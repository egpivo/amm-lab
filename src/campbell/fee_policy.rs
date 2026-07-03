use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

pub const RL_ACTIONS_BPS: [f64; 8] = [3.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0];
const N_ACTIONS: usize = 8;
type QEntry = ([f64; N_ACTIONS], [u32; N_ACTIONS]);

pub struct FeeObservation {
    pub step: usize,
    pub external_price: f64,
    pub amm_price: f64,
    pub oracle_gap_bps: f64,
    pub inventory_skew: f64,
    pub recent_vol: f64,
    pub recent_arb_frac: f64,
    pub recent_fund_frac: f64,
}

pub trait FeePolicy {
    fn name(&self) -> &'static str;
    fn fee(&mut self, obs: &FeeObservation) -> f64;
}

pub struct FixedFeePolicy {
    pub fee: f64,
}

impl FixedFeePolicy {
    pub fn new(fee: f64) -> Self {
        Self { fee }
    }
}

impl FeePolicy for FixedFeePolicy {
    fn name(&self) -> &'static str {
        "fixed"
    }
    fn fee(&mut self, _obs: &FeeObservation) -> f64 {
        self.fee
    }
}

pub struct OracleGapFeePolicy {
    pub base_fee: f64,
    pub gap_multiplier: f64,
    pub min_fee: f64,
    pub max_fee: f64,
}

impl FeePolicy for OracleGapFeePolicy {
    fn name(&self) -> &'static str {
        "oracle_gap"
    }
    fn fee(&mut self, obs: &FeeObservation) -> f64 {
        let f = self.base_fee + self.gap_multiplier * obs.oracle_gap_bps.abs() / 10_000.0;
        f.clamp(self.min_fee, self.max_fee)
    }
}

pub struct InventoryGapFeePolicy {
    pub base_fee: f64,
    pub gap_multiplier: f64,
    pub min_fee: f64,
    pub max_fee: f64,
}

impl FeePolicy for InventoryGapFeePolicy {
    fn name(&self) -> &'static str {
        "inventory_gap"
    }
    fn fee(&mut self, obs: &FeeObservation) -> f64 {
        let f = self.base_fee + self.gap_multiplier * obs.inventory_skew.abs();
        f.clamp(self.min_fee, self.max_fee)
    }
}

#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
pub struct RlState {
    pub gap_bucket: u8,
    pub vol_bucket: u8,
    pub flow_bucket: u8,
}

fn bucket_gap(abs_gap_bps: f64) -> u8 {
    match abs_gap_bps {
        g if g < 2.0 => 0,
        g if g < 5.0 => 1,
        g if g < 10.0 => 2,
        g if g < 20.0 => 3,
        g if g < 50.0 => 4,
        _ => 5,
    }
}

fn bucket_vol(vol: f64) -> u8 {
    // sigma step ~ 0.00105 at sigma = 0.04 annual
    match vol {
        v if v < 0.0007 => 0,
        v if v < 0.0016 => 1,
        _ => 2,
    }
}

fn bucket_flow(arb_frac: f64, fund_frac: f64) -> u8 {
    let total = arb_frac + fund_frac;
    if total < 0.25 {
        return 0;
    } // quite: fewer than 5/20 steps active
    match (arb_frac > 2.0 * fund_frac, fund_frac > 2.0 * arb_frac) {
        (true, _) => 1,
        (_, true) => 2,
        _ => 3,
    }
}

pub struct TabularLearnedFeePolicy {
    pub q_table: HashMap<RlState, QEntry>,
    pub epsilon: f64,
    pub alpha: f64,
    trajectory: Vec<(RlState, usize)>,
    rng: StdRng,
    pub inference: bool,
}

impl TabularLearnedFeePolicy {
    pub fn new(epsilon: f64, alpha: f64, rng_seed: u64) -> Self {
        Self {
            q_table: HashMap::new(),
            epsilon,
            alpha,
            trajectory: Vec::new(),
            rng: StdRng::seed_from_u64(rng_seed),
            inference: false,
        }
    }
    pub fn set_inference(&mut self) {
        self.inference = true;
        self.epsilon = 0.0;
    }

    fn obs_to_state(&self, obs: &FeeObservation) -> RlState {
        RlState {
            gap_bucket: bucket_gap(obs.oracle_gap_bps.abs()),
            vol_bucket: bucket_vol(obs.recent_vol),
            flow_bucket: bucket_flow(obs.recent_arb_frac, obs.recent_fund_frac),
        }
    }

    fn best_action(&self, state: RlState) -> usize {
        self.q_table
            .get(&state)
            .map(|(q, _): &QEntry| {
                q.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(2)
            })
            .unwrap_or(2) // default 6 bps
    }

    fn choose_action(&mut self, state: RlState) -> usize {
        if self.rng.r#gen::<f64>() < self.epsilon {
            self.rng.gen_range(0..N_ACTIONS)
        } else {
            self.best_action(state)
        }
    }

    pub fn update_episode(&mut self, reward: f64) {
        for &(state, action) in &self.trajectory {
            let (q, counts) = self
                .q_table
                .entry(state)
                .or_insert(([0.0; N_ACTIONS], [0u32; N_ACTIONS]));
            q[action] += self.alpha * (reward - q[action]);
            counts[action] = counts[action].saturating_add(1);
        }
        self.trajectory.clear();
    }

    pub fn decay_epsilon(&mut self, factor: f64, min_epsilon: f64) {
        self.epsilon = (self.epsilon * factor).max(min_epsilon);
    }

    pub fn save_q_table(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        writeln!(
            f,
            "gap_bucket,vol_bucket,flow_bucket,action_fee_bps,value_estimate,visit_count"
        )?;
        let mut keys: Vec<_> = self.q_table.keys().copied().collect();
        keys.sort_by_key(|s| (s.gap_bucket, s.vol_bucket, s.flow_bucket));
        for key in keys {
            let (q, counts) = &self.q_table[&key];
            for a in 0..N_ACTIONS {
                writeln!(
                    f,
                    "{},{},{},{},{:.6},{}",
                    key.gap_bucket,
                    key.vol_bucket,
                    key.flow_bucket,
                    RL_ACTIONS_BPS[a],
                    q[a],
                    counts[a]
                )?;
            }
        }
        Ok(())
    }

    pub fn from_csv(path: &str) -> std::io::Result<Self> {
        use std::io::BufRead;
        let mut policy = TabularLearnedFeePolicy::new(0.0, 0.0, 0);
        policy.inference = true;
        let reader = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut lines = reader.lines();

        let _ = lines.next(); // skip header

        for line in lines.map_while(Result::ok) {
            let p: Vec<&str> = line.trim().split(',').collect();
            if p.len() < 6 {
                continue;
            }
            let state = RlState {
                gap_bucket: p[0].parse().unwrap_or(0),
                vol_bucket: p[1].parse().unwrap_or(0),
                flow_bucket: p[2].parse().unwrap_or(0),
            };
            let bps: f64 = p[3].parse().unwrap_or(6.0);
            let qv: f64 = p[4].parse().unwrap_or(0.0);
            let ct: u32 = p[5].parse().unwrap_or(0);
            let a = RL_ACTIONS_BPS
                .iter()
                .position(|&x| (x - bps).abs() < 0.01)
                .unwrap_or(2);
            let e = policy
                .q_table
                .entry(state)
                .or_insert(([0.0; N_ACTIONS], [0u32; N_ACTIONS]));
            e.0[a] = qv;
            e.1[a] = ct;
        }
        Ok(policy)
    }
}

impl FeePolicy for TabularLearnedFeePolicy {
    fn name(&self) -> &'static str {
        "tabular_rl"
    }

    fn fee(&mut self, obs: &FeeObservation) -> f64 {
        let state = self.obs_to_state(obs);
        let action = if self.inference {
            self.best_action(state)
        } else {
            self.choose_action(state)
        };
        if !self.inference {
            self.trajectory.push((state, action));
        }
        RL_ACTIONS_BPS[action] / 10_000.0
    }
}
