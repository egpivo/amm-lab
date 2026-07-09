#![allow(clippy::type_complexity)]

use rand::rngs::StdRng;
/// Contextual fee-control v2.
///
/// Extends v1 with: 6-dimensional state, 9-action fee space, regime persistence
/// (Markov chain), fee switching cost, and volume bucket.
///
/// State: gap × vol × flow × volume × persistence × prev_fee
/// Action: {1,3,5,6,8,10,15,20,30} bps
/// Reward: fee_revenue − toxic_loss − switch_cost
///
/// Key design constraint: the same (gap, flow) can still require different fees
/// depending on whether the toxic state has persisted long enough to justify
/// raising fees (vs a single-step blip that doesn't warrant a costly switch).
///
/// Usage: cargo run --release --bin campbell_bandit_v2
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;

// ── Actions ───────────────────────────────────────────────────────────────────

const ACTIONS: [f64; 9] = [1.0, 3.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0];
const N_ACTIONS: usize = 9;
const DEFAULT_ACTION: usize = 3; // 6 bps

// ── Environment constants ─────────────────────────────────────────────────────

const N_TRAIN: usize = 600_000;
const N_EVAL: usize = 100_000;
const ALPHA: f64 = 0.05;
const GAMMA: f64 = 0.99;
const EPSILON_START: f64 = 1.0;
const EPSILON_MIN: f64 = 0.05;
const NOISE_SIGMA: f64 = 1.5;
const SWITCH_LAMBDA: f64 = 0.5; // switching cost multiplier

// Regime initial frequencies (used for environment initialization only)
const REGIME_INIT: [f64; 3] = [0.60, 0.30, 0.10]; // Normal, Toxic, HighVolToxic

// Regime Markov transition matrix [from][to]
const REGIME_TRANS: [[f64; 3]; 3] = [
    [0.97, 0.02, 0.01], // Normal → stays normal 97%
    [0.05, 0.92, 0.03], // Toxic  → persists 92% per step (~13 steps avg)
    [0.10, 0.10, 0.80], // HighVolToxic → persists 80%
];

// ── Hidden regime ─────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Regime {
    Normal,
    Toxic,
    HighVolToxic,
}

impl Regime {
    fn index(self) -> usize {
        match self {
            Self::Normal => 0,
            Self::Toxic => 1,
            Self::HighVolToxic => 2,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Normal => "Normal",
            Self::Toxic => "Toxic",
            Self::HighVolToxic => "HiVolToxic",
        }
    }
}

fn transition_regime(regime: Regime, rng: &mut StdRng) -> Regime {
    let row = &REGIME_TRANS[regime.index()];
    let u: f64 = rng.gen_range(0.0..1.0);
    if u < row[0] {
        Regime::Normal
    } else if u < row[0] + row[1] {
        Regime::Toxic
    } else {
        Regime::HighVolToxic
    }
}

fn sample_regime_init(rng: &mut StdRng) -> Regime {
    let u: f64 = rng.gen_range(0.0..1.0);
    if u < REGIME_INIT[0] {
        Regime::Normal
    } else if u < REGIME_INIT[0] + REGIME_INIT[1] {
        Regime::Toxic
    } else {
        Regime::HighVolToxic
    }
}

// ── Context sampling (observables from hidden regime) ─────────────────────────

fn bucket3(rng: &mut StdRng, p0: f64, p1: f64) -> u8 {
    let u: f64 = rng.gen_range(0.0..1.0);
    if u < p0 {
        0
    } else if u < p0 + p1 {
        1
    } else {
        2
    }
}

fn sample_obs(rng: &mut StdRng, regime: Regime) -> (u8, u8, u8, u8) {
    // returns (gap, vol, flow, volume)
    let (gp0, gp1, vp0, vp1, fp0, fp1, mp0, mp1) = match regime {
        Regime::Normal => (0.50, 0.40, 0.60, 0.30, 0.70, 0.20, 0.35, 0.45),
        Regime::Toxic => (0.20, 0.45, 0.20, 0.45, 0.10, 0.20, 0.15, 0.40),
        Regime::HighVolToxic => (0.10, 0.30, 0.05, 0.15, 0.05, 0.20, 0.05, 0.20),
    };
    let gap = bucket3(rng, gp0, gp1);
    let vol = bucket3(rng, vp0, vp1);
    let flow = bucket3(rng, fp0, fp1);
    let volume = bucket3(rng, mp0, mp1); // recent trading volume bucket
    (gap, vol, flow, volume)
}

fn persistence_bucket(streak: usize) -> u8 {
    if streak <= 1 {
        0
    } else if streak <= 5 {
        1
    } else {
        2
    }
}

fn prev_fee_bucket(fee_bps: f64) -> u8 {
    ACTIONS
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            ((*a - fee_bps).abs())
                .partial_cmp(&((*b - fee_bps).abs()))
                .unwrap()
        })
        .map(|(i, _)| i as u8)
        .unwrap_or(DEFAULT_ACTION as u8)
}

// ── Q-learning state ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
struct QState {
    gap: u8,         // 0-2
    vol: u8,         // 0-2
    flow: u8,        // 0-2
    volume: u8,      // 0-2
    persistence: u8, // 0-2
    prev_fee: u8,    // 0-8 (action index)
}

// ── Reward function ───────────────────────────────────────────────────────────

fn reward_det(
    fee_bps: f64,
    prev_fee_bps: f64,
    vol: u8,
    flow: u8,
    volume: u8,
    regime: Regime,
) -> f64 {
    // fee_revenue: fund traders elastic (zero at 15 bps), arb inelastic (zero at 40 bps)
    let fund_vol = (1.0 - fee_bps / 15.0).max(0.0);
    let arb_vol = (1.0 - fee_bps / 40.0).max(0.0);
    let (fw, aw) = match flow {
        0 => (2.0, 0.2),
        1 => (1.0, 0.8),
        _ => (0.3, 1.5),
    };
    let vol_scale: f64 = match volume {
        0 => 0.7,
        1 => 1.0,
        _ => 1.4,
    };
    let fee_revenue = fee_bps * (fw * fund_vol + aw * arb_vol) * vol_scale;

    // toxic_loss: regime sets base, vol amplifies, volume amplifies for arb states
    let regime_scale: f64 = match regime {
        Regime::Normal => 0.1,
        Regime::Toxic => 3.0,
        Regime::HighVolToxic => 6.0,
    };
    let vol_amp: f64 = match vol {
        0 => 0.5,
        1 => 1.0,
        _ => 2.0,
    };
    // volume amplifies adverse selection in arb-heavy states
    let tox_vol_amp: f64 = 1.0 + 0.5 * (volume as f64) * (flow as f64 / 2.0);
    let adverse_cost = regime_scale * vol_amp * tox_vol_amp / (1.0 + fee_bps / 10.0);

    // switch_cost: penalizes large fee jumps
    let switch_cost = SWITCH_LAMBDA * (fee_bps - prev_fee_bps).abs() / 10.0;

    fee_revenue - adverse_cost - switch_cost
}

fn reward_noisy(
    fee_bps: f64,
    prev_fee_bps: f64,
    vol: u8,
    flow: u8,
    volume: u8,
    regime: Regime,
    rng: &mut StdRng,
) -> f64 {
    reward_det(fee_bps, prev_fee_bps, vol, flow, volume, regime)
        + Normal::new(0.0, NOISE_SIGMA).unwrap().sample(rng)
}

// ── Environment (stateful, Markov regime) ─────────────────────────────────────

struct Env {
    regime: Regime,
    gap: u8,
    vol: u8,
    flow: u8,
    volume: u8,
    flow_streak: usize,
    prev_fee_bps: f64,
}

impl Env {
    fn new(rng: &mut StdRng) -> Self {
        let regime = sample_regime_init(rng);
        let (gap, vol, flow, volume) = sample_obs(rng, regime);
        Self {
            regime,
            gap,
            vol,
            flow,
            volume,
            flow_streak: 1,
            prev_fee_bps: 6.0,
        }
    }

    fn current_qstate(&self) -> QState {
        QState {
            gap: self.gap,
            vol: self.vol,
            flow: self.flow,
            volume: self.volume,
            persistence: persistence_bucket(self.flow_streak),
            prev_fee: prev_fee_bucket(self.prev_fee_bps),
        }
    }

    fn step(&mut self, rng: &mut StdRng, action_idx: usize, noisy: bool) -> (QState, f64, QState) {
        let cur_state = self.current_qstate();
        let fee_bps = ACTIONS[action_idx];
        let r = if noisy {
            reward_noisy(
                fee_bps,
                self.prev_fee_bps,
                self.vol,
                self.flow,
                self.volume,
                self.regime,
                rng,
            )
        } else {
            reward_det(
                fee_bps,
                self.prev_fee_bps,
                self.vol,
                self.flow,
                self.volume,
                self.regime,
            )
        };
        // Transition
        self.regime = transition_regime(self.regime, rng);
        let (ng, nv, nf, nm) = sample_obs(rng, self.regime);
        if nf == self.flow {
            self.flow_streak += 1;
        } else {
            self.flow_streak = 1;
        }
        self.gap = ng;
        self.vol = nv;
        self.flow = nf;
        self.volume = nm;
        self.prev_fee_bps = fee_bps;
        let next_state = self.current_qstate();
        (cur_state, r, next_state)
    }
}

// ── Heuristic policies ────────────────────────────────────────────────────────
// These see only the observable state, not the hidden regime.

fn oracle_gap_fee(_gap: u8, _vol: u8, _flow: u8, _volume: u8, _pers: u8, _prev_fee: f64) -> f64 {
    // gap-only: same as v1
    match _gap {
        0 => 3.0,
        1 => 6.0,
        _ => 10.0,
    }
}

fn flow_aware_fee(_gap: u8, _vol: u8, flow: u8, _volume: u8, _pers: u8, _prev_fee: f64) -> f64 {
    match flow {
        0 => 6.0,
        1 => 10.0,
        _ => 20.0,
    }
}

fn vol_flow_aware_fee(_gap: u8, vol: u8, flow: u8, _volume: u8, _pers: u8, _prev_fee: f64) -> f64 {
    match (flow, vol) {
        (0, _) => 6.0,
        (1, 0) => 6.0,
        (1, _) => 10.0,
        (2, 0) => 10.0,
        (2, 1) => 20.0,
        _ => 20.0,
    }
}

fn persistence_aware_fee(_gap: u8, vol: u8, flow: u8, volume: u8, pers: u8, _prev_fee: f64) -> f64 {
    // Also uses volume and persistence:
    // - Arb-dom + sustained → 20 bps regardless
    // - Arb-dom + new blip + low volume → 10 bps (don't overreact)
    // - Arb-dom + new blip + high volume → 15 bps
    // - Switching cost: if already at high fee and regime may be ending (pers drops) → stay
    match flow {
        0 => match vol {
            2 => 8.0,
            _ => 6.0,
        }, // fund
        1 => match (vol, volume) {
            // mixed
            (_, 2) => 10.0,
            (2, _) => 10.0,
            _ => 8.0,
        },
        _ => match pers {
            // arb
            0 => match volume {
                2 => 15.0,
                1 => 10.0,
                _ => 8.0,
            }, // new: cautious
            1 => 20.0, // ongoing: raise
            _ => 20.0, // sustained: hold
        },
    }
}

// ── Q-learning agent ──────────────────────────────────────────────────────────

struct QLearner {
    q: HashMap<QState, [f64; N_ACTIONS]>,
    visits: HashMap<QState, [u32; N_ACTIONS]>,
    epsilon: f64,
}

impl QLearner {
    fn new() -> Self {
        Self {
            q: HashMap::new(),
            visits: HashMap::new(),
            epsilon: EPSILON_START,
        }
    }

    fn best_action(&self, s: QState) -> usize {
        self.q
            .get(&s)
            .map(|q| {
                q.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(DEFAULT_ACTION)
            })
            .unwrap_or(DEFAULT_ACTION)
    }

    fn best_q(&self, s: QState) -> f64 {
        self.q
            .get(&s)
            .map(|q| q.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
            .unwrap_or(0.0)
    }

    fn choose(&mut self, s: QState, rng: &mut StdRng) -> usize {
        if rng.gen_range(0.0..1.0f64) < self.epsilon {
            rng.gen_range(0..N_ACTIONS)
        } else {
            self.best_action(s)
        }
    }

    fn update(&mut self, s: QState, a: usize, r: f64, s_next: QState) {
        let max_next = self.best_q(s_next);
        let q = self.q.entry(s).or_insert([0.0; N_ACTIONS]);
        q[a] += ALPHA * (r + GAMMA * max_next - q[a]);
        self.visits.entry(s).or_insert([0u32; N_ACTIONS])[a] += 1;
    }

    fn decay(&mut self) {
        let per_step_decay = (EPSILON_START - EPSILON_MIN) / N_TRAIN as f64;
        self.epsilon = (self.epsilon - per_step_decay).max(EPSILON_MIN);
    }
}

// ── Evaluation harness ────────────────────────────────────────────────────────

struct EvalRecord {
    reward: f64,
    fee: f64,
    regime: Regime,
    switched: bool, // fee changed vs prev step
}

fn eval_policy(
    policy: &dyn Fn(u8, u8, u8, u8, u8, f64) -> f64,
    n_steps: usize,
    seed: u64,
) -> Vec<EvalRecord> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut env = Env::new(&mut rng);
    let mut records = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        let s = env.current_qstate();
        let fee = policy(
            s.gap,
            s.vol,
            s.flow,
            s.volume,
            s.persistence,
            env.prev_fee_bps,
        );
        let r = reward_det(fee, env.prev_fee_bps, s.vol, s.flow, s.volume, env.regime);
        let switched = (fee - env.prev_fee_bps).abs() > 0.5;
        records.push(EvalRecord {
            reward: r,
            fee,
            regime: env.regime,
            switched,
        });
        // Advance env without affecting the policy's fee choice
        env.step(&mut rng, prev_fee_bucket(fee) as usize, false);
    }
    records
}

fn eval_q(learner: &QLearner, n_steps: usize, seed: u64) -> Vec<EvalRecord> {
    eval_policy(
        &|gap, vol, flow, volume, pers, prev_fee| {
            let s = QState {
                gap,
                vol,
                flow,
                volume,
                persistence: pers,
                prev_fee: prev_fee_bucket(prev_fee),
            };
            ACTIONS[learner.best_action(s)]
        },
        n_steps,
        seed,
    )
}

fn stats(records: &[EvalRecord]) -> (f64, f64, f64, f64, f64, [f64; 3], [f64; 3]) {
    let mut rs: Vec<f64> = records.iter().map(|r| r.reward).collect();
    let mean = rs.iter().sum::<f64>() / rs.len() as f64;
    rs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p05 = rs[(0.05 * (rs.len() - 1) as f64).round() as usize];
    let mean_fee = records.iter().map(|r| r.fee).sum::<f64>() / rs.len() as f64;
    let switch_frac = records.iter().filter(|r| r.switched).count() as f64 / rs.len() as f64;
    let avg_abs_change = records
        .windows(2)
        .map(|w| (w[1].fee - w[0].fee).abs())
        .sum::<f64>()
        / (rs.len() - 1) as f64;
    let mut by_regime = [0.0f64; 3];
    let mut by_regime_n = [0usize; 3];
    let mut fee_by_regime = [0.0f64; 3];
    for r in records {
        by_regime[r.regime.index()] += r.reward;
        fee_by_regime[r.regime.index()] += r.fee;
        by_regime_n[r.regime.index()] += 1;
    }
    let mut mean_reward_regime = [0.0f64; 3];
    let mut mean_fee_regime = [0.0f64; 3];
    for i in 0..3 {
        if by_regime_n[i] > 0 {
            mean_reward_regime[i] = by_regime[i] / by_regime_n[i] as f64;
            mean_fee_regime[i] = fee_by_regime[i] / by_regime_n[i] as f64;
        }
    }
    (
        mean,
        p05,
        mean_fee,
        switch_frac,
        avg_abs_change,
        mean_reward_regime,
        mean_fee_regime,
    )
}

// ── Phase 0 ───────────────────────────────────────────────────────────────────

fn phase0() {
    println!("=== Phase 0: Fee Sweep by Regime ===");
    let n_mc: usize = 50_000;
    let mut rng = StdRng::seed_from_u64(1);
    let regimes = [Regime::Normal, Regime::Toxic, Regime::HighVolToxic];
    let sep = "─".repeat(86);
    println!("{sep}");
    print!("{:>14}", "");
    for a in ACTIONS {
        print!("  {:>6.0}bps", a);
    }
    println!();
    println!("{sep}");
    let mut best_fees = Vec::new();
    for regime in regimes {
        let samples: Vec<(u8, u8, u8, u8)> =
            (0..n_mc).map(|_| sample_obs(&mut rng, regime)).collect();
        let means: Vec<f64> = ACTIONS
            .iter()
            .map(|&fee| {
                samples
                    .iter()
                    .map(|&(_, vol, flow, volume)| {
                        reward_det(fee, fee, vol, flow, volume, regime) // prev_fee = fee (no switch cost)
                    })
                    .sum::<f64>()
                    / n_mc as f64
            })
            .collect();
        let best = means
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(3);
        best_fees.push((regime, ACTIONS[best], means[best]));
        print!("{:>14}", regime.name());
        for (i, &m) in means.iter().enumerate() {
            if i == best {
                print!("  {:>7.2}*", m);
            } else {
                print!("  {:>7.2} ", m);
            }
        }
        println!();
    }
    println!("{sep}");
    println!("\n* = best fee for that regime");
    for (r, fee, mr) in &best_fees {
        println!(
            "  {:>14}: {:>5.0} bps  (mean reward = {:.3})",
            r.name(),
            fee,
            mr
        );
    }
    println!();
}

// ── Phase 1 ───────────────────────────────────────────────────────────────────

fn phase1() {
    println!("=== Phase 1: Heuristic Policies ({N_EVAL} eval steps) ===\n");
    let policies: Vec<(&str, Box<dyn Fn(u8, u8, u8, u8, u8, f64) -> f64>)> = vec![
        ("fixed_6bps", Box::new(|_, _, _, _, _, _| 6.0)),
        ("fixed_10bps", Box::new(|_, _, _, _, _, _| 10.0)),
        ("fixed_20bps", Box::new(|_, _, _, _, _, _| 20.0)),
        ("oracle_gap", Box::new(oracle_gap_fee)),
        ("flow_aware", Box::new(flow_aware_fee)),
        ("vol_flow_aware", Box::new(vol_flow_aware_fee)),
        ("persistence_aware", Box::new(persistence_aware_fee)),
    ];

    let mut all_records: Vec<(&str, Vec<EvalRecord>)> = Vec::new();
    let eval_seed = 42_000u64;

    let sep = "─".repeat(110);
    println!("{sep}");
    println!(
        "{:<18} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>12} {:>14}",
        "policy", "mean", "p05", "fee", "sw_frac", "Δfee", "Normal", "Toxic", "HiVolTox"
    );
    println!("{sep}");
    for (name, policy) in &policies {
        let records = eval_policy(policy.as_ref(), N_EVAL, eval_seed);
        let (mean, p05, mfee, sw, dfe, mr, _) = stats(&records);
        println!(
            "{:<18} {:>8.3} {:>8.3} {:>8.2} {:>8.3} {:>8.3} {:>10.3} {:>12.3} {:>14.3}",
            name, mean, p05, mfee, sw, dfe, mr[0], mr[1], mr[2]
        );
        all_records.push((name, records));
    }
    println!("{sep}");

    // Paired deltas vs oracle_gap and vs vol_flow_aware
    let og_idx = all_records
        .iter()
        .position(|(n, _)| *n == "oracle_gap")
        .unwrap();
    let vfa_idx = all_records
        .iter()
        .position(|(n, _)| *n == "vol_flow_aware")
        .unwrap();
    let og_recs = &all_records[og_idx].1;
    let vfa_recs = &all_records[vfa_idx].1;
    println!("\n  Paired delta vs oracle_gap  /  vs vol_flow_aware:");
    for (name, records) in &all_records {
        if *name == "oracle_gap" {
            continue;
        }
        let d_og: f64 = records
            .iter()
            .zip(og_recs)
            .map(|(r, o)| r.reward - o.reward)
            .sum::<f64>()
            / records.len() as f64;
        let d_vfa: f64 = records
            .iter()
            .zip(vfa_recs)
            .map(|(r, v)| r.reward - v.reward)
            .sum::<f64>()
            / records.len() as f64;
        let beat_og = records
            .iter()
            .zip(og_recs)
            .filter(|(r, o)| r.reward > o.reward)
            .count() as f64
            / records.len() as f64
            * 100.0;
        println!(
            "  {:<18}  Δ_og={:>+7.3}  beat_og={:.0}%   Δ_vfa={:>+7.3}",
            name, d_og, beat_og, d_vfa
        );
    }
    println!();
}

// ── Phase 2 ───────────────────────────────────────────────────────────────────

fn phase2() {
    println!("=== Phase 2: Q-Learning ({N_TRAIN} train steps, γ={GAMMA}) ===\n");
    let mut agent = QLearner::new();
    let mut train_rng = StdRng::seed_from_u64(7);
    let mut env = Env::new(&mut train_rng);

    for step in 0..N_TRAIN {
        let s = env.current_qstate();
        let a = agent.choose(s, &mut train_rng);
        let (_, r, s_next) = env.step(&mut train_rng, a, true);
        agent.update(s, a, r, s_next);
        agent.decay();
        let _ = step;
    }
    println!(
        "  States visited: {}  /  Q-entries: {}",
        agent.q.len(),
        agent.q.len() * N_ACTIONS
    );

    let eval_seed = 99_000u64;
    let learned = eval_q(&agent, N_EVAL, eval_seed);
    let og_recs = eval_policy(&oracle_gap_fee, N_EVAL, eval_seed);
    let vfa_recs = eval_policy(&vol_flow_aware_fee, N_EVAL, eval_seed);
    let pa_recs = eval_policy(&persistence_aware_fee, N_EVAL, eval_seed);

    let (mean, p05, mfee, sw, dfe, mr, mfr) = stats(&learned);
    let (og_m, _, _, _, _, og_mr, og_mfr) = stats(&og_recs);
    let (vfa_m, _, _, _, _, vfa_mr, _) = stats(&vfa_recs);
    let (pa_m, _, _, _, _, pa_mr, _) = stats(&pa_recs);

    let sep = "─".repeat(70);
    println!("{sep}");
    println!(
        "{:<20} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "policy", "mean", "p05", "fee", "sw_frac", "Δfee"
    );
    println!("{sep}");
    println!(
        "{:<20} {:>8.3} {:>8.3} {:>8.2} {:>8.3} {:>8.3}",
        "oracle_gap", og_m, 0.0, 0.0, 0.0, 0.0
    );
    println!(
        "{:<20} {:>8.3} {:>8.3} {:>8.2} {:>8.3} {:>8.3}",
        "vol_flow_aware", vfa_m, 0.0, 0.0, 0.0, 0.0
    );
    println!(
        "{:<20} {:>8.3} {:>8.3} {:>8.2} {:>8.3} {:>8.3}",
        "persistence_aware", pa_m, 0.0, 0.0, 0.0, 0.0
    );
    println!(
        "{:<20} {:>8.3} {:>8.3} {:>8.2} {:>8.3} {:>8.3}",
        "q_learner", mean, p05, mfee, sw, dfe
    );
    println!("{sep}");

    // Paired deltas
    let d_og: f64 = learned
        .iter()
        .zip(&og_recs)
        .map(|(r, o)| r.reward - o.reward)
        .sum::<f64>()
        / N_EVAL as f64;
    let d_vfa: f64 = learned
        .iter()
        .zip(&vfa_recs)
        .map(|(r, v)| r.reward - v.reward)
        .sum::<f64>()
        / N_EVAL as f64;
    let d_pa: f64 = learned
        .iter()
        .zip(&pa_recs)
        .map(|(r, v)| r.reward - v.reward)
        .sum::<f64>()
        / N_EVAL as f64;
    let beat_og = learned
        .iter()
        .zip(&og_recs)
        .filter(|(r, o)| r.reward > o.reward)
        .count() as f64
        / N_EVAL as f64
        * 100.0;
    println!(
        "\n  Paired Δ vs oracle_gap:       {:>+.3}  beat={:.1}%",
        d_og, beat_og
    );
    println!("  Paired Δ vs vol_flow_aware:   {:>+.3}", d_vfa);
    println!("  Paired Δ vs persistence_aware:{:>+.3}", d_pa);

    // Per-regime breakdown
    println!("\n  Per-regime reward and mean fee:");
    println!(
        "  {:>14}  {:>8} {:>8}  {:>8} {:>8}  {:>8} {:>8}  {:>8} {:>8}",
        "regime", "q_r", "q_fee", "og_r", "og_fee", "vfa_r", "pa_r", "Δ_og", "Δ_vfa"
    );
    for (i, name) in ["Normal", "Toxic", "HiVolTox"].iter().enumerate() {
        println!(
            "  {:>14}  {:>8.3} {:>8.2}  {:>8.3} {:>8.2}  {:>8.3} {:>8.3}  {:>+8.3} {:>+8.3}",
            name,
            mr[i],
            mfr[i],
            og_mr[i],
            og_mfr[i],
            vfa_mr[i],
            pa_mr[i],
            mr[i] - og_mr[i],
            mr[i] - vfa_mr[i]
        );
    }

    // Policy table: grouped by (flow, persistence) × (vol, volume)
    println!("\n  Policy table (fee bps chosen by Q-learner, grouped by flow × persistence)");
    println!("  Columns: vol×volume = (lo,lo) (lo,hi) (mid,lo) (mid,hi) (hi,lo) (hi,hi)");
    println!("  {}", "─".repeat(72));
    for flow in 0u8..3 {
        let flow_label = match flow {
            0 => "fund",
            1 => "mix",
            _ => "arb",
        };
        for pers in 0u8..3 {
            let pers_label = match pers {
                0 => "new",
                1 => "ongoing",
                _ => "sustained",
            };
            print!("  flow={flow_label} pers={pers_label:<10}  gap=*:");
            for vol in [0u8, 2] {
                for volume in [0u8, 2] {
                    // Marginalize over gap and prev_fee: pick most common action
                    let mut action_votes = [0u32; N_ACTIONS];
                    for gap in 0u8..3 {
                        for prev_fee in 0u8..9 {
                            let s = QState {
                                gap,
                                vol,
                                flow,
                                volume,
                                persistence: pers,
                                prev_fee,
                            };
                            let a = agent.best_action(s);
                            action_votes[a] += 1;
                        }
                    }
                    let best = action_votes
                        .iter()
                        .enumerate()
                        .max_by_key(|&(_, v)| *v)
                        .map(|(i, _)| i)
                        .unwrap_or(DEFAULT_ACTION);
                    print!("  {:>5.0}", ACTIONS[best]);
                }
            }
            println!();
        }
    }
    println!("  (vol,volume) order: (lo,lo) (lo,hi) (mid,lo) (mid,hi) (hi,lo) (hi,hi)");

    // Per-flow-persistence-vol_volume: detailed fee table
    println!("\n  Detailed: fee by (flow, pers, vol, volume)  [marginalized over gap, prev_fee]");
    println!(
        "  {:>4} {:>10} {:>4} {:>7}  |  fee  visits",
        "flow", "pers", "vol", "volume"
    );
    println!("  {}", "─".repeat(52));
    for flow in 0u8..3 {
        for pers in 0u8..3 {
            for vol in 0u8..3 {
                for volume in 0u8..3 {
                    let mut action_votes = [0u32; N_ACTIONS];
                    let mut total_visits = 0u32;
                    for gap in 0u8..3 {
                        for prev_fee in 0u8..9 {
                            let s = QState {
                                gap,
                                vol,
                                flow,
                                volume,
                                persistence: pers,
                                prev_fee,
                            };
                            let a = agent.best_action(s);
                            let v = agent
                                .visits
                                .get(&s)
                                .map(|vv| vv.iter().sum::<u32>())
                                .unwrap_or(0);
                            action_votes[a] += v;
                            total_visits += v;
                        }
                    }
                    if total_visits == 0 {
                        continue;
                    }
                    let best = action_votes
                        .iter()
                        .enumerate()
                        .max_by_key(|&(_, v)| *v)
                        .map(|(i, _)| i)
                        .unwrap_or(DEFAULT_ACTION);
                    let fl = match flow {
                        0 => "fund",
                        1 => "mix",
                        _ => "arb",
                    };
                    let pl = match pers {
                        0 => "new",
                        1 => "ongoing",
                        _ => "sustained",
                    };
                    let vl = match vol {
                        0 => "lo",
                        1 => "mid",
                        _ => "hi",
                    };
                    let ml = match volume {
                        0 => "lo",
                        1 => "mid",
                        _ => "hi",
                    };
                    println!(
                        "  {:>4} {:>10} {:>4} {:>7}  |  {:5.1}  {}",
                        fl, pl, vl, ml, ACTIONS[best], total_visits
                    );
                }
            }
        }
    }

    // Success criterion
    println!("\n  Success criteria:");
    let mut ok = true;
    // Fund-dom states → low fee
    for vol in 0u8..3 {
        for volume in 0u8..3 {
            for pers in 0u8..3 {
                for gap in 0u8..3 {
                    for prev_fee in 0u8..9 {
                        let fee = ACTIONS[agent.best_action(QState {
                            gap,
                            vol,
                            flow: 0,
                            volume,
                            persistence: pers,
                            prev_fee,
                        })];
                        if fee > 10.0 {
                            println!("  [WARN] fund-dom state chose fee={fee:.0} bps");
                            ok = false;
                        }
                    }
                }
            }
        }
    }
    // Arb-dom + sustained + hi-vol → high fee
    for gap in 0u8..3 {
        for volume in 0u8..3 {
            for prev_fee in 0u8..9 {
                let fee = ACTIONS[agent.best_action(QState {
                    gap,
                    vol: 2,
                    flow: 2,
                    volume,
                    persistence: 2,
                    prev_fee,
                })];
                if fee < 10.0 {
                    println!("  [WARN] arb+hi-vol+sustained chose fee={fee:.0} bps");
                    ok = false;
                }
            }
        }
    }
    if d_og <= 0.0 {
        println!("  [FAIL] did not beat oracle_gap: Δ={d_og:.3}");
        ok = false;
    }
    if d_vfa < -1.0 {
        println!("  [WARN] >1 unit below vol_flow_aware: Δ={d_vfa:.3}");
    }
    if ok {
        println!("  [PASS] All hard success criteria met.");
    }

    // Multi-seed robustness
    println!("\n=== Multi-seed robustness ===");
    let sep2 = "─".repeat(72);
    println!("{sep2}");
    println!(
        "{:>12}  {:>10} {:>10} {:>10} {:>10}",
        "train_seed", "q_learner", "oracle_gap", "Δ_og", "beat%"
    );
    println!("{sep2}");
    for &tseed in &[0u64, 42, 123, 456, 789] {
        let mut a2 = QLearner::new();
        let mut rng2 = StdRng::seed_from_u64(tseed);
        let mut env2 = Env::new(&mut rng2);
        for _ in 0..N_TRAIN {
            let s = env2.current_qstate();
            let act = a2.choose(s, &mut rng2);
            let (_, r, sn) = env2.step(&mut rng2, act, true);
            a2.update(s, act, r, sn);
            a2.decay();
        }
        let qr = eval_q(&a2, N_EVAL, eval_seed);
        let ogr = eval_policy(&oracle_gap_fee, N_EVAL, eval_seed);
        let q_mean = qr.iter().map(|r| r.reward).sum::<f64>() / N_EVAL as f64;
        let og_mean2 = ogr.iter().map(|r| r.reward).sum::<f64>() / N_EVAL as f64;
        let d: Vec<f64> = qr
            .iter()
            .zip(&ogr)
            .map(|(a, b)| a.reward - b.reward)
            .collect();
        let delta2 = d.iter().sum::<f64>() / d.len() as f64;
        let beat2 = d.iter().filter(|&&x| x > 0.0).count() as f64 / d.len() as f64 * 100.0;
        println!(
            "{:>12}  {:>10.3} {:>10.3} {:>+10.3} {:>9.1}%",
            tseed, q_mean, og_mean2, delta2, beat2
        );
    }
    println!("{sep2}");
    println!();
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    println!("Campbell Contextual Fee-Control v2");
    println!("State: gap × vol × flow × volume × persistence × prev_fee");
    println!("Actions: {:?} bps", ACTIONS);
    println!("Regime: Markov chain  Normal(60%) Toxic(30%) HiVolTox(10%)");
    println!("Reward: fee_revenue − toxic_loss − switch_cost (λ={SWITCH_LAMBDA})");
    println!("{}\n", "═".repeat(72));

    phase0();
    phase1();
    phase2();
}
