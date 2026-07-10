//! Tabular Q-learning over a coarse discretization of the observation.
//!
//! Small by design: the M1 question is whether the closed-loop environment
//! contains exploitable sequential structure beyond tuned heuristics, not
//! whether a big learner can memorize the simulator. 225 states x 8 actions.
//!
//! State features (all decision-time, no future information):
//!   steps-left bin (6, log-spaced so the terminal region is fine-grained)
//!   x remaining bin (5) x route-gap bin (3) x premium bin (3).

use crate::sim::env::{ExecEnv, N_ACTIONS, Observation};
use crate::sim::execution_agent::ExecutionPolicy;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

pub const N_STATES: usize = 6 * 5 * 3 * 5;
/// Fine spec (M3A): steps-left(8) x remaining(6) x route-gap(5) x premium(6).
pub const N_STATES_FINE: usize = 8 * 6 * 5 * 6;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum StateSpec {
    Coarse,
    Fine,
}

impl StateSpec {
    pub fn n_states(self) -> usize {
        match self {
            StateSpec::Coarse => N_STATES,
            StateSpec::Fine => N_STATES_FINE,
        }
    }
    pub fn index(self, obs: &Observation, horizon: usize) -> usize {
        match self {
            StateSpec::Coarse => state_index(obs, horizon),
            StateSpec::Fine => state_index_fine(obs, horizon),
        }
    }
}

/// Marginal all-in buy quote per pool: fee-adjusted mid as a premium over
/// the oracle, in bps. Independent of the agent's remaining inventory, so
/// "market is cheap" cannot be confounded with "my order is large".
fn marginal_premium_bps(obs: &Observation) -> (f64, f64) {
    let prem =
        |mid: f64, fee: f64| (mid / (1.0 - fee) - obs.oracle_price) / obs.oracle_price * 10_000.0;
    (
        prem(obs.pool_a_mid, obs.pool_a_fee_buy),
        prem(obs.pool_b_mid, obs.pool_b_fee_buy),
    )
}

pub fn state_index(obs: &Observation, horizon: usize) -> usize {
    let bin = |x: f64, n: usize| ((x * n as f64) as usize).min(n - 1);
    let steps_left = ((obs.remaining_time_frac * horizon as f64).round() as usize).max(1);
    let tb = match steps_left {
        1 => 0,
        2..=3 => 1,
        4..=7 => 2,
        8..=15 => 3,
        16..=31 => 4,
        _ => 5,
    };
    let rb = bin(obs.remaining_frac.clamp(0.0, 1.0 - 1e-12), 5);
    let (prem_a, prem_b) = marginal_premium_bps(obs);
    let route = prem_a - prem_b;
    let gb = if route < -5.0 {
        0
    } else if route <= 5.0 {
        1
    } else {
        2
    };
    let best = prem_a.min(prem_b);
    let pb = if best < 30.0 {
        0
    } else if best < 45.0 {
        1
    } else if best < 60.0 {
        2
    } else if best < 80.0 {
        3
    } else {
        4
    };
    ((tb * 5 + rb) * 3 + gb) * 5 + pb
}

/// Fine discretization for M3A: same feature families, higher resolution
/// around the terminal region, the route gap, and the premium band.
pub fn state_index_fine(obs: &Observation, horizon: usize) -> usize {
    let steps_left = ((obs.remaining_time_frac * horizon as f64).round() as usize).max(1);
    let tb = match steps_left {
        1 => 0,
        2 => 1,
        3 => 2,
        4..=5 => 3,
        6..=9 => 4,
        10..=15 => 5,
        16..=31 => 6,
        _ => 7,
    };
    let rb = ((obs.remaining_frac.clamp(0.0, 1.0 - 1e-12) * 6.0) as usize).min(5);
    let (prem_a, prem_b) = marginal_premium_bps(obs);
    let route = prem_a - prem_b;
    let gb = if route < -15.0 {
        0
    } else if route < -5.0 {
        1
    } else if route <= 5.0 {
        2
    } else if route <= 15.0 {
        3
    } else {
        4
    };
    let best = prem_a.min(prem_b);
    let pb = if best < 25.0 {
        0
    } else if best < 40.0 {
        1
    } else if best < 55.0 {
        2
    } else if best < 70.0 {
        3
    } else if best < 90.0 {
        4
    } else {
        5
    };
    ((tb * 6 + rb) * 5 + gb) * 6 + pb
}

fn default_spec() -> StateSpec {
    StateSpec::Coarse
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QTable {
    pub q: Vec<Vec<f64>>, // [n_states][N_ACTIONS]
    pub visits: Vec<Vec<u64>>,
    pub mode: String,
    pub n_episodes: usize,
    pub train_seed_base: u64,
    #[serde(default = "default_spec")]
    pub spec: StateSpec,
}

impl QTable {
    pub fn new(mode: &str) -> Self {
        Self::with_spec(mode, StateSpec::Coarse)
    }

    pub fn with_spec(mode: &str, spec: StateSpec) -> Self {
        let n = spec.n_states();
        Self {
            q: vec![vec![0.0; N_ACTIONS]; n],
            visits: vec![vec![0; N_ACTIONS]; n],
            mode: mode.to_string(),
            n_episodes: 0,
            train_seed_base: 0,
            spec,
        }
    }

    pub fn greedy(&self, s: usize) -> usize {
        self.q[s]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        if let Some(dir) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, serde_json::to_string(self).unwrap())
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        Ok(serde_json::from_str(&std::fs::read_to_string(path)?).unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub n_episodes: usize,
    pub alpha0: f64,
    pub alpha_final: f64,
    pub eps0: f64,
    pub eps_final: f64,
    /// Undiscounted (1.0): finite horizon with time in the state.
    pub gamma: f64,
    pub train_seed_base: u64,
    pub rng_seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            n_episodes: 2_000_000,
            alpha0: 0.20,
            alpha_final: 0.005,
            eps0: 1.0,
            eps_final: 0.05,
            gamma: 1.0,
            train_seed_base: 1_000_000,
            rng_seed: 7,
        }
    }
}

/// Train on `env` (its config decides the market mode). Each episode uses a
/// fresh seed `train_seed_base + episode`, so the learner sees new GBM/noise
/// draws every episode and cannot memorize a path.
pub fn train(env: &mut ExecEnv, cfg: &TrainConfig, mut table: QTable) -> QTable {
    let mut rng = StdRng::seed_from_u64(cfg.rng_seed);
    let horizon = env.cfg.order.horizon;
    table.n_episodes += cfg.n_episodes;
    table.train_seed_base = cfg.train_seed_base;
    for ep in 0..cfg.n_episodes {
        let progress = ep as f64 / cfg.n_episodes.max(1) as f64;
        let eps = cfg.eps0 + (cfg.eps_final - cfg.eps0) * progress;
        let alpha = cfg.alpha0 + (cfg.alpha_final - cfg.alpha0) * progress;
        env.reset(cfg.train_seed_base + ep as u64);
        let mut s = table.spec.index(&env.observe(), horizon);
        // roll out with eps-greedy, then update BACKWARD through the episode
        // so the terminal penalty propagates in one pass (gamma = 1 finite
        // horizon; one-step updates take ~horizon episodes to carry it back).
        let mut transitions: Vec<(usize, usize, f64)> = Vec::with_capacity(64);
        while !env.is_done() {
            let a = if rng.gen_range(0.0..1.0) < eps {
                rng.gen_range(0..N_ACTIONS)
            } else {
                table.greedy(s)
            };
            let res = env.step(a);
            transitions.push((s, a, res.reward));
            s = table.spec.index(&env.observe(), horizon);
        }
        // Monte Carlo control on the return-to-go: no bootstrapping, so the
        // optimistic-zero-init + max-bias trap (Q(wait) pinned at 0 while
        // any downstream action is under-visited) cannot occur, and the
        // terminal penalty reaches every step of the episode exactly.
        let mut g = 0.0;
        for &(s, a, r) in transitions.iter().rev() {
            g = r + cfg.gamma * g;
            table.q[s][a] += alpha * (g - table.q[s][a]);
            table.visits[s][a] += 1;
        }
    }
    table
}

/// Greedy policy over a trained table, usable anywhere a baseline is.
pub struct QPolicy {
    pub table: QTable,
    pub horizon: usize,
}

impl ExecutionPolicy for QPolicy {
    fn name(&self) -> &'static str {
        match self.table.spec {
            StateSpec::Coarse => "q_learner",
            StateSpec::Fine => "q_learner_fine",
        }
    }
    fn act(&mut self, obs: &Observation) -> usize {
        if obs.remaining_inventory <= 1e-9 {
            return 0;
        }
        self.table.greedy(self.table.spec.index(obs, self.horizon))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::env::{EnvConfig, MarketMode};

    #[test]
    fn state_index_in_range() {
        let env = ExecEnv::new(EnvConfig::baseline(MarketMode::DynamicDuopoly, 1));
        let horizon = env.cfg.order.horizon;
        let s = state_index(&env.observe(), horizon);
        assert!(s < N_STATES);
    }

    #[test]
    fn training_is_deterministic() {
        let run = || {
            let mut env = ExecEnv::new(EnvConfig::baseline(MarketMode::DynamicDuopoly, 0));
            let cfg = TrainConfig {
                n_episodes: 50,
                ..TrainConfig::default()
            };
            let t = train(&mut env, &cfg, QTable::new("test"));
            t.q.iter().flatten().sum::<f64>()
        };
        assert_eq!(run(), run());
    }
}
