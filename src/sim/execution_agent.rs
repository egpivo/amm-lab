//! Execution policies over the discrete action space.
//!
//! These are the non-RL baselines. An RL agent plugs in through the same
//! trait (or through the JSON bridge in `export_rl_env`).

use crate::sim::env::{N_ACTIONS, Observation};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

pub trait ExecutionPolicy {
    fn name(&self) -> &'static str;
    fn act(&mut self, obs: &Observation) -> usize;
    fn reset(&mut self) {}
}

/// Liquidate as fast as the action set allows: 50% of remaining on the
/// better-quoted pool every step.
pub struct ImmediatePolicy;

fn best_pool_is_a(obs: &Observation) -> bool {
    // For a buy order the cheaper pool is better; est_slippage features are
    // already side-aware, so compare effective quotes directly via gaps+fees.
    let cost_a = obs.pool_a_oracle_gap_bps + obs.pool_a_fee_buy * 10_000.0;
    let cost_b = obs.pool_b_oracle_gap_bps + obs.pool_b_fee_buy * 10_000.0;
    cost_a <= cost_b
}

impl ExecutionPolicy for ImmediatePolicy {
    fn name(&self) -> &'static str {
        "immediate"
    }
    fn act(&mut self, obs: &Observation) -> usize {
        if obs.remaining_inventory <= 0.0 {
            return 0;
        }
        if best_pool_is_a(obs) { 5 } else { 6 }
    }
}

/// Even schedule: with N steps left, execute 1/N of remaining inventory on
/// the better-quoted pool, using the smallest action size that keeps pace.
pub struct TwapPolicy {
    pub horizon: usize,
}

impl ExecutionPolicy for TwapPolicy {
    fn name(&self) -> &'static str {
        "twap"
    }
    fn act(&mut self, obs: &Observation) -> usize {
        if obs.remaining_inventory <= 0.0 {
            return 0;
        }
        let n_left = (self.horizon - obs.step).max(1) as f64;
        let frac = 1.0 / n_left;
        pick_size(frac, best_pool_is_a(obs))
    }
}

fn pick_size(frac: f64, pool_a: bool) -> usize {
    // Smallest available size >= needed fraction of remaining.
    if frac <= 0.10 {
        if pool_a { 1 } else { 2 }
    } else if frac <= 0.25 {
        if pool_a { 3 } else { 4 }
    } else if pool_a {
        5
    } else {
        6
    }
}

/// Greedy router: fixed 25% clip each step, routed to whichever pool has the
/// lower estimated all-in cost right now; waits only if quotes are worse than
/// the terminal penalty would be.
pub struct MyopicRouterPolicy;

impl ExecutionPolicy for MyopicRouterPolicy {
    fn name(&self) -> &'static str {
        "myopic_router"
    }
    fn act(&mut self, obs: &Observation) -> usize {
        if obs.remaining_inventory <= 0.0 {
            return 0;
        }
        if best_pool_is_a(obs) { 3 } else { 4 }
    }
}

/// Estimated all-in X cost (fee included, gas excluded) to buy `qty` on a
/// pool reconstructed from observation fields (mid, Y inventory, buy fee).
/// Baselines only see decision-time state; no future information.
fn est_buy_cost(mid: f64, reserve_y: f64, fee_buy: f64, qty: f64) -> Option<f64> {
    if qty <= 0.0 || qty >= reserve_y {
        return None;
    }
    let reserve_x = mid * reserve_y;
    let dx_net = reserve_x * qty / (reserve_y - qty);
    Some(dx_net / (1.0 - fee_buy))
}

pub(crate) fn pool_cost(obs: &Observation, pool_a: bool, qty: f64) -> Option<f64> {
    if pool_a {
        est_buy_cost(
            obs.pool_a_mid,
            obs.pool_a_inventory_y,
            obs.pool_a_fee_buy,
            qty,
        )
    } else {
        est_buy_cost(
            obs.pool_b_mid,
            obs.pool_b_inventory_y,
            obs.pool_b_fee_buy,
            qty,
        )
    }
}

/// Cost of an action (premium over oracle, gas included) plus the fraction
/// of remaining it executes. Buy-side only; all closed-loop orders are buys.
fn action_cost(obs: &Observation, action: usize) -> Option<(f64, f64)> {
    let (_, fracs) = crate::sim::env::action_spec(action);
    let mut cost = 0.0;
    for (i, &frac) in fracs.iter().enumerate() {
        if frac == 0.0 {
            continue;
        }
        let q = obs.remaining_inventory * frac;
        let c = pool_cost(obs, i == 0, q)?;
        cost += c - q * obs.oracle_price + obs.gas_cost;
    }
    Some((cost, fracs[0] + fracs[1]))
}

/// TWAP completion schedule, but each slice goes to whichever single pool
/// (or the 50/50 split when the pace allows) has the lowest estimated
/// all-in per-unit cost, gas included.
pub struct FeeAwareTwapPolicy {
    pub horizon: usize,
}

impl ExecutionPolicy for FeeAwareTwapPolicy {
    fn name(&self) -> &'static str {
        "fee_aware_twap"
    }
    fn act(&mut self, obs: &Observation) -> usize {
        if obs.remaining_inventory <= 1e-9 {
            return 0;
        }
        let n_left = (self.horizon - obs.step).max(1) as f64;
        let frac = 1.0 / n_left;
        let base = pick_size(frac, true); // size class on pool A
        // candidates at the same pace: A, B, and split if pace >= 50%
        let mut candidates = vec![base, base + 1];
        if base == 5 {
            candidates.push(7);
        }
        candidates
            .into_iter()
            .filter_map(|a| {
                let (cost, f) = action_cost(obs, a)?;
                let per_unit = cost / (obs.remaining_inventory * f);
                Some((a, per_unit))
            })
            .min_by(|x, y| x.1.total_cmp(&y.1))
            .map(|(a, _)| a)
            .unwrap_or(base)
    }
}

/// One-step lookahead over all discrete actions: immediate execution premium
/// (vs current oracle, gas included) plus an urgency carry term for the
/// inventory left after the action. kappa is tuned on validation seeds.
/// Uses only decision-time quotes; no peeking at future shocks.
pub struct LookaheadPolicy {
    pub horizon: usize,
    pub kappa: f64,
    pub unfinished_penalty: f64,
}

impl ExecutionPolicy for LookaheadPolicy {
    fn name(&self) -> &'static str {
        "lookahead"
    }
    fn act(&mut self, obs: &Observation) -> usize {
        if obs.remaining_inventory <= 1e-9 {
            return 0;
        }
        let n_left = (self.horizon - obs.step).max(1);
        let mut best = (0usize, f64::INFINITY);
        for action in 0..N_ACTIONS {
            let Some((exec_cost, f)) = action_cost(obs, action) else {
                continue;
            };
            let remaining_after = obs.remaining_inventory * (1.0 - f);
            let penalty_value = self.unfinished_penalty * obs.oracle_price * remaining_after;
            let carry = if n_left <= 1 {
                penalty_value // last step: full terminal penalty is real
            } else {
                self.kappa * penalty_value / (n_left - 1) as f64
            };
            let total = exec_cost + carry;
            if total < best.1 {
                best = (action, total);
            }
        }
        best.0
    }
}

/// Uniform random action, seeded.
pub struct RandomPolicy {
    rng: StdRng,
    seed: u64,
}

impl RandomPolicy {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            seed,
        }
    }
}

impl ExecutionPolicy for RandomPolicy {
    fn name(&self) -> &'static str {
        "random"
    }
    fn act(&mut self, _obs: &Observation) -> usize {
        self.rng.gen_range(0..N_ACTIONS)
    }
    fn reset(&mut self) {
        self.rng = StdRng::seed_from_u64(self.seed);
    }
}
