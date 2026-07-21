//! Multi-step planners over a forward model rebuilt from the observation.
//!
//! Promoted from the value-boundary/baseline-duopoly binaries so the model semantics live in one
//! reviewed place. Two planners share the model:
//! - `DeterministicPlanner` (value-boundary): expectimax on expected noise volumes,
//!   deterministic arbitrage, martingale oracle.
//! - `StochasticPlanner` (baseline-duopoly-D): Monte Carlo rollouts with shocks drawn
//!   from the simulator's own distributions, continuation actions from the
//!   one-step lookahead heuristic on the model.
//!
//! Operator order in `advance_*` matches `ExecEnv::step` exactly:
//! agent -> noise -> arbitrage -> oracle advance -> fee update (fees are
//! part of the NEXT state). An earlier binary version updated fees before
//! the sampled oracle move; fixed here and re-run (see
//! .local/rl_equilibrium/m3r_stochastic_planner.md).
//!
//! No planner sees realized future shocks: rollout RNG streams are keyed
//! to the decision index, never to the episode seed.

use crate::sim::amm::Pool;
use crate::sim::arbitrage::arbitrage_step;
use crate::sim::env::{EnvConfig, N_ACTIONS, Observation, action_spec};
use crate::sim::execution_agent::ExecutionPolicy;
use crate::sim::fee::{FeeInputs, FeeRule};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

#[derive(Clone)]
pub struct Model {
    pub pools: [Pool; 2],
    pub oracle: f64,
    pub remaining: f64,
    pub steps_left: usize,
}

fn model_from_obs(cfg: &EnvConfig, obs: &Observation) -> Model {
    let mk = |mid: f64, y: f64, fb: f64, fs: f64| Pool::new(mid * y, y, fb, fs);
    Model {
        pools: [
            mk(
                obs.pool_a_mid,
                obs.pool_a_inventory_y,
                obs.pool_a_fee_buy,
                obs.pool_a_fee_sell,
            ),
            mk(
                obs.pool_b_mid,
                obs.pool_b_inventory_y,
                obs.pool_b_fee_buy,
                obs.pool_b_fee_sell,
            ),
        ],
        oracle: obs.oracle_price,
        remaining: obs.remaining_inventory,
        steps_left: (obs.remaining_time_frac * cfg.order.horizon as f64).round() as usize,
    }
}

/// Immediate premium-over-oracle cost of an action on the model, gas
/// included. Returns (cost, executed fraction of remaining).
fn action_cost(cfg: &EnvConfig, m: &Model, action: usize) -> Option<(f64, f64)> {
    let (_, fracs) = action_spec(action);
    let mut cost = 0.0;
    for (i, &frac) in fracs.iter().enumerate() {
        if frac == 0.0 {
            continue;
        }
        let q = m.remaining * frac;
        let fill = m.pools[i].buy_cost(q)?;
        cost += fill.amount_x - q * m.oracle + cfg.agent_gas_cost;
    }
    Some((cost, fracs[0] + fracs[1]))
}

/// Terminal carry, same functional form as the lookahead baseline.
fn carry(cfg: &EnvConfig, kappa: f64, m: &Model) -> f64 {
    let penalty = cfg.unfinished_penalty * m.oracle * m.remaining;
    if m.steps_left <= 1 {
        penalty
    } else {
        kappa * penalty / (m.steps_left - 1) as f64
    }
}

/// Softmax routing share on pool A, mirroring `sim::noise`.
fn route_share(cfg: &EnvConfig, oracle: f64, cost_a: f64, cost_b: f64) -> f64 {
    let gap_bps = (cost_b - cost_a) / oracle * 10_000.0;
    1.0 / (1.0 + (-cfg.noise.route_sensitivity * gap_bps).exp())
}

/// Apply agent action + noise flow (given volumes) + arbitrage + oracle
/// move + fee update, in the environment's operator order.
#[allow(clippy::too_many_arguments)]
fn advance_core(
    cfg: &EnvConfig,
    m: &Model,
    action: usize,
    total_buy: f64,
    total_sell: f64,
    next_oracle: f64,
    arb_rng: &mut StdRng,
    arb_speed: f64,
) -> Option<Model> {
    let mut next = m.clone();
    let (_, fracs) = action_spec(action);
    for (i, &frac) in fracs.iter().enumerate() {
        if frac > 0.0 {
            next.pools[i].buy(next.remaining * frac)?;
        }
    }
    next.remaining *= 1.0 - fracs[0] - fracs[1];

    // both routing shares are computed on pre-flow pools, matching
    // sim::noise::noise_flows, before any flow executes
    let probe = cfg.noise.probe_size;
    let ba = next.pools[0]
        .effective_buy_price(probe)
        .unwrap_or(f64::INFINITY);
    let bb = next.pools[1]
        .effective_buy_price(probe)
        .unwrap_or(f64::INFINITY);
    let buy_share = route_share(cfg, next.oracle, ba, bb);
    let pa = next.pools[0].effective_sell_price(probe).unwrap_or(0.0);
    let pb = next.pools[1].effective_sell_price(probe).unwrap_or(0.0);
    let sell_share = route_share(cfg, next.oracle, -pa, -pb);
    let buys = [total_buy * buy_share, total_buy * (1.0 - buy_share)];
    let sells = [total_sell * sell_share, total_sell * (1.0 - sell_share)];
    for i in 0..2 {
        if buys[i] > 0.0 {
            next.pools[i].buy(buys[i]);
        }
        if sells[i] > 0.0 {
            next.pools[i].sell(sells[i]);
        }
    }

    let mut arb = cfg.arb.clone();
    arb.speed = arb_speed;
    for i in 0..2 {
        arbitrage_step(&mut next.pools[i], next.oracle, &arb, arb_rng);
    }

    // oracle advances, THEN fees update for the next state (env order)
    next.oracle = next_oracle;
    let imb = [
        next.pools[0].inventory_imbalance(next.oracle),
        next.pools[1].inventory_imbalance(next.oracle),
    ];
    for i in 0..2 {
        let pair = cfg.dynamic_fee.fees(&FeeInputs {
            own_imbalance: imb[i],
            rival_imbalance: imb[1 - i],
            oracle_misalignment: (next.pools[i].mid_price() - next.oracle) / next.oracle,
        });
        next.pools[i].fee_buy = pair.buy;
        next.pools[i].fee_sell = pair.sell;
    }
    next.steps_left = next.steps_left.saturating_sub(1);
    Some(next)
}

/// One-step lookahead action on the model (planner continuation policy).
fn lookahead_action(cfg: &EnvConfig, kappa: f64, m: &Model) -> usize {
    if m.remaining <= 1e-9 {
        return 0;
    }
    let mut best = (0usize, f64::INFINITY);
    for a in 0..N_ACTIONS {
        let Some((cost, f)) = action_cost(cfg, m, a) else {
            continue;
        };
        let mut after = m.clone();
        after.remaining *= 1.0 - f;
        after.steps_left = after.steps_left.saturating_sub(1);
        let v = cost + carry(cfg, kappa, &after);
        if v < best.1 {
            best = (a, v);
        }
    }
    best.0
}

/// value-boundary deterministic expectimax planner: expected noise volumes,
/// deterministic arbitrage, martingale oracle (oracle unchanged, so the
/// oracle/fee operator order is inert here).
pub struct DeterministicPlanner {
    pub depth: usize,
    pub kappa: f64,
    pub cfg: EnvConfig,
    exp_buy: f64,
    exp_sell: f64,
    pub name: &'static str,
}

impl DeterministicPlanner {
    pub fn new(depth: usize, kappa: f64, cfg: EnvConfig, name: &'static str) -> Self {
        let s = cfg.noise.volume_sigma;
        let factor = if s > 0.0 { s.sinh() / s } else { 1.0 };
        Self {
            depth,
            kappa,
            exp_buy: cfg.noise.buy_intensity * factor,
            exp_sell: cfg.noise.sell_intensity * factor,
            cfg,
            name,
        }
    }

    fn advance(&self, m: &Model, action: usize) -> Option<Model> {
        let mut dummy = StdRng::seed_from_u64(0);
        advance_core(
            &self.cfg,
            m,
            action,
            self.exp_buy,
            self.exp_sell,
            m.oracle,
            &mut dummy,
            1.0,
        )
    }

    fn value(&self, m: &Model, depth: usize) -> f64 {
        if m.remaining <= 1e-9 {
            return 0.0;
        }
        if m.steps_left == 0 {
            return self.cfg.unfinished_penalty * m.oracle * m.remaining;
        }
        let mut best = f64::INFINITY;
        for a in 0..N_ACTIONS {
            let Some((cost, f)) = action_cost(&self.cfg, m, a) else {
                continue;
            };
            let tail = if depth <= 1 {
                let mut after = m.clone();
                after.remaining *= 1.0 - f;
                after.steps_left -= 1;
                carry(&self.cfg, self.kappa, &after)
            } else {
                match self.advance(m, a) {
                    Some(next) => self.value(&next, depth - 1),
                    None => continue,
                }
            };
            best = best.min(cost + tail);
        }
        best
    }
}

impl ExecutionPolicy for DeterministicPlanner {
    fn name(&self) -> &'static str {
        self.name
    }
    fn act(&mut self, obs: &Observation) -> usize {
        let m = model_from_obs(&self.cfg, obs);
        if m.remaining <= 1e-9 {
            return 0;
        }
        let mut best = (0usize, f64::INFINITY);
        for a in 0..N_ACTIONS {
            let Some((cost, f)) = action_cost(&self.cfg, &m, a) else {
                continue;
            };
            let tail = if self.depth <= 1 {
                let mut after = m.clone();
                after.remaining *= 1.0 - f;
                after.steps_left -= 1;
                carry(&self.cfg, self.kappa, &after)
            } else {
                match self.advance(&m, a) {
                    Some(next) => self.value(&next, self.depth - 1),
                    None => continue,
                }
            };
            if cost + tail < best.1 {
                best = (a, cost + tail);
            }
        }
        best.0
    }
}

/// baseline-duopoly-D stochastic rollout planner: sampled shocks, lookahead
/// continuation, tuned kappa carry at the rollout leaf.
pub struct StochasticPlanner {
    pub depth: usize,
    pub n_rollouts: usize,
    pub kappa: f64,
    pub cfg: EnvConfig,
}

impl StochasticPlanner {
    fn advance_sampled(&self, m: &Model, action: usize, rng: &mut StdRng) -> Option<Model> {
        let s = self.cfg.noise.volume_sigma;
        let draw = |rng: &mut StdRng, mean: f64| {
            let z: f64 = rng.gen_range(-1.0..1.0);
            (mean * (s * z).exp()).max(0.0)
        };
        let total_buy = draw(rng, self.cfg.noise.buy_intensity);
        let total_sell = draw(rng, self.cfg.noise.sell_intensity);
        let z: f64 = Normal::new(0.0, 1.0).unwrap().sample(rng);
        let dt = self.cfg.dt;
        let next_oracle = m.oracle
            * ((self.cfg.mu - 0.5 * self.cfg.sigma * self.cfg.sigma) * dt
                + self.cfg.sigma * dt.sqrt() * z)
                .exp();
        advance_core(
            &self.cfg,
            m,
            action,
            total_buy,
            total_sell,
            next_oracle,
            rng,
            self.cfg.arb.speed,
        )
    }
}

impl ExecutionPolicy for StochasticPlanner {
    fn name(&self) -> &'static str {
        "stochastic_planner"
    }
    fn act(&mut self, obs: &Observation) -> usize {
        let m = model_from_obs(&self.cfg, obs);
        if m.remaining <= 1e-9 {
            return 0;
        }
        let mut best = (0usize, f64::INFINITY);
        for a0 in 0..N_ACTIONS {
            let Some((root_cost, _)) = action_cost(&self.cfg, &m, a0) else {
                continue;
            };
            let mut total = 0.0;
            let mut ok = true;
            for r in 0..self.n_rollouts {
                // planner-private stream keyed to (decision step, rollout,
                // root action) only — never to the episode seed
                let mut rng = StdRng::seed_from_u64(
                    0x9E3779B97F4A7C15u64
                        .wrapping_mul(obs.step as u64 + 1)
                        .wrapping_add(r as u64 * 0x2545F4914F6CDD1D)
                        .wrapping_add(a0 as u64),
                );
                let Some(mut cur) = self.advance_sampled(&m, a0, &mut rng) else {
                    ok = false;
                    break;
                };
                let mut cost = 0.0;
                for _ in 1..self.depth {
                    if cur.remaining <= 1e-9 || cur.steps_left == 0 {
                        break;
                    }
                    let ak = lookahead_action(&self.cfg, self.kappa, &cur);
                    cost += action_cost(&self.cfg, &cur, ak)
                        .map(|(c, _)| c)
                        .unwrap_or(0.0);
                    match self.advance_sampled(&cur, ak, &mut rng) {
                        Some(n) => cur = n,
                        None => break,
                    }
                }
                total += cost + carry(&self.cfg, self.kappa, &cur);
            }
            if !ok {
                continue;
            }
            let v = root_cost + total / self.n_rollouts as f64;
            if v < best.1 {
                best = (a0, v);
            }
        }
        best.0
    }
}
