//! Closed-loop execution environment.
//!
//! One episode: an execution trader must fill a target order of Q units of Y
//! over H steps across two competing AMM pools. Every agent trade moves pool
//! inventory, which moves quotes, which moves the dynamic fee rule, noise
//! routing, and the arbitrageur's response. Historical data plays no role at
//! runtime; it only calibrates the config.
//!
//! Step order: fees are already part of the step's starting state (set at
//! the end of the previous step); then (1) agent acts, (2) noise flow
//! arrives, (3) arbitrageur responds per pool, (4) the oracle advances,
//! (5) fee rules update from the post-step state, producing the fees of
//! the NEXT state. Fees never respond within a step to the agent's own
//! trade. The oracle-pool gap is measured right after the agent's trade
//! (market-impact diagnostic) and again at end of step.

use crate::sim::amm::Pool;
use crate::sim::arbitrage::{ArbConfig, arbitrage_step};
use crate::sim::fee::{ConstantFee, FeeInputs, FeeRule, LinearDynamicFee};
use crate::sim::noise::{NoiseConfig, NoiseFlows, noise_flows};
use crate::sim::oracle::{gbm_path, rolling_vol};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

pub const N_ACTIONS: usize = 8;

/// Discrete action space: fraction of *remaining* inventory per pool.
/// Index 7 splits 50% of remaining equally across both pools.
pub fn action_spec(action: usize) -> (&'static str, [f64; 2]) {
    match action {
        0 => ("wait", [0.0, 0.0]),
        1 => ("exec_10_A", [0.10, 0.0]),
        2 => ("exec_10_B", [0.0, 0.10]),
        3 => ("exec_25_A", [0.25, 0.0]),
        4 => ("exec_25_B", [0.0, 0.25]),
        5 => ("exec_50_A", [0.50, 0.0]),
        6 => ("exec_50_B", [0.0, 0.50]),
        7 => ("split_50", [0.25, 0.25]),
        _ => ("invalid", [0.0, 0.0]),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Side {
    Buy,
    Sell,
}

/// Intra-step ordering of the agent relative to noise flow + arbitrage.
/// `Before` is frozen v0 semantics (agent has priority). `After` and
/// `Random` exist for the policy-selected priority-artifact diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AgentOrder {
    Before,
    After,
    Random,
}

/// LP-adaptation: LP depth-adaptation regimes. Motivated by the causal paper's
/// finding of no large short-run average LP response (Frozen default is
/// the calibrated case); Weak/Aggressive are SENSITIVITY layers, not
/// causal evidence. LPs scale pool depth (price-preserving) in response
/// to toxic flow, proxied by arbitrage hits in a trailing window.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LpRegime {
    /// baseline-duopoly default: depth fixed over the horizon (as in arXiv:2603.09669).
    Frozen,
    /// Small, delayed withdrawal: >=3 arb hits in the last 10 steps ->
    /// depth x0.997 per step; else recovery x1.001; floor 0.90.
    Weak,
    /// Stress case: >=2 hits -> x0.98 per step; floor 0.50; recovery
    /// x1.0005.
    Aggressive,
}

/// JIT-searcher: minimal JIT/searcher layer. When an agent trade on a pool exceeds
/// `threshold_y` (and a seeded coin at `prob` lands), a searcher
/// sandwiches it: buys `frac * q` before the fill and sells it back
/// after. Deterministic by seed; not a full MEV market model.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum JitRegime {
    None,
    /// threshold 5 Y, frac 0.25, prob 0.5
    Weak,
    /// threshold 2 Y, frac 1.0, prob 1.0
    Aggressive,
}

impl JitRegime {
    fn params(self) -> Option<(f64, f64, f64)> {
        match self {
            JitRegime::None => None,
            JitRegime::Weak => Some((5.0, 0.25, 0.5)),
            JitRegime::Aggressive => Some((2.0, 1.0, 1.0)),
        }
    }
}

/// Completion handling at the final step (baseline-duopoly-A).
/// `ForcedTerminal`: any inventory left after the agent's final action is
/// executed immediately, split equally across live pools, at prevailing
/// all-in cost (gas charged per pool). Applied identically to every
/// policy; the forced leg's premium is tracked separately in the summary.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CompletionRule {
    Standard,
    ForcedTerminal,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MarketMode {
    /// Two pools, both constant fee.
    ConstantDuopoly,
    /// Single pool (A) with the dynamic rule; B is dead.
    DynamicMonopoly,
    /// Two pools, both running the linear dynamic rule against each other.
    DynamicDuopoly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderSpec {
    pub side: Side,
    /// Target quantity in Y units.
    pub quantity: f64,
    /// Horizon in steps.
    pub horizon: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvConfig {
    pub mode: MarketMode,
    pub order: OrderSpec,
    // oracle
    pub s0: f64,
    pub mu: f64,
    pub sigma: f64,
    pub dt: f64,
    pub vol_window: usize,
    // pools (each pool starts identical)
    pub pool_reserve_x: f64,
    pub pool_reserve_y: f64,
    pub constant_fee: f64,
    pub dynamic_fee: LinearDynamicFee,
    // flows
    pub noise: NoiseConfig,
    pub arb: ArbConfig,
    /// Gas charged to the agent per pool touched per step, X units.
    pub agent_gas_cost: f64,
    /// Terminal penalty per unfilled unit, as a fraction of arrival price.
    pub unfinished_penalty: f64,
    /// Intra-step ordering of agent vs noise+arb (v0 default: Before).
    pub agent_order: AgentOrder,
    /// Terminal completion handling (v0 default: Standard).
    pub completion_rule: CompletionRule,
    /// LP-adaptation LP depth adaptation (default Frozen = baseline-duopoly semantics).
    pub lp_regime: LpRegime,
    /// JIT-searcher JIT/searcher layer (default None = baseline-duopoly semantics).
    pub jit_regime: JitRegime,
    pub seed: u64,
}

impl EnvConfig {
    pub fn baseline(mode: MarketMode, seed: u64) -> Self {
        Self {
            mode,
            order: OrderSpec {
                side: Side::Buy,
                quantity: 50.0,
                horizon: 50,
            },
            s0: 1_000.0,
            mu: 0.0,
            sigma: 0.5,
            dt: 1.0 / (365.0 * 24.0 * 60.0), // one-minute steps
            vol_window: 20,
            pool_reserve_x: 1_000_000.0,
            pool_reserve_y: 1_000.0,
            constant_fee: 0.003,
            dynamic_fee: LinearDynamicFee::duopoly_default(),
            noise: NoiseConfig::default(),
            arb: ArbConfig::default(),
            agent_gas_cost: 2.0,
            unfinished_penalty: 0.02,
            agent_order: AgentOrder::Before,
            completion_rule: CompletionRule::Standard,
            lp_regime: LpRegime::Frozen,
            jit_regime: JitRegime::None,
            seed,
        }
    }

    pub fn n_live_pools(&self) -> usize {
        match self.mode {
            MarketMode::DynamicMonopoly => 1,
            _ => 2,
        }
    }
}

/// Full observation exposed to the agent (raw units; `to_vec` scales it).
#[derive(Debug, Clone, Serialize)]
pub struct Observation {
    pub step: usize,
    pub remaining_inventory: f64,
    pub remaining_frac: f64,
    pub remaining_time_frac: f64,
    pub oracle_price: f64,
    pub pool_a_inventory_y: f64,
    pub pool_b_inventory_y: f64,
    pub pool_a_mid: f64,
    pub pool_b_mid: f64,
    pub pool_a_fee_buy: f64,
    pub pool_a_fee_sell: f64,
    pub pool_b_fee_buy: f64,
    pub pool_b_fee_sell: f64,
    pub pool_a_oracle_gap_bps: f64,
    pub pool_b_oracle_gap_bps: f64,
    pub rival_quote_gap_bps: f64,
    pub recent_vol: f64,
    pub est_slippage_small_bps: f64,
    pub est_slippage_medium_bps: f64,
    pub est_slippage_large_bps: f64,
    pub gas_cost: f64,
    pub prev_action: usize,
}

impl Observation {
    /// Flat, roughly unit-scaled feature vector for RL consumption.
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.remaining_frac,
            self.remaining_time_frac,
            self.oracle_price.ln(),
            self.pool_a_oracle_gap_bps / 100.0,
            self.pool_b_oracle_gap_bps / 100.0,
            self.rival_quote_gap_bps / 100.0,
            self.pool_a_fee_buy * 100.0,
            self.pool_a_fee_sell * 100.0,
            self.pool_b_fee_buy * 100.0,
            self.pool_b_fee_sell * 100.0,
            self.recent_vol,
            self.est_slippage_small_bps / 100.0,
            self.est_slippage_medium_bps / 100.0,
            self.est_slippage_large_bps / 100.0,
            self.gas_cost / self.oracle_price,
            self.prev_action as f64 / N_ACTIONS as f64,
        ]
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct StepLog {
    pub step: usize,
    pub oracle_price: f64,
    pub pool_a_mid: f64,
    pub pool_b_mid: f64,
    pub pool_a_fee_buy: f64,
    pub pool_a_fee_sell: f64,
    pub pool_b_fee_buy: f64,
    pub pool_b_fee_sell: f64,
    pub action: String,
    pub agent_qty_a: f64,
    pub agent_qty_b: f64,
    pub agent_cash_flow_x: f64,
    pub agent_fee_paid_x: f64,
    pub agent_gas_x: f64,
    pub remaining_after: f64,
    pub post_trade_gap_a_bps: f64,
    pub post_trade_gap_b_bps: f64,
    pub noise_buy_a: f64,
    pub noise_buy_b: f64,
    pub noise_sell_a: f64,
    pub noise_sell_b: f64,
    pub arb_delta_a: f64,
    pub arb_delta_b: f64,
    pub reward: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct EpisodeSummary {
    pub mode: MarketMode,
    pub policy: String,
    pub seed: u64,
    /// Implementation shortfall vs oracle arrival price, bps of order value
    /// (positive = worse than arrival). Includes gas and terminal penalty.
    pub shortfall_bps: f64,
    pub completion_rate: f64,
    pub filled_qty: f64,
    pub avg_slippage_bps: f64,
    pub fees_paid_x: f64,
    pub gas_paid_x: f64,
    /// Herfindahl index over route volumes (1.0 = single pool).
    pub route_concentration: f64,
    /// Mean |oracle - pool mid| right after agent trades, bps.
    pub mean_post_trade_gap_bps: f64,
    pub arb_trade_count: usize,
    pub arb_profit_x: f64,
    pub n_steps: usize,
    pub total_reward: f64,
    // --- decomposition (bps of arrival notional; the identity
    // shortfall = drift + slippage_ex_fee + fee + gas + terminal holds) ---
    /// Timing/drift: qty * (contemporaneous oracle - arrival).
    pub drift_bps: f64,
    /// Execution cost vs contemporaneous oracle, excluding pool fees.
    pub slippage_ex_fee_bps: f64,
    pub fee_paid_bps: f64,
    pub gas_paid_bps: f64,
    pub terminal_penalty_bps: f64,
    /// Premium (incl. gas) of the ForcedTerminal completion leg, bps.
    pub forced_terminal_cost_bps: f64,
    // --- sensitivity sensitivity-layer telemetry (1.0 / 1.0 / 0 under baseline-duopoly defaults) ---
    pub avg_depth_factor: f64,
    pub min_depth_factor: f64,
    pub jit_event_count: usize,
    // --- behavior / market stats ---
    pub route_share_a: f64,
    pub route_share_b: f64,
    /// Fraction of steps with no executed trade.
    pub wait_share: f64,
    /// Mean start-of-step buy fee per pool.
    pub avg_fee_a: f64,
    pub avg_fee_b: f64,
    /// Mean start-of-step oracle gap per pool, bps.
    pub avg_oracle_gap_a_bps: f64,
    pub avg_oracle_gap_b_bps: f64,
}

pub struct StepResult {
    pub reward: f64,
    pub done: bool,
}

pub struct ExecEnv {
    pub cfg: EnvConfig,
    pools: Vec<Pool>,
    prices: Vec<f64>,
    t: usize,
    remaining: f64,
    filled: f64,
    /// X paid (buy) or received (sell) by the agent, fees included.
    cash_flow_x: f64,
    fees_paid_x: f64,
    gas_paid_x: f64,
    route_qty: [f64; 2],
    slippage_num: f64,
    drift_num: f64,
    forced_cost_x: f64,
    // LP-adaptation LP adaptation state
    depth_factor: [f64; 2],
    depth_factor_sum: f64,
    depth_factor_min: f64,
    arb_hits: [std::collections::VecDeque<bool>; 2],
    // JIT-searcher JIT state
    rng_jit: StdRng,
    jit_events: usize,
    wait_steps: usize,
    fee_sums: [f64; 2],
    gap_sums: [f64; 2],
    arb_count: usize,
    arb_profit: f64,
    prev_action: usize,
    total_reward: f64,
    done: bool,
    logs: Vec<StepLog>,
    rng_noise: StdRng,
    rng_arb: StdRng,
    rng_order: StdRng,
}

impl ExecEnv {
    pub fn new(cfg: EnvConfig) -> Self {
        let mut env = Self {
            pools: Vec::new(),
            prices: Vec::new(),
            t: 0,
            remaining: 0.0,
            filled: 0.0,
            cash_flow_x: 0.0,
            fees_paid_x: 0.0,
            gas_paid_x: 0.0,
            route_qty: [0.0; 2],
            slippage_num: 0.0,
            drift_num: 0.0,
            forced_cost_x: 0.0,
            depth_factor: [1.0; 2],
            depth_factor_sum: 0.0,
            depth_factor_min: 1.0,
            arb_hits: [Default::default(), Default::default()],
            rng_jit: StdRng::seed_from_u64(0),
            jit_events: 0,
            wait_steps: 0,
            fee_sums: [0.0; 2],
            gap_sums: [0.0; 2],
            arb_count: 0,
            arb_profit: 0.0,
            prev_action: 0,
            total_reward: 0.0,
            done: false,
            logs: Vec::new(),
            rng_noise: StdRng::seed_from_u64(0),
            rng_arb: StdRng::seed_from_u64(0),
            rng_order: StdRng::seed_from_u64(0),
            cfg,
        };
        env.reset(env.cfg.seed);
        env
    }

    pub fn reset(&mut self, seed: u64) {
        self.cfg.seed = seed;
        self.prices = gbm_path(
            self.cfg.order.horizon + 1,
            self.cfg.s0,
            self.cfg.mu,
            self.cfg.sigma,
            self.cfg.dt,
            seed,
        );
        let fee = self.cfg.constant_fee;
        self.pools = (0..2)
            .map(|_| Pool::new(self.cfg.pool_reserve_x, self.cfg.pool_reserve_y, fee, fee))
            .collect();
        self.t = 0;
        self.remaining = self.cfg.order.quantity;
        self.filled = 0.0;
        self.cash_flow_x = 0.0;
        self.fees_paid_x = 0.0;
        self.gas_paid_x = 0.0;
        self.route_qty = [0.0; 2];
        self.slippage_num = 0.0;
        self.drift_num = 0.0;
        self.forced_cost_x = 0.0;
        self.depth_factor = [1.0; 2];
        self.depth_factor_sum = 0.0;
        self.depth_factor_min = 1.0;
        self.arb_hits = [Default::default(), Default::default()];
        self.jit_events = 0;
        self.wait_steps = 0;
        self.fee_sums = [0.0; 2];
        self.gap_sums = [0.0; 2];
        self.arb_count = 0;
        self.arb_profit = 0.0;
        self.prev_action = 0;
        self.total_reward = 0.0;
        self.done = false;
        self.logs.clear();
        self.rng_noise = StdRng::seed_from_u64(seed.wrapping_mul(2654435761).wrapping_add(1));
        self.rng_arb = StdRng::seed_from_u64(seed.wrapping_mul(2246822519).wrapping_add(2));
        self.rng_order = StdRng::seed_from_u64(seed.wrapping_mul(3266489917).wrapping_add(3));
        self.rng_jit = StdRng::seed_from_u64(seed.wrapping_mul(668265263).wrapping_add(4));
        self.update_fees();
    }

    pub fn arrival_price(&self) -> f64 {
        self.prices[0]
    }

    fn oracle_now(&self) -> f64 {
        self.prices[self.t]
    }

    fn update_fees(&mut self) {
        let oracle = self.oracle_now();
        let imb: Vec<f64> = self
            .pools
            .iter()
            .map(|p| p.inventory_imbalance(oracle))
            .collect();
        for i in 0..self.cfg.n_live_pools() {
            let inputs = FeeInputs {
                own_imbalance: imb[i],
                rival_imbalance: if self.cfg.n_live_pools() > 1 {
                    imb[1 - i]
                } else {
                    0.0
                },
                oracle_misalignment: (self.pools[i].mid_price() - oracle) / oracle,
            };
            let pair = match self.cfg.mode {
                MarketMode::ConstantDuopoly => ConstantFee {
                    fee: self.cfg.constant_fee,
                }
                .fees(&inputs),
                _ => self.cfg.dynamic_fee.fees(&inputs),
            };
            self.pools[i].fee_buy = pair.buy;
            self.pools[i].fee_sell = pair.sell;
        }
    }

    /// Effective execution price for a fraction of remaining on the best
    /// live pool, as slippage vs oracle in bps (positive = costly).
    fn est_slippage_bps(&self, frac: f64) -> f64 {
        let qty = (self.remaining * frac).max(1e-9);
        let oracle = self.oracle_now();
        let best = (0..self.cfg.n_live_pools())
            .filter_map(|i| match self.cfg.order.side {
                Side::Buy => self.pools[i].effective_buy_price(qty),
                Side::Sell => self.pools[i].effective_sell_price(qty),
            })
            .fold(f64::NAN, |acc, p| {
                if acc.is_nan() {
                    p
                } else {
                    match self.cfg.order.side {
                        Side::Buy => acc.min(p),
                        Side::Sell => acc.max(p),
                    }
                }
            });
        if best.is_nan() {
            return f64::MAX;
        }
        match self.cfg.order.side {
            Side::Buy => (best - oracle) / oracle * 10_000.0,
            Side::Sell => (oracle - best) / oracle * 10_000.0,
        }
    }

    pub fn observe(&self) -> Observation {
        let oracle = self.oracle_now();
        let (a, b) = (&self.pools[0], &self.pools[1]);
        Observation {
            step: self.t,
            remaining_inventory: self.remaining,
            remaining_frac: self.remaining / self.cfg.order.quantity,
            remaining_time_frac: (self.cfg.order.horizon - self.t) as f64
                / self.cfg.order.horizon as f64,
            oracle_price: oracle,
            pool_a_inventory_y: a.reserve_y,
            pool_b_inventory_y: b.reserve_y,
            pool_a_mid: a.mid_price(),
            pool_b_mid: b.mid_price(),
            pool_a_fee_buy: a.fee_buy,
            pool_a_fee_sell: a.fee_sell,
            pool_b_fee_buy: b.fee_buy,
            pool_b_fee_sell: b.fee_sell,
            pool_a_oracle_gap_bps: (a.mid_price() - oracle) / oracle * 10_000.0,
            pool_b_oracle_gap_bps: (b.mid_price() - oracle) / oracle * 10_000.0,
            rival_quote_gap_bps: (a.mid_price() - b.mid_price()) / oracle * 10_000.0,
            recent_vol: rolling_vol(&self.prices, self.t, self.cfg.vol_window, self.cfg.dt),
            est_slippage_small_bps: self.est_slippage_bps(0.10),
            est_slippage_medium_bps: self.est_slippage_bps(0.25),
            est_slippage_large_bps: self.est_slippage_bps(0.50),
            gas_cost: self.cfg.agent_gas_cost,
            prev_action: self.prev_action,
        }
    }

    /// Agent leg of a step: execute `fracs` of remaining on each pool.
    /// Returns (qty, cash, fee_paid, gas, post_trade_gap_a, post_trade_gap_b).
    fn agent_trades(
        &mut self,
        fracs: &[f64; 2],
        oracle: f64,
    ) -> ([f64; 2], f64, f64, f64, f64, f64) {
        let arrival = self.arrival_price();
        let mut qty = [0.0; 2];
        let mut cash = 0.0;
        let mut fee_paid = 0.0;
        let mut gas = 0.0;
        for i in 0..2 {
            let q = self.remaining * fracs[i];
            if q <= 1e-12 {
                continue;
            }
            // JIT-searcher: searcher sandwich around large agent trades
            let mut jit_q = 0.0;
            if let Some((threshold, frac, prob)) = self.cfg.jit_regime.params()
                && q >= threshold
                && self.rng_jit.gen_range(0.0..1.0) < prob
            {
                jit_q = frac * q;
                let front = match self.cfg.order.side {
                    Side::Buy => self.pools[i].buy(jit_q).is_some(),
                    Side::Sell => self.pools[i].sell(jit_q).is_some(),
                };
                if front {
                    self.jit_events += 1;
                } else {
                    jit_q = 0.0;
                }
            }
            let fill = match self.cfg.order.side {
                Side::Buy => self.pools[i].buy(q),
                Side::Sell => self.pools[i].sell(q),
            };
            if jit_q > 0.0 {
                // back leg: searcher unwinds after the agent's fill
                match self.cfg.order.side {
                    Side::Buy => self.pools[i].sell(jit_q),
                    Side::Sell => self.pools[i].buy(jit_q),
                };
            }
            if let Some(f) = fill {
                qty[i] = q;
                cash += f.amount_x;
                fee_paid += f.fee_x;
                gas += self.cfg.agent_gas_cost;
                self.route_qty[i] += q;
                let (slip, drift) = match self.cfg.order.side {
                    Side::Buy => (f.amount_x - q * oracle, q * (oracle - arrival)),
                    Side::Sell => (q * oracle - f.amount_x, q * (arrival - oracle)),
                };
                self.slippage_num += slip;
                self.drift_num += drift;
            }
        }
        let traded = qty[0] + qty[1];
        self.remaining -= traded;
        self.filled += traded;
        self.cash_flow_x += cash;
        self.fees_paid_x += fee_paid;
        self.gas_paid_x += gas;
        let post_gap = |p: &Pool| (p.mid_price() - oracle) / oracle * 10_000.0;
        (
            qty,
            cash,
            fee_paid,
            gas,
            post_gap(&self.pools[0]),
            post_gap(&self.pools[1]),
        )
    }

    /// Agent leg: the chosen action, plus (under ForcedTerminal) mandatory
    /// execution of any inventory left at the final step, split equally
    /// across live pools. The forced premium is tracked separately.
    fn agent_leg(&mut self, fracs: &[f64; 2], oracle: f64) -> ([f64; 2], f64, f64, f64, f64, f64) {
        let (mut qty, mut cash, mut fee, mut gas, mut pa, mut pb) =
            self.agent_trades(fracs, oracle);
        if self.cfg.completion_rule == CompletionRule::ForcedTerminal
            && self.t + 1 >= self.cfg.order.horizon
            && self.remaining > 1e-9
        {
            let forced_fracs = if self.cfg.n_live_pools() == 1 {
                [1.0, 0.0]
            } else {
                [0.5, 0.5]
            };
            let arrival = self.arrival_price();
            let (fq, fc, ff, fg, fpa, fpb) = self.agent_trades(&forced_fracs, oracle);
            let traded = fq[0] + fq[1];
            let premium = match self.cfg.order.side {
                Side::Buy => fc - traded * arrival,
                Side::Sell => traded * arrival - fc,
            } + fg;
            self.forced_cost_x += premium;
            qty[0] += fq[0];
            qty[1] += fq[1];
            cash += fc;
            fee += ff;
            gas += fg;
            pa = fpa;
            pb = fpb;
        }
        (qty, cash, fee, gas, pa, pb)
    }

    /// Market leg of a step: noise flow then arbitrage on each live pool.
    fn market_flows(&mut self, oracle: f64) -> (NoiseFlows, [f64; 2]) {
        let live = self.cfg.n_live_pools();
        let flows = noise_flows(
            &self.cfg.noise,
            &self.pools[..live],
            oracle,
            &mut self.rng_noise,
        );
        for i in 0..live {
            if flows.buys[i] > 0.0 {
                self.pools[i].buy(flows.buys[i]);
            }
            if flows.sells[i] > 0.0 {
                self.pools[i].sell(flows.sells[i]);
            }
        }
        let mut arb_deltas = [0.0; 2];
        for (i, delta) in arb_deltas.iter_mut().enumerate().take(live) {
            if let Some(t) =
                arbitrage_step(&mut self.pools[i], oracle, &self.cfg.arb, &mut self.rng_arb)
            {
                *delta = t.delta_y;
                self.arb_count += 1;
                self.arb_profit += t.profit_x;
            }
        }
        self.apply_lp_adaptation(live, &arb_deltas);
        (flows, arb_deltas)
    }

    /// LP-adaptation: depth response to toxic flow (arb hits in a trailing window).
    /// No-op under LpRegime::Frozen (baseline-duopoly semantics).
    fn apply_lp_adaptation(&mut self, live: usize, arb_deltas: &[f64; 2]) {
        let (hits_needed, withdraw, recover, floor) = match self.cfg.lp_regime {
            LpRegime::Frozen => {
                self.depth_factor_sum += 1.0;
                return;
            }
            LpRegime::Weak => (3usize, 0.997f64, 1.001f64, 0.90f64),
            LpRegime::Aggressive => (2, 0.98, 1.0005, 0.50),
        };
        for (i, &delta) in arb_deltas.iter().enumerate().take(live) {
            let w = &mut self.arb_hits[i];
            w.push_back(delta != 0.0);
            if w.len() > 10 {
                w.pop_front();
            }
            let hits = w.iter().filter(|h| **h).count();
            let g = if hits >= hits_needed {
                withdraw
            } else {
                recover
            };
            let new_factor = (self.depth_factor[i] * g).clamp(floor, 1.0);
            let scale = new_factor / self.depth_factor[i];
            self.pools[i].reserve_x *= scale;
            self.pools[i].reserve_y *= scale;
            self.depth_factor[i] = new_factor;
            self.depth_factor_min = self.depth_factor_min.min(new_factor);
        }
        self.depth_factor_sum += (self.depth_factor[0] + self.depth_factor[1]) / 2.0;
    }

    /// Execute one step. Returns reward (normalized by order notional at
    /// arrival, so -1e-4 == 1 bps of shortfall) and done flag.
    pub fn step(&mut self, action: usize) -> StepResult {
        assert!(!self.done, "episode finished; call reset()");
        let action = action.min(N_ACTIONS - 1);
        let (name, mut fracs) = action_spec(action);
        if self.cfg.n_live_pools() == 1 {
            // Monopoly: all volume routes to pool A.
            fracs = [fracs[0] + fracs[1], 0.0];
        }
        let oracle = self.oracle_now();
        let arrival = self.arrival_price();
        let notional = self.cfg.order.quantity * arrival;

        // start-of-step market stats (decision-time fees and gaps)
        for i in 0..2 {
            self.fee_sums[i] += match self.cfg.order.side {
                Side::Buy => self.pools[i].fee_buy,
                Side::Sell => self.pools[i].fee_sell,
            };
            self.gap_sums[i] += (self.pools[i].mid_price() - oracle) / oracle * 10_000.0;
        }

        // decision-time fees, captured before any leg executes: update_fees
        // at the end of this step overwrites pool fees with NEXT-step values,
        // so the trajectory log must not read them back afterwards
        let decision_fees = [
            (self.pools[0].fee_buy, self.pools[0].fee_sell),
            (self.pools[1].fee_buy, self.pools[1].fee_sell),
        ];

        let agent_first = match self.cfg.agent_order {
            AgentOrder::Before => true,
            AgentOrder::After => false,
            AgentOrder::Random => self.rng_order.gen_range(0.0..1.0) < 0.5,
        };

        let (qty, cash, fee_paid, gas, post_a, post_b, flows, arb_deltas);
        if agent_first {
            (qty, cash, fee_paid, gas, post_a, post_b) = self.agent_leg(&fracs, oracle);
            (flows, arb_deltas) = self.market_flows(oracle);
        } else {
            (flows, arb_deltas) = self.market_flows(oracle);
            (qty, cash, fee_paid, gas, post_a, post_b) = self.agent_leg(&fracs, oracle);
        }
        let traded = qty[0] + qty[1];
        if traded <= 0.0 {
            self.wait_steps += 1;
        }

        // 5) advance time, fees for next step
        self.t += 1;
        self.update_fees();

        // reward: negative execution cost vs arrival, normalized by notional
        let step_cost = match self.cfg.order.side {
            Side::Buy => cash - traded * arrival,
            Side::Sell => traded * arrival - cash,
        } + gas;
        let mut reward = -step_cost / notional;

        if self.t >= self.cfg.order.horizon {
            self.done = true;
            let penalty = self.remaining * arrival * self.cfg.unfinished_penalty;
            reward -= penalty / notional;
        }
        self.total_reward += reward;
        self.prev_action = action;

        self.logs.push(StepLog {
            step: self.t - 1,
            oracle_price: oracle,
            pool_a_mid: self.pools[0].mid_price(),
            pool_b_mid: self.pools[1].mid_price(),
            pool_a_fee_buy: decision_fees[0].0,
            pool_a_fee_sell: decision_fees[0].1,
            pool_b_fee_buy: decision_fees[1].0,
            pool_b_fee_sell: decision_fees[1].1,
            action: name.to_string(),
            agent_qty_a: qty[0],
            agent_qty_b: qty[1],
            agent_cash_flow_x: cash,
            agent_fee_paid_x: fee_paid,
            agent_gas_x: gas,
            remaining_after: self.remaining,
            post_trade_gap_a_bps: post_a,
            post_trade_gap_b_bps: post_b,
            noise_buy_a: flows.buys[0],
            noise_buy_b: flows.buys[1],
            noise_sell_a: flows.sells[0],
            noise_sell_b: flows.sells[1],
            arb_delta_a: arb_deltas[0],
            arb_delta_b: arb_deltas[1],
            reward,
        });

        StepResult {
            reward,
            done: self.done,
        }
    }

    pub fn is_done(&self) -> bool {
        self.done
    }

    pub fn summary(&self, policy: &str) -> EpisodeSummary {
        let arrival = self.arrival_price();
        let notional = self.cfg.order.quantity * arrival;
        let exec_cost = match self.cfg.order.side {
            Side::Buy => self.cash_flow_x - self.filled * arrival,
            Side::Sell => self.filled * arrival - self.cash_flow_x,
        };
        let penalty = self.remaining * arrival * self.cfg.unfinished_penalty;
        let shortfall = exec_cost + self.gas_paid_x + penalty;
        let route_total = self.route_qty[0] + self.route_qty[1];
        let hhi = if route_total > 0.0 {
            (self.route_qty[0] / route_total).powi(2) + (self.route_qty[1] / route_total).powi(2)
        } else {
            1.0
        };
        let mean_post_gap = if self.logs.is_empty() {
            0.0
        } else {
            self.logs
                .iter()
                .filter(|l| l.agent_qty_a + l.agent_qty_b > 0.0)
                .map(|l| {
                    l.post_trade_gap_a_bps
                        .abs()
                        .max(l.post_trade_gap_b_bps.abs())
                })
                .sum::<f64>()
                / self
                    .logs
                    .iter()
                    .filter(|l| l.agent_qty_a + l.agent_qty_b > 0.0)
                    .count()
                    .max(1) as f64
        };
        let n_steps = self.t.max(1) as f64;
        let bps = |x: f64| x / notional * 10_000.0;
        EpisodeSummary {
            mode: self.cfg.mode,
            policy: policy.to_string(),
            seed: self.cfg.seed,
            drift_bps: bps(self.drift_num),
            slippage_ex_fee_bps: bps(self.slippage_num - self.fees_paid_x),
            fee_paid_bps: bps(self.fees_paid_x),
            gas_paid_bps: bps(self.gas_paid_x),
            terminal_penalty_bps: bps(penalty),
            forced_terminal_cost_bps: bps(self.forced_cost_x),
            avg_depth_factor: self.depth_factor_sum / n_steps,
            min_depth_factor: self.depth_factor_min,
            jit_event_count: self.jit_events,
            route_share_a: if route_total > 0.0 {
                self.route_qty[0] / route_total
            } else {
                0.0
            },
            route_share_b: if route_total > 0.0 {
                self.route_qty[1] / route_total
            } else {
                0.0
            },
            wait_share: self.wait_steps as f64 / n_steps,
            avg_fee_a: self.fee_sums[0] / n_steps,
            avg_fee_b: self.fee_sums[1] / n_steps,
            avg_oracle_gap_a_bps: self.gap_sums[0] / n_steps,
            avg_oracle_gap_b_bps: self.gap_sums[1] / n_steps,
            shortfall_bps: shortfall / notional * 10_000.0,
            completion_rate: self.filled / self.cfg.order.quantity,
            filled_qty: self.filled,
            avg_slippage_bps: if self.filled > 0.0 {
                self.slippage_num / (self.filled * arrival) * 10_000.0
            } else {
                0.0
            },
            fees_paid_x: self.fees_paid_x,
            gas_paid_x: self.gas_paid_x,
            route_concentration: hhi,
            mean_post_trade_gap_bps: mean_post_gap,
            arb_trade_count: self.arb_count,
            arb_profit_x: self.arb_profit,
            n_steps: self.t,
            total_reward: self.total_reward,
        }
    }

    pub fn write_trajectory_csv(&self, path: &str) -> std::io::Result<()> {
        if let Some(dir) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut w = csv::Writer::from_path(path)?;
        for log in &self.logs {
            w.serialize(log)?;
        }
        w.flush()?;
        Ok(())
    }

    pub fn logs(&self) -> &[StepLog] {
        &self.logs
    }
}

/// Write a batch of episode summaries to CSV.
pub fn write_summaries_csv(path: &str, rows: &[EpisodeSummary]) -> std::io::Result<()> {
    if let Some(dir) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut w = csv::Writer::from_path(path)?;
    for s in rows {
        w.serialize(s)?;
    }
    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn episode_is_deterministic_given_seed() {
        let run = || {
            let mut env = ExecEnv::new(EnvConfig::baseline(MarketMode::DynamicDuopoly, 42));
            let mut total = 0.0;
            while !env.is_done() {
                total += env.step(7).reward;
            }
            (total, env.summary("test").shortfall_bps)
        };
        let (r1, s1) = run();
        let (r2, s2) = run();
        assert_eq!(r1, r2);
        assert_eq!(s1, s2);
    }

    #[test]
    fn agent_trades_move_the_pool() {
        let mut env = ExecEnv::new(EnvConfig::baseline(MarketMode::ConstantDuopoly, 7));
        let mid_before = env.observe().pool_a_mid;
        env.step(5); // 50% of a 50 Y order on pool A
        let log = env.logs().last().unwrap();
        assert!(log.agent_qty_a > 0.0);
        assert!(log.pool_a_mid != mid_before || log.arb_delta_a != 0.0);
    }

    /// baseline-duopoly-E: observation schema whitelist. Every field must be a
    /// decision-time quantity; adding a future-information field (future
    /// oracle, upcoming noise/arb) breaks this test until reviewed.
    #[test]
    fn observation_schema_is_current_state_only() {
        let env = ExecEnv::new(EnvConfig::baseline(MarketMode::DynamicDuopoly, 1));
        let val = serde_json::to_value(env.observe()).unwrap();
        let allowed = [
            "step",
            "remaining_inventory",
            "remaining_frac",
            "remaining_time_frac",
            "oracle_price",
            "pool_a_inventory_y",
            "pool_b_inventory_y",
            "pool_a_mid",
            "pool_b_mid",
            "pool_a_fee_buy",
            "pool_a_fee_sell",
            "pool_b_fee_buy",
            "pool_b_fee_sell",
            "pool_a_oracle_gap_bps",
            "pool_b_oracle_gap_bps",
            "rival_quote_gap_bps",
            "recent_vol",
            "est_slippage_small_bps",
            "est_slippage_medium_bps",
            "est_slippage_large_bps",
            "gas_cost",
            "prev_action",
        ];
        let fields: Vec<&str> = val
            .as_object()
            .unwrap()
            .keys()
            .map(|s| s.as_str())
            .collect();
        for f in &fields {
            assert!(allowed.contains(f), "unreviewed observation field: {f}");
        }
        assert_eq!(fields.len(), allowed.len());
    }

    /// baseline-duopoly-E: reward decomposition regression. The identity
    /// shortfall = drift + slippage_ex_fee + fee + gas + terminal_penalty
    /// must hold for every policy path, both completion rules.
    #[test]
    fn shortfall_decomposition_identity() {
        for rule in [CompletionRule::Standard, CompletionRule::ForcedTerminal] {
            for seed in [1u64, 7, 99] {
                for action in [0usize, 1, 5, 7] {
                    let mut cfg = EnvConfig::baseline(MarketMode::DynamicDuopoly, seed);
                    cfg.completion_rule = rule;
                    let mut env = ExecEnv::new(cfg);
                    while !env.is_done() {
                        env.step(action);
                    }
                    let s = env.summary("test");
                    let sum = s.drift_bps
                        + s.slippage_ex_fee_bps
                        + s.fee_paid_bps
                        + s.gas_paid_bps
                        + s.terminal_penalty_bps;
                    assert!(
                        (sum - s.shortfall_bps).abs() < 1e-6,
                        "decomposition broke: {sum} vs {} (rule {rule:?}, seed {seed}, action {action})",
                        s.shortfall_bps
                    );
                }
            }
        }
    }

    /// sensitivity: sensitivity layers must be exact no-ops at their defaults.
    #[test]
    fn m4_defaults_preserve_m3r_semantics() {
        let run = |lp: LpRegime, jit: JitRegime| {
            let mut cfg = EnvConfig::baseline(MarketMode::DynamicDuopoly, 42);
            cfg.completion_rule = CompletionRule::ForcedTerminal;
            cfg.lp_regime = lp;
            cfg.jit_regime = jit;
            let mut env = ExecEnv::new(cfg);
            while !env.is_done() {
                env.step(5);
            }
            let s = env.summary("t");
            (s.shortfall_bps, s.avg_depth_factor, s.jit_event_count)
        };
        let (base, depth, jits) = run(LpRegime::Frozen, JitRegime::None);
        assert_eq!(depth, 1.0);
        assert_eq!(jits, 0);
        // regimes actually bite
        let (aggr_lp, depth_a, _) = run(LpRegime::Aggressive, JitRegime::None);
        assert!(depth_a < 1.0);
        assert_ne!(base, aggr_lp);
        let (aggr_jit, _, jits_a) = run(LpRegime::Frozen, JitRegime::Aggressive);
        assert!(jits_a > 0);
        assert!(aggr_jit > base, "sandwich must worsen execution");
    }

    #[test]
    fn forced_terminal_completes_fully() {
        let mut cfg = EnvConfig::baseline(MarketMode::DynamicDuopoly, 5);
        cfg.completion_rule = CompletionRule::ForcedTerminal;
        let mut env = ExecEnv::new(cfg);
        while !env.is_done() {
            env.step(0); // wait the entire episode
        }
        let s = env.summary("wait");
        assert!((s.completion_rate - 1.0).abs() < 1e-9);
        assert!(s.terminal_penalty_bps.abs() < 1e-9);
        assert!(s.forced_terminal_cost_bps > 0.0);
    }

    #[test]
    fn waiting_forever_pays_unfinished_penalty() {
        let mut env = ExecEnv::new(EnvConfig::baseline(MarketMode::ConstantDuopoly, 3));
        let mut total = 0.0;
        while !env.is_done() {
            total += env.step(0).reward;
        }
        let s = env.summary("wait");
        assert_eq!(s.completion_rate, 0.0);
        assert!(total < 0.0);
        assert!((s.shortfall_bps - env.cfg.unfinished_penalty * 10_000.0).abs() < 1e-6);
    }
}
