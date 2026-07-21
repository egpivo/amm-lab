use crate::campbell::fee_policy::{FeeObservation, FeePolicy, TabularLearnedFeePolicy};
use crate::campbell::pool::CampbellPool;
use crate::campbell::trader::{arb_delta, fundamental_buy_delta, fundamental_sell_delta};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Poisson};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Default scenario shared by `campbell_rl_fee_train` and `campbell_rl_fee_compare`.
pub const DEFAULT_RL_SCENARIO: &str = "scenarios/campbell_rl_normal.toml";

pub fn load_sim_config(path: Option<&str>) -> SimConfig {
    let path = path.unwrap_or(DEFAULT_RL_SCENARIO);
    let toml_str =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    toml::from_str(&toml_str).unwrap_or_else(|e| panic!("invalid TOML {path}: {e}"))
}

// ── Flow regime ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FlowRegime {
    #[default]
    Normal,
    ToxicBurst,
    RegimeSwitch,
}

fn default_scale_one() -> f64 {
    1.0
}
fn default_e1_fee_ref() -> f64 {
    0.0006
}

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimConfig {
    pub name: String,
    pub description: String,
    // pool parameters
    pub amm_fee: f64,
    pub cex_fee: f64,
    pub buy_demand: f64,
    pub sell_demand: f64,
    pub reserve_x: f64,
    pub reserve_y: f64,
    // GBM parameters
    pub sigma: f64,
    pub mu: f64,
    pub n_steps: usize,
    pub seed: u64,
    // Flow regime (all optional; existing TOML files without these fields use defaults)
    #[serde(default)]
    pub flow_regime: FlowRegime,
    #[serde(default)]
    pub toxic_burst_prob: f64,
    #[serde(default = "default_scale_one")]
    pub toxic_burst_arb_scale: f64,
    #[serde(default = "default_scale_one")]
    pub toxic_burst_fund_scale: f64,
    #[serde(default)]
    pub regime_switch_period: usize,
    // C1a/E1 router-substitution stress (additive; default 0 = disabled, existing TOMLs unaffected)
    #[serde(default)]
    pub e1_lambda: f64,
    #[serde(default = "default_e1_fee_ref")]
    pub e1_fee_ref: f64,
    // E5 arbitrage-latency stress (additive; default 1.0 = arb every step, no latency)
    #[serde(default = "default_scale_one")]
    pub e5_arb_prob: f64,
    // policy-selected (lvr paper): policy information-set lag in steps. 0 = zero-lag
    // baseline (fee decision sees the contemporaneous observation and
    // applies to the same step's fills); 1 = one-step lag f_t = pi(Z_{t-1}),
    // the PRIMARY specification per round 11.
    #[serde(default)]
    pub policy_lag: usize,
    // physical-clock (lvr paper): physical step length in hours. Converts arrival
    // rates into per-step probabilities and defines the GBM dt via
    // `SimConfig::dt_years()` — experiment binaries MUST generate paths
    // from that helper (round 13: no mixed clocks) and write dt_hours /
    // dt_years into their output manifests.
    #[serde(default = "default_scale_one")]
    pub dt_hours: f64,
    // physical-clock (round 13, POOLED semantics): latent fundamental-demand arrival
    // rate in events per hour, POOLED across sides, split by
    // `buy_arrival_share` r into lambda_buy = r*lam, lambda_sell =
    // (1-r)*lam; each side independently arrives with
    // p_side = 1 - exp(-lambda_side * dt_hours), drawn from a dedicated
    // seeded RNG so the arrival sequence is POLICY-INVARIANT given the
    // seed (paired comparisons). None = legacy behavior, demand present
    // every step. No arrival => zero potential demand on that side
    // (pot_* = 0 ledger row). NOTE: this rate is a latent input calibrated
    // so that the static baseline's TOTAL realized swap incidence
    // (arb + fundamental) matches observed activity quantiles; observed
    // swap counts are never plugged in directly (they mix arbitrage and
    // other flow).
    #[serde(default)]
    pub pooled_fund_arrival_rate_per_hour: Option<f64>,
    // physical-clock (round 13): share of pooled arrivals on the buy side.
    // Primary 0.5; observed side imbalance is a robustness axis.
    #[serde(default = "default_half")]
    pub buy_arrival_share: f64,
    // physical-clock (round 13): physical-time arbitrage activation. Some(lam):
    // p_arb = 1 - exp(-lam * dt_hours), overriding e5_arb_prob, so the
    // arb-speed axis survives clock changes. None = legacy per-step
    // Bernoulli e5_arb_prob (whose physical meaning changes with the
    // clock and must then be reported as a mean waiting time).
    #[serde(default)]
    pub arb_arrival_rate_per_hour: Option<f64>,
    // physical-clock (round 13): physical lookback for the policy observation
    // window (rolling vol, arb/fund fractions, volume). window_steps =
    // round(lookback_hours / dt_hours), so the information set is
    // clock-invariant. Default 20 h reproduces the legacy 20-step window
    // on the hourly clock.
    #[serde(default = "default_lookback_hours")]
    pub lookback_hours: f64,
    // Poisson-arrival (round 21): arrival model. Bernoulli = legacy at-most-one
    // demand event per side per step. Poisson = side-specific counts
    // K ~ Poisson(lambda_side * dt) with policy-invariant intra-step
    // times; each arrival is a separate PRIMITIVE demand event executed
    // sequentially (never batched: CPMM execution is nonlinear in size
    // and incidence is defined per event); the hook policy re-evaluates
    // per primitive swap using CURRENT pool state and the SAME (possibly
    // lagged) external signal. Poisson mode is Eval-only.
    #[serde(default)]
    pub arrival_model: ArrivalModel,
    /// When true, emit an arb ledger row at every step (including inactive
    /// legs with delta=0). Used for paired-ledger decomposition only.
    #[serde(default)]
    pub log_inactive_arb: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ArrivalModel {
    #[default]
    Bernoulli,
    Poisson,
}

/// One PRIMITIVE market event (round 21): the unit of the destination
/// ledger and of fill-level A/B accounting. Emitted in BOTH arrival
/// modes (Bernoulli steps produce up to arb + buy + sell), so summaries
/// and low-rate equivalence tests share one representation.
#[derive(Debug, Clone, Serialize)]
pub struct EventRecord {
    pub step: usize,
    /// Intra-step order key in [0,1); 0 for the arbitrage leg.
    pub time_frac: f64,
    pub kind: EventKind,
    pub delta: f64,
    pub pbar: f64,
    pub ell: f64,
    pub pot: f64,
    pub cex: f64,
    pub unserved: f64,
    pub fee_used: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    Arb,
    FundBuy,
    FundSell,
}

fn default_half() -> f64 {
    0.5
}
fn default_lookback_hours() -> f64 {
    20.0
}

impl SimConfig {
    /// GBM step length in YEARS implied by the physical clock. Experiment
    /// binaries must generate price paths with this dt (round 13).
    pub fn dt_years(&self) -> f64 {
        self.dt_hours / (24.0 * 365.0)
    }
    /// Observation-window length in steps implied by the physical
    /// lookback (minimum 1).
    pub fn window_steps(&self) -> usize {
        (self.lookback_hours / self.dt_hours).round().max(1.0) as usize
    }
}

// ── Records ───────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct StepRecord {
    pub step: usize,
    pub cex_price: f64,
    pub amm_price: f64,
    pub arb_delta: f64,
    pub buy_delta: f64,
    pub sell_delta: f64,
    pub step_fee: f64,
    pub pool_value: f64,
    pub hedging_portfolio: f64,
    pub pool_x: f64,
    pub pool_y: f64,
    pub fee_used: f64,
    pub oracle_gap_bps: f64,
    pub inventory_skew: f64,
    // C0 instrumentation (additive): step fee revenue split by trade leg
    pub step_fee_arb: f64,
    pub step_fee_fund: f64,
    // C1a/E1 instrumentation (additive)
    pub fund_retention: f64,
    pub fund_demand_lost: f64,
    // E5 instrumentation (additive)
    pub arb_active: bool,
    // policy-selected (lvr paper) per-fill gap accounting: ell = delta * (P - pbar)
    // with pbar the exact CPMM average execution price of that fill.
    // A_T = sum of positive parts, B_T = sum of negative parts
    // (Jordan decomposition on primitive fill events).
    pub ell_arb: f64,
    pub ell_buy: f64,
    pub ell_sell: f64,
    pub pbar_arb: f64,
    pub pbar_buy: f64,
    pub pbar_sell: f64,
    // policy-selected destination ledger per fundamental demand event (risky units):
    // potential = AMM fill + CEX-routed + unserved (router-substitution).
    pub pot_buy: f64,
    pub pot_sell: f64,
    pub cex_buy: f64,
    pub cex_sell: f64,
    pub unserved_buy: f64,
    pub unserved_sell: f64,
    // policy-selected quote accuracy: |log P - log p_amm| at end of step (post-fills).
    pub log_gap_abs: f64,
}

#[derive(Debug, Serialize)]
pub struct SimSummary {
    pub scenario_name: String,
    pub config: SimConfig,
    pub n_steps: usize,
    pub initial_cex_price: f64,
    pub final_cex_price: f64,
    pub final_amm_price: f64,
    pub final_pool_value: f64,
    pub final_hedging_portfolio: f64,
    pub total_fee_revenue: f64,
    pub tracking_error: f64,
    pub hedged_pnl: f64,
}

// ── Simulation ────────────────────────────────────────────────────────────────

pub fn run_simulation(
    config: &SimConfig,
    cex_prices: &[f64],
    policy: &mut dyn FeePolicy,
) -> Vec<StepRecord> {
    run_simulation_with_events(config, cex_prices, policy).0
}

/// External schedules for the clock-convergence diagnostic (round 23):
/// per-step fundamental primitive events (sorted (time_frac, is_buy))
/// and per-step arbitrage activation, generated ONCE from a shared
/// continuous-time base (15-second Brownian increments, continuous-time
/// Poisson event/opportunity times) and BINNED per clock by the caller.
/// Injection bypasses the internal arrival/latency draws so that
/// different clocks compare the same market path, latent demand, and
/// arb opportunities.
pub struct InjectedSchedules {
    pub fund: Vec<Vec<(f64, bool)>>,
    pub arb_active: Vec<bool>,
}

/// Poisson-mode run with externally injected schedules (Eval only).
pub fn run_simulation_with_injected_schedules(
    config: &SimConfig,
    cex_prices: &[f64],
    policy: &mut dyn FeePolicy,
    injected: &InjectedSchedules,
) -> (Vec<StepRecord>, Vec<EventRecord>) {
    assert_eq!(config.arrival_model, ArrivalModel::Poisson);
    assert_eq!(injected.fund.len(), config.n_steps);
    assert_eq!(injected.arb_active.len(), config.n_steps);
    let mut records = Vec::new();
    let mut events = Vec::new();
    run_episode_inner(
        config,
        cex_prices,
        EpisodeMode::Eval(policy),
        &mut |_, _, _, _| {},
        &mut records,
        &mut events,
        Some(injected),
        false,
    );
    (records, events)
}

/// Like [`run_simulation`], additionally returning the PRIMITIVE event
/// ledger (round 21). The ledger is the source of truth for fill-level
/// A/B accounting and per-event destination conservation; step records
/// carry aggregates.
pub fn run_simulation_with_events(
    config: &SimConfig,
    cex_prices: &[f64],
    policy: &mut dyn FeePolicy,
) -> (Vec<StepRecord>, Vec<EventRecord>) {
    let mut records = Vec::new();
    let mut events = Vec::new();
    run_episode(
        config,
        cex_prices,
        EpisodeMode::Eval(policy),
        &mut |_, _, _, _| {},
        &mut records,
        &mut events,
    );
    (records, events)
}

/// Event-ledger run retaining only fee-bearing steps plus the terminal step.
///
/// [`crate::campbell::summary::summarize_events`] uses step records only for
/// fee sums and the terminal tracking error. This compact representation is
/// therefore summary-equivalent while avoiding one `StepRecord` allocation
/// per one-second clock tick in large policy-grid diagnostics.
pub fn run_simulation_with_events_compact(
    config: &SimConfig,
    cex_prices: &[f64],
    policy: &mut dyn FeePolicy,
) -> (Vec<StepRecord>, Vec<EventRecord>) {
    let mut records = Vec::new();
    let mut events = Vec::new();
    run_episode_inner(
        config,
        cex_prices,
        EpisodeMode::Eval(policy),
        &mut |_, _, _, _| {},
        &mut records,
        &mut events,
        None,
        true,
    );
    (records, events)
}

/// One training episode on the same path engine as [`run_simulation`], with TD(0) updates.
pub fn run_rl_training_episode(
    config: &SimConfig,
    cex_prices: &[f64],
    policy: &mut TabularLearnedFeePolicy,
    gamma: f64,
) -> (Vec<StepRecord>, f64) {
    let mut records = Vec::new();
    let mut episode_reward = 0.0;
    let mut events = Vec::new();
    run_episode(
        config,
        cex_prices,
        EpisodeMode::Train(policy, gamma),
        &mut |_, reward, _, _| {
            episode_reward += reward;
        },
        &mut records,
        &mut events,
    );
    (records, episode_reward)
}

enum EpisodeMode<'a> {
    Eval(&'a mut dyn FeePolicy),
    Train(&'a mut TabularLearnedFeePolicy, f64),
}

fn run_episode(
    config: &SimConfig,
    cex_prices: &[f64],
    mode: EpisodeMode<'_>,
    on_step_reward: &mut dyn FnMut(usize, f64, &FeeObservation, &FeeObservation),
    records: &mut Vec<StepRecord>,
    events: &mut Vec<EventRecord>,
) {
    run_episode_inner(
        config,
        cex_prices,
        mode,
        on_step_reward,
        records,
        events,
        None,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_episode_inner(
    config: &SimConfig,
    cex_prices: &[f64],
    mut mode: EpisodeMode<'_>,
    on_step_reward: &mut dyn FnMut(usize, f64, &FeeObservation, &FeeObservation),
    records: &mut Vec<StepRecord>,
    events: &mut Vec<EventRecord>,
    injected: Option<&InjectedSchedules>,
    compact_records: bool,
) {
    if config.arrival_model == ArrivalModel::Poisson {
        assert!(
            matches!(mode, EpisodeMode::Eval(_)),
            "Poisson arrival model is Eval-only (RL per-event training is unsupported)"
        );
        assert!(
            config.pooled_fund_arrival_rate_per_hour.is_some(),
            "Poisson arrival model requires pooled_fund_arrival_rate_per_hour"
        );
    }
    let mut pool = CampbellPool::new(config.reserve_x, config.reserve_y, config.amm_fee);
    let mut hedging = pool.pool_value(cex_prices[0]);
    let mut regime_rng = StdRng::seed_from_u64(config.seed.wrapping_add(99_991));
    let mut latency_rng = StdRng::seed_from_u64(config.seed ^ 77_777);
    let mut arrival_rng = StdRng::seed_from_u64(config.seed ^ 424_243);
    // Pooled rate split into per-side rates (round 13).
    let p_arrival = config.pooled_fund_arrival_rate_per_hour.map(|lam| {
        let r = config.buy_arrival_share.clamp(0.0, 1.0);
        (
            1.0 - (-r * lam * config.dt_hours).exp(),
            1.0 - (-(1.0 - r) * lam * config.dt_hours).exp(),
        )
    });
    let window = config.window_steps();

    let mut win = RollingWindows::new(window);

    let mut previous_fee = config.amm_fee;
    let mut prev_position = hedging - pool.pool_value(cex_prices[0]);
    // policy-selected/lagged-policy: lagged information set. Under policy_lag = L the fee
    // decision at step t sees the observation built at step t-L; the
    // first L decisions see the genuine t_0 observation (signal frozen at
    // t_0 during warm-up, disclosed). L is in STEPS; under a finer market
    // clock the 5-minute physical staleness converts as
    // L = round(5 min / dt) (round 22: clock refinement must not change
    // the physical signal staleness). RL training supports only L <= 1.
    if config.policy_lag > 1 {
        assert!(
            matches!(mode, EpisodeMode::Eval(_)),
            "policy_lag > 1 is Eval-only"
        );
    }
    let obs0 = build_observation(0, cex_prices[0], &pool, &win, previous_fee);
    let mut lag_buf: VecDeque<FeeObservation> = VecDeque::new();
    lag_buf.push_back(obs0);

    for (step, &cex_price) in cex_prices[1..].iter().enumerate() {
        let prev_cex = cex_prices[step];
        hedging += pool.reserve_y * (cex_price - prev_cex);
        let fee_before = pool.cumulative_fee_revenue;
        // Arrival draws happen every step, before and independent of the
        // policy decision, so the arrival sequence is identical across
        // policies on the same seed.
        let (arrivals, schedule) = if let Some(inj) = injected {
            ((false, false), inj.fund[step].clone())
        } else {
            match config.arrival_model {
                ArrivalModel::Bernoulli => (
                    match p_arrival {
                        None => (true, true),
                        Some((p_buy, p_sell)) => (
                            arrival_rng.gen_range(0.0f64..1.0) < p_buy,
                            arrival_rng.gen_range(0.0f64..1.0) < p_sell,
                        ),
                    },
                    Vec::new(),
                ),
                ArrivalModel::Poisson => {
                    // side-specific Poisson counts + uniform intra-step times,
                    // sorted; each entry is one PRIMITIVE demand event.
                    let lam = config.pooled_fund_arrival_rate_per_hour.unwrap();
                    let r = config.buy_arrival_share.clamp(0.0, 1.0);
                    let mut sched: Vec<(f64, bool)> = Vec::new();
                    for (rate, is_buy) in [(r * lam, true), ((1.0 - r) * lam, false)] {
                        let mean = rate * config.dt_hours;
                        let k = if mean > 0.0 {
                            Poisson::new(mean).unwrap().sample(&mut arrival_rng) as usize
                        } else {
                            0
                        };
                        for _ in 0..k {
                            sched.push((arrival_rng.gen_range(0.0f64..1.0), is_buy));
                        }
                    }
                    sched.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                    ((false, false), sched)
                }
            }
        };
        let arb_override = injected.map(|inj| inj.arb_active[step]);

        let obs = build_observation(step, cex_price, &pool, &win, previous_fee);
        let obs_used = if config.policy_lag >= 1 {
            // oldest buffered observation = the one from `lag` steps ago
            // (or t_0 during warm-up).
            lag_buf.front().unwrap().clone()
        } else {
            obs.clone()
        };
        lag_buf.push_back(obs.clone());
        while lag_buf.len() > config.policy_lag.max(1) {
            lag_buf.pop_front();
        }

        let mut decision_obs = obs_used.clone();
        decision_obs.contemporaneous_gap_bps = obs.oracle_gap_bps;
        let fee = match &mut mode {
            EpisodeMode::Eval(policy) => {
                if config.arrival_model == ArrivalModel::Poisson {
                    let (fills, last_fee) = poisson_step_eval(
                        config,
                        step,
                        cex_price,
                        &decision_obs,
                        *policy,
                        &mut pool,
                        &mut latency_rng,
                        &mut regime_rng,
                        &schedule,
                        fee_before,
                        &win,
                        events,
                        arb_override,
                    );
                    previous_fee = last_fee;
                    let cur_position = hedging - pool.pool_value(cex_price);
                    let step_reward = fills.step_fee - (cur_position - prev_position);
                    prev_position = cur_position;
                    on_step_reward(step, step_reward, &obs_used, &obs_used);
                    let record =
                        make_record(step, cex_price, &pool, hedging, last_fee, &obs, &fills);
                    if !compact_records || fills.step_fee != 0.0 || step + 2 == cex_prices.len() {
                        records.push(record);
                    }
                    win.push(
                        (cex_price / prev_cex).ln(),
                        fills.arb_d.abs() > 1e-12,
                        (fills.buy_d.abs() + fills.sell_d.abs()) > 1e-12,
                        fills.arb_d.abs() + fills.buy_d.abs() + fills.sell_d.abs(),
                    );
                    continue;
                }
                policy.fee(&decision_obs)
            }
            EpisodeMode::Train(policy, gamma) => {
                let state = policy.obs_to_state(&obs_used);
                let action = policy.choose_action(state);
                let fee = crate::campbell::fee_policy::RL_ACTIONS_BPS[action] / 10_000.0;
                pool.amm_fee = fee;
                previous_fee = fee;

                let fills = execute_trades(
                    config,
                    step,
                    cex_price,
                    fee,
                    &mut pool,
                    &mut regime_rng,
                    &mut latency_rng,
                    fee_before,
                    arrivals,
                );

                let cur_position = hedging - pool.pool_value(cex_price);
                let step_reward = fills.step_fee - (cur_position - prev_position);
                prev_position = cur_position;

                win.push(
                    (cex_price / prev_cex).ln(),
                    fills.arb_d.abs() > 1e-12,
                    (fills.buy_d.abs() + fills.sell_d.abs()) > 1e-12,
                    fills.arb_d.abs() + fills.buy_d.abs() + fills.sell_d.abs(),
                );

                let next_obs = build_observation(step + 1, cex_price, &pool, &win, fee);
                // Under lag, the observation the policy will act on next
                // step is the one built THIS step (pre-trade), not the
                // post-trade approximation.
                let next_state = if config.policy_lag >= 1 {
                    policy.obs_to_state(&obs)
                } else {
                    policy.obs_to_state(&next_obs)
                };
                policy.update_step(state, action, step_reward, next_state, *gamma);
                on_step_reward(step, step_reward, &obs_used, &next_obs);

                records.push(make_record(
                    step, cex_price, &pool, hedging, fee, &obs, &fills,
                ));
                push_step_events(events, step, &fills, fee);
                continue;
            }
        };

        pool.amm_fee = fee;
        previous_fee = fee;

        let fills = execute_trades(
            config,
            step,
            cex_price,
            fee,
            &mut pool,
            &mut regime_rng,
            &mut latency_rng,
            fee_before,
            arrivals,
        );

        let cur_position = hedging - pool.pool_value(cex_price);
        let step_reward = fills.step_fee - (cur_position - prev_position);
        prev_position = cur_position;
        on_step_reward(step, step_reward, &obs_used, &obs_used);

        let record = make_record(step, cex_price, &pool, hedging, fee, &obs, &fills);
        if !compact_records || fills.step_fee != 0.0 || step + 2 == cex_prices.len() {
            records.push(record);
        }
        push_step_events(events, step, &fills, fee);

        win.push(
            (cex_price / prev_cex).ln(),
            fills.arb_d.abs() > 1e-12,
            (fills.buy_d.abs() + fills.sell_d.abs()) > 1e-12,
            fills.arb_d.abs() + fills.buy_d.abs() + fills.sell_d.abs(),
        );
    }
}

/// One step's fill outcomes: aggregate fees/deltas plus policy-selected per-fill gap
/// accounting and the potential-demand destination ledger.
struct StepFills {
    step_fee: f64,
    arb_d: f64,
    buy_d: f64,
    sell_d: f64,
    step_fee_arb: f64,
    step_fee_fund: f64,
    fund_retention: f64,
    fund_demand_lost: f64,
    arb_active: bool,
    ell_arb: f64,
    ell_buy: f64,
    ell_sell: f64,
    pbar_arb: f64,
    pbar_buy: f64,
    pbar_sell: f64,
    pot_buy: f64,
    pot_sell: f64,
    cex_buy: f64,
    cex_sell: f64,
    unserved_buy: f64,
    unserved_sell: f64,
}

/// Execute one fill, returning (ell, pbar): the per-fill tracking gap
/// delta * (P - pbar) and the exact CPMM average execution price.
fn fill(pool: &mut CampbellPool, delta: f64, cex_price: f64) -> (f64, f64) {
    if delta == 0.0 {
        return (0.0, 0.0);
    }
    let pbar = pool.reserve_x / (pool.reserve_y - delta);
    pool.apply_delta(delta);
    (delta * (cex_price - pbar), pbar)
}

fn make_record(
    step: usize,
    cex_price: f64,
    pool: &CampbellPool,
    hedging: f64,
    fee: f64,
    obs: &FeeObservation,
    fills: &StepFills,
) -> StepRecord {
    StepRecord {
        step,
        cex_price,
        amm_price: pool.marginal_price(),
        arb_delta: fills.arb_d,
        buy_delta: fills.buy_d,
        sell_delta: fills.sell_d,
        step_fee: fills.step_fee,
        pool_value: pool.pool_value(cex_price),
        hedging_portfolio: hedging,
        pool_x: pool.reserve_x,
        pool_y: pool.reserve_y,
        fee_used: fee,
        oracle_gap_bps: obs.oracle_gap_bps,
        inventory_skew: obs.inventory_skew,
        step_fee_arb: fills.step_fee_arb,
        step_fee_fund: fills.step_fee_fund,
        fund_retention: fills.fund_retention,
        fund_demand_lost: fills.fund_demand_lost,
        arb_active: fills.arb_active,
        ell_arb: fills.ell_arb,
        ell_buy: fills.ell_buy,
        ell_sell: fills.ell_sell,
        pbar_arb: fills.pbar_arb,
        pbar_buy: fills.pbar_buy,
        pbar_sell: fills.pbar_sell,
        pot_buy: fills.pot_buy,
        pot_sell: fills.pot_sell,
        cex_buy: fills.cex_buy,
        cex_sell: fills.cex_sell,
        unserved_buy: fills.unserved_buy,
        unserved_sell: fills.unserved_sell,
        log_gap_abs: (pool.marginal_price() / cex_price).ln().abs(),
    }
}

/// Fixed-capacity rolling windows for the policy observation, with O(1)
/// incremental statistics (round 25). The previous per-step recompute
/// over the whole deque was O(n_steps * window); at fine clocks the
/// window is tens of thousands of steps (72k at 1 s), which dominated
/// runtime. These fields feed only the tabular RL policy's buckets;
/// fixed/gap/defensive policies ignore them, so incremental f64 drift
/// (negligible for log-return magnitudes) never affects results here.
struct RollingWindows {
    cap: usize,
    rets: VecDeque<f64>,
    ret_sum: f64,
    ret_sq: f64,
    arb: VecDeque<bool>,
    arb_true: usize,
    fund: VecDeque<bool>,
    fund_true: usize,
    vols: VecDeque<f64>,
    vol_sum: f64,
}

impl RollingWindows {
    fn new(cap: usize) -> Self {
        RollingWindows {
            cap,
            rets: VecDeque::with_capacity(cap + 1),
            ret_sum: 0.0,
            ret_sq: 0.0,
            arb: VecDeque::with_capacity(cap + 1),
            arb_true: 0,
            fund: VecDeque::with_capacity(cap + 1),
            fund_true: 0,
            vols: VecDeque::with_capacity(cap + 1),
            vol_sum: 0.0,
        }
    }
    fn push(&mut self, ret: f64, arb_hit: bool, fund_hit: bool, vol: f64) {
        if self.rets.len() == self.cap {
            let old_ret = self.rets.pop_front().unwrap();
            self.ret_sum -= old_ret;
            self.ret_sq -= old_ret * old_ret;
            let old = self.arb.pop_front().unwrap();
            if old {
                self.arb_true -= 1;
            }
            let old = self.fund.pop_front().unwrap();
            if old {
                self.fund_true -= 1;
            }
            self.vol_sum -= self.vols.pop_front().unwrap();
        }
        self.rets.push_back(ret);
        self.ret_sum += ret;
        self.ret_sq += ret * ret;
        self.arb.push_back(arb_hit);
        if arb_hit {
            self.arb_true += 1;
        }
        self.fund.push_back(fund_hit);
        if fund_hit {
            self.fund_true += 1;
        }
        self.vols.push_back(vol);
        self.vol_sum += vol;
    }
    fn std(&self) -> f64 {
        let n = self.rets.len();
        if n < 2 {
            return 0.0;
        }
        let mean = self.ret_sum / n as f64;
        let var = (self.ret_sq / n as f64 - mean * mean).max(0.0);
        var.sqrt()
    }
    fn arb_frac(&self) -> f64 {
        if self.arb.is_empty() {
            0.0
        } else {
            self.arb_true as f64 / self.arb.len() as f64
        }
    }
    fn fund_frac(&self) -> f64 {
        if self.fund.is_empty() {
            0.0
        } else {
            self.fund_true as f64 / self.fund.len() as f64
        }
    }
    fn volume(&self) -> f64 {
        self.vol_sum
    }
}

#[allow(clippy::too_many_arguments)]
fn build_observation(
    step: usize,
    cex_price: f64,
    pool: &CampbellPool,
    win: &RollingWindows,
    previous_fee: f64,
) -> FeeObservation {
    let oracle_gap_bps = (pool.marginal_price() - cex_price) / cex_price * 10_000.0;
    let inventory_skew = (pool.reserve_x - pool.reserve_y * cex_price)
        / (pool.reserve_x + pool.reserve_y * cex_price);
    FeeObservation {
        step,
        external_price: cex_price,
        amm_price: pool.marginal_price(),
        oracle_gap_bps,
        contemporaneous_gap_bps: oracle_gap_bps,
        inventory_skew,
        recent_vol: win.std(),
        recent_arb_frac: win.arb_frac(),
        recent_fund_frac: win.fund_frac(),
        recent_volume: win.volume(),
        previous_fee,
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_trades(
    config: &SimConfig,
    step: usize,
    cex_price: f64,
    fee: f64,
    pool: &mut CampbellPool,
    regime_rng: &mut StdRng,
    latency_rng: &mut StdRng,
    fee_before: f64,
    arrivals: (bool, bool),
) -> StepFills {
    pool.amm_fee = fee;

    let fund_scale = effective_fund_scale(config, step, regime_rng);
    let fund_retention = (1.0
        - config.e1_lambda * ((fee - config.e1_fee_ref).max(0.0) / config.e1_fee_ref))
        .clamp(0.0, 1.0);
    // physical-clock arrival thinning: no arrival => zero potential demand on that
    // side (the destination ledger records a pot_* = 0 row).
    let eff_buy = if arrivals.0 {
        config.buy_demand * fund_scale
    } else {
        0.0
    };
    let eff_sell = if arrivals.1 {
        config.sell_demand * fund_scale
    } else {
        0.0
    };

    let p_arb = match config.arb_arrival_rate_per_hour {
        Some(lam) => 1.0 - (-lam * config.dt_hours).exp(),
        None => config.e5_arb_prob,
    };
    let arb_active = latency_rng.gen_range(0.0f64..1.0) < p_arb;
    let arb_d = if arb_active {
        arb_delta(pool, cex_price, config.cex_fee)
    } else {
        0.0
    };
    let (ell_arb, pbar_arb) = fill(pool, arb_d, cex_price);
    let fee_after_arb = pool.cumulative_fee_revenue;

    // Destination ledger: potential demand = AMM fill + CEX-routed
    // (beyond the marginal-routing band capacity) + unserved
    // (router-substitution loss, e1).
    let buy_full = fundamental_buy_delta(eff_buy, pool, cex_price, config.cex_fee);
    let buy_d = buy_full * fund_retention;
    let (ell_buy, pbar_buy) = fill(pool, buy_d, cex_price);
    let sell_full = fundamental_sell_delta(-eff_sell, pool, cex_price, config.cex_fee);
    let sell_d = sell_full * fund_retention;
    let (ell_sell, pbar_sell) = fill(pool, sell_d, cex_price);
    let fund_demand_lost = (buy_full.abs() + sell_full.abs()) * (1.0 - fund_retention);

    StepFills {
        step_fee: pool.cumulative_fee_revenue - fee_before,
        arb_d,
        buy_d,
        sell_d,
        step_fee_arb: fee_after_arb - fee_before,
        step_fee_fund: pool.cumulative_fee_revenue - fee_after_arb,
        fund_retention,
        fund_demand_lost,
        arb_active,
        ell_arb,
        ell_buy,
        ell_sell,
        pbar_arb,
        pbar_buy,
        pbar_sell,
        pot_buy: eff_buy,
        pot_sell: eff_sell,
        cex_buy: eff_buy - buy_full.abs(),
        cex_sell: eff_sell - sell_full.abs(),
        unserved_buy: buy_full.abs() * (1.0 - fund_retention),
        unserved_sell: sell_full.abs() * (1.0 - fund_retention),
    }
}

/// Emit the primitive event ledger for one BERNOULLI-mode step: up to
/// arb + fund-buy + fund-sell events, with fixed intra-step order keys
/// (0, 1/3, 2/3). Fund events are emitted whenever a demand event
/// occurred (pot > 0), so zero-fill demand events stay in the ledger.
fn push_step_events(events: &mut Vec<EventRecord>, step: usize, fills: &StepFills, fee: f64) {
    if fills.arb_d != 0.0 {
        events.push(EventRecord {
            step,
            time_frac: 0.0,
            kind: EventKind::Arb,
            delta: fills.arb_d,
            pbar: fills.pbar_arb,
            ell: fills.ell_arb,
            pot: 0.0,
            cex: 0.0,
            unserved: 0.0,
            fee_used: fee,
        });
    }
    if fills.pot_buy > 0.0 {
        events.push(EventRecord {
            step,
            time_frac: 1.0 / 3.0,
            kind: EventKind::FundBuy,
            delta: fills.buy_d,
            pbar: fills.pbar_buy,
            ell: fills.ell_buy,
            pot: fills.pot_buy,
            cex: fills.cex_buy,
            unserved: fills.unserved_buy,
            fee_used: fee,
        });
    }
    if fills.pot_sell > 0.0 {
        events.push(EventRecord {
            step,
            time_frac: 2.0 / 3.0,
            kind: EventKind::FundSell,
            delta: fills.sell_d,
            pbar: fills.pbar_sell,
            ell: fills.ell_sell,
            pot: fills.pot_sell,
            cex: fills.cex_sell,
            unserved: fills.unserved_sell,
            fee_used: fee,
        });
    }
}

/// One PRIMITIVE fundamental demand event (round 21, Poisson mode):
/// single-side fill at the CONTEMPORANEOUS external price with the fee
/// chosen for this event; e1 retention is fee-dependent per event.
fn fund_event(
    config: &SimConfig,
    pool: &mut CampbellPool,
    cex_price: f64,
    fee: f64,
    is_buy: bool,
    fund_scale: f64,
) -> (f64, f64, f64, f64, f64, f64) {
    pool.amm_fee = fee;
    let retention = (1.0
        - config.e1_lambda * ((fee - config.e1_fee_ref).max(0.0) / config.e1_fee_ref))
        .clamp(0.0, 1.0);
    let pot = if is_buy {
        config.buy_demand
    } else {
        config.sell_demand
    } * fund_scale;
    let full = if is_buy {
        fundamental_buy_delta(pot, pool, cex_price, config.cex_fee)
    } else {
        fundamental_sell_delta(-pot, pool, cex_price, config.cex_fee)
    };
    let d = full * retention;
    let (ell, pbar) = fill(pool, d, cex_price);
    (
        d,
        ell,
        pbar,
        pot,
        pot - full.abs(),
        full.abs() * (1.0 - retention),
    )
}

/// Poisson-mode step (Eval only): arbitrage at step open, then the
/// scheduled primitive demand events in time order. The hook policy
/// re-evaluates per event using CURRENT pool state and the SAME
/// (possibly lagged) external signal; fills route at the CONTEMPORANEOUS
/// external price. Returns aggregate fills for the step record plus the
/// last fee used.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn poisson_step_eval(
    config: &SimConfig,
    step: usize,
    cex_price: f64,
    obs_used: &FeeObservation,
    policy: &mut dyn FeePolicy,
    pool: &mut CampbellPool,
    latency_rng: &mut StdRng,
    regime_rng: &mut StdRng,
    schedule: &[(f64, bool)],
    fee_before: f64,
    win: &RollingWindows,
    events: &mut Vec<EventRecord>,
    arb_override: Option<bool>,
) -> (StepFills, f64) {
    let signal_price = obs_used.external_price;
    let fund_scale = effective_fund_scale(config, step, regime_rng);

    // fee for the step open / arbitrage leg
    let mut last_fee = policy.fee(obs_used);
    pool.amm_fee = last_fee;
    let p_arb = match config.arb_arrival_rate_per_hour {
        Some(lam) => 1.0 - (-lam * config.dt_hours).exp(),
        None => config.e5_arb_prob,
    };
    let arb_active = match arb_override {
        Some(a) => a,
        None => latency_rng.gen_range(0.0f64..1.0) < p_arb,
    };
    let arb_d = if arb_active {
        arb_delta(pool, cex_price, config.cex_fee)
    } else {
        0.0
    };
    let (ell_arb, pbar_arb) = fill(pool, arb_d, cex_price);
    let fee_after_arb = pool.cumulative_fee_revenue;
    // Log every scheduled arbitrage opportunity when requested; always
    // log fills (delta != 0).
    if config.log_inactive_arb || arb_d != 0.0 {
        events.push(EventRecord {
            step,
            time_frac: 0.0,
            kind: EventKind::Arb,
            delta: arb_d,
            pbar: pbar_arb,
            ell: ell_arb,
            pot: 0.0,
            cex: 0.0,
            unserved: 0.0,
            fee_used: last_fee,
        });
    }

    let (mut buy_d, mut sell_d) = (0.0f64, 0.0f64);
    let (mut ell_buy, mut ell_sell) = (0.0f64, 0.0f64);
    let (mut pot_buy, mut pot_sell) = (0.0f64, 0.0f64);
    let (mut cex_buy, mut cex_sell) = (0.0f64, 0.0f64);
    let (mut uns_buy, mut uns_sell) = (0.0f64, 0.0f64);
    let (mut n_buy, mut n_sell) = (0u32, 0u32);
    let (mut last_pbar_buy, mut last_pbar_sell) = (0.0f64, 0.0f64);
    for &(u, is_buy) in schedule {
        // per-event observation: CURRENT pool state, SAME stale signal.
        let mut obs_e = build_observation(step, signal_price, pool, win, last_fee);
        obs_e.contemporaneous_gap_bps = (pool.marginal_price() - cex_price) / cex_price * 10_000.0;
        let fee_e = policy.fee(&obs_e);
        last_fee = fee_e;
        let (d, ell, pbar, pot, cexq, uns) =
            fund_event(config, pool, cex_price, fee_e, is_buy, fund_scale);
        events.push(EventRecord {
            step,
            time_frac: u,
            kind: if is_buy {
                EventKind::FundBuy
            } else {
                EventKind::FundSell
            },
            delta: d,
            pbar,
            ell,
            pot,
            cex: cexq,
            unserved: uns,
            fee_used: fee_e,
        });
        if is_buy {
            buy_d += d;
            ell_buy += ell;
            pot_buy += pot;
            cex_buy += cexq;
            uns_buy += uns;
            n_buy += 1;
            last_pbar_buy = pbar;
        } else {
            sell_d += d;
            ell_sell += ell;
            pot_sell += pot;
            cex_sell += cexq;
            uns_sell += uns;
            n_sell += 1;
            last_pbar_sell = pbar;
        }
    }

    let fills = StepFills {
        step_fee: pool.cumulative_fee_revenue - fee_before,
        arb_d,
        buy_d,
        sell_d,
        step_fee_arb: fee_after_arb - fee_before,
        step_fee_fund: pool.cumulative_fee_revenue - fee_after_arb,
        fund_retention: f64::NAN, // fee-dependent per event; see ledger
        fund_demand_lost: uns_buy + uns_sell,
        arb_active,
        ell_arb,
        ell_buy,
        ell_sell,
        pbar_arb,
        // aggregates: meaningful only when a single event hit the side;
        // fill-level analysis must use the event ledger.
        pbar_buy: if n_buy == 1 { last_pbar_buy } else { 0.0 },
        pbar_sell: if n_sell == 1 { last_pbar_sell } else { 0.0 },
        pot_buy,
        pot_sell,
        cex_buy,
        cex_sell,
        unserved_buy: uns_buy,
        unserved_sell: uns_sell,
    };
    (fills, last_fee)
}

// ── Private helpers ───────────────────────────────────────────────────────────

fn effective_fund_scale(config: &SimConfig, step: usize, rng: &mut StdRng) -> f64 {
    match &config.flow_regime {
        FlowRegime::Normal => 1.0,
        FlowRegime::ToxicBurst => {
            if rng.gen_range(0.0f64..1.0) < config.toxic_burst_prob {
                config.toxic_burst_fund_scale
            } else {
                1.0
            }
        }
        FlowRegime::RegimeSwitch => {
            let period = config.regime_switch_period.max(1);
            if (step / period).is_multiple_of(2) {
                1.0
            } else {
                config.toxic_burst_fund_scale
            }
        }
    }
}

// ── metric-identity metric identity audit tests (.local/lvr) ──────────────────────────────
//
// These tests pin the definition of the realized tracking error
// L_T = hedging_portfolio_T - pool_value_T as computed by this engine:
// self-financing benchmark marked every step with pre-step inventory,
// fees outside reserves, trades executed at the exact CPMM average price.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::campbell::fee_policy::FixedFeePolicy;
    use crate::campbell::gbm::generate_gbm;
    use crate::campbell::simulation::{
        run_simulation_with_events, run_simulation_with_events_compact,
    };
    use crate::campbell::summary::summarize_events;

    fn base_config(amm_fee: f64, cex_fee: f64, buy: f64, sell: f64, arb_prob: f64) -> SimConfig {
        SimConfig {
            name: "audit".into(),
            description: "m0 metric identity audit".into(),
            amm_fee,
            cex_fee,
            buy_demand: buy,
            sell_demand: sell,
            reserve_x: 2.0e7,
            reserve_y: 1.0e4, // marginal price 2000 = s0
            sigma: 0.4,
            mu: 0.0,
            n_steps: 800,
            seed: 42,
            flow_regime: FlowRegime::Normal,
            toxic_burst_prob: 0.0,
            toxic_burst_arb_scale: 1.0,
            toxic_burst_fund_scale: 1.0,
            regime_switch_period: 0,
            e1_lambda: 0.0,
            e1_fee_ref: 0.0006,
            e5_arb_prob: arb_prob,
            policy_lag: 0,
            dt_hours: 1.0,
            pooled_fund_arrival_rate_per_hour: None,
            buy_arrival_share: 0.5,
            arb_arrival_rate_per_hour: None,
            lookback_hours: 20.0,
            arrival_model: ArrivalModel::Bernoulli,
            log_inactive_arb: false,
        }
    }

    fn run(config: &SimConfig) -> Vec<StepRecord> {
        let prices = generate_gbm(
            config.n_steps,
            2000.0,
            config.mu,
            config.sigma,
            1.0 / (365.0 * 24.0),
            config.seed,
        );
        let mut policy = FixedFeePolicy::new(config.amm_fee);
        run_simulation(config, &prices, &mut policy)
    }

    fn tracking_error(records: &[StepRecord]) -> f64 {
        let last = records.last().unwrap();
        last.hedging_portfolio - last.pool_value
    }

    /// No-trade => L_T is exactly the telescoped mark, i.e. zero.
    /// This is the capped-state exact-shutdown fact (finite path, finite fee),
    /// NOT a claim about GBM's unbounded support.
    #[test]
    fn shutdown_no_trades_zero_tracking_error() {
        let config = base_config(0.5, 0.0005, 5.0, 5.0, 1.0);
        let records = run(&config);
        let volume: f64 = records
            .iter()
            .map(|r| r.arb_delta.abs() + r.buy_delta.abs() + r.sell_delta.abs())
            .sum();
        assert_eq!(volume, 0.0, "50% fee must shut every fill on this path");
        let v0 = 4.0e7;
        assert!(
            tracking_error(&records).abs() < 1e-6 * v0,
            "no-trade tracking error must vanish, got {}",
            tracking_error(&records)
        );
    }

    /// L_T telescopes over executed fills: replaying the recorded deltas and
    /// summing per-fill gaps ell_k = delta * (P - avg_exec_price) reproduces
    /// hedging - pool_value. Also pins the Jordan decomposition
    /// L_T = A_T - B_T (gross unfavorable minus favorable components) and
    /// the per-fill nonnegativity of the ARB leg (every arb fill executes
    /// outside the fee band on the adverse side, so its ell_k >= 0).
    ///
    /// Config uses amm_fee < cex_fee: favorable ex-fee fundamental fills
    /// (B_T > 0) exist only in that regime, because a fundamental buy fills
    /// while p_bar(1+f) <= P(1+c), so p_bar can exceed the external mid only
    /// when f < c. With f > c every fundamental fill is ex-fee adverse and
    /// B_T = 0 (verified: the original f=30bps/c=10bps config fails the
    /// B_T > 0 assertion).
    #[test]
    fn tracking_error_telescopes_over_fills() {
        let config = base_config(0.0005, 0.003, 5.0, 5.0, 1.0);
        let records = run(&config);
        let mut pool = CampbellPool::new(config.reserve_x, config.reserve_y, 0.0);
        let mut loss = 0.0;
        let mut gross_a = 0.0; // A_T: sum of max(ell_k, 0)
        let mut gross_b = 0.0; // B_T: sum of max(-ell_k, 0)
        for r in &records {
            for (leg, delta) in [
                ("arb", r.arb_delta),
                ("buy", r.buy_delta),
                ("sell", r.sell_delta),
            ] {
                if delta != 0.0 {
                    let avg_price = pool.reserve_x / (pool.reserve_y - delta);
                    let ell = delta * (r.cex_price - avg_price);
                    if leg == "arb" {
                        assert!(
                            ell >= 0.0,
                            "arb fill must be per-fill adverse, got ell = {ell} at step {}",
                            r.step
                        );
                    }
                    loss += ell;
                    gross_a += ell.max(0.0);
                    gross_b += (-ell).max(0.0);
                    pool.apply_delta(delta);
                }
            }
            assert!(
                (pool.reserve_y - r.pool_y).abs() < 1e-9 * r.pool_y.abs().max(1.0),
                "replayed reserves diverged at step {}",
                r.step
            );
        }
        let te = tracking_error(&records);
        assert!(
            (loss - te).abs() < 1e-6 * te.abs().max(1.0),
            "telescoped {loss} vs engine {te}"
        );
        assert!(te != 0.0, "test must exercise actual fills");
        assert!(
            ((gross_a - gross_b) - loss).abs() < 1e-9 * gross_a.max(1.0),
            "Jordan decomposition must be exact: A {gross_a} - B {gross_b} vs L {loss}"
        );
        assert!(
            gross_b > 0.0,
            "mixed flow must produce favorable in-band fills (B_T > 0)"
        );
    }

    /// Arbitrage-only flow: every fill executes strictly outside the fee band,
    /// so each per-trade term is a loss and L_T > 0.
    #[test]
    fn arb_only_tracking_error_positive() {
        let config = base_config(0.0005, 0.0005, 0.0, 0.0, 1.0);
        let records = run(&config);
        let volume: f64 = records.iter().map(|r| r.arb_delta.abs()).sum();
        assert!(volume > 0.0, "path must trigger arbitrage");
        assert!(
            tracking_error(&records) > 0.0,
            "arb-only tracking error must be positive, got {}",
            tracking_error(&records)
        );
    }

    /// policy-selected: the engine's per-fill gap records (ell_arb/ell_buy/ell_sell)
    /// must sum exactly to the tracking error, independently of the replay
    /// construction used in `tracking_error_telescopes_over_fills`.
    #[test]
    fn engine_ell_records_sum_to_tracking_error() {
        let config = base_config(0.0005, 0.003, 5.0, 5.0, 1.0);
        let records = run(&config);
        let ell_sum: f64 = records
            .iter()
            .map(|r| r.ell_arb + r.ell_buy + r.ell_sell)
            .sum();
        let te = tracking_error(&records);
        assert!(
            (ell_sum - te).abs() < 1e-6 * te.abs().max(1.0),
            "engine ell sum {ell_sum} vs tracking error {te}"
        );
    }

    /// policy-selected destination ledger conservation: potential demand = AMM fill +
    /// CEX-routed + unserved, per side, every step (risky units).
    #[test]
    fn destination_ledger_conserves_potential_demand() {
        let mut config = base_config(0.003, 0.001, 5.0, 5.0, 1.0);
        config.e1_lambda = 0.5; // exercise the unserved leg too
        let records = run(&config);
        for r in &records {
            let buy_amm = r.buy_delta.abs();
            let sell_amm = r.sell_delta.abs();
            assert!(
                (r.pot_buy - (buy_amm + r.cex_buy + r.unserved_buy)).abs() < 1e-9,
                "buy ledger leak at step {}",
                r.step
            );
            assert!(
                (r.pot_sell - (sell_amm + r.cex_sell + r.unserved_sell)).abs() < 1e-9,
                "sell ledger leak at step {}",
                r.step
            );
        }
        let unserved: f64 = records.iter().map(|r| r.unserved_buy).sum();
        assert!(unserved > 0.0, "e1_lambda must produce unserved volume");
    }

    /// policy-selected one-step-lag mode: with a constant-fee policy the lag cannot
    /// matter (the decision ignores the observation), so records must be
    /// identical; this pins that lag only reroutes the information set.
    #[test]
    fn lag_is_noop_for_constant_fee() {
        let config0 = base_config(0.003, 0.001, 5.0, 5.0, 1.0);
        let mut config1 = base_config(0.003, 0.001, 5.0, 5.0, 1.0);
        config1.policy_lag = 1;
        let (r0, r1) = (run(&config0), run(&config1));
        assert_eq!(r0.len(), r1.len());
        for (a, b) in r0.iter().zip(r1.iter()) {
            assert_eq!(a.pool_x, b.pool_x, "diverged at step {}", a.step);
            assert_eq!(a.step_fee, b.step_fee, "diverged at step {}", a.step);
        }
    }

    /// policy-selected one-step-lag mode: a state-dependent policy must actually see
    /// stale information under lag=1 — fee decisions differ from zero-lag.
    #[test]
    fn lag_changes_state_dependent_policy_decisions() {
        use crate::campbell::fee_policy::OracleGapFeePolicy;
        let config0 = base_config(0.003, 0.001, 5.0, 5.0, 1.0);
        let mut config1 = base_config(0.003, 0.001, 5.0, 5.0, 1.0);
        config1.policy_lag = 1;
        let prices = generate_gbm(
            config0.n_steps,
            2000.0,
            0.0,
            config0.sigma,
            1.0 / (365.0 * 24.0),
            config0.seed,
        );
        let run_gap = |cfg: &SimConfig| {
            let mut policy = OracleGapFeePolicy {
                base_fee: 0.0005,
                gap_multiplier: 0.5,
                min_fee: 0.0001,
                max_fee: 0.01,
            };
            run_simulation(cfg, &prices, &mut policy)
        };
        let (r0, r1) = (run_gap(&config0), run_gap(&config1));
        let differs = r0
            .iter()
            .zip(r1.iter())
            .any(|(a, b)| (a.fee_used - b.fee_used).abs() > 1e-15);
        assert!(differs, "lag=1 must change a gap-reactive policy's fees");
        // First-step pin: lagged decision uses the genuine t_0 observation
        // (initial pool aligned with P_0, gap = 0 => base fee), while the
        // zero-lag decision sees the nonzero gap opened by the first price
        // move. Guards against an undisclosed zero-lag first step.
        assert!(
            (r1[0].fee_used - 0.0005).abs() < 1e-15,
            "lagged first step must use t_0 gap (base fee), got {}",
            r1[0].fee_used
        );
        assert!(
            (r0[0].fee_used - r1[0].fee_used).abs() > 1e-15,
            "zero-lag and lagged first-step fees must differ on this path"
        );
    }

    #[test]
    fn lag_diagnostic_separates_signal_and_contemporaneous_gap() {
        struct ProbePolicy {
            seen: Vec<(f64, f64)>,
        }
        impl FeePolicy for ProbePolicy {
            fn name(&self) -> &'static str {
                "probe"
            }
            fn fee(&mut self, obs: &FeeObservation) -> f64 {
                self.seen
                    .push((obs.oracle_gap_bps, obs.contemporaneous_gap_bps));
                0.0005 + 0.1 * obs.oracle_gap_bps.abs() / 10_000.0
            }
        }

        let mut config = base_config(0.003, 0.001, 5.0, 5.0, 1.0);
        config.policy_lag = 1;
        let prices = generate_gbm(
            config.n_steps,
            2000.0,
            0.0,
            config.sigma,
            config.dt_years(),
            config.seed,
        );
        let mut policy = ProbePolicy { seen: Vec::new() };
        let records = run_simulation(&config, &prices, &mut policy);
        assert_eq!(policy.seen.len(), records.len());
        assert!(
            policy
                .seen
                .iter()
                .any(|(signal, current)| { (signal - current).abs() > 1e-9 })
        );
        for ((signal, _), record) in policy.seen.iter().zip(records) {
            let expected = 0.0005 + 0.1 * signal.abs() / 10_000.0;
            assert!((record.fee_used - expected).abs() < 1e-15);
        }
    }

    /// physical-clock arrival thinning: the arrival sequence is policy-invariant
    /// given the seed (pot_* patterns identical across different fee
    /// policies on the same path), the ledger conserves on zero-potential
    /// rows, and the realized arrival frequency is near p = 1 - exp(-l*dt).
    #[test]
    fn arrival_thinning_policy_invariant_and_conserving() {
        let mut cfg = base_config(0.003, 0.001, 5.0, 5.0, 1.0);
        cfg.dt_hours = 1.0 / 12.0; // 5-minute clock
        cfg.pooled_fund_arrival_rate_per_hour = Some(4.1); // pooled; per-side lam = 2.05
        let prices = generate_gbm(
            cfg.n_steps,
            2000.0,
            0.0,
            cfg.sigma,
            1.0 / (365.0 * 24.0),
            cfg.seed,
        );
        let mut low = FixedFeePolicy::new(0.0005);
        let mut high = FixedFeePolicy::new(0.05);
        let r_low = run_simulation(&cfg, &prices, &mut low);
        let r_high = run_simulation(&cfg, &prices, &mut high);
        for (a, b) in r_low.iter().zip(r_high.iter()) {
            assert_eq!(a.pot_buy, b.pot_buy, "arrivals must be policy-invariant");
            assert_eq!(a.pot_sell, b.pot_sell, "arrivals must be policy-invariant");
            assert!(
                (a.pot_buy - (a.buy_delta.abs() + a.cex_buy + a.unserved_buy)).abs() < 1e-9,
                "ledger leak on thinned row at step {}",
                a.step
            );
        }
        // Per-side rate = share * pooled = 0.5 * 4.1; p per 5-min step.
        let p_expect = 1.0 - (-0.5 * 4.1f64 / 12.0).exp(); // ~0.157
        let arrived = r_low.iter().filter(|r| r.pot_buy > 0.0).count() as f64 / r_low.len() as f64;
        assert!(
            (arrived - p_expect).abs() < 0.06,
            "arrival frequency {arrived} far from p = {p_expect}"
        );
        assert!(
            arrived > 0.0 && arrived < 1.0,
            "thinning must actually thin"
        );
    }

    /// Round 13 physical-clock helpers: dt_years and window_steps math,
    /// and the physical arb-rate override (rate 0 => arb never activates;
    /// huge rate => arb active whenever profitable band exists).
    #[test]
    fn physical_clock_helpers_and_arb_rate_override() {
        let mut cfg = base_config(0.0005, 0.0005, 0.0, 0.0, 0.0); // e5 prob 0
        cfg.dt_hours = 1.0 / 12.0;
        cfg.lookback_hours = 20.0;
        assert!((cfg.dt_years() - (1.0 / 12.0) / (24.0 * 365.0)).abs() < 1e-18);
        assert_eq!(cfg.window_steps(), 240, "20h lookback on 5-min clock");
        // Physical arb rate overrides the (zero) e5 probability.
        cfg.arb_arrival_rate_per_hour = Some(1.0e6);
        let records = run(&cfg);
        let arb_volume: f64 = records.iter().map(|r| r.arb_delta.abs()).sum();
        assert!(arb_volume > 0.0, "huge arb rate must activate arbitrage");
        cfg.arb_arrival_rate_per_hour = Some(0.0);
        let records = run(&cfg);
        let arb_volume: f64 = records.iter().map(|r| r.arb_delta.abs()).sum();
        assert_eq!(arb_volume, 0.0, "zero arb rate must disable arbitrage");
    }

    /// physical-clock arrival rate semantics: Some(0.0) means demand NEVER arrives
    /// (distinct from None = legacy always-on).
    #[test]
    fn arrival_rate_zero_means_no_demand() {
        let mut cfg = base_config(0.003, 0.001, 5.0, 5.0, 1.0);
        cfg.pooled_fund_arrival_rate_per_hour = Some(0.0);
        let records = run(&cfg);
        assert!(
            records
                .iter()
                .all(|r| r.pot_buy == 0.0 && r.pot_sell == 0.0)
        );
        let served: f64 = records
            .iter()
            .map(|r| r.buy_delta.abs() + r.sell_delta.abs())
            .sum();
        assert_eq!(served, 0.0);
    }

    /// Poisson-arrival gate 1: the Poisson primitive-event schedule is
    /// policy-invariant — two different fee policies on the same seed see
    /// identical (step, time_frac, kind, pot) fundamental-event sequences.
    #[test]
    fn poisson_schedule_policy_invariant() {
        let mut cfg = base_config(0.003, 0.001, 5.0, 5.0, 1.0);
        cfg.arrival_model = ArrivalModel::Poisson;
        cfg.dt_hours = 1.0 / 12.0;
        cfg.pooled_fund_arrival_rate_per_hour = Some(24.0); // mean 1/side/step
        let prices = generate_gbm(
            cfg.n_steps,
            2000.0,
            0.0,
            cfg.sigma,
            cfg.dt_years(),
            cfg.seed,
        );
        let mut low = FixedFeePolicy::new(0.0005);
        let mut high = FixedFeePolicy::new(0.05);
        let (_, e1) = run_simulation_with_events(&cfg, &prices, &mut low);
        let (_, e2) = run_simulation_with_events(&cfg, &prices, &mut high);
        let sched = |ev: &[EventRecord]| {
            ev.iter()
                .filter(|e| e.kind != EventKind::Arb)
                .map(|e| (e.step, e.time_frac.to_bits(), e.kind, e.pot.to_bits()))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            sched(&e1),
            sched(&e2),
            "fund-event schedule must be policy-invariant"
        );
        assert!(!sched(&e1).is_empty());
    }

    #[test]
    fn compact_event_run_preserves_event_summary() {
        let mut cfg = base_config(0.0005, 0.001, 5.0, 5.0, 1.0);
        cfg.arrival_model = ArrivalModel::Poisson;
        cfg.dt_hours = 1.0 / 12.0;
        cfg.pooled_fund_arrival_rate_per_hour = Some(24.0);
        let prices = generate_gbm(
            cfg.n_steps,
            2000.0,
            0.0,
            cfg.sigma,
            cfg.dt_years(),
            cfg.seed,
        );
        let mut full_policy = FixedFeePolicy::new(cfg.amm_fee);
        let mut compact_policy = FixedFeePolicy::new(cfg.amm_fee);
        let (full_records, full_events) =
            run_simulation_with_events(&cfg, &prices, &mut full_policy);
        let (compact_records, compact_events) =
            run_simulation_with_events_compact(&cfg, &prices, &mut compact_policy);
        let full = summarize_events(&full_events, &full_records);
        let compact = summarize_events(&compact_events, &compact_records);

        assert!(compact_records.len() < full_records.len());
        assert_eq!(full_events.len(), compact_events.len());
        assert_eq!(full.l_total.to_bits(), compact.l_total.to_bits());
        assert_eq!(full.a_fill.to_bits(), compact.a_fill.to_bits());
        assert_eq!(full.b_fill.to_bits(), compact.b_fill.to_bits());
        assert_eq!(full.fees_total.to_bits(), compact.fees_total.to_bits());
        assert_eq!(full.u_lp_rel.to_bits(), compact.u_lp_rel.to_bits());
        assert_eq!(
            full.served_fund_volume.to_bits(),
            compact.served_fund_volume.to_bits()
        );
        assert_eq!(
            full.potential_volume.to_bits(),
            compact.potential_volume.to_bits()
        );
        assert_eq!(
            full.tracking_error.to_bits(),
            compact.tracking_error.to_bits()
        );
    }

    /// Poisson-arrival gate 2+3: per-primitive-event destination conservation and the
    /// A/B/L identity on the EVENT ledger (sum of per-event ell equals the
    /// step-based tracking error), with multi-event steps present.
    #[test]
    fn poisson_event_ledger_identities() {
        let mut cfg = base_config(0.0005, 0.003, 5.0, 5.0, 1.0);
        cfg.arrival_model = ArrivalModel::Poisson;
        cfg.dt_hours = 1.0 / 12.0;
        cfg.pooled_fund_arrival_rate_per_hour = Some(60.0); // mean 2.5/side/step
        let prices = generate_gbm(
            cfg.n_steps,
            2000.0,
            0.0,
            cfg.sigma,
            cfg.dt_years(),
            cfg.seed,
        );
        let mut policy = FixedFeePolicy::new(cfg.amm_fee);
        let (records, events) = run_simulation_with_events(&cfg, &prices, &mut policy);
        let mut per_step = std::collections::HashMap::new();
        let mut ell_sum = 0.0;
        for e in &events {
            ell_sum += e.ell;
            if e.kind != EventKind::Arb {
                assert!(
                    (e.pot - (e.delta.abs() + e.cex + e.unserved)).abs() < 1e-9,
                    "per-event ledger leak at step {}",
                    e.step
                );
                *per_step.entry(e.step).or_insert(0u32) += 1;
            }
        }
        let te = tracking_error(&records);
        assert!(
            (ell_sum - te).abs() < 1e-6 * te.abs().max(1.0),
            "event-ledger ell sum {ell_sum} vs tracking error {te}"
        );
        assert!(
            per_step.values().any(|&n| n > 2),
            "rate 60/hr must produce multi-event steps"
        );
    }

    /// Poisson-arrival gate 4a: Poisson with zero hazard equals Bernoulli with zero
    /// hazard exactly (arb-only records identical; arrival RNG consumption
    /// does not touch the arb path).
    #[test]
    fn poisson_zero_rate_equals_bernoulli_zero_rate() {
        let mut a = base_config(0.0005, 0.0005, 5.0, 5.0, 1.0);
        a.pooled_fund_arrival_rate_per_hour = Some(0.0);
        let mut b = a.clone();
        b.arrival_model = ArrivalModel::Poisson;
        let (ra, rb) = (run(&a), run(&b));
        assert_eq!(ra.len(), rb.len());
        for (x, y) in ra.iter().zip(rb.iter()) {
            assert_eq!(x.pool_x, y.pool_x, "diverged at step {}", x.step);
            assert_eq!(x.arb_delta, y.arb_delta);
        }
    }

    /// Poisson-arrival gate 4b: low-rate convergence — realized fund-event frequency
    /// approaches lambda*dt and identities still hold.
    #[test]
    fn poisson_low_rate_convergence() {
        let mut cfg = base_config(0.003, 0.001, 5.0, 5.0, 1.0);
        cfg.arrival_model = ArrivalModel::Poisson;
        cfg.dt_hours = 1.0 / 12.0;
        cfg.n_steps = 4000;
        cfg.pooled_fund_arrival_rate_per_hour = Some(1.2); // mean 0.05/side/step
        let prices = generate_gbm(
            cfg.n_steps,
            2000.0,
            0.0,
            cfg.sigma,
            cfg.dt_years(),
            cfg.seed,
        );
        let mut policy = FixedFeePolicy::new(cfg.amm_fee);
        let (records, events) = run_simulation_with_events(&cfg, &prices, &mut policy);
        let n_fund = events.iter().filter(|e| e.kind != EventKind::Arb).count() as f64;
        let expect = 1.2 / 12.0 * cfg.n_steps as f64; // pooled both sides
        assert!(
            (n_fund - expect).abs() < 0.25 * expect,
            "fund events {n_fund} far from expected {expect}"
        );
        let ell_sum: f64 = events.iter().map(|e| e.ell).sum();
        let te = tracking_error(&records);
        assert!((ell_sum - te).abs() < 1e-6 * te.abs().max(1.0));
    }

    /// Poisson-arrival/round-26 gate: two-moment calibration hits BOTH the total
    /// activity target and the arb-fill target (or reports the arb
    /// ceiling as unreachable), solving for latent lambda_arb*/fund*.
    #[test]
    fn two_moment_calibration_hits_both_targets() {
        use crate::campbell::calibrate::calibrate_two_moment;
        let mut cfg = base_config(0.003, 0.001, 5.0, 5.0, 1.0);
        cfg.arrival_model = ArrivalModel::Poisson;
        cfg.dt_hours = 1.0 / 12.0;
        cfg.n_steps = 2016; // one week at 5-min steps
        cfg.sigma = 0.6;
        cfg.pooled_fund_arrival_rate_per_hour = Some(1.0);
        cfg.arb_arrival_rate_per_hour = Some(1.0);
        let total_target = 600.0; // swaps/week
        let arb_target_hr = 0.5; // 84/week — small, reachable
        let r = calibrate_two_moment(&cfg, 2000.0, 100..103, total_target, arb_target_hr, 0.10);
        assert!(r.lambda_fund > 0.0 && r.lambda_arb > 0.0);
        assert!(
            (r.total_achieved - total_target).abs() <= 0.12 * total_target,
            "total {} vs target {total_target}",
            r.total_achieved
        );
        if r.arb_reachable {
            assert!(
                (r.arb_achieved - arb_target_hr * 168.0).abs() <= 0.20 * arb_target_hr * 168.0,
                "arb {} vs target {}",
                r.arb_achieved,
                arb_target_hr * 168.0
            );
        }
    }

    /// Reachability is measured after re-fitting fundamental flow at the
    /// arb scheduling ceiling.  In particular, an arb target larger than
    /// the total-fill target cannot be mislabeled reachable.
    #[test]
    fn two_moment_calibration_flags_unreachable_arb_target() {
        use crate::campbell::calibrate::calibrate_two_moment;
        let mut cfg = base_config(0.003, 0.001, 5.0, 5.0, 1.0);
        cfg.arrival_model = ArrivalModel::Poisson;
        cfg.dt_hours = 1.0 / 12.0;
        cfg.n_steps = 2016;
        cfg.sigma = 0.6;
        cfg.pooled_fund_arrival_rate_per_hour = Some(1.0);
        cfg.arb_arrival_rate_per_hour = Some(1.0);
        let total_target = 600.0;
        let arb_target_hr = 5.0; // 840/week > 600 total fills/week
        let r = calibrate_two_moment(&cfg, 2000.0, 100..103, total_target, arb_target_hr, 0.10);
        assert!(!r.arb_reachable);
        assert_eq!(r.lambda_arb, 1.0e6);
        assert!(r.arb_achieved < arb_target_hr * 168.0);
        assert!((r.total_achieved - total_target).abs() <= 0.12 * total_target);
    }

    /// Poisson-arrival gate 5: target-moment calibration reaches a tier-style activity
    /// target within tolerance on a small cell.
    #[test]
    fn poisson_hazard_calibration_hits_target() {
        use crate::campbell::calibrate::calibrate_pooled_hazard;
        let mut cfg = base_config(0.003, 0.001, 5.0, 5.0, 1.0);
        cfg.arrival_model = ArrivalModel::Poisson;
        cfg.dt_hours = 1.0 / 12.0;
        cfg.n_steps = 2016; // one week at 5-min steps
        cfg.pooled_fund_arrival_rate_per_hour = Some(1.0);
        let target = 500.0; // swaps/week, small test cell
        let (lam, achieved) = calibrate_pooled_hazard(&cfg, 2000.0, 100..103, target, 0.05);
        assert!(lam > 0.0);
        assert!(
            (achieved - target).abs() <= 0.05 * target,
            "calibrated {achieved}/wk vs target {target}/wk (lambda {lam})"
        );
    }

    /// Nonnegativity COUNTEREXAMPLE: one-sided fundamental flow inside a wide
    /// CEX-fee band, zero AMM fee, arbitrage disabled, near-zero volatility.
    /// Fills execute with the pool buying risky below the external mid while
    /// the quadratic-variation (stale-quote) component is negligible, so
    /// L_T < 0. The engine's tracking error is a SIGNED execution difference,
    /// not a nonnegative loss; an inf E[L_T] = 0 claim needs class-level
    /// nonnegativity, which fails here. (Same config at sigma = 0.4 yields
    /// L_T = +178: the sign is a volatility-vs-band-width race, which is the
    /// audit finding.)
    #[test]
    fn fundamental_in_band_tracking_error_negative() {
        let mut config = base_config(0.0, 0.003, 0.0, 10.0, 0.0);
        config.sigma = 0.01;
        let records = run(&config);
        let volume: f64 = records.iter().map(|r| r.sell_delta.abs()).sum();
        assert!(volume > 0.0, "path must produce fundamental sells");
        assert!(
            tracking_error(&records) < 0.0,
            "in-band one-sided flow must produce negative tracking error, got {}",
            tracking_error(&records)
        );
    }
}
