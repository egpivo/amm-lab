use crate::campbell::fee_policy::{FeeObservation, FeePolicy, TabularLearnedFeePolicy};
use crate::campbell::pool::CampbellPool;
use crate::campbell::trader::{arb_delta, fundamental_buy_delta, fundamental_sell_delta};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
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

const VOL_WINDOWS: usize = 20;

pub fn run_simulation(
    config: &SimConfig,
    cex_prices: &[f64],
    policy: &mut dyn FeePolicy,
) -> Vec<StepRecord> {
    let mut records = Vec::new();
    run_episode(
        config,
        cex_prices,
        EpisodeMode::Eval(policy),
        &mut |_, _, _, _| {},
        &mut records,
    );
    records
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
    run_episode(
        config,
        cex_prices,
        EpisodeMode::Train(policy, gamma),
        &mut |_, reward, _, _| {
            episode_reward += reward;
        },
        &mut records,
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
    mut mode: EpisodeMode<'_>,
    on_step_reward: &mut dyn FnMut(usize, f64, &FeeObservation, &FeeObservation),
    records: &mut Vec<StepRecord>,
) {
    let mut pool = CampbellPool::new(config.reserve_x, config.reserve_y, config.amm_fee);
    let mut hedging = pool.pool_value(cex_prices[0]);
    let mut regime_rng = StdRng::seed_from_u64(config.seed.wrapping_add(99_991));
    let mut latency_rng = StdRng::seed_from_u64(config.seed ^ 77_777);

    let mut log_rets: VecDeque<f64> = VecDeque::with_capacity(VOL_WINDOWS + 1);
    let mut arb_flags: VecDeque<bool> = VecDeque::with_capacity(VOL_WINDOWS + 1);
    let mut fund_flags: VecDeque<bool> = VecDeque::with_capacity(VOL_WINDOWS + 1);
    let mut vol_sums: VecDeque<f64> = VecDeque::with_capacity(VOL_WINDOWS + 1);

    let mut previous_fee = config.amm_fee;
    let mut prev_position = hedging - pool.pool_value(cex_prices[0]);

    for (step, &cex_price) in cex_prices[1..].iter().enumerate() {
        let prev_cex = cex_prices[step];
        hedging += pool.reserve_y * (cex_price - prev_cex);
        let fee_before = pool.cumulative_fee_revenue;

        let obs = build_observation(
            step,
            cex_price,
            &pool,
            &log_rets,
            &arb_flags,
            &fund_flags,
            &vol_sums,
            previous_fee,
        );

        let fee = match &mut mode {
            EpisodeMode::Eval(policy) => policy.fee(&obs),
            EpisodeMode::Train(policy, gamma) => {
                let state = policy.obs_to_state(&obs);
                let action = policy.choose_action(state);
                let fee = crate::campbell::fee_policy::RL_ACTIONS_BPS[action] / 10_000.0;
                pool.amm_fee = fee;
                previous_fee = fee;

                let (
                    step_fee,
                    arb_d,
                    buy_d,
                    sell_d,
                    step_fee_arb,
                    step_fee_fund,
                    fund_retention,
                    fund_demand_lost,
                    arb_active,
                ) = execute_trades(
                    config,
                    step,
                    cex_price,
                    fee,
                    &mut pool,
                    &mut regime_rng,
                    &mut latency_rng,
                    fee_before,
                );

                let cur_position = hedging - pool.pool_value(cex_price);
                let step_reward = step_fee - (cur_position - prev_position);
                prev_position = cur_position;

                push_window(&mut log_rets, (cex_price / prev_cex).ln(), VOL_WINDOWS);
                push_window(&mut arb_flags, arb_d.abs() > 1e-12, VOL_WINDOWS);
                push_window(
                    &mut fund_flags,
                    (buy_d.abs() + sell_d.abs()) > 1e-12,
                    VOL_WINDOWS,
                );
                push_window(
                    &mut vol_sums,
                    arb_d.abs() + buy_d.abs() + sell_d.abs(),
                    VOL_WINDOWS,
                );

                let next_obs = build_observation(
                    step + 1,
                    cex_price,
                    &pool,
                    &log_rets,
                    &arb_flags,
                    &fund_flags,
                    &vol_sums,
                    fee,
                );
                let next_state = policy.obs_to_state(&next_obs);
                policy.update_step(state, action, step_reward, next_state, *gamma);
                on_step_reward(step, step_reward, &obs, &next_obs);

                records.push(StepRecord {
                    step,
                    cex_price,
                    amm_price: pool.marginal_price(),
                    arb_delta: arb_d,
                    buy_delta: buy_d,
                    sell_delta: sell_d,
                    step_fee,
                    pool_value: pool.pool_value(cex_price),
                    hedging_portfolio: hedging,
                    pool_x: pool.reserve_x,
                    pool_y: pool.reserve_y,
                    fee_used: fee,
                    oracle_gap_bps: obs.oracle_gap_bps,
                    inventory_skew: obs.inventory_skew,
                    step_fee_arb,
                    step_fee_fund,
                    fund_retention,
                    fund_demand_lost,
                    arb_active,
                });
                continue;
            }
        };

        pool.amm_fee = fee;
        previous_fee = fee;

        let (
            step_fee,
            arb_d,
            buy_d,
            sell_d,
            step_fee_arb,
            step_fee_fund,
            fund_retention,
            fund_demand_lost,
            arb_active,
        ) = execute_trades(
            config,
            step,
            cex_price,
            fee,
            &mut pool,
            &mut regime_rng,
            &mut latency_rng,
            fee_before,
        );

        let cur_position = hedging - pool.pool_value(cex_price);
        let step_reward = step_fee - (cur_position - prev_position);
        prev_position = cur_position;
        on_step_reward(step, step_reward, &obs, &obs);

        records.push(StepRecord {
            step,
            cex_price,
            amm_price: pool.marginal_price(),
            arb_delta: arb_d,
            buy_delta: buy_d,
            sell_delta: sell_d,
            step_fee,
            pool_value: pool.pool_value(cex_price),
            hedging_portfolio: hedging,
            pool_x: pool.reserve_x,
            pool_y: pool.reserve_y,
            fee_used: fee,
            oracle_gap_bps: obs.oracle_gap_bps,
            inventory_skew: obs.inventory_skew,
            step_fee_arb,
            step_fee_fund,
            fund_retention,
            fund_demand_lost,
            arb_active,
        });

        push_window(&mut log_rets, (cex_price / prev_cex).ln(), VOL_WINDOWS);
        push_window(&mut arb_flags, arb_d.abs() > 1e-12, VOL_WINDOWS);
        push_window(
            &mut fund_flags,
            (buy_d.abs() + sell_d.abs()) > 1e-12,
            VOL_WINDOWS,
        );
        push_window(
            &mut vol_sums,
            arb_d.abs() + buy_d.abs() + sell_d.abs(),
            VOL_WINDOWS,
        );
    }
}

fn build_observation(
    step: usize,
    cex_price: f64,
    pool: &CampbellPool,
    log_rets: &VecDeque<f64>,
    arb_flags: &VecDeque<bool>,
    fund_flags: &VecDeque<bool>,
    vol_sums: &VecDeque<f64>,
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
        inventory_skew,
        recent_vol: rolling_std(log_rets),
        recent_arb_frac: frac_true(arb_flags),
        recent_fund_frac: frac_true(fund_flags),
        recent_volume: vol_sums.iter().sum(),
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
) -> (f64, f64, f64, f64, f64, f64, f64, f64, bool) {
    pool.amm_fee = fee;

    let fund_scale = effective_fund_scale(config, step, regime_rng);
    let fund_retention = (1.0
        - config.e1_lambda * ((fee - config.e1_fee_ref).max(0.0) / config.e1_fee_ref))
        .clamp(0.0, 1.0);
    let eff_buy = config.buy_demand * fund_scale;
    let eff_sell = config.sell_demand * fund_scale;

    let arb_active = latency_rng.gen_range(0.0f64..1.0) < config.e5_arb_prob;
    let arb_d = if arb_active {
        arb_delta(pool, cex_price, config.cex_fee)
    } else {
        0.0
    };
    pool.apply_delta(arb_d);
    let fee_after_arb = pool.cumulative_fee_revenue;

    let buy_full = fundamental_buy_delta(eff_buy, pool, cex_price, config.cex_fee);
    let buy_d = buy_full * fund_retention;
    pool.apply_delta(buy_d);
    let sell_full = fundamental_sell_delta(-eff_sell, pool, cex_price, config.cex_fee);
    let sell_d = sell_full * fund_retention;
    pool.apply_delta(sell_d);
    let fund_demand_lost = (buy_full.abs() + sell_full.abs()) * (1.0 - fund_retention);

    let step_fee = pool.cumulative_fee_revenue - fee_before;
    let step_fee_arb = fee_after_arb - fee_before;
    let step_fee_fund = pool.cumulative_fee_revenue - fee_after_arb;

    (
        step_fee,
        arb_d,
        buy_d,
        sell_d,
        step_fee_arb,
        step_fee_fund,
        fund_retention,
        fund_demand_lost,
        arb_active,
    )
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

fn push_window<T>(buf: &mut VecDeque<T>, val: T, capacity: usize) {
    if buf.len() == capacity {
        buf.pop_front();
    }
    buf.push_back(val);
}

fn rolling_std(buf: &VecDeque<f64>) -> f64 {
    if buf.len() < 2 {
        return 0.0;
    }
    let n = buf.len() as f64;
    let mean = buf.iter().sum::<f64>() / n;
    (buf.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n).sqrt()
}

fn frac_true(buf: &VecDeque<bool>) -> f64 {
    if buf.is_empty() {
        return 0.0;
    }
    buf.iter().filter(|&&b| b).count() as f64 / buf.len() as f64
}
