use crate::campbell::fee_policy::{FeeObservation, FeePolicy};
use crate::campbell::pool::CampbellPool;
use crate::campbell::trader::{arb_delta, fundamental_buy_delta, fundamental_sell_delta};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

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

pub fn run_simulation(
    config: &SimConfig,
    cex_prices: &[f64],
    policy: &mut dyn FeePolicy,
) -> Vec<StepRecord> {
    let mut pool = CampbellPool::new(config.reserve_x, config.reserve_y, config.amm_fee);
    let mut hedging = pool.pool_value(cex_prices[0]);
    let mut regime_rng = StdRng::seed_from_u64(config.seed.wrapping_add(99_991));
    // E5: dedicated exogenous latency stream (identical across policies on the same seed)
    let mut latency_rng = StdRng::seed_from_u64(config.seed ^ 77_777);

    let mut records = Vec::new();
    const VOL_WINDOWS: usize = 20;
    let mut log_rets: VecDeque<f64> = VecDeque::with_capacity(VOL_WINDOWS + 1);
    let mut arb_flags: VecDeque<bool> = VecDeque::with_capacity(VOL_WINDOWS + 1);
    let mut fund_flags: VecDeque<bool> = VecDeque::with_capacity(VOL_WINDOWS + 1);
    let mut vol_sums: VecDeque<f64> = VecDeque::with_capacity(VOL_WINDOWS + 1);

    let mut previous_fee = config.amm_fee;

    for (step, &cex_price) in cex_prices[1..].iter().enumerate() {
        let prev_cex = cex_prices[step];
        hedging += pool.reserve_y * (cex_price - prev_cex);
        let fee_before = pool.cumulative_fee_revenue;

        let oracle_gap_bps = (pool.marginal_price() - cex_price) / cex_price * 10_000.0;
        let inventory_skew = (pool.reserve_x - pool.reserve_y * cex_price)
            / (pool.reserve_x + pool.reserve_y * cex_price);
        let recent_vol = rolling_std(&log_rets);
        let recent_arb_frac = frac_true(&arb_flags);
        let recent_fund_frac = frac_true(&fund_flags);
        let recent_volume: f64 = vol_sums.iter().sum();

        let obs = FeeObservation {
            step,
            external_price: cex_price,
            amm_price: pool.marginal_price(),
            oracle_gap_bps,
            inventory_skew,
            recent_vol,
            recent_arb_frac,
            recent_fund_frac,
            recent_volume,
            previous_fee,
        };
        let fee = policy.fee(&obs);
        pool.amm_fee = fee;
        previous_fee = fee;

        let fund_scale = effective_fund_scale(config, step, &mut regime_rng);
        // C1a/E1: linear-threshold retention on fundamental demand only (spec-frozen form);
        // retention(f) = clamp(1 - lambda*max(0, f - f_ref)/f_ref, 0, 1); lambda=0 => 1.0
        let fund_retention = (1.0
            - config.e1_lambda * ((fee - config.e1_fee_ref).max(0.0) / config.e1_fee_ref))
            .clamp(0.0, 1.0);
        let eff_buy = config.buy_demand * fund_scale;
        let eff_sell = config.sell_demand * fund_scale;

        // E5: arbitrage inclusion is Bernoulli(q) per step from the dedicated stream;
        // the draw happens EVERY step (stream alignment across q levels and policies).
        let arb_active = latency_rng.gen_range(0.0f64..1.0) < config.e5_arb_prob;
        let arb_d = if arb_active {
            arb_delta(&pool, cex_price, config.cex_fee)
        } else {
            0.0
        };
        pool.apply_delta(arb_d);
        let fee_after_arb = pool.cumulative_fee_revenue;
        // E1 v1.1 (spec amendment): retention scales the EXECUTED fundamental trade
        // (routing share to the outside venue); demand cap is slack in this engine.
        let buy_full = fundamental_buy_delta(eff_buy, &pool, cex_price, config.cex_fee);
        let buy_d = buy_full * fund_retention;
        pool.apply_delta(buy_d);
        let sell_full = fundamental_sell_delta(-eff_sell, &pool, cex_price, config.cex_fee);
        let sell_d = sell_full * fund_retention;
        pool.apply_delta(sell_d);
        let fund_demand_lost = (buy_full.abs() + sell_full.abs()) * (1.0 - fund_retention);

        let step_fee = pool.cumulative_fee_revenue - fee_before;
        let step_fee_arb = fee_after_arb - fee_before;
        let step_fee_fund = pool.cumulative_fee_revenue - fee_after_arb;
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
            oracle_gap_bps,
            inventory_skew,
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
    records
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
