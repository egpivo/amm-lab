use crate::campbell::fee_policy::{FeeObservation, FeePolicy};
use crate::campbell::pool::CampbellPool;
use crate::campbell::trader::{arb_delta, fundamental_buy_delta, fundamental_sell_delta};
use serde::{Deserialize, Serialize};

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
}

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

pub fn run_simulation(
    config: &SimConfig,
    cex_prices: &[f64],
    policy: &mut dyn FeePolicy,
) -> Vec<StepRecord> {
    let mut pool = CampbellPool::new(config.reserve_x, config.reserve_y, config.amm_fee);
    let mut hedging = pool.pool_value(cex_prices[0]);

    let mut records = Vec::new();
    for (step, &cex_price) in cex_prices[1..].iter().enumerate() {
        let prev_cex = cex_prices[step];
        hedging += pool.reserve_y * (cex_price - prev_cex);
        let fee_before = pool.cumulative_fee_revenue;

        let oracle_gap_bps = (pool.marginal_price() - cex_price) / cex_price * 10_000.0;
        let inventory_skew = (pool.reserve_x - pool.reserve_y * cex_price)
            / (pool.reserve_x + pool.reserve_y * cex_price);
        let obs = FeeObservation {
            step,
            external_price: cex_price,
            amm_price: pool.marginal_price(),
            oracle_gap_bps,
            inventory_skew,
            recent_vol: 0.0,
        };
        let fee = policy.fee(&obs);
        pool.amm_fee = fee;

        let arb_delta = arb_delta(&pool, cex_price, config.cex_fee);
        pool.apply_delta(arb_delta);

        let buy_delta = fundamental_buy_delta(config.buy_demand, &pool, cex_price, config.cex_fee);
        pool.apply_delta(buy_delta);

        let sell_delta =
            fundamental_sell_delta(-config.sell_demand, &pool, cex_price, config.cex_fee);
        pool.apply_delta(sell_delta);

        let step_fee = pool.cumulative_fee_revenue - fee_before;
        records.push(StepRecord {
            step,
            cex_price,
            amm_price: pool.marginal_price(),
            arb_delta,
            buy_delta,
            sell_delta,
            step_fee,
            pool_value: pool.pool_value(cex_price),
            hedging_portfolio: hedging,
            pool_x: pool.reserve_x,
            pool_y: pool.reserve_y,
            fee_used: fee,
            oracle_gap_bps,
            inventory_skew,
        });
    }
    records
}
