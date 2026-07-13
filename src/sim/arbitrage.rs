//! Arbitrageur closing the AMM–CEX gap when profitable after fees and gas.
//!
//! Closed form under the fee-outside-reserves convention: the arb trades
//! until the pool's fee-adjusted marginal price equals the CEX fee-adjusted
//! price, then executes only if realized profit exceeds gas. `speed` < 1.0
//! makes the arb probabilistic per step (latency / competition proxy).

use crate::sim::amm::Pool;
use rand::Rng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbConfig {
    pub cex_fee: f64,
    /// Fixed gas cost per arb transaction, X units.
    pub gas_cost: f64,
    /// Probability the arbitrageur acts in a given step.
    pub speed: f64,
}

impl Default for ArbConfig {
    fn default() -> Self {
        Self {
            cex_fee: 0.001,
            gas_cost: 5.0,
            speed: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ArbTrade {
    /// Positive = arb bought Y from the pool; negative = sold Y to the pool.
    pub delta_y: f64,
    /// Realized profit in X after fees and gas.
    pub profit_x: f64,
}

/// Attempt one arb on `pool` against the oracle. Mutates the pool only if
/// the trade clears the gas hurdle. Returns the executed trade, if any.
pub fn arbitrage_step(
    pool: &mut Pool,
    oracle_price: f64,
    cfg: &ArbConfig,
    rng: &mut StdRng,
) -> Option<ArbTrade> {
    if cfg.speed < 1.0 && rng.gen_range(0.0..1.0) > cfg.speed {
        return None;
    }
    let k = pool.reserve_x * pool.reserve_y;
    let p = pool.mid_price();

    // Buy Y from pool, sell on CEX: profitable while
    // marginal cost p/(1-fee_buy) < oracle*(1-cex_fee).
    let target_buy = oracle_price * (1.0 - cfg.cex_fee) * (1.0 - pool.fee_buy);
    if p < target_buy {
        let y_target = (k / target_buy).sqrt();
        let dy = pool.reserve_y - y_target;
        if let Some(fill) = pool.buy_cost(dy) {
            let profit = dy * oracle_price * (1.0 - cfg.cex_fee) - fill.amount_x - cfg.gas_cost;
            if profit > 0.0 {
                pool.buy(dy);
                return Some(ArbTrade {
                    delta_y: dy,
                    profit_x: profit,
                });
            }
        }
        return None;
    }

    // Sell Y to pool, buy back on CEX: profitable while
    // marginal proceeds p*(1-fee_sell) > oracle*(1+cex_fee).
    let target_sell = oracle_price * (1.0 + cfg.cex_fee) / (1.0 - pool.fee_sell);
    if p > target_sell {
        let y_target = (k / target_sell).sqrt();
        let dy = y_target - pool.reserve_y;
        if let Some(fill) = pool.sell_proceeds(dy) {
            let profit = fill.amount_x - dy * oracle_price * (1.0 + cfg.cex_fee) - cfg.gas_cost;
            if profit > 0.0 {
                pool.sell(dy);
                return Some(ArbTrade {
                    delta_y: -dy,
                    profit_x: profit,
                });
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn arb_closes_gap_toward_oracle() {
        let mut pool = Pool::new(1_100_000.0, 1_000.0, 0.003, 0.003);
        let oracle = 1_000.0; // pool mid = 1100, overpriced
        let mut rng = StdRng::seed_from_u64(1);
        let cfg = ArbConfig::default();
        let gap_before = (pool.mid_price() - oracle).abs();
        let trade = arbitrage_step(&mut pool, oracle, &cfg, &mut rng).unwrap();
        assert!(trade.profit_x > 0.0);
        assert!(trade.delta_y < 0.0); // sells Y into the rich pool
        assert!((pool.mid_price() - oracle).abs() < gap_before);
    }

    #[test]
    fn no_arb_inside_fee_band() {
        let mut pool = Pool::new(1_000_000.0, 1_000.0, 0.003, 0.003);
        let mut rng = StdRng::seed_from_u64(1);
        let cfg = ArbConfig::default();
        assert!(arbitrage_step(&mut pool, 1_000.5, &cfg, &mut rng).is_none());
    }
}
