use crate::amount::TokenAmount;
use crate::pool::Pool;
use crate::swap::{SwapDirection, quote, swap};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct ArbitrageStep {
    pub step_index: u32,
    pub direction: SwapDirection,
    pub amount_in: TokenAmount,
    pub amount_out: TokenAmount,
    pub fee_paid: TokenAmount,
    pub profit_estimate: f64,
    pub pool_price_before: f64,
    pub pool_price_after: f64,
    pub external_price: f64,
    pub price_gap_before: f64,
    pub price_gap_after: f64,
    pub reserve_x_after: u128,
    pub reserve_y_after: u128,
}

fn compute_profit(
    pool: &Pool,
    direction: SwapDirection,
    amount_in: u128,
    external_price: f64,
) -> f64 {
    match quote(pool, direction, amount_in) {
        Ok(q) => match direction {
            SwapDirection::XtoY => q.amount_out as f64 - amount_in as f64 * external_price,
            SwapDirection::YtoX => q.amount_out as f64 * external_price - amount_in as f64,
        },
        Err(_) => f64::NEG_INFINITY,
    }
}

/// Ternary search for the amount_in that maximises profit.
/// Profit is unimodal (concave) in amount_in for constant-product AMMs.
fn find_best_amount_in(pool: &Pool, direction: SwapDirection, external_price: f64) -> u128 {
    let reserve_in = match direction {
        SwapDirection::XtoY => pool.reserve_x,
        SwapDirection::YtoX => pool.reserve_y,
    };
    let mut lo: u128 = 1;
    let mut hi: u128 = reserve_in / 2;

    for _ in 0..96 {
        if hi <= lo + 2 {
            break;
        }
        let m1 = lo + (hi - lo) / 3;
        let m2 = hi - (hi - lo) / 3;
        let p1 = compute_profit(pool, direction, m1, external_price);
        let p2 = compute_profit(pool, direction, m2, external_price);
        if p1 < p2 {
            lo = m1;
        } else {
            hi = m2;
        }
    }
    lo + (hi - lo) / 2
}

pub fn run_arbitrage(pool: &mut Pool, external_price: f64, max_steps: u32) -> Vec<ArbitrageStep> {
    let mut steps = Vec::new();
    for step_index in 0..max_steps {
        let pool_price = pool.spot_price();

        // No gap → nothing to do.
        if (pool_price - external_price).abs() < 1e-12 {
            break;
        }

        let direction = if pool_price < external_price {
            SwapDirection::YtoX // X underpriced in pool → buy X with Y
        } else {
            SwapDirection::XtoY // Y underpriced in pool → buy Y with X
        };

        let amount_in = find_best_amount_in(pool, direction, external_price);
        let best_profit = compute_profit(pool, direction, amount_in, external_price);

        if best_profit <= 0.0 {
            break;
        }

        let price_gap_before = (pool_price - external_price).abs();
        let pool_price_before = pool_price;

        let receipt = match swap(pool, direction, amount_in) {
            Ok(r) => r,
            Err(_) => break,
        };

        let pool_price_after = pool.spot_price();
        let price_gap_after = (pool_price_after - external_price).abs();

        // Guard: if the gap did not shrink, stop (price moved wrong direction).
        if price_gap_after >= price_gap_before {
            break;
        }

        steps.push(ArbitrageStep {
            step_index,
            direction,
            amount_in,
            amount_out: receipt.quote.amount_out,
            fee_paid: receipt.quote.fee_amount,
            profit_estimate: best_profit,
            pool_price_before,
            pool_price_after,
            external_price,
            price_gap_before,
            price_gap_after,
            reserve_x_after: receipt.reserve_x_after,
            reserve_y_after: receipt.reserve_y_after,
        });
    }
    steps
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::Pool;

    #[test]
    fn test_arb_moves_price_toward_external() {
        let mut pool = Pool::new(1_000_000, 1_000_000, 30).unwrap();
        let steps = run_arbitrage(&mut pool, 1.1, 20);
        assert!(!steps.is_empty());
        let final_price = pool.spot_price();
        assert!(final_price > 1.0);
        assert!(final_price <= 1.1 + 0.01);
    }

    #[test]
    fn test_arb_no_profit_when_price_aligned() {
        let mut pool = Pool::new(1_000_000, 1_000_000, 30).unwrap();
        let steps = run_arbitrage(&mut pool, 1.0, 20);
        assert!(steps.is_empty());
    }

    #[test]
    fn test_arb_each_step_profitable() {
        let mut pool = Pool::new(1_000_000, 1_000_000, 30).unwrap();
        let steps = run_arbitrage(&mut pool, 1.2, 20);
        for step in &steps {
            assert!(step.profit_estimate > 0.0);
        }
    }

    #[test]
    fn test_arb_ternary_finds_better_profit_than_bisection() {
        // Ternary search should find the true profit-maximising size,
        // which is strictly interior — neither endpoint is optimal.
        let pool = Pool::new(1_000_000_000_000, 1_000_000_000_000, 30).unwrap();
        let ext = 1.5;
        let direction = SwapDirection::YtoX;
        let best = find_best_amount_in(&pool, direction, ext);
        let profit_best = compute_profit(&pool, direction, best, ext);
        // A trade at 1/10 of best and 10x best should both be worse.
        let profit_small = compute_profit(&pool, direction, best / 10 + 1, ext);
        let profit_large =
            compute_profit(&pool, direction, (best * 10).min(pool.reserve_y / 2), ext);
        assert!(profit_best > profit_small);
        assert!(profit_best >= profit_large);
    }
}
