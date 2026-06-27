use crate::amount::TokenAmount;
use crate::pool::Pool;
use crate::swap::{SwapDirection, quote, swap};

#[derive(Debug)]
pub struct ArbitrageStep {
    pub step: u32,
    pub direction: SwapDirection,
    pub amount_in: TokenAmount,
    pub amount_out: TokenAmount,
    pub fee_paid: TokenAmount,
    pub pool_price_after: f64,
    pub arb_profit_est: f64,
}

pub fn run_arbitrage(pool: &mut Pool, external_price: f64, max_steps: u32) -> Vec<ArbitrageStep> {
    let mut steps = Vec::new();
    for step in 0..max_steps {
        let pool_price = pool.spot_price();

        let direction = if pool_price < external_price {
            SwapDirection::YtoX
        } else {
            SwapDirection::XtoY
        };

        let reserve_in = match direction {
            SwapDirection::XtoY => pool.reserve_x,
            SwapDirection::YtoX => pool.reserve_y,
        };
        let mut low: u128 = 1;
        let mut high: u128 = reserve_in / 2;

        for _ in 0..50 {
            let mid = low + (high - low) / 2;
            if mid == 0 {
                break;
            }
            let profit = match quote(pool, direction, mid) {
                Ok(q) => {
                    let out = q.amount_out as f64;
                    let inp = mid as f64;
                    match direction {
                        SwapDirection::XtoY => out - inp * external_price,
                        SwapDirection::YtoX => out * external_price - inp,
                    }
                }
                Err(_) => -1.0,
            };
            if profit > 0.0 {
                low = mid;
            } else {
                high = mid;
            }
        }
        let amount_in = low;

        let final_quote = match quote(pool, direction, amount_in) {
            Ok(q) => q,
            Err(_) => break,
        };
        let profit = match direction {
            SwapDirection::XtoY => {
                final_quote.amount_out as f64 - amount_in as f64 * external_price
            }
            SwapDirection::YtoX => {
                final_quote.amount_out as f64 * external_price - amount_in as f64
            }
        };
        if profit <= 0.0 {
            break;
        }

        let receipt = match swap(pool, direction, amount_in) {
            Ok(r) => r,
            Err(_) => break,
        };
        steps.push(ArbitrageStep {
            step,
            direction,
            amount_in,
            amount_out: receipt.quote.amount_out,
            fee_paid: receipt.quote.fee_amount,
            pool_price_after: pool.spot_price(),
            arb_profit_est: profit,
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
        // pool price = 1.0, external = 1.1
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
    fn test_arb_each_step_profitiable() {
        let mut pool = Pool::new(1_000_000, 1_000_000, 30).unwrap();
        let steps = run_arbitrage(&mut pool, 1.2, 20);
        for step in &steps {
            assert!(step.arb_profit_est > 0.0);
        }
    }
}
