use crate::amount::{TokenAmount, mul_div};
use crate::error::AmmError;
use crate::pool::Pool;

#[derive(Debug)]
pub struct LiquidityPosition {
    pub lp_shares: TokenAmount,
    pub entry_reserve_x: TokenAmount,
    pub entry_reserve_y: TokenAmount,
    pub entry_lp_supply: TokenAmount,
}

#[derive(Debug)]
pub struct LpPerformanceReport {
    pub withdraw_x: TokenAmount,
    pub withdraw_y: TokenAmount,
    /// Proxy: LP's pro-rata share of total fees, valued at external_price.
    /// Assumes the scenario LP is the sole liquidity provider.
    pub fee_income_value_in_y: f64,
    pub hold_value_in_y: f64,
    pub lp_value_in_y: f64,
    pub impermanent_loss_pct: f64,
    pub net_profit_loss_in_y: f64,
}

pub fn compute_lp_performance(
    position: &LiquidityPosition,
    pool: &Pool,
    external_price: f64,
) -> Result<LpPerformanceReport, AmmError> {
    // Current redemption value.
    let withdraw_x = mul_div(position.lp_shares, pool.reserve_x, pool.lp_supply)?;
    let withdraw_y = mul_div(position.lp_shares, pool.reserve_y, pool.lp_supply)?;

    // What the LP deposited (in token units), reconstructed from entry state.
    let entry_x = mul_div(
        position.lp_shares,
        position.entry_reserve_x,
        position.entry_lp_supply,
    )?;
    let entry_y = mul_div(
        position.lp_shares,
        position.entry_reserve_y,
        position.entry_lp_supply,
    )?;

    let hold_value_in_y = entry_x as f64 * external_price + entry_y as f64;
    let lp_value_in_y = withdraw_x as f64 * external_price + withdraw_y as f64;

    // IL via the closed-form formula: 2√r/(1+r) − 1, r = p1/p0.
    let p0 = position.entry_reserve_y as f64 / position.entry_reserve_x as f64;
    let r = external_price / p0;
    let il = 2.0 * r.sqrt() / (1.0 + r) - 1.0;

    // Fee income proxy: LP's share of accumulated fees valued at external price.
    // This is a proxy because fee income is already embedded in reserve growth;
    // we report it separately for article clarity.
    let share = position.lp_shares as f64 / pool.lp_supply as f64;
    let fee_income_value_in_y = pool.fee_x_accumulated as f64 * external_price * share
        + pool.fee_y_accumulated as f64 * share;

    let net_profit_loss_in_y = lp_value_in_y - hold_value_in_y;

    Ok(LpPerformanceReport {
        withdraw_x,
        withdraw_y,
        fee_income_value_in_y,
        hold_value_in_y,
        lp_value_in_y,
        impermanent_loss_pct: il * 100.0,
        net_profit_loss_in_y,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arbitrage::run_arbitrage;
    use crate::pool::Pool;

    #[test]
    fn test_no_il_when_price_unchanged() {
        let pool = Pool::new(1_000_000, 1_000_000, 30).unwrap();
        let position = LiquidityPosition {
            lp_shares: pool.lp_supply,
            entry_reserve_x: pool.reserve_x,
            entry_reserve_y: pool.reserve_y,
            entry_lp_supply: pool.lp_supply,
        };

        let report = compute_lp_performance(&position, &pool, 1.0).unwrap();
        assert!(report.impermanent_loss_pct.abs() < 0.001);
        assert!((report.lp_value_in_y - report.hold_value_in_y).abs() < 1.0);
    }

    #[test]
    fn test_il_increases_with_price_movement() {
        let mut pool = Pool::new(1_000_000, 1_000_000, 30).unwrap();
        let position = LiquidityPosition {
            lp_shares: pool.lp_supply,
            entry_reserve_x: pool.reserve_x,
            entry_reserve_y: pool.reserve_y,
            entry_lp_supply: pool.lp_supply,
        };
        run_arbitrage(&mut pool, 2.0, 50);
        let report = compute_lp_performance(&position, &pool, 2.0).unwrap();
        assert!(report.impermanent_loss_pct < 0.0);
        assert!(report.net_profit_loss_in_y < 0.0);
    }

    #[test]
    fn test_fee_income_positive_after_swaps() {
        use crate::swap::SwapDirection;
        use crate::swap::swap;
        let mut pool = Pool::new(1_000_000, 1_000_000, 30).unwrap();
        let position = LiquidityPosition {
            lp_shares: pool.lp_supply,
            entry_reserve_x: pool.reserve_x,
            entry_reserve_y: pool.reserve_y,
            entry_lp_supply: pool.lp_supply,
        };
        swap(&mut pool, SwapDirection::XtoY, 10_000).unwrap();
        let report = compute_lp_performance(&position, &pool, 1.0).unwrap();
        assert!(report.fee_income_value_in_y > 0.0);
    }
}
