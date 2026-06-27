use crate::amount::{TokenAmount, mul_div};
use crate::error::AmmError;
use crate::pool::Pool;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwapDirection {
    XtoY,
    YtoX,
}

#[derive(Debug)]
pub struct SwapQuote {
    pub direction: SwapDirection,
    pub amount_in: TokenAmount,
    pub fee_amount: TokenAmount,
    pub amount_out: TokenAmount,
    pub spot_price_before: f64,
    pub exec_price: f64,
    pub price_impact_pct: f64,
    pub invariant_before: u128,
    pub invariant_after: u128,
}

#[derive(Debug)]
pub struct SwapReceipt {
    pub quote: SwapQuote,
    pub reserve_x_before: TokenAmount,
    pub reserve_y_before: TokenAmount,
    pub reserve_x_after: TokenAmount,
    pub reserve_y_after: TokenAmount,
}

pub fn quote(
    pool: &Pool,
    direction: SwapDirection,
    amount_in: TokenAmount,
) -> Result<SwapQuote, AmmError> {
    if amount_in == 0 {
        return Err(AmmError::ZeroInput);
    }

    let (reserve_in, reserve_out) = match direction {
        SwapDirection::XtoY => (pool.reserve_x, pool.reserve_y),
        SwapDirection::YtoX => (pool.reserve_y, pool.reserve_x),
    };

    let net_factor = 10_000u128 - pool.fee_bps as u128;
    let fee_amount = amount_in - (amount_in * net_factor / 10_000);
    let denominator = reserve_in * 10_000 + amount_in * net_factor;
    let amount_out = mul_div(amount_in * net_factor, reserve_out, denominator)?;
    if amount_out == 0 {
        return Err(AmmError::ZeroOutput);
    }
    if amount_out >= reserve_out {
        return Err(AmmError::InsufficientLiquidity);
    }
    let spot_price_before = pool.spot_price();
    let exec_price = amount_in as f64 / amount_out as f64;
    let price_impact_pct = (exec_price - spot_price_before) / spot_price_before * 100.0;
    let invariant_before = pool.invariant();

    let (reserve_x_after, reserve_y_after) = match direction {
        SwapDirection::XtoY => (pool.reserve_x + amount_in, pool.reserve_y - amount_out),
        SwapDirection::YtoX => (pool.reserve_x - amount_out, pool.reserve_y + amount_in),
    };
    let invariant_after = reserve_x_after.saturating_mul(reserve_y_after);

    Ok(SwapQuote {
        direction,
        amount_in,
        fee_amount,
        amount_out,
        spot_price_before,
        exec_price,
        price_impact_pct,
        invariant_before,
        invariant_after,
    })
}

pub fn swap(
    pool: &mut Pool,
    direction: SwapDirection,
    amount_in: TokenAmount,
) -> Result<SwapReceipt, AmmError> {
    let quote = quote(pool, direction, amount_in)?;
    let reserve_x_before = pool.reserve_x;
    let reserve_y_before = pool.reserve_y;
    let (reserve_x_after, reserve_y_after) = match direction {
        SwapDirection::XtoY => (
            reserve_x_before + amount_in,
            reserve_y_before - quote.amount_out,
        ),
        SwapDirection::YtoX => (
            reserve_x_before - quote.amount_out,
            reserve_y_before + amount_in,
        ),
    };
    pool.reserve_x = reserve_x_after;
    pool.reserve_y = reserve_y_after;
    Ok(SwapReceipt {
        quote,
        reserve_x_before,
        reserve_y_before,
        reserve_x_after,
        reserve_y_after,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::Pool;

    fn make_pool() -> Pool {
        Pool::new(1_000_000, 1_000_000, 30).unwrap()
    }

    #[test]
    fn test_quote_zero_input() {
        let pool = make_pool();
        assert!(quote(&pool, SwapDirection::XtoY, 0).is_err());
    }

    #[test]
    fn test_swap_basic_x_to_y() {
        let mut pool = make_pool();
        let receipt = swap(&mut pool, SwapDirection::XtoY, 10_000).unwrap();

        assert!(receipt.quote.amount_out > 0);
        assert!(receipt.quote.amount_out < 10_000);
        assert_eq!(pool.reserve_x, 1_010_000);
        assert!(pool.reserve_y < 1_000_000);
    }

    #[test]
    fn test_invariant_non_decreasing() {
        let mut pool = make_pool();
        let before = pool.invariant();
        swap(&mut pool, SwapDirection::XtoY, 10_000).unwrap();
        let after = pool.invariant();
        assert!(after >= before);
    }

    #[test]
    fn test_larger_swap_worse_price() {
        let pool_small = make_pool();
        let pool_large = make_pool();
        let mut p1 = pool_small;
        let mut p2 = pool_large;
        let r1 = swap(&mut p1, SwapDirection::XtoY, 10_000).unwrap();
        let r2 = swap(&mut p2, SwapDirection::YtoX, 100_000).unwrap();
        assert!(r2.quote.exec_price > r1.quote.exec_price);
    }
}
