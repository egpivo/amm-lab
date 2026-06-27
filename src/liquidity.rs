use crate::amount::{TokenAmount, mul_div};
use crate::error::AmmError;
use crate::pool::Pool;

#[derive(Debug)]
pub struct AddLiquidityResult {
    pub amount_x: TokenAmount,
    pub amount_y: TokenAmount,
    pub lp_minted: TokenAmount,
    pub reserve_x_new: TokenAmount,
    pub reserve_y_new: TokenAmount,
    pub lp_supply_new: TokenAmount,
}

#[derive(Debug)]
pub struct RemoveLiquidityResult {
    pub amount_x: TokenAmount,
    pub amount_y: TokenAmount,
    pub reserve_x_new: TokenAmount,
    pub reserve_y_new: TokenAmount,
    pub lp_supply_new: TokenAmount,
}

pub fn add_liquidity(
    pool: &mut Pool,
    amount_x: TokenAmount,
    amount_y: TokenAmount,
) -> Result<AddLiquidityResult, AmmError> {
    if amount_x == 0 || amount_y == 0 {
        return Err(AmmError::ZeroInput);
    }
    if amount_x * pool.reserve_y != amount_y * pool.reserve_x {
        return Err(AmmError::NonProportionalDeposit);
    }

    let lp_minted = mul_div(pool.lp_supply, amount_x, pool.reserve_x)?;
    if lp_minted == 0 {
        return Err(AmmError::ZeroLpShares);
    }

    pool.reserve_x += amount_x;
    pool.reserve_y += amount_y;
    pool.lp_supply += lp_minted;

    Ok(AddLiquidityResult {
        amount_x,
        amount_y,
        lp_minted,
        reserve_x_new: pool.reserve_x,
        reserve_y_new: pool.reserve_y,
        lp_supply_new: pool.lp_supply,
    })
}

pub fn remove_liquidity(
    pool: &mut Pool,
    lp_shares: TokenAmount,
) -> Result<RemoveLiquidityResult, AmmError> {
    if lp_shares == 0 {
        return Err(AmmError::ZeroLpShares);
    }

    if lp_shares > pool.lp_supply {
        return Err(AmmError::InsufficientLiquidity);
    }

    let amount_x = mul_div(lp_shares, pool.reserve_x, pool.lp_supply)?;
    let amount_y = mul_div(lp_shares, pool.reserve_y, pool.lp_supply)?;

    pool.reserve_x -= amount_x;
    pool.reserve_y -= amount_y;
    pool.lp_supply -= lp_shares;

    Ok(RemoveLiquidityResult {
        amount_x,
        amount_y,
        reserve_x_new: pool.reserve_x,
        reserve_y_new: pool.reserve_y,
        lp_supply_new: pool.lp_supply,
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
    fn test_add_liquidity_proportional() {
        let mut pool = make_pool();
        let result = add_liquidity(&mut pool, 100_000, 100_000).unwrap();
        assert_eq!(pool.reserve_x, 1_100_000);
        assert_eq!(pool.reserve_y, 1_100_000);
        assert!(result.lp_minted > 0);
    }

    #[test]
    fn test_add_liquidity_non_proportional() {
        let mut pool = make_pool();
        assert!(add_liquidity(&mut pool, 100_000, 200_000).is_err());
    }

    #[test]
    fn test_remove_liquidity_partial() {
        let mut pool = make_pool();
        let lp_total = pool.lp_supply;
        let result = remove_liquidity(&mut pool, lp_total / 2).unwrap();
        assert!(result.amount_x > 0);
        assert!(result.amount_y > 0);
        assert!(pool.reserve_x < 1_000_000);
    }

    #[test]
    fn test_remove_liquidity_full() {
        let mut pool = make_pool();
        let lp_total = pool.lp_supply;
        remove_liquidity(&mut pool, lp_total).unwrap();
        assert_eq!(pool.lp_supply, 0);
    }

    #[test]
    fn test_remove_zero_shares() {
        let mut pool = make_pool();
        assert!(remove_liquidity(&mut pool, 0).is_err());
    }
}
