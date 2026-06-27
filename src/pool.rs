use crate::amount::{BasisPoints, TokenAmount, isqrt};
use crate::error::AmmError;

pub struct Pool {
    pub reserve_x: TokenAmount,
    pub reserve_y: TokenAmount,
    pub lp_supply: TokenAmount,
    pub fee_bps: BasisPoints,
}

impl Pool {
    pub fn new(
        reserve_x: TokenAmount,
        reserve_y: TokenAmount,
        fee_bps: BasisPoints,
    ) -> Result<Self, AmmError> {
        if reserve_x == 0 || reserve_y == 0 {
            return Err(AmmError::InvalidReserves);
        }
        if fee_bps > 10_000 {
            return Err(AmmError::InvalidFeeBps);
        }
        let product = reserve_x.checked_mul(reserve_y).ok_or(AmmError::Overflow)?;
        let lp_supply = isqrt(product);
        Ok(Pool {
            reserve_x,
            reserve_y,
            lp_supply,
            fee_bps,
        })
    }
    pub fn invariant(&self) -> u128 {
        self.reserve_x.saturating_mul(self.reserve_y)
    }
    pub fn spot_price(&self) -> f64 {
        // reserve_y / reserve_x
        self.reserve_y as f64 / self.reserve_x as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_new_valid() {
        let pool = Pool::new(1_000_000, 1_000_000, 30).unwrap();
        assert_eq!(pool.reserve_x, 1_000_000);
        assert_eq!(pool.reserve_y, 1_000_000);
        assert_eq!(pool.fee_bps, 30);
        assert!(pool.lp_supply > 0);
    }

    #[test]
    fn test_pool_new_zero_reserve() {
        assert!(Pool::new(0, 1_000_000, 30).is_err());
        assert!(Pool::new(1_000_000, 0, 30).is_err());
    }

    #[test]
    fn test_pool_new_invalid_fee() {
        assert!(Pool::new(1_000_000, 1_000_000, 10_001).is_err());
    }

    #[test]
    fn test_pool_invariant() {
        let pool = Pool::new(1_000_000, 2_000_000, 30).unwrap();
        assert_eq!(pool.invariant(), 2_000_000_000_000);
    }

    #[test]
    fn test_pool_spot_price() {
        let pool = Pool::new(1_000_000, 2_000_000, 30).unwrap();
        let price = pool.spot_price();
        assert!((price - 2.0).abs() < 1e-9);
    }
}
