use thiserror::Error;

#[derive(Debug, Error)]
pub enum AmmError {
    #[error("input amount must be greater than zero")]
    ZeroInput,
    #[error("empty pool")]
    EmptyPool,
    #[error("zero output")]
    ZeroOutput,
    #[error("insufficient liquidity")]
    InsufficientLiquidity,
    #[error("overflow")]
    Overflow,
    #[error("invalid fee bps")]
    InvalidFeeBps,
    #[error("nonproportional deposit")]
    NonProportionalDeposit,
    #[error("zero liquid pool shares")]
    ZeroLpShares,
    #[error("invalid reserves")]
    InvalidReserves,
    #[error("division by zero")]
    DivisionByZero,
    #[error("slippage limit exceeded: expected at least {min_out}, got {actual}")]
    SlippageFailed { min_out: u128, actual: u128 },
}
