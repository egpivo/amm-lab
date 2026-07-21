//! Empirical signed execution-loss marks. Not canonical LVR.

use crate::pstt::error::{PsttError, Result};
use crate::pstt::orientation::Orientation;
use crate::pstt::schema::OrientedFill;

#[derive(Debug, Clone, Copy)]
pub struct MarkComponents {
    pub q: f64,
    pub p_exec: f64,
    pub direction: f64,
    pub ell: f64,
    pub a: f64,
    pub b: f64,
    pub l: f64,
}

impl MarkComponents {
    pub fn identity_ok(self, abs_tol: f64) -> bool {
        (self.l - (self.a - self.b)).abs() <= abs_tol
    }
}

/// Build oriented fill quantities from raw token amounts.
///
/// Panel Stage-4 convention when base is token0:
/// `q = |amount0| / 10^base_decimals`,
/// `p_exec = (|amount1|/10^quote_decimals) / q`,
/// `direction = -1` if `amount0 > 0` else `+1`.
pub fn oriented_from_amounts(
    orientation: &Orientation,
    amount0: i128,
    amount1: i128,
) -> Result<(f64, f64, f64)> {
    let (amount_base, amount_quote) = if orientation.base_is_token0 {
        (amount0, amount1)
    } else {
        (amount1, amount0)
    };
    if amount_base == 0 {
        return Err(PsttError::parse("zero base amount"));
    }
    let q = (amount_base.unsigned_abs() as f64) / 10f64.powi(orientation.base_decimals as i32);
    let quote_q =
        (amount_quote.unsigned_abs() as f64) / 10f64.powi(orientation.quote_decimals as i32);
    if !(q.is_finite() && q > 0.0 && quote_q.is_finite()) {
        return Err(PsttError::parse("non-finite oriented quantities"));
    }
    let p_exec = quote_q / q;
    if !(p_exec.is_finite() && p_exec > 0.0) {
        return Err(PsttError::parse("non-finite execution price"));
    }
    let direction = if amount_base > 0 { -1.0 } else { 1.0 };
    Ok((q, p_exec, direction))
}

/// `ell = direction * q * (p_ref - p_exec)`; `A=max(ell,0)`, `B=max(-ell,0)`, `L=A-B`.
pub fn signed_mark(direction: f64, q: f64, p_ref: f64, p_exec: f64) -> Result<MarkComponents> {
    if ![direction, q, p_ref, p_exec].iter().all(|x| x.is_finite())
        || q <= 0.0
        || p_ref <= 0.0
        || p_exec <= 0.0
    {
        return Err(PsttError::parse("non-finite or non-positive mark inputs"));
    }
    let ell = direction * q * (p_ref - p_exec);
    if !ell.is_finite() {
        return Err(PsttError::parse("non-finite ell"));
    }
    let a = ell.max(0.0);
    let b = (-ell).max(0.0);
    let l = a - b;
    let mark = MarkComponents {
        q,
        p_exec,
        direction,
        ell,
        a,
        b,
        l,
    };
    if !mark.identity_ok(1e-12) {
        return Err(PsttError::invariant("L != A - B beyond tolerance"));
    }
    Ok(mark)
}

pub fn mark_fill(fill: &OrientedFill, p_ref: f64) -> Result<MarkComponents> {
    signed_mark(fill.direction, fill.q, p_ref, fill.p_exec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pstt::schema::Address;
    use std::collections::BTreeMap;

    fn orient(base_is_token0: bool) -> Orientation {
        let t0 = Address::normalize("0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48").unwrap();
        let t1 = Address::normalize("0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2").unwrap();
        Orientation {
            pool: Address::normalize("0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640").unwrap(),
            pair: "USDC/WETH".into(),
            base_symbol: "WETH".into(),
            quote_symbol: "USDC".into(),
            token0: t0,
            token1: t1,
            base_is_token0,
            base_decimals: if base_is_token0 { 6 } else { 18 },
            quote_decimals: if base_is_token0 { 18 } else { 6 },
            cex_symbol: "ETHUSDC".into(),
            invert: false,
        }
    }

    #[test]
    fn both_base_flow_signs_preserve_identity() {
        let o = orient(false);
        let (q, p_exec, dir) =
            oriented_from_amounts(&o, 2_500_000_000, -1_000_000_000_000_000_000).unwrap();
        assert!((dir - 1.0).abs() < 1e-15);
        let m = signed_mark(dir, q, 2600.0, p_exec).unwrap();
        assert!(m.identity_ok(1e-12));
        assert!((m.l - (m.a - m.b)).abs() < 1e-12);

        let (q2, p2, d2) =
            oriented_from_amounts(&o, -2_500_000_000, 1_000_000_000_000_000_000).unwrap();
        assert!((d2 + 1.0).abs() < 1e-15);
        let m2 = signed_mark(d2, q2, 2400.0, p2).unwrap();
        assert!(m2.identity_ok(1e-12));
        let _ = BTreeMap::<String, Address>::new();
    }
}
