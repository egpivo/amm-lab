//! Canonical token / base-quote orientation before any mark is computed.

use crate::pstt::error::{PsttError, Result};
use crate::pstt::schema::Address;
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct Orientation {
    pub pool: Address,
    pub pair: String,
    pub base_symbol: String,
    pub quote_symbol: String,
    pub token0: Address,
    pub token1: Address,
    pub base_is_token0: bool,
    pub base_decimals: u8,
    pub quote_decimals: u8,
    pub cex_symbol: String,
    pub invert: bool,
}

#[derive(Debug, Clone)]
pub struct OrientationSpec {
    pub pool: Address,
    pub pair: String,
    pub base_symbol: String,
    pub quote_symbol: String,
    pub token0: Address,
    pub token1: Address,
    pub expected_token0: Address,
    pub expected_token1: Address,
    pub base_decimals: u8,
    pub quote_decimals: u8,
    pub cex_symbol: String,
    pub invert: bool,
    pub canonical_tokens: BTreeMap<String, Address>,
}

pub fn validate_orientation(spec: &OrientationSpec) -> Result<Orientation> {
    let t0 = &spec.token0.0;
    let t1 = &spec.token1.0;
    if t0 == t1 {
        return Err(PsttError::schema(format!(
            "token0 == token1 for pool {}",
            spec.pool
        )));
    }
    if spec.expected_token0.0 != *t0 || spec.expected_token1.0 != *t1 {
        return Err(PsttError::schema(format!(
            "canonical token mismatch for pool {}: got ({t0},{t1}) expected ({},{})",
            spec.pool, spec.expected_token0, spec.expected_token1
        )));
    }
    let base_addr = spec
        .canonical_tokens
        .get(&spec.base_symbol)
        .ok_or_else(|| {
            PsttError::schema(format!(
                "missing canonical address for base {}",
                spec.base_symbol
            ))
        })?;
    let quote_addr = spec
        .canonical_tokens
        .get(&spec.quote_symbol)
        .ok_or_else(|| {
            PsttError::schema(format!(
                "missing canonical address for quote {}",
                spec.quote_symbol
            ))
        })?;
    if base_addr.0 == quote_addr.0 {
        return Err(PsttError::schema("base and quote addresses collide"));
    }
    let observed = {
        let mut s = [t0.clone(), t1.clone()];
        s.sort();
        s
    };
    let expected = {
        let mut s = [base_addr.0.clone(), quote_addr.0.clone()];
        s.sort();
        s
    };
    if observed != expected {
        return Err(PsttError::schema(format!(
            "symbol/address set mismatch for {}: symbols {},{} do not match token addresses",
            spec.pool, spec.base_symbol, spec.quote_symbol
        )));
    }
    let base_is_token0 = base_addr.0 == *t0;
    if !base_is_token0 && base_addr.0 != *t1 {
        return Err(PsttError::schema(format!(
            "base {} is neither token0 nor token1 for {}",
            spec.base_symbol, spec.pool
        )));
    }
    if spec.cex_symbol.trim().is_empty() {
        return Err(PsttError::schema("cex_symbol must be non-empty"));
    }
    Ok(Orientation {
        pool: spec.pool.clone(),
        pair: spec.pair.clone(),
        base_symbol: spec.base_symbol.clone(),
        quote_symbol: spec.quote_symbol.clone(),
        token0: spec.token0.clone(),
        token1: spec.token1.clone(),
        base_is_token0,
        base_decimals: spec.base_decimals,
        quote_decimals: spec.quote_decimals,
        cex_symbol: spec.cex_symbol.clone(),
        invert: spec.invert,
    })
}

/// Stablecoin helper used by candidate assembly (not mark construction).
pub fn choose_base_symbol(token0_symbol: &str, token1_symbol: &str, stables: &[&str]) -> String {
    let s0 = stables.contains(&token0_symbol);
    let s1 = stables.contains(&token1_symbol);
    match (s0, s1) {
        // Exactly one non-stable: choose that side. Otherwise token0.
        (true, false) => token1_symbol.to_string(),
        _ => token0_symbol.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn addr(s: &str) -> Address {
        Address::normalize(s).unwrap()
    }

    #[test]
    fn accepts_token0_base_and_rejects_symbol_only_match() {
        let weth = addr("0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2");
        let usdc = addr("0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48");
        let mut canon = BTreeMap::new();
        canon.insert("WETH".into(), weth.clone());
        canon.insert("USDC".into(), usdc.clone());
        let ok = validate_orientation(&OrientationSpec {
            pool: addr("0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"),
            pair: "USDC/WETH".into(),
            base_symbol: "WETH".into(),
            quote_symbol: "USDC".into(),
            token0: usdc.clone(),
            token1: weth.clone(),
            expected_token0: usdc,
            expected_token1: weth,
            base_decimals: 18,
            quote_decimals: 6,
            cex_symbol: "ETHUSDC".into(),
            invert: false,
            canonical_tokens: canon.clone(),
        })
        .unwrap();
        assert!(!ok.base_is_token0);

        let bad = validate_orientation(&OrientationSpec {
            pool: addr("0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"),
            pair: "USDC/WETH".into(),
            base_symbol: "WETH".into(),
            quote_symbol: "USDC".into(),
            token0: addr("0x0000000000000000000000000000000000000001"),
            token1: addr("0x0000000000000000000000000000000000000002"),
            expected_token0: addr("0x0000000000000000000000000000000000000001"),
            expected_token1: addr("0x0000000000000000000000000000000000000002"),
            base_decimals: 18,
            quote_decimals: 6,
            cex_symbol: "ETHUSDC".into(),
            invert: false,
            canonical_tokens: canon,
        });
        assert!(bad.is_err());
    }
}
