//! Binance aggTrade and BBO parsers. Parsers return data; they do not choose
//! venue, symbol, orientation, or gates.

use crate::pstt::block_time::normalize_aggtrade_timestamp;
use crate::pstt::error::{PsttError, Result};
use crate::pstt::schema::AggTrade;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Parse one headerless Binance aggTrades CSV line.
///
/// Columns: `id,price,quantity,first_id,last_id,timestamp,is_buyer_maker,is_best_match`
pub fn parse_aggtrade_line(line: &str, invert: bool) -> Result<Option<AggTrade>> {
    let line = line.trim();
    if line.is_empty() {
        return Ok(None);
    }
    let fields: Vec<&str> = line.split(',').collect();
    if fields.len() < 6 {
        return Ok(None);
    }
    let price: f64 = match fields[1].parse::<f64>() {
        Ok(v) if v.is_finite() && v > 0.0 => v,
        _ => return Ok(None),
    };
    let quantity: f64 = match fields[2].parse::<f64>() {
        Ok(v) if v.is_finite() && v >= 0.0 => v,
        _ => return Ok(None),
    };
    let stamp: i64 = match fields[5].parse::<i64>() {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };
    let timestamp_secs = normalize_aggtrade_timestamp(stamp)?;
    let price = if invert { 1.0 / price } else { price };
    if !price.is_finite() || price <= 0.0 {
        return Ok(None);
    }
    Ok(Some(AggTrade {
        timestamp_secs,
        price,
        quantity,
    }))
}

/// Stream a headerless aggTrades CSV file into normalized records.
/// Archive traversal order must be sorted by the caller; this function preserves
/// file order and then sorts by timestamp for join readiness.
pub fn load_aggtrades_csv(path: &Path, invert: bool) -> Result<Vec<AggTrade>> {
    let file = File::open(path).map_err(|e| PsttError::io(path, e))?;
    let reader = BufReader::new(file);
    let mut rows = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| PsttError::io(path, e))?;
        match parse_aggtrade_line(&line, invert) {
            Ok(Some(row)) => rows.push(row),
            Ok(None) => continue,
            Err(e) => {
                return Err(PsttError::parse(format!(
                    "{}:{}: {e}",
                    path.display(),
                    idx + 1
                )));
            }
        }
    }
    rows.sort_by(|a, b| {
        a.timestamp_secs
            .partial_cmp(&b.timestamp_secs)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(rows)
}

/// Native BBO midpoint when bid/ask are positive and `bid <= ask`.
pub fn native_bbo_mid(bid: f64, ask: f64) -> Option<f64> {
    if bid.is_finite() && ask.is_finite() && bid > 0.0 && ask > 0.0 && bid <= ask {
        Some(0.5 * (bid + ask))
    } else {
        None
    }
}

/// Synthetic cross mid from bid-safe and ask-safe leg composition.
///
/// For base/quote via USDT legs: mid ≈ sqrt-free ratio of safe legs is not used.
/// Instead:
/// - bid_cross = bid_base_usdt / ask_quote_usdt
/// - ask_cross = ask_base_usdt / bid_quote_usdt
pub fn synthetic_cross_mid(
    bid_base_quote_leg: f64,
    ask_base_quote_leg: f64,
    bid_quote_quote_leg: f64,
    ask_quote_quote_leg: f64,
) -> Option<(f64, f64, f64)> {
    if ![
        bid_base_quote_leg,
        ask_base_quote_leg,
        bid_quote_quote_leg,
        ask_quote_quote_leg,
    ]
    .iter()
    .all(|x| x.is_finite() && *x > 0.0)
    {
        return None;
    }
    let bid = bid_base_quote_leg / ask_quote_quote_leg;
    let ask = ask_base_quote_leg / bid_quote_quote_leg;
    if !(bid.is_finite() && ask.is_finite() && bid > 0.0 && ask > 0.0 && bid <= ask) {
        return None;
    }
    Some((bid, ask, 0.5 * (bid + ask)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ms_and_us_lines() {
        let ms = parse_aggtrade_line("1,2500.5,0.1,1,1,1720051200000,true,true", false)
            .unwrap()
            .unwrap();
        assert!((ms.timestamp_secs - 1_720_051_200.0).abs() < 1e-9);
        assert!((ms.price - 2500.5).abs() < 1e-12);

        let us = parse_aggtrade_line("1,0.05,1.0,1,1,1735689600000000,true,true", true)
            .unwrap()
            .unwrap();
        assert!((us.timestamp_secs - 1_735_689_600.0).abs() < 1e-9);
        assert!((us.price - 20.0).abs() < 1e-12);
    }

    #[test]
    fn native_and_synthetic_bbo() {
        assert_eq!(native_bbo_mid(100.0, 102.0), Some(101.0));
        assert_eq!(native_bbo_mid(102.0, 100.0), None);
        let (bid, ask, mid) = synthetic_cross_mid(2000.0, 2002.0, 1.0, 1.001).unwrap();
        assert!(bid < ask);
        assert!((mid - 0.5 * (bid + ask)).abs() < 1e-12);
    }
}
