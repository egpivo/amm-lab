//! Strict as-of joins. Timestamp joins and chain-order joins are separate APIs.

use crate::pstt::error::{PsttError, Result};
use crate::pstt::schema::AggTrade;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinMissReason {
    NoPriorObservation,
    EmptyReference,
}

#[derive(Debug, Clone, Copy)]
pub struct TimestampJoin {
    pub price: f64,
    pub reference_timestamp: f64,
    pub staleness_seconds: f64,
    pub index: usize,
}

/// Largest reference timestamp strictly less than block time `t`.
/// Equal-time observations are excluded.
pub fn strict_pre_timestamp(
    sorted_trades: &[AggTrade],
    t: f64,
) -> std::result::Result<TimestampJoin, JoinMissReason> {
    if sorted_trades.is_empty() {
        return Err(JoinMissReason::EmptyReference);
    }
    let idx = sorted_trades.partition_point(|row| row.timestamp_secs < t);
    if idx == 0 {
        return Err(JoinMissReason::NoPriorObservation);
    }
    let i = idx - 1;
    let row = &sorted_trades[i];
    Ok(TimestampJoin {
        price: row.price,
        reference_timestamp: row.timestamp_secs,
        staleness_seconds: t - row.timestamp_secs,
        index: i,
    })
}

#[derive(Debug, Clone, Copy)]
pub struct VwapJoin {
    pub price: f64,
    pub latest_trade_timestamp: f64,
    pub staleness_seconds: f64,
    pub quantity: f64,
}

/// Strict `[t-1s, t)` quantity-weighted VWAP.
pub fn vwap_1s(
    sorted_trades: &[AggTrade],
    t: f64,
) -> std::result::Result<VwapJoin, JoinMissReason> {
    if sorted_trades.is_empty() {
        return Err(JoinMissReason::EmptyReference);
    }
    let lo = sorted_trades.partition_point(|row| row.timestamp_secs < t - 1.0);
    let hi = sorted_trades.partition_point(|row| row.timestamp_secs < t);
    if hi <= lo {
        return Err(JoinMissReason::NoPriorObservation);
    }
    let mut pq = 0.0;
    let mut q = 0.0;
    for row in &sorted_trades[lo..hi] {
        pq += row.price * row.quantity;
        q += row.quantity;
    }
    if q <= 0.0 {
        return Err(JoinMissReason::NoPriorObservation);
    }
    let latest = sorted_trades[hi - 1].timestamp_secs;
    Ok(VwapJoin {
        price: pq / q,
        latest_trade_timestamp: latest,
        staleness_seconds: t - latest,
        quantity: q,
    })
}

/// Chain-order key: `(block, tx_index, log_index)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ChainKey {
    pub block: u64,
    pub tx_index: u32,
    pub log_index: u32,
}

/// Largest chain key strictly less than `target`, optionally excluding the
/// same transaction (`tx_index` equal and same block).
pub fn strict_pre_chain(
    sorted_keys: &[ChainKey],
    target: ChainKey,
    exclude_same_tx: bool,
) -> Result<Option<usize>> {
    if sorted_keys.windows(2).any(|w| w[0] > w[1]) {
        return Err(PsttError::invariant(
            "chain keys must be sorted ascending before join",
        ));
    }
    let idx = sorted_keys.partition_point(|k| *k < target);
    if idx == 0 {
        return Ok(None);
    }
    let mut i = idx - 1;
    if exclude_same_tx {
        while sorted_keys[i].block == target.block && sorted_keys[i].tx_index == target.tx_index {
            if i == 0 {
                return Ok(None);
            }
            i -= 1;
        }
    }
    Ok(Some(i))
}

/// Cache one timestamp reference per block so every fill in the block shares it.
#[derive(Debug, Default)]
pub struct BlockReferenceCache {
    last_block: Option<u64>,
    last_join: Option<TimestampJoin>,
}

impl BlockReferenceCache {
    pub fn get_or_join(
        &mut self,
        block: u64,
        t: f64,
        sorted_trades: &[AggTrade],
    ) -> std::result::Result<TimestampJoin, JoinMissReason> {
        if self.last_block == Some(block) {
            return self.last_join.ok_or(JoinMissReason::NoPriorObservation);
        }
        let join = strict_pre_timestamp(sorted_trades, t)?;
        self.last_block = Some(block);
        self.last_join = Some(join);
        Ok(join)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trades(ts: &[(f64, f64)]) -> Vec<AggTrade> {
        ts.iter()
            .map(|(t, p)| AggTrade {
                timestamp_secs: *t,
                price: *p,
                quantity: 1.0,
            })
            .collect()
    }

    #[test]
    fn equal_time_excluded() {
        let rows = trades(&[(100.0, 1.0), (200.0, 2.0)]);
        let j = strict_pre_timestamp(&rows, 200.0).unwrap();
        assert_eq!(j.index, 0);
        assert!((j.price - 1.0).abs() < 1e-12);
        assert!(strict_pre_timestamp(&rows, 100.0).is_err());
    }

    #[test]
    fn vwap_half_open_window() {
        let rows = vec![
            AggTrade {
                timestamp_secs: 98.0,
                price: 10.0,
                quantity: 1.0,
            },
            AggTrade {
                timestamp_secs: 99.0,
                price: 20.0,
                quantity: 1.0,
            },
            AggTrade {
                timestamp_secs: 100.0,
                price: 30.0,
                quantity: 1.0,
            },
        ];
        let j = vwap_1s(&rows, 100.0).unwrap();
        assert!((j.price - 20.0).abs() < 1e-12);
        assert!((j.quantity - 1.0).abs() < 1e-12);
    }

    #[test]
    fn same_block_shares_cached_reference() {
        let rows = trades(&[(100.0, 5.0), (150.0, 6.0)]);
        let mut cache = BlockReferenceCache::default();
        let a = cache.get_or_join(10, 160.0, &rows).unwrap();
        let b = cache.get_or_join(10, 160.0, &rows).unwrap();
        assert_eq!(a.index, b.index);
        assert!((a.price - 6.0).abs() < 1e-12);
    }
}
