//! Deterministic candidate / adjacent-contrast / sharing logic (no network).

use crate::pstt::error::{PsttError, Result};
use crate::pstt::orientation::choose_base_symbol;
use crate::pstt::schema::{Address, ContrastRecord, PoolRecord};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const UNISWAP_V3_FACTORY: &str = "0x1f98431c8ad98523631ae4a59f267346ea31f984";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolMeta {
    pub pool: Address,
    pub factory: Address,
    pub fee: u32,
    pub token0: Address,
    pub token1: Address,
    pub token0_symbol: String,
    pub token1_symbol: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateSlot {
    pub pair: String,
    pub pair_id: String,
    pub fee: u32,
    pub pool: Address,
    pub base: String,
    pub token0: String,
    pub token1: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdjacentContrast {
    pub pair: String,
    pub pair_id: String,
    pub lower_fee: u32,
    pub higher_fee: u32,
    pub pool_lower: Address,
    pub pool_higher: Address,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedPoolNote {
    pub pool: Address,
    pub contrast_ids: Vec<String>,
}

/// Canonical-factory filter and exact one-pool-per-(pair,fee) validation.
pub fn assemble_candidates(
    pools: &[PoolMeta],
    stables: &[&str],
) -> Result<(Vec<CandidateSlot>, Vec<AdjacentContrast>)> {
    let mut by_pair: BTreeMap<(String, String), Vec<&PoolMeta>> = BTreeMap::new();
    for p in pools {
        if p.factory.as_str() != UNISWAP_V3_FACTORY {
            continue;
        }
        let mut key = [p.token0.as_str().to_string(), p.token1.as_str().to_string()];
        key.sort();
        by_pair
            .entry((key[0].clone(), key[1].clone()))
            .or_default()
            .push(p);
    }

    let mut candidates = Vec::new();
    let mut contrasts = Vec::new();
    for ((a0, a1), entries) in &by_pair {
        let fees: BTreeSet<u32> = entries.iter().map(|e| e.fee).collect();
        if fees.len() < 2 {
            continue;
        }
        let mut by_fee: BTreeMap<u32, &PoolMeta> = BTreeMap::new();
        for e in entries {
            if by_fee.insert(e.fee, *e).is_some() {
                return Err(PsttError::invariant(format!(
                    "residual duplicate pair/fee slot: {a0}/{a1} fee={}",
                    e.fee
                )));
            }
        }
        let any = *by_fee.values().next().unwrap();
        let base = choose_base_symbol(&any.token0_symbol, &any.token1_symbol, stables);
        let pname = {
            let mut syms = [any.token0_symbol.clone(), any.token1_symbol.clone()];
            syms.sort();
            format!("{}/{}", syms[0], syms[1])
        };
        let pair_id = format!("{a0}/{a1}");
        let tiers: Vec<u32> = by_fee.keys().copied().collect();
        for fee in &tiers {
            let e = by_fee[fee];
            candidates.push(CandidateSlot {
                pair: pname.clone(),
                pair_id: pair_id.clone(),
                fee: *fee,
                pool: e.pool.clone(),
                base: base.clone(),
                token0: e.token0_symbol.clone(),
                token1: e.token1_symbol.clone(),
            });
        }
        for w in tiers.windows(2) {
            contrasts.push(AdjacentContrast {
                pair: pname.clone(),
                pair_id: pair_id.clone(),
                lower_fee: w[0],
                higher_fee: w[1],
                pool_lower: by_fee[&w[0]].pool.clone(),
                pool_higher: by_fee[&w[1]].pool.clone(),
            });
        }
    }
    candidates.sort_by(|a, b| {
        (&a.pair_id, a.fee, a.pool.as_str()).cmp(&(&b.pair_id, b.fee, b.pool.as_str()))
    });
    contrasts.sort_by(|a, b| {
        (&a.pair_id, a.lower_fee, a.higher_fee).cmp(&(&b.pair_id, b.lower_fee, b.higher_fee))
    });
    Ok((candidates, contrasts))
}

pub fn contrast_id(pair: &str, lower_fee: u32, higher_fee: u32) -> String {
    format!("{pair}:{lower_fee}-{higher_fee}")
}

pub fn shared_pool_notes(contrasts: &[ContrastRecord]) -> Vec<SharedPoolNote> {
    let mut map: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for c in contrasts {
        map.entry(c.pool_lower.as_str().to_string())
            .or_default()
            .insert(c.contrast_id.clone());
        map.entry(c.pool_higher.as_str().to_string())
            .or_default()
            .insert(c.contrast_id.clone());
    }
    let mut out = Vec::new();
    for (pool, ids) in map {
        if ids.len() > 1 {
            out.push(SharedPoolNote {
                pool: Address(pool),
                contrast_ids: ids.into_iter().collect(),
            });
        }
    }
    out
}

/// Extract eligible swap blocks in `[start, end)`.
pub fn extract_eligible_blocks(
    swaps: impl Iterator<Item = (Address, u64, i64, String)>,
    wanted_pools: &BTreeSet<String>,
    start: i64,
    end: i64,
) -> Result<BTreeMap<String, BTreeSet<u64>>> {
    let mut by_pool: BTreeMap<String, BTreeSet<u64>> = BTreeMap::new();
    for (pool, block, ts, ty) in swaps {
        if !wanted_pools.contains(pool.as_str()) || ty != "swap" {
            continue;
        }
        if start <= ts && ts < end {
            by_pool.entry(pool.0).or_default().insert(block);
        }
    }
    if by_pool.keys().cloned().collect::<BTreeSet<_>>() != *wanted_pools {
        let missing: Vec<_> = wanted_pools
            .difference(&by_pool.keys().cloned().collect())
            .cloned()
            .collect();
        return Err(PsttError::invariant(format!(
            "pools without eligible swaps: {missing:?}"
        )));
    }
    Ok(by_pool)
}

pub fn union_sorted_blocks(by_pool: &BTreeMap<String, BTreeSet<u64>>) -> Vec<u64> {
    let mut union = BTreeSet::new();
    for blocks in by_pool.values() {
        union.extend(blocks.iter().copied());
    }
    union.into_iter().collect()
}

pub fn pool_records_sorted(pools: Vec<PoolRecord>) -> Vec<PoolRecord> {
    let mut pools = pools;
    pools.sort_by(|a, b| (&a.pair, a.fee, a.pool.as_str()).cmp(&(&b.pair, b.fee, b.pool.as_str())));
    pools
}

/// Documented v1 defect: `setdefault`-style first-writer-wins on display pair
/// can shadow a later canonical factory pool. This helper reproduces the bad
/// behavior for a negative fixture; production code must use address keys.
pub fn defective_v1_shadow_by_display_pair(
    rows: &[(String, String, u32)],
) -> BTreeMap<(String, u32), String> {
    let mut out = BTreeMap::new();
    for (display_pair, pool, fee) in rows {
        out.entry((display_pair.clone(), *fee))
            .or_insert_with(|| pool.clone());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(pool: &str, fee: u32, t0: &str, t1: &str, s0: &str, s1: &str) -> PoolMeta {
        PoolMeta {
            pool: Address::normalize(pool).unwrap(),
            factory: Address::normalize(UNISWAP_V3_FACTORY).unwrap(),
            fee,
            token0: Address::normalize(t0).unwrap(),
            token1: Address::normalize(t1).unwrap(),
            token0_symbol: s0.into(),
            token1_symbol: s1.into(),
        }
    }

    #[test]
    fn adjacent_contrasts_and_shared_pool() {
        let usdc = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48";
        let weth = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2";
        let p100 = "0x0000000000000000000000000000000000000001";
        let p500 = "0x0000000000000000000000000000000000000002";
        let p3000 = "0x0000000000000000000000000000000000000003";
        let pools = vec![
            meta(p100, 100, usdc, weth, "USDC", "WETH"),
            meta(p500, 500, usdc, weth, "USDC", "WETH"),
            meta(p3000, 3000, usdc, weth, "USDC", "WETH"),
        ];
        let (cands, contrasts) = assemble_candidates(&pools, &["USDC", "USDT", "DAI"]).unwrap();
        assert_eq!(cands.len(), 3);
        assert_eq!(contrasts.len(), 2);
        assert_eq!(contrasts[0].lower_fee, 100);
        assert_eq!(contrasts[0].higher_fee, 500);
        let records: Vec<ContrastRecord> = contrasts
            .iter()
            .map(|c| ContrastRecord {
                contrast_id: contrast_id(&c.pair, c.lower_fee, c.higher_fee),
                pair: c.pair.clone(),
                base: "WETH".into(),
                quote: "USDC".into(),
                cex_symbol: "ETHUSDC".into(),
                invert: false,
                lower_fee: c.lower_fee,
                higher_fee: c.higher_fee,
                pool_lower: c.pool_lower.clone(),
                pool_higher: c.pool_higher.clone(),
            })
            .collect();
        let notes = shared_pool_notes(&records);
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].pool.as_str(), p500);
        assert_eq!(notes[0].contrast_ids.len(), 2);
    }

    #[test]
    fn v1_shadow_defect_negative_fixture() {
        let shadowed = defective_v1_shadow_by_display_pair(&[
            ("USDC/WETH".into(), "fork".into(), 500),
            ("USDC/WETH".into(), "canonical".into(), 500),
        ]);
        assert_eq!(
            shadowed.get(&("USDC/WETH".into(), 500)).map(String::as_str),
            Some("fork")
        );
    }
}
