//! Reconstruct the frozen [`Panel`] from normalized on-chain events, porting
//! `build_outcomes.py` v3's per-pool logic exactly so the Rust output can be validated
//! row-for-row against the Python golden via [`crate::data::panel::compare`].
//!
//! Parity-critical details mirrored from Python: week label is `strftime("%Y-%W")`;
//! `week_start` is the Monday 00:00 UTC of the containing week (which is why, at a
//! year-boundary week `00`, a segment can be bucketed under a label whose own week_start
//! sits in the prior year --- this is the reference behaviour, reproduced here); time-
//! weighted active liquidity splits intervals at week boundaries; depth is the mean of the
//! tick-book cumulative at `tick-band`, `tick`, `tick+band`; amounts are scaled by
//! per-token decimals. A pool's events are sorted internally by
//! `(block, tx_index, log_index)` (the Python SELECT's order), so callers need not
//! pre-sort; output rows are sorted by `(pool, week)` for a stable audit artifact.

use crate::data::book::Book;
use crate::data::panel::{Panel, PoolWeek, UnitRole};
use chrono::{Datelike, Duration, TimeZone, Utc};
use std::collections::{HashMap, HashSet};

const SECWK: i64 = 7 * 86400;

/// `strftime("%Y-%W")` of a UTC timestamp (Monday-first week, `00..53`).
pub fn week_id(ts: i64) -> String {
    Utc.timestamp_opt(ts, 0)
        .unwrap()
        .format("%Y-%W")
        .to_string()
}

/// Monday 00:00 UTC of the week containing `ts` (matches Python's
/// `strptime(f"{y} {w} 1", "%Y %W %w")`).
pub fn week_start(ts: i64) -> i64 {
    let d = Utc.timestamp_opt(ts, 0).unwrap().date_naive();
    let dfm = d.weekday().num_days_from_monday() as i64; // Mon=0
    (d - Duration::days(dfm))
        .and_hms_opt(0, 0, 0)
        .unwrap()
        .and_utc()
        .timestamp()
}

/// Price-band tick offset for `pct`: `floor(ln(1+pct)/ln(1.0001))`, as in `build_outcomes`.
fn band_ticks(pct: f64) -> i32 {
    ((1.0 + pct).ln() / 1.0001_f64.ln()) as i32
}

fn scale(x: &str, tok: &str, dec: &HashMap<String, u32>) -> f64 {
    // Python parses the raw amount as an arbitrary-precision int before dividing. i128
    // covers every realistic token amount exactly; only genuinely uint256-scale raws fall
    // back to a lossy f64 parse (flagged for the golden parity run on large amounts).
    let v = x
        .parse::<i128>()
        .map(|n| n as f64)
        .or_else(|_| x.parse::<f64>())
        .unwrap_or(0.0);
    v / 10f64.powi(*dec.get(tok).unwrap_or(&18) as i32)
}

/// One normalized event (subset of the raw schema needed for reconstruction).
///
/// `tx_index`/`log_index` are the intra-block ordering keys; reconstruction sorts on
/// `(block, tx_index, log_index)` so that JIT detection, book state, and depth-at-swap are
/// order-exact regardless of the order events are handed in.
#[derive(Clone, Debug)]
pub struct Event {
    pub ts: i64,
    pub block: i64,
    pub tx_index: i64,
    pub log_index: i64,
    pub kind: String, // swap | mint | burn | collect
    pub owner: String,
    pub tick_lower: Option<i32>,
    pub tick_upper: Option<i32>,
    pub liquidity_delta: i128,
    /// `None` when the raw `swap_liquidity` was empty/absent; Python carries the prior
    /// `cur_L` forward in that case rather than zeroing it.
    pub swap_liquidity: Option<i128>,
    pub amount0: String,
    pub amount1: String,
    pub tick: Option<i32>,
    pub token0: String,
    pub token1: String,
}

#[derive(Default)]
struct WeekAgg {
    twl_num: f64,
    twl_den: f64,
    swaps: i64,
    vol0: f64,
    vol1: f64,
    fee_income: f64,
    depth1: f64,
    depth2: f64,
    depth5: f64,
    lp_entry: i64,
    lp_exit: i64,
    mint_liq: i128,
    burn_liq: i128,
    jit_liq: i128,
    dur_sum: i64,
    dur_n: i64,
    collect_amt1: f64,
}

/// Reconstruct one pool's weekly rows from its `(block,tx_index,log_index)`-ordered events,
/// seeded by its pre-window tick book. Mirrors `build_outcomes.py` v3.
pub fn reconstruct_pool(
    pool: &str,
    role: UnitRole,
    tier: i64,
    events: &[Event],
    tickbook_seed: &HashMap<i32, i128>,
    dec: &HashMap<String, u32>,
) -> Vec<PoolWeek> {
    let bands = [
        ("1", band_ticks(0.01)),
        ("2", band_ticks(0.02)),
        ("5", band_ticks(0.05)),
    ];
    let mut book = Book::new();
    for (&t, &v) in tickbook_seed {
        book.apply(t, v);
    }
    let mut cur_l: i128 = 0;
    let mut cur_tick: Option<i32> = None;
    let mut last_ts: Option<i64> = None;
    let mut wk: HashMap<String, WeekAgg> = HashMap::new();
    type PosKey = (String, Option<i32>, Option<i32>);
    let mut open_pos: HashMap<PosKey, i64> = HashMap::new();
    let mut sameblk: HashMap<i64, HashMap<PosKey, i128>> = HashMap::new();
    let mut owners_wk: HashMap<String, HashSet<String>> = HashMap::new();

    // Enforce the Python SELECT's `ORDER BY block,tx_index,log_index`. Stable so ties (which
    // should not occur, log_index being unique within a block) keep their input order.
    let mut order: Vec<usize> = (0..events.len()).collect();
    order.sort_by_key(|&i| (events[i].block, events[i].tx_index, events[i].log_index));

    for &i in &order {
        let e = &events[i];
        // BO-1: split [last_ts, ts] across week boundaries, weighting cur_L
        if let Some(lt) = last_ts
            && cur_l > 0
            && e.ts > lt
        {
            let mut a = lt;
            while a < e.ts {
                let wend = week_start(a) + SECWK;
                let seg = e.ts.min(wend) - a;
                let w = wk.entry(week_id(a)).or_default();
                w.twl_num += cur_l as f64 * seg as f64;
                w.twl_den += seg as f64;
                a = e.ts.min(wend);
            }
        }
        last_ts = Some(e.ts);
        let wid = week_id(e.ts);
        match e.kind.as_str() {
            "swap" => {
                let v0 = scale(&e.amount0, &e.token0, dec).abs();
                let v1 = scale(&e.amount1, &e.token1, dec).abs();
                if let Some(l) = e.swap_liquidity {
                    cur_l = l; // else carry the prior cur_L forward (Python semantics)
                }
                if let Some(t) = e.tick {
                    cur_tick = Some(t);
                }
                let w = wk.entry(wid).or_default();
                w.swaps += 1;
                w.vol0 += v0;
                w.vol1 += v1;
                w.fee_income += v1 * tier as f64 / 1e6;
                if let Some(ct) = cur_tick {
                    let mut d = |off: i32| book.active_l(ct + off) as f64;
                    for (b, wd) in bands {
                        let val = (d(-wd) + d(0) + d(wd)) / 3.0;
                        match b {
                            "1" => w.depth1 = val,
                            "2" => w.depth2 = val,
                            _ => w.depth5 = val,
                        }
                    }
                }
            }
            "mint" | "burn" => {
                let l = e.liquidity_delta;
                let sgn: i128 = if e.kind == "mint" { 1 } else { -1 };
                if let (Some(tl), Some(tu)) = (e.tick_lower, e.tick_upper) {
                    book.apply(tl, sgn * l);
                    book.apply(tu, -sgn * l);
                }
                // Key on the raw Option ticks (Python uses (owner, None, None) for null
                // ticks) so malformed null-tick rows can't collide with a real (0,0) position.
                let key = (e.owner.clone(), e.tick_lower, e.tick_upper);
                owners_wk
                    .entry(wid.clone())
                    .or_default()
                    .insert(e.owner.clone());
                let w = wk.entry(wid).or_default();
                if e.kind == "mint" {
                    w.lp_entry += 1;
                    w.mint_liq += l;
                    open_pos.insert(key.clone(), e.ts);
                    *sameblk.entry(e.block).or_default().entry(key).or_insert(0) += l;
                } else {
                    w.lp_exit += 1;
                    w.burn_liq += l;
                    if let Some(t0) = open_pos.remove(&key) {
                        w.dur_sum += e.ts - t0;
                        w.dur_n += 1;
                    }
                    if let Some(m) = sameblk.get(&e.block).and_then(|b| b.get(&key)) {
                        w.jit_liq += l.min(*m);
                    }
                }
            }
            "collect" => {
                let w = wk.entry(wid).or_default();
                w.collect_amt1 += scale(&e.amount1, &e.token1, dec).abs();
            }
            _ => {}
        }
    }

    let mut out = Vec::with_capacity(wk.len());
    for (week, a) in wk {
        let twl = if a.twl_den > 0.0 {
            a.twl_num / a.twl_den
        } else {
            0.0
        };
        let fee = a.fee_income;
        let uniq = owners_wk.get(&week).map_or(0, |s| s.len()) as i64;
        out.push(PoolWeek {
            pool: pool.to_string(),
            unit_role: role,
            week,
            swaps: a.swaps,
            vol0: round(a.vol0, 6),
            vol1: round(a.vol1, 6),
            twl_active_liquidity: round(twl, 2),
            depth_1pct: round(a.depth1, 2),
            depth_2pct: round(a.depth2, 2),
            depth_5pct: round(a.depth5, 2),
            lp_entry_count: a.lp_entry,
            lp_exit_count: a.lp_exit,
            unique_lp_count: uniq,
            jit_share_same_block: if a.mint_liq > 0 {
                round(a.jit_liq as f64 / a.mint_liq as f64, 4)
            } else {
                0.0
            },
            lp_fee_income_native1: round(fee, 6),
            lp_fee_income_per_active_liquidity: if twl > 0.0 { round(fee / twl, 12) } else { 0.0 },
            collect_amount1_native: round(a.collect_amt1, 6),
            position_duration_days: if a.dur_n > 0 {
                round(a.dur_sum as f64 / a.dur_n as f64 / 86400.0, 2)
            } else {
                0.0
            },
            net_liq: a.mint_liq - a.burn_liq,
        });
    }
    // Deterministic order (HashMap iteration is not) so the CSV audit artifact is stable.
    out.sort_by(|x, y| x.week.cmp(&y.week));
    out
}

/// Round half-away-from-zero to `nd` decimals (Python `round` is banker's; the panel
/// tolerances in [`crate::data::panel::compare`] absorb the sub-ULP difference).
fn round(x: f64, nd: i32) -> f64 {
    let f = 10f64.powi(nd);
    (x * f).round() / f
}

/// Reconstruct the full [`Panel`] from events grouped by pool (each group ordered by
/// `(block,tx_index,log_index)`). `meta` gives (role, tier) per pool; `tickbook` the
/// pre-window seed; `dec` the token decimals.
pub fn reconstruct(
    pools: &[(String, UnitRole, i64, Vec<Event>)],
    tickbook: &HashMap<String, HashMap<i32, i128>>,
    dec: &HashMap<String, u32>,
) -> Panel {
    let empty = HashMap::new();
    let mut rows = Vec::new();
    for (pool, role, tier, events) in pools {
        let seed = tickbook.get(pool).unwrap_or(&empty);
        rows.extend(reconstruct_pool(pool, *role, *tier, events, seed, dec));
    }
    rows.sort_by(|x, y| x.pool.cmp(&y.pool).then(x.week.cmp(&y.week)));
    Panel { rows }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn week_functions_match_python_oracle() {
        // (ts, week_id, week_start) generated from build_outcomes' Python week functions
        let cases = [
            (1750248000i64, "2025-24", 1750032000i64),
            (1704067200, "2024-01", 1704067200),
            (1735689600, "2025-00", 1735516800), // year-boundary week 00
            (1767225540, "2025-52", 1766966400),
            (1767225600, "2026-00", 1766966400),
            (1782863940, "2026-26", 1782691200),
        ];
        for (ts, wid, ws) in cases {
            assert_eq!(week_id(ts), wid, "week_id({ts})");
            assert_eq!(week_start(ts), ws, "week_start({ts})");
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn ev(
        ts: i64,
        block: i64,
        kind: &str,
        owner: &str,
        tl: Option<i32>,
        tu: Option<i32>,
        ld: i128,
        swl: Option<i128>,
        a1: &str,
        tick: Option<i32>,
    ) -> Event {
        Event {
            ts,
            block,
            tx_index: 0,
            log_index: 0,
            kind: kind.into(),
            owner: owner.into(),
            tick_lower: tl,
            tick_upper: tu,
            liquidity_delta: ld,
            swap_liquidity: swl,
            amount0: "0".into(),
            amount1: a1.into(),
            tick,
            token0: "t0".into(),
            token1: "t1".into(),
        }
    }

    #[test]
    fn reconstruct_pool_basic_outcomes() {
        // one position [-100,100] L=1000 minted, then swaps in range across one week
        let dec: HashMap<String, u32> = [("t1".to_string(), 6u32)].into_iter().collect();
        let seed = HashMap::new();
        let base = 1750000000i64; // mid-2025
        // position wide enough that the +/-2% band (~198 ticks) stays in range at tick 0
        let evs = vec![
            ev(
                base,
                1,
                "mint",
                "alice",
                Some(-300),
                Some(300),
                1000,
                None,
                "0",
                None,
            ),
            ev(
                base + 3600,
                2,
                "swap",
                "x",
                None,
                None,
                0,
                Some(1000),
                "1000000",
                Some(0),
            ), // vol1=1.0 (6 dec)
            ev(
                base + 7200,
                3,
                "swap",
                "x",
                None,
                None,
                0,
                Some(1000),
                "2000000",
                Some(0),
            ), // vol1=2.0
        ];
        let rows = reconstruct_pool("p", UnitRole::MatchedTreated, 3000, &evs, &seed, &dec);
        assert_eq!(rows.len(), 1);
        let r = &rows[0];
        assert_eq!(r.swaps, 2);
        assert!((r.vol1 - 3.0).abs() < 1e-9, "vol1={}", r.vol1);
        assert_eq!(r.lp_entry_count, 1);
        assert_eq!(r.unique_lp_count, 1);
        assert_eq!(r.net_liq, 1000);
        // depth at tick 0 with the [-100,100] position in range = 1000 (mean of in-range points)
        assert!(
            (r.depth_2pct - 1000.0).abs() < 1e-6,
            "depth2={}",
            r.depth_2pct
        );
        // fee income = vol1 * tier/1e6 = 3.0 * 3000/1e6 = 0.009
        assert!(
            (r.lp_fee_income_native1 - 0.009).abs() < 1e-9,
            "fee={}",
            r.lp_fee_income_native1
        );
    }

    #[test]
    fn jit_same_block_detected() {
        let dec: HashMap<String, u32> = HashMap::new();
        let seed = HashMap::new();
        let base = 1750000000i64;
        // mint then burn same block, same position => JIT
        let evs = vec![
            ev(
                base,
                5,
                "mint",
                "bot",
                Some(-10),
                Some(10),
                500,
                None,
                "0",
                None,
            ),
            ev(
                base,
                5,
                "burn",
                "bot",
                Some(-10),
                Some(10),
                500,
                None,
                "0",
                None,
            ),
        ];
        let rows = reconstruct_pool("p", UnitRole::MatchedTreated, 3000, &evs, &seed, &dec);
        let r = &rows[0];
        assert_eq!(r.lp_entry_count, 1);
        assert_eq!(r.lp_exit_count, 1);
        assert!(
            (r.jit_share_same_block - 1.0).abs() < 1e-9,
            "jit={}",
            r.jit_share_same_block
        );
        assert_eq!(r.net_liq, 0);
    }

    #[test]
    fn swap_missing_liquidity_carries_forward() {
        // A swap with swap_liquidity=None must keep the prior cur_L for the TWL weighting,
        // not zero it (Python: `cur_L=int(swl) if swl not in ("",None) else cur_L`).
        let dec: HashMap<String, u32> = HashMap::new();
        let seed = HashMap::new();
        let base = 1750000000i64;
        let evs = vec![
            ev(
                base,
                1,
                "swap",
                "x",
                None,
                None,
                0,
                Some(1000),
                "0",
                Some(0),
            ),
            ev(
                base + 3600,
                2,
                "swap",
                "x",
                None,
                None,
                0,
                None,
                "0",
                Some(0),
            ), // carry 1000
            ev(
                base + 7200,
                3,
                "swap",
                "x",
                None,
                None,
                0,
                None,
                "0",
                Some(0),
            ),
        ];
        let rows = reconstruct_pool("p", UnitRole::MatchedTreated, 3000, &evs, &seed, &dec);
        // TWL is time-weighted cur_L over the two 3600s inter-swap gaps, both at cur_L=1000.
        assert!(
            (rows[0].twl_active_liquidity - 1000.0).abs() < 1e-6,
            "twl={}",
            rows[0].twl_active_liquidity
        );
    }

    #[test]
    fn intra_block_order_is_enforced() {
        // Events handed in reversed intra-block order must still reconstruct as if sorted by
        // (block, tx_index, log_index): the mint (log 0) precedes the burn (log 1), so the
        // same-block pair is a JIT round-trip.
        let dec: HashMap<String, u32> = HashMap::new();
        let seed = HashMap::new();
        let base = 1750000000i64;
        let mint = Event {
            tx_index: 0,
            log_index: 0,
            ..ev(
                base,
                9,
                "mint",
                "bot",
                Some(-10),
                Some(10),
                500,
                None,
                "0",
                None,
            )
        };
        let burn = Event {
            tx_index: 0,
            log_index: 1,
            ..ev(
                base,
                9,
                "burn",
                "bot",
                Some(-10),
                Some(10),
                500,
                None,
                "0",
                None,
            )
        };
        let evs = vec![burn, mint]; // reversed
        let rows = reconstruct_pool("p", UnitRole::MatchedTreated, 3000, &evs, &seed, &dec);
        let r = &rows[0];
        assert_eq!(r.lp_entry_count, 1);
        assert_eq!(r.lp_exit_count, 1);
        assert!(
            (r.jit_share_same_block - 1.0).abs() < 1e-9,
            "jit={}",
            r.jit_share_same_block
        );
    }
}
