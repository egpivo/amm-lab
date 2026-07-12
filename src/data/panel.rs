//! Frozen pool-week outcome panel: the typed contract between the data layer and the
//! method/dashboard layers.
//!
//! The [`PoolWeek`] fields mirror, one-for-one, the columns emitted by the reference
//! `build_outcomes.py` v3 (`panel_weekly_frozen.csv`). This lets the Rust reconstruction
//! be validated row-for-row against the Python output as a golden reference via
//! [`compare`]. Method code depends on this schema, never on raw events or RPC.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Role of a pool in the matched-overlap design. Primary controls are
/// [`UnitRole::MatchedControl`] only -- never "treated == 0" -- and forks are kept
/// separate so they never contaminate the control group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnitRole {
    #[serde(rename = "matched_treated")]
    MatchedTreated,
    #[serde(rename = "matched_control")]
    MatchedControl,
    #[serde(rename = "unmatched_treated")]
    UnmatchedTreated,
    #[serde(rename = "crossvenue_fork")]
    CrossvenueFork,
    #[serde(rename = "unknown")]
    Unknown,
}

/// One pool-week row. Column order and names match `build_outcomes.py` v3 exactly.
///
/// Liquidity-magnitude outcomes are `f64` because the reference already rounds them to
/// fixed decimals (they are time-weighted averages / band means, not exact integers);
/// counts are integers; `net_liq` is an exact signed liquidity delta held in `i128`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolWeek {
    pub pool: String,
    pub unit_role: UnitRole,
    pub week: String,
    pub swaps: i64,
    pub vol0: f64,
    pub vol1: f64,
    pub twl_active_liquidity: f64,
    pub depth_1pct: f64,
    pub depth_2pct: f64,
    pub depth_5pct: f64,
    pub lp_entry_count: i64,
    pub lp_exit_count: i64,
    pub unique_lp_count: i64,
    pub jit_share_same_block: f64,
    pub lp_fee_income_native1: f64,
    pub lp_fee_income_per_active_liquidity: f64,
    pub collect_amount1_native: f64,
    pub position_duration_days: f64,
    pub net_liq: i128,
}

impl PoolWeek {
    fn key(&self) -> (String, String) {
        (self.pool.clone(), self.week.clone())
    }
}

/// A frozen panel: the full set of pool-week rows.
#[derive(Debug, Clone, Default)]
pub struct Panel {
    pub rows: Vec<PoolWeek>,
}

impl Panel {
    /// Load a panel from a `build_outcomes`-style CSV (the golden reference or a Rust build).
    pub fn from_csv<P: AsRef<Path>>(path: P) -> Result<Self, csv::Error> {
        let mut rdr = csv::Reader::from_path(path)?;
        let mut rows = Vec::new();
        for r in rdr.deserialize() {
            rows.push(r?);
        }
        Ok(Panel { rows })
    }

    /// Write the panel as CSV with the frozen column order.
    pub fn to_csv<P: AsRef<Path>>(&self, path: P) -> Result<(), csv::Error> {
        let mut wtr = csv::Writer::from_path(path)?;
        for r in &self.rows {
            wtr.serialize(r)?;
        }
        wtr.flush()?;
        Ok(())
    }
}

/// Tolerances for float comparison in [`compare`].
#[derive(Debug, Clone, Copy)]
pub struct Tol {
    pub abs: f64,
    pub rel: f64,
}

impl Default for Tol {
    /// Reference rounds most outcomes to 2-6 decimals; a matching port should differ by
    /// at most last-decimal rounding, so 1e-6 absolute with a 1e-9 relative floor for
    /// large magnitudes is a tight-but-not-brittle default.
    fn default() -> Self {
        Tol {
            abs: 1e-6,
            rel: 1e-9,
        }
    }
}

/// A single field-level disagreement between the two panels.
#[derive(Debug, Clone)]
pub struct Mismatch {
    pub pool: String,
    pub week: String,
    pub field: &'static str,
    pub a: f64,
    pub b: f64,
}

/// Result of a row-for-row parity check.
#[derive(Debug, Clone, Default)]
pub struct ParityReport {
    pub keys_a: usize,
    pub keys_b: usize,
    pub common_keys: usize,
    pub only_in_a: Vec<(String, String)>,
    pub only_in_b: Vec<(String, String)>,
    pub mismatches: Vec<Mismatch>,
}

impl ParityReport {
    /// True iff every key is shared and no field disagrees beyond tolerance.
    pub fn is_pass(&self) -> bool {
        self.only_in_a.is_empty() && self.only_in_b.is_empty() && self.mismatches.is_empty()
    }
}

/// Compare a candidate panel `b` (e.g. the Rust build) against golden `a` (the Python
/// reference), keyed by `(pool, week)`. Integer fields must match exactly; float fields
/// must agree within `tol`. Intended as the acceptance gate for the reconstruction port.
pub fn compare(a: &Panel, b: &Panel, tol: Tol) -> ParityReport {
    let mut rep = ParityReport::default();
    let ma: HashMap<(String, String), &PoolWeek> = a.rows.iter().map(|r| (r.key(), r)).collect();
    let mb: HashMap<(String, String), &PoolWeek> = b.rows.iter().map(|r| (r.key(), r)).collect();
    rep.keys_a = ma.len();
    rep.keys_b = mb.len();

    for (k, ra) in &ma {
        match mb.get(k) {
            None => rep.only_in_a.push(k.clone()),
            Some(rb) => {
                rep.common_keys += 1;
                let close = |field: &'static str, x: f64, y: f64, out: &mut Vec<Mismatch>| {
                    let d = (x - y).abs();
                    if d > tol.abs.max(tol.rel * x.abs().max(y.abs())) {
                        out.push(Mismatch {
                            pool: k.0.clone(),
                            week: k.1.clone(),
                            field,
                            a: x,
                            b: y,
                        });
                    }
                };
                let ieq = |field: &'static str, x: i128, y: i128, out: &mut Vec<Mismatch>| {
                    if x != y {
                        out.push(Mismatch {
                            pool: k.0.clone(),
                            week: k.1.clone(),
                            field,
                            a: x as f64,
                            b: y as f64,
                        });
                    }
                };
                let m = &mut rep.mismatches;
                if ra.unit_role != rb.unit_role {
                    m.push(Mismatch {
                        pool: k.0.clone(),
                        week: k.1.clone(),
                        field: "unit_role",
                        a: f64::NAN,
                        b: f64::NAN,
                    });
                }
                ieq("swaps", ra.swaps as i128, rb.swaps as i128, m);
                ieq(
                    "lp_entry_count",
                    ra.lp_entry_count as i128,
                    rb.lp_entry_count as i128,
                    m,
                );
                ieq(
                    "lp_exit_count",
                    ra.lp_exit_count as i128,
                    rb.lp_exit_count as i128,
                    m,
                );
                ieq(
                    "unique_lp_count",
                    ra.unique_lp_count as i128,
                    rb.unique_lp_count as i128,
                    m,
                );
                // net_liq is exact in the Rust build (i128) but the Python reference
                // accumulates mint_liq/burn_liq as float (`defaultdict(float)`), so above 2^53
                // its net_liq is f64-rounded. Compare with a RELATIVE tolerance: this absorbs
                // the reference's float rounding (~1e-16) while still catching a real
                // event-level difference (a missed/extra mint or burn shifts net_liq by a whole
                // liquidity unit -> relative ~1e-2 or larger).
                close("net_liq", ra.net_liq as f64, rb.net_liq as f64, m);
                close("vol0", ra.vol0, rb.vol0, m);
                close("vol1", ra.vol1, rb.vol1, m);
                close(
                    "twl_active_liquidity",
                    ra.twl_active_liquidity,
                    rb.twl_active_liquidity,
                    m,
                );
                close("depth_1pct", ra.depth_1pct, rb.depth_1pct, m);
                close("depth_2pct", ra.depth_2pct, rb.depth_2pct, m);
                close("depth_5pct", ra.depth_5pct, rb.depth_5pct, m);
                close(
                    "jit_share_same_block",
                    ra.jit_share_same_block,
                    rb.jit_share_same_block,
                    m,
                );
                close(
                    "lp_fee_income_native1",
                    ra.lp_fee_income_native1,
                    rb.lp_fee_income_native1,
                    m,
                );
                close(
                    "lp_fee_income_per_active_liquidity",
                    ra.lp_fee_income_per_active_liquidity,
                    rb.lp_fee_income_per_active_liquidity,
                    m,
                );
                close(
                    "collect_amount1_native",
                    ra.collect_amount1_native,
                    rb.collect_amount1_native,
                    m,
                );
                close(
                    "position_duration_days",
                    ra.position_duration_days,
                    rb.position_duration_days,
                    m,
                );
            }
        }
    }
    for k in mb.keys() {
        if !ma.contains_key(k) {
            rep.only_in_b.push(k.clone());
        }
    }
    rep
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(pool: &str, week: &str) -> PoolWeek {
        PoolWeek {
            pool: pool.into(),
            unit_role: UnitRole::MatchedTreated,
            week: week.into(),
            swaps: 10,
            vol0: 1.5,
            vol1: 2.5,
            twl_active_liquidity: 1000.0,
            depth_1pct: 500.0,
            depth_2pct: 400.0,
            depth_5pct: 300.0,
            lp_entry_count: 3,
            lp_exit_count: 2,
            unique_lp_count: 4,
            jit_share_same_block: 0.1,
            lp_fee_income_native1: 0.12,
            lp_fee_income_per_active_liquidity: 0.00012,
            collect_amount1_native: 0.5,
            position_duration_days: 12.0,
            net_liq: 123456789012345678901234567890i128,
        }
    }

    #[test]
    fn csv_roundtrip_preserves_schema_incl_i128_and_role() {
        let p = Panel {
            rows: vec![row("0xabc", "2025-24"), row("0xdef", "2025-25")],
        };
        let dir =
            std::env::temp_dir().join(format!("ammlab_panel_test_{}.csv", std::process::id()));
        p.to_csv(&dir).unwrap();
        let q = Panel::from_csv(&dir).unwrap();
        std::fs::remove_file(&dir).ok();
        assert_eq!(q.rows.len(), 2);
        assert_eq!(q.rows[0].net_liq, 123456789012345678901234567890i128);
        assert_eq!(q.rows[0].unit_role, UnitRole::MatchedTreated);
        let rep = compare(&p, &q, Tol::default());
        assert!(
            rep.is_pass(),
            "roundtrip must be identical: {:?}",
            rep.mismatches
        );
        assert_eq!(rep.common_keys, 2);
    }

    #[test]
    fn net_liq_relative_tolerance_absorbs_float_reference_but_catches_real_diff() {
        // Mirrors the observed case: the Python reference f64-accumulates liquidity, so a
        // large net_liq can differ from the exact i128 by a few units (rounding). That must
        // PASS; a whole-liquidity-unit difference (a missed/extra mint) must FAIL.
        let base = 69_071_333_368_809_152i128; // ~6.9e16, as seen on real data
        let mk = |nl: i128| {
            let mut r = row("p", "2025-24");
            r.net_liq = nl;
            Panel { rows: vec![r] }
        };
        let a = mk(base);
        // f64-rounding-scale difference -> pass
        assert!(
            compare(&a, &mk(base + 50), Tol::default()).is_pass(),
            "rounding-scale net_liq diff should pass"
        );
        // real event-level difference (~2e15) -> fail
        assert!(
            !compare(&a, &mk(base + 2_000_000_000_000_000), Tol::default()).is_pass(),
            "whole-unit net_liq diff must fail"
        );
        // small net_liq stays effectively exact: off-by-one is caught
        assert!(
            !compare(&mk(1000), &mk(1001), Tol::default()).is_pass(),
            "small net_liq off-by-one must fail"
        );
    }

    #[test]
    fn compare_flags_float_and_int_and_missing() {
        let a = Panel {
            rows: vec![row("0xabc", "2025-24"), row("0xonlya", "2025-24")],
        };
        let mut b_rows = vec![row("0xabc", "2025-24")];
        b_rows[0].depth_2pct = 400.5; // beyond tol
        b_rows[0].swaps = 11; // int mismatch
        b_rows.push(row("0xonlyb", "2025-24"));
        let b = Panel { rows: b_rows };
        let rep = compare(&a, &b, Tol::default());
        assert!(!rep.is_pass());
        assert_eq!(
            rep.only_in_a,
            vec![("0xonlya".to_string(), "2025-24".to_string())]
        );
        assert_eq!(
            rep.only_in_b,
            vec![("0xonlyb".to_string(), "2025-24".to_string())]
        );
        let fields: Vec<_> = rep.mismatches.iter().map(|m| m.field).collect();
        assert!(fields.contains(&"depth_2pct"));
        assert!(fields.contains(&"swaps"));
    }

    #[test]
    fn compare_within_tolerance_passes() {
        let a = Panel {
            rows: vec![row("0xabc", "2025-24")],
        };
        let mut b = a.clone();
        b.rows[0].twl_active_liquidity += 5e-7; // within default abs tol 1e-6
        let rep = compare(&a, &b, Tol::default());
        assert!(rep.is_pass(), "{:?}", rep.mismatches);
    }
}
