//! Panel completeness and outcome-sanity QA, porting `build_outcomes.py`'s W-3 report.
//!
//! This is the gate that runs *before* any estimation: it quantifies how much of the frozen
//! (unit set x week grid) design is actually observed, where outcomes are structurally zero,
//! and whether outcome ranges are plausible. It computes **no** pre-trend and **no** ATT.
//!
//! The numbers are defined to agree with the Python reference: expected pool-weeks =
//! |frozen unit set| x |frozen week grid|; observed = rows present in the panel; per-role
//! breakdowns iterate the unit set (roles from `panel_units.json`), while the outcome
//! zero-counts and sanity stats range over every panel row (including `unknown`-role pools).

use crate::data::panel::{Panel, PoolWeek, UnitRole};
use crate::data::reconstruct::week_id;
use chrono::{TimeZone, Utc};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// Human-readable role label (matches the serde renames / Python `unit_role` strings).
pub fn role_label(r: UnitRole) -> &'static str {
    match r {
        UnitRole::MatchedTreated => "matched_treated",
        UnitRole::MatchedControl => "matched_control",
        UnitRole::UnmatchedTreated => "unmatched_treated",
        UnitRole::CrossvenueFork => "crossvenue_fork",
        UnitRole::Unknown => "unknown",
    }
}

/// The 16 outcome columns (everything but pool/unit_role/week), as `(name, value)` with
/// counts widened to `f64`. Order matches `build_outcomes.py`'s column order.
pub fn outcomes(r: &PoolWeek) -> [(&'static str, f64); 16] {
    [
        ("swaps", r.swaps as f64),
        ("vol0", r.vol0),
        ("vol1", r.vol1),
        ("twl_active_liquidity", r.twl_active_liquidity),
        ("depth_1pct", r.depth_1pct),
        ("depth_2pct", r.depth_2pct),
        ("depth_5pct", r.depth_5pct),
        ("lp_entry_count", r.lp_entry_count as f64),
        ("lp_exit_count", r.lp_exit_count as f64),
        ("unique_lp_count", r.unique_lp_count as f64),
        ("jit_share_same_block", r.jit_share_same_block),
        ("lp_fee_income_native1", r.lp_fee_income_native1),
        (
            "lp_fee_income_per_active_liquidity",
            r.lp_fee_income_per_active_liquidity,
        ),
        ("collect_amount1_native", r.collect_amount1_native),
        ("position_duration_days", r.position_duration_days),
        ("net_liq", r.net_liq as f64),
    ]
}

/// Distinct `%Y-%W` week labels for timestamps stepping daily over `[b0, b1]` (inclusive),
/// the same construction as the Python `week_grid`.
pub fn week_grid(b0: i64, b1: i64) -> BTreeSet<String> {
    let mut ws = BTreeSet::new();
    let mut t = b0;
    while t <= b1 {
        ws.insert(week_id(t));
        t += 86_400;
    }
    ws
}

/// The paper's frozen week grid: 2024-01-01 00:00:00 UTC through 2026-06-30 23:59:59 UTC.
pub fn frozen_week_grid() -> BTreeSet<String> {
    let b0 = Utc
        .with_ymd_and_hms(2024, 1, 1, 0, 0, 0)
        .unwrap()
        .timestamp();
    let b1 = Utc
        .with_ymd_and_hms(2026, 6, 30, 23, 59, 59)
        .unwrap()
        .timestamp();
    week_grid(b0, b1)
}

#[derive(Debug, Serialize)]
pub struct SanityStat {
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub nonzero_frac: f64,
}

/// Full completeness + sanity report; `Serialize`s to a JSON shaped like the Python
/// `panel_completeness.json` merged with `outcome_sanity_report.json`.
#[derive(Debug, Serialize)]
pub struct CompletenessReport {
    pub frozen_unit_set_size: usize,
    pub frozen_week_grid_size: usize,
    pub expected_pool_weeks_total: usize,
    pub observed_pool_weeks_total: usize,
    pub expected_pool_weeks_by_role: BTreeMap<String, usize>,
    pub observed_pool_weeks_by_role: BTreeMap<String, usize>,
    pub missing_pool_weeks_by_role: BTreeMap<String, i64>,
    pub units_with_zero_observed_weeks_count: usize,
    pub units_with_zero_observed_weeks_sample: Vec<String>,
    pub panel_rows_by_role: BTreeMap<String, usize>,
    /// outcome -> role -> count of rows where the outcome is exactly zero
    pub outcome_specific_zero_counts_by_role: BTreeMap<String, BTreeMap<String, usize>>,
    pub sanity: BTreeMap<String, SanityStat>,
    /// pool-weeks in the panel whose week label is outside the frozen grid (should be 0)
    pub observed_weeks_outside_grid: usize,
}

fn median(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// Compute the report from a panel, the role map (its keys are the frozen unit set), and the
/// frozen week grid.
pub fn report(
    panel: &Panel,
    roles: &HashMap<String, UnitRole>,
    grid: &BTreeSet<String>,
) -> CompletenessReport {
    let grid_size = grid.len();
    let unit_set: HashSet<&String> = roles.keys().collect();

    // pool -> set(weeks) observed
    let mut observed: HashMap<&str, HashSet<&str>> = HashMap::new();
    for r in &panel.rows {
        observed.entry(&r.pool).or_default().insert(&r.week);
    }

    // expected / observed pool-weeks by role, over the unit set
    let mut expected_by_role: BTreeMap<String, usize> = BTreeMap::new();
    let mut observed_by_role: BTreeMap<String, usize> = BTreeMap::new();
    for (pool, role) in roles {
        let lbl = role_label(*role).to_string();
        *expected_by_role.entry(lbl.clone()).or_default() += grid_size;
        let obs = observed.get(pool.as_str()).map_or(0, |s| s.len());
        *observed_by_role.entry(lbl).or_default() += obs;
    }
    let missing_by_role: BTreeMap<String, i64> = expected_by_role
        .keys()
        .map(|r| {
            let e = expected_by_role[r] as i64;
            let o = *observed_by_role.get(r).unwrap_or(&0) as i64;
            (r.clone(), e - o)
        })
        .collect();

    // units in the frozen set with no panel rows at all
    let pools_with_data: HashSet<&str> = panel.rows.iter().map(|r| r.pool.as_str()).collect();
    let mut units_no_data: Vec<String> = unit_set
        .iter()
        .filter(|p| !pools_with_data.contains(p.as_str()))
        .map(|p| (*p).clone())
        .collect();
    units_no_data.sort();

    // panel rows by role (all rows, including unknown) + outcome zero-counts by role
    let mut rows_by_role: BTreeMap<String, usize> = BTreeMap::new();
    for r in &panel.rows {
        *rows_by_role
            .entry(role_label(r.unit_role).to_string())
            .or_default() += 1;
    }
    let mut zero_counts: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    for r in &panel.rows {
        let lbl = role_label(r.unit_role).to_string();
        for (name, v) in outcomes(r) {
            if v == 0.0 {
                *zero_counts
                    .entry(name.to_string())
                    .or_default()
                    .entry(lbl.clone())
                    .or_default() += 1;
            }
        }
    }
    // ensure every outcome key exists even if never zero (stable JSON shape)
    if let Some(first) = panel.rows.first() {
        for (name, _) in outcomes(first) {
            zero_counts.entry(name.to_string()).or_default();
        }
    }

    // sanity: per outcome min/max/median/nonzero_frac over all rows
    let n = panel.rows.len().max(1);
    let mut sanity: BTreeMap<String, SanityStat> = BTreeMap::new();
    if let Some(first) = panel.rows.first() {
        for (name, _) in outcomes(first) {
            let mut vals: Vec<f64> = panel
                .rows
                .iter()
                .map(|r| outcomes(r).into_iter().find(|(k, _)| *k == name).unwrap().1)
                .collect();
            let nonzero = vals.iter().filter(|x| **x != 0.0).count();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            sanity.insert(
                name.to_string(),
                SanityStat {
                    min: *vals.first().unwrap_or(&f64::NAN),
                    max: *vals.last().unwrap_or(&f64::NAN),
                    median: median(&vals),
                    nonzero_frac: (nonzero as f64 / n as f64 * 1e4).round() / 1e4,
                },
            );
        }
    }

    let outside = panel
        .rows
        .iter()
        .filter(|r| !grid.contains(&r.week))
        .count();

    CompletenessReport {
        frozen_unit_set_size: unit_set.len(),
        frozen_week_grid_size: grid_size,
        expected_pool_weeks_total: unit_set.len() * grid_size,
        observed_pool_weeks_total: panel.rows.len(),
        expected_pool_weeks_by_role: expected_by_role,
        observed_pool_weeks_by_role: observed_by_role,
        missing_pool_weeks_by_role: missing_by_role,
        units_with_zero_observed_weeks_count: units_no_data.len(),
        units_with_zero_observed_weeks_sample: units_no_data.into_iter().take(30).collect(),
        panel_rows_by_role: rows_by_role,
        outcome_specific_zero_counts_by_role: zero_counts,
        sanity,
        observed_weeks_outside_grid: outside,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pw(pool: &str, role: UnitRole, week: &str, swaps: i64, net: i128) -> PoolWeek {
        PoolWeek {
            pool: pool.into(),
            unit_role: role,
            week: week.into(),
            swaps,
            vol0: 0.0,
            vol1: 0.0,
            twl_active_liquidity: 0.0,
            depth_1pct: 0.0,
            depth_2pct: 0.0,
            depth_5pct: 0.0,
            lp_entry_count: 0,
            lp_exit_count: 0,
            unique_lp_count: 0,
            jit_share_same_block: 0.0,
            lp_fee_income_native1: 0.0,
            lp_fee_income_per_active_liquidity: 0.0,
            collect_amount1_native: 0.0,
            position_duration_days: 0.0,
            net_liq: net,
        }
    }

    #[test]
    fn frozen_grid_spans_expected_range() {
        let g = frozen_week_grid();
        assert!(g.contains("2024-00") || g.contains("2024-01"));
        assert!(g.contains("2025-52"));
        assert!(g.contains("2026-26")); // 2026-06-30 falls in week 26
        // ~131 weeks across 2.5 years; allow slack for %W week-00 boundaries
        assert!(g.len() >= 128 && g.len() <= 135, "grid size {}", g.len());
    }

    #[test]
    fn completeness_counts_match_definitions() {
        let roles: HashMap<String, UnitRole> = [
            ("A".to_string(), UnitRole::MatchedTreated),
            ("B".to_string(), UnitRole::MatchedControl),
            ("C".to_string(), UnitRole::MatchedControl), // in unit set, no data
        ]
        .into_iter()
        .collect();
        let grid: BTreeSet<String> = ["2025-01", "2025-02", "2025-03"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let panel = Panel {
            rows: vec![
                pw("A", UnitRole::MatchedTreated, "2025-01", 5, 100),
                pw("A", UnitRole::MatchedTreated, "2025-02", 0, 0), // swaps zero
                pw("B", UnitRole::MatchedControl, "2025-01", 3, -4),
            ],
        };
        let rep = report(&panel, &roles, &grid);

        assert_eq!(rep.frozen_unit_set_size, 3);
        assert_eq!(rep.frozen_week_grid_size, 3);
        assert_eq!(rep.expected_pool_weeks_total, 9);
        assert_eq!(rep.observed_pool_weeks_total, 3);
        // treated: expected 3, observed 2 (A has 2 weeks) -> missing 1
        assert_eq!(rep.expected_pool_weeks_by_role["matched_treated"], 3);
        assert_eq!(rep.observed_pool_weeks_by_role["matched_treated"], 2);
        assert_eq!(rep.missing_pool_weeks_by_role["matched_treated"], 1);
        // control: expected 6 (B,C), observed 1 (B has 1 week, C none) -> missing 5
        assert_eq!(rep.expected_pool_weeks_by_role["matched_control"], 6);
        assert_eq!(rep.observed_pool_weeks_by_role["matched_control"], 1);
        assert_eq!(rep.missing_pool_weeks_by_role["matched_control"], 5);
        // C has no data
        assert_eq!(rep.units_with_zero_observed_weeks_count, 1);
        assert_eq!(
            rep.units_with_zero_observed_weeks_sample,
            vec!["C".to_string()]
        );
        // one treated row has swaps==0
        assert_eq!(
            rep.outcome_specific_zero_counts_by_role["swaps"]["matched_treated"],
            1
        );
        // sanity: swaps over [5,0,3] -> min 0, max 5, median 3, nonzero 2/3
        let s = &rep.sanity["swaps"];
        assert_eq!(s.min, 0.0);
        assert_eq!(s.max, 5.0);
        assert_eq!(s.median, 3.0);
        assert!((s.nonzero_frac - 0.6667).abs() < 1e-4);
        assert_eq!(rep.observed_weeks_outside_grid, 0);
    }
}
