//! Adapter from the frozen [`Panel`](crate::data::panel::Panel) to the causal layer's
//! inputs: matching [`Unit`]s from treatment metadata and per-pool-week [`Obs`] with a
//! chosen outcome. Treatment timing, the token-pair cluster key, and the matching
//! selection variable are not in the `Panel` itself; the caller supplies them as
//! [`TreatmentMeta`], keeping the evidence layer (the panel) and the design layer
//! (treatment/matching) separate.
//!
//! Week indices come from a caller-supplied **frozen** [`WeekGrid`] (the canonical calendar
//! from `build_outcomes`' week grid), not from the observed panel labels: inferring the
//! grid from observed data would compress indices whenever a global week is absent and
//! shift every event time. Panel labels are validated against the grid.
//!
//! Contract: [`panel_to_obs`] is role-agnostic (it uses `TreatmentMeta.treated`), so it can
//! consume a neutral pre-match panel where Rust matching then defines roles. When the panel
//! already encodes the frozen matched roles, call [`check_roles_against_matches`] to assert
//! that `unit_role` agrees with the [`MatchResult`] before estimation.

use crate::causal::event_study::{EventStudyData, Obs, SampleComposition, build_matched_sample};
use crate::causal::matching::{MatchResult, Unit};
use crate::data::panel::{Panel, PoolWeek, UnitRole};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// Design-layer metadata for one pool, keyed by pool address in the [`Panel`].
#[derive(Clone, Debug)]
pub struct TreatmentMeta {
    pub treated: bool,
    /// activation week `"YYYY-WW"`; required for treated pools, ignored for controls
    pub t0_week: Option<String>,
    /// clustering unit (token pair)
    pub cluster_key: String,
    /// matching selection variable, on the matching scale (log pre-period fee revenue)
    pub s: f64,
    pub tier: i64,
    pub pair_class: String,
    pub low_exposure: bool,
}

impl TreatmentMeta {
    /// Shared validation used by both [`units_from_meta`] and [`panel_to_obs`].
    pub fn validate(&self, id: &str) -> Result<(), String> {
        if !self.s.is_finite() {
            return Err(format!("pool {id}: matching variable s is not finite"));
        }
        if self.pair_class.is_empty() {
            return Err(format!("pool {id}: empty pair_class"));
        }
        if self.cluster_key.is_empty() {
            return Err(format!("pool {id}: empty cluster_key"));
        }
        if self.treated && self.t0_week.is_none() {
            return Err(format!("pool {id}: treated but missing t0_week"));
        }
        Ok(())
    }
}

/// A canonical, frozen week calendar: label -> contiguous index. Built from the full grid
/// of `"YYYY-WW"` labels (`build_outcomes` week grid), independent of which weeks a given
/// pool or panel happens to observe.
pub struct WeekGrid {
    index: BTreeMap<String, i64>,
}

/// Validate a canonical `YYYY-WW` label: 4-digit year, `-`, 2-digit zero-padded `%W` week
/// in `00..=53`.
fn valid_week_label(l: &str) -> bool {
    let Some((y, w)) = l.split_once('-') else {
        return false;
    };
    if y.len() != 4 || !y.bytes().all(|b| b.is_ascii_digit()) {
        return false;
    }
    if w.len() != 2 || !w.bytes().all(|b| b.is_ascii_digit()) {
        return false;
    }
    w.parse::<u32>().map(|n| n <= 53).unwrap_or(false)
}

impl WeekGrid {
    /// Canonical grid generated from an inclusive date range, exactly as `build_outcomes`
    /// derives its week grid (step days, collect distinct `%Y-%W` labels). Contiguity is
    /// guaranteed by construction --- no global week can be silently missing --- so this is
    /// the preferred constructor. Dates are `"YYYY-MM-DD"`.
    pub fn from_range(start: &str, end: &str) -> Result<Self, String> {
        use chrono::NaiveDate;
        let s = NaiveDate::parse_from_str(start, "%Y-%m-%d").map_err(|e| e.to_string())?;
        let e = NaiveDate::parse_from_str(end, "%Y-%m-%d").map_err(|e| e.to_string())?;
        if e < s {
            return Err("end date precedes start date".into());
        }
        let mut set: BTreeSet<String> = BTreeSet::new();
        let mut d = s;
        while d <= e {
            set.insert(d.format("%Y-%W").to_string());
            d = d.succ_opt().ok_or("date overflow")?;
        }
        Ok(WeekGrid {
            index: set
                .into_iter()
                .enumerate()
                .map(|(i, w)| (w, i as i64))
                .collect(),
        })
    }

    /// Build from an explicit frozen label list (e.g. the weeks in `panel_completeness`).
    /// Indices are chronological (4-digit year + zero-padded week sorts lexically). Rejects
    /// malformed/out-of-range labels and duplicates. NOTE: this does not itself verify
    /// calendar \emph{completeness} between the first and last week --- prefer
    /// [`WeekGrid::from_range`], or cross-check the label count against the frozen grid.
    pub fn new(labels: &[String]) -> Result<Self, String> {
        for l in labels {
            if !valid_week_label(l) {
                return Err(format!(
                    "malformed week label in grid: {l:?} (expected YYYY-WW)"
                ));
            }
        }
        let set: BTreeSet<&str> = labels.iter().map(|s| s.as_str()).collect();
        if set.len() != labels.len() {
            return Err("duplicate week labels in grid".into());
        }
        Ok(WeekGrid {
            index: set
                .into_iter()
                .enumerate()
                .map(|(i, w)| (w.to_string(), i as i64))
                .collect(),
        })
    }
    pub fn idx(&self, label: &str) -> Option<i64> {
        self.index.get(label).copied()
    }
    pub fn contains(&self, label: &str) -> bool {
        self.index.contains_key(label)
    }
    /// All grid indices (e.g. to pass as `expected_weeks` for full-grid completeness).
    pub fn ids(&self) -> BTreeSet<i64> {
        self.index.values().copied().collect()
    }
}

/// Build matching [`Unit`]s from treatment metadata (deterministic order by pool id),
/// validating each via [`TreatmentMeta::validate`].
pub fn units_from_meta(meta: &HashMap<String, TreatmentMeta>) -> Result<Vec<Unit>, String> {
    let mut ids: Vec<&String> = meta.keys().collect();
    ids.sort();
    let mut out = Vec::with_capacity(ids.len());
    for id in ids {
        let m = &meta[id];
        m.validate(id)?;
        out.push(Unit {
            id: id.clone(),
            treated: m.treated,
            s: m.s,
            tier: m.tier,
            pair_class: m.pair_class.clone(),
            low_exposure: m.low_exposure,
        });
    }
    Ok(out)
}

/// Assert that the panel's frozen `unit_role`s agree with a [`MatchResult`]: pools labelled
/// `MatchedTreated` must be matched treated, `MatchedControl` must be used controls, and no
/// matched-treated pool may appear in the unmatched list. Call this on a post-match panel
/// before estimation; it is the contract enforcement the role-agnostic [`panel_to_obs`]
/// omits.
pub fn check_roles_against_matches(panel: &Panel, m: &MatchResult) -> Result<(), String> {
    let matched_treated: BTreeSet<&str> = m.pairs.iter().map(|p| p.treated.as_str()).collect();
    let used_controls: BTreeSet<&str> = m.control_freq.keys().map(|s| s.as_str()).collect();
    let unmatched: BTreeSet<&str> = m.unmatched_treated.iter().map(|s| s.as_str()).collect();

    // one role per pool; conflicting roles across a pool's rows is itself an error
    let mut role_of: BTreeMap<&str, UnitRole> = BTreeMap::new();
    for r in &panel.rows {
        match role_of.get(r.pool.as_str()) {
            None => {
                role_of.insert(r.pool.as_str(), r.unit_role);
            }
            Some(&prev) if prev != r.unit_role => {
                return Err(format!(
                    "pool {} has conflicting unit_role across rows",
                    r.pool
                ));
            }
            _ => {}
        }
    }

    // forward: each pool's panel role must agree with the match result. A matched unit may
    // NOT hide under a non-primary role (UnmatchedTreated / CrossvenueFork / Unknown).
    for (&pool, &role) in &role_of {
        let is_mt = matched_treated.contains(pool);
        let is_mc = used_controls.contains(pool);
        match role {
            UnitRole::MatchedTreated => {
                if !is_mt {
                    return Err(format!(
                        "pool {pool} is MatchedTreated but not in the match result"
                    ));
                }
                if unmatched.contains(pool) {
                    return Err(format!("pool {pool} is both MatchedTreated and unmatched"));
                }
            }
            UnitRole::MatchedControl => {
                if !is_mc {
                    return Err(format!(
                        "pool {pool} is MatchedControl but was never matched"
                    ));
                }
            }
            UnitRole::UnmatchedTreated | UnitRole::CrossvenueFork | UnitRole::Unknown => {
                if is_mt || is_mc {
                    return Err(format!(
                        "pool {pool} is a matched unit but is labelled {role:?} in the panel"
                    ));
                }
            }
        }
    }

    // reverse: every matched unit must be present in the panel with its primary role
    for &t in &matched_treated {
        if role_of.get(t) != Some(&UnitRole::MatchedTreated) {
            return Err(format!(
                "matched-treated pool {t} is absent or mislabelled in the panel"
            ));
        }
    }
    for &c in &used_controls {
        if role_of.get(c) != Some(&UnitRole::MatchedControl) {
            return Err(format!(
                "matched-control pool {c} is absent or mislabelled in the panel"
            ));
        }
    }
    Ok(())
}

/// Build per-pool-week [`Obs`] from a frozen `Panel`, treatment metadata, the frozen
/// [`WeekGrid`], and an outcome selector. Treated pools get `event_time = week - t0` (both
/// from the grid); controls get `event_time = 0` (unused). Rows whose pool has no metadata
/// are skipped. Errors if metadata is invalid, a row's week label is not in the frozen
/// grid, or a treated pool's `t0_week` is not in the grid.
pub fn panel_to_obs(
    panel: &Panel,
    meta: &HashMap<String, TreatmentMeta>,
    grid: &WeekGrid,
    outcome: impl Fn(&PoolWeek) -> f64,
) -> Result<Vec<Obs>, String> {
    let mut obs = Vec::with_capacity(panel.rows.len());
    for r in &panel.rows {
        let Some(m) = meta.get(&r.pool) else { continue };
        m.validate(&r.pool)?;
        let wk = grid.idx(&r.week).ok_or_else(|| {
            format!(
                "pool {}: week label {} not in the frozen grid",
                r.pool, r.week
            )
        })?;
        let event_time = if m.treated {
            let t0_label = m.t0_week.as_deref().unwrap(); // validated present above
            let t0 = grid.idx(t0_label).ok_or_else(|| {
                format!(
                    "treated pool {}: t0_week {} not in the frozen grid",
                    r.pool, t0_label
                )
            })?;
            wk - t0
        } else {
            0
        };
        obs.push(Obs {
            unit: r.pool.clone(),
            cluster_key: m.cluster_key.clone(),
            week_id: wk,
            treated: m.treated,
            event_time,
            y: outcome(r),
        });
    }
    Ok(obs)
}

/// Single enforced primary path: role-check the post-match panel against the match result,
/// build [`Obs`] on the frozen grid, and assemble the matched-overlap [`EventStudyData`]
/// with control-multiplicity weights and (optional) frozen-grid completeness. Returns one
/// error type so callers cannot skip a step. Use this rather than wiring the three
/// functions by hand.
pub fn build_primary_event_study_data(
    panel: &Panel,
    meta: &HashMap<String, TreatmentMeta>,
    grid: &WeekGrid,
    m: &MatchResult,
    expected_weeks: Option<&BTreeSet<i64>>,
    outcome: impl Fn(&PoolWeek) -> f64,
) -> Result<(EventStudyData, SampleComposition), String> {
    check_roles_against_matches(panel, m)?;
    let obs = panel_to_obs(panel, meta, grid, outcome)?;
    build_matched_sample(&obs, m, expected_weeks).map_err(|e| format!("{e:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::causal::event_study::run;
    use crate::causal::matching::nn_caliper_match;

    fn pw(pool: &str, role: UnitRole, week: &str, twl: f64) -> PoolWeek {
        PoolWeek {
            pool: pool.into(),
            unit_role: role,
            week: week.into(),
            swaps: 0,
            vol0: 0.0,
            vol1: 0.0,
            twl_active_liquidity: twl,
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
            net_liq: 0,
        }
    }

    #[test]
    fn event_time_uses_frozen_grid_not_observed_labels() {
        // frozen grid has 2026-00; a panel that never observed it must NOT compress indices
        let grid_labels: Vec<String> = ["2025-51", "2025-52", "2026-00", "2026-01"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let grid = WeekGrid::new(&grid_labels).unwrap();
        // 2026-01 minus 2025-52 must be 2 on the frozen grid even if 2026-00 is unobserved
        assert_eq!(
            grid.idx("2026-01").unwrap() - grid.idx("2025-52").unwrap(),
            2
        );
    }

    #[test]
    fn rejects_label_outside_frozen_grid() {
        let grid = WeekGrid::new(&["2025-03".to_string()]).unwrap();
        let panel = Panel {
            rows: vec![pw("p0", UnitRole::MatchedControl, "2025-99", 1.0)],
        };
        let mut meta = HashMap::new();
        meta.insert(
            "p0".to_string(),
            TreatmentMeta {
                treated: false,
                t0_week: None,
                cluster_key: "w".into(),
                s: 10.0,
                tier: 3000,
                pair_class: "w".into(),
                low_exposure: true,
            },
        );
        assert!(panel_to_obs(&panel, &meta, &grid, |r| r.twl_active_liquidity).is_err());
    }

    #[test]
    fn panel_to_obs_rejects_invalid_meta() {
        let grid = WeekGrid::new(&["2025-03".to_string()]).unwrap();
        let panel = Panel {
            rows: vec![pw("p0", UnitRole::MatchedControl, "2025-03", 1.0)],
        };
        let mut meta = HashMap::new();
        meta.insert(
            "p0".to_string(),
            TreatmentMeta {
                treated: false,
                t0_week: None,
                cluster_key: "".into(),
                s: 10.0, // empty cluster key
                tier: 3000,
                pair_class: "w".into(),
                low_exposure: true,
            },
        );
        assert!(panel_to_obs(&panel, &meta, &grid, |r| r.twl_active_liquidity).is_err());
    }

    #[test]
    fn role_check_against_matches_detects_leak() {
        let units = vec![
            Unit {
                id: "t1".into(),
                treated: true,
                s: 10.0,
                tier: 3000,
                pair_class: "w".into(),
                low_exposure: false,
            },
            Unit {
                id: "c1".into(),
                treated: false,
                s: 10.1,
                tier: 3000,
                pair_class: "w".into(),
                low_exposure: true,
            },
        ];
        let m = nn_caliper_match(&units, 0.5, 1);
        // panel labels a fork pool as MatchedTreated -> must be rejected
        let panel = Panel {
            rows: vec![pw("x9", UnitRole::MatchedTreated, "2025-03", 1.0)],
        };
        assert!(check_roles_against_matches(&panel, &m).is_err());
        // a consistent panel passes
        let ok = Panel {
            rows: vec![
                pw("t1", UnitRole::MatchedTreated, "2025-03", 1.0),
                pw("c1", UnitRole::MatchedControl, "2025-03", 1.0),
            ],
        };
        assert!(check_roles_against_matches(&ok, &m).is_ok());
    }

    #[test]
    fn end_to_end_panel_to_event_study_recovers_effect() {
        let weeks: Vec<String> = (1..=8).map(|w| format!("2025-{:02}", w)).collect();
        let grid = WeekGrid::new(&weeks).unwrap();
        let t0 = "2025-05";
        let effect = 2.0;
        let mut rows = Vec::new();
        let mut meta: HashMap<String, TreatmentMeta> = HashMap::new();
        for i in 0..8 {
            let treated = i < 4;
            let pool = format!("p{i}");
            let role = if treated {
                UnitRole::MatchedTreated
            } else {
                UnitRole::MatchedControl
            };
            meta.insert(
                pool.clone(),
                TreatmentMeta {
                    treated,
                    t0_week: if treated { Some(t0.to_string()) } else { None },
                    cluster_key: pool.clone(),
                    // separated so each treated matches a distinct control 1:1 (every labelled
                    // MatchedControl is actually used -> role check is consistent)
                    s: if treated {
                        10.0 * (i as f64 + 1.0)
                    } else {
                        10.0 * (i as f64 - 3.0) + 0.01
                    },
                    tier: 3000,
                    pair_class: "w".into(),
                    low_exposure: !treated,
                },
            );
            for wk in &weeks {
                let wi = grid.idx(wk).unwrap();
                let post = treated && wk.as_str() >= t0;
                let twl = 100.0 * i as f64 + 3.0 * wi as f64 + if post { effect } else { 0.0 };
                rows.push(pw(&pool, role, wk, twl));
            }
        }
        let panel = Panel { rows };
        let units = units_from_meta(&meta).unwrap();
        let m = nn_caliper_match(&units, 0.5, 1);
        // single enforced primary path (role check + obs + matched sample + completeness)
        let (data, comp) =
            build_primary_event_study_data(&panel, &meta, &grid, &m, Some(&grid.ids()), |r| {
                r.twl_active_liquidity
            })
            .unwrap();
        assert_eq!(comp.matched_treated, 4);
        let res = run(&data, 40, 0.05, 3).unwrap();
        assert!(res.fe_converged);
        for (k, c) in res.bins.iter().zip(&res.coefs) {
            if *k >= 0 {
                assert!((c.beta - effect).abs() < 1e-6, "post {k}: {}", c.beta);
            } else {
                assert!(c.beta.abs() < 1e-6, "pre {k}: {}", c.beta);
            }
        }
    }

    #[test]
    fn week_grid_rejects_malformed_labels() {
        assert!(WeekGrid::new(&["2025-99".to_string()]).is_err()); // week out of range
        assert!(WeekGrid::new(&["2025-1".to_string()]).is_err()); // not zero-padded
        assert!(WeekGrid::new(&["25-03".to_string()]).is_err()); // year not 4 digits
        assert!(WeekGrid::new(&["2025-03".to_string(), "2025-04".to_string()]).is_ok());
    }

    #[test]
    fn week_grid_new_rejects_duplicates() {
        assert!(WeekGrid::new(&["2025-03".to_string(), "2025-03".to_string()]).is_err());
    }

    #[test]
    fn week_grid_from_range_is_contiguous_across_year_boundary() {
        // range spanning 2025-12 into 2026-01 must place 2025-52 and 2026-00 adjacent,
        // and be gap-free by construction
        let g = WeekGrid::from_range("2025-12-20", "2026-01-10").unwrap();
        let a = g.idx("2025-52").expect("2025-52 present");
        let b = g.idx("2026-00").expect("2026-00 present");
        assert_eq!(b - a, 1, "consecutive weeks must have consecutive indices");
        // every index in [min,max] is used (contiguous)
        let ids = g.ids();
        let (lo, hi) = (*ids.iter().next().unwrap(), *ids.iter().last().unwrap());
        assert_eq!(ids.len() as i64, hi - lo + 1);
    }

    #[test]
    fn role_check_rejects_matched_unit_hidden_as_fork() {
        let units = vec![
            Unit {
                id: "t1".into(),
                treated: true,
                s: 10.0,
                tier: 3000,
                pair_class: "w".into(),
                low_exposure: false,
            },
            Unit {
                id: "c1".into(),
                treated: false,
                s: 10.1,
                tier: 3000,
                pair_class: "w".into(),
                low_exposure: true,
            },
        ];
        let m = nn_caliper_match(&units, 0.5, 1);
        // t1 is matched but the panel hides it under a non-primary role -> must fail
        let panel = Panel {
            rows: vec![
                pw("t1", UnitRole::CrossvenueFork, "2025-03", 1.0),
                pw("c1", UnitRole::MatchedControl, "2025-03", 1.0),
            ],
        };
        assert!(check_roles_against_matches(&panel, &m).is_err());
    }
}
