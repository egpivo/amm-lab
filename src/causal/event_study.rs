//! Orchestration of the primary event-study estimator (Equation~(3)): validate inputs,
//! build the event-time design, absorb pool and week fixed effects by (weighted)
//! alternating projections, run OLS/WLS on the residualized design, and attach token-pair
//! cluster-robust inference and a wild cluster bootstrap. Writes the coefficient path and
//! a SHA256 estimator manifest covering the full design, sample, cluster, and weights.
//!
//! The estimation sample must be the matched-overlap sample: use
//! [`build_matched_sample`], which includes only matched treated and matched controls,
//! carries with-replacement control multiplicity as frequency weights, and records the
//! composition. Feeding an arbitrary [`EventStudyData`] is possible but bypasses that
//! contract. Nothing here runs on the frozen panel until it clears the no-pretrend /
//! no-ATT gate; the tests run on synthetic panels only.

use crate::causal::design::{bins_from, event_time_columns};
use crate::causal::fe::{TwoWayFeSpec, residualize_twoway_weighted};
use crate::causal::matching::MatchResult;
use crate::causal::ols::{
    BootstrapSpec, CoefInference, OlsError, ols, wcr_pvalues, wild_cluster_bootstrap,
};
use nalgebra::{DMatrix, DVector};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

/// Long-format estimation sample for one outcome. All vectors have length `n_obs`; `pool`,
/// `week`, and `cluster` are 0-based group ids. `weights` are frequency weights (treated
/// units 1, matched controls their multiplicity). `extra` holds already-built additional
/// regressors (e.g. baseline-time interactions), each a column of length `n_obs`.
pub struct EventStudyData {
    pub pool: Vec<usize>,
    pub n_pool: usize,
    pub week: Vec<usize>,
    pub n_week: usize,
    pub cluster: Vec<usize>,
    pub n_clusters: usize,
    pub treated: Vec<bool>,
    pub event_time: Vec<i64>,
    pub y: Vec<f64>,
    pub weights: Vec<f64>,
    pub extra: Vec<Vec<f64>>,
    /// set by [`build_matched_sample`]; carried into the manifest
    pub composition: Option<SampleComposition>,
}

pub struct EventStudyResult {
    pub bins: Vec<i64>,
    pub coefs: Vec<CoefInference>, // aligned with `bins`
    pub n_obs: usize,
    pub n_eff: f64, // frequency-weighted observation count
    pub rank: usize,
    pub cond: f64,
    pub fe_converged: bool,
    pub fe_max_residual_group_mean: f64,
    pub seed: u64,
    pub n_boot: usize,
    pub digest: String, // SHA256 hex over the full input
    pub composition: Option<SampleComposition>,
}

#[derive(Debug)]
pub enum EsError {
    Empty,
    Invalid(String),
    RankDeficient { rank: usize, k: usize, cond: f64 },
    NoResidualDof { n: usize, k: usize },
}

impl EventStudyData {
    /// Hard input-validation gate: uniform lengths, in-range group ids, >= 2 clusters,
    /// positive finite weights, finite outcomes/regressors, and sane bootstrap params.
    fn validate(&self, n_boot: usize, alpha: f64) -> Result<usize, EsError> {
        let n = self.y.len();
        if n == 0 {
            return Err(EsError::Empty);
        }
        let same = self.pool.len() == n
            && self.week.len() == n
            && self.cluster.len() == n
            && self.treated.len() == n
            && self.event_time.len() == n
            && self.weights.len() == n;
        if !same {
            return Err(EsError::Invalid("input vectors have unequal length".into()));
        }
        if self.extra.iter().any(|c| c.len() != n) {
            return Err(EsError::Invalid(
                "an extra regressor column has wrong length".into(),
            ));
        }
        if self.pool.iter().any(|&g| g >= self.n_pool) {
            return Err(EsError::Invalid("pool id out of range".into()));
        }
        if self.week.iter().any(|&g| g >= self.n_week) {
            return Err(EsError::Invalid("week id out of range".into()));
        }
        if self.cluster.iter().any(|&g| g >= self.n_clusters) {
            return Err(EsError::Invalid("cluster id out of range".into()));
        }
        if self.n_clusters < 2 {
            return Err(EsError::Invalid(
                "cluster-robust inference needs >= 2 clusters".into(),
            ));
        }
        if self.weights.iter().any(|&w| !(w.is_finite() && w > 0.0)) {
            return Err(EsError::Invalid(
                "weights must be finite and positive".into(),
            ));
        }
        if self.y.iter().any(|v| !v.is_finite()) {
            return Err(EsError::Invalid("outcome has non-finite values".into()));
        }
        if self.extra.iter().flatten().any(|v| !v.is_finite()) {
            return Err(EsError::Invalid(
                "an extra regressor has non-finite values".into(),
            ));
        }
        if n_boot == 0 {
            return Err(EsError::Invalid("n_boot must be > 0".into()));
        }
        if !(alpha > 0.0 && alpha < 1.0) {
            return Err(EsError::Invalid("alpha must be in (0,1)".into()));
        }
        Ok(n)
    }

    fn digest(&self) -> String {
        let mut h = Sha256::new();
        for c in [self.n_pool, self.n_week, self.n_clusters] {
            h.update((c as u64).to_le_bytes());
        }
        for i in 0..self.y.len() {
            h.update((self.pool[i] as u64).to_le_bytes());
            h.update((self.week[i] as u64).to_le_bytes());
            h.update((self.cluster[i] as u64).to_le_bytes());
            h.update([self.treated[i] as u8]);
            h.update(self.event_time[i].to_le_bytes());
            h.update(self.y[i].to_bits().to_le_bytes());
            h.update(self.weights[i].to_bits().to_le_bytes());
        }
        h.update((self.extra.len() as u64).to_le_bytes());
        for col in &self.extra {
            for v in col {
                h.update(v.to_bits().to_le_bytes());
            }
        }
        format!("{:x}", h.finalize())
    }
}

/// Estimate the event-study path. `alpha` is the two-sided level (e.g. 0.05).
pub fn run(
    data: &EventStudyData,
    n_boot: usize,
    alpha: f64,
    seed: u64,
) -> Result<EventStudyResult, EsError> {
    let n = data.validate(n_boot, alpha)?;
    let bins = bins_from(&data.event_time, &data.treated);
    if bins.is_empty() {
        return Err(EsError::Empty);
    }
    let n_et = bins.len();
    let k = n_et + data.extra.len();
    if n <= k {
        return Err(EsError::NoResidualDof { n, k });
    }
    let n_eff: f64 = data.weights.iter().sum();
    if n_eff <= k as f64 {
        // frequency-weighted sample size must exceed K for a valid CR1 correction
        return Err(EsError::NoResidualDof {
            n: n_eff.floor() as usize,
            k,
        });
    }

    // [y | event-time columns | extra], weighted-residualized on pool+week FE, then
    // scaled by sqrt(weight) so the OLS below is WLS with the matching frequency weights.
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(1 + k);
    cols.push(data.y.clone());
    cols.extend(event_time_columns(&data.event_time, &data.treated, &bins));
    cols.extend(data.extra.iter().cloned());
    let fe = residualize_twoway_weighted(
        &mut cols,
        TwoWayFeSpec {
            w: &data.weights,
            pool: &data.pool,
            n_pool: data.n_pool,
            week: &data.week,
            n_week: data.n_week,
            tol: 1e-10,
            max_iter: 5000,
        },
    );
    if !fe.converged {
        return Err(EsError::Invalid(format!(
            "two-way FE absorption did not converge (max residual group mean {:.3e})",
            fe.max_residual_group_mean
        )));
    }
    let sw: Vec<f64> = data.weights.iter().map(|w| w.sqrt()).collect();
    for col in cols.iter_mut() {
        for i in 0..n {
            col[i] *= sw[i];
        }
    }

    let y = DVector::from_vec(cols[0].clone());
    let x_cols: Vec<DVector<f64>> = cols[1..]
        .iter()
        .map(|c| DVector::from_vec(c.clone()))
        .collect();
    let x = DMatrix::from_columns(&x_cols);

    let fit = match ols(&x, &y) {
        Ok(f) => f,
        Err(OlsError::RankDeficient { rank, k, cond }) => {
            return Err(EsError::RankDeficient { rank, k, cond });
        }
        Err(OlsError::NoResidualDof { n, k }) => return Err(EsError::NoResidualDof { n, k }),
        Err(OlsError::Svd) => return Err(EsError::Invalid("SVD failed".into())),
    };
    let mut inf = wild_cluster_bootstrap(
        &x,
        &fit,
        BootstrapSpec {
            cluster: &data.cluster,
            n_clusters: data.n_clusters,
            n_eff,
            n_boot,
            alpha,
            seed,
        },
    );
    // Restricted (WCR) p-values: the reported default with few clusters. Uses the same seed
    // (per-coefficient derived stream) so results are reproducible from the manifest.
    let wcr = wcr_pvalues(
        &x,
        &fit,
        &data.cluster,
        data.n_clusters,
        n_eff,
        n_boot,
        seed,
    );
    for (c, p) in inf.iter_mut().zip(&wcr) {
        c.wcr_p = *p;
    }

    Ok(EventStudyResult {
        bins,
        coefs: inf.into_iter().take(n_et).collect(),
        n_obs: n,
        n_eff,
        rank: fit.rank,
        cond: fit.cond,
        fe_converged: fe.converged,
        fe_max_residual_group_mean: fe.max_residual_group_mean,
        seed,
        n_boot,
        digest: data.digest(),
        composition: data.composition.clone(),
    })
}

impl EventStudyResult {
    /// Write the coefficient path as CSV and a JSON estimator manifest.
    pub fn write(&self, dir: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut w = std::fs::File::create(format!("{dir}/event_study_coefficients.csv"))?;
        // wcr_p is the reported default; boot_p (WCU) retained for comparison.
        writeln!(
            w,
            "event_time,beta,se,t,ci_lo,ci_hi,boot_ci_lo,boot_ci_hi,wcr_p,boot_p"
        )?;
        for (k, c) in self.bins.iter().zip(&self.coefs) {
            writeln!(
                w,
                "{k},{:.10},{:.10},{:.6},{:.10},{:.10},{:.10},{:.10},{:.6},{:.6}",
                c.beta, c.se, c.t, c.ci_lo, c.ci_hi, c.boot_ci_lo, c.boot_ci_hi, c.wcr_p, c.boot_p
            )?;
        }
        let manifest = serde_json::json!({
            "estimator": "two-way FE event-study WLS (Eq. 3), weighted-residualized by pool+week",
            "inference": "token-pair cluster-robust CR1 + wild cluster bootstrap-t (Rademacher)",
            "reported_pvalue": "wcr_p = restricted (WCR, null-imposed) bootstrap-t; boot_p = unrestricted (WCU), retained for comparison",
            "omitted_event_bin": -1,
            "n_obs": self.n_obs,
            "n_eff_weighted": self.n_eff,
            "n_bins": self.bins.len(),
            "design_rank": self.rank,
            "design_condition_number": self.cond,
            "fe_converged": self.fe_converged,
            "fe_max_residual_group_mean": self.fe_max_residual_group_mean,
            "sample_composition": self.composition,
            "n_boot": self.n_boot,
            "seed": self.seed,
            "input_sha256": self.digest,
            "note": "phase-1 primary estimator; HonestDiD / robustness-value / entropy-balancing not yet implemented"
        });
        std::fs::write(
            format!("{dir}/estimator_manifest.json"),
            serde_json::to_string_pretty(&manifest)?,
        )?;
        Ok(())
    }
}

/// One observation for the matched-sample builder, keyed by unit and cluster.
pub struct Obs {
    pub unit: String,
    pub cluster_key: String,
    pub week_id: i64,
    pub treated: bool,
    pub event_time: i64,
    pub y: f64,
}

/// Recorded composition of the estimation sample (written to the manifest / reports).
#[derive(Debug, Clone, serde::Serialize)]
pub struct SampleComposition {
    pub matched_treated: usize,
    pub control_units: usize,
    pub excluded_unmatched_treated: usize,
    pub control_multiplicity: BTreeMap<String, usize>,
}

/// Enforce the matched-overlap contract: build an [`EventStudyData`] from observations and
/// a [`MatchResult`], including only matched treated units (weight 1) and matched controls
/// (weight = with-replacement multiplicity), excluding unmatched treated and any control
/// never matched. Unmatched treated and exposed / unused controls cannot leak into the
/// primary DiD through this path.
pub fn build_matched_sample(
    obs: &[Obs],
    m: &MatchResult,
    expected_weeks: Option<&BTreeSet<i64>>,
) -> Result<(EventStudyData, SampleComposition), EsError> {
    let matched_treated: BTreeSet<&str> = m.pairs.iter().map(|p| p.treated.as_str()).collect();
    // unit -> frequency weight; treated 1, control = multiplicity
    let mut weight_of: BTreeMap<&str, f64> = BTreeMap::new();
    for t in &matched_treated {
        weight_of.insert(t, 1.0);
    }
    for (c, &f) in &m.control_freq {
        weight_of.insert(c.as_str(), f as f64);
    }

    // contiguous id maps (deterministic: sorted keys)
    let unit_ids: BTreeSet<&str> = weight_of.keys().copied().collect();
    let unit_ix: BTreeMap<&str, usize> =
        unit_ids.iter().enumerate().map(|(i, &u)| (u, i)).collect();
    let weeks: BTreeSet<i64> = obs
        .iter()
        .filter(|o| unit_ix.contains_key(o.unit.as_str()))
        .map(|o| o.week_id)
        .collect();
    let week_ix: BTreeMap<i64, usize> = weeks.iter().enumerate().map(|(i, &w)| (w, i)).collect();
    let clusters: BTreeSet<&str> = obs
        .iter()
        .filter(|o| unit_ix.contains_key(o.unit.as_str()))
        .map(|o| o.cluster_key.as_str())
        .collect();
    let cl_ix: BTreeMap<&str, usize> = clusters.iter().enumerate().map(|(i, &c)| (c, i)).collect();

    let (mut pool, mut week, mut cluster, mut treated, mut event_time, mut y, mut weights) =
        (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    let included: BTreeSet<String> = unit_ids.iter().map(|u| u.to_string()).collect();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut unit_weeks: BTreeMap<String, BTreeSet<i64>> = BTreeMap::new();
    for o in obs {
        if let Some(&ui) = unit_ix.get(o.unit.as_str()) {
            // role consistency: matched treated rows must be treated, controls must not
            let expect_treated = matched_treated.contains(o.unit.as_str());
            if o.treated != expect_treated {
                return Err(EsError::Invalid(format!(
                    "unit {} has row treated={} but its matched role is treated={}",
                    o.unit, o.treated, expect_treated
                )));
            }
            seen.insert(o.unit.clone());
            unit_weeks
                .entry(o.unit.clone())
                .or_default()
                .insert(o.week_id);
            pool.push(ui);
            week.push(week_ix[&o.week_id]);
            cluster.push(cl_ix[o.cluster_key.as_str()]);
            treated.push(o.treated);
            event_time.push(o.event_time);
            y.push(o.y);
            weights.push(weight_of[o.unit.as_str()]);
        }
    }

    // completeness: every matched unit must have panel rows, else the manifest would count
    // it as included while it is silently absent from the estimation arrays
    let missing: Vec<String> = included.difference(&seen).cloned().collect();
    if !missing.is_empty() {
        return Err(EsError::Invalid(format!(
            "{} matched unit(s) have no panel rows (e.g. {})",
            missing.len(),
            missing
                .iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }

    // opt-in frozen-grid completeness: every included unit must cover the expected week set
    // (else a silently sparse pool-week panel enters the primary DiD). When `None`, the
    // external completeness gate (panel_completeness.json) is relied on instead.
    if let Some(exp) = expected_weeks {
        let gaps: Vec<String> = included
            .iter()
            .filter(|u| {
                let uw = unit_weeks.get(*u);
                uw.is_none_or(|s| !exp.iter().all(|w| s.contains(w)))
            })
            .cloned()
            .collect();
        if !gaps.is_empty() {
            return Err(EsError::Invalid(format!(
                "{} matched unit(s) miss expected pool-weeks (e.g. {})",
                gaps.len(),
                gaps.iter().take(5).cloned().collect::<Vec<_>>().join(", ")
            )));
        }
    }

    let comp = SampleComposition {
        matched_treated: matched_treated.len(),
        control_units: m.control_freq.len(),
        excluded_unmatched_treated: m.unmatched_treated.len(),
        control_multiplicity: m
            .control_freq
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect(),
    };
    let data = EventStudyData {
        n_pool: unit_ix.len(),
        n_week: week_ix.len(),
        n_clusters: cl_ix.len(),
        pool,
        week,
        cluster,
        treated,
        event_time,
        y,
        weights,
        extra: vec![],
        composition: Some(comp.clone()),
    };
    Ok((data, comp))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::causal::matching::{Unit, nn_caliper_match};

    fn synth(effect: f64) -> EventStudyData {
        let n_pool = 10;
        let n_week = 8;
        let t0 = 4i64;
        let (mut pool, mut week, mut cluster, mut treated, mut et, mut y) =
            (vec![], vec![], vec![], vec![], vec![], vec![]);
        for p in 0..n_pool {
            let is_t = p < 5;
            for w in 0..n_week {
                pool.push(p);
                week.push(w);
                cluster.push(p);
                treated.push(is_t);
                let e = w as i64 - t0;
                et.push(e);
                let post = is_t && e >= 0;
                y.push(3.0 * p as f64 + 1.5 * w as f64 + if post { effect } else { 0.0 });
            }
        }
        let n = y.len();
        EventStudyData {
            pool,
            n_pool,
            week,
            n_week,
            cluster,
            n_clusters: n_pool,
            treated,
            event_time: et,
            y,
            weights: vec![1.0; n],
            extra: vec![],
            composition: None,
        }
    }

    #[test]
    fn recovers_constant_post_effect() {
        let d = synth(2.0);
        let r = run(&d, 50, 0.05, 7).unwrap();
        for (k, c) in r.bins.iter().zip(&r.coefs) {
            if *k >= 0 {
                assert!((c.beta - 2.0).abs() < 1e-6, "post bin {k}: beta={}", c.beta);
            } else {
                assert!(c.beta.abs() < 1e-6, "pre bin {k}: beta={}", c.beta);
            }
        }
        assert_eq!(r.rank, r.bins.len());
        assert_eq!(r.digest.len(), 64); // sha256 hex
    }

    #[test]
    fn rejects_length_mismatch() {
        let mut d = synth(1.0);
        d.pool.pop();
        assert!(matches!(run(&d, 10, 0.05, 1), Err(EsError::Invalid(_))));
    }

    #[test]
    fn rejects_out_of_range_group_and_single_cluster() {
        let mut d = synth(1.0);
        d.n_clusters = 1; // now cluster ids exceed range AND < 2 clusters
        assert!(matches!(run(&d, 10, 0.05, 1), Err(EsError::Invalid(_))));
    }

    #[test]
    fn rejects_bad_alpha_and_zero_boot() {
        let d = synth(1.0);
        assert!(matches!(run(&d, 10, 1.5, 1), Err(EsError::Invalid(_))));
        assert!(matches!(run(&d, 0, 0.05, 1), Err(EsError::Invalid(_))));
    }

    #[test]
    fn rank_deficient_design_is_rejected() {
        // duplicate an event-time column via extra to force collinearity
        let mut d = synth(1.0);
        let bins = bins_from(&d.event_time, &d.treated);
        let dup = event_time_columns(&d.event_time, &d.treated, &bins)[0].clone();
        d.extra.push(dup); // identical to an existing event-time column
        assert!(matches!(
            run(&d, 10, 0.05, 1),
            Err(EsError::RankDeficient { .. })
        ));
    }

    #[test]
    fn builder_enforces_contract_and_multiplicity() {
        // t1,t2 treated share the one control c1 (mult 2); t3 unmatched; c2 never used
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
                id: "t2".into(),
                treated: true,
                s: 10.1,
                tier: 3000,
                pair_class: "w".into(),
                low_exposure: false,
            },
            Unit {
                id: "t3".into(),
                treated: true,
                s: 99.0,
                tier: 3000,
                pair_class: "w".into(),
                low_exposure: false,
            },
            Unit {
                id: "c1".into(),
                treated: false,
                s: 10.05,
                tier: 3000,
                pair_class: "w".into(),
                low_exposure: true,
            },
            Unit {
                id: "c2".into(),
                treated: false,
                s: 50.0,
                tier: 3000,
                pair_class: "w".into(),
                low_exposure: true,
            },
        ];
        let m = nn_caliper_match(&units, 0.5, 1);
        assert_eq!(m.control_freq.get("c1"), Some(&2));
        assert_eq!(m.unmatched_treated, vec!["t3".to_string()]);
        // one obs per unit (week 0) for a minimal composition check
        let obs: Vec<Obs> = ["t1", "t2", "t3", "c1", "c2"]
            .iter()
            .map(|u| Obs {
                unit: u.to_string(),
                cluster_key: "w".into(),
                week_id: 0,
                treated: u.starts_with('t'),
                event_time: 0,
                y: 1.0,
            })
            .collect();
        let (data, comp) = build_matched_sample(&obs, &m, None).unwrap();
        assert_eq!(comp.matched_treated, 2); // t1,t2
        assert_eq!(comp.excluded_unmatched_treated, 1); // t3
        assert_eq!(comp.control_multiplicity.get("c1"), Some(&2));
        // t3 and c2 excluded from the estimation arrays
        assert_eq!(data.pool.len(), 3); // t1,t2,c1 (one obs each)
        // c1 carries weight 2
        let c1_ix = data.weights.iter().filter(|&&w| w == 2.0).count();
        assert_eq!(c1_ix, 1);
        assert!(data.composition.is_some());
    }

    fn units_min() -> Vec<Unit> {
        vec![
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
        ]
    }

    #[test]
    fn builder_rejects_role_mismatch() {
        let m = nn_caliper_match(&units_min(), 0.5, 1);
        // c1 is a matched control but its row is (wrongly) marked treated
        let obs = vec![
            Obs {
                unit: "t1".into(),
                cluster_key: "w".into(),
                week_id: 0,
                treated: true,
                event_time: 0,
                y: 1.0,
            },
            Obs {
                unit: "c1".into(),
                cluster_key: "w".into(),
                week_id: 0,
                treated: true,
                event_time: 0,
                y: 1.0,
            },
        ];
        assert!(matches!(
            build_matched_sample(&obs, &m, None),
            Err(EsError::Invalid(_))
        ));
    }

    #[test]
    fn builder_rejects_missing_panel_rows() {
        let m = nn_caliper_match(&units_min(), 0.5, 1);
        // control c1 has no observations => must hard-fail, not be silently absent
        let obs = vec![Obs {
            unit: "t1".into(),
            cluster_key: "w".into(),
            week_id: 0,
            treated: true,
            event_time: 0,
            y: 1.0,
        }];
        assert!(matches!(
            build_matched_sample(&obs, &m, None),
            Err(EsError::Invalid(_))
        ));
    }

    #[test]
    fn builder_rejects_incomplete_frozen_grid() {
        let m = nn_caliper_match(&units_min(), 0.5, 1);
        // expected grid has weeks {0,1}; c1 covers only week 0 -> must be rejected
        let obs = vec![
            Obs {
                unit: "t1".into(),
                cluster_key: "wt".into(),
                week_id: 0,
                treated: true,
                event_time: -1,
                y: 1.0,
            },
            Obs {
                unit: "t1".into(),
                cluster_key: "wt".into(),
                week_id: 1,
                treated: true,
                event_time: 0,
                y: 1.0,
            },
            Obs {
                unit: "c1".into(),
                cluster_key: "wc".into(),
                week_id: 0,
                treated: false,
                event_time: 0,
                y: 1.0,
            },
        ];
        let expected: BTreeSet<i64> = [0, 1].into_iter().collect();
        assert!(matches!(
            build_matched_sample(&obs, &m, Some(&expected)),
            Err(EsError::Invalid(_))
        ));
        // without the expected-grid check, the same sample is accepted (>=1 row each)
        assert!(build_matched_sample(&obs, &m, None).is_ok());
    }
}
