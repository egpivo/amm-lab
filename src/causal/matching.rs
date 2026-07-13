//! Nearest-neighbour caliper matching on the pre-period selection variable, with exact
//! strata on fee tier and pair class and low-exposure controls only. This is the frozen
//! primary matching rule of the design (caliper 0.5 log-points, `k <= 3`, with
//! replacement); it never touches post-period outcomes.

use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug)]
pub struct Unit {
    pub id: String,
    pub treated: bool,
    /// selection variable, already on the matching scale (e.g. log pre-period fee revenue)
    pub s: f64,
    pub tier: i64,
    pub pair_class: String,
    pub low_exposure: bool,
}

#[derive(Debug)]
pub struct MatchPair {
    pub treated: String,
    pub controls: Vec<String>,
    pub distances: Vec<f64>,
}

#[derive(Debug)]
pub struct MatchResult {
    pub pairs: Vec<MatchPair>,
    pub unmatched_treated: Vec<String>,
    pub controls_used: HashSet<String>,
    /// with-replacement multiplicity: how many treated units each control was matched to.
    /// This is the frequency weight the estimation sample must carry; a plain used-set
    /// would silently drop it.
    pub control_freq: HashMap<String, usize>,
}

impl MatchResult {
    pub fn match_rate(&self) -> f64 {
        let m = self.pairs.len();
        let t = m + self.unmatched_treated.len();
        if t == 0 { 0.0 } else { m as f64 / t as f64 }
    }
}

/// Match each treated unit to up to `k` nearest low-exposure controls within `caliper`,
/// exact on (tier, pair class), matching with replacement. Treated units with no control
/// inside the caliper are returned as unmatched (they carry no counterfactual).
pub fn nn_caliper_match(units: &[Unit], caliper: f64, k: usize) -> MatchResult {
    let controls: Vec<&Unit> = units
        .iter()
        .filter(|u| !u.treated && u.low_exposure)
        .collect();
    let mut pairs = Vec::new();
    let mut unmatched = Vec::new();
    let mut used = HashSet::new();
    let mut freq: HashMap<String, usize> = HashMap::new();

    for t in units.iter().filter(|u| u.treated) {
        let mut cand: Vec<(f64, &&Unit)> = controls
            .iter()
            .filter(|c| c.tier == t.tier && c.pair_class == t.pair_class)
            .map(|c| ((t.s - c.s).abs(), c))
            .filter(|(d, _)| *d <= caliper)
            .collect();
        cand.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        if cand.is_empty() {
            unmatched.push(t.id.clone());
            continue;
        }
        let chosen: Vec<(f64, &&Unit)> = cand.into_iter().take(k).collect();
        for (_, c) in &chosen {
            used.insert(c.id.clone());
            *freq.entry(c.id.clone()).or_insert(0) += 1;
        }
        pairs.push(MatchPair {
            treated: t.id.clone(),
            controls: chosen.iter().map(|(_, c)| c.id.clone()).collect(),
            distances: chosen.iter().map(|(d, _)| *d).collect(),
        });
    }
    MatchResult {
        pairs,
        unmatched_treated: unmatched,
        controls_used: used,
        control_freq: freq,
    }
}

/// Standardized mean difference between two groups (pooled-SD denominator).
pub fn smd(treated: &[f64], control: &[f64]) -> f64 {
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    let var = |v: &[f64], m: f64| {
        if v.len() < 2 {
            0.0
        } else {
            v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() as f64 - 1.0)
        }
    };
    let (mt, mc) = (mean(treated), mean(control));
    let denom = ((var(treated, mt) + var(control, mc)) / 2.0).sqrt();
    if denom == 0.0 { 0.0 } else { (mt - mc) / denom }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u(id: &str, treated: bool, s: f64, tier: i64, pc: &str, low: bool) -> Unit {
        Unit {
            id: id.into(),
            treated,
            s,
            tier,
            pair_class: pc.into(),
            low_exposure: low,
        }
    }

    #[test]
    fn caliper_and_strata_respected() {
        let units = vec![
            u("t1", true, 10.0, 3000, "weth", false),
            u("c_close", false, 10.2, 3000, "weth", true), // in caliper, right tier/class, low-exp
            u("c_far", false, 12.0, 3000, "weth", true),   // outside caliper 0.5
            u("c_wrongtier", false, 10.1, 500, "weth", true),
            u("c_exposed", false, 10.1, 3000, "weth", false), // not low-exposure
            u("c_wrongclass", false, 10.1, 3000, "btc", true),
        ];
        let r = nn_caliper_match(&units, 0.5, 3);
        assert_eq!(r.pairs.len(), 1);
        assert_eq!(r.pairs[0].controls, vec!["c_close".to_string()]);
        assert!(r.unmatched_treated.is_empty());
    }

    #[test]
    fn unmatched_when_no_control_in_caliper() {
        let units = vec![
            u("t1", true, 10.0, 3000, "weth", false),
            u("c_far", false, 20.0, 3000, "weth", true),
        ];
        let r = nn_caliper_match(&units, 0.5, 3);
        assert_eq!(r.pairs.len(), 0);
        assert_eq!(r.unmatched_treated, vec!["t1".to_string()]);
        assert_eq!(r.match_rate(), 0.0);
    }

    #[test]
    fn smd_zero_for_identical_groups() {
        assert!(smd(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]).abs() < 1e-12);
        assert!(smd(&[2.0, 3.0, 4.0], &[1.0, 2.0, 3.0]) > 0.0);
    }
}
