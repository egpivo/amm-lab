//! Assemble the *frozen design* metadata the event-study needs but the [`Panel`] does not
//! carry: per-pool treatment status / timing / cluster / matching covariate, and the frozen
//! matched-overlap set. These come from the Paper C design artifacts, not the outcome panel,
//! keeping the evidence layer (reconstruction) separate from the design layer.
//!
//! Sources:
//! - `feerev_panelvars.csv` -> treated flag, fee tier, pair class, selection variable `s`
//!   (12-month fee-revenue proxy `fr12_usd`);
//! - `ckpt_tokens.json` (pool -> `[token0, token1]`) -> token-pair `cluster_key`;
//! - `matched_pairs.json` (frozen NN-caliper match set) -> [`MatchResult`]. The match is a
//!   pre-registered design choice, so it is *loaded*, never recomputed here.
//!
//! [`Panel`]: crate::data::panel::Panel

use crate::causal::adapter::TreatmentMeta;
use crate::causal::matching::{MatchPair, MatchResult};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum MetaError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("csv: {0}")]
    Csv(#[from] csv::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Deserialize)]
struct FeerevRow {
    pool: String,
    treated: String,
    tier: String,
    class: String,
    #[serde(default)]
    fr12_usd: String,
}

/// Build `pool -> TreatmentMeta` from the fee-revenue panel and the token map.
/// `t0_week` is the (single) fee-switch treatment week (e.g. `"2025-51"`), attached to
/// treated units only. `cluster_key` is the sorted token pair; if a pool is missing from
/// the token map it falls back to its pair class so the key is never empty.
pub fn load_treatment_meta(
    feerev_csv: &Path,
    tokens_json: &Path,
    t0_week: &str,
) -> Result<HashMap<String, TreatmentMeta>, MetaError> {
    let tokens: HashMap<String, Vec<String>> =
        serde_json::from_reader(std::fs::File::open(tokens_json)?)?;

    let mut out = HashMap::new();
    let mut rdr = csv::Reader::from_path(feerev_csv)?;
    for row in rdr.deserialize() {
        let r: FeerevRow = row?;
        let treated = r.treated.trim() == "1";
        let tier = r.tier.trim().parse::<i64>().unwrap_or(0);
        let s = r.fr12_usd.trim().parse::<f64>().unwrap_or(0.0);
        let pair_class = if r.class.trim().is_empty() {
            "unknown".to_string()
        } else {
            r.class.trim().to_string()
        };
        let cluster_key = match tokens.get(&r.pool) {
            Some(t) if t.len() == 2 => {
                let (a, b) = (t[0].to_lowercase(), t[1].to_lowercase());
                if a <= b {
                    format!("{a}-{b}")
                } else {
                    format!("{b}-{a}")
                }
            }
            // no token pair on file: cluster by pair class (never empty -> passes validate)
            _ => pair_class.clone(),
        };
        out.insert(
            r.pool.clone(),
            TreatmentMeta {
                treated,
                t0_week: if treated {
                    Some(t0_week.to_string())
                } else {
                    None
                },
                cluster_key,
                s,
                tier,
                pair_class,
                low_exposure: !treated,
            },
        );
    }
    Ok(out)
}

#[derive(Debug, Deserialize)]
struct MatchedPair {
    treated: String,
    #[serde(default)]
    controls: Vec<String>,
}

/// Load the frozen matched-overlap set. `control_freq` is with-replacement multiplicity
/// (how many treated units a control serves), used downstream as the control frequency
/// weight. `unmatched_treated` = `all_treated` minus the matched treated units, so the
/// sample composition can report the excluded treated correctly.
pub fn load_matched_pairs(
    path: &Path,
    all_treated: &HashSet<String>,
) -> Result<MatchResult, MetaError> {
    let raw: Vec<MatchedPair> = serde_json::from_reader(std::fs::File::open(path)?)?;

    let mut pairs = Vec::with_capacity(raw.len());
    let mut control_freq: HashMap<String, usize> = HashMap::new();
    let mut controls_used: HashSet<String> = HashSet::new();
    let mut matched_treated: HashSet<String> = HashSet::new();

    for mp in raw {
        matched_treated.insert(mp.treated.clone());
        for c in &mp.controls {
            *control_freq.entry(c.clone()).or_insert(0) += 1;
            controls_used.insert(c.clone());
        }
        let distances = vec![0.0; mp.controls.len()]; // not carried in the frozen file
        pairs.push(MatchPair {
            treated: mp.treated,
            controls: mp.controls,
            distances,
        });
    }

    let unmatched_treated: Vec<String> = all_treated
        .iter()
        .filter(|t| !matched_treated.contains(*t))
        .cloned()
        .collect();

    Ok(MatchResult {
        pairs,
        unmatched_treated,
        controls_used,
        control_freq,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("dmeta_{}_{name}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn treatment_meta_from_feerev_and_tokens() {
        let d = tmp("meta");
        let feerev = d.join("feerev.csv");
        let mut f = std::fs::File::create(&feerev).unwrap();
        writeln!(
            f,
            "pool,treated,tier,class,fr12_usd,fr4_usd,sw12,old,covered"
        )
        .unwrap();
        writeln!(f, "0xAAA,1,3000,weth-pair,1000.5,1,1,1,1").unwrap();
        writeln!(f, "0xBBB,0,3000,weth-pair,900.0,1,1,1,1").unwrap();
        writeln!(f, "0xCCC,0,500,,,1,1,1,1").unwrap(); // empty class + empty fr12
        let tokens = d.join("tokens.json");
        // 0xBBB tokens out of order -> cluster key must be sorted; 0xCCC absent -> class fallback
        std::fs::write(
            &tokens,
            r#"{"0xAAA":["0xWETH","0xUSDC"],"0xBBB":["0xZZZ","0xAAA1"]}"#,
        )
        .unwrap();

        let m = load_treatment_meta(&feerev, &tokens, "2025-51").unwrap();

        let a = &m["0xAAA"];
        assert!(a.treated);
        assert_eq!(a.t0_week.as_deref(), Some("2025-51"));
        assert_eq!(a.tier, 3000);
        assert!((a.s - 1000.5).abs() < 1e-9);
        assert_eq!(a.pair_class, "weth-pair");
        assert_eq!(a.cluster_key, "0xusdc-0xweth"); // lowercased + sorted
        assert!(!a.low_exposure);

        let b = &m["0xBBB"];
        assert!(!b.treated);
        assert_eq!(b.t0_week, None);
        assert_eq!(b.cluster_key, "0xaaa1-0xzzz"); // sorted
        assert!(b.low_exposure);

        let c = &m["0xCCC"];
        assert_eq!(c.pair_class, "unknown"); // empty class -> unknown
        assert_eq!(c.cluster_key, "unknown"); // absent from tokens -> pair_class fallback
        assert_eq!(c.s, 0.0); // empty fr12 -> 0
        // all must pass validate (non-empty pair_class/cluster_key, finite s)
        for (id, tm) in &m {
            tm.validate(id).unwrap();
        }
    }

    #[test]
    fn matched_pairs_freq_and_unmatched() {
        let d = tmp("pairs");
        let path = d.join("mp.json");
        // c1 serves both t1 and t2 -> multiplicity 2
        std::fs::write(
            &path,
            r#"[{"treated":"t1","controls":["c1","c2"]},{"treated":"t2","controls":["c1"]}]"#,
        )
        .unwrap();
        let all_treated: HashSet<String> =
            ["t1", "t2", "t3"].iter().map(|s| s.to_string()).collect();

        let m = load_matched_pairs(&path, &all_treated).unwrap();
        assert_eq!(m.pairs.len(), 2);
        assert_eq!(m.control_freq["c1"], 2);
        assert_eq!(m.control_freq["c2"], 1);
        assert_eq!(m.controls_used.len(), 2);
        // t3 was treated but never matched
        assert_eq!(m.unmatched_treated, vec!["t3".to_string()]);
    }
}
