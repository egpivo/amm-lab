//! Standalone M6-style application composition (weekly records -> signed
//! regions -> ranking status), mirroring the frozen `build_m6_public.py`
//! stage-2 formulas. This is additive parity tooling: it never regenerates
//! or replaces the frozen historical result, and it does not claim bitwise
//! bootstrap parity with the unrecoverable Python hash-salted seed stream.

use crate::pstt::bootstrap::{IndexSchedule, block_length};
use crate::pstt::classification::{IdentificationStatus, classify_grid};
use crate::pstt::error::{PsttError, Result};
use crate::pstt::projection::{RidgePolicy, ellipsoid_from_draws};
use crate::pstt::sensitivity::{
    Envelope, SignedProjection, contrast_interval, d1r_signed_ranges, envelope_from_weekly,
};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// One weekly primitive record (a frozen `m6_public_weekly.json` row).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeeklyRecord {
    pub week: String,
    #[serde(rename = "L")]
    pub l: f64,
    #[serde(rename = "A")]
    pub a: f64,
    #[serde(rename = "B")]
    pub b: f64,
    #[serde(rename = "S")]
    pub s: f64,
    #[serde(rename = "Om")]
    pub om: f64,
    pub q2: f64,
    pub n: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stale_med: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PoolInference {
    pub m_point: Option<f64>,
    pub n_weeks: usize,
    pub block_len: usize,
    pub n_eff: f64,
    pub env: (f64, f64, f64, f64),
    /// r_bar (as string key, frozen convention) -> [[Mlo.lo,Mlo.hi],[Mhi.lo,Mhi.hi]]
    pub outer_regions: BTreeMap<String, Option<[[f64; 2]; 2]>>,
}

/// Stable per-(label, reference) seed derivation for future Rust runs.
/// Documented explicitly: this is NOT the historical Python `hash(...)`
/// derivation, which is unrecoverable without `PYTHONHASHSEED`.
pub fn derive_seed(base_seed: u64, label: &str, reference: &str) -> u64 {
    let mut h = Sha256::new();
    h.update(base_seed.to_be_bytes());
    h.update(label.as_bytes());
    h.update([0x1f]);
    h.update(reference.as_bytes());
    let d = h.finalize();
    u64::from_be_bytes(d[..8].try_into().expect("8 bytes"))
}

pub fn frozen_r_bar_grid() -> Vec<f64> {
    vec![0.0, 0.05, 0.10, 0.50, 1.0]
}

/// Frozen n_eff rule: `(sum S)^2 / sum q2`.
pub fn n_eff(records: &[WeeklyRecord]) -> f64 {
    let tot_s: f64 = records.iter().map(|r| r.s).sum();
    let tot_q2: f64 = records.iter().map(|r| r.q2).sum();
    if tot_q2 > 0.0 {
        tot_s * tot_s / tot_q2
    } else {
        0.0
    }
}

fn grid_key(r_bar: f64) -> String {
    // Match Python str(float) for the frozen grid values.
    if r_bar == 0.0 {
        "0.0".to_string()
    } else if (r_bar - 0.05).abs() < 1e-12 {
        "0.05".to_string()
    } else if (r_bar - 0.10).abs() < 1e-12 {
        "0.1".to_string()
    } else if (r_bar - 0.50).abs() < 1e-12 {
        "0.5".to_string()
    } else if (r_bar - 1.0).abs() < 1e-12 {
        "1.0".to_string()
    } else {
        format!("{r_bar}")
    }
}

/// Frozen stage-2 per-pool signed region construction over an explicit
/// bootstrap schedule (synchronized across components within the pool).
pub fn pool_signed_regions(
    records: &[WeeklyRecord],
    schedule: &IndexSchedule,
    r_bar_grid: &[f64],
    nominal: f64,
) -> Result<PoolInference> {
    if records.is_empty() {
        return Err(PsttError::invariant("no weekly records for pool"));
    }
    let n_weeks = records.len();
    if schedule.n_weeks != n_weeks {
        return Err(PsttError::invariant(format!(
            "schedule length {} != weekly records {n_weeks}",
            schedule.n_weeks
        )));
    }
    let l: Vec<f64> = records.iter().map(|r| r.l).collect();
    let s: Vec<f64> = records.iter().map(|r| r.s).collect();
    let om: Vec<f64> = records.iter().map(|r| r.om).collect();
    let env = envelope_from_weekly(&l, &s, &om)?;
    let theta = DVector::from_column_slice(&[
        l.iter().sum::<f64>(),
        s.iter().sum::<f64>(),
        om.iter().sum::<f64>(),
    ]);
    let star: DMatrix<f64> = schedule.stacked_sums(&[&l, &s, &om])?;
    let region = ellipsoid_from_draws(&star, &theta, RidgePolicy::StandaloneM6, nominal)
        .map_err(|e| PsttError::invariant(format!("ellipsoid failure: {e:?}")))?;
    let mut outer_regions = BTreeMap::new();
    for &rb in r_bar_grid {
        let proj = d1r_signed_ranges(
            &region.theta,
            &region.sigma,
            region.c_alpha,
            rb,
            Envelope {
                ell_lo: env.ell_lo,
                ell_hi: env.ell_hi,
                s_lo: env.s_lo.max(0.0),
                s_hi: env.s_hi,
            },
        );
        let entry = proj
            .region()
            .map(|(m_lo, m_hi)| [[m_lo.0, m_lo.1], [m_hi.0, m_hi.1]]);
        outer_regions.insert(grid_key(rb), entry);
    }
    let tot_s: f64 = s.iter().sum();
    Ok(PoolInference {
        m_point: if tot_s > 0.0 {
            Some(l.iter().sum::<f64>() / tot_s)
        } else {
            None
        },
        n_weeks,
        block_len: block_length(n_weeks),
        n_eff: n_eff(records),
        env: (env.ell_lo, env.ell_hi, env.s_lo, env.s_hi),
        outer_regions,
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct RankingInference {
    pub n_common_weeks: usize,
    pub block_len: usize,
    /// r_bar key -> Some([lo, hi]) or None when either side abstains.
    pub i_delta_grid: BTreeMap<String, Option<[f64; 2]>>,
    pub status: String,
}

/// Frozen ranking: align on common weeks, resample the same week blocks for
/// both pools (6-dim stacked), project marginal 3x3 blocks with the joint
/// `c_alpha`, and compose `I_Delta` per grid point.
pub fn ranking_signed(
    lower: &[WeeklyRecord],
    higher: &[WeeklyRecord],
    schedule: &IndexSchedule,
    r_bar_grid: &[f64],
    nominal: f64,
) -> Result<RankingInference> {
    let by_week_l: BTreeMap<&str, &WeeklyRecord> =
        lower.iter().map(|r| (r.week.as_str(), r)).collect();
    let by_week_h: BTreeMap<&str, &WeeklyRecord> =
        higher.iter().map(|r| (r.week.as_str(), r)).collect();
    let common: Vec<&str> = by_week_l
        .keys()
        .filter(|w| by_week_h.contains_key(**w))
        .copied()
        .collect();
    let n_w = common.len();
    if n_w < 8 {
        return Err(PsttError::invariant(format!(
            "fewer than 8 common weeks: {n_w}"
        )));
    }
    if schedule.n_weeks != n_w {
        return Err(PsttError::invariant(format!(
            "schedule length {} != common weeks {n_w}",
            schedule.n_weeks
        )));
    }
    let pick = |m: &BTreeMap<&str, &WeeklyRecord>, f: fn(&WeeklyRecord) -> f64| -> Vec<f64> {
        common.iter().map(|w| f(m[*w])).collect()
    };
    let l5 = pick(&by_week_l, |r| r.l);
    let s5 = pick(&by_week_l, |r| r.s);
    let o5 = pick(&by_week_l, |r| r.om);
    let l3 = pick(&by_week_h, |r| r.l);
    let s3 = pick(&by_week_h, |r| r.s);
    let o3 = pick(&by_week_h, |r| r.om);
    let theta = DVector::from_column_slice(&[
        l5.iter().sum(),
        s5.iter().sum(),
        o5.iter().sum(),
        l3.iter().sum(),
        s3.iter().sum(),
        o3.iter().sum(),
    ]);
    let star = schedule.stacked_sums(&[&l5, &s5, &o5, &l3, &s3, &o3])?;
    let region = ellipsoid_from_draws(&star, &theta, RidgePolicy::StandaloneM6, nominal)
        .map_err(|e| PsttError::invariant(format!("ellipsoid failure: {e:?}")))?;

    let env5 = envelope_from_weekly(&l5, &s5, &o5)?;
    let env3 = envelope_from_weekly(&l3, &s3, &o3)?;
    let sub = |range: std::ops::Range<usize>| -> (DVector<f64>, DMatrix<f64>) {
        let th = DVector::from_iterator(3, range.clone().map(|i| region.theta[i]));
        let mut sig = DMatrix::zeros(3, 3);
        for (a, i) in range.clone().enumerate() {
            for (b, j) in range.clone().enumerate() {
                sig[(a, b)] = region.sigma[(i, j)];
            }
        }
        (th, sig)
    };
    let (th5, sig5) = sub(0..3);
    let (th3, sig3) = sub(3..6);

    let mut i_delta_grid = BTreeMap::new();
    for &rb in r_bar_grid {
        let r5 = d1r_signed_ranges(&th5, &sig5, region.c_alpha, rb, env5);
        let r3 = d1r_signed_ranges(&th3, &sig3, region.c_alpha, rb, env3);
        let entry = match (r5, r3) {
            (
                SignedProjection::Region {
                    m_lo: lo5,
                    m_hi: hi5,
                },
                SignedProjection::Region {
                    m_lo: lo3,
                    m_hi: hi3,
                },
            ) => {
                let (lo, hi) = contrast_interval(lo5, hi5, lo3, hi3);
                Some([lo, hi])
            }
            _ => None,
        };
        i_delta_grid.insert(grid_key(rb), entry);
    }
    let intervals: Vec<Option<(f64, f64)>> = i_delta_grid
        .values()
        .map(|v| v.map(|a| (a[0], a[1])))
        .collect();
    let status = classify_grid(&intervals);
    Ok(RankingInference {
        n_common_weeks: n_w,
        block_len: block_length(n_w),
        i_delta_grid,
        status: status.as_str().to_string(),
    })
}

pub fn status_from_str(s: &str) -> Option<IdentificationStatus> {
    match s {
        "POSITIVITY-LIMITED" => Some(IdentificationStatus::PositivityLimited),
        "IDENTIFIED" => Some(IdentificationStatus::Identified),
        "SENSITIVITY-DEPENDENT" => Some(IdentificationStatus::SensitivityDependent),
        "UNDETERMINED" => Some(IdentificationStatus::Undetermined),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(week: &str, l: f64, s: f64, om: f64) -> WeeklyRecord {
        WeeklyRecord {
            week: week.into(),
            l,
            a: l.max(0.0),
            b: (-l).max(0.0),
            s,
            om,
            q2: s * s / om.max(1.0),
            n: om as u64,
            stale_med: None,
        }
    }

    #[test]
    fn seed_derivation_is_stable_and_distinct() {
        let a = derive_seed(990_000_000, "5bp", "last_trade");
        let b = derive_seed(990_000_000, "5bp", "vwap1s");
        let c = derive_seed(990_000_000, "30bp", "last_trade");
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_eq!(a, derive_seed(990_000_000, "5bp", "last_trade"));
    }

    #[test]
    fn pool_regions_run_on_stable_synthetic_panel() {
        let records: Vec<WeeklyRecord> = (0..25)
            .map(|i| {
                rec(
                    &format!("2024-{:02}", i + 1),
                    -50.0 + (i as f64 % 5.0),
                    20.0 + (i as f64 % 3.0),
                    10.0,
                )
            })
            .collect();
        let sched = IndexSchedule::generate(7, 400, 25, block_length(25)).unwrap();
        let inf = pool_signed_regions(&records, &sched, &frozen_r_bar_grid(), 0.95).unwrap();
        assert_eq!(inf.n_weeks, 25);
        assert_eq!(inf.block_len, 5);
        assert!(inf.m_point.unwrap() < 0.0);
        // r=0 region present and negative for this strongly negative panel.
        let r0 = inf.outer_regions.get("0.0").unwrap().unwrap();
        assert!(r0[1][1] < 0.0, "sup of set should be negative");
        // monotone nesting of outer endpoints across the frozen grid
        let mut prev: Option<[[f64; 2]; 2]> = None;
        for key in ["0.0", "0.05", "0.1", "0.5", "1.0"] {
            let cur = inf.outer_regions.get(key).unwrap().unwrap();
            if let Some(p) = prev {
                assert!(cur[0][0] <= p[0][0] + 1e-9);
                assert!(cur[1][1] >= p[1][1] - 1e-9);
            }
            prev = Some(cur);
        }
    }
}
