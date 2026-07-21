//! Coverage, staleness, concentration, and quantile helpers.
//!
//! Nearest-rank and linear-interpolated quantiles are intentionally separate
//! so Stage-3 staleness gates and Stage-4 diagnostics cannot be confused.

use crate::pstt::error::{PsttError, Result};
use crate::pstt::schema::PoolStalenessStats;

/// Frozen nearest-rank quantile: `sorted[clamp(ceil(p*n)-1, 0, n-1)]`.
pub fn nearest_rank_quantile(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() || !(0.0..=1.0).contains(&p) {
        return None;
    }
    let n = sorted.len();
    let idx = ((p * n as f64).ceil() as isize - 1).clamp(0, (n as isize) - 1) as usize;
    Some(sorted[idx])
}

/// NumPy-default linear quantile on a sorted sample.
pub fn linear_quantile(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() || !(0.0..=1.0).contains(&p) {
        return None;
    }
    if sorted.len() == 1 {
        return Some(sorted[0]);
    }
    let n = sorted.len() as f64;
    let pos = p * (n - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        Some(sorted[lo])
    } else {
        let w = pos - lo as f64;
        Some(sorted[lo] * (1.0 - w) + sorted[hi] * w)
    }
}

pub fn coverage(joined: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        joined as f64 / total as f64
    }
}

pub fn pool_staleness_stats(
    total_blocks: u64,
    staleness_seconds: &mut [f64],
) -> PoolStalenessStats {
    let joined = staleness_seconds.len() as u64;
    staleness_seconds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    PoolStalenessStats {
        blocks: total_blocks,
        joined,
        coverage: coverage(joined, total_blocks),
        q50_seconds: nearest_rank_quantile(staleness_seconds, 0.5),
        q90_seconds: nearest_rank_quantile(staleness_seconds, 0.9),
        q99_seconds: nearest_rank_quantile(staleness_seconds, 0.99),
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StalenessGate {
    pub coverage_min: f64,
    pub q99_seconds_max: f64,
}

impl Default for StalenessGate {
    fn default() -> Self {
        Self {
            coverage_min: 0.99,
            q99_seconds_max: 30.0,
        }
    }
}

pub fn gate_pass(stats: &PoolStalenessStats, gate: StalenessGate) -> bool {
    match stats.q99_seconds {
        Some(q99) => stats.coverage >= gate.coverage_min && q99 <= gate.q99_seconds_max,
        None => false,
    }
}

pub fn top_share(values: &[f64], k: usize) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let total: f64 = values.iter().sum();
    if total <= 0.0 {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let take = k.min(sorted.len());
    Some(sorted[..take].iter().sum::<f64>() / total)
}

pub fn require_finite(name: &str, x: f64) -> Result<f64> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(PsttError::parse(format!("non-finite {name}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_rank_matches_frozen_rule() {
        let v = [1.0, 2.0, 3.0, 4.0];
        // ceil(0.5*4)-1 = 1 -> 2.0
        assert_eq!(nearest_rank_quantile(&v, 0.5), Some(2.0));
        // ceil(0.99*4)-1 = 3 -> 4.0
        assert_eq!(nearest_rank_quantile(&v, 0.99), Some(4.0));
    }

    #[test]
    fn linear_differs_from_nearest_rank() {
        let v = [1.0, 2.0, 3.0, 4.0];
        let nearest = nearest_rank_quantile(&v, 0.5).unwrap();
        let linear = linear_quantile(&v, 0.5).unwrap();
        assert!((nearest - 2.0).abs() < 1e-12);
        assert!((linear - 2.5).abs() < 1e-12);
    }

    #[test]
    fn staleness_gate() {
        let mut vals = vec![1.0, 2.0, 3.0, 10.0];
        let stats = pool_staleness_stats(4, &mut vals);
        assert!(gate_pass(&stats, StalenessGate::default()));
        let mut bad = vec![40.0; 4];
        let stats_bad = pool_staleness_stats(4, &mut bad);
        assert!(!gate_pass(&stats_bad, StalenessGate::default()));
    }
}
