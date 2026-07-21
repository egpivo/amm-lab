//! Synchronized calendar moving-block bootstrap.
//!
//! Index construction mirrors the frozen `moving_block_indices` rule of
//! `build_m6_public.py`: noncircular blocks with uniform starts in
//! `[0, n-bl]`, concatenated then truncated to `n`.
//!
//! IMPORTANT PROVENANCE LIMIT: the frozen standalone runner derived its
//! NumPy seeds through Python's salted built-in `hash(...)`, and the frozen
//! manifest records no `PYTHONHASHSEED`. This module therefore never claims
//! bitwise parity with the historical draw stream. It supports:
//!  - validation and consumption of an externally serialized index matrix;
//!  - deterministic Rust-side generation (rand `StdRng`, explicitly NOT a
//!    NumPy PCG64 stream) for future runs on a new draw schedule.

use crate::pstt::error::{PsttError, Result};
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Frozen block-length rule: `max(2, round(sqrt(n_weeks)))`.
pub fn block_length(n_weeks: usize) -> usize {
    ((n_weeks as f64).sqrt().round() as usize).max(2)
}

/// One draw of moving-block indices (frozen construction rule).
pub fn moving_block_indices<R: Rng>(rng: &mut R, n: usize, bl: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(n + bl);
    let hi = n.saturating_sub(bl).max(0) + 1; // uniform start in [0, n-bl]
    while out.len() < n {
        let s = rng.gen_range(0..hi.max(1));
        for i in s..(s + bl).min(n) {
            out.push(i);
        }
    }
    out.truncate(n);
    out
}

/// A synchronized index schedule: `draws x n_weeks`, one schedule reused by
/// every pool and reference in a run.
#[derive(Debug, Clone)]
pub struct IndexSchedule {
    pub indices: Vec<Vec<usize>>,
    pub n_weeks: usize,
}

impl IndexSchedule {
    /// Validate an externally supplied matrix: every draw has the calendar
    /// length and every index is in range.
    pub fn from_matrix(indices: Vec<Vec<usize>>, n_weeks: usize) -> Result<Self> {
        if indices.is_empty() {
            return Err(PsttError::invariant("empty bootstrap index matrix"));
        }
        for (d, draw) in indices.iter().enumerate() {
            if draw.len() != n_weeks {
                return Err(PsttError::invariant(format!(
                    "draw {d} has length {} != calendar length {n_weeks}",
                    draw.len()
                )));
            }
            if let Some(bad) = draw.iter().find(|&&i| i >= n_weeks) {
                return Err(PsttError::invariant(format!(
                    "draw {d} contains out-of-range index {bad}"
                )));
            }
        }
        Ok(Self { indices, n_weeks })
    }

    /// Deterministic Rust-side generation. Explicitly NOT NumPy PCG64 parity.
    pub fn generate(seed: u64, draws: usize, n_weeks: usize, bl: usize) -> Result<Self> {
        if draws == 0 || n_weeks == 0 || bl == 0 {
            return Err(PsttError::invariant("zero-size bootstrap request"));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let indices = (0..draws)
            .map(|_| moving_block_indices(&mut rng, n_weeks, bl))
            .collect();
        Ok(Self { indices, n_weeks })
    }

    pub fn draws(&self) -> usize {
        self.indices.len()
    }

    /// Consume the schedule against weekly component arrays, producing the
    /// `draws x components` matrix of resampled sums (frozen stage-2 layout).
    pub fn stacked_sums(&self, components: &[&[f64]]) -> Result<DMatrix<f64>> {
        for comp in components {
            if comp.len() != self.n_weeks {
                return Err(PsttError::invariant(format!(
                    "component length {} != calendar length {}",
                    comp.len(),
                    self.n_weeks
                )));
            }
        }
        let b = self.draws();
        let d = components.len();
        let mut star = DMatrix::zeros(b, d);
        for (row, draw) in self.indices.iter().enumerate() {
            for (col, comp) in components.iter().enumerate() {
                let mut acc = 0.0;
                for &i in draw {
                    acc += comp[i];
                }
                star[(row, col)] = acc;
            }
        }
        Ok(star)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_length_rule() {
        assert_eq!(block_length(1), 2);
        assert_eq!(block_length(4), 2);
        assert_eq!(block_length(104), 10); // sqrt(104)=10.198 -> round 10
        assert_eq!(block_length(100), 10);
    }

    #[test]
    fn indices_cover_and_truncate() {
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..50 {
            let idx = moving_block_indices(&mut rng, 10, 3);
            assert_eq!(idx.len(), 10);
            assert!(idx.iter().all(|&i| i < 10));
        }
    }

    #[test]
    fn matrix_validation_rejects_bad_shapes() {
        assert!(IndexSchedule::from_matrix(vec![vec![0, 1]], 3).is_err());
        assert!(IndexSchedule::from_matrix(vec![vec![0, 1, 3]], 3).is_err());
        assert!(IndexSchedule::from_matrix(vec![vec![0, 1, 2]], 3).is_ok());
    }

    #[test]
    fn synchronized_consumption_is_shared() {
        // Same schedule applied to two "pools" gives identical index selections.
        let sched = IndexSchedule::from_matrix(vec![vec![1, 1, 0], vec![2, 0, 2]], 3).unwrap();
        let l1 = [10.0, 20.0, 30.0];
        let l2 = [1.0, 2.0, 3.0];
        let star = sched.stacked_sums(&[&l1, &l2]).unwrap();
        assert_eq!(star.nrows(), 2);
        // draw0: idx (1,1,0) -> 20+20+10=50; 2+2+1=5
        assert!((star[(0, 0)] - 50.0).abs() < 1e-12);
        assert!((star[(0, 1)] - 5.0).abs() < 1e-12);
        // draw1: idx (2,0,2) -> 30+10+30=70; 3+1+3=7
        assert!((star[(1, 0)] - 70.0).abs() < 1e-12);
        assert!((star[(1, 1)] - 7.0).abs() < 1e-12);
    }
}
