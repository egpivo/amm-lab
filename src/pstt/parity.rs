//! Field-specific floating comparison helpers for PSTT golden checks.

#[derive(Debug, Clone, Copy)]
pub struct FloatTol {
    pub abs: f64,
    pub rel: f64,
}

impl FloatTol {
    pub const PRICE: Self = Self {
        abs: 1e-12,
        rel: 5e-13,
    };
    pub const WEEKLY: Self = Self {
        abs: 1e-8,
        rel: 5e-12,
    };
    pub const STALENESS: Self = Self {
        abs: 1e-12,
        rel: 1e-12,
    };
}

pub fn float_close(x: f64, y: f64, tol: FloatTol) -> bool {
    if x.is_nan() || y.is_nan() {
        return false;
    }
    if x == y {
        return true;
    }
    let diff = (x - y).abs();
    diff <= tol.abs + tol.rel * x.abs().max(y.abs())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParityMismatch {
    pub key: String,
    pub field: String,
    pub detail: String,
}

pub fn compare_f64(
    key: &str,
    field: &str,
    expected: f64,
    actual: f64,
    tol: FloatTol,
    out: &mut Vec<ParityMismatch>,
) {
    if !float_close(expected, actual, tol) {
        out.push(ParityMismatch {
            key: key.to_string(),
            field: field.to_string(),
            detail: format!("expected={expected} actual={actual}"),
        });
    }
}

pub fn compare_u64(
    key: &str,
    field: &str,
    expected: u64,
    actual: u64,
    out: &mut Vec<ParityMismatch>,
) {
    if expected != actual {
        out.push(ParityMismatch {
            key: key.to_string(),
            field: field.to_string(),
            detail: format!("expected={expected} actual={actual}"),
        });
    }
}

pub fn compare_str(
    key: &str,
    field: &str,
    expected: &str,
    actual: &str,
    out: &mut Vec<ParityMismatch>,
) {
    if expected != actual {
        out.push(ParityMismatch {
            key: key.to_string(),
            field: field.to_string(),
            detail: format!("expected={expected} actual={actual}"),
        });
    }
}
