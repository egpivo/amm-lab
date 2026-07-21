//! Certification classification for signed identified sets, matching the
//! frozen standalone status rule in `build_m6_public.py` stage 2 and the
//! M5-S certification predicates. Strict inequalities are the contract:
//! an endpoint exactly at zero never certifies a sign.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrictSign {
    Positive,
    Negative,
    Unsigned,
}

/// Strict sign of an interval: `lo > 0` positive, `hi < 0` negative,
/// anything touching zero (including exactly zero endpoints) unsigned.
pub fn strict_sign(lo: f64, hi: f64) -> StrictSign {
    if lo > 0.0 {
        StrictSign::Positive
    } else if hi < 0.0 {
        StrictSign::Negative
    } else {
        StrictSign::Unsigned
    }
}

/// M5-S certification predicates over the signed region composites.
/// `cert_neg`: sup of the set confidence region `< 0`;
/// `cert_pos`: inf of the set confidence region `> 0`.
pub fn cert_neg(m_hi: (f64, f64)) -> bool {
    m_hi.1 < 0.0
}

pub fn cert_pos(m_lo: (f64, f64)) -> bool {
    m_lo.0 > 0.0
}

/// Frozen standalone grid status (`build_m6_public.py`):
/// - no usable grid interval: `POSITIVITY-LIMITED`
/// - every interval strictly signed: `IDENTIFIED`
/// - at least one strictly signed: `SENSITIVITY-DEPENDENT`
/// - otherwise: `UNDETERMINED`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentificationStatus {
    PositivityLimited,
    Identified,
    SensitivityDependent,
    Undetermined,
}

impl IdentificationStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PositivityLimited => "POSITIVITY-LIMITED",
            Self::Identified => "IDENTIFIED",
            Self::SensitivityDependent => "SENSITIVITY-DEPENDENT",
            Self::Undetermined => "UNDETERMINED",
        }
    }
}

pub fn classify_grid(intervals: &[Option<(f64, f64)>]) -> IdentificationStatus {
    let vals: Vec<(f64, f64)> = intervals.iter().filter_map(|v| *v).collect();
    if vals.is_empty() {
        return IdentificationStatus::PositivityLimited;
    }
    let signed = |v: &(f64, f64)| v.0 > 0.0 || v.1 < 0.0;
    if vals.iter().all(signed) {
        IdentificationStatus::Identified
    } else if vals.iter().any(signed) {
        IdentificationStatus::SensitivityDependent
    } else {
        IdentificationStatus::Undetermined
    }
}

/// Frozen reference-robustness rule: `STABLE` iff both references produced
/// the same non-missing status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceRobustness {
    Stable,
    ReferenceSensitive,
}

impl ReferenceRobustness {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stable => "STABLE",
            Self::ReferenceSensitive => "REFERENCE-SENSITIVE",
        }
    }
}

pub fn reference_robustness(
    primary: Option<IdentificationStatus>,
    robustness: Option<IdentificationStatus>,
) -> ReferenceRobustness {
    match (primary, robustness) {
        (Some(a), Some(b)) if a == b => ReferenceRobustness::Stable,
        _ => ReferenceRobustness::ReferenceSensitive,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_endpoints_never_certify() {
        assert_eq!(strict_sign(0.0, 1.0), StrictSign::Unsigned);
        assert_eq!(strict_sign(-1.0, 0.0), StrictSign::Unsigned);
        assert_eq!(strict_sign(f64::MIN_POSITIVE, 1.0), StrictSign::Positive);
        assert_eq!(strict_sign(-1.0, -f64::MIN_POSITIVE), StrictSign::Negative);
        assert!(!cert_neg((-1.0, 0.0)));
        assert!(cert_neg((-1.0, -1e-300)));
        assert!(!cert_pos((0.0, 1.0)));
        assert!(cert_pos((1e-300, 1.0)));
    }

    #[test]
    fn grid_status_matches_frozen_rule() {
        use IdentificationStatus::*;
        assert_eq!(classify_grid(&[None, None]), PositivityLimited);
        assert_eq!(
            classify_grid(&[Some((1.0, 2.0)), Some((0.5, 3.0))]),
            Identified
        );
        assert_eq!(
            classify_grid(&[Some((1.0, 2.0)), Some((-0.5, 3.0))]),
            SensitivityDependent
        );
        assert_eq!(classify_grid(&[Some((-1.0, 2.0)), None]), Undetermined);
        // Interval with endpoint exactly zero is not strictly signed.
        assert_eq!(classify_grid(&[Some((0.0, 2.0))]), Undetermined);
    }

    #[test]
    fn robustness_requires_equal_nonmissing() {
        use IdentificationStatus::*;
        assert_eq!(
            reference_robustness(Some(Undetermined), Some(Undetermined)),
            ReferenceRobustness::Stable
        );
        assert_eq!(
            reference_robustness(Some(Undetermined), Some(Identified)),
            ReferenceRobustness::ReferenceSensitive
        );
        assert_eq!(
            reference_robustness(None, None),
            ReferenceRobustness::ReferenceSensitive
        );
    }
}
