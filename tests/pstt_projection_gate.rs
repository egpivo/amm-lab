//! Projection/classification parity gate against a stored NumPy oracle
//! fixture (`m5s_projection_fixture_v1.json`), which transcribes the frozen
//! `mc_m5r.lfrac_ext` / `mc_m5s.d1r_signed_ranges` formulas. The frozen
//! Python programs themselves are never rerun.

use amm_lab::pstt::classification::{IdentificationStatus, cert_neg, cert_pos, classify_grid};
use amm_lab::pstt::projection::{LinearFractional, lfrac_ext};
use amm_lab::pstt::sensitivity::{
    Envelope, SignedProjection, d1r_signed_ranges, monotone_nesting_holds, signed_true,
};
use nalgebra::{DMatrix, DVector};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct Expected {
    #[serde(rename = "Mlo")]
    m_lo: [f64; 2],
    #[serde(rename = "Mhi")]
    m_hi: [f64; 2],
}

#[derive(Debug, Deserialize)]
struct Case {
    #[serde(default)]
    name: Option<String>,
    theta: [f64; 3],
    sigma: [[f64; 3]; 3],
    c: f64,
    r_bar: f64,
    env: [f64; 4],
    expected: Option<Expected>,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    randomized: Vec<Case>,
    authority: Vec<Case>,
    nesting: std::collections::BTreeMap<String, Option<Expected>>,
}

fn load_fixture() -> Fixture {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pstt/m5s_projection_fixture_v1.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("fixture readable"))
        .expect("fixture parses")
}

fn run_case(case: &Case) -> SignedProjection {
    let theta = DVector::from_column_slice(&case.theta);
    let sigma = DMatrix::from_fn(3, 3, |i, j| case.sigma[i][j]);
    let env = Envelope {
        ell_lo: case.env[0],
        ell_hi: case.env[1],
        s_lo: case.env[2],
        s_hi: case.env[3],
    };
    d1r_signed_ranges(&theta, &sigma, case.c, case.r_bar, env)
}

fn close(a: f64, b: f64) -> bool {
    let tol = 1e-9 * a.abs().max(b.abs()).max(1.0);
    (a - b).abs() <= tol
}

fn assert_case_matches(case: &Case, ctx: &str) {
    let got = run_case(case);
    match (&case.expected, got) {
        (None, SignedProjection::PositivityLimited) | (None, SignedProjection::CornerUnbounded) => {
        }
        (Some(exp), SignedProjection::Region { m_lo, m_hi }) => {
            assert!(
                close(m_lo.0, exp.m_lo[0])
                    && close(m_lo.1, exp.m_lo[1])
                    && close(m_hi.0, exp.m_hi[0])
                    && close(m_hi.1, exp.m_hi[1]),
                "{ctx}: got {m_lo:?}/{m_hi:?}, expected {:?}/{:?}",
                exp.m_lo,
                exp.m_hi
            );
        }
        (exp, got) => panic!("{ctx}: outcome mismatch, expected {exp:?}, got {got:?}"),
    }
}

#[test]
fn randomized_five_corner_cases_match_numpy_oracle() {
    let fx = load_fixture();
    assert!(fx.randomized.len() >= 200);
    let mut abstained = 0usize;
    for (i, case) in fx.randomized.iter().enumerate() {
        if case.expected.is_none() {
            abstained += 1;
        }
        assert_case_matches(case, &format!("randomized[{i}]"));
    }
    // Fixture generation kept theta_S/omega positive, so most cases project.
    assert!(abstained < fx.randomized.len());
}

#[test]
fn s1_s2_s3_authority_cases_match() {
    let fx = load_fixture();
    let names: Vec<&str> = fx
        .authority
        .iter()
        .filter_map(|c| c.name.as_deref())
        .collect();
    for want in [
        "S1_negative_set",
        "S2_zero_straddle",
        "S3_denom_near_zero",
        "near_degenerate_r1",
    ] {
        assert!(names.contains(&want), "missing authority case {want}");
    }
    for case in &fx.authority {
        let name = case.name.as_deref().unwrap_or("?");
        assert_case_matches(case, name);
        match name {
            // S1: whole set strictly negative -> negative certification.
            "S1_negative_set" => {
                let (m_lo, m_hi) = run_case(case).region().unwrap();
                assert!(cert_neg(m_hi), "S1 must certify negative");
                assert!(!cert_pos(m_lo));
            }
            // S2: straddles zero -> no certification either way.
            "S2_zero_straddle" => {
                let (m_lo, m_hi) = run_case(case).region().unwrap();
                assert!(!cert_neg(m_hi) && !cert_pos(m_lo));
            }
            // S3: denominator degenerate -> abstention, not an interval.
            "S3_denom_near_zero" => {
                assert!(!run_case(case).is_usable());
            }
            _ => {}
        }
    }
}

#[test]
fn monotone_nesting_in_r_bar_matches_oracle_and_holds() {
    let fx = load_fixture();
    let mut regions = Vec::new();
    for (key, exp) in &fx.nesting {
        let rb: f64 = key.parse().unwrap();
        let exp = exp.as_ref().expect("nesting cell projects");
        regions.push((rb, ((exp.m_lo[0], exp.m_lo[1]), (exp.m_hi[0], exp.m_hi[1]))));
        // and the Rust projection reproduces each grid point
        let case = Case {
            name: None,
            theta: [-50.0, 20.0, 1.0],
            sigma: [[4.0, 0.5, 0.1], [0.5, 2.0, 0.05], [0.1, 0.05, 0.02]],
            c: 7.81,
            r_bar: rb,
            env: [-8.0, -4.0, 10.0, 30.0],
            expected: None,
        };
        let got = run_case(&case).region().unwrap();
        assert!(close(got.0.0, exp.m_lo[0]) && close(got.1.1, exp.m_hi[1]));
    }
    assert!(
        monotone_nesting_holds(&regions),
        "outer endpoints must nest"
    );
}

#[test]
fn denominator_screen_boundary_cases() {
    // Screen quantity is theta_S/omega. Construct Sigma so its Fieller lower
    // endpoint sits exactly at / near zero.
    let sigma = DMatrix::from_fn(3, 3, |i, j| if i == j { [0.0, 1.0, 1e-12][i] } else { 0.0 });
    let env = Envelope {
        ell_lo: -1.0,
        ell_hi: 1.0,
        s_lo: 1.0,
        s_hi: 2.0,
    };
    // theta_S = 2, var = 1, c = 4 -> denominator CI [0,4] touches zero
    // inside lfrac_ext -> abstain (inclusive rule).
    let theta = DVector::from_column_slice(&[1.0, 2.0, 1.0]);
    let touching = d1r_signed_ranges(&theta, &sigma, 4.0, 1.0, env);
    assert_eq!(touching, SignedProjection::PositivityLimited);
    // c slightly smaller -> strictly positive screen -> usable.
    let inside = d1r_signed_ranges(&theta, &sigma, 4.0 - 1e-6, 1.0, env);
    assert!(inside.is_usable());
    // Negative theta_S: screen never passes.
    let neg = DVector::from_column_slice(&[1.0, -2.0, 1.0]);
    assert_eq!(
        d1r_signed_ranges(&neg, &sigma, 0.01, 1.0, env),
        SignedProjection::PositivityLimited
    );
}

#[test]
fn screen_lower_endpoint_exactly_zero_abstains() {
    // Direct lfrac_ext check: lower endpoint == 0 must NOT pass the strict
    // `> 0` screen used by d1r_signed_ranges.
    let theta = DVector::from_column_slice(&[0.0, 2.0, 1.0]);
    let sigma = DMatrix::from_fn(3, 3, |i, j| if i == j { [1.0, 1.0, 1e-18][i] } else { 0.0 });
    // theta_S/omega ~ 2 with sd 1: c = 4 -> lower endpoint ~ 0.
    let screen = lfrac_ext(
        &[0.0, 1.0, 0.0],
        0.0,
        &[0.0, 0.0, 1.0],
        0.0,
        &theta,
        &sigma,
        4.0,
    );
    match screen {
        LinearFractional::Projected { lo, .. } => {
            assert!(lo.abs() < 1e-6, "constructed boundary case, lo={lo}");
            let env = Envelope {
                ell_lo: -1.0,
                ell_hi: 1.0,
                s_lo: 1.0,
                s_hi: 2.0,
            };
            let out = d1r_signed_ranges(&theta, &sigma, 4.0, 1.0, env);
            if lo <= 0.0 {
                assert_eq!(out, SignedProjection::PositivityLimited);
            }
        }
        LinearFractional::DenominatorTouchesZero => {}
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn certification_boundary_inequalities_are_strict() {
    // Interval endpoints exactly at zero never certify and never count as
    // signed in the grid status.
    assert!(!cert_neg((-2.0, 0.0)));
    assert!(!cert_pos((0.0, 2.0)));
    assert_eq!(
        classify_grid(&[Some((0.0, 1.0)), Some((-1.0, 0.0))]),
        IdentificationStatus::Undetermined
    );
    assert_eq!(
        classify_grid(&[Some((1e-300, 1.0)), Some((2.0, 3.0))]),
        IdentificationStatus::Identified
    );
    assert_eq!(
        classify_grid(&[Some((1e-300, 1.0)), Some((-1.0, 1.0))]),
        IdentificationStatus::SensitivityDependent
    );
    assert_eq!(
        classify_grid(&[None]),
        IdentificationStatus::PositivityLimited
    );
}

#[test]
fn signed_true_agrees_with_projection_center_for_tiny_ellipsoid() {
    // With a vanishing ellipsoid the projected set collapses to the plug-in
    // corner values, whose min/max is signed_true at the theta ratios.
    let mu_l = -50.0;
    let mu_s = 20.0;
    let env = Envelope {
        ell_lo: -8.0,
        ell_hi: -4.0,
        s_lo: 10.0,
        s_hi: 30.0,
    };
    let theta = DVector::from_column_slice(&[mu_l, mu_s, 1.0]);
    let sigma = DMatrix::from_fn(3, 3, |i, j| if i == j { 1e-18 } else { 0.0 });
    let (m_lo, m_hi) = d1r_signed_ranges(&theta, &sigma, 1.0, 1.0, env)
        .region()
        .unwrap();
    let (t_lo, t_hi) = signed_true(mu_l, mu_s, 1.0, env);
    assert!((m_lo.0 - t_lo).abs() < 1e-6);
    assert!((m_hi.1 - t_hi).abs() < 1e-6);
}
