//! External CEX / oracle mid-price process (seeded GBM) plus rolling
//! realized volatility used as a state feature.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

pub fn gbm_path(n_steps: usize, s0: f64, mu: f64, sigma: f64, dt: f64, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let z = Normal::new(0.0, 1.0).unwrap();
    let mut prices = Vec::with_capacity(n_steps + 1);
    prices.push(s0);
    for _ in 0..n_steps {
        let shock: f64 = z.sample(&mut rng);
        let next = prices.last().unwrap()
            * ((mu - 0.5 * sigma * sigma) * dt + sigma * dt.sqrt() * shock).exp();
        prices.push(next);
    }
    prices
}

/// Realized volatility of log returns over the trailing `window` steps,
/// annualized by 1/sqrt(dt). Returns 0.0 until enough history exists.
pub fn rolling_vol(prices: &[f64], t: usize, window: usize, dt: f64) -> f64 {
    if t < 2 {
        return 0.0;
    }
    let start = t.saturating_sub(window);
    let rets: Vec<f64> = (start + 1..=t)
        .map(|i| (prices[i] / prices[i - 1]).ln())
        .collect();
    if rets.len() < 2 {
        return 0.0;
    }
    let mean = rets.iter().sum::<f64>() / rets.len() as f64;
    let var = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (rets.len() - 1) as f64;
    (var / dt).sqrt()
}

// ---------------------------------------------------------------------------
// VTR-1 scenario price generators (Gate 1; docs: VTR1_SCENARIO_SPEC.md §§2-6).
// `PathSpec::Baseline` is the default and routes to `gbm_path` unchanged.
// Scenarios reuse the baseline oracle stream for z_t and draw extra
// randomness (xi, eta) from separate seed-keyed streams, so the noise, arb,
// order and JIT streams of `ExecEnv::reset` are untouched.
// ---------------------------------------------------------------------------

use serde::{Deserialize, Serialize};
use statrs::distribution::{Continuous, ContinuousCDF, Normal as StatNormal, StudentsT};

/// (multiplier, offset) of the episode-volatility (xi) stream: seed*K + c, wrapping u64.
pub const XI_STREAM: (u64, u64) = (1_597_334_677, 5);
/// (multiplier, offset) of the volatility-shock (eta) stream.
pub const ETA_STREAM: (u64, u64) = (374_761_393, 6);

fn stream_seed(seed: u64, (k, c): (u64, u64)) -> u64 {
    seed.wrapping_mul(k).wrapping_add(c)
}

/// Bounded heavy-tailed innovation: Student-t(nu) truncated at `cap` standardized
/// units and re-standardized (spec §5).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TInnov {
    pub nu: f64,
    pub cap: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VolState {
    /// Log-volatility sd s (Group 2: truncated-normal xi; Group 3: AR(1) state).
    pub s: f64,
    /// Group 3 only: e-folding time in steps (minutes). None = Group 2.
    pub tau: Option<f64>,
    /// Group 2 only: truncation of xi.
    pub k: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub enum PathSpec {
    /// Constant-sigma GBM from `EnvConfig::sigma` (Group 1 = baseline with another sigma).
    #[default]
    Baseline,
    /// Groups 2, 3 and Group 5 candidates: volatility state, optional t innovations.
    Scenario {
        sigma_bar: f64,
        vol: Option<VolState>,
        t: Option<TInnov>,
    },
}

impl PathSpec {
    /// Parameter constraints of the scenario specification (§§3-5): a spec that fails here is
    /// rejected, never repaired or replaced by a default.
    pub fn validate(&self) -> Result<(), String> {
        let pos = |name: &str, x: f64| {
            if x.is_finite() && x > 0.0 {
                Ok(())
            } else {
                Err(format!("{name} must be finite and > 0, got {x}"))
            }
        };
        let PathSpec::Scenario { sigma_bar, vol, t } = self else {
            return Ok(());
        };
        pos("sigma_bar", *sigma_bar)?;
        if let Some(v) = vol {
            if !(v.s.is_finite() && v.s >= 0.0) {
                return Err(format!("s must be finite and >= 0, got {}", v.s));
            }
            pos("k", v.k)?;
            if let Some(tau) = v.tau {
                pos("tau", tau)?;
            }
        }
        if let Some(t) = t {
            if !(t.nu.is_finite() && t.nu > 2.0) {
                return Err(format!("nu must be finite and > 2, got {}", t.nu));
            }
            pos("cap", t.cap)?;
        }
        Ok(())
    }
}

/// Quantities the W0 harness logs next to the returns.
#[derive(Debug, Clone, Default)]
pub struct PathTrace {
    pub xi: Option<f64>,
    pub sigma_t: Vec<f64>,
    pub x_t: Vec<f64>,
    pub z_t: Vec<f64>,
    pub eta_t: Vec<f64>,
    pub eps_t: Vec<f64>,
}

thread_local! {
    /// Per-thread count of `TruncT::new` constructions (tests assert caching).
    pub static TRUNC_T_BUILDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Truncated standardized t: constants needed by the generator.
pub struct TruncT {
    dist: StudentsT,
    norm: StatNormal,
    p_lo: f64,
    p_hi: f64,
    pub sd: f64,
    pub cap_raw: f64,
    /// raw moments of the standardized truncated law, orders 0..=MOM
    moments: [f64; MOM + 1],
}

const MOM: usize = 10;

impl TruncT {
    pub fn new(t: TInnov) -> Self {
        TRUNC_T_BUILDS.with(|c| c.set(c.get() + 1));
        let dist = StudentsT::new(0.0, 1.0, t.nu).expect("nu > 0");
        let norm = StatNormal::new(0.0, 1.0).unwrap();
        let k = t.cap * (t.nu / (t.nu - 2.0)).sqrt();
        let (p_lo, p_hi) = (dist.cdf(-k), dist.cdf(k));
        // composite Simpson over [-k, k] on a fine even grid
        let n = 400_000usize;
        let h = 2.0 * k / n as f64;
        let w = |i: usize| {
            if i == 0 || i == n {
                1.0
            } else if i % 2 == 1 {
                4.0
            } else {
                2.0
            }
        };
        let mass: f64 = (0..=n)
            .map(|i| w(i) * dist.pdf(-k + i as f64 * h))
            .sum::<f64>()
            * h
            / 3.0;
        let m2: f64 = (0..=n)
            .map(|i| {
                let x = -k + i as f64 * h;
                w(i) * x * x * dist.pdf(x)
            })
            .sum::<f64>()
            * h
            / 3.0
            / mass;
        let sd = m2.sqrt();
        let mut moments = [0.0; MOM + 1];
        for (j, m) in moments.iter_mut().enumerate() {
            *m = (0..=n)
                .map(|i| {
                    let x = -k + i as f64 * h;
                    w(i) * (x / sd).powi(j as i32) * dist.pdf(x)
                })
                .sum::<f64>()
                * h
                / 3.0
                / mass;
        }
        Self {
            dist,
            norm,
            p_lo,
            p_hi,
            sd,
            cap_raw: k,
            moments,
        }
    }

    /// Strictly increasing map of z (spec §5): no clipping, no ties.
    pub fn eps(&self, z: f64) -> f64 {
        let u = self.p_lo + self.norm.cdf(z) * (self.p_hi - self.p_lo);
        self.dist.inverse_cdf(u) / self.sd
    }

    /// m(a) = ln E exp(a * eps) from the moment series, accurate to relative ~1e-12
    /// for |a| <= 0.01 (higher-order remainder < 1e-20); ln_1p avoids cancellation.
    pub fn log_mgf(&self, a: f64) -> f64 {
        let mut term = 1.0;
        let mut rest = 0.0;
        for j in 1..=MOM {
            term *= a / j as f64;
            rest += term * self.moments[j];
        }
        rest.ln_1p()
    }

    /// Independent check value: direct quadrature of E[expm1(a*eps)] on a different grid.
    pub fn log_mgf_quadrature(&self, a: f64, n: usize) -> f64 {
        let k = self.cap_raw;
        let n = n + n % 2;
        let h = 2.0 * k / n as f64;
        let w = |i: usize| {
            if i == 0 || i == n {
                1.0
            } else if i % 2 == 1 {
                4.0
            } else {
                2.0
            }
        };
        let mass: f64 = (0..=n)
            .map(|i| w(i) * self.dist.pdf(-k + i as f64 * h))
            .sum();
        let num: f64 = (0..=n)
            .map(|i| {
                let x = -k + i as f64 * h;
                w(i) * (a * x / self.sd).exp_m1() * self.dist.pdf(x)
            })
            .sum();
        (num / mass).ln_1p()
    }

    pub fn truncation_mass(&self) -> f64 {
        self.p_lo + (1.0 - self.p_hi)
    }

    pub fn moment(&self, j: usize) -> f64 {
        self.moments[j]
    }
}

/// Precomputed scenario generator (build once per spec, reuse across episodes).
pub struct ScenarioGen {
    spec: PathSpec,
    trunc_t: Option<TruncT>,
    norm: StatNormal,
}

impl ScenarioGen {
    pub fn new(spec: PathSpec) -> Self {
        let trunc_t = match spec {
            PathSpec::Scenario { t: Some(t), .. } => Some(TruncT::new(t)),
            _ => None,
        };
        Self {
            spec,
            trunc_t,
            norm: StatNormal::new(0.0, 1.0).unwrap(),
        }
    }

    /// Spec §3: c_s = 1/2 ln( exp(2 s^2) [Phi(k-2s) - Phi(-k-2s)] / [Phi(k) - Phi(-k)] ).
    pub fn episode_correction(&self, s: f64, k: f64) -> f64 {
        let n = &self.norm;
        let num = n.cdf(k - 2.0 * s) - n.cdf(-k - 2.0 * s);
        let den = n.cdf(k) - n.cdf(-k);
        0.5 * ((2.0 * s * s).exp() * num / den).ln()
    }

    pub fn spec(&self) -> PathSpec {
        self.spec
    }

    pub fn trunc_t(&self) -> Option<&TruncT> {
        self.trunc_t.as_ref()
    }

    /// Drift correction m_t = ln E exp(sigma_t sqrt(dt) eps): closed form 0.5 sigma_t^2 dt for
    /// Gaussian eps, the truncated-law cumulant otherwise (spec §§1, 5).
    pub fn drift_constant(&self, sigma_t: f64, dt: f64) -> f64 {
        match &self.trunc_t {
            Some(tt) => tt.log_mgf(sigma_t * dt.sqrt()),
            None => 0.5 * sigma_t * sigma_t * dt,
        }
    }

    pub fn path(
        &self,
        n_steps: usize,
        s0: f64,
        mu: f64,
        sigma: f64,
        dt: f64,
        seed: u64,
    ) -> (Vec<f64>, PathTrace) {
        let PathSpec::Scenario { sigma_bar, vol, .. } = self.spec else {
            return (
                gbm_path(n_steps, s0, mu, sigma, dt, seed),
                PathTrace::default(),
            );
        };
        let mut rng = StdRng::seed_from_u64(seed);
        let z = Normal::new(0.0, 1.0).unwrap();
        let mut rng_xi = StdRng::seed_from_u64(stream_seed(seed, XI_STREAM));
        let mut rng_eta = StdRng::seed_from_u64(stream_seed(seed, ETA_STREAM));
        let mut tr = PathTrace::default();
        // per-episode level (Group 2) or initial state (Group 3)
        let mut x = 0.0;
        let mut sig_const = sigma_bar;
        let (mut phi, mut innov_sd) = (0.0, 0.0);
        if let Some(v) = vol {
            match v.tau {
                None => {
                    // truncated normal by inverse cdf: one uniform draw, no rejection
                    let (lo, hi) = (self.norm.cdf(-v.k), self.norm.cdf(v.k));
                    let u: f64 = rand::Rng::gen_range(&mut rng_xi, 0.0..1.0);
                    let xi = self.norm.inverse_cdf(lo + u * (hi - lo));
                    sig_const = sigma_bar * (v.s * xi - self.episode_correction(v.s, v.k)).exp();
                    tr.xi = Some(xi);
                }
                Some(tau) => {
                    phi = (-1.0 / tau).exp();
                    innov_sd = v.s * (1.0 - phi * phi).sqrt();
                    x = v.s * z.sample(&mut rng_eta); // stationary start x_0 ~ N(0, s^2)
                }
            }
        }
        let mut prices = Vec::with_capacity(n_steps + 1);
        prices.push(s0);
        for _ in 0..n_steps {
            let zt: f64 = z.sample(&mut rng);
            let sig_t = match vol {
                Some(v) if v.tau.is_some() => sigma_bar * (x - v.s * v.s).exp(),
                _ => sig_const,
            };
            let eps = self.trunc_t.as_ref().map_or(zt, |tt| tt.eps(zt));
            let m = self.drift_constant(sig_t, dt);
            let next = prices.last().unwrap() * (mu * dt - m + sig_t * dt.sqrt() * eps).exp();
            prices.push(next);
            tr.sigma_t.push(sig_t);
            tr.x_t.push(x);
            tr.z_t.push(zt);
            tr.eps_t.push(eps);
            if let Some(v) = vol
                && v.tau.is_some()
            {
                let eta: f64 = z.sample(&mut rng_eta);
                tr.eta_t.push(eta);
                x = phi * x + innov_sd * eta;
            }
        }
        (prices, tr)
    }
}

#[cfg(test)]
mod vtr1_tests {
    use super::*;

    #[test]
    fn baseline_spec_is_the_unchanged_gbm_path() {
        let g = ScenarioGen::new(PathSpec::Baseline);
        let (p, tr) = g.path(51, 1000.0, 0.0, 0.5, 1.0 / 525_600.0, 30_001);
        assert_eq!(p, gbm_path(51, 1000.0, 0.0, 0.5, 1.0 / 525_600.0, 30_001));
        assert!(tr.sigma_t.is_empty());
    }

    #[test]
    fn student_t_quantiles_match_tables() {
        let d5 = StudentsT::new(0.0, 1.0, 5.0).unwrap();
        let d3 = StudentsT::new(0.0, 1.0, 3.0).unwrap();
        assert!((d5.inverse_cdf(0.975) - 2.570_581_836).abs() < 1e-6);
        assert!((d3.inverse_cdf(0.975) - 3.182_446_305).abs() < 1e-6);
        for p in [1e-9, 1e-4, 0.3, 0.5, 0.9, 1.0 - 1e-9] {
            assert!((d5.cdf(d5.inverse_cdf(p)) - p).abs() < 1e-9 * p.max(1e-3));
        }
    }

    #[test]
    fn truncated_t_constants_match_spec_section_5() {
        for (nu, sd, mass, kurt) in [(5.0, 1.2878, 2.0e-5, 3.907), (3.0, 1.6382, 2.4e-4, 12.611)] {
            let t = TruncT::new(TInnov { nu, cap: 12.0 });
            assert!((t.sd - sd).abs() < 5e-4, "nu={nu} sd={}", t.sd);
            assert!(
                (t.truncation_mass() - mass).abs() < 0.06 * mass,
                "mass {}",
                t.truncation_mass()
            );
            assert!((t.moment(2) - 1.0).abs() < 1e-9 && t.moment(1).abs() < 1e-12);
            assert!(
                (t.moment(4) - 3.0 - kurt).abs() < 0.02 * kurt,
                "kurt {}",
                t.moment(4) - 3.0
            );
        }
    }

    #[test]
    fn t_innovation_map_is_strictly_increasing_and_bounded() {
        let t = TruncT::new(TInnov { nu: 5.0, cap: 12.0 });
        let mut prev = f64::NEG_INFINITY;
        for i in -6000..=6000 {
            let e = t.eps(i as f64 * 1e-3);
            assert!(e > prev);
            assert!(e.abs() <= 12.0301);
            prev = e;
        }
    }

    #[test]
    fn drift_constant_matches_independent_quadrature() {
        let dt = 1.0 / 525_600.0;
        for nu in [5.0, 3.0] {
            let t = TruncT::new(TInnov { nu, cap: 12.0 });
            for sigma in [0.1, 0.5, 0.65, 1.0, 2.0] {
                let a = sigma * f64::sqrt(dt);
                let (m, q) = (t.log_mgf(a), t.log_mgf_quadrature(a, 1_000_000));
                assert!(
                    (m - q).abs() <= 1e-9 * 0.5 * a * a,
                    "nu={nu} sigma={sigma} {m} {q}"
                );
            }
        }
    }

    #[test]
    fn episode_correction_matches_spec_section_3() {
        let g = ScenarioGen::new(PathSpec::Baseline);
        assert!((g.episode_correction(0.4, 2.5) - 0.14320).abs() < 1e-4);
    }

    #[test]
    fn scenario_generator_builds_the_truncated_t_once() {
        let g = ScenarioGen::new(PathSpec::Scenario {
            sigma_bar: 0.5,
            vol: None,
            t: Some(TInnov { nu: 5.0, cap: 12.0 }),
        });
        let built = TRUNC_T_BUILDS.with(|c| c.get());
        for seed in 0..50 {
            g.path(51, 1000.0, 0.0, 0.5, DT, 200_000 + seed);
        }
        assert_eq!(TRUNC_T_BUILDS.with(|c| c.get()), built);
    }

    #[test]
    fn stream_constants_are_the_frozen_ones() {
        assert_eq!(XI_STREAM, (1_597_334_677, 5));
        assert_eq!(ETA_STREAM, (374_761_393, 6));
    }

    fn rand_z(seed: u64, n: usize) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let z = Normal::new(0.0, 1.0).unwrap();
        (0..n).map(|_| z.sample(&mut rng)).collect()
    }

    const DT: f64 = 1.0 / 525_600.0;

    #[test]
    fn scenarios_reuse_the_baseline_z_stream_and_are_deterministic() {
        let g2 = ScenarioGen::new(PathSpec::Scenario {
            sigma_bar: 0.5,
            vol: Some(VolState {
                s: 0.4,
                tau: None,
                k: 2.5,
            }),
            t: None,
        });
        let g3 = ScenarioGen::new(PathSpec::Scenario {
            sigma_bar: 0.5,
            vol: Some(VolState {
                s: 0.4,
                tau: Some(30.0),
                k: 2.5,
            }),
            t: Some(TInnov { nu: 5.0, cap: 12.0 }),
        });
        for g in [&g2, &g3] {
            let (p, tr) = g.path(51, 1000.0, 0.0, 0.5, DT, 200_007);
            assert_eq!(tr.z_t, rand_z(200_007, 51));
            assert_eq!(p, g.path(51, 1000.0, 0.0, 0.5, DT, 200_007).0);
            assert!(p.iter().all(|x| x.is_finite() && *x > 0.0));
        }
    }

    #[test]
    fn group2_sigma_is_constant_within_episode_and_inside_bounds() {
        let g = ScenarioGen::new(PathSpec::Scenario {
            sigma_bar: 0.5,
            vol: Some(VolState {
                s: 0.4,
                tau: None,
                k: 2.5,
            }),
            t: None,
        });
        for seed in 200_000..200_300 {
            let (_, tr) = g.path(51, 1000.0, 0.0, 0.5, DT, seed);
            assert!(
                tr.sigma_t
                    .iter()
                    .all(|s| s.to_bits() == tr.sigma_t[0].to_bits())
            );
            let r = tr.sigma_t[0] / 0.5;
            assert!((0.31880 - 1e-4..=2.35561 + 1e-4).contains(&r), "ratio {r}");
        }
    }
}
