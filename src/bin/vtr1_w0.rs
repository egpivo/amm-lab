//! VTR-1 W0 scenario diagnostics harness (spec §7). Price generator only: no
//! policy, no pools, no training. **Not authorized to run**: it refuses to start
//! unless the authorization file exists, and it reads only the W0 seed blocks.
//! Tolerances that the spec leaves OPEN are CLI options; without a value the
//! affected check reports its numbers with status TOLERANCE_OPEN.

use amm_lab::sim::oracle::{PathSpec, ScenarioGen, TInnov, TruncT, VolState};
use clap::Parser;
use serde_json::{Value, json};
use statrs::distribution::{Continuous, ContinuousCDF, Normal as StatNormal, StudentsT};

const H: usize = 50;
const DT: f64 = 1.0 / 525_600.0;
const DB1: (u64, u64) = (200_000, 300_000);
const DB2: (u64, u64) = (300_000, 358_000);

#[derive(Parser)]
struct Args {
    /// 1 magnitude, 2 episode heterogeneity, 3 stochastic volatility, 4 t innovations, 5 combined
    #[arg(long)]
    group: u8,
    #[arg(long)]
    sigma_bar: f64,
    #[arg(long, default_value_t = 0.4)]
    s: f64,
    #[arg(long, default_value_t = 2.5)]
    k: f64,
    /// Group 3 / Group 5 C-1, C-3 e-folding time in minutes
    #[arg(long)]
    tau: Option<f64>,
    /// Group 4 / Group 5 C-2, C-3 degrees of freedom
    #[arg(long)]
    nu: Option<f64>,
    #[arg(long, default_value_t = 12.0)]
    cap: f64,
    #[arg(long, default_value_t = 100_000)]
    n_ep: u64,
    #[arg(long, default_value_t = 2_000)]
    n_days: u64,
    #[arg(long, default_value = "out/VTR1_W0_AUTHORIZED.txt")]
    authorization: String,
    #[arg(long, default_value = "out/vtr1_w0_diagnostics.json")]
    out: String,
    /// OPEN tolerances (owner): without a value the check is reported TOLERANCE_OPEN.
    #[arg(long)]
    d2_drift_tol: Option<f64>,
    #[arg(long)]
    d2x_rel_tol: Option<f64>,
    #[arg(long)]
    d5b_rel_tol: Option<f64>,
    /// OPEN: gate for the nu = 3 kurtosis (none unless given).
    #[arg(long)]
    kurt3_gate: Option<f64>,
}

#[derive(Default, Clone)]
struct Acc {
    n: f64,
    mean: f64,
    m2: f64,
}
impl Acc {
    fn add(&mut self, x: f64) {
        self.n += 1.0;
        let d = x - self.mean;
        self.mean += d / self.n;
        self.m2 += d * (x - self.mean);
    }
    fn sd(&self) -> f64 {
        (self.m2 / (self.n - 1.0)).sqrt()
    }
    fn se(&self) -> f64 {
        self.sd() / self.n.sqrt()
    }
}

fn verdict(a: bool, r: Option<bool>) -> &'static str {
    match (r, a) {
        (Some(false), _) => "INCONCLUSIVE",
        (_, false) => "FAIL",
        _ => "PASS",
    }
}

/// Accuracy (A) check at z = 4, with optional resolution (R) tolerance.
fn diag(id: &str, target: f64, est: f64, se: f64, tol: Option<f64>) -> Value {
    let a = (est - target).abs() <= 4.0 * se;
    let r = tol.map(|t| 4.0 * se <= t);
    json!({"id": id, "target": target, "estimate": est, "se": se, "resolution_4se": 4.0 * se,
           "status": verdict(a, r)})
}

fn report(id: &str, fields: Value) -> Value {
    json!({"id": id, "status": "REPORTED", "values": fields})
}

fn simpson_sd(nu: f64, k: f64, n: usize) -> f64 {
    let d = StudentsT::new(0.0, 1.0, nu).unwrap();
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
    let mass: f64 = (0..=n).map(|i| w(i) * d.pdf(-k + i as f64 * h)).sum();
    let m2: f64 = (0..=n)
        .map(|i| w(i) * (-k + i as f64 * h).powi(2) * d.pdf(-k + i as f64 * h))
        .sum();
    (m2 / mass).sqrt()
}

fn returns(p: &[f64]) -> Vec<f64> {
    (0..H).map(|t| (p[t + 1] / p[t]).ln()).collect()
}

fn main() {
    let a = Args::parse();
    if !std::path::Path::new(&a.authorization).exists() {
        eprintln!(
            "W0 diagnostics are not authorized: {} does not exist",
            a.authorization
        );
        std::process::exit(2);
    }
    assert!(DB1.0 + a.n_ep <= DB1.1, "n_ep exceeds the DB1 block");
    assert!(
        DB2.0 + 29 * a.n_days <= DB2.1,
        "n_days exceeds the DB2 block"
    );
    let vol = match (a.group, a.tau) {
        (2, _) => Some(VolState {
            s: a.s,
            tau: None,
            k: a.k,
        }),
        (3 | 5, Some(tau)) => Some(VolState {
            s: a.s,
            tau: Some(tau),
            k: a.k,
        }),
        _ => None,
    };
    let t = a.nu.map(|nu| TInnov { nu, cap: a.cap });
    let spec = if a.group == 1 {
        PathSpec::Baseline
    } else {
        PathSpec::Scenario {
            sigma_bar: a.sigma_bar,
            vol,
            t,
        }
    };
    let gen_ = ScenarioGen::new(spec);
    let tt = t.map(TruncT::new);
    let norm = StatNormal::new(0.0, 1.0).unwrap();
    let phi = a.tau.map(|tau| (-1.0 / tau).exp());
    let target_var = a.sigma_bar * a.sigma_bar;

    let (mut q, mut drift) = (Acc::default(), Acc::default());
    let (mut lsig, mut s2r, mut x_all) = (Acc::default(), Acc::default(), Acc::default());
    let (mut x0sq, mut xlsq, mut sig2) = (Acc::default(), Acc::default(), Acc::default());
    let lags = [1usize, 5, 10, 25, 49];
    let mut cov = vec![Acc::default(); lags.len()];
    let (mut czeta, mut czx) = (Acc::default(), Acc::default());
    let (mut e2, mut t3, mut t5, mut e4) = (0f64, 0f64, 0f64, 0f64);
    let (mut e_max, mut xi_lo, mut xi_hi, mut const_ok) =
        (0f64, f64::INFINITY, f64::NEG_INFINITY, true);
    let mut n_eps = 0f64;
    for j in 0..a.n_ep {
        let (p, tr) = gen_.path(H + 1, 1_000.0, 0.0, a.sigma_bar, DT, DB1.0 + j);
        let r = returns(&p);
        q.add(r.iter().map(|x| x * x / DT).sum::<f64>() / H as f64);
        drift.add(p.windows(2).take(H).map(|w| w[1] / w[0] - 1.0).sum::<f64>() / H as f64);
        if vol.is_some_and(|v| v.tau.is_none()) {
            let sig = tr.sigma_t[0];
            const_ok &= tr.sigma_t[..H]
                .iter()
                .all(|&x| x.to_bits() == sig.to_bits());
            let ratio = sig / a.sigma_bar;
            lsig.add(ratio.ln());
            s2r.add(ratio * ratio);
            xi_lo = xi_lo.min(ratio);
            xi_hi = xi_hi.max(ratio);
        }
        if let (Some(v), Some(ph)) = (vol, phi)
            && v.tau.is_some()
        {
            let x = &tr.x_t[..H];
            x_all.add(x.iter().sum::<f64>() / H as f64);
            x0sq.add(x[0] * x[0]);
            xlsq.add(x[H - 1] * x[H - 1]);
            sig2.add(tr.sigma_t[..H].iter().map(|s| s * s).sum::<f64>() / H as f64 / target_var);
            for (i, &k) in lags.iter().enumerate() {
                cov[i].add((0..H - k).map(|t| x[t] * x[t + k]).sum::<f64>() / (H - k) as f64);
            }
            czeta.add((0..H).map(|t| tr.z_t[t] * tr.eta_t[t]).sum::<f64>() / H as f64);
            czx.add((0..H).map(|t| tr.z_t[t] * x[t]).sum::<f64>() / H as f64);
            let _ = ph;
        }
        if let Some(tt) = &tt {
            let cap_std = tt.cap_raw / tt.sd;
            for &e in &tr.eps_t[..H] {
                e2 += e * e;
                e4 += e.powi(4);
                t3 += (e.abs() > 3.0) as u64 as f64;
                t5 += (e.abs() > 5.0) as u64 as f64;
                e_max = e_max.max(e.abs() / cap_std);
            }
            n_eps += H as f64;
        }
    }

    let mut d: Vec<Value> = vec![];
    let n = a.n_ep as f64;
    // D1: R tolerance 2% (relative), A on the relative deviation
    let (v_hat, se_rel) = (q.mean, q.sd() / n.sqrt() / target_var);
    d.push(diag("D1", 1.0, v_hat / target_var, se_rel, Some(0.02)));
    // D2: accuracy only; practical tolerance is OPEN
    let mut d2 = diag("D2", 0.0, drift.mean, drift.se(), a.d2_drift_tol);
    d2["resolution_bps_per_episode"] = json!(4.0 * drift.se() * H as f64 * 1e4);
    d2["practical_tolerance"] = json!(a.d2_drift_tol.map_or("OPEN".into(), |t| json!(t)));
    d.push(d2);
    // D2x: independent quadrature for the t-law; Gaussian eps uses the closed form 0.5 a^2 (identity)
    if let Some(tt) = &tt {
        let rows: Vec<Value> = [0.1, 0.5, a.sigma_bar, 1.0, 2.0]
            .iter()
            .map(|&s| {
                let am = s * DT.sqrt();
                let (m, mq) = (tt.log_mgf(am), tt.log_mgf_quadrature(am, 1_000_000));
                json!({"sigma": s, "m_impl": m, "m_indep": mq, "rel_dev": (m - mq).abs() / (0.5 * am * am)})
            })
            .collect();
        let status = match a.d2x_rel_tol {
            None => "TOLERANCE_OPEN",
            Some(tol) => {
                if rows.iter().all(|r| r["rel_dev"].as_f64().unwrap() <= tol) {
                    "PASS"
                } else {
                    "FAIL"
                }
            }
        };
        d.push(json!({"id": "D2x", "status": status, "rows": rows}));
    }
    // D3 / D3x (Group 2)
    if let Some(v) = vol.filter(|v| v.tau.is_none()) {
        let kk = v.k;
        let cs = gen_.episode_correction(v.s, kk);
        let sd_target =
            v.s * (1.0 - 2.0 * kk * norm.pdf(kk) / (norm.cdf(kk) - norm.cdf(-kk))).sqrt();
        d.push(diag("D3a", -cs, lsig.mean, lsig.se(), None));
        let z = norm.cdf(kk) - norm.cdf(-kk);
        let mu2 = 1.0 - 2.0 * kk * norm.pdf(kk) / z;
        let mu4 = 3.0 * mu2 - 2.0 * kk.powi(3) * norm.pdf(kk) / z;
        let kappa_xi = mu4 / (mu2 * mu2);
        d.push(diag(
            "D3b",
            sd_target,
            lsig.sd(),
            sd_target * ((kappa_xi - 1.0) / (4.0 * n)).sqrt(),
            None,
        ));
        d.push(diag("D3c", 1.0, s2r.mean, s2r.se(), None));
        let (lo, hi) = ((-kk * v.s - cs).exp(), (kk * v.s - cs).exp());
        let ok = xi_lo >= lo - 1e-12 && xi_hi <= hi + 1e-12 && const_ok;
        d.push(
            json!({"id": "D3x", "status": if ok {"PASS"} else {"FAIL"}, "min": xi_lo, "max": xi_hi,
            "expected_range": [lo, hi], "sigma_bitwise_constant_within_episode": const_ok}),
        );
    }
    // D4 (Group 3 state)
    if let (Some(v), Some(ph)) = (vol.filter(|v| v.tau.is_some()), phi) {
        let s2 = v.s * v.s;
        d.push(diag("D4a", 0.0, x_all.mean, x_all.se(), None));
        d.push(diag(
            "D4b0",
            s2,
            x0sq.mean,
            s2 * 2f64.sqrt() / n.sqrt(),
            None,
        ));
        d.push(diag(
            "D4b1",
            s2,
            xlsq.mean,
            s2 * 2f64.sqrt() / n.sqrt(),
            None,
        ));
        for (i, &k) in lags.iter().enumerate() {
            d.push(diag(
                &format!("D4c_lag{k}"),
                s2 * ph.powi(k as i32),
                cov[i].mean,
                cov[i].se(),
                None,
            ));
        }
        d.push(diag("D4d", 1.0, sig2.mean, sig2.se(), None));
        d.push(diag("D4e_z_eta", 0.0, czeta.mean, czeta.se(), None));
        d.push(diag("D4e_z_x", 0.0, czx.mean, czx.se(), None));
    }
    // D5 (t innovations)
    if let (Some(tt), Some(ti)) = (&tt, t) {
        let nn = n_eps;
        let grid_ok = |lo: f64, hi: f64, m: usize, strict: bool| {
            let mut prev = tt.eps(lo);
            (1..=m).all(|i| {
                let e = tt.eps(lo + (hi - lo) * i as f64 / m as f64);
                let ok = if strict { e > prev } else { e >= prev };
                prev = e;
                ok
            })
        };
        let mono = grid_ok(-8.5, 8.5, 1_000_000, false) && grid_ok(-6.0, 6.0, 100_000, true);
        let sd_q = simpson_sd(ti.nu, tt.cap_raw, 1_000_000);
        let sd_rel = (tt.sd - sd_q).abs() / sd_q;
        let b_status = match a.d5b_rel_tol {
            None => "TOLERANCE_OPEN",
            Some(t) => {
                if sd_rel <= t && e_max <= 1.0 + 1e-12 {
                    "PASS"
                } else {
                    "FAIL"
                }
            }
        };
        d.push(json!({"id": "D5a", "status": if mono {"PASS"} else {"FAIL"}}));
        d.push(json!({"id": "D5b", "status": b_status, "max_abs_over_cap": e_max, "sd_rel_dev_vs_quadrature": sd_rel}));
        let (m4, m6, m8) = (tt.moment(4), tt.moment(6), tt.moment(8));
        d.push(diag("D5c", 1.0, e2 / nn, ((m4 - 1.0) / nn).sqrt(), None));
        let dist = StudentsT::new(0.0, 1.0, ti.nu).unwrap();
        let mass = dist.cdf(tt.cap_raw) - dist.cdf(-tt.cap_raw);
        for (c, cnt, id) in [(3.0, t3, "D5d_3"), (5.0, t5, "D5d_5")] {
            let p = 2.0 * (dist.cdf(tt.cap_raw) - dist.cdf(c * tt.sd)) / mass;
            d.push(diag(id, p, cnt / nn, (p * (1.0 - p) / nn).sqrt(), None));
        }
        let kappa = m4 - 3.0;
        let se_th =
            ((m8 - m4 * m4 - 4.0 * m4 * (m6 - m4) + 4.0 * m4 * m4 * (m4 - 1.0)) / nn).sqrt();
        let k_hat = e4 / nn / (e2 / nn).powi(2) - 3.0;
        let mut k = json!({"id": "D5e", "target": kappa, "estimate": k_hat, "se_theory": se_th});
        k["status"] = if ti.nu >= 5.0 {
            json!(verdict((k_hat - kappa).abs() <= 4.0 * se_th, None))
        } else {
            match a.kurt3_gate {
                None => json!("REPORTED_GATE_OPEN"),
                Some(g) => json!(verdict((k_hat - kappa).abs() <= g, None)),
            }
        };
        d.push(k);
    }
    // D7: daily exceedance rate (E1 rule), episodes assembled into 1,439-return days
    let (mut ex_all, mut ex_sec) = (Acc::default(), Acc::default());
    for day in 0..a.n_days {
        let mut r: Vec<f64> = Vec::with_capacity(1_439);
        for j in 0..29u64 {
            let (p, _) = gen_.path(H + 1, 1_000.0, 0.0, a.sigma_bar, DT, DB2.0 + 29 * day + j);
            let rr = returns(&p);
            r.extend_from_slice(if j == 28 { &rr[..39] } else { &rr[..] });
        }
        let count = |pairs: &dyn Fn(usize) -> bool| {
            let (mut s, mut np) = (0.0, 0.0);
            for i in 1..r.len() {
                if pairs(i) {
                    s += r[i].abs() * r[i - 1].abs();
                    np += 1.0;
                }
            }
            let v = std::f64::consts::FRAC_PI_2 * s / np;
            r.iter().filter(|x| x.abs() > 4.0 * v.sqrt()).count() as f64
        };
        ex_all.add(count(&|_| true));
        ex_sec.add(count(&|i| i % H != 0));
    }
    if a.group == 1 {
        let se_cal = (0.088f64 / 2000.0).sqrt();
        let se = (ex_all.se().powi(2) + se_cal.powi(2)).sqrt();
        d.push(diag("D7", 0.088, ex_all.mean, se, None));
    } else {
        d.push(report(
            "D7",
            json!({"exceedances_per_day": ex_all.mean, "se": ex_all.se(),
            "secondary_excl_straddling_pairs": ex_sec.mean, "gated": false}),
        ));
    }
    let doc = json!({"scenario": {"group": a.group, "sigma_bar": a.sigma_bar, "tau": a.tau, "nu": a.nu,
        "s": a.s, "k": a.k, "cap": a.cap}, "n_ep": a.n_ep, "n_days": a.n_days, "diagnostics": d});
    std::fs::write(&a.out, serde_json::to_string_pretty(&doc).unwrap()).expect("write");
}
