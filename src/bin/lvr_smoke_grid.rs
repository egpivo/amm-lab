//! M3 smoke grid for the LVR paper (.local/lvr/plan.md, round 10).
//!
//! Diagnostic run on TRAINING-DOMAIN seeds only (1000..1199); the
//! untouched final block is never read here. Order per round-10 review:
//!   1. canonical sigma^2/8 convergence anchor (dt grid, arb-only, f=c=0)
//!   2. C2 same-curve/different-rule constructive validation
//!   3. floor_hat_grid(f_max) with service / quote accuracy / allocation
//!   4. six-margin decomposition (fixed vs defensive)
//!   5. zero-lag vs one-step-lag for the core cells
//!   6. matched-service exploratory scan (validation candidates only)

use amm_lab::campbell::fee_policy::{FeePolicy, FixedFeePolicy, OracleGapFeePolicy};
use amm_lab::campbell::gbm::generate_gbm;
use amm_lab::campbell::simulation::{ArrivalModel, FlowRegime, SimConfig, run_simulation};
use amm_lab::campbell::summary::{LvrSummary, summarize};

const SEEDS: std::ops::Range<u64> = 1000..1200; // training block; final seeds untouched
const S0: f64 = 2000.0;
const T_YEARS: f64 = 30.0 / 365.0;

fn base_config(
    amm_fee: f64,
    cex_fee: f64,
    buy: f64,
    sell: f64,
    n_steps: usize,
    lag: usize,
) -> SimConfig {
    SimConfig {
        name: "smoke".into(),
        description: "m3 smoke grid".into(),
        amm_fee,
        cex_fee,
        buy_demand: buy,
        sell_demand: sell,
        reserve_x: 2.0e7,
        reserve_y: 1.0e4,
        sigma: 0.4,
        mu: 0.0,
        n_steps,
        seed: 0,
        flow_regime: FlowRegime::Normal,
        toxic_burst_prob: 0.0,
        toxic_burst_arb_scale: 1.0,
        toxic_burst_fund_scale: 1.0,
        regime_switch_period: 0,
        e1_lambda: 0.0,
        e1_fee_ref: 0.0006,
        e5_arb_prob: 1.0,
        policy_lag: lag,
        dt_hours: 1.0,
        pooled_fund_arrival_rate_per_hour: None,
        buy_arrival_share: 0.5,
        arb_arrival_rate_per_hour: None,
        lookback_hours: 20.0,
        arrival_model: ArrivalModel::Bernoulli,
        log_inactive_arb: false,
    }
}

#[derive(Clone)]
enum Policy {
    Fixed(f64),
    Gap { base: f64, mult: f64, max: f64 },
}

impl Policy {
    fn name(&self) -> String {
        match self {
            Policy::Fixed(f) => format!("fixed_{:.0}bps", f * 1e4),
            Policy::Gap { base, mult, max } => {
                format!("gap_b{:.0}_m{}_x{:.0}bps", base * 1e4, mult, max * 1e4)
            }
        }
    }
    fn make(&self) -> Box<dyn FeePolicy> {
        match self {
            Policy::Fixed(f) => Box::new(FixedFeePolicy::new(*f)),
            Policy::Gap { base, mult, max } => Box::new(OracleGapFeePolicy {
                base_fee: *base,
                gap_multiplier: *mult,
                min_fee: *base,
                max_fee: *max,
            }),
        }
    }
}

/// Mean summaries for one policy over the seed block (paired paths are
/// guaranteed by seeding the GBM with the seed itself).
fn run_policy(policy: &Policy, config: &SimConfig) -> MeanStats {
    let mut acc = MeanStats::default();
    let dt = T_YEARS / config.n_steps as f64;
    for seed in SEEDS {
        let prices = generate_gbm(config.n_steps, S0, 0.0, config.sigma, dt, seed);
        let mut cfg = config.clone();
        cfg.seed = seed;
        cfg.amm_fee = match policy {
            Policy::Fixed(f) => *f,
            Policy::Gap { base, .. } => *base,
        };
        let mut p = policy.make();
        let records = run_simulation(&cfg, &prices, p.as_mut());
        acc.add(&summarize(&records));
    }
    acc.finish((SEEDS.end - SEEDS.start) as f64);
    acc
}

#[derive(Default, Clone)]
struct MeanStats {
    n: f64,
    a_fill: f64,
    b_fill: f64,
    l_total: f64,
    a_arb: f64,
    a_fund: f64,
    fees: f64,
    u_lp: f64,
    served: f64,
    potential: f64,
    alloc_amm: f64,
    alloc_amm_n: f64,
    incidence_pooled: f64,
    incidence_n: f64,
    cond_size: f64,
    cond_size_n: f64,
    adverse_per_unit: f64,
    adverse_per_unit_n: f64,
    log_gap: f64,
}

impl MeanStats {
    fn add(&mut self, s: &LvrSummary) {
        self.n += 1.0;
        self.a_fill += s.a_fill;
        self.b_fill += s.b_fill;
        self.l_total += s.l_total;
        self.a_arb += s.a_arb;
        self.a_fund += s.a_fund;
        self.fees += s.fees_total;
        self.u_lp += s.u_lp_rel;
        self.served += s.served_fund_volume;
        self.potential += s.potential_volume;
        if let Some(v) = s.alloc_amm_share {
            self.alloc_amm += v;
            self.alloc_amm_n += 1.0;
        }
        if let Some(v) = s.incidence_pooled_event {
            self.incidence_pooled += v;
            self.incidence_n += 1.0;
        }
        if let Some(v) = s.cond_fill_size_pooled {
            self.cond_size += v;
            self.cond_size_n += 1.0;
        }
        if let Some(v) = s.a_fund_per_served_unit {
            self.adverse_per_unit += v;
            self.adverse_per_unit_n += 1.0;
        }
        self.log_gap += s.mean_abs_log_gap.unwrap_or(0.0);
    }
    fn finish(&mut self, n: f64) {
        for v in [
            &mut self.a_fill,
            &mut self.b_fill,
            &mut self.l_total,
            &mut self.a_arb,
            &mut self.a_fund,
            &mut self.fees,
            &mut self.u_lp,
            &mut self.served,
            &mut self.potential,
            &mut self.log_gap,
        ] {
            *v /= n;
        }
        if self.alloc_amm_n > 0.0 {
            self.alloc_amm /= self.alloc_amm_n;
        }
        if self.incidence_n > 0.0 {
            self.incidence_pooled /= self.incidence_n;
        }
        if self.cond_size_n > 0.0 {
            self.cond_size /= self.cond_size_n;
        }
        if self.adverse_per_unit_n > 0.0 {
            self.adverse_per_unit /= self.adverse_per_unit_n;
        }
    }
    fn opt(&self, v: f64, n: f64) -> String {
        if n > 0.0 {
            format!("{v:.6}")
        } else {
            "n/a".into()
        }
    }
}

fn main() {
    println!(
        "# LVR M3 smoke grid (training seeds {:?}, final block untouched)",
        SEEDS
    );

    // ── 1. canonical sigma^2/8 anchor ──────────────────────────────────
    println!("\n## 1. sigma^2/8 anchor (arb-only, f=c=0, T=30d, sigma=0.4)");
    let v0 = 2.0e7 + 1.0e4 * S0;
    let target = 0.4f64.powi(2) * T_YEARS / 8.0;
    println!("target A_T/V0 = sigma^2 T/8 = {target:.6}");
    for n_steps in [720usize, 2880, 11520] {
        let cfg = base_config(0.0, 0.0, 0.0, 0.0, n_steps, 0);
        let stats = run_policy(&Policy::Fixed(0.0), &cfg);
        println!(
            "n_steps={n_steps:6}  E[A_T]/V0 = {:.6}  (ratio to target {:.4})",
            stats.a_fill / v0,
            stats.a_fill / v0 / target
        );
    }

    // ── policies for parts 2-6 ─────────────────────────────────────────
    let fee_grid_bps = [1.0, 5.0, 10.0, 30.0, 50.0, 100.0, 300.0, 1000.0, 3000.0];
    let mut policies: Vec<Policy> = fee_grid_bps
        .iter()
        .map(|b| Policy::Fixed(b / 1e4))
        .collect();
    policies.push(Policy::Gap {
        base: 0.0005,
        mult: 0.5,
        max: 0.01,
    });
    policies.push(Policy::Gap {
        base: 0.0005,
        mult: 2.0,
        max: 0.05,
    });
    policies.push(Policy::Gap {
        base: 0.0030,
        mult: 5.0,
        max: 0.30,
    }); // defensive
    policies.push(Policy::Gap {
        base: 0.0005,
        mult: 50.0,
        max: 0.30,
    }); // defensive-hard

    let market = |lag: usize| base_config(0.0005, 0.0010, 5.0, 5.0, 720, lag);

    for lag in [0usize, 1] {
        println!("\n## 2-4. core table (policy_lag = {lag})");
        println!(
            "{:26} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8} {:>10} {:>10} {:>10}",
            "policy",
            "A_fill",
            "B_fill",
            "L",
            "A_arb",
            "A_fund",
            "U_LP",
            "served",
            "allocAMM",
            "incid",
            "condSize",
            "advPerUnit"
        );
        let mut rows: Vec<(String, MeanStats)> = Vec::new();
        for pol in &policies {
            let stats = run_policy(pol, &market(lag));
            println!(
                "{:26} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>8.1} {:>8.4} {:>10.4} {:>10.4} {:>10}",
                pol.name(),
                stats.a_fill,
                stats.b_fill,
                stats.l_total,
                stats.a_arb,
                stats.a_fund,
                stats.u_lp,
                stats.served,
                stats.alloc_amm,
                stats.incidence_pooled,
                stats.cond_size,
                stats.opt(stats.adverse_per_unit, stats.adverse_per_unit_n),
            );
            rows.push((pol.name(), stats));
        }

        // 3. floor curve over nested fixed-fee grid. NAMING (round 11):
        // this minimizes over Pi_static only, so it is floor_hat_STATIC;
        // the full floor_hat_grid must minimize over the cumulative nested
        // union of static/gap/defensive families with sup f_t <= f_max.
        println!("\n### floor_hat_static(f_max) over the nested FIXED-FEE grid only");
        println!(
            "{:>10} {:>12} {:>14} {:>10} {:>10} {:>12}",
            "f_max_bps", "floor A", "argmin", "served", "allocAMM", "logGap"
        );
        let mut best: Option<(String, MeanStats)> = None;
        for (i, b) in fee_grid_bps.iter().enumerate() {
            let (name, stats) = &rows[i];
            let better = match &best {
                None => true,
                Some((_, bs)) => stats.a_fill < bs.a_fill,
            };
            if better {
                best = Some((name.clone(), stats.clone()));
            }
            let (bn, bs) = best.as_ref().unwrap();
            println!(
                "{:>10.0} {:>12.2} {:>14} {:>10.1} {:>10.4} {:>12.6}",
                b, bs.a_fill, bn, bs.served, bs.alloc_amm, bs.log_gap
            );
        }

        // 6. matched-service exploratory scan (validation candidates only).
        // Round-11 tie handling: STRICT inequalities on both A and U, and
        // near-identical outcome pairs (e.g. two fully-shutdown policies)
        // are excluded — a reversal requires genuinely different policies.
        println!("\n### matched-service candidate scan (|served diff| <= 5% of level, strict)");
        for i in 0..rows.len() {
            for j in (i + 1)..rows.len() {
                let (n1, s1) = &rows[i];
                let (n2, s2) = &rows[j];
                let served_close =
                    (s1.served - s2.served).abs() <= 0.05 * s1.served.max(s2.served).max(1e-9);
                if !served_close || s1.served == 0.0 {
                    continue;
                }
                let a_gap = (s1.a_fill - s2.a_fill).abs();
                let u_gap = (s1.u_lp - s2.u_lp).abs();
                let distinct = a_gap > 1e-6 * s1.a_fill.abs().max(1.0)
                    && u_gap > 1e-6 * s1.u_lp.abs().max(1.0);
                if !distinct {
                    continue;
                }
                let reversal = (s1.a_fill < s2.a_fill) == (s1.u_lp < s2.u_lp);
                if reversal {
                    println!(
                        "  CANDIDATE: {n1} (A={:.1}, U={:.1}, S={:.0}) vs {n2} (A={:.1}, U={:.1}, S={:.0})",
                        s1.a_fill, s1.u_lp, s1.served, s2.a_fill, s2.u_lp, s2.served
                    );
                }
            }
        }
    }
    println!("\n(done; diagnostics only — no final seeds touched, no preregistered run)");
}
