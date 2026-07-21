//! validation-grid Step A: seed-size pilot (round-25 gate; runs BEFORE the formal
//! training/validation/final blocks are frozen).
//!
//! The defensive policy's arbitrage leg is heavy-tailed across seed
//! batches (80x A-difference between 20-seed pilots was observed), so
//! formal block sizes must come from a frozen ABSOLUTE precision
//! target, not a default. Preregistered spec:
//! - paired means stay the primary estimand (NO winsorization);
//! - paired-effect (DeltaU, DeltaA, DeltaS) between-batch variance from
//!   multiple independent pilot batches;
//! - target: final paired-U 95% CI half-width / V0 <= 2e-5;
//! - median-of-means reported as heavy-tail robustness;
//! - max-seed and top-1% seed contributions reported for every
//!   expectation, to disclose domination by rare episodes.
//!
//! Seeds: pilot domain (< 10,000), fresh range 4000..4150 (6 batches x
//! 25 seeds). Hazards are read from the frozen
//! calibration_54_manifest.json (single source of truth); this binary
//! never re-tunes them. Representative cells: both strata x {central
//! cell, worst-case cell}.

use amm_lab::campbell::fee_policy::{FeePolicy, FixedFeePolicy, OracleGapFeePolicy};
use amm_lab::campbell::gbm::generate_gbm;
use amm_lab::campbell::simulation::{
    ArrivalModel, FlowRegime, SimConfig, run_simulation_with_events,
};
use amm_lab::campbell::summary::summarize_events;
use std::fs::File;

const S0_PRICE: f64 = 2000.0;
const V0: f64 = 4.0e7;
const TARGET_HALFWIDTH_OVER_V0: f64 = 2.0e-5;
const N_BATCHES: usize = 6;
const BATCH_SIZE: usize = 25;
const SEED_START: u64 = 4000; // pilot domain, fresh

#[derive(Clone)]
struct Cell {
    stratum: String,
    fee: f64,
    sigma: f64,
    z: f64,
    speed: String,
    lam_arb: f64,
    lam_fund: f64,
}

fn load_cells() -> Vec<Cell> {
    let manifest: serde_json::Value = serde_json::from_reader(
        File::open("/Users/joseph/amm-lab/.local/lvr/calibration_54_manifest.json").unwrap(),
    )
    .unwrap();
    let want = [
        ("5bp", 0.64, 0.0055, "medium"),
        ("30bp", 0.64, 0.0055, "medium"),
        ("5bp", 0.92, 0.03, "fast"),
        ("30bp", 0.92, 0.03, "fast"),
    ];
    let mut out = Vec::new();
    for c in manifest["cells"].as_array().unwrap() {
        for (st, sg, zz, sp) in want {
            if c["stratum"] == st
                && (c["sigma"].as_f64().unwrap() - sg).abs() < 1e-9
                && (c["z"].as_f64().unwrap() - zz).abs() < 1e-9
                && c["arb_speed"] == sp
            {
                out.push(Cell {
                    stratum: st.into(),
                    fee: c["fee"].as_f64().unwrap(),
                    sigma: sg,
                    z: zz,
                    speed: sp.into(),
                    lam_arb: c["lambda_arb_star_SOLVED"].as_f64().unwrap(),
                    lam_fund: c["lambda_fund_star_SOLVED"].as_f64().unwrap(),
                });
            }
        }
    }
    assert_eq!(out.len(), 4, "expected the 4 representative cells");
    out
}

fn config_for(cell: &Cell) -> SimConfig {
    let y0 = 1.0e4;
    let d_ref = y0 * (1.0 - (1.0f64 + 0.01).powf(-0.5));
    SimConfig {
        name: "seedpilot".into(),
        description: "validation-grid step A seed-size pilot".into(),
        amm_fee: cell.fee,
        cex_fee: 0.0010,
        buy_demand: cell.z * d_ref,
        sell_demand: cell.z * d_ref,
        reserve_x: 2.0e7,
        reserve_y: y0,
        sigma: cell.sigma,
        mu: 0.0,
        n_steps: 604_800,
        seed: 0,
        flow_regime: FlowRegime::Normal,
        toxic_burst_prob: 0.0,
        toxic_burst_arb_scale: 1.0,
        toxic_burst_fund_scale: 1.0,
        regime_switch_period: 0,
        e1_lambda: 0.0,
        e1_fee_ref: 0.0006,
        e5_arb_prob: 1.0,
        policy_lag: 300, // 5-minute physical staleness at 1-second clock
        dt_hours: 1.0 / 3600.0,
        pooled_fund_arrival_rate_per_hour: Some(cell.lam_fund),
        buy_arrival_share: 0.5,
        arb_arrival_rate_per_hour: Some(cell.lam_arb),
        lookback_hours: 20.0,
        arrival_model: ArrivalModel::Poisson,
        log_inactive_arb: false,
    }
}

fn make_policy(which: usize, fee: f64) -> Box<dyn FeePolicy> {
    match which {
        0 => Box::new(FixedFeePolicy::new(fee)),
        1 => Box::new(OracleGapFeePolicy {
            base_fee: fee,
            gap_multiplier: 2.0,
            min_fee: fee,
            max_fee: 10.0 * fee,
        }),
        _ => Box::new(OracleGapFeePolicy {
            base_fee: fee,
            gap_multiplier: 50.0,
            min_fee: fee,
            max_fee: 0.30,
        }),
    }
}

fn quantile_desc(mut v: Vec<f64>) -> (f64, f64) {
    // (max share, top-1% share) of the total absolute sum
    v.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap());
    let total: f64 = v.iter().map(|x| x.abs()).sum();
    if total == 0.0 {
        return (0.0, 0.0);
    }
    let top1_n = (v.len() as f64 * 0.01).ceil() as usize;
    let max_share = v[0].abs() / total;
    let top1_share: f64 = v[..top1_n.max(1)].iter().map(|x| x.abs()).sum::<f64>() / total;
    (max_share, top1_share)
}

fn main() {
    let cells = load_cells();
    let names = ["static", "gap", "defensive"];
    let n_total = N_BATCHES * BATCH_SIZE;
    let mut report = Vec::new();

    for cell in &cells {
        let cfg0 = config_for(cell);
        // per-seed metrics [policy][seed]
        let mut u = vec![vec![0.0f64; n_total]; 3];
        let mut a = vec![vec![0.0f64; n_total]; 3];
        let mut s = vec![vec![0.0f64; n_total]; 3];
        for i in 0..n_total {
            let seed = SEED_START + i as u64;
            let prices = generate_gbm(
                cfg0.n_steps,
                S0_PRICE,
                0.0,
                cfg0.sigma,
                cfg0.dt_years(),
                seed,
            );
            for (pi, _) in names.iter().enumerate() {
                let mut cfg = cfg0.clone();
                cfg.seed = seed;
                let mut pol = make_policy(pi, cell.fee);
                let (records, events) = run_simulation_with_events(&cfg, &prices, pol.as_mut());
                let es = summarize_events(&events, &records);
                u[pi][i] = es.u_lp_rel;
                a[pi][i] = es.a_fill;
                s[pi][i] = es.served_fund_volume;
            }
        }

        let cell_name = format!(
            "{} s{} z{} {}",
            cell.stratum, cell.sigma, cell.z, cell.speed
        );
        println!(
            "\n== {cell_name} (lam_arb={}, lam_fund={}) ==",
            cell.lam_arb, cell.lam_fund
        );

        // heavy-tail disclosure on defensive A and per-policy U
        for (pi, nm) in names.iter().enumerate() {
            let (mx, t1) = quantile_desc(a[pi].clone());
            println!(
                "  {nm}: mean A={:.1} mean U/V0={:.3e} mean S={:.1}  A max-seed share={:.3} top-1%={:.3}",
                a[pi].iter().sum::<f64>() / n_total as f64,
                u[pi].iter().sum::<f64>() / n_total as f64 / V0,
                s[pi].iter().sum::<f64>() / n_total as f64,
                mx,
                t1
            );
        }

        // paired contrasts
        for (i, j, label) in [
            (1usize, 0usize, "gap-static"),
            (2, 0, "def-static"),
            (2, 1, "def-gap"),
        ] {
            let d: Vec<f64> = (0..n_total).map(|k| u[i][k] - u[j][k]).collect();
            let mean = d.iter().sum::<f64>() / n_total as f64;
            // between-batch variance of batch means
            let mut batch_means = Vec::new();
            for b in 0..N_BATCHES {
                let bm: f64 =
                    d[b * BATCH_SIZE..(b + 1) * BATCH_SIZE].iter().sum::<f64>() / BATCH_SIZE as f64;
                batch_means.push(bm);
            }
            let bb_mean = batch_means.iter().sum::<f64>() / N_BATCHES as f64;
            let bb_var = batch_means
                .iter()
                .map(|m| (m - bb_mean).powi(2))
                .sum::<f64>()
                / (N_BATCHES - 1) as f64;
            // per-seed variance implied by between-batch variance
            let per_seed_var_bb = bb_var * BATCH_SIZE as f64;
            // direct per-seed variance
            let per_seed_var_direct =
                d.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n_total - 1) as f64;
            let target_hw = TARGET_HALFWIDTH_OVER_V0 * V0;
            let n_req_bb = (1.96 * per_seed_var_bb.sqrt() / target_hw).powi(2).ceil();
            let n_req_direct = (1.96 * per_seed_var_direct.sqrt() / target_hw)
                .powi(2)
                .ceil();
            // median-of-means
            let mut bm_sorted = batch_means.clone();
            bm_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let mom = bm_sorted[N_BATCHES / 2];
            let (mx, t1) = quantile_desc(d.clone());
            println!(
                "  pair {label}: mean dU/V0={:.3e}  MoM/V0={:.3e}  sd_seed(direct)={:.0} (bb)={:.0}  N_req direct={} bb={}  max-seed share={:.3} top-1%={:.3}",
                mean / V0,
                mom / V0,
                per_seed_var_direct.sqrt(),
                per_seed_var_bb.sqrt(),
                n_req_direct as u64,
                n_req_bb as u64,
                mx,
                t1
            );
            report.push(serde_json::json!({
                "cell": cell_name, "pair": label,
                "mean_dU_over_V0": mean / V0,
                "median_of_means_over_V0": mom / V0,
                "sd_per_seed_direct": per_seed_var_direct.sqrt(),
                "sd_per_seed_between_batch": per_seed_var_bb.sqrt(),
                "n_required_direct": n_req_direct,
                "n_required_between_batch": n_req_bb,
                "max_seed_share": mx, "top1pct_share": t1,
            }));
        }
    }

    let manifest = serde_json::json!({
        "step": "seed-pilot seed-size pilot",
        "target_halfwidth_over_V0": TARGET_HALFWIDTH_OVER_V0,
        "batches": N_BATCHES, "batch_size": BATCH_SIZE,
        "seed_range": [SEED_START, SEED_START + n_total as u64],
        "estimand": "paired means, no winsorization; median-of-means reported as robustness",
        "pairs": report,
    });
    let path = "/Users/joseph/amm-lab/.local/lvr/seed_pilot_manifest.json";
    serde_json::to_writer_pretty(File::create(path).unwrap(), &manifest).unwrap();
    eprintln!("wrote {path}");
}
