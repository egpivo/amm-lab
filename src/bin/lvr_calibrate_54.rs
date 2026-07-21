//! Round-26 formal 54-cell two-moment hazard calibration.
//!
//! 2 strata x 3 sigma x 3 z x 3 arb-speed = 54 cells. Each solves for
//! the LATENT hazards (lambda_arb*, lambda_fund*) on the FORMAL
//! CALIBRATION seed block so the static baseline's realized fills match
//! BOTH the tier activity target and the proxy-calibrated arb-intensity
//! anchor (round 26). Primary clock = 1 second (frozen round-25
//! certification correction; the unchanged 2% criterion rejects 5 s).
//!
//! Manifest labels lambda_arb*/lambda_fund* as SOLVED HAZARDS and the
//! six rates as OBSERVED TARGETS — never conflated. Calibrated hazards
//! are frozen here and MUST NOT be re-tuned in training/validation/final.
//!
//! Usage: lvr_calibrate_54 [--cells N]  (N limits cells for sanity).

use amm_lab::campbell::calibrate::calibrate_two_moment;
use amm_lab::campbell::simulation::{ArrivalModel, FlowRegime, SimConfig};
use std::fs::File;
use std::io::{self, Write};

const S0: f64 = 2000.0;
// FORMAL calibration seed block (>= 10,000; disjoint from pilot < 10,000
// and from the yet-to-be-frozen training/validation/final blocks).
const CAL_SEEDS: std::ops::Range<u64> = 10_000..10_008;
const DT_HOURS: f64 = 1.0 / 3600.0; // 1-second clock
const N_STEPS: usize = 604_800; // one week
const LAG: usize = 300; // 5-minute physical staleness

struct Stratum {
    name: &'static str,
    fee: f64,
    total_target: f64,
    arb_anchors: [f64; 3], // slow, medium, fast (per hour)
}

fn strata() -> [Stratum; 2] {
    [
        Stratum {
            name: "5bp",
            fee: 0.0005,
            total_target: 41_500.0,
            arb_anchors: [5.3, 9.6, 27.8],
        },
        Stratum {
            name: "30bp",
            fee: 0.0030,
            total_target: 3_000.0,
            arb_anchors: [0.35, 0.70, 2.37],
        },
    ]
}

const SIGMAS: [f64; 3] = [0.48, 0.64, 0.92];
const ZS: [f64; 3] = [0.00087, 0.0055, 0.030];
const SPEED_NAMES: [&str; 3] = ["slow", "medium", "fast"];

fn config_for(fee: f64, sigma: f64, z: f64) -> SimConfig {
    let y0 = 1.0e4;
    let d_ref = y0 * (1.0 - (1.0f64 + 0.01).powf(-0.5)); // 1% directional depth
    SimConfig {
        name: "cal54".into(),
        description: "round-26 formal 54-cell calibration".into(),
        amm_fee: fee,
        cex_fee: 0.0010,
        buy_demand: z * d_ref,
        sell_demand: z * d_ref,
        reserve_x: 2.0e7,
        reserve_y: y0,
        sigma,
        mu: 0.0,
        n_steps: N_STEPS,
        seed: 0,
        flow_regime: FlowRegime::Normal,
        toxic_burst_prob: 0.0,
        toxic_burst_arb_scale: 1.0,
        toxic_burst_fund_scale: 1.0,
        regime_switch_period: 0,
        e1_lambda: 0.0,
        e1_fee_ref: 0.0006,
        e5_arb_prob: 1.0,
        policy_lag: LAG,
        dt_hours: DT_HOURS,
        pooled_fund_arrival_rate_per_hour: Some(1.0),
        buy_arrival_share: 0.5,
        arb_arrival_rate_per_hour: Some(1.0),
        lookback_hours: 20.0,
        arrival_model: ArrivalModel::Poisson,
        log_inactive_arb: false,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let limit: usize = args
        .iter()
        .position(|a| a == "--cells")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(54);

    let mut cells = Vec::new();
    let start = std::time::Instant::now();
    let mut done = 0usize;
    'outer: for st in strata() {
        for &sigma in &SIGMAS {
            for &z in &ZS {
                for (si, sp_name) in SPEED_NAMES.iter().enumerate() {
                    if done >= limit {
                        break 'outer;
                    }
                    let cfg = config_for(st.fee, sigma, z);
                    let arb_hr = st.arb_anchors[si];
                    let r =
                        calibrate_two_moment(&cfg, S0, CAL_SEEDS, st.total_target, arb_hr, 0.05);
                    done += 1;
                    let elapsed = start.elapsed().as_secs_f64();
                    println!(
                        "[{done:2}/{limit}] {} sigma={sigma} z={z} {sp_name}: \
lambda_arb*={:.2} lambda_fund*={:.2} arb_ach={:.0}/wk (tgt {:.0}) tot_ach={:.0}/wk (tgt {:.0}) reachable={} [{elapsed:.0}s]",
                        st.name,
                        r.lambda_arb,
                        r.lambda_fund,
                        r.arb_achieved,
                        arb_hr * 168.0,
                        r.total_achieved,
                        st.total_target,
                        r.arb_reachable
                    );
                    io::stdout().flush().expect("flush calibration progress");
                    cells.push(serde_json::json!({
                        "stratum": st.name,
                        "fee": st.fee,
                        "sigma": sigma,
                        "z": z,
                        "arb_speed": sp_name,
                        "arb_target_per_hour_OBSERVED": arb_hr,
                        "total_target_per_week": st.total_target,
                        "lambda_arb_star_SOLVED": r.arb_reachable.then_some(r.lambda_arb),
                        "lambda_fund_star_SOLVED": r.arb_reachable.then_some(r.lambda_fund),
                        "lambda_arb_ceiling_DIAGNOSTIC": (!r.arb_reachable).then_some(r.lambda_arb),
                        "lambda_fund_at_arb_ceiling_DIAGNOSTIC": (!r.arb_reachable).then_some(r.lambda_fund),
                        "arb_fills_achieved_per_week": r.arb_achieved,
                        "total_fills_achieved_per_week": r.total_achieved,
                        "arb_reachable": r.arb_reachable,
                    }));
                }
            }
        }
    }

    let manifest = serde_json::json!({
        "round": 26,
        "clock": "1-second (frozen round-25 corrected certification)",
        "dt_hours": DT_HOURS,
        "n_steps_per_week": N_STEPS,
        "policy_lag_steps": LAG,
        "calibration_seed_block": [CAL_SEEDS.start, CAL_SEEDS.end],
        "note": "For reachable cells, lambda_*_SOLVED are back-solved LATENT hazards. For unreachable cells they are null and lambda_*_DIAGNOSTIC reports the joint-calibration ceiling evaluation only. arb_target_per_hour_OBSERVED is a proxy-calibrated compatible-fill rate, not a latent hazard. Only reachable solved hazards may be frozen for training/validation/final.",
        "cells": cells,
    });
    if limit >= 54 {
        let path = "/Users/joseph/amm-lab/.local/lvr/calibration_54_manifest.json";
        serde_json::to_writer_pretty(File::create(path).unwrap(), &manifest).unwrap();
        eprintln!("wrote {path}");
    }
}
