//! M3 Step C: training-block dial-grid evaluation.
//!
//! For every (cell, family, dial) the binary evaluates one week at the
//! 1-second primary clock on the TRAINING seed block and writes
//! per-seed rows (U, A, B, S, alloc) to a gzipped CSV. Downstream
//! selection (service targets rho*S0, argmin-A member, frontier trace)
//! is post-processing and never re-runs the engine.
//!
//! DESIGN (round 31): each family's service dial is its BASE FEE with
//! the response shape frozen — static: f; gap: (f0, mult 2, max 10*f0);
//! defensive: (f0, mult 50, max 30%). Dial grid = stratum_fee x
//! {0.5, 0.75, 1, 1.5, 2, 3, 4.5, 7, 10, 15, 25, 40}. Hazards come from
//! the frozen calibration manifest and are never re-tuned.
//!
//! Sharding: --shard K --of M runs cells with index % M == K, so the
//! grid can be split across processes. Usage:
//!   lvr_m3_train --n-seeds 100 --seed-start 20000 [--shard 0 --of 4]

use amm_lab::campbell::fee_policy::{FeePolicy, FixedFeePolicy, OracleGapFeePolicy};
use amm_lab::campbell::gbm::generate_gbm;
use amm_lab::campbell::simulation::{
    ArrivalModel, FlowRegime, SimConfig, run_simulation_with_events,
};
use amm_lab::campbell::summary::summarize_events;
use flate2::Compression;
use flate2::write::GzEncoder;
use std::fs::File;
use std::io::Write;

const S0_PRICE: f64 = 2000.0;
const DIAL_MULTS: [f64; 12] = [
    0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.5, 7.0, 10.0, 15.0, 25.0, 40.0,
];
const FAMILIES: [&str; 3] = ["static", "gap", "defensive"];

struct Cell {
    idx: usize,
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
    manifest["cells"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(idx, c)| Cell {
            idx,
            stratum: c["stratum"].as_str().unwrap().into(),
            fee: c["fee"].as_f64().unwrap(),
            sigma: c["sigma"].as_f64().unwrap(),
            z: c["z"].as_f64().unwrap(),
            speed: c["arb_speed"].as_str().unwrap().into(),
            lam_arb: c["lambda_arb_star_SOLVED"].as_f64().unwrap(),
            lam_fund: c["lambda_fund_star_SOLVED"].as_f64().unwrap(),
        })
        .collect()
}

fn config_for(cell: &Cell) -> SimConfig {
    let y0 = 1.0e4;
    let d_ref = y0 * (1.0 - (1.0f64 + 0.01).powf(-0.5));
    SimConfig {
        name: "m3train".into(),
        description: "M3 step C training dial grid".into(),
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
        policy_lag: 300,
        dt_hours: 1.0 / 3600.0,
        pooled_fund_arrival_rate_per_hour: Some(cell.lam_fund),
        buy_arrival_share: 0.5,
        arb_arrival_rate_per_hour: Some(cell.lam_arb),
        lookback_hours: 20.0,
        arrival_model: ArrivalModel::Poisson,
        log_inactive_arb: false,
    }
}

fn make_policy(family: &str, dial_fee: f64) -> Box<dyn FeePolicy> {
    match family {
        "static" => Box::new(FixedFeePolicy::new(dial_fee)),
        "gap" => Box::new(OracleGapFeePolicy {
            base_fee: dial_fee,
            gap_multiplier: 2.0,
            min_fee: dial_fee,
            max_fee: 10.0 * dial_fee,
        }),
        _ => Box::new(OracleGapFeePolicy {
            base_fee: dial_fee,
            gap_multiplier: 50.0,
            min_fee: dial_fee,
            max_fee: 0.30,
        }),
    }
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_seeds: usize = arg(&args, "--n-seeds")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    assert!(n_seeds > 0, "--n-seeds is required (from the seed pilot)");
    let seed_start: u64 = arg(&args, "--seed-start")
        .and_then(|v| v.parse().ok())
        .unwrap_or(20_000);
    let shard: usize = arg(&args, "--shard")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let of: usize = arg(&args, "--of").and_then(|v| v.parse().ok()).unwrap_or(1);

    let cells = load_cells();
    let out_path = format!("/Users/joseph/amm-lab/.local/lvr/m3_training_rows_shard{shard}.csv.gz");
    let mut out = GzEncoder::new(File::create(&out_path).unwrap(), Compression::default());
    writeln!(
        out,
        "cell_idx,stratum,sigma,z,speed,family,dial_mult,dial_fee,seed,u,a,b,s,alloc"
    )
    .unwrap();

    let start = std::time::Instant::now();
    let mut n_done = 0usize;
    for cell in cells.iter().filter(|c| c.idx % of == shard) {
        let cfg0 = config_for(cell);
        for si in 0..n_seeds {
            let seed = seed_start + si as u64;
            let mut cfg = cfg0.clone();
            cfg.seed = seed;
            let prices = generate_gbm(cfg.n_steps, S0_PRICE, 0.0, cfg.sigma, cfg.dt_years(), seed);
            for family in FAMILIES {
                for dm in DIAL_MULTS {
                    let dial_fee = cell.fee * dm;
                    let mut pol = make_policy(family, dial_fee);
                    let (records, events) = run_simulation_with_events(&cfg, &prices, pol.as_mut());
                    let es = summarize_events(&events, &records);
                    writeln!(
                        out,
                        "{},{},{},{},{},{family},{dm},{dial_fee},{seed},{:.6},{:.6},{:.6},{:.6},{}",
                        cell.idx,
                        cell.stratum,
                        cell.sigma,
                        cell.z,
                        cell.speed,
                        es.u_lp_rel,
                        es.a_fill,
                        es.b_fill,
                        es.served_fund_volume,
                        es.alloc_amm_share
                            .map(|v| format!("{v:.6}"))
                            .unwrap_or_default(),
                    )
                    .unwrap();
                }
            }
        }
        n_done += 1;
        eprintln!(
            "[shard {shard}/{of}] cell {} ({} s{} z{} {}) done — {n_done} cells, {:.0}s elapsed",
            cell.idx,
            cell.stratum,
            cell.sigma,
            cell.z,
            cell.speed,
            start.elapsed().as_secs_f64()
        );
    }
    out.finish().unwrap();
    eprintln!("wrote {out_path}");
}
