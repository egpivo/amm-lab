//! M3 disclosed design amendment: genuine (f0, alpha) gap-family grid.
//!
//! Training seeds only. The alpha=0 gap members are exact aliases of the
//! static simulations and are written explicitly without duplicate engine
//! runs. Positive-alpha members use a cap fixed independently of f0.

use amm_lab::campbell::fee_policy::{FixedFeePolicy, OracleGapFeePolicy};
use amm_lab::campbell::gbm::generate_gbm;
use amm_lab::campbell::simulation::{
    ArrivalModel, FlowRegime, SimConfig, run_simulation_with_events,
};
use amm_lab::campbell::summary::{EventSummary, summarize, summarize_events};
use flate2::Compression;
use flate2::write::GzEncoder;
use std::fs::File;
use std::io::Write;

const S0_PRICE: f64 = 2000.0;
const FEE_CAP: f64 = 0.30;
const DIAL_MULTS: [f64; 12] = [
    0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.5, 7.0, 10.0, 15.0, 25.0, 40.0,
];
const POSITIVE_ALPHAS: [f64; 6] = [0.05, 0.10, 0.25, 0.50, 1.0, 2.0];

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
        name: "m3train2d".into(),
        description: "M3 amended two-dimensional gap grid, training only".into(),
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

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

#[allow(clippy::too_many_arguments)]
fn write_row(
    out: &mut GzEncoder<File>,
    cell: &Cell,
    family: &str,
    dial_mult: f64,
    f0: f64,
    alpha: f64,
    alpha_zero_static_alias: bool,
    seed: u64,
    es: &EventSummary,
    quote_error: Option<f64>,
) {
    let option = |value: Option<f64>| value.map(|v| format!("{v:.12}")).unwrap_or_default();
    writeln!(
        out,
        "{},{},{},{},{},{family},{dial_mult},{f0},{alpha},{FEE_CAP},{alpha_zero_static_alias},{seed},\
         {:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{},\
         {},{},{},{},{},{},{},{},{},{}",
        cell.idx,
        cell.stratum,
        cell.sigma,
        cell.z,
        cell.speed,
        es.l_total,
        es.a_fill,
        es.b_fill,
        es.a_arb,
        es.a_fund,
        es.b_fund,
        es.u_lp_rel,
        es.fees_total,
        es.fees_arb,
        es.fees_fund,
        es.served_fund_volume,
        es.potential_volume,
        option(es.alloc_amm_share),
        option(es.alloc_cex_share),
        option(es.alloc_unserved_share),
        option(es.incidence_event),
        option(es.cond_fill_size),
        option(es.a_arb_per_served_unit),
        option(es.a_fund_per_served_unit),
        option(es.a_total_per_served_unit),
        option(quote_error),
        es.n_fund_events,
    )
    .unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_seeds: usize = arg(&args, "--n-seeds")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    assert!(n_seeds > 0, "--n-seeds is required");
    let seed_start: u64 = arg(&args, "--seed-start")
        .and_then(|v| v.parse().ok())
        .unwrap_or(20_000);
    assert_eq!(seed_start, 20_000, "amended runner is training-block only");
    assert_eq!(
        n_seeds, 100,
        "amended training block is frozen at 100 seeds"
    );
    let shard: usize = arg(&args, "--shard")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let of: usize = arg(&args, "--of").and_then(|v| v.parse().ok()).unwrap_or(1);
    assert!(of > 0 && shard < of);

    let out_path =
        format!("/Users/joseph/amm-lab/.local/lvr/m3_amended_training_rows_shard{shard}.csv.gz");
    let mut out = GzEncoder::new(File::create(&out_path).unwrap(), Compression::default());
    writeln!(
        out,
        "cell_idx,stratum,sigma,z,speed,family,dial_mult,f0,alpha,fee_cap,alpha_zero_static_alias,seed,\
         l,a,b,a_arb,a_fund,b_fund,u,fees,fees_arb,fees_fund,s,potential,alloc_amm,alloc_cex,\
         alloc_unserved,fill_incidence,conditional_fill_size,a_arb_per_served,a_fund_per_served,\
         a_total_per_served,quote_error,n_fund_events"
    )
    .unwrap();

    let cells = load_cells();
    let start = std::time::Instant::now();
    let mut n_done = 0usize;
    for cell in cells.iter().filter(|c| c.idx % of == shard) {
        let cfg0 = config_for(cell);
        for si in 0..n_seeds {
            let seed = seed_start + si as u64;
            let mut cfg = cfg0.clone();
            cfg.seed = seed;
            let prices = generate_gbm(cfg.n_steps, S0_PRICE, 0.0, cfg.sigma, cfg.dt_years(), seed);
            for dial_mult in DIAL_MULTS {
                let f0 = cell.fee * dial_mult;
                let mut policy = FixedFeePolicy::new(f0);
                let (records, events) = run_simulation_with_events(&cfg, &prices, &mut policy);
                let es = summarize_events(&events, &records);
                let quote_error = summarize(&records).mean_abs_log_gap;
                write_row(
                    &mut out,
                    cell,
                    "static",
                    dial_mult,
                    f0,
                    0.0,
                    false,
                    seed,
                    &es,
                    quote_error,
                );
                write_row(
                    &mut out,
                    cell,
                    "gap",
                    dial_mult,
                    f0,
                    0.0,
                    true,
                    seed,
                    &es,
                    quote_error,
                );
            }
            for alpha in POSITIVE_ALPHAS {
                for dial_mult in DIAL_MULTS {
                    let f0 = cell.fee * dial_mult;
                    let mut policy = OracleGapFeePolicy {
                        base_fee: f0,
                        gap_multiplier: alpha,
                        min_fee: f0,
                        max_fee: FEE_CAP,
                    };
                    let (records, events) = run_simulation_with_events(&cfg, &prices, &mut policy);
                    let es = summarize_events(&events, &records);
                    let quote_error = summarize(&records).mean_abs_log_gap;
                    write_row(
                        &mut out,
                        cell,
                        "gap",
                        dial_mult,
                        f0,
                        alpha,
                        false,
                        seed,
                        &es,
                        quote_error,
                    );
                }
            }
        }
        n_done += 1;
        eprintln!(
            "[amended shard {shard}/{of}] cell {} ({} s{} z{} {}) done - {n_done} cells, {:.0}s elapsed",
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
