//! Round-34 P2: local finite-grid refinement on training seeds only.
//!
//! The policy manifest is frozen before this runner executes. Original coarse
//! policies are not rerun; the analyzer combines their frozen means with these
//! newly simulated midpoint policies.

use amm_lab::campbell::fee_policy::{FixedFeePolicy, OracleGapFeePolicy};
use amm_lab::campbell::gbm::generate_gbm;
use amm_lab::campbell::simulation::{
    ArrivalModel, FlowRegime, SimConfig, run_simulation_with_events_compact,
};
use amm_lab::campbell::summary::summarize_events;
use flate2::Compression;
use flate2::write::GzEncoder;
use serde::Deserialize;
use std::env;
use std::fs::File;
use std::io::Write;
use std::sync::Arc;

const ROOT: &str = "/Users/joseph/amm-lab";
const POLICY_MANIFEST: &str = "/Users/joseph/amm-lab/.local/lvr/m3_local_refinement_policies.json";
const SEED_START: u64 = 20_000;
const SEED_END_EXCLUSIVE: u64 = 20_100;
const S0_PRICE: f64 = 2_000.0;
const FEE_CAP: f64 = 0.30;
const N_STEPS: usize = 604_800;
const N_SHARDS: usize = 6;
const IDENTITY_TOLERANCE: f64 = 1e-10;

#[derive(Clone)]
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

#[derive(Clone, Deserialize)]
struct PolicySpec {
    policy_id: String,
    dial_mult: f64,
    f0: f64,
    alpha: f64,
    fee_cap: f64,
}

#[derive(Deserialize)]
struct ManifestCell {
    cell_idx: usize,
    new_policies: Vec<PolicySpec>,
}

#[derive(Deserialize)]
struct PolicyManifest {
    schema_version: String,
    seed_block: SeedBlock,
    cells: Vec<ManifestCell>,
}

#[derive(Deserialize)]
struct SeedBlock {
    start_inclusive: u64,
    end_exclusive: u64,
    n: usize,
}

fn required_hash_arg(name: &str) -> String {
    let args: Vec<String> = env::args().collect();
    let position = args
        .iter()
        .position(|arg| arg == name)
        .unwrap_or_else(|| panic!("missing {name}"));
    let value = args
        .get(position + 1)
        .unwrap_or_else(|| panic!("missing value for {name}"));
    assert_eq!(value.len(), 64, "{name} must be a SHA-256 digest");
    assert!(value.bytes().all(|byte| byte.is_ascii_hexdigit()));
    value.clone()
}

fn load_cells() -> Vec<Cell> {
    let path = format!("{ROOT}/.local/lvr/calibration_54_manifest.json");
    let manifest: serde_json::Value =
        serde_json::from_reader(File::open(path).expect("open calibration manifest"))
            .expect("parse calibration manifest");
    manifest["cells"]
        .as_array()
        .expect("calibration cells array")
        .iter()
        .enumerate()
        .map(|(idx, cell)| Cell {
            idx,
            stratum: cell["stratum"].as_str().unwrap().into(),
            fee: cell["fee"].as_f64().unwrap(),
            sigma: cell["sigma"].as_f64().unwrap(),
            z: cell["z"].as_f64().unwrap(),
            speed: cell["arb_speed"].as_str().unwrap().into(),
            lam_arb: cell["lambda_arb_star_SOLVED"].as_f64().unwrap(),
            lam_fund: cell["lambda_fund_star_SOLVED"].as_f64().unwrap(),
        })
        .collect()
}

fn load_policies() -> Vec<Vec<PolicySpec>> {
    let manifest: PolicyManifest =
        serde_json::from_reader(File::open(POLICY_MANIFEST).expect("open refinement manifest"))
            .expect("parse refinement manifest");
    assert_eq!(manifest.schema_version, "m3-local-refinement-policies-v1");
    assert_eq!(manifest.seed_block.start_inclusive, SEED_START);
    assert_eq!(manifest.seed_block.end_exclusive, SEED_END_EXCLUSIVE);
    assert_eq!(manifest.seed_block.n, 100);
    assert_eq!(manifest.cells.len(), 54);
    let mut policies = vec![Vec::new(); 54];
    for cell in manifest.cells {
        assert!(cell.cell_idx < 54);
        assert!(policies[cell.cell_idx].is_empty());
        for policy in &cell.new_policies {
            assert!(policy.alpha >= 0.0 && policy.alpha <= 2.0);
            assert!(policy.dial_mult >= 0.5 && policy.dial_mult <= 40.0);
            assert_eq!(policy.fee_cap, FEE_CAP);
        }
        policies[cell.cell_idx] = cell.new_policies;
    }
    assert!(policies.iter().all(|cell| !cell.is_empty()));
    policies
}

fn config_for(cell: &Cell, seed: u64) -> SimConfig {
    let reserve_y = 1.0e4;
    let d_ref = reserve_y * (1.0 - (1.0f64 + 0.01).powf(-0.5));
    SimConfig {
        name: "m3_local_refinement".into(),
        description: "Round-34 P2 local refinement, training only".into(),
        amm_fee: cell.fee,
        cex_fee: 0.0010,
        buy_demand: cell.z * d_ref,
        sell_demand: cell.z * d_ref,
        reserve_x: 2.0e7,
        reserve_y,
        sigma: cell.sigma,
        mu: 0.0,
        n_steps: N_STEPS,
        seed,
        flow_regime: FlowRegime::Normal,
        toxic_burst_prob: 0.0,
        toxic_burst_arb_scale: 1.0,
        toxic_burst_fund_scale: 1.0,
        regime_switch_period: 0,
        e1_lambda: 0.0,
        e1_fee_ref: 0.0006,
        e5_arb_prob: 1.0,
        policy_lag: 300,
        dt_hours: 1.0 / 3_600.0,
        pooled_fund_arrival_rate_per_hour: Some(cell.lam_fund),
        buy_arrival_share: 0.5,
        arb_arrival_rate_per_hour: Some(cell.lam_arb),
        lookback_hours: 20.0,
        arrival_model: ArrivalModel::Poisson,
        log_inactive_arb: false,
    }
}

fn close(left: f64, right: f64) -> bool {
    let scale = left.abs().max(right.abs()).max(1.0);
    (left - right).abs() <= IDENTITY_TOLERANCE * scale
}

fn run_shard(shard: usize, cells: &[Cell], policies: &[Vec<PolicySpec>]) {
    let output_path = format!("{ROOT}/.local/lvr/m3_local_refinement_rows_shard{shard}.csv.gz");
    let output = File::create(&output_path).expect("create refinement shard");
    let mut writer = GzEncoder::new(output, Compression::default());
    writeln!(
        writer,
        "cell_idx,stratum,sigma,z,speed,policy_id,dial_mult,f0,alpha,fee_cap,seed,l,a,b,u,fees,s"
    )
    .unwrap();

    let started = std::time::Instant::now();
    let mut cell_count = 0usize;
    let mut row_count = 0usize;
    for cell in cells.iter().filter(|cell| cell.idx % N_SHARDS == shard) {
        for seed in SEED_START..SEED_END_EXCLUSIVE {
            let config = config_for(cell, seed);
            let prices = generate_gbm(
                config.n_steps,
                S0_PRICE,
                0.0,
                config.sigma,
                config.dt_years(),
                seed,
            );
            for policy in &policies[cell.idx] {
                let summary = if policy.alpha == 0.0 {
                    let mut fee_policy = FixedFeePolicy::new(policy.f0);
                    let (records, events) =
                        run_simulation_with_events_compact(&config, &prices, &mut fee_policy);
                    summarize_events(&events, &records)
                } else {
                    let mut fee_policy = OracleGapFeePolicy {
                        base_fee: policy.f0,
                        gap_multiplier: policy.alpha,
                        min_fee: policy.f0,
                        max_fee: policy.fee_cap,
                    };
                    let (records, events) =
                        run_simulation_with_events_compact(&config, &prices, &mut fee_policy);
                    summarize_events(&events, &records)
                };
                assert!(close(summary.l_total, summary.a_fill - summary.b_fill));
                assert!(close(
                    summary.u_lp_rel,
                    summary.fees_total - summary.l_total
                ));
                writeln!(
                    writer,
                    "{},{},{},{},{},{},{:.17e},{:.17e},{:.17e},{:.17e},{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e}",
                    cell.idx,
                    cell.stratum,
                    cell.sigma,
                    cell.z,
                    cell.speed,
                    policy.policy_id,
                    policy.dial_mult,
                    policy.f0,
                    policy.alpha,
                    policy.fee_cap,
                    seed,
                    summary.l_total,
                    summary.a_fill,
                    summary.b_fill,
                    summary.u_lp_rel,
                    summary.fees_total,
                    summary.served_fund_volume,
                )
                .unwrap();
                row_count += 1;
            }
        }
        cell_count += 1;
        eprintln!(
            "[P2 shard {shard}/{N_SHARDS}] cell {} done; cells={cell_count}, rows={row_count}, elapsed={:.0}s",
            cell.idx,
            started.elapsed().as_secs_f64()
        );
    }
    writer.finish().expect("finish refinement shard");
    eprintln!("[P2 shard {shard}] wrote {output_path}; rows={row_count}");
}

fn main() {
    let plan_sha256 = required_hash_arg("--plan-sha256");
    let runner_sha256 = required_hash_arg("--runner-sha256");
    let policy_manifest_sha256 = required_hash_arg("--policy-manifest-sha256");
    eprintln!(
        "P2 frozen inputs plan={plan_sha256} runner={runner_sha256} policies={policy_manifest_sha256}"
    );

    let cells = Arc::new(load_cells());
    let policies = Arc::new(load_policies());
    std::thread::scope(|scope| {
        for shard in 0..N_SHARDS {
            let cells = Arc::clone(&cells);
            let policies = Arc::clone(&policies);
            scope.spawn(move || run_shard(shard, &cells, &policies));
        }
    });
}
