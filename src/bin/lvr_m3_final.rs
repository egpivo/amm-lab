//! M3 Step E: untouched-final verification of one validation-selected pair.

use amm_lab::campbell::fee_policy::{FeePolicy, FixedFeePolicy, OracleGapFeePolicy};
use amm_lab::campbell::gbm::generate_gbm;
use amm_lab::campbell::simulation::{
    ArrivalModel, FlowRegime, SimConfig, run_simulation_with_events,
};
use amm_lab::campbell::summary::{summarize, summarize_events};
use flate2::Compression;
use flate2::write::GzEncoder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::Path;

const ROOT: &str = "/Users/joseph/amm-lab";
const FINAL_PLAN_REL: &str = ".local/lvr/m3_final_plan.json";
const FINAL_PLAN_HASH_REL: &str = ".local/lvr/m3_final_plan.sha256";
const MANIFEST_REL: &str = ".local/lvr/calibration_54_manifest.json";
const S0_PRICE: f64 = 2000.0;

#[derive(Deserialize)]
struct SeedBlock {
    start_inclusive: u64,
    end_exclusive: u64,
    n: usize,
}

#[derive(Deserialize)]
struct PlanCell {
    cell_idx: usize,
    stratum: String,
    sigma: f64,
    z: f64,
    speed: String,
}

#[derive(Deserialize)]
struct PlanPolicy {
    family: String,
    dial_mult: f64,
    dial_fee: f64,
}

#[derive(Deserialize)]
struct Candidate {
    candidate_id: String,
    cell: PlanCell,
    rho: f64,
    #[serde(rename = "policy_1_lower_A")]
    policy_1_lower_a: PlanPolicy,
    policy_2: PlanPolicy,
}

#[derive(Deserialize)]
struct FinalPlan {
    source_stage: String,
    final_seed_block: SeedBlock,
    candidate: Candidate,
    input_artifacts_sha256: BTreeMap<String, String>,
}

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

#[derive(Serialize)]
struct OutputRow<'a> {
    candidate_id: &'a str,
    policy_role: &'a str,
    cell_idx: usize,
    stratum: &'a str,
    sigma: f64,
    z: f64,
    speed: &'a str,
    rho: f64,
    family: &'a str,
    dial_mult: f64,
    dial_fee: f64,
    seed: u64,
    l: f64,
    a: f64,
    b: f64,
    l_arb: f64,
    u: f64,
    fees: f64,
    fees_arb: f64,
    fees_fund: f64,
    s: f64,
    potential: f64,
    alloc_amm: Option<f64>,
    alloc_cex: Option<f64>,
    alloc_unserved: Option<f64>,
    quote_error: Option<f64>,
    a_arb_per_served: Option<f64>,
    a_fund_per_served: Option<f64>,
    a_total_per_served: Option<f64>,
}

fn sha256(path: &Path) -> String {
    let mut file =
        File::open(path).unwrap_or_else(|e| panic!("cannot open {}: {e}", path.display()));
    let mut hasher = Sha256::new();
    let mut buf = [0_u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    format!("{:x}", hasher.finalize())
}

fn load_and_verify_plan() -> FinalPlan {
    let root = Path::new(ROOT);
    let path = root.join(FINAL_PLAN_REL);
    let expected = fs::read_to_string(root.join(FINAL_PLAN_HASH_REL))
        .expect("missing final plan hash")
        .split_whitespace()
        .next()
        .expect("empty final plan hash")
        .to_owned();
    assert_eq!(sha256(&path), expected, "final plan hash mismatch");
    let plan: FinalPlan =
        serde_json::from_reader(BufReader::new(File::open(&path).unwrap())).unwrap();
    assert_eq!(
        plan.source_stage,
        "validation-selected; no final-seed search"
    );
    assert_eq!(
        plan.final_seed_block.end_exclusive - plan.final_seed_block.start_inclusive,
        plan.final_seed_block.n as u64
    );
    for (relative, digest) in &plan.input_artifacts_sha256 {
        let input = root.join(relative);
        assert_eq!(
            sha256(&input),
            *digest,
            "frozen final-plan input changed: {}",
            input.display()
        );
    }
    plan
}

fn load_cell(idx: usize) -> Cell {
    let manifest: serde_json::Value = serde_json::from_reader(
        File::open(Path::new(ROOT).join(MANIFEST_REL)).expect("missing calibration manifest"),
    )
    .unwrap();
    let c = &manifest["cells"].as_array().unwrap()[idx];
    Cell {
        idx,
        stratum: c["stratum"].as_str().unwrap().into(),
        fee: c["fee"].as_f64().unwrap(),
        sigma: c["sigma"].as_f64().unwrap(),
        z: c["z"].as_f64().unwrap(),
        speed: c["arb_speed"].as_str().unwrap().into(),
        lam_arb: c["lambda_arb_star_SOLVED"].as_f64().unwrap(),
        lam_fund: c["lambda_fund_star_SOLVED"].as_f64().unwrap(),
    }
}

fn config_for(cell: &Cell) -> SimConfig {
    let y0 = 1.0e4;
    let d_ref = y0 * (1.0 - (1.0f64 + 0.01).powf(-0.5));
    SimConfig {
        name: "m3final".into(),
        description: "M3 step E untouched-final verification".into(),
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

fn make_policy(spec: &PlanPolicy) -> Box<dyn FeePolicy> {
    match spec.family.as_str() {
        "static" => Box::new(FixedFeePolicy::new(spec.dial_fee)),
        "gap" => Box::new(OracleGapFeePolicy {
            base_fee: spec.dial_fee,
            gap_multiplier: 2.0,
            min_fee: spec.dial_fee,
            max_fee: 10.0 * spec.dial_fee,
        }),
        other => panic!("unsupported final policy family: {other}"),
    }
}

fn main() {
    let plan = load_and_verify_plan();
    let cell = load_cell(plan.candidate.cell.cell_idx);
    assert_eq!(cell.idx, plan.candidate.cell.cell_idx);
    assert_eq!(cell.stratum, plan.candidate.cell.stratum);
    assert_eq!(cell.speed, plan.candidate.cell.speed);
    assert_eq!(cell.sigma.to_bits(), plan.candidate.cell.sigma.to_bits());
    assert_eq!(cell.z.to_bits(), plan.candidate.cell.z.to_bits());

    let out_path = Path::new(ROOT).join(".local/lvr/m3_final_rows.csv.gz");
    let encoder = GzEncoder::new(File::create(&out_path).unwrap(), Compression::default());
    let mut out = csv::Writer::from_writer(encoder);
    let cfg0 = config_for(&cell);
    let policies = [
        ("policy_1_lower_A", &plan.candidate.policy_1_lower_a),
        ("policy_2", &plan.candidate.policy_2),
    ];
    let start = std::time::Instant::now();

    for (i, seed) in
        (plan.final_seed_block.start_inclusive..plan.final_seed_block.end_exclusive).enumerate()
    {
        let mut cfg = cfg0.clone();
        cfg.seed = seed;
        let prices = generate_gbm(cfg.n_steps, S0_PRICE, 0.0, cfg.sigma, cfg.dt_years(), seed);
        for (role, spec) in policies {
            let mut policy = make_policy(spec);
            let (records, events) = run_simulation_with_events(&cfg, &prices, policy.as_mut());
            let es = summarize_events(&events, &records);
            let quote_error = summarize(&records).mean_abs_log_gap;
            out.serialize(OutputRow {
                candidate_id: &plan.candidate.candidate_id,
                policy_role: role,
                cell_idx: cell.idx,
                stratum: &cell.stratum,
                sigma: cell.sigma,
                z: cell.z,
                speed: &cell.speed,
                rho: plan.candidate.rho,
                family: &spec.family,
                dial_mult: spec.dial_mult,
                dial_fee: spec.dial_fee,
                seed,
                l: es.l_total,
                a: es.a_fill,
                b: es.b_fill,
                l_arb: es.a_arb,
                u: es.u_lp_rel,
                fees: es.fees_total,
                fees_arb: es.fees_arb,
                fees_fund: es.fees_fund,
                s: es.served_fund_volume,
                potential: es.potential_volume,
                alloc_amm: es.alloc_amm_share,
                alloc_cex: es.alloc_cex_share,
                alloc_unserved: es.alloc_unserved_share,
                quote_error,
                a_arb_per_served: es.a_arb_per_served_unit,
                a_fund_per_served: es.a_fund_per_served_unit,
                a_total_per_served: es.a_total_per_served_unit,
            })
            .unwrap();
        }
        if (i + 1) % 50 == 0 {
            eprintln!(
                "final seeds {}/{} done - {:.0}s elapsed",
                i + 1,
                plan.final_seed_block.n,
                start.elapsed().as_secs_f64()
            );
        }
    }
    out.flush().unwrap();
    let encoder = out.into_inner().unwrap();
    encoder.finish().unwrap();
    eprintln!("wrote {}", out_path.display());
}
