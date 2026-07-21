//! M3 Step D: validation of the training-frozen candidate universe.
//!
//! The binary refuses to run unless the validation plan and every input
//! artifact recorded by that plan still match their frozen SHA-256 hashes.
//! Validation seeds and policies come only from the plan; sharding changes
//! execution placement, not the experiment definition.

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
use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

const ROOT: &str = "/Users/joseph/amm-lab";
const PLAN_REL: &str = ".local/lvr/m3_validation_plan.json";
const PLAN_HASH_REL: &str = ".local/lvr/m3_validation_plan.sha256";
const MANIFEST_REL: &str = ".local/lvr/calibration_54_manifest.json";
const S0_PRICE: f64 = 2000.0;

#[derive(Deserialize)]
struct SeedBlock {
    start_inclusive: u64,
    end_exclusive: u64,
    n: usize,
}

#[derive(Clone, Deserialize)]
struct PlanCell {
    cell_idx: usize,
    stratum: String,
    sigma: f64,
    z: f64,
    speed: String,
}

#[derive(Clone, Deserialize)]
struct PlanPolicy {
    family: String,
    dial_mult: f64,
    dial_fee: f64,
}

#[derive(Deserialize)]
struct MatchedCandidate {
    candidate_id: String,
    cell: PlanCell,
    rho: f64,
    #[serde(rename = "policy_1_lower_A")]
    policy_1_lower_a: PlanPolicy,
    policy_2: PlanPolicy,
}

#[derive(Deserialize)]
struct FrontierPair {
    frontier_id: String,
    cell: PlanCell,
    rho: f64,
    static_policy: PlanPolicy,
    adaptive_policy: PlanPolicy,
}

#[derive(Deserialize)]
struct ValidationPlan {
    source_stage: String,
    validation_seed_block: SeedBlock,
    input_artifacts_sha256: BTreeMap<String, String>,
    matched_candidates: Vec<MatchedCandidate>,
    frontier_pairs: Vec<FrontierPair>,
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

#[derive(Clone)]
struct Assignment {
    record_id: String,
    record_kind: String,
    policy_role: String,
    rho: f64,
    policy: PlanPolicy,
}

#[derive(Serialize)]
struct OutputRow<'a> {
    record_id: &'a str,
    record_kind: &'a str,
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
        let n = file
            .read(&mut buf)
            .unwrap_or_else(|e| panic!("cannot hash {}: {e}", path.display()));
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    format!("{:x}", hasher.finalize())
}

fn load_and_verify_plan() -> ValidationPlan {
    let root = Path::new(ROOT);
    let plan_path = root.join(PLAN_REL);
    let recorded_plan_hash = fs::read_to_string(root.join(PLAN_HASH_REL))
        .expect("missing frozen plan hash")
        .split_whitespace()
        .next()
        .expect("empty frozen plan hash")
        .to_owned();
    assert_eq!(
        sha256(&plan_path),
        recorded_plan_hash,
        "validation plan differs from its frozen hash"
    );

    let plan: ValidationPlan =
        serde_json::from_reader(BufReader::new(File::open(&plan_path).unwrap())).unwrap();
    assert_eq!(plan.source_stage, "training only");
    assert_eq!(
        plan.validation_seed_block.end_exclusive - plan.validation_seed_block.start_inclusive,
        plan.validation_seed_block.n as u64,
        "invalid validation seed block"
    );
    assert_eq!(
        plan.matched_candidates.len(),
        34,
        "candidate universe changed"
    );
    assert_eq!(plan.frontier_pairs.len(), 270, "frontier universe changed");

    for (relative, expected) in &plan.input_artifacts_sha256 {
        let path = root.join(relative);
        assert_eq!(
            sha256(&path),
            *expected,
            "frozen input artifact changed: {}",
            path.display()
        );
    }
    plan
}

fn load_cells() -> Vec<Cell> {
    let manifest: serde_json::Value = serde_json::from_reader(
        File::open(Path::new(ROOT).join(MANIFEST_REL)).expect("missing calibration manifest"),
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
        name: "m3validate".into(),
        description: "M3 step D frozen-policy validation".into(),
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
        other => panic!("validation plan contains unsupported policy family: {other}"),
    }
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn check_cell(plan_cell: &PlanCell, cell: &Cell) {
    assert_eq!(plan_cell.cell_idx, cell.idx);
    assert_eq!(plan_cell.stratum, cell.stratum);
    assert_eq!(plan_cell.speed, cell.speed);
    assert_eq!(plan_cell.sigma.to_bits(), cell.sigma.to_bits());
    assert_eq!(plan_cell.z.to_bits(), cell.z.to_bits());
}

fn assignments_by_cell(plan: &ValidationPlan, cells: &[Cell]) -> HashMap<usize, Vec<Assignment>> {
    let mut result: HashMap<usize, Vec<Assignment>> = HashMap::new();
    for candidate in &plan.matched_candidates {
        check_cell(&candidate.cell, &cells[candidate.cell.cell_idx]);
        for (role, policy) in [
            ("policy_1_lower_A", &candidate.policy_1_lower_a),
            ("policy_2", &candidate.policy_2),
        ] {
            result
                .entry(candidate.cell.cell_idx)
                .or_default()
                .push(Assignment {
                    record_id: candidate.candidate_id.clone(),
                    record_kind: "matched".into(),
                    policy_role: role.into(),
                    rho: candidate.rho,
                    policy: policy.clone(),
                });
        }
    }
    for frontier in &plan.frontier_pairs {
        check_cell(&frontier.cell, &cells[frontier.cell.cell_idx]);
        for (role, policy) in [
            ("static", &frontier.static_policy),
            ("adaptive", &frontier.adaptive_policy),
        ] {
            result
                .entry(frontier.cell.cell_idx)
                .or_default()
                .push(Assignment {
                    record_id: frontier.frontier_id.clone(),
                    record_kind: "frontier".into(),
                    policy_role: role.into(),
                    rho: frontier.rho,
                    policy: policy.clone(),
                });
        }
    }
    result
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let shard: usize = arg(&args, "--shard")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let of: usize = arg(&args, "--of").and_then(|v| v.parse().ok()).unwrap_or(1);
    assert!(of > 0 && shard < of, "require 0 <= shard < of");

    let plan = load_and_verify_plan();
    let cells = load_cells();
    let assignments = assignments_by_cell(&plan, &cells);
    let out_path = PathBuf::from(format!(
        "{ROOT}/.local/lvr/m3_validation_rows_shard{shard}.csv.gz"
    ));
    let encoder = GzEncoder::new(File::create(&out_path).unwrap(), Compression::default());
    let mut out = csv::Writer::from_writer(encoder);
    let start = std::time::Instant::now();
    let mut n_done = 0usize;

    for cell in cells.iter().filter(|c| c.idx % of == shard) {
        let cell_assignments = assignments.get(&cell.idx).cloned().unwrap_or_default();
        let mut policies: BTreeMap<(String, u64), Vec<Assignment>> = BTreeMap::new();
        for assignment in cell_assignments {
            let key = (
                assignment.policy.family.clone(),
                assignment.policy.dial_mult.to_bits(),
            );
            policies.entry(key).or_default().push(assignment);
        }

        let cfg0 = config_for(cell);
        for seed in
            plan.validation_seed_block.start_inclusive..plan.validation_seed_block.end_exclusive
        {
            let mut cfg = cfg0.clone();
            cfg.seed = seed;
            let prices = generate_gbm(cfg.n_steps, S0_PRICE, 0.0, cfg.sigma, cfg.dt_years(), seed);
            for group in policies.values() {
                let policy_spec = &group[0].policy;
                assert!(group.iter().all(|a| {
                    a.policy.family == policy_spec.family
                        && a.policy.dial_mult.to_bits() == policy_spec.dial_mult.to_bits()
                        && a.policy.dial_fee.to_bits() == policy_spec.dial_fee.to_bits()
                }));
                let mut policy = make_policy(&policy_spec.family, policy_spec.dial_fee);
                let (records, events) = run_simulation_with_events(&cfg, &prices, policy.as_mut());
                let es = summarize_events(&events, &records);
                let quote_error = summarize(&records).mean_abs_log_gap;

                for assignment in group {
                    out.serialize(OutputRow {
                        record_id: &assignment.record_id,
                        record_kind: &assignment.record_kind,
                        policy_role: &assignment.policy_role,
                        cell_idx: cell.idx,
                        stratum: &cell.stratum,
                        sigma: cell.sigma,
                        z: cell.z,
                        speed: &cell.speed,
                        rho: assignment.rho,
                        family: &assignment.policy.family,
                        dial_mult: assignment.policy.dial_mult,
                        dial_fee: assignment.policy.dial_fee,
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
            }
        }
        n_done += 1;
        eprintln!(
            "[shard {shard}/{of}] cell {} ({} s{} z{} {}) done - {n_done} cells, {:.0}s elapsed",
            cell.idx,
            cell.stratum,
            cell.sigma,
            cell.z,
            cell.speed,
            start.elapsed().as_secs_f64()
        );
    }
    out.flush().unwrap();
    let encoder = out.into_inner().unwrap();
    encoder.finish().unwrap();
    eprintln!("wrote {}", out_path.display());
}
