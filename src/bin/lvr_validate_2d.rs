//! Fresh validation for the training-frozen amended validation-grid policy family.
//!
//! The binary verifies the immutable plan and its inputs, and refuses any
//! seed block other than the fresh amended-validation block in that plan.

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
const PLAN_REL: &str = ".local/lvr/m3_amended_validation_plan.json";
const PLAN_HASH_REL: &str = ".local/lvr/m3_amended_validation_plan.sha256";
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
    f0: f64,
    alpha: f64,
    fee_cap: f64,
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
    gap_policy: PlanPolicy,
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
    f0: f64,
    alpha: f64,
    fee_cap: f64,
    seed: u64,
    l: f64,
    a: f64,
    b: f64,
    a_arb: f64,
    a_fund: f64,
    b_fund: f64,
    u: f64,
    fees: f64,
    fees_arb: f64,
    fees_fund: f64,
    s: f64,
    potential: f64,
    alloc_amm: Option<f64>,
    alloc_cex: Option<f64>,
    alloc_unserved: Option<f64>,
    fill_incidence: Option<f64>,
    conditional_fill_size: Option<f64>,
    quote_error: Option<f64>,
    a_arb_per_served: Option<f64>,
    a_fund_per_served: Option<f64>,
    a_total_per_served: Option<f64>,
    n_fund_events: u64,
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
    let recorded_hash = fs::read_to_string(root.join(PLAN_HASH_REL))
        .expect("missing frozen plan hash")
        .split_whitespace()
        .next()
        .expect("empty frozen plan hash")
        .to_owned();
    assert_eq!(sha256(&plan_path), recorded_hash, "validation plan changed");
    let plan: ValidationPlan =
        serde_json::from_reader(BufReader::new(File::open(&plan_path).unwrap())).unwrap();
    assert_eq!(
        plan.source_stage,
        "amended two-dimensional family; training-only selection and diagnostics"
    );
    assert_eq!(plan.validation_seed_block.start_inclusive, 40_000);
    assert_eq!(plan.validation_seed_block.end_exclusive, 40_200);
    assert_eq!(plan.validation_seed_block.n, 200);
    assert_eq!(
        plan.matched_candidates.len(),
        106,
        "candidate universe changed"
    );
    assert_eq!(plan.frontier_pairs.len(), 270, "frontier universe changed");
    for (relative, expected) in &plan.input_artifacts_sha256 {
        let path = root.join(relative);
        assert_eq!(
            sha256(&path),
            *expected,
            "frozen input changed: {}",
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
        name: "m3validate2d".into(),
        description: "validation-grid amended fresh held-out validation".into(),
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
    assert!((0.0..=0.30).contains(&spec.f0));
    assert!((0.0..=2.0).contains(&spec.alpha));
    assert_eq!(spec.fee_cap.to_bits(), 0.30_f64.to_bits());
    if spec.alpha == 0.0 {
        Box::new(FixedFeePolicy::new(spec.f0))
    } else {
        assert_eq!(spec.family, "gap");
        Box::new(OracleGapFeePolicy {
            base_fee: spec.f0,
            gap_multiplier: spec.alpha,
            min_fee: spec.f0,
            max_fee: spec.fee_cap,
        })
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
            ("gap_family", &frontier.gap_policy),
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
    if args.iter().any(|arg| arg == "--verify-only") {
        let n_assignments: usize = assignments.values().map(Vec::len).sum();
        assert_eq!(n_assignments, 752);
        eprintln!(
            "verified amended validation plan: {} cells, {n_assignments} assignments",
            cells.len()
        );
        return;
    }
    let out_path = PathBuf::from(format!(
        "{ROOT}/.local/lvr/m3_amended_validation_rows_shard{shard}.csv.gz"
    ));
    assert!(
        !out_path.exists(),
        "refusing to overwrite {}",
        out_path.display()
    );
    let encoder = GzEncoder::new(File::create(&out_path).unwrap(), Compression::default());
    let mut out = csv::Writer::from_writer(encoder);
    let start = std::time::Instant::now();
    let mut n_done = 0usize;

    for cell in cells.iter().filter(|c| c.idx % of == shard) {
        let cell_assignments = assignments.get(&cell.idx).cloned().unwrap_or_default();
        let mut policies: BTreeMap<(u64, u64, u64), Vec<Assignment>> = BTreeMap::new();
        for assignment in cell_assignments {
            let key = (
                assignment.policy.f0.to_bits(),
                assignment.policy.alpha.to_bits(),
                assignment.policy.fee_cap.to_bits(),
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
                let spec = &group[0].policy;
                assert!(group.iter().all(|a| {
                    a.policy.f0.to_bits() == spec.f0.to_bits()
                        && a.policy.alpha.to_bits() == spec.alpha.to_bits()
                        && a.policy.fee_cap.to_bits() == spec.fee_cap.to_bits()
                }));
                let mut policy = make_policy(spec);
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
                        f0: assignment.policy.f0,
                        alpha: assignment.policy.alpha,
                        fee_cap: assignment.policy.fee_cap,
                        seed,
                        l: es.l_total,
                        a: es.a_fill,
                        b: es.b_fill,
                        a_arb: es.a_arb,
                        a_fund: es.a_fund,
                        b_fund: es.b_fund,
                        u: es.u_lp_rel,
                        fees: es.fees_total,
                        fees_arb: es.fees_arb,
                        fees_fund: es.fees_fund,
                        s: es.served_fund_volume,
                        potential: es.potential_volume,
                        alloc_amm: es.alloc_amm_share,
                        alloc_cex: es.alloc_cex_share,
                        alloc_unserved: es.alloc_unserved_share,
                        fill_incidence: es.incidence_event,
                        conditional_fill_size: es.cond_fill_size,
                        quote_error,
                        a_arb_per_served: es.a_arb_per_served_unit,
                        a_fund_per_served: es.a_fund_per_served_unit,
                        a_total_per_served: es.a_total_per_served_unit,
                        n_fund_events: es.n_fund_events,
                    })
                    .unwrap();
                }
            }
        }
        n_done += 1;
        eprintln!(
            "[amended validation {shard}/{of}] cell {} done - {n_done} cells, {:.0}s elapsed",
            cell.idx,
            start.elapsed().as_secs_f64()
        );
    }
    out.flush().unwrap();
    let encoder = out.into_inner().unwrap();
    encoder.finish().unwrap();
    eprintln!("wrote {}", out_path.display());
}
