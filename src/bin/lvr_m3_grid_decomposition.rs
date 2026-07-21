//! Validation-grid paired-ledger decomposition (frontier + selector contrasts).

use amm_lab::campbell::fee_policy::{FeePolicy, FixedFeePolicy, OracleGapFeePolicy};
use amm_lab::campbell::gbm::generate_gbm;
use amm_lab::campbell::paired_decomposition::{
    AggregateDeltas, SeedDecomp, decompose_pair, summarize_run,
};
use amm_lab::campbell::simulation::{
    ArrivalModel, EventRecord, FlowRegime, SimConfig, StepRecord, run_simulation_with_events,
};
use flate2::Compression;
use flate2::write::GzEncoder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

const ROOT: &str = env!("CARGO_MANIFEST_DIR");
const VALIDATION_PLAN_REL: &str = ".local/lvr/m3_amended_validation_plan.json";
const VALIDATION_PLAN_HASH_REL: &str = ".local/lvr/m3_amended_validation_plan.sha256";
const SELECTOR_CSV_REL: &str = ".local/lvr/m3_constrained_divergence.csv";
const MANIFEST_REL: &str = ".local/lvr/calibration_54_manifest.json";
const S0_PRICE: f64 = 2000.0;

#[derive(Deserialize)]
struct SeedBlock {
    start_inclusive: u64,
    end_exclusive: u64,
}

#[derive(Clone, Deserialize)]
struct PlanCell {
    cell_idx: usize,
    stratum: String,
    sigma: f64,
    z: f64,
    speed: String,
}

#[derive(Clone, Deserialize, Serialize)]
struct PlanPolicy {
    family: String,
    dial_mult: f64,
    f0: f64,
    alpha: f64,
    fee_cap: f64,
}

#[derive(Deserialize)]
struct FrontierPair {
    frontier_id: String,
    cell: PlanCell,
    rho: f64,
    target_s_training: f64,
    static_policy: PlanPolicy,
    gap_policy: PlanPolicy,
    empirical_support: EmpiricalSupport,
}

#[derive(Clone, Deserialize, Serialize)]
struct EmpiricalSupport {
    support_label: String,
    observed_pool_weeks: u64,
    stratum: String,
}

#[derive(Deserialize)]
struct ValidationPlan {
    validation_seed_block: SeedBlock,
    frontier_pairs: Vec<FrontierPair>,
}

struct Cell {
    idx: usize,
    fee: f64,
    sigma: f64,
    z: f64,
    lam_arb: f64,
    lam_fund: f64,
}

#[derive(Hash, Eq, PartialEq, Clone, Copy)]
struct PolicyKey {
    f0_bits: u64,
    alpha_bits: u64,
    fee_cap_bits: u64,
    dial_bits: u64,
}

impl PolicyKey {
    fn from_policy(p: &PlanPolicy) -> Self {
        Self {
            f0_bits: p.f0.to_bits(),
            alpha_bits: p.alpha.to_bits(),
            fee_cap_bits: p.fee_cap.to_bits(),
            dial_bits: p.dial_mult.to_bits(),
        }
    }
}

struct RunCache {
    events: Vec<EventRecord>,
    records: Vec<StepRecord>,
}

#[derive(Serialize)]
struct GridRow {
    comparison: String,
    record_id: String,
    cell_idx: usize,
    stratum: String,
    sigma: f64,
    z: f64,
    speed: String,
    rho: f64,
    support_label: String,
    observed_pool_weeks: u64,
    target_s_training: f64,
    seed: u64,
    pi1_family: String,
    pi1_f0: f64,
    pi1_alpha: f64,
    pi1_dial_mult: f64,
    pi0_family: String,
    pi0_f0: f64,
    pi0_alpha: f64,
    pi0_dial_mult: f64,
    #[serde(rename = "total.delta_a")]
    total_delta_a: f64,
    #[serde(rename = "total.delta_qty_c")]
    total_delta_qty_c: f64,
    #[serde(rename = "total.delta_sev_c")]
    total_delta_sev_c: f64,
    #[serde(rename = "total.delta_entry")]
    total_delta_entry: f64,
    #[serde(rename = "total.delta_exit")]
    total_delta_exit: f64,
    #[serde(rename = "total.delta_common")]
    total_delta_common: f64,
    #[serde(rename = "total.delta_selection")]
    total_delta_selection: f64,
    #[serde(rename = "fund.delta_a")]
    fund_delta_a: f64,
    #[serde(rename = "fund.delta_qty_c")]
    fund_delta_qty_c: f64,
    #[serde(rename = "fund.delta_sev_c")]
    fund_delta_sev_c: f64,
    #[serde(rename = "fund.delta_entry")]
    fund_delta_entry: f64,
    #[serde(rename = "fund.delta_exit")]
    fund_delta_exit: f64,
    #[serde(rename = "fund.delta_common")]
    fund_delta_common: f64,
    #[serde(rename = "fund.delta_selection")]
    fund_delta_selection: f64,
    #[serde(rename = "arb.delta_a")]
    arb_delta_a: f64,
    #[serde(rename = "arb.delta_qty_c")]
    arb_delta_qty_c: f64,
    #[serde(rename = "arb.delta_sev_c")]
    arb_delta_sev_c: f64,
    #[serde(rename = "arb.delta_entry")]
    arb_delta_entry: f64,
    #[serde(rename = "arb.delta_exit")]
    arb_delta_exit: f64,
    #[serde(rename = "arb.delta_common")]
    arb_delta_common: f64,
    #[serde(rename = "arb.delta_selection")]
    arb_delta_selection: f64,
    delta_s: f64,
    delta_fees: f64,
    delta_u: f64,
    reconstruct_ok: bool,
    max_reconstruct_err: f64,
}

#[derive(Clone)]
struct ComparisonSpec {
    comparison: String,
    record_id: String,
    cell: PlanCell,
    rho: f64,
    target_s_training: f64,
    support: EmpiricalSupport,
    pi1: PlanPolicy,
    pi0: PlanPolicy,
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

fn load_validation_plan() -> ValidationPlan {
    let root = Path::new(ROOT);
    let plan_path = root.join(VALIDATION_PLAN_REL);
    let recorded = fs::read_to_string(root.join(VALIDATION_PLAN_HASH_REL))
        .expect("missing validation plan hash")
        .split_whitespace()
        .next()
        .unwrap()
        .to_owned();
    assert_eq!(sha256(&plan_path), recorded, "validation plan changed");
    serde_json::from_reader(BufReader::new(File::open(plan_path).unwrap())).unwrap()
}

fn load_cells() -> Vec<Cell> {
    let manifest: serde_json::Value = serde_json::from_reader(
        File::open(Path::new(ROOT).join(MANIFEST_REL)).expect("missing manifest"),
    )
    .unwrap();
    manifest["cells"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(idx, c)| Cell {
            idx,
            fee: c["fee"].as_f64().unwrap(),
            sigma: c["sigma"].as_f64().unwrap(),
            z: c["z"].as_f64().unwrap(),
            lam_arb: c["lambda_arb_star_SOLVED"].as_f64().unwrap(),
            lam_fund: c["lambda_fund_star_SOLVED"].as_f64().unwrap(),
        })
        .collect()
}

fn config_for(cell: &Cell) -> SimConfig {
    let y0 = 1.0e4;
    let d_ref = y0 * (1.0 - (1.0f64 + 0.01).powf(-0.5));
    SimConfig {
        name: "m3griddecomp".into(),
        description: "M3 validation grid decomposition".into(),
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
    if spec.alpha == 0.0 {
        Box::new(FixedFeePolicy::new(spec.f0))
    } else {
        Box::new(OracleGapFeePolicy {
            base_fee: spec.f0,
            gap_multiplier: spec.alpha,
            min_fee: spec.f0,
            max_fee: spec.fee_cap,
        })
    }
}

fn ensure_cached(
    cache: &mut HashMap<PolicyKey, RunCache>,
    cfg: &SimConfig,
    prices: &[f64],
    policy: &PlanPolicy,
) {
    let key = PolicyKey::from_policy(policy);
    if cache.contains_key(&key) {
        return;
    }
    let mut pol = make_policy(policy);
    let (records, events) = run_simulation_with_events(cfg, prices, pol.as_mut());
    cache.insert(key, RunCache { events, records });
}

fn leg_err(decomp: &SeedDecomp) -> f64 {
    decomp
        .total
        .reconstruct_err()
        .max(decomp.fund.reconstruct_err())
        .max(decomp.arb.reconstruct_err())
}

fn emit_row(
    spec: &ComparisonSpec,
    seed: u64,
    decomp: SeedDecomp,
    agg: AggregateDeltas,
    err: f64,
    tol: f64,
) -> GridRow {
    GridRow {
        comparison: spec.comparison.clone(),
        record_id: spec.record_id.clone(),
        cell_idx: spec.cell.cell_idx,
        stratum: spec.cell.stratum.clone(),
        sigma: spec.cell.sigma,
        z: spec.cell.z,
        speed: spec.cell.speed.clone(),
        rho: spec.rho,
        support_label: spec.support.support_label.clone(),
        observed_pool_weeks: spec.support.observed_pool_weeks,
        target_s_training: spec.target_s_training,
        seed,
        pi1_family: spec.pi1.family.clone(),
        pi1_f0: spec.pi1.f0,
        pi1_alpha: spec.pi1.alpha,
        pi1_dial_mult: spec.pi1.dial_mult,
        pi0_family: spec.pi0.family.clone(),
        pi0_f0: spec.pi0.f0,
        pi0_alpha: spec.pi0.alpha,
        pi0_dial_mult: spec.pi0.dial_mult,
        total_delta_a: decomp.total.delta_a,
        total_delta_qty_c: decomp.total.delta_qty_c,
        total_delta_sev_c: decomp.total.delta_sev_c,
        total_delta_entry: decomp.total.delta_entry,
        total_delta_exit: decomp.total.delta_exit,
        total_delta_common: decomp.total.delta_common(),
        total_delta_selection: decomp.total.delta_selection(),
        fund_delta_a: decomp.fund.delta_a,
        fund_delta_qty_c: decomp.fund.delta_qty_c,
        fund_delta_sev_c: decomp.fund.delta_sev_c,
        fund_delta_entry: decomp.fund.delta_entry,
        fund_delta_exit: decomp.fund.delta_exit,
        fund_delta_common: decomp.fund.delta_common(),
        fund_delta_selection: decomp.fund.delta_selection(),
        arb_delta_a: decomp.arb.delta_a,
        arb_delta_qty_c: decomp.arb.delta_qty_c,
        arb_delta_sev_c: decomp.arb.delta_sev_c,
        arb_delta_entry: decomp.arb.delta_entry,
        arb_delta_exit: decomp.arb.delta_exit,
        arb_delta_common: decomp.arb.delta_common(),
        arb_delta_selection: decomp.arb.delta_selection(),
        delta_s: agg.delta_s,
        delta_fees: agg.delta_fees,
        delta_u: agg.delta_u,
        reconstruct_ok: err <= tol,
        max_reconstruct_err: err,
    }
}

fn primary_specs(plan: &ValidationPlan) -> Vec<ComparisonSpec> {
    plan.frontier_pairs
        .iter()
        .map(|f| ComparisonSpec {
            comparison: "gap_frontier_vs_static_frontier".into(),
            record_id: f.frontier_id.clone(),
            cell: f.cell.clone(),
            rho: f.rho,
            target_s_training: f.target_s_training,
            support: f.empirical_support.clone(),
            pi1: f.gap_policy.clone(),
            pi0: f.static_policy.clone(),
        })
        .collect()
}

fn parse_policy(row: &csv::StringRecord, headers: &csv::StringRecord, prefix: &str) -> PlanPolicy {
    let get = |name: &str| -> String {
        let idx = headers.iter().position(|h| h == name).unwrap();
        row[idx].to_string()
    };
    let alpha: f64 = get(&format!("{prefix}_alpha")).parse().unwrap();
    PlanPolicy {
        family: if alpha == 0.0 {
            "static".into()
        } else {
            "gap".into()
        },
        dial_mult: get(&format!("{prefix}_dial_mult")).parse().unwrap(),
        f0: get(&format!("{prefix}_f0")).parse().unwrap(),
        alpha,
        fee_cap: 0.30,
    }
}

fn secondary_specs() -> Vec<ComparisonSpec> {
    let path = Path::new(ROOT).join(SELECTOR_CSV_REL);
    let mut rdr = csv::Reader::from_path(path).expect("selector csv");
    let headers = rdr.headers().unwrap().clone();
    let mut specs = Vec::new();
    for row in rdr.records() {
        let row = row.unwrap();
        let idx = headers
            .iter()
            .position(|h| h == "selection_diverges")
            .unwrap();
        if &row[idx] != "True" {
            continue;
        }
        let cell_idx: usize = row[headers.iter().position(|h| h == "cell_idx").unwrap()]
            .parse()
            .unwrap();
        let get = |name: &str| -> String {
            let i = headers.iter().position(|h| h == name).unwrap();
            row[i].to_string()
        };
        specs.push(ComparisonSpec {
            comparison: "pi_a_star_vs_pi_u_star".into(),
            record_id: format!("cell{cell_idx:03}_rho{}", get("rho")),
            cell: PlanCell {
                cell_idx,
                stratum: get("stratum"),
                sigma: get("sigma").parse().unwrap(),
                z: get("z").parse().unwrap(),
                speed: get("speed"),
            },
            rho: get("rho").parse().unwrap(),
            target_s_training: get("service_target").parse().unwrap(),
            support: EmpiricalSupport {
                support_label: get("support_label"),
                observed_pool_weeks: get("observed_pool_weeks").parse().unwrap(),
                stratum: get("stratum"),
            },
            pi1: parse_policy(&row, &headers, "pi_a_star"),
            pi0: parse_policy(&row, &headers, "pi_u_star"),
        });
    }
    specs
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args
        .iter()
        .position(|a| a == "--mode")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
        .unwrap_or("primary");
    let shard: usize = args
        .iter()
        .position(|a| a == "--shard")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let of: usize = args
        .iter()
        .position(|a| a == "--of")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    assert!(shard < of, "require shard < of");
    let force = args.iter().any(|a| a == "--force");

    let plan = load_validation_plan();
    let cells = load_cells();
    let specs = match mode {
        "primary" => primary_specs(&plan),
        "secondary" => secondary_specs(),
        other => panic!("unknown mode {other}"),
    };

    let mut by_cell: BTreeMap<usize, Vec<ComparisonSpec>> = BTreeMap::new();
    for spec in specs {
        by_cell.entry(spec.cell.cell_idx).or_default().push(spec);
    }

    let out_path = PathBuf::from(format!(
        "{ROOT}/.local/lvr/m3_validation_grid_decomposition_{mode}_shard{shard}.csv.gz"
    ));
    if out_path.exists() && !force {
        panic!(
            "refusing to overwrite {} (pass --force)",
            out_path.display()
        );
    }

    let encoder = GzEncoder::new(File::create(&out_path).unwrap(), Compression::default());
    let mut writer = csv::Writer::from_writer(encoder);
    let start = std::time::Instant::now();
    let mut n_rows = 0usize;
    let mut n_fail = 0usize;

    for cell in cells.iter().filter(|c| c.idx % of == shard) {
        let cell_specs = match by_cell.get(&cell.idx) {
            Some(v) if !v.is_empty() => v,
            _ => continue,
        };
        let cfg0 = config_for(cell);
        for seed in
            plan.validation_seed_block.start_inclusive..plan.validation_seed_block.end_exclusive
        {
            let mut cfg = cfg0.clone();
            cfg.seed = seed;
            let prices = generate_gbm(cfg.n_steps, S0_PRICE, 0.0, cfg.sigma, cfg.dt_years(), seed);
            let mut cache: HashMap<PolicyKey, RunCache> = HashMap::new();

            for spec in cell_specs {
                ensure_cached(&mut cache, &cfg, &prices, &spec.pi1);
                ensure_cached(&mut cache, &cfg, &prices, &spec.pi0);
                let key1 = PolicyKey::from_policy(&spec.pi1);
                let key0 = PolicyKey::from_policy(&spec.pi0);
                let run1 = cache.get(&key1).unwrap();
                let run0 = cache.get(&key0).unwrap();
                let decomp = decompose_pair(&run1.events, &run0.events);
                let es1 = summarize_run(&run1.events, &run1.records);
                let es0 = summarize_run(&run0.events, &run0.records);
                let agg = AggregateDeltas {
                    delta_a: es1.a_fill - es0.a_fill,
                    delta_s: es1.served_fund_volume - es0.served_fund_volume,
                    delta_fees: es1.fees_total - es0.fees_total,
                    delta_u: es1.u_lp_rel - es0.u_lp_rel,
                };
                let err = leg_err(&decomp);
                let tol = 1e-8_f64.max(1e-8 * decomp.total.delta_a.abs().max(1.0));
                if err > tol {
                    n_fail += 1;
                }
                writer
                    .serialize(emit_row(spec, seed, decomp, agg, err, tol))
                    .unwrap();
                n_rows += 1;
            }
        }
        eprintln!(
            "[grid {mode} {shard}/{of}] cell {} done, {} rows, {:.0}s",
            cell.idx,
            n_rows,
            start.elapsed().as_secs_f64()
        );
    }
    writer.flush().unwrap();
    writer.into_inner().unwrap().finish().unwrap();
    assert_eq!(n_fail, 0, "{n_fail} rows failed reconstruction");
    eprintln!("wrote {} ({} rows)", out_path.display(), n_rows);
}
