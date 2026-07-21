//! Paired-ledger decomposition for the frozen validation-grid final held-out block.
//!
//! Exports per-seed ΔA = Δ_qty,C + Δ_sev,C + Δ_entry + Δ_exit on shared
//! primitive opportunities, with fund / arb / total splits.

use amm_lab::campbell::fee_policy::{FeePolicy, FixedFeePolicy, OracleGapFeePolicy};
use amm_lab::campbell::gbm::generate_gbm;
use amm_lab::campbell::simulation::{
    ArrivalModel, EventKind, EventRecord, FlowRegime, SimConfig, run_simulation_with_events,
};
use flate2::Compression;
use flate2::write::GzEncoder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

const ROOT: &str = "/Users/joseph/amm-lab";
const PLAN_REL: &str = ".local/lvr/m3_amended_final_plan.json";
const PLAN_HASH_REL: &str = ".local/lvr/m3_amended_final_plan.sha256";
const MANIFEST_REL: &str = ".local/lvr/calibration_54_manifest.json";
const S0_PRICE: f64 = 2000.0;
const OUT_REL: &str = ".local/lvr/m3_amended_final_decomposition_seeds.csv.gz";
const AUDIT_REL: &str = ".local/lvr/m3_amended_final_decomposition_audit.json";

#[derive(Deserialize)]
#[allow(dead_code)]
struct SeedBlock {
    start_inclusive: u64,
    end_exclusive: u64,
    n: usize,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct PlanCell {
    cell_idx: usize,
    stratum: String,
    sigma: f64,
    z: f64,
    speed: String,
}

#[derive(Clone, Deserialize)]
#[allow(dead_code)]
struct PlanPolicy {
    family: String,
    dial_mult: f64,
    f0: f64,
    alpha: f64,
    fee_cap: f64,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct Candidate {
    candidate_id: String,
    cell: PlanCell,
    rho: f64,
    #[serde(rename = "policy_1_lower_A")]
    policy_1_lower_a: PlanPolicy,
    policy_2: PlanPolicy,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct FinalPlan {
    source_stage: String,
    final_seed_block: SeedBlock,
    candidate: Candidate,
    input_artifacts_sha256: BTreeMap<String, String>,
}

struct Cell {
    fee: f64,
    sigma: f64,
    z: f64,
    lam_arb: f64,
    lam_fund: f64,
}

#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
struct OppKey {
    step: usize,
    time_frac_bits: u64,
    kind_tag: u8,
}

fn kind_tag(kind: EventKind) -> u8 {
    match kind {
        EventKind::Arb => 0,
        EventKind::FundBuy => 1,
        EventKind::FundSell => 2,
    }
}

fn tag_kind(tag: u8) -> EventKind {
    match tag {
        0 => EventKind::Arb,
        1 => EventKind::FundBuy,
        2 => EventKind::FundSell,
        _ => unreachable!("invalid kind tag"),
    }
}

impl OppKey {
    fn from_event(e: &EventRecord) -> Self {
        Self {
            step: e.step,
            time_frac_bits: e.time_frac.to_bits(),
            kind_tag: kind_tag(e.kind),
        }
    }
}

#[derive(Clone, Copy, Default, Debug)]
#[allow(dead_code)] // fees/pot/cex/unserved retained for ledger completeness
struct OppFields {
    e: u8,
    q: f64,
    a: f64,
    c: f64,
    fees: f64,
    pot: f64,
    cex: f64,
    unserved: f64,
}

impl OppFields {
    fn from_event(e: &EventRecord) -> Self {
        let q = e.delta.abs();
        let inc = q > 0.0;
        let a = e.ell.max(0.0);
        let c = if inc { a / q } else { 0.0 };
        // Fee cash flow proxy: served notional × fee rate (y-denominated leg).
        let fees = q * e.fee_used;
        Self {
            e: u8::from(inc),
            q,
            a,
            c,
            fees,
            pot: e.pot,
            cex: e.cex,
            unserved: e.unserved,
        }
    }

    fn zero() -> Self {
        Self::default()
    }
}

#[derive(Clone, Copy, Default, Debug, Serialize)]
struct LegDecomp {
    delta_a: f64,
    delta_qty_c: f64,
    delta_sev_c: f64,
    delta_entry: f64,
    delta_exit: f64,
    n_opportunities: u64,
    n_common: u64,
    n_entry: u64,
    n_exit: u64,
    n_both_zero: u64,
}

#[derive(Serialize)]
struct SeedRow {
    seed: u64,
    delta_a_total: f64,
    delta_qty_c_total: f64,
    delta_sev_c_total: f64,
    delta_entry_total: f64,
    delta_exit_total: f64,
    delta_a_fund: f64,
    delta_qty_c_fund: f64,
    delta_sev_c_fund: f64,
    delta_entry_fund: f64,
    delta_exit_fund: f64,
    delta_a_arb: f64,
    delta_qty_c_arb: f64,
    delta_sev_c_arb: f64,
    delta_entry_arb: f64,
    delta_exit_arb: f64,
    n_fund_opportunities: u64,
    n_arb_opportunities: u64,
    n_common_fund: u64,
    n_entry_fund: u64,
    n_exit_fund: u64,
    n_common_arb: u64,
    n_entry_arb: u64,
    n_exit_arb: u64,
    reconstruct_ok: bool,
    max_reconstruct_err: f64,
}

#[derive(Serialize)]
struct LedgerAudit {
    paired_on: String,
    opportunity_key: String,
    fund_events_always_paired: bool,
    arb_events_pair_on_step: bool,
    fields_per_opportunity: Vec<&'static str>,
    note: String,
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

fn load_and_verify_plan() -> FinalPlan {
    let root = Path::new(ROOT);
    let plan_path = root.join(PLAN_REL);
    let recorded_hash = fs::read_to_string(root.join(PLAN_HASH_REL))
        .expect("missing frozen final-plan hash")
        .split_whitespace()
        .next()
        .expect("empty final-plan hash")
        .to_owned();
    assert_eq!(sha256(&plan_path), recorded_hash, "final plan changed");
    let plan: FinalPlan =
        serde_json::from_reader(BufReader::new(File::open(&plan_path).unwrap())).unwrap();
    assert_eq!(plan.final_seed_block.start_inclusive, 91_000);
    assert_eq!(plan.final_seed_block.end_exclusive, 91_400);
    plan
}

fn load_cell(idx: usize) -> Cell {
    let manifest: serde_json::Value = serde_json::from_reader(
        File::open(Path::new(ROOT).join(MANIFEST_REL)).expect("missing calibration manifest"),
    )
    .unwrap();
    let c = &manifest["cells"].as_array().unwrap()[idx];
    Cell {
        fee: c["fee"].as_f64().unwrap(),
        sigma: c["sigma"].as_f64().unwrap(),
        z: c["z"].as_f64().unwrap(),
        lam_arb: c["lambda_arb_star_SOLVED"].as_f64().unwrap(),
        lam_fund: c["lambda_fund_star_SOLVED"].as_f64().unwrap(),
    }
}

fn config_for(cell: &Cell) -> SimConfig {
    let y0 = 1.0e4;
    let d_ref = y0 * (1.0 - (1.0f64 + 0.01).powf(-0.5));
    SimConfig {
        name: "m3decomp".into(),
        description: "validation-grid paired decomposition".into(),
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
        log_inactive_arb: true,
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

fn index_events(events: &[EventRecord]) -> HashMap<OppKey, OppFields> {
    let mut map = HashMap::new();
    for e in events {
        map.insert(OppKey::from_event(e), OppFields::from_event(e));
    }
    map
}

fn is_fund(kind: EventKind) -> bool {
    matches!(kind, EventKind::FundBuy | EventKind::FundSell)
}

fn decompose_leg(
    p1: &HashMap<OppKey, OppFields>,
    p0: &HashMap<OppKey, OppFields>,
    filter: impl Fn(EventKind) -> bool,
) -> LegDecomp {
    let keys: HashSet<OppKey> = p1
        .keys()
        .chain(p0.keys())
        .copied()
        .filter(|k| filter(tag_kind(k.kind_tag)))
        .collect();
    let mut out = LegDecomp {
        n_opportunities: keys.len() as u64,
        ..Default::default()
    };
    for key in keys {
        let f1 = p1.get(&key).copied().unwrap_or_else(OppFields::zero);
        let f0 = p0.get(&key).copied().unwrap_or_else(OppFields::zero);
        let da = f1.a - f0.a;
        out.delta_a += da;
        match (f1.e, f0.e) {
            (1, 1) => {
                out.n_common += 1;
                out.delta_qty_c += (f1.q - f0.q) * f0.c;
                out.delta_sev_c += f1.q * (f1.c - f0.c);
            }
            (1, 0) => {
                out.n_entry += 1;
                out.delta_entry += f1.a;
            }
            (0, 1) => {
                out.n_exit += 1;
                out.delta_exit -= f0.a;
            }
            (0, 0) => out.n_both_zero += 1,
            _ => unreachable!("incidence is binary"),
        }
    }
    out
}

fn decompose_pair(
    events1: &[EventRecord],
    events0: &[EventRecord],
) -> (LegDecomp, LegDecomp, LegDecomp) {
    let p1 = index_events(events1);
    let p0 = index_events(events0);
    let total = decompose_leg(&p1, &p0, |_| true);
    let fund = decompose_leg(&p1, &p0, is_fund);
    let arb = decompose_leg(&p1, &p0, |k| k == EventKind::Arb);
    (total, fund, arb)
}

fn reconstruct_err(d: &LegDecomp) -> f64 {
    (d.delta_a - (d.delta_qty_c + d.delta_sev_c + d.delta_entry + d.delta_exit)).abs()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let verify_only = args.iter().any(|a| a == "--verify-only");
    let max_seeds: Option<u64> = args
        .iter()
        .position(|a| a == "--max-seeds")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok());

    let plan = load_and_verify_plan();
    if verify_only {
        eprintln!(
            "verified plan for paired decomposition: {}",
            plan.candidate.candidate_id
        );
        return;
    }

    let out_path = PathBuf::from(format!("{ROOT}/{OUT_REL}"));
    if out_path.exists() && !args.iter().any(|a| a == "--force") {
        panic!(
            "refusing to overwrite {} (pass --force)",
            out_path.display()
        );
    }

    let cell = load_cell(plan.candidate.cell.cell_idx);
    let cfg0 = config_for(&cell);
    let policy1_spec = &plan.candidate.policy_1_lower_a;
    let policy0_spec = &plan.candidate.policy_2;

    let seed_end = plan.final_seed_block.end_exclusive;
    let seed_start = plan.final_seed_block.start_inclusive;
    let seed_limit = max_seeds.map(|n| seed_start + n).unwrap_or(seed_end);

    let mut seed_rows = Vec::new();
    let start = std::time::Instant::now();

    for seed in seed_start..seed_limit.min(seed_end) {
        let mut cfg = cfg0.clone();
        cfg.seed = seed;
        let prices = generate_gbm(cfg.n_steps, S0_PRICE, 0.0, cfg.sigma, cfg.dt_years(), seed);

        let mut pol1 = make_policy(policy1_spec);
        let (_, ev1) = run_simulation_with_events(&cfg, &prices, pol1.as_mut());
        let mut pol0 = make_policy(policy0_spec);
        let (_, ev0) = run_simulation_with_events(&cfg, &prices, pol0.as_mut());

        let (total, fund, arb) = decompose_pair(&ev1, &ev0);
        let err = reconstruct_err(&total)
            .max(reconstruct_err(&fund))
            .max(reconstruct_err(&arb));
        let tol = 1e-6 * total.delta_a.abs().max(1.0);
        let ok = err <= tol;

        seed_rows.push(SeedRow {
            seed,
            delta_a_total: total.delta_a,
            delta_qty_c_total: total.delta_qty_c,
            delta_sev_c_total: total.delta_sev_c,
            delta_entry_total: total.delta_entry,
            delta_exit_total: total.delta_exit,
            delta_a_fund: fund.delta_a,
            delta_qty_c_fund: fund.delta_qty_c,
            delta_sev_c_fund: fund.delta_sev_c,
            delta_entry_fund: fund.delta_entry,
            delta_exit_fund: fund.delta_exit,
            delta_a_arb: arb.delta_a,
            delta_qty_c_arb: arb.delta_qty_c,
            delta_sev_c_arb: arb.delta_sev_c,
            delta_entry_arb: arb.delta_entry,
            delta_exit_arb: arb.delta_exit,
            n_fund_opportunities: fund.n_opportunities,
            n_arb_opportunities: arb.n_opportunities,
            n_common_fund: fund.n_common,
            n_entry_fund: fund.n_entry,
            n_exit_fund: fund.n_exit,
            n_common_arb: arb.n_common,
            n_entry_arb: arb.n_entry,
            n_exit_arb: arb.n_exit,
            reconstruct_ok: ok,
            max_reconstruct_err: err,
        });

        if (seed + 1 - seed_start).is_multiple_of(25) {
            eprintln!(
                "decomp seeds {}/{} done, {:.0}s elapsed",
                seed + 1 - seed_start,
                seed_limit - seed_start,
                start.elapsed().as_secs_f64()
            );
        }
    }

    let failed: Vec<_> = seed_rows.iter().filter(|r| !r.reconstruct_ok).collect();
    assert!(
        failed.is_empty(),
        "{} seeds failed exact reconstruction (first seed {}, err {})",
        failed.len(),
        failed.first().map(|r| r.seed).unwrap_or(0),
        failed.first().map(|r| r.max_reconstruct_err).unwrap_or(0.0)
    );

    let encoder = GzEncoder::new(
        File::create(&out_path).expect("create output"),
        Compression::default(),
    );
    let mut w = csv::Writer::from_writer(encoder);
    for row in &seed_rows {
        w.serialize(row).unwrap();
    }
    w.flush().unwrap();
    w.into_inner().unwrap().finish().unwrap();

    let audit = LedgerAudit {
        paired_on: "(seed, step, time_frac, event_kind)".into(),
        opportunity_key: "step + time_frac + FundBuy|FundSell|Arb".into(),
        fund_events_always_paired: true,
        arb_events_pair_on_step: true,
        fields_per_opportunity: vec!["e", "q", "a", "c", "fees", "pot", "cex", "unserved"],
        note: "Poisson-mode fund primitives are policy-invariant; arb rows appear only on fill."
            .into(),
    };
    let audit_path = PathBuf::from(format!("{ROOT}/{AUDIT_REL}"));
    fs::write(
        &audit_path,
        serde_json::to_string_pretty(&audit).unwrap() + "\n",
    )
    .unwrap();

    eprintln!("wrote {} ({} seeds)", out_path.display(), seed_rows.len());
    eprintln!("wrote {}", audit_path.display());
}
