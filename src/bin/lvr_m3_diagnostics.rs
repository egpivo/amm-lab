//! Training-only diagnostics for policies selected from the amended M3 grid.

use amm_lab::campbell::fee_policy::{FeeObservation, FeePolicy};
use amm_lab::campbell::gbm::generate_gbm;
use amm_lab::campbell::simulation::{
    ArrivalModel, EventKind, EventRecord, FlowRegime, SimConfig, StepRecord,
    run_simulation_with_events,
};
use amm_lab::campbell::summary::{summarize, summarize_events};
use flate2::Compression;
use flate2::write::GzEncoder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

const ROOT: &str = "/Users/joseph/amm-lab";
const PLAN_REL: &str = ".local/lvr/m3_amended_diagnostic_plan.json";
const PLAN_HASH_REL: &str = ".local/lvr/m3_amended_diagnostic_plan.sha256";
const SELECTION_REL: &str = ".local/lvr/m3_amended_training_selection.json";
const MANIFEST_REL: &str = ".local/lvr/calibration_54_manifest.json";
const S0_PRICE: f64 = 2000.0;
const N_FEE_BINS: usize = 1024;
const MIN_FEE_BPS: f64 = 0.01;
const MAX_FEE_BPS: f64 = 3000.0;
const GAP_EDGES_BPS: [f64; 8] = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 250.0];
const HORIZONS_HOURS: [f64; 4] = [0.0, 1.0, 5.0, 20.0];

#[derive(Deserialize)]
struct SeedBlock {
    start_inclusive: u64,
    end_exclusive: u64,
    n: usize,
}

#[derive(Clone, Deserialize, Serialize)]
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

#[derive(Clone, Deserialize, Serialize)]
struct DiagnosticPolicy {
    cell: PlanCell,
    policy: PlanPolicy,
    assignments: serde_json::Value,
}

#[derive(Deserialize)]
struct DiagnosticPlan {
    selection_sha256: String,
    seed_block: SeedBlock,
    n_unique_policies: usize,
    policies: Vec<DiagnosticPolicy>,
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

#[derive(Clone, Default, Serialize)]
struct GapBin {
    n: u64,
    sum_fee: f64,
    sum_abs_gap: f64,
}

#[derive(Clone, Serialize)]
struct DecisionAggregate {
    n: u64,
    sum_fee: f64,
    min_fee: f64,
    max_fee: f64,
    at_lower_clip: u64,
    at_upper_clip: u64,
    fee_histogram: Vec<u64>,
    stale_gap_bins: Vec<GapBin>,
    contemporaneous_gap_bins: Vec<GapBin>,
}

impl Default for DecisionAggregate {
    fn default() -> Self {
        Self {
            n: 0,
            sum_fee: 0.0,
            min_fee: f64::INFINITY,
            max_fee: f64::NEG_INFINITY,
            at_lower_clip: 0,
            at_upper_clip: 0,
            fee_histogram: vec![0; N_FEE_BINS],
            stale_gap_bins: vec![GapBin::default(); GAP_EDGES_BPS.len() + 1],
            contemporaneous_gap_bins: vec![GapBin::default(); GAP_EDGES_BPS.len() + 1],
        }
    }
}

impl DecisionAggregate {
    fn observe(&mut self, fee: f64, lower: f64, upper: f64, stale_gap: f64, true_gap: f64) {
        self.n += 1;
        self.sum_fee += fee;
        self.min_fee = self.min_fee.min(fee);
        self.max_fee = self.max_fee.max(fee);
        let tolerance = 1e-12;
        if (fee - lower).abs() <= tolerance {
            self.at_lower_clip += 1;
        }
        if (fee - upper).abs() <= tolerance {
            self.at_upper_clip += 1;
        }
        self.fee_histogram[fee_bin(fee)] += 1;
        observe_gap(&mut self.stale_gap_bins, stale_gap, fee);
        observe_gap(&mut self.contemporaneous_gap_bins, true_gap, fee);
    }

    fn merge(&mut self, other: Self) {
        self.n += other.n;
        self.sum_fee += other.sum_fee;
        self.min_fee = self.min_fee.min(other.min_fee);
        self.max_fee = self.max_fee.max(other.max_fee);
        self.at_lower_clip += other.at_lower_clip;
        self.at_upper_clip += other.at_upper_clip;
        for (left, right) in self.fee_histogram.iter_mut().zip(other.fee_histogram) {
            *left += right;
        }
        for (left, right) in self.stale_gap_bins.iter_mut().zip(other.stale_gap_bins) {
            left.n += right.n;
            left.sum_fee += right.sum_fee;
            left.sum_abs_gap += right.sum_abs_gap;
        }
        for (left, right) in self
            .contemporaneous_gap_bins
            .iter_mut()
            .zip(other.contemporaneous_gap_bins)
        {
            left.n += right.n;
            left.sum_fee += right.sum_fee;
            left.sum_abs_gap += right.sum_abs_gap;
        }
    }
}

struct AuditedPolicy {
    spec: PlanPolicy,
    decisions: DecisionAggregate,
}

impl AuditedPolicy {
    fn new(spec: PlanPolicy) -> Self {
        Self {
            spec,
            decisions: DecisionAggregate::default(),
        }
    }
}

impl FeePolicy for AuditedPolicy {
    fn name(&self) -> &'static str {
        "m3_diagnostic"
    }

    fn fee(&mut self, obs: &FeeObservation) -> f64 {
        let fee = if self.spec.family == "static" {
            self.spec.f0
        } else {
            (self.spec.f0 + self.spec.alpha * obs.oracle_gap_bps.abs() / 10_000.0)
                .clamp(self.spec.f0, self.spec.fee_cap)
        };
        let upper = if self.spec.family == "static" {
            self.spec.f0
        } else {
            self.spec.fee_cap
        };
        self.decisions.observe(
            fee,
            self.spec.f0,
            upper,
            obs.oracle_gap_bps,
            obs.contemporaneous_gap_bps,
        );
        fee
    }
}

#[derive(Clone, Default, Serialize)]
struct MomentSums {
    n: u64,
    sum_x: f64,
    sum_y: f64,
    sum_x2: f64,
    sum_y2: f64,
    sum_xy: f64,
}

impl MomentSums {
    fn observe(&mut self, x: f64, y: f64) {
        self.n += 1;
        self.sum_x += x;
        self.sum_y += y;
        self.sum_x2 += x * x;
        self.sum_y2 += y * y;
        self.sum_xy += x * y;
    }

    fn merge(&mut self, other: Self) {
        self.n += other.n;
        self.sum_x += other.sum_x;
        self.sum_y += other.sum_y;
        self.sum_x2 += other.sum_x2;
        self.sum_y2 += other.sum_y2;
        self.sum_xy += other.sum_xy;
    }
}

#[derive(Clone, Default, Serialize)]
struct RiskBin {
    fill_count: u64,
    volume: f64,
    ell_positive: f64,
    sum_ell_positive_per_unit: f64,
    markout_count: [u64; 4],
    markout_volume: [f64; 4],
    markout_sum: [f64; 4],
    sum_markout_per_unit: [f64; 4],
}

impl RiskBin {
    fn merge(&mut self, other: Self) {
        self.fill_count += other.fill_count;
        self.volume += other.volume;
        self.ell_positive += other.ell_positive;
        self.sum_ell_positive_per_unit += other.sum_ell_positive_per_unit;
        for h in 0..4 {
            self.markout_count[h] += other.markout_count[h];
            self.markout_volume[h] += other.markout_volume[h];
            self.markout_sum[h] += other.markout_sum[h];
            self.sum_markout_per_unit[h] += other.sum_markout_per_unit[h];
        }
    }
}

#[derive(Clone, Default, Serialize)]
struct ChannelSums {
    episodes: u64,
    potential: f64,
    served: f64,
    alloc_amm: f64,
    alloc_cex: f64,
    alloc_unserved: f64,
    fill_incidence: f64,
    conditional_fill_size: f64,
    a: f64,
    a_arb: f64,
    a_fund: f64,
    b_fund: f64,
    fees: f64,
    fees_arb: f64,
    fees_fund: f64,
    u: f64,
    quote_error: f64,
    n_fund_events: u64,
    fund_fill_count: u64,
}

#[derive(Default, Serialize)]
struct PolicyAggregate {
    decisions: DecisionAggregate,
    channels: ChannelSums,
    risk_bins: BTreeMap<String, RiskBin>,
    fee_severity_correlation_sums: MomentSums,
    fee_markout_correlation_sums: [MomentSums; 4],
}

#[derive(Serialize)]
struct PolicyOutput {
    cell: PlanCell,
    policy: PlanPolicy,
    assignments: serde_json::Value,
    aggregate: PolicyAggregate,
}

#[derive(Serialize)]
struct ShardOutput {
    step: &'static str,
    shard: usize,
    of: usize,
    seed_start_inclusive: u64,
    seed_end_exclusive: u64,
    fee_histogram: serde_json::Value,
    gap_bin_upper_edges_bps: [f64; 8],
    markout_horizons_hours: [f64; 4],
    policies: Vec<PolicyOutput>,
}

fn fee_bin(fee: f64) -> usize {
    let bps = (fee * 10_000.0).clamp(MIN_FEE_BPS, MAX_FEE_BPS);
    let fraction = (bps / MIN_FEE_BPS).ln() / (MAX_FEE_BPS / MIN_FEE_BPS).ln();
    (fraction * (N_FEE_BINS - 1) as f64).floor() as usize
}

fn gap_bin(gap: f64) -> usize {
    GAP_EDGES_BPS
        .iter()
        .position(|edge| gap.abs() < *edge)
        .unwrap_or(GAP_EDGES_BPS.len())
}

fn observe_gap(bins: &mut [GapBin], gap: f64, fee: f64) {
    let bin = &mut bins[gap_bin(gap)];
    bin.n += 1;
    bin.sum_fee += fee;
    bin.sum_abs_gap += gap.abs();
}

fn event_kind_name(kind: EventKind) -> &'static str {
    match kind {
        EventKind::Arb => "arb",
        EventKind::FundBuy => "fund_buy",
        EventKind::FundSell => "fund_sell",
    }
}

fn risk_key(kind: &str, bin: usize) -> String {
    format!("{kind}:{bin:04}")
}

fn accumulate_risk(
    events: &[EventRecord],
    records: &[StepRecord],
) -> (BTreeMap<String, RiskBin>, MomentSums, [MomentSums; 4]) {
    let mut bins = BTreeMap::new();
    let mut fee_severity = MomentSums::default();
    let mut fee_markout: [MomentSums; 4] = Default::default();
    for event in events {
        let volume = event.delta.abs();
        if volume == 0.0 {
            continue;
        }
        let bin_idx = fee_bin(event.fee_used);
        let severity = event.ell.max(0.0) / volume;
        fee_severity.observe(event.fee_used, severity);
        let mut targets = [None; 4];
        for (h, horizon) in HORIZONS_HOURS.iter().enumerate() {
            let target = if *horizon == 0.0 {
                event.step
            } else {
                event.step + (event.time_frac + horizon / (1.0 / 3600.0)).ceil() as usize
            };
            if let Some(record) = records.get(target) {
                let markout = event.delta * (record.cex_price - event.pbar);
                targets[h] = Some(markout);
                fee_markout[h].observe(event.fee_used, markout / volume);
            }
        }
        for kind in ["all", event_kind_name(event.kind)] {
            let bin = bins
                .entry(risk_key(kind, bin_idx))
                .or_insert_with(RiskBin::default);
            bin.fill_count += 1;
            bin.volume += volume;
            bin.ell_positive += event.ell.max(0.0);
            bin.sum_ell_positive_per_unit += severity;
            for (h, markout) in targets.iter().enumerate() {
                if let Some(markout) = markout {
                    bin.markout_count[h] += 1;
                    bin.markout_volume[h] += volume;
                    bin.markout_sum[h] += markout;
                    bin.sum_markout_per_unit[h] += markout / volume;
                }
            }
        }
    }
    (bins, fee_severity, fee_markout)
}

fn sha256(path: &Path) -> String {
    let mut file = File::open(path).unwrap();
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer).unwrap();
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    format!("{:x}", hasher.finalize())
}

fn load_plan() -> DiagnosticPlan {
    let root = Path::new(ROOT);
    let plan_path = root.join(PLAN_REL);
    let expected = std::fs::read_to_string(root.join(PLAN_HASH_REL))
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .to_owned();
    assert_eq!(sha256(&plan_path), expected);
    let plan: DiagnosticPlan =
        serde_json::from_reader(BufReader::new(File::open(plan_path).unwrap())).unwrap();
    assert_eq!(plan.selection_sha256, sha256(&root.join(SELECTION_REL)));
    assert_eq!(plan.policies.len(), plan.n_unique_policies);
    assert_eq!(plan.seed_block.start_inclusive, 20_000);
    assert_eq!(plan.seed_block.end_exclusive, 20_100);
    assert_eq!(plan.seed_block.n, 100);
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
        name: "m3diagnostics".into(),
        description: "M3 amended selected-policy training diagnostics".into(),
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

fn merge_policy(
    aggregate: &mut PolicyAggregate,
    decisions: DecisionAggregate,
    events: &[EventRecord],
    records: &[StepRecord],
) {
    aggregate.decisions.merge(decisions);
    let es = summarize_events(events, records);
    let ls = summarize(records);
    let c = &mut aggregate.channels;
    c.episodes += 1;
    c.potential += es.potential_volume;
    c.served += es.served_fund_volume;
    c.alloc_amm += es.alloc_amm_share.unwrap_or(0.0);
    c.alloc_cex += es.alloc_cex_share.unwrap_or(0.0);
    c.alloc_unserved += es.alloc_unserved_share.unwrap_or(0.0);
    c.fill_incidence += es.incidence_event.unwrap_or(0.0);
    c.conditional_fill_size += es.cond_fill_size.unwrap_or(0.0);
    c.a += es.a_fill;
    c.a_arb += es.a_arb;
    c.a_fund += es.a_fund;
    c.b_fund += es.b_fund;
    c.fees += es.fees_total;
    c.fees_arb += es.fees_arb;
    c.fees_fund += es.fees_fund;
    c.u += es.u_lp_rel;
    c.quote_error += ls.mean_abs_log_gap.unwrap_or(0.0);
    c.n_fund_events += es.n_fund_events;
    c.fund_fill_count += events
        .iter()
        .filter(|event| event.kind != EventKind::Arb && event.delta != 0.0)
        .count() as u64;

    let (risk, severity, markouts) = accumulate_risk(events, records);
    for (key, value) in risk {
        aggregate.risk_bins.entry(key).or_default().merge(value);
    }
    aggregate.fee_severity_correlation_sums.merge(severity);
    for (left, right) in aggregate
        .fee_markout_correlation_sums
        .iter_mut()
        .zip(markouts)
    {
        left.merge(right);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let shard: usize = arg(&args, "--shard")
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let of: usize = arg(&args, "--of")
        .and_then(|value| value.parse().ok())
        .unwrap_or(1);
    assert!(of > 0 && shard < of);

    let plan = load_plan();
    let cells = load_cells();
    let mut by_cell: HashMap<usize, Vec<DiagnosticPolicy>> = HashMap::new();
    for policy in plan.policies {
        by_cell
            .entry(policy.cell.cell_idx)
            .or_default()
            .push(policy);
    }
    let mut output = Vec::new();
    let start = std::time::Instant::now();
    let mut n_done = 0usize;
    for cell in cells.iter().filter(|cell| cell.idx % of == shard) {
        let policies = by_cell.remove(&cell.idx).unwrap_or_default();
        let mut aggregates: Vec<PolicyAggregate> = (0..policies.len())
            .map(|_| PolicyAggregate::default())
            .collect();
        let cfg0 = config_for(cell);
        for seed in plan.seed_block.start_inclusive..plan.seed_block.end_exclusive {
            let mut cfg = cfg0.clone();
            cfg.seed = seed;
            let prices = generate_gbm(cfg.n_steps, S0_PRICE, 0.0, cfg.sigma, cfg.dt_years(), seed);
            for (index, selected) in policies.iter().enumerate() {
                let mut policy = AuditedPolicy::new(selected.policy.clone());
                let (records, events) = run_simulation_with_events(&cfg, &prices, &mut policy);
                merge_policy(&mut aggregates[index], policy.decisions, &events, &records);
            }
        }
        for (selected, aggregate) in policies.into_iter().zip(aggregates) {
            output.push(PolicyOutput {
                cell: selected.cell,
                policy: selected.policy,
                assignments: selected.assignments,
                aggregate,
            });
        }
        n_done += 1;
        eprintln!(
            "[diagnostic shard {shard}/{of}] cell {} ({} s{} z{} {}) done - {n_done} cells, {:.0}s elapsed",
            cell.idx,
            cell.stratum,
            cell.sigma,
            cell.z,
            cell.speed,
            start.elapsed().as_secs_f64()
        );
    }

    let result = ShardOutput {
        step: "M3 amended selected-policy diagnostics, training only",
        shard,
        of,
        seed_start_inclusive: plan.seed_block.start_inclusive,
        seed_end_exclusive: plan.seed_block.end_exclusive,
        fee_histogram: serde_json::json!({
            "n_bins": N_FEE_BINS,
            "scale": "log",
            "min_fee_bps": MIN_FEE_BPS,
            "max_fee_bps": MAX_FEE_BPS,
        }),
        gap_bin_upper_edges_bps: GAP_EDGES_BPS,
        markout_horizons_hours: HORIZONS_HOURS,
        policies: output,
    };
    let path = format!("{ROOT}/.local/lvr/m3_amended_diagnostics_shard{shard}.json.gz");
    let mut writer = GzEncoder::new(File::create(&path).unwrap(), Compression::default());
    serde_json::to_writer(&mut writer, &result).unwrap();
    writer.finish().unwrap();
    eprintln!("wrote {path}");
}
