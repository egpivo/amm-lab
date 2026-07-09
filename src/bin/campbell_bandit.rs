#![allow(clippy::type_complexity)]

use rand::rngs::StdRng;
/// Contextual bandit toy experiment.
///
/// Clean controlled test of whether adaptive fee control using (gap + vol + flow)
/// adds value beyond oracle_gap (gap only).
///
/// No AMM reserves. No hedging portfolio. No LVR accounting.
/// Reward = fee_revenue - adverse_cost, where the optimal fee differs by hidden regime.
///
/// Usage: cargo run --release --bin campbell_bandit
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;

// ── Actions ───────────────────────────────────────────────────────────────────

const ACTIONS: [f64; 6] = [1.0, 3.0, 6.0, 10.0, 20.0, 30.0];
const N_ACTIONS: usize = 6;

// ── Training / eval constants ─────────────────────────────────────────────────

const N_TRAIN: usize = 300_000;
const N_EVAL: usize = 100_000;
const ALPHA: f64 = 0.05;
const EPSILON_START: f64 = 1.0;
const EPSILON_MIN: f64 = 0.05;
const EPSILON_DECAY: f64 = 1.0 - (EPSILON_START - EPSILON_MIN) / N_TRAIN as f64;
const NOISE_SIGMA: f64 = 1.5; // reward noise during training
const REGIME_P: [f64; 3] = [0.50, 0.35, 0.15]; // Normal, Toxic, HighVolToxic

// ── Hidden regime ─────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Regime {
    Normal,
    Toxic,
    HighVolToxic,
}

impl Regime {
    fn name(self) -> &'static str {
        match self {
            Self::Normal => "Normal",
            Self::Toxic => "Toxic",
            Self::HighVolToxic => "HighVolToxic",
        }
    }
    fn index(self) -> usize {
        match self {
            Self::Normal => 0,
            Self::Toxic => 1,
            Self::HighVolToxic => 2,
        }
    }
}

fn sample_regime(rng: &mut StdRng) -> Regime {
    let u: f64 = rng.gen_range(0.0..1.0);
    if u < REGIME_P[0] {
        Regime::Normal
    } else if u < REGIME_P[0] + REGIME_P[1] {
        Regime::Toxic
    } else {
        Regime::HighVolToxic
    }
}

// ── Context sampling ──────────────────────────────────────────────────────────
// Given regime, sample observable (gap_bucket, vol_bucket, flow_bucket).
// Distributions overlap so the policy cannot perfectly infer the regime from context.

fn sample_bucket(rng: &mut StdRng, p0: f64, p1: f64) -> u8 {
    let u: f64 = rng.gen_range(0.0..1.0);
    if u < p0 {
        0
    } else if u < p0 + p1 {
        1
    } else {
        2
    }
}

fn sample_context(rng: &mut StdRng, regime: Regime) -> (u8, u8, u8) {
    let (gp0, gp1, vp0, vp1, fp0, fp1) = match regime {
        Regime::Normal => (0.50, 0.40, 0.60, 0.30, 0.70, 0.20), // low-gap, low-vol, fund-heavy
        Regime::Toxic => (0.20, 0.45, 0.20, 0.45, 0.10, 0.20),  // spread; arb-heavy
        Regime::HighVolToxic => (0.10, 0.30, 0.05, 0.15, 0.05, 0.20), // high-gap, high-vol, arb-heavy
    };
    let gap = sample_bucket(rng, gp0, gp1);
    let vol = sample_bucket(rng, vp0, vp1);
    let flow = sample_bucket(rng, fp0, fp1);
    (gap, vol, flow)
}

// ── Reward function ───────────────────────────────────────────────────────────
//
// fee_revenue = fee × (fund_weight × fund_volume(fee) + arb_weight × arb_volume(fee))
//   fund_volume: drops to 0 at 15 bps (elastic)
//   arb_volume:  drops to 0 at 40 bps (inelastic)
//   Weights depend on flow_bucket.
//
// adverse_cost = regime_scale × vol_amp × 1 / (1 + fee / 10)
//   Higher fee reduces adverse cost (fee protection), but fund traders also leave faster.
//   Normal: tiny adverse cost → fee_rev dominates → optimal ≈ 6 bps
//   Toxic:  large adverse cost → trade-off → optimal ≈ 20 bps
//   HighVolToxic: vol amplifies adverse cost → optimal ≈ 20 bps

fn reward_det(fee_bps: f64, vol: u8, flow: u8, regime: Regime) -> f64 {
    let fund_vol = (1.0 - fee_bps / 15.0).max(0.0);
    let arb_vol = (1.0 - fee_bps / 40.0).max(0.0);
    let (fw, aw) = match flow {
        0 => (2.0, 0.2), // fund-dom
        1 => (1.0, 0.8), // mixed
        _ => (0.3, 1.5), // arb-dom
    };
    let fee_revenue = fee_bps * (fw * fund_vol + aw * arb_vol);
    let regime_scale = match regime {
        Regime::Normal => 0.1,
        Regime::Toxic => 3.0,
        Regime::HighVolToxic => 6.0,
    };
    let vol_amp: f64 = match vol {
        0 => 0.5,
        1 => 1.0,
        _ => 2.0,
    };
    let adverse_cost = regime_scale * vol_amp / (1.0 + fee_bps / 10.0);
    fee_revenue - adverse_cost
}

fn reward_noisy(fee_bps: f64, vol: u8, flow: u8, regime: Regime, rng: &mut StdRng) -> f64 {
    let det = reward_det(fee_bps, vol, flow, regime);
    let noise = Normal::new(0.0, NOISE_SIGMA).unwrap().sample(rng);
    det + noise
}

// ── Heuristic policies ────────────────────────────────────────────────────────

fn oracle_gap_fee(gap: u8, _vol: u8, _flow: u8) -> f64 {
    match gap {
        0 => 3.0,
        1 => 6.0,
        _ => 10.0,
    }
}

fn flow_aware_fee(gap: u8, _vol: u8, flow: u8) -> f64 {
    match (gap, flow) {
        (0, 0) => 3.0,
        (0, 1) => 3.0,
        (0, 2) => 6.0,
        (1, 0) => 6.0,
        (1, 1) => 6.0,
        (1, 2) => 20.0,
        (_, 0) => 6.0,
        (_, 1) => 10.0,
        _ => 20.0,
    }
}

fn vol_flow_aware_fee(_gap: u8, vol: u8, flow: u8) -> f64 {
    match (flow, vol) {
        (0, _) => 6.0,  // fund-dom: 6 bps regardless
        (1, 0) => 6.0,  // mixed, low-vol
        (1, _) => 10.0, // mixed, mid/hi-vol
        (2, 0) => 10.0, // arb-dom, low-vol
        (2, 1) => 20.0, // arb-dom, mid-vol
        (2, _) => 20.0, // arb-dom, hi-vol
        _ => 6.0,
    }
}

// ── Phase 0: fee sweep by regime ──────────────────────────────────────────────

fn phase0_sweep() {
    println!("=== Phase 0: Fee Sweep by Regime (expected deterministic reward) ===");
    println!("Rewards are averaged over (gap, vol, flow) sampled from regime distribution.\n");
    let sep = "─".repeat(72);
    println!("{sep}");
    println!(
        "{:>14}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}",
        "fee (bps)", "1", "3", "6", "10", "20", "30"
    );
    println!("{sep}");

    let regimes = [Regime::Normal, Regime::Toxic, Regime::HighVolToxic];
    let n_mc: usize = 50_000;
    let mut rng = StdRng::seed_from_u64(9999);

    let mut best_fees = Vec::new();

    for regime in regimes {
        // Sample (vol, flow) from regime distribution
        let samples: Vec<(u8, u8, u8)> = (0..n_mc)
            .map(|_| sample_context(&mut rng, regime))
            .collect();

        let means: Vec<f64> = ACTIONS
            .iter()
            .map(|&fee| {
                samples
                    .iter()
                    .map(|&(_, vol, flow)| reward_det(fee, vol, flow, regime))
                    .sum::<f64>()
                    / n_mc as f64
            })
            .collect();

        let best_idx = means
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(2);
        best_fees.push((regime, ACTIONS[best_idx], means[best_idx]));

        let row_str: Vec<String> = means
            .iter()
            .enumerate()
            .map(|(i, &m)| {
                if i == best_idx {
                    format!("{:>7.2}*", m)
                } else {
                    format!("{:>7.2} ", m)
                }
            })
            .collect();
        println!("{:>14}  {}", regime.name(), row_str.join("  "));
    }
    println!("{sep}");
    println!("\n* = best fee for that regime");
    println!("\nBest fixed fee by regime:");
    for (r, fee, mean_r) in &best_fees {
        println!(
            "  {:>14}: {:>5.0} bps  (mean reward = {:.3})",
            r.name(),
            fee,
            mean_r
        );
    }
    println!();
}

// ── Evaluation harness ────────────────────────────────────────────────────────

struct StepRecord {
    reward: f64,
    fee_chosen: f64,
    regime: Regime,
}

fn eval_policy(policy: &dyn Fn(u8, u8, u8) -> f64, n_steps: usize, seed: u64) -> Vec<StepRecord> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n_steps)
        .map(|_| {
            let regime = sample_regime(&mut rng);
            let (gap, vol, flow) = sample_context(&mut rng, regime);
            let fee = policy(gap, vol, flow);
            StepRecord {
                reward: reward_det(fee, vol, flow, regime), // deterministic for eval
                fee_chosen: fee,
                regime,
            }
        })
        .collect()
}

fn summarize(records: &[StepRecord]) -> (f64, f64, f64, [f64; 3], [usize; 3]) {
    let mut rewards: Vec<f64> = records.iter().map(|r| r.reward).collect();
    let mean = rewards.iter().sum::<f64>() / rewards.len() as f64;
    rewards.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p05 = rewards[(0.05 * (rewards.len() - 1) as f64).round() as usize];
    let mean_fee = records.iter().map(|r| r.fee_chosen).sum::<f64>() / records.len() as f64;
    let mut by_regime = [0.0f64; 3];
    let mut by_regime_n = [0usize; 3];
    for r in records {
        by_regime[r.regime.index()] += r.reward;
        by_regime_n[r.regime.index()] += 1;
    }
    for i in 0..3 {
        if by_regime_n[i] > 0 {
            by_regime[i] /= by_regime_n[i] as f64;
        }
    }
    (mean, p05, mean_fee, by_regime, by_regime_n)
}

// ── Tabular bandit ────────────────────────────────────────────────────────────

struct BanditPolicy {
    q: HashMap<(u8, u8, u8), [f64; N_ACTIONS]>,
    visits: HashMap<(u8, u8, u8), [u32; N_ACTIONS]>,
    epsilon: f64,
}

impl BanditPolicy {
    fn new() -> Self {
        Self {
            q: HashMap::new(),
            visits: HashMap::new(),
            epsilon: EPSILON_START,
        }
    }

    fn best_action(&self, s: (u8, u8, u8)) -> usize {
        self.q
            .get(&s)
            .map(|q| {
                q.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(2)
            })
            .unwrap_or(2) // default: 6 bps
    }

    fn choose(&mut self, s: (u8, u8, u8), rng: &mut StdRng) -> usize {
        if rng.gen_range(0.0..1.0f64) < self.epsilon {
            rng.gen_range(0..N_ACTIONS)
        } else {
            self.best_action(s)
        }
    }

    fn update(&mut self, s: (u8, u8, u8), a: usize, r: f64) {
        let q = self.q.entry(s).or_insert([0.0; N_ACTIONS]);
        q[a] += ALPHA * (r - q[a]);
        self.visits.entry(s).or_insert([0u32; N_ACTIONS])[a] += 1;
    }

    fn decay(&mut self) {
        self.epsilon = (self.epsilon * EPSILON_DECAY).max(EPSILON_MIN);
    }
}

fn train_bandit(seed: u64) -> BanditPolicy {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut policy = BanditPolicy::new();
    for _ in 0..N_TRAIN {
        let regime = sample_regime(&mut rng);
        let (gap, vol, flow) = sample_context(&mut rng, regime);
        let s = (gap, vol, flow);
        let a = policy.choose(s, &mut rng);
        let r = reward_noisy(ACTIONS[a], vol, flow, regime, &mut rng);
        policy.update(s, a, r);
        policy.decay();
    }
    policy
}

fn bandit_fee(policy: &BanditPolicy, gap: u8, vol: u8, flow: u8) -> f64 {
    ACTIONS[policy.best_action((gap, vol, flow))]
}

// ── Phase 1: compare heuristic policies ──────────────────────────────────────

fn phase1_compare() {
    println!("=== Phase 1: Heuristic Policy Comparison ({N_EVAL} eval steps) ===\n");
    let sep = "─".repeat(84);
    println!("{sep}");
    println!(
        "{:<16} {:>8} {:>8} {:>8}  {:>10} {:>12} {:>14}",
        "policy", "mean", "p05", "fee_bps", "Normal", "Toxic", "HighVolToxic"
    );
    println!("{sep}");

    let policies: &[(&str, &dyn Fn(u8, u8, u8) -> f64)] = &[
        ("fixed_3bps", &|_, _, _| 3.0),
        ("fixed_6bps", &|_, _, _| 6.0),
        ("fixed_10bps", &|_, _, _| 10.0),
        ("oracle_gap", &oracle_gap_fee),
        ("flow_aware", &flow_aware_fee),
        ("vol_flow_aware", &vol_flow_aware_fee),
    ];

    let mut results: Vec<(&str, Vec<StepRecord>)> = Vec::new();
    for (name, policy) in policies {
        let records = eval_policy(policy, N_EVAL, 42_000);
        let (mean, p05, fee, by_regime, _) = summarize(&records);
        println!(
            "{:<16} {:>8.3} {:>8.3} {:>8.2}  {:>10.3} {:>12.3} {:>14.3}",
            name, mean, p05, fee, by_regime[0], by_regime[1], by_regime[2]
        );
        results.push((name, records));
    }
    println!("{sep}");

    // Paired delta vs oracle_gap
    let og_idx = results
        .iter()
        .position(|(n, _)| *n == "oracle_gap")
        .unwrap();
    let og_records = &results[og_idx].1;
    println!("\n  Paired delta vs oracle_gap:");
    for (name, records) in &results {
        if *name == "oracle_gap" {
            continue;
        }
        let delta: f64 = records
            .iter()
            .zip(og_records)
            .map(|(r, og)| r.reward - og.reward)
            .sum::<f64>()
            / records.len() as f64;
        let beat_pct = records
            .iter()
            .zip(og_records)
            .filter(|(r, og)| r.reward > og.reward)
            .count() as f64
            / records.len() as f64
            * 100.0;
        println!("  {:<16} Δ={:>+7.3}  beat={:.1}%", name, delta, beat_pct);
    }
    println!();
}

// ── Phase 2: train and evaluate tabular bandit ────────────────────────────────

fn phase2_bandit() {
    println!("=== Phase 2: Tabular Contextual Bandit ===");
    println!("Training: {N_TRAIN} steps, α={ALPHA}, ε: {EPSILON_START}→{EPSILON_MIN}");
    println!(
        "State: (gap_bucket, vol_bucket, flow_bucket)  Actions: {:?}\n",
        ACTIONS
    );

    let policy = train_bandit(42);

    // Evaluate
    let records = eval_policy(
        &|gap, vol, flow| bandit_fee(&policy, gap, vol, flow),
        N_EVAL,
        99_000,
    );
    let (mean, p05, fee, by_regime, by_regime_n) = summarize(&records);

    // Compare oracle_gap on same eval steps
    let og_records = eval_policy(&oracle_gap_fee, N_EVAL, 99_000);
    let (og_mean, og_p05, og_fee, og_by_regime, _) = summarize(&og_records);

    let sep = "─".repeat(60);
    println!("{sep}");
    println!(
        "{:<20} {:>8} {:>8} {:>8}",
        "policy", "mean", "p05", "fee_bps"
    );
    println!("{sep}");
    println!(
        "{:<20} {:>8.3} {:>8.3} {:>8.2}",
        "oracle_gap", og_mean, og_p05, og_fee
    );
    println!(
        "{:<20} {:>8.3} {:>8.3} {:>8.2}",
        "tabular_bandit", mean, p05, fee
    );
    println!("{sep}");

    let deltas: Vec<f64> = records
        .iter()
        .zip(&og_records)
        .map(|(r, og)| r.reward - og.reward)
        .collect();
    let delta = deltas.iter().sum::<f64>() / deltas.len() as f64;
    let beat_pct = deltas.iter().filter(|&&d| d > 0.0).count() as f64 / deltas.len() as f64 * 100.0;
    println!(
        "\n  Paired Δ vs oracle_gap: {:>+.3}  beat={:.1}%",
        delta, beat_pct
    );

    // Per-regime breakdown
    println!("\n  Per-regime breakdown:");
    println!(
        "  {:>14} {:>10} {:>10} {:>8}",
        "regime", "bandit", "oracle_gap", "Δ"
    );
    for (i, name) in ["Normal", "Toxic", "HighVolToxic"].iter().enumerate() {
        println!(
            "  {:>14} {:>10.3} {:>10.3} {:>+8.3}  (n={})",
            name,
            by_regime[i],
            og_by_regime[i],
            by_regime[i] - og_by_regime[i],
            by_regime_n[i]
        );
    }

    // Policy table
    println!("\n  Learned policy table (fee chosen per state, visits in parens):");
    println!(
        "  {:>5} {:>5} {:>5} | {:>6}  {:>8}",
        "gap", "vol", "flow", "fee", "visits"
    );
    println!("  {}", "─".repeat(40));
    let mut states: Vec<(u8, u8, u8)> = policy.q.keys().copied().collect();
    states.sort();
    for (gap, vol, flow) in &states {
        let s = (*gap, *vol, *flow);
        let a = policy.best_action(s);
        let visits: u32 = policy.visits.get(&s).map(|v| v.iter().sum()).unwrap_or(0);
        let flow_label = match flow {
            0 => "fund",
            1 => "mix",
            _ => "arb",
        };
        let vol_label = match vol {
            0 => "lo",
            1 => "mid",
            _ => "hi",
        };
        println!(
            "  {:>5} {:>5}({}) {:>4}({}) | {:>6.1}  {:>8}",
            gap, vol, vol_label, flow, flow_label, ACTIONS[a], visits
        );
    }

    // Success criterion check
    println!("\n  Success criterion:");
    let mut ok = true;
    // Fund-dom states should have low fees
    for gap in 0u8..3 {
        for vol in 0u8..3 {
            let fee = bandit_fee(&policy, gap, vol, 0);
            if fee > 10.0 {
                println!(
                    "  [FAIL] fund-dom ({},{},flow=0): fee={:.0} bps (expected ≤10)",
                    gap, vol, fee
                );
                ok = false;
            }
        }
    }
    // Arb-dom + hi-vol states should have higher fees
    for gap in 0u8..3 {
        let fee = bandit_fee(&policy, gap, 2, 2);
        if fee < 10.0 {
            println!(
                "  [FAIL] arb+hi-vol ({},vol=2,flow=2): fee={:.0} bps (expected ≥10)",
                gap, fee
            );
            ok = false;
        }
    }
    // Overall win
    if delta <= 0.0 {
        println!("  [FAIL] overall paired delta not positive: {:.3}", delta);
        ok = false;
    }
    // Must beat oracle_gap especially in Toxic and HighVolToxic
    if by_regime[1] <= og_by_regime[1] {
        println!(
            "  [FAIL] Toxic regime: bandit {:.3} ≤ oracle_gap {:.3}",
            by_regime[1], og_by_regime[1]
        );
        ok = false;
    }
    if by_regime[2] <= og_by_regime[2] {
        println!(
            "  [FAIL] HighVolToxic: bandit {:.3} ≤ oracle_gap {:.3}",
            by_regime[2], og_by_regime[2]
        );
        ok = false;
    }
    if ok {
        println!("  [PASS] All success criteria met.");
    }
    println!();

    // Multi-seed robustness for the bandit
    println!("=== Multi-seed robustness ({} seeds) ===", 5);
    let sep2 = "─".repeat(60);
    println!("{sep2}");
    println!(
        "{:>12} {:>10} {:>10} {:>10} {:>8}",
        "train_seed", "bandit", "oracle_gap", "delta", "beat%"
    );
    println!("{sep2}");
    for &seed in &[0u64, 42, 123, 456, 789] {
        let bp = train_bandit(seed);
        // Use same eval seed for all runs to get true paired comparison
        let b_recs = eval_policy(&|g, v, f| bandit_fee(&bp, g, v, f), N_EVAL, 99_000);
        let og_recs = eval_policy(&oracle_gap_fee, N_EVAL, 99_000);
        let b_mean = b_recs.iter().map(|r| r.reward).sum::<f64>() / N_EVAL as f64;
        let og_mean2 = og_recs.iter().map(|r| r.reward).sum::<f64>() / N_EVAL as f64;
        let d: Vec<f64> = b_recs
            .iter()
            .zip(&og_recs)
            .map(|(b, o)| b.reward - o.reward)
            .collect();
        let delta2 = d.iter().sum::<f64>() / d.len() as f64;
        let beat2 = d.iter().filter(|&&x| x > 0.0).count() as f64 / d.len() as f64 * 100.0;
        println!(
            "{:>12} {:>10.3} {:>10.3} {:>+10.3} {:>7.1}%",
            seed, b_mean, og_mean2, delta2, beat2
        );
    }
    println!("{sep2}");
    println!();
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    println!("Campbell Contextual Bandit Experiment");
    println!(
        "State: (gap, vol, flow) buckets  |  Actions: {:?} bps",
        ACTIONS
    );
    println!("Hidden regimes: Normal (50%) / Toxic (35%) / HighVolToxic (15%)");
    println!("Reward: fee_revenue(fee, flow) - adverse_cost(fee, vol, regime)");
    println!("{}\n", "═".repeat(72));

    phase0_sweep();
    phase1_compare();
    phase2_bandit();
}
