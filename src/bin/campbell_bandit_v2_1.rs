#![allow(dead_code, clippy::too_many_arguments, clippy::type_complexity)]

use rand::rngs::StdRng;
/// Contextual fee-control v2.1 — diagnostics and ablations.
///
/// Three experiments:
///   1. State ablation  — which dims actually add value beyond oracle_gap?
///   2. Calibrated oracle_gap — best (base, multiplier, band) via grid search
///   3. Switching-cost sensitivity — 4 lambda levels on full G-state model
///
/// Reward and regime generator are identical to v2; only state projection,
/// oracle calibration, and lambda vary.
///
/// Usage: cargo run --release --bin campbell_bandit_v2_1
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;

// ── Constants ─────────────────────────────────────────────────────────────────

const ACTIONS: [f64; 9] = [1.0, 3.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0];
const N_ACTIONS: usize = 9;
const DEFAULT_ACTION: usize = 3; // 6 bps

const N_TRAIN: usize = 400_000;
const N_EVAL: usize = 100_000;
const N_CALIB: usize = 50_000;

const ALPHA: f64 = 0.05;
const GAMMA: f64 = 0.99;
const EPSILON_START: f64 = 1.0;
const EPSILON_MIN: f64 = 0.05;
const NOISE_SIGMA: f64 = 1.5;

const TRAIN_SEED: u64 = 7;
const EVAL_SEED: u64 = 99_000;
const CALIB_SEED: u64 = 77_000;

const SWITCH_LAMBDA_BASE: f64 = 0.5;
const SWITCH_LAMBDAS: [f64; 4] = [0.0, 0.1, 0.5, 1.5];

const REGIME_INIT: [f64; 3] = [0.60, 0.30, 0.10];
const REGIME_TRANS: [[f64; 3]; 3] = [[0.97, 0.02, 0.01], [0.05, 0.92, 0.03], [0.10, 0.10, 0.80]];

// ── Regime ────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Regime {
    Normal,
    Toxic,
    HighVolToxic,
}

impl Regime {
    fn index(self) -> usize {
        match self {
            Self::Normal => 0,
            Self::Toxic => 1,
            Self::HighVolToxic => 2,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Normal => "Normal",
            Self::Toxic => "Toxic",
            Self::HighVolToxic => "HiVolTox",
        }
    }
}

fn transition_regime(r: Regime, rng: &mut StdRng) -> Regime {
    let row = &REGIME_TRANS[r.index()];
    let u: f64 = rng.gen_range(0.0..1.0);
    if u < row[0] {
        Regime::Normal
    } else if u < row[0] + row[1] {
        Regime::Toxic
    } else {
        Regime::HighVolToxic
    }
}

fn sample_regime_init(rng: &mut StdRng) -> Regime {
    let u: f64 = rng.gen_range(0.0..1.0);
    if u < REGIME_INIT[0] {
        Regime::Normal
    } else if u < REGIME_INIT[0] + REGIME_INIT[1] {
        Regime::Toxic
    } else {
        Regime::HighVolToxic
    }
}

fn bucket3(rng: &mut StdRng, p0: f64, p1: f64) -> u8 {
    let u: f64 = rng.gen_range(0.0..1.0);
    if u < p0 {
        0
    } else if u < p0 + p1 {
        1
    } else {
        2
    }
}

fn sample_obs(rng: &mut StdRng, regime: Regime) -> (u8, u8, u8, u8) {
    let (gp0, gp1, vp0, vp1, fp0, fp1, mp0, mp1) = match regime {
        Regime::Normal => (0.50, 0.40, 0.60, 0.30, 0.70, 0.20, 0.35, 0.45),
        Regime::Toxic => (0.20, 0.45, 0.20, 0.45, 0.10, 0.20, 0.15, 0.40),
        Regime::HighVolToxic => (0.10, 0.30, 0.05, 0.15, 0.05, 0.20, 0.05, 0.20),
    };
    (
        bucket3(rng, gp0, gp1),
        bucket3(rng, vp0, vp1),
        bucket3(rng, fp0, fp1),
        bucket3(rng, mp0, mp1),
    )
}

fn persistence_bucket(streak: usize) -> u8 {
    if streak <= 1 {
        0
    } else if streak <= 5 {
        1
    } else {
        2
    }
}

fn prev_fee_bucket(fee: f64) -> u8 {
    ACTIONS
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| ((*a - fee).abs()).partial_cmp(&((*b - fee).abs())).unwrap())
        .map(|(i, _)| i as u8)
        .unwrap_or(DEFAULT_ACTION as u8)
}

// ── Reward (lambda as explicit parameter) ─────────────────────────────────────

fn reward_det(
    fee: f64,
    prev_fee: f64,
    vol: u8,
    flow: u8,
    volume: u8,
    regime: Regime,
    lam: f64,
) -> f64 {
    let fund_vol = (1.0 - fee / 15.0).max(0.0);
    let arb_vol = (1.0 - fee / 40.0).max(0.0);
    let (fw, aw) = match flow {
        0 => (2.0, 0.2),
        1 => (1.0, 0.8),
        _ => (0.3, 1.5),
    };
    let vol_scale: f64 = match volume {
        0 => 0.7,
        1 => 1.0,
        _ => 1.4,
    };
    let fee_rev = fee * (fw * fund_vol + aw * arb_vol) * vol_scale;
    let rs: f64 = match regime {
        Regime::Normal => 0.1,
        Regime::Toxic => 3.0,
        Regime::HighVolToxic => 6.0,
    };
    let va: f64 = match vol {
        0 => 0.5,
        1 => 1.0,
        _ => 2.0,
    };
    let tva: f64 = 1.0 + 0.5 * (volume as f64) * (flow as f64 / 2.0);
    let adverse = rs * va * tva / (1.0 + fee / 10.0);
    let switch = lam * (fee - prev_fee).abs() / 10.0;
    fee_rev - adverse - switch
}

fn reward_noisy(
    fee: f64,
    prev_fee: f64,
    vol: u8,
    flow: u8,
    volume: u8,
    regime: Regime,
    lam: f64,
    rng: &mut StdRng,
) -> f64 {
    reward_det(fee, prev_fee, vol, flow, volume, regime, lam)
        + Normal::new(0.0, NOISE_SIGMA).unwrap().sample(rng)
}

// ── Environment ───────────────────────────────────────────────────────────────

struct Env {
    regime: Regime,
    gap: u8,
    vol: u8,
    flow: u8,
    volume: u8,
    flow_streak: usize,
    prev_fee_bps: f64,
}

impl Env {
    fn new(rng: &mut StdRng) -> Self {
        let r = sample_regime_init(rng);
        let (g, v, fl, m) = sample_obs(rng, r);
        Self {
            regime: r,
            gap: g,
            vol: v,
            flow: fl,
            volume: m,
            flow_streak: 1,
            prev_fee_bps: 6.0,
        }
    }

    // Returns (gap, vol, flow, volume, persistence, prev_fee_idx, prev_fee_bps)
    fn obs(&self) -> (u8, u8, u8, u8, u8, u8, f64) {
        let pers = persistence_bucket(self.flow_streak);
        let pfi = prev_fee_bucket(self.prev_fee_bps);
        (
            self.gap,
            self.vol,
            self.flow,
            self.volume,
            pers,
            pfi,
            self.prev_fee_bps,
        )
    }

    fn advance(&mut self, rng: &mut StdRng, fee_bps: f64) {
        self.regime = transition_regime(self.regime, rng);
        let (ng, nv, nf, nm) = sample_obs(rng, self.regime);
        if nf == self.flow {
            self.flow_streak += 1;
        } else {
            self.flow_streak = 1;
        }
        self.gap = ng;
        self.vol = nv;
        self.flow = nf;
        self.volume = nm;
        self.prev_fee_bps = fee_bps;
    }
}

// ── Flexible Q-learner (Vec<u8> key = projection of 6-dim obs) ────────────────
// mask order: [gap, vol, flow, volume, persistence, prev_fee_idx]

struct FlexQLearner {
    q: HashMap<Vec<u8>, [f64; N_ACTIONS]>,
    visits: HashMap<Vec<u8>, [u32; N_ACTIONS]>,
    epsilon: f64,
    mask: [bool; 6],
}

impl FlexQLearner {
    fn new(mask: [bool; 6]) -> Self {
        Self {
            q: HashMap::new(),
            visits: HashMap::new(),
            epsilon: EPSILON_START,
            mask,
        }
    }

    fn project(&self, gap: u8, vol: u8, flow: u8, volume: u8, pers: u8, pfi: u8) -> Vec<u8> {
        let all = [gap, vol, flow, volume, pers, pfi];
        (0..6).filter(|&i| self.mask[i]).map(|i| all[i]).collect()
    }

    fn best_action(&self, key: &[u8]) -> usize {
        self.q
            .get(key)
            .and_then(|q| {
                q.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
            })
            .unwrap_or(DEFAULT_ACTION)
    }

    fn best_q(&self, key: &[u8]) -> f64 {
        self.q
            .get(key)
            .map(|q| q.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
            .unwrap_or(0.0)
    }

    fn choose(&mut self, key: &[u8], rng: &mut StdRng) -> usize {
        if rng.gen_range(0.0..1.0f64) < self.epsilon {
            rng.gen_range(0..N_ACTIONS)
        } else {
            self.best_action(key)
        }
    }

    fn update(&mut self, key: Vec<u8>, a: usize, r: f64, next_key: &[u8]) {
        let max_next = self.best_q(next_key);
        let q = self.q.entry(key.clone()).or_insert([0.0; N_ACTIONS]);
        q[a] += ALPHA * (r + GAMMA * max_next - q[a]);
        self.visits.entry(key).or_insert([0u32; N_ACTIONS])[a] += 1;
    }
}

fn train_flex(mask: [bool; 6], n_steps: usize, lam: f64, seed: u64) -> FlexQLearner {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut env = Env::new(&mut rng);
    let mut learner = FlexQLearner::new(mask);
    let decay = (EPSILON_START - EPSILON_MIN) / n_steps as f64;
    for _ in 0..n_steps {
        let (gap, vol, flow, volume, pers, pfi, pfb) = env.obs();
        let key = learner.project(gap, vol, flow, volume, pers, pfi);
        let a = learner.choose(&key, &mut rng);
        let fee = ACTIONS[a];
        let r = reward_noisy(fee, pfb, vol, flow, volume, env.regime, lam, &mut rng);
        env.advance(&mut rng, fee);
        let (ng, nv, nf, nm, np, npfi, _) = env.obs();
        let next_key = learner.project(ng, nv, nf, nm, np, npfi);
        learner.update(key, a, r, &next_key);
        learner.epsilon = (learner.epsilon - decay).max(EPSILON_MIN);
    }
    learner
}

// ── Evaluation harness ────────────────────────────────────────────────────────

struct EvalRec {
    reward: f64,
    fee: f64,
    regime: Regime,
    switched: bool,
}

fn eval_policy<F: Fn(u8, u8, u8, u8, u8, f64) -> f64 + ?Sized>(
    policy: &F,
    n: usize,
    seed: u64,
    lam: f64,
) -> Vec<EvalRec> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut env = Env::new(&mut rng);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let (gap, vol, flow, volume, pers, _, pfb) = env.obs();
        let fee = policy(gap, vol, flow, volume, pers, pfb);
        let r = reward_det(fee, pfb, vol, flow, volume, env.regime, lam);
        let switched = (fee - pfb).abs() > 0.5;
        let regime = env.regime;
        out.push(EvalRec {
            reward: r,
            fee,
            regime,
            switched,
        });
        env.advance(&mut rng, fee);
    }
    out
}

fn eval_flex_q(learner: &FlexQLearner, n: usize, seed: u64, lam: f64) -> Vec<EvalRec> {
    eval_policy(
        &|gap, vol, flow, volume, pers, pfb| {
            let pfi = prev_fee_bucket(pfb);
            let key = learner.project(gap, vol, flow, volume, pers, pfi);
            ACTIONS[learner.best_action(&key)]
        },
        n,
        seed,
        lam,
    )
}

struct Stats {
    mean: f64,
    p05: f64,
    mean_fee: f64,
    sw_frac: f64,
    avg_dfe: f64,
    r_regime: [f64; 3],
    f_regime: [f64; 3],
}

fn stats(recs: &[EvalRec]) -> Stats {
    let n = recs.len() as f64;
    let mean = recs.iter().map(|r| r.reward).sum::<f64>() / n;
    let mut sorted: Vec<f64> = recs.iter().map(|r| r.reward).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p05 = sorted[(0.05 * (recs.len() - 1) as f64).round() as usize];
    let mean_fee = recs.iter().map(|r| r.fee).sum::<f64>() / n;
    let sw_frac = recs.iter().filter(|r| r.switched).count() as f64 / n;
    let avg_dfe = recs
        .windows(2)
        .map(|w| (w[1].fee - w[0].fee).abs())
        .sum::<f64>()
        / (n - 1.0);
    let mut rr = [0.0f64; 3];
    let mut fr = [0.0f64; 3];
    let mut cnt = [0usize; 3];
    for r in recs {
        let i = r.regime.index();
        rr[i] += r.reward;
        fr[i] += r.fee;
        cnt[i] += 1;
    }
    for i in 0..3 {
        if cnt[i] > 0 {
            rr[i] /= cnt[i] as f64;
            fr[i] /= cnt[i] as f64;
        }
    }
    Stats {
        mean,
        p05,
        mean_fee,
        sw_frac,
        avg_dfe,
        r_regime: rr,
        f_regime: fr,
    }
}

fn mean_reward(recs: &[EvalRec]) -> f64 {
    recs.iter().map(|r| r.reward).sum::<f64>() / recs.len() as f64
}

fn paired_delta(a: &[EvalRec], b: &[EvalRec]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| x.reward - y.reward)
        .sum::<f64>()
        / a.len() as f64
}

fn beat_rate(a: &[EvalRec], b: &[EvalRec]) -> f64 {
    a.iter().zip(b).filter(|(x, y)| x.reward > y.reward).count() as f64 / a.len() as f64 * 100.0
}

// ── Heuristic policies ────────────────────────────────────────────────────────

fn oracle_gap_fee(gap: u8, _: u8, _: u8, _: u8, _: u8, _: f64) -> f64 {
    match gap {
        0 => 3.0,
        1 => 6.0,
        _ => 10.0,
    }
}

fn flow_aware_fee(_: u8, _: u8, flow: u8, _: u8, _: u8, _: f64) -> f64 {
    match flow {
        0 => 6.0,
        1 => 10.0,
        _ => 20.0,
    }
}

fn vol_flow_aware_fee(_: u8, vol: u8, flow: u8, _: u8, _: u8, _: f64) -> f64 {
    match (flow, vol) {
        (0, _) => 6.0,
        (1, 0) => 6.0,
        (1, _) => 10.0,
        (2, 0) => 10.0,
        (2, 1) => 20.0,
        _ => 20.0,
    }
}

fn persistence_aware_fee(_: u8, vol: u8, flow: u8, volume: u8, pers: u8, _: f64) -> f64 {
    match flow {
        0 => match vol {
            2 => 8.0,
            _ => 6.0,
        },
        1 => match (vol, volume) {
            (_, 2) | (2, _) => 10.0,
            _ => 8.0,
        },
        _ => match pers {
            0 => match volume {
                2 => 15.0,
                1 => 10.0,
                _ => 8.0,
            },
            _ => 20.0,
        },
    }
}

// ── Calibrated oracle_gap ─────────────────────────────────────────────────────

fn nearest_action(fee: f64) -> f64 {
    *ACTIONS
        .iter()
        .min_by(|a, b| ((*a - fee).abs()).partial_cmp(&((*b - fee).abs())).unwrap())
        .unwrap()
}

fn calib_og(gap: u8, prev_fee: f64, base: f64, mult: f64, band: f64) -> f64 {
    let rep = [2.0f64, 6.0, 15.0][gap as usize]; // representative bps for each gap bucket
    let target = nearest_action((base + mult * rep).clamp(1.0, 30.0));
    if (target - prev_fee).abs() <= band {
        prev_fee
    } else {
        target
    }
}

// ── Ablation specs ────────────────────────────────────────────────────────────
// Dim order: [gap, vol, flow, volume, persistence, prev_fee_idx]

const ABLATION_SPECS: &[(&str, [bool; 6])] = &[
    ("A:gap", [true, false, false, false, false, false]),
    ("B:gap+vol", [true, true, false, false, false, false]),
    ("C:gap+flow", [true, false, true, false, false, false]),
    ("D:gap+flow+vol", [true, true, true, false, false, false]),
    ("E:+volume", [true, true, true, true, false, false]),
    ("F:+persistence", [true, true, true, true, true, false]),
    ("G:+prev_fee", [true, true, true, true, true, true]),
];

// ── Phase 1: State ablation ───────────────────────────────────────────────────

fn phase1_ablation() {
    println!(
        "=== Phase 1: State Ablation  ({N_TRAIN} train / {N_EVAL} eval / λ={SWITCH_LAMBDA_BASE}) ==="
    );
    println!("Research question: which observable dimensions add value beyond oracle_gap?");

    let og = eval_policy(&oracle_gap_fee, N_EVAL, EVAL_SEED, SWITCH_LAMBDA_BASE);
    let vfa = eval_policy(&vol_flow_aware_fee, N_EVAL, EVAL_SEED, SWITCH_LAMBDA_BASE);
    let og_s = stats(&og);
    let vfa_s = stats(&vfa);

    let sep = "─".repeat(108);
    println!("\n{sep}");
    println!(
        "{:<18} {:>7} {:>7} {:>6} {:>6} {:>6} {:>7} {:>7} {:>8} {:>8} {:>8}",
        "spec", "mean", "p05", "fee", "sw%", "Δfee", "Δ_og", "Δ_vfa", "Normal", "Toxic", "HiVol"
    );
    println!("{sep}");

    let mut results: Vec<(String, Stats, f64, f64)> = Vec::new();

    for &(name, mask) in ABLATION_SPECS {
        let learner = train_flex(mask, N_TRAIN, SWITCH_LAMBDA_BASE, TRAIN_SEED);
        let recs = eval_flex_q(&learner, N_EVAL, EVAL_SEED, SWITCH_LAMBDA_BASE);
        let s = stats(&recs);
        let d_og = paired_delta(&recs, &og);
        let d_vfa = paired_delta(&recs, &vfa);
        println!(
            "{:<18} {:>7.3} {:>7.3} {:>6.2} {:>5.1}% {:>6.3} {:>+7.3} {:>+7.3} {:>8.3} {:>8.3} {:>8.3}",
            name,
            s.mean,
            s.p05,
            s.mean_fee,
            s.sw_frac * 100.0,
            s.avg_dfe,
            d_og,
            d_vfa,
            s.r_regime[0],
            s.r_regime[1],
            s.r_regime[2]
        );
        results.push((name.to_string(), s, d_og, d_vfa));
    }

    println!("{sep}");
    println!(
        "{:<18} {:>7.3} {:>7} {:>6.2} {:>6} {:>6} {:>7} {:>7} {:>8.3} {:>8.3} {:>8.3}  ← baseline",
        "oracle_gap",
        og_s.mean,
        "─",
        og_s.mean_fee,
        "─",
        "─",
        "+0.000",
        "─",
        og_s.r_regime[0],
        og_s.r_regime[1],
        og_s.r_regime[2]
    );
    println!(
        "{:<18} {:>7.3} {:>7} {:>6.2} {:>6} {:>6} {:>+7.3} {:>7} {:>8.3} {:>8.3} {:>8.3}  ← best heuristic",
        "vol_flow_aware",
        vfa_s.mean,
        "─",
        vfa_s.mean_fee,
        "─",
        "─",
        vfa_s.mean - og_s.mean,
        "─",
        vfa_s.r_regime[0],
        vfa_s.r_regime[1],
        vfa_s.r_regime[2]
    );
    println!("{sep}");

    // Marginal gain analysis: A→C (adding flow), B→D (adding flow to gap+vol)
    println!("\n  Marginal gains:");
    let get = |name: &str| {
        results
            .iter()
            .find(|(n, _, _, _)| n == name)
            .map(|(_, s, _, _)| s.mean)
            .unwrap_or(0.0)
    };
    let ga = get("A:gap");
    let gc = get("C:gap+flow");
    let gd = get("D:gap+flow+vol");
    let ge = get("E:+volume");
    let gf = get("F:+persistence");
    let gg = get("G:+prev_fee");
    println!(
        "  A→C  adding flow to gap-only:          {:>+.3}  (largest expected gain)",
        gc - ga
    );
    println!("  C→D  adding vol to gap+flow:            {:>+.3}", gd - gc);
    println!("  D→E  adding volume:                     {:>+.3}", ge - gd);
    println!("  E→F  adding persistence:                {:>+.3}", gf - ge);
    println!("  F→G  adding prev_fee (switch memory):   {:>+.3}", gg - gf);
    println!();
}

// ── Phase 2: Calibrated oracle grid search ────────────────────────────────────

fn phase2_calibration() {
    println!("=== Phase 2: Calibrated Oracle Gap  (grid search on calib_seed={CALIB_SEED}) ===");
    println!("Goal: best gap-only policy with (base_fee, gap_multiplier, no-change band).\n");

    let base_fees = [5.0f64, 6.0, 7.0, 8.0, 9.0, 10.0];
    let multipliers = [0.0f64, 0.01, 0.02, 0.05, 0.10];
    let bands = [0.0f64, 2.0, 5.0];

    let mut best_mean = f64::NEG_INFINITY;
    let mut best_base = 6.0f64;
    let mut best_mult = 0.0f64;
    let mut best_band = 0.0f64;

    for &base in &base_fees {
        for &mult in &multipliers {
            for &band in &bands {
                let recs = eval_policy(
                    &|gap, _, _, _, _, pf| calib_og(gap, pf, base, mult, band),
                    N_CALIB,
                    CALIB_SEED,
                    SWITCH_LAMBDA_BASE,
                );
                let m = mean_reward(&recs);
                if m > best_mean {
                    best_mean = m;
                    best_base = base;
                    best_mult = mult;
                    best_band = band;
                }
            }
        }
    }

    println!(
        "  Best params (calib seed={CALIB_SEED}): base={best_base:.0} bps  multiplier={best_mult:.2}  band={best_band:.0} bps"
    );
    println!("  Calib mean reward: {best_mean:.4}");
    println!("  Fee mapping:");
    for (gap_b, rep) in [(0u8, 2.0f64), (1, 6.0), (2, 15.0)] {
        let target = nearest_action((best_base + best_mult * rep).clamp(1.0, 30.0));
        println!(
            "    gap={gap_b} (rep≈{rep:.0} bps) → {target:.0} bps  (no-change band: {best_band:.0} bps)"
        );
    }

    // Evaluate all policies on held-out eval seed
    let og_recs = eval_policy(&oracle_gap_fee, N_EVAL, EVAL_SEED, SWITCH_LAMBDA_BASE);
    let calib_recs = eval_policy(
        &|gap, _, _, _, _, pf| calib_og(gap, pf, best_base, best_mult, best_band),
        N_EVAL,
        EVAL_SEED,
        SWITCH_LAMBDA_BASE,
    );
    let vfa_recs = eval_policy(&vol_flow_aware_fee, N_EVAL, EVAL_SEED, SWITCH_LAMBDA_BASE);
    let pa_recs = eval_policy(
        &persistence_aware_fee,
        N_EVAL,
        EVAL_SEED,
        SWITCH_LAMBDA_BASE,
    );

    let og_s = stats(&og_recs);
    let calib_s = stats(&calib_recs);
    let vfa_s = stats(&vfa_recs);
    let pa_s = stats(&pa_recs);

    let sep = "─".repeat(92);
    println!("\n  Eval results ({N_EVAL} steps, seed={EVAL_SEED}, λ={SWITCH_LAMBDA_BASE}):");
    println!("{sep}");
    println!(
        "{:<20} {:>8} {:>8} {:>8} {:>7} {:>7} {:>8} {:>8} {:>8}",
        "policy", "mean", "p05", "fee", "sw%", "Δfee", "Normal", "Toxic", "HiVol"
    );
    println!("{sep}");
    for (name, s) in [
        ("oracle_gap", &og_s),
        ("calibrated_og", &calib_s),
        ("vol_flow_aware", &vfa_s),
        ("persistence_aware", &pa_s),
    ] {
        println!(
            "{:<20} {:>8.3} {:>8.3} {:>8.2} {:>6.1}% {:>7.3} {:>8.3} {:>8.3} {:>8.3}",
            name,
            s.mean,
            s.p05,
            s.mean_fee,
            s.sw_frac * 100.0,
            s.avg_dfe,
            s.r_regime[0],
            s.r_regime[1],
            s.r_regime[2]
        );
    }
    println!("{sep}");
    println!(
        "  calibrated_og vs oracle_gap:     Δ = {:>+.3}  beat={:.1}%",
        paired_delta(&calib_recs, &og_recs),
        beat_rate(&calib_recs, &og_recs)
    );
    println!(
        "  calibrated_og vs vol_flow_aware: Δ = {:>+.3}",
        paired_delta(&calib_recs, &vfa_recs)
    );
    println!(
        "  calibrated_og vs pers_aware:     Δ = {:>+.3}",
        paired_delta(&calib_recs, &pa_recs)
    );
    println!();
}

// ── Phase 3: Switching-cost sensitivity ───────────────────────────────────────

fn phase3_switch_cost() {
    println!(
        "=== Phase 3: Switching-Cost Sensitivity  ({N_TRAIN} train / {N_EVAL} eval, full G state) ==="
    );
    println!("λ tested: {SWITCH_LAMBDAS:?}\n");

    let mask_g = [true; 6];
    let sep = "─".repeat(100);
    let sep2 = "─".repeat(60);

    // Compact summary across lambdas
    println!("{sep}");
    println!(
        "{:>6} {:<20} {:>8} {:>8} {:>8} {:>7} {:>7} {:>8} {:>8} {:>8}",
        "λ", "policy", "mean", "p05", "fee", "sw%", "Δfee", "Normal", "Toxic", "HiVol"
    );
    println!("{sep}");

    let heuristics: &[(&str, &dyn Fn(u8, u8, u8, u8, u8, f64) -> f64)] = &[
        ("oracle_gap", &oracle_gap_fee),
        ("flow_aware", &flow_aware_fee),
        ("vol_flow_aware", &vol_flow_aware_fee),
        ("persistence_aware", &persistence_aware_fee),
    ];

    let mut sweep_summary: Vec<(f64, f64, f64, f64, f64, f64)> = Vec::new(); // (lam, q_mean, og_mean, d_og, beat, sw_frac)

    for &lam in &SWITCH_LAMBDAS {
        // Evaluate heuristics
        let og_recs = eval_policy(&oracle_gap_fee, N_EVAL, EVAL_SEED, lam);
        for (name, policy) in heuristics {
            let recs = eval_policy(*policy, N_EVAL, EVAL_SEED, lam);
            let s = stats(&recs);
            println!(
                "{:>6.1} {:<20} {:>8.3} {:>8.3} {:>8.2} {:>6.1}% {:>7.3} {:>8.3} {:>8.3} {:>8.3}",
                lam,
                name,
                s.mean,
                s.p05,
                s.mean_fee,
                s.sw_frac * 100.0,
                s.avg_dfe,
                s.r_regime[0],
                s.r_regime[1],
                s.r_regime[2]
            );
        }

        // Train and eval Q-learner
        let learner = train_flex(mask_g, N_TRAIN, lam, TRAIN_SEED);
        let q_recs = eval_flex_q(&learner, N_EVAL, EVAL_SEED, lam);
        let s = stats(&q_recs);
        let d_og = paired_delta(&q_recs, &og_recs);
        let beat = beat_rate(&q_recs, &og_recs);
        let og_mean = mean_reward(&og_recs);
        println!(
            "{:>6.1} {:<20} {:>8.3} {:>8.3} {:>8.2} {:>6.1}% {:>7.3} {:>8.3} {:>8.3} {:>8.3}  Δ_og={:>+.3} beat={:.0}%",
            lam,
            "q_learner(G)",
            s.mean,
            s.p05,
            s.mean_fee,
            s.sw_frac * 100.0,
            s.avg_dfe,
            s.r_regime[0],
            s.r_regime[1],
            s.r_regime[2],
            d_og,
            beat
        );
        sweep_summary.push((lam, s.mean, og_mean, d_og, beat, s.sw_frac));
        println!("{sep}");

        // Key-state policy table for this lambda
        println!("  Policy table (Q-learner at λ={lam:.1})  avg fee over gap×prev_fee:");
        println!("  {sep2}");
        println!("  {:>6} {:>4} {:>5}  →  fee bps", "flow", "vol", "pers");
        for (flow, fl_l) in [(0u8, "fund"), (2u8, "arb")] {
            for (vol, v_l) in [(0u8, "lo"), (1u8, "mid"), (2u8, "hi")] {
                for (pers, p_l) in [(0u8, "new"), (1u8, "ong"), (2u8, "sus")] {
                    let mut fee_sum = 0.0f64;
                    let mut cnt = 0usize;
                    for gap in 0u8..3 {
                        for pfi in 0u8..9 {
                            let key = learner.project(gap, vol, flow, 1, pers, pfi);
                            fee_sum += ACTIONS[learner.best_action(&key)];
                            cnt += 1;
                        }
                    }
                    let avg = fee_sum / cnt as f64;
                    println!("    flow={fl_l} vol={v_l} pers={p_l}  →  {avg:.1} bps");
                }
            }
        }
        println!("  {sep2}\n");
    }

    // Summary table: how Q-learner's behaviour changes with lambda
    println!("  Sweep summary (Q-learner only):");
    println!("  {sep2}");
    println!(
        "  {:>6} {:>9} {:>9} {:>8} {:>7} {:>7}",
        "λ", "q_mean", "og_mean", "Δ_og", "beat%", "sw%"
    );
    println!("  {sep2}");
    for (lam, qm, ogm, d, b, sw) in &sweep_summary {
        println!(
            "  {:>6.1} {:>9.3} {:>9.3} {:>+8.3} {:>6.1}% {:>6.1}%",
            lam,
            qm,
            ogm,
            d,
            b,
            sw * 100.0
        );
    }
    println!("  {sep2}");
    println!();
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    println!("Campbell Contextual Fee-Control — v2.1 Diagnostics");
    println!("State dims: gap / vol / flow / volume / persistence / prev_fee");
    println!("Actions:    {:?} bps", ACTIONS);
    println!("Regime:     Markov  Normal(60%) Toxic(30%) HiVolTox(10%)");
    println!("Reward:     fee_revenue − toxic_loss − λ×|Δfee|/10");
    println!("{}", "═".repeat(80));
    println!();

    phase1_ablation();
    phase2_calibration();
    phase3_switch_cost();
}
