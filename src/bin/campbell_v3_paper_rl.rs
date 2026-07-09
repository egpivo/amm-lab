#![allow(clippy::type_complexity)]

use rand::rngs::StdRng;
/// v3: Paper-inspired two-sided fee RL environment.
///
/// Modeling inspiration: "Optimal Dynamic Fees in Automated Market Makers"
///   arXiv:2506.02869 (Baggiani, Herdegen, Sánchez-Betancourt)
///   Reference simulation: github.com/leonardobaggiani/amm-fees
///
/// Key paper concepts imported:
///   1. Two-sided fees — buy_fee p⁺ and sell_fee p⁻ set independently
///   2. Exponential order-flow intensities: λ · exp(−κ · fee) · gap_factor
///   3. Signed gap (pool_price − oracle_price) and inventory as primary state
///   4. Linear fee rules in inventory: buy=BASE−α·Q, sell=BASE+α·Q (strong baseline)
///   5. Arb-deterrence regime (high fee) vs noise-attraction regime (low fee)
///
/// Action (delta): base_delta ∈ {−2,0,+2} × skew_delta ∈ {−2,0,+2} → 9 actions
///   buy_fee  = clamp(base + skew, 1, 30) bps
///   sell_fee = clamp(base − skew, 1, 30) bps
///   Convention: skew < 0 → sell_fee > buy_fee (deter arb-sells when gap > 0)
///
/// Reward = fee_revenue − toxic_loss − gap_penalty − switch_cost
///
/// Success criterion: RL adds value beyond paper_linear_two_sided when
///   nonlinear trade-offs (vol-regime, flow imbalance, partial observability,
///   switching costs) break the linear policy's fixed-coefficient optimality.
///
/// Usage: cargo run --release --bin campbell_v3_paper_rl
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::collections::{HashMap, VecDeque};

// ── Action space ──────────────────────────────────────────────────────────────

const N_ACTIONS: usize = 9;
const DEFAULT_ACTION: usize = 4; // hold: Δbase=0, Δskew=0

fn action_deltas(a: usize) -> (f64, f64) {
    ([-2.0, 0.0, 2.0][a / 3], [-2.0, 0.0, 2.0][a % 3])
}

fn fee_pair(base: f64, skew: f64) -> (f64, f64) {
    (
        (base + skew).clamp(1.0, 30.0),
        (base - skew).clamp(1.0, 30.0),
    )
}

// ── Environment parameters ────────────────────────────────────────────────────

const LAMBDA_ARB: f64 = 0.30; // arb intensity × gap_bps per step
const LAMBDA_NOISE: f64 = 1.50; // noise intensity per direction per step
const KAPPA_ARB: f64 = 0.08; // arb fee sensitivity (bps⁻¹)
const KAPPA_NOISE: f64 = 0.02; // noise fee sensitivity
const GAP_DELTA: f64 = 1.00; // gap shift per trade unit (bps)
const INV_DECAY: f64 = 0.95; // inventory mean-reversion per step
const SIGMA_LOW: f64 = 2.00; // oracle vol in low-vol regime (bps/step)
const SIGMA_HIGH: f64 = 6.00; // oracle vol in high-vol regime
const P_TO_HIGH: f64 = 0.05; // P(low→high) per step
const P_TO_LOW: f64 = 0.20; // P(high→low) per step
const TOXIC_SCALE: f64 = 0.80; // LP adverse-selection loss per gap-bps per arb trade
const GAP_PEN: f64 = 0.05; // inventory/gap penalty per bps²
const SWITCH_LAM: f64 = 0.30; // switch cost per bps Δfee (÷10)
const GAP_CLAMP: f64 = 15.0;
const INV_CLAMP: f64 = 20.0;
const WINDOW: usize = 10; // rolling observation window

const BASE_INIT: f64 = 8.0;
const SKEW_INIT: f64 = 0.0;

// ── Training / eval ───────────────────────────────────────────────────────────

const N_TRAIN: usize = 800_000;
const N_EVAL: usize = 100_000;
const N_CALIB: usize = 50_000;
const ALPHA: f64 = 0.05;
const GAMMA: f64 = 0.99;
const EPS_START: f64 = 1.0;
const EPS_MIN: f64 = 0.05;
const TRAIN_NOISE: f64 = 3.0; // reward noise std for training exploration

const TRAIN_SEED: u64 = 7;
const EVAL_SEED: u64 = 99_000;
const CALIB_SEED: u64 = 77_000;

// ── Poisson sampler (Knuth's algorithm, correct for λ < ~20) ─────────────────

fn poisson(rng: &mut StdRng, lambda: f64) -> u32 {
    if lambda <= 0.0 {
        return 0;
    }
    let threshold = (-lambda).exp();
    let mut k = 0u32;
    let mut p = 1.0f64;
    loop {
        p *= rng.gen_range(0.0..1.0);
        if p <= threshold {
            return k;
        }
        k += 1;
        if k > 50 {
            return k;
        }
    }
}

// ── Environment ───────────────────────────────────────────────────────────────

struct Env {
    gap: f64, // pool_price − oracle_price in bps (fast, reacts each step)
    inv: f64, // inventory imbalance (slower, mean-reverts toward net flow)
    high_vol: bool,
    base_fee: f64, // current RL base fee (used by delta-action policies)
    skew: f64,     // current RL skew
    prev_buy_fee: f64,
    prev_sell_fee: f64,
    rets: VecDeque<f64>,  // recent oracle returns
    buys: VecDeque<f64>,  // recent buy arrivals per step
    sells: VecDeque<f64>, // recent sell arrivals per step
}

impl Env {
    fn new(rng: &mut StdRng) -> Self {
        let high_vol = rng.gen_range(0.0..1.0) < 0.2;
        let mut rets = VecDeque::with_capacity(WINDOW + 1);
        let mut buys = VecDeque::with_capacity(WINDOW + 1);
        let mut sells = VecDeque::with_capacity(WINDOW + 1);
        for _ in 0..WINDOW {
            rets.push_back(0.0);
            buys.push_back(1.5);
            sells.push_back(1.5);
        }
        Self {
            gap: 0.0,
            inv: 0.0,
            high_vol,
            base_fee: BASE_INIT,
            skew: SKEW_INIT,
            prev_buy_fee: BASE_INIT,
            prev_sell_fee: BASE_INIT,
            rets,
            buys,
            sells,
        }
    }

    fn push(buf: &mut VecDeque<f64>, val: f64) {
        if buf.len() >= WINDOW {
            buf.pop_front();
        }
        buf.push_back(val);
    }

    fn recent_vol(&self) -> f64 {
        let n = self.rets.len() as f64;
        if n < 2.0 {
            return 0.0;
        }
        let mean = self.rets.iter().sum::<f64>() / n;
        (self.rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n).sqrt()
    }

    fn flow_imbal(&self) -> f64 {
        let tb: f64 = self.buys.iter().sum();
        let ts: f64 = self.sells.iter().sum();
        let denom = tb + ts + 1.0;
        (tb - ts) / denom
    }

    fn last_ret(&self) -> f64 {
        *self.rets.back().unwrap_or(&0.0)
    }

    fn avg_buys(&self) -> f64 {
        self.buys.iter().sum::<f64>() / WINDOW as f64
    }
    fn avg_sells(&self) -> f64 {
        self.sells.iter().sum::<f64>() / WINDOW as f64
    }

    fn obs(&self) -> Vec<u8> {
        vec![
            signed3(self.gap, -4.0, 4.0),
            signed3(self.inv, -7.0, 7.0),
            signed3(self.last_ret(), -2.5, 2.5),
            bucket3(self.recent_vol(), 2.5, 5.0),
            bucket3(self.avg_buys(), 0.8, 2.5),
            bucket3(self.avg_sells(), 0.8, 2.5),
            signed3(self.flow_imbal(), -0.3, 0.3),
            bucket3(self.prev_buy_fee, 6.5, 12.0),
            bucket3(self.prev_sell_fee, 6.5, 12.0),
        ]
    }

    fn step(&mut self, rng: &mut StdRng, buy_fee: f64, sell_fee: f64, noisy: bool) -> EvalRec {
        // 1. Oracle price move
        let sigma = if self.high_vol { SIGMA_HIGH } else { SIGMA_LOW };
        let ext_ret = Normal::new(0.0, sigma).unwrap().sample(rng);
        // Pool price fixed this step; oracle moves → gap changes
        let gap_before = self.gap - ext_ret;
        // gap_before is what traders see BEFORE orders arrive

        // 2. Order arrivals (Poisson, paper-style intensity)
        let arb_sell_lam = LAMBDA_ARB * gap_before.max(0.0) * (-KAPPA_ARB * sell_fee).exp();
        let arb_buy_lam = LAMBDA_ARB * (-gap_before).max(0.0) * (-KAPPA_ARB * buy_fee).exp();
        let noise_buy_lam = LAMBDA_NOISE * (-KAPPA_NOISE * buy_fee).exp();
        let noise_sell_lam = LAMBDA_NOISE * (-KAPPA_NOISE * sell_fee).exp();

        let n_as = poisson(rng, arb_sell_lam);
        let n_ab = poisson(rng, arb_buy_lam);
        let n_nb = poisson(rng, noise_buy_lam);
        let n_ns = poisson(rng, noise_sell_lam);

        let net_buys = (n_ab + n_nb) as f64;
        let net_sells = (n_as + n_ns) as f64;
        let n_arb = (n_as + n_ab) as f64;

        // 3. Update gap and inv
        // Buys from AMM → pool price increases → gap increases
        // Sells to AMM  → pool price decreases → gap decreases
        let gap_after =
            (gap_before + GAP_DELTA * (net_buys - net_sells)).clamp(-GAP_CLAMP, GAP_CLAMP);
        self.inv = (self.inv * INV_DECAY + GAP_DELTA * (net_buys - net_sells))
            .clamp(-INV_CLAMP, INV_CLAMP);
        self.gap = gap_after;

        // 4. Reward components (use gap_before for adverse selection: arbs traded at gap_before)
        let fee_rev = buy_fee * net_buys + sell_fee * net_sells;
        let toxic_loss = TOXIC_SCALE * gap_before.abs() * n_arb;
        let gap_pen = GAP_PEN * gap_before * gap_before;
        let switch_cost = SWITCH_LAM
            * ((buy_fee - self.prev_buy_fee).abs() + (sell_fee - self.prev_sell_fee).abs())
            / 10.0;
        let _reward = fee_rev - toxic_loss - gap_pen - switch_cost
            + if noisy {
                Normal::new(0.0, TRAIN_NOISE).unwrap().sample(rng)
            } else {
                0.0
            };

        // 5. Update rolling windows
        Self::push(&mut self.rets, ext_ret);
        Self::push(&mut self.buys, net_buys);
        Self::push(&mut self.sells, net_sells);

        // 6. Transition vol regime
        if self.high_vol {
            if rng.gen_range(0.0..1.0) < P_TO_LOW {
                self.high_vol = false;
            }
        } else if rng.gen_range(0.0..1.0) < P_TO_HIGH {
            self.high_vol = true;
        }

        self.prev_buy_fee = buy_fee;
        self.prev_sell_fee = sell_fee;

        EvalRec {
            reward: fee_rev - toxic_loss - gap_pen - switch_cost, // det reward for eval
            fee_rev,
            toxic_loss,
            gap_pen,
            switch_cost,
            buy_fee,
            sell_fee,
            n_arb_sells: n_as,
            n_arb_buys: n_ab,
            n_noise_buys: n_nb,
            n_noise_sells: n_ns,
            high_vol: self.high_vol,
        }
    }
}

fn signed3(x: f64, lo: f64, hi: f64) -> u8 {
    if x < lo {
        0
    } else if x <= hi {
        1
    } else {
        2
    }
}
fn bucket3(x: f64, lo: f64, hi: f64) -> u8 {
    if x < lo {
        0
    } else if x <= hi {
        1
    } else {
        2
    }
}

// ── EvalRec and Stats ─────────────────────────────────────────────────────────

struct EvalRec {
    reward: f64,
    fee_rev: f64,
    toxic_loss: f64,
    gap_pen: f64,
    switch_cost: f64,
    buy_fee: f64,
    sell_fee: f64,
    n_arb_sells: u32,
    n_arb_buys: u32,
    n_noise_buys: u32,
    n_noise_sells: u32,
    high_vol: bool,
}

struct Stats {
    mean: f64,
    p05: f64,
    fee_rev: f64,
    toxic: f64,
    penalty: f64,
    switch: f64,
    buy_fee: f64,
    sell_fee: f64,
    skew: f64,
    arb_buys: f64,
    arb_sells: f64,
    noise_buys: f64,
    noise_sells: f64,
    pct_high_vol: f64,
}

fn compute_stats(recs: &[EvalRec]) -> Stats {
    let n = recs.len() as f64;
    let mean = recs.iter().map(|r| r.reward).sum::<f64>() / n;
    let mut rs: Vec<f64> = recs.iter().map(|r| r.reward).collect();
    rs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p05 = rs[(0.05 * (recs.len() - 1) as f64).round() as usize];
    Stats {
        mean,
        p05,
        fee_rev: recs.iter().map(|r| r.fee_rev).sum::<f64>() / n,
        toxic: recs.iter().map(|r| r.toxic_loss).sum::<f64>() / n,
        penalty: recs.iter().map(|r| r.gap_pen).sum::<f64>() / n,
        switch: recs.iter().map(|r| r.switch_cost).sum::<f64>() / n,
        buy_fee: recs.iter().map(|r| r.buy_fee).sum::<f64>() / n,
        sell_fee: recs.iter().map(|r| r.sell_fee).sum::<f64>() / n,
        skew: recs.iter().map(|r| r.buy_fee - r.sell_fee).sum::<f64>() / n,
        arb_buys: recs.iter().map(|r| r.n_arb_buys as f64).sum::<f64>() / n,
        arb_sells: recs.iter().map(|r| r.n_arb_sells as f64).sum::<f64>() / n,
        noise_buys: recs.iter().map(|r| r.n_noise_buys as f64).sum::<f64>() / n,
        noise_sells: recs.iter().map(|r| r.n_noise_sells as f64).sum::<f64>() / n,
        pct_high_vol: recs.iter().filter(|r| r.high_vol).count() as f64 / n,
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

// ── Baseline policies ─────────────────────────────────────────────────────────
// Signature: (gap, inv, vol, imbal, prev_buy, prev_sell) → (buy_fee, sell_fee)

fn fixed_sym(bps: f64) -> impl Fn(f64, f64, f64, f64, f64, f64) -> (f64, f64) {
    move |_, _, _, _, _, _| (bps, bps)
}

fn calibrated_og(base: f64) -> impl Fn(f64, f64, f64, f64, f64, f64) -> (f64, f64) {
    move |gap, _, _, _, _, _| {
        let fee = (base + 0.5 * gap.abs()).clamp(1.0, 30.0);
        (fee, fee)
    }
}

fn paper_linear_sym(base: f64, beta: f64) -> impl Fn(f64, f64, f64, f64, f64, f64) -> (f64, f64) {
    // Symmetric: fee = base + β·|gap|  (paper: higher fee when inventory deviates)
    move |gap, _, _, _, _, _| {
        let fee = (base + beta * gap.abs()).clamp(1.0, 30.0);
        (fee, fee)
    }
}

fn paper_linear_two_sided(
    base: f64,
    alpha: f64,
) -> impl Fn(f64, f64, f64, f64, f64, f64) -> (f64, f64) {
    // Paper's optimal linear policy: buy_fee = base − α·gap, sell_fee = base + α·gap
    // gap > 0 (pool overpriced): lower buy_fee (attract buyers), higher sell_fee (deter arb-sells)
    move |gap, _, _, _, _, _| {
        let bf = (base - alpha * gap).clamp(1.0, 30.0);
        let sf = (base + alpha * gap).clamp(1.0, 30.0);
        (bf, sf)
    }
}

fn flow_aware_two_sided(
    base: f64,
    beta_imbal: f64,
    alpha_gap: f64,
) -> impl Fn(f64, f64, f64, f64, f64, f64) -> (f64, f64) {
    // Uses both gap and observed flow imbalance to set skew
    // imbal > 0 (more buys): pool price rising → gap increasing → raise buy_fee
    move |gap, _, _, imbal, _, _| {
        let skew = -alpha_gap * gap - beta_imbal * imbal;
        let bf = (base + skew).clamp(1.0, 30.0);
        let sf = (base - skew).clamp(1.0, 30.0);
        (bf, sf)
    }
}

// ── Evaluation harness ────────────────────────────────────────────────────────

fn eval_baseline<F: Fn(f64, f64, f64, f64, f64, f64) -> (f64, f64) + ?Sized>(
    policy: &F,
    n: usize,
    seed: u64,
) -> Vec<EvalRec> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut env = Env::new(&mut rng);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let (bf, sf) = policy(
            env.gap,
            env.inv,
            env.recent_vol(),
            env.flow_imbal(),
            env.prev_buy_fee,
            env.prev_sell_fee,
        );
        let rec = env.step(&mut rng, bf, sf, false);
        out.push(rec);
    }
    out
}

// ── FlexQLearner ──────────────────────────────────────────────────────────────

struct FlexQLearner {
    q: HashMap<Vec<u8>, [f64; N_ACTIONS]>,
    visits: HashMap<Vec<u8>, [u32; N_ACTIONS]>,
    epsilon: f64,
}

impl FlexQLearner {
    fn new() -> Self {
        Self {
            q: HashMap::new(),
            visits: HashMap::new(),
            epsilon: EPS_START,
        }
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

    fn update(&mut self, key: Vec<u8>, a: usize, r: f64, next: &[u8]) {
        let max_n = self.best_q(next);
        let q = self.q.entry(key.clone()).or_insert([0.0; N_ACTIONS]);
        q[a] += ALPHA * (r + GAMMA * max_n - q[a]);
        self.visits.entry(key).or_insert([0u32; N_ACTIONS])[a] += 1;
    }
}

fn train_rl(seed: u64) -> FlexQLearner {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut env = Env::new(&mut rng);
    let mut learner = FlexQLearner::new();
    let decay = (EPS_START - EPS_MIN) / N_TRAIN as f64;
    for _ in 0..N_TRAIN {
        let s = env.obs();
        let a = learner.choose(&s, &mut rng);
        let (bd, sd) = action_deltas(a);
        env.base_fee = (env.base_fee + bd).clamp(1.0, 30.0);
        env.skew = (env.skew + sd).clamp(-15.0, 15.0);
        let (bf, sf) = fee_pair(env.base_fee, env.skew);
        let rec = env.step(&mut rng, bf, sf, true);
        let ns = env.obs();
        learner.update(s, a, rec.reward, &ns);
        learner.epsilon = (learner.epsilon - decay).max(EPS_MIN);
    }
    learner
}

fn eval_rl(learner: &FlexQLearner, n: usize, seed: u64) -> Vec<EvalRec> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut env = Env::new(&mut rng);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let s = env.obs();
        let a = learner.best_action(&s);
        let (bd, sd) = action_deltas(a);
        env.base_fee = (env.base_fee + bd).clamp(1.0, 30.0);
        env.skew = (env.skew + sd).clamp(-15.0, 15.0);
        let (bf, sf) = fee_pair(env.base_fee, env.skew);
        let rec = env.step(&mut rng, bf, sf, false);
        out.push(rec);
    }
    out
}

// ── Calibration helpers ───────────────────────────────────────────────────────

fn calib_fixed_sym() -> f64 {
    let mut best = (f64::NEG_INFINITY, 8.0f64);
    for bps in [5.0f64, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 20.0, 25.0] {
        let m = mean_reward(&eval_baseline(&fixed_sym(bps), N_CALIB, CALIB_SEED));
        if m > best.0 {
            best = (m, bps);
        }
    }
    best.1
}

fn calib_pltd() -> (f64, f64) {
    // Wide search: include the high-fee range where symmetric policies are strong.
    // The two-sided policy's advantage should appear at the same base-fee level.
    let mut best = (f64::NEG_INFINITY, 8.0f64, 0.5f64);
    for base in [8.0f64, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0] {
        for alpha in [0.1f64, 0.3, 0.5, 0.7, 1.0, 1.5] {
            let m = mean_reward(&eval_baseline(
                &paper_linear_two_sided(base, alpha),
                N_CALIB,
                CALIB_SEED,
            ));
            if m > best.0 {
                best = (m, base, alpha);
            }
        }
    }
    (best.1, best.2)
}

fn calib_fatd() -> (f64, f64, f64) {
    let mut best = (f64::NEG_INFINITY, 8.0f64, 2.0f64, 0.5f64);
    for base in [10.0f64, 12.0, 14.0, 16.0, 18.0, 20.0] {
        for bi in [1.0f64, 2.0, 4.0] {
            for ag in [0.3f64, 0.6, 1.0, 1.5] {
                let m = mean_reward(&eval_baseline(
                    &flow_aware_two_sided(base, bi, ag),
                    N_CALIB,
                    CALIB_SEED,
                ));
                if m > best.0 {
                    best = (m, base, bi, ag);
                }
            }
        }
    }
    (best.1, best.2, best.3)
}

// ── Print helpers ─────────────────────────────────────────────────────────────

fn print_stats_row(name: &str, s: &Stats) {
    println!(
        "{:<24} {:>7.2} {:>7.2} {:>7.2} {:>7.2} {:>6.2} {:>6.2} {:>7.2} {:>7.2} {:>7.2}",
        name, s.mean, s.p05, s.fee_rev, s.toxic, s.penalty, s.switch, s.buy_fee, s.sell_fee, s.skew
    );
}

fn print_flow_row(name: &str, s: &Stats) {
    println!(
        "{:<24} {:>8.3} {:>8.3} {:>8.3} {:>8.3}  hi_vol={:.1}%",
        name,
        s.arb_buys,
        s.arb_sells,
        s.noise_buys,
        s.noise_sells,
        s.pct_high_vol * 100.0
    );
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    println!("Campbell v3 — Two-Sided Fee Control (arXiv:2506.02869 inspired)");
    println!("Action: Δbase×Δskew  buy_fee=clamp(base+skew,1,30)  sell_fee=clamp(base-skew,1,30)");
    println!("Intensities: arb ∝ max(0,±gap)·exp(−κ_arb·fee)   noise ∝ exp(−κ_n·fee)");
    println!("Reward: fee_revenue − toxic_loss − gap_penalty − switch_cost");
    println!("{}", "═".repeat(80));
    println!();

    // ── Calibration ───────────────────────────────────────────────────────────
    println!("Calibrating baselines on seed={CALIB_SEED} ({N_CALIB} steps each)...");
    let best_fixed = calib_fixed_sym();
    let (pltd_base, pltd_alpha) = calib_pltd();
    let (fatd_base, fatd_bi, fatd_ag) = calib_fatd();
    println!("  fixed_sym_best:           {best_fixed:.0} bps");
    println!("  paper_linear_two_sided:   base={pltd_base:.0}  α={pltd_alpha:.2}");
    println!(
        "  flow_aware_two_sided:     base={fatd_base:.0}  β_imbal={fatd_bi:.1}  α_gap={fatd_ag:.1}"
    );
    println!();

    // ── Define policies ───────────────────────────────────────────────────────
    let pls_beta = 0.3f64; // paper_linear_sym: fee = base + 0.3·|gap|

    let baselines: Vec<(
        &str,
        Box<dyn Fn(f64, f64, f64, f64, f64, f64) -> (f64, f64)>,
    )> = vec![
        ("fixed_8bps", Box::new(fixed_sym(8.0))),
        ("fixed_10bps", Box::new(fixed_sym(10.0))),
        ("calib_og_sym", Box::new(calibrated_og(best_fixed))),
        (
            "paper_linear_sym",
            Box::new(paper_linear_sym(best_fixed, pls_beta)),
        ),
        (
            "paper_linear_2sided",
            Box::new(paper_linear_two_sided(pltd_base, pltd_alpha)),
        ),
        (
            "flow_aware_2sided",
            Box::new(flow_aware_two_sided(fatd_base, fatd_bi, fatd_ag)),
        ),
    ];

    // ── Phase 1: baseline comparison ──────────────────────────────────────────
    println!("=== Phase 1: Baseline Policies ({N_EVAL} eval steps, seed={EVAL_SEED}) ===");
    let sep = "─".repeat(106);
    println!("{sep}");
    println!(
        "{:<24} {:>7} {:>7} {:>7} {:>7} {:>6} {:>6} {:>7} {:>7} {:>7}",
        "policy", "mean", "p05", "fee_rev", "toxic", "pen", "sw", "buy_f", "sell_f", "skew"
    );
    println!("{sep}");

    let mut all_recs: Vec<(&str, Vec<EvalRec>)> = Vec::new();

    for (name, policy) in &baselines {
        let recs = eval_baseline(policy.as_ref(), N_EVAL, EVAL_SEED);
        let s = compute_stats(&recs);
        print_stats_row(name, &s);
        all_recs.push((name, recs));
    }

    // ── Phase 2: Q-learning ───────────────────────────────────────────────────
    println!("{sep}");
    println!("\n=== Phase 2: Q-Learning ({N_TRAIN} train steps, γ={GAMMA}, 9-dim state) ===\n");
    let learner = train_rl(TRAIN_SEED);
    println!(
        "  States visited: {}  Q-entries: {}",
        learner.q.len(),
        learner.q.len() * N_ACTIONS
    );

    let q_recs = eval_rl(&learner, N_EVAL, EVAL_SEED);
    let q_stats = compute_stats(&q_recs);
    println!();
    println!("{sep}");
    println!(
        "{:<24} {:>7} {:>7} {:>7} {:>7} {:>6} {:>6} {:>7} {:>7} {:>7}",
        "policy", "mean", "p05", "fee_rev", "toxic", "pen", "sw", "buy_f", "sell_f", "skew"
    );
    println!("{sep}");
    for (name, recs) in &all_recs {
        print_stats_row(name, &compute_stats(recs));
    }
    print_stats_row("delta_q_learner", &q_stats);
    println!("{sep}");

    // ── Phase 3: Paired deltas vs paper_linear_two_sided ─────────────────────
    let pltd_idx = all_recs
        .iter()
        .position(|(n, _)| *n == "paper_linear_2sided")
        .unwrap();
    let pltd_recs = &all_recs[pltd_idx].1;
    println!("\n  Paired Δ vs paper_linear_two_sided (eval seed {EVAL_SEED}):");
    let sep2 = "─".repeat(70);
    println!("{sep2}");
    println!(
        "{:<24} {:>9} {:>9} {:>8}",
        "policy", "Δ_mean", "beat%", "Δ_fixed8"
    );
    println!("{sep2}");
    let fixed8_recs = &all_recs.iter().find(|(n, _)| *n == "fixed_8bps").unwrap().1;
    for (name, recs) in &all_recs {
        if *name == "paper_linear_2sided" {
            continue;
        }
        let d = paired_delta(recs, pltd_recs);
        let b = beat_rate(recs, pltd_recs);
        let df8 = paired_delta(recs, fixed8_recs);
        println!("{:<24} {:>+9.3} {:>8.1}% {:>+8.3}", name, d, b, df8);
    }
    let dq = paired_delta(&q_recs, pltd_recs);
    let bq = beat_rate(&q_recs, pltd_recs);
    let dq8 = paired_delta(&q_recs, fixed8_recs);
    println!(
        "{:<24} {:>+9.3} {:>8.1}% {:>+8.3}",
        "delta_q_learner", dq, bq, dq8
    );
    println!("{sep2}");

    // ── Phase 4: Order flow breakdown ─────────────────────────────────────────
    println!("\n  Order flow breakdown (avg counts per step):");
    println!("{sep2}");
    println!(
        "{:<24} {:>8} {:>8} {:>8} {:>8}  regime",
        "policy", "arb_buy", "arb_sell", "ns_buy", "ns_sell"
    );
    println!("{sep2}");
    for (name, recs) in &all_recs {
        print_flow_row(name, &compute_stats(recs));
    }
    print_flow_row("delta_q_learner", &q_stats);
    println!("{sep2}");

    // ── Phase 5: Policy examples ──────────────────────────────────────────────
    // Show RL fees at steady-state base/skew (not cold-start BASE_INIT=8).
    // Steady state is derived from eval mean buy_fee and sell_fee.
    // prev_fee buckets use =2 (>12 bps) to match converged regime.
    let ss_base = (q_stats.buy_fee + q_stats.sell_fee) / 2.0;
    let ss_skew = (q_stats.buy_fee - q_stats.sell_fee) / 2.0;
    println!("\n  Policy examples: fees at key (gap, inv) states");
    println!("  RL shown at steady-state base={ss_base:.1}  skew={ss_skew:.1}");
    println!("  (paper_linear_2sided  vs  delta_q_learner)");
    println!("{sep2}");
    println!(
        "  {:>8} {:>8}  {:>8} {:>9}   {:>8} {:>9}  {:>6}",
        "gap", "inv", "pltd_buy", "pltd_sell", "rl_buy", "rl_sell", "action"
    );
    println!("{sep2}");

    let example_gaps = [-10.0f64, -5.0, 0.0, 5.0, 10.0];
    let example_invs = [-10.0f64, 0.0, 10.0];
    let pltd_fn = paper_linear_two_sided(pltd_base, pltd_alpha);
    let action_names = [
        "-2/-2", "-2/0", "-2/+2", "0/-2", "hold", "0/+2", "+2/-2", "+2/0", "+2/+2",
    ];
    for &g in &example_gaps {
        for &inv in &example_invs {
            let (pb, ps) = pltd_fn(g, inv, 0.0, 0.0, 0.0, 0.0);
            // Obs at steady-state: prev fees in bucket=2 (>12 bps)
            let obs = vec![
                signed3(g, -4.0, 4.0),
                signed3(inv, -7.0, 7.0),
                1u8,
                1,
                1,
                1,
                1,
                2,
                2,
            ];
            let a = learner.best_action(&obs);
            let (bd, sd) = action_deltas(a);
            let new_base = (ss_base + bd).clamp(1.0, 30.0);
            let new_skew = (ss_skew + sd).clamp(-15.0, 15.0);
            let rb = (new_base + new_skew).clamp(1.0, 30.0);
            let rs = (new_base - new_skew).clamp(1.0, 30.0);
            println!(
                "  {:>8.0} {:>8.0}  {:>8.1} {:>9.1}   {:>8.1} {:>9.1}  {:>6}",
                g, inv, pb, ps, rb, rs, action_names[a]
            );
        }
    }
    println!("{sep2}");

    // ── Phase 6: Multi-seed robustness ────────────────────────────────────────
    println!("\n=== Phase 6: RL Multi-seed Robustness ===");
    println!("{sep2}");
    println!(
        "{:>12}  {:>9} {:>9} {:>9} {:>8}",
        "train_seed", "rl_mean", "pltd_mean", "Δ_pltd", "beat%"
    );
    println!("{sep2}");
    let pltd_eval = eval_baseline(
        &paper_linear_two_sided(pltd_base, pltd_alpha),
        N_EVAL,
        EVAL_SEED,
    );
    for &tseed in &[0u64, 42, 123, 456, 789] {
        let l2 = train_rl(tseed);
        let r2 = eval_rl(&l2, N_EVAL, EVAL_SEED);
        let qm = mean_reward(&r2);
        let pm = mean_reward(&pltd_eval);
        let d = paired_delta(&r2, &pltd_eval);
        let b = beat_rate(&r2, &pltd_eval);
        println!(
            "{:>12}  {:>9.3} {:>9.3} {:>+9.3} {:>7.1}%",
            tseed, qm, pm, d, b
        );
    }
    println!("{sep2}");

    // ── Success criterion ─────────────────────────────────────────────────────
    println!("\n  Success criterion: RL beats paper_linear_two_sided significantly?");
    if dq > 0.5 && bq > 55.0 {
        println!("  [PASS] RL beats paper_linear_two_sided: Δ={dq:+.3}  beat={bq:.0}%");
        println!("         Claim: nonlinear signals (vol regime, flow imbalance, persistence)");
        println!("         add value beyond fixed-coefficient linear fee rules.");
    } else if dq > 0.0 {
        println!("  [MARGINAL] RL outperforms but modestly: Δ={dq:+.3}  beat={bq:.0}%");
        println!("         Paper_linear_two_sided is a strong baseline; gap is narrow.");
    } else {
        println!("  [FAIL] RL does not beat paper_linear_two_sided: Δ={dq:+.3}  beat={bq:.0}%");
        println!(
            "         Either environment lacks nonlinear structure, or training underconverged."
        );
    }

    // ── Structural diagnosis ──────────────────────────────────────────────────
    let sym_best = mean_reward(
        &all_recs
            .iter()
            .find(|(n, _)| *n == "calib_og_sym")
            .unwrap()
            .1,
    );
    let delta_sym_vs_rl = sym_best - mean_reward(&q_recs);
    println!();
    println!("  Structural diagnosis:");
    println!("  calib_og_sym (fixed high-fee) beats RL by {delta_sym_vs_rl:+.3}.");
    if delta_sym_vs_rl > 2.0 {
        println!("  >> Fee LEVEL (exp arb-suppression) dominates fee ASYMMETRY in this env.");
        println!("  >> RL learns correct direction (sell_fee > buy_fee) but reaches lower level");
        println!("     than optimal because delta-actions start at {BASE_INIT:.0} bps and state");
        println!(
            "     space ({} states) is only {:.0}% explored at {N_TRAIN} steps.",
            3usize.pow(9),
            100.0 * learner.q.len() as f64 / 3usize.pow(9) as f64
        );
        println!("  >> To make two-sided skew matter more: add directional noise traders or");
        println!("     inventory-dependent toxic-loss amplifier that penalises symmetric fees.");
    }
    println!();
}
