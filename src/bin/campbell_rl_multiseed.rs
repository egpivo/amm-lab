/// Multi-seed robustness diagnostic.
/// Usage: cargo run --release --bin campbell_rl_multiseed -- scenarios/campbell_rl_normal.toml
///
/// Trains 5 independent RL policies (different exploration seeds) on the same scenario.
/// Evaluates each on holdout eval seeds 5000-5499 alongside oracle_gap.
/// Reports: train_seed, RL mean_pnl, oracle_gap mean_pnl, paired delta, beat rate.
/// Purpose: determine whether the RL win is reproducible or seed-sensitive.
use amm_lab::campbell::fee_policy::{
    FeeObservation, FeePolicy, OracleGapFeePolicy, RL_ACTIONS_BPS, TabularLearnedFeePolicy,
};
use amm_lab::campbell::gbm::generate_gbm;
use amm_lab::campbell::pool::CampbellPool;
use amm_lab::campbell::simulation::{FlowRegime, SimConfig, run_simulation};
use amm_lab::campbell::trader::{arb_delta, fundamental_buy_delta, fundamental_sell_delta};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::VecDeque;
use std::io::Write;

// ── Training parameters (match campbell_rl_fee_train.rs) ─────────────────────
const TRAIN_PATHS: u64 = 5_000;
const GAMMA: f64 = 0.99;
const ALPHA: f64 = 0.05;
const EPSILON_START: f64 = 1.0;
const EPSILON_MIN: f64 = 0.05;
const EPSILON_DECAY: f64 = 0.9994;
const VOL_WINDOW: usize = 20;
const INITIAL_PRICE: f64 = 1.0;

// ── Eval parameters ───────────────────────────────────────────────────────────
const EVAL_START: u64 = 5_000;
const EVAL_PATHS: u64 = 500;

// Policy seeds to test
const POLICY_SEEDS: &[u64] = &[0, 42, 123, 456, 789];

// ── Helpers (copied from train binary) ───────────────────────────────────────

fn push_w<T>(buf: &mut VecDeque<T>, val: T, cap: usize) {
    if buf.len() == cap {
        buf.pop_front();
    }
    buf.push_back(val);
}

fn rolling_std(buf: &VecDeque<f64>) -> f64 {
    if buf.len() < 2 {
        return 0.0;
    }
    let n = buf.len() as f64;
    let mean = buf.iter().sum::<f64>() / n;
    (buf.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n).sqrt()
}

fn frac_true(buf: &VecDeque<bool>) -> f64 {
    if buf.is_empty() {
        return 0.0;
    }
    buf.iter().filter(|&&b| b).count() as f64 / buf.len() as f64
}

fn effective_fund_scale(config: &SimConfig, step: usize, rng: &mut StdRng) -> f64 {
    match &config.flow_regime {
        FlowRegime::Normal => 1.0,
        FlowRegime::ToxicBurst => {
            if rng.gen_range(0.0f64..1.0) < config.toxic_burst_prob {
                config.toxic_burst_fund_scale
            } else {
                1.0
            }
        }
        FlowRegime::RegimeSwitch => {
            let period = config.regime_switch_period.max(1);
            if (step / period).is_multiple_of(2) {
                1.0
            } else {
                config.toxic_burst_fund_scale
            }
        }
    }
}

// ── Train one policy ──────────────────────────────────────────────────────────

fn train(config: &SimConfig, policy_seed: u64) -> TabularLearnedFeePolicy {
    let dt = 1.0 / config.n_steps as f64;
    let mut policy = TabularLearnedFeePolicy::new(EPSILON_START, ALPHA, policy_seed);

    for ep in 0..TRAIN_PATHS {
        let seed = ep;
        let cex_prices = generate_gbm(
            config.n_steps,
            INITIAL_PRICE,
            config.mu,
            config.sigma,
            dt,
            seed,
        );
        let mut pool = CampbellPool::new(config.reserve_x, config.reserve_y, config.amm_fee);
        let mut hedging = pool.pool_value(cex_prices[0]);
        let mut regime_rng = StdRng::seed_from_u64(seed.wrapping_add(99_991));

        let mut log_rets: VecDeque<f64> = VecDeque::with_capacity(VOL_WINDOW + 1);
        let mut arb_flags: VecDeque<bool> = VecDeque::with_capacity(VOL_WINDOW + 1);
        let mut fund_flags: VecDeque<bool> = VecDeque::with_capacity(VOL_WINDOW + 1);
        let mut vol_sums: VecDeque<f64> = VecDeque::with_capacity(VOL_WINDOW + 1);
        let mut prev_fee = config.amm_fee;

        for i in 0..config.n_steps {
            let prev_cex = cex_prices[i];
            let cex_price = cex_prices[i + 1];

            let prev_position = hedging - pool.pool_value(prev_cex);
            hedging += pool.reserve_y * (cex_price - prev_cex);
            let fee_before = pool.cumulative_fee_revenue;

            let oracle_gap_bps = (pool.marginal_price() - cex_price) / cex_price * 10_000.0;
            let inventory_skew = (pool.reserve_x - pool.reserve_y * cex_price)
                / (pool.reserve_x + pool.reserve_y * cex_price);
            let obs = FeeObservation {
                step: i,
                external_price: cex_price,
                amm_price: pool.marginal_price(),
                oracle_gap_bps,
                inventory_skew,
                recent_vol: rolling_std(&log_rets),
                recent_arb_frac: frac_true(&arb_flags),
                recent_fund_frac: frac_true(&fund_flags),
                recent_volume: vol_sums.iter().sum(),
                previous_fee: prev_fee,
            };
            let state = policy.obs_to_state(&obs);
            let action = policy.choose_action(state);
            let fee = RL_ACTIONS_BPS[action] / 10_000.0;
            pool.amm_fee = fee;
            prev_fee = fee;

            let fund_scale = effective_fund_scale(config, i, &mut regime_rng);
            let arb_d = arb_delta(&pool, cex_price, config.cex_fee);
            pool.apply_delta(arb_d);
            let buy_d = fundamental_buy_delta(
                config.buy_demand * fund_scale,
                &pool,
                cex_price,
                config.cex_fee,
            );
            pool.apply_delta(buy_d);
            let sell_d = fundamental_sell_delta(
                -config.sell_demand * fund_scale,
                &pool,
                cex_price,
                config.cex_fee,
            );
            pool.apply_delta(sell_d);

            let step_fee = pool.cumulative_fee_revenue - fee_before;
            let cur_position = hedging - pool.pool_value(cex_price);
            let step_reward = step_fee - (cur_position - prev_position);

            push_w(&mut log_rets, (cex_price / prev_cex).ln(), VOL_WINDOW);
            push_w(&mut arb_flags, arb_d.abs() > 1e-12, VOL_WINDOW);
            push_w(
                &mut fund_flags,
                (buy_d.abs() + sell_d.abs()) > 1e-12,
                VOL_WINDOW,
            );
            push_w(
                &mut vol_sums,
                arb_d.abs() + buy_d.abs() + sell_d.abs(),
                VOL_WINDOW,
            );

            let next_obs = FeeObservation {
                step: i + 1,
                external_price: cex_price,
                amm_price: pool.marginal_price(),
                oracle_gap_bps: (pool.marginal_price() - cex_price) / cex_price * 10_000.0,
                inventory_skew: (pool.reserve_x - pool.reserve_y * cex_price)
                    / (pool.reserve_x + pool.reserve_y * cex_price),
                recent_vol: rolling_std(&log_rets),
                recent_arb_frac: frac_true(&arb_flags),
                recent_fund_frac: frac_true(&fund_flags),
                recent_volume: vol_sums.iter().sum(),
                previous_fee: fee,
            };
            let next_state = policy.obs_to_state(&next_obs);
            policy.update_step(state, action, step_reward, next_state, GAMMA);
        }
        policy.decay_epsilon(EPSILON_DECAY, EPSILON_MIN);
    }
    policy
}

// ── Eval one policy ───────────────────────────────────────────────────────────

fn eval(base_config: &SimConfig, policy: &mut dyn FeePolicy) -> Vec<f64> {
    let dt = 1.0 / base_config.n_steps as f64;
    (EVAL_START..EVAL_START + EVAL_PATHS)
        .map(|seed| {
            let mut config = base_config.clone();
            config.seed = seed;
            let cex = generate_gbm(
                config.n_steps,
                INITIAL_PRICE,
                config.mu,
                config.sigma,
                dt,
                seed,
            );
            let records = run_simulation(&config, &cex, policy);
            let last = records.last().unwrap();
            let fee_rev: f64 = records.iter().map(|r| r.step_fee).sum();
            let lvr = last.hedging_portfolio - last.pool_value;
            fee_rev - lvr
        })
        .collect()
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let toml_path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("scenarios/campbell_rl_normal.toml");
    let toml_str = std::fs::read_to_string(toml_path)
        .unwrap_or_else(|e| panic!("cannot read {toml_path}: {e}"));
    let config: SimConfig =
        toml::from_str(&toml_str).unwrap_or_else(|e| panic!("invalid TOML: {e}"));

    println!(
        "Scenario: {}  ({TRAIN_PATHS} train / {EVAL_PATHS} eval)",
        config.name
    );

    // Evaluate oracle_gap once (same eval seeds for all comparisons)
    print!("Evaluating oracle_gap baseline... ");
    let _ = std::io::stdout().flush();
    let mut og = OracleGapFeePolicy {
        base_fee: 0.0006,
        gap_multiplier: 0.1,
        min_fee: 0.0001,
        max_fee: 0.0020,
    };
    let og_pnls = eval(&config, &mut og);
    println!("mean={:.4}", mean(&og_pnls));

    // Train + eval each policy seed
    let sep = "─".repeat(78);
    println!("\n{sep}");
    println!(
        "{:>12} {:>10} {:>10} {:>10} {:>10} {:>8}",
        "train_seed", "rl_pnl", "og_pnl", "delta", "pct_delta", "beat%"
    );
    println!("{sep}");

    let mut csv_rows: Vec<(u64, f64, f64, f64, f64, f64)> = Vec::new();

    for &pseed in POLICY_SEEDS {
        print!("Training seed={pseed}... ");
        let _ = std::io::stdout().flush();
        let mut policy = train(&config, pseed);
        policy.set_inference();
        let rl_pnls = eval(&config, &mut policy);

        let rl_mean = mean(&rl_pnls);
        let og_mean = mean(&og_pnls);
        let paired_deltas: Vec<f64> = rl_pnls.iter().zip(&og_pnls).map(|(r, o)| r - o).collect();
        let delta = mean(&paired_deltas);
        let pct_delta = delta / og_mean.abs() * 100.0;
        let beat_pct = paired_deltas.iter().filter(|&&d| d > 0.0).count() as f64
            / paired_deltas.len() as f64
            * 100.0;

        println!(
            "{:>12} {:>10.4} {:>10.4} {:>10.4} {:>9.1}% {:>7.1}%",
            pseed, rl_mean, og_mean, delta, pct_delta, beat_pct
        );
        csv_rows.push((pseed, rl_mean, og_mean, delta, pct_delta, beat_pct));
    }
    println!("{sep}");

    // Summary: mean delta and consistency
    let mean_delta = csv_rows.iter().map(|r| r.3).sum::<f64>() / csv_rows.len() as f64;
    let n_positive = csv_rows.iter().filter(|r| r.3 > 0.0).count();
    println!(
        "Mean paired Δ across seeds: {:.4}  ({}/{} seeds beat oracle_gap)",
        mean_delta,
        n_positive,
        csv_rows.len()
    );

    // CSV
    std::fs::create_dir_all("data/processed").unwrap();
    let csv_path = format!("data/processed/campbell_rl_multiseed_{}.csv", config.name);
    let mut f = std::fs::File::create(&csv_path).unwrap();
    writeln!(
        f,
        "scenario,train_seed,rl_mean_pnl,og_mean_pnl,paired_delta,pct_delta,beat_pct"
    )
    .unwrap();
    for (seed, rl, og, delta, pct, beat) in &csv_rows {
        writeln!(
            f,
            "{},{},{:.4},{:.4},{:.4},{:.2},{:.1}",
            config.name, seed, rl, og, delta, pct, beat
        )
        .unwrap();
    }
    println!("Saved → {csv_path}");
}
