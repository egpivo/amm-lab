use amm_lab::campbell::fee_policy::{FeeObservation, RL_ACTIONS_BPS, TabularLearnedFeePolicy};
use amm_lab::campbell::gbm::generate_gbm;
use amm_lab::campbell::pool::CampbellPool;
use amm_lab::campbell::simulation::{FlowRegime, SimConfig};
use amm_lab::campbell::trader::{arb_delta, fundamental_buy_delta, fundamental_sell_delta};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::VecDeque;
use std::io::Write;

// ── MDP parameters ────────────────────────────────────────────────────────────
const TRAIN_PATHS: u64 = 5_000;
const GAMMA: f64 = 0.99;
const ALPHA: f64 = 0.05;
const EPSILON_START: f64 = 1.0;
const EPSILON_MIN: f64 = 0.05;
const EPSILON_DECAY: f64 = 0.9994;
const VOL_WINDOW: usize = 20;
const INITIAL_PRICE: f64 = 1.0;
const PRINT_EVERY: u64 = 500;

// ── Rolling window helpers ────────────────────────────────────────────────────

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

// ── Scenario config ───────────────────────────────────────────────────────────

fn load_config(path: Option<&str>) -> SimConfig {
    if let Some(p) = path {
        let s = std::fs::read_to_string(p).unwrap_or_else(|e| panic!("cannot read {p}: {e}"));
        toml::from_str(&s).unwrap_or_else(|e| panic!("invalid TOML {p}: {e}"))
    } else {
        SimConfig {
            name: "campbell_rl_normal".to_string(),
            description: "RL training — normal flow".to_string(),
            amm_fee: 0.0006,
            cex_fee: 0.0010,
            buy_demand: 100.0,
            sell_demand: 100.0,
            reserve_x: 1000.0,
            reserve_y: 1000.0,
            sigma: 0.04,
            mu: 0.0,
            n_steps: 1440,
            seed: 0,
            flow_regime: FlowRegime::Normal,
            toxic_burst_prob: 0.0,
            toxic_burst_arb_scale: 1.0,
            toxic_burst_fund_scale: 1.0,
            regime_switch_period: 0,
            e1_lambda: 0.0, // E1 substitution disabled (neutral default)
            e1_fee_ref: 0.0006,
            e5_arb_prob: 1.0, // E5 latency disabled (neutral default)
        }
    }
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

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let toml_path = std::env::args().nth(1);
    let config = load_config(toml_path.as_deref());
    let dt = 1.0 / config.n_steps as f64;

    println!(
        "Scenario: {} | regime: {:?}",
        config.name, config.flow_regime
    );
    println!(
        "Training {} episodes, γ={}, α={}, ε: {}→{}",
        TRAIN_PATHS, GAMMA, ALPHA, EPSILON_START, EPSILON_MIN
    );

    let mut policy = TabularLearnedFeePolicy::new(EPSILON_START, ALPHA, 42);
    let mut episode_rewards: Vec<f64> = Vec::with_capacity(TRAIN_PATHS as usize);
    let mut episode_fees: Vec<f64> = Vec::with_capacity(TRAIN_PATHS as usize);

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
        let mut episode_reward = 0.0f64;
        let mut total_fee_bps = 0.0f64;

        for i in 0..config.n_steps {
            let prev_cex = cex_prices[i];
            let cex_price = cex_prices[i + 1];

            // Pre-step position (before price update and trades)
            let prev_position = hedging - pool.pool_value(prev_cex);

            // Delta hedge update
            hedging += pool.reserve_y * (cex_price - prev_cex);
            let fee_before = pool.cumulative_fee_revenue;

            // Build observation from current rolling windows
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

            // Choose action (ε-greedy)
            let state = policy.obs_to_state(&obs);
            let action = policy.choose_action(state);
            let fee = RL_ACTIONS_BPS[action] / 10_000.0;
            pool.amm_fee = fee;
            prev_fee = fee;
            total_fee_bps += RL_ACTIONS_BPS[action];

            // Execute trades
            let fund_scale = effective_fund_scale(&config, i, &mut regime_rng);
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

            // Step reward = step_fee − Δ(hedging − pool_value)
            let cur_position = hedging - pool.pool_value(cex_price);
            let step_reward = step_fee - (cur_position - prev_position);
            episode_reward += step_reward;

            // Update rolling windows BEFORE computing next_state
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

            // Build next observation for TD target
            let next_gap_bps = (pool.marginal_price() - cex_price) / cex_price * 10_000.0;
            let next_inv_skew = (pool.reserve_x - pool.reserve_y * cex_price)
                / (pool.reserve_x + pool.reserve_y * cex_price);
            let next_obs = FeeObservation {
                step: i + 1,
                external_price: cex_price,
                amm_price: pool.marginal_price(),
                oracle_gap_bps: next_gap_bps,
                inventory_skew: next_inv_skew,
                recent_vol: rolling_std(&log_rets),
                recent_arb_frac: frac_true(&arb_flags),
                recent_fund_frac: frac_true(&fund_flags),
                recent_volume: vol_sums.iter().sum(),
                previous_fee: fee,
            };
            let next_state = policy.obs_to_state(&next_obs);

            // TD(0) update
            policy.update_step(state, action, step_reward, next_state, GAMMA);
        }

        episode_rewards.push(episode_reward);
        episode_fees.push(total_fee_bps / config.n_steps as f64);
        policy.decay_epsilon(EPSILON_DECAY, EPSILON_MIN);

        if (ep + 1) % PRINT_EVERY == 0 {
            let n = PRINT_EVERY as usize;
            let start = episode_rewards.len().saturating_sub(n);
            let mean_r = episode_rewards[start..].iter().sum::<f64>() / n as f64;
            let mean_f = episode_fees[start..].iter().sum::<f64>() / n as f64;
            println!(
                "ep {:5} | mean reward={:.4} | mean fee={:.2} bps | ε={:.4} | states={}",
                ep + 1,
                mean_r,
                mean_f,
                policy.epsilon,
                policy.q_table.len()
            );
        }
    }

    std::fs::create_dir_all("data/processed").unwrap();

    let q_path = format!("data/processed/campbell_rl_fee_table_{}.csv", config.name);
    policy.save_q_table(&q_path).unwrap();
    println!("Q-table → {q_path}");

    // Also write to the canonical path for the compare binary
    policy
        .save_q_table("data/processed/campbell_rl_fee_table.csv")
        .unwrap();
    println!("Q-table → data/processed/campbell_rl_fee_table.csv");

    let sum_path = format!(
        "data/processed/campbell_rl_training_summary_{}.csv",
        config.name
    );
    let mut f = std::fs::File::create(&sum_path).unwrap();
    writeln!(f, "episode,seed,episode_reward,avg_fee_bps,epsilon").unwrap();
    for (i, (&r, &fee)) in episode_rewards.iter().zip(&episode_fees).enumerate() {
        writeln!(
            f,
            "{},{},{:.4},{:.4},{:.4}",
            i,
            i,
            r,
            fee,
            EPSILON_MIN + (EPSILON_START - EPSILON_MIN) * EPSILON_DECAY.powi(i as i32).max(0.0)
        )
        .unwrap();
    }
    println!("Training summary → {sum_path}");

    let n = 500.min(episode_rewards.len());
    let mean_r = episode_rewards[episode_rewards.len() - n..]
        .iter()
        .sum::<f64>()
        / n as f64;
    println!("States visited: {}", policy.q_table.len());
    println!("Mean reward (last {} episodes): {:.4}", n, mean_r);
}
