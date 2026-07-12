use amm_lab::campbell::fee_policy::TabularLearnedFeePolicy;
use amm_lab::campbell::gbm::generate_gbm;
use amm_lab::campbell::simulation::{load_sim_config, run_rl_training_episode};
use std::io::Write;

const TRAIN_PATHS: u64 = 5_000;
const GAMMA: f64 = 0.99;
const ALPHA: f64 = 0.05;
const EPSILON_START: f64 = 1.0;
const EPSILON_MIN: f64 = 0.05;
const EPSILON_DECAY: f64 = 0.9994;
const INITIAL_PRICE: f64 = 1.0;
const PRINT_EVERY: u64 = 500;

fn main() {
    let toml_path = std::env::args().nth(1);
    let mut config = load_sim_config(toml_path.as_deref());
    let dt = 1.0 / config.n_steps as f64;

    println!(
        "Scenario: {} | regime: {:?}",
        config.name, config.flow_regime
    );
    println!(
        "Config:    {}",
        toml_path
            .as_deref()
            .unwrap_or(amm_lab::campbell::simulation::DEFAULT_RL_SCENARIO)
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
        config.seed = seed;
        let cex_prices = generate_gbm(
            config.n_steps,
            INITIAL_PRICE,
            config.mu,
            config.sigma,
            dt,
            seed,
        );

        let (records, episode_reward) =
            run_rl_training_episode(&config, &cex_prices, &mut policy, GAMMA);

        let avg_fee_bps =
            records.iter().map(|r| r.fee_used * 10_000.0).sum::<f64>() / records.len() as f64;

        episode_rewards.push(episode_reward);
        episode_fees.push(avg_fee_bps);
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
