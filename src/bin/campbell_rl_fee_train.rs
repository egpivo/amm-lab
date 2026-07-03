use amm_lab::campbell::fee_policy::TabularLearnedFeePolicy;
use amm_lab::campbell::gbm::generate_gbm;
use amm_lab::campbell::simulation::{SimConfig, run_simulation};
use std::io::Write;

fn main() {
    let config = SimConfig {
        name: "rl_train".to_string(),
        description: "RL tabular training".to_string(),
        amm_fee: 0.0006,
        cex_fee: 0.0010,
        buy_demand: 100.0,
        sell_demand: 100.0,
        reserve_x: 10000.0,
        reserve_y: 10000.0,
        sigma: 0.04,
        mu: 0.0,
        n_steps: 1440,
        seed: 0,
    };
    let dt = 1.0 / config.n_steps as f64;

    const TRAIN_PATHS: u64 = 5000;
    const EPSILON_START: f64 = 1.0;
    const EPSILON_MIN: f64 = 0.05;
    const EPSILON_DECAY: f64 = 0.9994; // 1.0 * 0.9994^5000 ≈ 0.05
    const ALPHA: f64 = 0.1;

    // training
    let mut policy = TabularLearnedFeePolicy::new(EPSILON_START, ALPHA, 42);
    let mut summary_rows: Vec<String> = Vec::with_capacity(TRAIN_PATHS as usize);

    for seed in 0..TRAIN_PATHS {
        let cex_prices = generate_gbm(config.n_steps, 1.0, config.mu, config.sigma, dt, seed);
        let records = run_simulation(&config, &cex_prices, &mut policy);

        // terminal reward: hedged_pnl = fee_revenue - LVR
        let fee_revenue: f64 = records.iter().map(|r| r.step_fee).sum();
        let last = records.last().unwrap();
        let lvr = last.hedging_portfolio - last.pool_value;
        let hedged_pnl = fee_revenue - lvr;

        let avg_fee_bps =
            records.iter().map(|r| r.fee_used * 10_000.0).sum::<f64>() / records.len() as f64;
        let eps_snapshot = policy.epsilon;

        policy.update_episode(hedged_pnl);
        policy.decay_epsilon(EPSILON_DECAY, EPSILON_MIN);

        summary_rows.push(format!(
            "{},{},{:.4},{:.4},{:.6}",
            seed, seed, hedged_pnl, avg_fee_bps, eps_snapshot
        ));

        // save outputs + print stats
        std::fs::create_dir_all("data/processed").unwrap();
        policy
            .save_q_table("data/processed/campbell_rl_fee_table.csv")
            .unwrap();
        println!("Q-table -> data/processed/campbell_rl_fee_table.csv");

        {
            let mut f =
                std::fs::File::create("data/processed/campbell_rl_training_summary.csv").unwrap();
            writeln!(f, "episode,seed,episode_reward,avg_fee_bps,epsilon").unwrap();
            for row in &summary_rows {
                writeln!(f, "{}", row).unwrap();
            }
        }

        println!("Training summary -> data/processed/campbell_rl_training_summary.csv");

        // quick stats
        let last_500_reward: f64 = summary_rows
            .iter()
            .rev()
            .take(500)
            .map(|r| r.split(',').nth(2).unwrap().parse::<f64>().unwrap())
            .sum::<f64>()
            / 500.0;
        println!("States visited: {}", policy.q_table.len());
        println!("Mean reward (last 500 episodes): {:.2}", last_500_reward);
    }
}
