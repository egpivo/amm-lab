//! Run one deterministic episode and export the full trajectory to CSV.
//!
//! Usage: rl_equilibrium_sim [--mode dynamic_duopoly|dynamic_monopoly|constant_duopoly]
//!                          [--policy twap|immediate|myopic|random] [--seed N] [--out PATH]

use amm_lab::sim::env::{EnvConfig, ExecEnv, MarketMode};
use amm_lab::sim::execution_agent::{
    ExecutionPolicy, FeeAwareTwapPolicy, ImmediatePolicy, LookaheadPolicy, MyopicRouterPolicy,
    RandomPolicy, TwapPolicy,
};
use amm_lab::sim::q_learner::{QPolicy, QTable};
use clap::Parser;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "dynamic_duopoly")]
    mode: String,
    #[arg(long, default_value = "twap")]
    policy: String,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value = "experiments/rl_execution/out/trajectory.csv")]
    out: String,
}

pub fn parse_mode(s: &str) -> MarketMode {
    match s {
        "constant_duopoly" => MarketMode::ConstantDuopoly,
        "dynamic_monopoly" => MarketMode::DynamicMonopoly,
        _ => MarketMode::DynamicDuopoly,
    }
}

fn main() {
    let args = Args::parse();
    let cfg = EnvConfig::baseline(parse_mode(&args.mode), args.seed);
    let horizon = cfg.order.horizon;
    let mut env = ExecEnv::new(cfg);

    let mut policy: Box<dyn ExecutionPolicy> = match args.policy.as_str() {
        "immediate" => Box::new(ImmediatePolicy),
        "myopic" => Box::new(MyopicRouterPolicy),
        "random" => Box::new(RandomPolicy::new(args.seed)),
        "fee_aware_twap" => Box::new(FeeAwareTwapPolicy { horizon }),
        "lookahead" => Box::new(LookaheadPolicy {
            horizon,
            kappa: 16.0,
            unfinished_penalty: 0.02,
        }),
        // "q:PATH" replays a trained table exported by rl_equilibrium_train_tabular
        p if p.starts_with("q:") => Box::new(QPolicy {
            table: QTable::load(&p[2..]).expect("load q table"),
            horizon,
        }),
        _ => Box::new(TwapPolicy { horizon }),
    };

    while !env.is_done() {
        let obs = env.observe();
        let action = policy.act(&obs);
        env.step(action);
    }

    env.write_trajectory_csv(&args.out).expect("write csv");
    let summary = env.summary(policy.name());
    println!("{}", serde_json::to_string_pretty(&summary).unwrap());
    eprintln!("trajectory written to {}", args.out);
}
