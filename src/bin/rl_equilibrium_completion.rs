//! M3R-A: baseline policies under Standard vs ForcedTerminal completion.
//! DQN rows are appended to the same CSV by dqn_m3r_eval.py.
//!
//! Usage: rl_equilibrium_completion [--n-seeds 500]

use amm_lab::sim::env::{CompletionRule, EnvConfig, ExecEnv, MarketMode};
use amm_lab::sim::execution_agent::{
    ExecutionPolicy, FeeAwareTwapPolicy, LookaheadPolicy, TwapPolicy,
};
use clap::Parser;
use std::io::Write as _;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 500)]
    n_seeds: u64,
    #[arg(
        long,
        default_value = "experiments/rl_execution/out/m3r_completion.csv"
    )]
    out: String,
}

pub const HEADER: &str = "seed,seed_set,mode,policy,completion_rule,shortfall_bps,completion_rate,terminal_penalty_bps,forced_terminal_cost_bps,fee_bps,gas_bps,slippage_ex_fee_bps,drift_bps,route_share_A,route_share_B,wait_share";

fn main() {
    let args = Args::parse();
    let horizon = EnvConfig::baseline(MarketMode::DynamicDuopoly, 0)
        .order
        .horizon;
    let mut f = std::fs::File::create(&args.out).expect("csv");
    writeln!(f, "{HEADER}").unwrap();

    for (rule, rule_name) in [
        (CompletionRule::Standard, "standard"),
        (CompletionRule::ForcedTerminal, "forced_terminal"),
    ] {
        let mut sums: std::collections::BTreeMap<&str, (f64, f64)> = Default::default();
        for (label, base) in [("test", 30_000u64), ("fresh", 40_000u64)] {
            for seed in base..base + args.n_seeds {
                let mut cfg = EnvConfig::baseline(MarketMode::DynamicDuopoly, seed);
                cfg.completion_rule = rule;
                let mut policies: Vec<Box<dyn ExecutionPolicy>> = vec![
                    Box::new(TwapPolicy { horizon }),
                    Box::new(FeeAwareTwapPolicy { horizon }),
                    Box::new(LookaheadPolicy {
                        horizon,
                        kappa: 16.0,
                        unfinished_penalty: 0.02,
                    }),
                ];
                for p in policies.iter_mut() {
                    let mut env = ExecEnv::new(cfg.clone());
                    p.reset();
                    while !env.is_done() {
                        let obs = env.observe();
                        let a = p.act(&obs);
                        env.step(a);
                    }
                    let s = env.summary(p.name());
                    writeln!(
                        f,
                        "{seed},{label},DynamicDuopoly,{},{rule_name},{:.4},{:.6},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
                        s.policy, s.shortfall_bps, s.completion_rate,
                        s.terminal_penalty_bps, s.forced_terminal_cost_bps,
                        s.fee_paid_bps, s.gas_paid_bps, s.slippage_ex_fee_bps,
                        s.drift_bps, s.route_share_a, s.route_share_b, s.wait_share
                    )
                    .unwrap();
                    if label == "test" {
                        let e = sums
                            .entry(match s.policy.as_str() {
                                "twap" => "twap",
                                "fee_aware_twap" => "fee_aware_twap",
                                _ => "lookahead",
                            })
                            .or_insert((0.0, 0.0));
                        e.0 += s.shortfall_bps;
                        e.1 += s.completion_rate;
                    }
                }
            }
        }
        println!(
            "--- rule {rule_name} (test means, {} seeds) ---",
            args.n_seeds
        );
        for (name, (is, comp)) in &sums {
            println!(
                "{name:<16} IS {:>8.2}  completion {:.4}",
                is / args.n_seeds as f64,
                comp / args.n_seeds as f64
            );
        }
    }
    println!("wrote {}", args.out);
}
