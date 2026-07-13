//! M3R reference numbers: tuned lookahead and TWAP under every agent
//! ordering and market mode, ForcedTerminal completion, test + fresh seeds.
//! DQN matrices from dqn_m3r_eval.py are compared against these rows.

use amm_lab::sim::env::{AgentOrder, CompletionRule, EnvConfig, ExecEnv, MarketMode};
use amm_lab::sim::execution_agent::{ExecutionPolicy, LookaheadPolicy, TwapPolicy};
use clap::Parser;
use std::io::Write as _;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 300)]
    n_seeds: u64,
    #[arg(long, default_value = "experiments/rl_execution/out/m3r_reference.csv")]
    out: String,
    /// Override: evaluate ONLY dynamic_duopoly at this seed base (used for
    /// the frozen final paper block), all three orderings.
    #[arg(long)]
    seed_base: Option<u64>,
    #[arg(long, default_value = "final")]
    seed_label: String,
    /// Re-run the lookahead kappa validation grid per intra-step ordering
    /// (dynamic duopoly, validation seeds 20,000-20,199) and exit.
    #[arg(long, default_value_t = false)]
    tune_kappa: bool,
    /// Lookahead carry parameter (default matches the frozen M3R runs).
    #[arg(long, default_value_t = 16.0)]
    kappa: f64,
}

fn tune_kappa(horizon: usize) {
    let cells = [
        (
            MarketMode::DynamicDuopoly,
            AgentOrder::Before,
            "dynamic/before",
        ),
        (
            MarketMode::DynamicDuopoly,
            AgentOrder::Random,
            "dynamic/random",
        ),
        (
            MarketMode::DynamicDuopoly,
            AgentOrder::After,
            "dynamic/after",
        ),
        (
            MarketMode::ConstantDuopoly,
            AgentOrder::Before,
            "constant/before",
        ),
        (
            MarketMode::DynamicMonopoly,
            AgentOrder::Before,
            "monopoly/before",
        ),
    ];
    for (mode, order, order_name) in cells {
        let mut best = (f64::NAN, f64::INFINITY);
        for kappa in [4.0, 8.0, 16.0, 32.0] {
            let mut sum = 0.0;
            for seed in 20_000u64..20_200 {
                let mut cfg = EnvConfig::baseline(mode, seed);
                cfg.agent_order = order;
                cfg.completion_rule = CompletionRule::ForcedTerminal;
                let mut la = LookaheadPolicy {
                    horizon,
                    kappa,
                    unfinished_penalty: 0.02,
                };
                let mut env = ExecEnv::new(cfg);
                la.reset();
                while !env.is_done() {
                    let a = la.act(&env.observe());
                    env.step(a);
                }
                sum += env.summary("lookahead").shortfall_bps;
            }
            let mean = sum / 200.0;
            println!("{order_name} kappa={kappa}: val IS {mean:.2}");
            if mean < best.1 {
                best = (kappa, mean);
            }
        }
        println!("{order_name}: selected kappa={}", best.0);
    }
}

fn main() {
    let args = Args::parse();
    if args.tune_kappa {
        let horizon = EnvConfig::baseline(MarketMode::DynamicDuopoly, 0)
            .order
            .horizon;
        tune_kappa(horizon);
        return;
    }
    let horizon = EnvConfig::baseline(MarketMode::DynamicDuopoly, 0)
        .order
        .horizon;
    let mut f = std::fs::File::create(&args.out).expect("csv");
    writeln!(
        f,
        "policy,mode,agent_order,seed_set,seed,shortfall_bps,completion_rate"
    )
    .unwrap();

    let orders = [
        (AgentOrder::Before, "before"),
        (AgentOrder::Random, "random"),
        (AgentOrder::After, "after"),
    ];
    let modes = [
        (MarketMode::ConstantDuopoly, "constant_duopoly"),
        (MarketMode::DynamicMonopoly, "dynamic_monopoly"),
        (MarketMode::DynamicDuopoly, "dynamic_duopoly"),
    ];
    for (mode, mode_name) in modes {
        for (order, order_name) in orders {
            // non-duopoly modes only need the default ordering
            if mode != MarketMode::DynamicDuopoly && order != AgentOrder::Before {
                continue;
            }
            if args.seed_base.is_some() && mode != MarketMode::DynamicDuopoly {
                continue;
            }
            let seed_sets: Vec<(&str, u64)> = match args.seed_base {
                Some(base) => vec![(args.seed_label.as_str(), base)],
                None => vec![("test", 30_000u64), ("fresh", 40_000u64)],
            };
            let mut sums: std::collections::BTreeMap<&str, f64> = Default::default();
            for (label, base) in seed_sets {
                for seed in base..base + args.n_seeds {
                    let mut cfg = EnvConfig::baseline(mode, seed);
                    cfg.agent_order = order;
                    cfg.completion_rule = CompletionRule::ForcedTerminal;
                    let mut la = LookaheadPolicy {
                        horizon,
                        kappa: args.kappa,
                        unfinished_penalty: 0.02,
                    };
                    let mut tw = TwapPolicy { horizon };
                    for (name, p) in [
                        ("lookahead", &mut la as &mut dyn ExecutionPolicy),
                        ("twap", &mut tw),
                    ] {
                        let mut env = ExecEnv::new(cfg.clone());
                        p.reset();
                        while !env.is_done() {
                            let obs = env.observe();
                            let a = p.act(&obs);
                            env.step(a);
                        }
                        let s = env.summary(name);
                        writeln!(
                            f,
                            "{name},{mode_name},{order_name},{label},{seed},{:.4},{:.6}",
                            s.shortfall_bps, s.completion_rate
                        )
                        .unwrap();
                        if label == "test" || args.seed_base.is_some() {
                            *sums.entry(name).or_insert(0.0) += s.shortfall_bps;
                        }
                    }
                }
            }
            println!(
                "{mode_name}/{order_name}: lookahead {:.2}  twap {:.2}",
                sums["lookahead"] / args.n_seeds as f64,
                sums["twap"] / args.n_seeds as f64
            );
        }
    }
    println!("wrote {}", args.out);
}
