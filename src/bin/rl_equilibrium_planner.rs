//! baseline-duopoly-D: stochastic rollout planner baseline.
//!
//! For each decision: for each root action, average the K-step cost over N
//! sampled model rollouts (GBM oracle steps, lognormal noise volumes,
//! Bernoulli arbitrage — the simulator's own distributions, but sampled
//! from planner-private RNG streams derived from (episode step, rollout
//! index), never from the episode seed, so realized future shocks are
//! inaccessible). Rollout continuation actions come from the one-step
//! lookahead heuristic on the model; the tail beyond K uses the tuned
//! kappa carry. Depth, rollout count, and kappa are tuned on validation
//! seeds only.
//!
//! Usage: rl_equilibrium_planner [--n-val 200] [--n-seeds 300]
//!        [--completion-rule forced_terminal|standard]

use amm_lab::sim::env::{CompletionRule, EnvConfig, ExecEnv, MarketMode};
use amm_lab::sim::execution_agent::{ExecutionPolicy, LookaheadPolicy, TwapPolicy};
use amm_lab::sim::planner::StochasticPlanner;
use clap::Parser;
use std::io::Write as _;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 200)]
    n_val: u64,
    #[arg(long, default_value_t = 300)]
    n_seeds: u64,
    #[arg(long, default_value = "forced_terminal")]
    completion_rule: String,
    #[arg(long, default_value = "experiments/rl_execution/out")]
    out_dir: String,
    /// Optional single evaluation block override (replaces test+fresh).
    #[arg(long)]
    seed_base: Option<u64>,
    #[arg(long, default_value = "m3r_stochastic_planner.csv")]
    out_name: String,
}

const VAL_BASE: u64 = 20_000;
const TEST_BASE: u64 = 30_000;
const FRESH_BASE: u64 = 40_000;

fn run(cfg: EnvConfig, policy: &mut dyn ExecutionPolicy) -> (f64, f64) {
    let mut env = ExecEnv::new(cfg);
    policy.reset();
    while !env.is_done() {
        let obs = env.observe();
        let a = policy.act(&obs);
        env.step(a);
    }
    let s = env.summary(policy.name());
    (s.shortfall_bps, s.completion_rate)
}

fn main() {
    let args = Args::parse();
    let rule = if args.completion_rule == "standard" {
        CompletionRule::Standard
    } else {
        CompletionRule::ForcedTerminal
    };
    let mk_cfg = |seed: u64| {
        let mut c = EnvConfig::baseline(MarketMode::DynamicDuopoly, seed);
        c.completion_rule = rule;
        c
    };
    let horizon = mk_cfg(0).order.horizon;

    // --- validation grid (validation seeds only) ---
    let mut best = (2usize, 8usize, 8.0f64, f64::INFINITY);
    for depth in [2usize, 3] {
        for n_rollouts in [8usize, 16, 32] {
            for kappa in [8.0, 16.0, 32.0] {
                let t0 = std::time::Instant::now();
                let mut sum = 0.0;
                for seed in VAL_BASE..VAL_BASE + args.n_val {
                    let cfg = mk_cfg(seed);
                    let mut p = StochasticPlanner {
                        depth,
                        n_rollouts,
                        kappa,
                        cfg: cfg.clone(),
                    };
                    sum += run(cfg, &mut p).0;
                }
                let m = sum / args.n_val as f64;
                eprintln!(
                    "K={depth} N={n_rollouts} kappa={kappa}: val IS {m:.2} ({:.1}s)",
                    t0.elapsed().as_secs_f64()
                );
                if m < best.3 {
                    best = (depth, n_rollouts, kappa, m);
                }
            }
        }
    }
    let (depth, n_rollouts, kappa, val_is) = best;
    eprintln!("selected K={depth} N={n_rollouts} kappa={kappa} (val {val_is:.2})");

    // --- test + fresh evaluation, with lookahead/twap references ---
    let out_path = format!("{}/{}", args.out_dir, args.out_name);
    let mut f = std::fs::File::create(&out_path).expect("csv");
    writeln!(f, "policy,seed_set,seed,shortfall_bps,completion_rate").unwrap();
    let seed_sets: Vec<(&str, u64, u64)> = match args.seed_base {
        Some(base) => vec![("final", base, args.n_seeds)],
        None => vec![
            ("test", TEST_BASE, args.n_seeds),
            ("fresh", FRESH_BASE, args.n_seeds),
        ],
    };
    for (label, base, n) in seed_sets {
        let mut sums = std::collections::BTreeMap::new();
        for seed in base..base + n {
            let cfg = mk_cfg(seed);
            let mut planner = StochasticPlanner {
                depth,
                n_rollouts,
                kappa,
                cfg: cfg.clone(),
            };
            let mut la = LookaheadPolicy {
                horizon,
                kappa: 16.0,
                unfinished_penalty: 0.02,
            };
            let mut tw = TwapPolicy { horizon };
            for (name, p) in [
                (
                    "stochastic_planner",
                    &mut planner as &mut dyn ExecutionPolicy,
                ),
                ("lookahead", &mut la),
                ("twap", &mut tw),
            ] {
                let (is, comp) = run(cfg.clone(), p);
                writeln!(f, "{name},{label},{seed},{is:.4},{comp:.4}").unwrap();
                let e = sums.entry(name).or_insert((0.0, 0.0));
                e.0 += is;
                e.1 += comp;
            }
        }
        println!("--- {label} ({n} seeds, rule {rule:?}) ---");
        for (name, (is, comp)) in &sums {
            println!(
                "{name:<20} IS {:>8.2} bps  completion {:.4}",
                is / n as f64,
                comp / n as f64
            );
        }
    }
    println!("wrote {out_path}");
}
