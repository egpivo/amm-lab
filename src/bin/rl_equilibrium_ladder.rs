//! value-boundary: how much execution value exists beyond one-step lookahead?
//!
//! Fair planners (valid policies, decision-time info only):
//!   two_step / three_step: expectimax over the discrete actions on a
//!   deterministic forward model built FROM THE OBSERVATION (reserves
//!   reconstructed from mid + inventory; expected noise volume; determin-
//!   istic arb; martingale oracle). No access to the episode's future
//!   shocks. Terminal carry has the same tuned-kappa form as lookahead.
//!
//! Invalid bound (diagnostic only, NOT a policy):
//!   clairvoyant: per-seed hill-climb over full action sequences evaluated
//!   on the true environment (sees future oracle/noise/arb). Lower bound
//!   on achievable shortfall for the discrete action set.

use amm_lab::sim::env::{CompletionRule, EnvConfig, ExecEnv, MarketMode, N_ACTIONS};
use amm_lab::sim::execution_agent::{ExecutionPolicy, LookaheadPolicy, TwapPolicy};
use amm_lab::sim::planner::DeterministicPlanner;
use amm_lab::sim::q_learner::{QPolicy, QTable};
use clap::Parser;
use std::io::Write as _;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 300)]
    n_seeds: u64,
    #[arg(long, default_value_t = 200)]
    n_val: u64,
    #[arg(long, default_value = "experiments/rl_execution/out")]
    out_dir: String,
    #[arg(long, default_value_t = 16.0)]
    lookahead_kappa: f64,
    /// Hill-climb sweep limit for the clairvoyant bound.
    #[arg(long, default_value_t = 6)]
    max_sweeps: usize,
    /// Evaluation seed base (default: development test block).
    #[arg(long, default_value_t = TEST_BASE)]
    seed_base: u64,
    /// "standard" (validation-grid default) or "forced_terminal" (paper headline rule).
    #[arg(long, default_value = "standard")]
    completion_rule: String,
    /// Optional trained Q-tables to include in the ladder.
    #[arg(long)]
    q_coarse: Option<String>,
    #[arg(long)]
    q_fine: Option<String>,
    /// Output CSV name (within out_dir).
    #[arg(long, default_value = "m3_value_boundary.csv")]
    out_name: String,
}

const VAL_BASE: u64 = 20_000;
const TEST_BASE: u64 = 30_000;

// ---------- episode helpers ----------

fn run_policy(cfg: EnvConfig, policy: &mut dyn ExecutionPolicy) -> (f64, f64, Vec<usize>) {
    let mut env = ExecEnv::new(cfg);
    policy.reset();
    let mut actions = Vec::new();
    while !env.is_done() {
        let obs = env.observe();
        let a = policy.act(&obs);
        actions.push(a);
        env.step(a);
    }
    let s = env.summary(policy.name());
    (s.shortfall_bps, s.completion_rate, actions)
}

fn run_sequence(cfg: EnvConfig, actions: &[usize]) -> f64 {
    let mut env = ExecEnv::new(cfg);
    let mut i = 0;
    while !env.is_done() {
        env.step(*actions.get(i).unwrap_or(&0));
        i += 1;
    }
    env.summary("seq").shortfall_bps
}

/// Clairvoyant bound: coordinate-descent over the action sequence on the
/// TRUE env (sees future shocks). Not a valid policy.
fn clairvoyant(cfg: &EnvConfig, init: &[usize], max_sweeps: usize) -> (f64, f64) {
    let horizon = cfg.order.horizon;
    let mut seq = init.to_vec();
    seq.resize(horizon, 0);
    let mut best = run_sequence(cfg.clone(), &seq);
    for _ in 0..max_sweeps {
        let mut improved = false;
        for pos in 0..horizon {
            let orig = seq[pos];
            for a in 0..N_ACTIONS {
                if a == orig {
                    continue;
                }
                seq[pos] = a;
                let c = run_sequence(cfg.clone(), &seq);
                if c < best - 1e-9 {
                    best = c;
                    improved = true;
                } else {
                    seq[pos] = orig;
                }
                if seq[pos] == a {
                    break; // keep improvement, move on
                }
            }
        }
        if !improved {
            break;
        }
    }
    // completion of the final sequence
    let mut env = ExecEnv::new(cfg.clone());
    let mut i = 0;
    while !env.is_done() {
        env.step(*seq.get(i).unwrap_or(&0));
        i += 1;
    }
    (best, env.summary("clairvoyant").completion_rate)
}

fn main() {
    let args = Args::parse();
    let rule = if args.completion_rule == "forced_terminal" {
        CompletionRule::ForcedTerminal
    } else {
        CompletionRule::Standard
    };
    let base = EnvConfig::baseline(MarketMode::DynamicDuopoly, 0);
    let horizon = base.order.horizon;
    let mk_cfg = |seed: u64| {
        let mut c = EnvConfig::baseline(MarketMode::DynamicDuopoly, seed);
        c.completion_rule = rule;
        c
    };

    // tune planner kappa on validation seeds (same protocol as lookahead)
    let mut kappas = std::collections::HashMap::new();
    for depth in [2usize, 3] {
        let mut best = (8.0, f64::INFINITY);
        for kappa in [4.0, 8.0, 16.0, 32.0] {
            let mut sum = 0.0;
            for seed in VAL_BASE..VAL_BASE + args.n_val {
                let cfg = mk_cfg(seed);
                let mut p = DeterministicPlanner::new(depth, kappa, cfg.clone(), "planner");
                sum += run_policy(cfg, &mut p).0;
            }
            let m = sum / args.n_val as f64;
            eprintln!("depth={depth} kappa={kappa}: val IS {m:.2}");
            if m < best.1 {
                best = (kappa, m);
            }
        }
        eprintln!("depth={depth}: selected kappa={}", best.0);
        kappas.insert(depth, best.0);
    }

    let out_path = format!("{}/{}", args.out_dir, args.out_name);
    let mut f = std::fs::File::create(&out_path).expect("create csv");
    writeln!(f, "policy,seed,shortfall_bps,completion_rate").unwrap();

    let mut sums: Vec<(String, f64, f64)> = Vec::new();
    let mut record = |name: &str, seed: u64, is: f64, comp: f64, f: &mut std::fs::File| {
        writeln!(f, "{name},{seed},{is:.4},{comp:.4}").unwrap();
        if let Some(e) = sums.iter_mut().find(|e| e.0 == name) {
            e.1 += is;
            e.2 += comp;
        } else {
            sums.push((name.to_string(), is, comp));
        }
    };

    let t0 = std::time::Instant::now();
    let q_policies: Vec<(&str, QTable)> = [
        ("q_learner", &args.q_coarse),
        ("q_learner_fine", &args.q_fine),
    ]
    .into_iter()
    .filter_map(|(name, path)| {
        path.as_ref()
            .map(|p| (name, QTable::load(p).expect("load q table")))
    })
    .collect();

    for seed in args.seed_base..args.seed_base + args.n_seeds {
        let cfg = mk_cfg(seed);
        let (is, comp, _) = run_policy(cfg.clone(), &mut TwapPolicy { horizon });
        record("twap", seed, is, comp, &mut f);
        let (is, comp, la_actions) = run_policy(
            cfg.clone(),
            &mut LookaheadPolicy {
                horizon,
                kappa: args.lookahead_kappa,
                unfinished_penalty: 0.02,
            },
        );
        record("lookahead", seed, is, comp, &mut f);
        for depth in [2usize, 3] {
            let (is, comp, _) = run_policy(
                cfg.clone(),
                &mut DeterministicPlanner::new(
                    depth,
                    kappas[&depth],
                    cfg.clone(),
                    if depth == 2 { "two_step" } else { "three_step" },
                ),
            );
            record(
                if depth == 2 { "two_step" } else { "three_step" },
                seed,
                is,
                comp,
                &mut f,
            );
        }
        for (name, table) in &q_policies {
            let (is, comp, _) = run_policy(
                cfg.clone(),
                &mut QPolicy {
                    table: table.clone(),
                    horizon,
                },
            );
            record(name, seed, is, comp, &mut f);
        }
        let (is, comp) = clairvoyant(&cfg, &la_actions, args.max_sweeps);
        record("clairvoyant", seed, is, comp, &mut f);
    }
    eprintln!(
        "evaluated {} seeds in {:.1}s",
        args.n_seeds,
        t0.elapsed().as_secs_f64()
    );

    let n = args.n_seeds as f64;
    println!(
        "\n=== value-boundary value boundary (DynamicDuopoly, {} test seeds) ===",
        args.n_seeds
    );
    for (name, is, comp) in &sums {
        println!(
            "{name:<12} IS {:>8.2} bps   completion {:.4}",
            is / n,
            comp / n
        );
    }
    println!("wrote {out_path}");
}
