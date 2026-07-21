//! closed-loop: train the tabular Q-learner per market mode, tune the lookahead
//! baseline's kappa on validation seeds, then evaluate everything on
//! held-out test seeds.
//!
//! Seed protocol (disjoint by construction):
//!   train      : 1_000_000 + episode      (fresh path every episode)
//!   validation : 20_000 .. 20_000 + n_val (kappa tuning only)
//!   test       : 30_000 .. 30_000 + n_test (reported numbers)
//!
//! Outputs: q_table_<mode>.json, m1_results.csv, m1_actions.csv.

use amm_lab::sim::env::{EnvConfig, EpisodeSummary, ExecEnv, MarketMode, write_summaries_csv};
use amm_lab::sim::execution_agent::{
    ExecutionPolicy, FeeAwareTwapPolicy, ImmediatePolicy, LookaheadPolicy, MyopicRouterPolicy,
    RandomPolicy, TwapPolicy,
};
use amm_lab::sim::q_learner::{QPolicy, QTable, StateSpec, TrainConfig, train};
use clap::Parser;
use std::io::Write as _;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 30_000)]
    train_episodes: usize,
    #[arg(long, default_value_t = 200)]
    n_val: u64,
    #[arg(long, default_value_t = 500)]
    n_test: u64,
    #[arg(long, default_value = "experiments/rl_execution/out")]
    out_dir: String,
    /// State discretization: "coarse" (closed-loop) or "fine" (fine-tabular).
    #[arg(long, default_value = "coarse")]
    spec: String,
    /// Prefix for results/actions CSVs (m1 or m3_fine etc.).
    #[arg(long, default_value = "m1")]
    out_prefix: String,
}

const MODES: [(MarketMode, &str); 3] = [
    (MarketMode::ConstantDuopoly, "constant_duopoly"),
    (MarketMode::DynamicMonopoly, "dynamic_monopoly"),
    (MarketMode::DynamicDuopoly, "dynamic_duopoly"),
];
const VAL_BASE: u64 = 20_000;
const TEST_BASE: u64 = 30_000;
const KAPPA_GRID: [f64; 6] = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0];

/// Run one episode; optionally record per-step action rows.
fn run_episode(
    cfg: EnvConfig,
    policy: &mut dyn ExecutionPolicy,
    action_log: Option<&mut Vec<String>>,
) -> EpisodeSummary {
    let mut env = ExecEnv::new(cfg);
    policy.reset();
    let mut rows = Vec::new();
    while !env.is_done() {
        let obs = env.observe();
        let action = policy.act(&obs);
        if action_log.is_some() {
            rows.push(format!(
                "{},{:?},{},{},{:.4},{:.4},{:.2},{:.2},{:.2},{:.2},{}",
                policy.name(),
                env.cfg.mode,
                env.cfg.seed,
                obs.step,
                obs.remaining_frac,
                obs.remaining_time_frac,
                obs.rival_quote_gap_bps,
                obs.est_slippage_medium_bps,
                obs.pool_a_oracle_gap_bps.min(obs.pool_b_oracle_gap_bps),
                (obs.pool_a_fee_buy - obs.pool_b_fee_buy) * 10_000.0,
                action,
            ));
        }
        env.step(action);
    }
    if let Some(log) = action_log {
        log.extend(rows);
    }
    env.summary(policy.name())
}

fn mean_shortfall(rows: &[EpisodeSummary]) -> f64 {
    rows.iter().map(|r| r.shortfall_bps).sum::<f64>() / rows.len() as f64
}

fn main() {
    let args = Args::parse();
    let horizon = EnvConfig::baseline(MarketMode::DynamicDuopoly, 0)
        .order
        .horizon;
    let mut all_rows: Vec<EpisodeSummary> = Vec::new();
    let mut action_rows: Vec<String> = Vec::new();

    for (mode, mode_name) in MODES {
        // --- train ---
        let mut env = ExecEnv::new(EnvConfig::baseline(mode, 0));
        let tcfg = TrainConfig {
            n_episodes: args.train_episodes,
            ..TrainConfig::default()
        };
        let spec = if args.spec == "fine" {
            StateSpec::Fine
        } else {
            StateSpec::Coarse
        };
        let t0 = std::time::Instant::now();
        let table = train(&mut env, &tcfg, QTable::with_spec(mode_name, spec));
        eprintln!(
            "[{mode_name}] trained {} episodes ({:?}) in {:.1}s",
            args.train_episodes,
            spec,
            t0.elapsed().as_secs_f64()
        );
        let suffix = if args.spec == "fine" { "_fine" } else { "" };
        table
            .save(&format!(
                "{}/q_table_{}{}.json",
                args.out_dir, mode_name, suffix
            ))
            .expect("save q table");

        // --- tune lookahead kappa on validation seeds ---
        let mut best_kappa = KAPPA_GRID[0];
        let mut best_val = f64::INFINITY;
        for &kappa in &KAPPA_GRID {
            let rows: Vec<EpisodeSummary> = (VAL_BASE..VAL_BASE + args.n_val)
                .map(|seed| {
                    let mut p = LookaheadPolicy {
                        horizon,
                        kappa,
                        unfinished_penalty: 0.02,
                    };
                    run_episode(EnvConfig::baseline(mode, seed), &mut p, None)
                })
                .collect();
            let m = mean_shortfall(&rows);
            eprintln!("[{mode_name}] lookahead kappa={kappa}: val IS {m:.2} bps");
            if m < best_val {
                best_val = m;
                best_kappa = kappa;
            }
        }
        eprintln!("[{mode_name}] selected kappa={best_kappa}");

        // --- evaluate on held-out test seeds ---
        let log_actions = mode == MarketMode::DynamicDuopoly;
        for seed in TEST_BASE..TEST_BASE + args.n_test {
            let mut policies: Vec<Box<dyn ExecutionPolicy>> = vec![
                Box::new(ImmediatePolicy),
                Box::new(TwapPolicy { horizon }),
                Box::new(MyopicRouterPolicy),
                Box::new(RandomPolicy::new(seed ^ 0xBEEF)),
                Box::new(FeeAwareTwapPolicy { horizon }),
                Box::new(LookaheadPolicy {
                    horizon,
                    kappa: best_kappa,
                    unfinished_penalty: 0.02,
                }),
                Box::new(QPolicy {
                    table: table.clone(),
                    horizon,
                }),
            ];
            for policy in policies.iter_mut() {
                let want_log = log_actions
                    && matches!(
                        policy.name(),
                        "q_learner" | "q_learner_fine" | "fee_aware_twap" | "lookahead" | "twap"
                    );
                let log = if want_log {
                    Some(&mut action_rows)
                } else {
                    None
                };
                all_rows.push(run_episode(
                    EnvConfig::baseline(mode, seed),
                    policy.as_mut(),
                    log,
                ));
            }
        }
    }

    let results_path = format!("{}/{}_results.csv", args.out_dir, args.out_prefix);
    write_summaries_csv(&results_path, &all_rows).expect("write results");
    let actions_path = format!("{}/{}_actions.csv", args.out_dir, args.out_prefix);
    let mut f = std::fs::File::create(&actions_path).expect("create actions csv");
    writeln!(
        f,
        "policy,mode,seed,step,remaining_frac,remaining_time_frac,rival_quote_gap_bps,est_slippage_medium_bps,min_oracle_gap_bps,buy_fee_gap_bps,action"
    )
    .unwrap();
    for row in &action_rows {
        writeln!(f, "{row}").unwrap();
    }
    eprintln!("wrote {} rows to {results_path}", all_rows.len());
    eprintln!("wrote {} rows to {actions_path}", action_rows.len());

    // --- console summary: mean IS + paired diff vs strongest baseline ---
    for (mode, _name) in MODES {
        println!("\n=== {:?} (test, n={}) ===", mode, args.n_test);
        let learner_name = if args.spec == "fine" {
            "q_learner_fine"
        } else {
            "q_learner"
        };
        let policies = [
            "immediate",
            "twap",
            "myopic_router",
            "random",
            "fee_aware_twap",
            "lookahead",
            learner_name,
        ];
        let per_policy: Vec<(&str, Vec<&EpisodeSummary>)> = policies
            .iter()
            .map(|p| {
                (
                    *p,
                    all_rows
                        .iter()
                        .filter(|r| r.mode == mode && r.policy == *p)
                        .collect(),
                )
            })
            .collect();
        for (p, rows) in &per_policy {
            let n = rows.len() as f64;
            let is = rows.iter().map(|r| r.shortfall_bps).sum::<f64>() / n;
            let comp = rows.iter().map(|r| r.completion_rate).sum::<f64>() / n;
            let wait = rows.iter().map(|r| r.wait_share).sum::<f64>() / n;
            println!("{p:<16} IS {is:>8.2} bps   completion {comp:.4}   wait {wait:.3}");
        }
        // strongest baseline = min mean IS among non-learner policies
        let (best_name, best_rows) = per_policy
            .iter()
            .filter(|(p, _)| *p != learner_name)
            .min_by(|a, b| mean_is(&a.1).total_cmp(&mean_is(&b.1)))
            .unwrap();
        let q_rows = &per_policy.last().unwrap().1;
        let diffs: Vec<f64> = q_rows
            .iter()
            .zip(best_rows.iter())
            .map(|(q, b)| {
                assert_eq!(q.seed, b.seed);
                q.shortfall_bps - b.shortfall_bps
            })
            .collect();
        let n = diffs.len() as f64;
        let mean = diffs.iter().sum::<f64>() / n;
        let var = diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let t = mean / (var / n).sqrt();
        println!(
            "paired {learner_name} - {best_name}: mean {mean:+.2} bps, t = {t:.2} (negative favors learner)"
        );
    }
}

fn mean_is(rows: &[&EpisodeSummary]) -> f64 {
    rows.iter().map(|r| r.shortfall_bps).sum::<f64>() / rows.len() as f64
}
