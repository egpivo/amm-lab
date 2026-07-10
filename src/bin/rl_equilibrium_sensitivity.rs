//! M4: baseline rows (lookahead, twap) for the LP-adaptation and JIT
//! sensitivity grids. DQN rows are appended by dqn_m4_eval.py.
//!
//! Reserved seed blocks (never used before M4): LP 95_000-95_499,
//! JIT 96_000-96_499. ForcedTerminal completion, agent-first ordering.

use amm_lab::sim::env::{CompletionRule, EnvConfig, ExecEnv, JitRegime, LpRegime, MarketMode};
use amm_lab::sim::execution_agent::{ExecutionPolicy, LookaheadPolicy, TwapPolicy};
use clap::Parser;
use std::io::Write as _;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 500)]
    n_seeds: u64,
    #[arg(long, default_value = "experiments/rl_execution/out")]
    out_dir: String,
}

pub const HEADER: &str = "extension,regime,policy,seed,shortfall_bps,completion_rate,fee_bps,gas_bps,slippage_ex_fee_bps,drift_bps,forced_terminal_cost_bps,route_share_A,route_share_B,wait_share,avg_depth_factor,min_depth_factor,jit_event_count";

fn run_rows(
    f: &mut std::fs::File,
    extension: &str,
    regime_name: &str,
    mutate: &dyn Fn(&mut EnvConfig),
    seed_base: u64,
    n_seeds: u64,
    horizon: usize,
) {
    let mut sums: std::collections::BTreeMap<&str, f64> = Default::default();
    for seed in seed_base..seed_base + n_seeds {
        let mut cfg = EnvConfig::baseline(MarketMode::DynamicDuopoly, seed);
        cfg.completion_rule = CompletionRule::ForcedTerminal;
        mutate(&mut cfg);
        let mut la = LookaheadPolicy {
            horizon,
            kappa: 16.0,
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
                "{extension},{regime_name},{name},{seed},{:.4},{:.6},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{}",
                s.shortfall_bps, s.completion_rate, s.fee_paid_bps, s.gas_paid_bps,
                s.slippage_ex_fee_bps, s.drift_bps, s.forced_terminal_cost_bps,
                s.route_share_a, s.route_share_b, s.wait_share,
                s.avg_depth_factor, s.min_depth_factor, s.jit_event_count
            )
            .unwrap();
            *sums.entry(name).or_insert(0.0) += s.shortfall_bps;
        }
    }
    println!(
        "{extension}/{regime_name}: lookahead {:.2}  twap {:.2}",
        sums["lookahead"] / n_seeds as f64,
        sums["twap"] / n_seeds as f64
    );
}

fn main() {
    let args = Args::parse();
    let horizon = EnvConfig::baseline(MarketMode::DynamicDuopoly, 0)
        .order
        .horizon;

    let mut f_lp = std::fs::File::create(format!("{}/m4_lp_adaptation.csv", args.out_dir)).unwrap();
    writeln!(f_lp, "{HEADER}").unwrap();
    for (name, regime) in [
        ("frozen", LpRegime::Frozen),
        ("weak", LpRegime::Weak),
        ("aggressive", LpRegime::Aggressive),
    ] {
        run_rows(
            &mut f_lp,
            "lp",
            name,
            &|c: &mut EnvConfig| c.lp_regime = regime,
            95_000,
            args.n_seeds,
            horizon,
        );
    }

    let mut f_jit = std::fs::File::create(format!("{}/m4_jit_mev.csv", args.out_dir)).unwrap();
    writeln!(f_jit, "{HEADER}").unwrap();
    for (name, regime) in [
        ("none", JitRegime::None),
        ("weak", JitRegime::Weak),
        ("aggressive", JitRegime::Aggressive),
    ] {
        run_rows(
            &mut f_jit,
            "jit",
            name,
            &|c: &mut EnvConfig| c.jit_regime = regime,
            96_000,
            args.n_seeds,
            horizon,
        );
    }
    println!("wrote m4_lp_adaptation.csv, m4_jit_mev.csv");
}
