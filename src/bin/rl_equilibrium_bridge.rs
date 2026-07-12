//! Gymnasium-style JSON bridge over stdin/stdout.
//!
//! Protocol (one JSON object per line):
//!   -> {"cmd": "reset", "seed": 7, "mode": "dynamic_duopoly"}
//!   <- {"type": "state", "obs": [...], "raw": {...}}
//!   -> {"cmd": "step", "action": 3}
//!   <- {"type": "transition", "obs": [...], "reward": -1.2e-4, "done": false, "raw": {...}}
//!      (when done, a "summary" field is included)
//!   -> {"cmd": "close"}
//!
//! On startup emits {"type": "hello", "n_actions": 8, "obs_dim": D}.
//! See scripts/rl_equilibrium/gym_env.py for the Python wrapper.

use amm_lab::sim::env::{EnvConfig, ExecEnv, MarketMode, N_ACTIONS};
use serde_json::{Value, json};
use std::io::{BufRead, Write};

fn parse_mode(s: &str) -> MarketMode {
    match s {
        "constant_duopoly" => MarketMode::ConstantDuopoly,
        "dynamic_monopoly" => MarketMode::DynamicMonopoly,
        _ => MarketMode::DynamicDuopoly,
    }
}

fn state_msg(env: &ExecEnv, kind: &str, reward: Option<f64>, done: bool, policy: &str) -> Value {
    let obs = env.observe();
    let mut msg = json!({
        "type": kind,
        "obs": obs.to_vec(),
        "raw": serde_json::to_value(&obs).unwrap(),
    });
    if let Some(r) = reward {
        msg["reward"] = json!(r);
        msg["done"] = json!(done);
    }
    if done {
        msg["summary"] = serde_json::to_value(env.summary(policy)).unwrap();
    }
    msg
}

fn main() {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout().lock();

    let mut env = ExecEnv::new(EnvConfig::baseline(MarketMode::DynamicDuopoly, 0));
    let obs_dim = env.observe().to_vec().len();
    writeln!(
        stdout,
        "{}",
        json!({"type": "hello", "n_actions": N_ACTIONS, "obs_dim": obs_dim})
    )
    .unwrap();
    stdout.flush().unwrap();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) if !l.trim().is_empty() => l,
            Ok(_) => continue,
            Err(_) => break,
        };
        let req: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                writeln!(
                    stdout,
                    "{}",
                    json!({"type": "error", "message": e.to_string()})
                )
                .unwrap();
                stdout.flush().unwrap();
                continue;
            }
        };
        let reply = match req["cmd"].as_str() {
            Some("reset") => {
                let seed = req["seed"].as_u64().unwrap_or(0);
                let mode = parse_mode(req["mode"].as_str().unwrap_or("dynamic_duopoly"));
                let mut cfg = EnvConfig::baseline(mode, seed);
                if let Some(sigma) = req["sigma"].as_f64() {
                    cfg.sigma = sigma;
                }
                if let Some(speed) = req["arb_speed"].as_f64() {
                    cfg.arb.speed = speed;
                }
                if let Some(gas) = req["gas"].as_f64() {
                    cfg.agent_gas_cost = gas;
                }
                if let Some(order) = req["agent_order"].as_str() {
                    cfg.agent_order = match order {
                        "after" => amm_lab::sim::env::AgentOrder::After,
                        "random" => amm_lab::sim::env::AgentOrder::Random,
                        _ => amm_lab::sim::env::AgentOrder::Before,
                    };
                }
                if let Some(scale) = req["noise_intensity_scale"].as_f64() {
                    cfg.noise.buy_intensity *= scale;
                    cfg.noise.sell_intensity *= scale;
                }
                if let Some(p) = req["unfinished_penalty"].as_f64() {
                    cfg.unfinished_penalty = p;
                }
                if req["completion_rule"].as_str() == Some("forced_terminal") {
                    cfg.completion_rule = amm_lab::sim::env::CompletionRule::ForcedTerminal;
                }
                if let Some(r) = req["lp_regime"].as_str() {
                    cfg.lp_regime = match r {
                        "weak" => amm_lab::sim::env::LpRegime::Weak,
                        "aggressive" => amm_lab::sim::env::LpRegime::Aggressive,
                        _ => amm_lab::sim::env::LpRegime::Frozen,
                    };
                }
                if let Some(r) = req["jit_regime"].as_str() {
                    cfg.jit_regime = match r {
                        "weak" => amm_lab::sim::env::JitRegime::Weak,
                        "aggressive" => amm_lab::sim::env::JitRegime::Aggressive,
                        _ => amm_lab::sim::env::JitRegime::None,
                    };
                }
                env = ExecEnv::new(cfg);
                state_msg(&env, "state", None, false, "rl")
            }
            Some("step") => {
                if env.is_done() {
                    json!({"type": "error", "message": "episode done; send reset"})
                } else {
                    let action = req["action"].as_u64().unwrap_or(0) as usize;
                    let res = env.step(action);
                    state_msg(&env, "transition", Some(res.reward), res.done, "rl")
                }
            }
            Some("close") => break,
            _ => json!({"type": "error", "message": "unknown cmd"}),
        };
        writeln!(stdout, "{reply}").unwrap();
        stdout.flush().unwrap();
    }
}
