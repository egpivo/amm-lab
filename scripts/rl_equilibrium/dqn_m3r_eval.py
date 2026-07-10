"""M3R evaluation matrices for the trained DQN checkpoints.

Produces (in out/):
- DQN rows appended to m3r_completion.csv  (A: rules x {unconstrained, completion-aware})
- m3r_priority.csv                          (B: train-order x test-order matrix)
- m3r_dynamic_fee_ablation.csv              (C: mode-trained + cross-mode transfer)
- m3r_symmetry.csv                          (G: pool-label swap)

All evaluations use ForcedTerminal completion unless the cell says otherwise
(completion.csv covers both rules). Reference lookahead/twap rows live in
m3r_reference.csv / m3r_completion.csv (Rust-generated).

Usage: python dqn_m3r_eval.py [--n-seeds 300] [--n-completion 500]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from dqn_train import QNet, normalize, reset_env
from gym_env import AmmExecutionEnv

OUT = Path(__file__).resolve().parents[2] / "experiments/rl_execution/out"
FT = {"completion_rule": "forced_terminal"}

# to_vec index map for pool-label swap (see Observation::to_vec):
# [rem, time, lnp, gapA, gapB, rivalgap, feeAb, feeAs, feeBb, feeBs,
#  vol, slipS, slipM, slipL, gas, prev]
SWAP_IDX = [0, 1, 2, 4, 3, 5, 8, 9, 6, 7, 10, 11, 12, 13, 14, 15]
NEGATE_IDX = {5}  # rival gap = A - B flips sign
ACTION_SWAP = {0: 0, 1: 2, 2: 1, 3: 4, 4: 3, 5: 6, 6: 5, 7: 7}


def swap_obs(obs: list[float]) -> list[float]:
    out = [obs[i] for i in SWAP_IDX]
    for i in NEGATE_IDX:
        out[i] = -out[i]
    # prev_action feature: map through the action swap
    prev = round(obs[15] * 8)
    out[15] = ACTION_SWAP.get(prev, prev) / 8.0
    return out


def load_net(env, tag: str) -> QNet:
    net = QNet(env.obs_dim, env.n_actions)
    net.load_state_dict(torch.load(OUT / f"dqn_{tag}.pt", weights_only=True))
    net.eval()
    return net


def episode(env, net, seed, mode, swap=False, **ov) -> dict:
    obs = reset_env(env, seed, mode, **ov)
    done, info = False, {}
    while not done:
        x = swap_obs(obs) if swap else obs
        with torch.no_grad():
            a = int(net(normalize(x)).argmax())
        if swap:
            a = ACTION_SWAP[a]
        obs, _, done, _, info = env.step(a)
    return info["summary"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=300)
    ap.add_argument("--n-completion", type=int, default=500)
    args = ap.parse_args()
    env = AmmExecutionEnv(mode="dynamic_duopoly")

    # ---------- A: completion (append DQN rows to m3r_completion.csv) ----------
    # idempotency guard: refuse to append twice (rerun rl_equilibrium_completion first)
    with open(OUT / "m3r_completion.csv") as f:
        if any(",dqn" in line for line in f):
            raise SystemExit("m3r_completion.csv already contains DQN rows; "
                             "regenerate it with rl_equilibrium_completion before appending")
    with open(OUT / "m3r_completion.csv", "a") as f:
        for tag, policy in [("dynamic_duopoly", "dqn_unconstrained"),
                            ("completion_aware", "dqn_completion_aware")]:
            net = load_net(env, tag)
            for rule_name, ov in [("standard", {}), ("forced_terminal", FT)]:
                agg = [0.0, 0.0]
                for label, base in [("test", 30_000), ("fresh", 40_000)]:
                    for seed in range(base, base + args.n_completion):
                        s = episode(env, net, seed, "dynamic_duopoly", **ov)
                        f.write(
                            f"{seed},{label},DynamicDuopoly,{policy},{rule_name},"
                            f"{s['shortfall_bps']:.4f},{s['completion_rate']:.6f},"
                            f"{s['terminal_penalty_bps']:.4f},{s['forced_terminal_cost_bps']:.4f},"
                            f"{s['fee_paid_bps']:.4f},{s['gas_paid_bps']:.4f},"
                            f"{s['slippage_ex_fee_bps']:.4f},{s['drift_bps']:.4f},"
                            f"{s['route_share_a']:.4f},{s['route_share_b']:.4f},"
                            f"{s['wait_share']:.4f}\n")
                        if label == "test":
                            agg[0] += s["shortfall_bps"]
                            agg[1] += s["completion_rate"]
                print(f"A {policy:<24} {rule_name:<16} test IS "
                      f"{agg[0]/args.n_completion:8.2f} comp {agg[1]/args.n_completion:.4f}",
                      flush=True)

    # ---------- B: priority matrix ----------
    with open(OUT / "m3r_priority.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["train_order", "test_order", "policy", "seed_set", "seed",
                    "shortfall_bps", "completion_rate"])
        for train_order, tag in [("before", "dynamic_duopoly"),
                                 ("random", "order_random"),
                                 ("after", "order_after")]:
            net = load_net(env, tag)
            for test_order in ["before", "random", "after"]:
                agg = [0.0, 0.0]
                for label, base in [("test", 30_000), ("fresh", 40_000)]:
                    for seed in range(base, base + args.n_seeds):
                        s = episode(env, net, seed, "dynamic_duopoly",
                                    agent_order=test_order, **FT)
                        w.writerow([train_order, test_order, f"dqn_{tag}", label, seed,
                                    round(s["shortfall_bps"], 4),
                                    round(s["completion_rate"], 6)])
                        if label == "test":
                            agg[0] += s["shortfall_bps"]
                            agg[1] += s["completion_rate"]
                print(f"B train={train_order:<7} test={test_order:<7} IS "
                      f"{agg[0]/args.n_seeds:8.2f} comp {agg[1]/args.n_seeds:.4f}",
                      flush=True)

    # ---------- C: dynamic-fee ablation + cross-mode transfer ----------
    with open(OUT / "m3r_dynamic_fee_ablation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["train_mode", "test_mode", "policy", "seed_set", "seed",
                    "shortfall_bps", "completion_rate"])
        cells = [("constant_duopoly", "constant_duopoly"),
                 ("dynamic_monopoly", "dynamic_monopoly"),
                 ("dynamic_duopoly", "dynamic_duopoly"),
                 ("dynamic_duopoly", "constant_duopoly"),
                 ("constant_duopoly", "dynamic_duopoly")]
        for train_mode, test_mode in cells:
            net = load_net(env, train_mode)
            agg = [0.0, 0.0]
            for label, base in [("test", 30_000), ("fresh", 40_000)]:
                for seed in range(base, base + args.n_seeds):
                    s = episode(env, net, seed, test_mode, **FT)
                    w.writerow([train_mode, test_mode, f"dqn_{train_mode}", label, seed,
                                round(s["shortfall_bps"], 4),
                                round(s["completion_rate"], 6)])
                    if label == "test":
                        agg[0] += s["shortfall_bps"]
                        agg[1] += s["completion_rate"]
            print(f"C train={train_mode:<18} test={test_mode:<18} IS "
                  f"{agg[0]/args.n_seeds:8.2f} comp {agg[1]/args.n_seeds:.4f}",
                  flush=True)

    # ---------- G: pool-label symmetry ----------
    with open(OUT / "m3r_symmetry.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "seed", "shortfall_bps", "completion_rate",
                    "route_share_a"])
        net = load_net(env, "dynamic_duopoly")
        for variant, swap in [("original", False), ("labels_swapped", True)]:
            agg = [0.0, 0.0, 0.0]
            for seed in range(30_000, 30_000 + args.n_seeds):
                s = episode(env, net, seed, "dynamic_duopoly", swap=swap, **FT)
                w.writerow([variant, seed, round(s["shortfall_bps"], 4),
                            round(s["completion_rate"], 6),
                            round(s["route_share_a"], 4)])
                agg[0] += s["shortfall_bps"]
                agg[1] += s["completion_rate"]
                agg[2] += s["route_share_a"]
            print(f"G {variant:<16} IS {agg[0]/args.n_seeds:8.2f} "
                  f"comp {agg[1]/args.n_seeds:.4f} routeA {agg[2]/args.n_seeds:.3f}",
                  flush=True)
    env.close()


if __name__ == "__main__":
    main()
