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

import argparse

from common import OUT, append_csv_rows, read_csv, write_csv
from dqn_core import QNet, greedy_episode, load_q_network
from gym_env import AmmExecutionEnv

FT = {"completion_rule": "forced_terminal"}

SWAP_IDX = [0, 1, 2, 4, 3, 5, 8, 9, 6, 7, 10, 11, 12, 13, 14, 15]
NEGATE_IDX = {5}
ACTION_SWAP = {0: 0, 1: 2, 2: 1, 3: 4, 4: 3, 5: 6, 6: 5, 7: 7}


def swap_obs(obs: list[float]) -> list[float]:
    out = [obs[i] for i in SWAP_IDX]
    for i in NEGATE_IDX:
        out[i] = -out[i]
    prev = round(obs[15] * 8)
    out[15] = ACTION_SWAP.get(prev, prev) / 8.0
    return out


def episode(
    env: AmmExecutionEnv, net: QNet, seed: int, mode: str, swap: bool = False, **ov
) -> dict:
    return greedy_episode(
        env,
        net,
        seed,
        mode,
        observation_transform=swap_obs if swap else None,
        action_transform=ACTION_SWAP.__getitem__ if swap else None,
        **ov,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=300)
    ap.add_argument("--n-completion", type=int, default=500)
    args = ap.parse_args()
    with AmmExecutionEnv(mode="dynamic_duopoly") as env:
        completion_path = OUT / "m3r_completion.csv"
        if any(row["policy"].startswith("dqn") for row in read_csv(completion_path)):
            raise SystemExit(
                "m3r_completion.csv already contains DQN rows; "
                "regenerate it with rl_equilibrium_completion before appending"
            )
        completion_rows = []
        for tag, policy in [
            ("dynamic_duopoly", "dqn_unconstrained"),
            ("completion_aware", "dqn_completion_aware"),
        ]:
            net = load_q_network(env, tag)
            for rule_name, ov in [("standard", {}), ("forced_terminal", FT)]:
                agg = [0.0, 0.0]
                for label, base in [("test", 30_000), ("fresh", 40_000)]:
                    for seed in range(base, base + args.n_completion):
                        s = episode(env, net, seed, "dynamic_duopoly", **ov)
                        completion_rows.append(
                            [
                                seed,
                                label,
                                "DynamicDuopoly",
                                policy,
                                rule_name,
                                f"{s['shortfall_bps']:.4f}",
                                f"{s['completion_rate']:.6f}",
                                f"{s['terminal_penalty_bps']:.4f}",
                                f"{s['forced_terminal_cost_bps']:.4f}",
                                f"{s['fee_paid_bps']:.4f}",
                                f"{s['gas_paid_bps']:.4f}",
                                f"{s['slippage_ex_fee_bps']:.4f}",
                                f"{s['drift_bps']:.4f}",
                                f"{s['route_share_a']:.4f}",
                                f"{s['route_share_b']:.4f}",
                                f"{s['wait_share']:.4f}",
                            ]
                        )
                        if label == "test":
                            agg[0] += s["shortfall_bps"]
                            agg[1] += s["completion_rate"]
                print(
                    f"A {policy:<24} {rule_name:<16} test IS "
                    f"{agg[0] / args.n_completion:8.2f} comp {agg[1] / args.n_completion:.4f}",
                    flush=True,
                )
        append_csv_rows(completion_path, completion_rows)

        priority_rows = []
        for train_order, tag in [
            ("before", "dynamic_duopoly"),
            ("random", "order_random"),
            ("after", "order_after"),
        ]:
            net = load_q_network(env, tag)
            for test_order in ["before", "random", "after"]:
                agg = [0.0, 0.0]
                for label, base in [("test", 30_000), ("fresh", 40_000)]:
                    for seed in range(base, base + args.n_seeds):
                        s = episode(
                            env,
                            net,
                            seed,
                            "dynamic_duopoly",
                            agent_order=test_order,
                            **FT,
                        )
                        priority_rows.append(
                            {
                                "train_order": train_order,
                                "test_order": test_order,
                                "policy": f"dqn_{tag}",
                                "seed_set": label,
                                "seed": seed,
                                "shortfall_bps": round(s["shortfall_bps"], 4),
                                "completion_rate": round(s["completion_rate"], 6),
                            }
                        )
                        if label == "test":
                            agg[0] += s["shortfall_bps"]
                            agg[1] += s["completion_rate"]
                print(
                    f"B train={train_order:<7} test={test_order:<7} IS "
                    f"{agg[0] / args.n_seeds:8.2f} comp {agg[1] / args.n_seeds:.4f}",
                    flush=True,
                )
        write_csv(
            OUT / "m3r_priority.csv",
            [
                "train_order",
                "test_order",
                "policy",
                "seed_set",
                "seed",
                "shortfall_bps",
                "completion_rate",
            ],
            priority_rows,
        )

        ablation_rows = []
        cells = [
            ("constant_duopoly", "constant_duopoly"),
            ("dynamic_monopoly", "dynamic_monopoly"),
            ("dynamic_duopoly", "dynamic_duopoly"),
            ("dynamic_duopoly", "constant_duopoly"),
            ("constant_duopoly", "dynamic_duopoly"),
        ]
        for train_mode, test_mode in cells:
            net = load_q_network(env, train_mode)
            agg = [0.0, 0.0]
            for label, base in [("test", 30_000), ("fresh", 40_000)]:
                for seed in range(base, base + args.n_seeds):
                    s = episode(env, net, seed, test_mode, **FT)
                    ablation_rows.append(
                        {
                            "train_mode": train_mode,
                            "test_mode": test_mode,
                            "policy": f"dqn_{train_mode}",
                            "seed_set": label,
                            "seed": seed,
                            "shortfall_bps": round(s["shortfall_bps"], 4),
                            "completion_rate": round(s["completion_rate"], 6),
                        }
                    )
                    if label == "test":
                        agg[0] += s["shortfall_bps"]
                        agg[1] += s["completion_rate"]
            print(
                f"C train={train_mode:<18} test={test_mode:<18} IS "
                f"{agg[0] / args.n_seeds:8.2f} comp {agg[1] / args.n_seeds:.4f}",
                flush=True,
            )
        write_csv(
            OUT / "m3r_dynamic_fee_ablation.csv",
            [
                "train_mode",
                "test_mode",
                "policy",
                "seed_set",
                "seed",
                "shortfall_bps",
                "completion_rate",
            ],
            ablation_rows,
        )

        symmetry_rows = []
        net = load_q_network(env, "dynamic_duopoly")
        for variant, swap in [("original", False), ("labels_swapped", True)]:
            agg = [0.0, 0.0, 0.0]
            for seed in range(30_000, 30_000 + args.n_seeds):
                s = episode(env, net, seed, "dynamic_duopoly", swap=swap, **FT)
                symmetry_rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "shortfall_bps": round(s["shortfall_bps"], 4),
                        "completion_rate": round(s["completion_rate"], 6),
                        "route_share_a": round(s["route_share_a"], 4),
                    }
                )
                agg[0] += s["shortfall_bps"]
                agg[1] += s["completion_rate"]
                agg[2] += s["route_share_a"]
            print(
                f"G {variant:<16} IS {agg[0] / args.n_seeds:8.2f} "
                f"comp {agg[1] / args.n_seeds:.4f} routeA {agg[2] / args.n_seeds:.3f}",
                flush=True,
            )
        write_csv(
            OUT / "m3r_symmetry.csv",
            ["variant", "seed", "shortfall_bps", "completion_rate", "route_share_a"],
            symmetry_rows,
        )


if __name__ == "__main__":
    main()
