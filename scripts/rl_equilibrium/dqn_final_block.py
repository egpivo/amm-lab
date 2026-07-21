"""baseline-duopoly-F: final untouched paper seed block, 90_000-90_999.

All design choices are FROZEN before this script runs (see
m3r_run_manifest.json). Evaluates the frozen DQN checkpoints under the
frozen protocol (ForcedTerminal completion) at all three orderings: the
agent-last deterministic headline, agent-first, and randomized (the
worst-case stochastic ordering disclosed alongside the headline). Writes
out/m3r_final_paper_seeds.csv. Baseline rows come from
`rl_equilibrium_reference --seed-base 90000 --n-seeds 1000`.

Usage: python dqn_final_block.py
"""

from common import OUT, write_csv
from dqn_core import greedy_episode, load_q_network
from gym_env import AmmExecutionEnv

FT = {"completion_rule": "forced_terminal"}
BASE, N = 90_000, 1_000

CELLS = [
    ("order_random", "random"),
    ("dynamic_duopoly", "before"),
    ("order_after", "after"),
]


def main() -> None:
    rows = []
    with AmmExecutionEnv(mode="dynamic_duopoly") as env:
        for tag, order in CELLS:
            net = load_q_network(env, tag)
            agg = [0.0, 0.0]
            for seed in range(BASE, BASE + N):
                summary = greedy_episode(
                    env,
                    net,
                    seed,
                    "dynamic_duopoly",
                    agent_order=order,
                    **FT,
                )
                rows.append(
                    {
                        "policy": f"dqn_{tag}",
                        "agent_order": order,
                        "seed": seed,
                        "shortfall_bps": round(summary["shortfall_bps"], 4),
                        "completion_rate": round(summary["completion_rate"], 6),
                        "forced_terminal_cost_bps": round(
                            summary["forced_terminal_cost_bps"], 4
                        ),
                        "wait_share": round(summary["wait_share"], 4),
                    }
                )
                agg[0] += summary["shortfall_bps"]
                agg[1] += summary["completion_rate"]
            print(
                f"final dqn_{tag:<18} order={order:<7} IS {agg[0] / N:8.2f} "
                f"comp {agg[1] / N:.4f}",
                flush=True,
            )
    write_csv(
        OUT / "m3r_final_paper_seeds.csv",
        [
            "policy",
            "agent_order",
            "seed",
            "shortfall_bps",
            "completion_rate",
            "forced_terminal_cost_bps",
            "wait_share",
        ],
        rows,
    )


if __name__ == "__main__":
    main()
