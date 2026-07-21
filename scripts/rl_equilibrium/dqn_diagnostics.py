"""validation-grid: artifact diagnostics for the frozen DQN checkpoint.

Evaluates the selected checkpoint greedily under perturbed environments on
the same 300 test seeds as rl_equilibrium_artifact_battery, writing rows compatible with
m2_diagnostics.csv so lookahead/twap references can be reused.

Usage: python dqn_diagnostics.py [--n-seeds 300]
"""

import argparse
from common import OUT, write_csv
from dqn_core import greedy_episode, load_q_network
from gym_env import AmmExecutionEnv

CELLS = [
    ("priority", "before", {}),
    ("priority", "after", {"agent_order": "after"}),
    ("priority", "random", {"agent_order": "random"}),
    ("gas", "gas=0", {"gas": 0.0}),
    ("gas", "gas=2", {"gas": 2.0}),
    ("gas", "gas=10", {"gas": 10.0}),
    ("arb_speed", "speed=0.2", {"arb_speed": 0.2}),
    ("arb_speed", "speed=0.5", {"arb_speed": 0.5}),
    ("arb_speed", "speed=1", {"arb_speed": 1.0}),
    ("noise", "base", {}),
    ("noise", "intensity_x2", {"noise_intensity_scale": 2.0}),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=300)
    args = ap.parse_args()

    rows = []
    with AmmExecutionEnv(mode="dynamic_duopoly") as env:
        net = load_q_network(env, "dynamic_duopoly")
        for diag, variant, overrides in CELLS:
            total_is, total_comp = 0.0, 0.0
            for seed in range(30_000, 30_000 + args.n_seeds):
                summary = greedy_episode(env, net, seed, "dynamic_duopoly", **overrides)
                rows.append(
                    {
                        "diagnostic": diag,
                        "variant": variant,
                        "policy": "dqn",
                        "seed": seed,
                        "shortfall_bps": round(summary["shortfall_bps"], 4),
                        "completion_rate": round(summary["completion_rate"], 4),
                    }
                )
                total_is += summary["shortfall_bps"]
                total_comp += summary["completion_rate"]
            print(
                f"{diag:<10} {variant:<16} dqn IS {total_is / args.n_seeds:8.2f} "
                f"comp {total_comp / args.n_seeds:.4f}",
                flush=True,
            )
    write_csv(
        OUT / "m3_dqn_diagnostics.csv",
        ["diagnostic", "variant", "policy", "seed", "shortfall_bps", "completion_rate"],
        rows,
    )


if __name__ == "__main__":
    main()
