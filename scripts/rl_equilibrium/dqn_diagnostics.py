"""M3: artifact diagnostics for the frozen DQN checkpoint.

Evaluates the selected checkpoint greedily under perturbed environments on
the same 300 test seeds as run_m2_diagnostics, writing rows compatible with
m2_diagnostics.csv so lookahead/twap references can be reused.

Usage: python dqn_diagnostics.py [--n-seeds 300]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from dqn_train import QNet, normalize
from gym_env import AmmExecutionEnv

OUT = Path(__file__).resolve().parents[2] / "experiments/rl_execution/out"

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

    env = AmmExecutionEnv(mode="dynamic_duopoly")
    net = QNet(env.obs_dim, env.n_actions)
    net.load_state_dict(torch.load(OUT / "dqn_dynamic_duopoly.pt", weights_only=True))
    net.eval()

    with open(OUT / "m3_dqn_diagnostics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["diagnostic", "variant", "policy", "seed", "shortfall_bps",
                    "completion_rate"])
        for diag, variant, overrides in CELLS:
            total_is, total_comp = 0.0, 0.0
            for seed in range(30_000, 30_000 + args.n_seeds):
                obs, _ = env.reset(seed=seed, mode="dynamic_duopoly", **overrides)
                done, info = False, {}
                while not done:
                    with torch.no_grad():
                        a = int(net(normalize(obs)).argmax())
                    obs, _, done, _, info = env.step(a)
                s = info["summary"]
                w.writerow([diag, variant, "dqn", seed,
                            round(s["shortfall_bps"], 4), round(s["completion_rate"], 4)])
                total_is += s["shortfall_bps"]
                total_comp += s["completion_rate"]
            print(f"{diag:<10} {variant:<16} dqn IS {total_is/args.n_seeds:8.2f} "
                  f"comp {total_comp/args.n_seeds:.4f}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
