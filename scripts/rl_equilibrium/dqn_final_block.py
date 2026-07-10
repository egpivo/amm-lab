"""M3R-F: final untouched paper seed block, 90_000-90_999.

All design choices are FROZEN before this script runs (see
m3r_run_manifest.json). Evaluates the frozen DQN checkpoints under the
frozen protocol (ForcedTerminal completion) at all three orderings: the
agent-last deterministic headline, agent-first, and randomized (the
worst-case stochastic ordering disclosed alongside the headline). Writes
out/m3r_final_paper_seeds.csv. Baseline rows come from
`rl_equilibrium_reference --seed-base 90000 --n-seeds 1000`.

Usage: python dqn_final_block.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from dqn_train import QNet, normalize, reset_env
from gym_env import AmmExecutionEnv

OUT = Path(__file__).resolve().parents[2] / "experiments/rl_execution/out"
FT = {"completion_rule": "forced_terminal"}
BASE, N = 90_000, 1_000

# (checkpoint tag, evaluation ordering) — each checkpoint is evaluated
# under the ordering it was trained for.
CELLS = [
    ("order_random", "random"),
    ("dynamic_duopoly", "before"),
    ("order_after", "after"),
]


def main() -> None:
    env = AmmExecutionEnv(mode="dynamic_duopoly")
    with open(OUT / "m3r_final_paper_seeds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "agent_order", "seed", "shortfall_bps",
                    "completion_rate", "forced_terminal_cost_bps", "wait_share"])
        for tag, order in CELLS:
            net = QNet(env.obs_dim, env.n_actions)
            net.load_state_dict(torch.load(OUT / f"dqn_{tag}.pt", weights_only=True))
            net.eval()
            agg = [0.0, 0.0]
            for seed in range(BASE, BASE + N):
                obs = reset_env(env, seed, "dynamic_duopoly",
                                agent_order=order, **FT)
                done, info = False, {}
                while not done:
                    with torch.no_grad():
                        a = int(net(normalize(obs)).argmax())
                    obs, _, done, _, info = env.step(a)
                s = info["summary"]
                w.writerow([f"dqn_{tag}", order, seed,
                            round(s["shortfall_bps"], 4),
                            round(s["completion_rate"], 6),
                            round(s["forced_terminal_cost_bps"], 4),
                            round(s["wait_share"], 4)])
                agg[0] += s["shortfall_bps"]
                agg[1] += s["completion_rate"]
            print(f"final dqn_{tag:<18} order={order:<7} IS {agg[0]/N:8.2f} "
                  f"comp {agg[1]/N:.4f}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
