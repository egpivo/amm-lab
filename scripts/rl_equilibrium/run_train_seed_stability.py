"""Train R independent DQNs and evaluate forced-terminal edges on a fixed block."""
from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

from common import OUT, REPO_ROOT, write_csv
from dqn_core import QNet, greedy_episode, normalize
from gym_env import AmmExecutionEnv

FT = {"completion_rule": "forced_terminal"}
SEEDS = [11, 13, 17, 19, 23]
EVAL_SEEDS = range(30_000, 30_300)


def train_one(seed: int, episodes: int) -> Path:
    tag = f"trainseed_{seed}"
    cmd = [
        sys.executable,
        "dqn_train.py",
        "--episodes",
        str(episodes),
        "--n-val",
        "50",
        "--n-test",
        "1",  # skip large default eval; we re-eval with FT below
        "--tag",
        tag,
        "--seed",
        str(seed),
        "--mode",
        "dynamic_duopoly",
        "--agent-order",
        "before",
    ]
    log = OUT / "train_seed_stability" / f"train_{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"training seed={seed} -> {log}", flush=True)
    with log.open("w") as fh:
        subprocess.run(cmd, cwd=Path(__file__).resolve().parent, stdout=fh, stderr=subprocess.STDOUT, check=True)
    return OUT / f"dqn_{tag}.pt"


def eval_edge(ckpt: Path, seed_label: int) -> dict[str, float]:
    # Load lookahead means from reference for pairing
    ref_path = OUT / "m3r_reference.csv"
    la = {}
    with ref_path.open() as fh:
        for row in csv.DictReader(fh):
            if (
                row["policy"] == "lookahead"
                and row["mode"] == "dynamic_duopoly"
                and row["agent_order"] == "before"
                and int(row["seed"]) in EVAL_SEEDS
            ):
                la[int(row["seed"])] = float(row["shortfall_bps"])
    with AmmExecutionEnv(mode="dynamic_duopoly") as env:
        net = QNet(env.obs_dim, env.n_actions)
        net.load_state_dict(torch.load(ckpt, map_location="cpu"))
        net.eval()
        diffs = []
        dqn_vals = []
        for seed in EVAL_SEEDS:
            s = greedy_episode(env, net, seed, "dynamic_duopoly", agent_order="before", **FT)
            dqn_vals.append(s["shortfall_bps"])
            diffs.append(s["shortfall_bps"] - la[seed])
    import numpy as np

    diffs_a = np.array(diffs)
    return {
        "train_seed": float(seed_label),
        "dqn_mean": float(np.mean(dqn_vals)),
        "la_mean": float(np.mean([la[s] for s in EVAL_SEEDS])),
        "edge_mean": float(diffs_a.mean()),
        "edge_std_across_seeds": float(diffs_a.std(ddof=1)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()
    rows = []
    for seed in args.seeds:
        t0 = time.monotonic()
        ckpt = train_one(seed, args.episodes)
        row = eval_edge(ckpt, seed)
        row["train_seconds"] = time.monotonic() - t0
        print(row, flush=True)
        rows.append(row)
        write_csv(
            OUT / "train_seed_stability" / "summary.csv",
            list(rows[0].keys()),
            rows,
        )
    edges = [r["edge_mean"] for r in rows]
    print(
        "SUMMARY edges",
        edges,
        "mean",
        sum(edges) / len(edges),
        "min",
        min(edges),
        "max",
        max(edges),
        flush=True,
    )


if __name__ == "__main__":
    main()
