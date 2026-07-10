"""M3R figures: priority train x test heatmap, completion-rule comparison,
final-block summary. Reads m3r_*.csv; writes PNGs to out/.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2] / "experiments/rl_execution/out"
ORDERS = ["before", "random", "after"]


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fig_priority_heatmap() -> None:
    rows = read(OUT / "m3r_priority.csv")
    ref = read(OUT / "m3r_reference.csv")
    la = defaultdict(list)
    for r in ref:
        if (r["policy"] == "lookahead" and r["mode"] == "dynamic_duopoly"
                and r["seed_set"] == "test"):
            la[r["agent_order"]].append(float(r["shortfall_bps"]))
    la_mean = {o: sum(v) / len(v) for o, v in la.items()}

    cells = defaultdict(list)
    for r in rows:
        if r["seed_set"] == "test":
            cells[(r["train_order"], r["test_order"])].append(float(r["shortfall_bps"]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, mode in zip(axes, ["absolute", "edge"]):
        grid = []
        for tr in ORDERS:
            row = []
            for te in ORDERS:
                v = sum(cells[(tr, te)]) / len(cells[(tr, te)])
                row.append(v if mode == "absolute" else v - la_mean[te])
            grid.append(row)
        vmax = max(abs(x) for row in grid for x in row)
        im = ax.imshow(grid, cmap="RdBu_r" if mode == "edge" else "Blues",
                       vmin=-vmax if mode == "edge" else None,
                       vmax=vmax if mode == "edge" else None)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{grid[i][j]:+.1f}" if mode == "edge" else f"{grid[i][j]:.1f}",
                        ha="center", va="center", fontsize=10,
                        color="#1f1f1f")
        ax.set_xticks(range(3), [f"test: {o}" for o in ORDERS], fontsize=9)
        ax.set_yticks(range(3), [f"train: {o}" for o in ORDERS], fontsize=9)
        ax.set_title("DQN shortfall (bps)" if mode == "absolute"
                     else "DQN - lookahead(same ordering) (bps, negative = DQN wins)",
                     fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("M3R-B: priority retraining, ForcedTerminal, 300 test seeds")
    fig.tight_layout()
    fig.savefig(OUT / "m3r_priority_heatmap.png", dpi=150)
    plt.close(fig)


def fig_completion() -> None:
    rows = read(OUT / "m3r_completion.csv")
    cells = defaultdict(list)
    for r in rows:
        if r["seed_set"] == "test":
            cells[(r["policy"], r["completion_rule"])].append(float(r["shortfall_bps"]))
    policies = ["twap", "fee_aware_twap", "lookahead", "dqn_unconstrained",
                "dqn_completion_aware"]
    colors = {"standard": "#9ecae1", "forced_terminal": "#4269d0"}
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for k, rule in enumerate(["standard", "forced_terminal"]):
        xs = [i + (k - 0.5) * 0.36 for i in range(len(policies))]
        means = [sum(cells[(p, rule)]) / max(1, len(cells[(p, rule)])) for p in policies]
        ax.bar(xs, means, 0.34, color=colors[rule], label=rule)
        for x, m in zip(xs, means):
            ax.text(x, m + 1, f"{m:.0f}", ha="center", fontsize=8, color="#333333")
    ax.set_xticks(range(len(policies)), policies, fontsize=9)
    ax.set_ylabel("shortfall (bps, test mean)")
    ax.set_title("M3R-A: standard vs forced-terminal completion (500 test seeds)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "m3r_completion.png", dpi=150)
    plt.close(fig)


def main() -> None:
    fig_priority_heatmap()
    fig_completion()
    print("wrote m3r_priority_heatmap.png, m3r_completion.png")


if __name__ == "__main__":
    main()
