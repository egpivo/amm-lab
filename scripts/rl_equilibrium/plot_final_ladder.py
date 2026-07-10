"""Paper Figure 1: the benchmark ladder on the final untouched seed block
(90,000-90,999), forced-terminal completion, agent-first ordering.

Inputs: out/final_ladder.csv (Rust policies + clairvoyant),
out/m3r_stochastic_planner_final.csv, out/m3r_final_paper_seeds.csv (DQN).
Output: out/final_ladder.png.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2] / "experiments/rl_execution/out"
COLORS = {
    "twap": "#efb118",
    "two_step": "#9c6b4e",
    "stochastic_planner": "#6cc5b0",
    "three_step": "#3ca951",
    "lookahead": "#a463f2",
    "q_learner": "#ff8ab7",
    "q_learner_fine": "#ff725c",
    "dqn": "#4269d0",
    "clairvoyant": "#555555",
}
ORDER = ["twap", "two_step", "stochastic_planner", "three_step", "lookahead",
         "q_learner", "q_learner_fine", "dqn", "clairvoyant"]


def boot_ci(vals, n_boot=2000, seed=0):
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(rng.choices(vals, k=n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def main() -> None:
    vals: dict[str, list[float]] = {}
    for r in csv.DictReader(open(OUT / "final_ladder.csv")):
        vals.setdefault(r["policy"], []).append(float(r["shortfall_bps"]))
    for r in csv.DictReader(open(OUT / "m3r_stochastic_planner_final.csv")):
        if r["policy"] == "stochastic_planner":
            vals.setdefault("stochastic_planner", []).append(float(r["shortfall_bps"]))
    for r in csv.DictReader(open(OUT / "m3r_final_paper_seeds.csv")):
        if r["policy"] == "dqn_dynamic_duopoly" and r["agent_order"] == "before":
            vals.setdefault("dqn", []).append(float(r["shortfall_bps"]))

    fig, ax = plt.subplots(figsize=(9, 5))
    for p in ORDER:
        v = vals[p]
        m = sum(v) / len(v)
        lo, hi = boot_ci(v)
        ax.barh(p, m, color=COLORS[p], height=0.62)
        ax.errorbar([m], [p], xerr=[[m - lo], [hi - m]], fmt="none",
                    ecolor="#333333", capsize=3)
        ax.text(m + 1.8, p, f"{m:.1f}", va="center", fontsize=9, color="#333333")
    ax.invert_yaxis()
    ax.set_xlabel("implementation shortfall (bps), final untouched seeds 90000-90999")
    ax.set_title("Benchmark ladder, forced-terminal completion, agent-first ordering\n"
                 "(clairvoyant sees future shocks; not a policy)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "final_ladder.png", dpi=150)
    print({p: round(sum(v)/len(v), 2) for p, v in vals.items() if p in ORDER})


if __name__ == "__main__":
    main()
