"""Paper Figure: the benchmark ladder on the final reserved seed block
(90,000-90,999), forced-terminal completion, agent-first ordering.

Style matches the blog ladder (zen palette, hatched non-deployable
reference, readable policy labels) with two paper-specific differences:
white background and 95% bootstrap CI whiskers, which the paper caption
promises.

Inputs: out/final_ladder.csv (Rust policies + clairvoyant),
out/m3r_stochastic_planner_final.csv, out/m3r_final_paper_seeds.csv (DQN).
Output: out/final_ladder.png.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import OUT, bootstrap_ci, read_csv

TEXT = "#2B2B2B"
SECONDARY = "#5F6368"
GRID = "#D9D7D1"

COLORS = {
    "twap": "#B7905E",
    "two_step": "#9A7B45",
    "stochastic_planner": "#7C8A78",
    "three_step": "#6A8F73",
    "lookahead": "#8A6799",
    "q_learner": "#7A8A9A",
    "q_learner_fine": "#5F7A8A",
    "dqn": "#4C6A91",
    "clairvoyant": "#7A7A7A",
}
LABELS = {
    "twap": "TWAP",
    "two_step": "Two-step planner",
    "stochastic_planner": "Stochastic planner",
    "three_step": "Three-step planner",
    "lookahead": "Tuned one-step lookahead",
    "q_learner": "Q-learner",
    "q_learner_fine": "Q-learner (fine)",
    "dqn": "DQN",
    "clairvoyant": "Achieved hindsight reference",
}
ORDER = [
    "twap",
    "two_step",
    "stochastic_planner",
    "three_step",
    "lookahead",
    "q_learner",
    "q_learner_fine",
    "dqn",
    "clairvoyant",
]


def main() -> None:
    vals: dict[str, list[float]] = {}
    for r in read_csv(OUT / "final_ladder.csv"):
        vals.setdefault(r["policy"], []).append(float(r["shortfall_bps"]))
    for r in read_csv(OUT / "m3r_stochastic_planner_final.csv"):
        if r["policy"] == "stochastic_planner":
            vals.setdefault("stochastic_planner", []).append(float(r["shortfall_bps"]))
    for r in read_csv(OUT / "m3r_final_paper_seeds.csv"):
        if r["policy"] == "dqn_dynamic_duopoly" and r["agent_order"] == "before":
            vals.setdefault("dqn", []).append(float(r["shortfall_bps"]))

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Inter",
                "Source Sans 3",
                "Arial",
                "Helvetica",
                "DejaVu Sans",
            ],
            "axes.labelcolor": TEXT,
            "axes.edgecolor": GRID,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
        }
    )

    means, lows, highs = [], [], []
    for p in ORDER:
        v = vals[p]
        m = sum(v) / len(v)
        lo, hi = bootstrap_ci(v)
        means.append(m)
        lows.append(m - lo)
        highs.append(hi - m)

    fig, ax = plt.subplots(figsize=(10.5, 6), dpi=200)
    y = list(range(len(ORDER)))
    bars = ax.barh(
        y, means, color=[COLORS[p] for p in ORDER], height=0.62, zorder=3
    )
    hatched = bars[ORDER.index("clairvoyant")]
    hatched.set_hatch("////")
    hatched.set_edgecolor(SECONDARY)
    hatched.set_linewidth(0.6)
    ax.errorbar(
        means,
        y,
        xerr=[lows, highs],
        fmt="none",
        ecolor=TEXT,
        elinewidth=1.0,
        capsize=3,
        zorder=4,
    )

    xmax = max(means) * 1.14
    ax.set_xlim(0, xmax)
    ax.set_yticks(y, [LABELS[p] for p in ORDER], fontsize=16)
    ax.invert_yaxis()
    for yi, m in zip(y, means):
        ax.text(
            m + xmax * 0.018,
            yi,
            f"{m:.1f}",
            va="center",
            ha="left",
            fontsize=15,
            color=TEXT,
        )

    ax.set_xlabel("Implementation shortfall (bps)", fontsize=16, labelpad=8)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "final_ladder.png", bbox_inches="tight")
    print({p: round(sum(v) / len(v), 2) for p, v in vals.items() if p in ORDER})


if __name__ == "__main__":
    main()
