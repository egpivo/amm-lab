"""Paper figure: sensitivity-layer paired edges as a forest plot,
replacing the appendix table. Point = mean paired DQN - lookahead edge,
whiskers = pointwise 95% percentile bootstrap CI, dashed line at zero.
White background, no in-figure title.

Inputs: out/m4_lp_adaptation.csv, out/m4_jit_mev.csv.
Output: out/m4_sensitivity_edges.png.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import OUT, bootstrap_ci, mean, read_csv

TEXT = "#2B2B2B"
GRID = "#D9D7D1"
LP_COLOR = "#6A8F73"
SEARCHER_COLOR = "#4C6A91"

ROWS = [
    ("m4_lp_adaptation.csv", "frozen", "LP: frozen (baseline)", LP_COLOR),
    ("m4_lp_adaptation.csv", "weak", "LP: weak withdrawal", LP_COLOR),
    ("m4_lp_adaptation.csv", "aggressive", "LP: aggressive exit", LP_COLOR),
    ("m4_jit_mev.csv", "none", "Searcher: none (baseline)", SEARCHER_COLOR),
    ("m4_jit_mev.csv", "weak", "Searcher: weak", SEARCHER_COLOR),
    ("m4_jit_mev.csv", "aggressive", "Searcher: aggressive", SEARCHER_COLOR),
]


def paired_diffs(filename: str, regime: str) -> list[float]:
    by_policy: dict[str, dict[str, float]] = {}
    for r in read_csv(OUT / filename):
        if r["regime"] == regime:
            by_policy.setdefault(r["policy"], {})[r["seed"]] = float(r["shortfall_bps"])
    dqn, la = by_policy["dqn"], by_policy["lookahead"]
    return [dqn[seed] - la[seed] for seed in dqn if seed in la]


def main() -> None:
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
            "xtick.color": TEXT,
            "ytick.color": TEXT,
        }
    )
    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=200)
    y_positions = list(range(len(ROWS)))
    for y, (filename, regime, label, color) in zip(y_positions, ROWS):
        diffs = paired_diffs(filename, regime)
        m = mean(diffs)
        lo, hi = bootstrap_ci(diffs)
        ax.errorbar(
            [m], [y], xerr=[[m - lo], [hi - m]], fmt="o", markersize=9,
            color=color, ecolor=color, elinewidth=2.0, capsize=4, zorder=3,
        )
        ax.text(m, y - 0.32, f"{m:+.1f}", ha="center", fontsize=14, color=TEXT)
    ax.axvline(0.0, color=TEXT, linewidth=1.0, linestyle="--", zorder=1)
    ax.set_yticks(y_positions, [r[2] for r in ROWS], fontsize=15)
    ax.invert_yaxis()
    ax.set_xlabel("Paired DQN $-$ lookahead edge (bps)", fontsize=16, labelpad=8)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    fig.tight_layout()
    fig.savefig(OUT / "m4_sensitivity_edges.png", bbox_inches="tight")
    print("wrote m4_sensitivity_edges.png")


if __name__ == "__main__":
    main()
