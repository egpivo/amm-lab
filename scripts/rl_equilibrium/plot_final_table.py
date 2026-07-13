"""Paper figure: the final reserved-block headline comparison, replacing the
main-text table. Two standalone panels for LaTeX subfigures: (a) mean absolute
shortfall by intra-step ordering for TWAP, tuned lookahead, and the DQN;
(b) paired DQN - lookahead edge with pointwise 95% percentile bootstrap CIs
(B = 4,000, matching the reported headline intervals), dashed line at zero.
White background, no in-figure titles.

Inputs: out/m3r_final_paper_seeds.csv (DQN, one checkpoint per ordering),
out/m3r_reference_final.csv (lookahead and TWAP per evaluation ordering).
Outputs: out/m3r_final_abs.png, out/m3r_final_edge.png.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import OUT, bootstrap_ci, mean, read_csv

TEXT = "#2B2B2B"
GRID = "#D9D7D1"
COLORS = {"twap": "#B7905E", "lookahead": "#8A6799", "dqn": "#4C6A91"}
LABELS = {"twap": "TWAP", "lookahead": "Tuned lookahead", "dqn": "DQN"}

ORDERS = ["before", "random", "after"]
ORDER_LABELS = ["Agent-first", "Randomized", "Agent-last\n(headline)"]


def load() -> tuple[dict, dict]:
    dqn: dict[str, dict[str, float]] = {o: {} for o in ORDERS}
    for r in read_csv(OUT / "m3r_final_paper_seeds.csv"):
        dqn[r["agent_order"]][r["seed"]] = float(r["shortfall_bps"])
    ref: dict[tuple[str, str], dict[str, float]] = {}
    for r in read_csv(OUT / "m3r_reference_final.csv"):
        ref.setdefault((r["policy"], r["agent_order"]), {})[r["seed"]] = float(
            r["shortfall_bps"]
        )
    return dqn, ref


def style(ax) -> None:
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()


def plot_absolute(dqn: dict, ref: dict) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=200)
    for y, order in enumerate(ORDERS):
        for policy in ("twap", "lookahead", "dqn"):
            values = (
                dqn[order] if policy == "dqn" else ref[(policy, order)]
            ).values()
            m = mean(list(values))
            ax.plot([m], [y], "o", markersize=10, color=COLORS[policy],
                    label=LABELS[policy] if y == 0 else None, zorder=3)
            ax.text(m, y - 0.22, f"{m:.1f}", ha="center", fontsize=12.5,
                    color=TEXT)
    ax.set_yticks(range(len(ORDERS)), ORDER_LABELS, fontsize=14)
    ax.set_xlabel("Implementation shortfall (bps)", fontsize=15, labelpad=8)
    ax.margins(y=0.18)
    style(ax)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=3, fontsize=12.5, labelcolor=TEXT, columnspacing=1.2,
              handletextpad=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "m3r_final_abs.png", bbox_inches="tight")
    plt.close(fig)


def plot_edge(dqn: dict, ref: dict) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=200)
    for y, order in enumerate(ORDERS):
        la = ref[("lookahead", order)]
        diffs = [v - la[seed] for seed, v in dqn[order].items() if seed in la]
        m = mean(diffs)
        lo, hi = bootstrap_ci(diffs, n_boot=4_000)
        ax.errorbar([m], [y], xerr=[[m - lo], [hi - m]], fmt="o",
                    markersize=10, color=COLORS["dqn"], ecolor=COLORS["dqn"],
                    elinewidth=2.0, capsize=4, zorder=3)
        ax.text(m, y - 0.22, f"{m:+.2f}", ha="center", fontsize=12.5,
                color=TEXT)
        print(f"{order}: {m:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    ax.axvline(0.0, color=TEXT, linewidth=1.0, linestyle="--", zorder=1)
    ax.set_yticks(range(len(ORDERS)), ORDER_LABELS, fontsize=14)
    ax.set_xlabel("Paired DQN $-$ lookahead edge (bps)", fontsize=15,
                  labelpad=8)
    ax.set_xlim(right=1.5)
    ax.margins(y=0.18)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "m3r_final_edge.png", bbox_inches="tight")
    plt.close(fig)


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
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
        }
    )
    dqn, ref = load()
    plot_absolute(dqn, ref)
    plot_edge(dqn, ref)
    print("wrote m3r_final_abs.png, m3r_final_edge.png")


if __name__ == "__main__":
    main()
