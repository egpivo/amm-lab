"""Paper figures: state-conditional behavior (DQN vs. tuned lookahead),
split into two standalone panels for LaTeX subfigures. White background,
no in-figure titles (captions carry them), paper-sized fonts.

Inputs: out/m3_fine_actions.csv (lookahead rows), out/m3_dqn_actions.csv.
Outputs: out/m3_behavior_wait.png, out/m3_behavior_routing.png.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import OUT, read_csv

TEXT = "#2B2B2B"
GRID = "#D9D7D1"
DQN = "#4C6A91"
LOOKAHEAD = "#8A6799"

GAP_BINS = [(-1e9, -20), (-20, -5), (-5, 5), (5, 20), (20, 1e9)]
GAP_LABELS = ["<-20", "-20..-5", "-5..5", "5..20", ">20"]
FEE_BINS = [(-1e9, -10), (-10, -2), (-2, 2), (2, 10), (10, 1e9)]
FEE_LABELS = ["<-10", "-10..-2", "-2..2", "2..10", ">10"]
TRADE_ACTIONS = {"1", "2", "3", "4", "5", "6"}
ROUTE_A = {"1", "3", "5"}


def load_rows() -> list[dict]:
    rows = read_csv(OUT / "m3_fine_actions.csv") + read_csv(OUT / "m3_dqn_actions.csv")
    return [
        r
        for r in rows
        if r["mode"] == "DynamicDuopoly" and r["policy"] in ("dqn", "lookahead")
    ]


def wait_curve(rows: list[dict], policy: str) -> list[float]:
    sub = [r for r in rows if r["policy"] == policy]
    shares = []
    for lo, hi in GAP_BINS:
        sel = [
            r
            for r in sub
            if lo <= float(r["min_oracle_gap_bps"]) < hi
            and float(r["remaining_frac"]) > 1e-6
        ]
        shares.append(
            sum(1 for r in sel if r["action"] == "0") / len(sel) if sel else float("nan")
        )
    return shares


def route_curve(rows: list[dict], policy: str) -> list[float]:
    sub = [r for r in rows if r["policy"] == policy]
    shares = []
    for lo, hi in FEE_BINS:
        sel = [
            r
            for r in sub
            if lo <= float(r["buy_fee_gap_bps"]) < hi and r["action"] in TRADE_ACTIONS
        ]
        shares.append(
            sum(1 for r in sel if r["action"] in ROUTE_A) / len(sel)
            if sel
            else float("nan")
        )
    return shares


def styled_panel() -> tuple[plt.Figure, plt.Axes]:
    # Each image occupies half of one ACM column, so size the source canvas to
    # the final subfigure instead of shrinking a full-width plot in LaTeX.
    fig, ax = plt.subplots(figsize=(3.6, 2.7), dpi=200)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", labelsize=13, colors=TEXT)
    return fig, ax


def main() -> None:
    rows = load_rows()
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
        }
    )

    fig, ax = styled_panel()
    x = range(len(GAP_LABELS))
    ax.plot(x, wait_curve(rows, "dqn"), marker="o", markersize=6.5, color=DQN,
            linewidth=2.2, label="DQN")
    ax.plot(x, wait_curve(rows, "lookahead"), marker="o", markersize=6.5,
            color=LOOKAHEAD, linewidth=2.2, label="Lookahead")
    ax.set_xlabel("Best-pool oracle gap (bps)", fontsize=14, labelpad=6)
    ax.set_ylabel("Wait share", fontsize=14)
    ax.set_xticks(list(x), GAP_LABELS)
    ax.set_ylim(0.35, 0.85)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=2, fontsize=11.5, columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "m3_behavior_wait.png", bbox_inches="tight")

    fig, ax = styled_panel()
    x = range(len(FEE_LABELS))
    ax.plot(x, route_curve(rows, "dqn"), marker="o", markersize=6.5, color=DQN,
            linewidth=2.2, label="DQN")
    ax.plot(x, route_curve(rows, "lookahead"), marker="o", markersize=6.5,
            color=LOOKAHEAD, linewidth=2.2, label="Lookahead")
    ax.set_xlabel("Buy-fee gap $A-B$ (bps)", fontsize=14, labelpad=6)
    ax.set_ylabel("Trades routed to $A$", fontsize=14)
    ax.set_xticks(list(x), FEE_LABELS)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=2, fontsize=11.5, columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "m3_behavior_routing.png", bbox_inches="tight")
    print("wrote m3_behavior_wait.png, m3_behavior_routing.png")


if __name__ == "__main__":
    main()
