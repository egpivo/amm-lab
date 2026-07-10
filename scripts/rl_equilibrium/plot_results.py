"""Diagnostic figures for the execution-routing simulator.

Reads the CSVs produced by `run_execution_sim` and `run_baselines` and writes
three figures to out/:
  1. trajectory.png  - oracle vs pool mids, fees, remaining inventory, actions
  2. shortfall.png   - implementation shortfall by policy x market mode
  3. sensitivity.png - TWAP shortfall over volatility x arb-speed grid

Usage: python plot_results.py [--out-dir out]
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import OUT, read_csv

COLORS = {
    "immediate": "#4269d0",
    "twap": "#efb118",
    "myopic_router": "#3ca951",
    "random": "#9c6b4e",
    "oracle": "#1f1f1f",
    "pool_a": "#4269d0",
    "pool_b": "#ff725c",
}
MODES = ["ConstantDuopoly", "DynamicMonopoly", "DynamicDuopoly"]
POLICIES = ["immediate", "twap", "myopic_router", "random"]


def style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)


def plot_trajectory(rows: list[dict], out: Path) -> None:
    t = [int(r["step"]) for r in rows]
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)

    ax = axes[0]
    ax.plot(
        t,
        [float(r["oracle_price"]) for r in rows],
        color=COLORS["oracle"],
        lw=1.6,
        label="oracle",
    )
    ax.plot(
        t,
        [float(r["pool_a_mid"]) for r in rows],
        color=COLORS["pool_a"],
        lw=1.2,
        label="pool A mid",
    )
    ax.plot(
        t,
        [float(r["pool_b_mid"]) for r in rows],
        color=COLORS["pool_b"],
        lw=1.2,
        label="pool B mid",
    )
    ax.set_ylabel("price (X per Y)")
    ax.legend(frameon=False, ncols=3)
    ax.set_title("Closed-loop episode: prices, fees, inventory, actions")

    ax = axes[1]
    for key, color, label in [
        ("pool_a_fee_buy", COLORS["pool_a"], "A buy fee"),
        ("pool_a_fee_sell", COLORS["pool_a"], "A sell fee"),
        ("pool_b_fee_buy", COLORS["pool_b"], "B buy fee"),
        ("pool_b_fee_sell", COLORS["pool_b"], "B sell fee"),
    ]:
        ls = "-" if "buy" in key else "--"
        ax.plot(
            t,
            [float(r[key]) * 1e4 for r in rows],
            color=color,
            lw=1.2,
            linestyle=ls,
            label=label,
        )
    ax.set_ylabel("fee (bps)")
    ax.legend(frameon=False, ncols=4, fontsize=8)

    ax = axes[2]
    ax.plot(
        t, [float(r["remaining_after"]) for r in rows], color=COLORS["oracle"], lw=1.6
    )
    ax.set_ylabel("remaining Y")

    ax = axes[3]
    qty_a = [float(r["agent_qty_a"]) for r in rows]
    qty_b = [float(r["agent_qty_b"]) for r in rows]
    ax.bar(t, qty_a, color=COLORS["pool_a"], label="routed to A", width=0.85)
    ax.bar(
        t, qty_b, bottom=qty_a, color=COLORS["pool_b"], label="routed to B", width=0.85
    )
    ax.set_ylabel("agent qty (Y)")
    ax.set_xlabel("step")
    ax.legend(frameon=False, ncols=2)

    for ax in axes:
        style_ax(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_shortfall(rows: list[dict], out: Path) -> None:
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        cells[(r["mode"], r["policy"])].append(float(r["shortfall_bps"]))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.2
    for j, policy in enumerate(POLICIES):
        xs = [i + (j - 1.5) * width for i in range(len(MODES))]
        means = []
        for mode in MODES:
            vals = cells.get((mode, policy), [0.0])
            means.append(sum(vals) / len(vals))
        bars = ax.bar(xs, means, width * 0.92, color=COLORS[policy], label=policy)
        for b, m in zip(bars, means):
            ax.text(
                b.get_x() + b.get_width() / 2,
                m + 3,
                f"{m:.0f}",
                ha="center",
                fontsize=8,
                color="#444444",
            )
    ax.set_xticks(range(len(MODES)), MODES)
    ax.set_ylabel("implementation shortfall (bps, mean over seeds)")
    ax.set_title("Execution baselines by market mode (lower is better)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.legend(frameon=False, ncols=4, loc="upper center")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_sensitivity(rows: list[dict], out: Path) -> None:
    cells: dict[tuple[float, float], list[float]] = defaultdict(list)
    for r in rows:
        parts = dict(p.split("=") for p in r["policy"].split("|")[1:])
        key = (float(parts["sigma"]), float(parts["arb_speed"]))
        cells[key].append(float(r["shortfall_bps"]))
    sigmas = sorted({k[0] for k in cells})
    speeds = sorted({k[1] for k in cells})
    grid = [[sum(cells[(s, a)]) / len(cells[(s, a)]) for a in speeds] for s in sigmas]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(grid, cmap="Blues", aspect="auto", origin="lower")
    ax.set_xticks(range(len(speeds)), [f"{a:g}" for a in speeds])
    ax.set_yticks(range(len(sigmas)), [f"{s:g}" for s in sigmas])
    ax.set_xlabel("arbitrageur speed (P(act) per step)")
    ax.set_ylabel("oracle volatility (annualized)")
    ax.set_title("TWAP shortfall (bps) over volatility x arb speed")
    vmax = max(max(row) for row in grid)
    for i in range(len(sigmas)):
        for j in range(len(speeds)):
            dark = grid[i][j] > 0.6 * vmax
            ax.text(
                j,
                i,
                f"{grid[i][j]:.0f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if dark else "#1f1f1f",
            )
    fig.colorbar(im, ax=ax, shrink=0.85, label="shortfall (bps)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=OUT, type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    made = []
    if (out_dir / "trajectory.csv").exists():
        plot_trajectory(
            read_csv(out_dir / "trajectory.csv"), out_dir / "trajectory.png"
        )
        made.append("trajectory.png")
    if (out_dir / "baselines.csv").exists():
        plot_shortfall(read_csv(out_dir / "baselines.csv"), out_dir / "shortfall.png")
        made.append("shortfall.png")
    if (out_dir / "sensitivity.csv").exists():
        plot_sensitivity(
            read_csv(out_dir / "sensitivity.csv"), out_dir / "sensitivity.png"
        )
        made.append("sensitivity.png")
    print(f"wrote {', '.join(made)} to {out_dir}")


if __name__ == "__main__":
    main()
