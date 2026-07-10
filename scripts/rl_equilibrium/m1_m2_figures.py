"""M1/M2 figures from m1_results.csv, m1_actions.csv, m1_trajectory_q.csv,
and m2_diagnostics.csv. Writes PNGs to out/.

Usage: python m1_m2_figures.py
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2] / "experiments/rl_execution/out"

# Fixed categorical order, colorblind-aware (Observable-10 subset).
COLORS = {
    "immediate": "#4269d0",
    "twap": "#efb118",
    "myopic_router": "#3ca951",
    "random": "#9c6b4e",
    "fee_aware_twap": "#6cc5b0",
    "lookahead": "#a463f2",
    "q_learner": "#ff725c",
}
POLICIES = list(COLORS)
MODES = ["ConstantDuopoly", "DynamicMonopoly", "DynamicDuopoly"]
ACTIONS = ["wait", "10%A", "10%B", "25%A", "25%B", "50%A", "50%B", "split"]


def read(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)


def boot_ci(vals: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(rng.choices(vals, k=n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def fig_comparison(rows: list[dict]) -> None:
    cells = defaultdict(list)
    for r in rows:
        cells[(r["mode"], r["policy"])].append(float(r["shortfall_bps"]))
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    width = 0.11
    for j, p in enumerate(POLICIES):
        xs, means, lo, hi = [], [], [], []
        for i, m in enumerate(MODES):
            vals = cells[(m, p)]
            xs.append(i + (j - 3) * width)
            mean = sum(vals) / len(vals)
            a, b = boot_ci(vals)
            means.append(mean)
            lo.append(mean - a)
            hi.append(b - mean)
        ax.bar(xs, means, width * 0.9, color=COLORS[p], label=p)
        ax.errorbar(xs, means, yerr=[lo, hi], fmt="none", ecolor="#333333",
                    elinewidth=0.9, capsize=2)
    ax.set_xticks(range(len(MODES)), MODES)
    ax.set_ylabel("implementation shortfall (bps)")
    ax.set_title("M1: shortfall by policy and market mode, 500 held-out seeds (95% bootstrap CI)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    ax.legend(frameon=False, ncols=4, loc="upper center", fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "m1_comparison.png", dpi=150)
    plt.close(fig)


def fig_paired(rows: list[dict]) -> None:
    by = defaultdict(dict)
    for r in rows:
        if r["mode"] == "DynamicDuopoly":
            by[r["policy"]][r["seed"]] = float(r["shortfall_bps"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, base in zip(axes, ["lookahead", "twap"]):
        diffs = [by["q_learner"][s] - by[base][s] for s in by["q_learner"]]
        mean = sum(diffs) / len(diffs)
        a, b = boot_ci(diffs)
        ax.hist(diffs, bins=40, color=COLORS["q_learner"], alpha=0.85)
        ax.axvline(0, color="#333333", lw=1)
        ax.axvline(mean, color="#1f1f1f", lw=1.4, linestyle="--")
        ax.set_title(f"q_learner - {base}\nmean {mean:+.2f} bps, 95% CI [{a:+.2f}, {b:+.2f}]",
                     fontsize=10)
        ax.set_xlabel("per-seed shortfall difference (bps)")
        style(ax)
    axes[0].set_ylabel("seeds")
    fig.suptitle("M1 paired seed-level comparison, DynamicDuopoly (negative favors learner)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "m1_paired_diff.png", dpi=150)
    plt.close(fig)


def fig_trajectory(rows: list[dict]) -> None:
    t = [int(r["step"]) for r in rows]
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    ax = axes[0]
    ax.plot(t, [float(r["oracle_price"]) for r in rows], color="#1f1f1f", lw=1.6, label="oracle")
    ax.plot(t, [float(r["pool_a_mid"]) for r in rows], color="#4269d0", lw=1.2, label="pool A mid")
    ax.plot(t, [float(r["pool_b_mid"]) for r in rows], color="#ff725c", lw=1.2, label="pool B mid")
    ax.set_ylabel("price (X per Y)")
    ax.legend(frameon=False, ncols=3)
    ax.set_title("Trained Q-policy episode (DynamicDuopoly, fresh seed 40007)")
    ax = axes[1]
    for key, color, ls, label in [("pool_a_fee_buy", "#4269d0", "-", "A buy"),
                                  ("pool_b_fee_buy", "#ff725c", "-", "B buy"),
                                  ("pool_a_fee_sell", "#4269d0", "--", "A sell"),
                                  ("pool_b_fee_sell", "#ff725c", "--", "B sell")]:
        ax.plot(t, [float(r[key]) * 1e4 for r in rows], color=color, lw=1.1, linestyle=ls, label=label)
    ax.set_ylabel("fee (bps)")
    ax.legend(frameon=False, ncols=4, fontsize=8)
    ax = axes[2]
    ax.plot(t, [float(r["remaining_after"]) for r in rows], color="#1f1f1f", lw=1.6)
    ax.set_ylabel("remaining Y")
    ax = axes[3]
    qa = [float(r["agent_qty_a"]) for r in rows]
    qb = [float(r["agent_qty_b"]) for r in rows]
    ax.bar(t, qa, color="#4269d0", label="routed to A", width=0.85)
    ax.bar(t, qb, bottom=qa, color="#ff725c", label="routed to B", width=0.85)
    ax.set_ylabel("agent qty (Y)")
    ax.set_xlabel("step")
    ax.legend(frameon=False, ncols=2)
    for ax in axes:
        style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "m1_trajectory.png", dpi=150)
    plt.close(fig)


def fig_actions(rows: list[dict]) -> None:
    # action mix by remaining-time decile, q_learner vs lookahead
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    cmap = ["#bbbbbb", "#4269d0", "#89a6e8", "#3ca951", "#8fd0a1",
            "#efb118", "#f6d573", "#a463f2"]
    for ax, pol in zip(axes, ["q_learner", "lookahead"]):
        counts = defaultdict(lambda: [0] * 8)
        for r in rows:
            if r["policy"] != pol:
                continue
            tb = min(9, int((1.0 - float(r["remaining_time_frac"])) * 10))
            counts[tb][int(r["action"])] += 1
        xs = sorted(counts)
        bottoms = [0.0] * len(xs)
        for a in range(8):
            share = [counts[x][a] / max(1, sum(counts[x])) for x in xs]
            ax.bar(xs, share, bottom=bottoms, color=cmap[a], width=0.85,
                   label=ACTIONS[a] if pol == "q_learner" else None)
            bottoms = [b + s for b, s in zip(bottoms, share)]
        ax.set_title(pol)
        ax.set_xlabel("episode progress decile")
        style(ax)
    axes[0].set_ylabel("action share")
    fig.legend(frameon=False, ncols=8, loc="upper center", fontsize=8,
               bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("M1: action distribution by remaining time (DynamicDuopoly test seeds)")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "m1_actions_time.png", dpi=150)
    plt.close(fig)

    # wait share by best-pool premium bin
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = [(0, 40), (40, 55), (55, 70), (70, 90), (90, 1e9)]
    labels = ["<40", "40-55", "55-70", "70-90", ">90"]
    for pol in ["q_learner", "lookahead"]:
        shares = []
        for lo, hi in bins:
            sel = [r for r in rows
                   if r["policy"] == pol and lo <= float(r["est_slippage_medium_bps"]) < hi]
            waits = sum(1 for r in sel if r["action"] == "0")
            shares.append(waits / max(1, len(sel)))
        ax.plot(labels, shares, marker="o", color=COLORS[pol], label=pol)
    ax.set_xlabel("estimated slippage for a 25% clip (bps)")
    ax.set_ylabel("wait share")
    ax.set_title("M1: waiting vs current execution premium")
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "m1_actions_state.png", dpi=150)
    plt.close(fig)


def fig_m2(rows: list[dict]) -> None:
    panels = [
        ("priority", ["before", "after", "random"], "agent order vs noise+arb"),
        ("gas", ["gas=0", "gas=2", "gas=10"], "agent gas cost (X)"),
        ("arb_speed", ["speed=0.2", "speed=0.5", "speed=1"], "arbitrageur speed"),
        ("coeff", ["a_own_x0.5", "a_own_x2", "a_rival_x0.5", "a_rival_x2",
                   "a_oracle_x0.5", "a_oracle_x2"], "dynamic-fee coefficient perturbation"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for ax, (diag, variants, title) in zip(axes.flat, panels):
        for pol in ["q_learner", "lookahead", "twap"]:
            means, lo, hi = [], [], []
            for v in variants:
                vals = [float(r["shortfall_bps"]) for r in rows
                        if r["diagnostic"] == diag and r["variant"] == v and r["policy"] == pol]
                mean = sum(vals) / len(vals)
                a, b = boot_ci(vals)
                means.append(mean)
                lo.append(mean - a)
                hi.append(b - mean)
            xs = range(len(variants))
            ax.errorbar(xs, means, yerr=[lo, hi], marker="o", color=COLORS[pol],
                        label=pol, capsize=3, lw=1.4)
        ax.set_xticks(range(len(variants)), variants,
                      rotation=20 if diag == "coeff" else 0, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("shortfall (bps)")
        style(ax)
    axes[0][0].legend(frameon=False, fontsize=9)
    fig.suptitle("M2 artifact diagnostics: frozen DynamicDuopoly learner under perturbation "
                 "(300 test seeds, 95% bootstrap CI)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "m2_artifacts.png", dpi=150)
    plt.close(fig)


def main() -> None:
    m1 = read(OUT / "m1_results.csv")
    fig_comparison(m1)
    fig_paired(m1)
    fig_trajectory(read(OUT / "m1_trajectory_q.csv"))
    fig_actions(read(OUT / "m1_actions.csv"))
    fig_m2(read(OUT / "m2_diagnostics.csv"))
    print("wrote m1_comparison, m1_paired_diff, m1_trajectory, m1_actions_time, "
          "m1_actions_state, m2_artifacts to", OUT)


if __name__ == "__main__":
    main()
