"""M3 figures + tables.

Inputs (out/): m1_results.csv, m3_fine_results.csv, m3_dqn_results.csv,
m3_value_boundary.csv, m3_fine_actions.csv, m3_traj_*.csv, m3_dqn_curve.csv.
Outputs: m3_learners.png, m3_value_boundary.png, m3_behavior_state.png,
m3_trajectories.png, m3_dqn_curve.png, m3_learner_results.csv, and printed
paired stats + behavior table (paste into notes).
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
COLORS = {
    "twap": "#efb118",
    "fee_aware_twap": "#6cc5b0",
    "lookahead": "#a463f2",
    "q_learner": "#ff8ab7",
    "q_learner_fine": "#ff725c",
    "dqn": "#4269d0",
    "two_step": "#9c6b4e",
    "three_step": "#3ca951",
    "clairvoyant": "#555555",
}


def read(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)


def boot_ci(vals, n_boot=2000, seed=0):
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(rng.choices(vals, k=n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def load_all() -> dict[str, dict[str, dict]]:
    """policy -> seed -> row (DynamicDuopoly test seeds)."""
    data: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in read(OUT / "m1_results.csv"):
        if r["mode"] == "DynamicDuopoly":
            data[r["policy"]][r["seed"]] = r
    for r in read(OUT / "m3_fine_results.csv"):
        if r["mode"] == "DynamicDuopoly" and r["policy"] == "q_learner_fine":
            data[r["policy"]][r["seed"]] = r
    for r in read(OUT / "m3_dqn_results.csv"):
        if r["seed_set"] == "test":
            data["dqn"][r["seed"]] = r
    for r in read(OUT / "m3_value_boundary.csv"):
        if r["policy"] in ("two_step", "three_step", "clairvoyant"):
            data[r["policy"]][r["seed"]] = r
    return data


def paired(data, a: str, b: str) -> tuple[float, float, float]:
    seeds = sorted(set(data[a]) & set(data[b]))
    diffs = [float(data[a][s]["shortfall_bps"]) - float(data[b][s]["shortfall_bps"])
             for s in seeds]
    lo, hi = boot_ci(diffs)
    return sum(diffs) / len(diffs), lo, hi


def fig_ladder(data) -> None:
    order = ["twap", "fee_aware_twap", "two_step", "three_step", "lookahead",
             "q_learner", "q_learner_fine", "dqn", "clairvoyant"]
    fig, ax = plt.subplots(figsize=(9, 5))
    names, means = [], []
    for p in order:
        if p not in data:
            continue
        vals = [float(r["shortfall_bps"]) for r in data[p].values()]
        m = sum(vals) / len(vals)
        lo, hi = boot_ci(vals)
        names.append(p)
        means.append(m)
        ax.barh(p, m, color=COLORS[p], height=0.62)
        ax.errorbar([m], [p], xerr=[[m - lo], [hi - m]], fmt="none",
                    ecolor="#333333", capsize=3)
        ax.text(m + 1.5, p, f"{m:.1f}", va="center", fontsize=9, color="#333333")
    ax.invert_yaxis()
    ax.set_xlabel("implementation shortfall (bps), held-out seeds")
    ax.set_title("M3 value ladder, DynamicDuopoly (clairvoyant sees future shocks; not a policy)")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "m3_learners.png", dpi=150)
    plt.close(fig)


def fig_boundary(data) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    groups = [("schedule", ["twap"]), ("planning", ["two_step", "three_step", "lookahead"]),
              ("learning", ["q_learner", "q_learner_fine", "dqn"]),
              ("hindsight bound", ["clairvoyant"])]
    x = 0
    ticks, labels = [], []
    for gname, ps in groups:
        for p in ps:
            if p not in data:
                continue
            vals = [float(r["shortfall_bps"]) for r in data[p].values()]
            m = sum(vals) / len(vals)
            lo, hi = boot_ci(vals)
            ax.errorbar([x], [m], yerr=[[m - lo], [hi - m]], fmt="o",
                        color=COLORS[p], capsize=4, markersize=7)
            ticks.append(x)
            labels.append(p)
            x += 1
        x += 0.6
    la = sum(float(r["shortfall_bps"]) for r in data["lookahead"].values()) / len(data["lookahead"])
    ax.axhline(la, color=COLORS["lookahead"], lw=0.8, linestyle="--", alpha=0.6)
    ax.set_xticks(ticks, labels, rotation=20, fontsize=9)
    ax.set_ylabel("shortfall (bps)")
    ax.set_title("M3B: schedule vs planning vs learning vs hindsight (dashed = tuned lookahead)")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "m3_value_boundary.png", dpi=150)
    plt.close(fig)


def fig_behavior_state() -> None:
    rows = read(OUT / "m3_fine_actions.csv")
    dqn_path = OUT / "m3_dqn_actions.csv"
    if dqn_path.exists():
        rows += read(dqn_path)
        learner = "dqn"
    else:
        learner = "q_learner_fine"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    # wait share vs best-pool oracle gap
    gap_bins = [(-1e9, -20), (-20, -5), (-5, 5), (5, 20), (20, 1e9)]
    gap_labels = ["<-20", "-20..-5", "-5..5", "5..20", ">20"]
    ax = axes[0]
    for pol in [learner, "lookahead"]:
        shares = []
        for lo, hi in gap_bins:
            sel = [r for r in rows if r["policy"] == pol
                   and lo <= float(r["min_oracle_gap_bps"]) < hi
                   and float(r["remaining_frac"]) > 1e-6]
            shares.append(sum(1 for r in sel if r["action"] == "0") / max(1, len(sel)))
        ax.plot(gap_labels, shares, marker="o", color=COLORS[pol], label=pol)
    ax.set_xlabel("best pool oracle gap (bps; negative = pool cheap)")
    ax.set_ylabel("wait share (remaining > 0)")
    ax.set_title("waiting vs oracle gap")
    ax.legend(frameon=False, fontsize=9)
    # route share vs buy-fee gap
    fee_bins = [(-1e9, -10), (-10, -2), (-2, 2), (2, 10), (10, 1e9)]
    fee_labels = ["<-10", "-10..-2", "-2..2", "2..10", ">10"]
    ax = axes[1]
    for pol in [learner, "lookahead"]:
        shares = []
        for lo, hi in fee_bins:
            sel = [r for r in rows if r["policy"] == pol
                   and lo <= float(r["buy_fee_gap_bps"]) < hi
                   and r["action"] in ("1", "3", "5", "2", "4", "6")]
            a_routed = sum(1 for r in sel if r["action"] in ("1", "3", "5"))
            shares.append(a_routed / max(1, len(sel)))
        ax.plot(fee_labels, shares, marker="o", color=COLORS[pol], label=pol)
    ax.set_xlabel("buy fee gap A-B (bps; negative = A cheaper)")
    ax.set_ylabel("share of single-pool trades routed to A")
    ax.set_title("routing vs fee gap")
    ax.legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    fig.suptitle("M3C: state-conditional behavior (DynamicDuopoly test seeds)")
    fig.tight_layout()
    fig.savefig(OUT / "m3_behavior_state.png", dpi=150)
    plt.close(fig)


def fig_trajectories(seeds: dict[str, str]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, (label, seed) in zip(axes, seeds.items()):
        for pol, fname, color in [("dqn", f"m3_traj_dqn_{seed}.csv", COLORS["dqn"]),
                                  ("lookahead", f"m3_traj_lookahead_{seed}.csv", COLORS["lookahead"])]:
            rows = read(OUT / fname)
            ax.plot([int(r["step"]) for r in rows],
                    [float(r["remaining_after"]) for r in rows],
                    color=color, lw=1.6, label=pol)
        ax2 = ax.twinx()
        rows = read(OUT / f"m3_traj_lookahead_{seed}.csv")
        ax2.plot([int(r["step"]) for r in rows],
                 [float(r["oracle_price"]) for r in rows],
                 color="#999999", lw=0.9, alpha=0.8)
        ax2.set_yticks([])
        ax.set_title(f"{label} (seed {seed})", fontsize=10)
        ax.set_xlabel("step")
        ax.spines[["top"]].set_visible(False)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("remaining Y (gray = oracle path)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("M3C: representative episodes, DQN vs lookahead")
    fig.tight_layout()
    fig.savefig(OUT / "m3_trajectories.png", dpi=150)
    plt.close(fig)


def fig_curve() -> None:
    rows = read(OUT / "m3_dqn_curve.csv")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot([int(r["episode"]) for r in rows],
            [float(r["val_shortfall_bps"]) for r in rows],
            color=COLORS["dqn"], marker="o", markersize=3, lw=1.3)
    ax.axhline(99.8, color=COLORS["lookahead"], lw=0.9, linestyle="--",
               label="lookahead (test)")
    ax.set_xlabel("training episode")
    ax.set_ylabel("validation shortfall (bps)")
    ax.set_title("DQN training curve (greedy eval on 50 validation seeds)")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "m3_dqn_curve.png", dpi=150)
    plt.close(fig)


def behavior_table(data) -> None:
    cols = ["shortfall_bps", "completion_rate", "wait_share", "route_share_a",
            "fee_paid_bps", "gas_paid_bps", "slippage_ex_fee_bps", "drift_bps",
            "terminal_penalty_bps"]
    print("\n=== M3C behavior table (DynamicDuopoly test means) ===")
    print(f"{'policy':<16}" + "".join(f"{c[:12]:>13}" for c in cols))
    for p in ["twap", "lookahead", "q_learner", "q_learner_fine", "dqn"]:
        if p not in data:
            continue
        vals = []
        for c in cols:
            xs = [float(r[c]) for r in data[p].values() if c in r and r[c] != ""]
            vals.append(sum(xs) / len(xs) if xs else float("nan"))
        print(f"{p:<16}" + "".join(f"{v:>13.2f}" for v in vals))


def main() -> None:
    data = load_all()
    # merged per-seed results file for the record
    with open(OUT / "m3_learner_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "seed", "shortfall_bps", "completion_rate"])
        for p, seeds in data.items():
            for s, r in sorted(seeds.items()):
                w.writerow([p, s, r["shortfall_bps"], r["completion_rate"]])

    print("=== paired vs lookahead (negative favors row policy) ===")
    for p in ["q_learner", "q_learner_fine", "dqn", "two_step", "three_step", "clairvoyant"]:
        if p in data:
            m, lo, hi = paired(data, p, "lookahead")
            print(f"{p:<16} {m:+7.2f} bps  95% CI [{lo:+.2f}, {hi:+.2f}]")

    fig_ladder(data)
    fig_boundary(data)
    fig_behavior_state()
    fig_trajectories({"DQN wins": "30409", "DQN loses": "30138", "tie": "30266"})
    fig_curve()
    behavior_table(data)
    print("\nfigures written to", OUT)


if __name__ == "__main__":
    main()
