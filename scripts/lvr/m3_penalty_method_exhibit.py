#!/usr/bin/env python3
"""Build the no-new-seed M3 exact-penalty method exhibit."""

from __future__ import annotations

import csv
import gzip
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / ".local" / "lvr"
V0 = 40_000_000.0


def as_bool(value: str) -> bool:
    return value.lower() == "true"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def policy_id(dial: float, alpha: float) -> str:
    d = format(dial, "g").replace(".", "p")
    a = format(alpha, "g").replace(".", "p")
    return f"gap_d{d}_a{a}"


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=2e-10, abs_tol=2e-10)


def aggregate_cell(cell_idx: int) -> list[dict]:
    metrics = ("a", "b", "l", "fees", "u", "s")
    sums = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    specs = {}
    for path in sorted(LOCAL.glob("m3_amended_training_rows_shard*.csv.gz")):
        with gzip.open(path, "rt", newline="") as handle:
            for row in csv.DictReader(handle):
                if int(row["cell_idx"]) != cell_idx or row["family"] != "gap":
                    continue
                dial, alpha = float(row["dial_mult"]), float(row["alpha"])
                key = (dial, alpha)
                counts[key] += 1
                for metric in metrics:
                    sums[key][metric] += float(row[metric])
                specs[key] = {
                    "dial_mult": dial, "f0": float(row["f0"]),
                    "alpha": alpha, "fee_cap": float(row["fee_cap"]),
                }
    policies = []
    for key in sorted(sums, key=lambda x: (x[1], x[0])):
        if counts[key] != 100:
            raise RuntimeError(f"unexpected training count for {key}")
        means = {f"mean_{m}": sums[key][m] / counts[key] for m in metrics}
        if not close(means["mean_u"], means["mean_fees"] - means["mean_l"]):
            raise RuntimeError(f"accounting identity failed for {key}")
        policies.append({**specs[key], **means, "policy_id": policy_id(*key)})
    if len(policies) != 84:
        raise RuntimeError(f"expected 84 canonical gap policies, found {len(policies)}")
    return policies


def envelope(policies: list[dict], target: float, metric: str) -> list[dict]:
    """Upper envelope for U, or lower envelope for L, on lambda >= 0."""
    lines = []
    for policy in policies:
        shortfall = max(target - policy["mean_s"], 0.0)
        value = policy[f"mean_{metric}"]
        intercept = value if metric == "u" else -value
        lines.append({**policy, "shortfall": shortfall, "hull_intercept": intercept,
                      "hull_slope": -shortfall})
    by_slope = defaultdict(list)
    for line in lines:
        by_slope[line["hull_slope"]].append(line)
    candidates = []
    for slope in sorted(by_slope):
        group = by_slope[slope]
        best = max(x["hull_intercept"] for x in group)
        tied = [x for x in group if close(x["hull_intercept"], best)]
        candidates.append(sorted(tied, key=lambda x: (x["alpha"], x["dial_mult"], x["policy_id"]))[0])
    hull, starts = [], []
    for line in candidates:
        start = -math.inf
        while hull:
            prev = hull[-1]
            start = ((prev["hull_intercept"] - line["hull_intercept"])
                     / (line["hull_slope"] - prev["hull_slope"]))
            if starts[-1] == -math.inf or start > starts[-1] and not close(start, starts[-1]):
                break
            hull.pop(); starts.pop()
        if not hull:
            start = -math.inf
        hull.append(line); starts.append(start)
    segments = []
    for idx, line in enumerate(hull):
        raw_start = starts[idx]
        raw_end = starts[idx + 1] if idx + 1 < len(starts) else math.inf
        if raw_end < 0 and not close(raw_end, 0):
            continue
        start = max(0.0, raw_start)
        if math.isfinite(raw_end) and (raw_end < start or close(raw_end, start)):
            continue
        segments.append({**line, "lambda_start": start,
                         "lambda_end": None if math.isinf(raw_end) else raw_end})
    if not segments or segments[0]["lambda_start"] != 0 or segments[-1]["lambda_end"] is not None:
        raise RuntimeError("incomplete envelope")
    return segments


def plot_panel(ax, segments: list[dict], s0: float, target: float, lambda_star: float,
               objective: str, constrained_u_id: str) -> None:
    finite = [s["lambda_start"] for s in segments[1:]]
    x_end = max([lambda_star * 1.18, *(x * 1.12 for x in finite), 1e-8])
    starts = [s["lambda_start"] for s in segments]
    ends = [s["lambda_end"] if s["lambda_end"] is not None else x_end for s in segments]
    x = [v * s0 / V0 * 1e4 for v in starts] + [x_end * s0 / V0 * 1e4]
    surplus = [s["mean_u"] / V0 * 1e4 for s in segments] + [segments[-1]["mean_u"] / V0 * 1e4]
    loss = [s["mean_l"] / V0 * 1e4 for s in segments] + [segments[-1]["mean_l"] / V0 * 1e4]
    service = [s["mean_s"] / s0 for s in segments] + [segments[-1]["mean_s"] / s0]
    ax2 = ax.twinx()
    if objective == "surplus":
        ax.step(x, surplus, where="post", color="#2166ac", linewidth=1.9, label=r"selected $U/V_0$")
        ax.set_ylabel(r"Selected surplus / $V_0$ ($\times 10^4$)", color="#2166ac")
    else:
        ax.step(x, loss, where="post", color="#b2182b", linewidth=1.9, label=r"selected $L/V_0$")
        ax.step(x, surplus, where="post", color="#2166ac", linewidth=1.4,
                linestyle="--", label=r"selected $U/V_0$")
        ax.set_ylabel(r"Selected metric / $V_0$ ($\times 10^4$)")
    ax2.step(x, service, where="post", color="#1b7837", linewidth=1.5, label=r"selected $S/S_0$")
    ax2.axhline(target / s0, color="#1b7837", linestyle=":", linewidth=1.1)
    ax2.set_ylabel(r"Selected service / $S_0$", color="#1b7837")
    for segment, start, end in zip(segments, starts, ends):
        left, right = start * s0 / V0 * 1e4, end * s0 / V0 * 1e4
        if segment["shortfall"] > 0:
            ax.axvspan(left, right, color="#f4cccc", alpha=0.22, linewidth=0)
        mid = left + 0.5 * (right - left)
        label = f"{segment['dial_mult']:g}/{segment['alpha']:g}"
        ax.annotate(label, (mid, 0.98), xycoords=("data", "axes fraction"),
                    ha="center", va="top", rotation=45, fontsize=6.0, color="#333333")
    lambda_x = lambda_star * s0 / V0 * 1e4
    if objective == "surplus":
        ax.axvline(lambda_x, color="#111111", linestyle="--", linewidth=1.1)
        ax.text(lambda_x, 0.04, r" $\widetilde{\lambda}^{\star}$", rotation=90,
                transform=ax.get_xaxis_transform(), va="bottom", fontsize=8)
    first_feasible = next(s["lambda_start"] for s in segments if s["shortfall"] == 0)
    if objective == "loss":
        first_x = first_feasible * s0 / V0 * 1e4
        ax.axvline(first_x, color="#111111", linestyle=":", linewidth=1.1)
        ax.text(first_x, 0.04, " first feasible", rotation=90,
                transform=ax.get_xaxis_transform(), va="bottom", fontsize=8)
        limit = segments[-1]
        status = "same" if limit["policy_id"] == constrained_u_id else "different"
        text = (f"large-$\\lambda$ limit ({status})\n{limit['policy_id']}\n"
                f"fees={limit['mean_fees']:.0f}, L={limit['mean_l']:.0f}, U={limit['mean_u']:.0f}")
        ax.text(0.98, 0.06, text, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7.2, bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "pad": 3})
    ax.set_xlabel(r"Normalized penalty $\widetilde{\lambda}$ ($\times 10^4$)")
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend_loc = "upper left" if objective == "loss" else "lower right"
    ax.legend(lines1 + lines2, labels1 + labels2, loc=legend_loc, frameon=False, fontsize=7.5)


def main() -> None:
    exact = read_csv(LOCAL / "m3_exact_penalties.csv")
    normalized = {(r["cell_idx"], r["rho"]): r for r in read_csv(LOCAL / "m3_normalized_penalties.csv")}
    alignment = read_csv(LOCAL / "m3_loss_alignment.csv")
    align_by_key = {(r["cell_idx"], r["rho"]): r for r in alignment}
    if len(exact) != 270 or len(alignment) != 270:
        raise RuntimeError("expected 270 finite-grid states")

    cells = []
    for row in exact:
        key = (row["cell_idx"], row["rho"])
        a = align_by_key[key]
        n = normalized[key]
        cells.append({
            "cell_idx": int(row["cell_idx"]), "stratum": row["stratum"],
            "sigma": float(row["sigma"]), "z": float(row["z"]), "speed": row["speed"],
            "support_label": row["support_label"], "rho": float(row["rho"]),
            "training_S0": float(row["training_S0"]), "service_target": float(row["service_target"]),
            "lambda_star_raw": float(row["lambda_star"]),
            "lambda_tilde_star": float(n["lambda_tilde_star"]),
            "surplus_recovery": as_bool(row["penalized_surplus_recovers_constrained_optimum"]),
            "surplus_segments": int(row["surplus_envelope_segments"]),
            "constrained_u_policy_id": a["pi_u_star_id"],
            "constrained_l_policy_id": a["pi_l_star_id"],
            "constrained_l_u_disagreement": as_bool(a["l_vs_u_diverges"]),
            "constrained_u_mean_fees": float(a["pi_u_star_mean_fees"]),
            "constrained_u_mean_l": float(a["pi_u_star_mean_l"]),
            "constrained_u_mean_u": float(a["pi_u_star_mean_u"]),
            "constrained_u_mean_s": float(a["pi_u_star_mean_s"]),
            "constrained_l_mean_fees": float(a["pi_l_star_mean_fees"]),
            "constrained_l_mean_l": float(a["pi_l_star_mean_l"]),
            "constrained_l_mean_u": float(a["pi_l_star_mean_u"]),
            "constrained_l_mean_s": float(a["pi_l_star_mean_s"]),
            "local_lambda_loss_differs": as_bool(row["penalized_loss_differs_above_lambda_star"]),
            "representative_example": False,
        })
    representatives = sorted(
        (r for r in cells if r["support_label"] == "supported" and r["constrained_l_u_disagreement"]),
        key=lambda r: (r["cell_idx"], r["rho"]),
    )
    example = representatives[0]
    example["representative_example"] = True
    write_csv(LOCAL / "m3_penalty_method_cells.csv", cells)

    policies = aggregate_cell(example["cell_idx"])
    target = example["service_target"]
    surplus_path = envelope(policies, target, "u")
    loss_path = envelope(policies, target, "l")
    if surplus_path[-1]["policy_id"] != example["constrained_u_policy_id"]:
        raise RuntimeError("surplus path does not recover constrained U selector")
    if loss_path[-1]["policy_id"] != example["constrained_l_policy_id"]:
        raise RuntimeError("loss path does not recover constrained L selector")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7))
    plot_panel(axes[0], surplus_path, example["training_S0"], target,
               example["lambda_star_raw"], "surplus", example["constrained_u_policy_id"])
    plot_panel(axes[1], loss_path, example["training_S0"], target,
               example["lambda_star_raw"], "loss", example["constrained_u_policy_id"])
    axes[0].set_title("A. Penalized surplus recovers the constrained optimum", fontsize=10)
    axes[1].set_title("B. Penalized tracking difference converges elsewhere", fontsize=10)
    fig.suptitle(
        f"Exact penalty paths: supported cell {example['cell_idx']}, rho={example['rho']:g}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(LOCAL / "m3_penalty_method_figure.pdf", bbox_inches="tight")
    plt.close(fig)

    supported = [r for r in cells if r["support_label"] == "supported"]
    lambda_values = sorted(r["lambda_tilde_star"] for r in cells)
    by_rho = defaultdict(list); by_support = defaultdict(list)
    for row in cells:
        by_rho[row["rho"]].append(row["lambda_tilde_star"])
        by_support[row["support_label"]].append(row["lambda_tilde_star"])
    def q(values, p):
        values = sorted(values); pos = (len(values) - 1) * p
        lo, hi = math.floor(pos), math.ceil(pos)
        return values[lo] if lo == hi else values[lo] * (hi - pos) + values[hi] * (pos - lo)
    dist_lines = "\n".join(
        f"- rho={rho:g}: median={q(vals, .5):.6g}, p10={q(vals, .1):.6g}, p90={q(vals, .9):.6g}."
        for rho, vals in sorted(by_rho.items()))
    support_lines = "\n".join(
        f"- {label}: n={len(vals)}, median={q(vals, .5):.6g}, p10={q(vals, .1):.6g}, p90={q(vals, .9):.6g}."
        for label, vals in sorted(by_support.items()))
    report = f"""# M3 Exact-Penalty Method Summary

## Method hierarchy

The service-constrained LP-surplus frontier is the economic benchmark. The
shortfall-penalized surplus objective is its operational representation on the
finite policy set. Penalized gross unfavorable loss A is the primary
loss-only comparison. Signed tracking difference L=A-B is a secondary
diagnostic: its penalty can exclude low-service policies but cannot restore
omitted fee revenue.

This exhibit uses frozen training means, exact breakpoints, normalized
penalties, and A/L/U selectors only. It does not call the simulator or consume
a seed.

## Complete-grid results

- Exact penalized-surplus recovery for every lambda strictly above the
  grid-specific threshold: **270/270**.
- Primary constrained A/U selector disagreement: **158/270 (58.5%)**.
- Primary directly supported subset: **79/135 (58.5%)**.
- Secondary constrained L/U selector disagreement: **196/270 (72.6%)**.
- Secondary directly supported subset: **90/135 (66.7%)**.
- The A- and L-minimizing selectors differ in **100/270 (37.0%)** states
  and **43/135 (31.9%)** directly supported states.
- Penalized gross-loss selection immediately above the surplus threshold
  differs from constrained surplus in **242/270** states. This remains a
  secondary local-lambda diagnostic because it mixes infeasible-policy
  exclusion with feasible-set ranking.
- Fees minus L reproduces U in **270/270** states; the negative result concerns
loss-only selection, not LVR accounting.

The 158/270 and 196/270 counts compare selectors on the common feasible set.
Equivalently, they compare the objective-specific large-penalty limits after
each penalty has excluded infeasible policies. They are not the selector count
immediately to the right of the surplus threshold; that is the separate
242/270 local-lambda diagnostic above.

## Normalized threshold distribution

The normalization is `lambda_tilde_star = lambda_star * S0 / V0`, with
`V0=40,000,000`.

By service target:

{dist_lines}

By support class:

{support_lines}

Across all states, the median is {q(lambda_values, .5):.6g}, with p10
{q(lambda_values, .1):.6g} and p90 {q(lambda_values, .9):.6g}.

## Frozen representative example

The figure uses the lexicographically first `(cell_idx, rho)` among directly
supported states with constrained L/U disagreement: cell {example['cell_idx']},
rho={example['rho']:g} (sigma={example['sigma']}, z={example['z']},
speed={example['speed']}). This rule does not use effect magnitude.

Panel A shows the complete penalized-surplus envelope and marks
`lambda_tilde_star={example['lambda_tilde_star']:.6g}`. Above the threshold the
path selects `{example['constrained_u_policy_id']}` and remains service-feasible.
Panel B reconstructs the complete penalized-L envelope. Its large-lambda limit
is `{example['constrained_l_policy_id']}`, whereas constrained surplus selects
`{example['constrained_u_policy_id']}`. The policies are both feasible, but the
L minimizer has fees={example['constrained_l_mean_fees']:.1f},
L={example['constrained_l_mean_l']:.1f}, and U={example['constrained_l_mean_u']:.1f};
the surplus maximizer has fees={example['constrained_u_mean_fees']:.1f},
L={example['constrained_u_mean_l']:.1f}, and U={example['constrained_u_mean_u']:.1f}.

## Boundaries

`lambda_star` is finite-grid and training-block specific. It is not an
identified preference parameter, market shadow price, or structural service
price. The method constrains expected service, not pathwise reliability or
tail-state service. The final confirmed pair remains a separate held-out
illustration and was not used to choose the method figure.
"""
    (LOCAL / "m3_penalty_method_summary.md").write_text(report)
    caption = r"""\caption{Exact-penalty representation of service-constrained LP-surplus evaluation. Panel A traces the complete penalized-surplus solution path for the first directly supported cell--target state, in fixed lexicographic order, with constrained tracking-difference and surplus selectors that disagree. Above the finite-grid threshold $\widetilde{\lambda}^{\star}$, penalized surplus recovers the constrained-surplus optimizer. Panel B applies the same expected-service shortfall penalty to tracking difference. The selected policy eventually becomes feasible, but its large-penalty limit is the constrained-$L$ minimizer rather than the constrained-$U$ maximizer because fee revenue remains outside the loss-only objective. Shading marks infeasible path segments. Thresholds and paths use frozen training means and impose expected, not pathwise, service.}
"""
    (LOCAL / "m3_penalty_method_figure_caption.tex").write_text(caption)
    print(f"example=cell{example['cell_idx']},rho{example['rho']:g} surplus={len(surplus_path)} L={len(loss_path)}")


if __name__ == "__main__":
    main()
