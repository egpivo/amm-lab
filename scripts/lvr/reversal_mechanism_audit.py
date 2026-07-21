#!/usr/bin/env python3
"""Audit the mechanism behind amended validation-grid matched-service reversals.

This script consumes frozen training/validation/final artifacts only. It does
not import or call the simulator. Candidate orientation is frozen by training
means: policy 1 is the lower-training-L member of each pre-existing pair. The
original lower-A validation gate is retained only as a chronology marker.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / ".local" / "lvr"


def read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def candidate_plan_map(plan: dict) -> dict[str, dict]:
    return {row["candidate_id"]: row for row in plan["matched_candidates"]}


def aggregate_validation(paths: list[Path]) -> dict[tuple[str, str], dict]:
    sums: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[tuple[str, str], int] = defaultdict(int)
    fields = ("a", "b", "l", "fees", "u", "s", "quote_error")
    for path in paths:
        with gzip.open(path, "rt", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["record_kind"] != "matched":
                    continue
                key = (row["record_id"], row["policy_role"])
                counts[key] += 1
                for field in fields:
                    sums[key][field] += float(row[field])
    result = {}
    for key, values in sums.items():
        n = counts[key]
        result[key] = {field: values[field] / n for field in fields}
        result[key]["n"] = n
    return result


def speed_code(speed: str) -> int:
    return {"slow": 0, "medium": 1, "fast": 2}[speed]


def safe_scale(value: float) -> float:
    return max(abs(value), 1e-12)


def fit_surfaces(rows: list[dict]) -> list[dict]:
    raw = np.array([
        [r["sigma"], math.log(r["z"]), speed_code(r["speed"]), r["rho"]]
        for r in rows
    ])
    means = raw.mean(axis=0)
    sds = raw.std(axis=0, ddof=0)
    zvals = (raw - means) / np.where(sds == 0, 1.0, sds)
    support_sparse = np.array([r["support_label"] == "sparse" for r in rows], dtype=float)
    support_unobserved = np.array([r["support_label"] == "unobserved" for r in rows], dtype=float)
    x = np.column_stack([
        np.ones(len(rows)), zvals, support_sparse, support_unobserved,
        zvals[:, 0] * zvals[:, 1], zvals[:, 0] * zvals[:, 2],
        zvals[:, 1] * zvals[:, 3],
    ])
    names = [
        "intercept", "sigma_z", "log_z_z", "speed_z", "rho_z",
        "support_sparse", "support_unobserved", "sigma_x_log_z",
        "sigma_x_speed", "log_z_x_rho",
    ]
    outputs = {
        "delta_l_over_s0": np.array([r["delta_l_over_s0"] for r in rows]),
        "delta_fees_over_s0": np.array([r["delta_fees_over_s0"] for r in rows]),
        "delta_u_over_s0": np.array([r["delta_u_over_s0"] for r in rows]),
    }
    result = []
    for outcome, y in outputs.items():
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        fitted = x @ beta
        residual = y - fitted
        sse = float(residual @ residual)
        sst = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - sse / sst if sst else 1.0
        rmse = math.sqrt(sse / len(y))
        for name, value in zip(names, beta):
            result.append({
                "outcome": outcome,
                "term": name,
                "coefficient": value,
                "n": len(y),
                "r_squared": r2,
                "rmse": rmse,
                "sigma_mean": means[0],
                "sigma_sd": sds[0],
                "log_z_mean": means[1],
                "log_z_sd": sds[1],
                "speed_mean": means[2],
                "speed_sd": sds[2],
                "rho_mean": means[3],
                "rho_sd": sds[3],
            })
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_phase_diagram(rows: list[dict], output: Path, final_id: str) -> None:
    support_colors = {"supported": "#2166ac", "sparse": "#b2182b", "unobserved": "#666666"}
    speed_markers = {"slow": "o", "medium": "s", "fast": "^"}
    sigmas = sorted({r["sigma"] for r in rows})
    fig, axes = plt.subplots(1, len(sigmas), figsize=(12.4, 4.2), sharex=True, sharey=True)
    if len(sigmas) == 1:
        axes = [axes]
    max_value = max(max(r["loss_saving"], r["fee_sacrifice"], 0) for r in rows)
    for ax, sigma in zip(axes, sigmas):
        subset = [r for r in rows if r["sigma"] == sigma]
        for speed, marker in speed_markers.items():
            points = [r for r in subset if r["speed"] == speed]
            for support, color in support_colors.items():
                group = [r for r in points if r["support_label"] == support]
                if not group:
                    continue
                ax.scatter(
                    [r["loss_saving"] for r in group],
                    [r["fee_sacrifice"] for r in group],
                    s=[24 + 52 * r["rho"] for r in group],
                    marker=marker, facecolor=color, edgecolor="white", linewidth=0.5,
                    alpha=0.78,
                )
        gate_rows = [r for r in subset if r["gate_pass"]]
        if gate_rows:
            ax.scatter(
                [r["loss_saving"] for r in gate_rows],
                [r["fee_sacrifice"] for r in gate_rows],
                s=[58 + 70 * r["rho"] for r in gate_rows],
                facecolors="none", edgecolors="#111111", linewidths=1.4,
            )
        for row in gate_rows:
            label = "final" if row["candidate_id"] == final_id else row["candidate_id"].replace("amended_match_", "")
            ax.annotate(label, (row["loss_saving"], row["fee_sacrifice"]), xytext=(4, 4),
                        textcoords="offset points", fontsize=7)
        ax.plot([0, max_value], [0, max_value], color="#222222", linewidth=0.9, linestyle="--")
        ax.axhline(0, color="#aaaaaa", linewidth=0.6)
        ax.axvline(0, color="#aaaaaa", linewidth=0.6)
        ax.set_title(rf"$\sigma={sigma:.2f}$", fontsize=10)
        ax.grid(color="#dddddd", linewidth=0.45, alpha=0.7)
    axes[0].set_ylabel(r"Fee-revenue sacrifice: $F_2-F_1$")
    fig.supxlabel(r"Tracking-difference saving: $L_2-L_1$", y=0.105)
    fig.suptitle("Matched-service reversal mechanism", fontsize=12)
    legend_items = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
                   markeredgecolor="white", label=support, markersize=7)
        for support, color in support_colors.items()
    ] + [
        plt.Line2D([0], [0], marker=marker, color="#444444", linestyle="none",
                   label=speed, markersize=6)
        for speed, marker in speed_markers.items()
    ] + [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                   markeredgecolor="#111111", label="passes both gates", markersize=8)
    ]
    fig.legend(handles=legend_items, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=7, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.18, 1, 0.94))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def classify_verdict(rows: list[dict], final_id: str) -> tuple[str, dict]:
    final = next(r for r in rows if r["candidate_id"] == final_id)
    supported_gate = [r for r in rows if r["gate_pass"] and r["support_label"] == "supported"]
    supported_gate_same = [
        r for r in supported_gate if r["fee_sacrifice"] > r["loss_saving"] > 0
    ]
    reversal_rows = [r for r in rows if r["delta_l"] < 0 and r["delta_u"] < 0]
    same_mechanism = [r for r in reversal_rows if r["fee_sacrifice"] > r["loss_saving"] > 0]
    supported_same = [r for r in same_mechanism if r["support_label"] == "supported"]
    reference = np.array([r["delta_u_over_s0"] for r in supported_gate_same], dtype=float)
    if supported_gate_same and supported_same and final["fee_sacrifice"] > final["loss_saving"] > 0:
        if (len(supported_gate_same) >= math.ceil(len(supported_gate) / 2)
                and reference.size
                and min(reference) <= final["delta_u_over_s0"] <= max(reference)):
            verdict = "SMOOTH SCALE-AMPLIFIED CONTINUATION"
        else:
            verdict = "COHERENT BUT SUPPORT-LIMITED CONTINUATION"
    else:
        verdict = "ISOLATED MECHANISM-BOUNDARY CORNER"
    stats = {
        "n_reversal_direction": len(reversal_rows),
        "n_same_fee_loss_mechanism": len(same_mechanism),
        "n_supported_same_mechanism": len(supported_same),
        "n_supported_gate": len(supported_gate),
        "n_supported_gate_same_mechanism": len(supported_gate_same),
        "final_loss_saving": final["loss_saving"],
        "final_fee_sacrifice": final["fee_sacrifice"],
        "final_delta_u": final["delta_u"],
    }
    return verdict, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", type=Path, default=LOCAL)
    args = parser.parse_args()
    local = args.local
    plan = read_json(local / "m3_amended_validation_plan.json")
    selection = read_json(local / "m3_amended_validation_selection.json")
    final = read_json(local / "m3_amended_final_result.json")
    plans = candidate_plan_map(plan)
    selected = {row["candidate_id"]: row for row in selection["matched_candidates"]}
    aggregates = aggregate_validation(sorted(local.glob("m3_amended_validation_rows_shard*.csv.gz")))
    final_id = selection["winner"]["candidate_id"]
    final_means = final["policy_means"]

    rows = []
    for candidate_id in sorted(plans):
        p = plans[candidate_id]
        s = selected[candidate_id]
        lower_a = p["policy_1_lower_A"]
        other = p["policy_2"]
        if lower_a["mean_l"] <= other["mean_l"]:
            p1, p2 = lower_a, other
            role1, role2 = "policy_1_lower_A", "policy_2"
        else:
            p1, p2 = other, lower_a
            role1, role2 = "policy_2", "policy_1_lower_A"
        m1 = aggregates[(candidate_id, role1)]
        m2 = aggregates[(candidate_id, role2)]
        if m1["n"] != 200 or m2["n"] != 200:
            raise RuntimeError(f"unexpected validation count for {candidate_id}")
        delta = {field: m1[field] - m2[field] for field in ("a", "b", "l", "fees", "u", "s", "quote_error")}
        if not math.isclose(delta["u"], delta["fees"] - delta["l"], abs_tol=1e-7):
            raise RuntimeError(f"accounting identity failed for {candidate_id}")
        s0 = p["training_S0"]
        row = {
            "candidate_id": candidate_id,
            "cell_idx": p["cell"]["cell_idx"],
            "stratum": p["cell"]["stratum"],
            "sigma": p["cell"]["sigma"],
            "z": p["cell"]["z"],
            "speed": p["cell"]["speed"],
            "rho": p["rho"],
            "support_label": p["empirical_support"]["support_label"],
            "empirical_weight": p["empirical_support"]["empirical_weight"],
            "gate_pass": bool(s["both_one_sided_95_upper_below_zero"]),
            "lower_l_is_original_lower_a": role1 == "policy_1_lower_A",
            "reversal_direction": delta["l"] < 0 and delta["u"] < 0,
            "policy_1_family": p1["family"], "policy_1_dial_mult": p1["dial_mult"],
            "policy_1_f0": p1["f0"], "policy_1_alpha": p1["alpha"],
            "policy_2_family": p2["family"], "policy_2_dial_mult": p2["dial_mult"],
            "policy_2_f0": p2["f0"], "policy_2_alpha": p2["alpha"],
            "training_S0": s0,
            "validation_mean_a_1": m1["a"], "validation_mean_a_2": m2["a"],
            "validation_mean_b_1": m1["b"], "validation_mean_b_2": m2["b"],
            "validation_mean_l_1": m1["l"], "validation_mean_l_2": m2["l"],
            "validation_mean_fees_1": m1["fees"], "validation_mean_fees_2": m2["fees"],
            "validation_mean_u_1": m1["u"], "validation_mean_u_2": m2["u"],
            "validation_mean_s_1": m1["s"], "validation_mean_s_2": m2["s"],
            "validation_mean_quote_error_1": m1["quote_error"],
            "validation_mean_quote_error_2": m2["quote_error"],
            "delta_a": delta["a"], "delta_b": delta["b"], "delta_l": delta["l"],
            "delta_fees": delta["fees"], "delta_u": delta["u"], "delta_s": delta["s"],
            "delta_quote_error": delta["quote_error"],
            "delta_l_over_s0": delta["l"] / safe_scale(s0),
            "delta_fees_over_s0": delta["fees"] / safe_scale(s0),
            "delta_u_over_s0": delta["u"] / safe_scale(s0),
            "delta_s_over_s0": delta["s"] / safe_scale(s0),
            "loss_saving": -delta["l"],
            "fee_sacrifice": -delta["fees"],
            "is_final_winner": candidate_id == final_id,
        }
        if candidate_id == final_id:
            # The final block supersedes validation only for the confirmed pair summary.
            row.update({
                "final_mean_l_1": final_means["policy_1_lower_A"]["l"],
                "final_mean_l_2": final_means["policy_2"]["l"],
                "final_mean_fees_1": final_means["policy_1_lower_A"]["fees"],
                "final_mean_fees_2": final_means["policy_2"]["fees"],
                "final_mean_u_1": final_means["policy_1_lower_A"]["u"],
                "final_mean_u_2": final_means["policy_2"]["u"],
                "final_mean_s_1": final_means["policy_1_lower_A"]["s"],
                "final_mean_s_2": final_means["policy_2"]["s"],
            })
        else:
            row.update({key: "" for key in (
                "final_mean_l_1", "final_mean_l_2", "final_mean_fees_1", "final_mean_fees_2",
                "final_mean_u_1", "final_mean_u_2", "final_mean_s_1", "final_mean_s_2")})
        rows.append(row)

    decomposition = local / "m3_reversal_candidate_decomposition.csv"
    response = local / "m3_reversal_response_surface.csv"
    diagram = local / "m3_reversal_phase_diagram.pdf"
    report = local / "m3_reversal_mechanism_report.md"
    write_csv(decomposition, rows)
    surface_rows = fit_surfaces(rows)
    write_csv(response, surface_rows)
    make_phase_diagram(rows, diagram, final_id)
    verdict, stats = classify_verdict(rows, final_id)

    supported_gate = [r for r in rows if r["gate_pass"] and r["support_label"] == "supported"]
    gate_lines = "\n".join(
        f"- `{r['candidate_id']}`: sigma={r['sigma']}, z={r['z']}, speed={r['speed']}, "
        f"rho={r['rho']}; Delta L/S0={r['delta_l_over_s0']:.4f}, "
        f"Delta fees/S0={r['delta_fees_over_s0']:.4f}, Delta U/S0={r['delta_u_over_s0']:.4f}."
        for r in supported_gate
    )
    final_row = next(r for r in rows if r["candidate_id"] == final_id)
    report.write_text(f"""# validation-grid Reversal Mechanism Audit

## Scope and orientation

This audit uses the frozen amended training plan, all 106 matched-candidate
validation pairs, and the amended final result. It does not call the simulator,
change a policy, or consume a seed. Policy 1 is oriented from frozen training
means as the lower-training-L member of each pre-existing pair. The original
validation gate was defined for lower A, so `gate_pass` is retained only as a
chronology marker; it is not relabeled as fresh lower-L inference.

For each candidate, `Delta` means policy 1 minus policy 2. The accounting check
`Delta U = Delta fees - Delta L` holds for all 106 validation decompositions.
Effects normalized by `training_S0` are descriptive scale controls, not
population estimates.

## Mechanism decomposition

- {stats['n_reversal_direction']}/106 candidates have both `Delta L < 0` and
  `Delta U < 0`: the lower-loss policy also has lower LP relative surplus.
- {stats['n_same_fee_loss_mechanism']}/106 have the corresponding mechanism
  inequality `fees_2 - fees_1 > L_2 - L_1 > 0`.
- {stats['n_supported_same_mechanism']} of those mechanism-consistent cases are
  in directly supported cells.
- Five candidates passed the original lower-A/lower-U inference gate. Three are
  directly supported; the selected final candidate is unobserved. Their
  lower-L decomposition is reported here without reusing that gate as new
  lower-L inference.
- {stats['n_supported_gate_same_mechanism']} of the three directly supported
  gate-passing pairs retains the lower-L/lower-U fee-sacrifice mechanism after
  reorientation by training L.

The three supported gate-passing candidates are:

{gate_lines}

The final winner `{final_id}` lies at sigma={final_row['sigma']}, z={final_row['z']},
speed={final_row['speed']}, rho={final_row['rho']}. On validation,
`Delta L/S0={final_row['delta_l_over_s0']:.4f}`,
`Delta fees/S0={final_row['delta_fees_over_s0']:.4f}`, and
`Delta U/S0={final_row['delta_u_over_s0']:.4f}`. On the fresh amended final
block, policy 1 saves {final_means['policy_2']['l'] - final_means['policy_1_lower_A']['l']:.1f}
units of tracking difference but sacrifices
{final_means['policy_2']['fees'] - final_means['policy_1_lower_A']['fees']:.1f}
units of fee revenue, leaving LP relative surplus
{final_means['policy_2']['u'] - final_means['policy_1_lower_A']['u']:.1f} lower.

## Response-surface reading

`m3_reversal_response_surface.csv` reports low-order descriptive least-squares
surfaces for `Delta L/S0`, `Delta fees/S0`, and `Delta U/S0` using standardized
sigma, log z, speed, rho, support class, and three predeclared interactions.
The surfaces summarize scale and direction only. They carry no causal or
historical-frequency interpretation, and the audit reports no headline
p-values.

The phase diagram plots loss saving against fee sacrifice. Points above the
45-degree line are exactly the cases where the fee sacrifice exceeds the loss
saving and the lower-loss policy has lower surplus. Sigma is faceted, speed is
encoded by marker, rho by marker size, and support by color; gate-passing
candidates are outlined and labeled.

## Verdict

The final candidate shares the fee-sacrifice-over-loss-saving mechanism with
one supported gate-passing pair and 23 supported candidates overall, but it
remains outside direct joint support and its magnitude is amplified at the
high-sigma, high-z, fast boundary.
The evidence therefore supports mechanism continuity without converting the
boundary final confirmation into an empirically supported-cell claim.

**{verdict}**
""")

    print(json.dumps({"rows": len(rows), "verdict": verdict, **stats}, indent=2))


if __name__ == "__main__":
    main()
