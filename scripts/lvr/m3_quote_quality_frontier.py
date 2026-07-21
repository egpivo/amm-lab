#!/usr/bin/env python3
"""Construct the frozen-training service/quote-quality surplus frontier."""

from __future__ import annotations

import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / ".local" / "lvr"
RHOS = (0.2, 0.4, 0.6, 0.8, 0.95)
METRICS = ("a", "b", "l", "fees", "u", "s", "quote_error")


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=2e-10, abs_tol=2e-10)


def pid(dial: float, alpha: float) -> str:
    return f"gap_d{format(dial, 'g').replace('.', 'p')}_a{format(alpha, 'g').replace('.', 'p')}"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def quantile(values: list[float], p: float) -> float:
    values = sorted(values)
    pos = (len(values) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] if lo == hi else values[lo] * (hi - pos) + values[hi] * (pos - lo)


def main() -> None:
    selection = json.loads((LOCAL / "m3_amended_training_selection.json").read_text())
    support = json.loads((LOCAL / "m3_joint_support.json").read_text())
    support_lookup = {
        (r["stratum"], float(r["sigma"]), float(r["z"])): r
        for r in support["market_state_cells"]
    }
    sums = defaultdict(lambda: defaultdict(float)); counts = defaultdict(int)
    specs, cells = {}, {}
    row_count = 0
    for path in sorted(LOCAL.glob("m3_amended_training_rows_shard*.csv.gz")):
        with gzip.open(path, "rt", newline="") as handle:
            for row in csv.DictReader(handle):
                row_count += 1
                cell_idx = int(row["cell_idx"])
                key = (cell_idx, row["family"], float(row["dial_mult"]), float(row["alpha"]))
                counts[key] += 1
                for metric in METRICS:
                    sums[key][metric] += float(row[metric])
                specs[key] = {
                    "family": row["family"], "dial_mult": float(row["dial_mult"]),
                    "f0": float(row["f0"]), "alpha": float(row["alpha"]),
                    "fee_cap": float(row["fee_cap"]),
                }
                cells[cell_idx] = {
                    "cell_idx": cell_idx, "stratum": row["stratum"],
                    "sigma": float(row["sigma"]), "z": float(row["z"]), "speed": row["speed"],
                }
    if row_count != 518_400 or set(counts.values()) != {100}:
        raise RuntimeError("training row integrity failed")
    means = {
        key: {f"mean_{m}": values[m] / counts[key] for m in METRICS}
        for key, values in sums.items()
    }
    for cell_idx in range(54):
        for dial in selection["grid"]["dial_mults"]:
            if means[(cell_idx, "static", float(dial), 0.0)] != means[(cell_idx, "gap", float(dial), 0.0)]:
                raise RuntimeError("static boundary alias failed")

    breakpoint_rows, frontier_rows, case_summaries = [], [], []
    for cell_idx in range(54):
        info = cells[cell_idx]
        sup = support_lookup[(info["stratum"], info["sigma"], info["z"])]
        s0 = means[(cell_idx, "static", 1.0, 0.0)]["mean_s"]
        policies = []
        for key, values in means.items():
            ci, family, dial, alpha = key
            if ci != cell_idx or family != "gap":
                continue
            policy = {**specs[key], **values, "policy_id": pid(dial, alpha)}
            if not close(policy["mean_l"], policy["mean_a"] - policy["mean_b"]):
                raise RuntimeError("L=A-B identity failed")
            if not close(policy["mean_u"], policy["mean_fees"] - policy["mean_l"]):
                raise RuntimeError("U=fees-L identity failed")
            policies.append(policy)
        if len(policies) != 84:
            raise RuntimeError("canonical policy count failed")

        for rho in RHOS:
            target = rho * s0
            service_feasible = [p for p in policies if p["mean_s"] >= target - 1e-10]
            service_optimum = sorted(
                service_feasible,
                key=lambda p: (-p["mean_u"], p["alpha"], p["dial_mult"], p["policy_id"]),
            )[0]
            thresholds = sorted({p["mean_quote_error"] for p in service_feasible})
            prior_id = None; interval_index = -1
            case_changed = False; nonterminal_differs = 0; costs = []
            for index, threshold in enumerate(thresholds):
                feasible = [p for p in service_feasible if p["mean_quote_error"] <= threshold + 1e-14]
                selected = sorted(
                    feasible,
                    key=lambda p: (-p["mean_u"], p["alpha"], p["dial_mult"], p["policy_id"]),
                )[0]
                optimum_feasible = service_optimum["mean_quote_error"] <= threshold + 1e-14
                cost = service_optimum["mean_u"] - selected["mean_u"]
                if cost < -1e-7:
                    raise RuntimeError("negative surplus cost")
                cost = max(cost, 0.0); costs.append(cost)
                differs = selected["policy_id"] != service_optimum["policy_id"]
                if index < len(thresholds) - 1 and differs:
                    nonterminal_differs += 1
                case_changed = case_changed or differs
                common = {
                    **info, "support_label": sup["support_label"],
                    "observed_pool_weeks": sup["observed_pool_weeks"],
                    "rho": rho, "training_S0": s0, "service_target": target,
                }
                row = {
                    **common,
                    "threshold_index": index,
                    "mean_quote_error_threshold": threshold,
                    "n_service_feasible": len(service_feasible),
                    "n_joint_feasible": len(feasible),
                    "selected_policy_id": selected["policy_id"],
                    "selected_class": "static_boundary" if selected["alpha"] == 0 else "adaptive_gap",
                    "selected_dial_mult": selected["dial_mult"], "selected_f0": selected["f0"],
                    "selected_alpha": selected["alpha"], "selected_mean_a": selected["mean_a"],
                    "selected_mean_b": selected["mean_b"], "selected_mean_l": selected["mean_l"],
                    "selected_mean_fees": selected["mean_fees"], "selected_mean_u": selected["mean_u"],
                    "selected_mean_s": selected["mean_s"],
                    "selected_mean_quote_error": selected["mean_quote_error"],
                    "service_only_optimum_id": service_optimum["policy_id"],
                    "service_only_optimum_quote_error": service_optimum["mean_quote_error"],
                    "service_only_optimum_feasible": optimum_feasible,
                    "differs_from_service_only_optimum": differs,
                    "surplus_cost": cost,
                }
                breakpoint_rows.append(row)
                if selected["policy_id"] != prior_id:
                    interval_index += 1
                    if frontier_rows and frontier_rows[-1]["cell_idx"] == cell_idx and frontier_rows[-1]["rho"] == rho:
                        frontier_rows[-1]["mean_quote_error_end_exclusive"] = threshold
                    frontier_rows.append({
                        **common, "frontier_interval_index": interval_index,
                        "mean_quote_error_start_inclusive": threshold,
                        "mean_quote_error_end_exclusive": "inf",
                        "n_joint_feasible_at_start": len(feasible),
                        "selected_policy_id": selected["policy_id"],
                        "selected_class": row["selected_class"],
                        "selected_dial_mult": selected["dial_mult"], "selected_f0": selected["f0"],
                        "selected_alpha": selected["alpha"], "selected_mean_a": selected["mean_a"],
                        "selected_mean_b": selected["mean_b"], "selected_mean_l": selected["mean_l"],
                        "selected_mean_fees": selected["mean_fees"], "selected_mean_u": selected["mean_u"],
                        "selected_mean_s": selected["mean_s"],
                        "selected_mean_quote_error": selected["mean_quote_error"],
                        "surplus_cost_at_start": cost,
                    })
                    prior_id = selected["policy_id"]
            if breakpoint_rows[-1]["selected_policy_id"] != service_optimum["policy_id"]:
                raise RuntimeError("terminal threshold does not recover service-only optimum")
            case_summaries.append({
                "cell_idx": cell_idx, "rho": rho, "support_label": sup["support_label"],
                "n_thresholds": len(thresholds), "n_policy_intervals": interval_index + 1,
                "constraint_changes_selector": case_changed,
                "n_nonterminal_thresholds_differing": nonterminal_differs,
                "max_surplus_cost": max(costs), "median_surplus_cost": quantile(costs, 0.5),
            })

    write_csv(LOCAL / "m3_quote_quality_breakpoints.csv", breakpoint_rows)
    write_csv(LOCAL / "m3_quote_quality_frontier.csv", frontier_rows)
    final = json.loads((LOCAL / "m3_amended_final_result.json").read_text())
    p1 = final["policy_means"]["policy_1_lower_A"]
    p2 = final["policy_means"]["policy_2"]
    common_threshold = max(p1["quote_error"], p2["quote_error"])
    changed = [r for r in case_summaries if r["constraint_changes_selector"]]
    supported_cases = [r for r in case_summaries if r["support_label"] == "supported"]
    supported_changed = [r for r in supported_cases if r["constraint_changes_selector"]]
    all_costs = [r["surplus_cost"] for r in breakpoint_rows]
    positive_costs = [x for x in all_costs if x > 1e-10]
    report = f"""# M3 Quote-Quality Constrained Frontier

## Scope

This audit uses the frozen 518,400-row amended training block. It evaluates the
canonical 84-member gap family, including its alpha-zero static boundary, under
both `mean_S >= rho*S0` and `mean_quote_error <= m_bar`. It does not call the
simulator, change a policy, or consume a seed.

No quote threshold is selected after observing an outcome. The breakpoint CSV
contains every distinct attainable mean-quote-error threshold among
service-feasible policies for all 270 cell-target states. The frontier CSV
compresses the same surface into intervals over which the surplus selector is
constant. Ties use maximum U, lower alpha, lower dial, then stable policy id.

## Complete surface

- Cell-target states: **270**; directly supported states: **135**.
- Attainable threshold rows: **{len(breakpoint_rows):,}**.
- Constant-selector frontier intervals: **{len(frontier_rows):,}**.
- The quote-quality constraint changes the service-only surplus selector at
  one or more attainable thresholds in **{len(changed)}/270** states and
  **{len(supported_changed)}/135** directly supported states.
- At the least restrictive attainable threshold, the selector recovers the
  service-only constrained-surplus optimum in **270/270** states.
- Across all attainable thresholds, the median surplus cost is
  **{quantile(all_costs, .5):.3f}**, p90 is **{quantile(all_costs, .9):.3f}**,
  and the maximum is **{max(all_costs):.3f}** in simulator value units.
- Conditional on a positive cost ({len(positive_costs):,} thresholds), the
  median is **{quantile(positive_costs, .5):.3f}** and p90 is
  **{quantile(positive_costs, .9):.3f}**.

These counts describe the frozen finite training grid. They are not historical
market frequencies and do not identify a preferred quote-quality threshold.

## Final confirmed pair

The final lower-loss policy has mean quote error {p1['quote_error']:.6g}; the
higher-surplus policy has {p2['quote_error']:.6g}. A common threshold must be at
least **{common_threshold:.6g}** for both policies to remain eligible. At that
threshold the pair remains directly comparable and its L, fees, U, and service
values are unchanged; any tighter threshold excludes the lower-loss policy
before it excludes the higher-surplus policy. This is a quote-quality
eligibility observation, not a new held-out test.

## Interpretation boundary

Expected served volume does not by itself certify quote freshness. Conversely,
mean quote error does not establish pathwise execution quality or trader
welfare. This audit adds a complete two-constraint finite-grid robustness
surface without choosing a favorable operating threshold.
"""
    (LOCAL / "m3_quote_quality_frontier_report.md").write_text(report)
    print(f"breakpoints={len(breakpoint_rows)} intervals={len(frontier_rows)} changed={len(changed)}/270")


if __name__ == "__main__":
    main()
