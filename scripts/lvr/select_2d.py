#!/usr/bin/env python3
"""Training-only selection for the amended two-dimensional validation-grid gap grid."""

import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path("/Users/joseph/amm-lab")
LVR = ROOT / ".local/lvr/workspace"
ROW_PATHS = [LVR / f"m3_amended_training_rows_shard{i}.csv.gz" for i in range(6)]
SUPPORT_PATH = LVR / "m3_joint_support.json"
AUDIT_PATH = LVR / "m3_policy_support_audit.json"
OUT = LVR / "m3_amended_training_selection.json"
FRONTIER_CSV = LVR / "m3_amended_training_frontier.csv"
DIAGNOSTIC_PLAN = LVR / "m3_amended_diagnostic_plan.json"
RHOS = [0.2, 0.4, 0.6, 0.8, 0.95]
SUPPORT_TOL = 0.10
PAIR_TOL = 0.05
METRICS = (
    "l",
    "a",
    "b",
    "a_arb",
    "a_fund",
    "b_fund",
    "u",
    "fees",
    "fees_arb",
    "fees_fund",
    "s",
    "potential",
    "alloc_amm",
    "alloc_cex",
    "alloc_unserved",
    "fill_incidence",
    "conditional_fill_size",
    "a_arb_per_served",
    "a_fund_per_served",
    "a_total_per_served",
    "quote_error",
    "n_fund_events",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n").encode()


def freeze(path: Path, value: object) -> str:
    payload = json_bytes(value)
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"refusing to overwrite frozen artifact: {path}")
    if not path.exists():
        path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n")
    return digest


def point(cell_idx: int, family: str, dial: float, alpha: float) -> dict:
    values = means[(cell_idx, family, dial, alpha)]
    return {
        "family": family,
        "dial_mult": dial,
        "f0": specs[(cell_idx, family, dial, alpha)]["f0"],
        "alpha": alpha,
        "fee_cap": specs[(cell_idx, family, dial, alpha)]["fee_cap"],
        **{f"mean_{metric}": values[metric] for metric in METRICS},
    }


def best_u(points: list[dict]) -> dict:
    assert points
    return min(
        points,
        key=lambda p: (-p["mean_u"], p["alpha"], p["dial_mult"]),
    )


def nearest_service(points: list[dict], target: float) -> dict:
    return min(
        points,
        key=lambda p: (abs(p["mean_s"] - target), p["alpha"], p["dial_mult"]),
    )


sums = defaultdict(lambda: defaultdict(float))
metric_counts = defaultdict(lambda: defaultdict(int))
counts = defaultdict(int)
specs = {}
cell_info = {}
row_counts = {}
for path in ROW_PATHS:
    n = 0
    with gzip.open(path, "rt", newline="") as f:
        for row in csv.DictReader(f):
            cell_idx = int(row["cell_idx"])
            family = row["family"]
            dial = float(row["dial_mult"])
            alpha = float(row["alpha"])
            key = (cell_idx, family, dial, alpha)
            counts[key] += 1
            n += 1
            for metric in METRICS:
                if row[metric] == "":
                    continue
                value = float(row[metric])
                assert math.isfinite(value)
                sums[key][metric] += value
                metric_counts[key][metric] += 1
            specs[key] = {
                "f0": float(row["f0"]),
                "fee_cap": float(row["fee_cap"]),
                "alpha_zero_static_alias": row["alpha_zero_static_alias"] == "true",
            }
            cell_info[cell_idx] = {
                "cell_idx": cell_idx,
                "stratum": row["stratum"],
                "sigma": float(row["sigma"]),
                "z": float(row["z"]),
                "speed": row["speed"],
            }
    row_counts[path.name] = n

assert sum(row_counts.values()) == 54 * 96 * 100
assert len(counts) == 54 * 96
assert set(counts.values()) == {100}
means = {
    key: {
        metric: (
            values[metric] / metric_counts[key][metric]
            if metric_counts[key][metric]
            else None
        )
        for metric in METRICS
    }
    for key, values in sums.items()
}

dials = sorted({key[2] for key in means})
alphas = sorted({key[3] for key in means if key[1] == "gap"})
assert dials == [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.5, 7.0, 10.0, 15.0, 25.0, 40.0]
assert alphas == [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
for cell_idx in range(54):
    for dial in dials:
        static = means[(cell_idx, "static", dial, 0.0)]
        alias = means[(cell_idx, "gap", dial, 0.0)]
        assert static == alias
        assert specs[(cell_idx, "gap", dial, 0.0)]["alpha_zero_static_alias"]

support = json.loads(SUPPORT_PATH.read_text())
support_lookup = {
    (c["stratum"], c["sigma"], c["z"]): c for c in support["market_state_cells"]
}

cells = []
frontier_rows = []
matched_candidates = []
diagnostic_assignments = []
for cell_idx in range(54):
    info = cell_info[cell_idx]
    static_points = [point(cell_idx, "static", dial, 0.0) for dial in dials]
    gap_points = [point(cell_idx, "gap", dial, alpha) for alpha in alphas for dial in dials]
    positive_gap_points = [p for p in gap_points if p["alpha"] > 0]
    s0 = means[(cell_idx, "static", 1.0, 0.0)]["s"]
    cell_result = {
        "cell": info,
        "empirical_support": support_lookup[(info["stratum"], info["sigma"], info["z"])],
        "S0_training": s0,
        "targets": [],
        "argmin_A_static": min(static_points, key=lambda p: (p["mean_a"], p["dial_mult"])),
        "argmin_A_gap": min(
            gap_points, key=lambda p: (p["mean_a"], p["alpha"], p["dial_mult"])
        ),
    }
    for role, policy in (
        ("argmin_A_static", cell_result["argmin_A_static"]),
        ("argmin_A_gap", cell_result["argmin_A_gap"]),
    ):
        diagnostic_assignments.append(
            {"record_id": f"cell{cell_idx:03d}_{role}", "role": role, "cell": info, "policy": policy}
        )

    for rho in RHOS:
        target = rho * s0
        static_matched = nearest_service(static_points, target)
        gap_positive_matched = nearest_service(positive_gap_points, target)
        static_frontier = best_u([p for p in static_points if p["mean_s"] >= target])
        gap_frontier = best_u([p for p in gap_points if p["mean_s"] >= target])
        assert gap_frontier["mean_u"] >= static_frontier["mean_u"] - 1e-12
        static_gap = abs(static_matched["mean_s"] - target) / s0
        gap_gap = abs(gap_positive_matched["mean_s"] - target) / s0
        pair_gap = abs(static_matched["mean_s"] - gap_positive_matched["mean_s"]) / s0
        orientation = None
        if (
            static_matched["mean_a"] < gap_positive_matched["mean_a"]
            and static_matched["mean_u"] < gap_positive_matched["mean_u"]
        ):
            orientation = (static_matched, gap_positive_matched)
        elif (
            gap_positive_matched["mean_a"] < static_matched["mean_a"]
            and gap_positive_matched["mean_u"] < static_matched["mean_u"]
        ):
            orientation = (gap_positive_matched, static_matched)
        candidate_id = None
        if (
            static_gap <= SUPPORT_TOL
            and gap_gap <= SUPPORT_TOL
            and pair_gap <= PAIR_TOL
            and orientation is not None
        ):
            lower_a, other = orientation
            candidate_id = f"amended_match_c{cell_idx:03d}_rho{int(round(100*rho)):02d}"
            matched_candidates.append(
                {
                    "candidate_id": candidate_id,
                    "cell": info,
                    "empirical_support": cell_result["empirical_support"],
                    "rho": rho,
                    "training_S0": s0,
                    "target_s_training": target,
                    "policy_1_lower_A": lower_a,
                    "policy_2": other,
                    "service_mismatch_over_training_S0": pair_gap,
                    "policy_1_target_gap_over_training_S0": abs(lower_a["mean_s"] - target) / s0,
                    "policy_2_target_gap_over_training_S0": abs(other["mean_s"] - target) / s0,
                    "training_delta_A": lower_a["mean_a"] - other["mean_a"],
                    "training_delta_U": lower_a["mean_u"] - other["mean_u"],
                }
            )
        target_result = {
            "rho": rho,
            "target_s_training": target,
            "static_matched": static_matched,
            "gap_positive_matched": gap_positive_matched,
            "static_target_gap_over_S0": static_gap,
            "gap_positive_target_gap_over_S0": gap_gap,
            "pair_mismatch_over_S0": pair_gap,
            "matched_candidate_id": candidate_id,
            "static_frontier": static_frontier,
            "gap_frontier": gap_frontier,
            "strict_gap_frontier_improvement": (
                gap_frontier["alpha"] > 0
                and gap_frontier["mean_u"] > static_frontier["mean_u"]
            ),
            "training_delta_U_gap_minus_static": gap_frontier["mean_u"]
            - static_frontier["mean_u"],
        }
        cell_result["targets"].append(target_result)
        frontier_rows.append({"cell": info, "empirical_support": cell_result["empirical_support"], **target_result})
        for role, policy in (
            ("static_matched", static_matched),
            ("gap_positive_matched", gap_positive_matched),
            ("static_frontier", static_frontier),
            ("gap_frontier", gap_frontier),
        ):
            diagnostic_assignments.append(
                {
                    "record_id": f"cell{cell_idx:03d}_rho{int(round(100*rho)):02d}_{role}",
                    "role": role,
                    "rho": rho,
                    "cell": info,
                    "policy": policy,
                }
            )
    cells.append(cell_result)

frontier_by_rho = {}
for rho in RHOS:
    selected = [row for row in frontier_rows if row["rho"] == rho]
    by_label = {}
    for label in ("supported", "sparse", "unobserved"):
        subset = [row for row in selected if row["empirical_support"]["support_label"] == label]
        by_label[label] = {
            "n": len(subset),
            "strict": sum(row["strict_gap_frontier_improvement"] for row in subset),
        }
    by_speed = {}
    for speed in ("slow", "medium", "fast"):
        subset = [row for row in selected if row["cell"]["speed"] == speed]
        weight_total = sum(row["empirical_support"]["observed_pool_weeks"] for row in subset)
        by_speed[speed] = {
            "observed_weight_total": weight_total,
            "weighted_strict_share": (
                sum(
                    row["empirical_support"]["observed_pool_weeks"]
                    * row["strict_gap_frontier_improvement"]
                    for row in subset
                )
                / weight_total
                if weight_total
                else None
            ),
            "weighted_mean_delta_U": (
                sum(
                    row["empirical_support"]["observed_pool_weeks"]
                    * row["training_delta_U_gap_minus_static"]
                    for row in subset
                )
                / weight_total
                if weight_total
                else None
            ),
        }
    frontier_by_rho[str(rho)] = {
        "full_grid_strict": sum(row["strict_gap_frontier_improvement"] for row in selected),
        "full_grid_n": len(selected),
        "by_support_label": by_label,
        "empirical_weighting_by_speed": by_speed,
    }

unique_diagnostics = {}
for assignment in diagnostic_assignments:
    p = assignment["policy"]
    key = (
        assignment["cell"]["cell_idx"],
        p["family"],
        p["dial_mult"],
        p["alpha"],
    )
    entry = unique_diagnostics.setdefault(
        key,
        {"cell": assignment["cell"], "policy": p, "assignments": []},
    )
    entry["assignments"].append(
        {k: assignment[k] for k in ("record_id", "role", "rho") if k in assignment}
    )

selection = {
    "step": "validation-grid amended 2D selection, training only",
    "seed_block": {"start_inclusive": 20000, "end_exclusive": 20100, "n": 100},
    "grid": {
        "dial_mults": dials,
        "alphas": alphas,
        "fee_cap": 0.30,
        "gap_contains_static_boundary": True,
    },
    "row_counts": row_counts,
    "rho_targets": RHOS,
    "matched_candidates": matched_candidates,
    "frontier_summary": frontier_by_rho,
    "cells": cells,
    "input_sha256": {
        str(path.relative_to(ROOT)): sha256(path)
        for path in (*ROW_PATHS, SUPPORT_PATH, AUDIT_PATH, Path(__file__).resolve())
    },
}
selection_hash = freeze(OUT, selection)

with FRONTIER_CSV.open("w", newline="") as f:
    fields = [
        "cell_idx",
        "stratum",
        "sigma",
        "z",
        "speed",
        "support_label",
        "observed_pool_weeks",
        "rho",
        "target_s",
        "static_dial",
        "static_u",
        "static_s",
        "gap_dial",
        "gap_alpha",
        "gap_u",
        "gap_s",
        "delta_u",
        "strict",
    ]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in frontier_rows:
        s, g, info, support_cell = (
            row["static_frontier"],
            row["gap_frontier"],
            row["cell"],
            row["empirical_support"],
        )
        writer.writerow(
            {
                "cell_idx": info["cell_idx"],
                "stratum": info["stratum"],
                "sigma": info["sigma"],
                "z": info["z"],
                "speed": info["speed"],
                "support_label": support_cell["support_label"],
                "observed_pool_weeks": support_cell["observed_pool_weeks"],
                "rho": row["rho"],
                "target_s": row["target_s_training"],
                "static_dial": s["dial_mult"],
                "static_u": s["mean_u"],
                "static_s": s["mean_s"],
                "gap_dial": g["dial_mult"],
                "gap_alpha": g["alpha"],
                "gap_u": g["mean_u"],
                "gap_s": g["mean_s"],
                "delta_u": row["training_delta_U_gap_minus_static"],
                "strict": row["strict_gap_frontier_improvement"],
            }
        )

diagnostic_plan = {
    "step": "validation-grid amended selected-policy diagnostics, training only",
    "selection_sha256": selection_hash,
    "seed_block": selection["seed_block"],
    "n_unique_policies": len(unique_diagnostics),
    "policies": list(unique_diagnostics.values()),
    "required_outputs": [
        "decision fee distribution and clip fractions",
        "fee by stale and contemporaneous gap bins",
        "event adverse severity and markout by fee decile",
        "fee-risk correlations",
        "allocation/incidence/size/severity decomposition",
    ],
}
freeze(DIAGNOSTIC_PLAN, diagnostic_plan)
print(f"rows={sum(row_counts.values())} keys={len(counts)}")
print(f"matched_candidates={len(matched_candidates)}")
for rho, summary in frontier_by_rho.items():
    print(f"rho={rho} strict={summary['full_grid_strict']}/{summary['full_grid_n']}")
print(f"diagnostic_unique_policies={len(unique_diagnostics)}")
print(f"wrote {OUT}, {FRONTIER_CSV}, {DIAGNOSTIC_PLAN}")
