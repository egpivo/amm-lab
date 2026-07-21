#!/usr/bin/env python3
"""M3 Step C post-processing: selection on the training block.

Reads m3_training_rows_shard*.csv.gz (per-seed rows from lvr_m3_train),
computes per-(cell, family, dial) training means, then per cell:
  - S0 = static family's mean service at dial_mult 1.0 (stratum fee);
  - per family, the dial member nearest each service target rho*S0,
    rho in {0.2, 0.4, 0.6, 0.8, 0.95} (support gap = |E[S]-target|/S0,
    reported; a target with gap > 0.5*rho_step is flagged unreachable);
  - the argmin-E[A] member per family (Prop B gross-loss selector);
  - the frontier trace: per family the (E[S], E[U]) curve over dials;
    the pooled upper envelope is derived from these points downstream.
Writes m3_training_selection.json (members to carry to validation) and
m3_training_frontier.csv (curve points for the paper figure).
Selection only; no inference is done on the training block.
"""
import glob
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/Users/joseph/amm-lab")
LVR = ROOT / ".local/lvr"
RHOS = [0.2, 0.4, 0.6, 0.8, 0.95]
FAMILIES = ["static", "gap", "defensive"]
MATCH_TOLERANCE_OVER_S0 = 0.05
SUPPORT_TOLERANCE_OVER_S0 = 0.10


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=1, sort_keys=True) + "\n").encode()


def freeze_plan(path: Path, value: object) -> str:
    payload = json_bytes(value)
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(
            f"refusing to overwrite frozen validation plan with different content: {path}"
        )
    if not path.exists():
        path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n")
    return digest


def policy_point(ci: int, family: str, dial: float) -> dict:
    mm = means[(ci, family, dial)]
    return {
        "family": family,
        "dial_mult": dial,
        "dial_fee": cellinfo[ci]["dial_fee_at_1"] * dial,
        "mean_s": mm["s"],
        "mean_u": mm["u"],
        "mean_a": mm["a"],
        "mean_b": mm["b"],
    }


def frontier_pick(points: list[dict]) -> dict:
    if not points:
        raise RuntimeError("frontier feasible set is empty")
    # Maximize training U. Exact ties prefer static, then the lower dial,
    # making the finite-grid union deterministic without assuming set inclusion.
    return min(
        points,
        key=lambda p: (
            -p["mean_u"],
            0 if p["family"] == "static" else 1,
            p["dial_mult"],
        ),
    )

rows = defaultdict(lambda: defaultdict(list))  # (cell,fam,dial) -> metric -> [per-seed]
cellinfo = {}
training_paths = [Path(p) for p in sorted(glob.glob(str(LVR / "m3_training_rows_shard*.csv.gz")))]
for path in training_paths:
    with gzip.open(path, "rt") as f:
        for r in csv.DictReader(f):
            key = (int(r["cell_idx"]), r["family"], float(r["dial_mult"]))
            for m in ("u", "a", "b", "s"):
                rows[key][m].append(float(r[m]))
            cellinfo[int(r["cell_idx"])] = {
                "stratum": r["stratum"], "sigma": float(r["sigma"]),
                "z": float(r["z"]), "speed": r["speed"],
                "dial_fee_at_1": float(r["dial_fee"]) / float(r["dial_mult"]),
            }

means = {k: {m: sum(v) / len(v) for m, v in d.items()} | {"n": len(d["u"])}
         for k, d in rows.items()}

cells = sorted(cellinfo)
dials = sorted({k[2] for k in means})

selection = []
frontier_lines = ["cell_idx,stratum,sigma,z,speed,family,dial_mult,mean_s,mean_u,mean_a,mean_b"]
for ci in cells:
    info = cellinfo[ci]
    s0 = means[(ci, "static", 1.0)]["s"]
    cell_sel = {"cell_idx": ci, **{k: v for k, v in info.items() if k != "dial_fee_at_1"},
                "S0_training": s0, "families": {}}
    for fam in FAMILIES:
        curve = []
        for dm in dials:
            k = (ci, fam, dm)
            if k not in means:
                continue
            mm = means[k]
            curve.append((dm, mm["s"], mm["u"], mm["a"], mm["b"]))
            frontier_lines.append(
                f"{ci},{info['stratum']},{info['sigma']},{info['z']},{info['speed']},"
                f"{fam},{dm},{mm['s']:.4f},{mm['u']:.4f},{mm['a']:.4f},{mm['b']:.4f}")
        # service-matched members
        matched = {}
        for rho in RHOS:
            target = rho * s0
            best = min(curve, key=lambda c: abs(c[1] - target))
            gap = abs(best[1] - target) / s0 if s0 > 0 else float("nan")
            matched[str(rho)] = {
                "dial_mult": best[0], "mean_s": best[1], "mean_u": best[2],
                "mean_a": best[3], "mean_b": best[4],
                "support_gap_over_S0": gap,
                "unreachable": bool(gap > SUPPORT_TOLERANCE_OVER_S0),
            }
        argmin_a = min(curve, key=lambda c: c[3])
        cell_sel["families"][fam] = {
            "matched": matched,
            "argmin_A": {"dial_mult": argmin_a[0], "mean_a": argmin_a[3],
                          "mean_s": argmin_a[1], "mean_u": argmin_a[2]},
            "baseline": {"dial_mult": 1.0},
        }
    selection.append(cell_sel)

selection_path = LVR / "m3_training_selection.json"
frontier_path = LVR / "m3_training_frontier.csv"
selection_path.write_bytes(json_bytes({
    "step": "M3-C selection (training block)",
    "seed_block": {"start_inclusive": 20000, "end_exclusive": 20100, "n": 100},
    "rho_targets": RHOS,
    "unreachable_rule": "support gap > 0.1*S0",
    "cells": selection,
}))
frontier_path.write_text("\n".join(frontier_lines) + "\n")

# Freeze the validation candidate universe before any validation seed is read.
matched_candidates = []
frontier_pairs = []
for c in selection:
    ci = c["cell_idx"]
    s0 = c["S0_training"]
    info = {k: c[k] for k in ("cell_idx", "stratum", "sigma", "z", "speed")}
    for rho in RHOS:
        target = rho * s0
        rho_key = str(rho)

        sm = c["families"]["static"]["matched"][rho_key]
        gm = c["families"]["gap"]["matched"][rho_key]
        sp = policy_point(ci, "static", sm["dial_mult"])
        gp = policy_point(ci, "gap", gm["dial_mult"])
        pair_gap = abs(sp["mean_s"] - gp["mean_s"]) / s0
        orientation = None
        if sp["mean_a"] < gp["mean_a"] and sp["mean_u"] < gp["mean_u"]:
            orientation = (sp, gp)
        elif gp["mean_a"] < sp["mean_a"] and gp["mean_u"] < sp["mean_u"]:
            orientation = (gp, sp)
        if (
            not sm["unreachable"]
            and not gm["unreachable"]
            and pair_gap <= MATCH_TOLERANCE_OVER_S0
            and orientation is not None
        ):
            p1, p2 = orientation
            matched_candidates.append({
                "candidate_id": f"match_c{ci:03d}_rho{int(round(100*rho)):02d}",
                "cell": info,
                "rho": rho,
                "training_S0": s0,
                "target_s_training": target,
                "policy_1_lower_A": p1,
                "policy_2": p2,
                "training_deltas_policy1_minus_policy2": {
                    "s": p1["mean_s"] - p2["mean_s"],
                    "a": p1["mean_a"] - p2["mean_a"],
                    "u": p1["mean_u"] - p2["mean_u"],
                },
                "service_mismatch_over_training_S0": pair_gap,
                "policy_1_target_gap_over_training_S0": abs(p1["mean_s"] - target) / s0,
                "policy_2_target_gap_over_training_S0": abs(p2["mean_s"] - target) / s0,
            })

        static_feasible = [
            policy_point(ci, "static", dm)
            for dm in dials
            if means[(ci, "static", dm)]["s"] >= target
        ]
        gap_feasible = [
            policy_point(ci, "gap", dm)
            for dm in dials
            if means[(ci, "gap", dm)]["s"] >= target
        ]
        static_best = frontier_pick(static_feasible)
        adaptive_best = frontier_pick(static_feasible + gap_feasible)
        frontier_pairs.append({
            "frontier_id": f"frontier_c{ci:03d}_rho{int(round(100*rho)):02d}",
            "cell": info,
            "rho": rho,
            "training_S0": s0,
            "target_s_training": target,
            "static_policy": static_best,
            "adaptive_policy": adaptive_best,
            "gap_has_training_support": bool(gap_feasible),
            "strict_adaptive_improvement_on_training": (
                adaptive_best["family"] == "gap"
                and adaptive_best["mean_u"] > static_best["mean_u"]
            ),
            "training_delta_u_adaptive_minus_static": (
                adaptive_best["mean_u"] - static_best["mean_u"]
            ),
        })

input_paths = [
    LVR / "calibration_54_manifest.json",
    *training_paths,
    selection_path,
    frontier_path,
    Path(__file__).resolve(),
]
plan = {
    "step": "M3-D immutable validation plan",
    "source_stage": "training only",
    "validation_seed_block": {"start_inclusive": 30000, "end_exclusive": 30200, "n": 200},
    "final_seed_block": {"start_inclusive": 90000, "end_exclusive": 90400, "n": 400},
    "rho_targets": RHOS,
    "matched_service_rule": {
        "pair_tolerance_over_training_S0": MATCH_TOLERANCE_OVER_S0,
        "per_policy_support_tolerance_over_training_S0": SUPPORT_TOLERANCE_OVER_S0,
        "orientation": "policy_1 is the lower-A policy on training and also has lower U on training",
        "validation_survival": "pair mismatch <= 0.05 training S0; both target gaps <= 0.10 training S0; mean delta A < 0; mean delta U < 0",
        "winner_rule": "both one-sided paired 95% CI upper bounds < 0; maximize min(-t_delta_A,-t_delta_U); tie-break by lower service mismatch, then candidate_id",
    },
    "frontier_rule": {
        "static": "maximize training mean U over finite static-grid members with training mean S >= rho*S0",
        "adaptive": "maximize training mean U over the finite union static U gap with training mean S >= rho*S0",
        "tie_break": "exact U ties prefer static, then lower dial_mult",
        "validation": "evaluate only the frozen policies; do not re-optimize on validation",
        "defensive_scope": "boundary evidence only; no member of the preregistered defensive grid supported the positive-service targets",
    },
    "input_artifacts_sha256": {
        str(p.relative_to(ROOT)): sha256(p) for p in input_paths
    },
    "matched_candidates": matched_candidates,
    "frontier_pairs": frontier_pairs,
}
plan_path = LVR / "m3_validation_plan.json"
plan_hash = freeze_plan(plan_path, plan)

# console summary: how many matched targets are unreachable per family
unreach = defaultdict(int)
total = defaultdict(int)
for c in selection:
    for fam, d in c["families"].items():
        for rho, m in d["matched"].items():
            total[fam] += 1
            unreach[fam] += m["unreachable"]
print(f"cells={len(cells)} dials={len(dials)} rows/key n={means[(cells[0],'static',1.0)]['n']}")
for fam in FAMILIES:
    print(f"{fam}: unreachable matched targets {unreach[fam]}/{total[fam]}")
strict_frontier = sum(p["strict_adaptive_improvement_on_training"] for p in frontier_pairs)
print(f"matched validation candidates: {len(matched_candidates)}")
print(f"strict adaptive training-frontier points: {strict_frontier}/{len(frontier_pairs)}")
print(f"validation plan sha256: {plan_hash}")
print("wrote m3_training_selection.json, m3_training_frontier.csv, m3_validation_plan.json")
