#!/usr/bin/env python3
"""Analyze fresh held-out validation for the amended two-dimensional family."""

import csv
import gzip
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from scipy.stats import t as student_t


ROOT = Path("/Users/joseph/amm-lab")
LVR = ROOT / ".local/lvr"
PLAN_PATH = LVR / "m3_amended_validation_plan.json"
PLAN_HASH_PATH = LVR / "m3_amended_validation_plan.sha256"
ROW_PATHS = [LVR / f"m3_amended_validation_rows_shard{i}.csv.gz" for i in range(6)]
RESULT_PATH = LVR / "m3_amended_validation_selection.json"
MATCHED_CSV_PATH = LVR / "m3_amended_validation_matched.csv"
FRONTIER_CSV_PATH = LVR / "m3_amended_validation_frontier.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mean(xs: list[float]) -> float:
    return statistics.fmean(xs)


def paired_stats(xs: list[float], confidence: float = 0.95) -> dict:
    n = len(xs)
    m = mean(xs)
    sd = statistics.stdev(xs)
    se = sd / math.sqrt(n)
    if se == 0:
        t_stat = math.copysign(math.inf, m) if m else 0.0
        half = 0.0
    else:
        t_stat = m / se
        half = float(student_t.ppf(confidence, n - 1)) * se
    return {
        "n": n,
        "mean": m,
        "sd": sd,
        "se": se,
        "t": t_stat,
        "one_sided_95_lower": m - half,
        "one_sided_95_upper": m + half,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    assert rows
    with path.open("w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)


plan_hash = sha256(PLAN_PATH)
assert PLAN_HASH_PATH.read_text().split()[0] == plan_hash
plan = json.loads(PLAN_PATH.read_text())
seeds = range(
    plan["validation_seed_block"]["start_inclusive"],
    plan["validation_seed_block"]["end_exclusive"],
)
assert len(seeds) == plan["validation_seed_block"]["n"] == 200
assert seeds.start == 40000 and seeds.stop == 40200

expected = {}
for candidate in plan["matched_candidates"]:
    for role in ("policy_1_lower_A", "policy_2"):
        expected[(candidate["candidate_id"], role)] = {
            "kind": "matched",
            "cell_idx": candidate["cell"]["cell_idx"],
            "rho": candidate["rho"],
            "policy": candidate[role],
        }
for frontier in plan["frontier_pairs"]:
    for role, key in (("static", "static_policy"), ("gap_family", "gap_policy")):
        expected[(frontier["frontier_id"], role)] = {
            "kind": "frontier",
            "cell_idx": frontier["cell"]["cell_idx"],
            "rho": frontier["rho"],
            "policy": frontier[key],
        }
assert len(expected) == 752

numeric = (
    "l", "a", "b", "a_arb", "a_fund", "b_fund", "u", "fees", "fees_arb",
    "fees_fund", "s", "potential", "alloc_amm", "alloc_cex", "alloc_unserved",
    "fill_incidence", "conditional_fill_size", "quote_error", "a_arb_per_served",
    "a_fund_per_served", "a_total_per_served", "n_fund_events",
)
rows = {}
row_counts = Counter()
for path in ROW_PATHS:
    with gzip.open(path, "rt", newline="") as f:
        for raw in csv.DictReader(f):
            assignment = (raw["record_id"], raw["policy_role"])
            assert assignment in expected, assignment
            spec = expected[assignment]
            policy = spec["policy"]
            assert raw["record_kind"] == spec["kind"]
            assert int(raw["cell_idx"]) == spec["cell_idx"]
            assert float(raw["rho"]) == spec["rho"]
            assert raw["family"] == policy["family"]
            for name in ("dial_mult", "f0", "alpha", "fee_cap"):
                assert float(raw[name]) == policy[name]
            seed = int(raw["seed"])
            assert seed in seeds
            key = (*assignment, seed)
            assert key not in rows, key
            parsed = {name: float(raw[name]) if raw[name] else None for name in numeric}
            assert all(math.isfinite(v) for v in parsed.values() if v is not None)
            assert all(parsed[name] is not None for name in numeric)
            assert abs(parsed["l"] - (parsed["a"] - parsed["b"])) <= 1e-7 * max(
                1.0, abs(parsed["l"]), abs(parsed["a"]), abs(parsed["b"])
            )
            assert abs(parsed["u"] - (parsed["fees"] - parsed["l"])) <= 1e-7 * max(
                1.0, abs(parsed["u"]), abs(parsed["fees"]), abs(parsed["l"])
            )
            rows[key] = parsed
            row_counts[path.name] += 1

assert len(rows) == len(expected) * len(seeds) == 150_400
assert all((*assignment, seed) in rows for assignment in expected for seed in seeds)

matched_results = []
matched_csv = []
for candidate in plan["matched_candidates"]:
    cid = candidate["candidate_id"]
    p1 = [rows[(cid, "policy_1_lower_A", seed)] for seed in seeds]
    p2 = [rows[(cid, "policy_2", seed)] for seed in seeds]
    stats_s = paired_stats([a["s"] - b["s"] for a, b in zip(p1, p2)])
    stats_a = paired_stats([a["a"] - b["a"] for a, b in zip(p1, p2)])
    stats_u = paired_stats([a["u"] - b["u"] for a, b in zip(p1, p2)])
    mean_s1, mean_s2 = mean([r["s"] for r in p1]), mean([r["s"] for r in p2])
    s0 = candidate["training_S0"]
    target = candidate["target_s_training"]
    mismatch = abs(mean_s1 - mean_s2) / s0
    gap1 = abs(mean_s1 - target) / s0
    gap2 = abs(mean_s2 - target) / s0
    support_ok = mismatch <= 0.05 and gap1 <= 0.10 and gap2 <= 0.10
    reversal_ok = stats_a["mean"] < 0 and stats_u["mean"] < 0
    inference_ok = (
        support_ok
        and reversal_ok
        and stats_a["one_sided_95_upper"] < 0
        and stats_u["one_sided_95_upper"] < 0
    )
    score = min(-stats_a["t"], -stats_u["t"]) if inference_ok else None
    result = {
        "candidate_id": cid,
        "cell": candidate["cell"],
        "empirical_support": candidate["empirical_support"],
        "rho": candidate["rho"],
        "training_S0": s0,
        "target_s_training": target,
        "validation_mean_s_policy_1": mean_s1,
        "validation_mean_s_policy_2": mean_s2,
        "service_mismatch_over_training_S0": mismatch,
        "policy_1_target_gap_over_training_S0": gap1,
        "policy_2_target_gap_over_training_S0": gap2,
        "delta_s_policy_1_minus_policy_2": stats_s,
        "delta_a_policy_1_minus_policy_2": stats_a,
        "delta_u_policy_1_minus_policy_2": stats_u,
        "support_ok": support_ok,
        "reversal_ok": reversal_ok,
        "both_one_sided_95_upper_below_zero": inference_ok,
        "studentized_margin_score": score,
    }
    matched_results.append(result)
    matched_csv.append({
        "candidate_id": cid,
        "cell_idx": candidate["cell"]["cell_idx"],
        "support_label": candidate["empirical_support"]["support_label"],
        "rho": candidate["rho"],
        "mean_s_1": mean_s1,
        "mean_s_2": mean_s2,
        "service_mismatch_over_training_S0": mismatch,
        "target_gap_1_over_training_S0": gap1,
        "target_gap_2_over_training_S0": gap2,
        "mean_delta_a": stats_a["mean"],
        "upper95_delta_a": stats_a["one_sided_95_upper"],
        "t_delta_a": stats_a["t"],
        "mean_delta_u": stats_u["mean"],
        "upper95_delta_u": stats_u["one_sided_95_upper"],
        "t_delta_u": stats_u["t"],
        "inference_ok": inference_ok,
        "score": score,
    })

eligible = [x for x in matched_results if x["both_one_sided_95_upper_below_zero"]]
eligible.sort(key=lambda x: (
    -x["studentized_margin_score"],
    x["service_mismatch_over_training_S0"],
    x["candidate_id"],
))
winner = eligible[0] if eligible else None

frontier_results = []
frontier_csv = []
for frontier in plan["frontier_pairs"]:
    fid = frontier["frontier_id"]
    static = [rows[(fid, "static", seed)] for seed in seeds]
    gap = [rows[(fid, "gap_family", seed)] for seed in seeds]
    delta_s = paired_stats([g["s"] - s["s"] for g, s in zip(gap, static)])
    delta_u = paired_stats([g["u"] - s["u"] for g, s in zip(gap, static)])
    mean_s_static = mean([r["s"] for r in static])
    mean_s_gap = mean([r["s"] for r in gap])
    target = frontier["target_s_training"]
    service_feasible = mean_s_static >= target and mean_s_gap >= target
    evidence_gate = service_feasible and delta_u["one_sided_95_lower"] > 0
    result = {
        "frontier_id": fid,
        "cell": frontier["cell"],
        "empirical_support": frontier["empirical_support"],
        "rho": frontier["rho"],
        "training_S0": frontier["training_S0"],
        "target_s_training": target,
        "static_policy": frontier["static_policy"],
        "gap_policy": frontier["gap_policy"],
        "strict_gap_improvement_on_training": frontier["strict_gap_improvement_on_training"],
        "validation_mean_s_static": mean_s_static,
        "validation_mean_s_gap": mean_s_gap,
        "validation_service_feasible": service_feasible,
        "delta_s_gap_minus_static": delta_s,
        "delta_u_gap_minus_static": delta_u,
        "validation_evidence_gate": evidence_gate,
    }
    frontier_results.append(result)
    frontier_csv.append({
        "frontier_id": fid,
        "cell_idx": frontier["cell"]["cell_idx"],
        "support_label": frontier["empirical_support"]["support_label"],
        "rho": frontier["rho"],
        "gap_alpha": frontier["gap_policy"]["alpha"],
        "static_dial": frontier["static_policy"]["dial_mult"],
        "gap_dial": frontier["gap_policy"]["dial_mult"],
        "training_strict": frontier["strict_gap_improvement_on_training"],
        "mean_s_static": mean_s_static,
        "mean_s_gap": mean_s_gap,
        "service_feasible": service_feasible,
        "mean_delta_s": delta_s["mean"],
        "mean_delta_u": delta_u["mean"],
        "lower95_delta_u": delta_u["one_sided_95_lower"],
        "upper95_delta_u": delta_u["one_sided_95_upper"],
        "evidence_gate": evidence_gate,
    })

result = {
    "step": "M3 amended fresh-validation selection",
    "validation_plan_sha256": plan_hash,
    "validation_rows_sha256": {path.name: sha256(path) for path in ROW_PATHS},
    "row_counts": dict(sorted(row_counts.items())),
    "integrity": {
        "expected_assignments_per_seed": len(expected),
        "expected_seeds": len(seeds),
        "observed_rows": len(rows),
        "duplicate_rows": 0,
        "metric_identities_checked": ["L=A-B", "U=fees-L"],
    },
    "matched_summary": {
        "n_candidates": len(matched_results),
        "n_support_ok": sum(x["support_ok"] for x in matched_results),
        "n_reversal_direction": sum(x["support_ok"] and x["reversal_ok"] for x in matched_results),
        "n_inference_eligible": len(eligible),
        "winner_candidate_id": winner["candidate_id"] if winner else None,
        "winner_rule": plan["matched_service_rule"]["winner_rule"],
    },
    "winner": winner,
    "matched_candidates": matched_results,
    "frontier_summary": {
        "n_frozen_pairs": len(frontier_results),
        "n_training_strict": sum(x["strict_gap_improvement_on_training"] for x in frontier_results),
        "n_validation_service_feasible": sum(x["validation_service_feasible"] for x in frontier_results),
        "n_validation_evidence_gate": sum(x["validation_evidence_gate"] for x in frontier_results),
        "n_supported_state_evidence_gate": sum(
            x["validation_evidence_gate"] and x["empirical_support"]["support_label"] == "supported"
            for x in frontier_results
        ),
        "selection_note": "No validation re-optimization; all policies were frozen from training.",
    },
    "frontier_pairs": frontier_results,
}
RESULT_PATH.write_text(json.dumps(result, indent=1, sort_keys=True, allow_nan=False) + "\n")
write_csv(MATCHED_CSV_PATH, matched_csv)
write_csv(FRONTIER_CSV_PATH, frontier_csv)

print(f"rows={len(rows)} assignments={len(expected)} seeds={len(seeds)}")
print(
    f"matched support={result['matched_summary']['n_support_ok']}/106; "
    f"direction={result['matched_summary']['n_reversal_direction']}/106; "
    f"inference={len(eligible)}/106"
)
print(f"winner={result['matched_summary']['winner_candidate_id']}")
print(
    "frontier evidence gate="
    f"{result['frontier_summary']['n_validation_evidence_gate']}/270; "
    f"supported={result['frontier_summary']['n_supported_state_evidence_gate']}"
)
print(f"wrote {RESULT_PATH}, {MATCHED_CSV_PATH}, {FRONTIER_CSV_PATH}")
