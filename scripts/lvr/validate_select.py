#!/usr/bin/env python3
"""validation-grid Step D validation audit and preregistered matched-candidate selection."""

import csv
import gzip
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import t as student_t


ROOT = Path("/Users/joseph/amm-lab")
LVR = ROOT / ".local/lvr/workspace"
PLAN_PATH = LVR / "m3_validation_plan.json"
PLAN_HASH_PATH = LVR / "m3_validation_plan.sha256"
ROW_PATHS = [LVR / f"m3_validation_rows_shard{i}.csv.gz" for i in range(6)]
RESULT_PATH = LVR / "m3_validation_selection.json"
MATCHED_CSV_PATH = LVR / "m3_validation_matched.csv"
FRONTIER_CSV_PATH = LVR / "m3_validation_frontier.csv"


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
    assert n > 1
    m = mean(xs)
    sd = statistics.stdev(xs)
    se = sd / math.sqrt(n)
    if se == 0:
        t_stat = math.copysign(math.inf, m) if m != 0 else 0.0
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

expected = {}
for candidate in plan["matched_candidates"]:
    for role in ("policy_1_lower_A", "policy_2"):
        policy = candidate[role]
        expected[(candidate["candidate_id"], role)] = {
            "kind": "matched",
            "cell_idx": candidate["cell"]["cell_idx"],
            "rho": candidate["rho"],
            "family": policy["family"],
            "dial_mult": policy["dial_mult"],
            "dial_fee": policy["dial_fee"],
        }
for frontier in plan["frontier_pairs"]:
    for role, key in (("static", "static_policy"), ("adaptive", "adaptive_policy")):
        policy = frontier[key]
        expected[(frontier["frontier_id"], role)] = {
            "kind": "frontier",
            "cell_idx": frontier["cell"]["cell_idx"],
            "rho": frontier["rho"],
            "family": policy["family"],
            "dial_mult": policy["dial_mult"],
            "dial_fee": policy["dial_fee"],
        }
assert len(expected) == 608

required_numeric = (
    "l",
    "a",
    "b",
    "l_arb",
    "u",
    "fees",
    "fees_arb",
    "fees_fund",
    "s",
    "potential",
)
optional_numeric = (
    "alloc_amm",
    "alloc_cex",
    "alloc_unserved",
    "quote_error",
    "a_arb_per_served",
    "a_fund_per_served",
    "a_total_per_served",
)
rows = {}
row_counts = Counter()
for path in ROW_PATHS:
    with gzip.open(path, "rt", newline="") as f:
        for raw in csv.DictReader(f):
            assignment = (raw["record_id"], raw["policy_role"])
            assert assignment in expected, assignment
            spec = expected[assignment]
            assert raw["record_kind"] == spec["kind"]
            assert int(raw["cell_idx"]) == spec["cell_idx"]
            assert float(raw["rho"]) == spec["rho"]
            assert raw["family"] == spec["family"]
            assert float(raw["dial_mult"]) == spec["dial_mult"]
            assert float(raw["dial_fee"]) == spec["dial_fee"]
            seed = int(raw["seed"])
            assert seed in seeds
            key = (*assignment, seed)
            assert key not in rows, key
            parsed = {name: float(raw[name]) for name in required_numeric}
            parsed.update(
                {
                    name: float(raw[name]) if raw[name] else None
                    for name in optional_numeric
                }
            )
            assert all(math.isfinite(value) for value in parsed.values() if value is not None)
            assert abs(parsed["l"] - (parsed["a"] - parsed["b"])) <= 1e-7 * max(
                1.0, abs(parsed["l"]), abs(parsed["a"]), abs(parsed["b"])
            )
            assert abs(parsed["u"] - (parsed["fees"] - parsed["l"])) <= 1e-7 * max(
                1.0, abs(parsed["u"]), abs(parsed["fees"]), abs(parsed["l"])
            )
            rows[key] = parsed
            row_counts[path.name] += 1

assert len(rows) == len(expected) * len(seeds) == 121_600
for assignment in expected:
    assert all((*assignment, seed) in rows for seed in seeds)

matched_results = []
matched_csv = []
for candidate in plan["matched_candidates"]:
    cid = candidate["candidate_id"]
    p1 = [rows[(cid, "policy_1_lower_A", seed)] for seed in seeds]
    p2 = [rows[(cid, "policy_2", seed)] for seed in seeds]
    delta_s = [a["s"] - b["s"] for a, b in zip(p1, p2)]
    delta_a = [a["a"] - b["a"] for a, b in zip(p1, p2)]
    delta_u = [a["u"] - b["u"] for a, b in zip(p1, p2)]
    stats_s = paired_stats(delta_s)
    stats_a = paired_stats(delta_a)
    stats_u = paired_stats(delta_u)
    mean_s1, mean_s2 = mean([r["s"] for r in p1]), mean([r["s"] for r in p2])
    s0 = candidate["training_S0"]
    target = candidate["target_s_training"]
    mismatch = abs(mean_s1 - mean_s2) / s0
    gap1 = abs(mean_s1 - target) / s0
    gap2 = abs(mean_s2 - target) / s0
    support_ok = mismatch <= 0.05 and gap1 <= 0.10 and gap2 <= 0.10
    reversal_ok = stats_a["mean"] < 0 and stats_u["mean"] < 0
    survives = support_ok and reversal_ok
    inference_ok = (
        survives
        and stats_a["one_sided_95_upper"] < 0
        and stats_u["one_sided_95_upper"] < 0
    )
    score = min(-stats_a["t"], -stats_u["t"]) if inference_ok else None
    result = {
        "candidate_id": cid,
        "cell": candidate["cell"],
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
        "survives_validation": survives,
        "both_one_sided_95_upper_below_zero": inference_ok,
        "studentized_margin_score": score,
    }
    matched_results.append(result)
    matched_csv.append(
        {
            "candidate_id": cid,
            "cell_idx": candidate["cell"]["cell_idx"],
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
            "survives": survives,
            "inference_ok": inference_ok,
            "score": score,
        }
    )

eligible = [x for x in matched_results if x["both_one_sided_95_upper_below_zero"]]
eligible.sort(
    key=lambda x: (
        -x["studentized_margin_score"],
        x["service_mismatch_over_training_S0"],
        x["cell"]["cell_idx"],
        x["rho"],
    )
)
winner = eligible[0] if eligible else None

frontier_results = []
frontier_csv = []
for frontier in plan["frontier_pairs"]:
    fid = frontier["frontier_id"]
    static = [rows[(fid, "static", seed)] for seed in seeds]
    adaptive = [rows[(fid, "adaptive", seed)] for seed in seeds]
    delta_u = paired_stats([a["u"] - s["u"] for a, s in zip(adaptive, static)])
    delta_s = paired_stats([a["s"] - s["s"] for a, s in zip(adaptive, static)])
    result = {
        "frontier_id": fid,
        "cell": frontier["cell"],
        "rho": frontier["rho"],
        "static_policy": frontier["static_policy"],
        "adaptive_policy": frontier["adaptive_policy"],
        "strict_adaptive_improvement_on_training": frontier[
            "strict_adaptive_improvement_on_training"
        ],
        "validation_mean_s_static": mean([r["s"] for r in static]),
        "validation_mean_s_adaptive": mean([r["s"] for r in adaptive]),
        "delta_s_adaptive_minus_static": delta_s,
        "delta_u_adaptive_minus_static": delta_u,
    }
    frontier_results.append(result)
    frontier_csv.append(
        {
            "frontier_id": fid,
            "cell_idx": frontier["cell"]["cell_idx"],
            "rho": frontier["rho"],
            "adaptive_family": frontier["adaptive_policy"]["family"],
            "static_dial": frontier["static_policy"]["dial_mult"],
            "adaptive_dial": frontier["adaptive_policy"]["dial_mult"],
            "training_strict": frontier["strict_adaptive_improvement_on_training"],
            "mean_s_static": result["validation_mean_s_static"],
            "mean_s_adaptive": result["validation_mean_s_adaptive"],
            "mean_delta_s": delta_s["mean"],
            "mean_delta_u": delta_u["mean"],
            "lower95_delta_u": delta_u["one_sided_95_lower"],
            "upper95_delta_u": delta_u["one_sided_95_upper"],
            "t_delta_u": delta_u["t"],
        }
    )

result = {
    "step": "validation validation selection",
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
        "n_support_and_reversal_survivors": sum(
            x["survives_validation"] for x in matched_results
        ),
        "n_inference_eligible": len(eligible),
        "winner_candidate_id": winner["candidate_id"] if winner else None,
        "winner_rule": plan["matched_service_rule"]["winner_rule"],
    },
    "winner": winner,
    "matched_candidates": matched_results,
    "frontier_summary": {
        "n_frozen_pairs": len(frontier_results),
        "n_training_strict": sum(
            x["strict_adaptive_improvement_on_training"] for x in frontier_results
        ),
        "n_training_strict_with_validation_lower95_above_zero": sum(
            x["strict_adaptive_improvement_on_training"]
            and x["delta_u_adaptive_minus_static"]["one_sided_95_lower"] > 0
            for x in frontier_results
        ),
        "selection_note": "No validation re-optimization; no frontier alternative is automatically selected without an explicit interval gate.",
    },
    "frontier_pairs": frontier_results,
}
RESULT_PATH.write_text(json.dumps(result, indent=1, sort_keys=True, allow_nan=False) + "\n")
write_csv(MATCHED_CSV_PATH, matched_csv)
write_csv(FRONTIER_CSV_PATH, frontier_csv)

print(f"rows={len(rows)} assignments={len(expected)} seeds={len(seeds)}")
print(
    "matched survivors="
    f"{result['matched_summary']['n_support_and_reversal_survivors']}/34; "
    f"inference eligible={len(eligible)}/34"
)
print(f"winner={result['matched_summary']['winner_candidate_id']}")
print(
    "training-strict frontier points with validation lower95>0="
    f"{result['frontier_summary']['n_training_strict_with_validation_lower95_above_zero']}"
    f"/{result['frontier_summary']['n_training_strict']}"
)
print(f"wrote {RESULT_PATH}, {MATCHED_CSV_PATH}, {FRONTIER_CSV_PATH}")
