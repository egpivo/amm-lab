#!/usr/bin/env python3
"""Audit and analyze the single validation-grid untouched-final candidate."""

import csv
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

from scipy.stats import t as student_t


ROOT = Path("/Users/joseph/amm-lab")
LVR = ROOT / ".local/lvr/workspace"
PLAN_PATH = LVR / "m3_final_plan.json"
PLAN_HASH_PATH = LVR / "m3_final_plan.sha256"
ROWS_PATH = LVR / "m3_final_rows.csv.gz"
RESULT_PATH = LVR / "m3_final_result.json"
BATCH_SIZE = 25
ABSOLUTE_PRECISION_TARGET = 800.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def paired_stats(xs: list[float]) -> dict:
    n = len(xs)
    m = statistics.fmean(xs)
    sd = statistics.stdev(xs)
    se = sd / math.sqrt(n)
    one_sided_half = float(student_t.ppf(0.95, n - 1)) * se
    two_sided_half = float(student_t.ppf(0.975, n - 1)) * se
    return {
        "n": n,
        "mean": m,
        "sd": sd,
        "se": se,
        "t": m / se if se else math.copysign(math.inf, m),
        "one_sided_95_lower": m - one_sided_half,
        "one_sided_95_upper": m + one_sided_half,
        "two_sided_95_halfwidth": two_sided_half,
    }


def tail_contribution(xs: list[float]) -> dict:
    values = sorted((abs(x) for x in xs), reverse=True)
    total = sum(values)
    if total == 0:
        return {"max_seed_share": 0.0, "top_1pct_share": 0.0}
    top_n = max(1, math.ceil(0.01 * len(values)))
    return {
        "max_seed_share": values[0] / total,
        "top_1pct_share": sum(values[:top_n]) / total,
    }


def median_of_means(xs: list[float]) -> dict:
    assert len(xs) % BATCH_SIZE == 0
    batch_means = [
        statistics.fmean(xs[i : i + BATCH_SIZE])
        for i in range(0, len(xs), BATCH_SIZE)
    ]
    ordered = sorted(batch_means)
    return {
        "batch_size": BATCH_SIZE,
        "n_batches": len(batch_means),
        "upper_median_of_batch_means": ordered[len(ordered) // 2],
        "batch_means": batch_means,
    }


plan_hash = sha256(PLAN_PATH)
assert PLAN_HASH_PATH.read_text().split()[0] == plan_hash
plan = json.loads(PLAN_PATH.read_text())
candidate = plan["candidate"]
seed_block = plan["final_seed_block"]
seeds = range(seed_block["start_inclusive"], seed_block["end_exclusive"])
assert len(seeds) == seed_block["n"] == 400

specs = {
    "policy_1_lower_A": candidate["policy_1_lower_A"],
    "policy_2": candidate["policy_2"],
}
numeric = (
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
    "alloc_amm",
    "alloc_cex",
    "alloc_unserved",
    "quote_error",
    "a_arb_per_served",
    "a_fund_per_served",
    "a_total_per_served",
)
rows = {}
with gzip.open(ROWS_PATH, "rt", newline="") as f:
    for raw in csv.DictReader(f):
        assert raw["candidate_id"] == candidate["candidate_id"]
        role = raw["policy_role"]
        assert role in specs
        spec = specs[role]
        assert int(raw["cell_idx"]) == candidate["cell"]["cell_idx"]
        assert float(raw["rho"]) == candidate["rho"]
        assert raw["family"] == spec["family"]
        assert float(raw["dial_mult"]) == spec["dial_mult"]
        assert float(raw["dial_fee"]) == spec["dial_fee"]
        seed = int(raw["seed"])
        assert seed in seeds
        key = (role, seed)
        assert key not in rows
        values = {name: float(raw[name]) if raw[name] else None for name in numeric}
        assert all(math.isfinite(x) for x in values.values() if x is not None)
        assert all(values[name] is not None for name in numeric)
        assert abs(values["l"] - (values["a"] - values["b"])) <= 1e-7 * max(
            1.0, abs(values["l"]), abs(values["a"]), abs(values["b"])
        )
        assert abs(values["u"] - (values["fees"] - values["l"])) <= 1e-7 * max(
            1.0, abs(values["u"]), abs(values["fees"]), abs(values["l"])
        )
        rows[key] = values

assert len(rows) == 800
assert all((role, seed) in rows for role in specs for seed in seeds)
p1 = [rows[("policy_1_lower_A", seed)] for seed in seeds]
p2 = [rows[("policy_2", seed)] for seed in seeds]
deltas = {
    metric: [a[metric] - b[metric] for a, b in zip(p1, p2)]
    for metric in ("s", "a", "u")
}
delta_stats = {metric: paired_stats(values) for metric, values in deltas.items()}

mean_s1 = statistics.fmean(x["s"] for x in p1)
mean_s2 = statistics.fmean(x["s"] for x in p2)
s0 = candidate["training_S0"]
target = candidate["target_s_training"]
service = {
    "mean_s_policy_1": mean_s1,
    "mean_s_policy_2": mean_s2,
    "pair_mismatch_over_training_S0": abs(mean_s1 - mean_s2) / s0,
    "policy_1_target_gap_over_training_S0": abs(mean_s1 - target) / s0,
    "policy_2_target_gap_over_training_S0": abs(mean_s2 - target) / s0,
}
support_ok = (
    service["pair_mismatch_over_training_S0"] <= 0.05
    and service["policy_1_target_gap_over_training_S0"] <= 0.10
    and service["policy_2_target_gap_over_training_S0"] <= 0.10
)
direction_ok = delta_stats["a"]["mean"] < 0 and delta_stats["u"]["mean"] < 0
inference_ok = (
    delta_stats["a"]["one_sided_95_upper"] < 0
    and delta_stats["u"]["one_sided_95_upper"] < 0
)

result = {
    "step": "final-block untouched-final verification",
    "final_plan_sha256": plan_hash,
    "final_rows_sha256": sha256(ROWS_PATH),
    "candidate_id": candidate["candidate_id"],
    "candidate": candidate,
    "integrity": {
        "expected_rows": 800,
        "observed_rows": len(rows),
        "duplicate_rows": 0,
        "seed_start_inclusive": seed_block["start_inclusive"],
        "seed_end_exclusive": seed_block["end_exclusive"],
        "metric_identities_checked": ["L=A-B", "U=fees-L"],
    },
    "policy_means": {
        role: {
            metric: statistics.fmean(rows[(role, seed)][metric] for seed in seeds)
            for metric in numeric
        }
        for role in specs
    },
    "service": service,
    "paired_deltas_policy_1_minus_policy_2": delta_stats,
    "heavy_tail_robustness": {
        metric: {
            "median_of_means": median_of_means(values),
            "absolute_contribution": tail_contribution(values),
        }
        for metric, values in deltas.items()
    },
    "precision": {
        "paired_delta_u_two_sided_95_halfwidth": delta_stats["u"][
            "two_sided_95_halfwidth"
        ],
        "absolute_target": ABSOLUTE_PRECISION_TARGET,
        "target_met": delta_stats["u"]["two_sided_95_halfwidth"]
        <= ABSOLUTE_PRECISION_TARGET,
    },
    "gates": {
        "support_ok": support_ok,
        "direction_ok": direction_ok,
        "both_one_sided_95_upper_below_zero": inference_ok,
        "final_verified": support_ok and direction_ok and inference_ok,
    },
}
RESULT_PATH.write_text(json.dumps(result, indent=1, sort_keys=True, allow_nan=False) + "\n")
print(f"rows={len(rows)} seeds={len(seeds)}")
print(f"support={support_ok} direction={direction_ok} inference={inference_ok}")
print(
    f"delta_A={delta_stats['a']['mean']:.6f} "
    f"upper95={delta_stats['a']['one_sided_95_upper']:.6f}"
)
print(
    f"delta_U={delta_stats['u']['mean']:.6f} "
    f"upper95={delta_stats['u']['one_sided_95_upper']:.6f}"
)
print(
    f"service_mismatch/S0={service['pair_mismatch_over_training_S0']:.6f} "
    f"precision_halfwidth={delta_stats['u']['two_sided_95_halfwidth']:.6f}"
)
print(f"final_verified={result['gates']['final_verified']}")
print(f"wrote {RESULT_PATH}")
