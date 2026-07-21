#!/usr/bin/env python3
"""Analyze amended validation-grid fee/risk sorting and exact channel decompositions."""

import gzip
import hashlib
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path("/Users/joseph/amm-lab")
LVR = ROOT / ".local/lvr/workspace"
PLAN_PATH = LVR / "m3_amended_diagnostic_plan.json"
SHARDS = [LVR / f"m3_amended_diagnostics_shard{i}.json.gz" for i in range(6)]
OUT = LVR / "m3_amended_diagnostics.json"
N_FEE_BINS = 1024
MIN_FEE_BPS = 0.01
MAX_FEE_BPS = 3000.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def policy_key(cell: dict, policy: dict) -> tuple:
    return (
        cell["cell_idx"],
        policy["family"],
        policy["dial_mult"],
        policy["alpha"],
    )


def fee_bin_bps(index: int) -> float:
    fraction = (index + 0.5) / N_FEE_BINS
    return MIN_FEE_BPS * (MAX_FEE_BPS / MIN_FEE_BPS) ** fraction


def histogram_quantile(hist: list[int], quantile: float) -> float:
    target = quantile * sum(hist)
    cumulative = 0
    for index, count in enumerate(hist):
        cumulative += count
        if cumulative >= target:
            return fee_bin_bps(index) / 10_000.0
    return fee_bin_bps(len(hist) - 1) / 10_000.0


def correlation(sums: dict) -> float | None:
    n = sums["n"]
    if n < 2:
        return None
    numerator = n * sums["sum_xy"] - sums["sum_x"] * sums["sum_y"]
    xpart = n * sums["sum_x2"] - sums["sum_x"] ** 2
    ypart = n * sums["sum_y2"] - sums["sum_y"] ** 2
    denominator = math.sqrt(max(0.0, xpart) * max(0.0, ypart))
    return numerator / denominator if denominator > 0 else None


def add_scaled(target: dict, source: dict, weight: float) -> None:
    for key in (
        "fill_count",
        "volume",
        "ell_positive",
        "sum_ell_positive_per_unit",
    ):
        target[key] += source[key] * weight
    for key in (
        "markout_count",
        "markout_volume",
        "markout_sum",
        "sum_markout_per_unit",
    ):
        for index in range(4):
            target[key][index] += source[key][index] * weight


def empty_risk() -> dict:
    return {
        "fill_count": 0.0,
        "volume": 0.0,
        "ell_positive": 0.0,
        "sum_ell_positive_per_unit": 0.0,
        "markout_count": [0.0] * 4,
        "markout_volume": [0.0] * 4,
        "markout_sum": [0.0] * 4,
        "sum_markout_per_unit": [0.0] * 4,
    }


def risk_deciles(risk_bins: dict, kind: str = "all") -> list[dict]:
    bins = []
    for key, values in risk_bins.items():
        key_kind, index = key.split(":")
        if key_kind == kind:
            bins.append((int(index), values))
    bins.sort()
    total = sum(values["fill_count"] for _, values in bins)
    if total == 0:
        return []
    deciles = [empty_risk() for _ in range(10)]
    boundaries = [total * i / 10 for i in range(11)]
    position = 0.0
    for _, values in bins:
        count = values["fill_count"]
        start, end = position, position + count
        for decile in range(10):
            overlap = max(0.0, min(end, boundaries[decile + 1]) - max(start, boundaries[decile]))
            if overlap > 0:
                add_scaled(deciles[decile], values, overlap / count)
        position = end
    result = []
    for index, values in enumerate(deciles):
        count, volume = values["fill_count"], values["volume"]
        result.append(
            {
                "decile": index + 1,
                "fill_count": count,
                "mean_ell_positive_per_abs_delta": (
                    values["sum_ell_positive_per_unit"] / count if count else None
                ),
                "volume_weighted_ell_positive_per_abs_delta": (
                    values["ell_positive"] / volume if volume else None
                ),
                "mean_markout_per_abs_delta": [
                    values["sum_markout_per_unit"][h] / values["markout_count"][h]
                    if values["markout_count"][h]
                    else None
                    for h in range(4)
                ],
                "volume_weighted_markout_per_abs_delta": [
                    values["markout_sum"][h] / values["markout_volume"][h]
                    if values["markout_volume"][h]
                    else None
                    for h in range(4)
                ],
            }
        )
    return result


def channel_factors(channels: dict) -> dict:
    n_events = channels["n_fund_events"]
    n_fills = channels["fund_fill_count"]
    served = channels["served"]
    a_fund = channels["a_fund"]
    return {
        "potential_events": float(n_events),
        "fill_incidence": n_fills / n_events if n_events else 0.0,
        "conditional_fill_size": served / n_fills if n_fills else 0.0,
        "conditional_adverse_severity": a_fund / served if served else 0.0,
    }


def product(factors: dict) -> float:
    return math.prod(factors.values())


def shapley_product(left: dict, right: dict) -> dict:
    names = list(left)
    contributions = dict.fromkeys(names, 0.0)
    permutations = list(itertools.permutations(names))
    for ordering in permutations:
        state = dict(left)
        before = product(state)
        for name in ordering:
            state[name] = right[name]
            after = product(state)
            contributions[name] += (after - before) / len(permutations)
            before = after
    return contributions


def comparison(left: dict, right: dict, comparison_id: str) -> dict:
    left_channels = left["channel_totals_100_seeds"]
    right_channels = right["channel_totals_100_seeds"]
    left_factors = channel_factors(left_channels)
    right_factors = channel_factors(right_channels)
    fund = shapley_product(left_factors, right_factors)
    delta_a_arb = right_channels["a_arb"] - left_channels["a_arb"]
    delta_a = right_channels["a"] - left_channels["a"]
    residual = delta_a - delta_a_arb - sum(fund.values())
    assert abs(residual) <= 1e-7 * max(1.0, abs(delta_a)), residual
    return {
        "comparison_id": comparison_id,
        "left_policy": left["policy"],
        "right_policy": right["policy"],
        "delta_right_minus_left": {
            key: right_channels[key] - left_channels[key]
            for key in (
                "a",
                "a_arb",
                "a_fund",
                "b_fund",
                "u",
                "fees",
                "served",
                "potential",
            )
        },
        "a_decomposition": {
            "delta_a_arb": delta_a_arb,
            "delta_a_fund_shapley": fund,
            "sum": delta_a_arb + sum(fund.values()),
            "residual": residual,
        },
        "left_factors": left_factors,
        "right_factors": right_factors,
    }


plan = json.loads(PLAN_PATH.read_text())
expected = plan["n_unique_policies"]
raw_policies = []
metadata = None
for path in SHARDS:
    with gzip.open(path, "rt") as f:
        shard = json.load(f)
    if metadata is None:
        metadata = {
            "fee_histogram": shard["fee_histogram"],
            "gap_bin_upper_edges_bps": shard["gap_bin_upper_edges_bps"],
            "markout_horizons_hours": shard["markout_horizons_hours"],
        }
    raw_policies.extend(shard["policies"])
assert len(raw_policies) == expected
assert len({policy_key(p["cell"], p["policy"]) for p in raw_policies}) == expected

policies = {}
assignment_lookup = defaultdict(dict)
for raw in raw_policies:
    aggregate = raw["aggregate"]
    decisions = aggregate["decisions"]
    channels = aggregate["channels"]
    assert channels["episodes"] == 100
    deciles = risk_deciles(aggregate["risk_bins"], "all")
    key = policy_key(raw["cell"], raw["policy"])
    result = {
        "cell": raw["cell"],
        "policy": raw["policy"],
        "assignments": raw["assignments"],
        "fee_distribution": {
            "n_decisions": decisions["n"],
            "mean": decisions["sum_fee"] / decisions["n"],
            "p10_approx": histogram_quantile(decisions["fee_histogram"], 0.10),
            "median_approx": histogram_quantile(decisions["fee_histogram"], 0.50),
            "p90_approx": histogram_quantile(decisions["fee_histogram"], 0.90),
            "maximum": decisions["max_fee"],
            "fraction_at_lower_clip": decisions["at_lower_clip"] / decisions["n"],
            "fraction_at_upper_clip": decisions["at_upper_clip"] / decisions["n"],
        },
        "fee_by_stale_gap_bin": decisions["stale_gap_bins"],
        "fee_by_contemporaneous_gap_bin": decisions["contemporaneous_gap_bins"],
        "risk_by_realized_fee_decile": deciles,
        "fee_risk_correlations": {
            "ell_positive_per_abs_delta": correlation(
                aggregate["fee_severity_correlation_sums"]
            ),
            "markout_per_abs_delta": [
                correlation(sums) for sums in aggregate["fee_markout_correlation_sums"]
            ],
        },
        "channel_totals_100_seeds": channels,
        "channel_means_per_episode": {
            key: value / channels["episodes"]
            for key, value in channels.items()
            if key not in {"episodes", "n_fund_events", "fund_fill_count"}
        },
        "channel_factors": channel_factors(channels),
    }
    policies[key] = result
    for assignment in raw["assignments"]:
        record_id = assignment["record_id"]
        suffix = f"_{assignment['role']}"
        if record_id.endswith(suffix):
            record_id = record_id[: -len(suffix)]
        assignment_lookup[record_id][assignment["role"]] = result

comparisons = []
for record_id, roles in sorted(assignment_lookup.items()):
    for left_role, right_role, suffix in (
        ("static_frontier", "gap_frontier", "frontier_gap_minus_static"),
        ("static_matched", "gap_positive_matched", "matched_gap_minus_static"),
    ):
        if left_role in roles and right_role in roles:
            comparisons.append(
                comparison(roles[left_role], roles[right_role], f"{record_id}:{suffix}")
            )

positive_gap = [
    policy
    for policy in policies.values()
    if policy["policy"]["family"] == "gap" and policy["policy"]["alpha"] > 0
]
severity_corr = [
    p["fee_risk_correlations"]["ell_positive_per_abs_delta"]
    for p in positive_gap
    if p["fee_risk_correlations"]["ell_positive_per_abs_delta"] is not None
]
markout_corr = {
    str(h): [
        p["fee_risk_correlations"]["markout_per_abs_delta"][index]
        for p in positive_gap
        if p["fee_risk_correlations"]["markout_per_abs_delta"][index] is not None
    ]
    for index, h in enumerate(metadata["markout_horizons_hours"])
}
top_bottom_severity = []
for policy in positive_gap:
    deciles = policy["risk_by_realized_fee_decile"]
    if deciles and deciles[0]["mean_ell_positive_per_abs_delta"] is not None:
        top_bottom_severity.append(
            deciles[-1]["mean_ell_positive_per_abs_delta"]
            - deciles[0]["mean_ell_positive_per_abs_delta"]
        )

risk_summary = {
    "n_positive_alpha_selected_policies": len(positive_gap),
    "fee_severity_correlation": {
        "median": statistics.median(severity_corr) if severity_corr else None,
        "fraction_positive": sum(value > 0 for value in severity_corr) / len(severity_corr)
        if severity_corr
        else None,
    },
    "fee_markout_correlation_by_horizon": {
        horizon: {
            "median": statistics.median(values) if values else None,
            "fraction_positive": sum(value > 0 for value in values) / len(values)
            if values
            else None,
        }
        for horizon, values in markout_corr.items()
    },
    "top_minus_bottom_fee_decile_severity": {
        "median": statistics.median(top_bottom_severity) if top_bottom_severity else None,
        "fraction_positive": sum(value > 0 for value in top_bottom_severity)
        / len(top_bottom_severity)
        if top_bottom_severity
        else None,
    },
}
risk_summary["classification_rule"] = (
    "weak signal when fewer than 60% of selected positive-alpha policies have positive "
    "fee/severity correlation or positive fee/5h-markout correlation"
)
severity_fraction = risk_summary["fee_severity_correlation"]["fraction_positive"]
markout_5h_fraction = risk_summary["fee_markout_correlation_by_horizon"]["5.0"][
    "fraction_positive"
]
risk_summary["classification"] = (
    "weak_policy_signal"
    if severity_fraction is None
    or markout_5h_fraction is None
    or severity_fraction < 0.60
    or markout_5h_fraction < 0.60
    else "risk_sorting_supported"
)

result = {
    "step": "validation-grid amended selected-policy diagnostics, training only",
    "diagnostic_plan_sha256": sha256(PLAN_PATH),
    "diagnostic_shards_sha256": {path.name: sha256(path) for path in SHARDS},
    "metadata": metadata,
    "risk_sorting_summary": risk_summary,
    "channel_comparisons": comparisons,
    "policies": list(policies.values()),
}
OUT.write_text(json.dumps(result, indent=1, sort_keys=True, allow_nan=False) + "\n")
OUT.with_suffix(".sha256").write_text(f"{sha256(OUT)}  {OUT.name}\n")
print(f"policies={len(policies)} comparisons={len(comparisons)}")
print(json.dumps(risk_summary, indent=1))
print(f"wrote {OUT}")
