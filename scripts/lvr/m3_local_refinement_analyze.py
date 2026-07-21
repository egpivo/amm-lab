#!/usr/bin/env python3
"""Analyze Round-34 P2 local refinement from training rows only."""

import csv
import gzip
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path


ROOT = Path("/Users/joseph/amm-lab")
LVR = ROOT / ".local/lvr"
PLAN = LVR / "m3_local_refinement_plan_v3.json"
POLICIES = LVR / "m3_local_refinement_policies.json"
SUPPORT = LVR / "m3_joint_support.json"
COARSE_ROWS = [LVR / f"m3_amended_training_rows_shard{i}.csv.gz" for i in range(6)]
REFINED_ROWS = [LVR / f"m3_local_refinement_rows_shard{i}.csv.gz" for i in range(6)]
ALIGNMENT_OUT = LVR / "m3_local_refinement_alignment.csv"
FRONTIER_OUT = LVR / "m3_local_refinement_frontier.csv"
CHANGES_OUT = LVR / "m3_local_refinement_selector_changes.csv"
PENALTIES_OUT = LVR / "m3_local_refinement_penalties.csv"
BREAKPOINTS_OUT = LVR / "m3_local_refinement_penalty_breakpoints.csv"
REPORT_OUT = LVR / "m3_local_refinement_report.md"
RHOS = (0.2, 0.4, 0.6, 0.8, 0.95)
METRICS = ("a", "b", "l", "fees", "u", "s")
V0 = 40_000_000.0
VALUE_TOLERANCE = 2e-5 * V0
IDENTITY_TOLERANCE = 1e-10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left: float, right: float, scale: float | None = None) -> bool:
    if scale is None:
        scale = max(1.0, abs(left), abs(right))
    return abs(left - right) <= 2e-10 * scale


def number_id(value: float) -> str:
    return format(value, ".12g").replace(".", "p")


def coarse_policy_id(dial: float, alpha: float) -> str:
    return f"gap_d{number_id(dial)}_a{number_id(alpha)}"


def freeze_csv(path: Path, rows: list[dict]) -> None:
    assert rows
    fields = list(rows[0])
    assert all(list(row) == fields for row in rows)
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    payload = buffer.getvalue().encode()
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"refusing to overwrite different frozen output: {path}")
    path.write_bytes(payload)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{hashlib.sha256(payload).hexdigest()}  {path.name}\n"
    )


def freeze_text(path: Path, text: str) -> None:
    payload = text.encode()
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"refusing to overwrite different frozen output: {path}")
    path.write_bytes(payload)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{hashlib.sha256(payload).hexdigest()}  {path.name}\n"
    )


def read_means(paths: list[Path], refined: bool) -> tuple[dict, dict, int]:
    sums = defaultdict(lambda: {metric: 0.0 for metric in METRICS})
    counts = Counter()
    specs = {}
    row_count = 0
    for path in paths:
        with gzip.open(path, "rt", newline="") as handle:
            for row in csv.DictReader(handle):
                if not refined and row["family"] != "gap":
                    continue
                row_count += 1
                cell_idx = int(row["cell_idx"])
                dial = float(row["dial_mult"])
                alpha = float(row["alpha"])
                policy_id = row["policy_id"] if refined else coarse_policy_id(dial, alpha)
                key = (cell_idx, policy_id)
                counts[key] += 1
                values = {metric: float(row[metric]) for metric in METRICS}
                scale = max(1.0, abs(values["a"]), abs(values["l"]), abs(values["u"]), abs(values["fees"]))
                assert abs(values["l"] - values["a"] + values["b"]) <= IDENTITY_TOLERANCE * scale
                assert abs(values["u"] - values["fees"] + values["l"]) <= IDENTITY_TOLERANCE * scale
                for metric, value in values.items():
                    assert math.isfinite(value)
                    sums[key][metric] += value
                specs[key] = {
                    "cell_idx": cell_idx,
                    "stratum": row["stratum"],
                    "sigma": float(row["sigma"]),
                    "z": float(row["z"]),
                    "speed": row["speed"],
                    "policy_id": policy_id,
                    "dial_mult": dial,
                    "f0": float(row["f0"]),
                    "alpha": alpha,
                    "fee_cap": float(row["fee_cap"]),
                    "source": "refined" if refined else "coarse",
                }
    assert set(counts.values()) == {100}
    means = {
        key: {
            **specs[key],
            **{f"mean_{metric}": value / counts[key] for metric, value in totals.items()},
        }
        for key, totals in sums.items()
    }
    return means, dict(counts), row_count


def ordered(points: list[dict], metric: str, maximize: bool) -> dict:
    assert points
    if maximize:
        return min(
            points,
            key=lambda point: (
                -point[f"mean_{metric}"],
                point["alpha"],
                point["dial_mult"],
                point["policy_id"],
            ),
        )
    return min(
        points,
        key=lambda point: (
            point[f"mean_{metric}"],
            point["alpha"],
            point["dial_mult"],
            point["policy_id"],
        ),
    )


def same_policy(left: dict, right: dict) -> bool:
    return left["policy_id"] == right["policy_id"]


def selector_fields(prefix: str, point: dict) -> dict:
    return {
        f"{prefix}_id": point["policy_id"],
        f"{prefix}_source": point["source"],
        f"{prefix}_dial_mult": point["dial_mult"],
        f"{prefix}_f0": point["f0"],
        f"{prefix}_alpha": point["alpha"],
        f"{prefix}_mean_a": point["mean_a"],
        f"{prefix}_mean_b": point["mean_b"],
        f"{prefix}_mean_l": point["mean_l"],
        f"{prefix}_mean_fees": point["mean_fees"],
        f"{prefix}_mean_u": point["mean_u"],
        f"{prefix}_mean_s": point["mean_s"],
    }


def surplus_envelope(lines: list[dict]) -> list[dict]:
    by_slope: dict[float, list[dict]] = defaultdict(list)
    for line in lines:
        item = {
            **line,
            "hull_intercept": line["mean_u"],
            "hull_slope": -line["shortfall"],
        }
        by_slope[item["hull_slope"]].append(item)

    candidates = []
    for slope in sorted(by_slope):
        group = by_slope[slope]
        best_intercept = max(item["hull_intercept"] for item in group)
        candidates.append(
            min(
                (item for item in group if close(item["hull_intercept"], best_intercept)),
                key=lambda item: (item["alpha"], item["dial_mult"], item["policy_id"]),
            )
        )

    hull = []
    starts = []
    for line in candidates:
        start = -math.inf
        while hull:
            previous = hull[-1]
            denominator = line["hull_slope"] - previous["hull_slope"]
            assert denominator > 0
            start = (previous["hull_intercept"] - line["hull_intercept"]) / denominator
            if starts[-1] == -math.inf or (start > starts[-1] and not close(start, starts[-1])):
                break
            hull.pop()
            starts.pop()
        if not hull:
            start = -math.inf
        hull.append(line)
        starts.append(start)

    segments = []
    for index, line in enumerate(hull):
        raw_start = starts[index]
        raw_end = starts[index + 1] if index + 1 < len(starts) else math.inf
        if raw_end < 0 and not close(raw_end, 0.0):
            continue
        start = max(0.0, raw_start)
        end = raw_end
        if math.isfinite(end) and (end < start or close(end, start)):
            continue
        values = [item["mean_u"] - start * item["shortfall"] for item in lines]
        optimum = max(values)
        selected_value = line["mean_u"] - start * line["shortfall"]
        assert close(selected_value, optimum, max(1.0, abs(optimum), *(abs(value) for value in values)))
        ties = sorted(
            item["policy_id"]
            for item, value in zip(lines, values)
            if close(value, optimum, max(1.0, abs(optimum)))
        )
        segments.append(
            {
                "lambda_start": start,
                "lambda_end": None if math.isinf(end) else end,
                "policy": line,
                "ties_at_start": ties,
            }
        )
    assert segments and segments[0]["lambda_start"] == 0.0
    assert segments[-1]["lambda_end"] is None
    return segments


def segment_strictly_after(segments: list[dict], threshold: float) -> dict:
    epsilon = 2e-10 * max(1.0, abs(threshold))
    for segment in segments:
        end = segment["lambda_end"]
        if segment["lambda_start"] <= threshold + epsilon and (
            end is None or end > threshold + epsilon
        ):
            return segment
    for segment in segments:
        if segment["lambda_start"] >= threshold - epsilon:
            return segment
    raise AssertionError((threshold, segments))


plan = json.loads(PLAN.read_text())
assert plan["phase"] == "Round 34 Phase C / P2"
for relative, expected in plan["input_sha256"].items():
    assert sha256(ROOT / relative) == expected, relative
assert plan["prohibited_inputs"]["validation_or_final_rows"] == "must not be read"

policy_manifest = json.loads(POLICIES.read_text())
expected_new = int(policy_manifest["counts"]["new_cell_policies"])
expected_ids = {
    (int(cell["cell_idx"]), policy["policy_id"])
    for cell in policy_manifest["cells"]
    for policy in cell["new_policies"]
}
assert len(expected_ids) == expected_new

coarse_means, coarse_counts, coarse_row_count = read_means(COARSE_ROWS, refined=False)
refined_means, refined_counts, refined_row_count = read_means(REFINED_ROWS, refined=True)
assert len(coarse_means) == 54 * 84
assert coarse_row_count == 54 * 84 * 100
assert set(refined_counts) == expected_ids
assert refined_row_count == expected_new * 100

support_data = json.loads(SUPPORT.read_text())
support_lookup = {
    (cell["stratum"], float(cell["sigma"]), float(cell["z"])): cell
    for cell in support_data["market_state_cells"]
}

coarse_by_cell = defaultdict(list)
refined_only_by_cell = defaultdict(list)
for point in coarse_means.values():
    coarse_by_cell[point["cell_idx"]].append(point)
for point in refined_means.values():
    refined_only_by_cell[point["cell_idx"]].append(point)
for cell_idx in range(54):
    assert len(coarse_by_cell[cell_idx]) == 84

alignment_rows = []
frontier_rows = []
change_rows = []
penalty_rows = []
breakpoint_rows = []
coarse_counts_summary = Counter()
refined_counts_summary = Counter()
coarse_supported = Counter()
refined_supported = Counter()
supported_u_gains = []
supported_u_changes = 0
lambda_normalized = []

for cell_idx in range(54):
    coarse_points = coarse_by_cell[cell_idx]
    refined_points = coarse_points + refined_only_by_cell[cell_idx]
    baseline = next(
        point
        for point in coarse_points
        if close(point["dial_mult"], 1.0) and close(point["alpha"], 0.0)
    )
    s0 = baseline["mean_s"]
    info = baseline
    support_cell = support_lookup[(info["stratum"], info["sigma"], info["z"])]
    supported = support_cell["support_label"] == "supported"

    for rho in RHOS:
        target = rho * s0
        coarse_feasible = [point for point in coarse_points if point["mean_s"] >= target]
        refined_feasible = [point for point in refined_points if point["mean_s"] >= target]
        assert coarse_feasible and refined_feasible

        coarse_selectors = {
            "a": ordered(coarse_feasible, "a", False),
            "l": ordered(coarse_feasible, "l", False),
            "u": ordered(coarse_feasible, "u", True),
        }
        refined_selectors = {
            "a": ordered(refined_feasible, "a", False),
            "l": ordered(refined_feasible, "l", False),
            "u": ordered(refined_feasible, "u", True),
        }

        for label, selectors, counter in (
            ("coarse", coarse_selectors, coarse_counts_summary),
            ("refined", refined_selectors, refined_counts_summary),
        ):
            counter["a_u"] += not same_policy(selectors["a"], selectors["u"])
            counter["l_u"] += not same_policy(selectors["l"], selectors["u"])
            counter["a_l"] += not same_policy(selectors["a"], selectors["l"])
            if supported:
                target_counter = coarse_supported if label == "coarse" else refined_supported
                target_counter["a_u"] += not same_policy(selectors["a"], selectors["u"])
                target_counter["l_u"] += not same_policy(selectors["l"], selectors["u"])
                target_counter["a_l"] += not same_policy(selectors["a"], selectors["l"])

        u_gain = refined_selectors["u"]["mean_u"] - coarse_selectors["u"]["mean_u"]
        assert u_gain >= -1e-8
        if supported:
            supported_u_gains.append(abs(u_gain))
            supported_u_changes += not same_policy(coarse_selectors["u"], refined_selectors["u"])

        common = {
            "cell_idx": cell_idx,
            "stratum": info["stratum"],
            "sigma": info["sigma"],
            "z": info["z"],
            "speed": info["speed"],
            "support_label": support_cell["support_label"],
            "observed_pool_weeks": support_cell["observed_pool_weeks"],
            "rho": rho,
            "training_S0": s0,
            "service_target": target,
            "n_coarse_policies": len(coarse_points),
            "n_refined_policies": len(refined_points),
            "n_coarse_feasible": len(coarse_feasible),
            "n_refined_feasible": len(refined_feasible),
        }
        alignment_rows.append(
            {
                **common,
                **selector_fields("coarse_a", coarse_selectors["a"]),
                **selector_fields("refined_a", refined_selectors["a"]),
                **selector_fields("coarse_l", coarse_selectors["l"]),
                **selector_fields("refined_l", refined_selectors["l"]),
                **selector_fields("coarse_u", coarse_selectors["u"]),
                **selector_fields("refined_u", refined_selectors["u"]),
                "refined_a_gain": coarse_selectors["a"]["mean_a"] - refined_selectors["a"]["mean_a"],
                "refined_l_gain": coarse_selectors["l"]["mean_l"] - refined_selectors["l"]["mean_l"],
                "refined_u_gain": u_gain,
                "refined_a_u_diverges": not same_policy(refined_selectors["a"], refined_selectors["u"]),
                "refined_l_u_diverges": not same_policy(refined_selectors["l"], refined_selectors["u"]),
                "refined_a_l_diverges": not same_policy(refined_selectors["a"], refined_selectors["l"]),
            }
        )

        coarse_static = ordered([point for point in coarse_feasible if close(point["alpha"], 0.0)], "u", True)
        refined_static = ordered([point for point in refined_feasible if close(point["alpha"], 0.0)], "u", True)
        coarse_gap = coarse_selectors["u"]
        refined_gap = refined_selectors["u"]
        frontier_rows.append(
            {
                **common,
                **selector_fields("coarse_static", coarse_static),
                **selector_fields("refined_static", refined_static),
                **selector_fields("coarse_gap", coarse_gap),
                **selector_fields("refined_gap", refined_gap),
                "static_u_gain": refined_static["mean_u"] - coarse_static["mean_u"],
                "gap_u_gain": refined_gap["mean_u"] - coarse_gap["mean_u"],
                "coarse_strict_gap_over_static": coarse_gap["alpha"] > 0 and coarse_gap["mean_u"] > coarse_static["mean_u"],
                "refined_strict_gap_over_static": refined_gap["alpha"] > 0 and refined_gap["mean_u"] > refined_static["mean_u"],
            }
        )

        selector_specs = (
            ("A", coarse_selectors["a"], refined_selectors["a"], "a", False),
            ("L", coarse_selectors["l"], refined_selectors["l"], "l", False),
            ("U", coarse_selectors["u"], refined_selectors["u"], "u", True),
            ("STATIC_FRONTIER", coarse_static, refined_static, "u", True),
            ("GAP_FRONTIER", coarse_gap, refined_gap, "u", True),
        )
        for selector, coarse_point, refined_point, metric, maximize in selector_specs:
            if same_policy(coarse_point, refined_point):
                continue
            gain = (
                refined_point[f"mean_{metric}"] - coarse_point[f"mean_{metric}"]
                if maximize
                else coarse_point[f"mean_{metric}"] - refined_point[f"mean_{metric}"]
            )
            change_rows.append(
                {
                    **common,
                    "selector": selector,
                    "objective_metric": metric,
                    "objective_gain": gain,
                    **selector_fields("coarse", coarse_point),
                    **selector_fields("refined", refined_point),
                }
            )

        lines = [
            {**point, "shortfall": max(target - point["mean_s"], 0.0)}
            for point in refined_points
        ]
        u_star = refined_selectors["u"]["mean_u"]
        ratios = [
            (max(point["mean_u"] - u_star, 0.0) / point["shortfall"], point)
            for point in lines
            if point["shortfall"] > 0.0
        ]
        lambda_star = max((ratio for ratio, _ in ratios), default=0.0)
        binding = sorted(point["policy_id"] for ratio, point in ratios if close(ratio, lambda_star))
        envelope = surplus_envelope(lines)
        recovered = segment_strictly_after(envelope, lambda_star)["policy"]
        recovery_ok = recovered["shortfall"] == 0.0 and close(recovered["mean_u"], u_star)
        assert recovery_ok
        normalized = lambda_star * s0 / V0
        lambda_normalized.append(normalized)
        penalty_rows.append(
            {
                **common,
                "n_refined_gap_policies": len(refined_points),
                "lambda_star": lambda_star,
                "lambda_star_normalized_S0_over_V0": normalized,
                "lambda_star_binding_ids": ";".join(binding),
                "surplus_envelope_segments": len(envelope),
                "constrained_u_id": refined_selectors["u"]["policy_id"],
                "penalized_u_strictly_above_threshold_id": recovered["policy_id"],
                "exact_surplus_recovery": recovery_ok,
                "penalized_l_large_penalty_id": refined_selectors["l"]["policy_id"],
                "large_penalty_l_differs_from_u": not same_policy(refined_selectors["l"], refined_selectors["u"]),
            }
        )
        for segment_index, segment in enumerate(envelope):
            point = segment["policy"]
            breakpoint_rows.append(
                {
                    **common,
                    "segment_index": segment_index,
                    "lambda_start_inclusive": segment["lambda_start"],
                    "lambda_end_exclusive": "" if segment["lambda_end"] is None else segment["lambda_end"],
                    "ties_at_lambda_start": ";".join(segment["ties_at_start"]),
                    **selector_fields("selected", point),
                    "service_shortfall": point["shortfall"],
                    "line_intercept": point["mean_u"],
                    "line_slope": -point["shortfall"],
                    "feasible": point["shortfall"] == 0.0,
                }
            )

assert coarse_counts_summary == Counter({"l_u": 196, "a_u": 158, "a_l": 100})
assert coarse_supported == Counter({"l_u": 90, "a_u": 79, "a_l": 43})
coarse_frontier_strict = sum(row["coarse_strict_gap_over_static"] for row in frontier_rows)
assert coarse_frontier_strict == 249

supported_n = len(supported_u_gains)
assert supported_n == 135
value_stable_count = sum(gain <= VALUE_TOLERANCE + 1e-9 for gain in supported_u_gains)
value_stable_share = value_stable_count / supported_n
supported_l_u_share = refined_supported["l_u"] / supported_n
supported_u_change_share = supported_u_changes / supported_n
if supported_l_u_share < 0.50 or value_stable_share < 0.80:
    classification = "GRID-SENSITIVE"
elif supported_u_change_share > 0.20:
    classification = "VALUE-STABLE, SELECTOR-SENSITIVE"
else:
    classification = "STABLE"

freeze_csv(ALIGNMENT_OUT, alignment_rows)
freeze_csv(FRONTIER_OUT, frontier_rows)
freeze_csv(CHANGES_OUT, change_rows)
freeze_csv(PENALTIES_OUT, penalty_rows)
freeze_csv(BREAKPOINTS_OUT, breakpoint_rows)

row_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in REFINED_ROWS}
change_counts = Counter(row["selector"] for row in change_rows)
refined_frontier_strict = sum(row["refined_strict_gap_over_static"] for row in frontier_rows)
report = f"""# M3 P2 Local-Refinement Report

## Scope and integrity

This training-only diagnostic combines the frozen 84-member coarse gap grid
with `{expected_new}` new cell-policy midpoint evaluations. It uses seeds
`20000..20099` only. No validation or final row was read, no held-out candidate
was selected, and no policy was extrapolated beyond the frozen grid bounds.

- New simulator rows: **{refined_row_count:,}**.
- Rows per new cell-policy: **100/100 for all {expected_new} policies**.
- Row identities `L=A-B` and `U=fees-L`: **PASS** at relative tolerance
  `{IDENTITY_TOLERANCE}`.
- Coarse baseline reproduction: A/U **158/270**, L/U **196/270**, A/L
  **100/270**, supported L/U **90/135**, and strict gap frontier **249/270**.

## Refined selectors

| comparison | all cell-targets | directly supported |
|---|---:|---:|
| A/U disagreement | {refined_counts_summary['a_u']}/270 | {refined_supported['a_u']}/135 |
| L/U disagreement | {refined_counts_summary['l_u']}/270 | {refined_supported['l_u']}/135 |
| A/L disagreement | {refined_counts_summary['a_l']}/270 | {refined_supported['a_l']}/135 |

- Strict gap-over-static frontier improvement: **{refined_frontier_strict}/270**.
- Supported U-selector changes: **{supported_u_changes}/135
  ({100 * supported_u_change_share:.1f}%)**.
- Supported states with absolute refined U gain no larger than 800:
  **{value_stable_count}/135 ({100 * value_stable_share:.1f}%)**.
- Supported refined L/U disagreement share: **{100 * supported_l_u_share:.1f}%**.
- Classification: **{classification}**.

Every selector identity change is listed in
`m3_local_refinement_selector_changes.csv`. Change counts are A
{change_counts['A']}, L {change_counts['L']}, U {change_counts['U']}, static
frontier {change_counts['STATIC_FRONTIER']}, and gap frontier
{change_counts['GAP_FRONTIER']}.

## Refined exact penalties

The exact surplus threshold and full penalized-surplus envelope were
recomputed on the expanded finite set for every cell-target state.

- Exact surplus recovery above the refined threshold: **270/270**.
- Penalized-L large-penalty selector differs from constrained U in
  **{refined_counts_summary['l_u']}/270**, including
  **{refined_supported['l_u']}/135** directly supported states.
- Normalized threshold `lambda_star*S0/V0`: min
  **{min(lambda_normalized):.6g}**, median
  **{statistics.median(lambda_normalized):.6g}**, max
  **{max(lambda_normalized):.6g}**.

The thresholds are grid-specific training constructions, not identified
service prices. Intermediate penalties remain uninterpreted.

## Integrity hashes

- Plan SHA-256: `{sha256(PLAN)}`.
- Policy manifest SHA-256: `{sha256(POLICIES)}`.
- Analyzer SHA-256: `{sha256(Path(__file__).resolve())}`.
- Refined row shards: `{json.dumps(row_hashes, sort_keys=True)}`.

## Stop gate

P2 is complete with classification **{classification}**. P3 and P4 remain
blocked pending reviewer review.
"""
freeze_text(REPORT_OUT, report)
print(f"classification={classification}")
print(
    f"refined A/U={refined_counts_summary['a_u']}/270 "
    f"L/U={refined_counts_summary['l_u']}/270 A/L={refined_counts_summary['a_l']}/270"
)
print(f"supported L/U={refined_supported['l_u']}/135 U changes={supported_u_changes}/135")
print(f"wrote {REPORT_OUT}")
