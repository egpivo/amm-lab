#!/usr/bin/env python3
"""Magnitude-threshold sensitivity for frozen M3 selector divergences."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / ".local/lvr/m3_loss_alignment.csv"
OUTPUT = ROOT / ".local/lvr/m3_divergence_threshold_sensitivity.csv"
REPORT = ROOT / ".local/lvr/m3_divergence_threshold_sensitivity_report.md"
EXPECTED_INPUT_SHA256 = "78e14777ddf3c76ea2e7cd6883fff874fea72fcf2ec623e6643143a526675ce3"
V0 = 40_000_000.0
THETAS = (0.0, 1e-5, 1e-4, 1e-3)
COMPARISONS = {
    "A_vs_U": ("a_vs_u_diverges", "pi_a_star_mean_u"),
    "L_vs_U": ("l_vs_u_diverges", "pi_l_star_mean_u"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(value: str) -> bool:
    if value not in {"True", "False"}:
        raise ValueError(value)
    return value == "True"


def main() -> None:
    observed_hash = sha256(INPUT)
    if observed_hash != EXPECTED_INPUT_SHA256:
        raise RuntimeError(f"frozen input hash mismatch: {observed_hash}")
    with INPUT.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 270:
        raise RuntimeError(f"expected 270 rows, found {len(rows)}")
    if sum(row["support_label"] == "supported" for row in rows) != 135:
        raise RuntimeError("supported denominator changed")
    for row in rows:
        if any(int(row[field]) != 1 for field in (
            "argmin_a_tie_count",
            "argmin_l_tie_count",
            "argmax_u_tie_count",
        )):
            raise RuntimeError("primary-objective tie detected")

    output_rows: list[dict[str, str | int | float]] = []
    for comparison, (divergence_field, loss_u_field) in COMPARISONS.items():
        base_all = sum(as_bool(row[divergence_field]) for row in rows)
        expected_base = 158 if comparison == "A_vs_U" else 196
        if base_all != expected_base:
            raise RuntimeError(f"{comparison} baseline changed: {base_all}")
        for theta in THETAS:
            absolute_threshold = theta * V0
            for subset in ("all", "supported"):
                selected = rows if subset == "all" else [
                    row for row in rows if row["support_label"] == "supported"
                ]
                count = 0
                for row in selected:
                    if not as_bool(row[divergence_field]):
                        continue
                    delta_u = abs(float(row[loss_u_field]) - float(row["pi_u_star_mean_u"]))
                    if delta_u > absolute_threshold:
                        count += 1
                denominator = len(selected)
                output_rows.append(
                    {
                        "comparison": comparison,
                        "theta": f"{theta:.0e}" if theta else "0",
                        "absolute_delta_u_threshold": f"{absolute_threshold:.12g}",
                        "subset": subset,
                        "count": count,
                        "denominator": denominator,
                        "share": f"{count / denominator:.12g}",
                    }
                )

    fieldnames = [
        "comparison",
        "theta",
        "absolute_delta_u_threshold",
        "subset",
        "count",
        "denominator",
        "share",
    ]
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    def result(comparison: str, theta: float, subset: str) -> dict[str, str | int | float]:
        theta_text = f"{theta:.0e}" if theta else "0"
        return next(
            row
            for row in output_rows
            if row["comparison"] == comparison
            and row["theta"] == theta_text
            and row["subset"] == subset
        )

    lines = [
        "# M3 Divergence Magnitude-Threshold Sensitivity",
        "",
        "## Scope",
        "",
        "This no-new-seed audit uses only frozen training selectors from",
        "`m3_loss_alignment.csv`. A disagreement is retained only when the",
        "loss-selected and surplus-selected policies differ and",
        "`abs(Delta U) > theta*V0`, with `V0=40,000,000`. The original",
        "deterministic tie rule is unchanged; no primary-objective tie occurs.",
        "",
        "## Counts",
        "",
        "| theta | A/U all | A/U supported | L/U all | L/U supported |",
        "|---:|---:|---:|---:|---:|",
    ]
    for theta in THETAS:
        a_all = result("A_vs_U", theta, "all")
        a_sup = result("A_vs_U", theta, "supported")
        l_all = result("L_vs_U", theta, "all")
        l_sup = result("L_vs_U", theta, "supported")
        lines.append(
            f"| `{a_all['theta']}` | {a_all['count']}/{a_all['denominator']} "
            f"| {a_sup['count']}/{a_sup['denominator']} "
            f"| {l_all['count']}/{l_all['denominator']} "
            f"| {l_sup['count']}/{l_sup['denominator']} |"
        )
    lines.extend(
        [
            "",
            "The threshold is a descriptive magnitude filter on LP-relative-surplus",
            "differences, not a new selector, inference gate, or re-tuning rule.",
            "Counts describe the frozen finite policy grid and calibrated model cells.",
            "",
            "## Integrity",
            "",
            f"- Input SHA-256: `{observed_hash}`.",
            f"- Output rows: {len(output_rows)}.",
            "- Simulator runs: none.",
            "- New seeds: none.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUTPUT}")
    print(f"wrote {REPORT}")
    for comparison in COMPARISONS:
        all_1e4 = result(comparison, 1e-4, "all")
        sup_1e4 = result(comparison, 1e-4, "supported")
        print(
            f"{comparison}@1e-4={all_1e4['count']}/{all_1e4['denominator']} "
            f"supported={sup_1e4['count']}/{sup_1e4['denominator']}"
        )


if __name__ == "__main__":
    main()
