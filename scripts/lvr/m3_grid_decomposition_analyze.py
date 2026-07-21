#!/usr/bin/env python3
"""Aggregate validation-grid paired-ledger decomposition."""

import csv
import gzip
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path("/Users/joseph/amm-lab")
LVR = ROOT / ".local/lvr"
MODES = ("primary", "secondary")
SHARDS = 6


def read_rows(mode: str) -> list[dict]:
    rows = []
    for shard in range(SHARDS):
        path = LVR / f"m3_validation_grid_decomposition_{mode}_shard{shard}.csv.gz"
        if not path.exists():
            continue
        with gzip.open(path, "rt", newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def f(x: str) -> float:
    return float(x)


def cell_means(rows: list[dict]) -> dict[str, dict]:
    by_record: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_record[r["record_id"]].append(r)
    out = {}
    for rid, rs in by_record.items():
        n = len(rs)
        assert n > 0

        def mean(col: str) -> float:
            return statistics.fmean(f(r[col]) for r in rs)

        total_a = mean("total.delta_a")
        total_common = mean("total.delta_common")
        total_sel = mean("total.delta_selection")
        fund_a = mean("fund.delta_a")
        arb_a = mean("arb.delta_a")
        out[rid] = {
            "record_id": rid,
            "comparison": rs[0]["comparison"],
            "cell_idx": int(rs[0]["cell_idx"]),
            "stratum": rs[0]["stratum"],
            "sigma": f(rs[0]["sigma"]),
            "z": f(rs[0]["z"]),
            "speed": rs[0]["speed"],
            "rho": f(rs[0]["rho"]),
            "support_label": rs[0]["support_label"],
            "observed_pool_weeks": int(float(rs[0]["observed_pool_weeks"])),
            "target_s_training": f(rs[0]["target_s_training"]),
            "n_seeds": n,
            "delta_a": total_a,
            "delta_common": total_common,
            "delta_selection": total_sel,
            "delta_qty_c": mean("total.delta_qty_c"),
            "delta_sev_c": mean("total.delta_sev_c"),
            "delta_entry": mean("total.delta_entry"),
            "delta_exit": mean("total.delta_exit"),
            "delta_a_fund": fund_a,
            "delta_a_arb": arb_a,
            "delta_s": mean("delta_s"),
            "delta_fees": mean("delta_fees"),
            "delta_u": mean("delta_u"),
            "D_sel": int(abs(total_sel) > abs(total_common)),
            "D_hidden_reversal": int(total_a < 0 and total_common > 0),
            "D_leg_opposition": int(
                fund_a != 0 and arb_a != 0 and (fund_a > 0) != (arb_a > 0)
            ),
            "reconstruct_ok": all(r["reconstruct_ok"] in {"true", "True", "1"} for r in rs),
        }
    return out


def quantile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    pos = p * (len(xs) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def summarize_group(cells: list[dict]) -> dict:
    if not cells:
        return {"n": 0}
    lower_a = [c for c in cells if c["delta_a"] < 0]
    return {
        "n": len(cells),
        "n_lower_a": len(lower_a),
        "share_selection_dominated_among_lower_a": (
            statistics.fmean(c["D_sel"] for c in lower_a) if lower_a else None
        ),
        "share_hidden_reversal_among_lower_a": (
            statistics.fmean(c["D_hidden_reversal"] for c in lower_a) if lower_a else None
        ),
        "share_leg_opposition": statistics.fmean(c["D_leg_opposition"] for c in cells),
        "median_delta_a": statistics.median(c["delta_a"] for c in cells),
        "median_delta_common": statistics.median(c["delta_common"] for c in cells),
        "median_delta_selection": statistics.median(c["delta_selection"] for c in cells),
        "iqr_delta_common": (
            quantile([c["delta_common"] for c in cells], 0.75)
            - quantile([c["delta_common"] for c in cells], 0.25)
        ),
        "iqr_delta_selection": (
            quantile([c["delta_selection"] for c in cells], 0.75)
            - quantile([c["delta_selection"] for c in cells], 0.25)
        ),
    }


def analyze_mode(mode: str) -> dict:
    rows = read_rows(mode)
    assert rows, f"no rows for mode={mode}"
    assert all(r["reconstruct_ok"] in {"true", "True", "1"} for r in rows), "reconstruction failed"
    cells = list(cell_means(rows).values())
    by_support: dict[str, list[dict]] = defaultdict(list)
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    by_speed: dict[str, list[dict]] = defaultdict(list)
    for c in cells:
        by_support[c["support_label"]].append(c)
        by_stratum[c["stratum"]].append(c)
        by_speed[c["speed"]].append(c)

    lower_a = [c for c in cells if c["delta_a"] < 0]
    return {
        "mode": mode,
        "n_rows": len(rows),
        "n_cells": len(cells),
        "overall": summarize_group(cells),
        "among_lower_a": summarize_group(lower_a),
        "by_support_label": {k: summarize_group(v) for k, v in sorted(by_support.items())},
        "by_stratum": {k: summarize_group(v) for k, v in sorted(by_stratum.items())},
        "by_speed": {k: summarize_group(v) for k, v in sorted(by_speed.items())},
        "cells": sorted(cells, key=lambda c: (c["cell_idx"], c["rho"])),
    }


result = {mode: analyze_mode(mode) for mode in MODES if any(
    (LVR / f"m3_validation_grid_decomposition_{mode}_shard{i}.csv.gz").exists()
    for i in range(SHARDS)
)}
out = LVR / "m3_validation_grid_decomposition_result.json"
out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
print(f"wrote {out}")
for mode, res in result.items():
    o = res["overall"]
    la = res["among_lower_a"]
    print(f"\n=== {mode} ===")
    print(f"cells={res['n_cells']} rows={res['n_rows']}")
    print(
        f"lower-A cells: n={la['n_lower_a']} "
        f"selection-dominated={la.get('share_selection_dominated_among_lower_a')} "
        f"hidden-reversal={la.get('share_hidden_reversal_among_lower_a')}"
    )
    for label, grp in res["by_support_label"].items():
        la_g = [c for c in res["cells"] if c["support_label"] == label and c["delta_a"] < 0]
        if not la_g:
            continue
        print(
            f"  support={label}: lower-A n={len(la_g)} "
            f"hidden-reversal={statistics.fmean(c['D_hidden_reversal'] for c in la_g):.2f}"
        )
