#!/usr/bin/env python3
"""Map observed target-pool weeks to the nearest M3 market-state cell."""

import csv
import datetime
import gzip
import hashlib
import json
import math
import statistics
from array import array
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/Users/joseph/amm-lab")
LVR = ROOT / ".local/lvr"
EVENTS = LVR / "deep_subset_3pools_all_events.csv.gz"
Z_ROWS = LVR / "deep_subset_z_rows.csv.gz"
SETFEE = ROOT / "data/causality/setfeeprotocol_events.csv"
CALIBRATION = LVR / "calibration_54_manifest.json"
REFERENCE_MANIFEST = LVR / "reference_series_manifest.json"
OUT = LVR / "m3_joint_support.json"
TABLE = LVR / "m3_joint_support.csv"

REF_POOL = "0xe0554a476a092703abdb3ef35c80e0d76d32939f"
TARGETS = {
    "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640": "5bp",
    "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8": "30bp",
}
Q96 = 2**96
BLOCKS_5MIN = 25
GATE_BLOCKS = 75
SEC_PER_YEAR = 365 * 24 * 3600
SIGMA_ANCHORS = [0.48, 0.64, 0.92]
Z_ANCHORS = [0.00087, 0.0055, 0.03]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iso_week(ts: int) -> str:
    return datetime.datetime.fromtimestamp(ts, datetime.UTC).strftime("%G-%V")


def nearest(value: float, anchors: list[float], log_scale: bool = False) -> float:
    distance = (
        (lambda x: abs(math.log(value) - math.log(x)))
        if log_scale
        else (lambda x: abs(value - x))
    )
    return min(anchors, key=distance)


treat_block = {}
with SETFEE.open() as f:
    for row in csv.DictReader(f):
        pool = row["pool"].lower()
        if pool in TARGETS or pool == REF_POOL:
            block = int(row["block"])
            treat_block[pool] = min(block, treat_block.get(pool, block))
window_end = min(treat_block[pool] for pool in TARGETS)

ref_blocks = array("q")
ref_ts = array("q")
ref_logp = array("d")
activity = Counter()
first_block = None
last_ref_key = None
with gzip.open(EVENTS, "rt", newline="") as f:
    for row in csv.DictReader(f):
        if row["type"] != "swap":
            continue
        block = int(row["block"])
        first_block = block if first_block is None else min(first_block, block)
        if block >= window_end:
            continue
        pool = row["pool"].lower()
        if pool in TARGETS:
            activity[(pool, iso_week(int(row["ts"])))] += 1
        elif pool == REF_POOL and row["sqrtP"]:
            key = (block, int(row["tx_index"]), int(row["log_index"]))
            assert last_ref_key is None or key >= last_ref_key, "reference rows not chain ordered"
            last_ref_key = key
            sqrtp = int(row["sqrtP"])
            ref_blocks.append(block)
            ref_ts.append(int(row["ts"]))
            ref_logp.append(math.log(1e12) - 2 * math.log(sqrtp / Q96))
assert first_block is not None and ref_blocks

weekly_r2 = defaultdict(list)
pointer = -1
previous = None
for grid_block in range(first_block + BLOCKS_5MIN, window_end, BLOCKS_5MIN):
    while pointer + 1 < len(ref_blocks) and ref_blocks[pointer + 1] <= grid_block:
        pointer += 1
    current = None
    if pointer >= 0:
        stale = grid_block - ref_blocks[pointer]
        if stale <= GATE_BLOCKS:
            current = (ref_logp[pointer], iso_week(ref_ts[pointer]))
    if previous is not None and current is not None:
        weekly_r2[current[1]].append((current[0] - previous[0]) ** 2)
    previous = current

reference_manifest = json.loads(REFERENCE_MANIFEST.read_text())
ramp_week = reference_manifest["post_ramp_start_week"]
weeks = sorted(weekly_r2)
complete = [week for week in weeks[1:-1] if week >= ramp_week]
annualization = SEC_PER_YEAR / (BLOCKS_5MIN * 12)
weekly_sigma = {}
for week in complete:
    values = weekly_r2[week]
    if len(values) >= 0.8 * 2016:
        weekly_sigma[week] = math.sqrt(annualization * statistics.fmean(values))
assert len(weekly_sigma) == reference_manifest["sigma_weeks"]

z_values = defaultdict(list)
with gzip.open(Z_ROWS, "rt", newline="") as f:
    for row in csv.DictReader(f):
        if int(row["eps_bps"]) != 100:
            continue
        pool = row["pool"].lower()
        if pool in TARGETS:
            z_values[(pool, iso_week(int(row["ts"])))].append(float(row["z"]))
weekly_z = {key: statistics.median(values) for key, values in z_values.items() if values}

manifest = json.loads(CALIBRATION.read_text())
activity_target = {}
for cell in manifest["cells"]:
    activity_target.setdefault(cell["stratum"], cell["total_target_per_week"])
    assert activity_target[cell["stratum"]] == cell["total_target_per_week"]

observed = []
support = Counter()
for pool, stratum in TARGETS.items():
    common_weeks = sorted(
        set(week for p, week in weekly_z if p == pool)
        & set(weekly_sigma)
        & set(week for p, week in activity if p == pool)
    )
    for week in common_weeks:
        sigma = weekly_sigma[week]
        z = weekly_z[(pool, week)]
        swaps = activity[(pool, week)]
        sigma_cell = nearest(sigma, SIGMA_ANCHORS)
        z_cell = nearest(z, Z_ANCHORS, log_scale=True)
        key = (stratum, sigma_cell, z_cell)
        support[key] += 1
        observed.append(
            {
                "pool": pool,
                "stratum": stratum,
                "week": week,
                "sigma": sigma,
                "z_weekly_median_eps100": z,
                "activity_swaps": swaps,
                "activity_target": activity_target[stratum],
                "activity_ratio_to_grid_target": swaps / activity_target[stratum],
                "nearest_sigma": sigma_cell,
                "nearest_z": z_cell,
            }
        )

market_cells = []
total = len(observed)
for stratum in ("5bp", "30bp"):
    for sigma in SIGMA_ANCHORS:
        for z in Z_ANCHORS:
            count = support[(stratum, sigma, z)]
            market_cells.append(
                {
                    "stratum": stratum,
                    "sigma": sigma,
                    "z": z,
                    "observed_pool_weeks": count,
                    "empirical_weight": count / total if total else 0.0,
                    "support_label": (
                        "unobserved" if count == 0 else "sparse" if count < 5 else "supported"
                    ),
                }
            )

speed_replicas = []
for cell in manifest["cells"]:
    count = support[(cell["stratum"], cell["sigma"], cell["z"])]
    speed_replicas.append(
        {
            "cell_idx": manifest["cells"].index(cell),
            "stratum": cell["stratum"],
            "sigma": cell["sigma"],
            "z": cell["z"],
            "arb_speed": cell["arb_speed"],
            "market_state_pool_weeks": count,
            "arb_speed_jointly_observed": False,
            "support_label": "boundary" if count == 0 else "market-state-supported_speed-sensitivity",
        }
    )

result = {
    "stage": "M3 empirical joint-support audit",
    "sample": {
        "target_pools": TARGETS,
        "weeks_with_joint_sigma_z_activity": total,
        "sigma_source": "post-ramp 1bp cross-pool reference proxy, 5-minute grid",
        "z_source": "exact 100bp directional-depth normalized swap size, weekly median",
        "activity_source": "target-pool landed swaps per week",
    },
    "mapping": {
        "sigma_distance": "absolute",
        "z_distance": "absolute log distance",
        "activity_axis_status": (
            "The 54-cell grid has one fixed total-activity target per stratum, not an activity axis; "
            "the observed activity ratio is reported but cannot select another cell level."
        ),
        "arb_speed_status": (
            "Arbitrage speed is not jointly observed per pool-week. Support counts identify 18 "
            "stratum-sigma-z market states; the three speed members remain sensitivity replicas."
        ),
    },
    "market_state_cells": market_cells,
    "speed_replicas": speed_replicas,
    "observed_pool_weeks": observed,
    "summary": {
        "supported_market_cells": sum(c["support_label"] == "supported" for c in market_cells),
        "sparse_market_cells": sum(c["support_label"] == "sparse" for c in market_cells),
        "unobserved_market_cells": sum(c["support_label"] == "unobserved" for c in market_cells),
        "weighting_rule": (
            "Weight market states by observed pool-week counts. For an aggregate over arb speed, "
            "report each speed separately or disclose equal one-third sensitivity weights; the data "
            "do not identify speed weights jointly."
        ),
    },
    "input_sha256": {
        str(path.relative_to(ROOT)): sha256(path)
        for path in (EVENTS, Z_ROWS, SETFEE, CALIBRATION, REFERENCE_MANIFEST)
    },
}
OUT.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
with TABLE.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(market_cells[0]))
    writer.writeheader()
    writer.writerows(market_cells)
print(f"joint_pool_weeks={total}")
print(f"summary={result['summary']}")
print(f"wrote {OUT} and {TABLE}")
