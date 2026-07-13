#!/usr/bin/env python3
"""Fast subset golden for early parity validation: run build_outcomes on the top-N pools by
12-month fee revenue (the busiest pools, incl. USDC/USDT/WBTC pools that exercise non-18
decimals, JIT, week-splitting, all outcome fields) via pool_filter, so the SQLite stays tiny
and finishes in minutes instead of the full-index-thrash hours.

Writes panel_weekly_subset_frozen.csv (+ subset_pools.json) into AMMLAB_DATA. Compare against
the full Rust panel with `panel_compare GOLDEN RUST --subset`.
"""
import csv, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_outcomes as bo  # noqa: E402  (module-level loads use AMMLAB_ROOT/DATA + .env)

PER_CLASS = int(os.environ.get("SUBSET_PER_CLASS", "4"))

# Moderate-activity band: enough events to exercise mints/burns/swaps/collects/JIT/week-split,
# few enough to ingest in seconds (avoids the giants' UNIQUE-index thrash). Take a few per
# pair-class so 18- (weth), 8- (btc), and 6-decimal (stable) tokens are all covered.
from collections import defaultdict  # noqa: E402

rows = [r for r in csv.DictReader(open(os.path.join(bo.DATA, "feerev_panelvars.csv")))
        if r.get("sw12", "").isdigit() and 200 <= int(r["sw12"]) <= 3000]
by_class = defaultdict(list)
for r in rows:
    by_class[r["class"]].append(r)
pick = []
for cls, rs in by_class.items():
    rs.sort(key=lambda r: int(r["sw12"]))  # smallest-but-nontrivial within class
    pick += rs[:PER_CLASS]
pools = {r["pool"] for r in pick}
json.dump(sorted(pools), open(os.path.join(bo.DATA, "subset_pools.json"), "w"))
print(f"subset: top {len(pools)} pools by fr12", flush=True)

db = os.path.join(bo.DATA, "events_subset.sqlite")
if os.path.exists(db):
    os.remove(db)
panel, meta = bo.build(db=db, pool_filter=pools)
bo.write_artifacts(panel, meta, prefix="subset_")
print(f"SUBSET BUILD DONE pools {len(pools)} rows {len(panel)}", flush=True)
