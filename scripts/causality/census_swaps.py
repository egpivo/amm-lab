"""Paper C task 2a: one-week all-pools v3-architecture swap activity census.

Topic-only Swap scan over a pre-proposal reference week (2025-10-01..10-08 UTC).
Counts swap events per emitting pool across ALL v3-architecture venues (canonical
Uniswap v3 + forks sharing the event signature). Output: activity ranking joined with
the treated flag from task 1.
"""
import json, os, sys, time, csv, calendar, urllib.request
from collections import Counter

ROOT = "/Users/joseph/amm-lab"
def load_rpc_url():
    for line in open(os.path.join(ROOT, ".env")):
        if line.strip().startswith("ALCHEMY_ETHEREUM_URL"):
            return line.strip().split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("no rpc")
RPC = load_rpc_url()

def rpc(method, params, retries=5):
    data = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request(RPC, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                out = json.loads(r.read())
            if "error" in out:
                raise RuntimeError(str(out["error"])[:200])
            return out["result"]
        except Exception:
            if a == retries - 1: raise
            time.sleep(1.2 * (a + 1))

def block_ts(n): return int(rpc("eth_getBlockByNumber", [hex(n), False])["timestamp"], 16)
def block_at(ts, lo, hi):
    while lo < hi:
        mid = (lo + hi) // 2
        if block_ts(mid) < ts: lo = mid + 1
        else: hi = mid
    return lo

LATEST = int(rpc("eth_blockNumber", []), 16)
B0 = block_at(calendar.timegm((2025, 10, 1, 0, 0, 0)), 18_000_000, LATEST)
B1 = block_at(calendar.timegm((2025, 10, 8, 0, 0, 0)), B0, LATEST)
print("census window", B0, "->", B1, f"({B1-B0} blocks)", flush=True)

TOPIC = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"
counts = Counter()
def scan(frm, to):
    try:
        logs = rpc("eth_getLogs", [{"fromBlock": hex(frm), "toBlock": hex(to), "topics": [TOPIC]}])
        for lg in logs: counts[lg["address"].lower()] += 1
        return True
    except RuntimeError:
        return False

frm, step, done = B0, 400, 0
while frm <= B1:
    to = min(frm + step - 1, B1)
    if scan(frm, to):
        frm = to + 1; done += 1
        if step < 400: step = min(400, step * 2)
    else:
        step = max(50, step // 2)          # adaptive: response too large
    if done % 25 == 0 and done:
        print(f"  {done} chunks, at block {frm}, pools {len(counts)}, swaps {sum(counts.values())}", flush=True)
        done += 1  # avoid repeat print
print("census done: pools", len(counts), "swaps", sum(counts.values()), flush=True)

treated = set()
for r in csv.DictReader(open(os.path.join(ROOT, ".local/amm_paper_c/data/setfeeprotocol_events.csv"))):
    treated.add(r["pool"])
rows = [{"pool": p, "swaps_week": c, "treated": int(p in treated)} for p, c in counts.most_common()]
out = os.path.join(ROOT, ".local/amm_paper_c/data/swap_census_2025-10-01_wk.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["pool", "swaps_week", "treated"]); w.writeheader(); w.writerows(rows)

tr = [r for r in rows if r["treated"]]; un = [r for r in rows if not r["treated"]]
summary = {"window_blocks": [B0, B1], "pools_active": len(rows), "total_swaps": sum(counts.values()),
           "treated_active": len(tr), "untreated_active": len(un),
           "treated_swap_share": round(sum(r["swaps_week"] for r in tr) / max(1, sum(counts.values())), 4),
           "top_untreated": un[:15]}
json.dump(summary, open(os.path.join(ROOT, ".local/amm_paper_c/data/swap_census_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1), flush=True)
