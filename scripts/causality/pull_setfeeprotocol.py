"""Paper C data task 1: treated-pool list + activation blocks.

Scans all Uniswap v3 SetFeeProtocol events (topic0-only eth_getLogs) over the
UNIfication window, decodes per-pool protocol-fee activations, fetches each treated
pool's fee tier, and binary-searches the v2 factory feeTo change block.
"""
import json, os, sys, time, csv, urllib.request
from Crypto.Hash import keccak

ROOT = "/Users/joseph/amm-lab"
def load_rpc_url():
    for line in open(os.path.join(ROOT, ".env")):
        if line.strip().startswith("ALCHEMY_ETHEREUM_URL"):
            return line.strip().split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("REFUSING: ALCHEMY_ETHEREUM_URL missing")
RPC = load_rpc_url()

def rpc(method, params, retries=6):
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    data = json.dumps(payload).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(RPC, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                out = json.loads(r.read())
            if "error" in out:
                raise RuntimeError(out["error"])
            return out["result"]
        except Exception as e:
            if attempt == retries - 1: raise
            time.sleep(1.5 * (attempt + 1))

def block_ts(n):
    b = rpc("eth_getBlockByNumber", [hex(n), False])
    return int(b["timestamp"], 16)

def block_at_time(target_ts, lo, hi):
    while lo < hi:
        mid = (lo + hi) // 2
        if block_ts(mid) < target_ts: lo = mid + 1
        else: hi = mid
    return lo

LATEST = int(rpc("eth_blockNumber", []), 16)
print("latest block", LATEST)
# window: 2025-11-01 .. 2026-02-15 UTC
import calendar
t0 = calendar.timegm((2025, 11, 1, 0, 0, 0))
t1 = calendar.timegm((2026, 2, 15, 0, 0, 0))
B_LO = block_at_time(t0, 18_000_000, LATEST)
B_HI = block_at_time(t1, B_LO, LATEST)
print("scan window blocks", B_LO, "->", B_HI, f"({B_HI-B_LO} blocks)")

k = keccak.new(digest_bits=256); k.update(b"SetFeeProtocol(uint8,uint8,uint8,uint8)")
TOPIC = "0x" + k.hexdigest()
CHUNK = 5000
logs = []
n_calls = 0
frm = B_LO
while frm <= B_HI:
    to = min(frm + CHUNK - 1, B_HI)
    try:
        res = rpc("eth_getLogs", [{"fromBlock": hex(frm), "toBlock": hex(to), "topics": [TOPIC]}])
    except RuntimeError as e:
        # range too large fallback
        res = []
        for f2 in range(frm, to + 1, 1000):
            res += rpc("eth_getLogs", [{"fromBlock": hex(f2), "toBlock": hex(min(f2+999, to)), "topics": [TOPIC]}])
    logs += res
    n_calls += 1
    if n_calls % 10 == 0:
        print(f"  chunk {n_calls}: block {to} events so far {len(logs)}", flush=True)
    frm = to + 1
print("getLogs chunks:", n_calls, "raw events:", len(logs))

rows = []
for lg in logs:
    d = lg["data"][2:]
    vals = [int(d[i*64:(i+1)*64], 16) for i in range(4)]
    rows.append({"pool": lg["address"].lower(), "block": int(lg["blockNumber"], 16),
                 "tx": lg["transactionHash"], "old0": vals[0], "old1": vals[1],
                 "new0": vals[2], "new1": vals[3]})
pools = sorted({r["pool"] for r in rows})
print("unique pools with SetFeeProtocol:", len(pools))

# fee tier + timestamps
FEE_SEL = "0xddca3f43"
tiers = {}
for p in pools:
    try:
        tiers[p] = int(rpc("eth_call", [{"to": p, "data": FEE_SEL}, "latest"]), 16)
    except Exception:
        tiers[p] = None
blocks = sorted({r["block"] for r in rows})
ts_map = {b: block_ts(b) for b in blocks}
for r in rows:
    r["fee_tier"] = tiers.get(r["pool"])
    r["timestamp"] = ts_map[r["block"]]
    r["utc"] = time.strftime("%Y-%m-%d %H:%M", time.gmtime(r["timestamp"]))

out = os.path.join(ROOT, ".local/amm_paper_c/data/setfeeprotocol_events.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["pool","fee_tier","block","timestamp","utc","old0","old1","new0","new1","tx"])
    w.writeheader(); w.writerows(sorted(rows, key=lambda r: (r["block"], r["pool"])))
print("wrote", out)

# our two pools check
for name, addr in [("USDC-WETH 5bp", "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"),
                   ("USDC-WETH 30bp", "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8")]:
    hit = [r for r in rows if r["pool"] == addr]
    print(name, "->", [(r["utc"], r["new0"], r["new1"]) for r in hit] or "NOT in window")

# v2 factory feeTo binary search
V2F = "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f"
FEETO_SEL = "0x017e7e58"
def feeTo_at(b):
    return rpc("eth_call", [{"to": V2F, "data": FEETO_SEL}, hex(b)])
lo_v, hi_v = feeTo_at(B_LO), feeTo_at(B_HI)
print("v2 feeTo at window start:", lo_v[-8:], "end:", hi_v[-8:])
v2_block = None
if lo_v != hi_v:
    lo, hi = B_LO, B_HI
    while lo < hi:
        mid = (lo + hi) // 2
        if feeTo_at(mid) == lo_v: lo = mid + 1
        else: hi = mid
    v2_block = lo
    print("v2 feeTo change block:", v2_block, time.strftime("%Y-%m-%d %H:%M", time.gmtime(block_ts(v2_block))), "new feeTo:", feeTo_at(v2_block))

summary = {"scan_window_blocks": [B_LO, B_HI], "n_events": len(rows), "n_pools": len(pools),
           "tier_counts": {}, "v2_feeTo_change_block": v2_block,
           "activation_utc_range": [min(r["utc"] for r in rows), max(r["utc"] for r in rows)] if rows else None}
for r in rows:
    tkey = str(r["fee_tier"])
    summary["tier_counts"][tkey] = summary["tier_counts"].get(tkey, 0) + 1
json.dump(summary, open(os.path.join(ROOT, ".local/amm_paper_c/data/setfeeprotocol_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
