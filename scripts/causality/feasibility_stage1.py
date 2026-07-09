"""Task 2b stage 1: match-feasibility on pre-period data only.

Phases (checkpointed):
 A. token0/token1 map for all census-active pools;
 B. numeraire identification (USDC/USDT/DAI/WETH/WBTC, symbol-verified) + WETH/WBTC
    USD prices from on-chain slot0 at the reference block (proposal eve);
 C. 12-week swap scan ending 2025-11-09 decoding amounts -> realized fee revenue
    (USD, numeraire leg x tier), with trailing-4-week sub-window;
 D. age flag: active in a 2024-12 reference week (proxy for created < 2025-01-01);
 E. matching per spec: exact tier + pair-class, NN on log(fr_12wk), caliper 0.5,
    exposure flags -> feasibility report. NO post-period data touched.
"""
import json, os, sys, time, csv, calendar, urllib.request
from collections import Counter, defaultdict

ROOT = "/Users/joseph/amm-lab"
DATA = os.path.join(ROOT, ".local/amm_paper_c/data")
def load_rpc():
    for line in open(os.path.join(ROOT, ".env")):
        if line.strip().startswith("ALCHEMY_ETHEREUM_URL"):
            return line.strip().split("=", 1)[1].strip().strip('"')
RPC = load_rpc()
def rpc(method, params, retries=5):
    data = json.dumps({"jsonrpc":"2.0","id":1,"method":method,"params":params}).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request(RPC, data=data, headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                out = json.loads(r.read())
            if "error" in out: raise RuntimeError(str(out["error"])[:150])
            return out["result"]
        except Exception:
            if a == retries-1: raise
            time.sleep(1.2*(a+1))
def block_ts(n): return int(rpc("eth_getBlockByNumber", [hex(n), False])["timestamp"], 16)
def block_at(ts, lo, hi):
    while lo < hi:
        mid = (lo+hi)//2
        if block_ts(mid) < ts: lo = mid+1
        else: hi = mid
    return lo
def ckpt(name): return os.path.join(DATA, f"ckpt_{name}.json")
def load_ck(name):
    p = ckpt(name)
    return json.load(open(p)) if os.path.exists(p) else None
def save_ck(name, obj): json.dump(obj, open(ckpt(name), "w"))

census = list(csv.DictReader(open(os.path.join(DATA, "swap_census_2025-10-01_wk.csv"))))
pools = [r["pool"] for r in census]
treated = {r["pool"] for r in census if r["treated"] == "1"}
print(f"census pools {len(pools)}, treated {len(treated)}", flush=True)

# ---- Phase A: token map ----
tok = load_ck("tokens") or {}
if len(tok) < len(pools):
    for i, p in enumerate(pools):
        if p in tok: continue
        try:
            t0 = rpc("eth_call", [{"to": p, "data": "0x0dfe1681"}, "latest"])
            t1 = rpc("eth_call", [{"to": p, "data": "0xd21220a7"}, "latest"])
            tok[p] = ["0x"+t0[-40:], "0x"+t1[-40:]]
        except Exception:
            tok[p] = [None, None]
        if i % 300 == 0:
            save_ck("tokens", tok); print(f"  tokens {i}/{len(pools)}", flush=True)
    save_ck("tokens", tok)
print("phase A done", flush=True)

# ---- Phase B: numeraires + prices ----
NUM = {"0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": ("USDC", 6, 1.0),
       "0xdac17f958d2ee523a2206206994597c13d831ec7": ("USDT", 6, 1.0),
       "0x6b175474e89094c44da98b954eedeac495271d0f": ("DAI", 18, 1.0),
       "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": ("WETH", 18, None),
       "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599": ("WBTC", 8, None)}
def symbol(t):
    r = rpc("eth_call", [{"to": t, "data": "0x95d89b41"}, "latest"])
    h = r[2:]
    try:
        if len(h) > 128:
            ln = int(h[64:128], 16); return bytes.fromhex(h[128:128+ln*2]).decode("utf-8","replace")
        return bytes.fromhex(h).rstrip(b"\x00").decode("utf-8","replace")
    except Exception: return "?"
for a, (sym, dec, px) in NUM.items():
    got = symbol(a)
    assert sym.lower() in got.lower() or got.lower() in sym.lower(), (a, sym, got)
LATEST = int(rpc("eth_blockNumber", []), 16)
REF = block_at(calendar.timegm((2025,11,9,0,0,0)), 18_000_000, LATEST)
def spot_price(pool, base_is_token0, dec0, dec1):
    r = rpc("eth_call", [{"to": pool, "data": "0x3850c7bd"}, hex(REF)])  # slot0
    sqrtp = int(r[2:66], 16)
    p_raw = (sqrtp / 2**96) ** 2          # token1 per token0 (raw units)
    p_adj = p_raw * 10**dec0 / 10**dec1   # human units token1/token0
    return p_adj if base_is_token0 else 1.0 / p_adj
# USDC/WETH 5bp: token0 USDC(6), token1 WETH(18) -> WETH per USDC; want USD per WETH
usdc_per_weth = 1.0 / spot_price("0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640", True, 6, 18)
# WBTC/WETH 5bp 0x4585fe77...: token0 WBTC(8), token1 WETH(18) -> WETH per WBTC
weth_per_wbtc = spot_price("0x4585fe77225b41b697c938b018e2ac67ac5a20c0", True, 8, 18)
PX = {"USDC":1.0, "USDT":1.0, "DAI":1.0, "WETH": usdc_per_weth, "WBTC": weth_per_wbtc*usdc_per_weth}
print("prices at proposal eve:", {k: round(v,1) for k,v in PX.items()}, flush=True)

# ---- Phase C: 12-week amount scan ending proposal eve ----
B_END = REF
B_12 = block_at(calendar.timegm((2025,8,17,0,0,0)), 18_000_000, B_END)
B_4  = block_at(calendar.timegm((2025,10,12,0,0,0)), B_12, B_END)
TOPIC = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"
# numeraire slot per pool: prefer stable > WETH > WBTC
def num_slot(p):
    t0, t1 = tok.get(p, [None, None])
    best = None
    for i, t in enumerate([t0, t1]):
        if t in NUM:
            sym = NUM[t][0]
            rank = 0 if sym in ("USDC","USDT","DAI") else (1 if sym == "WETH" else 2)
            if best is None or rank < best[0]: best = (rank, i, t)
    return (best[1], best[2]) if best else (None, None)
NUMSLOT = {p: num_slot(p) for p in pools}
tiers = {}
for r in csv.DictReader(open(os.path.join(DATA, "setfeeprotocol_events.csv"))):
    tiers[r["pool"]] = int(r["fee_tier"])

state = load_ck("scanC") or {"frm": B_12, "fr12": {}, "fr4": {}, "sw12": {}, "chunks": 0}
frm = state["frm"]; fr12 = defaultdict(float, state["fr12"]); fr4 = defaultdict(float, state["fr4"])
sw12 = Counter(state["sw12"]); step = 300
def to_i256(h):
    v = int(h, 16)
    return v - 2**256 if v >= 2**255 else v
while frm <= B_END:
    to = min(frm + step - 1, B_END)
    try:
        logs = rpc("eth_getLogs", [{"fromBlock": hex(frm), "toBlock": hex(to), "topics": [TOPIC]}])
    except RuntimeError:
        step = max(40, step//2); continue
    for lg in logs:
        p = lg["address"].lower()
        slot, t = NUMSLOT.get(p, (None, None))
        sw12[p] += 1
        if slot is None: continue
        d = lg["data"][2:]
        amt = abs(to_i256(d[slot*64:(slot+1)*64]))
        sym, dec, _ = NUM[t]
        usd = amt / 10**dec * PX[sym]
        # fee tier: from setfee CSV (treated) else eth_call cache
        tier = tiers.get(p)
        if tier is None:
            try: tier = int(rpc("eth_call", [{"to": p, "data": "0xddca3f43"}, "latest"]), 16)
            except Exception: tier = 0
            tiers[p] = tier
        fee_usd = usd * tier / 1e6
        fr12[p] += fee_usd
        if int(lg["blockNumber"], 16) >= B_4: fr4[p] += fee_usd
    frm = to + 1
    state = {"frm": frm, "fr12": dict(fr12), "fr4": dict(fr4), "sw12": dict(sw12), "chunks": state["chunks"]+1}
    if state["chunks"] % 100 == 0:
        save_ck("scanC", state); print(f"  scanC chunk {state['chunks']} at block {frm} pools {len(fr12)}", flush=True)
    if step < 300: step = min(300, step+40)
save_ck("scanC", state)
print("phase C done: pools with fee revenue", len(fr12), flush=True)

# ---- Phase D: age flag via 2024-12 activity week ----
age = load_ck("age")
if age is None:
    A0 = block_at(calendar.timegm((2024,12,11,0,0,0)), 18_000_000, B_END)
    A1 = block_at(calendar.timegm((2024,12,18,0,0,0)), A0, B_END)
    act = set(); frm2, step2 = A0, 400
    while frm2 <= A1:
        to2 = min(frm2+step2-1, A1)
        try:
            logs = rpc("eth_getLogs", [{"fromBlock": hex(frm2), "toBlock": hex(to2), "topics": [TOPIC]}])
        except RuntimeError:
            step2 = max(50, step2//2); continue
        for lg in logs: act.add(lg["address"].lower())
        frm2 = to2+1
    age = {"active_2024_12": sorted(act)}
    save_ck("age", age)
old_pools = set(age["active_2024_12"])
print("phase D done: pools active 2024-12:", len(old_pools), flush=True)

# ---- Phase E: classify + match + report ----
STABLE_SYMS = {"USDC","USDT","DAI","USDe","sUSDe","USD1","USDf","PYUSD","RLUSD","FRAX","LUSD","GHO","EURC","TUSD","USDP","crvUSD"}
WETH_ADDR = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
BTC_SYMS = {"WBTC","tBTC","cbBTC"}
symcache = load_ck("symcache") or {}
def tok_sym(t):
    if t in symcache: return symcache[t]
    try: s_ = symbol(t)
    except Exception: s_ = "?"
    symcache[t] = s_; return s_
def pair_class(p):
    t0, t1 = tok.get(p, [None, None])
    if not t0: return "unknown"
    s0, s1 = tok_sym(t0), tok_sym(t1)
    st0, st1 = s0 in STABLE_SYMS, s1 in STABLE_SYMS
    if st0 and st1: return "stable-stable"
    if WETH_ADDR in (t0, t1): return "weth-pair"
    if s0 in BTC_SYMS or s1 in BTC_SYMS: return "btc-pair"
    return "other"
import math
rows = []
for p in pools:
    fr = fr12.get(p, 0.0)
    rows.append({"pool": p, "treated": int(p in treated), "tier": tiers.get(p, 0),
                 "class": pair_class(p), "fr12_usd": round(fr, 2), "fr4_usd": round(fr4.get(p, 0.0), 2),
                 "sw12": sw12.get(p, 0), "old": int(p in old_pools),
                 "covered": int(NUMSLOT.get(p, (None,))[0] is not None)})
save_ck("symcache", symcache)
w = csv.DictWriter(open(os.path.join(DATA, "feerev_panelvars.csv"), "w", newline=""),
                   fieldnames=list(rows[0].keys()))
w.writeheader(); w.writerows(rows)

# exposure: same unordered pair as any treated pool -> spillover; shared major token -> exposed
treated_pairs = set(); treated_tok = set()
for p in treated:
    t = tok.get(p, [None, None])
    if t[0]: treated_pairs.add(frozenset(t)); treated_tok.update(t)
def exposure(p):
    # spec: exposure = w_pair * w_sub. same pair = 1.0 (spillover unit);
    # shared token only, non-same-pair = routes only, w_pair=0.5 x w_sub=0.25 = 0.125 (pure);
    # no shared token = 0.0.
    t = tok.get(p, [None, None])
    if not t[0]: return 1.0
    if frozenset(t) in treated_pairs: return 1.0
    if t[0] in treated_tok or t[1] in treated_tok: return 0.125
    return 0.0
# matching: main sample old==1, covered==1, fr12>0
cands = defaultdict(list)
for r in rows:
    if not r["treated"] and r["old"] and r["covered"] and r["fr12_usd"] > 0 and exposure(r["pool"]) <= 0.25:
        cands[(r["tier"], r["class"])].append(r)
matched, unmatched = [], []
for r in rows:
    if not r["treated"] or not r["old"] or not r["covered"] or r["fr12_usd"] <= 0: continue
    pool_c = cands.get((r["tier"], r["class"]), [])
    lg = math.log(r["fr12_usd"])
    near = sorted(pool_c, key=lambda c: abs(math.log(max(c["fr12_usd"], 1e-9)) - lg))[:3]
    near = [c for c in near if abs(math.log(max(c["fr12_usd"], 1e-9)) - lg) <= 0.5]
    if near: matched.append({"treated": r["pool"], "fr12": r["fr12_usd"], "tier": r["tier"],
                             "class": r["class"], "controls": [c["pool"] for c in near]})
    else: unmatched.append({"pool": r["pool"], "fr12": r["fr12_usd"], "tier": r["tier"], "class": r["class"]})
unmatched.sort(key=lambda x: -x["fr12"])
by_tc = Counter((m["tier"], m["class"]) for m in matched)
n_treated_main = sum(1 for r in rows if r["treated"] and r["old"] and r["covered"] and r["fr12_usd"] > 0)
report = {"reference_block_proposal_eve": REF,
          "prices": {k: round(v, 2) for k, v in PX.items()},
          "n_treated_census_active": len([r for r in rows if r["treated"]]),
          "n_treated_main_sample": n_treated_main,
          "n_matched_treated": len(matched),
          "match_rate_main": round(len(matched)/max(1, n_treated_main), 3),
          "match_by_tier_class": {f"{k[0]}|{k[1]}": v for k, v in sorted(by_tc.items())},
          "n_pure_controls_used": len({c for m in matched for c in m["controls"]}),
          "n_candidate_pure_controls": sum(len(v) for v in cands.values()),
          "top_unmatched_by_feerev": unmatched[:15]}
json.dump(report, open(os.path.join(DATA, "match_feasibility.json"), "w"), indent=1)
json.dump(matched, open(os.path.join(DATA, "matched_pairs.json"), "w"), indent=1)
print(json.dumps(report, indent=1), flush=True)
print("STAGE 1 COMPLETE", flush=True)
