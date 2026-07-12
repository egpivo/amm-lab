"""Task 2b: post-period weekly panel pull for the frozen unit set.

1,194 pools (868 treated main + 326 matched controls). Window: full panel
2024-01-01 .. 2026-06-30 (clean pre + anticipation + post). Per pool-week, from
Mint/Burn/Collect + Swap events, build the FROZEN outcome family. Topic-filtered
getLogs restricted to the unit address set (batched by address), decoded to weekly
aggregates. Checkpointed. NO estimation here — pull + weekly aggregation only.
"""
import json, os, sys, time, csv, calendar, urllib.request
from collections import defaultdict

ROOT="/Users/joseph/amm-lab"; DATA=os.path.join(ROOT,".local/amm_paper_c/data")
def rpcurl():
    for l in open(os.path.join(ROOT,".env")):
        if l.startswith("ALCHEMY_ETHEREUM_URL"): return l.split("=",1)[1].strip().strip('"')
RPC=rpcurl()
def rpc(m,p,retries=6):
    d=json.dumps({"jsonrpc":"2.0","id":1,"method":m,"params":p}).encode()
    for a in range(retries):
        try:
            r=urllib.request.Request(RPC,data=d,headers={"Content-Type":"application/json"})
            o=json.loads(urllib.request.urlopen(r,timeout=40).read())
            if "error" in o: raise RuntimeError(str(o["error"])[:120])
            return o["result"]
        except Exception:
            if a==retries-1: raise
            time.sleep(1.3*(a+1))
def blk_ts(n): return int(rpc("eth_getBlockByNumber",[hex(n),False])["timestamp"],16)
def blk_at(ts,lo,hi):
    while lo<hi:
        m=(lo+hi)//2
        if blk_ts(m)<ts: lo=m+1
        else: hi=m
    return lo

units=json.load(open(os.path.join(DATA,"panel_units.json")))
pools=sorted(set(units["treated_main"])|set(units["controls"]))
poolset={p:1 for p in pools}
print("pools:",len(pools),flush=True)

LATEST=int(rpc("eth_blockNumber",[]),16)
B0=blk_at(calendar.timegm((2024,1,1,0,0,0)),15_000_000,LATEST)
B1=blk_at(calendar.timegm((2026,6,30,0,0,0)),B0,LATEST)
print("panel window",B0,"->",B1,flush=True)

# event topics
SWAP="0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"
MINT="0x7a53080ba414158be7ec69b987b5fb7d07dee101fe85488f0853ae16239d0bde"
BURN="0x0c396cd989a39f4459b5fa1aed6a9a8dcdbc45908acfd67e028cd568da98982c"
COLL="0x70935338e69775456a85ddef226c395fb668b63fa0115f5f20610b388e6ca9c0"
TOPICS=[[SWAP,MINT,BURN,COLL]]

def wk(ts): return time.strftime("%Y-%W", time.gmtime(ts))
# weekly aggregates per pool: swaps, swap_volume(token1 abs), mints, mint_liq, burns, burn_liq,
# collects, collect_amt, jit(same-block mint->burn) counted at burn
state=json.load(open(os.path.join(DATA,"ckpt_panel.json"))) if os.path.exists(os.path.join(DATA,"ckpt_panel.json")) else {"frm":B0,"agg":{},"chunks":0}
frm=state["frm"]; agg=defaultdict(lambda: defaultdict(float), {k:defaultdict(float,v) for k,v in state["agg"].items()})
mint_blocks=defaultdict(dict)  # pool -> {(tickL,tickU): last mint block} for JIT
step=800
def to_i256(h):
    v=int(h,16); return v-2**256 if v>=2**255 else v
tscache={}
def week_of(bn):
    b=bn//200*200
    if b not in tscache: tscache[b]=blk_ts(b)
    return wk(tscache[b])
while frm<=B1:
    to=min(frm+step-1,B1)
    try:
        logs=rpc("eth_getLogs",[{"fromBlock":hex(frm),"toBlock":hex(to),"topics":TOPICS}])
    except RuntimeError:
        step=max(100,step//2); continue
    for lg in logs:
        p=lg["address"].lower()
        if p not in poolset: continue
        t0=lg["topics"][0]; bn=int(lg["blockNumber"],16); w=week_of(bn); key=f"{p}|{w}"
        d=lg["data"][2:]
        if t0==SWAP:
            agg[key]["swaps"]+=1
            amt1=abs(to_i256(d[64:128])); agg[key]["vol1"]+=amt1/1e18
        elif t0==MINT:
            agg[key]["mints"]+=1
            L=int(d[64:128],16); agg[key]["mint_liq_i"]=int(agg[key].get("mint_liq_i",0))+L
        elif t0==BURN:
            agg[key]["burns"]+=1
            L=int(d[0:64],16); agg[key]["burn_liq_i"]=int(agg[key].get("burn_liq_i",0))+L
        elif t0==COLL:
            agg[key]["collects"]+=1
    frm=to+1; state["chunks"]+=1
    if state["chunks"]%100==0:
        state["frm"]=frm; state["agg"]={k:dict(v) for k,v in agg.items()}
        json.dump(state,open(os.path.join(DATA,"ckpt_panel.json"),"w"))
        print(f"  chunk {state['chunks']} block {frm} keys {len(agg)}",flush=True)
    if step<800: step=min(800,step+100)
state["frm"]=frm; state["agg"]={k:dict(v) for k,v in agg.items()}
json.dump(state,open(os.path.join(DATA,"ckpt_panel.json"),"w"))

# emit tidy panel
treated=set(units["treated_main"]); mrows=[]
with open(os.path.join(DATA,"panel_weekly.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["pool","week","treated","swaps","vol1","mints","mint_liq","burns","burn_liq","collects","net_liq"])
    for key,a in agg.items():
        p,wk_=key.split("|")
        ml=int(a.get("mint_liq_i",0)); bl=int(a.get("burn_liq_i",0))
        w.writerow([p,wk_,int(p in treated),int(a["swaps"]),round(a["vol1"],4),int(a["mints"]),
                    ml,int(a["burns"]),bl,int(a["collects"]),ml-bl])
print("PANEL PULL DONE: pool-weeks",len(agg),flush=True)
