"""BO-2: initialize each pool's tick-liquidity book from creation to panel start.

Depth within price bands cannot be computed from an empty 2024-01-01 tick book. This
pulls Mint/Burn from before the panel window per pool, accumulates liquidityNet per
tick, and emits tickbook_init.json (pool -> {tick: net_liquidity}) plus a
state_initialization_report.json flagging pools whose init is incomplete (failed ranges
or no pre-window events). Runs independently of the panel pull; checkpointed by pool.
"""
import json, os, time, csv, calendar, urllib.request, urllib.error
from collections import defaultdict
ROOT=os.environ.get("AMMLAB_ROOT","/Users/joseph/amm-lab")
DATA=os.environ.get("AMMLAB_DATA", os.path.join(ROOT,".local/amm_paper_c/data"))
def rpcurl():
    u=os.environ.get("ALCHEMY_ETHEREUM_URL")
    if u: return u.strip().strip('"')
    for l in open(os.path.join(ROOT,".env")):
        if l.startswith("ALCHEMY_ETHEREUM_URL"): return l.split("=",1)[1].strip().strip('"')
RPC=rpcurl()
class RangeTooLarge(Exception): pass
def rpc(m,p,retries=6):
    d=json.dumps({"jsonrpc":"2.0","id":1,"method":m,"params":p}).encode()
    for a in range(retries):
        try:
            r=urllib.request.Request(RPC,data=d,headers={"Content-Type":"application/json"})
            o=json.loads(urllib.request.urlopen(r,timeout=45).read())
            if "error" in o:
                msg=str(o["error"]).lower()
                if any(s in msg for s in ("more than","exceed","too large","range","limit","response size")):
                    raise RangeTooLarge(str(o["error"])[:120])
                raise RuntimeError(str(o["error"])[:120])
            return o["result"]
        except urllib.error.HTTPError as he:
            # Alchemy surfaces "range too large / too many results" as HTTP 400 (deterministic
            # for the same params): do not retry -- signal the caller to halve the range.
            if he.code==400: raise RangeTooLarge(f"HTTP400 {m}")
            if a==retries-1: raise
            time.sleep(1.2*(a+1))
        except (RangeTooLarge,):
            raise
        except Exception:
            if a==retries-1: raise
            time.sleep(1.2*(a+1))
def blk_at(ts,lo,hi):
    def bts(n): return int(rpc("eth_getBlockByNumber",[hex(n),False])["timestamp"],16)
    while lo<hi:
        m=(lo+hi)//2
        if bts(m)<ts: lo=m+1
        else: hi=m
    return lo
units=json.load(open(os.path.join(DATA,"panel_units.json")))
pools=sorted(set(units["treated_main"])|set(units["controls"])|set(units["crossvenue_forks"]))
LATEST=int(rpc("eth_blockNumber",[]),16)
PANEL_START=blk_at(calendar.timegm((2024,1,1,0,0,0)),15_000_000,LATEST)
V3_DEPLOY=12_369_621   # Uniswap v3 factory deployment
MINT="0x7a53080ba414158be7ec69b987b5fb7d07dee101fe85488f0853ae16239d0bde"
BURN="0x0c396cd989a39f4459b5fa1aed6a9a8dcdbc45908acfd67e028cd568da98982c"
def i24(topic):
    v=int(topic,16)&((1<<24)-1); return v-2**24 if v>=2**23 else v
book=json.load(open(os.path.join(DATA,"ckpt_tickbook.json"))) if os.path.exists(os.path.join(DATA,"ckpt_tickbook.json")) else {}
report=json.load(open(os.path.join(DATA,"ckpt_initreport.json"))) if os.path.exists(os.path.join(DATA,"ckpt_initreport.json")) else {}
for idx,p in enumerate(pools):
    if p in book: continue
    net=defaultdict(int); failed=0; nev=0; frm=V3_DEPLOY; step=50000
    while frm<PANEL_START:
        to=min(frm+step-1,PANEL_START-1)
        try:
            logs=rpc("eth_getLogs",[{"address":p,"fromBlock":hex(frm),"toBlock":hex(to),"topics":[[MINT,BURN]]}])
        except (RangeTooLarge,RuntimeError,Exception):
            if step<=2000: failed+=1; frm=to+1; continue
            step=max(2000,step//2); continue
        for lg in logs:
            d=lg["data"][2:]; sgn=1 if lg["topics"][0]==MINT else -1
            tl=i24(lg["topics"][2]); tu=i24(lg["topics"][3])
            L=int(d[64:128],16) if sgn==1 else int(d[0:64],16)
            net[str(tl)]+=sgn*L; net[str(tu)]-=sgn*L; nev+=1
        frm=to+1
        if step<200000: step=min(200000,step*2)
    book[p]={k:v for k,v in net.items() if v!=0}
    report[p]={"pre_window_events":nev,"failed_ranges":failed,"complete":failed==0}
    if idx%50==0:
        json.dump(book,open(os.path.join(DATA,"ckpt_tickbook.json"),"w"))
        json.dump(report,open(os.path.join(DATA,"ckpt_initreport.json"),"w"))
        print(f"  init {idx}/{len(pools)} {p[:10]} events {nev} failed {failed}",flush=True)
json.dump(book,open(os.path.join(DATA,"tickbook_init.json"),"w"))
inc=[p for p,r in report.items() if not r["complete"]]
json.dump({"pools":len(book),"incomplete_init":inc,"n_incomplete":len(inc),"per_pool":report},
          open(os.path.join(DATA,"state_initialization_report.json"),"w"),indent=1)
print(f"TICKBOOK INIT DONE pools {len(book)} incomplete {len(inc)}",flush=True)
