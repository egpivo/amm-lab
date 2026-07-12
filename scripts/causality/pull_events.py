"""Task 2b raw pull v2: full-schema normalized events for the frozen unit set.

Fixes over v1: persists tx_hash / block / transaction_index / log_index / ts /
unit_role / token0 / token1 / removed; separate liquidity_delta (mint/burn) vs
swap_liquidity (in-range L); dedup by (tx_hash, log_index) with resume reload;
integer-exact liquidity. Roles: matched_treated / matched_control /
unmatched_treated / crossvenue_fork. Checkpointed; records failed RPC ranges.
"""
import json, os, sys, time, csv, gzip, calendar, urllib.request
ROOT=os.environ.get("AMMLAB_ROOT","/Users/joseph/amm-lab")
DATA=os.environ.get("AMMLAB_DATA", os.path.join(ROOT,".local/amm_paper_c/data"))
def rpcurl():
    u=os.environ.get("ALCHEMY_ETHEREUM_URL")
    if u: return u.strip().strip('"')
    for l in open(os.path.join(ROOT,".env")):
        if l.startswith("ALCHEMY_ETHEREUM_URL"): return l.split("=",1)[1].strip().strip('"')
RPC=rpcurl()
def rpc(m,p,retries=8):
    d=json.dumps({"jsonrpc":"2.0","id":1,"method":m,"params":p}).encode()
    for a in range(retries):
        try:
            r=urllib.request.Request(RPC,data=d,headers={"Content-Type":"application/json"})
            o=json.loads(urllib.request.urlopen(r,timeout=60).read())
            if "error" in o: raise RuntimeError(str(o["error"])[:120])
            return o["result"]
        except Exception:
            if a==retries-1: raise
            time.sleep(1.5*(a+1))
def blk_ts(n): return int(rpc("eth_getBlockByNumber",[hex(n),False])["timestamp"],16)
def blk_at(ts,lo,hi):
    while lo<hi:
        m=(lo+hi)//2
        if blk_ts(m)<ts: lo=m+1
        else: hi=m
    return lo

units=json.load(open(os.path.join(DATA,"panel_units.json")))
role={}
for p in units["treated_matched"]: role[p]="matched_treated"
for p in units["controls"]: role[p]="matched_control"
for p in units["unmatched_treated"]: role[p]="unmatched_treated"
for p in units["crossvenue_forks"]: role[p]="crossvenue_fork"
tok=json.load(open(os.path.join(DATA,"ckpt_tokens.json")))
POOLS=sorted(role)  # address filter: fetch only our units, not the whole-mainnet v3 firehose
print("units:",len(role),flush=True)

LATEST=int(rpc("eth_blockNumber",[]),16)
B0=blk_at(calendar.timegm((2024,1,1,0,0,0)),15_000_000,LATEST)
B1=blk_at(calendar.timegm((2026,6,30,0,0,0)),B0,LATEST)
print("window",B0,"->",B1,flush=True)
SWAP="0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"
MINT="0x7a53080ba414158be7ec69b987b5fb7d07dee101fe85488f0853ae16239d0bde"
BURN="0x0c396cd989a39f4459b5fa1aed6a9a8dcdbc45908acfd67e028cd568da98982c"
COLL="0x70935338e69775456a85ddef226c395fb668b63fa0115f5f20610b388e6ca9c0"
def i256(h):
    v=int(h,16); return v-2**256 if v>=2**255 else v
def i24(topic):
    v=int(topic,16)&((1<<24)-1); return v-2**24 if v>=2**23 else v

shard=os.path.join(DATA,"events","events.csv.gz"); st_path=os.path.join(DATA,"ckpt_events.json")
HDR=["pool","unit_role","tx_hash","block","tx_index","log_index","ts","type","owner",
     "tickLower","tickUpper","liquidity_delta","swap_liquidity","amount0","amount1",
     "sqrtP","tick","token0","token1","removed"]
# resume: reload dedup keys
seen=set()
if os.path.exists(shard) and os.path.exists(st_path):
    for r in csv.DictReader(gzip.open(shard,"rt")):
        seen.add((r["tx_hash"],r["log_index"]))
    state=json.load(open(st_path)); mode="at"
else:
    state={"frm":B0,"chunks":0,"nrows":0,"failed":[]}; mode="wt"
frm=state["frm"]; step=1000; STEP_MAX=2000
out=gzip.open(shard,mode,newline=""); w=csv.writer(out)
if mode=="wt": w.writerow(HDR)
tscache={}
def wts(bn):
    b=bn//120*120
    if b not in tscache: tscache[b]=blk_ts(b)
    return tscache[b]
while frm<=B1:
    to=min(frm+step-1,B1)
    try:
        logs=rpc("eth_getLogs",[{"fromBlock":hex(frm),"toBlock":hex(to),"address":POOLS,"topics":[[SWAP,MINT,BURN,COLL]]}])
    except Exception:
        # any failure (RPC error, or a transient network timeout that survived retries)
        # degrades to a range halve; only a persistent failure at the floor is recorded
        # and skipped, so a network blip never crashes the whole pull.
        if step<=50: state["failed"].append([frm,to]); json.dump(state,open(st_path,"w")); frm=to+1; continue
        step=max(50,step//2); continue
    for lg in logs:
        p=lg["address"].lower()
        if p not in role: continue
        k=(lg["transactionHash"],str(int(lg["logIndex"],16)))
        if k in seen: continue
        seen.add(k)
        t0=lg["topics"][0]; bn=int(lg["blockNumber"],16); ts=wts(bn); d=lg["data"][2:]
        txi=int(lg["transactionIndex"],16); li=int(lg["logIndex"],16); rm=lg.get("removed",False)
        t=tok.get(p,[None,None])
        base=[p,role[p],lg["transactionHash"],bn,txi,li,ts]
        if t0==SWAP:
            w.writerow(base+["swap","","","","",int(d[192:256],16),i256(d[0:64]),i256(d[64:128]),
                             int(d[128:192],16),i24(lg["data"][2+256:2+320]),t[0],t[1],int(rm)])
        elif t0==MINT:
            w.writerow(base+["mint","0x"+lg["topics"][1][-40:],i24(lg["topics"][2]),i24(lg["topics"][3]),
                             int(d[64:128],16),"",int(d[128:192],16),int(d[192:256],16),"","",t[0],t[1],int(rm)])
        elif t0==BURN:
            w.writerow(base+["burn","0x"+lg["topics"][1][-40:],i24(lg["topics"][2]),i24(lg["topics"][3]),
                             int(d[0:64],16),"",int(d[64:128],16),int(d[128:192],16),"","",t[0],t[1],int(rm)])
        elif t0==COLL:
            w.writerow(base+["collect","0x"+lg["topics"][1][-40:],i24(lg["topics"][2]),i24(lg["topics"][3]),
                             0,"",int(d[64:128],16),int(d[128:192],16),"","",t[0],t[1],int(rm)])
        state["nrows"]+=1
    frm=to+1; state["chunks"]+=1; state["frm"]=frm
    if state["chunks"]%100==0:
        out.flush(); json.dump(state,open(st_path,"w"))
        print(f"  chunk {state['chunks']} block {frm} rows {state['nrows']} failed {len(state['failed'])}",flush=True)
    if step<STEP_MAX: step=min(STEP_MAX,step+200)
out.close(); json.dump(state,open(st_path,"w"))
print(f"RAW EVENTS DONE rows {state['nrows']} failed_ranges {len(state['failed'])}",flush=True)
