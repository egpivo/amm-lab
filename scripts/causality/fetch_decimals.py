#!/usr/bin/env python3
"""Standalone token-decimals fetcher, mirroring build_outcomes.py's load_decimals exactly
(selector 0x313ce567, int(res,16), default 18 on empty/failure) so the emitted
token_decimals.json is identical to what the full build would produce. This lets the Rust
reconstruction get correct vol/fee scaling without waiting for the slow SQLite build.

Token set = distinct token0/token1 across ckpt_tokens.json (pool -> [token0, token1]),
which is exactly the token set present in events (each event's tokens are its pool's tokens).

Usage: AMMLAB_ROOT/AMMLAB_DATA + ALCHEMY_ETHEREUM_URL (env or ROOT/.env), then run.
"""
import json, os, time, urllib.request

ROOT = os.environ.get("AMMLAB_ROOT", "/Users/joseph/amm-lab")
DATA = os.environ.get("AMMLAB_DATA", os.path.join(ROOT, ".local/amm_paper_c/data"))
DECPATH = os.path.join(DATA, "token_decimals.json")
TOKENS = os.path.join(DATA, "ckpt_tokens.json")


def rpcurl():
    u = os.environ.get("ALCHEMY_ETHEREUM_URL")
    if u:
        return u.strip().strip('"')
    for l in open(os.path.join(ROOT, ".env")):
        if l.startswith("ALCHEMY_ETHEREUM_URL"):
            return l.split("=", 1)[1].strip().strip('"')
    raise SystemExit("no ALCHEMY_ETHEREUM_URL in env or .env")


RPC = rpcurl()


def rpc(m, p, retries=5):
    d = json.dumps({"jsonrpc": "2.0", "id": 1, "method": m, "params": p}).encode()
    for a in range(retries):
        try:
            r = urllib.request.Request(RPC, data=d, headers={"Content-Type": "application/json"})
            o = json.loads(urllib.request.urlopen(r, timeout=45).read())
            if "error" in o:
                raise RuntimeError(str(o["error"])[:120])
            return o["result"]
        except Exception:
            if a == retries - 1:
                raise
            time.sleep(1.0 * (a + 1))


def main():
    toks = set()
    ck = json.load(open(TOKENS))
    for pool, pair in ck.items():
        for t in pair:
            if t:
                toks.add(t)
    toks = sorted(toks)
    print(f"{len(toks)} distinct tokens from ckpt_tokens.json", flush=True)

    dec = json.load(open(DECPATH)) if os.path.exists(DECPATH) else {}
    dec_default = []
    miss = [t for t in toks if t and t not in dec]
    print(f"{len(miss)} missing -> fetching decimals()", flush=True)
    for i, t in enumerate(miss):
        try:
            res = rpc("eth_call", [{"to": t, "data": "0x313ce567"}, "latest"])  # decimals()
            dec[t] = int(res, 16) if res and res != "0x" else 18
            if not res or res == "0x":
                dec_default.append(t)
        except Exception:
            dec[t] = 18
            dec_default.append(t)
        if i % 100 == 0:
            json.dump(dec, open(DECPATH, "w"))
            print(f"  decimals {i}/{len(miss)}", flush=True)
    json.dump(dec, open(DECPATH, "w"))
    print(f"DECIMALS DONE {len(dec)} tokens; defaulted-to-18 {len(dec_default)}", flush=True)


if __name__ == "__main__":
    main()
