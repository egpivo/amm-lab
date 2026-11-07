"""Pure-arithmetic audit of derived RNG seeds over every registered seed block.

Reads the (multiplier, offset) pairs from the Rust sources, derives each stream seed
(seed*K + c, wrapping u64; the oracle stream uses the seed itself) for every seed in
every block, and reports any derived value reached from two different (seed, stream)
pairs. Runs no simulator. Exit status 1 on any collision.
"""

import re
import sys
from pathlib import Path

import numpy as np

from seed_blocks import BLOCKS, assert_registry_disjoint

ROOT = Path(__file__).resolve().parents[2]


def stream_constants() -> dict[str, tuple[int, int]]:
    env = (ROOT / "src/sim/env.rs").read_text()
    streams = {"oracle": (1, 0)}
    for name in ("noise", "arb", "order", "jit"):
        m = re.search(
            rf"rng_{name} = StdRng::seed_from_u64\(seed\.wrapping_mul\((\d+)\)\.wrapping_add\((\d+)\)\)",
            env,
        )
        assert m, f"stream {name} not found in env.rs"
        streams[name] = (int(m[1]), int(m[2]))
    orc = (ROOT / "src/sim/oracle.rs").read_text()
    for name, const in (("xi", "XI_STREAM"), ("eta", "ETA_STREAM")):
        m = re.search(rf"pub const {const}: \(u64, u64\) = \(([\d_]+), (\d+)\);", orc)
        assert m, f"{const} not found in oracle.rs"
        streams[name] = (int(m[1].replace("_", "")), int(m[2]))
    return streams


def audit() -> dict:
    assert_registry_disjoint()
    streams = stream_constants()
    seeds = np.concatenate([np.arange(b.start, b.end, dtype=np.uint64) for b in BLOCKS])
    vals, owner_seed, owner_stream = [], [], []
    for idx, (k, c) in enumerate(streams.values()):
        with np.errstate(over="ignore"):
            vals.append(seeds * np.uint64(k) + np.uint64(c))  # wraps mod 2**64
        owner_seed.append(seeds)
        owner_stream.append(np.full(seeds.shape, idx, dtype=np.uint8))
    v, s, st = map(np.concatenate, (vals, owner_seed, owner_stream))
    order = np.argsort(v, kind="stable")
    v, s, st = v[order], s[order], st[order]
    dup = np.nonzero(v[1:] == v[:-1])[0]
    collisions = [
        (int(v[i]), (int(s[i]), int(st[i])), (int(s[i + 1]), int(st[i + 1]))) for i in dup
    ]
    return {"streams": streams, "derived": int(v.size), "collisions": collisions}


if __name__ == "__main__":
    result = audit()
    print(f"streams={list(result['streams'])} derived={result['derived']} collisions={len(result['collisions'])}")
    for c in result["collisions"][:10]:
        print("COLLISION", c)
    sys.exit(1 if result["collisions"] else 0)
