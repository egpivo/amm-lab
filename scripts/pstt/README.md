# PSTT standalone WETH-USDC replication (M6 application)

## Authority and roles

- **Frozen Python is the historical evidence authority.** Everything under
  `.local/selection_paper/` (in particular `mc_m5r.py`, `mc_m5s.py`,
  `build_m6_public.py`, and the frozen outputs `m6_public_weekly.json` /
  `m6_public_manifest.json`) is the record of what the paper reports. It is
  never modified, moved, regenerated, or rerun by anything in this repo.
- **Rust (`src/pstt/`, `src/bin/pstt_*`) is additive parity/reconstruction
  tooling only.** It re-implements the frozen deterministic kernels with
  regression tests against a stored NumPy oracle fixture. It does not change
  the frozen M6 verdict (UNDETERMINED, reference-robust, not
  positivity-limited) and computes no new empirical results.

## Bootstrap provenance limit

The frozen stage-2 runner derived NumPy seeds through Python's salted
built-in `hash(...)` and the manifest records no `PYTHONHASHSEED`. The
historical bootstrap draw stream is therefore unrecoverable, and **no tool
here claims bitwise bootstrap parity**. `pstt_build_application` supports:

- `--index-file`: exact replay of an externally serialized index schedule;
- `--seed`: a new deterministic Rust draw schedule (rand `StdRng`,
  explicitly not a NumPy PCG64 stream).

Everything upstream of the bootstrap (fills, joins, marks, weekly
primitives, envelopes, Fieller projection, screening, classification) is
deterministic and parity-tested.

## Required public inputs

| Input | Description |
|---|---|
| WETH-USDC swap tape | CSV(.gz) with `type,block,pool,token0,token1,amount0,amount1`; target pools `0x88e6...5640` (5bp) and `0x8ad5...e6d8` (30bp), canonical USDC/WETH token order |
| Block timestamps | Ethereum headers for every swap block (fetch + verify below) |
| Binance ETHUSDC aggTrades | Daily `ETHUSDC-aggTrades-YYYY-MM-DD.{zip,csv,csv.gz}` archives, 2024-01-01 through 2025-12-27 (plus one prior day for as-of joins) |

Reference marks are public trade-price proxies (last trade PRIMARY, 1s VWAP
ROBUSTNESS), never bid-ask midpoints.

## Standalone execution order

All tools refuse a nonempty output directory unless
`--allow-nonempty-output` is passed. Only step 2 uses the network.

1. **Extract eligible blocks** from the swap tape:

   ```sh
   cargo run --release --bin pstt_extract_target_blocks -- \
     --events swaps.csv.gz --pools-json pools.json \
     --start-unix 1704067200 --end-unix 1766793600 --output-dir out/blocks
   ```

2. **Fetch block headers** (JSON-RPC, resumable), then **verify offline**:

   ```sh
   cargo run --release --bin pstt_fetch_block_headers -- \
     --rpc-url $RPC --blocks-file out/blocks/pstt_target_blocks.txt \
     --output-csv out/headers.csv
   cargo run --release --bin pstt_verify_block_headers -- \
     --headers-csv out/headers.csv \
     --blocks-file out/blocks/pstt_target_blocks.txt
   ```

3. **Normalize Binance aggTrades archives**:

   ```sh
   cargo run --release --bin pstt_normalize_aggtrades -- \
     --input-dir binance_ethusdc_full --symbol ETHUSDC --output-dir out/agg
   ```

4. **Spool oriented fills** (frozen stage-1 orientation:
   `q = |amount1|/1e18`, `p_exec = |amount0/amount1|*1e12`,
   `direction = -1 if amount1 > 0 else +1`; window `2024-01-01 <= t < 2025-12-27` UTC):

   ```sh
   cargo run --release --bin pstt_spool_fills -- \
     --swaps-csv swaps.csv.gz --block-ts-csv out/headers.csv \
     --block-col 0 --ts-col 3 --pools-json pools.json \
     --window-start-unix 1704067200 --window-end-unix 1766793600 \
     --output-dir out/fills
   ```

5. **Build weekly primitives** (strict-pre last-trade + strict `[T-1s,T)`
   VWAP joins, ISO `%G-%V` weeks):

   ```sh
   cargo run --release --bin pstt_build_weekly_primitives -- \
     --fills-csv out/fills/pstt_oriented_fills.csv \
     --trades-dir out/agg --output-dir out/weekly
   ```

6. **Run the standalone application** (signed five-corner projection —
   `r=0` plus all four `r=r_bar` envelope corners — over the frozen grid
   `r_bar ∈ {0, 0.05, 0.10, 0.50, 1.0}`, standalone ridge `1e-9·I`,
   region-wise denominator screen, synchronized calendar moving-block
   bootstrap with block length `max(2, round(sqrt(n_weeks)))`):

   ```sh
   cargo run --release --bin pstt_build_application -- \
     --weekly-json weekly_frozen_format.json \
     --lower-label 5bp --higher-label 30bp \
     --draws 1000 --seed <new-seed> --output-dir out/app
   ```

7. **Hash verification.** Record digests of every input and output with
   `pstt_manifest`, and compare weekly primitives against a reference with
   `pstt_verify_parity`:

   ```sh
   cargo run --release --bin pstt_manifest -- \
     --base out --file out/weekly/pstt_weekly.json --file out/headers.csv \
     --output-dir out/manifest
   cargo run --release --bin pstt_verify_parity -- \
     --expected reference_weekly.json --actual out/weekly/pstt_weekly.json
   ```

   The 17 frozen `freeze_manifest_*.sha256` files under
   `.local/selection_paper/` remain the authority for historical artifacts
   and must continue to verify unchanged.

## Parity test surface

`cargo test --test pstt_projection_gate --test pstt_standalone_gate \
  --test pstt_parity_gate --test pstt_prepare_gate --test pstt_staleness_gate`

covers: M5-S S1/S2/S3 authority cases and 200 randomized five-corner cases
against a stored NumPy oracle fixture
(`tests/fixtures/pstt/m5s_projection_fixture_v1.json`), denominator-screen
boundary cases (inclusive zero-touch abstention, strict `> 0` screen),
monotone nesting in `r_bar`, strict-pre-block last-trade and 1s-VWAP join
semantics, weekly primitive parity, certification boundary inequalities
(endpoints exactly at zero never certify), the three parameterized ridge
rules, and end-to-end determinism/replay of the standalone orchestrator.
