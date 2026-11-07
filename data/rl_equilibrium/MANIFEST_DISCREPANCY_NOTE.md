# m3r_run_manifest.json — discrepancy reconciled

`m3r_run_manifest.json` records SHA-256 hashes for 15 result CSVs. **5 match the files in this directory as committed; 10 differ — and the difference is line endings only.** The manifest and the CSVs are unchanged by this note.

## Cause (verified)
Those 10 files are written by Python's `csv.writer` / `csv.DictWriter` with the default `\r\n` terminator. The author's global Git configuration has `core.autocrlf=input` (and the repository has no `.gitattributes`), so Git stored them with LF endings. For each of the 10 files below, **the SHA-256 of the committed bytes with every line ending rewritten from LF to CRLF equals the manifest hash exactly**; the content is identical.

| File | Manifest SHA-256 (reproduced by LF→CRLF re-expansion) |
|---|---|
| `m3r_dynamic_fee_ablation.csv` | `09111b44aa53b166e573121791cbf916da62f416fee7af511b46e97d74aed1af` |
| `m3r_final_paper_seeds.csv` | `d0834dcca6e551dea7f4bff2362a833c18f2c7cec81a589091848c819b371ebb` |
| `m3r_priority.csv` | `a1fb9d7546586515eb62ba7a1c11efa0613ac79e25aa035678ac33e0bf31ed5e` |
| `m3r_symmetry.csv` | `044f989f4117c97aba56e6f5df99a2b7c9dd767e76560b2e83ffb72d4191e189` |
| `m3_dqn_curve_completion_aware.csv` | `eb5d378845cf7664b8064d6fac9e9331d633a5087ac2f0ad4eaa1c6d53b80425` |
| `m3_dqn_curve_constant_duopoly.csv` | `64b018daa15099245c76328f364aa931ecd05143604a26956fc2c95daccded56` |
| `m3_dqn_curve_dynamic_monopoly.csv` | `e06c130aa75c48359c4ec17e344e6161c4fca6b297b33d3df90c6f0c364860a1` |
| `m3_dqn_curve_order_after.csv` | `185cc5ec04d8c3076cbd5b6530553297973af0e55941fd49f06ff00d95d265ff` |
| `m3_dqn_curve_order_random.csv` | `77ede4a6ffd1c9bf10fb2876fc1444a25a00b9e71d0fc3a3ebd3dee46a11c483` |
| `m3_dqn_curve_smoke.csv` | `1205a552a0399bfc76431859c8ed7f68f013eda1cbd7f249b1a5eaa83ac23d07` |

The 5 files that match as committed are the Rust-written CSVs and `m3r_completion.csv` (whose Python-appended rows use an explicit LF terminator).

To verify: for each file above, read the committed bytes, replace every `\r\n` with `\n` and then every `\n` with `\r\n`, and compute SHA-256.

## What this supersedes
An earlier (unpublished) version of this note attributed the mismatch to the pipeline re-writing the CSVs after the manifest was generated, citing equal modification times. That explanation is wrong: 89 of the 116 files in this directory, including the matching CSVs and the checkpoints, share one modification time (the time a merge wrote them into a working tree), which carries no information about when they were generated.

## Manifest provenance (bounded)
The manifest records `git_commit = ab79436…` with `git_dirty = true`. `ab79436` contains no RL code; it is the **parent of the first RL commit `3a6518f`**, and the dirty tree was the uncommitted RL working tree of that period. `Cargo.toml` and `Cargo.lock` are byte-identical in `ab79436`, `3a6518f` and `1a4555f`, and the lock hash in the manifest equals them. The exact bytes of the uncommitted Python scripts at generation time were not preserved; the candidate committed states are `3a6518f` (most likely), `1a4555f` and `1eb1c79`, and none is claimed to be the exact historical working tree. Original outputs were untracked at the time (`experiments/` was gitignored), which is why no earlier commit contains them.

## Limits of the claims
(1) *Authenticity by hash*: all 15 CSVs and 6 checkpoints agree with the manifest (CSVs after line-ending reconstruction). (2) *Independent regeneration*: only a bounded subset has been regenerated (400 of 480 development rows, seeds 30,000–30,019, exact agreement on identifying fields and outcome values). (3) *Exact historical execution provenance*: bounded, not recovered. The manifest's command list omits the final-block Rust runs for `m3r_reference_final.csv` and `m3r_stochastic_planner_final.csv` and a smoke run, and the training logs were never committed (`*.log` is ignored).
