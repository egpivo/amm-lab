"""baseline-duopoly-E: write out/m3r_run_manifest.json — environment, commands, seeds,
checkpoint + CSV hashes, selected checkpoints. Run AFTER all baseline-duopoly artifacts
exist and choices are frozen.
"""

import hashlib
import platform
import subprocess
import sys
from pathlib import Path

from common import OUT, REPO_ROOT, write_json


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def cmd(args: list[str]) -> str:
    try:
        result = subprocess.run(
            args, check=True, capture_output=True, text=True, encoding="utf-8"
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"
    return result.stdout.strip()


def main() -> None:
    import torch

    manifest = {
        "python": sys.version,
        "torch": torch.__version__,
        "rustc": cmd(["rustc", "--version"]),
        "cargo_lock_sha256": sha256(REPO_ROOT / "Cargo.lock"),
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "gpu": "none (CPU-only, torch.set_num_threads(1))",
        "git_commit": cmd(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]),
        "git_dirty": cmd(["git", "-C", str(REPO_ROOT), "status", "--porcelain"]) != "",
        "seed_protocol": {
            "train": "1_000_000 + episode (fresh path per episode)",
            "validation": "20_000-20_199 (baseline/planner tuning); "
            "20_000-20_049 (DQN checkpoint selection)",
            "test": "30_000-30_499 (development-visible)",
            "fresh": "40_000-40_499 (used once per milestone)",
            "final_paper": "90_000-90_999 (untouched until baseline-duopoly-F freeze)",
        },
        "frozen_choices": {
            "learner": "DQN MLP 16-64-64-8, checkpoint by 50-seed validation",
            "headline_checkpoint": "dqn_order_after.pt evaluated agent-last "
            "(conservative headline per baseline-duopoly-B discipline); "
            "dqn_dynamic_duopoly.pt agent-first and dqn_order_random.pt "
            "random-order disclosed alongside",
            "completion_rule": "forced_terminal (all policies)",
            "priority_ordering": "agent-last headline; before/random disclosed",
            "mode": "dynamic_duopoly",
            "lookahead": "kappa=16 (validation-tuned)",
            "stochastic_planner": "K=3, N=16, kappa=16 (validation-tuned)",
        },
        "commands": [
            "# python steps run from scripts/rl_equilibrium/; see repo Makefile",
            "cargo build --release --bins",
            "cargo test",
            "python3 dqn_train.py --episodes 12000 --tag dynamic_duopoly",
            "python3 dqn_train.py --episodes 12000 --tag completion_aware --train-penalty 0.08",
            "python3 dqn_train.py --episodes 12000 --tag order_random --agent-order random",
            "python3 dqn_train.py --episodes 12000 --tag order_after --agent-order after",
            "python3 dqn_train.py --episodes 12000 --tag constant_duopoly --mode constant_duopoly",
            "python3 dqn_train.py --episodes 12000 --tag dynamic_monopoly --mode dynamic_monopoly",
            "./target/release/rl_equilibrium_completion --n-seeds 500",
            "./target/release/rl_equilibrium_reference --n-seeds 300",
            "./target/release/rl_equilibrium_planner --n-val 200 --n-seeds 300",
            "python3 dqn_baseline_eval.py --n-seeds 300 --n-completion 500",
            "python3 dqn_final_block.py",
            "python3 verify_paper_artifacts.py",
            "python3 make_manifest.py",
        ],
        "checkpoints": {},
        "csv_sha256": {},
    }
    for pt in sorted(OUT.glob("dqn_*.pt")):
        manifest["checkpoints"][pt.name] = sha256(pt)
    for c in sorted(OUT.glob("m3r_*.csv")) + sorted(OUT.glob("m3_dqn_curve_*.csv")):
        manifest["csv_sha256"][c.name] = sha256(c)
    path = OUT / "m3r_run_manifest.json"
    write_json(path, manifest)
    print("wrote", path)


if __name__ == "__main__":
    main()
