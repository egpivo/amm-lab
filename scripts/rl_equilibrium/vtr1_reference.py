"""Guarded entry point for the Rust reference policies (lookahead, TWAP) and kappa tuning.

Every VTR-1 invocation of `rl_equilibrium_reference` goes through here: the seed range is
checked against the registry (protected blocks need an authorization file) and the same
scenario JSON as DQN training / evaluation is passed on. The Rust binary itself is not guarded.
"""

import argparse
import subprocess
from pathlib import Path

import scenario_config
import seed_blocks

BINARY = Path(__file__).resolve().parents[2] / "target/release/rl_equilibrium_reference"


def build_command(args: argparse.Namespace) -> list[str]:
    if args.tune_kappa:  # VTR-1 kappa tuning: protected validation block B, scope "kappa_tuning"
        seed_blocks.require("vtr1_val", args.seed_base, args.n, args.authorization, scope="kappa_tuning")
    else:
        seed_blocks.require(args.role, args.seed_base, args.n, args.authorization)
    scenario_config.check_cell(args.cell)
    if not args.tune_kappa:
        seed_blocks.check_new_outputs([args.out])
    scenario_config.load(args.scenario)  # validate keys before launching
    cmd = [str(args.binary), "--cell", args.cell]
    if args.scenario:
        cmd += ["--scenario", args.scenario]
    if args.tune_kappa:
        cmd += ["--tune-kappa", "--tune-seed-base", str(args.seed_base), "--tune-n", str(args.n)]
    else:
        cmd += [
            "--seed-base", str(args.seed_base),
            "--n-seeds", str(args.n),
            "--seed-label", args.seed_label,
            "--kappa", str(args.kappa),
            "--out", args.out,
        ]  # fmt: skip
    return cmd


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--tune-kappa", action="store_true", help="kappa grid on a validation block")
    ap.add_argument("--role", choices=["eval", "pilot", "final"], default="eval")
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--authorization", default=None)
    ap.add_argument("--kappa", type=float, default=16.0)
    ap.add_argument("--seed-label", default="vtr1")
    ap.add_argument("--out", default="vtr1_reference.csv")
    ap.add_argument("--binary", default=str(BINARY))
    ap.add_argument("--dry-run", action="store_true", help="validate and print the command only")
    return ap


def main() -> None:
    args = parser().parse_args()
    cmd = build_command(args)
    if args.dry_run:
        print(" ".join(cmd))
        return
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
