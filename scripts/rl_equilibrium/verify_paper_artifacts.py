"""Verify the artifact values used by the paper."""

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

from common import CsvRow, OUT, mean, read_csv


TOLERANCE_BPS = 0.05
FINAL_SEEDS = set(range(90_000, 91_000))

DQN_EXPECTED = {
    ("dqn_dynamic_duopoly", "before"): 85.73,
    ("dqn_order_random", "random"): 96.56,
    ("dqn_order_after", "after"): 100.32,
}
LOOKAHEAD_EXPECTED = {"before": 100.63, "random": 102.18, "after": 113.61}
TWAP_EXPECTED = {"before": 111.24, "random": 112.92, "after": 123.12}
LADDER_EXPECTED = {
    "twap": 111.24,
    "lookahead": 100.63,
    "two_step": 111.45,
    "three_step": 102.25,
    "q_learner": 108.86,
    "q_learner_fine": 100.21,
    "clairvoyant": 78.20,
}


class Verifier:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        print(
            f"{'PASS' if condition else 'FAIL'}  {name}"
            + (f"  ({detail})" if detail else "")
        )
        if not condition:
            self.failures.append(name)

    def check_mean(self, name: str, rows: Sequence[CsvRow], expected: float) -> None:
        if not rows:
            self.check(name, False, "no rows")
            return
        actual = mean(float(row["shortfall_bps"]) for row in rows)
        self.check(
            name,
            abs(actual - expected) < TOLERANCE_BPS,
            f"got {actual:.2f}, n={len(rows)}",
        )


def selected(rows: Iterable[CsvRow], **criteria: str) -> list[CsvRow]:
    return [
        row
        for row in rows
        if all(row.get(column) == value for column, value in criteria.items())
    ]


def verify_final_block(verifier: Verifier) -> None:
    reference = read_csv(OUT / "m3r_reference_final.csv")
    dqn = read_csv(OUT / "m3r_final_paper_seeds.csv")

    for (policy, order), expected in DQN_EXPECTED.items():
        rows = selected(dqn, policy=policy, agent_order=order)
        label = f"table1 {policy}/{order}"
        verifier.check(f"{label} n=1000", len(rows) == 1_000, f"n={len(rows)}")
        seeds = {int(row["seed"]) for row in rows}
        verifier.check(f"{label} seed block", seeds == FINAL_SEEDS)
        verifier.check_mean(f"{label} mean {expected}", rows, expected)
        verifier.check(
            f"{label} completion==1.0",
            bool(rows) and all(float(row["completion_rate"]) == 1.0 for row in rows),
        )

    for policy, expected_by_order in (
        ("lookahead", LOOKAHEAD_EXPECTED),
        ("twap", TWAP_EXPECTED),
    ):
        for order, expected in expected_by_order.items():
            rows = selected(
                reference,
                policy=policy,
                agent_order=order,
                mode="dynamic_duopoly",
            )
            verifier.check_mean(
                f"table1 {policy}/{order} mean {expected}", rows, expected
            )


def verify_ladder(verifier: Verifier) -> None:
    rows = read_csv(OUT / "final_ladder.csv")
    for policy, expected in LADDER_EXPECTED.items():
        policy_rows = selected(rows, policy=policy)
        verifier.check(
            f"fig1 {policy} n=1000", len(policy_rows) == 1_000, f"n={len(policy_rows)}"
        )
        verifier.check_mean(f"fig1 {policy} mean {expected}", policy_rows, expected)


def verify_planner(verifier: Verifier) -> None:
    rows = read_csv(OUT / "m3r_stochastic_planner_final.csv")
    by_policy: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_policy[row["policy"]][row["seed"]] = float(row["shortfall_bps"])

    planner = by_policy.get("stochastic_planner", {})
    lookahead = by_policy.get("lookahead", {})
    shared_seeds = set(planner) & set(lookahead)
    verifier.check(
        "planner and lookahead have identical seeds",
        bool(planner) and set(planner) == set(lookahead),
        f"shared={len(shared_seeds)}",
    )
    if not shared_seeds:
        verifier.check("planner final tie |paired| < 0.7 bps", False, "no paired rows")
        return
    difference = mean(planner[seed] - lookahead[seed] for seed in shared_seeds)
    verifier.check(
        "planner final tie |paired| < 0.7 bps",
        abs(difference) < 0.7,
        f"paired {difference:+.2f}",
    )


def verify_duplicates(verifier: Verifier) -> None:
    files = {
        "m3r_completion.csv": ("policy", "completion_rule", "seed_set", "seed"),
        "m4_lp_adaptation.csv": ("extension", "regime", "policy", "seed"),
        "m4_jit_mev.csv": ("extension", "regime", "policy", "seed"),
    }
    for filename, key_columns in files.items():
        keys = Counter(
            tuple(row[column] for column in key_columns)
            for row in read_csv(OUT / filename)
        )
        duplicate_count = sum(count > 1 for count in keys.values())
        verifier.check(
            f"no duplicate rows in {filename}",
            duplicate_count == 0,
            f"{duplicate_count} duplicated keys" if duplicate_count else "",
        )


def verify_m4(verifier: Verifier) -> None:
    for filename in ("m4_lp_adaptation.csv", "m4_jit_mev.csv"):
        rows = read_csv(OUT / filename)
        verifier.check(
            f"{filename} completion==1.0",
            bool(rows) and all(float(row["completion_rate"]) == 1.0 for row in rows),
        )
        cells: dict[str, dict[str, dict[str, float]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        for row in rows:
            cells[row["regime"]][row["policy"]][row["seed"]] = float(
                row["shortfall_bps"]
            )
        for regime, policies in cells.items():
            dqn = policies.get("dqn", {})
            lookahead = policies.get("lookahead", {})
            shared_seeds = set(dqn) & set(lookahead)
            verifier.check(
                f"{filename} {regime} paired seeds",
                bool(dqn) and set(dqn) == set(lookahead),
                f"shared={len(shared_seeds)}",
            )
            if not shared_seeds:
                verifier.check(
                    f"{filename} {regime} DQN edge negative", False, "no pairs"
                )
                continue
            edge = mean(dqn[seed] - lookahead[seed] for seed in shared_seeds)
            verifier.check(
                f"{filename} {regime} DQN edge negative",
                edge < 0,
                f"{edge:+.2f}",
            )


def main() -> int:
    verifier = Verifier()
    verify_final_block(verifier)
    verify_ladder(verifier)
    verify_planner(verifier)
    verify_duplicates(verifier)
    verify_m4(verifier)
    print(f"\n{len(verifier.failures)} failure(s)")
    return int(bool(verifier.failures))


if __name__ == "__main__":
    raise SystemExit(main())
