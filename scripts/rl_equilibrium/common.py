import csv
import json
import os
import random
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "experiments/rl_execution/out"

CsvRow = dict[str, str]


def read_csv(path: Path) -> list[CsvRow]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", newline="", encoding="utf-8", dir=path.parent, delete=False
        ) as file:
            temp_path = Path(file.name)
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        assert temp_path is not None
        os.replace(temp_path, path)
    except BaseException:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def append_csv_rows(path: Path, rows: Iterable[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, delete=False
        ) as destination:
            temp_path = Path(destination.name)
            with path.open("rb") as source:
                contents = source.read()
            destination.write(contents)
            if contents and not contents.endswith((b"\n", b"\r")):
                destination.write(b"\n")
        assert temp_path is not None
        with temp_path.open("a", newline="", encoding="utf-8") as file:
            csv.writer(file, lineterminator="\n").writerows(rows)
        os.replace(temp_path, path)
    except BaseException:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as file:
            temp_path = Path(file.name)
            json.dump(value, file, indent=2)
            file.write("\n")
        assert temp_path is not None
        os.replace(temp_path, path)
    except BaseException:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        raise ValueError("mean requires at least one value")
    return sum(items) / len(items)


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 2_000, seed: int = 0
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap_ci requires at least one value")
    rng = random.Random(seed)
    size = len(values)
    means = sorted(mean(rng.choices(values, k=size)) for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]
