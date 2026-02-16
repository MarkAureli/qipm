"""Tests for benchmark: gate counts for qipm1/2/3 on fixture instances (.sde + .init).

For each test instance (fixture stem with both .sde and .init) and each qipm method (1, 2, 3),
runs the gate-count benchmark and asserts the resulting gate count is a positive integer.
Uses the same fixture stems as test_initialise; skips stems that do not have .sde and .init.
"""

import json
import shutil
from pathlib import Path

import pytest

from benchmark import benchmark_instance

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Same fixture stems as test_initialise; instances need .sde or .std and .init
BENCHMARK_FIXTURE_STEMS = [
    "min_sum",
    "equality",
    "three_var",
    "bounded_var",
    "lower_row",
    "free_var",
    "upper_var",
    "range_row",
]

QIPM_METHODS = [1, 2, 3]


def _instance_has_required_files(stem: str) -> bool:
    """True if fixture has .sde and .init (benchmark uses SDE + embedding triple)."""
    return (FIXTURES / f"{stem}.sde").is_file() and (FIXTURES / f"{stem}.init").is_file()


@pytest.mark.parametrize("stem", BENCHMARK_FIXTURE_STEMS)
@pytest.mark.parametrize("method", QIPM_METHODS)
def test_gate_count_positive(stem: str, method: int, tmp_path: Path) -> None:
    """For each instance and qipm method, gate count is a positive integer."""
    if not _instance_has_required_files(stem):
        pytest.skip(f"Fixture {stem} missing .sde/.std or .init in {FIXTURES}")

    instance_class = "fixtures"
    instance_dir = tmp_path / instance_class / stem
    instance_dir.mkdir(parents=True)

    shutil.copy(FIXTURES / f"{stem}.sde", instance_dir / f"{stem}.sde")
    shutil.copy(FIXTURES / f"{stem}.init", instance_dir / f"{stem}.init")

    benchmark_instance(
        instance_class,
        stem,
        cache_dir=tmp_path,
        qipm_numbers=[method],
    )

    data_path = instance_dir / f"{stem}.data"
    assert data_path.is_file(), "benchmark_instance should write .data"
    data = json.loads(data_path.read_text())
    key = f"gate_count_qipm{method}"
    assert key in data, f"benchmark should write {key}"
    count = data[key]
    assert isinstance(count, int), f"{key} should be an integer"
    assert count > 0, f"{key} should be positive, got {count}"
