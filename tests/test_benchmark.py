"""Tests for benchmark: cycle counts for mnes/oss on fixture instances (.std).

For each test instance (fixture stem with .std) and each variant (mnes, oss),
runs the cycle-count benchmark and asserts the resulting cycle count is a non-negative integer.
"""

import json
import shutil
from pathlib import Path

import pytest

from benchmark import benchmark_instance

FIXTURES = Path(__file__).resolve().parent / "fixtures"


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

VARIANTS = ["mnes", "oss"]


@pytest.mark.parametrize("stem", BENCHMARK_FIXTURE_STEMS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_cycle_count_positive(stem: str, variant: str, tmp_path: Path) -> None:
    """For each instance and variant, cycle count is a non-negative integer."""
    std_path = FIXTURES / f"{stem}.std"
    if not std_path.is_file():
        pytest.skip(f"Fixture {stem} missing .std in {FIXTURES}")

    instance_class = "fixtures"
    instance_dir = tmp_path / instance_class / stem
    instance_dir.mkdir(parents=True)

    shutil.copy(std_path, instance_dir / f"{stem}.std")

    benchmark_instance(
        instance_class,
        stem,
        cache_dir=tmp_path,
        variant=variant,
    )

    data_path = instance_dir / f"{stem}.data"
    assert data_path.is_file(), "benchmark_instance should write .data"
    data = json.loads(data_path.read_text())
    key = f"cycle_count_{variant}"
    assert key in data, f"benchmark should write {key}"
    count = data[key]
    assert isinstance(count, int), f"{key} should be an integer"
    assert count >= 0, f"{key} should be non-negative, got {count}"
