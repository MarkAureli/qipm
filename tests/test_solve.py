"""Tests for solve_instance: MPS or .std LP, solve with HiGHS, write runtime to instance .data.

Uses the same fixture stems as test_transform; reference .std files are used for standard-form tests.
"""

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("highspy", reason="highspy required for solve tests")
from solve import solve_instance, _solve_instance_from_path

FIXTURES = Path(__file__).resolve().parent / "fixtures"

SOLVE_FIXTURE_STEMS = [
    "min_sum",
    "equality",
    "three_var",
    "bounded_var",
    "lower_row",
    "free_var",
    "upper_var",
    "range_row",
]


@pytest.mark.parametrize("stem", SOLVE_FIXTURE_STEMS)
def test_solve_instance_std_completes_and_writes_data(stem: str, tmp_path: Path) -> None:
    """Solve .std instance; assert no error and runtime_highs_std written to .data."""
    std_path = FIXTURES / f"{stem}.std"
    if not std_path.is_file():
        pytest.skip(f"Fixture not found: {std_path}")

    instance_class = "cls"
    instance_dir = tmp_path / instance_class / stem
    instance_dir.mkdir(parents=True)
    shutil.copy(std_path, instance_dir / f"{stem}.std")
    solve_instance(instance_class, stem, cache_dir=tmp_path, formats="std")

    data_path = instance_dir / f"{stem}.data"
    assert data_path.is_file(), "solve_instance should write .data for .std"
    data = json.loads(data_path.read_text())
    assert "runtime_highs_std" in data
    assert data["runtime_highs_std"] >= 0.0, "Solve time should be non-negative"


@pytest.mark.parametrize("stem", SOLVE_FIXTURE_STEMS)
def test_solve_instance_mps_completes_and_writes_data(stem: str, tmp_path: Path) -> None:
    """Solve .mps instance; assert no error and runtime_highs_mps written to .data."""
    mps_path = FIXTURES / f"{stem}.mps"
    if not mps_path.is_file():
        pytest.skip(f"Fixture not found: {mps_path}")

    instance_class = "cls"
    instance_dir = tmp_path / instance_class / stem
    instance_dir.mkdir(parents=True)
    shutil.copy(mps_path, instance_dir / f"{stem}.mps")
    solve_instance(instance_class, stem, cache_dir=tmp_path, formats="mps")

    data_path = instance_dir / f"{stem}.data"
    assert data_path.is_file(), "solve_instance should write .data for .mps"
    data = json.loads(data_path.read_text())
    assert "runtime_highs_mps" in data
    assert data["runtime_highs_mps"] >= 0.0, "Solve time should be non-negative"


def test_solve_instance_file_not_found(tmp_path: Path) -> None:
    """solve_instance raises FileNotFoundError when instance subdir does not exist."""
    with pytest.raises(FileNotFoundError, match="Instance directory not found"):
        solve_instance("x", "nonexistent", cache_dir=tmp_path)


def test_solve_instance_unsupported_format(tmp_path: Path) -> None:
    """_solve_instance_from_path raises ValueError for unsupported extension."""
    bad_path = tmp_path / "dummy.xyz"
    bad_path.write_text("not an instance")
    with pytest.raises(ValueError, match="Unsupported instance format"):
        _solve_instance_from_path(bad_path)
