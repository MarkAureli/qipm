"""Tests for solve_instance: MPS or .npz LP, solve with HiGHS, write .mps_time or .std_time.

Instance-level tests only. Solving .mps and solving .npz are tested separately.
Uses the same fixture stems as test_transform; reference .npz files are used for NPZ tests.
"""

import shutil
from pathlib import Path

import pytest

pytest.importorskip("highspy", reason="highspy required for solve tests")
from solve import solve_instance

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
def test_solve_instance_npz_completes_and_writes_std_time(stem: str, tmp_path: Path) -> None:
    """Solve .npz instance; assert no error and .std_time written with valid time."""
    npz_path = FIXTURES / f"{stem}.npz"
    if not npz_path.is_file():
        pytest.skip(f"Fixture not found: {npz_path}")

    npz_tmp = tmp_path / f"{stem}.npz"
    shutil.copy(npz_path, npz_tmp)
    solve_instance(npz_tmp)

    std_time_path = tmp_path / f"{stem}.std_time"
    assert std_time_path.is_file(), "solve_instance should write .std_time for .npz"
    elapsed = float(std_time_path.read_text().strip())
    assert elapsed >= 0.0, "Solve time should be non-negative"


@pytest.mark.parametrize("stem", SOLVE_FIXTURE_STEMS)
def test_solve_instance_mps_completes_and_writes_mps_time(stem: str, tmp_path: Path) -> None:
    """Solve .mps instance; assert no error and .mps_time written with valid time."""
    mps_path = FIXTURES / f"{stem}.mps"
    if not mps_path.is_file():
        pytest.skip(f"Fixture not found: {mps_path}")

    mps_tmp = tmp_path / f"{stem}.mps"
    shutil.copy(mps_path, mps_tmp)
    solve_instance(mps_tmp)

    mps_time_path = tmp_path / f"{stem}.mps_time"
    assert mps_time_path.is_file(), "solve_instance should write .mps_time for .mps"
    elapsed = float(mps_time_path.read_text().strip())
    assert elapsed >= 0.0, "Solve time should be non-negative"


def test_solve_instance_file_not_found() -> None:
    """solve_instance raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError, match="not found"):
        solve_instance("/nonexistent/path.npz")


def test_solve_instance_unsupported_format(tmp_path: Path) -> None:
    """solve_instance raises ValueError for unsupported extension."""
    bad_path = tmp_path / "dummy.xyz"
    bad_path.write_text("not an instance")
    with pytest.raises(ValueError, match="Unsupported instance format"):
        solve_instance(bad_path)
