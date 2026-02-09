"""Tests for solve_instance: MPS or .npz LP, solve with HiGHS, write .mps_time or .std_time.

Uses the same fixtures as test_transform. For .npz we transform MPS first; for .mps we solve
directly. We assert each run completes without error and writes the correct time file.
"""

import shutil
from pathlib import Path

import pytest

pytest.importorskip("highspy", reason="highspy required for solve tests")
from solve import solve_instance, solve_instance_class
from transform import transform_instance

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# All MPS fixtures (same as transform tests): parametrized + edge cases
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
    """Transform MPS to .npz, run solve_instance; assert no error and .std_time written with valid time."""
    mps_path = FIXTURES / f"{stem}.mps"
    if not mps_path.is_file():
        pytest.skip(f"Fixture not found: {mps_path}")

    mps_tmp = tmp_path / f"{stem}.mps"
    shutil.copy(mps_path, mps_tmp)
    transform_instance(mps_tmp)

    npz_path = tmp_path / f"{stem}.npz"
    assert npz_path.is_file(), "transform_instance should have produced .npz"

    solve_instance(npz_path)

    std_time_path = tmp_path / f"{stem}.std_time"
    assert std_time_path.is_file(), "solve_instance should write .std_time for .npz"
    elapsed = float(std_time_path.read_text().strip())
    assert elapsed >= 0.0, "Solve time should be non-negative"


@pytest.mark.parametrize("stem", SOLVE_FIXTURE_STEMS)
def test_solve_instance_mps_completes_and_writes_mps_time(stem: str, tmp_path: Path) -> None:
    """Run solve_instance on .mps; assert no error and .mps_time written with valid time."""
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


def test_solve_instance_class_formats_mps_only(tmp_path: Path) -> None:
    """solve_instance_class with formats='mps' only solves .mps files."""
    class_dir = tmp_path / "myclass"
    class_dir.mkdir()
    shutil.copy(FIXTURES / "min_sum.mps", class_dir / "a.mps")
    transform_instance(class_dir / "a.mps")
    # Now we have a.mps and a.npz
    solve_instance_class("myclass", cache_dir=tmp_path, formats="mps")
    assert (class_dir / "a.mps_time").is_file()
    assert not (class_dir / "a.std_time").is_file()


def test_solve_instance_class_formats_npz_only(tmp_path: Path) -> None:
    """solve_instance_class with formats='npz' only solves .npz files."""
    class_dir = tmp_path / "myclass"
    class_dir.mkdir()
    shutil.copy(FIXTURES / "min_sum.mps", class_dir / "a.mps")
    transform_instance(class_dir / "a.mps")
    solve_instance_class("myclass", cache_dir=tmp_path, formats="npz")
    assert (class_dir / "a.std_time").is_file()
    assert not (class_dir / "a.mps_time").is_file()


def test_solve_instance_class_formats_both(tmp_path: Path) -> None:
    """solve_instance_class with formats='both' solves .mps and .npz."""
    class_dir = tmp_path / "myclass"
    class_dir.mkdir()
    shutil.copy(FIXTURES / "min_sum.mps", class_dir / "a.mps")
    transform_instance(class_dir / "a.mps")
    solve_instance_class("myclass", cache_dir=tmp_path, formats="both")
    assert (class_dir / "a.mps_time").is_file()
    assert (class_dir / "a.std_time").is_file()


def test_solve_instance_class_formats_invalid() -> None:
    """solve_instance_class raises ValueError for invalid formats."""
    with pytest.raises(ValueError, match="formats must be"):
        solve_instance_class("x", formats="invalid")
