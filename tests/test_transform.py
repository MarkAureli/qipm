"""Tests for transform_instance: MPS -> presolve -> standard form (.npz).

Fixture coverage: see tests/fixtures/README.md. Reference .npz files in tests/fixtures/
are the expected standard form for each .mps. Parametrized tests compare transform output
to these references; edge-case fixtures are validated for well-formed standard form only.
"""

import numpy as np
import pytest
from pathlib import Path
from scipy.sparse import csr_matrix

pytest.importorskip("highspy", reason="highspy required for transform tests")
from transform import transform_instance

# Fixture directory
FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Stems that have reference .npz for exact comparison (all MPS fixtures have reference .npz)
REFERENCE_NPZ_STEMS = ["min_sum", "equality", "three_var", "bounded_var", "lower_row", "free_var", "upper_var", "range_row"]


def load_standard_npz(path: Path) -> tuple[np.ndarray, np.ndarray, csr_matrix]:
    """Load c, b, A from our .npz format (A in CSR via data/indices/indptr/shape)."""
    data = np.load(path, allow_pickle=False)
    c = data["c"]
    b = data["b"]
    shape = tuple(data["A_shape"])
    A = csr_matrix(
        (data["A_data"], data["A_indices"], data["A_indptr"]),
        shape=shape,
    )
    return c, b, A


def assert_standard_form_equal(
    c: np.ndarray,
    b: np.ndarray,
    A: csr_matrix,
    c_exp: np.ndarray,
    b_exp: np.ndarray,
    A_exp: csr_matrix,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> None:
    np.testing.assert_allclose(c, c_exp, rtol=rtol, atol=atol)
    np.testing.assert_allclose(b, b_exp, rtol=rtol, atol=atol)
    assert A.shape == A_exp.shape
    Ad = A.toarray()
    Ae = A_exp.toarray()
    np.testing.assert_allclose(Ad, Ae, rtol=rtol, atol=atol)


@pytest.mark.parametrize("stem", REFERENCE_NPZ_STEMS)
def test_transform_instance_matches_expected(stem: str, tmp_path: Path) -> None:
    """Transform MPS fixture to .npz and compare to reference .npz in fixtures."""
    mps_path = FIXTURES / f"{stem}.mps"
    ref_npz_path = FIXTURES / f"{stem}.npz"
    if not mps_path.is_file():
        pytest.skip(f"Fixture not found: {mps_path}")
    if not ref_npz_path.is_file():
        pytest.skip(f"Reference .npz not found: {ref_npz_path}")

    import shutil
    mps_tmp = tmp_path / f"{stem}.mps"
    shutil.copy(mps_path, mps_tmp)

    transform_instance(mps_tmp)
    out_path = tmp_path / f"{stem}.npz"
    assert out_path.suffix == ".npz"
    assert out_path.name == f"{stem}.npz"

    c, b, A = load_standard_npz(out_path)
    c_exp, b_exp, A_exp = load_standard_npz(ref_npz_path)
    assert c.size > 0, "Presolved formulation has no variables (empty c)"
    assert b.size > 0, "Presolved formulation has no constraints (empty b)"
    assert A.shape[0] > 0 and A.shape[1] > 0, "Presolved constraint matrix A is empty"
    assert_standard_form_equal(c, b, A, c_exp, b_exp, A_exp)


def test_transform_instance_writes_npz(tmp_path: Path) -> None:
    """transform_instance writes .npz file next to the MPS file."""
    mps_path = FIXTURES / "equality.mps"
    if not mps_path.is_file():
        pytest.skip("Fixture equality.mps not found")
    mps_tmp = tmp_path / "equality.mps"
    import shutil
    shutil.copy(mps_path, mps_tmp)

    transform_instance(mps_tmp)
    out = tmp_path / "equality.npz"
    assert out.is_file()


def test_transform_instance_file_not_found() -> None:
    """transform_instance raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError, match="not found"):
        transform_instance("/nonexistent/path.mps")


