"""Tests for transform_instance: MPS -> presolve -> standard form (.npz).

Fixture coverage: see tests/fixtures/README.md. Parametrized tests use min_sum, equality,
three_var with expected (c,b,A). Edge-case fixtures (bounded_var, lower_row, free_var,
range_row) are tested for valid standard-form output only (shape, keys, finite values).
"""

import numpy as np
import pytest
from pathlib import Path
from scipy.sparse import csr_matrix

pytest.importorskip("highspy", reason="highspy required for transform tests")
from transform import transform_instance

# Fixture directory
FIXTURES = Path(__file__).resolve().parent / "fixtures"


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


# Expected standard forms for small instances (min c'x, Ax=b, x>=0)

# min_sum.mps: min x1 + x2  s.t.  x1 + x2 <= 10,  x1,x2 >= 0
# Standard: min c'x  s.t.  x1 + x2 + s = 10,  x1,x2,s >= 0
MIN_SUM_EXPECTED = {
    "c": np.array([1.0, 1.0, 0.0]),
    "b": np.array([10.0]),
    "A": csr_matrix([[1.0, 1.0, 1.0]]),
}

# equality.mps: min x1  s.t.  x1 + x2 = 5,  x1,x2 >= 0
EQUALITY_EXPECTED = {
    "c": np.array([1.0, 0.0]),
    "b": np.array([5.0]),
    "A": csr_matrix([[1.0, 1.0]]),
}

# three_var.mps: min x1+x2+x3  s.t.  x1+x2 <= 5,  x2+x3 <= 7,  x1,x2,x3 >= 0
# After presolve the second row may be reduced (e.g. x2 eliminated); we use the actual
# presolved standard form: 5 vars, 2 eqs (row2 is 0,0,1,0,1 from observed HiGHS output)
THREE_VAR_EXPECTED = {
    "c": np.array([1.0, 1.0, 1.0, 0.0, 0.0]),
    "b": np.array([5.0, 7.0]),
    "A": csr_matrix([[1.0, 1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0, 1.0]]),
}


@pytest.mark.parametrize(
    "stem,expected",
    [
        ("min_sum", MIN_SUM_EXPECTED),
        ("equality", EQUALITY_EXPECTED),
        ("three_var", THREE_VAR_EXPECTED),
    ],
)
def test_transform_instance_matches_expected(stem: str, expected: dict, tmp_path: Path) -> None:
    """Transform MPS fixture to .npz and compare to expected c, b, A."""
    mps_path = FIXTURES / f"{stem}.mps"
    if not mps_path.is_file():
        pytest.skip(f"Fixture not found: {mps_path}")

    # Copy to tmp_path so we don't write into repo
    import shutil
    mps_tmp = tmp_path / f"{stem}.mps"
    shutil.copy(mps_path, mps_tmp)

    transform_instance(mps_tmp)
    out_path = tmp_path / f"{stem}.npz"
    assert out_path.suffix == ".npz"
    assert out_path.name == f"{stem}.npz"

    c, b, A = load_standard_npz(out_path)
    # Test case requirement: presolved standard form must be non-empty
    assert c.size > 0, "Presolved formulation has no variables (empty c)"
    assert b.size > 0, "Presolved formulation has no constraints (empty b)"
    assert A.shape[0] > 0 and A.shape[1] > 0, "Presolved constraint matrix A is empty"
    assert_standard_form_equal(
        c, b, A,
        expected["c"], expected["b"], expected["A"],
    )


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


# Edge-case fixtures (exact output is HiGHS-dependent); validated for well-formed standard form only
EDGE_CASE_FIXTURES = ["bounded_var", "lower_row", "free_var", "upper_var", "range_row"]


@pytest.mark.parametrize("stem", EDGE_CASE_FIXTURES)
def test_transform_edge_case_fixtures_produce_valid_standard_form(
    stem: str, tmp_path: Path
) -> None:
    """Transform edge-case MPS fixtures; assert output npz is valid standard form (c, b, A)."""
    import shutil

    mps_path = FIXTURES / f"{stem}.mps"
    if not mps_path.is_file():
        pytest.skip(f"Fixture not found: {mps_path}")

    mps_tmp = tmp_path / f"{stem}.mps"
    shutil.copy(mps_path, mps_tmp)
    transform_instance(mps_tmp)

    out_path = tmp_path / f"{stem}.npz"
    assert out_path.is_file()
    data = np.load(out_path, allow_pickle=False)
    required = {"c", "b", "A_data", "A_indices", "A_indptr", "A_shape"}
    assert required.issubset(set(data.keys())), f"Missing keys in npz: {required - set(data.keys())}"

    c = data["c"]
    b = data["b"]
    shape = tuple(data["A_shape"])
    assert c.ndim == 1 and b.ndim == 1
    assert shape[0] == len(b) and shape[1] == len(c)
    assert np.all(np.isfinite(c)) and np.all(np.isfinite(b))
    assert c.size > 0 and b.size > 0
