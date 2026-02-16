"""Tests for transform_instance: MPS -> presolve -> standard form (.std).

Fixture coverage: see tests/fixtures/README.md. Reference .std files in tests/fixtures/
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

# Stems that have reference .std for exact comparison (all MPS fixtures have reference .std)
REFERENCE_STD_STEMS = ["min_sum", "equality", "three_var", "bounded_var", "lower_row", "free_var", "upper_var", "range_row"]


def load_standard_std(path: Path) -> tuple[np.ndarray, np.ndarray, csr_matrix]:
    """Load c, b, A from our .std format (A in CSR via data/indices/indptr/shape)."""
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


@pytest.mark.parametrize("stem", REFERENCE_STD_STEMS)
def test_transform_instance_matches_expected(stem: str, tmp_path: Path) -> None:
    """Transform MPS fixture to .std and compare to reference .std in fixtures."""
    mps_path = FIXTURES / f"{stem}.mps"
    ref_std_path = FIXTURES / f"{stem}.std"
    if not mps_path.is_file():
        pytest.skip(f"Fixture not found: {mps_path}")
    if not ref_std_path.is_file():
        pytest.skip(f"Reference .std not found: {ref_std_path}")

    import shutil
    instance_class = "cls"
    instance_dir = tmp_path / instance_class / stem
    instance_dir.mkdir(parents=True)
    mps_tmp = instance_dir / f"{stem}.mps"
    shutil.copy(mps_path, mps_tmp)

    transform_instance(instance_class, stem, cache_dir=tmp_path)
    out_path = instance_dir / f"{stem}.std"
    assert out_path.suffix == ".std"
    assert out_path.name == f"{stem}.std"

    c, b, A = load_standard_std(out_path)
    c_exp, b_exp, A_exp = load_standard_std(ref_std_path)
    assert c.size > 0, "Presolved formulation has no variables (empty c)"
    assert b.size > 0, "Presolved formulation has no constraints (empty b)"
    assert A.shape[0] > 0 and A.shape[1] > 0, "Presolved constraint matrix A is empty"
    assert_standard_form_equal(c, b, A, c_exp, b_exp, A_exp)


def test_transform_instance_writes_std(tmp_path: Path) -> None:
    """transform_instance writes .std file next to the MPS file in the instance subfolder."""
    mps_path = FIXTURES / "equality.mps"
    if not mps_path.is_file():
        pytest.skip("Fixture equality.mps not found")
    import shutil
    instance_class = "cls"
    instance_name = "equality"
    instance_dir = tmp_path / instance_class / instance_name
    instance_dir.mkdir(parents=True)
    shutil.copy(mps_path, instance_dir / "equality.mps")

    transform_instance(instance_class, instance_name, cache_dir=tmp_path)
    out = instance_dir / "equality.std"
    assert out.is_file()


def test_transform_instance_file_not_found(tmp_path: Path) -> None:
    """transform_instance raises FileNotFoundError when instance subdir has no .mps."""
    with pytest.raises(FileNotFoundError, match="Expected exactly one .mps|Instance directory not found"):
        transform_instance("x", "nonexistent", cache_dir=tmp_path)


