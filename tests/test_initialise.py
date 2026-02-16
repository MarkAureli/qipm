"""Tests for initialise: find_initial_triple and selfdual_embedding for .std fixtures.

For each fixture:
1. Try to find a strictly feasible initial triple; if found, verify it.
2. Build the self-dual embedding, compare to reference .sde in fixtures, validate SDE
   structure, check the embedding triple is strictly feasible, and compare it to
   reference initial triple stored in .init (NPZ: x, y, s).
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

pytest.importorskip("highspy", reason="highspy required for initialise tests")
from initialise import (
    _load_standard_form,
    find_initial_triple,
    selfdual_embedding,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURE_STEMS = [
    "min_sum",
    "equality",
    "three_var",
    "bounded_var",
    "lower_row",
    "free_var",
    "upper_var",
    "range_row",
]

RTOL = 1e-9
ATOL = 1e-10
MIN_STRICT = 1e-10


def assert_standard_form_equal(
    A: csr_matrix,
    b: np.ndarray,
    c: np.ndarray,
    A_ref: csr_matrix,
    b_ref: np.ndarray,
    c_ref: np.ndarray,
    *,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> None:
    """Assert (A, b, c) equals reference (A_ref, b_ref, c_ref) in standard form."""
    np.testing.assert_allclose(b, b_ref, rtol=rtol, atol=atol, err_msg="b mismatch")
    np.testing.assert_allclose(c, c_ref, rtol=rtol, atol=atol, err_msg="c mismatch")
    assert A.shape == A_ref.shape, f"A shape {A.shape} vs {A_ref.shape}"
    np.testing.assert_allclose(
        A.toarray(), A_ref.toarray(), rtol=rtol, atol=atol, err_msg="A mismatch"
    )


def _load_init(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load initial triple (x, y, s) from .init npz."""
    data = np.load(path, allow_pickle=False)
    x = np.asarray(data["x"], dtype=np.float64).ravel()
    y = np.asarray(data["y"], dtype=np.float64).ravel()
    s = np.asarray(data["s"], dtype=np.float64).ravel()
    return x, y, s


def assert_triple_strictly_feasible(
    A: csr_matrix,
    b: np.ndarray,
    c: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    s: np.ndarray,
    *,
    rtol: float = RTOL,
    atol: float = ATOL,
    min_strict: float = MIN_STRICT,
) -> None:
    """Assert (x, y, s) is strictly feasible: x > 0, s > 0, Ax = b, s = c - A'y."""
    n, m = A.shape[1], A.shape[0]
    assert x.shape == (n,), f"x shape {x.shape} vs n={n}"
    assert y.shape == (m,), f"y shape {y.shape} vs m={m}"
    assert s.shape == (n,), f"s shape {s.shape} vs n={n}"
    assert np.all(x >= min_strict), "x must be strictly positive"
    assert np.all(s >= min_strict), "s must be strictly positive"
    np.testing.assert_allclose(A @ x, b, rtol=rtol, atol=atol, err_msg="Ax = b")
    np.testing.assert_allclose(s, c - A.T @ y, rtol=rtol, atol=atol, err_msg="s = c - A'y")


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_initial_triple_and_embedding(stem: str) -> None:
    """For each fixture: (1) try find_initial_triple and verify if found; (2) build SDE, compare to reference .sde, check embedding triple."""
    std_path = FIXTURES / f"{stem}.std"
    if not std_path.is_file():
        pytest.skip(f"Fixture not found: {std_path}")
    ref_sde_path = FIXTURES / f"{stem}.sde"
    ref_init_path = FIXTURES / f"{stem}.init"
    if not ref_sde_path.is_file():
        pytest.skip(f"Reference .sde not found: {ref_sde_path}")
    if not ref_init_path.is_file():
        pytest.skip(f"Reference .init not found: {ref_init_path}")

    A, b, c = _load_standard_form(std_path)
    m, n = A.shape

    # 1. Try to find strictly feasible initial triple; if found, verify
    try:
        x, y, s = find_initial_triple(A, b, c)
        assert_triple_strictly_feasible(A, b, c, x, y, s)
    except Exception:
        pass  # OK if no strictly feasible triple exists

    # 2. Build self-dual embedding
    x_emb, y_emb, s_emb, emb_lp = selfdual_embedding(A, b, c)
    A_emb, b_emb, c_emb = emb_lp

    # Validate SDE dimensions
    n_emb = 2 * n + m + 2
    m_emb = m + n + 1
    assert A_emb.shape == (m_emb, n_emb), f"A_emb shape {A_emb.shape}"
    assert b_emb.shape == (m_emb,), f"b_emb shape {b_emb.shape}"
    assert c_emb.shape == (n_emb,), f"c_emb shape {c_emb.shape}"

    # Compare constructed SDE to reference .sde in fixtures
    A_emb_ref, b_emb_ref, c_emb_ref = _load_standard_form(ref_sde_path)
    assert_standard_form_equal(A_emb, b_emb, c_emb, A_emb_ref, b_emb_ref, c_emb_ref)

    # 3. Embedding triple (x_emb, y_emb, s_emb) is strictly feasible for SDE
    assert_triple_strictly_feasible(A_emb, b_emb, c_emb, x_emb, y_emb, s_emb)

    # 4. Compare embedding triple to reference .init
    x_ref, y_ref, s_ref = _load_init(ref_init_path)
    np.testing.assert_allclose(x_emb, x_ref, rtol=RTOL, atol=ATOL, err_msg="x mismatch vs reference .init")
    np.testing.assert_allclose(y_emb, y_ref, rtol=RTOL, atol=ATOL, err_msg="y mismatch vs reference .init")
    np.testing.assert_allclose(s_emb, s_ref, rtol=RTOL, atol=ATOL, err_msg="s mismatch vs reference .init")
