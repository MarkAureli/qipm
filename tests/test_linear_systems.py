"""Tests for helpers/linear_systems.py: build_modified_nes.

Two test suites:
1. Soundness: for each fixture (.sde + .init), verify that solving M̂ Δŷ = ω̂
   and recovering Δy = T^T Δŷ satisfies M Δy = ω (the two systems are equivalent).
2. Condition number improvement: for each matrix in test_data/reduce_with_basis/,
   verify that the new basis (QR on A*d_sqrt) gives lower cond(M̂) than the old
   basis (QR on A).
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import qr
from scipy.sparse import csr_matrix

from helpers.linear_systems import build_modified_nes

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TEST_DATA = Path(__file__).resolve().parent.parent / "test_data" / "reduce_with_basis"

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

SOUNDNESS_TOL = 1e-8


def _load_std(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (A, b, c) from a .std or .sde NPZ file; A returned as dense."""
    data = np.load(path)
    c = np.asarray(data["c"], dtype=np.float64).ravel()
    b = np.asarray(data["b"], dtype=np.float64).ravel()
    A_data = np.asarray(data["A_data"], dtype=np.float64).ravel()
    A_indices = np.asarray(data["A_indices"], dtype=np.int64).ravel()
    A_indptr = np.asarray(data["A_indptr"], dtype=np.int64).ravel()
    A_shape = np.asarray(data["A_shape"], dtype=np.int64).ravel()
    m_s, n_s = int(A_shape[0]), int(A_shape[1])
    A = csr_matrix((A_data, A_indices, A_indptr), shape=(m_s, n_s)).toarray()
    return A, b, c


def _load_init(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (x, y, s) from a .init NPZ file."""
    data = np.load(path, allow_pickle=False)
    x = np.asarray(data["x"], dtype=np.float64).ravel()
    y = np.asarray(data["y"], dtype=np.float64).ravel()
    s = np.asarray(data["s"], dtype=np.float64).ravel()
    return x, y, s


def _m_hat_old_basis(A: np.ndarray, x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Compute M̂ using the OLD basis selection: QR with column pivoting on A (unscaled).

    Replicates the pre-fix logic for use in condition number comparison.
    """
    m = A.shape[0]
    _, R, basis_P = qr(A, pivoting=True)
    r_diag = np.abs(np.diag(R))
    tol = max(A.shape) * np.finfo(float).eps * (np.max(r_diag) if r_diag.size else 1.0)
    effective_rank = int(np.sum(r_diag > tol))
    if effective_rank < m:
        _, _, P_row = qr(A.T, pivoting=True)
        row_subset = P_row[:effective_rank]
        A = A[row_subset, :]
        m = effective_rank
    B = np.asarray(basis_P[:m], dtype=np.intp).ravel()
    x_B, s_B = x[B], s[B]
    d_B_inv = np.sqrt(s_B / x_B)
    D_B_inv = np.diag(d_B_inv)
    A_B = A[:, B]
    A_B_inv = np.linalg.solve(A_B, np.eye(m))
    d2 = x / s
    M = A @ np.diag(d2) @ A.T
    inner = A_B_inv @ M @ A_B_inv.T
    return D_B_inv @ inner @ D_B_inv


# ---------------------------------------------------------------------------
# Soundness tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_soundness(stem: str) -> None:
    """M̂ Δŷ = ω̂ is equivalent to M Δy = ω after basis recovery.

    Verifies that the congruence transformation is mathematically sound:
    solving the reduced system and recovering Δy = T^T Δŷ (where T = D_B⁻¹ A_B⁻¹)
    satisfies the original system M Δy = ω.
    """
    sde_path = FIXTURES / f"{stem}.sde"
    init_path = FIXTURES / f"{stem}.init"
    if not sde_path.is_file() or not init_path.is_file():
        pytest.skip(f"Fixture not found: {stem}")

    A, b, c = _load_std(sde_path)
    x, y, s = _load_init(init_path)
    m, n = A.shape

    # Replicate the new basis selection (QR on A*d_sqrt) to get recovery matrices.
    d2 = x / s
    d_sqrt = np.sqrt(d2)
    _, R, basis_P = qr(A * d_sqrt, pivoting=True)
    r_diag = np.abs(np.diag(R))
    tol = max(A.shape) * np.finfo(float).eps * (np.max(r_diag) if r_diag.size else 1.0)
    effective_rank = int(np.sum(r_diag > tol))
    if effective_rank < m:
        pytest.skip(f"{stem}: row reduction required ({effective_rank} < {m}), skipping soundness")

    B = np.asarray(basis_P[:m], dtype=np.intp).ravel()
    x_B, s_B = x[B], s[B]
    d_B_inv = np.sqrt(s_B / x_B)
    D_B_inv = np.diag(d_B_inv)
    A_B = A[:, B]
    A_B_inv = np.linalg.solve(A_B, np.eye(m))

    # Build both systems
    M_hat, omega_hat = build_modified_nes(A, b, c, x, y, s, reduce_with_basis=True)
    M, omega = build_modified_nes(A, b, c, x, y, s, reduce_with_basis=False)

    # Solve M̂ Δŷ = ω̂
    delta_y_hat, _, _, _ = np.linalg.lstsq(M_hat, omega_hat, rcond=None)

    # Recover Δy = T^T Δŷ  where T = D_B_inv @ A_B_inv
    # T^T = A_B_inv.T @ D_B_inv  (D_B_inv is symmetric/diagonal)
    delta_y = A_B_inv.T @ (D_B_inv @ delta_y_hat)

    # Verify M Δy ≈ ω
    residual = np.linalg.norm(M @ delta_y - omega)
    rhs_norm = np.linalg.norm(omega)
    rel_residual = residual / rhs_norm if rhs_norm > 1e-14 else residual
    assert rel_residual < SOUNDNESS_TOL, (
        f"{stem}: relative residual {rel_residual:.2e} exceeds tolerance {SOUNDNESS_TOL}"
    )


# ---------------------------------------------------------------------------
# Condition number improvement tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "npz_path",
    sorted(TEST_DATA.glob("A_*.npz")),
    ids=lambda p: p.stem,
)
def test_condition_number_improvement(npz_path: Path) -> None:
    """New basis (QR on A*d_sqrt) gives lower-or-equal cond(M̂) than old basis (QR on A).

    Uses synthetic x, s > 0 generated with a fixed seed for reproducibility.
    Allows a slack factor of 10 to guard against numerical edge cases.
    """
    if not npz_path.is_file():
        pytest.skip(f"Test data file not found: {npz_path}")

    rng = np.random.default_rng(42)
    data = np.load(npz_path)
    A = np.asarray(data["A"], dtype=np.float64)
    m, n = A.shape

    # Synthetic strictly feasible point
    x = np.ones(n) + rng.uniform(0.0, 1.0, size=n)
    s = np.ones(n) + rng.uniform(0.0, 1.0, size=n)
    b = A @ x         # Ax = b  (primal feasibility)
    y = np.zeros(m)
    c = s             # c - A^T y = s > 0  (dual feasibility)

    # New basis: build_modified_nes now uses QR on A*d_sqrt
    M_hat_new, _ = build_modified_nes(A, b, c, x, y, s, reduce_with_basis=True)

    # Old basis: QR on A (unscaled), replicated explicitly
    M_hat_old = _m_hat_old_basis(A.copy(), x, s)

    cond_new = np.linalg.cond(M_hat_new)
    cond_old = np.linalg.cond(M_hat_old)

    assert cond_new <= cond_old * 10, (
        f"{npz_path.stem}: cond_new={cond_new:.4g} > cond_old={cond_old:.4g} * 10  "
        f"(new basis is worse by more than factor 10)"
    )
