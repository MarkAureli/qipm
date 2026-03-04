#!/usr/bin/env python3
"""Estimate condition number of the modified NES matrix M̂ from IPM."""

from __future__ import annotations

import numpy as np
from scipy.linalg import lu_factor, lu_solve, qr
from scipy.sparse import spmatrix, issparse


def estimate_mnes_cond(
    A: np.ndarray | spmatrix,
    *,
    tol: float = 0.0,
    maxiter: int | None = None,
) -> float:
    """Estimate κ(M̂) without materialising the m×m matrix M̂.

    Uses the identity M̂ = I + F̄F̄ᵀ where F̄ = A_B⁻¹ A_N (m × (n−m)),
    i.e. x and s are replaced by identity (D_B = D_N = I).
    M̂ is represented as a ``scipy.sparse.linalg.LinearOperator``
    (matvec: v → v + F̄(F̄ᵀv)) and ``scipy.sparse.linalg.eigsh`` is used to
    obtain λ_max and λ_min without forming M̂ as a dense array.

    M̂ is symmetric positive definite with λ_min ≥ 1 (it equals 1 whenever
    rank(F̄) < m, i.e. when n − m < m).  eigsh converges reliably for both
    extremes because all eigenvalues are ≥ 1 and bounded away from zero.

    The basis B is selected by QR with column pivoting on A.  Row reduction
    (if A is rank-deficient) is handled identically to build_modified_nes.

    Parameters
    ----------
    A : (m, n) dense ndarray or scipy sparse matrix
        Constraint matrix.
    tol : float
        ARPACK tolerance for eigsh (0 = machine precision).
    maxiter : int or None
        Maximum ARPACK iterations (None = ARPACK default).

    Returns
    -------
    kappa : float
        2-norm condition number of M̂.
    """
    from scipy.sparse.linalg import LinearOperator, eigsh

    if issparse(A):
        from scipy.sparse import csr_matrix
        A = csr_matrix(A, dtype=np.float64)
    else:
        A = np.asarray(A, dtype=np.float64)

    m, n = A.shape

    # --- Basis selection: QR with column pivoting on A (D = I, so no scaling) ---
    # For sparse A, use SuiteSparse SPQR (via sparseqr).
    if issparse(A):
        import sparseqr
        _, _, basis_P, effective_rank = sparseqr.qr(A)
        basis_P = np.asarray(basis_P, dtype=np.intp)
    else:
        _, R, basis_P = qr(A, pivoting=True)
        r_diag = np.abs(np.diag(R))
        tol_rank = max(A.shape) * np.finfo(float).eps * (r_diag[0] if r_diag.size else 1.0)
        effective_rank = int(np.sum(r_diag > tol_rank))

    # --- Row reduction (consistent with build_modified_nes) ---
    if effective_rank < m:
        if issparse(A):
            _, _, P_row, _ = sparseqr.qr(A.T)
            P_row = np.asarray(P_row, dtype=np.intp)
        else:
            _, _, P_row = qr(A.T, pivoting=True)
        row_subset = P_row[:effective_rank]
        A = A[row_subset, :]
        m = effective_rank

    # --- Basis and non-basis column indices ---
    B = np.asarray(basis_P[:m], dtype=np.intp)
    N_mask = np.ones(n, dtype=bool)
    N_mask[B] = False
    N = np.where(N_mask)[0]
    n_N = len(N)

    # n_N = 0 → F̄ has no columns → M̂ = I → κ = 1
    if n_N == 0 or m <= 1:
        return 1.0

    # --- F̄ components (D_B = D_N = I) ---
    if issparse(A):
        from scipy.sparse.linalg import splu
        A_B_sparse = A[:, B]    # sparse m × m — keep sparse to avoid O(m²) densification
        A_N = A[:, N]           # sparse m × n_N
        A_B_lu = splu(A_B_sparse.tocsc())

        def _fbar_mv(v: np.ndarray) -> np.ndarray:
            """F̄ v = A_B⁻¹ @ (A_N @ v)"""
            v = np.asarray(v, dtype=np.float64).ravel()
            return A_B_lu.solve(np.asarray(A_N @ v).ravel())

        def _fbar_rmv(u: np.ndarray) -> np.ndarray:
            """F̄ᵀ u = A_N.T @ (A_B⁻ᵀ @ u)"""
            u = np.asarray(u, dtype=np.float64).ravel()
            return np.asarray(A_N.T @ A_B_lu.solve(u, trans="T")).ravel()

    else:
        A_B = A[:, B]
        A_N = A[:, N]
        lu_fac = lu_factor(A_B)

        def _fbar_mv(v: np.ndarray) -> np.ndarray:
            """F̄ v = A_B⁻¹ @ (A_N @ v)"""
            v = np.asarray(v, dtype=np.float64).ravel()
            return lu_solve(lu_fac, np.asarray(A_N @ v).ravel())

        def _fbar_rmv(u: np.ndarray) -> np.ndarray:
            """F̄ᵀ u = A_N.T @ (A_B⁻ᵀ @ u)"""
            u = np.asarray(u, dtype=np.float64).ravel()
            return np.asarray(A_N.T @ lu_solve(lu_fac, u, trans=1)).ravel()

    def _mhat_mv(v: np.ndarray) -> np.ndarray:
        """M̂ v = v + F̄(F̄ᵀ v)"""
        v = np.asarray(v, dtype=np.float64).ravel()
        return v + _fbar_mv(_fbar_rmv(v))

    M_op = LinearOperator((m, m), matvec=_mhat_mv, dtype=np.float64)

    eigsh_kwargs: dict = {"tol": tol}
    if maxiter is not None:
        eigsh_kwargs["maxiter"] = maxiter

    lam_max, _ = eigsh(M_op, k=1, which="LM", **eigsh_kwargs)
    lam_max_val = float(lam_max[0])

    # When n_N < m, rank(F̄) ≤ n_N < m so F̄F̄ᵀ has a nullspace of dimension
    # ≥ m − n_N and λ_min(M̂) = 1 exactly.  The SM eigsh would need to
    # converge to an eigenvalue of exactly 1 amid many degenerate copies,
    # which ARPACK reliably fails to do on large matrices.  Skip it.
    if n_N < m:
        lam_min_val = 1.0
    else:
        lam_min, _ = eigsh(M_op, k=1, which="SM", **eigsh_kwargs)
        lam_min_val = float(lam_min[0])

    if lam_min_val <= 0.0:
        return float("inf")
    return lam_max_val / lam_min_val
