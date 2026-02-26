#!/usr/bin/env python3
"""Estimate condition number of the modified NES matrix M̂ from IPM."""

from __future__ import annotations

import numpy as np
from scipy.linalg import lu_factor, lu_solve, qr
from scipy.sparse import spmatrix, diags as sp_diags, issparse


def estimate_mnes_cond(
    A: np.ndarray | spmatrix,
    x: np.ndarray,
    s: np.ndarray,
    *,
    tol: float = 0.0,
    maxiter: int | None = None,
) -> float:
    """Estimate κ(M̂) without materialising the m×m matrix M̂.

    Uses the identity M̂ = I + F̄F̄ᵀ where F̄ = D_B⁻¹ A_B⁻¹ A_N D_N (m × (n−m)).
    M̂ is represented as a ``scipy.sparse.linalg.LinearOperator``
    (matvec: v → v + F̄(F̄ᵀv)) and ``scipy.sparse.linalg.eigsh`` is used to
    obtain λ_max and λ_min without forming M̂ as a dense array.

    M̂ is symmetric positive definite with λ_min ≥ 1 (it equals 1 whenever
    rank(F̄) < m, i.e. when n − m < m).  eigsh converges reliably for both
    extremes because all eigenvalues are ≥ 1 and bounded away from zero.

    The basis B is selected by QR with column pivoting on A·diag(√(x/s)),
    matching the logic in build_modified_nes.  Row reduction (if A is rank-
    deficient) is handled identically to build_modified_nes.

    Parameters
    ----------
    A : (m, n) dense ndarray or scipy sparse matrix
        Constraint matrix.
    x, s : (n,) arrays
        Strictly positive primal variables and dual slacks.
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

    x = np.asarray(x, dtype=np.float64).ravel()
    s = np.asarray(s, dtype=np.float64).ravel()

    if issparse(A):
        from scipy.sparse import csr_matrix
        A = csr_matrix(A, dtype=np.float64)
    else:
        A = np.asarray(A, dtype=np.float64)

    m, n = A.shape

    # --- Basis selection: QR with column pivoting on A·diag(√(x/s)) ---
    # For sparse A, use SuiteSparse SPQR (via sparseqr) to avoid the O(m×n) dense
    # allocation that (A @ sp_diags(d_sqrt)).toarray() would require.  SPQR uses
    # COLAMD/AMD column ordering instead of LAPACK DGEQP3, so the selected basis
    # will differ from the dense path but is equally valid.
    d_sqrt = np.sqrt(x / s)
    if issparse(A):
        import sparseqr
        A_scaled_sp = A @ sp_diags(d_sqrt, 0, format="csr")
        _, _, basis_P, effective_rank = sparseqr.qr(A_scaled_sp)
        basis_P = np.asarray(basis_P, dtype=np.intp)
    else:
        A_scaled = A * d_sqrt
        _, R, basis_P = qr(A_scaled, pivoting=True)
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

    # --- F̄ components ---
    d_B_inv = np.sqrt(s[B] / x[B])
    d_N = np.sqrt(x[N] / s[N])

    if issparse(A):
        from scipy.sparse.linalg import splu
        A_B_sparse = A[:, B]    # sparse m × m — keep sparse to avoid O(m²) densification
        A_N = A[:, N]           # sparse m × n_N
        # splu uses SuperLU with fill-reducing ordering; memory ~ O(nnz(A_B) × fill-factor)
        # rather than O(m²) for dense LU.  For netlib instances nnz(A_B)/m ≈ 2–15,
        # giving 100×–10 000× less memory than the dense LU factor.
        A_B_lu = splu(A_B_sparse.tocsc())

        def _fbar_mv(v: np.ndarray) -> np.ndarray:
            """F̄ v = D_B_inv * (A_B⁻¹ @ (A_N @ (D_N * v)))"""
            v = np.asarray(v, dtype=np.float64).ravel()
            return d_B_inv * A_B_lu.solve(np.asarray(A_N @ (d_N * v)).ravel())

        def _fbar_rmv(u: np.ndarray) -> np.ndarray:
            """F̄ᵀ u = D_N * (A_N.T @ (A_B⁻ᵀ @ (D_B_inv * u)))"""
            u = np.asarray(u, dtype=np.float64).ravel()
            return d_N * np.asarray(A_N.T @ A_B_lu.solve(d_B_inv * u, trans="T")).ravel()

    else:
        A_B = A[:, B]
        A_N = A[:, N]
        lu_fac = lu_factor(A_B)

        def _fbar_mv(v: np.ndarray) -> np.ndarray:
            """F̄ v = D_B_inv * (A_B⁻¹ @ (A_N @ (D_N * v)))"""
            v = np.asarray(v, dtype=np.float64).ravel()
            return d_B_inv * lu_solve(lu_fac, np.asarray(A_N @ (d_N * v)).ravel())

        def _fbar_rmv(u: np.ndarray) -> np.ndarray:
            """F̄ᵀ u = D_N * (A_N.T @ (A_B⁻ᵀ @ (D_B_inv * u)))"""
            u = np.asarray(u, dtype=np.float64).ravel()
            return d_N * np.asarray(A_N.T @ lu_solve(lu_fac, d_B_inv * u, trans=1)).ravel()

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
