#!/usr/bin/env python3
"""Build linear systems of equations (e.g. modified NES from IPM)."""

from __future__ import annotations

import numpy as np
from scipy.linalg import lu_factor, lu_solve, qr
from scipy.sparse import spmatrix, diags as sp_diags, issparse


def _build_modified_nes_sparse(
    A: spmatrix,
    b: np.ndarray,
    c: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    s: np.ndarray,
    mu: float,
    sigma: float,
    reduce_with_basis: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Sparse path for build_modified_nes.

    Avoids densifying A and avoids forming the n×n diagonal matrix D² = diag(x/s).
    Uses scipy.sparse.diags for column scaling and sparse matrix products for M = A D² Aᵀ
    and all matrix–vector products in ω. Only the m×n scaled A (for QR basis selection)
    and the m×m basis submatrix A_B are densified.

    This makes large-instance computation feasible: for 80bau3b (m=21292) the dense path
    requires an n×n diagonal (>≈4×10⁸ entries) while this path never materialises it.
    """
    from scipy.sparse import csr_matrix

    A = csr_matrix(A, dtype=np.float64)
    m, n = A.shape
    d2 = x / s

    basis_P = None
    if reduce_with_basis:
        d_sqrt = np.sqrt(d2)
        # Column-scale A by d_sqrt via sparse diagonal; densify only the m×n result for QR.
        # This avoids building the n×n dense D_sqrt matrix used in the dense path.
        A_scaled = (A @ sp_diags(d_sqrt, 0, format="csr")).toarray()
        _, R, basis_P = qr(A_scaled, pivoting=True)
        r_diag = np.abs(np.diag(R))
        tol = max(A.shape) * np.finfo(float).eps * (np.max(r_diag) if r_diag.size else 1.0)
        effective_rank = int(np.sum(r_diag > tol))
        if effective_rank < m:
            # Row reduction: densify A only for the column-pivoted QR on A^T
            _, _, P_row = qr(A.toarray().T, pivoting=True)
            row_subset = P_row[:effective_rank]
            A = A[row_subset, :]
            b = b[row_subset]
            y = y[row_subset]
            m = effective_rank

    # M = A D² Aᵀ via sparse operations: avoids the n×n np.diag(d2) of the dense path.
    A_d2 = A @ sp_diags(d2, 0, format="csr")  # sparse, same nnz as A
    M = (A_d2 @ A.T).toarray()  # m×m dense result

    # ω = A(d2·c) − M y − σμ A(1/s) + b − A x  (sparse mat-vec throughout)
    omega = (
        np.asarray(A @ (d2 * c)).ravel()
        - M @ y
        - (sigma * mu) * np.asarray(A @ (1.0 / s)).ravel()
        + b
        - np.asarray(A @ x).ravel()
    )

    if not reduce_with_basis:
        return M, omega

    B = np.asarray(basis_P[:m], dtype=np.intp).ravel()
    x_B, s_B = x[B], s[B]
    d_B_inv = np.sqrt(s_B / x_B)
    D_B_inv = np.diag(d_B_inv)
    A_B = A[:, B].toarray()  # m×m dense submatrix
    A_B_inv = np.linalg.solve(A_B, np.eye(m))
    inner = A_B_inv @ M @ A_B_inv.T
    M_hat = D_B_inv @ inner @ D_B_inv
    omega_hat = D_B_inv @ (A_B_inv @ omega)
    return M_hat, omega_hat


def build_modified_nes(
    A: np.ndarray | spmatrix,
    b: np.ndarray,
    c: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    s: np.ndarray,
    mu: float | None = None,
    sigma: float = 1.0,
    reduce_with_basis: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build M̂ and ω̂ from the modified NES (12): M̂ z = ω̂.

    Notation (paper): X = diag(x), S = diag(s), D = X^{1/2} S^{-1/2};
    M = A D^2 A^T; ω = A D^2 c - M y - σμ A S^{-1} 1 + b - A x.
    Basis B ⊆ {0..n-1} with |B| = m s.t. A_B = A[:, B] is invertible (chosen via QR with column pivoting on the scaled A D, where D = diag(sqrt(x/s))).
    Â = A_B^{-1} A, b̂ = A_B^{-1} b; D_B = diag(x_B)^{1/2} diag(s_B)^{-1/2}.
    Then:
      M̂ = D_B^{-1} A_B^{-1} M (D_B^{-1} A_B^{-1})^T,
      ω̂ = D_B^{-1} A_B^{-1} ω.

    When A is a scipy sparse matrix the computation is dispatched to a sparse
    implementation that avoids densifying A and never forms the n×n diagonal
    matrix D² = diag(x/s), making large instances (e.g. m=21292) feasible.

    Parameters
    ----------
    A : (m, n) matrix
        Constraint matrix (full row rank). May be a dense ndarray or any scipy
        sparse matrix; sparse input triggers the memory-efficient sparse path.
    b : (m,) array
    c : (n,) array
    x, s : (n,) arrays
        Primal and dual slacks (x > 0, s > 0).
    y : (m,) array
    mu : float or None
        Barrier parameter. If None, μ = x^T s / n.
    sigma : float
        Centering parameter (default 1.0).
    reduce_with_basis : bool, optional
        If True (default), compute a basis B via QR with column pivoting, reduce
        A to full row rank if rank deficient, and return the basis-transformed
        M̂ and ω̂. If False, return M and ω directly without basis reduction.

    Returns
    -------
    M_hat : (m, m) array
        Modified constraint matrix in (12).
    omega_hat : (m,) array
        RHS in (12).
    """
    b = np.asarray(b, dtype=np.float64).ravel()
    c = np.asarray(c, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    s = np.asarray(s, dtype=np.float64).ravel()

    if isinstance(A, spmatrix):
        m, n = A.shape
    else:
        A = np.asarray(A) if not isinstance(A, np.ndarray) else A
        A = np.atleast_2d(A)
        m, n = A.shape

    if b.size != m or c.size != n or x.size != n or s.size != n or y.size != m:
        raise ValueError("A (m,n), b (m), c (n), x (n), y (m), s (n) size mismatch")

    if mu is None:
        mu = float(np.dot(x, s)) / n

    # Dispatch to sparse path when A is a scipy sparse matrix.
    if isinstance(A, spmatrix):
        return _build_modified_nes_sparse(A, b, c, x, y, s, mu, sigma, reduce_with_basis)

    # --- Dense path (unchanged) ---
    # M = A D^2 A^T, ω = A D^2 c - M y - σμ A S^{-1} 1 + b - A x (single place)
    d2 = x / s

    basis_P = None  # column permutation from first QR when reduce_with_basis
    if reduce_with_basis:
        # Reduce to full row rank if rank deficient; keep column perm P for basis.
        # QR on scaled A*d_sqrt selects a basis aligned with M = A D² Aᵀ,
        # minimising κ(M̂) compared to QR on unscaled A.
        d_sqrt = np.sqrt(d2)
        _, R, basis_P = qr(A * d_sqrt, pivoting=True)
        r_diag = np.abs(np.diag(R))
        tol = max(A.shape) * np.finfo(float).eps * (np.max(r_diag) if r_diag.size else 1.0)
        effective_rank = int(np.sum(r_diag > tol))
        if effective_rank < m:
            _, _, P_row = qr(A.T, pivoting=True)
            row_subset = P_row[:effective_rank]
            A = A[row_subset, :]
            b = b[row_subset]
            y = y[row_subset]
            m = effective_rank
    D2 = np.diag(d2)
    S_inv_one = 1.0 / s
    M = A @ D2 @ A.T
    omega = (
        A @ (d2 * c)
        - M @ y
        - (sigma * mu) * (A @ S_inv_one)
        + b
        - A @ x
    )

    if reduce_with_basis:
        # Basis B from column perm of first QR; return M̂, ω̂
        B = np.asarray(basis_P[:m].copy(), dtype=np.intp).ravel()
        x_B = x[B]
        s_B = s[B]
        d_B_inv = np.sqrt(s_B / x_B)
        A_B = A[:, B]
        A_B_inv = np.linalg.solve(A_B, np.eye(m))
        D_B_inv = np.diag(d_B_inv)
        inner = A_B_inv @ M @ A_B_inv.T
        M_hat = D_B_inv @ inner @ D_B_inv
        omega_hat = D_B_inv @ (A_B_inv @ omega)
        return M_hat, omega_hat
    else:
        return M, omega


def estimate_cond_mhat(
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
    d_sqrt = np.sqrt(x / s)
    if issparse(A):
        A_scaled = (A @ sp_diags(d_sqrt, 0, format="csr")).toarray()
    else:
        A_scaled = A * d_sqrt

    _, R, basis_P = qr(A_scaled, pivoting=True)
    r_diag = np.abs(np.diag(R))
    tol_rank = max(A.shape) * np.finfo(float).eps * (r_diag[0] if r_diag.size else 1.0)
    effective_rank = int(np.sum(r_diag > tol_rank))

    # --- Row reduction (consistent with build_modified_nes) ---
    if effective_rank < m:
        if issparse(A):
            _, _, P_row = qr(A.toarray().T, pivoting=True)
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
    lam_min, _ = eigsh(M_op, k=1, which="SM", **eigsh_kwargs)

    lam_max_val = float(lam_max[0])
    lam_min_val = float(lam_min[0])
    if lam_min_val <= 0.0:
        return float("inf")
    return lam_max_val / lam_min_val
