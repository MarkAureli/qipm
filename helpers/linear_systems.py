#!/usr/bin/env python3
"""Estimate condition number of the modified NES matrix M̂ from IPM."""

from __future__ import annotations

import numpy as np
from scipy.sparse import spmatrix, csr_matrix, issparse


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

    The basis B is selected by SuiteSparse SPQR (via sparseqr) with column
    pivoting on A.  Row reduction (if A is rank-deficient) is handled via a
    second SPQR call on Aᵀ.

    Parameters
    ----------
    A : (m, n) dense ndarray or scipy sparse matrix
        Constraint matrix.  Dense arrays are converted to CSR internally.
    tol : float
        ARPACK tolerance for eigsh (0 = machine precision).
    maxiter : int or None
        Maximum ARPACK iterations (None = ARPACK default).

    Returns
    -------
    kappa : float
        2-norm condition number of M̂.
    """
    import sparseqr
    from scipy.sparse.linalg import LinearOperator, eigsh, splu

    A = csr_matrix(A, dtype=np.float64) if not issparse(A) else csr_matrix(A, dtype=np.float64)

    m, n = A.shape

    # --- Basis selection: SPQR with column pivoting on A ---
    _, _, basis_P, effective_rank = sparseqr.qr(A)
    basis_P = np.asarray(basis_P, dtype=np.intp)

    # --- Row reduction (if A is rank-deficient) ---
    if effective_rank < m:
        _, _, P_row, _ = sparseqr.qr(A.T)
        P_row = np.asarray(P_row, dtype=np.intp)
        A = A[P_row[:effective_rank], :]
        m = effective_rank

    # --- Basis and non-basis column indices ---
    B = basis_P[:m]
    N_mask = np.ones(n, dtype=bool)
    N_mask[B] = False
    N = np.where(N_mask)[0]
    n_N = len(N)

    # n_N = 0 → F̄ has no columns → M̂ = I → κ = 1
    if n_N == 0 or m <= 1:
        return 1.0

    # --- F̄ components: A_B_lu for triangular solves, A_N for matvecs ---
    A_B_lu = splu(A[:, B].tocsc())
    A_N = A[:, N]

    def _fbar_mv(v: np.ndarray) -> np.ndarray:
        """F̄ v = A_B⁻¹ @ (A_N @ v)"""
        return A_B_lu.solve(np.asarray(A_N @ v, dtype=np.float64).ravel())

    def _fbar_rmv(u: np.ndarray) -> np.ndarray:
        """F̄ᵀ u = A_N.T @ (A_B⁻ᵀ @ u)"""
        return np.asarray(A_N.T @ A_B_lu.solve(u, trans="T"), dtype=np.float64).ravel()

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


def estimate_oss_cond(
    A: np.ndarray | spmatrix,
    x: np.ndarray,
    s: np.ndarray,
    *,
    tol: float = 0.0,
    maxiter: int | None = None,
) -> float:
    """Estimate κ(M) for the OSS matrix M = [-XAᵀ  SV] ∈ ℝⁿˣⁿ.

    The Orthogonal Subspaces System (OSS) arises in the IF-IPM of
    Mohammadisiahroudi et al. (2025).  Given a feasible interior point
    (x, s) > 0 the system matrix is

        M = [-XAᵀ  SV]

    where X = diag(x), S = diag(s), and V ∈ ℝⁿˣ⁽ⁿ⁻ᵐ⁾ is a null-space basis
    of A built from the SPQR pivot basis B:

        V[B, :] = -A_B⁻¹ A_N,   V[N, :] = I_{n-m}.

    M is represented as a ``LinearOperator`` and κ(M) = σ_max / σ_min is
    estimated via ``svds`` without materialising M as a dense array.

    Parameters
    ----------
    A : (m, n) dense ndarray or scipy sparse matrix
        Constraint matrix.
    x : (n,) array
        Primal interior point (x > 0).
    s : (n,) array
        Dual slack interior point (s > 0).
    tol : float
        ARPACK tolerance for svds (0 = machine precision).
    maxiter : int or None
        Maximum ARPACK iterations (None = ARPACK default).

    Returns
    -------
    kappa : float
        2-norm condition number of M.
    """
    import sparseqr
    from scipy.sparse.linalg import LinearOperator, svds, splu

    A = csr_matrix(A, dtype=np.float64) if not issparse(A) else csr_matrix(A, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64).ravel()
    s = np.asarray(s, dtype=np.float64).ravel()

    m, n = A.shape

    if n <= 1:
        return 1.0

    # --- Basis selection: SPQR with column pivoting on A ---
    _, _, basis_P, effective_rank = sparseqr.qr(A)
    basis_P = np.asarray(basis_P, dtype=np.intp)

    # --- Row reduction (if A is rank-deficient) ---
    if effective_rank < m:
        _, _, P_row, _ = sparseqr.qr(A.T)
        P_row = np.asarray(P_row, dtype=np.intp)
        A = A[P_row[:effective_rank], :]
        m = effective_rank

    # --- Basis and non-basis column indices ---
    B = basis_P[:m]
    N_mask = np.ones(n, dtype=bool)
    N_mask[B] = False
    N = np.where(N_mask)[0]
    n_N = len(N)

    s_B = s[B]
    s_N = s[N]

    A_B_lu = splu(A[:, B].tocsc())
    A_N = A[:, N]

    # M @ z  where  z = [z_y (m);  z_λ (n_N)]
    # M z = -XAᵀ z_y + SV z_λ
    #   at B indices: -x_B*(Aᵀ z_y)_B  -  s_B * A_B⁻¹ A_N z_λ
    #   at N indices: -x_N*(Aᵀ z_y)_N  +  s_N * z_λ
    def _matvec(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64).ravel()
        z_y = z[:m]
        z_lam = z[m:]

        out = -(x * np.asarray(A.T @ z_y, dtype=np.float64).ravel())
        if n_N > 0:
            sv = np.empty(n, dtype=np.float64)
            sv[B] = -s_B * A_B_lu.solve(np.asarray(A_N @ z_lam, dtype=np.float64).ravel())
            sv[N] = s_N * z_lam
            out += sv
        return out

    # Mᵀ @ u  where  u ∈ ℝⁿ,  result ∈ ℝⁿ  (m + n_N components)
    # First m:   (Mᵀ u)_y = -AX u = -A @ (x * u)
    # Last n_N:  (Mᵀ u)_λ = Vᵀ S u = -A_Nᵀ A_B⁻ᵀ (s_B * u_B) + s_N * u_N
    def _rmatvec(u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64).ravel()

        out = np.empty(n, dtype=np.float64)
        out[:m] = -np.asarray(A @ (x * u), dtype=np.float64).ravel()
        if n_N > 0:
            solved = A_B_lu.solve(s_B * u[B], trans="T")
            out[m:] = -np.asarray(A_N.T @ solved, dtype=np.float64).ravel() + s_N * u[N]
        return out

    M_op = LinearOperator((n, n), matvec=_matvec, rmatvec=_rmatvec, dtype=np.float64)

    svds_kwargs: dict = {"tol": tol, "return_singular_vectors": False}
    if maxiter is not None:
        svds_kwargs["maxiter"] = maxiter

    sv_max = float(svds(M_op, k=1, which="LM", **svds_kwargs)[0])
    sv_min = float(svds(M_op, k=1, which="SM", **svds_kwargs)[0])

    if sv_min <= 0.0:
        return float("inf")
    return sv_max / sv_min
