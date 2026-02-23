#!/usr/bin/env python3
"""Build linear systems of equations (e.g. modified NES from IPM)."""

from __future__ import annotations

import numpy as np
from scipy.linalg import qr
from scipy.sparse import spmatrix


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
    Basis B ⊆ {0..n-1} with |B| = m s.t. A_B = A[:, B] is invertible (chosen via QR with column pivoting).
    Â = A_B^{-1} A, b̂ = A_B^{-1} b; D_B = diag(x_B)^{1/2} diag(s_B)^{-1/2}.
    Then:
      M̂ = D_B^{-1} A_B^{-1} M (D_B^{-1} A_B^{-1})^T,
      ω̂ = D_B^{-1} A_B^{-1} ω.

    Parameters
    ----------
    A : (m, n) matrix
        Constraint matrix (full row rank).
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
    if isinstance(A, spmatrix):
        A = A.toarray()
    A = np.asarray(A) if not isinstance(A, np.ndarray) else A
    A = np.atleast_2d(A)
    b = np.asarray(b, dtype=np.float64).ravel()
    c = np.asarray(c, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    s = np.asarray(s, dtype=np.float64).ravel()

    m, n = A.shape
    if b.size != m or c.size != n or x.size != n or s.size != n or y.size != m:
        raise ValueError("A (m,n), b (m), c (n), x (n), y (m), s (n) size mismatch")

    if mu is None:
        mu = float(np.dot(x, s)) / n

    basis_P = None  # column permutation from first QR when reduce_with_basis
    if reduce_with_basis:
        # Reduce to full row rank if rank deficient; keep column perm P for basis
        _, R, basis_P = qr(A, pivoting=True)
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

    # M = A D^2 A^T, ω = A D^2 c - M y - σμ A S^{-1} 1 + b - A x (single place)
    d2 = x / s
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
