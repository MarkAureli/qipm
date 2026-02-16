#!/usr/bin/env python3
"""Gate counts for quantum linear system algorithm (QLSA)."""

from __future__ import annotations

import numpy as np


#######################################################################################################################
#                                                       QLS Chebyshev
#######################################################################################################################
def qls_chebyshev_queries(
    alpha: float, x_norm: float, d: int, k: float, epsilon: float
) -> int:
    """
    Compute the number of queries that QLS Chebyshev makes to O_H and O_F (P_A).

    Args:
        alpha (float): Parameter used to compute the success probabilities p and p0.
        x_norm (float): Norm of the solution vector.
        d (int): Maximum sparsity.
        k (float): Condition number.
        epsilon (float): Precision.

    Returns:
        int: The number of queries that QLS Chebyshev makes to O_H and O_F (P_A).
    """
    p0 = 1 / alpha ** 2
    p = (x_norm / alpha) ** 2

    j0inst = j0(d * k, epsilon)
    QPa = 8 * j0inst

    return int(amplitude_amplification(p, p0) * QPa)


def compute_alpha_qls_chebyshev(d: int, k: float, epsilon: float) -> float:
    """
    Compute the alpha parameter for QLS-Chebyshev.

    Args:
        d (int): Maximum sparsity per row/column.
        k (float): Condition number.
        epsilon (float): Precision.

    Returns:
        float: Computed alpha parameter.
    """
    log_term = np.log2(d * k / epsilon)

    s = int(np.ceil(log_term * (d * k) ** 2))
    j0_val = int(
        np.ceil(
            d * k * np.sqrt(log_term * np.log2(log_term * 4 * (d * k) ** 2 / epsilon)),
        )
    )

    i_arr = np.arange(s)
    eta_i = np.prod((s - 0.5 - i_arr) / (s - i_arr))

    alpha = 2 * (j0_val + 1) * (1 - eta_i) / d

    for i in range(1, j0_val + 1):
        eta_i *= (s - i + 1) / (s + i)
        alpha -= 4 * (j0_val + 1 - i) * eta_i / d

    return alpha


#######################################################################################################################
#                                          Helping functions (Chebyshev)
#######################################################################################################################


def j0(k: float, epsilon: float) -> float:
    """Compute bound on j0 by computing a bound on b (binst)."""
    binst = np.ceil(np.log(k / epsilon) * k ** 2)
    insqrt = binst * np.log(4 * binst / epsilon)
    return np.ceil(np.sqrt(insqrt))


def infinisum(f, start: int = 0, epsilon: float = 1e-1) -> float:
    """
    Compute an infinite sum of a function with a given precision.

    Args:
        f (callable): Function to compute the infinite sum in quantum amplitude amplification.
        start (int): Starting point for the sum calculation.
        epsilon (float): Precision for stopping criterion.

    Returns:
        float: Result of the infinite sum.
    """
    n, res = start, f(0)
    while True:
        lo, hi = 2**n, 2 ** (n + 1)
        term_sum = sum(f(k) for k in range(lo, hi))
        if abs(term_sum) < epsilon:
            break
        n, res = n + 1, res + term_sum
    return res


def product_in_qaa(
    theta: float,
    k: int,
    p0: float,
    *,
    sin_2_theta: float | None = None,
    p0_inv_sqrt: float | None = None,
) -> float:
    """
    Compute the product term for Quantum Amplitude Amplification (QAA).

    Args:
        theta (float): Parameter derived from success_probability.
        k (int): Condition number.
        p0 (float): Lower bound on success probability.
        sin_2_theta: Precomputed sin(2*theta); computed if None.
        p0_inv_sqrt: Precomputed p0**(-1/2); computed if None.

    Returns:
        float: Product term value.
    """
    if sin_2_theta is None:
        sin_2_theta = np.sin(2 * theta)
    if p0_inv_sqrt is None:
        p0_inv_sqrt = p0 ** (-0.5)
    l_values = np.arange(1, k, dtype=np.float64)
    if l_values.size == 0:
        return 1.0
    pow_vals = np.power(1.2, l_values)
    min_vals = np.minimum(pow_vals, p0_inv_sqrt)
    denom = sin_2_theta * 4 * min_vals
    terms = 0.5 + np.sin(4 * theta * (1 + min_vals)) / denom
    return float(np.prod(terms))


def amplitude_amplification(p: float, p0: float) -> float:
    """
    The expected number of times quantum amplitude amplification is called.

    Args:
        p (float): Success probability.
        p0 (float): Lower bound on success probability.

    Returns:
        float: The expected number of times quantum amplitude amplification is called.
    """
    theta = np.arcsin(np.sqrt(p))
    sin_2_theta = np.sin(2 * theta)
    p0_inv_sqrt = p0 ** (-0.5)

    def term(k: int) -> float:
        return min((1.2) ** k, p0_inv_sqrt) * product_in_qaa(
            theta, k, p0, sin_2_theta=sin_2_theta, p0_inv_sqrt=p0_inv_sqrt
        )

    return infinisum(term, start=1)


def gate_count_qlsa(A: np.ndarray, b: np.ndarray, epsilon: float = 1e-1) -> int:
    """
    Return the QLSA Chebyshev query count for the linear system A x = b.

    Computes sparsity and condition number of A, solves A x = b for x_norm,
    then returns the number of queries to O_H and O_F (P_A) as in qls_chebyshev_queries.

    Args:
        A: Square matrix of the linear system.
        b: Right-hand side vector.
        epsilon: Precision (default 1e-1).

    Returns:
        int: The number of queries that QLS Chebyshev makes to O_H and O_F (P_A).
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).ravel()
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    if b.size != A.shape[0]:
        raise ValueError("b size must match A shape")

    # Maximum sparsity (non-zeros per row or column)
    nz_row = np.count_nonzero(A, axis=1)
    nz_col = np.count_nonzero(A, axis=0)
    d = int(max(nz_row.max(), nz_col.max()))

    # Condition number (2-norm)
    k = float(np.linalg.cond(A))

    # Solve A x = b and get ||x||
    x = np.linalg.solve(A, b)
    x_norm = float(np.linalg.norm(x))

    alpha = compute_alpha_qls_chebyshev(d, k, epsilon)
    return qls_chebyshev_queries(alpha, x_norm, d, k, epsilon)
