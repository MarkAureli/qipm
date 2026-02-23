#!/usr/bin/env python3
"""Gate counts for quantum linear system algorithm (QLSA)."""

from __future__ import annotations

import math


def gate_count_qlsa(
    *,
    d: int,
    k: float,
    epsilon: float = 1e-1,
) -> int:
    """
    Return the QLSA Chebyshev query count.

    Args:
        d: Maximum sparsity (non-zeros per row or column) of M̂.
        k: Condition number (2-norm) of M̂.
        epsilon: Precision (default 1e-1).

    Returns:
        int: The number of queries that QLS Chebyshev makes to O_H and O_F (P_A).
    """
    binst = math.ceil(math.log(d * k / epsilon) * (d * k) ** 2)
    insqrt = binst * math.log(4 * binst / epsilon)
    j0_val = int(math.ceil(math.sqrt(insqrt)))
    return 8 * j0_val
