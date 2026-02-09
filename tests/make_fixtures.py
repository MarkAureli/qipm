#!/usr/bin/env python3
"""Generate expected standard-form .npz fixtures for test MPS instances.

Run from repo root: python tests/make_fixtures.py
Requires: numpy, scipy
"""

import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# min_sum.mps: min x1 + x2  s.t.  x1 + x2 <= 10,  x1,x2 >= 0
# Standard form: min c'x  s.t.  x1 + x2 + s = 10,  x1,x2,s >= 0
MIN_SUM = {
    "c": np.array([1.0, 1.0, 0.0]),
    "b": np.array([10.0]),
    "A": csr_matrix([[1.0, 1.0, 1.0]]),
}

# equality.mps: min x1  s.t.  x1 + x2 = 5,  x1,x2 >= 0
# Standard form: already Ax = b, x >= 0
EQUALITY = {
    "c": np.array([1.0, 0.0]),
    "b": np.array([5.0]),
    "A": csr_matrix([[1.0, 1.0]]),
}


def save_standard_npz(path: Path, c: np.ndarray, b: np.ndarray, A: csr_matrix) -> None:
    np.savez_compressed(
        path,
        c=c,
        b=b,
        A_data=A.data,
        A_indices=A.indices,
        A_indptr=A.indptr,
        A_shape=np.array(A.shape),
    )


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    d = MIN_SUM
    save_standard_npz(FIXTURES / "min_sum.npz", d["c"], d["b"], d["A"])
    d = EQUALITY
    save_standard_npz(FIXTURES / "equality.npz", d["c"], d["b"], d["A"])
    print("Wrote min_sum.npz and equality.npz in tests/fixtures/")


if __name__ == "__main__":
    main()
