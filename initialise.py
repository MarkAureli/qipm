#!/usr/bin/env python3
"""Find an initial triple (x, y, s) with x > 0, s > 0 for standard-form LPs; write to .init (npz format). On failure, use self-dual embedding."""

from __future__ import annotations

import warnings
from pathlib import Path

import highspy
import numpy as np
from scipy.linalg import lstsq, null_space
from scipy.sparse import csr_matrix, eye, hstack, vstack
from tqdm import tqdm

try:
    _HIGHS_INF = highspy.kHighsInf
except AttributeError:
    _HIGHS_INF = 1e30

# Minimum strict positivity for x and s
_STRICT_DELTA = 1e-8


def _load_standard_form(path: Path) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    """Load (A, b, c) from .std standard-form LP. A returned as CSR."""
    data = np.load(path)
    c = np.asarray(data["c"], dtype=np.float64).ravel()
    b = np.asarray(data["b"], dtype=np.float64).ravel()
    A_data = np.asarray(data["A_data"], dtype=np.float64).ravel()
    A_indices = np.asarray(data["A_indices"], dtype=np.int64).ravel()
    A_indptr = np.asarray(data["A_indptr"], dtype=np.int64).ravel()
    A_shape = np.asarray(data["A_shape"], dtype=np.int64).ravel()
    m, n = int(A_shape[0]), int(A_shape[1])
    A = csr_matrix((A_data, A_indices, A_indptr), shape=(m, n))
    return A, b, c


def _find_dual_feasible_strict(A: csr_matrix, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find y and s = c - A'y with s > 0 (elementwise). Returns (y, s). Raises if not found."""
    m, n = A.shape
    delta = _STRICT_DELTA
    if np.all(c > delta):
        y = np.zeros(m, dtype=np.float64)
        s = c.copy()
        return y, s
    # Solve LP: find y such that A'y <= c - delta (so s = c - A'y >= delta).
    # HiGHS: min 0, s.t. (A') y <= c - delta. Vars: y (m), rows: n.
    A_t = A.T.tocsr()
    row_upper = (c - delta).astype(np.float64)
    row_lower = np.full(n, -_HIGHS_INF, dtype=np.float64)
    col_cost = np.zeros(m, dtype=np.float64)
    col_lower = np.full(m, -_HIGHS_INF, dtype=np.float64)
    col_upper = np.full(m, _HIGHS_INF, dtype=np.float64)
    h = highspy.Highs()
    h.setOptionValue("log_to_console", False)
    h.addVars(m, col_lower, col_upper)
    h.changeColsCost(m, np.arange(m, dtype=np.int64), col_cost)
    num_nz = int(A_t.nnz)
    starts = np.asarray(A_t.indptr[:-1], dtype=np.int32) if n > 0 else np.array([], dtype=np.int32)
    h.addRows(
        n,
        row_lower,
        row_upper,
        num_nz,
        starts,
        A_t.indices.astype(np.int32),
        A_t.data.astype(np.float64),
    )
    status = h.run()
    if status != highspy.HighsStatus.kOk and status != highspy.HighsStatus.kWarning:
        raise RuntimeError("LP for dual feasible (y, s) failed")
    sol = h.getSolution()
    y = np.asarray(sol.col_value, dtype=np.float64).ravel()
    s = c - A_t @ y
    if not np.all(s >= delta):
        raise RuntimeError("Dual slack s not strictly positive after LP")
    # Ensure stored s is strictly positive (LP may return s_i = delta)
    s = np.maximum(s, 2 * delta)
    return y, s


def _find_primal_feasible_strict(A: csr_matrix, b: np.ndarray, n: int) -> np.ndarray:
    """Find x > 0 with Ax = b. Raises if not found."""
    m = A.shape[0]
    delta = _STRICT_DELTA
    # Minimum-norm solution to Ax = b (dense for lstsq).
    Ad = A.toarray()
    x0, residues, rank, sing = lstsq(Ad, b)
    x0 = np.asarray(x0, dtype=np.float64).ravel()
    if np.all(x0 > delta):
        return x0
    # Try to shift by null space: x = x0 + N @ alpha, require x >= delta.
    N = null_space(Ad)
    if N.size == 0:
        raise RuntimeError("No null space; unique solution not strictly positive")
    N = np.asarray(N, dtype=np.float64)
    # N is (n, k). We need alpha (k,) such that x0 + N @ alpha >= delta.
    k = N.shape[1]
    # LP: min 0 s.t. x0 + N @ alpha >= delta. So N @ alpha >= delta - x0.
    # Variables: alpha (k). Constraints: N.T is (k, n), we need (N.T).T @ alpha = N @ alpha >= delta - x0.
    # So constraint matrix in HiGHS: one row per component: row_j = N[j, :], lower = (delta - x0)_j, upper = inf.
    # So we have n constraints, k variables. Matrix: (n, k) = N.T (rows are constraints). So matrix is N (as n x k).
    row_lower = (delta - x0).astype(np.float64)
    row_upper = np.full(n, _HIGHS_INF, dtype=np.float64)
    col_cost = np.zeros(k, dtype=np.float64)
    col_lower = np.full(k, -_HIGHS_INF, dtype=np.float64)
    col_upper = np.full(k, _HIGHS_INF, dtype=np.float64)
    N_csr = csr_matrix(N)
    h = highspy.Highs()
    h.setOptionValue("log_to_console", False)
    h.addVars(k, col_lower, col_upper)
    h.changeColsCost(k, np.arange(k, dtype=np.int64), col_cost)
    num_nz = int(N_csr.nnz)
    starts = np.asarray(N_csr.indptr[:-1], dtype=np.int32) if n > 0 else np.array([], dtype=np.int32)
    h.addRows(
        n,
        row_lower,
        row_upper,
        num_nz,
        starts,
        N_csr.indices.astype(np.int32),
        N_csr.data.astype(np.float64),
    )
    status = h.run()
    if status != highspy.HighsStatus.kOk and status != highspy.HighsStatus.kWarning:
        raise RuntimeError("LP for primal strictly feasible x failed")
    sol = h.getSolution()
    alpha = np.asarray(sol.col_value, dtype=np.float64).ravel()
    x = x0 + N @ alpha
    if not np.all(x >= delta):
        raise RuntimeError("Primal x not strictly positive after null-space shift")
    x = np.maximum(x, 2 * delta)
    return x


def find_initial_triple(
    A: csr_matrix,
    b: np.ndarray,
    c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, y, s) with x > 0, s > 0, Ax = b, and s = c - A'y.

    Raises if such a triple cannot be found.
    """
    m, n = A.shape
    raise RuntimeError("Skip for testing purposes")
    y, s = _find_dual_feasible_strict(A, c)
    x = _find_primal_feasible_strict(A, b, n)
    return x, y, s


def _find_initial_triple_clique(
    A: csr_matrix,
    b: np.ndarray,
    c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clique specialisation: x = 0.25 for all primal variables; (y, s) satisfy dual constraint s = c - A'y with s > 0."""
    _, n = A.shape
    x = np.full(n, 0.25, dtype=np.float64)
    y, s = _find_dual_feasible_strict(A, c)
    return x, y, s


def selfdual_embedding(
    A: csr_matrix,
    b: np.ndarray,
    c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[csr_matrix, np.ndarray, np.ndarray] | None]:
    """Return an initial triple (x, y, s) and the embedded LP via the self-dual embedding.

    Builds the homogeneous self-dual embedding in standard form:
      min c_emb' x_emb  s.t.  A_emb x_emb = b_emb,  x_emb >= 0
    with variable vector x_emb = (x, τ, y, s, κ). Chooses b_emb = A_emb @ z0 so that
    z0 = (e, 1, 0, e, 1) is primal feasible; uses c_emb = 1 so that ỹ = 0 yields
    s_emb = c_emb > 0. Returns (x, y, s) grouped to match the (x, y, s) pattern:
    x = primal point, y = dual multipliers, s = dual slack for the embedded problem.
    """
    m, n = A.shape
    b = np.asarray(b, dtype=np.float64).ravel()
    c = np.asarray(c, dtype=np.float64).ravel()
    assert b.size == m and c.size == n

    # Variable layout: x_emb = (x [n], τ [1], y [m], s [n], κ [1]); total n_emb = 2*n + m + 2
    # Constraints: (1) Ax - b*τ = 0 [m]; (2) -A'y + c*τ - s = 0 [n]; (3) b'y - c'x + κ = 0 [1]
    n_emb = 2 * n + m + 2
    m_emb = m + n + 1

    # Block rows for A_emb (m_emb x n_emb)
    # Row block 1 (m rows): [A, -b, 0, 0, 0]
    A_csr = A.tocsr()
    col_b = csr_matrix(-b.reshape(-1, 1))
    col_zeros_m = csr_matrix((m, m))
    col_zeros_n = csr_matrix((m, n))
    col_zeros_1 = csr_matrix((m, 1))
    B1 = hstack([A_csr, col_b, col_zeros_m, col_zeros_n, col_zeros_1], format="csr")

    # Row block 2 (n rows): [0, c, -A', -I, 0]  (coefficients on x, τ, y, s, κ)
    zeros_nn = csr_matrix((n, n))
    col_c = csr_matrix(c.reshape(-1, 1))
    neg_I = -eye(n, format="csr")
    B2 = hstack([zeros_nn, col_c, -A_csr.T, neg_I, csr_matrix((n, 1))], format="csr")

    # Row block 3 (1 row): [c', 0, -b', 0, 1]  (literature convention)
    row_c = csr_matrix(c.reshape(1, -1))
    row_b = csr_matrix((-b).reshape(1, -1))
    row_zeros_n = csr_matrix((1, n))
    B3 = hstack(
        [row_c, csr_matrix([[0.0]]), row_b, row_zeros_n, csr_matrix([[1.0]])],
        format="csr",
    )

    A_emb = vstack([B1, B2, B3], format="csr")

    # Strictly feasible point: z0 = (x0, τ0, y0, s0, κ0) with all entries > 0
    x0 = np.ones(n, dtype=np.float64)
    tau0 = 1.0
    y0 = np.full(m, _STRICT_DELTA, dtype=np.float64)
    s0 = np.ones(n, dtype=np.float64)
    kappa0 = 1.0
    z0 = np.concatenate([x0, [tau0], y0, s0, [kappa0]])
    assert z0.size == n_emb

    # Set b_emb so that A_emb @ z0 = b_emb (primal feasible)
    b_emb = A_emb @ z0

    # Objective: c_emb = 1 so that (z0, ỹ=0, s_emb=c_emb) has s_emb > 0
    c_emb = np.ones(n_emb, dtype=np.float64)

    # Group as (x, y, s) for the embedded problem: x = z0, y = 0, s = c_emb - A_emb' @ 0 = c_emb
    x = z0
    y_emb = np.zeros(m_emb, dtype=np.float64)
    s = c_emb - A_emb.T @ y_emb  # = c_emb
    assert np.all(s > 0) and np.all(x > 0)

    return x, y_emb, s, (A_emb, b_emb, c_emb)


def _write_std_npz(path: Path, A: csr_matrix, b: np.ndarray, c: np.ndarray) -> None:
    """Write standard-form LP (c, b, A) to path in npz format (no .npz suffix appended)."""
    with open(path, "wb") as f:
        np.savez_compressed(
            f,
            c=c,
            b=b,
            A_data=A.data,
            A_indices=A.indices,
            A_indptr=A.indptr,
            A_shape=np.array(A.shape),
        )


def _initialise_instance_from_path(
    path: Path,
    instance_class: str | None = None,
) -> None:
    """Load .std at path, find initial triple (x, y, s) with x > 0, s > 0, write .init (and optionally .sde) next to it."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Instance file not found: {path}")
    if path.suffix.lower() != ".std":
        raise ValueError(f"Only .std instances are supported; got {path.suffix!r}")

    A, b, c = _load_standard_form(path)
    embedding_used = False
    emb_lp = None
    if instance_class == "clique":
        x, y, s = _find_initial_triple_clique(A, b, c)
    else:
        try:
            x, y, s = find_initial_triple(A, b, c)
        except Exception:
            result = selfdual_embedding(A, b, c)
            embedding_used = True
            if len(result) == 4:
                x, y, s, emb_lp = result
            else:
                x, y, s = result

    out_path = path.with_suffix(".init")
    with open(out_path, "wb") as f:
        np.savez_compressed(
            f,
            x=x,
            y=y,
            s=s,
            embedding_used=np.array(embedding_used),
        )

    if emb_lp is not None:
        A_emb, b_emb, c_emb = emb_lp
        sde_path = path.with_suffix(".sde")
        _write_std_npz(sde_path, A_emb, b_emb, c_emb)


def initialise_instance(
    instance_class: str,
    instance_name: str,
    cache_dir: str | Path | None = None,
) -> None:
    """Compute initial triple for the .std instance in cache_dir/instance_class/instance_name/.

    Discovers the single .std file in that subdirectory; writes .init (and optionally .sde) next to it.
    instance_class: e.g. "netlib", "miplib".
    instance_name: subfolder name (instance stem).
    cache_dir: root containing instance-class subfolders; defaults to "cache_dir".
    """
    root = Path(cache_dir).resolve() if cache_dir is not None else Path("cache_dir").resolve()
    instance_dir = root / instance_class / instance_name
    if not instance_dir.is_dir():
        raise FileNotFoundError(f"Instance directory not found: {instance_dir}")
    std_files = sorted(instance_dir.glob("*.std"))
    if len(std_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one .std in {instance_dir}; found {len(std_files)}"
        )
    _initialise_instance_from_path(std_files[0], instance_class=instance_class)


def initialise_instance_class(
    instance_class: str,
    cache_dir: str | Path | None = None,
) -> None:
    """Compute initial triples for all .std instances in the given instance-class subfolder of cache_dir.

    instance_class: name of the subfolder (e.g. "netlib", "miplib").
    cache_dir: directory containing instance-class subfolders; defaults to "cache_dir" in the current directory.
    """
    root = Path(cache_dir).resolve() if cache_dir is not None else Path("cache_dir").resolve()
    folder = root / instance_class
    if not folder.is_dir():
        raise FileNotFoundError(f"Instance class folder not found: {folder}")

    subdirs = sorted(d for d in folder.iterdir() if d.is_dir())
    for subdir in tqdm(subdirs, desc=instance_class, unit="instance"):
        try:
            initialise_instance(instance_class, subdir.name, cache_dir=root)
        except Exception as e:
            warnings.warn(f"Failed {instance_class}/{subdir.name}: {e}", stacklevel=2)


def initialise_all_instance_classes(
    instance_classes: list[str] | None = None,
    cache_dir: str | Path | None = None,
) -> None:
    """Compute initial triples for .std instances (main entry point).

    instance_classes: optional list of instance class names (subfolder names under cache_dir).
        If None, all instance classes (all subdirectories of cache_dir) are processed.
    cache_dir: directory containing instance-class subfolders; defaults to "cache_dir" in the current directory.
    """
    root = Path(cache_dir).resolve() if cache_dir is not None else Path("cache_dir").resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Cache directory not found: {root}")

    if instance_classes is None:
        instance_classes = [f.name for f in sorted(root.iterdir()) if f.is_dir()]

    for name in instance_classes:
        initialise_instance_class(name, root)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Find initial (x,y,s) with x>0, s>0 for standard-form .std LPs; write to .init (npz format).",
    )
    parser.add_argument(
        "instance_classes",
        nargs="*",
        help="Instance class names (subfolders under cache_dir). If none given, process all.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory (default: cache_dir in current directory).",
    )
    args = parser.parse_args()
    initialise_all_instance_classes(
        instance_classes=args.instance_classes or None,
        cache_dir=args.cache_dir,
    )
