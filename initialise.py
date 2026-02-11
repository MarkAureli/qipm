#!/usr/bin/env python3
"""Find an initial triple (x, y, s) with x > 0, s > 0 for standard-form LPs; write to .init (npz format). On failure, use selfdual-embedding (stub)."""

from __future__ import annotations

import warnings
from pathlib import Path

import highspy
import numpy as np
from scipy.linalg import lstsq, null_space
from scipy.sparse import csr_matrix
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
    y, s = _find_dual_feasible_strict(A, c)
    x = _find_primal_feasible_strict(A, b, n)
    return x, y, s


def selfdual_embedding(
    A: csr_matrix,
    b: np.ndarray,
    c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[csr_matrix, np.ndarray, np.ndarray] | None]:
    """Return an initial triple (x, y, s) and optionally the embedded LP via the self-dual embedding.

    When implemented, returns (x, y, s, emb_lp) where emb_lp is (A_emb, b_emb, c_emb) for the
    embedded problem in standard form; the embedded LP is stored as <base>.sde (npz format).
    Not implemented; raise NotImplementedError.
    """
    raise NotImplementedError("selfdual_embedding is not implemented")


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


def initialise_instance(filepath: str | Path) -> None:
    """Load standard-form LP from .std, find initial triple (x, y, s) with x > 0, s > 0, and save to .init (npz format).

    If finding a triple fails, reverts to selfdual_embedding (stub). Writes x, y, s and embedding_used to <base>.init.
    When embedding is used, the embedded LP is written to <base>.sde (npz format).
    """
    path = Path(filepath).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Instance file not found: {path}")
    if path.suffix.lower() != ".std":
        raise ValueError(f"Only .std instances are supported; got {path.suffix!r}")

    A, b, c = _load_standard_form(path)
    embedding_used = False
    emb_lp = None
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

    paths = sorted(folder.glob("*.std"))
    for p in tqdm(paths, desc=instance_class, unit="instance"):
        try:
            initialise_instance(p)
        except NotImplementedError:
            warnings.warn(
                f"selfdual_embedding not implemented; skipping {p}",
                stacklevel=2,
            )
        except Exception as e:
            warnings.warn(f"Failed {p}: {e}", stacklevel=2)


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
