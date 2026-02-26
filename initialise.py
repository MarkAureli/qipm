#!/usr/bin/env python3
"""Find an initial triple (x, y, s) with x > 0, s > 0 for standard-form LPs; write to .init (npz format). On failure, use self-dual embedding."""

from __future__ import annotations

import warnings
from pathlib import Path

import highspy
import numpy as np
from scipy.sparse import csr_matrix, eye, hstack, vstack
from scipy.sparse.linalg import lsqr as sparse_lsqr
from tqdm import tqdm

try:
    _HIGHS_INF = highspy.kHighsInf
except AttributeError:
    _HIGHS_INF = 1e30

# Minimum strict positivity for x and s
_STRICT_DELTA = 1e-8
# Tolerance for post-solve slack validation (LP solver floating-point noise)
_SOLVER_TOL = 1e-7


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
    """Find y and s = c - A'y with s > 0, maximising min(s). Returns (y, s). Raises if not found."""
    m, n = A.shape
    delta = _STRICT_DELTA
    # Chebyshev-centre LP: max t  s.t.  A'y + t·e ≤ c,  t ≥ delta
    # Variables: [y (m), t (1)]; objective: min -t
    # Constraint matrix: [A' | e]  (n × m+1)
    A_t = A.T.tocsr()
    ones_col = csr_matrix(np.ones((n, 1)))
    C = hstack([A_t, ones_col], format="csr")
    n_vars = m + 1
    col_lower = np.full(n_vars, -_HIGHS_INF, dtype=np.float64)
    col_lower[m] = delta  # t ≥ delta: LP infeasible iff no strictly feasible s exists
    col_upper = np.full(n_vars, _HIGHS_INF, dtype=np.float64)
    col_cost = np.zeros(n_vars, dtype=np.float64)
    col_cost[m] = -1.0  # minimise -t
    row_lower = np.full(n, -_HIGHS_INF, dtype=np.float64)
    row_upper = c.astype(np.float64)
    num_nz = int(C.nnz)
    starts = C.indptr[:-1].astype(np.int32) if n > 0 else np.array([], dtype=np.int32)
    h = highspy.Highs()
    h.setOptionValue("log_to_console", False)
    h.setOptionValue("time_limit", 60.0)
    h.addVars(n_vars, col_lower, col_upper)
    h.changeColsCost(n_vars, np.arange(n_vars, dtype=np.int64), col_cost)
    h.addRows(n, row_lower, row_upper, num_nz, starts,
              C.indices.astype(np.int32), C.data.astype(np.float64))
    status = h.run()
    if status != highspy.HighsStatus.kOk and status != highspy.HighsStatus.kWarning:
        raise RuntimeError(f"LP for dual feasible (y, s) failed (status={status})")
    model_status = h.getModelStatus()
    if model_status == highspy.HighsModelStatus.kInfeasible:
        raise RuntimeError("No dual feasible (y, s) exists — LP is infeasible")
    if model_status == highspy.HighsModelStatus.kTimeLimit:
        raise RuntimeError("Dual feasibility LP hit time limit")
    sol = h.getSolution()
    vars_sol = np.asarray(sol.col_value, dtype=np.float64).ravel()
    y = vars_sol[:m]
    t_opt = max(float(vars_sol[m]), delta)  # guard against numerical t < delta
    s = c - A_t @ y
    if not np.all(s >= t_opt - _SOLVER_TOL):
        raise RuntimeError("Dual slack s not strictly positive after LP")
    s = np.maximum(s, t_opt)
    return y, s


def _find_primal_feasible_strict(A: csr_matrix, b: np.ndarray) -> np.ndarray:
    """Find x > 0 with Ax = b, maximising min(x). Raises if not found."""
    m, n = A.shape
    delta = _STRICT_DELTA
    # Fast path: if lsqr minimum-norm solution is already strictly positive, use it.
    # Infeasibility of Ax=b is determined authoritatively by the LP below.
    x0 = np.asarray(sparse_lsqr(A, b)[0], dtype=np.float64).ravel()
    if np.all(x0 > delta) and np.linalg.norm(A @ x0 - b) <= 1e-6 * (1.0 + float(np.linalg.norm(b))):
        return x0
    # Chebyshev-centre LP: max t  s.t.  Ax = b,  x ≥ t·e,  t ≥ delta
    # Variables: [x (n), t (1)]; objective: min -t
    # Constraint matrix: [A | 0] (equality, m rows) stacked with [I | -e] (lower bound, n rows)
    A_csr = A.tocsr()
    eq_block = hstack([A_csr, csr_matrix((m, 1))], format="csr")
    ineq_block = hstack([eye(n, format="csr"), csr_matrix(-np.ones((n, 1)))], format="csr")
    C = vstack([eq_block, ineq_block], format="csr")
    n_vars = n + 1
    col_lower = np.zeros(n_vars, dtype=np.float64)
    col_lower[n] = delta  # t ≥ delta: LP infeasible iff no strictly feasible x exists
    col_upper = np.full(n_vars, _HIGHS_INF, dtype=np.float64)
    col_cost = np.zeros(n_vars, dtype=np.float64)
    col_cost[n] = -1.0  # minimise -t
    row_lower = np.concatenate([b.astype(np.float64), np.zeros(n)])
    row_upper = np.concatenate([b.astype(np.float64), np.full(n, _HIGHS_INF)])
    num_nz = int(C.nnz)
    starts = C.indptr[:-1].astype(np.int32) if (m + n) > 0 else np.array([], dtype=np.int32)
    h = highspy.Highs()
    h.setOptionValue("log_to_console", False)
    h.setOptionValue("time_limit", 60.0)
    h.addVars(n_vars, col_lower, col_upper)
    h.changeColsCost(n_vars, np.arange(n_vars, dtype=np.int64), col_cost)
    h.addRows(m + n, row_lower, row_upper, num_nz, starts,
              C.indices.astype(np.int32), C.data.astype(np.float64))
    status = h.run()
    model_status = h.getModelStatus()
    if model_status == highspy.HighsModelStatus.kInfeasible:
        raise RuntimeError("No strictly feasible primal solution exists (LP infeasible)")
    if status not in (highspy.HighsStatus.kOk, highspy.HighsStatus.kWarning):
        raise RuntimeError(f"LP for primal strictly feasible x failed (status={status})")
    if model_status == highspy.HighsModelStatus.kTimeLimit:
        raise RuntimeError("Primal feasibility LP hit time limit")
    sol = h.getSolution()
    vars_sol = np.asarray(sol.col_value, dtype=np.float64).ravel()
    x = vars_sol[:n]
    t_opt = max(float(vars_sol[n]), delta)  # guard against numerical t < delta
    if not np.all(x >= t_opt - _SOLVER_TOL):
        raise RuntimeError("Primal x not strictly positive after LP")
    x = np.maximum(x, t_opt)
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
    x = _find_primal_feasible_strict(A, b)
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
        except RuntimeError:
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
