#!/usr/bin/env python3
"""Transform MPS instances to standard-form LP (min c'x, Ax=b, x>=0) after presolve."""

from __future__ import annotations

import warnings
from pathlib import Path

import highspy
from tqdm import tqdm
import numpy as np
from scipy.sparse import csr_matrix

# HiGHS uses ±inf for unbounded bounds
try:
    _HIGHS_INF = highspy.kHighsInf
except AttributeError:
    _HIGHS_INF = 1e30  # fallback if constant not exposed


def _lp_to_standard_form(
    num_col: int,
    num_row: int,
    col_cost: np.ndarray,
    col_lower: np.ndarray,
    col_upper: np.ndarray,
    row_lower: np.ndarray,
    row_upper: np.ndarray,
    a_start: np.ndarray,
    a_index: np.ndarray,
    a_value: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, csr_matrix]:
    """Convert HiGHS LP (col_lower <= x <= col_upper, row_lower <= Ax <= row_upper) to standard form.

    Standard form: min c'x  s.t.  Ax = b,  x >= 0.

    Returns (c, b, A) with A in CSR sparse format.
    """
    inf = _HIGHS_INF
    # Convert to numpy arrays once (avoid repeated conversion in caller)
    col_cost = np.asarray(col_cost, dtype=np.float64).ravel()
    col_lower = np.asarray(col_lower, dtype=np.float64).ravel()
    col_upper = np.asarray(col_upper, dtype=np.float64).ravel()
    row_lower = np.asarray(row_lower, dtype=np.float64).ravel()
    row_upper = np.asarray(row_upper, dtype=np.float64).ravel()
    a_start = np.asarray(a_start, dtype=np.int64).ravel()
    a_index = np.asarray(a_index, dtype=np.int64).ravel()
    a_value = np.asarray(a_value, dtype=np.float64).ravel()

    new_col_count = 0
    c_list: list[float] = []
    b_list: list[float] = []
    extra_b_list: list[float] = []
    row_list: list[int] = []
    col_list: list[int] = []
    val_list: list[float] = []
    extra_row_list: list[int] = []
    extra_col_list: list[int] = []
    extra_val_list: list[float] = []
    row_constant = np.zeros(num_row, dtype=np.float64)

    def add_var(cost: float) -> int:
        nonlocal new_col_count
        j = new_col_count
        new_col_count += 1
        c_list.append(cost)
        return j

    # Single pass over columns: map to non-negative variables and accumulate row_constant
    for j in range(num_col):
        lj = col_lower[j]
        uj = col_upper[j]
        cj = col_cost[j]
        beg = a_start[j]
        end = a_start[j + 1]
        row_ind = a_index[beg:end]
        row_vals = a_value[beg:end]

        if lj > -inf and uj < inf:
            j1 = add_var(cj)
            j2 = add_var(0.0)
            width = uj - lj
            r = num_row + len(extra_b_list)
            extra_b_list.append(width)
            extra_row_list.append(r)
            extra_col_list.append(j1)
            extra_val_list.append(1.0)
            extra_row_list.append(r)
            extra_col_list.append(j2)
            extra_val_list.append(1.0)
            for idx in range(len(row_ind)):
                i = row_ind[idx]
                v = row_vals[idx]
                row_list.append(i)
                col_list.append(j1)
                val_list.append(v)
                row_constant[i] += v * lj
            continue

        if lj > -inf and uj >= inf:
            j1 = add_var(cj)
            for idx in range(len(row_ind)):
                i = row_ind[idx]
                v = row_vals[idx]
                row_list.append(i)
                col_list.append(j1)
                val_list.append(v)
                row_constant[i] += v * lj
            continue
        if lj <= -inf and uj < inf:
            j1 = add_var(-cj)
            for idx in range(len(row_ind)):
                i = row_ind[idx]
                v = row_vals[idx]
                row_list.append(i)
                col_list.append(j1)
                val_list.append(-v)
                row_constant[i] += v * uj
            continue
        # Free variable
        j_plus = add_var(cj)
        j_minus = add_var(-cj)
        for idx in range(len(row_ind)):
            i = row_ind[idx]
            v = row_vals[idx]
            row_list.append(i)
            col_list.append(j_plus)
            val_list.append(v)
            row_list.append(i)
            col_list.append(j_minus)
            val_list.append(-v)

    # Add slacks and set b for original rows
    for i in range(num_row):
        lo = row_lower[i]
        hi = row_upper[i]
        if lo == hi:
            b_list.append(lo - row_constant[i])
        elif lo <= -inf and hi < inf:
            j_slack = add_var(0.0)
            row_list.append(i)
            col_list.append(j_slack)
            val_list.append(1.0)
            b_list.append(hi - row_constant[i])
        elif lo > -inf and hi >= inf:
            j_slack = add_var(0.0)
            row_list.append(i)
            col_list.append(j_slack)
            val_list.append(-1.0)
            b_list.append(lo - row_constant[i])
        else:
            j_slack = add_var(0.0)
            row_list.append(i)
            col_list.append(j_slack)
            val_list.append(1.0)
            b_list.append(hi - row_constant[i])
            extra_b_list.append(hi - lo)
            r = num_row + len(extra_b_list) - 1
            extra_row_list.append(r)
            extra_col_list.append(j_slack)
            extra_val_list.append(1.0)
            j_s2 = add_var(0.0)
            extra_row_list.append(r)
            extra_col_list.append(j_s2)
            extra_val_list.append(1.0)

    n_std = new_col_count
    m_std = len(b_list) + len(extra_b_list)
    b_std = np.empty(m_std, dtype=np.float64)
    b_std[:len(b_list)] = b_list
    b_std[len(b_list):] = extra_b_list
    c_std = np.fromiter(c_list, dtype=np.float64, count=len(c_list))

    nnz = len(row_list) + len(extra_row_list)
    if nnz == 0:
        A_std = csr_matrix((m_std, n_std))
    else:
        row_arr = np.empty(nnz, dtype=np.int64)
        col_arr = np.empty(nnz, dtype=np.int64)
        val_arr = np.empty(nnz, dtype=np.float64)
        n1 = len(row_list)
        row_arr[:n1] = row_list
        col_arr[:n1] = col_list
        val_arr[:n1] = val_list
        row_arr[n1:] = extra_row_list
        col_arr[n1:] = extra_col_list
        val_arr[n1:] = extra_val_list
        A_std = csr_matrix((val_arr, (row_arr, col_arr)), shape=(m_std, n_std))
    return c_std, b_std, A_std


def _transform_instance_from_path(path: Path) -> None:
    """Read MPS file at path, presolve with HiGHS, convert to standard form, and save .std next to it."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MPS file not found: {path}")
    if path.stat().st_size == 0:
        warnings.warn(f"Skipping empty file: {path}", stacklevel=2)
        return

    h = highspy.Highs()
    h.setOptionValue("log_to_console", False)
    status = h.readModel(str(path))
    if status == highspy.HighsStatus.kWarning:
        warnings.warn(f"HiGHS readModel returned kWarning for {path}", stacklevel=2)
    elif status != highspy.HighsStatus.kOk:
        raise RuntimeError(f"HiGHS readModel failed: {status}")

    status = h.presolve()
    if status == highspy.HighsStatus.kWarning:
        warnings.warn(f"HiGHS presolve returned kWarning for {path}", stacklevel=2)
    elif status != highspy.HighsStatus.kOk:
        raise RuntimeError(f"HiGHS presolve failed: {status}")

    # After presolve(), the incumbent model in HiGHS is the presolved LP.
    lp = h.getLp()
    if lp is None or (lp.num_col_ == 0 and lp.num_row_ == 0):
        raise RuntimeError("HiGHS presolved LP not available or empty")

    num_col = lp.num_col_
    num_row = lp.num_row_
    a = lp.a_matrix_
    c, b, A = _lp_to_standard_form(
        num_col, num_row,
        lp.col_cost_, lp.col_lower_, lp.col_upper_,
        lp.row_lower_, lp.row_upper_,
        a.start_, a.index_, a.value_,
    )

    out_std = path.with_suffix(".std")
    # savez_compressed appends .npz if missing; write to .npz then rename to .std
    out_npz = path.with_suffix(".npz")
    np.savez_compressed(
        str(out_npz),
        c=c,
        b=b,
        A_data=A.data,
        A_indices=A.indices,
        A_indptr=A.indptr,
        A_shape=np.array(A.shape),
    )
    out_npz.rename(out_std)


def transform_instance(
    instance_class: str,
    instance_name: str,
    cache_dir: str | Path | None = None,
) -> None:
    """Transform the MPS instance in cache_dir/instance_class/instance_name/ to standard form.

    Discovers the single .mps file in that subdirectory and writes .std next to it.
    instance_class: e.g. "netlib", "miplib".
    instance_name: subfolder name (instance stem).
    cache_dir: root containing instance-class subfolders; defaults to "cache_dir".
    """
    root = Path(cache_dir).resolve() if cache_dir is not None else Path("cache_dir").resolve()
    instance_dir = root / instance_class / instance_name
    if not instance_dir.is_dir():
        raise FileNotFoundError(f"Instance directory not found: {instance_dir}")
    mps_files = sorted(instance_dir.glob("*.mps"))
    if len(mps_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one .mps in {instance_dir}; found {len(mps_files)}"
        )
    _transform_instance_from_path(mps_files[0])


def transform_instance_class(
    instance_class: str,
    cache_dir: str | Path | None = None,
) -> None:
    """Transform all MPS instances in the given instance-class subfolder of cache_dir to standard form.

    instance_class: name of the subfolder (e.g. "netlib", "miplib", "clique").
    cache_dir: directory containing instance-class subfolders; defaults to "cache_dir" in the current directory.
    """
    root = Path(cache_dir).resolve() if cache_dir is not None else Path("cache_dir").resolve()
    folder = root / instance_class
    if not folder.is_dir():
        raise FileNotFoundError(f"Instance class folder not found: {folder}")

    subdirs = sorted(d for d in folder.iterdir() if d.is_dir())
    for subdir in tqdm(subdirs, desc=instance_class, unit="instance"):
        transform_instance(instance_class, subdir.name, cache_dir=root)


def transform_all_instance_classes(
    instance_classes: list[str] | None = None,
    cache_dir: str | Path | None = None,
) -> None:
    """Transform MPS instances to standard form (main entry point).

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
        transform_instance_class(name, root)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Transform MPS instances to standard-form LP.")
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
    transform_all_instance_classes(
        instance_classes=args.instance_classes or None,
        cache_dir=args.cache_dir,
    )
