#!/usr/bin/env python3
"""Solve LP instances (MPS or standard-form .npz) with HiGHS and record solve time."""

from __future__ import annotations

import time
from pathlib import Path

import highspy
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

try:
    _HIGHS_INF = highspy.kHighsInf
except AttributeError:
    _HIGHS_INF = 1e30


def _solve_mps(path: Path) -> float:
    """Read MPS into HiGHS, run solver (no presolve/transform). Return solve time in seconds."""
    h = highspy.Highs()
    h.setOptionValue("log_to_console", False)
    status = h.readModel(str(path))
    if status != highspy.HighsStatus.kOk and status != highspy.HighsStatus.kWarning:
        raise RuntimeError(f"HiGHS readModel failed: {status}")
    t0 = time.perf_counter()
    status = h.run()
    elapsed = time.perf_counter() - t0
    if status != highspy.HighsStatus.kOk and status != highspy.HighsStatus.kWarning:
        raise RuntimeError(f"HiGHS solve failed: {status}")
    return elapsed


def _solve_npz(path: Path) -> float:
    """Load .npz standard-form LP, build HiGHS model, run solver. Return solve time in seconds."""
    data = np.load(path)
    c = np.asarray(data["c"], dtype=np.float64).ravel()
    b = np.asarray(data["b"], dtype=np.float64).ravel()
    A_data = np.asarray(data["A_data"], dtype=np.float64).ravel()
    A_indices = np.asarray(data["A_indices"], dtype=np.int64).ravel()
    A_indptr = np.asarray(data["A_indptr"], dtype=np.int64).ravel()
    A_shape = np.asarray(data["A_shape"], dtype=np.int64).ravel()
    n, m = int(A_shape[1]), int(A_shape[0])  # A is (m, n)
    A = csr_matrix((A_data, A_indices, A_indptr), shape=(m, n))

    h = highspy.Highs()
    h.setOptionValue("log_to_console", False)
    col_lower = np.zeros(n, dtype=np.float64)
    col_upper = np.full(n, _HIGHS_INF, dtype=np.float64)
    h.addVars(n, col_lower, col_upper)
    h.changeColsCost(n, np.arange(n, dtype=np.int64), c)
    row_lower = b.copy()
    row_upper = b.copy()
    num_nz = int(A.nnz)
    starts = np.asarray(A.indptr[:-1], dtype=np.int32) if m > 0 else np.array([], dtype=np.int32)
    h.addRows(m, row_lower, row_upper, num_nz, starts, A.indices.astype(np.int32), A.data.astype(np.float64))

    t0 = time.perf_counter()
    status = h.run()
    elapsed = time.perf_counter() - t0
    if status != highspy.HighsStatus.kOk and status != highspy.HighsStatus.kWarning:
        # Retry with IPM when default solver fails (e.g. badly scaled RHS); IPM often handles scaling better
        h.setOptionValue("solver", "ipm")
        t0 = time.perf_counter()
        status = h.run()
        elapsed = time.perf_counter() - t0
        if status != highspy.HighsStatus.kOk and status != highspy.HighsStatus.kWarning:
            raise RuntimeError(f"HiGHS solve failed: {status}")
    return elapsed


def solve_instance(filepath: str | Path) -> None:
    """Solve an LP instance with HiGHS and write solve time to a sidecar file.

    - If filepath is .mps: read into HiGHS and solve directly (no transformation).
      Writes solve time to the same path with extension .mps_time (e.g. instance.mps -> instance.mps_time).
    - If filepath is .npz: load standard form (c, b, A), build HiGHS model, solve.
      Writes solve time to the same path with extension .std_time (e.g. instance.npz -> instance.std_time).

    Raises FileNotFoundError if the file does not exist, ValueError for unsupported extension.
    """
    path = Path(filepath).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Instance file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".mps":
        elapsed = _solve_mps(path)
        out_path = path.with_suffix(".mps_time")
    elif suffix == ".npz":
        elapsed = _solve_npz(path)
        out_path = path.with_suffix(".std_time")
    else:
        raise ValueError(f"Unsupported instance format: {suffix}. Use .mps or .npz.")

    out_path.write_text(f"{elapsed}\n")


def solve_instance_class(
    instance_class: str,
    cache_dir: str | Path | None = None,
    formats: str = "both",
) -> None:
    """Solve instances in the given instance-class subfolder of cache_dir.

    instance_class: name of the subfolder (e.g. "netlib", "miplib", "clique").
    cache_dir: directory containing instance-class subfolders; defaults to "cache_dir" in the current directory.
    formats: "mps" | "npz" | "both" — which instance formats to solve (default "both").
    """
    if formats not in ("mps", "npz", "both"):
        raise ValueError(f"formats must be 'mps', 'npz', or 'both'; got {formats!r}")

    root = Path(cache_dir).resolve() if cache_dir is not None else Path("cache_dir").resolve()
    folder = root / instance_class
    if not folder.is_dir():
        raise FileNotFoundError(f"Instance class folder not found: {folder}")

    paths: list[Path] = []
    if formats in ("mps", "both"):
        paths.extend(sorted(folder.glob("*.mps")))
    if formats in ("npz", "both"):
        paths.extend(sorted(folder.glob("*.npz")))
    for p in tqdm(paths, desc=instance_class, unit="instance"):
        solve_instance(p)


def solve_all_instance_classes(
    instance_classes: list[str] | None = None,
    cache_dir: str | Path | None = None,
    formats: str = "both",
) -> None:
    """Solve instances for given or all instance classes (main entry point).

    instance_classes: optional list of instance class names (subfolder names under cache_dir).
        If None, all instance classes (all subdirectories of cache_dir) are processed.
    cache_dir: directory containing instance-class subfolders; defaults to "cache_dir" in the current directory.
    formats: "mps" | "npz" | "both" — which instance formats to solve (default "both").
    """
    root = Path(cache_dir).resolve() if cache_dir is not None else Path("cache_dir").resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Cache directory not found: {root}")

    if instance_classes is None:
        instance_classes = [f.name for f in sorted(root.iterdir()) if f.is_dir()]

    for name in instance_classes:
        solve_instance_class(name, root, formats=formats)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Solve standard-form LP instances with HiGHS and record solve time.")
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
    parser.add_argument(
        "--formats",
        choices=("mps", "npz", "both"),
        default="both",
        help="Instance formats to solve: mps, npz, or both (default: both).",
    )
    args = parser.parse_args()
    solve_all_instance_classes(
        instance_classes=args.instance_classes or None,
        cache_dir=args.cache_dir,
        formats=args.formats,
    )
