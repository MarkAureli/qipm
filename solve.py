#!/usr/bin/env python3
"""Solve LP instances (MPS or standard-form .std) with HiGHS and record solve time."""

from __future__ import annotations

import json
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


def _solve_std(path: Path) -> float:
    """Load .std standard-form LP, build HiGHS model, run solver. Return solve time in seconds."""
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


def _solve_instance_from_path(path: Path) -> None:
    """Solve the instance at path (.mps or .std) with HiGHS; write solve time to sidecar and to instance .data."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Instance file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".mps":
        elapsed = _solve_mps(path)
        out_path = path.with_suffix(".mps_time")
        data_key = "runtime_highs_mps"
    elif suffix == ".std":
        elapsed = _solve_std(path)
        out_path = path.with_suffix(".std_time")
        data_key = "runtime_highs_std"
    else:
        raise ValueError(f"Unsupported instance format: {suffix}. Use .mps or .std.")

    out_path.write_text(f"{elapsed}\n")

    data_path = path.with_suffix(".data")
    if data_path.exists():
        data = json.loads(data_path.read_text())
    else:
        data = {}
    data[data_key] = elapsed
    data_path.write_text(json.dumps(data, indent=None))


def solve_instance(
    instance_class: str,
    instance_name: str,
    cache_dir: str | Path | None = None,
    formats: str = "both",
) -> None:
    """Solve the instance(s) in cache_dir/instance_class/instance_name/ with HiGHS.

    Discovers .mps and/or .std in that subdirectory according to formats; writes .mps_time / .std_time next to each.
    instance_class: e.g. "netlib", "miplib".
    instance_name: subfolder name (instance stem).
    cache_dir: root containing instance-class subfolders; defaults to "cache_dir".
    formats: "mps" | "std" | "both" — which formats to solve (default "both").
    """
    if formats not in ("mps", "std", "both"):
        raise ValueError(f"formats must be 'mps', 'std', or 'both'; got {formats!r}")

    root = Path(cache_dir).resolve() if cache_dir is not None else Path("cache_dir").resolve()
    instance_dir = root / instance_class / instance_name
    if not instance_dir.is_dir():
        raise FileNotFoundError(f"Instance directory not found: {instance_dir}")

    paths: list[Path] = []
    if formats in ("mps", "both"):
        paths.extend(sorted(instance_dir.glob("*.mps")))
    if formats in ("std", "both"):
        paths.extend(sorted(instance_dir.glob("*.std")))
    for p in paths:
        _solve_instance_from_path(p)


def solve_instance_class(
    instance_class: str,
    cache_dir: str | Path | None = None,
    formats: str = "both",
) -> None:
    """Solve instances in the given instance-class subfolder of cache_dir.

    instance_class: name of the subfolder (e.g. "netlib", "miplib", "clique").
    cache_dir: directory containing instance-class subfolders; defaults to "cache_dir" in the current directory.
    formats: "mps" | "std" | "both" — which instance formats to solve (default "both").
    """
    if formats not in ("mps", "std", "both"):
        raise ValueError(f"formats must be 'mps', 'std', or 'both'; got {formats!r}")

    root = Path(cache_dir).resolve() if cache_dir is not None else Path("cache_dir").resolve()
    folder = root / instance_class
    if not folder.is_dir():
        raise FileNotFoundError(f"Instance class folder not found: {folder}")

    subdirs = sorted(d for d in folder.iterdir() if d.is_dir())
    for subdir in tqdm(subdirs, desc=instance_class, unit="instance"):
        solve_instance(instance_class, subdir.name, cache_dir=root, formats=formats)


def solve_all_instance_classes(
    instance_classes: list[str] | None = None,
    cache_dir: str | Path | None = None,
    formats: str = "both",
) -> None:
    """Solve instances for given or all instance classes (main entry point).

    instance_classes: optional list of instance class names (subfolder names under cache_dir).
        If None, all instance classes (all subdirectories of cache_dir) are processed.
    cache_dir: directory containing instance-class subfolders; defaults to "cache_dir" in the current directory.
    formats: "mps" | "std" | "both" — which instance formats to solve (default "both").
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
        choices=("mps", "std", "both"),
        default="both",
        help="Instance formats to solve: mps, std, or both (default: both).",
    )
    args = parser.parse_args()
    solve_all_instance_classes(
        instance_classes=args.instance_classes or None,
        cache_dir=args.cache_dir,
        formats=args.formats,
    )
