#!/usr/bin/env python3
"""Benchmark LP instances (standard-form .std): compute gate counts for qipm1/2/3 and write to .qipm1/.qipm2/.qipm3."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm


def _gate_count_qipm1(A: csr_matrix, b: np.ndarray, c: np.ndarray) -> int:
    """Return gate count for qipm1. Standard form: min c'x s.t. Ax = b, x >= 0."""
    raise NotImplementedError


def _gate_count_qipm2(A: csr_matrix, b: np.ndarray, c: np.ndarray) -> int:
    """Return gate count for qipm2. Standard form: min c'x s.t. Ax = b, x >= 0."""
    raise NotImplementedError


def _gate_count_qipm3(A: csr_matrix, b: np.ndarray, c: np.ndarray) -> int:
    """Return gate count for qipm3. Standard form: min c'x s.t. Ax = b, x >= 0."""
    raise NotImplementedError


_GATE_COUNT_FUNCS = {1: _gate_count_qipm1, 2: _gate_count_qipm2, 3: _gate_count_qipm3}


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


def _benchmark_instance_from_path(
    path: Path,
    qipm_numbers: list[int] | None = None,
) -> None:
    """Load .std at path, compute gate counts for requested qipm(s), write .qipm1/.qipm2/.qipm3 next to it."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Instance file not found: {path}")
    if path.suffix.lower() != ".std":
        raise ValueError(f"Only .std instances are supported; got {path.suffix!r}")

    numbers = qipm_numbers if qipm_numbers else [1, 2, 3]
    if not numbers:
        numbers = [1, 2, 3]
    for n in numbers:
        if n not in (1, 2, 3):
            raise ValueError(f"qipm_numbers must contain only 1, 2, or 3; got {n}")

    A, b, c = _load_standard_form(path)
    for n in numbers:
        count = _GATE_COUNT_FUNCS[n](A, b, c)
        out_path = path.with_suffix(f".qipm{n}")
        out_path.write_text(f"{count}\n")


def benchmark_instance(
    instance_class: str,
    instance_name: str,
    cache_dir: str | Path | None = None,
    qipm_numbers: list[int] | None = None,
) -> None:
    """Run gate-count benchmark for the .std instance in cache_dir/instance_class/instance_name/.

    Discovers the single .std file in that subdirectory; writes .qipm1/.qipm2/.qipm3 next to it.
    instance_class: e.g. "netlib", "miplib".
    instance_name: subfolder name (instance stem).
    cache_dir: root containing instance-class subfolders; defaults to "cache_dir".
    qipm_numbers: which qipm variants (1, 2, 3). If None, all three are run.
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
    _benchmark_instance_from_path(std_files[0], qipm_numbers=qipm_numbers)


def benchmark_instance_class(
    instance_class: str,
    qipm_numbers: list[int] | None = None,
    cache_dir: str | Path | None = None,
) -> None:
    """Run gate-count benchmark for all .std instances in the given instance-class subfolder of cache_dir.

    instance_class: name of the subfolder (e.g. "netlib", "miplib").
    qipm_numbers: which qipm variants (1, 2, 3). If None, all three are run.
    cache_dir: directory containing instance-class subfolders; defaults to "cache_dir" in the current directory.
    """
    root = Path(cache_dir).resolve() if cache_dir is not None else Path("cache_dir").resolve()
    folder = root / instance_class
    if not folder.is_dir():
        raise FileNotFoundError(f"Instance class folder not found: {folder}")

    subdirs = sorted(d for d in folder.iterdir() if d.is_dir())
    for subdir in tqdm(subdirs, desc=instance_class, unit="instance"):
        benchmark_instance(instance_class, subdir.name, cache_dir=root, qipm_numbers=qipm_numbers)


def benchmark_all_instance_classes(
    instance_classes: list[str] | None = None,
    qipm_numbers: list[int] | None = None,
    cache_dir: str | Path | None = None,
) -> None:
    """Run gate-count benchmark for .std instances (main entry point).

    instance_classes: optional list of instance class names (subfolder names under cache_dir).
        If None, all instance classes (all subdirectories of cache_dir) are processed.
    qipm_numbers: which qipm variants (1, 2, 3). If None, all three are run.
    cache_dir: directory containing instance-class subfolders; defaults to "cache_dir" in the current directory.
    """
    root = Path(cache_dir).resolve() if cache_dir is not None else Path("cache_dir").resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Cache directory not found: {root}")

    if instance_classes is None:
        instance_classes = [f.name for f in sorted(root.iterdir()) if f.is_dir()]

    for name in instance_classes:
        benchmark_instance_class(name, qipm_numbers=qipm_numbers, cache_dir=root)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark .std LP instances: compute qipm gate counts and write to .qipm1/.qipm2/.qipm3.",
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
    parser.add_argument(
        "--qipm",
        nargs="*",
        type=int,
        choices=[1, 2, 3],
        default=None,
        metavar="N",
        help="Which qipm variants to run (1, 2, 3). If not given, run all three.",
    )
    args = parser.parse_args()
    benchmark_all_instance_classes(
        instance_classes=args.instance_classes or None,
        qipm_numbers=args.qipm,
        cache_dir=args.cache_dir,
    )
