#!/usr/bin/env python3
"""Benchmark LP instances: (A,b,c) from .sde if present else .std; initial triple from .init; compute gate counts for qipm1/2/3 and write to instance .data (JSON)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

from helpers.gate_count_qlsa import gate_count_qlsa
from helpers.gate_count_state_prep import gate_count_state_preparation
from helpers.linear_systems import build_modified_nes


def _sparsity_and_cond(M: np.ndarray) -> tuple[int, float]:
    """Return maximum sparsity (non-zeros per row/column) and 2-norm condition number of M."""
    M = np.asarray(M)
    nz_row = np.count_nonzero(M, axis=1)
    nz_col = np.count_nonzero(M, axis=0)
    d = int(max(nz_row.max(), nz_col.max()))
    k = float(np.linalg.cond(M))
    return d, k


def _gate_count_qipm1(
    A: csr_matrix,
    b: np.ndarray,
    c: np.ndarray,
    x_init: np.ndarray | None,
    y_init: np.ndarray | None,
    s_init: np.ndarray | None,
) -> tuple[int, int, float]:
    """Return (gate_count, sparsity, cond) for qipm1. Requires initial triple (x_init, y_init, s_init) from .init."""
    if x_init is None or y_init is None or s_init is None:
        raise ValueError("qipm1 requires initial triple (x_init, y_init, s_init) from .init")
    M_hat, omega_hat = build_modified_nes(A, b, c, x_init, y_init, s_init, mu=1.0)
    d, k = _sparsity_and_cond(M_hat)
    norm = np.linalg.norm(M_hat, 2)
    if norm <= 0:
        raise ValueError("spectral norm of M_hat is zero")
    M_hat = M_hat / norm
    omega_hat = omega_hat / norm
    count = (
        gate_count_qlsa(M_hat, omega_hat, d=d, k=k)
        + gate_count_state_preparation(omega_hat)
        + M_hat.shape[0]
    )
    return count, d, k


def _gate_count_qipm2(
    A: csr_matrix,
    b: np.ndarray,
    c: np.ndarray,
    x_init: np.ndarray | None,
    y_init: np.ndarray | None,
    s_init: np.ndarray | None,
) -> tuple[int, int, float]:
    """Return (gate_count, sparsity, cond) for qipm2. Requires initial triple (x_init, y_init, s_init) from .init."""
    raise NotImplementedError


def _gate_count_qipm3(
    A: csr_matrix,
    b: np.ndarray,
    c: np.ndarray,
    x_init: np.ndarray | None,
    y_init: np.ndarray | None,
    s_init: np.ndarray | None,
) -> tuple[int, int, float]:
    """Return (gate_count, sparsity, cond) for qipm3. Requires initial triple (x_init, y_init, s_init) from .init."""
    raise NotImplementedError


_GATE_COUNT_FUNCS = {1: _gate_count_qipm1, 2: _gate_count_qipm2, 3: _gate_count_qipm3}


def _load_standard_form(path: Path) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    """Load (A, b, c) from .std or .sde standard-form LP (same npz format). A returned as CSR."""
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


def _load_init(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load initial triple (x, y, s) from .init npz."""
    data = np.load(path)
    x = np.asarray(data["x"], dtype=np.float64).ravel()
    y = np.asarray(data["y"], dtype=np.float64).ravel()
    s = np.asarray(data["s"], dtype=np.float64).ravel()
    return x, y, s


def _benchmark_instance_from_path(
    path: Path,
    qipm_numbers: list[int] | None = None,
) -> None:
    """Load (A, b, c) from .sde if present else .std; load initial triple from .init; compute gate counts, write to instance .data (JSON). path must be .std or .sde."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Instance file not found: {path}")
    suf = path.suffix.lower()
    if suf not in (".std", ".sde"):
        raise ValueError(f"Path must be .std or .sde; got {path.suffix!r}")

    # Strip only the known extension so names with dots (e.g. clique "name.dimac_clq.std") stay correct
    base_name = path.name[: -len(suf)]
    base = path.parent / base_name
    sde_path = path.parent / (base_name + ".sde")
    std_path = path.parent / (base_name + ".std")
    lp_path = sde_path if sde_path.is_file() else std_path
    if not lp_path.is_file():
        raise FileNotFoundError(f"Neither {sde_path} nor {std_path} found")

    numbers = qipm_numbers if qipm_numbers else [1, 2, 3]
    if not numbers:
        numbers = [1, 2, 3]
    for n in numbers:
        if n not in (1, 2, 3):
            raise ValueError(f"qipm_numbers must contain only 1, 2, or 3; got {n}")

    A, b, c = _load_standard_form(lp_path)

    init_path = path.parent / (base_name + ".init")
    if init_path.is_file():
        x_init, y_init, s_init = _load_init(init_path)
    else:
        x_init, y_init, s_init = None, None, None

    data_path = path.parent / (base_name + ".data")
    if data_path.exists():
        data = json.loads(data_path.read_text())
    else:
        data = {}

    for n in numbers:
        count, sparsity, cond = _GATE_COUNT_FUNCS[n](A, b, c, x_init, y_init, s_init)
        data[f"gate_count_qipm{n}"] = count
        data[f"sparsity_qipm{n}"] = sparsity
        data[f"cond_qipm{n}"] = cond

    data_path.write_text(json.dumps(data, indent=None))


def benchmark_instance(
    instance_class: str,
    instance_name: str,
    cache_dir: str | Path | None = None,
    qipm_numbers: list[int] | None = None,
) -> None:
    """Run gate-count benchmark for the instance in cache_dir/instance_class/instance_name/.

    Discovers the instance by .sde if present (exactly one), else by .std (exactly one). Loads (A,b,c) from that file and initial triple from .init; writes gate counts to instance .data (JSON).
    instance_class: e.g. "netlib", "miplib".
    instance_name: subfolder name (instance stem).
    cache_dir: root containing instance-class subfolders; defaults to "cache_dir".
    qipm_numbers: which qipm variants (1, 2, 3). If None, all three are run.
    """
    root = Path(cache_dir).resolve() if cache_dir is not None else Path("cache_dir").resolve()
    instance_dir = root / instance_class / instance_name
    if not instance_dir.is_dir():
        raise FileNotFoundError(f"Instance directory not found: {instance_dir}")
    sde_files = sorted(instance_dir.glob("*.sde"))
    std_files = sorted(instance_dir.glob("*.std"))
    if sde_files:
        if len(sde_files) != 1:
            raise FileNotFoundError(
                f"Expected exactly one .sde in {instance_dir}; found {len(sde_files)}"
            )
        instance_path = sde_files[0]
    else:
        if len(std_files) != 1:
            raise FileNotFoundError(
                f"Expected exactly one .std in {instance_dir}; found {len(std_files)}"
            )
        instance_path = std_files[0]
    _benchmark_instance_from_path(instance_path, qipm_numbers=qipm_numbers)


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
        description="Benchmark LP instances: compute qipm gate counts and write to instance .data (JSON).",
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
