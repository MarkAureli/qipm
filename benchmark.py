#!/usr/bin/env python3
"""Benchmark LP instances: A from .std; compute gate counts for qipm1/2 and write to instance .data (JSON)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

from helpers.gate_count_qlsa import gate_count_qlsa
from helpers.gate_count_state_prep import gate_count_state_preparation


def _gate_count_qipm1(A: csr_matrix) -> tuple[int, int, float]:
    """Return (gate_count, sparsity, cond) for qipm1.

    Estimates κ(M̂) via M̂ = I + F̄F̄ᵀ, F̄ = A_B⁻¹ A_N (D_B = D_N = I).
    M̂ is m×m and generically dense, so d = m.
    """
    import sparseqr
    from scipy.sparse.linalg import LinearOperator, eigsh, splu

    A = csr_matrix(A, dtype=np.float64)
    m, n = A.shape

    _, _, basis_P, effective_rank = sparseqr.qr(A)
    basis_P = np.asarray(basis_P, dtype=np.intp)

    if effective_rank < m:
        _, _, P_row, _ = sparseqr.qr(A.T)
        P_row = np.asarray(P_row, dtype=np.intp)
        A = A[P_row[:effective_rank], :]
        m = effective_rank

    B = basis_P[:m]
    N_mask = np.ones(n, dtype=bool)
    N_mask[B] = False
    N = np.where(N_mask)[0]
    n_N = len(N)

    d = m  # M̂ is generically dense m×m

    if n_N == 0 or m <= 1:
        k = 1.0
    else:
        A_B_lu = splu(A[:, B].tocsc())
        A_N = A[:, N]

        def _fbar_mv(v: np.ndarray) -> np.ndarray:
            return A_B_lu.solve(np.asarray(A_N @ v, dtype=np.float64).ravel())

        def _fbar_rmv(u: np.ndarray) -> np.ndarray:
            return np.asarray(A_N.T @ A_B_lu.solve(u, trans="T"), dtype=np.float64).ravel()

        def _mhat_mv(v: np.ndarray) -> np.ndarray:
            v = np.asarray(v, dtype=np.float64).ravel()
            return v + _fbar_mv(_fbar_rmv(v))

        M_op = LinearOperator((m, m), matvec=_mhat_mv, dtype=np.float64)
        lam_max = float(eigsh(M_op, k=1, which="LM")[0][0])
        lam_min = 1.0 if n_N < m else float(eigsh(M_op, k=1, which="SM")[0][0])
        k = lam_max / lam_min if lam_min > 0.0 else float("inf")

    count = (
        gate_count_qlsa(d=d, k=k)
        + gate_count_state_preparation(np.arange(1.0, m + 1))
        + m
    )
    return count, d, k


def _gate_count_qipm2(A: csr_matrix) -> tuple[int, int, float]:
    """Return (gate_count, sparsity, cond) for qipm2.

    Estimates κ(M) for the OSS matrix M = [-Aᵀ | V] ∈ ℝⁿˣⁿ (x = s = 1).
    V ∈ ℝⁿˣ⁽ⁿ⁻ᵐ⁾ is the null-space basis built from the SPQR pivot basis B:
        V[B, :] = -A_B⁻¹ A_N,  V[N, :] = I_{n-m}.

    Sparsity d = max(max column nnz of A, m + 1):
    - z_y columns of M have the same sparsity as columns of A,
    - z_λ columns have m entries in B-rows (dense A_B⁻¹ A_N column) + 1 in N-rows.
    """
    import sparseqr
    from scipy.sparse.linalg import LinearOperator, svds, splu

    A = csr_matrix(A, dtype=np.float64)
    m, n = A.shape

    if n <= 1:
        k = 1.0
        count = gate_count_qlsa(d=1, k=k) + gate_count_state_preparation(np.array([1.0])) + 1
        return count, 1, k

    _, _, basis_P, effective_rank = sparseqr.qr(A)
    basis_P = np.asarray(basis_P, dtype=np.intp)

    if effective_rank < m:
        _, _, P_row, _ = sparseqr.qr(A.T)
        P_row = np.asarray(P_row, dtype=np.intp)
        A = A[P_row[:effective_rank], :]
        m = effective_rank

    B = basis_P[:m]
    N_mask = np.ones(n, dtype=bool)
    N_mask[B] = False
    N = np.where(N_mask)[0]
    n_N = len(N)

    # Sparsity: z_y columns mirror A's column nnz; z_λ columns have m+1 entries.
    d = max(int(A.getnnz(axis=0).max()) if A.nnz > 0 else 0, m + 1)

    A_B_lu = splu(A[:, B].tocsc())
    A_N = A[:, N]

    # M z = [-Aᵀ z_y + V z_λ]  (x = s = 1)
    def _matvec(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64).ravel()
        z_y, z_lam = z[:m], z[m:]
        out = -np.asarray(A.T @ z_y, dtype=np.float64).ravel()
        if n_N > 0:
            sv = np.empty(n, dtype=np.float64)
            sv[B] = -A_B_lu.solve(np.asarray(A_N @ z_lam, dtype=np.float64).ravel())
            sv[N] = z_lam
            out += sv
        return out

    # Mᵀ u: first m → -A u; last n_N → -A_Nᵀ A_B⁻ᵀ u_B + u_N
    def _rmatvec(u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64).ravel()
        out = np.empty(n, dtype=np.float64)
        out[:m] = -np.asarray(A @ u, dtype=np.float64).ravel()
        if n_N > 0:
            out[m:] = -np.asarray(
                A_N.T @ A_B_lu.solve(u[B], trans="T"), dtype=np.float64
            ).ravel() + u[N]
        return out

    M_op = LinearOperator((n, n), matvec=_matvec, rmatvec=_rmatvec, dtype=np.float64)
    sv_max = float(svds(M_op, k=1, which="LM", return_singular_vectors=False)[0])
    sv_min = float(svds(M_op, k=1, which="SM", return_singular_vectors=False)[0])
    k = sv_max / sv_min if sv_min > 0.0 else float("inf")

    count = (
        gate_count_qlsa(d=d, k=k)
        + gate_count_state_preparation(np.arange(1.0, n + 1))
        + n
    )
    return count, d, k



def _load_standard_form(path: Path) -> csr_matrix:
    """Load A from .std standard-form LP (npz format). A returned as CSR."""
    data = np.load(path)
    A_data = np.asarray(data["A_data"], dtype=np.float64).ravel()
    A_indices = np.asarray(data["A_indices"], dtype=np.int64).ravel()
    A_indptr = np.asarray(data["A_indptr"], dtype=np.int64).ravel()
    A_shape = np.asarray(data["A_shape"], dtype=np.int64).ravel()
    m, n = int(A_shape[0]), int(A_shape[1])
    return csr_matrix((A_data, A_indices, A_indptr), shape=(m, n))


def _benchmark_instance_from_path(
    path: Path,
    variant: str = "both",
) -> None:
    """Load A from .std; compute gate counts, write to instance .data (JSON). path must be .std."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Instance file not found: {path}")
    if path.suffix.lower() != ".std":
        raise ValueError(f"Path must be .std; got {path.suffix!r}")
    if variant not in ("mnes", "oss", "both"):
        raise ValueError(f"variant must be 'mnes', 'oss', or 'both'; got {variant!r}")

    base_name = path.name[: -len(".std")]
    A = _load_standard_form(path)

    if A.shape[0] > 100_000:
        return

    data_path = path.parent / (base_name + ".data")
    data = json.loads(data_path.read_text()) if data_path.exists() else {}

    if variant in ("mnes", "both"):
        count, sparsity, cond = _gate_count_qipm1(A)
        data["gate_count_qipm1"] = count
        data["sparsity_qipm1"] = sparsity
        data["cond_qipm1"] = cond

    if variant in ("oss", "both"):
        count, sparsity, cond = _gate_count_qipm2(A)
        data["gate_count_qipm2"] = count
        data["sparsity_qipm2"] = sparsity
        data["cond_qipm2"] = cond

    data_path.write_text(json.dumps(data, indent=None))


def benchmark_instance(
    instance_class: str,
    instance_name: str,
    cache_dir: str | Path | None = None,
    variant: str = "both",
) -> None:
    """Run gate-count benchmark for the instance in cache_dir/instance_class/instance_name/.

    Discovers the instance by .std (exactly one). Loads A from that file; writes gate counts to instance .data (JSON).
    instance_class: e.g. "netlib", "miplib".
    instance_name: subfolder name (instance stem).
    cache_dir: root containing instance-class subfolders; defaults to "cache_dir".
    variant: 'mnes' (qipm1), 'oss' (qipm2), or 'both' (default).
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
    _benchmark_instance_from_path(std_files[0], variant=variant)


def benchmark_instance_class(
    instance_class: str,
    variant: str = "both",
    cache_dir: str | Path | None = None,
) -> None:
    """Run gate-count benchmark for all .std instances in the given instance-class subfolder of cache_dir.

    instance_class: name of the subfolder (e.g. "netlib", "miplib").
    variant: 'mnes' (qipm1), 'oss' (qipm2), or 'both' (default).
    cache_dir: directory containing instance-class subfolders; defaults to "cache_dir" in the current directory.
    """
    root = Path(cache_dir).resolve() if cache_dir is not None else Path("cache_dir").resolve()
    folder = root / instance_class
    if not folder.is_dir():
        raise FileNotFoundError(f"Instance class folder not found: {folder}")

    subdirs = sorted(d for d in folder.iterdir() if d.is_dir())
    for subdir in tqdm(subdirs, desc=instance_class, unit="instance"):
        benchmark_instance(instance_class, subdir.name, cache_dir=root, variant=variant)


def benchmark_all_instance_classes(
    instance_classes: list[str] | None = None,
    variant: str = "both",
    cache_dir: str | Path | None = None,
) -> None:
    """Run gate-count benchmark for .std instances (main entry point).

    instance_classes: optional list of instance class names (subfolder names under cache_dir).
        If None, all instance classes (all subdirectories of cache_dir) are processed.
    variant: 'mnes' (qipm1), 'oss' (qipm2), or 'both' (default).
    cache_dir: directory containing instance-class subfolders; defaults to "cache_dir" in the current directory.
    """
    root = Path(cache_dir).resolve() if cache_dir is not None else Path("cache_dir").resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Cache directory not found: {root}")

    if instance_classes is None:
        instance_classes = [f.name for f in sorted(root.iterdir()) if f.is_dir()]

    for name in instance_classes:
        benchmark_instance_class(name, variant=variant, cache_dir=root)


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
        choices=["mnes", "oss", "both"],
        default="both",
        help="Which qipm variant to run: 'mnes' (qipm1), 'oss' (qipm2), or 'both' (default).",
    )
    args = parser.parse_args()
    benchmark_all_instance_classes(
        instance_classes=args.instance_classes or None,
        variant=args.qipm,
        cache_dir=args.cache_dir,
    )
