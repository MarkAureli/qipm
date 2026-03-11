#!/usr/bin/env python3
"""Benchmark LP instances: A from .std; compute gate counts for qipm1/2 and write to instance .data (JSON)."""

from __future__ import annotations

import json
import math
import multiprocessing
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

_EPSILON = 1e-1  # precision shared by QLSA and outer Newton-step count
_PREPROCESS_TIMEOUT = 600  # seconds; basis preprocessing time limit


def gate_count_qlsa(
    *,
    d: int,
    k: float,
    epsilon: float = 1e-1,
) -> int:
    """
    Return the QLSA Chebyshev query count.

    Args:
        d: Maximum sparsity (non-zeros per row or column) of M̂.
        k: Condition number (2-norm) of M̂.
        epsilon: Precision (default 1e-1).

    Returns:
        int: The number of queries that QLS Chebyshev makes to O_H and O_F (P_A).
    """
    binst = math.ceil(math.log(d * k / epsilon) * (d * k) ** 2)
    insqrt = binst * math.log(4 * binst / epsilon)
    j0_val = int(math.ceil(math.sqrt(insqrt)))
    return 8 * j0_val


def _preprocess_basis_worker(queue: multiprocessing.Queue, A: csr_matrix) -> None:
    """Subprocess worker for _preprocess_basis; puts result or exception into queue."""
    import sparseqr
    try:
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

        queue.put((A, m, n, B, N, n_N, A[:, B].tocsc(), A[:, N]))
    except Exception as exc:  # noqa: BLE001
        queue.put(exc)


def _preprocess_basis(A: csr_matrix):
    """Shared preprocessing for both qipm variants.

    Returns (A, m, n, B, N, n_N, A_B_lu, A_N) after SPQR basis selection,
    optional rank-deficiency row reduction, and LU factorisation of A_B.
    Raises RuntimeError if preprocessing exceeds _PREPROCESS_TIMEOUT seconds.
    """
    from scipy.sparse.linalg import splu

    q: multiprocessing.Queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_preprocess_basis_worker, args=(q, A))
    p.start()
    try:
        p.join(_PREPROCESS_TIMEOUT)
        if p.is_alive():
            p.terminate()
            p.join(5)
            if p.is_alive():
                p.kill()
                p.join()
            raise RuntimeError(
                f"Basis preprocessing exceeded {_PREPROCESS_TIMEOUT // 60}-minute time limit"
            )
        result = q.get_nowait()
        if isinstance(result, Exception):
            raise result
        A, m, n, B, N, n_N, A_B_csc, A_N = result
        return A, m, n, B, N, n_N, splu(A_B_csc), A_N
    finally:
        q.close()
        q.cancel_join_thread()
        p.close()


def _gate_count_qipm1_from_basis(
    A: csr_matrix,
    m: int,
    n: int,
    B: np.ndarray,
    N: np.ndarray,
    n_N: int,
    A_B_lu,
    A_N: csr_matrix,
) -> tuple[int, int, float]:
    """Compute (gate_count, sparsity, cond) for qipm1 from preprocessed basis."""
    from scipy.sparse.linalg import LinearOperator, eigsh

    d = m  # M̂ is generically dense m×m

    if n_N == 0 or m <= 1:
        k = 1.0
    else:
        def _fbar_mv(v: np.ndarray) -> np.ndarray:
            return A_B_lu.solve(np.asarray(A_N @ v, dtype=np.float64).ravel())

        def _fbar_rmv(u: np.ndarray) -> np.ndarray:
            return np.asarray(A_N.T @ A_B_lu.solve(u, trans="T"), dtype=np.float64).ravel()

        def _mhat_mv(v: np.ndarray) -> np.ndarray:
            v = np.asarray(v, dtype=np.float64).ravel()
            return v + _fbar_mv(_fbar_rmv(v))

        M_op = LinearOperator((m, m), matvec=_mhat_mv, dtype=np.float64)
        k = float(eigsh(M_op, k=1, which="LM")[0][0])

    count = int(gate_count_qlsa(d=d, k=k, epsilon=_EPSILON) * (m - 1) / _EPSILON**2)
    return count, d, k


def _gate_count_qipm1(A: csr_matrix) -> tuple[int, int, float]:
    """Return (gate_count, sparsity, cond) for qipm1.

    Estimates κ(M̂) via M̂ = I + F̄F̄ᵀ, F̄ = A_B⁻¹ A_N (D_B = D_N = I).
    M̂ is m×m and generically dense, so d = m.
    """
    return _gate_count_qipm1_from_basis(*_preprocess_basis(A))


def _gate_count_qipm2_from_basis(
    A: csr_matrix,
    m: int,
    n: int,
    B: np.ndarray,
    N: np.ndarray,
    n_N: int,
    A_B_lu,
    A_N: csr_matrix,
) -> tuple[int, int, float]:
    """Compute (gate_count, sparsity, cond) for qipm2 from preprocessed basis.

    Estimates κ(M) for the OSS matrix M = [-Aᵀ | V] ∈ ℝⁿˣⁿ (x = s = 1).
    V ∈ ℝⁿˣ⁽ⁿ⁻ᵐ⁾ is the null-space basis built from the SPQR pivot basis B:
        V[B, :] = -A_B⁻¹ A_N,  V[N, :] = I_{n-m}.

    Sparsity d = max(max column nnz of A, m + 1):
    - z_y columns of M have the same sparsity as columns of A,
    - z_λ columns have m entries in B-rows (dense A_B⁻¹ A_N column) + 1 in N-rows.
    """
    from scipy.sparse.linalg import LinearOperator, svds

    # Sparsity: z_y columns mirror A's column nnz; z_λ columns have m+1 entries.
    d = max(int(A.getnnz(axis=0).max()) if A.nnz > 0 else 0, m + 1)

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
    k = float(svds(M_op, k=1, which="LM", return_singular_vectors=False)[0])
    count = int(gate_count_qlsa(d=d, k=k, epsilon=_EPSILON) * (n - 1) / _EPSILON**2)
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
    A = csr_matrix(A, dtype=np.float64)
    m, n = A.shape

    if n <= 1:
        k = 1.0
        count = int(gate_count_qlsa(d=1, k=k, epsilon=_EPSILON) * (n - 1) / _EPSILON**2)
        return count, 1, k

    return _gate_count_qipm2_from_basis(*_preprocess_basis(A))


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

    if variant == "both":
        m, n = A.shape
        if n > 1:
            try:
                basis = _preprocess_basis(A)
                count, sparsity, cond = _gate_count_qipm1_from_basis(*basis)
                data["gate_count_qipm1"] = count
                data["sparsity_qipm1"] = sparsity
                data["cond_qipm1"] = None if not math.isfinite(cond) else cond
                count, sparsity, cond = _gate_count_qipm2_from_basis(*basis)
                data["gate_count_qipm2"] = count
                data["sparsity_qipm2"] = sparsity
                data["cond_qipm2"] = None if not math.isfinite(cond) else cond
            except RuntimeError:
                data["gate_count_qipm1"] = data["sparsity_qipm1"] = data["cond_qipm1"] = None
                data["gate_count_qipm2"] = data["sparsity_qipm2"] = data["cond_qipm2"] = None
        else:
            count, sparsity, cond = _gate_count_qipm1(A)
            data["gate_count_qipm1"] = count
            data["sparsity_qipm1"] = sparsity
            data["cond_qipm1"] = None if not math.isfinite(cond) else cond
            count, sparsity, cond = _gate_count_qipm2(A)
            data["gate_count_qipm2"] = count
            data["sparsity_qipm2"] = sparsity
            data["cond_qipm2"] = None if not math.isfinite(cond) else cond
    elif variant == "mnes":
        try:
            count, sparsity, cond = _gate_count_qipm1(A)
            data["gate_count_qipm1"] = count
            data["sparsity_qipm1"] = sparsity
            data["cond_qipm1"] = None if not math.isfinite(cond) else cond
        except RuntimeError:
            data["gate_count_qipm1"] = data["sparsity_qipm1"] = data["cond_qipm1"] = None
    else:
        try:
            count, sparsity, cond = _gate_count_qipm2(A)
            data["gate_count_qipm2"] = count
            data["sparsity_qipm2"] = sparsity
            data["cond_qipm2"] = None if not math.isfinite(cond) else cond
        except RuntimeError:
            data["gate_count_qipm2"] = data["sparsity_qipm2"] = data["cond_qipm2"] = None

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


_BENCHMARK_DATA_KEYS = {
    "mnes": ("gate_count_qipm1", "sparsity_qipm1", "cond_qipm1"),
    "oss":  ("gate_count_qipm2", "sparsity_qipm2", "cond_qipm2"),
}


def show_benchmark_status(
    instance_classes: list[str] | None = None,
    variant: str = "both",
    cache_dir: str | Path | None = None,
) -> None:
    """Print how many instances per class have all benchmark keys present in their .data files.

    For each instance class, prints one line per active variant showing
    "<class>  [mnes: x/total]  [oss: x/total]".
    An instance counts as done when all _BENCHMARK_DATA_KEYS for the variant are present.
    """
    root = Path(cache_dir).resolve() if cache_dir is not None else Path("cache_dir").resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Cache directory not found: {root}")

    if instance_classes is None:
        instance_classes = [f.name for f in sorted(root.iterdir()) if f.is_dir()]

    active_variants = ["mnes", "oss"] if variant == "both" else [variant]

    for cls in instance_classes:
        folder = root / cls
        if not folder.is_dir():
            print(f"{cls}: directory not found")
            continue

        subdirs = sorted(d for d in folder.iterdir() if d.is_dir())
        total = len(subdirs)

        counts: dict[str, int] = {v: 0 for v in active_variants}
        for subdir in subdirs:
            data_files = list(subdir.glob("*.data"))
            data = json.loads(data_files[0].read_text()) if data_files else {}
            for v in active_variants:
                if all(k in data for k in _BENCHMARK_DATA_KEYS[v]):
                    counts[v] += 1

        parts = "  ".join(f"{v}: {counts[v]}/{total}" for v in active_variants)
        print(f"{cls}:  {parts}")


def clear_benchmark_data(
    instance_classes: list[str] | None = None,
    cache_dir: str | Path | None = None,
    variant: str = "both",
) -> None:
    """Remove benchmark gate-count entries from .data files."""
    root = Path(cache_dir).resolve() if cache_dir is not None else Path("cache_dir").resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Cache directory not found: {root}")

    keys = (
        _BENCHMARK_DATA_KEYS["mnes"] + _BENCHMARK_DATA_KEYS["oss"]
        if variant == "both"
        else _BENCHMARK_DATA_KEYS[variant]
    )

    search_roots = [root / name for name in instance_classes] if instance_classes else [root]
    for search_root in search_roots:
        for data_path in search_root.rglob("*.data"):
            data = json.loads(data_path.read_text())
            if any(k in data for k in keys):
                for k in keys:
                    data.pop(k, None)
                data_path.write_text(json.dumps(data, indent=None))


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
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Remove benchmark entries from .data files instead of benchmarking. Other flags are ignored.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show how many instances per class have benchmark data for the selected variant(s). Other flags are ignored.",
    )
    args = parser.parse_args()
    if args.show:
        show_benchmark_status(
            instance_classes=args.instance_classes or None,
            variant=args.qipm,
            cache_dir=args.cache_dir,
        )
    elif args.clear:
        clear_benchmark_data(
            instance_classes=args.instance_classes or None,
            cache_dir=args.cache_dir,
            variant=args.qipm,
        )
    else:
        benchmark_all_instance_classes(
            instance_classes=args.instance_classes or None,
            variant=args.qipm,
            cache_dir=args.cache_dir,
        )
