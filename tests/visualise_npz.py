#!/usr/bin/env python3
"""Visualise npz fixture files (.std, .sde, .init) in human-readable form on the console.

Usage:
    python tests/visualise_npz.py tests/fixtures/min_sum.std
    python tests/visualise_npz.py tests/fixtures/equality.sde
    python tests/visualise_npz.py tests/fixtures/equality.init
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


def _format_array(arr: np.ndarray, max_entries: int = 50) -> str:
    """Format a 1D array for console output, optionally truncating."""
    arr = np.asarray(arr).ravel()
    n = arr.size
    if n == 0:
        return "[]"
    if n <= max_entries:
        return np.array2string(arr, precision=6, suppress_small=True)
    head = np.array2string(arr[: max_entries // 2], precision=6, suppress_small=True)
    tail = np.array2string(arr[-(max_entries - max_entries // 2) :], precision=6, suppress_small=True)
    return f"{head[:-1]} ... {tail[1:]}  # length {n}"


def _print_standard_form(data: np.lib.npyio.NpzFile, path: Path) -> None:
    """Print standard-form LP (A, b, c) from .std / .sde npz."""
    suffix = path.suffix.lower()
    label = "Self-dual embedding (SDE)" if suffix == ".sde" else "Standard form (std)"
    print(f"# {path.name} — {label}")
    print()

    c = np.asarray(data["c"], dtype=np.float64).ravel()
    b = np.asarray(data["b"], dtype=np.float64).ravel()
    A_data = np.asarray(data["A_data"], dtype=np.float64).ravel()
    A_indices = np.asarray(data["A_indices"], dtype=np.int64).ravel()
    A_indptr = np.asarray(data["A_indptr"], dtype=np.int64).ravel()
    A_shape = np.asarray(data["A_shape"], dtype=np.int64).ravel()
    m, n = int(A_shape[0]), int(A_shape[1])
    A = csr_matrix((A_data, A_indices, A_indptr), shape=(m, n))

    print("Dimensions: m (rows) =", m, ", n (cols) =", n)
    print()
    print("c (objective, length n):")
    print(_format_array(c))
    print()
    print("b (RHS, length m):")
    print(_format_array(b))
    print()
    print("A (m×n, CSR): nnz =", A.nnz)
    print("A dense:")
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    print(A.toarray())
    np.set_printoptions()  # reset


def _print_init(data: np.lib.npyio.NpzFile, path: Path) -> None:
    """Print initial triple (x, y, s) and embedding_used from .init npz."""
    print(f"# {path.name} — Initial triple (.init)")
    print()

    x = np.asarray(data["x"], dtype=np.float64).ravel()
    y = np.asarray(data["y"], dtype=np.float64).ravel()
    s = np.asarray(data["s"], dtype=np.float64).ravel()
    embedding_used = np.asarray(data["embedding_used"], dtype=bool)
    if embedding_used.size == 1:
        embedding_used = bool(embedding_used.flat[0])
    else:
        embedding_used = embedding_used.tolist()

    print("embedding_used:", embedding_used)
    print()
    print("x (primal, length n):")
    print(_format_array(x))
    print()
    print("y (dual, length m):")
    print(_format_array(y))
    print()
    print("s (slack, length n):")
    print(_format_array(s))


def _print_generic_npz(data: np.lib.npyio.NpzFile, path: Path) -> None:
    """Print all arrays in an npz file with names and shapes."""
    print(f"# {path.name} — npz contents")
    print()
    for key in sorted(data.files):
        arr = data[key]
        arr = np.asarray(arr)
        print(f"{key}: shape {arr.shape}, dtype {arr.dtype}")
        if arr.size <= 100:
            np.set_printoptions(precision=6, suppress=True, linewidth=100)
            print(arr)
            np.set_printoptions()
        else:
            print(_format_array(arr.ravel(), max_entries=30))
        print()


def visualise(path: Path) -> None:
    """Load npz at path and print human-readable content to stdout."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    data = np.load(path, allow_pickle=False)
    suffix = path.suffix.lower()

    if suffix in (".std", ".sde"):
        if not all(k in data.files for k in ("c", "b", "A_data", "A_indices", "A_indptr", "A_shape")):
            _print_generic_npz(data, path)
        else:
            _print_standard_form(data, path)
    elif suffix == ".init":
        if not all(k in data.files for k in ("x", "y", "s", "embedding_used")):
            _print_generic_npz(data, path)
        else:
            _print_init(data, path)
    else:
        _print_generic_npz(data, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print npz fixture files (.std, .sde, .init) in human-readable form."
    )
    parser.add_argument(
        "file",
        type=Path,
        help="Path to .npz-style file (e.g. tests/fixtures/min_sum.std or .sde or .init)",
    )
    args = parser.parse_args()
    try:
        visualise(args.file)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
