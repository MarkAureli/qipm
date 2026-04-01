# QIPM

Benchmarks quantum gate counts for solving linear programs (LPs) of the form

$$\min\ c^\top x \quad \text{s.t.}\ Ax = b,\ x \geq 0$$

via quantum interior-point methods (QIPMs). The key quantity is $\kappa(\hat{M})$, the condition number of the system matrix, which drives the gate count of the underlying Quantum Linear System Algorithm (QLSA).

## Background

Two QIPM variants are benchmarked:

| Variant | System | Matrix | Size |
|---------|--------|--------|------|
| `mnes` | Modified Normal Equation System | $\hat{M} = I + \bar{F}\bar{F}^\top$ | $m \times m$ |
| `oss` | Orthogonal Subspaces System | $M = [-A^\top \mid V]$ | $n \times n$ |

Both condition numbers are estimated matrix-free via ARPACK on a `LinearOperator`, with basis selection via SuiteSparse SPQR.

### MNES — `mnes`

A basis $B \subset \{1,\ldots,n\}$ of size $m$ is selected by column-pivoted QR (SPQR) on $A$, giving $A = [A_B \mid A_N]$. The reduced matrix $\bar{F} = A_B^{-1} A_N \in \mathbb{R}^{m \times (n-m)}$ is never formed explicitly. Instead, $\hat{M}$ is wrapped as a `LinearOperator` with matvec

$$v \mapsto v + \bar{F}(\bar{F}^\top v) = v + A_B^{-1}\bigl(A_N (A_N^\top (A_B^{-\top} v))\bigr),$$

where solves against $A_B$ use a sparse LU factorisation. Since $\hat{M} \succeq I$, all eigenvalues are $\geq 1$. `eigsh` finds $\lambda_\max$ (largest magnitude) and $\lambda_\min$ (smallest magnitude) separately; when $n - m < m$ the nullspace of $\bar{F}\bar{F}^\top$ guarantees $\lambda_\min = 1$ exactly and the second `eigsh` call is skipped. The condition number is $\kappa(\hat{M}) = \lambda_\max / \lambda_\min$.

### OSS — `oss`

Using the same SPQR basis $B$, the null-space basis $V \in \mathbb{R}^{n \times (n-m)}$ is defined implicitly by

$$V_B = -A_B^{-1} A_N, \qquad V_N = I_{n-m}.$$

The system matrix $M = [-A^\top \mid V]$ (evaluated at $x = s = \mathbf{1}$) is wrapped as a `LinearOperator` whose matvec and adjoint-matvec are computed via sparse matrix–vector products and triangular solves against $A_B$. The condition number is $\kappa(M) = \sigma_\max / \sigma_\min$, estimated by two calls to `svds` (largest and smallest singular value). The QLSA sparsity parameter is $d = \max(\text{max col-nnz}(A),\ m+1)$: the first $m$ columns of $M$ inherit the sparsity of $A$, while the remaining $n-m$ columns each have $m$ non-zeros in the $B$-rows (from $A_B^{-1}A_N$) plus one in the $N$-rows.

## Pipeline

```
extract → transform → solve → benchmark
```

Each stage writes results into `cache_dir/<class>/<name>/`:

| File | Contents |
|------|----------|
| `<name>.mps` | Original MPS file |
| `<name>.std` | Standard-form LP (NPZ: sparse `A`, vectors `b`, `c`) |
| `<name>.data` | JSON accumulating gate counts, sparsity, condition numbers, runtimes |

Instance classes: `clique`, `independent_set`, `max_flow`, `miplib`, `misc`, `netlib`, `stochlp`, `vertex_cover`.

## Usage

### 1. Extract

Clone [simplex-benchmarks](https://github.com/mtanneau/simplex-benchmarks) (requires Git LFS) and populate `cache_dir`:

```bash
python extract.py
# or point to an existing clone:
python extract.py /path/to/simplex-benchmarks
```

### 2. Transform

Convert MPS instances to standard-form LP via HiGHS presolve:

```bash
python transform.py                        # all instance classes
python transform.py netlib miplib          # selected classes
python transform.py --cache-dir /my/cache
```

### 3. Solve (optional)

Solve instances with HiGHS and record solve time:

```bash
python solve.py
python solve.py --formats std netlib
```

### 4. Benchmark

Compute QLSA gate counts and write to `.data`:

```bash
python benchmark.py                        # all classes, both variants
python benchmark.py --qipm mnes            # MNES only
python benchmark.py --qipm oss netlib      # OSS, netlib class only
python benchmark.py --cache-dir /my/cache
```

`--qipm` accepts `mnes`, `oss`, or `both` (default).

## Installation

```bash
# System dependency (macOS)
brew install suite-sparse

pip install -r requirements.txt
```

**Dependencies:** `numpy`, `scipy`, `highspy`, `qiskit`, `sparseqr`, `tqdm`.

## Tests

```bash
python -m pytest tests/
```
