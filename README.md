# QIPM

Benchmarks quantum cycle counts for solving linear programs (LPs) of the form

$$\min\ c^\top x \quad \text{s.t.}\ Ax = b,\ x \geq 0$$

via quantum interior-point methods (QIPMs). The key quantity is $\kappa(\hat{M})$, the condition number of the system matrix, which drives the cycle count of the underlying Quantum Linear System Algorithm (QLSA).

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

The system matrix $M = [-A^\top \mid V]$ (evaluated at $x = s = \mathbf{1}$) is wrapped as a `LinearOperator` whose matvec and adjoint-matvec are computed via sparse matrix–vector products and triangular solves against $A_B$. The condition number is $\kappa(M) = \sigma_\max / \sigma_\min$, estimated by two calls to `svds` (largest and smallest singular value). The QLSA sparsity parameter is $s = \max(\text{max row-nnz}(A),\ m+1)$: the first $m$ columns of $M$ are the columns of $-A^\top$, so column $j$ has as many non-zeros as row $j$ of $A$, while the remaining $n-m$ columns each have $m$ non-zeros in the $B$-rows (from $A_B^{-1}A_N$) plus one in the $N$-rows.

## Pipeline

```
extract → transform → solve → benchmark
```

Each stage writes results into `cache_dir/<class>/<name>/`:

| File | Contents |
|------|----------|
| <name>.mps | Original MPS file |
| <name>.std | Standard-form LP (NPZ: sparse `A`, vectors `b`, `c`) |
| <name>.data | JSON accumulating cycle counts, sparsity, condition numbers, runtimes |

Instance classes: `clique`, `independent_set`, `max_flow`, `miplib`, `misc`, `netlib`, `stochlp`, `vertex_cover`.

## Usage

### 1. Extract

Populate `cache_dir` from the simplex-benchmarks repository (requires Git LFS):

```bash
python extract.py                          # clone automatically, then clean up
python extract.py /path/to/simplex-benchmarks  # use existing clone
```

Without an argument, the script performs a shallow clone (`--depth 1`) of the repository, pulls LFS objects, extracts the relevant data, then deletes the temporary clone. Passing a path skips the clone entirely.

Two types of data are extracted:

**MPS instances** — each zip in `mps/` is unpacked and its `.mps` files (from `min/` or `max/` subdirectories) are placed under `cache_dir/<class>/<stem>/`. Instance classes are assigned as follows:

| Zip | Class |
|-----|-------|
| netlib, miplib, stochlp, misc | same name |
| max_flow, random_directed_graphs | max_flow |
| clq_mis_vc_dimacs, clq_mis_vc_random | by filename prefix: `clq`→clique, `is`→independent_set, `vc`→vertex_cover |

**GLPK runtimes** — the evaluation results in `benchmark/01_evaluation/` contain compressed archives of `.data` files with per-instance GLPK solve times (`runtime_primal`). These are extracted and written as `{"runtime_glpk": ...}` into each instance's `.data` file.

### 2. Transform

Convert MPS instances to standard-form LP via HiGHS presolve:

```bash
python transform.py                        # all instance classes
python transform.py netlib miplib          # selected classes
python transform.py --cache-dir /my/cache
```

Each MPS file is first presolved by HiGHS, then the reduced LP is algebraically converted to standard form $\min c^\top x$ s.t. $Ax = b,\ x \geq 0$. The conversion handles all bound and row types:

| Variable type | Transformation |
|---|---|
| Bounded $l \leq x \leq u$ | $x = l + x_1$, add row $x_1 + s = u - l$,\ $x_1, s \geq 0$ |
| Lower-bounded $l \leq x$ | Shift $x_1 = x - l \geq 0$ |
| Upper-bounded $x \leq u$ | Negate $x_1 = u - x \geq 0$ |
| Free $x \in \mathbb{R}$ | Split $x = x^+ - x^-$,\ $x^+, x^- \geq 0$ |

| Row type | Treatment |
|---|---|
| Equality $Ax = b$ | Kept as-is |
| $\leq$ inequality | Add slack $s \geq 0$ |
| $\geq$ inequality | Add surplus $s \geq 0$ (negated) |
| Range $l \leq Ax \leq u$ | Add slack for upper bound; extra row $s_1 + s_2 = u - l$ |
| Free row | Dropped entirely |

After conversion, zero rows (ghost equality rows that survive presolve unreferenced) are removed, as they make $A$ rank-deficient without adding any constraint. The result is saved as a compressed NPZ file with extension `.std`.

### 3. Solve (optional)

Solve instances with HiGHS and record solve time:

```bash
python solve.py                            # all classes, both formats
python solve.py --formats std netlib       # .std only, netlib class
python solve.py --formats mps              # .mps only
```

Each instance is solved in two independent modes, controlled by `--formats`:

| Format | Input | HiGHS model | Output key |
|--------|-------|-------------|------------|
| mps | Raw `.mps` | Read directly, HiGHS selects solver | runtime_highs_mps |
| std | NPZ `.std` | Equality LP $(Ax = b,\ x \geq 0)$ reconstructed from $A, b, c$ | runtime_highs_std |

For `.std`, if the default solver fails (e.g. due to poor scaling), the solve is automatically retried with HiGHS's interior-point method.

Each solve runs in a subprocess with a 10-minute timeout. In `both` mode, if the `.mps` solve times out, the `.std` solve is skipped for that instance. Wall-clock solve times are written to the instance's `.data` JSON and serve as the classical baseline for the quantum advantage comparison.

### 4. Benchmark

Compute QLSA cycle counts and write to `.data`:

```bash
python benchmark.py                        # all classes, both variants
python benchmark.py --qipm mnes            # MNES only
python benchmark.py --qipm oss netlib      # OSS, netlib class only
python benchmark.py --cache-dir /my/cache
```

`--qipm` accepts `mnes`, `oss`, or `both` (default).

For each instance, the script reads $A$ from the `.std` file and writes three keys per variant into the instance's `.data` JSON: the cycle count (`cycle_count_mnes` / `cycle_count_oss`), the sparsity parameter $s$ (`sparsity_mnes` / `sparsity_oss`), and the condition number $\kappa$ (`cond_mnes` / `cond_oss`).

**Basis preprocessing** — shared by both variants: SPQR (column-pivoted QR on $A$) selects a basis $B$ of size $m$ and identifies the non-basic columns $N$. If $A$ is rank-deficient, a secondary SPQR on $A^\top$ drops redundant rows. A sparse LU factorisation of $A_B$ is then computed once and reused by both variants for all subsequent triangular solves.

**Condition estimation** — see the [Background](#background) section for the matrix definitions. In brief: MNES uses two `eigsh` calls on $\hat{M} = I + \bar{F}\bar{F}^\top$ (wrapped as a `LinearOperator`) to find $\lambda_\max$ and $\lambda_\min$, giving $\kappa = \lambda_\max / \lambda_\min$. OSS uses two `svds` calls on $M = [-A^\top \mid V]$ for $\sigma_\max$ and $\sigma_\min$, giving $\kappa = \sigma_\max / \sigma_\min$.

**Cycle count formula** — a single QLSA call costs `cycle_count_qlsa(s, κ, ε)` cycles (Chebyshev query count). At least $(d-1)/\varepsilon^2$ measurements, hence that many repetitions of the QLSA, are required in order to obtain an approximate classical solution. The total cycle count is therefore

$$\text{cycle count} = \texttt{cycle\_count\_qlsa}(s,\kappa,\varepsilon) \times \frac{\dim - 1}{\varepsilon^2},$$

where $\dim = m$ (MNES) or $\dim = n$ (OSS), and $\varepsilon = 0.1$.

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
