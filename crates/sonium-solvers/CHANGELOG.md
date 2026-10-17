# Changelog

## 0.5.0

### Changed

- Bumped the package version from 0.4.6 to 0.5.0 as part of the coordinated minor release.

## 0.4.5

### Architecture

- Split the solver implementation into the focused `sonium-solvers-core`,
  `sonium-solvers-sys`, `sonium-solvers-sparse`, `sonium-solvers-eigen`,
  `sonium-solvers-direct`, `sonium-solvers-preconditioners`,
  `sonium-solvers-iterative`, `sonium-solvers-optimization`,
  `sonium-solvers-nonlinear`, and `sonium-solvers-time-integration` crates.
  `sonium-solvers` remains the compatibility facade.

### Scripts

- Added a script to generate performance status, results in PERF.md

### Direct

- Blocked LU
- Implemented sparse LU still far away from SuperLU, partial COLAMD algo
- Perf improvemement (x10, still not competitive with Accelerate, 2x to 3x slower)

### Iterative

- Improved BiCGSTAB: added support for unconjugate dots, initial guess and residual replacement
- Fixed p-GMRES convergence
- Implemented HILU properly


### Optimisation

- Implemented proper line search with HZ

### Eigenvalues

- Better implementation across the board: LOGPCG, Newmark-Beta, Newton-Rraphson, mixed precision, CSR SpMV,
- Mini partionning and NUMA dispatch
	

## 0.4.4

- Improve parallelisation
- Upgraded to new sonium-compute

## 0.4.3

### Fixed

- Givens rotation in GMRES/FGMRES used `sqrt(a² + b²)` which overflows for large norms
  (`|a| > 1e154` for f64). Replaced with the numerically safe scaled form
  `scale * sqrt((a/scale)² + (b/scale)²)` where `scale = max(|a|, |b|)`.
- ILU(0) factorization used `break` on `k >= i` to skip upper-triangular entries, which
  silently produced wrong factors when CSR rows had unsorted column indices. Changed to
  `continue` so all lower-triangular entries are processed regardless of column order.
- ILU(0) U-row update fallback search started at `row_ptrs[k] + 1` instead of
  `diag_indices[k] + 1`, potentially skipping the first upper-triangular entry.
- IDR(s) shadow space was generated with zero imaginary parts for complex types (`Complex32`,
  `Complex64`), reducing convergence and increasing breakdown probability. Now generates both
  real and imaginary components.
- GMRES adaptive restart ratio used `g[m]` (least-squares residual from within the completed
  cycle) vs the true residual at restart entry, which could diverge in floating-point and cause
  incorrect restart window decisions.
