# sonium-solvers-preconditioners

Preconditioners for the iterative solver family, from diagonal
scaling to algebraic multigrid.

## Scope

- Jacobi/diagonal, ILU(0), ILUT, hierarchical ILU (HILU), and
  parallel ILU variants (coloring, fixed-point, ParILUT).
- Block Jacobi, additive Schwarz domain decomposition, and AMG
  with parallel coarsening and smoothing.
- Saddle-point block preconditioners (PCD/LSC-style Schur
  strategies) and polynomial/Chebyshev backend preconditioners.

The crate README in `sonium-solvers` carries the physics-to-
preconditioner selection table; as a rule, an unpreconditioned
Krylov solve on an indefinite Helmholtz system will stall, so
pairing guidance matters more than defaults.

## Development

```bash
cargo test -p sonium-solvers-preconditioners
```

See `CHANGELOG.md` for release history.
