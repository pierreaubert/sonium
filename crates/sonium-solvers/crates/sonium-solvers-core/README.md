# sonium-solvers-core

Core traits and dense matrix primitives for the Sonium solver
family: the `ComplexField` scalar contract, `LinearOperator` and
`Preconditioner` traits, shared solver-result contracts, and the
row-major dense matrix used across solver crates.

## Scope

- Scalar-generic contracts (`f32`, `f64`, `Complex32`, `Complex64`)
  implemented once and reused by every solver crate.
- Dense matrix storage and BLAS-style helpers with explicit
  ownership, so hot paths stay allocation-free.
- Precision-cast and vector-operation helpers shared by direct,
  iterative, eigenvalue, and nonlinear solvers.

Most users should depend on the `sonium-solvers` umbrella crate;
this crate is the foundation layer for solver implementers.

## Development

```bash
cargo test -p sonium-solvers-core
```

See `CHANGELOG.md` for release history.
