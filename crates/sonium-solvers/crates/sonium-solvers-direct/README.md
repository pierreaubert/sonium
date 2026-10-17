# sonium-solvers-direct

Direct solvers for dense and sparse linear systems.

## Scope

- Dense LU with partial pivoting and dense Cholesky, generic over
  the solver scalar types.
- Sparse LU (COLAMD ordering) and simplicial sparse Cholesky for
  SPD/HPD systems, including a supernodal BLAS-3 numeric path.
- Multifrontal BLR and LDLᵀ/LDLᴴ factorisations with compressed
  off-diagonal frontal panels and configurable pivoting.

Checked factor solves report their attained status explicitly;
attach an independently recomputed residual before publishing an
accuracy claim (see the `sonium-solvers` umbrella documentation).

## Development

```bash
cargo test -p sonium-solvers-direct
```

See `CHANGELOG.md` for release history.
