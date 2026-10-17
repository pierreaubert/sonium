# sonium-solvers-eigen

Eigenvalue solvers for sparse and dense pencils.

## Scope

- LOBPCG for a few smallest eigenvalues of large sparse SPD
  problems, including generalized `A x = λ B x` pencils.
- Complex shift-invert Arnoldi for complex-shift problems.
- Dense nonsymmetric eigensolvers and divide-and-conquer paths for
  smaller pencils.
- Eigenpair sensitivities, continuation, and adjoint derivatives
  for optimisation workflows.

LOBPCG entry points are `f64` real-symmetric; sensitivity and
continuation APIs are generic over the solver scalar field.

## Development

```bash
cargo test -p sonium-solvers-eigen
```

See `CHANGELOG.md` for release history.
