# sonium-solvers-iterative

Iterative Krylov solvers for large sparse and operator-form linear
systems.

## Scope

- GMRES(m), flexible GMRES, deflated and recycled variants
  (including GCRO-DR) for general nonsymmetric systems.
- BiCGSTAB, IDR(s) and IDR(s)stab, TFQMR, and CGS for short-recurrence
  nonsymmetric solves.
- CG for SPD systems and MINRES for Hermitian indefinite systems.
- Pipelined and block variants where the workload amortises them,
  with checkpointable solver state for long runs.

All solvers share the workspace result contract: convergence is
reported from residuals, never assumed from iteration counts.

## Development

```bash
cargo test -p sonium-solvers-iterative
```

See `CHANGELOG.md` for release history.
