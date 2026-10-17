# sonium-solvers-nonlinear

Nonlinear solvers for systems `F(x) = 0`.

## Scope

- Newton–Raphson with exact, inexact, and continuation variants
  and configurable damping.
- Anderson acceleration for fixed-point and nonlinear iterations.

Solvers are generic over the workspace scalar field and report
convergence through the shared solver-result contract.

## Development

```bash
cargo test -p sonium-solvers-nonlinear
```

See `CHANGELOG.md` for release history.
