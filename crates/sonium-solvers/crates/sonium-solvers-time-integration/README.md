# sonium-solvers-time-integration

Time-integration methods for second-order systems
`M u″ + C u′ + K u = f(t)` and related transient problems.

## Scope

- Implicit Newmark-β and generalized-α schemes with consistent
  tangent handling.
- Explicit central-difference integration for lumped-mass,
  conditionally stable workloads.
- Adjoint checkpointing (revolve-style) and transient adjoint
  sensitivities for optimisation over time-dependent responses.

Schemes report their stability and accuracy assumptions in their
configuration types; explicit methods document their step-size
limits rather than discovering them at runtime.

## Development

```bash
cargo test -p sonium-solvers-time-integration
```

See `CHANGELOG.md` for release history.
