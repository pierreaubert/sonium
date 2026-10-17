# sonium-solvers-optimization

Unconstrained and constrained optimisation algorithms.

## Scope

- Derivative-free: Nelder–Mead, Powell, BOBYQA, CMA-ES,
  differential evolution, simulated annealing, and seeded random
  search.
- Quasi-Newton and gradient methods: BFGS, L-BFGS and L-BFGS-B,
  nonlinear CG, SR1/DFP, and Hessian-free truncated Newton.
- Constrained: SLSQP/SQP, trust-region constrained, primal-dual
  interior-point, MMA, augmented Lagrangian, penalty, and
  epsilon-constraint methods.
- Gaussian-process Bayesian optimisation and NSGA-II/III
  multi-objective search.

Bound handling and convergence reporting are explicit in each
solver's configuration; no method silently relaxes constraints.

## Development

```bash
cargo test -p sonium-solvers-optimization
```

See `CHANGELOG.md` for release history.
