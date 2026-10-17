# sonium-solvers-sparse

Sparse matrix formats and SpMV execution plans for the Sonium
solver family.

## Scope

- CSR and BSR storage with builders, triplet assembly, and
  submatrix extraction.
- SpMV plans (standard, row-bucketed, SELL-C-σ, NUMA-partitioned)
  that analyse a matrix once and reuse the schedule across solves.
- Composite, scaled, summed, and function-backed operator adapters
  implementing the shared `LinearOperator` contract.

Plans are explicit values: callers can inspect the analysis report
before committing to a format, and hot-path SpMV stays
allocation-free once a plan is built.

## Development

```bash
cargo test -p sonium-solvers-sparse
```

See `CHANGELOG.md` for release history.
