# sonium-solvers-sys

System-level runtime support for the solver family: runtime
detection, NUMA topology, lane execution, and parallel helpers.

## Scope

- Runtime capability and topology detection shared by solver
  crates (consumers should not re-probe the machine).
- A lane executor for bounded parallel solver work with explicit
  thread budgets.
- Rayon helpers that keep solver parallelism machine-adaptive
  rather than hard-coded to one host.

Placement and threading never change numerics; they only decide
where and how widely work runs.

## Development

```bash
cargo test -p sonium-solvers-sys
```

See `CHANGELOG.md` for release history.
