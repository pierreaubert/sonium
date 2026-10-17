# sonium-topology

Machine placement for parallel work: topology discovery, CPU-set
selection, thread pinning, and bandwidth-width calibration.

## Scope

- Discovers NUMA nodes, cache domains, and physical-versus-SMT core
  structure from portable Linux sysfs topology handles — no vendor
  or CPUID dispatch.
- Selects CPU sets and streaming widths that saturate DRAM without
  oversubscribing shared cache domains.
- Respects the inherited affinity mask (containers, cgroups,
  `taskset`, `numactl`) and never changes numerics; placement is
  best-effort and degrades to scheduler placement off Linux.

Consumers include `sonium-compute` (thread pools and BLAS widths)
and `sonium-fft` (batch dispatch budgets). Escape hatches:
`SONIUM_SET_AFFINITY=0`, `SONIUM_TOPO_PACK=0`,
`SONIUM_STREAM_WIDTH=n`, `SONIUM_BW_CALIB=0`.

## Development

```bash
cargo test -p sonium-topology
```

See `CHANGELOG.md` for release history.
