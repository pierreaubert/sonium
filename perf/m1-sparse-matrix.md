# Apple M1 (macOS, aarch64) — Sparse matrix

_Generated 2026-10-15T08:20:20+00:00 from commit `39fde79e789e-dirty`._

- status: measured
- host: Darwin 27.0.0 arm64 · 8 logical CPUs · mem 16 GB
- rustc: rustc 1.99.0 (b940084d7 2026-09-28)

## Commands

```text
cargo bench -p sonium-compute --bench roofline_preflight
```
```text
bash scripts/iterative_solver_perf.sh /var/folders/8z/fqzm1txx5p50qp15_8rft79m0000gn/T/perf-cells-0l1alb0d/m1-iterative.json
```
## Results

### CSR SpMV (roofline preflight)

| benchmark | median | bytes | eff. BW |
|---|---|---|---|
| `roofline_preflight_spmv/cpu/csr_f32_prebaked` | 75.42 µs | 589812 | 7.8 GB/s |

### Effective SpMV bandwidth inside iterative solves

| n | solve | iters | residual | eff. SpMV BW |
|---|---|---|---|---|
| 961 | 0.41 ms | 12 | 8.5e-07 | 2.86 GB/s |
| 10000 | 4.62 ms | 14 | 5.2e-07 | 3.13 GB/s |
| 99856 | 72.98 ms | 18 | 6.7e-07 | 2.56 GB/s |

