# Apple M1 (macOS, aarch64) — Dense solvers

_Generated 2026-10-15T08:20:20+00:00 from commit `39fde79e789e-dirty`._

- status: measured
- host: Darwin 27.0.0 arm64 · 8 logical CPUs · mem 16 GB
- rustc: rustc 1.99.0 (b940084d7 2026-09-28)

## Commands

```text
bash scripts/direct_solver_perf.sh /var/folders/8z/fqzm1txx5p50qp15_8rft79m0000gn/T/perf-cells-0l1alb0d/m1-direct.md
```
## Results

End-to-end dense LU: one factorization + one right-hand-side solve per timing.

| N | Sonium blocked (ms) | Sonium factor (ms) | Sonium solve (ms) | Accelerate (ms) | Sonium / Accelerate | Sonium residual | Accelerate residual | Sonium factor nnz | Sonium factor bytes | Modeled peak bytes | Peak RSS bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.113 | 0.097 | 0.009 | 0.052 | 2.17× | 8.71e-16 | 1.20e-15 | 4096 | 32768 | 67072 | n/a |
| 128 | 0.423 | 0.391 | 0.031 | 0.152 | 2.78× | 1.22e-15 | 1.44e-15 | 16384 | 131072 | 265216 | n/a |
| 256 | 1.788 | 1.318 | 0.110 | 0.863 | 2.07× | 1.66e-15 | 2.65e-15 | 65536 | 524288 | 1054720 | n/a |
| 512 | 3.791 | 2.691 | 0.206 | 2.168 | 1.75× | 3.21e-15 | 3.10e-15 | 262144 | 2097152 | 4206592 | n/a |
