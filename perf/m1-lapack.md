# Apple M1 (macOS, aarch64) — Dense LAPACK

_Generated 2026-10-15T08:18:22+00:00 from commit `39fde79e789e-dirty`._

- status: measured
- host: Darwin 27.0.0 arm64 · 8 logical CPUs · mem 16 GB
- rustc: rustc 1.99.0 (b940084d7 2026-09-28)

## Commands

```text
bash scripts/direct_solver_perf.sh /var/folders/8z/fqzm1txx5p50qp15_8rft79m0000gn/T/perf-cells-0l1alb0d/m1-direct.md
```
```text
cargo bench -p sonium-solvers-direct --bench lapack_compare
```
## Results

### Factorization kernels (`lapack_compare`)

Median of criterion samples; gap = sonium ÷ accelerate.

#### `geqrf_f64`

| size | sonium | accelerate | gap |
|---|---|---|---|
| 512 | 9.697 ms | 6.072 ms | 1.60× |
| 1024 | 52.302 ms | 37.046 ms | 1.41× |
| 2048 | 665.609 ms | 357.880 ms | 1.86× |
| 4096 | 6.344 s | 2.446 s | 2.59× |

#### `getrf_f32`

| size | sonium | accelerate | gap |
|---|---|---|---|
| 1024 | 7.217 ms | 4.224 ms | 1.71× |
| 2048 | 28.757 ms | 44.173 ms | 0.65× |

#### `getrf_f64`

| size | sonium | accelerate | gap |
|---|---|---|---|
| 512 | 2.444 ms | 1.160 ms | 2.11× |
| 1024 | 9.774 ms | 7.036 ms | 1.39× |
| 2048 | 70.679 ms | 51.922 ms | 1.36× |
| 4096 | 764.924 ms | 320.200 ms | 2.39× |

#### `potrf_f32`

| size | sonium | accelerate | gap |
|---|---|---|---|
| 1024 | 4.939 ms | 751.38 µs | 6.57× |
| 2048 | 26.830 ms | 6.753 ms | 3.97× |

#### `potrf_f64`

| size | sonium | accelerate | gap |
|---|---|---|---|
| 512 | 1.794 ms | 355.77 µs | 5.04× |
| 1024 | 6.916 ms | 2.981 ms | 2.32× |
| 2048 | 41.410 ms | 15.436 ms | 2.68× |
| 4096 | 298.598 ms | 141.331 ms | 2.11× |

### Dense LU via `direct-solver-perf`

One factorization + one solve per timing; end-to-end reading lives in `m1-dense-solvers.md`.

| N | Sonium blocked (ms) | Sonium factor (ms) | Sonium solve (ms) | Accelerate (ms) | Sonium / Accelerate | Sonium residual | Accelerate residual | Sonium factor nnz | Sonium factor bytes | Modeled peak bytes | Peak RSS bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.113 | 0.097 | 0.009 | 0.052 | 2.17× | 8.71e-16 | 1.20e-15 | 4096 | 32768 | 67072 | n/a |
| 128 | 0.423 | 0.391 | 0.031 | 0.152 | 2.78× | 1.22e-15 | 1.44e-15 | 16384 | 131072 | 265216 | n/a |
| 256 | 1.788 | 1.318 | 0.110 | 0.863 | 2.07× | 1.66e-15 | 2.65e-15 | 65536 | 524288 | 1054720 | n/a |
| 512 | 3.791 | 2.691 | 0.206 | 2.168 | 1.75× | 3.21e-15 | 3.10e-15 | 262144 | 2097152 | 4206592 | n/a |
