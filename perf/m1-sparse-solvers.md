# Apple M1 (macOS, aarch64) — Sparse solvers + preconditioners

_Generated 2026-10-15T08:20:20+00:00 from commit `39fde79e789e-dirty`._

- status: measured
- host: Darwin 27.0.0 arm64 · 8 logical CPUs · mem 16 GB
- rustc: rustc 1.99.0 (b940084d7 2026-09-28)

## Commands

```text
bash scripts/direct_solver_perf.sh /var/folders/8z/fqzm1txx5p50qp15_8rft79m0000gn/T/perf-cells-0l1alb0d/m1-direct.md
```
```text
bash scripts/iterative_solver_perf.sh /var/folders/8z/fqzm1txx5p50qp15_8rft79m0000gn/T/perf-cells-0l1alb0d/m1-iterative.json
```
## Results

### Sparse direct (LU)

| Grid | N | NNZ | Simplicial (ms) | Simplicial factor (ms) | Simplicial solve (ms) | Supernodal (ms) | Supernodal factor (ms) | Supernodal solve (ms) | Reused symbolic (ms) | Reused factor (ms) | Reused solve (ms) | SuperLU (ms) | Simplicial / SuperLU | Supernodal / SuperLU | Simplicial factor nnz | Supernodal factor nnz | Reused factor nnz | Simplicial factor bytes | Supernodal factor bytes | Reused factor bytes | Modeled peak bytes | Peak RSS bytes | Max residual |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8×8 | 64 | 288 | 0.063 | 0.057 | 0.002 | 0.092 | 0.078 | 0.001 | 0.048 | 0.042 | 0.001 | — | — | — | 844 | 844 | 844 | 6752 | 6752 | 6752 | 13416 | n/a | 1.05e-15 |
| 12×12 | 144 | 672 | 0.189 | 0.175 | 0.004 | 0.267 | 0.248 | 0.004 | 0.170 | 0.157 | 0.004 | — | — | — | 2598 | 2598 | 2631 | 20784 | 20784 | 21048 | 36416 | n/a | 2.35e-15 |
| 16×16 | 256 | 1216 | 0.392 | 0.350 | 0.008 | 0.524 | 0.497 | 0.008 | 0.359 | 0.337 | 0.008 | — | — | — | 6060 | 6060 | 6103 | 48480 | 48480 | 48824 | 76480 | n/a | 2.67e-15 |
| 20×20 | 400 | 1920 | 0.654 | 0.612 | 0.014 | 0.932 | 0.899 | 0.013 | 0.679 | 0.631 | 0.014 | — | — | — | 10864 | 10864 | 10887 | 86912 | 86912 | 87096 | 130624 | n/a | 1.04e-14 |
| 28×28 | 784 | 3808 | 1.524 | 1.473 | 0.031 | 1.975 | 1.942 | 0.031 | 1.499 | 1.462 | 0.032 | — | — | — | 26090 | 26090 | 26177 | 208720 | 208720 | 209416 | 295440 | n/a | 4.68e-15 |

### Iterative + preconditioners

62 runs · pairs: bicgstab+none, cg+amg, cg+jacobi, cg+none, fgmres+amg, gmres+amg, gmres+ilu, gmres+ilut, gmres+jacobi, gmres+none, idrs+jacobi, idrs+none

matrices: anisotropic_diffusion_2d, helmholtz_2d, laplacian_3d, poisson_2d, saddle_point_2d

#### CG + AMG on Poisson 2-D (f64)

| n | solve | iters | residual | eff. SpMV BW |
|---|---|---|---|---|
| 961 | 0.41 ms | 12 | 8.5e-07 | 2.86 GB/s |
| 10000 | 4.62 ms | 14 | 5.2e-07 | 3.13 GB/s |
| 99856 | 72.98 ms | 18 | 6.7e-07 | 2.56 GB/s |

