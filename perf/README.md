# Per-platform performance records

One committed markdown file per (platform, workload cell):

| | dense BLAS | dense LAPACK | sparse matrix | dense solvers | sparse solvers + precond |
|---|---|---|---|---|---|
| Apple M1 | `m1-blas.md` | `m1-lapack.md` | `m1-sparse-matrix.md` | `m1-dense-solvers.md` | `m1-sparse-solvers.md` |
| Apple M2 | `m2-blas.md` | `m2-lapack.md` | `m2-sparse-matrix.md` | `m2-dense-solvers.md` | `m2-sparse-solvers.md` |
| Apple M4 | `m4-blas.md` | `m4-lapack.md` | `m4-sparse-matrix.md` | `m4-dense-solvers.md` | `m4-sparse-solvers.md` |
| Linux x86-64 | `x86-blas.md` | `x86-lapack.md` | `x86-sparse-matrix.md` | `x86-dense-solvers.md` | `x86-sparse-solvers.md` |
| NVIDIA CUDA | `cuda-blas.md` | `cuda-lapack.md` | `cuda-sparse-matrix.md` | `cuda-dense-solvers.md` | `cuda-sparse-solvers.md` |

## Run every benchmark supported by this machine

```bash
just bench                 # All native CPU targets, then available CUDA arms
just bench --dry-run       # Inspect hardware detection and exact commands
just bench --backend cpu   # CPU only
```

The runner discovers Cargo workspace benchmark targets. It probes a visible
CUDA device and the cuBLAS/cuSPARSE/cuSOLVER/NVRTC libraries before enabling
CUDA paths. It enables OpenBLAS comparison arms when the library is installed,
and skips platform stubs such as Apple Accelerate on Linux. On the current
x86/CUDA box, the plan contains 44 CPU targets and eight CUDA targets.
Benchmarks run sequentially. A failure is retained in the summary while the
remaining targets continue; the command exits unsuccessfully if any target fails.

Logs and a JSON summary are retained under `target/bench-runs/<run>/`.
Criterion measurements remain under `target/criterion/`, with separate
`cpu` and `cuda` saved baselines. Thread budgets default to at most 16;
`RAYON_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `OMP_NUM_THREADS` override them.

For a bounded selection:

```bash
just bench --package sonium-compute --bench blas1_compare --quick
```

`--package` and `--bench` can be repeated. `--quick` (or `PERF_QUICK=1`) uses
Criterion quick mode; custom benchmark harnesses retain their normal workload.
`--filter` selects Criterion benchmark names. The default command runs full
benchmarks, including expensive FEM/FMM/scaling workloads.

This command runs the benchmark inventory. Use the recipes below to publish
formatted per-platform performance tables for the website.

## Workflow

Each platform owner regenerates and commits their six files:

```bash
just perf-m1     # on the Apple M1 Mac (perf-m2 / perf-m4 likewise, per chip)
just perf-x86    # on the Linux x86-64 box
just perf-cuda   # on the Linux + NVIDIA GPU box
just perf-cell x86 blas   # a single cell
```

Adding a new Apple chip (e.g. M6): add its id to `APPLE_PLATFORMS` in
`scripts/perf_cells.py`, a `just perf-m6` recipe, and a row above.
Auto-detection parses any `Apple M<N>` brand string, so the generator
titles the files correctly even before the id is listed.

`PERF_QUICK=1` selects a bounded run; quick mode is recorded in each file
header and is not comparable to full runs. `PERF_KEEP_WORK=1` retains raw
harness logs under `perf/.work/` (git-ignored).

## Cell → harness mapping

All generation goes through `scripts/perf_cells.py`:

| cell | harness | parsed from |
|---|---|---|
| `blas` | `cargo bench -p sonium-compute` blas1/2/3_compare (+ `bench-openblas` on x86, `cuda` on CUDA) | `target/criterion/**/new/estimates.json` medians, sonium vs vendor arm |
| `lapack` | `lapack_compare` criterion bench (potrf/getrf/geqrf + f32, residual-gated) + direct dense section | criterion medians sonium vs vendor + LU table |
| `sparse-matrix` | `roofline_preflight` bench (+ iterative effective BW) | `csr_spmv` estimates + `sonium_roofline_preflight` byte models |
| `dense-solvers` | `scripts/direct_solver_perf.sh` dense section | end-to-end factor+solve table |
| `sparse-solvers` | direct sparse section + `scripts/iterative_solver_perf.sh` JSON | sparse-LU table + scaling/coverage + PETSc KSP comparison (when `ITERATIVE_PERF_PETSC` is set) |
| `fft` | `cargo bench -p sonium-fft --bench fft_compare` | criterion medians sonium vs vendor (complex/real f32/f64, audio + large sizes) |

Vendor arms: Accelerate (m4), OpenBLAS/SuperLU/CHOLMOD/PETSc/rustfft (x86),
cuBLAS/cuSPARSE/cuSOLVER (cuda, plus host-CPU rustfft for the fft cell).
Gap = Sonium ÷ vendor throughout.

Standalone vendor benches (see `just --list`, group `bench`):
`bench-lapack`, `bench-lapack-cuda`, `bench-sparse-cpu`,
`bench-gmres-cuda`, `bench-petsc-build`, `bench-petsc-run`, `bench-fft`.

## Relation to `PERF.md` / `ITERATIVE_PERF.json`

`perf/PERF.md` (direct solvers) and `perf/ITERATIVE_PERF.json`
(iterative solvers) are the committed solver reports; both live in
`perf/` since 2026-10-12 (moved from the repo root). Regenerate
with `just perf-solvers` (full matrix) or the bounded form below;
each platform owner regenerates and commits on their own box.
`DIRECT_PERF_*` / `ITERATIVE_PERF_*` env vars pass through to the
underlying scripts (`scripts/direct_solver_perf.sh`,
`scripts/iterative_solver_perf.sh`).

```bash
just perf-solvers            # full direct + iterative matrices
DIRECT_PERF_DENSE_SIZES=64 DIRECT_PERF_SPARSE_GRIDS=8 ITERATIVE_PERF_QUICK=1 \
  just perf-solvers          # bounded smoke of the regen plumbing
```

Seed content transcribed from the old root worklog is marked
`status: seed` until a real `just perf-*` run overwrites it.
