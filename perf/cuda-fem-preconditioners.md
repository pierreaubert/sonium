# Retained CUDA high-order FEM preconditioning

Low-order refined AMG now solves all three high-order hex cases that failed
with AMG built from the high-order matrix. The high-order system is unchanged:
K−25M, complex64, natural Neumann conditions, FGMRES restart 60, maximum 600
iterations and relative tolerance 1e-9. This establishes a working retained
route for these cases; iteration counts and hierarchy growth still need work.

## Implementation

[`prepare_hex_lor_shifted_laplacian`](../crates/sonium-fem/src/physics/acoustics/pressure/device_hex.rs)
constructs a Q1 auxiliary mesh by subdividing each Q1/Q2/Q3 hex along its nodal
grid. Auxiliary node indices equal the original H1 DOF indices. AMG built from
K_LOR+(25+25i)M_LOR therefore acts on the high-order vectors directly, without
additional restriction/prolongation arrays or host vector exchange. Both the
CSR and partial-assembly system operators continue to represent the original
high-order discretization and use the same auxiliary preconditioner.

The auxiliary matrix is approximate, including a piecewise-trilinear geometry
approximation on curved meshes. Host setup uses two Gauss points per axis on
each subcell. Runtime cases below use affine geometry; curved geometry and
frequency-varying updates need further coverage. Prescribed-pressure numbering
is preserved; CPU tests compare constrained actions against an independently
generated refined mesh for Q1/Q2/Q3.

A separate backend correctness fix makes L1-Jacobi use inverse row norms with
unit relaxation, matching the host implementation. Previously it uploaded the
ordinary inverse diagonal and applied Jacobi's damping. Regression tests check
independent row-norm relaxation at every hierarchy level and after checkpoint
restoration on CPU and CUDA. Corrected L1 alone does not fix the high-order
convergence problem and is not the selected performance route.

## Measurements

Same RTX PRO 6000 Blackwell and release benchmark methodology as the
[initial kernel report](cuda-fem-kernels.md). Four Rayon threads, five timed
zero-initial-guess solves after warmup, same preconditioner for both operators.
The table includes the previously passing Q2/729 control, which becomes slower
with this auxiliary matrix. Do not select low-order refinement universally.

| Space | DOFs | Iterations, both routes | CSR solve ms | Partial assembly solve ms | Independent relative residual |
| --- | --- | --- | --- | --- | --- |
| Q2 | 729 | 135 | 185.95 | 190.15 | 8.60e-10 |
| Q3 | 2,197 | 212 | 479.47 | 483.71 | 9.81e-10 |
| Q2 | 4,913 | 123 | 258.36 | 267.54 | 8.04e-10 |
| Q3 | 15,625 | 329 | 1035.07 | 1030.01 | 9.69e-10 |

The three larger rows previously exhausted 600 iterations. Each now passes
independent residual verification against the original high-order matrix.
Warm solves allocate no additional device storage. A separately warmed
preconditioner application asserts zero allocation, H2D and D2H counts and zero
recorded smoother fallback. Small Krylov control transfers remain recorded in
the solve data. All eight benchmark cases (including unchanged Tet controls)
at subdivisions 4 and 8 completed successfully.

The Q3/15,625 hierarchy has level sizes 15,625 → 11,100 → 6,874 → 3,789 →
1,892 → 608 → 205 → 91 → 54. This weak initial coarsening and 329 outer
iterations are concrete next optimisation targets. Convergence is not proof of
mesh-independent scaling or improved speed versus a tuned alternative solver.

Multicolor Gauss-Seidel was also measured. Its completed CSR solves took
132.82 ms for Tet P1/125, 737.38 ms for Tet P2/729 and 974.77 ms for hex Q2/729,
despite fewer iterations. The run was stopped after 308 seconds while the
Q3/2,197 case was unfinished. Its partial evidence is retained; the unfinished
case has no convergence or timing claim.

## Reproduce

```bash
RAYON_NUM_THREADS=4 SONIUM_FEM_BENCH_PRECONDITIONER=lor \
  cargo run --release -p sonium-fem --features cuda \
  --example cuda_fem_compare -- 4 8
```

`SONIUM_FEM_BENCH_PRECONDITIONER=assembled` preserves the original benchmark.
`SONIUM_FEM_BENCH_SMOOTHER` accepts `jacobi` (default), `l1-jacobi` and
`multicolor-gs`. These are explicit selections; there is no silent solver or
preconditioner substitution. The example fails if any solve fails convergence,
an independent residual fails, or a warmed preconditioner allocates/stages data.

[Raw results and metadata](cuda-fem-preconditioners-2026-10-21.json) retain
passing, failing and interrupted runs. The auxiliary producer and repeated-RHS
native regression are in `device_hex.rs` and `device_hex/native_lor.rs`.
Broader production integration, GPU numeric facet/preconditioner updates,
curved/heterogeneous sweeps and time-to-tolerance optimisation remain open.

Validation: 1,920 FEM CPU tests passed (one ignored), and 334 preconditioner
unit/integration tests passed (one doctest ignored). The native per-level L1
and checkpoint tests passed, as did the native Q3 auxiliary regression with
changing RHS, retained warm start, no second-solve allocation and independent
original-system residual checks. Root check, root Clippy and CUDA test/example
Clippy passed with existing warnings. Logs: `/tmp/amg-l1-native.log`,
`/tmp/amg-l1-suite.log`, `/tmp/hex-lor-fem.log`, `/tmp/hex-lor-native.log`,
`/tmp/hex-lor-final-check.log`, `/tmp/hex-lor-final-clippy.log`, and
`/tmp/hex-lor-cuda-clippy.log`.

## Reproducible hierarchy comparisons (2026-10-22)

The comparison example also accepts `SONIUM_FEM_BENCH_FAMILY=hex` (or `tet`,
default `all`) and `SONIUM_FEM_BENCH_AMG`:

- `baseline`: existing CLJP/standard interpolation, one pre/post Jacobi sweep.
- `extended`: Extended+i interpolation, otherwise baseline.
- `two-sweeps`: two pre/post sweeps, otherwise baseline.
- `w-cycle`: W-cycle, otherwise baseline.
- `aggregation`: existing smoothed-aggregation profile, with the selected
  smoother, coarse size 16 and Jacobi weight 0.6.

The JSON output records the full effective AMG configuration, hierarchy sizes,
operator complexity, kernel launches and vendor calls. Solve operation counts
cover all five measured solves; divide by five for per-solve counts. Counts are
work indicators, not elapsed-time estimates. All configurations retain the same
physical matrix, RHS, initial guess, FGMRES controls and residual checks.

```bash
RAYON_NUM_THREADS=4 SONIUM_FEM_BENCH_PRECONDITIONER=lor \
  SONIUM_FEM_BENCH_FAMILY=hex SONIUM_FEM_BENCH_AMG=extended \
  cargo run -p sonium-fem --features cuda --example cuda_fem_compare -- 2 4
```

These options do not change production defaults. The new comparison is performed
while a separate BEM sanitizer is active, so its timings cannot establish a
speedup. Iterations, independent residuals, allocations and operation counts
remain useful evidence for selecting the next isolated performance experiment.

### Iterations and work counts

| Profile | Q2 / 125 DOFs | Q3 / 343 DOFs | Q2 / 729 DOFs | Q3 / 2,197 DOFs |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 49 | 217 | 135 | 212 |
| Extended+i | 50 | 157 | 102 | 199 |
| Two sweeps | 48 | 168 | 114 | 204 |
| W-cycle | 49 | 225 | 128 | 204 |
| Aggregation | Coarse LU failed | Not run | Not run | Not run |

Both CSR and partial-assembly routes produced the same iteration counts for
all successful cases. All 160 measured solves had independent relative residuals
below 2e-9 and zero warm device allocations. Standalone preconditioner actions
had no host vector transfers or device allocations.

Extended+i reduces the Q3/343 solve from 2,832 to 2,049 kernel launches and
4,597 to 3,327 vendor operations, but increases hierarchy operator complexity
from 1.55 to 1.87. Q2/729 launches drop from 2,168 to 1,331; Q3/2,197 improves
only modestly. Larger/coarser operators can offset launch savings, so isolated
scaling and timing validation is still required. The smallest case regresses.

The W-cycle Q3/2,197 case issues 78,551 kernels and 169,564 vendor calls per
solve, versus 5,311 and 8,732 for baseline, for only eight fewer iterations.
Two sweeps also increase kernel count on every case. Neither is selected as the
next performance route. Aggregation fails the first coarse factorization at
one-based pivot 5; there is no fallback or regularized result presented as a pass.
Its coarse hierarchy needs diagnosis before this profile can be admitted.

[Full profiles, failures and raw diagnostic timings](cuda-fem-amg-variants-2026-10-22.json).
Root check/clippy, CUDA example clippy and 1,946 CPU FEM tests pass (one ignored;
existing warnings). No production defaults changed.

## Aggregation truncation repair (2026-10-22)

The failed aggregation profile exposed a setup defect. Magnitude-only truncation
kept four entries per row and erased all entries of four coarse columns in the
125-to-27 transfer. The Galerkin matrix consequently contained unsupported coarse
unknowns. Truncation now retains the row's nonzero own-aggregate entry and uses
the configured `max_interp_elements` instead of a hard-coded four. Existing
normalization remains. SA and optional GNN setup share the corrected helper.
This addresses the observed loss of support; it is not a general rank guarantee.

An all-scalar regression checks row bounds, own-aggregate support and constant
preservation for limits 1, 2, 4 and 6. The failing physical auxiliary hierarchy
now has no empty prolongation columns. The CUDA profile now completes all four
workloads through both CSR and partial-assembly operators without regularization
or solver fallback.

| Space / DOFs | Classical baseline iterations | Repaired aggregation iterations | Baseline / aggregation operator complexity |
| --- | ---: | ---: | ---: |
| Q2 / 125 | 49 | 48 | 1.05 / 1.26 |
| Q3 / 343 | 217 | 118 | 1.55 / 1.12 |
| Q2 / 729 | 135 | 82 | 1.53 / 1.16 |
| Q3 / 2,197 | 212 | 176 | 4.01 / 1.10 |

All 40 measured solves pass independent residual checks below 2e-9 with zero
warm allocation. At Q3/2,197, vendor calls per CSR solve drop from 8,732 to 6,014
and kernel launches from 5,311 to 4,232. The smallest case increases work despite
one fewer iteration. These are workload-specific improvements in convergence
and stored hierarchy entries, not an elapsed-time speedup. The baseline uses
one smoothing sweep and the aggregation profile uses two; compare full profiles,
not a single isolated coarsening parameter.

The Q2/Q3 subdivision-2 run passes memcheck with zero errors. CPU preconditioner
(241) and FEM (1,947; one ignored) suites, root check/clippy, CUDA test/example
clippy and the optional GNN build pass. Production coarsening defaults remain
unchanged. [Repair evidence and full results](cuda-amg-aggregation-support-2026-10-22.json).

## Release timing and retained production sweeps (2026-10-22)

After the long BEM sanitizer completed, release runs compared baseline and
repaired aggregation in baseline–aggregation–aggregation–baseline order.
`nvidia-smi` reported no compute process before each run; our builds, tests and
sanitizers had finished. Each table value is the median of ten warm zero-start
solves across two processes, at the same original-system tolerance of 1e-9.

| Space / DOFs | CSR baseline ms | CSR aggregation ms | Speedup | Partial assembly baseline / aggregation ms |
| --- | ---: | ---: | ---: | ---: |
| Q2 / 729 | 131.75 | 84.87 | 1.55× | 136.32 / 87.51 |
| Q3 / 2,197 | 322.88 | 244.75 | 1.32× | 329.25 / 250.03 |
| Q2 / 4,913 | 180.79 | 133.22 | 1.36× | 190.02 / 139.53 |
| Q3 / 15,625 | 720.65 | 314.03 | 2.29× | 716.25 / 312.16 |

Q3/15,625 iterations drop from 329 to 152, with initial coarsening from
15,625 → 823 instead of 15,625 → 11,100. All 32 operator cases converge;
160 measured solves retain zero warm allocations, and each case's final solution
passes independent host residual verification. Setup is separately recorded:
for the first Q3/15,625 pass, auxiliary assembly/hierarchy/upload/factorization
costs 715 ms for baseline and 404 ms for aggregation. These setup observations
are not the warm-solve samples and are not a claim about GPU-only setup.

This supports an explicit aggregation configuration for the tested medium/large
Q2/Q3 LOR workloads, while the smaller-case regression remains a reason to avoid
changing every default:

```rust
AmgConfig {
    coarse_size: 16,
    smoother: AmgSmoother::Jacobi,
    jacobi_weight: 0.6,
    ..AmgConfig::for_aggregation()
}
```

The retained warped Q1/Q2/Q3 session fixture now covers both profiles through
material, Robin/source, frequency and auxiliary-hierarchy changes. All 24 solves
pass independent assembled-system checks, structural refresh rejection and
failure recovery. Q3's hardest boundary/material case drops from 224 to 166
iterations. The test exposed a missing restart buffer in explicit Krylov
reservation; reservation now includes that buffer, preserving zero warm
allocations when a later solve first needs a restart. Full expanded native
coverage passes memcheck with zero errors. CPU iterative (321) and FEM (1,947;
one ignored) suites and root/CUDA checks pass.

[Release samples, hardware checks and binary/source hashes](cuda-fem-amg-release-2026-10-22.json).
[Retained production sweep and restart reservation evidence](cuda-hex-aggregation-production-2026-10-22.json).
The timing workload remains affine, natural-Neumann Helmholtz at k=5 on one GPU;
production defaults and broader coverage are unchanged. Partial assembly does
not consistently beat cuSPARSE in these cases, so this result does not justify
replacing the vendor sparse path globally.
