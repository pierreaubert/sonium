# CUDA FEM kernel and persistent solve measurements

On the RTX PRO 6000 Blackwell, the closed-form P2 stiffness expansion improves
the existing node-owned P2 implementation, but cuSPARSE remains faster on all
measured meshes. The current shifted-AMG configuration also fails on three
larger high-order hex cases. Keep the CSR route available; neither kernel
availability nor small integration tests justify switching production dispatch.

A [follow-up auxiliary-preconditioner implementation](cuda-fem-preconditioners.md)
now converges on the three failed hex cases below without changing the original
system or tolerance. The table below remains the original assembled-AMG baseline.

## Method

Run the [benchmark example](../crates/sonium-fem/examples/cuda_fem_compare.rs):

```bash
RAYON_NUM_THREADS=4 cargo run --release -p sonium-fem --features cuda \
  --example cuda_fem_compare -- 2 4 8
```

Requires actual CUDA access. This session requires execution outside its device
sandbox. Hardware: RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB,
600 W power limit, driver 610.57.04; Rust 1.99.1. Release build; four Rayon
threads for host setup. Geometry is an affine unit cube; Tet P1/P2 and hex Q2/Q3,
complex64, k=5, natural homogeneous Neumann conditions. There are no impedance
or prescribed-pressure facets in this benchmark. It is a GPU-to-GPU comparison,
not a CPU speedup study or a geometry/physics coverage test.

Both operators solve the same independently assembled K−25M system, using the
same retained AMG preconditioner built from K+(25+25i)M, Jacobi smoothing
(weight 0.6), coarse size 16 and native complex coarse LU. FGMRES uses restart
60, maximum 600 iterations, tolerance 1e-9. Every timed solve starts from zero;
workspace is warmed and retained. Five solve samples are timed with completion
synchronization. Each action sample batches 100 launches and synchronizes once;
seven samples report amortized per-action wall time. No CUDA graphs are used.
CSR is measured before the specialized operator; measurements are exploratory,
not randomized controlled trials. Raw samples and setup costs are retained.

Action parity is checked against CPU quadrature (relative error <1e-11).
Converged solutions are checked against the independent assembled residual
(<2e-9). Warm actions assert zero allocations and host/device transfers.
Warm solves report allocations and control-transfer bytes; they do not pretend
scalar/Hessenberg traffic is zero. Failed solves receive no timing claim and
cause a nonzero final exit status after all cases have been recorded.

## Optional solve phase attribution

Set `SONIUM_FEM_BENCH_PROFILE=sync` to run one additional diagnostic solve after
the normal five timing samples for each operator. Its `synchronized_profile`
object separates operator application, original-system residual evaluation,
and preconditioner application. It synchronizes before and after each phase;
the reported times include dispatch and completion overhead and perturb normal
execution. They are not CUDA-event kernel times. The remaining
`other_solve_work_ms` includes Krylov algebra, reductions, control transfers,
validation, host work and unattributed synchronization; it is not a measurement
of orthogonalization alone.

The diagnostic preserves the actual operator/residual/preconditioner methods,
requires unchanged iteration counts, verifies the original-system residual
independently and rejects new warm device allocations. Normal timing samples
remain separate. The default is `off`; unknown values fail explicitly.

```bash
SONIUM_FEM_BENCH_PROFILE=sync SONIUM_FEM_BENCH_PRECONDITIONER=lor \
  RAYON_NUM_THREADS=4 cargo run --release -p sonium-fem --features cuda \
  --example cuda_fem_compare -- 4 8
```

The diagnostic now also enables `BackendAmgPreconditioner::set_phase_profiling`
for that solve only. Its `amg_levels` array reports pre-smoothing, residual
formation, restriction, prolongation plus correction, post-smoothing and coarse
solution separately for every level. Recursive visits are excluded from the
parent's phase times; W/F revisits increase the visited level's counts. Coarse
fallback smoothing belongs to the coarse-solve phase. Only successful phase
completions are recorded; a partial profile does not certify a successful solve.
The API is disabled by default and does not add synchronization when disabled.

Use an otherwise idle GPU for bottleneck decisions. Synchronized stage times
include dispatch and completion overhead and can expose the cost of many small
level operations. Host-only AMG application timers must not be interpreted as
completed GPU execution time. Numeric hierarchy refresh retains the profiling
state; a rebuilt or checkpoint-restored hierarchy starts with profiling disabled.

## Final measurements

All times below are median milliseconds. Hex uses Q2/Q3 tensor-product spaces
(the table labels follow the code's PolynomialDegree enum).

| Element/order | Subdivisions | DOFs | cuSPARSE action | Specialized action | cuSPARSE solve | Specialized solve |
| --- | --- | --- | --- | --- | --- | --- |
| tet P1 | 2 | 27 | 0.0248 | 0.1594 | 9.81 | 12.74 |
| tet P2 | 2 | 125 | 0.0267 | 0.4682 | 25.09 | 40.97 |
| hex P2 | 2 | 125 | 0.0267 | 0.0539 | 25.18 | 26.34 |
| hex P3 | 2 | 343 | 0.0261 | 0.0590 | 57.26 | 58.99 |
| tet P1 | 4 | 125 | 0.0264 | 0.1652 | 40.96 | 46.24 |
| tet P2 | 4 | 729 | 0.0274 | 0.9209 | 77.40 | 129.35 |
| hex P2 | 4 | 729 | 0.0275 | 0.0567 | 66.84 | 68.59 |
| hex P3 | 4 | 2,197 | 0.0389 | 0.0650 | not converged | not converged |
| tet P1 | 8 | 729 | 0.0271 | 0.6498 | 60.49 | 86.07 |
| tet P2 | 8 | 4,913 | 0.0295 | 1.7442 | 468.68 | 880.26 |
| hex P2 | 8 | 4,913 | 0.0399 | 0.1098 | not converged | not converged |
| hex P3 | 8 | 15,625 | 0.1290 | 0.1260 | not converged | not converged |

The three nonconvergent cases fail with both operators, at essentially the same
residual. This points to solver/preconditioner behavior rather than an operator
parity defect. All converged cases have matching iteration counts between the
two routes and zero warm allocations. The current Jacobi-AMG configuration is
not established as a suitable scalable high-order Helmholtz preconditioner.

## P2 change and next decisions

The original P2 action accumulated a complex gradient at each of four
barycentric vertices, then integrated that representation. The new kernel
expands E[lambda_a lambda_b]=(1+delta_ab)/20 into scalar stiffness entries,
removing twelve complex gradient accumulators and their repeated updates.
Numeric CSR assembly shares that expression. No fast-math or reduced-precision
change is involved.

At 125 DOFs, P2 action time fell from 1.5638 to 0.4682 ms (3.34×), and complete
solve time from 80.38 to 40.98 ms (1.96×), with 34 iterations. At 729 DOFs,
action time fell from 2.5542 to 0.9209 ms (2.77×), and solve time from 224.38
to 129.35 ms (1.73×), with 56 iterations. These compare the same coercive shift,
mesh, solver settings and GPU. They are improvements over the prior custom
kernel, not over cuSPARSE. Broader repetition is needed for production claims.

Next work should address high-order preconditioning (p-multigrid or a suitable
low-order auxiliary operator, with stronger smoothing) and measure restart
sensitivity on the failing cases. For Tet action, compare element-owned work
and reduction against the current repeated node-owned arithmetic. Profile launch
and memory overhead before choosing a layout; Nsight Compute was attempted but
hardware counters are denied (`ERR_NVGPUCTRPERM`). No occupancy/bandwidth
measurement is inferred from wall-clock timing. GPU facet updates and numeric
preconditioner reuse remain separate unfinished work.

## Evidence

[Raw samples and metadata](cuda-fem-kernels-2026-10-21.json) include the initial
indefinite-shift failure, partial pre-change coercive-shift run, and complete
post-change run. The initial K+(-25+25i)M configuration stalled already at Q2
hex/729 DOFs; changing the shift did not solve the broader high-order problem.

The changed Tet kernel passed native P1/P2 action, transpose/Hermitian, block,
constraint, fused-mode and numeric-assembly tests. CUDA memcheck reported zero
errors, and the native persistent P2 solve passed. Root check/Clippy and example
Clippy passed; CPU compute had 646 passing tests, FEM 1,919 passing and one
ignored. Current changes do not complete the overall CUDA coverage program.
