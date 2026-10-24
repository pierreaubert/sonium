# CUDA execution audit and FEM performance plan

## Latest audit and delivery contract

`CudaIlutPreconditioner` now supplies retained native ILUT/ILUTP application
without weakening the CPU dropping, fill, pivot or symmetric-ordering policy.
It shares the existing native dependency-level triangular machinery with ILU(0),
and uses resident CSR maps for both permutations. Compatible numeric factors
refresh in place; changed maps or factor sparsity require replacement. Native
four-scalar action tests and actual resolved-speaker PML matrices pass, including
the default 191-node/sigma-18 model at its configured tolerance. All four
sanitizer runs are clean. See [ILUT qualification](cuda-ilut-session-2026-10-22.json).
This closes a preconditioner availability gap for the resolved-speaker default;
the public resolved continuation/full-result APIs still need explicit CUDA
routing and preserved diagnostics/reuse policy. PML qualification currently
uses their host-assembled matrices through the existing persistent FGMRES API.

Preassembled FEM producers now have direct retained CUDA CSR preparation through
`CudaFemSolveSession::try_from_csr` and
`CudaConfiguredFemSolveSession::try_from_csr`. The room treatment producer uses
this path for forward/Hermitian setup, eliminating its CSR-to-COO-to-CSR adapter.
Forward preparation borrows canonical host values; transpose/Hermitian perform
one host CSR transpose. Validation rejects malformed, unsorted/duplicate and
non-finite data before device allocation or preconditioner setup. Configuration,
admission, native iteration and resident updates share the existing session
implementation. The legacy CPU convenience function remains CPU-only and now
documents this explicit retained CUDA entry point; this is not automatic backend
routing for every legacy caller. See [CSR preparation evidence](cuda-fem-csr-preparation-2026-10-22.json).

The existing body-fitted treatment handoff now has explicit CUDA routes for
target reanalysis, promotion, and source optimization followed by target
promotion. They accept retained source/target sessions and share CPU projection,
dimension, source-convergence and target non-regression checks. Native tests use
distinct source/target discretizations and exercise rejection/recovery; the new
handoff passes memcheck with zero errors. See
[projection evidence](cuda-treatment-projection-2026-10-22.json).

Audit correction: inspection found the treatment optimizer's downstream caller
in this library projection workflow, not in an existing treatment CLI. Adding a
new treatment CLI would be new product scope, not closure of a currently exposed
CPU dispatch path. The earlier generic CLI follow-up statements do not establish
such a missing route. Broader existing compute/solver/BEM/FEM paths remain open.

Configured FEM sessions and both treatment producers now support explicit
preconditioner refresh while retaining resident physical operators, bases,
Krylov workspaces and warm-start fields. Candidate preparation validates the
canonical CSR and completes GPU setup before installation. The room producer
stages both actions; the band producer stages all frequency pairs before any
commit. Failed preparation leaves installed actions unchanged. Native tests
and memcheck pass; see [refresh evidence](cuda-treatment-preconditioner-refresh-2026-10-22.json).
Refresh still performs host setup and allocates replacement actions; staging
raises peak memory. Automatic refresh heuristics and in-place factor refresh
are separate remaining performance work.

`CudaMultiFrequencyRoomTreatmentSession` now connects the band objective and
shared MMA loop to retained forward/Hermitian state per frequency. Production
and seat-variance objectives share CPU pressure-level terms and weighted/cost
aggregation; density filtering and constraints use the same optimizer code.
Three-frequency native comparisons, finite differences, cancellation/resume,
failure recovery and memcheck pass. Frequencies execute serially in sorted
order; memory grows with retained frequency state. See
[band evidence](cuda-room-treatment-band-2026-10-22.json). Automatic refresh
heuristics and broader CUDA workflow coverage remain open.

The frequency-local room-treatment application now has an explicit
`CudaRoomTreatmentSession`: its existing MMA loop calls retained configured
forward and Hermitian solves. ILU and configured Schwarz pass physical P1
boundary-treatment objective/gradient, finite-difference, warm-allocation,
recovery and four-iteration MMA comparisons against CPU direct solves; CUDA
memcheck reports zero errors. Base and treatment operators now upload once;
forward/Hermitian numeric design combinations run on CUDA. Both solved fields,
the forward RHS and the assembled adjoint RHS now stay resident. Sparse listener
maps and treatment sensitivities execute on CUDA; only listener-sized objective
vectors, treatment scalars and compact solver controls cross the host boundary.
MMA and the listener-sized nonlinear objective remain host work. See
[resident-field evidence](cuda-room-treatment-resident-fields-2026-10-22.json),
[resident-update evidence](cuda-room-treatment-resident-updates-2026-10-22.json)
and the [initial integration evidence](cuda-room-treatment-session-2026-10-22.json).
The band integration above extends this producer. Broader existing workflow
coverage remains open; these bounded paths do not establish complete CUDA coverage.

`CudaConfiguredFemSolveSession` now accepts the public FEM `SolverConfig` and
prepares retained standard GMRES with identity, Jacobi, ILU, explicit Schwarz
or AMG. It preserves restart/tolerance/exact-residual controls and Schwarz
partition/overlap settings. Unsupported variants reject before assembly.
Native configuration/reuse tests and CUDA memcheck pass; see
[configured-session evidence](cuda-fem-configured-session-2026-10-22.json).
This supplies an explicit retained producer entry point; legacy CPU callers
still require migration, and broad CUDA workflow coverage remains incomplete.

The retained generic FEM session now supports **standard GMRES**, in addition to
FGMRES and PCG. Its native physical Helmholtz fixture connects retained ILU and
explicitly configured Schwarz (eight subdomains, overlap two) through P1/P2
assembly, forward/transpose/Hermitian actions, numeric refresh and warm starts.
This closes the method-selection gap in the generic retained API; it does not
yet migrate the legacy `solve_csr_with_guess` producer or establish support for
pipelined GMRES. See the implementation evidence at the end of this document.

### Retry: public FEM integration boundary

Current source inspection confirms an actionable gap between available CUDA
solver primitives and the public FEM producer:

- `solver/mod.rs::solve_csr_with_guess` still calls CPU `solve_with_guess` /
  `solve_complex`; it has no CUDA backend selection. Its `GmresSchwarz` branch
  explicitly rejects the request, although retained Schwarz now exists in the
  solver layer. Route this producer through an explicit backend policy and a
  reusable session; preserve RHS-only reuse and distinguish numeric refresh
  from a structural rebuild.
- Preserve FEM's configured Schwarz partition count and overlap. Do not silently
  substitute the facade's automatic partitioning and overlap policy. Validate
  the preconditioner action against the CPU configuration on an overlapping mesh.
- `solver/device.rs::FemDeviceMethod` now exposes standard GMRES alongside
  FGMRES and PCG. Standard GMRES requires a fixed linear preconditioner within
  each solve; it uses the existing non-flexible device Krylov implementation.
  The legacy mapping still aliases pipelined and alternative ILU
  variants; track those algorithms individually rather than counting an alias
  as implementation coverage.

This is the next concrete integration slice after runtime admission and registry
maintenance. Its acceptance gate is a physical FEM assembly through the public
API, CPU/native solution parity, original-system residual certification,
configuration preservation, repeated RHS and numeric updates without unexpected
bulk staging, and failure recovery. Host assembly is an explicit first-stage
boundary; moving recurring assembly to CUDA is a subsequent milestone.

This retry changes the audit only. Previously recorded native measurements below
remain historical evidence for their exact workloads, not new GPU validation.
Root `cargo check` followed by `cargo clippy` passed on this retry (logs:
`/tmp/cuda-audit-current-check.log`, `/tmp/cuda-audit-current-clippy.log`).
The source-inventory freshness check, workflow-registry validator and all ten
inventory/registry tests also passed. These checks do not establish native GPU
correctness or full workflow coverage.

The first [reviewed workflow subset](cuda-workflow-registry.json) now covers
all ten solver facade strategies across four scalars for single forward solves.
All forty rows have retained CUDA dispatch; no rows are complete. Auto resolves
once to the CPU-selected solver family and retains it across numeric updates.
CPU optional pipelined-GMRES remains a separate CUDA gap. Both named Schwarz
strategies preserve CPU overlap
and partition-of-unity weights through resident sparse maps and packed local
ILU. GMRES–ILU retains checked host ILU(0) factors with native
dependency-level triangular application and compatible numeric refresh.
GMRES–Chebyshev retains device state under the CPU facade's
real-SPD coefficient contract, including complex scalar storage and RHS. Seven
current native session/update tests plus the large-row-cutoff Auto fixture pass
normally and under memcheck (eight tests); the
separate four-scalar ILU factor parity/recovery fixture also passes memcheck. Run
`python3 scripts/check_cuda_workflow_registry.py` to validate source-review
freshness, exhaustive strategy/scalar rows and attached evidence. This subset
does not establish project-wide coverage.

Source discovery now has an executable inventory: run
`python3 scripts/cuda_workflow_inventory.py --check` to detect source drift,
or `--write` to regenerate after reviewing changes. The checked-in
[source inventory](cuda-source-inventory.json) contains 22 packages and 1361
Rust source review units, including all split solver crates and simulation
applications. These are **not workflow counts or CUDA coverage percentages**.
All units remain unclassified until public exports, producers and variant
requirements have been reviewed. Source hashes and lexical declaration hints
make that review reproducible; macros, re-exports and trait implementations
still require explicit resolution.

Latest implementation evidence (2026-10-22): specialized Tet P1/P2 and Hex
pressure sessions now expose the opt-in fused CSR Jacobi smoother and preserve
its selection across preconditioner refresh/rebuild. Native tests and memory
checks pass. A bounded tetrahedral ABBA comparison shows P1 improvements but
P2 regressions, so the option remains disabled by default. See
[specialized session evidence](cuda-specialized-fused-jacobi-2026-10-22.json).
Full CUDA workflow coverage remains incomplete.

Retry audit: current source confirms bounded retained solver strategies and the
fixed-state contract in common BEM dispatch. The former Auto admission gap is
closed by the CUDA-specific 1000-panel crossover described below. TokenSave is
rebuilding, so current source takes precedence over graph results. This refresh
is an audit and implementation plan; it does not introduce new CUDA kernels or
rerun native GPU tests or benchmarks.

The immediate deliverable should be a public-workflow coverage registry, followed
by persistent execution with appropriate preconditioners. A callable GPU primitive
does not establish coverage of its application producer. Track setup, solve,
numeric updates and outputs separately, and distinguish explicit CPU setup from
unexpected bulk transfers inside a resident iteration. Existing FEM capability
rows in `src/backend.rs` provide a starting point, not a complete cross-crate
workflow inventory.

The earlier statement that small-CSR Jacobi had only primitive validation is
superseded by the recorded [cooperative AMG experiment](cuda-small-jacobi-cooperative-2026-10-22.json):
native cycle validation and complete-solve comparisons exist. Six of eight
bounded comparisons improved by 1.29–5.20%; two regressed by 0.95–1.10%.
The feature remains opt-in. These historical results justify profiling and
selective tuning, not enabling it universally or claiming complete FEM speedup.

Reviewed again on 2026-10-22 against the current mixed working tree. CUDA
primitives and several persistent production sessions already exist. The
remaining task is to connect every supported public workflow, make unsupported
requests explicit, and eliminate recurring bulk host work where residency is
claimed. Complete CUDA coverage is **not established**. Historical native test
and benchmark records below apply to their stated workloads and source snapshots.

### Current findings by component

| Component | Current execution | Remaining coverage work |
| --- | --- | --- |
| Compute | Vendor BLAS and sparse actions across the four principal scalar types; persistent CSR; native FEM/BEM and Krylov helpers. | Runtime module failures can change casts, gather, contraction and reductions into host-staged operations. Check the entire dependency set before execution, and account for legacy allocations outside pool quotas. See `CudaBackend::capabilities` in `cuda/cublas.rs`. |
| Solvers | `CudaLinearSolveSession` retains CSR, Krylov and preconditioner state; prepared FGMRES accepts auxiliary actions. AMG setup remains on the host. | Migrate one-shot application callers; audit direct, eigen, other iterative, nonlinear, optimization and time algorithms individually. Preserve symmetry for PCG, shifted auxiliary preconditioning for Helmholtz and field splits for mixed systems. |
| BEM | Separate retained dense, H-matrix and non-directional MLFMM sessions support bounded constant discontinuous Tri3 problems. Compressed setup remains host work. | Complete application integration through common dispatch; extend frequency/material changes, shared/higher-order bases, directional FMM, singular/facet cases, fields and derivatives independently. Preserve compressed storage for large systems. |
| FEM | Generic admission includes bounded acoustic, elastic, Maxwell, Stokes, heat-conduction, scalar magnetic and electrostatic-potential systems. Specialized Tet/hex pressure, thermal time and resident field sessions extend this set. | Reconcile specialized producers with generic admission. Extend the family backlog below through physical forms, constraints, updates and outputs, not just tagged synthetic matrices. Arbitrary Rust form callbacks need explicit device lowering. |

A common retained BEM dispatcher is present in
[`bem_solver_cuda_scattering.rs`](../crates/sonium-bem/src/core/bem_solver_cuda_scattering.rs).
Its native tests now exercise dense, H-matrix and MLFMM selection, retention,
CPU parity and recovery. Application migration remains open. Its CUDA-specific
`Auto` policy now selects H-matrix at 1000 panels, matching the exclusive dense
storage ceiling; the CPU crossover remains 1280. The requested linear solver
is preserved: `Auto + Direct` rejects compressed sizes, while `Auto + GMRES`
uses the compressed route. Explicit assembly choices remain unchanged.

### Execution plan and completion gates

The next implementable work packages are below. These are planned changes,
not claims that the paths are already closed. The later implementation log
preserves the evidence for completed, bounded increments.

| Order | Work package and source anchor | Required result |
| --- | --- | --- |
| A | Compute runtime admission: [`capabilities`](../crates/sonium-compute/src/cuda/cublas.rs), shared device operators and allocation statistics. | Check the transitive operation requirements of the selected solver/preconditioner. Missing cast, gather, contraction or reduction modules must reject strict-native requests before expensive preparation. Account for vendor scratch and raw allocations as well as pooled buffers. |
| B | Coverage registry: [`FemWorkload`](../crates/sonium-fem/src/backend.rs), solver facade, BEM dispatch and application producers. | Enumerate meaningful CPU workflows with separate setup, solve, update and output status. Generate admission tests and the report from this registry; an explicit rejection is a tracked gap, not CUDA completion. |
| C | Retained application integration: [`CudaLinearSolveSession`](../crates/sonium-solvers/src/solve/cuda_session.rs) and [`CudaScatteringSession`](../crates/sonium-bem/src/core/bem_solver_cuda_scattering.rs). | Keep context, operator, Krylov buffers and suitable preconditioners across RHS changes. Define numeric refresh and structural rebuild boundaries. Verify failure invalidation and forward/transpose/Hermitian semantics through each public producer. |
| D | BEM sweep producer: [`NumCalc run_project`](../crates/sonium-bem/src/numcalc/runner.rs) and [`frequency_sweep`](../crates/sonium-bem/src/core/solver/frequency_sweep.rs). | Add explicit CUDA dispatch preserving prepared BCs, frequency-dependent admittance, harmonic/interior conventions and configured Burton–Miller scaling. Retain state across compatible updates; rebuild quadrature when its schedule changes. Track GCRO-DR separately from previous-solution warm starts. Connect field evaluation and report any remaining host output boundary. |
| E | Broader solver and FEM execution: split solver crates and the family backlog in section 3. | First connect host-assembled systems to native solves with appropriate preconditioners. Then lower recurring forms, constraints, material/history updates and outputs. Close each remaining direct/eigen/iterative algorithm individually; a replacement algorithm is not feature parity. |
| F | Resident FEM performance experiments: Tet/hex pressure sessions and compute kernels. | Profile complete solves at matched accuracy, then extend the four kernel priorities below. Require benefit over the existing resident implementation, not just over CPU execution. |

Packages A and B establish the contract for C–E. Package F can proceed for a
family once its resident workflow passes; it does not need to wait for every
other physics family. For “all paths”, the registry must include applicable
single/block RHS, four principal scalar types, real/complex adjoints,
frequency/time/nonlinear updates, nullspaces, constraints, export and failure
paths. Unsupported combinations remain open until implemented or explicitly
removed from the agreed scope.

NumCalc's default route calls the host `frequency_sweep` and
`compute_total_field` functions. The explicit `cuda-jacobi` route now connects
the retained CUDA sweep and native off-surface fields, downloading results for
JSON export. Evaluation grids require fixed velocity/impedance constant-Tri3
traces; unsupported cases reject. Broader compressed and field-representation
migration remains open. The common scattering
session currently keeps frequency and materials fixed. Its dense GMRES route
requires an explicit point-Jacobi selection; this is not evidence that Jacobi
will converge efficiently for every BEM problem.

1. **P0 — enumerate the denominator.** Build a machine-readable registry from
   public CPU entrypoints. Each row identifies producer, physics, space,
   topology/order, scalar, boundary/constraint, action, solver/preconditioner,
   RHS layout, update type and output. Record implementation, dispatch, native
   correctness, residency and performance as separate fields. Include missing
   CUDA modules, failure recovery and explicit unsupported cases. Audit the
   split solver families and every subordinate FEM module, not only the common
   acoustic path. Keep common BEM dispatch tests required when enabling application routes.
2. **P1 — connect retained execution.** Route public producers through one
   retained context/pool, operator, suitable preconditioner and Krylov workspace.
   Define RHS-only reuse, numeric refresh and structural rebuild separately.
   Retain certified device solutions through downstream field evaluation.
   Failed updates must invalidate output and warm starts until repaired.
3. **P2 — close physics and representation coverage.** Use host assembly plus
   resident sparse solves as the first milestone for each remaining FEM family.
   Use dense/compressed/FMM BEM sessions appropriate to problem size. Add direct,
   eigen, nonlinear, optimization and time algorithms as individual work items;
   substituting another algorithm does not close a row.
4. **P3 — move recurring bulk work.** Lower numeric forms, BC lifting, frequency
   combinations, nonlinear residuals/tangents, accepted/trial history and field
   outputs onto CUDA. Host mesh construction and initial hierarchy setup may
   remain explicit setup boundaries. Compact convergence/control transfers are
   distinct from bulk vector staging.
5. **P4 — optimize complete workflows.** Establish CPU and resident CUDA baselines
   with the same accuracy targets. Report cold setup, warm update/solve/output,
   iteration counts, true residual, physical error, peak memory, allocations and
   transfer bytes. Require native tests, sanitizer checks and measured benefit
   before promoting a faster path. Every registry row must be closed or explicitly
   outside the agreed scope; no coverage percentage until the denominator exists.

### FEM numeric-combination timing decision

The [retained Helmholtz comparison](cuda-fem-fused-combination-benchmark-2026-10-22.json)
measured the fused two-base update against the original device copy/scale/AXPY
route through the same public session API. Across four P1/P2 cases (81–2401
DOFs), the update phase improved, but occupied only 0.2–0.3% of update-plus-solve
time. The first clean run had median paired sweep speedups of 1.0066x, 0.9971x,
0.9992x and 0.9998x. A second isolated run with the final binary gave 1.0067x,
0.9964x, 1.0001x and 0.9997x, confirming the same decision.
These sub-percent, mixed total-time differences do not justify default promotion.
Fusion is therefore **opt-in** through
`CudaBackend::with_fused_numeric_combinations(true)`; default sessions retain
the existing device operations. This supersedes the previous unconditional
fusion dispatch, while keeping the implementation and native validation.
Further work should profile solver/preconditioner iteration costs before tuning
this already inexpensive update phase. Larger meshes and update-dominated
workloads need separate measurements.

### Retained CSR preprocessing result

Profiling the retained Helmholtz harness identified CSR multiply/partition kernels
as roughly 55% of aggregate GPU kernel time. The CUDA backend now offers
`try_with_csr_preprocessing(true)`: it checks runtime API availability and performs
one preprocessing call per retained descriptor, reusing the existing workspace
across vector and numeric-value changes. SpMV and SpMM share the descriptor lock.
This applies to physical CSR actions and every retained AMG level without changing
the solver or hierarchy.

Two isolated complete update-plus-solve comparisons measured **1.09–1.11x** speedup
on four P1/P2 Helmholtz cases (81–2401 DOFs), with equal iteration counts and
unchanged independent residual/error gates. Tracing confirms fewer partition
kernels with unchanged multiplication counts. This is an **opt-in** result on one
GPU and a bounded workload, not a default promotion or complete CUDA coverage.
See [preprocessing evidence](cuda-csr-preprocessing-2026-10-22.json) and
[baseline profile](cuda-fem-retained-profile-2026-10-22.json). Warm timings exclude
first-action preprocessing, setup and explicit output/reference calculations.

### Fused CSR smoothing result

Kernel priority 1 now includes a fused CSR residual/Jacobi step for all four
principal scalar types. Four lanes cooperate per row; the old iterate remains
immutable while a separate output is written. AMG reuses its `ax` scratch and
swaps buffers after each successful sweep, preserving simultaneous Jacobi
semantics across blocks without extra copies or allocations. Enable explicitly
with `BackendAmgPreconditioner::set_fused_jacobi_enabled(true)`. Existing
single-block small-level smoothing takes precedence if both options are enabled.

Stable complete-solve comparisons show about **1.05–1.10x** improvement over the
already-preprocessed CSR baseline on four P1/P2 cases. One repeat became
nonstationary on the largest case; all samples are retained and a confirmation
run recovered stable results. The option remains disabled by default pending
larger and more varied workloads. Independent AMG-cycle, primitive and physical
adjoint memchecks passed; the primitive also passed synchronization checking.
See [fused CSR Jacobi evidence](cuda-fused-csr-jacobi-2026-10-22.json).

### FEM kernel decision after resident execution

Bounded implementations of the original four kernel priorities already exist.
The next work should extend and measure them in this order, adjusted by profiles:

| Priority | Kernel or supporting change | Acceptance criterion |
| --- | --- | --- |
| 1 | Extend fused AMG residual and Jacobi/Chebyshev update kernels; first select and reuse a suitable hierarchy. | Lower complete time to tolerance with unchanged residual/physical gates; include setup and coarse-level costs. |
| 2 | Extend fixed-pattern numeric CSR assembly, coefficient combinations and Dirichlet/Robin/facet updates. | Remove recurring host numeric assembly and uploads in sweeps/time loops; preserve symmetry and adjoints. |
| 3 | Tune Tet P1/P2 gather, local action and scatter with cached geometry and specialized layouts. | Beat the existing cuSPARSE or cached-element baseline in a full solve; measure atomic contention and register pressure. |
| 4 | Extend high-order quad/hex sum-factorized partial assembly together with its auxiliary coarse operator. | Improve order/mesh scaling and total solve time without degrading preconditioning. |

**Concrete next kernel experiment:** extend the existing fused CSR residual /
weighted-Jacobi smoother, `x_new = x + omega * inv_diag * (b - A*x)`, in the
auxiliary AMG cycle. Keep the old iterate separate throughout each sweep;
cross-block synchronization cannot be replaced by an ordinary block barrier.
Use cooperative multi-sweep execution only on levels that fit its supported
single-block limits. Benchmark level-size/row-length dispatch against the
existing implementation before selecting a default. If a fresh phase profile
instead shows numeric assembly dominating, promote priority 2. Existing Tet
and hex kernels mean priorities 3 and 4 are extensions, not first implementations.

For each kernel experiment, retain the same hierarchy, stopping criterion and
physical problem; compare cold preparation and warm time-to-tolerance separately.
Require unchanged physical accuracy, original-system residual certification,
recorded iteration counts, zero unexpected bulk transfers, and sanitizer checks.
A faster operator with a slower complete solve fails the performance gate.

Use a CPU/GPU timeline to locate transfers, allocations and launch gaps before
kernel tuning ([NVIDIA Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)).
High-order partial assembly requires a compatible preconditioning strategy
([MFEM](https://mfem.org/performance/)). A new universal SpMV kernel is not the
default next task.

Existing [thermal evidence](cuda-thermal-lagged-amg-2026-10-22.json) reports
3.20x and 3.33x faster warm update-plus-solve windows from explicitly retaining
AMG on two small physical meshes. Those results exclude initial preparation
and assembly and are not a general GPU speedup claim. They support testing
hierarchy reuse before optimizing an already inexpensive coefficient-combination
kernel. Large operator changes still need a refresh policy and convergence gates.

### Verification of this audit refresh

- Latest retry: root `cargo check` and then `cargo clippy` passed (Clippy
  warnings remain); logs `/tmp/cuda-audit-final-check.log` and
  `/tmp/cuda-audit-final-clippy.log`. No Rust or CUDA implementation was changed
  by this refresh. These default-feature checks do not certify CUDA execution.
- Retry source checks confirmed runtime fallback states, the five facade
  strategies plus prepared FGMRES, common BEM dispatch restrictions, NumCalc's
  CPU producer and the generic FEM admission table. TokenSave was rebuilding;
  current file contents were used for these findings.
- `cargo check`: passed on this retry (`/tmp/cuda-audit-retry-check.log`).
- `cargo clippy`: passed with warnings (`/tmp/cuda-audit-retry-clippy.log`).
- This retry changes documentation only. Native test and performance results
  below are existing workload-specific evidence, not newly rerun GPU validation.
- The audit initially found a missing dispatcher test module. The implementation
  continuation below repairs this blocker and records native validation.
- The audit itself changed documentation only; subsequent implementation and
  validation are recorded separately below.

## Delivery order and immediate decisions

### Current working tree review

Reviewed on 2026-10-22 against HEAD
`bf781ac41d8bc590375420660c3044198e6c2125` plus the existing staged,
unstaged and untracked implementation. HEAD alone does not reproduce this
snapshot. TokenSave was rebuilding; current source was used to verify the
dispatch and residency findings. Section 8 records earlier implementation and
validation. Subsequent implementation updates below record bounded CUDA
coverage and their validation separately from the audit findings.

### Retry findings and next implementation order

CUDA execution is available for bounded workflows, but complete coverage is
not established. Keep the existing implementations and close the following
gates in order:

1. **Connect the validated block solver lifecycle to production.** Restart
   storage now covers `n * nrhs` and survives calls without implicitly reusing
   old iterates. The fixture type error and misplaced active-width cache lookup
   are repaired. All-scalar CPU/CUDA restart and repeated-RHS checks pass;
   native memory checking reports zero errors. Pooled restart quota and
   final-cycle behavior have explicit regressions. Checked native H-matrix
   block actions and oriented CSR inverse application now connect retained BEM
   block sessions for fixed constant-Tri3 systems, with native production tests.
   Admission must account for cached-width memory before enabling this opt-in
   policy in application sweeps; legacy block basis allocations are not yet
   covered by the pool quota.
2. **Make the coverage denominator executable.** Generate the inventory in P0
   from public CPU workflows, including application producers. A primitive,
   action-only fixture or unsupported row cannot close an end-to-end row.
   Record host setup separately from recurring host execution. Runtime module
   availability must participate in admission.
3. **Finish retained solver integration.** Route each application through the
   appropriate persistent operator and preconditioner. Preserve PCG symmetry,
   shifted auxiliary FGMRES for indefinite Helmholtz, and Schur/auxiliary-space
   methods for mixed systems. Track the remaining direct, eigen, nonlinear,
   optimization and time-integration algorithms individually.
4. **Close BEM production routes.** Extend the fixed-frequency constant-Tri3
   dense and compressed sessions to missing block, update, basis, field and
   derivative cases. Non-directional MLFMM now has retained core and public scattering solve APIs;
   connect application dispatch, then extend directional and shared-DOF
   representations. Preserve compressed storage for large systems.
5. **Close FEM execution family by family.** Start with host assembly plus
   resident sparse solves for each valid physics/space/scalar combination;
   subsequently move recurring form, boundary, nonlinear and history updates
   onto CUDA. The detailed family backlog below is the implementation scope.
6. **Optimize a validated resident baseline.** Profile complete solves before
   selecting another kernel. Existing Tet and hex implementations already
   cover bounded versions of kernel priorities 1–4; extend and measure them
   rather than scheduling duplicate first implementations.

The first kernel experiment should extend **fused residual and smoother
updates in the auxiliary AMG cycle**, provided a fresh phase profile confirms
that cost. Compare it against the existing fused Jacobi/Chebyshev paths, with
unchanged hierarchy, tolerance and original-system residual checks. In parallel
planning, retain fixed-pattern numeric assembly/BC updates for sweeps, specialized
Tet gather/action/scatter for low orders, and tensor-product partial assembly
for higher-order hexes. These target different bottlenecks; none is a universal
replacement for cuSPARSE.

The repository's [controlled AMG evidence](cuda-fem-amg-release-2026-10-22.json)
already reports benefits from hierarchy selection on bounded hex workloads.
That makes preconditioner quality a prerequisite to kernel ranking, not a new
performance claim for this checkout. Use a CPU/GPU timeline to identify launch,
transfer and synchronization costs ([NVIDIA Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)),
then profile the dominant kernels. High-order partial assembly must include an
effective auxiliary or multigrid preconditioner; operator-only speed is
insufficient ([MFEM partial assembly](https://mfem.org/performance/)).

The immediate execution blockers are:

| Priority | Finding in current source | Required change before claiming coverage |
| --- | --- | --- |
| P0 | CUDA capabilities depend on successfully loaded optional modules. Casts, gathers, local contractions and deterministic reductions can report host staging. | Admit the actual operator/preconditioner dependency set at runtime. Test missing-module and strict-native rejection paths, not only successful CUDA initialization. |
| P0 | `CudaLinearSolveSession` now retains the five admitted facade strategies across RHS changes; the one-shot convenience call uses the same implementation but creates a pool per call. | Migrate application callers to retained sessions. Fixed-pattern numeric refresh now retains physical CSR/Krylov storage and refreshes or rebuilds AMG; structural changes require a new session. Prepared FGMRES now admits caller-selected native preconditioners; extend production integration and validate each action’s residency. |
| P0 | Generic FEM admission and specialized Tet, hex, room and DG producers describe different supported sets. | Generate one coverage inventory containing each public producer, including its setup, solve, update and output boundaries. Do not broaden admission merely because a compute kernel exists. |
| P0 | Shared block restart/cache regressions now pass on CPU/CUDA, with zero allocations for repeated identical RHS and zero memcheck errors. | Native adapters now connect bounded BEM block sessions. Account for cached-width memory and extend producer coverage; shared solver evidence alone does not close application rows. |
| P0 | `prepare_cuda_hmatrix` now connects incident evaluation, compressed action, cluster-block/Jacobi preconditioning and persistent GMRES. The validator call and missing test module are fixed; native small and 1,280-panel cases pass. | Extend frequency updates, broader block producers, memory estimates and field/derivative coverage. A separate retained resident-RHS session now connects forward/transpose/Hermitian actions with oriented preconditioning. Fixed off-surface fields now retain device data. Keep optional exact-TBEM host certification visible; bounded production tests do not close all compressed BEM paths. |
| P1 | Host form callbacks, hierarchy construction, nonlinear/time history and several output paths remain CPU work. | First provide correct assembled CUDA solves for each family; then lower recurring bulk work to device adapters. A host callback is not automatically GPU executable. |

The H-matrix implementation freezes geometry, frequency, material and
formulation. Repeated incident fields and fixed off-surface field evaluation are its reuse
scope. The separate linear session also retains forward/transpose/Hermitian
solves for resident RHS vectors and blocks; frequency sweeps, per-session budget estimates and
broader field quadrature still require separate work. Native tests now verify this bounded producer,
including independent exact residuals and zero warm device allocations. The
host-factor compression ratio is modest and excludes the device adapter's
orientation copies, sparse preconditioner and Krylov storage.

For the first implementation milestone, use the following complete workflows:
an SPD FEM solve with stationary symmetric AMG/PCG; an indefinite Helmholtz
frequency sweep with shifted auxiliary AMG/FGMRES; a Stokes solve with a pressure
gauge and Schur preconditioning; and a compressed BEM solve with repeated
incident fields. Each must retain its state, verify the original-system residual
and physical outputs, and report cold setup separately from warm execution.
These establish the shared lifecycle; the family inventory below remains the
checklist for extending coverage to all CPU-supported paths.

### Audit summary

| Component | Execution already available | Coverage blocker | First action |
| --- | --- | --- | --- |
| Compute | Native dense/sparse algebra, device operators, Krylov helpers and specialized BEM/FEM kernels. | Some optional operations stage through the host; allocation and capability guarantees vary by route. | Make strict CUDA admission and transfer/allocation accounting consistent across all operator paths. |
| Solvers | Retained PCG/FGMRES, block methods, mixed-precision infrastructure and device AMG cycles exist. | Public dispatch and lifecycle coverage are narrower than primitive coverage; hierarchy setup, several algorithms and direct/eigen workflows still use host work. | Connect persistent producer sessions to physics-appropriate preconditioners, with numeric refresh and explicit rebuild rules. |
| BEM | Small constant-Tri3 dense sessions connect incident evaluation, assembly, LU and certification; a bounded constant-Tri3 H-matrix producer connects persistent preconditioned GMRES and repeated incidents. | Broader compressed production solves, higher-order/shared-DOF and directional cases, general field quadrature and derivatives remain incomplete. | Extend the retained H-matrix producer and integrate FMM solves without expanding large problems to dense storage. |
| FEM | Bounded assembled sessions, Tet/hex pressure sessions, room sweeps, cached BDF linear solves and resident compressible DG exist. | Generic forms, application dispatch, other physics/constraints and repeated nonlinear/history work are not uniformly resident. | Broaden assembled execution family by family, then lower repeated forms, BCs and state updates to CUDA. |

“All paths” means all meaningful CPU-supported combinations, including errors,
updates, transpose/Hermitian actions and outputs—not merely one GPU solve per
crate. The registry in P0 is the completion checklist; this source audit scopes
that registry but does not claim exhaustive runtime validation. Existing evidence
records are historical, workload-specific results, not fresh tests of every
uncommitted implementation.

The audit covers compute, every split solver family, BEM assembly/solve/field
routes, and FEM infrastructure and physics families. CUDA coverage is incomplete.
Existing working-tree implementations must be integrated and tested through
public producers before adding more isolated kernels. The detailed inventory
below distinguishes native actions from host setup and repeated CPU execution.

| Order | Work package | Exit criterion |
| --- | --- | --- |
| 1 | Build the executable coverage registry described in P0. Include scalar, topology/order, physics, BC, action, solver/preconditioner and RHS layout. | Every supported CPU workflow is represented; unsupported CUDA requests fail explicitly; each admitted row names a required native production test. |
| 2 | Unify feature forwarding and retained device sessions across compute and solver APIs. Migrate public producers away from per-call contexts, uploads and downloads. | Repeated solves retain operators, Krylov storage and appropriate preconditioners; transfer/allocation counters verify the claimed hot-loop residency. |
| 3 | Complete solver/preconditioner routes: PCG+AMG for SPD/HPD, FGMRES+shifted auxiliary solves for Helmholtz, field-split/Schur for saddle points; inventory remaining direct, eigen, nonlinear and time algorithms separately. | Original-system residual and physical-output parity pass without substituting an unsuitable algorithm or hiding a CPU fallback. |
| 4 | Connect BEM pair assembly directly to device matrix/RHS and solve; retain compressed H-matrix/FMM representations for large systems. Extend missing basis, directional/shared-DOF, field and derivative routes. | End-to-end constant/higher-order, near/singular, BC, sweep and adjoint cases pass for each admitted representation within its memory envelope. |
| 5 | Broaden FEM execution first through assembled device operators, then move repeated numeric assembly, BCs, nonlinear/history updates and outputs to device implementations. Reconcile generic admission with specialized producers. | Each physics-family backlog row has production evidence; mesh/constraint changes invalidate caches correctly; host callbacks are explicit gaps until lowered. |
| 6 | Optimize FEM using complete-solve profiles and the kernel priorities in section 4. | Improve time to the same accuracy, recording setup, iterations, transfers and peak memory as well as kernel time. |

For the next FEM performance experiment, prioritize **auxiliary AMG hierarchy
and smoother cost**, then extend **residual/smoother fusion**. Existing
measurements show high outer iteration counts and specialized actions slower
than cuSPARSE; a new generic SpMV kernel is not justified. The broader kernel
backlog remains Tet P1/P2 fused element action, residual/smoother fusion,
fixed-pattern numeric assembly and boundary kernels, and high-order hex/quad
sum factorization. Initial implementations already exist for these four areas;
their remaining integration and optimization work is described below.

### Coverage closure checklist

The following deliverables make the execution-first plan reviewable. They remain
open until the public workflow passes its native CUDA gate. Existing kernels and
historical measurements are implementation evidence, not blanket completion.

| Deliverable | Concrete change | Closure test |
| --- | --- | --- |
| One coverage inventory | Reconcile compute runtime capabilities, solver dispatch, generic FEM admission and specialised BEM/FEM producers in a machine-readable registry. Include optional kernel-loading failures. | Every public CPU workflow maps to a supported row or an explicit backlog row; strict CUDA never silently stages bulk operations or falls back. |
| Persistent public solver API | Let callers retain context, CSR descriptors, preconditioner and Krylov storage across calls. Distinguish RHS-only, numeric-value and structural updates. | Repeated RHS preserves factors/hierarchy; value updates refresh or explicitly lag preconditioning; structural changes rebuild; failed updates cannot publish stale solutions. |
| Scalable BEM producer | Connect compressed H-matrix/FMM actions to retained preconditioned solves and field evaluation. Add basis, singular integration, directional and shared-DOF cases separately. | A large production case stays compressed within its memory budget and matches CPU/analytic surface and field outputs. Small dense success cannot close this item. |
| FEM production coverage | Extend assembled device solves family by family, with suitable preconditioners and validated BCs, orientation, gauges and nullspaces. Reconcile specialised Tet/hex/DG routes with admission. | Physical production references pass for each row, including repeated RHS and supported adjoints. Synthetic matrices only validate algebra. |
| Resident repeated workflows | Move numeric form/BC evaluation, nonlinear residuals and history updates into device adapters; retain state through frequency and time loops. | Transfer counters distinguish input/output from internal bulk staging; warm allocation counts and invalidation rules pass explicit assertions. |
| Remaining solver families | Give direct, eigen, nonlinear, optimization and time-integration algorithms individual rows and device contracts. | The requested algorithm and precision execute as advertised; a different CUDA algorithm is not counted as feature parity. |

For each row, record separately: source implementation, public dispatch,
native correctness test, residency test and performance result. Require all five
before calling a path both covered and optimised. Coverage percentages should
wait until the CPU workflow denominator has been enumerated.

### Work package details

These slices turn the coverage backlog into reviewable changes. They are proposed
work, not completed CUDA coverage:

1. **Make backend selection auditable.** Generate admission and rejection reasons
   from the P0 registry. Cover generic FEM, specialized Tet/hex, room sweeps,
   BDF, DG and BEM producers independently. An explicit strict CUDA request must
   return a checked error on unsupported work or device failure; an automatic
   backend choice may recover on CPU with recorded provenance.
2. **Complete preconditioner refresh.** BDF now retains concrete polynomial,
   Chebyshev and f64 Jacobi–Schur preconditioners. Fixed-pattern Schur updates
   refresh full/velocity/pressure values, inverse diagonals and spectral bounds;
   partially completed uploads invalidate the action until complete recovery.
   Pressure AMG now refreshes matching hierarchy storage and rebuilds on structural
   changes. Complex-double CUDA coarse LU now retains storage on compatible hierarchy refresh; host hierarchy construction and complex-single coarse refresh remain open.
   Extend refresh to remaining precision modes and reduce host numeric work,
   while preserving fresh-setup parity and failure recovery.
3. **Close production assembled FEM gaps.** Migrate application callers onto
   retained sessions, then extend the bounded Tri3/Tet4 PEC Maxwell producer and the
   remaining family rows. Maxwell now has scalar/vector auxiliary AMG corrections;
   PML, other boundary conditions and higher orders remain open. Verify constraints, gauges,
   nullspaces and adjoints through each producer. A synthetic matrix tagged
   with a physics name does not qualify as that physics' production test.
4. **Close scalable BEM execution.** Connect retained H-matrix and admitted FMM
   actions to appropriate preconditioned solves, repeated RHS and field output.
   Add explicit cases for shared DOFs, directional plans and higher-order
   singular integration. Keep large systems compressed throughout the solve.
5. **Tune the measured FEM bottleneck.** Profile auxiliary AMG setup, levels,
   coarse solve and smoothing in retained Tet/hex solves. Extend residual and
   smoother fusion only where the profile predicts an end-to-end benefit.
   Compare against the current zero-start smoother baseline with unchanged
   tolerance and report iteration changes separately from kernel savings.

For every slice, record cold and warm timings, original-system residual,
physical error, peak device memory, allocations and transfer bytes. Exercise
both public dispatch and the device implementation: a passing device unit test
does not prove that an application selected it. BDF currently uses an
unrestarted Krylov budget (`restart = max_iterations`); include retained basis
memory in scaling tests before increasing the iteration limit.

### Evidence and remaining coverage

BDF now connects retained pressure AMG to the checked hierarchy-refresh API.
Changed pressure matrices refresh all level and transfer values, smoother data
and checkpoints when hierarchy structure matches; incompatible hierarchies
rebuild. The physical BDF fixture retains one device setup rather than three
across changed timesteps, with zero new device allocations, unchanged iterations
and CPU velocity/pressure/history parity. A separate larger algebra fixture
verifies multilevel reuse and rebuilds even when finest CSR sparsity is unchanged.
Injected partial uploads evict the BDF cache. Both native suites pass memcheck.
See [BDF AMG refresh evidence](cuda-bdf-amg-refresh-2026-10-21.json).

Host AMG hierarchy construction still runs for changed pressure values. Device
setup counts therefore do not describe total host setup work or establish a
speedup. Complex-double CUDA direct coarse factors now support retained numeric refresh; complex-single coarse factors still require rebuilding.
The small physical BDF refresh fixture has a single AMG level. The working tree
also contains an optional pressure-only hierarchy with retained device
restriction/extension buffers, selected by
`linear_solver_bdf_pressure_only_amg` (default `false`). The default embedding
includes velocity identity rows. Pressure-only execution passes native physical
and block-action parity, layout invalidation and residency tests under memcheck.
Matched complete warm linear solves at 91, 190, 325 and 703 pressure DOFs favor
pressure-only only at 325 DOFs; keep it opt-in. Fewer unknowns or iterations
alone do not establish a speedup. See
[pressure layout evidence](cuda-pressure-only-amg-2026-10-21.json).

Profiling exposed CSR applications of the inverse diagonal inside Chebyshev
smoothing. Connecting the existing native diagonal-scale primitive removes
that sparse dispatch and folds in the recurrence coefficient. Both embedded
and pressure-only BDF solves improve 1.39–1.46x across the four measured sizes,
with unchanged iterations and zero warm device allocations. These timings
include complete warm linear solves but exclude assembly and setup; host BDF
history and hierarchy construction remain open. The five-test BDF suite and
three-test AMG refresh suite pass CUDA memcheck. See
[Chebyshev diagonal evidence](cuda-chebyshev-diagonal-2026-10-21.json).

Fresh corrections now also skip the first Chebyshev matrix action and residual
construction, initializing both the correction and recurrence direction from
the scaled inverse diagonal. W/F revisits preserve their accumulated correction.
Independent Complex64 scalar references cover V/W/F cycles, zero/one/three
pre-sweeps, repeated block RHS and NaN output sentinels on CPU and CUDA. Both
that suite and the physical BDF suite pass memcheck. The same complete warm
BDF solves improve a further 1.03–1.09x relative to diagonal scaling alone,
with unchanged iterations. The matched trace removes 1,245 CSR actions and
6,225 runtime kernel launches. See
[zero-start Chebyshev evidence](cuda-chebyshev-zero-2026-10-21.json).
AMG Chebyshev now also uses a fused native residual/diagonal/recurrence/solution
update for f32/f64/c32/c64. The checked hook preserves portable fallback and
overwrites stale direction values when the old-direction coefficient is zero.
Primitive parity, invalid-input/foreign-stream/launch-failure tests, independent
complex V/W/F-cycle references and physical BDF tests pass CUDA memcheck.
The four matched BDF meshes improve another 1.12–1.18x relative to the zero-start
baseline, with unchanged iterations and zero warm device allocations. The
matched trace replaces 33,348 kernel launches while retaining the same CSR
action count. See [fused Chebyshev evidence](cuda-chebyshev-fused-2026-10-21.json).
Standalone checked Chebyshev scalar/block preconditioners now use the same
native operations. Column-major hooks broadcast one diagonal across all RHS
columns with one launch per operation, including explicit empty-shape handling.
Warm scalar/block interleaving allocates no device buffers and transfers no host
arrays. CPU-reference parity covers all four scalars and degrees 1/4; primitive,
numeric-refresh, physical BDF and room block-solve suites pass memcheck. The room
fixture accepts all four frequencies on CUDA with zero warm allocations and
passes CPU physical parity. Alternating baseline/candidate microbenchmarks at
4096 rows show 2.01–2.52x faster degree-4 applications and 2.51–4.66x faster
degree-1 applications. These are preconditioner timings, not whole-simulation
speedups. See [standalone Chebyshev evidence](cuda-chebyshev-standalone-2026-10-21.json).
Host spectral estimation, device recurrence coefficients and broader production
coverage remain open.

The underlying AMG refresh API passes f64/Complex64 fresh-hierarchy and
checkpoint parity with zero refresh allocations. See
[AMG hierarchy refresh evidence](cuda-amg-hierarchy-refresh-2026-10-21.json).
Earlier bounded records cover
[unchanged-pressure retention](cuda-pressure-amg-retention-2026-10-21.json),
[Schur numeric refresh](cuda-schur-numeric-refresh-2026-10-21.json) and
[outer polynomial/Chebyshev refresh](cuda-bdf-numeric-refresh-2026-10-21.json).
The current BDF integration supersedes their pressure-AMG rebuild limitations
for matching hierarchy structures.

Audit scope: this is an execution-coverage and optimization plan, not a claim
that every CUDA route is complete. The existing uncommitted implementation and
its historical evidence are included; unverified additions stay marked as such.


The implementation history in section 8 is cumulative. The latest bounded
results supersede earlier failures and allocation counts for the same fixtures:

| Area | Latest evidence | Next concrete task |
| --- | --- | --- |
| Scalar room sweeps | Retained physical and RHS CSR descriptors plus Krylov workspaces give allocation counts 84/0/0/0 across four frequencies for both tested precision modes. Pure-f32 certification now meets the requested tolerance. | Extend the residency tests to changing demand and larger production workloads; remove recurring diagnostic bulk transfers. |
| Block room solves | The producer retains `BlockGmresWorkspace`; point-Jacobi, shifted-AMG, polynomial-Jacobi and Chebyshev routes yield accepted block results at all four tested frequencies. Typed diagnostics distinguish acceptance from producer scalar recovery. Native initialization, QR and recovery keep full vectors on CUDA. | Active-prefix copies now reduce coefficient transfers; reduce remaining warm allocations and verify each operator/preconditioner and producer boundary independently. Accepted block results alone do not establish whole-workflow residency. |
| FEM smoother | Native fused Jacobi updates and zero-start diagonal scaling are integrated into retained AMG cycles. The latest matched Tet solves improve a further 1.05–1.12x, with identical iterations; see section 8 for distinct baselines. | Improve auxiliary hierarchy/coarse correction and profile the remaining cycle cost; measure any further fusion against this updated baseline. |
| FEM element and assembly kernels | Initial Tet P1/P2, fixed-pattern numeric/boundary assembly and high-order hex partial-assembly implementations exist. | Broaden production integration and coverage before claiming priorities 1–4 complete; retain cuSPARSE where it wins. |
| Retained tetrahedral pressure solves | Affine P1/P2 volume sessions connect numeric CUDA updates, Dirichlet lifting, shifted AMG and FGMRES in both CSR and element modes. Recorded repeated solves allocate no device buffers; numeric frequency updates transfer no host data. | Integrate device coefficient laws, block RHS and generic production admission; keep CSR as the default: matched complete solves still favor cuSPARSE after the P2 geometry optimization. |
| Shared assembled FEM session | `solver::device::CudaFemSolveSession` admits before invoking checked host assembly, then retains the requested CSR action, RHS, preconditioner and PCG/FGMRES workspace. Native production fixtures cover 3D elasticity, Tri3 P1/P2 pressure, Taylor–Hood Stokes with an owned Schur preconditioner, and PEC Maxwell with scalar/vector auxiliary AMG (Tri3 forward; Tet4 forward/transpose/Hermitian). | Integrate application callers, the nonlinear/BDF lifecycle, broader Maxwell physics and block RHS; reconcile specialized Tet/hex admission separately. Host setup and preconditioner rebuilding remain explicit. |
| BDF linear solves | The public cached BDF f64 CUDA branch retains context, physical CSR, buffers and flexible Krylov workspace. Identical matrix/Schur/controls reuse polynomial, Chebyshev or Jacobi–Schur setup, including owned pressure AMG. | Move host numeric assembly, RHS/history and output state onto CUDA; retain mixed-precision and steady nonlinear sessions; remove host hierarchy/numeric work, add pressure-only AMG and strict fallback policy. |
| Public dense BEM solves | `BemSolver::prepare_cuda` connects incident evaluation, assembly, retained LU and residual certification for constant Tri3 rigid/impedance scattering below 1000 DOFs. | Add device field evaluation and broader boundary/basis coverage; integrate compressed representations for larger systems. |

Owned polynomial-Jacobi and Chebyshev preconditioners now retain their device
CSR copies across numeric updates. The recorded four-frequency production
fixtures report allocation counts of 227/0/0/3 and 222/0/0/0 respectively.
These are allocation reductions, not measured elapsed-time speedups. Chebyshev
spectral estimation and diagonal uploads still use the host, and the extra CSR
copy must be included in memory budgets. See the
[owned-preconditioner evidence](cuda-owned-polynomials-2026-10-21.json).

Evidence: [room descriptor reuse](cuda-room-persistent-csr-2026-10-21.json),
[f32 certification](cuda-room-f32-certification-2026-10-21.json),
[complete-solve smoother comparison](cuda-jacobi-fem-solves-2026-10-21.json),
and the current `solve_f32_sources_block` and block-Krylov initialization sources.
These are workload-specific results, not a claim of complete CUDA coverage.

This plan does not equate an inventoried unsupported path with completed CUDA
coverage. One-time host geometry/hierarchy setup is permitted for the first
execution milestone; recurring bulk host work remains a separate residency gap.
The existing implementation history and benchmark evidence are retained below;
they apply only to their named workloads and source revisions.

Audit refreshed: 2026-10-21. Base revision:
`bf781ac41d8bc590375420660c3044198e6c2125`, including the current uncommitted
compute, solver and room-sweep changes. Graph results were used as locators;
current source is authoritative for the changed paths. Native CUDA execution now works outside the filesystem/device sandbox. The
Tet P1/P2 and hex kernel tests pass on the RTX PRO 6000 Blackwell, including
CUDA memcheck. Broader production parity and measured speedups remain open.

The main gaps are production integration and residency: compute already has
extensive CUDA primitives, while many producers still assemble, update or
exchange bulk data on the CPU. Persistent GMRES/PCG sessions, Tet P1/P2 volume and numeric assembly
kernels, and a first hexahedral partial-assembly primitive now exist in the
working tree, but do not yet close the complete FEM workflow. Sections 2–6 describe current coverage and remaining work;
section 8 preserves the implementation history.

The priority is to connect the existing CUDA primitives to complete production
workflows, preserve device state across repeated solves, and provide appropriate
preconditioners. Only then should FEM kernel tuning become the main workstream.
CUDA already implements considerably more than BLAS and regular BEM pairs.
Conversely, a CUDA feature, successful GPU matrix action, or QA capability row
does not establish a fully accelerated simulation.

## 1. What complete coverage means

Track each supported CPU workflow through input preparation, assembly, boundary
conditions, preconditioner setup, solve, updates, derivatives, and outputs.
Give each stage one of these statuses:

| Status | Meaning |
| --- | --- |
| Device native | Bulk data and arithmetic remain on CUDA, including vendor libraries. |
| Host setup / device execution | Explicit one-time mesh, sparsity, hierarchy, or factor preparation on CPU; reusable GPU action. |
| Hybrid repeated execution | Bulk arrays cross the host boundary during frequency, Newton, Krylov, or time loops. |
| CPU only / unsupported | No admitted CUDA implementation for the exact workload. |
| Unverified | Implementation exists, but the production workload has no passing GPU evidence. |

Two milestones are required: **execution coverage** (every supported workflow has
a working CUDA route, with setup boundaries stated) and **resident execution**
(no recurring bulk CPU work or transfers in the chosen hot loop). A CPU fallback
or an unsupported row accounts for a gap; neither counts as CUDA coverage.
Mesh import, visualization, and tiny control decisions can remain on the CPU.
Cold setup is measured separately and moved only when it matters to total time.

The coverage key must include physics, dimension, element topology and order,
function space, scalar, assembled/element-local/partial-assembly mode, forward /
transpose / Hermitian action, BCs, solver, preconditioner, and single/block RHS.
Only meaningful CPU-supported combinations belong in the denominator. Arbitrary
Rust form callbacks cannot execute on CUDA without an explicit device lowering.
New CPU forms must either supply that lowering or create a visible coverage gap.

## 2. Current source audit

### Compute

| Area | Present implementation | Remaining work |
| --- | --- | --- |
| Dense and sparse algebra | CUDA capability table reports vendor-native BLAS-1, GEMV, GEMM, TRSM, SYRK, CSR SpMV/SpMM and CSR transpose for f32/f64/c32/c64. | Exercise every action, view, stride, batch and scalar through production callers; reuse descriptors and workspaces. |
| Krylov helpers | Fused multi-dot and basis operations, casts, weighted gather, segmented reduction and batched local contraction exist. | Kernel availability is conditional. Cast/gather/local contraction and some reductions can stage on host; strict requests must reject those cells. |
| Reductions | Vendor-native ordinary reduction and explicit host-staged reduction capability coexist. | Preserve deterministic semantics; add a native deterministic mode if required, or reject it under strict CUDA. Avoid bulk downloads for norms. |
| Memory and execution | MemoryPool, persistent CSR, transfer/runtime statistics, checked errors, failure injection and CUDA graph capture exist. | Audit allocations through raw backend paths, vendor scratch, cached descriptors and pool ownership together; prove quota accounting rather than assuming that creating a pool covers all allocations. |
| Transforms | cuFFT plus real-transform helpers exist. | Keep capability/runtime checks and plan reuse; transforms are dependencies of some solvers, not a universal FEM requirement. |
| Specialized arithmetic | BEM quadrature, H-matrix/MLFMM actions, local FEM contraction, colored GS, explicit gather/scatter/hourglass/min-dt and compressible DG kernels exist. | Integrate and extend these; do not schedule them as entirely new implementations. |

Evidence: [CUDA capability table](../crates/sonium-compute/src/cuda/cublas.rs)
(`capabilities`),
[local kernels](../crates/sonium-compute/src/cuda/kernels/local_kernels.cu),
[device operators](../crates/sonium-compute/src/operator.rs), and
[explicit kernels](../crates/sonium-compute/src/cuda/kernels/explicit_kernels.cu).
The capability table explicitly marks hierarchical BEM setup as host work.

### Solvers, including the split crates

| Layer | Present implementation | Gap / action |
| --- | --- | --- |
| High-level facade | CUDA admits `Gmres`, `GmresJacobi`, `GmresAmg`, `Cg`, and `CgAmg`. Public `CudaLinearSolveSession` retains context via a caller pool, CSR, preconditioner, RHS and GMRES/PCG state across calls; host and device results and warm starts are explicit. CG and AMG retain SPD/HPD admission; CG+AMG uses a stationary symmetric cycle. | Fixed-pattern numeric updates now refresh Jacobi/AMG, with explicit AMG rebuild reporting. Canonical sparsity and controls remain fixed. Prepared FGMRES accepts suitable auxiliary actions for indefinite Helmholtz; migrate applications and validate additional prepared actions. The one-shot convenience API still creates a pool and downloads host output. |
| Iterative | Checked GMRES, FGMRES, block GMRES and block FGMRES wrappers; compute also has resident PCG and mixed-precision GMRES-IR. | Reusable `DeviceGmresSession` and `DevicePcgSession` now retain scratch and certify published results; finish producer integration and preconditioner selection. Audit remaining host algorithms (BiCGSTAB, MINRES, TFQMR, IDR(s), etc.) individually; port requested algorithms without silently substituting GMRES. |
| Preconditioners | Device point/polynomial Jacobi, Chebyshev, block Jacobi and saddle-point Jacobi/Schur exist; backend AMG uploads a host hierarchy and performs device cycles. CUDA c32/c64 coarse LU exists. | Expose these consistently. Establish native ILU/ILUT/IC/Schwarz alternatives and adjoint actions; complete AMG setup/update/coarse-solver coverage. Host setup is acceptable for the execution milestone, but not a claim of GPU setup. |
| Sparse/core/sys | Host CSR/BSR/triplets and host operator traits coexist with compute's device operators and persistent CSR. | Define an explicit host/device operator bridge and persistent sparse lifecycle. CPU runtime/NUMA scheduling remains CPU infrastructure. Add format conversions only where needed. |
| Direct | Focused direct crate has CUDA f64 dense LU/Cholesky; compute exposes complex dense LU helpers. | Unify typed facade dispatch and device-output solves; audit sparse LU/Cholesky/LDLT, multifrontal/BLR, complex and mixed-precision coverage separately. Dense BLAS offload does not establish a native sparse factorization. |
| Eigen | Focused eigen crate has CUDA dense symmetric eigendecomposition and automatic routing. | Distinguish this from resident LOBPCG/generalized eigenproblems and adjoint eigen sensitivities; connect device operators and preconditioners there. |
| Nonlinear, optimization, time integration | Primarily host control interfaces; physics-specific device paths exist elsewhere. | Introduce device residual/Jacobian/JVP/VJP and history-state contracts. Keep small optimizer decisions on CPU but retain large state, gradients and linear solves on device. Arbitrary host callbacks need explicit adapters. |

Evidence: [facade](../crates/sonium-solvers/src/solve.rs) (`solve_cuda_backend`), [checked Krylov API](../crates/sonium-solvers/crates/sonium-solvers-iterative/src/backend.rs),
[PCG](../crates/sonium-compute/src/pcg.rs),
[AMG](../crates/sonium-solvers/crates/sonium-solvers-preconditioners/src/amg/mod.rs),
[dense direct](../crates/sonium-solvers/crates/sonium-solvers-direct/src/gpu.rs),
and [dense eigen](../crates/sonium-solvers/crates/sonium-solvers-eigen/src/cuda.rs).

The newly added `SaddlePointSchurSolver` in
[`saddle_schur.rs`](../crates/sonium-solvers/crates/sonium-solvers-direct/src/saddle_schur.rs)
factors host sparse primal/Schur blocks and solves each constraint column on the
host. Include its reusable factorization, repeated RHS and transpose convention
in the direct-solver inventory; it does not close CUDA saddle-point coverage.

CUDA feature forwarding now connects both BEM and FEM to compute and the solver
facade. The facade enables compute, iterative, preconditioner, direct and eigen
CUDA features. Package-scoped builds and resolved-feature inspection verify
this wiring independently of workspace-wide feature activation. This exposes
the existing backends; it does not make every solver strategy device-native or
remove the host-input/output boundaries of the dense direct/eigen APIs.

Two solver details are immediate correctness/residency gates, before tuning:

* Scalar `try_fgmres_backend_preconditioned` now retains each Z_j = M_j(V_j)
  and combines Z*y without reapplying the preconditioner. Pooled storage survives
  restart cycles and repeated solves, and `DeviceGmresSession::try_new_flexible`
  exposes this convention to persistent producers. All-scalar variable-action
  regression tests cover restarts and warm starts. Native retained Tet and hex
  pressure fixtures connect FGMRES to shifted auxiliary AMG; broader variable
  inner-solve policies and production integration remain open.
* Block Krylov now uses native CUDA finite checks, zero initialization,
  residual/QR operations, recovery column copies and retained scalar fallback
  Krylov storage. Backends without the optional native initialization hooks
  still stage inputs explicitly. All-scalar dependent/zero RHS and direct QR
  tests cover the new boundaries; check each production operator/preconditioner
  and keep compact Hessenberg/QR traffic distinct from bulk-vector transfers.
  Warm allocations, coefficient-transfer volume and full quota accounting remain
  open; no generic callback or producer is made resident merely by this solver.

Evidence: [scalar dispatch](../crates/sonium-solvers/crates/sonium-solvers-iterative/src/backend.rs)
(`try_scalar_backend`, line 901), [terminal preconditioner application](../crates/sonium-compute/src/gmres.rs)
(lines 1092–1115), and [block initialization](../crates/sonium-compute/src/block_krylov.rs)
(native initialization and recovery). These findings apply to the audited working tree.

### BEM

| Path | Present implementation | Gap / action |
| --- | --- | --- |
| TBEM constant Tri3 | Checked CUDA workspace for regular, near, singular/Duffy and hypersingular arithmetic; selected derivatives exist. The working tree also contains `assemble_device` and a retained dense `CudaTbemDeviceSystem` for fewer than 1000 DOFs. | The established `assemble` route still downloads pair outputs. Native matrix/RHS parity now passes for an 80-panel mixed-BC fixture with zero warm allocations and compact validation downloads. A retained dense LU session now certifies complete solves across three nearby frequencies and reuses factors for an RHS-only update. `BemSolver::prepare_cuda` now connects rigid/impedance scattering with native incident evaluation, source-descriptor updates and optional solution download. Geometry/quadrature rejection tests pass. Adjoint/block, budget estimates and broader producer coverage remain open. |
| Other bases/elements | Quad4 regular quadrature and real-channel helpers exist. CPU linear/quadratic/continuous-quadratic builders also exist. | The main checked TBEM validator still requires Tri3 with one DOF per panel. Helper/QA breadth must not be reported as full Quad4 or higher-order production coverage. Complete each singular/near/field/derivative combination. |
| H-matrix | Host-built compressed factors upload through `try_to_cuda`; device actions and wavenumber-update adapter exist. A checked `DeviceOperator` bridge and `prepare_cuda_hmatrix` retained GMRES producer pass bounded native solve/residency/failure tests. | Fixed off-surface fields now use resident traces. Retained arbitrary-RHS forward/transpose/Hermitian solves now use oriented diagonal/cluster preconditioning. Fixed-operator block sessions now pass production tests with oriented sparse inverse preconditioning. Extend frequency-update and broader basis/field/derivative routes. Tree/admissibility/ACA and self-block factorization remain host setup. The adapter retains all three orientations; host factor compression is not total GPU memory reduction. |
| MLFMM | An admitted non-directional hierarchy uploads to resident forward/transpose/Hermitian actions. A checked borrowed-buffer operator bridge and `MlfmmSystem::try_prepare_cuda_solve` connect retained scalar GMRES with oriented Jacobi/leaf-block Jacobi; a 320-panel physical fixture passes native tests. | Integrate application dispatch and native blocks/updates. Directional plans and repeated/shared leaf DOFs are explicitly rejected. Multi-RHS trait requests reject instead of using the generic host-staged default. Add cone layouts/translations and shared-DOF owner/reduction rules before claiming those cells. |
| SLFMM | Existing host assembly/action path. | No matching CUDA adapter was found in `slfmm.rs`; provide an adapter or a documented equivalent path with accuracy and memory validation. |
| Fields, sweeps, adjoints | Field and selected derivative kernels exist; host sweep/solver APIs remain. | Keep solution and evaluation data resident across frequencies/RHS; enumerate all exposed derivatives, not only wavenumber/source-x/source-y. Validate compression derivatives independently. |

Evidence: [TBEM](../crates/sonium-bem/src/core/assembly/tbem.rs)
(`CudaTbemAssemblyWorkspace::assemble`, 4230; validator, 4659),
[H-matrix](../crates/sonium-bem/src/core/assembly/hmatrix.rs) (`try_to_cuda`),
[MLFMM](../crates/sonium-bem/src/core/assembly/mlfmm.rs) (`try_to_cuda`, 4824),
and [solver integration](../crates/sonium-bem/src/core/solver/fmm_interface.rs).
Large BEM coverage must retain compression/FMM; a dense GPU allocation is not
an acceptable replacement for a scalable CPU algorithm.

The new [device TBEM handoff](../crates/sonium-bem/src/core/assembly/tbem_device.rs)
retains the dense matrix, RHS and pair-routing map. It uploads O(n) boundary
weights, combines pair channels on CUDA and exposes checked GEMV/GEMM actions.
It rejects changed geometry or quadrature schedules and dense systems at 1000
DOFs or above. The native mixed-boundary regression validates three assemblies against CPU
reference at relative tolerance 1e-9. Warm calls allocate no device buffers,
upload 12,800 bytes of coefficients and download 12 bytes of validation flags;
CUDA memcheck reports zero errors. The retained `CudaTbemSolveSession` now connects this assembly to device-input
LU and original-system residual certification. Native tests cover three nearby
frequencies, an RHS-only factor reuse, invalid tolerance and injected-launch
recovery. Warm solves allocate no device buffers; tracked downloads are 20 bytes
for refactorization or 12 bytes with LU reuse, plus cuBLAS norm scalars.
CPU original-system residuals and a two-column matrix action pass at 1e-9;
computed solve residuals are approximately 1.2e-15. This is bounded dense-session
coverage. Geometry/quadrature rebuild, adjoint/block solve and broader application
integration remain open. See [resident solve evidence](cuda-bem-resident-solve-2026-10-21.json). See
[assembly evidence](cuda-bem-device-assembly-2026-10-21.json). Transpose/Hermitian
operator and adjoint-solve coverage must be tracked separately.

For BEM, expand each assembly-family row across constant, linear, quadratic
and shared-DOF layouts; Tri3/Quad4; regular/near/self/hypersingular terms;
pressure/velocity/impedance BCs; direct/iterative solves; single/block RHS;
fields and parameter/shape adjoints. Include Burton–Miller combinations and
both harmonic conventions where the CPU supports them. A passing regular-pair
kernel cannot close a row whose singular integration, RHS, BCs or field
evaluation still executes on the host. The shared-DOF MLFMM rejection is a
specific representation gap, distinct from directional high-frequency support.

### FEM

The shared [admission table](../crates/sonium-fem/src/backend.rs)
(`cuda_supports`) currently admits these bounded cells:

| Physics / space / element | Mode and scalar | Actions |
| --- | --- | --- |
| Helmholtz H1 / Tri3 / P1 or P2 | Assembled f64 | Forward, transpose, Hermitian |
| Same | Assembled c64 | All three |
| Helmholtz H1 / tetrahedron / P1 or P2 | Assembled c64 | All three |
| Helmholtz H1 / hexahedron / Q1, Q2 or Q3 | Assembled c64 | All three |
| Helmholtz H1 / Tri3 / P1 | Element-tensor matrix-free f64/c64 | All three |
| Linear elasticity H1 / tetrahedron / P1 | Assembled f64 | All three |
| Maxwell H(curl) / Tri3 or Tet4 / lowest order | Assembled c64 | All three |
| Taylor–Hood Stokes / triangle / P2 velocity + P1 pressure | Assembled f64 | All three |
| Heat conduction H1 / Tri3 or tetrahedron / P1 or P2 | Assembled f64 | All three |
| Linear scalar magnetic potential H1 / Tri3 or tetrahedron / P1 or P2 | Assembled f64 | All three |
| Linear electrostatic potential H1 / Tri3 or tetrahedron / P1 or P2 | Assembled f64 | All three |

These rows chiefly establish **device action**, with host matrix/element tensor
preparation. They do not establish GPU quadrature or generic FEM support.
Generic partial assembly, 3D acoustic elements outside the bounded tetrahedral/hexahedral rows, H(div), higher orders and
f32/c32 are not admitted by this table. Separate application paths have broader
behavior, so this table must not be treated as a complete runtime inventory.
Some rejection messages also lag the accepted cells; generate both from the
same capability data when consolidating admission.

Important additional paths:

* [Room sweep](../crates/sonium-sim-room/src/fem_runner/gpu.rs): host stiffness,
  mass and boundary assembly followed by device basis combinations, RHS blocks,
  mixed/single-precision solves, AMG and listener evaluation. This already
  supplies a reusable design for frequency sweeps. Scalar `F32Pure` now retains
  a pooled GMRES workspace across sources/frequencies/corrections. F64Outer now
  retains quota-accounted mixed-precision scratch across sources/frequencies and
  physical-system/RHS sparse descriptors across numeric frequency updates.
  Block initialization now has native CUDA hooks. Production coverage beyond
  the tested routes, diagnostic transfers and whole-workflow quota accounting
  remain open. Evidence/diagnostic branches
  can download nodal solutions and full residual vectors; time these separately
  and replace routine residual downloads with device reductions.
* [Navier–Stokes](../crates/sonium-fem/src/physics/fluid/stokes/navier_stokes.rs):
  host Picard/Newton/BDF work invokes GPU linear solves, uploads CSR/RHS and
  downloads the solution. The cached BDF f64 CUDA branch now retains its
  context, physical CSR descriptors and Krylov buffers; identical systems also
  reuse preconditioning. Fixed-pattern value changes reuse physical CSR storage. Outer polynomial
  and Chebyshev preconditioners refresh numeric storage in place. Jacobi–Schur
  with polynomial/Chebyshev pressure actions also refreshes in place; external
  pressure AMG refreshes matching hierarchy storage and rebuilds when structure changes.
  Host AMG hierarchy construction remains. Steady nonlinear and mixed-precision calls still
  use the older per-call route; BDF assembly and history remain on the host.
  Initialization/feature failures can warn and fall back to CPU. The updated dispatcher records fallback provenance; retain that behavior and
  unify the warning/fallback policy with strict backend requests.
* [Compressible DG](../crates/sonium-fem/src/physics/fluid/compressible_flow/dg_gpu.rs):
  genuine resident state and SSPRK3 execution already exist, with RHS, limiter,
  positivity and CFL kernels. The compute loader bounds dimension to 2–3 and
  local nodes to 1–10; verify exact supported topology/order/BC combinations
  through the producer. This specialized route is absent from the generic
  admission table.
* The working-tree [Tet P1/P2 kernels](../crates/sonium-compute/src/cuda/fem_tet.rs)
  support affine complex volume actions, block/adjoint application, prescribed
  pressure lifting, fused residual/Jacobi and fixed-pattern numeric CSR assembly.
  Checked FEM H1 producers and retained P1/P2 Robin quadrature maps exist.
  Native fixed-map facet numeric updates are tested; curved P2 geometry and
  broader application solve parity remain open; these APIs do not expand the generic admission table.
  A retained [P1/P2 pressure session](../crates/sonium-fem/src/physics/acoustics/pressure/device_tet_solve.rs)
  now joins volume actions, Dirichlet lifting, numeric CSR updates, shifted AMG
  and FGMRES. Both CSR and element modes pass three-frequency native solves
  against independent FEM assembly with zero warm device allocations. Numeric
  updates require no host transfers. This closes the bounded volume-session
  integration. The P1/P2 sessions now also retain fixed Robin/Neumann facets,
  including prescribed-pressure coupling at intersecting boundaries and their
  contribution to auxiliary AMG. Resident P1/P2 boundary coefficient/source
  updates now use fixed CUDA quadrature maps. Device coefficient laws, curved
  geometry and broader production admission remain open.
* The working-tree [hex producer](../crates/sonium-fem/src/physics/acoustics/pressure/device_hex.rs)
  connects Q1/Q2/Q3 H1 geometry to CUDA tensor-product partial assembly, with
  constraints, residual/smoother fusion and a shifted-AMG/FGMRES integration test.
  The new [hex Robin producer](../crates/sonium-fem/src/physics/acoustics/pressure/device_boundary/hex.rs)
  uses host quadrature with complete geometry face traces. Curved Q2/Q3
  surface-area/trace-energy and boundary validation tests pass, as does the
  native retained volume-plus-Robin shifted-AMG solve. Fixed-geometry Q1/Q2/Q3
  Robin numeric assembly now has a CUDA quadrature-map route. Other facet forms,
  and device geometry setup remain open; the retained frequency-update route
  is described below.
  The public [CUDA pressure session](../crates/sonium-fem/src/physics/acoustics/pressure/device_hex_solve.rs)
  now owns the volume, fixed marked boundaries, low-order refined shifted AMG,
  RHS lifting and flexible Krylov state. Its optional updatable constructor also
  retains native boundary assembly maps. Repeated RHS, prescribed values and
  Robin/source samples reuse device state. Wavenumber updates reuse a canonical
  K-M volume representation; geometry and constraints still require a rebuild.
  Updates retain AMG as an explicit lagged preconditioner, with optional AMG
  replacement that preserves the rest of the session. This API does not expand
  generic admission.
* The shared [assembled solve session](../crates/sonium-fem/src/solver/device.rs)
  now connects admitted FEM systems to retained PCG/FGMRES, caller-prepared
  device preconditioners and resident CSR/RHS updates. Admission runs before
  the assembly closure; unsupported modes and scalar mismatches do not assemble.
  Host setup canonicalizes the requested forward/transpose/Hermitian matrix.
  Native production tests use elasticity with stationary symmetric AMG and
  complex Tri3 and tetrahedral P1/P2 Helmholtz with shifted auxiliary AMG,
  plus hexahedral Q1–Q3 with low-order-refined auxiliary AMG.
  Triangular P1/P2, tetrahedral P1/P2 and hexahedral Q1–Q3 Complex64 assembled
  requests now have explicit admission for all three actions; host assembly and hierarchy setup remain explicit. Taylor–Hood Stokes now passes physical
  Poiseuille, pressure-gauge and divergence checks with owned Schur preconditioning
  and repeated RHS. Maxwell now has a bounded Tri3/Tet4 PEC producer;
  broader Maxwell coverage and resident nonlinear/BDF state remain open;
  the BDF f64 linear cache is described separately above.
* [Generic assembly](../crates/sonium-fem/src/assembly/generic.rs) remains a host
  form integration/scatter route. The generic
  [matrix-free operator](../crates/sonium-fem/src/solver/matrix_free.rs) caches
  dense element matrices: it is not high-order quadrature partial assembly.
* Explicit structural CUDA helpers cover gather/scatter, hourglass and min-dt.
  Their presence does not establish a resident constitutive/contact/time loop
  for every structural model. Existing `perf/oc` proxy reports are performance
  investigations, not production CUDA coverage evidence.

## 3. Implementation sequence and acceptance gates

| Phase | Deliverables and owning area | Dependencies / completion gate |
| --- | --- | --- |
| P0: executable inventory | Compute capability records + solver/FEM/BEM workload registry. Enumerate all public CPU paths, specialized adapters and application entrypoints. Record setup/iteration/output residency separately. | Every existing CPU workflow has a row, owner, reference case and explicit CUDA state. Negative tests reject unsupported requests before expensive work. |
| P1: shared device lifecycle | Compute owns execution context, quota accounting, persistent buffers/descriptors, updates, statistics and checked failure propagation. Solvers expose a reusable solve session and device-output API. Fix feature forwarding. | Repeated solve with unchanged structure allocates/uploads no structural arrays after warmup; injected failures and quota exhaustion preserve valid state. |
| P2: complete solver routes | Connect resident PCG, (F)GMRES, blocks and IR; expose AMG, Chebyshev, block and Schur preconditioners. Add missing requested Krylov methods, direct/eigen routes and transpose/Hermitian preconditioning. | PDE references converge with correct method/assumptions; no per-iteration vector staging. Small convergence scalars are separately counted. |
| P3: broad assembled execution | FEM generic host assembly -> device CSR solve for every applicable CPU physics/space/element; BEM resident TBEM construction, H-matrix and non-directional FMM integration. | End-to-end solutions, BCs, fields and adjoints pass CPU/analytic parity. Host setup is explicitly reported. This is broad execution coverage, not native assembly. |
| P4: resident repeated workflows | FEM device numeric assembly/BC updates, frequency combinations, Newton/BDF state and material history; BEM repeated coefficient/compression updates, directional MLFMM, SLFMM route and missing basis/field derivatives. | No bulk host round trips inside frequency/Newton/time loops except explicitly requested output/checkpoints. Rebuild only when topology changes. |
| P5: remaining CPU feature parity | Cover all remaining nonlinear, transient, coupled, direct/eigen/optimization and postprocessing rows from P0. Add device form lowering for each unsupported family. | All meaningful CPU-supported rows run CUDA; no unsupported/fallback row is counted as completed. Specialized QA cells alone do not close a production row. |
| P6: FEM kernel optimization | Profile representative resident workloads; implement the ranked kernel work below. | Improve end-to-end time to the same accuracy, including setup and preconditioner costs; no numerical regression. |

P3 can proceed family by family after P1/P2. Kernel experiments can start once
that family's resident baseline passes, without waiting for unrelated physics.
P5 is a substantial program, not a promise that arbitrary user forms can be
ported automatically. Keep its backlog generated from P0 as the code grows.

P0 should produce a machine-readable registry checked in CI, not just a list
of kernel names. Each row needs its public producer/entrypoint, workload key,
CPU reference, owner, setup/action/update/output residency, capability decision,
fallback reason, and required GPU test command. Store runtime evidence with the
commit, hardware, tolerance, true residual, physical-output error, transfer
bytes and warm allocation count. Generate the human coverage report from these
rows. The family backlog below scopes the audit; it is not an exhaustive,
already-verified registry of every public function.

### FEM coverage backlog by family

| Family | First CUDA execution route | Resident work still required |
| --- | --- | --- |
| Pressure acoustics, viscothermal/equivalent fluid | Reuse assembled CSR and room-sweep basis machinery for 2D/3D elements and orders. | Native volume/facet forms, impedance/radiation/ports/Floquet constraints, source updates, velocity/intensity/listener reductions. Preserve complex phase conventions and adjoints. |
| Thermal, scalar electrostatic/magnetostatic, diffusion/transport | Shared scalar assembled operator with PCG for valid SPD cases, GMRES otherwise. | Diffusion/advection/reaction kernels, variable coefficients and nonlinear radiation/material updates. |
| Elasticity, viscoelasticity, axisymmetry, beam/shell/plate/composites | Vector/block assembled solves with appropriate nullspace/BC handling. | Element-specific strain/stress, shell/beam kinematics, stabilization, block preconditioning and element outputs. |
| Maxwell, eddy current, mixed H(div) | Oriented edge/face DOF operators, complex solves and suitable auxiliary/block preconditioners. | Piola transforms, orientation signs, tangential/normal facet terms, gauge/nullspace treatment and curl/div kernels. H1 kernels do not cover these. |
| Stokes, incompressible flow, turbulence, ALE | Retain existing GPU Schur/Jacobi/Chebyshev linear solves; introduce persistent nonlinear session. | Convective residual/Jacobian, pressure constraints, stabilization, turbulence/history updates, mesh-motion geometry and BDF combinations. |
| Compressible DG, shallow water, transport/reacting/porous/multiphase flow | Reuse DG implementation only for matching equations and discretizations; assembled route for other CPU forms. | Broaden fluxes, BCs, local state and limiters; add each model's conservation/positivity gates. |
| Nonlinear solids, plasticity, fracture/contact, explicit mechanics | Reuse device linear solves and existing explicit primitives. | Constitutive state/return mapping, residual/tangent, contact search and forces, damage, accepted-step history and adaptive time control. |
| Coupled FSI, electroacoustics, piezoelectric, thermoelastic, MHD, electrothermal, battery/electrochemistry and other multiphysics | Device block operator with field-specific solves. | Interface transfer, coupling Jacobians, field splits and consistent adjoints. Tiny lumped circuits may retain explicit CPU control, but recurring bulk coupling may not. |
| Modal, transient, sensitivity/optimization and output paths | Device operator/eigen/linear-solve adapters. | Resident history, objective/gradient reductions, JVP/VJP, checkpoint/restart and field evaluation. Validate gradients, not just forward residuals. |

Inventory subordinate modules as well as these families, including specialized
airbag, SPH, lubrication, semiconductor, radiation-transport and joint models.
Their presence does not imply that a common H1 or CSR adapter covers their
entire simulation loop. Mesh adaptivity also needs explicit cache invalidation,
state transfer and re-upload semantics.

## 4. Which FEM kernels to write or extend

First measure setup, assembly, BCs, preconditioner setup/apply, operator action,
orthogonalization/reductions, transfers, and output separately. Rank by total
time saved at equal solution accuracy, not isolated FLOP/s. The following order
is the default engineering priority; profiling may change it.

**Measured update:** the first implementations of priorities 1–4 exist, but the
coverage gates remain open. The [kernel measurements](cuda-fem-kernels.md) show
that the optimized P2 action still trails cuSPARSE on measured cases. The
[preconditioner measurements](cuda-fem-preconditioners.md) show that auxiliary
AMG restores convergence for three high-order hex cases, with up to 329 outer
iterations and weak early coarsening. The next performance work should therefore
reduce auxiliary hierarchy/cycle cost and iteration count, then measure fused
smoothers and specialized element/scatter layouts against that baseline. Do not
replace cuSPARSE globally or infer a speedup from fewer launches alone.

| Rank | Kernel work | Why / implementation outline | Required evidence |
| --- | --- | --- | --- |
| 1 | Fused low-order element action | Extend local contraction into gather -> geometry/coefficient evaluation -> mass/diffusion/elasticity action -> scatter. Start with production tetrahedral P1 and P2 acoustic cases plus Tri3 regression. Avoid materializing dense element matrices when cheaper. Keep DOFs/geometry resident; specialize by topology/order. | Compare against cached-element and CSR actions, including total solve time with the same preconditioner. Quantify register pressure, scatter contention and bytes/DOF. |
| 2 | Residual + diagonal/block smoother + vector update fusion | Existing Jacobi/Chebyshev/colored-GS components are a base. Fuse compatible SpMV/local-action residual and pointwise work, provide device scalar recurrence/reductions, batch small blocks. | AMG/PCG/FGMRES time-to-tolerance and true residual; do not weaken preconditioning to reduce kernel time. |
| 3 | Numeric CSR assembly and constraint kernels | Precompute element-to-CSR slots once. Integrate and accumulate values on GPU; apply Dirichlet lifting, Neumann/Robin/facet terms and block constraints consistently. Fuse frequency/Newmark/BDF coefficient combinations where structure is fixed. | Cold and repeated assembly times, symmetry/adjoint identity, BC correctness, memory use; no repeated host numeric assembly. |
| 4 | Quadrature partial assembly / tensor-product sum factorization | Introduce restriction, basis interpolation/gradient, quadrature physics and transpose basis/restriction. Prioritize higher-order quad/hex workloads where reuse amortizes complexity. Store quadrature data rather than dense local matrices. | Scaling with order, geometry and coefficient variation; compare assembled and partial-assembly solves including a workable coarse/preconditioning operator. |
| 5 | Incompressible-flow residual/Jacobian and block operations | Fuse element convective/viscous/body-force terms; retain Picard/Newton/BDF state. Improve Schur block action and pressure/velocity preconditioning together. | Cavity/Poiseuille and application references, divergence, nonlinear iterations, rejected-step behavior and drag/lift/adjoint parity. |
| 6 | Nonlinear constitutive and explicit force/update kernels | Batched Gauss-point return mapping, stress/tangent/internal force, contact and state updates. Extend existing explicit primitives; retain accepted/trial state separately. | Energy, plastic work, equilibrium, contact and restart parity; account for divergence and small-mesh occupancy. |
| 7 | Extend DG kernels and optimize outputs | Tune existing volume/face RHS, limiter and SSPRK kernels rather than rewriting them; add compact listener/stress/flux/objective reductions. | Conservation, positivity, CFL, outputs and total transient/sweep time; bounded output transfers. |

For low-order unstructured simplices, use specialized gather/local-action/scatter
kernels; tensor-product sum factorization is chiefly a higher-order quad/hex
opportunity. Treat the two as separate designs. Compare atomic accumulation,
coloring and node-owned/segmented reduction; no scatter scheme wins universally.
Offer deterministic accumulation when the workload requires it.

The current `sonium_batched_local_contract_*` kernels assign one output entry
to a thread and loop serially over the contraction dimension. They are a useful
correctness baseline. Specializing fixed local sizes, reusing geometry/basis
data and fusing gather/scatter are concrete opportunities, not measured wins yet.

### Next performance experiment

The checked `PersistentDeviceCsr::try_residual` path now also connects the
native residual update to retained sparse actions, including calls through
`DeviceOperator`. Warm native residual formation uses one cuSPARSE operation
and one vector kernel, with zero allocations and host transfers. CPU/CUDA tests
cover all four scalar types, rectangular matrices, numeric updates, invalid
lengths and launch-failure recovery; memcheck reports zero errors. Sixteen
retained FEM solve comparisons preserve iterations, transfers and independent
residual accuracy. This closes the forward persistent-CSR residual path only;
other wrappers and adjoint residual routes need their own inventory rows.
No elapsed-time gain is claimed from these contended diagnostic runs. See
[persistent CSR residual evidence](cuda-persistent-csr-residual-2026-10-22.json).

The checked sparse adapter paths now share this native update as well:
`PrebakedSpmv`, `PrebakedAdjointSpmv`, `DeviceCsr` and `PooledDeviceCsr`.
`AdjointDeviceCsr` forwards checked residual calls to its forward matrix.
All-scalar CPU/CUDA references cover forward and stored Hermitian actions,
invalid dimensions and retained-buffer residency. Native launch-failure recovery
and retained GMRES/FGMRES homogeneous-system regressions pass; memcheck reports
zero errors. Retained f64 residual counters show one sparse operation and one
vector kernel, with zero warm allocations/transfers. Unprepared CSR adapters
still rebuild descriptors and download validation data; use retained adapters
for iterative solves. Legacy infallible methods and the general operator default
are unchanged. This is not full adjoint-producer coverage or a measured speedup.
See [sparse adapter evidence](cuda-sparse-residual-adapters-2026-10-22.json).

AMG residual formation now uses a fused native `rhs - action` update after
SpMV for f32/f64/Complex32/Complex64, replacing separate scale and AXPY passes.
The optional hook returns false unchanged when unavailable, retaining the
existing backend fallback. Kernel tests cover all scalars, empty/tail sizes,
nonfinite arithmetic, malformed buffers, foreign streams, missing modules and
launch-failure recovery. Memcheck passes with zero errors. Independent V/W/F
cycle references and 16 complete FEM solves retain their accuracy, iterations,
transfers and zero warm allocations. CPU compute/preconditioner suites pass
651/339 tests. This establishes one fewer vector dispatch and no new storage;
elapsed-time gains remain unmeasured on an idle GPU. See
[residual fusion evidence](cuda-amg-residual-fusion-2026-10-22.json).

The benchmark now supports `SONIUM_FEM_BENCH_PROFILE=sync`: one separately
reported diagnostic solve attributes synchronized wall time to operator,
residual and preconditioner calls while preserving normal timing samples.
Sixteen native CSR/specialized solves across Tet P1/P2 and hex Q2/Q3 at two
mesh sizes preserve iteration counts and per-solve transfer counts, allocate
zero warm device buffers, and pass independent residual checks. These validate
the instrumentation. Their times are not bottleneck evidence because the long
BEM sanitizer was active on the same GPU. Repeat on an idle GPU before selecting
an optimization from the phase times. Per-level AMG attribution now also records
pre/post smoothing, residual formation, restriction, prolongation/correction and
coarse solution, excluding recursive visits from parent times. It is disabled
by default and enabled only for the separate diagnostic solve.
The per-level implementation passes independent CPU/CUDA V/W/F-cycle references
with Jacobi, L1-Jacobi and Chebyshev, plus native coarse-failure recovery. The
focused native suite passes memcheck with zero errors; 339 focused CPU
preconditioner tests pass. Sixteen complete FEM solves retain their iterations,
transfer counts and zero warm allocations with level profiling enabled.
See [per-level validation](cuda-amg-level-profile-2026-10-21.json). These runs
also shared the GPU with sanitizer work and do not establish performance gains.
See [profile validation](cuda-fem-phase-profile-2026-10-21.json) and the
[profiling method](cuda-fem-kernels.md#optional-solve-phase-attribution).

**First kernel target after resident execution:** extend the existing fused
residual and Jacobi/Chebyshev smoother path in AMG. Start with the levels that
dominate a complete retained solve. Combine residual formation with compatible
diagonal scaling and vector updates, reuse resident recurrence state, and batch
small block operations where the dependency graph permits. Do not fuse across
an AMG level dependency or change the mathematical smoother merely to reduce
launches. Existing fusion implementations make this an extension and profiling
task, not a request to start another standalone kernel.

Before writing that extension, measure whether cycle cost or iteration count
dominates. If weak coarsening or an unsuitable auxiliary operator dominates,
fix the hierarchy/preconditioner first. Retain cuSPARSE for assembled low-order
actions until a specialized implementation wins at equal accuracy. The four
kernel areas above remain the integration backlog; their rank is not evidence
that a new Tet kernel is currently the best performance investment.

Instrument retained AMG solves by level: operator action, restriction,
prolongation, smoothing, coarse solve and synchronization. AMG and standalone
scalar/block Chebyshev now share fused native updates. Broaden producer coverage
and evaluate retained device recurrence coefficients where useful.
Keep dependencies between successive sparse actions explicit. Compare identical
hierarchies and tolerances first, then evaluate hierarchy changes separately.
For small coarse levels, measure batched/fused work against the current solve
before introducing another kernel. Publish both wins and regressions across
small and large meshes; retain the existing path when it is faster.

### Tet element kernel specification

Finish integrating and benchmarking the existing **Tet P1/P2 acoustic operator
actions** under compute's checked device-operator contract. The first implementation uses node-owned accumulation and repeats
element arithmetic; compare it with element-owned alternatives on hardware. For a fixed geometry,
retain connectivity, inverse Jacobians/volume factors, coefficients and boundary
maps. Gather each element's pressure DOFs, evaluate its diffusion and mass
action, and accumulate into the global result without writing an intermediate
element vector or dense local matrix. Affine P1 tetrahedra can use closed-form
local actions; P2 and curved elements need the appropriate quadrature/basis
data. Support complex Helmholtz coefficients from the first production slice.

Keep facet impedance/Robin actions and Dirichlet lifting explicit until they
are included with matching semantics. Provide forward and Hermitian actions;
transpose must remain a distinct operation for complex nonsymmetric forms.
Benchmark one warp per element versus cooperative element batches, with a
structure-of-arrays geometry layout. Compare atomics against a precomputed
coloring or node-owned reduction. Choose from measurements, not a fixed claim
that one launch/scatter configuration is universally best.

The first acceptance case is the same 3D Helmholtz system solved through CSR
and this operator at the same tolerance, with the same shifted-Laplacian
preconditioner. Retain an assembled preconditioning operator where needed;
removing the system matrix does not remove the need for AMG/coarse operators.
Record operator time, complete solve time, iterations, bytes/DOF and peak VRAM.
This isolates the kernel benefit from changes in convergence or setup policy.

Keep cuBLAS/cuSPARSE as the baseline for generic dense/sparse operations.
The repository's [sparse matrix record](cuda-sparse-matrix.md) already reports
near-vendor SpMV/SpMM on its measured shapes; it does not justify a generic SpMV
rewrite or establish all FEM shapes are bandwidth-bound. Custom block-sparse or
matrix-free kernels need representative evidence. For missing sparse triangular
solves, evaluate vendor SpSV/SpSM before writing a general solver kernel.
[NVIDIA cuSPARSE reference](https://docs.nvidia.com/cuda/cusparse/index.html) and its
[SpSM example](https://github.com/NVIDIA/CUDALibrarySamples/tree/main/cuSPARSE/spsm_csr)
provide the library baseline. The proposed partial-assembly decomposition is
consistent with [MFEM's assembly-level design](https://mfem.org/howto/assembly_levels/)
and [performance documentation](https://mfem.org/performance/).

Use graph replay for stable launch sequences after allocations and descriptors
are prepared. Do not capture adaptive convergence decisions blindly. Consider
persistent kernels only when launch overhead is measured to dominate and global
synchronization is correct. Tiny structural decks can be faster on CPU even
with ideal residency; see [existing machine/proxy evidence](oc-gpu-machine.md).

Precision must be workload-specific: evaluate f32/c32 and mixed-precision
refinement on workstation GPUs, but retain f64/c64 references and reject accuracy
loss in ill-conditioned, high-frequency, nonlinear or adjoint cases. Tensor cores
are a later experiment for suitable dense/block contractions, not a coverage
requirement or permission to relax tolerances silently.

## 5. Validation and performance gates

Reuse existing `scripts/run_unix_c1_cuda.sh`, `run_unix_c2_cuda.sh`,
`run_unix_c4_cuda.sh`, `run_unix_c6_cuda.sh`, `run_unix_si2_cuda.sh`,
`run_unix_si3_cuda.sh`, `run_unix_sp2_cuda.sh`, `run_unix_b1_cuda.sh`,
`run_unix_b2_cuda.sh`, `run_unix_b3_cuda.sh`, `run_unix_fem1_cuda.sh`,
`run_unix_fem1_p2_cuda.sh`, `run_unix_fem1_matrix_free_cuda.sh`,
`run_unix_fem2_cuda.sh`, `run_unix_fem3_cuda.sh` and their breadth/physics
variants. Extend their production producers and checkers rather than creating
another incompatible definition of CUDA success. Historical artifacts and
unavailable-host results are not passing evidence for the current revision.

For each new workload:

1. Run CPU/analytic references, discretization convergence, BC checks and
   forward/transpose/Hermitian dot-product identities as applicable. Test
   nonsymmetric complex matrices so transpose cannot accidentally equal adjoint.
2. Compare true residual/backward error, solution and physical outputs at the
   same tolerance. For iterative changes record iterations as well as wall time;
   for transient/nonlinear changes also record accepted/rejected steps and
   conservation/history. Check derivatives against finite differences or an
   independent reference.
3. Require CUDA (`SONIUM_REQUIRE_CUDA=1` where supported), native admission and
   actual-backend evidence. Test unavailable driver/module, unsupported scalar,
   invalid geometry, OOM, device failure and cancellation. A skip is not a pass.
4. Record H2D/D2H bytes, allocations after warmup, kernel/library calls,
   synchronization, peak VRAM and host memory. Permit initial upload and requested
   output; reject hidden vector transfers in a claimed resident loop.
5. Benchmark small/medium/large problems, real/complex, P1/P2/higher order,
   single/multiple RHS, cold setup and repeated solves. Include difficult
   Helmholtz and saddle-point spectra, not only diagonal matrices. Use fixed
   meshes/accuracy, a tuned CPU baseline, and an idle GPU with hardware/toolchain
   metadata. Report medians/spread, not projected speedups as measurements.
6. Use Nsight Systems to locate transfers/launch gaps and Nsight Compute for
   bandwidth, occupancy, registers and divergence when counters are accessible.
   Otherwise report timing/traffic evidence and the profiling limitation.

Completion requires every P0 workload row to pass its production route; every
optimized row must additionally improve end-to-end time under its stated regime.
Retain CPU thresholds for small jobs and correctness references. Multi-GPU and
out-of-core execution need separate designs if required by the workload memory
envelope; never silently replace a scalable sparse/FMM path with dense storage.

## 6. First implementation slice

The retained tetrahedral pressure and generic assembled sessions already provide
the starting point. The following tasks turn those bounded implementations into
broader execution coverage. Ownership below refers to crates, not individuals.

| Task | Owner and concrete change | Completion evidence |
| --- | --- | --- |
| Coverage registry | Compute capabilities plus solver, BEM and FEM admission: reconcile generic admission with specialized Tet, hex, room and DG producers. Generate unsupported-request diagnostics from the same records. | Each public CPU route has a workload row and a required native test; distinguish action-only tests from complete solves. |
| Persistent public solve dispatch | Solver facade: accept a retained context/operator/preconditioner/session instead of creating these inside every `solve_cuda_backend` call. Preserve the convenient host-input API as an explicit boundary. | Repeated RHS and numeric updates reuse descriptors and scratch; certify the original residual; failed updates cannot publish stale solutions. |
| Stokes and Maxwell production solves | FEM shared session: connect Taylor–Hood assembly to an owned Schur preconditioner, and oriented Maxwell assembly to a suitable auxiliary/block preconditioner. | Poiseuille velocity, pressure gauge and divergence; Maxwell manufactured solution and orientation/adjoint identities. A synthetic matrix tagged as Maxwell is insufficient. |
| Nonlinear and BDF lifecycle | FEM Stokes driver/workspace: retain device CSR, velocity/pressure state, RHS and solve workspace across Picard/Newton/time steps; explicitly refresh or lag preconditioning. | Multiple accepted and rejected steps match CPU references; topology/constraint changes rebuild caches; strict CUDA cannot silently invoke CPU recovery. |
| Scalable BEM producer | BEM solver/FMM interface: expose retained H-matrix and admitted MLFMM sessions through the public API, including suitable preconditioning and device outputs. | Large scattering fixture preserves compression and accuracy; repeated RHS/frequencies stay within memory limits; no dense replacement. Directional/shared-DOF and SLFMM gaps remain separate rows. |
| BEM output and basis coverage | BEM assembly/field APIs: connect device field evaluation; add each higher-order, shared-DOF, singular and derivative combination. | Surface and receiver fields, BC residuals and derivative checks pass through the same public producer used by applications. |
| Remaining solver families | Direct/eigen/nonlinear/time/optimization crates: inventory each exposed algorithm, then add device operator/state adapters or native implementations. | Requested algorithms and scalar/action combinations run explicitly; dense offload does not count as sparse-factorization coverage. |
| Remaining FEM families | Physics producers and generic assembly: reuse assembled device solves first, then lower recurring form, facet, constitutive, coupling and output work. | Every family in section 3 has physical-output evidence and a declared setup/update boundary. Arbitrary Rust callbacks remain unsupported until explicitly lowered. |

The fixed Taylor–Hood system with changing RHS and owned Schur preconditioning
now passes the physical-output gates. The BDF f64 CUDA branch now retains its
linear solver state as well. Next move repeated numeric assembly and time-history
state to CUDA and retain the mixed-precision/steady nonlinear branches. The older
`solve_iterative_via_backend` route still uploads CSR and allocates RHS storage
per call; the f64 BDF cache bypasses it.

### Next FEM kernel experiment

**Extend residual and smoother fusion in the retained AMG cycle first.** The
initial zero-start diagonal kernel is already implemented. Measure the remaining
nonzero-start sweeps and test a fused row action
`x_next[i] = x[i] + omega * diagonal_inverse[i] * (b[i] - A*x[i])`.
Use separate input/output iterates for Jacobi; an in-place parallel update would
change the algorithm and introduce read/write hazards. Preserve complex values,
L1 scaling and constrained rows, and compare against cuSPARSE SpMV followed by
the existing update. Do not fuse across required multigrid dependencies.

The recorded zero-start change improved matched complete Tet solves by
1.05–1.12x with unchanged iterations, using optimized development builds on the
RTX PRO 6000 Blackwell. This supports further measurement, not a promised gain
from the next fusion. High-order hex cases still need hierarchy/coarse-correction
improvements: the recorded Q3/15,625 case takes 329 outer iterations. Algorithm
quality and kernel throughput must be measured separately.

After that experiment, prioritize fixed-pattern numeric assembly and boundary
updates for repeated sweeps/time steps, then high-order sum factorization where
order and size justify it. Continue Tet P1/P2 gather/action/scatter experiments
only against the retained CSR baseline: the current custom route is slower on
the recorded complete solves. The ranks in section 4 describe the original
implementation backlog; this paragraph gives the next measured-work priority.

Accept an optimization only when complete solve time improves at the same
physical accuracy, with setup, iterations, warm allocations, control transfers
and peak memory reported. Retain both implementations when their winning
size/order regimes differ. Existing evidence:
[AMG fusion](cuda-amg-zero-start-2026-10-21.json),
[high-order preconditioning](cuda-fem-preconditioners.md), and
[retained FEM session](cuda-generic-fem-session-2026-10-21.json).

## 7. Verification evidence

### Retry audit checks on 2026-10-22

The subsequent block lifecycle implementation resolves the failure recorded
below. See [block restart evidence](cuda-block-restart-retention-2026-10-22.json):
four CPU focused tests, five CUDA-enabled focused tests, 651 compute unit tests
and the full iterative crate suite pass. Native memory checking of the retained
all-scalar regression reports zero errors. Root check/Clippy and CUDA-targeted
Clippy pass with existing warnings. The native regression runs 144 solves with
72 identical-RHS repeats, each asserting zero new device allocations. This is
shared solver lifecycle evidence, not production BEM block coverage or a speedup
measurement.

This documentation refresh ran `cargo check`, then `cargo clippy`; both passed
for the default workspace configuration, with existing warnings. A separate
`cargo test -p sonium-solvers-iterative --lib block_restart_tests` failed at
compilation with E0308 in `backend/block_restart_tests/guards.rs:7` (`u64`
versus `usize`). No tests in that command executed. Logs are
`/tmp/cuda-audit-retry-check.log`, `/tmp/cuda-audit-retry-clippy.log` and
`/tmp/cuda-audit-block-tests.log`. These temporary logs are local diagnostics,
not durable native CUDA evidence. This retry changed documentation only and
did not rerun GPU benchmarks. Earlier evidence below applies to its recorded
source and workload snapshots, not automatically to all current changes.

### Audit refresh checks

The 2026-10-21 refresh ran root `cargo check` followed by `cargo clippy`:
both passed, with existing Clippy warnings. The CUDA-specific command
`cargo check -p sonium-bem --features cuda --tests` failed in
`bem_solver_cuda_hmatrix.rs`: line 123 passes four arguments to the five-argument
`validate_tbem_quadrature_for_basis` (missing the basis and correct borrowed
policy), and line 272 references absent `bem_solver_cuda_hmatrix_tests.rs`.
These failures belong to the existing working-tree implementation; this
documentation refresh does not repair them. Logs are
`/tmp/cuda-audit-check.log`, `/tmp/cuda-audit-clippy.log` and
`/tmp/cuda-audit-bem-tests.log`. Separately,
`cargo check -p sonium-fem --features cuda --lib` passed, including its CUDA
compute/solver dependencies (`/tmp/cuda-audit-fem.log`). This is compilation
evidence, not native execution. No native GPU benchmark was rerun for this refresh.

### Earlier implementation evidence

The table below preserves the earlier kernel-validation snapshot. Later bounded
runs are recorded in section 8 and their linked JSON evidence, rather than being
implicitly combined into a single full-suite run. The latest tetrahedral-session
record reports 650 compute CPU tests and 1,921 FEM CPU tests passing (one ignored),
plus three native tetrahedral tests under memcheck with zero errors. See
[retained tetrahedral evidence](cuda-tet-pressure-session-2026-10-21.json).
This audit refresh rechecked current source and feature wiring; it does not
claim to have rerun that historical GPU evidence.

Native CUDA needs device access outside this session's sandbox. Inside the
sandbox, CUDA initialization returns `CUDA_ERROR_NO_DEVICE` even though NVML
(`nvidia-smi`) sees the GPU. The approved native runs below executed kernels;
they did not skip. Hardware: RTX PRO 6000 Blackwell, driver 610.57.04,
approximately 96 GiB VRAM.

| Check | Current result |
| --- | --- |
| Root `cargo check`, then `cargo clippy` | Passed; existing warnings remain. |
| CUDA compute/FEM test Clippy | Passed with warnings. |
| Full CPU FEM library | 1,919 passed, one ignored. |
| Full CPU compute library | 646 passed. |
| Required-CUDA compute Tet/hex suite | Five passed: three native action suites and two NVRTC compile tests. |
| CUDA memcheck on the Tet/hex suite | Five passed; zero reported memory errors. |
| Required-CUDA focused FEM producers/solves | 19 passed, including native Tet P1/P2, Tet Robin and curved Q2/Q3 Robin persistent solves. |
| Complex64 CUDA AMG coarse factor/reuse/checkpoint | One native test passed. |
| Retained operator composition | Three tests passed, including native forward/adjoint/block actions. |
| End-to-end speedup and production workload breadth | Still unverified; no performance claim. |

The four curved-hex boundary failures found during the audit are resolved.
Fixtures now provide complete geometry face traces; the Robin producer checks
4/9/16-node records consistently and preserves corner-first orientation rules.
Validation remains enabled. Native testing then exposed excessive block size in
the P2 action: the default 1,024-thread block exceeded register resources.
Setup now retains CUDA occupancy-selected block sizes separately for action and
numeric assembly. Native tests include 129 RHS columns to exercise multiple
blocks. Launch-injection tests follow the backend's conservative output-invalid
contract and explicitly recover the context before reuse.

Reproduce native kernels and memory checks with GPU device access:

```bash
SONIUM_REQUIRE_CUDA=1 cargo test -p sonium-compute --features cuda --lib \
  cuda::fem_ -- --test-threads=1 --nocapture
# Run compute-sanitizer --tool memcheck --error-exitcode 99 on the test executable
# printed by Cargo, using the same filter and serial test arguments.
```

Logs: `/tmp/hex-trace-final-check.log`, `/tmp/hex-trace-final-clippy.log`,
`/tmp/hex-trace-cuda-clippy.log`, `/tmp/hex-trace-fem.log`,
`/tmp/hex-trace-compute.log`, `/tmp/fem-kernels-native-fixed.log`,
`/tmp/fem-kernels-memcheck.log`, `/tmp/fem-producers-native-fixed.log`,
`/tmp/amg-c64-native-fixed.log`, and `/tmp/operator-sum-native-fixed.log`.
Historical no-device results in section 8 describe earlier environments and
increments; they are superseded only for the specific native cases above.

## 8. Implementation progress (2026-10-21)

### Public retained CUDA solver facade (2026-10-22)

[`CudaLinearSolveSession`](../crates/sonium-solvers/src/solve/cuda_session.rs)
now exposes retained setup for the facade's five explicit CUDA strategies.
The caller retains the pool/context; the session owns canonical device CSR,
preconditioning, RHS storage and Krylov state. `solve_host` retains independent
CPU residual/backward-error certification. `solve_device` publishes certified
device output without bulk input/output transfers. Zero and previous-solution
starts are explicit; invalid input, nonconvergence and backend failures hide
published output and prevent stale warm starts. The one-shot facade delegates
to this implementation, retaining validation before context creation.

Native tests cover all five strategies across f32/f64/Complex32/Complex64:
20 combinations with changed RHS, warm starts and zero RHS. Warm host/device
solves allocate no new device storage. Matched zero-start host and device solves
have identical iterations; host transfer counts differ by exactly one RHS upload
and one solution download. Compact Krylov and validation traffic remains.
Duplicate CSR, invalid shape/values, unsupported strategies, indefinite CG+AMG,
nonconvergence and launch recovery have explicit regression gates.

The initial memcheck found foreign-stream RHS access before sparse-operator
validation. The public entry point now rejects mismatched dimensions, allocation
extent and stream before Krylov copies or reductions. Its rejection test verifies
zero dispatch/copy work. Final memcheck passes both new tests with zero errors;
all seven CUDA facade regression tests and 117 CPU facade tests pass. The public
API example compiles, and root check/clippy plus CUDA test clippy pass.

The numeric-refresh extension below now permits fixed-pattern value updates.
Canonical sparsity and solver controls remain fixed. The prepared FGMRES extension below adds caller-selected shifted/Schur actions; broader application migration remains open. AMG setup is host work; existing raw matrix/preconditioner allocation
contracts still limit whole-session quota claims. These are correctness and
residency results, not elapsed-time speedups. See
[public session evidence](cuda-public-solver-session-2026-10-22.json).

### Public fixed-pattern numeric refresh (2026-10-22)

`CudaLinearSolveSession::try_update_matrix` now retains physical CSR descriptors,
RHS and Krylov storage while replacing matrix values. It revalidates the original
SPD/HPD assumption and compares canonical CSR patterns, so duplicate/unsorted
input layouts may change without forcing a rebuild. Identity and Jacobi retain
storage. AMG constructs a new host hierarchy, refreshes device values when all
level/transfer patterns match, and rebuilds only preconditioning otherwise.
`CudaLinearUpdateStatus::preconditioner_rebuilt` reports that distinction.

A successful update hides the old solution but preserves it as a warm-start
candidate. A failed validation leaves the old numeric system usable with a zero
start. Any failure after uploads begin blocks all solves until a complete update
succeeds. Recoverable backend failures require recovery followed by a full update;
terminal synchronization failure requires a new pool/session. Failed attempts
never publish output or permit stale warm starts.

Forty matched refresh comparisons cover five strategies, four scalar types and
two value changes. These algebra fixtures allocate no device storage and download
no data during refresh; subsequent zero-start solves match fresh-session
iterations and independently known solutions. Additional tests force an AMG
hierarchy rebuild at unchanged finest sparsity, changed-pattern rejection,
duplicate layout changes, nonfinite/indefinite rejection, partial upload recovery
and terminal failure handling. Four focused native tests pass memcheck with zero
errors; all 11 CUDA facade regressions and 117 CPU facade tests pass. Root
check/clippy, CUDA test clippy and the public API example pass.

Host hierarchy construction and migration of broader physics/application loops remain open. These results do not establish a complete
production frequency sweep or elapsed-time speedup. See
[public numeric refresh evidence](cuda-public-solver-refresh-2026-10-22.json).

### Public prepared FGMRES and shifted auxiliary systems (2026-10-22)

`CudaLinearSolveSession::try_new_flexible` accepts a caller-prepared device
preconditioner on the session backend, without imposing SPD on the physical
matrix. Flexible Krylov retains preconditioned basis vectors; independent
certification still uses the original physical system. This connects shifted
auxiliary AMG and owned Schur actions to the public retained API.

Numeric updates explicitly choose `try_update_matrix_lagged` or
`try_update_matrix_with_preconditioner`. Automatic refresh rejects prepared
sessions. Validation/preparation errors hide output; partial uploads block
solving until a complete update. Backend recovery requirements still apply.
The factory must supply compatible DOFs, dimensions and a checked native action;
`require_native` cannot introspect arbitrary callback implementations for host
staging. Residency must be tested for each admitted action.

All four scalar types have variable-preconditioner algebra tests, plus explicit
lagging/replacement, strict-policy rejection and failure recovery. The assembled
P1/P2 Helmholtz fixture uses Dirichlet constraints, three frequencies and shifted
auxiliary AMG with a native complex-double coarse LU. Its reference is a known
discrete solution, not an analytic mesh-convergence study. Host assembly and
hierarchy construction remain visible setup work. Backend Krylov reduction
storage is reserved for the full restart capacity, avoiding growth when later
frequencies require longer bases.

All 13 CUDA facade regressions pass. The six FEM degree/frequency cases have
zero warm device allocations and original-system residuals below 1e-9. Both
prepared solver tests and the FEM sweep pass memcheck with zero errors. CPU
solver/FEM suites, root check/clippy and CUDA test clippy pass (existing warnings).
Validation results are recorded in
[prepared FGMRES evidence](cuda-public-prepared-fgmres-2026-10-22.json).
No elapsed-time speedup is claimed, and broader physics coverage remains open.

### Retained complex-double AMG coarse refactorization (2026-10-22)

Complex-double CUDA coarse LU now implements the AMG numeric-refresh contract.
`try_refresh_from_host_amg` checks all hierarchy patterns before mutation,
updates level/transfer/smoother data, refactors the coarsest matrix in existing
LU/pivot/workspace storage and synchronizes before publishing the new checkpoint.
The checkpoint retains its direct-coarse flag. A partial upload, factorization
or synchronization failure invalidates application and checkpoint export until
complete refresh succeeds. Complex-single coarse LU remains explicitly unsupported
by refresh; its existing regularization policy is unchanged.

The new `dense_lu_c64_refactor_column_major` provides the host-input counterpart
to resident refactorization, reusing factorization storage. Invalid host dimensions
or nonfinite input preserve old factors; a singular uploaded matrix invalidates
them until successful refactorization. No new device buffer allocation path is added.

`CudaTetPressureSession::try_refresh_preconditioner` connects this lifecycle to
retained P1/P2 CSR and element pressure solves. It builds auxiliary AMG on the
host and returns false when patterns require the existing explicit rebuild API.
Boundary-bearing setup still downloads boundary values. This is storage reuse,
not GPU hierarchy construction. The explicit rebuild API preserves its previous
failure contract.

One-level and four-level algebra fixtures match fresh AMG actions and restored
checkpoints for two numeric changes. P1/P2 tetrahedral CSR and element sessions
refresh a changed auxiliary shift with zero device allocations; each refresh
downloads only the 4-byte coarse LU status. Three coarse tests and the production
tetrahedral fixture pass memcheck with zero errors. CPU compute (651),
preconditioner (240) and FEM (1,945; one ignored) suites pass, alongside the
three existing hierarchy-refresh regressions and root check/clippy.
Validation and transfer counts are recorded in
[coarse-factor refresh evidence](cuda-amg-coarse-refresh-2026-10-22.json).
No timing speedup is claimed while the separate large BEM sanitizer is active.

### Controlled AMG hierarchy comparison (2026-10-22)

The retained FEM comparison now accepts explicit hierarchy/cycle profiles and
family selection, recording effective configuration and complete-solve operation
counts. Four profiles pass 32 operator cases (160 measured solves) with independent
residuals and zero warm allocation. Extended+i interpolation reduces iterations
and dispatch counts on three of four workloads, but slightly regresses the
smallest and can increase operator complexity. It is a candidate for isolated
scaling/timing validation, not a new universal default. Extra smoothing and
W-cycles generally add work; the largest W-cycle launches about 15 times as many
kernels. The aggregation profile initially failed coarse LU at pivot 5 on its
first case; the truncation repair below restores this bounded profile. See the
[comparison report](cuda-fem-preconditioners.md#iterations-and-work-counts) and
[raw evidence](cuda-fem-amg-variants-2026-10-22.json). Concurrent BEM sanitization
prevents a timing speedup claim.

### Aggregation coarse-support repair (2026-10-22)

The failed aggregation benchmark was traced to prolongation truncation deleting
four complete coarse columns. Smoothed prolongation now retains the row's
nonzero own-aggregate contribution and honors `max_interp_elements`. All-scalar
and physical-hierarchy regressions reproduce and prevent the observed loss of
support. The corrected profile passes all eight native operator cases with zero
warm allocations and independent residuals; small Q2/Q3 cases pass memcheck.
At Q3/2,197, iterations decrease from 212 to 176 versus the classical baseline,
operator complexity from 4.01 to 1.10, and vendor calls from 8,732 to 6,014 per
CSR solve. The smallest case increases work, so production defaults stay unchanged.
See [repair evidence](cuda-amg-aggregation-support-2026-10-22.json). Larger,
heterogeneous and boundary-rich workloads plus isolated timing remain required.

### Measured aggregation performance and retained workflow coverage (2026-10-22)

The completed original-snapshot BEM sanitizer allows a fresh release comparison.
Four medium/large Q2/Q3 cases, two operator routes and ABBA profile order give
1.32–2.29× faster warm solves for repaired aggregation at unchanged tolerance.
Q3/15,625 drops from 329 to 152 iterations and 720.65 to 314.03 ms for CSR.
All 160 measured solves retain zero allocations; 32 case-level independent host
residual checks pass. Setup costs are recorded separately. GPU compute-process
lists were empty before each run, after our builds and sanitizers completed.
See [release evidence](cuda-fem-amg-release-2026-10-22.json); this is a bounded
workload result, not universal backend or physics coverage.

The warped Q1/Q2/Q3 boundary/material sweep now validates both classical and
aggregation AMG, including numeric refresh, structural rejection and recovery.
Its 24 solves pass memcheck and original-system checks. A later first restart
exposed one missing reserved device vector; `DeviceGmresSession::try_reserve_cycle`
now includes the restart iterate. The expanded workflow retains zero warm
allocations. See [production evidence](cuda-hex-aggregation-production-2026-10-22.json).
Broader physics, scalable BEM extensions and coverage-registry closure remain open.

### High-order hex auxiliary refresh (2026-10-22)

`CudaHexPressureSession::try_refresh_preconditioner` connects retained AMG/coarse
factor refresh to partial-assembly pressure solves. The auxiliary matrix uses
physical H1 numbering and the same prescribed-DOF elimination. Hierarchy setup
remains host work. Compatible refresh retains device level, transfer, smoother
and coarse-factor buffers. Pattern changes return false before device mutation;
the existing explicit rebuild remains available. Refresh attempts hide published
output; partial updates invalidate the AMG action until complete recovery.

The warped-mesh frequency/material/Robin fixture now covers Q1, Q2 and Q3.
Each degree refreshes a changed auxiliary shift with zero device allocations and
only four status bytes downloaded. The fixture verifies unchanged hierarchy
checkpoints on a rejected structural update, injected coarse-factor failure and
successful retry. Twelve changed-system solves pass independent host-assembled
residual checks, including prescribed pressure and boundary source lifting.
Q3 still requires up to 224 iterations in this fixture: this closes a storage
reuse path, not the preconditioning performance backlog.

The native fixture passes CUDA memcheck with zero errors; 1,945 CPU FEM tests
pass (one ignored), as do root check/clippy and CUDA test clippy with existing
warnings.

See [hex refresh evidence](cuda-hex-coarse-refresh-2026-10-22.json) for checks,
transfer counts and limitations. No elapsed-time improvement is claimed.

### Persistent compressed BEM production solves

`BemSolver::prepare_cuda_hmatrix` now connects host-built H-matrix blocks to
native incident evaluation, checked device actions, spatial cluster-block or
point Jacobi preconditioning, and retained GMRES. Setup uses individual
self-block inverses; it does not assemble a global dense matrix or invoke the
CPU assembly route that builds a dense matrix solely to obtain the RHS.
Uncompressible individual blocks can still be dense. Geometry, frequency,
material and formulation remain fixed for the session.

The 1,280-panel rigid sphere with Burton–Miller uses 1,714 low-rank blocks at
ACA tolerance 1e-6. Three incident directions/amplitudes take 19/17/17 iterations
at solver tolerance 1e-9. Compressed residuals are below 5e-10; independent
uncompressed TBEM residuals are approximately 1.21e-8, below the unchanged 1e-7
gate. Both warm solves allocate zero device buffers. An additional solve with
host exact certification disabled takes 18 iterations, zero allocations,
336 tracked H2D bytes and 5,484 D2H bytes of solver/validation control traffic.
Optional exact certification explicitly downloads two vectors. Prepared
off-surface field output now has the resident route described below; the CPU
field API remains available after an explicit solution download.

The host factor ratio is 1.04765x. Earlier ACA tolerances 1e-10 and 1e-7 failed
the compression gate, rather than being counted as scalable execution. This
ratio excludes device orientation copies, indices, sparse inverses and Krylov
storage; no overall memory reduction or elapsed-time speedup is claimed.
Larger meshes, stronger compression, launch batching and forward-only storage
remain performance work.

Small rigid/impedance cases with Burton–Miller on/off and both preconditioners
match independent dense CPU surface and field references. Invalid sources,
failed exact certification and injected kernel failures unpublish output and
prevent stale warm starts; successful retry recovers the retained session.
The compute adapter tests raw-buffer action/residual reuse, dimension checks,
foreign-stream rejection and failure propagation.

Empty source lists exposed a shared GMRES zero-RHS rejection. GMRES/FGMRES now
use an absolute residual for homogeneous systems, including nonzero initial
guesses. CPU checks cover all four scalar types; the BEM CUDA test verifies the
warm-start residual against independent uncompressed TBEM and also tests a
zero initial guess. Compute CPU tests pass 651 and iterative solver tests 321.

Root check followed by Clippy passes, as do CUDA compute/BEM test Clippy and
the independent spatial-block inverse test. The native large/small production
cases and the corrected zero-source/failure case pass in focused runs. Full
CUDA memcheck completed on 2026-10-22: all three original-snapshot tests pass
with zero errors after 7,152.87 seconds. This executable predates the later
resident field-output extension. The native all-scalar homogeneous
test, adapter fault-injection test and missing-module admission test now pass
independent CUDA memcheck runs with zero errors. The latest small physical and
failure-recovery BEM cases also pass memcheck (two tests, zero errors). These
short correctness runs used separate contexts while the long run remained
active; no elapsed-time comparison is claimed. The large case is now memory-verified for that original snapshot. Tests live in
[`bem_solver_cuda_hmatrix_tests.rs`](../crates/sonium-bem/src/core/bem_solver_cuda_hmatrix_tests.rs)
and [`gmres_zero_rhs.rs`](../crates/sonium-compute/tests/gmres_zero_rhs.rs).

Preparation now also checks native FMM, incident and finite-validation module
availability before host element/hierarchy assembly. Missing incident or
validation modules return a typed unsupported error before device work, instead
of being reported as non-finite input. The missing-module regression proves
zero kernel launches, transfers and allocations, with preserved output.
Source hashes, native solve counters and completed short memchecks are recorded
in [compressed-session evidence](cuda-bem-hmatrix-session-2026-10-21.json).
That record includes the completed long large-case memcheck and its snapshot limit.

### Resident field output from compressed BEM solves

`CudaHMatrixBemSession::prepare_field_points` retains fixed evaluation points,
triangle geometry, boundary admittance and pooled field workspace.
`evaluate_pressure_device` derives velocity traces from the certified resident
pressure, invokes the existing CUDA field kernel, evaluates incident pressure
from retained source descriptors and combines both contributions on CUDA.
No new kernel or surface-vector download is required. The returned device
buffer is available only after finite validation; failed solves cannot expose
stale fields. Failed point preparation discards the previous field plan.

This route covers the session's constant discontinuous Tri3 rigid/impedance
scattering at fixed geometry, material and frequency. It uses the existing
seven-point field quadrature and rejects on-surface/nonfinite points; it does
not establish near-singular accuracy or higher-order field coverage. Empty
point sets are valid. Other BEM producers still need resident field integration.

The 1,280-panel native compressed solve also passes resident field evaluation,
matching CPU integration of its independently residual-certified surface
solution within 1e-10. Its warm field action records the same zero allocations,
zero H2D bytes and 16 D2H validation bytes. This large field case has native
correctness evidence; it has not yet completed CUDA memcheck.

The small physical suite compares against independent dense CPU solutions for
rigid/impedance boundaries, Burton–Miller on/off and both session preconditioners.
Repeated plane-wave, point-source and zero-source cases verify descriptor
updates. Warm field evaluation records zero device allocations, zero H2D bytes
and 16 D2H bytes of validation flags. Launch failure/recovery, failed solve,
invalid-point replacement and empty output are tested. Three focused CUDA tests
pass memcheck with zero errors. The CPU BEM suite passes 541 tests with three
ignored. Root check, root Clippy and CUDA BEM test Clippy pass with existing
warnings. No elapsed-time speedup is claimed while the separate large sanitizer
run is active. See [field evidence](cuda-bem-resident-fields-2026-10-21.json).

### Physical Maxwell with retained auxiliary corrections

[`maxwell::device::try_prepare_cuda`](../crates/sonium-fem/src/physics/electromagnetics/maxwell/device.rs)
connects checked H(curl) assembly and physical current loads to the shared retained
FGMRES session. The admitted producer uses affine 2D Tri3 or 3D Tet4, lowest-order edge
elements, constant coefficients, complex double and homogeneous PEC on all
exterior edges. It derives PEC edges from facet incidence rather than marker labels.
In 3D it first finds exterior triangular faces, then collects their edges;
edge incidence alone cannot distinguish tetrahedral interior and boundary edges.
The explicit mass coefficient includes its sign; the public constructor's
documentation now states that convention correctly.

The preconditioner adds edge Jacobi, a discrete-gradient nodal correction and
a vector-nodal interpolation correction. Positive auxiliary coefficients define
the scalar Poisson and vector reaction-diffusion operators. Host setup constructs
their AMG hierarchies; CUDA retains restriction/prolongation CSR descriptors,
AMG cycles, native coarse LU and work buffers. This uses the auxiliary-space
decomposition described in [hypre's AMS documentation](https://hypre.readthedocs.io/en/stable/solvers-ams.html),
but is a Sonium implementation with a narrower contract. Full AMS feature parity
and mesh-independent or resonance-robust convergence are not claimed.

Native tests independently assemble a physical current for a transverse field
plus an irrotational mode, compare against CPU LU, and integrate physical L2
error. At 56/208/456 edge DOFs, auxiliary FGMRES takes 28/86/102 iterations;
the same positive-diagonal Jacobi baseline takes 38/323/998 at the same 1e-10
tolerance, restart 60 and 2000-iteration cap. Physical L2 errors are
0.27709/0.14100/0.09432; independently recomputed residuals stay below 1e-10.
These iteration counts do not establish a wall-clock speedup.

Three RHS amplitudes per mesh verify warm reuse and CPU solution parity.
Warm solves allocate no device storage; warmed auxiliary applications allocate
nothing and transfer no host data. Exact-sequence tests verify curl(grad)=0,
the projected edge mass/nodal stiffness identity and PEC masks. Injected launch
failure preserves the caller's output, and explicit backend recovery restores
application. The new explicit-DOF complex Dirichlet helper reuses the checked
symmetric elimination implementation.

The Tet4 extension exposes `try_prepare_cuda_action` for forward, transpose and
Hermitian solves. The physical current is supplied for the requested action;
Hermitian tests independently conjugate the mass coefficient when assembling
the reference load and CPU matrix. At 98/279/604 edge DOFs, all three actions
converge in 13/45/65 iterations. Forward Jacobi baselines take 13/54/227.
Physical L2 errors decrease through 0.67483/0.49205/0.38179 on these coarse
meshes; this demonstrates refinement, not a high-accuracy production mesh.
CPU LU parity, original-system residuals below 1e-10 and warm allocation/transfer
gates pass. The middle mesh reverses global node numbering to exercise edge
signs. An independent geometric test checks every PEC edge; nonmanifold facets
are rejected. The generic admission table and capability rows now include Tet4
assembled complex-double actions while rejecting unsupported modes and orders.

Assembly, current callbacks and hierarchy setup remain CPU work. General boundary
data, higher order, PML, material variation, block RHS, numeric hierarchy refresh
and complete pool quota accounting remain open. The native physical action suite
covers Tri3 forward and all three Tet4 actions; it is not a sensitivity/gradient
validation suite. See [initial Maxwell evidence](cuda-maxwell-auxiliary-2026-10-21.json)
and [Tet4 action evidence](cuda-maxwell-tetrahedron-2026-10-21.json).

### BDF reuses CUDA linear state through the public driver

`solve_bdf_step_cached` now retains f64 CUDA linear state in `BdfWorkspace`:
context/pool, physical CSR descriptors, RHS/guess/output buffers, flexible Krylov
workspace and owned preconditioning. Exact matrix, Schur and control matches
reuse setup. Fixed-pattern value changes upload only physical values and rebuild
the selected preconditioner; changed patterns or pressure gauges rebuild the
cache. A failed setup, transfer or solve evicts it before caller recovery.

The selected preconditioners are preserved: outer polynomial Jacobi, outer
Chebyshev, and Jacobi–Schur with polynomial, Chebyshev or AMG pressure action.
The Schur action can now own its auxiliary pressure preconditioner instead of
borrowing a temporary hierarchy. BDF continues using the unsteady BtDiagB Schur
approximation, rather than a steady pressure-mass approximation. Setup reuse is
reported in the existing BDF telemetry. The old CPU fallback policy remains
explicit and records the actual backend; strict backend policy is still open.

Native production tests compare changing-traction affine probes and subsequent
BDF2 velocity, pressure and history against CPU direct solves. Warm probes
allocate zero device buffers and reuse both setup and physical matrix uploads.
On the recorded small channel, initial polynomial/Chebyshev/AMG pressure solves
take 78/69/55 iterations. The existing larger mass-dominated CUDA regression
passes in 680 iterations, below its 3,000-iteration stall gate, with zero memcheck
errors. These are correctness and allocation results, not elapsed-time speedups.
Native tests also verify configuration/pattern/gauge invalidation, cache eviction
on nonconvergence or injected transfers, truthful public CPU recovery telemetry,
and a fresh CUDA retry. CPU Stokes tests pass 103 cases (one ignored), compute
Schur tests pass nine, and three ownership tests cover scalar/block external
pressure actions. Root check, Clippy and focused CUDA test Clippy pass with
existing warnings.

This advances the linear-solve lifecycle only. BDF assembly, RHS/guess uploads,
solution downloads, true-residual host certification and accepted time-history
updates remain host work. Changed systems still rebuild preconditioning, and
mixed precision and steady Picard/Newton calls still use their prior per-call
routes. Pool-backed Krylov buffers do not establish full-session quota accounting
for sparse/preconditioner/vendor allocations. See
[BDF evidence](cuda-bdf-persistent-2026-10-21.json).

### Owned Schur preconditioning connects retained Stokes solves

`SaddlePointJacobiSchurBackend::try_from_host_csr` uploads and owns the full
matrix and embedded block actions without an intermediate device matrix. The
existing constructors no longer tie the returned action to the input matrix's
lifetime; only an external pressure preconditioner remains borrowed. Checked
construction rejects non-finite matrix/relaxation data and out-of-range velocity
blocks before device setup, including the previous spectral-estimation subtraction
underflow case.

The shared FEM session now has a production Taylor–Hood Poiseuille regression:
P2 velocity/P1 pressure, nonzero boundary lifting, pinned pressure, independently
integrated body force, and negative pressure-mass Schur approximation. Two meshes
(114 and 387 total DOFs), all three real operator actions and three RHS amplitudes
produce 18 passing solves. The largest measured relative residual is 7.9e-12,
velocity error 5.0e-11, pressure error 8.3e-9 and weak divergence error 5.0e-12.
The solver tolerance is 1e-11 because the pinned pressure error was too large
at 1e-9; the physical-output criterion was retained. Warm solves allocate zero
device buffers. RHS updates upload no host data and download a four-byte
finite-validation status. Setup and initial validation/workspace allocations
remain explicit. The production Schur action separately asserts zero warm
allocations and zero H2D/D2H traffic. The six-test retained FEM suite and final
Stokes regression pass native CUDA memcheck with zero errors; existing f32/f64
Schur CUDA regressions also pass. CPU validation passes two ownership/admission
tests, nine compute Schur tests and 103 Stokes tests (one ignored). Root check,
Clippy and focused CUDA test Clippy pass with existing warnings. See
[Stokes evidence](cuda-stokes-session-2026-10-21.json).

This connects the generic public session to production mixed assembly through
an owning preconditioner factory. That fixed-system regression alone does not
migrate the Navier–Stokes/BDF application loop (see the separate BDF cache above),
establish mesh-independent Schur convergence, or make host
assembly and pressure-mass setup native. The larger case needs 115–116 Krylov
iterations versus 32 on the smaller case. Preconditioning quality, nonlinear
numeric updates and full-session quota accounting remain open.

### Shared retained solves for host-assembled FEM systems

`solver::device::CudaFemSolveSession::try_assemble` joins the existing admission
contract, checked generic assembly/BCs, persistent CSR, device preconditioners
and retained PCG/FGMRES sessions. Unsupported workloads, scalar mismatches and
non-assembled modes are rejected before the assembly/preconditioner callbacks.
Malformed or non-finite systems fail before upload. Checked space and boundary
errors propagate directly. Host setup canonicalizes the requested matrix action;
the RHS is supplied for that action. No generic admission rows are widened.

The session retains its RHS and only publishes independently certified device
solutions. Device numeric updates preserve the CSR pattern and explicitly lag
the preconditioner; callers can replace a prepared preconditioner separately.
Successful updates retain the previous solution as a warm-start candidate while
unpublishing it. Failed matrix/RHS copies independently block solving until a
complete replacement succeeds. Failed or unconverged solves remove publication
and warm-start eligibility. Pattern, topology or constraint-layout changes need
a new session. PCG requires an explicit SPD/HPD assumption and rejects Stokes;
FGMRES accepts appropriate changing/auxiliary device preconditioners.

Native production fixtures cover 192-DOF P1 tetrahedral elasticity with
stationary symmetric AMG (forward/transpose/Hermitian) and 49/169-DOF Tri3
P1/P2 complex Helmholtz with shifted auxiliary AMG (forward/Hermitian).
Three numeric scaling/RHS changes per case preserve zero warm allocations.
Updates upload no host data and download two four-byte finite-validation flags.
Independent host actions verify residuals below 1.1e-9 and solution parity.
A separate nonsymmetric complex algebra fixture distinguishes all three actions;
its Maxwell admission metadata does not establish production Maxwell coverage.
Negative tests cover admission before assembly, invalid systems, failed copies,
recovery, missing warm starts and nonconvergence. See
[shared FEM session evidence](cuda-generic-fem-session-2026-10-21.json).

This is host setup followed by retained device solves, not GPU generic form
integration. The API does not automatically select or validate the mathematical
suitability of arbitrary preconditioners. Block RHS, native form/BC lowering,
application migration, broader Maxwell and nonlinear Stokes integration, and reconciliation
with specialized Tet/hex admission remain open. Existing matrix/preconditioner
allocation policies remain; full-session pool quota coverage is not claimed.

### Zero-start AMG smoothing avoids redundant sparse actions

Backend AMG now tracks whether each correction starts from zero. Its first
Jacobi or L1-Jacobi sweep overwrites the correction using a native diagonal
scaling kernel for f32/f64/c32/c64, avoiding both initialization and A*0.
The native hook does not read old output, allocate buffers or stage vectors.
Other backends use diagonal CSR plus scaling. W/F-cycle second visits preserve
the existing correction; zero pre-smoothing and other smoothers explicitly
initialize fresh corrections. Direct coarse solves still overwrite their output.

The independent cycle reference covers Jacobi/L1-Jacobi, V/W/F cycles, zero/one/
three pre-sweeps and repeated two-column RHS on CPU and CUDA. Native regression
checks also preserve PCG Hermiticity/positive definiteness, checkpoint replay,
coarse LU, and retained Tet/hex frequency and boundary solves. Fourteen native GPU tests and one CPU fallback test pass in the
memcheck runs, with zero errors. The full preconditioner suite has
337 passing tests and one ignored doc test. Root check/Clippy and CUDA-enabled
compute/preconditioner Clippy pass with existing warnings.

The same optimized-dev retained Tet benchmark records a further 1.05–1.12x
complete-solve speedup relative to the P2 geometry baseline: about 1.07–1.12x
for CSR and 1.05–1.10x for element mode across P1/P2 and subdivisions 4/8.
All 24 frequency/operator/mesh cases retain identical iteration counts and zero
warm device allocations. CPU-reference residual and solution checks pass.
These are sequential before/after measurements, not alternating binary trials
or an end-to-end simulation benchmark. CSR remains the faster default. AMG
hierarchy quality, setup/update residency and block-cycle batching remain open.
See [zero-start AMG evidence](cuda-amg-zero-start-2026-10-21.json).

### P2 geometry precomputation improves retained element solves

P2 setup now stores the ten symmetric barycentric-gradient dot products instead
of twelve gradient components. Together with volume, this reduces geometry from
13 to 11 doubles per element and removes repeated dot products from both the
node-owned action and numeric CSR assembly. P1 is unchanged. Geometry preparation
remains explicit one-time CPU setup; frequency updates reuse the device data.

Matched complete pressure solves use the same shifted AMG, zero initial guess,
three frequencies and five warm timing samples per frequency. On the RTX PRO
6000 Blackwell, P2 element-mode medians improve from 74.48–76.71 ms to
57.96–59.72 ms at 729 DOFs, and from 223.54–227.80 ms to 169.31–172.62 ms at
4913 DOFs: approximately 1.28–1.32x. Iteration counts are unchanged. Current CSR
medians remain faster at 40.25–41.54 ms and 118.58–121.00 ms respectively, so
this result does not justify changing the default operator.

Two alternatives (row-dot caching with unrolling, and arithmetic mass/edge
lookup alone) failed to improve complete solves and were reverted. Measurements
use the repository's optimized dev profile, not a release-build claim. Setup
shares driver caches and runs CSR first; no cold-setup speedup is inferred.
Native compute and retained FEM regressions cover actions, adjoints, lifting,
numeric assembly, boundary updates and recovery. See the
[P2 geometry evidence](cuda-tet-p2-gram-2026-10-21.json) for raw timing samples,
source hashes and validation output. Full CUDA coverage remains open.

### Element sessions omit unused physical CSR storage

Tetrahedral sessions now select a private physical-volume representation at
construction. CSR mode retains numeric assembly maps, value scratch and the
sparse operator. Element mode retains only the element operator and performs
frequency updates by changing coefficient scales, with no device launch,
allocation or transfer. Both modes still use assembled auxiliary AMG and
retained boundary operators. Lifting and Hermitian volume checks use the same
geometry regardless of physical action mode.

Six native tetrahedral tests pass under memcheck with zero errors, including
CSR-copy and element-launch failure recovery, boundary updates and repeated
solves. For the isolated 729-DOF P2 physical operator, tracked device allocation
bytes are 88,780 for element mode versus 1,062,579 for CSR plus assembly. This
excludes AMG, boundaries, Krylov storage and untracked driver/vendor memory.

The matched public-session benchmark passes again for subdivisions 4 and 8,
P1/P2 and three frequencies. All iteration counts remain unchanged. Combined
frequency/boundary updates in element mode drop from four launches to three;
observed update times fall from 0.25–0.47 ms to 0.03–0.15 ms. These update times
are single synchronized observations. Warm solve timings are effectively
unchanged, and CSR remains faster on these workloads. Root check/clippy and
CUDA FEM/example Clippy pass with existing warnings. See
[storage and benchmark evidence](cuda-tet-element-storage-2026-10-21.json).

### Complete retained tetrahedral solve comparison

The `cuda_tet_session_compare` example measures public P1/P2 sessions with
Dirichlet and changing Robin/source data, three frequencies, the same shifted
AMG/FGMRES settings and zero initial guesses. Each frequency has one warmup and
five synchronized timed solves. Independent CPU residuals and solution errors
are checked after timing; numeric updates must transfer no host data and warm
solves must allocate no device buffers. The example requires CUDA explicitly.

On RTX PRO 6000 Blackwell, with the repository's opt-level=2 development profile:

| Subdivisions | Order / DOFs | CSR median range | Element median range | Element / CSR |
| --- | --- | --- | --- | --- |
| 4 | P1 / 125 | 15.62–16.45 ms | 18.97–19.96 ms | 1.21× |
| 4 | P2 / 729 | 40.72–42.00 ms | 74.64–76.83 ms | 1.83× |
| 8 | P1 / 729 | 29.22–29.25 ms | 46.31–46.33 ms | 1.58× |
| 8 | P2 / 4,913 | 119.42–121.84 ms | 223.59–227.83 ms | 1.87× |

Iteration counts match between modes for every frequency. All residual,
allocation and update-residency assertions pass. Keep CSR as the default on
these workloads. Element mode still retains and updates an unused physical CSR
representation; removing it is a separate memory/setup optimization, not an
explanation for faster CSR Krylov iterations. AMG quality and element-action
cost remain performance work. Root check/clippy and CUDA example Clippy pass.
See [raw timings and comparisons](cuda-tet-session-solves-2026-10-21.json).
Setup timings share driver/context caches; host reference construction and
sample uploads are excluded from warm solve measurements. Peak VRAM was not
measured. These results do not generalize to all FEM meshes or physics.

### Resident tetrahedral boundary numeric updates

`prepare_tet_robin_quadratures` builds fixed P1/P2 action, prescribed-pressure
coupling and load maps. P1 preserves the existing centroid-coefficient/exact
face-mass contract; P2 uses checked degree-four triangular quadrature semantics.
Structural slots remain present when alpha is zero. The shared public
`RobinQuadrature` also retains the existing `HexRobinQuadrature` compatibility
name. CPU parity checks compare both orders against their established builders.

Tetrahedral sessions retain these maps and expose `boundary_samples` and
`try_update_boundary`. Existing CUDA quadrature kernels assemble action, coupling
and load values from resident samples, followed by device copies into retained
operators. No host data transfer or device allocation occurs in numeric updates.
AMG remains lagged; explicit rebuild downloads current boundary values so it
includes the updated physical boundary instead of the initial coefficients.
Failed updates unpublish the solution and block solves until a complete retry;
volume-frequency updates cannot clear that boundary failure state.

P1 zero/nonzero-alpha cases and P2 changing impedance/source cases pass native
CSR and element solves. P2 transitions from zero alpha to nonzero values without
rebuilding sparsity, with 34/24/22 iterations and zero warm allocations. Injected
copy failure/recovery passes. All five native tetrahedral tests pass CUDA
memcheck with zero errors; 1,923 CPU FEM tests pass (one ignored). Root
check/clippy and CUDA FEM Clippy pass with existing warnings. See
[boundary-update evidence](cuda-tet-boundary-updates-2026-10-21.json).
Geometry preparation, arbitrary callback sampling and AMG setup remain host
work. No elapsed-time speedup or whole-workflow GPU-residency claim is made.

### P2 tetrahedral boundary connection

`prepare_tet_p2_robin` promotes affine boundary faces and explicitly remaps
promoted nodes into the H1 volume numbering, checking shared-edge consistency
and bijectivity. Checked P2 surface quadrature then supplies the action, source
and prescribed-pressure coupling. `CudaTetP2PressureSession::try_new_with_boundaries`
retains these with the same operator composition and shifted AMG as P1.

Independent unit-cube surface polynomial integrals validate the trace basis and
numbering, including constrained edge coupling. Native CSR and element solves
pass three frequencies and an AMG rebuild, with 25/24/24 iterations and zero
warm device allocations. Frequency updates transfer no host data. Five native
tetrahedral tests pass under memcheck with zero errors; 1,922 CPU FEM tests pass
(one ignored). Root check/clippy and CUDA FEM Clippy pass with existing warnings.
See [P2 boundary evidence](cuda-tet-p2-robin-session-2026-10-21.json).
Boundary coefficients remain fixed host setup; resident numeric updates, curved
geometry, block production solves and generic admission remain open. This is
execution/residency evidence, not an elapsed-time speedup claim.

### Fixed P1 tetrahedral Robin/Neumann session integration

`CudaTetP1PressureSession::try_new_with_boundaries` connects existing checked
boundary preparation to both CSR and element-action solves. Setup retains the
boundary action, natural load, prescribed-pressure coupling and composition
scratch. Shifted AMG includes the boundary matrix, including after explicit
preconditioner rebuilds. Repeated solves perform lifting and boundary actions
on CUDA; volume-frequency updates preserve fixed boundary coefficients.

Native manufactured-solution checks cover Robin and zero-alpha Neumann loads,
intersecting prescribed faces, both operator modes and three nearby frequencies.
All repeated solves allocate zero device buffers; numeric updates allocate
nothing and transfer no host data. Missing/duplicate markers are rejected.
Four native tetrahedral tests pass under CUDA memcheck with zero errors. The
CPU FEM suite passes 1,921 tests (one ignored); root check/clippy and CUDA FEM
Clippy pass with existing warnings. See [evidence](cuda-tet-robin-session-2026-10-21.json).
This closes fixed P1 facet integration only: P2 facets and resident boundary
coefficient updates remain open, and no complete-solve speedup is claimed.

### CUDA feature forwarding through public packages

BEM's `cuda` feature now enables the solver facade's `cuda` feature. The facade
also forwards CUDA to its direct and eigen crates, alongside compute, iterative
and preconditioners. Package-scoped `cargo check` with `--no-default-features
--features cuda` passes separately for BEM and the facade. Resolved `cargo tree`
features for BEM confirm all four solver backends receive CUDA, without enabling
features on other workspace packages. Root `cargo check` followed by `cargo
clippy`, CUDA library Clippy for BEM/facade, and library tests pass: 541 BEM tests
(three ignored) and 117 facade tests. Existing warnings remain. No new GPU
runtime or performance claim follows from this build-wiring change; dense
direct/eigen residency and broader algorithm coverage remain separate work.

The requested implementation scope is persistent solves with appropriate
preconditioners, followed by kernel priorities 1–4 above. It remains open.

Historical first increment (superseded by the following increments):

The first change adds `backend::DeviceGmresSession` in the iterative crate.
It retains two pooled solution buffers and Krylov workspace, accepts resident
operators/preconditioners/RHS, supports a certified previous-solution guess,
and publishes only converged results. CPU-backend contract tests exercise
Jacobi, repeated RHS, changed RHS, nonconvergence, cancellation, and partial
allocation cleanup. This is shared lifecycle infrastructure, not proof of a
fully resident CUDA workflow: scalar restart storage, Krylov control transfers,
operator/preconditioner quota ownership, PCG/AMG and production FEM integration
still need completion. No GPU runtime or performance claim is made.

Next: retain restart storage, connect this lifecycle to production solve callers,
add PCG and suitable AMG/shifted-Laplacian preconditioning, then implement and
validate Tet P1/P2 actions, fused smoothers, numeric assembly/BC updates and
high-order partial assembly. CUDA execution and profiling remain required gates.

### Persistent-session integration increment

Scalar GMRES now retains pooled restart storage across solves, includes it in
workspace metrics, releases it on clear, and avoids allocating/copying a restart
at the iteration limit. A regression test compares reused and fresh sessions
with changed RHS to catch stale-initial-guess contamination.

The public CUDA facade now invokes `DeviceGmresSession` and admits explicit
`GmresJacobi` alongside `Gmres`. RHS and session vectors use pooled allocation.
The one-shot facade still creates its context/operator per call; this is not yet
a persistent production FEM frequency/time workflow. Matrix/preconditioner
allocation accounting is also still incomplete. The low-level public session
can be retained by those producers in the next integration step.

Both changed crates' CPU test suites pass. CUDA-enabled compilation passes.
Native validation is separately required; an explicit CUDA Jacobi facade test
fails rather than skips when `SONIUM_REQUIRE_CUDA=1`.

### Production room sweep and PCG warm starts

The scalar `F32Pure` room sweep now owns one pool around its existing CUDA
backend and retains a `GmresWorkspace` across sources, frequencies and residual
correction solves. Correction iterates remain internal; FP64 promoted residual
and physical-output gates still certify the final solution. The pool accounts
for Krylov scratch only; whole-workflow quota closure, block RHS and F64Outer
refinement residency remain open. Existing preconditioner reuse policy is
preserved. A four-frequency scalar FP32 production parity regression was added.

PCG now checks a supplied initial guess's residual before forming its search
direction. Exact and near-converged guesses return successfully with zero
iterations rather than breaking down. This is a prerequisite for persistent
PCG integration; it does not close PCG true-residual certification, zero-RHS
semantics or positive-definiteness admission.

### Persistent PCG and public CUDA solver integration

`DevicePcgSession` now retains pooled PCG, candidate and certification vectors.
It requires explicit caller certification of the SPD/HPD operator and stationary
positive-definite preconditioner, recomputes the original-system residual on the
device, and publishes only certified solutions. Zero RHS overwrites scratch
(including non-finite scratch) with the zero RHS. Tests cover all four scalar
types, complex Hermitian systems, repeated and changed RHS, zero RHS,
nonconvergence, quota cleanup, and an intentionally changing operator proving
that recurrence convergence cannot bypass certification. A retained multilevel
AMG hierarchy solves repeated Poisson RHS in the CPU backend; the matching CUDA
runtime test is present but skipped locally because CUDA reports no device.

The explicit CUDA facade now admits `Cg` (identity preconditioner) and
`GmresAmg`, in addition to `Gmres` and `GmresJacobi`, through persistent CSR
descriptors and sessions. `solve_with_backend_and_assumption` carries large
SPD/HPD assumptions through native and explicit-fallback dispatch; validation
precedes CUDA initialization for SPD-only strategies. The one-shot facade still
creates a session per call. `CgAmg` is not newly admitted by this change; the
low-level PCG session accepts caller-certified AMG configurations. Host AMG
setup, whole-workflow quotas, further producer integration, and native GPU
performance validation remain open. No kernel priority 1–4 is closed yet.

### Kernel priority 1: affine Tet P1 volume action

Added `fem_tet::TetP1AcousticC64` and CUDA `CudaTetP1C64`. The checked host
constructor prepares physical basis gradients, volumes and stable node-to-element
adjacency. The fused CUDA kernel gathers nodal pressure, evaluates isotropic
complex mass/diffusion terms, and accumulates directly into each output node.
It retains no dense element tensor or intermediate element vector. Forward,
transpose, Hermitian and column-major block actions share one kernel. The first
variant uses node-owned accumulation; element/warp and atomic variants still
need comparison on hardware. No speedup is asserted.

Independent host tests compare against degree-2 mass quadrature and analytic
physical gradients on two shared-node tetrahedra, including reversed orientation,
affine energy and the constant-field diffusion nullspace. NVRTC compilation was
executed successfully without a GPU. Native tests cover all actions, blocks,
graph replay, warm allocation/transfer accounting, launch-failure preservation
and malformed raw storage, but are skipped locally with `CUDA_ERROR_NO_DEVICE`.
The CUDA wrapper validates stream identity and actual byte storage, accounts
array allocations through the pool, and preserves typed device-loss errors.

Priority 1 remains incomplete: FEM mesh/BC producer integration, physical solve
parity with an assembled preconditioner, Tet P2 and curved geometry, additional
scalar policies, sanitizer runs and measured time-to-tolerance are pending.
Priorities 2–4 remain open. The existing generic FEM admission table is unchanged;
a kernel primitive is not reported as a newly supported complete FEM workload.


### FEM mesh and space integration for Tet P1

`pressure::device_tet::prepare_tet_p1_helmholtz` now prepares the volume operator
from FEM Mesh, H1Space and HelmholtzForm. It rejects unsupported dimensions,
orders, element geometry and DOF layouts before upload. CPU tests compare all
three actions against generic FEM assembly on a skewed 3D tetrahedral mesh for
real and lossy complex coefficients, and reject malformed connectivity and
non-finite coefficients. A CUDA-gated test retains one operator, Jacobi
preconditioner and GMRES session across three manufactured RHS. This small
lossy natural-Neumann case verifies the producer/session connection when run
on hardware; Jacobi is not a scalable indefinite Helmholtz preconditioning claim.
Boundary operators, Tet P2, shifted-Laplacian production integration and kernel
priorities 2–4 remain open.

Validation for this increment: CUDA-enabled FEM library suite reported 1912
passed and one ignored; hardware-gated cases are included in that count and
are not runtime CUDA evidence. The new native solve explicitly skipped with
`CUDA_ERROR_NO_DEVICE`. Root `cargo check` and `cargo clippy` passed with
existing warnings. Logs: `/tmp/fem-device-tet-full.log`,
`/tmp/fem-device-tet-native.log`, `/tmp/fem-device-tet-check.log`, and
`/tmp/fem-device-tet-clippy.log`.


### Flexible scalar Krylov correction

The checked scalar FGMRES route now saves each preconditioned Arnoldi vector
in pooled storage and forms the solution update from that basis. Stationary
GMRES retains its lower-memory path. `DeviceGmresSession::try_new_flexible`
provides retained solutions/workspaces with the same publication and warm-start
contracts. This removes the stationary-only limitation for scalar FGMRES;
block FGMRES is unchanged. Tests use changing preconditioner actions on
nonsymmetric real/complex systems, restart lengths one and three, repeated RHS,
retained workspace capacity, warm starts, cancellation and pool cleanup.
Native GPU testing and real Helmholtz inner-preconditioner integration remain
open; this correction alone does not close kernel priorities 1–4.

Validation: compute library suite 637 passed; CUDA-enabled iterative library
suite 322 passed before the added quota regression. The final targeted suite
passed both CPU regressions (all scalar types/restarts/session lifecycle and
partial-allocation cleanup); its native CUDA case explicitly skipped with
`CUDA_ERROR_NO_DEVICE`. Logs: `/tmp/fgmres-compute-tests.log`,
`/tmp/fgmres-iterative-tests.log`, and `/tmp/fgmres-final-tests.log`.


### Kernel priority 2: fused Tet P1 residual and Jacobi update

The Tet P1 CUDA action now supports a fused residual `b - A*x` and simultaneous
weighted-Jacobi step `x + omega * D^-1 * (b - A*x)`. The operator overrides the
checked residual contract, so Krylov true-residual certification uses the fused
launch automatically. The smoother accepts resident RHS blocks and one shared
nodal inverse diagonal. Inputs stay immutable while the output uses separate
storage; no inter-thread update ordering is assumed. Shape, stream, actual raw
storage and damping validation precede launch. Diagonal extraction from affine
geometry avoids constructing dense element or global matrices and is compared
with FEM's assembled diagonal.

CUDA regression coverage includes numerical comparison to the host action,
multiple RHS, damping values, one-launch/no-transfer/no-allocation counters,
invalid damping and short or malformed diagonal storage with preserved output.
NVRTC compilation passes without a GPU. Native numerical/sanitizer/performance
evidence remains required. Priority 2 remains open for multilevel smoother
integration, broader element/scalar support and measured time-to-tolerance.
The production room scalar preconditioners inspected here use fixed point,
polynomial, Chebyshev or AMG actions; their stationary GMRES convention is
preserved rather than allocating a flexible basis without a variable action.

Validation: CPU compute library 637 passed; FEM library 1906 passed and one
ignored. Isolated serial Tet CUDA tests compiled the kernel with NVRTC and
explicitly skipped native execution with `CUDA_ERROR_NO_DEVICE`. An attempted
full CUDA-enabled compute run stalled in several CUDA initialization tests
(including pre-existing explicit kernels) and was interrupted; it is not
passing evidence. Final root check and Clippy passed with existing warnings.
Logs: `/tmp/tet-fused-compute-cpu.log`, `/tmp/tet-fused-fem-full.log`,
`/tmp/tet-fused-cuda-isolated.log`, `/tmp/tet-fused-final-check.log`, and
`/tmp/tet-fused-final-clippy.log`.


### Kernel priority 3: Tet P1 Dirichlet elimination and lifting

Host preparation now accepts a validated nodal Dirichlet set, uploaded once
with the Tet P1 operator. The action implements `P*A*P + Q` (P selects free
nodes, Q prescribed nodes), preserving transpose/Hermitian semantics and unit
constrained diagonals. One CUDA launch lifts each RHS as
`P*(b - A*Q*g) + Q*g`; prescribed values can vary across RHS columns and repeated
solves without rebuilding geometry or the mask. Residual and weighted-Jacobi
modes use the same constrained action. No masked vector or global matrix is
materialized. Geometry/mask preparation remains host work.

FEM tests compare the host action and lifted RHS against existing complex
Dirichlet elimination, including changing nonzero data and all three matrix
actions. Edge cases cover no constraints, every node constrained, duplicates,
invalid indices and invalid vector lengths. Native test coverage exercises
empty/partial/full masks, block lifting, single-launch/traffic accounting, and
a persistent FEM solve with changing prescribed pressure. GPU execution,
facet Robin/Neumann assembly, periodic/block constraints, numeric CSR assembly,
P2 and higher-order kernels remain open.

Validation for Dirichlet integration: compute CPU suite 638 passed; FEM CPU
suite 1907 passed and one ignored; focused CPU elimination parity passed.
Isolated CUDA kernel compilation succeeded with NVRTC. Native compute and
constrained FEM solve cases compiled and explicitly skipped with
`CUDA_ERROR_NO_DEVICE`. Root check/Clippy and compute CUDA Clippy passed with
existing warnings. Logs: `/tmp/tet-dirichlet-compute.log`,
`/tmp/tet-dirichlet-fem.log`, `/tmp/tet-dirichlet-cuda.log`,
`/tmp/tet-dirichlet-native-fem.log`, `/tmp/tet-dirichlet-check.log`, and
`/tmp/tet-dirichlet-clippy.log`.


### Kernel priority 1: affine Tet P2 and shared checked execution

`TetP2AcousticC64` and `CudaTetP2C64` now share the checked tetrahedral operator
implementation with P1, specialized by local DOF count. P2 connectivity follows
four vertices then edges (01, 02, 03, 12, 13, 23). Construction validates midpoint
coordinates and rejects curved geometry. Exact barycentric moments produce
stiffness actions; a shared reference mass table avoids per-element dense
matrices. The node-owned CUDA variant fuses gathering, local action and
accumulation; its arithmetic/register tradeoff still requires GPU measurement.
P2 inherits block actions, transpose/Hermitian actions, residual/Jacobi fusion,
Dirichlet elimination and lifting with persistent geometry and mask storage.

`prepare_tet_p2_helmholtz` connects FEM H1 global vertex/edge numbering to this
operator. Independent FEM quadrature and boundary-elimination tests pass for
skew geometry, shared edges, real/complex coefficients, all actions, diagonals
and nonzero constraints. A native P2 persistent-solve test uses assembled RHS
references and retained Jacobi/GMRES state. Curved P2 geometry, broader scalar
policies, production shifted-Laplacian/multigrid integration and native hardware
performance evidence remain open; priority 1 is not yet closed.

Validation for the P2 increment: compute CPU suite 639 passed; FEM CPU suite
1909 passed and one ignored. Focused FEM tests cover P2 volume/diagonal/action
parity and nonzero boundary lifting. Both P1 and P2 CUDA specializations compile
with NVRTC; native action tests explicitly skip on this no-device host. Logs:
`/tmp/tet-p2-compute.log`, `/tmp/tet-p2-fem.log`, `/tmp/tet-p2-cuda.log`,
`/tmp/tet-p2-final-parity.log`, and `/tmp/tet-p2-native-fem.log`.
The final reversed-orientation P2 parity test passed after midpoint validation
was tightened to scale with coordinates. Final root check and Clippy passed
with existing warnings; the native P2 persistent FEM test compiled and explicitly
skipped with no CUDA device. No measured GPU speedup is claimed.


### Kernel priority 3: fixed-pattern numeric CSR assembly

`TetCsrPattern` builds sorted CSR topology and deterministic element-contribution
lists for P1/P2 tetrahedra once on the host. `CudaTetCsrAssembly` uploads that
mapping with geometry and material coefficients. One thread owns each CSR entry
and sums its contributions without atomics. Complex diffusion/mass scale factors
support repeated numeric rebuilds and shifted-Laplacian matrices while preserving
Dirichlet unit diagonals and structural zeros. All device mapping arrays are
pool-owned. The current mapping is also retained on the host, and memory cost
must be included in production sizing.

`PersistentDeviceCsr::try_copy_values_from_device` connects assembled values to
retained sparse descriptors without a host round trip. This first handoff uses
an explicit device-to-device nnz copy; it does not claim zero-copy descriptor
updates. Native tests cover P1/P2, constraint masks, changed complex scales,
CSR action parity and warm single-launch/no-host-transfer accounting. CPU tests
compare entries with FEM quadrature and boundary elimination. Actual multilevel
hierarchy numeric updates and setup reuse remain separate integration work.
Facet assembly, general constraints, higher-order partial assembly, GPU
sanitizers and measured time-to-tolerance remain open.

Validation: 640 compute CPU tests passed; 1909 FEM CPU tests passed and one was
ignored. Final targeted FEM parity includes complex coefficient scaling and
constrained CSR entries. Isolated CUDA tests compiled both specializations with
NVRTC; native cases explicitly skipped without a CUDA device. Root check and
Clippy passed with existing warnings. Logs: `/tmp/tet-csr-compute.log`,
`/tmp/tet-csr-fem.log`, `/tmp/tet-csr-final-parity.log`,
`/tmp/tet-csr-cuda.log`, `/tmp/tet-csr-final-check.log`, and
`/tmp/tet-csr-final-clippy.log`.

### Kernel priority 4: tensor-product partial assembly (first increment)

`HexPartialC64` represents lexicographic hexahedral connectivity, one-dimensional
basis/derivative tables and seven complex symmetric coefficients per quadrature
point (mass and transformed diffusion metric). It supports 2–8 basis nodes and
1–8 quadrature points per axis without storing dense element matrices.
`CudaHexPartialC64` retains those arrays and caller-sized block-RHS scratch in
the memory pool. One CUDA block per element/RHS performs the forward tensor
contractions, quadrature action and transpose contractions; a second kernel
reduces incident element results deterministically into global DOFs. Forward,
transpose and Hermitian actions share the operator. Warm execution requires two
launches, with no bulk host transfer or allocation in the implementation.

This is an execution primitive, not a completed production FEM route. It still
needs FEM Q2/Q3 ordering/geometry producers, boundary operators, suitable
preconditioners and persistent solve integration. Native sanitizer, graph replay,
profiling, occupancy and time-to-tolerance measurements remain open. Static
shared memory is 48 KiB per block; specialization and smaller-order scratch
sizing are profiling candidates, not demonstrated improvements.

Independent CPU direct-quadrature tests cover shared-face elements, complex
anisotropic coefficients, unequal basis/quadrature sizes, all three actions,
constant diffusion nullspace, and invalid input. NVRTC compilation passed.
Native CUDA tests cover scalar/block RHS, all actions, the largest supported
tensor size, rejected capacity and warm launch/allocation/transfer accounting;
they explicitly skipped on this host with `CUDA_ERROR_NO_DEVICE`. A skipped
runtime test is not GPU parity evidence. Focused logs: `/tmp/hex-host-tests.log`
and `/tmp/hex-cuda-tests.log`.

The compute CPU suite passed all 643 tests. Root `cargo check` and `cargo clippy`
passed; CUDA compilation and isolated NVRTC/runtime-gated tests passed their
available checks. Existing workspace lint warnings remain. No GPU speedup is
claimed. Validation logs: `/tmp/hex-compute-tests.log`,
`/tmp/hex-final-check.log`, `/tmp/hex-final-clippy.log`,
`/tmp/hex-final-cuda-clippy.log`, and `/tmp/hex-final-cuda-tests.log`.

### Hex FEM producer and retained solve integration

`physics/acoustics/pressure/device_hex.rs::prepare_hex_helmholtz` now lowers
FEM's Q1/Q2/Q3 H1 nodal basis to the lexicographic partial-assembly contract.
It derives the local permutation from FEM's canonical basis and uses H1's
global DOF mapping, preserving shared edge/face orientation. Trilinear and
explicit Q2/Q3 geometry use pointwise Jacobians; warped and curved elements
retain their quadrature metrics. Geometry/solution order combinations remain
subject to H1's existing admission rules. Quadrature order is explicit (1–8
points per axis); curved geometry still requires convergence studies.

The producer rejects malformed mesh connectivity before topology construction,
non-finite inputs, degenerate sampled Jacobians and sampled orientation changes.
Uniformly reversed orientation uses absolute integration measure. These checks
are not a proof of geometry validity between quadrature points.
`HexPartialC64::diagonal` extracts the assembled diagonal during host setup
without constructing dense element matrices. The CUDA-gated FEM integration
test retains the operator, Jacobi preconditioner and GMRES session across three
RHS solves on curved Q2/Q3 meshes. Its coercive shifted operator tests session
integration; it does not justify Jacobi for general indefinite Helmholtz.
Boundary facets/constraints and production shifted-Laplacian or multigrid
integration remain open for this path.

Producer validation: 1,912 FEM CPU tests passed (one existing ignored test),
and all 643 compute CPU tests passed. Root check and Clippy passed with existing
workspace warnings. FEM parity covers Q1/Q2/Q3 on shared warped elements,
reversed orientation, explicit curved Q2/Q3 geometry (including Q1 solution
spaces), diagonals and all operator actions. Invalid-input regression tests
include non-finite geometry, singular/folded samples and malformed connectivity.
The CUDA-enabled persistent-solve test compiled and explicitly skipped with
`CUDA_ERROR_NO_DEVICE`; no native solve or performance result is claimed.
Logs: `/tmp/hex-producer-final-check.log`,
`/tmp/hex-producer-final-clippy.log`, `/tmp/hex-producer-final-fem.log`,
`/tmp/hex-producer-final-compute.log`, `/tmp/hex-producer-final-cuda.log`.

### Hex prescribed-pressure constraints and resident RHS lifting

`HexPartialC64::try_with_dirichlet` now represents symmetric row/column
elimination, P A P + Q, with unit constrained diagonals. Its host and CUDA RHS
lifting computes P(b - A Q g) + Q g for changing nonzero pressure values. Free
entries of g are ignored, including non-finite placeholders. Constraint masks
are pool-owned on CUDA, and masking is fused into the existing gather and
scatter stages: both operator action and block-RHS lifting use two launches,
retained element scratch and no bulk host transfers in the implementation.

Tests compare empty/partial/full masks, all actions, diagonals and lifting
against unconstrained quadrature and FEM's independent boundary elimination for
Q1/Q2/Q3. CUDA-gated tests check multi-RHS action/lifting, unchanged hot-path
allocation/transfer counts and a retained curved Q2/Q3 solve with changing
prescribed values. This closes prescribed-pressure handling for this primitive;
Robin/Neumann facets, general constraints, production indefinite preconditioning
and native GPU validation remain open.

Constraint validation: 644 compute CPU tests and 1,913 FEM CPU tests passed
(one existing FEM test ignored). Root check and Clippy passed with existing
workspace warnings; CUDA compute Clippy passed cleanly. NVRTC compiled the
updated kernels. Native action and constrained persistent-solve tests explicitly
skipped with no CUDA device, so their runtime counters/parity remain unverified.
Logs: `/tmp/hex-bc-all-compute.log`, `/tmp/hex-bc-all-fem.log`,
`/tmp/hex-bc-root-check.log`, `/tmp/hex-bc-root-clippy.log`,
`/tmp/hex-bc-cuda.log`, `/tmp/hex-bc-native-fem.log`.

### Complex-double shifted-Laplacian AMG connection

`BackendAmgPreconditioner<CudaBackend, Complex64>` now supports retained
cuSOLVER coarse LU and checkpoint restoration through the CUDA-specific
constructors. Host hierarchy construction is followed by device uploads and
coarse factorization; repeated applications use device SpMV/BLAS and retained
factors. Singular C64 coarse factorizations return an error, with no implicit
regularization or CPU solve fallback. Existing raw backend/vendor allocations
are not newly covered by MemoryPool quota accounting.

`prepare_hex_shifted_laplacian` assembles K + s M during host setup with the
partial-assembly producer's geometry admission, quadrature, H1 numbering and
Dirichlet elimination. FEM's CUDA feature now forwards `sonium-solvers/cuda`,
making the coarse-factor constructor available without unrelated feature
unification. The retained FEM integration now uses FGMRES on the original
indefinite operator with a distinct damped shifted-Laplacian AMG preconditioner,
reusing both across changing RHS/prescribed values. Shift and AMG configuration
remain explicit workload choices; no universal convergence guarantee is claimed.

A CPU backend regression executes this setup on curved Q2/Q3 meshes with a
multilevel hierarchy, repeated solves and warm starts, then independently checks
the residual against the original partial-assembly operator. CUDA-gated tests
cover native coarse action, repeated applications, checkpoint reconstruction and
absence of warm allocations/bulk transfers, plus the complete FEM connection.
Native execution and time-to-tolerance profiling remain required on a GPU.

Validation for the C64 AMG connection: all 333 preconditioner unit/integration
tests passed (one ignored doctest), and 1,914 FEM CPU tests passed (one existing
ignored test). Root check, root Clippy and CUDA-enabled FEM Clippy passed with
existing warnings. The shifted-AMG FGMRES regression passed both ordinary and
CUDA-enabled CPU builds. Native coarse/checkpoint and FEM solve tests compiled
and explicitly skipped with `CUDA_ERROR_NO_DEVICE`. Logs:
`/tmp/c64-amg-all-preconditioners.log`, `/tmp/c64-amg-all-fem.log`,
`/tmp/c64-amg-root-check.log`, `/tmp/c64-amg-root-clippy.log`,
`/tmp/c64-amg-cuda-clippy.log`, `/tmp/c64-amg-native.log`,
`/tmp/c64-amg-native-fem.log`.

### Fused hex residual and weighted-Jacobi actions

The hex scatter kernel now optionally computes b - A x or the simultaneous
update x + omega D^-1 (b - A x), including constrained identity rows. Both use
the existing element action and scatter launches, retained scratch, and a
caller-retained inverse diagonal shared across RHS columns. The operator's
`DeviceOperator::try_residual` override connects this directly to persistent
Krylov residual checks. Damping is explicit; this smoother is not a replacement
for the shifted-Laplacian AMG preconditioner on indefinite problems.

CUDA-gated tests compare scalar/block residuals and updates against host volume
actions for empty/partial/full constraint masks, check two launches with no warm
allocation/bulk transfer, reject invalid damping/diagonal lengths without
changing output, and check zero damping with poisoned RHS/diagonal data. NVRTC
compilation passed. Native parity and performance remain unverified without a
CUDA device; fewer implementation passes do not establish a measured speedup.

Fused-hex validation: all 644 compute CPU tests passed. Root check, root Clippy
and CUDA compute test Clippy passed with existing warnings. CUDA-enabled FEM
parity and CPU shifted-AMG convergence tests passed; the native FEM solve and
kernel runtime tests explicitly skipped with no device. Logs:
`/tmp/hex-fused-compute.log`, `/tmp/hex-fused-root-check.log`,
`/tmp/hex-fused-root-clippy.log`, `/tmp/hex-fused-clippy.log`,
`/tmp/hex-fused-cuda.log`, `/tmp/hex-fused-fem.log`.

### Retained volume-plus-boundary composition

`operator_sum::DeviceSum` combines two device operators using pooled scratch
prepared for scalar and explicitly requested block sizes. It supports forward
and Hermitian actions and preserves the left child's fused scalar residual
path: r = (b - A x) - B x. The wrapper introduces no host transfers or hot-path
allocations. Children must themselves supply native actions for each requested
size; default child staging is not converted into a native action by composition.
The exact pool backend is required. Child/accumulation failures do not publish
partial scratch results; final copy failures retain backend dispositions.

`prepare_tet_p1_robin` prepares P1 tetrahedral Robin/impedance terms using FEM's
checked host quadrature. It separates P B P, P B Q and P f, so the volume supplies
Dirichlet identity rows and the boundary contributes no duplicate identities.
The lifted RHS receives P f - P B Q g. Matrices can be uploaded/prebaked once
alongside a matrix-free volume operator. A CUDA-gated integration composes the
Tet P1 kernel with the retained boundary sparse operator and uses shifted AMG
plus persistent FGMRES across repeated RHS solves. Native forward/adjoint/block
composition tests also check warm allocation and bulk-transfer counts.

This first connection covers host-prepared P1 triangular boundary facets.
GPU numeric facet assembly, higher-order traces, quadrilateral faces, changing
coefficients without host rebuild, and broader general constraints remain open.

Composition validation: all 646 compute CPU tests and 1,915 FEM CPU tests
passed (one existing FEM test ignored). Final root check and Clippy passed with
existing warnings. The independent FEM comparison covers intersecting Robin
and prescribed faces, nonzero data, the action and P B Q lifting. CPU sum tests
cover complex non-Hermitian children, forward/Hermitian/block actions, residuals,
unsupported block sizes and no publication after a child failure. CUDA builds
passed; native sum and Robin/volume persistent-solve tests explicitly skipped
without a device. Logs: `/tmp/boundary-sum-final-check.log`,
`/tmp/boundary-sum-final-clippy.log`, `/tmp/boundary-sum-final-compute.log`,
`/tmp/boundary-sum-fem.log`, `/tmp/boundary-sum-native.log`,
`/tmp/boundary-sum-native-fem.log`.

### Complete hex traces and native launch validation

The hex Robin producer now integrates Q1/Q2/Q3 solution traces on linear or
curved geometry, with complete geometry face records and owning-element maps.
It retains the host-prepared P B P, P B Q and P f split for volume composition
and prescribed-pressure lifting. Analytic curved-face area, polynomial trace
energy, shared-face loads, orientation independence and malformed-record tests
pass. Its persistent CUDA integration composes the partial-assembly volume with
retained sparse boundary terms and shifted-Laplacian AMG for FGMRES.

Native execution outside the sandbox exposed and fixed the P2 launch-resource
failure. Occupancy selection happens once per loaded action/assembly kernel;
hot execution retains the existing one-launch/no-transfer contract. Tet P1/P2
and hex native suites pass, including forward/transpose/Hermitian, constraints,
block RHS, numeric assembly, fused modes and warm allocation/transfer checks.
CUDA memcheck reports zero errors. These are numerical/residency results, not
a speedup benchmark; larger production runs, profiling, GPU numeric facets,
preconditioner updates and the rest of the coverage program remain open.

### Measured P2 optimization and high-order preconditioner gap

The [CUDA FEM benchmark](cuda-fem-kernels.md) now compares retained cuSPARSE
and specialized operators on identical Helmholtz systems with shared shifted
AMG. The P2 kernel uses expanded barycentric stiffness moments, reducing action
time by 2.77–3.34× and full solve time by 1.73–1.96× on two measured cases versus
the previous custom kernel, at identical iteration counts. cuSPARSE remains
faster on all measured cases. Native parity, memcheck and persistent P2 solve
checks pass. This is an incremental kernel improvement, not an overall GPU
performance milestone.

The broader benchmark exposes three nonconverging high-order hex cases with
both operator routes. The current Jacobi-AMG/restart configuration is therefore
not sufficient to close suitable-preconditioner integration. Preserve these
cases as gates for p-multigrid/auxiliary-operator and smoother work. Raw failed
and passing results, parameters, setup times and source hashes are retained in
the benchmark evidence. Hardware profiling counters are unavailable; profiling
and larger production measurements remain open.

### Low-order refined auxiliary AMG for high-order hexes

The [auxiliary-preconditioner follow-up](cuda-fem-preconditioners.md) closes the
three observed high-order hex convergence failures with the original operator,
restart and tolerance unchanged. `prepare_hex_lor_shifted_laplacian` assembles
Q1 subcells on the original H1 nodal grid, preserving DOF/constraint numbering.
Retained CUDA AMG acts on this auxiliary matrix while the outer FGMRES session
continues using the high-order CSR or partial-assembly operator. Eight benchmark
cases at subdivisions 4/8 pass; the three formerly failing hex cases take
212, 123 and 329 iterations. Warm preconditioner actions allocate and transfer
no data, and no smoother fallback is recorded. This is a verified execution
route, not a finished scalable preconditioning design.

Backend L1-Jacobi now uploads inverse row norms and uses unit damping to match
host semantics; per-level/checkpoint regression tests pass on CPU and CUDA.
Corrected L1 does not close the observed convergence gaps. Multicolor smoothing
reduced iterations on completed cases but had prohibitive cycle cost; its
interrupted diagnostic is retained separately. Next optimise auxiliary hierarchy
coarsening and suitable smoothing, and broaden curved/material/frequency-update
coverage without substituting the auxiliary matrix for the original residual.

### Reusable hex pressure solve API

`CudaHexPressureSession` now packages the Q1/Q2/Q3 volume, fixed Robin/Neumann
boundaries, nonzero Dirichlet lifting, low-order refined shifted AMG and retained
FGMRES workspace. Callers supply resident uneliminated volume loads and prescribed
values; the session adds the fixed boundary source exactly once. Successful
solutions remain on device and can seed the next RHS. Invalid input, cancellation
or unsuccessful convergence hides the previous solution. A subsequent zero-start
solve can recover after correctable input errors; backend failures retain their
own context-recovery requirements.

`DeviceSumWorkspace` separates scratch ownership from borrowed operators so the
session can bind volume and boundary actions without self-referential storage or
hot allocations. Shared scratch rejects concurrent/recursive use rather than
deadlocking, and a regression verifies that rejection preserves caller output
and leaves the workspace usable.

The native public-API regression passes for Q2/Q3 with absent boundary terms,
pure Neumann loads, multiple Robin markers, nonzero prescribed values and changed
RHS/warm starts. Free entries of the prescribed vector are poisoned with NaNs to
check masking. It checks the original assembled free equations and constrained
values independently, asserts zero additional device allocations on the second
solve, and verifies failed-publication, cancellation and marker validation.
Native sum tests also pass. This is a specialized reusable library route;
application migration, changing frequency/material/boundary coefficients, curved
geometry coverage for this wrapper, GPU facet assembly and whole-workflow quota
accounting remain open. It does not close the full CUDA coverage goal.

Validation: compute CPU 647 passed; FEM CPU 1,920 passed and one ignored; native
sum 4 passed; native pressure session 1 passed (six Q2/Q3 boundary configurations).
Root `cargo check`, root `cargo clippy` and CUDA FEM test Clippy passed with
existing warnings. The public error boxes the larger Krylov diagnostic, keeping
the successful-call result small. Logs:
`/tmp/hex-session-compute.log`, `/tmp/hex-session-fem.log`,
`/tmp/hex-session-native-sum.log`, `/tmp/hex-session-native.log`,
`/tmp/hex-session-root-check.log`, `/tmp/hex-session-root-clippy.log`,
`/tmp/hex-session-cuda-clippy.log`.

### Native Robin numeric assembly and persistent boundary updates

`QuadratureAssemblyMap` and `CudaQuadratureAssembly` retain fixed real
geometry/basis weights and reduce resident complex coefficient samples into
matrix values or loads. Each nonempty map uses one deterministic, output-owned
CUDA kernel with no atomics, warm allocation or host transfer. Structural zeros
remain in the template, allowing coefficients to become zero and later nonzero
without rebuilding sparse descriptors. Empty output and empty-contribution maps
have explicit checked behavior.

`prepare_hex_robin_quadrature(s)` lowers checked Q1/Q2/Q3 face geometry into
maps for P B P, P B Q and P f. Multiple markers share one combined pattern;
coordinates and marker IDs define coefficient-buffer ordering. Geometry and
basis tabulation remain host setup. The expanded contribution map costs
O(samples × trace-DOFs²) storage, so this is a correctness/residency baseline,
not a demonstrated optimal high-order representation. Arbitrary Rust callbacks
still run on the host; a device material law can supply resident samples.

`CudaHexPressureSession::try_new_updatable` retains these maps, CSR structure and
update scratch. `try_update_boundary` assembles action, prescribed-pressure
coupling and source load with at most three kernel launches and three device
copies. It invalidates published solutions; a successful update preserves the
previous iterate for a warm start. A failed numeric update blocks solves until
a full successful retry, preventing mixed old/new boundary state from being
used. AMG remains based on the initial boundary as an explicit lagged
preconditioner; convergence certification always uses the updated system.
Large coefficient changes may need a preconditioner rebuild.

The varying-boundary test exposed two allocation gaps hidden by equal-iteration
warmups: lazy Krylov vector growth and CUDA reduction pointer/result cache
growth. Full-cycle vector reservation is now available on GMRES workspaces and
sessions; CUDA has a matching reduction-cache reservation. The FEM session uses
both. Reduction-cache allocations retain the backend-ledger accounting policy,
outside MemoryPool quota. Changed short final restart lengths can still resize
small projection buffers; full variable-length-cycle residency remains open.

Native Q2/Q3 tests pass with multiple markers, zero-to-nonzero coefficients,
changed source loads, nonzero prescribed values and repeated solves using the
initial AMG. Boundary updates use three launches with zero allocation/H2D/D2H;
subsequent tested solves allocate no additional device storage. Independent
assembled equations verify the changed-system residual, and invalid-update
recovery/publication checks pass. Curved Q2/Q3 matrix/load updates separately
match direct quadrature and feed retained CSR actions without host staging.
The new kernel passes CUDA memcheck with zero errors. No wall-time speedup is
claimed; map memory, setup/JIT cost and update timing still need measurement.

Remaining priority-3 work includes tetrahedral/other facet forms, broader
constraints, spatially varying volume-coefficient changes, numeric AMG refresh and production sweep
migration. This increment does not close the overall CUDA coverage goal.

Validation: 649 compute CPU tests, 1,921 FEM CPU tests (one ignored), and 353
iterative-solver unit/integration tests passed. Native boundary tests (8),
pressure-session tests (2), and focused compute quadrature tests (3) passed.
Root check, root Clippy and CUDA FEM test Clippy passed with existing warnings.
Logs: `/tmp/facet-final-compute2.log`, `/tmp/facet-final-fem.log`,
`/tmp/facet-iterative.log`, `/tmp/facet-native-compute.log`,
`/tmp/facet-native-fem.log`, `/tmp/facet-native-session3.log`,
`/tmp/facet-memcheck.log`, `/tmp/facet-final-check2.log`,
`/tmp/facet-final-clippy2.log`, `/tmp/facet-final-cuda-clippy2.log`.

### Retained hex volume-frequency updates and explicit AMG replacement

`CudaHexPartialC64::try_set_scales` applies independent complex diffusion and
mass multipliers relative to the originally uploaded quadrature coefficients.
The scales affect forward/transpose/Hermitian actions, nonzero-pressure lifting,
residuals and Jacobi updates; constrained identity rows remain one. Callers must
supply an updated inverse diagonal or intentionally retain a lagged smoother.
Changing scales itself issues no allocation, transfer or GPU launch. Captured
CUDA graphs retain their old launch parameters and require recapture.

The pressure session now uploads canonical K-M geometry and applies k² as the
mass multiplier. `try_update_wavenumber` therefore supports complex k and
transitions from/to k=0 without dividing by the initial frequency or re-uploading
geometry. Successful updates unpublish the old solution but retain its iterate
for warm starts; non-finite k or k² is rejected without changing current state.
Frequency-dependent Robin/source samples still use the separate boundary API.

`try_rebuild_preconditioner` accepts a replacement auxiliary matrix in original
H1 numbering and constructs AMG before swapping it into the session. The volume,
boundaries and Krylov storage survive. Failure retains the old preconditioner,
subject to backend recovery requirements. This is host hierarchy setup followed
by device upload/coarse factorization, not native numeric multilevel refresh.
The auxiliary matrix never replaces the physical residual operator.

Native Q2/Q3 sweeps on warped hex meshes pass for k=0, 4+0.2i, 6-0.1i, then 0,
with changing multi-marker Robin/source data and nonzero prescribed pressures.
AMG is retained for the first two systems, explicitly replaced with a
K_LOR+(36+36i)M_LOR auxiliary matrix for the third, then retained again. Original
assembled equations independently certify each updated solve. Wavenumber plus
boundary updates retain the three-launch/no-allocation/no-bulk-transfer boundary
contract; later solves allocate no additional device storage. These small-mesh
checks do not establish a universal frequency-reuse policy or scalable AMG.

Final frequency-update verification and timing are retained in
[the raw evidence](cuda-hex-frequency-2026-10-21.json), including final source
hashes, native test logs and all three kernel trials. The fixed/scaled entrypoint
split avoids extra scale arguments and arithmetic for unit-scale operators.
Fixed Q2/Q3 action medians are 0.05682/0.06498 ms, compared with the earlier
0.05701/0.06523 ms baseline. The unconditional and runtime-branch variants were
slower at approximately 0.059/0.0676 ms. This recovers a regression; it does not
establish an overall speedup over cuSPARSE. Final specialized solves take
189.61/481.59 ms versus cuSPARSE 185.33/475.47 ms, at identical 135/212 iterations.
The measurements cover fixed operators, not total frequency-sweep wall time.
Cold setup/cache state was not controlled across variants; the final Q2 setup
of 239.66 ms must remain visible rather than being treated as a warm-path cost.

The final native compute suite passes both tests; both retained pressure-session
tests pass, and the final scaled/fixed action passes memcheck with zero errors.
The warped Q2/Q3 sweep residuals are below 1e-10 for all eight systems. Root
check, root Clippy and CUDA FEM test Clippy pass with existing warnings.
Application migration, spatial coefficient updates, scalable auxiliary AMG and
the remaining coverage gates are still open.

### Resident weighted volume coefficient replacement

`CudaHexPartialC64::try_update_quadrature_data` now replaces the seven weighted
coefficients per quadrature point using one device-to-device copy. Geometry,
restriction, constraints and scratch remain fixed; the current global diffusion
and mass scales apply to the new data. The update allocates nothing and transfers
no bulk data to or from the host. Invalid buffer metadata is rejected before
mutation. A failed copy blocks actions until a successful replacement; backend
recovery alone cannot make potentially incomplete data usable.

`CudaHexPressureSession::try_update_volume_data` connects this update to retained
FGMRES and auxiliary AMG. It expects canonical K-M coefficients, including
geometry/weights and the negative mass sign, but excluding the session's k².
Successful updates retain the old iterate as an initial guess and remove its
solution certification. Failed updates clear publication and warm-start access.
The auxiliary preconditioner remains explicitly lagged until rebuilt.

Native tests vary complex coefficients across quadrature points and check
forward/transpose/Hermitian, block, lifting, residual and smoother actions against
host references, including constrained rows. Update counters assert one full
device copy with zero allocations, launches or host transfers. Injected copy
failure verifies action rejection, unchanged caller output and successful retry.
Both compute tests pass under CUDA memcheck with zero errors. Both retained FEM
session tests also pass, including mass-coefficient changes checked against an
independently assembled physical operator, and failed-update recovery.

This closes resident replacement of already weighted coefficients, not GPU
material-law evaluation or geometry transformation. Tests upload input fixtures
outside the measured update; production device material producers remain to be
connected. The changed Q3 mass case takes 233 iterations with the selected
auxiliary AMG, reinforcing the need for scalable preconditioner updates. No new
kernel or wall-time speedup is claimed for this copy-based integration.

CPU validation: 649 compute tests and 1,921 FEM tests passed, with one existing
FEM test ignored. Root check, root Clippy and CUDA FEM test Clippy passed with
warnings. The full CUDA coverage objective remains open.

Validation logs: `/tmp/hex-data-native-compute.log` (including memcheck),
`/tmp/hex-data-native-final-fem.log`, `/tmp/hex-data-cpu-compute.log`,
`/tmp/hex-data-cpu-fem.log`, `/tmp/hex-data-final-check.log`,
`/tmp/hex-data-final-clippy.log`, and `/tmp/hex-data-final-cuda-clippy.log`.

### Native scalar material lowering into retained hex solves

The new `sonium_hex_material_c64` kernel maps two resident complex sample fields
into the seven weighted entries at each quadrature point. Scalar diffusion
multiplies the six symmetric metric entries and mass multiplies the weighted
mass entry. It uses a retained immutable reference, so updates are absolute and
can pass through zero without division or cumulative scaling. Global wavenumber
scaling remains separate. This covers scalar spatial materials, not general
physical anisotropic tensors or automatic lowering of arbitrary Rust callbacks.

`try_prepare_material_updates` opts into seven extra complex values per point,
charged to the pool, with a one-time allocation/copy. It is idempotent and later
raw-data updates do not redefine the reference. Each subsequent material update
uses exactly one launch and zero allocations or H2D/D2H/D2D copies. A failed
launch invalidates the volume until a successful replacement; recovery cannot
silently restore a potentially incomplete operator.

Updatable pressure sessions prepare canonical K-M reference data and expose
physical sample coordinates through `volume_samples`. Geometry coordinates are
produced with the full curved basis in element-major, x-fastest tensor order;
analytic Q2/Q3 geometry tests check that ordering and interpolation. The session
material API retains geometry, constraints, boundaries, Krylov storage and the
explicitly lagged AMG. Successful changes unpublish old solutions but preserve
their iterates as warm starts. Frequency and boundary dependencies stay explicit.

Three native compute tests pass under memcheck with zero errors. New cases cover
zero-to-heterogeneous-to-unit updates, complex global scales, raw-data/reference
independence, two RHS columns, transpose/adjoint actions, malformed shapes and
injected launch failure/retry for n=2/3/4/8 tensor bases. Both native pressure
session tests pass. Their warped-mesh Q2/Q3 sweep now includes spatially varying
complex diffusion and mass, checked against independent physical-coordinate form
assembly, with nonzero prescribed pressure and changing Robin/source terms. The
spatial Q2/Q3 systems take 49/224 iterations; this is execution evidence, not a
scalability or speedup claim.

Application material laws still need to generate the two resident sample fields.
The tests upload fixtures before measuring update counters. Device material
evaluation, anisotropic lowering, numeric AMG refresh and production migration
remain open. Raw validation logs and source hashes are in
[`cuda-hex-material-2026-10-21.json`](cuda-hex-material-2026-10-21.json).

CPU regression suites pass: 649 compute tests and 1,921 FEM tests, with one
existing FEM test ignored. Root check/Clippy and CUDA compute/FEM test Clippy
pass with existing warnings. The overall implementation goal remains open.

### CUDA CG with a stationary symmetric AMG cycle

The public CUDA facade now admits `CgAmg` for f32/f64/c32/c64 SPD/HPD systems.
It uses retained PCG workspace and `AmgConfig::for_backend_pcg`: equal L1-Jacobi
pre/post sweeps, conjugate-transpose restriction, undropped Galerkin coarse
matrices, a V-cycle and a fixed coarse smoothing polynomial. Row-norm scaling
bounds the normalized HPD spectrum by one; paired relaxation and positive coarse
correction provide the symmetric positive-definite action needed by PCG. This
preset is specifically for the backend implementation, not a claim about the
different host smoother or arbitrary modified configurations.

Existing matrix validation remains mandatory. Small matrices receive an
independent factorization check; large matrices retain the explicit
positive-definiteness assumption/certificate policy. Non-Hermitian complex
systems and symmetric-indefinite systems are rejected before device setup.
Indefinite Helmholtz and saddle-point systems must retain their suitable routes.
The convenience facade still constructs a context/session per call and downloads
the result for host certification; repeated callers can retain the lower-level
PCG session and the new preconditioner themselves.

Two integration defects were fixed during verification. Host AMG checkpoint v1
omitted `galerkin_drop_tol`, silently restoring 0.001 even when the source policy
was zero. Version 2 preserves the value; v1 remains readable with its historical
default. Tests cover nondefault/zero tolerances, invalid encoded values and
legacy decoding, and compare complete backend actions after restore. Separately,
the CUDA facade now coalesces additive CSR duplicates before upload and reuses
the canonical matrix for AMG setup. This avoids cuSPARSE descriptor rejection
when duplicate entries exceed the dense matrix size. Certification still uses
the original input system.

Native evidence includes the full applied AMG matrix for real and genuinely
complex multilevel cases: Hermiticity, independent positive-pivot factorization,
checkpoint equivalence, no host transfer during application, and no smoother
fallback. Public dispatch and retained-session tests cover all four scalar
types, independent residual certification, changed RHS/warm starts and zero
additional device allocations on the second solve. All five CUDA facade tests
and the native AMG cycle test pass. CPU suites pass 336 preconditioner tests and
322 facade tests, with three existing ignored tests across the suites.

This closes the missing CG+AMG dispatch combination, not scalable AMG tuning or
all production migration. Coarse smoothing still costs twenty sweeps; benchmark
cycle time and convergence before claiming a performance improvement. Validation
logs and source hashes are retained in
[`cuda-pcg-amg-2026-10-21.json`](cuda-pcg-amg-2026-10-21.json).

### Fused native AMG Jacobi update

Backend AMG Jacobi and L1-Jacobi now retain the cuSPARSE matrix action and
replace the following four operations (negation, residual AXPY, diagonal CSR
action, solution AXPY) with one pointwise CUDA kernel. The primitive supports
f32/f64/c32/c64, including complex inverse diagonals. It performs
x += omega * inverse * (b - Ax) without new buffers or host transfers.
Other backends retain the previous sequence. Missing native support returns
false before mutation; invalid inputs and launch failures return errors rather
than silently selecting another execution path.

Native tests cover all scalar types, empty and non-block-aligned sizes, zero
and negative damping, length/stream/damping rejection and injected launch
failure/recovery. They verify one launch for nonempty updates, no allocation
or transfer, and independent arithmetic parity. Three contract tests pass
under CUDA memcheck with zero errors. Native AMG L1, stationary HPD-cycle and
complex coarse/checkpoint tests pass, as do both retained FEM pressure-session
tests with changing RHS, frequency, material and boundaries.

The manual complex64 update diagnostic compares the previous four operations
against fusion with identical inputs, seven alternating-order trials and
independent output checks. Median update latency including synchronization is
49.47 -> 15.31 microseconds at 1,024 entries and 55.56 -> 15.72 microseconds at
65,536 entries. Setup, input restoration and the physical matrix action are
outside the timed interval. This is an update-level result, not an end-to-end
FEM speedup; complete AMG-cycle and time-to-tolerance benchmarks remain required.
Raw timings and source hashes are in
[cuda-jacobi-update-2026-10-21.json](cuda-jacobi-update-2026-10-21.json).

The compute CPU suite passes 649 tests; the preconditioner suite and CPU
unavailable-native contract also pass. Root check/Clippy and CUDA test Clippy
are recorded separately. This advances priority 2; application migration,
scalable preconditioning and the broader priorities 1–4 remain open.

### Matched complete-solve verification of fused AMG

The update-level result now has a matched complete-solve comparison. Two release
executables were built from identical working-tree sources except that the
baseline disables the new native-update branch in backend AMG. The production
source was restored byte-for-byte before the fused build. Three alternating-order
process trials each run five timed zero-start solves after warmup, with the
same physical matrix, shifted auxiliary operator, AMG hierarchy, tolerance and
restart. The report uses the median of the three process medians. Setup remains
separate; these are retained solve times, not total simulation times.

On the RTX PRO 6000 Blackwell, cuSPARSE-backed solves improve 1.22–1.28x and
specialized solves improve 1.12–1.27x across Tet P1/P2 and Hex Q2/Q3 at
subdivisions 4/8. All sixteen operator/workload combinations preserve iteration
counts across all runs. Independent original-system relative residuals remain
below 2e-9. Every measured warm solve reports zero additional device allocations;
every measured warm preconditioner action reports zero allocations or host
transfers and no substituted smoother. Small Krylov control transfers remain
counted in the raw records and are not claimed to be absent.

| Case / specialized operator | DOFs | Previous solve ms | Fused solve ms | Iterations |
| --- | ---: | ---: | ---: | ---: |
| Tet P1, subdivisions 4 | 125 | 46.42 | 37.43 | 36 |
| Tet P2, subdivisions 8 | 4,913 | 884.04 | 787.06 | 229 |
| Hex Q2, subdivisions 8 | 4,913 | 268.05 | 216.91 | 123 |
| Hex Q3, subdivisions 8 | 15,625 | 1,030.68 | 845.23 | 329 |

Single warmed AMG-action samples across the three trials improve 1.32–1.40x;
these are supporting diagnostics, while the repeated complete solves provide
the stronger performance evidence. The biggest case still requires 329
iterations. Fusion improves cycle cost without fixing weak coarsening or
establishing mesh-independent convergence. Specialized Tet actions still trail
cuSPARSE, so this result does not justify replacing the vendor sparse action.

The full raw records, per-case summaries, source hashes and executable hashes
are retained in
[cuda-jacobi-fem-solves-2026-10-21.json](cuda-jacobi-fem-solves-2026-10-21.json).
The executable benchmark is
`RAYON_NUM_THREADS=4 SONIUM_FEM_BENCH_PRECONDITIONER=lor cuda_fem_compare 4 8`.
No permanent benchmark toggle was added to production APIs. Root check and
Clippy pass after source restoration. Next work remains scalable auxiliary
preconditioning and producer integration, alongside the outstanding topology,
physics, boundary and update coverage for priorities 1–4.


### Retained mixed-precision workspace in the production room sweep

`GmresIrWorkspace` and
`gmres_ir_backend_preconditioned_with_workspace` retain both precisions'
Krylov, correction and true-residual scratch in the caller's MemoryPool.
The existing FGMRES-IR algorithm, adaptive inner tolerances, restart behavior
and original-system residual certification are unchanged. Scratch grows with
maximum simultaneous demand, is reused across inner solves/restarts/calls, and
is released on a dimension change or workspace drop. Allocation/backend/inner
solver errors discard cached storage before retry; input validation errors
preserve it. Host Hessenberg/diagnostic arrays and backend reduction caches
retain their separate allocation contracts.

The CUDA room F64Outer path now owns one such workspace for the sweep.
It also prebakes both CSR operators once per frequency and reuses their
descriptors across all sources and inner actions. A native allocation test
exposed that ordinary DeviceCsr actions recreate descriptors on every action;
scratch reuse alone did not remove those allocations. The retained-descriptor
test now observes zero additional device allocations on repeated real and
complex mixed-precision solves, with changed RHS, restarts and true-residual
telemetry. Tests also cover exact warm starts, dimension changes, quota failure
and a preconditioner that writes NaNs before returning an error. Three tests
pass under CUDA memcheck with zero errors.

The production four-frequency room fixture passes CPU/listener reciprocity and
physical-ledger parity. Its allocation counts are 88, 4, 4, 4; a regression gate
bounds the last three frequencies at four allocations each. These remaining
allocations are not claimed absent: descriptors currently renew at frequency
boundaries, and whole-workflow allocations remain broader than Krylov scratch.
The preconditioner ages are 0, 1, 2, 3 with no fallback. Native mixed-precision
parity, block-output, launch/device-loss recovery and the two cancellation/retry
lanes pass when run with their required SR1 environment settings.

The broader native run also exposed a pure-f32 physical-balance parity failure:
at 80 Hz, CPU remainder -1.143248441e-9 versus CUDA 1.906844778e-7 exceeds the
existing 1e-7 absolute gate for this fixture. It reproduces identically with
fused AMG disabled. The f32 path does not call the new IR workspace. This is
an open validation gap; the gate was not relaxed and the broad CUDA suite must
not be reported fully passing. Investigate f32 residual correction and physical
acceptance before declaring that production row covered.

CPU verification passes 649 compute tests, 329 room tests (one existing ignore)
and 12 pipeline tests. Root check/Clippy and focused CUDA workspace Clippy pass.
Raw production evidence, source hashes and validation logs are retained in
[cuda-ir-workspace-2026-10-21.json](cuda-ir-workspace-2026-10-21.json).

### Pure-f32 physical parity restored; descriptors retained for all room modes

The pure-f32 physical-balance failure reported above is now resolved for the
native room fixtures without relaxing their physical acceptance threshold.
The iterative solver targeted 1e-6, but promoted residual correction previously
stopped at 1e-4. Its acceptance/correction target now equals the configured f32
solver target. This causes correction to run when the promoted residual fails
the requested accuracy, rather than publishing the insufficiently accurate
candidate. Both the scalar four-frequency physical-ledger/reciprocity fixture
and separate block-RHS fixture pass. Promotion still certifies the rounded f32
system; it is not a claim of arbitrary f64-original-system accuracy.

Both CSR descriptors are now retained for each frequency in all CUDA room
precision modes. Scalar f32 solve/correction and promoted residual actions use
those descriptors; block actions use them as well. Promoted scalar certification
now uses checked casts, matrix action and reductions. Krylov scratch remains
retained across frequencies as established in the earlier increments.

For the scalar four-frequency fixture, retaining descriptors reduces device
allocation counts from 141/57/30/30 to 88/4/4/4 at otherwise identical stricter
certification. Iteration counts stay 43/41/20/20 and residuals are unchanged:
2.499e-7, 2.284e-7, 9.062e-7 and 7.857e-7. The regression now explicitly requires
residual <= 1e-6 and at most four allocations on each later frequency, in
addition to independent physical parity. No solver fallback or retry occurs.
These are allocation/correctness results, not a measured wall-time speedup.

All nine CUDA room regression tests pass when cancellation and fault-injection
tests receive their required per-lane environment. This includes scalar/block
parity, mixed-precision parity/reuse, cancellation before setup and during the
sweep, warm-state retry, launch failure and device-loss recovery. CPU room tests
pass 329 with one existing ignore; pipeline tests pass 12. Root check/Clippy and
CUDA integration-test Clippy pass with existing warnings.

Evidence, exact runner settings and source hashes are retained in
[cuda-room-f32-certification-2026-10-21.json](cuda-room-f32-certification-2026-10-21.json).
The historical failing logs remain in the previous artifact. Outstanding work
still includes descriptor reuse across numeric frequency updates, block
initialization residency, output/physical-evidence transfers and the broader
physics/topology coverage. This resolves the observed f32 validation gap, not
the complete CUDA goal.


### Persistent physical-system and RHS descriptors across frequencies

The room runner now moves its existing A32/A64 CSR storage into
`PersistentDeviceCsr` once at setup. New checked `try_combine_values` and
`try_cast_values_from_lower` methods refresh numeric values using existing
device bases while preserving storage addresses, sparsity and descriptors.
No duplicate matrix or intermediate numeric array is introduced. Coefficient
and shape checks precede mutation; a backend failure may leave a partial update,
so callers must complete a successful replacement before reuse. The room
runner propagates these errors instead of continuing with partial values.

The last two per-frequency allocations came from RHS scatter actions.
Scalar and block RHS assemblers now own persistent sparse descriptors as well.
Their coefficient uploads and actions return checked errors, and malformed
input is rejected before modifying output. Existing scatter weights and source
coefficient ordering are preserved.

Both scalar precision modes now report device allocation counts **84/0/0/0**
for the four-frequency production fixtures, down from **88/4/4/4** before
cross-frequency descriptor reuse. The first frequency warms Krylov demand;
all later frequencies reuse device storage. Matrix-only descriptor retention
first reduced the later count to two, and retaining RHS descriptors removed
the remaining two. Iterations, true residuals and physical parity are unchanged.
The regression gates now require exactly zero allocations after the first
frequency for these fixed fixtures. This is not a promise for a new mesh,
larger Krylov demand, a preconditioner rebuild or every block workflow.

New compute tests cover f32/f64/c32/c64 coefficient combinations, zero/nonzero
replacement, complex promotion, malformed shapes and non-finite coefficients.
Native updates allocate nothing and transfer no host arrays. Compute update
tests and the native repeated-RHS test pass CUDA memcheck with zero errors.
All nine CUDA room regression tests pass with their required lane settings.
CPU suites pass 649 compute tests, 329 room tests (one existing ignore) and
12 pipeline tests; root check/Clippy and CUDA test Clippy pass.

Raw evidence and final source hashes are in
[cuda-room-persistent-csr-2026-10-21.json](cuda-room-persistent-csr-2026-10-21.json).
This closes descriptor lifetime reuse for the measured production sweeps.
Host source laws, diagnostic/physical-evidence transfers, block initialization,
whole-workflow quota and the wider coverage/kernel backlog remain open.
Metal runtime was not exercised; shared helper changes retain CPU regression
coverage and checked backend dispatch.

### Retained block workspace in the room producer

The CUDA pure-f32 multi-source point-Jacobi route now owns one
`BlockGmresWorkspace` for the complete frequency sweep and calls the reusable
block solver entrypoint. Promoted certification uses checked casts, AXPY and
copies; backend errors flow through the existing failure handling.

The production block regression now covers 80/100/120/140 Hz, including a
scheduled preconditioner rebuild, with CPU physical parity and reciprocity.
Native CUDA memcheck passes with zero errors. Device allocation counts are
344/77/91/85 and reported residuals remain below 1e-6. The regression requires
warm frequencies to allocate less than the cold solve; it does not assert zero
allocations. Remaining block temporary/fallback allocations and full-array host
initialization/certification are still open work. There is no timing claim or
matched before/after allocation comparison in this increment.

CPU room tests pass 329 with one existing ignore. Root check/Clippy and
CUDA-enabled room test Clippy pass. Evidence and source hashes are in
[cuda-room-block-workspace-2026-10-21.json](cuda-room-block-workspace-2026-10-21.json).

### Device reductions for promoted block certification

Promoted block certification now copies contiguous device columns into the
existing scalar residual scratch and computes checked device norms. Only scalar
norm results return to host; the two full RHS/residual downloads are removed
from this stage. CUDA contiguous view copies now use one driver copy per column
instead of one per scalar. Strided copies retain the previous checked path.

A native helper test covers three complex columns, zero RHS/residual behavior,
non-finite rejection, invalid shapes and repeated calls. It asserts no device
allocation, no H2D traffic and at most scalar-sized D2H traffic. The helper and
four-frequency production parity test pass CUDA memcheck with zero errors.
Production transfer counts remain unchanged: this fixture does not demonstrate
successful block acceptance, since the block attempt can fall back before
certification. Therefore these results establish the helper's residency and
production compatibility, not an end-to-end transfer reduction or speedup.
Block initialization, rank handling and remaining fallback allocations remain
open. Add explicit block acceptance diagnostics and an accepted production
fixture before claiming resident block execution.

Evidence: [cuda-room-block-norms-2026-10-21.json](cuda-room-block-norms-2026-10-21.json).

### Certified block results accepted by the production room sweep

Typed `FemBlockSolveOutcome` diagnostics now distinguish no attempt, accepted
block results with a correction count, and rejection with a reason followed by
producer scalar recovery. The outcome is preserved in RoomEQ metadata; older
records without this field deserialize as no attempt. Existing scalar fallback
stage and backend provenance retain their separate meanings.

The new diagnostics exposed the earlier fixture's failure: all four initial
f32 block attempts stopped at residuals around 1.2–2.5e-6, above the requested
1e-6 threshold. The producer now retains those iterates and applies up to six
promoted-residual corrections using the existing block solver, retained
workspace and two reusable correction buffers. Each correction solves for
A*x-b and subtracts the update. Acceptance still requires promoted residual
and physical-output gates; exhausted corrections or backend errors remain
explicit rejections. No accuracy threshold was relaxed.

All four production frequencies now report accepted block results with one
correction each. The test requires acceptance, correction execution, residuals
at most 1e-6, CPU physical parity and reciprocity. Measured residuals are
2.499e-7, 2.284e-7, 1.940e-7 and 1.655e-7. D2H traffic falls from
555064/266544/276664/420144 bytes to 20268/20008/20268/20008 bytes. Allocation
counts are 384/154/158/159, increased relative to the prior producer-fallback
route. Remaining block temporaries and changes in active-column shape must be
addressed before claiming allocation-free warm solves.

The accepted production test passes CUDA memcheck with zero errors. CPU room
tests pass 331 with one existing ignore; root check/Clippy and CUDA room test
Clippy pass. The compute block solver may internally deflate dependent columns;
producer acceptance is not a claim that every inner solve used block Arnoldi.
There is no matched wall-clock speedup claim. Raw before/after outcomes and
source hashes are in
[cuda-room-block-correction-2026-10-21.json](cuda-room-block-correction-2026-10-21.json).

### Retained scalar recovery inside block GMRES

`BlockGmresWorkspace::with_fallback_pool` now connects the existing pooled scalar
GMRES workspace to block rank-deficiency and stagnation recovery. The three
scalar column buffers are retained independently of the active block shape.
The room producer supplies its existing Krylov pool. The solve rejects a
backend instance different from that pool's backend before touching output.
The default constructor remains available with the same numerical fallback.

The focused dependent-RHS test covers f32/f64/c32/c64, changing RHS values, a
zero column, repeated solves, clearing storage, wrong-backend rejection and
quota failure. Native warm fallback solves make zero device allocations. The
pool accounts for scalar Krylov storage only; block arrays, fallback columns
and vendor scratch remain outside that accounting. Workspace capacity methods
document this distinction and `fallback_cached_bytes` exposes retained pooled
storage separately.

The accepted four-frequency production sweep now reports device allocations
247/5/8/5, down from 384/154/158/159. All four residuals and correction counts
are unchanged. The production regression permits at most 16 warm allocations
on this fixture to allow small preconditioner/cache growth while rejecting a
reconstructed fallback basis. Full host staging in initialization and scalar
recovery remains; active-block shape changes still replace block buffers.

CPU compute tests pass 649; focused fallback CPU tests pass two; room tests
pass 331 with one existing ignore. All three fallback tests and the production
block sweep pass CUDA memcheck with zero errors. Root check/Clippy and CUDA
compute/room test Clippy pass. Evidence:
[cuda-block-fallback-workspace-2026-10-21.json](cuda-block-fallback-workspace-2026-10-21.json).

### Device column recovery and solution assembly

Block rank-deficiency/stagnation recovery now extracts RHS and initial-guess
columns from device buffers, runs the retained scalar solve, and scatters each
result back to the device full-solution buffer. Final block solution assembly
also uses device scatter/copy. An already-converged warm start no longer
reuploads its solution. Initialization validation and initial QR still stage
full arrays, so this increment removes recovery/output traffic without closing
the full block residency gate.

The all-scalar dependent-RHS regression now uses 257 rows to separate full-column
traffic from compact coefficients. It bounds transfers to the remaining initial
staging plus small Krylov data, requires zero warm device allocations and
verifies an exact warm start has no H2D traffic. The native and CPU tests,
including quota and backend-instance rejection, pass under CUDA memcheck.

Against the immediately preceding pooled baseline, production H2D bytes fall
by 6144 per frequency and D2H bytes by 2048. New D2H totals are
125740/126440/127660/131240 bytes; these include compact pooled Krylov coefficient
transfers and must not be compared to the earlier nonpooled 20 KB totals as if
only this increment changed. All four accepted results, correction counts,
residuals and 247/5/8/5 allocation counts are unchanged. Initial staging and
coefficient-transfer volume remain separate optimization targets.

CPU compute tests pass 649. Focused tests and production block parity pass CUDA
memcheck with zero errors. Root check/Clippy and CUDA compute/room test Clippy
pass. Evidence:
[cuda-block-device-recovery-2026-10-21.json](cuda-block-device-recovery-2026-10-21.json).

### Device initial residual and QR

Block initialization now forms b-A*x, computes RHS/residual column norms,
extracts active columns, and performs two-pass QR through device primitives.
The initial basis passes directly into resident Arnoldi storage; only the small
R factor and projection coefficients are host data. Unused N-sized host
residual, active-block, basis and solution scratch arrays were removed. Input
finiteness validation still downloads the full RHS and supplied initial guess,
and a zero guess is still uploaded; these are the remaining large initialization
transfers.

Direct CPU/native tests verify QR reconstruction, orthogonality and clearing of
stale basis data for f32/f64/c32/c64. The 257-row dependent-block regression now
allows only the remaining input download plus compact coefficient traffic.
All direct QR and fallback tests pass CUDA memcheck with zero errors. The
production four-frequency sweep also passes physical parity and memcheck;
all four results remain accepted with one promoted correction.

Relative to device recovery alone, H2D bytes fall by 3072 per frequency and D2H
bytes by 2048. Allocation counts are 249/5/8/5, versus 247/5/8/5 previously. Reported
residuals are unchanged. CPU compute tests pass 650, and root check/Clippy plus
CUDA compute/room test Clippy pass. Evidence:
[cuda-block-initial-qr-2026-10-21.json](cuda-block-initial-qr-2026-10-21.json).

### Native finite validation and zero initialization

Optional checked backend hooks now test finite values and zero device buffers.
CUDA finite validation scans all real/complex components, atomically updates
one retained status word and downloads four bytes. Native zero fill uses the
CUDA driver memset operation, correctly replacing NaNs and infinities without
multiplying their old values. Both hooks validate storage and stream ownership;
errors propagate rather than selecting the unsupported fallback.

Block initialization uses these hooks and allocates full host validation/zero
scratch lazily only when a hook is unavailable. The native CUDA path therefore
avoids full-vector input downloads and zero-guess uploads. Direct tests cover
f32/f64/c32/c64, empty/single/tail lengths, maximum finite values, non-finite real
and imaginary components, foreign-stream rejection and zeroing of invalid
values. Solver tests require invalid input to preserve output and bound valid
257-row solve transfers independently of vector length, allowing only compact
status and coefficient data.

All five focused tests and production block parity pass CUDA memcheck with
zero errors. All native room parity/cancellation/failure lanes pass; CPU compute
tests pass 650, and root check/Clippy plus CUDA compute/room test Clippy pass.
The four production results, residuals, correction counts and 249/5/8/5 device
allocation counts are unchanged. H2D traffic drops by 2048 bytes per frequency;
D2H drops by 2040 bytes after accounting for compact status reads. Producer
physical diagnostics, host source callbacks and compact Krylov coefficients
remain separate boundaries; this is not whole-application residency.
Evidence:
[cuda-block-native-initialization-2026-10-21.json](cuda-block-native-initialization-2026-10-21.json).


### Active-prefix Krylov transfers

CUDA now implements checked contiguous and strided host/device view copies
without staging the complete device allocation. Scalar and block GMRES transfer
only active projection and solution coefficients. Scalar GMRES retains the
largest reduction buffers across shorter cycles, avoiding two allocations when
the iteration budget shrinks. Unused tails remain untouched; native operations
consume only their active basis count.

Tests cover real/complex scalar types, strided and empty views, unchanged data
outside the view, invalid lengths, injected transfer failures, foreign streams,
and a retained scalar workspace with budgets 64/4/64. The latter requires no
warm allocations and bounds coefficient traffic independently of reserved
capacity. These tests, block recovery and production block parity pass CUDA
memcheck with zero errors. The 650 CPU compute tests and all nine native room
parity/cancellation/failure tests pass; root check and Clippy pass.

| Frequency (Hz) | H2D bytes before → after | D2H bytes before → after | Allocations |
| --- | --- | --- | --- |
| 80 | 5292 → 2348 | 121652 → 28532 | 249 |
| 100 | 3944 → 1008 | 122352 → 28736 | 5 |
| 120 | 4972 → 2044 | 123572 → 29460 | 8 |
| 140 | 3944 → 1048 | 127152 → 31088 | 5 |

All four block results remain accepted with one promoted correction and exactly
the same reported residuals. This measures transfer volume, not a wall-clock
speedup. Host Hessenberg algebra, physical diagnostics, source callbacks and
remaining warm allocations are still separate optimization targets. Evidence:
[cuda-krylov-prefix-transfers-2026-10-21.json](cuda-krylov-prefix-transfers-2026-10-21.json).


### Retained scalar and block Jacobi workspace

Allocation backtraces identified repeated replacement of point-Jacobi scratch
when block Arnoldi switches to scalar recovery. Point Jacobi now retains scalar
scratch separately from the latest multi-RHS block scratch. Single-column block
calls share scalar scratch. The room block producer updates an existing
Jacobi diagonal in place on scheduled refresh instead of reconstructing its CSR
and descriptors. The preconditioning algorithm and refresh cadence are unchanged.

The four-frequency production allocation counts fall from 249/5/8/5 to
248/3/0/3. In particular, the scheduled refresh at 120 Hz performs zero device
allocations. Its H2D/D2H traffic falls by 516/260 bytes; other frequencies retain
the previous transfer counts. Residuals and promoted correction counts are
unchanged. Remaining warm allocations were traced to increased scalar Krylov
demand: one vector and two device reduction pointer buffers. The production
regression now allows at most three warm allocations on this fixture.

Direct CPU/CUDA tests alternate scalar, single-column and two-column application
across diagonal changes for f32/f64/c32/c64 and require zero allocations after
warmup. A failed first block allocation preserves output and retries correctly.
Focused tests and production parity pass CUDA memcheck; 650 CPU compute tests,
all nine native room tests, root check/Clippy and CUDA compute/room test Clippy
pass (existing lint warnings remain). Temporary allocation tracing was removed.

This retains one additional scalar scratch vector; changing the multi-RHS width
can still resize block scratch. It does not establish complete CUDA residency or
a measured elapsed-time speedup. Evidence:
[cuda-jacobi-workspace-2026-10-21.json](cuda-jacobi-workspace-2026-10-21.json).


### Production shifted-AMG block solves and mesh-resolution parity

The pure-f32 CUDA room producer now admits shifted-Laplacian AMG in its retained
block source solve. Preconditioner setup occurs once at the frequency boundary,
and both block solves and scalar recovery consume the same cached object.
Scheduled point-Jacobi refreshes likewise update one object before either path.
AMG setup failure selects the existing Chebyshev fallback and clears the old
hierarchy, preventing a failed refresh from also executing a stale AMG solve.
Setup time is now included in preconditioner timing.

The refined regression exposed a mesh-resolution mismatch: CUDA used the fixed
six-elements-per-wavelength default while CPU honored the configured value.
CUDA now honors `FemMeshConfig::elements_per_wavelength`; parity thresholds were
not relaxed. Native logs prove two AMG levels on the 64-node/162-element mesh
and three levels on the 150-node/480-element mesh at twelve elements per
wavelength. Both meshes pass the four-frequency CPU field/physical-work parity
and reciprocity gates, with accepted block results and one promoted correction
at every frequency.

| Fixture | Iteration totals at 80/100/120/140 Hz | Device allocations |
| --- | --- | --- |
| 64 nodes, point Jacobi baseline | 112/114/116/120 | 248/3/0/3 |
| 64 nodes, shifted AMG | 44/44/44/44 | 232/0/42/0 |
| 150 nodes, shifted AMG | 44/44/46/48 | 255/0/68/0 |

The cached-frequency tests require zero device allocations and age/rebuild
metadata consistent with reuse. Maximum reported residual is below 4e-7.
Both AMG tests pass CUDA memcheck with zero errors; all eleven native room
parity/cancellation/failure tests pass. CPU room tests pass 331 with one ignored;
root check/Clippy and CUDA room test Clippy pass with existing lint warnings.

AMG hierarchy construction remains host setup, and scheduled rebuilds still
allocate. Block AMG applies resident columns sequentially through its existing
cycle workspace. This closes the production block dispatch gap for AMG, not
multivector AMG tuning, all precision/preconditioner combinations or large-mesh
scalability. Iteration reduction is not a wall-clock speedup claim. Evidence:
[cuda-room-amg-block-2026-10-21.json](cuda-room-amg-block-2026-10-21.json).


### Polynomial-Jacobi and Chebyshev block integration

The pure-f32 room producer now connects the existing polynomial-Jacobi and
Chebyshev device block actions. All currently admitted CUDA preconditioner
choices can enter block execution; the same per-frequency preconditioner serves
scalar recovery if block certification is rejected. Setup is performed once
before either path and included in preconditioner timing. The requested
preconditioner and residual/physical acceptance gates are unchanged.

Both polynomial implementations now retain separate scalar and multi-RHS
scratch, avoiding shape-driven replacement during block rank recovery. The
single-column block route reuses scalar scratch. Direct CPU/CUDA tests compare
block application against scalar application on a 257-row tridiagonal matrix
for f32/f64/c32/c64, alternate block/scalar/single-column calls, and require zero
warm allocations. Existing preconditioner failure tests still pass.

| Degree-three preconditioner | Production iterations (80/100/120/140 Hz) | Allocations before separate scratch | Allocations after |
| --- | --- | --- | --- |
| Polynomial Jacobi | 70/72/36/38 | 227/16/16/19 | 224/13/13/16 |
| Chebyshev | 30/30/58/30 | 224/22/22/22 | 219/17/17/17 |

All production block results are accepted, with unchanged residuals below 1e-6.
Both routes pass physical/field parity, reciprocity and CUDA memcheck. Four
focused scratch tests pass memcheck; all thirteen native room tests pass,
including cancellation and injected failures. CPU compute tests pass 650 and
room tests pass 331 (one ignored). Root check/Clippy and CUDA compute/room test
Clippy pass with existing warnings.

The preconditioners still borrow the shifted matrix and reconstruct once per
frequency. Closing that persistence gap requires an owned operator/update
contract; this change does not claim zero whole-frequency allocations. Separate
scratch retains three additional scalar vectors for polynomial Jacobi and five
for Chebyshev. Changing block width can still resize scratch. These bounded
results do not make Chebyshev appropriate for arbitrary indefinite systems or
establish an elapsed-time speedup. Evidence:
[cuda-room-polynomial-block-2026-10-21.json](cuda-room-polynomial-block-2026-10-21.json).


### Owned polynomial operators and cross-frequency numeric refresh

Polynomial Jacobi and Chebyshev now accept an owned `PersistentDeviceCsr` as
well as their existing borrowed API. Numeric refresh copies device matrix values
and uploads only the inverse diagonal, retaining sparse descriptors and scalar/
block scratch. Chebyshev also recomputes its host spectral bound from the current
numeric matrix, preserving the previous algorithm. The CUDA room producer caches
both preconditioners across frequencies and refreshes their numeric data.

The shared matrix wrapper validates host diagonal/shape before mutation and
marks an owned operator invalid while refreshing. A backend error after matrix
replacement cannot expose a mixed matrix/diagonal to later applications. After
backend recovery, a complete successful refresh is required. Tests inject a
diagonal-upload failure after matrix copy, verify rejection and preserved output,
and retry. The degree-one shortcut is tested explicitly because it does not
otherwise touch the matrix action.

| Route | Allocations before (80/100/120/140 Hz) | Allocations after |
| --- | --- | --- |
| Polynomial Jacobi | 224/13/13/16 | 227/0/0/3 |
| Chebyshev | 219/17/17/17 | 222/0/0/0 |

Iterations and residuals are unchanged. Production gates now require these warm
allocation bounds. The remaining three polynomial allocations reflect increased
Krylov/reduction capacity, not preconditioner reconstruction. Direct CPU/CUDA
numeric-update tests compare retained and fresh preconditioners for all four
scalar types, require zero warm allocations, and account for diagonal-only H2D
traffic with no warm D2H. Focused and production tests pass CUDA memcheck. All
thirteen native room tests, 650 CPU compute tests and 331 CPU room tests pass
(one room test ignored). Root check/Clippy and CUDA compute/room test Clippy pass
with existing warnings.

The producer retains an additional owned CSR alongside the live shifted operator;
the memory estimate includes that copy. Chebyshev spectral estimation, host
numeric assembly used for that estimate and diagonal uploads remain recurring
host work. This closes descriptor/scratch reconstruction for these routes, not
whole-application residency or elapsed-time performance. Evidence:
[cuda-owned-polynomials-2026-10-21.json](cuda-owned-polynomials-2026-10-21.json).

### Resident dense BEM solve and LU refresh

The CUDA complex LU API accepts a resident column-major matrix and retains
factors, pivots and vendor workspace for numeric refresh. A singular or failed
refresh invalidates factors until a successful refresh; invalid input rejected
before mutation preserves the prior factorization. Native tests cover forward,
transpose and Hermitian solves, singular-factor rejection, injected-copy failure
and recovery, with zero CUDA memcheck errors.

The small constant-Tri3 BEM session now uses this API and publishes only a
residual-certified device solution. It compares exact matrix coefficients to
reuse LU for RHS-only changes. Matrix assembly still runs each call; avoiding
that repeated work is a separate optimization. Native solve testing exposed and
fixed the column-major BEM versus row-major compute BLAS layout mismatch in
both scalar and block actions. This does not establish compressed, higher-order
or full BEM application coverage.

### Public BEM CUDA producer integration

`BemSolver::prepare_cuda` now returns a backend-bound `CudaBemSession`.
`solve_device` preserves the public solver's material preparation, quadrature
admission, Burton–Miller coupling and incident-field semantics. It retains the
certified surface vector on CUDA; `download_solution` explicitly creates the
existing `BemSolution` for CPU field/output APIs. Incident evaluation now runs on CUDA for all four source variants; only source
descriptors are uploaded. Material/boundary coefficient preparation remains on
the host.

Native validation covers rigid and impedance boundaries with Burton–Miller on
and off, for 12 solves across frequency and incident-amplitude changes. Warm
calls allocate no device buffers. With native incident evaluation, they upload
12,848 bytes for the 80-panel single-source fixture: boundary weights plus one
48-byte source descriptor. Tracked explicit downloads are 32 bytes when
refactoring and 24 bytes when reusing factors, plus cuBLAS norm scalars. Surface relative errors are below 1e-15; three field points per solve
pass the 1e-9 CPU-reference tolerance. CUDA memcheck reports zero errors.

Changed geometry or quadrature schedules reject the solve and unpublish its
output; restoring compatible inputs recovers the session. Compressed,
higher-order and soft-indirect routes reject explicitly. Per-session resource
budgets also reject pending CUDA memory estimates rather than silently using
the host-only estimate; compute backend quotas remain available. No large
system is converted to dense storage. See
[public-session evidence](cuda-bem-public-session-2026-10-21.json).

### Native incident RHS evaluation

The public CUDA session now retains collocation positions/normals and evaluates
plane-wave and point-source pressure plus normal derivative on device. Single
and multiple sources share a six-f64 descriptor layout. Source order, harmonic
phase, interior/exterior normal sign and the CPU point-source cutoff are
preserved. Empty source lists yield a zero RHS; invalid kinds, counts and
non-finite inputs fail explicitly.

Native tests compare all four source variants and empty sources against the CPU
RHS at 131 collocation points, including coincidence and near-cutoff points,
both harmonic signs and both tau signs. The 12 public scattering solves still
pass surface/field parity and zero-warm-allocation gates. CUDA memcheck reports
zero errors. The single-source incident upload falls from 1,280 to 48 bytes
for 80 panels, with no elapsed-time speedup claimed. Source capacity is retained
and can grow when needed. Material preparation, boundary weights and host field
output remain separate residency work. See
[incident evaluation evidence](cuda-bem-incident-2026-10-21.json).

### Retained tetrahedral pressure sessions

`CudaTetP1PressureSession` and `CudaTetP2PressureSession` now prepare affine H1
volume operators, prescribed-pressure constraints, a shifted auxiliary operator
and CUDA AMG/FGMRES state. A canonical K-M representation supports wavenumber
updates through the existing numeric assembly kernel. The fused element action
now accepts global coefficient scales consistently for action, Hermitian action,
lifting, residual and smoothing; constraint identity rows remain unchanged.

CSR and element-action modes share the same solve/preconditioner configuration.
On the recorded cube fixtures, both modes take 23/22/22 iterations for P1 and
37/36/36 for P2 across three frequencies. Warm solves make zero device allocations;
frequency updates make zero allocations and no host transfers. The tests compare
solutions and updated Hermitian actions to independent FEM quadrature assembly,
exercise explicit AMG shift rebuild and injected-update failure recovery, and
retain the legacy tetrahedral solve gates. All three native tests pass CUDA
memcheck with zero errors.

The auxiliary hierarchy stays lagged across frequency updates until explicitly
rebuilt; its setup remains on the host. The session currently covers volume and
fixed Dirichlet/natural homogeneous Neumann data. Robin/facet integration, curved
geometry, generic admission, block workflows and complete-solve performance
comparisons remain open. Element mode retains CSR storage for this matched
baseline, so no memory-saving or elapsed-time speedup is claimed. See
[tetrahedral session evidence](cuda-tet-pressure-session-2026-10-21.json).


### Generic assembled tetrahedral admission — 2026-10-22

The public workload contract now admits H1 tetrahedral P1/P2 Complex64
Helmholtz systems through `CudaFemSolveSession`, for forward, transpose and
Hermitian actions. This connects host form assembly and constraints to retained
CUDA CSR, FGMRES and caller-prepared shifted auxiliary AMG. It does not admit
generic GPU assembly, matrix-free/partial-assembly callbacks, higher orders or
other scalar types. Specialized element sessions retain their separate contracts.

Six physical assembly/action fixtures pass three numeric/RHS solve states each.
Independent original-system residuals stay below 1e-9; repeated solves and
updates allocate no device buffers. Updates consume already-device-resident
values/RHS and download two validation status words (8 bytes); initial host
staging is excluded from that update measurement. Unsupported workloads reject
before assembly or preconditioner callbacks, with zero kernels, allocations or
transfers. Failure recovery and stale-output rejection use the shared session
fixture. The capability inventory now contains 34 rows, including six new
positive tetrahedral cells; it is still not a complete workspace inventory.

Both native tests pass CUDA memcheck with zero errors. CPU validation passes
1,947 FEM library tests (one ignored) and five admission integration tests.
Workspace check, workspace Clippy and CUDA FEM Clippy complete with existing
warnings. See [tetrahedral admission evidence](cuda-generic-tetrahedral-admission-2026-10-22.json).


### Complex triangular transpose admission — 2026-10-22

P1/P2 H1 triangular Complex64 Helmholtz requests now admit transpose solves
through the retained assembled session. Host setup canonicalizes A-transpose
once, then CUDA executes CSR actions, shifted auxiliary AMG and FGMRES. This
uses the existing session implementation; it does not add a transpose method to
the separate `DeviceAdjointOperator` trait. The capability catalog replaces the
obsolete P1 rejection and adds the P2 transpose row (35 rows total).

Physical P1/P2 assembly tests now cover all three actions, each across three
numeric/RHS states. A separate nonsymmetric complex fixture checks all actions
through both triangular workload contracts, distinguishing transpose from
Hermitian using independent matrix-vector products and a known complex solution.
The new physical and admission tests failed before the dispatch change, confirming
they exercise the formerly rejected route. Host assembly and AMG setup remain
explicit; broader workload coverage remains open.

Five generic native CUDA regression tests and six admission integration tests
pass. CUDA memcheck reports zero errors. The CPU FEM library suite passes
1947 tests (one ignored); workspace check, workspace Clippy and CUDA FEM Clippy
complete with existing warnings. Repeated physical solves retain zero warm
allocations. See [triangle transpose evidence](cuda-triangle-transpose-2026-10-22.json).


### Generic assembled hexahedral admission — 2026-10-22

`FemWorkload::native_cuda_hex` connects H1 Q1/Q2/Q3 Complex64 Helmholtz
requests to the shared retained `CudaFemSolveSession` for forward, transpose
and Hermitian solves. The new explicit `FemOrder::P3` denotes cubic order
(Q3 for hexahedra); it does not widen the ambiguous `P2OrHigher` bucket or
admit cubic triangles/tetrahedra. The generic capability catalog has nine new
positive cells and retains an explicit f64 hex rejection (44 rows total).

Host assembly and constraints supply the high-order physical CSR matrix. The
caller prepares low-order-refined shifted-Laplacian AMG on the same global DOF
grid, retaining native complex coarse LU and FGMRES storage. The physical
operator remains the target for independent residual certification; the auxiliary
operator is only a preconditioner. Generic partial assembly and matrix-free
callbacks still reject; the specialized hex partial-assembly session has its
own contract. This admission extension does not claim GPU form integration.

Nine native physical cases on eight sheared hexahedra (27/125/343 DOFs) pass
three numeric/RHS states each, with independent original-system residuals below
1e-9 and zero warm device allocations. The Q3 cases require 77–78 iterations,
exercising retained restart storage beyond the restart length of 60. Updates
consume device-resident values/RHS and download 8 validation bytes, excluding
initial host staging. Failure recovery and stale-output rejection use the shared
session fixture. Unsupported requests reject before both callbacks with zero
device operations.

Both native tests pass CUDA memcheck with zero errors; seven admission tests
and 1,948 CPU FEM tests pass (one ignored). Workspace check, workspace Clippy
and CUDA FEM Clippy complete with existing warnings. See
[hexahedral admission evidence](cuda-generic-hexahedral-admission-2026-10-22.json).


### Retained compressed BEM linear and adjoint sessions — 2026-10-22

`BemSolver::prepare_cuda_hmatrix_linear` prepares a separate
`CudaHMatrixLinearSession` for a fixed forward, transpose or Hermitian action.
It reuses the constant discontinuous Tri3 rigid/impedance H-matrix producer,
resident low-rank/dense blocks, sparse diagonal or spatial cluster inverse and
GMRES workspace. The inverse preconditioner is transposed or conjugate-transposed
once during host setup to match the system. The compute layer now exposes checked
raw-buffer transpose/Hermitian adapters for Krylov, with dimension, allocation
extent and stream validation. Setup reserves full restart reduction capacity.

`solve_device` consumes a resident RHS and only publishes a certified device
vector. Invalid RHS, failed kernels, nonconvergence and exact-check failures
unpublish output and prohibit stale warm starts. Native finite validation downloads
a compact status word. The existing optional exact TBEM policy downloads two
vectors and evaluates the corresponding uncompressed host action without a
global dense matrix; its host work is explicit. Disabling that policy leaves
vectors resident but still transfers Krylov control scalars/coefficients.

The linear-session type exposes no scattering-solution or field-output methods:
an arbitrary or adjoint RHS is not assigned incident-field semantics. Physical
field evaluation remains on `CudaHMatrixBemSession`. Geometry, material,
frequency and formulation remain frozen, and block workflows, gradients through
assembly/compression, shared-DOF/higher-order bases and directional FMM remain
separate coverage work. No new kernel-speedup or whole-session memory-saving
claim accompanies this integration.

Native validation covers 72 solve states at 80 panels (rigid/impedance,
Burton–Miller on/off, both preconditioners, all three actions) and six at 320
panels (680 low-rank blocks, transpose/Hermitian, cluster preconditioning).
All repeated solves allocate no device buffers. The 320-panel cases converge
in 13 iterations, with independent residuals at most 6.1e-10; the test oracle
is a separately assembled dense matrix, not a production allocation. With
optional host exact certification disabled, these solves transfer 208 H2D
coefficient bytes and 2,916 D2H control/validation bytes each, excluding initial
RHS staging and test-only solution download. The 20-panel one-block inverse
solves each orientation in one iteration and unchanged warm starts in zero.

Four native linear-session tests and four existing small scattering/field
regressions pass. CUDA memcheck passes raw dense/low-rank action parity plus
small oriented-inverse and failure-recovery session tests with zero errors;
it does not cover the new 320-panel production adjoint case. CPU compute
(651) and BEM (541, three ignored) tests pass; workspace check/Clippy and
CUDA compute/BEM Clippy complete with existing warnings. The private common
setup still retains scattering geometry/source buffers for linear sessions;
removing that unused setup storage is a separate memory optimization. See
[linear/adjoint session evidence](cuda-bem-linear-adjoint-session-2026-10-22.json).

### Retained compressed BEM block sessions — 2026-10-22

`CudaHMatrixLinearSession::solve_block_device` now joins resident column-major
RHS, native forward/transpose/Hermitian compressed actions, oriented Jacobi or
cluster-block inverse SpMM, and restarted block GMRES. The block solution remains
on CUDA and is published only after every original RHS passes an independent
residual check. Optional exact TBEM certification remains explicit host work.
Scalar calls invalidate block publication and vice versa; errors invalidate
previous-solution eligibility. Width changes rebuild block storage. The fixed
geometry/frequency/material/formulation/action constraints remain unchanged.

The compute adapters reuse scalar H-matrix scratch column by column: native
execution is established, but a batched H-matrix kernel and performance gains
are not claimed. Action tests cover mixed dense/low-rank factors, all three
orientations and widths 1/3/2, with zero allocations and zero host transfers.
Foreign inputs, outputs and solver streams reject before device operations.

Production tests exercise 80-panel impedance Burton–Miller systems with both
preconditioners and all three actions, independent two-column RHS and duplicate
plus zero three-column RHS. A 320-panel compressed Hermitian case with cluster
preconditioning converges in 20 iterations and three restarts. Across these
fixtures, 52 zero-start solves include 26 identical-RHS repeats with zero device
allocations; 26 previous-solution starts require zero iterations. Independent
dense residuals pass 3e-7 and compressed residuals pass 1e-9.

All seven native scalar/adjoint/block session tests pass, as do the native
compute adapter test and 541 CPU BEM tests (three ignored). CUDA memcheck reports
zero errors for the compute block adapters and small production failure/recovery
case; the 320-panel block case was not memory-checked. Root check/Clippy and CUDA
compute/BEM Clippy pass with existing warnings. Failure tests cover invalid
widths/streams, NaN, injected launches, exact certification rejection,
nonconvergence, zero RHS and scalar/block publication transitions.

First-use active widths may allocate. Existing raw block-basis and sparse
allocation policies mean the pool is not a whole-session budget. The common
linear setup also retains scalar Krylov and incident geometry alongside block
state. Broader basis/FMM/frequency-update block producers remain open. See
[block session evidence](cuda-bem-block-session-2026-10-22.json).

### Persistent preconditioned MLFMM core solve API — 2026-10-22

`CudaMlfmmC64` now implements the checked scalar `DeviceOperator` contract and
accepts borrowed solver buffers for all three orientations. Shape, allocation
and stream checks run before dispatch. Existing pooled actions delegate to the
same bridge. Native phase, near and far actions retain their workspaces; the
existing synchronization before publishing output is preserved. Multi-RHS trait
calls explicitly reject instead of inheriting host staging.

`MlfmmSystem::try_prepare_cuda_solve` retains the uploaded hierarchy, oriented
Jacobi or leaf-block Jacobi sparse inverse, and reserved GMRES state. Setup
inverts only leaf self-blocks, never a global dense matrix. Missing, overlapping,
singular and non-finite self-blocks reject explicitly. The session exposes a
resident RHS/solution API, certifies the MLFMM-system residual, supports previous
solutions, and invalidates publication on every failed call. Directional and
repeated/shared-DOF admission remains unchanged. Matrix/hierarchy/orientation
updates require a new session.

The physical test uses 320 constant Tri3 panels at ka=1 with genuine far
translations. Independent dense TBEM supplies the RHS and checks the final
physical residual; CPU MLFMM separately checks GPU action/solve parity. Eighteen
zero-start solves cover both preconditioners, all orientations and three RHS
states. Twelve changed-RHS warm cases allocate no device buffers, and eighteen
previous-solution starts take zero iterations. Jacobi takes 38–39 iterations;
leaf-block Jacobi takes 33. These iteration counts are not timing measurements.
The maximum independent TBEM residual is below 9.4e-8 against a 1e-5 gate;
CPU MLFMM residuals pass 2e-9. Compact Krylov/control traffic remains visible.

Three focused BEM tests, native compute parity and 541 CPU BEM tests pass (three
CPU tests ignored). Memcheck reports zero errors for raw/pooled phase-action
parity and the small production failure/recovery session; it does not cover the
320-panel solve. Root check/Clippy and CUDA compute/BEM Clippy pass with existing
warnings. Tests distinguish recoverable launch errors from terminal
synchronization errors, and cover malformed RHS, invalid preconditioners,
nonconvergence, zero RHS and publication invalidation.

This connects the assembled MLFMM core API, not high-level `BemSolver` or
application dispatch. Native block FMM, frequency updates, directional/shared
layouts and general physical-output producers remain open. Sparse descriptor
allocations still have partial quota coverage, and per-action synchronization
remains a performance target. See [MLFMM evidence](cuda-mlfmm-persistent-solve-2026-10-22.json).


### Public retained MLFMM scattering API — 2026-10-22

`BemSolver::prepare_cuda_mlfmm` now connects incident evaluation, retained
MLFMM/GMRES and Jacobi or leaf-block Jacobi directly from a validated BEM problem.
The admitted scope is fixed constant/discontinuous Tri3, rigid or impedance
boundaries, with or without Burton–Miller. Setup uses the host; repeated RHS
and solves run on CUDA. Optional independent exact TBEM certification explicitly
downloads vectors. `download_solution` enables existing host field evaluation.
Failures invalidate publication and previous-solution eligibility.

The default leaf target remains 32, matching the CPU producer. The separate
`prepare_cuda_mlfmm_with_leaf_size` exposes a positive target for tuning. Tests
at 320 and 1,280 panels with the default produced no far translations; these
were rejected as far-coverage fixtures. The final 320-panel test uses target
four and asserts 156 far entries. Plane-wave and multiple-point-source cases
pass independent dense surface/field comparisons and the 1e-6 exact residual
gate. Both repeated zero-start solves take 18 iterations, allocate no device
buffers, and have MLFMM residual below 4.5e-10. Compact source/control transfers
remain; source-count growth may allocate. These are correctness/residency
measurements, not speedup measurements.

Three focused native tests pass, including 80-panel rigid/impedance,
Burton–Miller on/off, both preconditioners, source changes, previous solutions,
rejection and recovery. CPU BEM reports 541 passed and three ignored. Root check
then Clippy and CUDA BEM Clippy pass with existing warnings. Memcheck reports
zero errors for the small public failure/recovery case; the far test was not
memory-checked. Application dispatch, resident fields for this session, block
FMM, directional/shared layouts and operator updates remain open. See
[public MLFMM session evidence](cuda-mlfmm-public-session-2026-10-22.json).


### Speaker forward sweep Krylov retention — 2026-10-22

The speaker application already retained CSR operators, numerical preconditioner
updates, RHS and solution buffers, but rebuilt scalar Krylov scratch at each
frequency. `run_simulation_gpu_with_backend` now owns one scratch pool for the
sweep. Prescribed CUDA drivers use retained mixed-precision IR scratch; typed
CUDA drivers use retained Complex64 GMRES scratch with the existing block-Jacobi
preconditioner. The shared Metal scalar branch uses retained Complex32 scratch.
These workspaces start empty and grow on demand. The pool owns the same backend
used by assembly and operators; no second CUDA context is created.

Existing tolerances, warm-start decisions, CPU residual checks and physical
reference checks remain in place. Scratch quota coverage does not include the
legacy assembly/operator allocations. Host Hessenberg work in IR, adjoint
scratch, the small coupled-network solve and block width-cache admission remain
separate work. This does not extend the application's dense-only GPU admission
to compressed systems, and does not imply allocation-free whole sweeps.

The prescribed-driver whole-application qualification passes with 768 DOFs,
two frequencies and 72 CEA points: true residual 5.48e-7 and CPU-reference field
error 4.07e-6. Both frequency references run without correction. Device and
staging allocations are zero after teardown. Large host/reference transfers
remain, so this is bounded correctness evidence rather than a speedup claim.

The typed-driver production gate also passes at 752 DOFs: true residual
8.63e-7, field error 4.25e-5, coupled power error 5.40e-7 and coupled derivative
error 1.03e-8. The failure-contract gate passes all 17 checks, including native
transfer, launch, synchronization and device-loss lanes. Root check then Clippy
and CUDA speaker Clippy pass with existing warnings. The CPU regression result
is tracked in the [speaker retention evidence](cuda-speaker-forward-retention-2026-10-22.json).

CPU regression validation is **not green**: 233 passed, six ignored and
`batched_fd_gradient_matches_serial` failed. The isolated rerun reproduces a
3.05e-7 difference for a gradient near 169.78. The changed application function
is CUDA/Metal-gated and excluded from this default-feature CPU test build.
This separate CPU gradient regression remains unresolved; its tolerance was
not changed. Metal hardware validation and whole-sweep allocation profiling
were not performed.


### Speaker adjoint and coupled-network Krylov retention — 2026-10-22

The remaining scalar convenience calls in the speaker application now use
retained scratch. The adjoint frequency sweep owns a Complex64 GMRES workspace
and pool alongside its updated Hermitian CSR and block-Jacobi preconditioner.
The typed N-port network has a separate driver-count workspace. Reference
corrections reuse the field-sized final-solve workspace. This closes the scalar
scratch-lifetime gaps recorded in the preceding speaker section, without
extending dense-only application admission or claiming whole-sweep residency.

A native application-helper regression performs five two-driver network solves
at 1000/1000/1200/1200/1000 Hz. All four calls after warm-up allocate zero device
buffers while updating CSR and Jacobi values. Velocities pass the independent
CPU network comparison at 1e-5; CUDA memcheck reports zero errors. The CPU-backed
helper contract passes with an explicit pool capacity, since that backend does
not report device memory.

Whole-speaker prescribed/adjoint and typed-production gates pass. The adjoint
true residual is 9.96e-6, CPU-reference error 1.84e-3 and inner-product error
3.86e-16, within the existing gates. The typed field residual is 8.63e-7 and
field error 4.25e-5. The failure-contract gate again passes all 17 checks.
Root check then Clippy and CUDA speaker Clippy pass with existing warnings.
The prior default-feature CPU gradient failure remains unresolved; this change
does not modify that path or relax its tolerance. No whole-application speedup,
zero-allocation sweep, full quota coverage or Metal hardware result is claimed.
See [adjoint/network retention evidence](cuda-speaker-adjoint-retention-2026-10-22.json).


### Steady heat-conduction admission and retained solves — 2026-10-22

`FemPhysics::HeatConduction` and `FemWorkload::native_cuda_heat_conduction` now
connect assembled f64 H1 Tri3/tetrahedron P1/P2 systems to the public retained
CUDA FEM solver. The new enum variant is appended, preserving existing ordinal
positions. A separate `fem_cuda_thermal_capability_rows` table describes the 12
forward/transpose/Hermitian cells without relabeling acoustic evidence. Other
spaces, scalars, orders and assembly modes reject before setup callbacks.

Production conduction and heat-source forms, plus nonhomogeneous Dirichlet
elimination, run on the host. The tested solver is persistent PCG with symmetric
AMG, positive conductivity and constrained end faces; pure-Neumann nullspaces
need separate treatment. Insulated sides give the analytical P1 profile T=1+x
with zero volumetric source, and P2 profile T=1+x(1-x) with Q=2k. All 12 physical
cases pass: maximum temperature error 2.74e-9 and residual 4.65e-10. Real
transpose/Hermitian cases are symmetry-equivalent, not evidence for thermal
advection.

Thirty-six subsequent changed-value/RHS solves allocate no device buffers and
pass an independent CSR residual gate of 1.1e-9. The shared session regressions
also exercise warm starts, publication invalidation and injected-error recovery.
Native rejection tests verify unsupported requests perform no allocation,
launch or transfer before either callback. CUDA memcheck passes both focused
tests with zero errors. CPU FEM reports 1,963 passed and one ignored. Root check
then Clippy and CUDA FEM Clippy pass with existing warnings.

Host setup and uploads remain explicit. This does not cover transient thermal
history, nonlinear radiation, advection, pure-Neumann handling, other scalar or
element families, or all public thermal application producers. No speedup or
whole-sweep zero-allocation claim is made. See
[thermal admission evidence](cuda-thermal-admission-2026-10-22.json).


### Retained thermal theta integration — 2026-10-22

The existing `ThermalTimeStepper` now offers `prepare_cuda`, returning a
`CudaThermalTimeStepper` for fixed linear mass/stiffness matrices, dt and theta.
Host setup builds the effective matrix and symmetric AMG once. Device mass and
stiffness actions form the history RHS; caller-owned device sources supply the
next/current loads. Persistent PCG certifies the effective-system residual.
A successful candidate copy is synchronized before temperature/history count
is committed. Errors or nonconvergence leave the prior committed state intact
and remove warm-start eligibility. No partial iterate is published as history.

The input contract requires consistently reduced free-DOF systems and an
explicit SPD assumption. Time-dependent boundary lifting is caller-owned.
Changing matrices, constraints or time-step parameters requires preparation of
a new session. The source-current omission matches the CPU API's zero term;
for theta below one, changing sources require the actual current-time load.
Explicit theta stability remains a separate time-integration condition.

Twelve physical cases cover 2-D triangles and 3-D tetrahedra, P1/P2, and theta
0.5/0.75/1. Each runs eight steps: all 96 steps have zero H2D bytes during the
step call, and all 84 steps after first use allocate zero device buffers.
They match the CPU step implementation within 2e-8. Independent continuum
sine-mode decay is checked separately against a 0.025 maximum-error gate on
these coarse meshes (observed maximum 0.021991); this is not a high-resolution
physical accuracy claim. A separate exact affine-temperature forcing case
checks theta 0/0.5/0.75/1 to 1e-10.

Three native tests and CUDA memcheck pass with zero memory errors. Invalid
parameters/CSR, malformed and foreign sources, NaN, nonconvergence and injected
launch/D2D errors exercise rejection or preserved history and retry. CPU FEM
reports 1,963 passed and one ignored. Root check then Clippy and CUDA FEM Clippy
pass with existing warnings. Setup and optional output transfers remain host
boundaries; no whole-application speedup or full allocation-quota claim is made.
This closes a bounded linear transient API path, not nonlinear thermal,
automatic lifting, variable operators/time steps, or all thermal producers.
See [thermal time-session evidence](cuda-thermal-time-session-2026-10-22.json).


### Retained thermal scheme updates — 2026-10-22

`CudaThermalTimeStepper::try_update_scheme` now changes dt/theta without
replacing committed history, mass/stiffness device operators or Krylov buffers.
This extends the fixed-time-step scope above. Host mass/stiffness copies support
new effective-matrix combinations. Effective CSR values refresh at fixed
sparsity; symmetric AMG refreshes matching hierarchy storage or explicitly
reports a preconditioner rebuild. Changed effective sparsity requires a new
session. The caller renews the effective operator's SPD assumption.

Updates preserve temperature and the committed step count. Invalid scheme
parameters fail before device work. After uploads begin, failure blocks further
steps until a complete scheme update succeeds following backend recovery;
partially changed matrix/preconditioner state cannot enter a solve.

Native P2 triangular and tetrahedral diffusion tests run five successive
schemes each, including backward Euler, Crank–Nicolson, intermediate theta and
an explicit step. Maximum CPU-step discrepancy is 9.16e-10. Ten repeated
identical updates allocate no device memory; all ten subsequent solves upload
zero bytes, and eight post-first solves allocate no device memory. Different
schemes exercise both AMG refresh and hierarchy rebuild, so arbitrary updates
are not claimed allocation-free. Injected upload failure preserves history,
blocks stepping and recovers through a complete update and an exact scalar gate.

All five native thermal tests pass; both new tests pass CUDA memcheck with
zero errors. Root check then Clippy and CUDA FEM Clippy pass with existing
warnings. CPU FEM reports 1,963 passed and one ignored.
[Evidence](cuda-thermal-scheme-refresh-2026-10-22.json).

Matrix combination and hierarchy construction remain CPU setup work, and
scheme updates upload numeric data. Nonlinear material/geometry updates,
automatic boundary lifting and stability control remain open. This is retained
variable-step integration for fixed linear thermal operators, not adaptive
error control or complete transient FEM coverage.


### Consistent sparse input admission — 2026-10-22

Host CSR construction now rejects a nonzero initial row offset with
`HostCsrError::NonzeroRowStart`. The panicking constructor delegates to the
checked constructor, and transpose/Hermitian preparation inherits the check.
This closes a host preparation inconsistency: device CSR uploads already
rejected this malformed structure, while host validation previously accepted
unused leading entries. Empty valid shapes and dimension overflow retain
explicit regression coverage.

All 654 CPU compute tests pass. Ten CUDA-enabled host CSR tests pass, including
12 malformed-upload cases across four scalars and ordinary, persistent and
adjoint CSR entrypoints. Every rejection performs zero device allocations,
launches and host/device transfers. Root check then Clippy and CUDA compute
Clippy pass with existing warnings. [Evidence](cuda-csr-admission-2026-10-22.json).
This strengthens a shared admission gate; it adds no physics coverage and makes
no performance claim.


### Device effective-matrix combination for thermal updates — 2026-10-22

The retained thermal scheme-update route now connects resident numeric bases
to `PersistentDeviceCsr::try_combine_scaled_values`. It forms `M/dt + theta*K`
through device copy, scale and AXPY, preserving physical CSR storage and avoiding
its previous effective-value upload. The existing unscaled combination method
keeps its original call sequence. Setup aligns/coalesces mass and stiffness
entries to the effective pattern and retains two pooled f64 arrays, costing
16 bytes per effective nonzero. Host matrices still support AMG construction.

This extends kernel priority 3's fixed-pattern update integration. Host effective
combination and AMG hierarchy setup remain necessary for current preconditioning;
AMG still uploads refreshed hierarchy values and can rebuild changed storage.
There is no whole-update zero-transfer or wall-clock speedup claim.

All 655 CPU compute tests and 1,963 CPU FEM tests pass (one FEM test ignored).
Five native thermal tests pass, including ten variable-scheme physical steps
with maximum CPU discrepancy 9.16e-10. Both refresh/recovery tests pass CUDA
memcheck with zero errors. Device-copy and AMG-upload failures preserve history
and block stepping until a complete repair. A custom-kernel launch injection
was found inapplicable to cuBLAS scale/AXPY and is not counted as coverage.
Root check then Clippy and CUDA compute/FEM Clippy pass with existing warnings.
[Evidence](cuda-thermal-device-combination-2026-10-22.json).


### Resident thermal scheme updates with retained AMG — 2026-10-22

`CudaThermalTimeStepper::try_update_scheme_lagged` explicitly retains the current
stationary symmetric AMG action while changing the physical effective matrix
on CUDA. This removes host matrix/hierarchy construction and hierarchy uploads
from that update path. The caller renews the physical SPD assumption. The
ordinary `try_update_scheme` still refreshes AMG; it also repairs failed updates.
Lagging can increase iterations, so it is opt-in rather than a changed default.

Native finite validation checks the combined matrix and returns a 4-byte status.
Device-copy and validation-launch failures, plus arithmetic overflow, preserve
committed history and block steps/lagged updates until complete refresh succeeds.
No host fallback is used. Both new tests pass CUDA memcheck with zero errors;
eight release native thermal tests pass. CPU FEM reports 1,963 passed and one
ignored. Root check then Clippy and CUDA FEM Clippy pass with existing warnings.

A release ABBA comparison, repeated after builds and CPU tests finished, uses
P2 triangular and tetrahedral diffusion with fixed operators and changing dt/theta.
All 80 solves match the CPU trajectory within 2e-8 and satisfy 1e-10 residual
controls. The timing window contains four changed-scheme steps, each including
matrix/preconditioner update and committed PCG solve; preparation, host assembly
and the first step are excluded. Four trajectories per mode are compared:

| Physical case | Fresh AMG median | Retained AMG median | Warm update + solve speedup |
| --- | ---: | ---: | ---: |
| 2D P2, 3,969 free DOFs | 155.23 ms | 48.57 ms | 3.20x |
| 3D P2, 3,375 free DOFs | 208.97 ms | 62.72 ms | 3.33x |

All 32 scored retained-AMG steps perform zero H2D uploads and zero device
allocations. Iterations increase only slightly for this schedule; results do
not establish a universal lagging policy or tolerance for larger changes.
[Full measurements and source hashes](cuda-thermal-lagged-amg-2026-10-22.json).

The retained numeric update takes about 0.025 ms versus roughly 12–16 ms for
its solve. Another combination kernel is therefore low priority on these cases.
Preserving suitable preconditioning across repeated work has a measured benefit;
further kernel tuning should target measured solve cost instead. Other physics,
nonlinear material updates and automatic boundary lifting remain open.


### Retained scalar magnetostatic solves — 2026-10-22

`FemPhysics::MagneticScalarPotential` and
`FemWorkload::native_cuda_magnetic_scalar_potential` now admit assembled f64 H1
Tri3/tetrahedron P1/P2 through `CudaFemSolveSession`. The separate
`fem_cuda_magnetic_scalar_capability_rows` exposes twelve forward/transpose/
Hermitian rows. Existing enum discriminants are preserved by appending the new
physics value. Other spaces/scalars/orders and unassembled modes remain rejected.

Production permeability and permanent-remanence forms assemble a mesh-aligned
two-layer material. Prescribed end potentials remove the constant nullspace;
natural side conditions impose zero normal flux. Positive permeability feeds
PCG with stationary symmetric AMG. Three remanence amplitudes reuse each
physical operator and preconditioner. Independent piecewise-linear potential
and constant normal-flux references validate all element/order/action rows,
including production B-field reconstruction on the host.

All 36 physical solves pass: maximum potential error 3.05e-10, magnetic-flux
error 4.54e-9 and independent relative residual 8.05e-10. Twenty-four repeated
source solves allocate no device memory, and all physical solves upload no
internal bulk data (new source input uploads occur before the measured boundary).
Thirty-six additional generic numeric-refresh solves allocate no device memory.
Nine unsupported requests reject before assembly/preconditioner callbacks or
device work. Both native tests pass CUDA memcheck with zero errors. Root check
then Clippy and CUDA FEM Clippy pass with existing warnings; CPU FEM reports
1,964 passed and one ignored.
[Evidence](cuda-magnetic-scalar-admission-2026-10-22.json).

This closes a bounded linear scalar-potential execution route. Form assembly,
constraints and field recovery remain host work. Nonlinear B-H laws,
axisymmetric/vector potentials, eddy currents, higher orders, other elements
and application dispatch require separate integration and evidence.


### Resident magnetic centroid fields — 2026-10-22

`CudaMagneticCentroidField` now connects the retained scalar-potential solution
to CUDA field recovery. Host preparation tabulates a rectangular gradient CSR
weighted by per-element positive scalar permeability. Persistent SpMV and the
existing native residual update compute `B = Br - mu*grad(phi)` without a host
potential round trip. Remanence can be replaced from a resident buffer; geometry,
space and permeability stay fixed. Output is element-major, with two in-plane
components for affine triangles and three for affine tetrahedra.

All 36 physical magnetic solve cases now evaluate resident fields before the
potential is downloaded for independent test verification. GPU fields match
production CPU recovery within 1.0e-15 and retain the analytical layered-flux
gate. Twelve additional P1 affine/P2 quadratic potential cases independently
check centroid gradients within 1e-12. Twenty-four repeated material-source
update/field evaluations and eight repeated polynomial evaluations allocate
nothing. Update plus evaluation uploads zero bytes and downloads exactly three
4-byte validation results; evaluation alone downloads two. Requested output
downloads occur outside those counters.

Seven malformed/unsupported preparation cases and two wrong-buffer cases reject
before device work. Failed native validation unpublishes output; failed remanence
copy additionally blocks evaluation until a complete replacement. All four
native tests pass CUDA memcheck with zero errors. CPU FEM reports 1,964 passed
and one ignored. Final root check then Clippy, and focused CUDA FEM check/Clippy,
pass with existing warnings after unrelated workspace changes settled.
[Evidence](cuda-magnetic-resident-fields-2026-10-22.json).

This closes centroid output for the bounded scalar-potential route. Arbitrary
field points, coil/force reductions, permeability refresh, curved geometry and
other electromagnetic formulations remain open. Sparse setup remains on the
host and legacy CSR allocations are not fully covered by the pool quota.

### Common retained BEM scattering dispatch — 2026-10-22

`BemSolver::prepare_cuda_scattering` now selects the existing retained dense,
H-matrix or non-directional MLFMM session. `CudaScatteringOptions` explicitly
selects point or spatial-block preconditioning and the MLFMM leaf target.
Dense admits Direct or GMRES with explicit Jacobi; compressed representations require GMRES. The API
retains representation-specific certification reports and exposes a certified
resident solution or an explicit host download. Invalid input or solve failure
unpublishes the result and disables previous-solution reuse until a successful
zero-start solve.

The common reuse contract fixes geometry, frequency, material, basis and
formulation while allowing incident changes. This does not replace the broader
numeric-update contract of the dense-specific API. Unsupported representations
and incompatible solver methods return errors without CPU fallback. Auto uses
the existing assembly threshold; its 1000–1279-panel dense-limit gap remains an
explicit rejection, covered by a 1000-panel admission regression.

Four native CUDA tests pass. Eight small configurations cover Auto/TBEM/H-matrix/
MLFMM and both preconditioner choices, with plane and point sources, CPU surface
and field parity, factor/warm-start reuse, zero warm allocations, NaN rejection
and injected launch-error recovery. A 320-panel MLFMM test requires nonzero far
translations. A 1280-panel Auto test requires H-matrix low-rank blocks, an exact
residual below 1e-6 and CPU surface/field parity below 1e-5. Both larger tests
verify zero allocations for a repeated zero-start solve.

CPU BEM tests pass (541 passed, 3 ignored). Root `cargo check`, then `cargo clippy`,
and CUDA BEM Clippy pass with existing warnings; none name the new dispatcher
files. Evidence: [dispatch validation](cuda-bem-scattering-dispatch-2026-10-22.json).
Application migration, higher/shared bases, directional representations and
compressed numeric updates remain open. Host setup, optional exact certification,
source-buffer growth and incomplete pool-quota accounting remain explicit
boundaries. This change makes no new speedup claim.

Application integration follow-up: `numcalc::run_project` calls the lower-level
`frequency_sweep` with project elements, rather than `BemSolver::solve`. It also
uses per-frequency project physics for evaluation-grid pressure. The new common
incident-only scattering session cannot replace this call without a numeric
refresh/sweep adapter that preserves project velocity/material boundary data,
frequency recycling and evaluation conventions. Keep this producer open in the
coverage inventory. Similarly, speaker applications have their own forward and
adjoint assembly/solve state; common scattering API tests do not demonstrate
that those callers select this path.

The dense iterative sweep can reuse `CudaTbemDeviceSystem` directly: it already
implements checked device scalar and block actions. `CudaTbemSolveSession`
retains assembly and LU storage across numeric changes, but is explicitly direct
and limited to fewer than 1000 DOFs. Do not use it to silently satisfy a GMRES or
GCRO-DR request. A retained iterative sweep needs native diagonal extraction/
inversion and numeric preconditioner refresh (the current `PointJacobiBackend`
constructor/update accept host diagonals), plus device incident evaluation and
an explicit policy for quadrature-schedule changes. Subspace recycling requires
its own implementation/admission row; previous-solution reuse is not equivalent.

### Native dense inverse-diagonal refresh — 2026-10-22

`CudaBackend::try_dense_inverse_diagonal` now extracts and inverts the diagonal
of a resident padded square matrix for f32/f64/Complex32/Complex64. Row-major
and column-major matrices share the same diagonal offsets. Shape, byte extents,
stream ownership and native module availability are checked before launching.
The kernel rejects non-finite and epsilon-small diagonal values through a
compact native finite check. Complex reciprocals scale components before
forming their squared norm to avoid overflow. Output storage belongs to the
caller and must not be published after an error.

All-scalar native tests pass, including zero warm allocations, zero warm H2D
bytes, padded layout, zero/NaN/infinite/epsilon-small rejection, malformed shape
and a complex diagonal with 1e300 components. CUDA memcheck passes with zero
errors. CPU compute tests pass (655), as do root check followed by Clippy and
CUDA compute Clippy, with existing warnings. An initial NVRTC failure from an
external math-header include was corrected by using runtime compiler built-ins.
Evidence: [native diagonal refresh](cuda-dense-inverse-diagonal-2026-10-22.json).

This supplies the missing native numeric primitive for dense iterative BEM
preconditioning. Connecting it to retained preconditioner/session lifecycle and
frequency-sweep producers remains open; kernel correctness and allocation
checks are not an end-to-end sweep or speedup claim.

### Retained dense CUDA Jacobi preconditioning — 2026-10-22

`CudaDenseJacobi` connects native diagonal extraction to the shared
`DevicePreconditioner` contract. It retains one pool-allocated inverse buffer,
refreshes directly from a resident dense matrix, and applies scalar or
column-major block RHS vectors with the native diagonal-scaling primitive.
Every failed refresh disables application until a complete successful refresh;
backend recovery alone does not make stale preconditioner data usable.

All four scalar types pass three numeric states with two RHS columns, zero
allocations and zero H2D bytes inside measured update/application calls, invalid
diagonal rejection and injected launch-failure recovery. The combined primitive
and preconditioner tests pass natively and under CUDA memcheck (2 passed, zero
errors). CPU compute tests pass (655), root check then Clippy pass, and CUDA
compute Clippy passes with existing warnings and none in this module.
Evidence: [retained Jacobi refresh](cuda-dense-jacobi-refresh-2026-10-22.json).

This supplies native point preconditioning to the planned retained dense BEM
GMRES path; the BEM session and sweep producer still need integration. Point
Jacobi is not a replacement for stronger block/ILU/auxiliary preconditioners
where those are needed for convergence. No new solve-time speedup is claimed.

### Retained device-assembled TBEM GMRES — 2026-10-22

`CudaTbemGmresSession` connects `CudaTbemAssemblyWorkspace`, native
`CudaDenseJacobi` numeric refresh and `DeviceGmresSession`. The matrix, inverse
diagonal, RHS and Krylov vectors remain on CUDA. Frequency and boundary values
can change while geometry, DOF order and quadrature schedule remain fixed. An
optional resident incident RHS is added to the assembled boundary load. Warm
starts reuse a certified previous solution, and all failure paths unpublish
output until a successful zero-start solve. Nonconvergence returns an error.

The native fixture uses 80 constant Tri3 panels with pressure, velocity and
velocity/admittance boundaries. Four frequency/boundary states, alternating a
resident incident contribution, match dense LU within 1e-7 and independently
satisfy the CPU-assembled system with relative residuals below 5e-11. They take
21, 18, 18 and 18 GMRES iterations. The first solve allocates twice; all three
warm numeric-update/solve calls allocate zero times. Warm traffic is 13,088 H2D
bytes (including 12,800 boundary-coefficient bytes) and about 5,492 D2H bytes
for compact Krylov/certification data; this is not a zero-transfer claim.

Failure gates cover unavailable warm starts, launch failure, changed geometry,
non-finite incident values and forced nonconvergence, including successful
repair. The final native test passes under CUDA memcheck with zero errors.
CPU BEM tests pass (541 passed, 3 ignored), root check then Clippy pass, and
CUDA BEM Clippy passes with existing warnings and none in this module.
Evidence: [TBEM GMRES numeric refresh](cuda-tbem-gmres-refresh-2026-10-22.json).

This closes the retained iterative core connection for the admitted dense
representation. The public scattering/NumCalc sweep producers still need to
select it. It does not add GCRO-DR recycling, lift the dense memory limit,
replace stronger preconditioners where required, or establish a speedup.

### Public retained TBEM GMRES and common dispatch — 2026-10-22

`BemSolver::prepare_cuda_gmres_jacobi` now prepares the retained iterative core
through the public scattering API. Native incident evaluation feeds the resident
RHS, and the session exposes certified device output or an explicit host
`BemSolution` download. Geometry and quadrature remain fixed while compatible
numeric inputs may change. The method explicitly selects point Jacobi instead
of the CPU preconditioner setting. Common `prepare_cuda_scattering` now admits
TBEM/small Auto with GMRES when `CudaScatteringPreconditioner::Jacobi` is
explicitly requested, and returns `CudaScatteringReport::DenseGmres`. A spatial
block request on this dense route rejects before preparation, without downgrade.

Two native tests cover explicit TBEM/Auto dispatch, plane/point incidents,
rigid/impedance problems with and without Burton–Miller, three frequencies,
CPU surface and three field-point parity within 1e-7, zero warm allocations,
zero-iteration repeated-source warm starts and invalid-source/launch recovery.
Both tests pass CUDA memcheck with zero errors. Existing small direct/compressed
dispatch regressions pass (2 tests); CPU BEM passes (541 passed, 3 ignored).
Root check then Clippy and CUDA BEM Clippy pass with existing warnings and none
in the modified public CUDA files. Evidence:
[public TBEM GMRES](cuda-public-tbem-gmres-2026-10-22.json).

The common interface remains incident-only; numeric changes use the specific
GMRES session. The dense memory limit, unsupported formulations, partial quota
accounting, host coefficient/source uploads and explicit host field output
remain. This does not close NumCalc's frequency-sweep producer or GCRO-DR
recycling, and it makes no new performance speedup claim.

### Retained TBEM quadrature schedule refresh — 2026-10-22

`CudaTbemGmresSession` now replaces its assembly workspace when frequency
changes cross a quadrature schedule boundary. Geometry and DOF ordering remain
fixed. Krylov storage, the inverse-diagonal buffer and the previous solution
survive replacement; native Jacobi refresh uses the new matrix. Both the core
and public dense GMRES session expose `assembly_rebuilds()` so callers can
separate rebuild costs from ordinary numeric refresh. A completed assembly
replacement is counted even if the subsequent solve fails. Replacement prepares
the new assembly before releasing the old one, increasing temporary memory use.

Native tests pass across all four quadrature schedules and a return to the
lowest schedule on a 20-panel sphere. Each result passes an independently
assembled CPU residual gate of 1e-7; repeated calls within each schedule make
zero device allocations. Injected launch failure during replacement unpublishes
output and prevents warm starts until successful zero-start recovery. The
existing mixed-boundary numeric-refresh/recovery test also passes. Both tests
pass CUDA memcheck with zero errors. CPU BEM: 541 passed, 3 ignored. Root check,
root Clippy and CUDA BEM Clippy pass; no warnings identify the modified files.
See [quadrature refresh evidence](cuda-tbem-quadrature-refresh-2026-10-22.json).

This supersedes the fixed-quadrature restriction for the specific GMRES sessions
only. Direct LU and common incident-only scattering contracts are unchanged.
NumCalc integration, GCRO-DR, compressed frequency updates and complete memory
quota accounting remain open. No new speedup is claimed.

### Native retained TBEM frequency sweep — 2026-10-22

`frequency_sweep_cuda_tbem_jacobi` now connects prepared surface elements to
native incident evaluation, retained dense assembly, refreshed point Jacobi
and persistent GMRES across frequencies. It preserves supplied boundary values,
optional per-frequency admittances, density/sound speed, harmonic and interior
conventions, and the configured Burton–Miller beta scale. Source descriptors
upload once; compatible updates retain device storage, while quadrature changes
replace assembly without discarding Krylov storage or warm starts. GCRO-DR is
explicitly rejected. Empty grids return without device allocation.

The result reports preparation, combined update/assembly/preconditioning/solve,
and explicit solution-download times separately, together with iterations,
original-system residual, allocation counts and cumulative assembly rebuilds.
It exports each certified solution once for existing host consumers. This is
a native solve producer with a visible host output boundary; it does not make
field evaluation resident or connect NumCalc dispatch yet.

One native regression covers 28 physical frequency solves on a 20-panel sphere:
seven frequencies spanning four quadrature schedules, both harmonic signs,
interior/exterior physics, point/plane incident sources, mixed prescribed-pressure
and velocity/admittance data, and changing admittance. All solutions match the
existing CPU sweep within relative error 1e-7. Within-schedule warm updates have
zero device allocations. The regression passes CUDA memcheck with zero errors;
CPU BEM passes 541 tests with 3 ignored. CUDA BEM Clippy reports no warnings in
the new module. Evidence: [native frequency sweep](cuda-tbem-frequency-sweep-2026-10-22.json).

The remaining restrictions include ordered contiguous constant-Tri3 DOFs,
fewer than 1000 DOFs, explicit point-Jacobi selection, legacy assembly allocations
outside pool quotas, and unsupported compressed/GCRO-DR sweep variants. These
remain coverage work, not completed rows. No performance improvement is claimed.

### NumCalc retained CUDA producer — 2026-10-22

NumCalc now exposes `run_project_cuda_jacobi` with a caller-owned pool and CLI
selection `--backend cuda-jacobi` (build with `cli,cuda`). It connects the native
retained sweep to the existing prepared project data, range selection and output
layout. The default CPU route and CUDA route share output construction. CUDA
errors propagate without fallback; a CPU-only build rejects CUDA selection.

`SweepOutput.execution` records `cpu` or `cuda_jacobi_with_cpu_fields`; older
JSON without the optional field remains readable. Surface solutions are exported
for existing CPU evaluation-grid processing, so this does not claim a fully
resident output pipeline. The existing first-source selection behavior is
preserved, not broadened by this integration.

Native validation compares surface and two evaluation-grid points against CPU
for full and partial sweeps and checks empty/inverted ranges and serialization.
It passes CUDA memcheck with zero errors. The CLI runs both routes on the sample
four-panel, three-frequency project; relative surface difference is 1.52e-15.
CPU BEM passes 541 tests with 3 ignored. Root check and Clippy, CUDA CLI check
and Clippy, and CPU-only CLI check pass; modified files have no Clippy warnings.
Evidence: [NumCalc producer](cuda-numcalc-producer-2026-10-22.json).

Dense geometry/size restrictions, point-Jacobi suitability, host field evaluation,
compressed frequency sweeps and legacy allocation accounting remain open.

### Device output from retained TBEM sweeps — 2026-10-22

`frequency_sweep_cuda_tbem_jacobi_device` now exposes each certified surface
solution to a callback before the retained buffer is reused. The callback also
receives current physics and boundary metadata, including frequency-dependent
admittances. This entrypoint downloads no surface vector and returns empty
solution vectors with zero download time. It separately reports synchronized
callback time. The host-export entrypoint keeps its existing behavior. Callback
errors stop the sweep and propagate unchanged; callback residency remains the
consumer's responsibility.

The native regression now exercises 28 exported and 28 device-output physical
frequency solves. Callback device copies use zero host transfers and allocations,
retain results across subsequent solves, and match host exports within relative
1e-10. Physics/admittance delivery and immediate callback-error propagation pass.
CUDA memcheck reports zero errors; CPU BEM passes 541 tests with 3 ignored.
Root check/Clippy and CUDA BEM Clippy pass, with no modified-file warnings.
Evidence: [device sweep output](cuda-tbem-sweep-device-output-2026-10-22.json).

This provides the handoff needed for resident field evaluation and reductions.
The existing resident field evaluator is tied to H-matrix velocity/impedance
traces; adapting it to frequency-changing prepared systems remains work. NumCalc
still selects host export with explicit CPU-field provenance. No fully resident
NumCalc output or new performance speedup is claimed.

### Shared resident field workspace — 2026-10-22

`CudaBemFieldWorkspace` now owns the native field geometry, evaluation points,
velocity/admittance coefficients and output buffers independently of H-matrix
sessions. The H-matrix field API delegates to this implementation. A frequency
sweep callback can now evaluate directly from its certified device pressure,
with changing physics and resident incident descriptors. Fixed boundary data
uploads once. The constructor validates surface topology, node bounds, finite
geometry, DOF order and nonempty finite velocity data before allocating.

The new native sweep regression evaluates three off-surface points at four
frequencies on a 20-panel impedance sphere. Fields agree with the CPU evaluator
within relative 1e-7, allocate no device buffers and upload no host data per
evaluation. Compact validation downloads remain. Existing H-matrix fields retain
their <=16-byte download gate; the refactor initially added a redundant input
validation download, which was removed while preserving output validation.
Both producers pass launch-failure recovery and CUDA memcheck with zero errors.
Invalid connectivity/on-surface points reject before allocation. CPU BEM passes
541 tests with 3 ignored; root check/Clippy and CUDA BEM Clippy pass without
modified-file warnings. Evidence: [shared fields](cuda-shared-resident-fields-2026-10-22.json).

The workspace currently requires fixed velocity/impedance coefficients, exterior
physics and constant Tri3 surfaces. It uses the existing seven-point rule and
does not establish near-surface quadrature convergence. NumCalc selection,
changing boundary coefficients and broader trace representations remain open.

### NumCalc native field integration — 2026-10-22

`run_project_cuda_jacobi` now prepares the shared field workspace once and uses
the device-output sweep callback to evaluate each grid before exporting its
surface solution. Geometry and incident descriptors remain resident during the
sweep. Field and surface arrays are downloaded only for the requested JSON
output. `execution: cuda_jacobi` identifies this route; historical
`cuda_jacobi_with_cpu_fields` JSON remains readable. CPU dispatch remains
unchanged. Evaluation grids with unsupported pressure traces reject before
device allocation; there is no implicit CPU field fallback.

The NumCalc native test passes surface and two-point field parity against CPU
at relative 1e-7 for full/partial sweeps, checks empty/inverted ranges, historical
serialization and unsupported-trace rejection. CUDA memcheck reports zero errors.
The CLI succeeds on the sample project with native provenance. CPU BEM passes
541 tests with 3 ignored; root check/Clippy and CUDA CLI/lib Clippy pass.
Evidence: [NumCalc resident fields](cuda-numcalc-resident-fields-2026-10-22.json).

This closes the supported NumCalc velocity/impedance off-surface output handoff,
not general field coverage. Constant-Tri3 dense limits, seven-point quadrature,
near-surface convergence, broader traces and compressed sweeps remain open.

### Quiescent retained FEM phase profile — 2026-10-22

The current release benchmark now has a quiescent profile for aggregation LOR
AMG on hex Q2/Q3 at subdivisions 4 and 8. The live BEM sanitizer was temporarily
suspended, three consecutive GPU utilization samples were zero, and no other
compute process was listed. Its context and approximately 1550 MiB remained
resident. A `finally` block resumed it afterward; its process state was verified
running. The unfinished sanitizer result is not counted as passed.

Eight CSR/specialized operator cases and 40 timed solves converge with zero warm
device allocations and independent residuals below 2e-9. On Q3 with 15,625 DOFs,
the normal median solve times were 314.55 ms (CSR) and 313.87 ms (specialized),
both at 152 iterations. These are baseline observations, not an optimization
comparison. The separately synchronized CSR diagnostic attributed 229.06 ms
to preconditioning, 21.07 ms to the physical action, 0.71 ms to physical residual
formation and 99.29 ms to other solve work. Synchronization perturbs these times.

AMG post-smoothing accounts for 69.63 ms, pre-smoothing 41.17 ms, residuals
35.80 ms, restriction 29.13 ms, prolongation 33.70 ms and coarse solve 15.59 ms.
The hierarchy has 15,625 / 823 / 157 / 92 / 52 / 37 / 24 DOFs. The small
noncoarse levels each incur roughly 9.5–10 ms total post-smoothing overhead
across 152 visits despite shrinking arithmetic. The current implementation
dispatches SpMV and a separate fused vector update for each Jacobi sweep.

The next concrete kernel candidate is an opt-in **single-block small-CSR Jacobi
smoother**: cache the old iterate in shared memory, perform simultaneous row
updates, synchronize between sweeps, and write back after the last sweep.
Keep the configured weight/sweep count and hierarchy unchanged. Restrict the
initial experiment to levels fitting one block; preserve cuSPARSE plus the
existing fused update elsewhere. Validate all scalar types, zero/nonzero starts,
multiple sweeps, failure handling and full-cycle/solve parity, then compare
complete solve time before promotion. Do not change global defaults from this
profile alone. Evidence: [quiescent FEM profile](cuda-fem-quiescent-profile-2026-10-22.json).

### Small-level Jacobi integration and rejected performance candidate — 2026-10-22

`Backend::try_small_csr_jacobi_native` now exposes the resident operation with
an unchanged-output fallback. `BackendAmgPreconditioner::set_small_jacobi_enabled`
opts into it without changing the hierarchy, weight or sweep counts. Defaults
remain disabled; unsupported sizes use the previous resident SpMV/update path,
while launch errors propagate. The zero-start first sweep retains the existing
diagonal-scale shortcut. Numeric refresh preserves the option; reconstructed
hierarchies start disabled.

Validation covers all four scalar types on V/W/F cycles, including a 513-row
fine level outside the kernel limit and smaller coarse levels. Enabled/disabled
applications agree, enabled applications reduce dispatch count, and repeated
applications allocate nothing or transfer bulk vectors. The independent
complex64 cycle reference also passes with the opt-in enabled, including
zero-start, revisits, L1-Jacobi and untouched Chebyshev behavior. Its CUDA
memcheck reports zero errors. CPU compute passes 655 tests and preconditioners
241; root check/Clippy and CUDA preconditioner Clippy pass with existing warnings.

The release FEM benchmark accepts `SONIUM_FEM_BENCH_SMALL_JACOBI=on|off` and
records the choice per result. A quiescent off/on/on/off comparison used the
same aggregation LOR hierarchy on hex Q2/Q3 at subdivisions 4 and 8. All 160
timed solves converged, with identical iteration counts within each case,
independent residuals below 1e-9 and no warm device allocations. Preconditioner
applications have zero host transfers. The existing BEM sanitizer was suspended
for measurement and resumed afterward; its overall result remains pending.

**Do not promote this candidate.** Combined medians regress by 1.49–6.17% across
the eight cases. For Q3 at 15,625 DOFs, CSR goes from 314.28 to 333.41 ms and the
specialized action from 313.66 to 333.01 ms. Reduced dispatch count did not
improve time to tolerance. Investigate per-row parallelism and repeated CSR
validation cost before another candidate, preserving malformed-input safety.
Measure those hypotheses rather than assuming launch count predicts benefit.
The broader coverage registry and kernel priorities 2–4 remain open.

Full samples, source hashes, conditions and validation:
[small Jacobi AMG evidence](cuda-small-jacobi-amg-2026-10-22.json).

### Cooperative small-level Jacobi — 2026-10-22

The experimental small-CSR kernel now assigns four adjacent lanes to each row,
using subgroup reductions and whole-block barriers between simultaneous sweeps.
Launches round to complete warps, up to 1024 threads for 256 rows. Padding lanes
participate in every shuffle and barrier. CSR offset/column validation remains
on device before any dependent read; malformed offsets produce NaN output.
Tests now include negative, descending and oversized offsets across all four
scalar types, in addition to column errors and launch recovery.

Native primitive tests, all-scalar V/W/F parity and the independent complex64
cycle reference pass. Primitive CUDA memcheck and synchronization checking
report zero errors. Root check/Clippy and CUDA compute Clippy pass with existing
warnings. The four-lane reduction changes summation order; original numerical
tolerances and independent residual gates remain in force.

A second quiescent off/on/on/off release comparison used the same eight FEM
operator cases, with 160 timed solves. Iteration counts are unchanged, all true
residuals are below 1e-9, warm device allocations are zero, and preconditioner
applications have zero host transfers. Combined median solve time improves by
1.29–5.20% for the 729, 2197 and 4913 DOF cases. At 15,625 DOFs it regresses by
0.95% with CSR and 1.10% with the specialized action. This improves on the prior
single-thread-per-row candidate but does not justify a global default change.
The option remains disabled by default. A measured workload-selection rule or
further kernel work must pass complete solves before promotion; do not infer
one from mesh size alone. The existing BEM sanitizer was resumed and verified
running after measurement.

Samples, source hashes and conditions:
[cooperative Jacobi evidence](cuda-small-jacobi-cooperative-2026-10-22.json).

### Retained hex-pressure smoother selection — 2026-10-22

`CudaHexPressureSession` now exposes `set_small_jacobi_enabled` and a matching
query. The option defaults off, performs no device work when toggled, and
preserves published output and previous-solution warm starts. Both compatible
numeric refresh and explicit auxiliary-preconditioner rebuild retain the
selection. Failed rebuilds keep the previous preconditioner, and failed numeric
refresh retains the option while preserving the existing invalidation contract.
This connects the measured candidate to the retained physical solve API without
changing its default policy or auxiliary operator.

The native session tests pass for Q2/Q3 Dirichlet, Neumann and Robin systems,
changed RHS/warm starts, curved geometry, frequency/material/facet updates,
aggregation and classical hierarchies, coarse refresh/rebuild, cancellation and
failed-publication recovery. Physical residual checks use independently
assembled CPU operators. Toggling is checked for zero allocations, transfers or
dispatches; compatible warm solves retain their allocation gates. Both tests
pass CUDA memcheck with zero errors. Root check/Clippy pass with existing
warnings; CPU FEM passes 1964 tests with one ignored. The earlier bounded
performance comparison remains the basis for opt-in status; no new general
speedup claim follows from API integration.
CPU preconditioners also pass 241 tests; CUDA FEM Clippy passes with existing
warnings. Evidence: [retained hex smoother selection](cuda-hex-small-jacobi-session-2026-10-22.json).

### Runtime Krylov admission before retained preparation — 2026-10-22

The Krylov capability table previously exposed the CSR operator state but
omitted it from `native_execution`. A host-staged CSR action could therefore
leave that field incorrectly true. The decision now includes operator and
preconditioner states, and strict admission rejects such a cell.

`validate_backend_krylov` exposes the existing read-only runtime preflight for
preparation-time callers. Retained facade GMRES runs it before preconditioner
setup/device allocation; prepared FGMRES runs it before invoking the caller's
preparation callback. Iteration-time admission remains in place. This covers
the declared CSR/identity primitive cell; arbitrary operator/preconditioner
callbacks still require their own dependency and residency validation. PCG
preparation is not changed by this increment.

All-four-scalar tests vary required primitives across unsupported, host-shared,
host-staged, native and vendor-native states for single/block GMRES and FGMRES.
Explicit non-native mode still admits available fallbacks, unsupported cells
always reject, and missing unrelated gather/contraction kernels do not reject
a CSR solve. Native retained facade tests pass all 10 cases, including prepared
FGMRES, numeric refresh and failure recovery. CPU iterative tests pass 327 and
facade tests pass 117. Root check/Clippy and CUDA facade Clippy pass with existing
warnings. Missing-module combinations use synthetic capability reports rather
than a claim of physically unloading CUDA modules.

Evidence: [early Krylov admission](cuda-krylov-early-admission-2026-10-22.json).

### PCG preparation-time primitive admission — 2026-10-22

`validate_backend_pcg` checks the CSR PCG dependencies (SpMV, BLAS1 and fast
reductions) without allocations or device work. Strict mode rejects host
fallbacks; unsupported primitives reject in either mode and errors name the
failed primitive. Missing multi-dot, block projection, SpMM and fused-many-AXPY
kernels do not reject a PCG solve that does not use them. Retained facade CG and
CG+AMG now run this check before preconditioner preparation/device allocation.
SPD/HPD validation remains independent and unchanged.

All-four-scalar report tests cover unsupported, host-shared, host-staged, native
and vendor-native states. Native tests directly assert zero allocations,
launches/library operations and transfers during admission. The complete native
facade selection passes 10 tests; the focused PCG rerun passes both tests,
including non-Hermitian input rejection. CPU iterative passes 328 tests and the
facade passes 117. Root check/Clippy and CUDA facade Clippy pass with existing
warnings. No kernels changed, so this increment reuses the prior kernel memory
validation rather than claiming a new sanitizer run.

This closes preparation-time admission of the facade's declared CSR PCG
primitive set. Generic `DevicePcgSession` retains its existing backend-neutral
contract; custom operators/preconditioners and action-specific dependencies
still need their own residency/admission coverage.
Evidence: [early PCG admission](cuda-pcg-early-admission-2026-10-22.json).

### Backend-local FEM kernel preparation and reuse — 2026-10-22

Hex partial assembly and numeric quadrature assembly now retain their compiled
functions in backend-owned caches. Every required symbol is resolved before
operator/map uploads. Repeated operators and the three boundary maps in an
updatable pressure session reuse the same module rather than invoking NVRTC
for each object. Cache publication occurs only after successful loading; a
failed attempt leaves the cache empty for retry. Module ownership stays with
the backend/context and operator handles; no process-global cache is introduced.

`CudaHexPressureSession` performs strict FGMRES primitive admission and prepares
the required hex/quadrature kernels before host form preparation and auxiliary
AMG construction. This covers these declared dependencies, not arbitrary
callbacks or every optional validation helper. Driver module memory remains
outside pooled-array accounting; zero array-allocation counters must not be
interpreted as zero driver memory usage.

Native tests assert reuse of the same cached handle and no pooled-array
allocations, transfers or launches during kernel preparation. Existing hex
forward/adjoint/block/constraint actions and resident quadrature updates pass,
as do retained physical-session update and recovery tests. Both compute tests
pass CUDA memcheck with zero errors. Root check/Clippy and CUDA FEM Clippy pass
with existing warnings. No new full-solve performance claim is made: this
increment removes repeated setup compilation and moves failures earlier.
CPU compute passes 655 tests; CPU FEM passes 1964 with one ignored.
Evidence: [FEM kernel preparation cache](cuda-fem-kernel-cache-2026-10-22.json).

### Tetrahedral kernel preparation and reuse — 2026-10-22

P1 and P2 tetrahedral action/assembly functions now use separate backend-local
cache entries. Each retains its own occupancy-derived launch sizes, avoiding
repeated NVRTC compilation and occupancy queries for repeated operators.
Invalid node counts reject explicitly; cache publication occurs only after
both functions and their launch sizes are available. Driver module memory
remains separate from pooled arrays.

All retained Tet pressure constructors perform strict FGMRES admission and
prepare the selected tetrahedral kernels before host geometry, boundary or AMG
setup. Nonempty boundary routes also prepare the shared quadrature kernel.
The direct numeric-assembly constructor prepares kernels before building its
host CSR contribution pattern. Element mode still retains no physical CSR or
numeric assembly map; this change does not broaden its storage requirements.

Native compute tests verify separate P1/P2 cache entries, same-specialization
handle reuse, valid launch sizes, invalid-order rejection and zero pooled-array
allocations, launches or transfers during preparation. Both P1/P2 action and
assembly tests pass CUDA memcheck with zero errors. Six native FEM tests pass,
covering persistent solves, CSR/element selection, frequency changes, P1/P2
Robin lifting, failure recovery and the physical-CSR storage gate. CPU compute
passes 655 tests and FEM passes 1964 with one ignored. Root check/Clippy and CUDA
FEM Clippy pass with existing warnings. No new full-solve speedup is claimed.
Evidence: [Tet kernel preparation cache](cuda-tet-kernel-cache-2026-10-22.json).

### Electrostatic potential and resident electric fields — 2026-10-22

Generic FEM admission now includes assembled f64 H1 electrostatic potential on
Tri3/tetrahedron P1/P2 for forward, transpose and Hermitian actions. Positive
permittivity and removal of the constant nullspace are prerequisites for PCG;
nonlinear dielectric, floating-conductor and electromechanical systems remain
separate work. Host forms and symmetric constraint elimination feed the existing
retained CUDA solve session with symmetric AMG. Generic FEM session preparation
now also checks runtime Krylov/PCG dependencies before either assembly callback.

`CudaElectrostaticCentroidField` reuses the resident sparse gradient machinery
with unit weight and zero offset to recover E=-grad(V), retaining its output on
CUDA. Geometry preparation is host work. Invalid or failed evaluation hides
published output until repaired. Only compact finite-validation status reaches
the host during evaluation; energy/capacitance currently use the CPU
postprocessor after an explicit potential download.

Physical native validation covers 36 solves: two element families, P1/P2, three
actions and three voltages on a unit-area/unit-gap capacitor (unit depth in 2D).
Uniform positive permittivity is normalized out of the charge-free Laplace
solve and applied once to the CPU energy/capacitance output. Potential error is
below 1e-7, electric-field analytic error below 1e-6, GPU/CPU field parity below
1e-11, independent relative residual below 1.1e-9 and capacitance relative error
below 1e-7. Warm solves and field evaluations allocate nothing; field evaluation
has zero H2D bytes and at most 16 compact D2H bytes. Buffer and injected-launch
failures exercise field invalidation and recovery.

The physical test passes CUDA memcheck with zero errors. All 16 generic native
FEM tests pass, as do eight public admission tests and 1964 CPU FEM tests (one
ignored). Root check/Clippy and CUDA FEM Clippy pass with existing warnings.
Heterogeneous dielectric updates, GPU integral observables and broader coupled
models remain open; this is bounded execution coverage, not a general speedup.
Evidence: [electrostatic potential](cuda-electrostatic-potential-2026-10-22.json).

### Resident electrostatic energy, capacitance and charge — 2026-10-22

`CudaElectrostaticObservables` now connects the persistent solved potential to
native CSR action and dot reduction. It retains the unconstrained unit Laplacian
and action scratch, computes epsilon*V^T*K*V/2, and returns scalar energy,
capacitance and signed charge. This supersedes the CPU-only integral boundary
in the preceding increment. Uniform positive permittivity is applied once;
the caller must supply the physical voltage difference and original Laplacian,
not the Dirichlet-eliminated solve matrix. Host assembly remains setup work.

The physical test covers 36 triangle/tetrahedron P1/P2 solves across all three
actions and voltages 1, 2 and -0.5. Native observables match CPU results within
1e-10 relative error; analytic capacitance matches within 1e-7. Warm evaluation
allocates nothing, uploads zero bytes and downloads at most 64 compact bytes.
Invalid dimensions, voltage ranges, permittivity, nonfinite potentials, arithmetic
overflow, negative energy and injected launch failures are rejected; recovery
is tested. Independent original-system solve residual remains below 1.1e-9.

Fresh native validation passes, and CUDA memcheck reports zero errors. CPU FEM
passes 1964 tests with one ignored; root check/Clippy and CUDA FEM Clippy pass
with existing warnings. No speedup is claimed. Heterogeneous dielectric updates,
conductor-resolved charges and coupled models remain open. This evaluator does
not independently certify that its input solves the PDE.
Evidence: [resident electrostatic observables](cuda-electrostatic-observables-2026-10-22.json).

### Heterogeneous electrostatic material updates — 2026-10-22

`CudaElectrostaticObservables::try_from_stiffness` accepts an unconstrained
operator already weighted by absolute permittivity. The existing constructor
also accepts relative material weights with a fixed positive reference scale.
`try_update_values` copies finite resident values into the retained CSR without
rebuilding its descriptor or scratch. Every failed update disables evaluation
until complete replacement. Copy failures permit backend recovery; synchronization
failure is terminal and requires backend recreation.

Twelve physical cases exercise two-layer capacitors on conforming triangle and
tetrahedron P1/P2 meshes, relative dielectric contrasts 2, 4 and 0.5, and positive
and negative voltage. Generic retained PCG uses the initial symmetric AMG cycle
through material changes. The piecewise-linear potential matches within 1e-7;
energy, capacitance and signed charge match analytic results within 1e-7 relative
error. CPU energy parity is within 1e-10 and independent original-system residual
below 1.1e-9. Warm solve/update/evaluation allocates no device buffers. After
explicit host assembly/upload, resident energy update plus evaluation uploads
zero bytes and downloads at most 80 compact bytes.

Both native electrostatic tests pass, including the earlier 36 uniform cases;
CUDA memcheck reports zero errors. CPU FEM passes 1964 tests with one ignored.
Root check/Clippy and CUDA FEM Clippy pass with existing warnings. Host material
assembly and upload remain recurring boundaries; this does not establish native
material lowering, arbitrary-contrast AMG convergence, conductor-resolved charge
or coupled electromechanics. No speedup is claimed.
Evidence: [electrostatic material updates](cuda-electrostatic-material-updates-2026-10-22.json).

### Resident material-basis combination through FEM solves — 2026-10-22

The generic retained FEM session now exposes `try_combine_system_lagged` to
form both physical CSR values and RHS from resident bases. It reuses native
copy/scale/AXPY primitives and validates both results before marking them usable.
Failures invalidate matrix, RHS, published output and warm-start eligibility;
successful full replacement restores solvability. The prepared preconditioner
is explicitly retained, with its method assumptions remaining the caller's
responsibility. The electrostatic energy operator exposes the matching
`try_combine_values` operation.

The layered capacitor fixture prepares matrix, RHS-lifting and energy bases
once. A base includes constraint identity rows; material increments have zero
constraint rows. Material contrast and voltage then enter as scalar combination
coefficients, without repeated host numeric assembly or bulk uploads in this
route. Separate CPU assembly in the test supplies only the parity reference.
Geometry, initial bases and initial AMG preparation remain host setup.

Twelve combination solves on triangle/tetrahedron P1/P2 pass the same analytic
potential, energy, capacitance, signed-charge and original-system residual gates
as direct numeric replacement. Warm solves and energy updates allocate nothing;
the resident windows upload zero bytes, and energy combination plus evaluation
downloads at most 80 compact bytes. Invalid coefficients, combined-value overflow,
copy failure, failed publication and full-replacement recovery are exercised.

All three electrostatic native tests and all 18 generic FEM native tests pass.
CUDA memcheck reports zero errors for the electrostatic tests. CPU FEM passes
1964 tests with one ignored; root check/Clippy and CUDA FEM Clippy pass with
existing warnings. Physical combination evidence is f64 forward electrostatics;
other scalar/action combinations, non-affine material laws and changing geometry
need separate coverage. No complete-solve speedup is claimed.
Evidence: [resident material combination](cuda-electrostatic-material-combination-2026-10-22.json).

### Complex resident Helmholtz combinations and adjoints — 2026-10-22

The generic retained combination API now has complex64 Helmholtz integration
evidence for triangle P1/P2 forward, transpose and Hermitian systems. One-time
uploaded K/M and discrete-load bases form K-k²M on CUDA. Hermitian requests
conjugate the frequency coefficient as well as orienting the bases. Constraint
identity is carried by K and cancels from the mass increment. FGMRES retains
AMG for the positive shifted auxiliary K+16M, including native complex coarse LU.

Eighteen solves cover three nearby complex wavenumbers and all six order/action
combinations. Every solve starts from zero and must perform iterations, avoiding
a trivial warm-start success with an unchanged manufactured solution. Independent
host assembly after the solve confirms original-system residual below 1.1e-9
and relative solution error below 1e-7. Warm calls allocate no device buffers.
Matrix/RHS combination uploads zero bytes and downloads at most 32 compact
validation bytes. FGMRES control transfers remain nonzero and are recorded
separately; this is not a zero-transfer iteration claim.

The native test passes CUDA memcheck with zero errors. CPU FEM passes 1964 tests
with one ignored; root checks and CUDA library/test Clippy pass with existing
warnings. This increment adds integration evidence for the existing production
API, not another implementation. Arbitrary frequencies, boundary types and
application producers still require separate coverage. No speedup is claimed.
Evidence: [resident Helmholtz combinations](cuda-helmholtz-resident-combination-2026-10-22.json).

### CUDA-aware automatic BEM assembly selection — 2026-10-22

`AssemblyMethod::resolve_cuda` closes the 1000–1279-panel admission gap:
Auto selects H-matrix at the exclusive dense CUDA ceiling of 1000 DOFs.
Dense assembly, LU/GMRES preparation and automatic dispatch share the same
constant. The CPU crossover stays at 1280, explicit assembly requests stay
unchanged, and the linear solver is never substituted. Auto with Direct at
compressed sizes rejects before device allocation; Auto with GMRES can proceed.

A closed 1000-panel Fibonacci sphere passes native compressed scattering with
low-rank blocks, exact relative residual below 1e-6, CPU surface/field parity
at 1e-5 and zero repeated-solve device allocations. Pure policy tests cover
999/1000 and 1279/1280 boundaries and all explicit representations. CPU BEM
passes 541 tests with three ignored; root checks and CUDA BEM checks/Clippy pass
with existing warnings. Native memory checking is still running and is not
reported passed. This is an admission correction, not a speedup claim.
Evidence: [CUDA Auto admission](cuda-bem-auto-admission-2026-10-22.json).

### Early common GMRES admission for BEM preparation — 2026-10-22

Public dense GMRES, H-matrix and MLFMM preparation now checks pending context
errors and the common native complex64 GMRES dependencies before expensive
geometry/hierarchy setup. The lower-level TBEM GMRES and MLFMM solve constructors
perform the same checks before assembly or local-inverse preparation. Existing
representation-specific FMM, incident and layout checks remain in place; the
common Krylov check does not prove arbitrary operator/preconditioner dependencies.

A native regression injects terminal synchronization errors before preparing
each public representation. All three preserve the original error and execute
zero allocations, kernel launches, library operations or transfers after
rejection. Healthy dense/compressed parity, retention and recovery tests pass,
as does the explicit dense-GMRES dispatch test. The focused preflight memcheck
passes with zero errors. CPU BEM passes 541 tests with three ignored; root and
CUDA BEM checks/Clippy pass with existing warnings.

This does not test runtime removal of CUDA modules or complete block dependency
admission. The separate 1000-panel Auto memcheck remains running and is not
included in the passing preflight result. No performance speedup is claimed.
Evidence: [BEM early admission](cuda-bem-early-admission-2026-10-22.json).

### Block-specific compressed BEM admission — 2026-10-22

`BemSolver::prepare_cuda_hmatrix_block` checks the native block-GMRES capability
cell before building the host hierarchy or oriented inverse. This includes
block projection and CSR multi-vector dependencies beyond scalar GMRES.
Scalar-prepared sessions retain their ability to solve blocks, with the same
check now occurring before block validation/workspace allocation. The returned
session supports both scalar and block solves; dynamic-width workspaces and
active-width caches still warm on first use.

All four native block-session tests pass. Coverage includes both Jacobi and
cluster-block preconditioning, all three orientations, independent/dependent/
zero RHS columns, width changes, low-rank storage, warm allocation reuse and
failure publication/recovery. A new poisoned-context preparation test rejects
without allocations, launches, library operations or transfers; its focused
memcheck reports zero errors. CPU BEM passes 541 tests with three ignored;
root and CUDA BEM check/Clippy pass with existing warnings.

This admits the shared block Krylov dependency contract, not every custom
operator/preconditioner dependency. Runtime module removal and full block
sanitizer coverage were not rerun in this increment. The independent 1000-panel
Auto memcheck remains live and unpassed. No speedup is claimed.
Evidence: [BEM block admission](cuda-bem-block-admission-2026-10-22.json).

### H-matrix block actions eliminate full-vector staging copies — 2026-10-22

Multi-RHS H-matrix actions now apply existing indexed-view CUDA primitives
directly to caller input/output columns. The old two full-vector device copies
per RHS are removed, and output is cleared once per block action. Low-rank
projections retain the existing rank scratch; no width-dependent storage or
host transfers are introduced. Scalar actions keep their transactional scratch
publication, while block output retains its documented discard-on-error contract.

The compute fixture checks widths 1/3/2, all orientations, dense blocks and a
two-component complex factorization against an independent matrix. It asserts
zero allocations and zero H2D/D2H/D2D bytes. Native validation and memcheck pass,
with zero memory errors. All four complete BEM block-session tests pass,
including low-rank storage, preconditioners, dependent/zero columns, independent
residuals, warm allocation reuse and failure recovery. CPU compute passes 655
tests; root checks and CUDA compute checks/Clippy pass with existing warnings.

This eliminates `2*n*rhs*sizeof(Complex64)` copied payload bytes per action.
Columns still execute sequentially; batched indexed GEMM and isolated complete-
solve timing remain future work. No elapsed-time speedup is claimed. The separate
1000-panel Auto memcheck remains pending, independent of the passing compute
memcheck above.
Evidence: [H-matrix column views](cuda-hmatrix-column-views-2026-10-22.json).


### Fused two-base FEM numeric combinations (2026-10-22)

Added a checked optional native `Backend::try_linear_combination2_native` hook
and f32/f64/Complex32/Complex64 CUDA kernels. Persistent CSR combination and
generic FEM system RHS combination now use one native pass for two bases,
replacing copy/scale/AXPY. Unsupported native hooks retain the existing device
operations; arbitrary-length combinations retain the existing implementation.
Coefficient/extent/stream checks precede launch, and session failure invalidation
and native finite validation remain in place. No per-call allocation or host
staging is introduced.

Native primitive tests cover empty, singleton, block-boundary and large vectors,
all four scalars, wrong lengths, nonfinite coefficients, foreign streams,
nonfinite inputs with zero coefficients, and injected launch recovery. Counters
show one launch and zero allocations/H2D/D2H/D2D bytes for a nonempty primitive.
Retained electrostatic material combinations and complex Helmholtz combinations
now assert zero D2D bytes during the numeric system update. All three focused
memcheck runs passed with zero errors: primitive, Helmholtz (18 solves across
forward/transpose/Hermitian), and electrostatics (three fixtures). CPU compute
655 tests and CPU FEM 1964 tests passed (one FEM ignored). Root check then Clippy
and CUDA FEM check/Clippy passed with existing warnings.

Evidence: [fused combination validation](cuda-fem-fused-combination-2026-10-22.json).
This advances kernel priority 2 and its retained-session integration. It is not
an elapsed-time speedup claim; full workflow benchmarking and the remaining
coverage backlog are still required.

Sanitizer status correction: the older scattering/MLFMM process PID 1716879
was confirmed live outside the sandbox process namespace and left running.
The Auto-gap PID 2895228 is absent, but its log has no completion summary;
its evidence now records unverified rather than running or passed.


### Complete-solve gate for fused numeric updates (2026-10-22)

Added `CudaBackend::with_fused_numeric_combinations` and an ignored physical
Helmholtz benchmark that compares the original device BLAS update with fusion
through `CudaFemSolveSession::try_combine_system_lagged`. Both modes keep
shifted auxiliary AMG, use zero-start FGMRES with the same controls, and pass
independent host-assembled residual/error checks with identical iteration counts.
The benchmark covers P1/P2, 81/289/625/2401 DOFs, two warm sweeps and nine timed
three-frequency sweeps per mode. Mode order alternates. Native updates have
zero H2D bytes, warmed solves have zero allocations, and only the fused update
has zero D2D bytes. Timing excludes preparation and explicit host validation.

Two isolated runs show faster numeric updates but mixed sub-percent full-window
changes. Fusion is now disabled by default and remains available explicitly;
this corrects the previous unconditional dispatch without removing the kernel.
The generic FEM CUDA test pool explicitly enables fusion to keep its physical
integration coverage. A native disabled-hook test checks false/no-mutation with
zero launch, allocation and transfer counters. Final native validation passed:
two primitive tests under memcheck (zero errors), one Helmholtz test with 18
solves, three electrostatic tests, and the complete-solve benchmark. Root check
then Clippy and CUDA test Clippy passed with existing warnings. CPU compute
655 and FEM 1964 tests passed (one ignored). The existing MLFMM sanitizer was
resumed and confirmed live after both isolated timings; it remains unverified.

Raw samples, hardware, scope, commands and checks:
[complete-solve timing evidence](cuda-fem-fused-combination-benchmark-2026-10-22.json).
Next: profile persistent FGMRES and auxiliary AMG iteration costs. These cases
spend roughly 99.7% of the measured window outside numeric combination, so more
combination tuning cannot materially improve them. Larger/update-dominated
problems remain separate candidates, and the full coverage goal remains open.


### Retained cuSPARSE analysis in FEM and AMG (2026-10-22)

Connected optional `cusparseSpMV_preprocess` to persistent CUDA CSR dispatch.
The runtime symbol is loaded through the already-retained cuSPARSE library and
missing support produces an explicit admission error. First use runs analysis
under the descriptor mutex with real input/output descriptors; subsequent SpMV
calls retain the same workspace, operation, algorithm and scalar type. Numeric
value refreshes and vector retargeting preserve analysis. Block SpMM submissions
share the mutex and retain their separate workspace.

Native compute validation covers all four scalar types, rectangular matrices
with an empty row, changing values/vectors, interleaved block actions, and launch
failure/recovery. The operation counter verifies one analysis call on first use
and none on subsequent numeric refreshes. Compute memcheck and physical FEM
P1/P2 forward/transpose/Hermitian memcheck both passed with zero errors. The
shared ignored benchmark compares identical physical systems, shifted AMG and
FGMRES controls over two warm and nine measured frequency sweeps in alternating
mode order. Two isolated runs show 1.09–1.11x complete-window speedup across all
four tested cases. Profiling the paired harness reduced partition calls from
115082 to 57605 while multiplication calls remained 115082; half of the candidate
harness deliberately retains the baseline.

Root check then Clippy, CUDA check/test Clippy, 655 compute CPU tests and 1964 FEM
CPU tests passed (one FEM ignored; existing Clippy warnings). Evidence and raw
samples: [CSR preprocessing](cuda-csr-preprocessing-2026-10-22.json).
The old scattering/MLFMM sanitizer remains a separate, unverified run and was
resumed after isolated GPU work. Larger matrices, cold amortization, other
physics and broader application default selection remain open.


### General fused CSR Jacobi integration (2026-10-22)

Implemented the optional `Backend::try_csr_jacobi_step_native` contract with
CUDA f32/f64/Complex32/Complex64 kernels. Each lane group computes its row dot
product, residual and damped diagonal correction in one launch. CSR offsets
and columns are checked before access; malformed input produces NaNs rather
than unsafe reads. Empty input is a no-op, and unsupported paths return false
without modifying output. All extents, stream ownership and damping are checked.

The opt-in AMG path retains the existing hierarchy, sweep counts and weights,
uses the existing per-level scratch, and swaps old/new buffers only after a
successful launch. Existing vendor/native update operations remain the fallback.
The earlier single-block multi-sweep option remains separate and has precedence
when enabled. Independent cycle tests now exercise the new option alongside
baseline, profiled and small-level modes.

Validation: four-scalar primitive tests (0–16385 rows, three simultaneous sweeps,
immutable inputs, invalid indices/shapes/streams and recovery) pass memcheck and
synccheck with zero errors. V/W/F-cycle parity with zero/nonzero starts and block
RHS passes memcheck. Eighteen physical P1/P2 forward/transpose/Hermitian solves
pass memcheck. CPU compute 655, preconditioners 241, FEM 1964 tests pass (one
FEM ignored). Root check then Clippy and CUDA check/test Clippy pass with existing
warnings.

Three isolated retained FEM benchmark runs preserve equal iteration counts and
independent residual/discrete-solution error gates. Both modes enable CSR
preprocessing. Stable first/confirmation runs give roughly 1.05–1.10x speedup;
the largest case in the intervening run was nonstationary and is retained in
the evidence. No default promotion or general performance claim follows from
these bounded cases. Full raw samples, methodology and source hashes are in
[fused CSR Jacobi evidence](cuda-fused-csr-jacobi-2026-10-22.json). Larger workloads,
combined small-level dispatch and the remaining CUDA coverage remain open.

### Fused smoother in specialized pressure sessions (2026-10-22)

Tet P1/P2 and Hex sessions expose `set_fused_jacobi_enabled` and a getter.
Changing the option performs no device work or allocation and preserves a
published solution and warm start. Numeric refresh and explicit hierarchy
rebuild retain the selection. Existing small-level cooperative smoothing keeps
precedence when both options are enabled. The default remains disabled.

Native Tet coverage exercises CSR and element operators, changing frequencies,
nonzero boundary values, warm starts, rebuild/refresh and failure recovery.
Native Hex coverage exercises toggling on a live session, Q1–Q3 updates and
preconditioner refresh/rebuild. All three tests pass normally and under memcheck
with zero errors. CPU FEM passes 1964 tests with one ignored. Root check then
Clippy, CUDA test/example Clippy and diff whitespace checks pass; existing
warnings remain. The earlier separate BEM scattering sanitizer is still pending.

The tetrahedral comparison example accepts `SONIUM_FEM_BENCH_FUSED_JACOBI`
and `SONIUM_FEM_BENCH_CSR_PREPROCESS`, each strictly 0 or 1 and defaulting to 0.
Both choices are recorded in output. An isolated baseline/fused/fused/baseline
comparison uses subdivisions 4, three frequencies, one warmup and five timed
zero-start solves per frequency. Both modes enable CSR preprocessing. Iteration
counts match, warm allocations are zero, and independent residual/error gates
pass. Timings exclude setup, numeric updates and reference/output work.

| Case | DOFs | Median baseline/fused time ratio |
| --- | ---: | ---: |
| P1 CSR | 125 | 1.0427 |
| P1 element | 125 | 1.0331 |
| P2 CSR | 729 | 0.9763 |
| P2 element | 729 | 0.9847 |

Ratios above one favor fusion. These bounded results support keeping explicit
selection: P1 improves while P2 consistently regresses. They do not establish
Hex speedups or justify a general default. All samples and source hashes are in
[specialized fused smoother evidence](cuda-specialized-fused-jacobi-2026-10-22.json).
Next performance work should profile the P2 crossover and compare larger systems
and combined small-level policies before promoting any automatic dispatch rule.

### Executable source discovery for the coverage denominator (2026-10-22)

Added `scripts/cuda_workflow_inventory.py` and the generated
`perf/cuda-source-inventory.json`. Discovery includes the four requested core
packages, every split solver package and every `sonium-sim-*` package. Every
Rust source file under `src` remains a review unit, even when it has no public
declaration hints. That avoids dropping private producer modules, macros or
conditional implementations merely because lexical discovery cannot resolve
them. The inventory records manifest/source hashes, package, FEM physics
subdirectory group and declaration locations. It currently has 22 packages,
1350 files, 82 review groups and 13678 declaration hints. Hints include types
and modules as well as functions; they are not distinct executable workflows.

`--check` fails on source or manifest drift and reports added, removed and
changed units. `--write` regenerates discovery. The tool does not mutate the
TokenSave graph, load CUDA or claim execution evidence. Comment/string masking
handles nested block comments, raw strings, escaped strings and character
literals while preserving locations. Tests cover false hints, lifetimes,
automatic new-package discovery, files without public declarations, and body,
addition and deletion drift. Both tests pass, as does checking the generated
inventory against current source. Root cargo check then Clippy pass with
existing warnings. The separate BEM scattering sanitizer is still live and
pending; it is not counted as a passed gate.

This closes the missing reproducible *source-discovery* artifact only. The P0
workflow registry remains open: resolve public exports and aliases, classify
helpers/tests with reasons, and expand each producer into scalar, topology,
constraint, action, solver/preconditioner, RHS, update and output variants.
Then attach setup/solve/update/output status and native correctness, residency,
recovery and performance evidence separately. Keep those reviewed workflow
records separate from regenerated source discovery so regeneration cannot erase
review decisions. Examples, benchmarks, external workspaces and non-Rust
entrypoints are explicitly outside this discovery pass and still need review.

### Reviewed solver facade workflow subset (2026-10-22)

The [reviewed workflow registry](cuda-workflow-registry.json) is separate from
regenerated source discovery. Its initial subset enumerates every current
`LinearSolverType` variant across f32, f64, Complex32 and Complex64 for a single
forward CSR solve: 40 rows. Five strategies have retained CUDA dispatch
(GMRES identity/Jacobi/AMG and CG identity/AMG); Auto, GMRES ILU, GMRES Chebyshev,
GMRES Schwarz and FGMRES Schwarz remain explicit facade rejections. The separate
prepared-FGMRES callback API does not implement those named strategy contracts.
CPU strategy admission and SPD/HPD assumptions still apply independently.

Each row records its producer, scalar, matrix/constraint boundary, action, RHS,
assumption, setup/solve/update/output execution, correctness, residency,
performance and remaining work. No row is marked complete. Source discovery
and this bounded subset are not the full workflow denominator, and the 20
retained-dispatch rows must not be presented as a project coverage percentage.

`python3 scripts/check_cuda_workflow_registry.py` checks reviewed source hashes,
strategy/scalar exhaustiveness, source admission agreement, duplicate rows,
phase presence and bounded native evidence provenance. New enum variants require
new rows; source changes require review before hash updates. Native test claims
require an attached successful result with matching reviewed source anchors.
The checker does not itself run GPU tests or prove callback residency.

All six current `native_cuda_public_` tests passed, covering the five strategies
and four scalars, retained solves, numeric refresh, duplicate CSR canonicalization,
rebuilds, failed publication and terminal recovery. The exact output, binary hash
and reviewed source hashes are in
[validation evidence](cuda-workflow-registry-validation-2026-10-22.json).
Eight registry tests and two source inventory tests pass; root cargo check then
Clippy and whitespace checks pass with existing warnings. No performance or new
sanitizer claim is made. The separate BEM scattering sanitizer remains pending.

Next: implement the rejected named facade strategies using their appropriate
native preconditioners, while extending reviewed rows to split solver APIs,
prepared callbacks, block/adjoint actions, FEM/BEM producers and applications.
Recurring host assembly/update boundaries remain explicit open work.

### Persistent public GMRES–Chebyshev strategy (2026-10-22)

Connected the existing owned `ChebyshevBackend` to `CudaLinearSolveSession` and
the one-shot CUDA facade. The named strategy now executes native GMRES with
Chebyshev preconditioning instead of being rejected. It preserves the CPU
facade's checked real-SPD coefficient restriction, default degree 5, interval
ratio 30 and host spectral estimator. f32/f64/Complex32/Complex64 storage is
supported, including complex RHS; non-real Hermitian matrix coefficients remain
rejected under this strategy's existing CPU contract.

The session owns the preconditioner's CSR and polynomial scratch across RHS
changes. Fixed-pattern updates validate/reestimate the spectral bound before
physical upload, then copy the already-uploaded physical values to the retained
preconditioner and refresh its inverse diagonal. This incurs no temporary GPU
allocation or second bulk host matrix upload. Host coefficient production,
validation and spectral estimation remain explicit setup/update work. The
additional owned preconditioner CSR is an explicit memory cost.

Extended the native six-strategy/four-scalar retained-solve and numeric-refresh
fixtures. Warm solves and compatible refreshes allocate zero device buffers;
refresh has zero D2H bytes. Changed coefficients match freshly prepared solves.
Injected transfer/diagonal failures block publication until recovery. Indefinite
updates reject, while an additional non-real coefficient test proves rejection
before device allocation/upload and subsequent zero-start recovery. The same
fixture exercises one-shot dispatch with CPU fallback forbidden.

All seven native public session tests pass, including memcheck with zero errors.
CPU facade 117 tests, root check then Clippy, CUDA test Clippy and ten registry/
inventory tests pass with existing warnings. Evidence and source/binary hashes:
[public Chebyshev session](cuda-public-chebyshev-session-2026-10-22.json).
No performance claim follows from these correctness/residency gates. Auto,
GMRES ILU, GMRES Schwarz and FGMRES Schwarz remain rejected named facade routes.
The reviewed registry now has 24 retained-dispatch rows and 16 rejected rows;
none is complete across its remaining producer/variant scope. The separate
older BEM scattering sanitizer remains live and pending.

### Persistent public GMRES–ILU strategy (2026-10-22)

Added `CudaIluPreconditioner` in the split preconditioner crate and connected it
to the facade's explicit `GmresIlu` strategy. Checked default CPU ILU(0) supplies
the factors with no reordering. Forward/back substitution uses the existing
native colored-row primitive in dependency-level order, giving exact triangular
substitution rather than approximate iterative triangular sweeps. Independent
rows within a level run together. Upper rows are normalized to unit diagonal,
with a separate resident inverse-diagonal action, to avoid the primitive's
absolute pivot cutoff. Nonfinite normalized factors reject construction.

Native row-kernel admission precedes preconditioner allocation. Triangular CSR
buffers, dependency-row lists, diagonal action and three scratch vectors persist.
Compatible numeric updates refactor on the host and replace values in place;
changed factor patterns require replacement. Interrupted uploads invalidate the
adapter until a complete refresh succeeds. Intermediate application failures
preserve the caller's output, while the facade enforces its usual publication
and warm-start invalidation. Raw-buffer quota accounting remains an open
cross-cutting issue; this is not a claim of complete pool quota enforcement.

Direct action parity passes against CPU ILU for all four scalar representations,
nonsymmetric disconnected components, independent same-level rows, an isolated
row, multiple RHS and scales 1 and 1e-20. Application allocates no device memory
and transfers no vectors to/from the host. Launch failure and partial numeric
refresh recovery pass. The seven public CUDA tests now exercise seven retained
strategies across four scalar representations, including ILU zero-allocation
compatible updates and warm solves. Both the direct factor fixture and all seven
facade tests pass memcheck with zero errors. CPU compute 655, preconditioners 241,
facade 117, root check then Clippy, CUDA test Clippy, registry/inventory tests and
whitespace checks pass with existing warnings.

Evidence: [public ILU session](cuda-public-ilu-session-2026-10-22.json).
No timing improvement is claimed: long dependency chains require many launches.
A dedicated or vendor sparse triangular solve is a future performance candidate
once profiled against this retained implementation. Host factorization remains
explicit setup/update work. Block/adjoint/reordered variants remain open. Auto,
GMRES Schwarz and FGMRES Schwarz are the three remaining rejected named facade
routes. The registry has 28 retained-dispatch rows and 12 rejected rows, with no
project-wide completion claim. The older BEM scattering sanitizer is still pending.

### Persistent public GMRES/FGMRES–Schwarz strategies (2026-10-22)

Connected `GmresSchwarz` and `FgmresSchwarz` to a new resident one-level
`CudaSchwarzPreconditioner`. The facade preserves the CPU automatic subdomain
count/partition policy, overlap depth one and partition-of-unity weights.
FGMRES explicitly creates the flexible Krylov session; stationary GMRES retains
its normal session. This is not substitution with global ILU or block Jacobi.

The adapter packs the CPU local matrices into one block-diagonal ILU action and
uses a single sparse restriction plus weighted assembly map. It retains local
factors, dependency levels and scratch. Sparse maps scale with total overlapping
local DOFs, avoiding dense global storage or an entire global map per subdomain.
Compatible updates retain maps and refresh factors; changes to membership,
weights or local factor patterns rebuild the preconditioner. Partial factor
uploads block application until repaired. Attached coarse-space corrections
reject; one-level forward single-vector application is the implemented contract.

Independent CPU parity covers four scalars, nonsymmetric matrices, four actual
subdomains, overlap depths zero/one/two, two coefficient scales, multiple RHS,
repartition detection and failed-refresh recovery. Application allocates no
buffers and performs no H2D/D2H transfers; compatible numeric refresh allocates
no device buffers and downloads no vectors. All seven current native facade
tests now exercise nine strategies across four scalar representations. Both
the direct parity fixture and facade suite pass memcheck with zero errors.
CPU preconditioners 241 and facade 117 tests, root check then Clippy, CUDA test
Clippy, ten registry/inventory tests and whitespace checks pass with existing
warnings. Evidence: [public Schwarz sessions](cuda-public-schwarz-session-2026-10-22.json).

Host setup currently builds CPU subdomain factors and refactorizes the packed
local matrices for device upload. Removing that duplicated setup work is a
future optimization; no performance improvement is claimed here. Host partition
and numeric factorization work, raw-buffer quota accounting, coarse/block/adjoint
variants and broader producers remain open. Auto is now the only rejected named
facade strategy. The 36 retained-dispatch rows are not a project-wide coverage
percentage and none is marked complete. The older BEM scattering sanitizer
remains pending.

### Retained Auto selection and provenance (2026-10-22)

`CudaLinearSolveSession` and one-shot CUDA facade calls now admit Auto. They use
the existing CPU resolver: GMRES–ILU for small systems and large real systems,
FGMRES–Schwarz for large complex scalar storage, even when its matrix entries
are real. Selection happens once during construction and remains fixed through
compatible numeric updates. `requested_strategy()` and `selected_strategy()`
expose that distinction; caller-prepared FGMRES returns `None` rather than
misreporting its internal certification metadata as a named strategy. Creating
a new session re-evaluates selection thresholds.

CUDA uses its retained standard/flexible Krylov implementations. CPU Auto's
optional pipelined-GMRES path is not implemented here and remains explicitly
open; this is solver-family dispatch parity, not closure of every CPU execution
variant. The source resolver currently does not choose AMG automatically or
infer SPD from size/diagonal signs.

Extended native fixtures cover all ten named strategies across four scalar
representations, one-shot Auto with CPU fallback forbidden, and a 50000-row
signed-diagonal fixture at the default row cutoff. Negative real diagonals
prove no implicit SPD requirement; real coefficients in complex storage prove
scalar-type-dependent FGMRES selection. Large-system compatible numeric
refresh and warm solves allocate zero device buffers, and selected/requested
strategy provenance survives updates. Two prepared-FGMRES tests verify their
new provenance accessors return `None`.

Eight public native tests and two prepared-FGMRES tests pass. The eight public
tests also pass memcheck with zero errors. CPU facade 117 tests, root check then
Clippy, CUDA test Clippy, ten registry/inventory tests and whitespace checks pass
with existing warnings. Evidence: [Auto sessions](cuda-public-auto-session-2026-10-22.json).
An independent native nnz-only crossover fixture, the CPU pipelined variant,
block/adjoint workflows and broader producers remain open. No timing improvement
is claimed. The initial reviewed facade registry now has 40 retained-dispatch
rows and zero outright rejections, but no rows are complete across their
remaining scope and this is not a project-wide coverage percentage. The older
BEM scattering sanitizer remains live and pending.


### Standard GMRES in retained generic FEM sessions (2026-10-22)

`FemDeviceMethod::Gmres` now selects the standard device GMRES constructor and
its runtime admission checks. FGMRES retains its flexible constructor; PCG is
unchanged. Standard GMRES requires a fixed linear preconditioner during each
solve. Both GMRES modes reuse the existing update, residual certification and
failure-invalidation lifecycle.

The new native fixture uses actual constrained P1/P2 Helmholtz assembly with
ILU(0) and Schwarz configured for eight subdomains and two overlap layers.
Forward, transpose and Hermitian actions each run three changed-RHS and scaled
operator solves with a lagged preconditioner. Independent host CSR residuals
remain below 1.1e-9; warm solves allocate no backend buffers. Device numeric
updates upload no host data and download two finite-validation status words.
Injected matrix/RHS copy failures, invalid controls and nonconvergence exercise
publication invalidation and recovery.

Validation: the native fixture and CUDA memcheck passed (zero errors); retained
FEM regression tests passed 22 with three performance experiments ignored. The
CPU FEM library suite passed 1964 with one ignored. Root check then Clippy, CUDA
FEM test Clippy, registry validation and all ten audit tests passed; existing
warnings remain. Inventory freshness passed after refreshing an unrelated
concurrent edit's discovery hash. See
[captured evidence](cuda-fem-standard-gmres-2026-10-22.json).

No speedup is claimed: other GPU work was active. Legacy CSR producer migration,
configured preconditioner factories and broader workflow coverage remain open.


### Configured retained FEM producer entry point (2026-10-22)

`CudaConfiguredFemSolveSession::try_assemble` connects `SolverConfig` directly
to retained complex FEM execution without a caller-written preconditioner
closure. Five standard GMRES choices are supported: identity, Jacobi, ILU(0),
configured one-level Schwarz and default-hierarchy AMG with native coarse LU.
The session owns a copy of the configuration and its Krylov controls; `solve`
uses them across repeated RHS requests. `session_mut` exposes the existing
resident numeric update and publication-invalidation contract. Lagged numeric
updates do not silently refactor or replace the configured preconditioner.

Unsupported algorithms, adaptive restart, unsupported convergence cadence or
progress printing and invalid controls reject before assembly. The entry point
does not alias pipelined/colored/fixed-point variants to standard GMRES/ILU.
Shifted auxiliary systems continue to use the explicit generic preparation API.

Native tests exercise all five strategies on constrained P1/P2 complex Helmholtz
systems. Schwarz configurations (3 subdomains, overlap 0) and (8, 2) match their
CPU preconditioner actions directly, with zero device-application allocations or
host transfers. Three different RHS solves per case pass independent residual
and solution checks; warm solves allocate no backend buffers. Two native tests
and memcheck pass (zero errors). The CPU library suite passes 1964 tests with
one ignored; root check then Clippy and CUDA test Clippy pass with existing
warnings. Inventory/registry validation and all ten audit tests pass.

Evidence: [configured retained sessions](cuda-fem-configured-session-2026-10-22.json).
No speedup or complete workflow claim is made. The next integration step remains
migration of actual legacy application loops to this retained entry point,
including a defined numeric preconditioner refresh/rebuild policy.


### Room-treatment MMA connects retained configured FEM solves (2026-10-22)

`CudaRoomTreatmentSession::try_prepare` takes the existing
`RoomTreatmentProblem`, pool, mesh/space workload and preparation design. It
retains separate forward and Hermitian configured FEM sessions, value/RHS
staging buffers and warm-start state. The room crate's CUDA feature now forwards
`sonium-fem/cuda`. Unsupported configured solver variants and non-Helmholtz
workloads reject rather than silently selecting a replacement.

`objective_gradient` uses the existing objective and derivative routines with
native certified solves. `optimize` shares the original MMA implementation via
an evaluator callback, preserving density filters, spatial constraints and
stopping logic. The CPU API still calls the same shared implementation. To use
the explicit CUDA route, prepare the session and call `session.optimize(config)`.

The numeric policy is explicit lagging: matrix values change with each design,
but both preconditioners retain the preparation design. Every forward/adjoint
solve certifies its current operator. Nonconvergence is an error; callers can
prepare again at a more suitable design. Failed updates/solves clear that
phase's warm-start eligibility. No automatic refactor/rebuild is claimed.

Host sparse design combination, RHS construction, listener evaluation,
sensitivities and MMA remain. Each design uploads values/RHS and downloads the
two certified fields. Warm evaluations allocate zero backend buffers in the
bounded test, but this does not establish zero host transfers or full residency.

Validation uses an actual 4x4 P1 triangular Helmholtz mesh with two boundary
mass operators. ILU and Schwarz (3 subdomains, overlap 1) match CPU direct
objectives/gradients and finite differences across three designs. Four MMA
iterations match CPU history and final design; injected D2D failure recovery
passes. Native and memcheck pass (zero errors). CPU room tests pass 331 with
one ignored; all 12 pipeline tests pass. Root check then Clippy and CUDA test
Clippy pass with existing warnings. Inventory/registry checks and ten audit
tests pass. [Evidence](cuda-room-treatment-session-2026-10-22.json) records the
source/binary/log hashes. No performance claim is made.


### Resident room-treatment matrix combinations (2026-10-22)

The application now uploads its fixed base operator, each treatment derivative
and the corresponding Hermitian-oriented bases once during preparation. Each
design evaluation calls the existing retained numeric-combination primitive to
form `A_base + sum(t_i T_i)` on CUDA. The forward RHS stays resident; only the
changing adjoint RHS is uploaded. No host `operator(treatments)` or sparse
Hermitian reconstruction runs inside the CUDA objective loop. Host assembly at
the preparation design remains necessary for the lagged preconditioners.

The native physical room test measures update phases separately from Krylov
controls. Forward updates upload zero bytes; adjoint updates upload exactly one
complex RHS. Each phase downloads eight bytes for two finite-status checks.
Warm updates and solves allocate zero backend buffers. CPU-direct objective,
adjoint-gradient, finite-difference and four-iteration MMA comparisons still
pass, including interrupted-update recovery. Native execution and CUDA memcheck
pass (zero errors); 331 CPU room tests and 12 pipeline tests pass. Root check,
Clippy and CUDA test Clippy pass with existing warnings. Inventory freshness,
registry validation and ten audit tests pass.

[Evidence](cuda-room-treatment-resident-updates-2026-10-22.json) records this
source snapshot. This supersedes the earlier per-design matrix-upload boundary,
not the remaining host objective, adjoint RHS, sensitivities or full-field
downloads. Storing one device basis per treatment per action increases memory
with treatment count; pool-cap failures remain explicit. No speedup is claimed
without an isolated matched-accuracy timing comparison.


### Resident room-treatment fields and adjoint sensitivities (2026-10-22)

The application now keeps both certified fields on the device. An uploaded
sparse listener matrix computes pressures, and its explicit Hermitian matrix
assembles the adjoint RHS from the listener-sized objective gradient. The CPU
and CUDA paths share the same pressure-level log-power objective helper. Only
listener-sized pressures/gradients cross that boundary; the full state gradient
and adjoint RHS no longer upload, and neither solved field downloads.

Each treatment sensitivity applies its resident derivative to the forward field
and computes `2 Re(lambda^H T_i p)` through a checked native complex dot. One
retained derivative CSR descriptor and action buffer are reused; derivative
values come from the existing resident forward bases by D2D copy. This avoids
allocating a separate full CSR descriptor per treatment. Additional sparse
maps/descriptors retain the compute backend's existing allocation/quota policy;
pooled listener/action vectors enforce the supplied cap.

The physical fixture now includes complex interpolation weights and repeated
listener entries, validating conjugation and canonical duplicate accumulation.
For three listeners, listener processing uploads and downloads 48 bytes each,
with zero warm allocations. Both matrix-update phases upload zero bytes and
download eight validation bytes each. Gradient processing uploads zero bytes;
recorded downloads are bounded by one complex scalar per treatment. Native
vendor scalar-result reductions may not increment the generic transfer counter;
this is not a zero-scalar-transfer claim. Neither gradient inputs nor solved
fields are staged through the host.

CPU-direct objective/gradient, finite-difference, four-iteration MMA and update
failure recovery checks pass. Native execution and CUDA memcheck pass with zero
errors. CPU room tests pass 331 with one ignored; all 12 pipeline tests pass.
Root check then Clippy, CUDA test Clippy, inventory/registry validation and ten
audit tests pass with existing warnings. See
[resident-field evidence](cuda-room-treatment-resident-fields-2026-10-22.json).
No speedup is claimed. Multi-frequency treatment integration, refresh policy,
CLI routing and broader workflow coverage remain open.


### Multi-frequency treatment and production MMA on retained CUDA state (2026-10-22)

A reusable resident treatment state now underlies both the single-frequency
session and `CudaMultiFrequencyRoomTreatmentSession`. Band preparation retains
one configured forward/Hermitian pair per frequency, with resident numeric
bases, fields, listener maps and derivative scratch. Temporary host setup data
is released after each preparation; device states remain for later designs.
Frequencies execute serially on the supplied backend, in sorted problem order.
The CPU `max_frequency_lanes` setting does not request GPU lane parallelism.

Pressure-level production terms (target error, seat variance and minimum-level
penalty), frequency-weighted aggregation and treatment area/cost contributions
are shared with CPU evaluation. The existing multi-frequency MMA loop now accepts
an evaluator callback, so CUDA uses the same area constraint, density filter,
decay/spatial constraints and stopping rules. Cancellation is checked between
frequencies and before returning the aggregate; no partial result is published.
A resumed call evaluates the entire band, retaining valid per-frequency warm
starts. Preconditioners remain explicitly lagged at the preparation design.

Validation uses three supplied Helmholtz operators assembled on a 4x4 P1
triangular mesh, deliberately unsorted frequency metadata, unequal weights,
per-frequency targets, nonzero minimum-level and cost terms, complex/repeated
listener interpolation, ILU and configured Schwarz. Native production and
seat-variance evaluations match CPU direct breakdowns/gradients and finite
differences over three designs. Density-filtered three-iteration MMA matches CPU
history/design. Pre-cancellation, resume and interrupted-update recovery pass.
Warm evaluations allocate zero backend buffers. The single-frequency regression
also passes after sharing the resident implementation.

Both native tests and CUDA memcheck pass (zero errors); CPU room tests pass 331
with one ignored, and all 12 pipeline tests pass. Root check then Clippy, CUDA
test Clippy, inventory/registry validation and ten audit tests pass with existing
warnings. [Evidence](cuda-room-treatment-band-2026-10-22.json) captures the exact
source/binary/log hashes and supplied wavenumbers. No speedup is claimed.
Frequency-state memory growth, transitive quota accounting, automatic refresh,
CLI routing and broader physics/solver/BEM coverage remain open.


### Explicit preconditioner refresh without rebuilding resident solves (2026-10-22)

`CudaConfiguredFemSolveSession::try_prepare_preconditioner` validates finite
values on the canonical action pattern, creates a replacement using the retained
configuration and synchronizes setup before returning it. It does not mutate
the installed preconditioner, physical matrix/RHS, publication or warm start.
`try_refresh_preconditioner` installs a successfully prepared action, unpublishes
the solution and preserves warm-start eligibility. Transpose/Hermitian callers
supply an already oriented canonical matrix; physical values remain unchanged.

Both room-treatment session APIs expose `try_refresh_preconditioners(design)`.
The single-frequency producer prepares forward and Hermitian candidates before
either installation. The band producer prepares every candidate pair before
committing any frequency, so a reported preparation failure cannot leave a
partially refreshed band. Successful refresh retains physical CSR descriptors,
resident bases, fields and Krylov workspaces. Fields require recertification
before use; the next evaluation can warm-start from the prior certified vectors.

Tests exercise all five configured FEM strategies on P1/P2 systems, rejecting
bad dimensions, missing values and NaNs without uploads/allocations or lost
publication. Room tests cover ILU and explicit Schwarz in single/three-frequency
producers. An injected first-candidate H2D setup failure preserves installed
state. Refresh at a new design unpublishes all relevant fields; the next solve
retains solution-buffer object identity, allocates no backend buffers and
matches CPU direct objectives/gradients. The all-frequency commit ordering is
also inspected in source; a late-frequency injected failure is not claimed.

Four native tests and both corresponding memcheck runs pass with zero errors.
CPU tests pass 1964 FEM and 331 room tests (one ignored in each); all 12 pipeline
tests pass. Root check then Clippy, CUDA test Clippy, inventory/registry validation
and ten audit tests pass with existing warnings. See
[refresh evidence](cuda-treatment-preconditioner-refresh-2026-10-22.json).

This is explicit rebuild of the preconditioner only, not device factorization
or in-place numeric factor refresh. Host setup and new preconditioner uploads
remain; old and staged new actions coexist until commit. No automatic refresh
heuristic or speedup is claimed. CLI routing, general quota accounting and
broader compute/solver/FEM/BEM workflows remain open.


### Body-fitted treatment handoff routes through retained CUDA sessions (2026-10-22)

`reanalyze_projected_treatment_cuda` evaluates the explicit target baseline and
projected design on a retained target session. Geometry transfer and dimension
checks are shared with the CPU path. `promote_projected_treatment_cuda` applies
the existing non-regression gate to that complete target report.
`optimize_and_promote_body_fitted_treatment_cuda` runs source MMA, requires its
converged status, then performs independent target reanalysis/promotion. These
entry points return no report/promotion when a required solve fails; no CPU
solver fallback is introduced. The target session exposes its treatment count
for pre-solve topology admission.

The native test uses a 25-DOF source and a distinct 36-DOF target triangular FEM
discretization, with retained ILU/Schwarz strategies and a three-frequency band.
Baseline/projected target objectives and gradients match CPU direct results;
warm target reanalysis allocates no backend buffers. It rejects an explicitly
regressing projection, topology mismatch, injected target-solve failure and an
unconverged source. Recovery and a successful source-to-target handoff match CPU
results. The positive handoff smoke test uses an explicit permissive optimizer
stopping tolerance; it does not claim high-accuracy optimization convergence.

All three native treatment tests pass; memcheck for the new handoff passes with
zero errors. CPU room tests pass 331 with one ignored, and all 12 pipeline tests
pass. Root check then Clippy, CUDA test Clippy, formatting, inventory/registry
validation and ten audit tests pass with existing warnings. See
[projection evidence](cuda-treatment-projection-2026-10-22.json). No speedup,
mesh-convergence or measured-room validation is claimed. The projection remains
host geometry work; the existing retained backend performs the required solves.

### Direct CSR preparation for retained FEM and treatment sessions (2026-10-22)

Added `CudaFemSolveSession::try_from_csr` and the configuration-preserving
`CudaConfiguredFemSolveSession::try_from_csr`. Preassembled FEM callers can
supply constrained canonical CSR and the requested action's RHS directly.
Forward setup borrows host matrix values; transpose/Hermitian setup uses a
checked host CSR transpose. The generic assembled and CSR entries share
workload/native-method admission and device initialization. Existing resident
RHS/numeric update, lagged preconditioner, refresh, warm-start and publication
contracts are unchanged. Canonical input requires strictly increasing columns
per row; malformed, duplicate/unsorted, non-finite and invalid-dimension inputs
reject before device upload or preconditioner setup.

The treatment producer now uses direct CSR setup for both its forward and
Hermitian sessions. Its old CSR-to-COO-to-CSR adapter is removed. This applies to
single-frequency, band optimization and projected-treatment handoff through the
shared phase implementation. Assembly and preconditioner construction remain
host work. The legacy `solve_csr`/`solve_csr_with_guess` convenience calls remain
CPU-only; their documentation points to the explicit retained CUDA entry.
Automatic backend selection and migration of other existing producers remain
open. No new solver algorithm or workload is admitted by this change.

Validation:

- Four configured native FEM tests pass, including new direct-CSR P1/P2
  forward/transpose/Hermitian solves with ILU and configured Schwarz. The matrix
  is a constrained Helmholtz assembly with a small nonsymmetric complex
  perturbation; manufactured RHS/solutions distinguish all three actions.
  CPU matvec residuals and the existing COO CUDA entry agree. Three RHS solves
  reuse storage with zero backend allocations after warm-up. Injected RHS-copy
  failure unpublishes the field, blocks solving until repair, then recovers.
- Twelve malformed CSR/RHS cases across all three actions reject before the
  preconditioner callback, with zero backend allocations or H2D transfers.
- Five existing generic FEM native tests and three treatment native tests pass.
  The treatment tests retain their CPU objective/gradient/MMA comparisons and
  reuse/failure gates.
- CUDA memcheck passes all four configured tests and all three treatment tests,
  with zero errors in both runs.
- CPU FEM: 1,964 passed, one ignored. CPU room: 331 passed, one ignored. Pipeline:
  12 passed. Root `cargo check`, root `cargo clippy`, and CUDA FEM/room Clippy
  pass with existing warnings. Formatting and whitespace checks pass.
- Inventory remains 22 packages / 1,358 source review units. Registry validation
  and all ten audit tests pass; 40 retained-dispatch rows and zero complete rows.

Evidence: [direct CSR preparation](cuda-fem-csr-preparation-2026-10-22.json).
This removes a source-visible setup conversion; no controlled timing comparison
or end-to-end speedup is claimed. Full compute/solver/BEM/FEM coverage remains
incomplete.

### Native ILUT/ILUTP action and resolved PML qualification (2026-10-22)

The resolved-speaker default uses pivoted ILUT because ILU(0) can stall on its
PML systems. The existing native adapter covered ILU(0) only. Added
`sonium_solvers::preconditioners::CudaIlutPreconditioner`, with checked default
and configured CSR construction, import of existing host factors, and compatible
numeric refresh. Host factorization retains the CPU drop/fill/pivot policy.
Imported factors retain symmetric ordering and column pivots. Native CSR maps
gather the RHS and scatter factor-step solutions to
`perm[col_perm[step]]`; both triangular solves use the existing native row-sweep
primitive with dependency levels. A borrowed internal triangular view shares
this machinery without fabricating ILU(0) metadata or reconstructing pivots
from inverse diagonals.

All factor, permutation and scratch buffers are prepared once. Application has
zero backend allocations and zero recorded H2D/D2H transfers. Compatible refresh
retains storage and levels. Changed maps or factor patterns return false before
mutation. A partial upload invalidates application until successful repair;
validation errors preserve the old action. Setup and factorization remain host
work, allocations retain the existing backend quota policy, and long dependency
chains can require many launches. No custom kernel or speedup claim is added.
The adapter supplies the forward single-vector action; transpose/Hermitian and
block applications remain open.

Validation:

- Two native ILUT tests pass across f32/f64/Complex32/Complex64, pivoting on/off,
  and natural/non-involutive symmetric ordering. They check CPU action parity,
  three RHS applications, compatible numeric refresh, changed map/pattern
  rejection, malformed factors, partial refresh failure/recovery, and preserved
  output through an injected application launch failure.
- The existing ILU(0) all-scalar/small-pivot native test passes after the shared
  triangular refactor. Eight public facade and two prepared-FGMRES native tests
  pass, preserving the reviewed named-strategy dispatch and update contracts.
- Native prepared FGMRES solves actual resolved axisymmetric PML producer
  matrices: a 61-node/sigma-2 diagnostic mesh and the default
  191-node/sigma-18 mesh, at 400 and 450 Hz with two complex drives each.
  The model's restart, iteration limit, ILUT controls and tolerance are retained.
  Independent host original-system residuals and CPU direct field references
  pass; coarse/default solves take three/five iterations respectively. Warmed
  RHS solves allocate zero backend buffers. Frequency updates replace only the
  preconditioner while retaining the physical operator and Krylov session.
- CUDA memcheck passes ILUT (two tests), ILU regression (one), resolved PML
  qualification (one), and facade regression (eight), with zero errors in all
  four runs.
- CPU preconditioners: 241 passed. CPU facade: 117 passed. Resolved acoustics:
  all 24 tests passed. Root check, root Clippy and CUDA Clippy pass with existing
  warnings; no warnings arise in the changed CUDA files. Formatting and
  whitespace checks pass.
- The source inventory is 22 packages / 1,361 source review units. Registry source
  fingerprints are refreshed for the reviewed shared triangular changes and
  reference the new current evidence; the workflow denominator is unchanged.

Evidence: [ILUT session qualification](cuda-ilut-session-2026-10-22.json).
The public resolved-speaker continuation/full-result entry points still execute
CPU solves. Their next integration needs a retained CUDA session with explicit
selection, CPU-compatible factor reuse/rebuild behavior, correct diagnostics,
original-system certification, failure recovery, and existing postprocessing.
The tests above qualify the assembled matrices through the public persistent
FGMRES solver; they do not establish that application routing is implemented.
Low-frequency robustness, mesh convergence, product-scale performance, other
preconditioners and full CUDA workflow coverage remain open.
