# Changelog

## Unreleased

### Fixed

- Corotational shell force restored to the classic frame-rotation
  transform; the (dq/du)^T energy-gradient projection double-counted
  frame spin and degraded the elastica strip benchmark under refinement.
- Taylor-bar Wolfram golden regenerated for the dissipative bulk-viscosity
  sign (sigma - q I): the script identity and impact-face resultants now
  use (szz - q).


## 0.7.0
### Added

- Lagged BDF preconditioner setup reuse (`NavierStokesConfig::linear_solver_setup_max_lag_steps`,
  default 0 = legacy rebuild-on-any-change): the JacobiSchur setup is reused
  across steps while matrix drift stays within
  `linear_solver_setup_lag_tolerance`, with a retry-once fresh rebuild when a
  lagged solve fails to converge. 128-step 3D smoke validation: 128 → 43
  fresh setups, identical 81,081 total iterations, energy-closure agreement
  to 10 digits, ~10% wall savings (12.8 → 11.5 min at 32 threads).
- ILU(0) velocity-block saddle-point variant: `BlockPreconditionerConfig::Ilu`
  plus the `LinearSolverCpuPreconditionerKind::IluSchur` /
  `BdfLinearSolverKind::FgmresIluSchur` path through Picard, BDF, and backend
  fallback. On the 47.8k-DOF probe it cuts iterations 1814 → 580 at
  identical physics, but its 3.7× setup cost nets +42s setup+solves even
  with lag-5 at 128-step smoke scale (25,677 vs 81,081 iterations), so
  JacobiSchur stays the default and ILU(0) remains an opt-in for
  iteration-dominated regimes (unit test: ILU(0) block beats diagonal
  block on Stokes-1D).
- BDF body-force assembly cache (`NavierStokesConfig::cache_body_force_assembly`,
  default off): time-independent force RHS/Jacobian quadrature runs once per
  workspace and is reused across steps, guarded by a per-step sample
  fingerprint that reassembles on any mismatch (bitwise-equivalence and
  stale-guard tests).
- BDF phase timing on `SONIUM_BDF_SOLVER_PROGRESS=1`: new `BDF_ASSEMBLY_DONE`
  (with pattern-cache flag), `BDF_AUDIT_DONE`, and `BDF_WORKSPACE_DONE`
  lines, plus `setup_ms`/`solve_ms` on the linear setup/solve lines; the
  Schur-arm solver tag now reports `fgmres-ilut-schur` for IlutSchur.
- Added machine-adaptive `parallelism::ParallelPolicy` for assembly merges:
  the rayon gate scales with `available_parallelism`
  (`SONIUM_FEM_PAR_THRESHOLD` absolute override, `SONIUM_SERIAL=1` escape
  hatch) instead of a fixed element count.
- `assemble_bilinear` and `assemble_bilinear_per_element` now merge large
  per-element triplet lists on rayon workers via order-preserving sharded
  concatenation, so the push order — and duplicate summation in `to_csr` —
  is bitwise identical to the serial merge; small assembles stay serial.
- Element-tabulation and pointwise-combine loops (generic, region-routed,
  legacy mass/stiffness/Helmholtz assemblers, prebaked coefficient
  application, Helmholtz shift assembly, transport combine, nonlinear
  residual/tangent) auto-select serial below the machine-adaptive
  `ELEMENTS_PER_THREAD` / `POINTWISE_VALUES_PER_THREAD` grains via the new
  `collect_indexed` / `filter_collect_indexed` helpers; both paths visit
  indices in order, so results are identical.

### Fixed

- Capped the `ParallelPolicy::decide` gate at `MAX_PARALLEL_GATE_THREADS = 16`:
  the uncapped `grain × pool threads` threshold serialized mid-size meshes on
  many-core machines (a 10374-tet mesh needs 32768 elements to thread on a
  128-thread pool). Pools with 16 or fewer threads behave exactly as before;
  measured 22.5s → 1.2s per P2 BDF assembly with identical results.
- The CPU JacobiSchur/IlutSchur fallback in `solve_iterative_cpu` now builds
  the external Schur block with ILUT (same settings as the BDF cached path,
  diagonal fallback preserved) instead of point Jacobi: 16128 → 442
  iterations on the 47.8k-DOF port probe with identical converged residuals.
- Fixed the serial triplet-merge bottleneck after parallel element
  tabulation in bilinear assembly: large merges no longer single-thread the
  scatter into the global `TripletMatrix`.
- Replaced the `thread_local! UnsafeCell` quadrature-rule cache with a
  process-wide `OnceLock<Mutex<..>>` table returning owned clones: no more
  `'static` lifetime laundering or aliasing risk under rayon worker churn.
- `ElasticityForm`/`ViscoelasticForm` 2D assembly without a plane
  assumption now panics with a contract message (`plane_2d`, covered by a
  `should_panic` test) instead of a bare `Option::unwrap`; all `% dim`
  index math routes through the documented `vector_component` helper plus
  `BasisTabulation::grad`/`vector_value` accessors with debug layout
  asserts.
- Reconciled `stabilisation_tau` docs with the code: the viscous term is
  `C_inv·ν²/h⁴` with `C_inv = 36` absorbing the textbook `9·C` factor
  (Tezduyar/Osawa, Codina variants), pinned by a hand-computed unit test.
- `AdvectionForm` documents the `[u_x, u_y, …]` closure contract and trips
  a `debug_assert` on short returns instead of silently truncating.
- The 22 meshlang-transfer integration tests declare
  `required-features = ["meshlang"]`, so default-feature
  `cargo check --all-targets` no longer fails with E0432.
- The Navier–Stokes backend path uploads the saddle-point preconditioner
  via `try_from_csr_host_parts_with_pressure_preconditioner` and falls
  back to CPU with a warning instead of aborting on allocation failure.
- Fixed a tabulation rebuild storm that made multi-element H(curl)
  tabulation never finish: `HcurlSpace::try_tabulate` rebuilt the
  whole-mesh edge/face tables on every quadrature point (directly, and
  indirectly through per-call full-mesh validation). Edge signs are now
  computed locally in O(edges) — bit-identical to the global rule,
  pinned by a unit test — and validation is the caller's once-up-front
  duty per the `FunctionSpace` contract. The 48-tet exact-sequence
  check went from >30 min (killed) to ~3 s.

### Added (physics)

- New `JohnsonChampouxAllard` rigid-frame porous model
  (`physics::acoustics::impedance`): five-parameter microstructure
  equivalent fluid (σ, φ, α∞, Λ, Λ′) with `effective_density` /
  `effective_bulk_modulus` accessors, wired into the shared
  `PorousMaterialModel` trait (characteristic impedance, propagation
  constant, backed-layer surface impedance, Robin coefficient). Verified
  against an independent published-data-backed oracle (Z_c/z0 =
  2.321+2.075j at 200 Hz for the σ=20 kPa·s/m², φ=0.98, α∞=1,
  Λ=Λ′=87 µm specimen), DC/high-frequency analytical limits, audio-band
  passivity, and agreement with the in-crate Miki fit.
- "Which model when" decision docs for mean-flow acoustics
  (`physics::aeroacoustics`: LPF vs LEE vs LNS vs Lighthill/FW-H vs DRP,
  with the Galbrun/APE absence noted and justified) and for viscothermal
  modelling (LRF vs BLI vs full LNS).

### Added — aeroacoustic analogies and propagation

- Lighthill radiation in 2-D, 2-D-axisymmetric, and 3-D
  (`aeroacoustics::lighthill_2d_axi`): planar quadrupoles through the
  Hankel kernel plus spinning ring sources of any azimuthal order
  sharing the 3-D kernel sample by sample.
- Distributed quadrupole sums over mesh cells (`distributed`):
  per-cell stress moments through the point/planar evaluators, with an
  explicit compact-depth 2-D-to-3-D extrusion documenting the `d ≪ λ`
  approximation.
- Stationary permeable FW-H in 2-D and axisymmetric form
  (`fwh_2d_axi`): planar contours, spinning revolved strips of any
  azimuthal order, and uniform-flow convected Kirchhoff surfaces
  carrying `(p, ∇p)`.
- Annular-duct modal layer (`duct_modes` + `sonium-wave` annular modes):
  hard/soft-wall cut-on with flow, lined-wall roots, rotor-stator
  cut-on screening, and Tyler–Sofrin interaction orders; propeller
  rotor-alone/interaction directivity in `propeller_source`.
- Axisymmetric frequency-domain LEE with azimuthal mode `m`
  (`axisymmetric_lee`: `AxiLeeSystem`, meridional divergence,
  axis-aware DOFs) and spinning-mode axisymmetric LPF sweeps
  (`AxiLpfModeAssembler`, azimuthal mass form).
- Tam–Webb DRP Euler on structured 2-D/3-D Cartesian grids
  (`drp_structured`, characteristic outflow plus sponge) and
  axisymmetric modal pairs (`drp_axi`, `cos mφ`/`sin mφ`), with shared
  energy-centroid group-velocity diagnostics (`wave_diagnostics`).
- Unstructured P1–P3 DG-LEE on triangles/tetrahedra (`dg_lee`): exact
  upwind flux, SSPRK(3,3), outflow/slip-wall/prescribed boundaries,
  manufactured-solution convergence at degree-plus-one, packet
  group-velocity and wall-reflection gates, and a structured/DG
  equivalence gate.
- Snapshot-to-phasor coupling (`snapshot`): Hann DFT peak phasors,
  Lighthill stress tensors, FW-H patches/contours, azimuthal-mode
  extraction, and ring/revolved-strip assembly from spinning data.

### Fixed — aeroacoustics

- 3-D LEE/LNS continuity block corrected from `+ρ₀c₀²Bᵀ` to
  `-ρ₀c₀²Bᵀ` (integration by parts against the divergence form);
  pinned by a plane-wave regression test whose negative control fails
  under the old sign.

## 0.5.30

### Added

- Added first-class FEM physics modules for porous flow/poroelasticity, species transport,
  thermoelasticity, compressible-flow building blocks, turbulence transport, phase-field fracture,
  electrochemistry, semiconductor drift-diffusion, shallow-water waves, and composite laminates.
- Added FEM-side speaker multiphysics contracts for axisymmetric pressure/elasticity/shell/coupling
  forms, magnetic bias and perturbation workflows, distributed Lorentz force assembly, solid-shell
  ties, acoustic-structure boundary coupling, cyclic sector constraints, thermoviscous gap model
  selection, coupled frequency-sweep reuse, and stable speaker result extraction.
- Added generalized FEM QA coverage for Maxwell, electrostatics, biharmonic plates, shells,
  magnetostatics, nonlinear solids, plasticity, piezoelectricity, contact, aeroacoustics,
  electroacoustics, Navier-Stokes, and lumped-circuit analytical benchmarks.
- Added shell distributed pressure/load assembly, shell point loads, 6-DOF shell boundary helpers,
  and curved shell surface mesh generators.
- Added biharmonic Hessian cell assembly and plate load helpers.
- Added DSG/MITC-style projected shear stabilization for P1 triangular Mindlin plate elements,
  preserving the compatible shear formulation via `new_consistent` and validating the stabilized
  simply supported sinusoidal plate QA against the Navier Mindlin solution.
- Added Maxwell current-source helpers and standalone Maxwell PML assembly.
- Added electrostatic point-charge/image helpers, two-cylinder mesh support, contact reference
  helpers, and additional analytical QA formulas.

### Fixed

- Shell pressure loads now evaluate spatially varying load functions at the global element
  centroid instead of local coordinates.
- Q2 quadrilateral and Hex27 H1 tabulation now provide physical Hessians, preventing biharmonic
  cell stiffness from silently assembling as zero on tensor-product P2 meshes.
- Maxwell PML assembly now applies anisotropic curl-curl stretching and a separate mass
  coefficient instead of uniformly rescaling the full element contribution.

## 0.5.29

### Added

- Added transient thermal time stepping for `M dT/dt + K T = f`, including backward Euler,
  Crank-Nicolson, reusable effective-matrix/RHS helpers, and a `thermal_bdf_step` convenience API.
- Added vector traction Neumann assembly for elasticity `VectorH1Space` systems.
- Added scalar Saint-Venant torsion helpers for Prandtl stress-function and warping-function
  formulations.
- Added standalone Helmholtz PML assembly via `PmlHelmholtzForm`.
- Added spherical Bessel/Hankel helper wrappers for QA analytical scattering checks.
- Added axisymmetric magnetostatic `A_theta` forms for azimuthal solenoid/coil current,
  distributional point-dipole source assembly, and a tagged C-core mesh generator.

## 0.5.28

### Fixed

- `CachedShapeKey` now uses exact floating-point equality consistent with its hash, restoring the
  `HashMap` Eq/Hash invariant for shape-function caching.
- Added P2 tetrahedron shape-function evaluation and gradients, and made unsupported H1
  element/degree combinations fail explicitly instead of being advertised by DOF mapping and
  failing later during tabulation.
- 15-point tetrahedral quadrature rule (order >= 5) used coordinates with negative barycentric
  values (outside the reference domain) and wrong weight sum. Replaced with the correct Keast
  (1986) degree-5 15-point rule: 4 orbits (centroid, two S31, one S22), all coordinates
  non-negative, weights sum to 1/6.
- `BodyForceForm` in elasticity indexed shape function values as `values[dof * dim + c]` instead
  of `values[dof]`, reading the wrong shape function for all non-x components.
- `VectorMassForm` had the same incorrect indexing pattern (`values[a * dim + comp_a]`). Fixed
  to `values[a]`.
- Added comment documenting the RT0 basis function normalization issue (missing factor of 2 for
  triangles, 3 for tetrahedra) and incorrect divergence values.
- Added comment documenting that the multigrid restriction operator uses `R = P^T` without
  variational scaling, which may degrade convergence for non-uniform meshes.

### Known Issues

- RT0 (Raviart-Thomas) basis uses `N_i = λ_i ∇λ_i` instead of the correctly normalized
  `N_i = dim · λ_i ∇λ_i`. Divergence values are also wrong. This affects H(div) conforming
  discretizations.
- Neumann BC for 3D triangle faces uses centroid (1-point) quadrature, which is inconsistent
  with the consistent mass matrix used in Robin BCs.
