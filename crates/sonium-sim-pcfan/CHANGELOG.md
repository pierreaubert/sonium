# Changelog

## Unreleased

## 0.5.0
### Fixed — annular FW-H thrust scaling

- `modal_loading_amplitude_pa` now returns the thrust-conserving
  `Δp·sinc(hπσ)`: blades covering a pitch fraction `σ` carry the
  blade-local jump `Δp/σ`, whose Fourier coefficient cancels the
  solidity factor. The previous `Δp·σ·sinc(hπσ)` under-counted every
  tone by `σ`.

### Added — machine-adaptive parallel gates

- New `parallelism::use_pcfan_parallel` gate: the finite-difference
  gradient batches (`optimizer`, `evaluation`) parallelize only above
  `PCFAN_FD_COMPONENTS_PER_THREAD × available_parallelism`
  (`SONIUM_PCFAN_PAR_THRESHOLD` absolute override, `SONIUM_SERIAL=1`
  escape hatch) instead of always-on dispatch. Both paths share one
  evaluation closure, so serial and parallel results match.
- `optimize_pc_fan_multi_de` runs a single experiment in-process (no
  thread or pool spawn); multi-experiment fan-out uses scoped threads
  over a shared experiment body with identical seeds, gated by
  `PCFAN_EXPERIMENTS_PER_THREAD`.

### Added — annular FW-H tone path

- New `annular_fwh` module: rotor-alone BPF tones through revolved
  FW-H patches over the hub-to-tip annulus, with spinning order
  `m = -h·B`, top-hat-blade modal loading
  (`Δp·σ·sin(hπσ)/(hπσ)`), and thickness neglected (thin blades, low
  tip Mach). Silent at zero thrust, toneless for a full disk
  (`σ = 1`), null on the axis for `m ≠ 0`, beaming obliquely. Audits
  the empirical tone model; the optimizer hot path is unchanged.

## 0.3.0

### Added

- MeshLang DSL geometry path (`meshlang_geometry`) — alternative pcfan geometry described in `.mlg` and compiled to Gmsh `.msh` via `meshlang-compiler`. Reference fan checked in at `shapes/pcfan.mlg` / `shapes/pcfan.msh`.

## 0.2.0

### Added

- Incompressible CFD mean-flow solver (`PcFanFlowModel::IncompressibleCfd`) on a structured FEM grid: Picard nonlinear iteration, configurable cells per axis, residual reporting, and VTU export for both volume and boundary faces.
- Immersed-penalty representation for solid fan surfaces. Boundary DOFs now count only side-wall nodes; blades, struts, hub, motor mount, and shroud walls are forced via volumetric penalties so the linear system size stays independent of fan complexity.
- Actuator-line forcing derived from blade geometry — pressure rise and airflow now respond to chord, twist, and camber changes (verified by deformation regression test).
- Fan-forced flow semantics: airflow is determined by the geometry/operating point, not by `target_airflow_cfm`. Two configs with different targets but identical geometry produce identical CFD output.
- Boundary-condition tagging: Inlet, Outlet, OuterWall, ShroudWall, SupportWall, RotatingHub, RotatingBlade — each surface gets the correct stationary or rotating wall velocity.
- Fixed-hardware noise model — frame, struts, and motor mount produce broadband noise that scales with their blockage ratio and total loss factor.
- Airflow-band lock — warm-restart loop drives the optimised design into the ±`airflow_band_fraction` window of the target.
- Spectral spread, directivity spread, operating-point spread, and tip-leakage broadband objective terms (FD-tested).
- VTU export for the full fan geometry, the CFD volume mesh, the boundary mesh, and the CFD flow solution.

### Fixed

- Adjoint shape gradient now covers the airflow (squared) penalty and tonal acoustic level. FD parity test with `relative_error < 5e-3` over blade-path, blade-section, and shroud-radius design variables.
- Optimizer is monotone-improving on the deterministic small-mesh case — objective never exceeds the baseline.
- Frame mounts are immutable across all design variables; the spline deformation explicitly preserves them.
- Shroud outer radius is clamped to stay inside the frame inner side.
- Interior blade overlap is rejected; root-contact-at-hub is accepted.

## 0.1.0

### Added

- Initial PC fan optimization workflow:
  - 120 mm baseline geometry: 8 blades, 4 frame mounts, X-shaped support struts, central motor mount, configurable shroud.
  - Reduced-order mean-flow solver (`PcFanFlowModel::ReducedOrder`) for fast objective evaluation.
  - Harmonic aeroacoustic evaluation through the FEM aeroacoustic source model, indexed by blade-passing frequency × harmonic count.
  - Spline-based design variables: blade-path (tangential + axial offsets), per-station blade sections (chord, twist, camber, thickness), shroud radius offsets.
  - Collision/merge validation for blade-blade and blade-shroud interference.
  - L-BFGS optimizer with collision and clearance penalties, configurable max iterations and tolerances.
  - `pcfan-opt` binary for command-line optimization runs.
  - Geometry VTU export for inspection.
