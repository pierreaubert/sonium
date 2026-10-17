# Changelog

## Unreleased

## 0.2.0
### Added — machine-adaptive parallel gates

- New `parallelism::use_propeller_parallel` gate
  (`SONIUM_PROPELLER_PAR_THRESHOLD` absolute override, `SONIUM_SERIAL=1`
  escape hatch): blade replication, harmonic spectrum evaluation, and
  radial-row integration parallelize only above per-thread grains
  (`REPLICATED_PANELS_PER_THREAD`, `HARMONICS_PER_THREAD`,
  `INTEGRATION_POINTS_PER_THREAD` × `available_parallelism`). All loops
  collect into index order before reduction, so checksummed artifacts are
  identical on both paths.

### Added — FW-H audit

- Cross-check test pinning the far-field-only `fwh_reference` against
  the full-kernel `sonium-fem` stationary FW-H evaluator on identical
  elements: thickness agrees to roundoff (identical closed form) and
  loading/total within 2% at `kr ~ 180`, documenting the `1/R`
  near-field scope difference.

## 0.1.0

### Added

- Axisymmetric source-noise model for shrouded propellers (`source_model`):
  - `PropellerDesign` wraps `sonium_fem::physics::aeroacoustics::PropellerSource`.
  - `PropellerPlanform::ToroidalLoop` — modelled as a source-generation modifier (reduces open-tip vortex shedding) since the axisymmetric propagator cannot geometrically represent a loop blade.
  - `EdgeTreatment` — leading- and trailing-edge serrations as bio-inspired noise-reduction modifiers (LE pre-conditions incoming turbulence; TE breaks coherent wake eddies).
  - `SerrationGeometry` — amplitude / wavelength / span-fraction parameters with smooth-window effectiveness clamps. Default = `owl_like_micro_airfoil`.
  - `EvaluationMode::FastSurrogate` (optimization hot loop) and `ResearchControl` (higher-order quadrature, coherence control).
  - `PropellerSpectrum` and `HarmonicNoise` outputs.
- 3D blade representation (`blade3d`):
  - `Blade3dConfig`, `BladeMesh`, `BladePanel`, panel-level `PanelAcousticSummary`.
  - `compare_panel_acoustics` for delta studies.
  - Per-panel parallelism via `rayon`.
- 3D shape optimization (`optimization3d`):
  - `BladeShapeVariables` (chord, twist, camber, sweep, dihedral …), `Propeller3dOptConfig`.
  - `evaluate_blade_shape` and `optimize_blade_shape` driver.
  - `propeller-3d-opt` binary.
- ParaView (VTU) output for both 3D blade meshes and aeroacoustic surface fields, aligning with the workspace-wide migration to ParaView.
