# Changelog

## Unreleased

## 0.2.0
### Added

- Machine-adaptive frequency-lane gate (`use_frequency_lanes`,
  `FREQUENCY_SWEEP_ENTRIES_PER_THREAD`): small sweeps (`n_freqs × n_dofs²`
  below `entries_per_thread × available_parallelism`) now cap to one lane
  instead of fanning out per-lane operator/ILU/subspace state, via
  `SONIUM_SPEAKER_PAR_THRESHOLD` override and `SONIUM_SERIAL=1` hatch.
  `max_frequency_lanes: None` continues to derive the lane cap from the
  thread and memory budgets inside `plan_lanes`.
- New `ThdConfig::validate` (mirrors `CompressionConfig::validate`);
  `measure_thd` routes its input guard through it. Criterion bench
  `frequency_lane_scaling` pins the lane-gate crossover.

### Changed

- `optimization` is no longer gated behind the `cli` feature (it never
  used `clap`); `--no-default-features` builds expose it.
- `SimulationResult::woofer/tweeter_electrical_impedance_ohm` are
  `#[deprecated]` (prescribed-result compat only); producers must fill
  `driver_electrical_impedances_ohm` (woofer = driver 0, tweeter = 1).
  Metrics prefer the matrix whenever it carries data; all in-workspace
  compat readers carry targeted `#[allow(deprecated)]`.

## 0.1.1

### Fixed

- Large BEM solves now verify the true dense residual after FGMRES+ILU and reject non-converged
  solutions instead of returning acoustic fields from failed iterations.
- CEA-2034 curves and preference metrics now delegate to `sonium_roomeq::cea2034::score_peq`,
  using the CTA listening-window and early-reflection groups.
- Sound Power now uses 70 unique CTA measurement rows with spherical-slice weights instead of
  azimuthal `|cos(angle)|` weights.
- Woofer cone geometry is now honored when a tweeter waveguide is present.
- Cone shell triangle winding now matches the analytical cone node normals, so element normals
  point toward the front acoustic domain.
- FGMRES solution recycling now resets across frequency steps wider than a quarter octave, avoiding
  stale initial guesses on sparse sweeps.
- Default BEM solving now switches from dense LU to residual-checked FGMRES+ILU above 3000 DOFs,
  avoiding cubic factorization cost for medium meshes.
