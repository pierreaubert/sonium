# Changelog

## Unreleased

### Added

- Campaign candidate parallelism: `FunnelConfig::stage_parallelism` and
  `Step10GateConfig::stage_parallelism` (`1` = serial legacy) solve up to N
  candidates concurrently per funnel stage and Tier-2 verification wave.
  A serial-order admission simulation plus ordered replay of cache inserts,
  history records, and budget accounting keeps rankings bit-identical to
  the serial path (histories match up to baseline-record position and
  run-dependent wall times), locked by four serial-vs-parallel equivalence
  tests covering hits, rejections, solver failures, and budget exhaustion.
  Measured on the bounded Step-10 gate (121-candidate Tier-0 screen, quota-6
  Tier-1 re-rank, 7 capped Tier-2 runs): 1.95× at width 4 with byte-identical
  QA metrics, plateauing at width 8 (serial BOBYQA refine is the Amdahl
  limit), so width 4 is the recommended campaign setting.
- `SpotCheck3dDevice` opt-in for `run_3d_spot_check`: `SpotCheck3dFidelity::device`
  selects the CPU FGMRES–Jacobi/Schur path (default, unchanged) or the tuned
  CUDA backend Chebyshev–Schur path (measured 16.9s vs 26.6s CPU on the
  47.8k-DOF probe with matching physics). CUDA runs fail loudly if any solve
  falls back to CPU.
- Non-compact port noise on 2-D slot meshes: `PermeableFwh` now
  evaluates the planar FW-H contour (`stationary_fwh_pressure_2d`)
  instead of rejecting 2-D meshes, reporting the infinite-line field
  (cylindrical `1/√r` spreading, covered by a two-observer test).

### Changed

- Snapshot-to-phasor coupling goes through the shared
  `sonium-fem` `snapshot` helper (`dft_peak_phasor`,
  `fwh_patch_from_series`): sample-identical numerics to the legacy
  `dft_phasor(...).conj()` path, locked by a cross-convention test,
  with fail-closed errors replacing silent zeros on bad input.
