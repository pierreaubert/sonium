# Changelog

## Unreleased

## 0.6.0
### Added

- `suggested_sphere_terms_for_grid`: grid-aware Mie-series truncation
  that accounts for the observer radius (`n > kr` at the observer),
  for cases where `suggested_sphere_terms(ka)` under-counts at small
  `ka` with far observers.

### Fixed

- `run_richardson` reuses one scratch buffer across iterations instead of
  cloning a full-grid `domain.medium(&x)` per step; operation order and
  numerics are unchanged.
- Domain construction shares a single FFT planner across the three
  `laplace_kernel_axis` calls; new `laplace_kernel_axis_with_planner` exposes
  the planner-reusing core for repeated constructions.

### Added

- `tests/wavesim_convergence.rs`: 1D homogeneous point-source solve vs the
  `analytical_1d` discrete-kernel reference (residual + relative error).
- `tests/kernel_consistency.rs`: adjoint-consistency
  (`dg_dnx == -dg_dny` for shared normals) and coincidence policy
  (bare-kernel infinite sentinel vs checked-`try_*` rejection).
- Machine-adaptive rayon gates (`parallelism::ParallelPolicy`, replacing
  any fixed threshold thinking): FFT lane loops and pointwise passes
  (medium application, propagator divide, IFFT normalization) parallelize
  when `work_items >= items_per_thread × available_parallelism`, with
  `SONIUM_WAVE_PAR_THRESHOLD` override and `SONIUM_SERIAL` escape hatch.
  Strided axes gather into one scratch buffer, transform lanes in parallel,
  and scatter back; serial and parallel paths agree bitwise
  (`parallel_lanes_match_serial_bitwise`, `pointwise_kernels_match_on_both_paths`).

### Added — aeroacoustic kernels

- Annular-duct modal analysis (`analytical::annular_duct_modes`):
  hard/soft-wall radial modes with uniform axial flow, Tyler–Sofrin
  rotor-stator spinning orders, and axial-branch selection.
- Azimuthal-mode bookkeeping (`analytical::azimuthal::AzimuthalMode`):
  spinning orders with axis-power rules and `exp(imφ)` modal phases
  shared by rings, revolved patches, and modal propagators.
- Azimuthal ring quadrature (`special::ring`): fixed and adaptive
  `∫f(φ)exp(imφ)dφ` rules driving the spinning ring and revolved
  evaluators.
- 2-D Helmholtz kernel derivatives (`special::helmholtz`): Hankel
  gradient, normal derivative, and Hessian (bare and checked forms)
  plus uniform-flow Garrick convected kernels and a time-convention
  flip helper.

## 0.4.1

### Fixed

- Division by zero at `ka = 0` in `compute_rigid_sphere_coefficients`, `oscillating_sphere_3d`,
  and `impedance_sphere_scattering_3d`. Added guards that return the correct low-frequency limit.
- Phase sign mismatch in `damped_wave_1d`: implementation computed `exp(-αx)·exp(+ikx)` but the
  documented formula is `exp(-(α+ik)x) = exp(-αx)·exp(-ikx)`. Fixed to match the documented
  convention.
- Added `speed_of_sound` parameter to `impedance_sphere_scattering_3d` so the ka≈0 guard can
  compute the correct frequency for the returned `AnalyticalSolution`.
- Added clarifying comment on the Giebermann continued-fraction `gnu` formula in
  `spherical_hankel_first_kind` noting the convention matches NC_SphericalHankel.
