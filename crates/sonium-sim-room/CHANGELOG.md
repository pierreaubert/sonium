# Changelog

## Unreleased

## 0.5.0
### Added — machine-adaptive parallel gates

- FEM sweep source fan-out (per-source solves and block-solve RHS
  assembly) runs serially below half the Rayon pool width and lets nested
  assembly/GMRES/ILUT parallelism use full cores
  (`SONIUM_SIMROOM_PAR_MIN_SOURCES` absolute override, `SONIUM_SERIAL=1`
  escape hatch), instead of always-on outer dispatch oversubscribing the
  pool with 2–10 sources.

## 0.4.3

- added a set of parameters
- deal with limited memory on mac

## 0.4.2

### Added

- Exposed a selectable room BEM assembly mode so downstream room simulations can run either
  the fast legacy collocation approximation or the core TBEM formulation with singular
  integration and Burton-Miller stabilization.

### Fixed

- BEM mesh resolution increased from 3 to 8 elements/meter. At 3 elem/m the mesh had only
  0.5 elements/wavelength at 2 kHz, producing completely unreliable results. At 8 elem/m
  the mesh is adequate up to ~500 Hz. Added a warning comment noting results above 500 Hz
  remain qualitative.

### Known Issues

- HF extension `is_sub` detection uses `starts_with("sub")`, missing grouped sub sources named
  `left_sub` / `right_sub`. These sources incorrectly receive full-range HF extension instead
  of a steep sub rolloff.
- FEM impedance BC uses hardcoded air density `1.21` kg/m³ instead of the project constant
  `AIR_DENSITY_20C = 1.204`.
