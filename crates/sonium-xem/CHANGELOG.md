# Changelog

## Unreleased

## 0.5.0
### Fixed

- `runtime::observe_execution_contract` parks its 10 ms overlap window with
  `thread::sleep` instead of busy-spinning, and demoted its counters from
  `SeqCst` to acquire/release ordering.

### Added

- `io::try_sonium_output_root` / `io::try_binary_output_dir`: fallible twins
  of the `expect`-based directory resolvers for library callers.
- `runtime::ensure_global_rayon_pool`: idempotent pool installation that
  reports the installed width instead of failing on second init.
- `output::try_generate_spatial_slices_par`: machine-adaptive rayon
  parallelization over solutions sharing one evaluation core with the
  serial path (`parallelism::ParallelPolicy`:
  `SLICE_CELLS_PER_THREAD × available_parallelism`,
  `SONIUM_XEM_PAR_THRESHOLD` override, `SONIUM_SERIAL` escape hatch —
  fixed constants cannot be right per machine); output matches element-wise
  on both paths (`slices_core_matches_on_both_paths`).
- `ExternalMultiphysicsDocument::{try_id, try_domains, try_fields_array}`:
  fallible envelope accessors so validator/accessor drift becomes
  `MissingField` instead of a panic.
- `tests/xem_review_20260904.rs`: boundary-validation, ambient-medium
  propagation, output-dir twin parity, pool idempotency, par/serial slice
  parity (both sides of the gate), and accessor parity regressions.
- Documented the panicking-vs-`try_add_*` contract on the `vtu` module.

### Changed

- Fixed stale `AGENTS.md` (crate name `sonium-xem`, lib `sonium_xem`,
  version 0.4.2, real test commands).

## 0.4.2

### Fixed

- Panic on empty `listening_positions` in `create_output_json`, `create_output_json_with_sources`,
  and `generate_spatial_slices`. Functions now assert the vec is non-empty and serialize all
  listening positions (not just `[0]`).
- Division by zero / NaN when `elements_per_meter == 0` in `generate_mesh`. Added a panic with
  a descriptive message.
- `graded_parameter` was discontinuous at `t = grading` and `t = 1 - grading`. Rewrote all three
  branches to ensure C0 continuity: `t²/grading` (ramp), identity (linear middle),
  `1 - u²·grading` (symmetric ramp). Verified `f(0)=0`, `f(1)=1`.
- `log_space` and `lin_space` returned `[start]` when `num == 0` instead of an empty vec.
  Now returns `Vec::new()` for `num == 0` and `vec![start]` for `num == 1`.
- `create_output_json` and `create_output_json_with_sources` now serialize all listening positions
  as `"listening_positions"` (array) and per-LP frequency responses as `"frequency_responses"`.
- Added row-length validation for jagged `magnitude` arrays in `DirectivityConfig::Custom`.
