# Changelog

## Unreleased

## 0.2.0
### Added — machine-adaptive parallel gates

- New `cpu::parallelism::use_compute_parallel` gate
  (`SONIUM_COMPUTE_PAR_THRESHOLD` absolute override, `SONIUM_SERIAL=1`
  escape hatch) replacing the fixed `SPMV_PAR_THRESHOLD = 1024` row
  cutoff: CSR SpMV/SpMM parallelize above `SPMV_ROWS_PER_THREAD` /
  `SPMM_OUTPUTS_PER_THREAD × available_parallelism`, and
  `bem_cpu::regular_pair_batch` bands pair lists above
  `BEM_PAIRS_PER_THREAD × available_parallelism`. All gated loops keep
  one output row/pair per worker accumulated serially, so both paths are
  bitwise identical. The transpose-GEMV axpy fan-out stays serial by
  design (shared accumulator — sharding would make results
  machine-dependent) and now says so.

### Fixed — roofline, NaN semantics, singular diagnostics

- New `perf::csr_spmv_model_ncols` overload: the source-vector traffic
  term now uses the true column count instead of `nnz`, so SpMV
  arithmetic intensity is correct for rectangular matrices (the legacy
  `csr_spmv_model` keeps the square-matrix proxy). The roofline bench
  uses the accurate overload.
- `gemv_trans` (f32/f64) no longer skips `xj == 0.0` terms, so
  `0 * NaN` matrix entries propagate NaN exactly like the reference
  accumulation (regression test
  `gemv_trans_propagates_nan_with_zero_x`).
- New `bem_cpu::SINGULAR_SKIP_CUTOFF` constant plus
  `regular_pair_batch_with_skipped_count`: touching/adjacent triangle
  pairs integrated with dropped quadrature points are now countable
  instead of silently biased (test
  `skipped_count_flags_touching_pairs`).

### Changed — panicking constructors deprecated, bench registry, vulkan

- The infallible `from_host*` / `copy_values_from_host` /
  `record_input` / saddle-point `from_csr_host_parts*` constructors are
  `#[deprecated]` in favour of their `try_*` variants: every remaining
  production caller (notably the Metal CSR uploads in
  `sonium-sim-room`) is migrated, and in-crate fixtures carry targeted
  `#[allow(deprecated)]`. QA binaries, scaling bins, and examples now
  use `try_*` with typed propagation; solver tests/benches carry
  targeted `#[allow(deprecated)]`. Zero compute deprecation warnings
  workspace-wide; `scripts/check_compute_deprecated_callers.sh` fails
  review on new non-allowlisted callers.
- Monolith split (mechanical, ABI-preserving): inline test modules moved
  out of `backend.rs`, `metal/mod.rs`, and `cuda/cublas.rs` into
  `backend_tests.rs`, `backend_reduction_contract_tests.rs`,
  `metal/tests.rs`, and `cuda/cublas_tests.rs`; the cuBLAS BLAS-1 and
  BLAS-2/3 kernel tables moved to `cuda/cublas_blas1.rs` and
  `cuda/cublas_blas23.rs`. Parent files shrink from 9481/10210/18363 to
  6912/7363/15187 lines with identical module paths; the move exposed
  and fixed three vacuous source self-matches in the CUDA/Metal audit
  tests (needles now point at real production lines).
- Production split of `backend.rs` (mechanical, namespace-preserving):
  `backend.rs` becomes `backend/mod.rs` plus `backend/capabilities.rs`
  (capability tables), `backend/views.rs` (view descriptors),
  `backend/validate.rs` (checked-argument validators), and
  `backend/host.rs` (host fallback kernels), re-exported so every
  `crate::backend::X` path keeps resolving. `mod.rs` shrinks from 6912
  to 4464 lines. The only namespace delta is the removal of two
  re-exported-but-uncallable `#[allow(dead_code)]` helpers
  (`host_axpy`, `host_scale`); the CUDA-gated
  `validate_host_csr_structure` re-export is now `#[cfg(feature =
  "cuda")]`, and the test-only `host_syrk_compute` /
  `host_trsm_compute` re-exports are `#[cfg(test)]`. The split also
  repaired a stale `include_str!("../backend.rs")` in
  `cuda/cublas_tests.rs` (now `../backend/mod.rs`).
- Capability negatives: CPU/CUDA/Metal parity tests now pin every
  non-Complex64 hierarchical (`hmatrix*`, `fmm*`) cell, plus Metal's
  fully-unsupported forward tables.
- `Cargo.toml` sets `autobenches = false` and registers the two live
  criterion benches (`bem_regular_pair_compare`,
  `block_gmres_accelerator`) with `harness = false`; the
  `benches/perf_report.rs` helper is no longer a phantom bench target.
- The reserved no-op `vulkan = []` feature is removed until a backend
  exists; requesting it now fails the build instead of silently
  building to nothing.

## 0.1.5

- removed ndarray and other libraries : do everything in-house
- metal + sme2 are well supportedx

## 0.1.4

### Added

- `cuda` feature now refers to a **mature cuBLAS / cuSPARSE backend** built on `cudarc` 0.19. Drives `cublas::sys::culib()` for BLAS-1 (`axpy`, `dot`/`dotc`, `nrm2`, `scal`) across `f32` / `f64` / `Complex32` / `Complex64` and `cusparse::sys::culib()` for the modern generic CSR `cusparseSpMV`. `copy_d2d` is a real `cuMemcpyDtoDAsync_v2` via cudarc's safe `memcpy_dtod`, no longer host-staged.
- `tests/parity.rs::complex::cuda_matches_cpu_complex32` and `cuda_matches_cpu_complex64` — CPU vs cudarc parity for the new complex BLAS-1 + complex SpMV paths. Both skip gracefully when no NVIDIA driver is present.

### Changed

- `BufferElement` now requires `Default`. All in-tree impls (`f32`, `f64`, `Complex32`, `Complex64`, `i32`, `u32`, `usize`) already satisfy it; the bound enables host-staging buffers and is transparent for the cudarc path.

## 0.1.3

### Added

- Initial `Backend` trait abstraction with BLAS-1 (`axpy`, `dot`, `nrm2`, `scale`) and CSR SpMV (`spmv_csr`) for `f32`, `f64`, `Complex32`, `Complex64`, and `f16`.
- `MemoryPool` allocator with strict per-device byte quotas, defaulting to 80 % of `total_device_memory_bytes()`. Quota overruns return `OomError::QuotaExceeded` before the driver is touched, so the host process is no longer killed by CUDA OOM.
- `BufferElement` / `DeviceScalar` generic element abstraction, replacing per-backend type duplication.
- `CpuBackend` — `Vec<T>`-backed host buffers, always available, used as the parity reference.
- `CudaBackend` (feature `cuda`) — NVIDIA backend via `cudarc`. Compiles on any host (driver loaded dynamically) and exposes the full BLAS-1 + SpMV trait surface on real CUDA contexts.
- `MetalBackend` (feature `metal`) — Apple Metal backend via `objc2-metal`. Validation-grade BLAS-1 + SpMV with host-staged CPU fallback for some kernels; device alloc/copy/sync go through the real Metal API.
- Cross-backend parity test suite (`tests/parity.rs`) comparing CPU vs CUDA vs Metal output on BLAS-1 and SpMV for all supported scalar types.
- `tests/quota.rs` covering pool accounting, refund on driver OOM, and explicit-cap fallback when device memory discovery is unavailable.

### Known Issues

- BLAS-2/3 (`gemv`, `gemm`) and FFT are intentionally not on the trait yet; they will be added once a second backend is in production use.
- Metal SpMV is host-staged (CPU fallback), not a Metal compute kernel. Bandwidth-bound numerical workloads should prefer CPU or CUDA on this backend.
