//! Resident sum-factorized hex actions with deterministic shared-DOF reduction.
//!
//! One block per element/RHS evaluates tensor contractions in shared memory.
//! A second kernel reduces element outputs to global DOFs without atomics.
//! Scratch is allocated once for a caller-selected maximum RHS count.

// Rust guideline compliant 2026-02-21
use crate::cuda::{CudaBackend, CudaBuffer};
use crate::fem_hex::{HexPartialC64, invalid};
use crate::pool::Buffer;
use crate::{
    Backend, BackendError, BackendResult, BufferElement, DeviceAdjointOperator, DeviceBufferLike,
    DeviceOperator, MemoryPool, Transpose,
};
use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use num_complex::Complex64;
use std::sync::{Arc, Mutex};
const SOURCE: &str = include_str!("kernels/fem_hex.cu");
const OP: &str = "cuda_hex_partial";

pub(super) struct CudaHexKernels {
    action: CudaFunction,
    scaled_action: CudaFunction,
    scatter: CudaFunction,
    material: CudaFunction,
}

impl CudaBackend {
    /// Prepare and retain hex kernels before allocating operator storage.
    ///
    /// Reuses this backend's module on subsequent calls. Driver module storage
    /// is separate from pooled arrays; no mesh buffers are allocated or uploaded.
    ///
    /// # Errors
    /// Rejects graph capture and propagates pending, compilation and symbol errors.
    pub fn try_prepare_hex_kernels(&self) -> BackendResult<()> {
        self.hex_kernel_functions().map(|_| ())
    }

    fn hex_kernel_functions(&self) -> BackendResult<Arc<CudaHexKernels>> {
        self.check_pending_error()?;
        if self.stream().capture_status().map_err(failed)?
            != cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
        {
            return Err(invalid("prepare hex kernels before graph capture"));
        }
        let mut cache = self.hex_kernels.lock().map_err(failed)?;
        if let Some(kernels) = cache.as_ref() {
            return Ok(Arc::clone(kernels));
        }
        let ptx = crate::backend::catch_backend_panic(OP, || compile_ptx(SOURCE))?.map_err(failed)?;
        let module = self.context().load_module(ptx).map_err(failed)?;
        let kernels = Arc::new(CudaHexKernels {
            action: module.load_function("sonium_hex_partial_c64").map_err(failed)?,
            scaled_action: module.load_function("sonium_hex_partial_scaled_c64").map_err(failed)?,
            scatter: module.load_function("sonium_hex_scatter_c64").map_err(failed)?,
            material: module.load_function("sonium_hex_material_c64").map_err(failed)?,
        });
        *cache = Some(Arc::clone(&kernels));
        Ok(kernels)
    }
}
enum Output<'a> {
    Action,
    Lift(&'a CudaBuffer<Complex64>),
    Residual(&'a CudaBuffer<Complex64>),
    Jacobi {
        rhs: &'a CudaBuffer<Complex64>,
        inverse: &'a CudaBuffer<Complex64>,
        omega: f64,
    },
}
fn failed(e: impl std::fmt::Debug) -> BackendError {
    BackendError::OperationFailed {
        operation: OP,
        reason: format!("{e:?}"),
    }
}
fn upload<T: BufferElement>(
    pool: &MemoryPool<CudaBackend>,
    values: &[T],
) -> BackendResult<Buffer<T, CudaBackend>> {
    let mut b = pool.try_alloc(values.len())?;
    pool.backend().try_copy_h2d(values, b.raw_mut())?;
    Ok(b)
}
fn raw<T: BufferElement>(b: &Buffer<T, CudaBackend>) -> BackendResult<&CudaSlice<u8>> {
    b.raw()
        .raw()
        .ok_or_else(|| invalid("missing owned hex storage"))
}

/// Checked persistent CUDA hexahedral partial-assembly operator.
pub struct CudaHexPartialC64 {
    pool: MemoryPool<CudaBackend>,
    dimension: usize,
    basis_nodes: usize,
    q: usize,
    elements: usize,
    max_rhs: usize,
    nodes: Buffer<u32, CudaBackend>,
    constrained: Buffer<u32, CudaBackend>,
    basis: Buffer<f64, CudaBackend>,
    derivative: Buffer<f64, CudaBackend>,
    data: Buffer<Complex64, CudaBackend>,
    material_reference: Option<Buffer<Complex64, CudaBackend>>,
    offsets: Buffer<u32, CudaBackend>,
    slots: Buffer<u32, CudaBackend>,
    scratch: Mutex<Buffer<Complex64, CudaBackend>>,
    action: CudaFunction,
    scaled_action: CudaFunction,
    scatter: CudaFunction,
    material_kernel: CudaFunction,
    diffusion_scale: Complex64,
    mass_scale: Complex64,
    data_valid: bool,
}
impl std::fmt::Debug for CudaHexPartialC64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaHexPartialC64")
            .field("dimension", &self.dimension)
            .field("max_rhs", &self.max_rhs)
            .finish_non_exhaustive()
    }
}
impl CudaHexPartialC64 {
    /// Upload validated tensor data and allocate reusable element-output scratch.
    ///
    /// # Errors
    /// Rejects zero/excessive RHS capacity, graph capture, quota exhaustion, or
    /// JIT/backend failures. Partial construction refunds pooled arrays.
    pub fn try_new(
        pool: MemoryPool<CudaBackend>,
        host: &HexPartialC64,
        max_rhs: usize,
    ) -> BackendResult<Self> {
        let count = host
            .nodes
            .len()
            .checked_mul(max_rhs)
            .ok_or_else(|| invalid("hex scratch overflow"))?;
        if max_rhs == 0
            || count > u32::MAX as usize
            || host
                .element_count()
                .checked_mul(max_rhs)
                .is_none_or(|v| v > i32::MAX as usize)
            || host
                .dimension
                .checked_mul(max_rhs)
                .is_none_or(|v| v > u32::MAX as usize)
        {
            return Err(invalid("hex RHS capacity exceeds kernel limits"));
        }
        let backend = pool.backend();
        backend.check_pending_error()?;
        if backend.stream().capture_status().map_err(failed)?
            != cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
        {
            return Err(invalid("prepare hex operators before graph capture"));
        }
        let kernels = backend.hex_kernel_functions()?;
        Ok(Self {
            dimension: host.dimension,
            basis_nodes: host.n,
            q: host.q,
            elements: host.element_count(),
            max_rhs,
            nodes: upload(&pool, &host.nodes)?,
            constrained: upload(&pool, &host.constrained)?,
            basis: upload(&pool, &host.basis)?,
            derivative: upload(&pool, &host.derivative)?,
            data: upload(&pool, &host.data)?,
            material_reference: None,
            offsets: upload(&pool, &host.offsets)?,
            slots: upload(&pool, &host.slots)?,
            scratch: Mutex::new(pool.try_alloc(count)?),
            action: kernels.action.clone(),
            scaled_action: kernels.scaled_action.clone(),
            scatter: kernels.scatter.clone(),
            material_kernel: kernels.material.clone(),
            diffusion_scale: Complex64::new(1.0, 0.0),
            mass_scale: Complex64::new(1.0, 0.0),
            data_valid: true,
            pool,
        })
    }

    /// Retain current weighted coefficients as the reference for absolute material updates.
    ///
    /// Allocates and copies once; repeated calls preserve the first reference.
    /// Global diffusion/mass scales are not baked into it. Later raw coefficient
    /// replacements do not alter this reference. Uses seven complex values per
    /// quadrature point, charged to the operator's memory pool.
    ///
    /// # Errors
    /// Rejects incomplete data, graph capture, excessive sample counts and backend
    /// failures. Failed preparation retains the previous operator and can be retried.
    pub fn try_prepare_material_updates(&mut self) -> BackendResult<()> {
        let backend = self.pool.backend();
        backend.check_pending_error()?;
        if !self.data_valid || self.data.len() / 7 > u32::MAX as usize {
            return Err(invalid("invalid hex material reference"));
        }
        if backend.stream().capture_status().map_err(failed)?
            != cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
        {
            return Err(invalid("prepare hex material data outside graph capture"));
        }
        if self.material_reference.is_none() {
            let mut reference = self.pool.try_alloc(self.data.len())?;
            backend.try_copy_d2d(self.data.raw(), reference.raw_mut())?;
            self.material_reference = Some(reference);
        }
        Ok(())
    }

    /// Number of scalar samples per material field in tensor quadrature order.
    pub fn material_sample_count(&self) -> usize {
        self.data.len() / 7
    }

    /// Lower resident scalar material samples into weighted quadrature coefficients.
    ///
    /// Each input contains one complex multiplier per quadrature point, ordered
    /// by element and then x-fastest tensor quadrature. Mass multiplies the first
    /// reference entry; diffusion multiplies its six symmetric metric entries.
    /// Updates are absolute relative to `try_prepare_material_updates`, including
    /// transitions through zero. Global scales remain active. This supports scalar
    /// material variation, not a general physical anisotropic diffusion tensor.
    /// Inputs/products must be finite; there is no implicit numerical download.
    /// Uses one launch, with no allocation or copy. Preconditioners remain lagged.
    ///
    /// # Errors
    /// Rejects missing preparation, buffer/stream mismatches and graph capture
    /// before mutation. A failed launch blocks actions until a successful material
    /// or raw-data replacement, following backend context recovery requirements.
    pub fn try_update_material_samples(
        &mut self,
        diffusion: &CudaBuffer<Complex64>,
        mass: &CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        let backend = self.pool.backend();
        let reference = self
            .material_reference
            .as_ref()
            .ok_or_else(|| invalid("hex material updates were not prepared"))?;
        let count = self.material_sample_count();
        for buffer in [diffusion, mass] {
            let storage = buffer
                .raw()
                .ok_or_else(|| invalid("missing hex material samples"))?;
            if buffer.len() != count
                || storage.len() != count * size_of::<Complex64>()
                || !Arc::ptr_eq(storage.stream(), backend.stream())
            {
                return Err(invalid("hex material sample shape/storage/stream mismatch"));
            }
        }
        backend.check_pending_error()?;
        if backend.stream().capture_status().map_err(failed)?
            != cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
        {
            return Err(invalid("update hex material data outside graph capture"));
        }
        self.data_valid = false;
        backend.try_custom_kernel_preflight()?;
        let count = count as u32;
        backend.record_custom_kernel_launch();
        // SAFETY: prepared reference has seven entries per point and count fits
        // u32. Input lengths, raw byte counts and streams are checked above.
        // Private output has the same size as reference and cannot alias inputs.
        // Each thread writes seven disjoint entries using 64-bit output offsets.
        unsafe {
            backend
                .stream()
                .launch_builder(&self.material_kernel)
                .arg(raw(reference)?)
                .arg(
                    diffusion
                        .raw()
                        .ok_or_else(|| invalid("missing diffusion samples"))?,
                )
                .arg(mass.raw().ok_or_else(|| invalid("missing mass samples"))?)
                .arg(
                    self.data
                        .raw_mut()
                        .raw_mut()
                        .ok_or_else(|| invalid("missing hex data"))?,
                )
                .arg(&count)
                .launch(LaunchConfig {
                    grid_dim: (count.div_ceil(128), 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| backend.custom_kernel_error(OP, format!("{e:?}")))?;
        }
        self.data_valid = true;
        Ok(())
    }

    /// Replace weighted quadrature coefficients from a resident buffer.
    ///
    /// Uses one device copy without allocation or host staging. The layout matches
    /// `HexPartialC64::quadrature_data`; geometry, basis and constraints are fixed.
    /// Global scales remain active on the replacement data. Values must be finite
    /// and correctly transformed; no host download validates their numerical content.
    /// Refresh a smoother inverse or explicitly retain a lagged preconditioner.
    /// Existing action graphs reference the same storage and observe the new data.
    ///
    /// # Errors
    /// Rejects shape/storage/stream mismatches and graph capture before mutation.
    /// A failed copy blocks actions until a successful full replacement, subject
    /// to backend context recovery. Invalid input preserves the previous data.
    pub fn try_update_quadrature_data(
        &mut self,
        values: &CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        let backend = self.pool.backend();
        if values.len() != self.data.len() {
            return Err(invalid("hex quadrature data length mismatch"));
        }
        let source = values
            .raw()
            .ok_or_else(|| invalid("missing hex quadrature data"))?;
        if source.len() != raw(&self.data)?.len() || !Arc::ptr_eq(source.stream(), backend.stream())
        {
            return Err(invalid("hex quadrature data storage/stream mismatch"));
        }
        backend.check_pending_error()?;
        if backend.stream().capture_status().map_err(failed)?
            != cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
        {
            return Err(invalid("update hex data outside graph capture"));
        }
        self.data_valid = false;
        backend.try_copy_d2d(values, self.data.raw_mut())?;
        self.data_valid = true;
        Ok(())
    }

    /// Set global diffusion and mass multipliers without replacing quadrature data.
    ///
    /// Scales multiply the current quadrature coefficients, not the previous
    /// scales. All forward/transpose/Hermitian, residual, smoother and lifting
    /// actions use them. Constrained identity rows remain one. A supplied Jacobi
    /// inverse must match the updated operator (or explicitly be a lagged smoother).
    /// This host-side parameter update allocates/transfers nothing. Already
    /// captured CUDA graphs retain their captured scales and must be recaptured.
    ///
    /// # Errors
    /// Rejects non-finite scales or a pending backend error, preserving old scales.
    /// Products with geometry coefficients must remain representable.
    pub fn try_set_scales(&mut self, diffusion: Complex64, mass: Complex64) -> BackendResult<()> {
        if [diffusion, mass]
            .iter()
            .any(|v| !v.re.is_finite() || !v.im.is_finite())
        {
            return Err(invalid("hex coefficient scales must be finite"));
        }
        self.pool.backend().check_pending_error()?;
        self.diffusion_scale = diffusion;
        self.mass_scale = mass;
        Ok(())
    }
    /// Apply forward, transpose or Hermitian action to resident RHS columns.
    ///
    /// # Errors
    /// Rejects RHS capacity/shape, raw byte storage or stream mismatches before
    /// launch. Backend failures preserve typed recovery/output dispositions.
    pub fn try_apply_action(
        &self,
        backend: &CudaBackend,
        rhs: usize,
        x: &CudaBuffer<Complex64>,
        y: &mut CudaBuffer<Complex64>,
        action: Transpose,
    ) -> BackendResult<()> {
        self.launch(backend, rhs, x, y, action, Output::Action)
    }

    /// Lift resident nonzero prescribed values for column-major RHS vectors.
    ///
    /// Uses the retained element scratch and two launches without host transfers.
    /// Free entries of `prescribed` are ignored; constrained RHS entries are replaced.
    ///
    /// # Errors
    /// Rejects capacity, lengths, raw storage and stream mismatches before launch.
    pub fn try_lift_rhs_colmajor(
        &self,
        backend: &CudaBackend,
        rhs_count: usize,
        rhs: &CudaBuffer<Complex64>,
        prescribed: &CudaBuffer<Complex64>,
        output: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        self.launch(
            backend,
            rhs_count,
            prescribed,
            output,
            Transpose::NoTrans,
            Output::Lift(rhs),
        )
    }

    /// Compute resident column-major residuals in the two partial-assembly launches.
    ///
    /// Uses the constrained operator, including its identity rows.
    ///
    /// # Errors
    /// Rejects capacity, shapes, storage and stream mismatches before launch.
    pub fn try_residual_colmajor(
        &self,
        backend: &CudaBackend,
        rhs: usize,
        x: &CudaBuffer<Complex64>,
        b: &CudaBuffer<Complex64>,
        out: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        self.launch(
            backend,
            rhs,
            x,
            out,
            Transpose::NoTrans,
            Output::Residual(b),
        )
    }

    /// Apply a simultaneous weighted-Jacobi step within partial assembly.
    ///
    /// Computes y = x + omega D⁻¹(b - Ax). Supply a retained inverse diagonal
    /// of length n, shared across RHS columns. Choose damping for the spectrum;
    /// this smoother alone is not a general indefinite Helmholtz preconditioner.
    ///
    /// # Errors
    /// Rejects non-finite damping, capacity, shape, raw storage or stream
    /// mismatches before launch. Device diagonal values are not downloaded for
    /// validation. Output must be distinct from the immutable input buffers.
    pub fn try_jacobi_step(
        &self,
        backend: &CudaBackend,
        rhs: usize,
        x: &CudaBuffer<Complex64>,
        b: &CudaBuffer<Complex64>,
        inverse: &CudaBuffer<Complex64>,
        omega: f64,
        out: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        self.launch(
            backend,
            rhs,
            x,
            out,
            Transpose::NoTrans,
            Output::Jacobi {
                rhs: b,
                inverse,
                omega,
            },
        )
    }

    fn launch(
        &self,
        backend: &CudaBackend,
        rhs: usize,
        x: &CudaBuffer<Complex64>,
        y: &mut CudaBuffer<Complex64>,
        action: Transpose,
        output: Output<'_>,
    ) -> BackendResult<()> {
        if !self.data_valid {
            return Err(invalid("hex quadrature update is incomplete"));
        }
        let (mode, original_rhs, inverse, omega) = match output {
            Output::Action => (0_u32, None, None, 0.0),
            Output::Lift(b) => (1, Some(b), None, 0.0),
            Output::Residual(b) => (2, Some(b), None, 0.0),
            Output::Jacobi {
                rhs: b,
                inverse,
                omega,
            } => (3, Some(b), Some(inverse), omega),
        };
        if !omega.is_finite() || inverse.is_some_and(|b| b.len() != self.dimension) {
            return Err(invalid("hex inverse diagonal length or damping mismatch"));
        }
        if rhs > self.max_rhs || !Arc::ptr_eq(backend.stream(), self.pool.backend().stream()) {
            return Err(invalid("hex RHS capacity or backend stream mismatch"));
        }
        let total = self.dimension * rhs;
        if x.len() != total || y.len() != total || original_rhs.is_some_and(|b| b.len() != total) {
            return Err(invalid("hex vector length mismatch"));
        }
        backend.check_pending_error()?;
        if total == 0 {
            return Ok(());
        }
        let bytes = total
            .checked_mul(size_of::<Complex64>())
            .ok_or_else(|| invalid("hex byte size overflow"))?;
        for b in [x.raw(), y.raw(), original_rhs.unwrap_or(x).raw()] {
            let b = b.ok_or_else(|| invalid("missing hex vector storage"))?;
            if b.len() != bytes || !Arc::ptr_eq(b.stream(), backend.stream()) {
                return Err(invalid("hex vector storage/stream mismatch"));
            }
        }
        if let Some(inverse) = inverse {
            let raw = inverse
                .raw()
                .ok_or_else(|| invalid("missing inverse diagonal"))?;
            if raw.len() != self.dimension * size_of::<Complex64>()
                || !Arc::ptr_eq(raw.stream(), backend.stream())
            {
                return Err(invalid("hex inverse diagonal storage/stream mismatch"));
            }
        }
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| failed("hex scratch mutex poisoned"))?;
        backend.try_custom_kernel_preflight()?;
        let n = self.basis_nodes as u32;
        let q = self.q as u32;
        let dimension = self.dimension as u32;
        let elements = self.elements as u32;
        let adjoint = u32::from(matches!(action, Transpose::ConjTrans));
        let element_size = self.nodes.len() as u32;
        let total = total as u32;
        let lifting = u32::from(mode == 1);
        let scaled = self.diffusion_scale != Complex64::new(1.0, 0.0)
            || self.mass_scale != Complex64::new(1.0, 0.0);
        // SAFETY: constructor validates tensor sizes <=8 and all restriction and
        // quadrature lengths. Private arrays and scratch cannot be replaced by
        // callers. RHS capacity, raw vector byte counts and owning streams are
        // checked. One block owns each element/RHS output; the second kernel
        // sums validated incidence slots. Same-stream launches preserve ordering.
        unsafe {
            backend.record_custom_kernel_launch();
            let mut builder = backend.stream().launch_builder(if scaled {
                &self.scaled_action
            } else {
                &self.action
            });
            builder
                .arg(raw(&self.nodes)?)
                .arg(raw(&self.basis)?)
                .arg(raw(&self.derivative)?)
                .arg(raw(&self.data)?)
                .arg(x.raw().ok_or_else(|| invalid("missing input"))?)
                .arg(
                    scratch
                        .raw_mut()
                        .raw_mut()
                        .ok_or_else(|| invalid("missing scratch"))?,
                )
                .arg(&n)
                .arg(&q)
                .arg(&dimension)
                .arg(&elements)
                .arg(&adjoint)
                .arg(raw(&self.constrained)?)
                .arg(&lifting);
            if scaled {
                builder
                    .arg(&self.diffusion_scale.re)
                    .arg(&self.diffusion_scale.im)
                    .arg(&self.mass_scale.re)
                    .arg(&self.mass_scale.im);
            }
            builder
                .launch(LaunchConfig {
                    grid_dim: ((self.elements * rhs) as u32, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| backend.custom_kernel_error(OP, format!("{e:?}")))?;
            backend.record_custom_kernel_launch();
            backend
                .stream()
                .launch_builder(&self.scatter)
                .arg(raw(&self.offsets)?)
                .arg(raw(&self.slots)?)
                .arg(raw(&scratch)?)
                .arg(y.raw_mut().ok_or_else(|| invalid("missing output"))?)
                .arg(&dimension)
                .arg(&element_size)
                .arg(&total)
                .arg(raw(&self.constrained)?)
                .arg(x.raw().ok_or_else(|| invalid("missing input"))?)
                .arg(
                    original_rhs
                        .unwrap_or(x)
                        .raw()
                        .ok_or_else(|| invalid("missing RHS"))?,
                )
                .arg(&mode)
                .arg(
                    inverse
                        .unwrap_or(x)
                        .raw()
                        .ok_or_else(|| invalid("missing inverse diagonal"))?,
                )
                .arg(&omega)
                .launch(LaunchConfig::for_num_elems(total))
                .map_err(|e| backend.custom_kernel_error(OP, format!("{e:?}")))?;
        }
        Ok(())
    }
}
impl DeviceOperator<CudaBackend, Complex64> for CudaHexPartialC64 {
    fn n(&self) -> usize {
        self.dimension
    }
    fn apply(&self, b: &CudaBackend, x: &CudaBuffer<Complex64>, y: &mut CudaBuffer<Complex64>) {
        if let Err(e) = self.try_apply(b, x, y) {
            log::error!("hex action failed: {e}");
        }
    }
    fn try_apply(
        &self,
        b: &CudaBackend,
        x: &CudaBuffer<Complex64>,
        y: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        self.try_apply_action(b, 1, x, y, Transpose::NoTrans)
    }
    fn try_residual(
        &self,
        backend: &CudaBackend,
        x: &CudaBuffer<Complex64>,
        b: &CudaBuffer<Complex64>,
        out: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        self.try_residual_colmajor(backend, 1, x, b, out)
    }
    fn try_apply_colmajor(
        &self,
        b: &CudaBackend,
        rhs: usize,
        x: &CudaBuffer<Complex64>,
        y: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        self.try_apply_action(b, rhs, x, y, Transpose::NoTrans)
    }
}
impl DeviceAdjointOperator<CudaBackend, Complex64> for CudaHexPartialC64 {
    fn try_apply_hermitian(
        &self,
        b: &CudaBackend,
        x: &CudaBuffer<Complex64>,
        y: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        self.try_apply_action(b, 1, x, y, Transpose::ConjTrans)
    }
    fn try_apply_hermitian_colmajor(
        &self,
        b: &CudaBackend,
        rhs: usize,
        x: &CudaBuffer<Complex64>,
        y: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        self.try_apply_action(b, rhs, x, y, Transpose::ConjTrans)
    }
}

#[cfg(test)]
mod material_tests;

#[cfg(test)]
mod tests {
    #[test]
    fn hex_partial_kernel_compiles_with_nvrtc() {
        cudarc::nvrtc::compile_ptx(super::SOURCE).expect("hex partial assembly NVRTC compilation");
    }
    #[test]
    fn native_hex_partial_actions() {
        use super::*;
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                assert!(
                    std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                    "CUDA required: {e}"
                );
                eprintln!("SKIP native hex partial assembly: {e}");
                return;
            }
        };
        let pool = MemoryPool::new(backend, crate::MemoryPoolConfig::with_cap(1 << 24)).unwrap();
        let backend = pool.backend();
        assert!(backend.hex_kernels.lock().unwrap().is_none());
        let before = backend.begin_runtime_stats();
        let first = backend.hex_kernel_functions().unwrap();
        backend.try_prepare_hex_kernels().unwrap();
        let second = backend.hex_kernel_functions().unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        let stats = backend.runtime_stats_since(before);
        assert_eq!(stats.allocation_operations, 0);
        assert_eq!(stats.kernel_launches + stats.library_operations, 0);
        assert_eq!(stats.transfer.h2d_bytes + stats.transfer.d2h_bytes, 0);
        for (n, q) in [(2, 3), (3, 2), (4, 5), (8, 8)] {
            let host = crate::fem_hex::tests::fixture(n, q);
            for dofs in [
                vec![],
                vec![0, host.dimension() / 2],
                (0..host.dimension()).collect(),
            ] {
                let base = host.clone().try_with_dirichlet(&dofs).unwrap();
                let mut op = CudaHexPartialC64::try_new(pool.clone(), &base, 3).unwrap();
                let scales = [
                    (Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)),
                    (Complex64::new(0.7, 0.3), Complex64::new(-1.2, 0.4)),
                    (Complex64::new(1.0, 0.0), Complex64::default()),
                    (Complex64::default(), Complex64::new(0.8, -0.2)),
                ];
                for (update, &(diffusion, mass)) in scales
                    .iter()
                    .take(if n == 8 { 1 } else { scales.len() })
                    .enumerate()
                {
                    let mut updated = base.clone();
                    for (i, value) in updated.data.iter_mut().enumerate() {
                        *value *= Complex64::new(
                            1.0 + update as f64 * (i % 11) as f64 * 0.03,
                            update as f64 * (i % 5) as f64 * 0.01,
                        );
                    }
                    let values = upload(&pool, &updated.data).unwrap();
                    if n == 2 && dofs.is_empty() && update == 1 {
                        backend.inject_failure(crate::CudaFailureInjectionPoint::TransferD2D);
                        assert!(op.try_update_quadrature_data(values.raw()).is_err());
                        backend.recover_after_error().unwrap();
                        let x = upload(&pool, &vec![Complex64::new(1.0, 0.0); base.dimension()])
                            .unwrap();
                        let sentinel = vec![Complex64::new(3.0, -2.0); base.dimension()];
                        let mut y = upload(&pool, &sentinel).unwrap();
                        let before = backend.runtime_stats();
                        assert!(op.try_apply(backend, x.raw(), y.raw_mut()).is_err());
                        assert_eq!(
                            backend.runtime_stats().kernel_launches,
                            before.kernel_launches
                        );
                        let mut actual = vec![Complex64::default(); base.dimension()];
                        backend.try_copy_d2h(y.raw(), &mut actual).unwrap();
                        assert_eq!(actual, sentinel);
                    }
                    let before = backend.runtime_stats();
                    op.try_update_quadrature_data(values.raw()).unwrap();
                    op.try_set_scales(diffusion, mass).unwrap();
                    let after = backend.runtime_stats();
                    assert_eq!(after.kernel_launches, before.kernel_launches);
                    assert_eq!(after.allocation_operations, before.allocation_operations);
                    assert_eq!(after.transfer.h2d_bytes, before.transfer.h2d_bytes);
                    assert_eq!(after.transfer.d2h_bytes, before.transfer.d2h_bytes);
                    assert_eq!(
                        after.transfer.d2d_bytes - before.transfer.d2d_bytes,
                        updated.data.len() * size_of::<Complex64>()
                    );
                    let wrong = pool.try_alloc::<Complex64>(1).unwrap();
                    assert!(op.try_update_quadrature_data(wrong.raw()).is_err());
                    assert!(
                        op.try_set_scales(Complex64::new(f64::NAN, 0.0), mass)
                            .is_err()
                    );
                    assert!(
                        op.try_set_scales(diffusion, Complex64::new(0.0, f64::INFINITY))
                            .is_err()
                    );
                    let mut host = updated;
                    for data in host.data.chunks_exact_mut(7) {
                        data[0] *= mass;
                        for value in &mut data[1..] {
                            *value *= diffusion;
                        }
                    }
                    for rhs in [1, 3] {
                        let input: Vec<_> = (0..rhs * host.dimension())
                            .map(|i| Complex64::new((i as f64 * 0.3).sin(), (i as f64 * 0.2).cos()))
                            .collect();
                        let mut x = pool.try_alloc(input.len()).unwrap();
                        let mut y = pool.try_alloc(input.len()).unwrap();
                        backend.try_copy_h2d(&input, x.raw_mut()).unwrap();
                        for action in [Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans] {
                            op.try_apply_action(backend, rhs, x.raw(), y.raw_mut(), action)
                                .unwrap();
                            backend.try_synchronize().unwrap();
                            let before = backend.runtime_stats();
                            op.try_apply_action(backend, rhs, x.raw(), y.raw_mut(), action)
                                .unwrap();
                            let after = backend.runtime_stats();
                            assert_eq!(after.kernel_launches, before.kernel_launches + 2);
                            assert_eq!(after.allocation_operations, before.allocation_operations);
                            assert_eq!(after.transfer.h2d_bytes, before.transfer.h2d_bytes);
                            assert_eq!(after.transfer.d2h_bytes, before.transfer.d2h_bytes);
                            let mut actual = vec![Complex64::default(); input.len()];
                            backend.try_copy_d2h(y.raw(), &mut actual).unwrap();
                            for col in 0..rhs {
                                let size = host.dimension();
                                let mut expected = vec![Complex64::default(); size];
                                host.apply(
                                    &input[col * size..(col + 1) * size],
                                    &mut expected,
                                    action,
                                )
                                .unwrap();
                                for (a, b) in
                                    actual[col * size..(col + 1) * size].iter().zip(expected)
                                {
                                    assert!(
                                        (*a - b).norm() < 1e-9 * (1.0 + b.norm()),
                                        "n={n} q={q}: {a} != {b}"
                                    );
                                }
                            }
                            assert!(
                                op.try_apply_action(backend, 4, x.raw(), y.raw_mut(), action)
                                    .is_err()
                            );
                            let mut unchanged = vec![Complex64::default(); input.len()];
                            backend.try_copy_d2h(y.raw(), &mut unchanged).unwrap();
                            assert_eq!(actual, unchanged);
                        }
                        let size = host.dimension();
                        let rhs_host: Vec<_> = input
                            .iter()
                            .map(|v| v.conj() + Complex64::new(0.7, -0.2))
                            .collect();
                        let inverse_host: Vec<_> = host
                            .diagonal()
                            .into_iter()
                            .map(|v| Complex64::new(1.0, 0.0) / v)
                            .collect();
                        let mut b = pool.try_alloc(input.len()).unwrap();
                        let mut inverse = pool.try_alloc(size).unwrap();
                        backend.try_copy_h2d(&rhs_host, b.raw_mut()).unwrap();
                        backend
                            .try_copy_h2d(&inverse_host, inverse.raw_mut())
                            .unwrap();
                        let mut residual_reference = vec![Complex64::default(); input.len()];
                        for col in 0..rhs {
                            let range = col * size..(col + 1) * size;
                            host.apply(
                                &input[range.clone()],
                                &mut residual_reference[range],
                                Transpose::NoTrans,
                            )
                            .unwrap();
                        }
                        for (r, b) in residual_reference.iter_mut().zip(&rhs_host) {
                            *r = *b - *r;
                        }
                        for omega in [None, Some(0.0), Some(0.6)] {
                            let before = backend.runtime_stats();
                            match omega {
                                None if rhs == 1 => DeviceOperator::try_residual(
                                    &op,
                                    backend,
                                    x.raw(),
                                    b.raw(),
                                    y.raw_mut(),
                                )
                                .unwrap(),
                                None => op
                                    .try_residual_colmajor(
                                        backend,
                                        rhs,
                                        x.raw(),
                                        b.raw(),
                                        y.raw_mut(),
                                    )
                                    .unwrap(),
                                Some(w) => op
                                    .try_jacobi_step(
                                        backend,
                                        rhs,
                                        x.raw(),
                                        b.raw(),
                                        inverse.raw(),
                                        w,
                                        y.raw_mut(),
                                    )
                                    .unwrap(),
                            }
                            let after = backend.runtime_stats();
                            assert_eq!(after.kernel_launches, before.kernel_launches + 2);
                            assert_eq!(after.allocation_operations, before.allocation_operations);
                            assert_eq!(after.transfer.h2d_bytes, before.transfer.h2d_bytes);
                            assert_eq!(after.transfer.d2h_bytes, before.transfer.d2h_bytes);
                            let mut actual = vec![Complex64::default(); input.len()];
                            backend.try_copy_d2h(y.raw(), &mut actual).unwrap();
                            for (i, &v) in actual.iter().enumerate() {
                                let expected = omega.map_or(residual_reference[i], |w| {
                                    input[i] + w * inverse_host[i % size] * residual_reference[i]
                                });
                                assert!((v - expected).norm() < 2e-9 * (1.0 + expected.norm()));
                            }
                        }
                        let mut before = vec![Complex64::default(); input.len()];
                        backend.try_copy_d2h(y.raw(), &mut before).unwrap();
                        assert!(
                            op.try_jacobi_step(
                                backend,
                                rhs,
                                x.raw(),
                                b.raw(),
                                inverse.raw(),
                                f64::NAN,
                                y.raw_mut()
                            )
                            .is_err()
                        );
                        let short = pool.try_alloc::<Complex64>(size - 1).unwrap();
                        assert!(
                            op.try_jacobi_step(
                                backend,
                                rhs,
                                x.raw(),
                                b.raw(),
                                short.raw(),
                                0.6,
                                y.raw_mut()
                            )
                            .is_err()
                        );
                        let mut after = before.clone();
                        backend.try_copy_d2h(y.raw(), &mut after).unwrap();
                        assert_eq!(before, after);
                        // Zero damping must not propagate poisoned RHS/diagonal data.
                        backend
                            .try_copy_h2d(
                                &vec![Complex64::new(f64::NAN, 0.0); input.len()],
                                b.raw_mut(),
                            )
                            .unwrap();
                        backend
                            .try_copy_h2d(
                                &vec![Complex64::new(f64::NAN, 0.0); size],
                                inverse.raw_mut(),
                            )
                            .unwrap();
                        op.try_jacobi_step(
                            backend,
                            rhs,
                            x.raw(),
                            b.raw(),
                            inverse.raw(),
                            0.0,
                            y.raw_mut(),
                        )
                        .unwrap();
                        backend.try_copy_d2h(y.raw(), &mut after).unwrap();
                        assert_eq!(input, after);
                        let prescribed: Vec<_> = input
                            .iter()
                            .enumerate()
                            .map(|(i, &v)| {
                                if dofs.contains(&(i % size)) {
                                    v
                                } else {
                                    Complex64::new(f64::NAN, f64::NAN)
                                }
                            })
                            .collect();
                        let mut g = pool.try_alloc(input.len()).unwrap();
                        backend.try_copy_h2d(&prescribed, g.raw_mut()).unwrap();
                        let before = backend.runtime_stats();
                        op.try_lift_rhs_colmajor(backend, rhs, x.raw(), g.raw(), y.raw_mut())
                            .unwrap();
                        let after = backend.runtime_stats();
                        assert_eq!(after.kernel_launches, before.kernel_launches + 2);
                        assert_eq!(after.allocation_operations, before.allocation_operations);
                        assert_eq!(after.transfer.h2d_bytes, before.transfer.h2d_bytes);
                        assert_eq!(after.transfer.d2h_bytes, before.transfer.d2h_bytes);
                        let mut actual = vec![Complex64::default(); input.len()];
                        backend.try_copy_d2h(y.raw(), &mut actual).unwrap();
                        for col in 0..rhs {
                            let range = col * size..(col + 1) * size;
                            let mut expected = vec![Complex64::default(); size];
                            host.lift_rhs(
                                &input[range.clone()],
                                &prescribed[range.clone()],
                                &mut expected,
                            )
                            .unwrap();
                            for (a, b) in actual[range].iter().zip(expected) {
                                assert!((*a - b).norm() < 1e-9 * (1.0 + b.norm()));
                            }
                        }
                    }
                }
            }
        }
    }
}
