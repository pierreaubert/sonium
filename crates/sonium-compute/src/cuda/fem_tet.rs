//! CUDA affine P1/P2 tetrahedral acoustic volume actions without dense element tensors.
//!
//! Geometry, connectivity and coefficients upload once. One output-node thread
//! evaluates incident element actions and accumulates them in stable element
//! order. Forward, transpose, Hermitian and column-major block actions need one
//! launch, no temporary vectors, and no host vector transfers. Module/JIT memory
//! is driver-owned; all device arrays here are charged to the supplied pool.

// Rust guideline compliant 2026-02-21
use crate::cuda::{CudaBackend, CudaBuffer};
use crate::fem_tet::TetAcousticC64;
use crate::pool::Buffer;
use crate::{
    Backend, BackendError, BackendResult, BufferElement, DeviceAdjointOperator, DeviceBufferLike,
    DeviceOperator, MemoryPool, Transpose,
};
use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use num_complex::Complex64;
use std::sync::Arc;

mod assembly;
#[doc(inline)]
pub use assembly::CudaTetCsrAssembly;

const SOURCE: &str = include_str!("kernels/fem_tet_p1.cu");
const OPERATION: &str = "cuda_tet_acoustic";

pub(super) struct CudaTetKernels {
    action: CudaFunction,
    assembly: CudaFunction,
    action_block_size: u32,
    assembly_block_size: u32,
}

impl CudaBackend {
    /// Prepare and retain P1 or P2 tetrahedral action and assembly kernels.
    ///
    /// `N` is four for P1 or ten for P2. Each specialization is cached separately
    /// on this backend, including its occupancy-derived launch sizes. No element
    /// arrays are allocated or uploaded. Driver module memory is not pooled.
    ///
    /// # Errors
    /// Rejects unsupported order, capture and pending failures; propagates JIT,
    /// symbol-loading and occupancy-query errors before publishing the cache.
    pub fn try_prepare_tet_kernels<const N: usize>(&self) -> BackendResult<()> {
        self.tet_kernel_functions::<N>().map(|_| ())
    }

    fn tet_kernel_functions<const N: usize>(&self) -> BackendResult<Arc<CudaTetKernels>> {
        let slot = match N {
            4 => 0,
            10 => 1,
            _ => return Err(invalid("tet kernels require four or ten nodes per element")),
        };
        self.check_pending_error()?;
        if self.stream().capture_status().map_err(failed)?
            != cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
        {
            return Err(invalid("prepare tetrahedral kernels before graph capture"));
        }
        let mut cache = self.tet_kernels.lock().map_err(failed)?;
        if let Some(kernels) = &cache[slot] {
            return Ok(Arc::clone(kernels));
        }
        let ptx = crate::backend::catch_backend_panic(OPERATION, || {
            compile_ptx_with_opts(
                format!("#define TET_NODES {N}\n{SOURCE}"),
                CompileOptions::default(),
            )
        })?.map_err(failed)?;
        let module = self.context().load_module(ptx).map_err(failed)?;
        let action = module.load_function("sonium_tet_c64").map_err(failed)?;
        let assembly = module.load_function("sonium_tet_csr_c64").map_err(failed)?;
        let kernels = Arc::new(CudaTetKernels {
            action_block_size: kernel_block_size(&action)?,
            assembly_block_size: kernel_block_size(&assembly)?,
            action,
            assembly,
        });
        cache[slot] = Some(Arc::clone(&kernels));
        Ok(kernels)
    }
}
fn invalid(reason: impl Into<String>) -> BackendError {
    BackendError::InvalidArgument {
        operation: OPERATION,
        reason: reason.into(),
    }
}
fn failed(error: impl std::fmt::Debug) -> BackendError {
    BackendError::OperationFailed {
        operation: OPERATION,
        reason: format!("{error:?}"),
    }
}

// Neither tetrahedral kernel uses dynamic shared memory. Query the loaded
// specialization once: the generic 1,024-thread launch exceeds P2 register limits.
extern "C" fn no_dynamic_shared_memory(_: std::ffi::c_int) -> usize {
    0
}

fn kernel_block_size(kernel: &CudaFunction) -> BackendResult<u32> {
    let (_, size) = kernel
        .occupancy_max_potential_block_size(no_dynamic_shared_memory, 0, 0, None)
        .map_err(failed)?;
    if size == 0 {
        return Err(invalid("CUDA returned a zero kernel block size"));
    }
    Ok(size)
}

fn launch_config(count: u32, block_size: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (count.div_ceil(block_size), 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// CUDA affine P1 tetrahedral acoustic operator.
pub type CudaTetP1C64 = CudaTetC64<4>;
/// CUDA affine P2 tetrahedral acoustic operator.
pub type CudaTetP2C64 = CudaTetC64<10>;

/// Device-resident affine P1/P2 tetrahedral complex acoustic operator.
///
/// The supplied pool owns the execution backend. External buffers must belong
/// to that same stream; construction during graph capture is rejected. Boundary
/// facets are separate; an optional host-prepared Dirichlet mask is retained.
pub struct CudaTetC64<const N: usize> {
    pool: MemoryPool<CudaBackend>,
    n: usize,
    nodes: Buffer<u32, CudaBackend>,
    geometry: Buffer<f64, CudaBackend>,
    offsets: Buffer<u32, CudaBackend>,
    slots: Buffer<u32, CudaBackend>,
    coefficients: Buffer<Complex64, CudaBackend>,
    dirichlet: Buffer<u32, CudaBackend>,
    kernel: CudaFunction,
    assembly_kernel: CudaFunction,
    action_block_size: u32,
    assembly_block_size: u32,
    diffusion_scale: Complex64,
    mass_scale: Complex64,
}
impl<const N: usize> std::fmt::Debug for CudaTetC64<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaTetC64")
            .field("dimension", &self.n)
            .field("elements", &(self.nodes.len() / N))
            .finish_non_exhaustive()
    }
}
fn upload<T: BufferElement>(
    pool: &MemoryPool<CudaBackend>,
    values: &[T],
) -> BackendResult<Buffer<T, CudaBackend>> {
    let mut buffer = pool.try_alloc(values.len())?;
    pool.backend().try_copy_h2d(values, buffer.raw_mut())?;
    Ok(buffer)
}
// P2 stiffness needs only symmetric gradient dot products. Prepare their ten
// entries once rather than repeating three multiplies per entry in every action.
// P1 retains its gradients because its fused action evaluates a vector gradient.
fn upload_geometry<const N: usize>(
    pool: &MemoryPool<CudaBackend>,
    host: &TetAcousticC64<N>,
) -> BackendResult<Buffer<f64, CudaBackend>> {
    if N != 10 {
        return upload(pool, &host.geometry);
    }
    let mut packed = Vec::with_capacity(host.geometry.len() / 13 * 11);
    for g in host.geometry.as_chunks::<13>().0 {
        packed.push(g[0]);
        for i in 0..4 {
            for j in i..4 {
                let dot = g[1 + 3 * i] * g[1 + 3 * j]
                    + g[2 + 3 * i] * g[2 + 3 * j]
                    + g[3 + 3 * i] * g[3 + 3 * j];
                if !dot.is_finite() {
                    return Err(invalid("P2 geometry Gram matrix is non-finite"));
                }
                packed.push(dot);
            }
        }
    }
    upload(pool, &packed)
}

fn raw<T: BufferElement>(buffer: &Buffer<T, CudaBackend>) -> BackendResult<&CudaSlice<u8>> {
    buffer
        .raw()
        .raw()
        .ok_or_else(|| invalid("unexpected empty owned buffer"))
}
enum Output<'a> {
    Action,
    Residual(&'a CudaBuffer<Complex64>),
    Lift(&'a CudaBuffer<Complex64>),
    Jacobi {
        b: &'a CudaBuffer<Complex64>,
        inverse_diagonal: &'a CudaBuffer<Complex64>,
        omega: f64,
    },
}

impl<const N: usize> CudaTetC64<N> {
    /// Upload validated host geometry and load the fused volume-action kernel.
    ///
    /// # Errors
    /// Returns typed quota, upload, NVRTC/module or backend failures. Construction
    /// during capture fails before allocating. No CPU fallback is used.
    pub fn try_new(pool: MemoryPool<CudaBackend>, host: &TetAcousticC64<N>) -> BackendResult<Self> {
        let backend = pool.backend();
        backend.check_pending_error()?;
        if backend.stream().capture_status().map_err(failed)?
            != cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
        {
            return Err(invalid(
                "construct tetrahedral operators before graph capture",
            ));
        }
        let kernels = backend.tet_kernel_functions::<N>()?;
        Ok(Self {
            n: host.n,
            nodes: upload(&pool, &host.nodes)?,
            geometry: upload_geometry(&pool, host)?,
            offsets: upload(&pool, &host.offsets)?,
            slots: upload(&pool, &host.slots)?,
            coefficients: upload(&pool, &host.coefficients)?,
            dirichlet: upload(&pool, &host.dirichlet)?,
            kernel: kernels.action.clone(),
            assembly_kernel: kernels.assembly.clone(),
            action_block_size: kernels.action_block_size,
            assembly_block_size: kernels.assembly_block_size,
            diffusion_scale: Complex64::new(1.0, 0.0),
            mass_scale: Complex64::new(1.0, 0.0),
            pool,
        })
    }

    /// Set global coefficient scales without transferring element arrays.
    ///
    /// Scales multiply the constructor's diffusion and mass coefficients.
    /// They apply to actions, lifting, residuals and smoothing. Dirichlet
    /// identity rows remain one; Hermitian actions conjugate the full product.
    ///
    /// # Errors
    /// Rejects non-finite scales before changing either coefficient.
    pub fn try_set_scales(&mut self, diffusion: Complex64, mass: Complex64) -> BackendResult<()> {
        if [diffusion, mass]
            .iter()
            .any(|v| !v.re.is_finite() || !v.im.is_finite())
        {
            return Err(invalid("coefficient scales must be finite"));
        }
        self.diffusion_scale = diffusion;
        self.mass_scale = mass;
        Ok(())
    }

    /// Apply forward, algebraic transpose, or Hermitian action to resident RHS columns.
    ///
    /// # Errors
    /// Rejects shape, stream, byte-storage and kernel-index mismatches before
    /// launch. Backend errors after launch may leave output unusable; follow the
    /// typed context-recovery disposition before submitting further work.
    pub fn try_apply_action(
        &self,
        backend: &CudaBackend,
        rhs: usize,
        x: &CudaBuffer<Complex64>,
        y: &mut CudaBuffer<Complex64>,
        action: Transpose,
    ) -> BackendResult<()> {
        self.try_fused(backend, rhs, x, y, action, Output::Action)
    }

    /// Lift nonzero Dirichlet data into resident RHS columns in one launch.
    ///
    /// `prescribed` is a full column-major nodal array; free-node values are
    /// ignored. Computes `P*(b - A*Q*prescribed) + Q*prescribed`, using the
    /// original volume action and the constructor's fixed constraint mask.
    /// Geometry and the mask remain resident when prescribed values change.
    ///
    /// # Errors
    /// Rejects inconsistent lengths, streams or raw storage before launch;
    /// propagates checked backend errors without CPU fallback.
    pub fn try_lift_rhs_colmajor(
        &self,
        backend: &CudaBackend,
        rhs: usize,
        b: &CudaBuffer<Complex64>,
        prescribed: &CudaBuffer<Complex64>,
        out: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        self.try_fused(
            backend,
            rhs,
            prescribed,
            out,
            Transpose::NoTrans,
            Output::Lift(b),
        )
    }

    /// Compute `b - A*x` for resident column-major RHS blocks in one launch.
    ///
    /// # Errors
    /// Rejects mismatched lengths, streams or raw storage before launch. Backend
    /// failures follow the same output/context contract as `try_apply_action`.
    pub fn try_residual_colmajor(
        &self,
        backend: &CudaBackend,
        rhs: usize,
        x: &CudaBuffer<Complex64>,
        b: &CudaBuffer<Complex64>,
        out: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        self.try_fused(
            backend,
            rhs,
            x,
            out,
            Transpose::NoTrans,
            Output::Residual(b),
        )
    }

    /// Compute a simultaneous weighted-Jacobi step in one volume-action launch.
    ///
    /// Computes `y = x + omega * inverse_diagonal * (b - A*x)`. The inverse
    /// diagonal has one entry per node and is shared across RHS columns. Inputs
    /// remain immutable; output must be separate from the previous iterate.
    /// Callers supply the diagonal for this volume operator and choose damping
    /// suitable for their spectrum. This is a smoother, not a Helmholtz solver.
    ///
    /// # Errors
    /// Rejects non-finite damping, mismatched lengths/streams/raw storage, or
    /// backend failures. Device diagonal values are caller-owned numeric data;
    /// they are not downloaded for validation.
    pub fn try_jacobi_step(
        &self,
        backend: &CudaBackend,
        rhs: usize,
        x: &CudaBuffer<Complex64>,
        b: &CudaBuffer<Complex64>,
        inverse_diagonal: &CudaBuffer<Complex64>,
        omega: f64,
        y: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        self.try_fused(
            backend,
            rhs,
            x,
            y,
            Transpose::NoTrans,
            Output::Jacobi {
                b,
                inverse_diagonal,
                omega,
            },
        )
    }

    fn try_fused(
        &self,
        backend: &CudaBackend,
        rhs: usize,
        x: &CudaBuffer<Complex64>,
        y: &mut CudaBuffer<Complex64>,
        action: Transpose,
        output_mode: Output<'_>,
    ) -> BackendResult<()> {
        let total = self
            .n
            .checked_mul(rhs)
            .ok_or_else(|| invalid("RHS size overflow"))?;
        let total_u32 =
            u32::try_from(total).map_err(|_| invalid("RHS size exceeds kernel index range"))?;
        let bytes = total
            .checked_mul(size_of::<Complex64>())
            .ok_or_else(|| invalid("RHS byte count overflow"))?;
        if !Arc::ptr_eq(backend.stream(), self.pool.backend().stream())
            || x.len() != total
            || y.len() != total
        {
            return Err(invalid(
                "backend stream or input/output dimensions do not match operator",
            ));
        }
        let (b, inverse_diagonal, mode, omega) = match output_mode {
            Output::Action => (x, x, 0_u32, 0.0),
            Output::Residual(b) => (b, x, 1, 0.0),
            Output::Lift(b) => (b, x, 3, 0.0),
            Output::Jacobi {
                b,
                inverse_diagonal,
                omega,
            } => (b, inverse_diagonal, 2, omega),
        };
        if b.len() != total || !omega.is_finite() || (mode == 2 && inverse_diagonal.len() != self.n)
        {
            return Err(invalid("residual/smoother shape or damping mismatch"));
        }
        // Validate the shared diagonal even when there are zero RHS columns.
        if mode == 2 {
            let diagonal = inverse_diagonal
                .raw()
                .ok_or_else(|| invalid("missing diagonal storage"))?;
            let diagonal_bytes = self
                .n
                .checked_mul(size_of::<Complex64>())
                .ok_or_else(|| invalid("diagonal byte count overflow"))?;
            if diagonal.len() != diagonal_bytes || !Arc::ptr_eq(diagonal.stream(), backend.stream())
            {
                return Err(invalid("diagonal storage size or stream mismatch"));
            }
        }
        backend.check_pending_error()?;
        if total == 0 {
            return Ok(());
        }
        // CudaBuffer exposes mutable raw storage; validate its actual length
        // as well as its nominal scalar count before using pointers in CUDA.
        for buffer in [x.raw(), y.raw(), b.raw()] {
            let buffer = buffer.ok_or_else(|| invalid("missing nonempty device storage"))?;
            if buffer.len() != bytes || !Arc::ptr_eq(buffer.stream(), backend.stream()) {
                return Err(invalid("vector storage size or stream mismatch"));
            }
        }
        backend.try_custom_kernel_preflight()?;
        let input = x.raw().ok_or_else(|| invalid("missing input storage"))?;
        let output = y
            .raw_mut()
            .ok_or_else(|| invalid("missing output storage"))?;
        let rhs_input = b.raw().ok_or_else(|| invalid("missing RHS storage"))?;
        let diagonal_input = inverse_diagonal
            .raw()
            .ok_or_else(|| invalid("missing diagonal storage"))?;
        let n = self.n as u32;
        let adjoint = u32::from(matches!(action, Transpose::ConjTrans));
        backend.record_custom_kernel_launch();
        // SAFETY: arguments exactly match sonium_tet_c64's pointer/scalar
        // layout. Private geometry/indices originate from the checked host
        // constructor; the private constraint mask has exactly n entries.
        // These arrays cannot be modified through this API. Each thread
        // owns one output entry; RHS bounds, raw byte lengths and stream
        // identity are checked above, including RHS and node-sized diagonal.
        // Unused RHS/diagonal arguments alias the read-only input in action mode.
        // All buffers and the function stay live
        // during enqueue; cudarc tracks their uses on the owning stream.
        unsafe {
            backend
                .stream()
                .launch_builder(&self.kernel)
                .arg(raw(&self.nodes)?)
                .arg(raw(&self.geometry)?)
                .arg(raw(&self.offsets)?)
                .arg(raw(&self.slots)?)
                .arg(raw(&self.coefficients)?)
                .arg(input)
                .arg(output)
                .arg(&n)
                .arg(&total_u32)
                .arg(&adjoint)
                .arg(rhs_input)
                .arg(diagonal_input)
                .arg(&mode)
                .arg(&omega)
                .arg(raw(&self.dirichlet)?)
                .arg(&self.diffusion_scale.re)
                .arg(&self.diffusion_scale.im)
                .arg(&self.mass_scale.re)
                .arg(&self.mass_scale.im)
                .launch(launch_config(total_u32, self.action_block_size))
                .map_err(|error| backend.custom_kernel_error(OPERATION, format!("{error:?}")))?;
        }
        Ok(())
    }
}
impl<const N: usize> DeviceOperator<CudaBackend, Complex64> for CudaTetC64<N> {
    fn n(&self) -> usize {
        self.n
    }
    fn apply(
        &self,
        backend: &CudaBackend,
        x: &CudaBuffer<Complex64>,
        y: &mut CudaBuffer<Complex64>,
    ) {
        if let Err(error) = self.try_apply(backend, x, y) {
            log::error!("CUDA tetrahedral action failed: {error}");
        }
    }
    fn try_apply(
        &self,
        backend: &CudaBackend,
        x: &CudaBuffer<Complex64>,
        y: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        self.try_apply_action(backend, 1, x, y, Transpose::NoTrans)
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
        backend: &CudaBackend,
        rhs: usize,
        x: &CudaBuffer<Complex64>,
        y: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        self.try_apply_action(backend, rhs, x, y, Transpose::NoTrans)
    }
}
impl<const N: usize> DeviceAdjointOperator<CudaBackend, Complex64> for CudaTetC64<N> {
    fn try_apply_hermitian(
        &self,
        backend: &CudaBackend,
        x: &CudaBuffer<Complex64>,
        y: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        self.try_apply_action(backend, 1, x, y, Transpose::ConjTrans)
    }
    fn try_apply_hermitian_colmajor(
        &self,
        backend: &CudaBackend,
        rhs: usize,
        x: &CudaBuffer<Complex64>,
        y: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        self.try_apply_action(backend, rhs, x, y, Transpose::ConjTrans)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fem_tet::TetP1AcousticC64;
    #[test]
    fn tetrahedral_kernels_compile_without_device() {
        for nodes in [4, 10] {
            match std::panic::catch_unwind(|| {
                compile_ptx_with_opts(
                    format!("#define TET_NODES {nodes}\n{SOURCE}"),
                    CompileOptions::default(),
                )
            }) {
                Ok(result) => {
                    result.expect("tetrahedral NVRTC compilation");
                }
                Err(error) => {
                    assert!(
                        std::env::var_os("SONIUM_REQUIRE_NVRTC").is_none(),
                        "NVRTC required: {error:?}"
                    );
                    eprintln!("SKIP tetrahedral compiler test: NVRTC loader unavailable");
                }
            }
        }
    }
    #[test]
    fn native_tet_p1_actions_blocks_and_warm_residency() {
        let host = TetP1AcousticC64::try_new(
            &[
                [0., 0., 0.],
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.],
                [1., 1., 1.],
            ],
            &[[0, 1, 2, 3], [1, 2, 3, 4]],
            &[Complex64::new(1.2, 0.4), Complex64::new(0.7, -0.2)],
            &[Complex64::new(-2., 0.3), Complex64::new(-1., 0.8)],
        )
        .unwrap();
        native_tet_actions(host);
    }

    #[test]
    fn native_tet_p2_actions_blocks_and_warm_residency() {
        let host = crate::fem_tet::TetP2AcousticC64::try_new(
            &[
                [0., 0., 0.],
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.],
                [0.5, 0., 0.],
                [0., 0.5, 0.],
                [0., 0., 0.5],
                [0.5, 0.5, 0.],
                [0.5, 0., 0.5],
                [0., 0.5, 0.5],
            ],
            &[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
            &[Complex64::new(1.2, 0.4)],
            &[Complex64::new(-2., 0.3)],
        )
        .unwrap();
        native_tet_actions(host);
    }

    fn native_tet_actions<const N: usize>(host: TetAcousticC64<N>) {
        let backend = match CudaBackend::new(0) {
            Ok(backend) => backend,
            Err(error) => {
                assert!(
                    std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                    "CUDA required: {error}"
                );
                eprintln!("SKIP native tetrahedral operator ({N} DOFs/element): {error}");
                return;
            }
        };
        let pool = MemoryPool::new(backend, crate::MemoryPoolConfig::with_cap(1 << 24)).unwrap();
        let backend = pool.backend();
        assert!(backend.tet_kernels.lock().unwrap().iter().all(Option::is_none));
        assert!(backend.try_prepare_tet_kernels::<5>().is_err());
        let before = backend.begin_runtime_stats();
        let first = backend.tet_kernel_functions::<N>().unwrap();
        backend.try_prepare_tet_kernels::<N>().unwrap();
        let second = backend.tet_kernel_functions::<N>().unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        let other = if N == 4 {
            backend.tet_kernel_functions::<10>().unwrap()
        } else {
            backend.tet_kernel_functions::<4>().unwrap()
        };
        assert!(!Arc::ptr_eq(&first, &other));
        assert!(first.action_block_size > 0 && first.assembly_block_size > 0);
        let stats = backend.runtime_stats_since(before);
        assert_eq!(stats.allocation_operations, 0);
        assert_eq!(stats.kernel_launches + stats.library_operations, 0);
        assert_eq!(stats.transfer.h2d_bytes + stats.transfer.d2h_bytes, 0);
        let n = host.dimension();
        for dofs in [vec![], vec![0, n - 1], (0..n).collect()] {
            let host = host.clone().try_with_dirichlet(&dofs).unwrap();
            let operator = CudaTetC64::try_new(pool.clone(), &host).unwrap();
            {
                use crate::{DeviceCsr, PersistentDeviceCsr};
                let assembly = CudaTetCsrAssembly::try_new(pool.clone(), &host).unwrap();
                let pattern = assembly.pattern();
                let mut values = pool.try_alloc::<Complex64>(pattern.nnz()).unwrap();
                let matrix = DeviceCsr::try_from_host(
                    backend,
                    n,
                    n,
                    pattern.row_ptrs(),
                    pattern.columns(),
                    &vec![Complex64::default(); pattern.nnz()],
                )
                .unwrap();
                let mut persistent =
                    PersistentDeviceCsr::try_from_device_csr(backend, matrix).unwrap();
                let input: Vec<_> = (0..n)
                    .map(|i| Complex64::new(i as f64 * 0.2, 0.3))
                    .collect();
                let x = upload(&pool, &input).unwrap();
                let mut y = pool.try_alloc::<Complex64>(n).unwrap();
                for (ds, ms) in [
                    (Complex64::new(1., 0.), Complex64::new(1., 0.)),
                    (Complex64::new(1., 0.), Complex64::new(1., 0.5)),
                    (Complex64::new(2., 0.1), Complex64::new(-0.3, 0.2)),
                ] {
                    let before = backend.runtime_stats();
                    assembly
                        .try_assemble_values(ds, ms, values.raw_mut())
                        .unwrap();
                    let after = backend.runtime_stats();
                    assert_eq!(after.kernel_launches, before.kernel_launches + 1);
                    assert_eq!(after.allocation_operations, before.allocation_operations);
                    assert_eq!(after.transfer.h2d_bytes, before.transfer.h2d_bytes);
                    assert_eq!(after.transfer.d2h_bytes, before.transfer.d2h_bytes);
                    persistent
                        .try_copy_values_from_device(backend, values.raw())
                        .unwrap();
                    persistent.try_apply(backend, x.raw(), y.raw_mut()).unwrap();
                    let (_, reference) = host.assemble_csr(ds, ms).unwrap();
                    let mut actual = vec![Complex64::default(); pattern.nnz()];
                    backend.try_copy_d2h(values.raw(), &mut actual).unwrap();
                    for (a, b) in actual.iter().zip(&reference) {
                        assert!((*a - *b).norm() < 2e-12 * (1. + b.norm()));
                    }
                    let mut action = vec![Complex64::default(); n];
                    backend.try_copy_d2h(y.raw(), &mut action).unwrap();
                    for row in 0..n {
                        let expected: Complex64 = (pattern.row_ptrs()[row] as usize
                            ..pattern.row_ptrs()[row + 1] as usize)
                            .map(|j| reference[j] * input[pattern.columns()[j] as usize])
                            .sum();
                        assert!((action[row] - expected).norm() < 2e-12 * (1. + expected.norm()));
                    }
                }
                let mut short = pool.try_alloc::<Complex64>(1).unwrap();
                assert!(
                    assembly
                        .try_assemble_values(
                            Complex64::new(1., 0.),
                            Complex64::new(1., 0.),
                            short.raw_mut()
                        )
                        .is_err()
                );
                assert!(
                    assembly
                        .try_assemble_values(
                            Complex64::new(f64::NAN, 0.),
                            Complex64::new(1., 0.),
                            values.raw_mut()
                        )
                        .is_err()
                );
            }
            for rhs in [1, 3, 17, 129, 1] {
                let input: Vec<_> = (0..n * rhs)
                    .map(|i| Complex64::new(0.1 + i as f64 / 7.0, -0.4 + i as f64 / 13.0))
                    .collect();
                let mut x = pool.try_alloc::<Complex64>(input.len()).unwrap();
                let mut y = pool.try_alloc::<Complex64>(input.len()).unwrap();
                backend.try_copy_h2d(&input, x.raw_mut()).unwrap();
                for action in [Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans] {
                    // Warm first, then inspect only the kernel call's allocations and traffic.
                    operator
                        .try_apply_action(backend, rhs, x.raw(), y.raw_mut(), action)
                        .unwrap();
                    backend.try_synchronize().unwrap();
                    let before = backend.runtime_stats();
                    operator
                        .try_apply_action(backend, rhs, x.raw(), y.raw_mut(), action)
                        .unwrap();
                    let after = backend.runtime_stats();
                    assert_eq!(after.allocation_operations, before.allocation_operations);
                    assert_eq!(after.transfer.h2d_bytes, before.transfer.h2d_bytes);
                    assert_eq!(after.transfer.d2h_bytes, before.transfer.d2h_bytes);
                    assert_eq!(after.kernel_launches, before.kernel_launches + 1);
                    let mut observed = vec![Complex64::new(0., 0.); input.len()];
                    backend.try_copy_d2h(y.raw(), &mut observed).unwrap();
                    for col in 0..rhs {
                        let mut expected = vec![Complex64::new(0., 0.); n];
                        host.apply(&input[col * n..col * n + n], &mut expected, action)
                            .unwrap();
                        for i in 0..n {
                            assert!((observed[col * n + i] - expected[i]).norm() < 1e-12);
                        }
                    }
                }
                let rhs_host: Vec<_> = input
                    .iter()
                    .map(|v| v.conj() + Complex64::new(0.7, -0.2))
                    .collect();
                let inverse: Vec<_> = host
                    .diagonal()
                    .iter()
                    .map(|d| Complex64::new(1.0, 0.0) / d)
                    .collect();
                let b = upload(&pool, &rhs_host).unwrap();
                let mut diagonal = upload(&pool, &inverse).unwrap();
                let before_lift = backend.runtime_stats();
                operator
                    .try_lift_rhs_colmajor(backend, rhs, b.raw(), x.raw(), y.raw_mut())
                    .unwrap();
                let after_lift = backend.runtime_stats();
                assert_eq!(after_lift.kernel_launches, before_lift.kernel_launches + 1);
                assert_eq!(
                    after_lift.allocation_operations,
                    before_lift.allocation_operations
                );
                assert_eq!(
                    after_lift.transfer.h2d_bytes,
                    before_lift.transfer.h2d_bytes
                );
                assert_eq!(
                    after_lift.transfer.d2h_bytes,
                    before_lift.transfer.d2h_bytes
                );
                let mut lifted = vec![Complex64::default(); input.len()];
                backend.try_copy_d2h(y.raw(), &mut lifted).unwrap();
                for col in 0..rhs {
                    let mut expected = vec![Complex64::default(); n];
                    host.lift_rhs(
                        &rhs_host[col * n..col * n + n],
                        &input[col * n..col * n + n],
                        &mut expected,
                    )
                    .unwrap();
                    for i in 0..n {
                        assert!((lifted[col * n + i] - expected[i]).norm() < 2e-12);
                    }
                }

                for omega in [0.0, 0.6, 1.0] {
                    let before = backend.runtime_stats();
                    operator
                        .try_jacobi_step(
                            backend,
                            rhs,
                            x.raw(),
                            b.raw(),
                            diagonal.raw(),
                            omega,
                            y.raw_mut(),
                        )
                        .unwrap();
                    let after = backend.runtime_stats();
                    assert_eq!(after.kernel_launches, before.kernel_launches + 1);
                    assert_eq!(after.allocation_operations, before.allocation_operations);
                    assert_eq!(after.transfer.h2d_bytes, before.transfer.h2d_bytes);
                    assert_eq!(after.transfer.d2h_bytes, before.transfer.d2h_bytes);
                    let mut observed = vec![Complex64::default(); input.len()];
                    backend.try_copy_d2h(y.raw(), &mut observed).unwrap();
                    for col in 0..rhs {
                        let mut ax = vec![Complex64::default(); n];
                        host.apply(&input[col * n..col * n + n], &mut ax, Transpose::NoTrans)
                            .unwrap();
                        for i in 0..n {
                            let j = col * n + i;
                            let expected = input[j] + omega * inverse[i] * (rhs_host[j] - ax[i]);
                            assert!(
                                (observed[j] - expected).norm() < 2e-12 * (1.0 + expected.norm())
                            );
                        }
                    }
                }
                operator
                    .try_residual_colmajor(backend, rhs, x.raw(), b.raw(), y.raw_mut())
                    .unwrap();
                let mut residual = vec![Complex64::default(); input.len()];
                backend.try_copy_d2h(y.raw(), &mut residual).unwrap();
                for col in 0..rhs {
                    let mut ax = vec![Complex64::default(); n];
                    host.apply(&input[col * n..col * n + n], &mut ax, Transpose::NoTrans)
                        .unwrap();
                    for i in 0..n {
                        let j = col * n + i;
                        assert!((residual[j] - (rhs_host[j] - ax[i])).norm() < 2e-12);
                    }
                }
                if rhs == 1 {
                    let before = backend.runtime_stats();
                    operator
                        .try_residual(backend, x.raw(), b.raw(), y.raw_mut())
                        .unwrap();
                    assert_eq!(
                        backend.runtime_stats().kernel_launches,
                        before.kernel_launches + 1
                    );
                    let mut ax = vec![Complex64::default(); n];
                    host.apply(&input, &mut ax, Transpose::NoTrans).unwrap();
                    let mut observed = vec![Complex64::default(); n];
                    backend.try_copy_d2h(y.raw(), &mut observed).unwrap();
                    for i in 0..n {
                        assert!((observed[i] - (rhs_host[i] - ax[i])).norm() < 2e-12);
                    }
                }
                backend.try_copy_h2d(&input, y.raw_mut()).unwrap();
                assert!(
                    operator
                        .try_jacobi_step(
                            backend,
                            rhs,
                            x.raw(),
                            b.raw(),
                            diagonal.raw(),
                            f64::NAN,
                            y.raw_mut()
                        )
                        .is_err()
                );
                let short = pool.try_alloc::<Complex64>(4).unwrap();
                assert!(
                    operator
                        .try_jacobi_step(
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
                let mut malformed = pool.try_alloc::<u32>(1).unwrap();
                std::mem::swap(
                    diagonal.raw_mut().raw_mut().unwrap(),
                    malformed.raw_mut().raw_mut().unwrap(),
                );
                assert!(
                    operator
                        .try_jacobi_step(
                            backend,
                            rhs,
                            x.raw(),
                            b.raw(),
                            diagonal.raw(),
                            0.6,
                            y.raw_mut()
                        )
                        .is_err()
                );
                std::mem::swap(
                    diagonal.raw_mut().raw_mut().unwrap(),
                    malformed.raw_mut().raw_mut().unwrap(),
                );
                let mut sentinel = vec![Complex64::default(); input.len()];
                backend.try_copy_d2h(y.raw(), &mut sentinel).unwrap();
                assert_eq!(sentinel, input);
                // Exercise the checked operator contract and graph replay as used by solvers.
                let graph = backend
                    .try_capture_graph_checked(|b| {
                        operator.try_apply_colmajor(b, rhs, x.raw(), y.raw_mut())
                    })
                    .unwrap();
                graph.try_launch().unwrap();
                backend.try_synchronize().unwrap();
                drop(graph);
                let sentinel = vec![Complex64::new(7., -3.); input.len()];
                backend.try_copy_h2d(&sentinel, y.raw_mut()).unwrap();
                backend.inject_failure(crate::CudaFailureInjectionPoint::Launch);
                let error = operator
                    .try_apply_colmajor(backend, rhs, x.raw(), y.raw_mut())
                    .unwrap_err();
                assert_eq!(
                    error.output_disposition(),
                    crate::BackendOutputDisposition::Unusable
                );
                // Launch errors conservatively invalidate output even when injected
                // before dispatch. Synchronize before reusing the context; the
                // sentinel check below verifies this injection's no-write behavior.
                backend.recover_after_error().unwrap();
                let mut unchanged = vec![Complex64::new(0., 0.); input.len()];
                backend.try_copy_d2h(y.raw(), &mut unchanged).unwrap();
                assert_eq!(unchanged, sentinel);
                operator
                    .try_apply_colmajor(backend, rhs, x.raw(), y.raw_mut())
                    .unwrap();
                let mut short = pool.try_alloc::<u32>(1).unwrap();
                std::mem::swap(
                    x.raw_mut().raw_mut().unwrap(),
                    short.raw_mut().raw_mut().unwrap(),
                );
                let error = operator
                    .try_apply_colmajor(backend, rhs, x.raw(), y.raw_mut())
                    .unwrap_err();
                assert_eq!(
                    error.output_disposition(),
                    crate::BackendOutputDisposition::Preserved
                );
                std::mem::swap(
                    x.raw_mut().raw_mut().unwrap(),
                    short.raw_mut().raw_mut().unwrap(),
                );
            }
            drop(operator);
            assert_eq!(pool.bytes_in_use(), 0);
        }
    }
}
