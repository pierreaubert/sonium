//! CUDA compressible-DG operator behind NVRTC compilation.
//!
//! Device counterpart of the CPU element-block DG operator in
//! `sonium-fem`: the host stages the mesh-derived tables (quadrature
//! tabulations, facet/boundary CSR, limiter neighbours, Persson
//! matrices) and the initial state exactly once, after which every
//! SSPRK3 stage, limiter pass, and CFL reduction runs on the device.
//! Only scalars cross the bus per step — the accepted time step, retry
//! flags, and the step-report reductions — so the state stays resident.
//!
//! Each kernel assigns one thread block per element and replays the
//! CPU accumulation order inside that element, matching the CPU
//! operator to round-off. NVRTC compiles with `--fmad=false`, as in
//! [`super::explicit`]. Buffer staging flows through
//! [`MemoryPool`](crate::pool::MemoryPool) quota accounting.

// Rust guideline compliant 2026-02-21

use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};

use crate::backend::Backend;
use crate::cuda::{CudaBackend, CudaBuffer};
use crate::error::{BackendError, BackendResult};
use crate::pool::MemoryPool;

/// NVRTC source for the `sonium_dg_*` kernels.
const DG_KERNELS_CU: &str = include_str!("kernels/dg_kernels.cu");

/// Threads per block: four warps, one element per warp; local node
/// counts never exceed 10, so the extra lanes only idle (serial
/// per-element work runs on the warp's lane 0).
const BLOCK_THREADS: u32 = 128;

/// Scalar kernel parameters, layout-matched with `DgParams` in
/// `kernels/dg_kernels.cu`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DgParams {
    /// Spatial dimension (2 or 3).
    pub dim: i32,
    /// Local Lagrange nodes per element (uniform across the mesh).
    pub n: i32,
    /// Numerical flux: 0 = Rusanov, 1 = HLLC with Rusanov fallback.
    pub flux_kind: i32,
    /// Viscosity law: 0 = constant, 1 = Sutherland.
    pub visc_kind: i32,
    /// Troubled-cell sensor: 0 = Persson--Peraire, 1 = jump/deviation,
    /// 2 = always troubled.
    pub sensor_mode: i32,
    /// First highest-degree modal index for the Persson indicator.
    pub persson_highest: i32,
    /// Heat-capacity ratio.
    pub gamma: f64,
    /// Gas constant.
    pub gas_constant: f64,
    /// Prandtl number.
    pub prandtl: f64,
    /// Constant viscosity, or Sutherland reference viscosity.
    pub visc_a: f64,
    /// Sutherland reference temperature (unused for constant law).
    pub visc_b: f64,
    /// Sutherland temperature (unused for constant law).
    pub visc_c: f64,
    /// Density floor for physical-state checks.
    pub density_floor: f64,
    /// Pressure floor for physical-state checks.
    pub pressure_floor: f64,
    /// SIPG penalty coefficient.
    pub sipg_penalty: f64,
    /// `cfl * 3 / (2p + 1)`, folded on the host.
    pub cfl_scaled: f64,
    /// Limiter relaxation (overshoot margin factor).
    pub relaxation: f64,
    /// Jump/deviation troubled-cell threshold.
    pub threshold: f64,
    /// Persson--Peraire log10 threshold.
    pub persson_threshold: f64,
}

// SAFETY: `DgParams` is a plain-old-data `#[repr(C)]` struct of `i32`
// and `f64` fields with no references or padding-dependent semantics;
// cudarc copies it verbatim into the kernel parameter buffer.
unsafe impl cudarc::driver::DeviceRepr for DgParams {}

/// Host-staged compressible-DG problem tables.
///
/// All arrays use the layouts documented on [`CudaDgOperator`]; lengths
/// are validated by [`CudaDgOperator::load`]. Indices are `i32` to match
/// the kernel CSR types.
#[derive(Debug, Clone, PartialEq)]
pub struct CudaDgProblem {
    /// Scalar kernel parameters.
    pub params: DgParams,
    /// Number of elements.
    pub num_elements: usize,
    /// Volume quadrature points per element.
    pub nq_vol: usize,
    /// Volume basis values `[nq_vol][n]`.
    pub vol_basis: Vec<f64>,
    /// Physical basis gradients `[E][nq_vol][n][dim]` (zero-padded to 3).
    pub vol_grad: Vec<f64>,
    /// Quadrature weight times Jacobian determinant `[E][nq_vol]`.
    pub vol_wdet: Vec<f64>,
    /// Centroid-tabulation gradients `[E][n][dim]` (zero-padded to 3).
    pub cent_grad: Vec<f64>,
    /// Per-element inverse mass matrices `[E][n][n]`.
    pub mass_inv: Vec<f64>,
    /// Cell-average quadrature weights `[n]`.
    pub avg_weights: Vec<f64>,
    /// Per-element minimum altitude (stable-step length).
    pub elem_min_altitude: Vec<f64>,
    /// Per-element volume.
    pub elem_volume: Vec<f64>,
    /// Facet CSR element offsets `[E + 1]`.
    pub facet_elem_offsets: Vec<i32>,
    /// Facet neighbour element per record.
    pub facet_neighbor: Vec<i32>,
    /// Facet plus-side flag per record (1 when this element is plus).
    pub facet_is_plus: Vec<i32>,
    /// Facet unit normals `[records][3]`, oriented plus to minus.
    pub facet_normal: Vec<f64>,
    /// Facet penalty length per record (mean of the two diameters).
    pub facet_h: Vec<f64>,
    /// Facet quadrature-point offsets `[records + 1]`.
    pub facet_quad_offsets: Vec<i32>,
    /// Own-side basis values at facet quadrature points `[points][n]`.
    pub facet_basis_own: Vec<f64>,
    /// Neighbour-side basis values `[points][n]`.
    pub facet_basis_other: Vec<f64>,
    /// Facet quadrature weights `[points]` (measure included).
    pub facet_weights: Vec<f64>,
    /// Boundary CSR element offsets `[E + 1]`.
    pub bnd_elem_offsets: Vec<i32>,
    /// Boundary condition kind per record (see kernel source).
    pub bnd_kind: Vec<i32>,
    /// Boundary condition parameters `[records][8]`.
    pub bnd_params: Vec<f64>,
    /// Outward boundary normals `[records][3]`.
    pub bnd_normal: Vec<f64>,
    /// Boundary penalty length per record (element diameter).
    pub bnd_h: Vec<f64>,
    /// Boundary quadrature-point offsets `[records + 1]`.
    pub bnd_quad_offsets: Vec<i32>,
    /// Basis values at boundary quadrature points `[points][n]`.
    pub bnd_basis: Vec<f64>,
    /// Boundary quadrature weights `[points]` (measure included).
    pub bnd_weights: Vec<f64>,
    /// Limiter neighbour CSR element offsets `[E + 1]`.
    pub lim_offsets: Vec<i32>,
    /// Limiter neighbour element per record.
    pub lim_neighbor: Vec<i32>,
    /// Limiter normals `[records][3]`, pointing toward the neighbour.
    pub lim_normal: Vec<f64>,
    /// Persson--Peraire inverse Vandermonde `[n][n]` (empty if unused).
    pub persson_vinv: Vec<f64>,
    /// Persson--Peraire modal Gram matrix `[n][n]` (empty if unused).
    pub persson_gram: Vec<f64>,
}

/// State-buffer slots owned by [`CudaDgOperator`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DgSlot {
    /// The accepted (current) state.
    Current,
    /// Frozen step-initial state during a trial.
    Original,
    /// First stage/scratch slot.
    StageA,
    /// Second stage/scratch slot.
    StageB,
}

impl DgSlot {
    fn index(self) -> usize {
        match self {
            Self::Current => 0,
            Self::Original => 1,
            Self::StageA => 2,
            Self::StageB => 3,
        }
    }
}

type F64Buffer = crate::pool::Buffer<f64, CudaBackend>;
type I32Buffer = crate::pool::Buffer<i32, CudaBackend>;

/// NVRTC `--gpu-architecture` literal for a device compute capability,
/// mirroring the mapping used for the explicit-dynamics kernels.
fn dg_nvrtc_arch(major: i32, minor: i32) -> Option<&'static str> {
    match (major, minor) {
        (5, _) => Some("compute_50"),
        (6, _) => Some("compute_60"),
        (7, 0) | (7, 1) | (7, 2) => Some("compute_70"),
        (7, _) => Some("compute_75"),
        (8, 0) => Some("compute_80"),
        (8, 6) | (8, 7) => Some("compute_86"),
        (8, _) => Some("compute_89"),
        (9, _) => Some("compute_90"),
        (10, _) => Some("compute_100"),
        (12, _) => Some("compute_120"),
        _ => None,
    }
}

/// Compile the DG kernels, preferring the device arch and falling back
/// to NVRTC defaults; `--fmad=false` keeps device arithmetic on the
/// same rounding path as the CPU reference.
fn compile_dg_ptx(arch: Option<&'static str>) -> Result<Ptx, String> {
    if let Some(arch) = arch {
        let opts = CompileOptions {
            arch: Some(arch),
            fmad: Some(false),
            ..Default::default()
        };
        if let Ok(ptx) = compile_ptx_with_opts(DG_KERNELS_CU, opts) {
            return Ok(ptx);
        }
    }
    let opts = CompileOptions {
        fmad: Some(false),
        ..Default::default()
    };
    compile_ptx_with_opts(DG_KERNELS_CU, opts).map_err(|error| format!("{error:?}"))
}

fn load_failed(operation: &'static str, error: impl core::fmt::Debug) -> BackendError {
    BackendError::OperationFailed {
        operation,
        reason: format!("{error:?}"),
    }
}

fn check_u32(operation: &'static str, what: &str, value: usize) -> BackendResult<u32> {
    u32::try_from(value).map_err(|_| BackendError::InvalidArgument {
        operation,
        reason: format!("{what} {value} exceeds the CUDA kernel index range"),
    })
}

fn invalid(operation: &'static str, reason: impl Into<String>) -> BackendError {
    BackendError::InvalidArgument {
        operation,
        reason: reason.into(),
    }
}

/// Borrow the device slice behind a pool-backed buffer for a launch.
fn launch_slice<'a, T: crate::scalar::BufferElement>(
    operation: &'static str,
    what: &'static str,
    buffer: &'a CudaBuffer<T>,
) -> BackendResult<&'a cudarc::driver::CudaSlice<u8>> {
    buffer.raw().ok_or_else(|| BackendError::InvalidArgument {
        operation,
        reason: format!("{what} buffer is unexpectedly empty"),
    })
}

/// Mutably borrow the device slice behind a pool-backed buffer.
fn launch_slice_mut<'a, T: crate::scalar::BufferElement>(
    operation: &'static str,
    what: &'static str,
    buffer: &'a mut CudaBuffer<T>,
) -> BackendResult<&'a mut cudarc::driver::CudaSlice<u8>> {
    buffer
        .raw_mut()
        .ok_or_else(|| BackendError::InvalidArgument {
            operation,
            reason: format!("{what} buffer is unexpectedly empty"),
        })
}

/// Device-resident compressible-DG operator.
///
/// Owns the compiled kernels, the staged problem tables, four state
/// slots, and the reduction scalars. All methods launch on the pool
/// backend's stream; the ones that return scalars synchronize and copy
/// back only those scalars.
#[derive(Debug)]
pub struct CudaDgOperator {
    pool: MemoryPool<CudaBackend>,
    params: DgParams,
    num_elements: usize,
    total_slots: usize,
    nq_vol: i32,
    reset_scalars: CudaFunction,
    averages_kernel: CudaFunction,
    dt_kernel: CudaFunction,
    rhs_kernel: CudaFunction,
    limiter_kernel: CudaFunction,
    zhangshu_kernel: CudaFunction,
    add_scaled_kernel: CudaFunction,
    convex_kernel: CudaFunction,
    copy_kernel: CudaFunction,
    min_thermo_kernel: CudaFunction,
    totals_kernel: CudaFunction,
    vol_basis: F64Buffer,
    vol_grad: F64Buffer,
    vol_wdet: F64Buffer,
    cent_grad: F64Buffer,
    mass_inv: F64Buffer,
    avg_weights: F64Buffer,
    elem_min_altitude: F64Buffer,
    elem_volume: F64Buffer,
    facet_elem_offsets: I32Buffer,
    facet_neighbor: I32Buffer,
    facet_is_plus: I32Buffer,
    facet_normal: F64Buffer,
    facet_h: F64Buffer,
    facet_quad_offsets: I32Buffer,
    facet_basis_own: F64Buffer,
    facet_basis_other: F64Buffer,
    facet_weights: F64Buffer,
    bnd_elem_offsets: I32Buffer,
    bnd_kind: I32Buffer,
    bnd_params: F64Buffer,
    bnd_normal: F64Buffer,
    bnd_h: F64Buffer,
    bnd_quad_offsets: I32Buffer,
    bnd_basis: F64Buffer,
    bnd_weights: F64Buffer,
    lim_offsets: I32Buffer,
    lim_neighbor: I32Buffer,
    lim_normal: F64Buffer,
    persson_vinv: F64Buffer,
    persson_gram: F64Buffer,
    states: [F64Buffer; 4],
    rate: F64Buffer,
    averages: F64Buffer,
    dt_out: F64Buffer,
    flag: I32Buffer,
    min_thermo: F64Buffer,
    totals: F64Buffer,
}

fn check_csr(
    operation: &'static str,
    what: &str,
    offsets: &[i32],
    num_elements: usize,
    records: usize,
) -> BackendResult<()> {
    if offsets.len() != num_elements + 1 {
        return Err(invalid(
            operation,
            format!("{what} offsets length {} != elements + 1", offsets.len()),
        ));
    }
    if offsets[0] != 0 || offsets[num_elements] as usize != records {
        return Err(invalid(
            operation,
            format!("{what} offsets must span [0, {records}]"),
        ));
    }
    if offsets.windows(2).any(|pair| pair[1] < pair[0]) {
        return Err(invalid(operation, format!("{what} offsets not monotone")));
    }
    Ok(())
}

/// Borrow one mutable and one shared state slot from the slot array.
///
/// The split is scoped to the `states` field so other operator fields
/// stay accessible while the returned references are alive.
fn slot_pair(
    states: &mut [F64Buffer; 4],
    mutable: DgSlot,
    shared: DgSlot,
) -> BackendResult<(
    &mut cudarc::driver::CudaSlice<u8>,
    &cudarc::driver::CudaSlice<u8>,
)> {
    if mutable == shared {
        return Err(invalid("dg_slots", "mutable and shared slots must differ"));
    }
    let (mi, si) = (mutable.index(), shared.index());
    if mi < si {
        let (left, right) = states.split_at_mut(si);
        let out = launch_slice_mut("dg_slots", "out", left[mi].raw_mut())?;
        let input = launch_slice("dg_slots", "input", right[0].raw())?;
        Ok((out, input))
    } else {
        let (left, right) = states.split_at_mut(mi);
        let input = launch_slice("dg_slots", "input", left[si].raw())?;
        let out = launch_slice_mut("dg_slots", "out", right[0].raw_mut())?;
        Ok((out, input))
    }
}

/// Borrow one mutable and two shared state slots (convex combination).
fn slot_triple(
    states: &mut [F64Buffer; 4],
    mutable: DgSlot,
    first: DgSlot,
    second: DgSlot,
) -> BackendResult<(
    &mut cudarc::driver::CudaSlice<u8>,
    &cudarc::driver::CudaSlice<u8>,
    &cudarc::driver::CudaSlice<u8>,
)> {
    if mutable == first || mutable == second {
        return Err(invalid("dg_slots", "mutable slot must differ from inputs"));
    }
    let (oi, xi, yi) = (mutable.index(), first.index(), second.index());
    if oi < xi && oi < yi {
        let (head, tail) = states.split_at_mut(oi + 1);
        let out = launch_slice_mut("dg_slots", "out", head[oi].raw_mut())?;
        let x = launch_slice("dg_slots", "x", tail[xi - oi - 1].raw())?;
        let y = launch_slice("dg_slots", "y", tail[yi - oi - 1].raw())?;
        Ok((out, x, y))
    } else if oi > xi && oi > yi {
        let (head, tail) = states.split_at_mut(oi);
        let x = launch_slice("dg_slots", "x", head[xi].raw())?;
        let y = launch_slice("dg_slots", "y", head[yi].raw())?;
        let out = launch_slice_mut("dg_slots", "out", tail[0].raw_mut())?;
        Ok((out, x, y))
    } else if xi < oi && oi < yi {
        let (head, tail) = states.split_at_mut(oi);
        let x = launch_slice("dg_slots", "x", head[xi].raw())?;
        let (mid, rest) = tail.split_at_mut(1);
        let out = launch_slice_mut("dg_slots", "out", mid[0].raw_mut())?;
        let y = launch_slice("dg_slots", "y", rest[yi - oi - 1].raw())?;
        Ok((out, x, y))
    } else {
        // yi < oi < xi
        let (head, tail) = states.split_at_mut(oi);
        let y = launch_slice("dg_slots", "y", head[yi].raw())?;
        let (mid, rest) = tail.split_at_mut(1);
        let out = launch_slice_mut("dg_slots", "out", mid[0].raw_mut())?;
        let x = launch_slice("dg_slots", "x", rest[xi - oi - 1].raw())?;
        Ok((out, x, y))
    }
}

fn element_launch_config(num_elements: usize, operation: &'static str) -> BackendResult<LaunchConfig> {
    Ok(LaunchConfig {
        grid_dim: (check_u32(operation, "element warps", num_elements.div_ceil(4))?, 1, 1),
        block_dim: (BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    })
}

fn slot_launch_config(operation: &'static str, count: usize) -> BackendResult<LaunchConfig> {
    Ok(LaunchConfig::for_num_elems(check_u32(
        operation,
        "slot count",
        count,
    )?))
}

impl CudaDgOperator {
    /// Compile the kernels and stage a problem on the device.
    ///
    /// Validates every host array length and CSR invariant before the
    /// first upload; the upload itself is the only bulk host-to-device
    /// transfer the operator ever performs (besides state uploads the
    /// caller requests explicitly).
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::InvalidArgument`] on any layout mismatch
    /// and [`BackendError::OperationFailed`] when NVRTC compilation,
    /// module loading, or a staging copy fails.
    pub fn load(pool: MemoryPool<CudaBackend>, problem: CudaDgProblem) -> BackendResult<Self> {
        const OPERATION: &str = "dg_load";
        let params = problem.params;
        let num_elements = problem.num_elements;
        let n = params.n;
        if !(2..=3).contains(&params.dim) || !(1..=10).contains(&n) {
            return Err(invalid(
                OPERATION,
                format!("unsupported dim {} or local nodes {n}", params.dim),
            ));
        }
        let n = n as usize;
        let nq = problem.nq_vol;
        let expect = [
            ("vol_basis", problem.vol_basis.len(), nq * n),
            ("vol_grad", problem.vol_grad.len(), num_elements * nq * n * 3),
            ("vol_wdet", problem.vol_wdet.len(), num_elements * nq),
            ("cent_grad", problem.cent_grad.len(), num_elements * n * 3),
            ("mass_inv", problem.mass_inv.len(), num_elements * n * n),
            ("avg_weights", problem.avg_weights.len(), n),
            ("elem_min_altitude", problem.elem_min_altitude.len(), num_elements),
            ("elem_volume", problem.elem_volume.len(), num_elements),
        ];
        for (what, got, want) in expect {
            if got != want {
                return Err(invalid(
                    OPERATION,
                    format!("{what} length {got} != expected {want}"),
                ));
            }
        }
        let facet_records = problem.facet_neighbor.len();
        let bnd_records = problem.bnd_kind.len();
        check_csr(OPERATION, "facet", &problem.facet_elem_offsets, num_elements, facet_records)?;
        check_csr(OPERATION, "boundary", &problem.bnd_elem_offsets, num_elements, bnd_records)?;
        check_csr(
            OPERATION,
            "limiter",
            &problem.lim_offsets,
            num_elements,
            problem.lim_neighbor.len(),
        )?;
        if problem.facet_quad_offsets.len() != facet_records + 1
            || problem.bnd_quad_offsets.len() != bnd_records + 1
        {
            return Err(invalid(OPERATION, "quadrature offsets must have records + 1 entries"));
        }
        let facet_points = problem.facet_weights.len();
        let bnd_points = problem.bnd_weights.len();
        if problem.facet_basis_own.len() != facet_points * n
            || problem.facet_basis_other.len() != facet_points * n
            || problem.bnd_basis.len() != bnd_points * n
            || problem.facet_normal.len() != facet_records * 3
            || problem.facet_h.len() != facet_records
            || problem.facet_is_plus.len() != facet_records
            || problem.bnd_normal.len() != bnd_records * 3
            || problem.bnd_h.len() != bnd_records
            || problem.bnd_params.len() != bnd_records * 8
            || problem.lim_normal.len() != problem.lim_neighbor.len() * 3
        {
            return Err(invalid(OPERATION, "facet/boundary/limiter array length mismatch"));
        }
        if params.sensor_mode == 0
            && (problem.persson_vinv.len() != n * n || problem.persson_gram.len() != n * n)
        {
            return Err(invalid(
                OPERATION,
                "Persson sensor mode requires n x n Persson matrices",
            ));
        }
        if problem
            .facet_neighbor
            .iter()
            .chain(problem.lim_neighbor.iter())
            .any(|& idx| idx < 0 || idx as usize >= num_elements)
        {
            return Err(invalid(OPERATION, "neighbour index out of range"));
        }

        let backend = pool.backend();
        let arch = backend
            .context()
            .compute_capability()
            .ok()
            .and_then(|(major, minor)| dg_nvrtc_arch(major, minor));
        let ptx = compile_dg_ptx(arch).map_err(|reason| BackendError::OperationFailed {
            operation: "dg_nvrtc_compile",
            reason,
        })?;
        let module = backend
            .context()
            .load_module(ptx)
            .map_err(|error| load_failed("dg_module_load", error))?;
        let function = |name: &str| {
            module
                .load_function(name)
                .map_err(|error| load_failed("dg_function_load", error))
        };
        let reset_scalars = function("sonium_dg_reset_scalars")?;
        let averages_kernel = function("sonium_dg_averages")?;
        let dt_kernel = function("sonium_dg_dt")?;
        let rhs_kernel = function("sonium_dg_rhs")?;
        let limiter_kernel = function("sonium_dg_limiter")?;
        let zhangshu_kernel = function("sonium_dg_zhangshu")?;
        let add_scaled_kernel = function("sonium_dg_add_scaled")?;
        let convex_kernel = function("sonium_dg_convex")?;
        let copy_kernel = function("sonium_dg_copy")?;
        let min_thermo_kernel = function("sonium_dg_min_thermo")?;
        let totals_kernel = function("sonium_dg_totals")?;

        let stage_f64 = |data: &[f64]| -> BackendResult<F64Buffer> {
            let mut buffer = pool.try_alloc::<f64>(data.len().max(1))?;
            if !data.is_empty() {
                pool.backend().try_copy_h2d(data, buffer.raw_mut())?;
            }
            Ok(buffer)
        };
        let stage_i32 = |data: &[i32]| -> BackendResult<I32Buffer> {
            let mut buffer = pool.try_alloc::<i32>(data.len().max(1))?;
            if !data.is_empty() {
                pool.backend().try_copy_h2d(data, buffer.raw_mut())?;
            }
            Ok(buffer)
        };
        let total_slots = num_elements * n;
        let state_len = total_slots * 5;
        let states = [
            pool.try_alloc::<f64>(state_len)?,
            pool.try_alloc::<f64>(state_len)?,
            pool.try_alloc::<f64>(state_len)?,
            pool.try_alloc::<f64>(state_len)?,
        ];
        Ok(Self {
            vol_basis: stage_f64(&problem.vol_basis)?,
            vol_grad: stage_f64(&problem.vol_grad)?,
            vol_wdet: stage_f64(&problem.vol_wdet)?,
            cent_grad: stage_f64(&problem.cent_grad)?,
            mass_inv: stage_f64(&problem.mass_inv)?,
            avg_weights: stage_f64(&problem.avg_weights)?,
            elem_min_altitude: stage_f64(&problem.elem_min_altitude)?,
            elem_volume: stage_f64(&problem.elem_volume)?,
            facet_elem_offsets: stage_i32(&problem.facet_elem_offsets)?,
            facet_neighbor: stage_i32(&problem.facet_neighbor)?,
            facet_is_plus: stage_i32(&problem.facet_is_plus)?,
            facet_normal: stage_f64(&problem.facet_normal)?,
            facet_h: stage_f64(&problem.facet_h)?,
            facet_quad_offsets: stage_i32(&problem.facet_quad_offsets)?,
            facet_basis_own: stage_f64(&problem.facet_basis_own)?,
            facet_basis_other: stage_f64(&problem.facet_basis_other)?,
            facet_weights: stage_f64(&problem.facet_weights)?,
            bnd_elem_offsets: stage_i32(&problem.bnd_elem_offsets)?,
            bnd_kind: stage_i32(&problem.bnd_kind)?,
            bnd_params: stage_f64(&problem.bnd_params)?,
            bnd_normal: stage_f64(&problem.bnd_normal)?,
            bnd_h: stage_f64(&problem.bnd_h)?,
            bnd_quad_offsets: stage_i32(&problem.bnd_quad_offsets)?,
            bnd_basis: stage_f64(&problem.bnd_basis)?,
            bnd_weights: stage_f64(&problem.bnd_weights)?,
            lim_offsets: stage_i32(&problem.lim_offsets)?,
            lim_neighbor: stage_i32(&problem.lim_neighbor)?,
            lim_normal: stage_f64(&problem.lim_normal)?,
            persson_vinv: stage_f64(&problem.persson_vinv)?,
            persson_gram: stage_f64(&problem.persson_gram)?,
            rate: pool.try_alloc::<f64>(state_len)?,
            averages: pool.try_alloc::<f64>(num_elements * 5)?,
            dt_out: pool.try_alloc::<f64>(1)?,
            flag: pool.try_alloc::<i32>(1)?,
            min_thermo: pool.try_alloc::<f64>(2)?,
            totals: pool.try_alloc::<f64>(5)?,
            states,
            pool,
            params,
            num_elements,
            total_slots,
            nq_vol: i32::try_from(nq).map_err(|_| invalid(OPERATION, "nq_vol exceeds i32"))?,
            reset_scalars,
            averages_kernel,
            dt_kernel,
            rhs_kernel,
            limiter_kernel,
            zhangshu_kernel,
            add_scaled_kernel,
            convex_kernel,
            copy_kernel,
            min_thermo_kernel,
            totals_kernel,
        })
    }

    /// The pool backend that owns the stream all kernels launch on.
    #[must_use]
    pub fn backend(&self) -> &CudaBackend {
        self.pool.backend()
    }

    /// Number of nodal slots (`elements * local nodes`); state slices
    /// hold five conserved values per slot.
    #[must_use]
    pub fn total_slots(&self) -> usize {
        self.total_slots
    }


    /// Upload a full state into a slot (five values per nodal slot).
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::InvalidArgument`] on length mismatch and
    /// [`BackendError::OperationFailed`] when the copy fails.
    pub fn upload_state(&mut self, slot: DgSlot, values: &[f64]) -> BackendResult<()> {
        if values.len() != self.total_slots * 5 {
            return Err(invalid(
                "dg_upload_state",
                format!("state length {} != {}", values.len(), self.total_slots * 5),
            ));
        }
        self.pool
            .backend()
            .try_copy_h2d(values, self.states[slot.index()].raw_mut())
    }

    /// Download a full state slot into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::InvalidArgument`] on length mismatch and
    /// [`BackendError::OperationFailed`] when the copy fails.
    pub fn download_state(&self, slot: DgSlot, out: &mut [f64]) -> BackendResult<()> {
        if out.len() != self.total_slots * 5 {
            return Err(invalid("dg_download_state", "output length mismatch"));
        }
        self.pool
            .backend()
            .try_copy_d2h(self.states[slot.index()].raw(), out)
    }

    /// Download the most recent spatial-operator rate.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::InvalidArgument`] on length mismatch and
    /// [`BackendError::OperationFailed`] when the copy fails.
    pub fn download_rate(&self, out: &mut [f64]) -> BackendResult<()> {
        if out.len() != self.total_slots * 5 {
            return Err(invalid("dg_download_rate", "output length mismatch"));
        }
        self.pool.backend().try_copy_d2h(self.rate.raw(), out)
    }

    fn read_flags(&self) -> BackendResult<i32> {
        let mut flag = [0i32; 1];
        self.pool.backend().try_copy_d2h(self.flag.raw(), &mut flag)?;
        Ok(flag[0])
    }

    /// Synchronize and read the kernel flag bitmask.
    ///
    /// Bit 0 marks a nonphysical quadrature/facet/boundary trace; bit 1
    /// marks a nonphysical cell average. Kernels only ever set bits
    /// (`atomicOr`), so deferred launches can accumulate both across a
    /// whole SSPRK3 trial before a single read.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::OperationFailed`] when the sync or the
    /// scalar copy-back fails.
    pub fn flags(&self) -> BackendResult<i32> {
        self.pool.backend().synchronize()?;
        self.read_flags()
    }

    /// Reset the flag bitmask and reduction scalars.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::OperationFailed`] when the launch fails.
    pub fn clear_flags(&mut self) -> BackendResult<()> {
        self.launch_reset()
    }

    fn launch_reset(&mut self) -> BackendResult<()> {
        const OPERATION: &str = "dg_reset_scalars";
        let dt_out = launch_slice_mut(OPERATION, "dt_out", self.dt_out.raw_mut())?;
        let flag = launch_slice_mut(OPERATION, "flag", self.flag.raw_mut())?;
        let min_thermo = launch_slice_mut(OPERATION, "min_thermo", self.min_thermo.raw_mut())?;
        let totals = launch_slice_mut(OPERATION, "totals", self.totals.raw_mut())?;
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the four args match `sonium_dg_reset_scalars`; all
        // buffers are operator-owned live allocations of the right
        // lengths (1, 1, 2, and 5 elements), and launches order on the
        // backend stream.
        unsafe {
            self.pool
                .backend()
                .stream()
                .launch_builder(&self.reset_scalars)
                .arg(dt_out)
                .arg(flag)
                .arg(min_thermo)
                .arg(totals)
                .launch(config)
                .map_err(|error| load_failed(OPERATION, error))?;
        }
        Ok(())
    }

    fn launch_averages(&mut self, slot: DgSlot) -> BackendResult<()> {
        const OPERATION: &str = "dg_averages";
        let state = launch_slice(OPERATION, "state", self.states[slot.index()].raw())?;
        let avg_w = launch_slice(OPERATION, "avg_weights", self.avg_weights.raw())?;
        let averages = launch_slice_mut(OPERATION, "averages", self.averages.raw_mut())?;
        let params = self.params;
        let num_elements = i32::try_from(self.num_elements)
            .map_err(|_| invalid(OPERATION, "element count exceeds i32"))?;
        let config = element_launch_config(self.num_elements, OPERATION)?;
        // SAFETY: matches `sonium_dg_averages`; the state slot holds
        // `total_slots * 5` values and `averages` holds 5 per element,
        // as staged at load.
        unsafe {
            self.pool
                .backend()
                .stream()
                .launch_builder(&self.averages_kernel)
                .arg(state)
                .arg(avg_w)
                .arg(averages)
                .arg(&params)
                .arg(&num_elements)
                .launch(config)
                .map_err(|error| load_failed(OPERATION, error))?;
        }
        Ok(())
    }

    /// Stable time step of a state slot: `(dt, nonphysical_average)`.
    ///
    /// Runs the averages and dt reductions on the device and copies
    /// back only the two scalars.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::OperationFailed`] when a launch or the
    /// scalar copy-back fails.
    pub fn stable_dt(&mut self, slot: DgSlot) -> BackendResult<(f64, bool)> {
        const OPERATION: &str = "dg_dt";
        self.launch_reset()?;
        self.launch_averages(slot)?;
        let averages = launch_slice(OPERATION, "averages", self.averages.raw())?;
        let min_altitude = launch_slice(OPERATION, "min_altitude", self.elem_min_altitude.raw())?;
        let dt_out = launch_slice_mut(OPERATION, "dt_out", self.dt_out.raw_mut())?;
        let flag = launch_slice_mut(OPERATION, "flag", self.flag.raw_mut())?;
        let params = self.params;
        let num_elements = i32::try_from(self.num_elements)
            .map_err(|_| invalid(OPERATION, "element count exceeds i32"))?;
        let config = element_launch_config(self.num_elements, OPERATION)?;
        // SAFETY: matches `sonium_dg_dt`; `averages` was just produced
        // by `launch_averages` on the same stream, and `dt_out`/`flag`
        // were reset above.
        unsafe {
            self.pool
                .backend()
                .stream()
                .launch_builder(&self.dt_kernel)
                .arg(averages)
                .arg(min_altitude)
                .arg(dt_out)
                .arg(flag)
                .arg(&params)
                .arg(&num_elements)
                .launch(config)
                .map_err(|error| load_failed(OPERATION, error))?;
        }
        self.pool.backend().synchronize()?;
        let mut dt = [0.0f64; 1];
        self.pool.backend().try_copy_d2h(self.dt_out.raw(), &mut dt)?;
        Ok((dt[0], self.read_flags()? & 2 != 0))
    }

    /// Spatial operator of a state slot into the internal rate buffer.
    ///
    /// Returns `true` when a quadrature/facet/boundary trace was
    /// nonphysical (the CPU `NonPhysicalQuadratureState` error path).
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::OperationFailed`] when a launch fails.
    pub fn spatial_operator(&mut self, slot: DgSlot) -> BackendResult<bool> {
        self.launch_reset()?;
        self.launch_spatial(slot)?;
        self.pool.backend().synchronize()?;
        Ok(self.read_flags()? & 1 != 0)
    }

    /// Launch the spatial operator without resetting flags, syncing,
    /// or reading results (see [`Self::flags`]).
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::OperationFailed`] when a launch fails.
    pub fn spatial_operator_deferred(&mut self, slot: DgSlot) -> BackendResult<()> {
        self.launch_spatial(slot)
    }

    fn launch_spatial(&mut self, slot: DgSlot) -> BackendResult<()> {
        const OPERATION: &str = "dg_rhs";
        let params = self.params;
        let nq_vol = self.nq_vol;
        let num_elements = i32::try_from(self.num_elements)
            .map_err(|_| invalid(OPERATION, "element count exceeds i32"))?;
        let config = element_launch_config(self.num_elements, OPERATION)?;
        let state = launch_slice(OPERATION, "state", self.states[slot.index()].raw())?;
        let vol_basis = launch_slice(OPERATION, "vol_basis", self.vol_basis.raw())?;
        let vol_grad = launch_slice(OPERATION, "vol_grad", self.vol_grad.raw())?;
        let vol_wdet = launch_slice(OPERATION, "vol_wdet", self.vol_wdet.raw())?;
        let cent_grad = launch_slice(OPERATION, "cent_grad", self.cent_grad.raw())?;
        let f_offsets = launch_slice(OPERATION, "facet_offsets", self.facet_elem_offsets.raw())?;
        let f_neighbor = launch_slice(OPERATION, "facet_neighbor", self.facet_neighbor.raw())?;
        let f_is_plus = launch_slice(OPERATION, "facet_is_plus", self.facet_is_plus.raw())?;
        let f_normal = launch_slice(OPERATION, "facet_normal", self.facet_normal.raw())?;
        let f_h = launch_slice(OPERATION, "facet_h", self.facet_h.raw())?;
        let f_quad = launch_slice(OPERATION, "facet_quad", self.facet_quad_offsets.raw())?;
        let f_bown = launch_slice(OPERATION, "facet_basis_own", self.facet_basis_own.raw())?;
        let f_bother = launch_slice(OPERATION, "facet_basis_other", self.facet_basis_other.raw())?;
        let f_weights = launch_slice(OPERATION, "facet_weights", self.facet_weights.raw())?;
        let b_offsets = launch_slice(OPERATION, "bnd_offsets", self.bnd_elem_offsets.raw())?;
        let b_kind = launch_slice(OPERATION, "bnd_kind", self.bnd_kind.raw())?;
        let b_params = launch_slice(OPERATION, "bnd_params", self.bnd_params.raw())?;
        let b_normal = launch_slice(OPERATION, "bnd_normal", self.bnd_normal.raw())?;
        let b_h = launch_slice(OPERATION, "bnd_h", self.bnd_h.raw())?;
        let b_quad = launch_slice(OPERATION, "bnd_quad", self.bnd_quad_offsets.raw())?;
        let b_basis = launch_slice(OPERATION, "bnd_basis", self.bnd_basis.raw())?;
        let b_weights = launch_slice(OPERATION, "bnd_weights", self.bnd_weights.raw())?;
        let mass_inv = launch_slice(OPERATION, "mass_inv", self.mass_inv.raw())?;
        let rate = launch_slice_mut(OPERATION, "rate", self.rate.raw_mut())?;
        let flag = launch_slice_mut(OPERATION, "flag", self.flag.raw_mut())?;
        // SAFETY: matches `sonium_dg_rhs`; every table length and CSR
        // invariant was validated at load, and the kernel writes the
        // rate for the same slot it reads.
        unsafe {
            self.pool
                .backend()
                .stream()
                .launch_builder(&self.rhs_kernel)
                .arg(state)
                .arg(vol_basis)
                .arg(vol_grad)
                .arg(vol_wdet)
                .arg(cent_grad)
                .arg(f_offsets)
                .arg(f_neighbor)
                .arg(f_is_plus)
                .arg(f_normal)
                .arg(f_h)
                .arg(f_quad)
                .arg(f_bown)
                .arg(f_bother)
                .arg(f_weights)
                .arg(b_offsets)
                .arg(b_kind)
                .arg(b_params)
                .arg(b_normal)
                .arg(b_h)
                .arg(b_quad)
                .arg(b_basis)
                .arg(b_weights)
                .arg(mass_inv)
                .arg(rate)
                .arg(flag)
                .arg(&params)
                .arg(&nq_vol)
                .arg(&num_elements)
                .launch(config)
                .map_err(|error| load_failed(OPERATION, error))?;
        }
        Ok(())
    }

    /// Characteristic limiter plus Zhang--Shu positivity on a slot.
    ///
    /// Returns `true` when a cell average was nonphysical (the CPU
    /// `NonPhysicalCellAverage` error path).
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::OperationFailed`] when a launch fails.
    pub fn limit(&mut self, slot: DgSlot) -> BackendResult<bool> {
        self.launch_reset()?;
        self.launch_limit(slot)?;
        self.pool.backend().synchronize()?;
        Ok(self.read_flags()? & 2 != 0)
    }

    /// Launch the limiter plus Zhang--Shu positivity without syncing
    /// or reading the flag bitmask (see [`Self::flags`]).
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::OperationFailed`] when a launch fails.
    pub fn limit_deferred(&mut self, slot: DgSlot) -> BackendResult<()> {
        self.launch_limit(slot)
    }

    fn launch_limit(&mut self, slot: DgSlot) -> BackendResult<()> {
        const OPERATION: &str = "dg_limiter";
        self.launch_averages(slot)?;
        let params = self.params;
        let num_elements = i32::try_from(self.num_elements)
            .map_err(|_| invalid(OPERATION, "element count exceeds i32"))?;
        let config = element_launch_config(self.num_elements, OPERATION)?;
        let state = launch_slice_mut(OPERATION, "state", self.states[slot.index()].raw_mut())?;
        let averages = launch_slice(OPERATION, "averages", self.averages.raw())?;
        let avg_w = launch_slice(OPERATION, "avg_weights", self.avg_weights.raw())?;
        let offsets = launch_slice(OPERATION, "offsets", self.lim_offsets.raw())?;
        let neighbor = launch_slice(OPERATION, "neighbor", self.lim_neighbor.raw())?;
        let normal = launch_slice(OPERATION, "normal", self.lim_normal.raw())?;
        let vinv = launch_slice(OPERATION, "vinv", self.persson_vinv.raw())?;
        let gram = launch_slice(OPERATION, "gram", self.persson_gram.raw())?;
        // SAFETY: matches `sonium_dg_limiter`; the averages buffer was
        // just produced from this slot on the same stream, and the
        // limiter kernel mutates only each element's own nodal values.
        unsafe {
            self.pool
                .backend()
                .stream()
                .launch_builder(&self.limiter_kernel)
                .arg(state)
                .arg(averages)
                .arg(avg_w)
                .arg(offsets)
                .arg(neighbor)
                .arg(normal)
                .arg(vinv)
                .arg(gram)
                .arg(&params)
                .arg(&num_elements)
                .launch(config)
                .map_err(|error| load_failed(OPERATION, error))?;
        }
        let state = launch_slice_mut("dg_zhangshu", "state", self.states[slot.index()].raw_mut())?;
        let avg_w = launch_slice("dg_zhangshu", "avg_weights", self.avg_weights.raw())?;
        let flag = launch_slice_mut("dg_zhangshu", "flag", self.flag.raw_mut())?;
        let config = element_launch_config(self.num_elements, "dg_zhangshu")?;
        // SAFETY: matches `sonium_dg_zhangshu`; the flag buffer was
        // reset above and is read back after the stream synchronizes.
        unsafe {
            self.pool
                .backend()
                .stream()
                .launch_builder(&self.zhangshu_kernel)
                .arg(state)
                .arg(avg_w)
                .arg(flag)
                .arg(&params)
                .arg(&num_elements)
                .launch(config)
                .map_err(|error| load_failed("dg_zhangshu", error))?;
        }
        Ok(())
    }

    /// `out = x + dt * rate` over all slots (the SSPRK Euler update).
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::OperationFailed`] when the launch fails.
    pub fn add_scaled(&mut self, out: DgSlot, x: DgSlot, dt: f64) -> BackendResult<()> {
        const OPERATION: &str = "dg_add_scaled";
        let dim = self.params.dim;
        let total = i32::try_from(self.total_slots)
            .map_err(|_| invalid(OPERATION, "slot count exceeds i32"))?;
        let config = slot_launch_config(OPERATION, self.total_slots)?;
        let (out_raw, x_raw) = slot_pair(&mut self.states, out, x)?;
        let rate = launch_slice(OPERATION, "rate", self.rate.raw())?;
        // SAFETY: matches `sonium_dg_add_scaled`; `out` and `x` are
        // distinct live slots (checked by `two_slots` callers passing
        // different slots), and `rate` holds one rate per slot.
        unsafe {
            self.pool
                .backend()
                .stream()
                .launch_builder(&self.add_scaled_kernel)
                .arg(out_raw)
                .arg(x_raw)
                .arg(rate)
                .arg(&dt)
                .arg(&dim)
                .arg(&total)
                .launch(config)
                .map_err(|error| load_failed(OPERATION, error))?;
        }
        Ok(())
    }

    /// `out = a * x + b * y` over all slots (SSPRK convex combination).
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::OperationFailed`] when the launch fails.
    pub fn convex(&mut self, out: DgSlot, a: f64, x: DgSlot, b: f64, y: DgSlot) -> BackendResult<()> {
        const OPERATION: &str = "dg_convex";
        if out == x || out == y {
            return Err(invalid(OPERATION, "output slot must differ from inputs"));
        }
        let dim_total = (self.params.dim, self.total_slots);
        let (out_raw, x_raw, y_raw) = slot_triple(&mut self.states, out, x, y)?;
        let (dim, total_slots) = dim_total;
        let total = i32::try_from(total_slots)
            .map_err(|_| invalid(OPERATION, "slot count exceeds i32"))?;
        let config = slot_launch_config(OPERATION, total_slots)?;
        // SAFETY: matches `sonium_dg_convex`; the output slot was just
        // checked distinct from both input slots.
        unsafe {
            self.pool
                .backend()
                .stream()
                .launch_builder(&self.convex_kernel)
                .arg(out_raw)
                .arg(&a)
                .arg(x_raw)
                .arg(&b)
                .arg(y_raw)
                .arg(&dim)
                .arg(&total)
                .launch(config)
                .map_err(|error| load_failed(OPERATION, error))?;
        }
        Ok(())
    }

    /// Copy one state slot into another, entirely on the device.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::OperationFailed`] when the launch fails.
    pub fn copy_slot(&mut self, dst: DgSlot, src: DgSlot) -> BackendResult<()> {
        const OPERATION: &str = "dg_copy";
        if dst == src {
            return Ok(());
        }
        let total = i32::try_from(self.total_slots * 5)
            .map_err(|_| invalid(OPERATION, "state length exceeds i32"))?;
        let config = slot_launch_config(OPERATION, self.total_slots * 5)?;
        let (dst_raw, src_raw) = slot_pair(&mut self.states, dst, src)?;
        // SAFETY: matches `sonium_dg_copy`; the slots are distinct
        // live state buffers of equal length.
        unsafe {
            self.pool
                .backend()
                .stream()
                .launch_builder(&self.copy_kernel)
                .arg(dst_raw)
                .arg(src_raw)
                .arg(&total)
                .launch(config)
                .map_err(|error| load_failed(OPERATION, error))?;
        }
        Ok(())
    }

    /// Minimum nodal density and pressure of a slot.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::OperationFailed`] when a launch or the
    /// scalar copy-back fails.
    pub fn min_thermo(&mut self, slot: DgSlot) -> BackendResult<(f64, f64)> {
        self.launch_reset()?;
        self.launch_min_thermo(slot)?;
        self.pool.backend().synchronize()?;
        let mut out = [0.0f64; 2];
        self.pool
            .backend()
            .try_copy_d2h(self.min_thermo.raw(), &mut out)?;
        Ok((out[0], out[1]))
    }

    fn launch_min_thermo(&mut self, slot: DgSlot) -> BackendResult<()> {
        const OPERATION: &str = "dg_min_thermo";
        let state = launch_slice(OPERATION, "state", self.states[slot.index()].raw())?;
        let min_thermo = launch_slice_mut(OPERATION, "min_thermo", self.min_thermo.raw_mut())?;
        let params = self.params;
        let total = i32::try_from(self.total_slots)
            .map_err(|_| invalid(OPERATION, "slot count exceeds i32"))?;
        let config = slot_launch_config(OPERATION, self.total_slots)?;
        // SAFETY: matches `sonium_dg_min_thermo`; the reduction target
        // was reset to +infinity above.
        unsafe {
            self.pool
                .backend()
                .stream()
                .launch_builder(&self.min_thermo_kernel)
                .arg(state)
                .arg(min_thermo)
                .arg(&params)
                .arg(&total)
                .launch(config)
                .map_err(|error| load_failed(OPERATION, error))?;
        }
        Ok(())
    }

    /// Conserved totals `sum_e volume_e * average_e` of a slot.
    ///
    /// Accumulation uses device atomics, so the totals agree with the
    /// CPU's serial sum to reduction round-off (report-grade values).
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::OperationFailed`] when a launch or the
    /// copy-back fails.
    pub fn totals(&mut self, slot: DgSlot) -> BackendResult<[f64; 5]> {
        self.launch_reset()?;
        self.launch_totals(slot)?;
        self.pool.backend().synchronize()?;
        let mut out = [0.0f64; 5];
        self.pool.backend().try_copy_d2h(self.totals.raw(), &mut out)?;
        Ok(out)
    }

    fn launch_totals(&mut self, slot: DgSlot) -> BackendResult<()> {
        const OPERATION: &str = "dg_totals";
        self.launch_averages(slot)?;
        let averages = launch_slice(OPERATION, "averages", self.averages.raw())?;
        let volumes = launch_slice(OPERATION, "volumes", self.elem_volume.raw())?;
        let totals = launch_slice_mut(OPERATION, "totals", self.totals.raw_mut())?;
        let elements = i32::try_from(self.num_elements)
            .map_err(|_| invalid(OPERATION, "element count exceeds i32"))?;
        let config = element_launch_config(self.num_elements, OPERATION)?;
        // SAFETY: matches `sonium_dg_totals`; the totals buffer was
        // reset above and `averages` was just produced on this stream.
        unsafe {
            self.pool
                .backend()
                .stream()
                .launch_builder(&self.totals_kernel)
                .arg(averages)
                .arg(volumes)
                .arg(totals)
                .arg(&elements)
                .launch(config)
                .map_err(|error| load_failed(OPERATION, error))?;
        }
        Ok(())
    }

    /// Step-report scalars of a slot in one synchronization: minimum
    /// nodal density, minimum nodal pressure, and conserved totals.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::OperationFailed`] when a launch or the
    /// copy-back fails.
    pub fn report(&mut self, slot: DgSlot) -> BackendResult<(f64, f64, [f64; 5])> {
        self.launch_reset()?;
        self.launch_min_thermo(slot)?;
        self.launch_totals(slot)?;
        self.pool.backend().synchronize()?;
        let mut thermo = [0.0f64; 2];
        self.pool
            .backend()
            .try_copy_d2h(self.min_thermo.raw(), &mut thermo)?;
        let mut totals = [0.0f64; 5];
        self.pool
            .backend()
            .try_copy_d2h(self.totals.raw(), &mut totals)?;
        Ok((thermo[0], thermo[1], totals))
    }
}
