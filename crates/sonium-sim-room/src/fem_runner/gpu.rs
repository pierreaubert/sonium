//! Resident GPU frequency-sweep execution for the FEM room runner.
//!
//! This module owns the CUDA/Metal sweep setup, resident device lifecycle,
//! preconditioner selection and failure reporting behind [`FemRunDiagnostics`].
//! Room-specific scheduling lives in [`super::scheduler`]; shared sweep,
//! preconditioner and operator services belong in FEM/solvers/compute when
//! they mature beyond this room runner.
use super::build::build_source_rhs_basis_for_model;
use super::compute::{
    compute_barycentric_coords, compute_boundary_coefficients, find_containing_element_parallel,
};
use super::fem_mesh_config::{FemMeshConfig, GpuSolutionEvidence, create_room_mesh};
use super::fem_run_diagnostics::{
    FemBackendExecutionError, FemBackendFailureDiagnostics, FemRunDiagnostics,
};
use super::fmt_mod::fmt_duration;
#[cfg(feature = "cuda")]
use super::fmt_mod::fmt_gib;
use super::misc::{
    duration_ms, mesh_aabb, shifted_laplacian_amg_config, should_log_frequency,
    store_listener_pressures,
};
use super::types::{
    FemFrequencyStats, FemTransferStats, SourceRhsBasis, assemble_rhs_from_basis_into,
};
use super::validate::{validate_geometry, validate_mesh_config, validate_source_signal_ownership};
use super::*;
use crate::SimulationOutput;
use anyhow::Result;
use num_complex::{Complex32, Complex64};
use rayon::prelude::*;
#[cfg(feature = "cuda")]
use sonium_compute::CudaBackend;
#[cfg(feature = "metal")]
use sonium_compute::MetalBackend;
use sonium_compute::gmres::{BackendCsr, BackendPreconditioner};
#[cfg(feature = "cuda")]
use sonium_compute::gmres::{
    GmresBackendConfig, GmresBackendStatus, GmresWorkspace,
    gmres_backend_preconditioned_pooled_with_workspace,
};
use sonium_compute::gmres_ir::GmresIrConfig;
#[cfg(feature = "metal")]
use sonium_compute::gmres_ir::gmres_ir_backend_preconditioned;
#[cfg(feature = "cuda")]
use sonium_compute::gmres_ir::{GmresIrWorkspace, gmres_ir_backend_preconditioned_with_workspace};
use sonium_compute::{Backend, BufferElement, MixedPrecisionPolicy};
#[cfg(feature = "cuda")]
use sonium_compute::{BlockGmresConfig, CudaFailureInjectionPoint, DeviceOperator};
#[cfg(feature = "cuda")]
use sonium_compute::{MemoryPool, MemoryPoolConfig};
use sonium_xem::Point3D;

fn ensure_device_not_cancelled(
    cancellation: Option<&FemRunCancellation>,
    backend: &str,
) -> Result<()> {
    if let Some(cancellation) = cancellation
        && cancellation.is_cancelled()
    {
        let reason = cancellation
            .first_error()
            .unwrap_or_else(|| "cancelled cooperatively".into());
        return Err(anyhow::Error::new(FemBackendExecutionError::Cancelled {
            backend: backend.to_string(),
            reason,
        }));
    }
    Ok(())
}

/// Deterministic QA-only cancellation seam used to exercise cleanup after
/// a device frequency has started. The token remains the source of truth
/// for cancellation; this environment variable merely requests that the
/// production loop set it at a known frequency boundary. Keeping the seam
/// after the solve and before diagnostics/output publication ensures the
/// checked error path owns the live device allocations and no partial
/// `SimulationOutput` can escape.
fn inject_mid_frequency_cancellation(
    cancellation: Option<&FemRunCancellation>,
    frequency_index: usize,
) {
    let Some(cancellation) = cancellation else {
        return;
    };
    // Keep the established SR1 seam intact while allowing the breadth
    // lane to select a distinct nonzero warm-state boundary.  The latter
    // is deliberately opt-in and remains a QA-only environment seam.
    let target = std::env::var("SONIUM_SIM_ROOM_SR1_RESIDENT_CANCEL_FREQUENCY")
        .or_else(|_| std::env::var("SONIUM_SIM_ROOM_SR1_CANCEL_FREQUENCY"))
        .ok()
        .and_then(|value| value.parse::<usize>().ok());
    let Some(target) = target else {
        return;
    };
    if target == frequency_index && !cancellation.is_cancelled() {
        cancellation.cancel(format!(
            "mid-frequency cancellation injected at frequency {frequency_index}"
        ));
    }
}

fn accumulate_host_solution_evidence(
    assembler: &HelmholtzAssembler,
    wavenumber: Complex64,
    boundary_coeffs: &HashMap<usize, Complex64>,
    rhs_basis: &SourceRhsBasis,
    source: &Source,
    frequency_hz: f64,
    solution: &[Complex64],
    operator_work: &mut FemOperatorWorkEvidence,
    boundary_power_by_tag: &mut BTreeMap<usize, f64>,
) -> Result<()> {
    let mut rhs = vec![Complex64::new(0.0, 0.0); assembler.num_rows];
    assemble_rhs_from_basis_into(rhs_basis, source, frequency_hz, &mut rhs);
    let sample = fem_operator_work_evidence(assembler, wavenumber, boundary_coeffs, &rhs, solution)
        .map_err(anyhow::Error::msg)?;
    operator_work.source_work += sample.source_work;
    operator_work.volume_damping_work += sample.volume_damping_work;
    operator_work.boundary_work += sample.boundary_work;
    operator_work.balance_residual += sample.balance_residual;

    let flux = fem_boundary_flux_evidence(
        assembler,
        boundary_coeffs,
        frequency_hz,
        super::AIR_DENSITY,
        solution,
    )
    .map_err(anyhow::Error::msg)?;
    for sample in flux.samples {
        *boundary_power_by_tag
            .entry(sample.boundary_tag)
            .or_insert(0.0) += sample.active_power_w;
    }
    Ok(())
}

fn collect_device_solution_evidence<B: Backend>(
    backend: &B,
    solution_device: &<B as Backend>::Buffer<Complex64>,
    assembler: &HelmholtzAssembler,
    wavenumber: Complex64,
    boundary_coeffs: &HashMap<usize, Complex64>,
    rhs_basis: &SourceRhsBasis,
    source: &Source,
    frequency_hz: f64,
    operator_work: &mut FemOperatorWorkEvidence,
    boundary_power_by_tag: &mut BTreeMap<usize, f64>,
) -> Result<()> {
    let mut solution = vec![Complex64::new(0.0, 0.0); assembler.num_rows];
    backend.copy_d2h(solution_device, &mut solution);
    accumulate_host_solution_evidence(
        assembler,
        wavenumber,
        boundary_coeffs,
        rhs_basis,
        source,
        frequency_hz,
        &solution,
        operator_work,
        boundary_power_by_tag,
    )
}

/// Whether the sweep downloads full nodal solutions for host-side power
/// evidence. Listener pressures are evaluated on-device either way, so
/// [`GpuSolutionEvidence::ListenersOnly`] only drops the power ledger.
fn wants_full_solution_evidence(mesh_config: &FemMeshConfig) -> bool {
    mesh_config.gpu_solution_evidence == GpuSolutionEvidence::Full
}

fn finish_device_physical_evidence(
    operator_work: &mut FemOperatorWorkEvidence,
    boundary_power_by_tag: BTreeMap<usize, f64>,
    frequency_hz: f64,
) -> Result<(
    FemOperatorWorkEvidence,
    FemPhysicalPowerEvidence,
    FemBoundaryFluxEvidence,
)> {
    operator_work.validate().map_err(anyhow::Error::msg)?;
    let physical_power = FemPhysicalPowerEvidence::from_operator_work(
        operator_work,
        frequency_hz,
        super::AIR_DENSITY,
    )
    .map_err(anyhow::Error::msg)?;
    let boundary_flux = FemBoundaryFluxEvidence::from_tagged_powers(boundary_power_by_tag)
        .map_err(anyhow::Error::msg)?;
    Ok((*operator_work, physical_power, boundary_flux))
}

/// Refuse a CUDA FEM run when one of its solver primitives would silently
/// stage through host memory. The backend API exposes these capabilities so
/// a requested CUDA run is either genuinely device-resident or fails with
/// an explicit, actionable result.
#[cfg(feature = "cuda")]
fn require_native_cuda_fem_capabilities(
    backend: &CudaBackend,
    precision: GpuPrecision,
) -> Result<()> {
    let capabilities = backend.capabilities();
    let checks = [
        ("Complex32 CSR SpMV", capabilities.csr_spmv.c32),
        ("Complex64 CSR SpMV", capabilities.csr_spmv.c64),
        ("Complex32 BLAS-1", capabilities.blas1.c32),
        ("Complex64 BLAS-1", capabilities.blas1.c64),
        ("Complex32 multi-dot", capabilities.multi_dot.c32),
        ("Complex64 multi-dot", capabilities.multi_dot.c64),
        ("Complex32 axpy-many", capabilities.axpy_many.c32),
        ("Complex64 axpy-many", capabilities.axpy_many.c64),
        ("Complex32 fast reduction", capabilities.reduction.fast.c32),
        ("Complex64 fast reduction", capabilities.reduction.fast.c64),
        (
            "Complex64 listener gather",
            capabilities.gather4_weighted.c64,
        ),
        (
            "Complex64 to Complex32 cast",
            capabilities.precision_cast.c64_to_c32,
        ),
        (
            "Complex32 to Complex64 cast",
            capabilities.precision_cast.c32_to_c64,
        ),
    ];
    if let Some((operation, state)) = checks.into_iter().find(|(_, state)| !state.is_native()) {
        return Err(anyhow::Error::new(
            FemBackendExecutionError::UnsupportedNativePrimitive {
                backend: capabilities.backend_name.to_string(),
                precision: precision.label().to_string(),
                operation: operation.to_string(),
                capability: format!("{state:?}"),
            },
        ));
    }
    Ok(())
}
use sonium_fem::assembly::{
    HelmholtzAssembler, assemble_boundary_mass, assemble_mass, assemble_stiffness,
};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::Mesh;
#[cfg(feature = "cuda")]
use sonium_solvers::{BackendAmgPreconditioner, CsrMatrix};
use sonium_solvers::{ChebyshevBackend, PointJacobiBackend, PolynomialJacobiBackend};
use sonium_xem::{RoomSimulation, Source, SymmetryConfig};
use std::collections::{BTreeMap, HashMap};
use std::time::{Duration, Instant};

use crate::energy_validation::{
    FemBoundaryFluxEvidence, FemOperatorWorkEvidence, FemPhysicalPowerEvidence,
    fem_boundary_flux_evidence, fem_operator_work_evidence,
};

// Selection policy lives in `sonium_fem::backend_selection`; this module keeps
// the device runtime. (r2factor split output: keep this shim layout if the
// facade is re-split.)
pub use sonium_fem::backend_selection::{CHEBYSHEV_EIGENVALUE_RATIO, GpuPrecision, GpuPrecond};

/// Default damping for PolynomialJacobi on shifted-Laplacian spectra.
/// Conservative ω = 0.5 keeps the Neumann series stable for a wide
/// range of `k`. Values closer to 1.0 converge faster when the
/// effective spectrum is well-clustered but risk divergence near
/// resonant frequencies.
const POLY_JACOBI_DAMPING: f32 = 0.5;

/// Chebyshev degree used when an AMG setup failure forces a sweep-wide
/// fallback to the polynomial smoother. Matches the Metal `auto`
/// resolution and the CLI `chebyshev` default.
const GPU_AMG_SETUP_FALLBACK_CHEBYSHEV_DEGREE: usize = 8;

/// CUDA IR defaults. These are intentionally larger than the original
/// smoke-test budget: room Helmholtz modes around 250-350 Hz can need more
/// than 30 outer steps even when the residual is already near 1e-6.
/// 100 (was 60): at high wavenumbers the outer FGMRES makes slow but steady
/// progress (e.g. 1600 Hz reaches 1.4e-6 after 60 outers against the 1e-6
/// gate) because every f32 inner solve runs to its cap. Cutting the budget
/// at 60 turns a near-miss into an abort with zero data plus a full-cost
/// warm-start retry; the extra headroom lets such frequencies converge.
pub(super) const GPU_IR_MAX_OUTER_ITERATIONS: usize = 100;
// 160 (was 80): at high wavenumbers the f32 inner solve cannot reach its
// 1e-3 tolerance in 80 steps even with a good AMG hierarchy, so every outer
// step got a maxed-out, low-quality correction and the outer loop stalled.
// A deeper inner budget lets the correction actually converge, which cuts
// outer steps more than the extra inner work costs.
pub(super) const GPU_IR_MAX_INNER_ITERATIONS: usize = 160;
const GPU_IR_RESTART_SIZE: usize = 30;
const GPU_IR_OUTER_TOLERANCE: f64 = 1.0e-6;
const GPU_IR_INNER_TOLERANCE: f32 = 1.0e-3;
const GPU_IR_TRUE_RESIDUAL_CHECK_PERIOD: usize = 5;

/// Tuned shifted-Laplacian AMG setup for the GPU hierarchy.
///
/// Must match the CPU setup ([`super::misc::shifted_laplacian_amg_config`]:
/// parallel aggregation, Chebyshev smoother, strong threshold 0.5). The
/// generic aggregation defaults (threshold 0.08) coarsen the
/// high-wavenumber shifted operator poorly: the f32 inner solve then
/// maxes out its budget every outer step and iterative refinement stalls
/// around residual 3e-2 at 700 Hz instead of converging to 1e-6.
#[cfg(feature = "cuda")]
pub(super) fn gpu_amg_config() -> sonium_solvers::AmgConfig {
    shifted_laplacian_amg_config()
}

/// Pure-f32 defaults. The solver estimate is still driven to 1e-6, then
/// final acceptance recomputes the residual on-device after promoting the
/// f32 system to f64 arithmetic. That avoids downloading the full solution
/// while still checking an explicit `||Ax-b|| / ||b||`.
#[cfg(feature = "cuda")]
const GPU_F32_MAX_ITERATIONS: usize = 300;
#[cfg(feature = "cuda")]
const GPU_F32_TOLERANCE: f32 = 1.0e-6;
#[cfg(feature = "cuda")]
// Certification must meet the requested solve accuracy. A looser promoted
// gate can accept a drifted f32 recurrence without running residual correction.
const GPU_F32_PROMOTED_RESIDUAL_TOLERANCE: f64 = GPU_F32_TOLERANCE as f64;
#[cfg(feature = "cuda")]
const GPU_F32_RESIDUAL_CORRECTION_STEPS: usize = 6;
#[cfg(feature = "cuda")]
const GPU_F32_BLOCK_MAX_ITERATIONS: usize = 60;

fn usize_indices_to_i32(values: &[usize], label: &str) -> Result<Vec<i32>> {
    values
        .iter()
        .enumerate()
        .map(|(idx, &value)| {
            i32::try_from(value).map_err(|_| {
                anyhow::anyhow!("{label}[{idx}]={value} exceeds GPU CSR i32 index range")
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// GPU allocation helpers — return Result instead of panicking
// ---------------------------------------------------------------------------

fn gpu_alloc<B: Backend, T: BufferElement>(
    backend: &B,
    n: usize,
    label: &str,
) -> Result<<B as Backend>::Buffer<T>> {
    backend
        .raw_alloc::<T>(n)
        .map_err(|e| anyhow::anyhow!("GPU allocation failed for {label} ({n} elements): {e}"))
}

fn gpu_alloc_and_upload<B: Backend, T: BufferElement + Copy>(
    backend: &B,
    host: &[T],
    label: &str,
) -> Result<<B as Backend>::Buffer<T>> {
    let mut dev = gpu_alloc(backend, host.len(), label)?;
    backend.copy_h2d(host, &mut dev);
    Ok(dev)
}

#[cfg(feature = "cuda")]
fn promoted_f32_residual_on_device(
    backend: &CudaBackend,
    a_promoted: &dyn DeviceOperator<CudaBackend, Complex64>,
    x_32: &<CudaBackend as Backend>::Buffer<Complex32>,
    b_32: &<CudaBackend as Backend>::Buffer<Complex32>,
    x_64: &mut <CudaBackend as Backend>::Buffer<Complex64>,
    b_64: &mut <CudaBackend as Backend>::Buffer<Complex64>,
    residual_64: &mut <CudaBackend as Backend>::Buffer<Complex64>,
) -> sonium_compute::BackendResult<f64> {
    backend.try_cast_from_lower::<Complex64>(x_32, x_64)?;
    backend.try_cast_from_lower::<Complex64>(b_32, b_64)?;
    a_promoted.try_apply(backend, x_64, residual_64)?;
    backend.try_axpy(Complex64::new(-1.0, 0.0), b_64, residual_64)?;

    let residual_norm = backend
        .try_nrm2_with_mode::<Complex64>(sonium_compute::ReductionMode::Fast, residual_64)?;
    let rhs_norm =
        backend.try_nrm2_with_mode::<Complex64>(sonium_compute::ReductionMode::Fast, b_64)?;
    Ok(if rhs_norm == 0.0 {
        if residual_norm == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        residual_norm / rhs_norm
    })
}

#[cfg(feature = "cuda")]
struct F32ResidualCorrectionScratch<'a> {
    residual32: &'a mut <CudaBackend as Backend>::Buffer<Complex32>,
    correction32: &'a mut <CudaBackend as Backend>::Buffer<Complex32>,
    x64: &'a mut <CudaBackend as Backend>::Buffer<Complex64>,
    rhs64: &'a mut <CudaBackend as Backend>::Buffer<Complex64>,
    residual64: &'a mut <CudaBackend as Backend>::Buffer<Complex64>,
}

#[cfg(feature = "cuda")]
struct F32ResidualCorrectionProblem<'a> {
    pool: &'a MemoryPool<CudaBackend>,
    a32: &'a dyn DeviceOperator<CudaBackend, Complex32>,
    a64: &'a dyn DeviceOperator<CudaBackend, Complex64>,
    precond: &'a dyn BackendPreconditioner<CudaBackend, Complex32>,
    cfg: &'a GmresBackendConfig<Complex32>,
}

#[cfg(feature = "cuda")]
fn solve_f32_with_residual_correction(
    problem: &F32ResidualCorrectionProblem<'_>,
    rhs32: &<CudaBackend as Backend>::Buffer<Complex32>,
    x0_32: Option<&<CudaBackend as Backend>::Buffer<Complex32>>,
    x32: &mut <CudaBackend as Backend>::Buffer<Complex32>,
    workspace: &mut GmresWorkspace<CudaBackend, Complex32>,
    scratch: &mut F32ResidualCorrectionScratch<'_>,
) -> Result<(GmresBackendStatus<Complex32>, f64), String> {
    let backend = problem.pool.backend();
    let mut status = gmres_backend_preconditioned_pooled_with_workspace::<
        CudaBackend,
        dyn DeviceOperator<CudaBackend, Complex32>,
        dyn BackendPreconditioner<CudaBackend, Complex32>,
        Complex32,
    >(
        problem.pool,
        workspace,
        problem.a32,
        problem.precond,
        rhs32,
        x0_32,
        x32,
        problem.cfg,
    )
    .map_err(|e| e.to_string())?;
    let mut total_iterations = status.iterations;

    let mut promoted_residual = promoted_f32_residual_on_device(
        backend,
        problem.a64,
        x32,
        rhs32,
        scratch.x64,
        scratch.rhs64,
        scratch.residual64,
    )
    .map_err(|error| error.to_string())?;

    for correction_step in 0..GPU_F32_RESIDUAL_CORRECTION_STEPS {
        if promoted_residual <= GPU_F32_PROMOTED_RESIDUAL_TOLERANCE {
            break;
        }

        backend.scale::<Complex64>(Complex64::new(-1.0, 0.0), scratch.residual64);
        backend.cast_to_lower::<Complex64>(scratch.residual64, scratch.residual32);

        let correction_status = gmres_backend_preconditioned_pooled_with_workspace::<
            CudaBackend,
            dyn DeviceOperator<CudaBackend, Complex32>,
            dyn BackendPreconditioner<CudaBackend, Complex32>,
            Complex32,
        >(
            problem.pool,
            workspace,
            problem.a32,
            problem.precond,
            scratch.residual32,
            None,
            scratch.correction32,
            problem.cfg,
        )
        .map_err(|e| e.to_string())?;
        total_iterations += correction_status.iterations;
        backend.axpy::<Complex32>(Complex32::new(1.0, 0.0), scratch.correction32, x32);

        let corrected_residual = promoted_f32_residual_on_device(
            backend,
            problem.a64,
            x32,
            rhs32,
            scratch.x64,
            scratch.rhs64,
            scratch.residual64,
        )
        .map_err(|error| error.to_string())?;
        log::debug!(
            "FEM (cuda f32-pure): residual correction {} estimated_res={:.2e} promoted_res {:.2e} -> {:.2e}",
            correction_step + 1,
            correction_status.final_residual,
            promoted_residual,
            corrected_residual
        );
        promoted_residual = corrected_residual;
        status = correction_status;
    }

    status.iterations = total_iterations;
    status.final_residual = promoted_residual as f32;
    status.converged = promoted_residual <= GPU_F32_PROMOTED_RESIDUAL_TOLERANCE;

    Ok((status, promoted_residual))
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn solve_f32_sources_block(
    backend: &CudaBackend,
    workspace: &mut sonium_compute::BlockGmresWorkspace<CudaBackend, Complex32>,
    a_inner: &dyn DeviceOperator<CudaBackend, Complex32>,
    a_outer: &dyn DeviceOperator<CudaBackend, Complex64>,
    precond: &dyn BackendPreconditioner<CudaBackend, Complex32>,
    rhs_assembler: &mut DeviceRhsBlockAssembler<CudaBackend, Complex32>,
    coefficients: &[Complex32],
    rhs_32: &mut <CudaBackend as Backend>::Buffer<Complex32>,
    x_32: &mut <CudaBackend as Backend>::Buffer<Complex32>,
    correction_rhs: &mut <CudaBackend as Backend>::Buffer<Complex32>,
    correction: &mut <CudaBackend as Backend>::Buffer<Complex32>,
    rhs_64: &mut <CudaBackend as Backend>::Buffer<Complex64>,
    x_64: &mut <CudaBackend as Backend>::Buffer<Complex64>,
    residual_64: &mut <CudaBackend as Backend>::Buffer<Complex64>,
    norm_column: &mut <CudaBackend as Backend>::Buffer<Complex64>,
    listener_32: &mut DeviceListenerExtraction<CudaBackend, Complex32>,
    listener_64: &mut DeviceListenerExtraction<CudaBackend, Complex64>,
    rhs_count: usize,
    policy: MixedPrecisionPolicy,
) -> Result<(Vec<Vec<Complex64>>, Vec<GmresBackendStatus<Complex32>>, f64, usize), String> {
    rhs_assembler.assemble(backend, coefficients, rhs_32).map_err(|error| error.to_string())?;
    let mut status = sonium_compute::gmres_backend_block_preconditioned_with_workspace(
        backend,
        a_inner,
        precond,
        rhs_32,
        None,
        x_32,
        rhs_count,
        &BlockGmresConfig {
            max_block_iterations: GPU_F32_BLOCK_MAX_ITERATIONS,
            tolerance: GPU_F32_TOLERANCE,
        },
        workspace,
    )
    .map_err(|error| error.to_string())?;
    // The f32 Arnoldi recurrence may stagnate just above the requested
    // tolerance. Certify and correct the rounded system with promoted residuals
    // rather than accepting a loose inner status or discarding its iterate.
    backend.try_cast_from_lower::<Complex64>(rhs_32, rhs_64)
        .map_err(|error| error.to_string())?;
    let mut correction_steps = 0;
    let promoted_residuals = loop {
        backend.try_cast_from_lower::<Complex64>(x_32, x_64)
            .map_err(|error| error.to_string())?;
        a_outer.try_apply_colmajor(backend, rhs_count, x_64, residual_64)
            .map_err(|error| error.to_string())?;
        backend.try_axpy::<Complex64>(Complex64::new(-1.0, 0.0), rhs_64, residual_64)
            .map_err(|error| error.to_string())?;
        let residuals = super::device::block_relative_residuals(
            backend, rhs_64, residual_64, norm_column,
        ).map_err(|error| error.to_string())?;
        if residuals.iter().all(|&r| r <= GPU_F32_PROMOTED_RESIDUAL_TOLERANCE) {
            break residuals;
        }
        if correction_steps == GPU_F32_RESIDUAL_CORRECTION_STEPS {
            return Err(format!("block residual corrections exhausted: {residuals:?}"));
        }
        // residual_64 contains A*x-b, so subtract the correction below.
        backend.try_cast_to_lower::<Complex64>(residual_64, correction_rhs)
            .map_err(|error| error.to_string())?;
        let update = sonium_compute::gmres_backend_block_preconditioned_with_workspace(
            backend, a_inner, precond, correction_rhs, None, correction, rhs_count,
            &BlockGmresConfig {
                max_block_iterations: GPU_F32_BLOCK_MAX_ITERATIONS,
                tolerance: GPU_F32_TOLERANCE,
            }, workspace,
        ).map_err(|error| error.to_string())?;
        backend.try_axpy::<Complex32>(Complex32::new(-1.0, 0.0), correction, x_32)
            .map_err(|error| error.to_string())?;
        for (total, current) in status.rhs.iter_mut().zip(&update.rhs) {
            total.iterations += current.iterations;
        }
        correction_steps += 1;
    };
    for (column, &residual) in status.rhs.iter_mut().zip(&promoted_residuals) {
        column.final_residual = residual as f32;
        column.backward_error = residual as f32;
        column.converged = true;
    }
    let n = a_inner.n();

    let listener_values_32 = listener_32.evaluate_colmajor_to_host(backend, x_32, n, rhs_count);
    let listener_values_64 = listener_64.evaluate_colmajor_to_host(backend, x_64, n, rhs_count);
    let mut accepted_values = Vec::with_capacity(rhs_count);
    let mut max_residual = 0.0_f64;
    for column in 0..rhs_count {
        let low = &listener_values_32[column];
        let reference = &listener_values_64[column];
        let physical_error = low
            .iter()
            .zip(reference)
            .map(|(observed, expected)| {
                let observed = Complex64::new(observed.re as f64, observed.im as f64);
                (observed - *expected).norm() / expected.norm().max(1.0e-30)
            })
            .fold(0.0_f64, f64::max);
        if policy
            .validate_gates(promoted_residuals[column], physical_error)
            .is_err()
        {
            return Err(format!("block certification rejected column {column}: residual={}, physical_error={physical_error}", promoted_residuals[column]));
        }
        max_residual = max_residual.max(promoted_residuals[column]);
        accepted_values.push(
            low.iter()
                .map(|value| Complex64::new(value.re as f64, value.im as f64))
                .collect(),
        );
    }

    Ok((accepted_values, status.rhs, max_residual, correction_steps))
}

fn upload_real_basis_as_complex32<B: Backend>(
    backend: &B,
    values: &[f64],
    label: &str,
) -> Result<<B as Backend>::Buffer<Complex32>> {
    let host: Vec<Complex32> = values
        .iter()
        .map(|&v| Complex32::new(v as f32, 0.0))
        .collect();
    gpu_alloc_and_upload(backend, &host, &format!("{label} Complex32 basis"))
}

fn upload_real_basis_as_complex64<B: Backend>(
    backend: &B,
    values: &[f64],
    label: &str,
) -> Result<<B as Backend>::Buffer<Complex64>> {
    let host: Vec<Complex64> = values.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    gpu_alloc_and_upload(backend, &host, &format!("{label} Complex64 basis"))
}

fn upload_boundary_basis_as_complex32<B: Backend>(
    backend: &B,
    boundaries: &std::collections::HashMap<usize, Vec<f64>>,
) -> Result<Vec<(usize, <B as Backend>::Buffer<Complex32>)>> {
    boundaries
        .iter()
        .map(|(&tag, values)| {
            Ok((
                tag,
                upload_real_basis_as_complex32(backend, values, "boundary")?,
            ))
        })
        .collect()
}

fn upload_boundary_basis_as_complex64<B: Backend>(
    backend: &B,
    boundaries: &std::collections::HashMap<usize, Vec<f64>>,
) -> Result<Vec<(usize, <B as Backend>::Buffer<Complex64>)>> {
    boundaries
        .iter()
        .map(|(&tag, values)| {
            Ok((
                tag,
                upload_real_basis_as_complex64(backend, values, "boundary")?,
            ))
        })
        .collect()
}

fn build_diag_basis_32(
    row_ptrs: &[usize],
    col_indices: &[usize],
    k_values: &[f64],
    m_values: &[f64],
    n: usize,
) -> Vec<(f32, f32)> {
    let mut diag = vec![(0.0_f32, 0.0_f32); n];
    for row in 0..n {
        for idx in row_ptrs[row]..row_ptrs[row + 1] {
            if col_indices[idx] == row {
                diag[row] = (k_values[idx] as f32, m_values[idx] as f32);
                break;
            }
        }
    }
    diag
}

fn fill_shifted_diag_32(out: &mut [Complex32], diag_basis: &[(f32, f32)], p_shift: Complex64) {
    let p_shift = Complex32::new(p_shift.re as f32, p_shift.im as f32);
    for (dst, &(kv, mv)) in out.iter_mut().zip(diag_basis.iter()) {
        *dst = Complex32::new(kv, 0.0) + p_shift * Complex32::new(mv, 0.0);
    }
}

fn fill_shifted_values_32(
    out: &mut [Complex32],
    k_values: &[f64],
    m_values: &[f64],
    p_shift: Complex64,
) {
    let p_shift = Complex32::new(p_shift.re as f32, p_shift.im as f32);
    for ((dst, &kv), &mv) in out.iter_mut().zip(k_values.iter()).zip(m_values.iter()) {
        *dst = Complex32::new(kv as f32, 0.0) + p_shift * Complex32::new(mv as f32, 0.0);
    }
}

#[cfg(feature = "metal")]
fn update_helmholtz_values_32<B: Backend>(
    backend: &B,
    dst: &mut <B as Backend>::Buffer<Complex32>,
    k_basis: &<B as Backend>::Buffer<Complex32>,
    m_basis: &<B as Backend>::Buffer<Complex32>,
    boundary_basis: &[(usize, <B as Backend>::Buffer<Complex32>)],
    k_sq: Complex64,
    boundary_coeffs: &std::collections::HashMap<usize, Complex64>,
) {
    backend.copy_d2d(k_basis, dst);
    backend.axpy(
        Complex32::new(-k_sq.re as f32, -k_sq.im as f32),
        m_basis,
        dst,
    );
    for (tag, values) in boundary_basis {
        if let Some(coeff) = boundary_coeffs.get(tag) {
            backend.axpy(
                Complex32::new(coeff.re as f32, coeff.im as f32),
                values,
                dst,
            );
        }
    }
}

fn update_shifted_values_32<B: Backend>(
    backend: &B,
    dst: &mut <B as Backend>::Buffer<Complex32>,
    k_basis: &<B as Backend>::Buffer<Complex32>,
    m_basis: &<B as Backend>::Buffer<Complex32>,
    p_shift: Complex64,
) {
    backend.copy_d2d(k_basis, dst);
    backend.axpy(
        Complex32::new(p_shift.re as f32, p_shift.im as f32),
        m_basis,
        dst,
    );
}

#[cfg(feature = "metal")]
fn update_helmholtz_values_64<B: Backend>(
    backend: &B,
    dst: &mut <B as Backend>::Buffer<Complex64>,
    k_basis: &<B as Backend>::Buffer<Complex64>,
    m_basis: &<B as Backend>::Buffer<Complex64>,
    boundary_basis: &[(usize, <B as Backend>::Buffer<Complex64>)],
    k_sq: Complex64,
    boundary_coeffs: &std::collections::HashMap<usize, Complex64>,
) {
    backend.copy_d2d(k_basis, dst);
    backend.axpy(-k_sq, m_basis, dst);
    for (tag, values) in boundary_basis {
        if let Some(&coeff) = boundary_coeffs.get(tag) {
            backend.axpy(coeff, values, dst);
        }
    }
}

fn build_listener_interpolation(
    mesh: &Mesh,
    points: &[Point3D],
    containing_elements: &[Option<usize>],
) -> Result<(Vec<i32>, Vec<f64>)> {
    anyhow::ensure!(
        points.len() == containing_elements.len(),
        "listener interpolation input length mismatch"
    );

    let mut indices = Vec::with_capacity(points.len() * 4);
    let mut weights = Vec::with_capacity(points.len() * 4);

    for (&point, &elem_opt) in points.iter().zip(containing_elements.iter()) {
        if let Some(elem_idx) = elem_opt {
            let elem = &mesh.elements[elem_idx];
            let vertices = elem.vertices();
            if vertices.len() == 4 {
                let v0 = &mesh.nodes[vertices[0]];
                let v1 = &mesh.nodes[vertices[1]];
                let v2 = &mesh.nodes[vertices[2]];
                let v3 = &mesh.nodes[vertices[3]];
                if let Some(bary) = compute_barycentric_coords(point, v0, v1, v2, v3) {
                    for (&vertex, &weight) in vertices.iter().zip(bary.iter()) {
                        indices.push(i32::try_from(vertex).map_err(|_| {
                            anyhow::anyhow!(
                                "listener interpolation vertex index {vertex} exceeds i32"
                            )
                        })?);
                        weights.push(weight);
                    }
                    continue;
                }
            }
        }

        let mut min_dist = f64::MAX;
        let mut nearest_node = 0usize;
        for (node_idx, node) in mesh.nodes.iter().enumerate() {
            let dist = (node.x - point.x).powi(2)
                + (node.y - point.y).powi(2)
                + (node.z - point.z).powi(2);
            if dist < min_dist {
                min_dist = dist;
                nearest_node = node_idx;
            }
        }
        let nearest_i32 = i32::try_from(nearest_node).map_err(|_| {
            anyhow::anyhow!("nearest listener node index {nearest_node} exceeds i32")
        })?;
        indices.extend([nearest_i32; 4]);
        weights.extend([1.0, 0.0, 0.0, 0.0]);
    }

    Ok((indices, weights))
}

/// Run the FEM Helmholtz sweep on a CUDA backend. The shared CSR structure
/// and source RHS/solution work buffers are allocated once; each frequency
/// updates only A/P values and builds one preconditioner that is reused for
/// all sources. Pure-f32 sweeps use a retained block workspace; other
/// precisions use the retained scalar solve and retry path.
#[cfg(feature = "cuda")]
pub fn run_fem_gpu(
    simulation: &RoomSimulation,
    mesh_config: &FemMeshConfig,
    precision: GpuPrecision,
    precond: GpuPrecond,
    cuda_device: usize,
) -> Result<SimulationOutput> {
    let (output, _diagnostics) =
        run_fem_gpu_report(simulation, mesh_config, precision, precond, cuda_device)?;
    Ok(output)
}

/// Run the CUDA FEM sweep and retain per-frequency solver diagnostics.
#[cfg(feature = "cuda")]
pub fn run_fem_gpu_report(
    simulation: &RoomSimulation,
    mesh_config: &FemMeshConfig,
    precision: GpuPrecision,
    precond: GpuPrecond,
    cuda_device: usize,
) -> Result<(SimulationOutput, FemRunDiagnostics)> {
    run_fem_gpu_report_with_cancellation(
        simulation,
        mesh_config,
        precision,
        precond,
        cuda_device,
        None,
    )
}

/// Run CUDA FEM with cooperative frequency-boundary cancellation.
///
/// Device kernels remain atomic from the caller's perspective: a request
/// is observed before the next frequency starts or at the checked
/// post-solve boundary, and no partial response is returned.
#[cfg(feature = "cuda")]
pub fn run_fem_gpu_report_cancellation(
    simulation: &RoomSimulation,
    mesh_config: &FemMeshConfig,
    precision: GpuPrecision,
    precond: GpuPrecond,
    cuda_device: usize,
    cancellation: &FemRunCancellation,
) -> Result<(SimulationOutput, FemRunDiagnostics)> {
    run_fem_gpu_report_with_cancellation(
        simulation,
        mesh_config,
        precision,
        precond,
        cuda_device,
        Some(cancellation),
    )
}

/// Run the CUDA FEM sweep with one checked native failure armed on the
/// backend that owns the run's device context.
///
/// This is an acceptance/driver-shim seam: normal callers should use
/// [`run_fem_gpu_report`]. The injected error is returned through the same
/// structured diagnostic boundary as a real CUDA launch or device-loss
/// error, and all local device buffers are dropped before the error is
/// observed by the caller.
#[cfg(feature = "cuda")]
pub fn run_fem_gpu_report_with_failure_injection(
    simulation: &RoomSimulation,
    mesh_config: &FemMeshConfig,
    precision: GpuPrecision,
    precond: GpuPrecond,
    cuda_device: usize,
    failure: CudaFailureInjectionPoint,
) -> Result<(SimulationOutput, FemRunDiagnostics)> {
    run_fem_gpu_report_with_cancellation_inner(
        simulation,
        mesh_config,
        precision,
        precond,
        cuda_device,
        None,
        Some(failure),
        false,
    )
    .map_err(FemBackendFailureDiagnostics::attach)
}

/// Run the CUDA FEM sweep with a deterministic solver non-convergence
/// injection at the first source solve.  This is a room-runner acceptance
/// seam, not a physical failure model: it exercises the same publication
/// and unwind boundary as a real exhausted Krylov solve without relying
/// on a particular matrix becoming ill-conditioned on a given driver.
#[cfg(feature = "cuda")]
pub fn run_fem_gpu_report_with_solver_nonconvergence_injection(
    simulation: &RoomSimulation,
    mesh_config: &FemMeshConfig,
    precision: GpuPrecision,
    precond: GpuPrecond,
    cuda_device: usize,
) -> Result<(SimulationOutput, FemRunDiagnostics)> {
    run_fem_gpu_report_with_cancellation_inner(
        simulation,
        mesh_config,
        precision,
        precond,
        cuda_device,
        None,
        None,
        true,
    )
    .map_err(FemBackendFailureDiagnostics::attach)
}

#[cfg(feature = "cuda")]
fn run_fem_gpu_report_with_cancellation(
    simulation: &RoomSimulation,
    mesh_config: &FemMeshConfig,
    precision: GpuPrecision,
    precond: GpuPrecond,
    cuda_device: usize,
    cancellation: Option<&FemRunCancellation>,
) -> Result<(SimulationOutput, FemRunDiagnostics)> {
    run_fem_gpu_report_with_cancellation_inner(
        simulation,
        mesh_config,
        precision,
        precond,
        cuda_device,
        cancellation,
        None,
        false,
    )
    .map_err(FemBackendFailureDiagnostics::attach)
}

/// Implementation kept separate from the public error boundary so every
/// `?` in the production CUDA setup/solve path receives one structured
/// diagnostic on the way out.  The local backend and device buffers are
/// then dropped by normal Rust unwinding before the caller can observe the
/// error; no partially filled `SimulationOutput` is ever returned.
#[allow(clippy::too_many_arguments)]
fn run_fem_gpu_report_with_cancellation_inner(
    simulation: &RoomSimulation,
    mesh_config: &FemMeshConfig,
    precision: GpuPrecision,
    precond: GpuPrecond,
    cuda_device: usize,
    cancellation: Option<&FemRunCancellation>,
    failure_injection: Option<CudaFailureInjectionPoint>,
    solver_nonconvergence_injection: bool,
) -> Result<(SimulationOutput, FemRunDiagnostics)> {
    validate_mesh_config(mesh_config)?;
    validate_geometry(simulation)?;
    validate_source_signal_ownership(simulation, mesh_config)?;
    ensure_device_not_cancelled(cancellation, "cuda")?;
    let mut diagnostics = FemRunDiagnostics::new_with_preconditioner(
        "cuda",
        precond.label().to_string(),
        mesh_config.shifted_laplacian_shift,
        mesh_config.modal_damping_ratio,
    );
    anyhow::ensure!(
        !simulation.frequencies.is_empty(),
        "FEM solve requires at least one frequency"
    );

    let max_freq = simulation
        .frequencies
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let min_wavelength = simulation.speed_of_sound / max_freq;
    let elements_per_meter = (mesh_config.elements_per_wavelength / min_wavelength).ceil() as usize;

    log::info!(
        "FEM (cuda): building mesh for max_freq={:.0} Hz, grading {:.1}x",
        max_freq,
        mesh_config.grading_factor,
    );
    let mesh_start = Instant::now();
    // Force the non-symmetric path: GPU runner doesn't (yet) handle the
    // per-mode parity decomposition.
    let symmetry = SymmetryConfig::none();
    let mesh = create_room_mesh(simulation, elements_per_meter, &symmetry, mesh_config);
    log::info!(
        "FEM (cuda) mesh: {} nodes, {} elements (fine spacing {} elem/m, grading {:.1}x, radius {:.2} m) built in {}",
        mesh.num_nodes(),
        mesh.num_elements(),
        elements_per_meter,
        mesh_config.grading_factor,
        mesh_config.fine_radius,
        fmt_duration(mesh_start.elapsed()),
    );
    let image_walls_aabb = mesh_aabb(&mesh);

    let assembly_start = Instant::now();
    let stiffness = assemble_stiffness(&mesh, PolynomialDegree::P1);
    let mass = assemble_mass(&mesh, PolynomialDegree::P1);
    let phys_markers = [
        MARKER_FLOOR,
        MARKER_CEILING,
        MARKER_FRONT,
        MARKER_BACK,
        MARKER_LEFT,
        MARKER_RIGHT,
    ];
    let mut boundary_matrices = Vec::new();
    for &marker in &phys_markers {
        let b_mass = assemble_boundary_mass(&mesh, PolynomialDegree::P1, marker);
        if b_mass.nnz() > 0 {
            boundary_matrices.push((marker as usize, b_mass));
        }
    }
    let assembler = HelmholtzAssembler::from_matrices(&stiffness, &mass, &boundary_matrices);
    log::info!(
        "FEM (cuda): assembled K nnz={}, M nnz={}, {} boundary matrices in {}",
        stiffness.nnz(),
        mass.nnz(),
        boundary_matrices.len(),
        fmt_duration(assembly_start.elapsed()),
    );

    let resolved_boundaries = resolve_boundaries(simulation)?;

    let lp_elements: Vec<Option<usize>> = simulation
        .listening_positions
        .par_iter()
        .map(|lp| find_containing_element_parallel(&mesh, *lp))
        .collect();

    let backend = CudaBackend::new(cuda_device).map_err(|e| {
        anyhow::Error::new(FemBackendExecutionError::DeviceUnavailable {
            backend: "cuda".to_string(),
            detail: format!("device {cuda_device}: {e}"),
        })
    })?;
    if let Some(failure) = failure_injection {
        backend.inject_failure(failure);
    }
    require_native_cuda_fem_capabilities(&backend, precision)?;
    diagnostics.set_execution_profile("cuda", "cuda", precision.label(), "none");
    log::info!("FEM (cuda): CUDA device {cuda_device} acquired");
    let (gpu_free_bytes, gpu_total_bytes) = backend.context().mem_get_info().map_err(|e| {
        anyhow::Error::new(FemBackendExecutionError::DeviceUnavailable {
            backend: "cuda".to_string(),
            detail: format!("device {cuda_device} memory query: {e:?}"),
        })
    })?;
    let gpu_free_bytes = gpu_free_bytes as u64;
    let gpu_total_bytes = gpu_total_bytes as u64;
    let gpu_budget_bytes = (gpu_free_bytes as f64 * 0.8) as u64;
    // The pool owns the existing backend; no second CUDA context is created.
    // This quota covers retained Krylov scratch. Existing matrix and physical
    // output allocations still use the sweep's aggregate memory admission.
    let krylov_pool = MemoryPool::new(
        backend,
        MemoryPoolConfig::with_cap(usize::try_from(gpu_budget_bytes)?),
    )?;
    let backend = krylov_pool.backend();
    let mut ir_workspace = GmresIrWorkspace::<_, Complex64>::new(&krylov_pool);

    log::info!(
        "FEM (cuda): CUDA device {cuda_device} memory: free {}, total {}, conservative budget {}",
        fmt_gib(gpu_free_bytes),
        fmt_gib(gpu_total_bytes),
        fmt_gib(gpu_budget_bytes),
    );

    let n_sources = simulation.sources.len();
    let n_lps = simulation.listening_positions.len();
    let n_freqs = simulation.frequencies.len();
    let n_nodes = mesh.num_nodes();
    // Reused across sources, frequencies, and promoted-residual corrections.
    let mut f32_krylov_workspace = GmresWorkspace::new(n_nodes);
    let mut f32_block_workspace = sonium_compute::BlockGmresWorkspace::with_fallback_pool(n_nodes, &krylov_pool);

    let mut pressures: Vec<Vec<Vec<Complex64>>> =
        vec![vec![vec![Complex64::new(0.0, 0.0); n_freqs]; n_lps]; n_sources];

    let (lp_interp_indices, lp_interp_weights_64) =
        build_listener_interpolation(&mesh, &simulation.listening_positions, &lp_elements)?;
    let listener_nodes: Vec<[usize; 4]> = lp_interp_indices
        .chunks_exact(4)
        .map(|nodes| {
            [
                nodes[0] as usize,
                nodes[1] as usize,
                nodes[2] as usize,
                nodes[3] as usize,
            ]
        })
        .collect();
    let listener_weights_64: Vec<[f64; 4]> = lp_interp_weights_64
        .chunks_exact(4)
        .map(|weights| [weights[0], weights[1], weights[2], weights[3]])
        .collect();
    let listener_weights_32: Vec<[f32; 4]> = listener_weights_64
        .iter()
        .map(|weights| weights.map(|weight| weight as f32))
        .collect();
    let mut listener_32 = match precision {
        GpuPrecision::F32Pure => Some(DeviceListenerExtraction::<CudaBackend, Complex32>::new(
            backend,
            &listener_nodes,
            &listener_weights_32,
        )?),
        GpuPrecision::F64Outer => None,
    };
    // Keep one f64 listener extraction path even for f32 solves. It is
    // the physical-observable gate for the promoted f32 result.
    let mut listener_64 = Some(DeviceListenerExtraction::<CudaBackend, Complex64>::new(
        backend,
        &listener_nodes,
        &listener_weights_64,
    )?);
    let mut lp_values_host = vec![Complex64::new(0.0, 0.0); n_lps];

    // GMRES-IR config — outer Complex64 to keep accuracy at f64 level,
    // inner Complex32 to halve memory traffic per SpMV. Adaptive inner
    // tolerance loosens early iterations and tightens as we converge.
    //
    // Important: the inner GMRES has no restart — it allocates one
    // Complex32 device vector per iteration and accumulates them in the
    // Krylov basis. With max_inner_iterations = K, peak GPU memory for
    // the inner basis is K × n × 8 bytes (Complex32). We cap K
    // conservatively because (a) PointJacobi on indefinite Helmholtz
    // won't drive a deep Krylov to 1e-3 anyway, and (b) the OUTER FGMRES
    // refinement is what actually converges this problem — give the
    // inner just enough room to be a useful preconditioner.
    let ir_policy = MixedPrecisionPolicy {
        outer_tolerance: GPU_IR_OUTER_TOLERANCE,
        inner_tolerance: GPU_IR_INNER_TOLERANCE,
        true_residual_check_period: GPU_IR_TRUE_RESIDUAL_CHECK_PERIOD,
        physical_tolerance: GPU_F32_PROMOTED_RESIDUAL_TOLERANCE,
    };
    let ir_cfg = GmresIrConfig::<Complex64> {
        max_outer_iterations: GPU_IR_MAX_OUTER_ITERATIONS,
        outer_tolerance: ir_policy.outer_tolerance,
        max_inner_iterations: GPU_IR_MAX_INNER_ITERATIONS,
        inner_tolerance: ir_policy.inner_tolerance,
        restart_size: Some(GPU_IR_RESTART_SIZE),
        adaptive_inner_tolerance: true,
        true_residual_check_period: Some(ir_policy.true_residual_check_period),
    };

    // Plain Complex32 GMRES config for F32Pure mode. Same role as the
    // outer FGMRES in IR, but at f32 throughout. Tolerance is loose
    // relative to the f32 epsilon (~6e-8): 1e-6 is comfortably above
    // round-off and well below the FEM mesh discretisation error.
    let f32_cfg = GmresBackendConfig::<Complex32> {
        max_iterations: GPU_F32_MAX_ITERATIONS,
        tolerance: GPU_F32_TOLERANCE,
    };
    let f32_policy = MixedPrecisionPolicy {
        outer_tolerance: GPU_F32_PROMOTED_RESIDUAL_TOLERANCE,
        inner_tolerance: GPU_F32_TOLERANCE,
        true_residual_check_period: 1,
        physical_tolerance: GPU_F32_PROMOTED_RESIDUAL_TOLERANCE,
    };

    // Estimate peak GPU memory per frequency so the user can sanity-
    // check against device memory budget. Numbers below are for the
    // *worst-case* moment within one frequency solve, summed across
    // all device buffers in flight at that moment.
    //
    // `assembler.k_values.len()` is the actual union nnz of K, M and
    // boundary matrices (the sparsity pattern is shared across all
    // frequencies and across A and P), so it's the right size for the
    // CSR uploads.
    let mesh_nnz = assembler.k_values.len();
    let basis_count = 2_u64 + assembler.boundary_values.len() as u64; // K, M, boundaries
    let bytes_basis_c32 = basis_count * (mesh_nnz * 8) as u64;
    let bytes_basis_c64 = basis_count * (mesh_nnz * 16) as u64;
    let bytes_a_inner = (mesh_nnz * (8 + 4)) as u64 + ((n_nodes + 1) * 4) as u64;
    let bytes_p_inner = match precond {
        GpuPrecond::PolynomialJacobi { .. }
        | GpuPrecond::Chebyshev { .. }
        | GpuPrecond::Auto
        | GpuPrecond::Amg => 2 * bytes_a_inner,
        GpuPrecond::PointJacobi => 0,
    };
    let bytes_pointjacobi = (n_nodes * (8 + 4 + 4)) as u64;
    let bytes_precond_scratch = match precond {
        GpuPrecond::Chebyshev { .. } => (3 * n_nodes * 8) as u64,
        GpuPrecond::Auto | GpuPrecond::Amg => (8 * n_nodes * 8) as u64,
        GpuPrecond::PointJacobi | GpuPrecond::PolynomialJacobi { .. } => 0,
    };
    let bytes_a_outer = (mesh_nnz * (16 + 4)) as u64 + ((n_nodes + 1) * 4) as u64;
    let restart_outer = ir_cfg.restart_size.unwrap_or(ir_cfg.max_outer_iterations);
    let bytes_outer_basis = 2 * (restart_outer * n_nodes * 16) as u64; // V + Z
    let bytes_inner_basis_ir = (ir_cfg.max_inner_iterations * n_nodes * 8) as u64;
    let bytes_f32_basis = ((f32_cfg.max_iterations + 1) * n_nodes * 8) as u64;

    let peak_gpu_bytes = match precision {
        GpuPrecision::F32Pure => {
            let bytes_scratch_c32 = (8 * n_nodes * 8) as u64;
            let bytes_promoted_residual = bytes_a_outer + (3 * n_nodes * 16) as u64;
            let peak = bytes_a_inner
                + bytes_basis_c32
                + bytes_p_inner
                + bytes_pointjacobi
                + bytes_precond_scratch
                + bytes_f32_basis
                + bytes_scratch_c32
                + bytes_promoted_residual;
            log::info!(
                "FEM (cuda, {}, {}): estimated peak GPU memory per frequency: {} MiB \
                 (A32 {} MiB + basis32 {} MiB + P {} MiB + diag {} MiB + precond scratch {} MiB + Krylov basis {} MiB + scratch {} MiB + promoted residual {} MiB)",
                precision.label(),
                precond.label(),
                peak / (1024 * 1024),
                bytes_a_inner / (1024 * 1024),
                bytes_basis_c32 / (1024 * 1024),
                bytes_p_inner / (1024 * 1024),
                bytes_pointjacobi / (1024 * 1024),
                bytes_precond_scratch / (1024 * 1024),
                bytes_f32_basis / (1024 * 1024),
                bytes_scratch_c32 / (1024 * 1024),
                bytes_promoted_residual / (1024 * 1024),
            );
            peak
        }
        GpuPrecision::F64Outer => {
            let bytes_scratch_mixed = (5 * n_nodes * 16) as u64 + (3 * n_nodes * 8) as u64;
            let peak = bytes_a_outer
                + bytes_a_inner
                + bytes_basis_c64
                + bytes_basis_c32
                + bytes_p_inner
                + bytes_pointjacobi
                + bytes_precond_scratch
                + bytes_outer_basis
                + bytes_inner_basis_ir
                + bytes_scratch_mixed;
            log::info!(
                "FEM (cuda, {}, {}): estimated peak GPU memory per frequency: {} MiB \
                 (A_outer {} MiB + A_inner {} MiB + basis64 {} MiB + basis32 {} MiB + P {} MiB + precond scratch {} MiB + outer basis {} MiB + inner basis {} MiB + scratch {} MiB)",
                precision.label(),
                precond.label(),
                peak / (1024 * 1024),
                bytes_a_outer / (1024 * 1024),
                bytes_a_inner / (1024 * 1024),
                bytes_basis_c64 / (1024 * 1024),
                bytes_basis_c32 / (1024 * 1024),
                bytes_p_inner / (1024 * 1024),
                bytes_precond_scratch / (1024 * 1024),
                bytes_outer_basis / (1024 * 1024),
                bytes_inner_basis_ir / (1024 * 1024),
                bytes_scratch_mixed / (1024 * 1024),
            );
            peak
        }
    };
    anyhow::ensure!(
        peak_gpu_bytes <= gpu_budget_bytes,
        "estimated CUDA peak memory {} exceeds device {} conservative budget {} \
         (free {}, total {}). Lower --max-freq, increase --mesh-grading, \
         reduce CUDA solver memory, or choose a GPU with more VRAM.",
        fmt_gib(peak_gpu_bytes),
        cuda_device,
        fmt_gib(gpu_budget_bytes),
        fmt_gib(gpu_free_bytes),
        fmt_gib(gpu_total_bytes),
    );
    diagnostics.set_execution_memory(peak_gpu_bytes, Some(gpu_free_bytes), Some(gpu_total_bytes));

    log::info!(
        "FEM (cuda, {}): solving {} sources × {} frequencies with {} preconditioner",
        precision.label(),
        n_sources,
        n_freqs,
        precond.label(),
    );
    log::info!(
        "FEM (cuda, {}): solver budget: ir_outer<={} ir_inner<={} ir_restart={} ir_tol={:.1e}; f32_iter<={} f32_est_tol={:.1e} f32_promoted_res_tol={:.1e}",
        precision.label(),
        ir_cfg.max_outer_iterations,
        ir_cfg.max_inner_iterations,
        ir_cfg.restart_size.unwrap_or(ir_cfg.max_outer_iterations),
        ir_cfg.outer_tolerance,
        f32_cfg.max_iterations,
        f32_cfg.tolerance,
        GPU_F32_PROMOTED_RESIDUAL_TOLERANCE,
    );
    let rebuild_every = mesh_config.rebuild_precond_every.max(1);
    let mut cached_point_jacobi: Option<PointJacobiBackend<CudaBackend, Complex32>> = None;
    let mut cached_amg: Option<BackendAmgPreconditioner<CudaBackend, Complex32>> = None;
    let mut cached_polynomial: Option<PolynomialJacobiBackend<'static, CudaBackend, Complex32>> = None;
    let mut cached_chebyshev: Option<ChebyshevBackend<'static, CudaBackend, Complex32>> = None;
    // Sticky AMG-to-Chebyshev downgrade, armed by the first AMG setup
    // failure (e.g. a low-frequency coarse operator that is singular in
    // f32). Without it one bad rebuild aborts the whole sweep with zero
    // data even though the polynomial smoother solves the same system.
    let mut amg_setup_fallback = false;
    log::info!(
        "FEM (cuda): rebuilding reusable GPU preconditioners every {} freq{}",
        rebuild_every,
        if rebuild_every == 1 { "" } else { "s" },
    );

    let structure_upload_start = Instant::now();
    let row_ptrs_i32 = usize_indices_to_i32(&assembler.row_ptrs, "row_ptrs")?;
    let col_idx_i32 = usize_indices_to_i32(&assembler.col_indices, "col_indices")?;
    let a_inner = BackendCsr::<CudaBackend, Complex32>::try_from_host_structure(
        backend,
        n_nodes,
        n_nodes,
        &row_ptrs_i32,
        &col_idx_i32,
    )
    .map_err(|error| anyhow::anyhow!("CUDA structure allocation/upload failed for A32: {error}"))?;
    let a_outer = BackendCsr::<CudaBackend, Complex64>::try_from_host_structure(
        backend,
        n_nodes,
        n_nodes,
        &row_ptrs_i32,
        &col_idx_i32,
    )
    .map_err(|error| anyhow::anyhow!("CUDA structure allocation/upload failed for A64: {error}"))?;
    let mut a_inner = sonium_compute::PersistentDeviceCsr::try_from_device_csr(backend, a_inner)?;
    let mut a_outer = sonium_compute::PersistentDeviceCsr::try_from_device_csr(backend, a_outer)?;
    // P_inner is resident for every polynomial-family choice and for
    // Auto/AMG as well: an AMG setup failure downgrades the sweep to the
    // Chebyshev smoother, which needs the live P device CSR. One extra
    // P-sized structure (~the cost of A32) is negligible next to the
    // Krylov basis reservation.
    let mut p_inner_opt: Option<BackendCsr<CudaBackend, Complex32>> = match precond {
        GpuPrecond::PolynomialJacobi { .. }
        | GpuPrecond::Chebyshev { .. }
        | GpuPrecond::Auto
        | GpuPrecond::Amg => Some(
            BackendCsr::<CudaBackend, Complex32>::try_from_host_structure(
                backend,
                n_nodes,
                n_nodes,
                &row_ptrs_i32,
                &col_idx_i32,
            )
            .map_err(|error| {
                anyhow::anyhow!("CUDA structure allocation/upload failed for P32: {error}")
            })?,
        ),
        GpuPrecond::PointJacobi => None,
    };
    backend.synchronize()?;
    log::info!(
        "FEM (cuda): uploaded reusable CSR structure nnz={} in {}",
        mesh_nnz,
        fmt_duration(structure_upload_start.elapsed()),
    );

    let basis_upload_start = Instant::now();
    let k_basis_32 = upload_real_basis_as_complex32(backend, &assembler.k_values, "K")?;
    let m_basis_32 = upload_real_basis_as_complex32(backend, &assembler.m_values, "M")?;
    let boundary_basis_32 =
        upload_boundary_basis_as_complex32(backend, &assembler.boundary_values)?;
    let k_basis_64_opt = match precision {
        GpuPrecision::F64Outer => Some(upload_real_basis_as_complex64(
            backend,
            &assembler.k_values,
            "K",
        )?),
        GpuPrecision::F32Pure => None,
    };
    let m_basis_64_opt = match precision {
        GpuPrecision::F64Outer => Some(upload_real_basis_as_complex64(
            backend,
            &assembler.m_values,
            "M",
        )?),
        GpuPrecision::F32Pure => None,
    };
    let boundary_basis_64_opt = match precision {
        GpuPrecision::F64Outer => Some(upload_boundary_basis_as_complex64(
            backend,
            &assembler.boundary_values,
        )?),
        GpuPrecision::F32Pure => None,
    };
    let diag_basis_32 = build_diag_basis_32(
        &assembler.row_ptrs,
        &assembler.col_indices,
        &assembler.k_values,
        &assembler.m_values,
        n_nodes,
    );
    let mut p_diag_32 = vec![Complex32::new(0.0, 0.0); n_nodes];
    let mut p_values_host_32 = match precond {
        GpuPrecond::Chebyshev { .. } | GpuPrecond::Auto | GpuPrecond::Amg => {
            vec![Complex32::new(0.0, 0.0); mesh_nnz]
        }
        GpuPrecond::PointJacobi | GpuPrecond::PolynomialJacobi { .. } => Vec::new(),
    };
    backend.synchronize()?;
    log::info!(
        "FEM (cuda): uploaded reusable value bases ({} vectors, nnz={}) in {}",
        basis_count,
        mesh_nnz,
        fmt_duration(basis_upload_start.elapsed()),
    );

    let mut rhs_dev_32_opt = match precision {
        GpuPrecision::F32Pure => Some(gpu_alloc(backend, n_nodes, "rhs f32")?),
        GpuPrecision::F64Outer => None,
    };
    let mut x_dev_32_opt = match precision {
        GpuPrecision::F32Pure => Some(gpu_alloc(backend, n_nodes, "x f32")?),
        GpuPrecision::F64Outer => None,
    };
    let mut rhs_dev_64 = gpu_alloc(backend, n_nodes, "rhs/residual f64")?;
    let mut x_dev_64 = gpu_alloc(backend, n_nodes, "x/residual f64")?;
    let mut f32_residual_dev_opt = match precision {
        GpuPrecision::F32Pure => Some(gpu_alloc(backend, n_nodes, "residual scratch f32")?),
        GpuPrecision::F64Outer => None,
    };
    let mut f32_correction_dev_opt = match precision {
        GpuPrecision::F32Pure => Some(gpu_alloc(backend, n_nodes, "correction scratch f32")?),
        GpuPrecision::F64Outer => None,
    };
    let mut residual_dev_64_opt = match precision {
        GpuPrecision::F32Pure => Some(gpu_alloc(backend, n_nodes, "residual scratch f64")?),
        GpuPrecision::F64Outer => None,
    };
    let mut prev_x_32_opt = match precision {
        GpuPrecision::F32Pure => Some(
            (0..n_sources)
                .map(|src| gpu_alloc(backend, n_nodes, &format!("prev x f32 src {src}")))
                .collect::<Result<Vec<_>>>()?,
        ),
        GpuPrecision::F64Outer => None,
    };
    let mut prev_x_64_opt = match precision {
        GpuPrecision::F32Pure => None,
        GpuPrecision::F64Outer => Some(
            (0..n_sources)
                .map(|src| gpu_alloc(backend, n_nodes, &format!("prev x f64 src {src}")))
                .collect::<Result<Vec<_>>>()?,
        ),
    };
    let mut prev_solution_valid = vec![false; n_sources];
    let rhs_basis_start = Instant::now();
    let rhs_bases: Vec<SourceRhsBasis> = simulation
        .sources
        .iter()
        .map(|source| {
            build_source_rhs_basis_for_model(
                &mesh,
                source,
                image_walls_aabb,
                mesh_config.source_model,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    log::info!(
        "FEM (cuda): precomputed {} source RHS bases in {}",
        rhs_bases.len(),
        fmt_duration(rhs_basis_start.elapsed()),
    );
    let mut rhs_assemblers_32 = match precision {
        GpuPrecision::F32Pure => Some(
            rhs_bases
                .iter()
                .map(|basis| {
                    let entries: Vec<_> = basis
                        .entries
                        .iter()
                        // `assemble_source_coefficients_into` already
                        // multiplies each coefficient by its basis
                        // weight. The device matrix is therefore a pure
                        // scatter and must not apply that weight again.
                        .map(|&(node, _, _)| (node, 1.0_f32))
                        .collect();
                    DeviceRhsAssembler::<CudaBackend, Complex32>::new(backend, n_nodes, &entries)
                })
                .collect::<Result<Vec<_>>>()?,
        ),
        GpuPrecision::F64Outer => None,
    };
    let mut rhs_assemblers_64 = match precision {
        GpuPrecision::F64Outer => Some(
            rhs_bases
                .iter()
                .map(|basis| {
                    let entries: Vec<_> = basis
                        .entries
                        .iter()
                        .map(|&(node, _, _)| (node, 1.0_f64))
                        .collect();
                    DeviceRhsAssembler::<CudaBackend, Complex64>::new(backend, n_nodes, &entries)
                })
                .collect::<Result<Vec<_>>>()?,
        ),
        GpuPrecision::F32Pure => None,
    };
    let mut rhs_coefficients_64: Vec<Vec<Complex64>> = rhs_bases
        .iter()
        .map(|basis| vec![Complex64::new(0.0, 0.0); basis.entries.len()])
        .collect();
    let mut rhs_coefficients_32: Vec<Vec<Complex32>> = rhs_bases
        .iter()
        .map(|basis| vec![Complex32::new(0.0, 0.0); basis.entries.len()])
        .collect();
    let mut rhs_block_assembler_32 = if precision == GpuPrecision::F32Pure {
        let entries = rhs_bases
            .iter()
            .map(|basis| {
                basis
                    .entries
                    .iter()
                    // Coefficients are already basis-weighted; this
                    // matrix only scatters each entry into its node.
                    .map(|&(node, _, _)| (node, 1.0_f32))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        Some(DeviceRhsBlockAssembler::<CudaBackend, Complex32>::new(
            backend, n_nodes, &entries,
        )?)
    } else {
        None
    };
    let block_coefficient_len = rhs_bases
        .iter()
        .map(|basis| basis.entries.len())
        .sum::<usize>();
    let mut rhs_block_coefficients_32 = vec![Complex32::new(0.0, 0.0); block_coefficient_len];
    let mut rhs_block_dev_32_opt = if precision == GpuPrecision::F32Pure {
        Some(gpu_alloc(backend, n_nodes * n_sources, "batched RHS f32")?)
    } else {
        None
    };
    let mut x_block_dev_32_opt = if precision == GpuPrecision::F32Pure {
        Some(gpu_alloc(
            backend,
            n_nodes * n_sources,
            "batched solution f32",
        )?)
    } else {
        None
    };
    let mut rhs_block_dev_64_opt = if precision == GpuPrecision::F32Pure {
        Some(gpu_alloc(
            backend,
            n_nodes * n_sources,
            "batched RHS f64 gate",
        )?)
    } else {
        None
    };
    let mut x_block_dev_64_opt = if precision == GpuPrecision::F32Pure {
        Some(gpu_alloc(
            backend,
            n_nodes * n_sources,
            "batched solution f64 gate",
        )?)
    } else {
        None
    };
    let mut residual_block_dev_64_opt = if precision == GpuPrecision::F32Pure {
        Some(gpu_alloc(
            backend,
            n_nodes * n_sources,
            "batched residual f64 gate",
        )?)
    } else {
        None
    };
    let mut block_correction_rhs = if precision == GpuPrecision::F32Pure {
        Some(gpu_alloc(backend, n_nodes * n_sources, "block correction RHS")?)
    } else { None };
    let mut block_correction = if precision == GpuPrecision::F32Pure {
        Some(gpu_alloc(backend, n_nodes * n_sources, "block correction")?)
    } else { None };
    let setup_transfer_stats = FemTransferStats::from(backend.transfer_stats());
    backend.reset_transfer_stats();
    for (freq_idx, &freq) in simulation.frequencies.iter().enumerate() {
        ensure_device_not_cancelled(cancellation, "cuda")?;
        let transfer_start = backend.transfer_stats();
        let runtime_start = backend.runtime_stats();
        let freq_start = Instant::now();
        let k = simulation.wavenumber(freq);
        let k_complex = complex_wavenumber(k, freq, mesh_config.modal_damping_ratio);
        // After an AMG setup failure the sweep continues with the
        // Chebyshev smoother (see the AMG arm below); the downgrade is
        // sticky because the coarse operator stays singular across the
        // low-frequency band that triggered it.
        let mut active_precond = if amg_setup_fallback {
            GpuPrecond::Chebyshev {
                degree: GPU_AMG_SETUP_FALLBACK_CHEBYSHEV_DEGREE,
            }
        } else {
            precond
        };
        let rebuild_gpu_precond = match active_precond {
            GpuPrecond::PointJacobi => {
                cached_point_jacobi.is_none() || freq_idx % rebuild_every == 0
            }
            GpuPrecond::Auto | GpuPrecond::Amg => {
                cached_amg.is_none() || freq_idx % rebuild_every == 0
            }
            // Polynomial coefficients track the live shifted operator every
            // frequency; owned descriptors and application scratch stay cached.
            GpuPrecond::PolynomialJacobi { .. } | GpuPrecond::Chebyshev { .. } => true,
        };

        if should_log_frequency(freq_idx, n_freqs) {
            log::info!(
                "FEM (cuda): frequency {}/{} ({:.1} Hz) started",
                freq_idx + 1,
                n_freqs,
                freq,
            );
        }

        let matrix_start = Instant::now();
        let boundary_coeffs =
            compute_boundary_coefficients(simulation, &resolved_boundaries, freq)?;
        let p_shift = mesh_config.shifted_laplacian_shift.value(k);
        let k_sq = k_complex * k_complex;
        fill_shifted_diag_32(&mut p_diag_32, &diag_basis_32, p_shift);
        if matches!(active_precond, GpuPrecond::Chebyshev { .. })
            || (matches!(active_precond, GpuPrecond::Auto | GpuPrecond::Amg) && rebuild_gpu_precond)
        {
            fill_shifted_values_32(
                &mut p_values_host_32,
                &assembler.k_values,
                &assembler.m_values,
                p_shift,
            );
        }
        let matrix_time = matrix_start.elapsed();

        // K, M, and boundary basis values are already resident. Refresh
        // frequency-dependent A/P values by device-side linear
        // combinations instead of assembling full host CSR value arrays.
        let upload_start = Instant::now();
        let mut terms32 = Vec::with_capacity(1 + boundary_basis_32.len());
        terms32.push((Complex32::new(-k_sq.re as f32, -k_sq.im as f32), &m_basis_32));
        for (tag, values) in &boundary_basis_32 {
            if let Some(coeff) = boundary_coeffs.get(tag) {
                terms32.push((Complex32::new(coeff.re as f32, coeff.im as f32), values));
            }
        }
        a_inner.try_combine_values(backend, &k_basis_32, &terms32)?;
        match precision {
            GpuPrecision::F32Pure => {
                a_outer.try_cast_values_from_lower(backend, &a_inner.matrix().values)?;
            }
            GpuPrecision::F64Outer => {
                let boundaries = boundary_basis_64_opt.as_ref()
                    .expect("F64Outer mode requires boundary f64 basis");
                let mut terms64 = Vec::with_capacity(1 + boundaries.len());
                terms64.push((-k_sq, m_basis_64_opt.as_ref()
                    .expect("F64Outer mode requires M f64 basis")));
                for (tag, values) in boundaries {
                    if let Some(&coeff) = boundary_coeffs.get(tag) {
                        terms64.push((coeff, values));
                    }
                }
                a_outer.try_combine_values(backend,
                    k_basis_64_opt.as_ref().expect("F64Outer mode requires K f64 basis"),
                    &terms64)?;
            }
        }
        if let Some(p_inner) = p_inner_opt.as_mut() {
            update_shifted_values_32(
                backend,
                &mut p_inner.values,
                &k_basis_32,
                &m_basis_32,
                p_shift,
            );
        }
        backend.synchronize()?;
        let upload_time = upload_start.elapsed();


        let mut rhs_time = Duration::ZERO;
        let mut solve_time = Duration::ZERO;
        let mut total_outer = 0usize;
        let mut total_inner = 0usize;
        let mut max_residual = 0.0f64;
        let mut retry_count = 0usize;
        let mut operator_work = FemOperatorWorkEvidence {
            source_work: 0.0,
            volume_damping_work: 0.0,
            boundary_work: 0.0,
            balance_residual: 0.0,
        };
        let mut boundary_power_by_tag = BTreeMap::new();

        let preconditioner_start = Instant::now();
        if active_precond == GpuPrecond::PointJacobi && rebuild_gpu_precond {
            if let Some(pj) = cached_point_jacobi.as_mut() {
                pj.try_update_diagonal(backend, &p_diag_32)
                    .map_err(|error| anyhow::anyhow!("CUDA Point-Jacobi update failed: {error}"))?;
            } else {
                cached_point_jacobi = Some(
                    PointJacobiBackend::<CudaBackend, Complex32>::try_from_diagonal(
                        backend, &p_diag_32,
                    )
                    .map_err(|error| anyhow::anyhow!("CUDA Point-Jacobi setup failed: {error}"))?,
                );
            }
        }
        if matches!(active_precond, GpuPrecond::Auto | GpuPrecond::Amg) && rebuild_gpu_precond {
            let p_host = CsrMatrix::from_raw_parts(
                n_nodes,
                n_nodes,
                assembler.row_ptrs.clone(),
                assembler.col_indices.clone(),
                p_values_host_32.clone(),
            );
            match BackendAmgPreconditioner::<CudaBackend, Complex32>::try_from_csr_with_cuda_coarse_lu(
                backend,
                &p_host,
                gpu_amg_config(),
            ) {
                Ok(amg) => {
                    log::debug!(
                        "FEM (cuda): AMG(P) levels={} setup+upload={}",
                        amg.num_levels(),
                        fmt_duration(preconditioner_start.elapsed()),
                    );
                    cached_amg = Some(amg);
                }
                Err(error) => {
                    // A failed refresh must not leave the old hierarchy eligible
                    // after the sweep has explicitly selected its fallback.
                    cached_amg = None;
                    amg_setup_fallback = true;
                    active_precond = GpuPrecond::Chebyshev {
                        degree: GPU_AMG_SETUP_FALLBACK_CHEBYSHEV_DEGREE,
                    };
                    log::warn!(
                        "FEM (cuda): AMG setup failed at {freq:.1} Hz ({error}); using Chebyshev({}) for the remaining sweep",
                        GPU_AMG_SETUP_FALLBACK_CHEBYSHEV_DEGREE,
                    );
                    diagnostics.set_execution_profile(
                        "cuda",
                        "cuda",
                        precision.label(),
                        &format!(
                            "amg-setup-fallback:chebyshev-{}",
                            GPU_AMG_SETUP_FALLBACK_CHEBYSHEV_DEGREE
                        ),
                    );
                }
            }
        }
        if let GpuPrecond::PolynomialJacobi { degree } = active_precond {
            let matrix = p_inner_opt
                .as_ref()
                .expect("polynomial Jacobi requires P_inner");
            if let Some(polynomial) = cached_polynomial.as_mut() {
                polynomial.try_update_numeric(backend, &matrix.values, &p_diag_32)?;
            } else {
                let owned =
                    sonium_compute::PersistentDeviceCsr::try_clone_from_device_csr(backend, matrix)?;
                cached_polynomial = Some(
                    PolynomialJacobiBackend::try_from_owned_csr(
                        backend,
                        owned,
                        &p_diag_32,
                        degree,
                        Complex32::new(POLY_JACOBI_DAMPING, 0.0),
                    )
                    .map_err(|error| anyhow::anyhow!("CUDA polynomial-Jacobi setup failed: {error}"))?,
                );
            }
        }
        if let GpuPrecond::Chebyshev { degree } = active_precond {
            let matrix = p_inner_opt.as_ref().expect("Chebyshev requires P_inner");
            if let Some(chebyshev) = cached_chebyshev.as_mut() {
                chebyshev.try_update_from_csr_host(
                    backend,
                    &matrix.values,
                    &row_ptrs_i32,
                    &col_idx_i32,
                    &p_values_host_32,
                )?;
            } else {
                let owned =
                    sonium_compute::PersistentDeviceCsr::try_clone_from_device_csr(backend, matrix)?;
                cached_chebyshev = Some(
                    ChebyshevBackend::try_from_owned_csr_host(
                        backend,
                        owned,
                        &row_ptrs_i32,
                        &col_idx_i32,
                        &p_values_host_32,
                        degree,
                        CHEBYSHEV_EIGENVALUE_RATIO,
                    )
                    .map_err(|error| anyhow::anyhow!("CUDA Chebyshev setup failed: {error}"))?,
                );
            }
        }
        let preconditioner_time = preconditioner_start.elapsed();

        // All admitted preconditioners expose device block actions. Rejected
        // block certification uses the same preconditioner in scalar recovery.
        let mut batch_succeeded = false;
        let mut block_solve = FemBlockSolveOutcome::NotAttempted;
        if precision == GpuPrecision::F32Pure
            && n_sources > 1
            && !solver_nonconvergence_injection
        {
            let block_preconditioner: &dyn BackendPreconditioner<CudaBackend, Complex32> =
                match active_precond {
                    GpuPrecond::PointJacobi => cached_point_jacobi.as_ref()
                        .expect("point Jacobi initialized before block solve"),
                    GpuPrecond::Auto | GpuPrecond::Amg => cached_amg.as_ref()
                        .expect("AMG initialized before block solve"),
                    GpuPrecond::PolynomialJacobi { .. } => cached_polynomial.as_ref()
                        .expect("polynomial Jacobi initialized before block solve"),
                    GpuPrecond::Chebyshev { .. } => cached_chebyshev.as_ref()
                        .expect("Chebyshev initialized before block solve"),
                };
            let mut coefficient_offset = 0usize;
            for src_idx in 0..n_sources {
                assemble_source_coefficients_into(
                    &rhs_bases[src_idx],
                    &simulation.sources[src_idx],
                    freq,
                    &mut rhs_coefficients_64[src_idx],
                );
                for (dst, src) in rhs_block_coefficients_32
                    [coefficient_offset..coefficient_offset + rhs_bases[src_idx].entries.len()]
                    .iter_mut()
                    .zip(&rhs_coefficients_64[src_idx])
                {
                    *dst = Complex32::new(src.re as f32, src.im as f32);
                }
                coefficient_offset += rhs_bases[src_idx].entries.len();
            }
            let batch_start = Instant::now();
            let result = solve_f32_sources_block(
                backend,
                &mut f32_block_workspace,
                &a_inner,
                &a_outer,
                block_preconditioner,
                rhs_block_assembler_32
                    .as_mut()
                    .expect("batched f32 RHS assembler must be initialized"),
                &rhs_block_coefficients_32,
                rhs_block_dev_32_opt
                    .as_mut()
                    .expect("batched f32 RHS buffer must be initialized"),
                x_block_dev_32_opt
                    .as_mut()
                    .expect("batched f32 solution buffer must be initialized"),
                block_correction_rhs.as_mut().expect("block correction RHS initialized"),
                block_correction.as_mut().expect("block correction initialized"),
                rhs_block_dev_64_opt
                    .as_mut()
                    .expect("batched f64 RHS gate buffer must be initialized"),
                x_block_dev_64_opt
                    .as_mut()
                    .expect("batched f64 solution gate buffer must be initialized"),
                residual_block_dev_64_opt
                    .as_mut()
                    .expect("batched f64 residual gate buffer must be initialized"),
                residual_dev_64_opt
                    .as_mut()
                    .expect("f32 residual norm scratch must be initialized"),
                listener_32
                    .as_mut()
                    .expect("batched f32 listener extraction must be initialized"),
                listener_64
                    .as_mut()
                    .expect("batched f64 listener extraction must be initialized"),
                n_sources,
                f32_policy,
            );
            solve_time += batch_start.elapsed();
            match result {
                Ok((listener_values, statuses, residual, correction_steps)) => {
                    block_solve = FemBlockSolveOutcome::Accepted { correction_steps };
                    total_outer += statuses
                        .iter()
                        .map(|status| status.iterations)
                        .sum::<usize>();
                    max_residual = max_residual.max(residual);
                    for (src_idx, values) in listener_values.iter().enumerate() {
                        store_listener_pressures(
                            &simulation.listening_positions,
                            values,
                            &simulation.sources[src_idx],
                            freq,
                            &mut pressures[src_idx],
                            freq_idx,
                        );
                    }
                    if wants_full_solution_evidence(mesh_config) {
                        let mut solution_block_host =
                            vec![Complex64::new(0.0, 0.0); n_nodes * n_sources];
                        backend.copy_d2h(
                            x_block_dev_64_opt
                                .as_ref()
                                .expect("batched f64 solution gate buffer must be initialized"),
                            &mut solution_block_host,
                        );
                        for (src_idx, source) in simulation.sources.iter().enumerate() {
                            let start = src_idx * n_nodes;
                            accumulate_host_solution_evidence(
                                &assembler,
                                k_complex,
                                &boundary_coeffs,
                                &rhs_bases[src_idx],
                                source,
                                freq,
                                &solution_block_host[start..start + n_nodes],
                                &mut operator_work,
                                &mut boundary_power_by_tag,
                            )?;
                        }
                    }
                    prev_solution_valid.fill(false);
                    batch_succeeded = true;
                }
                Err(error) => {
                    block_solve = FemBlockSolveOutcome::Rejected { reason: error.clone() };
                    log::debug!(
                        "FEM (cuda f32-pure): batched source solve failed at {:.1} Hz: {error}; falling back to per-source retries",
                        freq,
                    );
                }
            }
        }

        if !batch_succeeded {
            // Per-source helper solves one RHS and downloads the solution.
            // The preconditioner is built once below, then reused across all
            // sources for this frequency.
            type SolveResult = (
                bool,     // converged
                usize,    // outer iters
                usize,    // inner iters
                f64,      // residual
                Duration, // rhs assemble time
                Duration, // solve time
            );
            let mut solve_for_source = |precond_dyn: &dyn BackendPreconditioner<CudaBackend, Complex32>,
                                        src_idx: usize,
                                        source: &Source,
                                        use_warm_start: bool,
                                        x_dev_64: &mut <CudaBackend as Backend>::Buffer<
                Complex64,
            >|
             -> Result<Option<SolveResult>> {
                if !use_warm_start {
                    prev_solution_valid[src_idx] = false;
                }
                let rhs_t = Instant::now();
                assemble_source_coefficients_into(
                    &rhs_bases[src_idx],
                    source,
                    freq,
                    &mut rhs_coefficients_64[src_idx],
                );
                let rhs_dt = rhs_t.elapsed();

                if solver_nonconvergence_injection {
                    anyhow::bail!(
                        "cuda_solver_nonconvergence: deterministic injected solver non-convergence for source '{}' at {:.1} Hz",
                        source.name,
                        freq,
                    );
                }

                let solve_t = Instant::now();
                let result = match precision {
                    GpuPrecision::F32Pure => {
                        for (dst, src) in rhs_coefficients_32[src_idx]
                            .iter_mut()
                            .zip(&rhs_coefficients_64[src_idx])
                        {
                            *dst = Complex32::new(src.re as f32, src.im as f32);
                        }
                        let rhs_dev = rhs_dev_32_opt
                            .as_mut()
                            .expect("F32Pure mode requires reusable f32 RHS buffer");
                        rhs_assemblers_32
                            .as_mut()
                            .expect("F32Pure mode requires device RHS assembler")[src_idx]
                            .assemble(backend, &rhs_coefficients_32[src_idx], rhs_dev)?;
                        let x_dev = x_dev_32_opt
                            .as_mut()
                            .expect("F32Pure mode requires reusable f32 solution buffer");
                        let x0_dev = if use_warm_start && prev_solution_valid[src_idx] {
                            Some(
                                &prev_x_32_opt
                                    .as_ref()
                                    .expect("F32Pure mode requires f32 warm-start buffers")
                                    [src_idx],
                            )
                        } else {
                            None
                        };
                        let residual32 = f32_residual_dev_opt
                            .as_mut()
                            .expect("F32Pure mode requires f32 residual scratch buffer");
                        let correction32 = f32_correction_dev_opt
                            .as_mut()
                            .expect("F32Pure mode requires f32 correction scratch buffer");
                        let residual64 = residual_dev_64_opt
                            .as_mut()
                            .expect("F32Pure mode requires f64 residual scratch buffer");
                        let mut scratch = F32ResidualCorrectionScratch {
                            residual32,
                            correction32,
                            x64: x_dev_64,
                            rhs64: &mut rhs_dev_64,
                            residual64,
                        };
                        let problem = F32ResidualCorrectionProblem {
                            pool: &krylov_pool,
                            a32: &a_inner,
                            a64: &a_outer,
                            precond: precond_dyn,
                            cfg: &f32_cfg,
                        };
                        let r = solve_f32_with_residual_correction(
                            &problem,
                            rhs_dev,
                            x0_dev,
                            x_dev,
                            &mut f32_krylov_workspace,
                            &mut scratch,
                        );
                        drop(scratch);
                        backend.synchronize()?;
                        match r {
                            Ok((status, promoted_residual)) => {
                                if promoted_residual <= GPU_F32_PROMOTED_RESIDUAL_TOLERANCE {
                                    let prev_x = &mut prev_x_32_opt
                                        .as_mut()
                                        .expect("F32Pure mode requires f32 warm-start buffers")
                                        [src_idx];
                                    backend.copy_d2d(x_dev, prev_x);
                                    prev_solution_valid[src_idx] = true;
                                }
                                let listener_values = listener_32
                                    .as_mut()
                                    .expect("F32Pure mode requires listener extraction")
                                    .evaluate_to_host(backend, x_dev);
                                for (dst, src) in lp_values_host.iter_mut().zip(listener_values) {
                                    *dst = Complex64::new(src.re as f64, src.im as f64);
                                }
                                let promoted_listener_values = listener_64
                                    .as_mut()
                                    .expect("f32 mode requires f64 listener extraction")
                                    .evaluate_to_host(backend, x_dev_64);
                                let physical_error = lp_values_host
                                    .iter()
                                    .zip(&promoted_listener_values)
                                    .map(|(observed, reference)| {
                                        (*observed - *reference).norm()
                                            / reference.norm().max(1.0e-30)
                                    })
                                    .fold(0.0_f64, f64::max);
                                let gates =
                                    f32_policy.validate_gates(promoted_residual, physical_error);
                                if gates.is_ok() && wants_full_solution_evidence(mesh_config) {
                                    if let Err(error) = collect_device_solution_evidence(
                                        backend,
                                        &x_dev_64,
                                        &assembler,
                                        k_complex,
                                        &boundary_coeffs,
                                        &rhs_bases[src_idx],
                                        source,
                                        freq,
                                        &mut operator_work,
                                        &mut boundary_power_by_tag,
                                    ) {
                                        return Err(anyhow::Error::new(
                                            FemBackendExecutionError::PhysicalEvidence {
                                                backend: "cuda".to_string(),
                                                detail: format!(
                                                    "source '{}' at {:.1} Hz: {error}",
                                                    source.name, freq
                                                ),
                                            },
                                        ));
                                    }
                                }
                                if let Err(error) = &gates {
                                    log::debug!(
                                        "FEM (cuda f32-pure) rejected f64 acceptance gates for source {} freq {:.1} Hz: {error}",
                                        source.name,
                                        freq,
                                    );
                                }

                                if log::log_enabled!(log::Level::Debug) {
                                    log::debug!(
                                        "FEM (cuda {}) source {} freq {:.1} Hz estimated_res={:.2e} promoted_res={:.2e} estimated_converged={}",
                                        precision.label(),
                                        source.name,
                                        freq,
                                        status.final_residual,
                                        promoted_residual,
                                        status.converged,
                                    );
                                }
                                Ok((gates.is_ok(), status.iterations, 0usize, promoted_residual))
                            }
                            Err(e) => Err(anyhow::anyhow!(e)),
                        }
                    }
                    GpuPrecision::F64Outer => {
                        let rhs_dev = &mut rhs_dev_64;
                        rhs_assemblers_64
                            .as_mut()
                            .expect("F64Outer mode requires device RHS assembler")[src_idx]
                            .assemble(backend, &rhs_coefficients_64[src_idx], rhs_dev)?;
                        let x_dev = &mut *x_dev_64;
                        let x0_dev = if use_warm_start && prev_solution_valid[src_idx] {
                            Some(
                                &prev_x_64_opt
                                    .as_ref()
                                    .expect("F64Outer mode requires f64 warm-start buffers")
                                    [src_idx],
                            )
                        } else {
                            None
                        };
                        let r = gmres_ir_backend_preconditioned_with_workspace::<
                            CudaBackend,
                            sonium_compute::PersistentDeviceCsr<CudaBackend, Complex64>,
                            sonium_compute::PersistentDeviceCsr<CudaBackend, Complex32>,
                            dyn BackendPreconditioner<CudaBackend, Complex32>,
                            Complex64,
                        >(
                            &mut ir_workspace,
                            &a_outer,
                            &a_inner,
                            precond_dyn,
                            rhs_dev,
                            x0_dev,
                            x_dev,
                            &ir_cfg,
                        );
                        backend.synchronize()?;
                        match r {
                            Ok(status) => {
                                if status.converged {
                                    let prev_x = &mut prev_x_64_opt
                                        .as_mut()
                                        .expect("F64Outer mode requires f64 warm-start buffers")
                                        [src_idx];
                                    backend.copy_d2d(x_dev, prev_x);
                                    prev_solution_valid[src_idx] = true;
                                }
                                if status.converged && wants_full_solution_evidence(mesh_config) {
                                    if let Err(error) = collect_device_solution_evidence(
                                        backend,
                                        x_dev,
                                        &assembler,
                                        k_complex,
                                        &boundary_coeffs,
                                        &rhs_bases[src_idx],
                                        source,
                                        freq,
                                        &mut operator_work,
                                        &mut boundary_power_by_tag,
                                    ) {
                                        return Err(anyhow::Error::new(
                                            FemBackendExecutionError::PhysicalEvidence {
                                                backend: "cuda".to_string(),
                                                detail: format!(
                                                    "source '{}' at {:.1} Hz: {error}",
                                                    source.name, freq
                                                ),
                                            },
                                        ));
                                    }
                                }
                                if log::log_enabled!(log::Level::Debug) {
                                    log::debug!(
                                        "FEM (cuda {}) source {} freq {:.1} Hz residual_history={:?} true_residual_checks={:?}",
                                        precision.label(),
                                        source.name,
                                        freq,
                                        status.residual_history,
                                        status.true_residual_checks,
                                    );
                                }
                                lp_values_host = listener_64
                                    .as_mut()
                                    .expect("F64Outer mode requires listener extraction")
                                    .evaluate_to_host(backend, x_dev);
                                Ok((
                                    status.converged,
                                    status.outer_iterations,
                                    status.total_inner_iterations,
                                    status.final_outer_residual,
                                ))
                            }
                            Err(e) => Err(anyhow::anyhow!("{e}")),
                        }
                    }
                };
                let solve_dt = solve_t.elapsed();
                match result {
                    Ok((conv, oi, ii, res)) => {
                        store_listener_pressures(
                            &simulation.listening_positions,
                            &lp_values_host,
                            source,
                            freq,
                            &mut pressures[src_idx],
                            freq_idx,
                        );
                        Ok(Some((conv, oi, ii, res, rhs_dt, solve_dt)))
                    }
                    Err(e) => {
                        log::warn!(
                            "FEM (cuda {}) solve failed for source {} freq {:.1} Hz: {e}",
                            precision.label(),
                            source.name,
                            freq,
                        );
                        Ok(None)
                    }
                }
            };

            let mut solve_and_record = |src_idx: usize,
                                        source: &Source,
                                        precond_dyn: &dyn BackendPreconditioner<CudaBackend, Complex32>|
             -> Result<()> {
                let mut record_attempt =
                    |outer_iters: usize,
                     inner_iters: usize,
                     residual: f64,
                     rhs_dt: Duration,
                     solve_dt: Duration| {
                        rhs_time += rhs_dt;
                        solve_time += solve_dt;
                        total_outer += outer_iters;
                        total_inner += inner_iters;
                        max_residual = max_residual.max(residual);
                    };

                match solve_for_source(precond_dyn, src_idx, source, true, &mut x_dev_64) {
                    Ok(Some((true, outer_iters, inner_iters, residual, rhs_dt, solve_dt))) => {
                        record_attempt(outer_iters, inner_iters, residual, rhs_dt, solve_dt);
                        return Ok(());
                    }
                    Ok(Some((false, outer_iters, inner_iters, residual, rhs_dt, solve_dt))) => {
                        record_attempt(outer_iters, inner_iters, residual, rhs_dt, solve_dt);
                        retry_count += 1;
                        log::warn!(
                            "FEM (cuda {}) did not converge for source {} freq {:.1} Hz: outer={} inner={} res={:.2e} (retrying without warm start)",
                            precision.label(),
                            source.name,
                            freq,
                            outer_iters,
                            inner_iters,
                            residual,
                        );
                    }
                    Ok(None) => {
                        retry_count += 1;
                        log::warn!(
                            "FEM (cuda {}) failed for source {} freq {:.1} Hz (retrying without warm start)",
                            precision.label(),
                            source.name,
                            freq,
                        );
                    }
                    Err(error) => return Err(error),
                }

                match solve_for_source(precond_dyn, src_idx, source, false, &mut x_dev_64) {
                    Ok(Some((true, outer_iters, inner_iters, residual, rhs_dt, solve_dt))) => {
                        record_attempt(outer_iters, inner_iters, residual, rhs_dt, solve_dt);
                        log::info!(
                            "FEM (cuda {}) retry succeeded for source {} freq {:.1} Hz: outer={} inner={} res={:.2e}",
                            precision.label(),
                            source.name,
                            freq,
                            outer_iters,
                            inner_iters,
                            residual,
                        );
                        Ok(())
                    }
                    Ok(Some((false, outer_iters, inner_iters, residual, rhs_dt, solve_dt))) => {
                        record_attempt(outer_iters, inner_iters, residual, rhs_dt, solve_dt);
                        anyhow::bail!(
                            "FEM (cuda {}) did not converge at {:.1} Hz for source '{}' even after warm-start reset: outer={} inner={} res={:.2e}",
                            precision.label(),
                            freq,
                            source.name,
                            outer_iters,
                            inner_iters,
                            residual,
                        );
                    }
                    Ok(None) => anyhow::bail!(
                        "FEM (cuda {}) solve failed at {:.1} Hz for source '{}' even after warm-start reset",
                        precision.label(),
                        freq,
                        source.name,
                    ),
                    Err(error) => return Err(error),
                }
            };

            match active_precond {
                GpuPrecond::PointJacobi => {
                    let pj = cached_point_jacobi
                        .as_ref()
                        .expect("PointJacobi preconditioner cache populated before solve");
                    for (src_idx, source) in simulation.sources.iter().enumerate() {
                        solve_and_record(src_idx, source, pj)?;
                    }
                }
                GpuPrecond::PolynomialJacobi { .. } => {
                    let pj = cached_polynomial.as_ref()
                        .expect("polynomial Jacobi initialized before scalar solve");
                    for (src_idx, source) in simulation.sources.iter().enumerate() {
                        solve_and_record(src_idx, source, pj)?;
                    }
                }
                GpuPrecond::Chebyshev { .. } => {
                    let cheb = cached_chebyshev.as_ref()
                        .expect("Chebyshev initialized before scalar solve");
                    for (src_idx, source) in simulation.sources.iter().enumerate() {
                        solve_and_record(src_idx, source, cheb)?;
                    }
                }
                GpuPrecond::Auto | GpuPrecond::Amg => {
                    let amg = cached_amg.as_ref()
                        .expect("AMG initialized before scalar solve");
                    for (src_idx, source) in simulation.sources.iter().enumerate() {
                        solve_and_record(src_idx, source, amg)?;
                    }
                }
            }
        }

        // This checked boundary deliberately follows the complete device
        // solve for the frequency and precedes runtime accounting and
        // result publication. It exercises the same typed cancellation
        // path as an asynchronous request arriving during a solve while
        // guaranteeing that no partially filled output is returned.
        inject_mid_frequency_cancellation(cancellation, freq_idx);
        ensure_device_not_cancelled(cancellation, "cuda")?;

        let runtime_stats = backend.runtime_stats().since(runtime_start);
        let (operator_work, physical_power, boundary_flux) =
            if wants_full_solution_evidence(mesh_config) {
                let (work, power, flux) = finish_device_physical_evidence(
                    &mut operator_work,
                    boundary_power_by_tag,
                    freq,
                )?;
                (Some(work), Some(power), Some(flux))
            } else {
                (None, None, None)
            };
        diagnostics.record_frequency(FemFrequencyStats {
            block_solve,
            frequency_hz: freq,
            symmetry_mode: None,
            matrix_ms: duration_ms(matrix_time),
            preconditioner_ms: duration_ms(upload_time + preconditioner_time),
            rhs_ms: duration_ms(rhs_time),
            solve_ms: duration_ms(solve_time),
            iterations: total_outer,
            inner_iterations: total_inner,
            retries: retry_count,
            preconditioner_age: if rebuild_gpu_precond {
                0
            } else {
                freq_idx % rebuild_every
            },
            preconditioner_rebuild_reason: if rebuild_gpu_precond {
                "scheduled".to_string()
            } else {
                "cached".to_string()
            },
            fallback_stage: usize::from(retry_count > 0),
            max_residual,
            converged: true,
            operator_work,
            physical_power,
            boundary_flux,
            transfer_stats: Some(FemTransferStats::from(
                backend.transfer_stats().saturating_sub(transfer_start),
            )),
            allocation_operations: runtime_stats.allocation_operations,
            allocation_high_water_bytes: runtime_stats.allocation_high_water_bytes,
            synchronization_calls: runtime_stats.synchronization_calls,
        });

        if should_log_frequency(freq_idx, n_freqs) {
            log::info!(
                "FEM (cuda): frequency {}/{} ({:.1} Hz) done in {}: matrix={}, upload={}, RHS={}, solve={}, outer={}, inner={}, max_res={:.2e}",
                freq_idx + 1,
                n_freqs,
                freq,
                fmt_duration(freq_start.elapsed()),
                fmt_duration(matrix_time),
                fmt_duration(upload_time),
                fmt_duration(rhs_time),
                fmt_duration(solve_time),
                total_outer,
                total_inner,
                max_residual,
            );
        }
    }

    diagnostics.set_execution_transfers(
        setup_transfer_stats,
        FemTransferStats::from(backend.transfer_stats()),
    );

    // Observe cancellation after the final frequency as well. Otherwise
    // a request arriving during the last device solve could publish a
    // completed-looking response despite the caller's cancellation.
    ensure_device_not_cancelled(cancellation, "cuda")?;
    Ok((
        SimulationOutput::solved(
            simulation.frequencies.clone(),
            pressures,
            simulation.sources.iter().map(|s| s.name.clone()).collect(),
        ),
        diagnostics,
    ))
}

/// Run the FEM Helmholtz sweep on Apple Metal using Complex64 outer
/// FGMRES-IR and a Complex32 inner solve/preconditioner path.
#[cfg(feature = "metal")]
pub fn run_fem_metal(
    simulation: &RoomSimulation,
    mesh_config: &FemMeshConfig,
    precision: GpuPrecision,
    precond: GpuPrecond,
) -> Result<SimulationOutput> {
    Ok(run_fem_metal_report(simulation, mesh_config, precision, precond)?.0)
}

/// Run the FEM Helmholtz sweep on Apple Metal and return solver diagnostics
/// alongside the listener-pressure output.
#[cfg(feature = "metal")]
pub fn run_fem_metal_report(
    simulation: &RoomSimulation,
    mesh_config: &FemMeshConfig,
    precision: GpuPrecision,
    precond: GpuPrecond,
) -> Result<(SimulationOutput, FemRunDiagnostics)> {
    run_fem_metal_report_with_cancellation(simulation, mesh_config, precision, precond, None)
}

/// Run Metal FEM with cooperative frequency-boundary cancellation.
#[cfg(feature = "metal")]
pub fn run_fem_metal_report_cancellation(
    simulation: &RoomSimulation,
    mesh_config: &FemMeshConfig,
    precision: GpuPrecision,
    precond: GpuPrecond,
    cancellation: &FemRunCancellation,
) -> Result<(SimulationOutput, FemRunDiagnostics)> {
    run_fem_metal_report_with_cancellation(
        simulation,
        mesh_config,
        precision,
        precond,
        Some(cancellation),
    )
}

#[cfg(feature = "metal")]
fn run_fem_metal_report_with_cancellation(
    simulation: &RoomSimulation,
    mesh_config: &FemMeshConfig,
    precision: GpuPrecision,
    precond: GpuPrecond,
    cancellation: Option<&FemRunCancellation>,
) -> Result<(SimulationOutput, FemRunDiagnostics)> {
    validate_mesh_config(mesh_config)?;
    validate_geometry(simulation)?;
    validate_source_signal_ownership(simulation, mesh_config)?;
    ensure_device_not_cancelled(cancellation, "metal")?;
    anyhow::ensure!(
        precision == GpuPrecision::F64Outer,
        "Metal FEM prototype currently supports only --gpu-precision f64-outer"
    );
    let precond = precond
        .resolve_for_metal()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut diagnostics = FemRunDiagnostics::new_with_preconditioner(
        "metal",
        precond.label().to_string(),
        mesh_config.shifted_laplacian_shift,
        mesh_config.modal_damping_ratio,
    );
    anyhow::ensure!(
        !simulation.frequencies.is_empty(),
        "FEM solve requires at least one frequency"
    );

    let max_freq = simulation
        .frequencies
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let min_wavelength = simulation.speed_of_sound / max_freq;
    let elements_per_meter = (ELEMENTS_PER_WAVELENGTH / min_wavelength).ceil() as usize;

    log::info!(
        "FEM (metal): building mesh for max_freq={:.0} Hz, grading {:.1}x",
        max_freq,
        mesh_config.grading_factor,
    );
    let mesh_start = Instant::now();
    // Match the current CUDA prototype: GPU runners do not yet handle the
    // per-mode parity decomposition for symmetric scenes.
    let symmetry = SymmetryConfig::none();
    let mesh = create_room_mesh(simulation, elements_per_meter, &symmetry, mesh_config);
    log::info!(
        "FEM (metal) mesh: {} nodes, {} elements (fine spacing {} elem/m, grading {:.1}x, radius {:.2} m) built in {}",
        mesh.num_nodes(),
        mesh.num_elements(),
        elements_per_meter,
        mesh_config.grading_factor,
        mesh_config.fine_radius,
        fmt_duration(mesh_start.elapsed()),
    );
    let image_walls_aabb = mesh_aabb(&mesh);

    let assembly_start = Instant::now();
    let stiffness = assemble_stiffness(&mesh, PolynomialDegree::P1);
    let mass = assemble_mass(&mesh, PolynomialDegree::P1);
    let phys_markers = [
        MARKER_FLOOR,
        MARKER_CEILING,
        MARKER_FRONT,
        MARKER_BACK,
        MARKER_LEFT,
        MARKER_RIGHT,
    ];
    let mut boundary_matrices = Vec::new();
    for &marker in &phys_markers {
        let b_mass = assemble_boundary_mass(&mesh, PolynomialDegree::P1, marker);
        if b_mass.nnz() > 0 {
            boundary_matrices.push((marker as usize, b_mass));
        }
    }
    let assembler = HelmholtzAssembler::from_matrices(&stiffness, &mass, &boundary_matrices);
    log::info!(
        "FEM (metal): assembled K nnz={}, M nnz={}, {} boundary matrices in {}",
        stiffness.nnz(),
        mass.nnz(),
        boundary_matrices.len(),
        fmt_duration(assembly_start.elapsed()),
    );

    let resolved_boundaries = resolve_boundaries(simulation)?;

    let lp_elements: Vec<Option<usize>> = simulation
        .listening_positions
        .par_iter()
        .map(|lp| find_containing_element_parallel(&mesh, *lp))
        .collect();

    let backend = MetalBackend::new().map_err(|e| {
        anyhow::Error::new(FemBackendExecutionError::DeviceUnavailable {
            backend: "metal".to_string(),
            detail: format!("initialization: {e:?}"),
        })
    })?;
    diagnostics.set_execution_profile("metal", "metal", precision.label(), "none");
    log::info!(
        "FEM (metal): Metal backend acquired; c64 outer operations use the backend fallback policy"
    );

    let n_sources = simulation.sources.len();
    let n_lps = simulation.listening_positions.len();
    let n_freqs = simulation.frequencies.len();
    let n_nodes = mesh.num_nodes();

    let mut pressures: Vec<Vec<Vec<Complex64>>> =
        vec![vec![vec![Complex64::new(0.0, 0.0); n_freqs]; n_lps]; n_sources];

    let (lp_interp_indices, lp_interp_weights_64) =
        build_listener_interpolation(&mesh, &simulation.listening_positions, &lp_elements)?;
    let listener_nodes: Vec<[usize; 4]> = lp_interp_indices
        .chunks_exact(4)
        .map(|nodes| {
            [
                nodes[0] as usize,
                nodes[1] as usize,
                nodes[2] as usize,
                nodes[3] as usize,
            ]
        })
        .collect();
    let listener_weights: Vec<[f64; 4]> = lp_interp_weights_64
        .chunks_exact(4)
        .map(|weights| [weights[0], weights[1], weights[2], weights[3]])
        .collect();
    let mut listener = DeviceListenerExtraction::<MetalBackend, Complex64>::new(
        &backend,
        &listener_nodes,
        &listener_weights,
    )?;
    let mut lp_values_host = vec![Complex64::new(0.0, 0.0); n_lps];

    let ir_policy = MixedPrecisionPolicy {
        outer_tolerance: GPU_IR_OUTER_TOLERANCE,
        inner_tolerance: GPU_IR_INNER_TOLERANCE,
        true_residual_check_period: GPU_IR_TRUE_RESIDUAL_CHECK_PERIOD,
        physical_tolerance: GPU_IR_OUTER_TOLERANCE,
    };
    let ir_cfg = GmresIrConfig::<Complex64> {
        max_outer_iterations: GPU_IR_MAX_OUTER_ITERATIONS,
        outer_tolerance: ir_policy.outer_tolerance,
        max_inner_iterations: GPU_IR_MAX_INNER_ITERATIONS,
        inner_tolerance: ir_policy.inner_tolerance,
        restart_size: Some(GPU_IR_RESTART_SIZE),
        adaptive_inner_tolerance: true,
        true_residual_check_period: Some(ir_policy.true_residual_check_period),
    };

    let mesh_nnz = assembler.k_values.len();
    let basis_count = 2_u64 + assembler.boundary_values.len() as u64;
    let bytes_basis_c32 = basis_count * (mesh_nnz * 8) as u64;
    let bytes_basis_c64 = basis_count * (mesh_nnz * 16) as u64;
    let bytes_a_inner = (mesh_nnz * (8 + 4)) as u64 + ((n_nodes + 1) * 4) as u64;
    let bytes_p_inner = match precond {
        GpuPrecond::PolynomialJacobi { .. } | GpuPrecond::Chebyshev { .. } => bytes_a_inner,
        GpuPrecond::PointJacobi => 0,
        GpuPrecond::Auto | GpuPrecond::Amg => unreachable!("Metal preconditioner resolved"),
    };
    let bytes_pointjacobi = (n_nodes * (8 + 4 + 4)) as u64;
    let bytes_precond_scratch = match precond {
        GpuPrecond::Chebyshev { .. } => (3 * n_nodes * 8) as u64,
        GpuPrecond::PointJacobi | GpuPrecond::PolynomialJacobi { .. } => 0,
        GpuPrecond::Auto | GpuPrecond::Amg => unreachable!("Metal preconditioner resolved"),
    };
    let bytes_a_outer = (mesh_nnz * (16 + 4)) as u64 + ((n_nodes + 1) * 4) as u64;
    let restart_outer = ir_cfg.restart_size.unwrap_or(ir_cfg.max_outer_iterations);
    let bytes_outer_basis = 2 * (restart_outer * n_nodes * 16) as u64;
    let bytes_inner_basis_ir = (ir_cfg.max_inner_iterations * n_nodes * 8) as u64;
    let bytes_scratch_mixed = (5 * n_nodes * 16) as u64 + (3 * n_nodes * 8) as u64;
    let peak_gpu_bytes = bytes_a_outer
        + bytes_a_inner
        + bytes_basis_c64
        + bytes_basis_c32
        + bytes_p_inner
        + bytes_pointjacobi
        + bytes_precond_scratch
        + bytes_outer_basis
        + bytes_inner_basis_ir
        + bytes_scratch_mixed;
    log::info!(
        "FEM (metal, {}, {}): estimated peak unified-memory footprint per frequency: {} MiB \
         (A_outer {} MiB + A_inner {} MiB + basis64 {} MiB + basis32 {} MiB + P {} MiB + diag {} MiB + precond scratch {} MiB + outer basis {} MiB + inner basis {} MiB + scratch {} MiB)",
        precision.label(),
        precond.label(),
        peak_gpu_bytes / (1024 * 1024),
        bytes_a_outer / (1024 * 1024),
        bytes_a_inner / (1024 * 1024),
        bytes_basis_c64 / (1024 * 1024),
        bytes_basis_c32 / (1024 * 1024),
        bytes_p_inner / (1024 * 1024),
        bytes_pointjacobi / (1024 * 1024),
        bytes_precond_scratch / (1024 * 1024),
        bytes_outer_basis / (1024 * 1024),
        bytes_inner_basis_ir / (1024 * 1024),
        bytes_scratch_mixed / (1024 * 1024),
    );

    log::info!(
        "FEM (metal, {}): solving {} sources × {} frequencies with {} preconditioner",
        precision.label(),
        n_sources,
        n_freqs,
        precond.label(),
    );
    log::info!(
        "FEM (metal, {}): solver budget: ir_outer<={} ir_inner<={} ir_restart={} ir_tol={:.1e}",
        precision.label(),
        ir_cfg.max_outer_iterations,
        ir_cfg.max_inner_iterations,
        ir_cfg.restart_size.unwrap_or(ir_cfg.max_outer_iterations),
        ir_cfg.outer_tolerance,
    );
    diagnostics.set_execution_memory(peak_gpu_bytes, None, None);
    let rebuild_every = mesh_config.rebuild_precond_every.max(1);
    let mut cached_point_jacobi: Option<PointJacobiBackend<MetalBackend, Complex32>> = None;
    log::info!(
        "FEM (metal): rebuilding reusable GPU preconditioners every {} freq{}",
        rebuild_every,
        if rebuild_every == 1 { "" } else { "s" },
    );

    let structure_upload_start = Instant::now();
    let row_ptrs_i32 = usize_indices_to_i32(&assembler.row_ptrs, "row_ptrs")?;
    let col_idx_i32 = usize_indices_to_i32(&assembler.col_indices, "col_indices")?;
    let mut a_inner = BackendCsr::<MetalBackend, Complex32>::try_from_host_structure(
        &backend,
        n_nodes,
        n_nodes,
        &row_ptrs_i32,
        &col_idx_i32,
    )
    .map_err(|error| {
        anyhow::anyhow!("Metal structure allocation/upload failed for A32: {error}")
    })?;
    let mut a_outer = BackendCsr::<MetalBackend, Complex64>::try_from_host_structure(
        &backend,
        n_nodes,
        n_nodes,
        &row_ptrs_i32,
        &col_idx_i32,
    )
    .map_err(|error| {
        anyhow::anyhow!("Metal structure allocation/upload failed for A64: {error}")
    })?;
    let mut p_inner_opt: Option<BackendCsr<MetalBackend, Complex32>> = match precond {
        GpuPrecond::PolynomialJacobi { .. } | GpuPrecond::Chebyshev { .. } => Some(
            BackendCsr::<MetalBackend, Complex32>::try_from_host_structure(
                &backend,
                n_nodes,
                n_nodes,
                &row_ptrs_i32,
                &col_idx_i32,
            )
            .map_err(|error| {
                anyhow::anyhow!("Metal structure allocation/upload failed for P32: {error}")
            })?,
        ),
        GpuPrecond::PointJacobi => None,
        GpuPrecond::Auto | GpuPrecond::Amg => unreachable!("Metal preconditioner resolved"),
    };
    backend.synchronize()?;
    log::info!(
        "FEM (metal): uploaded reusable CSR structure nnz={} in {}",
        mesh_nnz,
        fmt_duration(structure_upload_start.elapsed()),
    );

    let basis_upload_start = Instant::now();
    let k_basis_32 = upload_real_basis_as_complex32(&backend, &assembler.k_values, "K")?;
    let m_basis_32 = upload_real_basis_as_complex32(&backend, &assembler.m_values, "M")?;
    let boundary_basis_32 =
        upload_boundary_basis_as_complex32(&backend, &assembler.boundary_values)?;
    let k_basis_64 = upload_real_basis_as_complex64(&backend, &assembler.k_values, "K")?;
    let m_basis_64 = upload_real_basis_as_complex64(&backend, &assembler.m_values, "M")?;
    let boundary_basis_64 =
        upload_boundary_basis_as_complex64(&backend, &assembler.boundary_values)?;
    let diag_basis_32 = build_diag_basis_32(
        &assembler.row_ptrs,
        &assembler.col_indices,
        &assembler.k_values,
        &assembler.m_values,
        n_nodes,
    );
    let mut p_diag_32 = vec![Complex32::new(0.0, 0.0); n_nodes];
    let mut p_values_host_32 = match precond {
        GpuPrecond::Chebyshev { .. } => vec![Complex32::new(0.0, 0.0); mesh_nnz],
        GpuPrecond::PointJacobi | GpuPrecond::PolynomialJacobi { .. } => Vec::new(),
        GpuPrecond::Auto | GpuPrecond::Amg => unreachable!("Metal preconditioner resolved"),
    };
    backend.synchronize()?;
    log::info!(
        "FEM (metal): uploaded reusable value bases ({} vectors, nnz={}) in {}",
        basis_count,
        mesh_nnz,
        fmt_duration(basis_upload_start.elapsed()),
    );

    let mut rhs_dev_64 = gpu_alloc(&backend, n_nodes, "rhs/residual f64")?;
    let mut x_dev_64 = gpu_alloc(&backend, n_nodes, "x/residual f64")?;
    let mut prev_x_64 = (0..n_sources)
        .map(|src| gpu_alloc(&backend, n_nodes, &format!("prev x f64 src {src}")))
        .collect::<Result<Vec<_>>>()?;
    let mut prev_solution_valid = vec![false; n_sources];

    let rhs_basis_start = Instant::now();
    let rhs_bases: Vec<SourceRhsBasis> = simulation
        .sources
        .iter()
        .map(|source| {
            build_source_rhs_basis_for_model(
                &mesh,
                source,
                image_walls_aabb,
                mesh_config.source_model,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    log::info!(
        "FEM (metal): precomputed {} source RHS bases in {}",
        rhs_bases.len(),
        fmt_duration(rhs_basis_start.elapsed()),
    );
    let mut rhs_assemblers: Vec<DeviceRhsAssembler<MetalBackend, Complex64>> = rhs_bases
        .iter()
        .map(|basis| {
            let entries: Vec<_> = basis
                .entries
                .iter()
                // Host coefficients already include basis weights; keep
                // the device matrix as a unit-valued scatter.
                .map(|&(node, _, _)| (node, 1.0_f64))
                .collect();
            DeviceRhsAssembler::new(&backend, n_nodes, &entries)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut rhs_coefficients: Vec<Vec<Complex64>> = rhs_bases
        .iter()
        .map(|basis| vec![Complex64::new(0.0, 0.0); basis.entries.len()])
        .collect();
    let setup_transfer_stats = FemTransferStats::from(backend.transfer_stats());
    backend.reset_transfer_stats();

    for (freq_idx, &freq) in simulation.frequencies.iter().enumerate() {
        ensure_device_not_cancelled(cancellation, "metal")?;
        let transfer_start = backend.transfer_stats();
        let freq_start = Instant::now();
        let k = simulation.wavenumber(freq);
        let k_complex = complex_wavenumber(k, freq, mesh_config.modal_damping_ratio);
        let rebuild_gpu_precond = match precond {
            GpuPrecond::PointJacobi => {
                cached_point_jacobi.is_none() || freq_idx % rebuild_every == 0
            }
            GpuPrecond::PolynomialJacobi { .. } | GpuPrecond::Chebyshev { .. } => true,
            GpuPrecond::Auto | GpuPrecond::Amg => unreachable!("Metal preconditioner resolved"),
        };

        if should_log_frequency(freq_idx, n_freqs) {
            log::info!(
                "FEM (metal): frequency {}/{} ({:.1} Hz) started",
                freq_idx + 1,
                n_freqs,
                freq,
            );
        }

        let matrix_start = Instant::now();
        let boundary_coeffs =
            compute_boundary_coefficients(simulation, &resolved_boundaries, freq)?;
        let p_shift = mesh_config.shifted_laplacian_shift.value(k);
        let k_sq = k_complex * k_complex;
        fill_shifted_diag_32(&mut p_diag_32, &diag_basis_32, p_shift);
        if matches!(precond, GpuPrecond::Chebyshev { .. }) {
            fill_shifted_values_32(
                &mut p_values_host_32,
                &assembler.k_values,
                &assembler.m_values,
                p_shift,
            );
        }
        let matrix_time = matrix_start.elapsed();

        let upload_start = Instant::now();
        update_helmholtz_values_32(
            &backend,
            &mut a_inner.values,
            &k_basis_32,
            &m_basis_32,
            &boundary_basis_32,
            k_sq,
            &boundary_coeffs,
        );
        update_helmholtz_values_64(
            &backend,
            &mut a_outer.values,
            &k_basis_64,
            &m_basis_64,
            &boundary_basis_64,
            k_sq,
            &boundary_coeffs,
        );
        if let Some(p_inner) = p_inner_opt.as_mut() {
            update_shifted_values_32(
                &backend,
                &mut p_inner.values,
                &k_basis_32,
                &m_basis_32,
                p_shift,
            );
        }
        backend.synchronize()?;
        let upload_time = upload_start.elapsed();

        let mut rhs_time = Duration::ZERO;
        let mut solve_time = Duration::ZERO;
        let mut total_outer = 0usize;
        let mut total_inner = 0usize;
        let mut max_residual = 0.0f64;
        let mut retry_count = 0usize;
        let mut operator_work = FemOperatorWorkEvidence {
            source_work: 0.0,
            volume_damping_work: 0.0,
            boundary_work: 0.0,
            balance_residual: 0.0,
        };
        let mut boundary_power_by_tag = BTreeMap::new();

        type SolveResult = (bool, usize, usize, f64, Duration, Duration);
        let mut solve_for_source = |precond_dyn: &dyn BackendPreconditioner<MetalBackend, Complex32>,
                                    src_idx: usize,
                                    source: &Source,
                                    use_warm_start: bool,
                                    x_dev_64: &mut <MetalBackend as Backend>::Buffer<Complex64>|
         -> Result<Option<SolveResult>> {
            if !use_warm_start {
                prev_solution_valid[src_idx] = false;
            }
            let rhs_t = Instant::now();
            assemble_source_coefficients_into(
                &rhs_bases[src_idx],
                source,
                freq,
                &mut rhs_coefficients[src_idx],
            );
            let rhs_dt = rhs_t.elapsed();

            let solve_t = Instant::now();
            rhs_assemblers[src_idx].assemble(&backend, &rhs_coefficients[src_idx], &mut rhs_dev_64)?;
            let x0_dev = if use_warm_start && prev_solution_valid[src_idx] {
                Some(&prev_x_64[src_idx])
            } else {
                None
            };
            let r = gmres_ir_backend_preconditioned::<
                MetalBackend,
                BackendCsr<MetalBackend, Complex64>,
                BackendCsr<MetalBackend, Complex32>,
                dyn BackendPreconditioner<MetalBackend, Complex32>,
                Complex64,
            >(
                &backend,
                &a_outer,
                &a_inner,
                precond_dyn,
                &rhs_dev_64,
                x0_dev,
                x_dev_64,
                &ir_cfg,
            );
            backend.synchronize()?;
            let result = match r {
                Ok(status) => {
                    if status.converged {
                        backend.copy_d2d(&x_dev_64, &mut prev_x_64[src_idx]);
                        prev_solution_valid[src_idx] = true;
                    }
                    if log::log_enabled!(log::Level::Debug) {
                        log::debug!(
                            "FEM (metal {}) source {} freq {:.1} Hz residual_history={:?} true_residual_checks={:?}",
                            precision.label(),
                            source.name,
                            freq,
                            status.residual_history,
                            status.true_residual_checks,
                        );
                    }
                    lp_values_host = listener.evaluate_to_host(&backend, x_dev_64);
                    if status.converged && wants_full_solution_evidence(mesh_config) {
                        collect_device_solution_evidence(
                            &backend,
                            &*x_dev_64,
                            &assembler,
                            k_complex,
                            &boundary_coeffs,
                            &rhs_bases[src_idx],
                            source,
                            freq,
                            &mut operator_work,
                            &mut boundary_power_by_tag,
                        )
                        .map_err(|error| {
                            anyhow::Error::new(FemBackendExecutionError::PhysicalEvidence {
                                backend: "metal".to_string(),
                                detail: format!(
                                    "source '{}' at {:.1} Hz: {error}",
                                    source.name, freq
                                ),
                            })
                        })?;
                    }
                    Ok((
                        status.converged,
                        status.outer_iterations,
                        status.total_inner_iterations,
                        status.final_outer_residual,
                    ))
                }
                Err(e) => Err(anyhow::anyhow!("{e}")),
            };
            let solve_dt = solve_t.elapsed();
            match result {
                Ok((conv, oi, ii, res)) => {
                    store_listener_pressures(
                        &simulation.listening_positions,
                        &lp_values_host,
                        source,
                        freq,
                        &mut pressures[src_idx],
                        freq_idx,
                    );
                    Ok(Some((conv, oi, ii, res, rhs_dt, solve_dt)))
                }
                Err(e) => {
                    log::warn!(
                        "FEM (metal {}) solve failed for source {} freq {:.1} Hz: {e}",
                        precision.label(),
                        source.name,
                        freq,
                    );
                    Ok(None)
                }
            }
        };

        let mut solve_and_record = |src_idx: usize,
                                    source: &Source,
                                    precond_dyn: &dyn BackendPreconditioner<MetalBackend, Complex32>|
         -> Result<()> {
            let mut record_attempt = |outer_iters: usize,
                                      inner_iters: usize,
                                      residual: f64,
                                      rhs_dt: Duration,
                                      solve_dt: Duration| {
                rhs_time += rhs_dt;
                solve_time += solve_dt;
                total_outer += outer_iters;
                total_inner += inner_iters;
                max_residual = max_residual.max(residual);
            };

            match solve_for_source(precond_dyn, src_idx, source, true, &mut x_dev_64)? {
                Some((true, outer_iters, inner_iters, residual, rhs_dt, solve_dt)) => {
                    record_attempt(outer_iters, inner_iters, residual, rhs_dt, solve_dt);
                    return Ok(());
                }
                Some((false, outer_iters, inner_iters, residual, rhs_dt, solve_dt)) => {
                    record_attempt(outer_iters, inner_iters, residual, rhs_dt, solve_dt);
                    retry_count += 1;
                    log::warn!(
                        "FEM (metal {}) did not converge for source {} freq {:.1} Hz: outer={} inner={} res={:.2e} (retrying without warm start)",
                        precision.label(),
                        source.name,
                        freq,
                        outer_iters,
                        inner_iters,
                        residual,
                    );
                }
                None => {
                    retry_count += 1;
                    log::warn!(
                        "FEM (metal {}) failed for source {} freq {:.1} Hz (retrying without warm start)",
                        precision.label(),
                        source.name,
                        freq,
                    );
                }
            }

            match solve_for_source(precond_dyn, src_idx, source, false, &mut x_dev_64)? {
                Some((true, outer_iters, inner_iters, residual, rhs_dt, solve_dt)) => {
                    record_attempt(outer_iters, inner_iters, residual, rhs_dt, solve_dt);
                    log::info!(
                        "FEM (metal {}) retry succeeded for source {} freq {:.1} Hz: outer={} inner={} res={:.2e}",
                        precision.label(),
                        source.name,
                        freq,
                        outer_iters,
                        inner_iters,
                        residual,
                    );
                    Ok(())
                }
                Some((false, outer_iters, inner_iters, residual, rhs_dt, solve_dt)) => {
                    record_attempt(outer_iters, inner_iters, residual, rhs_dt, solve_dt);
                    anyhow::bail!(
                        "FEM (metal {}) did not converge at {:.1} Hz for source '{}' even after warm-start reset: outer={} inner={} res={:.2e}",
                        precision.label(),
                        freq,
                        source.name,
                        outer_iters,
                        inner_iters,
                        residual,
                    );
                }
                None => anyhow::bail!(
                    "FEM (metal {}) solve failed at {:.1} Hz for source '{}' even after warm-start reset",
                    precision.label(),
                    freq,
                    source.name,
                ),
            }
        };

        match precond {
            GpuPrecond::PointJacobi => {
                if rebuild_gpu_precond {
                    cached_point_jacobi = Some(
                        PointJacobiBackend::<MetalBackend, Complex32>::from_diagonal(
                            &backend, &p_diag_32,
                        ),
                    );
                }
                let pj = cached_point_jacobi
                    .as_ref()
                    .expect("PointJacobi preconditioner cache populated before solve");
                for (src_idx, source) in simulation.sources.iter().enumerate() {
                    solve_and_record(src_idx, source, pj)?;
                }
            }
            GpuPrecond::PolynomialJacobi { degree } => {
                let p_inner = p_inner_opt
                    .as_ref()
                    .expect("PolyJacobi mode requires P_inner; bug if missing");
                let pj = PolynomialJacobiBackend::<MetalBackend, Complex32>::new(
                    &backend,
                    p_inner,
                    &p_diag_32,
                    degree,
                    Complex32::new(POLY_JACOBI_DAMPING, 0.0),
                );
                for (src_idx, source) in simulation.sources.iter().enumerate() {
                    solve_and_record(src_idx, source, &pj)?;
                }
            }
            GpuPrecond::Chebyshev { degree } => {
                let p_inner = p_inner_opt
                    .as_ref()
                    .expect("Chebyshev mode requires P_inner; bug if missing");
                let cheb = ChebyshevBackend::<MetalBackend, Complex32>::from_csr_host(
                    &backend,
                    p_inner,
                    &row_ptrs_i32,
                    &col_idx_i32,
                    &p_values_host_32,
                    degree,
                    CHEBYSHEV_EIGENVALUE_RATIO,
                );
                log::debug!(
                    "FEM (metal): Chebyshev(P) degree={} lambda_max={:.3e} ratio={:.1}",
                    cheb.degree(),
                    cheb.lambda_max(),
                    cheb.eigenvalue_ratio(),
                );
                for (src_idx, source) in simulation.sources.iter().enumerate() {
                    solve_and_record(src_idx, source, &cheb)?;
                }
            }
            GpuPrecond::Auto | GpuPrecond::Amg => {
                unreachable!("Metal preconditioner resolved before solve")
            }
        }

        let (operator_work, physical_power, boundary_flux) =
            if wants_full_solution_evidence(mesh_config) {
                let (work, power, flux) = finish_device_physical_evidence(
                    &mut operator_work,
                    boundary_power_by_tag,
                    freq,
                )?;
                (Some(work), Some(power), Some(flux))
            } else {
                (None, None, None)
            };
        diagnostics.record_frequency(FemFrequencyStats {
            block_solve: Default::default(),
            frequency_hz: freq,
            symmetry_mode: None,
            matrix_ms: duration_ms(matrix_time),
            preconditioner_ms: duration_ms(upload_time),
            rhs_ms: duration_ms(rhs_time),
            solve_ms: duration_ms(solve_time),
            iterations: total_outer,
            inner_iterations: total_inner,
            retries: retry_count,
            preconditioner_age: 0,
            preconditioner_rebuild_reason: if rebuild_gpu_precond {
                "scheduled".to_string()
            } else {
                "cached".to_string()
            },
            fallback_stage: usize::from(retry_count > 0),
            max_residual,
            converged: true,
            operator_work,
            physical_power,
            boundary_flux,
            transfer_stats: Some(FemTransferStats::from(
                backend.transfer_stats().saturating_sub(transfer_start),
            )),
            allocation_operations: 0,
            allocation_high_water_bytes: 0,
            synchronization_calls: 0,
        });

        if should_log_frequency(freq_idx, n_freqs) {
            log::info!(
                "FEM (metal): frequency {}/{} ({:.1} Hz) done in {}: matrix={}, upload={}, RHS={}, solve={}, outer={}, inner={}, max_res={:.2e}",
                freq_idx + 1,
                n_freqs,
                freq,
                fmt_duration(freq_start.elapsed()),
                fmt_duration(matrix_time),
                fmt_duration(upload_time),
                fmt_duration(rhs_time),
                fmt_duration(solve_time),
                total_outer,
                total_inner,
                max_residual,
            );
        }
    }

    diagnostics.set_execution_transfers(
        setup_transfer_stats,
        FemTransferStats::from(backend.transfer_stats()),
    );

    // Keep the device contract atomic even when cancellation arrives
    // during the final frequency solve.
    ensure_device_not_cancelled(cancellation, "metal")?;
    Ok((
        SimulationOutput::solved(
            simulation.frequencies.clone(),
            pressures,
            simulation.sources.iter().map(|s| s.name.clone()).collect(),
        ),
        diagnostics,
    ))
}
