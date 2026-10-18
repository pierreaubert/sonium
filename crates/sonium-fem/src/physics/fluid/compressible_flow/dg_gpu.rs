//! Device-resident GPU counterpart of the compressible DG solver.
//!
//! [`CompressibleDgGpuSolver`] mirrors
//! [`CompressibleDgSolver`](super::CompressibleDgSolver) step for step —
//! same SSPRK3 stages, limiter, Zhang--Shu positivity, and retry
//! semantics — but keeps the state on the device: the caller uploads
//! the initial state once with
//! [`set_state`](CompressibleDgGpuSolver::set_state), then each
//! [`step`](CompressibleDgGpuSolver::step) moves only scalars (stable
//! time step, retry flags, conserved totals, min thermodynamics) across
//! the bus. [`fetch_state`](CompressibleDgGpuSolver::fetch_state)
//! downloads the full state only when the caller needs it (output,
//! probing), never per step.

// Rust guideline compliant 2026-02-21

use sonium_compute::cuda::CudaBackend;
use sonium_compute::cuda::dg::{CudaDgOperator, DgSlot};
use sonium_compute::error::BackendError;
use sonium_compute::{MemoryPool, MemoryPoolConfig};

use crate::mesh::Mesh;

use super::{
    CompressibleDgConfig, CompressibleDgError, CompressibleDgSolver, CompressibleDgState,
    CompressibleStepReport, ConservativeState,
};

/// Failure of the GPU DG path: a DG-semantics error identical to the
/// CPU solver's, or a backend (driver/NVRTC/pool) failure.
#[derive(Debug)]
pub enum CompressibleDgGpuError {
    /// DG-semantics failure, same variants as the CPU solver.
    Dg(CompressibleDgError),
    /// CUDA backend failure while staging or launching.
    Backend(BackendError),
}

impl From<CompressibleDgError> for CompressibleDgGpuError {
    fn from(error: CompressibleDgError) -> Self {
        Self::Dg(error)
    }
}

impl From<BackendError> for CompressibleDgGpuError {
    fn from(error: BackendError) -> Self {
        Self::Backend(error)
    }
}

impl From<sonium_compute::error::OomError> for CompressibleDgGpuError {
    fn from(error: sonium_compute::error::OomError) -> Self {
        Self::Backend(error.into())
    }
}

impl core::fmt::Display for CompressibleDgGpuError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Dg(error) => write!(formatter, "DG error: {error:?}"),
            Self::Backend(error) => write!(formatter, "CUDA backend error: {error}"),
        }
    }
}

impl std::error::Error for CompressibleDgGpuError {}

/// Pool quota for one DG operator: problem tables and six state-sized
/// buffers stay far below this on every gated mesh, and the pool fails
/// fast instead of over-committing device memory.
const DG_POOL_CAP_BYTES: usize = 1 << 32;

/// GPU compressible DG solver with a device-resident state.
pub struct CompressibleDgGpuSolver<'a> {
    solver: CompressibleDgSolver<'a>,
    gpu: Option<CudaDgOperator>,
}

impl<'a> CompressibleDgGpuSolver<'a> {
    /// Build the CPU operator (mesh validation and config checks are
    /// shared with the CPU solver); the device operator is staged on
    /// the first [`set_state`](Self::set_state) call.
    ///
    /// # Errors
    ///
    /// Returns [`CompressibleDgGpuError::Dg`] when the mesh or config
    /// fails CPU-side validation.
    pub fn new(
        mesh: &'a Mesh,
        config: CompressibleDgConfig,
    ) -> Result<Self, CompressibleDgGpuError> {
        Ok(Self {
            solver: CompressibleDgSolver::new(mesh, config)?,
            gpu: None,
        })
    }

    /// Upload the state, staging the device operator on first use.
    ///
    /// This is the only bulk host-to-device transfer of a run (besides
    /// problem staging itself); later calls re-upload into the current
    /// slot, which restarts the device state from the host copy.
    ///
    /// # Errors
    ///
    /// Returns [`CompressibleDgGpuError`] when staging or the upload
    /// fails.
    pub fn set_state(&mut self, state: &CompressibleDgState) -> Result<(), CompressibleDgGpuError> {
        if self.gpu.is_none() {
            let problem = self.solver.operator.build_cuda_problem(state)?;
            let backend = CudaBackend::new(0)?;
            let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(DG_POOL_CAP_BYTES))?;
            self.gpu = Some(CudaDgOperator::load(pool, problem)?);
        }
        let gpu = self.gpu.as_mut().expect("staged above");
        let flat: Vec<f64> = state
            .values
            .iter()
            .flat_map(|value| value.as_array())
            .collect();
        gpu.upload_state(DgSlot::Current, &flat)?;
        Ok(())
    }

    /// Download the current device state into `state`.
    ///
    /// # Errors
    ///
    /// Returns [`CompressibleDgGpuError`] when no state was staged or
    /// the copy-back fails.
    pub fn fetch_state(
        &mut self,
        state: &mut CompressibleDgState,
    ) -> Result<(), CompressibleDgGpuError> {
        let gpu = self.require_gpu()?;
        let mut flat = vec![0.0; state.values.len() * 5];
        gpu.download_state(DgSlot::Current, &mut flat)?;
        for (value, chunk) in state.values.iter_mut().zip(flat.as_chunks::<5>().0) {
            *value = ConservativeState::from_array(*chunk);
        }
        Ok(())
    }

    /// Spatial operator of the device current state (parity checks).
    ///
    /// Uploads nothing: the current slot must already hold the state
    /// (see [`set_state`](Self::set_state)).
    ///
    /// # Errors
    ///
    /// Returns [`CompressibleDgGpuError::Dg`] with
    /// `NonPhysicalQuadratureState` on the CPU error path, or a
    /// backend failure.
    pub fn spatial_operator(&mut self) -> Result<Vec<ConservativeState>, CompressibleDgGpuError> {
        let gpu = self.require_gpu()?;
        if gpu.spatial_operator(DgSlot::Current)? {
            return Err(CompressibleDgError::NonPhysicalQuadratureState.into());
        }
        let mut flat = vec![0.0; gpu.total_slots() * 5];
        gpu.download_rate(&mut flat)?;
        Ok(flat
            .as_chunks::<5>()
            .0
            .iter()
            .map(|chunk| ConservativeState::from_array(*chunk))
            .collect())
    }

    /// Stable time step of the device current state.
    ///
    /// # Errors
    ///
    /// Returns [`CompressibleDgGpuError::Dg`] with
    /// `NonPhysicalCellAverage` when a cell average is nonphysical.
    pub fn stable_time_step(&mut self) -> Result<f64, CompressibleDgGpuError> {
        let gpu = self.require_gpu()?;
        let (dt, nonphysical) = gpu.stable_dt(DgSlot::Current)?;
        if nonphysical {
            return Err(CompressibleDgError::NonPhysicalCellAverage.into());
        }
        Ok(dt)
    }

    /// Advance one SSPRK3 step on the device.
    ///
    /// Mirrors [`CompressibleDgSolver::step`]: the accepted step is
    /// `min(requested, stable)`, and a nonphysical trial halves the
    /// step up to the configured retry limit.
    ///
    /// # Errors
    ///
    /// Returns [`CompressibleDgGpuError::Dg`] with the same variants
    /// the CPU step returns (`NonPhysicalCellAverage`,
    /// `StepRetriesExhausted`), or a backend failure.
    pub fn step(
        &mut self,
        requested_time_step: f64,
    ) -> Result<CompressibleStepReport, CompressibleDgGpuError> {
        let max_retries = self.solver.operator.config.maximum_step_retries;
        let gpu = self.require_gpu()?;
        let before = gpu.totals(DgSlot::Current)?;
        let (stable, nonphysical) = gpu.stable_dt(DgSlot::Current)?;
        if nonphysical {
            return Err(CompressibleDgError::NonPhysicalCellAverage.into());
        }
        let mut dt = requested_time_step.min(stable);
        gpu.copy_slot(DgSlot::Original, DgSlot::Current)?;
        for retry in 0..max_retries {
            match self.trial(dt) {
                Ok(()) => {
                    let gpu = self.require_gpu()?;
                    let (minimum_density, minimum_pressure) = gpu.min_thermo(DgSlot::StageA)?;
                    gpu.copy_slot(DgSlot::Current, DgSlot::StageA)?;
                    let after = gpu.totals(DgSlot::Current)?;
                    return Ok(CompressibleStepReport {
                        accepted_time_step: dt,
                        retries: retry,
                        minimum_density,
                        minimum_pressure,
                        conserved_before: before,
                        conserved_after: after,
                    });
                }
                Err(CompressibleDgGpuError::Dg(
                    CompressibleDgError::NonPhysicalCellAverage
                    | CompressibleDgError::NonPhysicalQuadratureState,
                )) => dt *= 0.5,
                Err(error) => return Err(error),
            }
        }
        Err(CompressibleDgError::StepRetriesExhausted.into())
    }

    /// One SSPRK3 trial from the `Original` slot into `StageA`.
    fn trial(&mut self, dt: f64) -> Result<(), CompressibleDgGpuError> {
        let gpu = self.require_gpu()?;
        // Stage 1: u1 = u0 + dt * L(u0), then limit.
        if gpu.spatial_operator(DgSlot::Original)? {
            return Err(CompressibleDgError::NonPhysicalQuadratureState.into());
        }
        gpu.add_scaled(DgSlot::StageA, DgSlot::Original, dt)?;
        self.limit_slot(DgSlot::StageA)?;
        // Stage 2: u2 = 3/4 u0 + 1/4 (u1 + dt * L(u1)), then limit.
        let gpu = self.require_gpu()?;
        if gpu.spatial_operator(DgSlot::StageA)? {
            return Err(CompressibleDgError::NonPhysicalQuadratureState.into());
        }
        gpu.add_scaled(DgSlot::StageB, DgSlot::StageA, dt)?;
        gpu.convex(DgSlot::StageA, 0.75, DgSlot::Original, 0.25, DgSlot::StageB)?;
        self.limit_slot(DgSlot::StageA)?;
        // Stage 3: u3 = 1/3 u0 + 2/3 (u2 + dt * L(u2)), then limit.
        let gpu = self.require_gpu()?;
        if gpu.spatial_operator(DgSlot::StageA)? {
            return Err(CompressibleDgError::NonPhysicalQuadratureState.into());
        }
        gpu.add_scaled(DgSlot::StageB, DgSlot::StageA, dt)?;
        gpu.convex(
            DgSlot::StageA,
            1.0 / 3.0,
            DgSlot::Original,
            2.0 / 3.0,
            DgSlot::StageB,
        )?;
        self.limit_slot(DgSlot::StageA)?;
        Ok(())
    }

    fn limit_slot(&mut self, slot: DgSlot) -> Result<(), CompressibleDgGpuError> {
        let gpu = self.require_gpu()?;
        if gpu.limit(slot)? {
            return Err(CompressibleDgError::NonPhysicalCellAverage.into());
        }
        Ok(())
    }

    fn require_gpu(&mut self) -> Result<&mut CudaDgOperator, CompressibleDgGpuError> {
        self.gpu.as_mut().ok_or(CompressibleDgGpuError::Dg(
            CompressibleDgError::InvalidConfiguration,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::PolynomialDegree;
    use crate::mesh::{Mesh, unit_cube_tetrahedra, unit_square_triangles};
    use crate::physics::fluid::compressible_flow::{
        CompressibleBoundaryCondition, CompressibleDgOperator, CompressibleFluxKind, IdealGas,
        ViscosityModel,
    };

    fn nonuniform_config(mesh: &Mesh, degree: PolynomialDegree) -> CompressibleDgConfig {
        let free =
            ConservativeState::from_primitive(1.0, [30.0, 0.0, 0.0], 101_325.0, IdealGas::AIR);
        let mut boundaries = std::collections::HashMap::new();
        for face in &mesh.boundaries {
            boundaries.insert(face.marker, CompressibleBoundaryCondition::FarField(free));
        }
        CompressibleDgConfig {
            degree,
            flux: CompressibleFluxKind::Hllc,
            persson_threshold: -3.0,
            troubled_cell_threshold: 0.03,
            viscosity: ViscosityModel::Constant(1.8e-5),
            boundaries,
            ..CompressibleDgConfig::default()
        }
    }

    fn nonuniform_state(mesh: &Mesh, degree: PolynomialDegree) -> CompressibleDgState {
        CompressibleDgState::from_function_degree(mesh, degree, |point| {
            let density = 1.0 + 0.3 * (-4.0 * (point[0] - 0.5).powi(2)).exp();
            ConservativeState::from_primitive(
                density,
                [30.0 - 5.0 * point[0], 2.0 * (2.0 * point[1]).sin(), 0.0],
                density * 287.05 * 300.0,
                IdealGas::AIR,
            )
        })
        .expect("state")
    }

    /// Build a staged GPU solver, or `None` when no CUDA device is
    /// available (the suite then skips, mirroring the compute tests).
    fn staged_solver<'m>(
        mesh: &'m Mesh,
        config: CompressibleDgConfig,
        state: &CompressibleDgState,
    ) -> Option<CompressibleDgGpuSolver<'m>> {
        let mut solver = CompressibleDgGpuSolver::new(mesh, config)
            .map_err(|e| eprintln!("gpu new: {e}"))
            .ok()?;
        solver
            .set_state(state)
            .map_err(|e| eprintln!("gpu set_state: {e}"))
            .ok()?;
        Some(solver)
    }

    fn max_rel_diff(a: &[ConservativeState], b: &[ConservativeState]) -> f64 {
        let scale = a
            .iter()
            .flat_map(|value| value.as_array())
            .map(f64::abs)
            .fold(0.0f64, f64::max)
            .max(1.0e-300);
        a.iter()
            .zip(b)
            .flat_map(|(x, y)| {
                x.as_array()
                    .into_iter()
                    .zip(y.as_array())
                    .map(|(u, v)| (u - v).abs() / scale)
            })
            .fold(0.0f64, f64::max)
    }

    #[test]
    fn gpu_spatial_operator_matches_cpu_on_triangles_p2() {
        let mesh = unit_square_triangles(3);
        let config = nonuniform_config(&mesh, PolynomialDegree::P2);
        let state = nonuniform_state(&mesh, PolynomialDegree::P2);
        let cpu = CompressibleDgOperator::new(&mesh, config.clone()).expect("cpu operator");
        let expected = cpu.spatial_operator(&state).expect("cpu rate");
        let Some(mut gpu) = staged_solver(&mesh, config, &state) else {
            return;
        };
        let got = gpu.spatial_operator().expect("gpu rate");
        let diff = max_rel_diff(&expected, &got);
        assert!(diff < 1.0e-9, "GPU/CPU rate relative diff {diff:e}");
    }

    #[test]
    fn gpu_spatial_operator_matches_cpu_on_tetrahedra_p1() {
        let mesh = unit_cube_tetrahedra(2);
        let config = nonuniform_config(&mesh, PolynomialDegree::P1);
        let state = nonuniform_state(&mesh, PolynomialDegree::P1);
        let cpu = CompressibleDgOperator::new(&mesh, config.clone()).expect("cpu operator");
        let expected = cpu.spatial_operator(&state).expect("cpu rate");
        let Some(mut gpu) = staged_solver(&mesh, config, &state) else {
            return;
        };
        let got = gpu.spatial_operator().expect("gpu rate");
        let diff = max_rel_diff(&expected, &got);
        assert!(diff < 1.0e-9, "GPU/CPU rate relative diff {diff:e}");
    }

    #[test]
    fn gpu_steps_match_cpu_steps() {
        let mesh = unit_square_triangles(4);
        let config = nonuniform_config(&mesh, PolynomialDegree::P2);
        let initial = nonuniform_state(&mesh, PolynomialDegree::P2);
        let mut cpu_solver = CompressibleDgSolver::new(&mesh, config.clone()).expect("cpu solver");
        let mut cpu_state = initial.clone();
        let Some(mut gpu) = staged_solver(&mesh, config, &initial) else {
            return;
        };
        for _ in 0..5 {
            let cpu_report = cpu_solver.step(&mut cpu_state, 1.0e-4).expect("cpu step");
            let gpu_report = gpu.step(1.0e-4).expect("gpu step");
            assert_eq!(cpu_report.retries, gpu_report.retries);
            assert!(
                (cpu_report.accepted_time_step - gpu_report.accepted_time_step).abs()
                    / cpu_report.accepted_time_step
                    < 1.0e-9,
                "dt mismatch {} vs {}",
                cpu_report.accepted_time_step,
                gpu_report.accepted_time_step
            );
        }
        let mut gpu_state = initial.clone();
        gpu.fetch_state(&mut gpu_state).expect("fetch");
        let diff = max_rel_diff(&cpu_state.values, &gpu_state.values);
        assert!(diff < 1.0e-8, "GPU/CPU state relative diff {diff:e}");
    }
}
