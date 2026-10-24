//! Public retained dense scattering GMRES with native numeric preconditioning.

// Rust guideline compliant 2026-02-21
use super::*;
use crate::core::assembly::tbem::CudaTbemGmresSession;
use sonium_compute::{Buffer, MemoryPool};
use sonium_solvers::iterative::{BackendKrylovConfig, BackendKrylovStatus, DeviceInitialGuess};

/// Retained dense scattering GMRES with explicit native point-Jacobi preconditioning.
///
/// Geometry is fixed. Frequency, boundary material and incident data may change.
/// Quadrature schedule changes rebuild assembly while retaining Krylov storage.
/// Source descriptors and boundary coefficients are
/// uploaded; assembled matrices and Krylov vectors remain on CUDA. Assembly
/// legacy allocations are outside the pool quota. Rejected calls unpublish output.
pub struct CudaBemGmresSession<'a> {
    pool: &'a MemoryPool<CudaBackend>,
    solver: BemSolver,
    session: CudaTbemGmresSession<'a>,
    incident_rhs: Buffer<Complex64, CudaBackend>,
    incident_points: Buffer<f64, CudaBackend>,
    incident_sources: Buffer<f64, CudaBackend>,
    metadata: Option<BemSolution>,
}

impl std::fmt::Debug for CudaBemGmresSession<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaBemGmresSession")
            .field("session", &self.session)
            .field("published", &self.metadata.is_some())
            .finish_non_exhaustive()
    }
}
fn failed(error: impl std::fmt::Display) -> BemError {
    BemError::SolverFailed(error.to_string())
}
fn beta(problem: &BemProblem) -> Complex64 {
    problem
        .physics
        .burton_miller_beta_scaled(if problem.use_burton_miller {
            PhysicsParams::optimal_beta_scale(problem.ka())
        } else {
            0.0
        })
}

impl BemSolver {
    /// Prepare retained CUDA GMRES with explicit point-Jacobi preconditioning.
    ///
    /// This method selects native Jacobi instead of `self.preconditioner`.
    /// It preserves the requested GMRES method and requires TBEM below 1000 DOFs.
    ///
    /// # Errors
    /// Rejects unsupported solver, geometry, basis, formulation, budgets and
    /// quadrature, invalid controls or diagonals, allocation and backend failures.
    pub fn prepare_cuda_gmres_jacobi<'a>(
        &self,
        problem: &BemProblem,
        pool: &'a MemoryPool<CudaBackend>,
    ) -> Result<CudaBemGmresSession<'a>, BemError> {
        admit_cuda_gmres(pool.backend())?;
        let elements = self.prepare_cuda_dense_elements(problem, SolverMethod::Gmres)?;
        let n = count_dofs(&elements);
        let controls = BackendKrylovConfig {
            restart: self.gmres_restart,
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
            require_native: true,
            ..Default::default()
        };
        let session = CudaTbemGmresSession::try_new(
            pool,
            &elements,
            &problem.mesh.nodes,
            &problem.physics,
            beta(problem),
            controls,
        )?;
        let geometry: Vec<_> = elements
            .iter()
            .flat_map(|e| e.center.into_iter().chain(e.normal))
            .collect();
        let mut incident_points = pool.try_alloc::<f64>(geometry.len()).map_err(failed)?;
        pool.backend()
            .try_copy_h2d(&geometry, incident_points.raw_mut())
            .map_err(failed)?;
        Ok(CudaBemGmresSession {
            pool,
            solver: self.clone(),
            session,
            incident_rhs: pool.try_alloc(n).map_err(failed)?,
            incident_points,
            incident_sources: pool
                .try_alloc(pack_incident_sources(&problem.incident_field).len().max(6))
                .map_err(failed)?,
            metadata: None,
        })
    }
}

impl CudaBemGmresSession<'_> {
    /// Solve a compatible problem while retaining device Krylov storage.
    ///
    /// # Errors
    /// Rejects invalid input or unavailable warm starts, geometry changes,
    /// nonconvergence and backend failures. Retry with Zero after backend recovery.
    pub fn solve_device(
        &mut self,
        problem: &BemProblem,
        initial: DeviceInitialGuess,
    ) -> Result<BackendKrylovStatus<f64>, BemError> {
        let previous = self.metadata.take().is_some();
        if initial == DeviceInitialGuess::PreviousSolution && !previous {
            return Err(failed("no certified previous scattering solution"));
        }
        let elements = self
            .solver
            .prepare_cuda_dense_elements(problem, SolverMethod::Gmres)?;
        let backend = self.pool.backend();
        let beta = beta(problem);
        let mut sources = pack_incident_sources(&problem.incident_field);
        let source_count = sources.len() / 6;
        if sources.len() > self.incident_sources.len() {
            self.incident_sources = self.pool.try_alloc(sources.len()).map_err(failed)?;
        }
        sources.resize(self.incident_sources.len(), 0.0);
        backend
            .try_copy_h2d(&sources, self.incident_sources.raw_mut())
            .map_err(failed)?;
        backend
            .try_bem_incident_rhs_c64_device(
                self.incident_points.raw(),
                self.incident_sources.raw(),
                source_count,
                problem.physics.wave_number,
                problem.physics.harmonic_factor,
                problem.physics.gamma(),
                beta * problem.physics.tau,
                self.incident_rhs.raw_mut(),
            )
            .map_err(failed)?;
        let report = self.session.solve_device(
            &elements,
            &problem.mesh.nodes,
            &problem.physics,
            beta,
            Some(self.incident_rhs.raw()),
            initial,
        )?;
        self.metadata = Some(BemSolution {
            surface_pressure: Vec::new(),
            elements,
            nodes: problem.mesh.nodes.clone(),
            incident_field: problem.incident_field.clone(),
            physics: problem.physics.clone(),
            basis_order: BasisOrder::Constant,
            representation: BemSolutionRepresentation::DirectCauchy,
        });
        Ok(report)
    }

    /// Borrow the last certified solution on CUDA.
    pub fn solution_device(&self) -> Option<&CudaBuffer<Complex64>> {
        self.metadata.as_ref()?;
        self.session.solution_device()
    }

    /// Return quadrature workspace replacements, excluding initial preparation.
    pub fn assembly_rebuilds(&self) -> usize {
        self.session.assembly_rebuilds()
    }

    /// Download a certified solution for existing field and output APIs.
    ///
    /// # Errors
    /// Rejects unpublished output and propagates transfer failures.
    pub fn download_solution(&self) -> Result<BemSolution, BemError> {
        let mut result = self
            .metadata
            .clone()
            .ok_or_else(|| failed("no certified CUDA solution"))?;
        let device = self
            .solution_device()
            .ok_or_else(|| failed("no certified CUDA solution"))?;
        result
            .surface_pressure
            .resize(device.len(), Complex64::default());
        self.pool
            .backend()
            .try_copy_d2h(device, &mut result.surface_pressure)
            .map_err(failed)?;
        Ok(result)
    }
}
