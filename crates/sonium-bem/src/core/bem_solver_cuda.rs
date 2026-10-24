//! Explicit high-level CUDA session for small constant-panel scattering problems.

// Rust guideline compliant 2026-02-21
use super::*;
use crate::core::assembly::tbem::{CudaTbemSolveReport, CudaTbemSolveSession};
use sonium_compute::{Backend, CudaBackend, CudaBuffer, DeviceBufferLike};

// Check common iteration dependencies before host assembly or hierarchy setup.
// Representation-specific admission remains with each producer.
fn admit_cuda_gmres(backend: &CudaBackend) -> Result<(), BemError> {
    admit_cuda_krylov(backend, sonium_solvers::iterative::BackendKrylovMethod::Gmres)
}

fn admit_cuda_krylov(
    backend: &CudaBackend,
    method: sonium_solvers::iterative::BackendKrylovMethod,
) -> Result<(), BemError> {
    if let Some(error) = backend.pending_error().map_err(|e| BemError::SolverFailed(e.to_string()))? {
        return Err(BemError::SolverFailed(error.to_string()));
    }
    sonium_solvers::iterative::validate_backend_krylov::<_, Complex64>(
        backend,
        method,
        true,
    ).map_err(|e| BemError::SolverFailed(e.to_string()))
}

#[path = "bem_solver_cuda_hmatrix.rs"]
mod hmatrix;
#[doc(inline)]
pub use hmatrix::{
    CudaHMatrixBlockSolveReport,
    CudaHMatrixAction,
    CudaHMatrixBemSession,
    CudaHMatrixLinearSession,
    CudaHMatrixPreconditioner,
    CudaHMatrixSolveReport,
};

#[path = "bem_solver_cuda_mlfmm.rs"]
mod mlfmm;
#[doc(inline)]
pub use mlfmm::{CudaMlfmmBemSession, CudaMlfmmBemSolveReport};
#[doc(inline)]
pub use crate::core::assembly::mlfmm::CudaMlfmmPreconditioner;

#[path = "bem_solver_cuda_gmres.rs"]
mod gmres;
#[doc(inline)]
pub use gmres::CudaBemGmresSession;

#[path = "bem_solver_cuda_scattering.rs"]
mod scattering;
#[doc(inline)]
pub use scattering::{CudaScatteringOptions, CudaScatteringPreconditioner, CudaScatteringReport, CudaScatteringSession};

/// A retained CUDA solve configured through the public BEM solver API.
///
/// Material preparation uses the CPU. Incident evaluation, matrix assembly,
/// factorization, solve and residual certification use CUDA. The backend is
/// borrowed for the session lifetime, preventing accidental context switching.
/// Host field evaluation is available after an explicit solution download.
pub struct CudaBemSession<'a> {
    backend: &'a CudaBackend,
    solver: BemSolver,
    session: CudaTbemSolveSession,
    incident_rhs: CudaBuffer<Complex64>,
    incident_points: CudaBuffer<f64>,
    incident_sources: CudaBuffer<f64>,
    metadata: Option<BemSolution>,
}

impl std::fmt::Debug for CudaBemSession<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaBemSession")
            .field("n", &self.incident_rhs.len())
            .field("has_solution", &self.metadata.is_some())
            .finish_non_exhaustive()
    }
}

impl BemSolver {
    /// Prepare a persistent CUDA session for a small constant-Tri3 problem.
    ///
    /// The session freezes this solver configuration. Repeated calls may change
    /// incident fields, frequency and boundary material values while preserving
    /// geometry and the quadrature schedule. No CPU execution fallback occurs.
    ///
    /// # Errors
    /// Rejects non-direct, non-TBEM, higher-order, soft-indirect or evaluation-panel
    /// routes, systems at or above 1000 DOFs, per-session resource budgets,
    /// invalid inputs, failed quadrature admission and CUDA allocation failures.
    pub fn prepare_cuda<'a>(
        &self,
        problem: &BemProblem,
        backend: &'a CudaBackend,
    ) -> Result<CudaBemSession<'a>, BemError> {
        let elements = self.prepare_cuda_dense_elements(problem, SolverMethod::Direct)?;
        let session =
            CudaTbemSolveSession::new(&elements, &problem.mesh.nodes, &problem.physics, backend)
                .map_err(|error| BemError::SolverFailed(error.to_string()))?;
        let incident_rhs = backend
            .raw_alloc::<Complex64>(count_dofs(&elements))
            .map_err(|error| BemError::SolverFailed(error.to_string()))?;
        let geometry: Vec<_> = elements
            .iter()
            .flat_map(|e| e.center.into_iter().chain(e.normal))
            .collect();
        let mut incident_points = backend
            .raw_alloc::<f64>(geometry.len())
            .map_err(|error| BemError::SolverFailed(error.to_string()))?;
        backend
            .try_copy_h2d(&geometry, &mut incident_points)
            .map_err(|error| BemError::SolverFailed(error.to_string()))?;
        let incident_sources = backend
            .raw_alloc::<f64>(pack_incident_sources(&problem.incident_field).len().max(6))
            .map_err(|error| BemError::SolverFailed(error.to_string()))?;
        Ok(CudaBemSession {
            backend,
            solver: self.clone(),
            session,
            incident_rhs,
            incident_points,
            incident_sources,
            metadata: None,
        })
    }

    fn prepare_cuda_dense_elements(&self, problem: &BemProblem, method: SolverMethod) -> Result<Vec<Element>, BemError> {
        self.validate()?;
        problem.validate()?;
        if self.resource_budget.is_some() {
            return Err(BemError::NotImplemented(
                "CUDA per-session resource estimates are not yet available; configure the compute backend quota instead".into()));
        }
        if self.solver_method != method {
            return Err(BemError::NotImplemented(format!("CUDA dense session requires {method:?}")));
        }
        if self.basis_order != BasisOrder::Constant
            || self.basis_continuity != BasisContinuity::Discontinuous
            || problem.bc_type == BoundaryConditionType::Soft
            || problem
                .mesh
                .elements
                .iter()
                .any(|e| e.property.is_evaluation() || e.element_type != ElementType::Tri3)
        {
            return Err(BemError::NotImplemented(
                "CUDA BEM session requires constant discontinuous Tri3 with rigid or impedance boundaries".into()));
        }
        let elements = self.prepare_elements(problem)?;
        let n = count_dofs(&elements);
        if n >= crate::core::assembly::tbem::CudaTbemDeviceSystem::DOF_LIMIT
            || self.assembly_method.resolve(n) != AssemblyMethod::Tbem
        {
            return Err(BemError::NotImplemented(
                "CUDA dense session requires TBEM below 1000 DOFs; compressed requests are not replaced by dense assembly".into()));
        }
        if let Some(policy) = self.quadrature_policy {
            validate_tbem_quadrature_for_basis(
                &elements,
                &problem.mesh.nodes,
                &problem.physics,
                self.basis_order,
                &policy,
            )
            .map_err(|error| {
                BemError::InvalidParameters(format!(
                    "TBEM quadrature admission rejected assembly: {error}"
                ))
            })?;
        }
        Ok(elements)
    }
}

impl CudaBemSession<'_> {
    /// Solve a compatible problem and retain its certified surface solution.
    ///
    /// Source descriptors are uploaded once per call. Numeric matrix changes
    /// refactor retained LU storage; incident-only changes reuse its factors.
    ///
    /// # Errors
    /// Rejects unsupported inputs, changed geometry/quadrature, backend failures
    /// and failed residual certification. An error unpublishes the old solution.
    pub fn solve_device(&mut self, problem: &BemProblem) -> Result<CudaTbemSolveReport, BemError> {
        self.metadata = None;
        let elements = self.solver.prepare_cuda_dense_elements(problem, SolverMethod::Direct)?;
        let beta_scale = if problem.use_burton_miller {
            PhysicsParams::optimal_beta_scale(problem.ka())
        } else {
            0.0
        };
        let beta = problem.physics.burton_miller_beta_scaled(beta_scale);
        // The CPU stabilized route multiplies both A and b by the same real
        // sign. Cancel it here, including the incident load, to preserve x.
        let mut sources = pack_incident_sources(&problem.incident_field);
        let source_count = sources.len() / 6;
        if sources.len() > self.incident_sources.len() {
            self.incident_sources = self
                .backend
                .raw_alloc::<f64>(sources.len())
                .map_err(|error| BemError::SolverFailed(error.to_string()))?;
        }
        sources.resize(self.incident_sources.len(), 0.0);
        self.backend
            .try_copy_h2d(&sources, &mut self.incident_sources)
            .map_err(|error| BemError::SolverFailed(error.to_string()))?;
        self.backend
            .try_bem_incident_rhs_c64_device(
                &self.incident_points,
                &self.incident_sources,
                source_count,
                problem.physics.wave_number,
                problem.physics.harmonic_factor,
                problem.physics.gamma(),
                beta * problem.physics.tau,
                &mut self.incident_rhs,
            )
            .map_err(|error| BemError::SolverFailed(error.to_string()))?;
        let report = self
            .session
            .solve_with_incident_rhs(
                &elements,
                &problem.mesh.nodes,
                &problem.physics,
                beta,
                self.backend,
                self.solver.tolerance,
                Some(&self.incident_rhs),
            )
            .map_err(|error| BemError::SolverFailed(error.to_string()))?;
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

    /// Borrow the last certified surface solution without downloading it.
    pub fn solution_device(&self) -> Option<&CudaBuffer<Complex64>> {
        self.metadata.as_ref()?;
        self.session.solution()
    }

    /// Download the certified solution for existing CPU field/output APIs.
    ///
    /// # Errors
    /// Fails if no successful solve is published or the CUDA transfer fails.
    pub fn download_solution(&self) -> Result<BemSolution, BemError> {
        let mut solution = self
            .metadata
            .clone()
            .ok_or_else(|| BemError::SolverFailed("no certified CUDA solution".into()))?;
        let device = self
            .solution_device()
            .ok_or_else(|| BemError::SolverFailed("no certified CUDA solution".into()))?;
        solution
            .surface_pressure
            .resize(device.len(), Complex64::default());
        self.backend
            .try_copy_d2h(device, &mut solution.surface_pressure)
            .map_err(|error| BemError::SolverFailed(error.to_string()))?;
        Ok(solution)
    }
}

#[cfg(test)]
#[path = "bem_solver_cuda_tests.rs"]
mod tests;

// Descriptor conversion is O(source count); no collocation samples are built here.
pub(crate) fn pack_incident_sources(field: &IncidentField) -> Vec<f64> {
    let mut packed = Vec::new();
    let mut push = |coordinates: &[f64; 3], amplitude: Complex64, kind: f64| {
        packed.extend_from_slice(coordinates);
        packed.extend_from_slice(&[amplitude.re, amplitude.im, kind]);
    };
    match field {
        IncidentField::PlaneWave {
            direction,
            amplitude,
        } => push(direction, *amplitude, 0.0),
        IncidentField::PointSource { position, strength } => push(position, *strength, 1.0),
        IncidentField::MultiplePlaneWaves(waves) => {
            for (direction, amplitude) in waves {
                push(direction, *amplitude, 0.0);
            }
        }
        IncidentField::MultiplePointSources(sources) => {
            for (position, strength) in sources {
                push(position, *strength, 1.0);
            }
        }
    }
    packed
}
