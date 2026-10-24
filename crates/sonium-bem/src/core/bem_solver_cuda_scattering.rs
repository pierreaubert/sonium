//! Common retained CUDA dispatch for supported scattering representations.

// Rust guideline compliant 2026-02-21
use super::*;
use sonium_compute::MemoryPool;
use sonium_solvers::iterative::DeviceInitialGuess;

impl AssemblyMethod {
    /// Resolve automatic CUDA assembly while respecting the dense storage ceiling.
    ///
    /// Auto selects TBEM below 1000 DOFs and H-matrix otherwise. Explicit
    /// representations pass through unchanged. This selection never changes
    /// the linear solver: compressed CUDA dispatch still requires GMRES.
    pub fn resolve_cuda(self, num_dofs: usize) -> Self {
        match self {
            Self::Auto
                if num_dofs < crate::core::assembly::tbem::CudaTbemDeviceSystem::DOF_LIMIT =>
            {
                Self::Tbem
            }
            Self::Auto => Self::Hmatrix,
            other => other,
        }
    }
}

/// Explicit preconditioning for iterative CUDA scattering sessions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaScatteringPreconditioner {
    /// Inverse physical diagonal.
    Jacobi,
    /// H-matrix spatial-cluster or MLFMM leaf self-block inverses.
    SpatialBlockJacobi,
}

/// CUDA-specific controls shared by retained scattering preparation.
///
/// These explicitly select iterative CUDA preconditioning instead of the CPU
/// `BemSolver::preconditioner` field. Dense direct solves need no preconditioner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaScatteringOptions {
    /// Point or spatial-block preconditioning for iterative representations.
    pub preconditioner: CudaScatteringPreconditioner,
    /// Positive MLFMM leaf target; unused by other representations.
    pub mlfmm_target_elements_per_leaf: usize,
}
impl Default for CudaScatteringOptions {
    fn default() -> Self {
        Self {
            preconditioner: CudaScatteringPreconditioner::SpatialBlockJacobi,
            mlfmm_target_elements_per_leaf: 32,
        }
    }
}

/// Representation-specific certification returned by common CUDA dispatch.
#[derive(Debug, Clone, Copy)]
pub enum CudaScatteringReport {
    /// Dense LU residual and factor reuse.
    Dense(CudaTbemSolveReport),
    /// Dense GMRES with a native refreshed point-Jacobi inverse.
    DenseGmres(sonium_solvers::iterative::BackendKrylovStatus<f64>),
    /// Compressed and optional exact residuals for H-matrix GMRES.
    Hmatrix(CudaHMatrixSolveReport),
    /// Multipole and optional exact residuals for MLFMM GMRES.
    Mlfmm(CudaMlfmmBemSolveReport),
}

#[derive(Debug)]
enum Session<'a> {
    Dense {
        session: Box<CudaBemSession<'a>>,
        problem: Box<BemProblem>,
    },
    DenseGmres {
        session: Box<CudaBemGmresSession<'a>>,
        problem: Box<BemProblem>,
    },
    Hmatrix(Box<CudaHMatrixBemSession<'a>>),
    Mlfmm(Box<CudaMlfmmBemSession<'a>>),
}

/// Retains a selected CUDA representation across incident-field changes.
///
/// Geometry, frequency, materials, basis and formulation stay fixed. Reprepare
/// for changes. The dense-specific API additionally supports numeric updates;
/// the common interface deliberately exposes the shared incident-only contract.
/// Compressed setup and optional exact TBEM certification remain host work.
/// Pool quota does not cover every allocation made by the underlying sessions.
#[derive(Debug)]
pub struct CudaScatteringSession<'a> {
    inner: Session<'a>,
    published: bool,
}

impl BemSolver {
    /// Select and prepare a retained native CUDA scattering representation.
    ///
    /// TBEM admits Direct or GMRES with explicit Jacobi; compressed paths require
    /// GMRES. `Auto` uses [`AssemblyMethod::resolve_cuda`], selecting H-matrix
    /// at 1000 DOFs and above to respect the dense CUDA storage limit. The
    /// requested linear solver is preserved; Auto with Direct therefore rejects
    /// at that size instead of silently switching to GMRES.
    /// No unsupported representation is substituted or executed on the CPU.
    /// Source-count growth can resize the underlying incident descriptor buffer.
    ///
    /// # Errors
    /// Rejects unsupported basis/geometry, SLFMM/directional routes, incompatible
    /// linear methods and invalid leaf size. Propagates the selected session's
    /// quadrature, resource-budget, assembly, preconditioner and backend errors.
    pub fn prepare_cuda_scattering<'a>(
        &self,
        problem: &BemProblem,
        pool: &'a MemoryPool<CudaBackend>,
        options: CudaScatteringOptions,
    ) -> Result<CudaScatteringSession<'a>, BemError> {
        self.validate()?;
        problem.validate()?;
        if self.basis_order != BasisOrder::Constant
            || self.basis_continuity != BasisContinuity::Discontinuous
            || problem
                .mesh
                .elements
                .iter()
                .any(|e| e.property.is_evaluation() || e.element_type != ElementType::Tri3)
        {
            return Err(BemError::NotImplemented(
                "CUDA scattering dispatch requires constant discontinuous surface Tri3".into(),
            ));
        }
        let mut selected = self.clone();
        selected.assembly_method = self
            .assembly_method
            .resolve_cuda(problem.mesh.elements.len());
        let inner = match selected.assembly_method {
            AssemblyMethod::Tbem if selected.solver_method == SolverMethod::Gmres => {
                if options.preconditioner != CudaScatteringPreconditioner::Jacobi {
                    return Err(BemError::NotImplemented("dense CUDA GMRES requires explicit Jacobi; spatial block preconditioning is not implemented".into()));
                }
                Session::DenseGmres {
                    session: Box::new(selected.prepare_cuda_gmres_jacobi(problem, pool)?),
                    problem: Box::new(problem.clone()),
                }
            }
            AssemblyMethod::Tbem => Session::Dense {
                session: Box::new(selected.prepare_cuda(problem, pool.backend())?),
                problem: Box::new(problem.clone()),
            },
            AssemblyMethod::Hmatrix => Session::Hmatrix(Box::new(selected.prepare_cuda_hmatrix(
                problem,
                pool,
                match options.preconditioner {
                    CudaScatteringPreconditioner::Jacobi => CudaHMatrixPreconditioner::Jacobi,
                    CudaScatteringPreconditioner::SpatialBlockJacobi => {
                        CudaHMatrixPreconditioner::ClusterBlockJacobi
                    }
                },
            )?)),
            AssemblyMethod::Mlfmm => {
                Session::Mlfmm(Box::new(selected.prepare_cuda_mlfmm_with_leaf_size(
                    problem,
                    pool,
                    match options.preconditioner {
                        CudaScatteringPreconditioner::Jacobi => CudaMlfmmPreconditioner::Jacobi,
                        CudaScatteringPreconditioner::SpatialBlockJacobi => {
                            CudaMlfmmPreconditioner::LeafBlockJacobi
                        }
                    },
                    options.mlfmm_target_elements_per_leaf,
                )?))
            }
            _ => {
                return Err(BemError::NotImplemented(
                    "requested assembly has no retained CUDA scattering dispatch".into(),
                ));
            }
        };
        Ok(CudaScatteringSession {
            inner,
            published: false,
        })
    }
}

impl CudaScatteringSession<'_> {
    /// Return the resolved assembly representation.
    pub fn assembly_method(&self) -> AssemblyMethod {
        match self.inner {
            Session::Dense { .. } | Session::DenseGmres { .. } => AssemblyMethod::Tbem,
            Session::Hmatrix(_) => AssemblyMethod::Hmatrix,
            Session::Mlfmm(_) => AssemblyMethod::Mlfmm,
        }
    }

    /// Solve a new incident field using the retained representation and preconditioner.
    ///
    /// Direct solves reuse factors and do not use an iterative initial guess.
    /// For every representation, `PreviousSolution` requires a successful prior
    /// call. Any failure unpublishes output; retry with Zero after backend recovery.
    ///
    /// # Errors
    /// Rejects invalid sources or unavailable previous solutions. Propagates
    /// selected-session failures, nonconvergence and failed certification.
    pub fn solve_incident(
        &mut self,
        incident: &IncidentField,
        initial: DeviceInitialGuess,
    ) -> Result<CudaScatteringReport, BemError> {
        let previous = std::mem::replace(&mut self.published, false);
        if initial == DeviceInitialGuess::PreviousSolution && !previous {
            return Err(BemError::SolverFailed(
                "no certified previous CUDA scattering solution".into(),
            ));
        }
        let report = match &mut self.inner {
            Session::Dense { session, problem } => {
                problem.incident_field = incident.clone();
                CudaScatteringReport::Dense(session.solve_device(problem)?)
            }
            Session::DenseGmres { session, problem } => {
                problem.incident_field = incident.clone();
                CudaScatteringReport::DenseGmres(session.solve_device(problem, initial)?)
            }
            Session::Hmatrix(session) => {
                CudaScatteringReport::Hmatrix(session.solve_incident(incident, initial)?)
            }
            Session::Mlfmm(session) => {
                CudaScatteringReport::Mlfmm(session.solve_incident(incident, initial)?)
            }
        };
        self.published = true;
        Ok(report)
    }

    /// Borrow the last certified device solution without downloading it.
    pub fn solution_device(&self) -> Option<&CudaBuffer<Complex64>> {
        if !self.published {
            return None;
        }
        match &self.inner {
            Session::Dense { session, .. } => session.solution_device(),
            Session::DenseGmres { session, .. } => session.solution_device(),
            Session::Hmatrix(s) => s.solution_device(),
            Session::Mlfmm(s) => s.solution_device(),
        }
    }

    /// Download a certified solution for the existing host field and output APIs.
    ///
    /// # Errors
    /// Rejects unpublished output and propagates device transfer failures.
    pub fn download_solution(&self) -> Result<BemSolution, BemError> {
        if !self.published {
            return Err(BemError::SolverFailed(
                "no certified CUDA scattering solution".into(),
            ));
        }
        match &self.inner {
            Session::Dense { session, .. } => session.download_solution(),
            Session::DenseGmres { session, .. } => session.download_solution(),
            Session::Hmatrix(s) => s.download_solution(),
            Session::Mlfmm(s) => s.download_solution(),
        }
    }
}

#[cfg(test)]
#[path = "bem_solver_cuda_scattering_tests.rs"]
mod tests;
