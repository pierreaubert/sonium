//! Public retained MLFMM scattering solves with native incident evaluation.

// Rust guideline compliant 2026-02-21
use super::*;
use crate::core::assembly::mlfmm::{
    CudaMlfmmSolveSession, build_cluster_tree, build_mlfmm_system_stabilized_checked,
};
use sonium_compute::{Buffer, MemoryPool, Transpose};
use sonium_solvers::iterative::{BackendKrylovConfig, DeviceInitialGuess};

/// Residual certificates for a retained MLFMM scattering solve.
#[derive(Debug, Clone, Copy)]
pub struct CudaMlfmmBemSolveReport {
    /// Iterations against the uploaded MLFMM approximation.
    pub iterations: usize,
    /// Unpreconditioned relative MLFMM residual; absolute for zero RHS.
    pub mlfmm_relative_residual: f64,
    /// Optional independent uncompressed TBEM residual, computed on the host.
    pub exact_relative_residual: Option<f64>,
}

/// Fixed MLFMM scattering operator with resident incident RHS and retained solves.
///
/// Geometry, frequency, material, basis and formulation are fixed. Repeated
/// incident descriptors are uploaded, while RHS evaluation and linear solves
/// execute on CUDA. Setup and optional exact certification use the CPU.
/// Downloading the solution explicitly enables the existing host field APIs.
pub struct CudaMlfmmBemSession<'a> {
    pool: &'a MemoryPool<CudaBackend>,
    linear: CudaMlfmmSolveSession<'a>,
    rhs: Buffer<Complex64, CudaBackend>,
    points: Buffer<f64, CudaBackend>,
    sources: Buffer<f64, CudaBackend>,
    metadata: BemSolution,
    beta_scale: f64,
    exact_tolerance: Option<f64>,
    far_translations: usize,
    published: bool,
}

impl std::fmt::Debug for CudaMlfmmBemSession<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaMlfmmBemSession")
            .field("dofs", &self.rhs.len())
            .field("far_translations", &self.far_translations)
            .field("published", &self.published)
            .finish_non_exhaustive()
    }
}

fn failed(error: impl std::fmt::Display) -> BemError {
    BemError::SolverFailed(error.to_string())
}

impl BemSolver {
    /// Prepare repeated CUDA MLFMM scattering solves from a BEM problem.
    ///
    /// Uses the CPU producer's target of 32 elements per leaf. The explicitly
    /// supplied preconditioner selects Jacobi or leaf-block Jacobi. No global
    /// dense matrix is constructed. Pool quota coverage excludes existing sparse
    /// descriptor allocations; per-session resource estimates are not available.
    ///
    /// # Errors
    /// Rejects non-MLFMM/non-GMRES, higher-order/shared-DOF, soft, evaluation-panel,
    /// non-Tri3 and budgeted requests, invalid quadrature or unavailable native
    /// kernels. Propagates assembly, local-factorization and backend failures.
    pub fn prepare_cuda_mlfmm<'a>(
        &self,
        problem: &BemProblem,
        pool: &'a MemoryPool<CudaBackend>,
        preconditioner: CudaMlfmmPreconditioner,
    ) -> Result<CudaMlfmmBemSession<'a>, BemError> {
        self.prepare_cuda_mlfmm_with_leaf_size(problem, pool, preconditioner, 32)
    }

    /// Prepare a retained solve with an explicit target number of panels per leaf.
    ///
    /// Smaller leaves trade more translations for less near-field storage.
    ///
    /// # Errors
    /// Returns the same errors as [`Self::prepare_cuda_mlfmm`], and rejects zero
    /// target leaf size before allocating device storage.
    pub fn prepare_cuda_mlfmm_with_leaf_size<'a>(
        &self,
        problem: &BemProblem,
        pool: &'a MemoryPool<CudaBackend>,
        preconditioner: CudaMlfmmPreconditioner,
        target_elements_per_leaf: usize,
    ) -> Result<CudaMlfmmBemSession<'a>, BemError> {
        if target_elements_per_leaf == 0 {
            return Err(BemError::InvalidParameters(
                "MLFMM leaf size must be positive".into(),
            ));
        }
        self.validate()?;
        problem.validate()?;
        if self.solver_method != SolverMethod::Gmres
            || self.assembly_method != AssemblyMethod::Mlfmm
            || self.basis_order != BasisOrder::Constant
            || self.basis_continuity != BasisContinuity::Discontinuous
            || problem.bc_type == BoundaryConditionType::Soft
            || self.resource_budget.is_some()
            || problem
                .mesh
                .elements
                .iter()
                .any(|e| e.property.is_evaluation() || e.element_type != ElementType::Tri3)
        {
            return Err(BemError::NotImplemented("CUDA MLFMM scattering requires explicit MLFMM/GMRES, constant discontinuous Tri3, rigid/impedance boundaries and no per-session budget".into()));
        }
        let backend = pool.backend();
        admit_cuda_gmres(backend)?;
        if !backend.has_fmm_c64() || !backend.has_bem_incident_rhs_c64() {
            return Err(failed("native MLFMM and incident kernels are required"));
        }
        let elements = self.prepare_elements(problem)?;
        if let Some(policy) = self.quadrature_policy {
            validate_tbem_quadrature_for_basis(
                &elements,
                &problem.mesh.nodes,
                &problem.physics,
                self.basis_order,
                &policy,
            )
            .map_err(|e| BemError::InvalidParameters(e.to_string()))?;
        }
        let n = count_dofs(&elements);
        let beta_scale = if problem.use_burton_miller {
            PhysicsParams::optimal_beta_scale(problem.ka())
        } else {
            0.0
        };
        let tree = build_cluster_tree(&elements, target_elements_per_leaf, &problem.physics);
        let system = build_mlfmm_system_stabilized_checked(
            &elements,
            &problem.mesh.nodes,
            tree,
            &problem.physics,
            1.0,
            beta_scale,
        )
        .map_err(failed)?;
        let far_translations = system.d_matrices.iter().map(Vec::len).sum();
        let linear = system.try_prepare_cuda_solve(
            pool,
            Transpose::NoTrans,
            preconditioner,
            BackendKrylovConfig {
                restart: self.gmres_restart,
                max_iterations: self.max_iterations,
                tolerance: self.tolerance,
                require_native: true,
                exact_residual_check_interval: 0,
            },
        )?;
        let geometry: Vec<_> = elements
            .iter()
            .flat_map(|e| e.center.into_iter().chain(e.normal))
            .collect();
        let mut points = pool.try_alloc(geometry.len()).map_err(failed)?;
        backend
            .try_copy_h2d(&geometry, points.raw_mut())
            .map_err(failed)?;
        Ok(CudaMlfmmBemSession {
            pool,
            linear,
            rhs: pool.try_alloc(n).map_err(failed)?,
            points,
            sources: pool
                .try_alloc(pack_incident_sources(&problem.incident_field).len().max(6))
                .map_err(failed)?,
            metadata: BemSolution {
                surface_pressure: Vec::new(),
                elements,
                nodes: problem.mesh.nodes.clone(),
                incident_field: problem.incident_field.clone(),
                physics: problem.physics.clone(),
                basis_order: BasisOrder::Constant,
                representation: BemSolutionRepresentation::DirectCauchy,
            },
            beta_scale,
            exact_tolerance: (self.compressed_exact_residual_dof_limit > 0
                && n <= self.compressed_exact_residual_dof_limit)
                .then_some(self.compressed_exact_residual_tolerance),
            far_translations,
            published: false,
        })
    }
}

impl CudaMlfmmBemSession<'_> {
    /// Number of retained far-field translation entries.
    pub fn far_translations(&self) -> usize {
        self.far_translations
    }

    /// Evaluate and solve a new incident field using the retained MLFMM system.
    ///
    /// Source count growth may allocate a larger descriptor buffer. Optional
    /// exact certification downloads the RHS and solution for a host TBEM action.
    ///
    /// # Errors
    /// Rejects invalid sources, unavailable warm starts, nonconvergence and failed
    /// certificates. Every error invalidates publication and warm-start eligibility.
    /// Recoverable backend errors permit a zero-start retry after recovery;
    /// terminal synchronization failures require a new pool and session.
    pub fn solve_incident(
        &mut self,
        incident: &IncidentField,
        initial: DeviceInitialGuess,
    ) -> Result<CudaMlfmmBemSolveReport, BemError> {
        let previous = std::mem::replace(&mut self.published, false);
        if initial == DeviceInitialGuess::PreviousSolution && !previous {
            return Err(failed("no certified previous MLFMM scattering solution"));
        }
        incident.validate().map_err(failed)?;
        let mut descriptors = pack_incident_sources(incident);
        let count = descriptors.len() / 6;
        if descriptors.len() > self.sources.len() {
            self.sources = self.pool.try_alloc(descriptors.len()).map_err(failed)?;
        }
        descriptors.resize(self.sources.len(), 0.0);
        let backend = self.pool.backend();
        backend
            .try_copy_h2d(&descriptors, self.sources.raw_mut())
            .map_err(failed)?;
        let physics = &self.metadata.physics;
        backend
            .try_bem_incident_rhs_c64_device(
                self.points.raw(),
                self.sources.raw(),
                count,
                physics.wave_number,
                physics.harmonic_factor,
                physics.gamma(),
                physics.burton_miller_beta_scaled(self.beta_scale) * physics.tau,
                self.rhs.raw_mut(),
            )
            .map_err(failed)?;
        let status = self.linear.solve_device(self.rhs.raw(), initial)?;
        let exact_relative_residual = if let Some(tolerance) = self.exact_tolerance {
            let mut x = vec![Complex64::default(); self.rhs.len()];
            let mut b = x.clone();
            backend
                .try_copy_d2h(
                    self.linear
                        .solution_device()
                        .ok_or_else(|| failed("missing certified MLFMM solution"))?,
                    &mut x,
                )
                .map_err(failed)?;
            backend
                .try_copy_d2h(self.rhs.raw(), &mut b)
                .map_err(failed)?;
            let ax = apply_tbem_stabilized_exact(
                &self.metadata.elements,
                &self.metadata.nodes,
                physics,
                1.0,
                self.beta_scale,
                &x,
            )
            .map_err(failed)?;
            let norm = b.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
            let residual = ax
                .iter()
                .zip(&b)
                .map(|(a, b)| (*a - *b).norm_sqr())
                .sum::<f64>()
                .sqrt()
                / if norm > 0.0 { norm } else { 1.0 };
            if !residual.is_finite() || residual > tolerance {
                return Err(BemError::ResidualCheckFailed {
                    label: "cuda-mlfmm-independent-exact".into(),
                    residual,
                    tolerance,
                });
            }
            Some(residual)
        } else {
            None
        };
        self.metadata.incident_field = incident.clone();
        self.published = true;
        Ok(CudaMlfmmBemSolveReport {
            iterations: status.iterations,
            mlfmm_relative_residual: status.true_residual,
            exact_relative_residual,
        })
    }

    /// Borrow the last certified surface solution on CUDA.
    pub fn solution_device(&self) -> Option<&CudaBuffer<Complex64>> {
        self.published
            .then(|| self.linear.solution_device())
            .flatten()
    }

    /// Download a certified surface solution for existing host field APIs.
    ///
    /// # Errors
    /// Fails without a published solution or on a backend transfer error.
    pub fn download_solution(&self) -> Result<BemSolution, BemError> {
        let device = self
            .solution_device()
            .ok_or_else(|| failed("no certified MLFMM scattering solution"))?;
        let mut result = self.metadata.clone();
        result
            .surface_pressure
            .resize(self.rhs.len(), Complex64::default());
        self.pool
            .backend()
            .try_copy_d2h(device, &mut result.surface_pressure)
            .map_err(failed)?;
        Ok(result)
    }
}

#[cfg(test)]
#[path = "bem_solver_cuda_mlfmm_tests.rs"]
mod tests;
