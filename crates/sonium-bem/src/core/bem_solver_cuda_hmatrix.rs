//! Persistent compressed CUDA solves for constant-panel scattering.
//!
//! Tree construction, ACA and self-block factorization are host setup. Incident
//! evaluation, compressed actions, preconditioning and GMRES retain device data.
//! No dense global matrix is built for setup or RHS evaluation. Optional exact
//! TBEM certification is explicitly host work and downloads two vectors.

// Rust guideline compliant 2026-02-21
use super::*;
use crate::core::assembly::hmatrix::{HMatrix, build_hmatrix_stabilized};
use sonium_compute::{
    BackendResult, Buffer, CudaHMatrixC64, DevicePreconditioner, MemoryPool, PersistentDeviceCsr,
};
use sonium_solvers::iterative::{BackendKrylovConfig, DeviceGmresSession, DeviceInitialGuess};
use sonium_solvers::{CsrMatrix, TripletMatrix};

#[path = "bem_solver_cuda_hmatrix_field.rs"]
mod field;
#[path = "bem_solver_cuda_hmatrix_linear.rs"]
mod linear;
#[doc(inline)]
pub use linear::{
    CudaHMatrixBlockSolveReport,
    CudaHMatrixAction,
    CudaHMatrixLinearSession,
};

/// Device preconditioner selected explicitly for a compressed CUDA session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaHMatrixPreconditioner {
    /// Inverse of the physical diagonal.
    Jacobi,
    /// Exact inverses of the spatial cluster self-blocks.
    ClusterBlockJacobi,
}

/// Residual certification of a retained compressed solve.
#[derive(Debug, Clone, Copy)]
pub struct CudaHMatrixSolveReport {
    /// GMRES iteration count.
    pub iterations: usize,
    /// Recomputed residual against the compressed device operator.
    /// Uses the absolute norm when the incident RHS is zero.
    pub compressed_relative_residual: f64,
    /// Independent uncompressed TBEM residual when explicitly enabled.
    /// Uses the absolute norm when the incident RHS is zero.
    pub exact_relative_residual: Option<f64>,
}

struct Inverse(PersistentDeviceCsr<CudaBackend, Complex64>);

impl DevicePreconditioner<CudaBackend, Complex64> for Inverse {
    fn apply(
        &self,
        backend: &CudaBackend,
        r: &CudaBuffer<Complex64>,
        z: &mut CudaBuffer<Complex64>,
    ) {
        if let Err(error) = self.try_apply(backend, r, z) {
            log::error!("CUDA BEM block inverse failed: {error}");
        }
    }
    fn try_apply(
        &self,
        backend: &CudaBackend,
        r: &CudaBuffer<Complex64>,
        z: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        self.0.try_apply(backend, r, z)
    }
    fn try_apply_colmajor(
        &self,
        backend: &CudaBackend,
        rhs: usize,
        r: &CudaBuffer<Complex64>,
        z: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        self.0.try_apply_colmajor(backend, rhs, r, z)
    }
}

/// Retained H-matrix, spatial preconditioning and GMRES for repeated incident fields.
///
/// Geometry, material, frequency and formulation are frozen. Reprepare for those
/// changes. The pool owns H-matrix/RHS/Krylov storage; the sparse inverse retains
/// the compute backend's existing allocation policy. Global quota coverage is
/// not implied. Prepared off-surface field evaluation also retains device data.
pub struct CudaHMatrixBemSession<'a> {
    pool: &'a MemoryPool<CudaBackend>,
    operator: CudaHMatrixC64,
    preconditioner: Inverse,
    krylov: DeviceGmresSession<'a, CudaBackend, Complex64>,
    controls: BackendKrylovConfig<f64>,
    rhs: Buffer<Complex64, CudaBackend>,
    points: Buffer<f64, CudaBackend>,
    sources: Buffer<f64, CudaBackend>,
    source_count: usize,
    field: Option<field::FieldWorkspace>,
    beta_scale: f64,
    exact_tolerance: Option<f64>,
    metadata: BemSolution,
    published: bool,
    compression_ratio: f64,
    low_rank_blocks: usize,
}

impl std::fmt::Debug for CudaHMatrixBemSession<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaHMatrixBemSession")
            .field("dofs", &self.rhs.len())
            .field("compression_ratio", &self.compression_ratio)
            .field("low_rank_blocks", &self.low_rank_blocks)
            .field("published", &self.published)
            .finish_non_exhaustive()
    }
}

fn failed(error: impl std::fmt::Display) -> BemError {
    BemError::SolverFailed(error.to_string())
}

impl BemSolver {
    /// Prepare a compressed CUDA solve for fixed rigid/impedance scattering geometry.
    ///
    /// `preconditioner` explicitly selects this session's preconditioning. Solver
    /// tolerance, restart, iteration limit and ACA controls are retained. Assembly
    /// occurs once, without automatic compression retries or a global dense matrix.
    /// The host builder may store individual uncompressible blocks densely.
    /// The optional exact-TBEM certification policy is honored for its DOF range.
    ///
    /// # Errors
    /// Requires GMRES, H-matrix assembly, constant discontinuous Tri3 and rigid
    /// or impedance scattering. Rejects budgets without session estimates,
    /// unresolved ACA rank caps, singular self-blocks and backend failures.
    pub fn prepare_cuda_hmatrix<'a>(
        &self,
        problem: &BemProblem,
        pool: &'a MemoryPool<CudaBackend>,
        preconditioner: CudaHMatrixPreconditioner,
    ) -> Result<CudaHMatrixBemSession<'a>, BemError> {
        self.prepare_cuda_hmatrix_for_action(
            problem,
            pool,
            preconditioner,
            CudaHMatrixAction::Forward,
        )
    }

    fn prepare_cuda_hmatrix_for_action<'a>(
        &self,
        problem: &BemProblem,
        pool: &'a MemoryPool<CudaBackend>,
        preconditioner: CudaHMatrixPreconditioner,
        action: CudaHMatrixAction,
    ) -> Result<CudaHMatrixBemSession<'a>, BemError> {
        self.validate()?;
        problem.validate()?;
        if self.solver_method != SolverMethod::Gmres
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
            return Err(BemError::NotImplemented("CUDA H-matrix requires GMRES, constant discontinuous Tri3, rigid/impedance scattering and no per-session resource estimate".into()));
        }
        let backend = pool.backend();
        admit_cuda_gmres(backend)?;
        if !backend.has_fmm_c64() || !backend.has_bem_incident_rhs_c64() {
            return Err(failed(
                "CUDA H-matrix requires native FMM, incident and validation kernels",
            ));
        }
        let elements = self.prepare_elements(problem)?;
        let n = count_dofs(&elements);
        if self.assembly_method.resolve(n) != AssemblyMethod::Hmatrix {
            return Err(BemError::NotImplemented(
                "CUDA H-matrix session requires H-matrix assembly".into(),
            ));
        }
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
        let beta_scale = if problem.use_burton_miller {
            PhysicsParams::optimal_beta_scale(problem.ka())
        } else {
            0.0
        };
        let clusters = Self::build_hmatrix_clusters(&elements);
        let max_rank = if self.hmatrix_max_rank == 0 {
            n / 4 + 1
        } else {
            self.hmatrix_max_rank
        };
        // Cancel the common real stabilization sign in A and b. Prescribed
        // boundary velocity is zero, so scattering needs only the incident RHS.
        // Calling assemble_system_stabilized here would build dense TBEM for b.
        let host = build_hmatrix_stabilized(
            &elements,
            &problem.mesh.nodes,
            &clusters,
            &problem.physics,
            self.hmatrix_aca_tolerance,
            max_rank,
            1.0,
            beta_scale,
        );
        if host.rank_stats().saturated_count != 0 {
            return Err(failed(
                "ACA rank cap unresolved; increase max rank and reprepare",
            ));
        }
        let inverse = linear::orient_inverse(inverse_blocks(&host, preconditioner)?, action)?;
        let operator = host.try_to_cuda(pool.clone()).map_err(failed)?;
        let preconditioner = Inverse(
            PersistentDeviceCsr::try_from_host_csr_ref(backend, inverse.as_host_csr_ref())
                .map_err(failed)?,
        );
        let mut krylov = DeviceGmresSession::try_new(pool, n).map_err(failed)?;
        krylov
            .try_reserve_cycle(self.gmres_restart)
            .map_err(failed)?;
        backend
            .try_reserve_krylov_reductions::<Complex64>(self.gmres_restart.min(self.max_iterations))
            .map_err(failed)?;
        let geometry: Vec<_> = elements
            .iter()
            .flat_map(|e| e.center.into_iter().chain(e.normal))
            .collect();
        let mut points = pool.try_alloc(geometry.len()).map_err(failed)?;
        backend
            .try_copy_h2d(&geometry, points.raw_mut())
            .map_err(failed)?;
        Ok(CudaHMatrixBemSession {
            pool,
            operator,
            preconditioner,
            krylov,
            controls: BackendKrylovConfig {
                max_iterations: self.max_iterations,
                restart: self.gmres_restart,
                tolerance: self.tolerance,
                require_native: true,
                exact_residual_check_interval: 0,
            },
            rhs: pool.try_alloc(n).map_err(failed)?,
            points,
            sources: pool
                .try_alloc(pack_incident_sources(&problem.incident_field).len().max(6))
                .map_err(failed)?,
            source_count: 0,
            field: None,
            beta_scale,
            exact_tolerance: (self.compressed_exact_residual_dof_limit > 0
                && n <= self.compressed_exact_residual_dof_limit)
                .then_some(self.compressed_exact_residual_tolerance),
            metadata: BemSolution {
                surface_pressure: Vec::new(),
                elements,
                nodes: problem.mesh.nodes.clone(),
                incident_field: problem.incident_field.clone(),
                physics: problem.physics.clone(),
                basis_order: BasisOrder::Constant,
                representation: BemSolutionRepresentation::DirectCauchy,
            },
            published: false,
            compression_ratio: host.compression_ratio,
            low_rank_blocks: host.low_rank_blocks.len(),
        })
    }
}

impl CudaHMatrixBemSession<'_> {
    /// Ratio of global dense bytes to host compressed factor bytes at setup.
    pub fn compression_ratio(&self) -> f64 {
        self.compression_ratio
    }

    /// Number of retained low-rank far-field blocks.
    pub fn low_rank_blocks(&self) -> usize {
        self.low_rank_blocks
    }

    /// Solve a new incident field without rebuilding the compressed operator.
    ///
    /// Uploads source descriptors only. The compressed residual is certified on
    /// CUDA; an enabled independent TBEM check explicitly downloads solution and
    /// RHS and evaluates an uncompressed host action without a global dense matrix.
    ///
    /// # Errors
    /// Rejects invalid sources, unavailable warm starts, nonconvergence, failed
    /// exact certification and backend failures. Errors unpublish the solution.
    pub fn solve_incident(
        &mut self,
        incident: &IncidentField,
        initial: DeviceInitialGuess,
    ) -> Result<CudaHMatrixSolveReport, BemError> {
        let previous = std::mem::replace(&mut self.published, false);
        if initial == DeviceInitialGuess::PreviousSolution && !previous {
            return Err(failed("no certified previous H-matrix solution"));
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
        let status = self
            .krylov
            .solve(
                &self.operator,
                &self.preconditioner,
                self.rhs.raw(),
                initial,
                &self.controls,
                None,
            )
            .map_err(failed)?;
        if !status.converged {
            return Err(failed(format!(
                "compressed GMRES did not converge: {status:?}"
            )));
        }
        let exact_relative_residual = if let Some(tolerance) = self.exact_tolerance {
            let mut x = vec![Complex64::default(); self.rhs.len()];
            let mut rhs = x.clone();
            backend
                .try_copy_d2h(
                    self.krylov
                        .solution()
                        .ok_or_else(|| failed("missing certified iterate"))?,
                    &mut x,
                )
                .map_err(failed)?;
            backend
                .try_copy_d2h(self.rhs.raw(), &mut rhs)
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
            let norm = rhs.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
            let residual = ax
                .iter()
                .zip(&rhs)
                .map(|(a, b)| (*a - *b).norm_sqr())
                .sum::<f64>()
                .sqrt()
                / if norm > 0.0 { norm } else { 1.0 };
            if !residual.is_finite() || residual > tolerance {
                return Err(BemError::ResidualCheckFailed {
                    label: "cuda-hmatrix-independent-exact".into(),
                    residual,
                    tolerance,
                });
            }
            Some(residual)
        } else {
            None
        };
        self.metadata.incident_field = incident.clone();
        self.source_count = count;
        self.published = true;
        Ok(CudaHMatrixSolveReport {
            iterations: status.iterations,
            compressed_relative_residual: status.true_residual,
            exact_relative_residual,
        })
    }

    /// Borrow the last certified compressed-system solution on CUDA.
    pub fn solution_device(&self) -> Option<&CudaBuffer<Complex64>> {
        if self.published {
            self.krylov.solution()
        } else {
            None
        }
    }

    /// Download the published surface solution for CPU field/output APIs.
    ///
    /// # Errors
    /// Fails without a published solution or on a backend transfer error.
    pub fn download_solution(&self) -> Result<BemSolution, BemError> {
        let device = self
            .solution_device()
            .ok_or_else(|| failed("no certified H-matrix solution"))?;
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

// Only spatial self-blocks are inverted. No n-by-n matrix or global unit-vector
// applications are needed, and singular/overlapping blocks fail explicitly.
fn inverse_blocks(
    h: &HMatrix,
    kind: CudaHMatrixPreconditioner,
) -> Result<CsrMatrix<Complex64>, BemError> {
    let diagonal = hmatrix_diagonal(h)?;
    let mut inverse = TripletMatrix::new(h.num_dofs, h.num_dofs);
    if kind == CudaHMatrixPreconditioner::Jacobi {
        for (i, value) in diagonal.into_iter().enumerate() {
            if value.norm() == 0.0 {
                return Err(failed("zero H-matrix diagonal"));
            }
            inverse.push(i, i, value.inv());
        }
    } else {
        let mut covered = vec![false; h.num_dofs];
        for block in h
            .dense_blocks
            .iter()
            .filter(|b| b.source_cluster == b.field_cluster)
        {
            let dofs = &h.cluster_dof_indices[block.source_cluster];
            if block.matrix.nrows() != dofs.len() || block.matrix.ncols() != dofs.len() {
                return Err(failed("self-block shape mismatch"));
            }
            for &dof in dofs {
                if std::mem::replace(&mut covered[dof], true) {
                    return Err(failed("overlapping self-block DOFs"));
                }
            }
            let factor = sonium_solvers::lu_factorize(&block.matrix).map_err(failed)?;
            let mut unit = vec![Complex64::default(); dofs.len()];
            let mut column = unit.clone();
            for (j, &global_column) in dofs.iter().enumerate() {
                unit[j] = Complex64::new(1.0, 0.0);
                factor.solve_into(&unit, &mut column).map_err(failed)?;
                unit[j] = Complex64::default();
                for (i, &global_row) in dofs.iter().enumerate() {
                    inverse.push(global_row, global_column, column[i]);
                }
            }
        }
        if covered.iter().any(|&v| !v) {
            return Err(failed("self-blocks do not cover all DOFs"));
        }
    }
    inverse.try_to_csr().map_err(failed)
}

#[cfg(test)]
#[path = "bem_solver_cuda_hmatrix_tests.rs"]
mod tests;
