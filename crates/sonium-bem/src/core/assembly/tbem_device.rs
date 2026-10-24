// Device assembly handoff for the constant-panel CUDA TBEM workspace.

// Rust guideline compliant 2026-02-21
use super::*;
use sonium_compute::{DeviceOperator, Transpose};

mod iterative { include!("tbem_gmres.rs"); }
#[doc(inline)]
pub use iterative::CudaTbemGmresSession;

/// Resident dense TBEM matrix and RHS, published only after checked assembly.
///
/// Storage is retained by the assembly workspace. This operator can be passed
/// directly to a device solver; explicit downloads are optional caller actions.
pub struct CudaTbemDeviceSystem {
    n: usize,
    matrix: CudaBuffer<Complex64>,
    rhs: CudaBuffer<Complex64>,
    routes: CudaBuffer<i32>,
    coefficients: CudaBuffer<Complex64>,
    matrix_signature: Vec<Complex64>,
    matrix_version: u64,
}

impl CudaTbemDeviceSystem {
    /// Exclusive DOF ceiling for dense CUDA TBEM; larger systems require compression.
    ///
    /// This resource policy is shared by assembly and automatic representation
    /// selection. It is not a measured CPU/GPU performance crossover.
    pub const DOF_LIMIT: usize = 1000;

    /// Return the resident column-major matrix.
    pub fn matrix(&self) -> &CudaBuffer<Complex64> {
        &self.matrix
    }
    /// Return the resident right-hand side.
    pub fn rhs(&self) -> &CudaBuffer<Complex64> {
        &self.rhs
    }
}

impl DeviceOperator<CudaBackend, Complex64> for CudaTbemDeviceSystem {
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
            log::error!("CUDA TBEM matrix action failed: {error}");
        }
    }
    fn try_apply(
        &self,
        backend: &CudaBackend,
        x: &CudaBuffer<Complex64>,
        y: &mut CudaBuffer<Complex64>,
    ) -> sonium_compute::BackendResult<()> {
        // GEMV uses row-major storage; the column-major matrix is its transpose.
        backend.try_gemv(
            Transpose::Trans,
            self.n,
            self.n,
            Complex64::new(1.0, 0.0),
            &self.matrix,
            self.n,
            x,
            Complex64::new(0.0, 0.0),
            y,
        )
    }
    fn try_apply_colmajor(
        &self,
        backend: &CudaBackend,
        rhs: usize,
        x: &CudaBuffer<Complex64>,
        y: &mut CudaBuffer<Complex64>,
    ) -> sonium_compute::BackendResult<()> {
        // All buffers are column-major. In the row-major API, compute
        // Y^T = X^T A^T with dimensions rhs-by-n, preserving block layout.
        backend.try_gemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            rhs,
            self.n,
            self.n,
            Complex64::new(1.0, 0.0),
            x,
            self.n,
            &self.matrix,
            self.n,
            Complex64::new(0.0, 0.0),
            y,
            self.n,
        )
    }
}

pub(super) fn quadrature_schedule(
    elements: &[Element],
    physics: &PhysicsParams,
) -> Vec<[usize; 4]> {
    elements
        .iter()
        .filter(|e| !e.property.is_evaluation())
        .map(|e| {
            let p = QuadratureParams::for_ka(physics.wave_number * e.area.sqrt().max(1.0e-10));
            [
                p.edge_gauss_order,
                p.subelement_gauss_order,
                p.edge_sections,
                p.subtriangles_per_section,
            ]
        })
        .collect()
}

impl CudaTbemAssemblyWorkspace {
    /// Assemble and retain a dense device matrix/RHS without downloading pair channels.
    ///
    /// Constant Tri3 systems below 1000 DOFs are admitted. Larger systems must use
    /// a compressed representation. Geometry and quadrature schedules must match
    /// construction; boundary values and physics may change within that schedule.
    ///
    /// # Errors
    /// Rejects unsupported input, changed geometry/quadrature, allocation failures,
    /// and invalid device results. No system reference is published on failure.
    pub fn assemble_device(
        &mut self,
        elements: &[Element],
        nodes: &[[f64; 3]],
        physics: &PhysicsParams,
        beta: Complex64,
        backend: &CudaBackend,
    ) -> Result<&CudaTbemDeviceSystem, CudaTbemAssemblyError> {
        let invalid = |reason: &str| CudaTbemAssemblyError::InvalidInput {
            reason: reason.into(),
        };
        let n = validate_cuda_tbem_inputs(elements, nodes, physics, beta)?;
        if n >= CudaTbemDeviceSystem::DOF_LIMIT {
            return Err(invalid(
                "dense CUDA TBEM requires fewer than 1000 DOFs; use H-matrix or FMM",
            ));
        }
        if cuda_tbem_geometry_key(elements, nodes) != self.geometry_key {
            return Err(invalid("resident CUDA TBEM geometry cache is stale"));
        }
        if quadrature_schedule(elements, physics) != self.quadrature_schedule {
            return Err(invalid(
                "CUDA TBEM quadrature schedule changed; rebuild the workspace",
            ));
        }
        if self.device_system.is_none() {
            let mut routes = vec![i32::MIN; n * n];
            for (i, src) in elements
                .iter()
                .enumerate()
                .filter(|(_, e)| !e.property.is_evaluation())
            {
                for (j, fld) in elements
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| !e.property.is_evaluation())
                {
                    let route = match self.routes[i][j] {
                        PairRoute::Gpu(k) => {
                            -i32::try_from(k).map_err(|_| invalid("regular pair index overflow"))?
                                - 1
                        }
                        PairRoute::GpuQuadrature(k) => i32::try_from(k)
                            .map_err(|_| invalid("quadrature pair index overflow"))?,
                        PairRoute::Self_ => {
                            return Err(invalid("active pair has no native quadrature route"));
                        }
                    };
                    routes[src.dof_addresses[0] + fld.dof_addresses[0] * n] = route;
                }
            }
            let mut device_routes = backend
                .raw_alloc::<i32>(routes.len())
                .map_err(BackendError::from)?;
            backend.try_copy_h2d(&routes, &mut device_routes)?;
            self.device_system = Some(CudaTbemDeviceSystem {
                n,
                matrix_signature: Vec::new(),
                matrix_version: 0,
                routes: device_routes,
                matrix: backend
                    .raw_alloc::<Complex64>(n * n)
                    .map_err(BackendError::from)?,
                rhs: backend
                    .raw_alloc::<Complex64>(n)
                    .map_err(BackendError::from)?,
                coefficients: backend
                    .raw_alloc::<Complex64>(10 * n)
                    .map_err(BackendError::from)?,
            });
        }
        let mut weights = vec![Complex64::default(); 10 * n];
        let gamma = Complex64::new(physics.gamma(), 0.0);
        let tau = Complex64::new(physics.tau, 0.0);
        let pressure = Complex64::new(0.0, physics.pressure_factor);
        for e in elements.iter().filter(|e| !e.property.is_evaluation()) {
            let column = e.dof_addresses[0];
            let w = &mut weights[10 * column..10 * (column + 1)];
            let (kind, values) = get_bc_type_and_value(&e.boundary_condition);
            let avg = values.iter().sum::<Complex64>() / values.len() as f64;
            match kind {
                0 => {
                    w[1] = gamma * tau;
                    w[3] = beta;
                    if let Some(admittance) = e.boundary_condition.admittance() {
                        w[0] = -gamma * tau * admittance * pressure;
                        w[2] = -beta * admittance * pressure;
                        w[8] = -admittance * pressure * beta * tau * 0.5;
                    }
                    w[4] = gamma * tau * avg * pressure;
                    w[6] = beta * avg * pressure;
                    w[8] -= gamma * 0.5;
                    w[9] = avg * pressure * beta * tau * 0.5;
                }
                1 => {
                    w[0] = -gamma * tau;
                    w[2] = -beta;
                    w[5] = gamma * tau * avg;
                    w[7] = beta * avg;
                    w[8] = beta * tau * 0.5;
                    w[9] = -avg * gamma * 0.5;
                }
                _ => return Err(invalid("unsupported boundary condition")),
            }
        }
        // Host derivative caches cannot refer to pair outputs from a prior frequency.
        self.last_physics_key = None;
        if !self.regular_pairs.is_empty() {
            backend.try_bem_regular_pair_c64_device(
                &self.regular_pairs,
                &mut self.regular_output,
                physics.wave_number,
                physics.harmonic_factor,
            )?;
        }
        if !self.quadrature_pairs.is_empty() {
            backend.try_bem_quadrature_pair_c64_device(
                &self.quadrature_pairs,
                &self.quadrature_samples,
                &mut self.quadrature_output,
                physics.wave_number,
                physics.harmonic_factor,
            )?;
        }
        let system = self
            .device_system
            .as_mut()
            .ok_or_else(|| invalid("device system was not initialized"))?;
        backend.try_copy_h2d(&weights, &mut system.coefficients)?;
        backend.try_bem_dense_system_c64_device(
            n,
            &self.regular_output,
            &self.quadrature_output,
            &system.routes,
            &system.coefficients,
            &mut system.matrix,
            &mut system.rhs,
        )?;
        // Compare exact matrix coefficients, excluding RHS weights. Geometry
        // and quadrature are fixed by the workspace; no hash collision is possible.
        let mut signature = Vec::with_capacity(5 * n + 1);
        signature.push(Complex64::new(physics.wave_number, physics.harmonic_factor));
        for column in weights.as_chunks::<10>().0 {
            signature.extend_from_slice(&column[..4]);
            signature.push(column[8]);
        }
        if signature != system.matrix_signature {
            system.matrix_version = system
                .matrix_version
                .checked_add(1)
                .ok_or_else(|| invalid("matrix version overflow; rebuild workspace"))?;
            system.matrix_signature = signature;
        }
        Ok(system)
    }
}

/// Retained dense CUDA assembly, factorization and certified solution storage.
///
/// Constant Tri3 systems below 1000 DOFs use direct LU. Geometry and quadrature
/// schedules remain fixed; changed numeric coefficients reuse LU workspace.
/// This session does not admit larger systems through a dense fallback.
pub struct CudaTbemSolveSession {
    assembly: CudaTbemAssemblyWorkspace,
    lu: Option<sonium_compute::CudaDenseLuC64>,
    solution: CudaBuffer<Complex64>,
    residual: CudaBuffer<Complex64>,
    combined_rhs: CudaBuffer<Complex64>,
    published: bool,
    factored_version: Option<u64>,
}

impl std::fmt::Debug for CudaTbemSolveSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaTbemSolveSession")
            .field("n", &self.solution.len())
            .field("published", &self.published)
            .finish_non_exhaustive()
    }
}

/// Residual certification for a completed dense CUDA BEM solve.
#[derive(Debug, Clone, Copy)]
pub struct CudaTbemSolveReport {
    /// True when an unchanged matrix reused its existing LU factors.
    pub factorization_reused: bool,
    /// Euclidean norm of the original-system residual.
    pub residual_norm: f64,
    /// Euclidean norm of the assembled RHS.
    pub rhs_norm: f64,
    /// Residual divided by RHS norm, or absolute residual for a zero RHS.
    pub relative_residual: f64,
}

/// Failure before publication of a dense CUDA BEM solution.
#[derive(Debug, thiserror::Error)]
pub enum CudaTbemSolveError {
    /// Assembly input or device execution failed.
    #[error(transparent)]
    Assembly(#[from] CudaTbemAssemblyError),
    /// Factorization, solve or certification failed on the backend.
    #[error(transparent)]
    Backend(#[from] BackendError),
    /// The requested tolerance was invalid.
    #[error("CUDA TBEM solve tolerance must be finite and positive")]
    InvalidTolerance,
    /// The original-system residual exceeded the requested tolerance.
    #[error("CUDA TBEM relative residual {residual} exceeds tolerance {tolerance}")]
    Residual {
        /// Measured residual ratio.
        residual: f64,
        /// Requested upper bound.
        tolerance: f64,
    },
}

impl CudaTbemSolveSession {
    /// Prepare retained assembly and solution storage for a dense CUDA solve.
    ///
    /// # Errors
    /// Rejects unsupported meshes, systems at or above 1000 DOFs and CUDA failures.
    pub fn new(
        elements: &[Element],
        nodes: &[[f64; 3]],
        physics: &PhysicsParams,
        backend: &CudaBackend,
    ) -> Result<Self, CudaTbemSolveError> {
        let n = validate_cuda_tbem_inputs(elements, nodes, physics, Complex64::default())?;
        if n >= CudaTbemDeviceSystem::DOF_LIMIT {
            return Err(CudaTbemAssemblyError::InvalidInput {
                reason: "dense CUDA TBEM requires fewer than 1000 DOFs; use H-matrix or FMM".into(),
            }
            .into());
        }
        Ok(Self {
            assembly: CudaTbemAssemblyWorkspace::new(elements, nodes, physics, backend)?,
            lu: None,
            solution: backend
                .raw_alloc::<Complex64>(n)
                .map_err(BackendError::from)?,
            residual: backend
                .raw_alloc::<Complex64>(n)
                .map_err(BackendError::from)?,
            combined_rhs: backend
                .raw_alloc::<Complex64>(n)
                .map_err(BackendError::from)?,
            published: false,
            factored_version: None,
        })
    }

    /// Borrow the certified device solution, absent after any failed solve.
    pub fn solution(&self) -> Option<&CudaBuffer<Complex64>> {
        self.published.then_some(&self.solution)
    }

    /// Assemble, refactor and solve while retaining bulk data on CUDA.
    ///
    /// Numeric assembly runs each call; unchanged matrices reuse LU factors.
    /// Changed matrices reuse pivots, factor workspace and vectors. Compact validation
    /// flags, factorization status and residual norms cross the host boundary.
    ///
    /// # Errors
    /// Rejects changed geometry/quadrature, invalid tolerance, singular systems,
    /// backend failures or a residual exceeding tolerance. Any error unpublishes
    /// the previous solution; recovery requires a successful subsequent solve.
    pub fn solve(
        &mut self,
        elements: &[Element],
        nodes: &[[f64; 3]],
        physics: &PhysicsParams,
        beta: Complex64,
        backend: &CudaBackend,
        tolerance: f64,
    ) -> Result<CudaTbemSolveReport, CudaTbemSolveError> {
        self.solve_with_incident_rhs(elements, nodes, physics, beta, backend, tolerance, None)
    }

    /// Solve with an additional resident incident-field RHS.
    ///
    /// The extra RHS is added to the boundary-condition load. It must use the
    /// same DOF order and unscaled formulation as the assembled system.
    ///
    /// # Errors
    /// Returns the same errors as `solve`, plus incompatible incident-buffer
    /// dimensions, non-finite values or foreign-backend storage.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_with_incident_rhs(
        &mut self,
        elements: &[Element],
        nodes: &[[f64; 3]],
        physics: &PhysicsParams,
        beta: Complex64,
        backend: &CudaBackend,
        tolerance: f64,
        incident_rhs: Option<&CudaBuffer<Complex64>>,
    ) -> Result<CudaTbemSolveReport, CudaTbemSolveError> {
        self.published = false;
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(CudaTbemSolveError::InvalidTolerance);
        }
        let system = self
            .assembly
            .assemble_device(elements, nodes, physics, beta, backend)?;
        backend.try_copy_d2d(system.rhs(), &mut self.combined_rhs)?;
        if let Some(incident) = incident_rhs {
            backend.try_axpy(Complex64::new(1.0, 0.0), incident, &mut self.combined_rhs)?;
        }
        let factorization_reused = self.factored_version == Some(system.matrix_version);
        if !factorization_reused {
            self.factored_version = None;
            if let Some(lu) = self.lu.as_mut() {
                backend
                    .dense_lu_c64_refactor_device(lu, system.matrix())
                    .map_err(BackendError::from)?;
            } else {
                self.lu = Some(
                    backend
                        .dense_lu_c64_from_device(system.n, system.matrix())
                        .map_err(BackendError::from)?,
                );
            }
            self.factored_version = Some(system.matrix_version);
        }
        let lu = self
            .lu
            .as_ref()
            .ok_or_else(|| BackendError::InvalidArgument {
                operation: "tbem_solve",
                reason: "factorization unavailable".into(),
            })?;
        backend
            .dense_lu_c64_solve(
                lu,
                &self.combined_rhs,
                &mut self.solution,
                Transpose::NoTrans,
            )
            .map_err(BackendError::from)?;
        system.try_apply(backend, &self.solution, &mut self.residual)?;
        backend.try_axpy(
            Complex64::new(-1.0, 0.0),
            &self.combined_rhs,
            &mut self.residual,
        )?;
        let residual_norm =
            backend.try_nrm2_with_mode(sonium_compute::ReductionMode::Fast, &self.residual)?;
        let rhs_norm =
            backend.try_nrm2_with_mode(sonium_compute::ReductionMode::Fast, &self.combined_rhs)?;
        let relative_residual = if rhs_norm > 0.0 {
            residual_norm / rhs_norm
        } else {
            residual_norm
        };
        if !relative_residual.is_finite() || relative_residual > tolerance {
            return Err(CudaTbemSolveError::Residual {
                residual: relative_residual,
                tolerance,
            });
        }
        self.published = true;
        Ok(CudaTbemSolveReport {
            factorization_reused,
            residual_norm,
            rhs_norm,
            relative_residual,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::mesh::generators::generate_icosphere_mesh;

    #[test]
    fn native_device_tbem_mixed_boundaries_and_reuse() {
        let backend = match CudaBackend::new_default() {
            Ok(backend) => backend,
            Err(error) => {
                assert!(
                    std::env::var("SONIUM_REQUIRE_CUDA").is_err(),
                    "CUDA required: {error}"
                );
                return;
            }
        };
        let mesh = generate_icosphere_mesh(1.0, 1);
        let mut elements = mesh.elements.clone();
        for (i, element) in elements.iter_mut().enumerate() {
            element.dof_addresses = vec![i];
            element.boundary_condition = match i % 5 {
                0 => BoundaryCondition::Pressure(vec![Complex64::new(0.2, -0.1)]),
                1 => BoundaryCondition::Velocity(vec![Complex64::new(0.01, 0.02)]),
                2 => BoundaryCondition::VelocityWithAdmittance {
                    velocity: vec![Complex64::new(0.01, -0.01)],
                    admittance: Complex64::new(0.001, 0.002),
                },
                3 => BoundaryCondition::TransferAdmittance {
                    admittance: Complex64::new(0.002, -0.001),
                },
                _ => BoundaryCondition::TransferWithSurfaceAdmittance {
                    transfer_admittance: Complex64::new(0.001, 0.002),
                    surface_admittance: Complex64::new(0.002, -0.001),
                },
            };
        }
        let physics = PhysicsParams::new(343.0 / (2.0 * std::f64::consts::PI), 343.0, 1.21, false);
        let mut workspace =
            CudaTbemAssemblyWorkspace::new(&elements, &mesh.nodes, &physics, &backend).unwrap();
        let mut session =
            CudaTbemSolveSession::new(&elements, &mesh.nodes, &physics, &backend).unwrap();
        for pass in 0..4 {
            let physics = PhysicsParams::new(
                (1.0 + pass.min(2) as f64 * 0.01) * 343.0 / (2.0 * std::f64::consts::PI),
                343.0,
                1.21,
                false,
            );
            elements[0].boundary_condition =
                BoundaryCondition::Pressure(vec![Complex64::new(0.2 + pass as f64 * 0.1, -0.1)]);
            let beta = physics.burton_miller_beta();
            let cpu = build_tbem_system_with_beta_parallel(&elements, &mesh.nodes, &physics, beta);
            let before = backend.begin_runtime_stats();
            let system = workspace
                .assemble_device(&elements, &mesh.nodes, &physics, beta, &backend)
                .unwrap();
            let stats = backend.runtime_stats_since(before);
            if pass > 0 {
                assert_eq!(
                    stats.allocation_operations, 0,
                    "warm assembly allocations: {stats:?}"
                );
                assert_eq!(
                    stats.transfer.h2d_bytes,
                    10 * elements.len() * size_of::<Complex64>()
                );
                assert!(
                    stats.transfer.d2h_bytes <= 12,
                    "bulk download during assembly: {stats:?}"
                );
            }
            let n = elements.len();
            let mut matrix = vec![Complex64::default(); n * n];
            let mut rhs = vec![Complex64::default(); n];
            backend.try_copy_d2h(system.matrix(), &mut matrix).unwrap();
            backend.try_copy_d2h(system.rhs(), &mut rhs).unwrap();
            let mut error = 0.0;
            let mut reference = 0.0;
            for column in 0..n {
                for row in 0..n {
                    error += (matrix[row + column * n] - cpu.matrix[[row, column]]).norm_sqr();
                    reference += cpu.matrix[[row, column]].norm_sqr();
                }
            }
            assert!(
                (error / reference).sqrt() < 1e-9,
                "matrix relative error {}",
                (error / reference).sqrt()
            );
            let error = rhs
                .iter()
                .zip(cpu.rhs.iter())
                .map(|(a, b)| (*a - *b).norm_sqr())
                .sum::<f64>();
            let reference = cpu.rhs.iter().map(|x| x.norm_sqr()).sum::<f64>();
            assert!(
                (error / reference).sqrt() < 1e-9,
                "RHS relative error {}",
                (error / reference).sqrt()
            );
            let probe: Vec<_> = (0..2 * n)
                .map(|i| Complex64::new(0.1 + i as f64 * 0.002, -0.2))
                .collect();
            let mut input = backend.raw_alloc::<Complex64>(2 * n).unwrap();
            let mut output = backend.raw_alloc::<Complex64>(2 * n).unwrap();
            backend.try_copy_h2d(&probe, &mut input).unwrap();
            system
                .try_apply_colmajor(&backend, 2, &input, &mut output)
                .unwrap();
            let mut actual = vec![Complex64::default(); 2 * n];
            backend.try_copy_d2h(&output, &mut actual).unwrap();
            for column in 0..2 {
                for row in 0..n {
                    let expected = (0..n)
                        .map(|j| cpu.matrix[[row, j]] * probe[j + column * n])
                        .sum::<Complex64>();
                    assert!(
                        (actual[row + column * n] - expected).norm()
                            < 1e-9 * expected.norm().max(1.0)
                    );
                }
            }
            let before = backend.begin_runtime_stats();
            let report = session
                .solve(&elements, &mesh.nodes, &physics, beta, &backend, 1e-9)
                .unwrap();
            let stats = backend.runtime_stats_since(before);
            assert_eq!(report.factorization_reused, pass == 3);
            eprintln!(
                "BEM resident solve pass={pass} residual={} allocations={} h2d={} d2h={}",
                report.relative_residual,
                stats.allocation_operations,
                stats.transfer.h2d_bytes,
                stats.transfer.d2h_bytes
            );
            if pass > 0 {
                assert_eq!(stats.allocation_operations, 0);
                assert_eq!(stats.transfer.h2d_bytes, 10 * n * size_of::<Complex64>());
                assert!(
                    stats.transfer.d2h_bytes <= 64,
                    "bulk solve download: {stats:?}"
                );
            }
            let mut solution = vec![Complex64::default(); n];
            backend
                .try_copy_d2h(session.solution().unwrap(), &mut solution)
                .unwrap();
            let cpu_residual = (0..n)
                .map(|i| {
                    ((0..n)
                        .map(|j| cpu.matrix[[i, j]] * solution[j])
                        .sum::<Complex64>()
                        - cpu.rhs[i])
                        .norm_sqr()
                })
                .sum::<f64>()
                .sqrt()
                / reference.sqrt();
            assert!(
                cpu_residual < 1e-9,
                "CPU original-system residual {cpu_residual}"
            );
        }
        assert!(
            session
                .solve(
                    &elements,
                    &mesh.nodes,
                    &physics,
                    physics.burton_miller_beta(),
                    &backend,
                    f64::NAN
                )
                .is_err()
        );
        assert!(session.solution().is_none());
        backend.inject_failure(sonium_compute::CudaFailureInjectionPoint::Launch);
        assert!(
            session
                .solve(
                    &elements,
                    &mesh.nodes,
                    &physics,
                    physics.burton_miller_beta(),
                    &backend,
                    1e-9
                )
                .is_err()
        );
        assert!(session.solution().is_none());
        backend.recover_after_error().unwrap();
        session
            .solve(
                &elements,
                &mesh.nodes,
                &physics,
                physics.burton_miller_beta(),
                &backend,
                1e-9,
            )
            .unwrap();
        assert!(session.solution().is_some());
    }
}
