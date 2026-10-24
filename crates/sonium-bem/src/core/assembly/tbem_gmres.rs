// Retained GMRES for device-assembled constant-panel dense BEM systems.

// Rust guideline compliant 2026-02-21
use super::*;
use crate::core::bem_solver::BemError;
use sonium_compute::{Buffer, CudaDenseJacobi, MemoryPool};
use sonium_solvers::iterative::{
    BackendKrylovConfig, BackendKrylovStatus, DeviceGmresSession, DeviceInitialGuess,
};

/// Retained native GMRES with refreshed point-Jacobi for dense CUDA BEM.
///
/// Geometry and DOF order remain fixed. Frequency changes rebuild assembly
/// storage when the quadrature schedule changes, retaining Krylov storage and
/// the previous solution. Boundary coefficients may change. Assembly uploads
/// boundary coefficients, but matrix, diagonal and Krylov vectors stay resident.
/// The dense limit remains below 1000 DOFs; use compressed sessions above it.
/// No GCRO-DR subspace recycling is implied by previous-solution warm starts.
/// Assembly uses legacy allocations outside the pool quota; new solver storage
/// is pooled. Point Jacobi is not sufficient for every BEM configuration.
pub struct CudaTbemGmresSession<'a> {
    pool: &'a MemoryPool<CudaBackend>,
    assembly: CudaTbemAssemblyWorkspace,
    preconditioner: CudaDenseJacobi<'a, Complex64>,
    krylov: DeviceGmresSession<'a, CudaBackend, Complex64>,
    rhs: Buffer<Complex64, CudaBackend>,
    controls: BackendKrylovConfig<f64>,
    published: bool,
    assembly_rebuilds: usize,
}

impl std::fmt::Debug for CudaTbemGmresSession<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaTbemGmresSession")
            .field("dimension", &self.rhs.len())
            .field("published", &self.published)
            .field("assembly_rebuilds", &self.assembly_rebuilds)
            .finish_non_exhaustive()
    }
}

fn failed(error: impl std::fmt::Display) -> BemError {
    BemError::SolverFailed(error.to_string())
}

impl<'a> CudaTbemGmresSession<'a> {
    /// Prepare native assembly, diagonal preconditioning and persistent Krylov storage.
    ///
    /// # Errors
    /// Rejects invalid/native-disabled controls, unsupported geometry, large
    /// dense systems, invalid diagonals, quota exhaustion and backend failures.
    pub fn try_new(
        pool: &'a MemoryPool<CudaBackend>,
        elements: &[Element],
        nodes: &[[f64; 3]],
        physics: &PhysicsParams,
        beta: Complex64,
        controls: BackendKrylovConfig<f64>,
    ) -> Result<Self, BemError> {
        if !controls.require_native
            || controls.restart == 0
            || controls.max_iterations == 0
            || !controls.tolerance.is_finite()
            || controls.tolerance <= 0.0
        {
            return Err(BemError::InvalidParameters(
                "invalid native TBEM GMRES controls".into(),
            ));
        }
        if let Some(error) = pool.backend().pending_error().map_err(failed)? {
            return Err(failed(error));
        }
        sonium_solvers::iterative::validate_backend_krylov::<_, Complex64>(
            pool.backend(),
            sonium_solvers::iterative::BackendKrylovMethod::Gmres,
            true,
        ).map_err(failed)?;
        let n = validate_cuda_tbem_inputs(elements, nodes, physics, beta).map_err(failed)?;
        if n >= CudaTbemDeviceSystem::DOF_LIMIT {
            return Err(BemError::InvalidParameters(
                "dense CUDA TBEM requires fewer than 1000 DOFs".into(),
            ));
        }
        let backend = pool.backend();
        let mut assembly =
            CudaTbemAssemblyWorkspace::new(elements, nodes, physics, backend).map_err(failed)?;
        let system = assembly
            .assemble_device(elements, nodes, physics, beta, backend)
            .map_err(failed)?;
        let preconditioner =
            CudaDenseJacobi::try_new(pool, system.matrix(), n, n).map_err(failed)?;
        let mut krylov = DeviceGmresSession::try_new(pool, n).map_err(failed)?;
        let cycle = controls.restart.min(controls.max_iterations);
        krylov.try_reserve_cycle(cycle).map_err(failed)?;
        backend
            .try_reserve_krylov_reductions::<Complex64>(cycle)
            .map_err(failed)?;
        Ok(Self {
            pool,
            assembly,
            preconditioner,
            krylov,
            rhs: pool.try_alloc(n).map_err(failed)?,
            controls,
            published: false,
            assembly_rebuilds: 0,
        })
    }

    /// Reassemble numerics and solve with an optional resident incident RHS.
    ///
    /// The incident RHS uses the assembled operator's unscaled DOF convention.
    /// Successful solves certify the original assembled-system residual. Every
    /// error unpublishes output and requires a zero-start retry; recover backend
    /// errors before retrying. Geometry changes require a new session. Quadrature
    /// changes rebuild assembly storage and may temporarily retain both workspaces.
    ///
    /// # Errors
    /// Rejects changed geometry, invalid incident storage or values,
    /// unavailable warm starts, failed refresh, nonconvergence and backend errors.
    pub fn solve_device(
        &mut self,
        elements: &[Element],
        nodes: &[[f64; 3]],
        physics: &PhysicsParams,
        beta: Complex64,
        incident: Option<&CudaBuffer<Complex64>>,
        initial: DeviceInitialGuess,
    ) -> Result<BackendKrylovStatus<f64>, BemError> {
        let previous = std::mem::replace(&mut self.published, false);
        if initial == DeviceInitialGuess::PreviousSolution && !previous {
            return Err(failed("no certified previous TBEM GMRES solution"));
        }
        let backend = self.pool.backend();
        if let Some(incident) = incident {
            if incident.len() != self.rhs.len()
                || incident.raw().is_none_or(|raw| {
                    !std::sync::Arc::ptr_eq(raw.stream(), backend.stream())
                        || incident.len().checked_mul(std::mem::size_of::<Complex64>())
                            != Some(raw.len())
                })
            {
                return Err(failed("invalid incident RHS dimensions, storage or stream"));
            }
            if backend.try_all_finite_native(incident).map_err(failed)? != Some(true) {
                return Err(failed(
                    "invalid incident RHS or unavailable native validation",
                ));
            }
        }
        validate_cuda_tbem_inputs(elements, nodes, physics, beta).map_err(failed)?;
        if cuda_tbem_geometry_key(elements, nodes) != self.assembly.geometry_key {
            return Err(failed("resident CUDA TBEM geometry cache is stale"));
        }
        let rebuild = quadrature_schedule(elements, physics) != self.assembly.quadrature_schedule;
        if rebuild {
            // Prepare the replacement completely before retiring the old assembly.
            // Krylov vectors and the diagonal buffer depend only on the fixed DOFs.
            let mut replacement = CudaTbemAssemblyWorkspace::new(elements, nodes, physics, backend)
                .map_err(failed)?;
            replacement
                .assemble_device(elements, nodes, physics, beta, backend)
                .map_err(failed)?;
            self.assembly = replacement;
            self.assembly_rebuilds += 1;
        }
        let system = if rebuild {
            self.assembly
                .device_system
                .as_ref()
                .ok_or_else(|| failed("missing rebuilt TBEM system"))?
        } else {
            self.assembly
                .assemble_device(elements, nodes, physics, beta, backend)
                .map_err(failed)?
        };
        self.preconditioner
            .try_update(system.matrix(), self.rhs.len())
            .map_err(failed)?;
        backend
            .try_copy_d2d(system.rhs(), self.rhs.raw_mut())
            .map_err(failed)?;
        if let Some(incident) = incident {
            backend
                .try_axpy(Complex64::new(1.0, 0.0), incident, self.rhs.raw_mut())
                .map_err(failed)?;
        }
        let status = self
            .krylov
            .solve(
                system,
                &self.preconditioner,
                self.rhs.raw(),
                initial,
                &self.controls,
                None,
            )
            .map_err(failed)?;
        if !status.converged {
            return Err(failed(format!("TBEM GMRES did not converge: {status:?}")));
        }
        self.published = true;
        Ok(status)
    }

    /// Borrow the last certified device solution without a host transfer.
    pub fn solution_device(&self) -> Option<&CudaBuffer<Complex64>> {
        self.published.then(|| self.krylov.solution()).flatten()
    }

    /// Return completed quadrature workspace replacements, excluding initial preparation.
    ///
    /// A replacement is counted even if the subsequent linear solve fails.
    pub fn assembly_rebuilds(&self) -> usize {
        self.assembly_rebuilds
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::mesh::generators::generate_icosphere_mesh;
    use sonium_compute::{CudaFailureInjectionPoint, MemoryPoolConfig};

    #[test]
    fn native_tbem_gmres_quadrature_sweep() {
        let backend = match CudaBackend::new_default() {
            Ok(backend) => backend,
            Err(error) => {
                assert!(
                    std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                    "{error}"
                );
                return;
            }
        };
        let pool = MemoryPool::new(backend, MemoryPoolConfig::default()).unwrap();
        let backend = pool.backend();
        let mesh = generate_icosphere_mesh(1.0, 0);
        let mut elements = mesh.elements.clone();
        for (i, element) in elements.iter_mut().enumerate() {
            element.dof_addresses = vec![i];
            element.boundary_condition =
                BoundaryCondition::Velocity(vec![Complex64::new(0.01, 0.02)]);
        }
        let physics_at = |ka: f64| {
            PhysicsParams::new(
                ka * 343.0 / (2.0 * std::f64::consts::PI * elements[0].area.sqrt()),
                343.0,
                1.21,
                false,
            )
        };
        let physics = physics_at(0.1);
        let mut session = CudaTbemGmresSession::try_new(
            &pool,
            &elements,
            &mesh.nodes,
            &physics,
            physics.burton_miller_beta(),
            BackendKrylovConfig {
                restart: 20,
                max_iterations: 200,
                tolerance: 1e-10,
                require_native: true,
                ..Default::default()
            },
        )
        .unwrap();
        // Cross every schedule boundary, repeat within each schedule, then return.
        for (pass, ka) in [0.1, 0.11, 0.5, 0.51, 1.2, 1.21, 2.2, 2.21, 0.1, 0.11]
            .into_iter()
            .enumerate()
        {
            let physics = physics_at(ka);
            let beta = physics.burton_miller_beta();
            let before = backend.begin_runtime_stats();
            let report = session
                .solve_device(
                    &elements,
                    &mesh.nodes,
                    &physics,
                    beta,
                    None,
                    if pass == 0 {
                        DeviceInitialGuess::Zero
                    } else {
                        DeviceInitialGuess::PreviousSolution
                    },
                )
                .unwrap();
            let stats = backend.runtime_stats_since(before);
            assert!(report.converged);
            assert_eq!(session.assembly_rebuilds(), pass / 2);
            if pass % 2 == 1 {
                assert_eq!(stats.allocation_operations, 0, "{stats:?}");
            }
            let mut actual = vec![Complex64::default(); elements.len()];
            backend
                .try_copy_d2h(session.solution_device().unwrap(), &mut actual)
                .unwrap();
            let cpu = build_tbem_system_with_beta_parallel(&elements, &mesh.nodes, &physics, beta);
            let residual = (0..elements.len())
                .map(|i| {
                    let ax = (0..elements.len())
                        .map(|j| cpu.matrix[[i, j]] * actual[j])
                        .sum::<Complex64>();
                    (ax - cpu.rhs[i]).norm_sqr()
                })
                .sum::<f64>();
            let rhs_norm = cpu.rhs.iter().map(|x| x.norm_sqr()).sum::<f64>();
            assert!((residual / rhs_norm).sqrt() < 1e-7, "ka={ka}");
        }
        let physics = physics_at(0.5);
        backend.inject_failure(CudaFailureInjectionPoint::Launch);
        assert!(
            session
                .solve_device(
                    &elements,
                    &mesh.nodes,
                    &physics,
                    physics.burton_miller_beta(),
                    None,
                    DeviceInitialGuess::PreviousSolution
                )
                .is_err()
        );
        assert!(session.solution_device().is_none());
        backend.recover_after_error().unwrap();
        assert!(
            session
                .solve_device(
                    &elements,
                    &mesh.nodes,
                    &physics,
                    physics.burton_miller_beta(),
                    None,
                    DeviceInitialGuess::PreviousSolution
                )
                .is_err()
        );
        session
            .solve_device(
                &elements,
                &mesh.nodes,
                &physics,
                physics.burton_miller_beta(),
                None,
                DeviceInitialGuess::Zero,
            )
            .unwrap();
        assert_eq!(session.assembly_rebuilds(), 5);
    }

    #[test]
    fn native_tbem_gmres_numeric_refresh_and_recovery() {
        let backend = match CudaBackend::new_default() {
            Ok(backend) => backend,
            Err(error) => {
                assert!(
                    std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                    "{error}"
                );
                return;
            }
        };
        let pool = MemoryPool::new(backend, MemoryPoolConfig::default()).unwrap();
        let backend = pool.backend();
        let mesh = generate_icosphere_mesh(1.0, 1);
        let mut elements = mesh.elements.clone();
        for (i, element) in elements.iter_mut().enumerate() {
            element.dof_addresses = vec![i];
            element.boundary_condition = match i % 3 {
                0 => BoundaryCondition::Pressure(vec![Complex64::new(0.2, -0.1)]),
                1 => BoundaryCondition::Velocity(vec![Complex64::new(0.01, 0.02)]),
                _ => BoundaryCondition::VelocityWithAdmittance {
                    velocity: vec![Complex64::new(0.01, -0.01)],
                    admittance: Complex64::new(0.001, 0.002),
                },
            };
        }
        let n = elements.len();
        let frequency = 343.0 / (2.0 * std::f64::consts::PI);
        let mut physics = PhysicsParams::new(frequency, 343.0, 1.21, false);
        let controls = BackendKrylovConfig {
            restart: 80,
            max_iterations: 400,
            tolerance: 1e-10,
            require_native: true,
            ..Default::default()
        };
        let mut session = CudaTbemGmresSession::try_new(
            &pool,
            &elements,
            &mesh.nodes,
            &physics,
            physics.burton_miller_beta(),
            controls,
        )
        .unwrap();
        let mut direct =
            CudaTbemSolveSession::new(&elements, &mesh.nodes, &physics, backend).unwrap();
        let mut incident = pool.try_alloc::<Complex64>(n).unwrap();
        let incident_host = vec![Complex64::new(0.03, -0.02); n];
        backend
            .try_copy_h2d(&incident_host, incident.raw_mut())
            .unwrap();
        assert!(
            session
                .solve_device(
                    &elements,
                    &mesh.nodes,
                    &physics,
                    physics.burton_miller_beta(),
                    None,
                    DeviceInitialGuess::PreviousSolution
                )
                .is_err()
        );
        for pass in 0..4 {
            physics =
                PhysicsParams::new(frequency * (1.0 + pass as f64 * 0.01), 343.0, 1.21, false);
            elements[0].boundary_condition =
                BoundaryCondition::Pressure(vec![Complex64::new(0.2 + pass as f64 * 0.1, -0.1)]);
            let beta = physics.burton_miller_beta();
            let extra = if pass % 2 == 0 {
                None
            } else {
                Some(incident.raw())
            };
            let before = backend.begin_runtime_stats();
            let report = session
                .solve_device(
                    &elements,
                    &mesh.nodes,
                    &physics,
                    beta,
                    extra,
                    if pass == 0 {
                        DeviceInitialGuess::Zero
                    } else {
                        DeviceInitialGuess::PreviousSolution
                    },
                )
                .unwrap();
            let stats = backend.runtime_stats_since(before);
            assert!(report.converged);
            if pass > 0 {
                assert_eq!(stats.allocation_operations, 0, "{stats:?}");
            }
            let mut actual = vec![Complex64::default(); n];
            backend
                .try_copy_d2h(session.solution_device().unwrap(), &mut actual)
                .unwrap();
            direct
                .solve_with_incident_rhs(
                    &elements,
                    &mesh.nodes,
                    &physics,
                    beta,
                    backend,
                    1e-10,
                    extra,
                )
                .unwrap();
            let mut expected = vec![Complex64::default(); n];
            backend
                .try_copy_d2h(direct.solution().unwrap(), &mut expected)
                .unwrap();
            let error = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (*a - *b).norm_sqr())
                .sum::<f64>()
                .sqrt()
                / expected.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
            assert!(error < 1e-7, "LU parity {error}");
            let cpu = build_tbem_system_with_beta_parallel(&elements, &mesh.nodes, &physics, beta);
            let mut residual = 0.0;
            let mut rhs_norm = 0.0;
            for (i, &incident_value) in incident_host.iter().enumerate() {
                let rhs = cpu.rhs[i]
                    + if extra.is_some() {
                        incident_value
                    } else {
                        Complex64::default()
                    };
                let ax = (0..n)
                    .map(|j| cpu.matrix[[i, j]] * actual[j])
                    .sum::<Complex64>();
                residual += (ax - rhs).norm_sqr();
                rhs_norm += rhs.norm_sqr();
            }
            assert!((residual / rhs_norm).sqrt() < 1e-7);
            eprintln!(
                "TBEM GMRES pass={pass} iterations={} cpu_residual={} allocations={} h2d={} d2h={}",
                report.iterations,
                (residual / rhs_norm).sqrt(),
                stats.allocation_operations,
                stats.transfer.h2d_bytes,
                stats.transfer.d2h_bytes
            );
        }
        let beta = physics.burton_miller_beta();
        backend.inject_failure(CudaFailureInjectionPoint::Launch);
        assert!(
            session
                .solve_device(
                    &elements,
                    &mesh.nodes,
                    &physics,
                    beta,
                    None,
                    DeviceInitialGuess::Zero
                )
                .is_err()
        );
        assert!(session.solution_device().is_none());
        backend.recover_after_error().unwrap();
        assert!(
            session
                .solve_device(
                    &elements,
                    &mesh.nodes,
                    &physics,
                    beta,
                    None,
                    DeviceInitialGuess::PreviousSolution
                )
                .is_err()
        );
        session
            .solve_device(
                &elements,
                &mesh.nodes,
                &physics,
                beta,
                None,
                DeviceInitialGuess::Zero,
            )
            .unwrap();
        let mut changed = mesh.nodes.clone();
        changed[0][0] += 0.01;
        assert!(
            session
                .solve_device(
                    &elements,
                    &changed,
                    &physics,
                    beta,
                    None,
                    DeviceInitialGuess::Zero
                )
                .is_err()
        );
        assert!(session.solution_device().is_none());
        session
            .solve_device(
                &elements,
                &mesh.nodes,
                &physics,
                beta,
                None,
                DeviceInitialGuess::Zero,
            )
            .unwrap();
        let invalid_incident = vec![Complex64::new(f64::NAN, 0.0); n];
        backend
            .try_copy_h2d(&invalid_incident, incident.raw_mut())
            .unwrap();
        assert!(
            session
                .solve_device(
                    &elements,
                    &mesh.nodes,
                    &physics,
                    beta,
                    Some(incident.raw()),
                    DeviceInitialGuess::Zero
                )
                .is_err()
        );
        assert!(session.solution_device().is_none());
        session.controls.max_iterations = 1;
        assert!(
            session
                .solve_device(
                    &elements,
                    &mesh.nodes,
                    &physics,
                    beta,
                    None,
                    DeviceInitialGuess::Zero
                )
                .is_err()
        );
        assert!(session.solution_device().is_none());
        session.controls.max_iterations = 400;
        session
            .solve_device(
                &elements,
                &mesh.nodes,
                &physics,
                beta,
                None,
                DeviceInitialGuess::Zero,
            )
            .unwrap();
    }
}
