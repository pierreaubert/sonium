//! Native dense frequency sweeps with retained assembly and Krylov state.

// Rust guideline compliant 2026-02-21
use super::*;
use crate::core::assembly::tbem::CudaTbemGmresSession;
use crate::core::bem_solver::pack_incident_sources;
use sonium_compute::{Backend, CudaBackend, CudaBuffer, MemoryPool};
use sonium_solvers::iterative::{BackendKrylovConfig, DeviceInitialGuess};

/// Certified frequency result with explicit host export and phase timings.
#[derive(Debug)]
pub struct CudaFrequencySweepPoint {
    /// Frequency in hertz.
    pub frequency: f64,
    /// Certified solution exported in DOF order; empty for device-output sweeps.
    pub solution: Vec<Complex64>,
    /// GMRES iterations at this frequency.
    pub iterations: usize,
    /// Original-system residual reported by the retained solver.
    pub relative_residual: f64,
    /// Cumulative assembly replacements, excluding initial preparation.
    pub assembly_rebuilds: usize,
    /// Includes incident evaluation, numeric assembly, preconditioner refresh and solve.
    pub update_solve_seconds: f64,
    /// Wall time to allocate and download the host solution, including synchronization.
    pub download_seconds: f64,
    /// Device-output callback time, including completion synchronization.
    pub output_seconds: f64,
    /// Device allocations during update and solve, excluding explicit export.
    pub device_allocations: u64,
}

/// Results from a native TBEM sweep with point-Jacobi preconditioning.
#[derive(Debug)]
pub struct CudaFrequencySweepResult {
    /// Frequency results in the requested order.
    pub points: Vec<CudaFrequencySweepPoint>,
    /// Validation, first assembly, workspace reservation and descriptor uploads.
    pub preparation_seconds: f64,
    /// Complete wall time including preparation and all solution exports.
    pub total_seconds: f64,
}

fn failed(error: impl std::fmt::Display) -> BemError {
    BemError::SolverFailed(error.to_string())
}

/// Sweep prepared constant-Tri3 systems using persistent native GMRES and Jacobi.
///
/// Uses the supplied pool across all frequencies. Geometry and ordered contiguous
/// DOFs stay fixed; frequency and optional surface admittances change. Preserves
/// template density, sound speed, harmonic convention and interior/exterior sign,
/// and uses `config.beta_scale`. Quadrature changes rebuild assembly only.
/// Source descriptors upload once. Each certified solution is downloaded once
/// for existing field/export APIs; field evaluation is outside this function.
/// An empty grid returns an empty result without device allocation.
///
/// Point Jacobi is an explicit choice, not a replacement for every CPU
/// preconditioner. Legacy assembly allocations remain outside the pool quota.
///
/// # Errors
/// Rejects GCRO-DR requests, invalid controls/admittances/physics, unordered DOFs,
/// unsupported geometry or systems with 1000 or more DOFs. Propagates native
/// assembly, allocation, nonconvergence and transfer errors; no CPU fallback.
pub fn frequency_sweep_cuda_tbem_jacobi(
    elements: &[Element],
    nodes: &[[f64; 3]],
    incident: &IncidentField,
    config: &FrequencySweepConfig,
    physics_template: &PhysicsParams,
    surface_admittances: Option<&[Complex64]>,
    pool: &MemoryPool<CudaBackend>,
) -> Result<CudaFrequencySweepResult, BemError> {
    sweep_with_output(
        elements,
        nodes,
        incident,
        config,
        physics_template,
        surface_admittances,
        pool,
        true,
        |_, _, _| Ok(()),
    )
}

/// Consume certified frequency solutions on CUDA before their storage is reused.
///
/// The callback receives current physics, boundary metadata (including admittance
/// overrides), and a borrowed device solution in prepared DOF order. It may run
/// field evaluation or reductions on the supplied pool's backend. Copy to owned
/// device storage inside the callback if the result must outlive this frequency.
/// The sweep synchronizes callback work before continuing. No surface solution
/// is downloaded by this entrypoint; returned `solution` vectors are empty and
/// `download_seconds` is zero. Callback transfers and allocations are its own
/// responsibility, separate from the reported solve allocation count.
///
/// # Errors
/// Has the same admission/solve failures as [`frequency_sweep_cuda_tbem_jacobi`].
/// Callback errors stop the sweep immediately and are propagated unchanged.
pub fn frequency_sweep_cuda_tbem_jacobi_device(
    elements: &[Element],
    nodes: &[[f64; 3]],
    incident: &IncidentField,
    config: &FrequencySweepConfig,
    physics_template: &PhysicsParams,
    surface_admittances: Option<&[Complex64]>,
    pool: &MemoryPool<CudaBackend>,
    output: impl FnMut(&PhysicsParams, &[Element], &CudaBuffer<Complex64>) -> Result<(), BemError>,
) -> Result<CudaFrequencySweepResult, BemError> {
    sweep_with_output(
        elements,
        nodes,
        incident,
        config,
        physics_template,
        surface_admittances,
        pool,
        false,
        output,
    )
}

fn sweep_with_output(
    elements: &[Element],
    nodes: &[[f64; 3]],
    incident: &IncidentField,
    config: &FrequencySweepConfig,
    physics_template: &PhysicsParams,
    surface_admittances: Option<&[Complex64]>,
    pool: &MemoryPool<CudaBackend>,
    download: bool,
    mut output: impl FnMut(&PhysicsParams, &[Element], &CudaBuffer<Complex64>) -> Result<(), BemError>,
) -> Result<CudaFrequencySweepResult, BemError> {
    let start = Instant::now();
    config.validate()?;
    physics_template
        .validate()
        .map_err(|e| BemError::InvalidParameters(e.to_string()))?;
    if config.recycle_subspace_size != 0 {
        return Err(BemError::NotImplemented(
            "CUDA TBEM sweep does not implement GCRO-DR recycling".into(),
        ));
    }
    if let Some(values) = surface_admittances {
        if values.len() != config.frequencies.len()
            || values
                .iter()
                .any(|v| !v.re.is_finite() || !v.im.is_finite())
        {
            return Err(BemError::InvalidParameters(
                "invalid frequency-sweep admittances".into(),
            ));
        }
        if let Some(&first) = values.first() {
            elements_with_surface_admittance(elements, first)?;
        }
    }
    let mut result = CudaFrequencySweepResult {
        points: Vec::with_capacity(config.frequencies.len()),
        preparation_seconds: 0.0,
        total_seconds: 0.0,
    };
    if config.frequencies.is_empty() {
        result.total_seconds = start.elapsed().as_secs_f64();
        return Ok(result);
    }
    if elements
        .iter()
        .enumerate()
        .any(|(i, e)| e.property.is_evaluation() || e.dof_addresses != [i])
    {
        return Err(BemError::InvalidParameters(
            "CUDA sweep requires ordered contiguous surface DOFs".into(),
        ));
    }
    let physics = config
        .frequencies
        .iter()
        .map(|&frequency| {
            let mut p = PhysicsParams::try_new(
                frequency,
                physics_template.speed_of_sound,
                physics_template.density,
                physics_template.tau < 0.0,
            )
            .map_err(|e| BemError::InvalidParameters(e.to_string()))?;
            p.harmonic_factor = physics_template.harmonic_factor;
            p.pressure_factor = p.density * p.omega * p.harmonic_factor;
            p.validate()
                .map_err(|e| BemError::InvalidParameters(e.to_string()))?;
            if default_sweep_sign(elements, &p) != 1.0 {
                return Err(BemError::NotImplemented(
                    "CUDA sweep requires unit formulation sign".into(),
                ));
            }
            Ok(p)
        })
        .collect::<Result<Vec<_>, BemError>>()?;
    let mut sources = pack_incident_sources(incident);
    if sources.iter().any(|v| !v.is_finite()) {
        return Err(BemError::InvalidParameters(
            "incident descriptors must be finite".into(),
        ));
    }
    let source_count = sources.len() / 6;
    sources.resize(sources.len().max(6), 0.0);
    let first_elements = surface_admittances
        .map(|a| elements_with_surface_admittance(elements, a[0]))
        .transpose()?;
    let mut session = CudaTbemGmresSession::try_new(
        pool,
        first_elements.as_deref().unwrap_or(elements),
        nodes,
        &physics[0],
        physics[0].burton_miller_beta_scaled(config.beta_scale),
        BackendKrylovConfig {
            restart: config.gmres_restart,
            max_iterations: config.max_iterations,
            tolerance: config.tolerance,
            require_native: true,
            ..Default::default()
        },
    )?;
    let backend = pool.backend();
    let geometry: Vec<_> = elements
        .iter()
        .flat_map(|e| e.center.into_iter().chain(e.normal))
        .collect();
    let mut points = pool.try_alloc::<f64>(geometry.len()).map_err(failed)?;
    let mut descriptors = pool.try_alloc::<f64>(sources.len()).map_err(failed)?;
    let mut rhs = pool
        .try_alloc::<Complex64>(elements.len())
        .map_err(failed)?;
    backend
        .try_copy_h2d(&geometry, points.raw_mut())
        .map_err(failed)?;
    backend
        .try_copy_h2d(&sources, descriptors.raw_mut())
        .map_err(failed)?;
    backend.try_synchronize().map_err(failed)?;
    result.preparation_seconds = start.elapsed().as_secs_f64();
    for (index, p) in physics.iter().enumerate() {
        let update_start = Instant::now();
        let before = backend.begin_runtime_stats();
        let updated = surface_admittances
            .map(|a| elements_with_surface_admittance(elements, a[index]))
            .transpose()?;
        let beta = p.burton_miller_beta_scaled(config.beta_scale);
        backend
            .try_bem_incident_rhs_c64_device(
                points.raw(),
                descriptors.raw(),
                source_count,
                p.wave_number,
                p.harmonic_factor,
                p.gamma(),
                beta * p.tau,
                rhs.raw_mut(),
            )
            .map_err(failed)?;
        let status = session.solve_device(
            updated.as_deref().unwrap_or(elements),
            nodes,
            p,
            beta,
            Some(rhs.raw()),
            if index > 0 && config.recycle_solution {
                DeviceInitialGuess::PreviousSolution
            } else {
                DeviceInitialGuess::Zero
            },
        )?;
        backend.try_synchronize().map_err(failed)?;
        let update_solve_seconds = update_start.elapsed().as_secs_f64();
        let stats = backend.runtime_stats_since(before);
        let device = session
            .solution_device()
            .ok_or_else(|| failed("missing certified sweep solution"))?;
        let output_seconds = if download {
            0.0
        } else {
            let output_start = Instant::now();
            output(p, updated.as_deref().unwrap_or(elements), device)?;
            backend.try_synchronize().map_err(failed)?;
            output_start.elapsed().as_secs_f64()
        };
        let mut solution = Vec::new();
        let mut download_seconds = 0.0;
        if download {
            let download_start = Instant::now();
            solution.resize(elements.len(), Complex64::default());
            backend
                .try_copy_d2h(device, &mut solution)
                .map_err(failed)?;
            backend.try_synchronize().map_err(failed)?;
            download_seconds = download_start.elapsed().as_secs_f64();
        }
        result.points.push(CudaFrequencySweepPoint {
            frequency: config.frequencies[index],
            solution,
            iterations: status.iterations,
            relative_residual: status.true_residual,
            assembly_rebuilds: session.assembly_rebuilds(),
            update_solve_seconds,
            download_seconds,
            output_seconds,
            device_allocations: stats.allocation_operations as u64,
        });
    }
    result.total_seconds = start.elapsed().as_secs_f64();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::mesh::generators::generate_icosphere_mesh;
    use sonium_compute::MemoryPoolConfig;

    #[test]
    fn native_cuda_frequency_sweep_parity() {
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
        let mesh = generate_icosphere_mesh(1.0, 0);
        let mut elements = mesh.elements.clone();
        for (i, e) in elements.iter_mut().enumerate() {
            e.dof_addresses = vec![i];
            e.boundary_condition = if i == 0 {
                BoundaryCondition::Pressure(vec![Complex64::new(0.2, -0.1)])
            } else {
                BoundaryCondition::VelocityWithAdmittance {
                    velocity: vec![Complex64::new(0.01, 0.02)],
                    admittance: Complex64::new(0.001, 0.002),
                }
            };
        }
        let config = FrequencySweepConfig {
            frequencies: [0.1, 0.11, 0.5, 0.51, 1.2, 1.21, 2.2]
                .into_iter()
                .map(|ka| ka * 343.0 / (2.0 * std::f64::consts::PI * elements[0].area.sqrt()))
                .collect(),
            tolerance: 1e-10,
            gmres_restart: 20,
            max_iterations: 400,
            ..Default::default()
        };
        let admittances: Vec<_> = (0..config.frequencies.len())
            .map(|i| Complex64::new(0.001 + i as f64 * 0.0001, 0.002))
            .collect();
        for interior in [false, true] {
            for harmonic in [-1.0, 1.0] {
                let mut physics = PhysicsParams::new(config.frequencies[0], 343.0, 1.21, interior);
                physics.harmonic_factor = harmonic;
                physics.pressure_factor = physics.density * physics.omega * harmonic;
                let incident = if interior {
                    IncidentField::PointSource {
                        position: [0.1, 0.2, 0.0],
                        strength: Complex64::new(0.8, -0.2),
                    }
                } else {
                    IncidentField::PlaneWave {
                        direction: [1.0, 0.0, 0.0],
                        amplitude: Complex64::new(0.8, -0.2),
                    }
                };
                let cpu = frequency_sweep_with_physics_and_admittances(
                    &elements,
                    &mesh.nodes,
                    &incident,
                    &config,
                    AssemblyMethod::Tbem,
                    &physics,
                    Some(&admittances),
                )
                .unwrap();
                let gpu = frequency_sweep_cuda_tbem_jacobi(
                    &elements,
                    &mesh.nodes,
                    &incident,
                    &config,
                    &physics,
                    Some(&admittances),
                    &pool,
                )
                .unwrap();
                assert_eq!(gpu.points.len(), config.frequencies.len());
                let mut saved = (0..config.frequencies.len())
                    .map(|_| pool.try_alloc::<Complex64>(elements.len()).unwrap())
                    .collect::<Vec<_>>();
                let mut seen = 0;
                let streamed = frequency_sweep_cuda_tbem_jacobi_device(
                    &elements,
                    &mesh.nodes,
                    &incident,
                    &config,
                    &physics,
                    Some(&admittances),
                    &pool,
                    |p, boundary, solution| {
                        assert_eq!(p.frequency, config.frequencies[seen]);
                        assert_eq!(
                            boundary[1].boundary_condition.admittance(),
                            Some(admittances[seen])
                        );
                        let before = pool.backend().begin_runtime_stats();
                        pool.backend()
                            .try_copy_d2d(solution, saved[seen].raw_mut())
                            .map_err(failed)?;
                        let stats = pool.backend().runtime_stats_since(before);
                        assert_eq!(stats.transfer.d2h_bytes, 0);
                        assert_eq!(stats.transfer.h2d_bytes, 0);
                        assert_eq!(stats.allocation_operations, 0);
                        seen += 1;
                        Ok(())
                    },
                )
                .unwrap();
                assert_eq!(seen, config.frequencies.len());
                for (i, point) in streamed.points.iter().enumerate() {
                    assert!(point.solution.is_empty());
                    assert_eq!(point.download_seconds, 0.0);
                    let mut exported = vec![Complex64::default(); elements.len()];
                    pool.backend()
                        .try_copy_d2h(saved[i].raw(), &mut exported)
                        .unwrap();
                    let error = exported
                        .iter()
                        .zip(&gpu.points[i].solution)
                        .map(|(a, b)| (*a - *b).norm_sqr())
                        .sum::<f64>();
                    let norm = gpu.points[i]
                        .solution
                        .iter()
                        .map(|x| x.norm_sqr())
                        .sum::<f64>();
                    assert!((error / norm).sqrt() < 1e-10);
                }
                for (index, (point, expected)) in gpu.points.iter().zip(&cpu.solutions).enumerate()
                {
                    let difference = point
                        .solution
                        .iter()
                        .zip(expected)
                        .map(|(a, b)| (*a - *b).norm_sqr())
                        .sum::<f64>();
                    let norm = expected.iter().map(|x| x.norm_sqr()).sum::<f64>();
                    assert!(
                        (difference / norm).sqrt() < 1e-7,
                        "index={index} interior={interior} harmonic={harmonic}"
                    );
                    assert_eq!(point.assembly_rebuilds, index / 2);
                    if index % 2 == 1 {
                        assert_eq!(point.device_allocations, 0);
                    }
                }
            }
        }
        let physics = PhysicsParams::new(config.frequencies[0], 343.0, 1.21, false);
        let incident = IncidentField::PlaneWave {
            direction: [1.0, 0.0, 0.0],
            amplitude: Complex64::new(1.0, 0.0),
        };
        let mut unsupported = config;
        let mut callbacks = 0;
        let failure = frequency_sweep_cuda_tbem_jacobi_device(
            &elements,
            &mesh.nodes,
            &incident,
            &unsupported,
            &physics,
            None,
            &pool,
            |_, _, _| {
                callbacks += 1;
                Err(BemError::InvalidParameters("output rejected".into()))
            },
        );
        assert!(
            matches!(failure, Err(BemError::InvalidParameters(message)) if message == "output rejected")
        );
        assert_eq!(callbacks, 1);
        unsupported.recycle_subspace_size = 2;
        let before = pool.backend().begin_runtime_stats();
        assert!(
            frequency_sweep_cuda_tbem_jacobi(
                &elements,
                &mesh.nodes,
                &incident,
                &unsupported,
                &physics,
                None,
                &pool
            )
            .is_err()
        );
        assert_eq!(
            pool.backend()
                .runtime_stats_since(before)
                .allocation_operations,
            0
        );
        unsupported.recycle_subspace_size = 0;
        unsupported.frequencies.clear();
        let before = pool.backend().begin_runtime_stats();
        assert!(
            frequency_sweep_cuda_tbem_jacobi(
                &elements,
                &mesh.nodes,
                &incident,
                &unsupported,
                &physics,
                None,
                &pool
            )
            .unwrap()
            .points
            .is_empty()
        );
        assert_eq!(
            pool.backend()
                .runtime_stats_since(before)
                .allocation_operations,
            0
        );
    }
}
