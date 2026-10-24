//! Retained native off-surface fields for constant-triangle device solutions.

// Rust guideline compliant 2026-02-21
use super::point_on_flat_triangle;
use crate::core::bem_solver::BemError;
use crate::core::types::{BoundaryCondition, Element, ElementType, PhysicsParams};
use num_complex::Complex64;
use sonium_compute::{
    Backend, BemFieldElementC64, BemFieldPointC64, Buffer, CudaBackend, CudaBuffer,
    DeviceBufferLike, MemoryPool,
};

fn failed(error: impl std::fmt::Display) -> BemError {
    BemError::SolverFailed(error.to_string())
}

/// Fixed geometry and boundary traces for resident exterior pressure evaluation.
///
/// Uses seven-point constant-Tri3 quadrature; near-surface accuracy requires
/// independent convergence checks. Velocity/impedance coefficients remain fixed;
/// frequency and incident descriptors may change. Reprepare for geometry, points
/// or boundary changes. Results are borrowed only after native validation.
pub struct CudaBemFieldWorkspace {
    elements: Buffer<BemFieldElementC64, CudaBackend>,
    points: Buffer<BemFieldPointC64, CudaBackend>,
    incident_points: Buffer<f64, CudaBackend>,
    admittance: Buffer<Complex64, CudaBackend>,
    prescribed_velocity: Buffer<Complex64, CudaBackend>,
    velocity: Buffer<Complex64, CudaBackend>,
    incident: Buffer<Complex64, CudaBackend>,
    output: Buffer<Complex64, CudaBackend>,
}

impl std::fmt::Debug for CudaBemFieldWorkspace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaBemFieldWorkspace")
            .field("elements", &self.elements.len())
            .field("points", &self.points.len())
            .finish_non_exhaustive()
    }
}

impl CudaBemFieldWorkspace {
    /// Prepare fixed surface traces and off-surface evaluation points.
    ///
    /// Empty point sets are valid. Geometry and coefficient uploads occur once.
    ///
    /// # Errors
    /// Rejects invalid geometry, DOF ordering, empty or nonfinite boundary data,
    /// unsupported traces, on-surface points and backend allocation/transfer errors.
    pub fn try_new(
        pool: &MemoryPool<CudaBackend>,
        boundary: &[Element],
        nodes: &[[f64; 3]],
        points: &[[f64; 3]],
    ) -> Result<Self, BemError> {
        if boundary.is_empty() || nodes.iter().flatten().any(|x| !x.is_finite()) {
            return Err(failed("field geometry must be nonempty and finite"));
        }
        for element in boundary {
            if element.element_type != ElementType::Tri3
                || element.property.is_evaluation()
                || element.connectivity.iter().any(|&i| i >= nodes.len())
            {
                return Err(failed("invalid surface triangle connectivity"));
            }
            match &element.boundary_condition {
                BoundaryCondition::Velocity(v)
                | BoundaryCondition::VelocityWithAdmittance { velocity: v, .. }
                    if !v.is_empty() && v.iter().all(|z| z.re.is_finite() && z.im.is_finite()) => {}
                _ => return Err(failed("field traces require nonempty finite velocity data")),
            }
        }
        if points.iter().flatten().any(|v| !v.is_finite()) {
            return Err(failed("field points must be finite"));
        }
        let zero = Complex64::new(0.0, 0.0);
        let mut elements = Vec::with_capacity(boundary.len());
        let mut admittance = Vec::with_capacity(elements.capacity());
        let mut prescribed_velocity = Vec::with_capacity(elements.capacity());
        for (index, element) in boundary.iter().enumerate() {
            if element.dof_addresses.as_slice() != [index] {
                return Err(failed(
                    "field evaluation requires consecutive constant-panel DOFs",
                ));
            }
            let [a, b, c] = element.connectivity.as_slice() else {
                return Err(failed("field evaluation requires three-node triangles"));
            };
            let vertices = [nodes[*a], nodes[*b], nodes[*c]];
            if points.iter().any(|&p| point_on_flat_triangle(p, vertices)) {
                return Err(failed("CUDA field evaluation requires off-surface points"));
            }
            let u = std::array::from_fn::<_, 3, _>(|i| vertices[1][i] - vertices[0][i]);
            let v = std::array::from_fn::<_, 3, _>(|i| vertices[2][i] - vertices[0][i]);
            let cross = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            // Match the existing checked CUDA field operator's geometry threshold.
            if cross.iter().map(|x| x * x).sum::<f64>().sqrt() < 1e-15 {
                return Err(failed("degenerate field triangle"));
            }
            let (values, y) = match &element.boundary_condition {
                BoundaryCondition::Velocity(values) => (values, zero),
                BoundaryCondition::VelocityWithAdmittance {
                    velocity,
                    admittance,
                } => (velocity, *admittance),
                _ => {
                    return Err(failed(
                        "field evaluation requires velocity or impedance traces",
                    ));
                }
            };
            let prescribed = values.iter().copied().sum::<Complex64>() / values.len().max(1) as f64;
            if [y, prescribed]
                .iter()
                .any(|z| !z.re.is_finite() || !z.im.is_finite())
            {
                return Err(failed("nonfinite field boundary coefficients"));
            }
            admittance.push(y);
            prescribed_velocity.push(prescribed);
            elements.push(BemFieldElementC64::from_triangle(
                vertices[0],
                vertices[1],
                vertices[2],
                zero,
                zero,
            ));
        }
        let device_points: Vec<_> = points.iter().copied().map(BemFieldPointC64::new).collect();
        let incident_points: Vec<_> = points
            .iter()
            .flat_map(|p| p.iter().copied().chain([0.0; 3]))
            .collect();
        let backend = pool.backend();
        let mut workspace = Self {
            elements: pool.try_alloc(elements.len()).map_err(failed)?,
            points: pool.try_alloc(points.len()).map_err(failed)?,
            incident_points: pool.try_alloc(incident_points.len()).map_err(failed)?,
            admittance: pool.try_alloc(admittance.len()).map_err(failed)?,
            prescribed_velocity: pool.try_alloc(prescribed_velocity.len()).map_err(failed)?,
            velocity: pool.try_alloc(admittance.len()).map_err(failed)?,
            incident: pool.try_alloc(points.len()).map_err(failed)?,
            output: pool.try_alloc(points.len()).map_err(failed)?,
        };
        backend
            .try_copy_h2d(&elements, workspace.elements.raw_mut())
            .map_err(failed)?;
        backend
            .try_copy_h2d(&device_points, workspace.points.raw_mut())
            .map_err(failed)?;
        backend
            .try_copy_h2d(&incident_points, workspace.incident_points.raw_mut())
            .map_err(failed)?;
        backend
            .try_copy_h2d(&admittance, workspace.admittance.raw_mut())
            .map_err(failed)?;
        backend
            .try_copy_h2d(
                &prescribed_velocity,
                workspace.prescribed_velocity.raw_mut(),
            )
            .map_err(failed)?;
        Ok(workspace)
    }

    /// Evaluate total pressure from resident traces and incident descriptors.
    ///
    /// The caller must provide a certified surface pressure in prepared DOF order.
    /// Sources use the native six-value incident descriptor layout. Empty point
    /// sets return an empty buffer. No bulk host transfer or allocation is needed.
    ///
    /// # Errors
    /// Rejects invalid physics, interior evaluation, mismatched pressure,
    /// nonfinite results, missing native kernels, invalid descriptors and backend failures.
    pub fn try_evaluate(
        &mut self,
        backend: &CudaBackend,
        pressure: &CudaBuffer<Complex64>,
        physics: &PhysicsParams,
        sources: &CudaBuffer<f64>,
        source_count: usize,
    ) -> Result<&CudaBuffer<Complex64>, BemError> {
        physics.validate().map_err(failed)?;
        if physics.tau != 1.0 || pressure.len() != self.elements.len() {
            return Err(failed(
                "field evaluation requires exterior physics and matching pressure DOFs",
            ));
        }
        if self.output.is_empty() {
            return Ok(self.output.raw());
        }
        if !backend
            .try_diagonal_scale_native(
                self.admittance.raw(),
                pressure,
                self.velocity.raw_mut(),
                1.0,
            )
            .map_err(failed)?
        {
            return Err(failed("native field velocity scaling is unavailable"));
        }
        backend
            .try_axpy(
                Complex64::new(1.0, 0.0),
                self.prescribed_velocity.raw(),
                self.velocity.raw_mut(),
            )
            .map_err(failed)?;
        Backend::try_bem_field_c64_device(
            backend,
            self.elements.raw(),
            pressure,
            self.velocity.raw(),
            self.points.raw(),
            self.output.raw_mut(),
            physics.wave_number,
            physics.harmonic_factor,
            physics.pressure_factor,
        )
        .map_err(failed)?;
        // Incident RHS evaluates -(gamma*p + beta_tau*dp/dn). These parameters
        // request pressure itself at field points, with zero dummy normals.
        backend
            .try_bem_incident_rhs_c64_device(
                self.incident_points.raw(),
                sources,
                source_count,
                physics.wave_number,
                physics.harmonic_factor,
                -1.0,
                Complex64::new(0.0, 0.0),
                self.incident.raw_mut(),
            )
            .map_err(failed)?;
        backend
            .try_axpy(
                Complex64::new(1.0, 0.0),
                self.incident.raw(),
                self.output.raw_mut(),
            )
            .map_err(failed)?;
        if backend
            .try_all_finite_native(self.output.raw())
            .map_err(failed)?
            != Some(true)
        {
            return Err(failed("native total-field validation failed"));
        }
        Ok(self.output.raw())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::bem_solver::pack_incident_sources;
    use crate::core::incident::IncidentField;
    use crate::core::mesh::generators::generate_icosphere_mesh;
    use crate::core::postprocess::pressure::compute_total_field;
    use crate::core::solver::frequency_sweep::{
        FrequencySweepConfig, frequency_sweep_cuda_tbem_jacobi_device,
    };
    use crate::core::types::BasisOrder;
    use sonium_compute::{CudaFailureInjectionPoint, MemoryPoolConfig};

    #[test]
    fn native_sweep_resident_fields() {
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
        for (i, e) in elements.iter_mut().enumerate() {
            e.dof_addresses = vec![i];
            e.boundary_condition = BoundaryCondition::VelocityWithAdmittance {
                velocity: vec![Complex64::new(0.01, 0.02)],
                admittance: Complex64::new(0.001, 0.002),
            };
        }
        let points = [[2.0, 0.5, 0.2], [-2.0, 0.4, 0.1], [0.3, 2.2, 0.0]];
        let incident = IncidentField::PlaneWave {
            direction: [1.0, 0.0, 0.0],
            amplitude: Complex64::new(0.8, 0.2),
        };
        let packed = pack_incident_sources(&incident);
        let mut sources = pool.try_alloc::<f64>(packed.len()).unwrap();
        backend.try_copy_h2d(&packed, sources.raw_mut()).unwrap();
        let config = FrequencySweepConfig {
            frequencies: vec![10.0, 12.0, 40.0, 42.0],
            tolerance: 1e-10,
            max_iterations: 400,
            gmres_restart: 20,
            ..Default::default()
        };
        let physics = PhysicsParams::new(10.0, 343.0, 1.21, false);
        let mut field =
            CudaBemFieldWorkspace::try_new(&pool, &elements, &mesh.nodes, &points).unwrap();
        let mut count = 0;
        let result = frequency_sweep_cuda_tbem_jacobi_device(
            &elements,
            &mesh.nodes,
            &incident,
            &config,
            &physics,
            None,
            &pool,
            |p, boundary, pressure| {
                let before = backend.begin_runtime_stats();
                let output = field.try_evaluate(backend, pressure, p, sources.raw(), 1)?;
                let stats = backend.runtime_stats_since(before);
                assert_eq!(stats.allocation_operations, 0);
                assert_eq!(stats.transfer.h2d_bytes, 0);
                assert!(stats.transfer.d2h_bytes < elements.len() * 16, "{stats:?}");
                let mut actual = vec![Complex64::default(); points.len()];
                backend.try_copy_d2h(output, &mut actual).unwrap();
                let mut surface = vec![Complex64::default(); elements.len()];
                backend.try_copy_d2h(pressure, &mut surface).unwrap();
                let expected = compute_total_field(
                    &points,
                    boundary,
                    &mesh.nodes,
                    &surface,
                    None,
                    &incident,
                    p,
                    BasisOrder::Constant,
                );
                let error = actual
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (*a - b.p_total).norm_sqr())
                    .sum::<f64>();
                let norm = expected.iter().map(|x| x.p_total.norm_sqr()).sum::<f64>();
                assert!((error / norm).sqrt() < 1e-7);
                backend.inject_failure(CudaFailureInjectionPoint::Launch);
                assert!(
                    field
                        .try_evaluate(backend, pressure, p, sources.raw(), 1)
                        .is_err()
                );
                backend.recover_after_error().unwrap();
                field.try_evaluate(backend, pressure, p, sources.raw(), 1)?;
                count += 1;
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(count, config.frequencies.len());
        assert!(result.points.iter().all(|p| p.solution.is_empty()));
        let before = backend.begin_runtime_stats();
        let mut invalid = elements.clone();
        invalid[0].connectivity[0] = mesh.nodes.len();
        assert!(CudaBemFieldWorkspace::try_new(&pool, &invalid, &mesh.nodes, &points).is_err());
        assert!(
            CudaBemFieldWorkspace::try_new(&pool, &elements, &mesh.nodes, &[elements[0].center])
                .is_err()
        );
        assert_eq!(backend.runtime_stats_since(before).allocation_operations, 0);
    }
}
