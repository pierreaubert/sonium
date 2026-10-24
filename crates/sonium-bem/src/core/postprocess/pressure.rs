//! Field pressure evaluation at exterior points
//!
//! Computes the acoustic pressure field at evaluation points using
//! the BEM representation formula:
//!
//! p(x) = p_inc(x) + ∫_Γ [p(y) ∂G/∂n_y - (∂p/∂n)(y) G(x,y)] dS_y
//!
//! For a rigid scatterer (Neumann BC), the normal velocity is zero,
//! so the formula simplifies to:
//!
//! p(x) = p_inc(x) + ∫_Γ p(y) ∂G/∂n_y dS_y

use num_complex::Complex64;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::direct::lu_solve;
use std::f64::consts::PI;

// Note: Using inline Green's function computation for performance
use crate::core::incident::IncidentField;
use crate::core::integration::gauss::{quad_quadrature, triangle_quadrature};
use crate::core::mesh::element::{basis_values, quadrature_geometry};
use crate::core::types::{BasisOrder, BoundaryCondition, Element, ElementType, PhysicsParams};
#[cfg(all(feature = "cuda", test))]
use sonium_compute::MemoryPoolConfig;
#[cfg(feature = "cuda")]
use sonium_compute::{CudaBackend, CudaBemFieldOperatorC64, MemoryPool};

use crate::core::parallel::{BEM_QUADRATURE_PAIRS_PER_THREAD, gated_map, gated_map_indexed};

#[cfg(feature = "cuda")]
#[path = "pressure_cuda_device.rs"]
mod cuda_device;
#[cfg(feature = "cuda")]
#[doc(inline)]
pub use cuda_device::CudaBemFieldWorkspace;

/// Field evaluation result at a single point
#[derive(Debug, Clone)]
pub struct FieldPoint {
    /// Position coordinates [x, y, z]
    pub position: [f64; 3],
    /// Incident pressure
    pub p_incident: Complex64,
    /// Scattered pressure
    pub p_scattered: Complex64,
    /// Total pressure (incident + scattered)
    pub p_total: Complex64,
}

/// Typed admission failures for exterior field evaluation.
///
/// The historical field functions retain their infallible signatures for
/// trusted solver internals. External callers should use the checked
/// variants so malformed connectivity, non-finite samples, and mismatched
/// basis buffers are rejected before a parallel kernel can index them.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldEvaluationError {
    /// No physical boundary element was available for the representation.
    EmptyBoundary,
    /// An element has an unsupported or malformed connectivity list.
    InvalidConnectivity {
        /// Element position in the supplied array.
        element: usize,
        /// Required node count for the element type.
        expected: usize,
        /// Supplied node count.
        actual: usize,
    },
    /// An element references a node outside the coordinate array.
    InvalidNodeIndex {
        /// Element position in the supplied array.
        element: usize,
        /// Invalid node index.
        node: usize,
    },
    /// A node coordinate is not finite.
    NonFiniteNode {
        /// Node position in the coordinate array.
        node: usize,
    },
    /// An evaluation point is not finite.
    NonFiniteEvaluationPoint {
        /// Evaluation point position.
        point: usize,
    },
    /// The pressure buffer does not match the requested basis layout.
    SurfacePressureCount {
        /// Required buffer length.
        expected: usize,
        /// Supplied buffer length.
        actual: usize,
    },
    /// The optional surface-velocity buffer has the wrong basis-layout length.
    SurfaceVelocityCount {
        /// Required buffer length.
        expected: usize,
        /// Supplied buffer length.
        actual: usize,
    },
    /// A linear element does not provide the expected number of local DOFs.
    InvalidDofLayout {
        /// Element position in the supplied array.
        element: usize,
        /// Required number of local DOF addresses.
        expected: usize,
        /// Supplied number of local DOF addresses.
        actual: usize,
    },
    /// A linear element references a sentinel/out-of-range DOF address.
    InvalidDofIndex {
        /// Element position in the supplied array.
        element: usize,
        /// Local address position.
        local: usize,
        /// Invalid global DOF index.
        dof: usize,
    },
    /// A pressure or velocity sample is not finite.
    NonFiniteSurfaceSample {
        /// Sample position.
        index: usize,
    },
    /// Physical parameters failed their checked constructor contract.
    InvalidPhysics {
        /// Typed validation explanation.
        reason: String,
    },
    /// The computed field contains a non-finite value.
    NonFiniteResult {
        /// Evaluation point position.
        point: usize,
    },
    /// The dual-to-primal surface mass solve failed.
    DualBasisSolve {
        /// Typed factorization or non-finite-output explanation.
        reason: String,
    },
    /// A field kernel returned a result count different from the requested
    /// evaluation-point count.
    ResultCount {
        /// Number of requested evaluation points.
        expected: usize,
        /// Number of values returned by the kernel.
        actual: usize,
    },
    /// The requested CUDA field operation is outside the explicitly
    /// supported accelerator cell or failed in the CUDA backend.
    CudaFailure {
        /// Typed backend/admission explanation.
        reason: String,
    },
}

impl std::fmt::Display for FieldEvaluationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyBoundary => f.write_str("field evaluation requires a boundary element"),
            Self::InvalidConnectivity {
                element,
                expected,
                actual,
            } => write!(
                f,
                "element {element} has {actual} connectivity nodes; expected {expected}"
            ),
            Self::InvalidNodeIndex { element, node } => {
                write!(f, "element {element} references invalid node {node}")
            }
            Self::NonFiniteNode { node } => write!(f, "node {node} has non-finite coordinates"),
            Self::NonFiniteEvaluationPoint { point } => {
                write!(f, "evaluation point {point} is non-finite")
            }
            Self::SurfacePressureCount { expected, actual } => {
                write!(
                    f,
                    "surface pressure has {actual} values; expected {expected}"
                )
            }
            Self::SurfaceVelocityCount { expected, actual } => {
                write!(
                    f,
                    "surface velocity has {actual} values; expected {expected}"
                )
            }
            Self::InvalidDofLayout {
                element,
                expected,
                actual,
            } => write!(
                f,
                "element {element} has {actual} DOF addresses; expected {expected}"
            ),
            Self::InvalidDofIndex {
                element,
                local,
                dof,
            } => write!(
                f,
                "element {element} local DOF {local} has invalid index {dof}"
            ),
            Self::NonFiniteSurfaceSample { index } => {
                write!(f, "surface sample {index} is non-finite")
            }
            Self::InvalidPhysics { reason } => write!(f, "invalid field physics: {reason}"),
            Self::NonFiniteResult { point } => {
                write!(f, "field result at point {point} is non-finite")
            }
            Self::DualBasisSolve { reason } => {
                write!(f, "dual-basis surface mass solve failed: {reason}")
            }
            Self::ResultCount { expected, actual } => {
                write!(f, "field returned {actual} results; expected {expected}")
            }
            Self::CudaFailure { reason } => write!(f, "CUDA field evaluation failed: {reason}"),
        }
    }
}

impl std::error::Error for FieldEvaluationError {}

impl FieldPoint {
    /// Create new field point
    pub fn new(position: [f64; 3], p_incident: Complex64, p_scattered: Complex64) -> Self {
        Self {
            position,
            p_incident,
            p_scattered,
            p_total: p_incident + p_scattered,
        }
    }

    /// Get pressure magnitude in dB SPL (re: 20 μPa).
    ///
    /// The BEM solves for complex pressure amplitudes (peak), and SPL is
    /// referenced to RMS pressure: `SPL = 20 log10(|p|/sqrt(2)/p_ref)`.
    pub fn spl_db(&self) -> f64 {
        sonium_xem::pressure_to_spl(self.p_total)
    }

    /// Get pressure magnitude (absolute value)
    pub fn magnitude(&self) -> f64 {
        self.p_total.norm()
    }

    /// Get pressure phase in radians
    pub fn phase(&self) -> f64 {
        self.p_total.arg()
    }
}

/// Checked scattered-field evaluation.
///
/// This validates geometry, basis buffer lengths, finite samples, and
/// physical parameters before entering the allocation-free parallel kernel.
/// The returned values are the scattered pressure only; use
/// [`compute_total_field_checked`] when an incident field is available.
pub fn compute_scattered_field_checked(
    eval_points: &[[f64; 3]],
    elements: &[Element],
    nodes: &[[f64; 3]],
    surface_pressure: &[Complex64],
    surface_velocity: Option<&[Complex64]>,
    physics: &PhysicsParams,
    basis_order: BasisOrder,
) -> Result<Vec<Complex64>, FieldEvaluationError> {
    validate_field_inputs(
        eval_points,
        elements,
        nodes,
        surface_pressure,
        surface_velocity,
        physics,
        basis_order,
    )?;
    let values = compute_scattered_field(
        eval_points,
        elements,
        nodes,
        surface_pressure,
        surface_velocity,
        physics,
        basis_order,
    );
    validate_result_count(eval_points.len(), values.len())?;
    validate_finite_results(&values)?;
    Ok(values)
}

/// Checked CUDA field evaluation for the currently supported accelerator
/// cell: constant-basis, flat Tri3 boundary elements in Complex64.
///
/// The element geometry and constant Cauchy traces are uploaded once into a
/// [`CudaBemFieldOperatorC64`]. The returned host vector is the explicit
/// publication boundary; no CPU field integral is used as a fallback. Quad4,
/// higher-order traces, and on-surface singular evaluation should be routed
/// to the checked CPU reference through [`compute_scattered_field_checked`].
#[cfg(feature = "cuda")]
pub fn compute_scattered_field_cuda_checked(
    pool: &MemoryPool<CudaBackend>,
    eval_points: &[[f64; 3]],
    elements: &[Element],
    nodes: &[[f64; 3]],
    surface_pressure: &[Complex64],
    surface_velocity: Option<&[Complex64]>,
    physics: &PhysicsParams,
    basis_order: BasisOrder,
) -> Result<Vec<Complex64>, FieldEvaluationError> {
    validate_field_inputs(
        eval_points,
        elements,
        nodes,
        surface_pressure,
        surface_velocity,
        physics,
        basis_order,
    )?;
    if basis_order != BasisOrder::Constant {
        return Err(FieldEvaluationError::CudaFailure {
            reason: "CUDA field evaluation currently supports only constant-basis traces".into(),
        });
    }

    let boundary_elements: Vec<(usize, &Element)> = elements
        .iter()
        .enumerate()
        .filter(|(_, element)| !element.property.is_evaluation())
        .collect();
    for (point_index, &point) in eval_points.iter().enumerate() {
        for (element_index, element) in &boundary_elements {
            if element.element_type == ElementType::Tri3 {
                // `validate_field_inputs` has already checked the element
                // connectivity length, so indexed access is panic-free here
                // while keeping the checked CUDA admission path fallible.
                let node0 = element.connectivity[0];
                let node1 = element.connectivity[1];
                let node2 = element.connectivity[2];
                if point_on_flat_triangle(point, [nodes[node0], nodes[node1], nodes[node2]]) {
                    return Err(FieldEvaluationError::CudaFailure {
                        reason: format!(
                            "evaluation point {point_index} lies on boundary element {element_index}; CUDA field evaluation requires off-surface points"
                        ),
                    });
                }
            }
        }
    }
    let mut cuda_elements = Vec::with_capacity(boundary_elements.len());
    for (boundary_index, (element_index, element)) in boundary_elements.iter().enumerate() {
        if element.element_type != ElementType::Tri3 {
            return Err(FieldEvaluationError::CudaFailure {
                reason: format!(
                    "boundary element {element_index} has {:?}; CUDA field supports Tri3 only",
                    element.element_type
                ),
            });
        }
        let [node0, node1, node2] = element.connectivity.as_slice() else {
            return Err(FieldEvaluationError::InvalidConnectivity {
                element: *element_index,
                expected: 3,
                actual: element.connectivity.len(),
            });
        };
        let velocity =
            surface_velocity.map_or(Complex64::new(0.0, 0.0), |values| values[boundary_index]);
        cuda_elements.push(sonium_compute::BemFieldElementC64::from_triangle(
            nodes[*node0],
            nodes[*node1],
            nodes[*node2],
            surface_pressure[boundary_index],
            velocity,
        ));
    }

    let operator = CudaBemFieldOperatorC64::new(
        pool.clone(),
        cuda_elements,
        physics.wave_number,
        physics.harmonic_factor,
        physics.pressure_factor,
    )
    .map_err(|error| FieldEvaluationError::CudaFailure {
        reason: error.to_string(),
    })?;
    let values =
        operator
            .try_apply(eval_points)
            .map_err(|error| FieldEvaluationError::CudaFailure {
                reason: error.to_string(),
            })?;
    validate_result_count(eval_points.len(), values.len())?;
    validate_finite_results(&values)?;
    Ok(values)
}

#[cfg(feature = "cuda")]
pub(crate) fn point_on_flat_triangle(point: [f64; 3], vertices: [[f64; 3]; 3]) -> bool {
    let edge0 = [
        vertices[1][0] - vertices[0][0],
        vertices[1][1] - vertices[0][1],
        vertices[1][2] - vertices[0][2],
    ];
    let edge1 = [
        vertices[2][0] - vertices[0][0],
        vertices[2][1] - vertices[0][1],
        vertices[2][2] - vertices[0][2],
    ];
    let normal = [
        edge0[1] * edge1[2] - edge0[2] * edge1[1],
        edge0[2] * edge1[0] - edge0[0] * edge1[2],
        edge0[0] * edge1[1] - edge0[1] * edge1[0],
    ];
    let jacobian = normal.iter().map(|value| value * value).sum::<f64>().sqrt();
    if jacobian <= f64::MIN_POSITIVE {
        return false;
    }
    let scale = edge0
        .iter()
        .chain(edge1.iter())
        .map(|value| value.abs())
        .fold(1.0_f64, f64::max);
    let offset = [
        point[0] - vertices[0][0],
        point[1] - vertices[0][1],
        point[2] - vertices[0][2],
    ];
    let plane_distance = normal
        .iter()
        .zip(offset)
        .map(|(normal, offset)| normal * offset)
        .sum::<f64>()
        / jacobian;
    if plane_distance.abs() > 1.0e-10 * scale {
        return false;
    }
    let dot00 = edge0.iter().map(|value| value * value).sum::<f64>();
    let dot01 = edge0
        .iter()
        .zip(edge1)
        .map(|(left, right)| left * right)
        .sum::<f64>();
    let dot11 = edge1.iter().map(|value| value * value).sum::<f64>();
    let dot02 = edge0
        .iter()
        .zip(offset)
        .map(|(left, right)| left * right)
        .sum::<f64>();
    let dot12 = edge1
        .iter()
        .zip(offset)
        .map(|(left, right)| left * right)
        .sum::<f64>();
    let denominator = dot00 * dot11 - dot01 * dot01;
    if denominator.abs() <= f64::MIN_POSITIVE {
        return false;
    }
    let barycentric_v = (dot11 * dot02 - dot01 * dot12) / denominator;
    let barycentric_w = (dot00 * dot12 - dot01 * dot02) / denominator;
    let tolerance = 1.0e-10;
    barycentric_v >= -tolerance
        && barycentric_w >= -tolerance
        && barycentric_v + barycentric_w <= 1.0 + tolerance
}

/// Compute the spatial gradient of the scattered pressure field with checked
/// input and output admission.
///
/// The returned vectors contain derivatives with respect to the evaluation
/// point coordinates, `[∂p/∂x, ∂p/∂y, ∂p/∂z]`.  The kernel derivative is
/// evaluated analytically at every quadrature point, so this path is suitable
/// for field-sensitivity and coupled-physics consumers without introducing a
/// finite-difference step-size contract.
pub fn compute_scattered_field_gradient_checked(
    eval_points: &[[f64; 3]],
    elements: &[Element],
    nodes: &[[f64; 3]],
    surface_pressure: &[Complex64],
    surface_velocity: Option<&[Complex64]>,
    physics: &PhysicsParams,
    basis_order: BasisOrder,
) -> Result<Vec<[Complex64; 3]>, FieldEvaluationError> {
    validate_field_inputs(
        eval_points,
        elements,
        nodes,
        surface_pressure,
        surface_velocity,
        physics,
        basis_order,
    )?;

    let values = compute_scattered_field_gradient(
        eval_points,
        elements,
        nodes,
        surface_pressure,
        surface_velocity,
        physics,
        basis_order,
        None,
    );
    validate_result_count(eval_points.len(), values.len())?;
    validate_finite_gradients(&values)?;
    Ok(values)
}

/// Checked total-field evaluation, including the incident-field contribution.
pub fn compute_total_field_checked(
    eval_points: &[[f64; 3]],
    elements: &[Element],
    nodes: &[[f64; 3]],
    surface_pressure: &[Complex64],
    surface_velocity: Option<&[Complex64]>,
    incident_field: &IncidentField,
    physics: &PhysicsParams,
    basis_order: BasisOrder,
) -> Result<Vec<FieldPoint>, FieldEvaluationError> {
    validate_field_inputs(
        eval_points,
        elements,
        nodes,
        surface_pressure,
        surface_velocity,
        physics,
        basis_order,
    )?;

    let p_incident = incident_field.evaluate_pressure(eval_points, physics);
    validate_result_count(eval_points.len(), p_incident.len())?;
    validate_finite_results(&p_incident)?;
    let p_scattered = compute_scattered_field_with_incident(
        eval_points,
        elements,
        nodes,
        surface_pressure,
        surface_velocity,
        Some(incident_field),
        physics,
        basis_order,
    );
    validate_result_count(eval_points.len(), p_scattered.len())?;
    validate_finite_results(&p_scattered)?;

    let mut results = Vec::with_capacity(eval_points.len());
    for (index, ((position, &p_incident), &p_scattered)) in eval_points
        .iter()
        .zip(p_incident.iter())
        .zip(p_scattered.iter())
        .enumerate()
    {
        let total = p_incident + p_scattered;
        if !total.re.is_finite() || !total.im.is_finite() {
            return Err(FieldEvaluationError::NonFiniteResult { point: index });
        }
        results.push(FieldPoint::new(*position, p_incident, p_scattered));
    }
    Ok(results)
}

/// Compute the spatial gradient of the total pressure field with checked
/// input and output admission.
///
/// This is the sum of the analytic incident-field gradient and the analytic
/// boundary-representation gradient returned by
/// [`compute_scattered_field_gradient_checked`].
pub fn compute_total_field_gradient_checked(
    eval_points: &[[f64; 3]],
    elements: &[Element],
    nodes: &[[f64; 3]],
    surface_pressure: &[Complex64],
    surface_velocity: Option<&[Complex64]>,
    incident_field: &IncidentField,
    physics: &PhysicsParams,
    basis_order: BasisOrder,
) -> Result<Vec<[Complex64; 3]>, FieldEvaluationError> {
    validate_field_inputs(
        eval_points,
        elements,
        nodes,
        surface_pressure,
        surface_velocity,
        physics,
        basis_order,
    )?;

    let incident = incident_field.evaluate_pressure_point_position_derivative(eval_points, physics);
    validate_result_count(eval_points.len(), incident.len())?;
    validate_finite_gradients(&incident)?;
    let scattered = compute_scattered_field_gradient(
        eval_points,
        elements,
        nodes,
        surface_pressure,
        surface_velocity,
        physics,
        basis_order,
        Some(incident_field),
    );
    validate_result_count(eval_points.len(), scattered.len())?;
    validate_finite_gradients(&scattered)?;

    let mut total = Vec::with_capacity(eval_points.len());
    for (point, (incident, scattered)) in incident.iter().zip(scattered.iter()).enumerate() {
        let mut value = [Complex64::new(0.0, 0.0); 3];
        for axis in 0..3 {
            value[axis] = incident[axis] + scattered[axis];
        }
        if value
            .iter()
            .any(|component| !component.re.is_finite() || !component.im.is_finite())
        {
            return Err(FieldEvaluationError::NonFiniteResult { point });
        }
        total.push(value);
    }
    Ok(total)
}

fn validate_field_inputs(
    eval_points: &[[f64; 3]],
    elements: &[Element],
    nodes: &[[f64; 3]],
    surface_pressure: &[Complex64],
    surface_velocity: Option<&[Complex64]>,
    physics: &PhysicsParams,
    basis_order: BasisOrder,
) -> Result<(), FieldEvaluationError> {
    physics
        .validate()
        .map_err(|error| FieldEvaluationError::InvalidPhysics {
            reason: error.to_string(),
        })?;

    for (point, coordinates) in eval_points.iter().enumerate() {
        if coordinates.iter().any(|value| !value.is_finite()) {
            return Err(FieldEvaluationError::NonFiniteEvaluationPoint { point });
        }
    }
    for (node, coordinates) in nodes.iter().enumerate() {
        if coordinates.iter().any(|value| !value.is_finite()) {
            return Err(FieldEvaluationError::NonFiniteNode { node });
        }
    }

    for (element, value) in elements.iter().enumerate() {
        let expected = value.element_type.num_nodes();
        if value.connectivity.len() != expected {
            return Err(FieldEvaluationError::InvalidConnectivity {
                element,
                expected,
                actual: value.connectivity.len(),
            });
        }
        for &node in &value.connectivity {
            if node >= nodes.len() {
                return Err(FieldEvaluationError::InvalidNodeIndex { element, node });
            }
        }
    }

    let boundary_elements: Vec<(usize, &Element)> = elements
        .iter()
        .enumerate()
        .filter(|(_, element)| !element.property.is_evaluation())
        .collect();
    if boundary_elements.is_empty() {
        return Err(FieldEvaluationError::EmptyBoundary);
    }

    let expected_pressure = match basis_order {
        BasisOrder::Constant => boundary_elements.len(),
        BasisOrder::Linear | BasisOrder::Quadratic => {
            let mut max_dof = None;
            for (element, value) in &boundary_elements {
                let expected = value.element_type.num_dofs_for_basis(basis_order);
                if value.dof_addresses.len() != expected {
                    return Err(FieldEvaluationError::InvalidDofLayout {
                        element: *element,
                        expected,
                        actual: value.dof_addresses.len(),
                    });
                }
                for (local, &dof) in value.dof_addresses.iter().enumerate() {
                    if dof == usize::MAX {
                        return Err(FieldEvaluationError::InvalidDofIndex {
                            element: *element,
                            local,
                            dof,
                        });
                    }
                    max_dof = Some(max_dof.map_or(dof, |current: usize| current.max(dof)));
                }
            }
            match max_dof {
                Some(max_dof) => max_dof + 1,
                None => {
                    // Every supported element has at least one node, but keep
                    // the checked boundary panic-free if that invariant ever
                    // changes.
                    return Err(FieldEvaluationError::InvalidDofLayout {
                        element: boundary_elements[0].0,
                        expected: boundary_elements[0].1.element_type.num_nodes(),
                        actual: 0,
                    });
                }
            }
        }
    };
    if surface_pressure.len() != expected_pressure {
        return Err(FieldEvaluationError::SurfacePressureCount {
            expected: expected_pressure,
            actual: surface_pressure.len(),
        });
    }
    for (index, sample) in surface_pressure.iter().enumerate() {
        if !sample.re.is_finite() || !sample.im.is_finite() {
            return Err(FieldEvaluationError::NonFiniteSurfaceSample { index });
        }
    }

    if let Some(velocity) = surface_velocity {
        let expected_velocity = match basis_order {
            BasisOrder::Constant => boundary_elements.len(),
            BasisOrder::Linear | BasisOrder::Quadratic => expected_pressure,
        };
        if velocity.len() != expected_velocity {
            return Err(FieldEvaluationError::SurfaceVelocityCount {
                expected: expected_velocity,
                actual: velocity.len(),
            });
        }
        for (index, sample) in velocity.iter().enumerate() {
            if !sample.re.is_finite() || !sample.im.is_finite() {
                return Err(FieldEvaluationError::NonFiniteSurfaceSample { index });
            }
        }
    }

    Ok(())
}

fn validate_finite_results(values: &[Complex64]) -> Result<(), FieldEvaluationError> {
    for (point, value) in values.iter().enumerate() {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(FieldEvaluationError::NonFiniteResult { point });
        }
    }
    Ok(())
}

fn validate_result_count(expected: usize, actual: usize) -> Result<(), FieldEvaluationError> {
    if expected != actual {
        return Err(FieldEvaluationError::ResultCount { expected, actual });
    }
    Ok(())
}

fn validate_finite_gradients(values: &[[Complex64; 3]]) -> Result<(), FieldEvaluationError> {
    for (point, value) in values.iter().enumerate() {
        if value
            .iter()
            .any(|component| !component.re.is_finite() || !component.im.is_finite())
        {
            return Err(FieldEvaluationError::NonFiniteResult { point });
        }
    }
    Ok(())
}

/// Analytic spatial derivative of the boundary-representation field.
fn compute_scattered_field_gradient(
    eval_points: &[[f64; 3]],
    elements: &[Element],
    nodes: &[[f64; 3]],
    surface_pressure: &[Complex64],
    surface_velocity: Option<&[Complex64]>,
    physics: &PhysicsParams,
    basis_order: BasisOrder,
    incident_field: Option<&IncidentField>,
) -> Vec<[Complex64; 3]> {
    let k = physics.wave_number;
    let wavruim = k * physics.harmonic_factor;
    let boundary_elements: Vec<_> = elements
        .iter()
        .filter(|element| !element.property.is_evaluation())
        .collect();
    let all_coords: Vec<DenseMatrix<f64>> = boundary_elements
        .iter()
        .map(|element| get_element_coords(element, nodes))
        .collect();

    // Per-point field evaluation (rayon above the machine-adaptive gate,
    // serial for small point × element counts). Work is evaluation points ×
    // boundary elements kernel evaluations.
    let eval_work = eval_points.len().saturating_mul(boundary_elements.len());
    gated_map(
        eval_points,
        eval_work,
        BEM_QUADRATURE_PAIRS_PER_THREAD,
        |point| {
            let mut gradient = [Complex64::new(0.0, 0.0); 3];
            match basis_order {
                BasisOrder::Constant => {
                    for (index, element) in boundary_elements.iter().enumerate() {
                        let (pressure, velocity) = constant_cauchy_data(
                            element,
                            index,
                            surface_pressure,
                            surface_velocity,
                            incident_field,
                            physics,
                        );
                        let contribution = integrate_element_field_gradient(
                            point,
                            &all_coords[index],
                            element.element_type,
                            pressure,
                            velocity,
                            k,
                            wavruim,
                            physics.pressure_factor,
                            None,
                        );
                        for axis in 0..3 {
                            gradient[axis] += contribution[axis];
                        }
                    }
                }
                BasisOrder::Linear | BasisOrder::Quadratic => {
                    for (index, element) in boundary_elements.iter().enumerate() {
                        let contribution = integrate_element_field_discontinuous_gradient(
                            point,
                            &all_coords[index],
                            element.element_type,
                            basis_order,
                            &element.dof_addresses,
                            surface_pressure,
                            surface_velocity,
                            k,
                            wavruim,
                            physics.pressure_factor,
                            None,
                        );
                        for axis in 0..3 {
                            gradient[axis] += contribution[axis];
                        }
                    }
                }
            }
            gradient
        },
    )
}

/// Compute scattered field at evaluation points using surface solution
///
/// Uses the Kirchhoff-Helmholtz integral equation for exterior problems:
/// p_scat(x) = ∫_Γ [p(y) ∂G/∂n_y(x,y) - v_n(y) G(x,y)] dS_y
///
/// This implementation uses proper Gauss quadrature over each element,
/// matching the C++ NumCalc NC_IntegrationTBEM function for accuracy.
///
/// # Arguments
/// * `eval_points` - Evaluation points (N × 3 array)
/// * `elements` - Mesh elements with surface solution
/// * `nodes` - Node coordinates
/// * `surface_pressure` - Solved surface pressure values (one per element for constant,
///   or one per DOF for linear)
/// * `surface_velocity` - Surface normal velocity (usually from BC, can be zero for rigid)
/// * `physics` - Physical parameters
/// * `basis_order` - Basis function order
///
/// # Returns
/// Scattered pressure at each evaluation point
pub fn compute_scattered_field(
    eval_points: &[[f64; 3]],
    elements: &[Element],
    nodes: &[[f64; 3]],
    surface_pressure: &[Complex64],
    surface_velocity: Option<&[Complex64]>,
    physics: &PhysicsParams,
    basis_order: BasisOrder,
) -> Vec<Complex64> {
    compute_scattered_field_with_incident(
        eval_points,
        elements,
        nodes,
        surface_pressure,
        surface_velocity,
        None,
        physics,
        basis_order,
    )
}

/// Build the constant-basis exterior field operator.
///
/// Returns `(C, q)` such that the evaluated pressure is
/// `p_eval = C p_surface + q`. The affine term `q` contains the prescribed
/// surface-velocity contribution. Exposing this linear map lets application
/// adjoints apply `Cᴴ` directly instead of differentiating thousands of
/// individual field evaluations.
pub fn build_constant_field_operator(
    eval_points: &[[f64; 3]],
    elements: &[Element],
    nodes: &[[f64; 3]],
    surface_velocity: Option<&[Complex64]>,
    physics: &PhysicsParams,
) -> (DenseMatrix<Complex64>, Vec<Complex64>) {
    let boundary_elements: Vec<_> = elements
        .iter()
        .filter(|element| !element.property.is_evaluation())
        .collect();
    if let Some(velocity) = surface_velocity {
        assert_eq!(
            velocity.len(),
            boundary_elements.len(),
            "constant field operator needs one velocity per boundary element"
        );
    }
    let coordinates: Vec<_> = boundary_elements
        .iter()
        .map(|element| get_element_coords(element, nodes))
        .collect();
    let wavruim = physics.wave_number * physics.harmonic_factor;
    let eval_work = eval_points.len().saturating_mul(boundary_elements.len());
    let rows: Vec<(Vec<Complex64>, Complex64)> = gated_map(
        eval_points,
        eval_work,
        BEM_QUADRATURE_PAIRS_PER_THREAD,
        |point| {
            let mut row = Vec::with_capacity(boundary_elements.len());
            let mut affine = Complex64::new(0.0, 0.0);
            for (index, element) in boundary_elements.iter().enumerate() {
                row.push(integrate_element_field(
                    point,
                    &coordinates[index],
                    element.element_type,
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    physics.wave_number,
                    wavruim,
                    physics.pressure_factor,
                    None,
                ));
                if let Some(velocity) = surface_velocity {
                    affine += integrate_element_field(
                        point,
                        &coordinates[index],
                        element.element_type,
                        Complex64::new(0.0, 0.0),
                        velocity[index],
                        physics.wave_number,
                        wavruim,
                        physics.pressure_factor,
                        None,
                    );
                }
            }
            (row, affine)
        },
    );
    let affine = rows.iter().map(|(_, value)| *value).collect();
    let matrix = DenseMatrix::from_row_major_vec(
        eval_points.len(),
        boundary_elements.len(),
        rows.into_iter().flat_map(|(row, _)| row).collect(),
    );
    (matrix, affine)
}

/// Build a checked exterior field operator for constant, linear, or quadratic
/// surface data.
///
/// The returned `(C, q)` satisfies `p_eval = C p_surface + q`. Constant-basis
/// columns are ordered by boundary element. Higher-order columns use the
/// element `dof_addresses`, so shared continuous DOFs are accumulated into a
/// single column. The affine term contains only prescribed surface velocity.
pub fn build_field_operator_checked(
    eval_points: &[[f64; 3]],
    elements: &[Element],
    nodes: &[[f64; 3]],
    surface_velocity: Option<&[Complex64]>,
    physics: &PhysicsParams,
    basis_order: BasisOrder,
) -> Result<(DenseMatrix<Complex64>, Vec<Complex64>), FieldEvaluationError> {
    let boundary_elements: Vec<_> = elements
        .iter()
        .filter(|element| !element.property.is_evaluation())
        .collect();
    let pressure_dof_count = field_pressure_dof_count(&boundary_elements, basis_order)?;
    let zero_pressure = vec![Complex64::new(0.0, 0.0); pressure_dof_count];
    validate_field_inputs(
        eval_points,
        elements,
        nodes,
        &zero_pressure,
        surface_velocity,
        physics,
        basis_order,
    )?;

    let coordinates: Vec<_> = boundary_elements
        .iter()
        .map(|element| get_element_coords(element, nodes))
        .collect();
    let wavruim = physics.wave_number * physics.harmonic_factor;
    let eval_work = eval_points.len().saturating_mul(boundary_elements.len());
    let rows: Vec<(Vec<Complex64>, Complex64)> = gated_map(
        eval_points,
        eval_work,
        BEM_QUADRATURE_PAIRS_PER_THREAD,
        |point| {
            let mut row = vec![Complex64::new(0.0, 0.0); pressure_dof_count];
            let mut affine = Complex64::new(0.0, 0.0);

            for (element_index, element) in boundary_elements.iter().enumerate() {
                let element_coords = &coordinates[element_index];
                match basis_order {
                    BasisOrder::Constant => {
                        row[element_index] += integrate_element_field(
                            point,
                            element_coords,
                            element.element_type,
                            Complex64::new(1.0, 0.0),
                            Complex64::new(0.0, 0.0),
                            physics.wave_number,
                            wavruim,
                            physics.pressure_factor,
                            None,
                        );
                        if let Some(velocity) = surface_velocity {
                            affine += integrate_element_field(
                                point,
                                element_coords,
                                element.element_type,
                                Complex64::new(0.0, 0.0),
                                velocity[element_index],
                                physics.wave_number,
                                wavruim,
                                physics.pressure_factor,
                                None,
                            );
                        }
                    }
                    BasisOrder::Linear | BasisOrder::Quadratic => {
                        for (local_basis, &dof) in element.dof_addresses.iter().enumerate() {
                            row[dof] += integrate_element_field_discontinuous(
                                point,
                                element_coords,
                                element.element_type,
                                basis_order,
                                &element.dof_addresses,
                                &zero_pressure,
                                None,
                                physics.wave_number,
                                wavruim,
                                physics.pressure_factor,
                                None,
                                Some(local_basis),
                            );
                        }
                        if let Some(velocity) = surface_velocity {
                            affine += integrate_element_field_discontinuous(
                                point,
                                element_coords,
                                element.element_type,
                                basis_order,
                                &element.dof_addresses,
                                &zero_pressure,
                                Some(velocity),
                                physics.wave_number,
                                wavruim,
                                physics.pressure_factor,
                                None,
                                None,
                            );
                        }
                    }
                }
            }

            (row, affine)
        },
    );

    for (row, affine) in &rows {
        validate_finite_results(row)?;
        validate_finite_results(std::slice::from_ref(affine))?;
    }
    let affine = rows.iter().map(|(_, value)| *value).collect();
    let matrix = DenseMatrix::from_row_major_vec(
        eval_points.len(),
        pressure_dof_count,
        rows.into_iter().flat_map(|(row, _)| row).collect(),
    );
    Ok((matrix, affine))
}

fn field_pressure_dof_count(
    boundary_elements: &[&Element],
    basis_order: BasisOrder,
) -> Result<usize, FieldEvaluationError> {
    match basis_order {
        BasisOrder::Constant => Ok(boundary_elements.len()),
        BasisOrder::Linear | BasisOrder::Quadratic => {
            let mut max_dof = None;
            for (element, value) in boundary_elements.iter().enumerate() {
                let expected = value.element_type.num_dofs_for_basis(basis_order);
                if value.dof_addresses.len() != expected {
                    return Err(FieldEvaluationError::InvalidDofLayout {
                        element,
                        expected,
                        actual: value.dof_addresses.len(),
                    });
                }
                for (local, &dof) in value.dof_addresses.iter().enumerate() {
                    if dof == usize::MAX {
                        return Err(FieldEvaluationError::InvalidDofIndex {
                            element,
                            local,
                            dof,
                        });
                    }
                    max_dof = Some(max_dof.map_or(dof, |current: usize| current.max(dof)));
                }
            }
            max_dof
                .map(|value| value + 1)
                .ok_or(FieldEvaluationError::EmptyBoundary)
        }
    }
}

/// Convert Euclidean adjoint coefficients into primal trace coefficients.
///
/// A higher-order adjoint solve returns coordinates in the algebraic dual of
/// the trace basis. Field evaluation expects coefficients of the primal shape
/// functions, so this helper applies the inverse surface L² mass (Riesz) map.
/// Constant panels already use one coefficient per element and retain their
/// historical identity mapping.
pub fn map_dual_basis_coefficients_checked(
    dual_coefficients: &[Complex64],
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    basis_order: BasisOrder,
) -> Result<Vec<Complex64>, FieldEvaluationError> {
    validate_field_inputs(
        &[],
        elements,
        nodes,
        dual_coefficients,
        None,
        physics,
        basis_order,
    )?;

    if basis_order == BasisOrder::Constant {
        return Ok(dual_coefficients.to_vec());
    }

    let boundary_elements: Vec<&Element> = elements
        .iter()
        .filter(|element| !element.property.is_evaluation())
        .collect();
    let dof_count = field_pressure_dof_count(&boundary_elements, basis_order)?;
    let mut mass = DenseMatrix::<Complex64>::zeros(dof_count, dof_count);
    const QUADRATURE_ORDER: usize = 5;

    for (element_index, element) in boundary_elements.iter().enumerate() {
        let local_count = element.element_type.num_dofs_for_basis(basis_order);
        let coords = get_element_coords(element, nodes);
        let mut accumulate = |values: &[f64], weight: f64| {
            if values.len() != local_count
                || !weight.is_finite()
                || values.iter().any(|value| !value.is_finite())
            {
                return Err(FieldEvaluationError::DualBasisSolve {
                    reason: format!("non-finite mass quadrature data on element {element_index}"),
                });
            }
            for (local_row, &row) in element.dof_addresses.iter().enumerate() {
                for (local_column, &column) in element.dof_addresses.iter().enumerate() {
                    mass[[row, column]] +=
                        Complex64::new(weight * values[local_row] * values[local_column], 0.0);
                }
            }
            Ok(())
        };

        match element.element_type {
            ElementType::Tri3 => {
                let edge_1 = [
                    coords[[1, 0]] - coords[[0, 0]],
                    coords[[1, 1]] - coords[[0, 1]],
                    coords[[1, 2]] - coords[[0, 2]],
                ];
                let edge_2 = [
                    coords[[2, 0]] - coords[[0, 0]],
                    coords[[2, 1]] - coords[[0, 1]],
                    coords[[2, 2]] - coords[[0, 2]],
                ];
                let cross = [
                    edge_1[1] * edge_2[2] - edge_1[2] * edge_2[1],
                    edge_1[2] * edge_2[0] - edge_1[0] * edge_2[2],
                    edge_1[0] * edge_2[1] - edge_1[1] * edge_2[0],
                ];
                let jacobian =
                    (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
                if !jacobian.is_finite() || jacobian <= f64::MIN_POSITIVE {
                    return Err(FieldEvaluationError::DualBasisSolve {
                        reason: format!(
                            "element {element_index} has degenerate triangular geometry"
                        ),
                    });
                }
                for (s, t, reference_weight) in triangle_quadrature(QUADRATURE_ORDER) {
                    let values = basis_values(ElementType::Tri3, basis_order, s, t);
                    accumulate(&values, reference_weight * jacobian)?;
                }
            }
            ElementType::Quad4 => {
                for (s, t, reference_weight) in quad_quadrature(QUADRATURE_ORDER) {
                    let shape_fn = [
                        0.25 * (1.0 - s) * (1.0 - t),
                        0.25 * (1.0 + s) * (1.0 - t),
                        0.25 * (1.0 + s) * (1.0 + t),
                        0.25 * (1.0 - s) * (1.0 + t),
                    ];
                    let shape_ds = [
                        -0.25 * (1.0 - t),
                        0.25 * (1.0 - t),
                        0.25 * (1.0 + t),
                        -0.25 * (1.0 + t),
                    ];
                    let shape_dt = [
                        -0.25 * (1.0 - s),
                        -0.25 * (1.0 + s),
                        0.25 * (1.0 + s),
                        0.25 * (1.0 - s),
                    ];
                    let (_, jacobian, _) =
                        quadrature_geometry(&coords, &shape_fn, &shape_ds, &shape_dt, None);
                    if !jacobian.is_finite() || jacobian <= f64::MIN_POSITIVE {
                        return Err(FieldEvaluationError::DualBasisSolve {
                            reason: format!(
                                "element {element_index} has degenerate quadrilateral geometry"
                            ),
                        });
                    }
                    let values = basis_values(ElementType::Quad4, basis_order, s, t);
                    accumulate(&values, reference_weight * jacobian)?;
                }
            }
        }
    }

    if mass
        .as_slice()
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(FieldEvaluationError::DualBasisSolve {
            reason: "surface mass matrix is non-finite".to_string(),
        });
    }
    let primal = lu_solve(&mass, dual_coefficients).map_err(|error| {
        FieldEvaluationError::DualBasisSolve {
            reason: error.to_string(),
        }
    })?;
    if primal
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(FieldEvaluationError::DualBasisSolve {
            reason: "surface mass solve returned a non-finite coefficient".to_string(),
        });
    }
    Ok(primal)
}

/// Evaluate selected columns of the constant-basis exterior field operator
/// together with their affine velocity contributions.
///
/// Partial-evaluation counterpart of [`build_constant_field_operator`]: this
/// returns the operator values for each requested column at every evaluation
/// point, plus the affine contribution from exactly those columns. This is
/// useful for geometry-sensitivity code that only needs to rebuild perturbed
/// columns.
///
/// `columns` indexes the boundary (non-evaluation) element list, matching the
/// column indexing returned by [`build_constant_field_operator`].
pub fn eval_constant_field_operator_columns(
    eval_points: &[[f64; 3]],
    elements: &[Element],
    nodes: &[[f64; 3]],
    surface_velocity: Option<&[Complex64]>,
    physics: &PhysicsParams,
    columns: &[usize],
) -> (Vec<Vec<Complex64>>, Vec<Complex64>) {
    let boundary_elements: Vec<_> = elements
        .iter()
        .filter(|element| !element.property.is_evaluation())
        .collect();
    if let Some(velocity) = surface_velocity {
        assert_eq!(
            velocity.len(),
            boundary_elements.len(),
            "constant field operator needs one velocity per boundary element"
        );
    }
    let coordinates: Vec<_> = columns
        .iter()
        .map(|&column| get_element_coords(boundary_elements[column], nodes))
        .collect();
    let wavruim = physics.wave_number * physics.harmonic_factor;
    let eval_work = eval_points.len().saturating_mul(columns.len());
    let rows: Vec<(Vec<Complex64>, Complex64)> = gated_map(
        eval_points,
        eval_work,
        BEM_QUADRATURE_PAIRS_PER_THREAD,
        |point| {
            let mut values = Vec::with_capacity(columns.len());
            let mut affine = Complex64::new(0.0, 0.0);
            for (position, &column) in columns.iter().enumerate() {
                let element = boundary_elements[column];
                values.push(integrate_element_field(
                    point,
                    &coordinates[position],
                    element.element_type,
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    physics.wave_number,
                    wavruim,
                    physics.pressure_factor,
                    None,
                ));
                if let Some(velocity) = surface_velocity {
                    affine += integrate_element_field(
                        point,
                        &coordinates[position],
                        element.element_type,
                        Complex64::new(0.0, 0.0),
                        velocity[column],
                        physics.wave_number,
                        wavruim,
                        physics.pressure_factor,
                        None,
                    );
                }
            }
            (values, affine)
        },
    );
    let affine = rows.iter().map(|(_, value)| *value).collect();
    let mut column_values = vec![vec![Complex64::new(0.0, 0.0); eval_points.len()]; columns.len()];
    for (point_index, (values, _)) in rows.into_iter().enumerate() {
        for (position, value) in values.into_iter().enumerate() {
            column_values[position][point_index] = value;
        }
    }
    (column_values, affine)
}

/// Compute a single-layer potential field from per-element density values.
pub fn compute_single_layer_field(
    eval_points: &[[f64; 3]],
    elements: &[Element],
    nodes: &[[f64; 3]],
    density: &[Complex64],
    physics: &PhysicsParams,
    basis_order: BasisOrder,
) -> Vec<Complex64> {
    let i_omega_rho = Complex64::new(0.0, physics.pressure_factor);
    let velocity: Option<Vec<Complex64>> = if i_omega_rho.norm() > 1.0e-30 {
        Some(density.iter().map(|&sigma| -sigma / i_omega_rho).collect())
    } else {
        None
    };

    let zero_pressure = vec![Complex64::new(0.0, 0.0); density.len()];
    compute_scattered_field(
        eval_points,
        elements,
        nodes,
        &zero_pressure,
        velocity.as_deref(),
        physics,
        basis_order,
    )
}

fn compute_scattered_field_with_incident(
    eval_points: &[[f64; 3]],
    elements: &[Element],
    nodes: &[[f64; 3]],
    surface_pressure: &[Complex64],
    surface_velocity: Option<&[Complex64]>,
    incident_field: Option<&IncidentField>,
    physics: &PhysicsParams,
    basis_order: BasisOrder,
) -> Vec<Complex64> {
    let n_eval = eval_points.len();
    let k = physics.wave_number;
    let harmonic_factor = physics.harmonic_factor;
    let wavruim = k * harmonic_factor;

    // Get boundary elements (exclude evaluation-only elements)
    let boundary_elements: Vec<_> = elements
        .iter()
        .filter(|e| !e.property.is_evaluation())
        .collect();

    // Precompute coordinates for all boundary elements once
    let all_coords: Vec<DenseMatrix<f64>> = boundary_elements
        .iter()
        .map(|e| get_element_coords(e, nodes))
        .collect();

    let eval_work = n_eval.saturating_mul(boundary_elements.len());
    let p_scattered_vec: Vec<Complex64> =
        gated_map_indexed(n_eval, eval_work, BEM_QUADRATURE_PAIRS_PER_THREAD, |i| {
            let x: Vec<f64> = eval_points[i].to_vec();
            let mut p_scat = Complex64::new(0.0, 0.0);

            match basis_order {
                BasisOrder::Constant => {
                    for (j, elem) in boundary_elements.iter().enumerate() {
                        let (p_surf, v_surf) = constant_cauchy_data(
                            elem,
                            j,
                            surface_pressure,
                            surface_velocity,
                            incident_field,
                            physics,
                        );

                        let elem_coords = &all_coords[j];

                        let contribution = integrate_element_field(
                            &x,
                            elem_coords,
                            elem.element_type,
                            p_surf,
                            v_surf,
                            k,
                            wavruim,
                            physics.pressure_factor,
                            None,
                        );

                        p_scat += contribution;
                    }
                }
                BasisOrder::Linear | BasisOrder::Quadratic => {
                    for (j, elem) in boundary_elements.iter().enumerate() {
                        let elem_coords = &all_coords[j];
                        // For linear elements, interpolate pressure using shape functions
                        // at each quadrature point. Tri3 geometry remains flat,
                        // so its normal must remain the flat geometric normal.
                        let contribution = integrate_element_field_discontinuous(
                            &x,
                            elem_coords,
                            elem.element_type,
                            basis_order,
                            &elem.dof_addresses,
                            surface_pressure,
                            surface_velocity,
                            k,
                            wavruim,
                            physics.pressure_factor,
                            None,
                            None,
                        );

                        p_scat += contribution;
                    }
                }
            }
            p_scat
        });

    p_scattered_vec
}

fn constant_cauchy_data(
    elem: &Element,
    dof: usize,
    solution: &[Complex64],
    explicit_velocity: Option<&[Complex64]>,
    incident_field: Option<&IncidentField>,
    physics: &PhysicsParams,
) -> (Complex64, Complex64) {
    if let Some(velocity) = explicit_velocity {
        return (solution[dof], velocity[dof]);
    }

    let avg = |values: &[Complex64]| {
        values.iter().copied().sum::<Complex64>() / values.len().max(1) as f64
    };

    match &elem.boundary_condition {
        BoundaryCondition::Pressure(values) => {
            let total_pressure = avg(values);
            let total_normal_derivative = solution[dof];
            let (incident_pressure, incident_normal_derivative) =
                incident_cauchy_data(incident_field, &elem.center, &elem.normal, physics);
            let scattered_pressure = total_pressure - incident_pressure;
            let scattered_normal_derivative = total_normal_derivative - incident_normal_derivative;
            (
                scattered_pressure,
                normal_derivative_to_velocity(scattered_normal_derivative, physics),
            )
        }
        BoundaryCondition::Velocity(values) => (solution[dof], avg(values)),
        BoundaryCondition::VelocityWithAdmittance {
            velocity,
            admittance,
        } => {
            let pressure = solution[dof];
            (pressure, avg(velocity) + *admittance * pressure)
        }
        _ => (solution[dof], Complex64::new(0.0, 0.0)),
    }
}

fn incident_cauchy_data(
    incident_field: Option<&IncidentField>,
    point: &[f64; 3],
    normal: &[f64; 3],
    physics: &PhysicsParams,
) -> (Complex64, Complex64) {
    let Some(incident) = incident_field else {
        return (Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0));
    };

    let point_arr: [[f64; 3]; 1] = [*point];
    let normal_arr: [[f64; 3]; 1] = [*normal];
    let p_inc = incident.evaluate_pressure(&point_arr, physics)[0];
    let q_inc = incident.evaluate_normal_derivative(&point_arr, &normal_arr, physics)[0];
    (p_inc, q_inc)
}

fn normal_derivative_to_velocity(
    normal_derivative: Complex64,
    physics: &PhysicsParams,
) -> Complex64 {
    let i_omega_rho = Complex64::new(0.0, physics.pressure_factor);
    if i_omega_rho.norm() > 1e-30 {
        normal_derivative / i_omega_rho
    } else {
        Complex64::new(0.0, 0.0)
    }
}

/// Get element node coordinates from connectivity
fn get_element_coords(elem: &Element, nodes: &[[f64; 3]]) -> DenseMatrix<f64> {
    let n_nodes = elem.connectivity.len();
    let mut coords = DenseMatrix::<f64>::zeros(n_nodes, 3);
    for (i, &node_idx) in elem.connectivity.iter().enumerate() {
        for j in 0..3 {
            coords[[i, j]] = nodes[node_idx][j];
        }
    }
    coords
}

/// Integrate element contribution using Gauss quadrature
///
/// This matches the NC_IntegrationTBEM function from NumCalc.
fn integrate_element_field(
    x: &[f64],
    elem_coords: &DenseMatrix<f64>,
    elem_type: ElementType,
    p_surf: Complex64,
    v_surf: Complex64,
    _k: f64,
    wavruim: f64,
    pressure_factor: f64,
    node_normals: Option<&[[f64; 3]]>,
) -> Complex64 {
    let mut result = Complex64::new(0.0, 0.0);

    match elem_type {
        ElementType::Tri3 => {
            // Use order 3 quadrature (7-point for triangles)
            let gauss_order = 3;
            let quad_points = triangle_quadrature(gauss_order);

            for (xi, eta, weight) in quad_points {
                // Triangle shape functions: N0 = 1-xi-eta, N1 = xi, N2 = eta
                let shape_fn = [1.0 - xi - eta, xi, eta];
                let shape_ds = [-1.0, 1.0, 0.0];
                let shape_dt = [-1.0, 0.0, 1.0];

                result += integrate_quad_point(
                    x,
                    elem_coords,
                    &shape_fn,
                    &shape_ds,
                    &shape_dt,
                    weight,
                    p_surf,
                    v_surf,
                    wavruim,
                    3,
                    pressure_factor,
                    node_normals,
                );
            }
        }
        ElementType::Quad4 => {
            // Use 2x2 Gauss quadrature for quads (4 points)
            // Gauss points on [-1,1]: ±1/√3 ≈ ±0.577350269
            let g = 0.5773502691896257_f64;
            let gauss_pts = [(-g, -g), (g, -g), (g, g), (-g, g)];
            let weight = 1.0; // Each point has weight 1.0 for 2x2 Gauss on [-1,1]²

            for (xi, eta) in gauss_pts {
                // Bilinear quad shape functions on [-1,1]²
                let shape_fn = [
                    0.25 * (1.0 - xi) * (1.0 - eta),
                    0.25 * (1.0 + xi) * (1.0 - eta),
                    0.25 * (1.0 + xi) * (1.0 + eta),
                    0.25 * (1.0 - xi) * (1.0 + eta),
                ];
                let shape_ds = [
                    -0.25 * (1.0 - eta),
                    0.25 * (1.0 - eta),
                    0.25 * (1.0 + eta),
                    -0.25 * (1.0 + eta),
                ];
                let shape_dt = [
                    -0.25 * (1.0 - xi),
                    -0.25 * (1.0 + xi),
                    0.25 * (1.0 + xi),
                    0.25 * (1.0 - xi),
                ];

                result += integrate_quad_point(
                    x,
                    elem_coords,
                    &shape_fn,
                    &shape_ds,
                    &shape_dt,
                    weight,
                    p_surf,
                    v_surf,
                    wavruim,
                    4,
                    pressure_factor,
                    node_normals,
                );
            }
        }
    }

    result
}

fn integrate_element_field_gradient(
    x: &[f64; 3],
    elem_coords: &DenseMatrix<f64>,
    elem_type: ElementType,
    p_surf: Complex64,
    v_surf: Complex64,
    _k: f64,
    wavruim: f64,
    pressure_factor: f64,
    node_normals: Option<&[[f64; 3]]>,
) -> [Complex64; 3] {
    let mut result = [Complex64::new(0.0, 0.0); 3];
    let mut add = |contribution: [Complex64; 3]| {
        for axis in 0..3 {
            result[axis] += contribution[axis];
        }
    };

    match elem_type {
        ElementType::Tri3 => {
            for (xi, eta, weight) in triangle_quadrature(3) {
                let shape_fn = [1.0 - xi - eta, xi, eta];
                let shape_ds = [-1.0, 1.0, 0.0];
                let shape_dt = [-1.0, 0.0, 1.0];
                add(integrate_quad_point_gradient(
                    x,
                    elem_coords,
                    &shape_fn,
                    &shape_ds,
                    &shape_dt,
                    weight,
                    p_surf,
                    v_surf,
                    wavruim,
                    pressure_factor,
                    node_normals,
                ));
            }
        }
        ElementType::Quad4 => {
            let g = 0.5773502691896257_f64;
            for (xi, eta) in [(-g, -g), (g, -g), (g, g), (-g, g)] {
                let shape_fn = [
                    0.25 * (1.0 - xi) * (1.0 - eta),
                    0.25 * (1.0 + xi) * (1.0 - eta),
                    0.25 * (1.0 + xi) * (1.0 + eta),
                    0.25 * (1.0 - xi) * (1.0 + eta),
                ];
                let shape_ds = [
                    -0.25 * (1.0 - eta),
                    0.25 * (1.0 - eta),
                    0.25 * (1.0 + eta),
                    -0.25 * (1.0 + eta),
                ];
                let shape_dt = [
                    -0.25 * (1.0 - xi),
                    -0.25 * (1.0 + xi),
                    0.25 * (1.0 + xi),
                    0.25 * (1.0 - xi),
                ];
                add(integrate_quad_point_gradient(
                    x,
                    elem_coords,
                    &shape_fn,
                    &shape_ds,
                    &shape_dt,
                    1.0,
                    p_surf,
                    v_surf,
                    wavruim,
                    pressure_factor,
                    node_normals,
                ));
            }
        }
    }
    result
}

fn integrate_element_field_discontinuous_gradient(
    x: &[f64; 3],
    elem_coords: &DenseMatrix<f64>,
    elem_type: ElementType,
    basis_order: BasisOrder,
    dof_addresses: &[usize],
    surface_pressure: &[Complex64],
    surface_velocity: Option<&[Complex64]>,
    _k: f64,
    wavruim: f64,
    pressure_factor: f64,
    node_normals: Option<&[[f64; 3]]>,
) -> [Complex64; 3] {
    let mut result = [Complex64::new(0.0, 0.0); 3];
    let mut add = |contribution: [Complex64; 3]| {
        for axis in 0..3 {
            result[axis] += contribution[axis];
        }
    };
    let mut integrate =
        |xi: f64, eta: f64, shape_fn: &[f64], shape_ds: &[f64], shape_dt: &[f64], weight: f64| {
            let basis_values = panel_basis_values(elem_type, basis_order, xi, eta);
            let pressure = interpolate_discontinuous_data(
                Some(surface_pressure),
                dof_addresses,
                &basis_values,
            );
            let velocity =
                interpolate_discontinuous_data(surface_velocity, dof_addresses, &basis_values);
            add(integrate_quad_point_gradient(
                x,
                elem_coords,
                shape_fn,
                shape_ds,
                shape_dt,
                weight,
                pressure,
                velocity,
                wavruim,
                pressure_factor,
                node_normals,
            ));
        };

    match elem_type {
        ElementType::Tri3 => {
            let gauss_order = if matches!(basis_order, BasisOrder::Quadratic) {
                5
            } else {
                3
            };
            for (xi, eta, weight) in triangle_quadrature(gauss_order) {
                let shape_fn = [1.0 - xi - eta, xi, eta];
                let shape_ds = [-1.0, 1.0, 0.0];
                let shape_dt = [-1.0, 0.0, 1.0];
                integrate(xi, eta, &shape_fn, &shape_ds, &shape_dt, weight);
            }
        }
        ElementType::Quad4 => {
            let g = 0.5773502691896257_f64;
            for (xi, eta) in [(-g, -g), (g, -g), (g, g), (-g, g)] {
                let shape_fn = [
                    0.25 * (1.0 - xi) * (1.0 - eta),
                    0.25 * (1.0 + xi) * (1.0 - eta),
                    0.25 * (1.0 + xi) * (1.0 + eta),
                    0.25 * (1.0 - xi) * (1.0 + eta),
                ];
                let shape_ds = [
                    -0.25 * (1.0 - eta),
                    0.25 * (1.0 - eta),
                    0.25 * (1.0 + eta),
                    -0.25 * (1.0 + eta),
                ];
                let shape_dt = [
                    -0.25 * (1.0 - xi),
                    -0.25 * (1.0 + xi),
                    0.25 * (1.0 + xi),
                    0.25 * (1.0 - xi),
                ];
                integrate(xi, eta, &shape_fn, &shape_ds, &shape_dt, 1.0);
            }
        }
    }
    result
}

/// Integrate element contribution for linear elements
///
/// The surface pressure is interpolated using shape functions:
///   p(y_q) = sum_b N_b(y_q) * surface_pressure[dof_addresses[b]]
fn integrate_element_field_discontinuous(
    x: &[f64],
    elem_coords: &DenseMatrix<f64>,
    elem_type: ElementType,
    basis_order: BasisOrder,
    dof_addresses: &[usize],
    surface_pressure: &[Complex64],
    surface_velocity: Option<&[Complex64]>,
    _k: f64,
    wavruim: f64,
    pressure_factor: f64,
    node_normals: Option<&[[f64; 3]]>,
    basis_column: Option<usize>,
) -> Complex64 {
    let mut result = Complex64::new(0.0, 0.0);

    match elem_type {
        ElementType::Tri3 => {
            let gauss_order = if matches!(basis_order, BasisOrder::Quadratic) {
                5
            } else {
                3
            };
            let quad_points = triangle_quadrature(gauss_order);

            for (xi, eta, weight) in quad_points {
                let shape_fn = [1.0 - xi - eta, xi, eta];
                let shape_ds = [-1.0, 1.0, 0.0];
                let shape_dt = [-1.0, 0.0, 1.0];
                let basis_values = panel_basis_values(elem_type, basis_order, xi, eta);

                // Interpolate pressure at this quadrature point
                let p_surf = basis_column.map_or_else(
                    || {
                        interpolate_discontinuous_data(
                            Some(surface_pressure),
                            dof_addresses,
                            &basis_values,
                        )
                    },
                    |local_basis| Complex64::new(basis_values[local_basis], 0.0),
                );
                let v_surf = if basis_column.is_some() {
                    Complex64::new(0.0, 0.0)
                } else {
                    interpolate_discontinuous_data(surface_velocity, dof_addresses, &basis_values)
                };

                result += integrate_quad_point(
                    x,
                    elem_coords,
                    &shape_fn,
                    &shape_ds,
                    &shape_dt,
                    weight,
                    p_surf,
                    v_surf,
                    wavruim,
                    3,
                    pressure_factor,
                    node_normals,
                );
            }
        }
        ElementType::Quad4 => {
            let gauss_order = if matches!(basis_order, BasisOrder::Quadratic) {
                3
            } else {
                2
            };
            let gauss_pts = quad_quadrature(gauss_order);

            for (xi, eta, weight) in gauss_pts {
                let shape_fn = [
                    0.25 * (1.0 - xi) * (1.0 - eta),
                    0.25 * (1.0 + xi) * (1.0 - eta),
                    0.25 * (1.0 + xi) * (1.0 + eta),
                    0.25 * (1.0 - xi) * (1.0 + eta),
                ];
                let basis_values = panel_basis_values(elem_type, basis_order, xi, eta);
                let shape_ds = [
                    -0.25 * (1.0 - eta),
                    0.25 * (1.0 - eta),
                    0.25 * (1.0 + eta),
                    -0.25 * (1.0 + eta),
                ];
                let shape_dt = [
                    -0.25 * (1.0 - xi),
                    -0.25 * (1.0 + xi),
                    0.25 * (1.0 + xi),
                    0.25 * (1.0 - xi),
                ];

                let p_surf = basis_column.map_or_else(
                    || {
                        interpolate_discontinuous_data(
                            Some(surface_pressure),
                            dof_addresses,
                            &basis_values,
                        )
                    },
                    |local_basis| Complex64::new(basis_values[local_basis], 0.0),
                );
                let v_surf = if basis_column.is_some() {
                    Complex64::new(0.0, 0.0)
                } else {
                    interpolate_discontinuous_data(surface_velocity, dof_addresses, &basis_values)
                };

                result += integrate_quad_point(
                    x,
                    elem_coords,
                    &shape_fn,
                    &shape_ds,
                    &shape_dt,
                    weight,
                    p_surf,
                    v_surf,
                    wavruim,
                    4,
                    pressure_factor,
                    node_normals,
                );
            }
        }
    }

    result
}

fn interpolate_discontinuous_data(
    values: Option<&[Complex64]>,
    dof_addresses: &[usize],
    basis_values: &[f64],
) -> Complex64 {
    let Some(values) = values else {
        return Complex64::new(0.0, 0.0);
    };

    dof_addresses
        .iter()
        .zip(basis_values.iter())
        .fold(Complex64::new(0.0, 0.0), |sum, (&dof, &basis)| {
            sum + values[dof] * basis
        })
}

fn panel_basis_values(
    element_type: ElementType,
    basis_order: BasisOrder,
    s: f64,
    t: f64,
) -> Vec<f64> {
    basis_values(element_type, basis_order, s, t)
}

/// Helper function to integrate at a single quadrature point
///
/// When `node_normals` is `Some`, the field-element unit normal is replaced by
/// the interpolated (and re-normalized) normal from per-node normals. The
/// Jacobian (for dS) always comes from the flat cross-product.
#[inline]
fn integrate_quad_point(
    x: &[f64],
    elem_coords: &DenseMatrix<f64>,
    shape_fn: &[f64],
    shape_ds: &[f64],
    shape_dt: &[f64],
    weight: f64,
    p_surf: Complex64,
    v_surf: Complex64,
    wavruim: f64,
    _n_nodes: usize,
    pressure_factor: f64,
    node_normals: Option<&[[f64; 3]]>,
) -> Complex64 {
    let (crd_poi, jacobian, el_norm) =
        quadrature_geometry(elem_coords, shape_fn, shape_ds, shape_dt, node_normals);
    if jacobian < 1e-15 {
        return Complex64::new(0.0, 0.0);
    }

    // Distance from evaluation point to quadrature point
    let r_vec: [f64; 3] = [crd_poi[0] - x[0], crd_poi[1] - x[1], crd_poi[2] - x[2]];
    let r = (r_vec[0] * r_vec[0] + r_vec[1] * r_vec[1] + r_vec[2] * r_vec[2]).sqrt();

    if r < 1e-15 {
        return Complex64::new(0.0, 0.0);
    }

    // Weight factor
    let vjacwe = jacobian * weight;

    // Green's function: G = exp(ikr) / (4πr)
    let kr = wavruim * r;
    let re1 = 4.0 * PI * r;
    let zgrfu = sonium_fastmath::exp_i(kr) / re1;

    // Derivative factor: z1 = -1/r + ik
    let z1 = Complex64::new(-1.0 / r, wavruim);
    let zgikr = zgrfu * z1;

    // ∂r/∂n_y = (y-x)·n_y / r
    let drdn_y = (r_vec[0] * el_norm[0] + r_vec[1] * el_norm[1] + r_vec[2] * el_norm[2]) / r;

    // Double layer contribution: p * ∂G/∂n_y * dS
    let zdgrdn = zgikr * drdn_y;
    let mut contrib = p_surf * zdgrdn * vjacwe;

    // Single layer contribution: -v_n * iωρ₀ * G * dS
    // The Kirchhoff-Helmholtz integral uses the pressure_factor = ρ₀*ω*harmonic_factor
    if v_surf.norm() > 1e-15 {
        let iwrho = Complex64::new(0.0, pressure_factor);
        contrib -= v_surf * iwrho * zgrfu * vjacwe;
    }

    contrib
}

/// Integrate the analytic spatial derivative of one quadrature contribution.
#[inline]
fn integrate_quad_point_gradient(
    x: &[f64; 3],
    elem_coords: &DenseMatrix<f64>,
    shape_fn: &[f64],
    shape_ds: &[f64],
    shape_dt: &[f64],
    weight: f64,
    p_surf: Complex64,
    v_surf: Complex64,
    wavruim: f64,
    pressure_factor: f64,
    node_normals: Option<&[[f64; 3]]>,
) -> [Complex64; 3] {
    let (crd_poi, jacobian, el_norm) =
        quadrature_geometry(elem_coords, shape_fn, shape_ds, shape_dt, node_normals);
    if jacobian < 1.0e-15 {
        return [Complex64::new(0.0, 0.0); 3];
    }

    let r_vec = [crd_poi[0] - x[0], crd_poi[1] - x[1], crd_poi[2] - x[2]];
    let r = (r_vec[0] * r_vec[0] + r_vec[1] * r_vec[1] + r_vec[2] * r_vec[2]).sqrt();
    if r < 1.0e-15 {
        return [Complex64::new(0.0, 0.0); 3];
    }

    let green = sonium_fastmath::exp_i(wavruim * r) / (4.0 * PI * r);
    let z1 = Complex64::new(-1.0 / r, wavruim);
    let z2 = z1 * z1 + Complex64::new(1.0 / (r * r), 0.0);
    let unit = [r_vec[0] / r, r_vec[1] / r, r_vec[2] / r];
    let normal_projection = unit[0] * el_norm[0] + unit[1] * el_norm[1] + unit[2] * el_norm[2];
    let weighted_area = jacobian * weight;
    let iwrho = Complex64::new(0.0, pressure_factor);
    let mut gradient = [Complex64::new(0.0, 0.0); 3];

    for axis in 0..3 {
        // ∇_x (∂G/∂n_y), with n_y held fixed over this quadrature point.
        let normal_tangent = el_norm[axis] - normal_projection * unit[axis];
        let double_layer_gradient =
            -green * (z2 * normal_projection * unit[axis] + z1 * normal_tangent / r);
        // ∇_x G = -z1 G r̂, and the representation contains -iωρ v_n G.
        let single_layer_gradient = v_surf * iwrho * z1 * green * unit[axis];
        gradient[axis] = (p_surf * double_layer_gradient + single_layer_gradient) * weighted_area;
    }
    gradient
}

/// Compute total field (incident + scattered) at evaluation points
///
/// # Arguments
/// * `eval_points` - Evaluation points (N × 3 array)
/// * `elements` - Mesh elements
/// * `nodes` - Node coordinates
/// * `surface_pressure` - Solved surface pressure
/// * `incident_field` - Incident field specification
/// * `physics` - Physical parameters
/// * `basis_order` - Basis function order
///
/// # Returns
/// Vector of FieldPoint with all pressure components
pub fn compute_total_field(
    eval_points: &[[f64; 3]],
    elements: &[Element],
    nodes: &[[f64; 3]],
    surface_pressure: &[Complex64],
    surface_velocity: Option<&[Complex64]>,
    incident_field: &IncidentField,
    physics: &PhysicsParams,
    basis_order: BasisOrder,
) -> Vec<FieldPoint> {
    // Compute incident field
    let p_incident = incident_field.evaluate_pressure(eval_points, physics);

    // Compute scattered field
    let p_scattered = compute_scattered_field_with_incident(
        eval_points,
        elements,
        nodes,
        surface_pressure,
        surface_velocity,
        Some(incident_field),
        physics,
        basis_order,
    );

    // Combine results
    let n_eval = eval_points.len();
    let mut results = Vec::with_capacity(n_eval);

    for i in 0..n_eval {
        results.push(FieldPoint::new(
            eval_points[i],
            p_incident[i],
            p_scattered[i],
        ));
    }

    results
}

/// Generate evaluation points on a sphere around the origin
///
/// # Arguments
/// * `radius` - Sphere radius
/// * `n_theta` - Number of polar divisions
/// * `n_phi` - Number of azimuthal divisions
///
/// # Returns
/// Array of points on the sphere (N × 3)
pub fn generate_sphere_eval_points(radius: f64, n_theta: usize, n_phi: usize) -> DenseMatrix<f64> {
    let mut points = Vec::new();

    for i in 0..n_theta {
        let theta = PI * (i as f64 + 0.5) / n_theta as f64;
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        for j in 0..n_phi {
            let phi = 2.0 * PI * j as f64 / n_phi as f64;

            points.push(radius * sin_theta * phi.cos());
            points.push(radius * sin_theta * phi.sin());
            points.push(radius * cos_theta);
        }
    }

    let n_points = n_theta * n_phi;
    DenseMatrix::from_row_major_vec(n_points, 3, points)
}

/// Generate evaluation points along a line
///
/// # Arguments
/// * `start` - Start point
/// * `end` - End point
/// * `n_points` - Number of points
///
/// # Returns
/// Array of points along the line (N × 3)
pub fn generate_line_eval_points(
    start: [f64; 3],
    end: [f64; 3],
    n_points: usize,
) -> DenseMatrix<f64> {
    let mut points = Vec::new();

    for i in 0..n_points {
        let t = i as f64 / (n_points - 1).max(1) as f64;
        points.push(start[0] + t * (end[0] - start[0]));
        points.push(start[1] + t * (end[1] - start[1]));
        points.push(start[2] + t * (end[2] - start[2]));
    }

    DenseMatrix::from_row_major_vec(n_points, 3, points)
}

/// Generate evaluation points on a plane (for field maps)
///
/// # Arguments
/// * `center` - Center of the plane
/// * `normal` - Normal to the plane (determines orientation)
/// * `extent` - Half-size of the plane in each direction
/// * `n_points` - Number of points in each direction
///
/// # Returns
/// Array of points on the plane (N² × 3)
pub fn generate_plane_eval_points(
    center: [f64; 3],
    normal: [f64; 3],
    extent: f64,
    n_points: usize,
) -> DenseMatrix<f64> {
    // Find two vectors perpendicular to normal
    let n_norm = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    let n = [normal[0] / n_norm, normal[1] / n_norm, normal[2] / n_norm];

    // Choose an arbitrary vector not parallel to n
    let arbitrary: [f64; 3] = if n[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };

    // First basis vector (perpendicular to n)
    let u = {
        let cross = [
            n[1] * arbitrary[2] - n[2] * arbitrary[1],
            n[2] * arbitrary[0] - n[0] * arbitrary[2],
            n[0] * arbitrary[1] - n[1] * arbitrary[0],
        ];
        let norm = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
        [cross[0] / norm, cross[1] / norm, cross[2] / norm]
    };

    // Second basis vector (perpendicular to both n and u)
    let v = [
        n[1] * u[2] - n[2] * u[1],
        n[2] * u[0] - n[0] * u[2],
        n[0] * u[1] - n[1] * u[0],
    ];

    let mut points = Vec::new();

    for i in 0..n_points {
        let s = -extent + 2.0 * extent * i as f64 / (n_points - 1).max(1) as f64;
        for j in 0..n_points {
            let t = -extent + 2.0 * extent * j as f64 / (n_points - 1).max(1) as f64;

            points.push(center[0] + s * u[0] + t * v[0]);
            points.push(center[1] + s * u[1] + t * v[1]);
            points.push(center[2] + s * u[2] + t * v[2]);
        }
    }

    DenseMatrix::from_row_major_vec(n_points * n_points, 3, points)
}

/// Compute RCS (Radar Cross Section) from far-field pattern
///
/// For acoustic scattering, the RCS is defined as:
/// σ = lim_{r→∞} 4πr² |p_scat/p_inc|²
///
/// # Arguments
/// * `surface_pressure` - Solved surface pressure
/// * `elements` - Mesh elements
/// * `direction` - Far-field direction (unit vector)
/// * `physics` - Physical parameters
///
/// # Returns
/// RCS value
pub fn compute_rcs(
    surface_pressure: &[Complex64],
    elements: &[Element],
    direction: [f64; 3],
    physics: &PhysicsParams,
) -> f64 {
    let k = physics.wave_number;

    // Far-field pattern is computed using the stationary phase approximation
    // For constant elements: F(θ,φ) = Σ p_j * exp(-ik r_j · d) * A_j * (n_j · d)

    let mut far_field = Complex64::new(0.0, 0.0);

    let boundary_elements: Vec<_> = elements
        .iter()
        .filter(|e| !e.property.is_evaluation())
        .collect();

    let d: [f64; 3] = [direction[0], direction[1], direction[2]];

    for (j, elem) in boundary_elements.iter().enumerate() {
        let center = &elem.center;
        let normal = &elem.normal;
        let area = elem.area;

        // Phase term: exp(-ik r · d)
        let phase = -k * (center[0] * d[0] + center[1] * d[1] + center[2] * d[2]);
        let exp_phase = sonium_fastmath::exp_i(phase);

        // Normal factor: n · d (double layer contribution)
        let n_dot_d = normal[0] * d[0] + normal[1] * d[1] + normal[2] * d[2];

        // Contribution
        let ik = Complex64::new(0.0, k);
        far_field += surface_pressure[j] * exp_phase * area * ik * n_dot_d;
    }

    // The physical scattering amplitude is
    // f(d) = far_field / (4*pi).  For unit incident pressure,
    // sigma = 4*pi*|f|^2 = |far_field|^2/(4*pi).
    far_field.norm_sqr() / (4.0 * PI)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{BoundaryCondition, ElementProperty};

    fn make_physics(k: f64) -> PhysicsParams {
        let c = 343.0;
        let freq = k * c / (2.0 * PI);
        PhysicsParams::new(freq, c, 1.21, false)
    }

    fn test_triangle_element(dof_addresses: Vec<usize>) -> Element {
        Element {
            connectivity: vec![0, 1, 2],
            element_type: ElementType::Tri3,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0; 3]; 3],
            center: [0.5, 1.0 / 3.0, 0.0],
            area: 0.5,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
            group: 0,
            dof_addresses,
        }
    }

    #[test]
    fn dual_basis_mapping_recovers_constant_quadratic_trace() {
        let elements = vec![test_triangle_element((0..6).collect())];
        let nodes = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let dual = [
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0 / 6.0, 0.0),
            Complex64::new(1.0 / 6.0, 0.0),
            Complex64::new(1.0 / 6.0, 0.0),
        ];
        let mapped = map_dual_basis_coefficients_checked(
            &dual,
            &elements,
            &nodes,
            &make_physics(1.0),
            BasisOrder::Quadratic,
        )
        .expect("quadratic surface mass map");
        assert_eq!(mapped.len(), 6);
        for value in mapped {
            assert!((value - Complex64::new(1.0, 0.0)).norm() < 1.0e-12);
        }
    }

    #[test]
    fn test_field_point_creation() {
        let p_inc = Complex64::new(1.0, 0.0);
        let p_scat = Complex64::new(0.5, 0.3);

        let fp = FieldPoint::new([1.0, 0.0, 0.0], p_inc, p_scat);

        assert!((fp.p_total - (p_inc + p_scat)).norm() < 1e-10);
        assert!(fp.magnitude() > 0.0);
    }

    #[test]
    fn test_generate_sphere_eval_points() {
        let points = generate_sphere_eval_points(2.0, 10, 20);

        assert_eq!(points.nrows(), 200);
        assert_eq!(points.ncols(), 3);

        // Check all points are at radius 2.0
        for i in 0..points.nrows() {
            let r =
                (points[[i, 0]].powi(2) + points[[i, 1]].powi(2) + points[[i, 2]].powi(2)).sqrt();
            assert!((r - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_generate_line_eval_points() {
        let points = generate_line_eval_points([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 11);

        assert_eq!(points.nrows(), 11);

        // First point should be at origin
        assert!(points[[0, 0]].abs() < 1e-10);

        // Last point should be at (1, 0, 0)
        assert!((points[[10, 0]] - 1.0).abs() < 1e-10);

        // Point spacing should be uniform
        let spacing = points[[1, 0]] - points[[0, 0]];
        assert!((spacing - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_generate_plane_eval_points() {
        let points = generate_plane_eval_points([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.0, 5);

        assert_eq!(points.nrows(), 25);
        assert_eq!(points.ncols(), 3);

        // All points should have z = 0 (on the xy plane)
        for i in 0..points.nrows() {
            assert!(points[[i, 2]].abs() < 1e-10);
        }
    }

    #[test]
    fn rcs_includes_free_space_far_field_normalization() {
        use crate::core::types::BoundaryCondition;

        let physics = make_physics(2.0);
        let element = Element {
            connectivity: vec![0, 1, 2],
            element_type: crate::core::types::ElementType::Tri3,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0; 3]; 3],
            center: [0.0, 0.0, 0.0],
            area: 0.5,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
            group: 0,
            dof_addresses: vec![0],
        };
        let rcs = compute_rcs(
            &[Complex64::new(1.0, 0.0)],
            &[element],
            [0.0, 0.0, 1.0],
            &physics,
        );
        assert!((rcs - 1.0 / (4.0 * PI)).abs() < 1.0e-14);
    }

    #[test]
    fn test_scattered_field_basic() {
        // Simple test with a single element
        use crate::core::types::BoundaryCondition;

        let physics = make_physics(1.0);

        // Single triangular element
        let nodes: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]];

        let elem = Element {
            connectivity: vec![0, 1, 2],
            element_type: crate::core::types::ElementType::Tri3,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0; 3]; 3],
            center: [0.5, 1.0 / 3.0, 0.0],
            area: 0.5,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
            group: 0,
            dof_addresses: vec![0],
        };

        let elements = vec![elem];
        let surface_pressure: Vec<Complex64> = vec![Complex64::new(1.0, 0.0)];

        // Evaluation point above the element
        let eval_points: Vec<[f64; 3]> = vec![[0.5, 0.5, 1.0]];

        let p_scat = compute_scattered_field(
            &eval_points,
            &elements,
            &nodes,
            &surface_pressure,
            None,
            &physics,
            BasisOrder::Constant,
        );

        // Scattered field should be non-zero
        assert!(p_scat[0].norm() > 0.0);
    }

    #[test]
    fn checked_field_matches_legacy_evaluation() {
        let physics = make_physics(1.0);
        let nodes = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]];
        let elements = vec![test_triangle_element(vec![0])];
        let eval_points = vec![[0.5, 0.5, 1.0], [0.1, 0.2, 0.8]];
        let pressure = vec![Complex64::new(1.0, -0.25)];

        let legacy = compute_scattered_field(
            &eval_points,
            &elements,
            &nodes,
            &pressure,
            None,
            &physics,
            BasisOrder::Constant,
        );
        let checked = compute_scattered_field_checked(
            &eval_points,
            &elements,
            &nodes,
            &pressure,
            None,
            &physics,
            BasisOrder::Constant,
        )
        .expect("valid field inputs");
        assert_eq!(checked, legacy);

        let incident = IncidentField::plane_wave_z();
        let legacy_total = compute_total_field(
            &eval_points,
            &elements,
            &nodes,
            &pressure,
            None,
            &incident,
            &physics,
            BasisOrder::Constant,
        );
        let checked_total = compute_total_field_checked(
            &eval_points,
            &elements,
            &nodes,
            &pressure,
            None,
            &incident,
            &physics,
            BasisOrder::Constant,
        )
        .expect("valid total-field inputs");
        assert_eq!(checked_total.len(), legacy_total.len());
        for (checked, legacy) in checked_total.iter().zip(legacy_total.iter()) {
            assert_eq!(checked.position, legacy.position);
            assert_eq!(checked.p_incident, legacy.p_incident);
            assert_eq!(checked.p_scattered, legacy.p_scattered);
            assert_eq!(checked.p_total, legacy.p_total);
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn checked_cuda_field_matches_cpu_reference_for_constant_triangles() {
        let backend = match CudaBackend::new_default() {
            Ok(backend) => backend,
            Err(error) => {
                if std::env::var("SONIUM_REQUIRE_CUDA")
                    .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE"))
                {
                    panic!("CUDA required by SONIUM_REQUIRE_CUDA=1: {error}");
                }
                eprintln!("CUDA unavailable, skipping checked field parity: {error}");
                return;
            }
        };
        let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 20))
            .expect("CUDA field memory pool");
        let physics = make_physics(2.3);
        let nodes = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.1],
            [0.1, 1.0, -0.2],
            [1.4, 0.2, 0.5],
            [1.8, 0.3, 0.6],
            [1.5, 1.1, 0.7],
        ];
        let mut first = test_triangle_element(vec![0]);
        first.connectivity = vec![0, 1, 2];
        let mut second = test_triangle_element(vec![1]);
        second.connectivity = vec![3, 4, 5];
        second.center = [1.5666666666666667, 0.5333333333333333, 0.6];
        second.normal = [0.0, -0.19611613513818404, 0.9805806756909201];
        second.area = 0.5099019513592785;
        let elements = vec![first, second];
        let eval_points = vec![[0.25, 0.35, 1.7], [1.2, -0.4, 1.9], [-0.3, 0.8, 2.1]];
        let pressure = vec![Complex64::new(1.2, -0.4), Complex64::new(-0.3, 0.8)];
        let velocity = vec![Complex64::new(0.1, 0.05), Complex64::new(-0.07, 0.02)];

        let cpu = compute_scattered_field_checked(
            &eval_points,
            &elements,
            &nodes,
            &pressure,
            Some(&velocity),
            &physics,
            BasisOrder::Constant,
        )
        .expect("CPU field reference");
        let cuda = compute_scattered_field_cuda_checked(
            &pool,
            &eval_points,
            &elements,
            &nodes,
            &pressure,
            Some(&velocity),
            &physics,
            BasisOrder::Constant,
        )
        .expect("native CUDA field evaluation");

        assert_eq!(cpu.len(), cuda.len());
        for (index, (expected, actual)) in cpu.iter().zip(cuda.iter()).enumerate() {
            assert!(
                (*actual - *expected).norm() <= 3.0e-12 * expected.norm().max(1.0),
                "CUDA field value {index} differs: cpu={expected:?}, cuda={actual:?}"
            );
        }
        let stats = pool.backend().runtime_stats();
        assert!(
            stats.kernel_launches >= 1,
            "field must launch a CUDA kernel"
        );
        assert!(
            stats.transfer.h2d_operations >= 2,
            "field must upload traces and points"
        );
        assert!(
            stats.transfer.d2h_operations >= 1,
            "field must publish explicit readback"
        );
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn checked_cuda_field_rejects_quad4_and_on_surface_cells_explicitly() {
        let backend = match CudaBackend::new_default() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("CUDA unavailable, skipping CUDA admission test: {error}");
                return;
            }
        };
        let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 20))
            .expect("CUDA field memory pool");
        let physics = make_physics(1.0);
        let nodes = vec![
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ];
        let quad = Element {
            connectivity: vec![0, 1, 2, 3],
            element_type: ElementType::Quad4,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0, 0.0, 1.0]; 4],
            center: [0.0, 0.0, 0.0],
            area: 4.0,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::default()]),
            group: 0,
            dof_addresses: vec![0],
        };
        let error = compute_scattered_field_cuda_checked(
            &pool,
            &[[0.0, 0.0, 2.0]],
            std::slice::from_ref(&quad),
            &nodes,
            &[Complex64::new(1.0, 0.0)],
            None,
            &physics,
            BasisOrder::Constant,
        )
        .expect_err("Quad4 CUDA field support must remain explicit");
        assert!(
            matches!(error, FieldEvaluationError::CudaFailure { reason } if reason.contains("Tri3 only"))
        );

        let triangle = test_triangle_element(vec![0]);
        let error = compute_scattered_field_cuda_checked(
            &pool,
            &[[0.25, 0.25, 0.0]],
            std::slice::from_ref(&triangle),
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            &[Complex64::new(1.0, 0.0)],
            None,
            &physics,
            BasisOrder::Constant,
        )
        .expect_err("on-surface CUDA field evaluation must be explicit");
        assert!(
            matches!(error, FieldEvaluationError::CudaFailure { reason } if reason.contains("off-surface"))
        );
    }

    #[test]
    fn checked_gradient_consumers_preserve_result_cardinality() {
        let physics = make_physics(1.0);
        let nodes = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]];
        let elements = vec![test_triangle_element(vec![0])];
        let eval_points = vec![[0.5, 0.5, 1.0], [0.1, 0.2, 0.8]];
        let pressure = vec![Complex64::new(1.0, -0.25)];
        let incident = IncidentField::plane_wave_z();

        let scattered = compute_scattered_field_gradient_checked(
            &eval_points,
            &elements,
            &nodes,
            &pressure,
            None,
            &physics,
            BasisOrder::Constant,
        )
        .expect("valid scattered gradient inputs");
        assert_eq!(scattered.len(), eval_points.len());

        let total = compute_total_field_gradient_checked(
            &eval_points,
            &elements,
            &nodes,
            &pressure,
            None,
            &incident,
            &physics,
            BasisOrder::Constant,
        )
        .expect("valid total gradient inputs");
        assert_eq!(total.len(), eval_points.len());
    }

    #[test]
    fn checked_field_rejects_malformed_inputs() {
        let physics = make_physics(1.0);
        let nodes = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]];
        let eval_points = vec![[0.5, 0.5, 1.0]];
        let pressure = vec![Complex64::new(1.0, 0.0)];

        assert!(matches!(
            compute_scattered_field_checked(
                &eval_points,
                &[],
                &nodes,
                &[],
                None,
                &physics,
                BasisOrder::Constant,
            ),
            Err(FieldEvaluationError::EmptyBoundary)
        ));

        let mut malformed = test_triangle_element(vec![0]);
        malformed.connectivity.pop();
        assert!(matches!(
            compute_scattered_field_checked(
                &eval_points,
                &[malformed],
                &nodes,
                &pressure,
                None,
                &physics,
                BasisOrder::Constant,
            ),
            Err(FieldEvaluationError::InvalidConnectivity { .. })
        ));

        let mut invalid_node = test_triangle_element(vec![0]);
        invalid_node.connectivity[2] = nodes.len();
        assert!(matches!(
            compute_scattered_field_checked(
                &eval_points,
                &[invalid_node],
                &nodes,
                &pressure,
                None,
                &physics,
                BasisOrder::Constant,
            ),
            Err(FieldEvaluationError::InvalidNodeIndex { .. })
        ));

        let mut nonfinite_node = nodes.clone();
        nonfinite_node[1][0] = f64::NAN;
        assert!(matches!(
            compute_scattered_field_checked(
                &eval_points,
                &[test_triangle_element(vec![0])],
                &nonfinite_node,
                &pressure,
                None,
                &physics,
                BasisOrder::Constant,
            ),
            Err(FieldEvaluationError::NonFiniteNode { node: 1 })
        ));

        let nonfinite_eval = [[f64::INFINITY, 0.0, 1.0]];
        assert!(matches!(
            compute_scattered_field_checked(
                &nonfinite_eval,
                &[test_triangle_element(vec![0])],
                &nodes,
                &pressure,
                None,
                &physics,
                BasisOrder::Constant,
            ),
            Err(FieldEvaluationError::NonFiniteEvaluationPoint { point: 0 })
        ));

        assert!(matches!(
            compute_scattered_field_checked(
                &eval_points,
                &[test_triangle_element(vec![0])],
                &nodes,
                &[],
                None,
                &physics,
                BasisOrder::Constant,
            ),
            Err(FieldEvaluationError::SurfacePressureCount {
                expected: 1,
                actual: 0,
            })
        ));
        assert!(matches!(
            compute_scattered_field_checked(
                &eval_points,
                &[test_triangle_element(vec![0])],
                &nodes,
                &pressure,
                Some(&[]),
                &physics,
                BasisOrder::Constant,
            ),
            Err(FieldEvaluationError::SurfaceVelocityCount {
                expected: 1,
                actual: 0,
            })
        ));

        let nonfinite_pressure = [Complex64::new(f64::NAN, 0.0)];
        assert!(matches!(
            compute_scattered_field_checked(
                &eval_points,
                &[test_triangle_element(vec![0])],
                &nodes,
                &nonfinite_pressure,
                None,
                &physics,
                BasisOrder::Constant,
            ),
            Err(FieldEvaluationError::NonFiniteSurfaceSample { index: 0 })
        ));
        let nonfinite_velocity = [Complex64::new(0.0, f64::INFINITY)];
        assert!(matches!(
            compute_scattered_field_checked(
                &eval_points,
                &[test_triangle_element(vec![0])],
                &nodes,
                &pressure,
                Some(&nonfinite_velocity),
                &physics,
                BasisOrder::Constant,
            ),
            Err(FieldEvaluationError::NonFiniteSurfaceSample { index: 0 })
        ));

        let mut invalid_physics = physics.clone();
        invalid_physics.frequency = f64::NAN;
        assert!(matches!(
            compute_scattered_field_checked(
                &eval_points,
                &[test_triangle_element(vec![0])],
                &nodes,
                &pressure,
                None,
                &invalid_physics,
                BasisOrder::Constant,
            ),
            Err(FieldEvaluationError::InvalidPhysics { .. })
        ));

        let linear = test_triangle_element(vec![0]);
        assert!(matches!(
            compute_scattered_field_checked(
                &eval_points,
                &[linear],
                &nodes,
                &pressure,
                None,
                &physics,
                BasisOrder::Linear,
            ),
            Err(FieldEvaluationError::InvalidDofLayout { .. })
        ));
        let linear = test_triangle_element(vec![0, 1, 2]);
        assert!(matches!(
            compute_scattered_field_checked(
                &eval_points,
                &[linear],
                &nodes,
                &[Complex64::new(1.0, 0.0)],
                None,
                &physics,
                BasisOrder::Linear,
            ),
            Err(FieldEvaluationError::SurfacePressureCount {
                expected: 3,
                actual: 1,
            })
        ));
    }

    #[test]
    fn checked_field_result_count_contract_rejects_short_kernel_outputs() {
        assert_eq!(
            validate_result_count(2, 1),
            Err(FieldEvaluationError::ResultCount {
                expected: 2,
                actual: 1,
            })
        );
        assert!(validate_result_count(2, 2).is_ok());
    }

    #[test]
    fn checked_total_field_rejects_nonfinite_incident_result() {
        let physics = make_physics(1.0);
        let nodes = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]];
        let elements = vec![test_triangle_element(vec![0])];
        let incident = IncidentField::PlaneWave {
            direction: [f64::NAN, 0.0, 1.0],
            amplitude: Complex64::new(1.0, 0.0),
        };
        assert!(matches!(
            compute_total_field_checked(
                &[[0.5, 0.5, 1.0]],
                &elements,
                &nodes,
                &[Complex64::new(1.0, 0.0)],
                None,
                &incident,
                &physics,
                BasisOrder::Constant,
            ),
            Err(FieldEvaluationError::NonFiniteResult { point: 0 })
        ));
    }

    #[test]
    fn constant_field_operator_matches_direct_evaluation() {
        use crate::core::types::BoundaryCondition;

        let physics = make_physics(2.0);
        let nodes = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]];
        let elements = vec![Element {
            connectivity: vec![0, 1, 2],
            element_type: crate::core::types::ElementType::Tri3,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0; 3]; 3],
            center: [0.5, 1.0 / 3.0, 0.0],
            area: 0.5,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
            group: 0,
            dof_addresses: vec![0],
        }];
        let eval_points = vec![[0.5, 0.5, 1.0], [0.1, 0.2, 0.8]];
        let surface_pressure = vec![Complex64::new(0.7, -0.2)];
        let surface_velocity = vec![Complex64::new(0.01, 0.005)];
        let direct = compute_scattered_field(
            &eval_points,
            &elements,
            &nodes,
            &surface_pressure,
            Some(&surface_velocity),
            &physics,
            BasisOrder::Constant,
        );
        let (operator, affine) = build_constant_field_operator(
            &eval_points,
            &elements,
            &nodes,
            Some(&surface_velocity),
            &physics,
        );
        for row in 0..eval_points.len() {
            let reconstructed = operator[[row, 0]] * surface_pressure[0] + affine[row];
            assert!(
                (reconstructed - direct[row]).norm() < 1.0e-12,
                "row {row}: reconstructed={reconstructed}, direct={}",
                direct[row]
            );
        }
    }

    #[test]
    fn constant_field_operator_columns_match_full_operator() {
        use crate::core::types::BoundaryCondition;

        let physics = make_physics(2.0);
        let nodes = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [1.5, 1.0, 0.0],
        ];
        let make_element =
            |connectivity: Vec<usize>, center: [f64; 3], dof: usize, v: Complex64| Element {
                connectivity,
                element_type: crate::core::types::ElementType::Tri3,
                property: ElementProperty::Surface,
                normal: [0.0, 0.0, 1.0],
                node_normals: vec![[0.0; 3]; 3],
                center,
                area: 0.5,
                boundary_condition: BoundaryCondition::Velocity(vec![v]),
                group: 0,
                dof_addresses: vec![dof],
            };
        let elements = vec![
            make_element(
                vec![0, 1, 2],
                [0.5, 1.0 / 3.0, 0.0],
                0,
                Complex64::new(0.01, 0.005),
            ),
            make_element(
                vec![1, 3, 2],
                [1.0, 2.0 / 3.0, 0.0],
                1,
                Complex64::new(-0.02, 0.0),
            ),
        ];
        let eval_points = vec![[0.5, 0.5, 1.0], [0.1, 0.2, 0.8], [1.0, 0.5, 2.0]];
        let surface_velocity = vec![Complex64::new(0.01, 0.005), Complex64::new(-0.02, 0.0)];

        let (operator, affine) = build_constant_field_operator(
            &eval_points,
            &elements,
            &nodes,
            Some(&surface_velocity),
            &physics,
        );
        let (column_values, column_affine) = eval_constant_field_operator_columns(
            &eval_points,
            &elements,
            &nodes,
            Some(&surface_velocity),
            &physics,
            &[1, 0],
        );
        assert_eq!(column_values.len(), 2);
        for row in 0..eval_points.len() {
            // Requested order [1, 0] must map back to operator columns 1 and 0.
            assert!(
                (column_values[0][row] - operator[[row, 1]]).norm() < 1.0e-14,
                "column 1 mismatch at row {row}"
            );
            assert!(
                (column_values[1][row] - operator[[row, 0]]).norm() < 1.0e-14,
                "column 0 mismatch at row {row}"
            );
            // Both columns requested: the affine contributions sum to the full affine term.
            assert!(
                (column_affine[row] - affine[row]).norm() < 1.0e-14,
                "affine mismatch at row {row}"
            );
        }

        // A single-column request must isolate that column's affine share.
        let (_, affine_one) = eval_constant_field_operator_columns(
            &eval_points,
            &elements,
            &nodes,
            Some(&surface_velocity),
            &physics,
            &[0],
        );
        let (_, affine_other) = eval_constant_field_operator_columns(
            &eval_points,
            &elements,
            &nodes,
            Some(&surface_velocity),
            &physics,
            &[1],
        );
        for row in 0..eval_points.len() {
            assert!(
                (affine_one[row] + affine_other[row] - affine[row]).norm() < 1.0e-14,
                "affine split mismatch at row {row}"
            );
        }
    }

    #[test]
    fn higher_order_field_operator_matches_checked_evaluation() {
        use crate::core::types::BoundaryCondition;

        let physics = make_physics(2.0);
        let nodes = vec![
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ];
        let elements = vec![Element {
            connectivity: vec![0, 1, 2, 3],
            element_type: ElementType::Quad4,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0, 0.0, 1.0]; 4],
            center: [0.0, 0.0, 0.0],
            area: 4.0,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
            group: 0,
            dof_addresses: (0..9).collect(),
        }];
        let eval_points = [[0.23, -0.17, 2.0], [-0.41, 0.29, 1.7]];
        let surface_pressure: Vec<_> = (0..9)
            .map(|index| Complex64::new(0.1 * (index as f64 + 1.0), -0.03 * index as f64))
            .collect();
        let surface_velocity: Vec<_> = (0..9)
            .map(|index| Complex64::new(-0.02 * index as f64, 0.01 * (index as f64 + 1.0)))
            .collect();

        let direct = compute_scattered_field_checked(
            &eval_points,
            &elements,
            &nodes,
            &surface_pressure,
            Some(&surface_velocity),
            &physics,
            BasisOrder::Quadratic,
        )
        .expect("valid quadratic field evaluation");
        let (operator, affine) = build_field_operator_checked(
            &eval_points,
            &elements,
            &nodes,
            Some(&surface_velocity),
            &physics,
            BasisOrder::Quadratic,
        )
        .expect("valid quadratic field operator");

        for row in 0..eval_points.len() {
            let reconstructed = (0..surface_pressure.len())
                .map(|column| operator[[row, column]] * surface_pressure[column])
                .sum::<Complex64>()
                + affine[row];
            assert!(
                (reconstructed - direct[row]).norm() < 1.0e-12,
                "higher-order operator mismatch at row {row}: {reconstructed:?} vs {:?}",
                direct[row]
            );
        }
    }

    #[test]
    fn quadratic_field_interpolates_pressure_and_velocity_dofs() {
        let physics = make_physics(2.0);
        let nodes = vec![
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ];
        let element = Element {
            connectivity: vec![0, 1, 2, 3],
            element_type: ElementType::Quad4,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0, 0.0, 1.0]; 4],
            center: [0.0, 0.0, 0.0],
            area: 4.0,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
            group: 0,
            dof_addresses: (0..9).collect(),
        };
        let eval_points = [[0.23, -0.17, 2.0], [-0.41, 0.29, 1.7]];
        let zeros = vec![Complex64::new(0.0, 0.0); 9];
        let mut pressure = zeros.clone();
        pressure[8] = Complex64::new(1.0, 0.35);
        let mut velocity = zeros.clone();
        velocity[8] = Complex64::new(0.4, -0.2);

        let pressure_field = compute_scattered_field_checked(
            &eval_points,
            std::slice::from_ref(&element),
            &nodes,
            &pressure,
            Some(&zeros),
            &physics,
            BasisOrder::Quadratic,
        )
        .expect("quadratic pressure field must evaluate");
        let velocity_field = compute_scattered_field_checked(
            &eval_points,
            std::slice::from_ref(&element),
            &nodes,
            &zeros,
            Some(&velocity),
            &physics,
            BasisOrder::Quadratic,
        )
        .expect("quadratic velocity field must evaluate");

        assert!(pressure_field.iter().any(|value| value.norm() > 1.0e-12));
        assert!(velocity_field.iter().any(|value| value.norm() > 1.0e-12));

        assert!(matches!(
            compute_scattered_field_checked(
                &eval_points,
                std::slice::from_ref(&element),
                &nodes,
                &pressure,
                Some(&zeros[..1]),
                &physics,
                BasisOrder::Quadratic,
            ),
            Err(FieldEvaluationError::SurfaceVelocityCount {
                expected: 9,
                actual: 1,
            })
        ));
    }
}
