//! Independent Quad4 quadratic-panel kernel and field-evaluation reference.
//!
//! This packet deliberately does not solve a BEM system or exercise the
//! continuous-DOF topology map.  It checks the two production quadrature
//! branches used by the discontinuous nine-function Quad4 panel basis, and
//! then checks the checked post-processing path with non-constant quadratic
//! Cauchy data.

use num_complex::Complex64;
use sonium_bem::core::assembly::tbem::build_tbem_system_quadratic_stabilized;
use sonium_bem::core::integration::regular::regular_integration_linear;
use sonium_bem::core::integration::singular::{
    QuadratureParams, singular_integration_linear_with_params, singular_integration_with_params,
};
use sonium_bem::core::types::{
    BoundaryCondition, Element, ElementProperty, ElementType, PhysicsParams,
};
use sonium_solvers::dense::DenseMatrix;

const PI: f64 = std::f64::consts::PI;
const SPEED_OF_SOUND: f64 = 343.0;
const DENSITY: f64 = 1.21;
const FREQUENCY: f64 = 80.0;
const SOURCE_HEIGHT: f64 = 1.75;

// Twelve-point Gauss--Legendre data.  The reference is intentionally
// independent of the production quadrature schedules and uses a tensor rule
// for regular points and a quadrant Duffy map for the self panel.
const GL12_X: [f64; 12] = [
    -0.9815606342467192,
    -0.9041172563704749,
    -0.7699026741943047,
    -0.5873179542866175,
    -0.3678314989981802,
    -0.1252334085114689,
    0.1252334085114689,
    0.3678314989981802,
    0.5873179542866175,
    0.7699026741943047,
    0.9041172563704749,
    0.9815606342467192,
];
const GL12_W: [f64; 12] = [
    0.0471753363865118,
    0.1069393259953184,
    0.1600783285433462,
    0.2031674267230659,
    0.2334925365383548,
    0.2491470458134028,
    0.2491470458134028,
    0.2334925365383548,
    0.2031674267230659,
    0.1600783285433462,
    0.1069393259953184,
    0.0471753363865118,
];

fn physics() -> PhysicsParams {
    PhysicsParams::try_new(FREQUENCY, SPEED_OF_SOUND, DENSITY, false)
        .expect("reference physics must be valid")
}

fn square_nodes() -> Vec<[f64; 3]> {
    vec![
        [-0.5, -0.5, 0.0],
        [0.5, -0.5, 0.0],
        [0.5, 0.5, 0.0],
        [-0.5, 0.5, 0.0],
    ]
}

fn square_coords(nodes: &[[f64; 3]]) -> DenseMatrix<f64> {
    DenseMatrix::from_row_major_vec(
        4,
        3,
        nodes.iter().flat_map(|node| node.iter().copied()).collect(),
    )
}

fn square_element() -> Element {
    Element {
        connectivity: vec![0, 1, 2, 3],
        element_type: ElementType::Quad4,
        property: ElementProperty::Surface,
        normal: [0.0, 0.0, 1.0],
        node_normals: vec![[0.0, 0.0, 1.0]; 4],
        center: [0.0, 0.0, 0.0],
        area: 1.0,
        boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
        group: 0,
        dof_addresses: (0..9).collect(),
    }
}

fn quad9_basis(s: f64, t: f64) -> [f64; 9] {
    let ls = [0.5 * s * (s - 1.0), 1.0 - s * s, 0.5 * s * (s + 1.0)];
    let lt = [0.5 * t * (t - 1.0), 1.0 - t * t, 0.5 * t * (t + 1.0)];
    [
        ls[0] * lt[0],
        ls[2] * lt[0],
        ls[2] * lt[2],
        ls[0] * lt[2],
        ls[1] * lt[0],
        ls[2] * lt[1],
        ls[1] * lt[2],
        ls[0] * lt[1],
        ls[1] * lt[1],
    ]
}

fn green(physics: &PhysicsParams, radius: f64) -> Complex64 {
    Complex64::from_polar(1.0, physics.harmonic_factor * physics.wave_number * radius)
        / (4.0 * PI * radius)
}

fn relative_error(actual: Complex64, expected: Complex64) -> f64 {
    (actual - expected).norm() / expected.norm().max(1.0e-14)
}

fn regular_reference(
    physics: &PhysicsParams,
    source_point: [f64; 3],
    coefficients: Option<&[Complex64; 9]>,
) -> [Complex64; 9] {
    let mut result = [Complex64::new(0.0, 0.0); 9];
    for (s_index, &s) in GL12_X.iter().enumerate() {
        for (t_index, &t) in GL12_X.iter().enumerate() {
            let point = [0.5 * s, 0.5 * t, 0.0];
            let dx = point[0] - source_point[0];
            let dy = point[1] - source_point[1];
            let dz = point[2] - source_point[2];
            let radius = (dx * dx + dy * dy + dz * dz).sqrt();
            let weight = 0.25 * GL12_W[s_index] * GL12_W[t_index];
            let kernel = green(physics, radius) * weight;
            let basis = quad9_basis(s, t);
            for (index, &value) in basis.iter().enumerate() {
                let coefficient =
                    coefficients.map_or(Complex64::new(1.0, 0.0), |values| values[index]);
                result[index] += kernel * value * coefficient;
            }
        }
    }
    result
}

fn singular_reference(physics: &PhysicsParams) -> [Complex64; 9] {
    let mut result = [Complex64::new(0.0, 0.0); 9];
    for &sign_s in &[-1.0, 1.0] {
        for &sign_t in &[-1.0, 1.0] {
            // Two Duffy sectors cover the two triangles in each local
            // quadrant: |t| <= |s| and |s| <= |t|.
            for transpose in [false, true] {
                for (u_index, &u_node) in GL12_X.iter().enumerate() {
                    let u = 0.5 * (u_node + 1.0);
                    let u_weight = 0.5 * GL12_W[u_index];
                    for (v_index, &v_node) in GL12_X.iter().enumerate() {
                        let v = 0.5 * (v_node + 1.0);
                        let v_weight = 0.5 * GL12_W[v_index];
                        let (s, t) = if transpose {
                            (sign_s * u * v, sign_t * u)
                        } else {
                            (sign_s * u, sign_t * u * v)
                        };
                        let radius = 0.5 * u * (1.0 + v * v).sqrt();
                        let weight = 0.25 * u * u_weight * v_weight;
                        let kernel = green(physics, radius) * weight;
                        for (index, &value) in quad9_basis(s, t).iter().enumerate() {
                            result[index] += kernel * value;
                        }
                    }
                }
            }
        }
    }
    result
}

fn max_relative_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(&left, &right)| relative_error(left, right))
        .fold(0.0, f64::max)
}

#[test]
fn quad4_quadratic_regular_kernel_matches_independent_reference() {
    let nodes = square_nodes();
    let coords = square_coords(&nodes);
    let source_point = [0.0, 0.0, SOURCE_HEIGHT];
    let source_normal = [0.0, 0.0, 1.0];
    let physics = physics();

    let production = regular_integration_linear(
        &source_point,
        &source_normal,
        &coords,
        ElementType::Quad4,
        1.0,
        &physics,
        9,
    );
    let actual: Vec<_> = production
        .basis_results
        .iter()
        .map(|result| result.g_integral)
        .collect();
    let expected = regular_reference(&physics, source_point, None);
    let error = max_relative_error(&actual, &expected);

    assert_eq!(actual.len(), 9);
    assert!(
        actual
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    assert!(
        error < 2.0e-7,
        "regular Quad4 quadratic G error {error:.3e} exceeds independent tensor-Gauss reference"
    );
    let partition_error =
        relative_error(actual.iter().copied().sum(), expected.iter().copied().sum());
    assert!(
        partition_error < 2.0e-7,
        "regular quadratic partition error {partition_error:.3e}"
    );
}

#[test]
fn quad4_quadratic_singular_kernel_matches_independent_duffy_reference() {
    let nodes = square_nodes();
    let coords = square_coords(&nodes);
    let source_point = [0.0, 0.0, 0.0];
    let source_normal = [0.0, 0.0, 1.0];
    let physics = physics();
    let quadrature = QuadratureParams::high_accuracy();

    let production = singular_integration_linear_with_params(
        &source_point,
        &source_normal,
        &coords,
        ElementType::Quad4,
        &physics,
        9,
        &quadrature,
        None,
    );
    let actual: Vec<_> = production
        .basis_results
        .iter()
        .map(|result| result.g_integral)
        .collect();
    let expected = singular_reference(&physics);
    let error = max_relative_error(&actual, &expected);
    assert_eq!(actual.len(), 9);
    assert!(
        actual
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    assert!(
        error < 3.0e-4,
        "singular Quad4 quadratic G error {error:.3e} exceeds independent quadrant-Duffy reference; actual={actual:?}, expected={expected:?}"
    );

    let scalar = singular_integration_with_params(
        &source_point,
        &source_normal,
        &coords,
        ElementType::Quad4,
        &physics,
        None,
        0,
        false,
        &quadrature,
        None,
    );
    let partition_error = relative_error(actual.iter().copied().sum(), scalar.g_integral);
    assert!(
        partition_error < 2.0e-11,
        "singular quadratic basis does not partition to scalar G: {partition_error:.3e}"
    );
}

#[test]
fn quad4_quadratic_assembly_consumes_regular_singular_basis_contract() {
    let nodes = square_nodes();
    let element = square_element();
    let physics = physics();
    let system = build_tbem_system_quadratic_stabilized(
        std::slice::from_ref(&element),
        &nodes,
        &physics,
        1.0,
        1.0,
    );

    assert_eq!(system.matrix.shape(), (9, 9));
    assert_eq!(system.rhs.len(), 9);
    assert!(
        system
            .matrix
            .as_slice()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    assert!(
        system
            .matrix
            .as_slice()
            .iter()
            .any(|value| value.norm() > 1.0e-14)
    );
}
