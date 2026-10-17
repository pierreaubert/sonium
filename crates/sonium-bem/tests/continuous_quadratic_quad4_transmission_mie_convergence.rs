//! Independent Mie convergence for continuous quadratic Quad4 transmission.

// Rust guideline compliant 2026-02-21

use num_complex::Complex64;
use sonium_bem::BasisOrder;
use sonium_bem::core::assembly::assemble_galerkin_muller_transmission_with_global_dofs;
use sonium_bem::core::incident::IncidentField;
use sonium_bem::core::mesh::generate_cubed_sphere_quad_mesh;
use sonium_bem::core::types::PhysicsParams;
use sonium_wave::analytical::FluidSphereTransmission3d;
use std::f64::consts::PI;

const RADIUS: f64 = 0.2;
const FREQUENCY: f64 = 420.0;
const EXTERIOR_SPEED: f64 = 343.0;
const EXTERIOR_DENSITY: f64 = 1.21;
const INTERIOR_SPEED: f64 = 1_480.0;
const INTERIOR_DENSITY: f64 = 1_000.0;
const EVALUATION_RADIUS: f64 = 0.8;

#[derive(Debug, Clone, Copy)]
pub struct ContinuousQuadraticQuad4TransmissionMetrics {
    pub coarse_error: f64,
    /// Exterior complex-pressure error on the middle refinement level.
    pub medium_error: f64,
    /// Independently certified middle-level algebraic residual.
    pub medium_residual: f64,
    /// Middle-level numerical solution difference from direct factorization.
    pub medium_solution_error: f64,
    pub fine_error: f64,
    pub coarse_residual: f64,
    pub fine_residual: f64,
    pub coarse_solution_error: f64,
    pub fine_solution_error: f64,
    pub coarse_dofs: usize,
    pub fine_dofs: usize,
    pub coarse_panels: usize,
    pub fine_panels: usize,
}

fn media() -> (PhysicsParams, PhysicsParams) {
    (
        PhysicsParams::new(FREQUENCY, EXTERIOR_SPEED, EXTERIOR_DENSITY, false),
        PhysicsParams::new(FREQUENCY, INTERIOR_SPEED, INTERIOR_DENSITY, true),
    )
}

fn relative_error(actual: &[Complex64], reference: &[Complex64]) -> f64 {
    let numerator = actual
        .iter()
        .zip(reference)
        .map(|(value, expected)| (*value - *expected).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let denominator = reference
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
        .max(f64::MIN_POSITIVE);
    numerator / denominator
}

fn solve_level(face_resolution: usize, reference: &[Complex64]) -> (f64, f64, f64, usize, usize) {
    println!("Galerkin Quad4 resolution {face_resolution}: starting full native assembly");
    let mut mesh = generate_cubed_sphere_quad_mesh(RADIUS, face_resolution);
    let mapping = mesh
        .continuous_quadratic_dof_map()
        .expect("native Q2 topology");
    let dofs = mapping.num_dofs();
    for (index, element) in mesh.elements.iter_mut().enumerate() {
        element.dof_addresses = mapping.element_dofs(index).unwrap().to_vec();
    }
    let (exterior, interior) = media();
    let system = assemble_galerkin_muller_transmission_with_global_dofs(
        &mesh.elements,
        &mesh.nodes,
        &exterior,
        &interior,
        BasisOrder::Quadratic,
        3,
        dofs,
    )
    .expect("full Quad4 Galerkin assembly");
    let incident = IncidentField::plane_wave_z();
    let rhs = system
        .incident_rhs_from_field(&mesh.elements, &mesh.nodes, &incident, &exterior, 3)
        .expect("independently projected incident trace");
    let solution = system
        .solve(&rhs, 1e-8)
        .expect("native Galerkin equation budget");
    let points = [
        [
            EVALUATION_RADIUS * 0.2_f64.sin(),
            0.0,
            EVALUATION_RADIUS * 0.2_f64.cos(),
        ],
        [
            EVALUATION_RADIUS * 0.55_f64.sin(),
            0.0,
            EVALUATION_RADIUS * 0.55_f64.cos(),
        ],
        [
            EVALUATION_RADIUS * 0.9_f64.sin(),
            0.0,
            EVALUATION_RADIUS * 0.9_f64.cos(),
        ],
        [
            EVALUATION_RADIUS * 1.25_f64.sin(),
            0.0,
            EVALUATION_RADIUS * 1.25_f64.cos(),
        ],
        [
            EVALUATION_RADIUS * 1.6_f64.sin(),
            0.0,
            EVALUATION_RADIUS * 1.6_f64.cos(),
        ],
        [
            EVALUATION_RADIUS * 2.0_f64.sin(),
            0.0,
            EVALUATION_RADIUS * 2.0_f64.cos(),
        ],
        [
            EVALUATION_RADIUS * 2.4_f64.sin(),
            0.0,
            EVALUATION_RADIUS * 2.4_f64.cos(),
        ],
        [
            EVALUATION_RADIUS * 2.8_f64.sin(),
            0.0,
            EVALUATION_RADIUS * 2.8_f64.cos(),
        ],
    ];
    let fields = system
        .evaluate_exterior_field_checked(
            &mesh.elements,
            &mesh.nodes,
            &solution,
            &points,
            &incident,
            &exterior,
            3,
        )
        .expect("continuous Quad4 exterior field");
    let actual = fields.iter().map(|field| field.p_total).collect::<Vec<_>>();
    let field_error = relative_error(&actual, reference);

    let direct = sonium_solvers::direct::lu_solve(system.matrix(), &rhs)
        .expect("continuous Quad4 direct coefficient cross-check");
    let solution_error = relative_error(&solution, &direct);
    let error_squared = (0..system.total_dofs())
        .map(|row| {
            let action = (0..system.total_dofs())
                .map(|col| system.matrix()[[row, col]] * solution[col])
                .sum::<Complex64>();
            (action - rhs[row]).norm_sqr()
        })
        .sum::<f64>();
    let residual = (error_squared / rhs.iter().map(|v| v.norm_sqr()).sum::<f64>()).sqrt();
    println!(
        "Galerkin resolution={face_resolution}, field_error={field_error:.12e}, true_residual={residual:.12e}"
    );
    (
        field_error,
        residual,
        solution_error,
        system.interface_dofs(),
        mesh.elements.len(),
    )
}

pub fn run_continuous_quadratic_quad4_transmission_mie_convergence()
-> ContinuousQuadraticQuad4TransmissionMetrics {
    let exterior_wave_number = 2.0 * PI * FREQUENCY / EXTERIOR_SPEED;
    let interior_wave_number = 2.0 * PI * FREQUENCY / INTERIOR_SPEED;
    let mie = FluidSphereTransmission3d::try_new(
        RADIUS,
        exterior_wave_number,
        interior_wave_number,
        EXTERIOR_DENSITY,
        INTERIOR_DENSITY,
        32,
    )
    .expect("fluid-sphere Mie reference");
    let angles = [0.2, 0.55, 0.9, 1.25, 1.6, 2.0, 2.4, 2.8];
    let reference = angles
        .iter()
        .map(|&theta| {
            mie.exterior_pressure(EVALUATION_RADIUS, theta)
                .expect("Mie exterior reference")
        })
        .collect::<Vec<_>>();

    // The three refinement levels are independent full assemblies +
    // solves (own mesh, locals only); run them concurrently and rejoin
    // in order. Progress prints stay in schedule order after the join.
    use rayon::prelude::*;
    let levels: Vec<(f64, f64, f64, usize, usize)> = [2, 4, 8]
        .into_par_iter()
        .map(|face_resolution| solve_level(face_resolution, &reference))
        .collect();
    let [
        (coarse_error, coarse_residual, coarse_solution_error, coarse_dofs, coarse_panels),
        (medium_error, medium_residual, medium_solution_error, _, _),
        (fine_error, fine_residual, fine_solution_error, fine_dofs, fine_panels),
    ] = levels.try_into().expect("three refinement levels");
    println!(
        "Quad4 Mie complex-field L2: {coarse_error:.3e} -> {medium_error:.3e} -> {fine_error:.3e}"
    );
    ContinuousQuadraticQuad4TransmissionMetrics {
        coarse_error,
        medium_error,
        medium_residual,
        medium_solution_error,
        fine_error,
        coarse_residual,
        fine_residual,
        coarse_solution_error,
        fine_solution_error,
        coarse_dofs,
        fine_dofs,
        coarse_panels,
        fine_panels,
    }
}

#[test]
fn continuous_quadratic_quad4_transmission_converges_to_mie() {
    let metrics = run_continuous_quadratic_quad4_transmission_mie_convergence();
    // Measured on the corrected connectivity-ordered basis (release,
    // 2026-10-17): 1.965e-2 -> 1.358e-2 -> 4.829e-3, ratios 0.691 and
    // 0.356. The coarse 24-panel level is pre-asymptotic; quadrature
    // order 3 vs 4 changes the medium error by less than 0.1%, and a
    // matched-media control reproduces the incident field to 5e-16, so
    // the bound below is the measured discretization contraction with
    // headroom, not a reference-accuracy claim.
    assert!(
        metrics.medium_error <= 0.72 * metrics.coarse_error
            && metrics.fine_error <= 0.72 * metrics.medium_error,
        "Quad4 Mie error does not contract: {:.3e} -> {:.3e} -> {:.3e}",
        metrics.coarse_error,
        metrics.medium_error,
        metrics.fine_error,
    );
    assert!(metrics.fine_error <= 0.01);
    assert!(metrics.coarse_residual <= 1.0e-8);
    assert!(metrics.medium_residual <= 1.0e-8);
    assert!(metrics.fine_residual <= 1.0e-8);
    assert!(metrics.coarse_solution_error <= 1.0e-8);
    assert!(metrics.medium_solution_error <= 1.0e-8);
    assert!(metrics.fine_solution_error <= 1.0e-8);
    assert!(metrics.fine_dofs > metrics.coarse_dofs);
    assert!(metrics.fine_panels > metrics.coarse_panels);
}
