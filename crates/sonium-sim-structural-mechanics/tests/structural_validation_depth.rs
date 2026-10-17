//! Structural validation depth: independent convergence, patch/nullspace,
//! locking characterization, and transient energy accounting.

// Rust guideline compliant 2026-02-21

use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::box_mesh_tetrahedra;
use sonium_fem::physics::solid::elasticity::{ElasticityForm, IsotropicMaterial};
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};
use sonium_sim_structural_mechanics::{
    AxisymmetricCylinderStudy, NonlinearTransientFem3DStudy, run_axisymmetric_cylinder,
    run_nonlinear_transient_fem_3d,
};
use sonium_solvers::time_integration::{NewmarkConfig, NewmarkInitial, try_newmark_integrate};
use sonium_solvers::traits::LinearOperator;
use sonium_solvers::{CgConfig, TripletMatrix, cg};

fn lame_displacement_error(radial_elements: usize, poisson_ratio: f64) -> f64 {
    try_lame_displacement_error(radial_elements, poisson_ratio).expect("valid Lamé study")
}

fn try_lame_displacement_error(radial_elements: usize, poisson_ratio: f64) -> Result<f64, String> {
    let mut study = AxisymmetricCylinderStudy::introductory();
    study.geometry.radial_elements = radial_elements;
    study.geometry.axial_elements = 2;
    study.material.poisson_ratio = poisson_ratio;
    match run_axisymmetric_cylinder(&study) {
        Ok(result) => {
            assert!(result.converged);
            Ok(result.max_radial_displacement_relative_error)
        }
        Err(error) => Err(format!("{error:?}")),
    }
}

#[test]
fn lame_cylinder_converges_under_radial_refinement() {
    let grids = [6, 12, 24, 48];
    let errors: Vec<f64> = grids
        .iter()
        .map(|&radial| lame_displacement_error(radial, 0.3))
        .collect();
    for pair in errors.windows(2) {
        assert!(pair[1] < pair[0], "Lamé errors must decrease: {errors:?}");
    }
    // Least-squares slope of log2(error) over successive halving.
    let logs: Vec<f64> = errors.iter().map(|error| error.log2()).collect();
    let mean = logs.iter().sum::<f64>() / logs.len() as f64;
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (index, &value) in logs.iter().enumerate() {
        numerator += (index as f64 - 1.5) * (value - mean);
        denominator += (index as f64 - 1.5).powi(2);
    }
    let slope = -numerator / denominator;
    // Observed first-order in the max-nodal measure (lumped pressure trace
    // dominates the loaded surface); recovering O(h^2) is follow-up work.
    // The binding gates are independent convergence: monotone decrease,
    // a stable positive rate, and a small fine-grid error.
    assert!(slope >= 0.9, "Lamé slope = {slope}, errors = {errors:?}");
    assert!(errors[3] < 0.005, "fine-grid Lamé error = {}", errors[3]);
}

#[test]
fn p1_displacement_formulation_locks_as_nu_approaches_half() {
    // Characterization, not certification: the P1 displacement formulation
    // loses coarse-mesh accuracy near incompressibility (measured ~48x from
    // nu = 0.3 to nu = 0.4999 on the coarse grid). The refined solver now
    // reaches nu = 0.499 within budget (with locking-dominated errors),
    // while nu = 0.4999 still defeats it. A mixed u-p formulation is the
    // documented follow-up; this test pins the remaining gap.
    let coarse = [0.3, 0.49, 0.499, 0.4999].map(|nu| lame_displacement_error(6, nu));
    for pair in coarse.windows(2) {
        assert!(pair[1] > pair[0], "locking must worsen with nu: {coarse:?}");
    }
    assert!(
        coarse[3] > 10.0 * coarse[0],
        "expected severe locking degradation, got {coarse:?}"
    );
    // Moderate nu still converges under refinement.
    let refined_moderate = lame_displacement_error(24, 0.49);
    assert!(
        refined_moderate < coarse[1],
        "refinement must help at nu = 0.49: {refined_moderate} vs {}",
        coarse[1]
    );
    // Refined solves at nu = 0.495 / 0.499 now converge, but their errors
    // stay locking-dominated (worse than the refined nu = 0.49 error).
    for nu in [0.495, 0.499] {
        let refined = lame_displacement_error(24, nu);
        assert!(
            refined > refined_moderate,
            "expected locking-dominated refined error at nu = {nu}: {refined} vs {refined_moderate}"
        );
    }
    // nu = 0.4999 still defeats the default solver budget on the refined grid.
    let outcome = try_lame_displacement_error(24, 0.4999);
    assert!(
        matches!(outcome, Err(ref message) if message.contains("StaticNonConvergence")),
        "expected pinned solver-limit failure at nu = 0.4999, got {outcome:?}"
    );
}

fn linear_field(x: f64, y: f64, z: f64) -> [f64; 3] {
    [
        1.0 + 2.0 * x + 3.0 * y + 4.0 * z,
        5.0 - x + 2.0 * y - 3.0 * z,
        -2.0 + x - y + 5.0 * z,
    ]
}

#[test]
fn patch_test_reproduces_linear_field_on_p1_tets() {
    let mesh = box_mesh_tetrahedra(0.0, 2.0, 0.0, 1.0, 0.0, 0.5, 4, 3, 2);
    let space = VectorH1Space::new(PolynomialDegree::P1, 3);
    let dofs = space.num_global_dofs(&mesh);
    assert_eq!(dofs, mesh.nodes.len() * 3);
    let material = IsotropicMaterial::new(2.0e6, 0.3);
    let form = ElasticityForm::new_3d(material);
    let triplets = assemble_bilinear(&form, &space, &space, &mesh, 2);

    let tolerance = 1.0e-12;
    let on_surface = |index: usize| {
        let point = &mesh.nodes[index];
        point.x <= tolerance
            || (point.x - 2.0).abs() <= tolerance
            || point.y <= tolerance
            || (point.y - 1.0).abs() <= tolerance
            || point.z <= tolerance
            || (point.z - 0.5).abs() <= tolerance
    };
    let mut prescribed = vec![f64::NAN; dofs];
    for (node, _) in mesh.nodes.iter().enumerate() {
        if on_surface(node) {
            let point = &mesh.nodes[node];
            let values = linear_field(point.x, point.y, point.z);
            for component in 0..3 {
                prescribed[3 * node + component] = values[component];
            }
        }
    }
    let interior_dofs = prescribed.iter().filter(|value| value.is_nan()).count();
    assert!(interior_dofs > 0, "patch mesh needs interior DOFs");

    // Dirichlet lift: drop fixed rows, move fixed-column coupling to the RHS.
    let mut constrained = TripletMatrix::<f64>::new(dofs, dofs);
    let mut rhs = vec![0.0; dofs];
    for index in 0..triplets.values.len() {
        let row = triplets.rows[index];
        let column = triplets.cols[index];
        let value = triplets.values[index];
        if prescribed[row].is_nan() {
            if prescribed[column].is_nan() {
                constrained.push(row, column, value);
            } else {
                rhs[row] -= value * prescribed[column];
            }
        }
    }
    for (dof, &value) in prescribed.iter().enumerate() {
        if !value.is_nan() {
            constrained.push(dof, dof, 1.0);
            rhs[dof] = value;
        }
    }
    let csr = constrained.to_csr();
    let solved = cg(
        &csr,
        &rhs,
        &CgConfig {
            max_iterations: 20_000,
            tolerance: 1e-14,
            print_interval: 0,
        },
    );
    assert!(solved.converged);

    let mut max_error = 0.0f64;
    for (node, _) in mesh.nodes.iter().enumerate() {
        let point = &mesh.nodes[node];
        let expected = linear_field(point.x, point.y, point.z);
        for (component, &reference) in expected.iter().enumerate() {
            let error = (solved.x[3 * node + component] - reference).abs();
            max_error = max_error.max(error);
        }
    }
    assert!(max_error < 1.0e-8, "patch-test max error = {max_error}");
}

#[test]
fn free_box_stiffness_annihilates_six_rigid_modes() {
    let mesh = box_mesh_tetrahedra(0.0, 2.0, 0.0, 1.0, 0.0, 0.5, 3, 2, 2);
    let space = VectorH1Space::new(PolynomialDegree::P1, 3);
    let material = IsotropicMaterial::new(2.0e6, 0.3);
    let form = ElasticityForm::new_3d(material);
    let stiffness = assemble_bilinear(&form, &space, &space, &mesh, 2).to_csr();
    let stiffness_norm = stiffness
        .values
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    assert!(stiffness_norm > 0.0);

    let centroid = [1.0, 0.5, 0.25];
    let mut modes = vec![vec![0.0; stiffness.num_rows]; 6];
    for (node, _) in mesh.nodes.iter().enumerate() {
        let point = &mesh.nodes[node];
        let arm = [
            point.x - centroid[0],
            point.y - centroid[1],
            point.z - centroid[2],
        ];
        for component in 0..3 {
            modes[component][3 * node + component] = 1.0;
        }
        // Rotations about x, y, z through the centroid.
        modes[3][3 * node] = 0.0;
        modes[3][3 * node + 1] = -arm[2];
        modes[3][3 * node + 2] = arm[1];
        modes[4][3 * node] = arm[2];
        modes[4][3 * node + 1] = 0.0;
        modes[4][3 * node + 2] = -arm[0];
        modes[5][3 * node] = -arm[1];
        modes[5][3 * node + 1] = arm[0];
        modes[5][3 * node + 2] = 0.0;
    }
    for (index, mode) in modes.iter().enumerate() {
        let residual = stiffness.apply(mode);
        let mode_norm = mode.iter().map(|value| value * value).sum::<f64>().sqrt();
        let residual_norm = residual.iter().map(|value| value.abs()).fold(0.0, f64::max);
        let relative = residual_norm / (stiffness_norm * mode_norm);
        assert!(
            relative < 1.0e-12,
            "rigid mode {index} relative residual = {relative}"
        );
    }
}

#[test]
fn newmark_sdof_energy_stays_bounded_over_many_periods() {
    // Undamped 1 Hz oscillator, 100 steps per period, 100 periods.
    let omega_squared = (2.0 * std::f64::consts::PI).powi(2);
    let dt = 0.01;
    let beta = 0.25;
    let effective = 1.0 + beta * dt * dt * omega_squared;
    let mut triplets = TripletMatrix::<f64>::new(1, 1);
    triplets.push(0, 0, omega_squared);
    let stiffness = triplets.to_csr();
    let initial = NewmarkInitial {
        displacement: vec![1.0],
        velocity: vec![0.0],
        acceleration: vec![-omega_squared],
    };
    let config = NewmarkConfig::average_acceleration(dt, 10_000).with_history();
    let result = try_newmark_integrate(
        None::<&sonium_solvers::CsrMatrix<f64>>,
        &stiffness,
        |_| vec![0.0],
        &initial,
        |rhs: &[f64]| vec![rhs[0] / effective],
        &config,
    )
    .expect("valid Newmark run");
    let energy = |u: f64, v: f64| 0.5 * (v * v + omega_squared * u * u);
    let initial_energy = energy(1.0, 0.0);
    let displacements = result.displacement_history.expect("history");
    let velocities = result.velocity_history.expect("history");
    assert_eq!(displacements.len(), 10_001);
    let mut max_drift: f64 = 0.0;
    for step in 0..displacements.len() {
        let drift = ((energy(displacements[step][0], velocities[step][0]) - initial_energy)
            / initial_energy)
            .abs();
        max_drift = max_drift.max(drift);
    }
    assert!(max_drift < 0.02, "max energy drift = {max_drift}");
    let final_drift = ((energy(result.displacement[0], result.velocity[0]) - initial_energy)
        / initial_energy)
        .abs();
    assert!(final_drift < 0.01, "final energy drift = {final_drift}");
}

#[test]
fn nonlinear_transient_ledger_balances_work_against_energy() {
    let study = NonlinearTransientFem3DStudy::introductory();
    let result = run_nonlinear_transient_fem_3d(&study).expect("valid transient run");
    assert!(result.converged);
    let initial_energy = result.steps[0].total_energy_j;
    let final_energy = result.steps.last().expect("non-empty").total_energy_j;
    let stored = final_energy - initial_energy;
    let imbalance = (result.total_input_work_j - stored - result.total_damping_dissipation_j).abs();
    let scale = result
        .total_input_work_j
        .abs()
        .max(stored.abs())
        .max(result.total_damping_dissipation_j.abs())
        .max(1.0e-30);
    let relative = imbalance / scale;
    assert!(
        relative < 0.05,
        "energy-ledger relative imbalance = {relative} \
         (work = {}, stored = {stored}, dissipated = {})",
        result.total_input_work_j,
        result.total_damping_dissipation_j
    );
}
