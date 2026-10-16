//! Temporary probe: DFG fine-case linear-solve diagnostics. DELETE AFTER USE.

use sonium_fem::{
    basis::PolynomialDegree,
    boundary::locate_dofs_geometrical,
    mesh::generators::channel_with_cylinder_triangles,
    physics::fluid::stokes::navier_stokes::{NavierStokesConfig, solve_picard},
    spaces::{H1Space, VectorH1Space},
};

const CHANNEL_LENGTH_M: f64 = 2.2;
const CHANNEL_HEIGHT_M: f64 = 0.41;

fn run_case(use_direct_solver: bool, label: &str) {
    const NX: usize = 99;
    const NY: usize = 18;
    let (cylinder_x, cylinder_y) = (0.2, 0.2);
    let mesh = channel_with_cylinder_triangles(
        0.0, CHANNEL_LENGTH_M, 0.0, CHANNEL_HEIGHT_M, NX, NY, cylinder_x, cylinder_y, 0.05,
    );
    let velocity_space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let pressure_space = H1Space::new(PolynomialDegree::P1);
    let tolerance = CHANNEL_LENGTH_M / NX as f64 * 0.6;

    let mut constrained_dofs =
        locate_dofs_geometrical(&mesh, &velocity_space, |x, _, _| x < tolerance);
    constrained_dofs.extend(locate_dofs_geometrical(&mesh, &velocity_space, |_, y, _| {
        y < tolerance || CHANNEL_HEIGHT_M - y < tolerance
    }));
    constrained_dofs.extend(locate_dofs_geometrical(
        &mesh,
        &velocity_space,
        |x, y, _| {
            let distance = ((x - cylinder_x).powi(2) + (y - cylinder_y).powi(2)).sqrt();
            (distance - 0.05).abs() < 1.0e-10
        },
    ));
    constrained_dofs.sort_unstable();
    constrained_dofs.dedup();
    eprintln!("{label}: bc_dofs={} nodes={}", constrained_dofs.len(), mesh.num_nodes());

    let boundary_values = move |x: f64, y: f64, _: f64| {
        if x < tolerance {
            vec![4.0 * 0.3 * y * (CHANNEL_HEIGHT_M - y) / CHANNEL_HEIGHT_M.powi(2), 0.0]
        } else {
            vec![0.0, 0.0]
        }
    };
    let result = solve_picard(
        &mesh,
        &velocity_space,
        &pressure_space,
        &NavierStokesConfig {
            mu: 0.001,
            rho: 1.0,
            max_picard_iterations: 2,
            picard_tolerance: 1.0e-6,
            picard_relaxation: 1.0,
            use_direct_solver,
            linear_solver_max_iterations: 5_000,
            linear_solver_tolerance: 1.0e-9,
            ..Default::default()
        },
        &boundary_values,
        &constrained_dofs,
        |_| vec![0.0, 0.0],
    );
    eprintln!(
        "{label}: converged={} iterations={} residual={:.3e}",
        result.converged, result.iterations, result.residual
    );
}

#[test]
fn probe_direct() {
    run_case(true, "direct");
}

#[test]
fn probe_direct_full() {
    use sonium_fem::mesh::interpolation::interpolate_scalar_at_point;
    use sonium_fem::physics::fluid::stokes::navier_stokes::compute_drag_lift_variational;
    const NX: usize = 99;
    const NY: usize = 18;
    let (cylinder_x, cylinder_y) = (0.2, 0.2);
    let mesh = channel_with_cylinder_triangles(
        0.0, CHANNEL_LENGTH_M, 0.0, CHANNEL_HEIGHT_M, NX, NY, cylinder_x, cylinder_y, 0.05,
    );
    let velocity_space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let pressure_space = H1Space::new(PolynomialDegree::P1);
    let tolerance = CHANNEL_LENGTH_M / NX as f64 * 0.6;
    let mut constrained_dofs =
        locate_dofs_geometrical(&mesh, &velocity_space, |x, _, _| x < tolerance);
    constrained_dofs.extend(locate_dofs_geometrical(&mesh, &velocity_space, |_, y, _| {
        y < tolerance || CHANNEL_HEIGHT_M - y < tolerance
    }));
    constrained_dofs.extend(locate_dofs_geometrical(
        &mesh,
        &velocity_space,
        |x, y, _| {
            let distance = ((x - cylinder_x).powi(2) + (y - cylinder_y).powi(2)).sqrt();
            (distance - 0.05).abs() < 1.0e-10
        },
    ));
    constrained_dofs.sort_unstable();
    constrained_dofs.dedup();
    let boundary_values = move |x: f64, y: f64, _: f64| {
        if x < tolerance {
            vec![4.0 * 0.3 * y * (CHANNEL_HEIGHT_M - y) / CHANNEL_HEIGHT_M.powi(2), 0.0]
        } else {
            vec![0.0, 0.0]
        }
    };
    let result = solve_picard(
        &mesh,
        &velocity_space,
        &pressure_space,
        &NavierStokesConfig {
            mu: 0.001,
            rho: 1.0,
            max_picard_iterations: 100,
            picard_tolerance: 1.0e-6,
            picard_relaxation: 1.0,
            use_direct_solver: true,
            linear_solver_max_iterations: 5_000,
            linear_solver_tolerance: 1.0e-9,
            ..Default::default()
        },
        &boundary_values,
        &constrained_dofs,
        |_| vec![0.0, 0.0],
    );
    eprintln!(
        "direct-full: converged={} iterations={} residual={:.3e}",
        result.converged, result.iterations, result.residual
    );
    assert!(result.converged);
    let (drag, lift) =
        compute_drag_lift_variational(&mesh, &result.velocity, &result.pressure, 0.001, 1.0, 4, |_| {
            vec![0.0, 0.0]
        });
    let scale = 2.0 / (1.0 * 0.2_f64.powi(2) * 0.1);
    let front = interpolate_scalar_at_point(
        &mesh,
        &result.pressure,
        [cylinder_x - 0.05, cylinder_y, 0.0],
    )
    .unwrap();
    let back = interpolate_scalar_at_point(
        &mesh,
        &result.pressure,
        [cylinder_x + 0.05, cylinder_y, 0.0],
    )
    .unwrap();
    eprintln!(
        "direct-full: dp={:.6} cd={:.6} lift={:.6} (ref dp=0.1175 cd=5.5795 lift=0.0106)",
        front - back,
        scale * drag,
        scale * lift
    );
}

#[test]
fn probe_iterative() {
    run_case(false, "iterative");
}

#[test]
fn probe_schur() {
    use sonium_fem::physics::fluid::stokes::navier_stokes::LinearSolverCpuPreconditionerKind;
    const NX: usize = 99;
    const NY: usize = 18;
    let (cylinder_x, cylinder_y) = (0.2, 0.2);
    let mesh = channel_with_cylinder_triangles(
        0.0, CHANNEL_LENGTH_M, 0.0, CHANNEL_HEIGHT_M, NX, NY, cylinder_x, cylinder_y, 0.05,
    );
    let velocity_space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let pressure_space = H1Space::new(PolynomialDegree::P1);
    let tolerance = CHANNEL_LENGTH_M / NX as f64 * 0.6;
    let mut constrained_dofs =
        locate_dofs_geometrical(&mesh, &velocity_space, |x, _, _| x < tolerance);
    constrained_dofs.extend(locate_dofs_geometrical(&mesh, &velocity_space, |_, y, _| {
        y < tolerance || CHANNEL_HEIGHT_M - y < tolerance
    }));
    constrained_dofs.extend(locate_dofs_geometrical(
        &mesh,
        &velocity_space,
        |x, y, _| {
            let distance = ((x - cylinder_x).powi(2) + (y - cylinder_y).powi(2)).sqrt();
            (distance - 0.05).abs() < 1.0e-10
        },
    ));
    constrained_dofs.sort_unstable();
    constrained_dofs.dedup();
    let boundary_values = move |x: f64, y: f64, _: f64| {
        if x < tolerance {
            vec![4.0 * 0.3 * y * (CHANNEL_HEIGHT_M - y) / CHANNEL_HEIGHT_M.powi(2), 0.0]
        } else {
            vec![0.0, 0.0]
        }
    };
    for kind in [
        LinearSolverCpuPreconditionerKind::JacobiSchur,
        LinearSolverCpuPreconditionerKind::IlutSchur,
    ] {
        let result = solve_picard(
            &mesh,
            &velocity_space,
            &pressure_space,
            &NavierStokesConfig {
                mu: 0.001,
                rho: 1.0,
                max_picard_iterations: 2,
                picard_tolerance: 1.0e-6,
                picard_relaxation: 1.0,
                use_direct_solver: false,
                linear_solver_max_iterations: 5_000,
                linear_solver_tolerance: 1.0e-9,
                linear_solver_cpu_preconditioner: kind,
                ..Default::default()
            },
            &boundary_values,
            &constrained_dofs,
            |_| vec![0.0, 0.0],
        );
        eprintln!(
            "schur({kind:?}): converged={} iterations={} residual={:.3e}",
            result.converged, result.iterations, result.residual
        );
    }
}
