//! Bounded independent exact-TBEM residual certification contracts.

use num_complex::Complex64;
use sonium_bem::core::bem_solver::BemPreconditioner;
use sonium_bem::core::incident::IncidentField;
use sonium_bem::core::mesh::generate_cubed_sphere_quad_mesh;
use sonium_bem::core::types::{BasisOrder, PhysicsParams};
use sonium_bem::core::{
    AssemblyMethod, BemProblem, BemSolver, BoundaryConditionType, SolverMethod,
};

fn fixture() -> (BemProblem, Vec<Complex64>) {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 3, 6);
    let dense = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::Direct)
        .solve(&problem)
        .expect("dense fixture solve");
    let rhs = (0..dense.surface_pressure.len())
        .map(|row| Complex64::new(0.2 + row as f64 / 17.0, -((row % 5) as f64) / 23.0))
        .collect();
    (problem, rhs)
}

#[test]
fn hmatrix_forward_and_adjoint_report_independent_exact_residuals() {
    let (problem, rhs) = fixture();
    let solver = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Hmatrix)
        .with_solver_method(SolverMethod::Gmres)
        .with_max_iterations(300)
        .with_tolerance(1.0e-7)
        .with_compressed_exact_residual_certification(64, 1.0e-4)
        .with_compressed_retries(0);

    let forward = solver
        .solve_rhs_many(&problem, std::slice::from_ref(&rhs))
        .expect("H-matrix forward solve");
    let forward_residual = forward[0]
        .exact_reference_residual()
        .expect("forward exact residual certificate");
    assert!(forward_residual.is_finite());
    assert!(
        forward_residual <= 1.0e-4,
        "forward residual {forward_residual:e}"
    );

    let adjoint = solver
        .solve_adjoint_many(&problem, std::slice::from_ref(&rhs))
        .expect("H-matrix adjoint solve");
    let adjoint_residual = adjoint[0]
        .exact_reference_residual()
        .expect("adjoint exact residual certificate");
    assert!(adjoint_residual.is_finite());
    assert!(
        adjoint_residual <= 1.0e-4,
        "adjoint residual {adjoint_residual:e}"
    );
}

#[test]
fn certification_is_disabled_by_default_and_tolerance_is_validated() {
    let (problem, rhs) = fixture();
    let direct = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::Direct);
    let solution = direct
        .solve_rhs_many(&problem, std::slice::from_ref(&rhs))
        .expect("dense forward solve");
    assert_eq!(solution[0].exact_reference_residual(), None);

    let invalid = BemSolver::new().with_compressed_exact_residual_certification(64, 0.0);
    assert!(invalid.solve_rhs_many(&problem, &[]).is_err());
}

#[test]
fn discontinuous_higher_order_hmatrix_certifies_forward_and_adjoint_actions() {
    let problem = BemProblem {
        mesh: generate_cubed_sphere_quad_mesh(0.1, 1),
        physics: PhysicsParams::new(120.0, 343.0, 1.21, false),
        incident_field: IncidentField::plane_wave_z(),
        bc_type: BoundaryConditionType::Rigid,
        use_burton_miller: true,
    };
    for (basis, dof_count) in [(BasisOrder::Linear, 24), (BasisOrder::Quadratic, 54)] {
        let solver = BemSolver::new()
            .with_basis_order(basis)
            .with_assembly_method(AssemblyMethod::Hmatrix)
            .with_solver_method(SolverMethod::Gmres)
            .with_preconditioner(BemPreconditioner::Jacobi)
            .with_hmatrix_accuracy(1.0e-9, 64)
            .with_compressed_retries(0)
            .with_tolerance(1.0e-8)
            .with_max_iterations(300)
            .with_compressed_exact_residual_certification(64, 1.0e-4);

        let rhs = vec![Complex64::new(0.4, -0.2); dof_count];
        let solution = solver
            .solve_rhs_many(&problem, std::slice::from_ref(&rhs))
            .expect("higher-order H-matrix solve with exact certification");
        let forward_residual = solution[0]
            .exact_reference_residual()
            .expect("higher-order forward exact residual certificate");
        assert!(forward_residual.is_finite() && forward_residual <= 1.0e-4);

        let adjoint = solver
            .solve_adjoint_many(&problem, std::slice::from_ref(&rhs))
            .expect("higher-order H-matrix adjoint with exact certification");
        let adjoint_residual = adjoint[0]
            .exact_reference_residual()
            .expect("higher-order adjoint exact residual certificate");
        assert!(adjoint_residual.is_finite() && adjoint_residual <= 1.0e-4);
    }
}

#[test]
fn continuous_higher_order_hmatrix_certifies_shared_dof_incidence() {
    let problem = BemProblem {
        mesh: generate_cubed_sphere_quad_mesh(0.1, 1),
        physics: PhysicsParams::new(120.0, 343.0, 1.21, false),
        incident_field: IncidentField::plane_wave_z(),
        bc_type: BoundaryConditionType::Rigid,
        use_burton_miller: true,
    };
    let solver = BemSolver::new()
        .with_continuous_quadratic()
        .with_assembly_method(AssemblyMethod::Hmatrix)
        .with_solver_method(SolverMethod::Gmres)
        .with_preconditioner(BemPreconditioner::Jacobi)
        .with_hmatrix_accuracy(1.0e-9, 64)
        .with_compressed_retries(0)
        .with_tolerance(1.0e-8)
        .with_max_iterations(300)
        .with_compressed_exact_residual_certification(64, 1.0e-4);

    let rhs = vec![Complex64::new(0.4, -0.2); 26];
    let solution = solver
        .solve_rhs_many(&problem, std::slice::from_ref(&rhs))
        .expect("continuous higher-order H-matrix solve with exact certification");
    let forward_residual = solution[0]
        .exact_reference_residual()
        .expect("continuous forward exact residual certificate");
    assert!(forward_residual.is_finite() && forward_residual <= 1.0e-4);

    let adjoint = solver
        .solve_adjoint_many(&problem, std::slice::from_ref(&rhs))
        .expect("continuous higher-order H-matrix adjoint with exact certification");
    let adjoint_residual = adjoint[0]
        .exact_reference_residual()
        .expect("continuous adjoint exact residual certificate");
    assert!(adjoint_residual.is_finite() && adjoint_residual <= 1.0e-4);
}

#[test]
fn impedance_hmatrix_certifies_exact_admittance_diagonal_terms() {
    let problem = BemProblem {
        mesh: generate_cubed_sphere_quad_mesh(0.1, 1),
        physics: PhysicsParams::new(120.0, 343.0, 1.21, false),
        incident_field: IncidentField::plane_wave_z(),
        bc_type: BoundaryConditionType::Impedance,
        use_burton_miller: true,
    };
    let dof_count = problem.mesh.elements.len();
    problem
        .validate()
        .expect("valid impedance Quad4 sphere fixture");
    let solver = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Hmatrix)
        .with_solver_method(SolverMethod::Gmres)
        .with_preconditioner(BemPreconditioner::Jacobi)
        .with_hmatrix_accuracy(1.0e-9, 64)
        .with_compressed_retries(0)
        .with_tolerance(1.0e-8)
        .with_max_iterations(300)
        .with_compressed_exact_residual_certification(64, 1.0e-4);

    let rhs = vec![Complex64::new(0.4, -0.2); dof_count];
    let solution = solver
        .solve_rhs_many(&problem, std::slice::from_ref(&rhs))
        .expect("impedance H-matrix solve with exact certification");
    let forward_residual = solution[0]
        .exact_reference_residual()
        .expect("impedance forward exact residual certificate");
    assert!(forward_residual.is_finite() && forward_residual <= 1.0e-4);

    let adjoint = solver
        .solve_adjoint_many(&problem, std::slice::from_ref(&rhs))
        .expect("impedance H-matrix adjoint with exact certification");
    let adjoint_residual = adjoint[0]
        .exact_reference_residual()
        .expect("impedance adjoint exact residual certificate");
    assert!(adjoint_residual.is_finite() && adjoint_residual <= 1.0e-4);
}
