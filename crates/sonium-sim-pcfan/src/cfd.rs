// Rust guideline compliant 2026-02-21

use crate::geometry::{ShroudNotch, shroud_in_notch, shroud_strut_notches};
use crate::mesh_motion::{apply_pc_fan_mesh_motion, pc_fan_mesh_motion_design_jacobian};
use crate::model::*;
use crate::resolved_adjoint::{
    PcFanResolvedAdjointConfig, PcFanResolvedAdjointError, PcFanResolvedAdjointReport,
    PcFanResolvedDerivativeProbeConfig, PcFanResolvedFlowDerivativeData,
    PcFanResolvedFlowDerivativeProbeReport, PcFanResolvedFlowNewtonLinearization,
    PcFanResolvedFlowPicardLinearization, PcFanResolvedFlowSparseMatrix,
    probe_pc_fan_resolved_flow_derivatives, solve_pc_fan_resolved_flow_adjoint_from_newton,
};
use crate::state_transfer::{
    PcFanCfdKOmegaBdfState, PcFanCfdKOmegaBdfStateAdjointTransferChain,
    PcFanCfdKOmegaBdfStateTransferReport, PcFanCfdKOmegaStateTransferError,
    PcFanCfdP2P1StateAdjointTransfer, PcFanCfdP2P1StateAdjointTransferChain,
    PcFanCfdP2P1StateTransferReport, PcFanCfdStateTransferError,
    transfer_pc_fan_cfd_komega_bdf_state, transfer_pc_fan_cfd_komega_bdf_state_adjoint_chain,
    transfer_pc_fan_cfd_p2p1_bdf_state, transfer_pc_fan_cfd_p2p1_bdf_state_adjoint_chain,
    transfer_pc_fan_cfd_p2p1_state, transfer_pc_fan_cfd_p2p1_state_adjoint,
};
use crate::transient_adjoint::{
    PcFanResolvedTransientAdjointReport, PcFanResolvedTransientAdjointWithInitialHistory,
    PcFanResolvedTransientHistoryJacobian, PcFanResolvedTransientStageInitialHistoryJacobian,
    PcFanResolvedTransientStep, solve_pc_fan_resolved_transient_adjoint,
    solve_pc_fan_resolved_transient_adjoint_with_stage_initial_history,
};
use crate::transported_turbulence::{
    PcFanKOmegaCoupledLinearization, advance_pc_fan_cfd_komega_bdf_step_with_element_fields,
    apply_pc_fan_cfd_komega_mean_eddy_viscosity, assemble_pc_fan_cfd_komega_coupled_linearization,
    assemble_pc_fan_cfd_komega_monolithic_history_linearization,
    assemble_pc_fan_cfd_komega_monolithic_linearization,
    assemble_pc_fan_cfd_komega_nonlinear_residual_and_linearization,
    assemble_pc_fan_cfd_komega_scalar_mesh_motion_design_residual,
    assemble_pc_fan_cfd_komega_transient_history_linearization,
    derive_pc_fan_cfd_komega_element_fields, initialize_pc_fan_cfd_komega_bdf_state,
};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::locate_dofs_geometrical;
use sonium_fem::mesh::generators::{box_mesh_tetrahedra, box_mesh_tetrahedra_symmetric};
use sonium_fem::physics::fluid::stokes::navier_stokes::{
    NavierStokesConfig, assemble_body_force_design_residual_with_viscosity,
    assemble_mesh_motion_design_residual,
    solve_picard_with_velocity_force_and_jacobian_from_initial,
};
use sonium_fem::physics::fluid::turbulence::{
    KOmegaSstConstants, KOmegaTransportConfig, KOmegaTransportLinearization,
};
use sonium_fem::spaces::{FunctionSpace, H1Space, VectorH1Space};
use sonium_solvers::traits::LinearOperator;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Exact fixed-history flow tangent provenance for the transported
/// segregated BDF map. Momentum is differentiated at fixed previous
/// k--omega history; scalar cross-derivatives are retained in the adjacent
/// history block and do not claim a monolithic Newton solve.
pub const PC_FAN_TRANSPORTED_BDF_NEWTON_PROVENANCE: &str =
    "fem:navier-stokes:bdf-newton-transported-segregated:v1";
// The production transported runner publishes this only after assembling
// and solving the current coupled momentum/scalar Newton-BDF residual.
const PC_FAN_TRANSPORTED_MONOLITHIC_BDF_NEWTON_PROVENANCE: &str =
    "fem:navier-stokes:bdf-newton-transported-monolithic:v1";

const M3S_TO_CFM: f64 = 2_118.880_003_289_3;
const SOLID_BOUNDARY_INFLUENCE_CELLS: f64 = 0.55;
const IMMERSED_SOLID_MIN_RELAXATION_S: f64 = 0.005;
const IMMERSED_SOLID_MAX_RELAXATION_S: f64 = 0.050;
const FAN_PICARD_RELAXATION: f64 = 0.01;

/// Picard relaxation for loose continuation stages.
///
/// The production 1% relaxation keeps cold and final solves stable,
/// but on developed-flow ramp stages the exact Newton correction is
/// routinely rejected and the Picard update is the only progress:
/// at 1% it crawls at rel_change ~1e-3 until the stage budget
/// expires. At 10% the same stages converge (the 8-cell upper ramp
/// stalls at 0.36 speed fraction with 1% and converges with 10%),
/// while Newton remains the primary corrector on every stage.
const LOOSE_STAGE_PICARD_RELAXATION: f64 = 0.1;

/// Diagnostic override for the steady Picard relaxation factor.
///
/// Production uses [`FAN_PICARD_RELAXATION`] (or
/// [`LOOSE_STAGE_PICARD_RELAXATION`] for loose continuation stages);
/// `SONIUM_PCFAN_PICARD_RELAXATION` exists only to probe fixed-point
/// stability from examples and tests.
fn fan_picard_relaxation(default: f64) -> f64 {
    std::env::var("SONIUM_PCFAN_PICARD_RELAXATION")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0 && *value <= 1.0)
        .unwrap_or(default)
}
const ACTUATOR_INDUCED_VELOCITY_GAIN: f64 = 0.01;
const ACTUATOR_MIN_RADIAL_INFLUENCE_M: f64 = 0.010;
const ACTUATOR_MIN_AXIAL_INFLUENCE_M: f64 = 0.015;
/// Width of the C1 immersed-wall transition used by the full rotating-wall
/// model.  It is intentionally tied to the local mesh scale when the solver
/// constructs the penalty, while the standalone geometry fixture keeps this
/// conservative default.
const DEFAULT_BLADE_WALL_SMOOTHING_WIDTH_M: f64 = 5.0e-4;
const BLADE_WALL_SMOOTH_ABS_M: f64 = 1.0e-9;

pub fn solve_pc_fan_cfd_mean_flow(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
) -> PcFanCfdFlowSolution {
    let geometry = config.build_geometry(design);
    solve_pc_fan_cfd_mean_flow_for_geometry(config, design, &geometry)
}

#[cfg(test)]
mod transported_newton_merit_tests {
    use super::transported_newton_trial_is_accepted;

    #[test]
    fn residual_merit_requires_sufficient_decrease() {
        assert!(transported_newton_trial_is_accepted(10.0, 5.0, 1.0));
        assert!(transported_newton_trial_is_accepted(10.0, 9.0, 0.25));
        assert!(!transported_newton_trial_is_accepted(10.0, 10.0, 1.0));
        assert!(!transported_newton_trial_is_accepted(10.0, 11.0, 0.5));
        assert!(!transported_newton_trial_is_accepted(10.0, f64::NAN, 1.0));
        assert!(!transported_newton_trial_is_accepted(0.0, 0.0, 1.0));
    }
}

#[cfg(test)]
mod accelerator_observation_tests {
    use super::*;

    #[test]
    fn parity_observation_uses_flow_solution_backend_provenance() {
        let config = PcFanConfig::default_120mm();
        let design = config.default_design_variables();
        let geometry = config.build_geometry(&design);
        let domain = geometry.build_cfd_domain(&config);
        let solution = rejected_mesh_motion_solution(&domain, &config.cfd);

        let observation =
            crate::accelerator_parity::PcFanAcceleratorParityObservation::from_cfd_solution(
                config.operating_point.rpm,
                PcFanCfdBackend::Cuda,
                &solution,
            );

        assert_eq!(observation.requested_backend, PcFanCfdBackend::Cuda);
        assert_eq!(observation.actual_backend, PcFanCfdBackend::Cpu);
        assert_eq!(observation.residual, solution.residual);
    }
}

#[cfg(test)]
mod transient_convergence_tests {
    use super::*;

    fn small_transient_config() -> (PcFanConfig, PcFanDesignVariables) {
        let mut config = PcFanConfig::default_120mm();
        config.cfd.flow_model = PcFanFlowModel::IncompressibleCfdFull;
        config.cfd.backend = PcFanCfdBackend::Cpu;
        config.cfd.cells_x = 2;
        config.cfd.cells_y = 2;
        config.cfd.cells_z = 2;
        config.cfd.max_picard_iterations = 1;
        // This fixture checks transported-state/provenance wiring; strict
        // nonlinear admission is covered by the scalar residual/Jacobian tests.
        config.cfd.nonlinear_tolerance = 1.0e6;
        config.cfd.max_relative_mass_imbalance = 10.0;
        config.cfd.max_relative_momentum_imbalance = 10.0;
        config.cfd.max_relative_pressure_closure = 10.0;
        config.cfd.max_relative_energy_imbalance = 10.0;
        let design = config.default_design_variables();
        (config, design)
    }

    fn dot_values(left: &[f64], right: &[f64]) -> f64 {
        assert_eq!(left.len(), right.len());
        left.iter()
            .zip(right)
            .map(|(left, right)| left * right)
            .sum()
    }

    fn remesh_schedule_fixture(
        mesh_chain: &[&sonium_fem::mesh::Mesh],
        transported_turbulence: bool,
    ) -> PcFanCfdTransientRemeshSchedule {
        let stage_count = mesh_chain.len().saturating_sub(1);
        let zero_conservation = PcFanCfdConservation {
            inlet_mass_flow_kg_s: 0.0,
            outlet_mass_flow_kg_s: 0.0,
            relative_mass_imbalance: 0.0,
            inlet_axial_momentum_flux_n: 0.0,
            outlet_axial_momentum_flux_n: 0.0,
            pressure_force_n: 0.0,
            body_force_n: 0.0,
            axial_momentum_residual_n: 0.0,
            relative_momentum_imbalance: 0.0,
            pressure_force_from_loading_n: 0.0,
            relative_pressure_closure: 0.0,
            pressure_power_w: 0.0,
            shaft_power_to_fluid_w: 0.0,
            relative_energy_imbalance: 0.0,
        };
        let stages = (0..stage_count)
            .map(|stage| PcFanCfdTransientTrajectory {
                observation: PcFanCfdTransientObservation {
                    final_time_s: stage as f64,
                    airflow_cfm: 0.0,
                    pressure_drop_pa: 0.0,
                    axial_loading_n: 0.0,
                    residual: 0.0,
                    conservation: zero_conservation,
                    converged: true,
                },
                steps: Vec::new(),
                stage_initial_history_jacobians: Vec::new(),
            })
            .collect();
        let mean_flow_transfer_reports = mesh_chain
            .windows(2)
            .map(|meshes| {
                let source_velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
                let target_velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
                let source_pressure_space = H1Space::new(PolynomialDegree::P1);
                let target_pressure_space = H1Space::new(PolynomialDegree::P1);
                PcFanCfdP2P1StateTransferReport {
                    source_node_count: meshes[0].num_nodes(),
                    target_node_count: meshes[1].num_nodes(),
                    source_element_count: meshes[0].num_elements(),
                    target_element_count: meshes[1].num_elements(),
                    source_velocity_dof_count: source_velocity_space.num_global_dofs(meshes[0]),
                    target_velocity_dof_count: target_velocity_space.num_global_dofs(meshes[1]),
                    source_pressure_dof_count: source_pressure_space.num_global_dofs(meshes[0]),
                    target_pressure_dof_count: target_pressure_space.num_global_dofs(meshes[1]),
                }
            })
            .collect();
        let transported_turbulence_transfer_reports = transported_turbulence.then(|| {
            mesh_chain
                .windows(2)
                .map(|meshes| PcFanCfdKOmegaBdfStateTransferReport {
                    source_scalar_dof_count: H1Space::new(PolynomialDegree::P1)
                        .num_global_dofs(meshes[0]),
                    target_scalar_dof_count: H1Space::new(PolynomialDegree::P1)
                        .num_global_dofs(meshes[1]),
                    minimum_k: 0.0,
                    minimum_omega: 0.0,
                })
                .collect()
        });
        PcFanCfdTransientRemeshSchedule {
            stages,
            mean_flow_transfer_reports,
            transported_turbulence_transfer_reports,
            requires_operator_reassembly: true,
            topology_transfer_adjoint_admissible: false,
            provenance: "pcfan:external-remesh-transient-schedule:v1",
        }
    }

    #[test]
    fn transported_turbulence_is_retained_on_each_accepted_step() {
        let (mut config, design) = small_transient_config();
        config.cfd.turbulence = PcFanTurbulenceModel::KOmegaSstTransported;
        let trajectory = solve_pc_fan_cfd_rotating_wall_transient_with_trajectory(
            &config, &design, 1.0e-3, 1.0e-4,
        )
        .expect("tiny transported transient fixture should run");
        assert!(trajectory.steps.iter().all(|step| {
            step.state_jacobian_provenance == PC_FAN_TRANSPORTED_MONOLITHIC_BDF_NEWTON_PROVENANCE
        }));

        assert!(
            trajectory
                .steps
                .iter()
                .all(|step| step.transported_turbulence.is_some())
        );
        assert!(trajectory.steps.iter().all(|step| {
            step.transported_eddy_viscosity_per_element
                .as_ref()
                .is_some_and(|field| field.len() > 1)
        }));
        assert!(trajectory.steps.iter().all(|step| {
            step.transported_turbulence_linearization
                .as_ref()
                .is_some_and(|linearization| {
                    linearization.provenance
                    == sonium_fem::physics::fluid::turbulence::K_OMEGA_TRANSPORT_LINEARIZATION_PROVENANCE
                })
        }));
        assert!(trajectory.steps.iter().all(|step| {
            step.transported_turbulence_coupled_linearization
                .as_ref()
                .is_some_and(|linearization| {
                    linearization.provenance
                    == crate::transported_turbulence::PC_FAN_KOMEGA_COUPLED_LINEARIZATION_PROVENANCE
                && linearization.momentum_wrt_k.num_rows == step.bdf_state.velocity.len()
                && linearization.k_wrt_velocity.num_cols == step.bdf_state.velocity.len()
                })
        }));
        assert!(trajectory.steps.windows(2).all(|steps| {
            steps[1]
                .transported_turbulence
                .as_ref()
                .expect("transported state")
                .step
                > steps[0]
                    .transported_turbulence
                    .as_ref()
                    .expect("transported state")
                    .step
        }));

        let adjoint_config = PcFanResolvedAdjointConfig {
            // This runner fixture intentionally admits a wiring-only
            // nonlinear residual; strict residual admission is covered by
            // the scalar and coupled-linearization regressions.
            tolerance: 1.0e-2,
            max_state_residual: 1.0e6,
            ..PcFanResolvedAdjointConfig::default()
        };
        let adjoint = solve_pc_fan_cfd_resolved_transient_objective_adjoint(
            &adjoint_config,
            &config,
            &design,
            &trajectory,
            PcFanResolvedTransientObjective::PressureDrop,
        )
        .expect("transported production objective adjoint should solve");
        assert_eq!(adjoint.adjoints.len(), trajectory.steps.len());
        assert!(adjoint.adjoints.iter().all(|gradient| {
            gradient.iter().all(|value| value.is_finite())
                && gradient.len()
                    == trajectory.steps[0].state.len()
                        + 2 * trajectory.steps[0]
                            .transported_turbulence
                            .as_ref()
                            .expect("transported state")
                            .k
                            .len()
        }));
        assert!(adjoint.converged);
    }

    #[test]
    fn builtin_body_fitted_remesh_levels_generate_checked_targets() {
        let (config, design) = small_transient_config();
        let mut moved_design = design.clone();
        moved_design.hub_radius_offset_m = 1.0e-5;
        moved_design.tip_radius_offset_m = -1.0e-5;

        let coarse_level = PcFanCfdBodyFittedRemeshLevel::new(1, 2, 2);
        let fine_level = PcFanCfdBodyFittedRemeshLevel::new(2, 2, 2);
        let baseline = build_pc_fan_cfd_body_fitted_mesh(&config, &design, coarse_level)
            .expect("baseline built-in mesh should be admitted");
        let coarse = build_pc_fan_cfd_body_fitted_mesh(&config, &moved_design, coarse_level)
            .expect("coarse built-in mesh should be admitted");
        let fine = build_pc_fan_cfd_body_fitted_mesh(&config, &moved_design, fine_level)
            .expect("fine built-in mesh should be admitted");

        assert!(coarse.validate_connectivity().is_ok());
        assert!(fine.validate_connectivity().is_ok());
        assert_ne!(coarse.num_nodes(), fine.num_nodes());
        assert!(
            coarse
                .nodes
                .iter()
                .all(|point| { point.x.is_finite() && point.y.is_finite() && point.z.is_finite() })
        );
        assert!(
            coarse
                .nodes
                .iter()
                .zip(baseline.nodes.iter())
                .any(|(moved, original)| {
                    (moved.x - original.x).abs() > 0.0
                        || (moved.y - original.y).abs() > 0.0
                        || (moved.z - original.z).abs() > 0.0
                })
        );
    }

    #[test]
    fn builtin_body_fitted_remesher_rejects_invalid_level_schedule_before_solving() {
        let (config, design) = small_transient_config();
        let source_mesh = sonium_fem::mesh::unit_cube_tetrahedra(1);
        let velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
        let pressure_space = H1Space::new(PolynomialDegree::P1);
        let fem_state = sonium_fem::physics::fluid::stokes::time_stepping::BdfState::from_rest(
            velocity_space.num_global_dofs(&source_mesh),
            pressure_space.num_global_dofs(&source_mesh),
        );
        let source_state = PcFanCfdTransientBdfState::from_fem(&fem_state);

        for levels in [
            Vec::new(),
            vec![PcFanCfdBodyFittedRemeshLevel::new(1, 0, 1)],
        ] {
            assert!(matches!(
                solve_pc_fan_cfd_rotating_wall_transient_with_builtin_body_fitted_remesher(
                    &config,
                    &design,
                    1.0e-3,
                    1.0e-4,
                    &source_mesh,
                    &source_state,
                    None,
                    &levels,
                ),
                Err(PcFanCfdTransientStateTransferError::StateTransfer(
                    PcFanCfdStateTransferError::InvalidMesh { .. }
                ))
            ));
        }
    }

    #[test]
    fn builtin_body_fitted_remesher_routes_generated_level_through_schedule() {
        let (config, design) = small_transient_config();
        let geometry = config.build_geometry(&design);
        let domain = geometry.build_cfd_domain(&config);
        let source_mesh = box_mesh_tetrahedra_symmetric(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            2,
            2,
            2,
        );
        let velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
        let pressure_space = H1Space::new(PolynomialDegree::P1);
        let fem_state = sonium_fem::physics::fluid::stokes::time_stepping::BdfState::from_rest(
            velocity_space.num_global_dofs(&source_mesh),
            pressure_space.num_global_dofs(&source_mesh),
        );
        let source_state = PcFanCfdTransientBdfState::from_fem(&fem_state);

        let schedule = solve_pc_fan_cfd_rotating_wall_transient_with_builtin_body_fitted_remesher(
            &config,
            &design,
            1.0e-3,
            1.0e-4,
            &source_mesh,
            &source_state,
            None,
            &[PcFanCfdBodyFittedRemeshLevel::new(1, 2, 2)],
        )
        .expect("built-in remesh stage should route through transient schedule");

        assert_eq!(schedule.stages.len(), 1);
        assert_eq!(schedule.mean_flow_transfer_reports.len(), 1);
        assert!(schedule.requires_operator_reassembly);
        assert!(!schedule.topology_transfer_adjoint_admissible);
        assert_eq!(
            schedule.provenance,
            "pcfan:external-remesh-provider-transient-schedule:v1"
        );
    }

    #[test]
    fn transported_remesh_runner_transfers_history_and_reassembles_target() {
        let (mut config, design) = small_transient_config();
        config.cfd.turbulence = PcFanTurbulenceModel::KOmegaSstTransported;
        let geometry = config.build_geometry(&design);
        let domain = geometry.build_cfd_domain(&config);
        let source_mesh = box_mesh_tetrahedra_symmetric(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            2,
            2,
            2,
        );
        let target_mesh = box_mesh_tetrahedra_symmetric(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            1,
            2,
            2,
        );
        let velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
        let pressure_space = H1Space::new(PolynomialDegree::P1);
        let source_state = sonium_fem::physics::fluid::stokes::time_stepping::BdfState::from_rest(
            velocity_space.num_global_dofs(&source_mesh),
            pressure_space.num_global_dofs(&source_mesh),
        );
        let source_turbulence =
            initialize_pc_fan_cfd_komega_bdf_state(&source_mesh, 0.0, 0, 2.0e-6, 2.0)
                .expect("source transported history should initialise");

        let trajectory = solve_pc_fan_cfd_rotating_wall_transient_with_state_transfer_to_mesh(
            &config,
            &design,
            1.0e-3,
            1.0e-4,
            &source_mesh,
            &target_mesh,
            &source_state,
            Some(&source_turbulence),
        )
        .expect("target remesh handoff should run on changed topology");

        let first = trajectory.steps.first().expect("accepted target step");
        assert_eq!(
            first.bdf_state.velocity.len(),
            velocity_space.num_global_dofs(&target_mesh)
        );
        assert_eq!(
            first.bdf_state.pressure.len(),
            pressure_space.num_global_dofs(&target_mesh)
        );
        assert_eq!(
            first.transported_turbulence.as_ref().unwrap().k.len(),
            target_mesh.num_nodes()
        );
        assert!(!first.topology_transfer_adjoint_admissible);
    }

    #[test]
    fn transported_remesh_schedule_chains_accepted_history() {
        let (mut config, design) = small_transient_config();
        config.cfd.turbulence = PcFanTurbulenceModel::KOmegaSstTransported;
        let geometry = config.build_geometry(&design);
        let domain = geometry.build_cfd_domain(&config);
        let source_mesh = box_mesh_tetrahedra_symmetric(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            2,
            2,
            2,
        );
        let first_target_mesh = box_mesh_tetrahedra_symmetric(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            1,
            2,
            2,
        );
        let second_target_mesh = box_mesh_tetrahedra_symmetric(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            2,
            1,
            2,
        );
        let velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
        let pressure_space = H1Space::new(PolynomialDegree::P1);
        let source_turbulence =
            initialize_pc_fan_cfd_komega_bdf_state(&source_mesh, 0.0, 0, 2.0e-6, 2.0)
                .expect("source transported history initialise");

        let source_fem_state =
            sonium_fem::physics::fluid::stokes::time_stepping::BdfState::from_rest(
                velocity_space.num_global_dofs(&source_mesh),
                pressure_space.num_global_dofs(&source_mesh),
            );
        let source_state = PcFanCfdTransientBdfState::from_fem(&source_fem_state);
        let schedule = solve_pc_fan_cfd_rotating_wall_transient_with_state_transfer_schedule(
            &config,
            &design,
            1.0e-3,
            1.0e-4,
            &source_mesh,
            &source_state,
            Some(&source_turbulence),
            &[&first_target_mesh, &second_target_mesh],
        )
        .expect("multi-stage remesh schedule should run");

        assert_eq!(schedule.stages.len(), 2);
        assert_eq!(schedule.mean_flow_transfer_reports.len(), 2);
        assert_eq!(
            schedule
                .transported_turbulence_transfer_reports
                .as_ref()
                .expect("transported transfer reports")
                .len(),
            2
        );
        assert!(schedule.requires_operator_reassembly);
        assert!(!schedule.topology_transfer_adjoint_admissible);
        assert_eq!(
            schedule.provenance,
            "pcfan:external-remesh-transient-schedule:v1"
        );

        let first_final = schedule.stages[0]
            .steps
            .last()
            .expect("first stage final state");
        let second_first = schedule.stages[1]
            .steps
            .first()
            .expect("second stage first state");
        assert!(second_first.bdf_state.time > first_final.bdf_state.time);
        assert_eq!(
            second_first.bdf_state.velocity.len(),
            velocity_space.num_global_dofs(&second_target_mesh)
        );
        assert!(schedule.stages.iter().all(|stage| {
            stage
                .steps
                .iter()
                .all(|step| step.transported_turbulence.is_some())
        }));
    }

    #[test]
    fn transported_remesh_schedule_composes_fixed_map_state_adjoint() {
        let (mut config, design) = small_transient_config();
        config.cfd.turbulence = PcFanTurbulenceModel::KOmegaSstTransported;
        let geometry = config.build_geometry(&design);
        let domain = geometry.build_cfd_domain(&config);
        let source_mesh = box_mesh_tetrahedra_symmetric(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            2,
            2,
            2,
        );
        let first_target_mesh = box_mesh_tetrahedra_symmetric(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            1,
            2,
            2,
        );
        let second_target_mesh = box_mesh_tetrahedra_symmetric(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            2,
            1,
            2,
        );
        let velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
        let pressure_space = H1Space::new(PolynomialDegree::P1);
        let source_turbulence =
            initialize_pc_fan_cfd_komega_bdf_state(&source_mesh, 0.0, 0, 2.0e-6, 2.0)
                .expect("source transported history initialise");
        let schedule = remesh_schedule_fixture(
            &[&source_mesh, &first_target_mesh, &second_target_mesh],
            true,
        );

        let source_for_pairing = sonium_fem::physics::fluid::stokes::time_stepping::BdfState {
            velocity: (0..velocity_space.num_global_dofs(&source_mesh))
                .map(|index| 0.2 + 0.03 * index as f64)
                .collect(),
            velocity_prev: (0..velocity_space.num_global_dofs(&source_mesh))
                .map(|index| -0.1 + 0.05 * index as f64)
                .collect(),
            pressure: (0..pressure_space.num_global_dofs(&source_mesh))
                .map(|index| 0.8 + 0.02 * index as f64)
                .collect(),
            time: 0.25,
            step: 4,
        };
        let mesh_chain = [&source_mesh, &first_target_mesh, &second_target_mesh];
        let forward =
            crate::transfer_pc_fan_cfd_p2p1_bdf_state_chain(&mesh_chain, &source_for_pairing)
                .expect("forward mean-flow transfer chain should succeed");
        let terminal = forward.states.last().expect("terminal mean-flow state");
        let terminal_adjoint = sonium_fem::physics::fluid::stokes::time_stepping::BdfState {
            velocity: terminal
                .velocity
                .iter()
                .enumerate()
                .map(|(index, _)| -0.35 + 0.017 * index as f64)
                .collect(),
            velocity_prev: terminal
                .velocity_prev
                .iter()
                .enumerate()
                .map(|(index, _)| 0.45 - 0.013 * index as f64)
                .collect(),
            pressure: terminal
                .pressure
                .iter()
                .enumerate()
                .map(|(index, _)| -0.2 + 0.031 * index as f64)
                .collect(),
            time: terminal.time,
            step: terminal.step,
        };
        let source_turbulence_for_pairing = source_turbulence.clone();
        let forward_turbulence = crate::transfer_pc_fan_cfd_komega_bdf_state_chain(
            &mesh_chain,
            &source_turbulence_for_pairing,
        )
        .expect("forward k--omega transfer chain should succeed");
        let terminal_turbulence = forward_turbulence
            .states
            .last()
            .expect("terminal k--omega state");
        let terminal_turbulence_adjoint = PcFanCfdKOmegaBdfState {
            k: terminal_turbulence
                .k
                .iter()
                .enumerate()
                .map(|(index, _)| -0.3 + 0.01 * index as f64)
                .collect(),
            k_prev: terminal_turbulence
                .k_prev
                .iter()
                .enumerate()
                .map(|(index, _)| 0.2 - 0.015 * index as f64)
                .collect(),
            omega: terminal_turbulence
                .omega
                .iter()
                .enumerate()
                .map(|(index, _)| -0.1 + 0.02 * index as f64)
                .collect(),
            omega_prev: terminal_turbulence
                .omega_prev
                .iter()
                .enumerate()
                .map(|(index, _)| 0.4 - 0.01 * index as f64)
                .collect(),
            time: terminal_turbulence.time,
            step: terminal_turbulence.step,
        };

        let adjoint = transfer_pc_fan_cfd_transient_remesh_schedule_state_adjoint(
            &schedule,
            &mesh_chain,
            &terminal_adjoint,
            Some(&terminal_turbulence_adjoint),
        )
        .expect("schedule state adjoint should compose fixed maps");
        assert!(adjoint.fixed_map_transpose_admissible);
        assert!(!adjoint.topology_design_adjoint_admissible);
        assert_eq!(adjoint.mean_flow.states.len(), 2);
        assert_eq!(
            adjoint
                .transported_turbulence
                .as_ref()
                .expect("transported state adjoint")
                .states
                .len(),
            2
        );

        let mean_terminal_pairing = dot_values(&terminal.velocity, &terminal_adjoint.velocity)
            + dot_values(&terminal.velocity_prev, &terminal_adjoint.velocity_prev)
            + dot_values(&terminal.pressure, &terminal_adjoint.pressure);
        let mean_source_adjoint = adjoint
            .mean_flow
            .states
            .last()
            .expect("source mean-flow adjoint");
        let mean_source_pairing =
            dot_values(&source_for_pairing.velocity, &mean_source_adjoint.velocity)
                + dot_values(
                    &source_for_pairing.velocity_prev,
                    &mean_source_adjoint.velocity_prev,
                )
                + dot_values(&source_for_pairing.pressure, &mean_source_adjoint.pressure);
        assert!((mean_terminal_pairing - mean_source_pairing).abs() < 1.0e-8);

        let terminal_turbulence_pairing =
            dot_values(&terminal_turbulence.k, &terminal_turbulence_adjoint.k)
                + dot_values(
                    &terminal_turbulence.k_prev,
                    &terminal_turbulence_adjoint.k_prev,
                )
                + dot_values(
                    &terminal_turbulence.omega,
                    &terminal_turbulence_adjoint.omega,
                )
                + dot_values(
                    &terminal_turbulence.omega_prev,
                    &terminal_turbulence_adjoint.omega_prev,
                );
        let source_turbulence_adjoint = adjoint
            .transported_turbulence
            .as_ref()
            .expect("source transported state adjoint")
            .states
            .last()
            .expect("source transported adjoint");
        let source_turbulence_pairing = dot_values(
            &source_turbulence_for_pairing.k,
            &source_turbulence_adjoint.k,
        ) + dot_values(
            &source_turbulence_for_pairing.k_prev,
            &source_turbulence_adjoint.k_prev,
        ) + dot_values(
            &source_turbulence_for_pairing.omega,
            &source_turbulence_adjoint.omega,
        ) + dot_values(
            &source_turbulence_for_pairing.omega_prev,
            &source_turbulence_adjoint.omega_prev,
        );
        assert!((terminal_turbulence_pairing - source_turbulence_pairing).abs() < 1.0e-10);
    }

    #[test]
    fn remesh_schedule_state_adjoint_rejects_missing_mesh_chain_transactionally() {
        let (mut config, design) = small_transient_config();
        config.cfd.turbulence = PcFanTurbulenceModel::Laminar;
        let geometry = config.build_geometry(&design);
        let domain = geometry.build_cfd_domain(&config);
        let source_mesh = box_mesh_tetrahedra_symmetric(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            2,
            2,
            2,
        );
        let target_mesh = box_mesh_tetrahedra_symmetric(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            1,
            2,
            2,
        );
        let velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
        let pressure_space = H1Space::new(PolynomialDegree::P1);
        let schedule = remesh_schedule_fixture(&[&source_mesh, &target_mesh], false);
        let mut terminal_state =
            sonium_fem::physics::fluid::stokes::time_stepping::BdfState::from_rest(
                velocity_space.num_global_dofs(&target_mesh),
                pressure_space.num_global_dofs(&target_mesh),
            );
        for (index, value) in terminal_state.velocity.iter_mut().enumerate() {
            *value = -0.25 + 0.03 * index as f64;
        }
        for (index, value) in terminal_state.velocity_prev.iter_mut().enumerate() {
            *value = 0.15 - 0.02 * index as f64;
        }
        for (index, value) in terminal_state.pressure.iter_mut().enumerate() {
            *value = 0.4 + 0.01 * index as f64;
        }

        let valid_adjoint = transfer_pc_fan_cfd_transient_remesh_schedule_state_adjoint(
            &schedule,
            &[&source_mesh, &target_mesh],
            &terminal_state,
            None,
        )
        .expect("one-stage schedule state adjoint should compose fixed map");
        assert_eq!(valid_adjoint.mean_flow.states.len(), 1);
        let source_primal = sonium_fem::physics::fluid::stokes::time_stepping::BdfState {
            velocity: (0..velocity_space.num_global_dofs(&source_mesh))
                .map(|index| 0.1 + 0.02 * index as f64)
                .collect(),
            velocity_prev: (0..velocity_space.num_global_dofs(&source_mesh))
                .map(|index| -0.2 + 0.015 * index as f64)
                .collect(),
            pressure: (0..pressure_space.num_global_dofs(&source_mesh))
                .map(|index| 0.3 - 0.01 * index as f64)
                .collect(),
            time: terminal_state.time,
            step: terminal_state.step,
        };
        let forward = crate::transfer_pc_fan_cfd_p2p1_bdf_state_chain(
            &[&source_mesh, &target_mesh],
            &source_primal,
        )
        .expect("one-stage forward state transfer should succeed");
        let forward_terminal = forward.states.last().expect("forward terminal state");
        let source_adjoint = valid_adjoint
            .mean_flow
            .states
            .last()
            .expect("source state adjoint");
        let terminal_pairing = dot_values(&forward_terminal.velocity, &terminal_state.velocity)
            + dot_values(
                &forward_terminal.velocity_prev,
                &terminal_state.velocity_prev,
            )
            + dot_values(&forward_terminal.pressure, &terminal_state.pressure);
        let source_pairing = dot_values(&source_primal.velocity, &source_adjoint.velocity)
            + dot_values(&source_primal.velocity_prev, &source_adjoint.velocity_prev)
            + dot_values(&source_primal.pressure, &source_adjoint.pressure);
        assert!((terminal_pairing - source_pairing).abs() < 1.0e-8);

        let error = transfer_pc_fan_cfd_transient_remesh_schedule_state_adjoint(
            &schedule,
            &[&source_mesh],
            &terminal_state,
            None,
        )
        .expect_err("missing target mesh must be rejected before transfer");
        assert!(matches!(
            error,
            PcFanCfdTransientStateTransferError::StateTransfer(
                PcFanCfdStateTransferError::InvalidMesh {
                    role: "mesh chain",
                    ..
                }
            )
        ));

        let mut invalid_terminal_state = terminal_state;
        invalid_terminal_state.velocity.pop();
        let error = transfer_pc_fan_cfd_transient_remesh_schedule_state_adjoint(
            &schedule,
            &[&source_mesh, &target_mesh],
            &invalid_terminal_state,
            None,
        )
        .expect_err("incompatible terminal adjoint dimensions must be rejected");
        assert!(matches!(
            error,
            PcFanCfdTransientStateTransferError::StateTransfer(
                PcFanCfdStateTransferError::TargetVelocityDofCount { .. }
            )
        ));
    }

    #[test]
    fn transported_remesher_provider_receives_each_accepted_stage() {
        let (mut config, design) = small_transient_config();
        config.cfd.turbulence = PcFanTurbulenceModel::KOmegaSstTransported;
        let geometry = config.build_geometry(&design);
        let domain = geometry.build_cfd_domain(&config);
        let source_mesh = box_mesh_tetrahedra_symmetric(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            2,
            2,
            2,
        );
        let target_meshes = [
            box_mesh_tetrahedra_symmetric(
                domain.min_corner.x,
                domain.max_corner.x,
                domain.min_corner.y,
                domain.max_corner.y,
                domain.min_corner.z,
                domain.max_corner.z,
                1,
                2,
                2,
            ),
            box_mesh_tetrahedra_symmetric(
                domain.min_corner.x,
                domain.max_corner.x,
                domain.min_corner.y,
                domain.max_corner.y,
                domain.min_corner.z,
                domain.max_corner.z,
                2,
                1,
                2,
            ),
        ];
        let velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
        let pressure_space = H1Space::new(PolynomialDegree::P1);
        let source_fem_state =
            sonium_fem::physics::fluid::stokes::time_stepping::BdfState::from_rest(
                velocity_space.num_global_dofs(&source_mesh),
                pressure_space.num_global_dofs(&source_mesh),
            );
        let source_state = PcFanCfdTransientBdfState::from_fem(&source_fem_state);
        let source_turbulence =
            initialize_pc_fan_cfd_komega_bdf_state(&source_mesh, 0.0, 0, 2.0e-6, 2.0)
                .expect("source transported history initialise");
        let mut observed = Vec::new();

        let schedule = solve_pc_fan_cfd_rotating_wall_transient_with_state_transfer_with_remesher(
            &config,
            &design,
            1.0e-3,
            1.0e-4,
            &source_mesh,
            &source_state,
            Some(&source_turbulence),
            target_meshes.len(),
            |stage, previous_mesh, state, turbulence| {
                observed.push((
                    stage,
                    state.time,
                    previous_mesh.num_elements(),
                    turbulence.is_some(),
                ));
                target_meshes
                    .get(stage)
                    .cloned()
                    .ok_or_else(|| format!("missing target mesh for stage {stage}"))
            },
        )
        .expect("stage-by-stage remesher provider should run");

        assert_eq!(observed.len(), 2);
        assert_eq!(observed[0].0, 0);
        assert_eq!(observed[0].1, 0.0);
        assert_eq!(observed[0].2, source_mesh.num_elements());
        assert!(observed[0].3);
        assert_eq!(observed[1].0, 1);
        assert!(observed[1].1 > observed[0].1);
        assert_eq!(observed[1].2, target_meshes[0].num_elements());
        assert!(observed[1].3);
        assert_eq!(schedule.stages.len(), 2);
        assert_eq!(
            schedule.provenance,
            "pcfan:external-remesh-provider-transient-schedule:v1"
        );

        let rejected = solve_pc_fan_cfd_rotating_wall_transient_with_state_transfer_with_remesher(
            &config,
            &design,
            1.0e-3,
            1.0e-4,
            &source_mesh,
            &source_state,
            Some(&source_turbulence),
            1,
            |_stage, _previous_mesh, _state, _turbulence| Err("target quality barrier".to_string()),
        );
        assert!(matches!(
            rejected,
            Err(PcFanCfdTransientStateTransferError::Remesher {
                stage: 0,
                reason
            }) if reason == "target quality barrier"
        ));
    }

    #[test]
    fn transported_remesh_runner_rejects_missing_or_nonphysical_history() {
        let (mut config, design) = small_transient_config();
        config.cfd.turbulence = PcFanTurbulenceModel::KOmegaSstTransported;
        let geometry = config.build_geometry(&design);
        let domain = geometry.build_cfd_domain(&config);
        let mesh = pc_fan_cfd_mesh_for_domain(&config, &domain);
        let velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
        let pressure_space = H1Space::new(PolynomialDegree::P1);
        let source_state = sonium_fem::physics::fluid::stokes::time_stepping::BdfState::from_rest(
            velocity_space.num_global_dofs(&mesh),
            pressure_space.num_global_dofs(&mesh),
        );

        let missing = solve_pc_fan_cfd_rotating_wall_transient_with_state_transfer_to_mesh(
            &config,
            &design,
            1.0e-3,
            1.0e-4,
            &mesh,
            &mesh,
            &source_state,
            None,
        );
        assert!(matches!(
            missing,
            Err(PcFanCfdTransientStateTransferError::Solve(
                PcFanCfdTransientError::InvalidInitialState { .. }
            ))
        ));

        let mut invalid = initialize_pc_fan_cfd_komega_bdf_state(&mesh, 0.0, 0, 1.0e-6, 1.0)
            .expect("history should initialise");
        invalid.k[0] = -1.0;
        let rejected = solve_pc_fan_cfd_rotating_wall_transient_with_state_transfer_to_mesh(
            &config,
            &design,
            1.0e-3,
            1.0e-4,
            &mesh,
            &mesh,
            &source_state,
            Some(&invalid),
        );
        assert!(matches!(
            rejected,
            Err(PcFanCfdTransientStateTransferError::KOmegaStateTransfer(_))
        ));
    }

    #[test]
    fn transient_convergence_rejects_invalid_grid_before_solving() {
        let (config, design) = small_transient_config();
        let mut convergence = PcFanCfdTransientConvergenceConfig::default();
        convergence.fine_time_step_s = convergence.coarse_time_step_s;
        assert!(matches!(
            try_solve_pc_fan_cfd_transient_convergence(&config, &design, convergence),
            Err(PcFanCfdTransientConvergenceConfigError::FineTimeStepNotSmaller { .. })
        ));

        let mut convergence = PcFanCfdTransientConvergenceConfig::default();
        convergence.relative_tolerance = f64::NAN;
        assert!(matches!(
            try_solve_pc_fan_cfd_transient_convergence(&config, &design, convergence),
            Err(PcFanCfdTransientConvergenceConfigError::InvalidRelativeTolerance(_))
        ));

        let mut convergence = PcFanCfdTransientConvergenceConfig::default();
        convergence.max_final_residual = f64::NAN;
        assert!(matches!(
            try_solve_pc_fan_cfd_transient_convergence(&config, &design, convergence),
            Err(PcFanCfdTransientConvergenceConfigError::InvalidFinalResidual(_))
        ));

        let mut convergence = PcFanCfdTransientConvergenceConfig::default();
        convergence.coarse_time_step_s = 3.0e-4;
        assert!(matches!(
            try_solve_pc_fan_cfd_transient_convergence(&config, &design, convergence),
            Err(PcFanCfdTransientConvergenceConfigError::NonIntegralTimeGrid { .. })
        ));
    }

    #[test]
    fn transient_convergence_admits_bounded_two_level_cpu_fixture() {
        let (config, design) = small_transient_config();
        let report = try_solve_pc_fan_cfd_transient_convergence(
            &config,
            &design,
            PcFanCfdTransientConvergenceConfig {
                coarse_time_step_s: 5.0e-4,
                fine_time_step_s: 2.5e-4,
                rotor_revolutions: 0.02,
                relative_tolerance: 2.0,
                max_final_residual: 2.0,
            },
        )
        .expect("bounded transient time-step fixture should run");
        assert_eq!(report.levels.len(), 2);
        assert!(report.all_levels_converged_and_finite);
        assert!(report.converged);
        assert!(report.airflow_relative_change.unwrap().is_finite());
        assert!(report.pressure_relative_change.unwrap().is_finite());
        assert!(report.axial_loading_relative_change.unwrap().is_finite());
        assert!(report.final_time_relative_change.unwrap().is_finite());
        assert_eq!(report.levels[0].observation.final_time_s, 0.002);
        assert_eq!(report.levels[1].observation.final_time_s, 0.002);
    }

    #[test]
    fn transient_restart_routes_invalid_bdf_state_before_solving() {
        let (config, design) = small_transient_config();
        let source_mesh = sonium_fem::mesh::unit_cube_tetrahedra(1);
        let velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
        let pressure_space = H1Space::new(PolynomialDegree::P1);
        let mut source_state =
            sonium_fem::physics::fluid::stokes::time_stepping::BdfState::from_rest(
                velocity_space.num_global_dofs(&source_mesh),
                pressure_space.num_global_dofs(&source_mesh),
            );
        source_state.time = f64::NAN;

        assert!(matches!(
            solve_pc_fan_cfd_rotating_wall_transient_with_state_transfer(
                &config,
                &design,
                5.0e-4,
                0.02,
                &source_mesh,
                &source_state,
            ),
            Err(PcFanCfdTransientStateTransferError::StateTransfer(
                PcFanCfdStateTransferError::NonFiniteSourceTime
            ))
        ));
    }
}

#[cfg(test)]
mod transient_objective_tests {
    use super::*;

    #[test]
    fn transient_state_adjoint_reports_actual_step_backend() {
        let (config, _design, trajectory) = fixture();
        let objective_gradients = trajectory
            .steps
            .iter()
            .map(|step| vec![0.0; step.state.len()])
            .collect::<Vec<_>>();

        let report = solve_pc_fan_cfd_resolved_transient_state_adjoint(
            &PcFanResolvedAdjointConfig::default(),
            &config,
            &trajectory,
            &objective_gradients,
        )
        .expect("the finite CPU trajectory should be admissible");

        assert_eq!(report.backend, PcFanCfdBackend::Cpu);

        let (config, _design, mut forged_trajectory) = fixture();
        forged_trajectory
            .steps
            .first_mut()
            .expect("fixture must retain one transient step")
            .actual_backend = PcFanCfdBackend::Cuda;
        let objective_gradients = forged_trajectory
            .steps
            .iter()
            .map(|step| vec![0.0; step.state.len()])
            .collect::<Vec<_>>();
        assert!(matches!(
            solve_pc_fan_cfd_resolved_transient_state_adjoint(
                &PcFanResolvedAdjointConfig::default(),
                &config,
                &forged_trajectory,
                &objective_gradients,
            ),
            Err(PcFanResolvedAdjointError::DimensionMismatch(
                "transient backend provenance"
            ))
        ));
    }

    #[test]
    fn transported_turbulence_transient_adjoint_remains_fail_closed() {
        let (mut config, _design, trajectory) = fixture();
        config.cfd.turbulence = PcFanTurbulenceModel::KOmegaSstTransported;
        let objective_gradients = trajectory
            .steps
            .iter()
            .map(|step| vec![0.0; step.state.len()])
            .collect::<Vec<_>>();

        assert!(matches!(
            solve_pc_fan_cfd_resolved_transient_state_adjoint(
                &PcFanResolvedAdjointConfig::default(),
                &config,
                &trajectory,
                &objective_gradients,
            ),
            Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible)
        ));
    }

    #[test]
    fn coupled_transient_adjoint_requires_retained_turbulence_blocks() {
        let (mut config, _design, trajectory) = fixture();
        config.cfd.turbulence = PcFanTurbulenceModel::KOmegaSstTransported;
        let objective_gradients = trajectory
            .steps
            .iter()
            .map(|step| vec![0.0; step.state.len()])
            .collect::<Vec<_>>();
        assert!(matches!(
            solve_pc_fan_cfd_resolved_transient_turbulent_state_adjoint(
                &PcFanResolvedAdjointConfig::default(),
                &config,
                &trajectory,
                &objective_gradients,
            ),
            Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible)
        ));
    }

    #[test]
    fn coupled_transient_adjoint_uses_square_retained_block() {
        let (mut config, _design, mut trajectory) = fixture();
        config.cfd.turbulence = PcFanTurbulenceModel::KOmegaSstTransported;
        let step = trajectory.steps.first_mut().expect("fixture step");
        step.state_jacobian_provenance = PC_FAN_TRANSPORTED_MONOLITHIC_BDF_NEWTON_PROVENANCE.into();
        let flow_dimension = step.state.len();
        let velocity_dimension = step.bdf_state.velocity.len();
        let scalar_dimension = 2;
        let identity = |dimension: usize| {
            sonium_solvers::CsrMatrix::from_triplets(
                dimension,
                dimension,
                (0..dimension).map(|index| (index, index, 1.0)).collect(),
            )
        };
        let zero = |rows: usize, columns: usize| {
            sonium_solvers::CsrMatrix::from_triplets(rows, columns, Vec::new())
        };
        step.transported_turbulence = Some(PcFanCfdKOmegaBdfState {
            k: vec![1.0; scalar_dimension],
            k_prev: vec![1.0; scalar_dimension],
            omega: vec![2.0; scalar_dimension],
            omega_prev: vec![2.0; scalar_dimension],
            time: 1.0,
            step: 1,
        });
        step.transported_turbulence_linearization = Some(KOmegaTransportLinearization {
            provenance:
                sonium_fem::physics::fluid::turbulence::K_OMEGA_TRANSPORT_LINEARIZATION_PROVENANCE,
            k_state_jacobian: identity(scalar_dimension),
            omega_state_jacobian: identity(scalar_dimension),
            k_history_jacobian: zero(scalar_dimension, scalar_dimension),
            k_previous_history_jacobian: zero(scalar_dimension, scalar_dimension),
            omega_history_jacobian: zero(scalar_dimension, scalar_dimension),
            omega_previous_history_jacobian: zero(scalar_dimension, scalar_dimension),
        });
        step.transported_turbulence_coupled_linearization = Some(PcFanKOmegaCoupledLinearization {
            provenance:
                crate::transported_turbulence::PC_FAN_KOMEGA_COUPLED_LINEARIZATION_PROVENANCE,
            momentum_wrt_k: zero(velocity_dimension, scalar_dimension),
            momentum_wrt_omega: zero(velocity_dimension, scalar_dimension),
            k_wrt_velocity: zero(scalar_dimension, velocity_dimension),
            omega_wrt_velocity: zero(scalar_dimension, velocity_dimension),
            k_wrt_k: zero(scalar_dimension, scalar_dimension),
            k_wrt_omega: zero(scalar_dimension, scalar_dimension),
            omega_wrt_k: zero(scalar_dimension, scalar_dimension),
            omega_wrt_omega: zero(scalar_dimension, scalar_dimension),
        });
        step.transported_turbulence_residual =
            Some((vec![0.0; scalar_dimension], vec![0.0; scalar_dimension]));
        step.transported_turbulence_residual_design_jacobian =
            Some(vec![vec![0.0; DESIGN_VARIABLE_COUNT]; 2 * scalar_dimension]);
        let full_dimension = flow_dimension + 2 * scalar_dimension;
        let mut objective_gradient = vec![0.0; full_dimension];
        objective_gradient[0] = 1.0;

        let report = solve_pc_fan_cfd_resolved_transient_turbulent_state_adjoint(
            &PcFanResolvedAdjointConfig::default(),
            &config,
            &trajectory,
            &[objective_gradient],
        )
        .expect("identity coupled tangent should admit the turbulent adjoint");
        assert_eq!(report.adjoints.len(), 1);
        assert_eq!(report.adjoints[0].len(), full_dimension);
        assert!(report.converged);

        let objective_report = solve_pc_fan_cfd_resolved_transient_objective_adjoint(
            &PcFanResolvedAdjointConfig::default(),
            &config,
            &_design,
            &trajectory,
            PcFanResolvedTransientObjective::Airflow,
        )
        .expect("transported objective adjoint should dispatch to full state");
        assert_eq!(objective_report.adjoints.len(), 1);
        assert_eq!(objective_report.adjoints[0].len(), full_dimension);
        assert!(objective_report.converged);
    }

    #[test]
    fn transported_turbulent_adjoint_matches_manufactured_two_step_design_gradient() {
        let (mut config, _design, mut trajectory) = fixture();
        config.cfd.turbulence = PcFanTurbulenceModel::KOmegaSstTransported;

        let flow_dimension = trajectory.steps[0].state.len();
        let velocity_dimension = trajectory.steps[0].bdf_state.velocity.len();
        let scalar_dimension = 2;
        let full_dimension = flow_dimension + 2 * scalar_dimension;
        let diagonal = |dimension: usize, value: f64| {
            sonium_solvers::CsrMatrix::from_triplets(
                dimension,
                dimension,
                (0..dimension).map(|index| (index, index, value)).collect(),
            )
        };
        let zero = |rows: usize, columns: usize| {
            sonium_solvers::CsrMatrix::from_triplets(rows, columns, Vec::new())
        };
        let scalar_linearization = || KOmegaTransportLinearization {
            provenance:
                sonium_fem::physics::fluid::turbulence::K_OMEGA_TRANSPORT_LINEARIZATION_PROVENANCE,
            k_state_jacobian: diagonal(scalar_dimension, 1.0),
            omega_state_jacobian: diagonal(scalar_dimension, 1.0),
            k_history_jacobian: diagonal(scalar_dimension, 0.2),
            k_previous_history_jacobian: zero(scalar_dimension, scalar_dimension),
            omega_history_jacobian: diagonal(scalar_dimension, 0.3),
            omega_previous_history_jacobian: zero(scalar_dimension, scalar_dimension),
        };
        let coupled_linearization = || PcFanKOmegaCoupledLinearization {
            provenance:
                crate::transported_turbulence::PC_FAN_KOMEGA_COUPLED_LINEARIZATION_PROVENANCE,
            momentum_wrt_k: zero(velocity_dimension, scalar_dimension),
            momentum_wrt_omega: zero(velocity_dimension, scalar_dimension),
            k_wrt_velocity: zero(scalar_dimension, velocity_dimension),
            omega_wrt_velocity: zero(scalar_dimension, velocity_dimension),
            k_wrt_k: zero(scalar_dimension, scalar_dimension),
            k_wrt_omega: zero(scalar_dimension, scalar_dimension),
            omega_wrt_k: zero(scalar_dimension, scalar_dimension),
            omega_wrt_omega: zero(scalar_dimension, scalar_dimension),
        };
        let turbulence_state = PcFanCfdKOmegaBdfState {
            k: vec![1.0; scalar_dimension],
            k_prev: vec![1.0; scalar_dimension],
            omega: vec![2.0; scalar_dimension],
            omega_prev: vec![2.0; scalar_dimension],
            time: 1.0,
            step: 1,
        };
        let configure_step = |step: &mut PcFanCfdTransientTrajectoryStep| {
            step.state_jacobian_provenance =
                PC_FAN_TRANSPORTED_MONOLITHIC_BDF_NEWTON_PROVENANCE.into();
            step.transported_turbulence = Some(turbulence_state.clone());
            step.transported_turbulence_linearization = Some(scalar_linearization());
            step.transported_turbulence_coupled_linearization = Some(coupled_linearization());
            step.transported_turbulence_residual =
                Some((vec![0.0; scalar_dimension], vec![0.0; scalar_dimension]));
            step.transported_turbulence_residual_design_jacobian =
                Some(vec![vec![0.0; DESIGN_VARIABLE_COUNT]; 2 * scalar_dimension]);
        };

        let mut first = trajectory.steps[0].clone();
        let mut second = first.clone();
        configure_step(&mut first);
        configure_step(&mut second);
        first.step = 0;
        first.time_s = 0.0;
        first.bdf_state.step = 0;
        second.step = 1;
        second.time_s = 1.0;
        second.bdf_state.step = 1;
        second.history_jacobians = vec![PcFanResolvedTransientHistoryJacobian {
            previous_step: 0,
            matrix: PcFanResolvedFlowSparseMatrix::new(
                flow_dimension,
                (0..=flow_dimension).collect(),
                (0..flow_dimension).collect(),
                vec![0.4; flow_dimension],
            )
            .expect("manufactured flow history matrix"),
        }];
        trajectory.steps = vec![first, second];

        let mut residual_design_jacobians = vec![
            vec![vec![0.0; DESIGN_VARIABLE_COUNT]; full_dimension],
            vec![vec![0.0; DESIGN_VARIABLE_COUNT]; full_dimension],
        ];
        residual_design_jacobians[0][0][0] = 0.7;
        residual_design_jacobians[0][flow_dimension][0] = -0.2;
        residual_design_jacobians[0][flow_dimension + scalar_dimension][0] = 0.35;
        residual_design_jacobians[1][0][0] = 0.1;
        residual_design_jacobians[1][flow_dimension][0] = 0.4;
        residual_design_jacobians[1][flow_dimension + scalar_dimension][0] = -0.6;

        let mut objective_state_gradients = vec![vec![0.0; full_dimension]; 2];
        objective_state_gradients[0][0] = 0.3;
        objective_state_gradients[0][flow_dimension] = 0.5;
        objective_state_gradients[0][flow_dimension + scalar_dimension] = -0.2;
        objective_state_gradients[1][0] = 0.8;
        objective_state_gradients[1][flow_dimension] = -0.25;
        objective_state_gradients[1][flow_dimension + scalar_dimension] = 0.6;

        let mut objective_design_gradients = vec![vec![0.0; DESIGN_VARIABLE_COUNT]; 2];
        objective_design_gradients[0][0] = 1.25;
        let report = solve_pc_fan_cfd_resolved_transient_turbulent_adjoint_with_derivatives(
            &PcFanResolvedAdjointConfig::default(),
            &config,
            &trajectory,
            &objective_state_gradients,
            Some(&residual_design_jacobians),
            Some(&objective_design_gradients),
        )
        .expect("manufactured transported trajectory should admit its exact adjoint");
        assert!(report.converged);
        assert_eq!(report.adjoints.len(), 2);
        assert!(
            report
                .adjoints
                .iter()
                .all(|adjoint| adjoint.len() == full_dimension)
        );

        let d0 = residual_design_jacobians[0]
            .iter()
            .map(|row| row[0])
            .collect::<Vec<_>>();
        let d1 = residual_design_jacobians[1]
            .iter()
            .map(|row| row[0])
            .collect::<Vec<_>>();
        let mut history_scale = vec![0.4; full_dimension];
        for scale in &mut history_scale[flow_dimension..flow_dimension + scalar_dimension] {
            *scale = 0.2;
        }
        for scale in &mut history_scale[flow_dimension + scalar_dimension..] {
            *scale = 0.3;
        }
        let objective_value = |design_value: f64| {
            let first_state = d0
                .iter()
                .map(|derivative| -derivative * design_value)
                .collect::<Vec<_>>();
            let second_state = first_state
                .iter()
                .zip(&history_scale)
                .zip(&d1)
                .map(|((&state, &history), &derivative)| {
                    -(history * state + derivative * design_value)
                })
                .collect::<Vec<_>>();
            objective_state_gradients[0]
                .iter()
                .zip(&first_state)
                .map(|(&gradient, &state)| gradient * state)
                .sum::<f64>()
                + objective_state_gradients[1]
                    .iter()
                    .zip(&second_state)
                    .map(|(&gradient, &state)| gradient * state)
                    .sum::<f64>()
                + 1.25 * design_value
        };
        let epsilon = 1.0e-6;
        let finite_difference =
            (objective_value(epsilon) - objective_value(-epsilon)) / (2.0 * epsilon);
        assert!((report.design_gradient[0] - finite_difference).abs() < 1.0e-9);
    }

    #[test]
    fn transient_adjoint_rejects_step_residual_above_forward_budget() {
        let (config, _design, mut trajectory) = fixture();
        let tolerance = config.cfd.nonlinear_tolerance;
        trajectory
            .steps
            .first_mut()
            .expect("fixture must retain one transient step")
            .residual_norm = tolerance * 2.0;
        let objective_gradients = trajectory
            .steps
            .iter()
            .map(|step| vec![0.0; step.state.len()])
            .collect::<Vec<_>>();

        assert!(matches!(
            solve_pc_fan_cfd_resolved_transient_state_adjoint(
                &PcFanResolvedAdjointConfig::default(),
                &config,
                &trajectory,
                &objective_gradients,
            ),
            Err(PcFanResolvedAdjointError::ProductionStateNotConverged)
        ));
    }

    #[test]
    fn transient_adjoint_rejects_step_conservation_above_forward_budget() {
        let (config, _design, mut trajectory) = fixture();
        trajectory
            .steps
            .first_mut()
            .expect("fixture must retain one transient step")
            .conservation
            .relative_mass_imbalance = config.cfd.max_relative_mass_imbalance * 2.0;
        let objective_gradients = trajectory
            .steps
            .iter()
            .map(|step| vec![0.0; step.state.len()])
            .collect::<Vec<_>>();

        assert!(matches!(
            solve_pc_fan_cfd_resolved_transient_state_adjoint(
                &PcFanResolvedAdjointConfig::default(),
                &config,
                &trajectory,
                &objective_gradients,
            ),
            Err(PcFanResolvedAdjointError::ProductionStateNotConverged)
        ));
    }

    #[test]
    fn transient_adjoint_rejects_final_conservation_above_forward_budget() {
        let (config, _design, mut trajectory) = fixture();
        trajectory
            .observation
            .conservation
            .relative_energy_imbalance = config.cfd.max_relative_energy_imbalance * 2.0;
        let objective_gradients = trajectory
            .steps
            .iter()
            .map(|step| vec![0.0; step.state.len()])
            .collect::<Vec<_>>();

        assert!(matches!(
            solve_pc_fan_cfd_resolved_transient_state_adjoint(
                &PcFanResolvedAdjointConfig::default(),
                &config,
                &trajectory,
                &objective_gradients,
            ),
            Err(PcFanResolvedAdjointError::ProductionStateNotConverged)
        ));
    }

    #[test]
    fn transient_adjoint_rejects_nonfinite_retained_state_or_ledger() {
        let (config, _design, mut trajectory) = fixture();
        trajectory
            .steps
            .first_mut()
            .expect("fixture must retain one transient step")
            .state[0] = f64::NAN;
        let objective_gradients = trajectory
            .steps
            .iter()
            .map(|step| vec![0.0; step.state.len()])
            .collect::<Vec<_>>();
        assert!(matches!(
            solve_pc_fan_cfd_resolved_transient_state_adjoint(
                &PcFanResolvedAdjointConfig::default(),
                &config,
                &trajectory,
                &objective_gradients,
            ),
            Err(PcFanResolvedAdjointError::ProductionStateNotConverged)
        ));

        let (config, _design, mut trajectory) = fixture();
        trajectory
            .steps
            .first_mut()
            .expect("fixture must retain one transient step")
            .conservation
            .pressure_power_w = f64::INFINITY;
        let objective_gradients = trajectory
            .steps
            .iter()
            .map(|step| vec![0.0; step.state.len()])
            .collect::<Vec<_>>();
        assert!(matches!(
            solve_pc_fan_cfd_resolved_transient_state_adjoint(
                &PcFanResolvedAdjointConfig::default(),
                &config,
                &trajectory,
                &objective_gradients,
            ),
            Err(PcFanResolvedAdjointError::ProductionStateNotConverged)
        ));
    }

    #[test]
    fn transient_adjoint_rejects_transported_scalar_residual_above_forward_budget() {
        let (config, _design, mut trajectory) = fixture();
        let tolerance = config.cfd.nonlinear_tolerance;
        trajectory
            .steps
            .first_mut()
            .expect("fixture must retain one transient step")
            .transported_turbulence_residual = Some((vec![tolerance * 2.0], vec![0.0]));
        let objective_gradients = trajectory
            .steps
            .iter()
            .map(|step| vec![0.0; step.state.len()])
            .collect::<Vec<_>>();

        assert!(matches!(
            solve_pc_fan_cfd_resolved_transient_state_adjoint(
                &PcFanResolvedAdjointConfig::default(),
                &config,
                &trajectory,
                &objective_gradients,
            ),
            Err(PcFanResolvedAdjointError::ProductionStateNotConverged)
        ));
    }

    #[test]
    fn transient_state_adjoint_rejects_non_cfd_flow_model() {
        let (mut config, _design, trajectory) = fixture();
        config.cfd.flow_model = PcFanFlowModel::ReducedOrder;
        let objective_gradients = trajectory
            .steps
            .iter()
            .map(|step| vec![0.0; step.state.len()])
            .collect::<Vec<_>>();

        assert!(matches!(
            solve_pc_fan_cfd_resolved_transient_state_adjoint(
                &PcFanResolvedAdjointConfig::default(),
                &config,
                &trajectory,
                &objective_gradients,
            ),
            Err(PcFanResolvedAdjointError::UnsupportedFlowModel(
                PcFanFlowModel::ReducedOrder
            ))
        ));
    }

    #[test]
    fn transient_state_adjoint_rejects_algebraic_turbulence_without_bdf_tangent() {
        let (mut config, _design, trajectory) = fixture();
        config.cfd.turbulence = PcFanTurbulenceModel::KOmegaSstAlgebraic;
        let objective_gradients = trajectory
            .steps
            .iter()
            .map(|step| vec![0.0; step.state.len()])
            .collect::<Vec<_>>();

        assert!(matches!(
            solve_pc_fan_cfd_resolved_transient_state_adjoint(
                &PcFanResolvedAdjointConfig::default(),
                &config,
                &trajectory,
                &objective_gradients,
            ),
            Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible)
        ));
    }

    #[test]
    fn transient_state_adjoint_admits_exact_algebraic_bdf_tangent() {
        let (mut config, _design, mut trajectory) = fixture();
        config.cfd.turbulence = PcFanTurbulenceModel::KOmegaSstAlgebraic;
        for step in &mut trajectory.steps {
            step.state_jacobian_provenance =
                "fem:navier-stokes:bdf-newton-algebraic-turbulence:v1".into();
        }
        let objective_gradients = trajectory
            .steps
            .iter()
            .map(|step| vec![0.0; step.state.len()])
            .collect::<Vec<_>>();

        let report = solve_pc_fan_cfd_resolved_transient_state_adjoint(
            &PcFanResolvedAdjointConfig::default(),
            &config,
            &trajectory,
            &objective_gradients,
        )
        .expect("exact algebraic BDF tangent should be admitted");
        assert!(report.converged);
    }

    #[test]
    fn transient_step_admission_requires_solver_success_and_residual_budget() {
        assert!(transient_step_is_admitted(true, 1.0e-5, 1.0e-4, true, true));
        assert!(!transient_step_is_admitted(
            false, 1.0e-5, 1.0e-4, true, true
        ));
        assert!(!transient_step_is_admitted(
            true, 2.0e-4, 1.0e-4, true, true
        ));
        assert!(!transient_step_is_admitted(
            true,
            f64::NAN,
            1.0e-4,
            true,
            true
        ));
        assert!(!transient_step_is_admitted(
            true, 1.0e-5, 1.0e-4, false, true
        ));
        assert!(!transient_step_is_admitted(
            true, 1.0e-5, 1.0e-4, true, false
        ));
    }

    fn zero_step_conservation() -> PcFanCfdConservation {
        PcFanCfdConservation {
            inlet_mass_flow_kg_s: 0.0,
            outlet_mass_flow_kg_s: 0.0,
            relative_mass_imbalance: 0.0,
            inlet_axial_momentum_flux_n: 0.0,
            outlet_axial_momentum_flux_n: 0.0,
            pressure_force_n: 0.0,
            body_force_n: 0.0,
            axial_momentum_residual_n: 0.0,
            relative_momentum_imbalance: 0.0,
            pressure_force_from_loading_n: 0.0,
            relative_pressure_closure: 0.0,
            pressure_power_w: 0.0,
            shaft_power_to_fluid_w: 0.0,
            relative_energy_imbalance: 0.0,
        }
    }

    fn fixture() -> (
        PcFanConfig,
        PcFanDesignVariables,
        PcFanCfdTransientTrajectory,
    ) {
        let mut config = PcFanConfig::default_120mm();
        config.cfd.flow_model = PcFanFlowModel::IncompressibleCfdFull;
        config.cfd.backend = PcFanCfdBackend::Cpu;
        config.cfd.cells_x = 2;
        config.cfd.cells_y = 2;
        config.cfd.cells_z = 2;
        let design = config.default_design_variables();
        let geometry = config.build_geometry(&design);
        let domain = geometry.build_cfd_domain(&config);
        let mesh = box_mesh_tetrahedra_symmetric(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            2,
            2,
            2,
        );
        let velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
        let pressure_space = H1Space::new(PolynomialDegree::P1);
        let velocity_dimension = velocity_space.num_global_dofs(&mesh);
        let pressure_dimension = pressure_space.num_global_dofs(&mesh);
        let state_dimension = velocity_dimension + pressure_dimension;
        let mut state = vec![0.0; state_dimension];
        for node in 0..mesh.num_nodes() {
            state[3 * node + 2] = 1.0;
        }
        let state_jacobian = PcFanResolvedFlowSparseMatrix::new(
            state_dimension,
            (0..=state_dimension).collect(),
            (0..state_dimension).collect(),
            vec![1.0; state_dimension],
        )
        .expect("identity CSR fixture");
        let step = PcFanCfdTransientTrajectoryStep {
            step: 0,
            time_s: 1.0,
            state: state.clone(),
            state_jacobian,
            state_jacobian_provenance: "fixture:resolved-transient-bdf-newton:v1".into(),
            actual_backend: PcFanCfdBackend::Cpu,
            backend_provenance: PcFanCfdBackendProvenance::from_dispatch(
                PcFanCfdBackend::Cpu,
                PcFanCfdBackend::Cpu,
            ),
            execution_environment: PcFanCfdExecutionEnvironment::capture(
                &config.cfd,
                PcFanCfdBackendProvenance::from_dispatch(
                    PcFanCfdBackend::Cpu,
                    PcFanCfdBackend::Cpu,
                ),
            ),
            rhs: state,
            state_residual: vec![0.0; state_dimension],
            residual_norm: 0.0,
            conservation: zero_step_conservation(),
            history_jacobians: Vec::new(),
            transported_turbulence: None,
            transported_eddy_viscosity_per_element: None,
            transported_turbulence_linearization: None,
            transported_turbulence_coupled_linearization: None,
            transported_turbulence_residual: None,
            transported_turbulence_residual_design_jacobian: None,
            bdf_state: PcFanCfdTransientBdfState {
                velocity: vec![0.0; velocity_dimension],
                velocity_prev: vec![0.0; velocity_dimension],
                pressure: vec![0.0; pressure_dimension],
                time: 1.0,
                step: 0,
            },
            topology_transfer_adjoint_admissible: true,
            residual_design_jacobian: vec![vec![0.0; DESIGN_VARIABLE_COUNT]; state_dimension],
        };
        let conservation = PcFanCfdConservation {
            inlet_mass_flow_kg_s: 0.0,
            outlet_mass_flow_kg_s: 0.0,
            relative_mass_imbalance: 0.0,
            inlet_axial_momentum_flux_n: 0.0,
            outlet_axial_momentum_flux_n: 0.0,
            pressure_force_n: 0.0,
            body_force_n: 0.0,
            axial_momentum_residual_n: 0.0,
            relative_momentum_imbalance: 0.0,
            pressure_force_from_loading_n: 0.0,
            relative_pressure_closure: 0.0,
            pressure_power_w: 0.0,
            shaft_power_to_fluid_w: 0.0,
            relative_energy_imbalance: 0.0,
        };
        (
            config,
            design,
            PcFanCfdTransientTrajectory {
                observation: PcFanCfdTransientObservation {
                    final_time_s: 1.0,
                    airflow_cfm: 1.0,
                    pressure_drop_pa: 1.0,
                    axial_loading_n: 1.0,
                    residual: 0.0,
                    conservation,
                    converged: true,
                },
                steps: vec![step],
                stage_initial_history_jacobians: Vec::new(),
            },
        )
    }

    fn remesh_objective_schedule_fixture(
        trajectory: &PcFanCfdTransientTrajectory,
        mesh: &sonium_fem::mesh::Mesh,
        stage_count: usize,
    ) -> PcFanCfdTransientRemeshSchedule {
        let velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
        let pressure_space = H1Space::new(PolynomialDegree::P1);
        let report = PcFanCfdP2P1StateTransferReport {
            source_node_count: mesh.num_nodes(),
            target_node_count: mesh.num_nodes(),
            source_element_count: mesh.num_elements(),
            target_element_count: mesh.num_elements(),
            source_velocity_dof_count: velocity_space.num_global_dofs(mesh),
            target_velocity_dof_count: velocity_space.num_global_dofs(mesh),
            source_pressure_dof_count: pressure_space.num_global_dofs(mesh),
            target_pressure_dof_count: pressure_space.num_global_dofs(mesh),
        };
        PcFanCfdTransientRemeshSchedule {
            stages: (0..stage_count).map(|_| trajectory.clone()).collect(),
            mean_flow_transfer_reports: vec![report; stage_count],
            transported_turbulence_transfer_reports: None,
            requires_operator_reassembly: true,
            topology_transfer_adjoint_admissible: false,
            provenance: "pcfan:external-remesh-transient-schedule:v1",
        }
    }

    fn add_identity_startup_history(trajectory: &mut PcFanCfdTransientTrajectory) {
        let state_dimension = trajectory.steps[0].state.len();
        trajectory.stage_initial_history_jacobians =
            vec![PcFanResolvedTransientStageInitialHistoryJacobian {
                stage: 0,
                history_slot: 0,
                matrix: trajectory.steps[0].state_jacobian.clone(),
                objective_gradient: vec![0.0; state_dimension],
            }];
        if trajectory.steps.len() >= 2 {
            trajectory.stage_initial_history_jacobians.push(
                PcFanResolvedTransientStageInitialHistoryJacobian {
                    stage: 1,
                    history_slot: 1,
                    matrix: trajectory.steps[1].state_jacobian.clone(),
                    objective_gradient: vec![0.0; state_dimension],
                },
            );
        }
    }

    #[test]
    fn remesh_state_objective_adjoint_composes_startup_current_state() {
        let (config, _design, trajectory) = fixture();
        let geometry = config.build_geometry(&_design);
        let domain = geometry.build_cfd_domain(&config);
        let mesh = box_mesh_tetrahedra_symmetric(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            2,
            2,
            2,
        );
        let mut first = trajectory.clone();
        add_identity_startup_history(&mut first);
        let schedule = remesh_objective_schedule_fixture(&first, &mesh, 1);
        let state_dimension = first.steps[0].state.len();
        let result = solve_pc_fan_cfd_resolved_transient_remesh_schedule_state_objective_adjoint(
            &PcFanResolvedAdjointConfig::default(),
            &config,
            &PcFanCfdTransientRemeshSchedule {
                stages: vec![first],
                ..schedule
            },
            &[&mesh, &mesh],
            &[vec![vec![1.0; state_dimension]]],
        )
        .expect("one-stage remesh objective adjoint should solve");

        assert!(result.fixed_map_transpose_admissible);
        assert!(!result.topology_design_adjoint_admissible);
        assert_eq!(result.stages.len(), 1);
        assert!(
            result.stages[0].report.adjoints[0]
                .iter()
                .all(|value| (*value - 1.0).abs() < 1.0e-10)
        );
        assert_eq!(result.stages[0].initial_history_adjoints.len(), 1);
        assert!(
            result.stages[0].initial_history_adjoints[0]
                .adjoint
                .iter()
                .all(|value| (*value + 1.0).abs() < 1.0e-10)
        );
    }

    #[test]
    fn remesh_objective_adjoint_contracts_fixed_map_design_derivatives() {
        let (config, _design, trajectory) = fixture();
        let geometry = config.build_geometry(&_design);
        let domain = geometry.build_cfd_domain(&config);
        let mesh = box_mesh_tetrahedra_symmetric(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            2,
            2,
            2,
        );
        let mut stage = trajectory;
        add_identity_startup_history(&mut stage);
        let state_dimension = stage.steps[0].state.len();
        let schedule = remesh_objective_schedule_fixture(&stage, &mesh, 1);
        let mut residual_design = vec![vec![vec![0.0; DESIGN_VARIABLE_COUNT]; state_dimension]];
        residual_design[0][0][0] = 2.0;
        let objective_design = vec![vec![3.0; DESIGN_VARIABLE_COUNT]];

        let result =
            solve_pc_fan_cfd_resolved_transient_remesh_schedule_objective_adjoint_with_derivatives(
                &PcFanResolvedAdjointConfig::default(),
                &config,
                &PcFanCfdTransientRemeshSchedule {
                    stages: vec![stage],
                    ..schedule
                },
                &[&mesh, &mesh],
                &[vec![vec![1.0; state_dimension]]],
                "production:pcfan:fixed-map-design-residual:v1",
                &[residual_design],
                &[objective_design],
            )
            .expect("fixed-map remesh design adjoint should solve");

        assert_eq!(result.design_gradient.len(), DESIGN_VARIABLE_COUNT);
        assert!((result.design_gradient[0] - 1.0).abs() < 1.0e-10);
        assert!(
            result.design_gradient[1..]
                .iter()
                .all(|value| (*value - 3.0).abs() < 1.0e-10)
        );
        assert_eq!(
            result.stages[0].report.design_gradient,
            result.design_gradient
        );
        assert!(!result.topology_design_adjoint_admissible);
        assert_eq!(
            result.provenance,
            "pcfan:transient-remesh-fixed-map-objective-adjoint-with-design-derivatives:v1"
        );
    }

    #[test]
    fn remesh_map_design_derivative_contracts_incoming_history_adjoint() {
        let (config, _design, mut stage) = fixture();
        let geometry = config.build_geometry(&_design);
        let domain = geometry.build_cfd_domain(&config);
        let mesh = box_mesh_tetrahedra_symmetric(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            2,
            2,
            2,
        );
        let first_step = stage.steps[0].clone();
        stage.steps.push(PcFanCfdTransientTrajectoryStep {
            step: 1,
            time_s: 2.0,
            ..first_step
        });
        add_identity_startup_history(&mut stage);
        let velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
        let pressure_space = H1Space::new(PolynomialDegree::P1);
        let velocity_dimension = velocity_space.num_global_dofs(&mesh);
        let pressure_dimension = pressure_space.num_global_dofs(&mesh);
        let state_dimension = velocity_dimension + pressure_dimension;
        let mut source_state = stage.steps[0].bdf_state.clone();
        source_state.velocity[0] = 2.0;
        source_state.velocity_prev[0] = 4.0;
        source_state.pressure[0] = 3.0;
        let current_velocity_map_design_derivatives = vec![PcFanCfdTransientRemeshMapDesignEntry {
            target_dof: 0,
            source_dof: 0,
            design_index: 0,
            derivative: 1.0,
        }];
        let previous_velocity_map_design_derivatives =
            current_velocity_map_design_derivatives.clone();
        let pressure_map_design_derivatives = vec![PcFanCfdTransientRemeshMapDesignEntry {
            target_dof: 0,
            source_dof: 0,
            design_index: 0,
            derivative: 1.0,
        }];
        let derivative = PcFanCfdTransientRemeshMapDesignDerivative {
            source_state,
            current_velocity_map_design_derivatives,
            previous_velocity_map_design_derivatives,
            pressure_map_design_derivatives,
            provenance: "production:pcfan:analytic-remesh-map:v1".into(),
        };
        let schedule = remesh_objective_schedule_fixture(&stage, &mesh, 1);
        let result =
            solve_pc_fan_cfd_resolved_transient_remesh_schedule_objective_adjoint_with_map_design_derivatives(
                &PcFanResolvedAdjointConfig::default(),
                &config,
                &schedule,
                &[&mesh, &mesh],
                &[
                    vec![vec![1.0; state_dimension], vec![1.0; state_dimension]],
                ],
                "production:pcfan:analytic-stage-derivatives:v1",
                &[
                    vec![
                        vec![vec![0.0; DESIGN_VARIABLE_COUNT]; state_dimension],
                        vec![vec![0.0; DESIGN_VARIABLE_COUNT]; state_dimension],
                    ],
                ],
                &[vec![vec![0.0; DESIGN_VARIABLE_COUNT]; 2]],
                &[derivative],
            )
            .expect("analytic remesh-map derivative should contract");
        assert!((result.remesh_map_design_gradient[0] + 12.0).abs() < 1.0e-10);
        assert!(
            result.remesh_map_design_gradient[1..]
                .iter()
                .all(|value| value.abs() < 1.0e-12)
        );
        assert_eq!(result.design_gradient, result.remesh_map_design_gradient);
        assert_eq!(
            result.stages[0].report.design_gradient,
            result.remesh_map_design_gradient
        );
        assert!(result.fixed_map_transpose_admissible);
        assert!(result.interpolation_map_design_adjoint_admissible);
        assert!(!result.topology_design_adjoint_admissible);
        assert_eq!(
            result.provenance,
            "pcfan:transient-remesh-fixed-map-objective-adjoint-with-map-design-derivatives:v1"
        );
    }

    #[test]
    fn remesh_map_design_derivative_rejects_diagnostic_provenance() {
        let (config, _design, mut stage) = fixture();
        let geometry = config.build_geometry(&_design);
        let domain = geometry.build_cfd_domain(&config);
        let mesh = box_mesh_tetrahedra_symmetric(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            2,
            2,
            2,
        );
        add_identity_startup_history(&mut stage);
        let velocity_dimension = VectorH1Space::new(PolynomialDegree::P2, 3).num_global_dofs(&mesh);
        let pressure_dimension = H1Space::new(PolynomialDegree::P1).num_global_dofs(&mesh);
        let mut derivative = PcFanCfdTransientRemeshMapDesignDerivative {
            source_state: stage.steps[0].bdf_state.clone(),
            current_velocity_map_design_derivatives: vec![PcFanCfdTransientRemeshMapDesignEntry {
                target_dof: velocity_dimension,
                source_dof: 0,
                design_index: 0,
                derivative: 1.0,
            }],
            previous_velocity_map_design_derivatives: Vec::new(),
            pressure_map_design_derivatives: Vec::new(),
            provenance: "diagnostic:finite-difference-remesh-map".into(),
        };
        let schedule = remesh_objective_schedule_fixture(&stage, &mesh, 1);
        let error =
            solve_pc_fan_cfd_resolved_transient_remesh_schedule_objective_adjoint_with_map_design_derivatives(
                &PcFanResolvedAdjointConfig::default(),
                &config,
                &schedule,
                &[&mesh, &mesh],
                &[vec![vec![0.0; velocity_dimension + pressure_dimension]]],
                "production:pcfan:analytic-stage-derivatives:v1",
                &[vec![vec![vec![0.0; DESIGN_VARIABLE_COUNT]; velocity_dimension + pressure_dimension]]],
                &[vec![vec![0.0; DESIGN_VARIABLE_COUNT]]],
                &[derivative.clone()],
            )
            .expect_err("diagnostic remesh-map derivatives must be rejected");
        assert!(matches!(
            error,
            PcFanCfdTransientStateTransferError::ResolvedAdjoint(
                PcFanResolvedAdjointError::DerivativeProbeNotAdmissible
            )
        ));
        derivative.provenance = "production:pcfan:analytic-remesh-map:v1".into();
        let error =
            solve_pc_fan_cfd_resolved_transient_remesh_schedule_objective_adjoint_with_map_design_derivatives(
                &PcFanResolvedAdjointConfig::default(),
                &config,
                &schedule,
                &[&mesh, &mesh],
                &[vec![vec![0.0; velocity_dimension + pressure_dimension]]],
                "production:pcfan:analytic-stage-derivatives:v1",
                &[vec![vec![vec![0.0; DESIGN_VARIABLE_COUNT]; velocity_dimension + pressure_dimension]]],
                &[vec![vec![0.0; DESIGN_VARIABLE_COUNT]]],
                &[derivative],
            )
            .expect_err("malformed remesh-map dimensions must be rejected");
        assert!(matches!(
            error,
            PcFanCfdTransientStateTransferError::ResolvedAdjoint(
                PcFanResolvedAdjointError::DimensionMismatch(_)
            )
        ));
    }

    #[test]
    fn remesh_state_objective_adjoint_routes_later_stage_to_predecessor() {
        let (config, _design, trajectory) = fixture();
        let geometry = config.build_geometry(&_design);
        let domain = geometry.build_cfd_domain(&config);
        let mesh = box_mesh_tetrahedra_symmetric(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            2,
            2,
            2,
        );
        let mut first = trajectory.clone();
        let mut second = trajectory;
        let first_step = first.steps[0].clone();
        first.steps.push(PcFanCfdTransientTrajectoryStep {
            step: 1,
            time_s: 2.0,
            ..first_step
        });
        let second_step = second.steps[0].clone();
        second.steps.push(PcFanCfdTransientTrajectoryStep {
            step: 1,
            time_s: 2.0,
            ..second_step
        });
        add_identity_startup_history(&mut first);
        add_identity_startup_history(&mut second);
        let state_dimension = first.steps[0].state.len();
        let velocity_dimension = VectorH1Space::new(PolynomialDegree::P2, 3).num_global_dofs(&mesh);
        let pressure_dimension = H1Space::new(PolynomialDegree::P1).num_global_dofs(&mesh);
        let schedule = remesh_objective_schedule_fixture(&first, &mesh, 2);
        let result = solve_pc_fan_cfd_resolved_transient_remesh_schedule_state_objective_adjoint(
            &PcFanResolvedAdjointConfig::default(),
            &config,
            &PcFanCfdTransientRemeshSchedule {
                stages: vec![first, second],
                ..schedule
            },
            &[&mesh, &mesh, &mesh],
            &[
                vec![vec![0.0; state_dimension], vec![0.0; state_dimension]],
                vec![vec![1.0; state_dimension], vec![1.0; state_dimension]],
            ],
        )
        .expect("multi-stage remesh objective adjoint should solve");

        assert_eq!(result.stages.len(), 2);
        assert!(
            result.stages[1].report.adjoints[0]
                .iter()
                .all(|value| (*value - 1.0).abs() < 1.0e-10)
        );
        let target_initial_adjoint = vec![-1.0; state_dimension];
        let propagated = transfer_pc_fan_cfd_p2p1_state_adjoint(
            &mesh,
            &mesh,
            &target_initial_adjoint[..velocity_dimension],
            &target_initial_adjoint[velocity_dimension..],
        )
        .expect("identity remesh transpose should succeed");
        let expected_predecessor = propagated
            .velocity_dofs
            .iter()
            .chain(&propagated.pressure_dofs)
            .copied()
            .collect::<Vec<_>>();
        assert!(
            result.stages[0].report.adjoints[0][..velocity_dimension]
                .iter()
                .zip(&expected_predecessor[..velocity_dimension])
                .all(|(actual, expected)| (*actual - expected).abs() < 1.0e-10)
        );
        assert!(
            result.stages[0].report.adjoints[0][velocity_dimension..]
                .iter()
                .all(|value| value.abs() < 1.0e-10)
        );
        assert_eq!(
            result.stages[0].report.adjoints[0].len(),
            velocity_dimension + pressure_dimension
        );
        assert!(
            result.stages[0].report.adjoints[1][..velocity_dimension]
                .iter()
                .zip(&expected_predecessor[..velocity_dimension])
                .all(|(actual, expected)| (*actual - expected).abs() < 1.0e-10)
        );
        assert!(
            result.stages[0].report.adjoints[1][velocity_dimension..]
                .iter()
                .zip(&expected_predecessor[velocity_dimension..])
                .all(|(actual, expected)| (*actual - 2.0 * expected).abs() < 1.0e-10)
        );
    }

    #[test]
    fn remesh_state_objective_adjoint_rejects_missing_startup_history() {
        let (config, _design, trajectory) = fixture();
        let geometry = config.build_geometry(&_design);
        let domain = geometry.build_cfd_domain(&config);
        let mesh = box_mesh_tetrahedra_symmetric(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            2,
            2,
            2,
        );
        let schedule = remesh_objective_schedule_fixture(&trajectory, &mesh, 1);
        let error = solve_pc_fan_cfd_resolved_transient_remesh_schedule_state_objective_adjoint(
            &PcFanResolvedAdjointConfig::default(),
            &config,
            &schedule,
            &[&mesh, &mesh],
            &[vec![vec![0.0; trajectory.steps[0].state.len()]]],
        )
        .expect_err("missing startup history must be rejected");
        assert!(matches!(
            error,
            PcFanCfdTransientStateTransferError::ResolvedAdjoint(
                PcFanResolvedAdjointError::DimensionMismatch("remesh stage startup-history blocks")
            )
        ));
    }

    #[test]
    fn remesh_state_objective_adjoint_rejects_unsupported_scope() {
        let (config, _design, trajectory) = fixture();
        let geometry = config.build_geometry(&_design);
        let domain = geometry.build_cfd_domain(&config);
        let mesh = box_mesh_tetrahedra_symmetric(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            2,
            2,
            2,
        );
        let schedule = remesh_objective_schedule_fixture(&trajectory, &mesh, 1);
        let gradients = vec![vec![vec![0.0; trajectory.steps[0].state.len()]]];

        let mut turbulent_config = config.clone();
        turbulent_config.cfd.turbulence = PcFanTurbulenceModel::KOmegaSstTransported;
        let turbulent_error =
            solve_pc_fan_cfd_resolved_transient_remesh_schedule_state_objective_adjoint(
                &PcFanResolvedAdjointConfig::default(),
                &turbulent_config,
                &schedule,
                &[&mesh, &mesh],
                &gradients,
            )
            .expect_err("transported turbulence must remain outside this API");
        assert!(matches!(
            turbulent_error,
            PcFanCfdTransientStateTransferError::ResolvedAdjoint(
                PcFanResolvedAdjointError::DerivativeProbeNotAdmissible
            )
        ));

        let mut topology_schedule = schedule;
        topology_schedule.topology_transfer_adjoint_admissible = true;
        let topology_error =
            solve_pc_fan_cfd_resolved_transient_remesh_schedule_state_objective_adjoint(
                &PcFanResolvedAdjointConfig::default(),
                &config,
                &topology_schedule,
                &[&mesh, &mesh],
                &gradients,
            )
            .expect_err("topology derivatives must remain outside this API");
        assert!(matches!(
            topology_error,
            PcFanCfdTransientStateTransferError::ResolvedAdjoint(
                PcFanResolvedAdjointError::DerivativeProbeNotAdmissible
            )
        ));
    }

    #[test]
    fn transient_state_objective_gradients_use_production_operators() {
        let (config, design, trajectory) = fixture();
        for objective in [
            PcFanResolvedTransientObjective::Airflow,
            PcFanResolvedTransientObjective::PressureDrop,
            PcFanResolvedTransientObjective::PressureRise,
            PcFanResolvedTransientObjective::RotatingWallAxialLoading,
            PcFanResolvedTransientObjective::RotatingWallShaftPower,
        ] {
            let gradients = build_pc_fan_cfd_resolved_transient_objective_state_gradients(
                &config,
                &design,
                &trajectory,
                objective,
            )
            .expect("state objective derivative should be admitted");
            assert_eq!(gradients.len(), 1);
            assert_eq!(gradients[0].len(), trajectory.steps[0].state.len());
            assert!(gradients[0].iter().all(|value| value.is_finite()));
        }
        let airflow = build_pc_fan_cfd_resolved_transient_objective_state_gradients(
            &config,
            &design,
            &trajectory,
            PcFanResolvedTransientObjective::Airflow,
        )
        .expect("airflow derivative should be admitted");
        assert!(airflow[0][2..].iter().any(|value| *value > 0.0));
    }

    #[test]
    fn transient_state_objective_gradients_fail_closed_at_contract_boundary() {
        let (mut config, design, mut trajectory) = fixture();
        trajectory.observation.converged = false;
        assert!(matches!(
            build_pc_fan_cfd_resolved_transient_objective_state_gradients(
                &config,
                &design,
                &trajectory,
                PcFanResolvedTransientObjective::PressureRise,
            ),
            Err(PcFanResolvedAdjointError::ProductionStateNotConverged)
        ));
        config.cfd.flow_model = PcFanFlowModel::ReducedOrder;
        trajectory.observation.converged = true;
        assert!(matches!(
            build_pc_fan_cfd_resolved_transient_objective_state_gradients(
                &config,
                &design,
                &trajectory,
                PcFanResolvedTransientObjective::PressureRise,
            ),
            Err(PcFanResolvedAdjointError::UnsupportedFlowModel(_))
        ));
    }

    #[test]
    fn transient_direct_wall_design_gradient_uses_only_final_window() {
        let (config, design, mut trajectory) = fixture();
        trajectory.steps[0].time_s = 0.05;
        let mut final_step = trajectory.steps[0].clone();
        final_step.step = 1;
        final_step.time_s = 0.20;
        trajectory.steps.push(final_step);
        trajectory.observation.final_time_s = 0.20;

        let expected = build_pc_fan_cfd_resolved_state_objective_design_gradient_from_state(
            &config,
            &design,
            &trajectory.steps[1].state,
            PolynomialDegree::P2,
            PcFanResolvedFlowStateObjective::RotatingWallAxialLoading,
        )
        .expect("final-window wall design derivative should be admitted");
        assert!(expected.iter().any(|value| value.abs() > 1.0e-12));

        let report = solve_pc_fan_cfd_resolved_transient_objective_adjoint_from_trajectory(
            &PcFanResolvedAdjointConfig::default(),
            &config,
            &design,
            &trajectory,
            PcFanResolvedTransientObjective::RotatingWallAxialLoading,
        )
        .expect("transient wall objective adjoint should be admitted");

        assert_eq!(report.design_gradient.len(), DESIGN_VARIABLE_COUNT);
        for (actual, expected) in report.design_gradient.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 1.0e-11 * actual.abs().max(expected.abs()).max(1.0),
                "direct design gradient was not window weighted: actual={actual:.6e}, expected={expected:.6e}"
            );
        }
    }

    #[test]
    fn transient_objective_window_uses_time_quadrature_not_sample_count() {
        let (mut config, _design, mut trajectory) = fixture();
        config.operating_point.rpm = 60.0;
        trajectory.steps[0].time_s = 0.0;
        let mut middle = trajectory.steps[0].clone();
        middle.step = 1;
        middle.time_s = 0.1;
        let mut final_step = trajectory.steps[0].clone();
        final_step.step = 2;
        final_step.time_s = 0.2;
        trajectory.steps = vec![trajectory.steps[0].clone(), middle, final_step];
        trajectory.observation.final_time_s = 0.2;

        let weights = transient_objective_window_weights(&config, &trajectory)
            .expect("strictly increasing final-window times should be admitted");
        assert_eq!(weights.len(), 3);
        assert!((weights[0] - 0.25).abs() < 1.0e-12);
        assert!((weights[1] - 0.50).abs() < 1.0e-12);
        assert!((weights[2] - 0.25).abs() < 1.0e-12);
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1.0e-12);

        trajectory.steps[1].time_s = trajectory.steps[0].time_s;
        assert!(matches!(
            transient_objective_window_weights(&config, &trajectory),
            Err(PcFanResolvedAdjointError::DimensionMismatch(
                "transient objective step times"
            ))
        ));
    }

    #[test]
    fn transient_history_jacobian_skips_boundary_identity_rows() {
        let mesh = box_mesh_tetrahedra_symmetric(-0.1, 0.1, -0.1, 0.1, -0.1, 0.1, 1, 1, 1);
        let velocity_space = VectorH1Space::new(PolynomialDegree::P1, 3);
        let pressure_space = H1Space::new(PolynomialDegree::P1);
        let workspace = sonium_fem::physics::fluid::stokes::workspace::BdfWorkspace::new(
            &mesh,
            &velocity_space,
            &pressure_space,
        );
        let dimension = workspace.n_v + workspace.n_p;
        let mut row_ptrs = Vec::with_capacity(dimension + 1);
        let mut columns = Vec::with_capacity(dimension);
        let mut values = Vec::with_capacity(dimension);
        row_ptrs.push(0);
        for row in 0..dimension {
            columns.push(row);
            values.push(if row == 1 { 2.0 } else { 1.0 });
            row_ptrs.push(columns.len());
        }
        let state_jacobian =
            PcFanResolvedFlowSparseMatrix::new(dimension, row_ptrs, columns, values)
                .expect("manufactured state matrix");
        let histories = transient_history_jacobians(
            &workspace,
            &state_jacobian,
            &[0],
            1.0,
            sonium_fem::physics::fluid::stokes::time_stepping::BdfCoefficients::for_order(2),
            1,
        )
        .expect("history matrices should assemble");
        assert!(!histories.is_empty());
        let matrix = &histories[0].matrix;
        assert_eq!(matrix.row_ptrs[1], matrix.row_ptrs[0]);
        assert!(matrix.row_ptrs[2] > matrix.row_ptrs[1]);

        let first_initial =
            transient_stage_initial_history_jacobian(&workspace, &state_jacobian, &[0], 1.0, 0)
                .expect("first startup history should assemble")
                .expect("first startup history should be present");
        assert_eq!(first_initial.stage, 0);
        assert_eq!(first_initial.history_slot, 0);
        assert_eq!(first_initial.matrix.dimension, dimension);

        let second_initial =
            transient_stage_initial_history_jacobian(&workspace, &state_jacobian, &[0], 1.0, 1)
                .expect("second startup history should assemble")
                .expect("second startup history should be present");
        assert_eq!(second_initial.stage, 1);
        assert_eq!(second_initial.history_slot, 1);
        assert_eq!(second_initial.matrix.dimension, dimension);
        assert!(
            transient_stage_initial_history_jacobian(&workspace, &state_jacobian, &[0], 1.0, 2,)
                .expect("later history should be checked")
                .is_none()
        );
    }

    #[test]
    fn transient_state_adjoint_routes_stage_initial_history_blocks() {
        let (config, _design, trajectory) = fixture();
        let state_dimension = trajectory.steps[0].state.len();
        let objective_state_gradients = vec![vec![1.0; state_dimension]];
        let initial_history_jacobians = [PcFanResolvedTransientStageInitialHistoryJacobian {
            stage: 0,
            history_slot: 0,
            matrix: trajectory.steps[0].state_jacobian.clone(),
            objective_gradient: vec![0.0; state_dimension],
        }];

        let result = solve_pc_fan_cfd_resolved_transient_state_adjoint_with_stage_initial_history(
            &PcFanResolvedAdjointConfig::default(),
            &config,
            &trajectory,
            &objective_state_gradients,
            &initial_history_jacobians,
        )
        .expect("production stage-initial history adjoint should converge");

        assert!(result.report.converged);
        assert_eq!(result.initial_history_adjoints.len(), 1);
        let stage_adjoint = &result.report.adjoints[0];
        let history_adjoint = &result.initial_history_adjoints[0].adjoint;
        let matrix = &initial_history_jacobians[0].matrix;
        let mut expected = vec![0.0; state_dimension];
        for (row, window) in matrix.row_ptrs.windows(2).enumerate() {
            for entry in window[0]..window[1] {
                expected[matrix.column_indices[entry]] -= matrix.values[entry] * stage_adjoint[row];
            }
        }
        assert!(
            history_adjoint
                .iter()
                .zip(expected)
                .all(|(actual, expected)| (actual - expected).abs() <= 1.0e-12)
        );
    }

    #[test]
    fn transient_state_adjoint_uses_forward_retained_stage_initial_history() {
        let (config, _design, mut trajectory) = fixture();
        let state_dimension = trajectory.steps[0].state.len();
        let initial_history = PcFanResolvedTransientStageInitialHistoryJacobian {
            stage: 0,
            history_slot: 0,
            matrix: trajectory.steps[0].state_jacobian.clone(),
            objective_gradient: vec![0.0; state_dimension],
        };
        trajectory.stage_initial_history_jacobians = vec![initial_history.clone()];
        let objective_state_gradients = vec![vec![1.0; state_dimension]];

        let explicit =
            solve_pc_fan_cfd_resolved_transient_state_adjoint_with_stage_initial_history(
                &PcFanResolvedAdjointConfig::default(),
                &config,
                &trajectory,
                &objective_state_gradients,
                &[initial_history],
            )
            .expect("explicit production stage-initial history should converge");
        let retained =
            solve_pc_fan_cfd_resolved_transient_state_adjoint_with_retained_stage_initial_history(
                &PcFanResolvedAdjointConfig::default(),
                &config,
                &trajectory,
                &objective_state_gradients,
            )
            .expect("retained production stage-initial history should converge");

        assert_eq!(retained, explicit);
    }

    #[test]
    fn transient_objective_adjoint_wires_checked_state_gradients() {
        let (config, design, trajectory) = fixture();
        let report = solve_pc_fan_cfd_resolved_transient_objective_adjoint(
            &PcFanResolvedAdjointConfig::default(),
            &config,
            &design,
            &trajectory,
            PcFanResolvedTransientObjective::PressureDrop,
        )
        .expect("identity manufactured transient adjoint should converge");
        assert!(report.converged);
        assert_eq!(report.adjoints.len(), trajectory.steps.len());
        assert!(
            report
                .adjoints
                .iter()
                .flatten()
                .all(|value| value.is_finite())
        );
        assert!(report.design_gradient.is_empty());
    }

    #[test]
    fn transient_objective_adjoint_contracts_explicit_design_derivatives() {
        let (config, design, mut trajectory) = fixture();
        trajectory.steps[0].time_s = 0.05;
        let mut second = trajectory.steps[0].clone();
        second.step = 1;
        second.time_s = 0.10;
        let dimension = second.state_jacobian.dimension;
        let mut row_ptrs = Vec::with_capacity(dimension + 1);
        let mut columns = Vec::with_capacity(dimension);
        let mut values = Vec::with_capacity(dimension);
        row_ptrs.push(0);
        for row in 0..dimension {
            columns.push(row);
            values.push(0.25);
            row_ptrs.push(columns.len());
        }
        second.history_jacobians = vec![PcFanResolvedTransientHistoryJacobian {
            previous_step: 0,
            matrix: PcFanResolvedFlowSparseMatrix::new(dimension, row_ptrs, columns, values)
                .expect("manufactured history matrix"),
        }];
        trajectory.steps.push(second);
        trajectory.observation.final_time_s = 0.10;

        let state_gradients = build_pc_fan_cfd_resolved_transient_objective_state_gradients(
            &config,
            &design,
            &trajectory,
            PcFanResolvedTransientObjective::Airflow,
        )
        .expect("manufactured state gradients should be admitted");
        let derivative_index = state_gradients[0]
            .iter()
            .position(|value| value.abs() > f64::EPSILON)
            .expect("airflow must expose a state derivative");
        let mut first_design_jacobian = vec![vec![0.0; 1]; dimension];
        let mut second_design_jacobian = vec![vec![0.0; 1]; dimension];
        first_design_jacobian[derivative_index][0] = 2.0;
        second_design_jacobian[derivative_index][0] = 3.0;

        let report = solve_pc_fan_cfd_resolved_transient_objective_adjoint_with_derivatives(
            &PcFanResolvedAdjointConfig::default(),
            &config,
            &design,
            &trajectory,
            PcFanResolvedTransientObjective::Airflow,
            "analytic:manufactured-rotating-wall-shape:v1",
            &[first_design_jacobian, second_design_jacobian],
            &[vec![0.5], vec![-0.25]],
        )
        .expect("explicit transient design derivatives should be admitted");
        let expected = 0.25
            - 2.0 * report.adjoints[0][derivative_index]
            - 3.0 * report.adjoints[1][derivative_index];
        assert!((report.design_gradient[0] - expected).abs() < 1.0e-12);
        assert!(
            report
                .provenance
                .contains("manufactured-rotating-wall-shape")
        );

        assert!(matches!(
            solve_pc_fan_cfd_resolved_transient_objective_adjoint_with_derivatives(
                &PcFanResolvedAdjointConfig::default(),
                &config,
                &design,
                &trajectory,
                PcFanResolvedTransientObjective::Airflow,
                "finite-difference:diagnostic",
                &[],
                &[],
            ),
            Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible)
        ));
    }

    #[test]
    fn transient_design_adjoint_rejects_flow_only_nonlaminar_derivatives() {
        let (mut config, design, trajectory) = fixture();
        config.cfd.turbulence = PcFanTurbulenceModel::KOmegaSstTransported;

        let residual_design_jacobians = trajectory
            .steps
            .iter()
            .map(|step| vec![Vec::new(); step.state.len()])
            .collect::<Vec<_>>();
        let objective_design_gradients = trajectory
            .steps
            .iter()
            .map(|_| vec![0.0; DESIGN_VARIABLE_COUNT])
            .collect::<Vec<_>>();

        assert!(matches!(
            solve_pc_fan_cfd_resolved_transient_objective_adjoint_with_derivatives(
                &PcFanResolvedAdjointConfig::default(),
                &config,
                &design,
                &trajectory,
                PcFanResolvedTransientObjective::Airflow,
                "analytic:flow-only-fixture:v1",
                &residual_design_jacobians,
                &objective_design_gradients,
            ),
            Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible)
        ));
    }

    #[test]
    fn production_rotating_wall_trajectory_wires_objective_adjoint() {
        let mut config = PcFanConfig::default_120mm();
        config.cfd.flow_model = PcFanFlowModel::IncompressibleCfdFull;
        config.cfd.backend = PcFanCfdBackend::Cpu;
        config.cfd.cells_x = 2;
        config.cfd.cells_y = 2;
        config.cfd.cells_z = 2;
        config.cfd.max_picard_iterations = 1;
        config.cfd.linear_solver_max_iterations = 128;
        // This 2x2x2 trajectory is a bounded operator smoke fixture; use
        // explicit relaxed ledger budgets so it tests trajectory retention.
        config.cfd.max_relative_mass_imbalance = 10.0;
        config.cfd.max_relative_momentum_imbalance = 10.0;
        config.cfd.max_relative_pressure_closure = 10.0;
        config.cfd.max_relative_energy_imbalance = 10.0;
        config.cfd.nonlinear_tolerance = 10.0;
        let design = config.default_design_variables();
        let trajectory = solve_pc_fan_cfd_rotating_wall_transient_with_trajectory(
            &config, &design, 1.0e-3, 1.0e-3,
        )
        .expect("bounded rotating-wall trajectory should solve");
        assert!(trajectory.steps.len() >= 2);
        assert!(trajectory.observation.converged);
        assert_eq!(trajectory.stage_initial_history_jacobians.len(), 2);
        for (history, (stage, history_slot)) in trajectory
            .stage_initial_history_jacobians
            .iter()
            .zip([(0, 0), (1, 1)])
        {
            assert_eq!(history.stage, stage);
            assert_eq!(history.history_slot, history_slot);
            assert_eq!(
                history.matrix.dimension,
                trajectory.steps[stage].state.len()
            );
        }
        for step in &trajectory.steps {
            for history in &step.history_jacobians {
                for row in 0..step.state_jacobian.dimension {
                    let state_start = step.state_jacobian.row_ptrs[row];
                    let state_end = step.state_jacobian.row_ptrs[row + 1];
                    let history_start = history.matrix.row_ptrs[row];
                    let history_end = history.matrix.row_ptrs[row + 1];
                    if state_end - state_start == 1
                        && step.state_jacobian.column_indices[state_start] == row
                        && step.state_jacobian.values[state_start] == 1.0
                    {
                        assert_eq!(history_start, history_end);
                    }
                }
            }
        }
        let report = solve_pc_fan_cfd_resolved_transient_objective_adjoint(
            &PcFanResolvedAdjointConfig {
                max_state_residual: 1.0,
                ..PcFanResolvedAdjointConfig::default()
            },
            &config,
            &design,
            &trajectory,
            PcFanResolvedTransientObjective::PressureDrop,
        )
        .expect("captured trajectory objective adjoint should converge");
        assert!(report.converged);
        assert_eq!(report.adjoints.len(), trajectory.steps.len());

        let production_report =
            solve_pc_fan_cfd_resolved_transient_objective_adjoint_from_trajectory(
                &PcFanResolvedAdjointConfig {
                    max_state_residual: 1.0,
                    ..PcFanResolvedAdjointConfig::default()
                },
                &config,
                &design,
                &trajectory,
                PcFanResolvedTransientObjective::RotatingWallAxialLoading,
            )
            .expect("production frozen-topology rotating-wall adjoint should converge");
        assert!(production_report.converged);
        assert_eq!(
            production_report.design_gradient.len(),
            DESIGN_VARIABLE_COUNT
        );
        assert!(
            production_report
                .design_gradient
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(
            production_report
                .design_gradient
                .iter()
                .any(|value| value.abs() > 0.0)
        );

        let mut algebraic_config = config.clone();
        algebraic_config.cfd.turbulence = PcFanTurbulenceModel::KOmegaSstAlgebraic;
        assert!(matches!(
            solve_pc_fan_cfd_resolved_transient_objective_adjoint_from_trajectory(
                &PcFanResolvedAdjointConfig::default(),
                &algebraic_config,
                &design,
                &trajectory,
                PcFanResolvedTransientObjective::RotatingWallShaftPower,
            ),
            Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible)
        ));

        let mut moved_design = design.clone();
        moved_design.shroud_radius_offsets_m[0] = 1.0e-4;
        let moved_report = solve_pc_fan_cfd_resolved_transient_objective_adjoint_from_trajectory(
            &PcFanResolvedAdjointConfig {
                max_state_residual: 1.0,
                ..PcFanResolvedAdjointConfig::default()
            },
            &config,
            &moved_design,
            &trajectory,
            PcFanResolvedTransientObjective::RotatingWallShaftPower,
        )
        .expect("finite shroud-radius transient derivatives should be admitted");
        assert_eq!(moved_report.design_gradient.len(), DESIGN_VARIABLE_COUNT);
        assert!(
            moved_report
                .design_gradient
                .iter()
                .all(|value| value.is_finite())
        );
    }
}

/// A final-window observation from the resolved rotating-wall transient
/// runner.  The observation is deliberately separate from
/// [`PcFanCfdFlowSolution`]: a transient value is a window average, not a
/// converged steady field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcFanCfdTransientObservation {
    pub final_time_s: f64,
    pub airflow_cfm: f64,
    pub pressure_drop_pa: f64,
    pub axial_loading_n: f64,
    pub residual: f64,
    /// Final-window averaged control-volume conservation ledger.
    pub conservation: PcFanCfdConservation,
    pub converged: bool,
}

/// Public copy of the accepted mean-flow BDF history that does not depend on
/// FEM's internal state type. It can be converted back before a remesh handoff.
#[derive(Debug, Clone, PartialEq)]
pub struct PcFanCfdTransientBdfState {
    /// Current velocity DOFs.
    pub velocity: Vec<f64>,
    /// Previous velocity DOFs retained for BDF2.
    pub velocity_prev: Vec<f64>,
    /// Current pressure DOFs.
    pub pressure: Vec<f64>,
    /// Physical state time in seconds.
    pub time: f64,
    /// Accepted BDF step index.
    pub step: usize,
}

impl PcFanCfdTransientBdfState {
    fn from_fem(state: &sonium_fem::physics::fluid::stokes::time_stepping::BdfState) -> Self {
        Self {
            velocity: state.velocity.clone(),
            velocity_prev: state.velocity_prev.clone(),
            pressure: state.pressure.clone(),
            time: state.time,
            step: state.step,
        }
    }

    /// Convert this retained history to the FEM transient-state contract.
    pub fn to_fem(&self) -> sonium_fem::physics::fluid::stokes::time_stepping::BdfState {
        sonium_fem::physics::fluid::stokes::time_stepping::BdfState {
            velocity: self.velocity.clone(),
            velocity_prev: self.velocity_prev.clone(),
            pressure: self.pressure.clone(),
            time: self.time,
            step: self.step,
        }
    }
}

/// One production BDF step retained for a transient adjoint or audit.
#[derive(Debug, Clone, PartialEq)]
pub struct PcFanCfdTransientTrajectoryStep {
    pub step: usize,
    pub time_s: f64,
    pub state: Vec<f64>,
    pub state_jacobian: PcFanResolvedFlowSparseMatrix,
    /// Provenance of the published state derivative.  In particular, a
    /// lagged/Oseen turbulent matrix must not be mistaken for an exact
    /// algebraic-turbulence BDF Newton tangent.
    pub state_jacobian_provenance: String,
    /// Concrete linear-solve backend used for this accepted BDF step.
    ///
    /// This is deliberately separate from the requested backend in
    /// `PcFanCfdConfig`. Unavailable or failed accelerator setup is retained
    /// as explicit CPU fallback provenance rather than promoted to a hardware
    /// execution claim.
    pub actual_backend: PcFanCfdBackend,
    /// Requested/actual backend and execution-residency provenance for this
    /// accepted step.
    pub backend_provenance: PcFanCfdBackendProvenance,
    /// Build/runtime details captured with this step for cross-architecture
    /// reproducibility checks.
    pub execution_environment: PcFanCfdExecutionEnvironment,
    pub rhs: Vec<f64>,
    pub state_residual: Vec<f64>,
    pub residual_norm: f64,
    /// Control-volume conservation ledger for this accepted BDF step.
    ///
    /// Retaining the per-step ledger prevents a final-window average from
    /// hiding an inadmissible intermediate state during adjoint admission.
    pub conservation: PcFanCfdConservation,
    pub history_jacobians: Vec<PcFanResolvedTransientHistoryJacobian>,
    /// Accepted segregated k--omega state, when transported turbulence is on.
    pub transported_turbulence: Option<PcFanCfdKOmegaBdfState>,
    /// Element-local lagged eddy viscosity applied to this momentum step.
    /// This is the coefficient field used by the segregated momentum solve,
    /// not an exact scalar/momentum Newton derivative.
    pub transported_eddy_viscosity_per_element: Option<Vec<f64>>,
    /// Exact frozen-coefficient k--omega scalar tangent for the accepted
    /// update. Momentum cross-derivatives are intentionally not present.
    pub transported_turbulence_linearization: Option<KOmegaTransportLinearization>,
    /// Exact off-diagonal coupling blocks for the accepted transported step.
    pub transported_turbulence_coupled_linearization: Option<PcFanKOmegaCoupledLinearization>,
    /// Accepted scalar residual vectors paired with the retained scalar
    /// tangent. These are needed for full coupled-adjoint residual admission.
    pub transported_turbulence_residual: Option<(Vec<f64>, Vec<f64>)>,
    /// Exact fixed-state scalar ALE/design residual rows, ordered `[k, omega]`.
    /// This remains separate from the flow-only design residual so a partial
    /// transported derivative cannot be mistaken for a complete one.
    pub transported_turbulence_residual_design_jacobian: Option<Vec<Vec<f64>>>,
    /// Accepted current/previous mean-flow BDF state. Exposing both levels
    /// permits a later remesh stage to transfer the complete time history.
    pub bdf_state: PcFanCfdTransientBdfState,
    /// Whether topology-transfer terms required by a full trajectory design
    /// adjoint were published. External remesh handoffs set this false.
    pub topology_transfer_adjoint_admissible: bool,
    /// Production design residual derivative. In full
    /// rotating-wall mode it includes analytic mesh-scaled blade occupancy
    /// and topology-preserving ALE mesh-motion derivatives,
    /// including hub/tip/shroud controls.
    pub residual_design_jacobian: Vec<Vec<f64>>,
}

/// Resolved rotating-wall trajectory plus per-step BDF systems and derivative
/// provenance. Turbulent Oseen fallback steps remain explicitly marked and
/// are not admitted to the exact transient turbulent state adjoint.
#[derive(Debug, Clone, PartialEq)]
pub struct PcFanCfdTransientTrajectory {
    pub observation: PcFanCfdTransientObservation,
    pub steps: Vec<PcFanCfdTransientTrajectoryStep>,
    /// Checked residual-history Jacobians for BDF states entering before the
    /// retained trajectory. The stage indices are local to this trajectory;
    /// remesh schedule callers must transfer their contractions separately.
    pub stage_initial_history_jacobians: Vec<PcFanResolvedTransientStageInitialHistoryJacobian>,
}

/// Accepted trajectories and transfer diagnostics for a multi-stage external
/// remeshing schedule.
///
/// Each entry in `stages` starts from the last accepted BDF state of the
/// preceding entry.  Since external remeshing does not publish a differentiable
/// mesh-motion map, the complete schedule is fail-closed for topology-changing
/// design adjoints even though every forward handoff is checked.
#[derive(Debug, Clone, PartialEq)]
pub struct PcFanCfdTransientRemeshSchedule {
    /// One independently assembled transient trajectory per target mesh.
    pub stages: Vec<PcFanCfdTransientTrajectory>,
    /// P2/P1 transfer diagnostics in chronological remesh order.
    pub mean_flow_transfer_reports: Vec<PcFanCfdP2P1StateTransferReport>,
    /// k--omega transfer diagnostics, present only for transported turbulence.
    pub transported_turbulence_transfer_reports: Option<Vec<PcFanCfdKOmegaBdfStateTransferReport>>,
    /// Every target stage must rebuild topology-dependent FEM operators.
    pub requires_operator_reassembly: bool,
    /// External remeshing has no published exact topology derivative.
    pub topology_transfer_adjoint_admissible: bool,
    /// Stable provenance for audit and downstream admission checks.
    pub provenance: &'static str,
}

/// State-adjoint result for the fixed interpolation maps at a remesh
/// schedule boundary.
///
/// The returned chain is ordered backwards: the last entry is the adjoint on
/// the original source mesh. This is a fixed-map state adjoint only. It does
/// not differentiate the caller-owned remesher, its topology decision, or
/// the geometry dependence of the selected mesh pair, so it must not be used
/// as a topology-changing design adjoint.
#[derive(Debug, Clone)]
pub struct PcFanCfdTransientRemeshStateAdjoint {
    /// Mean-flow P2/P1 BDF adjoints, one state per remesh transition.
    pub mean_flow: PcFanCfdP2P1StateAdjointTransferChain,
    /// Transported k--omega BDF adjoints, when the schedule carries scalar
    /// history transfer reports.
    pub transported_turbulence: Option<PcFanCfdKOmegaBdfStateAdjointTransferChain>,
    /// The interpolation transpose is exact for the supplied fixed mesh
    /// chain.
    pub fixed_map_transpose_admissible: bool,
    /// Remesher/topology design derivatives are intentionally not supplied.
    pub topology_design_adjoint_admissible: bool,
    /// Stable provenance for downstream admission checks.
    pub provenance: &'static str,
}

/// Fixed-map state/objective adjoint composition for a remeshed transient
/// schedule.
///
/// The stage results are chronological. Each result includes the adjoints
/// with respect to that stage's incoming startup history; for stage zero,
/// those are the adjoints of the caller-owned source state. The composition
/// differentiates only the published P2/P1 interpolation maps. It does not
/// differentiate the remesher, its topology choice, mesh geometry, or any
/// turbulence equations.
#[derive(Debug, Clone)]
pub struct PcFanCfdTransientRemeshObjectiveAdjoint {
    /// One resolved transient adjoint result per accepted stage.
    pub stages: Vec<PcFanResolvedTransientAdjointWithInitialHistory>,
    /// Fixed-map design gradient from retained stage residual/objective
    /// derivatives, plus any explicitly supplied remesh-map contribution.
    pub design_gradient: Vec<f64>,
    /// Design gradient contributed by caller-supplied remesh-map derivatives.
    pub remesh_map_design_gradient: Vec<f64>,
    /// The remesh boundary was composed with exact fixed-map transposes.
    pub fixed_map_transpose_admissible: bool,
    /// Caller supplied an analytic derivative of each fixed-topology remesh
    /// interpolation map. This does not cover the discrete remesher choice.
    pub interpolation_map_design_adjoint_admissible: bool,
    /// Full topology/design derivatives are intentionally not supplied.
    pub topology_design_adjoint_admissible: bool,
    /// Stable provenance for downstream admission checks.
    pub provenance: &'static str,
}

/// Analytic design derivative of one fixed-topology BDF remesh map.
///
/// Each entry stores `(target_dof, source_dof, design_index, ∂T/∂d)`, not
/// `∂(T state)/∂d`. The sparse representation follows local FEM interpolation
/// and avoids an `O(target_dofs × source_dofs × designs)` dense allocation.
/// The source BDF state is retained so the adjoint can form the required
/// contraction without guessing which primal state the caller used to
/// evaluate the map derivative. The contract covers geometry-dependent node
/// locations and interpolation weights for the supplied mesh pair; it does
/// not represent the discrete remesher selection or a change in connectivity.
#[derive(Debug, Clone, PartialEq)]
pub struct PcFanCfdTransientRemeshMapDesignDerivative {
    /// Source state at the remesh boundary.
    pub source_state: PcFanCfdTransientBdfState,
    /// Sparse `∂T_current_velocity/∂design` entries.
    pub current_velocity_map_design_derivatives: Vec<PcFanCfdTransientRemeshMapDesignEntry>,
    /// Sparse `∂T_previous_velocity/∂design` entries.
    pub previous_velocity_map_design_derivatives: Vec<PcFanCfdTransientRemeshMapDesignEntry>,
    /// Sparse `∂T_pressure/∂design` entries.
    pub pressure_map_design_derivatives: Vec<PcFanCfdTransientRemeshMapDesignEntry>,
    /// Provenance of the analytic map derivative assembly.
    pub provenance: String,
}

/// One non-zero entry of a remesh-map design derivative.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcFanCfdTransientRemeshMapDesignEntry {
    /// Target field DOF row.
    pub target_dof: usize,
    /// Source field DOF column.
    pub source_dof: usize,
    /// Design-variable column.
    pub design_index: usize,
    /// Analytic derivative `∂T[target_dof, source_dof]/∂design_index`.
    pub derivative: f64,
}

fn transfer_pc_fan_cfd_remesh_initial_history_adjoint(
    source_mesh: &sonium_fem::mesh::Mesh,
    target_mesh: &sonium_fem::mesh::Mesh,
    target_adjoint: &[f64],
) -> Result<PcFanCfdP2P1StateAdjointTransfer, PcFanCfdTransientStateTransferError> {
    let target_velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
    let target_pressure_space = H1Space::new(PolynomialDegree::P1);
    let target_velocity_dof_count = target_velocity_space.num_global_dofs(target_mesh);
    let target_pressure_dof_count = target_pressure_space.num_global_dofs(target_mesh);
    let expected = target_velocity_dof_count + target_pressure_dof_count;
    if target_adjoint.len() != expected {
        return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
            PcFanResolvedAdjointError::DimensionMismatch("remesh startup history adjoint"),
        ));
    }

    transfer_pc_fan_cfd_p2p1_state_adjoint(
        source_mesh,
        target_mesh,
        &target_adjoint[..target_velocity_dof_count],
        &target_adjoint[target_velocity_dof_count..],
    )
    .map_err(PcFanCfdTransientStateTransferError::StateTransfer)
}

fn add_pc_fan_cfd_remesh_state_gradient(
    target: &mut [f64],
    velocity: &[f64],
    pressure: &[f64],
) -> Result<(), PcFanCfdTransientStateTransferError> {
    if target.len() != velocity.len() + pressure.len() {
        return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
            PcFanResolvedAdjointError::DimensionMismatch(
                "remesh propagated state objective gradient",
            ),
        ));
    }
    for (value, addition) in target.iter_mut().zip(velocity.iter().chain(pressure)) {
        *value += addition;
    }
    if target.iter().any(|value| !value.is_finite()) {
        return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
            PcFanResolvedAdjointError::NonFiniteInput("remesh propagated state objective gradient"),
        ));
    }
    Ok(())
}

fn add_pc_fan_cfd_remesh_velocity_gradient(
    target: &mut [f64],
    velocity: &[f64],
) -> Result<(), PcFanCfdTransientStateTransferError> {
    if target.len() < velocity.len() {
        return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
            PcFanResolvedAdjointError::DimensionMismatch(
                "remesh propagated velocity objective gradient",
            ),
        ));
    }
    for (value, addition) in target.iter_mut().take(velocity.len()).zip(velocity) {
        *value += addition;
    }
    if target.iter().any(|value| !value.is_finite()) {
        return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
            PcFanResolvedAdjointError::NonFiniteInput(
                "remesh propagated velocity objective gradient",
            ),
        ));
    }
    Ok(())
}

fn add_pc_fan_cfd_remesh_pressure_gradient(
    target: &mut [f64],
    velocity_dof_count: usize,
    pressure: &[f64],
) -> Result<(), PcFanCfdTransientStateTransferError> {
    if target.len() != velocity_dof_count + pressure.len() {
        return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
            PcFanResolvedAdjointError::DimensionMismatch(
                "remesh propagated pressure objective gradient",
            ),
        ));
    }
    for (value, addition) in target[velocity_dof_count..].iter_mut().zip(pressure) {
        *value += addition;
    }
    if target.iter().any(|value| !value.is_finite()) {
        return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
            PcFanResolvedAdjointError::NonFiniteInput(
                "remesh propagated pressure objective gradient",
            ),
        ));
    }
    Ok(())
}

/// Compose fixed-map state/objective adjoints through a remeshed transient
/// schedule.
///
/// This is deliberately scoped to full incompressible laminar CFD and state
/// objectives. It consumes each stage's published startup-history Jacobians,
/// solves the stage adjoint in reverse order, and routes the resulting startup
/// adjoints through the preceding remesh transpose. The pressure adjoint is
/// shared by both BDF history levels and is therefore accumulated on the
/// preceding current state.
pub fn solve_pc_fan_cfd_resolved_transient_remesh_schedule_state_objective_adjoint(
    adjoint_config: &PcFanResolvedAdjointConfig,
    config: &PcFanConfig,
    schedule: &PcFanCfdTransientRemeshSchedule,
    mesh_chain: &[&sonium_fem::mesh::Mesh],
    stage_objective_state_gradients: &[Vec<Vec<f64>>],
) -> Result<PcFanCfdTransientRemeshObjectiveAdjoint, PcFanCfdTransientStateTransferError> {
    if !matches!(config.cfd.flow_model, PcFanFlowModel::IncompressibleCfdFull) {
        return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
            PcFanResolvedAdjointError::UnsupportedFlowModel(config.cfd.flow_model),
        ));
    }
    if !matches!(config.cfd.turbulence, PcFanTurbulenceModel::Laminar) {
        return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
            PcFanResolvedAdjointError::DerivativeProbeNotAdmissible,
        ));
    }
    if schedule.topology_transfer_adjoint_admissible
        || schedule.transported_turbulence_transfer_reports.is_some()
    {
        return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
            PcFanResolvedAdjointError::DerivativeProbeNotAdmissible,
        ));
    }
    let stage_count = schedule.stages.len();
    if stage_count == 0 {
        return Err(PcFanCfdTransientStateTransferError::StateTransfer(
            PcFanCfdStateTransferError::InvalidMesh {
                role: "remesh schedule",
                reason: "at least one accepted stage required".into(),
            },
        ));
    }
    if mesh_chain.len() != stage_count + 1 {
        return Err(PcFanCfdTransientStateTransferError::StateTransfer(
            PcFanCfdStateTransferError::InvalidMesh {
                role: "mesh chain",
                reason: format!(
                    "expected {} meshes for {} stages, got {}",
                    stage_count + 1,
                    stage_count,
                    mesh_chain.len()
                ),
            },
        ));
    }
    if stage_objective_state_gradients.len() != stage_count {
        return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
            PcFanResolvedAdjointError::DimensionMismatch("remesh stage objective gradient count"),
        ));
    }

    // Reuse the existing checked schedule validator for mesh/report topology
    // consistency before any stage result is produced.
    let target_mesh = mesh_chain[stage_count];
    let target_velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
    let target_pressure_space = H1Space::new(PolynomialDegree::P1);
    let terminal_adjoint = sonium_fem::physics::fluid::stokes::time_stepping::BdfState {
        velocity: vec![0.0; target_velocity_space.num_global_dofs(target_mesh)],
        velocity_prev: vec![0.0; target_velocity_space.num_global_dofs(target_mesh)],
        pressure: vec![0.0; target_pressure_space.num_global_dofs(target_mesh)],
        time: 0.0,
        step: 0,
    };
    transfer_pc_fan_cfd_transient_remesh_schedule_state_adjoint(
        schedule,
        mesh_chain,
        &terminal_adjoint,
        None,
    )?;

    let mut objective_gradients = stage_objective_state_gradients.to_vec();
    let mut initial_history_jacobians = schedule
        .stages
        .iter()
        .map(|trajectory| trajectory.stage_initial_history_jacobians.clone())
        .collect::<Vec<_>>();

    for (stage, trajectory) in schedule.stages.iter().enumerate() {
        let expected_history_count =
            usize::from(!trajectory.steps.is_empty()) + usize::from(trajectory.steps.len() >= 2);
        if trajectory.steps.is_empty()
            || initial_history_jacobians[stage].len() != expected_history_count
        {
            return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
                PcFanResolvedAdjointError::DimensionMismatch("remesh stage startup-history blocks"),
            ));
        }
        for history in &initial_history_jacobians[stage] {
            if history.history_slot > 1
                || history.stage != history.history_slot
                || history.history_slot >= expected_history_count
            {
                return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
                    PcFanResolvedAdjointError::DimensionMismatch(
                        "remesh stage startup-history labels",
                    ),
                ));
            }
        }
        if initial_history_jacobians[stage]
            .iter()
            .map(|history| history.history_slot)
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != expected_history_count
        {
            return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
                PcFanResolvedAdjointError::DimensionMismatch("remesh stage startup-history labels"),
            ));
        }
    }

    let mut stage_results = (0..stage_count)
        .map(|_| None)
        .collect::<Vec<Option<PcFanResolvedTransientAdjointWithInitialHistory>>>();
    for stage in (0..stage_count).rev() {
        let result = solve_pc_fan_cfd_resolved_transient_state_adjoint_with_stage_initial_history(
            adjoint_config,
            config,
            &schedule.stages[stage],
            &objective_gradients[stage],
            &initial_history_jacobians[stage],
        )
        .map_err(PcFanCfdTransientStateTransferError::ResolvedAdjoint)?;

        if stage > 0 {
            let slot_zero = result
                .initial_history_adjoints
                .iter()
                .find(|history| history.history_slot == 0)
                .ok_or(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
                    PcFanResolvedAdjointError::DimensionMismatch(
                        "remesh stage startup-history slot zero",
                    ),
                ))?;
            let current = transfer_pc_fan_cfd_remesh_initial_history_adjoint(
                mesh_chain[stage],
                mesh_chain[stage + 1],
                &slot_zero.adjoint,
            )?;
            let previous_stage = stage - 1;
            let last_step = objective_gradients[previous_stage]
                .len()
                .checked_sub(1)
                .ok_or(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
                    PcFanResolvedAdjointError::DimensionMismatch(
                        "remesh preceding stage objective gradients",
                    ),
                ))?;
            add_pc_fan_cfd_remesh_state_gradient(
                &mut objective_gradients[previous_stage][last_step],
                &current.velocity_dofs,
                &current.pressure_dofs,
            )?;
            if schedule.stages[stage].steps.len() >= 2 {
                let slot_one = result
                    .initial_history_adjoints
                    .iter()
                    .find(|history| history.history_slot == 1)
                    .ok_or(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
                        PcFanResolvedAdjointError::DimensionMismatch(
                            "remesh stage startup-history slot one",
                        ),
                    ))?;
                let previous = transfer_pc_fan_cfd_remesh_initial_history_adjoint(
                    mesh_chain[stage],
                    mesh_chain[stage + 1],
                    &slot_one.adjoint,
                )?;
                let source_velocity_dof_count =
                    VectorH1Space::new(PolynomialDegree::P2, 3).num_global_dofs(mesh_chain[stage]);
                if schedule.stages[previous_stage].steps.len() >= 2 {
                    let penultimate = objective_gradients[previous_stage]
                        .len()
                        .checked_sub(2)
                        .ok_or(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
                            PcFanResolvedAdjointError::DimensionMismatch(
                                "remesh preceding stage penultimate state",
                            ),
                        ))?;
                    add_pc_fan_cfd_remesh_velocity_gradient(
                        &mut objective_gradients[previous_stage][penultimate],
                        &previous.velocity_dofs,
                    )?;
                    add_pc_fan_cfd_remesh_pressure_gradient(
                        &mut objective_gradients[previous_stage][last_step],
                        source_velocity_dof_count,
                        &previous.pressure_dofs,
                    )?;
                } else {
                    let startup = initial_history_jacobians[previous_stage]
                        .iter_mut()
                        .find(|history| history.history_slot == 0)
                        .ok_or(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
                            PcFanResolvedAdjointError::DimensionMismatch(
                                "remesh preceding stage startup slot zero",
                            ),
                        ))?;
                    add_pc_fan_cfd_remesh_state_gradient(
                        &mut startup.objective_gradient,
                        &previous.velocity_dofs,
                        &previous.pressure_dofs,
                    )?;
                }
            }
        }
        stage_results[stage] = Some(result);
    }

    let stages = stage_results
        .into_iter()
        .map(|result| {
            result.ok_or(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
                PcFanResolvedAdjointError::DimensionMismatch("remesh stage adjoint result"),
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PcFanCfdTransientRemeshObjectiveAdjoint {
        stages,
        design_gradient: vec![0.0; DESIGN_VARIABLE_COUNT],
        remesh_map_design_gradient: vec![0.0; DESIGN_VARIABLE_COUNT],
        fixed_map_transpose_admissible: true,
        interpolation_map_design_adjoint_admissible: false,
        topology_design_adjoint_admissible: false,
        provenance: "pcfan:transient-remesh-fixed-map-state-objective-adjoint:v1",
    })
}

/// Contract fixed-map remesh-schedule design derivatives in addition to the
/// state/objective adjoints returned by
/// [`solve_pc_fan_cfd_resolved_transient_remesh_schedule_state_objective_adjoint`].
///
/// The derivative bundles are indexed as
/// `stage -> retained step -> residual row -> design variable` and
/// `stage -> retained step -> design variable`. They must come from the same
/// converged production assembly as the published trajectory Jacobians. The
/// result includes `phi_d - R_d^T lambda` for every retained stage and routes
/// state multipliers through the fixed P2/P1 maps at remesh boundaries.
///
/// This is deliberately a fixed-map contract. It does not differentiate the
/// external remesher's discrete selection, target connectivity, target node
/// positions, or interpolation weights, so
/// `topology_design_adjoint_admissible` remains false.
pub fn solve_pc_fan_cfd_resolved_transient_remesh_schedule_objective_adjoint_with_derivatives(
    adjoint_config: &PcFanResolvedAdjointConfig,
    config: &PcFanConfig,
    schedule: &PcFanCfdTransientRemeshSchedule,
    mesh_chain: &[&sonium_fem::mesh::Mesh],
    stage_objective_state_gradients: &[Vec<Vec<f64>>],
    derivative_provenance: &str,
    stage_residual_design_jacobians: &[Vec<Vec<Vec<f64>>>],
    stage_objective_design_gradients: &[Vec<Vec<f64>>],
) -> Result<PcFanCfdTransientRemeshObjectiveAdjoint, PcFanCfdTransientStateTransferError> {
    if derivative_provenance.trim().is_empty() {
        return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
            PcFanResolvedAdjointError::MissingProvenance,
        ));
    }
    let provenance = derivative_provenance.to_ascii_lowercase();
    if [
        "picard",
        "oseen",
        "derivative-probe",
        "finite-difference",
        "black-box",
    ]
    .iter()
    .any(|marker| provenance.contains(marker))
    {
        return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
            PcFanResolvedAdjointError::DerivativeProbeNotAdmissible,
        ));
    }

    let stage_count = schedule.stages.len();
    if stage_residual_design_jacobians.len() != stage_count
        || stage_objective_design_gradients.len() != stage_count
    {
        return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
            PcFanResolvedAdjointError::DimensionMismatch("remesh stage design derivative count"),
        ));
    }
    for (stage_index, trajectory) in schedule.stages.iter().enumerate() {
        let residual_steps = &stage_residual_design_jacobians[stage_index];
        let objective_steps = &stage_objective_design_gradients[stage_index];
        if residual_steps.len() != trajectory.steps.len()
            || objective_steps.len() != trajectory.steps.len()
        {
            return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
                PcFanResolvedAdjointError::DimensionMismatch(
                    "remesh stage step design derivative count",
                ),
            ));
        }
        for (step_index, step) in trajectory.steps.iter().enumerate() {
            let residual_rows = &residual_steps[step_index];
            let objective_gradient = &objective_steps[step_index];
            if residual_rows.len() != step.state.len()
                || objective_gradient.len() != DESIGN_VARIABLE_COUNT
                || residual_rows
                    .iter()
                    .any(|row| row.len() != DESIGN_VARIABLE_COUNT)
            {
                return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
                    PcFanResolvedAdjointError::DimensionMismatch(
                        "remesh stage design derivative dimensions",
                    ),
                ));
            }
            if objective_gradient.iter().any(|value| !value.is_finite())
                || residual_rows
                    .iter()
                    .flatten()
                    .any(|value| !value.is_finite())
            {
                return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
                    PcFanResolvedAdjointError::NonFiniteInput("remesh stage design derivatives"),
                ));
            }
        }
    }

    let mut result = solve_pc_fan_cfd_resolved_transient_remesh_schedule_state_objective_adjoint(
        adjoint_config,
        config,
        schedule,
        mesh_chain,
        stage_objective_state_gradients,
    )?;
    let mut design_gradient = vec![0.0; DESIGN_VARIABLE_COUNT];

    for (stage_index, stage_result) in result.stages.iter_mut().enumerate() {
        let mut stage_design_gradient = vec![0.0; DESIGN_VARIABLE_COUNT];
        for (step_index, adjoint) in stage_result.report.adjoints.iter().enumerate() {
            let residual_rows = &stage_residual_design_jacobians[stage_index][step_index];
            let objective_gradient = &stage_objective_design_gradients[stage_index][step_index];
            for design_index in 0..DESIGN_VARIABLE_COUNT {
                let residual_design = residual_rows
                    .iter()
                    .zip(adjoint)
                    .map(|(row, &multiplier)| row[design_index] * multiplier)
                    .sum::<f64>();
                stage_design_gradient[design_index] +=
                    objective_gradient[design_index] - residual_design;
            }
        }
        if !stage_design_gradient.iter().all(|value| value.is_finite()) {
            return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
                PcFanResolvedAdjointError::NonFiniteInput("remesh stage design gradient"),
            ));
        }
        stage_result.report.design_gradient = stage_design_gradient.clone();
        for (total, stage_value) in design_gradient.iter_mut().zip(stage_design_gradient) {
            *total += stage_value;
        }
    }
    if !design_gradient.iter().all(|value| value.is_finite()) {
        return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
            PcFanResolvedAdjointError::NonFiniteInput("remesh schedule design gradient"),
        ));
    }
    result.design_gradient = design_gradient;
    result.provenance =
        "pcfan:transient-remesh-fixed-map-objective-adjoint-with-design-derivatives:v1";
    Ok(result)
}

fn validate_pc_fan_cfd_remesh_map_design_derivative(
    derivative: &PcFanCfdTransientRemeshMapDesignDerivative,
    source_mesh: &sonium_fem::mesh::Mesh,
    target_mesh: &sonium_fem::mesh::Mesh,
) -> Result<(), PcFanCfdTransientStateTransferError> {
    if derivative.provenance.trim().is_empty() {
        return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
            PcFanResolvedAdjointError::MissingProvenance,
        ));
    }
    let provenance = derivative.provenance.to_ascii_lowercase();
    if [
        "picard",
        "oseen",
        "derivative-probe",
        "finite-difference",
        "black-box",
    ]
    .iter()
    .any(|marker| provenance.contains(marker))
    {
        return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
            PcFanResolvedAdjointError::DerivativeProbeNotAdmissible,
        ));
    }

    let source_velocity_dof_count =
        VectorH1Space::new(PolynomialDegree::P2, 3).num_global_dofs(source_mesh);
    let target_velocity_dof_count =
        VectorH1Space::new(PolynomialDegree::P2, 3).num_global_dofs(target_mesh);
    let source_pressure_dof_count = H1Space::new(PolynomialDegree::P1).num_global_dofs(source_mesh);
    let target_pressure_dof_count = H1Space::new(PolynomialDegree::P1).num_global_dofs(target_mesh);
    if derivative.source_state.velocity.len() != source_velocity_dof_count
        || derivative.source_state.velocity_prev.len() != source_velocity_dof_count
        || derivative.source_state.pressure.len() != source_pressure_dof_count
    {
        return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
            PcFanResolvedAdjointError::DimensionMismatch("remesh map source BDF state"),
        ));
    }
    if !derivative.source_state.time.is_finite()
        || !derivative
            .source_state
            .velocity
            .iter()
            .chain(&derivative.source_state.velocity_prev)
            .chain(&derivative.source_state.pressure)
            .all(|value| value.is_finite())
    {
        return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
            PcFanResolvedAdjointError::NonFiniteInput("remesh map source BDF state"),
        ));
    }

    let validate_entries = |entries: &[PcFanCfdTransientRemeshMapDesignEntry],
                            target_count: usize,
                            source_count: usize,
                            name: &'static str| {
        if entries.iter().any(|entry| {
            entry.target_dof >= target_count
                || entry.source_dof >= source_count
                || entry.design_index >= DESIGN_VARIABLE_COUNT
        }) {
            return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
                PcFanResolvedAdjointError::DimensionMismatch(name),
            ));
        }
        if entries.iter().any(|entry| !entry.derivative.is_finite()) {
            return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
                PcFanResolvedAdjointError::NonFiniteInput(name),
            ));
        }
        let mut keys = entries
            .iter()
            .map(|entry| (entry.target_dof, entry.source_dof, entry.design_index))
            .collect::<Vec<_>>();
        keys.sort_unstable();
        if keys.windows(2).any(|window| window[0] == window[1]) {
            return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
                PcFanResolvedAdjointError::DimensionMismatch(name),
            ));
        }
        Ok(())
    };
    validate_entries(
        &derivative.current_velocity_map_design_derivatives,
        target_velocity_dof_count,
        source_velocity_dof_count,
        "remesh current-velocity map design derivative",
    )?;
    validate_entries(
        &derivative.previous_velocity_map_design_derivatives,
        target_velocity_dof_count,
        source_velocity_dof_count,
        "remesh previous-velocity map design derivative",
    )?;
    validate_entries(
        &derivative.pressure_map_design_derivatives,
        target_pressure_dof_count,
        source_pressure_dof_count,
        "remesh pressure map design derivative",
    )?;
    Ok(())
}

fn accumulate_pc_fan_cfd_remesh_map_design_contraction(
    target: &mut [f64],
    multipliers: &[f64],
    map_design_derivatives: &[PcFanCfdTransientRemeshMapDesignEntry],
    source_state: &[f64],
) -> Result<(), PcFanCfdTransientStateTransferError> {
    if target.len() != DESIGN_VARIABLE_COUNT {
        return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
            PcFanResolvedAdjointError::DimensionMismatch("remesh map design contraction"),
        ));
    }
    if map_design_derivatives.iter().any(|entry| {
        entry.target_dof >= multipliers.len()
            || entry.source_dof >= source_state.len()
            || entry.design_index >= DESIGN_VARIABLE_COUNT
            || !entry.derivative.is_finite()
    }) {
        return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
            PcFanResolvedAdjointError::DimensionMismatch("remesh map design contraction"),
        ));
    }
    for entry in map_design_derivatives {
        target[entry.design_index] +=
            multipliers[entry.target_dof] * entry.derivative * source_state[entry.source_dof];
    }
    if target.iter().any(|value| !value.is_finite()) {
        return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
            PcFanResolvedAdjointError::NonFiniteInput("remesh map design contraction"),
        ));
    }
    Ok(())
}

/// Compose a remeshed transient objective adjoint with caller-supplied
/// analytic derivatives of the fixed-topology interpolation maps.
///
/// The supplied map derivatives are contracted with the stage-initial
/// history adjoints after the ordinary fixed-map recurrence has been solved.
/// This admits geometry-dependent node and interpolation-weight derivatives
/// for a known mesh pair. The discrete remesher choice, connectivity changes,
/// and any omitted topology-selection derivative remain fail-closed, so the
/// returned `topology_design_adjoint_admissible` flag is still false.
#[allow(clippy::too_many_arguments)]
pub fn solve_pc_fan_cfd_resolved_transient_remesh_schedule_objective_adjoint_with_map_design_derivatives(
    adjoint_config: &PcFanResolvedAdjointConfig,
    config: &PcFanConfig,
    schedule: &PcFanCfdTransientRemeshSchedule,
    mesh_chain: &[&sonium_fem::mesh::Mesh],
    stage_objective_state_gradients: &[Vec<Vec<f64>>],
    derivative_provenance: &str,
    stage_residual_design_jacobians: &[Vec<Vec<Vec<f64>>>],
    stage_objective_design_gradients: &[Vec<Vec<f64>>],
    remesh_map_design_derivatives: &[PcFanCfdTransientRemeshMapDesignDerivative],
) -> Result<PcFanCfdTransientRemeshObjectiveAdjoint, PcFanCfdTransientStateTransferError> {
    if remesh_map_design_derivatives.len() != schedule.stages.len() {
        return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
            PcFanResolvedAdjointError::DimensionMismatch(
                "remesh map design derivative stage count",
            ),
        ));
    }
    for (stage, derivative) in remesh_map_design_derivatives.iter().enumerate() {
        let source_mesh = mesh_chain.get(stage).ok_or({
            PcFanCfdTransientStateTransferError::ResolvedAdjoint(
                PcFanResolvedAdjointError::DimensionMismatch("remesh map source mesh chain"),
            )
        })?;
        let target_mesh = mesh_chain.get(stage + 1).ok_or({
            PcFanCfdTransientStateTransferError::ResolvedAdjoint(
                PcFanResolvedAdjointError::DimensionMismatch("remesh map target mesh chain"),
            )
        })?;
        validate_pc_fan_cfd_remesh_map_design_derivative(derivative, source_mesh, target_mesh)?;
        if stage > 0 {
            let expected_source_state = schedule.stages[stage - 1]
                .steps
                .last()
                .ok_or(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
                    PcFanResolvedAdjointError::DimensionMismatch(
                        "remesh map preceding stage state",
                    ),
                ))?
                .bdf_state
                .clone();
            if derivative.source_state != expected_source_state {
                return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
                    PcFanResolvedAdjointError::DimensionMismatch(
                        "remesh map source state does not match preceding stage",
                    ),
                ));
            }
        }
    }

    let mut result =
        solve_pc_fan_cfd_resolved_transient_remesh_schedule_objective_adjoint_with_derivatives(
            adjoint_config,
            config,
            schedule,
            mesh_chain,
            stage_objective_state_gradients,
            derivative_provenance,
            stage_residual_design_jacobians,
            stage_objective_design_gradients,
        )?;
    let mut remesh_map_design_gradient = vec![0.0; DESIGN_VARIABLE_COUNT];
    for (stage, derivative) in remesh_map_design_derivatives.iter().enumerate() {
        let target_mesh = mesh_chain[stage + 1];
        let target_velocity_dof_count =
            VectorH1Space::new(PolynomialDegree::P2, 3).num_global_dofs(target_mesh);
        let target_pressure_dof_count =
            H1Space::new(PolynomialDegree::P1).num_global_dofs(target_mesh);
        let expected_state_dimension = target_velocity_dof_count + target_pressure_dof_count;
        let stage_result = &mut result.stages[stage];
        let slot_zero = stage_result
            .initial_history_adjoints
            .iter()
            .find(|history| history.history_slot == 0)
            .ok_or(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
                PcFanResolvedAdjointError::DimensionMismatch(
                    "remesh map stage startup-history slot zero",
                ),
            ))?;
        if slot_zero.adjoint.len() != expected_state_dimension {
            return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
                PcFanResolvedAdjointError::DimensionMismatch(
                    "remesh map stage startup-history state",
                ),
            ));
        }
        let (slot_zero_velocity, slot_zero_pressure) =
            slot_zero.adjoint.split_at(target_velocity_dof_count);
        let mut stage_map_design_gradient = vec![0.0; DESIGN_VARIABLE_COUNT];
        accumulate_pc_fan_cfd_remesh_map_design_contraction(
            &mut stage_map_design_gradient,
            slot_zero_velocity,
            &derivative.current_velocity_map_design_derivatives,
            &derivative.source_state.velocity,
        )?;
        accumulate_pc_fan_cfd_remesh_map_design_contraction(
            &mut stage_map_design_gradient,
            slot_zero_pressure,
            &derivative.pressure_map_design_derivatives,
            &derivative.source_state.pressure,
        )?;

        if schedule.stages[stage].steps.len() >= 2 {
            let slot_one = stage_result
                .initial_history_adjoints
                .iter()
                .find(|history| history.history_slot == 1)
                .ok_or(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
                    PcFanResolvedAdjointError::DimensionMismatch(
                        "remesh map stage startup-history slot one",
                    ),
                ))?;
            if slot_one.adjoint.len() != expected_state_dimension {
                return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
                    PcFanResolvedAdjointError::DimensionMismatch(
                        "remesh map stage startup-history state",
                    ),
                ));
            }
            let (slot_one_velocity, slot_one_pressure) =
                slot_one.adjoint.split_at(target_velocity_dof_count);
            accumulate_pc_fan_cfd_remesh_map_design_contraction(
                &mut stage_map_design_gradient,
                slot_one_velocity,
                &derivative.previous_velocity_map_design_derivatives,
                &derivative.source_state.velocity_prev,
            )?;
            accumulate_pc_fan_cfd_remesh_map_design_contraction(
                &mut stage_map_design_gradient,
                slot_one_pressure,
                &derivative.pressure_map_design_derivatives,
                &derivative.source_state.pressure,
            )?;
        }
        if stage_result.report.design_gradient.len() != DESIGN_VARIABLE_COUNT {
            return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
                PcFanResolvedAdjointError::DimensionMismatch("remesh stage design gradient"),
            ));
        }
        for (value, addition) in stage_result
            .report
            .design_gradient
            .iter_mut()
            .zip(&stage_map_design_gradient)
        {
            *value += addition;
        }
        for (value, addition) in remesh_map_design_gradient
            .iter_mut()
            .zip(stage_map_design_gradient)
        {
            *value += addition;
        }
    }
    for (value, addition) in result
        .design_gradient
        .iter_mut()
        .zip(&remesh_map_design_gradient)
    {
        *value += addition;
    }
    if !remesh_map_design_gradient
        .iter()
        .all(|value| value.is_finite())
        || !result.design_gradient.iter().all(|value| value.is_finite())
    {
        return Err(PcFanCfdTransientStateTransferError::ResolvedAdjoint(
            PcFanResolvedAdjointError::NonFiniteInput("remesh map design gradient"),
        ));
    }
    result.remesh_map_design_gradient = remesh_map_design_gradient;
    result.interpolation_map_design_adjoint_admissible = true;
    result.topology_design_adjoint_admissible = false;
    result.provenance =
        "pcfan:transient-remesh-fixed-map-objective-adjoint-with-map-design-derivatives:v1";
    Ok(result)
}

/// Compose fixed-map BDF state adjoints through a remesh schedule.
///
/// `mesh_chain[0]` is the mesh owning the schedule's source state and each
/// following mesh is the target of one schedule stage. The caller must retain
/// this chain because a schedule produced by an external remesher cannot
/// reconstruct its selected topologies from transfer reports alone.
///
/// This API deliberately stops at the exact transpose of the published
/// interpolation maps. It does not admit a full transient objective/design
/// adjoint: stage-initial BDF residual-history derivatives and the
/// remesher's topology/geometry derivatives are not represented by the
/// schedule contract.
pub fn transfer_pc_fan_cfd_transient_remesh_schedule_state_adjoint(
    schedule: &PcFanCfdTransientRemeshSchedule,
    mesh_chain: &[&sonium_fem::mesh::Mesh],
    terminal_mean_flow_adjoint: &sonium_fem::physics::fluid::stokes::time_stepping::BdfState,
    terminal_turbulence_adjoint: Option<&PcFanCfdKOmegaBdfState>,
) -> Result<PcFanCfdTransientRemeshStateAdjoint, PcFanCfdTransientStateTransferError> {
    let transition_count = schedule.stages.len();
    if transition_count == 0 {
        return Err(PcFanCfdTransientStateTransferError::StateTransfer(
            PcFanCfdStateTransferError::InvalidMesh {
                role: "remesh schedule",
                reason: "at least one accepted stage is required".into(),
            },
        ));
    }
    if mesh_chain.len() != transition_count + 1 {
        return Err(PcFanCfdTransientStateTransferError::StateTransfer(
            PcFanCfdStateTransferError::InvalidMesh {
                role: "mesh chain",
                reason: format!(
                    "expected {} meshes for {} stages, got {}",
                    transition_count + 1,
                    transition_count,
                    mesh_chain.len()
                ),
            },
        ));
    }
    if schedule.mean_flow_transfer_reports.len() != transition_count {
        return Err(PcFanCfdTransientStateTransferError::StateTransfer(
            PcFanCfdStateTransferError::InvalidMesh {
                role: "remesh schedule",
                reason: "mean-flow transfer report count does not match stages".into(),
            },
        ));
    }
    let source_velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
    let target_velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
    let source_pressure_space = H1Space::new(PolynomialDegree::P1);
    let target_pressure_space = H1Space::new(PolynomialDegree::P1);
    for (report, meshes) in schedule
        .mean_flow_transfer_reports
        .iter()
        .zip(mesh_chain.windows(2))
    {
        let expected = PcFanCfdP2P1StateTransferReport {
            source_node_count: meshes[0].num_nodes(),
            target_node_count: meshes[1].num_nodes(),
            source_element_count: meshes[0].num_elements(),
            target_element_count: meshes[1].num_elements(),
            source_velocity_dof_count: source_velocity_space.num_global_dofs(meshes[0]),
            target_velocity_dof_count: target_velocity_space.num_global_dofs(meshes[1]),
            source_pressure_dof_count: source_pressure_space.num_global_dofs(meshes[0]),
            target_pressure_dof_count: target_pressure_space.num_global_dofs(meshes[1]),
        };
        if *report != expected {
            return Err(PcFanCfdTransientStateTransferError::StateTransfer(
                PcFanCfdStateTransferError::InvalidMesh {
                    role: "remesh schedule",
                    reason: "mean-flow transfer report does not match mesh chain".into(),
                },
            ));
        }
    }
    if !schedule.requires_operator_reassembly {
        return Err(PcFanCfdTransientStateTransferError::StateTransfer(
            PcFanCfdStateTransferError::InvalidMesh {
                role: "remesh schedule",
                reason: "topology-changing schedule must reassemble target operators".into(),
            },
        ));
    }
    if !terminal_mean_flow_adjoint.time.is_finite() {
        return Err(PcFanCfdTransientStateTransferError::StateTransfer(
            PcFanCfdStateTransferError::NonFiniteSourceTime,
        ));
    }

    let mean_flow =
        transfer_pc_fan_cfd_p2p1_bdf_state_adjoint_chain(mesh_chain, terminal_mean_flow_adjoint)
            .map_err(PcFanCfdTransientStateTransferError::StateTransfer)?;

    let transported_turbulence = match (
        schedule.transported_turbulence_transfer_reports.as_ref(),
        terminal_turbulence_adjoint,
    ) {
        (Some(reports), Some(terminal_adjoint)) => {
            if reports.len() != transition_count {
                return Err(PcFanCfdTransientStateTransferError::StateTransfer(
                    PcFanCfdStateTransferError::InvalidMesh {
                        role: "remesh schedule",
                        reason: "k--omega transfer report count does not match stages".into(),
                    },
                ));
            }
            let scalar_space = H1Space::new(PolynomialDegree::P1);
            for (report, meshes) in reports.iter().zip(mesh_chain.windows(2)) {
                if report.source_scalar_dof_count != scalar_space.num_global_dofs(meshes[0])
                    || report.target_scalar_dof_count != scalar_space.num_global_dofs(meshes[1])
                {
                    return Err(PcFanCfdTransientStateTransferError::StateTransfer(
                        PcFanCfdStateTransferError::InvalidMesh {
                            role: "remesh schedule",
                            reason: "k--omega transfer report does not match mesh chain".into(),
                        },
                    ));
                }
            }
            if !terminal_adjoint.time.is_finite() {
                return Err(PcFanCfdTransientStateTransferError::KOmegaStateTransfer(
                    PcFanCfdKOmegaStateTransferError::NonFiniteSourceTime,
                ));
            }
            Some(
                transfer_pc_fan_cfd_komega_bdf_state_adjoint_chain(mesh_chain, terminal_adjoint)
                    .map_err(PcFanCfdTransientStateTransferError::KOmegaStateTransfer)?,
            )
        }
        (Some(_), None) => {
            return Err(PcFanCfdTransientStateTransferError::MissingTransportedTurbulenceAdjoint);
        }
        (None, Some(_)) => {
            return Err(
                PcFanCfdTransientStateTransferError::UnexpectedTransportedTurbulenceAdjoint,
            );
        }
        (None, None) => None,
    };

    Ok(PcFanCfdTransientRemeshStateAdjoint {
        mean_flow,
        transported_turbulence,
        fixed_map_transpose_admissible: true,
        topology_design_adjoint_admissible: false,
        provenance: "pcfan:transient-remesh-fixed-map-state-adjoint:v1",
    })
}

/// Cell counts for one built-in design-aware ALE remesh stage.
///
/// The stage mesh uses the configured tetrahedralisation pattern and then
/// applies the checked PC-fan radial mesh-motion map.  This is a bounded
/// body-fitted restart/remesh bridge: topology-dependent operators are still
/// rebuilt by the transient runner, while the topology-changing design
/// derivative remains deliberately inadmissible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcFanCfdBodyFittedRemeshLevel {
    pub cells_x: usize,
    pub cells_y: usize,
    pub cells_z: usize,
}

impl PcFanCfdBodyFittedRemeshLevel {
    /// Construct a remesh level, requiring positive cell counts at admission.
    pub const fn new(cells_x: usize, cells_y: usize, cells_z: usize) -> Self {
        Self {
            cells_x,
            cells_y,
            cells_z,
        }
    }
}

/// Continue a resolved rotating-wall transient through internally generated
/// design-aware body-fitted remesh levels.
///
/// Each level is generated from `config`/`design`, receives the checked ALE
/// mesh motion, and is passed through the same transactional state-transfer
/// and target-operator reassembly path as the caller-owned remesher API.  The
/// function publishes the external-remesh fail-closed provenance because the
/// generated topology map is not an exact differentiable remeshing operator.
#[allow(clippy::too_many_arguments)]
pub fn solve_pc_fan_cfd_rotating_wall_transient_with_builtin_body_fitted_remesher(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    time_step_s: f64,
    rotor_revolutions: f64,
    source_mesh: &sonium_fem::mesh::Mesh,
    source_state: &PcFanCfdTransientBdfState,
    source_turbulence_state: Option<&PcFanCfdKOmegaBdfState>,
    remesh_levels: &[PcFanCfdBodyFittedRemeshLevel],
) -> Result<PcFanCfdTransientRemeshSchedule, PcFanCfdTransientStateTransferError> {
    if remesh_levels.is_empty() {
        return Err(PcFanCfdTransientStateTransferError::StateTransfer(
            PcFanCfdStateTransferError::InvalidMesh {
                role: "body-fitted remesh schedule",
                reason: "at least one remesh level is required".into(),
            },
        ));
    }
    if let Some((stage, level)) = remesh_levels
        .iter()
        .enumerate()
        .find(|(_, level)| level.cells_x == 0 || level.cells_y == 0 || level.cells_z == 0)
    {
        return Err(PcFanCfdTransientStateTransferError::StateTransfer(
            PcFanCfdStateTransferError::InvalidMesh {
                role: "body-fitted remesh level",
                reason: format!(
                    "stage {stage} has non-positive cell counts ({}, {}, {})",
                    level.cells_x, level.cells_y, level.cells_z
                ),
            },
        ));
    }

    solve_pc_fan_cfd_rotating_wall_transient_with_state_transfer_with_remesher(
        config,
        design,
        time_step_s,
        rotor_revolutions,
        source_mesh,
        source_state,
        source_turbulence_state,
        remesh_levels.len(),
        |stage, _, _, _| {
            let level = remesh_levels
                .get(stage)
                .copied()
                .ok_or_else(|| format!("missing built-in remesh level {stage}"))?;
            build_pc_fan_cfd_body_fitted_mesh(config, design, level)
        },
    )
}

/// Configuration for a bounded two-level BDF time-step independence check.
///
/// Both levels integrate the same rotor interval.  The time steps must land
/// exactly on the interval so the comparison is made at the same physical
/// final time rather than between two different ceil-rounded trajectories.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcFanCfdTransientConvergenceConfig {
    pub coarse_time_step_s: f64,
    pub fine_time_step_s: f64,
    pub rotor_revolutions: f64,
    pub relative_tolerance: f64,
    pub max_final_residual: f64,
}

impl Default for PcFanCfdTransientConvergenceConfig {
    fn default() -> Self {
        Self {
            coarse_time_step_s: 5.0e-4,
            fine_time_step_s: 2.5e-4,
            rotor_revolutions: 0.02,
            relative_tolerance: 0.08,
            max_final_residual: 1.0e-3,
        }
    }
}

/// Configuration errors raised before a transient independence solve starts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PcFanCfdTransientConvergenceConfigError {
    InvalidCoarseTimeStep(f64),
    InvalidFineTimeStep(f64),
    FineTimeStepNotSmaller { coarse: f64, fine: f64 },
    InvalidRotorRevolutions(f64),
    InvalidRelativeTolerance(f64),
    InvalidFinalResidual(f64),
    NonIntegralTimeGrid { time_step_s: f64, step_count: f64 },
    TooFewTimeSteps { time_step_s: f64, step_count: f64 },
    SolveFailed(PcFanCfdTransientError),
}

impl fmt::Display for PcFanCfdTransientConvergenceConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCoarseTimeStep(value) => {
                write!(formatter, "invalid coarse transient time step {value}")
            }
            Self::InvalidFineTimeStep(value) => {
                write!(formatter, "invalid fine transient time step {value}")
            }
            Self::FineTimeStepNotSmaller { coarse, fine } => write!(
                formatter,
                "fine transient time step {fine} must be smaller than coarse step {coarse}"
            ),
            Self::InvalidRotorRevolutions(value) => {
                write!(formatter, "invalid transient rotor revolutions {value}")
            }
            Self::InvalidRelativeTolerance(value) => {
                write!(formatter, "invalid transient relative tolerance {value}")
            }
            Self::InvalidFinalResidual(value) => {
                write!(formatter, "invalid transient residual budget {value}")
            }
            Self::NonIntegralTimeGrid {
                time_step_s,
                step_count,
            } => write!(
                formatter,
                "time step {time_step_s} does not land on the rotor interval (steps {step_count})"
            ),
            Self::TooFewTimeSteps {
                time_step_s,
                step_count,
            } => write!(
                formatter,
                "time step {time_step_s} gives only {step_count} BDF steps"
            ),
            Self::SolveFailed(error) => {
                write!(formatter, "transient convergence solve failed: {error}")
            }
        }
    }
}

impl std::error::Error for PcFanCfdTransientConvergenceConfigError {}

/// One admitted level in a transient time-step independence report.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcFanCfdTransientConvergenceLevel {
    pub time_step_s: f64,
    pub observation: PcFanCfdTransientObservation,
}

/// Checked result of the coarse/fine transient time-step comparison.
#[derive(Debug, Clone, PartialEq)]
pub struct PcFanCfdTransientConvergence {
    pub levels: Vec<PcFanCfdTransientConvergenceLevel>,
    pub airflow_relative_change: Option<f64>,
    pub pressure_relative_change: Option<f64>,
    pub axial_loading_relative_change: Option<f64>,
    pub final_time_relative_change: Option<f64>,
    pub converged: bool,
    pub all_levels_converged_and_finite: bool,
}

/// A state objective supported by the production rotating-wall trajectory.
///
/// This enum covers the exact fixed-mesh observation operators already used
/// by the transient runner. It does not imply that the immersed geometry or
/// its design residual derivative is available.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcFanResolvedTransientObjective {
    /// Absolute outlet volumetric flow, reported in CFM.
    Airflow,
    /// Inlet mean pressure minus outlet mean pressure, matching the
    /// transient trajectory's `pressure_drop_pa` observation.
    PressureDrop,
    /// Outlet mean pressure minus inlet mean pressure, reported in pascals.
    PressureRise,
    /// Recovered axial force transferred from rotating walls to the fluid.
    RotatingWallAxialLoading,
    /// Recovered rotating-wall shaft power transferred to the fluid.
    RotatingWallShaftPower,
}

/// Fail-closed errors from the resolved rotating-wall BDF runner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PcFanCfdTransientError {
    UnsupportedFlowModel(PcFanFlowModel),
    InvalidNonlinearTolerance {
        tolerance: f64,
    },
    InvalidTimeStep,
    InvalidRotorRevolutions,
    TooManySteps,
    InvalidGeometry,
    InvalidInitialState {
        reason: &'static str,
    },
    BackendExecutionNotAdmitted {
        requested: PcFanCfdBackend,
        actual: PcFanCfdBackend,
        mode: PcFanCfdExecutionMode,
    },
    StepFailed {
        step: usize,
    },
    MissingLinearSystem {
        step: usize,
    },
    InvalidLinearSystem {
        step: usize,
    },
    TransportedTurbulenceFailed {
        step: usize,
    },
    NonFiniteObservation,
    NonConvergedObservation {
        residual: f64,
        tolerance: f64,
    },
    InvalidConservationTolerance {
        metric: &'static str,
        tolerance: f64,
    },
    ConservationViolation {
        metric: &'static str,
        value: f64,
        tolerance: f64,
    },
}

impl fmt::Display for PcFanCfdTransientError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedFlowModel(model) => {
                write!(
                    formatter,
                    "resolved transient runner requires full CFD, got {model:?}"
                )
            }
            Self::InvalidNonlinearTolerance { tolerance } => write!(
                formatter,
                "transient nonlinear tolerance must be finite and positive, got {tolerance}"
            ),
            Self::InvalidTimeStep => {
                write!(formatter, "transient time step must be finite and positive")
            }
            Self::InvalidRotorRevolutions => {
                write!(formatter, "rotor revolutions must be finite and positive")
            }
            Self::TooManySteps => write!(
                formatter,
                "transient request exceeds the bounded step budget"
            ),
            Self::InvalidGeometry => {
                write!(formatter, "invalid PC-fan geometry for transient CFD")
            }
            Self::InvalidInitialState { reason } => {
                write!(formatter, "invalid initial transient BDF state: {reason}")
            }
            Self::BackendExecutionNotAdmitted {
                requested,
                actual,
                mode,
            } => write!(
                formatter,
                "transient backend request {requested:?} resolved to {actual:?} in {mode:?}, which is not admitted"
            ),
            Self::StepFailed { step } => write!(formatter, "transient BDF step {step} failed"),
            Self::MissingLinearSystem { step } => {
                write!(
                    formatter,
                    "transient BDF step {step} did not publish its linear system"
                )
            }
            Self::InvalidLinearSystem { step } => {
                write!(
                    formatter,
                    "transient BDF step {step} published an invalid linear system"
                )
            }
            Self::TransportedTurbulenceFailed { step } => {
                write!(
                    formatter,
                    "transient transported k--omega step {step} failed"
                )
            }
            Self::NonFiniteObservation => write!(formatter, "transient observation is non-finite"),
            Self::NonConvergedObservation {
                residual,
                tolerance,
            } => write!(
                formatter,
                "transient observation residual {residual:.3e} exceeds tolerance {tolerance:.3e}"
            ),
            Self::InvalidConservationTolerance { metric, tolerance } => write!(
                formatter,
                "transient conservation tolerance {metric} is invalid: {tolerance}"
            ),
            Self::ConservationViolation {
                metric,
                value,
                tolerance,
            } => write!(
                formatter,
                "transient conservation {metric} {value:.3e} exceeds tolerance {tolerance:.3e}"
            ),
        }
    }
}

impl std::error::Error for PcFanCfdTransientError {}

/// Errors from a topology-changing transient restart handoff.
#[derive(Debug, Clone, PartialEq)]
pub enum PcFanCfdTransientStateTransferError {
    /// Source state could not be transferred to the target P2/P1 topology.
    StateTransfer(PcFanCfdStateTransferError),
    /// Transported P1 k--omega history could not be transferred to the
    /// target topology.
    KOmegaStateTransfer(PcFanCfdKOmegaStateTransferError),
    /// The target transient solve failed after the handoff was accepted.
    Solve(PcFanCfdTransientError),
    /// The caller-owned remesher rejected or could not produce a target mesh.
    Remesher { stage: usize, reason: String },
    /// The schedule carries transported scalar history but the terminal
    /// scalar adjoint was omitted.
    MissingTransportedTurbulenceAdjoint,
    /// A terminal scalar adjoint was supplied for a schedule without scalar
    /// history transfer.
    UnexpectedTransportedTurbulenceAdjoint,
    /// Local resolved-adjoint admission or solve failed while composing a
    /// fixed-map remesh schedule.
    ResolvedAdjoint(PcFanResolvedAdjointError),
}

impl fmt::Display for PcFanCfdTransientStateTransferError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StateTransfer(error) => {
                write!(formatter, "transient state transfer failed: {error}")
            }
            Self::KOmegaStateTransfer(error) => {
                write!(
                    formatter,
                    "transient k--omega history transfer failed: {error}"
                )
            }
            Self::Solve(error) => write!(formatter, "transient restart solve failed: {error}"),
            Self::Remesher { stage, reason } => {
                write!(
                    formatter,
                    "transient remesher failed at stage {stage}: {reason}"
                )
            }
            Self::MissingTransportedTurbulenceAdjoint => {
                write!(
                    formatter,
                    "terminal transported k--omega adjoint is required"
                )
            }
            Self::UnexpectedTransportedTurbulenceAdjoint => {
                write!(
                    formatter,
                    "terminal transported k--omega adjoint is unexpected"
                )
            }
            Self::ResolvedAdjoint(error) => {
                write!(formatter, "transient resolved adjoint failed: {error}")
            }
        }
    }
}

impl std::error::Error for PcFanCfdTransientStateTransferError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::StateTransfer(error) => Some(error),
            Self::KOmegaStateTransfer(error) => Some(error),
            Self::Solve(error) => Some(error),
            Self::Remesher { .. } => None,
            Self::MissingTransportedTurbulenceAdjoint
            | Self::UnexpectedTransportedTurbulenceAdjoint => None,
            Self::ResolvedAdjoint(error) => Some(error),
        }
    }
}

struct PcFanTransportedMonolithicStep {
    flow_state: sonium_fem::physics::fluid::stokes::time_stepping::BdfState,
    turbulence_state: PcFanCfdKOmegaBdfState,
    flow_residual: Vec<f64>,
    flow_jacobian: sonium_solvers::CsrMatrix<f64>,
    scalar_residual: (Vec<f64>, Vec<f64>),
    scalar_linearization: KOmegaTransportLinearization,
    coupled_linearization: PcFanKOmegaCoupledLinearization,
    residual_norm: f64,
}

struct PcFanTransportedMonolithicAssembly {
    candidate_flow: sonium_fem::physics::fluid::stokes::time_stepping::BdfState,
    candidate_turbulence: PcFanCfdKOmegaBdfState,
    flow_residual: Vec<f64>,
    flow_jacobian: sonium_solvers::CsrMatrix<f64>,
    scalar_residual: (Vec<f64>, Vec<f64>),
    scalar_linearization: KOmegaTransportLinearization,
    coupled_linearization: PcFanKOmegaCoupledLinearization,
    residual_norm: f64,
}

#[allow(clippy::too_many_arguments)]
fn assemble_pc_fan_cfd_transported_monolithic_trial(
    config: &PcFanConfig,
    mesh: &sonium_fem::mesh::Mesh,
    velocity_space: &VectorH1Space,
    pressure_space: &H1Space,
    bdf: &sonium_fem::physics::fluid::stokes::time_stepping::BdfConfig,
    previous_flow: &sonium_fem::physics::fluid::stokes::time_stepping::BdfState,
    previous_turbulence: &PcFanCfdKOmegaBdfState,
    base_ns_config: &NavierStokesConfig,
    bc_dofs: &[usize],
    solid_penalty: &ImmersedSolidPenalty,
    workspace: &sonium_fem::physics::fluid::stokes::workspace::BdfWorkspace,
    transport_config: &KOmegaTransportConfig,
    velocity: &[f64],
    pressure: &[f64],
    k: &[f64],
    omega: &[f64],
    step: usize,
) -> Result<PcFanTransportedMonolithicAssembly, PcFanCfdTransientError> {
    let candidate_turbulence = PcFanCfdKOmegaBdfState {
        k: k.to_vec(),
        k_prev: previous_turbulence.k.clone(),
        omega: omega.to_vec(),
        omega_prev: previous_turbulence.omega.clone(),
        time: previous_turbulence.time,
        step: previous_turbulence.step,
    };
    let fields =
        derive_pc_fan_cfd_komega_element_fields(mesh, velocity, &candidate_turbulence, 10.0)
            .map_err(|_| PcFanCfdTransientError::TransportedTurbulenceFailed { step })?;
    let mut ns_config = base_ns_config.clone();
    ns_config.transported_eddy_viscosity_per_element = Some(fields.eddy_viscosity.clone());
    ns_config.transported_eddy_viscosity = None;
    let candidate_flow = sonium_fem::physics::fluid::stokes::time_stepping::BdfState {
        velocity: velocity.to_vec(),
        velocity_prev: previous_flow.velocity.clone(),
        pressure: pressure.to_vec(),
        time: previous_flow.time,
        step: previous_flow.step,
    };
    let force_penalty = solid_penalty.clone();
    let jacobian_penalty = solid_penalty.clone();
    let (flow_residual, flow_jacobian) =
        sonium_fem::physics::fluid::stokes::navier_stokes::assemble_bdf_newton_linearization(
            mesh,
            velocity_space,
            pressure_space,
            &ns_config,
            bdf,
            &candidate_flow,
            &|_, _, _| vec![0.0, 0.0, 0.0],
            bc_dofs,
            &|coords| {
                force_penalty
                    .force_at(Point3::new(coords[0], coords[1], coords[2]), [0.0; 3])
                    .to_vec()
            },
            &|coords| {
                jacobian_penalty
                    .force_jacobian_at(Point3::new(coords[0], coords[1], coords[2]), [0.0; 3])
            },
            workspace,
            ns_config.mu,
            None,
        )
        .ok_or(PcFanCfdTransientError::StepFailed { step })?;
    let (k_residual, omega_residual, scalar_linearization) =
        assemble_pc_fan_cfd_komega_nonlinear_residual_and_linearization(
            mesh,
            transport_config,
            previous_turbulence,
            &candidate_turbulence,
            &fields,
        )
        .map_err(|_| PcFanCfdTransientError::TransportedTurbulenceFailed { step })?;
    let coupled_linearization = assemble_pc_fan_cfd_komega_coupled_linearization(
        mesh,
        velocity_space,
        pressure_space,
        velocity,
        &candidate_turbulence,
        &fields,
        config.operating_point.air_density_kg_m3,
        10.0,
    )
    .map_err(|_| PcFanCfdTransientError::TransportedTurbulenceFailed { step })?;
    assemble_pc_fan_cfd_komega_monolithic_linearization(
        &flow_jacobian,
        &scalar_linearization,
        &coupled_linearization,
    )
    .map_err(|_| PcFanCfdTransientError::TransportedTurbulenceFailed { step })?;
    let mut residual = flow_residual.clone();
    residual.extend_from_slice(&k_residual);
    residual.extend_from_slice(&omega_residual);
    let residual_norm = residual
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if !residual_norm.is_finite() {
        return Err(PcFanCfdTransientError::StepFailed { step });
    }
    Ok(PcFanTransportedMonolithicAssembly {
        candidate_flow,
        candidate_turbulence,
        flow_residual,
        flow_jacobian,
        scalar_residual: (k_residual, omega_residual),
        scalar_linearization,
        coupled_linearization,
        residual_norm,
    })
}

fn transported_newton_trial_is_accepted(
    current_residual_norm: f64,
    trial_residual_norm: f64,
    damping: f64,
) -> bool {
    if !current_residual_norm.is_finite()
        || current_residual_norm <= 0.0
        || !trial_residual_norm.is_finite()
        || damping <= 0.0
        || damping > 1.0
    {
        return false;
    }
    let ratio = trial_residual_norm / current_residual_norm;
    ratio.is_finite() && ratio * ratio <= 1.0 - 1.0e-4 * damping
}

/// Solve one transported k--omega BDF step with a genuine current-state
/// Newton system. The momentum, scalar, and cross blocks are rebuilt at each
/// iterate; scalar positivity is enforced by a damped correction.
#[allow(clippy::too_many_arguments)]
fn solve_pc_fan_cfd_transported_monolithic_step(
    config: &PcFanConfig,
    mesh: &sonium_fem::mesh::Mesh,
    velocity_space: &VectorH1Space,
    pressure_space: &H1Space,
    bdf: &sonium_fem::physics::fluid::stokes::time_stepping::BdfConfig,
    previous_flow: &sonium_fem::physics::fluid::stokes::time_stepping::BdfState,
    previous_turbulence: &PcFanCfdKOmegaBdfState,
    base_ns_config: &NavierStokesConfig,
    bc_dofs: &[usize],
    solid_penalty: &ImmersedSolidPenalty,
    workspace: &sonium_fem::physics::fluid::stokes::workspace::BdfWorkspace,
    step: usize,
) -> Result<PcFanTransportedMonolithicStep, PcFanCfdTransientError> {
    let scalar_dimension = H1Space::new(PolynomialDegree::P1).num_global_dofs(mesh);
    let flow_dimension =
        velocity_space.num_global_dofs(mesh) + pressure_space.num_global_dofs(mesh);
    let mut velocity = previous_flow.velocity.clone();
    let mut pressure = previous_flow.pressure.clone();
    let mut k = previous_turbulence.k.clone();
    let mut omega = previous_turbulence.omega.clone();
    if scalar_dimension != k.len()
        || scalar_dimension != omega.len()
        || previous_flow.velocity_prev.len() != velocity.len()
    {
        return Err(PcFanCfdTransientError::TransportedTurbulenceFailed { step });
    }

    let transport_config = KOmegaTransportConfig {
        dt: bdf.dt,
        bdf_order: bdf.order,
        molecular_viscosity: config.operating_point.dynamic_viscosity_pa_s
            / config.operating_point.air_density_kg_m3,
        constants: KOmegaSstConstants::default(),
        ..KOmegaTransportConfig::default()
    };
    // A one-iteration legacy Picard setting still needs one Newton
    // correction plus a post-correction residual evaluation.
    let max_iterations = config.cfd.max_picard_iterations.max(2);
    let floor = transport_config.positivity_floor;
    let mut final_assembly = None;
    for _ in 0..max_iterations {
        let candidate_turbulence = PcFanCfdKOmegaBdfState {
            k: k.clone(),
            k_prev: previous_turbulence.k.clone(),
            omega: omega.clone(),
            omega_prev: previous_turbulence.omega.clone(),
            time: previous_turbulence.time,
            step: previous_turbulence.step,
        };
        let fields =
            derive_pc_fan_cfd_komega_element_fields(mesh, &velocity, &candidate_turbulence, 10.0)
                .map_err(|_| PcFanCfdTransientError::TransportedTurbulenceFailed { step })?;
        let mut ns_config = base_ns_config.clone();
        ns_config.transported_eddy_viscosity_per_element = Some(fields.eddy_viscosity.clone());
        ns_config.transported_eddy_viscosity = None;
        let candidate_flow = sonium_fem::physics::fluid::stokes::time_stepping::BdfState {
            velocity: velocity.clone(),
            velocity_prev: previous_flow.velocity.clone(),
            pressure: pressure.clone(),
            time: previous_flow.time,
            step: previous_flow.step,
        };
        let force_penalty = solid_penalty.clone();
        let jacobian_penalty = solid_penalty.clone();
        let (flow_residual, flow_jacobian) =
            sonium_fem::physics::fluid::stokes::navier_stokes::assemble_bdf_newton_linearization(
                mesh,
                velocity_space,
                pressure_space,
                &ns_config,
                bdf,
                &candidate_flow,
                &|_, _, _| vec![0.0, 0.0, 0.0],
                bc_dofs,
                &|coords| {
                    force_penalty
                        .force_at(Point3::new(coords[0], coords[1], coords[2]), [0.0; 3])
                        .to_vec()
                },
                &|coords| {
                    jacobian_penalty
                        .force_jacobian_at(Point3::new(coords[0], coords[1], coords[2]), [0.0; 3])
                },
                workspace,
                ns_config.mu,
                None,
            )
            .ok_or(PcFanCfdTransientError::StepFailed { step })?;
        let (k_residual, omega_residual, scalar_linearization) =
            assemble_pc_fan_cfd_komega_nonlinear_residual_and_linearization(
                mesh,
                &transport_config,
                previous_turbulence,
                &candidate_turbulence,
                &fields,
            )
            .map_err(|_| PcFanCfdTransientError::TransportedTurbulenceFailed { step })?;
        let coupled_linearization = assemble_pc_fan_cfd_komega_coupled_linearization(
            mesh,
            velocity_space,
            pressure_space,
            &velocity,
            &candidate_turbulence,
            &fields,
            config.operating_point.air_density_kg_m3,
            10.0,
        )
        .map_err(|_| PcFanCfdTransientError::TransportedTurbulenceFailed { step })?;
        let monolithic = assemble_pc_fan_cfd_komega_monolithic_linearization(
            &flow_jacobian,
            &scalar_linearization,
            &coupled_linearization,
        )
        .map_err(|_| PcFanCfdTransientError::TransportedTurbulenceFailed { step })?;
        let mut residual = flow_residual.clone();
        residual.extend_from_slice(&k_residual);
        residual.extend_from_slice(&omega_residual);
        let residual_norm = residual
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        final_assembly = Some((
            candidate_flow,
            candidate_turbulence,
            flow_residual,
            flow_jacobian,
            (k_residual, omega_residual),
            scalar_linearization,
            coupled_linearization,
            residual_norm,
        ));
        if residual_norm <= config.cfd.nonlinear_tolerance {
            break;
        }

        let rhs = residual.iter().map(|value| -*value).collect::<Vec<_>>();
        let factor =
            sonium_solvers::direct::SparseLuFactorization::factorize(&monolithic.state_jacobian)
                .map_err(|_| PcFanCfdTransientError::StepFailed { step })?;
        let correction = factor
            .solve(&rhs)
            .map_err(|_| PcFanCfdTransientError::StepFailed { step })?;
        if correction.len() != flow_dimension + 2 * scalar_dimension
            || correction.iter().any(|value| !value.is_finite())
        {
            return Err(PcFanCfdTransientError::StepFailed { step });
        }
        let mut damping = 1.0;
        let mut accepted = false;
        while damping >= 1.0e-8 {
            let trial_k = k
                .iter()
                .enumerate()
                .map(|(index, value)| *value + damping * correction[flow_dimension + index])
                .collect::<Vec<_>>();
            let trial_omega = omega
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    *value + damping * correction[flow_dimension + scalar_dimension + index]
                })
                .collect::<Vec<_>>();
            let trial_velocity = velocity
                .iter()
                .enumerate()
                .map(|(index, value)| *value + damping * correction[index])
                .collect::<Vec<_>>();
            let trial_pressure = pressure
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    *value + damping * correction[velocity_space.num_global_dofs(mesh) + index]
                })
                .collect::<Vec<_>>();
            if trial_k
                .iter()
                .all(|value| value.is_finite() && *value >= floor)
                && trial_omega
                    .iter()
                    .all(|value| value.is_finite() && *value >= floor)
                && trial_velocity.iter().all(|value| value.is_finite())
                && trial_pressure.iter().all(|value| value.is_finite())
            {
                let trial_assembly = assemble_pc_fan_cfd_transported_monolithic_trial(
                    config,
                    mesh,
                    velocity_space,
                    pressure_space,
                    bdf,
                    previous_flow,
                    previous_turbulence,
                    base_ns_config,
                    bc_dofs,
                    solid_penalty,
                    workspace,
                    &transport_config,
                    &trial_velocity,
                    &trial_pressure,
                    &trial_k,
                    &trial_omega,
                    step,
                );
                if let Ok(trial_assembly) = trial_assembly
                    && transported_newton_trial_is_accepted(
                        residual_norm,
                        trial_assembly.residual_norm,
                        damping,
                    )
                {
                    k = trial_k;
                    omega = trial_omega;
                    velocity = trial_velocity;
                    pressure = trial_pressure;
                    accepted = true;
                    break;
                }
            }
            damping *= 0.5;
        }
        if !accepted {
            return Err(PcFanCfdTransientError::StepFailed { step });
        }
        final_assembly = None;
    }

    if final_assembly.is_none() {
        let assembly = assemble_pc_fan_cfd_transported_monolithic_trial(
            config,
            mesh,
            velocity_space,
            pressure_space,
            bdf,
            previous_flow,
            previous_turbulence,
            base_ns_config,
            bc_dofs,
            solid_penalty,
            workspace,
            &transport_config,
            &velocity,
            &pressure,
            &k,
            &omega,
            step,
        )?;
        final_assembly = Some((
            assembly.candidate_flow,
            assembly.candidate_turbulence,
            assembly.flow_residual,
            assembly.flow_jacobian,
            assembly.scalar_residual,
            assembly.scalar_linearization,
            assembly.coupled_linearization,
            assembly.residual_norm,
        ));
    }
    let (
        candidate_flow,
        candidate_turbulence,
        flow_residual,
        flow_jacobian,
        scalar_residual,
        scalar_linearization,
        coupled_linearization,
        residual_norm,
    ) = final_assembly.ok_or(PcFanCfdTransientError::StepFailed { step })?;
    if residual_norm > config.cfd.nonlinear_tolerance {
        return Err(PcFanCfdTransientError::StepFailed { step });
    }
    Ok(PcFanTransportedMonolithicStep {
        flow_state: sonium_fem::physics::fluid::stokes::time_stepping::BdfState {
            velocity: candidate_flow.velocity,
            velocity_prev: previous_flow.velocity.clone(),
            pressure: candidate_flow.pressure,
            time: previous_flow.time + bdf.dt,
            step: previous_flow.step.saturating_add(1),
        },
        turbulence_state: PcFanCfdKOmegaBdfState {
            k: candidate_turbulence.k,
            k_prev: previous_turbulence.k.clone(),
            omega: candidate_turbulence.omega,
            omega_prev: previous_turbulence.omega.clone(),
            time: previous_turbulence.time + bdf.dt,
            step: previous_turbulence.step.saturating_add(1),
        },
        flow_residual,
        flow_jacobian,
        scalar_residual,
        scalar_linearization,
        coupled_linearization,
        residual_norm,
    })
}

fn transient_step_is_admitted(
    solved: bool,
    residual: f64,
    nonlinear_tolerance: f64,
    finite_observables: bool,
    conservation_admitted: bool,
) -> bool {
    solved
        && finite_observables
        && conservation_admitted
        && residual.is_finite()
        && residual <= nonlinear_tolerance
}

/// Require every retained production transient state to satisfy the same
/// nonlinear admission contract used by the forward runner.
///
/// The transient adjoint consumes the published trajectory directly, so a
/// caller could otherwise construct a finite-looking trajectory whose final
/// observation is marked converged while one retained BDF step still exceeds
/// the configured nonlinear residual budget. That would make the adjoint
/// derivative appear valid for a state the forward production path rejected.
fn transient_backend_is_compatible(requested: PcFanCfdBackend, actual: PcFanCfdBackend) -> bool {
    match requested {
        PcFanCfdBackend::Auto => matches!(
            actual,
            PcFanCfdBackend::Cpu | PcFanCfdBackend::Cuda | PcFanCfdBackend::Metal
        ),
        PcFanCfdBackend::Cpu => actual == PcFanCfdBackend::Cpu,
        // The FEM dispatcher explicitly falls back to CPU when an
        // accelerator is unavailable or fails to initialize.
        PcFanCfdBackend::Cuda | PcFanCfdBackend::Metal => {
            actual == requested || actual == PcFanCfdBackend::Cpu
        }
    }
}

fn validate_production_transient_admission(
    config: &PcFanConfig,
    trajectory: &PcFanCfdTransientTrajectory,
) -> Result<(), PcFanResolvedAdjointError> {
    let tolerance = config.cfd.nonlinear_tolerance;
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(PcFanResolvedAdjointError::InvalidConfig(
            "CFD nonlinear tolerance",
            tolerance,
        ));
    }
    if trajectory.steps.iter().any(|step| {
        !transient_backend_is_compatible(config.cfd.backend, step.actual_backend)
            || step.backend_provenance.requested_backend != config.cfd.backend
            || step.backend_provenance.actual_backend != step.actual_backend
            || !step
                .execution_environment
                .matches_provenance(step.backend_provenance)
            || !step.backend_provenance.is_admitted(
                config.cfd.allow_host_fallback,
                config.cfd.require_device_resident,
            )
    }) {
        return Err(PcFanResolvedAdjointError::DimensionMismatch(
            "transient backend provenance",
        ));
    }
    let steps_admitted = trajectory.steps.iter().all(|step| {
        let fields_admitted = transient_step_fields_are_finite(step);
        let state_residual_admitted =
            finite_residual_within_budget(&step.state_residual, tolerance);
        let scalar_residual_admitted = step.transported_turbulence_residual.as_ref().is_none_or(
            |(k_residual, omega_residual)| {
                finite_residual_within_budget(k_residual, tolerance)
                    && finite_residual_within_budget(omega_residual, tolerance)
            },
        );

        fields_admitted
            && state_residual_admitted
            && scalar_residual_admitted
            && first_conservation_violation(config, step.conservation).is_none()
    });

    if trajectory.steps.is_empty()
        || !trajectory.observation.converged
        || !trajectory.observation.residual.is_finite()
        || !transient_observation_fields_are_finite(&trajectory.observation)
        || trajectory.observation.residual > tolerance
        || trajectory
            .steps
            .iter()
            .any(|step| !step.residual_norm.is_finite() || step.residual_norm > tolerance)
        || !steps_admitted
        || first_conservation_violation(config, trajectory.observation.conservation).is_some()
    {
        return Err(PcFanResolvedAdjointError::ProductionStateNotConverged);
    }
    Ok(())
}

fn transient_step_fields_are_finite(step: &PcFanCfdTransientTrajectoryStep) -> bool {
    step.time_s.is_finite()
        && step.state.iter().all(|value| value.is_finite())
        && step.rhs.iter().all(|value| value.is_finite())
        && step.state_residual.iter().all(|value| value.is_finite())
        && step.residual_norm.is_finite()
        && conservation_ledger_fields_are_finite(step.conservation)
        && step.bdf_state.time.is_finite()
        && step
            .bdf_state
            .velocity
            .iter()
            .all(|value| value.is_finite())
        && step
            .bdf_state
            .velocity_prev
            .iter()
            .all(|value| value.is_finite())
        && step
            .bdf_state
            .pressure
            .iter()
            .all(|value| value.is_finite())
        && step.transported_turbulence.as_ref().is_none_or(|state| {
            state.time.is_finite()
                && state.k.iter().all(|value| value.is_finite())
                && state.k_prev.iter().all(|value| value.is_finite())
                && state.omega.iter().all(|value| value.is_finite())
                && state.omega_prev.iter().all(|value| value.is_finite())
        })
        && step
            .transported_eddy_viscosity_per_element
            .as_ref()
            .is_none_or(|values| values.iter().all(|value| value.is_finite()))
        && step.transported_turbulence_residual.as_ref().is_none_or(
            |(k_residual, omega_residual)| {
                k_residual.iter().all(|value| value.is_finite())
                    && omega_residual.iter().all(|value| value.is_finite())
            },
        )
        && step
            .transported_turbulence_residual_design_jacobian
            .as_ref()
            .is_none_or(|rows| rows.iter().flatten().all(|value| value.is_finite()))
        && step
            .residual_design_jacobian
            .iter()
            .flatten()
            .all(|value| value.is_finite())
}

fn transient_observation_fields_are_finite(observation: &PcFanCfdTransientObservation) -> bool {
    observation.final_time_s.is_finite()
        && observation.airflow_cfm.is_finite()
        && observation.pressure_drop_pa.is_finite()
        && observation.axial_loading_n.is_finite()
        && observation.residual.is_finite()
        && conservation_ledger_fields_are_finite(observation.conservation)
}

fn conservation_ledger_fields_are_finite(conservation: PcFanCfdConservation) -> bool {
    [
        conservation.inlet_mass_flow_kg_s,
        conservation.outlet_mass_flow_kg_s,
        conservation.relative_mass_imbalance,
        conservation.inlet_axial_momentum_flux_n,
        conservation.outlet_axial_momentum_flux_n,
        conservation.pressure_force_n,
        conservation.body_force_n,
        conservation.axial_momentum_residual_n,
        conservation.relative_momentum_imbalance,
        conservation.pressure_force_from_loading_n,
        conservation.relative_pressure_closure,
        conservation.pressure_power_w,
        conservation.shaft_power_to_fluid_w,
        conservation.relative_energy_imbalance,
    ]
    .iter()
    .all(|value| value.is_finite())
}

fn finite_residual_within_budget(values: &[f64], budget: f64) -> bool {
    let norm = values.iter().map(|value| value * value).sum::<f64>().sqrt();
    norm.is_finite() && norm <= budget
}

/// Solve a state-only transient adjoint over a production trajectory.
///
/// The trajectory contributes the exact FEM BDF state/history systems. The
/// caller must provide the objective gradient at each retained state; no
/// acoustic or design derivative is inferred from a scalar observation.
pub fn solve_pc_fan_cfd_resolved_transient_state_adjoint(
    adjoint_config: &PcFanResolvedAdjointConfig,
    config: &PcFanConfig,
    trajectory: &PcFanCfdTransientTrajectory,
    objective_state_gradients: &[Vec<f64>],
) -> Result<PcFanResolvedTransientAdjointReport, PcFanResolvedAdjointError> {
    if !matches!(config.cfd.flow_model, PcFanFlowModel::IncompressibleCfdFull) {
        return Err(PcFanResolvedAdjointError::UnsupportedFlowModel(
            config.cfd.flow_model,
        ));
    }

    validate_production_transient_admission(config, trajectory)?;
    // The trajectory currently publishes the laminar production BDF/Oseen
    // blocks.  Algebraic turbulence has a checked steady-state tangent, but
    // it does not yet publish the exact per-step turbulent Newton/BDF
    // history operators required by this transient adjoint.  Keep this
    // state-only entry point fail-closed just like the design-objective
    // wrapper below; otherwise a caller could mistake a frozen turbulent
    // viscosity block for an exact turbulent transient derivative.
    if !matches!(config.cfd.turbulence, PcFanTurbulenceModel::Laminar)
        && trajectory.steps.iter().any(|step| {
            step.state_jacobian_provenance != "fem:navier-stokes:bdf-newton-algebraic-turbulence:v1"
        })
    {
        return Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible);
    }

    if objective_state_gradients.len() != trajectory.steps.len() {
        return Err(PcFanResolvedAdjointError::DimensionMismatch(
            "transient objective gradient count",
        ));
    }
    let steps = trajectory
        .steps
        .iter()
        .zip(objective_state_gradients)
        .map(|(step, objective_state_gradient)| {
            if objective_state_gradient.len() != step.state.len() {
                return Err(PcFanResolvedAdjointError::DimensionMismatch(
                    "transient objective state gradient",
                ));
            }
            Ok(PcFanResolvedTransientStep {
                flow_model: PcFanFlowModel::IncompressibleCfdFull,
                backend: step.actual_backend,
                provenance: format!(
                    "production:pcfan:rotating-wall-bdf-state:v1;state-jacobian:{}",
                    step.state_jacobian_provenance
                ),
                state_jacobian: step.state_jacobian.clone(),
                state_residual: step.state_residual.clone(),
                history_jacobians: step.history_jacobians.clone(),
                residual_design_jacobian: vec![Vec::new(); step.state.len()],
                objective_state_gradient: objective_state_gradient.clone(),
                objective_design_gradient: Vec::new(),
            })
        })
        .collect::<Result<Vec<_>, PcFanResolvedAdjointError>>()?;
    solve_pc_fan_resolved_transient_adjoint(adjoint_config, &steps)
}

/// Solve the production state-only transient adjoint for one checked
/// rotating-wall observation operator.
///
/// This convenience entry point wires the exact production objective state
/// gradients into the multi-step BDF adjoint. This state-only path
/// intentionally does not return a complete design gradient.
/// Solve the full coupled transient state adjoint for transported k--omega.
///
/// Objective gradients are ordered `[flow, k, omega]` for each retained
/// step. The API consumes retained scalar diagonal/off-diagonal blocks, but
/// admits only a genuine monolithic momentum/scalar Newton-BDF flow tangent.
/// The production transported runner publishes a coupled current-state block
/// and is admitted only when this provenance is present on every step.
/// Solve the production resolved transient state adjoint and contract
/// caller-retained BDF history states entering at stage boundaries.
///
/// The history blocks must already be expressed in each stage's state
/// coordinates. This wrapper routes them through the checked generic API; it
/// does not infer derivatives for a remesher, topology choice, or mesh
/// geometry. Use the fixed-map remesh state-adjoint API for the separate
/// state-transfer pairing contract.
pub fn solve_pc_fan_cfd_resolved_transient_state_adjoint_with_stage_initial_history(
    adjoint_config: &PcFanResolvedAdjointConfig,
    config: &PcFanConfig,
    trajectory: &PcFanCfdTransientTrajectory,
    objective_state_gradients: &[Vec<f64>],
    initial_history_jacobians: &[PcFanResolvedTransientStageInitialHistoryJacobian],
) -> Result<PcFanResolvedTransientAdjointWithInitialHistory, PcFanResolvedAdjointError> {
    validate_production_transient_admission(config, trajectory)?;
    if !matches!(config.cfd.flow_model, PcFanFlowModel::IncompressibleCfdFull) {
        return Err(PcFanResolvedAdjointError::UnsupportedFlowModel(
            config.cfd.flow_model,
        ));
    }
    if !matches!(config.cfd.turbulence, PcFanTurbulenceModel::Laminar)
        && trajectory.steps.iter().any(|step| {
            step.state_jacobian_provenance != "fem:navier-stokes:bdf-newton-algebraic-turbulence:v1"
        })
    {
        return Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible);
    }

    if objective_state_gradients.len() != trajectory.steps.len() {
        return Err(PcFanResolvedAdjointError::DimensionMismatch(
            "transient objective gradient count",
        ));
    }
    let steps = trajectory
        .steps
        .iter()
        .zip(objective_state_gradients)
        .map(|(step, objective_state_gradient)| {
            if objective_state_gradient.len() != step.state.len() {
                return Err(PcFanResolvedAdjointError::DimensionMismatch(
                    "transient objective state gradient",
                ));
            }
            Ok(PcFanResolvedTransientStep {
                flow_model: PcFanFlowModel::IncompressibleCfdFull,
                backend: step.actual_backend,
                provenance: format!(
                    "production:pcfan:rotating-wall-bdf-state:v1;state-jacobian:{}",
                    step.state_jacobian_provenance
                ),
                state_jacobian: step.state_jacobian.clone(),
                state_residual: step.state_residual.clone(),
                history_jacobians: step.history_jacobians.clone(),
                residual_design_jacobian: vec![vec![0.0; 0]; step.state.len()],
                objective_state_gradient: objective_state_gradient.clone(),
                objective_design_gradient: Vec::new(),
            })
        })
        .collect::<Result<Vec<_>, PcFanResolvedAdjointError>>()?;

    solve_pc_fan_resolved_transient_adjoint_with_stage_initial_history(
        adjoint_config,
        &steps,
        initial_history_jacobians,
    )
}

/// Solve the production transient state adjoint using the BDF history blocks
/// retained by the forward trajectory itself.
///
/// This is the automatic counterpart to
/// [`solve_pc_fan_cfd_resolved_transient_state_adjoint_with_stage_initial_history`].
/// It is valid only for the fixed-dimensional trajectory segment represented
/// by `trajectory`; remesh transfer and topology/geometry derivatives remain
/// outside this contract.
pub fn solve_pc_fan_cfd_resolved_transient_state_adjoint_with_retained_stage_initial_history(
    adjoint_config: &PcFanResolvedAdjointConfig,
    config: &PcFanConfig,
    trajectory: &PcFanCfdTransientTrajectory,
    objective_state_gradients: &[Vec<f64>],
) -> Result<PcFanResolvedTransientAdjointWithInitialHistory, PcFanResolvedAdjointError> {
    solve_pc_fan_cfd_resolved_transient_state_adjoint_with_stage_initial_history(
        adjoint_config,
        config,
        trajectory,
        objective_state_gradients,
        &trajectory.stage_initial_history_jacobians,
    )
}

fn solve_pc_fan_cfd_resolved_transient_turbulent_adjoint_with_derivatives(
    adjoint_config: &PcFanResolvedAdjointConfig,
    config: &PcFanConfig,
    trajectory: &PcFanCfdTransientTrajectory,
    objective_state_gradients: &[Vec<f64>],
    residual_design_jacobians: Option<&[Vec<Vec<f64>>]>,
    objective_design_gradients: Option<&[Vec<f64>]>,
) -> Result<PcFanResolvedTransientAdjointReport, PcFanResolvedAdjointError> {
    validate_production_transient_admission(config, trajectory)?;
    if !matches!(config.cfd.flow_model, PcFanFlowModel::IncompressibleCfdFull) {
        return Err(PcFanResolvedAdjointError::UnsupportedFlowModel(
            config.cfd.flow_model,
        ));
    }
    if !matches!(
        config.cfd.turbulence,
        PcFanTurbulenceModel::KOmegaSstTransported
    ) {
        return Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible);
    }
    if objective_state_gradients.len() != trajectory.steps.len() {
        return Err(PcFanResolvedAdjointError::DimensionMismatch(
            "turbulent transient objective gradient count",
        ));
    }
    if residual_design_jacobians
        .is_some_and(|derivatives| derivatives.len() != trajectory.steps.len())
        || objective_design_gradients
            .is_some_and(|gradients| gradients.len() != trajectory.steps.len())
    {
        return Err(PcFanResolvedAdjointError::DimensionMismatch(
            "turbulent transient design derivative count",
        ));
    }

    let mut steps = Vec::with_capacity(trajectory.steps.len());
    for (step_index, (step, objective_state_gradient)) in trajectory
        .steps
        .iter()
        .zip(objective_state_gradients)
        .enumerate()
    {
        let flow_provenance = step.state_jacobian_provenance.to_ascii_lowercase();
        if flow_provenance != PC_FAN_TRANSPORTED_MONOLITHIC_BDF_NEWTON_PROVENANCE {
            return Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible);
        }
        let turbulence = step
            .transported_turbulence
            .as_ref()
            .ok_or(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible)?;
        let scalar_linearization = step
            .transported_turbulence_linearization
            .as_ref()
            .ok_or(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible)?;
        let coupled_linearization = step
            .transported_turbulence_coupled_linearization
            .as_ref()
            .ok_or(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible)?;
        let (k_residual, omega_residual) = step
            .transported_turbulence_residual
            .as_ref()
            .ok_or(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible)?;
        let flow_jacobian = sonium_solvers::CsrMatrix::try_from_raw_parts(
            step.state_jacobian.dimension,
            step.state_jacobian.dimension,
            step.state_jacobian.row_ptrs.clone(),
            step.state_jacobian.column_indices.clone(),
            step.state_jacobian.values.clone(),
        )
        .map_err(|_| PcFanResolvedAdjointError::InvalidMatrix("transported flow block"))?;
        let current_state_jacobian = assemble_pc_fan_cfd_komega_monolithic_linearization(
            &flow_jacobian,
            scalar_linearization,
            coupled_linearization,
        )
        .map_err(|_| PcFanResolvedAdjointError::InvalidMatrix("transported current block"))?
        .state_jacobian;
        let flow_dimension = step.state.len();
        let k_dimension = turbulence.k.len();
        let omega_dimension = turbulence.omega.len();
        let full_dimension = flow_dimension
            .checked_add(k_dimension)
            .and_then(|dimension| dimension.checked_add(omega_dimension))
            .ok_or(PcFanResolvedAdjointError::DimensionMismatch(
                "turbulent transient state dimension",
            ))?;
        if objective_state_gradient.len() != full_dimension
            || k_residual.len() != k_dimension
            || omega_residual.len() != omega_dimension
        {
            return Err(PcFanResolvedAdjointError::DimensionMismatch(
                "turbulent transient state gradient or residual",
            ));
        }
        let current_matrix = PcFanResolvedFlowSparseMatrix::new(
            current_state_jacobian.num_rows,
            current_state_jacobian.row_ptrs,
            current_state_jacobian.col_indices,
            current_state_jacobian.values,
        )?;
        let mut state = step.state.clone();
        state.extend_from_slice(&turbulence.k);
        state.extend_from_slice(&turbulence.omega);
        let mut state_residual = step.state_residual.clone();
        state_residual.extend_from_slice(k_residual);
        state_residual.extend_from_slice(omega_residual);
        let mut history_jacobians = Vec::with_capacity(step.history_jacobians.len());
        for history in &step.history_jacobians {
            let Some(history_order) = step.step.checked_sub(history.previous_step) else {
                return Err(PcFanResolvedAdjointError::DimensionMismatch(
                    "turbulent transient history order",
                ));
            };
            if !matches!(history_order, 1 | 2) {
                return Err(PcFanResolvedAdjointError::DimensionMismatch(
                    "turbulent transient history order",
                ));
            }
            let flow_history = sonium_solvers::CsrMatrix::try_from_raw_parts(
                history.matrix.dimension,
                history.matrix.dimension,
                history.matrix.row_ptrs.clone(),
                history.matrix.column_indices.clone(),
                history.matrix.values.clone(),
            )
            .map_err(|_| PcFanResolvedAdjointError::InvalidMatrix("transported history block"))?;
            let transported_history = if flow_provenance
                == PC_FAN_TRANSPORTED_MONOLITHIC_BDF_NEWTON_PROVENANCE.to_ascii_lowercase()
            {
                assemble_pc_fan_cfd_komega_monolithic_history_linearization(
                    flow_dimension,
                    scalar_linearization,
                    history_order,
                )
            } else {
                assemble_pc_fan_cfd_komega_transient_history_linearization(
                    flow_dimension,
                    scalar_linearization,
                    coupled_linearization,
                    history_order,
                )
            }
            .map_err(|_| PcFanResolvedAdjointError::InvalidMatrix("transported history block"))?;
            if history.matrix.dimension != flow_dimension {
                return Err(PcFanResolvedAdjointError::DimensionMismatch(
                    "transported transient flow history block",
                ));
            }
            let total_history_dimension = flow_dimension + k_dimension + omega_dimension;
            let mut triplets = Vec::new();
            for (matrix, row_offset, column_offset) in
                [(&flow_history, 0, 0), (&transported_history, 0, 0)]
            {
                matrix.validate_structure().map_err(|_| {
                    PcFanResolvedAdjointError::InvalidMatrix("transported history block")
                })?;
                for row in 0..matrix.num_rows {
                    for entry in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
                        triplets.push((
                            row_offset + row,
                            column_offset + matrix.col_indices[entry],
                            matrix.values[entry],
                        ));
                    }
                }
            }
            let history_matrix = sonium_solvers::CsrMatrix::try_from_triplets(
                total_history_dimension,
                total_history_dimension,
                triplets,
            )
            .map_err(|_| PcFanResolvedAdjointError::InvalidMatrix("transported history"))?;
            history_jacobians.push(PcFanResolvedTransientHistoryJacobian {
                previous_step: history.previous_step,
                matrix: PcFanResolvedFlowSparseMatrix::new(
                    history_matrix.num_rows,
                    history_matrix.row_ptrs,
                    history_matrix.col_indices,
                    history_matrix.values,
                )?,
            });
        }
        let residual_design_jacobian = if let Some(all_derivatives) = residual_design_jacobians {
            let derivatives = all_derivatives[step_index].clone();
            if derivatives.len() != full_dimension {
                return Err(PcFanResolvedAdjointError::DimensionMismatch(
                    "turbulent transient design residual rows",
                ));
            }
            derivatives
        } else {
            vec![Vec::new(); full_dimension]
        };
        let objective_design_gradient = objective_design_gradients
            .map(|gradients| gradients[step_index].clone())
            .unwrap_or_default();
        steps.push(PcFanResolvedTransientStep {
            flow_model: PcFanFlowModel::IncompressibleCfdFull,
            backend: step.actual_backend,
            provenance:
                "production:pcfan:transported-komega-monolithic-adjoint:v1;block:current-plus-history"
                    .into(),
            state_jacobian: current_matrix,
            state_residual,
            history_jacobians,
            residual_design_jacobian,
            objective_state_gradient: objective_state_gradient.clone(),
            objective_design_gradient,
        });
    }
    solve_pc_fan_resolved_transient_adjoint(adjoint_config, &steps)
}

pub fn solve_pc_fan_cfd_resolved_transient_turbulent_state_adjoint(
    adjoint_config: &PcFanResolvedAdjointConfig,
    config: &PcFanConfig,
    trajectory: &PcFanCfdTransientTrajectory,
    objective_state_gradients: &[Vec<f64>],
) -> Result<PcFanResolvedTransientAdjointReport, PcFanResolvedAdjointError> {
    solve_pc_fan_cfd_resolved_transient_turbulent_adjoint_with_derivatives(
        adjoint_config,
        config,
        trajectory,
        objective_state_gradients,
        None,
        None,
    )
}

pub fn solve_pc_fan_cfd_resolved_transient_objective_adjoint(
    adjoint_config: &PcFanResolvedAdjointConfig,
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    trajectory: &PcFanCfdTransientTrajectory,
    objective: PcFanResolvedTransientObjective,
) -> Result<PcFanResolvedTransientAdjointReport, PcFanResolvedAdjointError> {
    if matches!(
        config.cfd.turbulence,
        PcFanTurbulenceModel::KOmegaSstTransported
    ) {
        return solve_pc_fan_cfd_resolved_transient_objective_adjoint_from_trajectory(
            adjoint_config,
            config,
            design,
            trajectory,
            objective,
        );
    }
    let objective_state_gradients = build_pc_fan_cfd_resolved_transient_objective_state_gradients(
        config, design, trajectory, objective,
    )?;
    solve_pc_fan_cfd_resolved_transient_state_adjoint(
        adjoint_config,
        config,
        trajectory,
        &objective_state_gradients,
    )
}

fn solve_pc_fan_cfd_resolved_transient_turbulent_objective_adjoint_from_trajectory(
    adjoint_config: &PcFanResolvedAdjointConfig,
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    trajectory: &PcFanCfdTransientTrajectory,
    objective: PcFanResolvedTransientObjective,
) -> Result<PcFanResolvedTransientAdjointReport, PcFanResolvedAdjointError> {
    if !matches!(config.cfd.flow_model, PcFanFlowModel::IncompressibleCfdFull) {
        return Err(PcFanResolvedAdjointError::UnsupportedFlowModel(
            config.cfd.flow_model,
        ));
    }
    if trajectory
        .steps
        .iter()
        .any(|step| !step.topology_transfer_adjoint_admissible)
    {
        return Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible);
    }
    let objective_state_gradients = build_pc_fan_cfd_resolved_transient_objective_state_gradients(
        config, design, trajectory, objective,
    )?;
    let full_objective_state_gradients = objective_state_gradients
        .into_iter()
        .zip(&trajectory.steps)
        .map(|(mut gradient, step)| {
            let turbulence = step
                .transported_turbulence
                .as_ref()
                .ok_or(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible)?;
            if gradient.len() != step.state.len() {
                return Err(PcFanResolvedAdjointError::DimensionMismatch(
                    "transported transient flow objective gradient",
                ));
            }
            gradient.extend(std::iter::repeat_n(
                0.0,
                turbulence.k.len().saturating_add(turbulence.omega.len()),
            ));
            Ok(gradient)
        })
        .collect::<Result<Vec<_>, PcFanResolvedAdjointError>>()?;

    let residual_design_jacobians = trajectory
        .steps
        .iter()
        .map(|step| {
            let turbulence = step
                .transported_turbulence
                .as_ref()
                .ok_or(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible)?;
            if step.residual_design_jacobian.len() != step.state.len() {
                return Err(PcFanResolvedAdjointError::DimensionMismatch(
                    "transported transient flow design residual rows",
                ));
            }
            let scalar_rows = step
                .transported_turbulence_residual_design_jacobian
                .as_ref()
                .ok_or(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible)?;
            let scalar_dimension = turbulence.k.len().saturating_add(turbulence.omega.len());
            if scalar_rows.len() != scalar_dimension
                || scalar_rows
                    .iter()
                    .any(|row| row.len() != DESIGN_VARIABLE_COUNT)
            {
                return Err(PcFanResolvedAdjointError::DimensionMismatch(
                    "transported transient scalar design residual rows",
                ));
            }
            let mut rows = step.residual_design_jacobian.clone();
            rows.extend(scalar_rows.iter().cloned());
            Ok(rows)
        })
        .collect::<Result<Vec<_>, PcFanResolvedAdjointError>>()?;

    let state_objective = match objective {
        PcFanResolvedTransientObjective::RotatingWallAxialLoading => {
            Some(PcFanResolvedFlowStateObjective::RotatingWallAxialLoading)
        }
        PcFanResolvedTransientObjective::RotatingWallShaftPower => {
            Some(PcFanResolvedFlowStateObjective::RotatingWallShaftPower)
        }
        PcFanResolvedTransientObjective::Airflow
        | PcFanResolvedTransientObjective::PressureDrop
        | PcFanResolvedTransientObjective::PressureRise => None,
    };
    let window_weights = transient_objective_window_weights(config, trajectory)?;
    let objective_design_gradients = trajectory
        .steps
        .iter()
        .zip(window_weights)
        .map(|(step, window_weight)| {
            state_objective.map_or_else(
                || Ok(vec![0.0; DESIGN_VARIABLE_COUNT]),
                |objective| {
                    if window_weight == 0.0 {
                        return Ok(vec![0.0; DESIGN_VARIABLE_COUNT]);
                    }
                    let gradient =
                        build_pc_fan_cfd_resolved_state_objective_design_gradient_from_state(
                            config,
                            design,
                            &step.state,
                            PolynomialDegree::P2,
                            objective,
                        )?;
                    Ok(gradient
                        .into_iter()
                        .map(|value| value * window_weight)
                        .collect())
                },
            )
        })
        .collect::<Result<Vec<_>, PcFanResolvedAdjointError>>()?;

    solve_pc_fan_cfd_resolved_transient_turbulent_adjoint_with_derivatives(
        adjoint_config,
        config,
        trajectory,
        &full_objective_state_gradients,
        Some(&residual_design_jacobians),
        Some(&objective_design_gradients),
    )
}

/// Solve the production rotating-wall transient objective adjoint.
///
/// The trajectory stores the exact topology-preserving ALE BDF residual
/// design derivative published by the production runner. The rotating-wall
/// objective also contributes analytic blade/shroud occupancy and mesh-motion
/// derivatives at each stored state.
pub fn solve_pc_fan_cfd_resolved_transient_objective_adjoint_from_trajectory(
    adjoint_config: &PcFanResolvedAdjointConfig,
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    trajectory: &PcFanCfdTransientTrajectory,
    objective: PcFanResolvedTransientObjective,
) -> Result<PcFanResolvedTransientAdjointReport, PcFanResolvedAdjointError> {
    if !matches!(config.cfd.flow_model, PcFanFlowModel::IncompressibleCfdFull) {
        return Err(PcFanResolvedAdjointError::UnsupportedFlowModel(
            config.cfd.flow_model,
        ));
    }
    if trajectory
        .steps
        .iter()
        .any(|step| !step.topology_transfer_adjoint_admissible)
    {
        return Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible);
    }
    if matches!(
        config.cfd.turbulence,
        PcFanTurbulenceModel::KOmegaSstTransported
    ) {
        return solve_pc_fan_cfd_resolved_transient_turbulent_objective_adjoint_from_trajectory(
            adjoint_config,
            config,
            design,
            trajectory,
            objective,
        );
    }
    if !matches!(config.cfd.turbulence, PcFanTurbulenceModel::Laminar)
        || design
            .shroud_radius_offsets_m
            .iter()
            .any(|offset| !offset.is_finite())
    {
        return Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible);
    }

    let residual_design_jacobians = trajectory
        .steps
        .iter()
        .map(|step| step.residual_design_jacobian.clone())
        .collect::<Vec<_>>();
    let state_objective = match objective {
        PcFanResolvedTransientObjective::RotatingWallAxialLoading => {
            Some(PcFanResolvedFlowStateObjective::RotatingWallAxialLoading)
        }
        PcFanResolvedTransientObjective::RotatingWallShaftPower => {
            Some(PcFanResolvedFlowStateObjective::RotatingWallShaftPower)
        }
        PcFanResolvedTransientObjective::Airflow
        | PcFanResolvedTransientObjective::PressureDrop
        | PcFanResolvedTransientObjective::PressureRise => None,
    };
    let window_weights = transient_objective_window_weights(config, trajectory)?;
    let objective_design_gradients = trajectory
        .steps
        .iter()
        .zip(window_weights)
        .map(|(step, window_weight)| {
            state_objective.map_or_else(
                || Ok(vec![0.0; DESIGN_VARIABLE_COUNT]),
                |objective| {
                    if window_weight == 0.0 {
                        return Ok(vec![0.0; DESIGN_VARIABLE_COUNT]);
                    }
                    let gradient =
                        build_pc_fan_cfd_resolved_state_objective_design_gradient_from_state(
                            config,
                            design,
                            &step.state,
                            PolynomialDegree::P2,
                            objective,
                        )?;
                    Ok(gradient
                        .into_iter()
                        .map(|value| value * window_weight)
                        .collect())
                },
            )
        })
        .collect::<Result<Vec<_>, PcFanResolvedAdjointError>>()?;

    solve_pc_fan_cfd_resolved_transient_objective_adjoint_with_derivatives(
        adjoint_config,
        config,
        design,
        trajectory,
        objective,
        "analytic:production-frozen-immersed-topology:v2",
        &residual_design_jacobians,
        &objective_design_gradients,
    )
}

/// Solve a checked rotating-wall transient adjoint with caller-supplied
/// analytic design derivatives.
///
/// The caller-supplied arrays remain explicit so this handoff can be used for
/// independently assembled analytic terms without treating a finite-difference
/// probe or a discontinuous penalty-support perturbation as an exact
/// derivative. The production convenience entry point assembles the admitted
/// topology-preserving ALE rotating-wall terms itself.
pub fn solve_pc_fan_cfd_resolved_transient_objective_adjoint_with_derivatives(
    adjoint_config: &PcFanResolvedAdjointConfig,
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    trajectory: &PcFanCfdTransientTrajectory,
    objective: PcFanResolvedTransientObjective,
    derivative_provenance: &str,
    residual_design_jacobians: &[Vec<Vec<f64>>],
    objective_design_gradients: &[Vec<f64>],
) -> Result<PcFanResolvedTransientAdjointReport, PcFanResolvedAdjointError> {
    // This entry point currently packages the flow-only transient state. A
    // non-laminar configuration needs the complete turbulence state and its
    // scalar/ALE design-residual blocks; accepting caller-supplied flow-only
    // derivatives here would publish a false gradient.
    if !matches!(config.cfd.turbulence, PcFanTurbulenceModel::Laminar) {
        return Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible);
    }
    if derivative_provenance.trim().is_empty() {
        return Err(PcFanResolvedAdjointError::MissingProvenance);
    }
    let provenance = derivative_provenance.to_ascii_lowercase();
    if [
        "picard",
        "oseen",
        "derivative-probe",
        "finite-difference",
        "black-box",
    ]
    .iter()
    .any(|marker| provenance.contains(marker))
    {
        return Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible);
    }
    if residual_design_jacobians.len() != trajectory.steps.len()
        || objective_design_gradients.len() != trajectory.steps.len()
    {
        return Err(PcFanResolvedAdjointError::DimensionMismatch(
            "transient design derivative count",
        ));
    }

    let objective_state_gradients = build_pc_fan_cfd_resolved_transient_objective_state_gradients(
        config, design, trajectory, objective,
    )?;
    let steps = trajectory
        .steps
        .iter()
        .zip(objective_state_gradients)
        .zip(residual_design_jacobians)
        .zip(objective_design_gradients)
        .map(
            |(((step, objective_state_gradient), residual_design_jacobian),
              objective_design_gradient)| {
                if residual_design_jacobian.len() != step.state.len() {
                    return Err(PcFanResolvedAdjointError::DimensionMismatch(
                        "transient design residual rows",
                    ));
                }
        Ok(PcFanResolvedTransientStep {
            flow_model: PcFanFlowModel::IncompressibleCfdFull,
            backend: step.actual_backend,
                    provenance: format!(
                        "production:pcfan:rotating-wall-bdf-objective:v1;state-jacobian:{}; derivatives:{derivative_provenance}",
                        step.state_jacobian_provenance
                    ),
                    state_jacobian: step.state_jacobian.clone(),
                    state_residual: step.state_residual.clone(),
                    history_jacobians: step.history_jacobians.clone(),
                    residual_design_jacobian: residual_design_jacobian.clone(),
                    objective_state_gradient,
                    objective_design_gradient: objective_design_gradient.clone(),
                })
            },
        )
        .collect::<Result<Vec<_>, PcFanResolvedAdjointError>>()?;
    solve_pc_fan_resolved_transient_adjoint(adjoint_config, &steps)
}

fn transient_objective_window_weights(
    config: &PcFanConfig,
    trajectory: &PcFanCfdTransientTrajectory,
) -> Result<Vec<f64>, PcFanResolvedAdjointError> {
    let angular_speed = std::f64::consts::TAU * config.operating_point.rpm / 60.0;
    if !angular_speed.is_finite() || angular_speed.abs() <= f64::EPSILON {
        return Err(PcFanResolvedAdjointError::NonFiniteInput(
            "transient objective rotor speed",
        ));
    }
    let final_time = trajectory.steps.last().map(|step| step.time_s).ok_or(
        PcFanResolvedAdjointError::DimensionMismatch("transient trajectory final time"),
    )?;
    if !final_time.is_finite() || final_time <= 0.0 {
        return Err(PcFanResolvedAdjointError::NonFiniteInput(
            "transient objective final time",
        ));
    }
    let window_start = (final_time - std::f64::consts::TAU / angular_speed.abs()).max(0.0);
    let mut selected = Vec::with_capacity(trajectory.steps.len());
    let mut previous_time = None;
    for step in &trajectory.steps {
        if !step.time_s.is_finite() {
            return Err(PcFanResolvedAdjointError::NonFiniteInput(
                "transient objective step time",
            ));
        }
        if previous_time.is_some_and(|previous| step.time_s <= previous) {
            return Err(PcFanResolvedAdjointError::DimensionMismatch(
                "transient objective step times",
            ));
        }
        previous_time = Some(step.time_s);
        selected.push(step.time_s >= window_start);
    }
    if !selected.iter().any(|is_selected| *is_selected) {
        selected.fill(true);
    }

    // Integrate the final-window objective with the trapezoidal rule. Equal
    // weights would make the reported average depend on the number of stored
    // time steps even when the physical interval and trajectory are the same.
    let mut weights = vec![0.0; trajectory.steps.len()];
    for index in 0..trajectory.steps.len().saturating_sub(1) {
        if selected[index] && selected[index + 1] {
            let delta_t = trajectory.steps[index + 1].time_s - trajectory.steps[index].time_s;
            weights[index] += 0.5 * delta_t;
            weights[index + 1] += 0.5 * delta_t;
        }
    }
    let total_weight = weights.iter().sum::<f64>();
    if !total_weight.is_finite() || total_weight <= 0.0 {
        // A window containing only one retained sample is still a valid
        // point observation, rather than an invalid zero-measure integral.
        let last_selected = selected
            .iter()
            .rposition(|is_selected| *is_selected)
            .ok_or(PcFanResolvedAdjointError::DimensionMismatch(
                "transient objective window",
            ))?;
        weights[last_selected] = 1.0;
        return Ok(weights);
    }
    Ok(weights
        .into_iter()
        .map(|weight| weight / total_weight)
        .collect())
}

/// Build exact fixed-mesh objective state gradients for a production
/// rotating-wall trajectory.
///
/// The transient observation is the average over the final rotor-period
/// window, so steps outside that window receive zero objective weight. The
/// returned gradients use the same nodal boundary quadrature and recovered
/// immersed-wall loading integration as the production observation path.
/// They intentionally contain no design gradients: moving a shroud, blade,
/// hub, or other immersed-solid membership changes the discontinuous penalty
/// support and requires a separate shape/residual derivative assembly.
pub fn build_pc_fan_cfd_resolved_transient_objective_state_gradients(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    trajectory: &PcFanCfdTransientTrajectory,
    objective: PcFanResolvedTransientObjective,
) -> Result<Vec<Vec<f64>>, PcFanResolvedAdjointError> {
    if !matches!(config.cfd.flow_model, PcFanFlowModel::IncompressibleCfdFull) {
        return Err(PcFanResolvedAdjointError::UnsupportedFlowModel(
            config.cfd.flow_model,
        ));
    }
    if !trajectory.observation.converged {
        return Err(PcFanResolvedAdjointError::ProductionStateNotConverged);
    }
    if trajectory.steps.is_empty() {
        return Err(PcFanResolvedAdjointError::DimensionMismatch(
            "transient trajectory steps",
        ));
    }
    config
        .validate_geometry_config()
        .map_err(|_| PcFanResolvedAdjointError::ProbeEvaluationFailed("CFD geometry"))?;

    let geometry = config.build_geometry(design);
    let domain = geometry.build_cfd_domain(config);
    let nx = config.cfd.cells_x.max(1);
    let ny = config.cfd.cells_y.max(1);
    let nz = config.cfd.cells_z.max(1);
    let mut mesh = match config.cfd.mesh_pattern {
        PcFanCfdMeshPattern::Symmetric => box_mesh_tetrahedra_symmetric(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            nx,
            ny,
            nz,
        ),
        PcFanCfdMeshPattern::Kuhn => box_mesh_tetrahedra(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            nx,
            ny,
            nz,
        ),
    };
    apply_pc_fan_mesh_motion(&mut mesh, config, &geometry, design)
        .map_err(|_| PcFanResolvedAdjointError::ProbeEvaluationFailed("CFD mesh motion"))?;
    let velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
    let pressure_space = H1Space::new(PolynomialDegree::P1);
    let velocity_dimension = velocity_space.num_global_dofs(&mesh);
    let pressure_dimension = pressure_space.num_global_dofs(&mesh);
    if pressure_dimension != mesh.num_nodes() || velocity_dimension < 3 * mesh.num_nodes() {
        return Err(PcFanResolvedAdjointError::DimensionMismatch(
            "transient nodal state layout",
        ));
    }
    let state_dimension = velocity_dimension + pressure_dimension;
    let nodes = mesh
        .nodes
        .iter()
        .map(|node| Point3::new(node.x, node.y, node.z))
        .collect::<Vec<_>>();
    let boundary_tolerance = boundary_tolerance(&domain, nx, ny, nz);

    let window_weights = transient_objective_window_weights(config, trajectory)?;

    let rotating_penalty = matches!(
        objective,
        PcFanResolvedTransientObjective::RotatingWallAxialLoading
            | PcFanResolvedTransientObjective::RotatingWallShaftPower
    )
    .then(|| {
        ImmersedSolidPenalty::from_geometry_with_design(
            config,
            &geometry,
            solid_boundary_influence_m(&domain, nx, ny, nz),
            Some(design),
        )
    });

    let mut gradients = Vec::with_capacity(trajectory.steps.len());
    for (step, &window_weight) in trajectory.steps.iter().zip(&window_weights) {
        if step.state.len() != state_dimension {
            return Err(PcFanResolvedAdjointError::DimensionMismatch(
                "transient objective state",
            ));
        }
        if step.state_residual.len() != state_dimension {
            return Err(PcFanResolvedAdjointError::DimensionMismatch(
                "transient objective residual",
            ));
        }
        if step.state.iter().any(|value| !value.is_finite()) {
            return Err(PcFanResolvedAdjointError::NonFiniteInput(
                "transient objective state",
            ));
        }
        if step.state_residual.iter().any(|value| !value.is_finite())
            || !step.residual_norm.is_finite()
        {
            return Err(PcFanResolvedAdjointError::NonFiniteInput(
                "transient objective residual",
            ));
        }
        let mut gradient = vec![0.0; state_dimension];
        if window_weight > 0.0 {
            match objective {
                PcFanResolvedTransientObjective::Airflow => {
                    let velocity = velocity_vectors(&step.state[..velocity_dimension], nodes.len());
                    let signed_flow = boundary_airflow_cfm(
                        &nodes,
                        &mesh.elements,
                        &velocity,
                        domain.max_corner.z,
                        domain.outlet_area_m2,
                        boundary_tolerance,
                    );
                    if !signed_flow.is_finite() {
                        return Err(PcFanResolvedAdjointError::NonFiniteInput(
                            "transient airflow objective",
                        ));
                    }
                    if signed_flow.abs() <= f64::EPSILON {
                        return Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible);
                    }
                    let sign = signed_flow.signum();
                    for (node, weight) in boundary_airflow_state_weights(
                        &nodes,
                        &mesh.elements,
                        domain.max_corner.z,
                        domain.outlet_area_m2,
                        boundary_tolerance,
                    )
                    .into_iter()
                    .enumerate()
                    {
                        gradient[3 * node + 2] = window_weight * sign * weight;
                    }
                }
                PcFanResolvedTransientObjective::PressureDrop
                | PcFanResolvedTransientObjective::PressureRise => {
                    let inlet = boundary_average_pressure_state_weights(
                        &nodes,
                        domain.min_corner.z,
                        boundary_tolerance,
                    );
                    let outlet = boundary_average_pressure_state_weights(
                        &nodes,
                        domain.max_corner.z,
                        boundary_tolerance,
                    );
                    if inlet.iter().all(|weight| *weight == 0.0)
                        || outlet.iter().all(|weight| *weight == 0.0)
                    {
                        return Err(PcFanResolvedAdjointError::DimensionMismatch(
                            "transient pressure objective boundary",
                        ));
                    }
                    let pressure_sign =
                        if matches!(objective, PcFanResolvedTransientObjective::PressureDrop) {
                            -1.0
                        } else {
                            1.0
                        };
                    for node in 0..nodes.len() {
                        gradient[velocity_dimension + node] =
                            window_weight * pressure_sign * (outlet[node] - inlet[node]);
                    }
                }
                PcFanResolvedTransientObjective::RotatingWallAxialLoading
                | PcFanResolvedTransientObjective::RotatingWallShaftPower => {
                    let penalty = rotating_penalty
                        .as_ref()
                        .ok_or(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible)?;
                    let state_scale = if matches!(
                        objective,
                        PcFanResolvedTransientObjective::RotatingWallShaftPower
                    ) {
                        penalty.omega_rad_s
                    } else {
                        1.0
                    };
                    for element in &mesh.elements {
                        if element.element_type != sonium_fem::mesh::ElementType::Tetrahedron
                            || element.nodes.len() != 4
                        {
                            continue;
                        }
                        let mut centroid = [0.0; 3];
                        for &node in &element.nodes {
                            let point = &mesh.nodes[node];
                            centroid[0] += point.x * 0.25;
                            centroid[1] += point.y * 0.25;
                            centroid[2] += point.z * 0.25;
                        }
                        let volume = tetrahedron_volume_m3(&mesh, &element.nodes);
                        if volume <= 0.0 {
                            continue;
                        }
                        let jacobian = penalty.rotating_force_jacobian_at(Point3::new(
                            centroid[0],
                            centroid[1],
                            centroid[2],
                        ));
                        let lever_x = centroid[0];
                        let lever_y = centroid[1];
                        for &node in &element.nodes {
                            for input in 0..3 {
                                let derivative = if matches!(
                                    objective,
                                    PcFanResolvedTransientObjective::RotatingWallAxialLoading
                                ) {
                                    jacobian[2][input]
                                } else {
                                    lever_x * jacobian[1][input] - lever_y * jacobian[0][input]
                                };
                                gradient[3 * node + input] +=
                                    window_weight * state_scale * derivative * volume * 0.25;
                            }
                        }
                    }
                }
            }
        }
        if gradient.iter().any(|value| !value.is_finite()) {
            return Err(PcFanResolvedAdjointError::NonFiniteInput(
                "transient objective state gradient",
            ));
        }
        gradients.push(gradient);
    }
    Ok(gradients)
}

fn transient_stage_initial_history_jacobian(
    workspace: &sonium_fem::physics::fluid::stokes::workspace::BdfWorkspace,
    state_jacobian: &PcFanResolvedFlowSparseMatrix,
    boundary_dofs: &[usize],
    rho_over_dt: f64,
    step: usize,
) -> Result<Option<PcFanResolvedTransientStageInitialHistoryJacobian>, PcFanResolvedAdjointError> {
    let (order, history_index, history_slot) = match step {
        0 => (1, 0, 0),
        1 => (2, 1, 1),
        _ => return Ok(None),
    };
    let history = transient_history_jacobians(
        workspace,
        state_jacobian,
        boundary_dofs,
        rho_over_dt,
        sonium_fem::physics::fluid::stokes::time_stepping::BdfCoefficients::for_order(order),
        step + 1,
    )?
    .get(history_index)
    .cloned()
    .ok_or(PcFanResolvedAdjointError::DimensionMismatch(
        "transient initial-history startup block",
    ))?;

    Ok(Some(PcFanResolvedTransientStageInitialHistoryJacobian {
        stage: step,
        history_slot,
        matrix: history.matrix,
        objective_gradient: vec![0.0; state_jacobian.dimension],
    }))
}

fn transient_history_jacobians(
    workspace: &sonium_fem::physics::fluid::stokes::workspace::BdfWorkspace,
    state_jacobian: &PcFanResolvedFlowSparseMatrix,
    boundary_dofs: &[usize],
    rho_over_dt: f64,
    coefficients: sonium_fem::physics::fluid::stokes::time_stepping::BdfCoefficients,
    step: usize,
) -> Result<Vec<PcFanResolvedTransientHistoryJacobian>, PcFanResolvedAdjointError> {
    if !rho_over_dt.is_finite() || rho_over_dt <= 0.0 {
        return Err(PcFanResolvedAdjointError::NonFiniteInput(
            "transient history coefficient",
        ));
    }
    if state_jacobian.dimension != workspace.n_v + workspace.n_p {
        return Err(PcFanResolvedAdjointError::DimensionMismatch(
            "transient history state matrix",
        ));
    }
    let mut is_boundary = vec![false; workspace.n_v];
    for &dof in boundary_dofs {
        if dof >= workspace.n_v {
            return Err(PcFanResolvedAdjointError::DimensionMismatch(
                "transient velocity boundary DOF",
            ));
        }
        is_boundary[dof] = true;
    }
    let mass = workspace.mass_triplet.to_csr();
    let make_history = |previous_step: usize, coefficient: f64| {
        let scale = rho_over_dt * coefficient;
        let mut row_ptrs = Vec::with_capacity(workspace.n_v + workspace.n_p + 1);
        let mut column_indices = Vec::new();
        let mut values = Vec::new();
        row_ptrs.push(0);
        #[expect(
            clippy::needless_range_loop,
            reason = "loop drives row pointers for velocity+pressure rows while the boundary mask covers velocity rows only; the range guard preserves the row count"
        )]
        for row in 0..(workspace.n_v + workspace.n_p) {
            if row < workspace.n_v && !is_boundary[row] {
                for entry in mass.row_ptrs[row]..mass.row_ptrs[row + 1] {
                    column_indices.push(mass.col_indices[entry]);
                    values.push(scale * mass.values[entry]);
                }
            }
            row_ptrs.push(values.len());
        }
        PcFanResolvedFlowSparseMatrix::new(
            workspace.n_v + workspace.n_p,
            row_ptrs,
            column_indices,
            values,
        )
        .map(|matrix| PcFanResolvedTransientHistoryJacobian {
            previous_step,
            matrix,
        })
    };

    let mut history = Vec::new();
    if step >= 1 && coefficients.a1 != 0.0 {
        history.push(make_history(step - 1, coefficients.a1)?);
    }
    if step >= 2 && coefficients.a2 != 0.0 {
        history.push(make_history(step - 2, coefficients.a2)?);
    }
    Ok(history)
}

/// Run the resolved rotating-wall FEM model for a bounded number of rotor
/// revolutions and report a final-one-revolution window average.
///
/// This is the production path used by the time-step and rotor-revolution
/// independence studies.  It intentionally accepts only
/// `IncompressibleCfdFull`: actuator-disk forcing is a steady screening model
/// and cannot be relabelled as a resolved transient rotor solution.
pub fn solve_pc_fan_cfd_rotating_wall_transient(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    time_step_s: f64,
    rotor_revolutions: f64,
) -> Result<PcFanCfdTransientObservation, PcFanCfdTransientError> {
    solve_pc_fan_cfd_rotating_wall_transient_with_trajectory(
        config,
        design,
        time_step_s,
        rotor_revolutions,
    )
    .map(|trajectory| trajectory.observation)
}

/// Run the resolved rotating-wall model and retain the exact boundary-
/// conditioned BDF systems used at every accepted step.
pub fn solve_pc_fan_cfd_rotating_wall_transient_with_trajectory(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    time_step_s: f64,
    rotor_revolutions: f64,
) -> Result<PcFanCfdTransientTrajectory, PcFanCfdTransientError> {
    solve_pc_fan_cfd_rotating_wall_transient_with_initial_state(
        config,
        design,
        time_step_s,
        rotor_revolutions,
        None,
        None,
        None,
    )
}

/// Continue the resolved rotating-wall BDF solve from a state on a different
/// tetrahedral topology.
///
/// The source current and previous P2 velocity levels plus P1 pressure are
/// transferred to the target ALE mesh before the production runner starts.
/// Target BDF matrices, residuals, and per-step Jacobians are then assembled
/// afresh; no topology-dependent operator or turbulence history is reused.
/// For transported k--omega history, use the target-mesh variant below so
/// scalar BDF levels cross the same topology handoff.
pub fn solve_pc_fan_cfd_rotating_wall_transient_with_state_transfer(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    time_step_s: f64,
    rotor_revolutions: f64,
    source_mesh: &sonium_fem::mesh::Mesh,
    source_state: &sonium_fem::physics::fluid::stokes::time_stepping::BdfState,
) -> Result<PcFanCfdTransientTrajectory, PcFanCfdTransientStateTransferError> {
    if !matches!(config.cfd.flow_model, PcFanFlowModel::IncompressibleCfdFull) {
        return Err(PcFanCfdTransientStateTransferError::Solve(
            PcFanCfdTransientError::UnsupportedFlowModel(config.cfd.flow_model),
        ));
    }
    config.validate_geometry_config().map_err(|_| {
        PcFanCfdTransientStateTransferError::Solve(PcFanCfdTransientError::InvalidGeometry)
    })?;

    let geometry = config.build_geometry(design);
    let domain = geometry.build_cfd_domain(config);
    let mut target_mesh = pc_fan_cfd_mesh_for_domain(config, &domain);
    apply_pc_fan_mesh_motion(&mut target_mesh, config, &geometry, design).map_err(|_| {
        PcFanCfdTransientStateTransferError::Solve(PcFanCfdTransientError::InvalidGeometry)
    })?;
    let transferred = transfer_pc_fan_cfd_p2p1_bdf_state(source_mesh, &target_mesh, source_state)
        .map_err(PcFanCfdTransientStateTransferError::StateTransfer)?;

    solve_pc_fan_cfd_rotating_wall_transient_with_initial_state(
        config,
        design,
        time_step_s,
        rotor_revolutions,
        Some(transferred),
        None,
        None,
    )
    .map_err(PcFanCfdTransientStateTransferError::Solve)
}

/// Solve a resolved rotating-wall transient on an externally remeshed target
/// topology, transferring both retained P2/P1 mean-flow BDF levels and the
/// transported P1 k--omega history when that model is enabled.
///
/// The target mesh is treated as the output of a checked body-fitted remesher
/// owned by the caller. Target-dependent FEM spaces, BDF workspace, immersed
/// force samples, and transported momentum coefficients are rebuilt on the
/// target. Because this API does not receive derivatives of the remesher, its
/// trajectory is explicitly not admissible for a full topology-changing
/// design adjoint.
pub fn solve_pc_fan_cfd_rotating_wall_transient_with_state_transfer_to_mesh(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    time_step_s: f64,
    rotor_revolutions: f64,
    source_mesh: &sonium_fem::mesh::Mesh,
    target_mesh: &sonium_fem::mesh::Mesh,
    source_state: &sonium_fem::physics::fluid::stokes::time_stepping::BdfState,
    source_turbulence_state: Option<&PcFanCfdKOmegaBdfState>,
) -> Result<PcFanCfdTransientTrajectory, PcFanCfdTransientStateTransferError> {
    if !matches!(config.cfd.flow_model, PcFanFlowModel::IncompressibleCfdFull) {
        return Err(PcFanCfdTransientStateTransferError::Solve(
            PcFanCfdTransientError::UnsupportedFlowModel(config.cfd.flow_model),
        ));
    }

    let transferred_state =
        transfer_pc_fan_cfd_p2p1_bdf_state(source_mesh, target_mesh, source_state)
            .map_err(PcFanCfdTransientStateTransferError::StateTransfer)?;
    let transferred_turbulence = if matches!(
        config.cfd.turbulence,
        PcFanTurbulenceModel::KOmegaSstTransported
    ) {
        let Some(source_turbulence_state) = source_turbulence_state else {
            return Err(PcFanCfdTransientStateTransferError::Solve(
                PcFanCfdTransientError::InvalidInitialState {
                    reason: "transported transient mesh handoff requires source k--omega history",
                },
            ));
        };
        Some(
            transfer_pc_fan_cfd_komega_bdf_state(source_mesh, target_mesh, source_turbulence_state)
                .map_err(PcFanCfdTransientStateTransferError::KOmegaStateTransfer)?
                .state,
        )
    } else {
        None
    };

    solve_pc_fan_cfd_rotating_wall_transient_with_initial_state(
        config,
        design,
        time_step_s,
        rotor_revolutions,
        Some(transferred_state),
        Some(target_mesh.clone()),
        transferred_turbulence,
    )
    .map_err(PcFanCfdTransientStateTransferError::Solve)
}

/// Run a sequence of resolved rotating-wall transient stages on externally
/// supplied remeshed topologies.
///
/// The first target receives `source_state`; every later target receives the
/// last accepted mean-flow and transported k--omega BDF histories from the
/// preceding stage.  Transfer reports are retained per transition, and each
/// target stage gets a fresh FEM workspace and topology-dependent assembly.
/// The schedule is intentionally not admissible for a full design adjoint
/// because the external remesher's topology derivative is not part of this
/// contract.
pub fn solve_pc_fan_cfd_rotating_wall_transient_with_state_transfer_schedule(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    time_step_s: f64,
    rotor_revolutions: f64,
    source_mesh: &sonium_fem::mesh::Mesh,
    source_state: &PcFanCfdTransientBdfState,
    source_turbulence_state: Option<&PcFanCfdKOmegaBdfState>,
    target_meshes: &[&sonium_fem::mesh::Mesh],
) -> Result<PcFanCfdTransientRemeshSchedule, PcFanCfdTransientStateTransferError> {
    if target_meshes.is_empty() {
        return Err(PcFanCfdTransientStateTransferError::StateTransfer(
            PcFanCfdStateTransferError::InvalidMesh {
                role: "mesh schedule",
                reason: "at least one target mesh is required".into(),
            },
        ));
    }

    if !matches!(config.cfd.flow_model, PcFanFlowModel::IncompressibleCfdFull) {
        return Err(PcFanCfdTransientStateTransferError::Solve(
            PcFanCfdTransientError::UnsupportedFlowModel(config.cfd.flow_model),
        ));
    }

    let transported_turbulence = matches!(
        config.cfd.turbulence,
        PcFanTurbulenceModel::KOmegaSstTransported
    );
    if transported_turbulence && source_turbulence_state.is_none() {
        return Err(PcFanCfdTransientStateTransferError::Solve(
            PcFanCfdTransientError::InvalidInitialState {
                reason: "transported transient remesh schedule requires source k--omega history",
            },
        ));
    }

    let mut current_mesh = source_mesh;
    let mut current_state = source_state.to_fem();
    let mut current_turbulence_state = source_turbulence_state.cloned();
    let mut stages = Vec::with_capacity(target_meshes.len());
    let mut mean_flow_transfer_reports = Vec::with_capacity(target_meshes.len());
    let mut transported_turbulence_transfer_reports =
        transported_turbulence.then(|| Vec::with_capacity(target_meshes.len()));

    for target_mesh in target_meshes {
        let transferred_current = transfer_pc_fan_cfd_p2p1_state(
            current_mesh,
            target_mesh,
            &current_state.velocity,
            &current_state.pressure,
        )
        .map_err(PcFanCfdTransientStateTransferError::StateTransfer)?;
        let transferred_previous = transfer_pc_fan_cfd_p2p1_state(
            current_mesh,
            target_mesh,
            &current_state.velocity_prev,
            &current_state.pressure,
        )
        .map_err(PcFanCfdTransientStateTransferError::StateTransfer)?;
        mean_flow_transfer_reports.push(transferred_current.report);
        let transferred_state = sonium_fem::physics::fluid::stokes::time_stepping::BdfState {
            velocity: transferred_current.velocity_dofs,
            velocity_prev: transferred_previous.velocity_dofs,
            pressure: transferred_current.pressure_dofs,
            time: current_state.time,
            step: current_state.step,
        };

        let transferred_turbulence_state = if transported_turbulence {
            let source_turbulence = current_turbulence_state.as_ref().ok_or(
                PcFanCfdTransientStateTransferError::Solve(
                    PcFanCfdTransientError::InvalidInitialState {
                        reason: "accepted remesh stage omitted transported k--omega history",
                    },
                ),
            )?;
            let transferred =
                transfer_pc_fan_cfd_komega_bdf_state(current_mesh, target_mesh, source_turbulence)
                    .map_err(PcFanCfdTransientStateTransferError::KOmegaStateTransfer)?;
            transported_turbulence_transfer_reports
                .as_mut()
                .expect("transported remesh report storage")
                .push(transferred.report);
            Some(transferred.state)
        } else {
            None
        };

        let trajectory = solve_pc_fan_cfd_rotating_wall_transient_with_initial_state(
            config,
            design,
            time_step_s,
            rotor_revolutions,
            Some(transferred_state),
            Some((*target_mesh).clone()),
            transferred_turbulence_state,
        )
        .map_err(PcFanCfdTransientStateTransferError::Solve)?;

        let last_step =
            trajectory
                .steps
                .last()
                .ok_or(PcFanCfdTransientStateTransferError::Solve(
                    PcFanCfdTransientError::InvalidInitialState {
                        reason: "remesh stage returned no accepted transient state",
                    },
                ))?;
        current_state = last_step.bdf_state.to_fem();
        if transported_turbulence {
            current_turbulence_state = Some(last_step.transported_turbulence.clone().ok_or(
                PcFanCfdTransientStateTransferError::Solve(
                    PcFanCfdTransientError::InvalidInitialState {
                        reason: "accepted remesh stage omitted transported k--omega history",
                    },
                ),
            )?);
        }
        current_mesh = target_mesh;
        stages.push(trajectory);
    }

    Ok(PcFanCfdTransientRemeshSchedule {
        stages,
        mean_flow_transfer_reports,
        transported_turbulence_transfer_reports,
        requires_operator_reassembly: true,
        topology_transfer_adjoint_admissible: false,
        provenance: "pcfan:external-remesh-transient-schedule:v1",
    })
}

/// Run a transient remesh schedule whose target topology is produced one
/// stage at a time by a caller-owned remesher.
///
/// The provider sees the accepted state and mesh from the preceding stage,
/// which permits quality- or physics-triggered body-fitted remeshing.  The
/// provider must return an owned tetrahedral target mesh; state transfer,
/// positivity checks, target assembly, and forward-stage admission remain
/// transactional in this runner.  The provider's topology/design derivative
/// is deliberately not inferred, so the returned schedule remains fail-closed
/// for the complete design adjoint.
#[allow(clippy::too_many_arguments)]
pub fn solve_pc_fan_cfd_rotating_wall_transient_with_state_transfer_with_remesher<F>(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    time_step_s: f64,
    rotor_revolutions: f64,
    source_mesh: &sonium_fem::mesh::Mesh,
    source_state: &PcFanCfdTransientBdfState,
    source_turbulence_state: Option<&PcFanCfdKOmegaBdfState>,
    stage_count: usize,
    mut remesher: F,
) -> Result<PcFanCfdTransientRemeshSchedule, PcFanCfdTransientStateTransferError>
where
    F: FnMut(
        usize,
        &sonium_fem::mesh::Mesh,
        &PcFanCfdTransientBdfState,
        Option<&PcFanCfdKOmegaBdfState>,
    ) -> Result<sonium_fem::mesh::Mesh, String>,
{
    const MAX_REMESH_STAGES: usize = 1024;

    if stage_count == 0 || stage_count > MAX_REMESH_STAGES {
        return Err(PcFanCfdTransientStateTransferError::StateTransfer(
            PcFanCfdStateTransferError::InvalidMesh {
                role: "mesh schedule",
                reason: format!(
                    "stage count must be in 1..={MAX_REMESH_STAGES}, got {stage_count}"
                ),
            },
        ));
    }
    if !matches!(config.cfd.flow_model, PcFanFlowModel::IncompressibleCfdFull) {
        return Err(PcFanCfdTransientStateTransferError::Solve(
            PcFanCfdTransientError::UnsupportedFlowModel(config.cfd.flow_model),
        ));
    }

    let transported_turbulence = matches!(
        config.cfd.turbulence,
        PcFanTurbulenceModel::KOmegaSstTransported
    );
    if transported_turbulence && source_turbulence_state.is_none() {
        return Err(PcFanCfdTransientStateTransferError::Solve(
            PcFanCfdTransientError::InvalidInitialState {
                reason: "transported transient remesh schedule requires source k--omega history",
            },
        ));
    }

    let mut current_mesh = source_mesh.clone();
    let mut current_state = source_state.to_fem();
    let mut current_turbulence_state = source_turbulence_state.cloned();
    let mut stages = Vec::with_capacity(stage_count);
    let mut mean_flow_transfer_reports = Vec::with_capacity(stage_count);
    let mut transported_turbulence_transfer_reports =
        transported_turbulence.then(|| Vec::with_capacity(stage_count));

    for stage in 0..stage_count {
        let public_state = PcFanCfdTransientBdfState::from_fem(&current_state);
        let target_mesh = remesher(
            stage,
            &current_mesh,
            &public_state,
            current_turbulence_state.as_ref(),
        )
        .map_err(|reason| PcFanCfdTransientStateTransferError::Remesher { stage, reason })?;

        let transferred_current = transfer_pc_fan_cfd_p2p1_state(
            &current_mesh,
            &target_mesh,
            &current_state.velocity,
            &current_state.pressure,
        )
        .map_err(PcFanCfdTransientStateTransferError::StateTransfer)?;
        let transferred_previous = transfer_pc_fan_cfd_p2p1_state(
            &current_mesh,
            &target_mesh,
            &current_state.velocity_prev,
            &current_state.pressure,
        )
        .map_err(PcFanCfdTransientStateTransferError::StateTransfer)?;
        mean_flow_transfer_reports.push(transferred_current.report);
        let transferred_state = sonium_fem::physics::fluid::stokes::time_stepping::BdfState {
            velocity: transferred_current.velocity_dofs,
            velocity_prev: transferred_previous.velocity_dofs,
            pressure: transferred_current.pressure_dofs,
            time: current_state.time,
            step: current_state.step,
        };

        let transferred_turbulence_state = if transported_turbulence {
            let source_turbulence = current_turbulence_state.as_ref().ok_or(
                PcFanCfdTransientStateTransferError::Solve(
                    PcFanCfdTransientError::InvalidInitialState {
                        reason: "accepted remesh stage omitted transported k--omega history",
                    },
                ),
            )?;
            let transferred = transfer_pc_fan_cfd_komega_bdf_state(
                &current_mesh,
                &target_mesh,
                source_turbulence,
            )
            .map_err(PcFanCfdTransientStateTransferError::KOmegaStateTransfer)?;
            transported_turbulence_transfer_reports
                .as_mut()
                .expect("transported remesh report storage")
                .push(transferred.report);
            Some(transferred.state)
        } else {
            None
        };

        let trajectory = solve_pc_fan_cfd_rotating_wall_transient_with_initial_state(
            config,
            design,
            time_step_s,
            rotor_revolutions,
            Some(transferred_state),
            Some(target_mesh.clone()),
            transferred_turbulence_state,
        )
        .map_err(PcFanCfdTransientStateTransferError::Solve)?;

        let last_step =
            trajectory
                .steps
                .last()
                .ok_or(PcFanCfdTransientStateTransferError::Solve(
                    PcFanCfdTransientError::InvalidInitialState {
                        reason: "remesh stage returned no accepted transient state",
                    },
                ))?;
        current_state = last_step.bdf_state.to_fem();
        if transported_turbulence {
            current_turbulence_state = Some(last_step.transported_turbulence.clone().ok_or(
                PcFanCfdTransientStateTransferError::Solve(
                    PcFanCfdTransientError::InvalidInitialState {
                        reason: "accepted remesh stage omitted transported k--omega history",
                    },
                ),
            )?);
        }
        current_mesh = target_mesh;
        stages.push(trajectory);
    }

    Ok(PcFanCfdTransientRemeshSchedule {
        stages,
        mean_flow_transfer_reports,
        transported_turbulence_transfer_reports,
        requires_operator_reassembly: true,
        topology_transfer_adjoint_admissible: false,
        provenance: "pcfan:external-remesh-provider-transient-schedule:v1",
    })
}

/// One transient conservation observation: `(time_s, outlet_airflow_cfm_abs,
/// pressure_drop_pa, axial_force_n, residual_norm, conservation, admitted)`.
type ConservationObservation = (f64, f64, f64, f64, f64, PcFanCfdConservation, bool);

fn solve_pc_fan_cfd_rotating_wall_transient_with_initial_state(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    time_step_s: f64,
    rotor_revolutions: f64,
    initial_state: Option<sonium_fem::physics::fluid::stokes::time_stepping::BdfState>,
    initial_mesh: Option<sonium_fem::mesh::Mesh>,
    initial_turbulence_state: Option<PcFanCfdKOmegaBdfState>,
) -> Result<PcFanCfdTransientTrajectory, PcFanCfdTransientError> {
    if !matches!(config.cfd.flow_model, PcFanFlowModel::IncompressibleCfdFull) {
        return Err(PcFanCfdTransientError::UnsupportedFlowModel(
            config.cfd.flow_model,
        ));
    }
    let nonlinear_tolerance = config.cfd.nonlinear_tolerance;
    if !nonlinear_tolerance.is_finite() || nonlinear_tolerance <= 0.0 {
        return Err(PcFanCfdTransientError::InvalidNonlinearTolerance {
            tolerance: nonlinear_tolerance,
        });
    }
    if let Some((metric, tolerance)) = first_invalid_cfd_conservation_tolerance(config) {
        return Err(PcFanCfdTransientError::InvalidConservationTolerance { metric, tolerance });
    }
    if !time_step_s.is_finite() || time_step_s <= 0.0 {
        return Err(PcFanCfdTransientError::InvalidTimeStep);
    }
    if !rotor_revolutions.is_finite() || rotor_revolutions <= 0.0 {
        return Err(PcFanCfdTransientError::InvalidRotorRevolutions);
    }
    config
        .validate_geometry_config()
        .map_err(|_| PcFanCfdTransientError::InvalidGeometry)?;

    let geometry = config.build_geometry(design);
    let domain = geometry.build_cfd_domain(config);
    let nx = config.cfd.cells_x.max(1);
    let ny = config.cfd.cells_y.max(1);
    let nz = config.cfd.cells_z.max(1);
    let externally_remeshed = initial_mesh.is_some();
    let mut mesh = initial_mesh.unwrap_or_else(|| match config.cfd.mesh_pattern {
        PcFanCfdMeshPattern::Symmetric => box_mesh_tetrahedra_symmetric(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            nx,
            ny,
            nz,
        ),
        PcFanCfdMeshPattern::Kuhn => box_mesh_tetrahedra(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            nx,
            ny,
            nz,
        ),
    });
    let mesh_motion_design_jacobian = if externally_remeshed {
        // An externally supplied body-fitted topology has no declared design
        // motion map at this API boundary. Keep the restart solve usable, but
        // mark its trajectory non-admissible for a full design adjoint.
        vec![[[0.0; 3]; DESIGN_VARIABLE_COUNT]; mesh.nodes.len()]
    } else {
        pc_fan_mesh_motion_design_jacobian(&mesh, config, design)
            .map_err(|_| PcFanCfdTransientError::InvalidGeometry)?
    };
    if !externally_remeshed {
        apply_pc_fan_mesh_motion(&mut mesh, config, &geometry, design)
            .map_err(|_| PcFanCfdTransientError::InvalidGeometry)?;
    }
    let velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
    let pressure_space = H1Space::new(PolynomialDegree::P1);
    let velocity_dimension = velocity_space.num_global_dofs(&mesh);
    let pressure_dimension = pressure_space.num_global_dofs(&mesh);
    let boundary_tolerance = boundary_tolerance(&domain, nx, ny, nz);
    let bc_dofs = locate_dofs_geometrical(&mesh, &velocity_space, |x, y, _| {
        is_side_wall(&domain, x, y, boundary_tolerance)
    });
    let solid_penalty = ImmersedSolidPenalty::from_geometry_with_design(
        config,
        &geometry,
        solid_boundary_influence_m(&domain, nx, ny, nz),
        Some(design),
    );
    let mut ns_config = NavierStokesConfig {
        mu: config.operating_point.dynamic_viscosity_pa_s,
        rho: config.operating_point.air_density_kg_m3,
        linear_solver_max_iterations: config.cfd.linear_solver_max_iterations.max(1),
        linear_solver_tolerance: config.cfd.linear_solver_tolerance,
        linear_solver_cpu_preconditioner: config.cfd.cpu_preconditioner.into_fem(),
        linear_solver_backend: config.cfd.backend.into_fem(),
        linear_solver_backend_preconditioner: config.cfd.precond_kind.into_fem(),
        linear_solver_polynomial_degree: config.cfd.precond_polynomial_degree.max(1),
        linear_solver_polynomial_damping: config.cfd.precond_polynomial_damping,
        linear_solver_schur_polynomial_degree: config.cfd.precond_schur_degree.max(1),
        linear_solver_schur_polynomial_damping: config.cfd.precond_schur_damping,
        linear_solver_schur_preconditioner: config.cfd.precond_schur_kind.into_fem(),
        linear_solver_schur_chebyshev_degree: config.cfd.precond_schur_chebyshev_degree.max(1),
        linear_solver_schur_chebyshev_eigenvalue_ratio: config
            .cfd
            .precond_schur_chebyshev_eigenvalue_ratio
            .max(1.0),
        linear_solver_chebyshev_degree: config.cfd.precond_chebyshev_degree.max(1),
        linear_solver_chebyshev_eigenvalue_ratio: config
            .cfd
            .precond_chebyshev_eigenvalue_ratio
            .max(1.0),
        linear_solver_kind: config.cfd.solver_kind.into_fem(),
        linear_solver_ir_inner_max_iterations: config.cfd.solver_ir_inner_max_iterations.max(1),
        linear_solver_ir_inner_tolerance: config.cfd.solver_ir_inner_tolerance,
        turbulence: config.cfd.turbulence.into_fem(),
        ..NavierStokesConfig::default()
    };
    // The BDF path advances one linearised step at a time; these steady Picard
    // controls must not accidentally turn it into a hidden convergence loop.
    ns_config.max_picard_iterations = 1;
    ns_config.min_picard_iterations = 1;

    let angular_speed = std::f64::consts::TAU * config.operating_point.rpm / 60.0;
    if !angular_speed.is_finite() || angular_speed.abs() <= f64::EPSILON {
        return Err(PcFanCfdTransientError::InvalidRotorRevolutions);
    }
    let duration_s = rotor_revolutions * std::f64::consts::TAU / angular_speed.abs();
    let step_count = (duration_s / time_step_s).ceil().max(2.0) as usize;
    if step_count > 2_000 {
        return Err(PcFanCfdTransientError::TooManySteps);
    }
    let bdf =
        sonium_fem::physics::fluid::stokes::time_stepping::BdfConfig::bdf2(time_step_s, step_count);
    let mut state = initial_state.unwrap_or_else(|| {
        sonium_fem::physics::fluid::stokes::time_stepping::BdfState::from_rest(
            velocity_dimension,
            pressure_dimension,
        )
    });
    if state.velocity.len() != velocity_dimension
        || state.velocity_prev.len() != velocity_dimension
        || state.pressure.len() != pressure_dimension
    {
        return Err(PcFanCfdTransientError::InvalidInitialState {
            reason: "P2 velocity/P1 pressure DOF dimensions do not match target mesh",
        });
    }
    if !state.time.is_finite() || state.time < 0.0 {
        return Err(PcFanCfdTransientError::InvalidInitialState {
            reason: "time must be finite and non-negative",
        });
    }
    if state
        .velocity
        .iter()
        .chain(&state.velocity_prev)
        .chain(&state.pressure)
        .any(|value| !value.is_finite())
    {
        return Err(PcFanCfdTransientError::InvalidInitialState {
            reason: "velocity and pressure DOFs must be finite",
        });
    }
    let initial_time_s = state.time;
    let mut transported_turbulence_state = if let Some(transferred) = initial_turbulence_state {
        let expected = H1Space::new(PolynomialDegree::P1).num_global_dofs(&mesh);
        let valid = matches!(
            config.cfd.turbulence,
            PcFanTurbulenceModel::KOmegaSstTransported
        ) && transferred.k.len() == expected
            && transferred.k_prev.len() == expected
            && transferred.omega.len() == expected
            && transferred.omega_prev.len() == expected
            && transferred.time.is_finite()
            && transferred.time >= 0.0
            && transferred.step == state.step
            && (transferred.time - initial_time_s).abs() <= 1.0e-12 * (1.0 + initial_time_s.abs())
            && transferred
                .k
                .iter()
                .chain(&transferred.k_prev)
                .all(|value| value.is_finite() && *value >= 0.0)
            && transferred
                .omega
                .iter()
                .chain(&transferred.omega_prev)
                .all(|value| value.is_finite() && *value > 0.0);
        if !valid {
            return Err(PcFanCfdTransientError::InvalidInitialState {
                reason: "transferred transported k--omega state has invalid target dimensions or values",
            });
        }
        Some(transferred)
    } else if matches!(
        config.cfd.turbulence,
        PcFanTurbulenceModel::KOmegaSstTransported
    ) {
        Some(
            initialize_pc_fan_cfd_komega_bdf_state(&mesh, initial_time_s, 0, 1.0e-6, 1.0).map_err(
                |_| PcFanCfdTransientError::InvalidInitialState {
                    reason: "could not initialise transported k--omega state",
                },
            )?,
        )
    } else {
        None
    };
    let workspace = sonium_fem::physics::fluid::stokes::workspace::BdfWorkspace::new(
        &mesh,
        &velocity_space,
        &pressure_space,
    );
    let nodes = mesh
        .nodes
        .iter()
        .map(|node| Point3::new(node.x, node.y, node.z))
        .collect::<Vec<_>>();
    let mut observations = Vec::with_capacity(step_count);
    let mut trajectory_steps = Vec::with_capacity(step_count);
    let mut stage_initial_history_jacobians = Vec::new();
    for step in 0..step_count {
        let mut transported_turbulence_linearization = None;
        let mut transported_turbulence_coupled_linearization = None;
        let mut transported_turbulence_residual = None;
        let mut transported_turbulence_history = None;
        let mut transported_monolithic_step = None;
        if matches!(
            config.cfd.turbulence,
            PcFanTurbulenceModel::KOmegaSstTransported
        ) {
            let previous_turbulence_state = transported_turbulence_state
                .as_ref()
                .ok_or(PcFanCfdTransientError::TransportedTurbulenceFailed { step })?;
            transported_turbulence_history = Some(previous_turbulence_state.clone());
            ns_config.transported_eddy_viscosity = None;
            ns_config.transported_eddy_viscosity_per_element = None;
            let monolithic = solve_pc_fan_cfd_transported_monolithic_step(
                config,
                &mesh,
                &velocity_space,
                &pressure_space,
                &bdf,
                &state,
                previous_turbulence_state,
                &ns_config,
                &bc_dofs,
                &solid_penalty,
                &workspace,
                step,
            )?;
            state = monolithic.flow_state.clone();
            transported_turbulence_state = Some(monolithic.turbulence_state.clone());
            transported_turbulence_linearization = Some(monolithic.scalar_linearization.clone());
            transported_turbulence_coupled_linearization =
                Some(monolithic.coupled_linearization.clone());
            transported_turbulence_residual = Some(monolithic.scalar_residual.clone());
            transported_monolithic_step = Some(monolithic);
        }
        let applied_transported_eddy_viscosity_per_element =
            if let Some(monolithic) = transported_monolithic_step.as_ref() {
                derive_pc_fan_cfd_komega_element_fields(
                    &mesh,
                    &state.velocity,
                    &monolithic.turbulence_state,
                    10.0,
                )
                .map_err(|_| PcFanCfdTransientError::TransportedTurbulenceFailed { step })?
                .eddy_viscosity
                .into()
            } else if let Some(turbulence_state) = transported_turbulence_state.as_ref() {
                apply_pc_fan_cfd_komega_mean_eddy_viscosity(
                    &mut ns_config,
                    &mesh,
                    turbulence_state,
                    10.0,
                )
                .map_err(|_| PcFanCfdTransientError::TransportedTurbulenceFailed { step })?;
                ns_config.transported_eddy_viscosity_per_element.clone()
            } else {
                None
            };
        let force_penalty = solid_penalty.clone();
        let jacobian_penalty = solid_penalty.clone();
        let solved = if transported_monolithic_step.is_some() {
            true
        } else {
            sonium_fem::physics::fluid::stokes::navier_stokes::solve_bdf_step_cached_with_force(
                &mesh,
                &velocity_space,
                &pressure_space,
                &ns_config,
                &bdf,
                &mut state,
                &|_, _, _| vec![0.0, 0.0, 0.0],
                &bc_dofs,
                &[],
                move |coords| {
                    force_penalty
                        .force_at(Point3::new(coords[0], coords[1], coords[2]), [0.0; 3])
                        .to_vec()
                },
                move |coords| {
                    jacobian_penalty
                        .force_jacobian_at(Point3::new(coords[0], coords[1], coords[2]), [0.0; 3])
                },
                &workspace,
            )
        };
        if !solved {
            return Err(PcFanCfdTransientError::StepFailed { step });
        }
        if transported_monolithic_step.is_none()
            && let Some(previous_turbulence_state) = transported_turbulence_state.as_ref()
        {
            let transport_config = KOmegaTransportConfig {
                dt: time_step_s,
                bdf_order: if step == 0 { 1 } else { 2 },
                molecular_viscosity: config.operating_point.dynamic_viscosity_pa_s
                    / config.operating_point.air_density_kg_m3,
                constants: KOmegaSstConstants::default(),
                ..KOmegaTransportConfig::default()
            };
            let fields = derive_pc_fan_cfd_komega_element_fields(
                &mesh,
                &state.velocity,
                previous_turbulence_state,
                10.0,
            )
            .map_err(|_| PcFanCfdTransientError::TransportedTurbulenceFailed { step })?;
            let updated = advance_pc_fan_cfd_komega_bdf_step_with_element_fields(
                &mesh,
                &transport_config,
                previous_turbulence_state,
                &fields,
            )
            .map_err(|_| PcFanCfdTransientError::TransportedTurbulenceFailed { step })?;
            transported_turbulence_coupled_linearization = Some(
                assemble_pc_fan_cfd_komega_coupled_linearization(
                    &mesh,
                    &velocity_space,
                    &pressure_space,
                    &state.velocity,
                    previous_turbulence_state,
                    &fields,
                    config.operating_point.air_density_kg_m3,
                    10.0,
                )
                .map_err(|_| PcFanCfdTransientError::TransportedTurbulenceFailed { step })?,
            );
            transported_turbulence_linearization = Some(updated.linearization);
            transported_turbulence_residual = Some((updated.k_residual, updated.omega_residual));
            transported_turbulence_state = Some(updated.state);
        }
        let state_vector = state
            .velocity
            .iter()
            .chain(&state.pressure)
            .copied()
            .collect::<Vec<_>>();
        let (lagged_state_jacobian, lagged_state_residual, rhs) =
            if let Some(monolithic) = transported_monolithic_step.as_ref() {
                let jacobian = PcFanResolvedFlowSparseMatrix::new(
                    monolithic.flow_jacobian.num_rows,
                    monolithic.flow_jacobian.row_ptrs.clone(),
                    monolithic.flow_jacobian.col_indices.clone(),
                    monolithic.flow_jacobian.values.clone(),
                )
                .map_err(|_| PcFanCfdTransientError::InvalidLinearSystem { step })?;
                let rhs = monolithic
                    .flow_jacobian
                    .apply(&state_vector)
                    .iter()
                    .zip(&monolithic.flow_residual)
                    .map(|(left, residual)| left - residual)
                    .collect::<Vec<_>>();
                (jacobian, monolithic.flow_residual.clone(), rhs)
            } else {
                let (linear_system, rhs) = workspace
                    .last_linear_system()
                    .ok_or(PcFanCfdTransientError::MissingLinearSystem { step })?;
                let residual = linear_system
                    .apply(&state_vector)
                    .iter()
                    .zip(&rhs)
                    .map(|(left, right)| left - right)
                    .collect::<Vec<_>>();
                let jacobian = PcFanResolvedFlowSparseMatrix::new(
                    linear_system.num_rows,
                    linear_system.row_ptrs.clone(),
                    linear_system.col_indices.clone(),
                    linear_system.values.clone(),
                )
                .map_err(|_| PcFanCfdTransientError::InvalidLinearSystem { step })?;
                (jacobian, residual, rhs)
            };

        let (state_jacobian, state_residual, state_jacobian_provenance) =
            if transported_monolithic_step.is_some() {
                (
                    lagged_state_jacobian.clone(),
                    lagged_state_residual.clone(),
                    PC_FAN_TRANSPORTED_MONOLITHIC_BDF_NEWTON_PROVENANCE.to_string(),
                )
            } else if matches!(config.cfd.turbulence, PcFanTurbulenceModel::Laminar) {
                let exact = sonium_fem::physics::fluid::stokes::navier_stokes::assemble_bdf_newton_linearization(
                    &mesh,
                    &velocity_space,
                    &pressure_space,
                    &ns_config,
                    &bdf,
                    &state,
                    &|_, _, _| vec![0.0, 0.0, 0.0],
                    &bc_dofs,
                    &|coords| {
                        solid_penalty
                            .force_at(Point3::new(coords[0], coords[1], coords[2]), [0.0; 3])
                            .to_vec()
                    },
                    &|coords| {
                        solid_penalty
                            .force_jacobian_at(Point3::new(coords[0], coords[1], coords[2]), [0.0; 3])
                    },
                    &workspace,
                    ns_config.mu,
                    None,
                );
                exact
                    .and_then(|(residual, jacobian)| {
                        PcFanResolvedFlowSparseMatrix::new(
                            jacobian.num_rows,
                            jacobian.row_ptrs,
                            jacobian.col_indices,
                            jacobian.values,
                        )
                        .ok()
                        .map(|matrix| {
                            (
                                matrix,
                                residual,
                                "fem:navier-stokes:bdf-newton-laminar:v1".to_string(),
                            )
                        })
                    })
                    .unwrap_or((
                        lagged_state_jacobian.clone(),
                        lagged_state_residual.clone(),
                        "fem:navier-stokes:bdf-oseen-laminar-fallback:v1".to_string(),
                    ))
    } else if matches!(
        config.cfd.turbulence,
        PcFanTurbulenceModel::KOmegaSstTransported
    ) {
        let exact = sonium_fem::physics::fluid::stokes::navier_stokes::assemble_bdf_newton_linearization(
            &mesh,
            &velocity_space,
            &pressure_space,
            &ns_config,
            &bdf,
            &state,
            &|_, _, _| vec![0.0, 0.0, 0.0],
            &bc_dofs,
            &|coords| {
                solid_penalty
                    .force_at(
                        Point3::new(coords[0], coords[1], coords[2]),
                        [0.0; 3],
                    )
                    .to_vec()
            },
            &|coords| {
                solid_penalty.force_jacobian_at(
                    Point3::new(coords[0], coords[1], coords[2]),
                    [0.0; 3],
                )
            },
            &workspace,
            ns_config.mu,
            None,
        );
        exact
            .and_then(|(residual, jacobian)| {
                PcFanResolvedFlowSparseMatrix::new(
                    jacobian.num_rows,
                    jacobian.row_ptrs,
                    jacobian.col_indices,
                    jacobian.values,
                )
                .ok()
                .map(|matrix| {
                    (
                        matrix,
                        residual,
                        PC_FAN_TRANSPORTED_BDF_NEWTON_PROVENANCE.to_string(),
                    )
                })
            })
            .unwrap_or((
                lagged_state_jacobian.clone(),
                lagged_state_residual.clone(),
                "fem:navier-stokes:bdf-oseen-transported-fallback:v1".to_string(),
            ))
    } else if let Some((peak, peak_gradient)) =
                sonium_fem::physics::fluid::stokes::turbulence::TurbulenceModel::unique_peak_velocity_subgradient(
                    &state.velocity,
                    3,
                )
            {
                let (effective_mu, d_mu_d_speed) = ns_config
                    .turbulence
                    .effective_mu_and_speed_derivative(ns_config.mu, ns_config.rho, peak);
                let gradient = peak_gradient
                    .into_iter()
                    .map(|value| value * d_mu_d_speed)
                    .collect::<Vec<_>>();
                let exact = sonium_fem::physics::fluid::stokes::navier_stokes::assemble_bdf_newton_linearization(
                    &mesh,
                    &velocity_space,
                    &pressure_space,
                    &ns_config,
                    &bdf,
                    &state,
                    &|_, _, _| vec![0.0, 0.0, 0.0],
                    &bc_dofs,
                    &|coords| {
                        solid_penalty
                            .force_at(Point3::new(coords[0], coords[1], coords[2]), [0.0; 3])
                            .to_vec()
                    },
                    &|coords| {
                        solid_penalty
                            .force_jacobian_at(Point3::new(coords[0], coords[1], coords[2]), [0.0; 3])
                    },
                    &workspace,
                    effective_mu,
                    Some(&gradient),
                );
                exact
                    .and_then(|(residual, jacobian)| {
                        PcFanResolvedFlowSparseMatrix::new(
                            jacobian.num_rows,
                            jacobian.row_ptrs,
                            jacobian.col_indices,
                            jacobian.values,
                        )
                        .ok()
                        .map(|matrix| {
                            (
                                matrix,
                                residual,
                                "fem:navier-stokes:bdf-newton-algebraic-turbulence:v1"
                                    .to_string(),
                            )
                        })
                    })
                    .unwrap_or((
                        lagged_state_jacobian.clone(),
                        lagged_state_residual.clone(),
                        "fem:navier-stokes:bdf-oseen-lagged-turbulence:v1".to_string(),
                    ))
            } else {
                (
                    lagged_state_jacobian.clone(),
                    lagged_state_residual.clone(),
                    "fem:navier-stokes:bdf-oseen-lagged-turbulence:v1".to_string(),
                )
            };
        let actual_backend = transported_monolithic_step
            .as_ref()
            .map(|_| PcFanCfdBackend::Cpu)
            .or_else(|| {
                workspace.last_linear_solve_telemetry().map(|telemetry| {
                    PcFanCfdBackend::from_fem(telemetry.resolved_linear_solver_backend)
                })
            })
            .unwrap_or(PcFanCfdBackend::Cpu);
        let backend_provenance =
            PcFanCfdBackendProvenance::from_dispatch(config.cfd.backend, actual_backend);
        let execution_environment =
            PcFanCfdExecutionEnvironment::capture(&config.cfd, backend_provenance);
        if !backend_provenance.is_admitted(
            config.cfd.allow_host_fallback,
            config.cfd.require_device_resident,
        ) {
            return Err(PcFanCfdTransientError::BackendExecutionNotAdmitted {
                requested: backend_provenance.requested_backend,
                actual: backend_provenance.actual_backend,
                mode: backend_provenance.execution_mode,
            });
        }

        let residual_norm = transported_monolithic_step
            .as_ref()
            .map(|step| step.residual_norm)
            .unwrap_or_else(|| {
                state_residual
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt()
            });
        let coefficients =
            sonium_fem::physics::fluid::stokes::time_stepping::BdfCoefficients::for_order(
                bdf.order,
            );
        if let Some(initial_history) = transient_stage_initial_history_jacobian(
            &workspace,
            &state_jacobian,
            &bc_dofs,
            config.operating_point.air_density_kg_m3 / time_step_s,
            step,
        )
        .map_err(|_| PcFanCfdTransientError::InvalidLinearSystem { step })?
        {
            stage_initial_history_jacobians.push(initial_history);
        }
        let history_jacobians = transient_history_jacobians(
            &workspace,
            &state_jacobian,
            &bc_dofs,
            config.operating_point.air_density_kg_m3 / time_step_s,
            coefficients,
            step,
        )
        .map_err(|_| PcFanCfdTransientError::InvalidLinearSystem { step })?;
        let velocity_magnitude_max = state
            .velocity
            .as_chunks::<3>()
            .0
            .iter()
            .take(mesh.nodes.len())
            .map(|node_velocity| {
                node_velocity
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt()
            })
            .fold(0.0_f64, f64::max);
        let viscous_mu =
            ns_config
                .turbulence
                .effective_mu(ns_config.mu, ns_config.rho, velocity_magnitude_max);
        let mut residual_design_jacobian = assemble_body_force_design_residual_with_viscosity(
            &mesh,
            &velocity_space,
            &pressure_space,
            &ns_config,
            viscous_mu,
            &state.velocity,
            DESIGN_VARIABLE_COUNT,
            |coords, local_velocity| {
                solid_penalty
                    .force_design_jacobian_at(
                        Point3::new(coords[0], coords[1], coords[2]),
                        local_velocity,
                    )
                    .into_iter()
                    .map(|row| row.to_vec())
                    .collect()
            },
            2,
        )
        .map_err(|_| PcFanCfdTransientError::InvalidLinearSystem { step })?;
        let mesh_residual_design_jacobian = assemble_mesh_motion_design_residual(
            &mesh,
            &velocity_space,
            &pressure_space,
            &ns_config,
            &state.velocity,
            &state.pressure,
            &mesh_motion_design_jacobian,
            |coords, local_velocity| {
                solid_penalty.force_at(Point3::new(coords[0], coords[1], coords[2]), local_velocity)
            },
            |coords, local_velocity| {
                solid_penalty.force_position_jacobian_at(
                    Point3::new(coords[0], coords[1], coords[2]),
                    local_velocity,
                )
            },
        )
        .map_err(|_| PcFanCfdTransientError::InvalidLinearSystem { step })?;
        for (row, mesh_row) in residual_design_jacobian
            .iter_mut()
            .zip(mesh_residual_design_jacobian)
        {
            for (value, mesh_value) in row.iter_mut().zip(mesh_row) {
                *value += mesh_value;
            }
        }
        for &dof in &bc_dofs {
            residual_design_jacobian[dof].fill(0.0);
        }
        if pressure_dimension > 0 {
            let pressure_pin = pressure_pin_dof(&mesh, &pressure_space);
            residual_design_jacobian[velocity_dimension + pressure_pin].fill(0.0);
        }
        let transported_turbulence_residual_design_jacobian =
            if let (Some(history), Some(monolithic)) = (
                transported_turbulence_history.as_ref(),
                transported_monolithic_step.as_ref(),
            ) {
                let transport_config = KOmegaTransportConfig {
                    dt: bdf.dt,
                    bdf_order: bdf.order,
                    molecular_viscosity: config.operating_point.dynamic_viscosity_pa_s
                        / config.operating_point.air_density_kg_m3,
                    constants: KOmegaSstConstants::default(),
                    ..KOmegaTransportConfig::default()
                };
                let fields = derive_pc_fan_cfd_komega_element_fields(
                    &mesh,
                    &state.velocity,
                    &monolithic.turbulence_state,
                    10.0,
                )
                .map_err(|_| PcFanCfdTransientError::TransportedTurbulenceFailed { step })?;
                Some(
                    assemble_pc_fan_cfd_komega_scalar_mesh_motion_design_residual::<
                        DESIGN_VARIABLE_COUNT,
                    >(
                        &mesh,
                        &transport_config,
                        history,
                        &monolithic.turbulence_state,
                        &state.velocity,
                        &fields,
                        &mesh_motion_design_jacobian,
                    )
                    .map_err(|_| PcFanCfdTransientError::TransportedTurbulenceFailed { step })?,
                )
            } else {
                None
            };
        let velocity_m_s = velocity_vectors(&state.velocity, nodes.len());
        let inlet_airflow_cfm = boundary_airflow_cfm(
            &nodes,
            &mesh.elements,
            &velocity_m_s,
            domain.min_corner.z,
            domain.inlet_area_m2,
            boundary_tolerance,
        );
        let outlet_airflow_cfm = boundary_airflow_cfm(
            &nodes,
            &mesh.elements,
            &velocity_m_s,
            domain.max_corner.z,
            domain.outlet_area_m2,
            boundary_tolerance,
        );
        let pressure_drop_pa = average_boundary_pressure(
            &nodes,
            &state.pressure,
            domain.min_corner.z,
            boundary_tolerance,
        ) - average_boundary_pressure(
            &nodes,
            &state.pressure,
            domain.max_corner.z,
            boundary_tolerance,
        );
        let loading = integrate_rotating_wall_loading(&mesh, &velocity_m_s, &solid_penalty);
        let conservation = build_conservation_ledger(
            config,
            &domain,
            &nodes,
            &velocity_m_s,
            &state.pressure,
            inlet_airflow_cfm,
            outlet_airflow_cfm,
            None,
            Some(loading),
            boundary_tolerance,
        );
        trajectory_steps.push(PcFanCfdTransientTrajectoryStep {
            step,
            time_s: initial_time_s + (step + 1) as f64 * time_step_s,
            state: state_vector,
            state_jacobian,
            state_jacobian_provenance,
            actual_backend,
            backend_provenance,
            execution_environment,
            rhs,
            state_residual,
            residual_norm,
            conservation,
            history_jacobians,
            transported_turbulence: transported_turbulence_state.clone(),
            transported_eddy_viscosity_per_element: applied_transported_eddy_viscosity_per_element,
            transported_turbulence_linearization,
            transported_turbulence_coupled_linearization,
            transported_turbulence_residual,
            transported_turbulence_residual_design_jacobian,
            bdf_state: PcFanCfdTransientBdfState::from_fem(&state),
            topology_transfer_adjoint_admissible: !externally_remeshed,
            residual_design_jacobian,
        });
        let residual = transported_monolithic_step.as_ref().map_or_else(
            || {
                workspace
                    .last_linear_solve_telemetry()
                    .map_or(f64::INFINITY, |telemetry| telemetry.true_relative_residual)
            },
            |monolithic| monolithic.residual_norm,
        );
        let time_s = state.time;
        observations.push((
            time_s,
            outlet_airflow_cfm.abs(),
            pressure_drop_pa,
            loading.force_on_fluid_n[2],
            residual,
            conservation,
            transient_step_is_admitted(
                solved,
                residual,
                nonlinear_tolerance,
                inlet_airflow_cfm.is_finite()
                    && outlet_airflow_cfm.is_finite()
                    && pressure_drop_pa.is_finite()
                    && loading
                        .force_on_fluid_n
                        .iter()
                        .all(|value| value.is_finite())
                    && state.velocity.iter().all(|value| value.is_finite())
                    && state.pressure.iter().all(|value| value.is_finite()),
                mesh_level_conservation_admitted(config, conservation),
            ),
        ));
    }
    let window_start = (duration_s - std::f64::consts::TAU / angular_speed.abs()).max(0.0);
    let window = observations
        .iter()
        .filter(|observation| observation.0 >= window_start)
        .collect::<Vec<_>>();
    let window = if window.is_empty() {
        observations.iter().collect::<Vec<_>>()
    } else {
        window
    };
    let count = window.len() as f64;
    let average = |selector: fn(&ConservationObservation) -> f64| {
        window
            .iter()
            .map(|observation| selector(observation))
            .sum::<f64>()
            / count
    };
    let average_conservation = |selector: fn(PcFanCfdConservation) -> f64| {
        window
            .iter()
            .map(|observation| selector(observation.5))
            .sum::<f64>()
            / count
    };
    let conservation = PcFanCfdConservation {
        inlet_mass_flow_kg_s: average_conservation(|value| value.inlet_mass_flow_kg_s),
        outlet_mass_flow_kg_s: average_conservation(|value| value.outlet_mass_flow_kg_s),
        relative_mass_imbalance: average_conservation(|value| value.relative_mass_imbalance),
        inlet_axial_momentum_flux_n: average_conservation(|value| {
            value.inlet_axial_momentum_flux_n
        }),
        outlet_axial_momentum_flux_n: average_conservation(|value| {
            value.outlet_axial_momentum_flux_n
        }),
        pressure_force_n: average_conservation(|value| value.pressure_force_n),
        body_force_n: average_conservation(|value| value.body_force_n),
        axial_momentum_residual_n: average_conservation(|value| value.axial_momentum_residual_n),
        relative_momentum_imbalance: average_conservation(|value| {
            value.relative_momentum_imbalance
        }),
        pressure_force_from_loading_n: average_conservation(|value| {
            value.pressure_force_from_loading_n
        }),
        relative_pressure_closure: average_conservation(|value| value.relative_pressure_closure),
        pressure_power_w: average_conservation(|value| value.pressure_power_w),
        shaft_power_to_fluid_w: average_conservation(|value| value.shaft_power_to_fluid_w),
        relative_energy_imbalance: average_conservation(|value| value.relative_energy_imbalance),
    };
    let converged = window.iter().all(|observation| observation.6);
    let result = PcFanCfdTransientObservation {
        final_time_s: observations.last().map_or(0.0, |observation| observation.0),
        airflow_cfm: average(|observation| observation.1),
        pressure_drop_pa: average(|observation| observation.2),
        axial_loading_n: average(|observation| observation.3),
        residual: window
            .iter()
            .map(|observation| observation.4)
            .fold(0.0, f64::max),
        conservation,
        converged,
    };
    if !result.final_time_s.is_finite()
        || !result.airflow_cfm.is_finite()
        || !result.pressure_drop_pa.is_finite()
        || !result.axial_loading_n.is_finite()
        || !result.residual.is_finite()
    {
        return Err(PcFanCfdTransientError::NonFiniteObservation);
    }
    if let Some((metric, value, tolerance)) = first_conservation_violation(config, conservation) {
        if !value.is_finite() {
            return Err(PcFanCfdTransientError::NonFiniteObservation);
        }
        return Err(PcFanCfdTransientError::ConservationViolation {
            metric,
            value,
            tolerance,
        });
    }
    if !result.converged {
        if let Some(observation) = window.iter().find(|observation| !observation.6)
            && let Some((metric, value, tolerance)) =
                first_conservation_violation(config, observation.5)
        {
            if !value.is_finite() {
                return Err(PcFanCfdTransientError::NonFiniteObservation);
            }
            return Err(PcFanCfdTransientError::ConservationViolation {
                metric,
                value,
                tolerance,
            });
        }
        return Err(PcFanCfdTransientError::NonConvergedObservation {
            residual: result.residual,
            tolerance: nonlinear_tolerance,
        });
    }
    Ok(PcFanCfdTransientTrajectory {
        observation: result,
        steps: trajectory_steps,
        stage_initial_history_jacobians,
    })
}

/// Run the bounded coarse/fine BDF time-step independence check.
///
/// The checked entry point validates the time grid before launching either
/// transient solve. The historical infallible wrapper is provided for
/// callers that already treat configuration errors as programmer errors.
pub fn solve_pc_fan_cfd_transient_convergence(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    convergence: PcFanCfdTransientConvergenceConfig,
) -> PcFanCfdTransientConvergence {
    try_solve_pc_fan_cfd_transient_convergence(config, design, convergence).unwrap_or_else(
        |error| panic!("invalid PC-fan transient convergence configuration: {error}"),
    )
}

/// Checked two-level time-step independence sweep for the same rotor interval.
pub fn try_solve_pc_fan_cfd_transient_convergence(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    convergence: PcFanCfdTransientConvergenceConfig,
) -> Result<PcFanCfdTransientConvergence, PcFanCfdTransientConvergenceConfigError> {
    validate_transient_convergence_config(config, convergence)?;

    let levels = [convergence.coarse_time_step_s, convergence.fine_time_step_s]
        .into_iter()
        .map(|time_step_s| {
            let observation = solve_pc_fan_cfd_rotating_wall_transient(
                config,
                design,
                time_step_s,
                convergence.rotor_revolutions,
            )
            .map_err(PcFanCfdTransientConvergenceConfigError::SolveFailed)?;
            Ok(PcFanCfdTransientConvergenceLevel {
                time_step_s,
                observation,
            })
        })
        .collect::<Result<Vec<_>, PcFanCfdTransientConvergenceConfigError>>()?;

    let coarse = levels[0].observation;
    let fine = levels[1].observation;
    let airflow_relative_change = Some(relative_change(fine.airflow_cfm, coarse.airflow_cfm));
    let pressure_relative_change = Some(relative_change(
        fine.pressure_drop_pa,
        coarse.pressure_drop_pa,
    ));
    let axial_loading_relative_change = Some(relative_change(
        fine.axial_loading_n,
        coarse.axial_loading_n,
    ));
    let final_time_relative_change = Some(relative_change(fine.final_time_s, coarse.final_time_s));
    let all_levels_converged_and_finite = levels.iter().all(|level| {
        transient_observation_is_admitted(config, level.observation, convergence.max_final_residual)
    });
    let converged = all_levels_converged_and_finite
        && [
            airflow_relative_change,
            pressure_relative_change,
            axial_loading_relative_change,
            final_time_relative_change,
        ]
        .into_iter()
        .flatten()
        .all(|change| change <= convergence.relative_tolerance);

    Ok(PcFanCfdTransientConvergence {
        levels,
        airflow_relative_change,
        pressure_relative_change,
        axial_loading_relative_change,
        final_time_relative_change,
        converged,
        all_levels_converged_and_finite,
    })
}

fn validate_transient_convergence_config(
    config: &PcFanConfig,
    convergence: PcFanCfdTransientConvergenceConfig,
) -> Result<(), PcFanCfdTransientConvergenceConfigError> {
    if !convergence.coarse_time_step_s.is_finite() || convergence.coarse_time_step_s <= 0.0 {
        return Err(
            PcFanCfdTransientConvergenceConfigError::InvalidCoarseTimeStep(
                convergence.coarse_time_step_s,
            ),
        );
    }
    if !convergence.fine_time_step_s.is_finite() || convergence.fine_time_step_s <= 0.0 {
        return Err(
            PcFanCfdTransientConvergenceConfigError::InvalidFineTimeStep(
                convergence.fine_time_step_s,
            ),
        );
    }
    if convergence.fine_time_step_s >= convergence.coarse_time_step_s {
        return Err(
            PcFanCfdTransientConvergenceConfigError::FineTimeStepNotSmaller {
                coarse: convergence.coarse_time_step_s,
                fine: convergence.fine_time_step_s,
            },
        );
    }
    if !convergence.rotor_revolutions.is_finite() || convergence.rotor_revolutions <= 0.0 {
        return Err(
            PcFanCfdTransientConvergenceConfigError::InvalidRotorRevolutions(
                convergence.rotor_revolutions,
            ),
        );
    }
    if !convergence.relative_tolerance.is_finite() || convergence.relative_tolerance < 0.0 {
        return Err(
            PcFanCfdTransientConvergenceConfigError::InvalidRelativeTolerance(
                convergence.relative_tolerance,
            ),
        );
    }
    if !convergence.max_final_residual.is_finite() || convergence.max_final_residual < 0.0 {
        return Err(
            PcFanCfdTransientConvergenceConfigError::InvalidFinalResidual(
                convergence.max_final_residual,
            ),
        );
    }

    let angular_speed = std::f64::consts::TAU * config.operating_point.rpm / 60.0;
    if !angular_speed.is_finite() || angular_speed.abs() <= f64::EPSILON {
        return Err(
            PcFanCfdTransientConvergenceConfigError::InvalidRotorRevolutions(
                config.operating_point.rpm,
            ),
        );
    }
    let duration_s = convergence.rotor_revolutions * std::f64::consts::TAU / angular_speed.abs();
    for time_step_s in [convergence.coarse_time_step_s, convergence.fine_time_step_s] {
        let step_count = duration_s / time_step_s;
        let rounded_step_count = step_count.round();
        let grid_error = (step_count - rounded_step_count).abs();
        if !step_count.is_finite() || grid_error > 1.0e-9 * step_count.abs().max(1.0) {
            return Err(
                PcFanCfdTransientConvergenceConfigError::NonIntegralTimeGrid {
                    time_step_s,
                    step_count,
                },
            );
        }
        if rounded_step_count < 2.0 {
            return Err(PcFanCfdTransientConvergenceConfigError::TooFewTimeSteps {
                time_step_s,
                step_count,
            });
        }
    }
    Ok(())
}

fn transient_observation_is_admitted(
    config: &PcFanConfig,
    observation: PcFanCfdTransientObservation,
    max_final_residual: f64,
) -> bool {
    observation.converged
        && observation.final_time_s.is_finite()
        && observation.airflow_cfm.is_finite()
        && observation.pressure_drop_pa.is_finite()
        && observation.axial_loading_n.is_finite()
        && observation.residual.is_finite()
        && observation.residual <= max_final_residual
        && mesh_level_conservation_admitted(config, observation.conservation)
}

/// Evaluate exactly one production FEM Picard update at a prescribed CFD
/// state. This is the state-map seam used by the checked derivative probe;
/// it is intentionally separate from the converged-flow API so callers cannot
/// mistake an Oseen observation for an exact Newton tangent.
pub fn evaluate_pc_fan_cfd_picard_map(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    initial_state: &[f64],
) -> Result<PcFanResolvedFlowPicardLinearization, PcFanResolvedAdjointError> {
    if !config.cfd.flow_model.is_cfd() {
        return Err(PcFanResolvedAdjointError::UnsupportedFlowModel(
            config.cfd.flow_model,
        ));
    }
    config
        .validate_geometry_config()
        .map_err(|_| PcFanResolvedAdjointError::ProbeEvaluationFailed("CFD geometry"))?;
    if initial_state.iter().any(|value| !value.is_finite()) {
        return Err(PcFanResolvedAdjointError::NonFiniteInput(
            "production Picard initial state",
        ));
    }
    let geometry = config.build_geometry(design);
    let domain = geometry.build_cfd_domain(config);
    // The production map uses P1 velocity and pressure spaces over the same
    // mesh. The tetrahedral generator can add nodes beyond the structured
    // domain-node list, so derive dimensions from the actual spaces rather
    // than assuming four DOFs per domain node.
    let nx = config.cfd.cells_x.max(1);
    let ny = config.cfd.cells_y.max(1);
    let nz = config.cfd.cells_z.max(1);
    let mut mesh = match config.cfd.mesh_pattern {
        PcFanCfdMeshPattern::Symmetric => box_mesh_tetrahedra_symmetric(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            nx,
            ny,
            nz,
        ),
        PcFanCfdMeshPattern::Kuhn => box_mesh_tetrahedra(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            nx,
            ny,
            nz,
        ),
    };
    apply_pc_fan_mesh_motion(&mut mesh, config, &geometry, design)
        .map_err(|_| PcFanResolvedAdjointError::ProbeEvaluationFailed("CFD mesh motion"))?;
    let velocity_space = VectorH1Space::new(PolynomialDegree::P1, 3);
    let pressure_space = H1Space::new(PolynomialDegree::P1);
    let velocity_dimension = velocity_space.num_global_dofs(&mesh);
    let pressure_dimension = pressure_space.num_global_dofs(&mesh);
    if initial_state.len() != velocity_dimension + pressure_dimension {
        return Err(PcFanResolvedAdjointError::DimensionMismatch(
            "production Picard initial state",
        ));
    }
    let solution = solve_cfd_domain_with_derivative_policy(
        config,
        design,
        &geometry,
        &domain,
        Some((
            &initial_state[..velocity_dimension],
            &initial_state[velocity_dimension..],
        )),
        true,
        SteadySolvePurpose::PicardMapProbe,
    );
    solution
        .picard_linearization
        .ok_or(PcFanResolvedAdjointError::MissingProductionPicardLinearization)
}

/// Evaluate the analytic Navier--Stokes tangent at one prescribed PC-fan CFD
/// state. The returned observation is state-only evidence: design residual
/// and objective derivatives are intentionally not fabricated here.
pub fn evaluate_pc_fan_cfd_newton_linearization(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    initial_state: &[f64],
) -> Result<PcFanResolvedFlowNewtonLinearization, PcFanResolvedAdjointError> {
    evaluate_pc_fan_cfd_newton_linearization_with_policy(config, design, initial_state, true)
}

/// Evaluate the native flow residual and tangent without design derivatives.
///
/// Uses the same constrained Navier--Stokes assembly as
/// [`evaluate_pc_fan_cfd_newton_linearization`], at the supplied state. Mesh
/// motion is still applied, but its design Jacobian and the residual's design
/// Jacobian are omitted. The returned `residual_design_jacobian` is `None`.
/// This path assembles the equations without performing a Picard linear solve.
/// Assembly runs on the CPU and dispatches no linear solver backend.
/// It is suitable for primal residual checks and nonlinear iteration.
///
/// # Errors
///
/// Returns an error for a non-CFD model, invalid geometry, a non-finite or
/// incorrectly sized state, or an unavailable production Newton tangent.
pub fn evaluate_pc_fan_cfd_newton_state_linearization(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    initial_state: &[f64],
) -> Result<PcFanResolvedFlowNewtonLinearization, PcFanResolvedAdjointError> {
    evaluate_pc_fan_cfd_newton_linearization_with_policy(config, design, initial_state, false)
}


fn evaluate_pc_fan_cfd_newton_linearization_with_policy(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    initial_state: &[f64],
    publish_design_derivatives: bool,
) -> Result<PcFanResolvedFlowNewtonLinearization, PcFanResolvedAdjointError> {
    if !config.cfd.flow_model.is_cfd() {
        return Err(PcFanResolvedAdjointError::UnsupportedFlowModel(
            config.cfd.flow_model,
        ));
    }
    config
        .validate_geometry_config()
        .map_err(|_| PcFanResolvedAdjointError::ProbeEvaluationFailed("CFD geometry"))?;
    if initial_state.iter().any(|value| !value.is_finite()) {
        return Err(PcFanResolvedAdjointError::NonFiniteInput(
            "production Newton initial state",
        ));
    }

    let geometry = config.build_geometry(design);
    let domain = geometry.build_cfd_domain(config);
    let nx = config.cfd.cells_x.max(1);
    let ny = config.cfd.cells_y.max(1);
    let nz = config.cfd.cells_z.max(1);
    let mesh = match config.cfd.mesh_pattern {
        PcFanCfdMeshPattern::Symmetric => box_mesh_tetrahedra_symmetric(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            nx,
            ny,
            nz,
        ),
        PcFanCfdMeshPattern::Kuhn => box_mesh_tetrahedra(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            nx,
            ny,
            nz,
        ),
    };
    let velocity_space = VectorH1Space::new(PolynomialDegree::P1, 3);
    let pressure_space = H1Space::new(PolynomialDegree::P1);
    let velocity_dimension = velocity_space.num_global_dofs(&mesh);
    let pressure_dimension = pressure_space.num_global_dofs(&mesh);
    if initial_state.len() != velocity_dimension + pressure_dimension {
        return Err(PcFanResolvedAdjointError::DimensionMismatch(
            "production Newton initial state",
        ));
    }

    let solution = solve_cfd_domain_with_derivative_policy(
        config,
        design,
        &geometry,
        &domain,
        Some((
            &initial_state[..velocity_dimension],
            &initial_state[velocity_dimension..],
        )),
        publish_design_derivatives,
        if publish_design_derivatives {
            SteadySolvePurpose::PicardMapProbe
        } else {
            SteadySolvePurpose::NewtonStateProbe
        },
    );
    solution
        .newton_linearization
        .ok_or(PcFanResolvedAdjointError::MissingProductionNewtonLinearization)
}

/// Solve the exact resolved-flow adjoint from a production FEM Newton
/// observation and caller-supplied analytic design/objective derivatives.
///
/// The production state tangent is assembled by the PC-fan CFD path; the
/// derivative bundle must be assembled by the caller from the same converged
/// state and is rejected when it carries finite-difference, probe, Picard,
/// Oseen, or black-box provenance. This explicit split prevents a state-only
/// tangent from being mistaken for a complete design adjoint.
pub fn solve_pc_fan_cfd_resolved_flow_adjoint(
    adjoint_config: &PcFanResolvedAdjointConfig,
    pc_fan_config: &PcFanConfig,
    design: &PcFanDesignVariables,
    initial_state: &[f64],
    derivatives: PcFanResolvedFlowDerivativeData,
) -> Result<PcFanResolvedAdjointReport, PcFanResolvedAdjointError> {
    let newton = evaluate_pc_fan_cfd_newton_linearization(pc_fan_config, design, initial_state)?;
    solve_pc_fan_resolved_flow_adjoint_from_newton(adjoint_config, &newton, derivatives)
}

/// Solve the exact resolved-flow adjoint using the production frozen-mesh
/// design residual derivative and explicit caller-supplied objective
/// derivatives. This convenience path is available only for the actuator-disk
/// configuration where all active design terms are assembled analytically.
pub fn solve_pc_fan_cfd_resolved_flow_adjoint_with_production_design_residual(
    adjoint_config: &PcFanResolvedAdjointConfig,
    pc_fan_config: &PcFanConfig,
    design: &PcFanDesignVariables,
    initial_state: &[f64],
    objective_provenance: impl Into<String>,
    objective_state_gradient: Vec<f64>,
    objective_design_gradient: Vec<f64>,
) -> Result<PcFanResolvedAdjointReport, PcFanResolvedAdjointError> {
    let newton = evaluate_pc_fan_cfd_newton_linearization(pc_fan_config, design, initial_state)?;
    let derivatives = PcFanResolvedFlowDerivativeData::from_production_newton(
        &newton,
        objective_provenance,
        objective_state_gradient,
        objective_design_gradient,
    )?;
    solve_pc_fan_resolved_flow_adjoint_from_newton(adjoint_config, &newton, derivatives)
}

/// A production resolved-flow objective whose complete topology-preserving ALE
/// derivative
/// is available from the actuator-disk assembly.
///
/// These objectives are deliberately limited to source loading observables.
/// They do not silently include acoustic, constraint, shroud-shape, or
/// rotating-wall terms using the same topology-preserving ALE contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcFanResolvedFlowActuatorObjective {
    /// Axial force transferred from the actuator disk to the fluid.
    AxialLoading,
    /// Mechanical power transferred from the actuator disk to the fluid.
    ShaftPower,
}

impl PcFanResolvedFlowActuatorObjective {
    fn provenance(self) -> &'static str {
        match self {
            Self::AxialLoading => "analytic-actuator-axial-loading",
            Self::ShaftPower => "analytic-actuator-shaft-power",
        }
    }
}

/// A resolved-flow objective whose analytic derivative is available from
/// the production PC-fan state and actuator/boundary assembly.
///
/// The `Actuator` variants use the volume-integrated loading ledger.  The
/// boundary observables use the same inlet/outlet quadrature as the published
/// CFD solution. Rotating-wall direct design terms include topology-
/// preserving ALE mesh motion; boundary terms retain their quadrature
/// contract and obtain design dependence through the residual Jacobian.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcFanResolvedFlowObjective {
    Actuator(PcFanResolvedFlowActuatorObjective),
    /// Absolute outlet volumetric flow, reported in CFM.
    Airflow,
    /// Outlet mean pressure minus inlet mean pressure, reported in pascals.
    PressureRise,
    /// Axial loading recovered from the immersed rotating-wall support.
    ///
    /// This objective uses the topology-preserving ALE immersed-wall
    /// derivative contract. Blade-shape hub/tip/shroud occupancy and
    /// mesh-motion terms are differentiated analytically.
    RotatingWallAxialLoading,
    /// Mechanical shaft power recovered from the immersed rotating-wall
    /// support under the same topology-preserving ALE contract.
    RotatingWallShaftPower,
}

/// A steady resolved-flow state objective with analytic rotating-wall
/// design derivatives, including topology-preserving ALE mesh motion.
///
/// This contract is intentionally separate from [`PcFanResolvedFlowObjective`]
/// so a state-only multiplier cannot be mistaken for a complete topology-
/// changing design adjoint. Complete design adjoints use the matching
/// production residual derivative supplied by the CFD linearization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcFanResolvedFlowStateObjective {
    /// Absolute outlet volumetric flow, reported in CFM.
    Airflow,
    /// Outlet mean pressure minus inlet mean pressure, reported in pascals.
    PressureRise,
    /// Recovered axial force transferred from rotating walls to the fluid.
    RotatingWallAxialLoading,
    /// Recovered rotating-wall shaft power transferred to the fluid.
    RotatingWallShaftPower,
}

impl PcFanResolvedFlowStateObjective {
    fn provenance(self) -> &'static str {
        match self {
            Self::Airflow => "analytic-steady-airflow-state",
            Self::PressureRise => "analytic-steady-pressure-rise-state",
            Self::RotatingWallAxialLoading => "analytic-steady-rotating-wall-axial-state",
            Self::RotatingWallShaftPower => "analytic-steady-rotating-wall-power-state",
        }
    }
}

#[cfg(test)]
mod actuator_objective_tests {
    use super::*;

    fn fixture() -> (
        PcFanConfig,
        PcFanDesignVariables,
        PcFanResolvedFlowNewtonLinearization,
        sonium_fem::mesh::Mesh,
    ) {
        let mut config = PcFanConfig::default_120mm();
        config.cfd.flow_model = PcFanFlowModel::IncompressibleCfd;
        config.cfd.backend = PcFanCfdBackend::Cpu;
        config.cfd.cells_x = 2;
        config.cfd.cells_y = 2;
        config.cfd.cells_z = 2;
        config.cfd.max_picard_iterations = 1;
        let design = config.default_design_variables();
        let seed = solve_pc_fan_cfd_mean_flow(&config, &design);
        let initial_state = seed
            .picard_linearization
            .expect("fixture solve should publish a state")
            .state;
        let tangent = evaluate_pc_fan_cfd_newton_linearization(&config, &design, &initial_state)
            .expect("fixture should publish a Newton tangent");
        let geometry = config.build_geometry(&design);
        let domain = geometry.build_cfd_domain(&config);
        let mesh = box_mesh_tetrahedra_symmetric(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            2,
            2,
            2,
        );
        (config, design, tangent, mesh)
    }

    fn objective_value(
        config: &PcFanConfig,
        design: &PcFanDesignVariables,
        mesh: &sonium_fem::mesh::Mesh,
        state: &[f64],
        objective: PcFanResolvedFlowActuatorObjective,
    ) -> f64 {
        let geometry = config.build_geometry(design);
        let forcing = BladeActuatorForcing::from_geometry(config, &geometry, design);
        let velocity = velocity_vectors(state, mesh.num_nodes());
        let loading = integrate_blade_actuator_loading(mesh, &velocity, &forcing);
        match objective {
            PcFanResolvedFlowActuatorObjective::AxialLoading => loading.force_on_fluid_n[2],
            PcFanResolvedFlowActuatorObjective::ShaftPower => loading.shaft_power_to_fluid_w,
        }
    }

    fn boundary_objective_value(
        config: &PcFanConfig,
        design: &PcFanDesignVariables,
        mesh: &sonium_fem::mesh::Mesh,
        state: &[f64],
        objective: PcFanResolvedFlowObjective,
    ) -> f64 {
        let geometry = config.build_geometry(design);
        let domain = geometry.build_cfd_domain(config);
        let nodes = mesh
            .nodes
            .iter()
            .map(|node| Point3::new(node.x, node.y, node.z))
            .collect::<Vec<_>>();
        let velocity_dimension = mesh.num_nodes() * 3;
        let velocity = velocity_vectors(&state[..velocity_dimension], nodes.len());
        let tolerance = boundary_tolerance(
            &domain,
            config.cfd.cells_x.max(1),
            config.cfd.cells_y.max(1),
            config.cfd.cells_z.max(1),
        );
        match objective {
            PcFanResolvedFlowObjective::Airflow => boundary_airflow_cfm(
                &nodes,
                &mesh.elements,
                &velocity,
                domain.max_corner.z,
                domain.outlet_area_m2,
                tolerance,
            )
            .abs(),
            PcFanResolvedFlowObjective::PressureRise => {
                average_boundary_pressure(
                    &nodes,
                    &state[velocity_dimension..],
                    domain.max_corner.z,
                    tolerance,
                ) - average_boundary_pressure(
                    &nodes,
                    &state[velocity_dimension..],
                    domain.min_corner.z,
                    tolerance,
                )
            }
            PcFanResolvedFlowObjective::Actuator(_) => {
                panic!("boundary objective helper received actuator objective")
            }
            PcFanResolvedFlowObjective::RotatingWallAxialLoading
            | PcFanResolvedFlowObjective::RotatingWallShaftPower => {
                panic!("boundary objective helper received rotating-wall objective")
            }
        }
    }

    #[test]
    fn actuator_objective_derivatives_match_state_and_design_centered_differences() {
        let (config, design, tangent, mesh) = fixture();
        for objective in [
            PcFanResolvedFlowActuatorObjective::AxialLoading,
            PcFanResolvedFlowActuatorObjective::ShaftPower,
        ] {
            let derivatives = build_pc_fan_cfd_actuator_objective_derivatives(
                &config, &design, &tangent, objective,
            )
            .expect("actuator objective derivative should be admitted");
            assert!(
                derivatives
                    .objective_state_gradient
                    .iter()
                    .all(|value| value.is_finite())
            );
            assert!(
                derivatives
                    .objective_design_gradient
                    .iter()
                    .all(|value| value.is_finite())
            );

            let state_indices = [0usize, 3, 6, tangent.state.len() / 3];
            for &index in &state_indices {
                if index >= mesh.num_nodes() * 3 {
                    continue;
                }
                let step = 1.0e-6;
                let mut plus = tangent.state.clone();
                let mut minus = tangent.state.clone();
                plus[index] += step;
                minus[index] -= step;
                let centered = (objective_value(&config, &design, &mesh, &plus, objective)
                    - objective_value(&config, &design, &mesh, &minus, objective))
                    / (2.0 * step);
                let analytic = derivatives.objective_state_gradient[index];
                let scale = centered.abs().max(analytic.abs()).max(1.0e-12);
                assert!((centered - analytic).abs() / scale < 2.0e-3);
            }

            let base = design.to_vector();
            for index in 0..DESIGN_VARIABLE_COUNT {
                let step = 1.0e-7 * base[index].abs().max(1.0);
                let mut plus = base;
                let mut minus = base;
                plus[index] += step;
                minus[index] -= step;
                let plus_design = PcFanDesignVariables::from_vector(plus);
                let minus_design = PcFanDesignVariables::from_vector(minus);
                let centered =
                    (objective_value(&config, &plus_design, &mesh, &tangent.state, objective)
                        - objective_value(
                            &config,
                            &minus_design,
                            &mesh,
                            &tangent.state,
                            objective,
                        ))
                        / (2.0 * step);
                let analytic = derivatives.objective_design_gradient[index];
                let scale = centered.abs().max(analytic.abs()).max(1.0e-10);
                assert!(
                    (centered - analytic).abs() / scale < 5.0e-3,
                    "objective {objective:?}, design index {index}: centered={centered:.6e}, analytic={analytic:.6e}"
                );
            }
        }
    }

    #[test]
    fn actuator_objective_adjoint_rejects_rotating_wall_and_accepts_fixed_mesh_shroud() {
        let (mut config, design, tangent, _) = fixture();
        let mut wall_config = config.clone();
        wall_config.cfd.flow_model = PcFanFlowModel::IncompressibleCfdFull;
        assert!(matches!(
            build_pc_fan_cfd_actuator_objective_derivatives(
                &wall_config,
                &design,
                &tangent,
                PcFanResolvedFlowActuatorObjective::ShaftPower,
            ),
            Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible)
        ));

        config.cfd.flow_model = PcFanFlowModel::IncompressibleCfd;
        let mut moved_design = design;
        moved_design.shroud_radius_offsets_m[0] = 1.0e-4;
        let derivatives = build_pc_fan_cfd_actuator_objective_derivatives(
            &config,
            &moved_design,
            &tangent,
            PcFanResolvedFlowActuatorObjective::AxialLoading,
        )
        .expect("fixed-mesh shroud occupancy derivatives should be admitted");
        assert_eq!(
            derivatives.objective_design_gradient.len(),
            DESIGN_VARIABLE_COUNT
        );
        assert!(
            derivatives
                .objective_design_gradient
                .iter()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn boundary_objective_derivatives_match_state_and_design_centered_differences() {
        let (config, design, tangent, mesh) = fixture();
        let velocity_dimension = mesh.num_nodes() * 3;
        let state_indices = [0usize, 3, velocity_dimension, velocity_dimension + 1];

        for objective in [
            PcFanResolvedFlowObjective::Airflow,
            PcFanResolvedFlowObjective::PressureRise,
        ] {
            let mut source = tangent.clone();
            if matches!(objective, PcFanResolvedFlowObjective::Airflow) {
                let geometry = config.build_geometry(&design);
                let domain = geometry.build_cfd_domain(&config);
                let tolerance = boundary_tolerance(
                    &domain,
                    config.cfd.cells_x.max(1),
                    config.cfd.cells_y.max(1),
                    config.cfd.cells_z.max(1),
                );
                for (node, point) in mesh.nodes.iter().enumerate() {
                    if (point.z - domain.max_corner.z).abs() <= tolerance {
                        source.state[3 * node + 2] += 1.0;
                    }
                }
            }
            let derivatives = build_pc_fan_cfd_resolved_objective_derivatives(
                &config, &design, &source, objective,
            )
            .expect("boundary objective derivative should be admitted");

            for &index in &state_indices {
                let step = 1.0e-6;
                let mut plus = source.state.clone();
                let mut minus = source.state.clone();
                plus[index] += step;
                minus[index] -= step;
                let centered =
                    (boundary_objective_value(&config, &design, &mesh, &plus, objective)
                        - boundary_objective_value(&config, &design, &mesh, &minus, objective))
                        / (2.0 * step);
                let analytic = derivatives.objective_state_gradient[index];
                let scale = centered.abs().max(analytic.abs()).max(1.0e-10);
                assert!(
                    (centered - analytic).abs() / scale < 2.0e-6,
                    "objective {objective:?}, state index {index}: centered={centered:.6e}, analytic={analytic:.6e}"
                );
            }

            assert!(
                derivatives
                    .objective_design_gradient
                    .iter()
                    .all(|value| value.abs() < 1.0e-14)
            );
        }
    }

    #[test]
    fn boundary_objective_admits_fixed_mesh_shroud_controls() {
        let (config, mut design, tangent, _) = fixture();
        design.shroud_radius_offsets_m[2] = 1.0e-4;

        let derivatives = build_pc_fan_cfd_resolved_objective_derivatives(
            &config,
            &design,
            &tangent,
            PcFanResolvedFlowObjective::PressureRise,
        )
        .expect("fixed-mesh shroud boundary derivatives should be admitted");
        assert_eq!(
            derivatives.objective_design_gradient.len(),
            DESIGN_VARIABLE_COUNT
        );
        assert!(
            derivatives
                .objective_design_gradient
                .iter()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn resolved_boundary_objective_adjoint_rejects_nonconverged_map_state() {
        let (config, design, tangent, _) = fixture();
        let error = solve_pc_fan_cfd_resolved_objective_adjoint(
            &PcFanResolvedAdjointConfig::default(),
            &config,
            &design,
            &tangent.state,
            PcFanResolvedFlowObjective::PressureRise,
        )
        .expect_err("a one-step map state must not enter the exact adjoint");
        assert!(matches!(
            error,
            PcFanResolvedAdjointError::StateResidualTooLarge { .. }
        ));

        let solution = solve_pc_fan_cfd_mean_flow(&config, &design);
        assert!(!solution.converged);
        assert!(matches!(
            solve_pc_fan_cfd_resolved_objective_adjoint_from_solution(
                &PcFanResolvedAdjointConfig::default(),
                &config,
                &design,
                &solution,
                PcFanResolvedFlowObjective::Airflow,
            ),
            Err(PcFanResolvedAdjointError::ProductionStateNotConverged)
        ));
    }
}

#[cfg(test)]
mod resolved_state_objective_tests {
    use super::*;

    fn source_fixture() -> (
        PcFanConfig,
        PcFanDesignVariables,
        PcFanResolvedFlowNewtonLinearization,
    ) {
        let mut config = PcFanConfig::default_120mm();
        config.cfd.flow_model = PcFanFlowModel::IncompressibleCfd;
        config.cfd.backend = PcFanCfdBackend::Cpu;
        config.cfd.cells_x = 2;
        config.cfd.cells_y = 2;
        config.cfd.cells_z = 2;
        config.cfd.max_picard_iterations = 1;
        let design = config.default_design_variables();
        let seed = solve_pc_fan_cfd_mean_flow(&config, &design);
        let state = seed
            .picard_linearization
            .expect("fixture solve should publish a state")
            .state;
        let mut source = evaluate_pc_fan_cfd_newton_linearization(&config, &design, &state)
            .expect("fixture should publish a Newton tangent");
        source.state_residual.fill(0.0);
        (config, design, source)
    }

    fn full_source_fixture() -> (
        PcFanConfig,
        PcFanDesignVariables,
        PcFanResolvedFlowNewtonLinearization,
    ) {
        let mut config = PcFanConfig::default_120mm();
        config.cfd.flow_model = PcFanFlowModel::IncompressibleCfdFull;
        config.cfd.backend = PcFanCfdBackend::Cpu;
        config.cfd.cells_x = 2;
        config.cfd.cells_y = 2;
        config.cfd.cells_z = 2;
        config.cfd.max_picard_iterations = 1;
        let design = config.default_design_variables();
        let seed = solve_pc_fan_cfd_mean_flow(&config, &design);
        let state = seed
            .picard_linearization
            .expect("full-CFD fixture solve should publish a state")
            .state;
        let mut source = evaluate_pc_fan_cfd_newton_linearization(&config, &design, &state)
            .expect("full-CFD fixture should publish a Newton tangent");
        source.state_residual.fill(0.0);
        (config, design, source)
    }

    fn fixed_state_rotating_wall_objective(
        config: &PcFanConfig,
        design: &PcFanDesignVariables,
        state: &[f64],
        objective: PcFanResolvedFlowStateObjective,
    ) -> f64 {
        let geometry = config.build_geometry(design);
        let domain = geometry.build_cfd_domain(config);
        let nx = config.cfd.cells_x.max(1);
        let ny = config.cfd.cells_y.max(1);
        let nz = config.cfd.cells_z.max(1);
        let mut mesh = match config.cfd.mesh_pattern {
            PcFanCfdMeshPattern::Symmetric => box_mesh_tetrahedra_symmetric(
                domain.min_corner.x,
                domain.max_corner.x,
                domain.min_corner.y,
                domain.max_corner.y,
                domain.min_corner.z,
                domain.max_corner.z,
                nx,
                ny,
                nz,
            ),
            PcFanCfdMeshPattern::Kuhn => box_mesh_tetrahedra(
                domain.min_corner.x,
                domain.max_corner.x,
                domain.min_corner.y,
                domain.max_corner.y,
                domain.min_corner.z,
                domain.max_corner.z,
                nx,
                ny,
                nz,
            ),
        };
        apply_pc_fan_mesh_motion(&mut mesh, config, &geometry, design)
            .expect("fixed-state objective mesh motion should be admissible");
        let velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
        let pressure_dimension = H1Space::new(PolynomialDegree::P1).num_global_dofs(&mesh);
        let velocity_dimension = velocity_space.num_global_dofs(&mesh);
        assert_eq!(state.len(), velocity_dimension + pressure_dimension);
        let penalty = ImmersedSolidPenalty::from_geometry_with_design(
            config,
            &geometry,
            solid_boundary_influence_m(&domain, nx, ny, nz),
            Some(design),
        );
        let shaft_power = matches!(
            objective,
            PcFanResolvedFlowStateObjective::RotatingWallShaftPower
        );
        let mut value = 0.0;
        for (element_index, element) in mesh.elements.iter().enumerate() {
            if element.element_type != sonium_fem::mesh::ElementType::Tetrahedron
                || element.nodes.len() != 4
            {
                continue;
            }
            let tabulation = velocity_space.tabulate(&mesh, element_index, 0.25, 0.25, 0.25);
            let dofs = velocity_space.local_to_global(&mesh, element_index);
            let mut centroid = [0.0; 3];
            let mut local_velocity = [0.0; 3];
            for &node in &element.nodes {
                let point = &mesh.nodes[node];
                centroid[0] += point.x * 0.25;
                centroid[1] += point.y * 0.25;
                centroid[2] += point.z * 0.25;
            }
            for (local_dof, &global_dof) in dofs.iter().enumerate() {
                for component in 0..3 {
                    local_velocity[component] +=
                        tabulation.vector_value(local_dof, component) * state[global_dof];
                }
            }
            let force = penalty.rotating_force_at(
                Point3::new(centroid[0], centroid[1], centroid[2]),
                local_velocity,
            );
            let integrand = if shaft_power {
                centroid[0] * force[1] - centroid[1] * force[0]
            } else {
                force[2]
            };
            value += integrand * tetrahedron_volume_m3(&mesh, &element.nodes);
        }
        if shaft_power {
            value * penalty.omega_rad_s
        } else {
            value
        }
    }

    #[test]
    fn rotating_wall_fixed_state_design_gradient_matches_p2_ale_difference() {
        let mut config = PcFanConfig::default_120mm();
        config.cfd.flow_model = PcFanFlowModel::IncompressibleCfdFull;
        config.cfd.backend = PcFanCfdBackend::Cpu;
        config.cfd.cells_x = 4;
        config.cfd.cells_y = 4;
        config.cfd.cells_z = 4;
        config.cfd.mesh_pattern = PcFanCfdMeshPattern::Symmetric;
        let design = config.default_design_variables();
        let geometry = config.build_geometry(&design);
        let domain = geometry.build_cfd_domain(&config);
        let mesh = box_mesh_tetrahedra_symmetric(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            config.cfd.cells_x,
            config.cfd.cells_y,
            config.cfd.cells_z,
        );
        let velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
        let velocity_dimension = velocity_space.num_global_dofs(&mesh);
        let pressure_dimension = H1Space::new(PolynomialDegree::P1).num_global_dofs(&mesh);
        let mut state = vec![0.0; velocity_dimension + pressure_dimension];
        for (index, value) in state[..velocity_dimension].iter_mut().enumerate() {
            *value = 0.02 * (index as f64 + 1.0);
        }

        let radial_start = BLADE_PATH_CONTROL_COUNT * 2 + BLADE_SECTION_STATION_COUNT * 4;
        let hub_index = radial_start;
        let tip_index = radial_start + 1;
        let shroud_index = radial_start + RADIAL_ENVELOPE_CONTROL_COUNT;
        let step = 1.0e-7;
        for objective in [
            PcFanResolvedFlowStateObjective::RotatingWallAxialLoading,
            PcFanResolvedFlowStateObjective::RotatingWallShaftPower,
        ] {
            let gradient = build_pc_fan_cfd_resolved_state_objective_design_gradient_from_state(
                &config,
                &design,
                &state,
                PolynomialDegree::P2,
                objective,
            )
            .expect("fixed-state P2 objective gradient should be admissible");
            for design_index in [hub_index, tip_index, shroud_index] {
                let mut plus = design.clone();
                let mut minus = design.clone();
                match design_index {
                    index if index == hub_index => {
                        plus.hub_radius_offset_m += step;
                        minus.hub_radius_offset_m -= step;
                    }
                    index if index == tip_index => {
                        plus.tip_radius_offset_m += step;
                        minus.tip_radius_offset_m -= step;
                    }
                    index => {
                        let shroud_control = index - shroud_index;
                        plus.shroud_radius_offsets_m[shroud_control] += step;
                        minus.shroud_radius_offsets_m[shroud_control] -= step;
                    }
                }
                let centered =
                    (fixed_state_rotating_wall_objective(&config, &plus, &state, objective)
                        - fixed_state_rotating_wall_objective(&config, &minus, &state, objective))
                        / (2.0 * step);
                let scale = gradient[design_index]
                    .abs()
                    .max(centered.abs())
                    .max(1.0e-10);
                assert!(
                    (gradient[design_index] - centered).abs() / scale < 3.0e-3,
                    "objective {objective:?}, design {design_index}: analytic={}, centered={centered}",
                    gradient[design_index]
                );
            }
        }
    }

    #[test]
    fn steady_state_objective_gradient_matches_boundary_operator_contract() {
        let (config, design, source) = source_fixture();
        let gradient = build_pc_fan_cfd_resolved_state_objective_gradient(
            &config,
            &design,
            &source,
            PcFanResolvedFlowStateObjective::PressureRise,
        )
        .expect("pressure-rise state gradient should be admitted");
        assert_eq!(gradient.len(), source.state.len());
        assert!(gradient.iter().all(|value| value.is_finite()));
        assert!(gradient.iter().any(|value| value.abs() > 0.0));

        let report = solve_pc_fan_cfd_resolved_state_adjoint(
            &PcFanResolvedAdjointConfig {
                tolerance: 1.0e-8,
                ..PcFanResolvedAdjointConfig::default()
            },
            &config,
            &design,
            &source,
            PcFanResolvedFlowStateObjective::PressureRise,
        )
        .expect("steady state-only adjoint should converge");
        assert!(report.converged);
        assert!(report.design_gradient.is_empty());
        assert!(report.adjoint.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn steady_state_objective_admits_full_wall_observables_but_rejects_surrogates() {
        let (mut config, design, source) = source_fixture();
        let mut full_source = source.clone();
        config.cfd.flow_model = PcFanFlowModel::IncompressibleCfdFull;
        full_source.flow_model = PcFanFlowModel::IncompressibleCfdFull;

        for objective in [
            PcFanResolvedFlowStateObjective::RotatingWallAxialLoading,
            PcFanResolvedFlowStateObjective::RotatingWallShaftPower,
        ] {
            let gradient = build_pc_fan_cfd_resolved_state_objective_gradient(
                &config,
                &design,
                &full_source,
                objective,
            )
            .expect("full-CFD wall state gradient should be admitted");
            assert!(gradient.iter().all(|value| value.is_finite()));
        }

        config.cfd.turbulence = PcFanTurbulenceModel::KOmegaSstAlgebraic;
        assert!(matches!(
            build_pc_fan_cfd_resolved_state_objective_gradient(
                &config,
                &design,
                &full_source,
                PcFanResolvedFlowStateObjective::PressureRise,
            ),
            Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible)
        ));
    }

    #[test]
    fn steady_wall_objective_uses_production_frozen_topology_derivatives() {
        let (config, design, source) = full_source_fixture();
        let residual_design_jacobian = source
            .residual_design_jacobian
            .as_ref()
            .expect("full-CFD production tangent should publish a design residual");
        assert_eq!(residual_design_jacobian.len(), source.state.len());
        assert!(residual_design_jacobian.iter().all(|row| {
            row.len() == DESIGN_VARIABLE_COUNT && row.iter().all(|value| value.is_finite())
        }));
        assert!(
            residual_design_jacobian
                .iter()
                .flat_map(|row| row.iter())
                .any(|value| value.abs() > 0.0)
        );

        for objective in [
            PcFanResolvedFlowObjective::RotatingWallAxialLoading,
            PcFanResolvedFlowObjective::RotatingWallShaftPower,
        ] {
            let derivatives = build_pc_fan_cfd_resolved_objective_derivatives(
                &config, &design, &source, objective,
            )
            .expect("frozen-topology wall objective should be admitted");
            assert_eq!(
                derivatives.objective_state_gradient.len(),
                source.state.len()
            );
            assert_eq!(
                derivatives.objective_design_gradient.len(),
                DESIGN_VARIABLE_COUNT
            );
            assert!(
                derivatives
                    .objective_state_gradient
                    .iter()
                    .all(|value| value.is_finite())
            );
            assert!(
                derivatives
                    .objective_design_gradient
                    .iter()
                    .all(|value| value.is_finite())
            );

            let report = solve_pc_fan_resolved_flow_adjoint_from_newton(
                &PcFanResolvedAdjointConfig {
                    tolerance: 1.0e-8,
                    ..PcFanResolvedAdjointConfig::default()
                },
                &source,
                derivatives,
            )
            .expect("frozen-topology wall objective adjoint should converge");
            assert!(report.converged);
            assert_eq!(report.design_gradient.len(), DESIGN_VARIABLE_COUNT);
            assert!(report.design_gradient.iter().all(|value| value.is_finite()));
        }

        let mut moved_config = config.clone();
        moved_config.cfd.turbulence = PcFanTurbulenceModel::KOmegaSstAlgebraic;
        assert!(matches!(
            build_pc_fan_cfd_resolved_objective_derivatives(
                &moved_config,
                &design,
                &source,
                PcFanResolvedFlowObjective::RotatingWallAxialLoading,
            ),
            Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible)
        ));

        let mut moved_design = design;
        moved_design.shroud_radius_offsets_m[0] = 1.0e-4;
        let moved_derivatives = build_pc_fan_cfd_resolved_objective_derivatives(
            &config,
            &moved_design,
            &source,
            PcFanResolvedFlowObjective::RotatingWallShaftPower,
        )
        .expect("finite shroud-radius steady derivatives should be admitted");
        assert_eq!(
            moved_derivatives.objective_state_gradient.len(),
            source.state.len()
        );
        assert!(
            moved_derivatives
                .objective_state_gradient
                .iter()
                .all(|value| value.is_finite())
        );

        let mut radial_design = moved_design;
        radial_design.tip_radius_offset_m = 1.0e-4;
        let radial_derivatives = build_pc_fan_cfd_resolved_objective_derivatives(
            &config,
            &radial_design,
            &source,
            PcFanResolvedFlowObjective::RotatingWallShaftPower,
        )
        .expect("fixed-mesh radial-envelope derivatives should be admitted");
        assert_eq!(
            radial_derivatives.objective_state_gradient.len(),
            source.state.len()
        );
        assert!(
            radial_derivatives
                .objective_state_gradient
                .iter()
                .all(|value| value.is_finite())
        );
        assert_eq!(
            radial_derivatives.objective_design_gradient.len(),
            DESIGN_VARIABLE_COUNT
        );
        assert!(
            radial_derivatives
                .objective_design_gradient
                .iter()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn full_cfd_boundary_objectives_use_ale_residual_derivatives() {
        let (config, design, mut source) = full_source_fixture();
        let geometry = config.build_geometry(&design);
        let domain = geometry.build_cfd_domain(&config);
        let mesh = box_mesh_tetrahedra_symmetric(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            config.cfd.cells_x,
            config.cfd.cells_y,
            config.cfd.cells_z,
        );
        for (node, point) in mesh.nodes.iter().enumerate() {
            if (point.z - domain.max_corner.z).abs() <= 1.0e-12 {
                source.state[3 * node + 2] += 1.0;
            }
        }
        for objective in [
            PcFanResolvedFlowObjective::Airflow,
            PcFanResolvedFlowObjective::PressureRise,
        ] {
            let derivatives = build_pc_fan_cfd_resolved_objective_derivatives(
                &config, &design, &source, objective,
            )
            .unwrap_or_else(|error| {
                panic!("full-CFD boundary objective {objective:?} should be admitted: {error:?}")
            });
            assert_eq!(
                derivatives.objective_design_gradient.len(),
                DESIGN_VARIABLE_COUNT
            );
            assert!(
                derivatives
                    .objective_design_gradient
                    .iter()
                    .all(|value| value.abs() <= 1.0e-14)
            );
            let report = solve_pc_fan_resolved_flow_adjoint_from_newton(
                &PcFanResolvedAdjointConfig {
                    tolerance: 1.0e-8,
                    ..PcFanResolvedAdjointConfig::default()
                },
                &source,
                derivatives,
            )
            .expect("full-CFD boundary objective adjoint should converge");
            assert!(report.converged);
            assert_eq!(report.design_gradient.len(), DESIGN_VARIABLE_COUNT);
            assert!(report.design_gradient.iter().all(|value| value.is_finite()));
        }
    }
}

#[test]
fn turbulent_state_objective_admits_unique_peak_newton_tangent() {
    let mut config = PcFanConfig::default_120mm();
    config.cfd.flow_model = PcFanFlowModel::IncompressibleCfdFull;
    config.cfd.backend = PcFanCfdBackend::Cpu;
    config.cfd.cells_x = 2;
    config.cfd.cells_y = 2;
    config.cfd.cells_z = 2;
    config.cfd.max_picard_iterations = 1;
    let design = config.default_design_variables();
    let laminar_source = solve_pc_fan_cfd_mean_flow(&config, &design)
        .picard_linearization
        .expect("fixture solve should publish state");
    config.cfd.turbulence = PcFanTurbulenceModel::KOmegaSstAlgebraic;

    // Break the symmetric fixture's peak tie so the algebraic max closure has
    // a valid local derivative. This also exercises the steady turbulent
    // design-residual and mesh-motion derivative handoff.
    let mut initial_state = laminar_source.state.clone();
    initial_state[12] += 10.0;
    let mut source = evaluate_pc_fan_cfd_newton_linearization(&config, &design, &initial_state)
        .expect("unique-peak turbulence tangent should be published");
    assert!(
        source
            .provenance
            .contains("newton-convection-turbulence-peak-subgradient")
    );
    let residual_design_jacobian = source
        .residual_design_jacobian
        .as_ref()
        .expect("turbulent mesh-motion design residual should be published");
    assert!(residual_design_jacobian.iter().all(|row| {
        row.len() == DESIGN_VARIABLE_COUNT && row.iter().all(|value| value.is_finite())
    }));
    let derivatives = build_pc_fan_cfd_resolved_objective_derivatives(
        &config,
        &design,
        &source,
        PcFanResolvedFlowObjective::RotatingWallAxialLoading,
    )
    .expect("turbulent rotating-wall design derivatives should be admitted");
    assert_eq!(
        derivatives.objective_design_gradient.len(),
        DESIGN_VARIABLE_COUNT
    );
    assert!(
        derivatives
            .objective_design_gradient
            .iter()
            .all(|value| value.is_finite())
    );
    source.state_residual.fill(0.0);

    let report = solve_pc_fan_cfd_resolved_state_adjoint(
        &PcFanResolvedAdjointConfig {
            tolerance: 1.0e-8,
            ..PcFanResolvedAdjointConfig::default()
        },
        &config,
        &design,
        &source,
        PcFanResolvedFlowStateObjective::PressureRise,
    )
    .expect("turbulent state-only adjoint should converge");
    assert!(report.converged);
    assert!(report.design_gradient.is_empty());
    assert!(report.adjoint.iter().all(|value| value.is_finite()));
}

#[test]
fn smooth_blade_wall_design_derivative_matches_centered_difference() {
    let config = PcFanConfig::default_120mm();
    let design = config.default_design_variables();
    let radius = config.dimensions.hub_radius_m
        + 0.5 * (config.dimensions.blade_tip_radius_m - config.dimensions.hub_radius_m);
    let transition = DEFAULT_BLADE_WALL_SMOOTHING_WIDTH_M;
    let y = 0.5 * design.blade_sections[2].thickness_m + 0.15 * transition;
    let x = (radius * radius - y * y).sqrt();
    let point = Point3::new(x, y, 0.0);
    let walls = BladeWallSet::from_design(&config, &design);
    let analytic =
        walls.smooth_weight_and_design_jacobian(point).1[BLADE_PATH_CONTROL_COUNT * 2 + 2 * 4 + 2];

    let step = 1.0e-7;
    let mut plus = design.clone();
    plus.blade_sections[2].thickness_m += step;
    let mut minus = design.clone();
    minus.blade_sections[2].thickness_m -= step;
    let plus_weight = BladeWallSet::from_design(&config, &plus)
        .smooth_weight_and_design_jacobian(point)
        .0;
    let minus_weight = BladeWallSet::from_design(&config, &minus)
        .smooth_weight_and_design_jacobian(point)
        .0;
    let centered = (plus_weight - minus_weight) / (2.0 * step);
    assert!(analytic.is_finite() && centered.is_finite());
    assert!(analytic.abs() > 1.0e-6);
    assert!((analytic - centered).abs() <= 2.0e-4 * analytic.abs().max(1.0));

    let geometry = config.build_geometry(&design);
    let velocity = [0.1, -0.2, 0.3];
    let penalty =
        ImmersedSolidPenalty::from_geometry_with_design(&config, &geometry, 1.0e-3, Some(&design));
    let force_derivative = penalty.rotating_force_design_jacobian_at(point, velocity)[0]
        [BLADE_PATH_CONTROL_COUNT * 2 + 2 * 4 + 2];
    let plus_penalty = ImmersedSolidPenalty::from_geometry_with_design(
        &config,
        &config.build_geometry(&plus),
        1.0e-3,
        Some(&plus),
    );
    let minus_penalty = ImmersedSolidPenalty::from_geometry_with_design(
        &config,
        &config.build_geometry(&minus),
        1.0e-3,
        Some(&minus),
    );
    let centered_force = (plus_penalty.rotating_force_at(point, velocity)[0]
        - minus_penalty.rotating_force_at(point, velocity)[0])
        / (2.0 * step);
    assert!(force_derivative.is_finite() && centered_force.is_finite());
    assert!((force_derivative - centered_force).abs() <= 2.0e-4 * force_derivative.abs().max(1.0));
}

#[test]
fn smooth_blade_wall_radial_envelope_derivative_matches_centered_difference() {
    let config = PcFanConfig::default_120mm();
    let design = config.default_design_variables();
    let eta = 0.37;
    let radius = config.dimensions.hub_radius_m
        + eta * (config.dimensions.blade_tip_radius_m - config.dimensions.hub_radius_m);
    let transition = DEFAULT_BLADE_WALL_SMOOTHING_WIDTH_M;
    let y = 0.5 * design.blade_sections[2].thickness_m + 0.15 * transition;
    let x = (radius * radius - y * y).sqrt();
    let point = Point3::new(x, y, 0.0);
    let radial_start = BLADE_PATH_CONTROL_COUNT * 2 + BLADE_SECTION_STATION_COUNT * 4;
    let walls = BladeWallSet::from_design(&config, &design);
    let analytic = walls.smooth_weight_and_design_jacobian(point).1[radial_start];

    let step = 1.0e-7;
    let mut hub_plus = design.clone();
    hub_plus.hub_radius_offset_m += step;
    let mut hub_minus = design.clone();
    hub_minus.hub_radius_offset_m -= step;
    let centered_hub = (BladeWallSet::from_design(&config, &hub_plus)
        .smooth_weight_and_design_jacobian(point)
        .0
        - BladeWallSet::from_design(&config, &hub_minus)
            .smooth_weight_and_design_jacobian(point)
            .0)
        / (2.0 * step);

    assert!(analytic.is_finite() && centered_hub.is_finite());
    assert!(analytic.abs() > 1.0e-6);
    assert!((analytic - centered_hub).abs() <= 2.0e-4 * analytic.abs().max(1.0));

    let mut tip_plus = design.clone();
    tip_plus.tip_radius_offset_m += step;
    let mut tip_minus = design.clone();
    tip_minus.tip_radius_offset_m -= step;
    let analytic_tip = walls.smooth_weight_and_design_jacobian(point).1[radial_start + 1];
    let centered_tip = (BladeWallSet::from_design(&config, &tip_plus)
        .smooth_weight_and_design_jacobian(point)
        .0
        - BladeWallSet::from_design(&config, &tip_minus)
            .smooth_weight_and_design_jacobian(point)
            .0)
        / (2.0 * step);
    assert!(analytic_tip.is_finite() && centered_tip.is_finite());
    assert!(analytic_tip.abs() > 1.0e-6);
    assert!((analytic_tip - centered_tip).abs() <= 2.0e-4 * analytic_tip.abs().max(1.0));

    let geometry = config.build_geometry(&design);
    let velocity = [0.1, -0.2, 0.3];
    let penalty =
        ImmersedSolidPenalty::from_geometry_with_design(&config, &geometry, 1.0e-3, Some(&design));
    let analytic_force =
        penalty.rotating_force_design_jacobian_at(point, velocity)[0][radial_start];
    let hub_plus_penalty = ImmersedSolidPenalty::from_geometry_with_design(
        &config,
        &config.build_geometry(&hub_plus),
        1.0e-3,
        Some(&hub_plus),
    );
    let hub_minus_penalty = ImmersedSolidPenalty::from_geometry_with_design(
        &config,
        &config.build_geometry(&hub_minus),
        1.0e-3,
        Some(&hub_minus),
    );
    let centered_force = (hub_plus_penalty.rotating_force_at(point, velocity)[0]
        - hub_minus_penalty.rotating_force_at(point, velocity)[0])
        / (2.0 * step);
    assert!(analytic_force.is_finite() && centered_force.is_finite());
    assert!((analytic_force - centered_force).abs() <= 2.0e-4 * analytic_force.abs().max(1.0));

    let analytic_tip_force =
        penalty.rotating_force_design_jacobian_at(point, velocity)[0][radial_start + 1];
    let tip_plus_penalty = ImmersedSolidPenalty::from_geometry_with_design(
        &config,
        &config.build_geometry(&tip_plus),
        1.0e-3,
        Some(&tip_plus),
    );
    let tip_minus_penalty = ImmersedSolidPenalty::from_geometry_with_design(
        &config,
        &config.build_geometry(&tip_minus),
        1.0e-3,
        Some(&tip_minus),
    );
    let centered_tip_force = (tip_plus_penalty.rotating_force_at(point, velocity)[0]
        - tip_minus_penalty.rotating_force_at(point, velocity)[0])
        / (2.0 * step);
    assert!(analytic_tip_force.is_finite() && centered_tip_force.is_finite());
    assert!(
        (analytic_tip_force - centered_tip_force).abs()
            <= 2.0e-4 * analytic_tip_force.abs().max(1.0)
    );
}

#[test]
fn blade_wall_force_position_jacobian_matches_centered_difference() {
    let config = PcFanConfig::default_120mm();
    let design = config.default_design_variables();
    let radius = config.dimensions.hub_radius_m
        + 0.5 * (config.dimensions.blade_tip_radius_m - config.dimensions.hub_radius_m);
    let transition = DEFAULT_BLADE_WALL_SMOOTHING_WIDTH_M;
    let y = 0.5 * design.blade_sections[2].thickness_m + 0.15 * transition;
    let point = Point3::new((radius * radius - y * y).sqrt(), y, 0.0);
    let geometry = config.build_geometry(&design);
    let penalty =
        ImmersedSolidPenalty::from_geometry_with_design(&config, &geometry, 1.0e-3, Some(&design));
    let velocity = [0.1, -0.2, 0.3];
    let jacobian = penalty.force_position_jacobian_at(point, velocity);
    let step = 1.0e-7;

    for column in 0..3 {
        let mut plus = [point.x, point.y, point.z];
        let mut minus = plus;
        plus[column] += step;
        minus[column] -= step;
        let plus_force = penalty.force_at(Point3::new(plus[0], plus[1], plus[2]), velocity);
        let minus_force = penalty.force_at(Point3::new(minus[0], minus[1], minus[2]), velocity);
        for row in 0..3 {
            let centered = (plus_force[row] - minus_force[row]) / (2.0 * step);
            let scale = jacobian[row][column].abs().max(centered.abs()).max(1.0);
            assert!(
                (jacobian[row][column] - centered).abs() / scale < 2.0e-4,
                "row {row}, column {column}: analytic={}, centered={centered}",
                jacobian[row][column]
            );
        }
    }
}

#[test]
fn actuator_force_position_jacobian_matches_centered_difference() {
    let config = PcFanConfig::default_120mm();
    let design = config.default_design_variables();
    let geometry = config.build_geometry(&design);
    let forcing = BladeActuatorForcing::from_geometry(&config, &geometry, &design);
    let radius = design.effective_hub_radius_m(&config)
        + 0.5 * (design.effective_tip_radius_m(&config) - design.effective_hub_radius_m(&config));
    let coords = [radius, 0.003, 0.0];
    let velocity = [0.1, -0.2, 0.3];
    let jacobian = forcing.force_position_jacobian_at(coords, velocity);
    let step = 1.0e-7;

    for column in 0..3 {
        let mut plus = coords;
        let mut minus = coords;
        plus[column] += step;
        minus[column] -= step;
        let plus_force = forcing.force_at(plus, velocity);
        let minus_force = forcing.force_at(minus, velocity);
        for row in 0..3 {
            let centered = (plus_force[row] - minus_force[row]) / (2.0 * step);
            let scale = jacobian[row][column].abs().max(centered.abs()).max(1.0);
            assert!(
                (jacobian[row][column] - centered).abs() / scale < 3.0e-4,
                "row {row}, column {column}: analytic={}, centered={centered}",
                jacobian[row][column]
            );
        }
    }
}

/// Build complete objective derivatives for a frozen-mesh actuator loading
/// observable at a production Newton state.
///
/// The state gradient differentiates the same centroid force integration used
/// by the published actuator ledger. The direct design gradient differentiates
/// the actuator loading law, Gaussian deposition, and panel geometry while
/// holding the FEM mesh/topology fixed. The design residual Jacobian still
/// comes from the production Newton observation and is required by the checked
/// adjoint handoff.
#[test]
fn smooth_shroud_wall_design_derivative_matches_centered_difference() {
    let config = PcFanConfig::default_120mm();
    let mut design = config.default_design_variables();
    // Keep the centered probe away from the frame-side radius clamp.
    design.shroud_radius_offsets_m = [-1.0e-3; SHROUD_CONTROL_COUNT];
    let geometry = config.build_geometry(&design);
    let depth_fraction = 0.5;
    let inner_radius = geometry.shroud.radius_at_depth_fraction(depth_fraction);
    let smoothing_width = DEFAULT_BLADE_WALL_SMOOTHING_WIDTH_M;
    let radius = inner_radius + 0.5 * smoothing_width;
    let angle: f64 = 0.2;
    let point = Point3::new(radius * angle.cos(), radius * angle.sin(), 0.0);
    let velocity = [0.1, -0.2, 0.3];
    let shroud_index = BLADE_PATH_CONTROL_COUNT * 2
        + BLADE_SECTION_STATION_COUNT * 4
        + RADIAL_ENVELOPE_CONTROL_COUNT
        + 2;
    let penalty =
        ImmersedSolidPenalty::from_geometry_with_design(&config, &geometry, 1.0e-3, Some(&design));
    let analytic = penalty.force_design_jacobian_at(point, velocity)[2][shroud_index];

    let step = 1.0e-7;
    let mut plus = design.clone();
    plus.shroud_radius_offsets_m[2] += step;
    let mut minus = design.clone();
    minus.shroud_radius_offsets_m[2] -= step;
    let plus_penalty = ImmersedSolidPenalty::from_geometry_with_design(
        &config,
        &config.build_geometry(&plus),
        1.0e-3,
        Some(&plus),
    );
    let minus_penalty = ImmersedSolidPenalty::from_geometry_with_design(
        &config,
        &config.build_geometry(&minus),
        1.0e-3,
        Some(&minus),
    );
    let centered = (plus_penalty.force_at(point, velocity)[2]
        - minus_penalty.force_at(point, velocity)[2])
        / (2.0 * step);

    assert!(analytic.is_finite() && centered.is_finite());
    assert!(analytic.abs() > 1.0e-6);
    assert!((analytic - centered).abs() <= 2.0e-4 * analytic.abs().max(1.0));
}

/// Build complete objective derivatives for the frozen-mesh actuator loading
/// observable and production Newton state.
pub fn build_pc_fan_cfd_actuator_objective_derivatives(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    source: &PcFanResolvedFlowNewtonLinearization,
    objective: PcFanResolvedFlowActuatorObjective,
) -> Result<PcFanResolvedFlowDerivativeData, PcFanResolvedAdjointError> {
    source.validate()?;
    if config.cfd.flow_model != PcFanFlowModel::IncompressibleCfd
        || config.cfd.flow_model.blade_as_wall()
        || !turbulence_newton_tangent_admitted(config, source)
        || design
            .shroud_radius_offsets_m
            .iter()
            .any(|offset| !offset.is_finite())
    {
        return Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible);
    }
    if source.residual_design_jacobian.is_none() {
        return Err(PcFanResolvedAdjointError::MissingProductionDesignResidualJacobian);
    }

    let geometry = config.build_geometry(design);
    let domain = geometry.build_cfd_domain(config);
    let nx = config.cfd.cells_x.max(1);
    let ny = config.cfd.cells_y.max(1);
    let nz = config.cfd.cells_z.max(1);
    let mut mesh = match config.cfd.mesh_pattern {
        PcFanCfdMeshPattern::Symmetric => box_mesh_tetrahedra_symmetric(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            nx,
            ny,
            nz,
        ),
        PcFanCfdMeshPattern::Kuhn => box_mesh_tetrahedra(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            nx,
            ny,
            nz,
        ),
    };
    apply_pc_fan_mesh_motion(&mut mesh, config, &geometry, design)
        .map_err(|_| PcFanResolvedAdjointError::ProbeEvaluationFailed("CFD mesh motion"))?;
    let velocity_space = VectorH1Space::new(PolynomialDegree::P1, 3);
    let pressure_space = H1Space::new(PolynomialDegree::P1);
    let velocity_dimension = velocity_space.num_global_dofs(&mesh);
    let pressure_dimension = pressure_space.num_global_dofs(&mesh);
    if source.state.len() != velocity_dimension + pressure_dimension {
        return Err(PcFanResolvedAdjointError::DimensionMismatch(
            "actuator objective state",
        ));
    }
    if velocity_dimension != mesh.num_nodes() * 3 {
        return Err(PcFanResolvedAdjointError::DimensionMismatch(
            "actuator objective velocity layout",
        ));
    }

    let forcing = BladeActuatorForcing::from_geometry(config, &geometry, design);
    let mut objective_state_gradient = vec![0.0; source.state.len()];
    let mut objective_design_gradient = vec![0.0; DESIGN_VARIABLE_COUNT];
    let angular_speed = forcing.omega_rad_s;

    for element in &mesh.elements {
        if element.element_type != sonium_fem::mesh::ElementType::Tetrahedron
            || element.nodes.len() != 4
        {
            continue;
        }
        let mut centroid = [0.0; 3];
        let mut local_velocity = [0.0; 3];
        for &node in &element.nodes {
            let point = &mesh.nodes[node];
            centroid[0] += point.x * 0.25;
            centroid[1] += point.y * 0.25;
            centroid[2] += point.z * 0.25;
            for (component, value) in local_velocity.iter_mut().enumerate() {
                *value += source.state[3 * node + component] * 0.25;
            }
        }
        let volume = tetrahedron_volume_m3(&mesh, &element.nodes);
        if volume <= 0.0 {
            continue;
        }
        let force_jacobian = forcing.force_jacobian_at(centroid, local_velocity);
        let force_design_jacobian = forcing.force_design_jacobian_at(centroid, local_velocity);
        let lever_x = centroid[0];
        let lever_y = centroid[1];
        let state_scale = match objective {
            PcFanResolvedFlowActuatorObjective::AxialLoading => 1.0,
            PcFanResolvedFlowActuatorObjective::ShaftPower => angular_speed,
        };
        let output_row = match objective {
            PcFanResolvedFlowActuatorObjective::AxialLoading => 2,
            PcFanResolvedFlowActuatorObjective::ShaftPower => usize::MAX,
        };
        for &node in &element.nodes {
            for input in 0..3 {
                let derivative = if output_row != usize::MAX {
                    force_jacobian[output_row][input]
                } else {
                    lever_x * force_jacobian[1][input] - lever_y * force_jacobian[0][input]
                };
                objective_state_gradient[3 * node + input] +=
                    state_scale * derivative * volume * 0.25;
            }
        }
        for design_index in 0..DESIGN_VARIABLE_COUNT {
            let derivative = if output_row != usize::MAX {
                force_design_jacobian[output_row][design_index]
            } else {
                lever_x * force_design_jacobian[1][design_index]
                    - lever_y * force_design_jacobian[0][design_index]
            };
            objective_design_gradient[design_index] += state_scale * derivative * volume;
        }
    }

    if objective_state_gradient
        .iter()
        .chain(&objective_design_gradient)
        .any(|value| !value.is_finite())
    {
        return Err(PcFanResolvedAdjointError::NonFiniteInput(
            "actuator objective derivatives",
        ));
    }
    PcFanResolvedFlowDerivativeData::from_production_newton(
        source,
        objective.provenance(),
        objective_state_gradient,
        objective_design_gradient,
    )
}

/// Build an exact state gradient for a steady resolved-flow observation.
///
/// The quadrature and recovered-wall operator match the production CFD
/// observables. Geometry, mesh, and immersed-support derivatives are not
/// included; callers that need a design gradient must use the stricter
/// [`build_pc_fan_cfd_resolved_objective_derivatives`] contract instead.
pub fn build_pc_fan_cfd_resolved_state_objective_gradient(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    source: &PcFanResolvedFlowNewtonLinearization,
    objective: PcFanResolvedFlowStateObjective,
) -> Result<Vec<f64>, PcFanResolvedAdjointError> {
    source.validate()?;
    if source.flow_model != config.cfd.flow_model
        || !turbulence_newton_tangent_admitted(config, source)
    {
        return Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible);
    }
    if matches!(
        objective,
        PcFanResolvedFlowStateObjective::RotatingWallAxialLoading
            | PcFanResolvedFlowStateObjective::RotatingWallShaftPower
    ) && !config.cfd.flow_model.blade_as_wall()
    {
        return Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible);
    }

    let geometry = config.build_geometry(design);
    let domain = geometry.build_cfd_domain(config);
    let nx = config.cfd.cells_x.max(1);
    let ny = config.cfd.cells_y.max(1);
    let nz = config.cfd.cells_z.max(1);
    let mut mesh = match config.cfd.mesh_pattern {
        PcFanCfdMeshPattern::Symmetric => box_mesh_tetrahedra_symmetric(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            nx,
            ny,
            nz,
        ),
        PcFanCfdMeshPattern::Kuhn => box_mesh_tetrahedra(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            nx,
            ny,
            nz,
        ),
    };
    apply_pc_fan_mesh_motion(&mut mesh, config, &geometry, design)
        .map_err(|_| PcFanResolvedAdjointError::ProbeEvaluationFailed("CFD mesh motion"))?;
    let velocity_space = VectorH1Space::new(PolynomialDegree::P1, 3);
    let pressure_space = H1Space::new(PolynomialDegree::P1);
    let velocity_dimension = velocity_space.num_global_dofs(&mesh);
    let pressure_dimension = pressure_space.num_global_dofs(&mesh);
    if velocity_dimension != mesh.num_nodes() * 3
        || pressure_dimension != mesh.num_nodes()
        || source.state.len() != velocity_dimension + pressure_dimension
    {
        return Err(PcFanResolvedAdjointError::DimensionMismatch(
            "steady resolved state objective layout",
        ));
    }

    let nodes = mesh
        .nodes
        .iter()
        .map(|node| Point3::new(node.x, node.y, node.z))
        .collect::<Vec<_>>();
    let boundary_tolerance = boundary_tolerance(&domain, nx, ny, nz);
    let mut gradient = vec![0.0; source.state.len()];

    match objective {
        PcFanResolvedFlowStateObjective::Airflow => {
            let velocity = velocity_vectors(&source.state[..velocity_dimension], nodes.len());
            let signed_flow = boundary_airflow_cfm(
                &nodes,
                &mesh.elements,
                &velocity,
                domain.max_corner.z,
                domain.outlet_area_m2,
                boundary_tolerance,
            );
            if !signed_flow.is_finite() {
                return Err(PcFanResolvedAdjointError::NonFiniteInput(
                    "steady airflow state objective",
                ));
            }
            if signed_flow.abs() <= f64::EPSILON {
                return Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible);
            }
            let sign = signed_flow.signum();
            for (node, weight) in boundary_airflow_state_weights(
                &nodes,
                &mesh.elements,
                domain.max_corner.z,
                domain.outlet_area_m2,
                boundary_tolerance,
            )
            .into_iter()
            .enumerate()
            {
                gradient[3 * node + 2] = sign * weight;
            }
        }
        PcFanResolvedFlowStateObjective::PressureRise => {
            let inlet = boundary_average_pressure_state_weights(
                &nodes,
                domain.min_corner.z,
                boundary_tolerance,
            );
            let outlet = boundary_average_pressure_state_weights(
                &nodes,
                domain.max_corner.z,
                boundary_tolerance,
            );
            if inlet.iter().all(|weight| *weight == 0.0)
                || outlet.iter().all(|weight| *weight == 0.0)
            {
                return Err(PcFanResolvedAdjointError::DimensionMismatch(
                    "steady pressure objective boundary",
                ));
            }
            for node in 0..nodes.len() {
                gradient[velocity_dimension + node] = outlet[node] - inlet[node];
            }
        }
        PcFanResolvedFlowStateObjective::RotatingWallAxialLoading
        | PcFanResolvedFlowStateObjective::RotatingWallShaftPower => {
            let penalty = ImmersedSolidPenalty::from_geometry_with_design(
                config,
                &geometry,
                solid_boundary_influence_m(&domain, nx, ny, nz),
                Some(design),
            );
            let state_scale = if matches!(
                objective,
                PcFanResolvedFlowStateObjective::RotatingWallShaftPower
            ) {
                penalty.omega_rad_s
            } else {
                1.0
            };
            for element in &mesh.elements {
                if element.element_type != sonium_fem::mesh::ElementType::Tetrahedron
                    || element.nodes.len() != 4
                {
                    continue;
                }
                let mut centroid = [0.0; 3];
                for &node in &element.nodes {
                    let point = &mesh.nodes[node];
                    centroid[0] += point.x * 0.25;
                    centroid[1] += point.y * 0.25;
                    centroid[2] += point.z * 0.25;
                }
                let volume = tetrahedron_volume_m3(&mesh, &element.nodes);
                if volume <= 0.0 {
                    continue;
                }
                let jacobian = penalty.rotating_force_jacobian_at(Point3::new(
                    centroid[0],
                    centroid[1],
                    centroid[2],
                ));
                let lever_x = centroid[0];
                let lever_y = centroid[1];
                for &node in &element.nodes {
                    for input in 0..3 {
                        let derivative = if matches!(
                            objective,
                            PcFanResolvedFlowStateObjective::RotatingWallAxialLoading
                        ) {
                            jacobian[2][input]
                        } else {
                            lever_x * jacobian[1][input] - lever_y * jacobian[0][input]
                        };
                        gradient[3 * node + input] += state_scale * derivative * volume * 0.25;
                    }
                }
            }
        }
    }

    if gradient.iter().any(|value| !value.is_finite()) {
        return Err(PcFanResolvedAdjointError::NonFiniteInput(
            "steady resolved state objective gradient",
        ));
    }
    Ok(gradient)
}

fn turbulence_newton_tangent_admitted(
    config: &PcFanConfig,
    source: &PcFanResolvedFlowNewtonLinearization,
) -> bool {
    let turbulent_provenance = source
        .provenance
        .contains("newton-convection-turbulence-peak-subgradient");
    match config.cfd.turbulence {
        PcFanTurbulenceModel::Laminar => !turbulent_provenance,
        PcFanTurbulenceModel::KOmegaSstAlgebraic => turbulent_provenance,
        PcFanTurbulenceModel::KOmegaSstTransported => false,
    }
}

/// Build the direct frozen-mesh design derivative of a rotating-wall
/// objective. The state derivative is assembled separately; this term is
/// the derivative of the immersed-wall loading integrand itself.
fn build_pc_fan_cfd_resolved_state_objective_design_gradient(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    source: &PcFanResolvedFlowNewtonLinearization,
    objective: PcFanResolvedFlowStateObjective,
) -> Result<Vec<f64>, PcFanResolvedAdjointError> {
    source.validate()?;
    if !turbulence_newton_tangent_admitted(config, source) {
        return Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible);
    }
    build_pc_fan_cfd_resolved_state_objective_design_gradient_from_state(
        config,
        design,
        &source.state,
        PolynomialDegree::P1,
        objective,
    )
}

fn build_pc_fan_cfd_resolved_state_objective_design_gradient_from_state(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    state: &[f64],
    velocity_degree: PolynomialDegree,
    objective: PcFanResolvedFlowStateObjective,
) -> Result<Vec<f64>, PcFanResolvedAdjointError> {
    if !config.cfd.flow_model.blade_as_wall() {
        return Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible);
    }
    let geometry = config.build_geometry(design);
    let domain = geometry.build_cfd_domain(config);
    let nx = config.cfd.cells_x.max(1);
    let ny = config.cfd.cells_y.max(1);
    let nz = config.cfd.cells_z.max(1);
    let mut mesh = match config.cfd.mesh_pattern {
        PcFanCfdMeshPattern::Symmetric => box_mesh_tetrahedra_symmetric(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            nx,
            ny,
            nz,
        ),
        PcFanCfdMeshPattern::Kuhn => box_mesh_tetrahedra(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            nx,
            ny,
            nz,
        ),
    };
    let mesh_motion_design_jacobian = pc_fan_mesh_motion_design_jacobian(&mesh, config, design)
        .map_err(|_| PcFanResolvedAdjointError::ProbeEvaluationFailed("CFD mesh motion"))?;
    apply_pc_fan_mesh_motion(&mut mesh, config, &geometry, design)
        .map_err(|_| PcFanResolvedAdjointError::ProbeEvaluationFailed("CFD mesh motion"))?;
    let velocity_space = VectorH1Space::new(velocity_degree, 3);
    let velocity_dimension = velocity_space.num_global_dofs(&mesh);
    let pressure_dimension = H1Space::new(PolynomialDegree::P1).num_global_dofs(&mesh);
    if state.len() != velocity_dimension + pressure_dimension {
        return Err(PcFanResolvedAdjointError::DimensionMismatch(
            "rotating-wall objective design state",
        ));
    }
    let penalty = ImmersedSolidPenalty::from_geometry_with_design(
        config,
        &geometry,
        solid_boundary_influence_m(&domain, nx, ny, nz),
        Some(design),
    );
    let shaft_power = matches!(
        objective,
        PcFanResolvedFlowStateObjective::RotatingWallShaftPower
    );
    let mut gradient = vec![0.0; DESIGN_VARIABLE_COUNT];
    for (element_index, element) in mesh.elements.iter().enumerate() {
        if element.element_type != sonium_fem::mesh::ElementType::Tetrahedron
            || element.nodes.len() != 4
        {
            continue;
        }
        let velocity_dofs = velocity_space.local_to_global(&mesh, element_index);
        let velocity_tabulation = velocity_space.tabulate(&mesh, element_index, 0.25, 0.25, 0.25);
        let mut centroid = [0.0; 3];
        let mut local_velocity = [0.0; 3];
        let mut centroid_design_jacobian = [[0.0; DESIGN_VARIABLE_COUNT]; 3];
        for &node in &element.nodes {
            let point = &mesh.nodes[node];
            centroid[0] += point.x * 0.25;
            centroid[1] += point.y * 0.25;
            centroid[2] += point.z * 0.25;
            for coordinate in 0..3 {
                for design_index in 0..DESIGN_VARIABLE_COUNT {
                    centroid_design_jacobian[coordinate][design_index] +=
                        mesh_motion_design_jacobian[node][design_index][coordinate] * 0.25;
                }
            }
        }
        for (local_dof, &global_dof) in velocity_dofs.iter().enumerate() {
            for (component, slot) in local_velocity.iter_mut().enumerate() {
                *slot += velocity_tabulation.vector_value(local_dof, component) * state[global_dof];
            }
        }
        let volume = tetrahedron_volume_m3(&mesh, &element.nodes);
        if volume <= 0.0 {
            continue;
        }
        let volume_design_jacobian =
            tetrahedron_volume_design_jacobian(&mesh, &element.nodes, &mesh_motion_design_jacobian);
        let force = penalty.rotating_force_at(
            Point3::new(centroid[0], centroid[1], centroid[2]),
            local_velocity,
        );
        let force_design_jacobian = penalty.rotating_force_design_jacobian_at(
            Point3::new(centroid[0], centroid[1], centroid[2]),
            local_velocity,
        );
        let force_position_jacobian = penalty.force_position_jacobian_at(
            Point3::new(centroid[0], centroid[1], centroid[2]),
            local_velocity,
        );
        for design_index in 0..DESIGN_VARIABLE_COUNT {
            let mut force_design_derivative = [0.0; 3];
            for force_component in 0..3 {
                force_design_derivative[force_component] = force_design_jacobian[force_component]
                    [design_index]
                    + (0..3)
                        .map(|coordinate| {
                            force_position_jacobian[force_component][coordinate]
                                * centroid_design_jacobian[coordinate][design_index]
                        })
                        .sum::<f64>();
            }
            let objective_value = if shaft_power {
                centroid[0] * force[1] - centroid[1] * force[0]
            } else {
                force[2]
            };
            let objective_derivative = if shaft_power {
                centroid_design_jacobian[0][design_index] * force[1]
                    + centroid[0] * force_design_derivative[1]
                    - centroid_design_jacobian[1][design_index] * force[0]
                    - centroid[1] * force_design_derivative[0]
            } else {
                force_design_derivative[2]
            };
            let scale = if shaft_power {
                penalty.omega_rad_s
            } else {
                1.0
            };
            gradient[design_index] += scale
                * (objective_derivative * volume
                    + objective_value * volume_design_jacobian[design_index]);
        }
    }
    if gradient.iter().any(|value| !value.is_finite()) {
        return Err(PcFanResolvedAdjointError::NonFiniteInput(
            "rotating-wall objective design derivatives",
        ));
    }
    Ok(gradient)
}

/// Solve the checked steady resolved-flow state adjoint.
///
/// This state-only path remains useful for state sensitivity and multiplier
/// diagnostics. Complete design adjoints use the production residual and
/// direct rotating-wall derivatives assembled by the full path.
pub fn solve_pc_fan_cfd_resolved_state_adjoint(
    adjoint_config: &PcFanResolvedAdjointConfig,
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    source: &PcFanResolvedFlowNewtonLinearization,
    objective: PcFanResolvedFlowStateObjective,
) -> Result<PcFanResolvedAdjointReport, PcFanResolvedAdjointError> {
    let objective_state_gradient =
        build_pc_fan_cfd_resolved_state_objective_gradient(config, design, source, objective)?;
    let dimension = source.state_jacobian.dimension;
    let linearization = crate::resolved_adjoint::PcFanResolvedFlowLinearization {
        flow_model: source.flow_model,
        backend: source.backend,
        provenance: format!(
            "{}; state-objective:{}",
            source.provenance,
            objective.provenance()
        ),
        state_jacobian: source.state_jacobian.clone(),
        state_residual: source.state_residual.clone(),
        residual_design_jacobian: vec![Vec::new(); dimension],
        objective_state_gradient,
        objective_design_gradient: Vec::new(),
    };
    crate::resolved_adjoint::solve_pc_fan_resolved_flow_adjoint(adjoint_config, &linearization)
}

/// Build complete topology-preserving ALE derivatives for a resolved-flow
/// objective.
// resolved-flow objective derivative dispatch
///
/// Actuator objectives delegate to the original, explicitly named helper.
/// Boundary objectives use analytic gradients of the production outlet-flow
/// Build checked topology-preserving ALE derivatives for a resolved-flow
/// objective.
///
/// The rotating-wall variants use the production state gradient plus the
/// topology-preserving ALE design-residual contract. The hard immersed-wall
/// occupancy is not differentiated; smooth blade/shroud support fields and
/// mesh-motion terms are. Topology-changing remeshing remains outside this
/// API.
pub fn build_pc_fan_cfd_resolved_objective_derivatives(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    source: &PcFanResolvedFlowNewtonLinearization,
    objective: PcFanResolvedFlowObjective,
) -> Result<PcFanResolvedFlowDerivativeData, PcFanResolvedAdjointError> {
    let state_objective = match objective {
        PcFanResolvedFlowObjective::RotatingWallAxialLoading => {
            Some(PcFanResolvedFlowStateObjective::RotatingWallAxialLoading)
        }
        PcFanResolvedFlowObjective::RotatingWallShaftPower => {
            Some(PcFanResolvedFlowStateObjective::RotatingWallShaftPower)
        }
        _ => None,
    };
    if let Some(state_objective) = state_objective {
        source.validate()?;
        if config.cfd.flow_model != PcFanFlowModel::IncompressibleCfdFull
            || !config.cfd.flow_model.blade_as_wall()
            || !turbulence_newton_tangent_admitted(config, source)
            || design
                .shroud_radius_offsets_m
                .iter()
                .any(|offset| !offset.is_finite())
        {
            return Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible);
        }
        if source.residual_design_jacobian.is_none() {
            return Err(PcFanResolvedAdjointError::MissingProductionDesignResidualJacobian);
        }
        let objective_state_gradient = build_pc_fan_cfd_resolved_state_objective_gradient(
            config,
            design,
            source,
            state_objective,
        )?;
        let objective_design_gradient = build_pc_fan_cfd_resolved_state_objective_design_gradient(
            config,
            design,
            source,
            state_objective,
        )?;
        return PcFanResolvedFlowDerivativeData::from_production_newton(
            source,
            state_objective.provenance(),
            objective_state_gradient,
            objective_design_gradient,
        );
    }
    build_pc_fan_cfd_resolved_objective_derivatives_legacy(config, design, source, objective)
}
/// and boundary-pressure quadratures, and retain the same checked production
/// Newton/design-residual provenance.
pub fn build_pc_fan_cfd_resolved_objective_derivatives_legacy(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    source: &PcFanResolvedFlowNewtonLinearization,
    objective: PcFanResolvedFlowObjective,
) -> Result<PcFanResolvedFlowDerivativeData, PcFanResolvedAdjointError> {
    if let PcFanResolvedFlowObjective::Actuator(actuator) = objective {
        return build_pc_fan_cfd_actuator_objective_derivatives(config, design, source, actuator);
    }

    source.validate()?;
    if !config.cfd.flow_model.is_cfd()
        || !turbulence_newton_tangent_admitted(config, source)
        || design
            .shroud_radius_offsets_m
            .iter()
            .any(|offset| !offset.is_finite())
    {
        return Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible);
    }
    if source.residual_design_jacobian.is_none() {
        return Err(PcFanResolvedAdjointError::MissingProductionDesignResidualJacobian);
    }

    let geometry = config.build_geometry(design);
    let domain = geometry.build_cfd_domain(config);
    let nx = config.cfd.cells_x.max(1);
    let ny = config.cfd.cells_y.max(1);
    let nz = config.cfd.cells_z.max(1);
    let mut mesh = match config.cfd.mesh_pattern {
        PcFanCfdMeshPattern::Symmetric => box_mesh_tetrahedra_symmetric(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            nx,
            ny,
            nz,
        ),
        PcFanCfdMeshPattern::Kuhn => box_mesh_tetrahedra(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            nx,
            ny,
            nz,
        ),
    };
    apply_pc_fan_mesh_motion(&mut mesh, config, &geometry, design)
        .map_err(|_| PcFanResolvedAdjointError::ProbeEvaluationFailed("CFD mesh motion"))?;
    let velocity_space = VectorH1Space::new(PolynomialDegree::P1, 3);
    let pressure_space = H1Space::new(PolynomialDegree::P1);
    let velocity_dimension = velocity_space.num_global_dofs(&mesh);
    let pressure_dimension = pressure_space.num_global_dofs(&mesh);
    if source.state.len() != velocity_dimension + pressure_dimension {
        return Err(PcFanResolvedAdjointError::DimensionMismatch(
            "resolved objective state",
        ));
    }
    if velocity_dimension != mesh.num_nodes() * 3 || pressure_dimension != mesh.num_nodes() {
        return Err(PcFanResolvedAdjointError::DimensionMismatch(
            "resolved objective nodal layout",
        ));
    }

    let nodes = mesh
        .nodes
        .iter()
        .map(|node| Point3::new(node.x, node.y, node.z))
        .collect::<Vec<_>>();
    let velocity = velocity_vectors(&source.state[..velocity_dimension], nodes.len());
    let boundary_tolerance = boundary_tolerance(&domain, nx, ny, nz);
    let mut objective_state_gradient = vec![0.0; source.state.len()];
    let objective_provenance = match objective {
        PcFanResolvedFlowObjective::Airflow => {
            let signed_flow = boundary_airflow_cfm(
                &nodes,
                &mesh.elements,
                &velocity,
                domain.max_corner.z,
                domain.outlet_area_m2,
                boundary_tolerance,
            );
            if signed_flow.abs() <= f64::EPSILON {
                return Err(PcFanResolvedAdjointError::DerivativeProbeNotAdmissible);
            }
            let sign = signed_flow.signum();
            for (node, weight) in boundary_airflow_state_weights(
                &nodes,
                &mesh.elements,
                domain.max_corner.z,
                domain.outlet_area_m2,
                boundary_tolerance,
            )
            .into_iter()
            .enumerate()
            {
                objective_state_gradient[3 * node + 2] = sign * weight;
            }
            "analytic-boundary-airflow"
        }
        PcFanResolvedFlowObjective::PressureRise => {
            let inlet_weights = boundary_average_pressure_state_weights(
                &nodes,
                domain.min_corner.z,
                boundary_tolerance,
            );
            let outlet_weights = boundary_average_pressure_state_weights(
                &nodes,
                domain.max_corner.z,
                boundary_tolerance,
            );
            if inlet_weights.iter().all(|weight| *weight == 0.0)
                || outlet_weights.iter().all(|weight| *weight == 0.0)
            {
                return Err(PcFanResolvedAdjointError::DimensionMismatch(
                    "resolved objective pressure boundary",
                ));
            }
            for node in 0..nodes.len() {
                objective_state_gradient[velocity_dimension + node] =
                    outlet_weights[node] - inlet_weights[node];
            }
            "analytic-boundary-pressure-rise"
        }
        PcFanResolvedFlowObjective::Actuator(_) => {
            unreachable!("actuator objectives are delegated before boundary derivative assembly")
        }
        PcFanResolvedFlowObjective::RotatingWallAxialLoading
        | PcFanResolvedFlowObjective::RotatingWallShaftPower => {
            unreachable!("rotating-wall objectives are dispatched before boundary assembly")
        }
    };

    let objective_design_gradient = vec![0.0; DESIGN_VARIABLE_COUNT];
    if objective_state_gradient
        .iter()
        .chain(&objective_design_gradient)
        .any(|value| !value.is_finite())
    {
        return Err(PcFanResolvedAdjointError::NonFiniteInput(
            "resolved objective derivatives",
        ));
    }
    PcFanResolvedFlowDerivativeData::from_production_newton(
        source,
        objective_provenance,
        objective_state_gradient,
        objective_design_gradient,
    )
}

/// Solve the checked resolved-flow adjoint for a production actuator loading
/// objective. This is the complete public handoff for the supported
/// frozen-mesh actuator-disk slice; unsupported objective/geometry terms fail
/// closed before an adjoint solve is attempted.
pub fn solve_pc_fan_cfd_actuator_adjoint(
    adjoint_config: &PcFanResolvedAdjointConfig,
    pc_fan_config: &PcFanConfig,
    design: &PcFanDesignVariables,
    initial_state: &[f64],
    objective: PcFanResolvedFlowActuatorObjective,
) -> Result<PcFanResolvedAdjointReport, PcFanResolvedAdjointError> {
    let newton = evaluate_pc_fan_cfd_newton_linearization(pc_fan_config, design, initial_state)?;
    let derivatives =
        build_pc_fan_cfd_actuator_objective_derivatives(pc_fan_config, design, &newton, objective)?;
    solve_pc_fan_resolved_flow_adjoint_from_newton(adjoint_config, &newton, derivatives)
}

/// Solve the checked resolved-flow adjoint for any currently supported
/// frozen-mesh objective, including actuator loading, shaft power, outlet
/// airflow, and pressure rise.
pub fn solve_pc_fan_cfd_resolved_objective_adjoint(
    adjoint_config: &PcFanResolvedAdjointConfig,
    pc_fan_config: &PcFanConfig,
    design: &PcFanDesignVariables,
    initial_state: &[f64],
    objective: PcFanResolvedFlowObjective,
) -> Result<PcFanResolvedAdjointReport, PcFanResolvedAdjointError> {
    let newton = evaluate_pc_fan_cfd_newton_linearization(pc_fan_config, design, initial_state)?;
    let derivatives =
        build_pc_fan_cfd_resolved_objective_derivatives(pc_fan_config, design, &newton, objective)?;
    solve_pc_fan_resolved_flow_adjoint_from_newton(adjoint_config, &newton, derivatives)
}

/// Solve a resolved-flow objective adjoint from an already completed
/// production CFD result.  Unlike the prescribed-state map wrapper, this
/// entry point requires the caller to hand over a converged solution so that
/// a one-step Picard/Oseen observation cannot be promoted accidentally.
pub fn solve_pc_fan_cfd_resolved_objective_adjoint_from_solution(
    adjoint_config: &PcFanResolvedAdjointConfig,
    pc_fan_config: &PcFanConfig,
    design: &PcFanDesignVariables,
    solution: &PcFanCfdFlowSolution,
    objective: PcFanResolvedFlowObjective,
) -> Result<PcFanResolvedAdjointReport, PcFanResolvedAdjointError> {
    if !solution.converged {
        return Err(PcFanResolvedAdjointError::ProductionStateNotConverged);
    }
    if solution.backend_provenance.actual_backend != solution.actual_backend
        || solution.backend_provenance.requested_backend != pc_fan_config.cfd.backend
        || !solution
            .execution_environment
            .matches_provenance(solution.backend_provenance)
    {
        return Err(PcFanResolvedAdjointError::BackendProvenanceMismatch);
    }
    if !solution.backend_provenance.is_admitted(
        pc_fan_config.cfd.allow_host_fallback,
        pc_fan_config.cfd.require_device_resident,
    ) {
        return Err(PcFanResolvedAdjointError::BackendExecutionNotAdmitted {
            requested: solution.backend_provenance.requested_backend,
            actual: solution.backend_provenance.actual_backend,
            mode: solution.backend_provenance.execution_mode,
        });
    }
    let newton = solution
        .newton_linearization
        .as_ref()
        .ok_or(PcFanResolvedAdjointError::MissingProductionNewtonLinearization)?;
    let derivatives =
        build_pc_fan_cfd_resolved_objective_derivatives(pc_fan_config, design, newton, objective)?;
    solve_pc_fan_resolved_flow_adjoint_from_newton(adjoint_config, newton, derivatives)
}

/// Run the checked derivative probe against the production PC-fan FEM map.
///
/// The objective callback is required from the caller because objective terms
/// must be declared explicitly at this resolved-flow seam. The helper wires
/// both state and design perturbations to the actual production mesh/force
/// assembly and preserves the probe's fail-closed, non-admissible provenance.
pub fn probe_pc_fan_cfd_resolved_flow_derivatives<ObjectiveFn>(
    config: &PcFanResolvedDerivativeProbeConfig,
    pc_fan_config: &PcFanConfig,
    design: &PcFanDesignVariables,
    base: &PcFanResolvedFlowPicardLinearization,
    objective: ObjectiveFn,
) -> Result<PcFanResolvedFlowDerivativeProbeReport, PcFanResolvedAdjointError>
where
    ObjectiveFn: Fn(&[f64], &[f64]) -> Result<f64, PcFanResolvedAdjointError>,
{
    let design_vector = design.to_vector();
    probe_pc_fan_resolved_flow_derivatives(
        config,
        base,
        &design_vector,
        |state| evaluate_pc_fan_cfd_picard_map(pc_fan_config, design, state),
        |values| {
            let perturbed_design = design_from_probe_values(values)?;
            evaluate_pc_fan_cfd_picard_map(pc_fan_config, &perturbed_design, &base.state)
        },
        objective,
    )
}

fn design_from_probe_values(
    values: &[f64],
) -> Result<PcFanDesignVariables, PcFanResolvedAdjointError> {
    if values.len() != DESIGN_VARIABLE_COUNT {
        return Err(PcFanResolvedAdjointError::DimensionMismatch(
            "production Picard design vector",
        ));
    }
    let mut vector = [0.0; DESIGN_VARIABLE_COUNT];
    vector.copy_from_slice(values);
    Ok(PcFanDesignVariables::from_vector(vector))
}

pub fn solve_pc_fan_cfd_mesh_convergence(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    convergence: PcFanCfdMeshConvergenceConfig,
) -> PcFanCfdMeshConvergence {
    try_solve_pc_fan_cfd_mesh_convergence(config, design, convergence).unwrap_or_else(|error| {
        panic!("invalid PC-fan CFD mesh convergence configuration: {error}")
    })
}

/// Run a bounded cubic-cell CFD refinement sweep after validating its
/// configuration.  The checked entry point is intended for production
/// callers that need to reject bad admission settings before paying for any
/// CFD solve; [`solve_pc_fan_cfd_mesh_convergence`] remains as the historical
/// compatibility wrapper.
fn cfd_conservation_tolerances(config: &PcFanConfig) -> [(&'static str, f64); 4] {
    [
        (
            "max_relative_mass_imbalance",
            config.cfd.max_relative_mass_imbalance,
        ),
        (
            "max_relative_momentum_imbalance",
            config.cfd.max_relative_momentum_imbalance,
        ),
        (
            "max_relative_pressure_closure",
            config.cfd.max_relative_pressure_closure,
        ),
        (
            "max_relative_energy_imbalance",
            config.cfd.max_relative_energy_imbalance,
        ),
    ]
}

fn first_invalid_cfd_conservation_tolerance(config: &PcFanConfig) -> Option<(&'static str, f64)> {
    cfd_conservation_tolerances(config)
        .into_iter()
        .find(|(_, tolerance)| !tolerance.is_finite() || *tolerance < 0.0)
}

fn validate_mesh_conservation_tolerances(
    config: &PcFanConfig,
) -> Result<(), PcFanCfdMeshConvergenceConfigError> {
    if let Some((metric, _)) = first_invalid_cfd_conservation_tolerance(config) {
        return Err(PcFanCfdMeshConvergenceConfigError::InvalidConservationTolerance { metric });
    }
    Ok(())
}

pub fn try_solve_pc_fan_cfd_mesh_convergence(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    convergence: PcFanCfdMeshConvergenceConfig,
) -> Result<PcFanCfdMeshConvergence, PcFanCfdMeshConvergenceConfigError> {
    convergence.validate()?;
    validate_mesh_conservation_tolerances(config)?;
    let geometry = config.build_geometry(design);
    let start_cells = convergence.start_cells;
    let max_cells = convergence.max_cells;
    let min_levels = convergence.min_levels;
    let mut levels = Vec::new();
    let mut previous: Option<(PcFanConfig, PcFanCfdFlowSolution)> = None;

    for cells in start_cells..=max_cells {
        let mut level_config = config.clone();
        level_config.cfd.cells_x = cells;
        level_config.cfd.cells_y = cells;
        level_config.cfd.cells_z = cells;
        // Warm-start each level from the previous level's field: the
        // solve is still the full production solve at this mesh, only
        // the initial guess is prolongated.
        let solution = match &previous {
            Some((previous_config, previous_solution)) => {
                let level_domain = geometry.build_cfd_domain(&level_config);
                let previous_domain = geometry.build_cfd_domain(previous_config);
                let source_mesh = pc_fan_cfd_mesh_for_domain(previous_config, &previous_domain);
                let target_mesh = pc_fan_cfd_mesh_for_domain(&level_config, &level_domain);
                match crate::state_transfer::transfer_pc_fan_cfd_state(
                    &source_mesh,
                    &target_mesh,
                    &previous_solution.velocity_m_s,
                    &previous_solution.pressure_pa,
                ) {
                    Ok(transfer) => {
                        let velocity: Vec<f64> = transfer
                            .velocity_m_s
                            .iter()
                            .flat_map(|velocity| velocity.iter().copied())
                            .collect();
                        solve_cfd_domain(
                            &level_config,
                            design,
                            &geometry,
                            &level_domain,
                            Some((&velocity, &transfer.pressure_pa)),
                        )
                    }
                    Err(_) => {
                        solve_pc_fan_cfd_mean_flow_for_geometry(&level_config, design, &geometry)
                    }
                }
            }
            None => solve_pc_fan_cfd_mean_flow_for_geometry(&level_config, design, &geometry),
        };
        previous = Some((level_config, solution.clone()));
        let airflow_relative_change = levels.last().map(|previous: &PcFanCfdMeshLevel| {
            relative_change(solution.airflow_cfm, previous.solution.airflow_cfm)
        });
        let pressure_relative_change = levels.last().map(|previous: &PcFanCfdMeshLevel| {
            relative_change(
                solution.pressure_drop_pa.abs(),
                previous.solution.pressure_drop_pa.abs(),
            )
        });
        let level = PcFanCfdMeshLevel {
            cells_x: cells,
            cells_y: cells,
            cells_z: cells,
            solution,
            airflow_relative_change,
            pressure_relative_change,
        };
        let converged = mesh_level_converged(
            levels.len() + 1,
            min_levels,
            level.solution.converged,
            airflow_relative_change,
            pressure_relative_change,
            convergence.relative_tolerance,
        );
        let residual_admitted =
            mesh_level_residual_admitted(level.solution.residual, convergence.max_final_residual);
        let conservation_admitted =
            mesh_level_conservation_admitted(config, level.solution.conservation);
        levels.push(level);
        if converged && residual_admitted && conservation_admitted {
            break;
        }
    }

    let selected_solution = levels
        .last()
        .expect("validated mesh convergence has at least one level")
        .solution
        .clone();
    let airflow_relative_change = levels
        .last()
        .and_then(|level| level.airflow_relative_change);
    let pressure_relative_change = levels
        .last()
        .and_then(|level| level.pressure_relative_change);
    let all_levels_converged_and_finite = levels.iter().all(|level| {
        level.solution.converged
            && mesh_solution_has_finite_diagnostics(&level.solution)
            && mesh_level_residual_admitted(level.solution.residual, convergence.max_final_residual)
            && mesh_level_conservation_admitted(config, level.solution.conservation)
    });
    let converged = all_levels_converged_and_finite
        && mesh_level_converged(
            levels.len(),
            min_levels,
            levels.last().is_some_and(|level| {
                level.solution.converged
                    && mesh_level_residual_admitted(
                        level.solution.residual,
                        convergence.max_final_residual,
                    )
                    && mesh_level_conservation_admitted(config, level.solution.conservation)
            }),
            airflow_relative_change,
            pressure_relative_change,
            convergence.relative_tolerance,
        );

    Ok(PcFanCfdMeshConvergence {
        levels,
        selected_solution,
        converged,
        all_levels_converged_and_finite,
        airflow_relative_change,
        pressure_relative_change,
    })
}

fn mesh_solution_has_finite_diagnostics(solution: &PcFanCfdFlowSolution) -> bool {
    let conservation = solution.conservation;
    [
        solution.airflow_cfm,
        solution.inlet_airflow_cfm,
        solution.mass_balance_error_kg_s,
        solution.pressure_drop_pa,
        solution.max_velocity_m_s,
        solution.reynolds_number,
        solution.residual,
        conservation.inlet_mass_flow_kg_s,
        conservation.outlet_mass_flow_kg_s,
        conservation.relative_mass_imbalance,
        conservation.inlet_axial_momentum_flux_n,
        conservation.outlet_axial_momentum_flux_n,
        conservation.pressure_force_n,
        conservation.body_force_n,
        conservation.axial_momentum_residual_n,
        conservation.relative_momentum_imbalance,
        conservation.pressure_force_from_loading_n,
        conservation.relative_pressure_closure,
        conservation.pressure_power_w,
        conservation.shaft_power_to_fluid_w,
        conservation.relative_energy_imbalance,
    ]
    .into_iter()
    .all(|value| value.is_finite())
}

fn mesh_level_conservation_admitted(
    config: &PcFanConfig,
    conservation: PcFanCfdConservation,
) -> bool {
    first_conservation_violation(config, conservation).is_none()
}

fn first_conservation_violation(
    config: &PcFanConfig,
    conservation: PcFanCfdConservation,
) -> Option<(&'static str, f64, f64)> {
    [
        (
            "mass imbalance",
            conservation.relative_mass_imbalance,
            config.cfd.max_relative_mass_imbalance,
        ),
        (
            "momentum imbalance",
            conservation.relative_momentum_imbalance,
            config.cfd.max_relative_momentum_imbalance,
        ),
        (
            "pressure closure",
            conservation.relative_pressure_closure,
            config.cfd.max_relative_pressure_closure,
        ),
        (
            "energy imbalance",
            conservation.relative_energy_imbalance,
            config.cfd.max_relative_energy_imbalance,
        ),
    ]
    .into_iter()
    .find(|(_, relative_error, tolerance)| {
        !relative_error.is_finite()
            || *relative_error < 0.0
            || !tolerance.is_finite()
            || *tolerance < 0.0
            || *relative_error > *tolerance
    })
}

fn mesh_level_residual_admitted(residual: f64, max_final_residual: f64) -> bool {
    residual.is_finite()
        && max_final_residual.is_finite()
        && max_final_residual >= 0.0
        && residual <= max_final_residual
}

fn mesh_level_converged(
    level_count: usize,
    min_levels: usize,
    nonlinear_converged: bool,
    airflow_relative_change: Option<f64>,
    pressure_relative_change: Option<f64>,
    relative_tolerance: f64,
) -> bool {
    level_count >= min_levels
        && nonlinear_converged
        && airflow_relative_change.is_some_and(|change| change <= relative_tolerance)
        && pressure_relative_change.is_some_and(|change| change <= relative_tolerance)
}

/// First reason a steady CFD solution fails the production admission
/// contract, or `None` when it is admitted.
///
/// Checks run in solver order: backend admission, nonlinear convergence
/// (distinguishing a failed inner linear solve from Picard stagnation),
/// residual budget, then the conservation ledger. Mesh ladders and dataset
/// exporters report this so a rejected level or sample carries an explicit,
/// actionable reason instead of a bare flag.
pub fn pc_fan_cfd_first_admission_violation(
    config: &PcFanConfig,
    solution: &PcFanCfdFlowSolution,
    max_final_residual: f64,
) -> Option<&'static str> {
    if !solution.backend_provenance.is_admitted(
        config.cfd.allow_host_fallback,
        config.cfd.require_device_resident,
    ) {
        return Some("backend-not-admitted");
    }
    if !solution.converged {
        // The Picard loop only returns early with `converged == false`
        // when an inner linear solve fails; exhausting the iteration
        // budget without meeting tolerance is stagnation.
        if solution.iterations < config.cfd.max_picard_iterations.max(1) {
            return Some("linear-solver-failure");
        }
        return Some("picard-stagnation");
    }
    if !mesh_level_residual_admitted(solution.residual, max_final_residual) {
        return Some("residual-above-budget");
    }
    first_conservation_violation(config, solution.conservation).map(|(metric, _, _)| match metric {
        "mass imbalance" => "conservation-mass-imbalance",
        "momentum imbalance" => "conservation-momentum-imbalance",
        "pressure closure" => "conservation-pressure-closure",
        "energy imbalance" => "conservation-energy-imbalance",
        _ => "conservation-violation",
    })
}

pub(crate) fn solve_pc_fan_cfd_mean_flow_for_geometry(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    geometry: &PcFanGeometry,
) -> PcFanCfdFlowSolution {
    let domain = geometry.build_cfd_domain(config);
    solve_cfd_domain(config, design, geometry, &domain, None)
}

/// Restart the production P1 mean-flow solve from a state on a different
/// generated tetrahedral topology.
///
/// The source state is transferred before the target runner applies its
/// topology-preserving ALE motion. This is the checked production handoff for
/// a future refinement/remeshing loop: the transfer itself is transactional,
/// while the returned solution retains the runner's normal convergence and
/// conservation admission diagnostics. Higher-order, turbulence-history, and
/// transient BDF state transfer remain separate contracts.
pub fn solve_pc_fan_cfd_mean_flow_with_state_transfer(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    source_mesh: &sonium_fem::mesh::Mesh,
    source_velocity_m_s: &[[f64; 3]],
    source_pressure_pa: &[f64],
) -> Result<PcFanCfdFlowSolution, PcFanCfdStateTransferError> {
    let geometry = config.build_geometry(design);
    let domain = geometry.build_cfd_domain(config);
    let target_mesh = pc_fan_cfd_mesh_for_domain(config, &domain);
    let transfer = crate::state_transfer::transfer_pc_fan_cfd_state(
        source_mesh,
        &target_mesh,
        source_velocity_m_s,
        source_pressure_pa,
    )?;
    let target_velocity = transfer
        .velocity_m_s
        .iter()
        .flat_map(|velocity| velocity.iter().copied())
        .collect::<Vec<_>>();

    Ok(solve_cfd_domain(
        config,
        design,
        &geometry,
        &domain,
        Some((&target_velocity, &transfer.pressure_pa)),
    ))
}

fn rejected_mesh_motion_solution(
    domain: &PcFanCfdDomain,
    config: &PcFanCfdConfig,
) -> PcFanCfdFlowSolution {
    let node_count = domain.nodes.len();
    let backend_provenance =
        PcFanCfdBackendProvenance::from_dispatch(config.backend, PcFanCfdBackend::Cpu);
    let execution_environment = PcFanCfdExecutionEnvironment::capture(config, backend_provenance);
    PcFanCfdFlowSolution {
        domain: domain.clone(),
        actual_backend: PcFanCfdBackend::Cpu,
        backend_provenance,
        execution_environment,
        nodes: domain.nodes.clone(),
        velocity_m_s: vec![[0.0; 3]; node_count],
        pressure_pa: vec![0.0; node_count],
        airflow_cfm: 0.0,
        inlet_airflow_cfm: 0.0,
        mass_balance_error_kg_s: 0.0,
        pressure_drop_pa: 0.0,
        max_velocity_m_s: 0.0,
        reynolds_number: 0.0,
        fem_node_count: node_count,
        fem_element_count: domain.cells.len(),
        boundary_dof_count: 0,
        actuator_loading: None,
        rotating_wall_loading: None,
        conservation: PcFanCfdConservation {
            inlet_mass_flow_kg_s: 0.0,
            outlet_mass_flow_kg_s: 0.0,
            relative_mass_imbalance: 0.0,
            inlet_axial_momentum_flux_n: 0.0,
            outlet_axial_momentum_flux_n: 0.0,
            pressure_force_n: 0.0,
            body_force_n: 0.0,
            axial_momentum_residual_n: 0.0,
            relative_momentum_imbalance: 0.0,
            pressure_force_from_loading_n: 0.0,
            relative_pressure_closure: 0.0,
            pressure_power_w: 0.0,
            shaft_power_to_fluid_w: 0.0,
            relative_energy_imbalance: 0.0,
        },
        picard_linearization: None,
        newton_linearization: None,
        residual_design_jacobian: None,
        iterations: 0,
        residual: f64::INFINITY,
        converged: false,
    }
}

fn pc_fan_cfd_mesh_for_domain(
    config: &PcFanConfig,
    domain: &PcFanCfdDomain,
) -> sonium_fem::mesh::Mesh {
    let nx = config.cfd.cells_x.max(1);
    let ny = config.cfd.cells_y.max(1);
    let nz = config.cfd.cells_z.max(1);
    match config.cfd.mesh_pattern {
        crate::model::PcFanCfdMeshPattern::Symmetric => box_mesh_tetrahedra_symmetric(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            nx,
            ny,
            nz,
        ),
        crate::model::PcFanCfdMeshPattern::Kuhn => box_mesh_tetrahedra(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            nx,
            ny,
            nz,
        ),
    }
}

fn build_pc_fan_cfd_body_fitted_mesh(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    level: PcFanCfdBodyFittedRemeshLevel,
) -> Result<sonium_fem::mesh::Mesh, String> {
    let mut level_config = config.clone();
    level_config.cfd.cells_x = level.cells_x;
    level_config.cfd.cells_y = level.cells_y;
    level_config.cfd.cells_z = level.cells_z;
    level_config
        .validate_geometry_config()
        .map_err(|_| "invalid PC-fan geometry for built-in remesh".to_owned())?;

    let geometry = level_config.build_geometry(design);
    let domain = geometry.build_cfd_domain(&level_config);
    let mut mesh = pc_fan_cfd_mesh_for_domain(&level_config, &domain);
    apply_pc_fan_mesh_motion(&mut mesh, &level_config, &geometry, design)
        .map_err(|error| error.to_string())?;
    Ok(mesh)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SteadySolvePurpose {
    Production,
    /// A bounded loose continuation stage: branch-tracking scaffolding
    /// whose state is carried onward whether or not it converged. Its
    /// Newton corrections use the cheap globalization (full step plus
    /// halving); the mass-shifted and Levenberg escalations are left
    /// to production solves, where their factorization cost is spent
    /// on a state that must actually converge.
    LooseStage,
    PicardMapProbe,
    NewtonStateProbe,
}

fn solve_cfd_domain(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    geometry: &PcFanGeometry,
    domain: &PcFanCfdDomain,
    initial_state: Option<(&[f64], &[f64])>,
) -> PcFanCfdFlowSolution {
    let target_cells = [
        config.cfd.cells_x.max(1),
        config.cfd.cells_y.max(1),
        config.cfd.cells_z.max(1),
    ];
    // Load continuation: at rest the exact tangent has no inertial
    // stiffness (the convection derivative vanishes), so a cold Newton
    // correction overshoots by orders of magnitude at the blade tip.
    // Production solves therefore continue in rotor speed and mesh size;
    // only the final stage runs the caller's exact configuration and
    // publishes derivatives. Small meshes and short-budget probes keep
    // the direct single-attempt path.
    let continuation = initial_state.is_none()
        && config.cfd.max_picard_iterations > 2
        && !target_cells.iter().all(|&cells| cells <= 2)
        && config.operating_point.rpm.abs() > f64::EPSILON;
    if !continuation {
        return solve_cfd_domain_with_derivative_policy(
            config,
            design,
            geometry,
            domain,
            initial_state,
            true,
            SteadySolvePurpose::Production,
        );
    }
    // Mesh continuation first: prolongation through unit mesh
    // increments raises the effective Reynolds number more gently
    // than a viscosity ladder on the target mesh, and the coarse
    // levels are cheap. It is only attempted when the target is
    // within the range whose intermediate levels converge: on finer
    // targets the unit-increment levels stall in the flat valley
    // (observed at the fourth level) and the attempt would only
    // consume the budget the parameter continuations need.
    let target_max = target_cells.iter().copied().max().unwrap_or(1);
    let mesh_continuation_attempt = target_max <= 4;
    // Coarse-mesh parameter traversal: the (viscosity, speed)
    // continuation that tracks the branch around the fold costs
    // about the same number of iterations on any mesh, so run its
    // descent where iterations are cheap, prolongate the converged
    // coarse field, and finish the ramp and production solve on the
    // target mesh. A failed finish falls through to the
    // target-mesh continuations below.
    if target_max >= 4
        && let Some(solution) =
            solve_cfd_domain_coarse_traversal(config, design, geometry, domain, target_cells)
    {
        return solution;
    }
    if mesh_continuation_attempt
        && std::env::var("SONIUM_PCFAN_NO_MESH_CONT").is_err()
        && let Some(solution) =
            solve_cfd_domain_mesh_continuation(config, design, geometry, target_cells)
    {
        return solution;
    }
    // Two-parameter continuation: descend viscosity and rotor speed
    // jointly around the fold that blocks each single-parameter path,
    // then ramp the speed back up at physical viscosity. This runs
    // before the strict viscosity continuation, whose bisection
    // around the fold would otherwise consume the caller's budget.
    if std::env::var("SONIUM_PCFAN_NO_TWO_PARAM_CONT").is_err()
        && let Some(solution) =
            solve_cfd_domain_two_parameter_continuation(config, design, geometry, domain)
    {
        return solution;
    }
    // Viscosity continuation: at a hundredfold viscosity the
    // system is diffusion-dominated and Newton converges from any
    // start; warm-started stages then track the solution branch down
    // to the physical viscosity. The target-mesh speed ramp remains
    // as the last fallback.
    if std::env::var("SONIUM_PCFAN_NO_VISC_CONT").is_err()
        && let Some(solution) =
            solve_cfd_domain_viscosity_continuation(config, design, geometry, domain, target_cells)
    {
        return solution;
    }
    // Mesh continuation: ramp the rotor speed on a coarse mesh
    // (cheap), then prolongate through warm-started solves up to the
    // target mesh at full speed. Only a converged target-mesh solution
    // is accepted from this path; otherwise the target-mesh speed ramp
    // below remains the fallback.
    if mesh_continuation_attempt
        && std::env::var("SONIUM_PCFAN_NO_MESH_CONT").is_err()
        && let Some(solution) =
            solve_cfd_domain_mesh_continuation(config, design, geometry, target_cells)
    {
        return solution;
    }
    solve_cfd_domain_rpm_ladder(config, design, geometry, domain)
}

/// State carried between continuation stages: flattened velocity and
/// pressure on the stage mesh.
type ContinuationState = (Vec<f64>, Vec<f64>);

/// A converged continuation point: (viscosity scale, speed fraction)
/// parameters with the state converged there.
type ContinuationPoint = ((f64, f64), ContinuationState);

/// Branch-track on a coarse mesh, then finish on the target mesh.
///
/// The two-parameter continuation below costs about the same number
/// of iterations on any mesh, but its iterations are far cheaper on
/// a coarse mesh. Run the viscosity/speed descent and the lower
/// speed ramp there, prolongate the transfer-point state onto the
/// target mesh, then run the remaining speed ramp and the production
/// finish on the target mesh so the discretization-sensitive part of
/// the path still tracks the target mesh's own branch. Returns the
/// target-mesh solution only when the production finish converges;
/// otherwise the caller's target-mesh continuations remain as the
/// fallback.
fn solve_cfd_domain_coarse_traversal(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    geometry: &PcFanGeometry,
    domain: &PcFanCfdDomain,
    target_cells: [usize; 3],
) -> Option<PcFanCfdFlowSolution> {
    const COARSE_MAX: usize = 3;
    // Bound the production finish like the mesh-continuation target
    // level: a prolongated state separated from the target-mesh
    // branch by the flat valley must not consume the caller's whole
    // budget before the fallbacks run.
    const WARM_FINISH_ITERATION_CAP: usize = 120;
    let target_max = target_cells.iter().copied().max().unwrap_or(1);
    let mut coarse_config = config.clone();
    for (axis, target) in [
        &mut coarse_config.cfd.cells_x,
        &mut coarse_config.cfd.cells_y,
        &mut coarse_config.cfd.cells_z,
    ]
    .into_iter()
    .zip(target_cells.iter())
    {
        *axis = (*target * COARSE_MAX / target_max).max(1);
    }
    let coarse_domain = geometry.build_cfd_domain(&coarse_config);
    // The coarse mesh runs the descent down to the (3x viscosity,
    // 3.5% speed) stage. There the coarse and target fields still
    // agree on the through-flow (within ~2x) and the problem is
    // diffusion-dominated, so the target mesh's solution is the
    // unique continuation of the transferred field: transferring
    // further down, where the coarse through-flow collapses ~20x
    // below the target's, hands the fine mesh a field that settles
    // into the reversed-flow basin (the converged target branch
    // then pumps backward, which the mass-balance gate rejects).
    // The target mesh re-solves the transfer stage itself, then
    // runs the remaining descent and the speed ramp on its own
    // branch.
    let split = 12;
    let descent = &TWO_PARAMETER_STAGES[..split];
    let ramp = &TWO_PARAMETER_STAGES[split..];
    let anchor_parameters = descent
        .last()
        .expect("descent stage sequence must be non-empty")
        .to_owned();
    let coarse_state =
        run_two_parameter_stages(&coarse_config, design, geometry, &coarse_domain, descent, None);
    let source_mesh = pc_fan_cfd_mesh_for_domain(&coarse_config, &coarse_domain);
    let target_mesh = pc_fan_cfd_mesh_for_domain(config, domain);
    // The carried state is a flat per-node velocity field; regroup
    // it into nodal vectors for the state transfer.
    let coarse_velocity: Vec<[f64; 3]> =
        coarse_state.0.as_chunks::<3>().0.to_vec();
    let transfer = crate::state_transfer::transfer_pc_fan_cfd_state(
        &source_mesh,
        &target_mesh,
        &coarse_velocity,
        &coarse_state.1,
    )
    .ok()?;
    let velocity: Vec<f64> = transfer
        .velocity_m_s
        .iter()
        .flat_map(|velocity| velocity.iter().copied())
        .collect();
    stage_trace(format!(
        "coarse traversal prolongated {}x{}x{} -> {}x{}x{}",
        coarse_config.cfd.cells_x,
        coarse_config.cfd.cells_y,
        coarse_config.cfd.cells_z,
        config.cfd.cells_x,
        config.cfd.cells_y,
        config.cfd.cells_z
    ));
    let anchored_state = run_two_parameter_stages(
        config,
        design,
        geometry,
        domain,
        &[anchor_parameters],
        Some((anchor_parameters, (velocity, transfer.pressure_pa))),
    );
    let anchor = (anchor_parameters, anchored_state);
    let state = run_two_parameter_stages(config, design, geometry, domain, ramp, Some(anchor));
    let mut finish_config = config.clone();
    finish_config.cfd.max_picard_iterations =
        finish_config.cfd.max_picard_iterations.min(WARM_FINISH_ITERATION_CAP);
    let solution = solve_cfd_domain_with_derivative_policy(
        &finish_config,
        design,
        geometry,
        domain,
        Some((state.0.as_slice(), state.1.as_slice())),
        true,
        SteadySolvePurpose::Production,
    );
    // A converged solution on the reversed-flow branch is not the
    // fan's operating point: reject it so the caller's remaining
    // continuations can still find the forward branch.
    (solution.converged && solution.conservation.inlet_mass_flow_kg_s > 0.0).then_some(solution)
}

/// Two-parameter (viscosity, rotor-speed) continuation on one mesh.
///
/// On refined meshes the steady branch folds in each single
/// parameter: at full speed it folds near thirteenfold viscosity, and
/// the fold scale drops roughly in proportion to rotor speed, so
/// descending viscosity while backing the speed off tracks the branch
/// around the fold to physical viscosity at a few percent of full
/// speed. Intermediate stages are loose (bounded iterations, caller
/// tolerance relaxed to `1e-3`) and the chain carries each stage's
/// state regardless of convergence, like the speed ladder: near-fold
/// patches only need a state close enough for the next stage, and the
/// final stage runs the caller's exact configuration and must
/// converge for the path to be accepted.
fn solve_cfd_domain_two_parameter_continuation(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    geometry: &PcFanGeometry,
    domain: &PcFanCfdDomain,
) -> Option<PcFanCfdFlowSolution> {
    let state = run_two_parameter_stages(config, design, geometry, domain, TWO_PARAMETER_STAGES, None);
    let solution = solve_cfd_domain_with_derivative_policy(
        config,
        design,
        geometry,
        domain,
        Some((state.0.as_slice(), state.1.as_slice())),
        true,
        SteadySolvePurpose::Production,
    );
    solution.converged.then_some(solution)
}

/// (viscosity scale, rotor-speed fraction) stages: a short
/// viscosity descent at full speed, then a lockstep joint descent
/// that holds the viscosity/speed ratio in the band that stays on
/// the easy side of the fold (descending viscosity to near-physical
/// values at full speed stalls on fine meshes, whose fold sits at
/// higher viscosity multiples), then the speed ramp at physical
/// viscosity. The descent ends at physical viscosity and the lowest
/// speed (index 15); the remaining stages are the speed ramp. The upper ramp
/// keeps each speed ratio at or below about 1.3, with an extra 0.31
/// step where the 8-cell branch is tightest (0.28 -> 0.36 otherwise
/// falls off and burns the full loose-stage budget plus bisection
/// retries, which cost far more than the extra converged stages).
const TWO_PARAMETER_STAGES: &[(f64, f64)] = &[
    (512.0, 1.0),
    (256.0, 1.0),
    (128.0, 1.0),
    (64.0, 1.0),
    (48.0, 0.65),
    (32.0, 0.45),
    (24.0, 0.3),
    (16.0, 0.2),
    (12.0, 0.14),
    (8.0, 0.1),
    (6.0, 0.07),
    (4.0, 0.05),
    (3.0, 0.035),
    (2.0, 0.025),
    (1.5, 0.018),
    (1.0, 0.015),
    (1.0, 0.0225),
    (1.0, 0.033),
    (1.0, 0.0415),
    (1.0, 0.05),
    (1.0, 0.075),
    (1.0, 0.11),
    (1.0, 0.17),
    (1.0, 0.22),
    (1.0, 0.28),
    (1.0, 0.31),
    (1.0, 0.36),
    (1.0, 0.46),
    (1.0, 0.59),
    (1.0, 0.75),
    (1.0, 0.97),
];

/// Run a sequence of two-parameter loose stages, warm-chaining the
/// state and bisecting failed steps toward the last converged
/// stage. `initial` is the last converged point before the sequence
/// (its parameters anchor the first bisection when a stage fails).
/// Returns the state carried out of the last stage.
fn run_two_parameter_stages(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    geometry: &PcFanGeometry,
    domain: &PcFanCfdDomain,
    stages: &[(f64, f64)],
    initial: Option<ContinuationPoint>,
) -> ContinuationState {
    // Loose stages only need to track the branch: they run with a
    // bounded iteration budget and carry their state onward either
    // way. When a stage does not converge, the parameter step is
    // bisected toward the last converged stage, restarting from that
    // stage's state (a failed attempt must not poison the retry), so
    // a mesh whose branch folds earlier (finer meshes fold at higher
    // viscosity multiples) inserts the intermediate points it needs
    // instead of crawling off-branch for the full budget.
    // Stage budget: the lowest-speed stage (physical viscosity,
    // 1.5% speed) crawls through the flat valley and reaches a
    // relative residual of 8.6e-3 at iteration 24, converging a few
    // iterations later; a 24 cap cuts it off and the bisection
    // retries then cost far more than the extra iterations.
    const LOOSE_STAGE_ITERATION_CAP: usize = 32;
    const MAX_STAGE_BISECTIONS: usize = 3;
    let mut state: Option<ContinuationState> =
        initial.as_ref().map(|(_, state)| state.clone());
    let mut last_converged: Option<ContinuationPoint> = initial;
    for &(target_viscosity, target_rpm) in stages {
        // One bounded loose attempt at the given parameters, started
        // from `initial`; returns the convergence flag and the state
        // to carry onward.
        let run_loose_stage = |viscosity_scale: f64,
                               rpm_fraction: f64,
                               initial: Option<&(Vec<f64>, Vec<f64>)>,
                               iteration_cap: usize|
         -> (bool, (Vec<f64>, Vec<f64>)) {
            stage_trace(format!(
                "two-param viscosity x{viscosity_scale} rpm x{rpm_fraction} cells={}x{}x{}",
                config.cfd.cells_x, config.cfd.cells_y, config.cfd.cells_z
            ));
            let mut stage_config = config.clone();
            stage_config.operating_point.dynamic_viscosity_pa_s *= viscosity_scale;
            stage_config.operating_point.rpm = config.operating_point.rpm * rpm_fraction;
            stage_config.cfd.max_picard_iterations =
                stage_config.cfd.max_picard_iterations.min(iteration_cap);
            // Branch tracking does not need the last decade of the
            // exact residual: at 1e-4 the flat-valley stages spend
            // 15-19 iterations per stage, while 1e-3 keeps the warm
            // start on-branch for the next stage. The production
            // finish still enforces the caller's exact tolerance.
            stage_config.cfd.nonlinear_tolerance = stage_config.cfd.nonlinear_tolerance.max(1.0e-3);
            let stage = solve_cfd_domain_with_derivative_policy(
                &stage_config,
                design,
                geometry,
                domain,
                initial.map(|(velocity, pressure)| (velocity.as_slice(), pressure.as_slice())),
                false,
                SteadySolvePurpose::LooseStage,
            );
            let stage_state = (
                stage
                    .velocity_m_s
                    .iter()
                    .flat_map(|velocity| velocity.iter().copied())
                    .collect(),
                stage.pressure_pa.clone(),
            );
            stage_trace(format!(
                "two-param viscosity x{viscosity_scale} rpm x{rpm_fraction} converged={} iterations={} inlet_kg_s={:.6e}",
                stage.converged, stage.iterations, stage.conservation.inlet_mass_flow_kg_s
            ));
            (stage.converged, stage_state)
        };
        let (converged, stage_state) = run_loose_stage(
            target_viscosity,
            target_rpm,
            state.as_ref(),
            LOOSE_STAGE_ITERATION_CAP,
        );
        state = Some(stage_state);
        if converged {
            last_converged = Some((
                (target_viscosity, target_rpm),
                state.clone().expect("stage state recorded"),
            ));
            continue;
        }
        // Step too large for this mesh: bisect from the last converged
        // stage, restoring its state for every retry. A failed
        // midpoint shrinks the pending step and bisects again instead
        // of giving up, so the sequence keeps inserting the
        // intermediate points this mesh needs rather than jumping to
        // the next, larger stage from a state that is already too far
        // off the branch.
        if let Some(((mut low_viscosity, mut low_rpm), mut low_state)) = last_converged.clone() {
            let mut pending_viscosity = target_viscosity;
            let mut pending_rpm = target_rpm;
            for _ in 0..MAX_STAGE_BISECTIONS {
                let mid_viscosity = (low_viscosity * pending_viscosity).sqrt();
                let mid_rpm = 0.5 * (low_rpm + pending_rpm);
                let (mid_converged, mid_state) =
                    run_loose_stage(mid_viscosity, mid_rpm, Some(&low_state), LOOSE_STAGE_ITERATION_CAP);
                if !mid_converged {
                    pending_viscosity = mid_viscosity;
                    pending_rpm = mid_rpm;
                    state = Some(low_state.clone());
                    continue;
                }
                last_converged = Some(((mid_viscosity, mid_rpm), mid_state.clone()));
                low_viscosity = mid_viscosity;
                low_rpm = mid_rpm;
                low_state = mid_state;
                let (target_converged, target_state) =
                    run_loose_stage(target_viscosity, target_rpm, Some(&low_state), LOOSE_STAGE_ITERATION_CAP);
                state = Some(target_state);
                if target_converged {
                    last_converged = Some((
                        (target_viscosity, target_rpm),
                        state.clone().expect("stage state recorded"),
                    ));
                    break;
                }
                // Keep the on-branch midpoint state, not the failed
                // target attempt's state, for the next stage.
                state = Some(low_state.clone());
                pending_viscosity = target_viscosity;
                pending_rpm = target_rpm;
            }
        }
    }
    state.expect("two-parameter stage sequence must be non-empty")
}

/// Temporary diagnostic: continuation stage tracing.
fn stage_trace(message: String) {
    if std::env::var("SONIUM_NS_PROFILE").as_deref() == Ok("1") {
        eprintln!("STAGE {message}");
    }
}

/// Rotor-speed continuation on one mesh: warm-started stages at
/// increasing fractions of the configured speed. Only the final stage
/// runs the caller's exact configuration and publishes derivatives.
fn solve_cfd_domain_rpm_ladder(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    geometry: &PcFanGeometry,
    domain: &PcFanCfdDomain,
) -> PcFanCfdFlowSolution {
    let mut state: Option<(Vec<f64>, Vec<f64>)> = None;
    let mut solution = None;
    // Fine ladder at the bottom: the first stages are nearly linear, and
    // each warm start keeps the exact tangent's inertial stiffness
    // representative of the developed swirl.
    for fraction in [
        1.0 / 600.0,
        1.0 / 300.0,
        1.0 / 150.0,
        1.0 / 60.0,
        1.0 / 30.0,
        1.0 / 12.0,
        1.0 / 6.0,
        1.0 / 3.0,
        1.0,
    ] {
        stage_trace(format!(
            "rpm ladder fraction={fraction:.5} cells={}x{}x{}",
            config.cfd.cells_x, config.cfd.cells_y, config.cfd.cells_z
        ));
        let mut stage_config = config.clone();
        stage_config.operating_point.rpm = config.operating_point.rpm * fraction;
        if fraction < 1.0 {
            stage_config.cfd.max_picard_iterations =
                stage_config.cfd.max_picard_iterations.min(48);
            stage_config.cfd.nonlinear_tolerance =
                stage_config.cfd.nonlinear_tolerance.max(1.0e-4);
        }
        let stage = solve_cfd_domain_with_derivative_policy(
            &stage_config,
            design,
            geometry,
            domain,
            state
                .as_ref()
                .map(|(velocity, pressure)| (velocity.as_slice(), pressure.as_slice())),
            fraction >= 1.0,
            if fraction >= 1.0 {
                SteadySolvePurpose::Production
            } else {
                SteadySolvePurpose::LooseStage
            },
        );
        state = Some((
            stage
                .velocity_m_s
                .iter()
                .flat_map(|velocity| velocity.iter().copied())
                .collect(),
            stage.pressure_pa.clone(),
        ));
        solution = Some(stage);
    }
    solution.expect("continuation ladder has a final stage")
}

/// Mesh continuation: ramp the rotor speed on the coarsest mesh, then
/// prolongate the state through warm-started full-speed solves up to
/// the target mesh. Returns the target-mesh solution only when it
/// converged; returns `None` (caller falls back to the target-mesh
/// speed ramp) when the mesh ladder is degenerate or a state transfer
/// fails.
fn solve_cfd_domain_mesh_continuation(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    geometry: &PcFanGeometry,
    target_cells: [usize; 3],
) -> Option<PcFanCfdFlowSolution> {
    let target_max = target_cells.iter().copied().max().unwrap_or(1);
    // Unit mesh increments: each prolongation is a mild perturbation,
    // so every warm-started solve stays inside the Newton basin (the
    // coarse-to-target jump does not).
    let mut level_maxes: Vec<usize> = (2..target_max).collect();
    level_maxes.push(target_max);
    if level_maxes.len() < 2 {
        return None;
    }
    let mut previous: Option<(PcFanConfig, PcFanCfdFlowSolution)> = None;
    let mut solution = None;
    for &level_max in &level_maxes {
        let mut level_config = config.clone();
        for (axis, target) in [
            &mut level_config.cfd.cells_x,
            &mut level_config.cfd.cells_y,
            &mut level_config.cfd.cells_z,
        ]
        .into_iter()
        .zip(target_cells.iter())
        {
            *axis = (*target * level_max / target_max).max(1);
        }
        let level_domain = geometry.build_cfd_domain(&level_config);
        let is_target = level_max == target_max;
        if !is_target {
            // Intermediate levels are the continuation path itself:
            // each must converge to the caller's tolerance so the next
            // level prolongates from an on-branch state (a loosely
            // converged intermediate returns a state about as far from
            // the fine-mesh branch as a cold start). Their budget stays
            // bounded so a struggling level cannot consume the caller's
            // whole solve allowance.
            level_config.cfd.max_picard_iterations =
                level_config.cfd.max_picard_iterations.min(200);
            level_config.cfd.nonlinear_tolerance = config.cfd.nonlinear_tolerance;
        } else {
            // Bound the target attempt as well: when the prolongated
            // state is separated from the target-mesh branch by the
            // flat valley, the direct solve crawls for thousands of
            // iterations, and the caller's remaining fallbacks (the
            // two-parameter continuation in particular) must get their
            // turn within the caller's budget.
            level_config.cfd.max_picard_iterations =
                level_config.cfd.max_picard_iterations.min(120);
        }
        stage_trace(format!(
            "mesh level_max={level_max} cells={}x{}x{} target={is_target}",
            level_config.cfd.cells_x, level_config.cfd.cells_y, level_config.cfd.cells_z
        ));
        let level_solution = match &previous {
            None => {
                // The coarsest mesh solves cold at full speed (its
                // numerical diffusion keeps the discrete problem
                // tame); the speed ramp is only the fallback.
                let direct = solve_cfd_domain_with_derivative_policy(
                    &level_config,
                    design,
                    geometry,
                    &level_domain,
                    None,
                    false,
                    SteadySolvePurpose::LooseStage,
                );
                if direct.converged {
                    direct
                } else {
                    solve_cfd_domain_rpm_ladder(&level_config, design, geometry, &level_domain)
                }
            }
            Some((previous_config, previous_solution)) => {
                let previous_domain = geometry.build_cfd_domain(previous_config);
                let source_mesh = pc_fan_cfd_mesh_for_domain(previous_config, &previous_domain);
                let target_mesh = pc_fan_cfd_mesh_for_domain(&level_config, &level_domain);
                let transfer = crate::state_transfer::transfer_pc_fan_cfd_state(
                    &source_mesh,
                    &target_mesh,
                    &previous_solution.velocity_m_s,
                    &previous_solution.pressure_pa,
                )
                .ok()?;
                let velocity: Vec<f64> = transfer
                    .velocity_m_s
                    .iter()
                    .flat_map(|velocity| velocity.iter().copied())
                    .collect();
                solve_cfd_domain_with_derivative_policy(
                    &level_config,
                    design,
                    geometry,
                    &level_domain,
                    Some((&velocity, &transfer.pressure_pa)),
                    is_target,
                    if is_target {
                        SteadySolvePurpose::Production
                    } else {
                        SteadySolvePurpose::LooseStage
                    },
                )
            }
        };
        if !level_solution.converged {
            // An unconverged level prolongates a state about as far
            // from the next level's branch as a cold start, so the
            // remaining levels would only burn the caller's budget.
            // Fail fast and let the caller's next fallback run.
            return None;
        }
        previous = Some((level_config, level_solution.clone()));
        solution = Some(level_solution);
    }
    let solution = solution?;
    solution.converged.then_some(solution)
}

/// Viscosity continuation: solve at a highly multiplied dynamic
/// viscosity (diffusion-dominated, Newton converges from a cold start
/// through the mesh-continuation path), then warm-start a ladder of
/// solves at decreasing viscosity on the target mesh, tracking the
/// solution branch to the physical viscosity. A stage that fails to
/// converge is retried at the geometric mean of the last accepted and
/// attempted scales, so the warm start always restarts from a
/// converged state; giving up returns `None` and lets the caller's
/// next fallback run instead of drifting along an unconverged chain.
/// Returns the physical-viscosity solution only when it converged.
fn solve_cfd_domain_viscosity_continuation(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    geometry: &PcFanGeometry,
    domain: &PcFanCfdDomain,
    target_cells: [usize; 3],
) -> Option<PcFanCfdFlowSolution> {
    // The descent starts at 512x: on the target meshes of interest the
    // tangent's near-null mode only becomes benign at several-hundredfold
    // viscosity, and the high stages then converge in a few iterations.
    // Intermediate stages solve to the caller's tolerance (a loosely
    // converged stage prolongates off-branch about as poorly as a cold
    // start), with a bounded per-stage budget.
    const MAX_STAGE_ATTEMPTS: usize = 40;
    let mut state: Option<(Vec<f64>, Vec<f64>)> = None;
    let mut accepted_scale = 512.0;
    let mut attempt_scale = 512.0;
    let mut attempts = 0usize;
    loop {
        attempts += 1;
        if attempts > MAX_STAGE_ATTEMPTS {
            return None;
        }
        let viscosity_scale = attempt_scale;
        let is_physical = viscosity_scale <= 1.0;
        stage_trace(format!("viscosity x{viscosity_scale}"));
        let mut stage_config = config.clone();
        stage_config.operating_point.dynamic_viscosity_pa_s *= viscosity_scale;
        if !is_physical {
            stage_config.cfd.max_picard_iterations =
                stage_config.cfd.max_picard_iterations.min(200);
            stage_config.cfd.nonlinear_tolerance = config.cfd.nonlinear_tolerance;
        }
        let stage_solution = if state.is_none() {
            // At the starting viscosity the system is diffusion-
            // dominated and the exact Newton iteration converges from a
            // cold start on the target mesh; only pay for the mesh-
            // continuation ramp when that direct attempt fails.
            let direct = solve_cfd_domain_with_derivative_policy(
                &stage_config,
                design,
                geometry,
                domain,
                None,
                false,
                SteadySolvePurpose::LooseStage,
            );
            if direct.converged {
                direct
            } else {
                match solve_cfd_domain_mesh_continuation(
                    &stage_config,
                    design,
                    geometry,
                    target_cells,
                ) {
                    Some(solution) => solution,
                    None => solve_cfd_domain_rpm_ladder(&stage_config, design, geometry, domain),
                }
            }
        } else {
            solve_cfd_domain_with_derivative_policy(
                &stage_config,
                design,
                geometry,
                domain,
                state
                    .as_ref()
                    .map(|(velocity, pressure)| (velocity.as_slice(), pressure.as_slice())),
                is_physical,
                if is_physical {
                    SteadySolvePurpose::Production
                } else {
                    SteadySolvePurpose::LooseStage
                },
            )
        };
        if stage_solution.converged {
            state = Some((
                stage_solution
                    .velocity_m_s
                    .iter()
                    .flat_map(|velocity| velocity.iter().copied())
                    .collect(),
                stage_solution.pressure_pa.clone(),
            ));
            accepted_scale = attempt_scale;
            if is_physical {
                return Some(stage_solution);
            }
            attempt_scale = (attempt_scale * 0.5).max(1.0);
        } else {
            // Bisect in log space between the last accepted scale and
            // the failed attempt, then retry from the accepted state.
            // Before the first acceptance, bisect upward instead: a
            // still more viscous cold start is strictly easier.
            let anchor = if state.is_some() {
                accepted_scale
            } else {
                attempt_scale * 2.0
            };
            let next_scale = (anchor * attempt_scale).sqrt();
            if !next_scale.is_finite() || next_scale <= 1.0 || next_scale >= attempt_scale * 2.0 {
                return None;
            }
            if (next_scale - attempt_scale).abs() <= 1.0e-12 * attempt_scale {
                return None;
            }
            attempt_scale = next_scale;
        }
    }
}

fn solve_cfd_domain_with_derivative_policy(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    geometry: &PcFanGeometry,
    domain: &PcFanCfdDomain,
    initial_state: Option<(&[f64], &[f64])>,
    publish_design_derivatives: bool,
    purpose: SteadySolvePurpose,
) -> PcFanCfdFlowSolution {
    let nx = config.cfd.cells_x.max(1);
    let ny = config.cfd.cells_y.max(1);
    let nz = config.cfd.cells_z.max(1);
    let mut mesh = match config.cfd.mesh_pattern {
        crate::model::PcFanCfdMeshPattern::Symmetric => box_mesh_tetrahedra_symmetric(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            nx,
            ny,
            nz,
        ),
        crate::model::PcFanCfdMeshPattern::Kuhn => box_mesh_tetrahedra(
            domain.min_corner.x,
            domain.max_corner.x,
            domain.min_corner.y,
            domain.max_corner.y,
            domain.min_corner.z,
            domain.max_corner.z,
            nx,
            ny,
            nz,
        ),
    };
    let mesh_motion_design_jacobian = if publish_design_derivatives {
        pc_fan_mesh_motion_design_jacobian(&mesh, config, design).ok()
    } else {
        None
    };
    if apply_pc_fan_mesh_motion(&mut mesh, config, geometry, design).is_err() {
        return rejected_mesh_motion_solution(domain, &config.cfd);
    }

    let v_space = VectorH1Space::new(PolynomialDegree::P1, 3);
    let p_space = H1Space::new(PolynomialDegree::P1);
    let boundary_tolerance = boundary_tolerance(domain, nx, ny, nz);
    let solid_boundary_conditions = SolidBoundaryConditions::from_domain(
        config,
        domain,
        solid_boundary_influence_m(domain, nx, ny, nz),
    );
    let bc_dofs = locate_dofs_geometrical(&mesh, &v_space, |x, y, _z| {
        is_side_wall(domain, x, y, boundary_tolerance)
    });
    let mut ns_config = NavierStokesConfig {
        mu: config.operating_point.dynamic_viscosity_pa_s,
        rho: config.operating_point.air_density_kg_m3,
        max_picard_iterations: config.cfd.max_picard_iterations.max(1),
        min_picard_iterations: 20.min(config.cfd.max_picard_iterations.max(1)),
        picard_tolerance: config.cfd.nonlinear_tolerance,
        picard_relaxation: fan_picard_relaxation(if purpose == SteadySolvePurpose::LooseStage {
            LOOSE_STAGE_PICARD_RELAXATION
        } else {
            FAN_PICARD_RELAXATION
        }),
        linear_solver_max_iterations: config.cfd.linear_solver_max_iterations.max(1),
        linear_solver_tolerance: config.cfd.linear_solver_tolerance,
        linear_solver_cpu_preconditioner: config.cfd.cpu_preconditioner.into_fem(),
        use_direct_solver: config.cfd.use_direct_solver,
        turbulence: config.cfd.turbulence.into_fem(),
        linear_solver_backend: config.cfd.backend.into_fem(),
        linear_solver_backend_preconditioner: config.cfd.precond_kind.into_fem(),
        linear_solver_polynomial_degree: config.cfd.precond_polynomial_degree.max(1),
        linear_solver_polynomial_damping: config.cfd.precond_polynomial_damping,
        linear_solver_schur_polynomial_degree: config.cfd.precond_schur_degree.max(1),
        linear_solver_schur_polynomial_damping: config.cfd.precond_schur_damping,
        linear_solver_schur_preconditioner: config.cfd.precond_schur_kind.into_fem(),
        linear_solver_schur_chebyshev_degree: config.cfd.precond_schur_chebyshev_degree.max(1),
        linear_solver_schur_chebyshev_eigenvalue_ratio: config
            .cfd
            .precond_schur_chebyshev_eigenvalue_ratio
            .max(1.0),
        linear_solver_chebyshev_degree: config.cfd.precond_chebyshev_degree.max(1),
        linear_solver_chebyshev_eigenvalue_ratio: config
            .cfd
            .precond_chebyshev_eigenvalue_ratio
            .max(1.0),
        linear_solver_kind: config.cfd.solver_kind.into_fem(),
        linear_solver_ir_inner_max_iterations: config.cfd.solver_ir_inner_max_iterations.max(1),
        linear_solver_ir_inner_tolerance: config.cfd.solver_ir_inner_tolerance,
        ..NavierStokesConfig::default()
    };
    if !matches!(
        purpose,
        SteadySolvePurpose::Production | SteadySolvePurpose::LooseStage
    ) && initial_state.is_some()
    {
        // A map probe must evaluate exactly one production Picard update at
        // the supplied state. It must not silently continue to a converged
        // solve, or the finite-difference residual would be for a different
        // operator than the published map.
        ns_config.max_picard_iterations = 1;
        ns_config.min_picard_iterations = 1;
        ns_config.picard_tolerance = 0.0;
    }
    if matches!(
        purpose,
        SteadySolvePurpose::Production | SteadySolvePurpose::LooseStage
    ) && config.cfd.max_picard_iterations > 1
    {
        // Damped Newton refinement on the exact tangent: the heavily
        // under-relaxed Picard iteration alone creeps toward the fixed
        // point and cannot reach production nonlinear tolerances in a
        // practical iteration count. A one-iteration budget is a linearized
        // map probe (backend parity fixtures), not a solve, and must keep
        // the single Picard update semantics.
        ns_config.use_newton_refinement = std::env::var("SONIUM_PCFAN_NO_NEWTON").is_err();
        ns_config.max_newton_iterations = ns_config.max_picard_iterations;
        // Loose continuation stages get the cheap globalization: the
        // outer parameter stepping, not per-iteration shift searches,
        // is what tracks the branch there.
        ns_config.newton_extended_globalization = purpose == SteadySolvePurpose::Production;
        if purpose == SteadySolvePurpose::LooseStage {
            // Loose stages only track the branch: refining the exact
            // residual to the production Newton tolerance (1e-10)
            // wastes the last quadratic-tail iterations of every
            // stage. The caller's stage tolerance is the target.
            ns_config.newton_tolerance = ns_config.picard_tolerance;
        }
        if initial_state.is_some() {
            // Warm-started solves are already near the branch. There
            // the mass-shifted/Levenberg escalation accepts near-null
            // decreasing steps and crawls along the valley (observed
            // in prolongated mesh-level solves), while the raw
            // correction with halving converges. Cold production
            // solves keep the full escalation ladder.
            ns_config.newton_extended_globalization = false;
        }

        if ns_config.use_newton_refinement {
            // Newton convergence is judged on the exact assembled
            // residual, which is meaningful from the first iteration.
            // The Picard settle floor (20 iterations) only delays that
            // return, and Picard-owned iterations after a failed
            // Newton step then drift a state Newton already converged.
            ns_config.min_picard_iterations = 1;
        }
    }
    let solid_boundary_velocity_conditions = solid_boundary_conditions.clone();
    let bc_fn = move |x: f64, y: f64, z: f64| -> Vec<f64> {
        if let Some(velocity) =
            solid_boundary_velocity_conditions.velocity_at_point(Point3::new(x, y, z))
        {
            vec![velocity[0], velocity[1], velocity[2]]
        } else {
            vec![0.0, 0.0, 0.0]
        }
    };

    // In `IncompressibleCfdFull` the blade is a rotating wall (penalty),
    // not a body force — the actuator-disk forcing is disabled, otherwise
    // the blade would be double-counted (wall + bulk thrust).
    let blade_as_wall = config.cfd.flow_model.blade_as_wall();
    let blade_forcing = if blade_as_wall {
        None
    } else {
        Some(BladeActuatorForcing::from_geometry(
            config, geometry, design,
        ))
    };
    let actuator_forcing_for_diagnostics = blade_forcing.clone();
    let immersed_solid_penalty = ImmersedSolidPenalty::from_geometry_with_design(
        config,
        geometry,
        solid_boundary_influence_m(domain, nx, ny, nz),
        Some(design),
    );
    let immersed_solid_penalty_for_solver = immersed_solid_penalty.clone();
    let actuator_forcing_for_jacobian = actuator_forcing_for_diagnostics.clone();
    let immersed_solid_penalty_for_jacobian = immersed_solid_penalty.clone();
    let force_fn = move |coords, local_velocity| {
        let mut force = match &blade_forcing {
            Some(f) => f.force_at(coords, local_velocity),
            None => vec![0.0, 0.0, 0.0],
        };
        let penalty = immersed_solid_penalty_for_solver
            .force_at(Point3::new(coords[0], coords[1], coords[2]), local_velocity);
        for component in 0..3 {
            force[component] += penalty[component];
        }
        force
    };
    let force_jacobian_fn = move |coords, local_velocity| {
        let mut jacobian = [[0.0; 3]; 3];
        if let Some(forcing) = &actuator_forcing_for_jacobian {
            add_force_jacobian(
                &mut jacobian,
                forcing.force_jacobian_at(coords, local_velocity),
            );
        }
        add_force_jacobian(
            &mut jacobian,
            immersed_solid_penalty_for_jacobian
                .force_jacobian_at(Point3::new(coords[0], coords[1], coords[2]), local_velocity),
        );
        jacobian
    };
    let result = if purpose == SteadySolvePurpose::NewtonStateProbe
        && let Some((velocity, pressure)) = initial_state
    {
        let linearization =
            sonium_fem::physics::fluid::stokes::navier_stokes::assemble_steady_newton_linearization(
                &mesh,
                &v_space,
                &p_space,
                &ns_config,
                &bc_fn,
                &bc_dofs,
                force_fn,
                force_jacobian_fn,
                velocity,
                pressure,
            );
        sonium_fem::physics::fluid::stokes::navier_stokes::NavierStokesResult {
            velocity: linearization.as_ref().map_or_else(
                || velocity.to_vec(),
                |observation| observation.state[..velocity.len()].to_vec(),
            ),
            pressure: linearization.as_ref().map_or_else(
                || pressure.to_vec(),
                |observation| observation.state[velocity.len()..].to_vec(),
            ),
            iterations: 0,
            residual: f64::INFINITY,
            converged: false,
            resolved_linear_solver_backend:
                sonium_fem::physics::fluid::stokes::navier_stokes::LinearSolverBackend::Cpu,
            picard_linearization: None,
            newton_linearization: linearization,
        }
    } else {
        solve_picard_with_velocity_force_and_jacobian_from_initial(
            &mesh,
            &v_space,
            &p_space,
            &ns_config,
            &bc_fn,
            &bc_dofs,
            force_fn,
            force_jacobian_fn,
            initial_state.map(|(velocity, _)| velocity),
            initial_state.map(|(_, pressure)| pressure),
        )
    };
    let actual_backend = PcFanCfdBackend::from_fem(result.resolved_linear_solver_backend);
    let backend_provenance =
        PcFanCfdBackendProvenance::from_dispatch(config.cfd.backend, actual_backend);
    let execution_environment =
        PcFanCfdExecutionEnvironment::capture(&config.cfd, backend_provenance);
    let picard_linearization = result.picard_linearization.as_ref().map(|source| {
        PcFanResolvedFlowPicardLinearization::from_fem(
            config.cfd.flow_model,
            actual_backend,
            "fem:navier-stokes:picard-oseen:v1",
            source,
        )
        .expect("FEM Picard linearization must satisfy the checked CSR contract")
    });

    let newton_provenance = if matches!(config.cfd.turbulence, PcFanTurbulenceModel::Laminar) {
        "fem:navier-stokes:newton-convection:v1"
    } else {
        "fem:navier-stokes:newton-convection-turbulence-peak-subgradient:v1"
    };
    let newton_linearization = result.newton_linearization.as_ref().map(|source| {
        PcFanResolvedFlowNewtonLinearization::from_fem(
            config.cfd.flow_model,
            actual_backend,
            newton_provenance,
            source,
        )
        .expect("FEM Newton linearization must satisfy checked CSR contract")
    });

    // Publish the exact production design residual derivative. Actuator
    // loading differentiates its analytic force law. Full rotating-wall mode
    // publishes analytic mesh-scaled blade, smooth shroud occupancy, and
    // topology-preserving mesh-motion derivatives; topology-changing remeshing
    // remains outside this contract.
    let residual_design_jacobian = if publish_design_derivatives
        && let Some(linearization) = result.newton_linearization.as_ref()
        && design
            .shroud_radius_offsets_m
            .iter()
            .all(|offset| offset.is_finite())
    {
        if !blade_as_wall && actuator_forcing_for_diagnostics.is_none() {
            None
        } else {
            let state_velocity_dimension = v_space.num_global_dofs(&mesh);
            let velocity_magnitude_max = linearization.state[..state_velocity_dimension]
                .as_chunks::<3>()
                .0
                .iter()
                .map(|node_velocity| {
                    node_velocity
                        .iter()
                        .map(|value| value * value)
                        .sum::<f64>()
                        .sqrt()
                })
                .fold(0.0_f64, f64::max);
            let viscous_mu = ns_config.turbulence.effective_mu(
                ns_config.mu,
                ns_config.rho,
                velocity_magnitude_max,
            );
            assemble_body_force_design_residual_with_viscosity(
                &mesh,
                &v_space,
                &p_space,
                &ns_config,
                viscous_mu,
                &linearization.state[..v_space.num_global_dofs(&mesh)],
                DESIGN_VARIABLE_COUNT,
                |coords, local_velocity| {
                    if blade_as_wall {
                        immersed_solid_penalty
                            .force_design_jacobian_at(
                                Point3::new(coords[0], coords[1], coords[2]),
                                local_velocity,
                            )
                            .into_iter()
                            .map(|row| row.to_vec())
                            .collect()
                    } else {
                        actuator_forcing_for_diagnostics
                            .as_ref()
                            .expect("actuator forcing checked above")
                            .force_design_jacobian_at(coords, local_velocity)
                            .into_iter()
                            .map(|row| row.to_vec())
                            .collect()
                    }
                },
                2,
            )
            .ok()
            .and_then(|mut jacobian| {
                let mesh_motion_design_jacobian = mesh_motion_design_jacobian.as_ref()?;
                let velocity_dimension = v_space.num_global_dofs(&mesh);
                let mesh_jacobian = assemble_mesh_motion_design_residual(
                    &mesh,
                    &v_space,
                    &p_space,
                    &ns_config,
                    &result
                        .newton_linearization
                        .as_ref()
                        .expect("checked above")
                        .state[..velocity_dimension],
                    &result
                        .newton_linearization
                        .as_ref()
                        .expect("checked above")
                        .state[velocity_dimension..],
                    mesh_motion_design_jacobian,
                    |coords, local_velocity| {
                        if blade_as_wall {
                            immersed_solid_penalty.force_at(
                                Point3::new(coords[0], coords[1], coords[2]),
                                local_velocity,
                            )
                        } else {
                            let force = actuator_forcing_for_diagnostics
                                .as_ref()
                                .expect("actuator checked above")
                                .force_at(coords, local_velocity);
                            [force[0], force[1], force[2]]
                        }
                    },
                    |coords, local_velocity| {
                        if blade_as_wall {
                            immersed_solid_penalty.force_position_jacobian_at(
                                Point3::new(coords[0], coords[1], coords[2]),
                                local_velocity,
                            )
                        } else {
                            actuator_forcing_for_diagnostics
                                .as_ref()
                                .expect("actuator checked above")
                                .force_position_jacobian_at(coords, local_velocity)
                        }
                    },
                )
                .ok()?;
                for (row, mesh_row) in jacobian.iter_mut().zip(mesh_jacobian) {
                    for (value, mesh_value) in row.iter_mut().zip(mesh_row) {
                        *value += mesh_value;
                    }
                }
                for &dof in &bc_dofs {
                    jacobian[dof].fill(0.0);
                }
                if p_space.num_global_dofs(&mesh) > 0 {
                    let pressure_pin = pressure_pin_dof(&mesh, &p_space);
                    let row = velocity_dimension + pressure_pin;
                    let matrix = &linearization.state_jacobian;
                    // Only erase an actual gauge constraint. Natural open
                    // boundaries anchor pressure through their traction, so
                    // the corresponding continuity derivative remains active.
                    let pinned = matrix.row_range(row).any(|entry| {
                        matrix.col_indices[entry] == row && matrix.values[entry] == 1.0
                    }) && matrix.row_range(row).all(|entry| {
                        matrix.col_indices[entry] == row || matrix.values[entry] == 0.0
                    });
                    if pinned {
                        jacobian[row].fill(0.0);
                    }
                }
                Some(jacobian)
            })
        }
    } else {
        None
    };
    let newton_linearization = newton_linearization.map(|mut linearization| {
        linearization.residual_design_jacobian = residual_design_jacobian.clone();
        linearization
    });

    let nodes = mesh
        .nodes
        .iter()
        .map(|node| Point3::new(node.x, node.y, node.z))
        .collect::<Vec<_>>();
    let velocity_m_s = velocity_vectors(&result.velocity, nodes.len());
    let actuator_loading = actuator_forcing_for_diagnostics
        .as_ref()
        .map(|forcing| integrate_blade_actuator_loading(&mesh, &velocity_m_s, forcing));
    let rotating_wall_loading =
        config.cfd.flow_model.blade_as_wall().then(|| {
            integrate_rotating_wall_loading(&mesh, &velocity_m_s, &immersed_solid_penalty)
        });
    let signed_inlet_airflow_cfm = boundary_airflow_cfm(
        &nodes,
        &mesh.elements,
        &velocity_m_s,
        domain.min_corner.z,
        domain.inlet_area_m2,
        boundary_tolerance,
    );
    let signed_outlet_airflow_cfm = boundary_airflow_cfm(
        &nodes,
        &mesh.elements,
        &velocity_m_s,
        domain.max_corner.z,
        domain.outlet_area_m2,
        boundary_tolerance,
    );
    let inlet_airflow_cfm = signed_inlet_airflow_cfm.abs();
    let airflow_cfm = signed_outlet_airflow_cfm.abs();
    let pressure_drop_pa = average_boundary_pressure(
        &nodes,
        &result.pressure,
        domain.min_corner.z,
        boundary_tolerance,
    ) - average_boundary_pressure(
        &nodes,
        &result.pressure,
        domain.max_corner.z,
        boundary_tolerance,
    );
    let conservation = build_conservation_ledger(
        config,
        domain,
        &nodes,
        &velocity_m_s,
        &result.pressure,
        signed_inlet_airflow_cfm,
        signed_outlet_airflow_cfm,
        actuator_loading,
        rotating_wall_loading,
        boundary_tolerance,
    );
    let max_velocity_m_s = velocity_m_s
        .iter()
        .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
        .fold(0.0_f64, f64::max);
    let hydraulic_diameter_m = domain.inlet_area_m2.sqrt();

    PcFanCfdFlowSolution {
        domain: domain.clone(),
        actual_backend,
        nodes,
        execution_environment,
        velocity_m_s,
        pressure_pa: result.pressure,
        airflow_cfm,
        inlet_airflow_cfm,
        mass_balance_error_kg_s: config.operating_point.air_density_kg_m3
            * ((signed_outlet_airflow_cfm - signed_inlet_airflow_cfm) / M3S_TO_CFM),
        pressure_drop_pa,
        max_velocity_m_s,
        reynolds_number: config.operating_point.air_density_kg_m3
            * max_velocity_m_s
            * hydraulic_diameter_m
            / config
                .operating_point
                .dynamic_viscosity_pa_s
                .max(f64::EPSILON),
        fem_node_count: mesh.num_nodes(),
        fem_element_count: mesh.num_elements(),
        boundary_dof_count: bc_dofs.len(),
        actuator_loading,
        rotating_wall_loading,
        conservation,
        picard_linearization,
        newton_linearization,
        residual_design_jacobian,
        iterations: result.iterations,
        residual: result.residual,
        converged: result.converged
            && backend_provenance.is_admitted(
                config.cfd.allow_host_fallback,
                config.cfd.require_device_resident,
            ),
        backend_provenance,
    }
}

/// Integrate the exact blade body-force field passed to the CFD assembly.
/// The P1 velocity is evaluated at each tetrahedron centroid, matching the
/// single-point volume quadrature used for this diagnostics artifact.
fn integrate_blade_actuator_loading(
    mesh: &sonium_fem::mesh::Mesh,
    velocity_m_s: &[[f64; 3]],
    forcing: &BladeActuatorForcing,
) -> PcFanActuatorLoading {
    let mut force_on_fluid_n = [0.0; 3];
    let mut torque_on_fluid_nm = 0.0;
    for element in &mesh.elements {
        if element.element_type != sonium_fem::mesh::ElementType::Tetrahedron
            || element.nodes.len() != 4
        {
            continue;
        }
        let mut centroid = [0.0; 3];
        let mut velocity = [0.0; 3];
        for &node in &element.nodes {
            let point = &mesh.nodes[node];
            centroid[0] += point.x * 0.25;
            centroid[1] += point.y * 0.25;
            centroid[2] += point.z * 0.25;
            for component in 0..3 {
                velocity[component] += velocity_m_s[node][component] * 0.25;
            }
        }
        let volume_m3 = tetrahedron_volume_m3(mesh, &element.nodes);
        if volume_m3 <= 0.0 {
            continue;
        }
        let density = forcing.force_at(centroid, velocity);
        let differential_force = [
            density[0] * volume_m3,
            density[1] * volume_m3,
            density[2] * volume_m3,
        ];
        for component in 0..3 {
            force_on_fluid_n[component] += differential_force[component];
        }
        torque_on_fluid_nm +=
            centroid[0] * differential_force[1] - centroid[1] * differential_force[0];
    }
    PcFanActuatorLoading {
        force_on_fluid_n,
        torque_on_fluid_nm,
        angular_speed_rad_s: forcing.omega_rad_s,
        shaft_power_to_fluid_w: torque_on_fluid_nm * forcing.omega_rad_s,
    }
}

fn integrate_rotating_wall_loading(
    mesh: &sonium_fem::mesh::Mesh,
    velocity_m_s: &[[f64; 3]],
    penalty: &ImmersedSolidPenalty,
) -> PcFanActuatorLoading {
    let mut force_on_fluid_n = [0.0; 3];
    let mut torque_on_fluid_nm = 0.0;
    for element in &mesh.elements {
        if element.element_type != sonium_fem::mesh::ElementType::Tetrahedron
            || element.nodes.len() != 4
        {
            continue;
        }
        let mut centroid = [0.0; 3];
        let mut velocity = [0.0; 3];
        for &node in &element.nodes {
            let point = &mesh.nodes[node];
            centroid[0] += point.x * 0.25;
            centroid[1] += point.y * 0.25;
            centroid[2] += point.z * 0.25;
            for component in 0..3 {
                velocity[component] += velocity_m_s[node][component] * 0.25;
            }
        }
        let volume_m3 = tetrahedron_volume_m3(mesh, &element.nodes);
        if volume_m3 <= 0.0 {
            continue;
        }
        let density =
            penalty.rotating_force_at(Point3::new(centroid[0], centroid[1], centroid[2]), velocity);
        let differential_force = [
            density[0] * volume_m3,
            density[1] * volume_m3,
            density[2] * volume_m3,
        ];
        for component in 0..3 {
            force_on_fluid_n[component] += differential_force[component];
        }
        torque_on_fluid_nm +=
            centroid[0] * differential_force[1] - centroid[1] * differential_force[0];
    }
    PcFanActuatorLoading {
        force_on_fluid_n,
        torque_on_fluid_nm,
        angular_speed_rad_s: penalty.omega_rad_s,
        shaft_power_to_fluid_w: torque_on_fluid_nm * penalty.omega_rad_s,
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "diagnostics entry point threading config, domain, state and conservation accumulators; grouping would churn the ledger call sites"
)]
fn build_conservation_ledger(
    config: &PcFanConfig,
    domain: &PcFanCfdDomain,
    nodes: &[Point3],
    velocity_m_s: &[[f64; 3]],
    pressure_pa: &[f64],
    signed_inlet_airflow_cfm: f64,
    signed_outlet_airflow_cfm: f64,
    actuator_loading: Option<PcFanActuatorLoading>,
    rotating_wall_loading: Option<PcFanActuatorLoading>,
    boundary_tolerance: f64,
) -> PcFanCfdConservation {
    let density = config.operating_point.air_density_kg_m3;
    let inlet_flow_m3_s = signed_inlet_airflow_cfm / M3S_TO_CFM;
    let outlet_flow_m3_s = signed_outlet_airflow_cfm / M3S_TO_CFM;
    let inlet_mass_flow_kg_s = density * inlet_flow_m3_s;
    let outlet_mass_flow_kg_s = density * outlet_flow_m3_s;
    let relative_mass_imbalance = (outlet_mass_flow_kg_s - inlet_mass_flow_kg_s).abs()
        / inlet_mass_flow_kg_s
            .abs()
            .max(outlet_mass_flow_kg_s.abs())
            .max(f64::MIN_POSITIVE);
    let inlet_velocity = average_boundary_axial_velocity(
        nodes,
        velocity_m_s,
        domain.min_corner.z,
        boundary_tolerance,
    );
    let outlet_velocity = average_boundary_axial_velocity(
        nodes,
        velocity_m_s,
        domain.max_corner.z,
        boundary_tolerance,
    );
    let inlet_axial_momentum_flux_n = inlet_mass_flow_kg_s * inlet_velocity;
    let outlet_axial_momentum_flux_n = outlet_mass_flow_kg_s * outlet_velocity;
    let inlet_pressure_pa =
        average_boundary_pressure(nodes, pressure_pa, domain.min_corner.z, boundary_tolerance);
    let outlet_pressure_pa =
        average_boundary_pressure(nodes, pressure_pa, domain.max_corner.z, boundary_tolerance);
    let pressure_force_n =
        inlet_pressure_pa * domain.inlet_area_m2 - outlet_pressure_pa * domain.outlet_area_m2;
    let loading = actuator_loading.or(rotating_wall_loading);
    let body_force_n = loading.map_or(0.0, |value| value.force_on_fluid_n[2]);
    let axial_momentum_residual_n = outlet_axial_momentum_flux_n
        - inlet_axial_momentum_flux_n
        - pressure_force_n
        - body_force_n;
    let momentum_scale = (outlet_axial_momentum_flux_n - inlet_axial_momentum_flux_n)
        .abs()
        .max(pressure_force_n.abs())
        .max(body_force_n.abs())
        .max(f64::MIN_POSITIVE);
    let relative_momentum_imbalance = axial_momentum_residual_n.abs() / momentum_scale;
    let pressure_force_from_loading_n = loading.map_or(0.0, |value| value.force_on_fluid_n[2]);
    let pressure_closure_scale = pressure_force_n
        .abs()
        .max(pressure_force_from_loading_n.abs())
        .max(f64::MIN_POSITIVE);
    // The pressure traction is positive in the +z momentum balance when the
    // inlet pressure exceeds the outlet pressure.  A fan raises pressure in
    // the +z direction, so its pressure-rise power is the negative of that
    // traction times the signed volumetric flow.  The recovered wall/body
    // loading is the opposing +z force and therefore closes the momentum
    // balance with `pressure_force_n + loading_force_n = 0`.
    let relative_pressure_closure =
        (pressure_force_n + pressure_force_from_loading_n).abs() / pressure_closure_scale;
    let pressure_power_w = -pressure_force_n * 0.5 * (inlet_flow_m3_s + outlet_flow_m3_s);
    let shaft_power_to_fluid_w = loading.map_or(0.0, |value| value.shaft_power_to_fluid_w);
    let energy_scale = pressure_power_w
        .abs()
        .max(shaft_power_to_fluid_w.abs())
        .max(f64::MIN_POSITIVE);
    let relative_energy_imbalance =
        (pressure_power_w - shaft_power_to_fluid_w).abs() / energy_scale;
    PcFanCfdConservation {
        inlet_mass_flow_kg_s,
        outlet_mass_flow_kg_s,
        relative_mass_imbalance,
        inlet_axial_momentum_flux_n,
        outlet_axial_momentum_flux_n,
        pressure_force_n,
        body_force_n,
        axial_momentum_residual_n,
        relative_momentum_imbalance,
        pressure_force_from_loading_n,
        relative_pressure_closure,
        pressure_power_w,
        shaft_power_to_fluid_w,
        relative_energy_imbalance,
    }
}

fn tetrahedron_volume_m3(mesh: &sonium_fem::mesh::Mesh, nodes: &[usize]) -> f64 {
    let [a, b, c, d] = [
        mesh.nodes[nodes[0]],
        mesh.nodes[nodes[1]],
        mesh.nodes[nodes[2]],
        mesh.nodes[nodes[3]],
    ];
    let ab = [b.x - a.x, b.y - a.y, b.z - a.z];
    let ac = [c.x - a.x, c.y - a.y, c.z - a.z];
    let ad = [d.x - a.x, d.y - a.y, d.z - a.z];
    (ab[0] * (ac[1] * ad[2] - ac[2] * ad[1]) - ab[1] * (ac[0] * ad[2] - ac[2] * ad[0])
        + ab[2] * (ac[0] * ad[1] - ac[1] * ad[0]))
        .abs()
        / 6.0
}

fn tetrahedron_volume_design_jacobian(
    mesh: &sonium_fem::mesh::Mesh,
    nodes: &[usize],
    mesh_motion_design_jacobian: &[[[f64; 3]; DESIGN_VARIABLE_COUNT]],
) -> [f64; DESIGN_VARIABLE_COUNT] {
    let [a, b, c, d] = [
        mesh.nodes[nodes[0]],
        mesh.nodes[nodes[1]],
        mesh.nodes[nodes[2]],
        mesh.nodes[nodes[3]],
    ];
    let ab = [b.x - a.x, b.y - a.y, b.z - a.z];
    let ac = [c.x - a.x, c.y - a.y, c.z - a.z];
    let ad = [d.x - a.x, d.y - a.y, d.z - a.z];
    let cross = |lhs: [f64; 3], rhs: [f64; 3]| {
        [
            lhs[1] * rhs[2] - lhs[2] * rhs[1],
            lhs[2] * rhs[0] - lhs[0] * rhs[2],
            lhs[0] * rhs[1] - lhs[1] * rhs[0],
        ]
    };
    let dot = |lhs: [f64; 3], rhs: [f64; 3]| lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
    let determinant = dot(ab, cross(ac, ad));
    if !determinant.is_finite() || determinant.abs() <= f64::EPSILON {
        return [0.0; DESIGN_VARIABLE_COUNT];
    }
    let orientation = determinant.signum();
    let mut jacobian = [0.0; DESIGN_VARIABLE_COUNT];
    for design_index in 0..DESIGN_VARIABLE_COUNT {
        let d_ab = [
            mesh_motion_design_jacobian[nodes[1]][design_index][0]
                - mesh_motion_design_jacobian[nodes[0]][design_index][0],
            mesh_motion_design_jacobian[nodes[1]][design_index][1]
                - mesh_motion_design_jacobian[nodes[0]][design_index][1],
            mesh_motion_design_jacobian[nodes[1]][design_index][2]
                - mesh_motion_design_jacobian[nodes[0]][design_index][2],
        ];
        let d_ac = [
            mesh_motion_design_jacobian[nodes[2]][design_index][0]
                - mesh_motion_design_jacobian[nodes[0]][design_index][0],
            mesh_motion_design_jacobian[nodes[2]][design_index][1]
                - mesh_motion_design_jacobian[nodes[0]][design_index][1],
            mesh_motion_design_jacobian[nodes[2]][design_index][2]
                - mesh_motion_design_jacobian[nodes[0]][design_index][2],
        ];
        let d_ad = [
            mesh_motion_design_jacobian[nodes[3]][design_index][0]
                - mesh_motion_design_jacobian[nodes[0]][design_index][0],
            mesh_motion_design_jacobian[nodes[3]][design_index][1]
                - mesh_motion_design_jacobian[nodes[0]][design_index][1],
            mesh_motion_design_jacobian[nodes[3]][design_index][2]
                - mesh_motion_design_jacobian[nodes[0]][design_index][2],
        ];
        let determinant_derivative =
            dot(d_ab, cross(ac, ad)) + dot(ab, cross(d_ac, ad)) + dot(ab, cross(ac, d_ad));
        jacobian[design_index] = orientation * determinant_derivative / 6.0;
    }
    jacobian
}

#[derive(Debug, Clone)]
pub(crate) struct SolidBoundaryConditions {
    samples: Vec<SolidBoundarySample>,
    influence_radius_m: f64,
    omega_rad_s: f64,
}

impl SolidBoundaryConditions {
    pub(crate) fn from_domain(
        config: &PcFanConfig,
        domain: &PcFanCfdDomain,
        influence_radius_m: f64,
    ) -> Self {
        let samples = domain
            .boundary_faces
            .iter()
            .filter(|face| face.source_surface.is_some() && face.area > 0.0)
            .map(|face| SolidBoundarySample {
                center: face_center(face.vertices),
                kind: face.kind,
            })
            .collect();

        Self {
            samples,
            influence_radius_m,
            omega_rad_s: std::f64::consts::TAU * config.operating_point.rpm / 60.0,
        }
    }

    pub(crate) fn velocity_at_point(&self, point: Point3) -> Option<[f64; 3]> {
        let influence_squared = self.influence_radius_m * self.influence_radius_m;
        let sample = self
            .samples
            .iter()
            .filter_map(|sample| {
                let distance_squared = distance_squared(point, sample.center);
                (distance_squared <= influence_squared).then_some((distance_squared, sample))
            })
            .min_by(|(a, _), (b, _)| a.total_cmp(b))
            .map(|(_, sample)| sample)?;

        Some(match sample.kind {
            PcFanCfdBoundaryKind::RotatingHub => {
                rotating_wall_velocity_m_s(self.omega_rad_s, point)
            }
            PcFanCfdBoundaryKind::RotatingBlade => return None,
            PcFanCfdBoundaryKind::FrameWall
            | PcFanCfdBoundaryKind::ShroudWall
            | PcFanCfdBoundaryKind::SupportWall
            | PcFanCfdBoundaryKind::MotorWall => [0.0, 0.0, 0.0],
            PcFanCfdBoundaryKind::Inlet
            | PcFanCfdBoundaryKind::Outlet
            | PcFanCfdBoundaryKind::OuterWall => return None,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct SolidBoundarySample {
    center: Point3,
    kind: PcFanCfdBoundaryKind,
}

#[derive(Debug, Clone)]
struct ImmersedSolidPenalty {
    frame_box: PcFanFrameBox,
    shroud: PcFanShroud,
    support_struts: [PcFanSupportStrut; SUPPORT_STRUT_COUNT],
    shroud_notches: Vec<ShroudNotch>,
    motor_mount: PcFanMotorMount,
    hub: PcFanHub,
    omega_rad_s: f64,
    penalty_coefficient_kg_m3_s: f64,
    penalty_coefficient_design_jacobian: [f64; DESIGN_VARIABLE_COUNT],
    /// Blade-as-wall set. `None` in `IncompressibleCfd` mode (where
    /// blades are the actuator-disk body force); `Some` in
    /// `IncompressibleCfdFull` mode where each blade is a rotating wall.
    blade_walls: Option<BladeWallSet>,
}

impl std::fmt::Debug for BladeWallSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BladeWallSet")
            .field("blade_count", &self.azimuths.len())
            .field("hub_radius_m", &self.hub_radius_m)
            .field("tip_radius_m", &self.tip_radius_m)
            .finish()
    }
}

/// Geometric description of the blade swept volumes used by the
/// `IncompressibleCfdFull` mode. The blade is treated as 8 thin twisted
/// plates rotating with the rotor; any quadrature point inside a swept
/// volume gets a stiff penalty force pulling its velocity toward
/// `u = ω × r`.
#[derive(Clone)]
struct BladeWallSet {
    azimuths: Vec<f64>,
    blade_path: [BladePathControl; BLADE_PATH_CONTROL_COUNT],
    blade_sections: [BladeSectionControl; BLADE_SECTION_STATION_COUNT],
    hub_radius_m: f64,
    tip_radius_m: f64,
    smoothing_width_m: f64,
}

impl BladeWallSet {
    #[cfg(test)]
    fn from_design(config: &PcFanConfig, design: &PcFanDesignVariables) -> Self {
        Self::from_design_with_smoothing(config, design, DEFAULT_BLADE_WALL_SMOOTHING_WIDTH_M)
    }

    fn from_design_with_smoothing(
        config: &PcFanConfig,
        design: &PcFanDesignVariables,
        smoothing_width_m: f64,
    ) -> Self {
        let n = config.blade_count.max(1);
        let azimuths = (0..n)
            .map(|k| std::f64::consts::TAU * k as f64 / n as f64)
            .collect();
        Self {
            azimuths,
            blade_path: design.blade_path,
            blade_sections: design.blade_sections,
            hub_radius_m: design.effective_hub_radius_m(config),
            tip_radius_m: design.effective_tip_radius_m(config),
            smoothing_width_m: smoothing_width_m.max(DEFAULT_BLADE_WALL_SMOOTHING_WIDTH_M * 0.25),
        }
    }

    /// True if `point` lies inside any blade's swept thin-plate volume.
    fn point_inside_blade(&self, point: Point3) -> bool {
        let r_world = point.x.hypot(point.y);
        if r_world < self.hub_radius_m || r_world > self.tip_radius_m {
            return false;
        }
        let span_fraction = ((r_world - self.hub_radius_m)
            / (self.tip_radius_m - self.hub_radius_m).max(f64::EPSILON))
        .clamp(0.0, 1.0);
        let section = sample_section(&self.blade_sections, span_fraction);
        let path = sample_path(&self.blade_path, span_fraction);
        let half_chord = 0.5 * section.chord_m.max(0.0);
        let half_thickness = 0.5 * section.thickness_m.max(0.0);
        if half_chord <= 0.0 || half_thickness <= 0.0 {
            return false;
        }
        let twist_rad = section.twist_deg.to_radians();
        let (sin_tw, cos_tw) = twist_rad.sin_cos();

        for &phi in &self.azimuths {
            // Rotate point into blade-local frame so the blade extends
            // along +X; +Y is tangential, +Z stays axial. The rotation is
            // `R(-phi) · (x, y) = (x cosφ + y sinφ, −x sinφ + y cosφ)`.
            let (sin_phi, cos_phi) = phi.sin_cos();
            let local_y = -point.x * sin_phi + point.y * cos_phi;
            // Subtract the path offsets at this span station — the blade
            // midline is centred at (path.tang, path.axial) in (Y, Z).
            let dy = local_y - path.tangential_offset_m;
            let dz = point.z - path.axial_offset_m;
            // Project onto chord direction (sin_tw, cos_tw) and the
            // chord-normal (cos_tw, −sin_tw).
            let chord_proj = dy * sin_tw + dz * cos_tw;
            if chord_proj.abs() > half_chord {
                continue;
            }
            // Camber-corrected normal projection. Camber bows the
            // midline by `camber·(1 − (2c)²)` in the +Y direction
            // (un-twisted). We project that bow onto the chord-normal so
            // a cambered plate is still treated as ~thickness/2 thick.
            let chord_centred = chord_proj / section.chord_m.max(f64::EPSILON);
            let camber_shape = 1.0 - 4.0 * chord_centred * chord_centred;
            let dy_corrected = dy - section.camber_m * camber_shape.max(0.0);
            let normal_proj = dy_corrected * cos_tw - dz * sin_tw;
            if normal_proj.abs() <= half_thickness {
                return true;
            }
        }
        false
    }

    /// Derivative of the union occupancy with respect to the normalized
    /// radial span coordinate `eta`.
    ///
    /// Hub and tip controls change `eta` at a fixed Eulerian quadrature
    /// point.  Keeping this contraction separate from the ordinary blade
    /// control derivatives makes the fixed-mesh radial-envelope contract
    /// explicit and avoids treating mesh motion as an immersed-wall term.
    fn smooth_weight_eta_derivative(&self, point: Point3) -> f64 {
        let r_world = point.x.hypot(point.y);
        if r_world < self.hub_radius_m || r_world > self.tip_radius_m {
            return 0.0;
        }

        let span = (self.tip_radius_m - self.hub_radius_m).max(f64::EPSILON);
        let eta = ((r_world - self.hub_radius_m) / span).clamp(0.0, 1.0);
        let section = sample_section(&self.blade_sections, eta);
        let path = sample_path(&self.blade_path, eta);

        let section_scaled = eta * (BLADE_SECTION_STATION_COUNT - 1) as f64;
        let section_i0 = section_scaled.floor() as usize;
        let section_i1 = (section_i0 + 1).min(BLADE_SECTION_STATION_COUNT - 1);
        let section_scale = (BLADE_SECTION_STATION_COUNT - 1) as f64;
        let d_section_chord = if section_i1 > section_i0 {
            (self.blade_sections[section_i1].chord_m - self.blade_sections[section_i0].chord_m)
                * section_scale
        } else {
            0.0
        };
        let d_section_camber = if section_i1 > section_i0 {
            (self.blade_sections[section_i1].camber_m - self.blade_sections[section_i0].camber_m)
                * section_scale
        } else {
            0.0
        };
        let d_section_thickness = if section_i1 > section_i0 {
            (self.blade_sections[section_i1].thickness_m
                - self.blade_sections[section_i0].thickness_m)
                * section_scale
        } else {
            0.0
        };
        let d_section_twist_deg = if section_i1 > section_i0 {
            (self.blade_sections[section_i1].twist_deg - self.blade_sections[section_i0].twist_deg)
                * section_scale
        } else {
            0.0
        };

        let path_scaled = eta * (BLADE_PATH_CONTROL_COUNT - 1) as f64;
        let path_i0 = path_scaled.floor() as usize;
        let path_i1 = (path_i0 + 1).min(BLADE_PATH_CONTROL_COUNT - 1);
        let path_scale = (BLADE_PATH_CONTROL_COUNT - 1) as f64;
        let d_path_tangential = if path_i1 > path_i0 {
            (self.blade_path[path_i1].tangential_offset_m
                - self.blade_path[path_i0].tangential_offset_m)
                * path_scale
        } else {
            0.0
        };
        let d_path_axial = if path_i1 > path_i0 {
            (self.blade_path[path_i1].axial_offset_m - self.blade_path[path_i0].axial_offset_m)
                * path_scale
        } else {
            0.0
        };

        let half_chord = 0.5 * section.chord_m.max(0.0);
        let half_thickness = 0.5 * section.thickness_m.max(0.0);
        if half_chord <= 0.0 || half_thickness <= 0.0 {
            return 0.0;
        }

        let d_twist = d_section_twist_deg.to_radians();
        let d_half_chord = if section.chord_m > 0.0 {
            0.5 * d_section_chord
        } else {
            0.0
        };
        let d_half_thickness = if section.thickness_m > 0.0 {
            0.5 * d_section_thickness
        } else {
            0.0
        };
        let mut contributions = Vec::with_capacity(self.azimuths.len());

        for &phi in &self.azimuths {
            let (sin_phi, cos_phi) = phi.sin_cos();
            let local_y = -point.x * sin_phi + point.y * cos_phi;
            let twist_rad = section.twist_deg.to_radians();
            let (sin_tw, cos_tw) = twist_rad.sin_cos();
            let dy = local_y - path.tangential_offset_m;
            let dz = point.z - path.axial_offset_m;
            let d_dy = -d_path_tangential;
            let d_dz = -d_path_axial;
            let chord_projection = dy * sin_tw + dz * cos_tw;
            let d_chord_projection =
                d_dy * sin_tw + dy * cos_tw * d_twist + d_dz * cos_tw - dz * sin_tw * d_twist;
            let chord_scale = section.chord_m.max(f64::EPSILON);
            let d_chord = if section.chord_m > 0.0 {
                d_section_chord
            } else {
                0.0
            };
            let chord_coordinate = chord_projection / chord_scale;
            let d_chord_coordinate = d_chord_projection / chord_scale
                - chord_projection * d_chord / (chord_scale * chord_scale);
            let raw_camber_shape = 1.0 - 4.0 * chord_coordinate * chord_coordinate;
            let camber_shape = raw_camber_shape.max(0.0);
            let d_camber_shape = if raw_camber_shape > 0.0 {
                -8.0 * chord_coordinate * d_chord_coordinate
            } else {
                0.0
            };
            let dy_corrected = dy - section.camber_m * camber_shape;
            let d_dy_corrected =
                d_dy - d_section_camber * camber_shape - section.camber_m * d_camber_shape;
            let normal_projection = dy_corrected * cos_tw - dz * sin_tw;
            let d_normal_projection = d_dy_corrected * cos_tw
                - dy_corrected * sin_tw * d_twist
                - d_dz * sin_tw
                - dz * cos_tw * d_twist;
            let abs_chord = (chord_projection * chord_projection
                + BLADE_WALL_SMOOTH_ABS_M * BLADE_WALL_SMOOTH_ABS_M)
                .sqrt();
            let abs_normal = (normal_projection * normal_projection
                + BLADE_WALL_SMOOTH_ABS_M * BLADE_WALL_SMOOTH_ABS_M)
                .sqrt();
            let d_abs_chord = chord_projection / abs_chord * d_chord_projection;
            let d_abs_normal = normal_projection / abs_normal * d_normal_projection;
            let chord_margin = half_chord - abs_chord;
            let thickness_margin = half_thickness - abs_normal;
            let (chord_weight, chord_slope) = smooth_window(chord_margin, self.smoothing_width_m);
            let (thickness_weight, thickness_slope) =
                smooth_window(thickness_margin, self.smoothing_width_m);
            let blade_weight = chord_weight * thickness_weight;
            let d_chord_margin = d_half_chord - d_abs_chord;
            let d_thickness_margin = d_half_thickness - d_abs_normal;
            let d_blade_weight = chord_slope * d_chord_margin * thickness_weight
                + chord_weight * thickness_slope * d_thickness_margin;
            contributions.push((blade_weight, d_blade_weight));
        }

        contributions
            .iter()
            .enumerate()
            .map(|(index, (_, derivative))| {
                let other_complement = contributions
                    .iter()
                    .enumerate()
                    .filter(|(other_index, _)| *other_index != index)
                    .map(|(_, (weight, _))| 1.0 - weight)
                    .product::<f64>();
                other_complement * derivative
            })
            .sum::<f64>()
    }

    /// Return the smooth swept-blade occupancy and its derivative with
    /// respect to the complete blade design vector.
    ///
    /// The production full-CFD penalty uses this occupancy instead of a hard
    /// point-in-volume test.  The union of the individual blade windows is
    /// differentiable, bounded in `[0, 1]`, and remains local to the fixed
    /// Eulerian mesh. Hub and tip radial-envelope controls are contracted
    /// through the span-coordinate derivative below; actual mesh motion
    /// remains a separate admission contract.
    fn smooth_weight_and_design_jacobian(
        &self,
        point: Point3,
    ) -> (f64, [f64; DESIGN_VARIABLE_COUNT]) {
        let mut zero = [0.0; DESIGN_VARIABLE_COUNT];
        let r_world = point.x.hypot(point.y);
        if r_world < self.hub_radius_m || r_world > self.tip_radius_m {
            return (0.0, zero);
        }

        let span = (self.tip_radius_m - self.hub_radius_m).max(f64::EPSILON);
        let eta = ((r_world - self.hub_radius_m) / span).clamp(0.0, 1.0);
        let d_weight_d_eta = self.smooth_weight_eta_derivative(point);
        let radial_start = BLADE_PATH_CONTROL_COUNT * 2 + BLADE_SECTION_STATION_COUNT * 4;
        let eta_hub_derivative = if eta > 0.0 && eta < 1.0 {
            (r_world - self.tip_radius_m) / (span * span)
        } else {
            0.0
        };
        let eta_tip_derivative = if eta > 0.0 && eta < 1.0 {
            -(r_world - self.hub_radius_m) / (span * span)
        } else {
            0.0
        };
        let section_scaled = eta * (BLADE_SECTION_STATION_COUNT - 1) as f64;
        let section_i0 = section_scaled.floor() as usize;
        let section_i1 = (section_i0 + 1).min(BLADE_SECTION_STATION_COUNT - 1);
        let section_t = section_scaled - section_i0 as f64;
        let mut section_weights = [0.0; BLADE_SECTION_STATION_COUNT];
        section_weights[section_i0] += 1.0 - section_t;
        section_weights[section_i1] += section_t;

        let path_scaled = eta * (BLADE_PATH_CONTROL_COUNT - 1) as f64;
        let path_i0 = path_scaled.floor() as usize;
        let path_i1 = (path_i0 + 1).min(BLADE_PATH_CONTROL_COUNT - 1);
        let path_t = path_scaled - path_i0 as f64;
        let mut path_weights = [0.0; BLADE_PATH_CONTROL_COUNT];
        path_weights[path_i0] += 1.0 - path_t;
        path_weights[path_i1] += path_t;

        let section = sample_section(&self.blade_sections, eta);
        let path = sample_path(&self.blade_path, eta);
        let half_chord = 0.5 * section.chord_m.max(0.0);
        let half_thickness = 0.5 * section.thickness_m.max(0.0);
        if half_chord <= 0.0 || half_thickness <= 0.0 {
            return (0.0, zero);
        }

        let mut contributions = Vec::with_capacity(self.azimuths.len());
        let section_start = BLADE_PATH_CONTROL_COUNT * 2;
        for &phi in &self.azimuths {
            let (sin_phi, cos_phi) = phi.sin_cos();
            let local_y = -point.x * sin_phi + point.y * cos_phi;
            let twist_rad = section.twist_deg.to_radians();
            let (sin_tw, cos_tw) = twist_rad.sin_cos();
            let dy = local_y - path.tangential_offset_m;
            let dz = point.z - path.axial_offset_m;
            let chord_projection = dy * sin_tw + dz * cos_tw;
            let chord_scale = section.chord_m.max(f64::EPSILON);
            let chord_coordinate = chord_projection / chord_scale;
            let raw_camber_shape = 1.0 - 4.0 * chord_coordinate * chord_coordinate;
            let camber_shape = raw_camber_shape.max(0.0);
            let dy_corrected = dy - section.camber_m * camber_shape;
            let normal_projection = dy_corrected * cos_tw - dz * sin_tw;
            let abs_chord = (chord_projection * chord_projection
                + BLADE_WALL_SMOOTH_ABS_M * BLADE_WALL_SMOOTH_ABS_M)
                .sqrt();
            let abs_normal = (normal_projection * normal_projection
                + BLADE_WALL_SMOOTH_ABS_M * BLADE_WALL_SMOOTH_ABS_M)
                .sqrt();
            let chord_margin = half_chord - abs_chord;
            let thickness_margin = half_thickness - abs_normal;
            let (chord_weight, chord_slope) = smooth_window(chord_margin, self.smoothing_width_m);
            let (thickness_weight, thickness_slope) =
                smooth_window(thickness_margin, self.smoothing_width_m);
            let blade_weight = chord_weight * thickness_weight;
            let mut blade_gradient = [0.0; DESIGN_VARIABLE_COUNT];

            for (design_index, slot) in blade_gradient.iter_mut().enumerate() {
                let (d_tangential, d_axial) = if design_index < section_start {
                    let control = design_index / 2;
                    let weight = path_weights[control];
                    if design_index % 2 == 0 {
                        (weight, 0.0)
                    } else {
                        (0.0, weight)
                    }
                } else {
                    // Section and remaining variables do not move the path frame.
                    (0.0, 0.0)
                };
                let (d_chord, d_camber, d_thickness, d_twist_deg) = if (section_start
                    ..section_start + BLADE_SECTION_STATION_COUNT * 4)
                    .contains(&design_index)
                {
                    let section_offset = design_index - section_start;
                    let control = section_offset / 4;
                    let weight = section_weights[control];
                    match section_offset % 4 {
                        0 => (weight, 0.0, 0.0, 0.0),
                        1 => (0.0, weight, 0.0, 0.0),
                        2 => (0.0, 0.0, weight, 0.0),
                        _ => (0.0, 0.0, 0.0, weight),
                    }
                } else {
                    (0.0, 0.0, 0.0, 0.0)
                };
                let d_twist = d_twist_deg.to_radians();
                let d_dy = -d_tangential;
                let d_dz = -d_axial;
                let d_chord_projection =
                    d_dy * sin_tw + dy * cos_tw * d_twist + d_dz * cos_tw - dz * sin_tw * d_twist;
                let d_half_chord = if section.chord_m > 0.0 {
                    0.5 * d_chord
                } else {
                    0.0
                };
                let d_half_thickness = if section.thickness_m > 0.0 {
                    0.5 * d_thickness
                } else {
                    0.0
                };
                let d_chord_coordinate = d_chord_projection / chord_scale
                    - chord_projection * d_chord / (chord_scale * chord_scale);
                let d_camber_shape = if raw_camber_shape > 0.0 {
                    -8.0 * chord_coordinate * d_chord_coordinate
                } else {
                    0.0
                };
                let d_dy_corrected =
                    d_dy - d_camber * camber_shape - section.camber_m * d_camber_shape;
                let d_normal_projection = d_dy_corrected * cos_tw
                    - dy_corrected * sin_tw * d_twist
                    - d_dz * sin_tw
                    - dz * cos_tw * d_twist;
                let d_abs_chord = chord_projection / abs_chord * d_chord_projection;
                let d_abs_normal = normal_projection / abs_normal * d_normal_projection;
                let d_chord_margin = d_half_chord - d_abs_chord;
                let d_thickness_margin = d_half_thickness - d_abs_normal;
                *slot = chord_slope * d_chord_margin * thickness_weight
                    + chord_weight * thickness_slope * d_thickness_margin;
            }
            contributions.push((blade_weight, blade_gradient));
        }

        let complement = contributions
            .iter()
            .map(|(weight, _)| 1.0 - weight)
            .product::<f64>();
        let union_weight = (1.0 - complement).clamp(0.0, 1.0);
        for (index, (_, gradient)) in contributions.iter().enumerate() {
            let other_complement = contributions
                .iter()
                .enumerate()
                .filter(|(other_index, _)| *other_index != index)
                .map(|(_, (weight, _))| 1.0 - weight)
                .product::<f64>();
            for design_index in 0..DESIGN_VARIABLE_COUNT {
                zero[design_index] += other_complement * gradient[design_index];
            }
        }
        zero[radial_start] = d_weight_d_eta * eta_hub_derivative;
        zero[radial_start + 1] = d_weight_d_eta * eta_tip_derivative;
        (union_weight, zero)
    }

    fn smooth_weight_and_spatial_gradient(&self, point: Point3) -> (f64, [f64; 3]) {
        let mut zero = [0.0; 3];
        let r_world = point.x.hypot(point.y);
        if r_world < self.hub_radius_m || r_world > self.tip_radius_m {
            return (0.0, zero);
        }

        let span = (self.tip_radius_m - self.hub_radius_m).max(f64::EPSILON);
        let eta = ((r_world - self.hub_radius_m) / span).clamp(0.0, 1.0);
        let d_weight_d_eta = self.smooth_weight_eta_derivative(point);
        let d_eta_dr = if eta > 0.0 && eta < 1.0 {
            1.0 / span
        } else {
            0.0
        };
        let eta_gradient = [
            d_eta_dr * point.x / r_world.max(f64::EPSILON),
            d_eta_dr * point.y / r_world.max(f64::EPSILON),
            0.0,
        ];
        let section = sample_section(&self.blade_sections, eta);
        let path = sample_path(&self.blade_path, eta);
        let half_chord = 0.5 * section.chord_m.max(0.0);
        let half_thickness = 0.5 * section.thickness_m.max(0.0);
        if half_chord <= 0.0 || half_thickness <= 0.0 {
            return (0.0, zero);
        }

        let mut contributions = Vec::with_capacity(self.azimuths.len());
        for &phi in &self.azimuths {
            let (sin_phi, cos_phi) = phi.sin_cos();
            let local_y = -point.x * sin_phi + point.y * cos_phi;
            let dy = local_y - path.tangential_offset_m;
            let dz = point.z - path.axial_offset_m;
            let twist_rad = section.twist_deg.to_radians();
            let (sin_tw, cos_tw) = twist_rad.sin_cos();
            let chord_projection = dy * sin_tw + dz * cos_tw;
            let chord_scale = section.chord_m.max(f64::EPSILON);
            let chord_coordinate = chord_projection / chord_scale;
            let raw_camber_shape = 1.0 - 4.0 * chord_coordinate * chord_coordinate;
            let camber_shape = raw_camber_shape.max(0.0);
            let dy_corrected = dy - section.camber_m * camber_shape;
            let normal_projection = dy_corrected * cos_tw - dz * sin_tw;
            let abs_chord =
                (chord_projection * chord_projection + BLADE_WALL_SMOOTH_ABS_M.powi(2)).sqrt();
            let abs_normal =
                (normal_projection * normal_projection + BLADE_WALL_SMOOTH_ABS_M.powi(2)).sqrt();
            let (chord_weight, chord_slope) =
                smooth_window(half_chord - abs_chord, self.smoothing_width_m);
            let (thickness_weight, thickness_slope) =
                smooth_window(half_thickness - abs_normal, self.smoothing_width_m);
            let blade_weight = chord_weight * thickness_weight;
            let mut blade_gradient = [0.0; 3];
            for (direction, slot) in blade_gradient.iter_mut().enumerate() {
                let d_local_y = if direction == 0 {
                    -sin_phi
                } else if direction == 1 {
                    cos_phi
                } else {
                    0.0
                };
                let d_dy = d_local_y;
                let d_dz = if direction == 2 { 1.0 } else { 0.0 };
                let d_chord_projection = d_dy * sin_tw + d_dz * cos_tw;
                let d_chord_coordinate = d_chord_projection / chord_scale;
                let d_camber_shape = if raw_camber_shape > 0.0 {
                    -8.0 * chord_coordinate * d_chord_coordinate
                } else {
                    0.0
                };
                let d_dy_corrected = d_dy - section.camber_m * d_camber_shape;
                let d_normal_projection = d_dy_corrected * cos_tw - d_dz * sin_tw;
                let d_abs_chord = chord_projection / abs_chord * d_chord_projection;
                let d_abs_normal = normal_projection / abs_normal * d_normal_projection;
                let d_chord_margin = -d_abs_chord;
                let d_thickness_margin = -d_abs_normal;
                *slot = chord_slope * d_chord_margin * thickness_weight
                    + chord_weight * thickness_slope * d_thickness_margin;
            }
            contributions.push((blade_weight, blade_gradient));
        }

        let complement = contributions
            .iter()
            .map(|(weight, _)| 1.0 - weight)
            .product::<f64>();
        let union_weight = (1.0 - complement).clamp(0.0, 1.0);
        for (index, (_, gradient)) in contributions.iter().enumerate() {
            let other_complement = contributions
                .iter()
                .enumerate()
                .filter(|(other_index, _)| *other_index != index)
                .map(|(_, (weight, _))| 1.0 - weight)
                .product::<f64>();
            for direction in 0..3 {
                zero[direction] += other_complement * gradient[direction];
            }
        }
        for direction in 0..3 {
            zero[direction] += d_weight_d_eta * eta_gradient[direction];
        }
        (union_weight, zero)
    }
}

fn smooth_window(margin: f64, width: f64) -> (f64, f64) {
    let argument = (margin / width.max(f64::EPSILON)).clamp(-60.0, 60.0);
    let weight = if argument >= 0.0 {
        1.0 / (1.0 + (-argument).exp())
    } else {
        let exp_argument = argument.exp();
        exp_argument / (1.0 + exp_argument)
    };
    let slope = weight * (1.0 - weight) / width.max(f64::EPSILON);
    (weight, slope)
}

fn shroud_radius_offset_basis(depth_fraction: f64) -> [f64; SHROUD_CONTROL_COUNT] {
    let scaled = depth_fraction.clamp(0.0, 1.0) * (SHROUD_CONTROL_COUNT - 1) as f64;
    let i1 = scaled.floor() as usize;
    let i2 = (i1 + 1).min(SHROUD_CONTROL_COUNT - 1);
    let i0 = i1.saturating_sub(1);
    let i3 = (i2 + 1).min(SHROUD_CONTROL_COUNT - 1);
    let t = scaled - i1 as f64;
    let t2 = t * t;
    let t3 = t2 * t;
    let weights = [
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    ];
    let mut basis = [0.0; SHROUD_CONTROL_COUNT];
    basis[i0] += weights[0];
    basis[i1] += weights[1];
    basis[i2] += weights[2];
    basis[i3] += weights[3];
    basis
}

fn sample_section(
    sections: &[BladeSectionControl; BLADE_SECTION_STATION_COUNT],
    eta: f64,
) -> BladeSectionControl {
    let n = BLADE_SECTION_STATION_COUNT;
    if n == 0 {
        return BladeSectionControl::default();
    }
    let scaled = eta.clamp(0.0, 1.0) * (n - 1) as f64;
    let i0 = scaled.floor() as usize;
    let i1 = (i0 + 1).min(n - 1);
    let t = scaled - i0 as f64;
    let a = sections[i0];
    let b = sections[i1];
    BladeSectionControl {
        chord_m: a.chord_m + t * (b.chord_m - a.chord_m),
        camber_m: a.camber_m + t * (b.camber_m - a.camber_m),
        thickness_m: a.thickness_m + t * (b.thickness_m - a.thickness_m),
        twist_deg: a.twist_deg + t * (b.twist_deg - a.twist_deg),
    }
}

fn sample_path(path: &[BladePathControl; BLADE_PATH_CONTROL_COUNT], eta: f64) -> BladePathControl {
    let n = BLADE_PATH_CONTROL_COUNT;
    if n == 0 {
        return BladePathControl::default();
    }
    let scaled = eta.clamp(0.0, 1.0) * (n - 1) as f64;
    let i0 = scaled.floor() as usize;
    let i1 = (i0 + 1).min(n - 1);
    let t = scaled - i0 as f64;
    let a = path[i0];
    let b = path[i1];
    BladePathControl {
        tangential_offset_m: a.tangential_offset_m
            + t * (b.tangential_offset_m - a.tangential_offset_m),
        axial_offset_m: a.axial_offset_m + t * (b.axial_offset_m - a.axial_offset_m),
    }
}

impl ImmersedSolidPenalty {
    fn from_geometry_with_design(
        config: &PcFanConfig,
        geometry: &PcFanGeometry,
        mesh_length_scale_m: f64,
        design: Option<&PcFanDesignVariables>,
    ) -> Self {
        let omega_rad_s = std::f64::consts::TAU * config.operating_point.rpm / 60.0;
        let tip_radius_m = design
            .map(|candidate| candidate.effective_tip_radius_m(config))
            .unwrap_or(config.dimensions.blade_tip_radius_m);
        let tip_speed_m_s = (omega_rad_s * tip_radius_m).max(0.25);
        let mesh_length_scale_m = mesh_length_scale_m.max(1.0e-4);
        let unclamped_relaxation_time_s = mesh_length_scale_m / tip_speed_m_s;
        let relaxation_time_s = (mesh_length_scale_m.max(1.0e-4) / tip_speed_m_s).clamp(
            IMMERSED_SOLID_MIN_RELAXATION_S,
            IMMERSED_SOLID_MAX_RELAXATION_S,
        );
        let penalty_coefficient_kg_m3_s =
            config.operating_point.air_density_kg_m3 / relaxation_time_s;

        let mut penalty_coefficient_design_jacobian = [0.0; DESIGN_VARIABLE_COUNT];
        if design.is_some() {
            let d_tip_speed_d_tip_control = if omega_rad_s * tip_radius_m > 0.25 {
                omega_rad_s
            } else {
                0.0
            };
            let d_relaxation_time_d_tip_control = if unclamped_relaxation_time_s
                > IMMERSED_SOLID_MIN_RELAXATION_S
                && unclamped_relaxation_time_s < IMMERSED_SOLID_MAX_RELAXATION_S
            {
                -mesh_length_scale_m * d_tip_speed_d_tip_control / tip_speed_m_s.powi(2)
            } else {
                0.0
            };
            let tip_control_index =
                BLADE_PATH_CONTROL_COUNT * 2 + BLADE_SECTION_STATION_COUNT * 4 + 1;
            penalty_coefficient_design_jacobian[tip_control_index] =
                -config.operating_point.air_density_kg_m3 * d_relaxation_time_d_tip_control
                    / relaxation_time_s.powi(2);
        }
        let shroud_notches =
            shroud_strut_notches(&geometry.support_struts, geometry.shroud.minimum_radius_m());

        let blade_walls = match (config.cfd.flow_model.blade_as_wall(), design) {
            (true, Some(d)) => Some(BladeWallSet::from_design_with_smoothing(
                config,
                d,
                (mesh_length_scale_m * 0.25).clamp(DEFAULT_BLADE_WALL_SMOOTHING_WIDTH_M, 2.0e-3),
            )),
            _ => None,
        };
        Self {
            frame_box: geometry.frame_box,
            shroud: geometry.shroud,
            support_struts: geometry.support_struts,
            shroud_notches,
            motor_mount: geometry.motor_mount,
            hub: geometry.hub,
            omega_rad_s,
            penalty_coefficient_kg_m3_s,
            penalty_coefficient_design_jacobian,
            blade_walls,
        }
    }

    fn force_at(&self, point: Point3, local_velocity: [f64; 3]) -> [f64; 3] {
        let (weight, target_velocity) = if self.point_inside_hub(point) {
            (1.0, rotating_wall_velocity_m_s(self.omega_rad_s, point))
        } else if let Some(walls) = &self.blade_walls {
            let (weight, _) = walls.smooth_weight_and_design_jacobian(point);
            if weight > f64::EPSILON {
                (weight, rotating_wall_velocity_m_s(self.omega_rad_s, point))
            } else if self.stationary_weight_and_design_jacobian(point).0 > f64::EPSILON {
                (
                    self.stationary_weight_and_design_jacobian(point).0,
                    [0.0, 0.0, 0.0],
                )
            } else if self.point_inside_motor_mount(point)
                || self.point_inside_support_strut(point)
                || self.point_inside_frame_box(point)
            {
                (1.0, [0.0, 0.0, 0.0])
            } else {
                return [0.0, 0.0, 0.0];
            }
        } else if self.stationary_weight_and_design_jacobian(point).0 > f64::EPSILON {
            (
                self.stationary_weight_and_design_jacobian(point).0,
                [0.0, 0.0, 0.0],
            )
        } else if self.point_inside_motor_mount(point)
            || self.point_inside_support_strut(point)
            || self.point_inside_frame_box(point)
        {
            (1.0, [0.0, 0.0, 0.0])
        } else {
            return [0.0, 0.0, 0.0];
        };
        [
            self.penalty_coefficient_kg_m3_s * weight * (target_velocity[0] - local_velocity[0]),
            self.penalty_coefficient_kg_m3_s * weight * (target_velocity[1] - local_velocity[1]),
            self.penalty_coefficient_kg_m3_s * weight * (target_velocity[2] - local_velocity[2]),
        ]
    }

    fn force_jacobian_at(&self, point: Point3, _local_velocity: [f64; 3]) -> [[f64; 3]; 3] {
        let weight = if self.point_inside_hub(point) {
            1.0
        } else if let Some(walls) = &self.blade_walls {
            let weight = walls.smooth_weight_and_design_jacobian(point).0;
            if weight > f64::EPSILON {
                weight
            } else if self.stationary_weight_and_design_jacobian(point).0 > f64::EPSILON {
                self.stationary_weight_and_design_jacobian(point).0
            } else if self.point_inside_motor_mount(point)
                || self.point_inside_support_strut(point)
                || self.point_inside_frame_box(point)
            {
                1.0
            } else {
                0.0
            }
        } else if self.stationary_weight_and_design_jacobian(point).0 > f64::EPSILON {
            self.stationary_weight_and_design_jacobian(point).0
        } else if self.point_inside_motor_mount(point)
            || self.point_inside_support_strut(point)
            || self.point_inside_frame_box(point)
        {
            1.0
        } else {
            0.0
        };
        let mut jacobian = [[0.0; 3]; 3];
        for (component, row) in jacobian.iter_mut().enumerate() {
            row[component] = -self.penalty_coefficient_kg_m3_s * weight;
        }
        jacobian
    }

    fn force_position_jacobian_at(&self, point: Point3, local_velocity: [f64; 3]) -> [[f64; 3]; 3] {
        let (weight, weight_gradient, target_velocity, target_gradient) =
            if self.point_inside_hub(point) {
                (
                    1.0,
                    [0.0; 3],
                    rotating_wall_velocity_m_s(self.omega_rad_s, point),
                    [
                        [0.0, -self.omega_rad_s, 0.0],
                        [self.omega_rad_s, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                )
            } else if let Some(walls) = &self.blade_walls {
                let (weight, weight_gradient) = walls.smooth_weight_and_spatial_gradient(point);
                (
                    weight,
                    weight_gradient,
                    rotating_wall_velocity_m_s(self.omega_rad_s, point),
                    [
                        [0.0, -self.omega_rad_s, 0.0],
                        [self.omega_rad_s, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                )
            } else {
                let (weight, weight_gradient) = self.stationary_weight_and_spatial_gradient(point);
                if weight > f64::EPSILON {
                    (weight, weight_gradient, [0.0; 3], [[0.0; 3]; 3])
                } else if self.point_inside_motor_mount(point)
                    || self.point_inside_support_strut(point)
                    || self.point_inside_frame_box(point)
                {
                    (1.0, [0.0; 3], [0.0; 3], [[0.0; 3]; 3])
                } else {
                    return [[0.0; 3]; 3];
                }
            };

        let mut jacobian = [[0.0; 3]; 3];
        for component in 0..3 {
            let force_without_weight = target_velocity[component] - local_velocity[component];
            for direction in 0..3 {
                jacobian[component][direction] = self.penalty_coefficient_kg_m3_s
                    * (force_without_weight * weight_gradient[direction]
                        + weight * target_gradient[component][direction]);
            }
        }
        jacobian
    }

    fn rotating_force_at(&self, point: Point3, local_velocity: [f64; 3]) -> [f64; 3] {
        let weight = if self.point_inside_hub(point) {
            1.0
        } else if let Some(walls) = &self.blade_walls {
            walls.smooth_weight_and_design_jacobian(point).0
        } else {
            0.0
        };
        if weight <= f64::EPSILON {
            return [0.0; 3];
        }
        let target = rotating_wall_velocity_m_s(self.omega_rad_s, point);
        [
            self.penalty_coefficient_kg_m3_s * weight * (target[0] - local_velocity[0]),
            self.penalty_coefficient_kg_m3_s * weight * (target[1] - local_velocity[1]),
            self.penalty_coefficient_kg_m3_s * weight * (target[2] - local_velocity[2]),
        ]
    }

    /// State Jacobian of the recovered rotating-wall force only.
    ///
    /// Stationary immersed solids use the same penalty coefficient in the
    /// Navier--Stokes residual, but they do not contribute to the rotating
    /// wall loading ledger. Keeping this support test beside
    /// [`rotating_force_at`] prevents a stationary shroud or frame from
    /// leaking into the rotating-wall objective derivative.
    fn rotating_force_jacobian_at(&self, point: Point3) -> [[f64; 3]; 3] {
        let weight = if self.point_inside_hub(point) {
            1.0
        } else if let Some(walls) = &self.blade_walls {
            walls.smooth_weight_and_design_jacobian(point).0
        } else {
            0.0
        };
        if weight <= f64::EPSILON {
            return [[0.0; 3]; 3];
        }
        [
            [-self.penalty_coefficient_kg_m3_s * weight, 0.0, 0.0],
            [0.0, -self.penalty_coefficient_kg_m3_s * weight, 0.0],
            [0.0, 0.0, -self.penalty_coefficient_kg_m3_s * weight],
        ]
    }

    /// Design derivative of the rotating-wall body force under the
    /// fixed-immersed-topology contract.
    ///
    /// Blade occupancy is represented by a smooth fixed-mesh support field;
    /// hub/tip span controls are differentiated through it, while actual
    /// mesh-motion shape derivatives remain outside this local contract.
    fn rotating_force_design_jacobian_at(
        &self,
        point: Point3,
        local_velocity: [f64; 3],
    ) -> [[f64; DESIGN_VARIABLE_COUNT]; 3] {
        let Some(walls) = &self.blade_walls else {
            return [[0.0; DESIGN_VARIABLE_COUNT]; 3];
        };
        if self.point_inside_hub(point) {
            return [[0.0; DESIGN_VARIABLE_COUNT]; 3];
        }
        let (weight, weight_gradient) = walls.smooth_weight_and_design_jacobian(point);
        let target = rotating_wall_velocity_m_s(self.omega_rad_s, point);
        let mut derivative = [[0.0; DESIGN_VARIABLE_COUNT]; 3];
        for component in 0..3 {
            let force_without_weight =
                self.penalty_coefficient_kg_m3_s * (target[component] - local_velocity[component]);
            for design_index in 0..DESIGN_VARIABLE_COUNT {
                derivative[component][design_index] = force_without_weight
                    * weight_gradient[design_index]
                    + weight
                        * (target[component] - local_velocity[component])
                        * self.penalty_coefficient_design_jacobian[design_index];
            }
        }
        derivative
    }

    fn stationary_weight_and_design_jacobian(
        &self,
        point: Point3,
    ) -> (f64, [f64; DESIGN_VARIABLE_COUNT]) {
        let mut gradient = [0.0; DESIGN_VARIABLE_COUNT];
        let z0 = -0.5 * self.shroud.depth_m;
        let z1 = 0.5 * self.shroud.depth_m;
        if point.z < z0 || point.z > z1 {
            return (0.0, gradient);
        }
        let depth_fraction =
            ((point.z - z0) / self.shroud.depth_m.max(f64::EPSILON)).clamp(0.0, 1.0);
        let radius = point.x.hypot(point.y);
        let inner_radius = self.shroud.radius_at_depth_fraction(depth_fraction);
        let outer_radius = self.shroud.outer_radius_at_depth_fraction(depth_fraction);
        if shroud_in_notch(
            point.y.atan2(point.x),
            point.z,
            &shroud_strut_notches(&self.support_struts, self.shroud.minimum_radius_m()),
        ) {
            return (0.0, gradient);
        }
        let (inner_weight, inner_slope) = smooth_window(
            radius - inner_radius,
            self.blade_walls
                .as_ref()
                .map_or(DEFAULT_BLADE_WALL_SMOOTHING_WIDTH_M, |walls| {
                    walls.smoothing_width_m
                }),
        );
        let (outer_weight, outer_slope) = smooth_window(
            outer_radius - radius,
            self.blade_walls
                .as_ref()
                .map_or(DEFAULT_BLADE_WALL_SMOOTHING_WIDTH_M, |walls| {
                    walls.smoothing_width_m
                }),
        );
        let weight = inner_weight * outer_weight;
        let basis = shroud_radius_offset_basis(depth_fraction);
        let design_start = BLADE_PATH_CONTROL_COUNT * 2
            + BLADE_SECTION_STATION_COUNT * 4
            + RADIAL_ENVELOPE_CONTROL_COUNT;
        let radius_derivative = -inner_slope * outer_weight + inner_weight * outer_slope;
        for (control, basis_value) in basis.into_iter().enumerate() {
            gradient[design_start + control] = radius_derivative * basis_value;
        }
        (weight, gradient)
    }

    fn stationary_weight_and_spatial_gradient(&self, point: Point3) -> (f64, [f64; 3]) {
        let mut gradient = [0.0; 3];
        let z0 = -0.5 * self.shroud.depth_m;
        let z1 = 0.5 * self.shroud.depth_m;
        if point.z < z0 || point.z > z1 {
            return (0.0, gradient);
        }
        let depth_fraction =
            ((point.z - z0) / self.shroud.depth_m.max(f64::EPSILON)).clamp(0.0, 1.0);
        let radius = point.x.hypot(point.y);
        let inner_radius = self.shroud.radius_at_depth_fraction(depth_fraction);
        let outer_radius = self.shroud.outer_radius_at_depth_fraction(depth_fraction);
        if shroud_in_notch(
            point.y.atan2(point.x),
            point.z,
            &shroud_strut_notches(&self.support_struts, self.shroud.minimum_radius_m()),
        ) {
            return (0.0, gradient);
        }
        let width = self
            .blade_walls
            .as_ref()
            .map_or(DEFAULT_BLADE_WALL_SMOOTHING_WIDTH_M, |walls| {
                walls.smoothing_width_m
            });
        let (inner_weight, inner_slope) = smooth_window(radius - inner_radius, width);
        let (outer_weight, outer_slope) = smooth_window(outer_radius - radius, width);
        let weight = inner_weight * outer_weight;
        let radial_x = point.x / radius.max(f64::EPSILON);
        let radial_y = point.y / radius.max(f64::EPSILON);
        let radius_slope_z = self.shroud.radius_slope_at_depth_fraction(depth_fraction)
            / self.shroud.depth_m.max(f64::EPSILON);
        gradient[0] = inner_slope * radial_x * outer_weight - inner_weight * outer_slope * radial_x;
        gradient[1] = inner_slope * radial_y * outer_weight - inner_weight * outer_slope * radial_y;
        gradient[2] = -inner_slope * radius_slope_z * outer_weight
            + inner_weight * outer_slope * radius_slope_z;
        (weight, gradient)
    }

    /// Combine rotating-blade and stationary-shroud fixed-mesh design terms.
    fn force_design_jacobian_at(
        &self,
        point: Point3,
        local_velocity: [f64; 3],
    ) -> [[f64; DESIGN_VARIABLE_COUNT]; 3] {
        let mut derivative = self.rotating_force_design_jacobian_at(point, local_velocity);
        let (_, weight_gradient) = self.stationary_weight_and_design_jacobian(point);
        for component in 0..3 {
            let force_without_weight =
                -self.penalty_coefficient_kg_m3_s * local_velocity[component];
            for design_index in 0..DESIGN_VARIABLE_COUNT {
                derivative[component][design_index] +=
                    force_without_weight * weight_gradient[design_index];
            }
        }
        derivative
    }

    #[allow(dead_code)]
    fn target_velocity_at(&self, point: Point3) -> Option<[f64; 3]> {
        if self.point_inside_hub(point) {
            return Some(rotating_wall_velocity_m_s(self.omega_rad_s, point));
        }

        // CFD-full mode: blade interiors get the rotating-wall target
        // velocity, same as the hub. Skipped when `blade_walls` is None
        // (legacy actuator-disk mode).
        if let Some(walls) = &self.blade_walls
            && walls.point_inside_blade(point)
        {
            return Some(rotating_wall_velocity_m_s(self.omega_rad_s, point));
        }

        (self.point_inside_motor_mount(point)
            || self.point_inside_support_strut(point)
            || self.point_inside_shroud(point)
            || self.point_inside_frame_box(point))
        .then_some([0.0, 0.0, 0.0])
    }

    fn point_inside_hub(&self, point: Point3) -> bool {
        let z0 = self.hub.center_z_m - 0.5 * self.hub.depth_m;
        let z1 = self.hub.center_z_m + 0.5 * self.hub.depth_m;
        point.z >= z0 && point.z <= z1 && point.x.hypot(point.y) <= self.hub.radius_m
    }

    fn point_inside_motor_mount(&self, point: Point3) -> bool {
        let z0 = self.motor_mount.center.z;
        let z1 = z0 + self.motor_mount.depth_m;
        point.z >= z0 && point.z <= z1 && point.x.hypot(point.y) <= self.motor_mount.radius_m
    }

    fn point_inside_support_strut(&self, point: Point3) -> bool {
        self.support_struts
            .iter()
            .any(|strut| point_inside_strut(point, *strut))
    }

    fn point_inside_shroud(&self, point: Point3) -> bool {
        let z0 = -0.5 * self.shroud.depth_m;
        let z1 = 0.5 * self.shroud.depth_m;
        if point.z < z0 || point.z > z1 {
            return false;
        }

        let depth_fraction =
            ((point.z - z0) / self.shroud.depth_m.max(f64::EPSILON)).clamp(0.0, 1.0);
        let radius = point.x.hypot(point.y);
        if radius < self.shroud.radius_at_depth_fraction(depth_fraction)
            || radius > self.shroud.outer_radius_at_depth_fraction(depth_fraction)
        {
            return false;
        }
        let angle = point.y.atan2(point.x);
        !shroud_in_notch(angle, point.z, &self.shroud_notches)
    }

    fn point_inside_frame_box(&self, point: Point3) -> bool {
        let outer = 0.5 * self.frame_box.outer_side_m;
        let inner = 0.5 * self.frame_box.inner_side_m;
        let z0 = -0.5 * self.frame_box.depth_m;
        let z1 = 0.5 * self.frame_box.depth_m;
        let max_abs_xy = point.x.abs().max(point.y.abs());

        point.z >= z0 && point.z <= z1 && max_abs_xy <= outer && max_abs_xy >= inner
    }
}

fn point_inside_strut(point: Point3, strut: PcFanSupportStrut) -> bool {
    let z1 = strut.start.z;
    let z0 = z1 - strut.thickness_m;
    if point.z < z0 || point.z > z1 {
        return false;
    }

    let dx = strut.end.x - strut.start.x;
    let dy = strut.end.y - strut.start.y;
    let length_squared = dx * dx + dy * dy;
    if length_squared <= f64::EPSILON {
        return false;
    }

    let t = ((point.x - strut.start.x) * dx + (point.y - strut.start.y) * dy) / length_squared;
    if !(0.0..=1.0).contains(&t) {
        return false;
    }

    let closest_x = strut.start.x + t * dx;
    let closest_y = strut.start.y + t * dy;
    let distance_m = (point.x - closest_x).hypot(point.y - closest_y);

    let length = length_squared.sqrt();
    let half_width = 0.5 * strut.width_m;
    let bevel = half_width.min(0.5 * length);
    let s = t * length;
    let s_to_tip = s.min(length - s);
    let allowed_half_width = if s_to_tip < bevel {
        s_to_tip
    } else {
        half_width
    };

    distance_m <= allowed_half_width
}

#[derive(Debug, Clone)]
struct BladeActuatorForcing {
    samples: Vec<BladeActuatorSample>,
    sample_design_derivatives: Vec<BladeActuatorSampleDerivative>,
    hub_radius_m: f64,
    tip_radius_m: f64,
    default_chord_m: f64,
    radial_influence_m: f64,
    radial_influence_design_derivative: [f64; DESIGN_VARIABLE_COUNT],
    axial_influence_m: f64,
    omega_rad_s: f64,
    air_density_kg_m3: f64,
    blade_count_scale: f64,
}

#[derive(Debug, Clone, Copy)]
struct BladeActuatorSampleDerivative {
    radius_m: [f64; DESIGN_VARIABLE_COUNT],
    z_m: [f64; DESIGN_VARIABLE_COUNT],
    axial_loading: [f64; DESIGN_VARIABLE_COUNT],
    tangential_loading: [f64; DESIGN_VARIABLE_COUNT],
    area_weight: [f64; DESIGN_VARIABLE_COUNT],
}

/// Tangential force direction for relative tangential speed `s`:
/// the discontinuous `signum(s)` of the baseline actuator model.
fn tangential_direction(relative_tangential_m_s: f64, _tip_speed_m_s: f64) -> f64 {
    relative_tangential_m_s.signum()
}

/// Derivative of [`tangential_direction`] with respect to its speed
/// argument (zero on either signum branch).
fn tangential_direction_derivative(
    _relative_tangential_m_s: f64,
    _tip_speed_m_s: f64,
) -> f64 {
    0.0
}

impl BladeActuatorForcing {
    fn from_geometry(
        config: &PcFanConfig,
        geometry: &PcFanGeometry,
        design: &PcFanDesignVariables,
    ) -> Self {
        let hub_radius_m = design.effective_hub_radius_m(config);
        let tip_radius_m = design.effective_tip_radius_m(config);
        let radial_span_m = (tip_radius_m - hub_radius_m).max(f64::EPSILON);
        let radial_segments = config.mesh.radial_segments.max(1) as f64;
        let raw_radial_influence_m = 1.5 * radial_span_m / radial_segments;
        let radial_influence_m = raw_radial_influence_m.max(ACTUATOR_MIN_RADIAL_INFLUENCE_M);
        let mut radial_influence_design_derivative = [0.0; DESIGN_VARIABLE_COUNT];
        if raw_radial_influence_m > ACTUATOR_MIN_RADIAL_INFLUENCE_M {
            let radial_start = BLADE_PATH_CONTROL_COUNT * 2 + BLADE_SECTION_STATION_COUNT * 4;
            let scale = 1.5 / radial_segments;
            radial_influence_design_derivative[radial_start] = -scale;
            radial_influence_design_derivative[radial_start + 1] = scale;
        }
        let axial_influence_m =
            (0.24 * config.dimensions.depth_m).max(ACTUATOR_MIN_AXIAL_INFLUENCE_M);
        let default_chord_m = BladeSectionControl::default().chord_m.max(1.0e-6);

        let mut samples = Vec::new();
        let mut sample_design_derivatives = Vec::new();
        for blade in &geometry.blades {
            for (panel_index, panel) in blade.panels.iter().enumerate() {
                if panel.area <= 0.0 {
                    continue;
                }
                samples.push(BladeActuatorSample::from_panel(panel, default_chord_m));
                sample_design_derivatives.push(BladeActuatorSampleDerivative::from_panel(
                    config,
                    design,
                    blade.index,
                    panel_index,
                    default_chord_m,
                ));
            }
        }

        Self {
            samples,
            sample_design_derivatives,
            hub_radius_m,
            tip_radius_m,
            default_chord_m,
            radial_influence_m,
            radial_influence_design_derivative,
            axial_influence_m,
            omega_rad_s: std::f64::consts::TAU * config.operating_point.rpm / 60.0,
            air_density_kg_m3: config.operating_point.air_density_kg_m3,
            // The actuator coefficients are calibrated for the default
            // eight-blade rotor. Kernel normalization otherwise cancels
            // duplicate blade samples and makes force independent of count.
            blade_count_scale: config.blade_count.max(1) as f64 / DEFAULT_BLADE_COUNT as f64,
        }
    }

    fn force_at(&self, coords: [f64; 3], local_velocity: [f64; 3]) -> Vec<f64> {
        let x = coords[0];
        let y = coords[1];
        let z = coords[2];
        let r = x.hypot(y);
        if r < self.hub_radius_m || r > self.tip_radius_m {
            return vec![0.0, 0.0, 0.0];
        }

        let (axial_loading, tangential_loading) = self.loading_at(r, z);
        let axial_loading = axial_loading * self.blade_count_scale;
        let tangential_loading = tangential_loading * self.blade_count_scale;
        if axial_loading <= 1.0e-8 && tangential_loading.abs() <= 1.0e-8 {
            return vec![0.0, 0.0, 0.0];
        }

        let inv_r = r.max(f64::EPSILON).recip();
        let tx = -y * inv_r;
        let ty = x * inv_r;
        let fluid_tangential_m_s =
            ACTUATOR_INDUCED_VELOCITY_GAIN * (local_velocity[0] * tx + local_velocity[1] * ty);
        let tip_speed_m_s = (self.omega_rad_s.abs() * self.tip_radius_m).max(0.25);
        let velocity_cap_m_s = 1.5 * tip_speed_m_s;
        let relative_tangential_m_s = (self.omega_rad_s * r - fluid_tangential_m_s)
            .clamp(-velocity_cap_m_s, velocity_cap_m_s);
        let axial_velocity_m_s = (ACTUATOR_INDUCED_VELOCITY_GAIN * local_velocity[2])
            .clamp(-velocity_cap_m_s, velocity_cap_m_s);
        let relative_speed_sq = relative_tangential_m_s * relative_tangential_m_s
            + axial_velocity_m_s * axial_velocity_m_s;
        let dynamic_pressure_pa = 0.5 * self.air_density_kg_m3 * relative_speed_sq;
        let axial_force_n_m3 = 0.18 * dynamic_pressure_pa * axial_loading / self.axial_influence_m;
        let tangential_force_n_m3 = 0.12
            * dynamic_pressure_pa
            * tangential_loading
            * tangential_direction(relative_tangential_m_s, tip_speed_m_s)
            / self.axial_influence_m;

        vec![
            tangential_force_n_m3 * tx,
            tangential_force_n_m3 * ty,
            // CFD domain labels min-z as inlet and max-z as outlet.
            axial_force_n_m3,
        ]
    }

    /// Local Jacobian of the actuator force with respect to the fluid
    /// velocity.  The force law contains explicit clamping at the declared
    /// velocity cap; the derivative therefore follows the active branch and
    /// is zero on a saturated branch, without differentiating the global CFD
    /// solve.
    fn force_jacobian_at(&self, coords: [f64; 3], local_velocity: [f64; 3]) -> [[f64; 3]; 3] {
        let x = coords[0];
        let y = coords[1];
        let z = coords[2];
        let r = x.hypot(y);
        if r < self.hub_radius_m || r > self.tip_radius_m {
            return [[0.0; 3]; 3];
        }
        let (axial_loading, tangential_loading) = self.loading_at(r, z);
        let axial_loading = axial_loading * self.blade_count_scale;
        let tangential_loading = tangential_loading * self.blade_count_scale;
        if axial_loading <= 1.0e-8 && tangential_loading.abs() <= 1.0e-8 {
            return [[0.0; 3]; 3];
        }

        let inv_r = r.max(f64::EPSILON).recip();
        let tangent = [-y * inv_r, x * inv_r, 0.0];
        let gain = ACTUATOR_INDUCED_VELOCITY_GAIN;
        let tip_speed_m_s = (self.omega_rad_s.abs() * self.tip_radius_m).max(0.25);
        let velocity_cap_m_s = 1.5 * tip_speed_m_s;
        let raw_tangential = self.omega_rad_s * r
            - gain * (local_velocity[0] * tangent[0] + local_velocity[1] * tangent[1]);
        let raw_axial = gain * local_velocity[2];
        let tangential = raw_tangential.clamp(-velocity_cap_m_s, velocity_cap_m_s);
        let axial = raw_axial.clamp(-velocity_cap_m_s, velocity_cap_m_s);
        let tangential_derivative = if raw_tangential.abs() < velocity_cap_m_s {
            [-gain * tangent[0], -gain * tangent[1], 0.0]
        } else {
            [0.0; 3]
        };
        let axial_derivative = if raw_axial.abs() < velocity_cap_m_s {
            [0.0, 0.0, gain]
        } else {
            [0.0; 3]
        };
        let dynamic_pressure_pa =
            0.5 * self.air_density_kg_m3 * (tangential * tangential + axial * axial);
        let direction = tangential_direction(tangential, tip_speed_m_s);
        let direction_derivative = tangential_direction_derivative(tangential, tip_speed_m_s);
        let mut jacobian = [[0.0; 3]; 3];
        for input in 0..3 {
            let dynamic_pressure_derivative = self.air_density_kg_m3
                * (tangential * tangential_derivative[input] + axial * axial_derivative[input]);
            let axial_derivative_force =
                0.18 * axial_loading * dynamic_pressure_derivative / self.axial_influence_m;
            // The direction factor is piecewise constant under the
            // signum law, so its derivative contributes only under the
            // smoothed law (through `direction_derivative`).
            let tangential_derivative_force = 0.12
                * tangential_loading
                * (dynamic_pressure_derivative * direction
                    + dynamic_pressure_pa * direction_derivative * tangential_derivative[input])
                / self.axial_influence_m;
            jacobian[0][input] = tangential_derivative_force * tangent[0];
            jacobian[1][input] = tangential_derivative_force * tangent[1];
            jacobian[2][input] = axial_derivative_force;
        }
        jacobian
    }

    /// Design derivative of the actuator force at fixed coordinates and fluid
    /// velocity.  This is analytic on the active loading/clamp branches; the
    /// derivative of a saturated clamp is zero.  It intentionally excludes
    /// mesh motion and immersed-solid indicator derivatives, which are
    /// outside the frozen-topology contract.
    fn force_position_jacobian_at(
        &self,
        coords: [f64; 3],
        local_velocity: [f64; 3],
    ) -> [[f64; 3]; 3] {
        let x = coords[0];
        let y = coords[1];
        let z = coords[2];
        let radius = x.hypot(y);
        if radius <= self.hub_radius_m || radius >= self.tip_radius_m {
            return [[0.0; 3]; 3];
        }
        let (
            axial_loading,
            d_axial_dr,
            d_axial_dz,
            tangential_loading,
            d_tangential_dr,
            d_tangential_dz,
        ) = self.loading_with_spatial_derivative(radius, z);
        let axial_loading = axial_loading * self.blade_count_scale;
        let d_axial_dr = d_axial_dr * self.blade_count_scale;
        let d_axial_dz = d_axial_dz * self.blade_count_scale;
        let tangential_loading = tangential_loading * self.blade_count_scale;
        let d_tangential_dr = d_tangential_dr * self.blade_count_scale;
        let d_tangential_dz = d_tangential_dz * self.blade_count_scale;
        if axial_loading <= 1.0e-8 && tangential_loading.abs() <= 1.0e-8 {
            return [[0.0; 3]; 3];
        }

        let inv_radius = radius.max(f64::EPSILON).recip();
        let tangent = [-y * inv_radius, x * inv_radius, 0.0];
        let tangent_gradient = [
            [x * y * inv_radius.powi(3), -x * x * inv_radius.powi(3), 0.0],
            [y * y * inv_radius.powi(3), -x * y * inv_radius.powi(3), 0.0],
            [0.0, 0.0, 0.0],
        ];
        let gain = ACTUATOR_INDUCED_VELOCITY_GAIN;
        let tip_speed_m_s = (self.omega_rad_s.abs() * self.tip_radius_m).max(0.25);
        let velocity_cap_m_s = 1.5 * tip_speed_m_s;
        let fluid_tangential_m_s =
            gain * (local_velocity[0] * tangent[0] + local_velocity[1] * tangent[1]);
        let raw_tangential = self.omega_rad_s * radius - fluid_tangential_m_s;
        let raw_axial = gain * local_velocity[2];
        let tangential = raw_tangential.clamp(-velocity_cap_m_s, velocity_cap_m_s);
        let axial = raw_axial.clamp(-velocity_cap_m_s, velocity_cap_m_s);
        let tangential_active = raw_tangential.abs() < velocity_cap_m_s;
        let tangential_sign = tangential.signum();
        let mut d_tangential = [0.0; 3];
        let d_axial = [0.0; 3];
        let radial_gradient = [x * inv_radius, y * inv_radius, 0.0];
        for direction in 0..3 {
            let d_fluid_tangential = gain
                * (local_velocity[0] * tangent_gradient[0][direction]
                    + local_velocity[1] * tangent_gradient[1][direction]);
            if tangential_active {
                d_tangential[direction] =
                    self.omega_rad_s * radial_gradient[direction] - d_fluid_tangential;
            }
        }
        let dynamic_pressure =
            0.5 * self.air_density_kg_m3 * (tangential * tangential + axial * axial);
        let mut d_dynamic_pressure = [0.0; 3];
        for direction in 0..3 {
            d_dynamic_pressure[direction] = self.air_density_kg_m3
                * (tangential * d_tangential[direction] + axial * d_axial[direction]);
        }
        let axial_scale = 0.18 * axial_loading / self.axial_influence_m;
        let tangential_scale = 0.12 * tangential_loading * tangential_sign / self.axial_influence_m;
        let tangential_force = tangential_scale * dynamic_pressure;
        let mut jacobian = [[0.0; 3]; 3];
        for direction in 0..3 {
            let d_axial_force = 0.18 / self.axial_influence_m
                * (d_axial_dr * radial_gradient[direction]
                    + d_axial_dz * if direction == 2 { 1.0 } else { 0.0 })
                * dynamic_pressure
                + axial_scale * d_dynamic_pressure[direction];
            let d_tangential_force = 0.12 / self.axial_influence_m
                * (d_tangential_dr * radial_gradient[direction]
                    + d_tangential_dz * if direction == 2 { 1.0 } else { 0.0 })
                * tangential_sign
                * dynamic_pressure
                + tangential_scale * d_dynamic_pressure[direction];
            jacobian[0][direction] =
                d_tangential_force * tangent[0] + tangential_force * tangent_gradient[0][direction];
            jacobian[1][direction] =
                d_tangential_force * tangent[1] + tangential_force * tangent_gradient[1][direction];
            jacobian[2][direction] = d_axial_force;
        }
        jacobian
    }

    fn force_design_jacobian_at(
        &self,
        coords: [f64; 3],
        local_velocity: [f64; 3],
    ) -> [[f64; DESIGN_VARIABLE_COUNT]; 3] {
        let mut jacobian = [[0.0; DESIGN_VARIABLE_COUNT]; 3];
        let x = coords[0];
        let y = coords[1];
        let z = coords[2];
        let r = x.hypot(y);
        if r < self.hub_radius_m || r > self.tip_radius_m {
            return jacobian;
        }
        let (axial_loading, axial_derivative, tangential_loading, tangential_derivative) =
            self.loading_with_design_derivative(r, z);
        let axial_loading = axial_loading * self.blade_count_scale;
        let tangential_loading = tangential_loading * self.blade_count_scale;
        if axial_loading <= 1.0e-8 && tangential_loading.abs() <= 1.0e-8 {
            return jacobian;
        }

        let inv_r = r.max(f64::EPSILON).recip();
        let tx = -y * inv_r;
        let ty = x * inv_r;
        let fluid_tangential_m_s =
            ACTUATOR_INDUCED_VELOCITY_GAIN * (local_velocity[0] * tx + local_velocity[1] * ty);
        let tip_speed_m_s = (self.omega_rad_s.abs() * self.tip_radius_m).max(0.25);
        let velocity_cap_m_s = 1.5 * tip_speed_m_s;
        let relative_tangential_m_s = (self.omega_rad_s * r - fluid_tangential_m_s)
            .clamp(-velocity_cap_m_s, velocity_cap_m_s);
        let axial_velocity_m_s = (ACTUATOR_INDUCED_VELOCITY_GAIN * local_velocity[2])
            .clamp(-velocity_cap_m_s, velocity_cap_m_s);
        let dynamic_pressure_pa = 0.5
            * self.air_density_kg_m3
            * (relative_tangential_m_s * relative_tangential_m_s
                + axial_velocity_m_s * axial_velocity_m_s);
        let axial_scale =
            0.18 * dynamic_pressure_pa * self.blade_count_scale / self.axial_influence_m;
        let tangential_scale =
            0.12 * dynamic_pressure_pa * relative_tangential_m_s.signum() * self.blade_count_scale
                / self.axial_influence_m;
        for design_index in 0..DESIGN_VARIABLE_COUNT {
            jacobian[0][design_index] = tangential_scale * tangential_derivative[design_index] * tx;
            jacobian[1][design_index] = tangential_scale * tangential_derivative[design_index] * ty;
            jacobian[2][design_index] = axial_scale * axial_derivative[design_index];
        }
        jacobian
    }

    fn loading_with_design_derivative(
        &self,
        radius_m: f64,
        z_m: f64,
    ) -> (
        f64,
        [f64; DESIGN_VARIABLE_COUNT],
        f64,
        [f64; DESIGN_VARIABLE_COUNT],
    ) {
        let mut weight_sum = 0.0;
        let mut axial_sum = 0.0;
        let mut tangential_sum = 0.0;
        let mut weight_derivative = [0.0; DESIGN_VARIABLE_COUNT];
        let mut axial_derivative_sum = [0.0; DESIGN_VARIABLE_COUNT];
        let mut tangential_derivative_sum = [0.0; DESIGN_VARIABLE_COUNT];
        for (sample, derivative) in self.samples.iter().zip(&self.sample_design_derivatives) {
            let radial_distance = (radius_m - sample.radius_m) / self.radial_influence_m;
            let axial_distance = (z_m - sample.z_m) / self.axial_influence_m;
            let kernel =
                (-(radial_distance * radial_distance + axial_distance * axial_distance)).exp();
            let weight = kernel * sample.area_weight;
            if weight <= 1.0e-10 {
                continue;
            }
            weight_sum += weight;
            axial_sum += weight * sample.axial_loading;
            tangential_sum += weight * sample.tangential_loading;
            for design_index in 0..DESIGN_VARIABLE_COUNT {
                let delta_radial = -derivative.radius_m[design_index] / self.radial_influence_m
                    - radial_distance * self.radial_influence_design_derivative[design_index]
                        / self.radial_influence_m;
                let delta_axial = -derivative.z_m[design_index] / self.axial_influence_m;
                let delta_kernel = kernel
                    * (-2.0 * radial_distance * delta_radial - 2.0 * axial_distance * delta_axial);
                let delta_weight = delta_kernel * sample.area_weight
                    + kernel * derivative.area_weight[design_index];
                weight_derivative[design_index] += delta_weight;
                axial_derivative_sum[design_index] += delta_weight * sample.axial_loading
                    + weight * derivative.axial_loading[design_index];
                tangential_derivative_sum[design_index] += delta_weight * sample.tangential_loading
                    + weight * derivative.tangential_loading[design_index];
            }
        }
        if weight_sum <= 1.0e-10 {
            return (
                self.fallback_loading(radius_m, z_m).0,
                [0.0; DESIGN_VARIABLE_COUNT],
                self.fallback_loading(radius_m, z_m).1,
                [0.0; DESIGN_VARIABLE_COUNT],
            );
        }
        let inverse_weight = weight_sum.recip();
        let axial = axial_sum * inverse_weight;
        let tangential = tangential_sum * inverse_weight;
        let mut axial_derivative = [0.0; DESIGN_VARIABLE_COUNT];
        let mut tangential_derivative = [0.0; DESIGN_VARIABLE_COUNT];
        for design_index in 0..DESIGN_VARIABLE_COUNT {
            axial_derivative[design_index] = (axial_derivative_sum[design_index]
                - axial * weight_derivative[design_index])
                * inverse_weight;
            tangential_derivative[design_index] = (tangential_derivative_sum[design_index]
                - tangential * weight_derivative[design_index])
                * inverse_weight;
        }
        (axial, axial_derivative, tangential, tangential_derivative)
    }

    fn loading_at(&self, radius_m: f64, z_m: f64) -> (f64, f64) {
        let mut weight_sum = 0.0;
        let mut axial_sum = 0.0;
        let mut tangential_sum = 0.0;
        for sample in &self.samples {
            let radial_distance = (radius_m - sample.radius_m) / self.radial_influence_m;
            let axial_distance = (z_m - sample.z_m) / self.axial_influence_m;
            let weight = (-(radial_distance * radial_distance + axial_distance * axial_distance))
                .exp()
                * sample.area_weight;
            if weight <= 1.0e-10 {
                continue;
            }

            weight_sum += weight;
            axial_sum += weight * sample.axial_loading;
            tangential_sum += weight * sample.tangential_loading;
        }

        if weight_sum > 1.0e-10 {
            (axial_sum / weight_sum, tangential_sum / weight_sum)
        } else {
            self.fallback_loading(radius_m, z_m)
        }
    }

    fn loading_with_spatial_derivative(
        &self,
        radius_m: f64,
        z_m: f64,
    ) -> (f64, f64, f64, f64, f64, f64) {
        let mut weight_sum = 0.0;
        let mut axial_sum = 0.0;
        let mut tangential_sum = 0.0;
        let mut d_weight_dr = 0.0;
        let mut d_weight_dz = 0.0;
        let mut d_axial_dr = 0.0;
        let mut d_axial_dz = 0.0;
        let mut d_tangential_dr = 0.0;
        let mut d_tangential_dz = 0.0;
        for sample in &self.samples {
            let radial_distance = (radius_m - sample.radius_m) / self.radial_influence_m;
            let axial_distance = (z_m - sample.z_m) / self.axial_influence_m;
            let kernel =
                (-(radial_distance * radial_distance + axial_distance * axial_distance)).exp();
            let weight = kernel * sample.area_weight;
            if weight <= 1.0e-10 {
                continue;
            }
            let weight_dr = weight * -2.0 * radial_distance / self.radial_influence_m;
            let weight_dz = weight * -2.0 * axial_distance / self.axial_influence_m;
            weight_sum += weight;
            axial_sum += weight * sample.axial_loading;
            tangential_sum += weight * sample.tangential_loading;
            d_weight_dr += weight_dr;
            d_weight_dz += weight_dz;
            d_axial_dr += weight_dr * sample.axial_loading;
            d_axial_dz += weight_dz * sample.axial_loading;
            d_tangential_dr += weight_dr * sample.tangential_loading;
            d_tangential_dz += weight_dz * sample.tangential_loading;
        }
        if weight_sum <= 1.0e-10 {
            let radial_span = (self.tip_radius_m - self.hub_radius_m).max(f64::EPSILON);
            let radial_fraction = (radius_m - self.hub_radius_m) / radial_span;
            let angle = std::f64::consts::PI * radial_fraction;
            let sine = angle.sin();
            let radial_shape = sine.max(0.0);
            let d_radial_shape_dr = if sine > 0.0 {
                std::f64::consts::PI / radial_span * angle.cos()
            } else {
                0.0
            };
            let axial_shape = (-(z_m / self.axial_influence_m).powi(2)).exp();
            let d_axial_shape_dz = axial_shape * -2.0 * z_m / self.axial_influence_m.powi(2);
            let chord_scale =
                (self.default_chord_m / BladeSectionControl::default().chord_m).clamp(0.25, 2.0);
            let value = radial_shape * axial_shape * chord_scale;
            return (
                value,
                d_radial_shape_dr * axial_shape * chord_scale,
                radial_shape * d_axial_shape_dz * chord_scale,
                value,
                d_radial_shape_dr * axial_shape * chord_scale,
                radial_shape * d_axial_shape_dz * chord_scale,
            );
        }
        let inverse_weight = weight_sum.recip();
        let axial = axial_sum * inverse_weight;
        let tangential = tangential_sum * inverse_weight;
        (
            axial,
            (d_axial_dr - axial * d_weight_dr) * inverse_weight,
            (d_axial_dz - axial * d_weight_dz) * inverse_weight,
            tangential,
            (d_tangential_dr - tangential * d_weight_dr) * inverse_weight,
            (d_tangential_dz - tangential * d_weight_dz) * inverse_weight,
        )
    }

    fn fallback_loading(&self, radius_m: f64, z_m: f64) -> (f64, f64) {
        let radial_fraction =
            (radius_m - self.hub_radius_m) / (self.tip_radius_m - self.hub_radius_m);
        let radial_shape = (std::f64::consts::PI * radial_fraction).sin().max(0.0);
        let axial_shape = (-(z_m / self.axial_influence_m).powi(2)).exp();
        let chord_scale =
            (self.default_chord_m / BladeSectionControl::default().chord_m).clamp(0.25, 2.0);
        let loading = radial_shape * axial_shape * chord_scale;

        (loading, loading)
    }
}

fn add_force_jacobian(target: &mut [[f64; 3]; 3], source: [[f64; 3]; 3]) {
    for (target_row, source_row) in target.iter_mut().zip(source) {
        for (target_value, source_value) in target_row.iter_mut().zip(source_row) {
            *target_value += source_value;
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct BladeActuatorSample {
    radius_m: f64,
    z_m: f64,
    axial_loading: f64,
    tangential_loading: f64,
    area_weight: f64,
}

impl BladeActuatorSample {
    fn from_panel(panel: &PcFanBladePanel, default_chord_m: f64) -> Self {
        let center = face_center(panel.vertices);
        let radius_m = center.x.hypot(center.y);
        let chord = panel_chord_vector(panel);
        let chord_length_m = point_norm(chord).max(1.0e-9);
        let tangent = tangential_unit(center);
        let tangential_pitch = point_dot(chord, tangent) / chord_length_m;
        let axial_pitch = (chord.z / chord_length_m).abs().clamp(0.0, 1.0);
        let radial_shape = (std::f64::consts::PI * panel.span_fraction).sin().max(0.0);
        let chord_scale = (chord_length_m / default_chord_m).clamp(0.25, 2.5);
        let thickness_scale = (panel.thickness_m / 0.0012).clamp(0.5, 2.0);
        let camber_proxy = (panel_chord_bow_m(panel) / chord_length_m).clamp(0.0, 0.30);
        let pitch_loading = (0.20 + 0.65 * axial_pitch + 0.45 * camber_proxy).clamp(0.05, 1.35);
        let axial_loading = radial_shape * chord_scale * thickness_scale.sqrt() * pitch_loading;
        let tangential_loading =
            axial_loading * (0.75 + 0.60 * tangential_pitch.clamp(-0.75, 0.75));

        Self {
            radius_m,
            z_m: center.z,
            axial_loading,
            tangential_loading,
            area_weight: panel.area.max(1.0e-9),
        }
    }
}

/// Forward-mode design dual used for the smooth actuator-disk geometry
/// sensitivity.  The fixed-size gradient keeps the production callback
/// allocation-free at quadrature points and makes the design ordering exactly
/// the same as `PcFanDesignVariables::to_vector`.
#[derive(Clone, Copy, Debug)]
struct DesignDual {
    value: f64,
    derivative: [f64; DESIGN_VARIABLE_COUNT],
}

impl DesignDual {
    fn constant(value: f64) -> Self {
        Self {
            value,
            derivative: [0.0; DESIGN_VARIABLE_COUNT],
        }
    }

    fn variable(value: f64, index: usize) -> Self {
        let mut derivative = [0.0; DESIGN_VARIABLE_COUNT];
        derivative[index] = 1.0;
        Self { value, derivative }
    }

    fn sqrt(self) -> Self {
        let root = self.value.sqrt();
        let scale = if root > f64::EPSILON { 0.5 / root } else { 0.0 };
        Self {
            value: root,
            derivative: self.derivative.map(|value| value * scale),
        }
    }

    fn sin_cos(self) -> (Self, Self) {
        let (sin, cos) = self.value.sin_cos();
        (
            Self {
                value: sin,
                derivative: self.derivative.map(|derivative| cos * derivative),
            },
            Self {
                value: cos,
                derivative: self.derivative.map(|derivative| -sin * derivative),
            },
        )
    }

    fn abs(self) -> Self {
        if self.value >= 0.0 { self } else { -self }
    }

    fn clamp(self, minimum: f64, maximum: f64) -> Self {
        if self.value <= minimum {
            Self::constant(minimum)
        } else if self.value >= maximum {
            Self::constant(maximum)
        } else {
            self
        }
    }

    fn max(self, minimum: f64) -> Self {
        if self.value <= minimum {
            Self::constant(minimum)
        } else {
            self
        }
    }
}

impl Add for DesignDual {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value + rhs.value,
            derivative: std::array::from_fn(|index| self.derivative[index] + rhs.derivative[index]),
        }
    }
}

impl Sub for DesignDual {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value - rhs.value,
            derivative: std::array::from_fn(|index| self.derivative[index] - rhs.derivative[index]),
        }
    }
}

impl Mul for DesignDual {
    type Output = Self;

    #[expect(
        clippy::suspicious_arithmetic_impl,
        reason = "forward-mode AD product rule: d(fg) = df*g + f*dg"
    )]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value * rhs.value,
            derivative: std::array::from_fn(|index| {
                self.derivative[index] * rhs.value + self.value * rhs.derivative[index]
            }),
        }
    }
}

impl Div for DesignDual {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let denominator = rhs.value * rhs.value;
        Self {
            value: self.value / rhs.value,
            derivative: std::array::from_fn(|index| {
                (self.derivative[index] * rhs.value - self.value * rhs.derivative[index])
                    / denominator.max(f64::EPSILON)
            }),
        }
    }
}

impl Neg for DesignDual {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            value: -self.value,
            derivative: self.derivative.map(|value| -value),
        }
    }
}

#[derive(Clone, Copy)]
struct DesignDualPoint {
    x: DesignDual,
    y: DesignDual,
    z: DesignDual,
}

fn design_dual_catmull(values: [DesignDual; 4], interpolation: f64) -> DesignDual {
    let t = DesignDual::constant(interpolation);
    let t2 = t * t;
    let t3 = t2 * t;
    DesignDual::constant(0.5)
        * (DesignDual::constant(2.0) * values[1]
            + (-values[0] + values[2]) * t
            + (DesignDual::constant(2.0) * values[0] - DesignDual::constant(5.0) * values[1]
                + DesignDual::constant(4.0) * values[2]
                - values[3])
                * t2
            + (-values[0] + DesignDual::constant(3.0) * values[1]
                - DesignDual::constant(3.0) * values[2]
                + values[3])
                * t3)
}

fn design_dual_section(design: &PcFanDesignVariables, span_fraction: f64) -> [DesignDual; 4] {
    let vector = design.to_vector();
    let scaled = span_fraction.clamp(0.0, 1.0) * (BLADE_SECTION_STATION_COUNT - 1) as f64;
    let i1 = scaled.floor() as usize;
    let i2 = (i1 + 1).min(BLADE_SECTION_STATION_COUNT - 1);
    let i0 = i1.saturating_sub(1);
    let i3 = (i2 + 1).min(BLADE_SECTION_STATION_COUNT - 1);
    let interpolation = scaled - i1 as f64;
    std::array::from_fn(|component| {
        design_dual_catmull(
            [
                DesignDual::variable(vector[10 + 4 * i0 + component], 10 + 4 * i0 + component),
                DesignDual::variable(vector[10 + 4 * i1 + component], 10 + 4 * i1 + component),
                DesignDual::variable(vector[10 + 4 * i2 + component], 10 + 4 * i2 + component),
                DesignDual::variable(vector[10 + 4 * i3 + component], 10 + 4 * i3 + component),
            ],
            interpolation,
        )
    })
}

fn design_dual_path(design: &PcFanDesignVariables, span_fraction: f64) -> [DesignDual; 2] {
    let vector = design.to_vector();
    let scaled = span_fraction.clamp(0.0, 1.0) * (BLADE_PATH_CONTROL_COUNT - 1) as f64;
    let i1 = scaled.floor() as usize;
    let i2 = (i1 + 1).min(BLADE_PATH_CONTROL_COUNT - 1);
    let i0 = i1.saturating_sub(1);
    let i3 = (i2 + 1).min(BLADE_PATH_CONTROL_COUNT - 1);
    let interpolation = scaled - i1 as f64;
    std::array::from_fn(|component| {
        design_dual_catmull(
            [
                DesignDual::variable(vector[2 * i0 + component], 2 * i0 + component),
                DesignDual::variable(vector[2 * i1 + component], 2 * i1 + component),
                DesignDual::variable(vector[2 * i2 + component], 2 * i2 + component),
                DesignDual::variable(vector[2 * i3 + component], 2 * i3 + component),
            ],
            interpolation,
        )
    })
}

fn design_dual_blade_point(
    config: &PcFanConfig,
    design: &PcFanDesignVariables,
    azimuth_rad: f64,
    span_fraction: f64,
    chord_fraction: f64,
) -> DesignDualPoint {
    let section = design_dual_section(design, span_fraction);
    let path = design_dual_path(design, span_fraction);
    let chord_centered = chord_fraction - 0.5;
    let camber_shape = (1.0 - (2.0 * chord_centered).powi(2)).max(0.0);
    let (sin_twist, cos_twist) =
        (section[3] * DesignDual::constant(std::f64::consts::PI / 180.0)).sin_cos();
    let local_chord = DesignDual::constant(chord_centered) * section[0];
    let tangential_offset =
        path[0] + local_chord * sin_twist + DesignDual::constant(camber_shape) * section[1];
    let axial_offset = path[1] + local_chord * cos_twist;
    let radial_start = BLADE_PATH_CONTROL_COUNT * 2 + BLADE_SECTION_STATION_COUNT * 4;
    let hub_radius = DesignDual::constant(config.dimensions.hub_radius_m)
        + DesignDual::variable(design.hub_radius_offset_m, radial_start);
    let tip_radius = DesignDual::constant(config.dimensions.blade_tip_radius_m)
        + DesignDual::variable(design.tip_radius_offset_m, radial_start + 1);
    let radius = hub_radius + DesignDual::constant(span_fraction) * (tip_radius - hub_radius);
    let (sin_azimuth, cos_azimuth) = azimuth_rad.sin_cos();
    DesignDualPoint {
        x: radius * DesignDual::constant(cos_azimuth)
            - tangential_offset * DesignDual::constant(sin_azimuth),
        y: radius * DesignDual::constant(sin_azimuth)
            + tangential_offset * DesignDual::constant(cos_azimuth),
        z: axial_offset,
    }
}

fn design_dual_sub(left: DesignDualPoint, right: DesignDualPoint) -> DesignDualPoint {
    DesignDualPoint {
        x: left.x - right.x,
        y: left.y - right.y,
        z: left.z - right.z,
    }
}

fn design_dual_cross(left: DesignDualPoint, right: DesignDualPoint) -> DesignDualPoint {
    DesignDualPoint {
        x: left.y * right.z - left.z * right.y,
        y: left.z * right.x - left.x * right.z,
        z: left.x * right.y - left.y * right.x,
    }
}

fn design_dual_norm(point: DesignDualPoint) -> DesignDual {
    (point.x * point.x + point.y * point.y + point.z * point.z).sqrt()
}

fn design_dual_average(left: DesignDualPoint, right: DesignDualPoint) -> DesignDualPoint {
    let half = DesignDual::constant(0.5);
    DesignDualPoint {
        x: (left.x + right.x) * half,
        y: (left.y + right.y) * half,
        z: (left.z + right.z) * half,
    }
}

fn design_dual_triangle_area(
    a: DesignDualPoint,
    b: DesignDualPoint,
    c: DesignDualPoint,
) -> DesignDual {
    DesignDual::constant(0.5)
        * design_dual_norm(design_dual_cross(
            design_dual_sub(b, a),
            design_dual_sub(c, a),
        ))
}

impl BladeActuatorSampleDerivative {
    fn from_panel(
        config: &PcFanConfig,
        design: &PcFanDesignVariables,
        blade_index: usize,
        panel_index: usize,
        default_chord_m: f64,
    ) -> Self {
        let radial_segments = config.mesh.radial_segments.max(1);
        let chord_segments = config.mesh.chord_segments.max(1);
        let radial_index = panel_index / chord_segments;
        let chord_index = panel_index % chord_segments;
        let r0 = radial_index as f64 / radial_segments as f64;
        let r1 = (radial_index + 1) as f64 / radial_segments as f64;
        let c0 = chord_index as f64 / chord_segments as f64;
        let c1 = (chord_index + 1) as f64 / chord_segments as f64;
        let azimuth = std::f64::consts::TAU * blade_index as f64 / config.blade_count.max(1) as f64;
        let vertices = [
            design_dual_blade_point(config, design, azimuth, r0, c0),
            design_dual_blade_point(config, design, azimuth, r1, c0),
            design_dual_blade_point(config, design, azimuth, r1, c1),
            design_dual_blade_point(config, design, azimuth, r0, c1),
        ];
        let center = design_dual_average(
            design_dual_average(vertices[0], vertices[1]),
            design_dual_average(vertices[2], vertices[3]),
        );
        let radius = (center.x * center.x + center.y * center.y).sqrt();
        let leading_mid = design_dual_average(vertices[0], vertices[1]);
        let trailing_mid = design_dual_average(vertices[3], vertices[2]);
        let chord = design_dual_sub(trailing_mid, leading_mid);
        let chord_length = design_dual_norm(chord).max(1.0e-9);
        let tangent_x = -center.y / radius.max(f64::EPSILON);
        let tangent_y = center.x / radius.max(f64::EPSILON);
        let tangential_pitch = (chord.x * tangent_x + chord.y * tangent_y) / chord_length;
        let axial_pitch = (chord.z / chord_length).abs().clamp(0.0, 1.0);
        let radial_shape = (std::f64::consts::PI * 0.5 * (r0 + r1)).sin().max(0.0);
        let chord_scale = (chord_length / DesignDual::constant(default_chord_m)).clamp(0.25, 2.5);
        let span_fraction = 0.5 * (r0 + r1);
        let thickness_scale = (design_dual_section(design, span_fraction)[2]
            / DesignDual::constant(0.0012))
        .max(0.0)
        .clamp(0.5, 2.0);
        let chord_mid = design_dual_average(leading_mid, trailing_mid);
        let camber = design_dual_norm(design_dual_sub(center, chord_mid));
        let camber_proxy = (camber / chord_length).clamp(0.0, 0.30);
        let pitch_loading = (DesignDual::constant(0.20)
            + DesignDual::constant(0.65) * axial_pitch
            + DesignDual::constant(0.45) * camber_proxy)
            .clamp(0.05, 1.35);
        let axial_loading = DesignDual::constant(radial_shape)
            * chord_scale
            * thickness_scale.sqrt()
            * pitch_loading;
        let tangential_loading = axial_loading
            * (DesignDual::constant(0.75)
                + DesignDual::constant(0.60) * tangential_pitch.clamp(-0.75, 0.75));
        let area = (design_dual_triangle_area(vertices[0], vertices[1], vertices[2])
            + design_dual_triangle_area(vertices[0], vertices[2], vertices[3]))
        .max(1.0e-9);
        Self {
            radius_m: radius.derivative,
            z_m: center.z.derivative,
            axial_loading: axial_loading.derivative,
            tangential_loading: tangential_loading.derivative,
            area_weight: area.derivative,
        }
    }
}

fn boundary_tolerance(domain: &PcFanCfdDomain, nx: usize, ny: usize, nz: usize) -> f64 {
    let dx = (domain.max_corner.x - domain.min_corner.x) / nx as f64;
    let dy = (domain.max_corner.y - domain.min_corner.y) / ny as f64;
    let dz = (domain.max_corner.z - domain.min_corner.z) / nz as f64;
    1.0e-8 * dx.min(dy).min(dz).max(1.0)
}

fn pressure_pin_dof<PSpace: FunctionSpace>(
    mesh: &sonium_fem::mesh::Mesh,
    p_space: &PSpace,
) -> usize {
    let coordinates = p_space.dof_coordinates(mesh).unwrap_or_default();
    let centroid = mesh.nodes.iter().fold([0.0; 3], |mut sum, point| {
        sum[0] += point.x;
        sum[1] += point.y;
        sum[2] += point.z;
        sum
    });
    let centroid = centroid.map(|value| value / mesh.num_nodes().max(1) as f64);
    coordinates
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| {
            let left_distance = (0..mesh.dimension)
                .map(|direction| (left[direction] - centroid[direction]).powi(2))
                .sum::<f64>();
            let right_distance = (0..mesh.dimension)
                .map(|direction| (right[direction] - centroid[direction]).powi(2))
                .sum::<f64>();
            left_distance.total_cmp(&right_distance)
        })
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn solid_boundary_influence_m(domain: &PcFanCfdDomain, nx: usize, ny: usize, nz: usize) -> f64 {
    let dx = (domain.max_corner.x - domain.min_corner.x) / nx as f64;
    let dy = (domain.max_corner.y - domain.min_corner.y) / ny as f64;
    let dz = (domain.max_corner.z - domain.min_corner.z) / nz as f64;

    SOLID_BOUNDARY_INFLUENCE_CELLS * dx.min(dy).min(dz).max(f64::EPSILON)
}

fn is_side_wall(domain: &PcFanCfdDomain, x: f64, y: f64, tolerance: f64) -> bool {
    (x - domain.min_corner.x).abs() <= tolerance
        || (x - domain.max_corner.x).abs() <= tolerance
        || (y - domain.min_corner.y).abs() <= tolerance
        || (y - domain.max_corner.y).abs() <= tolerance
}

fn rotating_wall_velocity_m_s(omega_rad_s: f64, point: Point3) -> [f64; 3] {
    [-omega_rad_s * point.y, omega_rad_s * point.x, 0.0]
}

fn panel_chord_vector(panel: &PcFanBladePanel) -> Point3 {
    let leading_mid = average_points(panel.vertices[0], panel.vertices[1]);
    let trailing_mid = average_points(panel.vertices[3], panel.vertices[2]);
    trailing_mid.sub(leading_mid)
}

fn panel_chord_bow_m(panel: &PcFanBladePanel) -> f64 {
    let leading_mid = average_points(panel.vertices[0], panel.vertices[1]);
    let trailing_mid = average_points(panel.vertices[3], panel.vertices[2]);
    let center = face_center(panel.vertices);
    let chord_mid = average_points(leading_mid, trailing_mid);

    point_distance(center, chord_mid)
}

fn tangential_unit(point: Point3) -> Point3 {
    let inv_r = point.x.hypot(point.y).max(f64::EPSILON).recip();
    Point3::new(-point.y * inv_r, point.x * inv_r, 0.0)
}

fn average_points(a: Point3, b: Point3) -> Point3 {
    Point3::new(0.5 * (a.x + b.x), 0.5 * (a.y + b.y), 0.5 * (a.z + b.z))
}

fn point_dot(a: Point3, b: Point3) -> f64 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

fn point_norm(point: Point3) -> f64 {
    (point.x * point.x + point.y * point.y + point.z * point.z).sqrt()
}

fn point_distance(a: Point3, b: Point3) -> f64 {
    point_norm(a.sub(b))
}

fn face_center(vertices: [Point3; 4]) -> Point3 {
    Point3::new(
        0.25 * vertices.iter().map(|point| point.x).sum::<f64>(),
        0.25 * vertices.iter().map(|point| point.y).sum::<f64>(),
        0.25 * vertices.iter().map(|point| point.z).sum::<f64>(),
    )
}

fn distance_squared(a: Point3, b: Point3) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;

    dx * dx + dy * dy + dz * dz
}

fn velocity_vectors(velocity: &[f64], node_count: usize) -> Vec<[f64; 3]> {
    (0..node_count)
        .map(|node| {
            [
                velocity.get(3 * node).copied().unwrap_or(0.0),
                velocity.get(3 * node + 1).copied().unwrap_or(0.0),
                velocity.get(3 * node + 2).copied().unwrap_or(0.0),
            ]
        })
        .collect()
}

/// Triangular facets of tetrahedral elements lying in the plane `z`,
/// each returned once. A facet shared by two elements is interior and
/// cannot lie in a domain boundary plane, so no sharing test is needed
/// beyond deduplication.
fn boundary_plane_facets(
    nodes: &[Point3],
    elements: &[sonium_fem::mesh::Element],
    z: f64,
    tolerance: f64,
) -> Vec<[usize; 3]> {
    let mut seen = std::collections::HashSet::new();
    let mut facets = Vec::new();
    for element in elements {
        if element.element_type != sonium_fem::mesh::ElementType::Tetrahedron
            || element.nodes.len() != 4
        {
            continue;
        }
        let n = &element.nodes;
        for face in [
            [n[0], n[1], n[2]],
            [n[0], n[1], n[3]],
            [n[0], n[2], n[3]],
            [n[1], n[2], n[3]],
        ] {
            if face
                .iter()
                .all(|&node| (nodes[node].z - z).abs() <= tolerance)
            {
                let mut key = face;
                key.sort_unstable();
                if seen.insert(key) {
                    facets.push(face);
                }
            }
        }
    }
    facets
}

fn triangle_area_m2(a: Point3, b: Point3, c: Point3) -> f64 {
    let ab = [b.x - a.x, b.y - a.y, b.z - a.z];
    let ac = [c.x - a.x, c.y - a.y, c.z - a.z];
    let cross = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt()
}

fn boundary_airflow_cfm(
    nodes: &[Point3],
    elements: &[sonium_fem::mesh::Element],
    velocity_m_s: &[[f64; 3]],
    z: f64,
    area_m2: f64,
    tolerance: f64,
) -> f64 {
    // Exact integration of the piecewise-linear axial velocity over the
    // boundary facets: on each triangle the integral is the facet area
    // times the mean nodal value. The separable nodal trapezoid this
    // replaces does not integrate the triangular facet layout of the
    // barycentric mesh (it manufactured double-digit inlet/outlet
    // disagreement on fields whose exact facet fluxes balance).
    let facets = boundary_plane_facets(nodes, elements, z, tolerance);
    if !facets.is_empty() {
        let mut flux_m3_s = 0.0;
        for [a, b, c] in facets {
            flux_m3_s += triangle_area_m2(nodes[a], nodes[b], nodes[c])
                * (velocity_m_s[a][2] + velocity_m_s[b][2] + velocity_m_s[c][2])
                / 3.0;
        }
        return flux_m3_s * M3S_TO_CFM;
    }
    boundary_airflow_cfm_trapezoid(nodes, velocity_m_s, z, area_m2, tolerance)
}

/// Nodal-trapezoid boundary flux, retained only as the fallback for
/// meshes without tetrahedral boundary facets at the plane.
fn boundary_airflow_cfm_trapezoid(
    nodes: &[Point3],
    velocity_m_s: &[[f64; 3]],
    z: f64,
    area_m2: f64,
    tolerance: f64,
) -> f64 {
    let samples = nodes
        .iter()
        .zip(velocity_m_s.iter())
        .filter_map(|(node, velocity)| {
            ((node.z - z).abs() <= tolerance).then_some((node.x, node.y, velocity[2]))
        })
        .collect::<Vec<_>>();

    if samples.is_empty() {
        return 0.0;
    }

    let x_coords = sorted_unique_coordinates(samples.iter().map(|(x, _, _)| *x), tolerance);
    let y_coords = sorted_unique_coordinates(samples.iter().map(|(_, y, _)| *y), tolerance);
    if x_coords.len() < 2 || y_coords.len() < 2 {
        let mean_velocity = samples.iter().map(|(_, _, vz)| vz).sum::<f64>() / samples.len() as f64;
        return mean_velocity * area_m2 * M3S_TO_CFM;
    }

    let mut flux_m3_s = 0.0;
    let mut integrated_area_m2 = 0.0;
    for (x, y, vz) in samples {
        let weight_m2 = coordinate_quadrature_weight(x, &x_coords, tolerance)
            * coordinate_quadrature_weight(y, &y_coords, tolerance);
        flux_m3_s += vz * weight_m2;
        integrated_area_m2 += weight_m2;
    }

    if integrated_area_m2 <= f64::EPSILON {
        0.0
    } else {
        flux_m3_s * area_m2 / integrated_area_m2 * M3S_TO_CFM
    }
}

/// Return the CFM coefficient multiplying each nodal axial velocity in the
/// production boundary-flow quadrature.  Keeping this beside
/// [`boundary_airflow_cfm`] makes the analytic objective derivative use the
/// exact same coordinate weights and degenerate-grid fallback.
fn boundary_airflow_state_weights(
    nodes: &[Point3],
    elements: &[sonium_fem::mesh::Element],
    z: f64,
    area_m2: f64,
    tolerance: f64,
) -> Vec<f64> {
    // Exact facet weights matching [`boundary_airflow_cfm`]: each
    // boundary node carries one third of the area of every boundary
    // facet it belongs to.
    let facets = boundary_plane_facets(nodes, elements, z, tolerance);
    if !facets.is_empty() {
        let mut weights = vec![0.0; nodes.len()];
        for [a, b, c] in facets {
            let weight =
                triangle_area_m2(nodes[a], nodes[b], nodes[c]) / 3.0 * M3S_TO_CFM;
            weights[a] += weight;
            weights[b] += weight;
            weights[c] += weight;
        }
        return weights;
    }
    let mut weights = vec![0.0; nodes.len()];
    let samples = nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| {
            ((node.z - z).abs() <= tolerance).then_some((index, node.x, node.y))
        })
        .collect::<Vec<_>>();
    if samples.is_empty() {
        return weights;
    }

    let x_coords = sorted_unique_coordinates(samples.iter().map(|(_, x, _)| *x), tolerance);
    let y_coords = sorted_unique_coordinates(samples.iter().map(|(_, _, y)| *y), tolerance);
    if x_coords.len() < 2 || y_coords.len() < 2 {
        let coefficient = area_m2 * M3S_TO_CFM / samples.len() as f64;
        for (index, _, _) in samples {
            weights[index] = coefficient;
        }
        return weights;
    }

    let raw_weights = samples
        .iter()
        .map(|(_, x, y)| {
            coordinate_quadrature_weight(*x, &x_coords, tolerance)
                * coordinate_quadrature_weight(*y, &y_coords, tolerance)
        })
        .collect::<Vec<_>>();
    let integrated_area_m2 = raw_weights.iter().sum::<f64>();
    if integrated_area_m2 <= f64::EPSILON {
        return weights;
    }
    let scale = area_m2 / integrated_area_m2 * M3S_TO_CFM;
    for ((index, _, _), raw_weight) in samples.into_iter().zip(raw_weights) {
        weights[index] = raw_weight * scale;
    }
    weights
}

fn boundary_average_pressure_state_weights(nodes: &[Point3], z: f64, tolerance: f64) -> Vec<f64> {
    let mut weights = vec![0.0; nodes.len()];
    let count = nodes
        .iter()
        .filter(|node| (node.z - z).abs() <= tolerance)
        .count();
    if count == 0 {
        return weights;
    }
    let weight = 1.0 / count as f64;
    for (index, node) in nodes.iter().enumerate() {
        if (node.z - z).abs() <= tolerance {
            weights[index] = weight;
        }
    }
    weights
}

fn sorted_unique_coordinates(coordinates: impl Iterator<Item = f64>, tolerance: f64) -> Vec<f64> {
    let mut values = coordinates.collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    values.dedup_by(|a, b| (*a - *b).abs() <= tolerance);
    values
}

fn coordinate_quadrature_weight(value: f64, coordinates: &[f64], tolerance: f64) -> f64 {
    let index = coordinates
        .iter()
        .position(|coordinate| (*coordinate - value).abs() <= tolerance)
        .unwrap_or_else(|| {
            coordinates
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| (*a - value).abs().total_cmp(&(*b - value).abs()))
                .map(|(index, _)| index)
                .unwrap_or(0)
        });

    match (index, coordinates.len()) {
        (_, 0 | 1) => 1.0,
        (0, _) => 0.5 * (coordinates[1] - coordinates[0]).abs(),
        (i, len) if i + 1 == len => 0.5 * (coordinates[i] - coordinates[i - 1]).abs(),
        (i, _) => 0.5 * (coordinates[i + 1] - coordinates[i - 1]).abs(),
    }
}

fn relative_change(current: f64, previous: f64) -> f64 {
    let scale = current.abs().max(previous.abs()).max(1.0e-12);
    (current - previous).abs() / scale
}

fn average_boundary_pressure(nodes: &[Point3], pressure: &[f64], z: f64, tolerance: f64) -> f64 {
    let mut sum_pressure = 0.0;
    let mut count = 0usize;
    for (node, pressure) in nodes.iter().zip(pressure.iter()) {
        if (node.z - z).abs() <= tolerance {
            sum_pressure += pressure;
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        sum_pressure / count as f64
    }
}

fn average_boundary_axial_velocity(
    nodes: &[Point3],
    velocity_m_s: &[[f64; 3]],
    z: f64,
    tolerance: f64,
) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for (node, velocity) in nodes.iter().zip(velocity_m_s.iter()) {
        if (node.z - z).abs() <= tolerance {
            sum += velocity[2];
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}

#[cfg(test)]
mod state_transfer_runner_tests {
    use super::*;

    fn transfer_runner_config(cells: usize) -> PcFanConfig {
        let mut config = PcFanConfig::default_120mm();
        config.cfd.flow_model = PcFanFlowModel::IncompressibleCfdFull;
        config.cfd.backend = PcFanCfdBackend::Cpu;
        config.cfd.cells_x = cells;
        config.cfd.cells_y = cells;
        config.cfd.cells_z = cells;
        config.cfd.max_picard_iterations = 1;
        config.cfd.nonlinear_tolerance = 2.0;
        config.cfd.max_relative_mass_imbalance = 10.0;
        config.cfd.max_relative_momentum_imbalance = 10.0;
        config.cfd.max_relative_pressure_closure = 10.0;
        config.cfd.max_relative_energy_imbalance = 10.0;
        config
    }

    #[test]
    fn state_transfer_runner_retains_production_iteration_budget() {
        let source_config = transfer_runner_config(1);
        let design = source_config.default_design_variables();
        let geometry = source_config.build_geometry(&design);
        let domain = geometry.build_cfd_domain(&source_config);
        let source_mesh = pc_fan_cfd_mesh_for_domain(&source_config, &domain);
        let mut target_config = transfer_runner_config(2);
        target_config.cfd.max_picard_iterations = 4;
        target_config.cfd.nonlinear_tolerance = 1e-7;
        target_config.cfd.use_direct_solver = true;
        let target = solve_pc_fan_cfd_mean_flow_with_state_transfer(
            &target_config,
            &design,
            &source_mesh,
            &vec![[0.0; 3]; source_mesh.num_nodes()],
            &vec![0.0; source_mesh.num_nodes()],
        )
        .expect("valid transferred state must enter the production solve");
        assert_eq!(
            target.iterations, 4,
            "warm start must retain the requested solve budget"
        );
        assert!(target.residual.is_finite());
    }

    #[test]
    fn state_transfer_runner_rejects_malformed_source_before_cfd_work() {
        let config = transfer_runner_config(2);
        let design = config.default_design_variables();
        let geometry = config.build_geometry(&design);
        let domain = geometry.build_cfd_domain(&config);
        let source_mesh = pc_fan_cfd_mesh_for_domain(&config, &domain);

        let error = solve_pc_fan_cfd_mean_flow_with_state_transfer(
            &config,
            &design,
            &source_mesh,
            &[],
            &[],
        )
        .expect_err("malformed source state must fail before running CFD");

        assert!(matches!(
            error,
            PcFanCfdStateTransferError::SourceVelocityCount { actual: 0, .. }
        ));
    }

    #[test]
    fn state_transfer_runner_consumes_changed_generated_topology() {
        let source_config = transfer_runner_config(1);
        let source_design = source_config.default_design_variables();
        let source_geometry = source_config.build_geometry(&source_design);
        let source_domain = source_geometry.build_cfd_domain(&source_config);
        let source_mesh = pc_fan_cfd_mesh_for_domain(&source_config, &source_domain);
        let source_velocity = vec![[0.0; 3]; source_mesh.num_nodes()];
        let source_pressure = vec![0.0; source_mesh.num_nodes()];

        let target_config = transfer_runner_config(2);
        let target_design = target_config.default_design_variables();
        let target = solve_pc_fan_cfd_mean_flow_with_state_transfer(
            &target_config,
            &target_design,
            &source_mesh,
            &source_velocity,
            &source_pressure,
        )
        .expect("coarse-to-fine production state handoff should be admitted");

        assert_eq!(target.fem_node_count, target.nodes.len());
        assert_eq!(target.velocity_m_s.len(), target.nodes.len());
        assert_eq!(target.pressure_pa.len(), target.nodes.len());
        assert!(
            target
                .velocity_m_s
                .iter()
                .flatten()
                .all(|value| value.is_finite())
        );
        assert!(target.pressure_pa.iter().all(|value| value.is_finite()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_convergence_requires_airflow_and_pressure_agreement() {
        assert!(mesh_level_converged(
            2,
            2,
            true,
            Some(0.02),
            Some(0.03),
            0.05,
        ));
        assert!(!mesh_level_converged(
            2,
            2,
            true,
            Some(0.02),
            Some(0.06),
            0.05,
        ));
        assert!(!mesh_level_converged(2, 2, true, Some(0.02), None, 0.05));
        assert!(!mesh_level_converged(
            2,
            2,
            false,
            Some(0.02),
            Some(0.03),
            0.05,
        ));
    }

    #[test]
    fn mesh_level_residual_admission_is_finite_and_bounded() {
        assert!(mesh_level_residual_admitted(1.0e-4, 1.0e-3));
        assert!(mesh_level_residual_admitted(1.0e-3, 1.0e-3));
        assert!(!mesh_level_residual_admitted(1.1e-3, 1.0e-3));
        assert!(!mesh_level_residual_admitted(f64::NAN, 1.0e-3));
        assert!(!mesh_level_residual_admitted(1.0e-4, f64::NAN));
        assert!(!mesh_level_residual_admitted(1.0e-4, -1.0));
    }

    #[test]
    fn mesh_level_conservation_admission_is_finite_and_bounded() {
        let config = PcFanConfig::default_120mm();
        let conservation = PcFanCfdConservation {
            inlet_mass_flow_kg_s: 0.0,
            outlet_mass_flow_kg_s: 0.0,
            relative_mass_imbalance: 0.0,
            inlet_axial_momentum_flux_n: 0.0,
            outlet_axial_momentum_flux_n: 0.0,
            pressure_force_n: 0.0,
            body_force_n: 0.0,
            axial_momentum_residual_n: 0.0,
            relative_momentum_imbalance: 0.0,
            pressure_force_from_loading_n: 0.0,
            relative_pressure_closure: 0.0,
            pressure_power_w: 0.0,
            shaft_power_to_fluid_w: 0.0,
            relative_energy_imbalance: 0.0,
        };
        assert!(mesh_level_conservation_admitted(&config, conservation));

        let mut rejected = conservation;
        rejected.relative_momentum_imbalance = config.cfd.max_relative_momentum_imbalance + 1.0;
        assert!(!mesh_level_conservation_admitted(&config, rejected));

        rejected.relative_momentum_imbalance = f64::NAN;
        assert!(!mesh_level_conservation_admitted(&config, rejected));
    }

    #[test]
    fn direct_linear_solver_defaults_off_and_parses_legacy_configs() {
        assert!(!PcFanCfdConfig::default().use_direct_solver);
        let mut encoded =
            serde_json::to_value(PcFanCfdConfig::default()).expect("config serializes");
        encoded
            .as_object_mut()
            .expect("config is an object")
            .remove("use_direct_solver");
        let legacy: PcFanCfdConfig = serde_json::from_value(encoded).expect("legacy configs parse");
        assert!(!legacy.use_direct_solver);
    }

    #[test]
    fn admission_violation_reports_each_rejection_reason_in_solver_order() {
        let mut config = PcFanConfig::default_120mm();
        config.cfd.flow_model = PcFanFlowModel::IncompressibleCfdFull;
        config.cfd.backend = PcFanCfdBackend::Cpu;
        config.cfd.cells_x = 2;
        config.cfd.cells_y = 2;
        config.cfd.cells_z = 2;
        config.cfd.max_picard_iterations = 1;
        config.cfd.nonlinear_tolerance = 2.0;
        config.cfd.max_relative_mass_imbalance = 10.0;
        config.cfd.max_relative_momentum_imbalance = 10.0;
        config.cfd.max_relative_pressure_closure = 10.0;
        config.cfd.max_relative_energy_imbalance = 10.0;
        let design = config.default_design_variables();
        let template = solve_pc_fan_cfd_mean_flow(&config, &design);
        let mut admitted = template.clone();
        admitted.converged = true;
        admitted.iterations = 1;
        admitted.residual = 1.0e-4;
        admitted.conservation.relative_mass_imbalance = 0.0;
        admitted.conservation.relative_momentum_imbalance = 0.0;
        admitted.conservation.relative_pressure_closure = 0.0;
        admitted.conservation.relative_energy_imbalance = 0.0;
        admitted.backend_provenance =
            PcFanCfdBackendProvenance::from_dispatch(PcFanCfdBackend::Cpu, PcFanCfdBackend::Cpu);
        assert_eq!(
            pc_fan_cfd_first_admission_violation(&config, &admitted, 1.0e-3),
            None
        );

        // Backend admission is checked before the nonlinear state.
        let mut backend = admitted.clone();
        backend.backend_provenance =
            PcFanCfdBackendProvenance::from_dispatch(PcFanCfdBackend::Cuda, PcFanCfdBackend::Cpu);
        backend.converged = false;
        let mut strict = config.clone();
        strict.cfd.allow_host_fallback = false;
        assert_eq!(
            pc_fan_cfd_first_admission_violation(&strict, &backend, 1.0e-3),
            Some("backend-not-admitted")
        );

        // Early Picard exit means the inner linear solve failed; reaching
        // the budget without tolerance means stagnation.
        let mut linear = admitted.clone();
        linear.converged = false;
        linear.iterations = 0;
        assert_eq!(
            pc_fan_cfd_first_admission_violation(&config, &linear, 1.0e-3),
            Some("linear-solver-failure")
        );
        let mut stagnated = admitted.clone();
        stagnated.converged = false;
        stagnated.iterations = config.cfd.max_picard_iterations.max(1);
        assert_eq!(
            pc_fan_cfd_first_admission_violation(&config, &stagnated, 1.0e-3),
            Some("picard-stagnation")
        );

        let mut over_budget = admitted.clone();
        over_budget.residual = 2.0e-3;
        assert_eq!(
            pc_fan_cfd_first_admission_violation(&config, &over_budget, 1.0e-3),
            Some("residual-above-budget")
        );

        let mut mass = admitted.clone();
        mass.conservation.relative_mass_imbalance = config.cfd.max_relative_mass_imbalance * 2.0;
        assert_eq!(
            pc_fan_cfd_first_admission_violation(&config, &mass, 1.0e-3),
            Some("conservation-mass-imbalance")
        );
        let mut momentum = admitted.clone();
        momentum.conservation.relative_momentum_imbalance =
            config.cfd.max_relative_momentum_imbalance * 2.0;
        assert_eq!(
            pc_fan_cfd_first_admission_violation(&config, &momentum, 1.0e-3),
            Some("conservation-momentum-imbalance")
        );
        let mut pressure = admitted.clone();
        pressure.conservation.relative_pressure_closure =
            config.cfd.max_relative_pressure_closure * 2.0;
        assert_eq!(
            pc_fan_cfd_first_admission_violation(&config, &pressure, 1.0e-3),
            Some("conservation-pressure-closure")
        );
        let mut energy = admitted.clone();
        energy.conservation.relative_energy_imbalance =
            config.cfd.max_relative_energy_imbalance * 2.0;
        assert_eq!(
            pc_fan_cfd_first_admission_violation(&config, &energy, 1.0e-3),
            Some("conservation-energy-imbalance")
        );
    }

    #[test]
    fn direct_linear_solver_passes_odd_cell_oseen_systems() {
        // Exact inner solves complete the odd-cell Oseen systems. With
        // shared face-center DOFs and natural outlet pressure, the second
        // update meets the original five-percent nonlinear tolerance.
        let mut config = PcFanConfig::bench_120mm();
        config.operating_point.rpm = 1_300.0;
        config.operating_point.target_airflow_cfm *= 1_300.0 / 1_800.0;
        config.cfd.flow_model = PcFanFlowModel::IncompressibleCfd;
        config.cfd.backend = PcFanCfdBackend::Cpu;
        config.cfd.cells_x = 3;
        config.cfd.cells_y = 3;
        config.cfd.cells_z = 3;
        config.cfd.use_direct_solver = true;
        config.cfd.max_picard_iterations = 2;
        config.cfd.nonlinear_tolerance = 0.05;
        let mut design = config.default_design_variables();
        for section in &mut design.blade_sections {
            section.twist_deg = 24.0;
        }
        design.hub_radius_offset_m =
            0.30 * config.dimensions.blade_tip_radius_m - config.dimensions.hub_radius_m;
        design.shroud_radius_offsets_m.fill(-0.003 * 0.05);
        let geometry = config.build_geometry(&design);
        let solution = solve_pc_fan_cfd_mean_flow_for_geometry(&config, &design, &geometry);
        assert_eq!(solution.iterations, 2);
        assert_eq!(
            pc_fan_cfd_first_admission_violation(&config, &solution, 0.05),
            None
        );
    }

    #[test]
    fn mesh_convergence_rejects_invalid_conservation_tolerance_before_solving() {
        let mut config = PcFanConfig::default_120mm();
        config.cfd.max_relative_momentum_imbalance = f64::NAN;
        let design = config.default_design_variables();

        assert_eq!(
            try_solve_pc_fan_cfd_mesh_convergence(
                &config,
                &design,
                PcFanCfdMeshConvergenceConfig::default(),
            ),
            Err(
                PcFanCfdMeshConvergenceConfigError::InvalidConservationTolerance {
                    metric: "max_relative_momentum_imbalance",
                }
            )
        );
    }

    #[test]
    fn mesh_convergence_config_rejects_unbounded_or_underspecified_sweeps() {
        assert!(PcFanCfdMeshConvergenceConfig::default().validate().is_ok());

        let mut config = PcFanCfdMeshConvergenceConfig::default();
        config.start_cells = 0;
        assert_eq!(
            config.validate(),
            Err(PcFanCfdMeshConvergenceConfigError::StartCellsMustBePositive)
        );

        let mut config = PcFanCfdMeshConvergenceConfig::default();
        config.max_cells = config.start_cells - 1;
        assert_eq!(
            config.validate(),
            Err(PcFanCfdMeshConvergenceConfigError::MaxCellsBeforeStart {
                start_cells: 6,
                max_cells: 5,
            })
        );

        let mut config = PcFanCfdMeshConvergenceConfig::default();
        config.relative_tolerance = f64::NAN;
        assert_eq!(
            config.validate(),
            Err(PcFanCfdMeshConvergenceConfigError::InvalidRelativeTolerance)
        );

        let mut config = PcFanCfdMeshConvergenceConfig::default();
        config.max_final_residual = f64::NAN;
        assert_eq!(
            config.validate(),
            Err(PcFanCfdMeshConvergenceConfigError::InvalidFinalResidual)
        );

        let mut config = PcFanCfdMeshConvergenceConfig::default();
        config.min_levels = 1;
        assert_eq!(
            config.validate(),
            Err(PcFanCfdMeshConvergenceConfigError::MinLevelsTooSmall)
        );

        let mut config = PcFanCfdMeshConvergenceConfig::default();
        config.min_levels = 4;
        assert_eq!(
            config.validate(),
            Err(PcFanCfdMeshConvergenceConfigError::InsufficientLevels {
                available: 3,
                required: 4,
            })
        );
    }

    #[test]
    fn immersed_solid_penalty_damps_fixed_solids_and_spins_hub() {
        let config = PcFanConfig::default_120mm();
        let design = config.default_design_variables();
        let geometry = config.build_geometry(&design);
        let penalty =
            ImmersedSolidPenalty::from_geometry_with_design(&config, &geometry, 0.004, None);

        let frame_radius_m =
            0.25 * (geometry.frame_box.inner_side_m + geometry.frame_box.outer_side_m);
        let fixed_force = penalty.force_at(Point3::new(frame_radius_m, 0.0, 0.0), [1.0, 0.0, 0.0]);
        assert!(
            fixed_force[0] < 0.0,
            "fixed frame solid should penalize non-zero local velocity"
        );

        let hub_point = Point3::new(0.5 * geometry.hub.radius_m, 0.0, geometry.hub.center_z_m);
        let hub_force = penalty.force_at(hub_point, [0.0, 0.0, 0.0]);
        assert!(
            hub_force[1] > 0.0,
            "rotating hub solid should drive tangential velocity"
        );

        let open_radius_m = 0.5 * (geometry.hub.radius_m + geometry.shroud.inner_radius_m);
        let open_force = penalty.force_at(Point3::new(open_radius_m, 0.0, 0.0), [1.0, 0.0, 0.0]);
        assert_eq!(open_force, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn blade_wall_set_detects_point_inside_default_blade() {
        // Compute the world-space point on the blade-1 midline at
        // mid-span / mid-chord. The baseline has non-zero path offsets
        // (backward sweep, axial sweep) and non-zero camber, so the
        // midline is offset from the rotation axis and the equator —
        // we sample it directly from the design.
        let config = PcFanConfig::default_120mm();
        let design = config.default_design_variables();
        let walls = BladeWallSet::from_design(&config, &design);
        let mid_r = 0.5 * (config.dimensions.hub_radius_m + config.dimensions.blade_tip_radius_m);
        let span = (mid_r - config.dimensions.hub_radius_m)
            / (config.dimensions.blade_tip_radius_m - config.dimensions.hub_radius_m);
        let section = sample_section(&design.blade_sections, span);
        let path = sample_path(&design.blade_path, span);
        // At chord-fraction 0.5 the camber bow is at its maximum and the
        // local-chord term vanishes, so the midline reduces to:
        //   y = path.tang + camber, z = path.axial
        let midline_y = path.tangential_offset_m + section.camber_m;
        let midline_z = path.axial_offset_m;
        let inside = walls.point_inside_blade(Point3::new(mid_r, midline_y, midline_z));
        assert!(
            inside,
            "point on blade midline at azimuth 0 should be inside"
        );
    }

    #[test]
    fn blade_wall_set_skips_points_outside_radial_range() {
        let config = PcFanConfig::default_120mm();
        let design = config.default_design_variables();
        let walls = BladeWallSet::from_design(&config, &design);
        assert!(!walls.point_inside_blade(Point3::new(0.001, 0.0, 0.0)));
        assert!(!walls.point_inside_blade(Point3::new(0.080, 0.0, 0.0)));
    }

    #[test]
    fn blade_wall_set_skips_between_blades() {
        // 8 blades on cardinal angles k·π/4. A point at azimuth π/8 — half
        // way between two blades — at mid-radius must not register.
        let config = PcFanConfig::default_120mm();
        let design = config.default_design_variables();
        let walls = BladeWallSet::from_design(&config, &design);
        let mid_r = 0.5 * (config.dimensions.hub_radius_m + config.dimensions.blade_tip_radius_m);
        let theta = std::f64::consts::PI / 8.0;
        let point = Point3::new(mid_r * theta.cos(), mid_r * theta.sin(), 0.0);
        assert!(!walls.point_inside_blade(point));
    }

    #[test]
    fn blade_actuator_force_responds_to_local_flow_velocity() {
        let config = PcFanConfig {
            mesh: PcFanMeshResolution {
                radial_segments: 2,
                chord_segments: 2,
                circumferential_segments: 8,
                harmonic_count: 2,
            },
            ..PcFanConfig::default_120mm()
        };
        let design = config.default_design_variables();
        let geometry = config.build_geometry(&design);
        let forcing = BladeActuatorForcing::from_geometry(&config, &geometry, &design);
        let radius_m =
            0.5 * (config.dimensions.hub_radius_m + config.dimensions.blade_tip_radius_m);
        let coords = [radius_m, 0.0, 0.0];

        let stationary_force = forcing.force_at(coords, [0.0, 0.0, 0.0]);
        let co_rotating_force =
            forcing.force_at(coords, [0.0, forcing.omega_rad_s * radius_m, 0.0]);

        assert!(
            stationary_force[2] > 0.0,
            "actuator must drive flow from min-z inlet toward max-z outlet"
        );
        assert!(
            stationary_force[2].abs() > co_rotating_force[2].abs(),
            "co-rotating local flow should reduce relative blade loading: stationary z-force {}, \
             co-rotating z-force {}",
            stationary_force[2],
            co_rotating_force[2]
        );
    }

    #[test]
    fn actuator_force_jacobian_matches_local_force_law() {
        let config = PcFanConfig::default_120mm();
        let design = config.default_design_variables();
        let geometry = config.build_geometry(&design);
        let forcing = BladeActuatorForcing::from_geometry(&config, &geometry, &design);
        let radius = 0.5 * (config.dimensions.hub_radius_m + config.dimensions.blade_tip_radius_m);
        let coords = [radius, 0.0, 0.0];
        let velocity = [0.37, -0.21, 0.43];
        let jacobian = forcing.force_jacobian_at(coords, velocity);
        let step = 1.0e-6;
        let mut maximum_relative_error: f64 = 0.0;
        for input in 0..3 {
            let mut plus = velocity;
            let mut minus = velocity;
            plus[input] += step;
            minus[input] -= step;
            let plus_force = forcing.force_at(coords, plus);
            let minus_force = forcing.force_at(coords, minus);
            for output in 0..3 {
                let finite_difference = (plus_force[output] - minus_force[output]) / (2.0 * step);
                let analytic = jacobian[output][input];
                let scale = analytic.abs().max(finite_difference.abs()).max(1.0);
                maximum_relative_error =
                    maximum_relative_error.max((analytic - finite_difference).abs() / scale);
            }
        }
        assert!(
            maximum_relative_error < 2.0e-8,
            "actuator local Jacobian mismatch = {maximum_relative_error:e}"
        );
    }

    #[test]
    fn actuator_design_force_jacobian_matches_centered_design_difference() {
        let config = PcFanConfig {
            mesh: PcFanMeshResolution {
                radial_segments: 4,
                chord_segments: 4,
                circumferential_segments: 8,
                harmonic_count: 2,
            },
            ..PcFanConfig::default_120mm()
        };
        let design = config.default_design_variables();
        let geometry = config.build_geometry(&design);
        let forcing = BladeActuatorForcing::from_geometry(&config, &geometry, &design);
        let panel = geometry.blades[0].panels[5];
        let point = face_center(panel.vertices);
        let coords = [point.x, point.y, point.z];
        let local_velocity = [0.11, -0.07, 0.13];
        let analytic = forcing.force_design_jacobian_at(coords, local_velocity);
        let base = design.to_vector();
        let mut maximum_relative_error: f64 = 0.0;
        for design_index in 0..DESIGN_VARIABLE_COUNT {
            let step = 1.0e-7 * base[design_index].abs().max(1.0);
            let mut plus = base;
            let mut minus = base;
            plus[design_index] += step;
            minus[design_index] -= step;
            let plus_design = PcFanDesignVariables::from_vector(plus);
            let minus_design = PcFanDesignVariables::from_vector(minus);
            let plus_geometry = config.build_geometry(&plus_design);
            let minus_geometry = config.build_geometry(&minus_design);
            let plus_force =
                BladeActuatorForcing::from_geometry(&config, &plus_geometry, &plus_design)
                    .force_at(coords, local_velocity);
            let minus_force =
                BladeActuatorForcing::from_geometry(&config, &minus_geometry, &minus_design)
                    .force_at(coords, local_velocity);
            for component in 0..3 {
                let finite_difference =
                    (plus_force[component] - minus_force[component]) / (2.0 * step);
                let error = (analytic[component][design_index] - finite_difference).abs()
                    / analytic[component][design_index]
                        .abs()
                        .max(finite_difference.abs())
                        .max(1.0);
                maximum_relative_error = maximum_relative_error.max(error);
            }
        }
        assert!(
            maximum_relative_error < 5.0e-6,
            "maximum relative design-force error {maximum_relative_error}"
        );
    }

    #[test]
    fn blade_actuator_force_scales_with_blade_count() {
        let make_force = |blade_count: usize| {
            let config = PcFanConfig {
                blade_count,
                mesh: PcFanMeshResolution {
                    radial_segments: 2,
                    chord_segments: 2,
                    circumferential_segments: 8,
                    harmonic_count: 2,
                },
                ..PcFanConfig::default_120mm()
            };
            let design = config.default_design_variables();
            let geometry = config.build_geometry(&design);
            let forcing = BladeActuatorForcing::from_geometry(&config, &geometry, &design);
            let radius_m =
                0.5 * (config.dimensions.hub_radius_m + config.dimensions.blade_tip_radius_m);
            forcing.force_at([radius_m, 0.0, 0.0], [0.0, 0.0, 0.0])[2]
        };

        let four_blades = make_force(4);
        let eight_blades = make_force(8);
        assert!(eight_blades > 1.8 * four_blades);
    }

    #[test]
    fn actuator_loading_matches_constant_force_volume_and_power_reference() {
        // One P1 tetrahedron with centroid (r, 0, z) and volume 1/3 m³ gives
        // an exact single-point quadrature reference for the production
        // actuator-loading integration.  The reference below is evaluated
        // independently from `integrate_blade_actuator_loading`.
        let mut mesh = sonium_fem::mesh::Mesh::new(3);
        for point in [
            sonium_fem::mesh::Point::new_3d(0.0, -0.5, -0.5),
            sonium_fem::mesh::Point::new_3d(1.0, 0.5, -0.5),
            sonium_fem::mesh::Point::new_3d(1.0, -0.5, 0.5),
            sonium_fem::mesh::Point::new_3d(0.0, 0.5, 0.5),
        ] {
            mesh.add_node(point);
        }
        mesh.add_element(sonium_fem::mesh::ElementType::Tetrahedron, vec![0, 1, 2, 3]);

        let forcing = BladeActuatorForcing {
            samples: vec![BladeActuatorSample {
                radius_m: 0.5,
                z_m: 0.0,
                axial_loading: 2.0,
                tangential_loading: 3.0,
                area_weight: 1.0,
            }],
            sample_design_derivatives: vec![BladeActuatorSampleDerivative {
                radius_m: [0.0; DESIGN_VARIABLE_COUNT],
                z_m: [0.0; DESIGN_VARIABLE_COUNT],
                axial_loading: [0.0; DESIGN_VARIABLE_COUNT],
                tangential_loading: [0.0; DESIGN_VARIABLE_COUNT],
                area_weight: [0.0; DESIGN_VARIABLE_COUNT],
            }],
            hub_radius_m: 0.0,
            tip_radius_m: 1.0,
            default_chord_m: 1.0,
            radial_influence_m: 1.0,
            radial_influence_design_derivative: [0.0; DESIGN_VARIABLE_COUNT],
            axial_influence_m: 1.0,
            omega_rad_s: 4.0,
            air_density_kg_m3: 2.0,
            blade_count_scale: 1.0,
        };
        let velocity = vec![[0.0; 3]; 4];
        let result = integrate_blade_actuator_loading(&mesh, &velocity, &forcing);

        let volume_m3 = 1.0 / 3.0;
        let radius_m = 0.5;
        let relative_tangential_m_s = forcing.omega_rad_s * radius_m;
        let dynamic_pressure_pa =
            0.5 * forcing.air_density_kg_m3 * relative_tangential_m_s * relative_tangential_m_s;
        let expected_axial_density_n_m3 = 0.18 * dynamic_pressure_pa * 2.0;
        let expected_tangential_density_n_m3 = 0.12 * dynamic_pressure_pa * 3.0;
        let expected_axial_force_n = expected_axial_density_n_m3 * volume_m3;
        let expected_tangential_force_n = expected_tangential_density_n_m3 * volume_m3;
        let expected_torque_nm = radius_m * expected_tangential_force_n;

        assert!((result.force_on_fluid_n[0]).abs() < 1.0e-12);
        assert!((result.force_on_fluid_n[1] - expected_tangential_force_n).abs() < 1.0e-12);
        assert!((result.force_on_fluid_n[2] - expected_axial_force_n).abs() < 1.0e-12);
        assert!((result.torque_on_fluid_nm - expected_torque_nm).abs() < 1.0e-12);
        assert!(
            (result.shaft_power_to_fluid_w - expected_torque_nm * forcing.omega_rad_s).abs()
                < 1.0e-12
        );
    }

    #[test]
    fn conservation_ledger_matches_manufactured_pressure_rise_and_shaft_power() {
        // A unit-area, unit-length control volume with uniform Q=2 m³/s,
        // p_in=0 Pa, p_out=10 Pa, and a +10 N recovered fan force is an
        // exact steady momentum/energy reference. This fixes the signs of
        // pressure-rise power and pressure/body-force closure without
        // relying on a particular nonlinear CFD iterate.
        let config = PcFanConfig::default_120mm();
        let domain = PcFanCfdDomain {
            nodes: Vec::new(),
            cells: Vec::new(),
            boundary_faces: Vec::new(),
            min_corner: Point3::new(0.0, 0.0, -0.5),
            max_corner: Point3::new(1.0, 1.0, 0.5),
            inlet_area_m2: 1.0,
            outlet_area_m2: 1.0,
            volume_m3: 1.0,
        };
        let nodes = vec![
            Point3::new(0.0, 0.0, -0.5),
            Point3::new(1.0, 1.0, -0.5),
            Point3::new(0.0, 0.0, 0.5),
            Point3::new(1.0, 1.0, 0.5),
        ];
        let velocity = vec![[0.0, 0.0, 2.0]; 4];
        let pressure = vec![0.0, 0.0, 10.0, 10.0];
        let loading = PcFanActuatorLoading {
            force_on_fluid_n: [0.0, 0.0, 10.0],
            torque_on_fluid_nm: 10.0,
            angular_speed_rad_s: 2.0,
            shaft_power_to_fluid_w: 20.0,
        };
        let flow_cfm = 2.0 * M3S_TO_CFM;

        let ledger = build_conservation_ledger(
            &config,
            &domain,
            &nodes,
            &velocity,
            &pressure,
            flow_cfm,
            flow_cfm,
            Some(loading),
            None,
            1.0e-9,
        );

        assert!(ledger.relative_mass_imbalance.abs() < 1.0e-12);
        assert!((ledger.pressure_force_n + 10.0).abs() < 1.0e-12);
        assert!((ledger.body_force_n - 10.0).abs() < 1.0e-12);
        assert!(ledger.axial_momentum_residual_n.abs() < 1.0e-12);
        assert!(ledger.relative_momentum_imbalance < 1.0e-12);
        assert!(ledger.relative_pressure_closure < 1.0e-12);
        assert!((ledger.pressure_power_w - 20.0).abs() < 1.0e-12);
        assert!((ledger.shaft_power_to_fluid_w - 20.0).abs() < 1.0e-12);
        assert!(ledger.relative_energy_imbalance < 1.0e-12);
    }

    #[test]
    fn production_picard_map_publishes_one_checked_update() {
        let mut config = PcFanConfig::default_120mm();
        config.cfd.flow_model = PcFanFlowModel::IncompressibleCfd;
        config.cfd.backend = PcFanCfdBackend::Cpu;
        config.cfd.cells_x = 2;
        config.cfd.cells_y = 2;
        config.cfd.cells_z = 2;
        config.cfd.max_picard_iterations = 1;
        let design = config.default_design_variables();
        let initial_state = solve_pc_fan_cfd_mean_flow(&config, &design)
            .picard_linearization
            .expect("seed solve should publish a Picard map state")
            .state;

        let observation = evaluate_pc_fan_cfd_picard_map(&config, &design, &initial_state)
            .expect("production one-step Picard map should publish its observation");
        observation.validate().unwrap();
        assert_eq!(observation.state.len(), initial_state.len());
        assert!(
            observation
                .fixed_point_residual
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(observation.provenance.contains("picard-oseen"));
    }

    #[test]
    fn production_newton_tangent_publishes_checked_state_residual() {
        let mut config = PcFanConfig::default_120mm();
        config.cfd.flow_model = PcFanFlowModel::IncompressibleCfd;
        config.cfd.backend = PcFanCfdBackend::Cpu;
        config.cfd.cells_x = 2;
        config.cfd.cells_y = 2;
        config.cfd.cells_z = 2;
        config.cfd.max_picard_iterations = 1;
        let design = config.default_design_variables();
        let initial_state = solve_pc_fan_cfd_mean_flow(&config, &design)
            .picard_linearization
            .expect("seed solve should publish a Picard state")
            .state;

        let tangent = evaluate_pc_fan_cfd_newton_linearization(&config, &design, &initial_state)
            .expect("production map should publish its analytic Newton tangent");
        tangent.validate().unwrap();
        assert!(tangent.provenance.contains("newton-convection"));
        assert_eq!(tangent.state.len(), initial_state.len());
        assert!(tangent.state_residual.iter().all(|value| value.is_finite()));
        assert!(
            tangent
                .state_jacobian
                .values
                .iter()
                .all(|value| value.is_finite())
        );
        let residual_design_jacobian = tangent
            .residual_design_jacobian
            .as_ref()
            .expect("frozen actuator path should publish design residual derivatives");
        assert_eq!(residual_design_jacobian.len(), tangent.state.len());
        assert!(
            residual_design_jacobian
                .iter()
                .all(|row| row.len() == DESIGN_VARIABLE_COUNT
                    && row.iter().all(|value| value.is_finite()))
        );
    }

    #[test]
    fn state_only_newton_preserves_primal_assembly_with_mesh_motion() {
        let mut config = PcFanConfig::default_120mm();
        config.cfd.flow_model = PcFanFlowModel::IncompressibleCfd;
        config.cfd.backend = PcFanCfdBackend::Cpu;
        config.cfd.cells_x = 2;
        config.cfd.cells_y = 2;
        config.cfd.cells_z = 2;
        config.cfd.max_picard_iterations = 1;
        config.cfd.use_direct_solver = true;
        let mut design = config.default_design_variables();
        design.shroud_radius_offsets_m[0] = 1.0e-6;
        let seed = solve_pc_fan_cfd_mean_flow(&config, &design);
        let mut state = seed
            .velocity_m_s
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        state.extend_from_slice(&seed.pressure_pa);
        let full = evaluate_pc_fan_cfd_newton_linearization(&config, &design, &state).unwrap();
        let primal =
            evaluate_pc_fan_cfd_newton_state_linearization(&config, &design, &state).unwrap();
        full.validate().unwrap();
        primal.validate().unwrap();
        assert!(full.residual_design_jacobian.is_some());
        assert!(primal.residual_design_jacobian.is_none());
        assert_eq!(full.state, primal.state);
        assert_eq!(full.provenance, primal.provenance);
        assert_eq!(full.backend, primal.backend);
        assert_eq!(
            full.state_jacobian.dimension,
            primal.state_jacobian.dimension
        );
        assert_eq!(full.state_jacobian.row_ptrs, primal.state_jacobian.row_ptrs);
        assert_eq!(
            full.state_jacobian.column_indices,
            primal.state_jacobian.column_indices
        );
        for (left, right) in full
            .state_residual
            .iter()
            .zip(&primal.state_residual)
            .chain(
                full.state_jacobian
                    .values
                    .iter()
                    .zip(&primal.state_jacobian.values),
            )
        {
            assert!((left - right).abs() <= 1.0e-12);
        }
    }

    #[test]
    fn state_only_newton_rejects_invalid_state_before_publication() {
        let mut config = PcFanConfig::default_120mm();
        config.cfd.flow_model = PcFanFlowModel::IncompressibleCfd;
        config.cfd.cells_x = 2;
        config.cfd.cells_y = 2;
        config.cfd.cells_z = 2;
        let design = config.default_design_variables();
        assert!(
            evaluate_pc_fan_cfd_newton_state_linearization(&config, &design, &[f64::NAN]).is_err()
        );
        assert!(evaluate_pc_fan_cfd_newton_state_linearization(&config, &design, &[]).is_err());
    }

    #[test]
    fn production_derivative_seam_rejects_reduced_flow_and_bad_design_shape() {
        let config = PcFanConfig::default_120mm();
        let design = config.default_design_variables();
        assert!(matches!(
            evaluate_pc_fan_cfd_picard_map(&config, &design, &[]),
            Err(PcFanResolvedAdjointError::UnsupportedFlowModel(_))
        ));
        assert!(matches!(
            design_from_probe_values(&[0.0]),
            Err(PcFanResolvedAdjointError::DimensionMismatch(_))
        ));
        let mut invalid_config = config;
        invalid_config.cfd.flow_model = PcFanFlowModel::IncompressibleCfd;
        invalid_config.dimensions.frame_side_m = f64::NAN;
        assert!(matches!(
            evaluate_pc_fan_cfd_picard_map(&invalid_config, &design, &[]),
            Err(PcFanResolvedAdjointError::ProbeEvaluationFailed(
                "CFD geometry"
            ))
        ));
    }

    #[test]
    fn resolved_adjoint_wrapper_requires_cfd_state_and_exact_derivatives() {
        let mut config = PcFanConfig::default_120mm();
        config.cfd.flow_model = PcFanFlowModel::IncompressibleCfd;
        let design = config.default_design_variables();
        let derivatives = PcFanResolvedFlowDerivativeData {
            provenance: "production:pcfan:analytic-design-objective:v1".into(),
            residual_design_jacobian: Vec::new(),
            objective_state_gradient: Vec::new(),
            objective_design_gradient: Vec::new(),
        };
        assert!(matches!(
            solve_pc_fan_cfd_resolved_flow_adjoint(
                &PcFanResolvedAdjointConfig::default(),
                &config,
                &design,
                &[],
                derivatives,
            ),
            Err(PcFanResolvedAdjointError::DimensionMismatch(_))
        ));

        config.cfd.flow_model = PcFanFlowModel::ReducedOrder;
        let derivatives = PcFanResolvedFlowDerivativeData {
            provenance: "production:pcfan:analytic-design-objective:v1".into(),
            residual_design_jacobian: Vec::new(),
            objective_state_gradient: Vec::new(),
            objective_design_gradient: Vec::new(),
        };
        assert!(matches!(
            solve_pc_fan_cfd_resolved_flow_adjoint(
                &PcFanResolvedAdjointConfig::default(),
                &config,
                &design,
                &[],
                derivatives,
            ),
            Err(PcFanResolvedAdjointError::UnsupportedFlowModel(_))
        ));
    }
}

#[cfg(test)]
mod steady_fixture_convergence_tests {
    use super::*;

    // Regression: the 2x2x2 production fixture must reach the actual
    // discrete steady solution (tight residual, forward pressure rise,
    // fixture-level mass conservation), not an early-exit state on the
    // globalization trajectory. The damped-Newton globalization
    // (step halving exhausted before Levenberg escalation, no
    // preemptive correction-norm bound) is what makes the cold
    // full-speed solve land on that solution at both ends of the
    // parity RPM sweep.
    #[test]
    fn steady_fixture_converges_to_discrete_solution_at_parity_rpms() {
        for rpm in [600.0, 1200.0] {
            let mut config = PcFanConfig::default_120mm();
            config.mesh = PcFanMeshResolution {
                radial_segments: 2,
                chord_segments: 2,
                circumferential_segments: 8,
                harmonic_count: 2,
            };
            config.cfd.flow_model = PcFanFlowModel::IncompressibleCfd;
            config.cfd.backend = PcFanCfdBackend::Cpu;
            config.cfd.cells_x = 2;
            config.cfd.cells_y = 2;
            config.cfd.cells_z = 2;
            config.cfd.use_direct_solver = true;
            config.cfd.max_picard_iterations = 40;
            config.cfd.nonlinear_tolerance = 2.0e-2;
            config.cfd.max_relative_mass_imbalance = 0.10;
            config.cfd.linear_solver_max_iterations = 300;
            config.cfd.linear_solver_tolerance = 1.0e-4;
            config.operating_point.rpm = rpm;
            let design = config.default_design_variables();
            let solution = solve_pc_fan_cfd_mean_flow(&config, &design);
            assert!(
                solution.converged,
                "rpm {rpm}: steady fixture did not converge (residual {:.3e})",
                solution.residual
            );
            assert!(
                solution.residual < 1.0e-6,
                "rpm {rpm}: converged residual {:.3e} is not the tight discrete solution",
                solution.residual
            );
            assert!(
                solution.conservation.relative_mass_imbalance < 0.10,
                "rpm {rpm}: mass imbalance {:.3e} exceeds the fixture gate",
                solution.conservation.relative_mass_imbalance
            );
            assert!(
                solution.pressure_drop_pa < 0.0,
                "rpm {rpm}: pressure drop {:.3e} Pa is not a forward fan rise",
                solution.pressure_drop_pa
            );
            assert!(
                solution.airflow_cfm > 10.0,
                "rpm {rpm}: airflow {:.3e} CFM is not a developed fan state",
                solution.airflow_cfm
            );
        }
    }
}
