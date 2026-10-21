//! PC fan aeroacoustic optimization workflow.
//!
//! The current model provides a deterministic 120 mm baseline geometry, spline
//! deformation controls, collision/merge validation, a crate-local mean-flow
//! solve, harmonic aeroacoustic evaluation through the FEM aeroacoustics source
//! model, and VTU export for geometry inspection.
//!
//! Alternative geometry can be described via MeshLang DSL - see `meshlang_geometry`.

// Rust guideline compliant 2026-02-21

mod accelerator_parity;
mod annular_fwh;
mod cfd;
mod cfd_independence;
mod correlation;
mod envelope_optimization;
mod evaluation;
mod fidelity;
mod geometry;
mod measured_product;
mod measured_validation;
mod mesh_motion;
pub mod meshlang_geometry;
mod model;
mod optimizer;
pub(crate) mod parallelism;
mod resolved_adjoint;
mod resolved_aeroacoustics;
mod resolved_envelope;
mod resolved_envelope_optimization;
mod state_transfer;
mod structural_calibration;
mod transient_adjoint;
mod transported_turbulence;

pub use accelerator_parity::*;
pub use annular_fwh::{
    AnnularFwhError, AnnularTone, AnnularToneSpec, annular_loading_tones,
    blade_passing_frequency_hz, modal_loading_amplitude_pa,
};
#[doc(inline)]
pub use cfd::evaluate_pc_fan_cfd_newton_state_linearization;
#[doc(hidden)]
pub use cfd::{
    PC_FAN_TRANSPORTED_BDF_NEWTON_PROVENANCE, PcFanCfdBodyFittedRemeshLevel,
    PcFanCfdTransientBdfState, PcFanCfdTransientConvergence, PcFanCfdTransientConvergenceConfig,
    PcFanCfdTransientConvergenceConfigError, PcFanCfdTransientConvergenceLevel,
    PcFanCfdTransientError, PcFanCfdTransientObservation, PcFanCfdTransientRemeshSchedule,
    PcFanCfdTransientRemeshStateAdjoint, PcFanCfdTransientStateTransferError,
    PcFanCfdTransientTrajectory, PcFanCfdTransientTrajectoryStep,
    PcFanResolvedFlowActuatorObjective, PcFanResolvedFlowObjective,
    PcFanResolvedFlowStateObjective, PcFanResolvedTransientObjective,
    build_pc_fan_cfd_actuator_objective_derivatives,
    build_pc_fan_cfd_resolved_objective_derivatives,
    build_pc_fan_cfd_resolved_state_objective_gradient,
    build_pc_fan_cfd_resolved_transient_objective_state_gradients,
    evaluate_pc_fan_cfd_newton_linearization, evaluate_pc_fan_cfd_picard_map,
    pc_fan_cfd_first_admission_violation, probe_pc_fan_cfd_resolved_flow_derivatives,
    solve_pc_fan_cfd_actuator_adjoint, solve_pc_fan_cfd_mean_flow,
    solve_pc_fan_cfd_mean_flow_with_state_transfer, solve_pc_fan_cfd_mesh_convergence,
    solve_pc_fan_cfd_resolved_flow_adjoint,
    solve_pc_fan_cfd_resolved_flow_adjoint_with_production_design_residual,
    solve_pc_fan_cfd_resolved_objective_adjoint,
    solve_pc_fan_cfd_resolved_objective_adjoint_from_solution,
    solve_pc_fan_cfd_resolved_state_adjoint, solve_pc_fan_cfd_resolved_transient_objective_adjoint,
    solve_pc_fan_cfd_resolved_transient_objective_adjoint_from_trajectory,
    solve_pc_fan_cfd_resolved_transient_objective_adjoint_with_derivatives,
    solve_pc_fan_cfd_resolved_transient_state_adjoint,
    solve_pc_fan_cfd_resolved_transient_state_adjoint_with_retained_stage_initial_history,
    solve_pc_fan_cfd_resolved_transient_state_adjoint_with_stage_initial_history,
    solve_pc_fan_cfd_resolved_transient_turbulent_state_adjoint,
    solve_pc_fan_cfd_rotating_wall_transient,
    solve_pc_fan_cfd_rotating_wall_transient_with_builtin_body_fitted_remesher,
    solve_pc_fan_cfd_rotating_wall_transient_with_state_transfer,
    solve_pc_fan_cfd_rotating_wall_transient_with_state_transfer_schedule,
    solve_pc_fan_cfd_rotating_wall_transient_with_state_transfer_to_mesh,
    solve_pc_fan_cfd_rotating_wall_transient_with_state_transfer_with_remesher,
    solve_pc_fan_cfd_rotating_wall_transient_with_trajectory,
    solve_pc_fan_cfd_transient_convergence,
    transfer_pc_fan_cfd_transient_remesh_schedule_state_adjoint,
    try_solve_pc_fan_cfd_mesh_convergence, try_solve_pc_fan_cfd_transient_convergence,
};
pub use cfd::{
    PcFanCfdTransientRemeshMapDesignDerivative, PcFanCfdTransientRemeshObjectiveAdjoint,
    solve_pc_fan_cfd_resolved_transient_remesh_schedule_objective_adjoint_with_derivatives,
    solve_pc_fan_cfd_resolved_transient_remesh_schedule_objective_adjoint_with_map_design_derivatives,
    solve_pc_fan_cfd_resolved_transient_remesh_schedule_state_objective_adjoint,
};
pub use cfd_independence::*;
pub use correlation::*;
pub use envelope_optimization::{
    OptimizedPcFanOperatingEnvelope, PcFanOperatingEnvelopeObjective,
    PcFanOperatingEnvelopeObjectiveConfig, PcFanOperatingEnvelopeOptimizationError,
    PcFanOperatingEnvelopeOptimizationSummary, evaluate_pc_fan_operating_envelope_objective,
    optimize_pc_fan_operating_envelope,
};
pub use evaluation::{
    PcFanCfdEvaluationError, PcFanDerivativePolicyError, PcFanFanCurveError,
    PcFanManufacturingEnsembleError, PcFanManufacturingOperatingEnvelopeError,
    attach_operating_point_spread, directivity_power_ratio, evaluate_pc_fan,
    evaluate_pc_fan_fan_curve, evaluate_pc_fan_fan_curve_objective,
    evaluate_pc_fan_from_cfd_solution, evaluate_pc_fan_manufacturing_ensemble,
    evaluate_pc_fan_manufacturing_operating_envelope, evaluate_pc_fan_operating_envelope,
    solve_pc_fan_acoustics, solve_pc_fan_mean_flow, solve_pc_fan_shape_gradient,
    solve_pc_fan_shape_gradient_reduced_adjoint, solve_pc_fan_shape_gradient_with_policy,
    try_evaluate_pc_fan, try_evaluate_pc_fan_from_cfd_solution,
};
pub use fidelity::*;
pub use measured_product::*;
pub use measured_validation::*;
pub use mesh_motion::*;
pub use model::*;
pub use optimizer::{
    MultiDEExperiment, MultiDEResult, optimize_pc_fan, optimize_pc_fan_de,
    optimize_pc_fan_fan_curve, optimize_pc_fan_multi_de, optimize_pc_fan_reduced_adjoint,
    optimize_pc_fan_with_derivative_policy, restore_pc_fan_feasibility,
    try_optimize_pc_fan_multi_de,
};
pub use resolved_adjoint::*;
pub use resolved_aeroacoustics::*;
pub use resolved_envelope::*;
pub use resolved_envelope_optimization::*;
pub use state_transfer::*;
pub use structural_calibration::*;

pub use transient_adjoint::*;
pub use transported_turbulence::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_geometry_has_eight_blades_and_four_mount_points() {
        let config = PcFanConfig::default_120mm();
        let design = config.default_design_variables();
        let geometry = config.build_geometry(&design);

        assert_close(config.operating_point.rpm, 600.0);
        assert_eq!(config.cfd.flow_model, PcFanFlowModel::ReducedOrder);
        assert_eq!(geometry.blades.len(), DEFAULT_BLADE_COUNT);
        assert_eq!(geometry.frame_mounts.len(), FRAME_MOUNT_COUNT);
        assert_eq!(geometry.support_struts.len(), SUPPORT_STRUT_COUNT);
        assert_close(
            geometry.frame_box.outer_side_m,
            config.dimensions.frame_side_m,
        );
        assert!(geometry.motor_mount.radius_m <= config.dimensions.hub_radius_m);
        assert!(geometry.blades.iter().all(|blade| !blade.panels.is_empty()));
    }

    #[test]
    fn empirical_operating_presets_scale_airflow_with_rpm() {
        let default = PcFanConfig::default_120mm();
        assert_close(default.operating_point.target_airflow_cfm, 20.0);

        let bench = PcFanOperatingPoint::bench(1_200.0);
        assert_close(bench.target_airflow_cfm, 40.0);
        assert_eq!(
            PcFanOperatingPoint::bench_airflow_prior_cfm(600.0),
            [14.0, 24.0]
        );

        let chassis = PcFanOperatingPoint::chassis_installed(1_200.0);
        assert_close(chassis.target_airflow_cfm, 30.0);
        assert_eq!(
            PcFanOperatingPoint::chassis_airflow_prior_cfm(600.0),
            [10.0, 20.0]
        );
    }

    #[test]
    fn catalogue_airflow_references_fit_rpm_scaled_bench_prior() {
        // These are the conventional 120 mm free-air catalogue points listed
        // in CALIBRATION.md, reduced to the model's 600 RPM reference with
        // the independent fan-similarity relation Q ∝ N. The 0.25 CFM gate
        // at 600 RPM allows catalogue rounding and the one source value
        // reported as approximate; it scales with RPM alongside the prior.
        let catalogue = [
            ("ARCTIC P12 PWM PST", 1_800.0, 56.3),
            ("ARCTIC P12 Silent", 1_050.0, 24.1),
            ("ARCTIC P12 Max", 3_300.0, 81.0),
            ("ARCTIC F12", 1_350.0, 53.0),
            (
                "be quiet! Silent Wings 4 120 mm PWM high-speed",
                2_500.0,
                76.7,
            ),
        ];

        for rpm in [600.0, 1_200.0] {
            let bench = PcFanOperatingPoint::bench(rpm);
            let prior = PcFanOperatingPoint::bench_airflow_prior_cfm(rpm);
            let tolerance_cfm = 0.25 * rpm / AIRFLOW_PRIOR_REFERENCE_RPM;

            assert!(
                bench.target_airflow_cfm >= prior[0] && bench.target_airflow_cfm <= prior[1],
                "bench target {:.2} CFM is outside [{:.2}, {:.2}] CFM at {:.0} RPM",
                bench.target_airflow_cfm,
                prior[0],
                prior[1],
                rpm
            );

            for (name, rated_rpm, rated_airflow_cfm) in catalogue {
                let scaled_airflow_cfm = rated_airflow_cfm * rpm / rated_rpm;
                let distance_from_prior = if scaled_airflow_cfm < prior[0] {
                    prior[0] - scaled_airflow_cfm
                } else if scaled_airflow_cfm > prior[1] {
                    scaled_airflow_cfm - prior[1]
                } else {
                    0.0
                };

                assert!(
                    distance_from_prior <= tolerance_cfm,
                    "{name}: {:.2} CFM at {:.0} RPM misses [{:.2}, {:.2}] by {:.2} CFM (gate {:.2})",
                    scaled_airflow_cfm,
                    rpm,
                    prior[0],
                    prior[1],
                    distance_from_prior,
                    tolerance_cfm
                );
            }
        }
    }

    #[test]
    fn fixed_frame_box_and_x_support_are_exported_as_surfaces() {
        let config = PcFanConfig {
            mesh: PcFanMeshResolution {
                radial_segments: 2,
                chord_segments: 2,
                circumferential_segments: 12,
                harmonic_count: 2,
            },
            ..PcFanConfig::default_120mm()
        };
        let design = config.default_design_variables();
        let geometry = config.build_geometry(&design);
        let panels = geometry.surface_panels();

        assert!(
            panels
                .iter()
                .any(|panel| panel.kind == PcFanSurfaceKind::FrameBox)
        );
        assert!(
            panels
                .iter()
                .any(|panel| panel.kind == PcFanSurfaceKind::SupportStrut)
        );
        assert!(
            panels
                .iter()
                .all(|panel| panel.kind != PcFanSurfaceKind::MotorMount)
        );
        assert!(
            panels
                .iter()
                .all(|panel| panel.kind != PcFanSurfaceKind::FrameMount)
        );
        assert!(
            geometry
                .support_struts
                .iter()
                .all(|strut| strut.end == geometry.motor_mount.center)
        );
        assert_close(
            geometry.shroud.wall_thickness_m,
            config.dimensions.shroud_wall_thickness_m,
        );
        assert!(
            geometry.frame_box.inner_side_m >= 2.0 * geometry.shroud.outer_radius_m - f64::EPSILON
        );
        assert!(
            geometry.frame_box.wall_thickness_m
                >= config.dimensions.frame_wall_thickness_m - f64::EPSILON
        );
        assert!(
            geometry
                .support_struts
                .iter()
                .all(|strut| strut.thickness_m > 0.0)
        );

        let support_z_values = sorted_unique_z_values(&panels, PcFanSurfaceKind::SupportStrut);
        assert!(support_z_values.len() >= 2);
        assert!(
            support_z_values[support_z_values.len() - 1] - support_z_values[0]
                >= config.dimensions.support_strut_thickness_m - 1.0e-9
        );

        let hub_rear_z = geometry.hub.center_z_m - 0.5 * geometry.hub.depth_m;
        let hub_top_z = geometry.hub.center_z_m + 0.5 * geometry.hub.depth_m;
        assert!(hub_rear_z - support_z_values[support_z_values.len() - 1] >= 0.002 - 1.0e-9);
        assert_close(hub_top_z, 0.5 * config.dimensions.depth_m);

        let hub_top_cap_count = panels
            .iter()
            .filter(|panel| {
                panel.kind == PcFanSurfaceKind::Hub
                    && panel
                        .vertices
                        .iter()
                        .all(|vertex| (vertex.z - hub_top_z).abs() < 1.0e-9)
                    && panel.area > 0.0
            })
            .count();
        assert_eq!(hub_top_cap_count, config.mesh.circumferential_segments);
    }

    #[test]
    fn shroud_wall_stays_inside_fixed_frame_box() {
        let config = PcFanConfig::default_120mm();
        let mut design = config.default_design_variables();
        for offset in &mut design.shroud_radius_offsets_m {
            *offset = 0.004;
        }

        let geometry = config.build_geometry(&design);

        assert!(
            geometry.shroud.outer_radius_m <= 0.5 * geometry.frame_box.inner_side_m + f64::EPSILON
        );
        assert!(
            geometry.frame_box.wall_thickness_m
                >= config.dimensions.frame_wall_thickness_m - f64::EPSILON
        );
    }

    #[test]
    fn straight_plane_baseline_has_no_invalid_blade_intersections() {
        let config = PcFanConfig::default_120mm();
        let design = config.default_design_variables();
        let geometry = config.build_geometry(&design);
        let status = geometry.validate_blade_intersections();

        assert!(status.is_valid);
        assert_eq!(status.invalid_intersection_count, 0);
        assert_eq!(status.root_merge_count, DEFAULT_BLADE_COUNT);
    }

    #[test]
    fn geometry_vtu_writer_emits_a_file() {
        let config = PcFanConfig {
            mesh: PcFanMeshResolution {
                radial_segments: 2,
                chord_segments: 2,
                circumferential_segments: 8,
                harmonic_count: 3,
            },
            ..PcFanConfig::default_120mm()
        };
        let optimized = optimize_pc_fan(config);
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("pcfan.vtu");

        optimized.geometry.write_vtu(&path).expect("write VTU");

        let metadata = std::fs::metadata(path).expect("VTU metadata");
        assert!(metadata.len() > 0);
    }

    #[test]
    fn cfd_domain_has_volume_regions_and_boundary_tags() {
        let config = PcFanConfig {
            mesh: PcFanMeshResolution {
                radial_segments: 2,
                chord_segments: 2,
                circumferential_segments: 8,
                harmonic_count: 2,
            },
            cfd: PcFanCfdConfig {
                cells_x: 4,
                cells_y: 4,
                cells_z: 6,
                ..PcFanCfdConfig::default()
            },
            ..PcFanConfig::default_120mm()
        };
        let design = config.default_design_variables();
        let domain = config.build_cfd_domain(&design);

        assert_eq!(domain.nodes.len(), (4 + 1) * (4 + 1) * (6 + 1));
        assert_eq!(domain.cells.len(), 4 * 4 * 6);
        assert!(domain.volume_m3 > 0.0);
        assert!(domain.inlet_area_m2 > 0.0);
        assert!(domain.outlet_area_m2 > 0.0);
        assert!(domain.min_corner.z < -0.5 * config.dimensions.depth_m);
        assert!(domain.max_corner.z > 0.5 * config.dimensions.depth_m);
        assert!(
            domain
                .cells
                .iter()
                .any(|cell| cell.region == PcFanCfdCellRegion::InletExtension)
        );
        assert!(
            domain
                .cells
                .iter()
                .any(|cell| cell.region == PcFanCfdCellRegion::FanBox)
        );
        assert!(
            domain
                .cells
                .iter()
                .any(|cell| cell.region == PcFanCfdCellRegion::OutletExtension)
        );

        assert_boundary_kind(&domain, PcFanCfdBoundaryKind::Inlet);
        assert_boundary_kind(&domain, PcFanCfdBoundaryKind::Outlet);
        assert_boundary_kind(&domain, PcFanCfdBoundaryKind::OuterWall);
        assert_boundary_kind(&domain, PcFanCfdBoundaryKind::ShroudWall);
        assert_boundary_kind(&domain, PcFanCfdBoundaryKind::SupportWall);
        assert_boundary_kind(&domain, PcFanCfdBoundaryKind::RotatingHub);
        assert_boundary_kind(&domain, PcFanCfdBoundaryKind::RotatingBlade);
    }

    #[test]
    fn cfd_domain_vtu_writers_emit_files() {
        let config = PcFanConfig {
            mesh: PcFanMeshResolution {
                radial_segments: 2,
                chord_segments: 2,
                circumferential_segments: 8,
                harmonic_count: 2,
            },
            cfd: PcFanCfdConfig {
                cells_x: 2,
                cells_y: 2,
                cells_z: 3,
                ..PcFanCfdConfig::default()
            },
            ..PcFanConfig::default_120mm()
        };
        let domain = config.build_cfd_domain(&config.default_design_variables());
        let dir = tempfile::tempdir().expect("temp dir");
        let volume_path = dir.path().join("pcfan-cfd-volume.vtu");
        let boundary_path = dir.path().join("pcfan-cfd-boundary.vtu");

        domain.write_vtu(&volume_path).expect("write volume VTU");
        domain
            .write_boundary_vtu(&boundary_path)
            .expect("write boundary VTU");

        assert!(
            std::fs::metadata(&volume_path)
                .expect("volume metadata")
                .len()
                > 0
        );
        assert!(
            std::fs::metadata(&boundary_path)
                .expect("boundary metadata")
                .len()
                > 0
        );
        sonium_xem::io::vtu::validate_vtu_file(&volume_path).expect("volume VTU must parse");
        sonium_xem::io::vtu::validate_vtu_file(&boundary_path).expect("boundary VTU must parse");
        let volume_xml = std::fs::read_to_string(&volume_path).expect("read volume VTU");
        let boundary_xml = std::fs::read_to_string(&boundary_path).expect("read boundary VTU");
        assert!(volume_xml.contains("region"));
        assert!(volume_xml.contains("volume_m3"));
        assert!(boundary_xml.contains("boundary_kind"));
        assert!(boundary_xml.contains("source_surface"));
    }

    #[test]
    fn cfd_solid_boundary_conditions_apply_stationary_and_rotating_walls() {
        let config = PcFanConfig {
            mesh: PcFanMeshResolution {
                radial_segments: 2,
                chord_segments: 2,
                circumferential_segments: 12,
                harmonic_count: 2,
            },
            ..PcFanConfig::default_120mm()
        };
        let design = config.default_design_variables();
        let geometry = config.build_geometry(&design);
        let domain = geometry.build_cfd_domain(&config);
        let conditions = crate::cfd::SolidBoundaryConditions::from_domain(&config, &domain, 0.012);
        let hub_top_z = geometry.hub.center_z_m + 0.5 * geometry.hub.depth_m;
        let rotating_velocity = conditions
            .velocity_at_point(Point3::new(config.dimensions.hub_radius_m, 0.0, hub_top_z))
            .expect("rotating hub boundary velocity");

        assert_close(rotating_velocity[0], 0.0);
        assert!(rotating_velocity[1] > 0.0);
        assert_close(rotating_velocity[2], 0.0);

        let shroud_wall_point = Point3::new(
            geometry.shroud.inner_radius_m + 0.5 * geometry.shroud.wall_thickness_m,
            0.0,
            0.0,
        );
        let stationary_velocity = conditions
            .velocity_at_point(shroud_wall_point)
            .expect("stationary shroud boundary velocity");

        assert_close(stationary_velocity[0], 0.0);
        assert_close(stationary_velocity[1], 0.0);
        assert_close(stationary_velocity[2], 0.0);
    }

    #[test]
    fn cfd_mean_flow_solve_produces_finite_fields() {
        let config = PcFanConfig {
            mesh: PcFanMeshResolution {
                radial_segments: 2,
                chord_segments: 2,
                circumferential_segments: 8,
                harmonic_count: 2,
            },
            cfd: PcFanCfdConfig {
                flow_model: PcFanFlowModel::IncompressibleCfd,
                cells_x: 2,
                cells_y: 2,
                cells_z: 2,
                max_picard_iterations: 1,
                nonlinear_tolerance: 1.0e-4,
                ..PcFanCfdConfig::default()
            },
            ..PcFanConfig::default_120mm()
        };
        let solution = solve_pc_fan_cfd_mean_flow(&config, &config.default_design_variables());

        assert_eq!(solution.fem_node_count, solution.nodes.len());
        assert_eq!(solution.velocity_m_s.len(), solution.nodes.len());
        assert_eq!(solution.pressure_pa.len(), solution.nodes.len());
        assert!(solution.fem_element_count > 0);
        assert!(solution.boundary_dof_count > 0);
        assert_eq!(
            solution.boundary_dof_count,
            side_wall_boundary_dof_count(&solution, &config),
            "solid fan surfaces are represented by immersed penalties, not extra Dirichlet DOFs"
        );
        assert_eq!(solution.iterations, 1);
        assert!(solution.residual.is_finite());
        assert!(solution.inlet_airflow_cfm.is_finite());
        assert!(solution.airflow_cfm.is_finite());
        assert!(solution.pressure_drop_pa.is_finite());
        assert!(solution.max_velocity_m_s > 0.0);
        assert!(solution.reynolds_number > 0.0);
        let actuator = solution
            .actuator_loading
            .expect("actuator-force CFD must publish its integrated loading");
        assert!(actuator.force_on_fluid_n.into_iter().all(f64::is_finite));
        assert!(actuator.torque_on_fluid_nm.is_finite());
        assert!(actuator.angular_speed_rad_s.is_finite());
        assert!(actuator.shaft_power_to_fluid_w.is_finite());
        assert!(
            actuator.force_on_fluid_n[2] > 0.0,
            "blade actuator must apply forward axial force to the fluid: {actuator:?}"
        );
    }

    #[test]
    fn cfd_mean_flow_vtu_writer_emits_field_file() {
        let config = PcFanConfig {
            mesh: PcFanMeshResolution {
                radial_segments: 2,
                chord_segments: 2,
                circumferential_segments: 8,
                harmonic_count: 2,
            },
            cfd: PcFanCfdConfig {
                flow_model: PcFanFlowModel::IncompressibleCfd,
                cells_x: 2,
                cells_y: 2,
                cells_z: 2,
                max_picard_iterations: 1,
                nonlinear_tolerance: 1.0e-4,
                ..PcFanCfdConfig::default()
            },
            ..PcFanConfig::default_120mm()
        };
        let solution = solve_pc_fan_cfd_mean_flow(&config, &config.default_design_variables());
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("pcfan-cfd-flow.vtu");

        solution.write_vtu(&path).expect("write CFD flow VTU");

        let vtu = std::fs::read_to_string(&path).expect("read CFD flow VTU");
        assert!(!vtu.is_empty());
        for field in [
            "velocity_m_s",
            "pressure_pa",
            "picard_converged",
            "conservation_relative_mass_imbalance",
            "conservation_relative_momentum_imbalance",
            "conservation_relative_pressure_closure",
            "conservation_relative_energy_imbalance",
            "conservation_pressure_power_w",
            "conservation_shaft_power_to_fluid_w",
        ] {
            assert!(vtu.contains(field), "VTU is missing field {field}");
        }
    }

    #[test]
    fn incompressible_cfd_flow_model_drives_evaluation_mean_flow() {
        let config = PcFanConfig {
            mesh: PcFanMeshResolution {
                radial_segments: 2,
                chord_segments: 2,
                circumferential_segments: 8,
                harmonic_count: 2,
            },
            cfd: PcFanCfdConfig {
                flow_model: PcFanFlowModel::IncompressibleCfd,
                cells_x: 2,
                cells_y: 2,
                cells_z: 2,
                max_picard_iterations: 1,
                nonlinear_tolerance: 1.0e-4,
                ..PcFanCfdConfig::default()
            },
            ..PcFanConfig::default_120mm()
        };
        let design = config.default_design_variables();
        let cfd = solve_pc_fan_cfd_mean_flow(&config, &design);
        let flow = solve_pc_fan_mean_flow(&config, &design);
        let evaluation = evaluate_pc_fan(&config, &design);

        assert_close(flow.airflow_cfm, cfd.airflow_cfm);
        assert_close(flow.pressure_rise_pa, cfd.pressure_drop_pa.abs());
        assert_close(flow.reynolds_number, cfd.reynolds_number);
        assert_close(evaluation.airflow_cfm, cfd.airflow_cfm);
        assert!(evaluation.acoustic_level_db.is_finite());
        assert!(evaluation.objective.is_finite());
    }

    #[test]
    fn checked_cfd_evaluation_rejects_unconverged_fields_before_objective_use() {
        let config = PcFanConfig {
            mesh: PcFanMeshResolution {
                radial_segments: 2,
                chord_segments: 2,
                circumferential_segments: 8,
                harmonic_count: 2,
            },
            cfd: PcFanCfdConfig {
                flow_model: PcFanFlowModel::IncompressibleCfd,
                cells_x: 2,
                cells_y: 2,
                cells_z: 2,
                max_picard_iterations: 1,
                nonlinear_tolerance: 1.0e-4,
                ..PcFanCfdConfig::default()
            },
            ..PcFanConfig::default_120mm()
        };
        let design = config.default_design_variables();
        let mut cfd = solve_pc_fan_cfd_mean_flow(&config, &design);
        cfd.converged = false;
        cfd.residual = 0.0;
        assert!(matches!(
            try_evaluate_pc_fan_from_cfd_solution(&config, &design, &cfd),
            Err(PcFanCfdEvaluationError::NonConverged { .. })
        ));

        cfd.converged = true;
        cfd.residual = 0.0;
        cfd.mass_balance_error_kg_s = 0.0;
        let evaluation = try_evaluate_pc_fan_from_cfd_solution(&config, &design, &cfd)
            .expect("converged finite CFD field must be admissible");
        assert!(evaluation.objective.is_finite());

        cfd.mass_balance_error_kg_s = config.operating_point.air_density_kg_m3;
        assert!(matches!(
            try_evaluate_pc_fan_from_cfd_solution(&config, &design, &cfd),
            Err(PcFanCfdEvaluationError::MassImbalance { .. })
        ));

        cfd.mass_balance_error_kg_s = 0.0;
        cfd.actuator_loading
            .as_mut()
            .expect("actuator-force fixture must publish loading")
            .shaft_power_to_fluid_w = f64::NAN;
        assert!(matches!(
            try_evaluate_pc_fan_from_cfd_solution(&config, &design, &cfd),
            Err(PcFanCfdEvaluationError::InvalidActuatorLoading)
        ));
    }

    #[test]
    fn cfd_flow_is_fan_forced_not_target_prescribed() {
        let base_config = PcFanConfig {
            mesh: PcFanMeshResolution {
                radial_segments: 2,
                chord_segments: 2,
                circumferential_segments: 8,
                harmonic_count: 2,
            },
            cfd: PcFanCfdConfig {
                flow_model: PcFanFlowModel::IncompressibleCfd,
                cells_x: 2,
                cells_y: 2,
                cells_z: 2,
                max_picard_iterations: 1,
                nonlinear_tolerance: 1.0e-4,
                ..PcFanCfdConfig::default()
            },
            ..PcFanConfig::default_120mm()
        };
        let mut low_target = base_config.clone();
        low_target.operating_point.target_airflow_cfm = 20.0;
        let mut high_target = base_config;
        high_target.operating_point.target_airflow_cfm = 80.0;
        let design = low_target.default_design_variables();

        let low_flow = solve_pc_fan_cfd_mean_flow(&low_target, &design);
        let high_flow = solve_pc_fan_cfd_mean_flow(&high_target, &design);

        assert_close(low_flow.airflow_cfm, high_flow.airflow_cfm);
        assert_close(low_flow.inlet_airflow_cfm, high_flow.inlet_airflow_cfm);
    }

    #[test]
    fn cfd_actuator_forcing_responds_to_blade_geometry() {
        let config = PcFanConfig {
            mesh: PcFanMeshResolution {
                radial_segments: 2,
                chord_segments: 2,
                circumferential_segments: 8,
                harmonic_count: 2,
            },
            cfd: PcFanCfdConfig {
                flow_model: PcFanFlowModel::IncompressibleCfd,
                cells_x: 2,
                cells_y: 2,
                cells_z: 2,
                max_picard_iterations: 1,
                nonlinear_tolerance: 1.0e-4,
                ..PcFanCfdConfig::default()
            },
            ..PcFanConfig::default_120mm()
        };
        let baseline_design = config.default_design_variables();
        let mut deformed_design = baseline_design.clone();
        for section in &mut deformed_design.blade_sections {
            section.chord_m *= 1.45;
            section.twist_deg += 18.0;
            section.camber_m += 0.001;
        }

        let baseline = solve_pc_fan_cfd_mean_flow(&config, &baseline_design);
        let deformed = solve_pc_fan_cfd_mean_flow(&config, &deformed_design);

        let velocity_delta = (deformed.max_velocity_m_s - baseline.max_velocity_m_s).abs();
        let pressure_delta = (deformed.pressure_drop_pa - baseline.pressure_drop_pa).abs();
        let airflow_delta = (deformed.airflow_cfm - baseline.airflow_cfm).abs();

        assert!(
            velocity_delta.max(pressure_delta).max(airflow_delta) > 1.0e-8,
            "blade-derived actuator forcing should change with blade geometry: \
             velocity delta {velocity_delta:.3e}, pressure delta {pressure_delta:.3e}, \
             airflow delta {airflow_delta:.3e}"
        );
    }

    #[test]
    fn spline_controls_deform_blades_and_shroud_without_moving_mounts() {
        let config = PcFanConfig {
            mesh: PcFanMeshResolution {
                radial_segments: 6,
                chord_segments: 3,
                circumferential_segments: 12,
                harmonic_count: 3,
            },
            ..PcFanConfig::default_120mm()
        };
        let baseline_design = config.default_design_variables();
        let baseline_geometry = config.build_geometry(&baseline_design);
        let mut deformed_design = baseline_design.clone();
        deformed_design.blade_path[2].tangential_offset_m = 0.006;
        deformed_design.shroud_radius_offsets_m[2] = -0.0015;

        let deformed_geometry = config.build_geometry(&deformed_design);

        assert_close(
            deformed_design.blade_path_at(0.5).tangential_offset_m,
            0.006,
        );
        assert_close(
            deformed_geometry.shroud.radius_at_depth_fraction(0.5),
            config.dimensions.shroud_inner_radius_m - 0.0015,
        );
        assert_eq!(
            baseline_geometry.frame_mounts,
            deformed_geometry.frame_mounts
        );
        assert!(
            deformed_geometry.blades[0]
                .panels
                .iter()
                .flat_map(|panel| panel.vertices)
                .any(|vertex| vertex.y.abs() > 1.0e-4)
        );
    }

    #[test]
    fn merged_root_contact_is_accepted() {
        let config = PcFanConfig::default_120mm();
        let design = config.default_design_variables();
        let mut geometry = config.build_geometry(&design);
        geometry.blades.truncate(2);

        let root = Point3::new(config.dimensions.hub_radius_m, 0.0, 0.0);
        let root_panel = PcFanBladePanel {
            vertices: [root; 4],
            blade_index: 0,
            span_fraction: 0.0,
            chord_fraction: 0.5,
            thickness_m: 0.0,
            area: 0.0,
        };
        geometry.blades[0].panels = vec![root_panel];
        geometry.blades[1].panels = vec![PcFanBladePanel {
            blade_index: 1,
            ..root_panel
        }];

        let status = geometry.validate_blade_intersections();

        assert!(status.is_valid);
        assert_eq!(status.invalid_intersection_count, 0);
    }

    #[test]
    fn interior_blade_overlap_is_rejected() {
        let config = PcFanConfig::default_120mm();
        let design = config.default_design_variables();
        let mut geometry = config.build_geometry(&design);
        geometry.blades.truncate(2);

        let mut overlapping_panel = geometry.blades[0]
            .panels
            .iter()
            .copied()
            .find(|panel| panel.span_fraction > 0.6)
            .expect("interior panel");
        geometry.blades[0].panels = vec![overlapping_panel];
        overlapping_panel.blade_index = 1;
        geometry.blades[1].panels = vec![overlapping_panel];

        let status = geometry.validate_blade_intersections();

        assert!(!status.is_valid);
        assert!(status.invalid_intersection_count > 0);
    }

    #[test]
    fn airflow_penalty_increases_when_target_flow_is_missed() {
        let config = PcFanConfig::default_120mm();
        let design = config.default_design_variables();
        let baseline = evaluate_pc_fan(&config, &design);

        let mut matched_config = config.clone();
        matched_config.operating_point.target_airflow_cfm = baseline.airflow_cfm;
        let matched = evaluate_pc_fan(&matched_config, &design);

        let mut missed_config = config;
        missed_config.operating_point.target_airflow_cfm = baseline.airflow_cfm + 25.0;
        let missed = evaluate_pc_fan(&missed_config, &design);

        assert!(baseline.airflow_cfm > 0.0);
        assert!(missed.airflow_penalty > matched.airflow_penalty);
        assert!(missed.objective > matched.objective);
    }

    #[test]
    fn flow_and_acoustic_solve_produces_harmonic_levels() {
        let config = PcFanConfig {
            mesh: PcFanMeshResolution {
                harmonic_count: 4,
                ..PcFanMeshResolution::default()
            },
            ..PcFanConfig::default_120mm()
        };
        let design = config.default_design_variables();
        let evaluation = evaluate_pc_fan(&config, &design);

        assert!(evaluation.flow.axial_velocity_m_s > 0.0);
        assert!(evaluation.flow.pressure_rise_pa > 0.0);
        assert!(evaluation.flow.reynolds_number > 0.0);
        assert!(evaluation.flow.fixed_hardware.blockage_ratio > 0.0);
        assert!(evaluation.flow.fixed_hardware.total_loss_factor < 1.0);
        assert_eq!(evaluation.acoustics.harmonics.len(), 4);
        assert_close(
            evaluation.acoustics.blade_passing_frequency_hz,
            config.blade_count as f64 * config.operating_point.rpm / 60.0,
        );
        assert!(evaluation.acoustic_level_db.is_finite());
        assert!(evaluation.sound_power_level_db.is_finite());
        assert!(
            evaluation
                .acoustics
                .fixed_hardware_noise
                .combined_spl_db
                .is_finite()
        );
    }

    #[test]
    fn installed_forward_model_scales_with_rpm() {
        let mut low_config = PcFanConfig::default_120mm();
        low_config.operating_point.rpm = 450.0;
        let low_design = low_config.default_design_variables();
        let low = evaluate_pc_fan(&low_config, &low_design);

        let mut high_config = low_config.clone();
        high_config.operating_point.rpm = 900.0;
        let high_design = high_config.default_design_variables();
        let high = evaluate_pc_fan(&high_config, &high_design);

        assert!(high.airflow_cfm > low.airflow_cfm);
        assert!(high.flow.pressure_rise_pa > low.flow.pressure_rise_pa);
        assert!(high.acoustic_level_db > low.acoustic_level_db);
    }

    #[test]
    fn installed_forward_model_responds_to_blade_loading() {
        let config = PcFanConfig::default_120mm();
        let baseline_design = config.default_design_variables();
        let mut loaded_design = baseline_design.clone();
        for section in &mut loaded_design.blade_sections {
            section.chord_m *= 1.20;
            section.twist_deg += 5.0;
            section.camber_m *= 1.15;
        }

        let baseline = evaluate_pc_fan(&config, &baseline_design);
        let loaded = evaluate_pc_fan(&config, &loaded_design);

        assert!(loaded.airflow_cfm > baseline.airflow_cfm);
        assert!(loaded.flow.pressure_rise_pa > baseline.flow.pressure_rise_pa);
        assert!(loaded.acoustic_level_db > baseline.acoustic_level_db);
    }

    #[test]
    fn installed_forward_model_penalizes_tip_clearance() {
        let mut config = PcFanConfig::default_120mm();
        config.dimensions.frame_side_m = 0.128;
        config.dimensions.shroud_inner_radius_m = 0.055;
        let tight_design = config.default_design_variables();
        let mut loose_design = tight_design.clone();
        for offset in &mut loose_design.shroud_radius_offsets_m {
            *offset += 0.002;
        }

        let tight = evaluate_pc_fan(&config, &tight_design);
        let loose = evaluate_pc_fan(&config, &loose_design);

        assert!(loose.airflow_cfm < tight.airflow_cfm);
        assert!(loose.acoustics.tip_leakage_spl_db > tight.acoustics.tip_leakage_spl_db);
    }

    #[test]
    fn fixed_hardware_blockage_reduces_flow_and_adds_noise() {
        let config = PcFanConfig {
            mesh: PcFanMeshResolution {
                radial_segments: 5,
                chord_segments: 3,
                circumferential_segments: 16,
                harmonic_count: 3,
            },
            ..PcFanConfig::default_120mm()
        };
        let mut open_config = config.clone();
        // Keep the envelope admissible while making its blockage negligible.
        open_config.dimensions.frame_wall_thickness_m = 1.0e-9;
        open_config.dimensions.support_strut_width_m = 1.0e-9;
        open_config.dimensions.motor_mount_radius_m = open_config.dimensions.hub_radius_m;
        open_config.dimensions.motor_mount_depth_m = 0.0;

        let design = config.default_design_variables();
        let open_design = open_config.default_design_variables();
        let blocked = evaluate_pc_fan(&config, &design);
        let open = evaluate_pc_fan(&open_config, &open_design);

        assert!(
            blocked.flow.fixed_hardware.blockage_ratio > open.flow.fixed_hardware.blockage_ratio
        );
        assert!(
            blocked.flow.fixed_hardware.total_loss_factor
                < open.flow.fixed_hardware.total_loss_factor
        );
        assert!(blocked.airflow_cfm < open.airflow_cfm);
        assert!(
            blocked.acoustics.fixed_hardware_noise.combined_spl_db
                > open.acoustics.fixed_hardware_noise.combined_spl_db
        );
    }

    #[test]
    fn shape_gradient_matches_finite_difference_objective() {
        // `solve_pc_fan_shape_gradient` returns the checked discrete-adjoint
        // full-objective gradient of the installed reduced-order model. This focused case
        // keeps broadband and geometric penalties off so regressions in the
        // airflow + tonal path remain easy to diagnose.
        let mut config = PcFanConfig {
            mesh: PcFanMeshResolution {
                radial_segments: 5,
                chord_segments: 3,
                circumferential_segments: 12,
                harmonic_count: 3,
            },
            optimizer: PcFanOptimizerConfig {
                finite_difference_step: 2.0e-5,
                ..PcFanOptimizerConfig::default()
            },
            weights: PcFanOptimizationWeights {
                airflow_band_fraction: 0.0,
                spectral_spread_weight: 0.0,
                directivity_spread_weight: 0.0,
                shroud_clearance_penalty_weight: 0.0,
                frame_containment_penalty_weight: 0.0,
                // Disable for this focused gradient test: the hub-attachment
                // penalty would add an unrelated geometric term.
                hub_attachment_penalty_weight: 0.0,
                ..PcFanOptimizationWeights::default()
            },
            broadband: PcFanBroadbandConfig::off(),
            ..PcFanConfig::default_120mm()
        };
        let mut design = config.default_design_variables();
        for (i, control) in design.blade_path.iter_mut().enumerate() {
            control.tangential_offset_m = 0.001 + 0.0001 * i as f64;
            control.axial_offset_m = 0.0003;
        }
        for (i, section) in design.blade_sections.iter_mut().enumerate() {
            section.camber_m = 0.0006 + 0.00005 * i as f64;
            section.thickness_m = 0.0011 + 0.00003 * i as f64;
            section.twist_deg = 7.0 + i as f64;
        }
        for offset in &mut design.shroud_radius_offsets_m {
            *offset = 0.0004;
        }
        let baseline = evaluate_pc_fan(&config, &design);
        config.operating_point.target_airflow_cfm = baseline.airflow_cfm + 6.0;

        let gradient = solve_pc_fan_shape_gradient(&config, &design);

        assert!(
            config
                .build_geometry(&design)
                .validate_blade_intersections()
                .is_valid
        );
        assert!(gradient.residual.max_abs < 1.0e-10);
        assert!(gradient.adjoint.airflow_multiplier.is_finite());
        assert!(gradient.adjoint.airflow_multiplier.abs() > f64::EPSILON);
        assert!(
            gradient
                .adjoint
                .acoustic_pressure_multipliers
                .iter()
                .any(|value| value.abs() > f64::EPSILON),
            "reduced-order legacy entry point must expose non-zero state multipliers"
        );
        for (index, actual) in gradient.objective_gradient.iter().enumerate() {
            let fd = finite_difference_objective_component(&config, &design, index);
            let scale = fd.abs().max(actual.abs()).max(1.0);
            let relative_error = (actual - fd).abs() / scale;
            assert!(
                relative_error < 5.0e-3,
                "gradient[{index}]: actual={actual:.6e}, fd={fd:.6e}, rel={relative_error:.2e}",
            );
        }
    }

    #[test]
    fn shape_gradient_matches_finite_difference_with_trailing_edge() {
        // Same setup as `shape_gradient_matches_finite_difference_objective`,
        // but with BPM trailing-edge noise enabled.
        let mut config = PcFanConfig {
            mesh: PcFanMeshResolution {
                radial_segments: 5,
                chord_segments: 3,
                circumferential_segments: 12,
                harmonic_count: 3,
            },
            optimizer: PcFanOptimizerConfig {
                finite_difference_step: 2.0e-5,
                ..PcFanOptimizerConfig::default()
            },
            weights: PcFanOptimizationWeights {
                airflow_band_fraction: 0.0,
                spectral_spread_weight: 0.0,
                directivity_spread_weight: 0.0,
                shroud_clearance_penalty_weight: 0.0,
                frame_containment_penalty_weight: 0.0,
                // Disable for this focused gradient test: the hub-attachment
                // penalty would add an unrelated geometric term.
                hub_attachment_penalty_weight: 0.0,
                ..PcFanOptimizationWeights::default()
            },
            broadband: PcFanBroadbandConfig {
                tip_leakage_enabled: false,
                strut_motor_enabled: true,
                trailing_edge_enabled: true,
                // Disable the duct cut-off filter for this test to isolate
                // trailing-edge objective-gradient propagation from the
                // filter's own shape sensitivities through
                // shroud_inner_radius_m + depth_m).
                duct_cutoff_enabled: false,
                // LE inflow is zero at TI=0 (the operating point's
                // default), so leaving the flag on costs nothing and
                // keeps this test isolated to the trailing-edge path.
                leading_edge_inflow_enabled: true,
            },
            ..PcFanConfig::default_120mm()
        };
        let mut design = config.default_design_variables();
        for (i, control) in design.blade_path.iter_mut().enumerate() {
            control.tangential_offset_m = 0.001 + 0.0001 * i as f64;
            control.axial_offset_m = 0.0003;
        }
        for (i, section) in design.blade_sections.iter_mut().enumerate() {
            section.camber_m = 0.0006 + 0.00005 * i as f64;
            section.thickness_m = 0.0011 + 0.00003 * i as f64;
            section.twist_deg = 7.0 + i as f64;
        }
        for offset in &mut design.shroud_radius_offsets_m {
            *offset = 0.0004;
        }
        let baseline = evaluate_pc_fan(&config, &design);
        config.operating_point.target_airflow_cfm = baseline.airflow_cfm + 6.0;

        // Sanity check: TE must actually be contributing — otherwise this
        // test silently degenerates to the existing "broadband off" case.
        assert!(
            baseline.acoustics.trailing_edge_spl_db > 0.0,
            "TE noise should contribute meaningfully at the test config: {} dB",
            baseline.acoustics.trailing_edge_spl_db
        );

        let gradient = solve_pc_fan_shape_gradient(&config, &design);

        assert!(gradient.residual.max_abs < 1.0e-10);
        for (index, actual) in gradient.objective_gradient.iter().enumerate() {
            let fd = finite_difference_objective_component(&config, &design, index);
            let scale = fd.abs().max(actual.abs()).max(1.0);
            let relative_error = (actual - fd).abs() / scale;
            assert!(
                relative_error < 5.0e-3,
                "TE gradient mismatch at gradient[{index}]: actual={actual:.6e}, fd={fd:.6e}, rel={relative_error:.2e}",
            );
        }
    }

    #[test]
    fn shape_gradient_matches_default_full_objective() {
        let config = PcFanConfig {
            mesh: PcFanMeshResolution {
                radial_segments: 4,
                chord_segments: 3,
                circumferential_segments: 12,
                harmonic_count: 3,
            },
            optimizer: PcFanOptimizerConfig {
                finite_difference_step: 2.0e-5,
                ..PcFanOptimizerConfig::default()
            },
            ..PcFanConfig::default_120mm()
        };
        let design = config.default_design_variables();
        let gradient = solve_pc_fan_shape_gradient(&config, &design);

        assert!(gradient.residual.max_abs < 1.0e-10);
        for (index, actual) in gradient.objective_gradient.iter().enumerate() {
            let fd = finite_difference_objective_component(&config, &design, index);
            let scale = fd.abs().max(actual.abs()).max(1.0);
            let relative_error = (actual - fd).abs() / scale;
            assert!(
                relative_error < 5.0e-3,
                "default gradient[{index}]: actual={actual:.6e}, fd={fd:.6e}, rel={relative_error:.2e}",
            );
        }
    }

    #[test]
    fn reduced_discrete_adjoint_matches_full_objective_finite_difference() {
        let config = PcFanConfig {
            mesh: PcFanMeshResolution {
                radial_segments: 4,
                chord_segments: 3,
                circumferential_segments: 12,
                harmonic_count: 3,
            },
            optimizer: PcFanOptimizerConfig {
                finite_difference_step: 2.0e-5,
                ..PcFanOptimizerConfig::default()
            },
            ..PcFanConfig::default_120mm()
        };
        let design = config.default_design_variables();
        let adjoint = solve_pc_fan_shape_gradient_reduced_adjoint(&config, &design)
            .expect("reduced-order adjoint should be available");
        let finite_difference = solve_pc_fan_shape_gradient(&config, &design);

        assert!(
            !adjoint.adjoint_is_finite_difference,
            "reduced adjoint must not carry the FD placeholder tag"
        );
        assert!(adjoint.adjoint.airflow_multiplier.is_finite());
        assert!(adjoint.adjoint.airflow_multiplier.abs() > 0.0);
        assert_eq!(adjoint.adjoint.acoustic_pressure_multipliers.len(), 3);
        assert!(
            adjoint
                .adjoint
                .acoustic_pressure_multipliers
                .iter()
                .all(|multiplier| multiplier.is_finite())
        );
        assert!(
            adjoint
                .adjoint
                .acoustic_pressure_multipliers
                .iter()
                .any(|multiplier| multiplier.abs() > 0.0)
        );
        assert!(adjoint.residual.max_abs < 1.0e-10);
        for (index, (actual, reference)) in adjoint
            .objective_gradient
            .iter()
            .zip(finite_difference.objective_gradient.iter())
            .enumerate()
        {
            let scale = actual.abs().max(reference.abs()).max(1.0);
            let relative_error = (actual - reference).abs() / scale;
            assert!(
                relative_error < 5.0e-3,
                "adjoint gradient[{index}]: actual={actual:.6e}, fd={reference:.6e}, rel={relative_error:.2e}"
            );
        }
    }

    #[test]
    fn optimizer_preserves_or_improves_tiny_deterministic_case() {
        let mut config = PcFanConfig {
            mesh: PcFanMeshResolution {
                radial_segments: 3,
                chord_segments: 2,
                circumferential_segments: 8,
                harmonic_count: 2,
            },
            weights: PcFanOptimizationWeights {
                acoustic_weight: 0.0,
                airflow_penalty_weight: 1.0,
                collision_penalty_weight: 1.0e6,
                airflow_band_fraction: 0.0,
                spectral_spread_weight: 0.0,
                directivity_spread_weight: 0.0,
                operating_spread_weight: 0.0,
                shroud_clearance_min_m: 0.0,
                shroud_clearance_penalty_weight: 0.0,
                frame_containment_penalty_weight: 0.0,
                hub_attachment_penalty_weight: 0.0,
                tonality_penalty_weight: 0.0,
                ..PcFanOptimizationWeights::default()
            },
            optimizer: PcFanOptimizerConfig {
                max_iterations: 2,
                finite_difference_step: 5.0e-4,
                gradient_tolerance: 1.0e-6,
                objective_tolerance: 1.0e-9,
                memory: 3,
                freeze_shroud: false,
                require_positive_camber: false,
            },
            ..PcFanConfig::default_120mm()
        };
        let baseline_design = config.default_design_variables();
        let baseline = evaluate_pc_fan(&config, &baseline_design);
        config.operating_point.target_airflow_cfm = baseline.airflow_cfm + 8.0;
        let baseline = evaluate_pc_fan(&config, &baseline_design);

        let optimized = optimize_pc_fan(config);

        assert!(optimized.evaluation.objective <= baseline.objective + 1.0e-9);
        assert_eq!(optimized.geometry.blades.len(), DEFAULT_BLADE_COUNT);
        assert!(optimized.optimization.function_evaluations > 0);
        assert!(
            optimized.geometry.shroud.outer_radius_m
                <= 0.5 * optimized.geometry.frame_box.inner_side_m + f64::EPSILON
        );
    }

    #[test]
    fn airflow_band_lock_keeps_optimised_design_inside_target_window() {
        // End-to-end check that the warm-restart loop drives the
        // converged design into the ±10% band. We pick a target equal to
        // the baseline airflow (so the band is reachable without large
        // shape changes) and verify the optimised design stays inside.
        let mut config = PcFanConfig {
            mesh: PcFanMeshResolution {
                radial_segments: 3,
                chord_segments: 2,
                circumferential_segments: 8,
                harmonic_count: 2,
            },
            optimizer: PcFanOptimizerConfig {
                max_iterations: 4,
                finite_difference_step: 5.0e-4,
                gradient_tolerance: 1.0e-6,
                objective_tolerance: 1.0e-9,
                memory: 3,
                ..PcFanOptimizerConfig::default()
            },
            ..PcFanConfig::default_120mm()
        };
        config.weights.airflow_band_fraction = 0.20;
        config.weights.airflow_penalty_weight = 0.5;
        let baseline_design = config.default_design_variables();
        let baseline = evaluate_pc_fan(&config, &baseline_design);
        // Target = baseline so the starting design is already inside the
        // band; the warm-restart logic should never need to fire, and
        // the optimiser is free to focus on the noise objective.
        config.operating_point.target_airflow_cfm = baseline.airflow_cfm;

        let optimized = optimize_pc_fan(config.clone());

        let band =
            config.weights.airflow_band_fraction * config.operating_point.target_airflow_cfm.abs();
        let margin =
            (optimized.evaluation.airflow_cfm - config.operating_point.target_airflow_cfm).abs();
        assert!(
            margin <= band + 1.0e-3,
            "airflow {} not within ±{} of target {}: margin {}",
            optimized.evaluation.airflow_cfm,
            band,
            config.operating_point.target_airflow_cfm,
            margin
        );
    }

    fn finite_difference_objective_component(
        config: &PcFanConfig,
        design: &PcFanDesignVariables,
        index: usize,
    ) -> f64 {
        let step = finite_difference_design_step(config, design, index);
        let mut plus = design.to_vector();
        let mut minus = design.to_vector();
        plus[index] += step;
        minus[index] -= step;

        let plus = evaluate_pc_fan(config, &PcFanDesignVariables::from_vector(plus)).objective;
        let minus = evaluate_pc_fan(config, &PcFanDesignVariables::from_vector(minus)).objective;

        (plus - minus) / (2.0 * step)
    }

    fn finite_difference_design_step(
        config: &PcFanConfig,
        design: &PcFanDesignVariables,
        index: usize,
    ) -> f64 {
        let scale = if index < BLADE_PATH_CONTROL_COUNT * 2 {
            0.01
        } else if index < BLADE_PATH_CONTROL_COUNT * 2 + BLADE_SECTION_STATION_COUNT * 4 {
            match (index - BLADE_PATH_CONTROL_COUNT * 2) % 4 {
                0 => config.dimensions.depth_m,
                1 => 0.004,
                2 => 0.001,
                _ => 10.0,
            }
        } else {
            0.003
        };

        config.optimizer.finite_difference_step * design.to_vector()[index].abs().max(scale)
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-9,
            "expected {expected}, got {actual}"
        );
    }

    fn assert_boundary_kind(domain: &PcFanCfdDomain, kind: PcFanCfdBoundaryKind) {
        assert!(
            domain.boundary_faces.iter().any(|face| face.kind == kind),
            "missing CFD boundary kind {kind:?}"
        );
    }

    fn sorted_unique_z_values(panels: &[PcFanSurfacePanel], kind: PcFanSurfaceKind) -> Vec<f64> {
        let mut values = panels
            .iter()
            .filter(|panel| panel.kind == kind)
            .flat_map(|panel| panel.vertices.iter().map(|vertex| vertex.z))
            .collect::<Vec<_>>();

        values.sort_by(f64::total_cmp);
        values.dedup_by(|a, b| (*a - *b).abs() < 1.0e-9);
        values
    }

    fn side_wall_boundary_dof_count(
        solution: &PcFanCfdFlowSolution,
        config: &PcFanConfig,
    ) -> usize {
        let domain = &solution.domain;
        let dx = (domain.max_corner.x - domain.min_corner.x) / config.cfd.cells_x.max(1) as f64;
        let dy = (domain.max_corner.y - domain.min_corner.y) / config.cfd.cells_y.max(1) as f64;
        let dz = (domain.max_corner.z - domain.min_corner.z) / config.cfd.cells_z.max(1) as f64;
        let tolerance = 1.0e-8 * dx.min(dy).min(dz).max(1.0);
        let side_wall_node_count = solution
            .nodes
            .iter()
            .filter(|node| {
                (node.x - domain.min_corner.x).abs() <= tolerance
                    || (node.x - domain.max_corner.x).abs() <= tolerance
                    || (node.y - domain.min_corner.y).abs() <= tolerance
                    || (node.y - domain.max_corner.y).abs() <= tolerance
            })
            .count();

        3 * side_wall_node_count
    }
}
