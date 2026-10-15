use std::f64::consts::TAU;

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::physics::coupled::lumped_circuit::VoiceCoilParams;
use sonium_fem::physics::fluid::stokes::navier_stokes::NavierStokesConfig;
use sonium_fem::physics::fluid::stokes::time_stepping::BdfConfig;
use sonium_fem::spaces::{H1Space, VectorH1Space};
use sonium_sim_speaker_modeling::port_flow::box_driver::BoxDriverModel;
use sonium_sim_speaker_modeling::port_flow::{
    PortFlowPromotionGate, PortFlowRuntimeCase, PortFlowRuntimePlan, PortFlowRuntimeRefinement,
    RoundPortGeometry, build_round_port_mesh, run_port_flow_promotion,
};

const FREQUENCY_HZ: f64 = 80.0;
const BOX_VOLUME_M3: f64 = 2.0e-5;
const DENSITY_KG_PER_M3: f64 = 1.2;
const SOUND_SPEED_M_PER_S: f64 = 343.0;

fn plan() -> PortFlowRuntimePlan {
    PortFlowRuntimePlan {
        spatial_mesh_sizes_m: vec![0.020, 0.015, 0.010],
        temporal_dt_s: vec![1.0 / (FREQUENCY_HZ * 8.0), 1.0 / (FREQUENCY_HZ * 16.0)],
        frequency_hz: FREQUENCY_HZ,
        tail_cycles: 1,
        discarded_cycles: 1,
        box_volume_m3: BOX_VOLUME_M3,
        density_kg_per_m3: DENSITY_KG_PER_M3,
        sound_speed_m_per_s: SOUND_SPEED_M_PER_S,
        provenance: "focused-round-port-runtime.v1".into(),
    }
}

fn case(refinement: PortFlowRuntimeRefinement) -> PortFlowRuntimeCase {
    let geometry = RoundPortGeometry {
        duct_length_m: 0.04,
        duct_radius_m: 0.01,
        chamber_length_m: 0.02,
        chamber_radius_m: 0.01,
        flare_radius_m: 0.0,
        mesh_size_m: refinement.mesh_size_m,
    };
    let mesh = build_round_port_mesh(&geometry).expect("focused round-port mesh");
    let driver = BoxDriverModel {
        mms_kg: 0.02,
        rms_kg_per_s: 1.0,
        kms_n_per_m: 4000.0,
        voice_coil: VoiceCoilParams::linear(6.0, 0.5e-3, 10.0),
        compliance_force_fn: None,
        compliance_potential_fn: None,
        sd_m2: 0.01,
        box_volume_m3: Some(BOX_VOLUME_M3),
        medium_density_kg_per_m3: DENSITY_KG_PER_M3,
        medium_sound_speed_m_per_s: SOUND_SPEED_M_PER_S,
        beta: 0.25,
        gamma: 0.5,
    }
    .with_nonlinear_suspension(
        |x| 4000.0 * x + 2.0e8 * x.powi(3),
        |x| 2000.0 * x.powi(2) + 5.0e7 * x.powi(4),
    );
    let config = sonium_sim_speaker_modeling::port_flow::coupled::CoupledPortConfig {
        driver,
        ns_config: NavierStokesConfig {
            mu: 1.85e-5,
            rho: DENSITY_KG_PER_M3,
            use_residual_stabilisation: false,
            // Zero body force through `solve_bdf_step_cached`: the
            // force quadrature is identical every step, so caching
            // skips two full assemblies per step after the first.
            cache_body_force_assembly: true,
            ..Default::default()
        },
        bdf: BdfConfig::bdf2(
            refinement.dt_s,
            (refinement.discarded_cycles + refinement.tail_cycles) * refinement.samples_per_cycle,
        ),
        voltage_fn: Box::new(|time_s| 1.0e-4 * (TAU * FREQUENCY_HZ * time_s).sin()),
        under_relaxation: 1.0,
        exterior_pressure_pa: 0.0,
    };
    PortFlowRuntimeCase {
        mesh,
        velocity_space: VectorH1Space::new(PolynomialDegree::P2, 3),
        pressure_space: H1Space::new(PolynomialDegree::P1),
        config,
    }
}

#[test]
#[ignore = "bounded 3-D runtime ladder; run only as an explicit promotion campaign"]
fn runtime_runner_executes_independent_3d_spatial_and_temporal_ladders() {
    let mut gate = PortFlowPromotionGate::default();
    // This test exercises the runtime plumbing and finite-state contract. The
    // production 1% physical gate remains intentionally separate until the
    // independent runtime ladder meets it on a sufficiently resolved mesh.
    gate.max_spatial_observable_relative_change = 10.0;
    gate.max_temporal_observable_relative_change = 10.0;
    gate.max_cross_ladder_observable_relative_change = 10.0;
    gate.max_linear_relative_residual = 1.0;
    gate.max_box_conservation_relative_residual = 1.0;
    gate.max_cap_continuity_relative_error = 1.0;
    gate.max_energy_algebraic_relative_closure = 1.0;
    gate.max_energy_physical_relative_closure = 1.0;
    gate.max_energy_interface_relative_residual = 1.0;

    let promotion = run_port_flow_promotion(&plan(), gate, |refinement| Ok(case(refinement)))
        .expect("focused runtime ladder should produce finite evidence");
    assert_eq!(promotion.evidence.spatial_levels.len(), 3);
    assert_eq!(promotion.evidence.temporal_levels.len(), 2);
    assert_eq!(
        promotion.evidence.spatial_levels.last(),
        promotion.evidence.temporal_levels.last()
    );
    assert!(
        promotion
            .evidence
            .spatial_levels
            .iter()
            .chain(promotion.evidence.temporal_levels.iter())
            .all(|level| level.port_flow.real.is_finite()
                && level.box_pressure.real.is_finite()
                && level.energy_physical_relative_closure.is_finite())
    );
}
