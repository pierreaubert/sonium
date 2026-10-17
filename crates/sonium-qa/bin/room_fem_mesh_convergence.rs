//! Release-facing physical RoomSim FEM three-level mesh-convergence gate.
//!
//! This binary is intentionally kept separate from the deterministic response
//! contract: it exercises three real full-domain FEM solves, retains nodal
//! fields and listener coordinates, and fails unless the adjacent response
//! errors meet the configured promotion budgets.

// Rust guideline compliant 2026-02-21

use sonium_qa::QaResult;
use sonium_sim_room::fem_runner::{
    FemMeshConfig, FemMeshConvergenceSpec, FemSourceModel, run_fem_mesh_convergence_ladder,
};
use sonium_xem::{
    BoundaryConfig, Point3D, RectangularRoom, RoomGeometry, RoomSimulation, Source, SurfaceConfig,
};

const CASE_ID: &str = "roomsim.fem-mesh-convergence.v1";
// Measured medium-to-fine listener changes are 7.0e-3 RMS / 8.7e-3 max
// (about 0.06 / 0.08 dB); gates sit at ~3x / ~2.3x measured. Phase remains
// in the complex-pressure comparison.
const RMS_TOLERANCE: f64 = 0.02;
const MAX_TOLERANCE: f64 = 0.02;
// Require a material decrease between the two adjacent mesh comparisons.
const CONTRACTION_TOLERANCE: f64 = 0.8;

fn build_simulation() -> RoomSimulation {
    let mut simulation = RoomSimulation::new(
        RoomGeometry::Rectangular(RectangularRoom::new(1.6, 1.5, 1.4)),
        vec![Source::omnidirectional(Point3D::new(0.4, 0.35, 0.35), 1.0)],
        vec![Point3D::new(1.2, 1.1, 1.0), Point3D::new(1.0, 0.55, 1.05)],
    );
    let matched = SurfaceConfig::Admittance {
        real: 1.0,
        imag: 0.0,
    };
    simulation.boundaries = BoundaryConfig {
        floor: matched.clone(),
        ceiling: matched.clone(),
        walls: matched,
        front_wall: None,
        back_wall: None,
        left_wall: None,
        right_wall: None,
    };
    simulation.frequencies = vec![400.0];
    simulation
}

fn run() -> Result<QaResult, String> {
    let simulation = build_simulation();
    // Resolve the point-source problem without changing its regularization:
    // 36/54/72 elements per wavelength on the three uniform meshes.
    let base = FemMeshConfig {
        elements_per_wavelength: 72.0,
        fine_radius: 0.4,
        modal_damping_ratio: 0.0,
        source_model: FemSourceModel::PointLoad,
        ..FemMeshConfig::default()
    };
    let mut coarse = base;
    coarse.grading_factor = 2.0;
    coarse.fine_radius = 0.0;
    let mut medium = base;
    medium.grading_factor = 4.0 / 3.0;
    medium.fine_radius = 0.0;
    let mut fine = base;
    fine.grading_factor = 1.0;
    fine.fine_radius = 0.0;

    let evidence = run_fem_mesh_convergence_ladder(
        &simulation,
        &[
            FemMeshConvergenceSpec::new("coarse", coarse),
            FemMeshConvergenceSpec::new("medium", medium),
            FemMeshConvergenceSpec::new("fine", fine),
        ],
        1.0e-12,
        1.0e-9,
        1.0e-12,
        RMS_TOLERANCE,
        MAX_TOLERANCE,
    )
    .map_err(|error| format!("physical FEM mesh-convergence failed: {error}"))?;

    let rms_contraction = evidence.report.medium_to_fine_rms_relative_error
        / evidence
            .report
            .coarse_to_medium_rms_relative_error
            .max(1e-12);
    let max_contraction = evidence.report.medium_to_fine_max_relative_error
        / evidence
            .report
            .coarse_to_medium_max_relative_error
            .max(1e-12);
    if !rms_contraction.is_finite()
        || !max_contraction.is_finite()
        || rms_contraction > CONTRACTION_TOLERANCE
        || max_contraction > CONTRACTION_TOLERANCE
    {
        return Err(format!(
            "FEM listener response does not contract sufficiently: RMS ratio {rms_contraction:.3e}, maximum ratio {max_contraction:.3e}"
        ));
    }

    let listener_count = evidence.fine.field.listener_positions.len() as f64;
    let solved_samples = evidence.report.sample_count as f64;
    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric(
            "coarse_node_count",
            evidence.ladder.coarse.node_count as f64,
        )
        .metric(
            "medium_node_count",
            evidence.ladder.medium.node_count as f64,
        )
        .metric("fine_node_count", evidence.ladder.fine.node_count as f64)
        .metric(
            "coarse_element_count",
            evidence.ladder.coarse.element_count as f64,
        )
        .metric(
            "medium_element_count",
            evidence.ladder.medium.element_count as f64,
        )
        .metric(
            "fine_element_count",
            evidence.ladder.fine.element_count as f64,
        )
        .metric("listener_count", listener_count)
        .metric("solved_sample_count", solved_samples)
        .metric(
            "medium_to_fine_rms_relative_error",
            evidence.report.medium_to_fine_rms_relative_error,
        )
        .metric(
            "medium_to_fine_max_relative_error",
            evidence.report.medium_to_fine_max_relative_error,
        )
        .metric("coarse_to_medium_rms_relative_error", evidence.report.coarse_to_medium_rms_relative_error)
        .metric("coarse_to_medium_max_relative_error", evidence.report.coarse_to_medium_max_relative_error)
        .metric("rms_refinement_contraction", rms_contraction).maximum("rms_refinement_contraction", CONTRACTION_TOLERANCE)
        .metric("max_refinement_contraction", max_contraction).maximum("max_refinement_contraction", CONTRACTION_TOLERANCE)
        .maximum("medium_to_fine_rms_relative_error", RMS_TOLERANCE)
        .maximum("medium_to_fine_max_relative_error", MAX_TOLERANCE)
        .minimum("listener_count", 2.0)
        .minimum("solved_sample_count", 2.0)
        .input_checksum("analytic/deterministic-fixture:roomsim-physical-mesh-v1")
        .reference_checksum("production/full-domain-fem:listener-position-retained-v1")
        .dimension("1.6 x 1.5 x 1.4 m; one source; two listeners; 400 Hz")
        .formulation("three sequential full-domain CPU FEM P1 solves with retained nodal fields")
        .tolerance("fine-grid complex RMS <= 0.02 and max <= 0.02; both refinement contraction ratios <=0.8; strict listener/topology gates")
        .seed("roomsim-fem-mesh-convergence:cpu:v1"))
}

fn main() {
    match run() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("RoomSim physical mesh-convergence result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, error)
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .to_json_line()
                    .expect("RoomSim physical mesh-convergence failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
