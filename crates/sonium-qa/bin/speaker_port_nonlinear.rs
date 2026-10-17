//! Release-mode QA gates for the M1 resolved nonlinear speaker-port model.
//!
//! The default path always executes the 2D subcase (plane Poiseuille plus the
//! coupled slot-port sweep) and executes the nightly 3D subcase (round-port
//! Poiseuille plus the exact resolved-FEM reference ladder) only when
//! `SPEAKER_PORT_3D=1`. Each subcase emits an independent `QA_RESULT` record
//! with per-branch executed flags so a whole-binary SUCCESS never implies 3D
//! coverage when the 3D leg did not run.

use std::f64::consts::{LN_2, PI, TAU};
use std::time::Instant;

use num_complex::Complex64;
use rayon::prelude::*;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::{BoundaryType, ElementType, Mesh, Point, triangulate};
use sonium_fem::physics::acoustics::pressure::FluidProperties;
use sonium_fem::physics::coupled::lumped_circuit::VoiceCoilParams;
use sonium_fem::physics::fluid::stokes::navier_stokes::{
    BackendPreconditionerKind, BackendPressureSchurPreconditionerKind, LinearSolverBackend,
    LinearSolverCpuPreconditionerKind, LinearSolverKind, NavierStokesConfig, solve_bdf_step_cached,
};
use sonium_fem::physics::fluid::stokes::time_stepping::{BdfConfig, BdfState};
use sonium_fem::physics::fluid::stokes::traction::PressureTraction;
use sonium_fem::physics::fluid::stokes::workspace::{BdfLinearSolverKind, BdfWorkspace};
use sonium_fem::spaces::{FunctionSpace, H1Space, VectorH1Space};
use sonium_qa::QaResult;
use sonium_sim_speaker_modeling::port_flow::analysis::{
    CoupledComplexObservables, LinearReferenceGateConfig, LinearReferenceGateInput,
    LinearReferenceGateReport, coupled_complex_observables, dft_amplitude, thd_fraction,
    validate_coupled_against_linear_reference,
};
use sonium_sim_speaker_modeling::port_flow::box_driver::BoxDriverModel;
use sonium_sim_speaker_modeling::port_flow::coupled::{
    CoupledEnergySummary, CoupledPortConfig, CoupledPortResult, run_coupled_port,
};
use sonium_sim_speaker_modeling::port_flow::{
    ComplexPortTransfer, MARKER_BOX_CAP, MARKER_EXTERIOR_CAP, MARKER_WALL, RoundPortGeometry,
    RoundPortLinearReferenceConfig, SlotPortGeometry, build_round_port_mesh, build_slot_port_mesh,
    solve_round_port_linear_reference,
};
use sonium_sim_speaker_modeling::{AcousticMedium, EnclosureSpec, PortSpec};

const RHO: f64 = 1.2;
const SOUND_SPEED: f64 = 343.0;
const MU: f64 = 1.85e-5;
const BOX_VOLUME_M3: f64 = 0.005;
const DIAPHRAGM_AREA_M2: f64 = 0.01;
const X_MAX_M: f64 = 0.005;
const DUCT_LENGTH_M: f64 = 0.1;
const DUCT_HEIGHT_M: f64 = 0.02;
const END_CORRECTION_M: f64 = 4.0 * LN_2 / PI * DUCT_HEIGHT_M;
const DRIVE_LEVELS_V_PEAK: [f64; 3] = [0.01, 0.1, 1.0];
// Full 40-cycle convergence evidence at 252 samples/cycle gives a 9.88493%
// tiny-drive lumped-limit error (unchanged <10% gate), with 126→252 settled
// deltas of 0.47748% in amplitude, 0.06214% in complex Z, and 1.83310% in
// complex H. See Task-14 report for raw phasors and runtimes.
const COUPLED_SAMPLES_PER_CYCLE: usize = 252;
const COUPLED_TOTAL_CYCLES: usize = 40;
const COUPLED_TAIL_CYCLES: usize = 8;
const EXACT_TAIL_CYCLES: usize = 16;
const EXACT_DISCARDED_CYCLES: usize = 12;
const EXACT_SAMPLES_PER_CYCLE: usize = 128;
const EXACT_VOLTAGE_PEAK_V: f64 = 1.0e-4;

/// Machine-readable subcase identifiers. The 2D and 3D legs report independent
/// statuses so a whole-binary SUCCESS never implies 3D coverage when the
/// optional 3D branch did not run.
const CASE_ID_2D: &str = "speaker.port-nonlinear-2d.v1";
const CASE_ID_3D: &str = "speaker.port-nonlinear-3d.v1";

/// Environment switch for the nightly 3D leg. Any value other than "1" leaves
/// the 3D subcase skipped with explicit non-coverage evidence.
const ENV_SPEAKER_PORT_3D: &str = "SPEAKER_PORT_3D";

// Named acceptance tolerances for the 2D leg. Values match the historically
// enforced bounds; they are centralized here so the printed gates and the
// structured thresholds cannot drift apart.
const PLANE_POISEUILLE_MAX_RELATIVE_ERROR: f64 = 0.04;
const LUMPED_LIMIT_MAX_RELATIVE_ERROR: f64 = 0.10;
const LINEAR_THD_MAX_FRACTION: f64 = 0.01;
const COUPLED_ENERGY_PHYSICAL_CLOSURE_MAX: f64 = 0.04;
const COUPLED_ENERGY_ALGEBRAIC_CLOSURE_MAX: f64 = 1.0e-8;
const COUPLED_ENERGY_INTERFACE_RESIDUAL_MAX: f64 = 1.0e-8;
/// Relative scale below which a negative dissipation channel counts as
/// roundoff rather than an energy audit failure.
const DISSIPATION_NEGATIVITY_RELATIVE_TOLERANCE: f64 = 1.0e-12;

// Named acceptance tolerances for the 3D leg.
const ROUND_PORT_POISEUILLE_MAX_RELATIVE_ERROR: f64 = 0.08;
const EXACT_REFERENCE_CONVERGENCE_MAX: f64 = 0.01;
const EXACT_NONLINEAR_MESH_DT_MAX: f64 = 0.02;
const EXACT_HALF_DRIVE_LINEARITY_MAX: f64 = 0.01;
const EXACT_TAIL_DELTA_MAX: f64 = 0.005;
const EXACT_THD_MAX_FRACTION: f64 = 0.005;
const EXACT_CAP_CONTINUITY_MAX: f64 = 1.0e-6;

// Linear-reference gate bounds. These are passed explicitly to
// `validate_coupled_against_linear_reference` (instead of relying on its
// defaults) so the enforced gate and the structured 3D thresholds share one
// definition. Values match `LinearReferenceGateConfig::default()` except
// `GATE_MAX_ENERGY_PHYSICAL_RELATIVE_CLOSURE`, which is tightened below.
const GATE_MAX_IMPEDANCE_RELATIVE_ERROR: f64 = 0.03;
const GATE_MAX_FLOW_RATIO_RELATIVE_ERROR: f64 = 0.03;
const GATE_MAX_TRANSFER_IDENTITY_RELATIVE_ERROR: f64 = 0.02;
const GATE_MAX_CAP_CONTINUITY_RELATIVE_ERROR: f64 = 1.0e-6;
/// Reduced-3D cap-continuity gate. The coarse 10k-tet smoke mesh resolves
/// boundary fluxes to ~8.1e-4 (direct-LU cross-validated by the parity
/// probe across CPU/CUDA backends), so this gate admits
/// discretization-level mismatch and only rejects solver-scale
/// conservation loss. Per-step BDF convergence is enforced separately by
/// `solve_bdf_step_cached` (unconverged steps fail the run outright).
const REDUCED_3D_MAX_CAP_CONTINUITY_RELATIVE_ERROR: f64 = 5.0e-3;
const GATE_MAX_REFERENCE_RELATIVE_RESIDUAL: f64 = 1.0e-8;
const GATE_MAX_REFERENCE_FLUX_RELATIVE_DIFFERENCE: f64 = 1.0e-6;
const GATE_MAX_ENERGY_ALGEBRAIC_RELATIVE_CLOSURE: f64 = 1.0e-8;
// Deliberately tighter than `LinearReferenceGateConfig::default()`: matches the
// operative 2D coupled gate (measured 8.4e-3, 4.8x margin) and the pre-existing
// reduced-3D 0.04 energy envelope in this file.
const GATE_MAX_ENERGY_PHYSICAL_RELATIVE_CLOSURE: f64 = 0.04;
const GATE_MAX_ENERGY_INTERFACE_RELATIVE_RESIDUAL: f64 = 1.0e-8;
const GATE_MAX_THD_FRACTION: f64 = 0.005;
const GATE_MAX_MACH: f64 = 0.05;

/// Measured 2D plane-Poiseuille gate quantities.
struct PlanePoiseuilleMetrics {
    relative_error: f64,
    numerical_u_max: f64,
    exact_u_max: f64,
}

/// Measured 3D round-port Poiseuille gate quantities.
struct RoundPortPoiseuilleMetrics {
    relative_error: f64,
    numerical_u_max: f64,
    exact_u_max: f64,
    nodes: usize,
}

/// Measured lumped-limit gate quantities.
struct LumpedLimitMetrics {
    relative_error: f64,
    coupled_flow_m3s: f64,
    reference_flow_m3s: f64,
}

/// Worst-case compression/energy quantities across the drive sweep.
struct CompressionEnergyMetrics {
    max_physical_closure: f64,
    max_algebraic_closure: f64,
    max_interface_residual: f64,
    max_compression_db: f64,
    levels: usize,
}

/// Executed coarse/fine reference-convergence state shared by the full 3D leg
/// and the bounded exact-reference diagnostic tiers.
struct ExactReferenceConvergence {
    coarse_mesh: Mesh,
    fine_mesh: Mesh,
    coarse_geometry: RoundPortGeometry,
    fine_geometry: RoundPortGeometry,
    reference_config: RoundPortLinearReferenceConfig,
    fine_reference: ComplexPortTransfer,
    frequencies: [f64; 3],
    reference_z_delta: f64,
    reference_h_delta: f64,
}

/// Worst-case quantities across the executed 3D reference gates. Every field
/// must come from an executed solve; a 3D claim requires all of them.
struct ExactReferenceMetrics {
    reference_z_delta: f64,
    reference_h_delta: f64,
    nonlinear_mesh_z_delta: f64,
    nonlinear_mesh_h_delta: f64,
    dt_z_delta: f64,
    dt_h_delta: f64,
    half_drive_z_delta: f64,
    half_drive_h_delta: f64,
    worst_tail_z_delta: f64,
    worst_tail_h_delta: f64,
    gate_impedance_error: f64,
    gate_flow_ratio_error: f64,
    gate_identity_error: f64,
    gate_reference_residual: f64,
    gate_flux_difference: f64,
    gate_cap_continuity: f64,
    gate_energy_algebraic: f64,
    gate_energy_physical: f64,
    gate_energy_interface: f64,
    gate_thd: f64,
    gate_mach: f64,
    frequencies_gated: usize,
}

/// Complete evidence for the always-executed 2D subcase.
struct Nonlinear2dEvidence {
    plane_poiseuille: PlanePoiseuilleMetrics,
    frequency_hz: f64,
    lumped_limit: LumpedLimitMetrics,
    compression_energy: CompressionEnergyMetrics,
    linear_thd: f64,
    steps_completed_min: usize,
    steps_expected: usize,
}

/// Complete evidence for one executed 3D leg.
struct Nonlinear3dEvidence {
    poiseuille: RoundPortPoiseuilleMetrics,
    exact_reference: ExactReferenceMetrics,
}

/// Outcome of the optional 3D leg.
enum ThreeDOutcome {
    Executed(Nonlinear3dEvidence),
    Skipped,
    Failed(String),
}

/// Full default-path evidence: the 2D subcase always runs, the 3D subcase
/// runs only when explicitly enabled.
struct FullOutcome {
    two_d: Nonlinear2dEvidence,
    three_d: ThreeDOutcome,
}

/// Top-level run outcome. Diagnostic tiers keep their legacy markers and
/// carry no subcase evidence.
enum RunOutcome {
    Diagnostic,
    Full(Box<FullOutcome>),
}

/// A stage failure tagged with the subcase it belongs to.
#[derive(Debug)]
struct BranchError {
    branch: &'static str,
    message: String,
}

impl BranchError {
    fn new(branch: &'static str, message: String) -> Self {
        Self { branch, message }
    }
}

fn three_d_enabled() -> bool {
    std::env::var(ENV_SPEAKER_PORT_3D).as_deref() == Ok("1")
}

fn linear_reference_gate_config() -> LinearReferenceGateConfig {
    LinearReferenceGateConfig {
        max_impedance_relative_error: GATE_MAX_IMPEDANCE_RELATIVE_ERROR,
        max_flow_ratio_relative_error: GATE_MAX_FLOW_RATIO_RELATIVE_ERROR,
        max_transfer_identity_relative_error: GATE_MAX_TRANSFER_IDENTITY_RELATIVE_ERROR,
        max_cap_continuity_relative_error: GATE_MAX_CAP_CONTINUITY_RELATIVE_ERROR,
        max_reference_relative_residual: GATE_MAX_REFERENCE_RELATIVE_RESIDUAL,
        max_reference_flux_relative_difference: GATE_MAX_REFERENCE_FLUX_RELATIVE_DIFFERENCE,
        max_energy_algebraic_relative_closure: GATE_MAX_ENERGY_ALGEBRAIC_RELATIVE_CLOSURE,
        max_energy_physical_relative_closure: GATE_MAX_ENERGY_PHYSICAL_RELATIVE_CLOSURE,
        max_energy_interface_relative_residual: GATE_MAX_ENERGY_INTERFACE_RELATIVE_RESIDUAL,
        max_thd_fraction: GATE_MAX_THD_FRACTION,
        max_mach: GATE_MAX_MACH,
    }
}

fn marked_channel_mesh(length: f64, height: f64, nx: usize, ny: usize) -> Mesh {
    let dx = length / nx as f64;
    let dy = height / ny as f64;
    let mut points = Vec::with_capacity((nx + 1) * (ny + 1));
    for j in 0..=ny {
        for i in 0..=nx {
            let interior = i > 0 && i < nx && j > 0 && j < ny;
            let (mut x, mut y) = (i as f64 * dx, j as f64 * dy);
            if interior {
                // Match the deterministic unstructured Task-2 channel fixture.
                let h1 = (i as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add((j as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
                let h2 = (j as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add((i as u64).wrapping_mul(0x1656_67B1_9E37_79F9));
                x += ((h1 >> 11) as f64 / (1_u64 << 53) as f64 - 0.5) * 0.4 * dx;
                y += ((h2 >> 11) as f64 / (1_u64 << 53) as f64 - 0.5) * 0.4 * dy;
            }
            points.push((x, y));
        }
    }

    let triangles = triangulate::triangulate_points_2d(&points);
    let mut mesh = Mesh::new(2);
    for &(x, y) in &points {
        mesh.add_node(Point::new_2d(x, y));
    }
    for triangle in triangles {
        mesh.add_element(ElementType::Triangle, triangle.to_vec());
    }
    mesh.detect_boundaries();
    let tolerance = 1.0e-9;
    mesh.set_boundary_condition(BoundaryType::Neumann, 1, |face| {
        face.iter().all(|point| point.x.abs() < tolerance)
    });
    mesh.set_boundary_condition(BoundaryType::Neumann, 2, |face| {
        face.iter()
            .all(|point| (point.x - length).abs() < tolerance)
    });
    mesh.set_boundary_condition(BoundaryType::Dirichlet, 3, |face| {
        face.iter()
            .all(|point| point.y.abs() < tolerance || (point.y - height).abs() < tolerance)
    });
    mesh
}

fn validate_plane_poiseuille() -> Result<PlanePoiseuilleMetrics, String> {
    let length = 2.0;
    let height = 0.1;
    let ny = 16;
    let mesh = marked_channel_mesh(length, height, 20, ny);
    let velocity_space = VectorH1Space::new(PolynomialDegree::P2, 2);
    let pressure_space = H1Space::new(PolynomialDegree::P1);
    let mut wall_dofs = velocity_space.boundary_dofs(&mesh, 3);
    wall_dofs.sort_unstable();
    wall_dofs.dedup();
    let config = NavierStokesConfig {
        mu: 1.0e-3,
        rho: 1.0,
        use_residual_stabilisation: false,
        // Zero body force: the force quadrature is identical every
        // step, so caching skips two assemblies per step after the first.
        cache_body_force_assembly: true,
        ..Default::default()
    };
    let bdf = BdfConfig::bdf2(0.125, 80);
    let mut state = BdfState::from_rest(
        velocity_space.num_global_dofs(&mesh),
        pressure_space.num_global_dofs(&mesh),
    );
    let mut workspace = BdfWorkspace::new(&mesh, &velocity_space, &pressure_space);
    let outlet_pressure_dof = mesh
        .boundaries
        .iter()
        .find(|face| face.marker == 2)
        .and_then(|face| face.nodes.first().copied())
        .ok_or_else(|| "2D channel has no outlet pressure boundary node".to_string())?;
    // A centroid pressure pin is algebraically valid for a closed Stokes
    // problem, but it is a point constraint inside this traction-driven open
    // channel. Pin the zero-pressure outlet instead so the pressure gauge
    // does not perturb the developed flow profile.
    workspace.set_pressure_gauge(outlet_pressure_dof, 0.0);
    let zero_velocity = |_: f64, _: f64, _: f64| vec![0.0, 0.0];
    let tractions = [
        PressureTraction {
            marker: 1,
            pressure_pa: 1.0,
        },
        PressureTraction {
            marker: 2,
            pressure_pa: 0.0,
        },
    ];
    for step in 0..bdf.num_steps {
        if !solve_bdf_step_cached(
            &mesh,
            &velocity_space,
            &pressure_space,
            &config,
            &bdf,
            &mut state,
            &zero_velocity,
            &wall_dofs,
            &tractions,
            &workspace,
        ) {
            return Err(format!("Poiseuille BDF solve failed at step {step}"));
        }
    }

    let exact = height * height / (8.0 * config.mu * length);
    let coordinates = velocity_space
        .dof_coordinates(&mesh)
        .ok_or_else(|| "P2 velocity space did not expose nodal coordinates".to_string())?;
    let dy = height / ny as f64;
    let numerical = coordinates
        .iter()
        .enumerate()
        .filter(|(dof, point)| {
            dof % 2 == 0
                && (point[1] - 0.5 * height).abs() < 0.3 * dy
                && point[0] > 0.25 * length
                && point[0] < 0.75 * length
        })
        .map(|(dof, _)| state.velocity[dof])
        .fold(0.0_f64, f64::max);
    let relative_error = (numerical - exact).abs() / exact;
    println!(
        "Poiseuille gate: u_max={numerical:.6e} m/s, exact={exact:.6e} m/s, relative error={:.3}%",
        100.0 * relative_error
    );
    // This coarse P2 profile historically resolves the centerline to better
    // than 4%; leave margin for backend roundoff without treating 5% as an
    // accuracy certificate.
    if !relative_error.is_finite() {
        return Err("plane Poiseuille relative error is non-finite".to_string());
    }
    if relative_error >= PLANE_POISEUILLE_MAX_RELATIVE_ERROR {
        return Err(format!(
            "Poiseuille relative error {:.3}% exceeds the measured 4% coarse-mesh bound",
            100.0 * relative_error
        ));
    }
    Ok(PlanePoiseuilleMetrics {
        relative_error,
        numerical_u_max: numerical,
        exact_u_max: exact,
    })
}

fn validate_round_port_poiseuille() -> Result<RoundPortPoiseuilleMetrics, String> {
    let reduced = std::env::var("SPEAKER_PORT_POISEUILLE_SMALL").as_deref() == Ok("1");
    let iterative = std::env::var("SPEAKER_PORT_POISEUILLE_BACKEND").as_deref() != Ok("direct");
    let geometry = RoundPortGeometry {
        duct_length_m: 0.15,
        duct_radius_m: 0.025,
        chamber_length_m: 0.005,
        chamber_radius_m: 0.025,
        flare_radius_m: 0.0,
        mesh_size_m: if reduced { 0.015 } else { 0.009 },
    };
    let mesh = build_round_port_mesh(&geometry).map_err(|error| error.to_string())?;
    if !reduced && mesh.num_nodes() <= 1000 {
        return Err(format!(
            "3D round-port fixture has only {} nodes; expected more than 1000",
            mesh.num_nodes()
        ));
    }
    println!(
        "3D round-port fixture: tier={}, backend={}, nodes={}, tetrahedra={}, boundary_facets={}",
        if reduced { "small" } else { "production" },
        if iterative { "iterative" } else { "direct" },
        mesh.num_nodes(),
        mesh.elements.len(),
        mesh.boundaries.len()
    );
    print_round_port_boundary_diagnostics(&mesh)?;
    let velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
    let pressure_space = H1Space::new(PolynomialDegree::P1);
    let mut wall_dofs = velocity_space.boundary_dofs(&mesh, MARKER_WALL);
    wall_dofs.sort_unstable();
    wall_dofs.dedup();
    let config = NavierStokesConfig {
        mu: 1.0e-3,
        rho: 1.0,
        use_residual_stabilisation: false,
        // Zero body force: the force quadrature is identical every
        // step, so caching skips two assemblies per step after the first.
        cache_body_force_assembly: true,
        use_direct_solver: !iterative,
        linear_solver_backend: LinearSolverBackend::Cpu,
        linear_solver_cpu_preconditioner: LinearSolverCpuPreconditionerKind::JacobiSchur,
        linear_solver_max_iterations: 5000,
        linear_solver_tolerance: 1.0e-10,
        ..Default::default()
    };
    let bdf = BdfConfig::bdf2(0.125, if reduced { 20 } else { 80 });
    let mut state = BdfState::from_rest(
        velocity_space.num_global_dofs(&mesh),
        pressure_space.num_global_dofs(&mesh),
    );
    let mut workspace = BdfWorkspace::new(&mesh, &velocity_space, &pressure_space);
    let pressure_pin = mesh
        .boundaries
        .iter()
        .filter(|face| face.marker == MARKER_EXTERIOR_CAP)
        .flat_map(|face| face.nodes.iter().copied())
        .min_by(|&a, &b| {
            let radial_a = mesh.nodes[a].y.hypot(mesh.nodes[a].z);
            let radial_b = mesh.nodes[b].y.hypot(mesh.nodes[b].z);
            radial_a.total_cmp(&radial_b).then(a.cmp(&b))
        })
        .ok_or_else(|| "3D round-port fixture has no exterior-cap pressure DOF".to_string())?;
    workspace.set_pressure_gauge(pressure_pin, 0.0);
    let pressure_pin_point = mesh.nodes[workspace.pressure_pin];
    println!(
        "3D pressure gauge: dof={}, point=({:.9e},{:.9e},{:.9e}) m",
        workspace.pressure_pin, pressure_pin_point.x, pressure_pin_point.y, pressure_pin_point.z
    );
    let zero_velocity = |_: f64, _: f64, _: f64| vec![0.0, 0.0, 0.0];
    let tractions = [
        PressureTraction {
            marker: MARKER_BOX_CAP,
            pressure_pa: 1.0,
        },
        PressureTraction {
            marker: MARKER_EXTERIOR_CAP,
            pressure_pa: 0.0,
        },
    ];
    for step in 0..bdf.num_steps {
        if !solve_bdf_step_cached(
            &mesh,
            &velocity_space,
            &pressure_space,
            &config,
            &bdf,
            &mut state,
            &zero_velocity,
            &wall_dofs,
            &tractions,
            &workspace,
        ) {
            return Err(format!("3D round-port BDF solve failed at step {step}"));
        }
    }

    let pressure_length = geometry.duct_length_m + 2.0 * geometry.chamber_length_m;
    let exact = geometry.duct_radius_m.powi(2) / (4.0 * config.mu * pressure_length);
    let coordinates = velocity_space
        .dof_coordinates(&mesh)
        .ok_or_else(|| "3D P2 velocity space did not expose nodal coordinates".to_string())?;
    let numerical = centerline_peak_velocity(
        &coordinates,
        &state.velocity,
        3,
        0.4 * geometry.duct_length_m,
        0.6 * geometry.duct_length_m,
        0.25 * geometry.mesh_size_m,
    )?;
    let relative_error = (numerical - exact).abs() / exact;
    print_round_port_solution_diagnostics(
        &mesh,
        &velocity_space,
        &state,
        &geometry,
        pressure_length,
    )?;
    println!(
        "3D round-port Poiseuille gate: nodes={}, u_max={numerical:.6e} m/s, exact={exact:.6e} m/s, relative error={:.3}%",
        mesh.num_nodes(),
        100.0 * relative_error
    );
    if !relative_error.is_finite() {
        return Err("3D round-port Poiseuille relative error is non-finite".to_string());
    }
    if relative_error >= ROUND_PORT_POISEUILLE_MAX_RELATIVE_ERROR {
        return Err(format!(
            "3D round-port Poiseuille relative error {:.3}% exceeds 8%",
            100.0 * relative_error
        ));
    }
    Ok(RoundPortPoiseuilleMetrics {
        relative_error,
        numerical_u_max: numerical,
        exact_u_max: exact,
        nodes: mesh.num_nodes(),
    })
}

fn triangle_area(mesh: &Mesh, nodes: &[usize]) -> Result<f64, String> {
    if nodes.len() != 3 {
        return Err(format!(
            "expected triangular boundary face, got {} nodes",
            nodes.len()
        ));
    }
    let a = mesh.nodes[nodes[0]];
    let b = mesh.nodes[nodes[1]];
    let c = mesh.nodes[nodes[2]];
    let ab = [b.x - a.x, b.y - a.y, b.z - a.z];
    let ac = [c.x - a.x, c.y - a.y, c.z - a.z];
    let cross = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    Ok(0.5 * cross.iter().map(|value| value * value).sum::<f64>().sqrt())
}

fn print_round_port_boundary_diagnostics(mesh: &Mesh) -> Result<(), String> {
    for marker in [MARKER_BOX_CAP, MARKER_EXTERIOR_CAP, MARKER_WALL] {
        let faces: Vec<_> = mesh
            .boundaries
            .iter()
            .filter(|face| face.marker == marker)
            .collect();
        let area = faces
            .iter()
            .map(|face| triangle_area(mesh, &face.nodes))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<f64>();
        println!(
            "3D boundary marker {marker}: facets={}, area={area:.9e} m^2",
            faces.len()
        );
    }
    Ok(())
}

fn print_round_port_solution_diagnostics(
    mesh: &Mesh,
    velocity_space: &VectorH1Space,
    state: &BdfState,
    geometry: &RoundPortGeometry,
    pressure_length: f64,
) -> Result<(), String> {
    let velocity_norm = state
        .velocity
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let last_step_change = state
        .velocity
        .iter()
        .zip(&state.velocity_prev)
        .map(|(current, previous)| (current - previous).powi(2))
        .sum::<f64>()
        .sqrt()
        / velocity_norm.max(f64::MIN_POSITIVE);

    let center_pressure: Vec<_> = mesh
        .nodes
        .iter()
        .zip(&state.pressure)
        .filter(|(point, _)| {
            point.y.hypot(point.z) <= 1.0e-12
                && point.x > 0.2 * geometry.duct_length_m
                && point.x < 0.8 * geometry.duct_length_m
        })
        .map(|(point, pressure)| (point.x, *pressure))
        .collect();
    let mean_x =
        center_pressure.iter().map(|(x, _)| x).sum::<f64>() / center_pressure.len().max(1) as f64;
    let mean_p =
        center_pressure.iter().map(|(_, p)| p).sum::<f64>() / center_pressure.len().max(1) as f64;
    let slope_numerator = center_pressure
        .iter()
        .map(|(x, p)| (x - mean_x) * (p - mean_p))
        .sum::<f64>();
    let slope_denominator = center_pressure
        .iter()
        .map(|(x, _)| (x - mean_x).powi(2))
        .sum::<f64>();
    let pressure_slope = slope_numerator / slope_denominator.max(f64::MIN_POSITIVE);

    let coordinates = velocity_space
        .dof_coordinates(mesh)
        .ok_or_else(|| "3D P2 velocity space did not expose nodal coordinates".to_string())?;
    let mut radial_samples: Vec<(f64, f64, f64)> = coordinates
        .iter()
        .enumerate()
        .filter(|(dof, point)| {
            dof % 3 == 0
                && point[0] > 0.45 * geometry.duct_length_m
                && point[0] < 0.55 * geometry.duct_length_m
        })
        .map(|(dof, point)| (point[0], point[1].hypot(point[2]), state.velocity[dof]))
        .collect();
    radial_samples.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.total_cmp(&b.0)));
    radial_samples.truncate(12);
    println!(
        "3D solution diagnostics: last_step_relative_change={last_step_change:.6e}, center_pressure_samples={}, dp_dx={pressure_slope:.6e} Pa/m, expected_dp_dx={:.6e} Pa/m",
        center_pressure.len(),
        -1.0 / pressure_length
    );
    println!("3D mid-duct samples (x_m,r_m,u_x_m_per_s): {radial_samples:?}");
    Ok(())
}

fn centerline_peak_velocity(
    coordinates: &[[f64; 3]],
    velocity: &[f64],
    components: usize,
    x_min: f64,
    x_max: f64,
    radial_limit: f64,
) -> Result<f64, String> {
    if components == 0 || coordinates.len() != velocity.len() {
        return Err(format!(
            "vector coordinate/state layout mismatch: coordinates={}, velocity={}, components={components}",
            coordinates.len(),
            velocity.len()
        ));
    }
    let peak = coordinates
        .iter()
        .enumerate()
        .filter(|(dof, point)| {
            dof % components == 0
                && point[0] > x_min
                && point[0] < x_max
                && point[1].hypot(point[2]) < radial_limit
        })
        .map(|(dof, _)| velocity[dof])
        .fold(f64::NEG_INFINITY, f64::max);
    if !peak.is_finite() {
        return Err("3D round-port fixture exposed no finite centerline velocity sample".into());
    }
    Ok(peak)
}

fn fixture_medium() -> AcousticMedium {
    AcousticMedium {
        density_kg_per_m3: RHO,
        sound_speed_m_per_s: SOUND_SPEED,
        dynamic_viscosity_pa_s: MU,
        thermal_diffusivity_m2_per_s: 2.2e-5,
    }
}

fn fixture_port() -> PortSpec {
    PortSpec {
        radius_m: (DUCT_HEIGHT_M / PI).sqrt(),
        length_m: DUCT_LENGTH_M,
        end_correction_m: END_CORRECTION_M,
        resistance_pa_s_per_m3: 0.0,
    }
}

fn fixture_enclosure() -> EnclosureSpec {
    EnclosureSpec::Vented {
        volume_m3: BOX_VOLUME_M3,
        port: fixture_port(),
        leakage_resistance_pa_s_per_m3: 0.0,
        absorption_resistance_pa_s_per_m3: 0.0,
    }
}

fn fixture_frequency_hz() -> Result<f64, String> {
    fixture_enclosure()
        .tuning_frequency_hz(fixture_medium())
        .map_err(|error| error.to_string())?
        .map(|frequency| frequency / 2.0)
        .ok_or_else(|| "vented fixture did not produce a tuning frequency".to_string())
}

fn fixture_driver() -> BoxDriverModel {
    BoxDriverModel {
        mms_kg: 0.02,
        rms_kg_per_s: 1.0,
        kms_n_per_m: 4000.0,
        voice_coil: VoiceCoilParams::linear(6.0, 0.5e-3, 10.0),
        compliance_force_fn: None,
        compliance_potential_fn: None,
        sd_m2: DIAPHRAGM_AREA_M2,
        box_volume_m3: Some(BOX_VOLUME_M3),
        medium_density_kg_per_m3: RHO,
        medium_sound_speed_m_per_s: SOUND_SPEED,
        beta: 0.25,
        gamma: 0.5,
    }
}

fn coupled_config(voltage_peak_v: f64, frequency_hz: f64) -> CoupledPortConfig {
    coupled_config_at_samples(voltage_peak_v, frequency_hz, COUPLED_SAMPLES_PER_CYCLE)
}

fn coupled_config_at_samples(
    voltage_peak_v: f64,
    frequency_hz: f64,
    samples_per_cycle: usize,
) -> CoupledPortConfig {
    // Coupled BDF2/Newmark full-tail convergence coverage shows 126 samples
    // per cycle is just outside the unchanged 10% small-signal lumped-limit
    // gate; 252 samples per cycle is the verified production QA step.
    let dt = 1.0 / (frequency_hz * samples_per_cycle as f64);
    let num_steps = COUPLED_TOTAL_CYCLES * samples_per_cycle;
    CoupledPortConfig {
        driver: fixture_driver(),
        ns_config: NavierStokesConfig {
            mu: MU,
            rho: RHO,
            use_residual_stabilisation: false,
            // Zero body force through `solve_bdf_step_cached`: the
            // force quadrature is identical every step, so caching
            // skips two full assemblies per step after the first.
            cache_body_force_assembly: true,
            ..Default::default()
        },
        bdf: BdfConfig::bdf2(dt, num_steps),
        voltage_fn: Box::new(move |time| voltage_peak_v * (TAU * frequency_hz * time).sin()),
        under_relaxation: 1.0,
        exterior_pressure_pa: 0.0,
    }
}

fn lumped_port_flow_amplitude(
    driver: &BoxDriverModel,
    frequency_hz: f64,
    voltage_peak_v: f64,
) -> Result<f64, String> {
    let medium = fixture_medium();
    let enclosure_impedance = fixture_enclosure()
        .acoustic_impedance(frequency_hz, medium)
        .map_err(|error| error.to_string())?;
    let port_impedance = fixture_port()
        .acoustic_impedance(frequency_hz, medium)
        .map_err(|error| error.to_string())?;
    let omega = TAU * frequency_hz;
    let resistance = driver.voice_coil.resistance;
    let inductance = (driver.voice_coil.inductance)(0.0);
    let force_factor = (driver.voice_coil.bl)(0.0);
    let electrical_impedance = Complex64::new(resistance, omega * inductance);
    let mechanical_impedance = Complex64::new(
        driver.rms_kg_per_s,
        omega * driver.mms_kg - driver.kms_n_per_m / omega,
    );
    let total_mechanical_impedance =
        mechanical_impedance + enclosure_impedance * driver.sd_m2.powi(2);
    let diaphragm_velocity = force_factor * voltage_peak_v
        / (electrical_impedance * total_mechanical_impedance + force_factor.powi(2));
    Ok((diaphragm_velocity * driver.sd_m2 * enclosure_impedance / port_impedance).norm())
}

struct LevelMeasurement {
    voltage_peak_v: f64,
    steps_completed: usize,
    port_flow_amplitude_m3s: f64,
    box_pressure_amplitude_pa: f64,
    peak_excursion_m: f64,
    thd: f64,
    energy: CoupledEnergySummary,
    largest_numerical_channel: &'static str,
    largest_numerical_channel_j: f64,
}

fn measure_coupled_levels() -> Result<(f64, Vec<LevelMeasurement>), String> {
    let frequency_hz = fixture_frequency_hz()?;
    let mesh = build_slot_port_mesh(&SlotPortGeometry {
        duct_length_m: DUCT_LENGTH_M,
        duct_height_m: DUCT_HEIGHT_M,
        chamber_length_m: 0.05,
        chamber_height_m: 0.08,
        mesh_size_m: 0.01,
    })
    .map_err(|error| error.to_string())?;
    let velocity_space = VectorH1Space::new(PolynomialDegree::P2, 2);
    let pressure_space = H1Space::new(PolynomialDegree::P1);
    let measurements = DRIVE_LEVELS_V_PEAK
        .into_par_iter()
        .map(|voltage_peak_v| -> Result<LevelMeasurement, String> {
        let config = coupled_config(voltage_peak_v, frequency_hz);
        let dt = config.bdf.dt;
        let expected_steps = config.bdf.num_steps;
        let result = run_coupled_port(&mesh, &velocity_space, &pressure_space, &config)
            .map_err(|error| format!("coupled run at {voltage_peak_v} V peak failed: {error}"))?;
        if result.steps_completed != expected_steps {
            return Err(format!(
                "coupled run at {voltage_peak_v} V completed {} of {expected_steps} steps",
                result.steps_completed
            ));
        }
        let tail_length = COUPLED_TAIL_CYCLES * COUPLED_SAMPLES_PER_CYCLE;
        if tail_length < 2 {
            return Err(format!(
                "coupled run at {voltage_peak_v} V returned too few samples"
            ));
        }
        let tail_start = result.port_flow_m3s.len() - tail_length;
        let flow_tail = &result.port_flow_m3s[tail_start..];
        let pressure_tail = &result.box_pressure_pa[tail_start..];
        let displacement_tail = &result.displacement_m[tail_start..];
        let energy = result.energy_summary(tail_start..result.energy_steps.len());
        if !energy.is_finite() {
            return Err(format!(
                "energy summary at {voltage_peak_v} V peak contains a non-finite or invalid-scale channel: {energy:?}"
            ));
        }
        let dissipation_tolerance_j =
            DISSIPATION_NEGATIVITY_RELATIVE_TOLERANCE * energy.energy_scale_j;
        for (offset, step) in result.energy_steps[tail_start..].iter().enumerate() {
            if !step.is_finite() {
                return Err(format!(
                    "energy step {} at {voltage_peak_v} V peak contains a non-finite channel",
                    tail_start + offset
                ));
            }
            if !step.physical_dissipations_nonnegative(dissipation_tolerance_j) {
                return Err(format!(
                    "energy step {} at {voltage_peak_v} V peak has a negative physical dissipation beyond tolerance {dissipation_tolerance_j:e} J: copper={:e}, mechanical={:e}, molecular_viscous={:e}, eddy_viscous={:e}",
                    tail_start + offset,
                    step.driver.copper_loss_j,
                    step.driver.mechanical_damping_midpoint_j,
                    step.fluid.molecular_viscous_dissipation_j,
                    step.fluid.eddy_viscous_dissipation_j,
                ));
            }
        }
        let mut numerical_channels = [
            ("electrical", 0.0_f64),
            ("electromagnetic_staggering", 0.0),
            ("suspension_newmark", 0.0),
            ("mechanical", 0.0),
            ("mechanical_damping_endpoint_midpoint", 0.0),
            ("box_endpoint", 0.0),
            ("bdf", 0.0),
            ("pressure", 0.0),
            ("fluid_solve", 0.0),
        ];
        for step in &result.energy_steps[tail_start..] {
            let values = [
                step.driver.electrical_numerical_j,
                step.driver.electromagnetic_staggering_j,
                step.driver.suspension_numerical_j,
                step.driver.mechanical_residual_j,
                step.driver.mechanical_damping_j - step.driver.mechanical_damping_midpoint_j,
                step.box_endpoint_numerical_j,
                step.fluid.bdf_numerical_j,
                step.fluid.pressure_work_j,
                step.fluid.solve_residual_j,
            ];
            for ((_, total), value) in numerical_channels.iter_mut().zip(values) {
                *total += value;
            }
        }
        let largest_numerical_channel = numerical_channels
            .into_iter()
            .max_by(|left, right| left.1.abs().total_cmp(&right.1.abs()))
            .unwrap();
        Ok(LevelMeasurement {
            voltage_peak_v,
            steps_completed: result.steps_completed,
            port_flow_amplitude_m3s: dft_amplitude(flow_tail, dt, frequency_hz),
            box_pressure_amplitude_pa: dft_amplitude(pressure_tail, dt, frequency_hz),
            peak_excursion_m: displacement_tail
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f64, f64::max),
            thd: thd_fraction(flow_tail, dt, frequency_hz, 5),
            energy,
            largest_numerical_channel: largest_numerical_channel.0,
            largest_numerical_channel_j: largest_numerical_channel.1,
        })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((frequency_hz, measurements))
}

fn validate_lumped_limit(
    frequency_hz: f64,
    measurements: &[LevelMeasurement],
) -> Result<LumpedLimitMetrics, String> {
    let small_signal = measurements
        .first()
        .ok_or_else(|| "coupled level set is empty".to_string())?;
    let reference =
        lumped_port_flow_amplitude(&fixture_driver(), frequency_hz, small_signal.voltage_peak_v)?;
    let relative_error = (small_signal.port_flow_amplitude_m3s - reference).abs() / reference;
    println!(
        "Lumped-limit gate: coupled={:.6e} m^3/s, reference={reference:.6e} m^3/s, relative error={:.3}%",
        small_signal.port_flow_amplitude_m3s,
        100.0 * relative_error
    );
    if !relative_error.is_finite() {
        return Err("lumped-limit relative error is non-finite".to_string());
    }
    if relative_error >= LUMPED_LIMIT_MAX_RELATIVE_ERROR {
        return Err(format!(
            "lumped-limit relative error {:.3}% exceeds 10%",
            100.0 * relative_error
        ));
    }
    Ok(LumpedLimitMetrics {
        relative_error,
        coupled_flow_m3s: small_signal.port_flow_amplitude_m3s,
        reference_flow_m3s: reference,
    })
}

fn validate_compression_and_energy(
    measurements: &[LevelMeasurement],
) -> Result<CompressionEnergyMetrics, String> {
    let reference = measurements
        .first()
        .ok_or_else(|| "coupled level set is empty".to_string())?;
    let compressions: Vec<f64> = measurements
        .iter()
        .map(|measurement| {
            20.0 * ((measurement.voltage_peak_v / reference.voltage_peak_v)
                / (measurement.port_flow_amplitude_m3s / reference.port_flow_amplitude_m3s))
                .log10()
        })
        .collect();
    if compressions.iter().any(|value| !value.is_finite()) {
        return Err("compression sweep produced a non-finite value".to_string());
    }
    if compressions.windows(2).any(|pair| pair[1] < pair[0]) {
        return Err(format!(
            "compression is not nondecreasing: {compressions:?} dB"
        ));
    }

    println!(
        "Energy closure (J unless marked %):\n  drive | source | delta stored | copper | mechanical | viscous | exterior export | physical % | numerical % | interface %"
    );
    let mut max_physical_closure = 0.0_f64;
    let mut max_algebraic_closure = 0.0_f64;
    let mut max_interface_residual = 0.0_f64;
    for measurement in measurements {
        let energy = measurement.energy;
        println!(
            "  {:>5.3} | {:>10.3e} | {:>12.3e} | {:>10.3e} | {:>10.3e} | {:>10.3e} | {:>15.3e} | {:>10.4} | {:>11.4} | {:>11.4}",
            measurement.voltage_peak_v,
            energy.source_j,
            energy.stored_delta_j,
            energy.copper_loss_j,
            energy.mechanical_damping_j,
            energy.viscous_dissipation_j,
            energy.exterior_export_j,
            100.0 * energy.physical_relative_closure,
            100.0 * energy.numerical_relative_residual,
            100.0 * energy.interface_relative_residual,
        );
        let dissipation_tolerance_j =
            DISSIPATION_NEGATIVITY_RELATIVE_TOLERANCE * energy.energy_scale_j;
        if !energy.is_finite()
            || energy.copper_loss_j < -dissipation_tolerance_j
            || energy.mechanical_damping_j < -dissipation_tolerance_j
            || energy.viscous_dissipation_j < -dissipation_tolerance_j
            || energy.physical_relative_closure > COUPLED_ENERGY_PHYSICAL_CLOSURE_MAX
            || energy.algebraic_relative_closure > COUPLED_ENERGY_ALGEBRAIC_CLOSURE_MAX
            || energy.interface_relative_residual > COUPLED_ENERGY_INTERFACE_RESIDUAL_MAX
        {
            return Err(format!(
                "energy closure failed at {} V peak: physical={:.6e}, algebraic={:.6e}, interface={:.6e}; largest per-step numerical channel {}={:.6e} J",
                measurement.voltage_peak_v,
                energy.physical_relative_closure,
                energy.algebraic_relative_closure,
                energy.interface_relative_residual,
                measurement.largest_numerical_channel,
                measurement.largest_numerical_channel_j,
            ));
        }
        max_physical_closure = max_physical_closure.max(energy.physical_relative_closure);
        max_algebraic_closure = max_algebraic_closure.max(energy.algebraic_relative_closure);
        max_interface_residual = max_interface_residual.max(energy.interface_relative_residual);
    }
    let pressure_bound_pa = RHO * SOUND_SPEED.powi(2) * DIAPHRAGM_AREA_M2 * X_MAX_M / BOX_VOLUME_M3;
    println!(
        "Compression gate: compression={compressions:?} dB\nPressure sanity (diagnostic only): reference bound={pressure_bound_pa:.6e} Pa"
    );
    for measurement in measurements {
        println!(
            "  {:.3} V peak: p_box={:.6e} Pa, peak excursion={:.6e} m",
            measurement.voltage_peak_v,
            measurement.box_pressure_amplitude_pa,
            measurement.peak_excursion_m
        );
    }
    Ok(CompressionEnergyMetrics {
        max_physical_closure,
        max_algebraic_closure,
        max_interface_residual,
        max_compression_db: compressions
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
        levels: measurements.len(),
    })
}

fn validate_linear_thd(measurements: &[LevelMeasurement]) -> Result<f64, String> {
    let thd = measurements
        .first()
        .ok_or_else(|| "coupled level set is empty".to_string())?
        .thd;
    println!("Linear-regime THD gate: {:.4}%", 100.0 * thd);
    if !thd.is_finite() {
        return Err("linear-regime THD is non-finite".to_string());
    }
    if thd >= LINEAR_THD_MAX_FRACTION {
        return Err(format!(
            "linear-regime THD {:.4}% is not below 1%",
            100.0 * thd
        ));
    }
    Ok(thd)
}

fn report_literature_comparison() {
    // Report-only observations digitized from Figure 17 of:
    // B. Doll, "Maximizing Bass Reflex System Performance Through Optimization
    // of Port Geometry," UCF Honors Undergraduate Theses 864 (2020), p. 20,
    // https://stars.library.ucf.edu/honorstheses/864.
    //
    // Shared test article: 30 Hz, 3.70 ft^3 box, 24.70 cm port length,
    // 8.25 cm throat diameter, 17.10 cm mouth diameter, 0.05--2.433 Vrms.
    // The two values below are visual digitizations at 65 dBV referenced to
    // 1 mV, with an estimated +/-0.05 dB reading uncertainty. They are not
    // acceptance gates because the QA fixture does not match this geometry,
    // enclosure, driver, room, or microphone transfer function.
    for (port, geometry, compression_db) in [
        (
            "Port 4",
            "continuous flare, curvature radius 21.34 cm, blend radius 1.25 cm",
            0.58,
        ),
        ("Port 5", "straight benchmark, blend radius 0.25 cm", 0.48),
    ] {
        println!(
            "LITERATURE_REPORT source=\"Doll 2020, UCF thesis 864, Figure 17\" frequency_hz=30 drive_dbv_re_1mv=65 port=\"{port}\" geometry=\"{geometry}\" length_cm=24.70 throat_diameter_cm=8.25 mouth_diameter_cm=17.10 box_ft3=3.70 compression_db={compression_db:.2} digitization_uncertainty_db=0.05 gate=report-only"
        );
    }
}

fn exact_geometry(mesh_size_m: f64) -> RoundPortGeometry {
    RoundPortGeometry {
        duct_length_m: 0.15,
        duct_radius_m: 0.025,
        chamber_length_m: 0.05,
        chamber_radius_m: 0.08,
        flare_radius_m: 0.01,
        mesh_size_m,
    }
}

fn exact_fluid() -> FluidProperties {
    FluidProperties {
        density: RHO,
        speed_of_sound: SOUND_SPEED,
        dynamic_viscosity: MU,
        ..FluidProperties::air_20c()
    }
}

#[derive(Clone, Copy)]
struct Reduced3dTier {
    name: &'static str,
    samples_per_cycle: usize,
    discarded_cycles: usize,
    tail_cycles: usize,
}

fn reduced_3d_tier_from_env() -> Result<Option<Reduced3dTier>, String> {
    let Ok(value) = std::env::var("SPEAKER_PORT_REDUCED_3D") else {
        return Ok(None);
    };
    match value.as_str() {
        "smoke" => Ok(Some(Reduced3dTier {
            name: "smoke",
            samples_per_cycle: 32,
            discarded_cycles: 2,
            tail_cycles: 2,
        })),
        "development" => Ok(Some(Reduced3dTier {
            name: "development",
            samples_per_cycle: 64,
            discarded_cycles: 4,
            tail_cycles: 4,
        })),
        _ => Err(format!(
            "SPEAKER_PORT_REDUCED_3D must be smoke or development, got {value:?}"
        )),
    }
}

fn validate_reduced_3d_energy(
    result: &CoupledPortResult,
    tail_start: usize,
    tier: Reduced3dTier,
) -> Result<CoupledEnergySummary, String> {
    let energy = result.energy_summary(tail_start..result.energy_steps.len());
    if !energy.is_finite() {
        return Err(format!(
            "reduced 3D {} energy summary contains a non-finite or invalid-scale channel: {energy:?}",
            tier.name
        ));
    }
    let dissipation_tolerance_j = 1.0e-12 * energy.energy_scale_j;
    for (offset, step) in result.energy_steps[tail_start..].iter().enumerate() {
        if !step.is_finite() {
            return Err(format!(
                "reduced 3D {} energy step {} contains a non-finite channel",
                tier.name,
                tail_start + offset
            ));
        }
        if !step.physical_dissipations_nonnegative(dissipation_tolerance_j) {
            return Err(format!(
                "reduced 3D {} energy step {} has negative physical dissipation beyond {dissipation_tolerance_j:e} J: copper={:e}, mechanical={:e}, molecular_viscous={:e}, eddy_viscous={:e}",
                tier.name,
                tail_start + offset,
                step.driver.copper_loss_j,
                step.driver.mechanical_damping_midpoint_j,
                step.fluid.molecular_viscous_dissipation_j,
                step.fluid.eddy_viscous_dissipation_j,
            ));
        }
    }
    // Resolution-scaled physical gate: the coarse reduced-3D tiers carry
    // first-order driver/box staggering dissipation that the 252-spc
    // production gate (0.04) does not see. Smoke (32 spc) characteristically
    // records ~0.282 (lag-0/lag-5 agree to 10 digits; algebraic bookkeeping
    // exact to 1e-16), matching 0.04 × 252/32 = 0.315, so the production
    // limit scales linearly with step size from the 0.04 production base.
    // The gate still trips on solver-scale energy blowups; bookkeeping
    // integrity is guarded separately by the exact algebraic/interface
    // closures.
    let max_physical_relative_closure = 0.04 * 252.0 / tier.samples_per_cycle as f64;
    if energy.copper_loss_j < -dissipation_tolerance_j
        || energy.mechanical_damping_j < -dissipation_tolerance_j
        || energy.viscous_dissipation_j < -dissipation_tolerance_j
        || energy.physical_relative_closure > max_physical_relative_closure
        || energy.algebraic_relative_closure > 1.0e-8
        || energy.interface_relative_residual > 1.0e-8
    {
        return Err(format!(
            "reduced 3D {} energy closure failed: physical={:.12e} (limit {max_physical_relative_closure:.3}), algebraic={:.12e}, interface={:.12e}, numerical={:.12e}",
            tier.name,
            energy.physical_relative_closure,
            energy.algebraic_relative_closure,
            energy.interface_relative_residual,
            energy.numerical_relative_residual,
        ));
    }
    Ok(energy)
}

fn run_reduced_3d_nonlinear(tier: Reduced3dTier) -> Result<(), String> {
    let geometry = exact_geometry(0.02);
    let area = PI * geometry.duct_radius_m.powi(2);
    let frequency_hz = SOUND_SPEED / TAU * (area / (BOX_VOLUME_M3 * geometry.duct_length_m)).sqrt();
    let voltage_peak_v = EXACT_VOLTAGE_PEAK_V;
    let mesh = build_round_port_mesh(&geometry).map_err(|error| error.to_string())?;
    let velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
    let pressure_space = H1Space::new(PolynomialDegree::P1);
    let velocity_dofs = velocity_space.num_global_dofs(&mesh);
    let pressure_dofs = pressure_space.num_global_dofs(&mesh);
    let dt = 1.0 / (frequency_hz * tier.samples_per_cycle as f64);
    let num_steps = (tier.discarded_cycles + tier.tail_cycles) * tier.samples_per_cycle;
    // Lagged setup reuse across BDF steps (default 0 = legacy
    // rebuild-on-any-change). Validation harness for long runs.
    let max_lag_steps: usize = std::env::var("SPEAKER_PORT_REDUCED_3D_LAG")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    // Iterative-arm preconditioner A/B for long runs (default jacobi-schur).
    let precond_name = std::env::var("SPEAKER_PORT_REDUCED_3D_PRECOND")
        .unwrap_or_else(|_| "jacobi-schur".to_owned());
    let (cpu_preconditioner, precond_tag) = match precond_name.as_str() {
        "jacobi-schur" | "jacobi_schur" | "schur" => (
            LinearSolverCpuPreconditionerKind::JacobiSchur,
            "jacobi-schur",
        ),
        "ilu-schur" | "ilu_schur" => (LinearSolverCpuPreconditionerKind::IluSchur, "ilu-schur"),
        "ilut-schur" | "ilut_schur" => (LinearSolverCpuPreconditionerKind::IlutSchur, "ilut-schur"),
        other => {
            return Err(format!(
                "SPEAKER_PORT_REDUCED_3D_PRECOND must be jacobi-schur, ilu-schur, or ilut-schur, got {other:?}"
            ));
        }
    };
    println!(
        "REDUCED_3D_START tier={} lag={max_lag_steps} precond={precond_tag} frequency_hz={frequency_hz:.9} voltage_peak_v={voltage_peak_v:.9e} samples_per_cycle={} discarded_cycles={} tail_cycles={} steps={num_steps} nodes={} tetrahedra={} velocity_p2_dofs={velocity_dofs} pressure_p1_dofs={pressure_dofs} total_saddle_dofs={} dt={dt:.9e} stabilisation=false",
        tier.name,
        tier.samples_per_cycle,
        tier.discarded_cycles,
        tier.tail_cycles,
        mesh.num_nodes(),
        mesh.elements.len(),
        velocity_dofs + pressure_dofs,
    );
    let config = CoupledPortConfig {
        driver: fixture_driver(),
        ns_config: NavierStokesConfig {
            mu: MU,
            rho: RHO,
            use_residual_stabilisation: false,
            // Zero body force through `solve_bdf_step_cached`: the
            // force quadrature is identical every step, so caching
            // skips two full assemblies per step after the first.
            cache_body_force_assembly: true,
            use_direct_solver: false,
            linear_solver_backend: LinearSolverBackend::Cpu,
            linear_solver_cpu_preconditioner: cpu_preconditioner,
            linear_solver_max_iterations: 5000,
            linear_solver_tolerance: 1.0e-8,
            linear_solver_setup_max_lag_steps: max_lag_steps,
            ..Default::default()
        },
        bdf: BdfConfig::bdf2(dt, num_steps),
        voltage_fn: Box::new(move |time| voltage_peak_v * (TAU * frequency_hz * time).sin()),
        under_relaxation: 1.0,
        exterior_pressure_pa: 0.0,
    };
    let started = std::time::Instant::now();
    let result = run_coupled_port(&mesh, &velocity_space, &pressure_space, &config)
        .map_err(|error| format!("reduced 3D {} solve failed: {error}", tier.name))?;
    let elapsed_s = started.elapsed().as_secs_f64();
    if result.steps_completed != num_steps {
        return Err(format!(
            "reduced 3D {} completed {} of {num_steps} steps",
            tier.name, result.steps_completed
        ));
    }
    let observables = coupled_complex_observables(
        &result,
        dt,
        frequency_hz,
        tier.samples_per_cycle,
        tier.tail_cycles,
        tier.discarded_cycles,
        BOX_VOLUME_M3,
        RHO,
        SOUND_SPEED,
    )
    .map_err(|error| error.to_string())?;
    let tail_start = result.port_flow_m3s.len() - observables.tail_samples;
    let thd = thd_fraction(&result.port_flow_m3s[tail_start..], dt, frequency_hz, 5);
    if !(thd.is_finite() && thd < 0.005) {
        return Err(format!(
            "reduced 3D {} low-drive THD {:.6}% is not below 0.5%",
            tier.name,
            100.0 * thd
        ));
    }
    if observables.cap_continuity_relative_error >= REDUCED_3D_MAX_CAP_CONTINUITY_RELATIVE_ERROR {
        return Err(format!(
            "reduced 3D {} cap-continuity error {:e} exceeds {:e}",
            tier.name,
            observables.cap_continuity_relative_error,
            REDUCED_3D_MAX_CAP_CONTINUITY_RELATIVE_ERROR
        ));
    }
    let energy = validate_reduced_3d_energy(&result, tail_start, tier)?;
    println!(
        "REDUCED_3D_RESULT tier={} frequency_hz={frequency_hz:.9} p_re={:.12e} p_im={:.12e} q_re={:.12e} q_im={:.12e} qd_re={:.12e} qd_im={:.12e} z_re={:.12e} z_im={:.12e} h_re={:.12e} h_im={:.12e} h_from_z_re={:.12e} h_from_z_im={:.12e} thd={thd:.12e} cap_continuity={:.12e} energy_physical_closure={:.12e} energy_algebraic_closure={:.12e} energy_interface_residual={:.12e} energy_numerical_residual={:.12e} energy_scale_j={:.12e} nodes={} tetrahedra={} velocity_p2_dofs={velocity_dofs} pressure_p1_dofs={pressure_dofs} steps={} dt={dt:.9e} elapsed_s={elapsed_s:.6}",
        tier.name,
        observables.box_pressure_pa.re,
        observables.box_pressure_pa.im,
        observables.port_flow_m3s.re,
        observables.port_flow_m3s.im,
        observables.diaphragm_flow_m3s.re,
        observables.diaphragm_flow_m3s.im,
        observables.port_impedance_pa_s_per_m3.re,
        observables.port_impedance_pa_s_per_m3.im,
        observables.direct_box_to_port_flow_ratio.re,
        observables.direct_box_to_port_flow_ratio.im,
        observables.box_to_port_flow_ratio_from_impedance.re,
        observables.box_to_port_flow_ratio_from_impedance.im,
        observables.cap_continuity_relative_error,
        energy.physical_relative_closure,
        energy.algebraic_relative_closure,
        energy.interface_relative_residual,
        energy.numerical_relative_residual,
        energy.energy_scale_j,
        mesh.num_nodes(),
        mesh.elements.len(),
        result.steps_completed,
    );
    println!("REDUCED_3D_SUCCESS tier={}", tier.name);
    Ok(())
}

fn run_reduced_3d_solver_parity() -> Result<(), String> {
    const STEPS: usize = 2;
    const SAMPLES_PER_CYCLE: usize = 128;
    let geometry = exact_geometry(0.02);
    let area = PI * geometry.duct_radius_m.powi(2);
    let frequency_hz = SOUND_SPEED / TAU * (area / (BOX_VOLUME_M3 * geometry.duct_length_m)).sqrt();
    let dt = 1.0 / (frequency_hz * SAMPLES_PER_CYCLE as f64);
    let mesh = build_round_port_mesh(&geometry).map_err(|error| error.to_string())?;
    let velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
    let pressure_space = H1Space::new(PolynomialDegree::P1);
    let velocity_dofs = velocity_space.num_global_dofs(&mesh);
    let pressure_dofs = pressure_space.num_global_dofs(&mesh);
    // Iterative-arm preconditioner A/B harness: the default keeps the
    // production JacobiSchur contract; ilut-schur selects the stronger
    // ILUT velocity block for timing/iteration comparisons at fixed
    // tolerance. The direct arm is unaffected either way.
    let precond_name = std::env::var("SPEAKER_PORT_SOLVER_PARITY_PRECOND")
        .unwrap_or_else(|_| "jacobi-schur".to_owned());
    let (cpu_preconditioner, mut expected_iterative_solver, precond_tag) = match precond_name
        .as_str()
    {
        "jacobi-schur" | "jacobi_schur" | "schur" => (
            LinearSolverCpuPreconditionerKind::JacobiSchur,
            BdfLinearSolverKind::FgmresJacobiSchur,
            "jacobi-schur",
        ),
        "ilut-schur" | "ilut_schur" => (
            LinearSolverCpuPreconditionerKind::IlutSchur,
            BdfLinearSolverKind::FgmresIlutSchur,
            "ilut-schur",
        ),
        "ilu-schur" | "ilu_schur" => (
            LinearSolverCpuPreconditionerKind::IluSchur,
            BdfLinearSolverKind::FgmresIluSchur,
            "ilu-schur",
        ),
        other => {
            return Err(format!(
                "SPEAKER_PORT_SOLVER_PARITY_PRECOND must be jacobi-schur, ilut-schur, or ilu-schur, got {other:?}"
            ));
        }
    };
    // Iterative-arm device A/B harness: the default keeps the CPU
    // contract; cuda routes the BDF linear solves through the GPU
    // backend with the Jacobi/Schur preconditioner fed by the memoized
    // unsteady BtDiagB product. Requires a `cuda`-feature build plus a
    // CUDA driver; a silent CPU fallback fails loudly below so a GPU
    // benchmark can never misreport CPU numbers.
    let device_name =
        std::env::var("SPEAKER_PORT_SOLVER_PARITY_DEVICE").unwrap_or_else(|_| "cpu".to_owned());
    let (linear_solver_backend, device_tag) = match device_name.as_str() {
        "cpu" => (LinearSolverBackend::Cpu, "cpu"),
        "cuda" => (LinearSolverBackend::Cuda, "cuda"),
        other => {
            return Err(format!(
                "SPEAKER_PORT_SOLVER_PARITY_DEVICE must be cpu or cuda, got {other:?}"
            ));
        }
    };
    let expected_backend = linear_solver_backend;
    if matches!(expected_backend, LinearSolverBackend::Cuda) {
        expected_iterative_solver = BdfLinearSolverKind::BackendFgmres;
    }
    // GPU preconditioner tuning knobs (ignored on the CPU device).
    // Defaults preserve the legacy diagonal/diagonal f64 behavior.
    let schur_degree: usize = std::env::var("SPEAKER_PORT_SOLVER_PARITY_SCHUR_DEGREE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1);
    let velocity_degree: usize = std::env::var("SPEAKER_PORT_SOLVER_PARITY_VELOCITY_DEGREE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1);
    let schur_kind_name = std::env::var("SPEAKER_PORT_SOLVER_PARITY_SCHUR_KIND")
        .unwrap_or_else(|_| "polynomial".to_owned());
    let schur_kind = match schur_kind_name.as_str() {
        "polynomial" => BackendPressureSchurPreconditionerKind::Polynomial,
        "chebyshev" => BackendPressureSchurPreconditionerKind::Chebyshev,
        "amg" => BackendPressureSchurPreconditionerKind::Amg,
        other => {
            return Err(format!(
                "SPEAKER_PORT_SOLVER_PARITY_SCHUR_KIND must be polynomial, chebyshev, or amg, got {other:?}"
            ));
        }
    };
    let mixed_precision = matches!(
        std::env::var("SPEAKER_PORT_SOLVER_PARITY_MIXED")
            .unwrap_or_default()
            .as_str(),
        "1" | "true" | "yes"
    );
    // Lagged setup reuse across BDF steps (CPU Schur arms only).
    // Default 0 preserves the legacy rebuild-on-any-change contract
    // and its exact base/probe reuse pattern.
    let max_lag_steps: usize = std::env::var("SPEAKER_PORT_SOLVER_PARITY_LAG")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    println!(
        "SOLVER_PARITY_START frequency_hz={frequency_hz:.12} dt={dt:.12e} steps={STEPS} precond={precond_tag} device={device_tag} schur={schur_kind_name}/{schur_degree} velocity_degree={velocity_degree} mixed={mixed_precision} lag={max_lag_steps} nodes={} tetrahedra={} velocity_p2_dofs={velocity_dofs} pressure_p1_dofs={pressure_dofs} total_saddle_dofs={}",
        mesh.num_nodes(),
        mesh.elements.len(),
        velocity_dofs + pressure_dofs,
    );

    let run = |use_direct_solver: bool| -> Result<(CoupledPortResult, f64), String> {
        let config = CoupledPortConfig {
            driver: fixture_driver(),
            ns_config: NavierStokesConfig {
                mu: MU,
                rho: RHO,
                use_residual_stabilisation: false,
                // Zero body force through `solve_bdf_step_cached`: the
                // force quadrature is identical every step, so caching
                // skips two full assemblies per step after the first.
                cache_body_force_assembly: true,
                use_direct_solver,
                linear_solver_backend,
                linear_solver_backend_preconditioner: BackendPreconditionerKind::JacobiSchur,
                linear_solver_polynomial_degree: velocity_degree,
                linear_solver_schur_polynomial_degree: schur_degree,
                linear_solver_schur_preconditioner: schur_kind,
                linear_solver_kind: if mixed_precision {
                    LinearSolverKind::MixedF32F64
                } else {
                    LinearSolverKind::F64
                },
                linear_solver_cpu_preconditioner: cpu_preconditioner,
                linear_solver_max_iterations: 5000,
                linear_solver_tolerance: 1.0e-10,
                linear_solver_setup_max_lag_steps: max_lag_steps,
                ..Default::default()
            },
            bdf: BdfConfig::bdf2(dt, STEPS),
            voltage_fn: Box::new(move |time| {
                EXACT_VOLTAGE_PEAK_V * (TAU * frequency_hz * time).cos()
            }),
            under_relaxation: 1.0,
            exterior_pressure_pa: 0.0,
        };
        let started = std::time::Instant::now();
        let result = run_coupled_port(&mesh, &velocity_space, &pressure_space, &config).map_err(
            |error| {
                format!(
                    "{} parity solve failed: {error}",
                    if use_direct_solver {
                        "direct"
                    } else {
                        "iterative"
                    }
                )
            },
        )?;
        Ok((result, started.elapsed().as_secs_f64()))
    };

    let backend =
        std::env::var("SPEAKER_PORT_SOLVER_PARITY_BACKEND").unwrap_or_else(|_| "both".to_owned());
    if backend == "iterative" {
        let (result, elapsed_s) = run(false)?;
        if result.steps_completed != STEPS
            || result.linear_solves.len() != 2 * STEPS
            || result.accepted_linear_relative_residual.len() != STEPS
        {
            return Err("iterative h20 probe did not complete its telemetry contract".into());
        }
        for (index, solve) in result.linear_solves.iter().enumerate() {
            // The backend path has no setup cache, and lagged reuse
            // deliberately reuses across changed matrices, so the exact
            // base/probe reuse pattern only applies to unlagged CPU.
            let reuse_ok = solve.solver == BdfLinearSolverKind::BackendFgmres
                || max_lag_steps > 0
                || solve.setup_reused == (index % 2 == 1);
            if solve.solver != expected_iterative_solver
                || solve.resolved_linear_solver_backend != expected_backend
                || !solve.converged
                || solve.true_relative_residual > 1.0e-9
                || !reuse_ok
            {
                return Err(format!("iterative h20 solve {index} failed: {solve:?}"));
            }
        }
        if let Some((index, residual)) = result
            .accepted_linear_relative_residual
            .iter()
            .copied()
            .enumerate()
            .find(|(_, residual)| !residual.is_finite() || *residual > 5.0e-9)
        {
            return Err(format!(
                "iterative h20 accepted state {index} residual {residual:e} exceeds 5e-9"
            ));
        }
        let iterations = result
            .linear_solves
            .iter()
            .map(|solve| solve.iterations)
            .sum::<usize>();
        println!(
            "SOLVER_ITERATIVE_PROBE_RESULT precond={precond_tag} device={device_tag} q={:.12e} p_box={:.12e} cap_continuity={:.12e} conservation={:.12e} max_probe_residual={:.12e} max_accepted_residual={:.12e} iterations={iterations} elapsed_s={elapsed_s:.6}",
            result.port_flow_m3s.last().unwrap(),
            result.box_pressure_pa.last().unwrap(),
            (result.exterior_cap_flow_m3s.last().unwrap()
                + result.box_cap_outward_flow_m3s.last().unwrap())
            .abs()
                / result
                    .exterior_cap_flow_m3s
                    .last()
                    .unwrap()
                    .abs()
                    .max(result.box_cap_outward_flow_m3s.last().unwrap().abs(),)
                    .max(f64::MIN_POSITIVE),
            result.box_conservation_residual_pa.last().unwrap(),
            result
                .linear_solves
                .iter()
                .map(|solve| solve.true_relative_residual)
                .fold(0.0_f64, f64::max),
            result
                .accepted_linear_relative_residual
                .iter()
                .copied()
                .fold(0.0_f64, f64::max),
        );
        println!("SOLVER_ITERATIVE_PROBE_SUCCESS");
        return Ok(());
    }
    if backend != "both" {
        return Err(format!(
            "SPEAKER_PORT_SOLVER_PARITY_BACKEND must be both or iterative, got {backend:?}"
        ));
    }

    let (direct, direct_elapsed_s) = run(true)?;
    let (iterative, iterative_elapsed_s) = run(false)?;
    if direct.steps_completed != STEPS || iterative.steps_completed != STEPS {
        return Err(format!(
            "parity run did not complete two steps: direct={}, iterative={}",
            direct.steps_completed, iterative.steps_completed
        ));
    }
    let direct_flow = direct
        .final_flow_state
        .as_ref()
        .ok_or_else(|| "direct parity run omitted final fluid state".to_owned())?;
    let iterative_flow = iterative
        .final_flow_state
        .as_ref()
        .ok_or_else(|| "iterative parity run omitted final fluid state".to_owned())?;

    let l2_relative = |actual: &[f64], reference: &[f64]| -> Result<f64, String> {
        if actual.len() != reference.len() {
            return Err(format!(
                "parity vector length mismatch: {} versus {}",
                actual.len(),
                reference.len()
            ));
        }
        let error = actual
            .iter()
            .zip(reference)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale = reference
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt()
            .max(f64::MIN_POSITIVE);
        Ok(error / scale)
    };
    let velocity_relative = l2_relative(&iterative_flow.velocity, &direct_flow.velocity)?;
    let gauge_align = |values: &[f64]| {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        values.iter().map(|value| value - mean).collect::<Vec<_>>()
    };
    let pressure_relative = l2_relative(
        &gauge_align(&iterative_flow.pressure),
        &gauge_align(&direct_flow.pressure),
    )?;
    let scalar_close = |name: &str,
                        actual: f64,
                        reference: f64,
                        relative: f64,
                        absolute: f64|
     -> Result<f64, String> {
        let error = (actual - reference).abs();
        let tolerance = absolute + relative * reference.abs();
        if !(error.is_finite() && error <= tolerance) {
            return Err(format!(
                "{name} parity failed: iterative={actual:e}, direct={reference:e}, error={error:e}, tolerance={tolerance:e}"
            ));
        }
        Ok(error)
    };
    if velocity_relative > 1.0e-7 || pressure_relative > 1.0e-6 {
        return Err(format!(
            "field parity failed: velocity={velocity_relative:e} (limit 1e-7), gauge_pressure={pressure_relative:e} (limit 1e-6)"
        ));
    }
    let q_direct = *direct.port_flow_m3s.last().unwrap();
    let q_iterative = *iterative.port_flow_m3s.last().unwrap();
    let q_error = scalar_close("cap Q", q_iterative, q_direct, 1.0e-6, 1.0e-14)?;
    let p_direct = *direct.box_pressure_pa.last().unwrap();
    let p_iterative = *iterative.box_pressure_pa.last().unwrap();
    let p_error = scalar_close("box pressure", p_iterative, p_direct, 1.0e-7, 1.0e-10)?;
    let conservation_error = scalar_close(
        "box conservation residual",
        *iterative.box_conservation_residual_pa.last().unwrap(),
        *direct.box_conservation_residual_pa.last().unwrap(),
        1.0e-6,
        1.0e-12,
    )?;
    let cap_continuity = |result: &CoupledPortResult| {
        let q_ext = *result.exterior_cap_flow_m3s.last().unwrap();
        let q_box = *result.box_cap_outward_flow_m3s.last().unwrap();
        (q_ext + q_box).abs() / q_ext.abs().max(q_box.abs()).max(f64::MIN_POSITIVE)
    };
    let direct_continuity = cap_continuity(&direct);
    let iterative_continuity = cap_continuity(&iterative);
    if direct_continuity > 1.0e-6 || iterative_continuity > 1.0e-6 {
        return Err(format!(
            "cap continuity failed: direct={direct_continuity:e}, iterative={iterative_continuity:e}"
        ));
    }

    let direct_energy = direct.energy_summary(0..STEPS);
    let iterative_energy = iterative.energy_summary(0..STEPS);
    let energy_scale = direct_energy
        .energy_scale_j
        .max(iterative_energy.energy_scale_j)
        .max(f64::MIN_POSITIVE);
    let energy_error = [
        (iterative_energy.source_j - direct_energy.source_j).abs(),
        (iterative_energy.stored_delta_j - direct_energy.stored_delta_j).abs(),
        (iterative_energy.physical_dissipation_j - direct_energy.physical_dissipation_j).abs(),
        (iterative_energy.exterior_export_j - direct_energy.exterior_export_j).abs(),
        (iterative_energy.numerical_residual_j - direct_energy.numerical_residual_j).abs(),
    ]
    .into_iter()
    .fold(0.0_f64, f64::max)
        / energy_scale;
    if energy_error > 1.0e-6 {
        return Err(format!(
            "energy parity error {energy_error:e} exceeds 1e-6 of scale"
        ));
    }

    for (name, result, expected_solver) in [
        ("direct", &direct, BdfLinearSolverKind::Direct),
        ("iterative", &iterative, expected_iterative_solver),
    ] {
        if result.linear_solves.len() != 2 * STEPS
            || result.accepted_linear_relative_residual.len() != STEPS
        {
            return Err(format!("{name} parity telemetry has wrong length"));
        }
        for (index, solve) in result.linear_solves.iter().enumerate() {
            // The backend path has no setup cache, and lagged reuse
            // deliberately reuses across changed matrices, so the exact
            // base/probe reuse pattern only applies to unlagged CPU.
            let reuse_ok = solve.solver == BdfLinearSolverKind::BackendFgmres
                || (name == "iterative" && max_lag_steps > 0)
                || solve.setup_reused == (index % 2 == 1);
            let backend_ok =
                name != "iterative" || solve.resolved_linear_solver_backend == expected_backend;
            if solve.solver != expected_solver
                || !solve.converged
                || solve.true_relative_residual > 1.0e-9
                || !reuse_ok
                || !backend_ok
            {
                return Err(format!(
                    "{name} linear solve {index} failed contract: {solve:?}"
                ));
            }
        }
        if let Some((index, residual)) = result
            .accepted_linear_relative_residual
            .iter()
            .copied()
            .enumerate()
            .find(|(_, residual)| !residual.is_finite() || *residual > 5.0e-9)
        {
            return Err(format!(
                "{name} accepted affine state {index} true residual {residual:e} exceeds 5e-9"
            ));
        }
    }
    let iterative_iterations = iterative
        .linear_solves
        .iter()
        .map(|solve| solve.iterations)
        .sum::<usize>();
    println!(
        "SOLVER_PARITY_RESULT precond={precond_tag} device={device_tag} velocity_l2_relative={velocity_relative:.12e} gauge_pressure_l2_relative={pressure_relative:.12e} q_error={q_error:.12e} box_pressure_error={p_error:.12e} conservation_error={conservation_error:.12e} direct_cap_continuity={direct_continuity:.12e} iterative_cap_continuity={iterative_continuity:.12e} energy_relative_error={energy_error:.12e} direct_max_true_residual={:.12e} iterative_max_true_residual={:.12e} direct_max_accepted_residual={:.12e} iterative_max_accepted_residual={:.12e} iterative_total_iterations={iterative_iterations} direct_elapsed_s={direct_elapsed_s:.6} iterative_elapsed_s={iterative_elapsed_s:.6}",
        direct
            .linear_solves
            .iter()
            .map(|solve| solve.true_relative_residual)
            .fold(0.0_f64, f64::max),
        iterative
            .linear_solves
            .iter()
            .map(|solve| solve.true_relative_residual)
            .fold(0.0_f64, f64::max),
        direct
            .accepted_linear_relative_residual
            .iter()
            .copied()
            .fold(0.0_f64, f64::max),
        iterative
            .accepted_linear_relative_residual
            .iter()
            .copied()
            .fold(0.0_f64, f64::max),
    );
    println!("SOLVER_PARITY_SUCCESS");
    Ok(())
}

fn exact_relative_error(actual: Complex64, reference: Complex64) -> Result<f64, String> {
    if reference.norm() <= f64::EPSILON {
        return Err("exact reference quantity is zero".into());
    }
    let error = (actual - reference).norm() / reference.norm();
    if !error.is_finite() {
        return Err("complex relative error is non-finite".into());
    }
    Ok(error)
}

struct ExactNonlinearPoint {
    observables: CoupledComplexObservables,
    energy: CoupledEnergySummary,
    thd: f64,
    nodes: usize,
    tetrahedra: usize,
    velocity_dofs: usize,
    dt: f64,
    elapsed_s: f64,
    tail_z_delta: f64,
    tail_h_delta: f64,
    reynolds: f64,
    mach: f64,
}

fn run_exact_nonlinear_point(
    mesh: &Mesh,
    geometry: &RoundPortGeometry,
    frequency_hz: f64,
    voltage_peak_v: f64,
    samples_per_cycle: usize,
) -> Result<ExactNonlinearPoint, String> {
    let velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
    let pressure_space = H1Space::new(PolynomialDegree::P1);
    let dt = 1.0 / (frequency_hz * samples_per_cycle as f64);
    let num_steps = (EXACT_DISCARDED_CYCLES + EXACT_TAIL_CYCLES) * samples_per_cycle;
    let velocity_dofs = velocity_space.num_global_dofs(mesh);
    let pressure_dofs = pressure_space.num_global_dofs(mesh);
    println!(
        "EXACT_NONLINEAR_START frequency_hz={frequency_hz:.9} voltage_peak_v={voltage_peak_v:.9e} samples_per_cycle={samples_per_cycle} discarded_cycles={} tail_cycles={} steps={num_steps} nodes={} tetrahedra={} velocity_p2_dofs={velocity_dofs} pressure_p1_dofs={pressure_dofs} total_saddle_dofs={} dt={dt:.9e}",
        EXACT_DISCARDED_CYCLES,
        EXACT_TAIL_CYCLES,
        mesh.num_nodes(),
        mesh.elements.len(),
        velocity_dofs + pressure_dofs,
    );
    let config = CoupledPortConfig {
        driver: fixture_driver(),
        ns_config: NavierStokesConfig {
            mu: MU,
            rho: RHO,
            use_residual_stabilisation: false,
            // Zero body force through `solve_bdf_step_cached`: the
            // force quadrature is identical every step, so caching
            // skips two full assemblies per step after the first.
            cache_body_force_assembly: true,
            use_direct_solver: false,
            linear_solver_backend: LinearSolverBackend::Cpu,
            linear_solver_cpu_preconditioner: LinearSolverCpuPreconditionerKind::JacobiSchur,
            linear_solver_max_iterations: 5000,
            linear_solver_tolerance: 1.0e-8,
            ..Default::default()
        },
        bdf: BdfConfig::bdf2(dt, num_steps),
        voltage_fn: Box::new(move |time| voltage_peak_v * (TAU * frequency_hz * time).sin()),
        under_relaxation: 1.0,
        exterior_pressure_pa: 0.0,
    };
    let started = std::time::Instant::now();
    let result = run_coupled_port(mesh, &velocity_space, &pressure_space, &config)
        .map_err(|error| error.to_string())?;
    let elapsed_s = started.elapsed().as_secs_f64();
    let observables = coupled_complex_observables(
        &result,
        dt,
        frequency_hz,
        samples_per_cycle,
        EXACT_TAIL_CYCLES,
        EXACT_DISCARDED_CYCLES,
        BOX_VOLUME_M3,
        RHO,
        SOUND_SPEED,
    )
    .map_err(|error| error.to_string())?;
    let observables_8 = coupled_complex_observables(
        &result,
        dt,
        frequency_hz,
        samples_per_cycle,
        8,
        EXACT_DISCARDED_CYCLES,
        BOX_VOLUME_M3,
        RHO,
        SOUND_SPEED,
    )
    .map_err(|error| error.to_string())?;
    let tail_start = result
        .energy_steps
        .len()
        .checked_sub(observables.tail_samples)
        .ok_or_else(|| "coupled energy ledger is shorter than the settled tail".to_string())?;
    let energy = result.energy_summary(tail_start..result.energy_steps.len());
    if !energy.is_finite() {
        return Err("settled coupled energy summary is non-finite".into());
    }
    let tail_z_delta = exact_relative_error(
        observables_8.port_impedance_pa_s_per_m3,
        observables.port_impedance_pa_s_per_m3,
    )?;
    let tail_h_delta = exact_relative_error(
        observables_8.direct_box_to_port_flow_ratio,
        observables.direct_box_to_port_flow_ratio,
    )?;
    if tail_z_delta >= EXACT_TAIL_DELTA_MAX || tail_h_delta >= EXACT_TAIL_DELTA_MAX {
        return Err(format!(
            "8-to-16-cycle tail convergence failed: Z={tail_z_delta:e}, H={tail_h_delta:e}"
        ));
    }
    let start = result.port_flow_m3s.len() - observables.tail_samples;
    let thd = thd_fraction(&result.port_flow_m3s[start..], dt, frequency_hz, 5);
    if !(thd.is_finite() && thd < EXACT_THD_MAX_FRACTION) {
        return Err(format!(
            "exact low-drive port-flow THD {:.4}% is not below 0.5%",
            100.0 * thd
        ));
    }
    if observables.cap_continuity_relative_error >= EXACT_CAP_CONTINUITY_MAX {
        return Err(format!(
            "exact cap-continuity error {:e} exceeds 1e-6",
            observables.cap_continuity_relative_error
        ));
    }
    let characteristic_velocity =
        observables.port_flow_m3s.norm() / (PI * geometry.duct_radius_m.powi(2));
    let reynolds = RHO * characteristic_velocity * 2.0 * geometry.duct_radius_m / MU;
    let mach = characteristic_velocity / SOUND_SPEED;
    println!(
        "EXACT_NONLINEAR_POINT frequency_hz={frequency_hz:.9} voltage_peak_v={voltage_peak_v:.9e} samples_per_cycle={samples_per_cycle} z_re={:.12e} z_im={:.12e} h_re={:.12e} h_im={:.12e} thd={thd:.12e} cap_continuity={:.12e} tail_z_delta={tail_z_delta:.12e} tail_h_delta={tail_h_delta:.12e} reynolds={reynolds:.12e} mach={mach:.12e} nodes={} tetrahedra={} velocity_p2_dofs={velocity_dofs} pressure_p1_dofs={pressure_dofs} elapsed_s={elapsed_s:.6}",
        observables.port_impedance_pa_s_per_m3.re,
        observables.port_impedance_pa_s_per_m3.im,
        observables.direct_box_to_port_flow_ratio.re,
        observables.direct_box_to_port_flow_ratio.im,
        observables.cap_continuity_relative_error,
        mesh.num_nodes(),
        mesh.elements.len(),
    );
    Ok(ExactNonlinearPoint {
        observables,
        energy,
        thd,
        nodes: mesh.num_nodes(),
        tetrahedra: mesh.elements.len(),
        velocity_dofs,
        dt,
        elapsed_s,
        tail_z_delta,
        tail_h_delta,
        reynolds,
        mach,
    })
}

/// Solve the coarse/fine linear references and gate their agreement. This is
/// the shared prefix of the full 3D leg and the bounded exact-reference
/// diagnostic tiers.
fn run_exact_reference_convergence() -> Result<ExactReferenceConvergence, String> {
    // Authoritative P2 lineage: canonical Task-13 h=20 mm mesh plus a
    // physically identical h=10 mm geometry refinement also refines the
    // angular boundary segmentation, not only the volume topology.
    let coarse_geometry = exact_geometry(0.02);
    let fine_geometry = exact_geometry(0.01);
    let area = PI * coarse_geometry.duct_radius_m.powi(2);
    let f0 = SOUND_SPEED / TAU * (area / (BOX_VOLUME_M3 * coarse_geometry.duct_length_m)).sqrt();
    let frequencies = [0.75 * f0, f0, 1.25 * f0];
    let fluid = exact_fluid();
    println!(
        "EXACT_CONFIG rho={} c={} mu={} thermal_conductivity={} cp={} gamma={} ambient_pressure={} box_volume_m3={} f0_hz={} voltage_peak_v={} tail_cycles={} discarded_cycles={} samples_per_cycle={}",
        fluid.density,
        fluid.speed_of_sound,
        fluid.dynamic_viscosity,
        fluid.thermal_conductivity,
        fluid.specific_heat_cp,
        fluid.ratio_of_specific_heats,
        fluid.ambient_pressure,
        BOX_VOLUME_M3,
        f0,
        EXACT_VOLTAGE_PEAK_V,
        EXACT_TAIL_CYCLES,
        EXACT_DISCARDED_CYCLES,
        EXACT_SAMPLES_PER_CYCLE
    );

    let coarse_mesh = build_round_port_mesh(&coarse_geometry).map_err(|e| e.to_string())?;
    let fine_mesh = build_round_port_mesh(&fine_geometry).map_err(|e| e.to_string())?;
    println!(
        "EXACT_GEOMETRY_REFINEMENT coarse_nodes={} coarse_tetrahedra={} fine_nodes={} fine_tetrahedra={} coarse_boundary_facets={} fine_boundary_facets={}",
        coarse_mesh.num_nodes(),
        coarse_mesh.elements.len(),
        fine_mesh.num_nodes(),
        fine_mesh.elements.len(),
        coarse_mesh.boundaries.len(),
        fine_mesh.boundaries.len(),
    );
    let reference_config = RoundPortLinearReferenceConfig {
        geometry: fine_geometry,
        fluid: fluid.clone(),
        box_volume_m3: BOX_VOLUME_M3,
    };
    let fine_reference =
        solve_round_port_linear_reference(&fine_mesh, &reference_config, frequencies[1])
            .map_err(|error| error.to_string())?;
    let coarse_reference = solve_round_port_linear_reference(
        &coarse_mesh,
        &RoundPortLinearReferenceConfig {
            geometry: coarse_geometry,
            fluid: fluid.clone(),
            box_volume_m3: BOX_VOLUME_M3,
        },
        frequencies[1],
    )
    .map_err(|error| error.to_string())?;
    let reference_z_delta = exact_relative_error(
        coarse_reference.port_impedance_pa_s_per_m3,
        fine_reference.port_impedance_pa_s_per_m3,
    )?;
    let reference_h_delta = exact_relative_error(
        coarse_reference.box_to_port_flow_ratio,
        fine_reference.box_to_port_flow_ratio,
    )?;
    println!(
        "EXACT_REFERENCE_RAW frequency_hz={:.9} polynomial_degree={} quadrature_order={} coarse_scalar_dofs={} fine_scalar_dofs={} coarse_z=({:.12e},{:.12e}) fine_z=({:.12e},{:.12e}) coarse_h=({:.12e},{:.12e}) fine_h=({:.12e},{:.12e}) coarse_strong_weak_delta={:.12e} fine_strong_weak_delta={:.12e} coarse_h_condition={:.12e} fine_h_condition={:.12e}",
        frequencies[1],
        fine_reference.polynomial_degree,
        fine_reference.quadrature_order,
        coarse_reference.scalar_dofs,
        fine_reference.scalar_dofs,
        coarse_reference.port_impedance_pa_s_per_m3.re,
        coarse_reference.port_impedance_pa_s_per_m3.im,
        fine_reference.port_impedance_pa_s_per_m3.re,
        fine_reference.port_impedance_pa_s_per_m3.im,
        coarse_reference.box_to_port_flow_ratio.re,
        coarse_reference.box_to_port_flow_ratio.im,
        fine_reference.box_to_port_flow_ratio.re,
        fine_reference.box_to_port_flow_ratio.im,
        coarse_reference.strong_weak_exterior_flux_relative_difference,
        fine_reference.strong_weak_exterior_flux_relative_difference,
        coarse_reference.box_to_port_flow_ratio_condition_number,
        fine_reference.box_to_port_flow_ratio_condition_number,
    );
    if !reference_z_delta.is_finite() || !reference_h_delta.is_finite() {
        return Err("reference mesh convergence deltas are non-finite".to_string());
    }
    if reference_z_delta >= EXACT_REFERENCE_CONVERGENCE_MAX
        || reference_h_delta >= EXACT_REFERENCE_CONVERGENCE_MAX
    {
        return Err(format!(
            "reference mesh convergence failed: Z={reference_z_delta:e}, H={reference_h_delta:e}"
        ));
    }
    println!(
        "EXACT_CONVERGENCE_REFERENCE frequency_hz={:.9} polynomial_degree={} quadrature_order={} coarse_nodes={} fine_nodes={} coarse_scalar_dofs={} fine_scalar_dofs={} z_delta={reference_z_delta:.9e} h_delta={reference_h_delta:.9e}",
        frequencies[1],
        fine_reference.polynomial_degree,
        fine_reference.quadrature_order,
        coarse_mesh.num_nodes(),
        fine_mesh.num_nodes(),
        coarse_reference.scalar_dofs,
        fine_reference.scalar_dofs,
    );

    Ok(ExactReferenceConvergence {
        coarse_mesh,
        fine_mesh,
        coarse_geometry,
        fine_geometry,
        reference_config,
        fine_reference,
        frequencies,
        reference_z_delta,
        reference_h_delta,
    })
}

/// Run the complete 3D exact-reference ladder: nonlinear mesh/timestep/
/// linearity convergence plus a linear-reference gate at every sweep
/// frequency. There are no partial returns; a 3D claim requires every gate.
fn validate_against_exact_resolved_fem() -> Result<ExactReferenceMetrics, String> {
    let ExactReferenceConvergence {
        coarse_mesh,
        fine_mesh,
        coarse_geometry,
        fine_geometry,
        reference_config,
        fine_reference,
        frequencies,
        reference_z_delta,
        reference_h_delta,
    } = run_exact_reference_convergence()?;

    let centre = run_exact_nonlinear_point(
        &fine_mesh,
        &fine_geometry,
        frequencies[1],
        EXACT_VOLTAGE_PEAK_V,
        EXACT_SAMPLES_PER_CYCLE,
    )?;
    let centre_coarse_mesh = run_exact_nonlinear_point(
        &coarse_mesh,
        &coarse_geometry,
        frequencies[1],
        EXACT_VOLTAGE_PEAK_V,
        EXACT_SAMPLES_PER_CYCLE,
    )?;
    let centre_coarse_dt = run_exact_nonlinear_point(
        &fine_mesh,
        &fine_geometry,
        frequencies[1],
        EXACT_VOLTAGE_PEAK_V,
        EXACT_SAMPLES_PER_CYCLE / 2,
    )?;
    let half_drive = run_exact_nonlinear_point(
        &fine_mesh,
        &fine_geometry,
        frequencies[1],
        EXACT_VOLTAGE_PEAK_V / 2.0,
        EXACT_SAMPLES_PER_CYCLE,
    )?;
    let nonlinear_mesh_z_delta = exact_relative_error(
        centre_coarse_mesh.observables.port_impedance_pa_s_per_m3,
        centre.observables.port_impedance_pa_s_per_m3,
    )?;
    if nonlinear_mesh_z_delta >= EXACT_NONLINEAR_MESH_DT_MAX {
        return Err(format!(
            "nonlinear_mesh_Z convergence/linearity error {nonlinear_mesh_z_delta:e} exceeds {EXACT_NONLINEAR_MESH_DT_MAX:e}"
        ));
    }
    let dt_z_delta = exact_relative_error(
        centre_coarse_dt.observables.port_impedance_pa_s_per_m3,
        centre.observables.port_impedance_pa_s_per_m3,
    )?;
    if dt_z_delta >= EXACT_NONLINEAR_MESH_DT_MAX {
        return Err(format!(
            "dt_Z convergence/linearity error {dt_z_delta:e} exceeds {EXACT_NONLINEAR_MESH_DT_MAX:e}"
        ));
    }
    let half_drive_z_delta = exact_relative_error(
        half_drive.observables.port_impedance_pa_s_per_m3,
        centre.observables.port_impedance_pa_s_per_m3,
    )?;
    if half_drive_z_delta >= EXACT_HALF_DRIVE_LINEARITY_MAX {
        return Err(format!(
            "half_drive_Z convergence/linearity error {half_drive_z_delta:e} exceeds {EXACT_HALF_DRIVE_LINEARITY_MAX:e}"
        ));
    }
    // The transfer-ratio deltas are computed here for the convergence report
    // below and gated right after it, preserving the historical check order.
    let nonlinear_mesh_h_delta = exact_relative_error(
        centre_coarse_mesh.observables.direct_box_to_port_flow_ratio,
        centre.observables.direct_box_to_port_flow_ratio,
    )?;
    let dt_h_delta = exact_relative_error(
        centre_coarse_dt.observables.direct_box_to_port_flow_ratio,
        centre.observables.direct_box_to_port_flow_ratio,
    )?;
    let half_drive_h_delta = exact_relative_error(
        half_drive.observables.direct_box_to_port_flow_ratio,
        centre.observables.direct_box_to_port_flow_ratio,
    )?;
    println!(
        "EXACT_CONVERGENCE_NONLINEAR frequency_hz={:.9} coarse_mesh_nodes={} fine_mesh_nodes={} mesh_z_delta={:.9e} mesh_h_delta={:.9e} coarse_dt={:.9e} fine_dt={:.9e} dt_z_delta={:.9e} dt_h_delta={:.9e} half_drive_z_delta={:.9e} half_drive_h_delta={:.9e} tail_z_delta={:.9e} tail_h_delta={:.9e}",
        frequencies[1],
        centre_coarse_mesh.nodes,
        centre.nodes,
        nonlinear_mesh_z_delta,
        nonlinear_mesh_h_delta,
        centre_coarse_dt.dt,
        centre.dt,
        dt_z_delta,
        dt_h_delta,
        half_drive_z_delta,
        half_drive_h_delta,
        centre.tail_z_delta,
        centre.tail_h_delta,
    );
    if nonlinear_mesh_h_delta >= EXACT_NONLINEAR_MESH_DT_MAX {
        return Err(format!(
            "nonlinear_mesh_H convergence/linearity error {nonlinear_mesh_h_delta:e} exceeds {EXACT_NONLINEAR_MESH_DT_MAX:e}"
        ));
    }
    if dt_h_delta >= EXACT_NONLINEAR_MESH_DT_MAX {
        return Err(format!(
            "dt_H convergence/linearity error {dt_h_delta:e} exceeds {EXACT_NONLINEAR_MESH_DT_MAX:e}"
        ));
    }
    if half_drive_h_delta >= EXACT_HALF_DRIVE_LINEARITY_MAX {
        return Err(format!(
            "half_drive_H convergence/linearity error {half_drive_h_delta:e} exceeds {EXACT_HALF_DRIVE_LINEARITY_MAX:e}"
        ));
    }

    let mut gate_reports = Vec::with_capacity(frequencies.len());
    let mut worst_tail_z_delta = centre.tail_z_delta;
    let mut worst_tail_h_delta = centre.tail_h_delta;
    for study_point in [&centre_coarse_mesh, &centre_coarse_dt, &half_drive] {
        worst_tail_z_delta = worst_tail_z_delta.max(study_point.tail_z_delta);
        worst_tail_h_delta = worst_tail_h_delta.max(study_point.tail_h_delta);
    }
    for (index, frequency_hz) in frequencies.into_iter().enumerate() {
        if index == 1 {
            gate_reports.push(validate_exact_frequency_row(
                frequency_hz,
                &centre,
                &fine_reference,
            )?);
            continue;
        }
        // Keep each completed point alive through the comparison/print.
        let point = run_exact_nonlinear_point(
            &fine_mesh,
            &fine_geometry,
            frequency_hz,
            EXACT_VOLTAGE_PEAK_V,
            EXACT_SAMPLES_PER_CYCLE,
        )?;
        worst_tail_z_delta = worst_tail_z_delta.max(point.tail_z_delta);
        worst_tail_h_delta = worst_tail_h_delta.max(point.tail_h_delta);
        let reference =
            solve_round_port_linear_reference(&fine_mesh, &reference_config, frequency_hz)
                .map_err(|error| error.to_string())?;
        gate_reports.push(validate_exact_frequency_row(
            frequency_hz,
            &point,
            &reference,
        )?);
    }
    if gate_reports.len() != frequencies.len() {
        return Err(format!(
            "exact reference sweep gated {} of {} frequencies",
            gate_reports.len(),
            frequencies.len()
        ));
    }
    let worst = |select: fn(&LinearReferenceGateReport) -> f64| {
        gate_reports.iter().map(select).fold(0.0_f64, f64::max)
    };
    Ok(ExactReferenceMetrics {
        reference_z_delta,
        reference_h_delta,
        nonlinear_mesh_z_delta,
        nonlinear_mesh_h_delta,
        dt_z_delta,
        dt_h_delta,
        half_drive_z_delta,
        half_drive_h_delta,
        worst_tail_z_delta,
        worst_tail_h_delta,
        gate_impedance_error: worst(|report| report.impedance_relative_error),
        gate_flow_ratio_error: worst(|report| report.flow_ratio_relative_error),
        gate_identity_error: worst(|report| report.transfer_identity_relative_error),
        gate_reference_residual: worst(|report| report.reference_relative_residual),
        gate_flux_difference: worst(|report| report.reference_flux_relative_difference),
        gate_cap_continuity: worst(|report| report.cap_continuity_relative_error),
        gate_energy_algebraic: worst(|report| report.energy_algebraic_relative_closure),
        gate_energy_physical: worst(|report| report.energy_physical_relative_closure),
        gate_energy_interface: worst(|report| report.energy_interface_relative_residual),
        gate_thd: worst(|report| report.thd_fraction),
        gate_mach: worst(|report| report.mach),
        frequencies_gated: gate_reports.len(),
    })
}

fn validate_exact_frequency_row(
    frequency_hz: f64,
    nonlinear: &ExactNonlinearPoint,
    reference: &ComplexPortTransfer,
) -> Result<LinearReferenceGateReport, String> {
    let z = nonlinear.observables.port_impedance_pa_s_per_m3;
    let h = nonlinear.observables.direct_box_to_port_flow_ratio;
    let _h_from_z = nonlinear.observables.box_to_port_flow_ratio_from_impedance;
    let gate = validate_coupled_against_linear_reference(
        &LinearReferenceGateInput {
            frequency_hz,
            observables: &nonlinear.observables,
            reference,
            energy: &nonlinear.energy,
            thd_fraction: nonlinear.thd,
            mach: nonlinear.mach,
        },
        linear_reference_gate_config(),
    )
    .map_err(|error| format!("CPU linear-reference gate failed: {error}"))?;
    let z_error = gate.impedance_relative_error;
    let h_error = gate.flow_ratio_relative_error;
    let identity_error = gate.transfer_identity_relative_error;
    let z_phase_error_deg = (z / reference.port_impedance_pa_s_per_m3)
        .arg()
        .to_degrees();
    let h_phase_error_deg = (h / reference.box_to_port_flow_ratio).arg().to_degrees();
    println!(
        "EXACT_GATE frequency_hz={frequency_hz:.9} z_nl_re={:.9e} z_nl_im={:.9e} z_nl_mag={:.9e} z_nl_phase_deg={:.9} z_ref_re={:.9e} z_ref_im={:.9e} z_ref_mag={:.9e} z_ref_phase_deg={:.9} z_phase_error_deg={z_phase_error_deg:.9} z_complex_rel_error={z_error:.9e} h_nl_re={:.9e} h_nl_im={:.9e} h_nl_mag={:.9e} h_nl_phase_deg={:.9} h_ref_re={:.9e} h_ref_im={:.9e} h_ref_mag={:.9e} h_ref_phase_deg={:.9} h_phase_error_deg={h_phase_error_deg:.9} h_complex_rel_error={h_error:.9e} h_identity_rel_error={identity_error:.9e} reference_residual={:.9e} reference_strong_weak_flux_delta={:.9e} reference_weak_box_out_re={:.9e} reference_weak_out_im={:.9e} reference_weak_net_storage_re={:.9e} reference_weak_net_storage_im={:.9e} reference_h_condition={:.9e} reference_energy_algebraic={:.9e} reference_energy_physical={:.9e} reference_energy_interface={:.9e} cap_continuity={:.9e} thd={:.9e} reynolds={:.9e} mach={:.9e} tail_z_delta={:.9e} tail_h_delta={:.9e} nodes={} tetrahedra={} p2_velocity_dofs={} dt={:.9e} tail_cycles={} elapsed_s={:.6}",
        z.re,
        z.im,
        z.norm(),
        z.arg().to_degrees(),
        reference.port_impedance_pa_s_per_m3.re,
        reference.port_impedance_pa_s_per_m3.im,
        reference.port_impedance_pa_s_per_m3.norm(),
        reference.port_impedance_pa_s_per_m3.arg().to_degrees(),
        h.re,
        h.im,
        h.norm(),
        h.arg().to_degrees(),
        reference.box_to_port_flow_ratio.re,
        reference.box_to_port_flow_ratio.im,
        reference.box_to_port_flow_ratio.norm(),
        reference.box_to_port_flow_ratio.arg().to_degrees(),
        reference.relative_residual,
        reference.strong_weak_exterior_flux_relative_difference,
        reference.weak_box_outward_volume_velocity_rms_m3_per_s.re,
        reference.weak_box_outward_volume_velocity_rms_m3_per_s.im,
        reference.weak_net_storage_volume_velocity_rms_m3_per_s.re,
        reference.weak_net_storage_volume_velocity_rms_m3_per_s.im,
        reference.box_to_port_flow_ratio_condition_number,
        gate.energy_algebraic_relative_closure,
        gate.energy_physical_relative_closure,
        gate.energy_interface_relative_residual,
        nonlinear.observables.cap_continuity_relative_error,
        nonlinear.thd,
        nonlinear.reynolds,
        nonlinear.mach,
        nonlinear.tail_z_delta,
        nonlinear.tail_h_delta,
        nonlinear.nodes,
        nonlinear.tetrahedra,
        nonlinear.velocity_dofs,
        nonlinear.dt,
        EXACT_TAIL_CYCLES,
        nonlinear.elapsed_s
    );
    Ok(gate)
}

/// Execute the always-on 2D leg and return its structured evidence.
fn run_2d_subcase() -> Result<Nonlinear2dEvidence, String> {
    report_literature_comparison();
    let plane_poiseuille = validate_plane_poiseuille()?;
    let (frequency_hz, measurements) = measure_coupled_levels()?;
    if measurements.len() != DRIVE_LEVELS_V_PEAK.len() {
        return Err(format!(
            "coupled level sweep returned {} of {} drive levels",
            measurements.len(),
            DRIVE_LEVELS_V_PEAK.len()
        ));
    }
    let steps_expected = COUPLED_TOTAL_CYCLES * COUPLED_SAMPLES_PER_CYCLE;
    let steps_completed_min = measurements
        .iter()
        .map(|measurement| measurement.steps_completed)
        .min()
        .unwrap_or(0);
    let compression_energy = validate_compression_and_energy(&measurements)?;
    let lumped_limit = validate_lumped_limit(frequency_hz, &measurements)?;
    let linear_thd = validate_linear_thd(&measurements)?;
    Ok(Nonlinear2dEvidence {
        plane_poiseuille,
        frequency_hz,
        lumped_limit,
        compression_energy,
        linear_thd,
        steps_completed_min,
        steps_expected,
    })
}

/// Execute the optional nightly 3D leg and return its structured evidence.
/// Every 3D reference gate must execute and converge; there is no partial
/// 3D claim.
fn run_3d_subcase() -> Result<Nonlinear3dEvidence, String> {
    if std::env::var("SPEAKER_PORT_REFERENCE_ONLY").as_deref() == Ok("1")
        || std::env::var("SPEAKER_PORT_EXACT_CENTER_ONLY").as_deref() == Ok("1")
    {
        return Err("diagnostic exact tiers (SPEAKER_PORT_REFERENCE_ONLY/SPEAKER_PORT_EXACT_CENTER_ONLY) require SPEAKER_PORT_EXACT_ONLY=1; refusing a truncated 3D claim"
            .to_string());
    }
    let poiseuille = validate_round_port_poiseuille()?;
    let exact_reference = validate_against_exact_resolved_fem()?;
    Ok(Nonlinear3dEvidence {
        poiseuille,
        exact_reference,
    })
}

/// Bounded exact-reference diagnostic tiers. These keep their legacy markers
/// and carry no subcase evidence.
fn run_exact_diagnostic() -> Result<(), String> {
    if std::env::var("SPEAKER_PORT_REFERENCE_ONLY").as_deref() == Ok("1") {
        run_exact_reference_convergence()?;
        println!("EXACT_REFERENCE_ONLY_SUCCESS");
        return Ok(());
    }
    if std::env::var("SPEAKER_PORT_EXACT_CENTER_ONLY").as_deref() == Ok("1") {
        let convergence = run_exact_reference_convergence()?;
        let _centre = run_exact_nonlinear_point(
            &convergence.fine_mesh,
            &convergence.fine_geometry,
            convergence.frequencies[1],
            EXACT_VOLTAGE_PEAK_V,
            EXACT_SAMPLES_PER_CYCLE,
        )?;
        println!("EXACT_CENTER_ONLY_SUCCESS");
        return Ok(());
    }
    let _metrics = validate_against_exact_resolved_fem()?;
    Ok(())
}

fn run() -> Result<RunOutcome, BranchError> {
    if std::env::var("SPEAKER_PORT_SOLVER_PARITY_ONLY").as_deref() == Ok("1") {
        return run_reduced_3d_solver_parity()
            .map(|()| RunOutcome::Diagnostic)
            .map_err(|message| BranchError::new("diagnostic", message));
    }
    if let Some(tier) =
        reduced_3d_tier_from_env().map_err(|message| BranchError::new("diagnostic", message))?
    {
        return run_reduced_3d_nonlinear(tier)
            .map(|()| RunOutcome::Diagnostic)
            .map_err(|message| BranchError::new("diagnostic", message));
    }
    if std::env::var("SPEAKER_PORT_POISEUILLE_ONLY").as_deref() == Ok("1") {
        let _metrics = validate_round_port_poiseuille()
            .map_err(|message| BranchError::new("diagnostic", message))?;
        println!("POISEUILLE_3D_ONLY_SUCCESS");
        return Ok(RunOutcome::Diagnostic);
    }
    if std::env::var("SPEAKER_PORT_PLANE_POISEUILLE_ONLY").as_deref() == Ok("1") {
        let _metrics = validate_plane_poiseuille()
            .map_err(|message| BranchError::new("diagnostic", message))?;
        println!("PLANE_POISEUILLE_ONLY_SUCCESS");
        return Ok(RunOutcome::Diagnostic);
    }
    if std::env::var("SPEAKER_PORT_EXACT_ONLY").as_deref() == Ok("1") {
        return run_exact_diagnostic()
            .map(|()| RunOutcome::Diagnostic)
            .map_err(|message| BranchError::new("diagnostic", message));
    }
    // Default path: the required 2D subcase always runs first so a 3D
    // failure can never mask completed 2D evidence. The 3D leg runs only
    // when explicitly enabled and reports its own status either way.
    let two_d = run_2d_subcase().map_err(|message| BranchError::new("2d", message))?;
    let three_d = if three_d_enabled() {
        match run_3d_subcase() {
            Ok(evidence) => ThreeDOutcome::Executed(evidence),
            Err(message) => ThreeDOutcome::Failed(message),
        }
    } else {
        println!("[SKIP] 3D Taylor-Hood pipe gate requires SPEAKER_PORT_3D=1 (nightly tier)");
        ThreeDOutcome::Skipped
    };
    Ok(RunOutcome::Full(Box::new(FullOutcome { two_d, three_d })))
}

fn emit_2d_result(evidence: &Nonlinear2dEvidence, elapsed_ms: u64) -> Result<(), String> {
    let plane = &evidence.plane_poiseuille;
    let lumped = &evidence.lumped_limit;
    let energy = &evidence.compression_energy;
    let result = QaResult::passed(CASE_ID_2D)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .reference("plane Poiseuille exact profile; lumped port-flow model")
        .dimension("2-D slot-port Taylor-Hood P2/P1 with coupled driver and box")
        .formulation("BDF2 Navier-Stokes coupled to a lumped driver/box circuit")
        .tolerance("plane Poiseuille error <= 4%; lumped-limit error <= 10%; linear THD < 1%; physical energy closure <= 4%; algebraic/interface energy residuals <= 1e-8")
        .metric("executed_2d", 1.0)
        .metric("drive_levels", energy.levels as f64)
        .metric("frequency_hz", evidence.frequency_hz)
        .metric(
            "coupled_steps_completed_min",
            evidence.steps_completed_min as f64,
        )
        .metric("coupled_steps_expected", evidence.steps_expected as f64)
        .metric("plane_poiseuille_relative_error", plane.relative_error)
        .metric("plane_poiseuille_u_max", plane.numerical_u_max)
        .metric("plane_poiseuille_u_exact", plane.exact_u_max)
        .metric("lumped_limit_relative_error", lumped.relative_error)
        .metric("lumped_coupled_flow_m3s", lumped.coupled_flow_m3s)
        .metric("lumped_reference_flow_m3s", lumped.reference_flow_m3s)
        .metric("linear_thd_fraction", evidence.linear_thd)
        .metric("max_physical_closure", energy.max_physical_closure)
        .metric("max_algebraic_closure", energy.max_algebraic_closure)
        .metric("max_interface_residual", energy.max_interface_residual)
        .metric("max_compression_db", energy.max_compression_db)
        .minimum("executed_2d", 1.0)
        .minimum("drive_levels", DRIVE_LEVELS_V_PEAK.len() as f64)
        .minimum(
            "coupled_steps_completed_min",
            evidence.steps_expected as f64,
        )
        .maximum(
            "plane_poiseuille_relative_error",
            PLANE_POISEUILLE_MAX_RELATIVE_ERROR,
        )
        .maximum(
            "lumped_limit_relative_error",
            LUMPED_LIMIT_MAX_RELATIVE_ERROR,
        )
        .maximum("linear_thd_fraction", LINEAR_THD_MAX_FRACTION)
        .maximum(
            "max_physical_closure",
            COUPLED_ENERGY_PHYSICAL_CLOSURE_MAX,
        )
        .maximum(
            "max_algebraic_closure",
            COUPLED_ENERGY_ALGEBRAIC_CLOSURE_MAX,
        )
        .maximum(
            "max_interface_residual",
            COUPLED_ENERGY_INTERFACE_RESIDUAL_MAX,
        )
        .elapsed_ms(elapsed_ms);
    println!("QA_RESULT: {}", result.to_json_line()?);
    Ok(())
}

fn emit_3d_passed(evidence: &Nonlinear3dEvidence, elapsed_ms: u64) -> Result<(), String> {
    let poiseuille = &evidence.poiseuille;
    let exact = &evidence.exact_reference;
    let result = QaResult::passed(CASE_ID_3D)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .reference("Hagen-Poiseuille profile; independent harmonic round-port FEM reference")
        .dimension("3-D round-port Taylor-Hood P2/P1 with coupled driver and box")
        .formulation("BDF2 Navier-Stokes coupled to a lumped driver/box circuit; coarse/fine reference ladder plus per-frequency linear-reference gates")
        .tolerance("round-port Poiseuille error <= 8%; reference convergence <= 1%; mesh/dt deltas <= 2%; half-drive linearity <= 1%; tail deltas <= 0.5%; gate impedance/transfer errors <= 3%/2%")
        .metric("executed_3d", 1.0)
        .metric("round_port_nodes", poiseuille.nodes as f64)
        .metric(
            "round_port_poiseuille_relative_error",
            poiseuille.relative_error,
        )
        .metric("round_port_u_max", poiseuille.numerical_u_max)
        .metric("round_port_u_exact", poiseuille.exact_u_max)
        .metric("reference_z_delta", exact.reference_z_delta)
        .metric("reference_h_delta", exact.reference_h_delta)
        .metric("nonlinear_mesh_z_delta", exact.nonlinear_mesh_z_delta)
        .metric("nonlinear_mesh_h_delta", exact.nonlinear_mesh_h_delta)
        .metric("dt_z_delta", exact.dt_z_delta)
        .metric("dt_h_delta", exact.dt_h_delta)
        .metric("half_drive_z_delta", exact.half_drive_z_delta)
        .metric("half_drive_h_delta", exact.half_drive_h_delta)
        .metric("worst_tail_z_delta", exact.worst_tail_z_delta)
        .metric("worst_tail_h_delta", exact.worst_tail_h_delta)
        .metric("gate_impedance_error", exact.gate_impedance_error)
        .metric("gate_flow_ratio_error", exact.gate_flow_ratio_error)
        .metric("gate_identity_error", exact.gate_identity_error)
        .metric("gate_reference_residual", exact.gate_reference_residual)
        .metric("gate_flux_difference", exact.gate_flux_difference)
        .metric("gate_cap_continuity", exact.gate_cap_continuity)
        .metric("gate_energy_algebraic", exact.gate_energy_algebraic)
        .metric("gate_energy_physical", exact.gate_energy_physical)
        .metric("gate_energy_interface", exact.gate_energy_interface)
        .metric("gate_thd", exact.gate_thd)
        .metric("gate_mach", exact.gate_mach)
        .metric("frequencies_gated", exact.frequencies_gated as f64)
        .minimum("executed_3d", 1.0)
        .minimum("frequencies_gated", 3.0)
        .maximum(
            "round_port_poiseuille_relative_error",
            ROUND_PORT_POISEUILLE_MAX_RELATIVE_ERROR,
        )
        .maximum("reference_z_delta", EXACT_REFERENCE_CONVERGENCE_MAX)
        .maximum("reference_h_delta", EXACT_REFERENCE_CONVERGENCE_MAX)
        .maximum("nonlinear_mesh_z_delta", EXACT_NONLINEAR_MESH_DT_MAX)
        .maximum("nonlinear_mesh_h_delta", EXACT_NONLINEAR_MESH_DT_MAX)
        .maximum("dt_z_delta", EXACT_NONLINEAR_MESH_DT_MAX)
        .maximum("dt_h_delta", EXACT_NONLINEAR_MESH_DT_MAX)
        .maximum("half_drive_z_delta", EXACT_HALF_DRIVE_LINEARITY_MAX)
        .maximum("half_drive_h_delta", EXACT_HALF_DRIVE_LINEARITY_MAX)
        .maximum("worst_tail_z_delta", EXACT_TAIL_DELTA_MAX)
        .maximum("worst_tail_h_delta", EXACT_TAIL_DELTA_MAX)
        .maximum(
            "gate_impedance_error",
            GATE_MAX_IMPEDANCE_RELATIVE_ERROR,
        )
        .maximum(
            "gate_flow_ratio_error",
            GATE_MAX_FLOW_RATIO_RELATIVE_ERROR,
        )
        .maximum(
            "gate_identity_error",
            GATE_MAX_TRANSFER_IDENTITY_RELATIVE_ERROR,
        )
        .maximum(
            "gate_reference_residual",
            GATE_MAX_REFERENCE_RELATIVE_RESIDUAL,
        )
        .maximum(
            "gate_flux_difference",
            GATE_MAX_REFERENCE_FLUX_RELATIVE_DIFFERENCE,
        )
        .maximum(
            "gate_cap_continuity",
            GATE_MAX_CAP_CONTINUITY_RELATIVE_ERROR,
        )
        .maximum(
            "gate_energy_algebraic",
            GATE_MAX_ENERGY_ALGEBRAIC_RELATIVE_CLOSURE,
        )
        .maximum(
            "gate_energy_physical",
            GATE_MAX_ENERGY_PHYSICAL_RELATIVE_CLOSURE,
        )
        .maximum(
            "gate_energy_interface",
            GATE_MAX_ENERGY_INTERFACE_RELATIVE_RESIDUAL,
        )
        .maximum("gate_thd", GATE_MAX_THD_FRACTION)
        .maximum("gate_mach", GATE_MAX_MACH)
        .elapsed_ms(elapsed_ms);
    println!("QA_RESULT: {}", result.to_json_line()?);
    Ok(())
}

fn emit_3d_skipped(reason: &str) -> Result<(), String> {
    // A skipped 3D leg carries only its executed flag. No 3D metrics or
    // thresholds appear unless the 3D gates actually ran.
    let result = QaResult::skipped(CASE_ID_3D, reason).metric("executed_3d", 0.0);
    println!("QA_RESULT: {}", result.to_json_line()?);
    Ok(())
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(RunOutcome::Diagnostic) => {
            println!("SUCCESS");
        }
        Ok(RunOutcome::Full(outcome)) => {
            let elapsed_ms = started.elapsed().as_millis() as u64;
            if let Err(error) = emit_2d_result(&outcome.two_d, elapsed_ms) {
                eprintln!("FAILED: 2D subcase evidence error: {error}");
                println!("FAILED");
                std::process::exit(1);
            }
            let mut failed_message: Option<String> = None;
            let status_3d = match &outcome.three_d {
                ThreeDOutcome::Executed(evidence) => {
                    if let Err(error) = emit_3d_passed(evidence, elapsed_ms) {
                        eprintln!("FAILED: 3D subcase evidence error: {error}");
                        println!("FAILED");
                        std::process::exit(1);
                    }
                    "passed"
                }
                ThreeDOutcome::Skipped => {
                    if let Err(error) = emit_3d_skipped(
                        "SPEAKER_PORT_3D != 1: nightly 3D gates did not run; whole-binary SUCCESS covers the 2D subcase only",
                    ) {
                        eprintln!("FAILED: 3D subcase evidence error: {error}");
                        println!("FAILED");
                        std::process::exit(1);
                    }
                    "skipped"
                }
                ThreeDOutcome::Failed(message) => {
                    let failed = QaResult::failed(CASE_ID_3D, message.clone())
                        .requested_backend("cpu")
                        .actual_backend("cpu")
                        .metric("executed_3d", 1.0);
                    match failed.to_json_line() {
                        Ok(line) => println!("QA_RESULT: {line}"),
                        Err(error) => {
                            eprintln!("FAILED: 3D failure evidence error: {error}");
                            println!("FAILED");
                            std::process::exit(1);
                        }
                    }
                    failed_message = Some(message.clone());
                    "failed"
                }
            };
            let executed_3d = i32::from(!matches!(outcome.three_d, ThreeDOutcome::Skipped));
            println!(
                "SUBCASE_SUMMARY subcase_2d=passed subcase_3d={status_3d} executed_2d=1 executed_3d={executed_3d} executed_cases={} requested_cases=2",
                1 + executed_3d,
            );
            if let Some(message) = failed_message {
                eprintln!("FAILED: 3D subcase failed: {message}");
                println!("FAILED");
                std::process::exit(1);
            }
            println!("SUCCESS");
        }
        Err(error) => {
            if error.branch == "diagnostic" {
                eprintln!("FAILED: {}", error.message);
                println!("FAILED");
                std::process::exit(1);
            }
            // The 2D leg is required: a 2D failure fails the binary and the
            // 3D leg is reported as not attempted.
            let failed = QaResult::failed(CASE_ID_2D, error.message.clone())
                .requested_backend("cpu")
                .actual_backend("cpu");
            match failed.to_json_line() {
                Ok(line) => println!("QA_RESULT: {line}"),
                Err(encode_error) => {
                    eprintln!("FAILED: 2D failure evidence error: {encode_error}");
                    println!("FAILED");
                    std::process::exit(1);
                }
            }
            if emit_3d_skipped("not attempted: the required 2D subcase failed first").is_err() {
                println!("FAILED");
                std::process::exit(1);
            }
            println!(
                "SUBCASE_SUMMARY subcase_2d=failed subcase_3d=skipped executed_2d=1 executed_3d=0 executed_cases=1 requested_cases=2"
            );
            eprintln!("FAILED: 2D subcase failed: {}", error.message);
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "slow release-mode resolved port-flow QA"]
    fn speaker_port_nonlinear_gates_pass() {
        run().expect("speaker port-nonlinear QA gates should pass");
    }

    #[test]
    #[ignore = "~20 minute release-only 126-to-252 sample settled convergence evidence"]
    fn tiny_drive_full_tail_timestep_convergence() {
        let frequency_hz = fixture_frequency_hz().unwrap();
        let voltage_peak_v = DRIVE_LEVELS_V_PEAK[0];
        let mesh = build_slot_port_mesh(&SlotPortGeometry {
            duct_length_m: DUCT_LENGTH_M,
            duct_height_m: DUCT_HEIGHT_M,
            chamber_length_m: 0.05,
            chamber_height_m: 0.08,
            mesh_size_m: 0.01,
        })
        .unwrap();
        let velocity_space = VectorH1Space::new(PolynomialDegree::P2, 2);
        let pressure_space = H1Space::new(PolynomialDegree::P1);
        let mut points = Vec::new();
        for samples_per_cycle in [126usize, 252] {
            let config = coupled_config_at_samples(voltage_peak_v, frequency_hz, samples_per_cycle);
            let result =
                run_coupled_port(&mesh, &velocity_space, &pressure_space, &config).unwrap();
            let observable = coupled_complex_observables(
                &result,
                config.bdf.dt,
                frequency_hz,
                samples_per_cycle,
                COUPLED_TAIL_CYCLES,
                COUPLED_TOTAL_CYCLES - COUPLED_TAIL_CYCLES,
                BOX_VOLUME_M3,
                RHO,
                SOUND_SPEED,
            )
            .unwrap();
            let start = result.port_flow_m3s.len() - observable.tail_samples;
            let amplitude =
                dft_amplitude(&result.port_flow_m3s[start..], config.bdf.dt, frequency_hz);
            points.push((amplitude, observable));
        }
        let amplitude_delta = (points[0].0 - points[1].0).abs() / points[1].0;
        let z_delta = exact_relative_error(
            points[0].1.port_impedance_pa_s_per_m3,
            points[1].1.port_impedance_pa_s_per_m3,
        )
        .unwrap();
        let h_delta = exact_relative_error(
            points[0].1.direct_box_to_port_flow_ratio,
            points[1].1.direct_box_to_port_flow_ratio,
        )
        .unwrap();
        // This is a timestep-convergence gate. Agreement with the reduced
        // lumped model is checked separately by `validate_lumped_limit`; an
        // arbitrary absolute model-error ceiling would not establish temporal
        // convergence here.
        assert!(
            amplitude_delta < 0.01,
            "amplitude delta={amplitude_delta:e}"
        );
        assert!(z_delta < 0.002, "complex Z delta={z_delta:e}");
        assert!(h_delta < 0.025, "complex H delta={h_delta:e}");
        assert!(points[0].1.cap_continuity_relative_error < 1.0e-12);
        assert!(points[1].1.cap_continuity_relative_error < 1.0e-12);
    }
}
#[test]
fn centerline_sampler_uses_vector_dof_coordinates_directly() {
    let coordinates = [
        [0.5, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.55, 0.001, 0.0],
        [0.55, 0.001, 0.0],
        [0.55, 0.001, 0.0],
        [0.9, 0.0, 0.0],
        [0.9, 0.0, 0.0],
        [0.9, 0.0, 0.0],
    ];
    let velocity = [1.0, 1.0e6, 1.0e6, 2.0, 1.0e6, 1.0e6, 9.0, 1.0e6, 1.0e6];
    let peak = centerline_peak_velocity(&coordinates, &velocity, 3, 0.4, 0.6, 0.01)
        .expect("synthetic vector state should expose x-component samples");
    assert_eq!(peak, 2.0);
}

#[test]
fn exact_round_port_reference_ladder_refines_geometry() {
    let coarse_geometry = exact_geometry(0.02);
    let fine_geometry = exact_geometry(0.01);
    let coarse = build_round_port_mesh(&coarse_geometry).unwrap();
    let fine = build_round_port_mesh(&fine_geometry).unwrap();

    assert!(fine.num_nodes() > coarse.num_nodes());
    assert!(fine.elements.len() > coarse.elements.len());
    for boundary in fine
        .boundaries
        .iter()
        .filter(|boundary| boundary.marker == MARKER_WALL)
    {
        for &node in &boundary.nodes {
            let point = fine.nodes[node];
            let expected_radius = fine_geometry.local_radius_m(point.x).unwrap();
            assert!((point.y.hypot(point.z) - expected_radius).abs() < 1.0e-12);
        }
    }
}
