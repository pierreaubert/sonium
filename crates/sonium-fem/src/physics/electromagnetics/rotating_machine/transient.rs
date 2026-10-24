//! Current-drive rotating transient over one electrical period.
//!
//! Each frame is a nonlinear magnetostatic `A_z` solve on the
//! sliding-band mesh at an integer ring shift, warm-started from the
//! previous frame. Outputs are terminal flux linkages and currents,
//! Arkkio and flux-linkage torque, per-element `B(t)` histories for the
//! iron-loss integration (stator in the stator frame, rotor rotated
//! into the rotor frame), and the solved field of every frame.
//!
//! Known simplifications versus the reference solver, in decreasing
//! impact order: rotor iron loss uses the raw one-period window (the
//! reference chains pole-pair images into a commensurate window),
//! and demagnetisation is wired into the current-drive march only
//! (not the eddy or voltage drivers).

// Rust guideline compliant 2026-02-21

use std::collections::BTreeMap;

use crate::basis::PolynomialDegree;
use crate::mesh::Mesh;

use super::super::magnetostatics::BhCurve;
use super::az::{
    AzError, AzProblem, AzQuadraturePoint, AzSolveConfig, ConductorBlock, TransientStep, solve_az,
};
use super::demag::{MagnetDemagCurve, apply_demag_update};
use super::losses::{IronLossReport, LossError, SteelLoss, iron_loss};
use super::materials::{AzMaterial, MagnetLaw};
use super::motor_mesh::{
    MARKER_AIR, MARKER_COIL_BASE, MARKER_MAGNET_BASE, MARKER_ROTOR, MARKER_SHAFT, MARKER_STATOR,
    MotorMesh,
};
use super::torque::{
    TorqueError, TorqueMetrics, arkkio_torque_nm, flux_linkage_mean_torque_nm,
    space_vector_mean_nm, terminal_work_mean_nm, torque_metrics,
};
use super::winding::{Winding, WindingError};

/// Builds the laminated (geometric) B-H curve from steel `(H, B)` data.
///
/// `B_geom(H) = k_f * B_steel(H) + (1 - k_f) * mu_0 * H`, returned in
/// the `(B, H)` order [`BhCurve`] expects.
///
/// # Errors
///
/// Propagates [`BhCurve`] validation failures as
/// [`TransientError::Material`].
pub fn laminated_bh_curve(
    steel_hb_points: &[(f64, f64)],
    stacking_factor: f64,
) -> Result<BhCurve, TransientError> {
    let kf = stacking_factor.clamp(0.05, 1.0);
    let points: Vec<(f64, f64)> = steel_hb_points
        .iter()
        .map(|&(h, b)| {
            (
                kf * b + (1.0 - kf) * super::super::magnetostatics::MU_0 * h,
                h,
            )
        })
        .collect();
    BhCurve::new(points).map_err(|error| TransientError::Material(error.to_string()))
}

/// Material set for a motor transient.
#[derive(Debug, Clone)]
pub struct MotorMaterials {
    /// Field material of stator and rotor iron (laminated curve).
    pub steel: AzMaterial,
    /// Loss record of the same steel.
    pub steel_loss: SteelLoss,
    /// Magnet remanence magnitude in tesla.
    pub magnet_br_t: f64,
    /// Magnet recoil relative permeability.
    pub magnet_mu_rec: f64,
    /// Magnet polarity (+1/-1, tangential CCW) per magnet marker order.
    pub magnet_polarity: Vec<f64>,
    /// Magnet conductivity in S/m (eddy stages).
    pub magnet_sigma_s_per_m: f64,
    /// Shaft relative permeability.
    pub shaft_mu_r: f64,
    /// Shaft conductivity in S/m (eddy stages).
    pub shaft_sigma_s_per_m: f64,
}

/// Operating point of a current-drive run.
#[derive(Debug, Clone, Copy)]
pub struct DrivePoint {
    /// Phase current in A rms (terminal).
    pub i_phase_rms_a: f64,
    /// Advance angle from the q-axis in degrees (0 = pure q-axis).
    pub gamma_deg: f64,
    /// Calibrated d-axis offset in electrical degrees.
    pub daxis_deg: f64,
    /// Mechanical speed in rpm.
    pub rpm: f64,
    /// Coil temperature in °C (copper conductivity for eddy runs).
    pub coil_temp_c: f64,
}

/// Frame plan and solver controls.
#[derive(Debug, Clone)]
pub struct TransientConfig {
    /// Frames per electrical period.
    pub steps_per_period: usize,
    /// Stack length in metres.
    pub stack_length_m: f64,
    /// Rotor outer radius in metres (Arkkio inner radius).
    pub rotor_outer_m: f64,
    /// Stator bore radius in metres (Arkkio outer radius).
    pub stator_inner_m: f64,
    /// Per-frame field solver controls.
    pub solve: AzSolveConfig,
    /// Irreversible-demagnetisation ratchet; `None` keeps magnets at
    /// full strength (current-drive march only).
    pub demag: Option<DemagConfig>,
}

/// Controls of the demagnetisation ratchet in a current-drive run.
#[derive(Debug, Clone)]
pub struct DemagConfig {
    /// The magnet's second-quadrant intrinsic curve.
    pub curve: MagnetDemagCurve,
    /// Discarded settling periods marched before the reported one,
    /// so reported frames see a magnet that has stopped moving.
    pub settle_periods: usize,
    /// Maximum same-frame re-solves while the magnet still moves
    /// (reported frames only; settling frames carry the weakening
    /// into the next frame).
    pub max_passes: usize,
}

/// One solved frame.
#[derive(Debug, Clone)]
pub struct TransientFrame {
    /// Rotor mechanical angle in degrees.
    pub rotor_angle_deg: f64,
    /// Phase currents in amperes (branch peak convention of the solve).
    pub currents_a: [f64; 3],
    /// Phase flux linkages in weber-turns.
    pub psi_wb_turn: [f64; 3],
    /// Nodal `A_z` in Wb/m.
    pub vector_potential_wb_per_m: Vec<f64>,
    /// Quadrature-point field state.
    pub quadrature_state: Vec<AzQuadraturePoint>,
}

/// Result of one current-drive electrical period.
#[derive(Debug, Clone)]
pub struct TransientResult {
    /// Solved frames in rotor-angle order.
    pub frames: Vec<TransientFrame>,
    /// Arkkio torque per frame in N m.
    pub torque_arkkio_nm: Vec<f64>,
    /// Mean torque from flux linkages in N m (discrete `sum i delta(psi)`).
    pub torque_flux_linkage_mean_nm: f64,
    /// Terminal-work mean torque in N m (reference headline mean).
    pub torque_terminal_work_mean_nm: f64,
    /// Space-vector mean torque in N m (diagnostic).
    pub torque_space_vector_mean_nm: f64,
    /// Mean/ripple summary of the Arkkio series.
    pub torque_arkkio_metrics: TorqueMetrics,
    /// Stator iron loss over the captured window.
    pub stator_iron_loss: IronLossReport,
    /// Rotor iron loss over the captured window (open-window, see docs).
    pub rotor_iron_loss: IronLossReport,
    /// Electrical frequency in Hz.
    pub f_elec_hz: f64,
    /// Plain mean of the per-element demagnetisation factors over
    /// magnet elements at the end of the run (1 when demag is off).
    pub demag_br_mean: f64,
    /// Area-weighted mean demagnetisation factor over magnet
    /// elements (the mesh-convergent "Br kept by volume"; the plain
    /// element-count mean above depends on the mesh population).
    pub demag_br_mean_area_weighted: f64,
    /// Worst (smallest) per-element demagnetisation factor.
    pub demag_br_min: f64,
    /// Per-element demagnetisation factors over all mesh elements
    /// (1 outside magnets; empty when demag is off).
    pub demag_factors: Vec<f64>,
}

/// Per-wire and per-slot copper cross-sections measured from the mesh.
struct CoilData {
    /// Copper area per slot (sum of its wires), in slot order.
    slot_areas_m2: Vec<f64>,
    /// Wire indices per slot.
    wires_of_slot: Vec<Vec<usize>>,
}

fn coil_data(motor: &MotorMesh, num_slots: usize) -> CoilData {
    let mesh = &motor.mesh;
    let mut wire_areas = vec![0.0; motor.coil_slot_of_wire.len()];
    for element in &mesh.elements {
        let wire = element.marker - MARKER_COIL_BASE;
        if !(0..wire_areas.len() as i32).contains(&wire) {
            continue;
        }
        let a = mesh.nodes[element.nodes[0]];
        let b = mesh.nodes[element.nodes[1]];
        let c = mesh.nodes[element.nodes[2]];
        wire_areas[wire as usize] +=
            0.5 * ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)).abs();
    }
    let mut slot_areas_m2 = vec![0.0; num_slots];
    let mut wires_of_slot: Vec<Vec<usize>> = vec![Vec::new(); num_slots];
    for (wire, &slot) in motor.coil_slot_of_wire.iter().enumerate() {
        if slot < num_slots {
            slot_areas_m2[slot] += wire_areas[wire];
            wires_of_slot[slot].push(wire);
        }
    }
    CoilData {
        slot_areas_m2,
        wires_of_slot,
    }
}

/// Area-weighted mean `A_z` over all wires of one slot.
fn slot_mean_a(state: &[AzQuadraturePoint], wires: &[usize]) -> f64 {
    let (mut integral, mut area) = (0.0, 0.0);
    for point in state {
        let wire = point.marker - MARKER_COIL_BASE;
        if wire >= 0 && wires.contains(&(wire as usize)) {
            integral += point.vector_potential_wb_per_m * point.integration_weight_m2;
            area += point.integration_weight_m2;
        }
    }
    if area > 0.0 { integral / area } else { 0.0 }
}

/// Marker-element groups with areas, for B-history capture.
fn element_groups(mesh: &Mesh, marker: i32) -> (Vec<usize>, Vec<f64>) {
    let mut indices = Vec::new();
    let mut areas = Vec::new();
    for (index, element) in mesh.elements.iter().enumerate() {
        if element.marker != marker {
            continue;
        }
        let a = mesh.nodes[element.nodes[0]];
        let b = mesh.nodes[element.nodes[1]];
        let c = mesh.nodes[element.nodes[2]];
        indices.push(index);
        areas.push(0.5 * ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)).abs());
    }
    (indices, areas)
}

/// Element-mean B from a quadrature state, one `[Bx, By]` per element.
fn element_mean_b(state: &[AzQuadraturePoint], num_elements: usize) -> Vec<[f64; 2]> {
    let mut sums = vec![[0.0; 2]; num_elements];
    let mut weights = vec![0.0; num_elements];
    for point in state {
        sums[point.element_index][0] += point.flux_density_t[0] * point.integration_weight_m2;
        sums[point.element_index][1] += point.flux_density_t[1] * point.integration_weight_m2;
        weights[point.element_index] += point.integration_weight_m2;
    }
    for (sum, weight) in sums.iter_mut().zip(&weights) {
        if *weight > 0.0 {
            sum[0] /= weight;
            sum[1] /= weight;
        }
    }
    sums
}

/// Runs one electrical period of current-drive frames.
///
/// The step plan must be commensurate with the ring: one electrical
/// period spans `n_ring / pole_pairs` pitches, each frame an integer
/// number of pitches.
///
/// # Errors
///
/// Returns [`TransientError`] for an incommensurate step plan, invalid
/// winding/material data, or a failed frame solve.
pub fn run_current_drive_transient(
    motor: &mut MotorMesh,
    degree: PolynomialDegree,
    winding: &Winding,
    materials: &MotorMaterials,
    drive: &DrivePoint,
    config: &TransientConfig,
) -> Result<TransientResult, TransientError> {
    winding.validate()?;
    let ring_pitches = motor.ring_nodes.len()
        / match degree {
            PolynomialDegree::P2 => 2,
            _ => 1,
        };
    let period_pitches = ring_pitches / winding.pole_pairs;
    if !ring_pitches.is_multiple_of(winding.pole_pairs)
        || !period_pitches.is_multiple_of(config.steps_per_period)
        || config.steps_per_period == 0
    {
        return Err(TransientError::IncommensurateSteps {
            ring_pitches,
            pole_pairs: winding.pole_pairs,
            steps_per_period: config.steps_per_period,
        });
    }
    let step_pitches = (period_pitches / config.steps_per_period) as i64;
    let num_slots = winding.layout.len();
    let coils = coil_data(motor, num_slots);
    if coils.slot_areas_m2.iter().any(|area| *area <= 0.0) {
        return Err(TransientError::Material(
            "a winding slot has no copper area in the mesh".to_string(),
        ));
    }
    // Magnet centroid angles at shift zero (rotor frame).
    motor.set_rotor_shift(0);
    let mut magnet_angles: Vec<f64> = Vec::with_capacity(materials.magnet_polarity.len());
    for index in 0..materials.magnet_polarity.len() {
        let marker = MARKER_MAGNET_BASE + index as i32;
        let (mut cx, mut cy, mut area) = (0.0, 0.0, 0.0);
        for element in motor.mesh.elements.iter().filter(|e| e.marker == marker) {
            let a = motor.mesh.nodes[element.nodes[0]];
            let b = motor.mesh.nodes[element.nodes[1]];
            let c = motor.mesh.nodes[element.nodes[2]];
            let element_area = 0.5 * ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)).abs();
            cx += (a.x + b.x + c.x) / 3.0 * element_area;
            cy += (a.y + b.y + c.y) / 3.0 * element_area;
            area += element_area;
        }
        if area <= 0.0 {
            return Err(TransientError::Material(format!(
                "magnet {index} has no elements in the mesh"
            )));
        }
        magnet_angles.push((cy / area).atan2(cx / area));
    }
    // Dirichlet: A_z = 0 on the outer boundary nodes.
    let outer_radius = motor
        .mesh
        .nodes
        .iter()
        .map(|node| node.x.hypot(node.y))
        .fold(0.0, f64::max);
    let dirichlet: Vec<usize> = motor
        .mesh
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| (node.x.hypot(node.y) - outer_radius).abs() < 1.0e-6)
        .map(|(index, _)| index)
        .collect();

    let i_peak_branch = drive.i_phase_rms_a * std::f64::consts::SQRT_2 / winding.n_parallel;
    let f_elec = drive.rpm * winding.pole_pairs as f64 / 60.0;
    let num_elements = motor.mesh.num_elements();
    let (stator_elements, stator_areas) = element_groups(&motor.mesh, MARKER_STATOR);
    let (rotor_elements, rotor_areas) = element_groups(&motor.mesh, MARKER_ROTOR);
    let frames_count = config.steps_per_period;
    let mut stator_bx = vec![0.0; frames_count * stator_elements.len()];
    let mut stator_by = vec![0.0; frames_count * stator_elements.len()];
    let mut rotor_bx = vec![0.0; frames_count * rotor_elements.len()];
    let mut rotor_by = vec![0.0; frames_count * rotor_elements.len()];

    let mut frames: Vec<TransientFrame> = Vec::with_capacity(frames_count);
    let mut torque_arkkio: Vec<f64> = Vec::with_capacity(frames_count);
    let mut warm_start: Option<Vec<f64>> = None;
    // Demagnetisation state: a discarded settling pass runs first so
    // the reported period measures a magnet that has stopped moving.
    let demag_settle = config
        .demag
        .as_ref()
        .map_or(0, |demag| demag.settle_periods);
    let mut magnet_scale: Vec<f64> = if config.demag.is_some() {
        vec![1.0; num_elements]
    } else {
        Vec::new()
    };
    let magnet_elements: Vec<(usize, usize)> = if config.demag.is_some() {
        motor
            .mesh
            .elements
            .iter()
            .enumerate()
            .filter(|(_, element)| {
                element.marker >= MARKER_MAGNET_BASE
                    && element.marker < MARKER_MAGNET_BASE + materials.magnet_polarity.len() as i32
            })
            .map(|(index, element)| (index, (element.marker - MARKER_MAGNET_BASE) as usize))
            .collect()
    } else {
        Vec::new()
    };
    let total_frames = frames_count * (demag_settle + 1);
    for global_frame in 0..total_frames {
        let frame = global_frame.checked_sub(demag_settle * frames_count);
        let shift = global_frame as i64 * step_pitches;
        motor.set_rotor_shift(shift);
        let rotor_angle_deg = motor.shift_angle_deg(shift);
        let currents = winding.phase_currents_a(
            rotor_angle_deg,
            drive.gamma_deg,
            drive.daxis_deg,
            i_peak_branch,
        );
        // Reported frames re-solve until the magnet stops moving:
        // torque and losses may not be billed on a magnet that no
        // longer exists. Settling frames just carry the weakening
        // forward (judged on the stronger, pre-weakening field).
        let mut pass = 0usize;
        let (solution, mean_b) = loop {
            let problem = frame_problem(
                motor,
                degree,
                winding,
                materials,
                &magnet_angles,
                &coils,
                &currents,
                rotor_angle_deg.to_radians(),
                config.stack_length_m,
                None,
                &magnet_scale,
            )?;
            let solution = solve_az(
                &problem,
                &dirichlet,
                &config.solve,
                warm_start.as_deref(),
                None,
            )?;
            warm_start = Some(solution.vector_potential_wb_per_m.clone());
            let mean_b = element_mean_b(&solution.quadrature_state, num_elements);
            let Some(demag) = &config.demag else {
                break (solution, mean_b);
            };
            let theta_rad = rotor_angle_deg.to_radians();
            let directions: Vec<(usize, [f64; 2])> = magnet_elements
                .iter()
                .map(|&(element, magnet)| {
                    let phi = magnet_angles[magnet] + theta_rad;
                    let polarity = materials.magnet_polarity[magnet];
                    (element, [-polarity * phi.sin(), polarity * phi.cos()])
                })
                .collect();
            let moved = apply_demag_update(&demag.curve, &mean_b, &directions, &mut magnet_scale);
            pass += 1;
            if !moved || frame.is_none() || pass >= demag.max_passes.max(1) {
                break (solution, mean_b);
            }
        };
        let Some(frame) = frame else {
            continue;
        };
        let slot_means: Vec<f64> = coils
            .wires_of_slot
            .iter()
            .map(|wires| slot_mean_a(&solution.quadrature_state, wires))
            .collect();
        let psi = winding.phase_flux_linkages_wb_turn(&slot_means, config.stack_length_m)?;
        torque_arkkio.push(arkkio_torque_nm(
            &solution.quadrature_state,
            config.rotor_outer_m,
            config.stator_inner_m,
            config.stack_length_m,
        ));
        for (slot, &element) in stator_elements.iter().enumerate() {
            stator_bx[frame * stator_elements.len() + slot] = mean_b[element][0];
            stator_by[frame * stator_elements.len() + slot] = mean_b[element][1];
        }
        let theta = rotor_angle_deg.to_radians();
        let (cos_t, sin_t) = (theta.cos(), theta.sin());
        for (slot, &element) in rotor_elements.iter().enumerate() {
            // Rotate B into the rotor frame: R(-theta) * B.
            let [bx, by] = mean_b[element];
            rotor_bx[frame * rotor_elements.len() + slot] = cos_t * bx + sin_t * by;
            rotor_by[frame * rotor_elements.len() + slot] = -sin_t * bx + cos_t * by;
        }
        frames.push(TransientFrame {
            rotor_angle_deg,
            currents_a: currents,
            psi_wb_turn: psi,
            vector_potential_wb_per_m: solution.vector_potential_wb_per_m,
            quadrature_state: solution.quadrature_state,
        });
    }
    motor.set_rotor_shift(0);
    let (demag_br_mean, demag_br_mean_area_weighted, demag_br_min) = if magnet_elements.is_empty() {
        (1.0, 1.0, 1.0)
    } else {
        let areas = super::thermal::element_areas_m2(&motor.mesh);
        let (mut sum, mut weighted, mut area_sum, mut min) = (0.0, 0.0, 0.0, 1.0f64);
        for &(element, _) in &magnet_elements {
            sum += magnet_scale[element];
            weighted += magnet_scale[element] * areas[element];
            area_sum += areas[element];
            min = min.min(magnet_scale[element]);
        }
        (sum / magnet_elements.len() as f64, weighted / area_sum, min)
    };

    let psi_series: [Vec<f64>; 3] = [
        frames.iter().map(|f| f.psi_wb_turn[0]).collect(),
        frames.iter().map(|f| f.psi_wb_turn[1]).collect(),
        frames.iter().map(|f| f.psi_wb_turn[2]).collect(),
    ];
    let current_series: [Vec<f64>; 3] = [
        frames.iter().map(|f| f.currents_a[0]).collect(),
        frames.iter().map(|f| f.currents_a[1]).collect(),
        frames.iter().map(|f| f.currents_a[2]).collect(),
    ];
    // Flux-linkage torque needs TERMINAL quantities: psi here is per
    // branch-summed linkage already scaled by 1/n_parallel, and the
    // currents are branch currents; the energy identity per branch set
    // multiplies back by n_parallel through the terminal current.
    let terminal_series: [Vec<f64>; 3] = current_series
        .iter()
        .map(|series| series.iter().map(|i| i * winding.n_parallel).collect())
        .collect::<Vec<_>>()
        .try_into()
        .expect("three phases");
    let torque_flux = flux_linkage_mean_torque_nm(
        &[&psi_series[0], &psi_series[1], &psi_series[2]],
        &[
            &terminal_series[0],
            &terminal_series[1],
            &terminal_series[2],
        ],
        winding.pole_pairs,
    )?;
    let mech_angles_rad: Vec<f64> = frames
        .iter()
        .map(|frame| frame.rotor_angle_deg.to_radians())
        .collect();
    let psi_refs: [&[f64]; 3] = [&psi_series[0], &psi_series[1], &psi_series[2]];
    let branch_refs: [&[f64]; 3] = [&current_series[0], &current_series[1], &current_series[2]];
    let torque_terminal = terminal_work_mean_nm(
        &psi_refs,
        &branch_refs,
        &mech_angles_rad,
        winding.pole_pairs,
        winding.n_parallel,
    )?;
    let torque_space_vector = space_vector_mean_nm(
        &psi_refs,
        &branch_refs,
        winding.pole_pairs,
        winding.n_parallel,
    );
    let dt = if f_elec > 0.0 {
        1.0 / (f_elec * frames_count as f64)
    } else {
        1.0
    };
    let stator_iron_loss = iron_loss(
        &stator_bx,
        &stator_by,
        &stator_areas,
        &materials.steel_loss,
        config.stack_length_m,
        f_elec,
        1.0,
        dt,
    )?;
    let rotor_iron_loss = iron_loss(
        &rotor_bx,
        &rotor_by,
        &rotor_areas,
        &materials.steel_loss,
        config.stack_length_m,
        f_elec,
        1.0,
        dt,
    )?;
    Ok(TransientResult {
        torque_arkkio_metrics: torque_metrics(&torque_arkkio),
        frames,
        torque_arkkio_nm: torque_arkkio,
        torque_flux_linkage_mean_nm: torque_flux,
        torque_terminal_work_mean_nm: torque_terminal,
        torque_space_vector_mean_nm: torque_space_vector,
        stator_iron_loss,
        rotor_iron_loss,
        f_elec_hz: f_elec,
        demag_br_mean,
        demag_br_mean_area_weighted,
        demag_br_min,
        demag_factors: magnet_scale,
    })
}

/// Eddy-mode frame inputs: copper conductivity and per-wire targets.
///
/// In eddy mode coils carry no imposed current density; each wire is a
/// conductor block whose net current is the branch current, and each
/// magnet and the shaft are floating (zero-net) blocks.
struct EddyFrame {
    copper_sigma_s_per_m: f64,
}

/// Builds the `A_z` problem of one frame (rotated magnet remanence).
#[allow(clippy::too_many_arguments)]
fn frame_problem(
    motor: &MotorMesh,
    degree: PolynomialDegree,
    winding: &Winding,
    materials: &MotorMaterials,
    magnet_angles_rad: &[f64],
    coils: &CoilData,
    currents_a: &[f64; 3],
    rotor_angle_rad: f64,
    stack_length_m: f64,
    eddy: Option<&EddyFrame>,
    magnet_scale: &[f64],
) -> Result<AzProblem, TransientError> {
    let mut material_map: BTreeMap<i32, AzMaterial> = BTreeMap::new();
    material_map.insert(MARKER_STATOR, materials.steel.clone());
    material_map.insert(MARKER_ROTOR, materials.steel.clone());
    material_map.insert(
        MARKER_SHAFT,
        AzMaterial::linear_mu_r(materials.shaft_mu_r)?
            .with_conductivity(materials.shaft_sigma_s_per_m)?,
    );
    material_map.insert(MARKER_AIR, AzMaterial::air());
    for wire in 0..motor.coil_slot_of_wire.len() {
        let coil_material = match eddy {
            Some(eddy) => AzMaterial::air().with_conductivity(eddy.copper_sigma_s_per_m)?,
            None => AzMaterial::air(),
        };
        material_map.insert(MARKER_COIL_BASE + wire as i32, coil_material);
    }
    for (index, (&polarity, &base_angle)) in materials
        .magnet_polarity
        .iter()
        .zip(magnet_angles_rad)
        .enumerate()
    {
        // Tangential spoke magnetization, CCW positive, rotated with
        // the rotor: phi-hat at the magnet's current centroid angle.
        let phi = base_angle + rotor_angle_rad;
        let br = [
            -polarity * materials.magnet_br_t * phi.sin(),
            polarity * materials.magnet_br_t * phi.cos(),
        ];
        material_map.insert(
            MARKER_MAGNET_BASE + index as i32,
            AzMaterial::magnet(MagnetLaw::new(materials.magnet_mu_rec, br)?)?
                .with_conductivity(materials.magnet_sigma_s_per_m)?,
        );
    }
    let mut current_density: BTreeMap<i32, f64> = BTreeMap::new();
    let mut conductor_blocks: Vec<ConductorBlock> = Vec::new();
    if eddy.is_some() {
        for (wire, &slot) in motor.coil_slot_of_wire.iter().enumerate() {
            let layout = winding.layout[slot];
            conductor_blocks.push(ConductorBlock {
                markers: vec![MARKER_COIL_BASE + wire as i32],
                target_current_a: f64::from(layout.direction) * currents_a[layout.phase.index()],
            });
        }
        for index in 0..materials.magnet_polarity.len() {
            conductor_blocks.push(ConductorBlock {
                markers: vec![MARKER_MAGNET_BASE + index as i32],
                target_current_a: 0.0,
            });
        }
        conductor_blocks.push(ConductorBlock {
            markers: vec![MARKER_SHAFT],
            target_current_a: 0.0,
        });
    } else {
        for (wire, &slot) in motor.coil_slot_of_wire.iter().enumerate() {
            current_density.insert(
                MARKER_COIL_BASE + wire as i32,
                winding.slot_current_density_a_per_m2(slot, currents_a, coils.slot_areas_m2[slot]),
            );
        }
    }
    Ok(AzProblem {
        mesh: motor.mesh.clone(),
        materials: material_map,
        current_density_a_per_m2: current_density,
        current_density_fn: None,
        degree,
        quadrature_order: 4,
        stack_length_m,
        conductor_blocks,
        element_magnet_scale: magnet_scale.to_vec(),
    })
}

/// Failure of a motor transient run.
#[derive(Debug, thiserror::Error)]
pub enum TransientError {
    /// Step plan does not land on ring pitches.
    #[error(
        "steps_per_period {steps_per_period} is incommensurate with ring \
         ({ring_pitches} pitches, {pole_pairs} pole pairs)"
    )]
    IncommensurateSteps {
        /// Ring pitches per revolution.
        ring_pitches: usize,
        /// Pole pairs.
        pole_pairs: usize,
        /// Requested frames per electrical period.
        steps_per_period: usize,
    },
    /// Material data was invalid.
    #[error("invalid motor material: {0}")]
    Material(String),
    /// The field solve failed.
    #[error(transparent)]
    Az(#[from] AzError),
    /// A material record was invalid.
    #[error(transparent)]
    AzMaterial(#[from] super::materials::AzMaterialError),
    /// The winding was invalid.
    #[error(transparent)]
    Winding(#[from] WindingError),
    /// Torque post-processing failed.
    #[error(transparent)]
    Torque(#[from] TorqueError),
    /// Iron-loss integration failed.
    #[error(transparent)]
    Loss(#[from] LossError),
}

/// Configuration of an eddy-current drive run.
#[derive(Debug, Clone)]
pub struct EddyTransientConfig {
    /// Backward-Euler steps per electrical period.
    pub steps_per_period: usize,
    /// Settling periods marched before the reported period.
    pub settle_periods: usize,
    /// Stack length in metres.
    pub stack_length_m: f64,
    /// Rotor outer radius in metres (Arkkio inner radius).
    pub rotor_outer_m: f64,
    /// Stator bore radius in metres (Arkkio outer radius).
    pub stator_inner_m: f64,
    /// Per-frame field solver controls.
    pub solve: AzSolveConfig,
}

/// Result of an eddy-current drive run (reported period only).
#[derive(Debug, Clone)]
pub struct EddyResult {
    /// Solved frames of the reported period.
    pub frames: Vec<TransientFrame>,
    /// Terminal-work mean torque in N m.
    pub torque_terminal_work_mean_nm: f64,
    /// Mean/ripple summary of the Arkkio series.
    pub torque_arkkio_metrics: TorqueMetrics,
    /// Mean solid-conductor copper eddy loss in watts (AC part only;
    /// the DC `I^2 R` share is [`super::losses::copper_loss_dc`]).
    pub eddy_loss_cu_w: f64,
    /// Mean magnet eddy loss in watts.
    pub eddy_loss_magnet_w: f64,
    /// Mean shaft eddy loss in watts.
    pub eddy_loss_shaft_w: f64,
    /// Stator iron loss over the reported period.
    pub stator_iron_loss: IronLossReport,
    /// Rotor iron loss over the reported period (open-window, see docs).
    pub rotor_iron_loss: IronLossReport,
    /// Electrical frequency in Hz.
    pub f_elec_hz: f64,
}

/// Copper conductivity at temperature, from the 20 °C reference value.
fn copper_sigma_at(temp_c: f64) -> f64 {
    super::losses::RHO_CU_20.recip() / (1.0 + super::losses::ALPHA_CU * (temp_c - 20.0))
}

/// Runs a current-drive eddy transient with settling.
///
/// Every step is a backward-Euler `A_z` solve with per-wire current
/// constraints and floating magnet/shaft blocks, warm-started from the
/// previous step (potential and `E0` fields). Only the final period is
/// reported. The step plan must be commensurate with the ring, as in
/// [`run_current_drive_transient`].
///
/// # Errors
///
/// Returns [`TransientError`] for an incommensurate step plan, invalid
/// winding/material data, or a failed step solve.
pub fn run_eddy_current_drive(
    motor: &mut MotorMesh,
    degree: PolynomialDegree,
    winding: &Winding,
    materials: &MotorMaterials,
    drive: &DrivePoint,
    config: &EddyTransientConfig,
) -> Result<EddyResult, TransientError> {
    winding.validate()?;
    let ring_pitches = motor.ring_nodes.len()
        / match degree {
            PolynomialDegree::P2 => 2,
            _ => 1,
        };
    let period_pitches = ring_pitches / winding.pole_pairs;
    if config.steps_per_period == 0
        || !ring_pitches.is_multiple_of(winding.pole_pairs)
        || !period_pitches.is_multiple_of(config.steps_per_period)
    {
        return Err(TransientError::IncommensurateSteps {
            ring_pitches,
            pole_pairs: winding.pole_pairs,
            steps_per_period: config.steps_per_period,
        });
    }
    let step_pitches = (period_pitches / config.steps_per_period) as i64;
    let coils = coil_data(motor, winding.layout.len());
    motor.set_rotor_shift(0);
    let magnet_angles = magnet_centroid_angles(motor, materials.magnet_polarity.len())?;
    let dirichlet = outer_dirichlet(&motor.mesh);
    let eddy = EddyFrame {
        copper_sigma_s_per_m: copper_sigma_at(drive.coil_temp_c),
    };
    let i_peak_branch = drive.i_phase_rms_a * std::f64::consts::SQRT_2 / winding.n_parallel;
    let f_elec = drive.rpm * winding.pole_pairs as f64 / 60.0;
    let dt = if f_elec > 0.0 {
        1.0 / (f_elec * config.steps_per_period as f64)
    } else {
        1.0
    };
    let total_steps = config.steps_per_period * (config.settle_periods + 1);
    let num_dofs = motor.mesh.num_nodes();
    let (stator_elements, stator_areas) = element_groups(&motor.mesh, MARKER_STATOR);
    let (rotor_elements, rotor_areas) = element_groups(&motor.mesh, MARKER_ROTOR);
    let num_elements = motor.mesh.num_elements();
    let reported = config.steps_per_period;
    let mut stator_bx = vec![0.0; reported * stator_elements.len()];
    let mut stator_by = vec![0.0; reported * stator_elements.len()];
    let mut rotor_bx = vec![0.0; reported * rotor_elements.len()];
    let mut rotor_by = vec![0.0; reported * rotor_elements.len()];

    let mut previous = vec![0.0; num_dofs];
    let mut previous_e0: Vec<f64> = Vec::new();
    let mut frames: Vec<TransientFrame> = Vec::with_capacity(reported);
    let mut torque_arkkio: Vec<f64> = Vec::with_capacity(reported);
    let (mut loss_cu, mut loss_mag, mut loss_shaft) = (0.0, 0.0, 0.0);
    for step_index in 0..total_steps {
        let shift = step_index as i64 * step_pitches;
        motor.set_rotor_shift(shift.rem_euclid(ring_pitches as i64));
        let rotor_angle_deg = motor.shift_angle_deg(shift);
        let currents = winding.phase_currents_a(
            rotor_angle_deg,
            drive.gamma_deg,
            drive.daxis_deg,
            i_peak_branch,
        );
        let problem = frame_problem(
            motor,
            degree,
            winding,
            materials,
            &magnet_angles,
            &coils,
            &currents,
            rotor_angle_deg.to_radians(),
            config.stack_length_m,
            Some(&eddy),
            &[],
        )?;
        let step = TransientStep {
            previous_potential: previous.clone(),
            dt_s: dt,
            conductor_e0_v_per_m: previous_e0.clone(),
        };
        let solution = solve_az(
            &problem,
            &dirichlet,
            &config.solve,
            Some(&previous),
            Some(&step),
        )?;
        previous_e0 = solution.conductor_e0_v_per_m.clone();
        previous = solution.vector_potential_wb_per_m.clone();
        let reported_index = step_index + reported;
        if reported_index < total_steps {
            continue;
        }
        let frame = reported_index - total_steps;
        let slot_means: Vec<f64> = coils
            .wires_of_slot
            .iter()
            .map(|wires| slot_mean_a(&solution.quadrature_state, wires))
            .collect();
        let psi = winding.phase_flux_linkages_wb_turn(&slot_means, config.stack_length_m)?;
        torque_arkkio.push(arkkio_torque_nm(
            &solution.quadrature_state,
            config.rotor_outer_m,
            config.stator_inner_m,
            config.stack_length_m,
        ));
        // Solid-conductor losses by group: integral J^2/sigma dA * L.
        for point in &solution.quadrature_state {
            let group_sigma = if point.marker >= MARKER_COIL_BASE
                && point.marker < MARKER_COIL_BASE + motor.coil_slot_of_wire.len() as i32
            {
                Some((&mut loss_cu, eddy.copper_sigma_s_per_m))
            } else if point.marker >= MARKER_MAGNET_BASE
                && point.marker < MARKER_MAGNET_BASE + materials.magnet_polarity.len() as i32
            {
                Some((&mut loss_mag, materials.magnet_sigma_s_per_m))
            } else if point.marker == MARKER_SHAFT {
                Some((&mut loss_shaft, materials.shaft_sigma_s_per_m))
            } else {
                None
            };
            if let Some((acc, sigma)) = group_sigma
                && sigma > 0.0
            {
                *acc += point.eddy_current_density_a_per_m2.powi(2) / sigma
                    * point.integration_weight_m2
                    * config.stack_length_m
                    / reported as f64;
            }
        }
        let mean_b = element_mean_b(&solution.quadrature_state, num_elements);
        for (slot, &element) in stator_elements.iter().enumerate() {
            stator_bx[frame * stator_elements.len() + slot] = mean_b[element][0];
            stator_by[frame * stator_elements.len() + slot] = mean_b[element][1];
        }
        let theta = rotor_angle_deg.to_radians();
        let (cos_t, sin_t) = (theta.cos(), theta.sin());
        for (slot, &element) in rotor_elements.iter().enumerate() {
            let [bx, by] = mean_b[element];
            rotor_bx[frame * rotor_elements.len() + slot] = cos_t * bx + sin_t * by;
            rotor_by[frame * rotor_elements.len() + slot] = -sin_t * bx + cos_t * by;
        }
        frames.push(TransientFrame {
            rotor_angle_deg,
            currents_a: currents,
            psi_wb_turn: psi,
            vector_potential_wb_per_m: solution.vector_potential_wb_per_m,
            quadrature_state: solution.quadrature_state,
        });
    }
    motor.set_rotor_shift(0);
    let psi_series: [Vec<f64>; 3] = [
        frames.iter().map(|f| f.psi_wb_turn[0]).collect(),
        frames.iter().map(|f| f.psi_wb_turn[1]).collect(),
        frames.iter().map(|f| f.psi_wb_turn[2]).collect(),
    ];
    let current_series: [Vec<f64>; 3] = [
        frames.iter().map(|f| f.currents_a[0]).collect(),
        frames.iter().map(|f| f.currents_a[1]).collect(),
        frames.iter().map(|f| f.currents_a[2]).collect(),
    ];
    let mech_angles_rad: Vec<f64> = frames
        .iter()
        .map(|frame| frame.rotor_angle_deg.to_radians())
        .collect();
    // One reported period is not special: unwrap angles for the DFT
    // derivative's integer-period check.
    let torque_terminal = terminal_work_mean_nm(
        &[&psi_series[0], &psi_series[1], &psi_series[2]],
        &[&current_series[0], &current_series[1], &current_series[2]],
        &mech_angles_rad,
        winding.pole_pairs,
        winding.n_parallel,
    )?;
    let stator_iron_loss = iron_loss(
        &stator_bx,
        &stator_by,
        &stator_areas,
        &materials.steel_loss,
        config.stack_length_m,
        f_elec,
        1.0,
        dt,
    )?;
    let rotor_iron_loss = iron_loss(
        &rotor_bx,
        &rotor_by,
        &rotor_areas,
        &materials.steel_loss,
        config.stack_length_m,
        f_elec,
        1.0,
        dt,
    )?;
    Ok(EddyResult {
        torque_arkkio_metrics: torque_metrics(&torque_arkkio),
        frames,
        torque_terminal_work_mean_nm: torque_terminal,
        eddy_loss_cu_w: loss_cu,
        eddy_loss_magnet_w: loss_mag,
        eddy_loss_shaft_w: loss_shaft,
        stator_iron_loss,
        rotor_iron_loss,
        f_elec_hz: f_elec,
    })
}

/// Magnet centroid angles (rotor frame, shift zero) by magnet index.
fn magnet_centroid_angles(
    motor: &MotorMesh,
    num_magnets: usize,
) -> Result<Vec<f64>, TransientError> {
    let mut angles = Vec::with_capacity(num_magnets);
    for index in 0..num_magnets {
        let marker = MARKER_MAGNET_BASE + index as i32;
        let (mut cx, mut cy, mut area) = (0.0, 0.0, 0.0);
        for element in motor.mesh.elements.iter().filter(|e| e.marker == marker) {
            let a = motor.mesh.nodes[element.nodes[0]];
            let b = motor.mesh.nodes[element.nodes[1]];
            let c = motor.mesh.nodes[element.nodes[2]];
            let element_area = 0.5 * ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)).abs();
            cx += (a.x + b.x + c.x) / 3.0 * element_area;
            cy += (a.y + b.y + c.y) / 3.0 * element_area;
            area += element_area;
        }
        if area <= 0.0 {
            return Err(TransientError::Material(format!(
                "magnet {index} has no elements in the mesh"
            )));
        }
        angles.push((cy / area).atan2(cx / area));
    }
    Ok(angles)
}

/// `A_z = 0` DOFs on the outermost mesh radius.
fn outer_dirichlet(mesh: &Mesh) -> Vec<usize> {
    let outer_radius = mesh
        .nodes
        .iter()
        .map(|node| node.x.hypot(node.y))
        .fold(0.0, f64::max);
    mesh.nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| (node.x.hypot(node.y) - outer_radius).abs() < 1.0e-6)
        .map(|(index, _)| index)
        .collect()
}

/// Operating point of a sinusoidal voltage-drive run.
#[derive(Debug, Clone)]
pub struct VoltageDrivePoint {
    /// Imposed phase-voltage peak in volts.
    pub v_phase_peak_v: f64,
    /// Voltage angle from the d axis in electrical degrees.
    pub v_delta_deg: f64,
    /// D-axis calibration offset in electrical degrees.
    pub daxis_deg: f64,
    /// Rotor speed in rpm.
    pub rpm: f64,
    /// Phase resistance in ohms, in per-branch circuit quantities
    /// (the reference passes the terminal `R_phase`; at one parallel
    /// branch the two coincide).
    pub r_phase_ohm: f64,
    /// Config current in A rms per terminal phase, used only to seed
    /// the circuit state before the first step.
    pub i_config_rms_a: f64,
    /// Applied waveform: plain sinusoid or two-level PWM.
    pub waveform: VoltageWaveform,
}

/// Voltage waveform applied by a voltage-drive run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VoltageWaveform {
    /// Imposed sinusoidal phase voltages.
    Sine,
    /// Ideal two-level inverter: centre-aligned sine-triangle PWM
    /// with asymmetric regular sampling (leading edge from the
    /// carrier-boundary sample, trailing from the mid-carrier
    /// sample), synchronous with the electrical period. The
    /// modulation reference is `2 * v_phase_peak / v_bus`
    /// UNCOMPENSATED: the applied fundamental carries the sampled-
    /// reference gain `sinc(pi / (2 N_c))` and half-carrier delay
    /// of a real DSP modulator (the reference application solves a
    /// compensated reference so its APPLIED fundamental equals the
    /// request; that compensation layer is not replicated).
    Pwm {
        /// DC-link voltage in volts.
        v_bus_v: f64,
        /// Carrier periods per electrical period.
        carriers_per_period: usize,
    },
}

/// Configuration of a voltage-drive run.
#[derive(Debug, Clone)]
pub struct VoltageTransientConfig {
    /// Steps per electrical period.
    pub steps_per_period: usize,
    /// Maximum settling periods marched before the reported period.
    pub settle_periods: usize,
    /// Stop settling once the period-boundary flux linkages repeat
    /// within this relative tolerance (0 disables the early stop).
    pub settle_tol_rel: f64,
    /// Stack length in metres.
    pub stack_length_m: f64,
    /// Rotor outer radius in metres (Arkkio inner radius).
    pub rotor_outer_m: f64,
    /// Stator bore radius in metres (Arkkio outer radius).
    pub stator_inner_m: f64,
    /// Per-frame field solver controls.
    pub solve: AzSolveConfig,
    /// Circuit residual convergence threshold in volts.
    pub circuit_tol_v: f64,
    /// Maximum circuit (Broyden) iterations per step.
    pub max_circuit_iterations: usize,
}

/// Line-to-line Crank-Nicolson circuit residual `[V_AB, V_BC]`.
///
/// The isolated neutral kills the zero-sequence EMF exactly, as in
/// the reference: phase-voltage equations would short the triplen
/// back-EMF through the zero-sequence inductance.
fn circuit_residual_ll(
    psi: &[f64; 3],
    currents: &[f64; 3],
    psi_prev: &[f64; 3],
    currents_prev: &[f64; 3],
    voltages: &[f64; 3],
    dt_s: f64,
    r_phase_ohm: f64,
) -> [f64; 2] {
    [
        (voltages[0] - voltages[1])
            - 0.5 * r_phase_ohm * (currents[0] - currents[1] + currents_prev[0] - currents_prev[1])
            - ((psi[0] - psi[1]) - (psi_prev[0] - psi_prev[1])) / dt_s,
        (voltages[1] - voltages[2])
            - 0.5 * r_phase_ohm * (currents[1] - currents[2] + currents_prev[1] - currents_prev[2])
            - ((psi[1] - psi[2]) - (psi_prev[1] - psi_prev[2])) / dt_s,
    ]
}

/// Balanced three-phase voltages at a mechanical rotor angle.
fn phase_voltages_v(
    point: &VoltageDrivePoint,
    pole_pairs: usize,
    rotor_angle_deg: f64,
) -> [f64; 3] {
    let te =
        (rotor_angle_deg * pole_pairs as f64 + point.v_delta_deg + point.daxis_deg).to_radians();
    let step = 2.0 * std::f64::consts::PI / 3.0;
    [
        point.v_phase_peak_v * te.cos(),
        point.v_phase_peak_v * (te - step).cos(),
        point.v_phase_peak_v * (te + step).cos(),
    ]
}

/// Duty commanded from a reference sample (sine modulation).
fn pwm_duty(point: &VoltageDrivePoint, m: f64, psi_sample_deg: f64, shift_deg: f64) -> f64 {
    let x = psi_sample_deg + point.v_delta_deg + point.daxis_deg;
    let reference = m * (x + shift_deg).to_radians().cos();
    (0.5 * (1.0 + reference)).clamp(0.0, 1.0)
}

/// `(rising, falling)` edges of one carrier's pulse, electrical degrees.
fn pwm_pulse(
    point: &VoltageDrivePoint,
    m: f64,
    carriers: usize,
    carrier_index: i64,
    shift_deg: f64,
) -> (f64, f64) {
    let d_ang = 360.0 / carriers as f64;
    let j0 = carrier_index as f64 * d_ang;
    let centre = j0 + 0.5 * d_ang;
    let d_lead = pwm_duty(point, m, j0, shift_deg);
    let d_trail = pwm_duty(point, m, centre, shift_deg);
    (
        centre - 0.5 * d_lead * d_ang,
        centre + 0.5 * d_trail * d_ang,
    )
}

/// Instantaneous pole voltage of one leg, in volts.
fn pwm_pole(
    point: &VoltageDrivePoint,
    m: f64,
    v_bus_v: f64,
    carriers: usize,
    psi_e_deg: f64,
    shift_deg: f64,
) -> f64 {
    let d_ang = 360.0 / carriers as f64;
    let j = (psi_e_deg / d_ang).floor() as i64;
    let (lo, hi) = pwm_pulse(point, m, carriers, j, shift_deg);
    if lo <= psi_e_deg && psi_e_deg < hi {
        0.5 * v_bus_v
    } else {
        -0.5 * v_bus_v
    }
}

/// Exact mean pole voltage over an electrical-angle interval.
///
/// The Crank-Nicolson circuit row multiplies V by dt, so the
/// interval mean is the quantity it integrates; a midpoint sample
/// of a square wave would be aliasing, not physics.
fn pwm_pole_mean(
    point: &VoltageDrivePoint,
    m: f64,
    v_bus_v: f64,
    carriers: usize,
    psi_a: f64,
    psi_b: f64,
    shift_deg: f64,
) -> f64 {
    if psi_b <= psi_a {
        return pwm_pole(point, m, v_bus_v, carriers, psi_a, shift_deg);
    }
    let d_ang = 360.0 / carriers as f64;
    let j0 = (psi_a / d_ang).floor() as i64;
    let j1 = ((psi_b - 1.0e-12) / d_ang).floor() as i64;
    let mut acc = 0.0;
    for j in j0..=j1 {
        let a = psi_a.max(j as f64 * d_ang);
        let b = psi_b.min((j + 1) as f64 * d_ang);
        if b <= a {
            continue;
        }
        let (on_lo, on_hi) = pwm_pulse(point, m, carriers, j, shift_deg);
        let on = (b.min(on_hi) - a.max(on_lo)).max(0.0);
        acc += 0.5 * v_bus_v * (2.0 * on - (b - a));
    }
    acc / (psi_b - psi_a)
}

/// Mean applied phase voltages over one step's rotor motion.
///
/// Sine drive: the midpoint sample (the Crank-Nicolson rule).
/// PWM: the exact volt-seconds of the chopped pole voltages; the
/// carrier is locked to the rotor electrical angle (the d-axis
/// offset enters the modulating reference, not the timebase).
pub fn step_mean_voltages_v(
    point: &VoltageDrivePoint,
    pole_pairs: usize,
    from_deg: f64,
    to_deg: f64,
) -> [f64; 3] {
    match point.waveform {
        VoltageWaveform::Sine => phase_voltages_v(point, pole_pairs, 0.5 * (from_deg + to_deg)),
        VoltageWaveform::Pwm {
            v_bus_v,
            carriers_per_period,
        } => {
            let m = 2.0 * point.v_phase_peak_v / v_bus_v;
            let psi_a = from_deg * pole_pairs as f64;
            let psi_b = to_deg * pole_pairs as f64;
            [0.0, -120.0, 120.0].map(|shift| {
                pwm_pole_mean(point, m, v_bus_v, carriers_per_period, psi_a, psi_b, shift)
            })
        }
    }
}

/// Electrical angle of the excitation frame at a mechanical angle.
fn excitation_angle_rad(rotor_deg: f64, pole_pairs: usize, daxis_deg: f64) -> f64 {
    (rotor_deg * pole_pairs as f64 + daxis_deg).to_radians()
}

/// Amplitude-invariant Park transform of a three-phase triple.
fn park_dq(values: &[f64; 3], theta_e: f64) -> [f64; 2] {
    let step = 2.0 * std::f64::consts::PI / 3.0;
    [
        (2.0 / 3.0)
            * (values[0] * theta_e.cos()
                + values[1] * (theta_e - step).cos()
                + values[2] * (theta_e + step).cos()),
        -(2.0 / 3.0)
            * (values[0] * theta_e.sin()
                + values[1] * (theta_e - step).sin()
                + values[2] * (theta_e + step).sin()),
    ]
}

/// Inverse Park transform (zero-sequence free).
fn inverse_park_dq(dq: [f64; 2], theta_e: f64) -> [f64; 3] {
    let step = 2.0 * std::f64::consts::PI / 3.0;
    let [d, q] = dq;
    [
        d * theta_e.cos() - q * theta_e.sin(),
        d * (theta_e - step).cos() - q * (theta_e - step).sin(),
        d * (theta_e + step).cos() - q * (theta_e + step).sin(),
    ]
}

/// Line-to-line residuals as a zero-sum triple in dq coordinates.
///
/// The circuit Jacobian is stationary in dq (only saturation and
/// slot ripple move it), while in abc coordinates it rotates with
/// the rotor and a secant method relearns it every step.
fn residual_dq(residual_ll: [f64; 2], theta_e: f64) -> [f64; 2] {
    park_dq(
        &[
            residual_ll[0],
            residual_ll[1],
            -residual_ll[0] - residual_ll[1],
        ],
        theta_e,
    )
}

/// Runs a sinusoidal voltage-drive transient with settling.
///
/// The phase currents are circuit unknowns: every step solves the
/// field at trial currents `(i_A, i_B)` (`i_C = -i_A - i_B`) and drives
/// the line-to-line Crank-Nicolson residual to zero by a Broyden
/// iteration (the circuit Jacobian is finite-differenced once and
/// rank-one updated). The step voltage is the midpoint sample, which
/// is the Crank-Nicolson rule. Settling stops early once the
/// period-boundary flux linkages repeat within `settle_tol_rel`.
/// Returns the reported period as a [`TransientResult`] whose frames
/// carry the solved currents.
///
/// # Errors
///
/// Returns [`TransientError`] for an incommensurate step plan, a
/// failed field solve, or a circuit iteration that does not converge.
#[allow(clippy::too_many_lines)]
pub fn run_voltage_drive_transient(
    motor: &mut MotorMesh,
    degree: PolynomialDegree,
    winding: &Winding,
    materials: &MotorMaterials,
    point: &VoltageDrivePoint,
    config: &VoltageTransientConfig,
) -> Result<TransientResult, TransientError> {
    winding.validate()?;
    let ring_pitches = motor.ring_nodes.len()
        / match degree {
            PolynomialDegree::P2 => 2,
            _ => 1,
        };
    let period_pitches = ring_pitches / winding.pole_pairs;
    if config.steps_per_period == 0
        || !ring_pitches.is_multiple_of(winding.pole_pairs)
        || !period_pitches.is_multiple_of(config.steps_per_period)
    {
        return Err(TransientError::IncommensurateSteps {
            ring_pitches,
            pole_pairs: winding.pole_pairs,
            steps_per_period: config.steps_per_period,
        });
    }
    let step_pitches = (period_pitches / config.steps_per_period) as i64;
    let coils = coil_data(motor, winding.layout.len());
    if coils.slot_areas_m2.iter().any(|area| *area <= 0.0) {
        return Err(TransientError::Material(
            "a winding slot has no copper area in the mesh".to_string(),
        ));
    }
    motor.set_rotor_shift(0);
    let magnet_angles = magnet_centroid_angles(motor, materials.magnet_polarity.len())?;
    let dirichlet = outer_dirichlet(&motor.mesh);
    let f_elec = point.rpm * winding.pole_pairs as f64 / 60.0;
    let dt = 1.0 / (f_elec * config.steps_per_period as f64);
    let num_elements = motor.mesh.num_elements();

    // One frame solve at given shift/currents; returns (psi, solution).
    type FrameSolve = ([f64; 3], Vec<f64>, Vec<AzQuadraturePoint>);
    let mut warm_start: Option<Vec<f64>> = None;
    let solve_frame = |motor: &mut MotorMesh,
                       shift: i64,
                       currents: &[f64; 3],
                       warm: &mut Option<Vec<f64>>|
     -> Result<FrameSolve, TransientError> {
        motor.set_rotor_shift(shift);
        let rotor_angle_deg = motor.shift_angle_deg(shift);
        let problem = frame_problem(
            motor,
            degree,
            winding,
            materials,
            &magnet_angles,
            &coils,
            currents,
            rotor_angle_deg.to_radians(),
            config.stack_length_m,
            None,
            &[],
        )?;
        let solution = match solve_az(&problem, &dirichlet, &config.solve, warm.as_deref(), None) {
            Ok(solution) => solution,
            Err(first) => {
                // One globalised retry: source continuation rescues a
                // trial-current overshoot that the chord Newton cannot
                // follow from the warm state. The warm potential is
                // kept — discarding it made every retry a cold solve.
                if std::env::var_os("SONIUM_VOLTAGE_TRACE").is_some() {
                    eprintln!("voltage trace: field solve retry after {first}");
                }
                let mut retry_config = config.solve.clone();
                retry_config.continuation_steps = retry_config.continuation_steps.max(4);
                solve_az(&problem, &dirichlet, &retry_config, warm.as_deref(), None)
                    .map_err(|_| first)?
            }
        };
        *warm = Some(solution.vector_potential_wb_per_m.clone());
        let slot_means: Vec<f64> = coils
            .wires_of_slot
            .iter()
            .map(|wires| slot_mean_a(&solution.quadrature_state, wires))
            .collect();
        let psi = winding.phase_flux_linkages_wb_turn(&slot_means, config.stack_length_m)?;
        Ok((
            psi,
            solution.vector_potential_wb_per_m,
            solution.quadrature_state,
        ))
    };

    // Seed state one step before the march, at the config currents.
    let i_peak_branch = point.i_config_rms_a * std::f64::consts::SQRT_2 / winding.n_parallel;
    let seed_shift = -step_pitches;
    let seed_angle = motor.shift_angle_deg(seed_shift);
    let mut i_prev = winding.phase_currents_a(seed_angle, 0.0, point.daxis_deg, i_peak_branch);
    let (mut psi_prev, _, _) = solve_frame(motor, seed_shift, &i_prev, &mut warm_start)?;

    // Circuit Jacobian by finite differences at the seed state
    // (held rotor angle), in dq coordinates; Broyden updates take
    // over from there.
    let voltages_seed = phase_voltages_v(point, winding.pole_pairs, seed_angle);
    let te_seed = excitation_angle_rad(seed_angle, winding.pole_pairs, point.daxis_deg);
    let residual_at = |psi: &[f64; 3], currents: &[f64; 3], voltages: &[f64; 3]| {
        residual_dq(
            circuit_residual_ll(
                psi,
                currents,
                &psi_prev,
                &i_prev,
                voltages,
                dt,
                point.r_phase_ohm,
            ),
            te_seed,
        )
    };
    let seed_dq = park_dq(&i_prev, te_seed);
    let mut jac = [[0.0; 2]; 2];
    {
        let base = residual_at(&psi_prev, &i_prev, &voltages_seed);
        let delta_a = 1.0;
        for (column, delta) in [([delta_a, 0.0]), ([0.0, delta_a])].iter().enumerate() {
            let trial = inverse_park_dq([seed_dq[0] + delta[0], seed_dq[1] + delta[1]], te_seed);
            let (psi_trial, _, _) = solve_frame(motor, seed_shift, &trial, &mut warm_start)?;
            let res_trial = residual_at(&psi_trial, &trial, &voltages_seed);
            jac[0][column] = (res_trial[0] - base[0]) / delta_a;
            jac[1][column] = (res_trial[1] - base[1]) / delta_a;
        }
    }

    let reported = config.steps_per_period;
    let (stator_elements, stator_areas) = element_groups(&motor.mesh, MARKER_STATOR);
    let (rotor_elements, rotor_areas) = element_groups(&motor.mesh, MARKER_ROTOR);
    let mut stator_bx = vec![0.0; reported * stator_elements.len()];
    let mut stator_by = vec![0.0; reported * stator_elements.len()];
    let mut rotor_bx = vec![0.0; reported * rotor_elements.len()];
    let mut rotor_by = vec![0.0; reported * rotor_elements.len()];
    let mut frames: Vec<TransientFrame> = Vec::with_capacity(reported);
    let mut torque_arkkio: Vec<f64> = Vec::with_capacity(reported);

    let mut boundary_psi: Option<[f64; 3]> = None;
    let mut step_index: i64 = 0;
    let mut period: usize = 0;
    loop {
        let is_reported = period > config.settle_periods
            || {
                // Decide at the period boundary whether settling is done.
                period >= 1
                    && config.settle_tol_rel > 0.0
                    && boundary_psi.is_some_and(|previous| {
                        let scale = previous
                            .iter()
                            .chain(psi_prev.iter())
                            .fold(0.0_f64, |acc, value| acc.max(value.abs()))
                            .max(1.0e-30);
                        previous
                            .iter()
                            .zip(psi_prev.iter())
                            .all(|(a, b)| (a - b).abs() <= config.settle_tol_rel * scale)
                    })
            }
            || period == config.settle_periods;
        boundary_psi = Some(psi_prev);
        for frame in 0..reported {
            let shift = step_index * step_pitches;
            let rotor_angle_deg = motor.shift_angle_deg(shift);
            let prev_angle_deg = motor.shift_angle_deg(shift - step_pitches);
            let voltages =
                step_mean_voltages_v(point, winding.pole_pairs, prev_angle_deg, rotor_angle_deg);
            // Broyden on (i_d, i_q) in the excitation frame: a
            // settled current set is a constant dq vector there, so
            // the previous step's dq is the predictor and the secant
            // Jacobian (stationary in dq) only has to polish
            // saturation and slot ripple.
            let te = excitation_angle_rad(rotor_angle_deg, winding.pole_pairs, point.daxis_deg);
            let te_prev = excitation_angle_rad(prev_angle_deg, winding.pole_pairs, point.daxis_deg);
            let mut trial_dq = park_dq(&i_prev, te_prev);
            type Accepted = ([f64; 3], [f64; 3], Vec<f64>, Vec<AzQuadraturePoint>);
            let mut accepted: Option<Accepted> = None;
            let mut previous_eval: Option<([f64; 2], [f64; 2])> = None;
            for circuit_iter in 0..config.max_circuit_iterations {
                let currents = inverse_park_dq(trial_dq, te);
                let (psi, potential, state) =
                    solve_frame(motor, shift, &currents, &mut warm_start)?;
                let residual_ll = circuit_residual_ll(
                    &psi,
                    &currents,
                    &psi_prev,
                    &i_prev,
                    &voltages,
                    dt,
                    point.r_phase_ohm,
                );
                let residual = residual_dq(residual_ll, te);
                if residual_ll[0].abs().max(residual_ll[1].abs()) < config.circuit_tol_v {
                    if std::env::var_os("SONIUM_VOLTAGE_TRACE").is_some() {
                        eprintln!(
                            "voltage trace: step {step_index} accepted at                              iteration {circuit_iter}, i = {currents:.2?}"
                        );
                    }
                    accepted = Some((currents, psi, potential, state));
                    break;
                }
                if let Some((x_old, f_old)) = previous_eval {
                    // Rank-one Broyden: J += (dF - J dx) dx^T / (dx^T dx).
                    let dx = [trial_dq[0] - x_old[0], trial_dq[1] - x_old[1]];
                    let df = [residual[0] - f_old[0], residual[1] - f_old[1]];
                    let jdx = [
                        jac[0][0] * dx[0] + jac[0][1] * dx[1],
                        jac[1][0] * dx[0] + jac[1][1] * dx[1],
                    ];
                    let denom = dx[0] * dx[0] + dx[1] * dx[1];
                    if denom > 0.0 {
                        for row in 0..2 {
                            let err = df[row] - jdx[row];
                            jac[row][0] += err * dx[0] / denom;
                            jac[row][1] += err * dx[1] / denom;
                        }
                    }
                }
                previous_eval = Some((trial_dq, residual));
                let det = jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0];
                if det.abs() < 1.0e-300 {
                    return Err(TransientError::Material(
                        "singular circuit Jacobian in voltage drive".to_string(),
                    ));
                }
                // Trust region: cap the current step so a stale secant
                // Jacobian cannot throw the field solve into a regime
                // the warm-started Newton cannot follow.
                let raw_step = [
                    (-residual[0] * jac[1][1] + residual[1] * jac[0][1]) / det,
                    (-residual[1] * jac[0][0] + residual[0] * jac[1][0]) / det,
                ];
                let step_norm = raw_step[0].abs().max(raw_step[1].abs());
                let step_scale = if step_norm > 40.0 {
                    40.0 / step_norm
                } else {
                    1.0
                };
                trial_dq = [
                    trial_dq[0] + step_scale * raw_step[0],
                    trial_dq[1] + step_scale * raw_step[1],
                ];
            }
            let Some((currents, psi, potential, state)) = accepted else {
                return Err(TransientError::Material(format!(
                    "voltage-drive circuit did not converge at step {step_index}"
                )));
            };
            i_prev = currents;
            psi_prev = psi;
            step_index += 1;
            if !is_reported {
                continue;
            }
            torque_arkkio.push(arkkio_torque_nm(
                &state,
                config.rotor_outer_m,
                config.stator_inner_m,
                config.stack_length_m,
            ));
            let mean_b = element_mean_b(&state, num_elements);
            for (slot, &element) in stator_elements.iter().enumerate() {
                stator_bx[frame * stator_elements.len() + slot] = mean_b[element][0];
                stator_by[frame * stator_elements.len() + slot] = mean_b[element][1];
            }
            let theta = rotor_angle_deg.to_radians();
            let (cos_t, sin_t) = (theta.cos(), theta.sin());
            for (slot, &element) in rotor_elements.iter().enumerate() {
                let [bx, by] = mean_b[element];
                rotor_bx[frame * rotor_elements.len() + slot] = cos_t * bx + sin_t * by;
                rotor_by[frame * rotor_elements.len() + slot] = -sin_t * bx + cos_t * by;
            }
            frames.push(TransientFrame {
                rotor_angle_deg,
                currents_a: currents,
                psi_wb_turn: psi,
                vector_potential_wb_per_m: potential,
                quadrature_state: state,
            });
        }
        period += 1;
        if is_reported {
            break;
        }
    }
    motor.set_rotor_shift(0);

    let psi_series: [Vec<f64>; 3] = [
        frames.iter().map(|f| f.psi_wb_turn[0]).collect(),
        frames.iter().map(|f| f.psi_wb_turn[1]).collect(),
        frames.iter().map(|f| f.psi_wb_turn[2]).collect(),
    ];
    let current_series: [Vec<f64>; 3] = [
        frames.iter().map(|f| f.currents_a[0]).collect(),
        frames.iter().map(|f| f.currents_a[1]).collect(),
        frames.iter().map(|f| f.currents_a[2]).collect(),
    ];
    let terminal_series: [Vec<f64>; 3] = current_series
        .iter()
        .map(|series| series.iter().map(|i| i * winding.n_parallel).collect())
        .collect::<Vec<_>>()
        .try_into()
        .expect("three phases");
    let torque_flux = flux_linkage_mean_torque_nm(
        &[&psi_series[0], &psi_series[1], &psi_series[2]],
        &[
            &terminal_series[0],
            &terminal_series[1],
            &terminal_series[2],
        ],
        winding.pole_pairs,
    )?;
    let mech_angles_rad: Vec<f64> = frames
        .iter()
        .map(|frame| frame.rotor_angle_deg.to_radians())
        .collect();
    let psi_refs: [&[f64]; 3] = [&psi_series[0], &psi_series[1], &psi_series[2]];
    let branch_refs: [&[f64]; 3] = [&current_series[0], &current_series[1], &current_series[2]];
    let torque_terminal = terminal_work_mean_nm(
        &psi_refs,
        &branch_refs,
        &mech_angles_rad,
        winding.pole_pairs,
        winding.n_parallel,
    )?;
    let torque_space_vector = space_vector_mean_nm(
        &psi_refs,
        &branch_refs,
        winding.pole_pairs,
        winding.n_parallel,
    );
    let stator_iron_loss = iron_loss(
        &stator_bx,
        &stator_by,
        &stator_areas,
        &materials.steel_loss,
        config.stack_length_m,
        f_elec,
        1.0,
        dt,
    )?;
    let rotor_iron_loss = iron_loss(
        &rotor_bx,
        &rotor_by,
        &rotor_areas,
        &materials.steel_loss,
        config.stack_length_m,
        f_elec,
        1.0,
        dt,
    )?;
    Ok(TransientResult {
        torque_arkkio_metrics: torque_metrics(&torque_arkkio),
        frames,
        torque_arkkio_nm: torque_arkkio,
        torque_flux_linkage_mean_nm: torque_flux,
        torque_terminal_work_mean_nm: torque_terminal,
        torque_space_vector_mean_nm: torque_space_vector,
        stator_iron_loss,
        rotor_iron_loss,
        f_elec_hz: f_elec,
        demag_br_mean: 1.0,
        demag_br_mean_area_weighted: 1.0,
        demag_br_min: 1.0,
        demag_factors: Vec::new(),
    })
}
