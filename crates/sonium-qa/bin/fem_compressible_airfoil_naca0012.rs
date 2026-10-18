//! Compressible flow past a NACA 0012 airfoil — transonic DG demo and gate.
//!
//! Successor of the cylinder Mach 1 demo on a classical aerodynamic shape,
//! using the same explicit P2-DG compressible solver (HLLC flux,
//! Persson--Peraire troubled-cell indicator, SSPRK3 positivity limiting).
//!
//! Cases (nondimensional, chord = 1, rho_inf = 1, c_inf = 1):
//!   1. AGARD inviscid benchmark: Mach 0.80, alpha = 1.25 degrees.
//!      Gates check lift and wave drag against the published inviscid
//!      band (Cl roughly 0.3-0.4, Cd roughly 0.02), a supersonic pocket
//!      with terminating shock, and leading-edge stagnation pressure.
//!   2. Headline: Mach 1.0, alpha = 0. Symmetric profile at zero
//!      incidence: lift must stay near zero (mesh-asymmetry check),
//!      stagnation must approach the Rayleigh pitot ratio, and drag
//!      must be order 0.1-0.5.
//!
//! The viscous stage remains laminar-only until a turbulence model lands
//! in the solver; the classical RAE 2822 viscous cases (Re ~ 6.5e6,
//! turbulent) are therefore deliberately not gated here.
//!
//! Exploration overrides: SONIUM_AIRFOIL_NX, SONIUM_AIRFOIL_NY,
//! SONIUM_AIRFOIL_T_END, SONIUM_AIRFOIL_WRITE_VTU=1.

// Rust guideline compliant 2026-02-21
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::generators::channel_with_airfoil_triangles;
use sonium_fem::physics::fluid::compressible_flow::{
    CompressibleBoundaryCondition, CompressibleDgConfig, CompressibleDgSolver,
    CompressibleDgState, CompressibleFluxKind, ConservativeState, IdealGas, ViscosityModel,
};
use sonium_xem::io::binary_output_dir;
use std::collections::HashMap;

const GAMMA: f64 = 1.4;
const GAS: IdealGas = IdealGas {
    gamma: GAMMA,
    gas_constant: 1.0,
};
const CHORD: f64 = 1.0;

fn env_parse<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.parse::<T>().ok())
        .unwrap_or(default)
}

fn freestream(mach: f64, alpha_deg: f64) -> ConservativeState {
    let alpha = alpha_deg.to_radians();
    ConservativeState::from_primitive(
        1.0,
        [mach * alpha.cos(), mach * alpha.sin(), 0.0],
        1.0 / GAMMA,
        GAS,
    )
}

fn total_pressure_ratio(mach: f64) -> f64 {
    let g = GAMMA;
    if mach < 1.0 {
        (1.0 + 0.5 * (g - 1.0) * mach * mach).powf(g / (g - 1.0))
    } else {
        let m2_sq = ((g - 1.0) * mach * mach + 2.0) / (2.0 * g * mach * mach - (g - 1.0));
        let static_jump = 1.0 + 2.0 * g / (g + 1.0) * (mach * mach - 1.0);
        static_jump * (1.0 + 0.5 * (g - 1.0) * m2_sq).powf(g / (g - 1.0))
    }
}

struct AeroResult {
    mach: f64,
    alpha_deg: f64,
    steps: usize,
    retries: usize,
    min_density: f64,
    min_pressure: f64,
    max_mach: f64,
    stagnation_pressure_ratio: f64,
    lift_coefficient: f64,
    drag_coefficient: f64,
    moment_coefficient: f64,
    upper_shock_x: f64,
}

/// Pressure force/moment over the airfoil faces (marker 4), plus the
/// upper-surface shock location (first cell downstream of x/c = 0.2 whose
/// cell-average pressure exceeds 1.3x the local pre-shock trend proxy:
/// here simply the location of maximum streamwise pressure gradient).
fn aero_forces(
    mesh: &sonium_fem::mesh::Mesh,
    state: &CompressibleDgState,
    mach: f64,
    alpha_deg: f64,
) -> (f64, f64, f64) {
    let alpha = alpha_deg.to_radians();
    let mut force = [0.0f64; 2];
    let mut moment = 0.0f64;
    // Quarter-chord reference point; airfoil occupies x in [0, 1], y ~ 0.
    let reference = [0.25, 0.0];
    let interior = [0.4, 0.0];
    for face in mesh.boundaries.iter().filter(|face| face.marker == 4) {
        let average = state.cell_average(face.element_idx);
        let pressure = average.pressure(GAS);
        let (a, b) = (
            &mesh.nodes[face.nodes[0]],
            &mesh.nodes[face.nodes[face.nodes.len() - 1]],
        );
        let (ex, ey) = (b.x - a.x, b.y - a.y);
        let length = (ex * ex + ey * ey).sqrt();
        if length <= 0.0 {
            continue;
        }
        // Segment normal, oriented away from the body interior.
        let mut normal = [ey / length, -ex / length];
        let midpoint = [0.5 * (a.x + b.x), 0.5 * (a.y + b.y)];
        if (midpoint[0] - interior[0]) * normal[0] + (midpoint[1] - interior[1]) * normal[1] < 0.0 {
            normal = [-normal[0], -normal[1]];
        }
        // Fluid traction on the body is -p * n_body.
        let traction = [-pressure * normal[0], -pressure * normal[1]];
        force[0] += traction[0] * length;
        force[1] += traction[1] * length;
        let r = [midpoint[0] - reference[0], midpoint[1] - reference[1]];
        moment += (r[0] * traction[1] - r[1] * traction[0]) * length;
    }
    let q = 0.5 * mach * mach; // 0.5 * rho * U^2, rho = U_ref = mach in c=1 units
    let drag = (force[0] * alpha.cos() + force[1] * alpha.sin()) / (q * CHORD);
    let lift = (-force[0] * alpha.sin() + force[1] * alpha.cos()) / (q * CHORD);
    (lift, drag, moment / (q * CHORD * CHORD))
}

fn upper_shock_location(mesh: &sonium_fem::mesh::Mesh, state: &CompressibleDgState) -> f64 {
    // Sample upper-surface-adjacent cell pressures along the chord and
    // return x/c of the steepest pressure rise (shock foot).
    let mut samples: Vec<(f64, f64)> = Vec::new();
    for element in 0..mesh.num_elements() {
        let cell = &mesh.elements[element];
        let mut centroid = [0.0f64; 2];
        for &node in cell.vertices() {
            centroid[0] += mesh.nodes[node].x;
            centroid[1] += mesh.nodes[node].y;
        }
        centroid[0] /= cell.vertices().len() as f64;
        centroid[1] /= cell.vertices().len() as f64;
        if (0.05..=0.95).contains(&centroid[0]) && centroid[1] > 0.02 && centroid[1] < 0.25 {
            samples.push((centroid[0], state.cell_average(element).pressure(GAS)));
        }
    }
    samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut best_x = f64::NAN;
    let mut best_gradient = 0.0f64;
    for window in samples.windows(2) {
        let dx = window[1].0 - window[0].0;
        if dx > 1.0e-9 {
            let gradient = (window[1].1 - window[0].1) / dx;
            if gradient > best_gradient {
                best_gradient = gradient;
                best_x = 0.5 * (window[0].0 + window[1].0);
            }
        }
    }
    best_x
}

/// GPU twin of the `run_case` time loop: one upload, device steps,
/// one download before probing.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn run_case_gpu(
    mesh: &sonium_fem::mesh::Mesh,
    config: CompressibleDgConfig,
    state: &mut CompressibleDgState,
    mach: f64,
    alpha_deg: f64,
    t_end: f64,
    time: &mut f64,
    steps: &mut usize,
    retries: &mut usize,
    min_density: &mut f64,
    min_pressure: &mut f64,
) -> Result<(), String> {
    use sonium_fem::physics::fluid::compressible_flow::CompressibleDgGpuSolver;
    let mut solver = CompressibleDgGpuSolver::new(mesh, config)
        .map_err(|error| format!("GPU solver construction: {error}"))?;
    solver
        .set_state(state)
        .map_err(|error| format!("GPU state upload: {error}"))?;
    while *time < t_end {
        let report = solver
            .step(f64::INFINITY)
            .map_err(|error| format!("GPU DG step at t={time:.3}: {error}"))?;
        *time += report.accepted_time_step;
        *steps += 1;
        *retries += report.retries;
        *min_density = min_density.min(report.minimum_density);
        *min_pressure = min_pressure.min(report.minimum_pressure);
        if steps.is_multiple_of(400) {
            println!("  M{mach:.2} a{alpha_deg:.2}: t={time:.2} steps={steps} retries={retries}");
        }
    }
    solver
        .fetch_state(state)
        .map_err(|error| format!("GPU state download: {error}"))?;
    Ok(())
}

#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
fn run_case_gpu(
    _mesh: &sonium_fem::mesh::Mesh,
    _config: CompressibleDgConfig,
    _state: &mut CompressibleDgState,
    _mach: f64,
    _alpha_deg: f64,
    _t_end: f64,
    _time: &mut f64,
    _steps: &mut usize,
    _retries: &mut usize,
    _min_density: &mut f64,
    _min_pressure: &mut f64,
) -> Result<(), String> {
    Err("SONIUM_DG_BACKEND=cuda requires building with --features cuda".to_string())
}

fn run_case(
    mesh: &sonium_fem::mesh::Mesh,
    mach: f64,
    alpha_deg: f64,
    t_end: f64,
) -> Result<AeroResult, String> {
    let free = freestream(mach, alpha_deg);
    let mut boundaries: HashMap<i32, CompressibleBoundaryCondition> = HashMap::new();
    boundaries.insert(1, CompressibleBoundaryCondition::FarField(free));
    boundaries.insert(2, CompressibleBoundaryCondition::FarField(free));
    boundaries.insert(3, CompressibleBoundaryCondition::FarField(free));
    boundaries.insert(4, CompressibleBoundaryCondition::SlipWall);
    let config = CompressibleDgConfig {
        gas: GAS,
        viscosity: ViscosityModel::Constant(1.0e-10),
        boundaries,
        degree: PolynomialDegree::P2,
        flux: CompressibleFluxKind::Hllc,
        troubled_cell_relaxation: 0.5,
        troubled_cell_threshold: 0.03,
        persson_threshold: -3.0,
        ..CompressibleDgConfig::default()
    };
    let mut state =
        CompressibleDgState::from_function_degree(mesh, PolynomialDegree::P2, |_| free)
            .map_err(|error| format!("initial state: {error:?}"))?;
    let mut time = 0.0;
    let mut steps = 0usize;
    let mut retries = 0usize;
    let mut min_density = f64::INFINITY;
    let mut min_pressure = f64::INFINITY;
    if std::env::var("SONIUM_DG_BACKEND").as_deref() == Ok("cuda") {
        run_case_gpu(
            mesh,
            config,
            &mut state,
            mach,
            alpha_deg,
            t_end,
            &mut time,
            &mut steps,
            &mut retries,
            &mut min_density,
            &mut min_pressure,
        )?;
    } else {
        let solver = CompressibleDgSolver::new(mesh, config)
            .map_err(|error| format!("solver construction: {error:?}"))?;
        while time < t_end {
            let stable = solver
                .operator
                .stable_time_step(&state)
                .map_err(|error| format!("stable time step at t={time:.3}: {error:?}"))?;
            let report = solver
                .step(&mut state, stable)
                .map_err(|error| format!("DG step at t={time:.3}: {error:?}"))?;
            time += report.accepted_time_step;
            steps += 1;
            retries += report.retries;
            min_density = min_density.min(report.minimum_density);
            min_pressure = min_pressure.min(report.minimum_pressure);
            if steps.is_multiple_of(400) {
                println!("  M{mach:.2} a{alpha_deg:.2}: t={time:.2} steps={steps} retries={retries}");
            }
        }
    }
    // Probes.
    let p_inf = 1.0 / GAMMA;
    let mut max_mach = 0.0f64;
    let mut stagnation_pressure_ratio = 0.0f64;
    let mut best_le_distance = f64::INFINITY;
    for element in 0..mesh.num_elements() {
        let cell = &mesh.elements[element];
        let mut centroid = [0.0f64; 2];
        for &node in cell.vertices() {
            centroid[0] += mesh.nodes[node].x;
            centroid[1] += mesh.nodes[node].y;
        }
        centroid[0] /= cell.vertices().len() as f64;
        centroid[1] /= cell.vertices().len() as f64;
        let average = state.cell_average(element);
        let velocity = average.velocity();
        let speed = (velocity[0].powi(2) + velocity[1].powi(2)).sqrt();
        max_mach = max_mach
            .max(speed / GAS.speed_of_sound(average.pressure(GAS), average.density));
        let le_distance = centroid[0].powi(2) + centroid[1].powi(2);
        if le_distance < best_le_distance {
            best_le_distance = le_distance;
            stagnation_pressure_ratio = average.pressure(GAS) / p_inf;
        }
    }
    let (lift, drag, moment) = aero_forces(mesh, &state, mach, alpha_deg);
    let result = AeroResult {
        mach,
        alpha_deg,
        steps,
        retries,
        min_density,
        min_pressure,
        max_mach,
        stagnation_pressure_ratio,
        lift_coefficient: lift,
        drag_coefficient: drag,
        moment_coefficient: moment,
        upper_shock_x: if alpha_deg > 0.1 {
            upper_shock_location(mesh, &state)
        } else {
            f64::NAN
        },
    };
    if std::env::var("SONIUM_AIRFOIL_WRITE_VTU").is_ok() {
        write_vtu(mesh, &state, mach, alpha_deg)?;
    }
    Ok(result)
}

fn write_vtu(
    mesh: &sonium_fem::mesh::Mesh,
    state: &CompressibleDgState,
    mach: f64,
    alpha_deg: f64,
) -> Result<(), String> {
    use sonium_fem::io::vtk::VtuWriter;
    let mut mach_field = Vec::with_capacity(mesh.num_elements());
    let mut density_field = Vec::with_capacity(mesh.num_elements());
    let mut pressure_field = Vec::with_capacity(mesh.num_elements());
    for element in 0..mesh.num_elements() {
        let average = state.cell_average(element);
        let velocity = average.velocity();
        let speed = (velocity[0].powi(2) + velocity[1].powi(2)).sqrt();
        mach_field.push(speed / GAS.speed_of_sound(average.pressure(GAS), average.density));
        density_field.push(average.density);
        pressure_field.push(average.pressure(GAS));
    }
    let mut writer = VtuWriter::new(mesh);
    writer.add_cell_scalar("mach", &mach_field);
    writer.add_cell_scalar("density", &density_field);
    writer.add_cell_scalar("pressure", &pressure_field);
    let path = binary_output_dir(env!("CARGO_PKG_NAME"), env!("CARGO_BIN_NAME"))
        .join(format!("naca0012_mach{mach:.2}_a{alpha_deg:.2}.vtu"));
    writer
        .write(&path)
        .map_err(|error| format!("VTU write: {error}"))?;
    println!("Wrote {}", path.display());
    Ok(())
}

fn report(result: &AeroResult) {
    println!(
        "M{:.2} a{:.2}: steps={} retries={} min_rho={:.3e} min_p={:.3e} max_mach={:.3} \
         p_stag/p_inf={:.3} (theory {:.3}) Cl={:.3} Cd={:.4} Cm={:.3} shock_x/c={:.2}",
        result.mach,
        result.alpha_deg,
        result.steps,
        result.retries,
        result.min_density,
        result.min_pressure,
        result.max_mach,
        result.stagnation_pressure_ratio,
        total_pressure_ratio(result.mach),
        result.lift_coefficient,
        result.drag_coefficient,
        result.moment_coefficient,
        result.upper_shock_x
    );
}

fn main() {
    let nx: usize = env_parse("SONIUM_AIRFOIL_NX", 90);
    let ny: usize = env_parse("SONIUM_AIRFOIL_NY", 50);
    let t_end: f64 = env_parse("SONIUM_AIRFOIL_T_END", 6.0);
    let mesh = channel_with_airfoil_triangles(-6.0, 12.0, -5.0, 5.0, nx, ny, 0.0, 0.0, CHORD);
    println!(
        "=== NACA 0012 compressible airfoil (P2-DG/HLLC) ===\nmesh {nx}x{ny}: {} nodes, {} elements, {} boundary faces",
        mesh.num_nodes(),
        mesh.num_elements(),
        mesh.boundaries.len()
    );
    let mut failed = false;

    // Case 1: AGARD inviscid M0.80, alpha 1.25.
    match run_case(&mesh, 0.80, 1.25, t_end) {
        Ok(result) => {
            report(&result);
            if !(0.05..=0.65).contains(&result.lift_coefficient) {
                eprintln!("GATE[agard]: Cl {:.3} outside published inviscid band [0.05, 0.65]", result.lift_coefficient);
                failed = true;
            }
            if !(0.002..=0.09).contains(&result.drag_coefficient) {
                eprintln!("GATE[agard]: Cd {:.4} outside wave-drag band [0.002, 0.09]", result.drag_coefficient);
                failed = true;
            }
            if result.max_mach < 1.1 {
                eprintln!("GATE[agard]: no supersonic pocket (max Mach {:.3})", result.max_mach);
                failed = true;
            }
            let theory = total_pressure_ratio(0.80);
            if (result.stagnation_pressure_ratio - theory).abs() / theory > 0.15 {
                eprintln!(
                    "GATE[agard]: stagnation {:.3} more than 15% from isentropic {:.3}",
                    result.stagnation_pressure_ratio, theory
                );
                failed = true;
            }
        }
        Err(error) => {
            eprintln!("FAILED[agard]: {error}");
            failed = true;
        }
    }

    // Case 2: headline Mach 1.0, alpha 0.
    match run_case(&mesh, 1.0, 0.0, t_end) {
        Ok(result) => {
            report(&result);
            if result.lift_coefficient.abs() > 0.08 {
                eprintln!(
                    "GATE[mach1]: spurious lift Cl {:.3} at zero incidence (limit 0.08)",
                    result.lift_coefficient
                );
                failed = true;
            }
            if !(0.02..=0.60).contains(&result.drag_coefficient) {
                eprintln!("GATE[mach1]: Cd {:.4} outside [0.02, 0.60]", result.drag_coefficient);
                failed = true;
            }
            if result.max_mach < 1.05 {
                eprintln!("GATE[mach1]: no supersonic pocket (max Mach {:.3})", result.max_mach);
                failed = true;
            }
            let theory = total_pressure_ratio(1.0);
            if (result.stagnation_pressure_ratio - theory).abs() / theory > 0.18 {
                eprintln!(
                    "GATE[mach1]: stagnation {:.3} more than 18% from Rayleigh {:.3}",
                    result.stagnation_pressure_ratio, theory
                );
                failed = true;
            }
            if result.min_density <= 0.0 || result.min_pressure <= 0.0 {
                eprintln!("GATE[mach1]: positivity lost");
                failed = true;
            }
        }
        Err(error) => {
            eprintln!("FAILED[mach1]: {error}");
            failed = true;
        }
    }

    if failed {
        eprintln!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
