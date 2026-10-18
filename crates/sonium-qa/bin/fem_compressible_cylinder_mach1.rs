//! Compressible flow past a cylinder — transonic (Mach 1) DG demo and gate.
//!
//! Companion to the incompressible DFG cylinder benchmark
//! (`fem_ns_flow_past_cylinder`): same body, but solved with the explicit
//! P1-DG compressible Navier--Stokes operator in conservative variables
//! `[rho, rho*u, rho*v, rho*E]` with an ideal-gas EOS.
//!
//! Setup (nondimensional, diameter = 1):
//!   rho_inf = 1, c_inf = 1, U_inf = Mach, p_inf = 1 / gamma
//!   Cylinder: centre (0, 0), radius 0.5, body-fitted triangular mesh
//!   Outer boundaries: characteristic far-field at the freestream state
//!
//! Default gate sequence on the demo mesh:
//!   1. Mach 0.1 low-Mach control: density variation and stagnation
//!      pressure must approach the incompressible/isentropic limits.
//!   2. Mach 0.4 -> 0.8 -> 1.0 continuation (state carried across stages):
//!      the 0.4 stage must stay subsonic (below the cylinder critical
//!      Mach ~0.44; at Mach 0.5 a local supersonic pocket is physical);
//!      the Mach 1 stage must show the Rayleigh stagnation ratio, a
//!      supersonic pocket at the shoulders, a bow-shock standoff ahead of
//!      the nose, and order-one drag.
//!
//! Default numerics are the sharpened configuration: P2 DG space, HLLC
//! interface flux, troubled-cell relaxation 0.5 (see `Numerics`).
//! SONIUM_CYL_DEGREE=1, SONIUM_CYL_FLUX=rusanov, and
//! SONIUM_CYL_RELAXATION=0.05 restore the legacy P1/Rusanov settings for
//! comparison.
//!   3. Mach 1.2 bracket: stronger shock, stagnation ratio against the
//!      Rayleigh pitot formula at 1.2.
//!   4. Viscous Mach 1 (Re = 100, no-slip adiabatic wall): the actual
//!      compressible Navier--Stokes operator with Newtonian stress and
//!      Fourier heat flux, not just its inviscid limit.
//!
//! Exploration overrides: SONIUM_CYL_MACHES (comma list of fresh cases),
//! SONIUM_CYL_NX, SONIUM_CYL_NY, SONIUM_CYL_T_END, SONIUM_CYL_WRITE_VTU=1,
//! SONIUM_CYL_CONVERGENCE=1 (three-mesh Mach 1 convergence, fresh starts).

// Rust guideline compliant 2026-02-21
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::generators::channel_with_cylinder_triangles;
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
const CYL_R: f64 = 0.5;

fn env_parse<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.parse::<T>().ok())
        .unwrap_or(default)
}

fn freestream(mach: f64) -> ConservativeState {
    // rho = 1, p = 1 / gamma gives c = sqrt(gamma * p / rho) = 1.
    ConservativeState::from_primitive(1.0, [mach, 0.0, 0.0], 1.0 / GAMMA, GAS)
}

/// Stagnation-to-static pressure ratio of the freestream: isentropic total
/// pressure below Mach 1, Rayleigh pitot (normal-shock total) at/above it.
fn total_pressure_ratio(mach: f64) -> f64 {
    let g = GAMMA;
    if mach < 1.0 {
        (1.0 + 0.5 * (g - 1.0) * mach * mach).powf(g / (g - 1.0))
    } else {
        // p02/p1 = (p2/p1) * (p02/p2): normal-shock static jump times the
        // isentropic total ratio at the downstream Mach number.
        let m2_sq = ((g - 1.0) * mach * mach + 2.0) / (2.0 * g * mach * mach - (g - 1.0));
        let static_jump = 1.0 + 2.0 * g / (g + 1.0) * (mach * mach - 1.0);
        static_jump * (1.0 + 0.5 * (g - 1.0) * m2_sq).powf(g / (g - 1.0))
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Wall {
    Slip,
    NoSlipAdiabatic,
}

struct CaseResult {
    mach: f64,
    steps: usize,
    retries: usize,
    final_time: f64,
    min_density: f64,
    min_pressure: f64,
    max_mach: f64,
    max_density_deviation: f64,
    stagnation_pressure_ratio: f64,
    nose_density_ratio: f64,
    shock_standoff: f64,
    drag_coefficient: f64,
}

fn pressure_drag(mesh: &sonium_fem::mesh::Mesh, state: &CompressibleDgState) -> f64 {
    // Integrate pressure over the cylinder faces (marker 4). The body
    // outward normal is approximated radially from the cylinder centre,
    // which is exact for the circular polygon in the mesh generator.
    let mut force_x = 0.0;
    for face in mesh.boundaries.iter().filter(|face| face.marker == 4) {
        let average = state.cell_average(face.element_idx);
        let pressure = average.pressure(GAS);
        let mut centroid = [0.0f64; 2];
        for &node in &face.nodes {
            centroid[0] += mesh.nodes[node].x;
            centroid[1] += mesh.nodes[node].y;
        }
        centroid[0] /= face.nodes.len() as f64;
        centroid[1] /= face.nodes.len() as f64;
        let radial = (centroid[0].powi(2) + centroid[1].powi(2)).sqrt().max(1.0e-12);
        let normal_body = [centroid[0] / radial, centroid[1] / radial];
        let (a, b) = (
            &mesh.nodes[face.nodes[0]],
            &mesh.nodes[face.nodes[face.nodes.len() - 1]],
        );
        let length = ((b.x - a.x).powi(2) + (b.y - a.y).powi(2)).sqrt();
        // Fluid traction on the body is -p * n_body: nose pressure pushes
        // the body downstream (+x), base suction pulls it downstream too.
        force_x -= pressure * normal_body[0] * length;
    }
    force_x
}

/// Density averaged from DG cell averages onto mesh nodes. The DG vertex
/// values themselves are not usable for gradients here: the default
/// troubled-cell limiter flattens element slopes in shocked regions, so
/// vertex-based gradients under-read by more than an order of magnitude.
fn nodal_density(mesh: &sonium_fem::mesh::Mesh, state: &CompressibleDgState) -> Vec<f64> {
    let mut density = vec![0.0f64; mesh.num_nodes()];
    let mut counts = vec![0.0f64; mesh.num_nodes()];
    for element in 0..mesh.num_elements() {
        let average = state.cell_average(element);
        for &node in mesh.elements[element].vertices() {
            density[node] += average.density;
            counts[node] += 1.0;
        }
    }
    for (value, count) in density.iter_mut().zip(&counts) {
        if *count > 0.0 {
            *value /= *count;
        }
    }
    density
}

/// Constant gradient of the nodal density field over one P1 triangle.
fn element_density_gradient(
    mesh: &sonium_fem::mesh::Mesh,
    element: usize,
    nodal_density: &[f64],
) -> [f64; 2] {
    let nodes = mesh.elements[element].vertices();
    let cell = [
        nodal_density[nodes[0]],
        nodal_density[nodes[1]],
        nodal_density[nodes[2]],
    ];
    let (p0, p1, p2) = (
        &mesh.nodes[nodes[0]],
        &mesh.nodes[nodes[1]],
        &mesh.nodes[nodes[2]],
    );
    let (d0, d1, d2) = (cell[0], cell[1], cell[2]);
    // f = d0 + gx (x - x0) + gy (y - y0) through the three vertex values.
    let a = p1.x - p0.x;
    let b = p2.x - p0.x;
    let c = p1.y - p0.y;
    let d = p2.y - p0.y;
    let det = a * d - b * c;
    if det.abs() < 1.0e-300 {
        return [0.0, 0.0];
    }
    [
        (d * (d1 - d0) - b * (d2 - d0)) / det,
        (a * (d2 - d0) - c * (d1 - d0)) / det,
    ]
}

/// Solver numerics selected once in `main` (env-overridable for P1/Rusanov
/// comparison runs). Defaults are the sharpened configuration: P2 space,
/// HLLC flux, troubled-cell relaxation 0.5.
#[derive(Clone, Copy)]
struct Numerics {
    degree: PolynomialDegree,
    flux: CompressibleFluxKind,
    relaxation: f64,
    troubled_threshold: f64,
    persson: f64,
}

static NUMERICS: std::sync::OnceLock<Numerics> = std::sync::OnceLock::new();

fn numerics() -> Numerics {
    NUMERICS.get().copied().unwrap_or(Numerics {
        degree: PolynomialDegree::P2,
        flux: CompressibleFluxKind::Hllc,
        relaxation: 0.5,
        troubled_threshold: 0.03,
        persson: -3.0,
    })
}

fn build_solver(
    mesh: &sonium_fem::mesh::Mesh,
    mach: f64,
    wall: Wall,
    viscosity: ViscosityModel,
) -> Result<CompressibleDgSolver<'_>, String> {
    let free = freestream(mach);
    let mut boundaries: HashMap<i32, CompressibleBoundaryCondition> = HashMap::new();
    boundaries.insert(1, CompressibleBoundaryCondition::FarField(free));
    boundaries.insert(2, CompressibleBoundaryCondition::FarField(free));
    boundaries.insert(3, CompressibleBoundaryCondition::FarField(free));
    boundaries.insert(
        4,
        match wall {
            Wall::Slip => CompressibleBoundaryCondition::SlipWall,
            Wall::NoSlipAdiabatic => CompressibleBoundaryCondition::NoSlipAdiabatic {
                wall_velocity: [0.0, 0.0, 0.0],
            },
        },
    );
    let numerics = numerics();
    let config = CompressibleDgConfig {
        gas: GAS,
        viscosity,
        boundaries,
        degree: numerics.degree,
        flux: numerics.flux,
        troubled_cell_relaxation: numerics.relaxation,
        troubled_cell_threshold: numerics.troubled_threshold,
        persson_threshold: numerics.persson,
        ..CompressibleDgConfig::default()
    };
    CompressibleDgSolver::new(mesh, config)
        .map_err(|error| format!("solver construction: {error:?}"))
}

/// Advance `state` at one Mach/wall/viscosity setting and probe the result.
fn advance(
    mesh: &sonium_fem::mesh::Mesh,
    state: &mut CompressibleDgState,
    mach: f64,
    t_end: f64,
    wall: Wall,
    viscosity: ViscosityModel,
) -> Result<CaseResult, String> {
    let solver = build_solver(mesh, mach, wall, viscosity)?;
    let mut time = 0.0;
    let mut steps = 0usize;
    let mut retries = 0usize;
    let mut min_density = f64::INFINITY;
    let mut min_pressure = f64::INFINITY;
    while time < t_end {
        let stable = solver
            .operator
            .stable_time_step(state)
            .map_err(|error| format!("stable time step at t={time:.3}: {error:?}"))?;
        let report = solver
            .step(state, stable)
            .map_err(|error| format!("DG step at t={time:.3}: {error:?}"))?;
        time += report.accepted_time_step;
        steps += 1;
        retries += report.retries;
        min_density = min_density.min(report.minimum_density);
        min_pressure = min_pressure.min(report.minimum_pressure);
        if steps.is_multiple_of(400) {
            println!("  Mach {mach:.2}: t={time:.2} steps={steps} retries={retries}");
        }
    }
    Ok(probe(mesh, state, mach, steps, retries, time, min_density, min_pressure))
}

fn probe(
    mesh: &sonium_fem::mesh::Mesh,
    state: &CompressibleDgState,
    mach: f64,
    steps: usize,
    retries: usize,
    final_time: f64,
    min_density: f64,
    min_pressure: f64,
) -> CaseResult {
    let p_inf = 1.0 / GAMMA;
    let mut max_mach = 0.0f64;
    let mut max_density_deviation = 0.0f64;
    let mut stagnation_pressure_ratio = 0.0f64;
    let mut nose_density_ratio = 0.0f64;
    let mut best_nose_distance = f64::INFINITY;
    // (Shock-standoff probe below uses axial density gradients.)
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
        let sound = GAS.speed_of_sound(average.pressure(GAS), average.density);
        max_mach = max_mach.max(speed / sound);
        max_density_deviation = max_density_deviation.max((average.density - 1.0).abs());
        let nose_distance = (centroid[0] + CYL_R).powi(2) + centroid[1].powi(2);
        if nose_distance < best_nose_distance {
            best_nose_distance = nose_distance;
            stagnation_pressure_ratio = average.pressure(GAS) / p_inf;
            nose_density_ratio = average.density;
        }
    }
    // Shock front: axial element with the peak density gradient (the
    // schlieren maximum) upstream of the nose. A fixed density threshold
    // is not usable at Mach 1: the compression ahead of the detached
    // shock is gradual, so threshold crossings move with mesh resolution
    // while the gradient peak stays at the shock.
    let mut shock_standoff = f64::NAN;
    let mut peak_gradient = 0.0f64;
    let nodal = nodal_density(mesh, state);
    for element in 0..mesh.num_elements() {
        let cell = &mesh.elements[element];
        let mut centroid = [0.0f64; 2];
        for &node in cell.vertices() {
            centroid[0] += mesh.nodes[node].x;
            centroid[1] += mesh.nodes[node].y;
        }
        centroid[0] /= cell.vertices().len() as f64;
        centroid[1] /= cell.vertices().len() as f64;
        if centroid[1].abs() < 0.1 && centroid[0] < -CYL_R {
            let gradient = element_density_gradient(mesh, element, &nodal);
            let magnitude = (gradient[0].powi(2) + gradient[1].powi(2)).sqrt();
            if magnitude > peak_gradient {
                peak_gradient = magnitude;
                shock_standoff = (-CYL_R - centroid[0]).max(0.0);
            }
        }
    }
    let drag = pressure_drag(mesh, state);
    // C_D = F_x / (0.5 * rho_inf * U_inf^2 * D), D = 1. At Mach 0.1 the
    // dynamic pressure is tiny; report the raw force instead of a ratio.
    let drag_coefficient = if mach > 0.05 {
        drag / (0.5 * mach * mach)
    } else {
        drag
    };
    CaseResult {
        mach,
        steps,
        retries,
        final_time,
        min_density,
        min_pressure,
        max_mach,
        max_density_deviation,
        stagnation_pressure_ratio,
        nose_density_ratio,
        shock_standoff,
        drag_coefficient,
    }
}

fn fresh_state(
    mesh: &sonium_fem::mesh::Mesh,
    mach: f64,
) -> Result<CompressibleDgState, String> {
    CompressibleDgState::from_function_degree(mesh, numerics().degree, |_| freestream(mach))
        .map_err(|error| format!("initial state: {error:?}"))
}

fn write_vtu(
    mesh: &sonium_fem::mesh::Mesh,
    state: &CompressibleDgState,
    mach: f64,
    nx: usize,
    ny: usize,
) -> Result<(), String> {
    use sonium_fem::io::vtk::VtuWriter;
    let mut mach_field = Vec::with_capacity(mesh.num_elements());
    let mut density_field = Vec::with_capacity(mesh.num_elements());
    let mut pressure_field = Vec::with_capacity(mesh.num_elements());
    let mut schlieren_field = Vec::with_capacity(mesh.num_elements());
    let nodal = nodal_density(mesh, state);
    for element in 0..mesh.num_elements() {
        let average = state.cell_average(element);
        let velocity = average.velocity();
        let speed = (velocity[0].powi(2) + velocity[1].powi(2)).sqrt();
        mach_field.push(speed / GAS.speed_of_sound(average.pressure(GAS), average.density));
        density_field.push(average.density);
        pressure_field.push(average.pressure(GAS));
        let gradient = element_density_gradient(mesh, element, &nodal);
        schlieren_field.push((gradient[0].powi(2) + gradient[1].powi(2)).sqrt());
    }
    let mut writer = VtuWriter::new(mesh);
    writer.add_cell_scalar("mach", &mach_field);
    writer.add_cell_scalar("density", &density_field);
    writer.add_cell_scalar("pressure", &pressure_field);
    writer.add_cell_scalar("schlieren", &schlieren_field);
    let path = binary_output_dir(env!("CARGO_PKG_NAME"), env!("CARGO_BIN_NAME"))
        .join(format!("cylinder_mach{mach:.2}_{nx}x{ny}.vtu"));
    writer
        .write(&path)
        .map_err(|error| format!("VTU write: {error}"))?;
    println!("Wrote {}", path.display());
    Ok(())
}

fn report(result: &CaseResult) {
    println!(
        "Mach {:.2}: steps={} retries={} t={:.2} min_rho={:.3e} min_p={:.3e} \
         max_mach={:.3} drho_max={:.3} p_stag/p_inf={:.3} (theory {:.3}) \
         rho_nose={:.3} standoff={:.3} C_D={:.3}",
        result.mach,
        result.steps,
        result.retries,
        result.final_time,
        result.min_density,
        result.min_pressure,
        result.max_mach,
        result.max_density_deviation,
        result.stagnation_pressure_ratio,
        total_pressure_ratio(result.mach),
        result.nose_density_ratio,
        result.shock_standoff,
        result.drag_coefficient
    );
}

/// Gate one finished stage; returns true on failure.
fn gate(result: &CaseResult, kind: &str) -> bool {
    let mut failed = false;
    let mut check = |condition: bool, message: String| {
        if !condition {
            eprintln!("GATE[{kind}]: {message}");
            failed = true;
        }
    };
    check(
        result.min_density > 0.0 && result.min_pressure > 0.0,
        format!(
            "thermodynamics lost positivity: min_rho={:.3e} min_p={:.3e}",
            result.min_density, result.min_pressure
        ),
    );
    match kind {
        "low-mach" => {
            check(
                result.max_density_deviation < 0.03,
                format!(
                    "Mach 0.1 density deviation {:.3} too large for the incompressible limit",
                    result.max_density_deviation
                ),
            );
            check(
                (0.99..=1.03).contains(&result.stagnation_pressure_ratio),
                format!(
                    "Mach 0.1 stagnation ratio {:.3} outside [0.99, 1.03]",
                    result.stagnation_pressure_ratio
                ),
            );
        }
        "subsonic" => {
            check(
                result.max_mach <= 1.0,
                format!("subsonic case went supersonic: max Mach {:.3}", result.max_mach),
            );
            let theory = total_pressure_ratio(result.mach);
            check(
                (result.stagnation_pressure_ratio - theory).abs() / theory < 0.08,
                format!(
                    "stagnation ratio {:.3} more than 8% from isentropic {:.3}",
                    result.stagnation_pressure_ratio, theory
                ),
            );
        }
        "transonic" => {
            let theory = total_pressure_ratio(result.mach);
            check(
                (result.stagnation_pressure_ratio - theory).abs() / theory < 0.18,
                format!(
                    "stagnation ratio {:.3} more than 18% from Rayleigh {:.3}",
                    result.stagnation_pressure_ratio, theory
                ),
            );
            check(
                result.max_mach > 1.05,
                format!("no supersonic pocket: max Mach {:.3}", result.max_mach),
            );
            check(
                result.shock_standoff.is_finite() && (0.0..=3.0).contains(&result.shock_standoff),
                format!("shock standoff {:.3} outside [0, 3]", result.shock_standoff),
            );
            check(
                (0.05..=2.5).contains(&result.drag_coefficient),
                format!("drag coefficient {:.3} outside [0.05, 2.5]", result.drag_coefficient),
            );
        }
        "continuation" => {
            // Continued from the Mach 0.8 state, the shoulder pocket is
            // weaker than in the impulsive fresh start (measured 1.01 vs
            // 1.25 on this mesh); gate the quantities that must match
            // either way and leave the pocket gate to the fresh case.
            let theory = total_pressure_ratio(result.mach);
            check(
                (result.stagnation_pressure_ratio - theory).abs() / theory < 0.18,
                format!(
                    "stagnation ratio {:.3} more than 18% from Rayleigh {:.3}",
                    result.stagnation_pressure_ratio, theory
                ),
            );
            check(
                result.max_mach > 0.95,
                format!("continued case lost the sonic shoulder: max Mach {:.3}", result.max_mach),
            );
            check(
                result.shock_standoff.is_finite() && (0.0..=3.0).contains(&result.shock_standoff),
                format!("shock standoff {:.3} outside [0, 3]", result.shock_standoff),
            );
            check(
                (0.05..=2.5).contains(&result.drag_coefficient),
                format!("drag coefficient {:.3} outside [0.05, 2.5]", result.drag_coefficient),
            );
        }
        "near-critical" => {
            // Between the critical Mach (~0.44) and 1 a small supersonic
            // pocket is physical; gate stagnation pressure and drag only.
            let theory = total_pressure_ratio(result.mach);
            check(
                (result.stagnation_pressure_ratio - theory).abs() / theory < 0.12,
                format!(
                    "stagnation ratio {:.3} more than 12% from isentropic {:.3}",
                    result.stagnation_pressure_ratio, theory
                ),
            );
            check(
                (-0.5..=2.5).contains(&result.drag_coefficient),
                format!("drag coefficient {:.3} outside [-0.5, 2.5]", result.drag_coefficient),
            );
        }
        "supersonic-bracket" => {
            let theory = total_pressure_ratio(result.mach);
            check(
                (result.stagnation_pressure_ratio - theory).abs() / theory < 0.20,
                format!(
                    "stagnation ratio {:.3} more than 20% from Rayleigh {:.3}",
                    result.stagnation_pressure_ratio, theory
                ),
            );
            check(
                result.max_mach > result.mach,
                format!(
                    "no shoulder acceleration beyond freestream: max Mach {:.3}",
                    result.max_mach
                ),
            );
            check(
                (0.05..=2.5).contains(&result.drag_coefficient),
                format!("drag coefficient {:.3} outside [0.05, 2.5]", result.drag_coefficient),
            );
        }
        "viscous" => {
            check(
                result.max_mach > 1.0,
                format!("viscous Mach 1 lost the supersonic pocket: {:.3}", result.max_mach),
            );
            check(
                (0.05..=3.0).contains(&result.drag_coefficient),
                format!("viscous drag {:.3} outside [0.05, 3.0]", result.drag_coefficient),
            );
        }
        _ => {}
    }
    failed
}

fn build_mesh(nx: usize, ny: usize) -> sonium_fem::mesh::Mesh {
    // Large domain: the DFG channel (blockage 24%) would choke at Mach 1.
    let mesh = channel_with_cylinder_triangles(-5.0, 12.0, -4.0, 4.0, nx, ny, 0.0, 0.0, CYL_R);
    println!(
        "mesh {nx}x{ny}: {} nodes, {} elements, {} boundary faces",
        mesh.num_nodes(),
        mesh.num_elements(),
        mesh.boundaries.len()
    );
    mesh
}

fn main() {
    let _ = NUMERICS.set(Numerics {
        degree: if env_parse::<usize>("SONIUM_CYL_DEGREE", 2) == 1 {
            PolynomialDegree::P1
        } else {
            PolynomialDegree::P2
        },
        flux: if std::env::var("SONIUM_CYL_FLUX").is_ok_and(|v| v == "rusanov") {
            CompressibleFluxKind::Rusanov
        } else {
            CompressibleFluxKind::Hllc
        },
        relaxation: env_parse("SONIUM_CYL_RELAXATION", 0.5),
        troubled_threshold: env_parse("SONIUM_CYL_TROUBLED", 0.03),
        persson: env_parse("SONIUM_CYL_PERSSON", -3.0),
    });
    println!(
        "numerics: degree={:?} flux={:?} relaxation={} troubled={}",
        numerics().degree,
        numerics().flux,
        numerics().relaxation,
        numerics().troubled_threshold
    );
    println!("persson threshold: {}", numerics().persson);
    let nx: usize = env_parse("SONIUM_CYL_NX", 68);
    let ny: usize = env_parse("SONIUM_CYL_NY", 32);
    let t_scale: f64 = env_parse("SONIUM_CYL_T_END", 6.0);
    let write_vtu_flag = std::env::var("SONIUM_CYL_WRITE_VTU").is_ok();
    let inviscid = ViscosityModel::Constant(1.0e-10);
    println!("=== Compressible cylinder (P1-DG, ideal gas gamma=1.4) ===");
    let mut failed = false;
    let run = |result: Result<CaseResult, String>, kind: &str, failed: &mut bool| -> Option<CaseResult> {
        match result {
            Ok(result) => {
                report(&result);
                if gate(&result, kind) {
                    *failed = true;
                }
                Some(result)
            }
            Err(error) => {
                eprintln!("FAILED[{kind}]: {error}");
                *failed = true;
                None
            }
        }
    };

    if std::env::var("SONIUM_CYL_CONVERGENCE").is_ok() {
        // Three-mesh Mach 1 convergence with fresh starts.
        let mut drags: Vec<f64> = Vec::new();
        for (cnx, cny) in [(34usize, 16usize), (68, 32), (102, 48)] {
            let mesh = build_mesh(cnx, cny);
            let result = fresh_state(&mesh, 1.0).and_then(|mut state| {
                advance(&mesh, &mut state, 1.0, t_scale, Wall::Slip, inviscid)
            });
            if let Some(result) = run(result, "transonic", &mut failed) {
                drags.push(result.drag_coefficient);
            }
        }
        if drags.len() == 3 {
            let first_change = (drags[1] - drags[0]).abs() / drags[0].abs();
            let second_change = (drags[2] - drags[1]).abs() / drags[1].abs();
            println!(
                "convergence: C_D = {:.3} -> {:.3} -> {:.3} (changes {:.1}%, {:.1}%)",
                drags[0],
                drags[1],
                drags[2],
                100.0 * first_change,
                100.0 * second_change
            );
            if !(second_change < first_change || second_change < 0.15) {
                eprintln!("GATE[convergence]: drag change did not decrease under refinement");
                failed = true;
            }
        }
    } else if let Ok(raw) = std::env::var("SONIUM_CYL_MACHES") {
        // Exploration mode: fresh case per listed Mach.
        let mesh = build_mesh(nx, ny);
        for mach in raw
            .split(',')
            .filter_map(|part| part.trim().parse::<f64>().ok())
        {
            let result = fresh_state(&mesh, mach).and_then(|mut state| {
                let outcome = advance(&mesh, &mut state, mach, t_scale, Wall::Slip, inviscid);
                if write_vtu_flag && outcome.is_ok() {
                    let _ = write_vtu(&mesh, &state, mach, nx, ny);
                }
                outcome
            });
            let kind = if mach < 0.2 {
                "low-mach"
            } else if mach <= 0.45 {
                "subsonic"
            } else if mach < 1.0 {
                "near-critical"
            } else if (mach - 1.0).abs() < 1.0e-9 {
                "transonic"
            } else {
                "supersonic-bracket"
            };
            run(result, kind, &mut failed);
        }
    } else {
        let mesh = build_mesh(nx, ny);
        // 1. Low-Mach control.
        let low = fresh_state(&mesh, 0.1).and_then(|mut state| {
            advance(&mesh, &mut state, 0.1, 4.0, Wall::Slip, inviscid)
        });
        run(low, "low-mach", &mut failed);
        // 2. Continuation 0.4 -> 0.8 -> 1.0 carrying the state. The
        // subsonic stage sits at Mach 0.4, below the cylinder critical
        // Mach (~0.44): at Mach 0.5 the sharpened solver produces a real
        // local supersonic pocket (measured max Mach 1.025), so a
        // must-stay-subsonic gate at 0.5 would gate against the physics.
        let continued = (|| -> Result<CompressibleDgState, String> {
            let mut state = fresh_state(&mesh, 0.4)?;
            let sub = advance(&mesh, &mut state, 0.4, t_scale, Wall::Slip, inviscid)?;
            report(&sub);
            if gate(&sub, "subsonic") {
                return Err("subsonic stage failed its gate".to_string());
            }
            let mid = advance(&mesh, &mut state, 0.8, 3.0, Wall::Slip, inviscid)?;
            report(&mid);
            Ok(state)
        })();
        match continued {
            Ok(mut state) => {
                let sonic = advance(&mesh, &mut state, 1.0, t_scale, Wall::Slip, inviscid);
                run(sonic, "continuation", &mut failed);
            }
            Err(error) => {
                eprintln!("FAILED[continuation]: {error}");
                failed = true;
            }
        }
        // 2b. Fresh impulsive Mach 1 start: primary transonic gate.
        let fresh_sonic = fresh_state(&mesh, 1.0).and_then(|mut state| {
            let outcome = advance(&mesh, &mut state, 1.0, t_scale, Wall::Slip, inviscid);
            if write_vtu_flag && outcome.is_ok() {
                let _ = write_vtu(&mesh, &state, 1.0, nx, ny);
            }
            outcome
        });
        run(fresh_sonic, "transonic", &mut failed);
        // 3. Mach 1.2 bracket, fresh.
        let bracket = fresh_state(&mesh, 1.2).and_then(|mut state| {
            advance(&mesh, &mut state, 1.2, 5.0, Wall::Slip, inviscid)
        });
        run(bracket, "supersonic-bracket", &mut failed);
        // 4. Viscous compressible Navier--Stokes at Mach 1, Re = 100:
        // mu = rho * U * D / Re with rho = U = D = 1.
        let viscous = fresh_state(&mesh, 1.0).and_then(|mut state| {
            advance(
                &mesh,
                &mut state,
                1.0,
                3.0,
                Wall::NoSlipAdiabatic,
                ViscosityModel::Constant(0.01),
            )
        });
        run(viscous, "viscous", &mut failed);
    }

    if failed {
        eprintln!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
