//! Minimum-drag shape optimisation for steady Navier–Stokes flow past a cylinder.
//!
//! Starts from the Schäfer–Turek channel cylinder and deforms its profile
//! to minimise drag at fixed cross-sectional area, following the
//! area-constrained formulation of Pironneau (1973, 1974). The fluid is
//! air (rho = 1.225 kg/m3, mu = 1.81e-5 Pa s); the Reynolds number sets
//! the mean inlet velocity. The Stokes counterpart is
//! `examples/stokes_shape_opt.rs`.
//!
//! Usage:
//!   cargo run -p sonium-fem --features cli --example ns_shape_opt
//!   cargo run -p sonium-fem --features cli --example ns_shape_opt -- --reynolds 20 --n-modes 3

// Rust guideline compliant 2026-02-21
use clap::Parser;

use sonium_fem::io::vtk::VtuWriter;
use sonium_fem::mesh::Mesh;
use sonium_fem::physics::fluid::stokes::drag_shape_opt::{
    DragShapeOptConfig, FlowEvaluation, optimise_drag, shape_radius,
};
use sonium_fem::physics::fluid::stokes::navier_stokes::compute_vorticity_2d;

#[derive(Parser)]
#[command(name = "ns-shape-opt")]
#[command(about = "Minimum-drag body shape in steady Navier-Stokes channel flow (Pironneau)")]
struct Cli {
    /// Reynolds number based on diameter and mean inlet velocity
    #[arg(long, default_value_t = 20.0)]
    reynolds: f64,
    /// Number of Fourier modes (even cosines: cos(0), cos(2t), cos(4t), ...)
    #[arg(long, default_value_t = 3)]
    n_modes: usize,
    /// Background channel mesh divisions in x
    #[arg(long, default_value_t = 64)]
    nx: usize,
    /// Background channel mesh divisions in y
    #[arg(long, default_value_t = 12)]
    ny: usize,
    /// FD step size for the drag gradient
    #[arg(long, default_value_t = 1e-4)]
    fd_eps: f64,
    /// Output directory
    #[arg(long, default_value = "viz_output")]
    output_dir: String,
}

fn write_vtu(mesh: &Mesh, flow: &FlowEvaluation, path: &str) {
    let n = mesh.num_nodes();
    let ux: Vec<f64> = (0..n).map(|i| flow.velocity[2 * i]).collect();
    let uy: Vec<f64> = (0..n).map(|i| flow.velocity[2 * i + 1]).collect();
    let uz = vec![0.0; n];
    let speed: Vec<f64> = (0..n)
        .map(|i| (ux[i] * ux[i] + uy[i] * uy[i]).sqrt())
        .collect();
    let vorticity = compute_vorticity_2d(mesh, &flow.velocity);
    let mut writer = VtuWriter::new(mesh);
    writer.add_point_vector("velocity", &ux, &uy, &uz);
    writer.add_point_scalar("pressure", &flow.pressure);
    writer.add_point_scalar("speed", &speed);
    writer.add_cell_scalar("vorticity", &vorticity);
    writer.write(path).expect("failed to write VTU");
}

fn main() {
    let cli = Cli::parse();
    let config = DragShapeOptConfig {
        reynolds: cli.reynolds,
        n_modes: cli.n_modes,
        nx: cli.nx,
        ny: cli.ny,
        fd_eps: cli.fd_eps,
        ..Default::default()
    };
    if let Err(error) = config.validate() {
        eprintln!("{error}");
        std::process::exit(1);
    }

    println!("=== Navier-Stokes Shape Optimisation (Pironneau) ===");
    println!();
    println!(
        "Channel:    {:.2} x {:.2} m, cylinder at ({:.2}, {:.2}), r = {:.3} m",
        config.channel_length,
        config.channel_height,
        config.center_x,
        config.center_y,
        config.radius
    );
    println!(
        "Air:        rho = {:.3} kg/m3, mu = {:.2e} Pa s",
        config.density, config.viscosity
    );
    println!(
        "Re = {:.1}, mean inlet velocity = {:.3e} m/s",
        config.reynolds,
        config.mean_velocity()
    );
    println!("Modes:      {} even Fourier cosines", config.n_modes);
    println!();

    let result = optimise_drag(&config).expect("shape optimisation failed");

    println!("--- Running optimisation ---");
    println!(
        "Initial drag (circle): {:.6e} N/m  (C_D = {:.4})",
        result.initial_flow.drag, result.initial_flow.drag_coefficient
    );
    println!();
    println!("=== Result ===");
    println!("Converged:  {}", result.converged);
    println!("Iterations: {}", result.optimiser_iterations);
    println!("Fun evals:  {}", result.fun_evals);
    println!();
    println!(
        "Optimal drag:   {:.6e} N/m  (C_D = {:.4})",
        result.optimal_flow.drag, result.optimal_flow.drag_coefficient
    );
    println!(
        "Optimal area:   {:.6} m2  (target: {:.6})",
        result.optimal_area, result.area_target
    );
    println!(
        "Drag reduction: {:.1}%",
        (1.0 - result.optimal_flow.drag / result.initial_flow.drag) * 100.0
    );
    println!();
    println!("Optimal shape:");
    println!(
        "  Fourier coeffs: {:?}",
        result
            .optimal_params
            .iter()
            .map(|c| format!("{c:.5}"))
            .collect::<Vec<_>>()
    );
    println!(
        "  half-length (flow dir)  = {:.4} m",
        shape_radius(0.0, &result.optimal_params)
    );
    println!(
        "  half-width  (perp)      = {:.4} m",
        shape_radius(std::f64::consts::PI / 2.0, &result.optimal_params)
    );
    println!("  aspect ratio            = {:.2}", result.aspect_ratio);
    println!();
    println!("Pironneau (1973, 1974) reference:");
    println!("  Fixed-area drag optimum is a streamlined, elongated body;");
    println!("  the Stokes limit is the ogival profile with pointed cusps.");

    std::fs::create_dir_all(&cli.output_dir).expect("failed to create output dir");
    let initial_path = format!("{}/ns_initial.vtu", cli.output_dir);
    let optimal_path = format!("{}/ns_optimal.vtu", cli.output_dir);
    write_vtu(
        &result.initial_flow.mesh,
        &result.initial_flow,
        &initial_path,
    );
    write_vtu(
        &result.optimal_flow.mesh,
        &result.optimal_flow,
        &optimal_path,
    );
    println!();
    println!("Wrote {initial_path}");
    println!("Wrote {optimal_path}");
}
