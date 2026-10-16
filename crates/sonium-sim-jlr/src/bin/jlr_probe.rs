//! Locates the first unphysical velocity in a CUDA JLR run.

// Rust guideline compliant 2026-02-21

use clap::Parser;
use sonium_sim_jlr::case::{CaseConfig, OPERATING_POINTS};
use sonium_sim_jlr::cuda::CudaLattice;
use sonium_sim_jlr::geometry::JlrGeometry;
use sonium_sim_jlr::lbm::tau_for_viscosity;

#[derive(Parser, Debug)]
#[command(about = "Probe a CUDA JLR run for the first unphysical cell")]
struct Args {
    #[arg(long, default_value_t = 4.5)]
    dx_mm: f64,
    #[arg(long, default_value_t = 3.6e-5)]
    dt_s: f64,
    #[arg(long, default_value_t = 30000)]
    steps: u64,
    #[arg(long, default_value_t = 500)]
    check_every: u64,
}

fn main() {
    let args = Args::parse();
    let cfg = CaseConfig {
        dx_mm: args.dx_mm,
        dt_s: args.dt_s,
        ..CaseConfig::default()
    };
    let geom = JlrGeometry::default();
    let dx_m = cfg.dx_mm * 1.0e-3;
    let nu_lb = cfg.nu_m2_s * cfg.dt_s / (dx_m * dx_m);
    let tau = tau_for_viscosity(nu_lb);
    let (cells, nx, ny, nz) = geom.voxelise(cfg.dx_mm);
    let op = OPERATING_POINTS[1];
    let a_feed = std::f64::consts::PI * (geom.nozzle_inlet_diameter_mm * 1.0e-3 / 2.0).powi(2);
    let v_feed_lb = (op.external_flow_m3_s / a_feed) * cfg.dt_s / dx_m;
    let mut lat = CudaLattice::with_periodic(
        nx,
        ny,
        nz,
        tau,
        cfg.smagorinsky,
        cfg.regularized,
        [0.0; 3],
        [0.0, 0.0, -v_feed_lb],
        [0, 0, 1],
        [false; 3],
        &cells,
    )
    .expect("CUDA lattice");
    lat.set_central_moment(cfg.central_moment);
    {
        let a_patch = std::f64::consts::PI * (geom.baffle_diameter_mm * 1.0e-3 / 2.0).powi(2);
        let v_out_lb = (op.external_flow_m3_s / a_patch) * cfg.dt_s / dx_m;
        lat.set_outlet_velocity([0.0, 0.0, -v_out_lb]);
    }
    let ramp_steps = cfg.ramp_time_s / cfg.dt_s;
    for step in 0..args.steps {
        let ramp = ((step as f64 + 1.0) / ramp_steps).min(1.0);
        lat.step(ramp, false).expect("step");
        if step % args.check_every == 0 || step == args.steps - 1 {
            let vel = lat.download_velocity().expect("download");
            let rho = lat.download_density().expect("density");
            let (mut rmin, mut rmax, mut rsum) = (f64::MAX, 0.0f64, 0.0f64);
            for r in &rho {
                if r.is_finite() {
                    rmin = rmin.min(*r);
                    rmax = rmax.max(*r);
                    rsum += r;
                }
            }
            println!("  rho min={rmin:.4} max={rmax:.4} mean={:.5}", rsum / rho.len() as f64);
            let mon = |x: usize, y: usize, z: usize, name: &str| {
                println!("    {name}: rho={:.4}", rho[x + nx * (y + ny * z)]);
            };
            mon(nx / 2, ny / 2, 200, "draft-tube mid");
            mon(nx / 2 + 15, ny / 2, 200, "annulus mid");
            mon(nx / 2, ny / 2, 20, "bottom space");
            mon(nx / 2 + 15, ny / 2, 340, "head annulus top");
            mon(nx / 2, ny / 2, 352, "feed pipe");
            let mut worst = (0usize, 0.0f64);
            let mut finite = true;
            let mut first_bad = 0usize;
            for (idx, u) in vel.iter().enumerate() {
                for v in u {
                    if !v.is_finite() {
                        if finite { first_bad = idx; }
                        finite = false;
                    }
                    if v.abs() > worst.1 {
                        worst = (idx, v.abs());
                    }
                }
            }
            let (x, y, z) = (
                worst.0 % nx,
                (worst.0 / nx) % ny,
                worst.0 / (nx * ny),
            );
            if !finite {
                let (bx, by, bz) = (first_bad % nx, (first_bad / nx) % ny, first_bad / (nx * ny));
                println!("first non-finite at ({bx},{by},{bz}) mm=({:.0},{:.0},{:.0})",
                    (bx as f64 + 0.5) * cfg.dx_mm - nx as f64 * cfg.dx_mm / 2.0,
                    (by as f64 + 0.5) * cfg.dx_mm - ny as f64 * cfg.dx_mm / 2.0,
                    bz as f64 * cfg.dx_mm);
            }
            println!(
                "step {step} finite={finite} max|u_lb|={:.4} at ({x},{y},{z}) mm=({:.0},{:.0},{:.0})",
                worst.1,
                (x as f64 + 0.5) * cfg.dx_mm - nx as f64 * cfg.dx_mm / 2.0,
                (y as f64 + 0.5) * cfg.dx_mm - ny as f64 * cfg.dx_mm / 2.0,
                z as f64 * cfg.dx_mm
            );
            if !finite || worst.1 > 0.35 {
                println!("BLOW-UP located, axis profile (x=y=centre):");
                for z in (300..nz).step_by(2) {
                    let idx = nx / 2 + nx * (ny / 2 + ny * z);
                    println!("  z={z} mm={:.0} uz={:.4}", z as f64 * cfg.dx_mm, vel[idx][2]);
                }
                for zz in [352usize, 341, 330, 320] {
                    let mut flux = 0.0;
                    for y in 0..ny { for x in 0..nx {
                        let idx = x + nx * (y + ny * zz);
                        flux += vel[idx][2];
                    }}
                    println!("layer z={zz} net uz sum={flux:.3}");
                }
                println!("feed-pipe row at z=352:");
                for x in 10..30 {
                    let idx = x + nx * (ny / 2 + ny * 352);
                    println!("  x={x} uz={:.4}", vel[idx][2]);
                }
                println!("cone-row profile at z=341:");
                for x in 12..28 {
                    let idx = x + nx * (ny / 2 + ny * 341);
                    println!("  x={x} uz={:.4} ux={:.4}", vel[idx][2], vel[idx][0]);
                }
                return;
            }
        }
    }
    println!("completed {} steps without blow-up", args.steps);
}
