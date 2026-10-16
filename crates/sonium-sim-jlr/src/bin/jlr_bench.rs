//! MLUPS benchmark for the CPU and CUDA lattice kernels.

// Rust guideline compliant 2026-02-21

use clap::Parser;
use sonium_sim_jlr::geometry::JlrGeometry;
use sonium_sim_jlr::lbm::{Lattice, LatticeConfig, tau_for_viscosity};

#[derive(Parser, Debug)]
#[command(about = "Benchmark JLR lattice kernels (MLUPS)")]
struct Args {
    /// Lattice spacing in mm for the JLR grid.
    #[arg(long, default_value_t = 1.5)]
    dx_mm: f64,
    /// Time step in s.
    #[arg(long, default_value_t = 4.0e-6)]
    dt_s: f64,
    /// Timed steps after 20 warmup steps.
    #[arg(long, default_value_t = 200)]
    steps: usize,
    /// Use the CUDA solver (requires --features cuda).
    #[arg(long)]
    cuda: bool,
}

fn main() {
    let args = Args::parse();
    let geom = JlrGeometry::default();
    let (cells, nx, ny, nz) = geom.voxelise(args.dx_mm);
    let nu_lb = 1.0e-6 * args.dt_s / (args.dx_mm * 1.0e-3).powi(2);
    let tau = tau_for_viscosity(nu_lb);
    let n = (nx * ny * nz) as f64;
    println!("grid {nx}x{ny}x{nz} = {n:.3e} cells, tau={tau:.6}");
    if args.cuda {
        #[cfg(feature = "cuda")]
        {
            let mut lat = sonium_sim_jlr::cuda::CudaLattice::new(
                nx,
                ny,
                nz,
                tau,
                Some(0.1),
                [0.0; 3],
                [0.0, 0.0, -0.001],
                [0, 0, 1],
                &cells,
            )
            .expect("CUDA lattice");
            lat.set_central_moment(true);
            for _ in 0..20 {
                lat.step(1.0, false).expect("warmup step");
            }
            lat.synchronize().expect("sync after warmup");
            let t0 = std::time::Instant::now();
            for _ in 0..args.steps {
                lat.step(1.0, false).expect("timed step");
            }
            lat.synchronize().expect("sync after timed steps");
            let wall = t0.elapsed().as_secs_f64();
            println!(
                "CUDA f32: {:.1} MLUPS ({:.2} GB/s at 19x4B x3 buffers traffic est.) wall={wall:.2}s for {} steps",
                n * args.steps as f64 / wall / 1.0e6,
                n * args.steps as f64 * 19.0 * 4.0 * 3.0 / wall / 1.0e9,
                args.steps
            );
        }
        #[cfg(not(feature = "cuda"))]
        panic!("--cuda requires building with --features cuda");
    } else {
        let mut lcfg = LatticeConfig::new(nx, ny, nz, tau);
        lcfg.smagorinsky = Some(0.1);
        let mut lat = Lattice::new(lcfg, cells);
        for _ in 0..5 {
            lat.step(1.0);
        }
        let t0 = std::time::Instant::now();
        for _ in 0..args.steps {
            lat.step(1.0);
        }
        let wall = t0.elapsed().as_secs_f64();
        println!(
            "CPU f64 rayon: {:.2} MLUPS wall={wall:.2}s for {} steps",
            n * args.steps as f64 / wall / 1.0e6,
            args.steps
        );
    }
}
