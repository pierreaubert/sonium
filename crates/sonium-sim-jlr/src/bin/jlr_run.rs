//! Runs one JLR operating point and writes a JSON result bundle.

// Rust guideline compliant 2026-02-21

use clap::Parser;
use sonium_sim_jlr::case::{CaseConfig, OPERATING_POINTS, run_case_cpu};
use sonium_sim_jlr::compare::compare_point;
use sonium_sim_jlr::geometry::JlrGeometry;

#[derive(Parser, Debug)]
#[command(about = "Run a jet loop reactor LBM case (Melchior et al. 2026 replication)")]
struct Args {
    /// Operating point index 0, 1, or 2 (phi = 2.243, 4.038, 5.833 kW/m3).
    #[arg(long, default_value_t = 1)]
    point: usize,
    /// Lattice spacing in mm.
    #[arg(long, default_value_t = 1.5)]
    dx_mm: f64,
    /// Physical time step in s.
    #[arg(long, default_value_t = 4.0e-6)]
    dt_s: f64,
    /// Total simulated time in s.
    #[arg(long, default_value_t = 20.0)]
    total_time_s: f64,
    /// Averaging start time in s.
    #[arg(long, default_value_t = 5.0)]
    averaging_start_s: f64,
    /// Geometry rotation in degrees.
    #[arg(long, default_value_t = 0.0)]
    rotation_deg: f64,
    /// Disable the Smagorinsky LES model.
    #[arg(long)]
    no_les: bool,
    /// Disable the regularised collision (unstable at the paper's tau).
    #[arg(long)]
    no_regularized: bool,
    /// Disable the central-moment collision (falls back to BGK family).
    #[arg(long)]
    no_central_moment: bool,
    /// Use a parabolic feed-pipe inlet profile instead of the plug.
    #[arg(long)]
    parabolic_inlet: bool,
    /// Use the CUDA solver (requires --features cuda).
    #[arg(long)]
    cuda: bool,
    /// Output JSON bundle path.
    #[arg(long, default_value = "jlr_result.json")]
    out: String,
    /// Log every N steps.
    #[arg(long, default_value_t = 0)]
    log_every: u64,
}

fn main() {
    let args = Args::parse();
    assert!(args.point < 3, "point must be 0, 1, or 2");
    let cfg = CaseConfig {
        dx_mm: args.dx_mm,
        dt_s: args.dt_s,
        total_time_s: args.total_time_s,
        averaging_start_s: args.averaging_start_s,
        rotation_deg: args.rotation_deg,
        smagorinsky: if args.no_les { None } else { Some(0.1) },
        regularized: !args.no_regularized,
        central_moment: !args.no_central_moment,
        inlet_profile: if args.parabolic_inlet {
            sonium_sim_jlr::case::InletProfile::Parabolic
        } else {
            sonium_sim_jlr::case::InletProfile::Plug
        },
        log_every_steps: args.log_every,
        ..CaseConfig::default()
    };
    let op = OPERATING_POINTS[args.point];
    let geom = JlrGeometry::default();
    let result = if args.cuda {
        #[cfg(feature = "cuda")]
        {
            sonium_sim_jlr::cuda::run_case_cuda(&cfg, op, &geom).expect("CUDA run failed")
        }
        #[cfg(not(feature = "cuda"))]
        {
            panic!("--cuda requires building with --features cuda");
        }
    } else {
        run_case_cpu(&cfg, op, &geom)
    };
    let comparison = compare_point(&result, 0.08);
    let bundle = serde_json::json!({
        "schema": "sonium.jlr-result/v1",
        "backend": if args.cuda { "cuda-f32" } else { "cpu-f64" },
        "config": cfg,
        "result": result,
        "comparison_vs_paper": comparison,
    });
    std::fs::write(&args.out, serde_json::to_string_pretty(&bundle).unwrap())
        .expect("write output bundle");
    println!(
        "phi={} kW/m3  Ncirc={:.3} (paper sim at this phi in bundle)  V_AG={:.4e} m3/s  steps={}  wall={:.1}s  MLUPS={:.1}",
        result.specific_power_kw_m3,
        result.circulation_number,
        result.annular_flow_m3_s,
        result.steps,
        result.wall_time_s,
        result.mlups
    );
    println!("comparison: {comparison:?}");
    println!("wrote {}", args.out);
}
