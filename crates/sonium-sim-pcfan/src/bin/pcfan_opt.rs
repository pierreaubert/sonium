use clap::{Parser, ValueEnum};
use sonium_sim_pcfan::{
    PcFanCfdFlowSolution, PcFanCfdMeshConvergence, PcFanCfdMeshConvergenceConfig, PcFanConfig,
    PcFanDesignVariables, PcFanEvaluation, PcFanFlowModel, PcFanOperatingPoint, optimize_pc_fan,
    solve_pc_fan_cfd_mean_flow, try_evaluate_pc_fan_from_cfd_solution,
    try_solve_pc_fan_cfd_mesh_convergence,
};
use sonium_xem::io::binary_output_dir;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Installation {
    /// Anechoic/free-air bench condition with clean inflow.
    Bench,
    /// Installed mid-tower chassis with disturbed inflow and cavity propagation.
    Chassis,
}

#[derive(Debug, Parser)]
#[command(about = "Optimize a 120 mm PC fan aeroacoustic design")]
struct Cli {
    /// Fan speed. The empirical 120 mm reference data span roughly
    /// 600–3300 RPM; the wider parser interval permits low-speed studies.
    #[arg(long, default_value_t = 600.0, value_parser = parse_rpm)]
    rpm: f64,
    /// Installation preset. Selects free-air/chassis propagation, inflow,
    /// and an RPM-scaled target of 20/15 CFM at 600 RPM respectively.
    #[arg(long, value_enum, default_value_t = Installation::Bench)]
    installation: Installation,
    /// Override the preset airflow target. At 600 RPM, measured-product
    /// priors are 14–24 CFM free-air and 10–20 CFM installed.
    #[arg(long, value_parser = parse_airflow_cfm)]
    target_airflow_cfm: Option<f64>,
    /// Half-width of the airflow tolerance band as a fraction of target.
    /// Penalty is zero inside `target ± band·target`, quadratic outside.
    /// Set to 0 for the legacy squared penalty.
    #[arg(long, default_value_t = 0.10)]
    airflow_band: f64,
    #[arg(long, default_value_t = 8)]
    iterations: usize,
    #[arg(long, default_value_t = 6)]
    harmonics: usize,
    /// Weight on absolute SPL (overall sound-pressure-level).
    #[arg(long = "w-spl", default_value_t = 1.0)]
    w_spl: f64,
    /// Weight on the spectral-spread penalty `(1 − spectral_flatness)`.
    /// Higher values push the optimiser toward flatter (less tonal) spectra.
    #[arg(long = "w-spectral-spread", default_value_t = 0.0)]
    w_spectral_spread: f64,
    /// Weight on the directivity-spread penalty (std-dev of per-observer SPL).
    /// Higher values push the optimiser toward more uniform polar patterns.
    #[arg(long = "w-directivity-spread", default_value_t = 0.0)]
    w_directivity_spread: f64,
    /// Number of azimuthal observer points in the directivity grid.
    #[arg(long = "observers-azimuth", default_value_t = 8)]
    observers_azimuth: usize,
    /// Number of elevation observer points in the directivity grid.
    #[arg(long = "observers-elevation", default_value_t = 3)]
    observers_elevation: usize,
    /// Far-field observer distance in metres.
    #[arg(long = "observer-distance", default_value_t = 1.0)]
    observer_distance_m: f64,
    /// Disable the broadband noise sources (tip-leakage, strut/motor).
    /// Defaults off so the legacy adjoint-gradient path remains valid.
    #[arg(long = "no-broadband")]
    no_broadband: bool,
    /// Override inlet turbulence intensity (RMS upwash / mean axial
    /// velocity). Preset defaults are 0 for bench and 0.10 for chassis.
    #[arg(long = "inflow-ti", value_parser = parse_inflow_ti)]
    inflow_ti: Option<f64>,
    /// Streamwise integral length scale of the inlet turbulence (m),
    /// used by the Amiet LE model. Sets the von Kármán cutoff
    /// frequency `f_c = U_local / (2π · Λ)`.
    /// Preset defaults are 5 mm for bench and 3 mm for chassis.
    #[arg(long = "inflow-length-scale-m", value_parser = parse_inflow_length_scale_m)]
    inflow_length_scale_m: Option<f64>,
    /// Override rigid-cavity modal loss factor eta (= 1/Q). The research
    /// prior is 0.08, with 0.03–0.20 required for sensitivity analysis.
    #[arg(long = "modal-loss-factor", value_parser = parse_modal_loss_factor)]
    modal_loss_factor: Option<f64>,
    /// Weight on the operating-point spread penalty (std-dev of overall
    /// SPL across `--operating-rpms`). 0 disables the multi-RPM evaluation
    /// (fast path).
    #[arg(long = "w-operating-spread", default_value_t = 0.0)]
    w_operating_spread: f64,
    /// Comma-separated list of RPMs to evaluate for the operating-point
    /// spread metric (only used when `--w-operating-spread > 0`).
    #[arg(long = "operating-rpms", value_delimiter = ',')]
    operating_rpms: Vec<f64>,
    /// Turbulence closure for the CFD Picard solver. `laminar` is the
    /// legacy behaviour; `komega` activates the algebraic k-ω SST
    /// surrogate (μ_eff = μ + ρ·ν_t).
    #[arg(long = "turbulence", default_value = "laminar")]
    turbulence: String,
    /// CFD tetrahedralisation pattern. `symmetric` (default, 24 tets per
    /// cube) preserves the cube's full octahedral symmetry so a
    /// rotationally-symmetric flow problem produces a symmetric pressure
    /// field. `kuhn` is the legacy 6-tet decomposition with a fixed
    /// `(0,0,0)→(1,1,1)` diagonal — cheaper but biases the pressure
    /// pattern along that diagonal.
    #[arg(long = "cfd-mesh", default_value = "symmetric")]
    cfd_mesh: String,
    /// Allow the shroud-radius design variables to vary. Off by default —
    /// the shroud is a fixed mechanical part, only the blade DOFs are
    /// admissible search directions.
    #[arg(long = "optimise-shroud")]
    optimise_shroud: bool,
    /// Allow camber to take negative values. Off by default — negative
    /// camber would invert the blade and reverse the airflow direction
    /// at fixed RPM.
    #[arg(long = "allow-negative-camber")]
    allow_negative_camber: bool,
    /// Minimum required gap between blade tip and shroud (m).
    #[arg(long = "shroud-clearance-min-m", default_value_t = 0.001)]
    shroud_clearance_min_m: f64,
    /// Penalty weight for tip-shroud clearance violations.
    #[arg(long = "shroud-clearance-penalty", default_value_t = 1.0e6)]
    shroud_clearance_penalty: f64,
    /// Penalty weight for blade vertices that exit the inner-frame cavity.
    #[arg(long = "frame-containment-penalty", default_value_t = 1.0e6)]
    frame_containment_penalty: f64,
    /// Penalty weight for blade-root deviation from the hub radius.
    /// The blade is mechanically attached to the hub at span_fraction
    /// = 0; this term penalises any blade panel near the root whose
    /// distance from the rotation axis differs from `hub_radius_m`.
    #[arg(long = "hub-attachment-penalty", default_value_t = 1.0e6)]
    hub_attachment_penalty: f64,
    /// Outer radius of the blade tip in metres. Smaller values give the
    /// optimiser more tangential sweep room before the shroud-clearance
    /// penalty bites; 0.054 m is the new default. Must stay below
    /// `shroud_inner_radius_m − shroud_clearance_min_m` so the unswept
    /// baseline tip is admissible.
    #[arg(long = "blade-tip-radius-m")]
    blade_tip_radius_m: Option<f64>,
    /// Use **L-SHADE** (gradient-free, global) instead of L-BFGS-B
    /// (gradient-based, local). L-SHADE is the SHADE variant of
    /// differential evolution with linear population-size reduction; it
    /// adapts mutation and crossover parameters per generation, shrinks
    /// the population from `np_init = popsize·dim` down to `np_final =
    /// 4` over the budget, and tends to find more aggressively curved
    /// blades than the gradient method. Ignores the L-BFGS-B
    /// `--iterations` flag; pass `--de-maxiter` and `--de-popsize`.
    #[arg(long)]
    de: bool,
    /// Maximum number of L-SHADE generations (only when `--de`).
    #[arg(long = "de-maxiter", default_value_t = 30)]
    de_maxiter: usize,
    /// L-SHADE initial-population multiplier (`np_init = popsize·dim`).
    /// Tanabe-Fukunaga 2014 paper recommends 18; smaller is faster but
    /// less exploratory, larger is slower but more global.
    #[arg(long = "de-popsize", default_value_t = 15)]
    de_popsize: usize,
    /// L-SHADE random seed for reproducibility (only when `--de`). When
    /// `--de-experiments > 1`, this is the *base* seed: experiment `i`
    /// uses `de_seed + i`.
    #[arg(long = "de-seed", default_value_t = 42)]
    de_seed: u64,
    /// Run `N` independent DE experiments in parallel, each with its
    /// own seed and its own rayon thread pool. Returns the best result
    /// across all experiments. Default 1 (single DE, legacy behaviour).
    /// Combine with `--de-cores-per` to size each experiment's pool —
    /// e.g. `--de-experiments 16 --de-cores-per 8` saturates a 128-core
    /// machine.
    #[arg(long = "de-experiments", default_value_t = 1)]
    de_experiments: usize,
    /// Threads per DE experiment when `--de-experiments > 1`. Each
    /// experiment runs on a private rayon pool of this size. When omitted,
    /// the resolved total thread budget is divided across the experiments.
    #[arg(long = "de-cores-per")]
    de_cores_per: Option<usize>,
    /// Run a standalone backend-aware GMRES smoke test on a synthetic
    /// 1-D Poisson problem of size `--gmres-smoke-n` and exit. Used to
    /// validate the `sonium-compute` Backend wiring on a given target
    /// (`cpu`, `cuda`, or `metal`) without involving the Picard loop.
    /// This is the first verification step for the GPU-offloaded CFD path;
    /// it validates the backend before involving the Picard loop.
    #[arg(long = "gmres-smoke")]
    gmres_smoke: Option<String>,
    /// Problem size for `--gmres-smoke`. Default 4096.
    #[arg(long = "gmres-smoke-n", default_value_t = 4096)]
    gmres_smoke_n: usize,
    /// Compute backend used for the per-Picard linear solve when `--cfd`
    /// is active. `auto` (default) picks the best backend for the host:
    /// Metal on macOS with `--features metal`, CUDA on Linux/Windows with
    /// `--features cuda`, CPU otherwise. `cpu` forces the legacy ILU(0) +
    /// GMRES path; `cuda` and `metal` force those backends explicitly and
    /// fall back to CPU with a warning when the feature isn't built or
    /// the device fails to initialise.
    #[arg(long = "cfd-backend", default_value = "auto")]
    cfd_backend: String,
    /// GPU-friendly preconditioner kind. `polynomial-jacobi` (default)
    /// uses the truncated Neumann series of `D⁻¹A` — robust on indefinite
    /// Stokes blocks with appropriate damping. `chebyshev` uses the
    /// optimal min-max polynomial on the eigenvalue interval — stronger
    /// on SPD/near-SPD systems. Both are pure SpMV + BLAS-1 (no host
    /// round-trip, no triangular solve). Ignored on the CPU backend.
    #[arg(long = "cfd-precond", default_value = "polynomial-jacobi")]
    cfd_precond: String,
    /// Polynomial Jacobi preconditioner degree on the GPU/backend path.
    /// `1` (default) is plain point-Jacobi; `2..4` gives a stronger
    /// preconditioner at the cost of `K-1` extra SpMVs per GMRES iter.
    /// Used when `--cfd-precond polynomial-jacobi`. Ignored on CPU.
    #[arg(long = "cfd-precond-degree", default_value_t = 1)]
    cfd_precond_degree: usize,
    /// Damping factor `ω` for the polynomial Jacobi preconditioner.
    /// `1.0` (default) is plain Jacobi-Richardson; `2/3` is the classical
    /// weighted-Jacobi smoother for diffusive problems. `0.5..0.8` for
    /// indefinite Stokes blocks. Used when `--cfd-precond polynomial-jacobi`.
    #[arg(long = "cfd-precond-damping", default_value_t = 1.0)]
    cfd_precond_damping: f64,
    /// Pressure Schur polynomial degree for `--cfd-precond jacobi-schur`.
    /// `1` keeps the stable diagonal Schur inverse; `2..4` adds sparse
    /// pressure-mass correction steps.
    #[arg(long = "cfd-schur-degree", default_value_t = 1)]
    cfd_schur_degree: usize,
    /// Pressure Schur correction damping for `--cfd-precond jacobi-schur`.
    #[arg(long = "cfd-schur-damping", default_value_t = 1.0)]
    cfd_schur_damping: f64,
    /// Pressure Schur solve inside `--cfd-precond jacobi-schur`.
    /// `polynomial` preserves the existing diagonal/polynomial path;
    /// `chebyshev` applies a backend Chebyshev solve on the pressure
    /// Schur block; `amg` applies a backend AMG V-cycle with Chebyshev
    /// smoothing.
    #[arg(long = "cfd-schur-precond", default_value = "polynomial")]
    cfd_schur_precond: String,
    /// Pressure Schur Chebyshev degree for `--cfd-schur-precond chebyshev`.
    #[arg(long = "cfd-schur-cheb-degree", default_value_t = 5)]
    cfd_schur_cheb_degree: usize,
    /// Pressure Schur Chebyshev eigenvalue-interval ratio.
    #[arg(long = "cfd-schur-cheb-eigenvalue-ratio", default_value_t = 30.0)]
    cfd_schur_cheb_eigenvalue_ratio: f64,
    /// Chebyshev preconditioner degree (number of 3-term-recurrence
    /// steps). `5` is the PETSc/HYPRE default. Used when
    /// `--cfd-precond chebyshev`.
    #[arg(long = "cfd-cheb-degree", default_value_t = 5)]
    cfd_cheb_degree: usize,
    /// Chebyshev eigenvalue-interval ratio `λ_max / λ_min`. `30.0`
    /// (default) matches PETSc/HYPRE. Used when `--cfd-precond chebyshev`.
    #[arg(long = "cfd-cheb-eigenvalue-ratio", default_value_t = 30.0)]
    cfd_cheb_eigenvalue_ratio: f64,
    /// Working precision for the per-Picard linear solve.
    /// `f64` (default) is the legacy path. `mixed-f32-f64` routes the
    /// inner Krylov through f32 (FGMRES-IR, Carson-Higham 2018) — the
    /// only way to actually accelerate the linear solve on Apple Silicon
    /// Metal, since the Metal compiler rejects `double`. The outer
    /// FGMRES still runs in f64, so f64 accuracy is preserved.
    #[arg(long = "cfd-solver", default_value = "f64")]
    cfd_solver: String,
    /// Maximum inner f32 GMRES iterations per outer FGMRES-IR step.
    /// Used only when `--cfd-solver mixed-f32-f64`.
    #[arg(long = "cfd-ir-inner-max", default_value_t = 50)]
    cfd_ir_inner_max: usize,
    /// Inner f32 GMRES relative-residual tolerance.
    /// Used only when `--cfd-solver mixed-f32-f64`.
    #[arg(long = "cfd-ir-inner-tol", default_value_t = 1.0e-3)]
    cfd_ir_inner_tol: f32,
    #[arg(long)]
    cfd: bool,
    /// Activate the full-CFD model: blade is treated as a real rotating
    /// wall (Brinkman penalty) rather than a distributed actuator-disk
    /// body force. Significantly more accurate but ~2× the per-iteration
    /// cost; the default inlet/outlet/side margins of the CFD enclosure
    /// are auto-enlarged so the no-slip outer walls don't choke the flow.
    /// Implies `--cfd`.
    #[arg(long = "cfd-full")]
    cfd_full: bool,
    #[arg(
        long,
        help = "Solve the default shape once with CFD and skip optimization"
    )]
    solve_only: bool,
    #[arg(
        long,
        help = "Run a CFD mesh-refinement sequence for the default shape"
    )]
    mesh_converged: bool,
    /// Run a Pareto sweep over `--w-spectral-spread ∈ {0, 0.5, 1, 2, 5}`,
    /// printing a comparison table of the converged design at each weight.
    /// Skips the rest of the run when set.
    #[arg(long = "pareto-sweep")]
    pareto_sweep: bool,
    #[arg(long, default_value_t = 2)]
    cfd_cells_x: usize,
    #[arg(long, default_value_t = 2)]
    cfd_cells_y: usize,
    #[arg(long, default_value_t = 2)]
    cfd_cells_z: usize,
    #[arg(long, default_value_t = 80)]
    cfd_picard_iterations: usize,
    #[arg(long, default_value_t = 2.0e-2)]
    cfd_tolerance: f64,
    /// Maximum GMRES iterations for each CFD Picard linear solve.
    /// GPU backends use weaker SpMV-only preconditioners than the CPU
    /// ILU path, so they often need hundreds or thousands of iterations
    /// on the indefinite Stokes block.
    #[arg(long = "cfd-linear-iterations", default_value_t = 5000)]
    cfd_linear_iterations: usize,
    /// Relative residual tolerance for each CFD Picard linear solve.
    #[arg(long = "cfd-linear-tolerance", default_value_t = 1.0e-4)]
    cfd_linear_tolerance: f64,
    /// CPU preconditioner for CFD linear solves and CPU fallback.
    /// `ilu` is the legacy monolithic ILU path; `jacobi-schur` uses
    /// diagonal velocity-block Jacobi plus pressure-mass Schur.
    #[arg(long = "cfd-cpu-precond", default_value = "ilu")]
    cfd_cpu_precond: String,
    /// Solve each CFD Picard linear system with the sparse-direct (LU)
    /// solver instead of preconditioned GMRES. Exact on the small
    /// conforming-mesh systems (2^3-6^3 cells); use it where monolithic
    /// ILU(0) stalls on the coupled saddle-point block.
    #[arg(long = "cfd-direct-solver")]
    cfd_direct_solver: bool,
    /// Finite-difference step (relative) used for the gradient when the
    /// CFD flow model is active. Default `0.05` (5 %): with the per-
    /// parameter scales in `finite_difference_step_for_variable`, this
    /// gives FD probes of ≈ 1 mm on chord, ≈ 0.2 mm on camber, ≈ 0.5°
    /// on twist — large enough that a coarse CFD grid registers a real
    /// objective change, small enough that the FD remains a sensible
    /// approximation of the gradient. The reduced-order path uses a
    /// much smaller step (`2e-5`) because its objective is analytical
    /// and noise-free.
    #[arg(long = "cfd-fd-step", default_value_t = 5.0e-2)]
    cfd_fd_step: f64,
    #[arg(long, default_value_t = 6)]
    mesh_start_cells: usize,
    #[arg(long, default_value_t = 8)]
    mesh_max_cells: usize,
    #[arg(long, default_value_t = 0.08)]
    mesh_convergence_tolerance: f64,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long)]
    cfd_output: Option<PathBuf>,
    /// Write a deterministic, website-ready gallery and provenance manifest.
    /// Implies the lightweight CFD path so the gallery contains a real flow
    /// field as well as the evaluated fan geometry and acoustic plots.
    #[arg(long, value_name = "DIR")]
    gallery: Option<PathBuf>,
    /// Skip the optional external `pvbatch` capture while retaining the
    /// deterministic native figures, VTU sources, and ParaView script.
    #[arg(long)]
    no_paraview: bool,
    /// Cap the Rayon thread pool to `N` threads. Zero or omission uses
    /// Rayon's automatic resolution. A positive value overrides the
    /// `RAYON_NUM_THREADS` environment variable. Useful when running
    /// multiple `pcfan-opt` processes in parallel on a many-core machine
    /// (e.g. 16 seeds × 8 threads = 128 cores).
    #[arg(long)]
    threads: Option<usize>,
    /// Print a bounded CPU execution contract as JSON and exit.
    #[arg(long)]
    execution_contract: bool,
}

fn parse_bounded_f64(value: &str, name: &str, min: f64, max: f64) -> Result<f64, String> {
    let parsed = value
        .parse::<f64>()
        .map_err(|error| format!("invalid {name} '{value}': {error}"))?;
    if !parsed.is_finite() || !(min..=max).contains(&parsed) {
        return Err(format!("{name} must be finite and in [{min}, {max}]"));
    }
    Ok(parsed)
}

fn parse_rpm(value: &str) -> Result<f64, String> {
    parse_bounded_f64(value, "RPM", 200.0, 4_000.0)
}

fn parse_airflow_cfm(value: &str) -> Result<f64, String> {
    parse_bounded_f64(value, "airflow", 0.1, 200.0)
}

fn parse_inflow_ti(value: &str) -> Result<f64, String> {
    parse_bounded_f64(value, "inflow turbulence intensity", 0.0, 0.30)
}

fn parse_inflow_length_scale_m(value: &str) -> Result<f64, String> {
    parse_bounded_f64(value, "inflow integral length scale", 5.0e-4, 5.0e-2)
}

fn parse_modal_loss_factor(value: &str) -> Result<f64, String> {
    parse_bounded_f64(value, "modal loss factor", 0.03, 0.20)
}

fn main() {
    let cli = Cli::parse();

    if let Err(err) = sonium_xem::runtime::install_global_rayon_pool(
        cli.threads.filter(|&n| n > 0).unwrap_or(0),
        None,
    ) {
        eprintln!("warning: failed to configure rayon thread pool: {err}");
    }

    if cli.execution_contract {
        let evidence = sonium_xem::runtime::observe_execution_contract(
            "pcfan-opt",
            "cpu",
            "cpu",
            std::collections::BTreeMap::from([
                ("finite_difference_components".to_string(), 30),
                (
                    "de_cores_per_experiment".to_string(),
                    resolved_de_cores_per_experiment(&cli),
                ),
            ]),
            false,
            "host-cpu",
        );
        println!("{}", serde_json::to_string_pretty(&evidence).unwrap());
        return;
    }

    if let Some(backend_name) = &cli.gmres_smoke {
        if let Err(error) = run_gmres_smoke(backend_name, cli.gmres_smoke_n) {
            eprintln!("GMRES smoke test failed: {error}");
            std::process::exit(3);
        }
        return;
    }

    let config = build_config(&cli);

    if cli.pareto_sweep {
        run_pareto_sweep(config);
    } else if cli.mesh_converged {
        run_mesh_converged_shape(config, &cli);
    } else if cli.solve_only {
        run_single_shape(config, &cli);
    } else {
        run_optimization(config, &cli);
    }
}

/// Standalone backend-aware GMRES smoke test. Builds an `n × n` 1-D
/// diagonally dominant tridiagonal problem, uploads
/// the CSR + RHS to the requested backend, runs the new
/// `gmres_backend_preconditioned` with point-Jacobi preconditioning,
/// and reports iteration count + wall time.
///
/// Checks the independently recomputed host residual of the converged result.
fn run_gmres_smoke(backend_name: &str, n: usize) -> Result<(), Box<dyn std::error::Error>> {
    use sonium_compute::gmres::GmresBackendConfig;

    // This is a wiring test, not an unbounded stress benchmark. Bound both
    // host CSR storage and the 100-vector Krylov workspace before allocation.
    if !(1..=100_000).contains(&n) {
        return Err("GMRES smoke size must be between 1 and 100000".into());
    }

    println!("=== GMRES Backend Smoke Test ===");
    println!("backend: {}, problem size: {}", backend_name, n);

    // Build host-side diagonally-dominant tridiagonal CSR. We use
    // `4·I − tridiag(−1, 0, −1)` rather than the pure 1-D Poisson
    // (2·I − tridiag) because the larger diagonal makes the system far
    // better-conditioned (cond ≈ 3 vs n²/π² for Poisson). With Jacobi
    // preconditioning GMRES converges in ~10 iterations regardless of n,
    // which is the right behaviour for a wiring smoke test — we want to
    // measure the *backend transfer + BLAS* cost, not the unhelpful
    // O(n) Krylov basis growth that a poorly-conditioned smoke test
    // would inflict.
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx: Vec<i32> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    row_ptr.push(0);
    for i in 0..n {
        if i > 0 {
            col_idx.push(i as i32 - 1);
            values.push(-1.0);
        }
        col_idx.push(i as i32);
        values.push(4.0);
        if i + 1 < n {
            col_idx.push(i as i32 + 1);
            values.push(-1.0);
        }
        row_ptr.push(col_idx.len() as i32);
    }
    let b_host: Vec<f64> = vec![1.0; n];

    let cfg = GmresBackendConfig {
        max_iterations: 100,
        tolerance: 1.0e-10,
    };

    match backend_name.to_ascii_lowercase().as_str() {
        "cpu" => smoke_one_backend(
            &sonium_compute::cpu::CpuBackend::new(),
            &row_ptr,
            &col_idx,
            &values,
            &b_host,
            n,
            &cfg,
        ),
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                let backend = sonium_compute::CudaBackend::new_default()?;
                smoke_one_backend(&backend, &row_ptr, &col_idx, &values, &b_host, n, &cfg)
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(
                    "this build does not include the CUDA backend; rebuild with --features cuda"
                        .into(),
                )
            }
        }
        "metal" => {
            #[cfg(feature = "metal")]
            {
                let backend = sonium_compute::MetalBackend::new()?;
                smoke_one_backend(&backend, &row_ptr, &col_idx, &values, &b_host, n, &cfg)
            }
            #[cfg(not(feature = "metal"))]
            {
                Err(
                    "this build does not include the Metal backend; rebuild with --features metal"
                        .into(),
                )
            }
        }
        other => Err(format!("unknown backend '{other}'; use cpu | cuda | metal").into()),
    }
}

fn smoke_one_backend<B: sonium_compute::Backend>(
    backend: &B,
    row_ptr: &[i32],
    col_idx: &[i32],
    values: &[f64],
    b_host: &[f64],
    n: usize,
    cfg: &sonium_compute::gmres::GmresBackendConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    use sonium_compute::PointJacobiBackend;
    use sonium_compute::gmres::{BackendCsr, gmres_backend_preconditioned};

    let upload_t0 = std::time::Instant::now();
    if b_host.len() != n || n == 0 || b_host.iter().any(|value| !value.is_finite()) {
        return Err("invalid smoke-test RHS".into());
    }
    let a = BackendCsr::try_from_host(backend, n, n, row_ptr, col_idx, values)?;
    let mut b = backend.raw_alloc::<f64>(n)?;
    backend.try_copy_h2d(b_host, &mut b)?;
    let mut out = backend.raw_alloc::<f64>(n)?;
    let upload_ms = upload_t0.elapsed().as_secs_f64() * 1000.0;

    let precond = PointJacobiBackend::try_from_csr_host(backend, n, row_ptr, col_idx, values)?;

    let solve_t0 = std::time::Instant::now();
    let status = gmres_backend_preconditioned(backend, &a, &precond, &b, None, &mut out, cfg)?;
    let solve_ms = solve_t0.elapsed().as_secs_f64() * 1000.0;

    let mut x_host = vec![0.0_f64; n];
    backend.try_copy_d2h(&out, &mut x_host)?;

    // Verify by computing ||A x − b||₂ on the host side.
    let mut residual_sq = 0.0_f64;
    for i in 0..n {
        let lo = row_ptr[i] as usize;
        let hi = row_ptr[i + 1] as usize;
        let mut row_sum = 0.0;
        for k in lo..hi {
            let j = col_idx[k] as usize;
            row_sum += values[k] * x_host[j];
        }
        let r = row_sum - b_host[i];
        residual_sq += r * r;
    }
    let resid = residual_sq.sqrt();
    let bnorm: f64 = b_host.iter().map(|v| v * v).sum::<f64>().sqrt();
    let rel_resid = if bnorm == 0.0 { resid } else { resid / bnorm };

    println!(
        "iterations: {}, final residual: {:.3e}, converged: {}",
        status.iterations, status.final_residual, status.converged
    );
    println!("upload (H→D + alloc): {:.2} ms", upload_ms);
    println!("solve:                 {:.2} ms", solve_ms);
    println!("verified ‖A x − b‖ / ‖b‖ = {:.3e}", rel_resid);
    if !status.converged
        || !status.final_residual.is_finite()
        || !rel_resid.is_finite()
        || rel_resid > 1.0e-8
        || x_host.iter().any(|value| !value.is_finite())
    {
        return Err("nonfinite, unconverged or over-tolerance smoke result".into());
    }
    Ok(())
}

fn run_pareto_sweep(base_config: PcFanConfig) {
    reject_unchecked_cfd_optimization(&base_config);
    let weights = [0.0, 0.5, 1.0, 2.0, 5.0];
    println!("=== Pareto Sweep: spectral-spread weight ===");
    println!(
        "operating point: {:.1} rpm, target airflow {:.2} CFM, band ±{:.0}%",
        base_config.operating_point.rpm,
        base_config.operating_point.target_airflow_cfm,
        100.0 * base_config.weights.airflow_band_fraction
    );
    println!(
        "{:>6}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "w_ss", "airflow", "SPL_dB", "flatness", "dir_σ", "op_σ", "objective"
    );

    for &w in &weights {
        let mut config = base_config.clone();
        config.weights.spectral_spread_weight = w;
        let result = optimize_pc_fan(config);
        let e = &result.evaluation;
        println!(
            "{:>6.2}  {:>10.2}  {:>10.2}  {:>10.4}  {:>10.2}  {:>10.2}  {:>10.3}",
            w,
            e.airflow_cfm,
            e.acoustic_level_db,
            e.spectral_flatness,
            e.directivity_spread_db,
            e.operating_point_spread_db,
            e.objective
        );
    }
    println!("(Higher w_ss should produce flatter spectra at the cost of higher absolute SPL.)");
}

fn run_mesh_converged_shape(config: PcFanConfig, cli: &Cli) {
    let design = config.default_design_variables();
    let geometry = config.build_geometry(&design);
    let convergence = try_solve_pc_fan_cfd_mesh_convergence(
        &config,
        &design,
        PcFanCfdMeshConvergenceConfig {
            start_cells: cli.mesh_start_cells,
            max_cells: cli.mesh_max_cells,
            relative_tolerance: cli.mesh_convergence_tolerance,
            ..PcFanCfdMeshConvergenceConfig::default()
        },
    )
    .unwrap_or_else(|error| {
        eprintln!("Invalid mesh-convergence configuration: {error}");
        std::process::exit(1);
    });
    if !convergence.converged {
        eprintln!(
            "Mesh-converged CFD result was not admitted: airflow and pressure changes must both be within the configured tolerance after nonlinear convergence"
        );
        print_mesh_convergence_report(&convergence);
        std::process::exit(1);
    }
    let cfd_solution = &convergence.selected_solution;
    let evaluation = try_evaluate_pc_fan_from_cfd_solution(&config, &design, cfd_solution)
        .unwrap_or_else(|error| {
            eprintln!("Mesh-converged CFD result is not admissible for evaluation: {error}");
            std::process::exit(1);
        });
    let out_path = cli
        .output
        .clone()
        .unwrap_or_else(|| default_output_path("mesh_converged_pcfan.vtu"));
    let cfd_path = cli
        .cfd_output
        .clone()
        .unwrap_or_else(|| default_output_path("mesh_converged_pcfan_cfd.vtu"));

    if let Err(err) = geometry.write_evaluation_vtu(&evaluation, &out_path) {
        eprintln!("Failed to write VTU: {err}");
    }
    if let Err(err) = cfd_solution.write_vtu(&cfd_path) {
        eprintln!("Failed to write CFD VTU: {err}");
    }
    let pv_path = sibling_path(&out_path, "_paraview").with_extension("py");
    let png_path = sibling_path(&out_path, "_render").with_extension("png");
    if let Err(err) = write_paraview_state(&pv_path, &cfd_path, &png_path) {
        eprintln!("Failed to write ParaView script: {err}");
    }

    println!("=== PC Fan Mesh-Converged CFD Solve ===");
    println!(
        "flow model: {:?}, CFD mesh levels {}x{}x{} through {}x{}x{}, Picard max {}",
        config.cfd.flow_model,
        cli.mesh_start_cells,
        cli.mesh_start_cells,
        cli.mesh_start_cells,
        cli.mesh_max_cells,
        cli.mesh_max_cells,
        cli.mesh_max_cells,
        config.cfd.max_picard_iterations
    );
    println!(
        "operating point: {:.1} rpm, target airflow {:.2} CFM",
        config.operating_point.rpm, config.operating_point.target_airflow_cfm
    );
    print_mesh_convergence_report(&convergence);
    println!("objective: {:.3}", evaluation.objective);
    print_evaluation_report(&evaluation);
    print_cfd_report(cfd_solution);
    print_design_variables(&design);
    println!("VTU: {}", out_path.display());
    println!("CFD VTU: {}", cfd_path.display());
    println!("ParaView visualisation: {}", pv_path.display());
    println!(
        "  paraview --script={}    →    interactive 3-D streamlines/glyphs",
        pv_path.display()
    );
    println!(
        "  pvbatch --no-mpi --disable-registry --force-offscreen-rendering {}  →  {}",
        pv_path.display(),
        png_path.display()
    );
}

fn run_single_shape(config: PcFanConfig, cli: &Cli) {
    if let Some(directory) = &cli.gallery
        && let Err(err) = std::fs::create_dir_all(directory)
    {
        eprintln!("Failed to create gallery directory: {err}");
        std::process::exit(1);
    }
    let design = config.default_design_variables();
    let geometry = config.build_geometry(&design);
    let cfd_solution = solve_pc_fan_cfd_mean_flow(&config, &design);
    let evaluation = try_evaluate_pc_fan_from_cfd_solution(&config, &design, &cfd_solution)
        .unwrap_or_else(|error| {
            eprintln!("CFD result is not admissible for evaluation: {error}");
            std::process::exit(1);
        });
    let out_path = cli.output.clone().unwrap_or_else(|| {
        cli.gallery
            .as_ref()
            .map(|directory| directory.join("single_shape_pcfan.vtu"))
            .unwrap_or_else(|| default_output_path("single_shape_pcfan.vtu"))
    });
    let cfd_path = cli.cfd_output.clone().unwrap_or_else(|| {
        cli.gallery
            .as_ref()
            .map(|directory| directory.join("single_shape_pcfan_cfd.vtu"))
            .unwrap_or_else(|| default_output_path("single_shape_pcfan_cfd.vtu"))
    });

    if let Err(err) = geometry.write_evaluation_vtu(&evaluation, &out_path) {
        eprintln!("Failed to write VTU: {err}");
    }
    if let Err(err) = cfd_solution.write_vtu(&cfd_path) {
        eprintln!("Failed to write CFD VTU: {err}");
    }
    let pv_path = sibling_path(&out_path, "_paraview").with_extension("py");
    let png_path = sibling_path(&out_path, "_render").with_extension("png");
    if let Err(err) = write_paraview_state(&pv_path, &cfd_path, &png_path) {
        eprintln!("Failed to write ParaView script: {err}");
    }
    let rendered_png = if cli.gallery.is_some() && !cli.no_paraview {
        match render_paraview_gallery(&pv_path, &png_path) {
            Ok(rendered) => rendered,
            Err(err) => {
                eprintln!("Failed to render ParaView gallery: {err}");
                std::process::exit(1);
            }
        }
    } else {
        None
    };
    if let Some(gallery_dir) = &cli.gallery
        && let Err(err) = write_gallery(GalleryRequest {
            directory: gallery_dir,
            config: &config,
            optimized: &evaluation,
            baseline: &evaluation,
            design: &design,
            geometry: &geometry,
            cfd: Some(&cfd_solution),
            paraview_render: rendered_png.as_deref(),
        })
    {
        eprintln!("Failed to write gallery: {err}");
        std::process::exit(1);
    }

    println!("=== PC Fan Single-Shape CFD Solve ===");
    print_run_setup(&config);
    println!("objective: {:.3}", evaluation.objective);
    print_evaluation_report(&evaluation);
    print_cfd_report(&cfd_solution);
    print_design_variables(&design);
    println!("VTU: {}", out_path.display());
    println!("CFD VTU: {}", cfd_path.display());
    println!("ParaView visualisation: {}", pv_path.display());
    println!(
        "  paraview --script={}    →    interactive 3-D streamlines/glyphs",
        pv_path.display()
    );
    println!(
        "  pvbatch {}             →    {}",
        pv_path.display(),
        png_path.display()
    );
}

/// Run the L-SHADE multistart harness, write per-seed surface VTUs
/// alongside the canonical `--output` path, print a comparison table,
/// and return the best-objective `OptimizedPcFan` so the caller can
/// drive the rest of the post-processing pipeline against it.
fn run_multi_de(config: &PcFanConfig, cli: &Cli) -> sonium_sim_pcfan::OptimizedPcFan {
    let cores_per_experiment = resolved_de_cores_per_experiment(cli);
    let multi = sonium_sim_pcfan::optimize_pc_fan_multi_de(
        config.clone(),
        cli.de_maxiter,
        cli.de_popsize,
        cli.de_seed,
        cli.de_experiments,
        cores_per_experiment,
    );

    let out_stem = cli
        .output
        .clone()
        .unwrap_or_else(|| default_output_path("optimized_pcfan.vtu"));
    for exp in &multi.experiments {
        let path = sibling_path(&out_stem, &format!("_seed{}", exp.seed));
        if let Err(err) = exp
            .result
            .geometry
            .write_evaluation_vtu(&exp.result.evaluation, &path)
        {
            eprintln!("Failed to write per-seed VTU (seed {}): {err}", exp.seed);
        }
    }

    println!(
        "=== Multi-DE results ({} experiments × {} threads each) ===",
        cli.de_experiments, cores_per_experiment
    );
    println!(
        "       {:>6}  {:>10}  {:>10}  {:>10}  {:>10}  {:<22}",
        "seed", "fun_evals", "objective", "SPL_dB", "airflow", "status"
    );
    for (i, exp) in multi.experiments.iter().enumerate() {
        let e = &exp.result.evaluation;
        let s = &exp.result.optimization;
        let marker = if i == multi.best_index { "*" } else { " " };
        println!(
            "  {}    {:>6}  {:>10}  {:>10.3}  {:>10.2}  {:>10.2}  {:<22}",
            marker,
            exp.seed,
            s.function_evaluations,
            e.objective,
            e.acoustic_level_db,
            e.airflow_cfm,
            format!("{:?}", s.status),
        );
    }
    println!("(* = best objective, used for canonical output paths below)");

    multi.experiments[multi.best_index].result.clone()
}

fn resolved_de_cores_per_experiment(cli: &Cli) -> usize {
    let total_threads = cli
        .threads
        .filter(|&threads| threads > 0)
        .unwrap_or_else(rayon::current_num_threads);
    cli.de_cores_per
        .filter(|&threads| threads > 0)
        .unwrap_or_else(|| total_threads / cli.de_experiments.max(1))
        .max(1)
}

fn run_optimization(config: PcFanConfig, cli: &Cli) {
    reject_unchecked_cfd_optimization(&config);
    let cfd_config = config.clone();
    let baseline_design = config.default_design_variables();
    let baseline_geometry = config.build_geometry(&baseline_design);
    let result = if cli.de {
        if cli.de_experiments > 1 {
            run_multi_de(&config, cli)
        } else {
            sonium_sim_pcfan::optimize_pc_fan_de(
                config,
                cli.de_maxiter,
                cli.de_popsize,
                cli.de_seed,
            )
        }
    } else {
        optimize_pc_fan(config)
    };
    let cfd_active = cfd_config.cfd.flow_model.is_cfd();

    // Resolve output paths. `--output foo.vtu` and `--cfd-output bar.vtu`
    // remain the canonical opt-in flags; when omitted we derive a
    // family of sibling paths so a single optimisation run produces
    //   optimized_pcfan.vtu                  (optimised surface mesh + per-panel constants)
    //   optimized_pcfan_cfd.vtu              (volumetric mesh + nodal velocity/pressure)
    //   optimized_pcfan_baseline.vtu         (baseline surface for visual diff)
    //   optimized_pcfan_baseline_cfd.vtu     (baseline volumetric for visual diff)
    // Only the surface pair is written when CFD is inactive — the
    // volumetric solve would be redundant for the reduced-order path.
    let out_path = cli
        .output
        .clone()
        .unwrap_or_else(|| default_output_path("optimized_pcfan.vtu"));
    let baseline_out_path = sibling_path(&out_path, "_baseline");
    let cfd_path = cli
        .cfd_output
        .clone()
        .unwrap_or_else(|| sibling_path(&out_path, "_cfd"));
    let baseline_cfd_path = sibling_path(&out_path, "_baseline_cfd");

    if let Err(err) = result
        .geometry
        .write_evaluation_vtu(&result.evaluation, &out_path)
    {
        eprintln!("Failed to write VTU: {err}");
    }

    // Baseline surface for side-by-side comparison.
    let baseline_evaluation =
        PcFanEvaluation::from_design(&cfd_config, &baseline_design, &baseline_geometry);
    if let Err(err) =
        baseline_geometry.write_evaluation_vtu(&baseline_evaluation, &baseline_out_path)
    {
        eprintln!("Failed to write baseline VTU: {err}");
    }

    // Volumetric CFD outputs — pressure & velocity fields on the air
    // domain mesh. Auto-written whenever CFD is active so the user sees
    // proper field data without needing the (easy to forget) `--cfd-output`
    // flag.
    let cfd_output = if cfd_active {
        let solution = solve_pc_fan_cfd_mean_flow(&cfd_config, &result.design_variables);
        if let Err(err) = solution.write_vtu(&cfd_path) {
            eprintln!("Failed to write CFD VTU: {err}");
        }

        // Baseline volumetric CFD too.
        let baseline_solution = solve_pc_fan_cfd_mean_flow(&cfd_config, &baseline_design);
        if let Err(err) = baseline_solution.write_vtu(&baseline_cfd_path) {
            eprintln!("Failed to write baseline CFD VTU: {err}");
        }
        // ParaView Python state script — open the optimised CFD file
        // with streamlines + slice + glyphs configured. Works either
        // interactively (`paraview --script=<path>`) or in batch
        // (`pvbatch <path>` to render PNGs without a display).
        let pv_path = sibling_path(&out_path, "_paraview").with_extension("py");
        let png_path = sibling_path(&out_path, "_render").with_extension("png");
        if let Err(err) = write_paraview_state(&pv_path, &cfd_path, &png_path) {
            eprintln!("Failed to write ParaView script: {err}");
        }
        Some((
            cfd_path.clone(),
            solution,
            baseline_cfd_path.clone(),
            pv_path,
            png_path,
        ))
    } else {
        None
    };

    if let Some(gallery_dir) = &cli.gallery
        && let Err(err) = write_gallery(GalleryRequest {
            directory: gallery_dir,
            config: &result.config,
            optimized: &result.evaluation,
            baseline: &baseline_evaluation,
            design: &result.design_variables,
            geometry: &result.geometry,
            cfd: cfd_output.as_ref().map(|(_, solution, _, _, _)| solution),
            paraview_render: None,
        })
    {
        eprintln!("Failed to write gallery: {err}");
    }

    println!("=== PC Fan Aeroacoustic Optimization ===");
    print_run_setup(&result.config);
    println!(
        "solver: {}",
        if cli.de {
            if cli.de_experiments > 1 {
                format!(
                    "L-SHADE multistart ({} experiments × {} threads, popsize={}·dim={}, maxiter={}, base_seed={}, np_final=4)",
                    cli.de_experiments,
                    resolved_de_cores_per_experiment(cli),
                    cli.de_popsize,
                    cli.de_popsize * 30,
                    cli.de_maxiter,
                    cli.de_seed
                )
            } else {
                format!(
                    "L-SHADE (popsize={}·dim={}, maxiter={}, seed={}, np_final=4)",
                    cli.de_popsize,
                    cli.de_popsize * 30,
                    cli.de_maxiter,
                    cli.de_seed
                )
            }
        } else {
            "L-BFGS-B (gradient-based, finite-difference gradient under CFD)".into()
        }
    );
    println!(
        "status: {:?}, converged: {}, iterations: {}, fun_evals: {}, grad_evals: {}",
        result.optimization.status,
        result.optimization.converged,
        result.optimization.iterations,
        result.optimization.function_evaluations,
        result.optimization.gradient_evaluations
    );
    if result.optimization.iterations <= 1 {
        eprintln!(
            "warning: optimiser converged in ≤1 iteration — gradient is likely too small. \
             Try a larger `--cfd-fd-step` (current: {:.1e}), a finer CFD grid (current {}×{}×{}), \
             or more `--cfd-picard-iterations` (current: {}).",
            result.config.optimizer.finite_difference_step,
            result.config.cfd.cells_x,
            result.config.cfd.cells_y,
            result.config.cfd.cells_z,
            result.config.cfd.max_picard_iterations,
        );
    }
    println!(
        "objective: {:.3} (baseline {:.3})",
        result.evaluation.objective, result.optimization.baseline_objective
    );
    print_evaluation_report(&result.evaluation);
    print_design_variables(&result.design_variables);
    println!();
    println!("Output files:");
    println!(
        "  optimised surface (per-panel constants): {}",
        out_path.display()
    );
    println!(
        "  baseline surface (for diff):              {}",
        baseline_out_path.display()
    );
    if let Some((path, solution, baseline_cfd, pv_script, render_png)) = cfd_output {
        print_cfd_report(&solution);
        println!(
            "  optimised volumetric CFD (pressure+velocity fields): {}",
            path.display()
        );
        println!(
            "  baseline volumetric CFD (for diff):                  {}",
            baseline_cfd.display()
        );
        println!(
            "  ParaView visualisation script:                       {}",
            pv_script.display()
        );
        println!();
        println!("To visualise the airflow in 3D:");
        println!("  Interactive:  paraview --script={}", pv_script.display());
        println!(
            "  Batch render: pvbatch {}    →    {}",
            pv_script.display(),
            render_png.display()
        );
        println!(
            "The script preconfigures streamlines (seeded on the inlet plane), \
             a centre-plane slice coloured by speed, and a glyph layer of velocity arrows."
        );
    } else {
        println!(
            "(reduced-order flow model — no volumetric CFD VTU. Pass `--cfd` for nodal pressure/velocity fields.)"
        );
    }
}

/// The current production CFD path is a checked solve/evaluation workflow,
/// not an optimizer fidelity. Refuse CLI optimization rather than exporting a
/// design selected from an unconverged CFD objective. `--solve-only` and
/// `--mesh-converged` use the checked admission API instead.
fn reject_unchecked_cfd_optimization(config: &PcFanConfig) {
    if config.cfd.flow_model.is_cfd() {
        eprintln!(
            "CFD optimization is not admissible yet: use --solve-only or --mesh-converged; \
             a CFD optimization must provide checked nonlinear/conservation closure."
        );
        std::process::exit(2);
    }
}

/// Generate a ParaView Python state script that opens the supplied CFD
/// VTU and configures a sensible 3-D flow visualisation:
///
/// - Streamlines seeded on the inlet plane (10 × 10 grid),
/// - A horizontal slice through the fan centre coloured by `speed_m_s`,
/// - Velocity-vector glyphs subsampled across the volume,
/// - Pressure isosurface (subtle, low opacity) for context,
/// - Camera positioned 30° above and ahead of the fan axis,
/// - PNG render saved at 1920×1080 when run via `pvbatch`.
///
/// The script is plain text — no ParaView SDK is required to write it.
/// At runtime ParaView interprets it via `paraview --script=...` (GUI)
/// or `pvbatch --no-mpi --disable-registry --force-offscreen-rendering ...`
/// (offscreen render). These flags avoid unnecessary MPI initialization and
/// ignore user-specific plugin settings for deterministic batch output.
fn write_paraview_state(
    script_path: &std::path::Path,
    cfd_vtu_path: &std::path::Path,
    render_png_path: &std::path::Path,
) -> std::io::Result<()> {
    let cfd_vtu = cfd_vtu_path.to_string_lossy();
    let render_png = render_png_path.to_string_lossy();
    let body = format!(
        r#"# Auto-generated by pcfan-opt. Open with:
#   paraview --script={script}        (interactive GUI)
#   pvbatch --no-mpi --disable-registry --force-offscreen-rendering {script}
#                                      (offscreen, writes PNG)
#
# On macOS the binaries are inside the app bundle:
#   /Applications/ParaView-X.Y.Z.app/Contents/MacOS/paraview --script=...
#   /Applications/ParaView-X.Y.Z.app/Contents/bin/pvbatch --no-mpi \\
#       --disable-registry --force-offscreen-rendering ...
# This script does not use volume rendering; OpenVKL diagnostics alone do not
# invalidate the render. Check the generated PNG and pvbatch exit status.
import os
from paraview.simple import *

CFD_VTU = "{cfd_vtu}"
RENDER_PNG = "{render_png}"

def _apply_first_preset(lut, candidates):
    """ParaView preset names changed across versions. Try each candidate
    and stop at the first one that exists. Silently no-op if none match."""
    for name in candidates:
        try:
            lut.ApplyPreset(name, True)
            return
        except Exception:
            continue

VIRIDIS_PRESETS = [
    "Viridis (matplotlib)",  # ParaView ≤ 5.x
    "Viridis",                # ParaView 6.x renamed it
    "Cool to Warm",           # universal fallback
]

# --- Load the CFD volumetric mesh ---
cfd = XMLUnstructuredGridReader(FileName=[CFD_VTU])
cfd.PointArrayStatus = [
    "velocity_m_s",
    "pressure_pa",
    "speed_m_s",
    "axial_position_m",
    "nearest_solid_boundary_distance_m",
]

view = GetActiveViewOrCreate("RenderView")
view.OrientationAxesVisibility = 1
view.Background = [0.95, 0.95, 0.95]
view.ViewSize = [1920, 1080]

# --- Bounding-box info to size the seeds + camera ---
cfd.UpdatePipeline()
bb = cfd.GetDataInformation().GetBounds()
xmin, xmax, ymin, ymax, zmin, zmax = bb
cx = 0.5 * (xmin + xmax)
cy = 0.5 * (ymin + ymax)
diag = ((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2) ** 0.5

# --- Streamlines, seeded on a 10x10 grid at the inlet (z = zmin) ---
seed_z = zmin + 0.05 * (zmax - zmin)
seed = PointSource(Center=[cx, cy, seed_z], Radius=0.45 * (xmax - xmin))
seed.NumberOfPoints = 100
streamlines = StreamTracerWithCustomSource(Input=cfd, SeedSource=seed)
streamlines.Vectors = ["POINTS", "velocity_m_s"]
streamlines.MaximumStreamlineLength = 4.0 * (zmax - zmin)
streamlines.IntegrationDirection = "FORWARD"
sl_disp = Show(streamlines, view)
ColorBy(sl_disp, ("POINTS", "speed_m_s"))
sl_disp.RescaleTransferFunctionToDataRange(True, False)
sl_lut = GetColorTransferFunction("speed_m_s")
_apply_first_preset(sl_lut, VIRIDIS_PRESETS)

# --- Centre-plane slice (Z = midpoint) coloured by speed ---
slice_z = Slice(Input=cfd)
slice_z.SliceType = "Plane"
slice_z.SliceType.Origin = [cx, cy, 0.5 * (zmin + zmax)]
slice_z.SliceType.Normal = [0.0, 0.0, 1.0]
slice_disp = Show(slice_z, view)
ColorBy(slice_disp, ("POINTS", "speed_m_s"))
slice_disp.Opacity = 0.6
slice_disp.Representation = "Surface"

# --- Velocity glyphs (subsampled arrows showing direction + magnitude) ---
glyph = Glyph(Input=cfd, GlyphType="Arrow")
glyph.OrientationArray = ["POINTS", "velocity_m_s"]
glyph.ScaleArray = ["POINTS", "speed_m_s"]
glyph.ScaleFactor = 0.15 * diag
glyph.GlyphMode = "Every Nth Point"
glyph.Stride = max(1, cfd.GetDataInformation().GetNumberOfPoints() // 200)
glyph_disp = Show(glyph, view)
ColorBy(glyph_disp, ("POINTS", "speed_m_s"))
glyph_disp.Opacity = 0.8

# --- Pressure isosurface (subtle, helps locate stagnation regions) ---
contour = Contour(Input=cfd)
contour.ContourBy = ["POINTS", "pressure_pa"]
contour.Isosurfaces = [0.0]
contour_disp = Show(contour, view)
contour_disp.Opacity = 0.25
contour_disp.AmbientColor = [0.7, 0.7, 0.7]
contour_disp.DiffuseColor = [0.7, 0.7, 0.7]

# --- Camera ---
view.CameraPosition = [
    cx + 1.6 * (xmax - xmin),
    cy - 1.6 * (ymax - ymin),
    zmax + 1.2 * (zmax - zmin),
]
view.CameraFocalPoint = [cx, cy, 0.5 * (zmin + zmax)]
view.CameraViewUp = [0.0, 0.0, 1.0]
ResetCamera(view)

# --- Save a PNG when run via pvbatch ---
try:
    os.makedirs(os.path.dirname(RENDER_PNG), exist_ok=True)
except OSError:
    pass
SaveScreenshot(RENDER_PNG, view, ImageResolution=[1920, 1080])
"#,
        script = script_path.display(),
        cfd_vtu = cfd_vtu,
        render_png = render_png,
    );
    std::fs::write(script_path, body)?;
    Ok(())
}

/// Insert `suffix` before the `.vtu` extension. e.g.
/// `optimized.vtu` + `_cfd` → `optimized_cfd.vtu`.
fn sibling_path(base: &std::path::Path, suffix: &str) -> PathBuf {
    let stem = base
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| String::from("output"));
    let extension = base
        .extension()
        .map(|e| e.to_string_lossy().to_string())
        .unwrap_or_else(|| String::from("vtu"));
    let parent = base.parent().map(|p| p.to_path_buf()).unwrap_or_default();
    parent.join(format!("{stem}{suffix}.{extension}"))
}

/// Render the generated ParaView state when the host provides `pvbatch`.
/// Missing ParaView is deliberately non-fatal: the gallery still contains its
/// deterministic data-derived figures and the batch script. A discovered
/// `pvbatch` that fails, however, is a failed gallery contract.
fn render_paraview_gallery(script: &Path, render: &Path) -> std::io::Result<Option<PathBuf>> {
    let Some(pvbatch) = pvbatch_executable() else {
        eprintln!(
            "ParaView pvbatch unavailable; retaining deterministic gallery figures and script"
        );
        return Ok(None);
    };
    let status = std::process::Command::new(pvbatch)
        .args([
            "--no-mpi",
            "--disable-registry",
            "--force-offscreen-rendering",
        ])
        .arg(script)
        .status()?;
    if !status.success() {
        return Err(std::io::Error::other("pvbatch gallery render failed"));
    }
    sonium_xem::io::validate_png_nonblank(render)?;
    Ok(Some(render.to_path_buf()))
}

fn pvbatch_executable() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("PVBATCH") {
        return Some(PathBuf::from(path));
    }
    if std::process::Command::new("pvbatch")
        .arg("--version")
        .output()
        .is_ok()
    {
        return Some(PathBuf::from("pvbatch"));
    }
    #[cfg(target_os = "macos")]
    {
        let applications = std::fs::read_dir("/Applications").ok()?;
        applications
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .find_map(|app| {
                app.file_name()
                    .is_some_and(|name| name.to_string_lossy().starts_with("ParaView-"))
                    .then(|| app.join("Contents/bin/pvbatch"))
                    .filter(|path| path.is_file())
            })
    }
    #[cfg(not(target_os = "macos"))]
    None
}

/// Write fixed-size SVG plots plus their authoritative VTU/CSV inputs and a
/// provenance manifest. This makes the documentation assets reproducible
/// without making rendered pixels the physics oracle.
struct GalleryRequest<'a> {
    directory: &'a Path,
    config: &'a PcFanConfig,
    optimized: &'a PcFanEvaluation,
    baseline: &'a PcFanEvaluation,
    design: &'a PcFanDesignVariables,
    geometry: &'a sonium_sim_pcfan::PcFanGeometry,
    cfd: Option<&'a PcFanCfdFlowSolution>,
    paraview_render: Option<&'a Path>,
}

fn write_gallery(request: GalleryRequest<'_>) -> std::io::Result<()> {
    use std::fmt::Write as _;
    let GalleryRequest {
        directory,
        config,
        optimized,
        baseline,
        design,
        geometry,
        cfd,
        paraview_render,
    } = request;
    // A gallery that includes a CFD field is a production claim about that
    // field. Admit it before creating any artifact, and use the CFD-backed
    // evaluation for the associated geometry/acoustic plots rather than the
    // surrogate evaluation that selected the candidate.
    let cfd_evaluation = cfd
        .map(|flow| {
            try_evaluate_pc_fan_from_cfd_solution(config, design, flow).map_err(|error| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("PC-fan gallery refuses inadmissible CFD field: {error}"),
                )
            })
        })
        .transpose()?;
    let optimized = cfd_evaluation.as_ref().unwrap_or(optimized);
    std::fs::create_dir_all(directory)?;
    geometry.write_evaluation_vtu(optimized, directory.join("optimized_geometry.vtu"))?;
    if let Some(flow) = cfd {
        flow.write_vtu(directory.join("flow_field.vtu"))?;
    }

    let mut csv = String::from("frequency_hz,spl_db,swl_db\n");
    for h in &optimized.acoustics.harmonics {
        writeln!(
            csv,
            "{:.6},{:.6},{:.6}",
            h.frequency_hz, h.sound_pressure_level_db, h.sound_power_level_db
        )
        .unwrap();
    }
    std::fs::write(directory.join("acoustic_spectrum.csv"), csv)?;
    std::fs::write(
        directory.join("objective_comparison.csv"),
        format!(
            "design,objective\nbaseline,{:.12}\noptimized,{:.12}\n",
            baseline.objective, optimized.objective
        ),
    )?;
    let mut fan_curve_csv =
        String::from("rpm,target_airflow_cfm,model_airflow_cfm,pressure_rise_pa,flow_model\n");
    let mut curve_airflow = Vec::with_capacity(5);
    let mut curve_pressure = Vec::with_capacity(5);
    for factor in [0.5, 0.75, 1.0, 1.25, 1.5] {
        let mut curve_config = config.clone();
        curve_config.operating_point.rpm *= factor;
        curve_config.operating_point.target_airflow_cfm *= factor;
        let point = PcFanEvaluation::from_design(&curve_config, design, geometry);
        if !point.airflow_cfm.is_finite() || !point.flow.pressure_rise_pa.is_finite() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "non-finite PC-fan curve point",
            ));
        }
        writeln!(
            fan_curve_csv,
            "{:.9},{:.9},{:.9},{:.9},{:?}",
            curve_config.operating_point.rpm,
            curve_config.operating_point.target_airflow_cfm,
            point.airflow_cfm,
            point.flow.pressure_rise_pa,
            curve_config.cfd.flow_model,
        )
        .unwrap();
        curve_airflow.push(point.airflow_cfm);
        curve_pressure.push(point.flow.pressure_rise_pa);
    }
    let has_distinct_flow = curve_airflow
        .windows(2)
        .any(|pair| (pair[1] - pair[0]).abs() > f64::EPSILON);
    let has_distinct_pressure = curve_pressure
        .windows(2)
        .any(|pair| (pair[1] - pair[0]).abs() > f64::EPSILON);
    if !has_distinct_flow || !has_distinct_pressure {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "PC-fan RPM sweep collapsed to a constant flow or pressure",
        ));
    }
    std::fs::write(directory.join("fan_curve.csv"), fan_curve_csv)?;
    let spectrum_frequency: Vec<f64> = optimized
        .acoustics
        .harmonics
        .iter()
        .map(|h| h.frequency_hz)
        .collect();
    let spectrum_spl: Vec<f64> = optimized
        .acoustics
        .harmonics
        .iter()
        .map(|h| h.sound_pressure_level_db)
        .collect();
    let panel_area: Vec<f64> = geometry
        .surface_panels()
        .into_iter()
        .map(|panel| panel.area)
        .collect();
    let mut rendered = sonium_xem::io::GalleryRenderer::new(directory)?;
    rendered.heatmap_png(
        "geometry_panel_area.png",
        &panel_area,
        "optimized_geometry.vtu",
        "Production surface-panel areas (m²) from the optimized fan geometry",
    )?;
    rendered.heatmap_png(
        "directivity_heatmap.png",
        &optimized.observer_spls_db,
        "optimized_geometry.vtu",
        "Fixed-camera directivity proxy from evaluated observer SPL values (dB re 20 uPa)",
    )?;
    rendered.line_svg(
        "fan_curve.svg",
        &curve_airflow,
        &curve_pressure,
        "PC-fan RPM-sweep pressure-rise curve",
        "fan_curve.csv",
        "Pressure rise (Pa) versus model airflow (CFM) for five RPM-scaled operating points on the pinned geometry",
    )?;
    rendered.line_svg(
        "acoustic_spectrum_plot.svg",
        &spectrum_frequency,
        &spectrum_spl,
        "PC fan acoustic spectrum",
        "acoustic_spectrum.csv",
        "Harmonic SPL (dB re 20 uPa) versus frequency (Hz)",
    )?;
    rendered.line_svg(
        "baseline_optimized_objective.svg",
        &[0.0, 1.0],
        &[baseline.objective, optimized.objective],
        "Baseline versus optimized objective",
        "objective_comparison.csv",
        "Two evaluated production objectives: baseline (x=0) and optimized design (x=1)",
    )?;
    if let Some(flow) = cfd {
        let speed: Vec<f64> = flow
            .velocity_m_s
            .iter()
            .map(|velocity| {
                (velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2])
                    .sqrt()
            })
            .collect();
        rendered.heatmap_png(
            "flow_speed.png",
            &speed,
            "flow_field.vtu",
            "CFD nodal speed (m/s), fixed deterministic false-colour map",
        )?;
        rendered.heatmap_png(
            "flow_pressure.png",
            &flow.pressure_pa,
            "flow_field.vtu",
            "CFD nodal static pressure (Pa), fixed deterministic false-colour map",
        )?;
    }
    if let Some(render) = paraview_render {
        let filename = render
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "invalid ParaView render name",
                )
            })?;
        if render.parent() != Some(directory) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "ParaView gallery render must be inside the gallery directory",
            ));
        }
        sonium_xem::io::validate_png_nonblank(render)?;
        rendered.existing_figure(
            filename,
            "png",
            "flow_field.vtu",
            "Fixed-camera ParaView streamline, slice, glyph, and pressure-contour render from the CFD VTU",
        )?;
    }
    let manifest_path = rendered.finish(
        "Pinned 120 mm PC fan gallery",
        vec![
            "fixed 1920x1080 viewport and deterministic colour mapping".into(),
            "VTU and CSV assets are authoritative production data".into(),
        ],
    )?;
    sonium_xem::io::validate_gallery_manifest(&manifest_path)?;
    let checksums = std::fs::read_dir(directory)?
        .filter_map(Result::ok)
        .filter_map(|entry| entry.file_name().into_string().ok())
        .map(|name| {
            let data = std::fs::read(directory.join(&name)).unwrap_or_default();
            (
                name,
                serde_json::json!(format!("fnv1a64:{:016x}", fnv1a64(&data))),
            )
        })
        .collect::<serde_json::Map<String, serde_json::Value>>();
    let git_sha = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|sha| sha.trim().to_owned())
        .unwrap_or_else(|| "unknown".to_owned());
    let manifest = serde_json::json!({
        "schema": "sonium-sim-pcfan-gallery/v1", "git_sha": git_sha,
        "scenario": "pinned-120mm-pc-fan", "viewport": {"width": 1920, "height": 1080},
        "units": {"airflow": "CFM", "pressure": "Pa", "spl": "dB re 20 uPa"},
        "solver": {"flow_model": format!("{:?}", config.cfd.flow_model), "rpm": config.operating_point.rpm},
        "provenance": "evaluated Sonium model; SVG plots are derived from accompanying VTU/CSV data; fan_curve.csv records its operating-point sweep and flow model",
        "optimized": {"objective": optimized.objective, "airflow_cfm": optimized.airflow_cfm, "pressure_rise_pa": optimized.flow.pressure_rise_pa},
        "checksums": checksums,
    });
    std::fs::write(
        directory.join("gallery-provenance.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

fn print_run_setup(config: &PcFanConfig) {
    println!(
        "flow model: {:?}, CFD grid {}x{}x{}, Picard max {}",
        config.cfd.flow_model,
        config.cfd.cells_x,
        config.cfd.cells_y,
        config.cfd.cells_z,
        config.cfd.max_picard_iterations
    );
    println!(
        "operating point: {:.1} rpm, target airflow {:.2} CFM",
        config.operating_point.rpm, config.operating_point.target_airflow_cfm
    );
}

fn print_evaluation_report(evaluation: &PcFanEvaluation) {
    println!(
        "airflow: {:.2} CFM (margin {:+.2} CFM)",
        evaluation.airflow_cfm, evaluation.constraint_margins.airflow_cfm
    );
    println!(
        "noise: {:.2} dB SPL, {:.2} dB SWL",
        evaluation.acoustic_level_db, evaluation.sound_power_level_db
    );
    println!(
        "spread: spectral flatness {:.3} (penalty {:.3}), directivity σ {:.2} dB (penalty {:.3}), operating-point σ {:.2} dB (penalty {:.3})",
        evaluation.spectral_flatness,
        evaluation.spectral_spread_penalty,
        evaluation.directivity_spread_db,
        evaluation.directivity_spread_penalty,
        evaluation.operating_point_spread_db,
        evaluation.operating_point_spread_penalty
    );
    if !evaluation.operating_point_spls_db.is_empty() {
        let s = evaluation
            .operating_point_spls_db
            .iter()
            .map(|v| format!("{:.2}", v))
            .collect::<Vec<_>>()
            .join(", ");
        println!("operating-point SPL (dB): {s}");
    }
    println!(
        "observer SPL: min {:.2}, max {:.2}, range {:.2} dB ({} obs)",
        observer_min_db(&evaluation.observer_spls_db),
        observer_max_db(&evaluation.observer_spls_db),
        observer_max_db(&evaluation.observer_spls_db)
            - observer_min_db(&evaluation.observer_spls_db),
        evaluation.observer_spls_db.len()
    );
    println!(
        "flow: {:.3} m/s axial, {:.2} Pa rise, Re {:.2e}, M {:.4}",
        evaluation.flow.axial_velocity_m_s,
        evaluation.flow.pressure_rise_pa,
        evaluation.flow.reynolds_number,
        evaluation.flow.mach_number
    );
    println!(
        "hardware: blockage {:.1}% (motor {:.1}%, X-support {:.1}%), loss factor {:.3}",
        100.0 * evaluation.flow.fixed_hardware.blockage_ratio,
        100.0 * evaluation.flow.fixed_hardware.motor_blockage_ratio,
        100.0 * evaluation.flow.fixed_hardware.strut_blockage_ratio,
        evaluation.flow.fixed_hardware.total_loss_factor
    );
    println!(
        "hardware noise: {:.2} dB SPL (X-support {:.2} tonal + {:.2} broadband, motor {:.2} tonal + {:.2} broadband)",
        evaluation.acoustics.fixed_hardware_noise.combined_spl_db,
        evaluation
            .acoustics
            .fixed_hardware_noise
            .support_wake_spl_db,
        evaluation
            .acoustics
            .fixed_hardware_noise
            .support_wake_broadband_spl_db,
        evaluation.acoustics.fixed_hardware_noise.motor_mount_spl_db,
        evaluation
            .acoustics
            .fixed_hardware_noise
            .motor_mount_broadband_spl_db,
    );
    println!(
        "tip-leakage broadband: {:.2} dB SPL",
        evaluation.acoustics.tip_leakage_spl_db
    );
    println!(
        "trailing-edge broadband (BPM): {:.2} dB SPL",
        evaluation.acoustics.trailing_edge_spl_db
    );
    println!(
        "leading-edge inflow (Amiet): {:.2} dB SPL",
        evaluation.acoustics.leading_edge_inflow_spl_db
    );
    println!(
        "collision: valid={}, invalid intersections={}, min gap {:.6} m",
        evaluation.collision_status.is_valid,
        evaluation.collision_status.invalid_intersection_count,
        evaluation.collision_status.minimum_blade_gap_m
    );
    println!(
        "constraints: shroud clearance {:+.5} m, airflow penalty {:.3}, collision penalty {:.3}, shroud-clearance penalty {:.3}, frame-containment penalty {:.3}, hub-attachment penalty {:.3}",
        evaluation.constraint_margins.minimum_shroud_clearance_m,
        evaluation.airflow_penalty,
        evaluation.collision_penalty,
        evaluation.shroud_clearance_penalty,
        evaluation.frame_containment_penalty,
        evaluation.hub_attachment_penalty,
    );
}

fn observer_min_db(spls: &[f64]) -> f64 {
    spls.iter().fold(f64::INFINITY, |a, &b| a.min(b))
}

fn observer_max_db(spls: &[f64]) -> f64 {
    spls.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
}

fn print_design_variables(design: &PcFanDesignVariables) {
    println!("blade sections:");
    for (i, section) in design.blade_sections.iter().enumerate() {
        println!(
            "  {:02}: chord {:.5} m, camber {:+.5} m, thickness {:.5} m, twist {:+.2} deg",
            i, section.chord_m, section.camber_m, section.thickness_m, section.twist_deg
        );
    }

    println!("blade path (sweep + lean):");
    for (i, control) in design.blade_path.iter().enumerate() {
        println!(
            "  {:02}: tangential {:+.5} m (− = backward), axial {:+.5} m",
            i, control.tangential_offset_m, control.axial_offset_m
        );
    }

    println!("shroud offsets:");
    for (i, offset) in design.shroud_radius_offsets_m.iter().enumerate() {
        println!("  {:02}: {:+.5} m", i, offset);
    }
}

fn print_cfd_report(solution: &PcFanCfdFlowSolution) {
    println!(
        "CFD: airflow {:.2} CFM, inlet airflow {:.2} CFM, mass balance {:.3e} kg/s",
        solution.airflow_cfm, solution.inlet_airflow_cfm, solution.mass_balance_error_kg_s
    );
    println!(
        "CFD: pressure drop {:.2} Pa, max velocity {:.3} m/s, Re {:.2e}",
        solution.pressure_drop_pa, solution.max_velocity_m_s, solution.reynolds_number
    );
    println!(
        "CFD solve: converged={}, iterations={}, residual {:.3e}, FEM nodes={}, elements={}, boundary DOFs={}",
        solution.converged,
        solution.iterations,
        solution.residual,
        solution.fem_node_count,
        solution.fem_element_count,
        solution.boundary_dof_count
    );
}

fn print_mesh_convergence_report(convergence: &PcFanCfdMeshConvergence) {
    println!(
        "mesh convergence: converged={} all-level-converged-and-finite={} (airflow and pressure gate)",
        convergence.converged, convergence.all_levels_converged_and_finite
    );
    for level in &convergence.levels {
        let airflow_change = level
            .airflow_relative_change
            .map_or(String::from("n/a"), |change| {
                format!("{:.2}%", 100.0 * change)
            });
        let pressure_change = level
            .pressure_relative_change
            .map_or(String::from("n/a"), |change| {
                format!("{:.2}%", 100.0 * change)
            });
        println!(
            "  grid {}x{}x{}: airflow {:.2} CFM, pressure {:.2} Pa, nonlinear residual {:.3e}, changes airflow {}, pressure {}",
            level.cells_x,
            level.cells_y,
            level.cells_z,
            level.solution.airflow_cfm,
            level.solution.pressure_drop_pa.abs(),
            level.solution.residual,
            airflow_change,
            pressure_change
        );
    }
}

fn default_output_path(filename: &str) -> PathBuf {
    binary_output_dir(env!("CARGO_PKG_NAME"), env!("CARGO_BIN_NAME")).join(filename)
}

fn build_config(cli: &Cli) -> PcFanConfig {
    let mut config = match cli.installation {
        Installation::Bench => PcFanConfig::bench_120mm(),
        Installation::Chassis => PcFanConfig::chassis_installed_120mm(),
    };
    config.operating_point = match cli.installation {
        Installation::Bench => PcFanOperatingPoint::bench(cli.rpm),
        Installation::Chassis => PcFanOperatingPoint::chassis_installed(cli.rpm),
    };
    let empirical_airflow_prior = match cli.installation {
        Installation::Bench => PcFanOperatingPoint::bench_airflow_prior_cfm(cli.rpm),
        Installation::Chassis => PcFanOperatingPoint::chassis_airflow_prior_cfm(cli.rpm),
    };
    if let Some(target_airflow_cfm) = cli.target_airflow_cfm {
        if !(empirical_airflow_prior[0]..=empirical_airflow_prior[1]).contains(&target_airflow_cfm)
        {
            eprintln!(
                "warning: {:.1} CFM is outside the RPM-scaled {:?} empirical prior {:.1}–{:.1} CFM; treat this as an extrapolation",
                target_airflow_cfm,
                cli.installation,
                empirical_airflow_prior[0],
                empirical_airflow_prior[1]
            );
        }
        config.operating_point.target_airflow_cfm = target_airflow_cfm;
    }
    if let Some(inflow_ti) = cli.inflow_ti {
        config.operating_point.inflow_turbulence_intensity = inflow_ti;
    }
    if let Some(inflow_length_scale_m) = cli.inflow_length_scale_m {
        config.operating_point.inflow_integral_length_scale_m = inflow_length_scale_m;
    }
    if let Some(modal_loss_factor) = cli.modal_loss_factor {
        config.enclosure.modal_damping = modal_loss_factor;
    }
    config.optimizer.max_iterations = cli.iterations;
    config.mesh.harmonic_count = cli.harmonics;
    if cli.gallery.is_some() {
        // Keep the gallery smoke path aligned with the bounded production QA
        // fixture.  Dense blade sampling on the deliberately coarse CFD mesh
        // changes the immersed forcing enough to fail the strict mass gate;
        // gallery assets are observability evidence, not a mesh-resolution
        // study.
        config.mesh.radial_segments = 2;
        config.mesh.chord_segments = 2;
        config.mesh.circumferential_segments = 8;
        config.mesh.harmonic_count = config.mesh.harmonic_count.min(2);
    }
    config.cfd.cells_x = cli.cfd_cells_x;
    config.cfd.cells_y = cli.cfd_cells_y;
    config.cfd.cells_z = cli.cfd_cells_z;
    config.cfd.max_picard_iterations = cli.cfd_picard_iterations;
    config.cfd.nonlinear_tolerance = cli.cfd_tolerance;
    config.cfd.linear_solver_max_iterations = cli.cfd_linear_iterations.max(1);
    config.cfd.linear_solver_tolerance = cli.cfd_linear_tolerance;
    config.cfd.use_direct_solver = cli.cfd_direct_solver;
    config.cfd.cpu_preconditioner = match cli.cfd_cpu_precond.to_ascii_lowercase().as_str() {
        "ilu" | "legacy" => sonium_sim_pcfan::PcFanCpuPreconditionerKind::Ilu,
        "jacobi-schur" | "jacobi_schur" | "schur" => {
            sonium_sim_pcfan::PcFanCpuPreconditionerKind::JacobiSchur
        }
        "ilut-schur" | "ilut_schur" => sonium_sim_pcfan::PcFanCpuPreconditionerKind::IlutSchur,
        other => {
            eprintln!("warning: unknown --cfd-cpu-precond '{other}', defaulting to ilu");
            sonium_sim_pcfan::PcFanCpuPreconditionerKind::Ilu
        }
    };
    if cli.cfd || cli.solve_only || cli.mesh_converged || cli.cfd_full || cli.gallery.is_some() {
        config.cfd.flow_model = if cli.cfd_full {
            PcFanFlowModel::IncompressibleCfdFull
        } else {
            PcFanFlowModel::IncompressibleCfd
        };
        // Coarse CFD grids cannot resolve the sub-µm geometry kicks the
        // reduced-order FD step uses, so the gradient evaluates to zero
        // and L-BFGS-B converges after one iteration. Bump to a CFD-
        // appropriate step. Users can still override via the
        // `--cfd-fd-step` flag.
        config.optimizer.finite_difference_step = cli.cfd_fd_step.max(1.0e-6);
    }
    if cli.cfd_full {
        // Full-CFD mode: enlarge the enclosure so the outer no-slip walls
        // don't constrain the flow. Reasonable defaults for a 120 mm fan:
        //   - inlet/outlet plenum ≈ 1× frame side (120 mm each side)
        //   - side clearance ≈ 0.5× frame side (60 mm of breathing room)
        // Users can still override via the existing CFD knobs after the
        // fact (the optimiser config is read-write here).
        let frame = config.dimensions.frame_side_m;
        if config.cfd.inlet_extension_m < frame {
            config.cfd.inlet_extension_m = frame;
        }
        if config.cfd.outlet_extension_m < frame {
            config.cfd.outlet_extension_m = frame;
        }
        if config.cfd.side_clearance_m < 0.5 * frame {
            config.cfd.side_clearance_m = 0.5 * frame;
        }
    }
    config.weights.airflow_band_fraction = cli.airflow_band;
    config.weights.acoustic_weight = cli.w_spl;
    config.weights.spectral_spread_weight = cli.w_spectral_spread;
    config.weights.directivity_spread_weight = cli.w_directivity_spread;
    config.weights.operating_spread_weight = cli.w_operating_spread;
    config.observers.azimuth_count = cli.observers_azimuth.max(1);
    config.observers.elevation_count = cli.observers_elevation.max(1);
    config.observers.distance_m = cli.observer_distance_m.max(0.05);
    if cli.no_broadband {
        config.broadband = sonium_sim_pcfan::PcFanBroadbandConfig::off();
    }
    if !cli.operating_rpms.is_empty() {
        config.operating_point_spread.rpms = cli.operating_rpms.clone();
    } else {
        config.operating_point_spread.rpms = vec![0.7 * cli.rpm, cli.rpm, 1.3 * cli.rpm];
    }
    config.cfd.turbulence = match cli.turbulence.to_ascii_lowercase().as_str() {
        "laminar" => sonium_sim_pcfan::PcFanTurbulenceModel::Laminar,
        "komega" | "komega-sst" | "k-omega-sst" => {
            sonium_sim_pcfan::PcFanTurbulenceModel::KOmegaSstAlgebraic
        }
        "komega-sst-transported"
        | "k-omega-sst-transported"
        | "komega-transported"
        | "k-omega-transported" => sonium_sim_pcfan::PcFanTurbulenceModel::KOmegaSstTransported,
        other => {
            eprintln!("warning: unknown --turbulence '{other}', defaulting to laminar");
            sonium_sim_pcfan::PcFanTurbulenceModel::Laminar
        }
    };
    config.optimizer.freeze_shroud = !cli.optimise_shroud;
    config.optimizer.require_positive_camber = !cli.allow_negative_camber;
    if let Some(r) = cli.blade_tip_radius_m {
        config.dimensions.blade_tip_radius_m = r.max(config.dimensions.hub_radius_m + 1.0e-4);
    }
    config.weights.shroud_clearance_min_m = cli.shroud_clearance_min_m.max(0.0);
    config.weights.shroud_clearance_penalty_weight = cli.shroud_clearance_penalty.max(0.0);
    config.weights.frame_containment_penalty_weight = cli.frame_containment_penalty.max(0.0);
    config.weights.hub_attachment_penalty_weight = cli.hub_attachment_penalty.max(0.0);
    config.cfd.mesh_pattern = match cli.cfd_mesh.to_ascii_lowercase().as_str() {
        "symmetric" | "sym" | "barycentric" => sonium_sim_pcfan::PcFanCfdMeshPattern::Symmetric,
        "kuhn" | "k" => sonium_sim_pcfan::PcFanCfdMeshPattern::Kuhn,
        other => {
            eprintln!("warning: unknown --cfd-mesh '{other}', defaulting to symmetric");
            sonium_sim_pcfan::PcFanCfdMeshPattern::Symmetric
        }
    };
    config.cfd.backend = match cli.cfd_backend.to_ascii_lowercase().as_str() {
        "auto" => sonium_sim_pcfan::PcFanCfdBackend::Auto,
        "cpu" => sonium_sim_pcfan::PcFanCfdBackend::Cpu,
        "cuda" | "nvidia" => sonium_sim_pcfan::PcFanCfdBackend::Cuda,
        "metal" | "apple" => sonium_sim_pcfan::PcFanCfdBackend::Metal,
        other => {
            eprintln!("warning: unknown --cfd-backend '{other}', defaulting to auto");
            sonium_sim_pcfan::PcFanCfdBackend::Auto
        }
    };
    config.cfd.precond_polynomial_degree = cli.cfd_precond_degree.max(1);
    config.cfd.precond_polynomial_damping = cli.cfd_precond_damping;
    config.cfd.precond_schur_degree = cli.cfd_schur_degree.max(1);
    config.cfd.precond_schur_damping = cli.cfd_schur_damping;
    config.cfd.precond_schur_kind = match cli.cfd_schur_precond.to_ascii_lowercase().as_str() {
        "polynomial" | "poly" | "jacobi" | "diagonal" => {
            sonium_sim_pcfan::PcFanBackendPressureSchurPreconditionerKind::Polynomial
        }
        "chebyshev" | "cheb" => {
            sonium_sim_pcfan::PcFanBackendPressureSchurPreconditionerKind::Chebyshev
        }
        "amg" => sonium_sim_pcfan::PcFanBackendPressureSchurPreconditionerKind::Amg,
        other => {
            eprintln!("warning: unknown --cfd-schur-precond '{other}', defaulting to polynomial");
            sonium_sim_pcfan::PcFanBackendPressureSchurPreconditionerKind::Polynomial
        }
    };
    config.cfd.precond_schur_chebyshev_degree = cli.cfd_schur_cheb_degree.max(1);
    config.cfd.precond_schur_chebyshev_eigenvalue_ratio =
        cli.cfd_schur_cheb_eigenvalue_ratio.max(1.0);
    config.cfd.precond_kind = match cli.cfd_precond.to_ascii_lowercase().as_str() {
        "polynomial-jacobi" | "polynomial" | "poly" | "jacobi" => {
            sonium_sim_pcfan::PcFanBackendPreconditionerKind::PolynomialJacobi
        }
        "chebyshev" | "cheb" => sonium_sim_pcfan::PcFanBackendPreconditionerKind::Chebyshev,
        "jacobi-schur" | "jacobi_schur" | "schur" => {
            sonium_sim_pcfan::PcFanBackendPreconditionerKind::JacobiSchur
        }
        other => {
            eprintln!("warning: unknown --cfd-precond '{other}', defaulting to polynomial-jacobi");
            sonium_sim_pcfan::PcFanBackendPreconditionerKind::PolynomialJacobi
        }
    };
    config.cfd.precond_chebyshev_degree = cli.cfd_cheb_degree.max(1);
    config.cfd.precond_chebyshev_eigenvalue_ratio = cli.cfd_cheb_eigenvalue_ratio.max(1.0);
    config.cfd.solver_kind = match cli.cfd_solver.to_ascii_lowercase().as_str() {
        "f64" | "double" => sonium_sim_pcfan::PcFanLinearSolverKind::F64,
        "mixed-f32-f64" | "mixed" | "ir" | "fgmres-ir" => {
            sonium_sim_pcfan::PcFanLinearSolverKind::MixedF32F64
        }
        other => {
            eprintln!("warning: unknown --cfd-solver '{other}', defaulting to f64");
            sonium_sim_pcfan::PcFanLinearSolverKind::F64
        }
    };
    config.cfd.solver_ir_inner_max_iterations = cli.cfd_ir_inner_max.max(1);
    config.cfd.solver_ir_inner_tolerance = cli.cfd_ir_inner_tol;
    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_rejects_bad_inputs_and_accepts_small_cpu_system() {
        for n in [0, 100_001, usize::MAX] {
            assert!(run_gmres_smoke("cpu", n).is_err());
        }
        assert!(run_gmres_smoke("unknown", 2).is_err());
        run_gmres_smoke("cpu", 16).unwrap();
        let backend = sonium_compute::cpu::CpuBackend::new();
        let config = sonium_compute::gmres::GmresBackendConfig {
            max_iterations: 100,
            tolerance: 1e-10,
        };
        assert!(smoke_one_backend(&backend, &[0, 1], &[9], &[4.0], &[1.0], 1, &config).is_err());
        assert!(
            smoke_one_backend(&backend, &[0, 1], &[0], &[4.0], &[f64::NAN], 1, &config).is_err()
        );
        assert!(smoke_one_backend(&backend, &[0, 1], &[0], &[4.0], &[], 1, &config).is_err());
    }

    #[test]
    fn automatic_thread_budget_is_default_and_splits_multi_de() {
        let automatic = Cli::parse_from(["pcfan-opt", "--de-experiments", "4"]);
        assert_eq!(automatic.threads, None);
        assert_eq!(automatic.de_cores_per, None);
        assert_eq!(
            resolved_de_cores_per_experiment(&automatic),
            (rayon::current_num_threads() / 4).max(1)
        );

        let explicit = Cli::parse_from([
            "pcfan-opt",
            "--threads",
            "16",
            "--de-experiments",
            "4",
            "--de-cores-per",
            "3",
        ]);
        assert_eq!(explicit.threads, Some(16));
        assert_eq!(resolved_de_cores_per_experiment(&explicit), 3);

        let derived = Cli::parse_from(["pcfan-opt", "--threads", "16", "--de-experiments", "4"]);
        assert_eq!(resolved_de_cores_per_experiment(&derived), 4);
    }

    #[test]
    fn empirical_presets_select_sensible_airflow_and_environment() {
        let bench = build_config(&Cli::parse_from(["pcfan-opt"]));
        assert_eq!(bench.operating_point.target_airflow_cfm, 20.0);
        assert_eq!(bench.operating_point.inflow_turbulence_intensity, 0.0);
        assert_eq!(
            bench.enclosure.mode,
            sonium_sim_pcfan::PcFanPropagationMode::FreeField
        );

        let chassis = build_config(&Cli::parse_from(["pcfan-opt", "--installation", "chassis"]));
        assert_eq!(chassis.operating_point.target_airflow_cfm, 15.0);
        assert_eq!(chassis.operating_point.inflow_turbulence_intensity, 0.10);
        assert_eq!(chassis.enclosure.modal_damping, 0.08);
        assert_eq!(
            chassis.enclosure.mode,
            sonium_sim_pcfan::PcFanPropagationMode::RigidCavity
        );
    }

    #[test]
    fn operating_point_overrides_and_ranges_are_enforced() {
        let cli = Cli::parse_from([
            "pcfan-opt",
            "--installation",
            "chassis",
            "--rpm",
            "1200",
            "--target-airflow-cfm",
            "32",
            "--inflow-ti",
            "0.15",
            "--inflow-length-scale-m",
            "0.004",
            "--modal-loss-factor",
            "0.12",
        ]);
        let config = build_config(&cli);
        assert_eq!(config.operating_point.rpm, 1200.0);
        assert_eq!(config.operating_point.target_airflow_cfm, 32.0);
        assert_eq!(config.operating_point.inflow_turbulence_intensity, 0.15);
        assert_eq!(config.operating_point.inflow_integral_length_scale_m, 0.004);
        assert_eq!(config.enclosure.modal_damping, 0.12);
        assert_eq!(config.operating_point_spread.rpms, [840.0, 1200.0, 1560.0]);

        assert!(Cli::try_parse_from(["pcfan-opt", "--modal-loss-factor", "0.01"]).is_err());
        assert!(Cli::try_parse_from(["pcfan-opt", "--inflow-ti", "0.5"]).is_err());
    }

    #[test]
    fn cfd_cli_flags_select_incompressible_flow_model() {
        let cli = Cli::parse_from([
            "pcfan-opt",
            "--cfd",
            "--cfd-cells-x",
            "2",
            "--cfd-cells-y",
            "3",
            "--cfd-cells-z",
            "4",
            "--cfd-picard-iterations",
            "5",
            "--cfd-tolerance",
            "1e-5",
            "--cfd-linear-iterations",
            "1234",
            "--cfd-linear-tolerance",
            "2e-4",
            "--cfd-cpu-precond",
            "jacobi-schur",
            "--cfd-direct-solver",
            "--cfd-precond",
            "jacobi-schur",
            "--cfd-schur-degree",
            "3",
            "--cfd-schur-damping",
            "0.4",
            "--cfd-schur-precond",
            "chebyshev",
            "--cfd-schur-cheb-degree",
            "7",
            "--cfd-schur-cheb-eigenvalue-ratio",
            "18",
        ]);
        let config = build_config(&cli);

        assert_eq!(config.cfd.flow_model, PcFanFlowModel::IncompressibleCfd);
        assert_eq!(config.cfd.cells_x, 2);
        assert_eq!(config.cfd.cells_y, 3);
        assert_eq!(config.cfd.cells_z, 4);
        assert_eq!(config.cfd.max_picard_iterations, 5);
        assert_eq!(config.cfd.nonlinear_tolerance, 1.0e-5);
        assert_eq!(config.cfd.linear_solver_max_iterations, 1234);
        assert_eq!(config.cfd.linear_solver_tolerance, 2.0e-4);
        assert_eq!(
            config.cfd.cpu_preconditioner,
            sonium_sim_pcfan::PcFanCpuPreconditionerKind::JacobiSchur
        );
        assert!(config.cfd.use_direct_solver);
        assert_eq!(
            config.cfd.precond_kind,
            sonium_sim_pcfan::PcFanBackendPreconditionerKind::JacobiSchur
        );
        assert_eq!(config.cfd.precond_schur_degree, 3);
        assert_eq!(config.cfd.precond_schur_damping, 0.4);
        assert_eq!(
            config.cfd.precond_schur_kind,
            sonium_sim_pcfan::PcFanBackendPressureSchurPreconditionerKind::Chebyshev
        );
        assert_eq!(config.cfd.precond_schur_chebyshev_degree, 7);
        assert_eq!(config.cfd.precond_schur_chebyshev_eigenvalue_ratio, 18.0);
    }

    #[test]
    fn solve_only_cli_selects_incompressible_flow_model() {
        let cli = Cli::parse_from([
            "pcfan-opt",
            "--solve-only",
            "--cfd-cells-x",
            "2",
            "--cfd-cells-y",
            "2",
            "--cfd-cells-z",
            "2",
        ]);
        let config = build_config(&cli);

        assert!(cli.solve_only);
        assert_eq!(config.cfd.flow_model, PcFanFlowModel::IncompressibleCfd);
        assert_eq!(config.cfd.cells_x, 2);
        assert_eq!(config.cfd.cells_y, 2);
        assert_eq!(config.cfd.cells_z, 2);
    }

    #[test]
    fn gallery_cli_selects_cfd_for_the_deterministic_smoke_path() {
        let cli = Cli::parse_from(["pcfan-opt", "--gallery", "gallery"]);
        let config = build_config(&cli);
        assert_eq!(cli.gallery, Some(PathBuf::from("gallery")));
        assert_eq!(config.cfd.flow_model, PcFanFlowModel::IncompressibleCfd);
        assert_eq!(config.mesh.radial_segments, 2);
        assert_eq!(config.mesh.chord_segments, 2);
        assert_eq!(config.mesh.circumferential_segments, 8);
        assert!(config.mesh.harmonic_count <= 2);
    }

    #[test]
    fn gallery_writes_valid_raster_and_manifest_from_evaluated_data() {
        let config = PcFanConfig::default_120mm();
        let design = config.default_design_variables();
        let geometry = config.build_geometry(&design);
        let evaluation = PcFanEvaluation::from_design(&config, &design, &geometry);
        let directory = tempfile::tempdir().unwrap();
        write_gallery(GalleryRequest {
            directory: directory.path(),
            config: &config,
            optimized: &evaluation,
            baseline: &evaluation,
            design: &design,
            geometry: &geometry,
            cfd: None,
            paraview_render: None,
        })
        .unwrap();
        assert_eq!(
            sonium_xem::io::validate_png_nonblank(directory.path().join("directivity_heatmap.png"))
                .unwrap(),
            (1920, 1080)
        );
        assert!(directory.path().join("gallery.json").is_file());
        assert!(directory.path().join("acoustic_spectrum.csv").is_file());
        let fan_curve = std::fs::read_to_string(directory.path().join("fan_curve.csv")).unwrap();
        assert_eq!(fan_curve.lines().count(), 6);
        assert_ne!(
            fan_curve.lines().nth(1).unwrap(),
            fan_curve.lines().nth(5).unwrap()
        );
    }

    #[test]
    fn gallery_rejects_an_unconverged_cfd_field_before_writing_artifacts() {
        let mut config = PcFanConfig::default_120mm();
        config.cfd.flow_model = PcFanFlowModel::IncompressibleCfd;
        config.cfd.cells_x = 2;
        config.cfd.cells_y = 2;
        config.cfd.cells_z = 2;
        let design = config.default_design_variables();
        let geometry = config.build_geometry(&design);
        let evaluation = PcFanEvaluation::from_design(&config, &design, &geometry);
        let mut cfd = solve_pc_fan_cfd_mean_flow(&config, &design);
        cfd.converged = false;
        cfd.residual = 0.0;
        let directory = tempfile::tempdir().unwrap();

        let error = write_gallery(GalleryRequest {
            directory: directory.path(),
            config: &config,
            optimized: &evaluation,
            baseline: &evaluation,
            design: &design,
            geometry: &geometry,
            cfd: Some(&cfd),
            paraview_render: None,
        })
        .expect_err("unconverged CFD must not be rendered as gallery evidence");

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(!directory.path().join("gallery.json").exists());
    }

    #[test]
    fn gallery_cli_flow_exports_the_pinned_production_asset_set() {
        let directory = tempfile::tempdir().unwrap();
        // Pinned gallery smoke mirrors the `simulation-galleries` recipe:
        // 4^3 conforming cells with the sparse-direct inner solver. The
        // 2^3 transient drifts to a stalled-signed pressure drop and the
        // monolithic ILU(0) path stalls on the finer coupled systems.
        let cli = Cli::parse_from([
            "pcfan-opt",
            "--solve-only",
            "--gallery",
            directory.path().to_str().unwrap(),
            "--cfd-cells-x",
            "4",
            "--cfd-cells-y",
            "4",
            "--cfd-cells-z",
            "4",
            "--cfd-picard-iterations",
            "2",
            "--cfd-tolerance",
            "0.02",
            "--cfd-direct-solver",
            "--cfd-linear-iterations",
            "40",
            "--no-paraview",
        ]);
        let config = build_config(&cli);
        run_single_shape(config, &cli);

        let required = [
            "gallery.json",
            "gallery-provenance.json",
            "optimized_geometry.vtu",
            "flow_field.vtu",
            "fan_curve.csv",
            "acoustic_spectrum.csv",
            "objective_comparison.csv",
            "geometry_panel_area.png",
            "directivity_heatmap.png",
            "flow_speed.png",
            "flow_pressure.png",
            "fan_curve.svg",
            "acoustic_spectrum_plot.svg",
            "baseline_optimized_objective.svg",
            "single_shape_pcfan_paraview.py",
        ];
        for filename in required {
            assert!(
                directory.path().join(filename).is_file(),
                "gallery is missing required asset {filename}"
            );
        }
        let manifest = directory.path().join("gallery.json");
        sonium_xem::io::validate_gallery_manifest(&manifest).unwrap();
        let provenance: serde_json::Value = serde_json::from_slice(
            &std::fs::read(directory.path().join("gallery-provenance.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(provenance["schema"], "sonium-sim-pcfan-gallery/v1");
        assert_eq!(provenance["scenario"], "pinned-120mm-pc-fan");
        assert_eq!(provenance["viewport"]["width"], 1920);
        assert_eq!(provenance["viewport"]["height"], 1080);
        assert!(provenance["checksums"].as_object().unwrap().len() >= required.len());
        assert_eq!(
            sonium_xem::io::validate_png_nonblank(directory.path().join("directivity_heatmap.png"))
                .unwrap(),
            (1920, 1080)
        );
        assert_eq!(
            sonium_xem::io::validate_png_nonblank(directory.path().join("flow_speed.png")).unwrap(),
            (1920, 1080)
        );
    }

    #[test]
    fn paraview_gallery_render_is_nonblank_when_pvbatch_is_available() {
        // Auto-discovering an application bundle is useful for interactive
        // runs, but a CI test must not treat an unrelated broken GUI install
        // as a renderer capability. Set PVBATCH explicitly to opt into this
        // strict external-render test; native gallery assets are tested above.
        if std::env::var_os("PVBATCH").is_none() {
            eprintln!("skipping ParaView gallery render: set PVBATCH to opt in");
            return;
        }
        if pvbatch_executable().is_none() {
            eprintln!("skipping ParaView gallery render: pvbatch is unavailable");
            return;
        }

        let mut config = PcFanConfig::default_120mm();
        config.cfd.cells_x = 2;
        config.cfd.cells_y = 2;
        config.cfd.cells_z = 2;
        config.cfd.max_picard_iterations = 2;
        config.cfd.linear_solver_max_iterations = 40;
        let design = config.default_design_variables();
        let flow = solve_pc_fan_cfd_mean_flow(&config, &design);
        let directory = tempfile::tempdir().unwrap();
        let flow_path = directory.path().join("flow_field.vtu");
        let script_path = directory.path().join("gallery.py");
        let render_path = directory.path().join("gallery.png");
        flow.write_vtu(&flow_path).unwrap();
        write_paraview_state(&script_path, &flow_path, &render_path).unwrap();

        let rendered = render_paraview_gallery(&script_path, &render_path)
            .expect("pvbatch gallery render failed");
        assert_eq!(rendered.as_deref(), Some(render_path.as_path()));
        assert_eq!(
            sonium_xem::io::validate_png_nonblank(&render_path).unwrap(),
            (1920, 1080)
        );
    }
}
