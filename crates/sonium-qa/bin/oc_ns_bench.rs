//! OpenCourant-vs-sonium Navier-Stokes comparison runner.
//!
//! Times the native incompressible Navier-Stokes path against the vendor
//! explicit ALE engine on matched-accuracy flow decks and emits one frozen
//! `sonium.oc-bench/v1` bundle. The engine arm stages checked-in ALE decks
//! to scratch and times vendor starter+engine runs; the sonium arm times
//! the in-process Picard/BDF solves on the same Reynolds number and
//! geometry. Both arms check accuracy probes before any timing wins.
//!
//! Decks: `ns_lid_cavity` (steady 2D lid-driven cavity at Re=100, gated
//! on Ghia et al. centerline data) and `ns_taylor_green` (unsteady 2D
//! Taylor-Green decay at Re=100, gated on the analytic solution).
//!
//! Verdict rule: OpenCourant runs explicit compressible ALE while sonium
//! runs implicit incompressible FEM, so solver steps are not the same
//! unit across arms (explicit cycles vs Picard iterations vs BDF steps).
//! The head-to-head winner is therefore decided on wall time at matched
//! accuracy (both arms' probes passing), not on the per-step diagnostic
//! `ns_per_dof_cycle` that `compare_pair` reports. The report carries
//! both; the `NS verdict` section states the binding comparison.
//!
//! ```sh
//! qa-oc-ns-bench --deck ns_lid_cavity --emit-bundle ns.json --side both
//! qa-oc-ns-bench --deck ns_taylor_green --emit-bundle tg.json --sonium-only
//! qa-oc-ns-bench --deck ns_lid_cavity --emit-bundle gpu.json --sonium-only --backend cuda
//! ```

// Rust guideline compliant 2026-02-21

#[path = "support/oc_engine.rs"]
mod oc_engine;

use clap::{Parser, ValueEnum};
use oc_engine::{
    EngineCtx, FlavorOpt, Probe, RepeatSample, SampleRule, SideOpt, aggregate_repeats, cpu_model,
    deck_dof, engine_argv, engine_context, first_existing, load_th_history, maybe_clean,
    out_candidates, plateau_mean, read_probe_file, repeat_scratch, resolve_ns_decks,
    resolve_repeats, resolve_sides, run_rootname, run_vendor_stage, self_peak_rss_bytes,
    starter_argv, th_candidates, total_cycles,
};
use sonium_bench::oc_compare::{
    BUNDLE_SCHEMA, BenchBundle, BenchRow, HardwareInfo, PROTOCOL_CORRECTNESS, ProtocolInfo, Side,
    bundle_json, render_bundle_markdown,
};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::locate_dofs_geometrical;
use sonium_fem::mesh::Mesh;
use sonium_fem::mesh::generators::unit_square_triangles;
use sonium_fem::physics::fluid::stokes::navier_stokes::{
    NavierStokesConfig, NavierStokesResult, solve_bdf_step_cached, solve_picard,
};
use sonium_fem::physics::fluid::stokes::time_stepping::{BdfConfig, BdfState};
use sonium_fem::physics::fluid::stokes::workspace::BdfWorkspace;
use sonium_fem::spaces::{FunctionSpace, H1Space, VectorH1Space};
use sonium_fem::{
    BackendPreconditionerKind, BackendPressureSchurPreconditionerKind, LinearSolverBackend,
    LinearSolverCpuPreconditionerKind, LinearSolverKind,
};
use sonium_qa::analytical::{
    ghia_lid_cavity_re100_u, interp_sorted_samples, ns_taylor_green_vortex,
};
use sonium_qa::rad::oc_golden::parse_engine_out;
use sonium_qa::rad::{TimeHistory, assert_probe_close};
use std::path::{Path, PathBuf};

/// Default deck: the steady lid-driven cavity, cheaper to bring up.
const DEFAULT_DECK: &str = "ns_lid_cavity";
/// Lid-cavity Reynolds number; Ghia data anchors Re=100.
const CAVITY_RE: f64 = 100.0;
/// Lid-cavity mesh subdivisions per side (P1-P1 triangles).
///
/// 64 subdivisions give 4225 nodes (12,675 velocity+pressure dofs), a
/// solve of several seconds that keeps bring-up iteration fast. Raise
/// via `--mesh-n` once the harness is green.
const CAVITY_MESH_N: usize = 64;
/// Lid speed; Ghia errors are fractions of this scale.
const CAVITY_LID_SPEED: f64 = 1.0;
/// Ghia centerline max-error gate, matching the QA smoke binary.
///
/// The QA binary requires the n=40 error below 3% of lid speed; the
/// benchmark mesh is finer, so the same bound applies with margin.
const CAVITY_GHIA_MAX_ERROR: f64 = 0.03;
/// Taylor-Green kinematic viscosity; U=L=1 gives Re=100.
const TG_NU: f64 = 0.01;
/// Taylor-Green mesh subdivisions per side over `[0, 2pi]^2`.
///
/// 128 subdivisions give 49,923 velocity+pressure dofs and 22-second
/// rows: enough timing weight for stable medians without making each
/// benchmark row a multi-minute commitment.
const TG_MESH_N: usize = 128;
/// Taylor-Green BDF2 step size in seconds.
const TG_DT: f64 = 0.1;
/// Taylor-Green BDF2 step count; `TG_DT * TG_STEPS = 1` second.
///
/// A 1-second march keeps the frozen-boundary modeling error out of
/// the interior gate: the boundary state overstates the decayed field
/// by only 2% at t=1 (vs 9.5% at t=5), and advection carries it ~1
/// unit into the 6.28-unit domain, short of the interior mask.
const TG_STEPS: usize = 10;
/// Taylor-Green interior gate margin from each wall.
///
/// Dirichlet boundaries hold the frozen t=0 analytic state while the
/// interior decays, so nodes within this distance of a wall are
/// excluded from the accuracy gate. The diffusion length over the
/// 5-second march is `sqrt(nu * t) ~ 0.22`, well inside the margin.
const TG_INTERIOR_MARGIN: f64 = 0.5;
/// Taylor-Green interior max-velocity-error gate.
///
/// Calibrated ~1.5x over the measured frozen-boundary floor: the interior
/// error sits at 0.0163-0.0167 across mesh (n=64/128) and dt (0.1/0.05)
/// variations, so it is the frozen-boundary modeling error, not
/// discretization, and no refinement will reduce it. The gate clears
/// that floor with margin while still catching solver regressions.
const TG_MAX_VELOCITY_ERROR: f64 = 0.025;
/// Geometric tolerance for boundary DOF location on the unit cavity.
const CAVITY_GEOM_TOL: f64 = 1e-10;
/// Relative residual tolerance for each linearized solve.
///
/// 1e-8 (not the QA smoke binary's 1e-10): GMRES+ILU stalls just above
/// 1e-10 on the n>=32 Taylor-Green saddle-point systems (measured
/// 1.5e-10 after 9222 iterations), while the outer accuracy gates
/// (Ghia 3%, Taylor-Green 10%) are insensitive to linear residuals far
/// above this: Picard converges the nonlinear problem to 1e-6 relative
/// change regardless. The accuracy gates still bind every row.
const LINEAR_SOLVER_TOLERANCE: f64 = 1e-8;
// CPU preconditioner for the steady cavity rows: the steady Oseen
// matrix suits monolithic ILU(0) + GMRES. It converges every Picard
// step at n=64 and solves each step 27% faster than the block
// ILUT-Schur alternative (measured 55.6s vs 71.2s per full solve).
// The unsteady Taylor-Green rows need the Schur path instead (see the
// comment at the Taylor-Green config below).

/// Sonium linear-solve backend for the timed rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum BackendOpt {
    /// In-house BLAS-1 + ILU GMRES on the CPU.
    Cpu,
    /// CUDA GMRES with the selected GPU preconditioner.
    Cuda,
}

/// GPU preconditioner for `--backend cuda` rows (CPU rows ignore it).
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum GpuPrecondOpt {
    /// Polynomial Jacobi (baseline; weakest, cheapest per iteration).
    Poly,
    /// Chebyshev (stronger smoothers, needs eigenvalue bounds).
    Chebyshev,
    /// Block Jacobi-Schur with AMG on the pressure block (strongest).
    SchurAmg,
    /// Block Jacobi-Schur with a polynomial pressure block (the only
    /// Schur variant available to the mixed-precision inner Krylov).
    Schur,
    /// Block Jacobi-Schur with a Chebyshev pressure block.
    SchurCheb,
}

#[derive(Parser, Debug)]
#[command(
    name = "qa-oc-ns-bench",
    about = "Time sonium-vs-engine Navier-Stokes and emit a comparison bundle."
)]
struct Args {
    /// Deck id: `ns_lid_cavity` or `ns_taylor_green`.
    #[arg(long)]
    deck: Option<String>,
    /// Path for the emitted `sonium.oc-bench/v1` bundle JSON.
    #[arg(long)]
    emit_bundle: PathBuf,
    /// Which side(s) to time.
    #[arg(long, value_enum, default_value_t = SideOpt::Both)]
    side: SideOpt,
    /// Worker threads (SMP threads; MPI ranks when `--flavor mpi`
    /// runs without `--mpi-ranks`).
    #[arg(long, default_value_t = 1)]
    threads: usize,
    /// Timed repeats per row (median wall time wins).
    #[arg(long)]
    repeats: Option<u32>,
    /// CI shortcut: one repeat, no warmup.
    #[arg(long)]
    quick: bool,
    /// Run the sonium arm only (no vendor engine needed).
    #[arg(long)]
    sonium_only: bool,
    /// Run the engine arm only.
    #[arg(long)]
    engine_only: bool,
    /// Engine launch mode; defaults to serial on one thread, else SMP.
    #[arg(long, value_enum)]
    flavor: Option<FlavorOpt>,
    /// MPI ranks for `--flavor mpi`; defaults to `--threads`.
    #[arg(long)]
    mpi_ranks: Option<usize>,
    /// Engine/starter exe or exec dir; else `RADISS_EXE`/`OPENCOURANT_EXE`.
    #[arg(long)]
    exe: Option<PathBuf>,
    /// Scratch root; defaults to `SONIUM_QA_SCRATCH`, else `TMPDIR`.
    #[arg(long)]
    scratch_root: Option<PathBuf>,
    /// Keep per-repeat scratch dirs after the run.
    #[arg(long)]
    keep_scratch: bool,
    /// Sonium linear-solve backend for the timed rows.
    #[arg(long, value_enum, default_value_t = BackendOpt::Cpu)]
    backend: BackendOpt,
    /// GPU preconditioner for `--backend cuda` rows; CPU rows ignore it.
    #[arg(long, value_enum, default_value_t = GpuPrecondOpt::Poly)]
    backend_precond: GpuPrecondOpt,
    /// Route the GPU inner Krylov through f32 (FGMRES-IR); f64 outer
    /// refinement keeps f64 accuracy. CUDA rows only.
    #[arg(long)]
    gpu_mixed: bool,
    /// Lid-cavity mesh subdivisions per side; defaults to 64.
    #[arg(long)]
    mesh_n: Option<usize>,
    /// Taylor-Green mesh subdivisions per side; defaults to 128.
    #[arg(long)]
    tg_n: Option<usize>,
    /// Taylor-Green BDF step size in seconds; defaults to 0.1.
    #[arg(long)]
    tg_dt: Option<f64>,
    /// Taylor-Green BDF step count; defaults to 10.
    #[arg(long)]
    tg_steps: Option<usize>,
}

/// Maps the CLI GPU preconditioner to solver config kinds.
fn gpu_precond_kinds(
    opt: GpuPrecondOpt,
) -> (
    BackendPreconditionerKind,
    BackendPressureSchurPreconditionerKind,
) {
    match opt {
        GpuPrecondOpt::Poly => (
            BackendPreconditionerKind::PolynomialJacobi,
            BackendPressureSchurPreconditionerKind::Polynomial,
        ),
        GpuPrecondOpt::Chebyshev => (
            BackendPreconditionerKind::Chebyshev,
            BackendPressureSchurPreconditionerKind::Chebyshev,
        ),
        GpuPrecondOpt::SchurAmg => (
            BackendPreconditionerKind::JacobiSchur,
            BackendPressureSchurPreconditionerKind::Amg,
        ),
        GpuPrecondOpt::Schur => (
            BackendPreconditionerKind::JacobiSchur,
            BackendPressureSchurPreconditionerKind::Polynomial,
        ),
        GpuPrecondOpt::SchurCheb => (
            BackendPreconditionerKind::JacobiSchur,
            BackendPressureSchurPreconditionerKind::Chebyshev,
        ),
    }
}

/// Lid-driven cavity model, built once per row outside the timing.
struct CavityModel {
    mesh: Mesh,
    v_space: VectorH1Space,
    p_space: H1Space,
    config: NavierStokesConfig,
    bc_dofs: Vec<usize>,
    mesh_n: usize,
    dof: u64,
}

/// Lid-driven cavity wall function: unit lid, no-slip walls.
fn cavity_bc(x: f64, y: f64, _z: f64) -> Vec<f64> {
    let _ = x;
    if (1.0 - y) < CAVITY_GEOM_TOL {
        vec![CAVITY_LID_SPEED, 0.0]
    } else {
        vec![0.0, 0.0]
    }
}

/// Builds the cavity model: mesh, P1-P1 spaces, and solver config.
///
/// The config mirrors the `qa-fem-ns-lid-driven-cavity` smoke binary
/// (iterative FGMRES+ILU, Picard 1e-6) except the linear-solve backend,
/// which the benchmark pins explicitly: CPU rows must never silently
/// inherit an auto-selected accelerator.
fn build_cavity_model(
    mesh_n: usize,
    backend: LinearSolverBackend,
    gpu_precond: GpuPrecondOpt,
    gpu_mixed: bool,
) -> Result<CavityModel, String> {
    if mesh_n == 0 {
        return Err("--mesh-n must be positive".to_string());
    }
    let mesh = unit_square_triangles(mesh_n);
    let v_space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let p_space = H1Space::new(PolynomialDegree::P1);
    let n_v = v_space.num_global_dofs(&mesh);
    let n_p = p_space.num_global_dofs(&mesh);
    let dof = u64::try_from(n_v + n_p).map_err(|_| "cavity dof count overflows u64".to_string())?;
    let bc_dofs = locate_dofs_geometrical(&mesh, &v_space, |x, y, _z| {
        x < CAVITY_GEOM_TOL
            || (1.0 - x) < CAVITY_GEOM_TOL
            || y < CAVITY_GEOM_TOL
            || (1.0 - y) < CAVITY_GEOM_TOL
    });
    if bc_dofs.is_empty() {
        return Err("cavity boundary location found no DOFs".to_string());
    }
    let mut config = NavierStokesConfig {
        mu: 1.0 / CAVITY_RE,
        rho: 1.0,
        max_picard_iterations: 50,
        picard_tolerance: 1e-6,
        picard_relaxation: 0.7,
        use_direct_solver: false,
        linear_solver_max_iterations: 3000,
        linear_solver_tolerance: LINEAR_SOLVER_TOLERANCE,
        // UNIT5 (oc-opt Phase M): a forcing-term schedule was tried here
        // and REVERTED: loose early linear tolerances stall the
        // warm-started Picard (50 iterations without converging), so the
        // production 1e-8 every-iteration solve stands.
        linear_solver_cpu_preconditioner: LinearSolverCpuPreconditionerKind::Ilu,
        linear_solver_backend: backend,
        linear_solver_backend_preconditioner: gpu_precond_kinds(gpu_precond).0,
        linear_solver_schur_preconditioner: gpu_precond_kinds(gpu_precond).1,
        linear_solver_kind: if gpu_mixed {
            LinearSolverKind::MixedF32F64
        } else {
            LinearSolverKind::F64
        },
        ..Default::default()
    };
    // UNIT5 (oc-opt G0): optional Picard cap for Stokes-reference runs.
    // Env-gated diagnostic; unset leaves the production iteration budget.
    if let Ok(text) = std::env::var("SONIUM_UNIT5_MAX_PICARD") {
        if let Ok(cap) = text.parse::<usize>() {
            config.max_picard_iterations = cap.min(config.max_picard_iterations);
        }
    }
    Ok(CavityModel {
        mesh,
        v_space,
        p_space,
        config,
        bc_dofs,
        mesh_n,
        dof,
    })
}

/// Max centerline u error against Ghia et al. (Re=100).
///
/// Extracts u along x=0.5 from the nodal velocity field and compares
/// against the shared Ghia table via piecewise-linear interpolation.
fn cavity_ghia_error(model: &CavityModel, result: &NavierStokesResult) -> Result<f64, String> {
    let h = 1.0 / model.mesh_n as f64;
    let center_tol = h * 0.6;
    let mut centerline: Vec<(f64, f64)> = Vec::new();
    for (ni, node) in model.mesh.nodes.iter().enumerate() {
        if (node.x - 0.5).abs() < center_tol {
            let u = *result
                .velocity
                .get(2 * ni)
                .ok_or_else(|| "cavity velocity field is short".to_string())?;
            centerline.push((node.y, u));
        }
    }
    centerline.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    if centerline.len() < 2 {
        return Err("cavity centerline holds fewer than two samples".to_string());
    }
    let mut max_err = 0.0_f64;
    for (y_ref, u_ref) in ghia_lid_cavity_re100_u() {
        let u_num = interp_sorted_samples(&centerline, y_ref)
            .ok_or_else(|| format!("centerline does not bracket y={y_ref}"))?;
        max_err = max_err.max((u_num - u_ref).abs());
    }
    if !max_err.is_finite() {
        return Err("cavity Ghia error is non-finite".to_string());
    }
    // UNIT5 (oc-opt G0): env-gated centerline dump for proxy validation.
    // Prints node-order (ascending y) rows; unset prints nothing.
    if std::env::var("SONIUM_UNIT5_CENTERLINE").as_deref() == Ok("1") {
        for (ni, node) in model.mesh.nodes.iter().enumerate() {
            if (node.x - 0.5).abs() < center_tol {
                let u = result.velocity.get(2 * ni).copied().unwrap_or(f64::NAN);
                let p = result.pressure.get(ni).copied().unwrap_or(f64::NAN);
                eprintln!("UNIT5_CENTERLINE y={:.6} u={:.12e} p={:.12e}", node.y, u, p);
            }
        }
    }
    Ok(max_err)
}

/// One timed cavity repeat: steady Picard solve plus Ghia gate.
fn cavity_repeat(
    model: &CavityModel,
    want_backend: LinearSolverBackend,
) -> Result<RepeatSample, String> {
    let start = std::time::Instant::now();
    let result = solve_picard(
        &model.mesh,
        &model.v_space,
        &model.p_space,
        &model.config,
        &cavity_bc,
        &model.bc_dofs,
        |_| vec![0.0, 0.0],
    );
    let wall_s = start.elapsed().as_secs_f64();
    if result.iterations == 0 {
        return Err("cavity solve performed zero Picard iterations".to_string());
    }
    let mut failures = Vec::new();
    if !result.converged {
        failures.push(format!(
            "Picard did not converge: iterations={}, residual={:.2e}",
            result.iterations, result.residual
        ));
    }
    if result.resolved_linear_solver_backend != want_backend {
        failures.push(format!(
            "backend mismatch: requested {want_backend:?}, resolved {:?}",
            result.resolved_linear_solver_backend
        ));
    }
    match cavity_ghia_error(model, &result) {
        Ok(error) => {
            println!("cavity Ghia max error: {error:.4} (gate {CAVITY_GHIA_MAX_ERROR})");
            if error > CAVITY_GHIA_MAX_ERROR {
                failures.push(format!(
                    "Ghia max error {error:.4} exceeds {CAVITY_GHIA_MAX_ERROR}"
                ));
            }
        }
        Err(error) => failures.push(error),
    }
    for failure in &failures {
        println!("FAIL: {failure}");
    }
    println!(
        "sonium ns_lid_cavity: wall {wall_s:.3}s, picard {}, dof {}, rss {} bytes, correct {}",
        result.iterations,
        model.dof,
        self_peak_rss_bytes(),
        failures.is_empty()
    );
    Ok(RepeatSample {
        wall_s,
        cycles: u64::try_from(result.iterations)
            .map_err(|_| "Picard iteration count overflows u64".to_string())?,
        dof: model.dof,
        peak_rss_bytes: self_peak_rss_bytes(),
        correct: failures.is_empty(),
    })
}

/// Taylor-Green model, built once per row outside the timing.
struct TgModel {
    mesh: Mesh,
    v_space: VectorH1Space,
    p_space: H1Space,
    config: NavierStokesConfig,
    bdf: BdfConfig,
    workspace: BdfWorkspace,
    bc_dofs: Vec<usize>,
    velocity_coords: Vec<[f64; 3]>,
    pressure_coords: Vec<[f64; 3]>,
    interior: Vec<bool>,
    t_end: f64,
    dof: u64,
}

/// Taylor-Green frozen boundary state: analytic solution at t=0.
fn tg_bc(x: f64, y: f64, _z: f64) -> Vec<f64> {
    let (u, v, _) = ns_taylor_green_vortex(x, y, 0.0, TG_NU);
    vec![u, v]
}

/// Builds the Taylor-Green model on `[0, 2pi]^2`.
///
/// Scales a unit triangulation to the vortex period, pins a corner
/// pressure gauge (enclosed Dirichlet flow leaves pressure defined up
/// to a constant), and precomputes the interior mask for the accuracy
/// gate. The linear-solve backend is pinned explicitly per row.
fn build_tg_model(
    mesh_n: usize,
    dt: f64,
    steps: usize,
    backend: LinearSolverBackend,
    gpu_precond: GpuPrecondOpt,
    gpu_mixed: bool,
) -> Result<TgModel, String> {
    if mesh_n == 0 {
        return Err("--tg-n must be positive".to_string());
    }
    if !dt.is_finite() || dt <= 0.0 {
        return Err("--tg-dt must be a positive finite step".to_string());
    }
    if steps == 0 {
        return Err("--tg-steps must be positive".to_string());
    }
    let period = std::f64::consts::TAU;
    let mut mesh = unit_square_triangles(mesh_n);
    for node in mesh.nodes.iter_mut() {
        node.x *= period;
        node.y *= period;
    }
    let v_space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let p_space = H1Space::new(PolynomialDegree::P1);
    let n_v = v_space.num_global_dofs(&mesh);
    let n_p = p_space.num_global_dofs(&mesh);
    let dof =
        u64::try_from(n_v + n_p).map_err(|_| "Taylor-Green dof count overflows u64".to_string())?;
    // Boundary location tolerance scales with the period: node
    // coordinates are exact multiples of period/n, so 1e-9 brackets
    // rounding without admitting interior nodes.
    let geom_tol = 1e-9;
    let bc_dofs = locate_dofs_geometrical(&mesh, &v_space, |x, y, _z| {
        x < geom_tol || (period - x) < geom_tol || y < geom_tol || (period - y) < geom_tol
    });
    if bc_dofs.is_empty() {
        return Err("Taylor-Green boundary location found no DOFs".to_string());
    }
    let config = NavierStokesConfig {
        mu: TG_NU,
        rho: 1.0,
        use_direct_solver: false,
        linear_solver_max_iterations: 3000,
        linear_solver_tolerance: LINEAR_SOLVER_TOLERANCE,
        // The unsteady mass-dominated saddle system defeats monolithic
        // ILU(0): measured Arnoldi breakdown at 1.5e-8/1.5e-10 after
        // thousands of GMRES iterations on the n=32 step. The block
        // ILUT-Schur preconditioner is the production saddle-point path
        // (speaker-port fine-3D precedent): 25 iterations to 5e-9.
        linear_solver_cpu_preconditioner: LinearSolverCpuPreconditionerKind::IlutSchur,
        linear_solver_backend: backend,
        linear_solver_backend_preconditioner: gpu_precond_kinds(gpu_precond).0,
        linear_solver_schur_preconditioner: gpu_precond_kinds(gpu_precond).1,
        linear_solver_kind: if gpu_mixed {
            LinearSolverKind::MixedF32F64
        } else {
            LinearSolverKind::F64
        },
        ..Default::default()
    };
    let bdf = BdfConfig::bdf2(dt, steps);
    let mut workspace = BdfWorkspace::new(&mesh, &v_space, &p_space);
    let gauge = mesh
        .nodes
        .iter()
        .position(|node| node.x < geom_tol && node.y < geom_tol)
        .ok_or_else(|| "Taylor-Green mesh has no origin corner node".to_string())?;
    workspace.set_pressure_gauge(gauge, 0.0);
    let velocity_coords = v_space
        .dof_coordinates(&mesh)
        .ok_or_else(|| "P1 velocity space did not expose nodal coordinates".to_string())?;
    let pressure_coords = p_space
        .dof_coordinates(&mesh)
        .ok_or_else(|| "P1 pressure space did not expose nodal coordinates".to_string())?;
    if velocity_coords.len() != n_v || pressure_coords.len() != n_p {
        return Err("Taylor-Green coordinate count mismatches dof count".to_string());
    }
    let interior: Vec<bool> = velocity_coords
        .iter()
        .map(|point| {
            point[0] > TG_INTERIOR_MARGIN
                && point[0] < period - TG_INTERIOR_MARGIN
                && point[1] > TG_INTERIOR_MARGIN
                && point[1] < period - TG_INTERIOR_MARGIN
        })
        .collect();
    if !interior.iter().any(|keep| *keep) {
        return Err("Taylor-Green interior mask is empty".to_string());
    }
    Ok(TgModel {
        mesh,
        v_space,
        p_space,
        config,
        bdf,
        workspace,
        bc_dofs,
        velocity_coords,
        pressure_coords,
        interior,
        t_end: dt * steps as f64,
        dof,
    })
}

/// One timed Taylor-Green repeat: BDF march plus analytic gate.
fn tg_repeat(model: &TgModel, want_backend: LinearSolverBackend) -> Result<RepeatSample, String> {
    let start = std::time::Instant::now();
    let n_v = model.velocity_coords.len();
    let n_p = model.pressure_coords.len();
    let mut state = BdfState::from_rest(n_v, n_p);
    for (dof, point) in model.velocity_coords.iter().enumerate() {
        let (u, v, _) = ns_taylor_green_vortex(point[0], point[1], 0.0, TG_NU);
        state.velocity[dof] = if dof % 2 == 0 { u } else { v };
        state.velocity_prev[dof] = state.velocity[dof];
    }
    for (dof, point) in model.pressure_coords.iter().enumerate() {
        let (_, _, p) = ns_taylor_green_vortex(point[0], point[1], 0.0, TG_NU);
        state.pressure[dof] = p;
    }
    // Per-step backend audit: every BDF solve should resolve to the
    // requested backend. A unanimous fallback still records a row with
    // correct=false (cavity precedent); mixed backends fail the row too.
    let mut resolved_backend: Option<LinearSolverBackend> = None;
    let mut mixed_backends = false;
    let mut failed_step: Option<usize> = None;
    for step in 0..model.bdf.num_steps {
        if !solve_bdf_step_cached(
            &model.mesh,
            &model.v_space,
            &model.p_space,
            &model.config,
            &model.bdf,
            &mut state,
            &tg_bc,
            &model.bc_dofs,
            &[],
            &model.workspace,
        ) {
            // A failed case still records a correct=false row (cavity
            // precedent); only internal corruption aborts the binary.
            failed_step = Some(step);
            break;
        }
        let resolved = model
            .workspace
            .last_linear_solve_telemetry()
            .map(|telemetry| telemetry.resolved_linear_solver_backend);
        match (resolved_backend, resolved) {
            (None, Some(first)) => resolved_backend = Some(first),
            (Some(seen), Some(now)) if seen == now => {}
            _ => mixed_backends = true,
        }
    }
    let wall_s = start.elapsed().as_secs_f64();
    let mut max_err = 0.0_f64;
    for (dof, point) in model.velocity_coords.iter().enumerate() {
        if !model.interior[dof] {
            continue;
        }
        let (u_ref, v_ref, _) = ns_taylor_green_vortex(point[0], point[1], model.t_end, TG_NU);
        let expected = if dof % 2 == 0 { u_ref } else { v_ref };
        max_err = max_err.max((state.velocity[dof] - expected).abs());
    }
    if !max_err.is_finite() {
        return Err("Taylor-Green velocity error is non-finite".to_string());
    }
    let mut failures = Vec::new();
    if let Some(step) = failed_step {
        failures.push(format!("Taylor-Green BDF solve failed at step {step}"));
    }
    if mixed_backends {
        failures.push(format!(
            "mixed linear-solver backends across BDF steps (requested {want_backend:?})"
        ));
    } else if resolved_backend != Some(want_backend) {
        failures.push(format!(
            "backend mismatch: requested {want_backend:?}, resolved {resolved_backend:?}"
        ));
    }
    println!("Taylor-Green interior max error: {max_err:.4} (gate {TG_MAX_VELOCITY_ERROR})");
    if max_err > TG_MAX_VELOCITY_ERROR {
        failures.push(format!(
            "interior max error {max_err:.4} exceeds {TG_MAX_VELOCITY_ERROR}"
        ));
    }
    for failure in &failures {
        println!("FAIL: {failure}");
    }
    println!(
        "sonium ns_taylor_green: wall {wall_s:.3}s, steps {}, dof {}, resolved {resolved_backend:?}, rss {} bytes, correct {}",
        model.bdf.num_steps,
        model.dof,
        self_peak_rss_bytes(),
        failures.is_empty()
    );
    Ok(RepeatSample {
        wall_s,
        cycles: u64::try_from(model.bdf.num_steps)
            .map_err(|_| "BDF step count overflows u64".to_string())?,
        dof: model.dof,
        peak_rss_bytes: self_peak_rss_bytes(),
        correct: failures.is_empty(),
    })
}

/// Runs the sonium arm over warmup plus timed repeats.
///
/// The model is built once per row outside the timing; each repeat
/// re-solves from the same initial state so iteration counts stay
/// deterministic across repeats.
fn run_sonium_arm(args: &Args, deck: &str, repeats: u32, warmup: u32) -> Result<BenchRow, String> {
    // Safety: no other thread reads the environment concurrently at
    // this point (single-threaded startup before any spawn).
    unsafe {
        std::env::set_var("RAYON_NUM_THREADS", args.threads.max(1).to_string());
    }
    let backend = match args.backend {
        BackendOpt::Cpu => LinearSolverBackend::Cpu,
        BackendOpt::Cuda => LinearSolverBackend::Cuda,
    };
    let flavor = match args.backend {
        BackendOpt::Cpu => "native",
        BackendOpt::Cuda => "cuda",
    };
    let mut samples = Vec::new();
    match deck {
        "ns_lid_cavity" => {
            let model = build_cavity_model(
                args.mesh_n.unwrap_or(CAVITY_MESH_N),
                backend,
                args.backend_precond,
                args.gpu_mixed,
            )?;
            println!("sonium {deck}: dof {}, backend {backend:?}", model.dof,);
            for warmup_index in 0..warmup {
                println!("sonium {deck}: warmup {}/{}", warmup_index + 1, warmup);
                let _ = cavity_repeat(&model, backend)?;
            }
            for repeat in 0..repeats {
                println!("sonium {deck}: repeat {}/{}", repeat + 1, repeats);
                samples.push(cavity_repeat(&model, backend)?);
            }
        }
        "ns_taylor_green" => {
            let model = build_tg_model(
                args.tg_n.unwrap_or(TG_MESH_N),
                args.tg_dt.unwrap_or(TG_DT),
                args.tg_steps.unwrap_or(TG_STEPS),
                backend,
                args.backend_precond,
                args.gpu_mixed,
            )?;
            println!(
                "sonium {deck}: dof {}, steps {}, backend {backend:?}",
                model.dof, model.bdf.num_steps,
            );
            for warmup_index in 0..warmup {
                println!("sonium {deck}: warmup {}/{}", warmup_index + 1, warmup);
                let _ = tg_repeat(&model, backend)?;
            }
            for repeat in 0..repeats {
                println!("sonium {deck}: repeat {}/{}", repeat + 1, repeats);
                samples.push(tg_repeat(&model, backend)?);
            }
        }
        _ => return Err(format!("sonium arm supports no deck {deck:?}")),
    }
    let total = aggregate_repeats(&samples)?;
    BenchRow::measured(
        deck,
        Side::Sonium,
        flavor,
        u32::try_from(args.threads.max(1)).map_err(|_| "thread count overflows u32".to_string())?,
        total.wall_s,
        total.cycles,
        total.dof,
        total.peak_rss_bytes,
        total.correct,
    )
    .map_err(|error| error.to_string())
}

/// Checks NS probes against TH channel values.
///
/// Each `probes.json` expectation carries its sampling rule: settled
/// steady-flow channels read the final TH value, while ringing
/// transients average the settled second half.
fn check_ns_probes(history: &TimeHistory, probes: &[Probe]) -> Vec<String> {
    let mut failures = Vec::new();
    for probe in probes {
        let sampled = match probe.rule {
            SampleRule::Final => history
                .final_value(&probe.channel)
                .map_err(|error| error.to_string()),
            SampleRule::Plateau => plateau_mean(history, &probe.channel),
        };
        match sampled {
            Ok(value) => {
                println!(
                    "probe {} [{}]: actual {value:.6e}, expected {:.6e}",
                    probe.label, probe.channel, probe.expected
                );
                if let Err(error) = assert_probe_close(
                    value,
                    probe.expected,
                    probe.abs_tol,
                    probe.rel_tol,
                    &probe.label,
                ) {
                    failures.push(error.to_string());
                }
            }
            Err(error) => failures.push(format!("probe {} unreadable: {error}", probe.label)),
        }
    }
    failures
}

/// One timed NS engine repeat: stage, starter, timed engine, probes.
fn engine_ns_repeat(ctx: &EngineCtx, deck: &str, index: usize) -> Result<RepeatSample, String> {
    let dir = repeat_scratch(&ctx.scratch_root, deck, "engine", index);
    if dir.exists() {
        std::fs::remove_dir_all(&dir)
            .map_err(|error| format!("cannot clean {}: {error}", dir.display()))?;
    }
    std::fs::create_dir_all(&dir)
        .map_err(|error| format!("cannot create {}: {error}", dir.display()))?;
    let outcome = engine_ns_repeat_inner(ctx, deck, &dir);
    match &outcome {
        Ok(_) => maybe_clean(&dir, ctx.keep_scratch),
        Err(_) => println!("note: engine scratch kept at {}", dir.display()),
    }
    outcome
}

/// Inner NS repeat so failures keep scratch for forensics.
fn engine_ns_repeat_inner(ctx: &EngineCtx, deck: &str, dir: &Path) -> Result<RepeatSample, String> {
    let decks = resolve_ns_decks(deck)?;
    let probes = read_probe_file(&decks.probes)?;
    let starter_name = format!("{deck}_0000.rad");
    let engine_name = format!("{deck}_0001.rad");
    std::fs::copy(&decks.starter, dir.join(&starter_name))
        .map_err(|error| format!("cannot stage starter deck: {error}"))?;
    std::fs::copy(&decks.engine, dir.join(&engine_name))
        .map_err(|error| format!("cannot stage engine deck: {error}"))?;
    let engine_text = std::fs::read_to_string(dir.join(&engine_name))
        .map_err(|error| format!("cannot read staged engine deck: {error}"))?;
    let rootname = run_rootname(&engine_text)?;
    let starter_np = (ctx.launch.flavor == FlavorOpt::Mpi).then_some(ctx.launch.mpi_ranks);
    run_vendor_stage(
        ctx,
        &ctx.starter_exe,
        &starter_argv(&starter_name, starter_np),
        dir,
        "starter",
        false,
    )?;
    let (exe, argv) = engine_argv(&ctx.launch, &ctx.engine_exe, &engine_name);
    let timed = run_vendor_stage(ctx, &exe, &argv, dir, "engine", true)?;
    let engine_out = first_existing(dir, &out_candidates(&rootname, 1))?;
    let (_scalars, normal_termination) =
        parse_engine_out(&engine_out).map_err(|error| error.to_string())?;
    if !normal_termination {
        return Err(format!(
            "engine run ended without NORMAL TERMINATION in {}",
            engine_out.display()
        ));
    }
    let cycles = total_cycles(std::slice::from_ref(&engine_out))?;
    let t01 = first_existing(dir, &th_candidates(&rootname, 1))?;
    let history = load_th_history(&ctx.engine_exe, &t01, dir)?;
    let failures = check_ns_probes(&history, &probes);
    for failure in &failures {
        println!("FAIL: {failure}");
    }
    let dof = deck_dof(&dir.join(&starter_name))?;
    println!(
        "engine {deck}: wall {:.3}s, cycles {cycles}, dof {dof}, rss {} bytes, correct {}",
        timed.wall_s,
        timed.peak_rss_bytes,
        failures.is_empty()
    );
    Ok(RepeatSample {
        wall_s: timed.wall_s,
        cycles,
        dof,
        peak_rss_bytes: timed.peak_rss_bytes,
        correct: failures.is_empty(),
    })
}

/// Runs the engine arm over warmup plus timed repeats.
fn run_engine_arm(args: &Args, deck: &str, repeats: u32, warmup: u32) -> Result<BenchRow, String> {
    if deck == "ns_taylor_green" {
        return Err(
            "engine arm supports ns_lid_cavity only; Taylor-Green needs a spatially-varying \
             initial velocity and OpenCourant offers only uniform-per-group /INIVEL"
                .to_string(),
        );
    }
    let ctx = engine_context(
        args.exe.clone(),
        args.threads.max(1),
        args.flavor,
        args.mpi_ranks,
        args.scratch_root.clone(),
        args.keep_scratch,
    )?;
    resolve_ns_decks(deck)?;
    let mut samples = Vec::new();
    for warmup_index in 0..warmup {
        println!("engine {deck}: warmup {}/{}", warmup_index + 1, warmup);
        let _ = engine_ns_repeat(&ctx, deck, warmup_index as usize)?;
    }
    for repeat in 0..repeats {
        println!("engine {deck}: repeat {}/{}", repeat + 1, repeats);
        samples.push(engine_ns_repeat(&ctx, deck, (warmup + repeat) as usize)?);
    }
    let total = aggregate_repeats(&samples)?;
    BenchRow::measured(
        deck,
        Side::Engine,
        &ctx.launch.label,
        u32::try_from(ctx.launch.row_threads)
            .map_err(|_| "thread count overflows u32".to_string())?,
        total.wall_s,
        total.cycles,
        total.dof,
        total.peak_rss_bytes,
        total.correct,
    )
    .map_err(|error| error.to_string())
}

/// Binding NS verdict: time-to-matched-accuracy per (deck, threads) group.
///
/// Solver steps differ across arms (explicit engine cycles vs Picard
/// iterations vs BDF steps), so the per-step `ns_per_dof_cycle` ratio
/// is diagnostic only. The winner is the faster wall time when both
/// arms pass their accuracy probes; a failed arm never wins.
fn render_ns_verdict(bundle: &BenchBundle) -> String {
    let mut report = String::from("\n## NS verdict (time-to-matched-accuracy)\n\n");
    report.push_str("| deck | threads | sonium wall_s | engine wall_s | speedup (sonium vs engine) | winner |\n");
    report.push_str("| --- | --- | --- | --- | --- | --- |\n");
    let mut groups: Vec<(String, u32)> = Vec::new();
    for row in &bundle.rows {
        let key = (row.deck.clone(), row.threads);
        if !groups.contains(&key) {
            groups.push(key);
        }
    }
    let mut any = false;
    for (deck, threads) in groups {
        let sonium = bundle
            .rows
            .iter()
            .find(|row| row.deck == deck && row.threads == threads && row.side == Side::Sonium);
        let engine = bundle
            .rows
            .iter()
            .find(|row| row.deck == deck && row.threads == threads && row.side == Side::Engine);
        let (Some(sonium), Some(engine)) = (sonium, engine) else {
            continue;
        };
        any = true;
        let speedup = engine.wall_s / sonium.wall_s;
        let winner = match (sonium.correctness_passed, engine.correctness_passed) {
            (true, true) if speedup > 1.0 => "Sonium",
            (true, true) if speedup < 1.0 => "Engine",
            (true, false) => "Sonium",
            (false, true) => "Engine",
            _ => "Neither",
        };
        report.push_str(&format!(
            "| {deck} | {threads} | {:.4} | {:.4} | {speedup:.3}x | {winner} |\n",
            sonium.wall_s, engine.wall_s,
        ));
    }
    if !any {
        report.push_str("No (deck, threads) group holds both sides; nothing to compare.\n");
    }
    report
}

fn run() -> Result<(), String> {
    let args = Args::parse();
    let deck = args
        .deck
        .clone()
        .unwrap_or_else(|| DEFAULT_DECK.to_string());
    if deck.trim().is_empty() {
        return Err("--deck must not be empty".to_string());
    }
    let (run_sonium, run_engine) = resolve_sides(args.sonium_only, args.engine_only, args.side)?;
    let (repeats, warmup) = resolve_repeats(args.repeats, args.quick)?;
    if args.threads == 0 {
        return Err("--threads must be positive".to_string());
    }
    let mut rows = Vec::new();
    // Engine first: forked vendor children inherit this process's RSS
    // hiwater, so the engine arm must run before the allocation-heavy
    // sonium solves (measured: engine reported the sonium 306 MB peak
    // instead of its own 49 MB when ordered second).
    if run_engine {
        rows.push(run_engine_arm(&args, &deck, repeats, warmup)?);
    }
    if run_sonium {
        rows.push(run_sonium_arm(&args, &deck, repeats, warmup)?);
    }
    let bundle = BenchBundle {
        schema: BUNDLE_SCHEMA.to_string(),
        hardware: HardwareInfo {
            arch: std::env::consts::ARCH.to_string(),
            model: cpu_model(),
            threads: args.threads.max(1).to_string(),
        },
        protocol: ProtocolInfo {
            repeats,
            warmup,
            correctness: PROTOCOL_CORRECTNESS.to_string(),
        },
        rows,
    };
    let text = bundle_json(&bundle).map_err(|error| error.to_string())?;
    if let Some(parent) = args
        .emit_bundle
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
    }
    std::fs::write(&args.emit_bundle, &text)
        .map_err(|error| format!("cannot write {}: {error}", args.emit_bundle.display()))?;
    println!("wrote {}", args.emit_bundle.display());
    let mut report = render_bundle_markdown(&bundle).map_err(|error| error.to_string())?;
    report.push_str(&render_ns_verdict(&bundle));
    println!("{report}");
    if bundle.rows.iter().all(|row| row.correctness_passed) {
        Ok(())
    } else {
        Err("one or more rows failed their correctness probes".to_string())
    }
}

fn main() {
    match run() {
        Ok(()) => println!("SUCCESS: qa-oc-ns-bench"),
        Err(error) => {
            eprintln!("error: {error}");
            println!("FAILED: qa-oc-ns-bench");
            std::process::exit(1);
        }
    }
}
