//! Wall-clock benchmark across CPU, CUDA, or Metal FEM preconditioners.
//!
//! Runs the same FEM frequency sweep once per selected preconditioner and
//! prints a sorted wall-clock leaderboard plus a solution-agreement check
//! against the first successful run at each physical damping value.
//!
//! The whole point is to settle "is AMG always faster on indefinite
//! Helmholtz?" with data instead of folklore. Run it across a few scenarios
//! and frequency ranges — preconditioner ranking is *strongly* problem-size
//! and frequency dependent.
//!
//! Usage:
//! ```text
//! cargo run --release -p sonium-sim-room --bin bench-fem-preconditioners -- \
//!     --scenario small_stereo_2_2 --max-freq 150 --num-freqs 15
//! ```

use anyhow::Result;
use clap::Parser;
use serde_json::json;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    time::Instant,
};

use sonium_sim_room::fem_runner::{
    self, FemFrequencyStats, FemMeshConfig, FemPreconditionerKind, FemRunDiagnostics, RecyclerKind,
    ShiftedLaplacianShift,
};
use sonium_sim_room::scenarios::{self, FrequencyConfig};

const ALL_PRECONDITIONERS: &[FemPreconditionerKind] = &[
    FemPreconditionerKind::Ilut,
    FemPreconditionerKind::BlockJacobi,
    FemPreconditionerKind::AdditiveSchwarz,
    FemPreconditionerKind::SweepingSchwarz,
    FemPreconditionerKind::IluColoring,
    FemPreconditionerKind::IluFixedPoint,
    FemPreconditionerKind::Amg,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchBackend {
    Cpu,
    Cuda,
    Metal,
}

impl BenchBackend {
    fn label(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum BenchPreconditioner {
    Cpu(FemPreconditionerKind),
    #[cfg(any(feature = "cuda", feature = "metal"))]
    Gpu(fem_runner::GpuPrecond),
}

impl BenchPreconditioner {
    fn label(self) -> String {
        match self {
            Self::Cpu(kind) => kind.to_string(),
            #[cfg(any(feature = "cuda", feature = "metal"))]
            Self::Gpu(kind) => kind.label(),
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "bench-fem-preconditioners",
    about = "Compare every FEM preconditioner on the same Helmholtz sweep."
)]
struct Args {
    /// Total Rayon worker budget. Zero uses Rayon's automatic default.
    #[arg(long, visible_alias = "cores", default_value_t = 0)]
    threads: usize,
    /// Scenario name. Defaults to small_stereo_2_0 — small enough to bench
    /// interactively, big enough to actually exercise the preconditioner.
    #[arg(long, default_value = "small_stereo_2_0")]
    scenario: String,
    /// Lowest frequency in the sweep (Hz).
    #[arg(long, default_value_t = 30.0)]
    min_freq: f64,
    /// Highest frequency in the sweep (Hz). Mesh DOFs scale as f^3, so keep
    /// this modest for interactive benches — 100–150 Hz is a good range.
    #[arg(long, default_value_t = 150.0)]
    max_freq: f64,
    /// Number of frequencies in the sweep.
    #[arg(long, default_value_t = 15)]
    num_freqs: usize,
    /// Backend under test: `cpu`, `cuda`, or `metal`. Device backends require
    /// their corresponding crate feature.
    #[arg(long, default_value = "cpu")]
    backend: String,
    /// Optional comma-separated subset, e.g. `ilut,amg,block-jacobi`. Defaults
    /// to all seven. Used only with `--backend cpu`.
    #[arg(long)]
    preconditioners: Option<String>,
    /// Optional comma-separated GPU preconditioner subset, e.g.
    /// `auto,chebyshev-8,poly-jacobi-4,point-jacobi`. Used only with
    /// `--backend cuda` or `--backend metal`.
    #[arg(long)]
    gpu_preconditioners: Option<String>,
    /// GPU precision mode. Metal currently supports only `f64-outer`.
    #[arg(long, default_value = "f64-outer")]
    gpu_precision: String,
    /// Recycled-subspace dimension. 0 = no recycling, so the raw
    /// preconditioner cost dominates. Defaults to the production value
    /// (`DEFAULT_RECYCLE_SUBSPACE_SIZE`) so the bench mirrors what the
    /// data-gen sweep actually runs.
    #[arg(long, default_value_t = fem_runner::DEFAULT_RECYCLE_SUBSPACE_SIZE)]
    recycle_size: usize,
    /// Comma-separated list of `rebuild-every-N` values to sweep. The bench
    /// runs the full preconditioner comparison once per N and prints a
    /// precond × N wall-clock grid, so a single invocation answers "how much
    /// does amortising ILUT/AMG setup buy us?". `1` is exact (rebuild every
    /// freq); larger N reuses the factorisation for N adjacent frequencies.
    #[arg(long, value_delimiter = ',', default_values_t = [1usize], value_name = "LIST")]
    rebuild_every: Vec<usize>,
    /// Comma-separated shifted-Laplacian shift families to sweep:
    /// `legacy` keeps the existing 0.5*k^2 + i*0.5*k shift; `k2` sweeps
    /// `+(alpha + i beta)*k^2`; `csl` sweeps the conventional
    /// `-(alpha + i beta)*k^2` using --shift-alpha and --shift-beta.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "legacy",
        value_name = "LIST"
    )]
    shift: Vec<String>,
    /// Comma-separated alpha values for `--shift k2`.
    #[arg(long, value_delimiter = ',', default_values_t = [0.5_f64], value_name = "LIST")]
    shift_alpha: Vec<f64>,
    /// Comma-separated beta values for `--shift k2`.
    #[arg(long, value_delimiter = ',', default_values_t = [0.5_f64], value_name = "LIST")]
    shift_beta: Vec<f64>,
    /// Comma-separated modal damping ratios to sweep.
    #[arg(
        long,
        value_delimiter = ',',
        default_values_t = [fem_runner::DEFAULT_MODAL_DAMPING_RATIO],
        value_name = "LIST"
    )]
    modal_damping: Vec<f64>,
    /// Optional path to write per-run diagnostics as JSON.
    #[arg(long)]
    json_out: Option<PathBuf>,
}

struct RunResult {
    backend: BenchBackend,
    preconditioner: String,
    rebuild_every: usize,
    shift: ShiftedLaplacianShift,
    modal_damping_ratio: f64,
    wall_clock_s: Option<f64>,
    output: Option<sonium_sim_room::SimulationOutput>,
    diagnostics: Option<FemRunDiagnostics>,
    error: Option<String>,
}

fn parse_preconditioners(arg: &Option<String>) -> Result<Vec<FemPreconditionerKind>> {
    let Some(list) = arg else {
        return Ok(ALL_PRECONDITIONERS.to_vec());
    };
    list.split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<FemPreconditionerKind>()
                .map_err(anyhow::Error::msg)
        })
        .collect()
}

fn parse_backend(value: &str) -> Result<BenchBackend> {
    match value.trim().to_ascii_lowercase().replace('_', "-").as_str() {
        "cpu" => Ok(BenchBackend::Cpu),
        "cuda" => Ok(BenchBackend::Cuda),
        "metal" => Ok(BenchBackend::Metal),
        other => anyhow::bail!("unknown --backend '{other}'. Valid values: cpu, cuda, metal"),
    }
}

fn parse_bench_preconditioners(
    backend: BenchBackend,
    args: &Args,
) -> Result<Vec<BenchPreconditioner>> {
    match backend {
        BenchBackend::Cpu => parse_preconditioners(&args.preconditioners).map(|kinds| {
            kinds
                .into_iter()
                .map(BenchPreconditioner::Cpu)
                .collect::<Vec<_>>()
        }),
        BenchBackend::Cuda | BenchBackend::Metal => {
            parse_gpu_preconditioners(backend, &args.gpu_preconditioners)
        }
    }
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn parse_gpu_preconditioners(
    backend: BenchBackend,
    arg: &Option<String>,
) -> Result<Vec<BenchPreconditioner>> {
    anyhow::ensure!(
        !matches!(backend, BenchBackend::Cuda) || cfg!(feature = "cuda"),
        "--backend cuda requires building sonium-sim-room with `--features cuda`"
    );
    anyhow::ensure!(
        !matches!(backend, BenchBackend::Metal) || cfg!(feature = "metal"),
        "--backend metal requires building sonium-sim-room with `--features metal`"
    );
    let specs = arg.as_deref().unwrap_or("auto");
    specs
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            fem_runner::GpuPrecond::parse(s)
                .map(BenchPreconditioner::Gpu)
                .map_err(anyhow::Error::msg)
        })
        .collect()
}

#[cfg(not(any(feature = "cuda", feature = "metal")))]
fn parse_gpu_preconditioners(
    backend: BenchBackend,
    _arg: &Option<String>,
) -> Result<Vec<BenchPreconditioner>> {
    anyhow::bail!(
        "--backend {} requires the corresponding sonium-sim-room feature",
        backend.label()
    )
}

fn parse_shift_cases(args: &Args) -> Result<Vec<ShiftedLaplacianShift>> {
    anyhow::ensure!(
        !args.shift_alpha.is_empty(),
        "--shift-alpha cannot be empty"
    );
    anyhow::ensure!(!args.shift_beta.is_empty(), "--shift-beta cannot be empty");
    for &alpha in &args.shift_alpha {
        anyhow::ensure!(
            alpha.is_finite() && alpha >= 0.0,
            "--shift-alpha values must be finite and non-negative"
        );
    }
    for &beta in &args.shift_beta {
        anyhow::ensure!(
            beta.is_finite() && beta >= 0.0,
            "--shift-beta values must be finite and non-negative"
        );
    }

    let mut shifts = Vec::new();
    for raw in &args.shift {
        match raw.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "legacy" => shifts.push(ShiftedLaplacianShift::Legacy),
            "k2" => {
                for &alpha in &args.shift_alpha {
                    for &beta in &args.shift_beta {
                        shifts.push(ShiftedLaplacianShift::K2 { alpha, beta });
                    }
                }
            }
            "csl" => {
                for &alpha in &args.shift_alpha {
                    for &beta in &args.shift_beta {
                        shifts.push(ShiftedLaplacianShift::Csl { alpha, beta });
                    }
                }
            }
            other => anyhow::bail!("unknown --shift '{other}'. Valid values: legacy,k2,csl"),
        }
    }

    let mut deduped = Vec::new();
    for shift in shifts {
        if !deduped
            .iter()
            .any(|existing: &ShiftedLaplacianShift| existing.label() == shift.label())
        {
            deduped.push(shift);
        }
    }
    anyhow::ensure!(!deduped.is_empty(), "no shifted-Laplacian shifts selected");
    Ok(deduped)
}

fn normalized_modal_damping(values: &[f64]) -> Result<Vec<f64>> {
    anyhow::ensure!(!values.is_empty(), "--modal-damping cannot be empty");
    let mut values = values.to_vec();
    for &value in &values {
        anyhow::ensure!(
            value.is_finite() && value >= 0.0,
            "--modal-damping values must be finite and non-negative"
        );
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values.dedup_by(|a, b| (*a - *b).abs() <= f64::EPSILON);
    Ok(values)
}

fn p90_pressure(output: &sonium_sim_room::SimulationOutput) -> f64 {
    let mut mags: Vec<f64> = output
        .pressures
        .iter()
        .flatten()
        .flatten()
        .map(|p| p.norm())
        .collect();
    mags.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    mags.get((mags.len() as f64 * 0.9) as usize)
        .copied()
        .unwrap_or(1.0)
        .max(1e-12)
}

fn agreement(
    reference: &sonium_sim_room::SimulationOutput,
    candidate: &sonium_sim_room::SimulationOutput,
) -> (f64, f64, f64) {
    let p90 = p90_pressure(reference);
    let mut max_abs = 0.0_f64;
    let mut sumsq_diff = 0.0_f64;
    let mut sumsq_ref = 0.0_f64;
    for (po, pn) in reference.pressures.iter().zip(candidate.pressures.iter()) {
        for (l_off, l_on) in po.iter().zip(pn.iter()) {
            for (a, b) in l_off.iter().zip(l_on.iter()) {
                let abs = (a - b).norm();
                if abs.is_nan() || abs > max_abs {
                    max_abs = abs;
                }
                sumsq_diff += abs * abs;
                sumsq_ref += a.norm_sqr();
            }
        }
    }
    let l2_rel = sumsq_diff.sqrt() / sumsq_ref.sqrt().max(1e-30);
    (max_abs, max_abs / p90, l2_rel)
}

fn agreement_reference_indices(results: &[RunResult]) -> Vec<Option<usize>> {
    let mut references = HashMap::<u64, usize>::new();
    for (index, result) in results.iter().enumerate() {
        if result.wall_clock_s.is_some() {
            references
                .entry(result.modal_damping_ratio.to_bits())
                .or_insert(index);
        }
    }
    results
        .iter()
        .map(|result| {
            references
                .get(&result.modal_damping_ratio.to_bits())
                .copied()
        })
        .collect()
}

fn frequency_stats_json(stats: &FemFrequencyStats) -> serde_json::Value {
    json!({
        "frequency_hz": stats.frequency_hz,
        "symmetry_mode": stats.symmetry_mode,
        "matrix_ms": stats.matrix_ms,
        "preconditioner_ms": stats.preconditioner_ms,
        "rhs_ms": stats.rhs_ms,
        "solve_ms": stats.solve_ms,
        "solve_ms_semantics": "summed_source_work",
        "iterations": stats.iterations,
        "inner_iterations": stats.inner_iterations,
        "retries": stats.retries,
        "preconditioner_age": stats.preconditioner_age,
        "preconditioner_rebuild_reason": stats.preconditioner_rebuild_reason.as_str(),
        "fallback_stage": stats.fallback_stage,
        "max_residual": stats.max_residual,
        "converged": stats.converged,
    })
}

fn write_json(path: &Path, args: &Args, results: &[RunResult]) -> Result<()> {
    let runs = results
        .iter()
        .map(|r| {
            let diagnostics = r.diagnostics.as_ref().map(|d| {
                json!({
                    "backend": d.backend.as_str(),
                    "preconditioner": d.preconditioner.as_str(),
                    "shift": d.shift.label(),
                    "modal_damping_ratio": d.modal_damping_ratio,
                    "total_iterations": d.total_iterations,
                    "total_inner_iterations": d.total_inner_iterations,
                    "retry_count": d.retry_count,
                    "nonconverged_count": d.nonconverged_count,
                    "max_residual": d.max_residual,
                    "frequency_stats": d.frequency_stats.iter().map(frequency_stats_json).collect::<Vec<_>>(),
                })
            });
            json!({
                "backend": r.backend.label(),
                "preconditioner": r.preconditioner.as_str(),
                "rebuild_every": r.rebuild_every,
                "shift": r.shift.label(),
                "modal_damping_ratio": r.modal_damping_ratio,
                "wall_clock_s": r.wall_clock_s,
                "error": r.error.as_deref(),
                "diagnostics": diagnostics,
            })
        })
        .collect::<Vec<_>>();

    let doc = json!({
        "scenario": args.scenario,
        "backend": args.backend.as_str(),
        "min_freq": args.min_freq,
        "max_freq": args.max_freq,
        "num_freqs": args.num_freqs,
        "runs": runs,
    });
    std::fs::write(path, serde_json::to_vec_pretty(&doc)?)?;
    Ok(())
}

fn main() -> Result<()> {
    // Default to warn so per-frequency progress lines don't drown the bench
    // output. Set RUST_LOG=info to see the FEM runner's iteration counts.
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let args = Args::parse();
    sonium_sim_room::runtime::install_fem_rayon_pool(args.threads)?;

    let backend = parse_backend(&args.backend)?;
    let preconditioners = parse_bench_preconditioners(backend, &args)?;
    anyhow::ensure!(!preconditioners.is_empty(), "no preconditioners selected");
    let shifts = parse_shift_cases(&args)?;
    let modal_damping = normalized_modal_damping(&args.modal_damping)?;

    let mut rebuild_ns: Vec<usize> = args
        .rebuild_every
        .iter()
        .copied()
        .map(|n| n.max(1))
        .collect();
    rebuild_ns.sort_unstable();
    rebuild_ns.dedup();
    anyhow::ensure!(!rebuild_ns.is_empty(), "no rebuild-every values selected");

    let freq_config = FrequencyConfig {
        min_freq: args.min_freq,
        max_freq: args.max_freq,
        num_points: args.num_freqs,
        spacing: sonium_xem::FrequencySpacing::default(),
    };
    let scenario = scenarios::scenario_by_name_with_config(&args.scenario, &freq_config)
        .ok_or_else(|| anyhow::anyhow!("scenario not found: {}", args.scenario))?;

    println!("Scenario: {} ({})", scenario.name, scenario.description);
    println!(
        "Sweep: {} Hz → {} Hz, {} points; backend: {}; recycle-size: k={}",
        args.min_freq,
        args.max_freq,
        args.num_freqs,
        backend.label(),
        args.recycle_size
    );
    println!(
        "Preconditioners under test ({}): {}",
        preconditioners.len(),
        preconditioners
            .iter()
            .map(|k| k.label())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "rebuild-every values under test ({}): {}",
        rebuild_ns.len(),
        rebuild_ns
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "shift values under test ({}): {}",
        shifts.len(),
        shifts
            .iter()
            .map(|s| s.label())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "modal damping values under test ({}): {}",
        modal_damping.len(),
        modal_damping
            .iter()
            .map(|d| format!("{d:.4}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!();

    let mut results: Vec<RunResult> = Vec::with_capacity(
        preconditioners.len() * rebuild_ns.len() * shifts.len() * modal_damping.len(),
    );
    for &preconditioner in &preconditioners {
        let preconditioner_label = preconditioner.label();
        for &n in &rebuild_ns {
            for &shift in &shifts {
                for &damping in &modal_damping {
                    let mut cfg = FemMeshConfig {
                        recycle_subspace_size: args.recycle_size,
                        recycler: RecyclerKind::GcroDr,
                        rebuild_precond_every: n,
                        modal_damping_ratio: damping,
                        shifted_laplacian_shift: shift,
                        ..FemMeshConfig::default()
                    };
                    match preconditioner {
                        BenchPreconditioner::Cpu(kind) => {
                            cfg.preconditioner = kind;
                        }
                        #[cfg(any(feature = "cuda", feature = "metal"))]
                        BenchPreconditioner::Gpu(_) => {}
                    }

                    print!(
                        "Running {:>17} rebuild-every={n:>3} shift={:<28} damping={damping:.4} … ",
                        preconditioner_label,
                        shift.label()
                    );
                    std::io::Write::flush(&mut std::io::stdout()).ok();

                    let t0 = Instant::now();
                    let run_result = match (backend, preconditioner) {
                        (BenchBackend::Cpu, BenchPreconditioner::Cpu(_)) => {
                            fem_runner::run_fem_with_mesh_report(&scenario.simulation, &cfg)
                        }
                        #[cfg(feature = "cuda")]
                        (BenchBackend::Cuda, BenchPreconditioner::Gpu(gpu_precond)) => {
                            let precision = fem_runner::GpuPrecision::parse(&args.gpu_precision)
                                .map_err(anyhow::Error::msg)?;
                            fem_runner::run_fem_gpu_report(
                                &scenario.simulation,
                                &cfg,
                                precision,
                                gpu_precond,
                                0,
                            )
                        }
                        #[cfg(feature = "metal")]
                        (BenchBackend::Metal, BenchPreconditioner::Gpu(gpu_precond)) => {
                            let precision = fem_runner::GpuPrecision::parse(&args.gpu_precision)
                                .map_err(anyhow::Error::msg)?;
                            fem_runner::run_fem_metal_report(
                                &scenario.simulation,
                                &cfg,
                                precision,
                                gpu_precond,
                            )
                        }
                        _ => unreachable!("backend/preconditioner admission mismatch"),
                    };

                    match run_result {
                        Ok((output, diagnostics)) => {
                            let wall_clock_s = t0.elapsed().as_secs_f64();
                            println!(
                                "{wall_clock_s:>7.2}s iters={} inner={} retries={} max_res={:.2e}",
                                diagnostics.total_iterations,
                                diagnostics.total_inner_iterations,
                                diagnostics.retry_count,
                                diagnostics.max_residual,
                            );
                            results.push(RunResult {
                                backend,
                                preconditioner: preconditioner_label.clone(),
                                rebuild_every: n,
                                shift,
                                modal_damping_ratio: damping,
                                wall_clock_s: Some(wall_clock_s),
                                output: Some(output),
                                diagnostics: Some(diagnostics),
                                error: None,
                            });
                        }
                        Err(e) => {
                            println!("FAILED: {e}");
                            results.push(RunResult {
                                backend,
                                preconditioner: preconditioner_label.clone(),
                                rebuild_every: n,
                                shift,
                                modal_damping_ratio: damping,
                                wall_clock_s: None,
                                output: None,
                                diagnostics: None,
                                error: Some(e.to_string()),
                            });
                        }
                    }
                }
            }
        }
    }

    let ran_any = results.iter().any(|r| r.wall_clock_s.is_some());
    anyhow::ensure!(
        ran_any,
        "no (preconditioner, rebuild-every) combination finished"
    );

    println!();
    if shifts.len() == 1 && modal_damping.len() == 1 {
        // Wall-clock grid: rows = preconditioner, cols = rebuild_every. Each
        // cell is the seconds for that combination, or 'FAILED' on failure.
        println!("Wall-clock grid (seconds; rows = preconditioner, cols = rebuild-every):");
        print!("  {:<17}", "preconditioner");
        for &n in &rebuild_ns {
            print!("  {:>9}", format!("N={n}"));
        }
        println!();
        println!("  {}", "─".repeat(17 + 2 + rebuild_ns.len() * 11));
        for preconditioner in &preconditioners {
            let label = preconditioner.label();
            print!("  {:<17}", label);
            for &n in &rebuild_ns {
                let cell = results
                    .iter()
                    .find(|r| r.preconditioner == label && r.rebuild_every == n)
                    .and_then(|r| r.wall_clock_s);
                match cell {
                    Some(s) => print!("  {s:>9.2}"),
                    None => print!("  {:>9}", "FAILED"),
                }
            }
            println!();
        }
        println!();
    } else {
        println!("Run matrix:");
        println!(
            "  {:<17}  {:>3}  {:<28}  {:>8}  {:>9}  {:>8}  {:>10}  {:>10}  {:>10}",
            "preconditioner",
            "N",
            "shift",
            "damping",
            "wall (s)",
            "retries",
            "iters",
            "inner",
            "max res"
        );
        println!(
            "  {}",
            "─".repeat(17 + 2 + 3 + 2 + 28 + 2 + 8 + 2 + 9 + 2 + 8 + 2 + 10 + 2 + 10 + 2 + 10)
        );
        for r in &results {
            match r.wall_clock_s {
                Some(wall) => println!(
                    "  {:<17}  {:>3}  {:<28}  {:>8.4}  {:>9.2}  {:>8}  {:>10}  {:>10}  {:>10.2e}",
                    r.preconditioner.as_str(),
                    r.rebuild_every,
                    r.shift.label(),
                    r.modal_damping_ratio,
                    wall,
                    r.diagnostics.as_ref().map(|d| d.retry_count).unwrap_or(0),
                    r.diagnostics
                        .as_ref()
                        .map(|d| d.total_iterations)
                        .unwrap_or(0),
                    r.diagnostics
                        .as_ref()
                        .map(|d| d.total_inner_iterations)
                        .unwrap_or(0),
                    r.diagnostics
                        .as_ref()
                        .map(|d| d.max_residual)
                        .unwrap_or(0.0),
                ),
                None => println!(
                    "  {:<17}  {:>3}  {:<28}  {:>8.4}  {:>9}",
                    r.preconditioner.as_str(),
                    r.rebuild_every,
                    r.shift.label(),
                    r.modal_damping_ratio,
                    "FAILED",
                ),
            }
        }
        println!();
    }

    // Damping changes the physical Helmholtz operator. Compare only runs with
    // identical damping, using the first successful run in each damping group.
    let reference_indices = agreement_reference_indices(&results);
    if reference_indices.iter().any(Option::is_some) {
        println!("Agreement vs the first successful reference at each damping value:");
        println!(
            "  {:<17}  {:>3}  {:<28}  {:>8}  {:>9}  {:>12}  {:>10}",
            "preconditioner", "N", "shift", "damping", "wall (s)", "max |Δp|/p90", "L2 rel."
        );
        println!(
            "  {}",
            "─".repeat(17 + 2 + 3 + 2 + 28 + 2 + 8 + 2 + 9 + 2 + 12 + 2 + 10)
        );
        for (run_index, r) in results.iter().enumerate() {
            let Some(wall) = r.wall_clock_s else {
                continue;
            };
            let Some(reference_index) = reference_indices[run_index] else {
                continue;
            };
            let reference = &results[reference_index];
            let (max_rel, l2_rel) = if run_index == reference_index {
                (f64::NAN, f64::NAN)
            } else if let (Some(ref_out), Some(out)) =
                (reference.output.as_ref(), r.output.as_ref())
            {
                let (_max_abs, mr, l2) = agreement(ref_out, out);
                (mr, l2)
            } else {
                (f64::NAN, f64::NAN)
            };
            if run_index == reference_index {
                println!(
                    "  {:<17}  {:>3}  {:<28}  {:>8.4}  {:>9.2}  {:>12}  {:>10}",
                    r.preconditioner.as_str(),
                    r.rebuild_every,
                    r.shift.label(),
                    r.modal_damping_ratio,
                    wall,
                    "(ref)",
                    "(ref)"
                );
            } else {
                println!(
                    "  {:<17}  {:>3}  {:<28}  {:>8.4}  {:>9.2}  {:>12.3e}  {:>10.3e}",
                    r.preconditioner.as_str(),
                    r.rebuild_every,
                    r.shift.label(),
                    r.modal_damping_ratio,
                    wall,
                    max_rel,
                    l2_rel
                );
            }
        }
        println!();
        println!(
            "max |Δp|/p90 above ~1e-2 typically indicates one solver converged to a different\n\
             answer (e.g. due to a near-singular modal frequency or a stale preconditioner),\n\
             not a real bench result."
        );
    } else {
        println!("No successful reference run — skipping agreement check.");
    }

    if let Some(path) = &args.json_out {
        write_json(path, &args, &results)?;
        println!("Wrote JSON diagnostics to {}", path.display());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_sim_room::fem_runner::FemFrequencyStats;

    fn successful_run(damping: f64) -> RunResult {
        RunResult {
            backend: BenchBackend::Cpu,
            preconditioner: "ilut".to_string(),
            rebuild_every: 1,
            shift: ShiftedLaplacianShift::Legacy,
            modal_damping_ratio: damping,
            wall_clock_s: Some(1.0),
            output: None,
            diagnostics: None,
            error: None,
        }
    }

    #[test]
    fn agreement_references_are_partitioned_by_physical_damping() {
        let results = vec![
            successful_run(0.005),
            successful_run(0.02),
            successful_run(0.005),
        ];

        assert_eq!(
            agreement_reference_indices(&results),
            vec![Some(0), Some(1), Some(0)]
        );
    }

    #[test]
    fn frequency_stats_json_keeps_solver_decision_diagnostics() {
        let stats = FemFrequencyStats {
            block_solve: Default::default(),
            frequency_hz: 410.6,
            symmetry_mode: Some(1),
            matrix_ms: 1.0,
            preconditioner_ms: 2.0,
            rhs_ms: 3.0,
            solve_ms: 4.0,
            iterations: 30_000,
            inner_iterations: 0,
            retries: 1,
            preconditioner_age: 2,
            preconditioner_rebuild_reason: "convergence-failure".to_string(),
            fallback_stage: 4,
            max_residual: 8.0e-7,
            converged: true,
            operator_work: None,
            physical_power: None,
            boundary_flux: None,
            transfer_stats: None,
            allocation_operations: 0,
            allocation_high_water_bytes: 0,
            synchronization_calls: 0,
        };

        let value = frequency_stats_json(&stats);

        assert_eq!(value["symmetry_mode"], 1);
        assert_eq!(value["preconditioner_age"], 2);
        assert_eq!(
            value["preconditioner_rebuild_reason"],
            "convergence-failure"
        );
        assert_eq!(value["fallback_stage"], 4);
        assert_eq!(value["solve_ms_semantics"], "summed_source_work");
    }
}
