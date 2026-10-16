//! `speaker-sim`: loudspeaker BEM simulation and optimisation CLI.

use std::path::{Path, PathBuf};

use clap::{Parser, ValueEnum};

use sonium_sim_speaker::{
    cea2034_output, directivity_export, mesh_gen, mlg_loader, model::LoudspeakerModel,
    optimization, simulation,
};
use sonium_xem::io::binary_output_dir;

fn default_output_dir() -> PathBuf {
    binary_output_dir(env!("CARGO_PKG_NAME"), env!("CARGO_BIN_NAME"))
}

/// Leave Rayon in charge unless the user requests an explicit pool width.
///
/// A zero value deliberately avoids calling `ThreadPoolBuilder::num_threads`,
/// so Rayon's normal `RAYON_NUM_THREADS`/available-parallelism resolution is
/// preserved.
const DEFAULT_THREADS: usize = 0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum OptimizeMode {
    /// Compatibility mode: optimize all legacy geometry variables.
    Legacy,
    /// Run only the bounded coarse waveguide search.
    WaveguidePreview,
    /// Run coarse search, capped refinement, and production validation.
    Waveguide,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
enum SimulationBackend {
    /// Portable CPU BEM path.
    #[default]
    Cpu,
    /// CUDA dense BEM path (requires the `cuda` feature).
    Cuda,
    /// Apple Metal dense BEM path (requires the `metal` feature).
    Metal,
}

#[derive(Debug, Parser)]
#[command(
    name = "speaker-sim",
    about = "Loudspeaker BEM simulation and optimisation"
)]
struct Cli {
    /// Path to loudspeaker model JSON file.
    #[arg(short, long, group = "input")]
    model: Option<PathBuf>,

    /// Path to MeshLang (.mlg) speaker description.
    #[arg(long, group = "input")]
    mlg: Option<PathBuf>,

    /// Output directory for results and CEA-2034 files.
    /// Defaults to `<sonium-out-root>/sonium-sim-speaker/speaker-sim/`.
    #[arg(short, long, default_value_os_t = default_output_dir())]
    output: PathBuf,

    /// Generate deterministic PNG/SVG gallery assets and gallery.json in this directory.
    #[arg(long, value_name = "DIR")]
    gallery: Option<PathBuf>,

    /// Run parameter optimisation. Bare `--optimize` retains legacy behavior.
    #[arg(
        long,
        num_args = 0..=1,
        default_missing_value = "legacy",
        value_name = "MODE",
        conflicts_with = "opt_config"
    )]
    optimize: Option<OptimizeMode>,

    /// Load a versioned staged optimization plan from JSON.
    #[arg(long, value_name = "FILE", conflicts_with = "optimize")]
    opt_config: Option<PathBuf>,

    /// Resume staged optimization from the output checkpoint.
    #[arg(long)]
    resume: bool,

    /// Maximum legacy optimisation iterations.
    #[arg(long)]
    max_iter: Option<usize>,

    /// Export mesh to .geo and exit (no simulation).
    #[arg(long)]
    export_mesh: bool,

    /// Run BEM at a single frequency and export a field-map .pos file for Gmsh.
    #[arg(long)]
    field_map: Option<f64>,

    /// Export the BEM complex far-field directivity grid as JSON (for import
    /// as a `complex_grid` source in room simulations), then exit. Runs the
    /// simulation on a full-sphere angular grid about the baffle center
    /// instead of the CEA-2034 arcs.
    #[arg(long, value_name = "FILE")]
    export_directivity: Option<PathBuf>,

    /// Angular step (degrees) of the --export-directivity sphere grid. Must
    /// divide 90° evenly so the grid contains the on-axis direction.
    #[arg(long, default_value_t = 10.0, requires = "export_directivity")]
    directivity_step: f64,

    /// Measurement radius (m) of the --export-directivity sphere.
    #[arg(long, default_value_t = 1.0, requires = "export_directivity")]
    directivity_radius: f64,

    /// Print progress to stderr.
    #[arg(long, short)]
    verbose: bool,

    /// Number of Rayon worker threads used by BEM assembly and frequency lanes.
    /// Zero uses Rayon's automatic default (including `RAYON_NUM_THREADS`).
    #[arg(long, visible_alias = "cores", default_value_t = DEFAULT_THREADS)]
    threads: usize,

    /// Backend used for the final BEM simulation and directivity export.
    #[arg(long, value_enum, default_value_t = SimulationBackend::Cpu)]
    backend: SimulationBackend,

    /// Print a bounded CPU execution contract as JSON and exit.
    #[arg(long)]
    execution_contract: bool,
}

fn invalid_data(message: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.into())
}

struct GalleryComparison<'a> {
    baseline: &'a simulation::SimulationResult,
    baseline_metrics: &'a cea2034_output::SpinoramaMetrics,
    objective_history: &'a [f64],
}

fn write_simulation_provenance(
    path: &Path,
    result: &simulation::SimulationResult,
) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(&result.provenance)?;
    std::fs::write(path, json)?;
    Ok(())
}

fn write_gpu_execution_report(
    path: &Path,
    report: &simulation::GpuExecutionReport,
) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(report)?;
    std::fs::write(path, json)?;
    Ok(())
}

fn copy_gallery_source(
    output_dir: &Path,
    gallery_dir: &Path,
    relative_path: &Path,
) -> std::io::Result<()> {
    std::fs::create_dir_all(gallery_dir)?;
    if std::fs::canonicalize(output_dir)? == std::fs::canonicalize(gallery_dir)? {
        return Ok(());
    }

    let destination = gallery_dir.join(relative_path);
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::copy(output_dir.join(relative_path), destination)?;
    Ok(())
}

fn validated_driver_row_count<T>(
    frequencies: &[f64],
    rows: &[Vec<T>],
    label: &'static str,
) -> Result<Option<usize>, std::io::Error> {
    if rows.is_empty() {
        return Ok(None);
    }
    if rows.len() != frequencies.len() {
        return Err(invalid_data(format!(
            "{label} row count {} does not match frequency count {}",
            rows.len(),
            frequencies.len()
        )));
    }
    let count = rows.first().map_or(0, Vec::len);
    if count == 0 || rows.iter().any(|row| row.len() != count) {
        return Err(invalid_data(format!(
            "{label} rows must contain the same non-zero driver count"
        )));
    }
    Ok(Some(count))
}

fn write_dynamic_driver_gallery(
    gallery: &mut sonium_xem::io::GalleryRenderer,
    result: &simulation::SimulationResult,
) -> Result<Option<usize>, Box<dyn std::error::Error>> {
    let impedance_count = validated_driver_row_count(
        &result.frequencies,
        &result.driver_electrical_impedances_ohm,
        "dynamic impedance",
    )?;
    let velocity_count = validated_driver_row_count(
        &result.frequencies,
        &result.driver_velocities_bem,
        "dynamic velocity",
    )?;
    if let (Some(impedance_count), Some(velocity_count)) = (impedance_count, velocity_count)
        && impedance_count != velocity_count
    {
        return Err(invalid_data(format!(
            "dynamic impedance driver count {impedance_count} does not match velocity count {velocity_count}"
        ))
        .into());
    }
    let driver_count = impedance_count.or(velocity_count);
    let Some(driver_count) = driver_count else {
        return Ok(None);
    };

    if result
        .frequencies
        .iter()
        .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
    {
        return Err(
            invalid_data("dynamic driver gallery requires finite positive frequencies").into(),
        );
    }

    for driver_index in 0..driver_count {
        let has_impedance_evidence = result
            .driver_electrical_impedances_ohm
            .iter()
            .any(|row| row[driver_index].is_some());
        if has_impedance_evidence {
            // Skip rows without typed evidence instead of plotting them as
            // 0-ohm points (a missing measurement is not a short circuit).
            let evidenced: Vec<(f64, num_complex::Complex64)> = result
                .frequencies
                .iter()
                .zip(&result.driver_electrical_impedances_ohm)
                .filter_map(|(&frequency, row)| {
                    row.get(driver_index)
                        .and_then(|entry| *entry)
                        .map(|value| (frequency, value))
                })
                .collect();
            if evidenced.is_empty() {
                continue;
            }
            let (evidenced_frequencies, evidenced_impedances): (Vec<f64>, Vec<_>) =
                evidenced.into_iter().unzip();
            let impedance: Vec<f64> = evidenced_impedances
                .iter()
                .map(|value| value.norm())
                .collect();
            let phase: Vec<f64> = evidenced_impedances
                .iter()
                .map(|value| value.arg())
                .collect();
            gallery.line_svg(
                &format!("driver-{driver_index}-impedance.svg"),
                &evidenced_frequencies,
                &impedance,
                &format!("Driver {driver_index} impedance magnitude"),
                "simulation_provenance.json",
                "Dynamic driver electrical impedance magnitude (ohm)",
            )?;
            gallery.line_svg(
                &format!("driver-{driver_index}-impedance-phase.svg"),
                &evidenced_frequencies,
                &phase,
                &format!("Driver {driver_index} impedance phase"),
                "simulation_provenance.json",
                "Dynamic driver electrical impedance phase (rad)",
            )?;
        }

        if !result.driver_velocities_bem.is_empty() {
            let excursion = result
                .driver_velocities_bem
                .iter()
                .zip(&result.frequencies)
                .map(|(row, frequency)| {
                    row[driver_index].norm() / (std::f64::consts::TAU * frequency)
                })
                .collect::<Vec<_>>();
            gallery.line_svg(
                &format!("driver-{driver_index}-excursion.svg"),
                &result.frequencies,
                &excursion,
                &format!("Driver {driver_index} cone excursion"),
                "simulation_provenance.json",
                "Dynamic driver displacement amplitude (m), derived from BEM boundary velocity",
            )?;
        }
    }

    if !result.driver_thermal_states.is_empty() {
        if result.driver_thermal_states.len() != result.frequencies.len()
            || result
                .driver_thermal_states
                .iter()
                .any(|row| row.len() != driver_count)
        {
            return Err(invalid_data(
                "dynamic thermal rows do not match frequency and driver counts",
            )
            .into());
        }
        for driver_index in 0..driver_count {
            let temperatures = result
                .driver_thermal_states
                .iter()
                .map(|row| {
                    row[driver_index]
                        .as_ref()
                        .map(|state| state.coil_temperature_k)
                })
                .collect::<Option<Vec<_>>>();
            if let Some(temperatures) = temperatures {
                gallery.line_svg(
                    &format!("driver-{driver_index}-coil-temperature.svg"),
                    &result.frequencies,
                    &temperatures,
                    &format!("Driver {driver_index} coil temperature"),
                    "simulation_provenance.json",
                    "Dynamic driver steady-state coil temperature (K)",
                )?;
            }
        }
    }

    Ok(Some(driver_count))
}

fn load_mlg(path: &PathBuf) -> Result<LoudspeakerModel, Box<dyn std::error::Error>> {
    let source = std::fs::read_to_string(path)?;
    mlg_loader::load_model(&source, &path.to_string_lossy())
        .map_err(|error| invalid_data(format!("MeshLang error: {error}")).into())
}

fn load_json(path: &PathBuf) -> Result<LoudspeakerModel, Box<dyn std::error::Error>> {
    let json = std::fs::read_to_string(path)?;
    let model: LoudspeakerModel = serde_json::from_str(&json)?;
    model
        .validate()
        .map_err(|error| invalid_data(format!("model validation failed: {error}")))?;
    Ok(model)
}

fn run_simulation_backend(
    backend: SimulationBackend,
    model: &LoudspeakerModel,
    eval_points: &[[f64; 3]],
) -> Result<
    (
        simulation::SimulationResult,
        Option<simulation::GpuExecutionReport>,
    ),
    simulation::SimulationError,
> {
    let solver_config = simulation::SolverConfig::default();
    match backend {
        SimulationBackend::Cpu => {
            simulation::run_simulation_with_config(model, eval_points, &solver_config)
                .map(|result| (result, None))
        }
        #[cfg(feature = "cuda")]
        SimulationBackend::Cuda => {
            simulation::run_simulation_cuda_with_report(model, eval_points, &solver_config)
                .map(|(result, report)| (result, Some(report)))
        }
        #[cfg(not(feature = "cuda"))]
        SimulationBackend::Cuda => Err(simulation::SimulationError::new(
            "CUDA backend requested but speaker-sim was built without the `cuda` feature",
        )),
        #[cfg(feature = "metal")]
        SimulationBackend::Metal => {
            simulation::run_simulation_metal_with_report(model, eval_points, &solver_config)
                .map(|(result, report)| (result, Some(report)))
        }
        #[cfg(not(feature = "metal"))]
        SimulationBackend::Metal => Err(simulation::SimulationError::new(
            "Metal backend requested but speaker-sim was built without the `metal` feature",
        )),
    }
}

fn main() {
    env_logger::init();
    if let Err(error) = run(Cli::parse()) {
        eprintln!("speaker-sim failed: {error}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    let output = cli.gallery.clone().unwrap_or_else(|| cli.output.clone());
    sonium_xem::runtime::install_global_rayon_pool(cli.threads, None)
        .map_err(|error| invalid_data(format!("failed to configure Rayon thread pool: {error}")))?;

    if cli.execution_contract {
        if cli.backend != SimulationBackend::Cpu {
            return Err(invalid_data(
                "--execution-contract is a host contract; run the requested accelerator path to record device identity",
            )
            .into());
        }
        let evidence = sonium_xem::runtime::observe_execution_contract(
            "speaker-sim",
            "cpu",
            "cpu",
            std::collections::BTreeMap::from([(
                "frequency_lane_budget".to_string(),
                rayon::current_num_threads(),
            )]),
            false,
            "host-cpu",
        );
        println!("{}", serde_json::to_string_pretty(&evidence)?);
        return Ok(());
    }

    if cli.verbose {
        eprintln!(
            "Using {} Rayon worker threads{}",
            rayon::current_num_threads(),
            if cli.threads == 0 {
                " (automatic)"
            } else {
                " (explicit)"
            }
        );
    }

    // Load model from JSON or MeshLang (auto-detect .mlg extension on --model)
    let model = if let Some(ref mlg_path) = cli.mlg {
        load_mlg(mlg_path)?
    } else if let Some(ref path) = cli.model {
        if path.extension().is_some_and(|ext| ext == "mlg") {
            load_mlg(path)?
        } else {
            load_json(path)?
        }
    } else if cli.gallery.is_some() {
        // The gallery command is deliberately self-contained: it uses the
        // versioned example shipped with this crate, so a clean checkout can
        // regenerate every public figure without an external model file.
        let mut model = mlg_loader::load_model(
            include_str!("../examples/two_way_speaker.mlg"),
            "examples/two_way_speaker.mlg",
        )
        .map_err(|error| invalid_data(format!("built-in gallery model invalid: {error}")))?;
        // A pinned public demonstration needs to complete in ordinary CI. The
        // full-resolution example remains available through --mlg; the gallery
        // intentionally uses a 4-point 200 Hz–800 Hz reference sweep and
        // mesh sizing at the same validated upper frequency.
        model.simulation.freq_start = 200.0;
        model.simulation.freq_end = 800.0;
        model.simulation.freq_points = 4;
        model.simulation.max_mesh_frequency = Some(800.0);
        model
    } else {
        return Err(invalid_data("provide either --model <json/mlg> or --mlg <meshlang>, or use --gallery <dir> for the built-in demonstration").into());
    };

    if cli.verbose {
        eprintln!(
            "Cabinet: {:.0}x{:.0}x{:.0} mm",
            model.cabinet.height * 1000.0,
            model.cabinet.width * 1000.0,
            model.cabinet.depth * 1000.0,
        );
        eprintln!(
            "Woofer: r={:.1} mm, y={:.1} mm{}",
            model.woofer.radius * 1000.0,
            model.woofer.y_offset * 1000.0,
            if model.woofer.cone.is_some() {
                " (cone)"
            } else {
                ""
            },
        );
        eprintln!(
            "Tweeter: r={:.1} mm, y={:.1} mm{}",
            model.tweeter.radius * 1000.0,
            model.tweeter.y_offset * 1000.0,
            if model.tweeter.cone.is_some() {
                " (cone)"
            } else {
                ""
            },
        );
    }

    // --- Mesh-only mode ---
    if cli.export_mesh {
        std::fs::create_dir_all(&output)?;
        let mesh = mesh_gen::generate_loudspeaker_mesh(&model);
        let geo_path = output.join("speaker_mesh.geo");
        mesh_gen::export_geo(&mesh, &geo_path)?;
        let vtu_path = output.join("speaker_mesh.vtu");
        mesh_gen::export_vtu(&mesh, &vtu_path)?;
        if cli.verbose {
            eprintln!(
                "Mesh: {} nodes, {} triangles",
                mesh.nodes.len(),
                mesh.triangles.len()
            );
        }
        eprintln!("Mesh exported to {}", geo_path.display());
        return Ok(());
    }

    // --- Field-map mode ---
    if let Some(freq) = cli.field_map {
        std::fs::create_dir_all(&output)?;
        let result = simulation::run_field_map(&model, freq, 1.0, 0.02)?;
        let pos_path = output.join(format!("field_{:.0}Hz.pos", freq));
        simulation::write_field_pos(&result, &pos_path)?;

        // Also write the mesh .geo for overlay
        let mesh = mesh_gen::generate_loudspeaker_mesh(&model);
        let geo_path = output.join("speaker_mesh.geo");
        mesh_gen::export_geo(&mesh, &geo_path)?;
        let vtu_path = output.join("speaker_mesh.vtu");
        mesh_gen::export_vtu(&mesh, &vtu_path)?;

        eprintln!("Field map written to {}", pos_path.display());
        return Ok(());
    }

    if cli.max_iter.is_some()
        && (cli.opt_config.is_some()
            || matches!(
                cli.optimize,
                Some(OptimizeMode::Waveguide | OptimizeMode::WaveguidePreview)
            ))
    {
        return Err(invalid_data("--max-iter is only valid with bare legacy --optimize").into());
    }
    if cli.resume
        && cli.opt_config.is_none()
        && !matches!(
            cli.optimize,
            Some(OptimizeMode::Waveguide | OptimizeMode::WaveguidePreview)
        )
    {
        return Err(
            invalid_data("--resume requires waveguide optimization or --opt-config").into(),
        );
    }

    // Retain the unmodified design so an optimization gallery can show the
    // actual before/after product rather than comparing against a second
    // reconstructed model.
    let baseline_model = model.clone();

    // --- Optimisation ---
    let staged_mode = cli.opt_config.is_some()
        || matches!(
            cli.optimize,
            Some(OptimizeMode::Waveguide | OptimizeMode::WaveguidePreview)
        );
    let (model, staged_summary) = if matches!(cli.optimize, Some(OptimizeMode::Legacy)) {
        let max_iterations = cli.max_iter.unwrap_or(20);
        let config = optimization::OptimizationConfig {
            max_iterations,
            ..Default::default()
        };
        if cli.verbose {
            eprintln!("Starting optimisation (max {max_iterations} iterations)...");
        }
        let result = optimization::optimize(&model, &config)?;
        if cli.verbose {
            eprintln!(
                "Optimisation: {} iterations, objective={:.4}, converged={}",
                result.iterations, result.objective, result.converged
            );
        }
        // Write optimised model JSON
        std::fs::create_dir_all(&output)?;
        let opt_json = serde_json::to_string_pretty(&result.model)?;
        let opt_path = output.join("optimised_model.json");
        std::fs::write(&opt_path, &opt_json)?;
        if cli.verbose {
            eprintln!("Optimised model written to {}", opt_path.display());
        }
        (result.model, None)
    } else if staged_mode {
        let plan = if let Some(path) = &cli.opt_config {
            let json = std::fs::read_to_string(path)?;
            serde_json::from_str::<optimization::OptimizationPlanV1>(&json)?
        } else if matches!(cli.optimize, Some(OptimizeMode::WaveguidePreview)) {
            optimization::OptimizationPlanV1::waveguide_preview()
        } else {
            optimization::OptimizationPlanV1::waveguide_balanced()
        };
        if cli.verbose {
            eprintln!(
                "Starting staged waveguide optimization ({})...",
                if plan.validation.is_some() {
                    "balanced"
                } else {
                    "preview"
                }
            );
        }
        let result = optimization::optimize_with_plan(&model, &plan, &output, cli.resume)?;
        let summary = result.summary;
        eprintln!(
            "Waveguide result: preference score {:.2} (200 Hz–5 kHz band-limited), accepted={}",
            summary.best_metrics.preference_score, summary.accepted
        );
        (result.model, Some(summary))
    } else {
        (model, None)
    };

    // --- Complex directivity export mode ---
    if let Some(path) = &cli.export_directivity {
        let step = cli.directivity_step;
        if !(step > 0.0 && step <= 90.0) || (90.0 / step).fract().abs() > 1e-9 {
            return Err(invalid_data(
                "--directivity-step must be in (0, 90] and divide 90° evenly so the grid contains the on-axis direction",
            )
            .into());
        }
        if !(cli.directivity_radius.is_finite() && cli.directivity_radius > 0.0) {
            return Err(invalid_data("--directivity-radius must be positive and finite").into());
        }
        let n_horizontal = (360.0 / step).round() as usize;
        let n_vertical = (180.0 / step).round() as usize;
        let horizontal: Vec<f64> = (0..n_horizontal).map(|i| i as f64 * step).collect();
        let vertical: Vec<f64> = (0..=n_vertical).map(|i| i as f64 * step).collect();
        let origin = model.baffle_center();
        let eval_points = directivity_export::spherical_eval_points(
            origin,
            cli.directivity_radius,
            &horizontal,
            &vertical,
        );
        if cli.verbose {
            eprintln!(
                "Running BEM simulation on a {}x{} directivity sphere ({} points)...",
                n_vertical,
                n_horizontal,
                eval_points.len()
            );
        }
        let (result, accelerator_report) =
            run_simulation_backend(cli.backend, &model, &eval_points)?;
        let artifact =
            directivity_export::export_complex_directivity_artifact(&result, &eval_points, origin)?;
        let json = serde_json::to_string_pretty(&artifact)?;
        std::fs::write(path, json)?;
        write_simulation_provenance(&path.with_extension("provenance.json"), &result)?;
        if let Some(report) = accelerator_report.as_ref() {
            write_gpu_execution_report(&path.with_extension("accelerator.json"), report)?;
        }
        eprintln!("Complex directivity grid written to {}", path.display());
        return Ok(());
    }

    // --- Simulation ---
    let eval_points = cea2034_output::generate_eval_points(model.baffle_center());

    if cli.verbose {
        eprintln!(
            "Running BEM simulation: {:.0}-{:.0} Hz, {} points...",
            model.simulation.freq_start, model.simulation.freq_end, model.simulation.freq_points
        );
    }

    let (result, accelerator_report) = run_simulation_backend(cli.backend, &model, &eval_points)?;

    // --- CEA-2034 output ---
    std::fs::create_dir_all(&output)?;
    cea2034_output::write_cea2034_files(&output, &result)?;
    write_simulation_provenance(&output.join("simulation_provenance.json"), &result)?;
    if let Some(report) = accelerator_report.as_ref() {
        write_gpu_execution_report(&output.join("accelerator_execution.json"), report)?;
    }

    // --- Spinorama scoring ---
    let metrics = cea2034_output::compute_spinorama_metrics(&result).map_err(invalid_data)?;
    eprintln!("\n--- Spinorama Metrics ---");
    if staged_mode {
        eprintln!(
            "Preference score: {:.2} (200 Hz–5 kHz band-limited)",
            metrics.preference_score
        );
    } else {
        eprintln!("Preference score: {:.2}", metrics.preference_score);
    }

    // Write mesh .geo alongside
    let mesh = mesh_gen::generate_loudspeaker_mesh(&model);
    let geo_path = output.join("speaker_mesh.geo");
    mesh_gen::export_geo(&mesh, &geo_path)?;
    mesh_gen::export_vtu(&mesh, output.join("speaker_mesh.vtu"))?;

    if let Some(gallery_dir) = &cli.gallery {
        for source in [
            "speaker_mesh.vtu",
            "simulation_provenance.json",
            "cea2034/manifest.json",
        ] {
            copy_gallery_source(&output, gallery_dir, Path::new(source))?;
        }
        if accelerator_report.is_some() {
            copy_gallery_source(
                &output,
                gallery_dir,
                Path::new("accelerator_execution.json"),
            )?;
        }
        if staged_summary.is_some() {
            copy_gallery_source(&output, gallery_dir, Path::new("optimization_summary.json"))?;
        }
        if let Some(summary) = staged_summary.as_ref() {
            let baseline_eval_points =
                cea2034_output::generate_eval_points(baseline_model.baffle_center());
            let (baseline_result, baseline_accelerator_report) =
                run_simulation_backend(cli.backend, &baseline_model, &baseline_eval_points)?;
            let baseline_metrics = cea2034_output::compute_spinorama_metrics(&baseline_result)
                .map_err(invalid_data)?;
            write_simulation_provenance(
                &gallery_dir.join("baseline_simulation_provenance.json"),
                &baseline_result,
            )?;
            if let Some(report) = baseline_accelerator_report.as_ref() {
                write_gpu_execution_report(
                    &gallery_dir.join("baseline_accelerator_execution.json"),
                    report,
                )?;
            }
            let comparison = GalleryComparison {
                baseline: &baseline_result,
                baseline_metrics: &baseline_metrics,
                objective_history: &summary.objective_history,
            };
            write_gallery(gallery_dir, &result, &metrics, &mesh, Some(&comparison))?;
        } else {
            write_gallery(gallery_dir, &result, &metrics, &mesh, None)?;
        }
        eprintln!(
            "Gallery manifest written to {}",
            gallery_dir.join("gallery.json").display()
        );
    }

    if cli.verbose {
        eprintln!(
            "Mesh: {} nodes, {} triangles",
            mesh.nodes.len(),
            mesh.triangles.len()
        );
    }
    eprintln!("Results written to {}", output.display());
    Ok(())
}

fn write_gallery(
    directory: &Path,
    result: &simulation::SimulationResult,
    metrics: &cea2034_output::SpinoramaMetrics,
    mesh: &mesh_gen::SpeakerMesh,
    comparison: Option<&GalleryComparison<'_>>,
) -> Result<(), Box<dyn std::error::Error>> {
    use sonium_xem::io::GalleryRenderer;
    if result.frequencies.len() < 2
        || result.spl.nrows() != result.frequencies.len()
        || result.spl.as_slice().iter().any(|value| !value.is_finite())
    {
        return Err(invalid_data(
            "speaker gallery requires finite, shape-validated simulation output",
        )
        .into());
    }
    let mut gallery = GalleryRenderer::new(directory)?;
    let spl = result.spl.as_slice().to_vec();
    let phase = result.phase.as_slice().to_vec();
    let pressure_magnitude = result
        .complex_pressure
        .as_slice()
        .iter()
        .map(|value| value.norm())
        .collect::<Vec<_>>();
    let geometry = mesh
        .nodes
        .iter()
        .flat_map(|node| [node[0], node[1], node[2]])
        .collect::<Vec<_>>();
    gallery.heatmap_png(
        "speaker-geometry.png",
        &geometry,
        "speaker_mesh.vtu",
        "Pinned cabinet/driver geometry view from production mesh",
    )?;
    gallery.grid_heatmap_png(
        "pressure-magnitude.png",
        &pressure_magnitude,
        result.complex_pressure.ncols(),
        "simulation_provenance.json",
        "Complex pressure magnitude (Pa peak) over the pinned frequency/observer grid",
    )?;
    gallery.grid_heatmap_png(
        "pressure-phase.png",
        &phase,
        result.phase.ncols(),
        "simulation_provenance.json",
        "Complex pressure phase (rad, e^{-iωt}) over the pinned frequency/observer grid",
    )?;
    gallery.heatmap_png(
        "horizontal-polar.png",
        &spl,
        "cea2034/manifest.json",
        "Horizontal polar heatmap; CEA-2034 normalized sampling grid",
    )?;
    gallery.heatmap_png(
        "vertical-polar.png",
        &spl.iter().rev().copied().collect::<Vec<_>>(),
        "cea2034/manifest.json",
        "Vertical polar heatmap; CEA-2034 normalized sampling grid",
    )?;
    gallery.line_svg(
        "spinorama-on-axis.svg",
        &result.frequencies,
        &metrics.on_axis,
        "CEA-2034 on-axis response",
        "cea2034/manifest.json",
        "Fixed CEA-2034 axes and smoothing contract",
    )?;
    for (filename, title, values) in [
        (
            "spinorama-listening-window.svg",
            "CEA-2034 listening window response",
            &metrics.listening_window,
        ),
        (
            "spinorama-early-reflections.svg",
            "CEA-2034 early reflections response",
            &metrics.early_reflections,
        ),
        (
            "spinorama-sound-power.svg",
            "CEA-2034 sound power response",
            &metrics.sound_power,
        ),
        (
            "spinorama-pir.svg",
            "CEA-2034 predicted in-room response",
            &metrics.pir,
        ),
    ] {
        gallery.line_svg(
            filename,
            &result.frequencies,
            values,
            title,
            "cea2034/manifest.json",
            "Fixed CEA-2034 axes and smoothing contract",
        )?;
    }
    // Legacy woofer series (deprecated fields): prescribed results carry
    // no driver matrix, and an empty series would fail the gallery's
    // finite-samples contract.
    #[allow(deprecated)]
    let impedance = result
        .woofer_electrical_impedance_ohm
        .iter()
        .map(|value| value.map(|z| z.norm()).unwrap_or(0.0))
        .collect::<Vec<_>>();
    gallery.line_svg(
        "impedance.svg",
        &result.frequencies,
        &impedance,
        "Woofer impedance magnitude",
        "simulation_provenance.json",
        "Electrical impedance magnitude (ohm)",
    )?;
    // Same legacy-series rationale as above.
    #[allow(deprecated)]
    let impedance_phase = result
        .woofer_electrical_impedance_ohm
        .iter()
        .map(|value| value.map(|z| z.arg()).unwrap_or(0.0))
        .collect::<Vec<_>>();
    gallery.line_svg(
        "impedance-phase.svg",
        &result.frequencies,
        &impedance_phase,
        "Woofer impedance phase",
        "simulation_provenance.json",
        "Electrical impedance phase (rad)",
    )?;
    let excursion = result
        .driver_velocities_bem
        .iter()
        .zip(&result.frequencies)
        .map(|(velocities, &frequency)| {
            velocities.first().map_or(0.0, |velocity| {
                velocity.norm() / (std::f64::consts::TAU * frequency)
            })
        })
        .collect::<Vec<_>>();
    gallery.line_svg(
        "woofer-excursion.svg",
        &result.frequencies,
        &excursion,
        "Woofer cone excursion",
        "simulation_provenance.json",
        "Woofer displacement amplitude (m), derived from validated BEM boundary velocity",
    )?;
    let thermal_load = result
        .power_ledgers
        .iter()
        .map(|ledger| {
            ledger
                .as_ref()
                .map(|ledger| ledger.copper_loss_w + ledger.magnetic_loss_w)
                .unwrap_or(0.0)
        })
        .collect::<Vec<_>>();
    gallery.line_svg(
        "woofer-thermal-load.svg",
        &result.frequencies,
        &thermal_load,
        "Woofer thermal load",
        "simulation_provenance.json",
        "Validated copper plus magnetic thermal loss (W); thermal-state temperature is emitted when the model enables it",
    )?;
    let thermal = result
        .driver_thermal_states
        .iter()
        .map(|states| {
            states
                .first()
                .and_then(|state| state.as_ref())
                .map(|state| state.coil_temperature_k)
        })
        .collect::<Option<Vec<_>>>();
    if let Some(thermal) = thermal {
        gallery.line_svg(
            "woofer-coil-temperature.svg",
            &result.frequencies,
            &thermal,
            "Woofer coil temperature",
            "simulation_provenance.json",
            "Steady-state coil temperature (K) from the validated thermal operating point",
        )?;
    }

    let dynamic_driver_count = write_dynamic_driver_gallery(&mut gallery, result)?;
    let mut conventions = vec![
        "1920x1080 fixed raster canvas; deterministic colour mapping and SVG axes".into(),
        "CEA-2034 convention, source mesh, frequency grid, backend and checksums are recorded alongside simulation provenance".into(),
        format!("CEA-2034 preference score: {:.6}", metrics.preference_score),
    ];
    if let Some(driver_count) = dynamic_driver_count {
        conventions.push(format!(
            "Dynamic per-driver impedance, excursion, and available thermal curves for {driver_count} drivers"
        ));
    }

    if let Some(comparison) = comparison {
        if comparison.baseline.frequencies.len() < 2
            || comparison.baseline.spl.nrows() != comparison.baseline.frequencies.len()
            || comparison.baseline.spl.ncols() == 0
            || comparison
                .baseline
                .spl
                .as_slice()
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err(invalid_data(
                "speaker gallery baseline requires finite, shape-validated simulation output",
            )
            .into());
        }

        let baseline_source = "baseline_simulation_provenance.json";
        let optimized_source = "simulation_provenance.json";
        gallery.line_svg(
            "baseline-spinorama-on-axis.svg",
            &comparison.baseline.frequencies,
            &comparison.baseline_metrics.on_axis,
            "Baseline CEA-2034 on-axis response",
            baseline_source,
            "Unoptimized baseline response on the pinned CEA-2034 frequency grid",
        )?;
        gallery.line_svg(
            "optimized-spinorama-on-axis.svg",
            &result.frequencies,
            &metrics.on_axis,
            "Optimized CEA-2034 on-axis response",
            optimized_source,
            "Optimized response on the pinned CEA-2034 frequency grid",
        )?;

        let baseline_directivity = comparison.baseline.spl.as_slice();
        let optimized_directivity = result.spl.as_slice();
        gallery.grid_heatmap_png(
            "baseline-directivity.png",
            baseline_directivity,
            comparison.baseline.spl.ncols(),
            baseline_source,
            "Baseline far-field directivity samples over the frequency and CEA observer grids",
        )?;
        gallery.grid_heatmap_png(
            "optimized-directivity.png",
            optimized_directivity,
            result.spl.ncols(),
            optimized_source,
            "Optimized far-field directivity samples over the frequency and CEA observer grids",
        )?;

        let baseline_directivity_x = (0..comparison.baseline.spl.ncols())
            .map(|index| index as f64)
            .collect::<Vec<_>>();
        let optimized_directivity_x = (0..result.spl.ncols())
            .map(|index| index as f64)
            .collect::<Vec<_>>();
        let baseline_directivity_curve =
            baseline_directivity[..comparison.baseline.spl.ncols()].to_vec();
        let optimized_directivity_curve = optimized_directivity[..result.spl.ncols()].to_vec();
        gallery.line_svg(
            "baseline-directivity.svg",
            &baseline_directivity_x,
            &baseline_directivity_curve,
            &format!(
                "Baseline directivity at {:.0} Hz",
                comparison.baseline.frequencies[0]
            ),
            baseline_source,
            "Baseline observer-grid directivity at the first pinned frequency; x is the deterministic CEA observer index",
        )?;
        gallery.line_svg(
            "optimized-directivity.svg",
            &optimized_directivity_x,
            &optimized_directivity_curve,
            &format!("Optimized directivity at {:.0} Hz", result.frequencies[0]),
            optimized_source,
            "Optimized observer-grid directivity at the first pinned frequency; x is the deterministic CEA observer index",
        )?;

        if comparison.objective_history.len() >= 2 {
            let objective_x = (0..comparison.objective_history.len())
                .map(|index| index as f64)
                .collect::<Vec<_>>();
            gallery.line_svg(
                "objective-history.svg",
                &objective_x,
                comparison.objective_history,
                "Staged optimization objective history",
                "optimization_summary.json",
                "Finite successful objective values in append-only evaluation order",
            )?;
        }

        conventions.push(format!(
            "Baseline/optimized response and directivity comparison; {} objective-history samples",
            comparison.objective_history.len()
        ));
    }

    let manifest_path = gallery.finish("Loudspeaker BEM gallery", conventions)?;
    sonium_xem::io::validate_gallery_manifest(&manifest_path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::error::ErrorKind;
    use num_complex::Complex64;
    use sonium_compute::HostMatrix;
    use sonium_xem::io::validate_png_nonblank;

    #[test]
    fn bare_optimize_remains_legacy() {
        let cli =
            Cli::try_parse_from(["speaker-sim", "--model", "speaker.json", "--optimize"]).unwrap();
        assert_eq!(cli.optimize, Some(OptimizeMode::Legacy));
    }

    #[test]
    fn waveguide_modes_parse() {
        let preview = Cli::try_parse_from([
            "speaker-sim",
            "--model",
            "speaker.json",
            "--optimize",
            "waveguide-preview",
        ])
        .unwrap();
        assert_eq!(preview.optimize, Some(OptimizeMode::WaveguidePreview));
        let balanced = Cli::try_parse_from([
            "speaker-sim",
            "--model",
            "speaker.json",
            "--optimize=waveguide",
        ])
        .unwrap();
        assert_eq!(balanced.optimize, Some(OptimizeMode::Waveguide));
    }

    #[test]
    fn threads_default_to_automatic_and_can_be_overridden() {
        let default = Cli::try_parse_from(["speaker-sim", "--model", "speaker.json"]).unwrap();
        assert_eq!(default.threads, DEFAULT_THREADS);
        assert_eq!(default.threads, 0);

        let automatic =
            Cli::try_parse_from(["speaker-sim", "--model", "speaker.json", "--threads", "0"])
                .unwrap();
        assert_eq!(automatic.threads, 0);

        let custom =
            Cli::try_parse_from(["speaker-sim", "--model", "speaker.json", "--threads", "8"])
                .unwrap();
        assert_eq!(custom.threads, 8);

        let alias = Cli::try_parse_from(["speaker-sim", "--model", "speaker.json", "--cores", "6"])
            .unwrap();
        assert_eq!(alias.threads, 6);
    }

    #[test]
    fn optimize_and_config_conflict() {
        let error = Cli::try_parse_from([
            "speaker-sim",
            "--model",
            "speaker.json",
            "--optimize",
            "--opt-config",
            "plan.json",
        ])
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ArgumentConflict);
    }

    #[test]
    fn gallery_sources_are_staged_for_separate_gallery_directory() {
        let output = tempfile::tempdir().unwrap();
        let gallery = tempfile::tempdir().unwrap();
        let relative_path = Path::new("cea2034/manifest.json");
        std::fs::create_dir_all(output.path().join(relative_path).parent().unwrap()).unwrap();
        std::fs::write(output.path().join(relative_path), b"manifest").unwrap();

        copy_gallery_source(output.path(), gallery.path(), relative_path).unwrap();

        assert_eq!(
            std::fs::read(gallery.path().join(relative_path)).unwrap(),
            b"manifest"
        );
    }

    #[test]
    fn dynamic_driver_gallery_shape_validation_tracks_all_drivers() {
        let frequencies = vec![100.0, 200.0];
        let impedance = vec![
            vec![Some(num_complex::Complex64::new(6.0, 1.0)), None],
            vec![Some(num_complex::Complex64::new(7.0, 1.0)), None],
        ];
        let velocity = vec![
            vec![
                num_complex::Complex64::new(0.1, 0.0),
                num_complex::Complex64::new(0.2, 0.0),
            ],
            vec![
                num_complex::Complex64::new(0.1, 0.0),
                num_complex::Complex64::new(0.2, 0.0),
            ],
        ];

        assert_eq!(
            validated_driver_row_count(&frequencies, &impedance, "impedance").unwrap(),
            Some(2)
        );
        assert_eq!(
            validated_driver_row_count(&frequencies, &velocity, "velocity").unwrap(),
            Some(2)
        );
    }

    #[test]
    fn dynamic_driver_gallery_shape_validation_rejects_misaligned_rows() {
        let frequencies = vec![100.0, 200.0];
        let rows = vec![vec![1.0], vec![1.0, 2.0]];
        let error = validated_driver_row_count(&frequencies, &rows, "velocity").unwrap_err();
        assert!(error.to_string().contains("same non-zero driver count"));
    }

    // Fixture fills the deprecated per-driver vectors alongside the matrix.
    #[allow(deprecated)]
    #[test]
    fn dynamic_driver_gallery_writes_per_driver_artifacts() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("simulation_provenance.json"), b"{}").unwrap();
        let result = simulation::SimulationResult {
            frequencies: vec![100.0, 200.0],
            spl: HostMatrix::zeros(2, 1),
            phase: HostMatrix::zeros(2, 1),
            complex_pressure: HostMatrix::from_elem(2, 1, Complex64::new(1.0, 0.0)),
            woofer_electrical_impedance_ohm: vec![None; 2],
            tweeter_electrical_impedance_ohm: vec![None; 2],
            driver_electrical_impedances_ohm: vec![
                vec![
                    Some(Complex64::new(6.0, 1.0)),
                    Some(Complex64::new(4.0, 0.5)),
                ],
                vec![
                    Some(Complex64::new(7.0, 1.0)),
                    Some(Complex64::new(5.0, 0.5)),
                ],
            ],
            radiation_matrices: vec![None; 2],
            radiation_matrix_diagnostics: vec![None; 2],
            thermal_driver_states: vec![[None, None]; 2],
            driver_thermal_states: vec![vec![None, None]; 2],
            power_ledgers: vec![None; 2],
            product_reference_sweep: None,
            structural_responses: Vec::new(),
            driver_velocities_bem: vec![
                vec![Complex64::new(0.1, 0.0), Complex64::new(0.2, 0.0)],
                vec![Complex64::new(0.1, 0.0), Complex64::new(0.2, 0.0)],
            ],
            exact_residual_evaluations: 0,
            provenance: Default::default(),
        };
        let mut gallery = sonium_xem::io::GalleryRenderer::new(directory.path()).unwrap();

        assert_eq!(
            write_dynamic_driver_gallery(&mut gallery, &result).unwrap(),
            Some(2)
        );
        let manifest = gallery.finish("dynamic driver test", Vec::new()).unwrap();
        sonium_xem::io::validate_gallery_manifest(&manifest).unwrap();
        for filename in [
            "driver-0-impedance.svg",
            "driver-0-impedance-phase.svg",
            "driver-0-excursion.svg",
            "driver-1-impedance.svg",
            "driver-1-impedance-phase.svg",
            "driver-1-excursion.svg",
        ] {
            assert!(
                directory.path().join(filename).is_file(),
                "missing {filename}"
            );
        }
    }

    // Fixture fills the deprecated per-driver vectors alongside the matrix.
    #[allow(deprecated)]
    #[test]
    fn dynamic_driver_gallery_skips_drivers_without_impedance_evidence() {
        // All-`None` impedance rows must not fabricate 0-ohm points: no
        // impedance SVGs are written, while the velocity-driven excursion
        // plots still are.
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("simulation_provenance.json"), b"{}").unwrap();
        let result = simulation::SimulationResult {
            frequencies: vec![100.0, 200.0],
            spl: HostMatrix::zeros(2, 1),
            phase: HostMatrix::zeros(2, 1),
            complex_pressure: HostMatrix::from_elem(2, 1, Complex64::new(1.0, 0.0)),
            woofer_electrical_impedance_ohm: vec![None; 2],
            tweeter_electrical_impedance_ohm: vec![None; 2],
            driver_electrical_impedances_ohm: vec![vec![None, None], vec![None, None]],
            radiation_matrices: vec![None; 2],
            radiation_matrix_diagnostics: vec![None; 2],
            thermal_driver_states: vec![[None, None]; 2],
            driver_thermal_states: vec![vec![None, None]; 2],
            power_ledgers: vec![None; 2],
            product_reference_sweep: None,
            structural_responses: Vec::new(),
            driver_velocities_bem: vec![
                vec![Complex64::new(0.1, 0.0), Complex64::new(0.2, 0.0)],
                vec![Complex64::new(0.1, 0.0), Complex64::new(0.2, 0.0)],
            ],
            exact_residual_evaluations: 0,
            provenance: Default::default(),
        };
        let mut gallery = sonium_xem::io::GalleryRenderer::new(directory.path()).unwrap();

        assert_eq!(
            write_dynamic_driver_gallery(&mut gallery, &result).unwrap(),
            Some(2)
        );
        for filename in ["driver-0-excursion.svg", "driver-1-excursion.svg"] {
            assert!(
                directory.path().join(filename).is_file(),
                "missing {filename}"
            );
        }
        for filename in [
            "driver-0-impedance.svg",
            "driver-0-impedance-phase.svg",
            "driver-1-impedance.svg",
            "driver-1-impedance-phase.svg",
        ] {
            assert!(
                !directory.path().join(filename).exists(),
                "fabricated impedance artifact {filename}"
            );
        }
    }

    #[test]
    fn built_in_gallery_has_nonblank_rasters_and_structured_vectors() {
        let directory = tempfile::tempdir().unwrap();
        run(Cli {
            model: None,
            mlg: None,
            output: directory.path().to_path_buf(),
            gallery: Some(directory.path().to_path_buf()),
            optimize: None,
            opt_config: None,
            resume: false,
            max_iter: None,
            export_mesh: false,
            field_map: None,
            export_directivity: None,
            directivity_step: 10.0,
            directivity_radius: 1.0,
            verbose: false,
            threads: 1,
            execution_contract: false,
            backend: SimulationBackend::Cpu,
        })
        .expect("built-in speaker gallery should regenerate");

        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.path().join("gallery.json")).unwrap())
                .unwrap();
        let figures = manifest["figures"].as_array().expect("figure array");
        assert!(
            figures.len() >= 10,
            "gallery should expose the product views"
        );
        let mut raster_count = 0;
        let mut vector_count = 0;
        for figure in figures {
            let path = figure["filename"].as_str().expect("figure filename");
            assert!(
                figure["source_checksum_fnv1a64"].is_string(),
                "figure {path} must retain source provenance"
            );
            let artifact = directory.path().join(path);
            assert!(artifact.is_file(), "missing gallery figure {path}");
            match figure["kind"].as_str().expect("figure kind") {
                "png" => {
                    assert_eq!(validate_png_nonblank(&artifact).unwrap(), (1920, 1080));
                    raster_count += 1;
                }
                "svg" => {
                    let svg = std::fs::read_to_string(&artifact).unwrap();
                    assert!(svg.contains("<svg"), "SVG figure {path} is not structured");
                    assert!(
                        svg.contains("width=\"1920\""),
                        "SVG figure {path} is not fixed-size"
                    );
                    vector_count += 1;
                }
                other => panic!("unexpected gallery figure kind {other}"),
            }
        }
        assert!(
            raster_count >= 4,
            "expected geometry/pressure/polar rasters"
        );
        assert!(
            vector_count >= 6,
            "expected spinorama/engineering SVG curves"
        );
    }
}
