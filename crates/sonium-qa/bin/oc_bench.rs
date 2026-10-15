//! OpenCourant-vs-sonium explicit-dynamics comparison runner.
//!
//! Lane 6A implementation of the frozen `qa-oc-bench` surface from
//! `PHASE6_CONTRACTS.md`. The engine arm stages decks to scratch and
//! times vendor starter+engine runs; the sonium arm times the native
//! explicit path on the same C3 decks. Both arms check probes and emit
//! one frozen `sonium.oc-bench/v1` bundle.
//!
//! Deck families: an id matching a case under the crate `decks/`
//! dir selects a C3 deck (both arms); anything else resolves as a
//! miniqa case id through `find_deck` (engine arm only; the sonium
//! arm fails closed with the reason). Same deck is the bar: both
//! arms consume the identical starter/engine deck pair, and the
//! sonium arm documents every deliberate modeling difference in its
//! equivalence basis.
//!
//! ```sh
//! qa-oc-bench --deck taylor_bar --emit-bundle bundle.json --side both
//! qa-oc-bench --deck taylor_bar --emit-bundle ci.json --quick --sonium-only
//! ```

// Rust guideline compliant 2026-02-21

use clap::{Parser, ValueEnum};
use rayon::prelude::*;
use sonium_bench::oc_compare::{
    BUNDLE_SCHEMA, BenchBundle, BenchRow, HardwareInfo, PROTOCOL_CORRECTNESS, ProtocolInfo, Side,
    bundle_json, render_bundle_markdown,
};
use sonium_compute::explicit_dt::critical_dt;
use sonium_compute::explicit_state::ExplicitState;
use sonium_fem::physics::solid::contact::explicit::{
    RigidWall, rigid_wall_contact, rigid_wall_multiplier, suggested_penalty,
};
use sonium_fem::physics::solid::fracture::ductile::{ErosionCriterion, ErosionSet};
use sonium_fem::physics::solid::nonlinear_solid::{
    HexExplicitError, HexExplicitGeom, HexExplicitParams, HexExplicitState, HexGaussPlasticGeom,
    HexGaussPlasticState, HexMesh, HexPlasticOut, NeoHookean, ObjectivePlasticMaterial,
    hex_gauss_plastic_forces, hex_jacobian, hex_plastic_forces, hex_plastic_forces_objective,
    hex_quadrature, hex_shape_derivatives, physical_gradients, q1_element_residual,
};
use sonium_fem::physics::solid::plasticity::law_metals::{
    Law2, Law2Flow, Law2State, Law36, Law36State, MaxRateEffect, RateCurve,
};
use sonium_qa::rad::oc_golden::{
    CompareOptions, MINIQA_ENV, compare_cycle, compare_history, find_deck, golden_path,
    load_golden, parse_engine_out, parse_th_csv, resolve_scratch_root,
};
use sonium_qa::rad::{TimeHistory, assert_probe_close, engine_exe, resolve_engine_pair};
use sonium_qa::rad_parse::{CardKind, DeckCard, ElemType, LoadCard, RadDeck, TypedCard};
use sonium_solvers::time_integration::{AdaptiveDtConfig, ExplicitStepDriver};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Default deck: the smallest C3 explicit case, suited for CI.
const DEFAULT_DECK: &str = "taylor_bar";
/// Penalty factor for the plastic arm (Taylor-plastic leg value: the
/// stiff LAW1 factor rings undamped on mushrooming/eroding impact).
const PLASTIC_PENALTY_FACTOR: f64 = 0.5;
/// Overdamped contact dashpot ratio for the plastic arm (leg value);
/// settles the penalty-contact bounce mode, footprint printed.
const PLASTIC_CONTACT_DAMPING: f64 = 3.0;
/// Native explicit step budget. Covers the rail crush run (about 2.6M
/// estimated steps); the pre-run estimate rejects slower decks instantly
/// instead of burning this budget on a doomed run.
const MAX_NATIVE_STEPS: u64 = 8_000_000;
/// Gap below which a wall starts in contact (sub-nanometer).
const TOUCHING_GAP_M: f64 = 1e-9;
/// Cap on sampled wall-force points per run; stride grows to fit.
const MAX_FORCE_SAMPLES: usize = 100_000;
/// Gate on the native energy ledger, mirroring the Taylor driver.
const DRIFT_TOL: f64 = 0.03;
/// Vendor OpenMP stack default, mirroring the harvest precedent.
const OMP_STACKSIZE: &str = "400m";

/// Which side(s) to time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SideOpt {
    /// Native sonium explicit run only.
    Sonium,
    /// Vendor engine run only.
    Engine,
    /// Both arms on the same deck.
    Both,
}

/// Vendor engine launch mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum FlavorOpt {
    /// One process, one thread, no `-nt`.
    Serial,
    /// One process, OpenMP threads via `-nt` and `OMP_NUM_THREADS`.
    Smp,
    /// MPI ranks via `mpirun` and the `_ompi` engine binary.
    Mpi,
}

#[derive(Parser, Debug)]
#[command(
    name = "qa-oc-bench",
    about = "Time sonium-vs-engine explicit dynamics and emit a comparison bundle."
)]
struct Args {
    /// Deck id: any case under the crate `decks/` dir (e.g.
    /// `taylor_bar`) or miniqa path (`LOIS/...`). Unknown ids and
    /// decks without a native arm fail closed with the reason.
    /// Defaults to the smallest C3 explicit deck.
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
    /// CI shortcut: one repeat, no warmup (cycle counts stay
    /// deck-dictated; combine with the default smallest deck).
    #[arg(long)]
    quick: bool,
    /// Run the sonium arm only (CI has no vendor engine).
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
    /// miniqa root (`.../qa-tests/miniqa`); else `OPENCOURANT_MINIQA`.
    #[arg(long)]
    miniqa: Option<PathBuf>,
    /// Goldens root; defaults to the crate `goldens/` dir.
    #[arg(long)]
    goldens: Option<PathBuf>,
    /// Scratch root; defaults to `SONIUM_QA_SCRATCH`, else `TMPDIR`.
    #[arg(long)]
    scratch_root: Option<PathBuf>,
    /// Keep per-repeat scratch dirs after the run.
    #[arg(long)]
    keep_scratch: bool,
}

/// Resolved side selection: (run sonium, run engine).
fn resolve_sides(args: &Args) -> Result<(bool, bool), String> {
    if args.sonium_only && args.engine_only {
        return Err("--sonium-only and --engine-only contradict".to_string());
    }
    if args.sonium_only && args.side != SideOpt::Both && args.side != SideOpt::Sonium {
        return Err("--sonium-only contradicts --side engine".to_string());
    }
    if args.engine_only && args.side != SideOpt::Both && args.side != SideOpt::Engine {
        return Err("--engine-only contradicts --side sonium".to_string());
    }
    if args.sonium_only {
        return Ok((true, false));
    }
    if args.engine_only {
        return Ok((false, true));
    }
    match args.side {
        SideOpt::Sonium => Ok((true, false)),
        SideOpt::Engine => Ok((false, true)),
        SideOpt::Both => Ok((true, true)),
    }
}

/// Resolved repeat counts: (timed repeats, warmup runs).
fn resolve_repeats(args: &Args) -> Result<(u32, u32), String> {
    // Three timed repeats plus one warmup is the default protocol.
    let repeats = args.repeats.unwrap_or(3);
    if args.quick && args.repeats.is_some_and(|repeats| repeats != 1) {
        return Err("--quick runs one repeat; drop --repeats".to_string());
    }
    let repeats = if args.quick { 1 } else { repeats };
    if repeats == 0 {
        return Err("--repeats must be positive".to_string());
    }
    // One timed repeat leaves no warmup to discard; otherwise one
    // warmup run absorbs cold caches and first-touch effects.
    let warmup = u32::from(repeats > 1);
    Ok((repeats, warmup))
}

/// One engine-verified probe expectation from `probes.json`.
#[derive(Debug, Clone)]
struct Probe {
    label: String,
    channel: String,
    expected: f64,
    abs_tol: f64,
    rel_tol: f64,
}

/// Reads probe expectations from a C3 `probes.json` file.
fn read_probe_file(path: &Path) -> Result<Vec<Probe>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|error| format!("cannot parse {}: {error}", path.display()))?;
    let probes = value
        .get("probes")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("{} has no probes array", path.display()))?;
    if probes.is_empty() {
        return Err(format!("{} holds no probes", path.display()));
    }
    let mut out = Vec::with_capacity(probes.len());
    for probe in probes {
        let get = |key: &str| {
            probe
                .get(key)
                .ok_or_else(|| format!("probe is missing {key}"))
        };
        out.push(Probe {
            label: get("label")?
                .as_str()
                .ok_or_else(|| "probe label is not a string".to_string())?
                .to_string(),
            channel: get("channel")?
                .as_str()
                .ok_or_else(|| "probe channel is not a string".to_string())?
                .to_string(),
            expected: get("expected")?
                .as_f64()
                .ok_or_else(|| "probe expected is not a number".to_string())?,
            abs_tol: get("abs_tol")?
                .as_f64()
                .ok_or_else(|| "probe abs_tol is not a number".to_string())?,
            rel_tol: get("rel_tol")?
                .as_f64()
                .ok_or_else(|| "probe rel_tol is not a number".to_string())?,
        });
    }
    Ok(out)
}

/// One timed repeat: wall time plus the correctness verdict.
#[derive(Debug, Clone)]
struct RepeatSample {
    wall_s: f64,
    cycles: u64,
    dof: u64,
    peak_rss_bytes: u64,
    correct: bool,
}

/// Aggregates repeats: median wall/cycles, worst RSS, AND verdict.
///
/// Sorting by wall time and taking the middle repeat resists one slow
/// outlier; RSS takes the maximum so the footprint never under-reports.
fn aggregate_repeats(samples: &[RepeatSample]) -> Result<RepeatSample, String> {
    if samples.is_empty() {
        return Err("no timed repeats to aggregate".to_string());
    }
    let mut ordered = samples.to_vec();
    ordered.sort_by(|a, b| a.wall_s.total_cmp(&b.wall_s));
    let median = ordered[ordered.len() / 2].clone();
    let peak_rss_bytes = samples
        .iter()
        .map(|sample| sample.peak_rss_bytes)
        .max()
        .unwrap_or(0);
    let correct = samples.iter().all(|sample| sample.correct);
    Ok(RepeatSample {
        wall_s: median.wall_s,
        cycles: median.cycles,
        dof: median.dof,
        peak_rss_bytes,
        correct,
    })
}

/// Piecewise-linear load curve with clamped ends.
fn eval_funct(points: &[[f64; 2]], time: f64) -> Result<f64, String> {
    if points.is_empty() {
        return Err("load curve holds no points".to_string());
    }
    if time <= points[0][0] {
        return Ok(points[0][1]);
    }
    for pair in points.windows(2) {
        let (x0, y0) = (pair[0][0], pair[0][1]);
        let (x1, y1) = (pair[1][0], pair[1][1]);
        if time <= x1 {
            if x1 <= x0 {
                return Ok(y1);
            }
            let weight = (time - x0) / (x1 - x0);
            return Ok(y0 * (1.0 - weight) + y1 * weight);
        }
    }
    Ok(points[points.len() - 1][1])
}

/// Reads the vendor rootname from an engine deck `/RUN/<name>/1` line.
fn run_rootname(engine_deck_text: &str) -> Result<String, String> {
    for line in engine_deck_text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("/RUN/") {
            let name = rest.split('/').next().unwrap_or_default().trim();
            if !name.is_empty() {
                return Ok(name.to_string());
            }
        }
    }
    Err("engine deck holds no /RUN/<name>/ line".to_string())
}

/// True when the starter deck carries an SI (kg/m/s) unit line.
fn has_si_units(starter_deck_text: &str) -> bool {
    starter_deck_text
        .lines()
        .any(|line| line.split_whitespace().collect::<Vec<_>>() == ["kg", "m", "s"])
}

/// Peak RSS of this process in bytes.
///
/// On Linux this reads the `VmHWM` high-water mark from
/// `/proc/self/status`: it is per address space, starts at `exec`,
/// and so measures this binary alone. `getrusage(RUSAGE_SELF)` is
/// not used there because its `ru_maxrss` survives `fork` + `exec`
/// and reports the launcher's peak instead (about 80 MB inherited
/// from `cargo run`, against a true footprint of a few MB). Other
/// platforms fall back to `getrusage`.
fn self_peak_rss_bytes() -> u64 {
    #[cfg(target_os = "linux")]
    if let Some(bytes) = linux_self_peak_rss_bytes() {
        return bytes;
    }
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 {
        return 0;
    }
    maxrss_to_bytes(usage.ru_maxrss)
}

/// Peak RSS of this process from `/proc/self/status`, when readable.
#[cfg(target_os = "linux")]
fn linux_self_peak_rss_bytes() -> Option<u64> {
    parse_vmhwm_bytes(&std::fs::read_to_string("/proc/self/status").ok()?)
}

/// Parses the `VmHWM` value (kibibytes) out of status text, in bytes.
#[cfg(target_os = "linux")]
fn parse_vmhwm_bytes(status: &str) -> Option<u64> {
    let rest = status
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))?;
    let kib: u64 = rest.trim().trim_end_matches("kB").trim().parse().ok()?;
    Some(kib.saturating_mul(1024))
}

/// Converts `ru_maxrss` to bytes (kilobytes on Linux, bytes on macOS).
fn maxrss_to_bytes(maxrss: libc::c_long) -> u64 {
    let units: u64 = maxrss.max(0) as u64;
    #[cfg(target_os = "macos")]
    {
        units
    }
    #[cfg(not(target_os = "macos"))]
    {
        units.saturating_mul(1024)
    }
}

/// Copies a directory tree (vendor `data/` into scratch).
fn copy_dir_all(from: &Path, to: &Path) -> Result<(), String> {
    std::fs::create_dir_all(to)
        .map_err(|error| format!("cannot create {}: {error}", to.display()))?;
    let mut entries: Vec<std::fs::DirEntry> = std::fs::read_dir(from)
        .map_err(|error| format!("cannot list {}: {error}", from.display()))?
        .flatten()
        .collect();
    entries.sort_by_key(std::fs::DirEntry::file_name);
    for entry in entries {
        let target = to.join(entry.file_name());
        if entry
            .file_type()
            .map_err(|error| format!("cannot stat {}: {error}", entry.path().display()))?
            .is_dir()
        {
            copy_dir_all(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), &target)
                .map_err(|error| format!("cannot copy {}: {error}", entry.path().display()))?;
        }
    }
    Ok(())
}

/// Output of one timed child process.
struct TimedOutput {
    success: bool,
    code: Option<i32>,
    combined_tail: String,
    wall_s: f64,
    peak_rss_bytes: u64,
}

/// Runs a child, timing wall seconds and reaping peak RSS via `wait4`.
///
/// Stdout/stderr drain on reader threads while the parent blocks in
/// `wait4`, so large vendor logs never deadlock the pipes. `ru_maxrss`
/// is that child's own peak, unlike the cumulative `RUSAGE_CHILDREN`.
fn run_timed(
    exe: &Path,
    args: &[String],
    dir: &Path,
    env: &[(String, String)],
) -> Result<TimedOutput, String> {
    let mut command = Command::new(exe);
    command
        .args(args)
        .current_dir(dir)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    for (key, value) in env {
        command.env(key, value);
    }
    let mut child = command
        .spawn()
        .map_err(|error| format!("cannot spawn {}: {error}", exe.display()))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "child stdout unavailable".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "child stderr unavailable".to_string())?;
    let stdout_thread = std::thread::spawn(move || {
        use std::io::Read;
        let mut pipe = stdout;
        let mut text = String::new();
        let _ = pipe.read_to_string(&mut text);
        text
    });
    let stderr_thread = std::thread::spawn(move || {
        use std::io::Read;
        let mut pipe = stderr;
        let mut text = String::new();
        let _ = pipe.read_to_string(&mut text);
        text
    });
    let start = std::time::Instant::now();
    let pid = child.id() as libc::pid_t;
    let mut status: libc::c_int = 0;
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    let waited = loop {
        let ret = unsafe { libc::wait4(pid, &mut status, 0, &mut usage) };
        if ret < 0 {
            let errno = std::io::Error::last_os_error();
            if errno.raw_os_error() == Some(libc::EINTR) {
                continue;
            }
            let _ = child.kill();
            return Err(format!("wait4 for {} failed ({errno})", exe.display()));
        }
        break ret;
    };
    let wall_s = start.elapsed().as_secs_f64();
    let stdout_text = stdout_thread.join().unwrap_or_default();
    let stderr_text = stderr_thread.join().unwrap_or_default();
    debug_assert!(waited > 0);
    let success = libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0;
    let code = if libc::WIFEXITED(status) {
        Some(libc::WEXITSTATUS(status))
    } else {
        None
    };
    let mut combined = stdout_text;
    combined.push_str(&stderr_text);
    let combined_tail: String = combined
        .chars()
        .rev()
        .take(2000)
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    Ok(TimedOutput {
        success,
        code,
        combined_tail,
        wall_s,
        peak_rss_bytes: maxrss_to_bytes(usage.ru_maxrss),
    })
}

/// Resolved engine launch: mode plus the thread/rank count on the row.
struct EngineLaunch {
    flavor: FlavorOpt,
    label: String,
    row_threads: usize,
    omp_threads: usize,
    mpi_ranks: usize,
}

/// Resolves the engine launch mode and row labels from CLI flags.
fn resolve_launch(args: &Args, engine_exe: &Path) -> Result<EngineLaunch, String> {
    if args.threads == 0 {
        return Err("--threads must be positive".to_string());
    }
    let flavor = args.flavor.unwrap_or(if args.threads > 1 {
        FlavorOpt::Smp
    } else {
        FlavorOpt::Serial
    });
    let mpi_ranks = args.mpi_ranks.unwrap_or(args.threads);
    if flavor == FlavorOpt::Mpi && mpi_ranks == 0 {
        return Err("--mpi-ranks must be positive".to_string());
    }
    if flavor == FlavorOpt::Serial && args.threads != 1 {
        return Err("--flavor serial runs one thread; pass --threads 1".to_string());
    }
    let mut label = match flavor {
        FlavorOpt::Serial => "serial".to_string(),
        FlavorOpt::Smp => "smp".to_string(),
        FlavorOpt::Mpi => "mpi".to_string(),
    };
    if engine_exe.to_string_lossy().to_lowercase().contains("_sp") {
        label.push_str("_sp");
    }
    // MPI rows count ranks; ranks own the parallelism, so each rank
    // runs single-threaded until hybrid mode is validated. Serial
    // always runs (and records) one thread per the guard above.
    let (row_threads, omp_threads) = match flavor {
        FlavorOpt::Mpi => (mpi_ranks, 1),
        FlavorOpt::Serial => (1, 1),
        FlavorOpt::Smp => (args.threads, args.threads),
    };
    Ok(EngineLaunch {
        flavor,
        label,
        row_threads,
        omp_threads,
        mpi_ranks,
    })
}

/// Picks the `_ompi` engine binary beside a resolved pair.
fn select_mpi_engine(exe: &Path) -> Result<PathBuf, String> {
    let dir = if exe.is_dir() {
        exe.to_path_buf()
    } else {
        exe.parent()
            .map(Path::to_path_buf)
            .ok_or_else(|| format!("--exe {} has no parent dir for MPI lookup", exe.display()))?
    };
    let mut candidates: Vec<(u32, PathBuf)> = Vec::new();
    let entries = std::fs::read_dir(&dir)
        .map_err(|error| format!("cannot list {}: {error}", dir.display()))?;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let lower = name.to_lowercase();
        if !lower.starts_with("engine") || !(lower.contains("ompi") || lower.contains("_mpi")) {
            continue;
        }
        // Double precision first; single precision only when the user
        // points --exe at an _sp binary directly is handled by label.
        let rank = u32::from(lower.contains("_sp"));
        candidates.push((rank, entry.path()));
    }
    candidates.sort();
    candidates
        .into_iter()
        .next()
        .map(|(_, path)| path)
        .ok_or_else(|| format!("no engine_*ompi* binary beside {}", exe.display()))
}

/// Vendor environment for one stage, mirroring the harvest precedent.
fn vendor_env(exec_dir: &Path, omp_threads: usize) -> Vec<(String, String)> {
    let mut env = Vec::new();
    if std::env::var("OMP_STACKSIZE").is_err() {
        env.push(("OMP_STACKSIZE".to_string(), OMP_STACKSIZE.to_string()));
    }
    env.push(("OMP_NUM_THREADS".to_string(), omp_threads.to_string()));
    let root = exec_dir.parent().unwrap_or(exec_dir);
    let cfg = root.join("hm_cfg_files");
    if std::env::var("RAD_CFG_PATH").is_err() && cfg.is_dir() {
        env.push((
            "RAD_CFG_PATH".to_string(),
            cfg.to_string_lossy().into_owned(),
        ));
    }
    let h3d = root.join("extlib").join("h3d").join("lib").join("linux64");
    if std::env::var("RAD_H3D_PATH").is_err() && h3d.is_dir() {
        env.push((
            "RAD_H3D_PATH".to_string(),
            h3d.to_string_lossy().into_owned(),
        ));
    }
    let reader = root.join("extlib").join("hm_reader").join("linux64");
    if reader.is_dir() {
        let prepend = reader.to_string_lossy().into_owned();
        let merged = std::env::var("LD_LIBRARY_PATH")
            .map(|current| format!("{prepend}:{current}"))
            .unwrap_or(prepend);
        env.push(("LD_LIBRARY_PATH".to_string(), merged));
    }
    env
}

/// Engine argv for one deck under the resolved launch mode.
fn engine_argv(launch: &EngineLaunch, engine_exe: &Path, deck: &str) -> (PathBuf, Vec<String>) {
    match launch.flavor {
        FlavorOpt::Mpi => (
            PathBuf::from("mpirun"),
            vec![
                "-np".to_string(),
                launch.mpi_ranks.to_string(),
                engine_exe.to_string_lossy().into_owned(),
                "-i".to_string(),
                deck.to_string(),
            ],
        ),
        FlavorOpt::Smp if launch.row_threads > 1 => (
            engine_exe.to_path_buf(),
            vec![
                "-i".to_string(),
                deck.to_string(),
                "-nt".to_string(),
                launch.row_threads.to_string(),
            ],
        ),
        _ => (
            engine_exe.to_path_buf(),
            vec!["-i".to_string(), deck.to_string()],
        ),
    }
}

/// Starter argv: `-np` only carries domain counts for MPI launches and
/// miniqa decks that declare them.
fn starter_argv(deck: &str, np: Option<usize>, extra: &[String]) -> Vec<String> {
    let mut args = vec!["-i".to_string(), deck.to_string()];
    if let Some(np) = np {
        args.push("-np".to_string());
        args.push(np.to_string());
    }
    args.extend(extra.iter().cloned());
    args
}

/// Locates the `th_to_csv` converter beside the engine or on `PATH`.
fn find_th_converter(engine_exe: &Path) -> Result<PathBuf, String> {
    let mut candidates = Vec::new();
    let dir = if engine_exe.is_dir() {
        Some(engine_exe.to_path_buf())
    } else {
        engine_exe.parent().map(Path::to_path_buf)
    };
    if let Some(dir) = dir {
        let plain = dir.join("th_to_csv");
        if plain.is_file() {
            return Ok(plain);
        }
        if let Ok(entries) = std::fs::read_dir(&dir) {
            let mut suffixed: Vec<PathBuf> = entries
                .flatten()
                .map(|entry| entry.path())
                .filter(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.starts_with("th_to_csv") && name != "th_to_csv")
                })
                .collect();
            suffixed.sort();
            if let Some(tool) = suffixed.into_iter().next() {
                return Ok(tool);
            }
        }
    }
    candidates.push(PathBuf::from("th_to_csv"));
    candidates
        .into_iter()
        .next()
        .ok_or_else(|| "th_to_csv converter missing".to_string())
}

/// True when a TH file already holds comma-separated text.
fn is_probably_csv(path: &Path) -> Result<bool, String> {
    let bytes =
        std::fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let head = &bytes[..bytes.len().min(4096)];
    Ok(!head.contains(&0) && head.contains(&b','))
}

/// Converts a binary T01 with `th_to_csv`, writing the sibling CSV.
fn convert_th(engine_exe: &Path, t01: &Path, dir: &Path) -> Result<(), String> {
    let tool = find_th_converter(engine_exe)?;
    // Vendor quirk: th_to_csv dies with SIGFPE on long absolute input
    // paths (seen past ~130 chars), so pass the bare file name with
    // the scratch dir as the working directory.
    let arg = t01
        .file_name()
        .ok_or_else(|| format!("{} has no file name", t01.display()))?;
    let output = Command::new(&tool)
        .arg(arg)
        .current_dir(dir)
        .stdin(std::process::Stdio::null())
        .output()
        .map_err(|error| format!("cannot spawn {}: {error}", tool.display()))?;
    if !output.status.success() {
        #[cfg(unix)]
        let signal = std::os::unix::process::ExitStatusExt::signal(&output.status);
        #[cfg(not(unix))]
        let signal: Option<i32> = None;
        return Err(format!(
            "{} failed on {} (code {:?}, signal {signal:?})",
            tool.display(),
            t01.display(),
            output.status.code()
        ));
    }
    Ok(())
}

/// Counts structural dofs from a starter deck: three translational
/// dofs per element-referenced node.
///
/// The starter `.out` carries no model node total (its `NUMBER OF
/// NODES` lines echo wall slave groups), so both arms parse the same
/// staged deck instead, which also keeps C3 dofs exactly equal.
/// Shell rotations and rigid-body condensation are not modeled; the
/// subset decks are brick models where the count is exact.
fn deck_dof(starter_deck: &Path) -> Result<u64, String> {
    let deck = RadDeck::parse_file(starter_deck).map_err(|error| error.to_string())?;
    let mut referenced = std::collections::BTreeSet::new();
    for card in &deck.cards {
        match card.typed.as_ref() {
            Some(TypedCard::Elems(block)) => {
                for elem in &block.elems {
                    referenced.extend(elem.nodes.iter().copied());
                }
            }
            Some(TypedCard::Sphcel(sph)) => {
                referenced.extend(sph.nodes.iter().copied());
            }
            _ => {}
        }
    }
    if referenced.is_empty() {
        return Err(format!(
            "{} references no structural nodes",
            starter_deck.display()
        ));
    }
    u64::try_from(referenced.len())
        .ok()
        .and_then(|nodes| nodes.checked_mul(3))
        .ok_or_else(|| "node count overflows dof".to_string())
}

/// Sums `TOTAL NUMBER OF CYCLES` over engine `.out` files.
///
/// The last printed cycle line goes stale under `/PRINT` strides (a
/// 24-cycle run with stride 1000 prints only cycle 0), while the run
/// summary carries the exact completed count, so the summary wins.
fn total_cycles(outs: &[PathBuf]) -> Result<u64, String> {
    let mut total = 0u64;
    for out in outs {
        let text = std::fs::read_to_string(out)
            .map_err(|error| format!("cannot read {}: {error}", out.display()))?;
        let mut found = None;
        for line in text.lines() {
            if line.contains("TOTAL NUMBER OF CYCLES") {
                found = line
                    .split_whitespace()
                    .next_back()
                    .and_then(|token| token.parse::<u64>().ok());
            }
        }
        let count = found
            .ok_or_else(|| format!("{} holds no TOTAL NUMBER OF CYCLES line", out.display()))?;
        total = total
            .checked_add(count)
            .ok_or_else(|| "cycle count overflows".to_string())?;
    }
    if total == 0 {
        return Err("engine completed zero cycles".to_string());
    }
    Ok(total)
}

/// First existing path among rootname-case variants.
fn first_existing(dir: &Path, names: &[String]) -> Result<PathBuf, String> {
    for name in names {
        let path = dir.join(name);
        if path.is_file() {
            return Ok(path);
        }
    }
    Err(format!("none of {names:?} exists in {}", dir.display()))
}

/// Candidate `.out` names: verbatim rootname, then uppercased vendor style.
fn out_candidates(rootname: &str, run: usize) -> Vec<String> {
    vec![
        format!("{rootname}_{run:04}.out"),
        format!("{}_{run:04}.out", rootname.to_uppercase()),
    ]
}

/// Candidate TH names: verbatim rootname, then uppercased vendor style.
fn th_candidates(rootname: &str, run: usize) -> Vec<String> {
    vec![
        format!("{rootname}T{run:02}"),
        format!("{}T{run:02}", rootname.to_uppercase()),
    ]
}

/// Shared engine-arm context for one deck.
struct EngineCtx {
    starter_exe: PathBuf,
    engine_exe: PathBuf,
    exec_dir: PathBuf,
    launch: EngineLaunch,
    scratch_root: PathBuf,
    keep_scratch: bool,
}

/// Runs one vendor stage, logging combined output beside the decks.
fn run_vendor_stage(
    ctx: &EngineCtx,
    exe: &Path,
    args: &[String],
    dir: &Path,
    stage: &str,
    timed: bool,
) -> Result<TimedOutput, String> {
    let env = vendor_env(&ctx.exec_dir, ctx.launch.omp_threads);
    let output = run_timed(exe, args, dir, &env)?;
    let log = if stage == "starter" {
        "starter_output.log"
    } else {
        "engine_output.log"
    };
    let _ = std::fs::write(dir.join(log), &output.combined_tail);
    if !output.success {
        return Err(format!(
            "{stage} failed (code {:?}): {}",
            output.code, output.combined_tail
        ));
    }
    if timed && output.wall_s <= 0.0 {
        return Err(format!("{stage} measured non-positive wall time"));
    }
    Ok(output)
}

/// Fresh per-repeat scratch dir for one arm and repeat index.
fn repeat_scratch(root: &Path, deck: &str, side: &str, index: usize) -> PathBuf {
    let safe = deck.replace(['/', '\\'], "_");
    root.join(format!(
        "oc_bench_{safe}_{side}_{}_{index}",
        std::process::id()
    ))
}

/// Cleans a scratch dir unless the run keeps forensics.
fn maybe_clean(dir: &Path, keep: bool) {
    if !keep {
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// C3 deck file set for one case.
#[derive(Debug)]
struct C3Decks {
    starter: PathBuf,
    engine: PathBuf,
    probes: PathBuf,
}

/// Resolves a C3 case dir, failing closed on missing artifacts.
fn resolve_c3_decks(deck: &str) -> Result<C3Decks, String> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("decks")
        .join(deck);
    let starter = dir.join(format!("{deck}_0000.rad"));
    let engine = dir.join(format!("{deck}_0001.rad"));
    let probes = dir.join("probes.json");
    for path in [&starter, &engine, &probes] {
        if !path.is_file() {
            return Err(format!("C3 deck {deck} misses {}", path.display()));
        }
    }
    Ok(C3Decks {
        starter,
        engine,
        probes,
    })
}

/// Parses one converted TH file, converting binary input first.
fn load_th_history(engine_exe: &Path, t01: &Path, dir: &Path) -> Result<TimeHistory, String> {
    if !is_probably_csv(t01)? {
        convert_th(engine_exe, t01, dir)?;
    }
    let csv = t01.with_extension("csv");
    let history_path = if csv.is_file() {
        csv
    } else {
        t01.to_path_buf()
    };
    sonium_qa::rad::parse_t01(&history_path).map_err(|error| error.to_string())
}

/// Second-half mean of a channel history (impact-plateau rule).
///
/// Returns `None` for empty histories.
fn second_half_mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let window = &values[values.len() / 2..];
    Some(window.iter().sum::<f64>() / window.len() as f64)
}

/// Settled wall-force value of a TH channel.
///
/// Rigid-wall contact forces ring on penalty impact, so the plateau
/// probes average the settled second half instead of reading the
/// phase-sensitive final sample (taylor_plastic engine FNZ: final
/// -5.77e4 vs plateau mean -3.82e4); mirrors the native arm.
fn wall_plateau_mean(history: &TimeHistory, channel: &str) -> Result<f64, String> {
    let values = history
        .channel(channel)
        .map_err(|error| error.to_string())?;
    second_half_mean(values).ok_or_else(|| format!("channel {channel} has no samples"))
}

/// Checks C3 probes against TH channel values (engine-leg rule).
///
/// Each `probes.json` expectation is anchored to the estimand that
/// produced it: taylor_plastic wall force is a settled second-half
/// plateau mean, while rail_crush and taylor_bar wall forces are
/// final-anchored; the thick cylinder follows its engine leg
/// positionally (bore DX at the 50 MPa crossing, outer DX one transit
/// later, hoop residuals as trailing means). Every other channel
/// reads the final TH value.
fn check_c3_probes(deck: &str, history: &TimeHistory, probes: &[Probe]) -> Vec<String> {
    const TRAILMEAN_SAMPLES: usize = 50;
    let sample_times = (deck == "plasticity_thick_cylinder").then(thick_cylinder_sample_times);
    let mut failures = Vec::new();
    for (index, probe) in probes.iter().enumerate() {
        let transient = match sample_times {
            Some((t_elastic, t_outer)) => Some(match index {
                0 => history
                    .value_at(&probe.channel, t_elastic)
                    .map_err(|error| error.to_string()),
                1 => history
                    .value_at(&probe.channel, t_outer)
                    .map_err(|error| error.to_string()),
                _ => history
                    .channel(&probe.channel)
                    .map_err(|error| error.to_string())
                    .and_then(|values| {
                        history_trailmean(values, TRAILMEAN_SAMPLES)
                            .ok_or_else(|| format!("channel {} has no samples", probe.channel))
                    }),
            }),
            None => None,
        };
        let plateau = deck == "taylor_plastic" && probe.channel.contains("/RWALL/");
        let sampled = if let Some(transient) = transient {
            transient
        } else if plateau {
            wall_plateau_mean(history, &probe.channel)
        } else {
            history
                .final_value(&probe.channel)
                .map_err(|error| error.to_string())
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

/// One timed C3 engine repeat: stage, starter, timed engine, probes.
fn engine_c3_repeat(
    ctx: &EngineCtx,
    deck: &str,
    decks: &C3Decks,
    probes: &[Probe],
    index: usize,
) -> Result<RepeatSample, String> {
    let dir = repeat_scratch(&ctx.scratch_root, deck, "engine", index);
    if dir.exists() {
        std::fs::remove_dir_all(&dir)
            .map_err(|error| format!("cannot clean {}: {error}", dir.display()))?;
    }
    std::fs::create_dir_all(&dir)
        .map_err(|error| format!("cannot create {}: {error}", dir.display()))?;
    let outcome = engine_c3_repeat_inner(ctx, deck, decks, probes, &dir);
    match &outcome {
        Ok(_) => maybe_clean(&dir, ctx.keep_scratch),
        Err(_) => println!("note: engine scratch kept at {}", dir.display()),
    }
    outcome
}

/// Inner C3 repeat so failures keep scratch for forensics.
fn engine_c3_repeat_inner(
    ctx: &EngineCtx,
    deck: &str,
    decks: &C3Decks,
    probes: &[Probe],
    dir: &Path,
) -> Result<RepeatSample, String> {
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
        &starter_argv(&starter_name, starter_np, &[]),
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
    let failures = check_c3_probes(deck, &history, probes);
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

/// One timed miniqa engine repeat against the checked-in golden.
fn engine_miniqa_repeat(
    ctx: &EngineCtx,
    miniqa_root: &Path,
    goldens_root: &Path,
    case: &str,
    index: usize,
) -> Result<RepeatSample, String> {
    let dir = repeat_scratch(&ctx.scratch_root, case, "engine", index);
    if dir.exists() {
        std::fs::remove_dir_all(&dir)
            .map_err(|error| format!("cannot clean {}: {error}", dir.display()))?;
    }
    let outcome = engine_miniqa_repeat_inner(ctx, miniqa_root, goldens_root, case, &dir);
    match &outcome {
        Ok(_) => maybe_clean(&dir, ctx.keep_scratch),
        Err(_) => println!("note: engine scratch kept at {}", dir.display()),
    }
    outcome
}

/// Inner miniqa repeat so failures keep scratch for forensics.
fn engine_miniqa_repeat_inner(
    ctx: &EngineCtx,
    miniqa_root: &Path,
    goldens_root: &Path,
    case: &str,
    dir: &Path,
) -> Result<RepeatSample, String> {
    let deck = find_deck(miniqa_root, case).map_err(|error| error.to_string())?;
    copy_dir_all(&deck.data_dir, dir)?;
    let starter_np = match ctx.launch.flavor {
        FlavorOpt::Mpi => ctx.launch.mpi_ranks,
        _ => deck.np,
    };
    run_vendor_stage(
        ctx,
        &ctx.starter_exe,
        &starter_argv(&deck.starter_deck, Some(starter_np), &deck.starter_args),
        dir,
        "starter",
        false,
    )?;
    let mut wall_s = 0.0;
    let mut peak_rss_bytes = 0;
    for engine_deck in &deck.engine_decks {
        let (exe, argv) = engine_argv(&ctx.launch, &ctx.engine_exe, engine_deck);
        let timed = run_vendor_stage(ctx, &exe, &argv, dir, "engine", true)?;
        wall_s += timed.wall_s;
        peak_rss_bytes = peak_rss_bytes.max(timed.peak_rss_bytes);
    }
    let runs = deck.engine_decks.len();
    let engine_out = first_existing(dir, &out_candidates(&deck.rootname, runs))?;
    let (scalars, normal_termination) =
        parse_engine_out(&engine_out).map_err(|error| error.to_string())?;
    if !normal_termination {
        return Err(format!(
            "engine run ended without NORMAL TERMINATION in {}",
            engine_out.display()
        ));
    }
    let mut run_outs = Vec::new();
    for run in 1..=runs.max(1) {
        run_outs.push(first_existing(dir, &out_candidates(&deck.rootname, run))?);
    }
    let cycles = total_cycles(&run_outs)?;
    // TH files are collected by existence: most multi-run decks write
    // T01..T0N, but some runs append to an earlier file instead.
    let mut history: Option<TimeHistory> = None;
    for run in 1..=runs.max(1) {
        let Ok(t01) = first_existing(dir, &th_candidates(&deck.rootname, run)) else {
            continue;
        };
        if !is_probably_csv(&t01)? {
            convert_th(&ctx.engine_exe, &t01, dir)?;
        }
        let csv = t01.with_extension("csv");
        let csv_path = if csv.is_file() { csv } else { t01.clone() };
        let run_history = parse_th_csv(&csv_path).map_err(|error| error.to_string())?;
        match &mut history {
            None => history = Some(run_history),
            Some(first) => sonium_qa::rad::oc_golden::concat_th_runs(first, &run_history)
                .map_err(|error| error.to_string())?,
        }
    }
    let history = history.ok_or_else(|| format!("engine runs for {case} produced no TH file"))?;
    let golden_file = golden_path(goldens_root, case);
    let golden = load_golden(&golden_file)
        .map_err(|error| format!("miniqa correctness needs the checked-in golden: {error}"))?;
    let report = compare_history(&golden, &history, &CompareOptions::default());
    println!("{}", report.summary());
    for failure in &report.failures {
        println!("FAIL: {failure}");
    }
    let cycle_failures = compare_cycle(&golden, &scalars);
    for failure in &cycle_failures {
        println!("FAIL: {failure}");
    }
    let dof = deck_dof(&dir.join(&deck.starter_deck))?;
    let correct = report.passed() && cycle_failures.is_empty();
    println!(
        "engine {case}: wall {wall_s:.3}s, cycles {cycles}, dof {dof}, rss {peak_rss_bytes} bytes, correct {correct}"
    );
    Ok(RepeatSample {
        wall_s,
        cycles,
        dof,
        peak_rss_bytes,
        correct,
    })
}

/// Native rigid wall parsed from one `/RWALL/PLANE` block.
#[derive(Debug, Clone)]
struct NativeWall {
    number: i64,
    point: [f64; 3],
    normal: [f64; 3],
    velocity: [f64; 3],
    slaves: Vec<usize>,
}

/// Observe the spring reaction without applying it to the physical state.
fn wall_reaction(
    wall: &NativeWall,
    time: f64,
    penalty: f64,
    positions: &[f64],
    velocities: &[f64],
    scratch: &mut [f64],
) -> [f64; 3] {
    scratch.fill(0.0);
    let point = std::array::from_fn(|axis| wall.point[axis] + wall.velocity[axis] * time);
    let mut rigid = RigidWall::new(point, wall.normal, penalty);
    rigid.velocity = wall.velocity;
    let pairs = rigid_wall_contact(&rigid, &wall.slaves, positions, velocities, scratch);
    let mut reaction = [0.0; 3];
    for pair in pairs {
        for (axis, value) in reaction.iter_mut().enumerate() {
            *value -= pair.force[axis];
        }
    }
    reaction
}

/// Adopt an engaging normal velocity and measure its physical impulse energy.
fn adopt_multiplier_wall_velocity(
    state: &mut ExplicitState,
    wall: &NativeWall,
    slave: usize,
) -> (f64, f64, f64, [f64; 3]) {
    let before: [f64; 3] = std::array::from_fn(|axis| state.velocities[3 * slave + axis]);
    let relative = (0..3)
        .map(|axis| (before[axis] - wall.velocity[axis]) * wall.normal[axis])
        .sum::<f64>();
    if relative >= 0.0 {
        return (0.0, 0.0, 0.0, [0.0; 3]);
    }
    let mass = state.masses[slave];
    let impulse = std::array::from_fn(|axis| -mass * relative * wall.normal[axis]);
    let mut kinetic_change = 0.0;
    let mut wall_work = 0.0;
    for axis in 0..3 {
        let after = before[axis] - relative * wall.normal[axis];
        state.velocities[3 * slave + axis] = after;
        kinetic_change += 0.5 * mass * (after * after - before[axis] * before[axis]);
        wall_work += impulse[axis] * wall.velocity[axis];
    }
    // Normal inelastic projection loses half m v_relative^2. Keep this
    // dissipation separate from actual moving-wall work and kinetic change.
    let dissipation = 0.5 * mass * relative * relative;
    (kinetic_change, wall_work, dissipation, impulse)
}

/// Native concentrated load parsed from one `/CLOAD` block.
#[derive(Debug, Clone)]
struct NativeCload {
    nodes: Vec<usize>,
    axis: usize,
    funct: i64,
    scale: f64,
}

/// Native explicit model parsed from a C3 deck pair.
///
/// Equivalence basis (documented, not silent): same mesh (deck nodes
/// and bricks, driver nodes excluded), same LAW1 E/nu/rho, same wall
/// planes/velocities/slave groups, same BCS pins and CLOAD curves,
/// same stop time. What differs: the engine integrates LAW1 bricks
/// with its QEPH formulation while sonium integrates Neo-Hookean Q1
/// bricks (identical at small strain; validated by the shared probes),
/// and contact is an exact multiplier rigid wall (constraint
/// force in the dynamics plus predictive velocity adoption;
/// frictionless, zero penetration, no contact CFL limit).
/// Plastic decks (LAW2/LAW36) run the same mesh/BCs through the
/// [`hex_plastic_forces`] path with deck-parsed hardening and
/// deck-driven erosion; damage tracking stays off (no damage card).
/// The plastic arm keeps explicit penalty walls (multiplier
/// migration pending), so its contact CFL cap stays.
#[derive(Debug, Clone)]
struct NativeModel {
    coords: Vec<[f64; 3]>,
    bricks: Vec<[usize; 8]>,
    /// Deck node id to compact native index.
    index_of: BTreeMap<i64, usize>,
    /// Deck brick id to compact native index (element probe channels).
    elem_index_of: BTreeMap<i64, usize>,
    youngs: f64,
    poisson: f64,
    density: f64,
    material: NativeMaterial,
    walls: Vec<NativeWall>,
    cloads: Vec<NativeCload>,
    /// Pinned translation axes per native node.
    pinned: Vec<[bool; 3]>,
    functs: BTreeMap<i64, Vec<[f64; 2]>>,
    stop: f64,
}

/// Native brick material parsed from the single `/MAT/*` block.
#[derive(Debug, Clone)]
enum NativeMaterial {
    /// Linear elastic (LAW1): Neo-Hookean Q1 bricks.
    Law1,
    /// Johnson-Cook (LAW2) with deck initial temperature and the
    /// deck failure strain driving erosion.
    Law2 {
        material: Law2,
        t0: f64,
        fail_ep: f64,
    },
    /// Tabulated hardening (LAW36); rupture/deletion stay off (the
    /// deck carries off-scale sentinels, verified at parse).
    Law36 { material: Law36 },
}

/// Parses one f64 token with deck context.
fn parse_scalar(token: &str, what: &str) -> Result<f64, String> {
    token
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite())
        .ok_or_else(|| format!("{what} token {token:?} is not a finite number"))
}

/// Parses one i64 token with deck context.
fn parse_int(token: &str, what: &str) -> Result<i64, String> {
    token
        .parse::<i64>()
        .ok()
        .ok_or_else(|| format!("{what} token {token:?} is not an integer"))
}

/// Flattens a `/GRNOD/NODE` block to deck node ids, skipping padding.
fn grnod_ids(card: &DeckCard) -> Result<Vec<i64>, String> {
    let mut ids = Vec::new();
    for row in &card.fields {
        for token in row {
            let id = parse_int(token, "GRNOD node")?;
            // Vendor groups pad short rows with zeros; the C3 writer
            // never emits them, but skipping is harmless either way.
            if id != 0 {
                ids.push(id);
            }
        }
    }
    Ok(ids)
}

/// Builds the native explicit model from a C3 deck pair.
fn parse_native_model(decks: &C3Decks) -> Result<NativeModel, String> {
    parse_native_model_with_validated_solid_contact(decks, false)
}

// The contact-enabled caller verifies byte-identical checked typed decks and
// builds their native operator before admitting these three contact cards.
fn parse_native_model_with_validated_solid_contact(
    decks: &C3Decks,
    validated_solid_contact: bool,
) -> Result<NativeModel, String> {
    let starter_text = std::fs::read_to_string(&decks.starter)
        .map_err(|error| format!("cannot read {}: {error}", decks.starter.display()))?;
    if !has_si_units(&starter_text) {
        return Err(format!(
            "sonium arm needs SI (kg/m/s) C3 decks: {} carries other units",
            decks.starter.display()
        ));
    }
    let starter = RadDeck::parse_file(&decks.starter).map_err(|error| error.to_string())?;
    let engine = RadDeck::parse_file(&decks.engine).map_err(|error| error.to_string())?;
    let mut stop: Option<f64> = None;
    for card in &engine.cards {
        match card.kind {
            CardKind::Run => {
                let value = card
                    .fields
                    .first()
                    .and_then(|row| row.first())
                    .ok_or_else(|| "engine /RUN block holds no stop time".to_string())?;
                stop = Some(parse_scalar(value, "RUN stop")?);
            }
            // Engine-side output controls do not affect the native run.
            CardKind::Title
            | CardKind::Begin
            | CardKind::End
            | CardKind::Tfile
            | CardKind::ThTitle
            | CardKind::AnimDt
            | CardKind::Print => {}
            _ => {
                return Err(format!(
                    "sonium arm supports no engine card {} (deck {})",
                    card.kind.as_str(),
                    decks.engine.display()
                ));
            }
        }
    }
    let stop = stop
        .filter(|stop| *stop > 0.0)
        .ok_or_else(|| "engine deck holds no positive /RUN stop time".to_string())?;

    let mut nodes: Vec<(i64, [f64; 3])> = Vec::new();
    let mut bricks: Vec<[i64; 8]> = Vec::new();
    let mut brick_ids: Vec<i64> = Vec::new();
    let mut groups: BTreeMap<i64, Vec<i64>> = BTreeMap::new();
    let mut functs: BTreeMap<i64, Vec<[f64; 2]>> = BTreeMap::new();
    for card in &starter.cards {
        match card.kind {
            CardKind::Node => {
                let TypedCard::Nodes(entries) = card.typed.as_ref().ok_or_else(|| {
                    format!("{} /NODE block undecodable", decks.starter.display())
                })?
                else {
                    return Err(format!(
                        "{} /NODE block undecodable",
                        decks.starter.display()
                    ));
                };
                for node in entries {
                    nodes.push((node.id, [node.x, node.y, node.z]));
                }
            }
            CardKind::Brick => {
                let TypedCard::Elems(block) = card.typed.as_ref().ok_or_else(|| {
                    format!("{} /BRICK block undecodable", decks.starter.display())
                })?
                else {
                    return Err(format!(
                        "{} /BRICK block undecodable",
                        decks.starter.display()
                    ));
                };
                if block.etype != ElemType::Brick {
                    return Err(format!("sonium arm supports no {:?} elements", block.etype));
                }
                for elem in &block.elems {
                    if elem.nodes.len() != 8 {
                        return Err(format!(
                            "brick {} has {} nodes, need 8",
                            elem.id,
                            elem.nodes.len()
                        ));
                    }
                    let mut conn = [0i64; 8];
                    conn.copy_from_slice(&elem.nodes);
                    bricks.push(conn);
                    brick_ids.push(elem.id);
                }
            }
            CardKind::GrnodNode => {
                let id = card
                    .id
                    .ok_or_else(|| "/GRNOD/NODE block misses its group id".to_string())?;
                groups.insert(id, grnod_ids(card)?);
            }
            CardKind::Funct => {
                let id = card
                    .id
                    .ok_or_else(|| "/FUNCT block misses its id".to_string())?;
                let TypedCard::Funct(funct) = card.typed.as_ref().ok_or_else(|| {
                    format!("/FUNCT {id} undecodable (need strictly increasing X Y pairs)")
                })?
                else {
                    return Err(format!("/FUNCT {id} undecodable"));
                };
                functs.insert(id, funct.points.clone());
            }
            CardKind::MatLaw1
            | CardKind::MatLaw2
            | CardKind::MatLaw36
            | CardKind::PropSolid
            | CardKind::Part
            | CardKind::Unit
            | CardKind::Title
            | CardKind::Begin
            | CardKind::End
            | CardKind::ThNode
            | CardKind::ThRwall
            | CardKind::ThTitle
            | CardKind::ThBric
            | CardKind::ThBrick => {}
            CardKind::Bcs | CardKind::Cload | CardKind::RwallPlane => {}
            CardKind::SurfSeg | CardKind::GrnodSurf | CardKind::InterType7
                if validated_solid_contact => {}
            _ => {
                return Err(format!(
                    "sonium arm supports no starter card {} (deferred: shells, other plastics, imposed motion)",
                    card.kind.as_str()
                ));
            }
        }
    }
    if nodes.is_empty() || bricks.is_empty() {
        return Err("starter deck holds no mesh (need /NODE + /BRICK)".to_string());
    }
    let (material, youngs, poisson, density) = parse_native_material(&starter, &functs)?;
    // Compact native numbering over brick-referenced nodes only, so
    // wall driver nodes carry no mass or dofs on either side.
    let mut referenced = std::collections::BTreeSet::new();
    for brick in &bricks {
        for id in brick {
            referenced.insert(*id);
        }
    }
    let coords_of: BTreeMap<i64, [f64; 3]> = nodes.into_iter().collect();
    let mut index_of: BTreeMap<i64, usize> = BTreeMap::new();
    let mut coords: Vec<[f64; 3]> = Vec::new();
    for id in &referenced {
        let position = coords_of
            .get(id)
            .ok_or_else(|| format!("brick references missing node {id}"))?;
        index_of.insert(*id, coords.len());
        coords.push(*position);
    }
    let mut native_bricks: Vec<[usize; 8]> = Vec::with_capacity(bricks.len());
    for brick in &bricks {
        let mut conn = [0usize; 8];
        for (slot, id) in brick.iter().enumerate() {
            conn[slot] = index_of[id];
        }
        native_bricks.push(conn);
    }
    normalize_brick_orientation(&coords, &mut native_bricks)?;
    let mut elem_index_of: BTreeMap<i64, usize> = BTreeMap::new();
    for (index, id) in brick_ids.iter().enumerate() {
        elem_index_of.insert(*id, index);
    }
    let pinned = parse_bcs(&starter, &groups, &index_of)?;
    let walls = parse_walls(&starter, &groups, &index_of, &coords_of)?;
    let cloads = parse_cloads(&starter, &groups, &index_of)?;
    Ok(NativeModel {
        coords,
        bricks: native_bricks,
        index_of,
        elem_index_of,
        youngs,
        poisson,
        density,
        material,
        walls,
        cloads,
        pinned,
        functs,
        stop,
    })
}

/// Reads E/nu/rho from the single `/MAT/LAW1` block.
fn parse_law1(starter: &RadDeck) -> Result<(f64, f64, f64), String> {
    let mut mats = Vec::new();
    for card in &starter.cards {
        if card.kind == CardKind::MatLaw1 {
            mats.push(card);
        }
    }
    let [mat] = mats.as_slice() else {
        return Err(format!(
            "sonium arm needs one /MAT/LAW1 block, found {}",
            mats.len()
        ));
    };
    let density = mat
        .fields
        .first()
        .and_then(|row| row.first())
        .ok_or_else(|| "/MAT/LAW1 block holds no density row".to_string())?;
    let density = parse_scalar(density, "LAW1 density")?;
    let moduli = mat
        .fields
        .get(1)
        .ok_or_else(|| "/MAT/LAW1 block holds no moduli row".to_string())?;
    if moduli.len() < 2 {
        return Err("/MAT/LAW1 moduli row needs E and nu".to_string());
    }
    let youngs = parse_scalar(&moduli[0], "LAW1 E")?;
    let poisson = parse_scalar(&moduli[1], "LAW1 nu")?;
    if youngs <= 0.0 || !(0.0..0.5).contains(&poisson) || density <= 0.0 {
        return Err(format!(
            "LAW1 parameters out of range: E={youngs}, nu={poisson}, rho={density}"
        ));
    }
    Ok((youngs, poisson, density))
}

/// Parses the single `/MAT/*` block into the native brick material.
///
/// Exactly one of LAW1/LAW2/LAW36 must be present; anything else (or
/// a mix) fails closed. Returns the material plus E/nu/rho for the
/// elastic wave-speed bound shared by all arms.
fn parse_native_material(
    starter: &RadDeck,
    functs: &BTreeMap<i64, Vec<[f64; 2]>>,
) -> Result<(NativeMaterial, f64, f64, f64), String> {
    let mut law1 = Vec::new();
    let mut law2 = Vec::new();
    let mut law36 = Vec::new();
    for card in &starter.cards {
        match card.kind {
            CardKind::MatLaw1 => law1.push(card),
            CardKind::MatLaw2 => law2.push(card),
            CardKind::MatLaw36 => law36.push(card),
            _ => {}
        }
    }
    let total = law1.len() + law2.len() + law36.len();
    if total != 1 {
        return Err(format!(
            "sonium arm needs one /MAT block (LAW1/LAW2/LAW36), found {total}"
        ));
    }
    if let [card] = law2.as_slice() {
        let (material, t0, fail_ep, youngs, poisson, density) = parse_law2_card(card)?;
        return Ok((
            NativeMaterial::Law2 {
                material,
                t0,
                fail_ep,
            },
            youngs,
            poisson,
            density,
        ));
    }
    if let [card] = law36.as_slice() {
        let (material, youngs, poisson, density) = parse_law36_card(card, functs)?;
        return Ok((NativeMaterial::Law36 { material }, youngs, poisson, density));
    }
    let (youngs, poisson, density) = parse_law1(starter)?;
    Ok((NativeMaterial::Law1, youngs, poisson, density))
}

/// Reads one data row of a `/MAT/*` card with deck context.
fn mat_row<'a>(card: &'a DeckCard, row: usize, what: &str) -> Result<&'a [String], String> {
    card.fields
        .get(row)
        .map(Vec::as_slice)
        .ok_or_else(|| format!("{} block holds no {what} row", card.kind.as_str()))
}

/// Parses a C3 `/MAT/LAW2` Johnson-Cook block (row layout mirrors
/// `write_extended_material`: classic input, total strain rate,
/// guide-default rate block, isothermal thermal line).
fn parse_law2_card(card: &DeckCard) -> Result<(Law2, f64, f64, f64, f64, f64), String> {
    let density = parse_scalar(
        mat_row(card, 0, "density")?
            .first()
            .ok_or_else(|| "/MAT/LAW2 block holds no density".to_string())?,
        "LAW2 density",
    )?;
    let moduli = mat_row(card, 1, "moduli")?;
    if moduli.len() < 5 {
        return Err("/MAT/LAW2 moduli row needs E, nu, Iflag, Icpre, cutoff".to_string());
    }
    let youngs = parse_scalar(&moduli[0], "LAW2 E")?;
    let poisson = parse_scalar(&moduli[1], "LAW2 nu")?;
    let iflag = parse_int(&moduli[2], "LAW2 Iflag")?;
    if iflag != 0 {
        return Err(format!(
            "LAW2 Iflag {iflag} is not classic Johnson-Cook (need 0)"
        ));
    }
    let flow_row = mat_row(card, 2, "flow")?;
    if flow_row.len() < 5 {
        return Err("/MAT/LAW2 flow row needs a, b, n, fail strain, max stress".to_string());
    }
    let flow_a = parse_scalar(&flow_row[0], "LAW2 a")?;
    let flow_b = parse_scalar(&flow_row[1], "LAW2 b")?;
    let flow_n = parse_scalar(&flow_row[2], "LAW2 n")?;
    let fail_ep = parse_scalar(&flow_row[3], "LAW2 fail strain")?;
    let max_stress = parse_scalar(&flow_row[4], "LAW2 max stress")?;
    let rate_row = mat_row(card, 3, "rate")?;
    if rate_row.len() < 6 {
        return Err("/MAT/LAW2 rate row needs C, ref rate, ICC, Fsmooth, fcut, Chard".to_string());
    }
    let rate_c = parse_scalar(&rate_row[0], "LAW2 C")?;
    let ref_rate = parse_scalar(&rate_row[1], "LAW2 ref rate")?;
    let icc = parse_int(&rate_row[2], "LAW2 ICC")?;
    if icc != 1 {
        return Err(format!(
            "LAW2 ICC {icc} is not the smoothed-rate treatment (need 1)"
        ));
    }
    let fcut = parse_scalar(&rate_row[4], "LAW2 fcut")?;
    let thermal = mat_row(card, 4, "thermal")?;
    if thermal.len() < 5 {
        return Err("/MAT/LAW2 thermal row needs T0, Tmelt, rhoCp, Tref, Tmax".to_string());
    }
    let t0 = parse_scalar(&thermal[3], "LAW2 T0")?;
    if youngs <= 0.0 || !(0.0..0.5).contains(&poisson) || density <= 0.0 {
        return Err(format!(
            "LAW2 elastic parameters out of range: E={youngs}, nu={poisson}, rho={density}"
        ));
    }
    if !(0.0..=1.0).contains(&flow_n) || rate_c < 0.0 || ref_rate <= 0.0 {
        return Err(format!(
            "LAW2 flow parameters out of range: n={flow_n}, C={rate_c}, ref rate={ref_rate}"
        ));
    }
    if fail_ep <= 0.0 || max_stress <= 0.0 || fcut <= 0.0 {
        return Err("LAW2 fail strain, max stress, and fcut must be positive".to_string());
    }
    // Isothermal run (writer: no thermal softening): melt/heat terms
    // off, reference temperature from the deck; damage tracking stays
    // off (Law2::new defaults damage to None; no damage card).
    let material = Law2::new(
        youngs,
        poisson,
        Law2Flow::JohnsonCook {
            a: flow_a,
            b: flow_b,
            n: flow_n,
            c: rate_c,
        },
        ref_rate,
        0.0,
        0.0,
        t0,
        0.0,
        max_stress,
        MaxRateEffect::Scaled,
        fail_ep,
        0.0,
        fcut,
    );
    Ok((material, t0, fail_ep, youngs, poisson, density))
}

/// Parses a C3 `/MAT/LAW36` tabulated-hardening block with its single
/// hardening curve resolved through the parsed `/FUNCT` table.
fn parse_law36_card(
    card: &DeckCard,
    functs: &BTreeMap<i64, Vec<[f64; 2]>>,
) -> Result<(Law36, f64, f64, f64), String> {
    let density = parse_scalar(
        mat_row(card, 0, "density")?
            .first()
            .ok_or_else(|| "/MAT/LAW36 block holds no density".to_string())?,
        "LAW36 density",
    )?;
    let moduli = mat_row(card, 1, "moduli")?;
    if moduli.len() < 5 {
        return Err("/MAT/LAW36 moduli row needs E, nu, rupture, t1, t2".to_string());
    }
    let youngs = parse_scalar(&moduli[0], "LAW36 E")?;
    let poisson = parse_scalar(&moduli[1], "LAW36 nu")?;
    let rupture = parse_scalar(&moduli[2], "LAW36 rupture")?;
    let tensile_t1 = parse_scalar(&moduli[3], "LAW36 t1")?;
    let tensile_t2 = parse_scalar(&moduli[4], "LAW36 t2")?;
    for (name, value) in [("rupture", rupture), ("t1", tensile_t1), ("t2", tensile_t2)] {
        if value < 1.0e19 {
            return Err(format!(
                "LAW36 {name} {value:.3e} is active (need off-scale sentinel)"
            ));
        }
    }
    let flags = mat_row(card, 2, "flags")?;
    if flags.len() < 6 {
        return Err(
            "/MAT/LAW36 flags row needs Nc, Ismooth, Fsmooth, Vref, deletion, Idel".to_string(),
        );
    }
    let deletion = parse_scalar(&flags[4], "LAW36 deletion")?;
    if deletion < 1.0e19 {
        return Err(format!(
            "LAW36 deletion {deletion:.3e} is active (need off-scale sentinel)"
        ));
    }
    let model_row = mat_row(card, 3, "model")?;
    if model_row.len() < 5 {
        return Err("LAW36 model row needs Iform, Ifil, Ipos, Fcut, Vrel".to_string());
    }
    let mut model_values = [0.0; 5];
    for (slot, token) in model_row.iter().take(5).enumerate() {
        model_values[slot] = parse_scalar(token, "LAW36 model")?;
    }
    if model_values != [0.0, 1.0, 0.0, 0.0, 0.0] {
        return Err("LAW36 model row differs from the C3 writer pattern".to_string());
    }
    let curve_id = parse_int(
        mat_row(card, 4, "curve")?
            .first()
            .ok_or_else(|| "/MAT/LAW36 block holds no curve id".to_string())?,
        "LAW36 curve",
    )?;
    let scale = parse_scalar(
        mat_row(card, 5, "scale")?
            .first()
            .ok_or_else(|| "/MAT/LAW36 block holds no curve scale".to_string())?,
        "LAW36 scale",
    )?;
    if youngs <= 0.0 || !(0.0..0.5).contains(&poisson) || density <= 0.0 {
        return Err(format!(
            "LAW36 elastic parameters out of range: E={youngs}, nu={poisson}, rho={density}"
        ));
    }
    let points = functs
        .get(&curve_id)
        .ok_or_else(|| format!("/MAT/LAW36 references missing /FUNCT {curve_id}"))?;
    let mut x = Vec::with_capacity(points.len());
    let mut y = Vec::with_capacity(points.len());
    for point in points {
        x.push(point[0]);
        y.push(point[1]);
    }
    // Single quasi-static curve (no rate table in the C3 subset).
    println!(
        "sonium LAW36 /FUNCT {curve_id}: {} points, y0 {:.6e}, ymax {:.6e}, scale {scale:.6e}",
        x.len(),
        y.first().copied().unwrap_or(f64::NAN),
        y.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );
    let curve = RateCurve::new(0.0, x, y, scale)
        .map_err(|error| format!("LAW36 curve invalid: {error}"))?;
    let material = Law36::new(
        youngs,
        poisson,
        vec![curve],
        0.0,
        rupture,
        tensile_t1,
        tensile_t2,
        deletion,
    );
    Ok((material, youngs, poisson, density))
}

/// Maps a deck group to compact native indices, skipping driver nodes.
fn group_indices(
    groups: &BTreeMap<i64, Vec<i64>>,
    group: i64,
    index_of: &BTreeMap<i64, usize>,
    what: &str,
) -> Result<Vec<usize>, String> {
    let members = groups
        .get(&group)
        .ok_or_else(|| format!("{what} references missing group {group}"))?;
    let mut indices = Vec::new();
    for id in members {
        // Driver nodes (wall masters) carry no structural mass; the
        // engine likewise excludes them from the model.
        if let Some(index) = index_of.get(id) {
            indices.push(*index);
        }
    }
    Ok(indices)
}

/// Parses `/BCS` pins into per-node locked-axis flags.
fn parse_bcs(
    starter: &RadDeck,
    groups: &BTreeMap<i64, Vec<i64>>,
    index_of: &BTreeMap<i64, usize>,
) -> Result<Vec<[bool; 3]>, String> {
    let mut pinned = vec![[false; 3]; index_of.len()];
    for card in &starter.cards {
        if card.kind != CardKind::Bcs {
            continue;
        }
        let TypedCard::Bcs(bcs) = card
            .typed
            .as_ref()
            .ok_or_else(|| "/BCS block undecodable".to_string())?
        else {
            return Err("/BCS block undecodable".to_string());
        };
        if bcs.skew.is_some_and(|skew| skew != 0) {
            return Err("/BCS in a skewed frame is not supported".to_string());
        }
        // Rotation locks are meaningless on bricks (no rotational
        // dofs); the C3 writer sets them, so ignore them openly.
        let tra: Vec<char> = bcs.tra.chars().collect();
        if tra.len() != 3 || !tra.iter().all(|flag| matches!(flag, '0' | '1')) {
            return Err(format!(
                "/BCS translation code {:?} is not three flags",
                bcs.tra
            ));
        }
        let group = bcs
            .grnod
            .ok_or_else(|| "/BCS block references no group".to_string())?;
        for index in group_indices(groups, group, index_of, "/BCS")? {
            for axis in 0..3 {
                pinned[index][axis] |= tra[axis] == '1';
            }
        }
    }
    Ok(pinned)
}

/// Parses `/RWALL/PLANE` blocks into native penalty walls.
fn parse_walls(
    starter: &RadDeck,
    groups: &BTreeMap<i64, Vec<i64>>,
    index_of: &BTreeMap<i64, usize>,
    coords_of: &BTreeMap<i64, [f64; 3]>,
) -> Result<Vec<NativeWall>, String> {
    let mut walls = Vec::new();
    for card in &starter.cards {
        if card.kind != CardKind::RwallPlane {
            continue;
        }
        let number = card
            .id
            .ok_or_else(|| "/RWALL/PLANE block misses its number".to_string())?;
        let TypedCard::Rwall(header) = card
            .typed
            .as_ref()
            .ok_or_else(|| format!("/RWALL/PLANE {number} header undecodable"))?
        else {
            return Err(format!("/RWALL/PLANE {number} header undecodable"));
        };
        if card.fields.len() < 3 {
            return Err(format!("/RWALL/PLANE {number} holds too few rows"));
        }
        let friction = card.fields[1]
            .get(1)
            .map(|token| parse_scalar(token, "RWALL friction"))
            .transpose()?
            .unwrap_or(0.0);
        if friction != 0.0 {
            return Err(format!(
                "/RWALL/PLANE {number} carries friction (unvalidated mapping)"
            ));
        }
        // Moving walls (driver node set) carry `0 VX VY VZ`; fixed
        // walls carry the plane point; both close with M1 = M + normal.
        let (point, velocity) = if header.node != 0 {
            let point = coords_of.get(&header.node).ok_or_else(|| {
                format!("/RWALL/PLANE {number} driver node {} missing", header.node)
            })?;
            if card.fields[2].len() < 4 {
                return Err(format!("/RWALL/PLANE {number} velocity row too short"));
            }
            let velocity = [
                parse_scalar(&card.fields[2][1], "RWALL VX")?,
                parse_scalar(&card.fields[2][2], "RWALL VY")?,
                parse_scalar(&card.fields[2][3], "RWALL VZ")?,
            ];
            (*point, velocity)
        } else {
            if card.fields[2].len() < 3 {
                return Err(format!("/RWALL/PLANE {number} point row too short"));
            }
            let point = [
                parse_scalar(&card.fields[2][0], "RWALL PX")?,
                parse_scalar(&card.fields[2][1], "RWALL PY")?,
                parse_scalar(&card.fields[2][2], "RWALL PZ")?,
            ];
            (point, [0.0; 3])
        };
        let orientation = card.fields.last().unwrap_or(&card.fields[2]);
        if orientation.len() < 3 {
            return Err(format!("/RWALL/PLANE {number} orientation row too short"));
        }
        let tip = [
            parse_scalar(&orientation[0], "RWALL M1X")?,
            parse_scalar(&orientation[1], "RWALL M1Y")?,
            parse_scalar(&orientation[2], "RWALL M1Z")?,
        ];
        let mut normal = [tip[0] - point[0], tip[1] - point[1], tip[2] - point[2]];
        let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if !length.is_finite() || length <= 0.0 {
            return Err(format!("/RWALL/PLANE {number} normal is degenerate"));
        }
        // The hertz deck carries a non-unit orientation offset; the
        // engine normalizes, so the native wall does the same.
        normal = [normal[0] / length, normal[1] / length, normal[2] / length];
        let mut slaves = Vec::new();
        for group in &header.grnds {
            if *group != 0 {
                slaves.extend(group_indices(groups, *group, index_of, "/RWALL")?);
            }
        }
        slaves.sort_unstable();
        slaves.dedup();
        if slaves.is_empty() {
            return Err(format!("/RWALL/PLANE {number} has no slave nodes"));
        }
        walls.push(NativeWall {
            number,
            point,
            normal,
            velocity,
            slaves,
        });
    }
    walls.sort_by_key(|wall| wall.number);
    Ok(walls)
}

/// Parses `/CLOAD` blocks into native nodal loads.
fn parse_cloads(
    starter: &RadDeck,
    groups: &BTreeMap<i64, Vec<i64>>,
    index_of: &BTreeMap<i64, usize>,
) -> Result<Vec<NativeCload>, String> {
    let mut cloads = Vec::new();
    for card in &starter.cards {
        if card.kind != CardKind::Cload {
            continue;
        }
        let TypedCard::Load(LoadCard::Cload(cload)) = card
            .typed
            .as_ref()
            .ok_or_else(|| "/CLOAD block undecodable".to_string())?
        else {
            return Err("/CLOAD block undecodable".to_string());
        };
        let axis = match cload.dir.to_uppercase().as_str() {
            "X" => 0,
            "Y" => 1,
            "Z" => 2,
            other => return Err(format!("/CLOAD direction {other} unsupported on bricks")),
        };
        // Writer row: fct Dir skew sens grnd 0 Fscale Magnitude; the
        // scale applies to EACH node of the set (rad writer precedent).
        let row = card
            .fields
            .first()
            .ok_or_else(|| "/CLOAD block holds no row".to_string())?;
        if row.len() < 8 {
            return Err("/CLOAD row too short for the writer shape".to_string());
        }
        let fscale = parse_scalar(&row[6], "CLOAD Fscale")?;
        let magnitude = parse_scalar(&row[7], "CLOAD magnitude")?;
        let nodes = group_indices(groups, cload.grnod, index_of, "/CLOAD")?;
        if nodes.is_empty() {
            return Err("/CLOAD group holds no structural nodes".to_string());
        }
        cloads.push(NativeCload {
            nodes,
            axis,
            funct: cload.funct,
            scale: fscale * magnitude,
        });
    }
    Ok(cloads)
}

/// Twelve H8 edges in `HEX_NODES` slot order.
const HEX_EDGES: [(usize, usize); 12] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];

/// Reference volume of one element by Gauss quadrature of `det J`.
fn element_volume(mesh: &HexMesh, elem: usize) -> f64 {
    signed_brick_volume(&mesh.element_coords(elem))
}

/// Signed quadrature volume of one brick from its 8 corner positions.
fn signed_brick_volume(coords: &[[f64; 3]; 8]) -> f64 {
    let mut volume = 0.0;
    for (natural, weight) in hex_quadrature() {
        let derivatives = hex_shape_derivatives(&natural);
        let (_, det, _) = hex_jacobian(coords, &derivatives);
        volume += weight * det;
    }
    volume
}

/// Reorders mirrored-winding bricks into the canonical slot convention.
///
/// Some C3 decks (beam_curved_arch) author bricks with both faces
/// wound clockwise seen from above; the engine accepts either
/// winding, but the isoparametric map is then orientation-reversing
/// and signed quadrature weights would negate the stiffness and
/// lumped mass. Reversing both faces (`[0,3,2,1,4,7,6,5]`) keeps
/// the same 8 nodes with straight edges while restoring the
/// canonical counter-clockwise convention. Each brick is handled
/// independently (already-positive bricks are untouched) and the
/// fix is verified per brick, failing closed on degenerate
/// geometry rather than a winding convention.
fn normalize_brick_orientation(
    coords: &[[f64; 3]],
    bricks: &mut [[usize; 8]],
) -> Result<(), String> {
    /// Slot permutation that reverses both face windings.
    const FACE_REVERSAL: [usize; 8] = [0, 3, 2, 1, 4, 7, 6, 5];
    for (elem, conn) in bricks.iter_mut().enumerate() {
        let gather = |conn: &[usize; 8]| {
            let mut corners = [[0.0; 3]; 8];
            for (slot, node) in conn.iter().enumerate() {
                corners[slot] = coords[*node];
            }
            corners
        };
        if signed_brick_volume(&gather(conn)) > 0.0 {
            continue;
        }
        let mut reordered = [0usize; 8];
        for (slot, source) in FACE_REVERSAL.iter().enumerate() {
            reordered[slot] = conn[*source];
        }
        *conn = reordered;
        let fixed = signed_brick_volume(&gather(conn));
        if !fixed.is_finite() || fixed <= 0.0 {
            return Err(format!(
                "element {elem} has non-positive volume {fixed:.6e} even after orientation fix"
            ));
        }
    }
    Ok(())
}

/// Minimum edge length of one element in the reference configuration.
fn min_edge_length(mesh: &HexMesh, elem: usize) -> f64 {
    let coords = mesh.element_coords(elem);
    let mut shortest = f64::INFINITY;
    for (a, b) in HEX_EDGES {
        let dx = coords[a][0] - coords[b][0];
        let dy = coords[a][1] - coords[b][1];
        let dz = coords[a][2] - coords[b][2];
        shortest = shortest.min((dx * dx + dy * dy + dz * dz).sqrt());
    }
    shortest
}

/// Native probe target: a deck node displacement, a wall force, or a
/// brick element value (centroid stress component / plastic strain).
#[derive(Clone, Copy)]
enum ProbeTarget {
    Displacement { axis: usize, node: usize },
    WallForce { axis: usize, wall: usize },
    ElemStress { comp: usize, elem: usize },
    ElemPlas { elem: usize },
}

/// Linearly interpolates a recorded probe history at a time.
///
/// Returns `None` when the history is empty, ragged, or the time
/// falls outside the sampled grid (no extrapolation).
fn history_lerp(times: &[f64], values: &[f64], time: f64) -> Option<f64> {
    if times.len() != values.len() || times.is_empty() {
        return None;
    }
    if time < times[0] || time > times[times.len() - 1] {
        return None;
    }
    for window in times.windows(2).zip(values.windows(2)) {
        let (bounds, pair) = window;
        if time >= bounds[0] && time <= bounds[1] {
            let span = bounds[1] - bounds[0];
            if span <= 0.0 {
                return Some(pair[0]);
            }
            let weight = (time - bounds[0]) / span;
            return Some(pair[0] * (1.0 - weight) + pair[1] * weight);
        }
    }
    values.last().copied()
}

/// Averages the trailing `n` history samples (ringdown-canceling mean).
///
/// Returns `None` for empty histories; shorter histories average over
/// whatever samples exist.
fn history_trailmean(values: &[f64], n: usize) -> Option<f64> {
    if values.is_empty() || n == 0 {
        return None;
    }
    let take = values.len().min(n);
    let window = &values[values.len() - take..];
    Some(window.iter().sum::<f64>() / take as f64)
}

/// Averages history samples at or after `start` (zero-hold mean).
///
/// Returns `None` when the history is ragged or no sample reaches
/// `start`. Averaging the full free-ringdown hold cancels the breathing
/// mode over many periods where a short trailing window keeps a
/// phase-dependent residue.
fn history_holdmean(times: &[f64], values: &[f64], start: f64) -> Option<f64> {
    if times.len() != values.len() {
        return None;
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    for (time, value) in times.iter().zip(values.iter()) {
        if *time >= start {
            sum += *value;
            count += 1;
        }
    }
    if count == 0 {
        None
    } else {
        Some(sum / count as f64)
    }
}

/// Transient sample times of the thick-cylinder engine leg.
///
/// Mirrors `fem_opencourant_plasticity_thick_cylinder`: the bore probe
/// reads where the raised cosine crosses 50 MPa on the loading leg,
/// the outer probe one dilatational transit later. Constants are
/// transcribed from that leg; the unit test pins both times.
fn thick_cylinder_sample_times() -> (f64, f64) {
    const A_M: f64 = 0.05;
    const B_M: f64 = 0.10;
    const E_PA: f64 = 200.0e9;
    const NU: f64 = 0.30;
    const RHO: f64 = 7800.0;
    const P_STAR_PA: f64 = 150.0e6;
    const LOAD_TIME_S: f64 = 0.001;
    const ELASTIC_PROBE_PA: f64 = 50.0e6;
    let t_elastic =
        LOAD_TIME_S * (1.0 - 2.0 * ELASTIC_PROBE_PA / P_STAR_PA).acos() / std::f64::consts::PI;
    let lame_lambda = E_PA * NU / ((1.0 + NU) * (1.0 - 2.0 * NU));
    let shear_mu = E_PA / (2.0 * (1.0 + NU));
    let c_dil = ((lame_lambda + 2.0 * shear_mu) / RHO).sqrt();
    (t_elastic, t_elastic + (B_M - A_M) / c_dil)
}

/// Resolves a C3 probe channel to a native target.
fn resolve_probe_channel(channel: &str, model: &NativeModel) -> Result<ProbeTarget, String> {
    if let Some(rest) = channel
        .strip_prefix("DX/")
        .map(|id| (0, id))
        .or_else(|| channel.strip_prefix("DY/").map(|id| (1, id)))
        .or_else(|| channel.strip_prefix("DZ/").map(|id| (2, id)))
    {
        let id: i64 = rest
            .1
            .parse()
            .ok()
            .ok_or_else(|| format!("probe channel {channel} names no node"))?;
        let node = model
            .index_of
            .get(&id)
            .copied()
            .ok_or_else(|| format!("probe channel {channel} names a non-structural node"))?;
        return Ok(ProbeTarget::Displacement { axis: rest.0, node });
    }
    for (prefix, axis) in [("FNX/RWALL/", 0), ("FNY/RWALL/", 1), ("FNZ/RWALL/", 2)] {
        if let Some(id) = channel.strip_prefix(prefix) {
            let number: i64 = id
                .parse()
                .ok()
                .ok_or_else(|| format!("probe channel {channel} names no wall"))?;
            let wall = model
                .walls
                .iter()
                .position(|wall| wall.number == number)
                .ok_or_else(|| format!("probe channel {channel} names a missing wall"))?;
            return Ok(ProbeTarget::WallForce { axis, wall });
        }
    }
    for (prefix, comp) in [
        ("SX/BRIC/", 0),
        ("SY/BRIC/", 1),
        ("SZ/BRIC/", 2),
        ("SXY/BRIC/", 3),
        ("SYZ/BRIC/", 4),
        ("SZX/BRIC/", 5),
    ] {
        if let Some(id) = channel.strip_prefix(prefix) {
            let number: i64 = id
                .parse()
                .ok()
                .ok_or_else(|| format!("probe channel {channel} names no brick"))?;
            let elem = model
                .elem_index_of
                .get(&number)
                .copied()
                .ok_or_else(|| format!("probe channel {channel} names a missing brick"))?;
            return Ok(ProbeTarget::ElemStress { comp, elem });
        }
    }
    if let Some(id) = channel.strip_prefix("PLAS/BRIC/") {
        let number: i64 = id
            .parse()
            .ok()
            .ok_or_else(|| format!("probe channel {channel} names no brick"))?;
        let elem = model
            .elem_index_of
            .get(&number)
            .copied()
            .ok_or_else(|| format!("probe channel {channel} names a missing brick"))?;
        return Ok(ProbeTarget::ElemPlas { elem });
    }
    Err(format!(
        "probe channel {channel} unsupported (need DX/DY/DZ, FN/RWALL, or SX/SY/SZ/SXY/SYZ/SZX/PLAS per BRIC)"
    ))
}

/// Linear Q1 element stiffness (row-major 24x24) at reference config.
///
/// Integrates `B^T C B` over the 8 Gauss points with the isotropic
/// `C(lambda, mu)` matching `NeoHookean::from_young` at small strain
/// (engineering-shear Voigt ordering). Precomputed once per run: the
/// LAW1 decks are small-strain, so the constant stiffness replaces
/// the finite-strain kernel (no per-step Jacobians, determinants, or
/// logarithms) at second-order strain error.
fn linear_hex_stiffness(lambda: f64, mu: f64, coords: &[[f64; 3]; 8]) -> [f64; 576] {
    let lame2 = lambda + 2.0 * mu;
    // Isotropic C in [exx, eyy, ezz, gxy, gyz, gzx] ordering.
    let mut stiff = [0.0; 576];
    for (natural, weight) in hex_quadrature() {
        let derivatives = hex_shape_derivatives(&natural);
        let (_, det, inv) = hex_jacobian(coords, &derivatives);
        let grad = physical_gradients(&inv, &derivatives);
        let det_w = weight * det;
        // B rows per node: [dNx/dx 0 0; 0 dNy/dy 0; 0 0 dNz/dz;
        // dNx/dy dNy/dx 0; 0 dNy/dz dNz/dy; dNx/dz 0 dNz/dx].
        let mut b = [[0.0; 24]; 6];
        for (node, g) in grad.iter().enumerate() {
            b[0][3 * node] = g[0];
            b[1][3 * node + 1] = g[1];
            b[2][3 * node + 2] = g[2];
            b[3][3 * node] = g[1];
            b[3][3 * node + 1] = g[0];
            b[4][3 * node + 1] = g[2];
            b[4][3 * node + 2] = g[1];
            b[5][3 * node] = g[2];
            b[5][3 * node + 2] = g[0];
        }
        // Ke += det_w * B^T C B with the 6x6 isotropic C inlined.
        for row in 0..24 {
            // C times B column `row`: normal rows couple via lambda,
            // shear rows scale by mu.
            let b_n0 = b[0][row];
            let b_n1 = b[1][row];
            let b_n2 = b[2][row];
            let cb = [
                lame2 * b_n0 + lambda * (b_n1 + b_n2),
                lame2 * b_n1 + lambda * (b_n0 + b_n2),
                lame2 * b_n2 + lambda * (b_n0 + b_n1),
                mu * b[3][row],
                mu * b[4][row],
                mu * b[5][row],
            ];
            for col in 0..24 {
                stiff[row * 24 + col] += det_w
                    * (b[0][col] * cb[0]
                        + b[1][col] * cb[1]
                        + b[2][col] * cb[2]
                        + b[3][col] * cb[3]
                        + b[4][col] * cb[4]
                        + b[5][col] * cb[5]);
            }
        }
    }
    stiff
}

/// Cache of the last internal-force evaluation for multiplier contact.
///
/// The driver evaluates the internal closure immediately before the
/// external closure with identical arguments, so the contact
/// multiplier inside the external closure reuses the cached bulk
/// forces instead of re-evaluating the mesh. The match is bitwise
/// on all inputs: any mismatch recomputes, so the cache stays
/// correct under any driver call order.
struct InternalForceCache {
    positions: Vec<f64>,
    velocities: Vec<f64>,
    time_bits: u64,
    forces: Vec<f64>,
    valid: bool,
}

impl InternalForceCache {
    fn new(dofs: usize) -> Self {
        Self {
            positions: vec![0.0; dofs],
            velocities: vec![0.0; dofs],
            time_bits: 0,
            forces: vec![0.0; dofs],
            valid: false,
        }
    }

    fn matches(&self, positions: &[f64], velocities: &[f64], time: f64) -> bool {
        self.valid
            && self.time_bits == time.to_bits()
            && self.positions.len() == positions.len()
            && self.velocities.len() == velocities.len()
            && self
                .positions
                .iter()
                .zip(positions.iter())
                .all(|(cached, fresh)| cached.to_bits() == fresh.to_bits())
            && self
                .velocities
                .iter()
                .zip(velocities.iter())
                .all(|(cached, fresh)| cached.to_bits() == fresh.to_bits())
    }

    fn store(&mut self, positions: &[f64], velocities: &[f64], time: f64, forces: &[f64]) {
        self.positions.copy_from_slice(positions);
        self.velocities.copy_from_slice(velocities);
        self.time_bits = time.to_bits();
        self.forces.copy_from_slice(forces);
        self.valid = true;
    }
}

/// Law1 internal forces through the finite-strain kernel.
///
/// Finite-strain fallback for decks whose strains exceed the linear
/// regime (taylor_bar impact reaches ~10%: the constant stiffness
/// misses its DZ probe by 5.5%). Same signature shape as the linear
/// path so both call sites branch on one flag.
fn eval_law1_internal_finite(
    material: &NeoHookean,
    mesh: &HexMesh,
    reference: &[f64],
    pinned: &[[bool; 3]],
    positions: &[f64],
    displacement: &mut [f64],
    forces: &mut [f64],
) {
    for (slot, (pos, pos0)) in displacement
        .iter_mut()
        .zip(positions.iter().zip(reference.iter()))
    {
        *slot = pos - pos0;
    }
    forces.fill(0.0);
    for (elem, conn_elem) in mesh.conn.iter().enumerate() {
        let residual = q1_element_residual(material, mesh, elem, displacement);
        for (slot, node) in conn_elem.iter().enumerate() {
            forces[3 * node] -= residual[3 * slot];
            forces[3 * node + 1] -= residual[3 * slot + 1];
            forces[3 * node + 2] -= residual[3 * slot + 2];
        }
    }
    // Ideal pins: zero force rows freeze pinned dofs exactly at
    // their zero initial state (the driver rejects zero masses,
    // so pinning lives in the force closures, not the mass list).
    for (node, axes) in pinned.iter().enumerate() {
        for axis in 0..3 {
            if axes[axis] {
                forces[3 * node + axis] = 0.0;
            }
        }
    }
}

/// Law1 internal forces through the precomputed linear stiffness.
///
/// Shared by the internal closure (which also stashes the cache)
/// and the external closure's cache-miss path, so both spellings
/// stay bit-identical by construction.
fn eval_law1_internal(
    conn: &[[usize; 8]],
    elem_stiffness: &[[f64; 576]],
    reference: &[f64],
    pinned: &[[bool; 3]],
    positions: &[f64],
    displacement: &mut [f64],
    forces: &mut [f64],
) {
    for (slot, (pos, pos0)) in displacement
        .iter_mut()
        .zip(positions.iter().zip(reference.iter()))
    {
        *slot = pos - pos0;
    }
    forces.fill(0.0);
    for (elem, conn_elem) in conn.iter().enumerate() {
        let stiff = &elem_stiffness[elem];
        let mut ue = [0.0; 24];
        for (slot, node) in conn_elem.iter().enumerate() {
            ue[3 * slot] = displacement[3 * node];
            ue[3 * slot + 1] = displacement[3 * node + 1];
            ue[3 * slot + 2] = displacement[3 * node + 2];
        }
        for (slot, node) in conn_elem.iter().enumerate() {
            let row = &stiff[3 * slot * 24..(3 * slot + 3) * 24];
            let fx: f64 = row[0..24].iter().zip(ue.iter()).map(|(k, v)| k * v).sum();
            let fy: f64 = row[24..48].iter().zip(ue.iter()).map(|(k, v)| k * v).sum();
            let fz: f64 = row[48..72].iter().zip(ue.iter()).map(|(k, v)| k * v).sum();
            forces[3 * node] -= fx;
            forces[3 * node + 1] -= fy;
            forces[3 * node + 2] -= fz;
        }
    }
    // Ideal pins: zero force rows freeze pinned dofs exactly at
    // their zero initial state (the driver rejects zero masses,
    // so pinning lives in the force closures, not the mass list).
    for (node, axes) in pinned.iter().enumerate() {
        for axis in 0..3 {
            if axes[axis] {
                forces[3 * node + axis] = 0.0;
            }
        }
    }
}

/// One timed native explicit repeat over the parsed deck model.
fn sonium_repeat(
    deck: &str,
    model: &NativeModel,
    probes: &[Probe],
) -> Result<RepeatSample, String> {
    sonium_repeat_observed(deck, model, probes, 1.0, None)
}

/// Original elastic integration with optional raw observation capture.
fn sonium_repeat_observed(
    deck: &str,
    model: &NativeModel,
    probes: &[Probe],
    timestep_scale: f64,
    capture: Option<&mut NativeElasticHistory>,
) -> Result<RepeatSample, String> {
    if !matches!(model.material, NativeMaterial::Law1) {
        return sonium_repeat_plastic(deck, model, probes);
    }
    let mesh = HexMesh::new(model.coords.clone(), model.bricks.clone());
    let num_nodes = mesh.num_nodes();
    let num_elems = mesh.conn.len();
    if num_nodes == 0 || num_elems == 0 {
        return Err("native model holds no mesh".to_string());
    }
    let mut masses = vec![0.0; num_nodes];
    let mut char_lengths = Vec::with_capacity(num_elems);
    for elem in 0..num_elems {
        let volume = element_volume(&mesh, elem);
        if !volume.is_finite() || volume <= 0.0 {
            return Err(format!(
                "element {elem} has non-positive volume {volume:.6e}"
            ));
        }
        for node in mesh.conn[elem] {
            masses[node] += model.density * volume / 8.0;
        }
        char_lengths.push(min_edge_length(&mesh, elem));
    }
    let min_mass = masses.iter().cloned().fold(f64::INFINITY, f64::min);
    if !min_mass.is_finite() || min_mass <= 0.0 {
        return Err(format!("non-positive lumped mass {min_mass:.6e}"));
    }
    let material = NeoHookean::from_young(model.youngs, model.poisson, 3);
    let dilatational = ((material.lambda + 2.0 * material.mu) / model.density).sqrt();
    let sound_speeds = vec![dilatational; num_elems];
    // Linearized bulk only where strains stay infinitesimal: hertz
    // contact (~0.1%) matches the finite-strain kernel to probe
    // tolerance, while taylor_bar impact (~10%) misses DZ by 5.5%.
    let linear_bulk = deck == "hertz_sphere";
    // Linearized bulk for the small-strain LAW1 decks: one element
    // stiffness precomputed per element replaces the finite-strain
    // kernel (Jacobians, determinants, logarithms at 8 Gauss points
    // per evaluation) with a single dgemv at second-order error.
    let elem_stiffness: Vec<[f64; 576]> = if linear_bulk {
        (0..num_elems)
            .map(|elem| {
                linear_hex_stiffness(material.lambda, material.mu, &mesh.element_coords(elem))
            })
            .collect()
    } else {
        Vec::new()
    };
    // Contact is an exact Lagrange multiplier in the dynamics (no
    // spring, no stiffness), so every deck rides the bulk CFL cap:
    // the old explicit-penalty `sqrt(m/k)` cap cost wall decks ~12x
    // in step count for identical statics.
    let dt_max = critical_dt(&char_lengths, &sound_speeds) * timestep_scale;
    if !dt_max.is_finite() || dt_max <= 0.0 {
        return Err("native dt bound is not positive".to_string());
    }
    // Sound pre-check: steps can only exceed stop/dt_max (dt never
    // passes the cap), so an over-budget estimate rejects instantly
    // instead of burning hours on a quasi-static deck.
    let estimated = (model.stop / dt_max).ceil() as u64;
    if estimated > MAX_NATIVE_STEPS {
        return Err(format!(
            "native run needs about {estimated} steps (stop {}s, dt_max {dt_max:.3e}s): quasi-static timescale deferred (needs implicit or mass-scaling support)",
            model.stop
        ));
    }
    println!(
        "sonium mesh: {num_nodes} nodes, {num_elems} bricks, dt_max {dt_max:.3e}s, about {estimated} steps"
    );
    let config = AdaptiveDtConfig::new(0.9, 1.0e-12, dt_max, 1.1)
        .map_err(|error| format!("adaptive dt config rejected: {error}"))?;
    let mut driver = ExplicitStepDriver::new(config)
        .map_err(|error| format!("step driver rejected: {error}"))?;
    let mut state = ExplicitState::new(num_nodes);
    for (node, xyz) in model.coords.iter().enumerate() {
        state.positions[3 * node] = xyz[0];
        state.positions[3 * node + 1] = xyz[1];
        state.positions[3 * node + 2] = xyz[2];
    }
    state.masses.copy_from_slice(&masses);
    let reference: Vec<f64> = state.positions.clone();
    // Wall-force history per wall: (time, force on wall). Stride keeps
    // sampling bounded on long runs without biasing the tail mean.
    let stride = ((estimated / MAX_FORCE_SAMPLES as u64).max(1)) as usize;

    // Reused scratch: the driver calls force closures sequentially (never
    // re-entrantly), so one displacement buffer serves every evaluation and
    // the per-step allocation drops to the returned force vector alone (the
    // driver takes ownership of it, hence it cannot be recycled here).
    let displacement_buf = RefCell::new(vec![0.0; 3 * num_nodes]);
    // Multiplier contact state shared by the external closure (which
    // computes the exact constraint force) and the loop (which
    // predicts engagement and adopts velocities).
    let contact_cache = RefCell::new(InternalForceCache::new(3 * num_nodes));
    let contact_recompute = RefCell::new(vec![0.0; 3 * num_nodes]);
    let engaged: RefCell<Vec<Vec<bool>>> = RefCell::new(
        model
            .walls
            .iter()
            .map(|wall| vec![false; wall.slaves.len()])
            .collect(),
    );
    let transitions: RefCell<Vec<(usize, usize)>> = RefCell::new(Vec::new());
    let wall_force: RefCell<Vec<[f64; 3]>> = RefCell::new(vec![[0.0; 3]; model.walls.len()]);
    let running_buf = RefCell::new(vec![0.0; 3 * num_nodes]);
    let internal_force = |positions: &[f64], velocities: &[f64], time: f64| {
        let mut forces = vec![0.0; positions.len()];
        {
            let mut displacement = displacement_buf.borrow_mut();
            if linear_bulk {
                eval_law1_internal(
                    &mesh.conn,
                    &elem_stiffness,
                    &reference,
                    &model.pinned,
                    positions,
                    &mut displacement,
                    &mut forces,
                );
            } else {
                eval_law1_internal_finite(
                    &material,
                    &mesh,
                    &reference,
                    &model.pinned,
                    positions,
                    &mut displacement,
                    &mut forces,
                );
            }
        }
        contact_cache
            .borrow_mut()
            .store(positions, velocities, time, &forces);
        forces
    };
    // Deck walls move at constant velocity (no acceleration card).
    let wall_acceleration = [0.0; 3];
    let external_force = |positions: &[f64], velocities: &[f64], time: f64| {
        let mut forces = vec![0.0; positions.len()];
        for cload in &model.cloads {
            let shape = model
                .functs
                .get(&cload.funct)
                .and_then(|points| eval_funct(points, time).ok());
            let Some(shape) = shape else { continue };
            for node in &cload.nodes {
                forces[3 * node + cload.axis] += cload.scale * shape;
            }
        }
        // Bulk forces for the multiplier: cached from the internal
        // closure's identical evaluation, recomputed on any miss.
        let cached = contact_cache.borrow().matches(positions, velocities, time);
        if !cached {
            let mut displacement = displacement_buf.borrow_mut();
            let mut recomputed = contact_recompute.borrow_mut();
            if linear_bulk {
                eval_law1_internal(
                    &mesh.conn,
                    &elem_stiffness,
                    &reference,
                    &model.pinned,
                    positions,
                    &mut displacement,
                    &mut recomputed,
                );
            } else {
                eval_law1_internal_finite(
                    &material,
                    &mesh,
                    &reference,
                    &model.pinned,
                    positions,
                    &mut displacement,
                    &mut recomputed,
                );
            }
            drop(displacement);
            contact_cache
                .borrow_mut()
                .store(positions, velocities, time, &recomputed);
        }
        let mut running = running_buf.borrow_mut();
        running.copy_from_slice(&forces);
        let cache = contact_cache.borrow();
        let mut wall_forces = wall_force.borrow_mut();
        let mut engaged_flags = engaged.borrow_mut();
        let mut fresh = transitions.borrow_mut();
        // Local transition staging: empty Vec allocates nothing, and
        // pushes only happen on the rare engagement event.
        let mut local: Vec<usize> = Vec::new();
        for (index, wall) in model.walls.iter().enumerate() {
            let point = [
                wall.point[0] + wall.velocity[0] * time,
                wall.point[1] + wall.velocity[1] * time,
                wall.point[2] + wall.velocity[2] * time,
            ];
            // Penalty unused: the multiplier is an exact constraint.
            let mut rigid = RigidWall::new(point, wall.normal, 0.0);
            rigid.velocity = wall.velocity;
            local.clear();
            let on_wall = rigid_wall_multiplier(
                &rigid,
                wall_acceleration,
                &wall.slaves,
                &masses,
                positions,
                velocities,
                &cache.forces,
                &running,
                &mut engaged_flags[index],
                &mut local,
                &mut forces,
            );
            fresh.extend(local.iter().map(|&slot| (index, slot)));
            wall_forces[index] = on_wall;
            running.copy_from_slice(&forces);
        }
        for (node, axes) in model.pinned.iter().enumerate() {
            for axis in 0..3 {
                if axes[axis] {
                    forces[3 * node + axis] = 0.0;
                }
            }
        }
        forces
    };

    let start = std::time::Instant::now();
    let mut steps: u64 = 0;
    // Wall histories sampled inline per step with stride (kept outside
    // the force closures so the driver sees pure evaluations).
    let mut sampled: Vec<Vec<(f64, [f64; 3])>> = vec![Vec::new(); model.walls.len()];
    // Accumulated adoption kinetic jumps (signed wall work): the
    // driver's ledger never sees these boundary-condition resets,
    // so the drift gate adds them back (E = D - J).
    let mut adopt_work = 0.0;
    let targets = if capture.is_some() {
        probes
            .iter()
            .map(|p| resolve_probe_channel(&p.channel, model))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        Vec::new()
    };
    let mut times = if capture.is_some() {
        vec![0.0]
    } else {
        Vec::new()
    };
    let mut values = targets.iter().map(|_| vec![0.0]).collect::<Vec<_>>();
    while state.time < model.stop {
        if steps >= MAX_NATIVE_STEPS {
            return Err(format!(
                "stalled: {MAX_NATIVE_STEPS} steps reached at t = {:.6e}s, short of stop {:.6e}s",
                state.time, model.stop
            ));
        }
        // Engagement is detected by the multiplier inside the
        // step (standard end-of-step contact detection, one step
        // of plunge like any explicit contact); the adoption
        // below then freezes the normal motion.
        transitions.borrow_mut().clear();
        let step = if capture.is_some() {
            // The observed run uses the same bulk CFL safety factor, and
            // clips its final step to the original physical stop time.
            let dt = (0.9 * critical_dt(&char_lengths, &sound_speeds) * timestep_scale)
                .min(model.stop - state.time);
            driver.step_fixed(&mut state, internal_force, external_force, dt)
        } else {
            driver.step(
                &mut state,
                internal_force,
                external_force,
                &char_lengths,
                &sound_speeds,
            )
        };
        if let Err(error) = step {
            return Err(format!(
                "unstable at t = {:.6e}s after {steps} steps: {error}",
                state.time
            ));
        }
        // Adoption for this step's engagements: kills the normal
        // approach velocity (one inelastic reset), freezing the
        // one-step plunge the detection incurred. Jumps tracked
        // for the drift ledger (wall work outside driver books).
        {
            let fresh = transitions.borrow();
            let engaged_flags = engaged.borrow();
            for &(index, slot) in fresh.iter() {
                if !engaged_flags[index][slot] {
                    continue;
                }
                let wall = &model.walls[index];
                let slave = wall.slaves[slot];
                let relative = (state.velocities[3 * slave] - wall.velocity[0]) * wall.normal[0]
                    + (state.velocities[3 * slave + 1] - wall.velocity[1]) * wall.normal[1]
                    + (state.velocities[3 * slave + 2] - wall.velocity[2]) * wall.normal[2];
                if relative < 0.0 {
                    let mass = state.masses[slave];
                    let before = state.velocities[3 * slave] * state.velocities[3 * slave]
                        + state.velocities[3 * slave + 1] * state.velocities[3 * slave + 1]
                        + state.velocities[3 * slave + 2] * state.velocities[3 * slave + 2];
                    state.velocities[3 * slave] -= relative * wall.normal[0];
                    state.velocities[3 * slave + 1] -= relative * wall.normal[1];
                    state.velocities[3 * slave + 2] -= relative * wall.normal[2];
                    let after = state.velocities[3 * slave] * state.velocities[3 * slave]
                        + state.velocities[3 * slave + 1] * state.velocities[3 * slave + 1]
                        + state.velocities[3 * slave + 2] * state.velocities[3 * slave + 2];
                    adopt_work += 0.5 * mass * (after - before);
                }
            }
        }
        let sample = ((steps + 1) as usize).is_multiple_of(stride);
        if sample {
            let wall_forces = wall_force.borrow();
            for (index, forces) in wall_forces.iter().enumerate() {
                sampled[index].push((state.time, *forces));
            }
        }
        steps += 1;
        if capture.is_some() {
            times.push(state.time);
            let wall_forces = wall_force.borrow();
            for (slot, target) in targets.iter().enumerate() {
                let value = match *target {
                    ProbeTarget::Displacement { axis, node } => {
                        state.positions[3 * node + axis] - reference[3 * node + axis]
                    }
                    ProbeTarget::WallForce { axis, wall } => wall_forces[wall][axis],
                    _ => {
                        return Err(
                            "elastic observations require displacement or wall force".into()
                        );
                    }
                };
                values[slot].push(value);
            }
        }
    }
    let wall_s = start.elapsed().as_secs_f64();
    println!(
        "sonium run: reached t = {:.6e}s in {steps} steps",
        state.time
    );
    let mut failures = Vec::new();
    for probe in probes {
        let target = match resolve_probe_channel(&probe.channel, model) {
            Ok(target) => target,
            Err(error) => {
                failures.push(error);
                continue;
            }
        };
        let actual = match target {
            ProbeTarget::Displacement { axis, node } => {
                state.positions[3 * node + axis] - reference[3 * node + axis]
            }
            ProbeTarget::WallForce { axis, wall } => {
                let history = &sampled[wall];
                if history.is_empty() {
                    failures.push(format!("probe {} has no wall-force samples", probe.label));
                    continue;
                }
                // Impact walls (initial gap) settle past the initial
                // transient, so the probe reads the second-half plateau
                // mean; touching walls load quasi-statically, so the
                // probe reads the final load.
                let wall = &model.walls[wall];
                let mut gap = f64::INFINITY;
                for slave in &wall.slaves {
                    let position = model.coords[*slave];
                    let distance = (position[0] - wall.point[0]) * wall.normal[0]
                        + (position[1] - wall.point[1]) * wall.normal[1]
                        + (position[2] - wall.point[2]) * wall.normal[2];
                    gap = gap.min(distance);
                }
                if gap <= TOUCHING_GAP_M {
                    history[history.len() - 1].1[axis]
                } else {
                    let half = history.len() / 2;
                    let window = &history[half..];
                    window.iter().map(|(_, force)| force[axis]).sum::<f64>() / window.len() as f64
                }
            }
            ProbeTarget::ElemStress { .. } | ProbeTarget::ElemPlas { .. } => {
                failures.push(format!(
                    "probe {} needs element state from a plastic run",
                    probe.label
                ));
                continue;
            }
        };
        if capture.is_some() {
            // Captured wall-force channels have no benchmark target. Their
            // actual histories are validated by the independent QA reference.
            println!(
                "measured {} [{}]: actual {actual:.6e}",
                probe.label, probe.channel
            );
            continue;
        }
        println!(
            "probe {} [{}]: actual {actual:.6e}, expected {:.6e}",
            probe.label, probe.channel, probe.expected
        );
        if let Err(error) = assert_probe_close(
            actual,
            probe.expected,
            probe.abs_tol,
            probe.rel_tol,
            &probe.label,
        ) {
            failures.push(error.to_string());
        }
    }
    for failure in &failures {
        println!("FAIL: {failure}");
    }
    let balance = driver.balance().clone();
    let energy_scale = balance.kinetic + balance.internal.abs();
    // The multiplier force flows through the driver's external
    // channel (its work is in Wext), but the adoption jumps reset
    // velocities outside the driver's books: true drift adds that
    // wall work back, E = D - J.
    let drift = balance.drift - adopt_work;
    let drift_frac = if energy_scale > 0.0 {
        drift.abs() / energy_scale
    } else {
        f64::INFINITY
    };
    println!(
        "energy: KE {:.6e}J, IE {:.6e}J, Wext {:.6e}J, Wadopt {:.6e}J, drift {:.3e} rel",
        balance.kinetic, balance.internal, balance.external_work, adopt_work, drift_frac
    );
    if !drift_frac.is_finite() || drift_frac > DRIFT_TOL {
        failures.push(format!(
            "energy drift {drift_frac:.3e} exceeds {DRIFT_TOL:.3e}"
        ));
    }
    let dof = u64::try_from(num_nodes)
        .ok()
        .and_then(|nodes| nodes.checked_mul(3))
        .ok_or_else(|| "node count overflows dof".to_string())?;
    if capture.is_some() {
        println!("native history: wall {wall_s:.3}s, cycles {steps}, dof {dof}");
    } else {
        println!(
            "sonium: wall {wall_s:.3}s, cycles {steps}, dof {dof}, correct {}",
            failures.is_empty()
        );
    }
    if let Some(capture) = capture {
        *capture = NativeElasticHistory {
            probe_channels: probes.iter().map(|p| p.channel.clone()).collect(),
            times_s: times,
            probe_values: values,
            final_time_s: state.time,
            steps,
            relative_energy_drift: drift_frac,
            ..NativeElasticHistory::default()
        };
    }
    Ok(RepeatSample {
        wall_s,
        cycles: steps,
        dof,
        peak_rss_bytes: self_peak_rss_bytes(),
        correct: failures.is_empty(),
    })
}

/// Raw native displacement and contact-force histories for elastic validation.
#[derive(Debug, Default, serde::Serialize)]
pub struct NativeElasticHistory {
    pub probe_channels: Vec<String>,
    pub times_s: Vec<f64>,
    pub probe_values: Vec<Vec<f64>>,
    pub final_time_s: f64,
    pub steps: u64,
    pub relative_energy_drift: f64,
    pub starter_sha256: String,
    pub engine_deck_sha256: String,
}

/// Measure the original full elastic sphere squeeze and both wall reactions.
///
/// Uses original checked-in decks and actual native elastic/multiplier contact
/// integration. Benchmark verdicts are not used to establish QA parity.
///
/// # Errors
/// Rejects invalid CFL scale, nonelastic material, integration failures, and
/// incomplete or nonfinite histories.
pub fn original_hertz_history(timestep_scale: f64) -> Result<NativeElasticHistory, String> {
    use sha2::{Digest, Sha256};
    if !timestep_scale.is_finite() || timestep_scale <= 0.0 || timestep_scale > 1.0 {
        return Err("native Hertz CFL scale must be in (0,1]".into());
    }
    let decks = resolve_c3_decks("hertz_sphere")?;
    let model = parse_native_model(&decks)?;
    if !matches!(model.material, NativeMaterial::Law1) {
        return Err("original elastic Hertz material required".into());
    }
    let mut probes = read_probe_file(&decks.probes)?;
    for wall in [1, 2] {
        probes.push(Probe {
            label: format!("wall {wall} force"),
            channel: format!("FNZ/RWALL/{wall}"),
            expected: 0.0,
            abs_tol: 0.0,
            rel_tol: 0.0,
        });
    }
    let mut history = NativeElasticHistory::default();
    sonium_repeat_observed(
        "hertz_sphere",
        &model,
        &probes,
        timestep_scale,
        Some(&mut history),
    )?;
    if history.steps == 0
        || history.times_s.len() < 2
        || (history.final_time_s - model.stop).abs() > 1e-12
        || history.probe_values.len() != probes.len()
        || history
            .probe_values
            .iter()
            .any(|v| v.len() != history.times_s.len() || v.iter().any(|x| !x.is_finite()))
        || !history.relative_energy_drift.is_finite()
    {
        return Err("incomplete finite original native Hertz history".into());
    }
    history.starter_sha256 = format!(
        "{:x}",
        Sha256::digest(std::fs::read(decks.starter).map_err(|e| e.to_string())?)
    );
    history.engine_deck_sha256 = format!(
        "{:x}",
        Sha256::digest(std::fs::read(decks.engine).map_err(|e| e.to_string())?)
    );
    Ok(history)
}

/// Minimum edge length of one brick at the given positions.
fn min_edge_length_at(positions: &[f64], conn: &[usize; 8]) -> f64 {
    let mut shortest = f64::INFINITY;
    for (a, b) in HEX_EDGES {
        let dx = positions[3 * conn[a]] - positions[3 * conn[b]];
        let dy = positions[3 * conn[a] + 1] - positions[3 * conn[b] + 1];
        let dz = positions[3 * conn[a] + 2] - positions[3 * conn[b] + 2];
        shortest = shortest.min((dx * dx + dy * dy + dz * dz).sqrt());
    }
    shortest
}

/// One timed native explicit repeat for LAW2/LAW36 brick decks.
///
/// Dispatches on the parsed material into the generic plastic core
/// below; LAW1 never reaches here (guarded by the caller).
fn sonium_repeat_plastic(
    deck: &str,
    model: &NativeModel,
    probes: &[Probe],
) -> Result<RepeatSample, String> {
    match &model.material {
        NativeMaterial::Law2 {
            material,
            t0,
            fail_ep,
        } => {
            let states: Vec<HexExplicitState<Law2State>> = model
                .bricks
                .iter()
                .map(|_| HexExplicitState::new(Law2State::new(*t0)))
                .collect();
            let criterion = ErosionCriterion {
                max_equiv_plastic_strain: *fail_ep,
                max_damage: 1.0,
                max_porosity: f64::INFINITY,
                max_tabular_damage: f64::INFINITY,
            };
            repeat_plastic_core(
                deck,
                model,
                probes,
                material,
                states,
                criterion,
                NativeRunOptions::default(),
            )
        }
        NativeMaterial::Law36 { material } => {
            let states: Vec<HexExplicitState<Law36State>> = model
                .bricks
                .iter()
                .map(|_| HexExplicitState::new(Law36State::new()))
                .collect();
            // The C3 LAW36 subset carries no failure data (verified
            // off-scale sentinels at parse), so erosion stays off and
            // the sweep below only deletes inverted elements.
            let criterion = ErosionCriterion {
                max_equiv_plastic_strain: f64::INFINITY,
                max_damage: 1.0,
                max_porosity: f64::INFINITY,
                max_tabular_damage: f64::INFINITY,
            };
            repeat_plastic_core(
                deck,
                model,
                probes,
                material,
                states,
                criterion,
                NativeRunOptions::default(),
            )
        }
        NativeMaterial::Law1 => Err("plastic repeat called on a LAW1 model".to_string()),
    }
}

/// Complete native plastic observations, independent of benchmark verdicts.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct NativePlasticityHistory {
    /// Artificial pressure enters tensile-positive Cauchy stress as minus q times identity.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artificial_bulk_pressure_convention: Option<String>,
    /// Actual quadratic and linear artificial bulk-viscosity coefficients, in that order.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bulk_viscosity_coefficients: Option<[f64; 2]>,
    /// Explicit diagnostic rigid-wall formulation, when selected.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rigid_wall_formulation: Option<String>,
    /// Actual kinetic-energy change from normal engagement velocity adoption.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wall_adoption_kinetic_change_j: Option<f64>,
    /// Physical moving-wall work during normal engagement impulses.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wall_adoption_work_j: Option<f64>,
    /// Inelastic loss during normal engagement impulses.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wall_adoption_dissipation_j: Option<f64>,
    /// Accumulated engagement impulses on nodes, in original wall order.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub wall_adoption_impulses_kg_m_s: Vec<[f64; 3]>,
    /// Optional explicitly selected integration formulation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub integration_formulation: Option<String>,
    /// Explicitly requested bulk execution mode; numerical equations are unchanged.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bulk_execution: Option<String>,
    /// Selected self-contact detection execution policy, when explicitly present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub self_contact_execution: Option<String>,
    /// Actual penetrating native self-contact pairs, if explicitly enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub self_contact_peak_pairs: Option<usize>,
    /// Final normal-spring contact energy; independent of the work ledger.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_self_contact_penalty_energy_j: Option<f64>,
    pub probe_channels: Vec<String>,
    pub times_s: Vec<f64>,
    pub probe_values: Vec<Vec<f64>>,
    pub element_ids: Vec<i64>,
    pub final_element_plastic_strain: Vec<f64>,
    pub final_time_s: f64,
    pub steps: u64,
    pub kinetic_energy_j: f64,
    pub internal_energy_j: f64,
    pub external_work_j: f64,
    pub relative_energy_drift: f64,
    pub eroded_elements: usize,
    pub starter_sha256: String,
    pub engine_deck_sha256: String,
    /// Original deck node ids for the measured final profile.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub node_ids: Vec<i64>,
    /// Final physical coordinates, in the same order as `node_ids`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub final_positions_m: Vec<[f64; 3]>,
    /// Requested observation time inside the complete physical run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile_sample_time_s: Option<f64>,
    /// Physical coordinates interpolated between actual steps at that time.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub profile_sample_positions_m: Vec<[f64; 3]>,
    /// Actual element plastic strains interpolated at the same observation time.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub profile_sample_element_plastic_strain: Vec<f64>,
    /// Current-volume averaged Gauss plastic strain, interpolated at the same profile time.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub profile_sample_element_volume_average_plastic_strain: Vec<f64>,
}

#[derive(Debug)]
struct NativeRunOptions<'a> {
    timestep_scale: f64,
    penalty_scale: f64,
    objective: bool,
    gauss_bbar: bool,
    parallel_gauss: bool,
    self_contact:
        Option<&'a RefCell<sonium_fem::physics::solid::contact::explicit::ExplicitContact>>,
    profile_sample_time_s: Option<f64>,
    capture: Option<&'a mut NativePlasticityHistory>,
}

impl Default for NativeRunOptions<'_> {
    fn default() -> Self {
        Self {
            timestep_scale: 1.0,
            penalty_scale: 1.0,
            objective: false,
            gauss_bbar: false,
            parallel_gauss: false,
            self_contact: None,
            profile_sample_time_s: None,
            capture: None,
        }
    }
}

fn native_gauss_parallel() -> Result<bool, String> {
    match std::env::var("SONIUM_NATIVE_GAUSS_PARALLEL").as_deref() {
        Ok("1") => Ok(true),
        Ok("0") | Err(_) => Ok(false),
        Ok(_) => Err("SONIUM_NATIVE_GAUSS_PARALLEL must be 0 or 1".into()),
    }
}

struct GaussBulkStep<'a> {
    positions: &'a [f64],
    velocities: &'a [f64],
    dt: f64,
    parallel: bool,
}

// Material histories belong exclusively to their element. Indexed collection
// preserves element order; the caller scatters forces serially in that order.
fn gauss_bulk_responses<M: ObjectivePlasticMaterial + Sync>(
    mesh: &HexMesh,
    states: &mut [HexGaussPlasticState<M::State>],
    material: &M,
    step: &GaussBulkStep<'_>,
    params: &HexExplicitParams,
    active: &[bool],
) -> Result<Vec<Option<Result<HexPlasticOut, HexExplicitError>>>, HexExplicitError>
where
    M::State: Send,
{
    if states.len() != mesh.conn.len()
        || active.len() != states.len()
        || step.positions.len() != 3 * mesh.num_nodes()
        || step.velocities.len() != step.positions.len()
    {
        return Err(HexExplicitError::new(
            "Gauss bulk states, connectivity and kinematics must align",
        ));
    }
    let update = |(elem, state): (usize, &mut HexGaussPlasticState<M::State>)| {
        if !active[elem] {
            return None;
        }
        let conn = &mesh.conn[elem];
        let coordinates =
            std::array::from_fn(|i| std::array::from_fn(|a| step.positions[3 * conn[i] + a]));
        let velocity = std::array::from_fn(|i| step.velocities[3 * conn[i / 3] + i % 3]);
        Some(HexGaussPlasticGeom::new(coordinates).and_then(|geom| {
            hex_gauss_plastic_forces(&geom, state, material, &velocity, step.dt, params)
        }))
    };
    Ok(if step.parallel {
        states.par_iter_mut().enumerate().map(update).collect()
    } else {
        states.iter_mut().enumerate().map(update).collect()
    })
}

fn native_centroid_parallel(elements: usize) -> Result<bool, String> {
    match std::env::var("SONIUM_NATIVE_CENTROID_PARALLEL").as_deref() {
        Ok("1") => Ok(true),
        Ok("0") => Ok(false),
        Err(std::env::VarError::NotPresent) => {
            // Larger independent material updates amortize pool dispatch;
            // smaller meshes retain the original serial path.
            Ok(elements >= 256 && rayon::current_num_threads() > 1)
        }
        _ => Err("SONIUM_NATIVE_CENTROID_PARALLEL must be0or1".into()),
    }
}

fn centroid_bulk_responses<M: ObjectivePlasticMaterial + Sync>(
    mesh: &HexMesh,
    states: &mut [HexExplicitState<M::State>],
    material: &M,
    step: &GaussBulkStep<'_>,
    params: &HexExplicitParams,
    active: &[bool],
    objective: bool,
) -> Result<Vec<Option<Result<HexPlasticOut, HexExplicitError>>>, HexExplicitError>
where
    M::State: Send,
{
    if states.len() != mesh.conn.len()
        || active.len() != states.len()
        || step.positions.len() != 3 * mesh.num_nodes()
        || step.velocities.len() != step.positions.len()
    {
        return Err(HexExplicitError::new(
            "centroid bulk states, connectivity and kinematics must align",
        ));
    }
    let update = |(elem, state): (usize, &mut HexExplicitState<M::State>)| {
        if !active[elem] {
            return None;
        }
        let conn = &mesh.conn[elem];
        let coordinates =
            std::array::from_fn(|i| std::array::from_fn(|a| step.positions[3 * conn[i] + a]));
        let velocity = std::array::from_fn(|i| step.velocities[3 * conn[i / 3] + i % 3]);
        Some(HexExplicitGeom::new(coordinates).and_then(|geom| {
            if objective {
                hex_plastic_forces_objective(&geom, state, material, &velocity, step.dt, params)
            } else {
                hex_plastic_forces(&geom, state, material, &velocity, step.dt, params)
            }
        }))
    };
    Ok(if step.parallel {
        states.par_iter_mut().enumerate().map(update).collect()
    } else {
        states.iter_mut().enumerate().map(update).collect()
    })
}

/// Measure the original rail's complete LAW1 elastic compression history.
///
/// Uses the isotropic C1 elastic stress update through a tabulated adapter whose
/// unreachable yield limit is checked to remain inactive. Geometry, E/nu/rho,
/// wall directions, velocities, eligibility groups and duration come from
/// the original decks. Penalty and timestep refinements are numerical only.
/// This supplies observations; full plastic crushing is a separate requirement.
///
/// # Errors
/// Rejects invalid options, non-LAW1 decks, failed integration, plasticity,
/// erosion, nonfinite fields or an incomplete original physical trajectory.
pub fn rail_elastic_history(
    directory: &Path,
    timestep_scale: f64,
    penalty_scale: f64,
) -> Result<NativePlasticityHistory, String> {
    objective_elastic_history(
        directory,
        "rail_crush",
        timestep_scale,
        penalty_scale,
        false,
    )
}

/// Measure the original full warping transient with objective eight-point solids.
///
/// Uses the unmodified LAW1 decks, original force couple and root restraints.
/// The elastic adapter rejects any plastic increment. Observations do not
/// certify reference parity, mesh convergence or release admission.
///
/// # Errors
/// Rejects invalid CFL scales, non-LAW1 decks, failed integration, plasticity,
/// erosion, nonfinite fields or incomplete original physical trajectories.
pub fn warping_elastic_history(
    directory: &Path,
    timestep_scale: f64,
) -> Result<NativePlasticityHistory, String> {
    objective_elastic_history(directory, "beam_warping", timestep_scale, 1.0, true)
}

fn objective_elastic_history(
    directory: &Path,
    deck: &str,
    timestep_scale: f64,
    penalty_scale: f64,
    gauss_bbar: bool,
) -> Result<NativePlasticityHistory, String> {
    use sha2::{Digest, Sha256};

    if !timestep_scale.is_finite()
        || !(0.0 < timestep_scale && timestep_scale <= 1.0)
        || !penalty_scale.is_finite()
        || penalty_scale < 1.0
    {
        return Err(
            "objective elastic solid requires CFL scale in(0,1] and penalty scale>=1".into(),
        );
    }
    let decks = C3Decks {
        starter: directory.join(format!("{deck}_0000.rad")),
        engine: directory.join(format!("{deck}_0001.rad")),
        probes: directory.join("probes.json"),
    };
    let model = parse_native_model(&decks)?;
    if !matches!(model.material, NativeMaterial::Law1) {
        return Err("objective elastic solid requires LAW1".into());
    }
    let probes = read_probe_file(&decks.probes)?;
    for probe in &probes {
        resolve_probe_channel(&probe.channel, &model)?;
    }
    // LAW1 has no yield surface. This adapter remains elastic even at the
    // original1% compression; any plastic increment rejects the observation.
    let material = rail_law1_adapter(model.youngs, model.poisson)?;
    let initial = model
        .bricks
        .iter()
        .map(|_| HexExplicitState::new(Law36State::new()))
        .collect();
    let mut history = NativePlasticityHistory::default();
    repeat_plastic_core(
        deck,
        &model,
        &probes,
        &material,
        initial,
        ErosionCriterion {
            max_equiv_plastic_strain: f64::INFINITY,
            max_damage: f64::INFINITY,
            max_porosity: f64::INFINITY,
            max_tabular_damage: f64::INFINITY,
        },
        NativeRunOptions {
            timestep_scale,
            penalty_scale,
            objective: true,
            gauss_bbar,
            parallel_gauss: gauss_bbar && native_gauss_parallel()?,
            capture: Some(&mut history),
            ..NativeRunOptions::default()
        },
    )?;
    history.times_s.insert(0, 0.0);
    for values in &mut history.probe_values {
        values.insert(0, 0.0);
    }
    if history.steps == 0
        || history.times_s.len() < 2
        || (history.final_time_s - model.stop).abs() > 1e-12
        || history.eroded_elements != 0
        || history.final_element_plastic_strain.len() != model.bricks.len()
        || history
            .final_element_plastic_strain
            .iter()
            .any(|v| *v != 0.0)
        || history.node_ids.len() != model.coords.len()
        || history.final_positions_m.len() != model.coords.len()
        || history.probe_values.len() != probes.len()
        || history
            .probe_values
            .iter()
            .any(|v| v.len() != history.times_s.len())
        || history
            .times_s
            .iter()
            .chain(history.probe_values.iter().flatten())
            .chain(history.final_positions_m.iter().flatten())
            .any(|v| !v.is_finite())
        || !history.relative_energy_drift.is_finite()
    {
        return Err("complete finite and purely elastic original solid history required".into());
    }
    history.starter_sha256 = format!(
        "{:x}",
        Sha256::digest(std::fs::read(&decks.starter).map_err(|e| e.to_string())?)
    );
    history.engine_deck_sha256 = format!(
        "{:x}",
        Sha256::digest(std::fs::read(&decks.engine).map_err(|e| e.to_string())?)
    );
    Ok(history)
}

fn rail_law1_adapter(youngs: f64, poisson: f64) -> Result<Law36, String> {
    // Reuse the existing objective tensor-history support. The yield table
    // only selects the elastic branch; no plastic event may be admitted.
    let curve = RateCurve::new(0.0, vec![0.0, 1.0], vec![1e20, 1e20], 1.0)
        .map_err(|e| format!("invalid elastic rail adapter: {e}"))?;
    Ok(Law36::new(
        youngs,
        poisson,
        vec![curve],
        0.0,
        f64::INFINITY,
        f64::INFINITY,
        f64::INFINITY,
        f64::INFINITY,
    ))
}

/// Run the original full plastic cylinder and return measured native histories.
///
/// Consumes the same checked-in C3 decks and native constitutive/integration
/// code as the benchmark. `timestep_scale` in `(0,1]` refines the bulk CFL bound.
/// It returns observations without promoting the benchmark verdict to QA parity.
///
/// # Errors
/// Returns an error for invalid timestep scale, unavailable/malformed original
/// decks, incompatible material, failed native integration or incomplete traces.
pub fn original_thick_cylinder_history(
    timestep_scale: f64,
) -> Result<NativePlasticityHistory, String> {
    use sha2::{Digest, Sha256};
    if !timestep_scale.is_finite() || timestep_scale <= 0.0 || timestep_scale > 1.0 {
        return Err("native CFL timestep scale must be in (0,1]".into());
    }
    let decks = resolve_c3_decks("plasticity_thick_cylinder")?;
    let model = parse_native_model(&decks)?;
    let probes = read_probe_file(&decks.probes)?;
    let NativeMaterial::Law36 { material } = &model.material else {
        return Err("original thick-cylinder LAW36 material required".into());
    };
    let states = model
        .bricks
        .iter()
        .map(|_| HexExplicitState::new(Law36State::new()))
        .collect();
    let criterion = ErosionCriterion {
        max_equiv_plastic_strain: f64::INFINITY,
        max_damage: 1.0,
        max_porosity: f64::INFINITY,
        max_tabular_damage: f64::INFINITY,
    };
    let mut history = NativePlasticityHistory::default();
    repeat_plastic_core(
        "plasticity_thick_cylinder",
        &model,
        &probes,
        material,
        states,
        criterion,
        NativeRunOptions {
            timestep_scale,
            capture: Some(&mut history),
            ..NativeRunOptions::default()
        },
    )?;
    // The original initial state is unstressed and at rest. Keep it explicit
    // in the exported physical trace; benchmark sampling remains unchanged.
    history.times_s.insert(0, 0.0);
    for values in &mut history.probe_values {
        values.insert(0, 0.0);
    }
    if history.steps == 0
        || history.times_s.len() < 2
        || history.probe_values.len() != probes.len()
        || (history.final_time_s - model.stop).abs() > 1e-12
        || history
            .probe_values
            .iter()
            .any(|v| v.len() != history.times_s.len() || v.iter().any(|x| !x.is_finite()))
    {
        return Err("incomplete finite original native cylinder history".into());
    }
    history.starter_sha256 = format!(
        "{:x}",
        Sha256::digest(std::fs::read(&decks.starter).map_err(|e| e.to_string())?)
    );
    history.engine_deck_sha256 = format!(
        "{:x}",
        Sha256::digest(std::fs::read(&decks.engine).map_err(|e| e.to_string())?)
    );
    Ok(history)
}

/// Measure the complete free-body hollow rail between both physical platens.
///
/// Requires byte-identical checked TYPE7 decks, uses objective eight-point
/// B-bar LAW36 integration, and applies the actual full-boundary native contact
/// operator. The explicit numerical penalty scale requires independent
/// convergence validation. Histories are measurements, not QA admission.
///
/// # Errors
/// Rejects invalid options, modified decks, unsupported contact behavior,
/// mismatched geometry, incompatible material, failed/incomplete integration,
/// erosion, nonfinite measurements or excessive force-work ledger drift.
pub fn rail_plastic_history(
    directory: &Path,
    case: &sonium_qa::rad::RadExtendedCase,
    surfaces: &[sonium_qa::rad::contact_surface::RadSurfSegments],
    extra: &sonium_qa::rad::RadExtra,
    timestep_scale: f64,
    penalty_scale: f64,
    profile_time_s: Option<f64>,
) -> Result<NativePlasticityHistory, String> {
    use sha2::{Digest, Sha256};
    use sonium_qa::rad::contact_surface::{
        native_solid_self_contact, write_explicit_solid_contact_starter,
    };
    if !timestep_scale.is_finite()
        || timestep_scale <= 0.0
        || timestep_scale > 1.0
        || !penalty_scale.is_finite()
        || penalty_scale < 1.0
        || profile_time_s.is_some_and(|t| !t.is_finite() || t <= 0.0 || t > case.base.stop_time_s)
        || case.base.name != "rail_crush_full_platens"
        || case.base.stop_time_s != 1.2
        || !case.base.bcs.is_empty()
    {
        return Err("full free-body rail requires original duration, valid CFL/penalty scale and profile time".into());
    }
    let starter =
        write_explicit_solid_contact_starter(case, surfaces, extra).map_err(|e| e.to_string())?;
    let engine = sonium_qa::rad::write_extended_engine(case);
    let decks = C3Decks {
        starter: directory.join(format!("{}_0000.rad", case.base.name)),
        engine: directory.join(format!("{}_0001.rad", case.base.name)),
        probes: directory.join("probes.json"),
    };
    if std::fs::read(&decks.starter).map_err(|e| e.to_string())? != starter.as_bytes()
        || std::fs::read(&decks.engine).map_err(|e| e.to_string())? != engine.as_bytes()
    {
        return Err("native rail requires byte-identical validated physical decks".into());
    }
    // Admit contact cards only after proving their complete typed native
    // counterpart; the generic benchmark parser remains strict by default.
    native_solid_self_contact(case, surfaces, extra, 1.0).map_err(|e| e.to_string())?;
    let model = parse_native_model_with_validated_solid_contact(&decks, true)?;
    let NativeMaterial::Law36 { material } = &model.material else {
        return Err("full hollow rail requires the declared LAW36 hardening curve".into());
    };
    if model.stop != case.base.stop_time_s
        || model.walls.len() != 2
        || model
            .walls
            .iter()
            .any(|wall| wall.slaves.len() != model.coords.len())
    {
        return Err("both physical platens must cover every structural rail node".into());
    }
    // Use coordinates read from the exact serialized reference deck. This
    // avoids giving the native arm unrounded, pre-serialization coordinates.
    let mut serialized_case = case.clone();
    for (&id, &index) in &model.index_of {
        serialized_case.base.mesh.nodes[id as usize - 1] = model.coords[index];
    }
    let mesh = HexMesh::new(model.coords.clone(), model.bricks.clone());
    let min_char = (0..model.bricks.len())
        .map(|e| min_edge_length(&mesh, e))
        .fold(f64::INFINITY, f64::min);
    let native = native_solid_self_contact(
        &serialized_case,
        surfaces,
        extra,
        model.youngs * min_char * penalty_scale,
    )
    .map_err(|e| e.to_string())?;
    if native.coordinates != model.coords
        || native.bricks != model.bricks
        || native
            .node_ids
            .iter()
            .enumerate()
            .any(|(i, &id)| model.index_of.get(&(id as i64)) != Some(&i))
    {
        return Err("contact operator and bulk solve must share exact compact geometry".into());
    }
    let fast_contact = match std::env::var("SONIUM_NATIVE_CONTACT_FAST").as_deref() {
        Ok("1") => true,
        Ok("0") | Err(std::env::VarError::NotPresent) => false,
        _ => return Err("SONIUM_NATIVE_CONTACT_FAST must be0 or1".into()),
    };
    let mut operator = native.operator;
    operator.set_parallel_projections(fast_contact);
    operator.set_projection_cache_enabled(fast_contact);
    let contact = RefCell::new(operator);
    let probes = (1..=2)
        .map(|wall| Probe {
            label: format!("physical platen{wall} axial force"),
            channel: format!("FNX/RWALL/{wall}"),
            expected: 0.0,
            abs_tol: 0.0,
            rel_tol: 0.0,
        })
        .collect::<Vec<_>>();
    let initial = model
        .bricks
        .iter()
        .map(|_| HexExplicitState::new(Law36State::new()))
        .collect();
    let mut history = NativePlasticityHistory::default();
    repeat_plastic_core(
        &case.base.name,
        &model,
        &probes,
        material,
        initial,
        ErosionCriterion {
            max_equiv_plastic_strain: f64::INFINITY,
            max_damage: 1.0,
            max_porosity: f64::INFINITY,
            max_tabular_damage: f64::INFINITY,
        },
        NativeRunOptions {
            timestep_scale,
            penalty_scale,
            objective: true,
            gauss_bbar: true,
            parallel_gauss: native_gauss_parallel()?,
            self_contact: Some(&contact),
            profile_sample_time_s: profile_time_s,
            capture: Some(&mut history),
        },
    )?;
    history.self_contact_execution = Some(if fast_contact {
        format!(
            "rayon:{};exact-coordinate-cache;ordered-force-scatter",
            rayon::current_num_threads()
        )
    } else {
        "serial;uncached".into()
    });
    history.times_s.insert(0, 0.0);
    for values in &mut history.probe_values {
        values.insert(0, 0.0);
    }
    if history.steps == 0
        || (history.final_time_s - model.stop).abs() > 1e-12
        || history.eroded_elements != 0
        || history.node_ids.len() != model.coords.len()
        || history.final_positions_m.len() != model.coords.len()
        || history.final_element_plastic_strain.len() != model.bricks.len()
        || history.probe_values.len() != 2
        || history.times_s.len() < 2
        || history
            .probe_values
            .iter()
            .any(|v| v.len() != history.times_s.len())
        || history
            .times_s
            .iter()
            .chain(history.probe_values.iter().flatten())
            .chain(history.final_positions_m.iter().flatten())
            .chain(&history.final_element_plastic_strain)
            .chain(history.profile_sample_positions_m.iter().flatten())
            .chain(&history.profile_sample_element_plastic_strain)
            .any(|v| !v.is_finite())
        || !history.relative_energy_drift.is_finite()
        || history.relative_energy_drift > DRIFT_TOL
        || history
            .final_self_contact_penalty_energy_j
            .is_none_or(|v| !v.is_finite() || v < 0.0)
        || profile_time_s.is_some_and(|t| {
            history.profile_sample_time_s != Some(t)
                || history.profile_sample_positions_m.len() != model.coords.len()
                || history.profile_sample_element_plastic_strain.len() != model.bricks.len()
        })
    {
        return Err("complete finite full-stroke native rail observations required".into());
    }
    history.starter_sha256 = format!("{:x}", Sha256::digest(starter.as_bytes()));
    history.engine_deck_sha256 = format!("{:x}", Sha256::digest(engine.as_bytes()));
    Ok(history)
}

/// Measure full plastic tensile deformation from the explicit necking decks.
///
/// Uses the actual native LAW36 material and objective stress integration.
/// Returns finite observations without admitting placeholder benchmark probes.
/// A requested profile is interpolated only between actual adjacent steps;
/// the complete physical trajectory and stop time remain unchanged.
///
/// # Errors
/// Rejects invalid run options, malformed decks, incompatible material,
/// failed integration, nonfinite observations and incomplete physical profiles.
pub fn necking_plastic_history(
    directory: &Path,
    timestep_scale: f64,
    profile_time_s: Option<f64>,
) -> Result<NativePlasticityHistory, String> {
    use sha2::{Digest, Sha256};
    let gauss_bbar = match std::env::var("P2_NECKING_FORMULATION").as_deref() {
        Ok("gauss-bbar") => true,
        Ok("centroid") | Err(_) => false,
        Ok(_) => return Err("P2_NECKING_FORMULATION must be centroid or gauss-bbar".into()),
    };
    if !timestep_scale.is_finite()
        || timestep_scale <= 0.0
        || timestep_scale > 1.0
        || profile_time_s.is_some_and(|t| !t.is_finite() || t <= 0.0)
    {
        return Err("necking requires CFL scale in(0,1] and finite positive profile time".into());
    }
    let decks = C3Decks {
        starter: directory.join("necking_bar_full_plastic_0000.rad"),
        engine: directory.join("necking_bar_full_plastic_0001.rad"),
        probes: directory.join("probes.json"),
    };
    let model = parse_native_model(&decks)?;
    if profile_time_s.is_some_and(|t| t > model.stop) {
        return Err("necking profile exceeds complete physical duration".into());
    }
    let probes = read_probe_file(&decks.probes)?;
    for probe in &probes {
        resolve_probe_channel(&probe.channel, &model)?;
    }
    let NativeMaterial::Law36 { material } = &model.material else {
        return Err("full necking requires the explicit LAW36 hardening curve".into());
    };
    let initial = model
        .bricks
        .iter()
        .map(|_| HexExplicitState::new(Law36State::new()))
        .collect();
    let mut history = NativePlasticityHistory::default();
    repeat_plastic_core(
        "necking_bar_full_plastic",
        &model,
        &probes,
        material,
        initial,
        ErosionCriterion {
            max_equiv_plastic_strain: f64::INFINITY,
            max_damage: 1.0,
            max_porosity: f64::INFINITY,
            max_tabular_damage: f64::INFINITY,
        },
        NativeRunOptions {
            timestep_scale,
            objective: true,
            gauss_bbar,
            parallel_gauss: gauss_bbar && native_gauss_parallel()?,
            profile_sample_time_s: profile_time_s,
            capture: Some(&mut history),
            ..NativeRunOptions::default()
        },
    )?;
    history.times_s.insert(0, 0.0);
    for values in &mut history.probe_values {
        values.insert(0, 0.0);
    }
    if history.steps == 0
        || (history.final_time_s - model.stop).abs() > 1e-12
        || history.node_ids.len() != model.coords.len()
        || history.final_positions_m.len() != model.coords.len()
        || history.final_element_plastic_strain.len() != model.bricks.len()
        || history.probe_values.len() != probes.len()
        || history
            .probe_values
            .iter()
            .any(|v| v.len() != history.times_s.len())
        || history
            .times_s
            .iter()
            .chain(history.probe_values.iter().flatten())
            .chain(history.final_positions_m.iter().flatten())
            .chain(&history.final_element_plastic_strain)
            .any(|v| !v.is_finite())
        || !history.relative_energy_drift.is_finite()
    {
        return Err("incomplete finite native full plastic tensile history".into());
    }
    if let Some(time) = profile_time_s
        && (history.profile_sample_time_s != Some(time)
            || history.profile_sample_positions_m.len() != model.coords.len()
            || history.profile_sample_element_plastic_strain.len() != model.bricks.len()
            || history
                .profile_sample_positions_m
                .iter()
                .flatten()
                .chain(&history.profile_sample_element_plastic_strain)
                .any(|v| !v.is_finite()))
    {
        return Err("complete finite necking profile at reference time required".into());
    }
    history.starter_sha256 = format!(
        "{:x}",
        Sha256::digest(std::fs::read(&decks.starter).map_err(|e| e.to_string())?)
    );
    history.engine_deck_sha256 = format!(
        "{:x}",
        Sha256::digest(std::fs::read(&decks.engine).map_err(|e| e.to_string())?)
    );
    Ok(history)
}

/// Measure a complete copper Taylor impact from an explicit deck directory.
///
/// The directory must contain `taylor_bar_full_plastic_0000.rad`, its engine
/// deck, and `probes.json`. Uses the actual native LAW2 integration path and
/// exports observations; benchmark probe verdicts do not establish QA parity.
/// `penalty_scale >= 1` refines contact stiffness while tightening the contact
/// timestep bound consistently. Other benchmark cases keep their default.
/// `objective` enables the spin-based corotational stress/history update.
///
/// # Errors
/// Rejects invalid CFL scale, nonplastic material, unresolved probe channels,
/// integration failures, nonfinite results, and incomplete histories.
pub fn taylor_plastic_history(
    directory: &Path,
    timestep_scale: f64,
    penalty_scale: f64,
    objective: bool,
) -> Result<NativePlasticityHistory, String> {
    taylor_plastic_history_sampled(directory, timestep_scale, penalty_scale, objective, None)
}

/// Measure full Taylor impact with a profile at an exact reference timestamp.
///
/// Interpolates only observations between actual adjacent native steps. It does
/// not change the trajectory, constitutive evolution, full physical stop time,
/// or benchmark behavior. Coordinates and plastic strains share one time.
///
/// # Errors
/// Rejects invalid sample time, invalid run options, nonplastic material,
/// incomplete integration and nonfinite profiles.
pub fn taylor_plastic_history_sampled(
    directory: &Path,
    timestep_scale: f64,
    penalty_scale: f64,
    objective: bool,
    profile_sample_time_s: Option<f64>,
) -> Result<NativePlasticityHistory, String> {
    if profile_sample_time_s.is_some_and(|t| !t.is_finite() || t <= 0.0) {
        return Err("native profile sample time must be finite positive".into());
    }
    use sha2::{Digest, Sha256};
    if !timestep_scale.is_finite() || timestep_scale <= 0.0 || timestep_scale > 1.0 {
        return Err("native CFL timestep scale must be in (0,1]".into());
    }
    if !penalty_scale.is_finite() || penalty_scale < 1.0 {
        return Err("native contact penalty scale must be finite and >=1".into());
    }
    let decks = C3Decks {
        starter: directory.join("taylor_bar_full_plastic_0000.rad"),
        engine: directory.join("taylor_bar_full_plastic_0001.rad"),
        probes: directory.join("probes.json"),
    };
    let model = parse_native_model(&decks)?;
    if profile_sample_time_s.is_some_and(|t| t > model.stop) {
        return Err("native profile sample time exceeds complete physical duration".into());
    }
    let probes = read_probe_file(&decks.probes)?;
    for probe in &probes {
        resolve_probe_channel(&probe.channel, &model)?;
    }
    let NativeMaterial::Law2 {
        material,
        t0,
        fail_ep,
    } = &model.material
    else {
        return Err("Taylor impact requires a LAW2 plastic material".into());
    };
    let initial = model
        .bricks
        .iter()
        .map(|_| HexExplicitState::new(Law2State::new(*t0)))
        .collect();
    let mut history = NativePlasticityHistory::default();
    repeat_plastic_core(
        "taylor_bar_full_plastic",
        &model,
        &probes,
        material,
        initial,
        ErosionCriterion {
            max_equiv_plastic_strain: *fail_ep,
            max_damage: 1.0,
            max_porosity: f64::INFINITY,
            max_tabular_damage: f64::INFINITY,
        },
        NativeRunOptions {
            timestep_scale,
            penalty_scale,
            objective,
            gauss_bbar: false,
            parallel_gauss: false,
            self_contact: None,
            profile_sample_time_s,
            capture: Some(&mut history),
        },
    )?;
    history.times_s.insert(0, 0.0);
    for values in &mut history.probe_values {
        values.insert(0, 0.0);
    }
    if history.steps == 0
        || history.times_s.len() < 2
        || history.probe_values.len() != probes.len()
        || (history.final_time_s - model.stop).abs() > 1e-12
        || history
            .probe_values
            .iter()
            .any(|v| v.len() != history.times_s.len() || v.iter().any(|x| !x.is_finite()))
        || history.final_positions_m.len() != model.coords.len()
        || history
            .final_positions_m
            .iter()
            .flatten()
            .any(|x| !x.is_finite())
        || history
            .final_element_plastic_strain
            .iter()
            .any(|x| !x.is_finite())
        || !history.relative_energy_drift.is_finite()
    {
        return Err("incomplete finite native Taylor impact history".into());
    }
    if let Some(time) = profile_sample_time_s
        && (history.profile_sample_time_s != Some(time)
            || history.profile_sample_positions_m.len() != model.coords.len()
            || history.profile_sample_element_plastic_strain.len() != model.bricks.len()
            || history
                .profile_sample_positions_m
                .iter()
                .flatten()
                .chain(&history.profile_sample_element_plastic_strain)
                .any(|x| !x.is_finite()))
    {
        return Err("complete finite matching-time native profile required".into());
    }
    history.starter_sha256 = format!(
        "{:x}",
        Sha256::digest(std::fs::read(&decks.starter).map_err(|e| e.to_string())?)
    );
    history.engine_deck_sha256 = format!(
        "{:x}",
        Sha256::digest(std::fs::read(&decks.engine).map_err(|e| e.to_string())?)
    );
    Ok(history)
}

// Interpolate actual adjacent observations; reject ragged or extrapolated data.
fn interpolate_plastic_profile(
    time: f64,
    before_time: f64,
    before_positions: &[f64],
    before_plastic: &[f64],
    after_time: f64,
    after_positions: &[f64],
    after_plastic: &[f64],
) -> Result<(Vec<[f64; 3]>, Vec<f64>), String> {
    if !time.is_finite()
        || !before_time.is_finite()
        || !after_time.is_finite()
        || after_time <= before_time
        || time < before_time
        || time > after_time
        || before_positions.len() != after_positions.len()
        || before_positions.is_empty()
        || !before_positions.len().is_multiple_of(3)
        || before_plastic.len() != after_plastic.len()
        || before_plastic.is_empty()
        || before_positions
            .iter()
            .chain(after_positions)
            .chain(before_plastic)
            .chain(after_plastic)
            .any(|v| !v.is_finite())
    {
        return Err("aligned finite bracketing physical profile required; no extrapolation".into());
    }
    let weight = (time - before_time) / (after_time - before_time);
    let positions = before_positions
        .as_chunks::<3>()
        .0
        .iter()
        .zip(after_positions.as_chunks::<3>().0)
        .map(|(a, b)| std::array::from_fn(|j| a[j] * (1.0 - weight) + b[j] * weight))
        .collect();
    let plastic = before_plastic
        .iter()
        .zip(after_plastic)
        .map(|(a, b)| a * (1.0 - weight) + b * weight)
        .collect();
    Ok((positions, plastic))
}

/// Generic plastic repeat: fixed-step CFL loop over
/// [`hex_plastic_forces`] with an erosion sweep, mirroring the
/// validated Taylor-plastic leg driver (fixed stepping threads the
/// true step rate into the flow stress; per-step edge lengths keep
/// the CFL bound honest under large deformation).
fn repeat_plastic_core<M: ObjectivePlasticMaterial + Sync>(
    deck: &str,
    model: &NativeModel,
    probes: &[Probe],
    material: &M,
    initial: Vec<HexExplicitState<M::State>>,
    criterion: ErosionCriterion,
    options: NativeRunOptions<'_>,
) -> Result<RepeatSample, String>
where
    M::State: Send,
{
    let capture_profile = options.capture.is_some()
        && matches!(
            deck,
            "taylor_bar_full_plastic"
                | "necking_bar_full_plastic"
                | "rail_crush"
                | "rail_crush_full_platens"
                | "beam_warping"
        );
    let multiplier_walls = match std::env::var("SONIUM_NATIVE_PLASTIC_MULTIPLIER_WALLS") {
        Ok(value) if value == "1" => true,
        Ok(value) if value == "0" => false,
        Err(std::env::VarError::NotPresent) => false,
        _ => return Err("SONIUM_NATIVE_PLASTIC_MULTIPLIER_WALLS must be0or1".into()),
    };
    if multiplier_walls
        && (!capture_profile
            || !matches!(deck, "necking_bar_full_plastic" | "rail_crush_full_platens"))
    {
        return Err(
            "plastic multiplier walls require an explicit full necking or rail diagnostic".into(),
        );
    }
    // Full one-second tension needs millions of physical contact-resolution
    // steps,including the independent half-CFL study and mesh refinements.
    // Only this explicit diagnostic gets more runtime; mass,CFL and duration
    // are unchanged,and every other benchmark keeps its original step guard.
    let step_budget = if capture_profile
        && matches!(
            deck,
            "necking_bar_full_plastic" | "rail_crush_full_platens" | "beam_warping"
        ) {
        8 * MAX_NATIVE_STEPS
    } else {
        MAX_NATIVE_STEPS
    };
    let mesh = HexMesh::new(model.coords.clone(), model.bricks.clone());
    let num_nodes = mesh.num_nodes();
    let num_elems = mesh.conn.len();
    if num_nodes == 0 || num_elems == 0 {
        return Err("native model holds no mesh".to_string());
    }
    let centroid_parallel = !options.gauss_bbar && native_centroid_parallel(num_elems)?;
    let centroid_threads = if centroid_parallel {
        rayon::current_num_threads().min(8)
    } else {
        1
    };
    let centroid_pool = if centroid_parallel && rayon::current_num_threads() > centroid_threads {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(centroid_threads)
                .build()
                .map_err(|error| format!("centroid material pool: {error}"))?,
        )
    } else {
        None
    };
    let mut masses = vec![0.0; num_nodes];
    let mut char_lengths = Vec::with_capacity(num_elems);
    for elem in 0..num_elems {
        let volume = element_volume(&mesh, elem);
        if !volume.is_finite() || volume <= 0.0 {
            return Err(format!(
                "element {elem} has non-positive volume {volume:.6e}"
            ));
        }
        for node in mesh.conn[elem] {
            masses[node] += model.density * volume / 8.0;
        }
        char_lengths.push(min_edge_length(&mesh, elem));
    }
    let min_mass = masses.iter().cloned().fold(f64::INFINITY, f64::min);
    if !min_mass.is_finite() || min_mass <= 0.0 {
        return Err(format!("non-positive lumped mass {min_mass:.6e}"));
    }
    let min_char = char_lengths.iter().cloned().fold(f64::INFINITY, f64::min);
    let elastic = NeoHookean::from_young(model.youngs, model.poisson, 3);
    let dilatational = ((elastic.lambda + 2.0 * elastic.mu) / model.density).sqrt();
    let sound_speeds = vec![dilatational; num_elems];
    let dt_crit = critical_dt(&char_lengths, &sound_speeds);
    let dt_max = if multiplier_walls {
        0.9 * dt_crit
    } else if model.walls.is_empty() {
        dt_crit
    } else {
        // Contact RESOLUTION cap (leg precedent), not just stability:
        // dt <= 2*sqrt(m/k) is stable but chatters (force zero most
        // steps, rare spikes, low plateau mean); omega dt = 0.2 holds
        // persistent contact and the pilot mean matches the engine.
        let penalty = suggested_penalty(
            model.youngs,
            min_char,
            PLASTIC_PENALTY_FACTOR * options.penalty_scale,
        );
        0.1 * 2.0 * (min_mass / penalty).sqrt()
    };
    let dt_max = dt_max * options.timestep_scale;
    if !dt_max.is_finite() || dt_max <= 0.0 {
        return Err("native dt bound is not positive".to_string());
    }
    let estimated = (model.stop / dt_max).ceil() as u64;
    if estimated > step_budget {
        return Err(format!(
            "native run needs about {estimated} steps (stop {}s, dt_max {dt_max:.3e}s),exceeding diagnostic step budget {step_budget}",
            model.stop
        ));
    }
    println!(
        "sonium mesh: {num_nodes} nodes, {num_elems} bricks, dt_max {dt_max:.3e}s, about {estimated} steps"
    );
    match &model.material {
        NativeMaterial::Law1 => {}
        NativeMaterial::Law2 { t0, fail_ep, .. } => {
            println!("sonium material: LAW2 T0 {t0:.1}K, fail ep {fail_ep:.3e}");
        }
        NativeMaterial::Law36 { material } => {
            let curves: Vec<String> = material
                .curves
                .iter()
                .map(|curve| format!("rate {:.3e} scale {:.3e}", curve.rate, curve.scale))
                .collect();
            println!("sonium material: LAW36 [{}]", curves.join(", "));
        }
    }
    let mut params = HexExplicitParams::new(model.density, dilatational);
    // Retain the existing zero linear coefficient for these benchmark paths.
    // Full diagnostics may select the reference coefficients explicitly below.
    // Compression pressure now has the dissipative sign in every explicit
    // solid kernel; earlier instability attributed to this coefficient was
    // measured before that correction. Export the actual coefficients.
    params.bulk_c1 = 0.0;
    let mut viscosity_overridden = false;
    for (name, coefficient) in [
        ("SONIUM_NATIVE_PLASTIC_BULK_C0", &mut params.bulk_c0),
        ("SONIUM_NATIVE_PLASTIC_BULK_C1", &mut params.bulk_c1),
    ] {
        let value = match std::env::var(name) {
            Ok(value) => value,
            Err(std::env::VarError::NotPresent) => continue,
            Err(error) => return Err(format!("invalid {name}: {error}")),
        };
        if !capture_profile
            || !matches!(
                deck,
                "necking_bar_full_plastic" | "rail_crush_full_platens" | "beam_warping"
            )
        {
            return Err(
                "bulk-viscosity override requires an explicit full solid diagnostic".into(),
            );
        }
        let value: f64 = value
            .parse()
            .map_err(|error| format!("invalid {name}: {error}"))?;
        if !value.is_finite() || value < 0.0 {
            return Err(format!("{name} requires a finite nonnegative coefficient"));
        }
        *coefficient = value;
        viscosity_overridden = true;
    }
    if viscosity_overridden {
        println!(
            "explicit plastic bulk viscosity: quadratic {}, linear {}; full physical QA unproven",
            params.bulk_c0, params.bulk_c1
        );
    }
    let gauss_states = RefCell::new(if options.gauss_bbar {
        initial
            .iter()
            .map(|s| HexGaussPlasticState::new(s.centroid.clone()))
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    });
    let states = RefCell::new(initial);
    let material_error = RefCell::new(None::<String>);
    let center_ep = |elem: usize| {
        if options.gauss_bbar {
            gauss_states.borrow()[elem].center_features(material).0
        } else {
            material.erosion_features(&states.borrow()[elem].centroid).0
        }
    };
    let center_stress = |elem: usize| -> [f64; 6] {
        if options.gauss_bbar {
            let points = gauss_states.borrow();
            let mut mean = [0.0; 6];
            for point in &points[elem].points {
                for (value, stress) in mean.iter_mut().zip(M::centroid_stress(point)) {
                    *value += stress / 8.0;
                }
            }
            mean
        } else {
            M::centroid_stress(&states.borrow()[elem].centroid)
        }
    };
    let plastic_values = || (0..num_elems).map(&center_ep).collect::<Vec<_>>();
    let volume_plastic_values = |positions: &[f64]| -> Result<Vec<f64>, String> {
        let states = gauss_states.borrow();
        if states.len() != num_elems || positions.len() != 3 * num_nodes {
            return Err("volume-average histories and complete mesh must align".into());
        }
        mesh.conn
            .iter()
            .zip(states.iter())
            .map(|(conn, state)| {
                let coordinates = std::array::from_fn(|slot| {
                    std::array::from_fn(|axis| positions[3 * conn[slot] + axis])
                });
                let geometry =
                    HexGaussPlasticGeom::new(coordinates).map_err(|error| error.to_string())?;
                let value = state.volume_features(material, &geometry).0;
                if !value.is_finite() {
                    return Err("nonfinite volume-average plastic observation".into());
                }
                Ok(value)
            })
            .collect()
    };
    let erosion = RefCell::new(ErosionSet::new(num_elems));
    let step_dt = RefCell::new(dt_max);
    let max_ep = RefCell::new(0.0f64);
    let mut driver = ExplicitStepDriver::new(
        AdaptiveDtConfig::new(0.9, 1.0e-12, dt_max, 1.1)
            .map_err(|error| format!("adaptive dt config rejected: {error}"))?,
    )
    .map_err(|error| format!("step driver rejected: {error}"))?;
    let mut state = ExplicitState::new(num_nodes);
    for (node, xyz) in model.coords.iter().enumerate() {
        state.positions[3 * node] = xyz[0];
        state.positions[3 * node + 1] = xyz[1];
        state.positions[3 * node + 2] = xyz[2];
    }
    state.masses.copy_from_slice(&masses);
    let reference: Vec<f64> = state.positions.clone();
    let penalty = suggested_penalty(
        model.youngs,
        min_char,
        PLASTIC_PENALTY_FACTOR * options.penalty_scale,
    );
    let stride = ((estimated / MAX_FORCE_SAMPLES as u64).max(1)) as usize;
    // Dashpot energy dissipated over the run (stabilization footprint;
    // wall-force probes read the penalty spring alone, leg precedent).
    let damp_dissipated = RefCell::new(0.0f64);
    let self_contact_peak_pairs = RefCell::new(0usize);
    let wall_bulk_cache =
        multiplier_walls.then(|| RefCell::new(InternalForceCache::new(3 * num_nodes)));
    let engaged = RefCell::new(
        model
            .walls
            .iter()
            .map(|wall| vec![false; wall.slaves.len()])
            .collect::<Vec<_>>(),
    );
    let wall_transitions = RefCell::new(Vec::<(usize, usize)>::new());
    let multiplier_reactions = RefCell::new(vec![[0.0; 3]; model.walls.len()]);
    let mut adoption_kinetic = 0.0;
    let mut adoption_work = 0.0;
    let mut adoption_dissipation = 0.0;
    let mut adoption_impulses = vec![[0.0; 3]; model.walls.len()];

    let internal_force = |positions: &[f64], velocities: &[f64], time: f64| {
        let dt = *step_dt.borrow();
        let mut forces = vec![0.0; positions.len()];
        let mut states = states.borrow_mut();
        let mut erosion = erosion.borrow_mut();
        let mut features = Vec::with_capacity(num_elems);
        let mut bulk_responses = if options.gauss_bbar || centroid_parallel {
            let active = (0..num_elems)
                .map(|e| erosion.is_active(e))
                .collect::<Vec<_>>();
            let step = GaussBulkStep {
                positions,
                velocities,
                dt,
                parallel: if options.gauss_bbar {
                    options.parallel_gauss
                } else {
                    true
                },
            };
            let response = if options.gauss_bbar {
                gauss_bulk_responses(
                    &mesh,
                    &mut gauss_states.borrow_mut(),
                    material,
                    &step,
                    &params,
                    &active,
                )
            } else {
                let states = states.as_mut_slice();
                let mut update = || {
                    centroid_bulk_responses(
                        &mesh,
                        states,
                        material,
                        &step,
                        &params,
                        &active,
                        options.objective,
                    )
                };
                if let Some(pool) = &centroid_pool {
                    pool.install(update)
                } else {
                    update()
                }
            };
            match response {
                Ok(responses) => responses,
                Err(error) => {
                    *material_error.borrow_mut() = Some(error.to_string());
                    return vec![f64::NAN; positions.len()];
                }
            }
        } else {
            Vec::new()
        };
        for (elem, conn) in mesh.conn.iter().enumerate() {
            if !erosion.is_active(elem) {
                features.push((0.0, 0.0, 0.0));
                continue;
            }
            let mut elem_coords = [[0.0; 3]; 8];
            for (slot, node) in conn.iter().enumerate() {
                elem_coords[slot] = [
                    positions[3 * node],
                    positions[3 * node + 1],
                    positions[3 * node + 2],
                ];
            }
            let mut vel = [0.0; 24];
            for (slot, node) in conn.iter().enumerate() {
                vel[slot * 3] = velocities[3 * node];
                vel[slot * 3 + 1] = velocities[3 * node + 1];
                vel[slot * 3 + 2] = velocities[3 * node + 2];
            }
            let response = if options.gauss_bbar || centroid_parallel {
                let response = bulk_responses[elem]
                    .take()
                    .expect("active element has bulk response");
                if let Err(error) = &response {
                    *material_error.borrow_mut() = Some(format!("element{elem}: {error}"));
                    return vec![f64::NAN; positions.len()];
                }
                response
            } else {
                let geom = match HexExplicitGeom::new(elem_coords) {
                    Ok(geom) => geom,
                    Err(_) => {
                        features.push((f64::INFINITY, 0.0, 0.0));
                        continue;
                    }
                };
                if options.objective {
                    hex_plastic_forces_objective(
                        &geom,
                        &mut states[elem],
                        material,
                        &vel,
                        dt,
                        &params,
                    )
                } else {
                    hex_plastic_forces(&geom, &mut states[elem], material, &vel, dt, &params)
                }
            };
            match response {
                Ok(out) => {
                    for (slot, node) in conn.iter().enumerate() {
                        forces[3 * node] -= out.base.forces[slot * 3];
                        forces[3 * node + 1] -= out.base.forces[slot * 3 + 1];
                        forces[3 * node + 2] -= out.base.forces[slot * 3 + 2];
                    }
                    let mut peak = max_ep.borrow_mut();
                    *peak = peak.max(out.features.0);
                    features.push(out.features);
                }
                Err(_) => {
                    features.push((f64::INFINITY, 0.0, 0.0));
                }
            }
        }
        erosion.sweep(&criterion, &features);
        if let Some(contact) = options.self_contact {
            let pairs = contact
                .borrow_mut()
                .apply_with_pairs(positions, velocities, &mut forces);
            let mut peak = self_contact_peak_pairs.borrow_mut();
            *peak = (*peak).max(pairs.len());
        }
        for (node, axes) in model.pinned.iter().enumerate() {
            for axis in 0..3 {
                if axes[axis] {
                    forces[3 * node + axis] = 0.0;
                }
            }
        }
        if let Some(cache) = &wall_bulk_cache {
            cache
                .borrow_mut()
                .store(positions, velocities, time, &forces);
        }
        forces
    };
    let external_force = |positions: &[f64], velocities: &[f64], time: f64| {
        let mut forces = vec![0.0; positions.len()];
        for wall in model.walls.iter().filter(|_| !multiplier_walls) {
            let point = [
                wall.point[0] + wall.velocity[0] * time,
                wall.point[1] + wall.velocity[1] * time,
                wall.point[2] + wall.velocity[2] * time,
            ];
            let mut rigid = RigidWall::new(point, wall.normal, penalty);
            rigid.velocity = wall.velocity;
            let pairs =
                rigid_wall_contact(&rigid, &wall.slaves, positions, velocities, &mut forces);
            // Overdamped dashpot on each penetrating slave (leg
            // precedent): settles the bounce mode; its reaction joins
            // the dynamics but stays out of the spring-force probe.
            let dt = *step_dt.borrow();
            let mut dissipated = damp_dissipated.borrow_mut();
            for pair in &pairs {
                let node = pair.slave;
                let c = 2.0 * PLASTIC_CONTACT_DAMPING * (penalty * masses[node]).sqrt();
                let vn_rel = (velocities[3 * node] - wall.velocity[0]) * wall.normal[0]
                    + (velocities[3 * node + 1] - wall.velocity[1]) * wall.normal[1]
                    + (velocities[3 * node + 2] - wall.velocity[2]) * wall.normal[2];
                let damp = -c * vn_rel;
                forces[3 * node] += damp * wall.normal[0];
                forces[3 * node + 1] += damp * wall.normal[1];
                forces[3 * node + 2] += damp * wall.normal[2];
                *dissipated += c * vn_rel * vn_rel * dt;
            }
        }
        for cload in &model.cloads {
            let shape = model
                .functs
                .get(&cload.funct)
                .and_then(|points| eval_funct(points, time).ok());
            let Some(shape) = shape else { continue };
            for node in &cload.nodes {
                forces[3 * node + cload.axis] += cload.scale * shape;
            }
        }
        if let Some(cache) = &wall_bulk_cache {
            let cache = cache.borrow();
            if !cache.matches(positions, velocities, time) {
                // Re-evaluation would advance plastic histories twice. Reject
                // a changed driver call order instead of silently recomputing.
                *material_error.borrow_mut() = Some("plastic wall bulk cache mismatch".into());
                return vec![f64::NAN; positions.len()];
            }
            let mut running = forces.clone();
            let mut flags = engaged.borrow_mut();
            let mut transitions = wall_transitions.borrow_mut();
            let mut reactions = multiplier_reactions.borrow_mut();
            for (index, wall) in model.walls.iter().enumerate() {
                let point =
                    std::array::from_fn(|axis| wall.point[axis] + wall.velocity[axis] * time);
                let mut rigid = RigidWall::new(point, wall.normal, 0.0);
                rigid.velocity = wall.velocity;
                let mut fresh = Vec::new();
                reactions[index] = rigid_wall_multiplier(
                    &rigid,
                    [0.0; 3],
                    &wall.slaves,
                    &masses,
                    positions,
                    velocities,
                    &cache.forces,
                    &running,
                    &mut flags[index],
                    &mut fresh,
                    &mut forces,
                );
                transitions.extend(fresh.into_iter().map(|slot| (index, slot)));
                running.copy_from_slice(&forces);
            }
        }
        for (node, axes) in model.pinned.iter().enumerate() {
            for axis in 0..3 {
                if axes[axis] {
                    forces[3 * node + axis] = 0.0;
                }
            }
        }
        forces
    };

    // The thick-cylinder engine leg samples its displacement probes at
    // transient times (50 MPa crossing, outer transit) and its stress
    // probes as trailing means, so the native arm records per-step
    // histories for those targets. Explicit observation requests also record
    // traces; other benchmark decks keep their original final-state sampling.
    let record_histories = deck == "plasticity_thick_cylinder" || options.capture.is_some();
    let targets: Vec<Option<ProbeTarget>> = if record_histories {
        probes
            .iter()
            .map(|probe| resolve_probe_channel(&probe.channel, model).ok())
            .collect()
    } else {
        Vec::new()
    };
    let mut probe_times: Vec<f64> = Vec::new();
    let mut probe_values: Vec<Vec<f64>> = vec![Vec::new(); targets.len()];
    let mut sampled_profile: Option<(f64, Vec<[f64; 3]>, Vec<f64>)> = None;
    let mut volume_average_profile = Vec::new();
    // Reuse the observation buffer; it never enters the physical force closure.
    let mut reaction_scratch = vec![0.0; state.positions.len()];
    let start = std::time::Instant::now();
    let mut steps: u64 = 0;
    let mut dt_last = dt_max;
    let mut sampled: Vec<Vec<(f64, [f64; 3])>> = vec![Vec::new(); model.walls.len()];
    let progress_stride: u64 = std::env::var("SONIUM_NATIVE_PROGRESS_STEPS")
        .ok()
        .map(|value| value.parse())
        .transpose()
        .map_err(|e| format!("invalid native progress interval: {e}"))?
        .unwrap_or(100_000);
    if progress_stride == 0 {
        return Err("SONIUM_NATIVE_PROGRESS_STEPS must be positive".into());
    }
    while state.time < model.stop {
        if steps >= step_budget {
            return Err(format!(
                "stalled: {step_budget} steps reached at t = {:.6e}s, short of stop {:.6e}s",
                state.time, model.stop
            ));
        }
        if (steps as usize).is_multiple_of(stride) {
            let mut scratch = vec![0.0; state.positions.len()];
            for (index, wall) in model.walls.iter().enumerate() {
                if multiplier_walls {
                    sampled[index].push((state.time, multiplier_reactions.borrow()[index]));
                    continue;
                }
                let point = [
                    wall.point[0] + wall.velocity[0] * state.time,
                    wall.point[1] + wall.velocity[1] * state.time,
                    wall.point[2] + wall.velocity[2] * state.time,
                ];
                let mut rigid = RigidWall::new(point, wall.normal, penalty);
                rigid.velocity = wall.velocity;
                let pairs = rigid_wall_contact(
                    &rigid,
                    &wall.slaves,
                    &state.positions,
                    &state.velocities,
                    &mut scratch,
                );
                let mut on_wall = [0.0; 3];
                for pair in &pairs {
                    on_wall[0] -= pair.force[0];
                    on_wall[1] -= pair.force[1];
                    on_wall[2] -= pair.force[2];
                }
                sampled[index].push((state.time, on_wall));
            }
        }
        let char_now: Vec<f64> = mesh
            .conn
            .iter()
            .map(|conn| min_edge_length_at(&state.positions, conn))
            .collect();
        let dt_crit = critical_dt(&char_now, &sound_speeds);
        let mut dt = (0.9 * dt_crit).min(dt_max).min(1.1 * dt_last);
        if let Some(contact) = options.self_contact {
            let bound = contact
                .borrow()
                .normal_stiffness_mass_bound(&state.positions, &masses)
                .map_err(|error| format!("contact timestep bound rejected: {error}"))?;
            if bound > 0.0 {
                // Resolve nearby normal contact before penetration. Frozen
                // projection stiffness supplements, rather than replaces,
                // the physical bulk and rigid-wall timestep bounds.
                dt = dt.min(0.2 / bound.sqrt());
            }
        }
        dt = dt.min(model.stop - state.time).max(1.0e-12);
        if !dt.is_finite() || dt <= 0.0 {
            return Err(format!(
                "non-positive step {dt:.6e} at t = {:.6e}s",
                state.time
            ));
        }
        // Copy one bracketing state only when the requested output is crossed.
        // Observation never changes dt or any constitutive/integrator state.
        let before_profile = options
            .profile_sample_time_s
            .filter(|time| {
                sampled_profile.is_none() && state.time <= *time && *time <= state.time + dt
            })
            .map(|time| (time, state.time, state.positions.clone(), plastic_values()));
        let before_volume_profile = if before_profile.is_some() && options.gauss_bbar {
            Some(volume_plastic_values(&state.positions)?)
        } else {
            None
        };
        *step_dt.borrow_mut() = dt;
        wall_transitions.borrow_mut().clear();
        if let Err(error) = driver.step_fixed(&mut state, internal_force, external_force, dt) {
            return Err(format!(
                "unstable at t = {:.6e}s after {steps} steps: {error};material error {:?}",
                state.time,
                material_error.borrow().as_ref()
            ));
        }
        if multiplier_walls {
            let flags = engaged.borrow();
            for &(index, slot) in wall_transitions.borrow().iter() {
                if flags[index][slot] {
                    let wall = &model.walls[index];
                    let (kinetic, work, dissipation, impulse) =
                        adopt_multiplier_wall_velocity(&mut state, wall, wall.slaves[slot]);
                    adoption_kinetic += kinetic;
                    adoption_work += work;
                    adoption_dissipation += dissipation;
                    for (sum, value) in adoption_impulses[index].iter_mut().zip(impulse) {
                        *sum += value;
                    }
                }
            }
        }
        if let Some((time, before_time, before_positions, before_plastic)) = before_profile {
            let after_plastic = plastic_values();
            let (positions, plastic) = interpolate_plastic_profile(
                time,
                before_time,
                &before_positions,
                &before_plastic,
                state.time,
                &state.positions,
                &after_plastic,
            )?;
            sampled_profile = Some((time, positions, plastic));
            if let Some(before) = before_volume_profile {
                let after = volume_plastic_values(&state.positions)?;
                let weight = (time - before_time) / (state.time - before_time);
                volume_average_profile = before
                    .iter()
                    .zip(after)
                    .map(|(a, b)| a * (1.0 - weight) + b * weight)
                    .collect();
            }
        }
        dt_last = dt;
        steps += 1;
        if capture_profile && steps.is_multiple_of(progress_stride) {
            println!(
                "Native {deck} progress: physical_time={:.9e}s/{:.9e}s,steps={steps},wall={:.3}s,max_ep={:.6e},self_contact_peak_pairs={}",
                state.time,
                model.stop,
                start.elapsed().as_secs_f64(),
                *max_ep.borrow(),
                *self_contact_peak_pairs.borrow(),
            );
        }
        if record_histories {
            probe_times.push(state.time);
            for (slot, target) in targets.iter().copied().enumerate() {
                let value = match target {
                    Some(ProbeTarget::Displacement { axis, node }) => {
                        state.positions[3 * node + axis] - reference[3 * node + axis]
                    }
                    Some(ProbeTarget::ElemStress { comp, elem }) => center_stress(elem)[comp],
                    Some(ProbeTarget::ElemPlas { elem }) => center_ep(elem),
                    Some(ProbeTarget::WallForce { axis, wall }) if multiplier_walls => {
                        multiplier_reactions.borrow()[wall][axis]
                    }
                    Some(ProbeTarget::WallForce { axis, wall }) => wall_reaction(
                        &model.walls[wall],
                        state.time,
                        penalty,
                        &state.positions,
                        &state.velocities,
                        &mut reaction_scratch,
                    )[axis],
                    _ => f64::NAN,
                };
                probe_values[slot].push(value);
            }
        }
    }
    let wall_s = start.elapsed().as_secs_f64();
    println!(
        "sonium run: reached t = {:.6e}s in {steps} steps",
        state.time
    );

    // Engine-leg sampling mirror for the thick cylinder: probe order is
    // positional there (bore DX at the 50 MPa crossing, outer DX one
    // transit later, hoop residuals as trailing means), so the native
    // arm applies the same schedule to its recorded histories, except
    // the residuals average the full zero hold: the native run is
    // undamped (c1 = 0 kills the only damper), so its ringdown never
    // decays and a 50-sample window keeps a phase-dependent residue,
    // while the hold mean cancels the breathing mode over ~19 periods.
    // Any unrecordable target falls back to final-state sampling below.
    // Hold start transcribed from the leg deck (raised cosine unloads
    // to zero at twice the load time, then holds to stop).
    const HOLD_START_S: f64 = 0.002;
    let sample_times = (deck == "plasticity_thick_cylinder").then(thick_cylinder_sample_times);
    let mut failures = Vec::new();
    for (index, probe) in probes.iter().enumerate() {
        let target = match resolve_probe_channel(&probe.channel, model) {
            Ok(target) => target,
            Err(error) => {
                failures.push(error);
                continue;
            }
        };
        let mut actual = match target {
            ProbeTarget::Displacement { axis, node } => {
                state.positions[3 * node + axis] - reference[3 * node + axis]
            }
            ProbeTarget::WallForce { axis, wall } => {
                let history = &sampled[wall];
                if history.is_empty() {
                    failures.push(format!("probe {} has no wall-force samples", probe.label));
                    continue;
                }
                let wall = &model.walls[wall];
                let mut gap = f64::INFINITY;
                for slave in &wall.slaves {
                    let position = model.coords[*slave];
                    let distance = (position[0] - wall.point[0]) * wall.normal[0]
                        + (position[1] - wall.point[1]) * wall.normal[1]
                        + (position[2] - wall.point[2]) * wall.normal[2];
                    gap = gap.min(distance);
                }
                if gap <= TOUCHING_GAP_M {
                    history[history.len() - 1].1[axis]
                } else {
                    let half = history.len() / 2;
                    let window = &history[half..];
                    window.iter().map(|(_, force)| force[axis]).sum::<f64>() / window.len() as f64
                }
            }
            ProbeTarget::ElemStress { comp, elem } => center_stress(elem)[comp],
            ProbeTarget::ElemPlas { elem } => center_ep(elem),
        };
        if let (Some((t_elastic, t_outer)), Some(values)) = (sample_times, probe_values.get(index))
        {
            let sampled = match index {
                0 => history_lerp(&probe_times, values, t_elastic),
                1 => history_lerp(&probe_times, values, t_outer),
                _ => history_holdmean(&probe_times, values, HOLD_START_S),
            };
            if let Some(value) = sampled.filter(|value| value.is_finite()) {
                actual = value;
            }
        }
        if capture_profile {
            // Full Taylor diagnostic targets are placeholders; independent
            // reference comparisons consume the actual exported measurements.
            println!(
                "measured {} [{}]: actual {actual:.6e}",
                probe.label, probe.channel
            );
            continue;
        }
        println!(
            "probe {} [{}]: actual {actual:.6e}, expected {:.6e}",
            probe.label, probe.channel, probe.expected
        );
        if let Err(error) = assert_probe_close(
            actual,
            probe.expected,
            probe.abs_tol,
            probe.rel_tol,
            &probe.label,
        ) {
            failures.push(error.to_string());
        }
    }
    for failure in &failures {
        println!("FAIL: {failure}");
    }
    let balance = driver.balance().clone();
    let final_kinetic = if multiplier_walls {
        state
            .velocities
            .as_chunks::<3>()
            .0
            .iter()
            .zip(&state.masses)
            .map(|(v, mass)| 0.5 * mass * v.iter().map(|x| x * x).sum::<f64>())
            .sum()
    } else {
        balance.kinetic
    };
    let corrected_drift = if multiplier_walls {
        balance.drift + (final_kinetic - balance.kinetic) - adoption_kinetic
    } else {
        balance.drift
    };
    let energy_scale = final_kinetic + balance.internal.abs();
    let drift_frac = if energy_scale > 0.0 {
        corrected_drift.abs() / energy_scale
    } else {
        f64::INFINITY
    };
    println!(
        "energy: KE {:.6e}J, IE {:.6e}J, Wext {:.6e}J, drift {:.3e} rel",
        final_kinetic,
        balance.internal,
        balance.external_work + adoption_work,
        drift_frac
    );
    if multiplier_walls {
        println!(
            "wall adoption: kinetic change {adoption_kinetic:.6e}J, moving-wall work {adoption_work:.6e}J, inelastic loss {adoption_dissipation:.6e}J"
        );
    }
    if !drift_frac.is_finite() || drift_frac > DRIFT_TOL {
        failures.push(format!(
            "energy drift {drift_frac:.3e} exceeds {DRIFT_TOL:.3e}"
        ));
    }
    let eroded = num_elems - erosion.borrow().active_count();
    println!(
        "plastic: max ep {:.6e}, eroded {eroded}/{num_elems}",
        *max_ep.borrow(),
    );
    println!(
        "contact: dashpot dissipated {:.6e} J over the run",
        *damp_dissipated.borrow(),
    );
    let dof = u64::try_from(num_nodes)
        .ok()
        .and_then(|nodes| nodes.checked_mul(3))
        .ok_or_else(|| "node count overflows dof".to_string())?;
    if capture_profile {
        println!("native history: wall {wall_s:.3}s, cycles {steps}, dof {dof}");
    } else {
        println!(
            "sonium: wall {wall_s:.3}s, cycles {steps}, dof {dof}, correct {}",
            failures.is_empty()
        );
    }
    let (profile_sample_time_s, profile_sample_positions_m, profile_sample_element_plastic_strain) =
        match sampled_profile {
            Some((time, positions, plastic)) => (Some(time), positions, plastic),
            None => (None, Vec::new(), Vec::new()),
        };
    if let Some(capture) = options.capture {
        let final_contact_energy = options.self_contact.map(|contact| {
            let mut scratch = vec![0.0; state.positions.len()];
            let mut contact = contact.borrow_mut();
            let stiffness = contact.config().penalty;
            contact
                .apply_with_pairs(&state.positions, &state.velocities, &mut scratch)
                .iter()
                .map(|pair| 0.5 * stiffness * pair.gap * pair.gap)
                .sum()
        });
        *capture = NativePlasticityHistory {
            artificial_bulk_pressure_convention: Some(
                "tensile-positive-Cauchy;compressive-artificial-pressure-minus-q-I".into(),
            ),
            bulk_viscosity_coefficients: Some([params.bulk_c0, params.bulk_c1]),
            rigid_wall_formulation: multiplier_walls
                .then(|| "frictionless-multiplier;normal-velocity-adoption".into()),
            wall_adoption_kinetic_change_j: multiplier_walls.then_some(adoption_kinetic),
            wall_adoption_work_j: multiplier_walls.then_some(adoption_work),
            wall_adoption_dissipation_j: multiplier_walls.then_some(adoption_dissipation),
            wall_adoption_impulses_kg_m_s: if multiplier_walls {
                adoption_impulses
            } else {
                Vec::new()
            },
            integration_formulation: options.gauss_bbar.then(|| "gauss-bbar".into()),
            bulk_execution: ((options.gauss_bbar && options.parallel_gauss) || centroid_parallel)
                .then(|| {
                    format!(
                        "rayon:{};ordered-force-scatter",
                        if centroid_parallel {
                            centroid_threads
                        } else {
                            rayon::current_num_threads()
                        }
                    )
                }),
            self_contact_execution: None,
            self_contact_peak_pairs: options
                .self_contact
                .map(|_| *self_contact_peak_pairs.borrow()),
            final_self_contact_penalty_energy_j: final_contact_energy,
            probe_channels: probes.iter().map(|p| p.channel.clone()).collect(),
            times_s: probe_times,
            probe_values,
            element_ids: {
                let mut ids = vec![0; num_elems];
                for (&id, &index) in &model.elem_index_of {
                    ids[index] = id;
                }
                ids
            },
            final_element_plastic_strain: plastic_values(),
            final_time_s: state.time,
            steps,
            kinetic_energy_j: final_kinetic,
            internal_energy_j: balance.internal,
            external_work_j: balance.external_work + adoption_work,
            relative_energy_drift: drift_frac,
            eroded_elements: eroded,
            node_ids: if capture_profile {
                let mut ids = vec![0; num_nodes];
                for (&id, &index) in &model.index_of {
                    ids[index] = id;
                }
                ids
            } else {
                Vec::new()
            },
            final_positions_m: if capture_profile {
                (0..num_nodes)
                    .map(|node| {
                        [
                            state.positions[3 * node],
                            state.positions[3 * node + 1],
                            state.positions[3 * node + 2],
                        ]
                    })
                    .collect()
            } else {
                Vec::new()
            },
            profile_sample_time_s,
            profile_sample_positions_m,
            profile_sample_element_plastic_strain,
            profile_sample_element_volume_average_plastic_strain: volume_average_profile,
            ..NativePlasticityHistory::default()
        };
    }
    Ok(RepeatSample {
        wall_s,
        cycles: steps,
        dof,
        peak_rss_bytes: self_peak_rss_bytes(),
        correct: failures.is_empty(),
    })
}

/// Resolved deck family: C3 cases live under the crate `decks/`
/// dir; anything else resolves as a miniqa case id.
#[derive(Debug)]
enum DeckFamily {
    C3(C3Decks),
    Miniqa(PathBuf),
}

/// Resolves a deck id to its family, failing closed on unknown ids.
fn resolve_deck_family(deck: &str, miniqa_root: Option<PathBuf>) -> Result<DeckFamily, String> {
    let c3_error = match resolve_c3_decks(deck) {
        Ok(c3) => return Ok(DeckFamily::C3(c3)),
        Err(error) => error,
    };
    // The C3 miss reason travels into both errors: new deck ids
    // usually fail here first, and the missing artifact (or the
    // unknown id) is the actionable part.
    let Some(root) = miniqa_root else {
        return Err(format!(
            "deck {deck} is not a C3 case ({c3_error}) and no --miniqa/{MINIQA_ENV} resolves it as miniqa"
        ));
    };
    find_deck(&root, deck).map_err(|error| {
        format!("deck {deck} resolves neither as C3 ({c3_error}) nor miniqa: {error}")
    })?;
    Ok(DeckFamily::Miniqa(root))
}

/// Builds the engine-arm context, resolving exes and launch mode.
fn engine_context(args: &Args) -> Result<EngineCtx, String> {
    let exe = args
        .exe
        .clone()
        .or_else(engine_exe)
        .ok_or_else(|| "engine arm needs --exe, RADISS_EXE, or OPENCOURANT_EXE".to_string())?;
    let (starter_exe, mut engine_exe) =
        resolve_engine_pair(&exe).map_err(|error| error.to_string())?;
    let provisional = resolve_launch(args, &engine_exe)?;
    if provisional.flavor == FlavorOpt::Mpi
        && !engine_exe.to_string_lossy().to_lowercase().contains("ompi")
        && !engine_exe.to_string_lossy().to_lowercase().contains("_mpi")
    {
        engine_exe = select_mpi_engine(&exe)?;
    }
    let launch = resolve_launch(args, &engine_exe)?;
    let exec_dir = if exe.is_dir() {
        exe.clone()
    } else {
        exe.parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| exe.clone())
    };
    Ok(EngineCtx {
        starter_exe,
        engine_exe,
        exec_dir,
        launch,
        scratch_root: resolve_scratch_root(args.scratch_root.clone()),
        keep_scratch: args.keep_scratch,
    })
}

/// Runs the engine arm over warmup plus timed repeats.
fn run_engine_arm(args: &Args, deck: &str, repeats: u32, warmup: u32) -> Result<BenchRow, String> {
    let ctx = engine_context(args)?;
    let miniqa = args.miniqa.clone().or_else(|| {
        std::env::var(MINIQA_ENV)
            .ok()
            .filter(|root| !root.trim().is_empty())
            .map(PathBuf::from)
    });
    let goldens_root = args
        .goldens
        .clone()
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("goldens"));
    let family = resolve_deck_family(deck, miniqa)?;
    let c3_probes = match &family {
        DeckFamily::C3(decks) => Some(read_probe_file(&decks.probes)?),
        DeckFamily::Miniqa(_) => None,
    };
    let one = |index: usize| match (&family, &c3_probes) {
        (DeckFamily::C3(decks), Some(probes)) => engine_c3_repeat(&ctx, deck, decks, probes, index),
        (DeckFamily::Miniqa(root), _) => {
            engine_miniqa_repeat(&ctx, root, &goldens_root, deck, index)
        }
        (DeckFamily::C3(_), None) => Err("C3 probes unreadable".to_string()),
    };
    for warmup_index in 0..warmup {
        println!("engine {deck}: warmup {}/{}", warmup_index + 1, warmup);
        one(warmup_index as usize)?;
    }
    let mut samples = Vec::new();
    for repeat in 0..repeats {
        println!("engine {deck}: repeat {}/{}", repeat + 1, repeats);
        samples.push(one((warmup + repeat) as usize)?);
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

/// Runs the sonium arm over warmup plus timed repeats (C3 only).
fn run_sonium_arm(args: &Args, deck: &str, repeats: u32, warmup: u32) -> Result<BenchRow, String> {
    if resolve_c3_decks(deck).is_err() && deck.contains('/') {
        return Err(format!(
            "sonium arm supports C3 decks only; miniqa case {deck} deferred (no native model for vendor decks)"
        ));
    }
    // The native explicit path is serial today; the setting is
    // recorded on the row so the 6B matrix stays explicit.
    // Safety: no other thread reads the environment concurrently at
    // this point (single-threaded startup before any spawn).
    unsafe {
        std::env::set_var("RAYON_NUM_THREADS", args.threads.max(1).to_string());
    }
    let decks = resolve_c3_decks(deck)?;
    let probes = read_probe_file(&decks.probes)?;
    let model = parse_native_model(&decks)?;
    for warmup_index in 0..warmup {
        println!("sonium {deck}: warmup {}/{}", warmup_index + 1, warmup);
        let _ = sonium_repeat(deck, &model, &probes)?;
    }
    let mut samples = Vec::new();
    for repeat in 0..repeats {
        println!("sonium {deck}: repeat {}/{}", repeat + 1, repeats);
        samples.push(sonium_repeat(deck, &model, &probes)?);
    }
    let total = aggregate_repeats(&samples)?;
    BenchRow::measured(
        deck,
        Side::Sonium,
        "native",
        u32::try_from(args.threads.max(1)).map_err(|_| "thread count overflows u32".to_string())?,
        total.wall_s,
        total.cycles,
        total.dof,
        total.peak_rss_bytes,
        total.correct,
    )
    .map_err(|error| error.to_string())
}

/// First CPU model string from `/proc/cpuinfo`, else `unknown`.
fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines().find_map(|line| {
                line.strip_prefix("model name")
                    .and_then(|rest| rest.split(':').nth(1))
                    .map(|model| model.trim().to_string())
                    .filter(|model| !model.is_empty())
            })
        })
        .unwrap_or_else(|| "unknown".to_string())
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
    let (run_sonium, run_engine) = resolve_sides(&args)?;
    let (repeats, warmup) = resolve_repeats(&args)?;
    if args.threads == 0 {
        return Err("--threads must be positive".to_string());
    }
    let mut rows = Vec::new();
    if run_sonium {
        rows.push(run_sonium_arm(&args, &deck, repeats, warmup)?);
    }
    if run_engine {
        rows.push(run_engine_arm(&args, &deck, repeats, warmup)?);
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
    let report = render_bundle_markdown(&bundle).map_err(|error| error.to_string())?;
    println!("{report}");
    if bundle.rows.iter().all(|row| row.correctness_passed) {
        Ok(())
    } else {
        Err("one or more rows failed their correctness probes".to_string())
    }
}

fn main() {
    match run() {
        Ok(()) => println!("SUCCESS: qa-oc-bench"),
        Err(error) => {
            eprintln!("error: {error}");
            println!("FAILED: qa-oc-bench");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn centroid_bulk_parallel_preserves_evolving_plastic_states_and_ordered_forces() {
        use std::time::{Duration, Instant};
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        let mesh = HexMesh::brick(10, 8, 7, [0.0; 3], [0.2, 0.05, 0.05]);
        let curve =
            RateCurve::new(0.0, vec![0.0, 0.1, 1.0], vec![250e6, 350e6, 550e6], 1.0).unwrap();
        let material = Law36::new(200e9, 0.3, vec![curve], 0.0, 1e30, 1e30, 2e30, 1e30);
        let params = HexExplicitParams::new(7800.0, 5000.0);
        for objective in [false, true] {
            let mut serial = vec![HexExplicitState::new(Law36State::new()); mesh.conn.len()];
            let mut parallel = serial.clone();
            let mut active = vec![true; mesh.conn.len()];
            active[17] = false;
            let mut positions = mesh.coords.iter().flatten().copied().collect::<Vec<_>>();
            let mut serial_time = Duration::ZERO;
            let mut parallel_time = Duration::ZERO;
            for _ in 0..80 {
                let velocities = positions
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .flat_map(|p| {
                        [
                            0.05 * p[0] - 0.3 * p[1],
                            0.3 * p[0] - 0.015 * p[1],
                            -0.015 * p[2],
                        ]
                    })
                    .collect::<Vec<_>>();
                for (p, v) in positions.iter_mut().zip(&velocities) {
                    *p += 1e-3 * v;
                }
                let mut step = GaussBulkStep {
                    positions: &positions,
                    velocities: &velocities,
                    dt: 1e-3,
                    parallel: false,
                };
                let start = Instant::now();
                let a = centroid_bulk_responses(
                    &mesh,
                    &mut serial,
                    &material,
                    &step,
                    &params,
                    &active,
                    objective,
                )
                .unwrap();
                serial_time += start.elapsed();
                step.parallel = true;
                let start = Instant::now();
                let b = pool
                    .install(|| {
                        centroid_bulk_responses(
                            &mesh,
                            &mut parallel,
                            &material,
                            &step,
                            &params,
                            &active,
                            objective,
                        )
                    })
                    .unwrap();
                parallel_time += start.elapsed();
                let mut f_serial = vec![0.0; positions.len()];
                let mut f_parallel = f_serial.clone();
                for (elem, (a, b)) in a.into_iter().zip(b).enumerate() {
                    match (a, b) {
                        (None, None) => assert!(!active[elem]),
                        (Some(Ok(a)), Some(Ok(b))) => {
                            assert!(a.base.forces.iter().all(|force| force.is_finite()));
                            assert!(a.features.0.is_finite());
                            assert_eq!(format!("{a:?}"), format!("{b:?}"));
                            for (slot, node) in mesh.conn[elem].iter().enumerate() {
                                for axis in 0..3 {
                                    f_serial[3 * node + axis] -= a.base.forces[3 * slot + axis];
                                    f_parallel[3 * node + axis] -= b.base.forces[3 * slot + axis];
                                }
                            }
                        }
                        _ => panic!("active centroid response must agree and be valid"),
                    }
                    assert_eq!(
                        format!("{:?}", serial[elem]),
                        format!("{:?}", parallel[elem])
                    );
                }
                for (a, b) in f_serial.iter().zip(&f_parallel) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
            }
            assert!(serial.iter().any(|s| s.centroid.equiv_plastic_strain > 0.0));
            assert_eq!(serial[17].centroid.equiv_plastic_strain, 0.0);
            println!(
                "Centroid objective={objective}:serial={serial_time:?},parallel8={parallel_time:?};bitwise ordered forces and full material debug state agree"
            );
        }
    }

    #[test]
    fn gauss_bulk_parallel_preserves_evolving_plastic_histories_and_force_order() {
        use std::time::{Duration, Instant};
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        let mesh = HexMesh::brick(10, 8, 7, [0.0; 3], [0.2, 0.05, 0.05]);
        let curve =
            RateCurve::new(0.0, vec![0.0, 0.1, 1.0], vec![250e6, 350e6, 550e6], 1.0).unwrap();
        let material = Law36::new(200e9, 0.3, vec![curve], 0.0, 1e30, 1e30, 2e30, 1e30);
        let params = HexExplicitParams::new(7800.0, 5000.0);
        let mut serial = vec![HexGaussPlasticState::new(Law36State::new()); mesh.conn.len()];
        let mut parallel = serial.clone();
        let mut active = vec![true; mesh.conn.len()];
        active[17] = false;
        let mut positions = mesh.coords.iter().flatten().copied().collect::<Vec<_>>();
        let mut serial_time = Duration::ZERO;
        let mut parallel_time = Duration::ZERO;
        for _ in 0..80 {
            let velocities = positions
                .chunks_exact(3)
                .flat_map(|p| {
                    // Uniaxial extension with Poisson contraction and rigid spin.
                    [
                        0.05 * p[0] - 0.3 * p[1],
                        0.3 * p[0] - 0.015 * p[1],
                        -0.015 * p[2],
                    ]
                })
                .collect::<Vec<_>>();
            for (p, v) in positions.iter_mut().zip(&velocities) {
                *p += 1e-3 * v;
            }
            let mut step = GaussBulkStep {
                positions: &positions,
                velocities: &velocities,
                dt: 1e-3,
                parallel: false,
            };
            let start = Instant::now();
            let a = gauss_bulk_responses(&mesh, &mut serial, &material, &step, &params, &active)
                .unwrap();
            serial_time += start.elapsed();
            step.parallel = true;
            let start = Instant::now();
            let b = pool
                .install(|| {
                    gauss_bulk_responses(&mesh, &mut parallel, &material, &step, &params, &active)
                })
                .unwrap();
            parallel_time += start.elapsed();
            for (i, (a, b)) in a.into_iter().zip(b).enumerate() {
                if !active[i] {
                    assert!(a.is_none() && b.is_none());
                    continue;
                }
                let (a, b) = (a.unwrap().unwrap(), b.unwrap().unwrap());
                for (x, y) in a.base.forces.iter().zip(b.base.forces) {
                    assert_eq!(x.to_bits(), y.to_bits());
                }
                assert_eq!(a.features, b.features);
            }
            for (a, b) in serial.iter().zip(&parallel) {
                for (a, b) in a.points.iter().zip(&b.points) {
                    for (x, y) in a
                        .stress
                        .iter()
                        .chain(&a.back_stress)
                        .chain(&a.plastic)
                        .chain([&a.equiv_plastic_strain, &a.tensile_scale])
                        .zip(
                            b.stress
                                .iter()
                                .chain(&b.back_stress)
                                .chain(&b.plastic)
                                .chain([&b.equiv_plastic_strain, &b.tensile_scale]),
                        )
                    {
                        assert_eq!(x.to_bits(), y.to_bits());
                    }
                    assert_eq!(
                        (a.ruptured, a.deleted, a.yielding),
                        (b.ruptured, b.deleted, b.yielding)
                    );
                }
            }
        }
        assert!(
            serial
                .iter()
                .any(|e| e.points.iter().any(|s| s.equiv_plastic_strain > 0.0))
        );
        assert!(
            serial[17]
                .points
                .iter()
                .all(|s| s.equiv_plastic_strain == 0.0 && s.stress == [0.0; 6])
        );
        println!(
            "Gauss bulk 560-element/80-step timing: serial_ms={}, parallel8_ms={}, speedup={}",
            serial_time.as_secs_f64() * 1e3,
            parallel_time.as_secs_f64() * 1e3,
            serial_time.as_secs_f64() / parallel_time.as_secs_f64()
        );
    }

    #[test]
    fn rail_law1_adapter_has_original_uniaxial_stress_without_yield() {
        use sonium_fem::physics::solid::plasticity::StressUpdateModel;
        let material = super::rail_law1_adapter(200e9, 0.3).unwrap();
        let out = material.update(
            &super::Law36State::new(),
            &[-0.01, 0.003, 0.003, 0.0, 0.0, 0.0],
        );
        assert_eq!(out.delta_equiv_plastic_strain, 0.0);
        assert!((out.stress[0] + 2e9).abs() < 1e-6);
        assert!(out.stress[1..].iter().all(|v| v.abs() < 1e-6));
    }
    use super::*;
    use sonium_fem::physics::solid::nonlinear_solid::q1_element_residual;

    #[test]
    fn multiplier_velocity_adoption_accounts_impulse_work_and_inelastic_loss() {
        for (normal, wall_velocity, initial_velocity) in [
            ([1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, -4.0]),
            ([0.6, 0.8, 0.0], [0.0; 3], [-3.0, -4.0, 2.0]),
        ] {
            let wall = NativeWall {
                number: 1,
                point: [0.0; 3],
                normal,
                velocity: wall_velocity,
                slaves: vec![0],
            };
            let mut state = ExplicitState::new(2);
            state.masses.copy_from_slice(&[3.0, 1.0]);
            state.velocities[..3].copy_from_slice(&initial_velocity);
            state.velocities[3..].copy_from_slice(&[1.0, 2.0, 3.0]);
            let before = state.velocities.clone();
            let (kinetic, work, loss, impulse) =
                adopt_multiplier_wall_velocity(&mut state, &wall, 0);
            assert!(loss > 0.0);
            assert!((work - kinetic - loss).abs() < 1e-12);
            let relative_normal = (0..3)
                .map(|axis| (state.velocities[axis] - wall_velocity[axis]) * normal[axis])
                .sum::<f64>();
            assert!(relative_normal.abs() < 1e-12);
            for axis in 0..3 {
                assert!(
                    (3.0 * (state.velocities[axis] - before[axis]) - impulse[axis]).abs() < 1e-12
                );
            }
            assert_eq!(state.velocities[3..], before[3..]);
            let after = state.velocities.clone();
            assert_eq!(
                adopt_multiplier_wall_velocity(&mut state, &wall, 0),
                (0.0, 0.0, 0.0, [0.0; 3])
            );
            assert_eq!(state.velocities, after);
        }
    }

    #[test]
    fn explicit_driver_reuses_each_bulk_evaluation_for_wall_constraints() {
        let mut state = ExplicitState::new(1);
        state.masses[0] = 1.0;
        let cache = RefCell::new(InternalForceCache::new(3));
        let calls = std::cell::Cell::new(0usize);
        let internal = |positions: &[f64], velocities: &[f64], time: f64| {
            calls.set(calls.get() + 1);
            let force = vec![-2.0, 0.0, 0.0];
            cache
                .borrow_mut()
                .store(positions, velocities, time, &force);
            force
        };
        let external = |positions: &[f64], velocities: &[f64], time: f64| {
            let cache = cache.borrow();
            assert!(cache.matches(positions, velocities, time));
            cache.forces.iter().map(|force| -force).collect::<Vec<_>>()
        };
        let mut driver =
            ExplicitStepDriver::new(AdaptiveDtConfig::new(0.9, 1e-12, 0.01, 1.1).unwrap()).unwrap();
        for _ in 0..4 {
            driver
                .step_fixed(&mut state, internal, external, 0.01)
                .unwrap();
        }
        assert_eq!(calls.get(), 5);
        assert_eq!(state.positions, vec![0.0; 3]);
        assert!(
            !cache
                .borrow()
                .matches(&state.positions, &[-0.0, 0.0, 0.0], state.time)
        );
        assert_eq!(calls.get(), 5);
    }

    #[test]
    fn wall_history_observation_tracks_moving_plane_and_action_reaction() {
        let wall = NativeWall {
            number: 1,
            point: [0.0; 3],
            normal: [1.0, 0.0, 0.0],
            velocity: [0.25, 0.0, 0.0],
            slaves: vec![0],
        };
        let positions = [0.1, 0.0, 0.0];
        let velocities = [0.0; 3];
        let mut scratch = [99.0; 3];
        assert_eq!(
            wall_reaction(&wall, 0.0, 1000.0, &positions, &velocities, &mut scratch),
            [0.0; 3]
        );
        let reaction = wall_reaction(&wall, 0.8, 1000.0, &positions, &velocities, &mut scratch);
        assert!((reaction[0] + 100.0).abs() < 1e-12);
        for axis in 0..3 {
            assert_eq!(reaction[axis], -scratch[axis]);
        }
        assert_eq!(positions, [0.1, 0.0, 0.0]);
        assert_eq!(velocities, [0.0; 3]);
    }

    fn args_with(side: SideOpt, sonium_only: bool, engine_only: bool) -> Args {
        Args {
            deck: None,
            emit_bundle: PathBuf::from("bundle.json"),
            side,
            threads: 1,
            repeats: None,
            quick: false,
            sonium_only,
            engine_only,
            flavor: None,
            mpi_ranks: None,
            exe: None,
            miniqa: None,
            goldens: None,
            scratch_root: None,
            keep_scratch: false,
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn physical_profile_interpolation_uses_one_time_and_rejects_bad_evidence() {
        let a = [0.0, 1.0, 2.0];
        let b = [4.0, 5.0, 6.0];
        let (p, ep) = interpolate_plastic_profile(3.0, 2.0, &a, &[0.1], 6.0, &b, &[0.5]).unwrap();
        assert_eq!(p, vec![[1.0, 2.0, 3.0]]);
        assert!((ep[0] - 0.2).abs() < 1e-15);
        for time in [1.0, 7.0, f64::NAN, f64::INFINITY] {
            assert!(interpolate_plastic_profile(time, 2.0, &a, &[0.1], 6.0, &b, &[0.5]).is_err());
        }
        assert!(interpolate_plastic_profile(3.0, 2.0, &a, &[0.1], 6.0, &b[..2], &[0.5]).is_err());
        assert!(interpolate_plastic_profile(3.0, 2.0, &a, &[0.1], 6.0, &b, &[f64::NAN]).is_err());
    }

    #[test]
    fn vmhwm_parse_reads_peak_kib() {
        let status = "Name:\tqa-oc-bench\nVmRSS:\t    3000 kB\nVmHWM:\t    80896 kB\nThreads:\t1\n";
        assert_eq!(parse_vmhwm_bytes(status), Some(80896 * 1024));
        assert_eq!(parse_vmhwm_bytes("VmRSS:\t    3000 kB\n"), None);
        assert_eq!(parse_vmhwm_bytes("VmHWM:\tgarbage kB\n"), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn self_peak_rss_is_own_footprint() {
        // The test binary's own high-water mark is far below the
        // hundreds of MB a launcher-inherited value would show.
        let bytes = self_peak_rss_bytes();
        assert!(bytes > 0, "peak RSS must be measurable");
        assert!(
            bytes < 512 * 1024 * 1024,
            "peak RSS {bytes} looks inherited, not own footprint"
        );
    }

    #[test]
    fn sides_default_to_both() {
        assert_eq!(
            resolve_sides(&args_with(SideOpt::Both, false, false)).unwrap(),
            (true, true)
        );
    }

    #[test]
    fn only_flags_select_single_arms() {
        assert_eq!(
            resolve_sides(&args_with(SideOpt::Both, true, false)).unwrap(),
            (true, false)
        );
        assert_eq!(
            resolve_sides(&args_with(SideOpt::Both, false, true)).unwrap(),
            (false, true)
        );
    }

    #[test]
    fn contradictory_sides_fail() {
        assert!(resolve_sides(&args_with(SideOpt::Both, true, true)).is_err());
        assert!(resolve_sides(&args_with(SideOpt::Engine, true, false)).is_err());
        assert!(resolve_sides(&args_with(SideOpt::Sonium, false, true)).is_err());
    }

    #[test]
    fn repeats_default_with_warmup() {
        let args = args_with(SideOpt::Both, false, false);
        assert_eq!(resolve_repeats(&args).unwrap(), (3, 1));
    }

    #[test]
    fn quick_forces_single_repeat() {
        let mut args = args_with(SideOpt::Both, false, false);
        args.quick = true;
        assert_eq!(resolve_repeats(&args).unwrap(), (1, 0));
        args.repeats = Some(3);
        assert!(resolve_repeats(&args).is_err());
    }

    #[test]
    fn funct_interpolates_and_clamps() {
        let points = [[0.0, 0.0], [1.0, 2.0]];
        assert_eq!(eval_funct(&points, -1.0).unwrap(), 0.0);
        assert_eq!(eval_funct(&points, 0.5).unwrap(), 1.0);
        assert_eq!(eval_funct(&points, 2.0).unwrap(), 2.0);
        assert!(eval_funct(&[], 0.0).is_err());
    }

    #[test]
    fn rootname_parses_run_line() {
        let text = "#RADIOSS ENGINE\n/RUN/taylor_bar/1\n   4.000000000000E-6\n";
        assert_eq!(run_rootname(text).unwrap(), "taylor_bar");
        assert!(run_rootname("no run here\n").is_err());
    }

    #[test]
    fn si_units_detected() {
        assert!(has_si_units("/BEGIN\n  kg   m   s\n"));
        assert!(!has_si_units("/BEGIN\n  g   mm   ms\n"));
    }

    #[test]
    fn deck_family_prefers_c3() {
        let family = resolve_deck_family("taylor_bar", None).unwrap();
        assert!(matches!(family, DeckFamily::C3(_)));
        let error = resolve_deck_family("no_such_deck", None).unwrap_err();
        assert!(error.contains("not a C3 case"));
    }

    #[test]
    fn every_deck_dir_resolves_or_fails_closed_with_reason() {
        // N-deck harness audit: parsing only, no timing. Each
        // checked-in deck either builds a native model or fails with
        // a reason naming the deck or the missing/unsupported
        // artifact; units adding arms only flip entries from the
        // second set to the first, never this shape.
        let decks_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("decks");
        let mut entries: Vec<String> = std::fs::read_dir(&decks_root)
            .expect("decks dir readable")
            .flatten()
            .filter(|entry| entry.path().is_dir())
            .filter_map(|entry| entry.file_name().to_str().map(str::to_string))
            .collect();
        entries.sort();
        assert!(!entries.is_empty(), "decks dir holds no cases");
        for deck in &entries {
            match resolve_c3_decks(deck) {
                Err(error) => {
                    assert!(!error.trim().is_empty(), "{deck}: empty C3 reason");
                    assert!(
                        error.contains(deck),
                        "{deck}: C3 reason names neither deck nor artifact: {error}"
                    );
                }
                Ok(c3) => {
                    read_probe_file(&c3.probes)
                        .unwrap_or_else(|error| panic!("{deck}: probes unreadable: {error}"));
                    if let Err(error) = parse_native_model(&c3) {
                        assert!(!error.trim().is_empty(), "{deck}: empty native reason");
                    }
                }
            }
        }
        // The Phase 6 subset keeps resolving on file and model level.
        for deck in [
            "taylor_bar",
            "hertz_sphere",
            "necking_bar",
            "rail_crush",
            "taylor_plastic",
            "plasticity_thick_cylinder",
        ] {
            let c3 = resolve_c3_decks(deck).unwrap_or_else(|error| panic!("{deck}: {error}"));
            parse_native_model(&c3).unwrap_or_else(|error| panic!("{deck}: {error}"));
        }
        assert!(resolve_c3_decks("no_such_deck").is_err());
    }

    fn sample(wall_s: f64, correct: bool) -> RepeatSample {
        RepeatSample {
            wall_s,
            cycles: 100,
            dof: 600,
            peak_rss_bytes: 1000,
            correct,
        }
    }

    #[test]
    fn aggregation_takes_median_and_and() {
        let total =
            aggregate_repeats(&[sample(3.0, true), sample(1.0, true), sample(2.0, true)]).unwrap();
        assert_eq!(total.wall_s, 2.0);
        assert!(total.correct);
        let mixed = aggregate_repeats(&[sample(1.0, true), sample(2.0, false)]).unwrap();
        assert!(!mixed.correct);
        assert!(aggregate_repeats(&[]).is_err());
    }

    #[test]
    fn launch_labels_precision_suffix() {
        let mut args = args_with(SideOpt::Both, false, false);
        args.threads = 4;
        let launch = resolve_launch(&args, Path::new("engine_linux64_gf")).unwrap();
        assert_eq!(launch.flavor, FlavorOpt::Smp);
        assert_eq!(launch.label, "smp");
        assert_eq!(launch.row_threads, 4);
        let launch = resolve_launch(&args, Path::new("engine_linux64_gf_sp")).unwrap();
        assert_eq!(launch.label, "smp_sp");
        args.flavor = Some(FlavorOpt::Mpi);
        let launch = resolve_launch(&args, Path::new("engine_linux64_gf_ompi")).unwrap();
        assert_eq!(launch.row_threads, 4);
        assert_eq!(launch.omp_threads, 1);
    }

    #[test]
    fn engine_argv_shapes() {
        let smp = EngineLaunch {
            flavor: FlavorOpt::Smp,
            label: "smp".to_string(),
            row_threads: 4,
            omp_threads: 4,
            mpi_ranks: 4,
        };
        let (exe, argv) = engine_argv(&smp, Path::new("engine"), "case_0001.rad");
        assert_eq!(exe, PathBuf::from("engine"));
        assert_eq!(argv, vec!["-i", "case_0001.rad", "-nt", "4"]);
        let mpi = EngineLaunch {
            flavor: FlavorOpt::Mpi,
            label: "mpi".to_string(),
            row_threads: 2,
            omp_threads: 1,
            mpi_ranks: 2,
        };
        let (exe, argv) = engine_argv(&mpi, Path::new("/exec/engine_ompi"), "case_0001.rad");
        assert_eq!(exe, PathBuf::from("mpirun"));
        assert!(argv.contains(&"-np".to_string()));
    }

    #[test]
    fn total_cycles_sums_run_outs() {
        let dir = std::env::temp_dir().join(format!("oc_bench_cycles_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let first = dir.join("a_0001.out");
        let second = dir.join("a_0002.out");
        std::fs::write(&first, "     TOTAL NUMBER OF CYCLES  :      24\n").unwrap();
        std::fs::write(&second, "     TOTAL NUMBER OF CYCLES  :      10\n").unwrap();
        assert_eq!(total_cycles(&[first, second]).unwrap(), 34);
        assert!(total_cycles(&[]).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn starter_argv_carries_np() {
        assert_eq!(starter_argv("a.rad", None, &[]), vec!["-i", "a.rad"]);
        assert_eq!(
            starter_argv("a.rad", Some(2), &["-x".to_string()]),
            vec!["-i", "a.rad", "-np", "2", "-x"]
        );
    }

    #[test]
    fn history_lerp_interpolates_and_rejects_extrapolation() {
        let times = [0.0, 1.0, 2.0];
        let values = [0.0, 10.0, 20.0];
        assert_eq!(history_lerp(&times, &values, 0.5).unwrap(), 5.0);
        assert_eq!(history_lerp(&times, &values, 2.0).unwrap(), 20.0);
        assert!(history_lerp(&times, &values, -0.1).is_none());
        assert!(history_lerp(&times, &values, 2.1).is_none());
        assert!(history_lerp(&[], &[], 0.0).is_none());
        assert!(history_lerp(&[0.0], &[0.0, 1.0], 0.0).is_none());
    }

    #[test]
    fn linear_stiffness_matches_neohookean_at_small_strain() {
        let coords = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        let nodes: Vec<[f64; 3]> = coords.to_vec();
        let mesh = HexMesh::new(nodes, vec![[0, 1, 2, 3, 4, 5, 6, 7]]);
        let material = NeoHookean::from_young(200.0e9, 0.3, 3);
        // Small but nonlinear-visible displacement: linear matches the
        // finite-strain kernel to second order in strain.
        let mut u = [0.0; 24];
        for (i, slot) in u.iter_mut().enumerate() {
            *slot = 1.0e-6 * ((i * 2654435761 % 97) as f64 / 97.0 - 0.5);
        }
        let stiff = linear_hex_stiffness(material.lambda, material.mu, &coords);
        let mut linear = [0.0; 24];
        for row in 0..24 {
            linear[row] = stiff[row * 24..(row + 1) * 24]
                .iter()
                .zip(u.iter())
                .map(|(k, v)| k * v)
                .sum();
        }
        let finite = q1_element_residual(&material, &mesh, 0, &u);
        let norm: f64 = finite.iter().map(|v| v * v).sum::<f64>().sqrt();
        for (l, f) in linear.iter().zip(finite.iter()) {
            assert!(
                (l - f).abs() / norm < 1.0e-6,
                "linear {l:.6e} vs finite {f:.6e} (norm {norm:.6e})"
            );
        }
    }

    #[test]
    fn second_half_mean_skips_transient() {
        assert_eq!(second_half_mean(&[1.0, 2.0, 3.0, 4.0]).unwrap(), 3.5);
        assert_eq!(second_half_mean(&[7.0]).unwrap(), 7.0);
        assert!(second_half_mean(&[]).is_none());
    }

    #[test]
    fn history_trailmean_averages_last_n() {
        let values = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(history_trailmean(&values, 2).unwrap(), 3.5);
        assert_eq!(history_trailmean(&values, 99).unwrap(), 2.5);
        assert!(history_trailmean(&[], 50).is_none());
        assert!(history_trailmean(&values, 0).is_none());
    }

    #[test]
    fn history_holdmean_averages_from_start() {
        let times = [0.0, 1.0, 2.0, 3.0];
        let values = [10.0, 20.0, 30.0, 40.0];
        assert_eq!(history_holdmean(&times, &values, 2.0).unwrap(), 35.0);
        assert!(history_holdmean(&times, &values, 9.0).is_none());
        assert!(history_holdmean(&[], &[], 0.0).is_none());
        assert!(history_holdmean(&[0.0], &[0.0, 1.0], 0.0).is_none());
    }

    #[test]
    fn thick_cylinder_times_match_engine_leg() {
        // Transcribed from fem_opencourant_plasticity_thick_cylinder:
        // 50 MPa crossing of the raised cosine plus one dilatational
        // transit (8.5 us) for the outer probe.
        let (t_elastic, t_outer) = thick_cylinder_sample_times();
        assert!((t_elastic - 3.918_266e-4).abs() < 1.0e-10);
        assert!((t_outer - t_elastic - 8.5e-6).abs() < 0.1e-6);
    }
}
