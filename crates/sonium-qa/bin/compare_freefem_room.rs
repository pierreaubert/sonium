//! Compare Sonium FEM room pressures with the same physical Helmholtz problem
//! solved by FreeFEM.

use anyhow::{Context, Result};
use clap::Parser;
use num_complex::Complex64;
use serde::Serialize;
use sonium_fem::mesh::{ElementType, Mesh};
use std::env;
use std::f64::consts::PI;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use sonium_fem::backend_selection::FemPreconditionerKind;
use sonium_sim_room::SimulationOutput;
use sonium_sim_room::fem_runner::{self, FemMeshConfig, RecyclerKind, complex_wavenumber};
use sonium_sim_room::scenarios::{self, FrequencyConfig};
use sonium_xem::FrequencySpacing;
use sonium_xem::{Point3D, RoomGeometry, RoomSimulation};

#[derive(Parser, Debug)]
#[command(
    name = "compare-freefem-room",
    about = "Compare sonium-sim-room FEM Helmholtz pressures against FreeFEM."
)]
struct Args {
    /// Scenario name from sonium-sim-room.
    #[arg(long, default_value = "small_stereo_2_0")]
    scenario: String,
    /// Lowest frequency in the comparison sweep (Hz).
    #[arg(long, default_value_t = 30.0)]
    min_freq: f64,
    /// Highest frequency in the comparison sweep (Hz).
    #[arg(long, default_value_t = 150.0)]
    max_freq: f64,
    /// Number of frequencies in the comparison sweep.
    #[arg(long, default_value_t = 15)]
    num_freqs: usize,
    /// FreeFEM executable.
    #[arg(long, default_value = "FreeFem++")]
    freefem_bin: String,
    /// Mesh source. `sonium` exports Sonium's full-domain tetra mesh for
    /// FreeFEM; `freefem-cube` lets FreeFEM generate a rectangular cube mesh.
    #[arg(long, default_value = "freefem-cube")]
    mesh_source: String,
    /// Output directory for generated FreeFEM script, CSVs, and JSON.
    #[arg(long, default_value = "/tmp/sonium-freefem-compare")]
    output_dir: PathBuf,
    /// Optional final JSON report path. Defaults to `<output-dir>/compare.json`.
    #[arg(long)]
    json_out: Option<PathBuf>,
    /// Generate inputs and run Sonium, but do not execute FreeFEM.
    #[arg(long)]
    emit_only: bool,
    /// Recycled-subspace dimension for the Sonium run.
    #[arg(long, default_value_t = fem_runner::DEFAULT_RECYCLE_SUBSPACE_SIZE)]
    recycle_size: usize,
    /// Sonium preconditioner.
    #[arg(long, default_value_t = FemPreconditionerKind::default(), value_parser = parse_preconditioner)]
    preconditioner: FemPreconditionerKind,
    /// Rebuild the Sonium preconditioner every N frequencies.
    #[arg(long, default_value_t = 5)]
    rebuild_every: usize,
    /// Sonium GMRES tolerance.
    #[arg(long, default_value_t = 1.0e-6)]
    gmres_tolerance: f64,
    /// Modal damping floor used by both Sonium and the generated FreeFEM script.
    #[arg(long, default_value_t = fem_runner::DEFAULT_MODAL_DAMPING_RATIO)]
    modal_damping: f64,
    /// Override FreeFEM cube subdivisions along x.
    #[arg(long)]
    freefem_nx: Option<usize>,
    /// Override FreeFEM cube subdivisions along y.
    #[arg(long)]
    freefem_ny: Option<usize>,
    /// Override FreeFEM cube subdivisions along z.
    #[arg(long)]
    freefem_nz: Option<usize>,
}

fn parse_preconditioner(s: &str) -> std::result::Result<FemPreconditionerKind, String> {
    s.parse()
}

#[derive(Debug, Clone, Copy, Serialize)]
struct RoomDims {
    width: f64,
    depth: f64,
    height: f64,
}

#[derive(Debug, Clone, Serialize)]
struct ArtifactPaths {
    output_dir: String,
    freefem_script: String,
    freefem_mesh: Option<String>,
    freefem_csv: String,
    freefem_timing_csv: String,
    sonium_json: String,
}

#[derive(Debug, Clone, Serialize)]
struct SolverSummary {
    backend: String,
    wall_clock_s: f64,
    total_iterations: Option<usize>,
    total_inner_iterations: Option<usize>,
    retry_count: Option<usize>,
    max_residual: Option<f64>,
    freefem_solve_s: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct ComparisonMetrics {
    max_abs: f64,
    max_abs_over_p90: f64,
    l2_rel: f64,
    max_spl_db: f64,
    rms_spl_db: f64,
    max_phase_deg: f64,
}

#[derive(Debug, Clone, Serialize)]
struct PressureRow {
    source_idx: usize,
    source_name: String,
    listener_idx: usize,
    freq_idx: usize,
    freq_hz: f64,
    re: f64,
    im: f64,
    magnitude: f64,
    spl_db: f64,
    phase_deg: f64,
}

#[derive(Debug, Clone, Serialize)]
struct BoundaryCoeffRow {
    freq_idx: usize,
    freq_hz: f64,
    floor: [f64; 2],
    ceiling: [f64; 2],
    front: [f64; 2],
    back: [f64; 2],
    left: [f64; 2],
    right: [f64; 2],
}

#[derive(Debug, Clone, Serialize)]
struct MeshSummary {
    nodes: usize,
    tetrahedra: usize,
    boundary_triangles: usize,
}

#[derive(Debug, Clone, Serialize)]
struct ProblemSummary {
    scenario: String,
    description: String,
    room: RoomDims,
    mesh_source: String,
    freefem_divisions: Option<[usize; 3]>,
    exported_mesh: Option<MeshSummary>,
    frequencies: Vec<f64>,
    source_width: f64,
    modal_damping: f64,
    boundary_coefficients: Vec<BoundaryCoeffRow>,
}

#[derive(Debug, Clone, Serialize)]
struct CompareReport {
    problem: ProblemSummary,
    sonium: SolverSummary,
    freefem: Option<SolverSummary>,
    metrics: Option<ComparisonMetrics>,
    sonium_pressures: Vec<PressureRow>,
    freefem_pressures: Vec<PressureRow>,
    artifacts: ArtifactPaths,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MeshSourceKind {
    FreeFemCube,
    Sonium,
}

impl MeshSourceKind {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "freefem-cube" => Ok(Self::FreeFemCube),
            "sonium" => Ok(Self::Sonium),
            _ => anyhow::bail!(
                "invalid --mesh-source {value:?}; expected `freefem-cube` or `sonium`"
            ),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::FreeFemCube => "freefem-cube",
            Self::Sonium => "sonium",
        }
    }
}

enum FreeFemMeshInput<'a> {
    Cube([usize; 3]),
    MeshFile(&'a Path),
}

fn main() {
    if let Err(error) = run() {
        eprintln!("FAILED: {error:#}");
        std::process::exit(1);
    }
    println!("SUCCESS: compare-freefem-room");
}

fn run() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let args = Args::parse();
    let mesh_source = MeshSourceKind::parse(&args.mesh_source)?;
    anyhow::ensure!(
        args.modal_damping.is_finite() && args.modal_damping >= 0.0,
        "--modal-damping must be finite and non-negative"
    );

    fs::create_dir_all(&args.output_dir).with_context(|| {
        format!(
            "failed to create output directory {}",
            args.output_dir.display()
        )
    })?;

    let freq_config = FrequencyConfig {
        min_freq: args.min_freq,
        max_freq: args.max_freq,
        num_points: args.num_freqs,
        spacing: FrequencySpacing::default(),
    };
    let scenario = scenarios::scenario_by_name_with_config(&args.scenario, &freq_config)
        .ok_or_else(|| anyhow::anyhow!("scenario not found: {}", args.scenario))?;
    let dims = rectangular_room_dims(&scenario.simulation)?;

    let script_path = args.output_dir.join("freefem_room.edp");
    let mesh_path = args.output_dir.join("sonium_room.mesh");
    let csv_path = args.output_dir.join("freefem_pressures.csv");
    let timing_path = args.output_dir.join("freefem_timing.csv");
    let sonium_json_path = args.output_dir.join("sonium_pressures.json");
    let json_out = args
        .json_out
        .clone()
        .unwrap_or_else(|| args.output_dir.join("compare.json"));

    let fem_config = FemMeshConfig {
        preconditioner: args.preconditioner,
        recycler: RecyclerKind::GcroDr,
        recycle_subspace_size: args.recycle_size,
        rebuild_precond_every: args.rebuild_every.max(1),
        gmres_tolerance: args.gmres_tolerance,
        modal_damping_ratio: args.modal_damping,
        ..FemMeshConfig::default()
    };

    let divisions = freefem_divisions(&args, dims, scenario.simulation.speed_of_sound);
    let mut exported_mesh = None;
    let freefem_mesh_input = match mesh_source {
        MeshSourceKind::FreeFemCube => FreeFemMeshInput::Cube(divisions),
        MeshSourceKind::Sonium => {
            let mesh = fem_runner::build_full_domain_room_mesh(&scenario.simulation, &fem_config)?;
            let summary = write_freefem_medit_mesh(&mesh, &mesh_path)?;
            exported_mesh = Some(summary);
            FreeFemMeshInput::MeshFile(&mesh_path)
        }
    };

    let freefem_script = generate_freefem_script(
        &scenario.simulation,
        dims,
        freefem_mesh_input,
        &csv_path,
        &timing_path,
        args.modal_damping,
    )?;
    fs::write(&script_path, freefem_script)
        .with_context(|| format!("failed to write {}", script_path.display()))?;

    println!("Scenario: {} ({})", scenario.name, scenario.description);
    match mesh_source {
        MeshSourceKind::FreeFemCube => println!(
            "Sweep: {:.1} Hz -> {:.1} Hz, {} points; FreeFEM mesh {}x{}x{}",
            args.min_freq, args.max_freq, args.num_freqs, divisions[0], divisions[1], divisions[2]
        ),
        MeshSourceKind::Sonium => {
            let mesh = exported_mesh.as_ref().ok_or_else(|| {
                anyhow::anyhow!("internal error: sonium mesh summary was not recorded")
            })?;
            println!(
                "Sweep: {:.1} Hz -> {:.1} Hz, {} points; exported Sonium mesh: {} nodes, {} tetrahedra",
                args.min_freq, args.max_freq, args.num_freqs, mesh.nodes, mesh.tetrahedra
            );
        }
    }

    let sonium_start = Instant::now();
    let (sonium_output, sonium_diag) = match mesh_source {
        MeshSourceKind::FreeFemCube => {
            fem_runner::run_fem_with_mesh_report(&scenario.simulation, &fem_config)?
        }
        MeshSourceKind::Sonium => {
            fem_runner::run_fem_full_domain_with_mesh_report(&scenario.simulation, &fem_config)?
        }
    };
    let sonium_wall = sonium_start.elapsed().as_secs_f64();
    let sonium_rows = pressure_rows(&sonium_output, &scenario.simulation)?;
    fs::write(&sonium_json_path, serde_json::to_vec_pretty(&sonium_rows)?)?;
    println!(
        "Sonium: {:.3}s, iters={}, retries={}, max_res={:.2e}",
        sonium_wall,
        sonium_diag.total_iterations,
        sonium_diag.retry_count,
        sonium_diag.max_residual
    );

    let mut freefem_summary = None;
    let mut freefem_rows = Vec::new();
    let mut metrics = None;

    if !args.emit_only {
        let freefem_start = Instant::now();
        run_freefem(&args.freefem_bin, &script_path)?;
        let freefem_wall = freefem_start.elapsed().as_secs_f64();
        let freefem_iso = parse_freefem_csv(
            &fs::read_to_string(&csv_path)
                .with_context(|| format!("failed to read {}", csv_path.display()))?,
            &scenario.simulation,
        )?;
        let freefem_output = apply_receiver_directivity(freefem_iso, &scenario.simulation);
        let freefem_solve_s = parse_freefem_timing_csv(
            &fs::read_to_string(&timing_path)
                .with_context(|| format!("failed to read {}", timing_path.display()))?,
        )?;
        metrics = Some(compare_outputs(&sonium_output, &freefem_output)?);
        freefem_rows = pressure_rows(&freefem_output, &scenario.simulation)?;
        println!(
            "FreeFEM: {:.3}s process wall, {:.3}s accumulated solve",
            freefem_wall, freefem_solve_s
        );
        if let Some(m) = &metrics {
            println!(
                "Agreement: max |dp|/p90={:.3e}, L2 rel={:.3e}, max SPL={:.3} dB",
                m.max_abs_over_p90, m.l2_rel, m.max_spl_db
            );
        }
        freefem_summary = Some(SolverSummary {
            backend: "freefem".to_string(),
            wall_clock_s: freefem_wall,
            total_iterations: None,
            total_inner_iterations: None,
            retry_count: None,
            max_residual: None,
            freefem_solve_s: Some(freefem_solve_s),
        });
    } else {
        println!(
            "FreeFEM: emit-only mode; generated {} but did not execute it",
            script_path.display()
        );
    }

    let report = CompareReport {
        problem: ProblemSummary {
            scenario: scenario.name,
            description: scenario.description,
            room: dims,
            mesh_source: mesh_source.as_str().to_string(),
            freefem_divisions: (mesh_source == MeshSourceKind::FreeFemCube).then_some(divisions),
            exported_mesh: exported_mesh.clone(),
            frequencies: sonium_output.frequencies.clone(),
            source_width: fem_runner::DEFAULT_SOURCE_WIDTH,
            modal_damping: args.modal_damping,
            boundary_coefficients: boundary_coeff_rows(&scenario.simulation)?,
        },
        sonium: SolverSummary {
            backend: "sonium-cpu-fem".to_string(),
            wall_clock_s: sonium_wall,
            total_iterations: Some(sonium_diag.total_iterations),
            total_inner_iterations: Some(sonium_diag.total_inner_iterations),
            retry_count: Some(sonium_diag.retry_count),
            max_residual: Some(sonium_diag.max_residual),
            freefem_solve_s: None,
        },
        freefem: freefem_summary,
        metrics,
        sonium_pressures: sonium_rows,
        freefem_pressures: freefem_rows,
        artifacts: ArtifactPaths {
            output_dir: args.output_dir.display().to_string(),
            freefem_script: script_path.display().to_string(),
            freefem_mesh: (mesh_source == MeshSourceKind::Sonium)
                .then(|| mesh_path.display().to_string()),
            freefem_csv: csv_path.display().to_string(),
            freefem_timing_csv: timing_path.display().to_string(),
            sonium_json: sonium_json_path.display().to_string(),
        },
    };
    fs::write(&json_out, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("failed to write {}", json_out.display()))?;
    println!("Wrote comparison report to {}", json_out.display());

    Ok(())
}

fn rectangular_room_dims(simulation: &RoomSimulation) -> Result<RoomDims> {
    match &simulation.room {
        RoomGeometry::Rectangular(room) => Ok(RoomDims {
            width: room.width,
            depth: room.depth,
            height: room.height,
        }),
        RoomGeometry::LShaped(_) => {
            anyhow::bail!("FreeFEM comparison currently supports rectangular scenarios only")
        }
    }
}

fn freefem_divisions(args: &Args, dims: RoomDims, speed_of_sound: f64) -> [usize; 3] {
    let min_wavelength = speed_of_sound / args.max_freq;
    let elements_per_meter = (fem_runner::ELEMENTS_PER_WAVELENGTH / min_wavelength).ceil();
    [
        args.freefem_nx
            .unwrap_or_else(|| (dims.width * elements_per_meter).ceil().max(1.0) as usize),
        args.freefem_ny
            .unwrap_or_else(|| (dims.depth * elements_per_meter).ceil().max(1.0) as usize),
        args.freefem_nz
            .unwrap_or_else(|| (dims.height * elements_per_meter).ceil().max(1.0) as usize),
    ]
}

fn write_freefem_medit_mesh(mesh: &Mesh, path: &Path) -> Result<MeshSummary> {
    anyhow::ensure!(
        mesh.dimension == 3,
        "FreeFEM mesh export requires a 3D mesh"
    );

    let mut tets = Vec::new();
    for elem in &mesh.elements {
        anyhow::ensure!(
            elem.element_type == ElementType::Tetrahedron,
            "FreeFEM mesh export only supports P1 tetrahedra"
        );
        tets.push(oriented_tet_vertices(mesh, elem.vertices())?);
    }

    let triangles: Vec<([usize; 3], i32)> = mesh
        .boundaries
        .iter()
        .filter(|face| face.nodes.len() == 3)
        .map(|face| ([face.nodes[0], face.nodes[1], face.nodes[2]], face.marker))
        .collect();

    let mut out = String::new();
    out.push_str("MeshVersionFormatted 1\n");
    out.push_str("Dimension 3\n\n");
    out.push_str("Vertices\n");
    out.push_str(&format!("{}\n", mesh.num_nodes()));
    for node in &mesh.nodes {
        anyhow::ensure!(
            node.x.is_finite() && node.y.is_finite() && node.z.is_finite(),
            "cannot export non-finite mesh node"
        );
        out.push_str(&format!(
            "{:.17e} {:.17e} {:.17e} 1\n",
            node.x, node.y, node.z
        ));
    }
    out.push('\n');

    out.push_str("Triangles\n");
    out.push_str(&format!("{}\n", triangles.len()));
    for (tri, marker) in &triangles {
        out.push_str(&format!(
            "{} {} {} {}\n",
            tri[0] + 1,
            tri[1] + 1,
            tri[2] + 1,
            (*marker).max(1)
        ));
    }
    out.push('\n');

    out.push_str("Tetrahedra\n");
    out.push_str(&format!("{}\n", tets.len()));
    for tet in &tets {
        out.push_str(&format!(
            "{} {} {} {} 1\n",
            tet[0] + 1,
            tet[1] + 1,
            tet[2] + 1,
            tet[3] + 1
        ));
    }
    out.push_str("\nEnd\n");

    fs::write(path, out).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(MeshSummary {
        nodes: mesh.num_nodes(),
        tetrahedra: tets.len(),
        boundary_triangles: triangles.len(),
    })
}

fn oriented_tet_vertices(mesh: &Mesh, vertices: &[usize]) -> Result<[usize; 4]> {
    anyhow::ensure!(
        vertices.len() == 4,
        "FreeFEM mesh export expected tetrahedron with 4 vertices"
    );
    let mut tet = [vertices[0], vertices[1], vertices[2], vertices[3]];
    let vol6 = signed_tet_volume6(mesh, tet);
    anyhow::ensure!(
        vol6.is_finite() && vol6.abs() > 1.0e-30,
        "cannot export degenerate tetrahedron"
    );
    if vol6 < 0.0 {
        tet.swap(1, 2);
    }
    Ok(tet)
}

fn signed_tet_volume6(mesh: &Mesh, tet: [usize; 4]) -> f64 {
    let a = mesh.node(tet[0]);
    let b = mesh.node(tet[1]);
    let c = mesh.node(tet[2]);
    let d = mesh.node(tet[3]);
    let ab = [b.x - a.x, b.y - a.y, b.z - a.z];
    let ac = [c.x - a.x, c.y - a.y, c.z - a.z];
    let ad = [d.x - a.x, d.y - a.y, d.z - a.z];
    let cross = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    cross[0] * ad[0] + cross[1] * ad[1] + cross[2] * ad[2]
}

fn generate_freefem_script(
    simulation: &RoomSimulation,
    dims: RoomDims,
    mesh_input: FreeFemMeshInput<'_>,
    csv_path: &Path,
    timing_path: &Path,
    modal_damping: f64,
) -> Result<String> {
    let mut script = String::new();
    script.push_str("load \"msh3\"\n");
    script.push_str("verbosity = 0;\n");
    script.push_str(&format!(
        "real Lx = {:.17e}; real Ly = {:.17e}; real Lz = {:.17e};\n",
        dims.width, dims.depth, dims.height
    ));
    match mesh_input {
        FreeFemMeshInput::Cube(divisions) => {
            script.push_str(&format!(
                "int nx = {}; int ny = {}; int nz = {};\n",
                divisions[0], divisions[1], divisions[2]
            ));
            script.push_str("mesh3 Th = cube(nx, ny, nz, [x*Lx, y*Ly, z*Lz]);\n");
        }
        FreeFemMeshInput::MeshFile(path) => {
            script.push_str(&format!("mesh3 Th = readmesh3(\"{}\");\n", ff_string(path)));
        }
    }
    script.push_str("fespace Vh(Th, P1);\n");
    script.push_str("Vh<complex> p, v;\n");
    script.push_str("complex k2 = 0;\n");
    script.push_str("complex rhsScale = 0;\n");
    script.push_str(
        "complex gFloor = 0, gCeiling = 0, gFront = 0, gBack = 0, gLeft = 0, gRight = 0;\n",
    );
    script.push_str("real solveStart = 0;\n");
    script.push_str("real epsBoundary = 1.0e-8 * max(Lx, max(Ly, Lz));\n");
    script.push_str("func gammaBoundary = (z < epsBoundary) ? gFloor : ((z > Lz - epsBoundary) ? gCeiling : ((y < epsBoundary) ? gFront : ((y > Ly - epsBoundary) ? gBack : ((x < epsBoundary) ? gLeft : gRight))));\n");
    script.push_str(&format!(
        "ofstream out(\"{}\"); out.precision(17);\n",
        ff_string(csv_path)
    ));
    script.push_str(&format!(
        "ofstream tout(\"{}\"); tout.precision(17);\n",
        ff_string(timing_path)
    ));
    script.push_str("out << \"source_idx,listener_idx,freq_idx,freq_hz,re,im\" << endl;\n");
    script.push_str("tout << \"freq_idx,source_idx,solve_s\" << endl;\n");

    let aabb = ([0.0, 0.0, 0.0], [dims.width, dims.depth, dims.height]);
    for (src_idx, source) in simulation.sources.iter().enumerate() {
        script.push_str(&format!(
            "func gauss{} = {};\n",
            src_idx,
            gaussian_expression(source.position, aabb, fem_runner::DEFAULT_SOURCE_WIDTH)
        ));
        script.push_str(&format!(
            "problem Helm{}(p, v, solver=sparsesolver) = int3d(Th)(dx(p)*dx(v) + dy(p)*dy(v) + dz(p)*dz(v) - k2*p*v) + int2d(Th)(gammaBoundary*p*v) - int3d(Th)(rhsScale*gauss{}*v);\n",
            src_idx, src_idx
        ));
    }

    for (freq_idx, &freq) in simulation.frequencies.iter().enumerate() {
        let k_real = simulation.wavenumber(freq);
        let k_complex = complex_wavenumber(k_real, freq, modal_damping);
        let coeffs = fem_runner::try_fem_boundary_coefficients(simulation, freq)?;
        script.push_str(&format!("// frequency {}: {:.17e} Hz\n", freq_idx, freq));
        script.push_str(&format!("k2 = {};\n", ff_complex(k_complex * k_complex)));
        script.push_str(&format!("gFloor = {}; ", ff_complex(coeffs.floor)));
        script.push_str(&format!("gCeiling = {}; ", ff_complex(coeffs.ceiling)));
        script.push_str(&format!("gFront = {}; ", ff_complex(coeffs.front)));
        script.push_str(&format!("gBack = {}; ", ff_complex(coeffs.back)));
        script.push_str(&format!("gLeft = {}; ", ff_complex(coeffs.left)));
        script.push_str(&format!("gRight = {};\n", ff_complex(coeffs.right)));

        for (src_idx, source) in simulation.sources.iter().enumerate() {
            let q = source.isotropic_amplitude(freq);
            let omega = 2.0 * PI * freq;
            let two_sigma_sq =
                2.0 * fem_runner::DEFAULT_SOURCE_WIDTH * fem_runner::DEFAULT_SOURCE_WIDTH;
            let norm = 1.0 / ((PI * two_sigma_sq).powf(1.5));
            let rhs_scale = Complex64::new(
                0.0,
                omega
                    * fem_runner::AIR_DENSITY
                    * fem_runner::SOURCE_VOLUME_VELOCITY_REFERENCE
                    * norm
                    * q,
            );
            script.push_str(&format!("rhsScale = {}; p = 0;\n", ff_complex(rhs_scale)));
            script.push_str("solveStart = clock();\n");
            script.push_str(&format!("Helm{};\n", src_idx));
            script.push_str(&format!(
                "tout << {} << \",\" << {} << \",\" << (clock() - solveStart) << endl;\n",
                freq_idx, src_idx
            ));
            for (lp_idx, lp) in simulation.listening_positions.iter().enumerate() {
                script.push_str(&format!(
                    "{{ complex val = p({:.17e}, {:.17e}, {:.17e}); out << {} << \",\" << {} << \",\" << {} << \",\" << {:.17e} << \",\" << real(val) << \",\" << imag(val) << endl; }}\n",
                    lp.x, lp.y, lp.z, src_idx, lp_idx, freq_idx, freq
                ));
            }
        }
    }

    Ok(script)
}

fn gaussian_expression(point: Point3D, aabb: ([f64; 3], [f64; 3]), source_width: f64) -> String {
    let cutoff = fem_runner::SOURCE_TRUNCATION_SIGMAS * source_width;
    let positions = image_positions(point, aabb, cutoff);
    let two_sigma_sq = 2.0 * source_width * source_width;
    positions
        .iter()
        .map(|p| {
            format!(
                "exp(-((x-({:.17e}))^2 + (y-({:.17e}))^2 + (z-({:.17e}))^2)/({:.17e}))",
                p.x, p.y, p.z, two_sigma_sq
            )
        })
        .collect::<Vec<_>>()
        .join(" + ")
}

fn image_positions(p: Point3D, aabb: ([f64; 3], [f64; 3]), cutoff: f64) -> Vec<Point3D> {
    let (lo, hi) = aabb;
    let mut xs = vec![p.x];
    if (p.x - lo[0]).abs() < cutoff {
        xs.push(2.0 * lo[0] - p.x);
    }
    if (p.x - hi[0]).abs() < cutoff {
        xs.push(2.0 * hi[0] - p.x);
    }
    let mut ys = vec![p.y];
    if (p.y - lo[1]).abs() < cutoff {
        ys.push(2.0 * lo[1] - p.y);
    }
    if (p.y - hi[1]).abs() < cutoff {
        ys.push(2.0 * hi[1] - p.y);
    }
    let mut zs = vec![p.z];
    if (p.z - lo[2]).abs() < cutoff {
        zs.push(2.0 * lo[2] - p.z);
    }
    if (p.z - hi[2]).abs() < cutoff {
        zs.push(2.0 * hi[2] - p.z);
    }

    let mut out = Vec::with_capacity(xs.len() * ys.len() * zs.len());
    for &x in &xs {
        for &y in &ys {
            for &z in &zs {
                out.push(Point3D::new(x, y, z));
            }
        }
    }
    out
}

fn run_freefem(freefem_bin: &str, script_path: &Path) -> Result<()> {
    let mut command = Command::new(freefem_bin);
    if env::var_os("FF_LOADPATH").is_none() {
        let plugin_dirs = [PathBuf::from("/usr/lib/freefem++")];
        if let Some(plugin_dir) = find_freefem_plugin_dir(&plugin_dirs) {
            command.env("FF_LOADPATH", plugin_dir);
        }
    }
    let output = command
        .arg("-nw")
        .arg(script_path)
        .output()
        .with_context(|| format!("failed to run {freefem_bin}"))?;
    if !output.status.success() {
        anyhow::bail!(
            "FreeFEM failed with status {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(())
}

fn find_freefem_plugin_dir(candidates: &[PathBuf]) -> Option<&Path> {
    candidates
        .iter()
        .map(PathBuf::as_path)
        .find(|path| path.join("msh3.so").is_file())
}

fn parse_freefem_csv(csv: &str, simulation: &RoomSimulation) -> Result<SimulationOutput> {
    let n_sources = simulation.sources.len();
    let n_lps = simulation.listening_positions.len();
    let n_freqs = simulation.frequencies.len();
    let mut pressures =
        vec![vec![vec![Complex64::new(f64::NAN, f64::NAN); n_freqs]; n_lps]; n_sources];
    let mut seen = vec![vec![vec![false; n_freqs]; n_lps]; n_sources];

    for (line_idx, line) in csv.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("source_idx") {
            continue;
        }
        let cols: Vec<&str> = line.split(',').map(str::trim).collect();
        anyhow::ensure!(
            cols.len() == 6,
            "invalid FreeFEM CSV line {}: expected 6 columns, got {}",
            line_idx + 1,
            cols.len()
        );
        let src_idx: usize = cols[0].parse()?;
        let lp_idx: usize = cols[1].parse()?;
        let freq_idx: usize = cols[2].parse()?;
        let re: f64 = cols[4].parse()?;
        let im: f64 = cols[5].parse()?;
        anyhow::ensure!(
            src_idx < n_sources && lp_idx < n_lps && freq_idx < n_freqs,
            "FreeFEM CSV index out of range on line {}",
            line_idx + 1
        );
        pressures[src_idx][lp_idx][freq_idx] = Complex64::new(re, im);
        seen[src_idx][lp_idx][freq_idx] = true;
    }

    for (src_idx, src_seen) in seen.iter().enumerate() {
        for (lp_idx, lp_seen) in src_seen.iter().enumerate() {
            for (freq_idx, &was_seen) in lp_seen.iter().enumerate() {
                anyhow::ensure!(
                    was_seen,
                    "missing FreeFEM pressure row for source {}, listener {}, frequency {}",
                    src_idx,
                    lp_idx,
                    freq_idx
                );
            }
        }
    }

    Ok(SimulationOutput::solved(
        simulation.frequencies.clone(),
        pressures,
        simulation.sources.iter().map(|s| s.name.clone()).collect(),
    ))
}

fn parse_freefem_timing_csv(csv: &str) -> Result<f64> {
    let mut total = 0.0;
    for line in csv.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("freq_idx") {
            continue;
        }
        let cols: Vec<&str> = line.split(',').map(str::trim).collect();
        anyhow::ensure!(cols.len() == 3, "invalid FreeFEM timing row: {line}");
        total += cols[2].parse::<f64>()?;
    }
    Ok(total)
}

fn apply_receiver_directivity(
    mut output: SimulationOutput,
    simulation: &RoomSimulation,
) -> SimulationOutput {
    for (src_idx, source) in simulation.sources.iter().enumerate() {
        for (lp_idx, lp) in simulation.listening_positions.iter().enumerate() {
            for (freq_idx, &freq) in simulation.frequencies.iter().enumerate() {
                let dir = source.directivity_factor(lp, freq);
                output.pressures[src_idx][lp_idx][freq_idx] *= Complex64::new(dir, 0.0);
            }
        }
    }
    output
}

fn pressure_rows(
    output: &SimulationOutput,
    simulation: &RoomSimulation,
) -> Result<Vec<PressureRow>> {
    anyhow::ensure!(
        output.frequencies.len() == simulation.frequencies.len(),
        "pressure output frequency count mismatch"
    );
    let mut rows = Vec::new();
    for (src_idx, src_pressures) in output.pressures.iter().enumerate() {
        let source_name = output
            .source_names
            .get(src_idx)
            .cloned()
            .unwrap_or_else(|| format!("source_{src_idx}"));
        for (lp_idx, lp_pressures) in src_pressures.iter().enumerate() {
            for (freq_idx, &p) in lp_pressures.iter().enumerate() {
                rows.push(PressureRow {
                    source_idx: src_idx,
                    source_name: source_name.clone(),
                    listener_idx: lp_idx,
                    freq_idx,
                    freq_hz: output.frequencies[freq_idx],
                    re: p.re,
                    im: p.im,
                    magnitude: p.norm(),
                    spl_db: spl_db(p),
                    phase_deg: p.arg().to_degrees(),
                });
            }
        }
    }
    Ok(rows)
}

fn compare_outputs(
    reference: &SimulationOutput,
    candidate: &SimulationOutput,
) -> Result<ComparisonMetrics> {
    anyhow::ensure!(
        reference.frequencies.len() == candidate.frequencies.len(),
        "frequency count mismatch"
    );
    let p90 = p90_pressure(reference);
    let mut max_abs = 0.0_f64;
    let mut sumsq_diff = 0.0_f64;
    let mut sumsq_ref = 0.0_f64;
    let mut max_spl_db = 0.0_f64;
    let mut sumsq_spl_db = 0.0_f64;
    let mut max_phase_deg = 0.0_f64;
    let mut count = 0usize;

    for (src_ref, src_cand) in reference.pressures.iter().zip(candidate.pressures.iter()) {
        for (lp_ref, lp_cand) in src_ref.iter().zip(src_cand.iter()) {
            for (&a, &b) in lp_ref.iter().zip(lp_cand.iter()) {
                let abs = (a - b).norm();
                max_abs = max_abs.max(abs);
                sumsq_diff += abs * abs;
                sumsq_ref += a.norm_sqr();

                let spl_diff = (spl_db(a) - spl_db(b)).abs();
                max_spl_db = max_spl_db.max(spl_diff);
                sumsq_spl_db += spl_diff * spl_diff;

                let phase_diff =
                    wrapped_phase_diff_deg(a.arg().to_degrees(), b.arg().to_degrees()).abs();
                max_phase_deg = max_phase_deg.max(phase_diff);
                count += 1;
            }
        }
    }

    Ok(ComparisonMetrics {
        max_abs,
        max_abs_over_p90: max_abs / p90,
        l2_rel: sumsq_diff.sqrt() / sumsq_ref.sqrt().max(1.0e-30),
        max_spl_db,
        rms_spl_db: (sumsq_spl_db / count.max(1) as f64).sqrt(),
        max_phase_deg,
    })
}

fn p90_pressure(output: &SimulationOutput) -> f64 {
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
        .max(1.0e-12)
}

fn spl_db(p: Complex64) -> f64 {
    20.0 * (p.norm() / 20.0e-6).max(1.0e-30).log10()
}

fn wrapped_phase_diff_deg(a: f64, b: f64) -> f64 {
    let mut d = a - b;
    while d > 180.0 {
        d -= 360.0;
    }
    while d < -180.0 {
        d += 360.0;
    }
    d
}

fn boundary_coeff_rows(simulation: &RoomSimulation) -> Result<Vec<BoundaryCoeffRow>> {
    simulation
        .frequencies
        .iter()
        .enumerate()
        .map(|(freq_idx, &freq)| {
            let c = fem_runner::try_fem_boundary_coefficients(simulation, freq)?;
            Ok(BoundaryCoeffRow {
                freq_idx,
                freq_hz: freq,
                floor: [c.floor.re, c.floor.im],
                ceiling: [c.ceiling.re, c.ceiling.im],
                front: [c.front.re, c.front.im],
                back: [c.back.re, c.back.im],
                left: [c.left.re, c.left.im],
                right: [c.right.re, c.right.im],
            })
        })
        .collect()
}

fn ff_complex(z: Complex64) -> String {
    format!("({:.17e})+({:.17e})*1i", z.re, z.im)
}

fn ff_string(path: &Path) -> String {
    path.display()
        .to_string()
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_fem::mesh::{BoundaryType, Point};
    use sonium_xem::{RectangularRoom, Source};
    use std::fs;

    #[test]
    fn parses_freefem_csv_into_pressure_cube() {
        let mut sim = RoomSimulation::new(
            RoomGeometry::Rectangular(RectangularRoom::new(1.0, 1.0, 1.0)),
            vec![Source::omnidirectional(Point3D::new(0.5, 0.5, 0.5), 1.0)],
            vec![Point3D::new(0.25, 0.25, 0.25)],
        );
        sim.frequencies = vec![30.0, 40.0];
        let csv =
            "source_idx,listener_idx,freq_idx,freq_hz,re,im\n0,0,0,30,1.0,2.0\n0,0,1,40,3.0,-4.0\n";
        let out = parse_freefem_csv(csv, &sim).expect("csv should parse");
        assert_eq!(out.pressures[0][0][0], Complex64::new(1.0, 2.0));
        assert_eq!(out.pressures[0][0][1], Complex64::new(3.0, -4.0));
    }

    #[test]
    fn parses_mesh_source_modes() {
        assert_eq!(
            MeshSourceKind::parse("freefem-cube").expect("cube mode parses"),
            MeshSourceKind::FreeFemCube
        );
        assert_eq!(
            MeshSourceKind::parse("sonium").expect("sonium mode parses"),
            MeshSourceKind::Sonium
        );
        assert!(MeshSourceKind::parse("gmsh").is_err());
    }

    #[test]
    fn discovers_freefem_msh3_plugin_directory() -> Result<()> {
        let missing = tempfile::tempdir()?;
        let installed = tempfile::tempdir()?;
        fs::write(installed.path().join("msh3.so"), [])?;
        let candidates = vec![missing.path().to_owned(), installed.path().to_owned()];

        assert_eq!(find_freefem_plugin_dir(&candidates), Some(installed.path()));
        Ok(())
    }

    #[test]
    fn exports_sonium_mesh_as_medit_file() -> Result<()> {
        let mut mesh = Mesh::new(3);
        let n0 = mesh.add_node(Point::new_3d(0.0, 0.0, 0.0));
        let n1 = mesh.add_node(Point::new_3d(1.0, 0.0, 0.0));
        let n2 = mesh.add_node(Point::new_3d(0.0, 1.0, 0.0));
        let n3 = mesh.add_node(Point::new_3d(0.0, 0.0, 1.0));
        mesh.add_element(ElementType::Tetrahedron, vec![n0, n1, n2, n3]);
        mesh.add_boundary(vec![n0, n1, n2], BoundaryType::Neumann, 3, 0, 0);

        let dir = tempfile::tempdir()?;
        let path = dir.path().join("room.mesh");
        let summary = write_freefem_medit_mesh(&mesh, &path)?;
        let text = fs::read_to_string(path)?;

        assert_eq!(summary.nodes, 4);
        assert_eq!(summary.tetrahedra, 1);
        assert_eq!(summary.boundary_triangles, 1);
        assert!(text.contains("MeshVersionFormatted 1"));
        assert!(text.contains("Vertices\n4"));
        assert!(text.contains("Triangles\n1"));
        assert!(text.contains("Tetrahedra\n1"));
        assert!(text.contains("1 2 3 4 1"));
        Ok(())
    }

    #[test]
    fn identical_outputs_have_zero_error_metrics() {
        let out = SimulationOutput::solved(
            vec![30.0],
            vec![vec![vec![Complex64::new(1.0, 1.0)]]],
            vec!["s".to_string()],
        );
        let m = compare_outputs(&out, &out).expect("identical outputs compare");
        assert_eq!(m.max_abs, 0.0);
        assert_eq!(m.l2_rel, 0.0);
        assert_eq!(m.max_spl_db, 0.0);
    }

    #[test]
    fn gaussian_expression_includes_near_wall_image() {
        let expr = gaussian_expression(
            Point3D::new(0.02, 0.5, 0.5),
            ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            fem_runner::DEFAULT_SOURCE_WIDTH,
        );
        assert!(expr.contains("x-(-2.00000000000000004e-2)"));
    }
}
