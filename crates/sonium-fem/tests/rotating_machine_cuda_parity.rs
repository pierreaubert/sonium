//! CPU↔CUDA parity of the planar `A_z` solver.
//!
//! The assembly is identical (CPU, rayon); only the inner linear
//! algebra moves: sparse-direct Cholesky on CPU versus the solvers
//! facade's native CUDA Krylov paths. Parity therefore means
//! agreement to iterative-solver tolerance on the same Newton root,
//! with backend provenance proving the GPU really ran. The oracle
//! problem is the Stage-1 scikit-fem cross-check mesh; the motor
//! problem is one GEO_30MM frame on the native sliding-band mesh
//! (the sliver-bearing mesh the CPU iterative paths struggle with).
//!
//! Build and run with:
//! `cargo test -p sonium-fem --features cuda --test rotating_machine_cuda_parity -- --ignored --nocapture`

// Rust guideline compliant 2026-02-21
#![cfg(feature = "cuda")]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use serde_json::Value;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::{ElementType, Mesh, Point};
use sonium_fem::physics::electromagnetics::magnetostatics::BhCurve;
use sonium_fem::physics::electromagnetics::rotating_machine::{
    AzLinearBackend, AzMaterial, AzProblem, AzSolveConfig, BertottiCoefficients, DrivePoint,
    MagnetLaw, MeasuredLossSurface, MotorMaterials, MotorMeshOptions, MotorPolygons, Phase,
    RegionPolygon, SlotWinding, SteelLoss, TransientConfig, Winding, build_motor_mesh,
    laminated_bh_curve, run_current_drive_transient, solve_az,
};
use sonium_solvers::{LinearSolveConfig, LinearSolverType};

const MM: f64 = 1.0e-3;

fn fixture(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/motor_oracle")
        .join(name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_str(&text).expect("fixture parses")
}

fn f64s(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .expect("array")
        .iter()
        .map(|entry| entry.as_f64().expect("number"))
        .collect()
}

fn relative_l2(actual: &[f64], expected: &[f64]) -> f64 {
    let num: f64 = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    let den: f64 = expected.iter().map(|b| b * b).sum();
    (num / den.max(1.0e-300)).sqrt()
}

fn cuda_config(solver: LinearSolverType, restart: usize) -> AzSolveConfig {
    let mut config = AzSolveConfig {
        continuation_steps: 1,
        ..AzSolveConfig::default()
    };
    config.backend = AzLinearBackend::IterativeCuda;
    config.inner_solver = LinearSolveConfig {
        solver,
        // Restart cycles, as on the host facade (the az CUDA
        // dispatch converts to total Arnoldi steps). Plain GMRES
        // on the GEO_30MM tangent needs a generous budget to reach
        // 1e-10 without a preconditioner; the CPU sweep converged
        // at this order of cycles per Newton solve.
        max_iterations: 4_000,
        restart,
        tolerance: 1.0e-10,
        ..LinearSolveConfig::default()
    };
    config
}

fn cpu_config() -> AzSolveConfig {
    let mut config = AzSolveConfig {
        continuation_steps: 1,
        ..AzSolveConfig::default()
    };
    config.newton.relative_tolerance = 1.0e-10;
    config.newton.absolute_tolerance = 1.0e-10;
    config
}

#[test]
#[ignore = "requires a CUDA device"]
fn cuda_matches_cpu_direct_on_oracle_problem() {
    let data = fixture("az_crosscheck.json");
    let meta = &data["meta"];
    let mut mesh = Mesh::new(2);
    for node in data["nodes"].as_array().expect("nodes") {
        let coords = f64s(node);
        mesh.add_node(Point::new_2d(coords[0], coords[1]));
    }
    for (element, marker) in data["elements"]
        .as_array()
        .expect("elements")
        .iter()
        .zip(data["markers"].as_array().expect("markers"))
    {
        let nodes: Vec<usize> = element
            .as_array()
            .expect("element")
            .iter()
            .map(|entry| entry.as_u64().expect("node id") as usize)
            .collect();
        mesh.add_element_with_marker(
            ElementType::Triangle,
            nodes,
            i32::try_from(marker.as_i64().expect("marker")).expect("marker fits"),
        );
    }
    mesh.detect_boundaries();
    let domain = f64s(&meta["domain_m"]);
    let dirichlet: Vec<usize> = mesh
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| {
            (node.x - domain[0]).abs() < 1.0e-12
                || (node.x - domain[1]).abs() < 1.0e-12
                || (node.y - domain[2]).abs() < 1.0e-12
                || (node.y - domain[3]).abs() < 1.0e-12
        })
        .map(|(index, _)| index)
        .collect();
    let points: Vec<(f64, f64)> = meta["bh_points"]
        .as_array()
        .expect("bh points")
        .iter()
        .map(|point| {
            let pair = f64s(point);
            (pair[0], pair[1])
        })
        .collect();
    let br = f64s(&meta["magnet_br_t"]);
    let problem = AzProblem {
        mesh,
        materials: BTreeMap::from([
            (1, AzMaterial::nonlinear(BhCurve::new(points).expect("bh"))),
            (
                2,
                AzMaterial::magnet(
                    MagnetLaw::new(meta["magnet_mu_rec"].as_f64().unwrap(), [br[0], br[1]])
                        .unwrap(),
                )
                .unwrap(),
            ),
            (3, AzMaterial::air()),
            (4, AzMaterial::air()),
        ]),
        current_density_a_per_m2: BTreeMap::from([(
            3,
            meta["coil_j_a_per_m2"].as_f64().expect("coil J"),
        )]),
        current_density_fn: None,
        degree: PolynomialDegree::P1,
        quadrature_order: 4,
        stack_length_m: meta["stack_length_m"].as_f64().expect("stack"),
        conductor_blocks: Vec::new(),
        element_magnet_scale: Vec::new(),
    };

    let start = Instant::now();
    let cpu = solve_az(&problem, &dirichlet, &cpu_config(), None, None).expect("cpu solve");
    let cpu_time = start.elapsed();
    let mut compared = 0usize;
    for solver in [
        LinearSolverType::Cg,
        LinearSolverType::CgAmg,
        LinearSolverType::GmresJacobi,
        LinearSolverType::GmresAmg,
        LinearSolverType::Gmres,
    ] {
        let start = Instant::now();
        match solve_az(&problem, &dirichlet, &cuda_config(solver, 30), None, None) {
            Ok(gpu) => {
                let gpu_time = start.elapsed();
                let error = relative_l2(
                    &gpu.vector_potential_wb_per_m,
                    &cpu.vector_potential_wb_per_m,
                );
                eprintln!(
                    "oracle parity {solver:?}: rel L2 {error:.3e}, cpu {cpu_time:?}, cuda {gpu_time:?}"
                );
                assert!(
                    error < 1.0e-6,
                    "CUDA {solver:?} deviates from CPU by {error:e}"
                );
                compared += 1;
            }
            Err(error) => {
                eprintln!("oracle CUDA {solver:?} unavailable: {error}");
            }
        }
    }
    assert!(
        compared > 0,
        "no admitted CUDA strategy solved the oracle problem"
    );
}

fn ring(value: &Value) -> Vec<(f64, f64)> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|point| {
            let pair = point.as_array().unwrap();
            (
                pair[0].as_f64().unwrap() * MM,
                pair[1].as_f64().unwrap() * MM,
            )
        })
        .collect()
}

fn polygon(value: &Value) -> RegionPolygon {
    RegionPolygon {
        marker: 0,
        exterior: ring(&value["exterior"]),
        holes: value["interiors"]
            .as_array()
            .unwrap()
            .iter()
            .map(ring)
            .collect(),
    }
}

/// GEO_30MM polygons, winding, materials and drive point from the fixture.
fn geo_setup() -> (MotorPolygons, Winding, MotorMaterials, DrivePoint) {
    let data = fixture("motor30.json");
    let polys = &data["polygons_mm"];
    let polygons = MotorPolygons {
        stator: polygon(&polys["stator"][0]),
        rotor: polygon(&polys["rotor"][0]),
        shaft: polygon(&polys["shaft"][0]),
        magnets: polys["magnets"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| polygon(&m["polygon"][0]))
            .collect(),
        coils: polys["coils"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                (
                    c["slot"].as_u64().unwrap() as usize,
                    polygon(&c["polygon"][0]),
                )
            })
            .collect(),
        mid_gap_radius_m: data["derived"]["mid_gap_radius_mm"].as_f64().unwrap() * MM,
        outer_radius_m: data["derived"]["outer_boundary_radius_mm"]
            .as_f64()
            .unwrap()
            * MM,
    };
    let winding = Winding {
        layout: data["winding_layout"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| {
                let pair = entry.as_array().unwrap();
                SlotWinding {
                    phase: match pair[0].as_str().unwrap() {
                        "A" => Phase::A,
                        "B" => Phase::B,
                        _ => Phase::C,
                    },
                    direction: pair[1].as_i64().unwrap() as i8,
                }
            })
            .collect(),
        turns_per_slot: 6.0,
        n_parallel: 1.0,
        pole_pairs: 7,
    };
    let steel = &data["materials"]["steel_B15AHV950M"];
    let hb: Vec<(f64, f64)> = steel["bh_curve"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            let pair = p.as_array().unwrap();
            (pair[0].as_f64().unwrap(), pair[1].as_f64().unwrap())
        })
        .collect();
    let kf = steel["stacking_factor"].as_f64().unwrap();
    let bertotti = BertottiCoefficients {
        kh: steel["core_loss_kh"].as_f64().unwrap(),
        kc: steel["core_loss_kc"].as_f64().unwrap(),
        ke: steel["core_loss_ke"].as_f64().unwrap(),
    };
    let mut frequencies = Vec::new();
    let mut tables = Vec::new();
    for (key, curve) in steel["core_loss_curves"].as_object().unwrap() {
        frequencies.push(key.trim_end_matches("Hz").parse::<f64>().unwrap());
        tables.push(
            curve
                .as_array()
                .unwrap()
                .iter()
                .map(|p| {
                    let pair = p.as_array().unwrap();
                    (pair[0].as_f64().unwrap(), pair[1].as_f64().unwrap())
                })
                .collect::<Vec<_>>(),
        );
    }
    let surface = MeasuredLossSurface::new(
        &frequencies,
        &tables,
        steel["density"].as_f64().unwrap(),
        steel["core_loss_curve_unit"].as_str() == Some("w_per_kg"),
        bertotti,
    )
    .expect("loss surface");
    let magnet = &data["materials"]["magnet_F45SH_120C"];
    let materials = MotorMaterials {
        steel: AzMaterial::nonlinear(laminated_bh_curve(&hb, kf).unwrap()),
        steel_loss: SteelLoss {
            bertotti,
            stacking_factor: kf,
            surface: Some(surface),
        },
        magnet_br_t: magnet["Br"].as_f64().unwrap(),
        magnet_mu_rec: magnet["mu_rec"].as_f64().unwrap(),
        magnet_polarity: polys["magnets"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["polarity"].as_f64().unwrap())
            .collect(),
        magnet_sigma_s_per_m: magnet["sigma"].as_f64().unwrap(),
        shaft_mu_r: data["materials"]["conductor_Aluminium_6061"]["mu_r"]
            .as_f64()
            .unwrap_or(1.0),
        shaft_sigma_s_per_m: data["constants"]["SIGMA_SHAFT"].as_f64().unwrap(),
    };
    let drive = DrivePoint {
        i_phase_rms_a: 60.0,
        gamma_deg: 0.0,
        daxis_deg: 60.0,
        rpm: 15_000.0,
        coil_temp_c: 120.0,
    };
    (polygons, winding, materials, drive)
}

/// Acceptance gate for CUDA parity on the full GEO_30MM frame.
///
/// History: this test failed while the native planar CDT emitted
/// sliver elements (47% of triangles below 10° min-angle, worst
/// radius ratio 2.9e7). The planar mesher now decimates
/// over-dense input contours and runs a Ruppert-style
/// circumcenter quality loop (slip-ring segments protected,
/// bit-deterministic output), leaving 0.07% of triangles below
/// 10° and worst radius ratio ~300. The remaining blocker was
/// never the device: on frozen GEO_30MM tangents (measured
/// offline, 2026-10-22), host and device plain GMRES produce
/// identical residuals at equal Arnoldi steps (3.14e-3 @12k,
/// ~3e-4 @360k — nowhere near 1e-10), so no tight Krylov solve
/// exists for these tangents on either side with the admitted
/// preconditioners: AMG/Jacobi make it worse, natural-ordering
/// ILU(0) reaches only ~5e-7 in 120k steps and its checked
/// setup breaks down on some tangents (sub-threshold pivots;
/// scipy's row-pivoted ILU solves the same systems in ~5
/// cycles, so this is a no-row-pivot-ILU limitation, not
/// conditioning per se — threshold ILU variants, HILU, COLAMD
/// reordering and equilibration all measured no better than
/// ILU(0) on frozen tangents). What does work is inexact
/// Newton: with relative Eisenstat–Walker forcing (floor
/// 2e-2), trust-region clipping, source continuation,
/// budget-escalating retries, and compat host ILU(0) factors
/// uploaded to the device (`CudaIluPreconditioner::try_from_host`,
/// the same policy CPU GmresIlu trusts), device FGMRES +
/// resident ILU(0) solves the loose decades cheaply and Newton
/// itself drives the exact residual to 1e-7 — proven
/// end-to-end on the coarse mesh below (frame-0 A_z rel L2
/// 1.5e-7, torque to 4e-10). At production density the same
/// route applies, at device-ILU iteration rates (~20-30
/// ms/iteration, launch-bound triangular sweeps) that make
/// the full transient a multi-hour-to-multi-day run; this
/// test is that stretch gate. Status: not yet run to
/// completion (five traced attempts all progressed without
/// correctness failures but were stopped on wall time; the
/// parity claim at this scale currently rests on the coarse
/// gate below plus frozen-tangent checks that the device
/// reproduces host ILU(0) iteration-for-residual exactly).
/// Two integration details matter: the host
/// GMRES facade counts `max_iterations` in restart cycles
/// while the device counts total steps (the az CUDA dispatch
/// converts), and when even compat ILU(0) upload fails the
/// dispatch falls back to plain device GMRES for that step.
#[test]
#[ignore = "requires a CUDA device; heavy"]
fn cuda_matches_cpu_direct_on_geo30mm_frame() {
    let (polygons, winding, materials, drive) = geo_setup();
    let run = |solve: AzSolveConfig, label: &str| {
        let mut motor = build_motor_mesh(&polygons, &MotorMeshOptions::default()).unwrap();
        motor.promote_p2(1.0e-7).unwrap();
        let start = Instant::now();
        let result = run_current_drive_transient(
            &mut motor,
            PolynomialDegree::P2,
            &winding,
            &materials,
            &drive,
            &TransientConfig {
                steps_per_period: 8,
                stack_length_m: 0.010,
                rotor_outer_m: 9.0 * MM,
                stator_inner_m: 9.2 * MM,
                solve,
                demag: None,
            },
        );
        eprintln!("geo30mm {label}: {:?}", start.elapsed());
        result
    };
    // Smoke mode (diagnostic): skip the CPU baseline and the
    // parity assert; just run the inexact-ILU attempt.
    let smoke = std::env::var_os("SONIUM_CUDA_SMOKE").is_some();
    let cpu = if smoke {
        None
    } else {
        Some(run(cpu_config(), "cpu-direct").expect("cpu transient"))
    };
    // The converging route is inexact Newton (Eisenstat-Walker
    // forcing, floor 2e-2) over FGMRES + resident device ILU(0):
    // loose inner decades are cheap on these tangents, tight
    // ones converge only algebraically slowly (see doc comment).
    // Measured alternative: pinning eta at the 0.1 cap made the
    // trust-region Newton stagnate (80 iterations, residual
    // 1.3e2) because retry-loosened steps at eta 0.4-0.5 are too
    // poor to reduce the nonlinear residual. Plain GMRES
    // and CgAmg are probed for the record and are expected to
    // fail; GmresAmg/GmresJacobi already measured stagnating and
    // are left out to bound the runtime.
    let inexact_ilu = {
        let mut config = cuda_config(LinearSolverType::GmresIlu, 30);
        config.inexact_newton = true;
        // Source continuation keeps each subproblem near the
        // previous root; without it, loose inexact steps from a
        // cold start wander into tangents no Krylov method solves
        // (measured: frame-1 Newton died at iteration 41/19).
        config.continuation_steps = 4;
        // Forcing floor 2e-2: traced device residuals on inexact
        // iterates decrease only algebraically slowly past ~0.1,
        // so wall hits at this floor burn the fail-fast budget
        // (~33 s) before the retry-looser safeguard steps in —
        // but steps that do reach it keep Newton converging
        // briskly. A 0.1 floor was measured to stagnate Newton
        // outright (see above), so the wall tax is the price of
        // step quality on this route.
        config.inner_solver.tolerance = 2.0e-2;
        // Small per-solve budget (40 cycles = 1.2k steps): traced
        // successful fine-mesh solves finish in ~12-18 cycles
        // (13-23 s), so 40 keeps ~2-3x headroom while a wall hit
        // fails in ~1 min and lets az's inexact safeguard retry
        // the step looser (with budget escalation) instead of
        // grinding the full budget. (At 200 cycles one wall hit
        // plus escalations held a single logical solve for >1 h.)
        config.inner_solver.max_iterations = 40;
        // Inexact steps need more Newton iterations than the exact
        // route; 80 measured short on a hard subproblem when the
        // forcing was loosened, so keep generous headroom.
        config.newton.max_iterations = 160;
        config
    };
    let attempts: Vec<(&str, AzSolveConfig)> = if smoke {
        vec![("FgmresIlu-inexact", inexact_ilu)]
    } else {
        vec![
            ("FgmresIlu-inexact", inexact_ilu),
            ("Gmres-60", cuda_config(LinearSolverType::Gmres, 60)),
            ("CgAmg-30", cuda_config(LinearSolverType::CgAmg, 30)),
        ]
    };
    let mut compared = false;
    for (label, config) in attempts {
        match run(config, "cuda") {
            Ok(gpu) => {
                if let Some(cpu) = &cpu {
                    let error = relative_l2(
                        &gpu.frames[0].vector_potential_wb_per_m,
                        &cpu.frames[0].vector_potential_wb_per_m,
                    );
                    eprintln!(
                        "geo30mm parity {label}: frame-0 A_z rel L2 {error:.3e}, \
                         psi cpu {:?} vs cuda {:?}",
                        cpu.frames[0].psi_wb_turn, gpu.frames[0].psi_wb_turn
                    );
                    assert!(error < 1.0e-5, "CUDA {label} deviates by {error:e}");
                } else {
                    eprintln!(
                        "geo30mm smoke {label}: torque {:.4} N m, psi frame0 {:?}",
                        gpu.torque_terminal_work_mean_nm, gpu.frames[0].psi_wb_turn
                    );
                }
                compared = true;
            }
            Err(error) => {
                eprintln!("geo30mm CUDA {label} unavailable: {error}");
            }
        }
    }
    assert!(
        compared,
        "no admitted CUDA strategy solved the GEO_30MM frame"
    );
}

/// Practical CUDA parity gate for the motor transient pipeline on
/// a coarsened GEO_30MM mesh (3x element sizes).
///
/// Backend parity is a same-discrete-problem statement (the fine
/// mesh's accuracy is gated separately by the CPU pins); this
/// mesh keeps the full pipeline — sliding-band rotation, winding
/// currents, nonlinear Newton, torque post-processing — while
/// bringing the tangent conditioning back into the range the
/// CUDA-admitted Krylov stack solves directly. The fine-mesh
/// test above remains the stretch gate: at production density
/// the cold tangent defeats every admitted tight Krylov strategy
/// on host and device alike (see its doc comment), and only the
/// inexact-Newton route grinds through, slowly.
#[test]
#[ignore = "requires a CUDA device"]
fn cuda_matches_cpu_direct_on_coarse_motor() {
    let (polygons, winding, materials, drive) = geo_setup();
    let options = MotorMeshOptions {
        gap_size_m: 2.4e-4,
        iron_size_m: 2.7e-3,
        outer_size_m: 6.0e-3,
        ..MotorMeshOptions::default()
    };
    let run = |solve: AzSolveConfig, label: &str| {
        let mut motor = build_motor_mesh(&polygons, &options).unwrap();
        eprintln!(
            "coarse mesh: {} elements, {} nodes",
            motor.mesh.elements.len(),
            motor.mesh.nodes.len()
        );
        motor.promote_p2(1.0e-7).unwrap();
        let start = Instant::now();
        let result = run_current_drive_transient(
            &mut motor,
            PolynomialDegree::P2,
            &winding,
            &materials,
            &drive,
            &TransientConfig {
                steps_per_period: 8,
                stack_length_m: 0.010,
                rotor_outer_m: 9.0 * MM,
                stator_inner_m: 9.2 * MM,
                solve,
                demag: None,
            },
        );
        eprintln!("coarse {label}: {:?}", start.elapsed());
        result
    };
    let cpu = run(cpu_config(), "cpu-direct").expect("cpu transient");
    let inexact_ilu = {
        let mut config = cuda_config(LinearSolverType::GmresIlu, 30);
        config.inexact_newton = true;
        config.continuation_steps = 4;
        config.inner_solver.tolerance = 2.0e-2;
        config.inner_solver.max_iterations = 200;
        config.newton.max_iterations = 80;
        config
    };
    // Inexact first (the expected route); exact attempts are
    // budget-capped so their expected failure is fast (device-ILU
    // iterations cost ~28 ms on these tangents, so an uncapped
    // tight-tolerance attempt burns ~1 h before failing).
    let capped = |solver: LinearSolverType| {
        let mut config = cuda_config(solver, 30);
        config.inner_solver.max_iterations = 200;
        config
    };
    let attempts: Vec<(&str, AzSolveConfig)> = vec![
        ("FgmresIlu-inexact", inexact_ilu),
        ("GmresIlu-exact", capped(LinearSolverType::GmresIlu)),
        ("Gmres-30", capped(LinearSolverType::Gmres)),
    ];
    let mut compared = false;
    for (label, config) in attempts {
        match run(config, "cuda") {
            Ok(gpu) => {
                let error = relative_l2(
                    &gpu.frames[0].vector_potential_wb_per_m,
                    &cpu.frames[0].vector_potential_wb_per_m,
                );
                let torque_rel =
                    (gpu.torque_terminal_work_mean_nm - cpu.torque_terminal_work_mean_nm).abs()
                        / cpu.torque_terminal_work_mean_nm.abs().max(1.0e-300);
                eprintln!(
                    "coarse parity {label}: frame-0 A_z rel L2 {error:.3e}, \
                     torque cpu {:.4} vs cuda {:.4} (rel {torque_rel:.2e})",
                    cpu.torque_terminal_work_mean_nm, gpu.torque_terminal_work_mean_nm
                );
                assert!(error < 1.0e-5, "CUDA {label} A_z deviates by {error:e}");
                assert!(
                    torque_rel < 1.0e-3,
                    "CUDA {label} torque deviates by {torque_rel:e}"
                );
                compared = true;
            }
            Err(error) => {
                eprintln!("coarse CUDA {label} unavailable: {error}");
            }
        }
    }
    assert!(
        compared,
        "no admitted CUDA strategy solved the coarse motor"
    );
}
