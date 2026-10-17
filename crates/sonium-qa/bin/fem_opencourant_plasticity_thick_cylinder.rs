//! External reference health and native parity gate for plasticity-thick-cylinder.
//!
//! Builds the `plasticity_thick_cylinder` OpenCourant case (ring mesh, bore pressure load-
//! unload history, analytic probes), verifies the checked-in decks match the `sonium_qa::rad`
//! writer, then runs the starter and engine when `OPENCOURANT_EXE` (or legacy `RADISS_EXE`) is
//! set.
//!
//! ACTIVATION STATUS (Sync 2): fully activated on the extended writer. The case uses an
//! elastic-perfectly plastic steel `/MAT/LAW36` card and `/TH/BRIC` stress output with group-
//! qualified `SY/BRIC/<id>` residual channels (probe bricks sit on the +X axis, where hoop is
//! Y). The elastic displacement probes run on the loading leg; the hoop residuals are read at
//! final time.
//!
//! Missing references emit a structured optional skip, or fail when
//! SONIUM_QA_REQUIRE_OPENCOURANT=1. External probe health is recorded separately; release
//! success also requires a matching native solve and full named phenomenon.

// Rust guideline compliant 2026-02-21

use sonium_qa::rad::{
    OpenCourantCase, RadBoundaryCondition, RadConcentratedLoad, RadElasticMaterial,
    RadExtendedCase, RadFunction, RadMaterial, RadMesh, RadNodeSet, RadProbe, RadThRequest,
    assert_probe_close, assert_probes_payload_eq, bric_channel, engine_exe, extended_probes_json,
    perfectly_plastic_steel_law36, run_extended_case, write_extended_engine,
    write_extended_starter,
};
use std::path::PathBuf;

#[path = "support/plasticity_native.rs"]
mod plasticity_native;

const A_M: f64 = 0.05;
const B_M: f64 = 0.10;
const LENGTH_M: f64 = 0.01;
const E_PA: f64 = 200.0e9;
const NU: f64 = 0.3;
const RHO_KG_M3: f64 = 7800.0;
const SIGMA_Y_PA: f64 = 250.0e6;
const P_STAR_PA: f64 = 150.0e6;
const LOAD_TIME_S: f64 = 0.001;
const UNLOAD_END_S: f64 = 0.002;
const STOP_TIME_S: f64 = 0.003;
/// Load level of the elastic-regime displacement probes (below first yield).
const ELASTIC_PROBE_PA: f64 = 50.0e6;
/// Residuals average the last samples over this hold window (5 ring periods).
const RESIDUAL_AVG_SAMPLES: usize = 50;
const NR: usize = 12;
const NTHETA: usize = 24;
const NZ: usize = 1;

/// Brick id of the outer probe brick (ir = nr-1, it = 0, iz = 0).
fn outer_brick_id() -> usize {
    (NR - 1) * NTHETA * NZ + 1
}

/// Lame radial displacement under internal pressure (plane strain).
fn lame_displacement(p: f64, r: f64) -> f64 {
    let amp = p * A_M * A_M / (E_PA * (B_M * B_M - A_M * A_M));
    amp * ((1.0 - NU) * r + (1.0 + NU) * B_M * B_M / r)
}

/// Elastic-plastic interface radius at overload (bisection, plane strain).
fn interface_radius() -> f64 {
    let sqrt3 = 3.0_f64.sqrt();
    let target = P_STAR_PA * sqrt3 / (2.0 * SIGMA_Y_PA);
    let (mut lo, mut hi) = (A_M, B_M);
    for _ in 0..300 {
        let mid = 0.5 * (lo + hi);
        let g = (mid / A_M).ln() + (B_M * B_M - mid * mid) / (2.0 * B_M * B_M) - target;
        if g < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Analytic radial residual after elastic unloading of the overload.
fn radial_residual(r: f64) -> f64 {
    let sqrt3 = 3.0_f64.sqrt();
    let flow = 2.0 * SIGMA_Y_PA / sqrt3;
    let c = interface_radius();
    let loaded = if r <= c {
        -P_STAR_PA + flow * (r / A_M).ln()
    } else {
        let qc = P_STAR_PA - flow * (c / A_M).ln();
        let amp = qc * c * c / (B_M * B_M - c * c);
        amp * (1.0 - B_M * B_M / (r * r))
    };
    let amp = P_STAR_PA * A_M * A_M / (B_M * B_M - A_M * A_M);
    loaded - amp * (1.0 - B_M * B_M / (r * r))
}

/// Analytic SY averaged over a probe brick span.
///
/// The probe bricks sit at polar centroid `phi` off the +X axis, so the
/// global SY mixes hoop and radial: `SY = hoop cos^2 + rad sin^2`.
/// Brick TH output is an element average, so the expectation integrates
/// the same mix over `[r0, r1]` (Simpson, 40 intervals).
fn hoop_sy_brick_avg(r0: f64, r1: f64, phi_rad: f64) -> f64 {
    let (s, c) = (phi_rad.sin(), phi_rad.cos());
    let mix = |r: f64| hoop_residual(r) * c * c + radial_residual(r) * s * s;
    let n = 40;
    let h = (r1 - r0) / n as f64;
    let mut sum = mix(r0) + mix(r1);
    for i in 1..n {
        sum += mix(r0 + h * i as f64) * if i % 2 == 0 { 2.0 } else { 4.0 };
    }
    sum * h / (3.0 * (r1 - r0))
}

/// Smooth load-unload-hold curve: raised cosines (no slope corners, which
/// ring the 50 kHz radial mode) plus a zero hold for residual averaging.
fn smooth_load_points() -> Vec<(f64, f64)> {
    let mut points = Vec::new();
    let up = |t: f64| 0.5 * (1.0 - (std::f64::consts::PI * t / LOAD_TIME_S).cos());
    for i in 0..=20 {
        let t = LOAD_TIME_S * i as f64 / 20.0;
        points.push((t, up(t)));
    }
    for i in 1..=20 {
        let t = LOAD_TIME_S + (UNLOAD_END_S - LOAD_TIME_S) * i as f64 / 20.0;
        points.push((t, 1.0 - up(t - LOAD_TIME_S)));
    }
    points.push((STOP_TIME_S, 0.0));
    points
}

/// Analytic hoop residual after elastic unloading of the overload.
fn hoop_residual(r: f64) -> f64 {
    let sqrt3 = 3.0_f64.sqrt();
    let flow = 2.0 * SIGMA_Y_PA / sqrt3;
    let c = interface_radius();
    let loaded = if r <= c {
        -P_STAR_PA + flow * (r / A_M).ln() + flow
    } else {
        let qc = P_STAR_PA - flow * (c / A_M).ln();
        let amp = qc * c * c / (B_M * B_M - c * c);
        amp * (1.0 + B_M * B_M / (r * r))
    };
    let amp = P_STAR_PA * A_M * A_M / (B_M * B_M - A_M * A_M);
    loaded - amp * (1.0 + B_M * B_M / (r * r))
}

/// Mesh a hollow cylinder with 8-node bricks (periodic in theta).
///
/// Returns the mesh, cap node sets, bore node ids with polar angles, the
/// probe bore/outer node ids at theta = 0, and the probe brick ids.
#[allow(clippy::type_complexity)]
fn cylinder_mesh(
    nr: usize,
    ntheta: usize,
    nz: usize,
) -> (
    RadMesh,
    Vec<usize>,
    Vec<usize>,
    Vec<(usize, f64, f64)>,
    usize,
    usize,
    usize,
    usize,
) {
    let id = |ir: usize, it: usize, iz: usize| (ir * ntheta + it) * (nz + 1) + iz + 1;
    let mut nodes = Vec::with_capacity((nr + 1) * ntheta * (nz + 1));
    for ir in 0..=nr {
        let r = A_M + (B_M - A_M) * ir as f64 / nr as f64;
        for it in 0..ntheta {
            let theta = 2.0 * std::f64::consts::PI * it as f64 / ntheta as f64;
            for iz in 0..=nz {
                nodes.push([
                    r * theta.cos(),
                    r * theta.sin(),
                    LENGTH_M * iz as f64 / nz as f64,
                ]);
            }
        }
    }
    let mut bricks = Vec::with_capacity(nr * ntheta * nz);
    for ir in 0..nr {
        for it in 0..ntheta {
            let jt = (it + 1) % ntheta;
            for iz in 0..nz {
                bricks.push([
                    id(ir, it, iz),
                    id(ir + 1, it, iz),
                    id(ir + 1, jt, iz),
                    id(ir, jt, iz),
                    id(ir, it, iz + 1),
                    id(ir + 1, it, iz + 1),
                    id(ir + 1, jt, iz + 1),
                    id(ir, jt, iz + 1),
                ]);
            }
        }
    }
    let mut cap_lo = Vec::new();
    let mut cap_hi = Vec::new();
    // Bore nodes with polar angle and axial area weight (cap rows are edges).
    let mut bore = Vec::new();
    for ir in 0..=nr {
        for it in 0..ntheta {
            cap_lo.push(id(ir, it, 0));
            cap_hi.push(id(ir, it, nz));
        }
    }
    for it in 0..ntheta {
        let theta = 2.0 * std::f64::consts::PI * it as f64 / ntheta as f64;
        for iz in 0..=nz {
            let axial = if iz == 0 || iz == nz { 0.5 } else { 1.0 } * LENGTH_M / nz as f64;
            bore.push((id(0, it, iz), theta, axial));
        }
    }
    let bore_node = id(0, 0, 0);
    let outer_node = id(nr, 0, 0);
    let bore_brick = 1;
    let outer_brick = (nr - 1) * ntheta * nz + 1;
    (
        RadMesh { nodes, bricks },
        cap_lo,
        cap_hi,
        bore,
        bore_node,
        outer_node,
        bore_brick,
        outer_brick,
    )
}

/// Signed brick volume via five-tet decomposition (orientation check).
fn brick_volume_signed(mesh: &RadMesh, brick: &[usize; 8]) -> f64 {
    let p = |i: usize| mesh.nodes[i - 1];
    let tet = |a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]| {
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let ad = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
        ((ab[1] * ac[2] - ab[2] * ac[1]) * ad[0]
            + (ab[2] * ac[0] - ab[0] * ac[2]) * ad[1]
            + (ab[0] * ac[1] - ab[1] * ac[0]) * ad[2])
            / 6.0
    };
    let n: Vec<[f64; 3]> = brick.iter().map(|id| p(*id)).collect();
    tet(n[0], n[1], n[3], n[4])
        + tet(n[1], n[2], n[3], n[6])
        + tet(n[1], n[3], n[4], n[6])
        + tet(n[3], n[4], n[6], n[7])
        + tet(n[1], n[4], n[5], n[6])
}

/// Builds the thick-cylinder autofrettage OpenCourant case.
///
/// Ring mesh with z-locked caps (plane-strain-like), bore pressure applied
/// as radial `/CLOAD` nodal forces on a smooth raised-cosine load-unload
/// plus a zero hold, and four probes: two elastic-regime bore/outer
/// displacements plus the bore/outer hoop residual SY brick averages on
/// `/TH/BRIC` channels. Residual expectations integrate the analytic
/// autofrettage solution over each probe brick span.
pub fn plasticity_thick_cylinder_case() -> RadExtendedCase {
    // 12 radial bricks: the 11.6 mm plastic zone holds ~3 bricks, so the
    // bore brick resolves the yield peak (nr = 3 smeared it elastic).
    // 24 sectors keep the probe-brick centroid 7.5 deg off the +X axis; the
    // residual expectation integrates the exact hoop/radial SY mix.
    let (nr, ntheta, nz) = (NR, NTHETA, NZ);
    let (mesh, cap_lo, cap_hi, bore, bore_node, outer_node, bore_brick, outer_brick) =
        cylinder_mesh(nr, ntheta, nz);
    mesh.validate().expect("brick indices must be valid");
    for brick in &mesh.bricks {
        assert!(
            brick_volume_signed(&mesh, brick) > 0.0,
            "inverted brick: {brick:?}"
        );
    }

    // Node sets: caps first (BCS), then one single-node set per bore-node
    // load component (X and Y of the radial force).
    let mut node_sets = vec![
        RadNodeSet {
            name: "cap z=0".to_string(),
            nodes: cap_lo,
        },
        RadNodeSet {
            name: "cap z=L".to_string(),
            nodes: cap_hi,
        },
    ];
    let circumferential = 2.0 * std::f64::consts::PI * A_M / ntheta as f64;
    let mut cloads = Vec::new();
    for (node, theta, axial) in &bore {
        let force = P_STAR_PA * circumferential * axial;
        for (direction, component) in [(1, theta.cos()), (2, theta.sin())] {
            // Skip zero components (theta = 0, pi/2, ...) to keep decks lean.
            if component.abs() < 1e-12 {
                continue;
            }
            node_sets.push(RadNodeSet {
                name: format!("bore {node} dir {direction}"),
                nodes: vec![*node],
            });
            cloads.push(RadConcentratedLoad {
                name: format!("bore pressure node {node} dir {direction}"),
                set: node_sets.len() - 1,
                direction,
                function: 0,
                force_scale_n: force * component,
            });
        }
    }

    // Analytic residual expectations: SY brick averages (plane strain,
    // von Mises, perfectly plastic), single-sourced from geometry/yield.
    let dr = (B_M - A_M) / nr as f64;
    let phi = std::f64::consts::PI / ntheta as f64;
    let bore_hoop_resid = hoop_sy_brick_avg(A_M, A_M + dr, phi);
    let outer_hoop_resid = hoop_sy_brick_avg(B_M - dr, B_M, phi);
    assert!(
        bore_hoop_resid < 0.0 && outer_hoop_resid > 0.0,
        "residual signs must be compressive at bore, tensile outside"
    );

    let base = OpenCourantCase {
        name: "plasticity_thick_cylinder".to_string(),
        title: "sonium_qa rad thick-cylinder autofrettage load-unload".to_string(),
        mesh,
        material: RadElasticMaterial {
            youngs_modulus_pa: E_PA,
            poisson_ratio: NU,
            density_kg_per_m3: RHO_KG_M3,
        },
        node_sets,
        bcs: vec![
            RadBoundaryCondition {
                name: "lock cap z=0 axial".to_string(),
                set: 0,
                lock: [false, false, true, false, false, false],
            },
            RadBoundaryCondition {
                name: "lock cap z=L axial".to_string(),
                set: 1,
                lock: [false, false, true, false, false, false],
            },
        ],
        functions: vec![RadFunction {
            name: "smooth overload 0 -> p* -> 0 + hold".to_string(),
            points: smooth_load_points(),
        }],
        cloads,
        walls: Vec::new(),
        th_nodes: vec![bore_node, outer_node],
        probes: vec![
            RadProbe {
                label: "bore radial DX at 50 MPa".to_string(),
                channel: format!("DX/{bore_node}"),
                expected: lame_displacement(ELASTIC_PROBE_PA, A_M),
                abs_tol: 1e-12,
                // Measured engine deviation = 1.9e-2; gate at ~1.6x.
                rel_tol: 0.03,
            },
            RadProbe {
                label: "outer radial DX at 50 MPa".to_string(),
                channel: format!("DX/{outer_node}"),
                expected: lame_displacement(ELASTIC_PROBE_PA, B_M),
                abs_tol: 1e-12,
                // Measured engine deviation = 4.4e-2; gate at ~1.25x.
                rel_tol: 0.055,
            },
            RadProbe {
                label: "bore hoop residual SY".to_string(),
                channel: bric_channel("SY", bore_brick),
                expected: bore_hoop_resid,
                abs_tol: 1e6,
                // Measured engine deviation beyond abs_tol = 3.4e-2; gate at ~1.3x.
                rel_tol: 0.045,
            },
            RadProbe {
                label: "outer hoop residual SY".to_string(),
                channel: bric_channel("SY", outer_brick),
                expected: outer_hoop_resid,
                abs_tol: 1e6,
                // Measured engine deviation beyond abs_tol = 1.6e-2; gate at ~1.9x.
                rel_tol: 0.03,
            },
        ],
        stop_time_s: STOP_TIME_S,
        th_period_s: 2.0e-6,
        anim_period_s: 3.0e-4,
    };
    let mut extended = RadExtendedCase::wrap_elastic(base);
    extended.material = RadMaterial::Plastic(perfectly_plastic_steel_law36(
        E_PA, NU, RHO_KG_M3, SIGMA_Y_PA,
    ));
    extended.th_brick = RadThRequest::brick_stress(vec![bore_brick, outer_brick]);
    extended
}

fn decks_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("decks/plasticity_thick_cylinder")
}

const CASE_ID: &str = "fem.opencourant.plasticity-thick-cylinder.native-parity.v1";

fn main() {
    let case = plasticity_thick_cylinder_case();
    case.validate()
        .expect("plasticity_thick_cylinder must validate");
    let dir = decks_dir();
    let starter_path = dir.join("plasticity_thick_cylinder_0000.rad");
    let engine_path = dir.join("plasticity_thick_cylinder_0001.rad");
    let probes_path = dir.join("probes.json");

    let starter = write_extended_starter(&case);
    let engine = write_extended_engine(&case);
    // One-time deck generation: the checked-in trio is writer output.
    if std::env::var("REGEN_DECKS").as_deref() == Ok("1") {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&starter_path, &starter).unwrap();
        std::fs::write(&engine_path, &engine).unwrap();
        std::fs::write(
            &probes_path,
            extended_probes_json(&case, "none (prove-out deferred)"),
        )
        .unwrap();
        println!("regenerated decks for plasticity_thick_cylinder");
        return;
    }

    // No hand-built decks: the checked-in trio must match the writer.
    let checked_starter = std::fs::read_to_string(&starter_path).unwrap_or_default();
    let checked_engine = std::fs::read_to_string(&engine_path).unwrap_or_default();
    let checked_probes = std::fs::read_to_string(&probes_path).unwrap_or_default();
    assert_eq!(
        checked_starter, starter,
        "plasticity_thick_cylinder_0000.rad drifted from the rad writer"
    );
    assert_eq!(
        checked_engine, engine,
        "plasticity_thick_cylinder_0001.rad drifted from the rad writer"
    );
    assert_probes_payload_eq(
        &checked_probes,
        &extended_probes_json(&case, "none (prove-out deferred)"),
        "probes.json drifted from the rad writer",
    );
    println!(
        "case {}: {} nodes, {} bricks, {} cloads, {} probes",
        case.base.name,
        case.base.mesh.num_nodes(),
        case.base.mesh.num_bricks(),
        case.base.cloads.len(),
        case.base.probes.len()
    );

    let Some(exe) = engine_exe() else {
        sonium_qa::rad::gate::report_missing_engine(CASE_ID);
    };

    // Run in a scratch dir so engine artifacts never pollute the repo.
    let work = std::env::temp_dir().join(format!(
        "sonium_rad_plasticity_thick_cylinder_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&work).unwrap();
    std::fs::write(work.join("plasticity_thick_cylinder_0000.rad"), &starter).unwrap();
    std::fs::write(work.join("plasticity_thick_cylinder_0001.rad"), &engine).unwrap();
    let output = run_extended_case(&work, &exe, &case).unwrap_or_else(|error| {
        sonium_qa::rad::gate::publish_and_exit(sonium_qa::rad::gate::execution_failure_result(
            CASE_ID,
            &error.to_string(),
        ));
    });
    println!("engine: {}", output.engine);
    // Elastic probes are read on the loading leg where the raised cosine
    // crosses 50 MPa: 50/150 = (1 - cos(pi t/T)) / 2. The outer probe is
    // retarded by one dilatational transit: the bore load reaches r = B
    // 8.5 us late, worth 4% of signal on this ramp (proven by experiment).
    let t_elastic =
        LOAD_TIME_S * (1.0 - 2.0 * ELASTIC_PROBE_PA / P_STAR_PA).acos() / std::f64::consts::PI;
    let lame_lambda = E_PA * NU / ((1.0 + NU) * (1.0 - 2.0 * NU));
    let shear_mu = E_PA / (2.0 * (1.0 + NU));
    let c_dil = ((lame_lambda + 2.0 * shear_mu) / RHO_KG_M3).sqrt();
    let t_outer = t_elastic + (B_M - A_M) / c_dil;
    let mut failed = false;
    for (index, probe) in case.base.probes.iter().enumerate() {
        let actual = if index == 0 {
            output.history.value_at(&probe.channel, t_elastic)
        } else if index == 1 {
            output.history.value_at(&probe.channel, t_outer)
        } else {
            // Residuals average over the zero hold to cancel ringdown.
            output.history.channel(&probe.channel).and_then(|values| {
                let n = values.len().min(RESIDUAL_AVG_SAMPLES);
                if n == 0 {
                    return Err(sonium_qa::rad::RadError::Parse(format!(
                        "channel {} has no samples",
                        probe.channel
                    )));
                }
                Ok(values[values.len() - n..].iter().sum::<f64>() / n as f64)
            })
        };
        match actual {
            Ok(value) => {
                println!(
                    "probe {}: actual {value:.6e}, expected {:.6e}",
                    probe.label, probe.expected
                );
                if let Err(error) = assert_probe_close(
                    value,
                    probe.expected,
                    probe.abs_tol,
                    probe.rel_tol,
                    &probe.label,
                ) {
                    eprintln!("FAIL: {error}");
                    failed = true;
                }
            }
            Err(error) => {
                eprintln!("FAIL: probe {} unreadable: {error}", probe.label);
                failed = true;
            }
        }
    }
    // Plasticity happened where it should: the bore brick yielded, the
    // outer brick (r = 96..100 mm, far outside the 61.6 mm front) did not.
    match (
        output.history.final_value(&bric_channel("PLAS", 1)),
        output
            .history
            .final_value(&bric_channel("PLAS", outer_brick_id())),
    ) {
        (Ok(bore_plas), Ok(outer_plas)) => {
            println!("probe bore PLAS: {bore_plas:.6e}, outer PLAS: {outer_plas:.6e}");
            if bore_plas < 5.0e-4 {
                eprintln!("FAIL: bore brick never yielded (PLAS {bore_plas:.3e})");
                failed = true;
            }
            if outer_plas > 1.0e-9 {
                eprintln!("FAIL: outer brick yielded (PLAS {outer_plas:.3e})");
                failed = true;
            }
        }
        _ => {
            eprintln!("FAIL: PLAS channels unreadable");
            failed = true;
        }
    }
    std::fs::remove_dir_all(&work).ok();
    let result =
        plasticity_native::parity_result(&case, CASE_ID, &output.engine, &output.history, !failed)
            .unwrap_or_else(|error| {
                sonium_qa::rad::gate::execution_failure_result(CASE_ID, &error.to_string())
            });
    sonium_qa::rad::gate::publish_and_exit(result);
}
