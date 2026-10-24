//! GEO_30MM current-drive transient against the reference solver pins.
//!
//! Runs a no-load sweep to calibrate the d-axis, then the pinned
//! `p2_load` operating point (60 A rms, gamma 0, 15000 rpm, 12
//! steps/period, P2) on the natively meshed full-360 sliding-band
//! mesh. The reference numbers come from
//! eMotres/Motor_Ai_Simulation `tests/physics_baseline.json`, whose own
//! regression tolerance is 0.5 % on torque/voltage (its mesh is a
//! 180-degree sector; this test's full mesh is independent, so the
//! comparison bands assert mesh-converged agreement, not identity).
//!
//! Heavy (24 nonlinear P2 solves): run with `--ignored`.

// Rust guideline compliant 2026-02-21

use std::path::PathBuf;

use serde_json::Value;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::physics::electromagnetics::rotating_machine::{
    AzMaterial, AzSolveConfig, BertottiCoefficients, DemagConfig, DrivePoint, EddyTransientConfig,
    MagnetDemagCurve, MeasuredLossSurface, MotorMaterials, MotorMeshOptions, MotorPolygons, Phase,
    RegionPolygon, SlotWinding, SteelLoss, TransientConfig, VoltageDrivePoint,
    VoltageTransientConfig, VoltageWaveform, Winding, build_motor_mesh, calibrate_daxis_deg,
    laminated_bh_curve, run_current_drive_transient, run_eddy_current_drive,
    run_voltage_drive_transient,
};

const MM: f64 = 1.0e-3;

fn fixture() -> Value {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/motor_oracle/motor30.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_str(&text).expect("motor fixture parses")
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

fn setup(data: &Value) -> (MotorPolygons, Winding, MotorMaterials) {
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
    let layout: Vec<SlotWinding> = data["winding_layout"]
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
        .collect();
    let winding = Winding {
        layout,
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
    (polygons, winding, materials)
}

#[test]
#[ignore = "heavy: 24 nonlinear P2 solves on the full motor mesh"]
fn p2_load_matches_reference_pins() {
    let data = fixture();
    let (polygons, winding, materials) = setup(&data);
    let mut motor = build_motor_mesh(&polygons, &MotorMeshOptions::default()).unwrap();
    motor.promote_p2(1.0e-7).unwrap();
    let config = TransientConfig {
        steps_per_period: 12,
        stack_length_m: 0.010,
        rotor_outer_m: 9.0 * MM,
        stator_inner_m: 9.2 * MM,
        // Warm-started frames converge without source continuation;
        // the cold first frame took 13 Newton iterations in probing.
        solve: AzSolveConfig {
            continuation_steps: 1,
            ..AzSolveConfig::default()
        },
        demag: None,
    };
    // No-load sweep -> d-axis calibration.
    let noload = run_current_drive_transient(
        &mut motor,
        PolynomialDegree::P2,
        &winding,
        &materials,
        &DrivePoint {
            i_phase_rms_a: 0.0,
            gamma_deg: 0.0,
            daxis_deg: 0.0,
            rpm: 15_000.0,
            coil_temp_c: 120.0,
        },
        &config,
    )
    .expect("no-load sweep");
    let angles: Vec<f64> = noload.frames.iter().map(|f| f.rotor_angle_deg).collect();
    let psi_a: Vec<f64> = noload.frames.iter().map(|f| f.psi_wb_turn[0]).collect();
    let daxis = calibrate_daxis_deg(&angles, &psi_a, 7).unwrap();
    eprintln!("calibrated daxis = {daxis:.3} deg (expect 60 for 12s14p)");

    let start = std::time::Instant::now();
    let loaded = run_current_drive_transient(
        &mut motor,
        PolynomialDegree::P2,
        &winding,
        &materials,
        &DrivePoint {
            i_phase_rms_a: 60.0,
            gamma_deg: 0.0,
            daxis_deg: daxis,
            rpm: 15_000.0,
            coil_temp_c: 120.0,
        },
        &config,
    )
    .expect("loaded transient");
    let pins = &data["pins"]["p2_load"];
    let pin = |key: &str| pins[key].as_f64().unwrap();
    let t_flux = loaded.torque_terminal_work_mean_nm;
    // Dump terminal series for offline estimator comparison.
    {
        let dump = serde_json::json!({
            "angles_deg": loaded.frames.iter().map(|f| f.rotor_angle_deg).collect::<Vec<_>>(),
            "psi": (0..3).map(|ph| loaded.frames.iter().map(|f| f.psi_wb_turn[ph]).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "currents": (0..3).map(|ph| loaded.frames.iter().map(|f| f.currents_a[ph]).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "torque_arkkio": loaded.torque_arkkio_nm,
            "torque_terminal_work": loaded.torque_terminal_work_mean_nm,
            "torque_space_vector": loaded.torque_space_vector_mean_nm,
            "torque_flux_discrete": loaded.torque_flux_linkage_mean_nm,
        });
        std::fs::write("/tmp/motor_transient_series.json", dump.to_string()).unwrap();
    }
    let t_arkkio = loaded.torque_arkkio_metrics.mean_nm;
    let p_fe = loaded.stator_iron_loss.total_w + loaded.rotor_iron_loss.total_w;
    let psi_amp = loaded
        .frames
        .iter()
        .map(|f| f.psi_wb_turn[0].abs())
        .fold(0.0f64, f64::max);
    let noload_psi_amp = psi_a.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    eprintln!(
        "loaded run {:?}: T_flux={t_flux:.4} (pin {:.4}) T_arkkio={t_arkkio:.4} \
         P_fe={p_fe:.3} (pin {:.3}) ripple_arkkio={:.2}% daxis={daxis:.3} \
         psiA_amp_loaded={psi_amp:.5} psiA_amp_noload={noload_psi_amp:.5} \
         P_fe_stator={:.3} P_fe_rotor={:.3}",
        start.elapsed(),
        pin("T_avg_Nm"),
        pin("P_fe_W"),
        loaded.torque_arkkio_metrics.ripple_pct,
        loaded.stator_iron_loss.total_w,
        loaded.rotor_iron_loss.total_w
    );
    assert!(
        (t_flux.abs() - pin("T_avg_Nm")).abs() / pin("T_avg_Nm") < 0.02,
        "flux-linkage torque {t_flux} vs pin {}",
        pin("T_avg_Nm")
    );
    assert!(
        (t_arkkio.abs() - pin("T_avg_Nm")).abs() / pin("T_avg_Nm") < 0.05,
        "Arkkio torque {t_arkkio} vs pin {}",
        pin("T_avg_Nm")
    );
    assert!(
        (p_fe - pin("P_fe_W")).abs() / pin("P_fe_W") < 0.10,
        "iron loss {p_fe} vs pin {}",
        pin("P_fe_W")
    );
    // Terminal voltage from the step-voltage identity with the DC phase
    // resistance (the reference current-drive V_peak carries no separate
    // end-winding inductance term).
    let copper = sonium_fem::physics::electromagnetics::rotating_machine::copper_loss_dc(
        &sonium_fem::physics::electromagnetics::rotating_machine::CopperLossInput {
            num_slots: 12,
            conductors_per_slot: 6.0,
            nominal_wire_area_m2: 1.0e-6,
            measured_copper_area_m2: Some(72.0e-6),
            stack_length_m: 0.01,
            n_parallel: 1.0,
            i_phase_rms_a: 60.0,
            coil_temp_c: 120.0,
            end_factor: 0.0,
            tooth_width_m: 2.6e-3,
            bundle_width_m: 2.0e-3,
        },
    );
    let omega_mech = 2.0 * std::f64::consts::PI * 15_000.0 / 60.0;
    let dt = (loaded.frames[1].rotor_angle_deg - loaded.frames[0].rotor_angle_deg).to_radians()
        / omega_mech;
    let dts = vec![dt; loaded.frames.len()];
    let mut v_peak = 0.0f64;
    for phase in 0..3 {
        let psi: Vec<f64> = loaded.frames.iter().map(|f| f.psi_wb_turn[phase]).collect();
        let cur: Vec<f64> = loaded.frames.iter().map(|f| f.currents_a[phase]).collect();
        let (voltages, _) =
            sonium_fem::physics::electromagnetics::rotating_machine::step_voltage_series(
                &psi,
                &cur,
                copper.r_phase_ohm,
                &dts,
                None,
            )
            .unwrap();
        v_peak = v_peak.max(voltages.iter().map(|v| v.abs()).fold(0.0, f64::max));
    }
    eprintln!(
        "V_peak={v_peak:.4} (pin {:.4}), P_cu_dc={:.3} (pin total {:.3} incl. AC proximity)",
        pin("V_peak"),
        copper.power_w,
        pin("P_cu_W")
    );
    assert!(
        (v_peak - pin("V_peak")).abs() / pin("V_peak") < 0.02,
        "V_peak {v_peak} vs pin {}",
        pin("V_peak")
    );
}

#[test]
#[ignore = "heavy: eddy settling plus reported period on the full motor mesh"]
fn p2_eddy_matches_reference_pins() {
    let data = fixture();
    let (polygons, winding, mut materials) = setup(&data);
    // Shaft conductivity from the Aluminium_6061 card (the reference
    // solver looks the assignment up; the constant fallback is 4.5e6).
    materials.shaft_sigma_s_per_m = data["materials"]["conductor_Aluminium_6061"]["sigma"]
        .as_f64()
        .unwrap();
    let mut motor = build_motor_mesh(&polygons, &MotorMeshOptions::default()).unwrap();
    motor.promote_p2(1.0e-7).unwrap();
    let config = EddyTransientConfig {
        steps_per_period: 12,
        settle_periods: 2,
        stack_length_m: 0.010,
        rotor_outer_m: 9.0 * MM,
        stator_inner_m: 9.2 * MM,
        solve: AzSolveConfig {
            continuation_steps: 1,
            ..AzSolveConfig::default()
        },
    };
    let start = std::time::Instant::now();
    let result = run_eddy_current_drive(
        &mut motor,
        PolynomialDegree::P2,
        &winding,
        &materials,
        &DrivePoint {
            i_phase_rms_a: 60.0,
            gamma_deg: 0.0,
            daxis_deg: 60.0,
            rpm: 15_000.0,
            coil_temp_c: 120.0,
        },
        &config,
    )
    .expect("eddy transient");
    let pins = &data["pins"]["p2_eddy"];
    let pin = |key: &str| pins[key].as_f64().unwrap();
    let p_fe = result.stator_iron_loss.total_w + result.rotor_iron_loss.total_w;
    eprintln!(
        "eddy run {:?}: T={:.4} (pin {:.4}) P_cu_solve={:.3} (pin total {:.3}) \
         P_mag={:.4} (pin {:.3}) P_shaft={:.4} (pin {:.3}) P_fe={p_fe:.3} (pin {:.3})",
        start.elapsed(),
        result.torque_terminal_work_mean_nm,
        pin("T_avg_Nm"),
        result.eddy_loss_cu_w,
        pin("P_cu_total_solve_W"),
        result.eddy_loss_magnet_w,
        pin("P_mag_solve_W"),
        result.eddy_loss_shaft_w,
        pin("P_shaft_solve_W"),
        pin("P_fe_W")
    );
    assert!(
        (result.torque_terminal_work_mean_nm.abs() - pin("T_avg_Nm")).abs() / pin("T_avg_Nm")
            < 0.02,
        "eddy torque {} vs pin {}",
        result.torque_terminal_work_mean_nm,
        pin("T_avg_Nm")
    );
    assert!(
        (result.eddy_loss_cu_w - pin("P_cu_total_solve_W")).abs() / pin("P_cu_total_solve_W")
            < 0.10,
        "copper solve loss {} vs pin {}",
        result.eddy_loss_cu_w,
        pin("P_cu_total_solve_W")
    );
    assert!(
        (result.eddy_loss_magnet_w - pin("P_mag_solve_W")).abs() / pin("P_mag_solve_W") < 0.25,
        "magnet eddy loss {} vs pin {}",
        result.eddy_loss_magnet_w,
        pin("P_mag_solve_W")
    );
    assert!(
        (p_fe - pin("P_fe_W")).abs() / pin("P_fe_W") < 0.10,
        "iron loss {p_fe} vs pin {}",
        pin("P_fe_W")
    );
}

#[test]
#[ignore = "heavy: voltage-drive settling on the full motor mesh"]
fn p2_voltage_matches_reference_pins() {
    let data = fixture();
    let (polygons, winding, materials) = setup(&data);
    let mut motor = build_motor_mesh(&polygons, &MotorMeshOptions::default()).unwrap();
    motor.promote_p2(1.0e-7).unwrap();
    let copper = sonium_fem::physics::electromagnetics::rotating_machine::copper_loss_dc(
        &sonium_fem::physics::electromagnetics::rotating_machine::CopperLossInput {
            num_slots: 12,
            conductors_per_slot: 6.0,
            nominal_wire_area_m2: 1.0e-6,
            measured_copper_area_m2: Some(72.0e-6),
            stack_length_m: 0.01,
            n_parallel: 1.0,
            i_phase_rms_a: 60.0,
            coil_temp_c: 120.0,
            end_factor: 0.0,
            tooth_width_m: 2.6e-3,
            bundle_width_m: 2.0e-3,
        },
    );
    let config = VoltageTransientConfig {
        steps_per_period: 12,
        settle_periods: 10,
        settle_tol_rel: 1.0e-4,
        stack_length_m: 0.010,
        rotor_outer_m: 9.0 * MM,
        stator_inner_m: 9.2 * MM,
        // Field Newton at the reference's own 1e-7 residual level:
        // the sliver floor of this mesh sits at ~3e-9 relative, above
        // the 1e-9 default, so every solve would stall and retry.
        solve: {
            let mut cfg = AzSolveConfig {
                continuation_steps: 1,
                ..AzSolveConfig::default()
            };
            cfg.newton.relative_tolerance = 1.0e-7;
            cfg.newton.absolute_tolerance = 1.0e-7;
            cfg
        },
        circuit_tol_v: 1.0e-6,
        max_circuit_iterations: 12,
    };
    let start = std::time::Instant::now();
    let result = run_voltage_drive_transient(
        &mut motor,
        PolynomialDegree::P2,
        &winding,
        &materials,
        &VoltageDrivePoint {
            v_phase_peak_v: 7.0,
            v_delta_deg: 10.0,
            daxis_deg: 60.0,
            rpm: 15_000.0,
            r_phase_ohm: copper.r_phase_ohm,
            i_config_rms_a: 60.0,
            waveform: VoltageWaveform::Sine,
        },
        &config,
    )
    .expect("voltage-drive transient");
    let pins = &data["pins"]["p2_voltage"];
    let pin = |key: &str| pins[key].as_f64().unwrap();
    // Fundamental peak of the solved phase-A current (12-sample DFT).
    let n = result.frames.len() as f64;
    let (mut re, mut im) = (0.0, 0.0);
    for (k, frame) in result.frames.iter().enumerate() {
        let angle = 2.0 * std::f64::consts::PI * k as f64 / n;
        re += frame.currents_a[0] * angle.cos();
        im -= frame.currents_a[0] * angle.sin();
    }
    let i_fund_peak = 2.0 / n * re.hypot(im);
    let i_peak = result
        .frames
        .iter()
        .map(|f| f.currents_a[0].abs())
        .fold(0.0, f64::max);
    let p_fe = result.stator_iron_loss.total_w + result.rotor_iron_loss.total_w;
    eprintln!(
        "voltage run {:?}: T={:.4} (pin {:.4}) I_fund={i_fund_peak:.3} (pin {:.3}) \
         I_peak={i_peak:.3} (pin {:.3}) P_fe={p_fe:.3} (pin {:.3}) R={:.5}",
        start.elapsed(),
        result.torque_terminal_work_mean_nm,
        pin("T_avg_Nm"),
        pin("I_A_fund_A"),
        pin("I_A_peak_A"),
        pin("P_fe_W"),
        copper.r_phase_ohm
    );
    assert!(
        (result.torque_terminal_work_mean_nm.abs() - pin("T_avg_Nm")).abs() / pin("T_avg_Nm")
            < 0.03,
        "voltage-drive torque {} vs pin {}",
        result.torque_terminal_work_mean_nm,
        pin("T_avg_Nm")
    );
    assert!(
        (i_fund_peak - pin("I_A_fund_A")).abs() / pin("I_A_fund_A") < 0.03,
        "solved fundamental current {i_fund_peak} vs pin {}",
        pin("I_A_fund_A")
    );
    assert!(
        (p_fe - pin("P_fe_W")).abs() / pin("P_fe_W") < 0.10,
        "iron loss {p_fe} vs pin {}",
        pin("P_fe_W")
    );
}

#[test]
#[ignore = "heavy: demag settling pass plus reported period on the full motor mesh"]
fn p2_demag_matches_reference_pins() {
    let data = fixture();
    let (polygons, winding, materials) = setup(&data);
    let magnet_card = &data["materials"]["magnet_F45SH_120C"];
    let points: Vec<[f64; 2]> = magnet_card["bh_curve"]
        .as_array()
        .unwrap()
        .iter()
        .map(|point| [point[0].as_f64().unwrap(), point[1].as_f64().unwrap()])
        .collect();
    let curve = MagnetDemagCurve::new(
        &points,
        magnet_card["mu_rec"].as_f64().unwrap(),
        materials.magnet_br_t,
    )
    .expect("demag curve from fixture card");
    let mut motor = build_motor_mesh(&polygons, &MotorMeshOptions::default()).unwrap();
    motor.promote_p2(1.0e-7).unwrap();
    let config = TransientConfig {
        steps_per_period: 12,
        stack_length_m: 0.010,
        rotor_outer_m: 9.0 * MM,
        stator_inner_m: 9.2 * MM,
        solve: AzSolveConfig {
            continuation_steps: 1,
            ..AzSolveConfig::default()
        },
        demag: Some(DemagConfig {
            curve,
            settle_periods: 1,
            max_passes: 11,
        }),
    };
    let start = std::time::Instant::now();
    let result = run_current_drive_transient(
        &mut motor,
        PolynomialDegree::P2,
        &winding,
        &materials,
        &DrivePoint {
            i_phase_rms_a: 60.0,
            gamma_deg: 0.0,
            daxis_deg: 60.0,
            rpm: 15_000.0,
            coil_temp_c: 120.0,
        },
        &config,
    )
    .expect("demag transient");
    let pins = &data["pins"]["p2_demag"];
    let pin = |key: &str| pins[key].as_f64().unwrap();
    let p_fe = result.stator_iron_loss.total_w + result.rotor_iron_loss.total_w;
    eprintln!(
        "demag run {:?}: T={:.4} (pin {:.4}) P_fe={p_fe:.3} (pin {:.3}) \
         br_mean={:.4} area-weighted={:.4} (pin plain {:.4}) br_min={:.4} (pin {:.4})",
        start.elapsed(),
        result.torque_terminal_work_mean_nm,
        pin("T_avg_Nm"),
        pin("P_fe_W"),
        result.demag_br_mean,
        result.demag_br_mean_area_weighted,
        pin("demag_br_mean"),
        result.demag_br_min,
        pin("demag_br_min")
    );
    assert!(
        (result.torque_terminal_work_mean_nm.abs() - pin("T_avg_Nm")).abs() / pin("T_avg_Nm")
            < 0.02,
        "demag torque {} vs pin {}",
        result.torque_terminal_work_mean_nm,
        pin("T_avg_Nm")
    );
    assert!(
        (p_fe - pin("P_fe_W")).abs() / pin("P_fe_W") < 0.10,
        "iron loss {p_fe} vs pin {}",
        pin("P_fe_W")
    );
    // The pin is a plain element-count mean on the reference's
    // sector mesh; that statistic depends on how the mesher
    // populates magnet corners (it moves 0.027 between the
    // reference's own gmsh and netgen meshes: 0.9224 vs the pinned
    // 0.8953). The area-weighted mean (Br kept by volume) is the
    // mesh-convergent quantity: the reference run in this sandbox
    // (tools: /tmp/run_p2demag.py, gmsh CDT) measures 0.98828 with
    // 4.99 % of magnet area de-rated, and the rule itself is gated
    // bitwise in rotating_machine_demag_oracle.
    const REFERENCE_BR_AREA_WEIGHTED: f64 = 0.98828;
    assert!(
        (result.demag_br_mean_area_weighted - REFERENCE_BR_AREA_WEIGHTED).abs() < 0.01,
        "area-weighted br_mean {} vs reference {REFERENCE_BR_AREA_WEIGHTED}",
        result.demag_br_mean_area_weighted,
    );
    assert!(
        (result.demag_br_mean - pin("demag_br_mean")).abs() < 0.07,
        "plain br_mean {} vs pin {}",
        result.demag_br_mean,
        pin("demag_br_mean")
    );
    assert!(
        result.demag_br_min < 0.15,
        "expected a deeply de-rated worst element, got br_min {}",
        result.demag_br_min
    );
}

#[test]
#[ignore = "heavy: PWM voltage-drive smoke on the full motor mesh"]
fn pwm_drive_matches_uncompensated_reference() {
    let data = fixture();
    let (polygons, winding, materials) = setup(&data);
    let mut motor = build_motor_mesh(&polygons, &MotorMeshOptions::default()).unwrap();
    motor.promote_p2(1.0e-7).unwrap();
    let copper = sonium_fem::physics::electromagnetics::rotating_machine::copper_loss_dc(
        &sonium_fem::physics::electromagnetics::rotating_machine::CopperLossInput {
            num_slots: 12,
            conductors_per_slot: 6.0,
            nominal_wire_area_m2: 1.0e-6,
            measured_copper_area_m2: Some(72.0e-6),
            stack_length_m: 0.01,
            n_parallel: 1.0,
            i_phase_rms_a: 60.0,
            coil_temp_c: 120.0,
            end_factor: 0.0,
            tooth_width_m: 2.6e-3,
            bundle_width_m: 2.0e-3,
        },
    );
    // 24 steps = 4 per carrier, the reference's own minimum for a
    // coarse PWM pass. The boundary-repeat tolerance decides when
    // the periodic orbit is reached, as in the sine pins.
    let config = VoltageTransientConfig {
        steps_per_period: 24,
        settle_periods: 6,
        settle_tol_rel: 1.0e-3,
        stack_length_m: 0.010,
        rotor_outer_m: 9.0 * MM,
        stator_inner_m: 9.2 * MM,
        solve: AzSolveConfig {
            continuation_steps: 1,
            ..AzSolveConfig::default()
        },
        circuit_tol_v: 1.0e-6,
        max_circuit_iterations: 12,
    };
    let start = std::time::Instant::now();
    let result = run_voltage_drive_transient(
        &mut motor,
        PolynomialDegree::P2,
        &winding,
        &materials,
        &VoltageDrivePoint {
            v_phase_peak_v: 7.0,
            v_delta_deg: 10.0,
            daxis_deg: 60.0,
            rpm: 15_000.0,
            r_phase_ohm: copper.r_phase_ohm,
            i_config_rms_a: 60.0,
            waveform: VoltageWaveform::Pwm {
                v_bus_v: 20.0,
                carriers_per_period: 6,
            },
        },
        &config,
    )
    .expect("pwm voltage transient");
    let torque = result.torque_terminal_work_mean_nm.abs();
    let n = result.frames.len() as f64;
    let (mut re, mut im) = (0.0, 0.0);
    for (k, frame) in result.frames.iter().enumerate() {
        let angle = 2.0 * std::f64::consts::PI * k as f64 / n;
        re += frame.currents_a[0] * angle.cos();
        im -= frame.currents_a[0] * angle.sin();
    }
    let i_fund_peak = 2.0 / n * re.hypot(im);
    eprintln!(
        "pwm smoke {:?}: T_terminal={torque:.4} T_arkkio={:.4} T_sv={:.4} \
         I_fund={i_fund_peak:.2} (sine pin T 0.2437, I_fund 72.585)",
        start.elapsed(),
        result.torque_arkkio_metrics.mean_nm.abs(),
        result.torque_space_vector_mean_nm.abs(),
    );
    // The uncompensated modulator at 6 carriers lands this machine
    // near a torque null: the reference solver with the same raw
    // modulator monkeypatched in (/tmp/run_pwm_uncomp.py) measures
    // T_avg = 0.0202 N m (Maxwell 0.0200) at |I1| ~ 76 A — the
    // current is there, at a near-reactive angle. The compensated
    // factory (build_pwm_source) exists precisely to undo this.
    // Assert agreement with that measured reference, loosely: near
    // a null, mesh and estimator differences are relatively large.
    const REFERENCE_PWM_TORQUE_NM: f64 = 0.0202;
    assert!(
        torque > 0.4 * REFERENCE_PWM_TORQUE_NM && torque < 2.5 * REFERENCE_PWM_TORQUE_NM,
        "PWM torque {torque} vs uncompensated reference {REFERENCE_PWM_TORQUE_NM}"
    );
    assert!(
        i_fund_peak > 50.0,
        "fundamental current should be sine-comparable, got {i_fund_peak}"
    );
}

/// Diagnostic: which iterative strategies converge on the GEO_30MM
/// P2 tangent system (the CUDA-admitted set is Gmres, GmresJacobi,
/// GmresAmg, Cg, CgAmg; the CPU-only strategies show whether the
/// wall is matrix conditioning or preconditioner strength).
///
/// Not a gate: prints one line per strategy. Failing strategies
/// error on the first frame; succeeding ones run the full 8-frame
/// transient and are compared against the sparse-direct baseline.
#[test]
#[ignore = "diagnostic sweep; heavy"]
fn iterative_strategy_sweep_on_geo30mm() {
    use std::time::Instant;

    use sonium_fem::physics::electromagnetics::rotating_machine::AzLinearBackend;
    use sonium_solvers::{LinearSolveConfig, LinearSolverType};

    let data = fixture();
    let (polygons, winding, materials) = setup(&data);
    let drive = DrivePoint {
        i_phase_rms_a: 60.0,
        gamma_deg: 0.0,
        daxis_deg: 60.0,
        rpm: 15_000.0,
        coil_temp_c: 120.0,
    };
    let run = |solve: AzSolveConfig| {
        let mut motor = build_motor_mesh(&polygons, &MotorMeshOptions::default()).unwrap();
        motor.promote_p2(1.0e-7).unwrap();
        run_current_drive_transient(
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
        )
    };
    let baseline_config = AzSolveConfig {
        continuation_steps: 1,
        ..AzSolveConfig::default()
    };
    let start = Instant::now();
    let baseline = run(baseline_config).expect("direct baseline");
    eprintln!("DirectSparse baseline: {:?}", start.elapsed());
    let reference = &baseline.frames[0].vector_potential_wb_per_m;
    for solver in [
        LinearSolverType::Auto,
        LinearSolverType::GmresIlu,
        LinearSolverType::GmresSchwarz,
        LinearSolverType::FgmresSchwarz,
        LinearSolverType::GmresChebyshev,
        LinearSolverType::GmresAmg,
        LinearSolverType::CgAmg,
        LinearSolverType::Cg,
        LinearSolverType::GmresJacobi,
        LinearSolverType::Gmres,
    ] {
        let mut config = AzSolveConfig {
            continuation_steps: 1,
            ..AzSolveConfig::default()
        };
        config.backend = AzLinearBackend::Iterative;
        config.inner_solver = LinearSolveConfig {
            solver,
            max_iterations: 12_000,
            tolerance: 1.0e-10,
            ..LinearSolveConfig::default()
        };
        let start = Instant::now();
        match run(config) {
            Ok(result) => {
                let frame = &result.frames[0].vector_potential_wb_per_m;
                let num: f64 = frame
                    .iter()
                    .zip(reference)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                let den: f64 = reference.iter().map(|b| b * b).sum();
                eprintln!(
                    "{solver:?}: OK in {:?}, frame-0 rel L2 {:.3e}",
                    start.elapsed(),
                    (num / den.max(1.0e-300)).sqrt()
                );
            }
            Err(error) => {
                eprintln!("{solver:?}: FAIL in {:?}: {error}", start.elapsed());
            }
        }
    }
}
