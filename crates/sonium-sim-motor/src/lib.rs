//! Showcase mini-app for the `sonium-fem` rotating-machine solver.
//!
//! This crate demonstrates how the planar `A_z` motor stack is
//! driven end to end on the GEO_30MM 12-slot/14-pole machine (the
//! cross-section and material cards exported from
//! eMotres/Motor_Ai_Simulation, see `data/geo30mm.json`): native
//! sliding-band meshing, d-axis calibration from a no-load sweep,
//! current-drive operating points with terminal-work torque,
//! Bertotti/measured-surface iron loss and DC copper loss, a
//! gamma sweep / MTPA search, a steady thermal check on the same
//! mesh, and a centrifugal rotor-stress check with limit speed.
//!
//! It is a showcase, not a product: the geometry is the one pinned
//! machine, and the knobs are operating-point quantities (current,
//! gamma, speed, stack length, turns). Numbers produced here are
//! the same code paths the `sonium-fem` pins tests validate against
//! the Python reference (torque within ~0.5 % on this machine).

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::physics::electromagnetics::rotating_machine::{
    AzMaterial, AzSolveConfig, BertottiCoefficients, CopperLossInput, DrivePoint, MARKER_AIR,
    MARKER_COIL_BASE, MARKER_MAGNET_BASE, MARKER_ROTOR, MARKER_SHAFT, MARKER_STATOR,
    MeasuredLossSurface, MechMaterial, MotorMaterials, MotorMesh, MotorMeshOptions, MotorPolygons,
    Phase, RegionHeat, RegionPolygon, RobinSurface, RotorStressProblem, RotorStressSolver,
    SlotWinding, SteelLoss, ThermalProblem, TransientConfig, TransientResult, Winding,
    build_motor_mesh, calibrate_daxis_deg, copper_loss_dc, element_centroids_m,
    heat_sources_from_regions, laminated_bh_curve, marker_mean_temperature_c,
    run_current_drive_transient, solve_steady_thermal, step_voltage_series,
};

const MM: f64 = 1.0e-3;
const GEO30MM_JSON: &str = include_str!("../data/geo30mm.json");

/// The pinned GEO_30MM design (cross-section, winding, materials).
#[derive(Debug, Clone)]
pub struct MotorDesign {
    polygons: MotorPolygons,
    winding: Winding,
    materials: MotorMaterials,
    /// Stack length in metres.
    pub stack_length_m: f64,
    /// Rotor outer radius in metres.
    pub rotor_outer_m: f64,
    /// Stator bore radius in metres.
    pub stator_inner_m: f64,
    /// Stator outer radius in metres.
    pub stator_outer_m: f64,
    /// Number of slots.
    pub num_slots: usize,
    /// Wires per slot of the reference winding.
    pub wires_per_slot: f64,
}

/// One current-drive operating point.
#[derive(Debug, Clone, Copy)]
pub struct OperatingPoint {
    /// Terminal phase current in A rms.
    pub i_phase_rms_a: f64,
    /// Current angle from the q axis in electrical degrees
    /// (0 = pure q-axis / MTPA reference of the pinned cases).
    pub gamma_deg: f64,
    /// Rotor speed in rpm.
    pub rpm: f64,
    /// Coil temperature in °C (copper resistivity).
    pub coil_temp_c: f64,
}

impl Default for OperatingPoint {
    fn default() -> Self {
        // The pinned p2_load corner.
        Self {
            i_phase_rms_a: 60.0,
            gamma_deg: 0.0,
            rpm: 15_000.0,
            coil_temp_c: 120.0,
        }
    }
}

/// Solver fidelity knobs for the showcase runs.
#[derive(Debug, Clone, Copy)]
pub struct SimOptions {
    /// Frames per electrical period (must divide the ring pitches
    /// per period; 12 matches the pinned cases).
    pub steps_per_period: usize,
}

impl Default for SimOptions {
    fn default() -> Self {
        Self {
            steps_per_period: 12,
        }
    }
}

/// Headline numbers of one operating point.
#[derive(Debug, Clone)]
pub struct MotorReport {
    /// Operating point solved.
    pub point: OperatingPoint,
    /// Calibrated d-axis offset in electrical degrees.
    pub daxis_deg: f64,
    /// Mean electromagnetic torque in N m (terminal-work estimator).
    pub torque_nm: f64,
    /// Arkkio torque ripple in percent of mean.
    pub torque_ripple_pct: f64,
    /// Total iron loss in watts.
    pub iron_loss_w: f64,
    /// DC copper loss in watts (no AC proximity term).
    pub copper_loss_w: f64,
    /// Peak phase terminal voltage in volts (step-voltage identity).
    pub v_peak_v: f64,
    /// Shaft power in watts.
    pub shaft_power_w: f64,
    /// Efficiency estimate in percent, from the losses above only.
    pub efficiency_pct: f64,
    /// Full transient result (frames, flux linkages, fields).
    pub transient: TransientResult,
}

/// Thermal check outcome at one operating point.
#[derive(Debug, Clone)]
pub struct ThermalReport {
    /// Mean coil temperature in °C.
    pub coil_mean_c: f64,
    /// Mean magnet temperature in °C.
    pub magnet_mean_c: f64,
    /// Mean stator temperature in °C.
    pub stator_mean_c: f64,
    /// Heat leaving through the housing film in watts.
    pub housing_heat_w: f64,
    /// Budget closure error in watts.
    pub heat_residual_w: f64,
}

/// Rotor mechanical check outcome.
#[derive(Debug, Clone)]
pub struct MechReport {
    /// Minimum safety factor at the solved speed.
    pub safety_factor_min: f64,
    /// Marker carrying the minimum safety factor.
    pub limiting_marker: i32,
    /// Estimated speed where SF reaches 1, when found.
    pub limit_speed_rpm: Option<f64>,
    /// Largest radial growth at speed, in metres.
    pub max_radial_growth_m: f64,
}

fn ring(value: &Value) -> Vec<(f64, f64)> {
    value
        .as_array()
        .expect("ring array")
        .iter()
        .map(|point| {
            let pair = point.as_array().expect("point pair");
            (
                pair[0].as_f64().expect("x") * MM,
                pair[1].as_f64().expect("y") * MM,
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
            .expect("interiors")
            .iter()
            .map(ring)
            .collect(),
    }
}

impl MotorDesign {
    /// The pinned GEO_30MM machine.
    ///
    /// # Errors
    ///
    /// Returns [`MotorSimError`] if the embedded data does not parse
    /// into valid materials (not expected for the shipped file).
    pub fn geo30mm() -> Result<Self, MotorSimError> {
        let data: Value =
            serde_json::from_str(GEO30MM_JSON).map_err(|e| MotorSimError::Data(e.to_string()))?;
        let polys = &data["polygons_mm"];
        let polygons = MotorPolygons {
            stator: polygon(&polys["stator"][0]),
            rotor: polygon(&polys["rotor"][0]),
            shaft: polygon(&polys["shaft"][0]),
            magnets: polys["magnets"]
                .as_array()
                .expect("magnets")
                .iter()
                .map(|m| polygon(&m["polygon"][0]))
                .collect(),
            coils: polys["coils"]
                .as_array()
                .expect("coils")
                .iter()
                .map(|c| {
                    (
                        c["slot"].as_u64().expect("slot") as usize,
                        polygon(&c["polygon"][0]),
                    )
                })
                .collect(),
            mid_gap_radius_m: data["derived"]["mid_gap_radius_mm"]
                .as_f64()
                .expect("mid gap")
                * MM,
            outer_radius_m: data["derived"]["outer_boundary_radius_mm"]
                .as_f64()
                .expect("outer radius")
                * MM,
        };
        let layout: Vec<SlotWinding> = data["winding_layout"]
            .as_array()
            .expect("layout")
            .iter()
            .map(|entry| {
                let pair = entry.as_array().expect("layout entry");
                SlotWinding {
                    phase: match pair[0].as_str().expect("phase") {
                        "A" => Phase::A,
                        "B" => Phase::B,
                        _ => Phase::C,
                    },
                    direction: i8::try_from(pair[1].as_i64().expect("direction"))
                        .expect("direction in i8"),
                }
            })
            .collect();
        let num_slots = layout.len();
        let winding = Winding {
            layout,
            turns_per_slot: 6.0,
            n_parallel: 1.0,
            pole_pairs: 7,
        };
        let steel = &data["materials"]["steel_B15AHV950M"];
        let hb: Vec<(f64, f64)> = steel["bh_curve"]
            .as_array()
            .expect("bh curve")
            .iter()
            .map(|p| {
                let pair = p.as_array().expect("bh point");
                (pair[0].as_f64().expect("H"), pair[1].as_f64().expect("B"))
            })
            .collect();
        let kf = steel["stacking_factor"].as_f64().expect("stacking factor");
        let bertotti = BertottiCoefficients {
            kh: steel["core_loss_kh"].as_f64().expect("kh"),
            kc: steel["core_loss_kc"].as_f64().expect("kc"),
            ke: steel["core_loss_ke"].as_f64().expect("ke"),
        };
        let mut frequencies = Vec::new();
        let mut tables = Vec::new();
        for (key, curve) in steel["core_loss_curves"].as_object().expect("loss curves") {
            frequencies.push(
                key.trim_end_matches("Hz")
                    .parse::<f64>()
                    .map_err(|e| MotorSimError::Data(e.to_string()))?,
            );
            tables.push(
                curve
                    .as_array()
                    .expect("loss curve")
                    .iter()
                    .map(|p| {
                        let pair = p.as_array().expect("loss point");
                        (pair[0].as_f64().expect("B"), pair[1].as_f64().expect("p"))
                    })
                    .collect::<Vec<_>>(),
            );
        }
        let surface = MeasuredLossSurface::new(
            &frequencies,
            &tables,
            steel["density"].as_f64().expect("density"),
            steel["core_loss_curve_unit"].as_str() == Some("w_per_kg"),
            bertotti,
        )
        .map_err(|e| MotorSimError::Data(e.to_string()))?;
        let magnet = &data["materials"]["magnet_F45SH_120C"];
        let materials = MotorMaterials {
            steel: AzMaterial::nonlinear(
                laminated_bh_curve(&hb, kf).map_err(|e| MotorSimError::Data(e.to_string()))?,
            ),
            steel_loss: SteelLoss {
                bertotti,
                stacking_factor: kf,
                surface: Some(surface),
            },
            magnet_br_t: magnet["Br"].as_f64().expect("Br"),
            magnet_mu_rec: magnet["mu_rec"].as_f64().expect("mu_rec"),
            magnet_polarity: polys["magnets"]
                .as_array()
                .expect("magnets")
                .iter()
                .map(|m| m["polarity"].as_f64().expect("polarity"))
                .collect(),
            magnet_sigma_s_per_m: magnet["sigma"].as_f64().expect("magnet sigma"),
            shaft_mu_r: data["materials"]["conductor_Aluminium_6061"]["mu_r"]
                .as_f64()
                .unwrap_or(1.0),
            shaft_sigma_s_per_m: data["constants"]["SIGMA_SHAFT"]
                .as_f64()
                .expect("shaft sigma"),
        };
        Ok(Self {
            polygons,
            winding,
            materials,
            stack_length_m: data["derived"]["stack_length_mm"].as_f64().expect("stack") * MM,
            rotor_outer_m: data["derived"]["rotor_outer_radius_mm"]
                .as_f64()
                .expect("rotor outer")
                * MM,
            stator_inner_m: data["derived"]["stator_inner_radius_mm"]
                .as_f64()
                .expect("stator inner")
                * MM,
            stator_outer_m: data["derived"]["stator_outer_radius_mm"]
                .as_f64()
                .expect("stator outer")
                * MM,
            num_slots,
            wires_per_slot: 6.0,
        })
    }

    /// Marker list of every coil wire region (per-wire markers).
    fn coil_markers(&self, motor: &MotorMesh) -> Vec<i32> {
        (0..motor.coil_slot_of_wire.len())
            .map(|wire| MARKER_COIL_BASE + wire as i32)
            .collect()
    }

    /// Marker list of every magnet region.
    fn magnet_markers(&self) -> Vec<i32> {
        (0..self.materials.magnet_polarity.len())
            .map(|index| MARKER_MAGNET_BASE + index as i32)
            .collect()
    }
}

/// A reusable simulator: owns the promoted sliding-band mesh and
/// the calibrated d-axis, so sweeps do not remesh or recalibrate.
pub struct MotorSimulator {
    design: MotorDesign,
    options: SimOptions,
    motor: MotorMesh,
    daxis_deg: Option<f64>,
}

impl MotorSimulator {
    /// Builds the native mesh and promotes it to P2.
    ///
    /// # Errors
    ///
    /// Returns [`MotorSimError`] on a meshing failure.
    pub fn new(design: MotorDesign, options: SimOptions) -> Result<Self, MotorSimError> {
        let mut motor = build_motor_mesh(&design.polygons, &MotorMeshOptions::default())
            .map_err(|e| MotorSimError::Design(e.to_string()))?;
        motor
            .promote_p2(1.0e-7)
            .map_err(|e| MotorSimError::Design(e.to_string()))?;
        Ok(Self {
            design,
            options,
            motor,
            daxis_deg: None,
        })
    }

    fn transient_config(&self) -> TransientConfig {
        TransientConfig {
            steps_per_period: self.options.steps_per_period,
            stack_length_m: self.design.stack_length_m,
            rotor_outer_m: self.design.rotor_outer_m,
            stator_inner_m: self.design.stator_inner_m,
            solve: AzSolveConfig {
                continuation_steps: 1,
                ..AzSolveConfig::default()
            },
            demag: None,
        }
    }

    /// Calibrated d-axis offset, from a no-load sweep on first use.
    ///
    /// # Errors
    ///
    /// Propagates transient failures.
    pub fn daxis_deg(&mut self) -> Result<f64, MotorSimError> {
        if let Some(daxis) = self.daxis_deg {
            return Ok(daxis);
        }
        let config = self.transient_config();
        let noload = run_current_drive_transient(
            &mut self.motor,
            PolynomialDegree::P2,
            &self.design.winding,
            &self.design.materials,
            &DrivePoint {
                i_phase_rms_a: 0.0,
                gamma_deg: 0.0,
                daxis_deg: 0.0,
                rpm: 15_000.0,
                coil_temp_c: 120.0,
            },
            &config,
        )?;
        let angles: Vec<f64> = noload.frames.iter().map(|f| f.rotor_angle_deg).collect();
        let psi_a: Vec<f64> = noload.frames.iter().map(|f| f.psi_wb_turn[0]).collect();
        let daxis = calibrate_daxis_deg(&angles, &psi_a, self.design.winding.pole_pairs)
            .map_err(|e| MotorSimError::Design(e.to_string()))?;
        self.daxis_deg = Some(daxis);
        Ok(daxis)
    }

    /// Solves one operating point.
    ///
    /// # Errors
    ///
    /// Propagates transient and post-processing failures.
    pub fn simulate(&mut self, point: OperatingPoint) -> Result<MotorReport, MotorSimError> {
        let daxis_deg = self.daxis_deg()?;
        let config = self.transient_config();
        let transient = run_current_drive_transient(
            &mut self.motor,
            PolynomialDegree::P2,
            &self.design.winding,
            &self.design.materials,
            &DrivePoint {
                i_phase_rms_a: point.i_phase_rms_a,
                gamma_deg: point.gamma_deg,
                daxis_deg,
                rpm: point.rpm,
                coil_temp_c: point.coil_temp_c,
            },
            &config,
        )?;
        let iron_loss_w = transient.stator_iron_loss.total_w + transient.rotor_iron_loss.total_w;
        let copper = copper_loss_dc(&CopperLossInput {
            num_slots: self.design.num_slots,
            conductors_per_slot: self.design.wires_per_slot,
            nominal_wire_area_m2: 1.0e-6,
            measured_copper_area_m2: Some(72.0e-6),
            stack_length_m: self.design.stack_length_m,
            n_parallel: self.design.winding.n_parallel,
            i_phase_rms_a: point.i_phase_rms_a,
            coil_temp_c: point.coil_temp_c,
            end_factor: 0.0,
            tooth_width_m: 2.6e-3,
            bundle_width_m: 2.0e-3,
        });
        let torque_nm = transient.torque_terminal_work_mean_nm.abs();
        let shaft_power_w = torque_nm * point.rpm * 2.0 * std::f64::consts::PI / 60.0;
        let losses = iron_loss_w + copper.power_w;
        let efficiency_pct = if shaft_power_w > 0.0 {
            100.0 * shaft_power_w / (shaft_power_w + losses)
        } else {
            0.0
        };
        // Peak terminal voltage from the step-voltage identity.
        let omega_mech = point.rpm * 2.0 * std::f64::consts::PI / 60.0;
        let v_peak_v = if transient.frames.len() > 1 && omega_mech > 0.0 {
            let dt = (transient.frames[1].rotor_angle_deg - transient.frames[0].rotor_angle_deg)
                .to_radians()
                / omega_mech;
            let dts = vec![dt; transient.frames.len()];
            let mut peak = 0.0f64;
            for phase in 0..3 {
                let psi: Vec<f64> = transient
                    .frames
                    .iter()
                    .map(|f| f.psi_wb_turn[phase])
                    .collect();
                let cur: Vec<f64> = transient
                    .frames
                    .iter()
                    .map(|f| f.currents_a[phase])
                    .collect();
                let (voltages, _) = step_voltage_series(&psi, &cur, copper.r_phase_ohm, &dts, None)
                    .map_err(|e| MotorSimError::Design(e.to_string()))?;
                peak = peak.max(voltages.iter().map(|v| v.abs()).fold(0.0, f64::max));
            }
            peak
        } else {
            0.0
        };
        Ok(MotorReport {
            point,
            daxis_deg,
            torque_nm,
            torque_ripple_pct: transient.torque_arkkio_metrics.ripple_pct,
            iron_loss_w,
            copper_loss_w: copper.power_w,
            v_peak_v,
            shaft_power_w,
            efficiency_pct,
            transient,
        })
    }

    /// Solves a list of gamma values at one operating point.
    ///
    /// # Errors
    ///
    /// Propagates the first failing solve.
    pub fn sweep_gamma(
        &mut self,
        point: OperatingPoint,
        gammas_deg: &[f64],
    ) -> Result<Vec<MotorReport>, MotorSimError> {
        gammas_deg
            .iter()
            .map(|&gamma_deg| self.simulate(OperatingPoint { gamma_deg, ..point }))
            .collect()
    }

    /// Golden-section search for the gamma of maximum torque.
    ///
    /// # Errors
    ///
    /// Propagates solve failures.
    pub fn max_torque_gamma(
        &mut self,
        point: OperatingPoint,
        lo_deg: f64,
        hi_deg: f64,
        iterations: usize,
    ) -> Result<MotorReport, MotorSimError> {
        let inv_phi = (5.0f64.sqrt() - 1.0) / 2.0;
        let (mut a, mut b) = (lo_deg, hi_deg);
        let mut c = b - inv_phi * (b - a);
        let mut d = a + inv_phi * (b - a);
        let mut report_c = self.simulate(OperatingPoint {
            gamma_deg: c,
            ..point
        })?;
        let mut report_d = self.simulate(OperatingPoint {
            gamma_deg: d,
            ..point
        })?;
        for _ in 0..iterations {
            if report_c.torque_nm < report_d.torque_nm {
                a = c;
                c = d;
                report_c = report_d;
                d = a + inv_phi * (b - a);
                report_d = self.simulate(OperatingPoint {
                    gamma_deg: d,
                    ..point
                })?;
            } else {
                b = d;
                d = c;
                report_d = report_c;
                c = b - inv_phi * (b - a);
                report_c = self.simulate(OperatingPoint {
                    gamma_deg: c,
                    ..point
                })?;
            }
        }
        Ok(if report_c.torque_nm >= report_d.torque_nm {
            report_c
        } else {
            report_d
        })
    }

    /// Steady thermal check: region losses mapped onto the mesh,
    /// housing film at the stator outer diameter, shaft bore film.
    ///
    /// Conductivities are showcase values (lamination in-plane,
    /// copper, NdFeB, aluminium shaft, effective gap air); they are
    /// parameters of this check, not fitted data.
    ///
    /// # Errors
    ///
    /// Propagates thermal-solve failures.
    pub fn thermal_check(
        &self,
        report: &MotorReport,
        housing_h_w_per_m2k: f64,
        t_ambient_c: f64,
    ) -> Result<ThermalReport, MotorSimError> {
        let mesh = &self.motor.mesh;
        let centroids = element_centroids_m(mesh);
        let (mut k_t, mut keep) = (Vec::with_capacity(mesh.num_elements()), Vec::new());
        for (index, element) in mesh.elements.iter().enumerate() {
            let radius = centroids[index][0].hypot(centroids[index][1]);
            // Drop the far-field air beyond the stator; keep gap and
            // slot air as low-conductivity solid.
            keep.push(!(element.marker == MARKER_AIR && radius > self.design.stator_outer_m));
            k_t.push(match element.marker {
                MARKER_STATOR | MARKER_ROTOR => 25.0,
                MARKER_SHAFT => 167.0,
                MARKER_AIR => 0.09,
                m if m >= MARKER_MAGNET_BASE => 7.6,
                _ => 385.0,
            });
        }
        let q = heat_sources_from_regions(
            mesh,
            &[
                RegionHeat {
                    markers: vec![MARKER_STATOR],
                    watts: report.transient.stator_iron_loss.total_w,
                },
                RegionHeat {
                    markers: vec![MARKER_ROTOR],
                    watts: report.transient.rotor_iron_loss.total_w,
                },
                RegionHeat {
                    markers: self.design.coil_markers(&self.motor),
                    watts: report.copper_loss_w,
                },
            ],
            self.design.stack_length_m,
        );
        let problem = ThermalProblem {
            k_tangential_w_per_mk: k_t,
            k_radial_w_per_mk: None,
            heat_source_w_per_m3: q,
            drop_markers: Vec::new(),
            element_keep: keep,
            surfaces: vec![
                RobinSurface {
                    name: "housing".to_string(),
                    radius_m: self.design.stator_outer_m,
                    radius_tol_m: 5.0e-5,
                    h_w_per_m2k: housing_h_w_per_m2k,
                    t_sink_c: t_ambient_c,
                },
                RobinSurface {
                    name: "shaft_bore".to_string(),
                    radius_m: mesh
                        .nodes
                        .iter()
                        .map(|node| node.x.hypot(node.y))
                        .fold(f64::INFINITY, f64::min),
                    radius_tol_m: 5.0e-5,
                    h_w_per_m2k: housing_h_w_per_m2k,
                    t_sink_c: t_ambient_c,
                },
            ],
            volume_sinks: Vec::new(),
            stack_length_m: self.design.stack_length_m,
        };
        let result = solve_steady_thermal(mesh, &problem)?;
        let mean = |markers: &[i32]| -> f64 {
            markers
                .iter()
                .filter_map(|&m| marker_mean_temperature_c(mesh, &result.temperature_c, m))
                .fold((0.0, 0usize), |(sum, n), t| (sum + t, n + 1))
                .0
                / markers.len().max(1) as f64
        };
        Ok(ThermalReport {
            coil_mean_c: mean(&self.design.coil_markers(&self.motor)),
            magnet_mean_c: mean(&self.design.magnet_markers()),
            stator_mean_c: mean(&[MARKER_STATOR]),
            housing_heat_w: result
                .surfaces
                .first()
                .map_or(0.0, |surface| surface.heat_removed_w),
            heat_residual_w: result.heat_residual_w,
        })
    }

    /// Centrifugal rotor-stress check at `rpm` with uniform
    /// temperature rise `delta_t_k` over the rotor assembly.
    ///
    /// # Errors
    ///
    /// Propagates mechanical-solve failures.
    pub fn mech_check(&self, rpm: f64, delta_t_k: f64) -> Result<MechReport, MotorSimError> {
        let steel = MechMaterial::Isotropic {
            youngs_modulus_pa: 200.0e9,
            poisson_ratio: 0.30,
            density_kg_per_m3: 7_650.0,
            thermal_expansion_per_k: 12.0e-6,
            yield_pa: 400.0e6,
        };
        let magnet = MechMaterial::Isotropic {
            youngs_modulus_pa: 160.0e9,
            poisson_ratio: 0.28,
            density_kg_per_m3: 7_500.0,
            thermal_expansion_per_k: 5.0e-6,
            yield_pa: 80.0e6,
        };
        let shaft = MechMaterial::Isotropic {
            youngs_modulus_pa: 69.0e9,
            poisson_ratio: 0.33,
            density_kg_per_m3: 2_700.0,
            thermal_expansion_per_k: 23.0e-6,
            yield_pa: 276.0e6,
        };
        let mut materials: std::collections::BTreeMap<i32, MechMaterial> =
            [(MARKER_ROTOR, steel), (MARKER_SHAFT, shaft)]
                .into_iter()
                .collect();
        for marker in self.design.magnet_markers() {
            materials.insert(marker, magnet);
        }
        let mut drop: Vec<i32> = vec![MARKER_STATOR, MARKER_AIR];
        drop.extend(self.design.coil_markers(&self.motor));
        let problem = RotorStressProblem {
            materials,
            drop_markers: drop,
            delta_t_k: vec![delta_t_k; self.motor.mesh.num_elements()],
        };
        let solver = RotorStressSolver::new(&self.motor.mesh, &problem)?;
        let omega = rpm * 2.0 * std::f64::consts::PI / 60.0;
        let result = solver.solve(omega)?;
        let limiting_marker = result
            .safety_factor_by_marker
            .iter()
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map_or(0, |entry| entry.0);
        Ok(MechReport {
            safety_factor_min: result.safety_factor_min,
            limiting_marker,
            limit_speed_rpm: solver.limit_speed_rpm(rpm)?,
            max_radial_growth_m: result.max_radial_growth_m,
        })
    }
}

/// Failure of a showcase motor simulation.
#[derive(Debug, thiserror::Error)]
pub enum MotorSimError {
    /// Embedded design data did not parse or validate.
    #[error("invalid motor design data: {0}")]
    Data(String),
    /// Design construction failed.
    #[error("motor design failure: {0}")]
    Design(String),
    /// A transient solve failed.
    #[error(transparent)]
    Transient(#[from] sonium_fem::physics::electromagnetics::rotating_machine::TransientError),
    /// The thermal check failed.
    #[error(transparent)]
    Thermal(#[from] sonium_fem::physics::electromagnetics::rotating_machine::ThermalError),
    /// The mechanical check failed.
    #[error(transparent)]
    Mech(#[from] sonium_fem::physics::electromagnetics::rotating_machine::MechError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geo30mm_loads_and_meshes() {
        let design = MotorDesign::geo30mm().expect("design loads");
        assert_eq!(design.num_slots, 12);
        assert_eq!(design.winding.pole_pairs, 7);
        assert!((design.stack_length_m - 0.010).abs() < 1.0e-12);
        let simulator = MotorSimulator::new(design, SimOptions::default()).expect("mesh builds");
        assert_eq!(simulator.motor.ring_nodes.len() % 2, 0, "P2 ring doubled");
        assert!(simulator.motor.mesh.num_elements() > 10_000);
    }

    #[test]
    fn mech_check_reports_a_safety_factor() {
        let design = MotorDesign::geo30mm().expect("design loads");
        let simulator = MotorSimulator::new(design, SimOptions::default()).expect("mesh builds");
        // Pure centrifugal load on a 9 mm rotor at 15000 rpm is a
        // sub-MPa stress state; the check must clear SF 1 by far.
        let mech = simulator.mech_check(15_000.0, 0.0).expect("mech check");
        assert!(
            mech.safety_factor_min > 1.0,
            "rotor should be safe at 15000 rpm, SF {}",
            mech.safety_factor_min
        );
        assert!(mech.max_radial_growth_m > 0.0);
        // The bonded model's thermal mismatch (magnet alpha below
        // iron's) loads the magnets in tension at temperature; the
        // reference models pocket contact instead, so here it is
        // only asserted to degrade the factor, not to clear it.
        let hot = simulator
            .mech_check(15_000.0, 80.0)
            .expect("hot mech check");
        assert!(
            hot.safety_factor_min < mech.safety_factor_min,
            "thermal mismatch should degrade SF: {} vs {}",
            hot.safety_factor_min,
            mech.safety_factor_min
        );
    }
}
