//! Simulator-owned computations with a common, validated visualization envelope.

// Rust guideline compliant 2026-02-21
use crate::data::{Curve, Frame, Mesh, Metric, Project, ResultData, Series, Surface};
use std::error::Error;

pub fn validate(p: &Project) -> Result<(), String> {
    if p.parameters.contains_key("min_hz") && p.number("min_hz")? >= p.number("max_hz")? {
        return Err("Sweep start must be lower than sweep end".into());
    }
    if p.study == "bracket" && p.number("force")?.abs() <= f64::EPSILON {
        return Err(
            "The harmonic bracket study needs a nonzero tip force to define magnification".into(),
        );
    }
    if p.simulator == "structural"
        && p.study == "cylinder"
        && p.number("inner_radius")? >= p.number("outer_radius")?
    {
        return Err("Inner radius must be smaller than outer radius".into());
    }
    Ok(())
}

fn result(p: &Project, provenance: &str) -> ResultData {
    ResultData {
        schema_version: 1,
        physics_revision: 1,
        project: p.clone(),
        actual_backend: "cpu".into(),
        precision: "f64".into(),
        provenance: provenance.into(),
        elapsed_s: 0.0,
        frames: vec![Frame {
            label: "Static result".into(),
            value: 0.0,
            unit: "".into(),
        }],
        surfaces: Vec::new(),
        curves: Vec::new(),
        metrics: Vec::new(),
        diagnostics: serde_json::json!({}),
    }
}
fn curve(
    title: &str,
    xlabel: &str,
    ylabel: &str,
    x: Vec<f64>,
    series: Vec<(&str, Vec<f64>)>,
    log: bool,
) -> Curve {
    Curve {
        title: title.into(),
        x_label: xlabel.into(),
        y_label: ylabel.into(),
        x,
        log_x: log,
        series: series
            .into_iter()
            .map(|(name, values)| Series {
                name: name.into(),
                values,
            })
            .collect(),
    }
}
fn metric(label: &str, value: f64, unit: &str) -> Metric {
    Metric {
        label: label.into(),
        value,
        unit: unit.into(),
    }
}

pub fn run(p: &Project) -> Result<ResultData, Box<dyn Error>> {
    match p.simulator.as_str() {
        "flow" => flow(p),
        "structural" => structural(p),
        "propeller" => propeller(p),
        "room" => room(p),
        "pcfan" => fan(p),
        "driver" => driver(p),
        _ => Err("Unknown simulator".into()),
    }
}

fn flow_surface(
    name: &str,
    flow: &sonium_fem::physics::fluid::stokes::drag_shape_opt::FlowEvaluation,
    description: &str,
) -> Surface {
    let mesh = Mesh {
        positions: flow
            .mesh
            .nodes
            .iter()
            .map(|node| [node.x, node.y, 0.0])
            .collect(),
        triangles: flow
            .mesh
            .elements
            .iter()
            .map(|element| {
                let vertices = element.vertices();
                [vertices[0] as u32, vertices[1] as u32, vertices[2] as u32]
            })
            .collect(),
    };
    let speed: Vec<f64> = (0..flow.mesh.num_nodes())
        .map(|i| (flow.velocity[2 * i].powi(2) + flow.velocity[2 * i + 1].powi(2)).sqrt())
        .collect();
    Surface {
        name: name.into(),
        mesh,
        association: "node".into(),
        field_name: "Speed".into(),
        field_unit: "m/s".into(),
        scalar_frames: vec![speed],
        displacement: Vec::new(),
        view: "plane".into(),
        description: description.into(),
    }
}

fn centreline_curve(
    flow: &sonium_fem::physics::fluid::stokes::drag_shape_opt::FlowEvaluation,
    center_y: f64,
    channel_length: f64,
) -> Curve {
    use sonium_fem::mesh::interpolation::interpolate_scalar_at_point;
    let ux: Vec<f64> = (0..flow.mesh.num_nodes())
        .map(|i| flow.velocity[2 * i])
        .collect();
    // 64 interpolated samples give a smooth wake profile; points inside
    // the obstacle have no fluid value and report the no-slip value 0.
    let samples = 64;
    let mut x = Vec::with_capacity(samples);
    let mut values = Vec::with_capacity(samples);
    for i in 0..samples {
        let xi = 0.02 + (channel_length - 0.04) * i as f64 / (samples - 1) as f64;
        x.push(xi);
        values
            .push(interpolate_scalar_at_point(&flow.mesh, &ux, [xi, center_y, 0.0]).unwrap_or(0.0));
    }
    curve(
        "Wake centreline velocity",
        "x (m)",
        "Axial velocity (m/s)",
        x,
        vec![("u(x, centreline)", values)],
        false,
    )
}

fn flow(p: &Project) -> Result<ResultData, Box<dyn Error>> {
    use sonium_fem::physics::fluid::stokes::drag_shape_opt::{
        DragShapeOptConfig, evaluate_flow, optimise_drag,
    };
    let base = DragShapeOptConfig {
        reynolds: p.number("reynolds")?,
        nx: p.number("nx")? as usize,
        ny: p.number("ny")? as usize,
        ..Default::default()
    };
    if p.study == "cylinder" {
        let flow = evaluate_flow(&base, &[base.radius, 0.0, 0.0])?;
        let mut data = result(
            p,
            "Native CPU/f64 steady incompressible Navier-Stokes (P1-P1, residual stabilisation, Picard/Oseen) for air (rho 1.225 kg/m3, mu 1.81e-5 Pa s) in the Schäfer–Turek channel; mean inlet velocity follows from the Reynolds number. Drag/lift are boundary-traction integrals on the obstacle. Steady claim only; no vortex-shedding, turbulence or measured validation claim.",
        );
        data.surfaces
            .push(flow_surface("Cylinder flow — speed", &flow, "Solved nodal speed on the retained FEM channel mesh; the circular obstacle is the mesh hole."));
        data.curves
            .push(centreline_curve(&flow, base.center_y, base.channel_length));
        data.metrics = vec![
            metric("Drag coefficient", flow.drag_coefficient, "dimensionless"),
            metric("Lift coefficient", flow.lift_coefficient, "dimensionless"),
            metric("Drag per unit depth", flow.drag, "N/m"),
            metric("Mean inlet velocity", base.mean_velocity(), "m/s"),
            metric(
                "Picard iterations",
                flow.picard_iterations as f64,
                "iterations",
            ),
        ];
        data.diagnostics = serde_json::json!({
            "reynolds": base.reynolds,
            "density_kg_m3": base.density,
            "viscosity_pa_s": base.viscosity,
            "mean_velocity_m_s": base.mean_velocity(),
            "picard_residual": flow.picard_residual,
            "converged": flow.converged,
        });
        return Ok(data);
    }

    let config = DragShapeOptConfig {
        n_modes: p.number("n_modes")? as usize,
        max_outer_iterations: p.number("outer_iterations")? as usize,
        max_inner_iterations: 15,
        ..base
    };
    let opt = optimise_drag(&config)?;
    let reduction = (1.0 - opt.optimal_flow.drag / opt.initial_flow.drag) * 100.0;
    let mut data = result(
        p,
        "Native CPU/f64 Pironneau fixed-area drag minimisation on the steady incompressible Navier-Stokes channel flow (P1-P1, residual stabilisation, Picard/Oseen) for air (rho 1.225 kg/m3, mu 1.81e-5 Pa s). Profile: even Fourier cosines with analytic area constraint; drag gradient by forward finite differences; projected-gradient optimiser with exact area projection. Mesh morphs smoothly from the cylinder mesh within three radii. Steady, laminar claim only; optimum is mesh- and mode-count-dependent, not a measured product claim.",
    );
    data.surfaces.push(flow_surface(
        "Initial cylinder — speed",
        &opt.initial_flow,
        "Solved nodal speed for the initial circular cylinder.",
    ));
    data.surfaces.push(flow_surface(
        "Optimised body — speed",
        &opt.optimal_flow,
        "Solved nodal speed for the fixed-area drag-optimal body.",
    ));
    data.curves.push(curve(
        "Drag convergence",
        "Objective evaluation",
        "Drag coefficient",
        opt.history
            .iter()
            .map(|entry| entry.evaluation as f64)
            .collect(),
        vec![(
            "C_D",
            opt.history
                .iter()
                .map(|entry| entry.drag_coefficient)
                .collect(),
        )],
        false,
    ));
    data.curves.push(centreline_curve(
        &opt.optimal_flow,
        config.center_y,
        config.channel_length,
    ));
    data.metrics = vec![
        metric(
            "Initial C_D",
            opt.initial_flow.drag_coefficient,
            "dimensionless",
        ),
        metric(
            "Optimised C_D",
            opt.optimal_flow.drag_coefficient,
            "dimensionless",
        ),
        metric("Drag reduction", reduction, "%"),
        metric(
            "Aspect ratio (length/width)",
            opt.aspect_ratio,
            "dimensionless",
        ),
        metric(
            "Area error",
            (opt.optimal_area - opt.area_target).abs() / opt.area_target * 100.0,
            "%",
        ),
        metric("Objective evaluations", opt.fun_evals as f64, "evaluations"),
    ];
    data.diagnostics = serde_json::json!({
        "initial_params": opt.initial_params,
        "optimal_params": opt.optimal_params,
        "area_target_m2": opt.area_target,
        "optimal_area_m2": opt.optimal_area,
        "optimiser_converged": opt.converged,
        "optimiser_iterations": opt.optimiser_iterations,
        "history": opt.history.iter().map(|entry| serde_json::json!({"evaluation": entry.evaluation, "drag_coefficient": entry.drag_coefficient, "params": entry.params})).collect::<Vec<_>>(),
    });
    Ok(data)
}

fn structural(p: &Project) -> Result<ResultData, Box<dyn Error>> {
    use sonium_sim_structural_mechanics::{
        BracketStudy, preview::bracket_static_preview, run_bracket_example,
    };
    if p.study == "cylinder" {
        return cylinder(p);
    }
    let mut study = BracketStudy::introductory_bracket();
    study.geometry.length_m = p.number("length")?;
    study.geometry.height_m = p.number("height")?;
    study.geometry.thickness_m = p.number("thickness")?;
    study.geometry.nx = p.number("nx")? as usize;
    study.geometry.ny = p.number("ny")? as usize;
    study.loads.vertical_tip_force_n = p.number("force")?;
    study.loads.axial_prestress_force_n = p.number("preload")?;
    study.material.youngs_modulus_pa = p.number("young_gpa")? * 1e9;
    study.material.density_kg_m3 = p.number("density")?;
    study.material.damping_ratio = p.number("damping")?;
    let count = p.number("points")? as usize;
    study.frequency_hz = (0..count)
        .map(|i| 20.0 * (100.0_f64).powf(i as f64 / (count - 1) as f64))
        .collect();
    let report = run_bracket_example(study)?;
    let preview = bracket_static_preview(&report.study, &report.static_result)?;
    let mut data = result(
        p,
        "Native CPU/f64 P1 plane-stress FEM. Static displacement and equilibrium are solved; modal and harmonic response use assembled stiffness/mass and signed preload. Centroid stress is diagnostic. Geometry deformation is visually exaggerated, with its scale shown in the UI. No general 3D/contact or measured validation claim.",
    );
    data.surfaces.push(Surface {
        name: "Bracket displacement".into(),
        mesh: Mesh {
            positions: preview.nodes,
            triangles: preview.triangles,
        },
        association: "node".into(),
        field_name: preview.scalar_name,
        field_unit: "m".into(),
        scalar_frames: vec![preview.scalar_values],
        displacement: preview.displacement,
        view: "plane".into(),
        description: "Solved static displacement magnitude; deformation scale is display-only."
            .into(),
    });
    data.metrics = vec![
        metric(
            "First bending frequency",
            report.modal.first_bending_frequency_hz,
            "Hz",
        ),
        metric(
            "Tip deflection",
            report.static_result.average_tip_deflection_m * 1000.0,
            "mm",
        ),
        metric(
            "Maximum displacement",
            report.static_result.max_displacement_m * 1000.0,
            "mm",
        ),
        metric(
            "Force balance residual",
            report.static_result.force_balance_residual_n,
            "N",
        ),
        metric(
            "Energy balance error",
            report.static_result.energy_balance_relative_error,
            "relative",
        ),
    ];
    data.curves.push(curve(
        "Harmonic tip response",
        "Frequency (Hz)",
        "Tip displacement magnitude (mm)",
        report
            .frequency_response
            .iter()
            .map(|x| x.frequency_hz)
            .collect(),
        vec![(
            "Harmonic response",
            report
                .frequency_response
                .iter()
                .map(|x| x.displacement_abs_m * 1000.0)
                .collect(),
        )],
        true,
    ));
    data.diagnostics = serde_json::to_value(&report)?;
    Ok(data)
}

fn propeller(p: &Project) -> Result<ResultData, Box<dyn Error>> {
    use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;
    use sonium_sim_propellers::{Blade3dConfig, EvaluationMode, PropellerDesign};
    let source = PropellerSource::drone_5inch(p.number("rpm")?);
    let baseline = PropellerDesign::conventional(source.clone());
    let mut candidate = PropellerDesign::toroidal_serrated(source);
    match p.text("candidate")? {
        "serrated" => candidate.planform = sonium_sim_propellers::PropellerPlanform::Conventional,
        "toroidal" => candidate.edge_treatment = sonium_sim_propellers::EdgeTreatment::Smooth,
        _ => {}
    }
    let config = Blade3dConfig {
        radial_segments: p.number("radial")? as usize,
        chord_segments: p.number("chord")? as usize,
        loop_segments: 16,
        pitch_deg_root: p.number("pitch_root")?,
        pitch_deg_tip: p.number("pitch_tip")?,
        ..Default::default()
    };
    let count = p.number("harmonics")? as usize;
    let spectra = [
        baseline.spectrum(count, EvaluationMode::ResearchControl),
        candidate.spectrum(count, EvaluationMode::ResearchControl),
    ];
    let mut data = result(
        p,
        "Native propeller panel geometry and ResearchControl source model. Harmonics are positive peak-pressure magnitudes at the source reference distance; no complex observer phase or resolved CFD is inferred. Toroidal/serration noise changes are model modifiers and do not establish a matched-thrust or measured-product improvement. Pitch and panel-resolution controls affect geometry; the spectrum uses native source-model parameters. Rotation, when shown, is prescribed kinematics.",
    );
    for (name, design) in [
        ("Conventional baseline", baseline),
        ("Candidate", candidate),
    ] {
        let blade = config.try_build_mesh(&design)?;
        let mut mesh = Mesh {
            positions: Vec::new(),
            triangles: Vec::new(),
        };
        let mut values = Vec::new();
        for panel in &blade.panels {
            let start = mesh.positions.len() as u32;
            mesh.positions
                .extend(panel.vertices.map(|v| [v.x, v.y, v.z]));
            mesh.triangles
                .extend([[start, start + 1, start + 2], [start, start + 2, start + 3]]);
            values.extend([panel.radius_fraction; 2]);
        }
        remove_degenerate_display_triangles(&mut mesh, &mut values);
        data.metrics.push(metric(
            &format!("{name} panel area"),
            blade.total_area(),
            "m²",
        ));
        data.surfaces.push(Surface {
            name: name.into(),
            mesh,
            association: "cell".into(),
            field_name: "Radial station".into(),
            field_unit: "fraction".into(),
            scalar_frames: vec![values],
            displacement: Vec::new(),
            view: "surface".into(),
            description:
                "Native replicated blade panel geometry; colour identifies radial station.".into(),
        });
    }
    data.curves.push(curve(
        "Harmonic source magnitude",
        "Frequency (Hz)",
        "Peak pressure (Pa)",
        spectra[0]
            .harmonics
            .iter()
            .map(|h| h.frequency_hz)
            .collect(),
        spectra
            .iter()
            .zip(["Baseline", "Candidate"])
            .map(|(s, n)| (n, s.harmonics.iter().map(|h| h.pressure_pa).collect()))
            .collect(),
        false,
    ));
    data.diagnostics = serde_json::json!({"baseline":spectra[0],"candidate":spectra[1]});
    Ok(data)
}

fn driver(p: &Project) -> Result<ResultData, Box<dyn Error>> {
    use sonium_sim_speaker_modeling::{EnclosureSpec, PortSpec, ResponseRow, SpeakerDriverStudy};
    let mut study = SpeakerDriverStudy::detailed_frequency_domain_driver();
    let radius = p.number("radius")?;
    study.geometry.diaphragm_radius_m = radius;
    study.geometry.voice_coil_radius_m = study.geometry.voice_coil_radius_m.min(radius * 0.6);
    study.voice_coil.resistance_ohm = p.number("re")?;
    study.voice_coil.inductance_h = p.number("le")?;
    study.voice_coil.bl_tm = p.number("bl")?;
    study.moving_system.moving_mass_kg = p.number("mass")?;
    study.moving_system.stiffness_n_per_m = 1.0 / p.number("cms")?;
    study.moving_system.mechanical_resistance_kg_per_s = p.number("rms")?;
    study.drive_voltage_rms_v = p.number("voltage")?;
    let n = if p.study == "nonlinear" {
        32
    } else {
        p.number("points")? as usize
    };
    let lo = if p.study == "nonlinear" {
        20.0
    } else {
        p.number("min_hz")?
    };
    let hi = if p.study == "nonlinear" {
        3000.0
    } else {
        p.number("max_hz")?
    };
    study.frequency_hz = (0..n)
        .map(|i| lo * (hi / lo).powf(i as f64 / (n - 1) as f64))
        .collect();
    let volume = p.number("volume_l")? * 0.001;
    study.enclosure = match p.text("enclosure")? {
        "sealed" => EnclosureSpec::Sealed {
            volume_m3: volume,
            leakage_resistance_pa_s_per_m3: 0.0,
            absorption_resistance_pa_s_per_m3: 0.0,
        },
        "vented" => EnclosureSpec::Vented {
            volume_m3: volume,
            port: PortSpec {
                radius_m: 0.015,
                length_m: 0.12,
                end_correction_m: 0.0219,
                resistance_pa_s_per_m3: 1000.0,
            },
            leakage_resistance_pa_s_per_m3: 0.0,
            absorption_resistance_pa_s_per_m3: 0.0,
        },
        _ => EnclosureSpec::FreeAir,
    };
    let responses = study.try_small_signal_response_sweep(study.drive_voltage_rms_v)?;
    let rows: Vec<_> = responses
        .iter()
        .map(|r| ResponseRow::from_response(r, &study))
        .collect();
    let mut data = result(
        p,
        "Native CPU/f64 RMS electromechanical model, exp(+iωt). Analytical infinite-baffle rigid-piston front radiation and lumped rear enclosure. Vented preset uses radius 15 mm, length 120 mm, total end correction 21.9 mm, resistance 1000 Pa s/m³. Geometry shows a rigid piston, not distributed structural FEM. Excursion is small-signal; no large-signal safety claim.",
    );
    data.frames = rows
        .iter()
        .map(|r| Frame {
            label: format!("{:.1} Hz", r.frequency_hz),
            value: r.frequency_hz,
            unit: "Hz".into(),
        })
        .collect();
    let mut positions = vec![[0.0, 0.0, 0.0]];
    let segments = 64;
    positions.extend((0..segments).map(|i| {
        let t = std::f64::consts::TAU * i as f64 / segments as f64;
        [radius * t.cos(), radius * t.sin(), 0.0]
    }));
    let triangles = (0..segments)
        .map(|i| [0, (i + 1) as u32, ((i + 1) % segments + 1) as u32])
        .collect();
    data.surfaces.push(Surface {
        name: "Rigid diaphragm".into(),
        mesh: Mesh {
            positions,
            triangles,
        },
        association: "node".into(),
        field_name: "RMS excursion".into(),
        field_unit: "m".into(),
        scalar_frames: rows
            .iter()
            .map(|r| vec![r.displacement_abs_m; segments + 1])
            .collect(),
        displacement: Vec::new(),
        view: "surface".into(),
        description:
            "Uniform rigid-piston excursion magnitude from the native electromechanical response."
                .into(),
    });
    if p.study == "thermal" {
        return thermal(p, &study, data);
    }
    if p.study == "nonlinear" {
        return nonlinear(p, &study, data);
    }
    let frequencies = study.frequency_hz.clone();
    data.curves.push(curve(
        "Electrical impedance",
        "Frequency (Hz)",
        "Impedance (ohm)",
        frequencies.clone(),
        vec![
            (
                "Magnitude",
                rows.iter()
                    .map(|r| r.impedance_real_ohm.hypot(r.impedance_imag_ohm))
                    .collect(),
            ),
            ("Real", rows.iter().map(|r| r.impedance_real_ohm).collect()),
        ],
        true,
    ));
    data.curves.push(curve(
        "Diaphragm excursion",
        "Frequency (Hz)",
        "RMS excursion (mm)",
        frequencies.clone(),
        vec![(
            "Excursion",
            rows.iter().map(|r| r.displacement_abs_m * 1000.0).collect(),
        )],
        true,
    ));
    data.curves.push(curve(
        "Power balance",
        "Frequency (Hz)",
        "Mean power (W)",
        frequencies,
        vec![
            ("Input", rows.iter().map(|r| r.input_power_w).collect()),
            ("Coil loss", rows.iter().map(|r| r.coil_loss_w).collect()),
            (
                "Radiation",
                rows.iter().map(|r| r.radiated_power_w).collect(),
            ),
            (
                "Rear loss",
                rows.iter().map(|r| r.enclosure_loss_power_w).collect(),
            ),
        ],
        true,
    ));
    data.metrics = vec![
        metric(
            "Peak RMS excursion",
            rows.iter()
                .map(|r| r.displacement_abs_m)
                .fold(0.0, f64::max)
                * 1000.0,
            "mm",
        ),
        metric(
            "Power balance residual",
            rows.iter()
                .map(|r| r.power_balance_error_w.abs())
                .fold(0.0, f64::max),
            "W",
        ),
    ];
    data.diagnostics = serde_json::json!({"study":study,"responses":rows});
    Ok(data)
}

fn fan(p: &Project) -> Result<ResultData, Box<dyn Error>> {
    use sonium_sim_pcfan::{PcFanConfig, try_evaluate_pc_fan};
    let mut config = PcFanConfig::default_120mm();
    config.operating_point.rpm = p.number("rpm")?;
    config.mesh.harmonic_count = p.number("harmonics")? as usize;
    let mut design = config.default_design_variables();
    for section in &mut design.blade_sections {
        section.twist_deg += p.number("pitch")?;
    }
    let evaluated = try_evaluate_pc_fan(&config, &design)?;
    let geometry = config.build_geometry(&design);
    let mut mesh = Mesh {
        positions: Vec::new(),
        triangles: Vec::new(),
    };
    let mut scalars = Vec::new();
    for panel in geometry.surface_panels() {
        let start = mesh.positions.len() as u32;
        mesh.positions
            .extend(panel.vertices.map(|v| [v.x, v.y, v.z]));
        mesh.triangles
            .extend([[start, start + 1, start + 2], [start, start + 2, start + 3]]);
        scalars.extend([panel.span_fraction; 2]);
    }
    let removed_triangles = remove_degenerate_display_triangles(&mut mesh, &mut scalars);
    let mut data = result(
        p,
        "Checked native reduced-order fan mean-flow and acoustic source model. Geometry includes rotor and fixed hardware. Colour represents blade span, not CFD pressure. Noise and airflow are model estimates, with configured observer/source conventions; no resolved-flow or measured fan performance claim. RPM sweep keeps this design fixed.",
    );
    data.surfaces.push(Surface {
        name: "Fan assembly".into(),
        mesh,
        association: "cell".into(),
        field_name: "Span station".into(),
        field_unit: "fraction".into(),
        scalar_frames: vec![scalars],
        displacement: Vec::new(),
        view: "surface".into(),
        description: "Native fan panels including rotor, frame, shroud and supports.".into(),
    });
    let harmonics = &evaluated.acoustics.harmonics;
    data.curves.push(curve(
        "Acoustic harmonics",
        "Frequency (Hz)",
        "Native harmonic SPL (dB)",
        harmonics.iter().map(|h| h.frequency_hz).collect(),
        vec![(
            "Harmonics",
            harmonics
                .iter()
                .map(|h| h.sound_pressure_level_db)
                .collect(),
        )],
        false,
    ));
    let count = p.number("rpm_points")? as usize;
    let rpms: Vec<_> = (0..count)
        .map(|i| config.operating_point.rpm * (0.5 + i as f64 / (count - 1) as f64))
        .collect();
    let mut flow = Vec::new();
    let mut pressure = Vec::new();
    let mut noise = Vec::new();
    for &rpm in &rpms {
        let mut point = config.clone();
        point.operating_point.rpm = rpm;
        let e = try_evaluate_pc_fan(&point, &design)?;
        flow.push(e.airflow_cfm);
        pressure.push(e.flow.pressure_rise_pa);
        noise.push(e.a_weighted_acoustic_level_db);
    }
    data.curves.push(curve(
        "Speed / flow",
        "Speed (RPM)",
        "Airflow (CFM)",
        rpms.clone(),
        vec![("Airflow", flow)],
        false,
    ));
    data.curves.push(curve(
        "Speed / pressure",
        "Speed (RPM)",
        "Pressure rise (Pa)",
        rpms.clone(),
        vec![("Pressure rise", pressure)],
        false,
    ));
    data.curves.push(curve(
        "Speed / noise",
        "Speed (RPM)",
        "A-weighted model level (dB(A))",
        rpms,
        vec![("Noise", noise)],
        false,
    ));
    data.metrics = vec![
        metric("Airflow", evaluated.airflow_cfm, "CFM"),
        metric("Pressure rise", evaluated.flow.pressure_rise_pa, "Pa"),
        metric(
            "Blade passing frequency",
            evaluated.acoustics.blade_passing_frequency_hz,
            "Hz",
        ),
        metric(
            "A-weighted noise",
            evaluated.a_weighted_acoustic_level_db,
            "dB(A)",
        ),
    ];
    data.diagnostics = serde_json::json!({"config":config,"design":design,"reynolds":evaluated.flow.reynolds_number,"mach":evaluated.flow.mach_number,"collision":format!("{:?}",evaluated.collision_status),"removed_zero_area_display_triangles":removed_triangles});
    Ok(data)
}

fn room(p: &Project) -> Result<ResultData, Box<dyn Error>> {
    use sonium_sim_room::{
        fem_runner::{FemMeshConfig, run_fem_with_mesh_field_report},
        scenarios::{FrequencyConfig, scenario_by_name_with_config},
    };
    use sonium_xem::{FrequencySpacing, Point3D};
    let frequency = FrequencyConfig {
        min_freq: p.number("min_hz")?,
        max_freq: p.number("max_hz")?,
        num_points: p.number("points")? as usize,
        spacing: FrequencySpacing::default(),
    };
    let scenario = scenario_by_name_with_config(p.text("scenario")?, &frequency)
        .ok_or("Unknown room preset")?;
    let (width, depth, height) = scenario.simulation.room.dimensions();
    let z = p.number("slice_height")?;
    if z >= height {
        return Err("Slice height must be inside the room".into());
    }
    let config = FemMeshConfig {
        modal_damping_ratio: p.number("damping")?,
        ..Default::default()
    };
    let (output, field, diagnostics) =
        run_fem_with_mesh_field_report(&scenario.simulation, &config)?;
    field.validate()?;
    let n = p.number("slice_side")? as usize;
    let points: Vec<_> = (0..n)
        .flat_map(|j| {
            (0..n).map(move |i| {
                Point3D::new(
                    width * (0.01 + 0.98 * i as f64 / (n - 1) as f64),
                    depth * (0.01 + 0.98 * j as f64 / (n - 1) as f64),
                    z,
                )
            })
        })
        .collect();
    let mut triangles = Vec::new();
    for j in 0..n - 1 {
        for i in 0..n - 1 {
            let a = (j * n + i) as u32;
            let b = a + 1;
            let c = a + n as u32;
            triangles.extend([[a, b, c + 1], [a, c + 1, c]]);
        }
    }
    let mesh = Mesh {
        positions: points.iter().map(|v| [v.x, v.y, v.z]).collect(),
        triangles,
    };
    let mut data = result(
        p,
        "Native full-domain Helmholtz FEM, CPU/f64, accepted true-residual convergence. Slice fields are P1 interpolations of retained complex nodal solutions, shown as phasor pressure magnitude in Pa. Sources are displayed independently; no assumed coherent summation. Modal damping is explicit. This is a numerical demonstration; mesh convergence and measured-room agreement are separate qualifications.",
    );
    data.frames = field
        .frequencies
        .iter()
        .map(|&f| Frame {
            label: format!("{f:.1} Hz"),
            value: f,
            unit: "Hz".into(),
        })
        .collect();
    let mut geometry = Mesh {
        positions: vec![
            [0.0, 0.0, 0.0],
            [width, 0.0, 0.0],
            [width, depth, 0.0],
            [0.0, depth, 0.0],
            [0.0, 0.0, height],
            [width, 0.0, height],
            [width, depth, height],
            [0.0, depth, height],
        ],
        triangles: vec![
            [0, 1, 2],
            [0, 2, 3],
            [3, 2, 6],
            [3, 6, 7],
            [0, 3, 7],
            [0, 7, 4],
        ],
    };
    let mut regions = vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    for (point, region) in scenario
        .simulation
        .sources
        .iter()
        .map(|s| (s.position, 2.0))
        .chain(
            scenario
                .simulation
                .listening_positions
                .iter()
                .map(|&p| (p, 3.0)),
        )
    {
        let start = geometry.positions.len() as u32;
        let radius = 0.08;
        geometry.positions.extend([
            [point.x - radius, point.y, point.z],
            [point.x + radius, point.y, point.z],
            [point.x, point.y - radius, point.z],
            [point.x, point.y + radius, point.z],
            [point.x, point.y, point.z - radius],
            [point.x, point.y, point.z + radius],
        ]);
        geometry.triangles.extend(
            [
                [0, 2, 4],
                [2, 1, 4],
                [1, 3, 4],
                [3, 0, 4],
                [2, 0, 5],
                [1, 2, 5],
                [3, 1, 5],
                [0, 3, 5],
            ]
            .map(|t| t.map(|i| i + start)),
        );
        regions.extend([region; 8]);
    }
    data.surfaces.push(Surface{name:"Room layout".into(),mesh:geometry,association:"cell".into(),field_name:"Region".into(),field_unit:"ID".into(),scalar_frames:vec![regions],displacement:Vec::new(),view:"surface".into(),description:"Cutaway dimensions from the native room preset. Region IDs: 0 floor, 1 walls, 2 sources, 3 receivers. Point glyph radius 8 cm is display-only.".into()});
    for (source, name) in field.source_names.iter().enumerate() {
        let values = (0..field.frequencies.len())
            .map(|f| {
                field
                    .sample_p1(source, f, &points)
                    .map(|x| x.iter().map(|v| v.norm()).collect())
            })
            .collect::<Result<Vec<_>, _>>()?;
        data.surfaces.push(Surface {
            name: format!("{name} pressure slice"),
            mesh: mesh.clone(),
            association: "node".into(),
            field_name: "Phasor pressure magnitude".into(),
            field_unit: "Pa".into(),
            scalar_frames: values,
            displacement: Vec::new(),
            view: "plane".into(),
            description: format!(
                "Physical horizontal slice at z = {z:.2} m; independent source {name}."
            ),
        });
        data.curves.push(curve(
            &format!("{name} / receivers"),
            "Frequency (Hz)",
            "Phasor pressure magnitude (Pa)",
            output.frequencies.clone(),
            output.pressures[source]
                .iter()
                .enumerate()
                .map(|(i, values)| {
                    (
                        format!("Receiver {}", i + 1),
                        values.iter().map(|v| v.norm()).collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>()
                .iter()
                .map(|(name, v)| (name.as_str(), v.clone()))
                .collect(),
            true,
        ));
    }
    data.metrics = vec![
        metric("Room volume", scenario.simulation.room.volume(), "m³"),
        metric("FEM nodes", field.mesh.nodes.len() as f64, "nodes"),
        metric(
            "Independent sources",
            field.source_names.len() as f64,
            "sources",
        ),
        metric(
            "Listeners",
            field.listener_positions.len() as f64,
            "positions",
        ),
    ];
    data.diagnostics = serde_json::json!({"scenario":scenario.name,"listeners":field.listener_positions,"solver":format!("{diagnostics:?}"),"room_dimensions_m":[width,depth,height]});
    Ok(data)
}

fn thermal(
    p: &Project,
    study: &sonium_sim_speaker_modeling::SpeakerDriverStudy,
    mut data: ResultData,
) -> Result<ResultData, Box<dyn Error>> {
    use sonium_sim_speaker_modeling::ThermalCompressionSpec;
    let spec = ThermalCompressionSpec {
        thermal_resistance_k_per_w: p.number("thermal_resistance")?,
        ..Default::default()
    };
    let voltage = p.number("voltage")?;
    let hot = study
        .frequency_hz
        .iter()
        .map(|&f| study.try_thermal_compressed_response_at(f, voltage, spec))
        .collect::<Result<Vec<_>, _>>()?;
    data.provenance="Native CPU/f64 steady-state electrothermal driver model. Copper resistance is iterated with the passive front/rear load and power ledger. Single-node coil thermal resistance; this study does not predict cooldown, distributed temperature or material damage. The rigid-piston view shows solved hot RMS excursion, not a temperature field.".into();
    data.surfaces[0].scalar_frames = hot
        .iter()
        .map(|r| vec![r.response.diaphragm_displacement.norm(); 65])
        .collect();
    data.curves = vec![
        curve(
            "Coil temperature",
            "Frequency (Hz)",
            "Steady-state coil temperature (K)",
            study.frequency_hz.clone(),
            vec![("Coil", hot.iter().map(|r| r.coil_temperature_k).collect())],
            true,
        ),
        curve(
            "Thermal compression",
            "Frequency (Hz)",
            "Velocity compression (dB)",
            study.frequency_hz.clone(),
            vec![(
                "Compression",
                hot.iter().map(|r| r.compression_db).collect(),
            )],
            true,
        ),
    ];
    data.metrics = vec![
        metric(
            "Maximum coil temperature",
            hot.iter().map(|r| r.coil_temperature_k).fold(0.0, f64::max),
            "K",
        ),
        metric(
            "Maximum compression",
            hot.iter().map(|r| r.compression_db).fold(0.0, f64::max),
            "dB",
        ),
    ];
    data.diagnostics = serde_json::json!({"thermal_spec":spec,"frequency_diagnostics":hot.iter().map(|r|serde_json::json!({"input_power_w":r.diagnostics.input_power_w,"power_balance_residual_w":r.diagnostics.power_balance_residual_w,"iteration_count":r.diagnostics.thermal_iteration_count,"residual_ohm":r.diagnostics.thermal_residual_ohm,"time_constant_s":r.diagnostics.thermal_time_constant_s,"hot_resistance_ohm":r.hot_resistance_ohm})).collect::<Vec<_>>()});
    Ok(data)
}
fn nonlinear(
    p: &Project,
    study: &sonium_sim_speaker_modeling::SpeakerDriverStudy,
    mut data: ResultData,
) -> Result<ResultData, Box<dyn Error>> {
    use sonium_sim_speaker_modeling::LargeSignalSpec;
    let spec = LargeSignalSpec {
        frequency_hz: p.number("tone")?,
        voltage_rms_v: p.number("voltage")?,
        duration_s: 20.0 / p.number("tone")?,
        samples_per_period: 200,
        bl_drop_fraction: p.number("bl_drop")?,
        suspension_cubic_n_per_m3: p.number("cubic")?,
        ..Default::default()
    };
    let native = study.try_large_signal_response(spec)?;
    data.provenance = format!(
        "Native CPU/f64 lumped nonlinear driver circuit: {:?}. 20 periods, 200 samples/period; specified BL drop and cubic suspension. Rear enclosure uses the native admitted circuit path. THD is derived from native displacement analysis. No resolved nonlinear port CFD or timestep-convergence claim. Rigid-piston view shows peak excursion.",
        native.provenance
    );
    data.frames = vec![Frame {
        label: "Transient peak".into(),
        value: spec.frequency_hz,
        unit: "Hz".into(),
    }];
    data.surfaces[0].field_name = "Peak excursion".into();
    data.surfaces[0].scalar_frames = vec![vec![native.peak_displacement_m; 65]];
    data.curves = vec![
        curve(
            "Displacement waveform",
            "Time (s)",
            "Displacement (mm)",
            native.time_s.clone(),
            vec![(
                "Displacement",
                native.displacement_m.iter().map(|v| v * 1000.0).collect(),
            )],
            false,
        ),
        curve(
            "Current waveform",
            "Time (s)",
            "Current (A)",
            native.time_s.clone(),
            vec![("Current", native.current_a.clone())],
            false,
        ),
    ];
    data.metrics = vec![
        metric("Peak excursion", native.peak_displacement_m * 1000.0, "mm"),
        metric("Displacement THD", native.displacement_thd * 100.0, "%"),
        metric("Peak current", native.peak_current_a, "A"),
    ];
    data.diagnostics = serde_json::to_value(&native)?;
    Ok(data)
}
fn cylinder(p: &Project) -> Result<ResultData, Box<dyn Error>> {
    use sonium_sim_structural_mechanics::{AxisymmetricCylinderStudy, run_axisymmetric_cylinder};
    let mut study = AxisymmetricCylinderStudy::introductory();
    study.geometry.inner_radius_m = p.number("inner_radius")?;
    study.geometry.outer_radius_m = p.number("outer_radius")?;
    study.geometry.length_m = p.number("cylinder_length")?;
    study.geometry.radial_elements = p.number("radial")? as usize;
    study.geometry.axial_elements = p.number("axial")? as usize;
    study.material.youngs_modulus_pa = p.number("young_gpa")? * 1e9;
    study.internal_pressure_pa = p.number("pressure")? * 1e6;
    let native = run_axisymmetric_cylinder(&study)?;
    if !native.converged {
        return Err("Axisymmetric cylinder solve did not converge".into());
    }
    let a = study.geometry.inner_radius_m;
    let b = study.geometry.outer_radius_m;
    let length = study.geometry.length_m;
    let mut data = result(
        p,
        "Native CPU/f64 axisymmetric P1 elasticity, 2πr-weighted assembly, axially constrained plane-strain cylinder. Boundary displacements are compared with the Lamé reference. The section view shows geometry and radial coordinate, not an interpolated displacement field: the native report currently retains boundary summaries. No general shell or nonlinear-contact claim.",
    );
    data.surfaces.push(Surface{name:"Cylinder section".into(),mesh:Mesh{positions:vec![[a,0.0,0.0],[b,0.0,0.0],[b,length,0.0],[a,length,0.0]],triangles:vec![[0,1,2],[0,2,3]]},association:"node".into(),field_name:"Radial coordinate".into(),field_unit:"m".into(),scalar_frames:vec![vec![a,b,b,a]],displacement:Vec::new(),view:"plane".into(),description:"Native study dimensions in the r–z section; boundary displacement summaries are in the plot.".into()});
    data.curves.push(curve(
        "Boundary displacement / reference",
        "Radius (m)",
        "Radial displacement (mm)",
        vec![a, b],
        vec![
            (
                "FEM",
                vec![
                    native.inner_radial_displacement_m * 1000.0,
                    native.outer_radial_displacement_m * 1000.0,
                ],
            ),
            (
                "Lamé",
                vec![
                    native.exact_inner_radial_displacement_m * 1000.0,
                    native.exact_outer_radial_displacement_m * 1000.0,
                ],
            ),
        ],
        false,
    ));
    data.metrics = vec![
        metric(
            "Inner radial displacement",
            native.inner_radial_displacement_m * 1000.0,
            "mm",
        ),
        metric(
            "Displacement relative error",
            native.max_radial_displacement_relative_error,
            "relative",
        ),
        metric(
            "Energy balance error",
            native.energy_balance_relative_error,
            "relative",
        ),
        metric("Solver residual", native.solver_residual, "relative"),
    ];
    data.diagnostics = serde_json::to_value(native)?;
    Ok(data)
}

// Native end-cap quads can contain a collapsed edge. Keep their nonzero triangle;
// this only changes the display tessellation, never native flow/acoustic geometry.
fn remove_degenerate_display_triangles(mesh: &mut Mesh, values: &mut Vec<f64>) -> usize {
    let mut retained = Vec::new();
    let mut scalar = Vec::new();
    for (triangle, &value) in mesh.triangles.iter().zip(values.iter()) {
        let [a, b, c] = triangle.map(|i| mesh.positions[i as usize]);
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let cross = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        if cross.iter().map(|x| x * x).sum::<f64>() > 1e-30 {
            retained.push(*triangle);
            scalar.push(value);
        }
    }
    let removed = mesh.triangles.len() - retained.len();
    mesh.triangles = retained;
    *values = scalar;
    removed
}
