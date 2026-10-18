//! Study catalogs and bounded input domains for the standalone demos.

// Rust guideline compliant 2026-02-21
use crate::data::{App, Field, Study};

fn number(key: &str, label: &str, value: f64, min: f64, max: f64) -> Field {
    Field {
        key: key.into(),
        label: label.into(),
        initial: value.to_string(),
        min,
        max,
        integer: false,
        choices: Vec::new(),
    }
}
fn integer(key: &str, label: &str, value: usize, min: usize, max: usize) -> Field {
    let mut f = number(key, label, value as f64, min as f64, max as f64);
    f.integer = true;
    f
}
fn choice(key: &str, label: &str, choices: &[&str]) -> Field {
    Field {
        key: key.into(),
        label: label.into(),
        initial: choices[0].into(),
        min: 0.0,
        max: 0.0,
        integer: false,
        choices: choices.iter().map(|s| (*s).into()).collect(),
    }
}
fn study(id: &str, title: &str, description: &str, fields: Vec<Field>) -> Study {
    Study {
        id: id.into(),
        title: title.into(),
        description: description.into(),
        fields,
    }
}

pub fn app(simulator: &str) -> Result<App, String> {
    let (title, description, studies) = match simulator {
        "room" => (
            "Room Lab",
            "Room geometry, listening positions and native FEM pressure fields",
            vec![study(
                "fem",
                "Room response",
                "Full-domain Helmholtz FEM with retained nodal pressures and explicit wall/source conventions.",
                vec![
                    choice(
                        "scenario",
                        "Room preset",
                        &[
                            "small_stereo_2_0",
                            "medium_multi_seat",
                            "medium_surround_5_1",
                        ],
                    ),
                    number("min_hz", "Sweep start (Hz)", 30.0, 20.0, 200.0),
                    number("max_hz", "Sweep end (Hz)", 120.0, 30.0, 250.0),
                    integer("points", "Frequency samples", 8, 2, 32),
                    number("damping", "Modal damping fraction", 0.0005, 0.0, 0.02),
                    number("slice_height", "Slice height (m)", 1.2, 0.1, 2.0),
                    integer("slice_side", "Slice samples per side", 24, 8, 48),
                ],
            )],
        ),
        "pcfan" => (
            "Fan Lab",
            "Fan geometry, airflow, pressure rise and harmonic acoustic studies",
            vec![study(
                "operating",
                "Operating point",
                "Checked native reduced-order mean flow and acoustic source model; no resolved CFD claim.",
                vec![
                    number("rpm", "Rotor speed (RPM)", 1500.0, 300.0, 5000.0),
                    number("pitch", "Pitch offset (degrees)", 0.0, -5.0, 5.0),
                    integer("harmonics", "Noise harmonics", 12, 3, 32),
                    integer("rpm_points", "RPM sweep samples", 8, 3, 20),
                ],
            )],
        ),
        "propeller" => (
            "Propeller Lab",
            "Blade geometry and source-model acoustic comparisons",
            vec![study(
                "panels",
                "Blade/source comparison",
                "Native panel geometry with source-generation modifiers; spectra are reduced source estimates, not resolved observer CFD.",
                vec![
                    number("rpm", "Rotor speed (RPM)", 18000.0, 3000.0, 24000.0),
                    choice(
                        "candidate",
                        "Candidate treatment",
                        &["serrated", "toroidal", "combined"],
                    ),
                    integer("harmonics", "Noise harmonics", 16, 3, 32),
                    integer("radial", "Radial panels", 24, 8, 64),
                    integer("chord", "Chord panels", 6, 3, 12),
                    number("pitch_root", "Root pitch (degrees)", 28.0, 10.0, 40.0),
                    number("pitch_tip", "Tip pitch (degrees)", 12.0, 5.0, 25.0),
                ],
            )],
        ),
        "driver" => {
            let common = vec![
                number("voltage", "Input drive (V RMS)", 1.0, 0.001, 10.0),
                number(
                    "radius",
                    "Effective diaphragm radius (m)",
                    0.065,
                    0.01,
                    0.15,
                ),
                number("re", "Coil Re (ohm)", 6.0, 0.5, 30.0),
                number("le", "Coil Le (H)", 0.0005, 0.0, 0.01),
                number("bl", "Force factor BL (T m)", 6.0, 0.5, 30.0),
                number("mass", "Moving mass (kg)", 0.014, 0.001, 0.1),
                number("cms", "Compliance (m/N)", 0.0005, 0.00001, 0.01),
                number("rms", "Mechanical loss (N s/m)", 1.0, 0.01, 10.0),
                number("min_hz", "Sweep start (Hz)", 20.0, 10.0, 1000.0),
                number("max_hz", "Sweep end (Hz)", 3000.0, 100.0, 10000.0),
                integer("points", "Frequency samples", 32, 8, 128),
                choice(
                    "enclosure",
                    "Rear enclosure",
                    &["free_air", "sealed", "vented"],
                ),
                number("volume_l", "Net chamber volume (litres)", 12.0, 0.1, 100.0),
            ];
            (
                "Driver Lab",
                "Electromechanical driver, enclosure and large-signal studies",
                vec![study(
                    "linear",
                    "Small signal",
                    "Native RMS electromechanical model with analytical baffled-piston radiation and lumped enclosure loading.",
                    common,
                )],
            )
        }
        "flow" => (
            "Flow Lab",
            "Steady incompressible air flow past a cylinder and Pironneau drag shape optimisation",
            vec![
                study(
                    "cylinder",
                    "Cylinder flow",
                    "Steady incompressible Navier-Stokes air flow past a circular cylinder in a channel (Schäfer–Turek geometry); drag from boundary traction integration.",
                    vec![
                        number(
                            "reynolds",
                            "Reynolds number (diameter, mean velocity)",
                            20.0,
                            5.0,
                            40.0,
                        ),
                        integer("nx", "Channel divisions in x", 64, 32, 96),
                        integer("ny", "Channel divisions in y", 12, 8, 20),
                    ],
                ),
                study(
                    "drag_opt",
                    "Drag shape optimisation",
                    "Pironneau fixed-area drag minimisation: the cylinder profile (even Fourier cosines) is deformed to minimise steady Navier-Stokes drag, starting from the circle.",
                    vec![
                        number(
                            "reynolds",
                            "Reynolds number (diameter, mean velocity)",
                            20.0,
                            5.0,
                            40.0,
                        ),
                        integer("n_modes", "Fourier modes", 3, 2, 4),
                        integer("nx", "Channel divisions in x", 48, 32, 96),
                        integer("ny", "Channel divisions in y", 10, 8, 20),
                        integer("outer_iterations", "Optimiser outer iterations", 8, 4, 20),
                    ],
                ),
            ],
        ),
        "structural" => (
            "Structure Lab",
            "Solved displacement, equilibrium and vibration response",
            vec![study(
                "bracket",
                "Bracket static / modal / harmonic",
                "Native plane-stress P1 FEM with assembled stiffness/mass, signed axial preload and damped harmonic response.",
                vec![
                    number("length", "Bracket length (m)", 0.18, 0.05, 0.5),
                    number("height", "Bracket height (m)", 0.036, 0.01, 0.1),
                    number("thickness", "Thickness (m)", 0.006, 0.001, 0.02),
                    number("force", "Vertical tip force (N)", -120.0, -500.0, 500.0),
                    number(
                        "preload",
                        "Axial preload: tension positive (N)",
                        250.0,
                        -1000.0,
                        1000.0,
                    ),
                    number("young_gpa", "Young's modulus (GPa)", 70.0, 1.0, 250.0),
                    number("density", "Density (kg/m³)", 2700.0, 500.0, 9000.0),
                    number("damping", "Modal damping ratio", 0.015, 0.001, 0.1),
                    integer("nx", "Length divisions", 24, 8, 48),
                    integer("ny", "Height divisions", 6, 3, 12),
                    integer("points", "Harmonic samples", 24, 8, 64),
                ],
            )],
        ),
        _ => return Err("Unknown simulator".into()),
    };
    let mut studies = studies;
    if simulator == "driver" {
        let common = studies[0].fields.clone();
        let mut thermal = common.clone();
        thermal.push(number(
            "thermal_resistance",
            "Coil thermal resistance (K/W)",
            3.0,
            0.1,
            20.0,
        ));
        studies.push(study("thermal", "Thermal compression", "Steady-state coil heating with iterated copper resistance and passive-load power balance.", thermal));
        let mut nonlinear = common;
        nonlinear.retain(|f| !["min_hz", "max_hz", "points"].contains(&f.key.as_str()));
        nonlinear.extend([
            number("tone", "Excitation tone (Hz)", 80.0, 30.0, 300.0),
            number("bl_drop", "BL drop at 5 mm (fraction)", 0.25, 0.0, 0.5),
            number("cubic", "Cubic suspension (N/m³)", 2e8, 0.0, 1e9),
        ]);
        studies.push(study("nonlinear", "Nonlinear transient", "Native lumped transient driver/enclosure model with displacement THD; waveform represents the solved circuit, not spatial port CFD.", nonlinear));
    }
    if simulator == "structural" {
        studies.push(study("cylinder", "Pressurized cylinder", "Axisymmetric P1 elasticity with 2πr integration and plane-strain Lamé boundary reference checks.", vec![
            number("inner_radius", "Inner radius (m)",0.03,0.01,0.1),number("outer_radius", "Outer radius (m)",0.05,0.015,0.15),number("cylinder_length","Cylinder length (m)",0.08,0.01,0.3),
            number("pressure", "Internal pressure (MPa)",1.0,0.001,10.0),number("young_gpa", "Young modulus (GPa)",70.0,1.0,250.0),integer("radial", "Radial divisions",24,8,64),integer("axial","Axial divisions",8,3,24)
        ]));
    }
    Ok(App {
        simulator: simulator.into(),
        title: title.into(),
        description: description.into(),
        studies,
    })
}
