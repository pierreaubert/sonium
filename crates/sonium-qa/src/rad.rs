//! OpenCourant comparison harness (contract C3).
//!
//! This module single-sources small solid-mechanics cases (mesh, SI units,
//! materials, boundary conditions, loads, probes) and renders them as
//! starter (`_0000.rad`) and engine (`_0001.rad`) decks, runs an available
//! engine as a subprocess, converts the binary time history through the
//! official `th_to_csv` tool, and compares probe channels against closed
//! forms. Card layouts follow the Altair Radioss Reference Guide block
//! format (fixed-width fields); engine verification is deferred until an
//! engine is available (see the worker report).
//!
//! Deck flow: starter reads `<case>_0000.rad` and writes `<case>_0000.rst`;
//! the engine reads `<case>_0001.rad` plus the restart and writes the `T01`
//! time history. Binary T01 wall channels are impulses, but `th_to_csv`
//! differentiates them like HyperGraph does, so the CSV already carries
//! forces (proven by energy balance on the hertz leg: `U = (2/5) F delta`);
//! engine decks request `/TH/TITLE` so `th_to_csv` emits full channel names.
//!
//! Phase B (ticket P1-B-W1-0) adds [`RadExtendedCase`]: shell elements with
//! `/PROP/SHELL` cards, quad/triangle shell meshing, plastic `/MAT/LAW2`
//! and `/MAT/LAW36` cards, `/GRAV` loading, and guide-shaped `/TH/SHEL`
//! plus `/TH/BRIC` element requests. The Phase-A [`OpenCourantCase`] writer is
//! frozen (other lanes assert byte-identical decks from it); every new
//! block follows the Altair Radioss Reference Guide block format with the
//! part identifier in the element-block header (`/SHELL/part_ID`,
//! `/BRICK/part_ID`) and variable cards on `/TH` requests.

// Rust guideline compliant 2026-02-21

/// Checked explicit contact surfaces for shell and solid reference fixtures.
#[path = "rad_contact_surface.rs"]
pub mod contact_surface;

/// Structured admission and results for the original native parity gates.
#[path = "rad_gate.rs"]
pub mod gate;

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

/// OpenCourant miniqa golden harvest + offline replay (Lane 0B).
///
/// Declared here so the crate root stays untouched; reachable as
/// `sonium_qa::rad::oc_golden`.
#[path = "oc_golden.rs"]
pub mod oc_golden;

/// One-command miniqa golden regen plus offline drift check (Phase-4
/// lane 4D).
#[path = "oc_regen.rs"]
pub mod oc_regen;

/// Failure of a OpenCourant harness operation.
#[derive(Debug)]
pub enum RadError {
    /// Filesystem or subprocess failure.
    Io(std::io::Error),
    /// A time-history file or probe table is malformed.
    Parse(String),
    /// The starter or engine subprocess reported failure.
    EngineFailed {
        /// Failing stage (`starter`, `engine`, or `th_to_csv`).
        stage: &'static str,
        /// Process exit code when available.
        code: Option<i32>,
        /// Tail of the combined output.
        output: String,
    },
    /// A requested time-history channel is absent.
    MissingChannel {
        /// Requested channel name.
        channel: String,
        /// Channels present in the file.
        available: Vec<String>,
    },
    /// A probe value misses its tolerance.
    ProbeMismatch {
        /// Probe label.
        label: String,
        /// Simulated value.
        actual: f64,
        /// Expected value.
        expected: f64,
        /// Absolute tolerance used.
        abs: f64,
        /// Relative tolerance used.
        rel: f64,
    },
}

impl fmt::Display for RadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "opencourant harness I/O failure: {error}"),
            Self::Parse(message) => write!(f, "opencourant harness parse failure: {message}"),
            Self::EngineFailed {
                stage,
                code,
                output,
            } => write!(f, "opencourant {stage} failed (code {code:?}): {output}"),
            Self::MissingChannel { channel, available } => {
                write!(
                    f,
                    "time history lacks channel {channel}; have {available:?}"
                )
            }
            Self::ProbeMismatch {
                label,
                actual,
                expected,
                abs,
                rel,
            } => write!(
                f,
                "probe {label} mismatch: actual {actual:.6e}, expected {expected:.6e} (abs {abs:.1e}, rel {rel:.1e})"
            ),
        }
    }
}

impl std::error::Error for RadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for RadError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

/// Elastic solid material rendered as `/MAT/LAW1`.
#[derive(Debug, Clone, Copy)]
pub struct RadElasticMaterial {
    /// Young's modulus in Pa.
    pub youngs_modulus_pa: f64,
    /// Poisson ratio, dimensionless.
    pub poisson_ratio: f64,
    /// Mass density in kg/m^3.
    pub density_kg_per_m3: f64,
}

/// Brick mesh with 1-based element connectivity.
#[derive(Debug, Clone)]
pub struct RadMesh {
    /// Node coordinates in metres, 0-based storage.
    pub nodes: Vec<[f64; 3]>,
    /// Eight-node brick connectivity, 1-based node ids.
    pub bricks: Vec<[usize; 8]>,
}

impl RadMesh {
    /// Count nodes of the mesh.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Append a free node (e.g. a rigid-wall driver) and return its 1-based id.
    pub fn add_node(&mut self, point_m: [f64; 3]) -> usize {
        self.nodes.push(point_m);
        self.nodes.len()
    }

    /// Count brick elements of the mesh.
    pub fn num_bricks(&self) -> usize {
        self.bricks.len()
    }

    /// Validate index ranges of every brick.
    ///
    /// # Errors
    ///
    /// Returns [`RadError::Parse`] when a brick references a missing node.
    pub fn validate(&self) -> Result<(), RadError> {
        for (index, brick) in self.bricks.iter().enumerate() {
            for node in brick {
                if *node == 0 || *node > self.nodes.len() {
                    return Err(RadError::Parse(format!(
                        "brick {} references missing node {node}",
                        index + 1
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Named node set rendered as `/GRNOD`.
#[derive(Debug, Clone)]
pub struct RadNodeSet {
    /// Set name used in deck comments.
    pub name: String,
    /// Member node ids, 1-based.
    pub nodes: Vec<usize>,
}

/// Clamped translational and rotational flags rendered as `/BCS`.
#[derive(Debug, Clone)]
pub struct RadBoundaryCondition {
    /// Name used in deck comments.
    pub name: String,
    /// Index into [`OpenCourantCase::node_sets`].
    pub set: usize,
    /// Locked `[Tx, Ty, Tz, Rx, Ry, Rz]` flags.
    pub lock: [bool; 6],
}

/// Piecewise-linear load curve rendered as `/FUNCT`.
#[derive(Debug, Clone)]
pub struct RadFunction {
    /// Curve name used in deck comments.
    pub name: String,
    /// Abscissa/ordinate pairs (time in s, scale factor).
    pub points: Vec<(f64, f64)>,
}

/// Concentrated nodal load rendered as `/CLOAD`.
#[derive(Debug, Clone)]
pub struct RadConcentratedLoad {
    /// Load name used in deck comments.
    pub name: String,
    /// Index into [`OpenCourantCase::node_sets`].
    pub set: usize,
    /// Load direction: 1..3 force X/Y/Z, 4..6 moment X/Y/Z.
    pub direction: usize,
    /// Index into [`OpenCourantCase::functions`].
    pub function: usize,
    /// Force scale in N.
    pub force_scale_n: f64,
}

/// Infinite-plane rigid wall rendered as `/RWALL/PLANE`.
#[derive(Debug, Clone)]
pub struct RadWall {
    /// Wall name used in deck comments.
    pub name: String,
    /// Outward unit normal of the wall plane.
    pub normal: [f64; 3],
    /// A point on the wall plane in metres.
    pub point_m: [f64; 3],
    /// Coulomb friction coefficient (0 = frictionless, selects Slide 0/2).
    pub friction: f64,
    /// Imposed wall velocity in m/s (`None` = fixed wall).
    pub velocity_m_per_s: Option<[f64; 3]>,
    /// 1-based driver node id for moving walls (`None` = fixed wall,
    /// rendered as node_ID 0 with an explicit M line). The driver node must
    /// exist in the mesh at [`RadWall::point_m`]; moving walls are rendered
    /// with Mass 0, which the reference guide defines as a constant imposed
    /// velocity (`VX0/VY0/VZ0`), so no `/IMPVEL` card is needed.
    pub driver_node: Option<usize>,
    /// Index into the case node sets attached as the secondary group
    /// (`grnd_ID1`; `None` = no group). Must exclude driver nodes.
    pub secondary_set: Option<usize>,
    /// Auto-capture distance in metres (`Dsearch`, 0 = group only).
    pub search_distance_m: f64,
}

/// A probe channel compared against a closed form.
#[derive(Debug, Clone)]
pub struct RadProbe {
    /// Probe label used in reports and `probes.json`.
    pub label: String,
    /// Time-history channel name, e.g. `DZ/123` or `FNZ/RWALL/2`.
    pub channel: String,
    /// Expected value in SI units.
    pub expected: f64,
    /// Absolute tolerance in SI units.
    pub abs_tol: f64,
    /// Relative tolerance, dimensionless.
    pub rel_tol: f64,
}

/// A single-sourced OpenCourant comparison case.
#[derive(Debug, Clone)]
pub struct OpenCourantCase {
    /// Case stem: decks are `<name>_0000.rad` / `<name>_0001.rad`.
    pub name: String,
    /// Title line written into both decks.
    pub title: String,
    /// Brick mesh in SI metres.
    pub mesh: RadMesh,
    /// Elastic solid material.
    pub material: RadElasticMaterial,
    /// Node sets referenced by BCs, loads, and outputs.
    pub node_sets: Vec<RadNodeSet>,
    /// Boundary conditions.
    pub bcs: Vec<RadBoundaryCondition>,
    /// Load curves.
    pub functions: Vec<RadFunction>,
    /// Concentrated loads.
    pub cloads: Vec<RadConcentratedLoad>,
    /// Rigid walls.
    pub walls: Vec<RadWall>,
    /// Node ids recorded with `/TH/NODE`.
    pub th_nodes: Vec<usize>,
    /// Probes compared after the run.
    pub probes: Vec<RadProbe>,
    /// Engine stop time in seconds.
    pub stop_time_s: f64,
    /// Time-history output period in seconds.
    pub th_period_s: f64,
    /// Animation output period in seconds.
    pub anim_period_s: f64,
}

/// Pre-rename alias kept for downstream users.
#[deprecated(since = "0.1.6", note = "renamed to `OpenCourantCase`")]
pub type RadCase = OpenCourantCase;

fn field_int(value: usize) -> String {
    format!("{value:>10}")
}

fn field_float(value: f64) -> String {
    format!("{value:>20.12E}")
}

/// Render one `/RWALL/PLANE` block (shared by the basic and extended writers).
///
/// Card order follows the reference guide: title, `node_ID/Slide/grnds`,
/// `Dsearch/fric`, then `M` (fixed wall) or `Mass/VX0/VY0/VZ0` (moving
/// wall), then the `M1 = M + normal` orientation point. Moving walls use
/// Mass 0, i.e. constant imposed velocity, so no `/IMPVEL` card is needed.
fn write_wall_block(deck: &mut String, index: usize, wall: &RadWall) {
    deck.push_str(&format!("/RWALL/PLANE/{}/1\n", index + 1));
    deck.push_str(&format!("{}\n", wall.name));
    let slide = if wall.friction > 0.0 { 2 } else { 0 };
    let grnd1 = wall.secondary_set.map_or(0, |set| set + 1);
    deck.push_str(&format!(
        "{}{}{}{}\n",
        field_int(wall.driver_node.unwrap_or(0)),
        field_int(slide),
        field_int(grnd1),
        field_int(0),
    ));
    deck.push_str(&format!(
        "{}{}{}{}{}\n",
        field_float(wall.search_distance_m),
        field_float(wall.friction),
        field_float(0.0),
        field_float(0.0),
        field_int(0),
    ));
    if wall.driver_node.is_some() {
        let velocity = wall.velocity_m_per_s.unwrap_or([0.0, 0.0, 0.0]);
        deck.push_str(&format!(
            "{}{}{}{}\n",
            field_float(0.0),
            field_float(velocity[0]),
            field_float(velocity[1]),
            field_float(velocity[2]),
        ));
    } else {
        deck.push_str(&format!(
            "{}{}{}\n",
            field_float(wall.point_m[0]),
            field_float(wall.point_m[1]),
            field_float(wall.point_m[2]),
        ));
    }
    deck.push_str(&format!(
        "{}{}{}\n",
        field_float(wall.point_m[0] + wall.normal[0]),
        field_float(wall.point_m[1] + wall.normal[1]),
        field_float(wall.point_m[2] + wall.normal[2]),
    ));
}

/// Render the starter deck (`<name>_0000.rad`) of a case.
///
/// Emits SI units, nodes, bricks, one elastic part, node sets, boundary
/// conditions, load curves, concentrated loads, rigid walls, and the
/// starter-side `/TH` requests.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{OpenCourantCase, cantilever_case, write_starter};
/// let case = cantilever_case();
/// let deck = write_starter(&case);
/// assert!(deck.contains("#RADIOSS STARTER"));
/// assert!(deck.contains("/END"));
/// ```
pub fn write_starter(case: &OpenCourantCase) -> String {
    let mut deck = String::new();
    deck.push_str("#RADIOSS STARTER\n");
    deck.push_str(&format!(
        "# sonium_qa rad case {} (SI units: kg, m, s)\n",
        case.name
    ));
    deck.push_str("/BEGIN\n");
    deck.push_str(&format!("{}\n", case.title));
    // Invers + input/work unit systems (mandatory /BEGIN cards).
    deck.push_str("      2022\n");
    deck.push_str("              kg                   m                   s\n");
    deck.push_str("              kg                   m                   s\n");
    deck.push_str("/UNIT/1\nSI mass-length-time\n");
    deck.push_str("              kg                   m                   s\n");
    deck.push_str("/NODE\n");
    for (index, node) in case.mesh.nodes.iter().enumerate() {
        deck.push_str(&format!(
            "{}{}{}{}\n",
            field_int(index + 1),
            field_float(node[0]),
            field_float(node[1]),
            field_float(node[2])
        ));
    }
    // Proven against the engine: part id lives in the header, not per row.
    deck.push_str("/BRICK/1\n");
    for (index, brick) in case.mesh.bricks.iter().enumerate() {
        let mut line = field_int(index + 1);
        for node in brick {
            line.push_str(&field_int(*node));
        }
        line.push('\n');
        deck.push_str(&line);
    }
    for (index, set) in case.node_sets.iter().enumerate() {
        deck.push_str(&format!("/GRNOD/NODE/{}/1\n", index + 1));
        deck.push_str(&format!("{}\n", set.name));
        for chunk in set.nodes.chunks(10) {
            let mut line = String::new();
            for node in chunk {
                line.push_str(&field_int(*node));
            }
            line.push('\n');
            deck.push_str(&line);
        }
    }
    deck.push_str("/MAT/LAW1/1/1\n");
    deck.push_str("linear elastic solid\n");
    // Proven against the engine: line 1 is density, line 2 is E and nu.
    deck.push_str(&format!(
        "{}\n",
        field_float(case.material.density_kg_per_m3)
    ));
    deck.push_str(&format!(
        "{}{}\n",
        field_float(case.material.youngs_modulus_pa),
        field_float(case.material.poisson_ratio)
    ));
    deck.push_str("/PROP/SOLID/1/1\nelastic brick property\n");
    // HEPH bricks + full geometric nonlinearities (best practice;
    // proven 1.6% vs EB on the coarse cantilever, Isolid=1 gave 41%).
    deck.push_str("        24         2\n");
    deck.push_str("/PART/1\nelastic part\n");
    deck.push_str("         1         1\n");
    for (index, function) in case.functions.iter().enumerate() {
        deck.push_str(&format!("/FUNCT/{}\n", index + 1));
        deck.push_str(&format!("{}\n", function.name));
        for (x, y) in &function.points {
            deck.push_str(&format!("{}{}\n", field_float(*x), field_float(*y)));
        }
    }
    for (index, bcs) in case.bcs.iter().enumerate() {
        deck.push_str(&format!("/BCS/{}/1\n", index + 1));
        deck.push_str(&format!("{}\n", bcs.name));
        // Proven against the engine + reference guide: one Trarot/Skew/grnd
        // line. Trarot packs the six lock flags as "TTT RRR" (1 = fixed),
        // right-justified in its field; skew 0 selects the global frame.
        let trarot = format!(
            "{}{}{} {}{}{}",
            i32::from(bcs.lock[0]),
            i32::from(bcs.lock[1]),
            i32::from(bcs.lock[2]),
            i32::from(bcs.lock[3]),
            i32::from(bcs.lock[4]),
            i32::from(bcs.lock[5]),
        );
        deck.push_str(&format!(
            "{:>10}{}{}\n",
            trarot,
            field_int(0),
            field_int(bcs.set + 1)
        ));
    }
    for (index, cload) in case.cloads.iter().enumerate() {
        deck.push_str(&format!("/CLOAD/{}/1\n", index + 1));
        deck.push_str(&format!("{}\n", cload.name));
        deck.push_str(&format!(
            "{}{:>10}{}{}{}{}{}{}\n",
            field_int(cload.function + 1),
            direction_code(cload.direction),
            field_int(0),
            field_int(0),
            field_int(cload.set + 1),
            field_int(0),
            field_float(1.0),
            field_float(cload.force_scale_n),
        ));
    }
    for (index, wall) in case.walls.iter().enumerate() {
        write_wall_block(&mut deck, index, wall);
    }
    // Proven against the engine: the variables line precedes the node ids.
    let mut th_group = 0;
    write_th_block(
        &mut deck,
        &mut th_group,
        "NODE",
        "time history nodes",
        &["DX".to_string(), "DY".to_string(), "DZ".to_string()],
        &case.th_nodes,
    );
    if !case.walls.is_empty() {
        // The FN group expands to FNX/FNY/FNZ; th_to_csv reports wall
        // forces directly (it differentiates the T01 impulses).
        let wall_ids: Vec<usize> = (1..=case.walls.len()).collect();
        write_th_block(
            &mut deck,
            &mut th_group,
            "RWALL",
            "time history walls",
            &["FN".to_string()],
            &wall_ids,
        );
    }
    deck.push_str("/END\n");
    deck
}

/// Render the engine deck (`<name>_0001.rad`) of a case.
///
/// Requests a quasi-static explicit run to `stop_time_s` with `/TFILE`
/// history output, `/TH/TITLE` full channel names for `th_to_csv`, and
/// animation states.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{OpenCourantCase, cantilever_case, write_engine};
/// let case = cantilever_case();
/// let deck = write_engine(&case);
/// assert!(deck.contains("#RADIOSS ENGINE"));
/// assert!(deck.contains("/TH/TITLE"));
/// ```
pub fn write_engine(case: &OpenCourantCase) -> String {
    // Proven against the engine: /RUN takes the bare stop time (no title,
    // no step count), /ANIM/DT takes start+period, and the deck ends after
    // the last card (no /STOP or /END terminator).
    let mut deck = String::new();
    deck.push_str("#RADIOSS ENGINE\n");
    deck.push_str(&format!("# engine controls for {}\n", case.name));
    deck.push_str(&format!("/RUN/{}/1\n", case.name));
    deck.push_str(&format!("{}\n", field_float(case.stop_time_s)));
    deck.push_str("/TFILE\n");
    deck.push_str(&format!("{}\n", field_float(case.th_period_s)));
    deck.push_str("/TH/TITLE\n");
    deck.push_str("/ANIM/DT\n");
    deck.push_str(&format!(
        "{}{}\n",
        field_float(0.0),
        field_float(case.anim_period_s)
    ));
    deck.push_str("/PRINT/-1000\n");
    deck
}

/// Build the quasi-static cantilever prove-out case.
///
/// A brick-meshed steel beam clamped at `x = 0` carries a ramped tip load;
/// the tip displacement probe replays the Euler-Bernoulli closed form.
pub fn cantilever_case() -> OpenCourantCase {
    let (mesh, root_set, tip_set, tip_node) = cantilever_brick_mesh(10, 2, 2, 1.0, 0.05, 0.05);
    let expected =
        super::beam_euler_cantilever_point(1.0, 1.0, 200.0e9, 0.05 * 0.05f64.powi(3) / 12.0, 100.0);
    // /CLOAD applies the scale to EACH node of the set.
    let tip_scale = 100.0 / tip_set.len() as f64;
    OpenCourantCase {
        name: "cantilever".to_string(),
        title: "sonium_qa rad cantilever quasi-static tip load".to_string(),
        mesh,
        material: RadElasticMaterial {
            youngs_modulus_pa: 200.0e9,
            poisson_ratio: 0.3,
            density_kg_per_m3: 7800.0,
        },
        node_sets: vec![
            RadNodeSet {
                name: "root".to_string(),
                nodes: root_set,
            },
            RadNodeSet {
                name: "tip".to_string(),
                nodes: tip_set,
            },
        ],
        bcs: vec![RadBoundaryCondition {
            name: "clamp root".to_string(),
            set: 0,
            lock: [true, true, true, true, true, true],
        }],
        functions: vec![RadFunction {
            name: "ramp 0 to 1 over stop time".to_string(),
            points: vec![(0.0, 0.0), (0.9, 0.9), (1.0, 1.0)],
        }],
        cloads: vec![RadConcentratedLoad {
            name: "tip load".to_string(),
            set: 1,
            direction: 3,
            function: 0,
            force_scale_n: tip_scale,
        }],
        walls: Vec::new(),
        th_nodes: vec![tip_node],
        probes: vec![RadProbe {
            label: "tip DZ".to_string(),
            channel: format!("DZ/{tip_node}"),
            expected,
            abs_tol: 1e-9,
            rel_tol: 0.05,
        }],
        stop_time_s: 1.0,
        th_period_s: 0.01,
        anim_period_s: 0.1,
    }
}

/// Build the Hertz sphere prove-out case.
///
/// An elastic ball rests on a fixed frictionless rigid plane while a top
/// plane presses it down at constant velocity. The case probe asserts the
/// pole kinematics (the top pole rides the imposed-velocity wall); the
/// force leg lives in the QA binary as custom checks (wall-force balance
/// from Newton's 3rd law plus a contact-stiffness window), because the
/// Hertz patch `a = sqrt(R delta) ~ 0.5 mm` is unresolvable on 10 mm
/// bricks, so the Hertz force magnitude itself is out of reach here.
/// Quasi-static Hertz is duration-independent, and the 1 s run grows a
/// contact-mode instability past ~150k cycles, so the press stops at 0.1 s
/// (137k cycles) where the parasitic kinetic energy is still 1e-7 J.
pub fn hertz_sphere_case() -> OpenCourantCase {
    let radius = 0.05;
    let mut mesh = sphere_ball_mesh(10, radius);
    let press_speed = 1.0e-4;
    let stop_time = 0.1;
    // Total top-wall travel; the top pole rides the wall down by this.
    let travel = press_speed * stop_time;
    let top_pole = mesh
        .nodes
        .iter()
        .enumerate()
        .max_by(|a, b| a.1[2].total_cmp(&b.1[2]))
        .map(|(index, _)| index + 1)
        .unwrap_or(1);
    // Split secondary groups (bottom/top halves): a node secondary to
    // both walls trips starter warning 312 (incompatible kinematic
    // conditions). Contact only touches the poles, so the split is exact.
    // The driver node is appended after so it stays out of both groups.
    let mut bottom_nodes = Vec::new();
    let mut top_nodes = Vec::new();
    for (index, node) in mesh.nodes.iter().enumerate() {
        if node[2] <= radius {
            bottom_nodes.push(index + 1);
        } else {
            top_nodes.push(index + 1);
        }
    }
    let top_point = [0.0, 0.0, 2.0 * radius];
    let driver = mesh.add_node(top_point);
    OpenCourantCase {
        name: "hertz_sphere".to_string(),
        title: "sonium_qa rad hertz sphere between rigid planes".to_string(),
        mesh,
        material: RadElasticMaterial {
            youngs_modulus_pa: 200.0e9,
            poisson_ratio: 0.3,
            density_kg_per_m3: 7800.0,
        },
        node_sets: vec![
            RadNodeSet {
                name: "ball bottom".to_string(),
                nodes: bottom_nodes,
            },
            RadNodeSet {
                name: "ball top".to_string(),
                nodes: top_nodes,
            },
        ],
        bcs: Vec::new(),
        functions: vec![RadFunction {
            name: "constant press velocity".to_string(),
            points: vec![(0.0, 1.0), (stop_time, 1.0)],
        }],
        cloads: Vec::new(),
        walls: vec![
            RadWall {
                name: "fixed bottom plane".to_string(),
                normal: [0.0, 0.0, 1.0],
                point_m: [0.0, 0.0, 0.0],
                friction: 0.0,
                velocity_m_per_s: None,
                driver_node: None,
                secondary_set: Some(0),
                search_distance_m: 0.0,
            },
            RadWall {
                name: "moving top plane".to_string(),
                normal: [0.0, 0.0, -1.0],
                point_m: top_point,
                friction: 0.0,
                velocity_m_per_s: Some([0.0, 0.0, -press_speed]),
                driver_node: Some(driver),
                secondary_set: Some(1),
                search_distance_m: 0.0,
            },
        ],
        th_nodes: vec![top_pole],
        probes: vec![RadProbe {
            label: "top pole DZ".to_string(),
            channel: format!("DZ/{top_pole}"),
            expected: -travel,
            abs_tol: 1e-9,
            rel_tol: 0.10,
        }],
        stop_time_s: stop_time,
        th_period_s: 0.001,
        anim_period_s: 0.025,
    }
}

/// Mesh a cantilever beam with 8-node bricks.
///
/// Returns the mesh, the root/tip node sets, and the tip-center node id.
/// Node ids are 1-based.
pub fn cantilever_brick_mesh(
    nx: usize,
    ny: usize,
    nz: usize,
    length_m: f64,
    width_m: f64,
    height_m: f64,
) -> (RadMesh, Vec<usize>, Vec<usize>, usize) {
    let mut nodes = Vec::with_capacity((nx + 1) * (ny + 1) * (nz + 1));
    let id = |i: usize, j: usize, k: usize| i * (ny + 1) * (nz + 1) + j * (nz + 1) + k + 1;
    for i in 0..=nx {
        for j in 0..=ny {
            for k in 0..=nz {
                nodes.push([
                    length_m * i as f64 / nx as f64,
                    width_m * (j as f64 / ny as f64 - 0.5),
                    height_m * (k as f64 / nz as f64 - 0.5),
                ]);
            }
        }
    }
    let mut bricks = Vec::with_capacity(nx * ny * nz);
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                bricks.push([
                    id(i, j, k),
                    id(i + 1, j, k),
                    id(i + 1, j + 1, k),
                    id(i, j + 1, k),
                    id(i, j, k + 1),
                    id(i + 1, j, k + 1),
                    id(i + 1, j + 1, k + 1),
                    id(i, j + 1, k + 1),
                ]);
            }
        }
    }
    let mut root_set = Vec::new();
    let mut tip_set = Vec::new();
    for j in 0..=ny {
        for k in 0..=nz {
            root_set.push(id(0, j, k));
            tip_set.push(id(nx, j, k));
        }
    }
    let tip_node = id(nx, ny / 2, nz / 2);
    (RadMesh { nodes, bricks }, root_set, tip_set, tip_node)
}

/// Mesh a solid ball with 8-node bricks.
///
/// Keeps the Cartesian cells fully inside the sphere and projects the
/// region boundary radially onto the surface, so every brick keeps a
/// positive volume. The ball is centered at `(0, 0, radius)`.
pub fn sphere_ball_mesh(cells_per_side: usize, radius_m: f64) -> RadMesh {
    let n = cells_per_side.max(2);
    let step = 2.0 * radius_m / n as f64;
    let corner = |i: usize| -radius_m + step * i as f64;
    let inside = |i: usize, j: usize, k: usize| {
        let (x, y, z) = (corner(i), corner(j), corner(k));
        x * x + y * y + z * z <= radius_m * radius_m
    };
    let mut kept = vec![false; n * n * n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let all_inside = (0..=1)
                    .all(|di| (0..=1).all(|dj| (0..=1).all(|dk| inside(i + di, j + dj, k + dk))));
                kept[(i * n + j) * n + k] = all_inside;
            }
        }
    }
    // Boundary nodes touch at least one dropped cell; project them outward.
    let mut node_id = vec![0usize; (n + 1) * (n + 1) * (n + 1)];
    let mut nodes = Vec::new();
    let mut is_boundary = Vec::new();
    for i in 0..=n {
        for j in 0..=n {
            for k in 0..=n {
                let mut used = false;
                let mut boundary = false;
                for di in 0..=1 {
                    for dj in 0..=1 {
                        for dk in 0..=1 {
                            let (ci, cj, ck) = (i + 1 - di, j + 1 - dj, k + 1 - dk);
                            if ci == 0 || cj == 0 || ck == 0 || ci > n || cj > n || ck > n {
                                continue;
                            }
                            if kept[((ci - 1) * n + (cj - 1)) * n + (ck - 1)] {
                                used = true;
                            } else {
                                boundary = true;
                            }
                        }
                    }
                }
                // Outer-grid nodes of kept cells are boundary by construction.
                if i == 0 || j == 0 || k == 0 || i == n || j == n || k == n {
                    boundary = true;
                }
                if !used {
                    continue;
                }
                let (mut x, mut y, mut z) = (corner(i), corner(j), corner(k));
                if boundary {
                    let r = (x * x + y * y + z * z).sqrt();
                    if r > 1e-12 {
                        let scale = radius_m / r;
                        x *= scale;
                        y *= scale;
                        z *= scale;
                    }
                }
                node_id[(i * (n + 1) + j) * (n + 1) + k] = nodes.len() + 1;
                nodes.push([x, y, z + radius_m]);
                is_boundary.push(boundary);
            }
        }
    }
    let _ = is_boundary;
    let mut bricks = Vec::new();
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                if !kept[(i * n + j) * n + k] {
                    continue;
                }
                let at = |di: usize, dj: usize, dk: usize| {
                    node_id[((i + di) * (n + 1) + (j + dj)) * (n + 1) + (k + dk)]
                };
                bricks.push([
                    at(0, 0, 0),
                    at(1, 0, 0),
                    at(1, 1, 0),
                    at(0, 1, 0),
                    at(0, 0, 1),
                    at(1, 0, 1),
                    at(1, 1, 1),
                    at(0, 1, 1),
                ]);
            }
        }
    }
    RadMesh { nodes, bricks }
}

/// Resolve the engine executable from the environment.
///
/// Returns `OPENCOURANT_EXE` when set, else legacy `RADISS_EXE`, else
/// `None` so binaries can report the documented skip message and exit 0.
pub fn engine_exe() -> Option<PathBuf> {
    for name in ["OPENCOURANT_EXE", "RADISS_EXE"] {
        if let Ok(path) = std::env::var(name) {
            let trimmed = path.trim();
            if !trimmed.is_empty() {
                return Some(PathBuf::from(trimmed));
            }
        }
    }
    None
}

/// Names the engine flavor behind an executable path.
pub fn engine_flavor(_exe: &Path) -> &'static str {
    // All supported engine paths currently resolve to OpenCourant.
    "OpenCourant"
}

/// Rank a starter/engine candidate filename (lower is better).
///
/// Prefers double-precision SMP builds (`*_linux64_gf` without `_sp` or
/// `ompi` suffixes); single-precision before MPI; anything else last.
fn candidate_rank(file_name: &str) -> u32 {
    let exact_dp = file_name == "starter_linux64_gf" || file_name == "engine_linux64_gf";
    if exact_dp {
        return 0;
    }
    let lower = file_name.to_lowercase();
    let is_sp = lower.contains("_sp");
    // MPI builds need mpirun; SMP/OpenMP builds run directly.
    let is_mpi = lower.contains("ompi") || lower.contains("_mpi");
    match (is_sp, is_mpi) {
        (false, false) => 1,
        (true, false) => 2,
        (false, true) => 3,
        (true, true) => 4,
    }
}

/// Best `prefix*` executable entry of `dir` by [`candidate_rank`].
pub(crate) fn best_sibling(dir: &Path, prefix: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut best: Option<(u32, String, PathBuf)> = None;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.starts_with(prefix) {
            continue;
        }
        let rank = candidate_rank(&name);
        let replace = best
            .as_ref()
            .map(|(best_rank, best_name, _)| (rank, &name) < (*best_rank, best_name))
            .unwrap_or(true);
        if replace {
            best = Some((rank, name, entry.path()));
        }
    }
    best.map(|(_, _, path)| path)
}

/// Resolve the starter/engine binary pair behind one configured path.
///
/// OpenCourant ships separate `starter_*` and `engine_*` binaries,
/// so one configured path cannot serve both stages directly. Accepts:
/// - a directory (e.g. `.../exec`): picks the best `starter_*`/`engine_*`
///   pair, preferring double-precision SMP builds;
/// - a `starter_*` binary: pairs it with the same-suffix `engine_*` sibling;
/// - an `engine_*` binary: pairs it with the same-suffix `starter_*`
///   sibling;
/// - anything else: uses the path for both stages (legacy single-binary).
///
/// # Errors
///
/// Returns [`RadError::Parse`] when a directory holds no usable pair.
pub fn resolve_engine_pair(exe: &Path) -> Result<(PathBuf, PathBuf), RadError> {
    if exe.is_dir() {
        let starter = best_sibling(exe, "starter");
        let engine = best_sibling(exe, "engine");
        match (starter, engine) {
            (Some(starter), Some(engine)) => return Ok((starter, engine)),
            _ => {
                return Err(RadError::Parse(format!(
                    "engine directory {} holds no starter_*/engine_* pair",
                    exe.display()
                )));
            }
        }
    }
    let file_name = exe
        .file_name()
        .and_then(|stem| stem.to_str())
        .unwrap_or_default();
    let parent = exe.parent();
    if file_name.starts_with("starter") {
        let sibling = file_name.replacen("starter", "engine", 1);
        if let Some(parent) = parent {
            let candidate = parent.join(&sibling);
            if candidate.is_file() {
                return Ok((exe.to_path_buf(), candidate));
            }
            if let Some(engine) = best_sibling(parent, "engine") {
                return Ok((exe.to_path_buf(), engine));
            }
        }
        return Ok((exe.to_path_buf(), exe.to_path_buf()));
    }
    if file_name.starts_with("engine") {
        let sibling = file_name.replacen("engine", "starter", 1);
        if let Some(parent) = parent {
            let candidate = parent.join(&sibling);
            if candidate.is_file() {
                return Ok((candidate, exe.to_path_buf()));
            }
            if let Some(starter) = best_sibling(parent, "starter") {
                return Ok((starter, exe.to_path_buf()));
            }
        }
        return Ok((exe.to_path_buf(), exe.to_path_buf()));
    }
    Ok((exe.to_path_buf(), exe.to_path_buf()))
}

/// Output of a starter-plus-engine run.
#[derive(Debug)]
pub struct RunOutput {
    /// Engine flavor that ran.
    pub engine: String,
    /// Converted time-history table.
    pub history: TimeHistory,
    /// Path of the converted CSV (or native T01 when already CSV).
    pub history_path: PathBuf,
}

/// Run the starter then the engine for a checked-in deck pair.
///
/// Both stages run with the working directory set to `decks_dir`; after a
/// successful engine run the `T01` file is converted with `th_to_csv`
/// (resolved beside the engine or on `PATH`) and parsed. Native CSV
/// histories are parsed directly. The `case` argument single-sources the
/// deck stems; C3 contract note: the frozen text sketches
/// `run_case(decks_dir, exe)`, but the stem must come from somewhere, so
/// the shipped form takes the case explicitly (owner clarification).
///
/// # Errors
///
/// Returns [`RadError`] for failed stages, explicit numerical aborts even with
/// a zero process exit code, unavailable binary conversion, malformed histories,
/// or histories that do not cover the original requested stop time.
pub fn run_case(
    decks_dir: &Path,
    exe: &Path,
    case: &OpenCourantCase,
) -> Result<RunOutput, RadError> {
    let starter_deck = decks_dir.join(format!("{}_0000.rad", case.name));
    let engine_deck = decks_dir.join(format!("{}_0001.rad", case.name));
    let (starter_exe, engine_exe) = resolve_engine_pair(exe)?;
    run_stage(&starter_exe, "starter", &starter_deck, decks_dir)?;
    run_stage(&engine_exe, "engine", &engine_deck, decks_dir)?;
    let t01 = decks_dir.join(format!("{}T01", case.name.to_uppercase()));
    let t01 = if t01.exists() {
        t01
    } else {
        decks_dir.join(format!("{}T01", case.name))
    };
    let csv = t01.with_extension("csv");
    if !is_probably_csv(&t01)? {
        convert_with_th_to_csv(exe, &t01, decks_dir)?;
    }
    let history_path = if csv.exists() { csv } else { t01.clone() };
    let history = parse_t01(&history_path)?;
    validate_history_extent(&history, case)?;
    Ok(RunOutput {
        engine: engine_flavor(exe).to_string(),
        history,
        history_path,
    })
}

fn run_stage(exe: &Path, stage: &'static str, deck: &Path, dir: &Path) -> Result<(), RadError> {
    let output = Command::new(exe)
        .arg("-i")
        .arg(deck)
        .current_dir(dir)
        // The engine blocks on stdin after abnormal ends; never inherit it.
        .stdin(std::process::Stdio::null())
        .output()?;
    if !output.status.success() {
        let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
        combined.push_str(&String::from_utf8_lossy(&output.stderr));
        let tail: String = combined
            .chars()
            .rev()
            .take(2000)
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        return Err(RadError::EngineFailed {
            stage,
            code: output.status.code(),
            output: tail,
        });
    }
    let mut log = String::from_utf8_lossy(&output.stdout).into_owned();
    log.push_str(&String::from_utf8_lossy(&output.stderr));
    let written_log = deck.with_extension("out");
    if written_log.exists() {
        log.push_str(&std::fs::read_to_string(written_log)?);
    }
    validate_stage_log(stage, output.status.code(), &log)?;
    Ok(())
}

fn validate_stage_log(stage: &'static str, code: Option<i32>, log: &str) -> Result<(), RadError> {
    if let Some(reason) = log.lines().find(|line| {
        ["RUN KILLED:", "ERROR TERMINATION", "FATAL ERROR"]
            .iter()
            .any(|marker| line.contains(marker))
    }) {
        return Err(RadError::EngineFailed {
            stage,
            code,
            output: format!(
                "OpenCourant numerical abort despite process exit success: {}",
                reason.trim()
            ),
        });
    }
    Ok(())
}

fn validate_history_extent(history: &TimeHistory, case: &OpenCourantCase) -> Result<(), RadError> {
    if !case.stop_time_s.is_finite()
        || case.stop_time_s <= 0.0
        || !case.th_period_s.is_finite()
        || case.th_period_s <= 0.0
    {
        return Err(RadError::Parse(
            "finite positive stop time and history period required".into(),
        ));
    }
    if history.times.len() < 2 || history.times.iter().any(|t| !t.is_finite()) {
        return Err(RadError::Parse(
            "external history needs at least two finite time samples".into(),
        ));
    }
    let first = history.times[0];
    let last = *history.times.last().unwrap();
    let tolerance = 1.05 * case.th_period_s + 1e-8 * case.stop_time_s;
    if first < -tolerance || first > tolerance || (last - case.stop_time_s).abs() > tolerance {
        return Err(RadError::Parse(format!(
            "external history [{first:.9e},{last:.9e}] does not cover original 0..{:.9e} s history (output period {:.9e} s)",
            case.stop_time_s, case.th_period_s
        )));
    }
    if history.times.windows(2).any(|pair| pair[1] < pair[0]) {
        return Err(RadError::Parse(
            "external history timestamps go backwards".into(),
        ));
    }
    Ok(())
}

fn is_probably_csv(path: &Path) -> Result<bool, RadError> {
    let bytes = std::fs::read(path)?;
    let head = &bytes[..bytes.len().min(4096)];
    Ok(!head.contains(&0) && head.contains(&b','))
}

/// Input argument for a `th_to_csv` conversion run.
///
/// Vendor defect workaround: `th_to_csv` SIGFPEs on input paths longer
/// than its fixed filename buffer (observed: 71 chars fine, 108 chars
/// fatal with OpenCourant CommitID 33e68517). The tool already runs with
/// `dir` as its working directory, so the bare file name is passed
/// whenever the T01 lives there; absolute paths are kept only for
/// out-of-directory inputs.
pub(crate) fn th_to_csv_arg(t01: &Path, dir: &Path) -> PathBuf {
    match (t01.file_name(), t01.parent()) {
        (Some(name), Some(parent)) if parent == dir => PathBuf::from(name),
        _ => t01.to_path_buf(),
    }
}

pub(crate) fn convert_with_th_to_csv(exe: &Path, t01: &Path, dir: &Path) -> Result<(), RadError> {
    // `exe` may be the exec directory itself or one engine binary; the
    // converter ships as `th_to_csv*` beside the engine binaries.
    let search_dir = if exe.is_dir() {
        Some(exe.to_path_buf())
    } else {
        exe.parent().map(Path::to_path_buf)
    };
    let mut candidates = Vec::new();
    if let Some(search_dir) = search_dir {
        candidates.push(search_dir.join("th_to_csv"));
        if let Some(suffixed) = best_sibling(&search_dir, "th_to_csv")
            && suffixed.file_name().map(|name| name != "th_to_csv") == Some(true)
        {
            candidates.push(suffixed);
        }
    }
    candidates.push(PathBuf::from("th_to_csv"));
    let arg = th_to_csv_arg(t01, dir);
    let mut spawn_error = None;
    let mut exec_error = None;
    for tool in candidates {
        match Command::new(&tool).arg(&arg).current_dir(dir).output() {
            Ok(output) if output.status.success() => return Ok(()),
            Ok(output) => {
                // The tool ran but failed; this beats path misses
                // below. File name only: this text can land in
                // checked-in golden notes, which must not embed
                // host-specific absolute paths. No exit code means a
                // signal death (Unix reports `None` there).
                let outcome = output
                    .status
                    .code()
                    .map_or("killed by signal".to_string(), |code| {
                        format!("exited {code}")
                    });
                exec_error = Some(format!(
                    "{} {outcome}",
                    tool.file_name().unwrap_or_default().to_string_lossy()
                ));
            }
            Err(error) => spawn_error = Some(error.to_string()),
        }
    }
    Err(RadError::Parse(format!(
        "binary T01 needs th_to_csv conversion: {}",
        exec_error
            .or(spawn_error)
            .unwrap_or_else(|| "converter missing".to_string())
    )))
}

/// A parsed time history: time grid plus named channels.
#[derive(Debug, Clone, Default)]
pub struct TimeHistory {
    /// Time stamps in seconds.
    pub times: Vec<f64>,
    /// Channel name to samples, one per time stamp.
    pub channels: BTreeMap<String, Vec<f64>>,
}

impl TimeHistory {
    /// Look up a channel by name.
    ///
    /// # Errors
    ///
    /// Returns [`RadError::MissingChannel`] when the name is absent.
    pub fn channel(&self, name: &str) -> Result<&[f64], RadError> {
        self.channels
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| RadError::MissingChannel {
                channel: name.to_string(),
                available: self.channels.keys().cloned().collect(),
            })
    }

    /// Final sample of a channel.
    ///
    /// # Errors
    ///
    /// Returns [`RadError`] when the channel is absent or empty.
    pub fn final_value(&self, name: &str) -> Result<f64, RadError> {
        let values = self.channel(name)?;
        values
            .last()
            .copied()
            .ok_or_else(|| RadError::Parse(format!("channel {name} has no samples")))
    }

    /// Linearly interpolate a channel at a time within the grid.
    ///
    /// # Errors
    ///
    /// Returns [`RadError`] when the channel is absent, empty, or the time
    /// falls outside the grid.
    pub fn value_at(&self, name: &str, time: f64) -> Result<f64, RadError> {
        let values = self.channel(name)?;
        if self.times.len() != values.len() || values.is_empty() {
            return Err(RadError::Parse(format!("channel {name} is malformed")));
        }
        if time < self.times[0] || time > self.times[self.times.len() - 1] {
            return Err(RadError::Parse(format!(
                "time {time} is outside [{}..{}]",
                self.times[0],
                self.times[self.times.len() - 1]
            )));
        }
        let upper = self
            .times
            .partition_point(|grid| *grid <= time)
            .min(values.len() - 1);
        if upper == 0 {
            return Ok(values[0]);
        }
        let (t0, t1) = (self.times[upper - 1], self.times[upper]);
        if t1 <= t0 {
            return Ok(values[upper]);
        }
        let weight = (time - t0) / (t1 - t0);
        Ok(values[upper - 1] * (1.0 - weight) + values[upper] * weight)
    }
}

/// Parse a `th_to_csv` time-history CSV file.
///
/// Expects a header row whose first column is time (any name containing
/// `time`, case-insensitive) followed by one column per channel. Binary
/// T01 input is rejected with guidance to convert first.
///
/// # Errors
///
/// Returns [`RadError::Parse`] for binary input, malformed rows, or
/// non-finite samples.
pub fn parse_t01(path: &Path) -> Result<TimeHistory, RadError> {
    let text = std::fs::read(path)?;
    if text.contains(&0) {
        return Err(RadError::Parse(format!(
            "binary T01 at {} needs th_to_csv conversion first",
            path.display()
        )));
    }
    let text = String::from_utf8(text)
        .map_err(|_| RadError::Parse(format!("{} is not valid UTF-8", path.display())))?;
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let header = lines
        .next()
        .ok_or_else(|| RadError::Parse(format!("{} is empty", path.display())))?;
    let names: Vec<String> = header
        .split(',')
        .map(|cell| cell.trim().to_string())
        .collect();
    if names.len() < 2 || !names[0].to_lowercase().contains("time") {
        return Err(RadError::Parse(format!(
            "{} header must start with a time column",
            path.display()
        )));
    }
    let mut times = Vec::new();
    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); names.len() - 1];
    for (row, line) in lines.enumerate() {
        let cells: Vec<&str> = line.split(',').collect();
        if cells.len() != names.len() {
            return Err(RadError::Parse(format!(
                "{} row {} has {} cells, expected {}",
                path.display(),
                row + 2,
                cells.len(),
                names.len()
            )));
        }
        let time: f64 = cells[0].trim().parse().map_err(|_| {
            RadError::Parse(format!(
                "{} row {} has a bad time stamp",
                path.display(),
                row + 2
            ))
        })?;
        if !time.is_finite() {
            return Err(RadError::Parse(format!(
                "{} row {} time stamp is non-finite",
                path.display(),
                row + 2
            )));
        }
        times.push(time);
        for (column, cell) in cells.iter().skip(1).enumerate() {
            let value: f64 = cell.trim().parse().map_err(|_| {
                RadError::Parse(format!(
                    "{} row {} column {} is not a number",
                    path.display(),
                    row + 2,
                    names[column + 1]
                ))
            })?;
            if !value.is_finite() {
                return Err(RadError::Parse(format!(
                    "{} row {} column {} is non-finite",
                    path.display(),
                    row + 2,
                    names[column + 1]
                )));
            }
            columns[column].push(value);
        }
    }
    let mut channels = BTreeMap::new();
    for (index, name) in names.iter().skip(1).enumerate() {
        channels.insert(normalize_channel(name), std::mem::take(&mut columns[index]));
    }
    Ok(TimeHistory { times, channels })
}

/// Normalize a `th_to_csv` column header to probe-channel form.
///
/// Element channels arrive as `"<title> <id> <VAR>"` (e.g.
/// `"time history nodes  95  DX"`); wall channels carry the wall title
/// between the id and the variable (`"time history walls  1  fixed
/// bottom plane  FNZ"`). Probes use `VAR/id` for nodes and `VAR/TYPE/id`
/// for shells (`SHEL`), bricks (`BRIC`), and walls (`RWALL`), with the
/// type recovered from the title keywords our writer emits.
/// Energy channels (`INTERNAL ENERGY`, ...) match no element pattern and
/// pass through with quotes stripped.
fn normalize_channel(raw: &str) -> String {
    let unquoted = raw.trim().trim_matches('"').trim().to_string();
    let tokens: Vec<&str> = unquoted.split_whitespace().collect();
    if tokens.len() >= 3 {
        let var = tokens[tokens.len() - 1];
        let var_ok = !var.is_empty() && var.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
        if var_ok {
            let head = &tokens[..tokens.len() - 1];
            let title = head.join(" ").to_lowercase();
            if title.contains("wall") {
                // Wall titles carry no digits, so the first integer token
                // is the wall id.
                if let Some(id) = head
                    .iter()
                    .find(|token| token.chars().all(|c| c.is_ascii_digit()))
                {
                    return format!("{var}/RWALL/{id}");
                }
            }
            let id = tokens[tokens.len() - 2];
            let id_ok = !id.is_empty() && id.chars().all(|c| c.is_ascii_digit());
            if id_ok {
                if title.contains("shell") {
                    return format!("{var}/SHEL/{id}");
                }
                if title.contains("brick") {
                    return format!("{var}/BRIC/{id}");
                }
                if title.contains("node") {
                    return format!("{var}/{id}");
                }
            }
        }
    }
    unquoted
}

/// Differentiate an impulse channel into a force channel.
///
/// Raw binary T01 records running impulses `J(t) = int F dt`; centered
/// differences recover the force at interior stamps (one-sided at the
/// ends). Not used on `th_to_csv` output, which already carries forces.
pub fn impulse_to_force(times: &[f64], impulse: &[f64]) -> Vec<f64> {
    let n = times.len().min(impulse.len());
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0.0];
    }
    let mut force = vec![0.0; n];
    force[0] = slope(times[0], impulse[0], times[1], impulse[1]);
    for i in 1..n - 1 {
        force[i] = slope(times[i - 1], impulse[i - 1], times[i + 1], impulse[i + 1]);
    }
    force[n - 1] = slope(times[n - 2], impulse[n - 2], times[n - 1], impulse[n - 1]);
    force
}

fn slope(t0: f64, j0: f64, t1: f64, j1: f64) -> f64 {
    if t1 > t0 { (j1 - j0) / (t1 - t0) } else { 0.0 }
}

/// Compare a probe value against its expectation.
///
/// Passes when `|actual - expected| <= abs + rel * |expected|`.
///
/// # Errors
///
/// Returns [`RadError::ProbeMismatch`] on missing tolerance or
/// non-finite inputs.
pub fn assert_probe_close(
    actual: f64,
    expected: f64,
    abs: f64,
    rel: f64,
    label: &str,
) -> Result<(), RadError> {
    if !actual.is_finite() || !expected.is_finite() || !abs.is_finite() || !rel.is_finite() {
        return Err(RadError::ProbeMismatch {
            label: label.to_string(),
            actual,
            expected,
            abs,
            rel,
        });
    }
    if (actual - expected).abs() <= abs + rel * expected.abs() {
        Ok(())
    } else {
        Err(RadError::ProbeMismatch {
            label: label.to_string(),
            actual,
            expected,
            abs,
            rel,
        })
    }
}

/// Compare two `probes.json` texts ignoring the `engine` stamp.
///
/// Prove-out stamps the producing engine into the checked-in file, so
/// the drift check compares the probe payload, not the bytes.
///
/// # Panics
///
/// Panics with `context` when either side is not valid JSON or the
/// payloads differ.
pub fn assert_probes_payload_eq(checked: &str, expected: &str, context: &str) {
    let mut checked_json: serde_json::Value =
        serde_json::from_str(checked).unwrap_or_else(|error| {
            panic!("{context}: checked probes.json is not valid JSON: {error}")
        });
    let mut expected_json: serde_json::Value = serde_json::from_str(expected)
        .unwrap_or_else(|error| panic!("{context}: writer probes.json is not valid JSON: {error}"));
    for value in [&mut checked_json, &mut expected_json] {
        if let Some(object) = value.as_object_mut() {
            object.remove("engine");
        }
    }
    assert_eq!(checked_json, expected_json, "{context}");
}

/// Render the checked-in `probes.json` artifact of a case.
///
/// Records expected probe values, tolerances, and the producing engine
/// version (`"none (prove-out deferred)"` until an engine runs).
pub fn probes_json(case: &OpenCourantCase, engine_version: &str) -> String {
    let mut out = String::from("{\n");
    out.push_str(&format!("  \"case\": \"{}\",\n", case.name));
    out.push_str(&format!("  \"engine\": \"{engine_version}\",\n"));
    out.push_str("  \"units\": \"SI\",\n");
    out.push_str("  \"probes\": [\n");
    for (index, probe) in case.probes.iter().enumerate() {
        out.push_str("    {\n");
        out.push_str(&format!("      \"label\": \"{}\",\n", probe.label));
        out.push_str(&format!("      \"channel\": \"{}\",\n", probe.channel));
        out.push_str(&format!("      \"expected\": {:.12e},\n", probe.expected));
        out.push_str(&format!("      \"abs_tol\": {:.3e},\n", probe.abs_tol));
        out.push_str(&format!("      \"rel_tol\": {:.3e}\n", probe.rel_tol));
        out.push_str(if index + 1 == case.probes.len() {
            "    }\n"
        } else {
            "    },\n"
        });
    }
    out.push_str("  ]\n}\n");
    out
}

// =====================================================================
// Phase-B extensions (P1-B-W1-0). Additive: nothing above this line
// changes behavior or writer output.
// =====================================================================

/// Shell property set rendered as `/PROP/SHELL` (TYPE1).
///
/// Field names and defaults follow the Reference Guide `/PROP/TYPE1`
/// page: `Ishell` selects the 4-node formulation, `Ismstr` the
/// small-strain switch, `Ish3n` the 3-node formulation, `N` the
/// through-thickness integration count, and `Thick` the thickness.
#[derive(Debug, Clone)]
pub struct RadShellProperty {
    /// Property title written on the card.
    pub name: String,
    /// Four-node formulation flag (24 = QEPH, recommended general use).
    pub ishell: i32,
    /// Small-strain flag (2 = full geometric nonlinearities).
    pub ismstr: i32,
    /// Three-node formulation flag (2 = C0 large rotation).
    pub ish3n: i32,
    /// Drilling stiffness flag (2 = off).
    pub idrill: i32,
    /// Through-thickness integration points, 1..10 (5 recommended).
    pub n_thickness: i32,
    /// Shell thickness in metres.
    pub thickness_m: f64,
    /// Thickness-change flag (2 = constant thickness).
    pub ithick: i32,
    /// Plane-stress plasticity flag (2 = radial return).
    pub iplas: i32,
}

impl RadShellProperty {
    /// QEPH shell property with best-practice defaults.
    ///
    /// Uses Ishell 24 (QEPH), Ismstr 2, Ish3n 2, Idrill 2, N 5,
    /// constant thickness, and radial-return plasticity per the
    /// Reference Guide and the OpenCourant modeling best practices.
    pub fn qeph(name: &str, thickness_m: f64) -> Self {
        Self {
            name: name.to_string(),
            ishell: 24,
            ismstr: 2,
            ish3n: 2,
            idrill: 2,
            n_thickness: 5,
            thickness_m,
            ithick: 2,
            iplas: 2,
        }
    }

    /// Check ranges of the property card.
    ///
    /// # Errors
    ///
    /// Returns [`RadError::Parse`] for non-positive thickness or an
    /// out-of-range through-thickness count.
    pub fn validate(&self) -> Result<(), RadError> {
        if !self.thickness_m.is_finite() || self.thickness_m <= 0.0 {
            return Err(RadError::Parse(format!(
                "shell property {} needs a positive thickness",
                self.name
            )));
        }
        if self.n_thickness < 1 || self.n_thickness > 9 {
            return Err(RadError::Parse(format!(
                "shell property {} needs 1..9 thickness points",
                self.name
            )));
        }
        Ok(())
    }
}

/// Shell element connectivity with 1-based node ids.
#[derive(Debug, Clone, Default)]
pub struct RadShellMesh {
    /// Four-node shell connectivity, 1-based node ids.
    pub quads: Vec<[usize; 4]>,
    /// Three-node shell connectivity, 1-based node ids.
    pub tris: Vec<[usize; 3]>,
}

impl RadShellMesh {
    /// Count four-node shells.
    pub fn num_quads(&self) -> usize {
        self.quads.len()
    }

    /// Count three-node shells.
    pub fn num_tris(&self) -> usize {
        self.tris.len()
    }

    /// Check whether the mesh holds no shells.
    pub fn is_empty(&self) -> bool {
        self.quads.is_empty() && self.tris.is_empty()
    }

    /// Validate index ranges against a node count.
    ///
    /// # Errors
    ///
    /// Returns [`RadError::Parse`] when a shell references a missing node.
    pub fn validate(&self, num_nodes: usize) -> Result<(), RadError> {
        for (index, quad) in self.quads.iter().enumerate() {
            for node in quad {
                if *node == 0 || *node > num_nodes {
                    return Err(RadError::Parse(format!(
                        "shell {} references missing node {node}",
                        index + 1
                    )));
                }
            }
        }
        for (index, tri) in self.tris.iter().enumerate() {
            for node in tri {
                if *node == 0 || *node > num_nodes {
                    return Err(RadError::Parse(format!(
                        "tria {} references missing node {node}",
                        index + 1
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Johnson-Cook plastic parameters rendered as `/MAT/LAW2`.
#[derive(Debug, Clone)]
pub struct RadJohnsonCook {
    /// Young's modulus in Pa.
    pub youngs_modulus_pa: f64,
    /// Poisson ratio, dimensionless.
    pub poisson_ratio: f64,
    /// Mass density in kg/m^3.
    pub density_kg_per_m3: f64,
    /// Yield stress `a` in Pa.
    pub yield_pa: f64,
    /// Hardening modulus `b` in Pa (0 = perfectly plastic).
    pub hardening_pa: f64,
    /// Hardening exponent `n` (1 = linear).
    pub hardening_exp: f64,
    /// Rate coefficient `c` (0 = rate independent).
    pub rate_coeff: f64,
    /// Reference strain rate in 1/s.
    pub ref_strain_rate: f64,
    /// Failure plastic strain (1e20 disables failure).
    pub fail_plastic_strain: f64,
    /// Maximum stress cutoff in Pa (1e20 disables the cap).
    pub max_stress_pa: f64,
}

impl RadJohnsonCook {
    /// Rate-independent elastic-perfectly plastic metal.
    pub fn perfectly_plastic(
        youngs_modulus_pa: f64,
        poisson_ratio: f64,
        density_kg_per_m3: f64,
        yield_pa: f64,
    ) -> Self {
        Self {
            youngs_modulus_pa,
            poisson_ratio,
            density_kg_per_m3,
            yield_pa,
            hardening_pa: 0.0,
            hardening_exp: 1.0,
            rate_coeff: 0.0,
            ref_strain_rate: 1.0,
            fail_plastic_strain: 1.0e20,
            max_stress_pa: 1.0e20,
        }
    }

    /// Rate-independent power-law hardening metal.
    pub fn power_hardening(
        youngs_modulus_pa: f64,
        poisson_ratio: f64,
        density_kg_per_m3: f64,
        yield_pa: f64,
        hardening_pa: f64,
        hardening_exp: f64,
    ) -> Self {
        Self {
            youngs_modulus_pa,
            poisson_ratio,
            density_kg_per_m3,
            yield_pa,
            hardening_pa,
            hardening_exp,
            rate_coeff: 0.0,
            ref_strain_rate: 1.0,
            fail_plastic_strain: 1.0e20,
            max_stress_pa: 1.0e20,
        }
    }
}

/// Tabulated hardening rendered as `/MAT/LAW36` with one curve.
#[derive(Debug, Clone)]
pub struct RadTabulatedHardening {
    /// Young's modulus in Pa.
    pub youngs_modulus_pa: f64,
    /// Poisson ratio, dimensionless.
    pub poisson_ratio: f64,
    /// Mass density in kg/m^3.
    pub density_kg_per_m3: f64,
    /// Yield stress vs plastic strain, at least two rising points.
    pub points: Vec<(f64, f64)>,
}

impl RadTabulatedHardening {
    /// Elastic-perfectly plastic response at one yield stress.
    pub fn perfectly_plastic(
        youngs_modulus_pa: f64,
        poisson_ratio: f64,
        density_kg_per_m3: f64,
        yield_pa: f64,
    ) -> Self {
        Self {
            youngs_modulus_pa,
            poisson_ratio,
            density_kg_per_m3,
            points: vec![(0.0, yield_pa), (0.5, yield_pa)],
        }
    }

    /// Check the hardening curve shape.
    ///
    /// # Errors
    ///
    /// Returns [`RadError::Parse`] for fewer than two points, negative
    /// strains, or non-increasing abscissae.
    pub fn validate(&self) -> Result<(), RadError> {
        if self.points.len() < 2 {
            return Err(RadError::Parse(
                "LAW36 hardening needs at least two points".to_string(),
            ));
        }
        for window in self.points.windows(2) {
            if !window[0].0.is_finite()
                || !window[1].0.is_finite()
                || !window[0].1.is_finite()
                || !window[1].1.is_finite()
                || window[0].0 < 0.0
                || window[1].0 <= window[0].0
                || window[0].1 <= 0.0
                || window[1].1 <= 0.0
            {
                return Err(RadError::Parse(
                    "LAW36 hardening needs rising strains and positive stresses".to_string(),
                ));
            }
        }
        Ok(())
    }
}

/// Plastic material selected by an extended case.
#[derive(Debug, Clone)]
pub enum RadPlasticMaterial {
    /// Johnson-Cook law (`/MAT/LAW2`).
    JohnsonCook(RadJohnsonCook),
    /// Tabulated isotropic hardening (`/MAT/LAW36`, one curve).
    Tabulated(RadTabulatedHardening),
}

/// Material of an extended case.
#[derive(Debug, Clone)]
pub enum RadMaterial {
    /// Linear elastic solid (`/MAT/LAW1`).
    Elastic(RadElasticMaterial),
    /// Elasto-plastic solid (`/MAT/LAW2` or `/MAT/LAW36`).
    Plastic(RadPlasticMaterial),
}

/// Gravity load rendered as `/GRAV`.
#[derive(Debug, Clone)]
pub struct RadGravity {
    /// Load title written on the card.
    pub name: String,
    /// Global direction `X`, `Y`, or `Z` (right-justified per the guide).
    pub direction: char,
    /// Gravity value in m/s^2 (constant) or curve scale.
    pub value_m_per_s2: f64,
    /// Index into [`OpenCourantCase::functions`] (`None` = constant load).
    pub function: Option<usize>,
    /// Index into [`OpenCourantCase::node_sets`] (`None` = whole model).
    pub set: Option<usize>,
}

impl RadGravity {
    /// Check the direction flag.
    ///
    /// # Errors
    ///
    /// Returns [`RadError::Parse`] for a direction outside `X`/`Y`/`Z`.
    pub fn validate(&self) -> Result<(), RadError> {
        if !matches!(self.direction, 'X' | 'Y' | 'Z') {
            return Err(RadError::Parse(format!(
                "gravity {} needs direction X, Y, or Z",
                self.name
            )));
        }
        if !self.value_m_per_s2.is_finite() {
            return Err(RadError::Parse(format!(
                "gravity {} needs a finite value",
                self.name
            )));
        }
        Ok(())
    }
}

/// Element time-history request with guide variable names.
#[derive(Debug, Clone, Default)]
pub struct RadThRequest {
    /// Variable names, e.g. `SX` or `M1` (8 characters max each).
    pub vars: Vec<String>,
    /// Element ids recorded (1-based).
    pub elems: Vec<usize>,
}

impl RadThRequest {
    /// Brick stress request (`SX SY SZ SXY SYZ SXZ PLAS`).
    pub fn brick_stress(elems: Vec<usize>) -> Self {
        Self {
            vars: ["SX", "SY", "SZ", "SXY", "SYZ", "SXZ", "PLAS"]
                .iter()
                .map(ToString::to_string)
                .collect(),
            elems,
        }
    }

    /// Shell resultant request (`F1 F2 F12 M1 M2 M12 THIC`).
    pub fn shell_resultants(elems: Vec<usize>) -> Self {
        Self {
            vars: ["F1", "F2", "F12", "M1", "M2", "M12", "THIC"]
                .iter()
                .map(ToString::to_string)
                .collect(),
            elems,
        }
    }

    /// Check variable spellings and element ranges.
    ///
    /// # Errors
    ///
    /// Returns [`RadError::Parse`] for over-long names or out-of-range ids.
    pub fn validate(&self, num_elems: usize, label: &str) -> Result<(), RadError> {
        for var in &self.vars {
            if var.is_empty() || var.len() > 8 {
                return Err(RadError::Parse(format!(
                    "{label} variable {var} must use 1..8 characters"
                )));
            }
        }
        for elem in &self.elems {
            if *elem == 0 || *elem > num_elems {
                return Err(RadError::Parse(format!(
                    "{label} references missing element {elem}"
                )));
            }
        }
        Ok(())
    }
}

/// An extended OpenCourant comparison case with shells or plasticity.
///
/// Wraps a base [`OpenCourantCase`] (nodes, bricks, sets, boundary conditions,
/// loads, walls, node probes, run controls) and adds shell elements, a
/// shell property, a possibly plastic material, gravity, and element
/// time-history requests. Brick-only plastic cases wrap with empty
/// shells; shell cases wrap with empty bricks.
#[derive(Debug, Clone)]
pub struct RadExtendedCase {
    /// Base case: mesh nodes, bricks, sets, BCs, loads, probes, controls.
    pub base: OpenCourantCase,
    /// Material card selected for every part.
    pub material: RadMaterial,
    /// Shell elements indexing `base.mesh.nodes`.
    pub shells: RadShellMesh,
    /// Shell property (`None` when the case has no shells).
    pub shell_prop: Option<RadShellProperty>,
    /// Gravity load (`None` disables `/GRAV`).
    pub gravity: Option<RadGravity>,
    /// Node time-history variables (default `DX DY DZ`).
    pub th_node_vars: Vec<String>,
    /// Four-node shell time-history request.
    pub th_shell: RadThRequest,
    /// Brick time-history request.
    pub th_brick: RadThRequest,
}

impl RadExtendedCase {
    /// Wrap a base case with its elastic material and node outputs.
    pub fn wrap_elastic(base: OpenCourantCase) -> Self {
        let material = RadMaterial::Elastic(base.material);
        Self {
            base,
            material,
            shells: RadShellMesh::default(),
            shell_prop: None,
            gravity: None,
            th_node_vars: ["DX", "DY", "DZ"].iter().map(ToString::to_string).collect(),
            th_shell: RadThRequest::default(),
            th_brick: RadThRequest::default(),
        }
    }

    /// Part id carrying shells (1 when shells exist).
    fn shell_part(&self) -> Option<usize> {
        if self.shells.is_empty() {
            None
        } else {
            Some(1)
        }
    }

    /// Part id carrying bricks (2 beside shells, else 1).
    fn solid_part(&self) -> Option<usize> {
        if self.base.mesh.bricks.is_empty() {
            None
        } else if self.shell_part().is_some() {
            Some(2)
        } else {
            Some(1)
        }
    }

    /// Check every index, id, and card range of the case.
    ///
    /// # Errors
    ///
    /// Returns [`RadError::Parse`] for the first inconsistency found.
    pub fn validate(&self) -> Result<(), RadError> {
        self.base.mesh.validate()?;
        self.shells.validate(self.base.mesh.num_nodes())?;
        if !self.shells.is_empty() {
            match &self.shell_prop {
                Some(prop) => prop.validate()?,
                None => {
                    return Err(RadError::Parse(
                        "shell elements need a shell property".to_string(),
                    ));
                }
            }
        }
        if let RadMaterial::Plastic(RadPlasticMaterial::Tabulated(table)) = &self.material {
            table.validate()?;
        }
        if let Some(gravity) = &self.gravity {
            gravity.validate()?;
            if let Some(function) = gravity.function
                && function >= self.base.functions.len()
            {
                return Err(RadError::Parse(format!(
                    "gravity {} references missing function {function}",
                    gravity.name
                )));
            }
            if let Some(set) = gravity.set
                && set >= self.base.node_sets.len()
            {
                return Err(RadError::Parse(format!(
                    "gravity {} references missing node set {set}",
                    gravity.name
                )));
            }
        }
        for node in &self.base.th_nodes {
            if *node == 0 || *node > self.base.mesh.num_nodes() {
                return Err(RadError::Parse(format!(
                    "TH/NODE references missing node {node}"
                )));
            }
        }
        self.th_shell
            .validate(self.shells.num_quads(), "/TH/SHEL")?;
        self.th_brick
            .validate(self.base.mesh.num_bricks(), "/TH/BRIC")?;
        Ok(())
    }
}

fn field_var(name: &str) -> String {
    format!("{name:<10}")
}

/// Concentrated-load direction code: 1..3 force X/Y/Z, 4..6 moment X/Y/Z.
///
/// Right-justified per the guide (`/CLOAD` rejects integer directions).
fn direction_code(direction: usize) -> &'static str {
    match direction {
        1 => "X",
        2 => "Y",
        3 => "Z",
        4 => "XX",
        5 => "YY",
        6 => "ZZ",
        other => panic!("RadConcentratedLoad direction {other} outside 1..=6"),
    }
}

fn field_blank() -> String {
    "          ".to_string()
}

/// Render the starter deck of an extended case.
///
/// Emits the guide-shaped blocks `/SHELL/part_ID`, `/SH3N/part_ID`,
/// `/BRICK/part_ID`, `/PROP/SHELL`, `/MAT/LAW2` or `/MAT/LAW36`,
/// `/GRAV`, and `/TH` requests with variable cards. The engine deck is
/// unchanged from [`write_engine`]; use [`write_extended_engine`].
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{shell_clamped_plate_case, write_extended_starter};
/// let case = shell_clamped_plate_case();
/// case.validate().unwrap();
/// let deck = write_extended_starter(&case);
/// assert!(deck.contains("/SHELL/1"));
/// assert!(deck.contains("/PROP/SHELL/1/1"));
/// assert!(deck.contains("/TH/SHEL/2"));
/// ```
pub fn write_extended_starter(case: &RadExtendedCase) -> String {
    let base = &case.base;
    let mut deck = String::new();
    deck.push_str("#RADIOSS STARTER\n");
    deck.push_str(&format!(
        "# sonium_qa rad extended case {} (SI units: kg, m, s)\n",
        base.name
    ));
    deck.push_str("/BEGIN\n");
    deck.push_str(&format!("{}\n", base.title));
    // Invers + input/work unit systems (mandatory /BEGIN cards).
    deck.push_str("      2022\n");
    deck.push_str("              kg                   m                   s\n");
    deck.push_str("              kg                   m                   s\n");
    deck.push_str("/UNIT/1\nSI mass-length-time\n");
    deck.push_str("              kg                   m                   s\n");
    deck.push_str("/NODE\n");
    for (index, node) in base.mesh.nodes.iter().enumerate() {
        deck.push_str(&format!(
            "{}{}{}{}\n",
            field_int(index + 1),
            field_float(node[0]),
            field_float(node[1]),
            field_float(node[2])
        ));
    }
    if let Some(part) = case.solid_part() {
        deck.push_str(&format!("/BRICK/{part}\n"));
        for (index, brick) in base.mesh.bricks.iter().enumerate() {
            let mut line = field_int(index + 1);
            for node in brick {
                line.push_str(&field_int(*node));
            }
            line.push('\n');
            deck.push_str(&line);
        }
    }
    if let Some(part) = case.shell_part() {
        if !case.shells.quads.is_empty() {
            deck.push_str(&format!("/SHELL/{part}\n"));
            for (index, quad) in case.shells.quads.iter().enumerate() {
                let mut line = field_int(index + 1);
                for node in quad {
                    line.push_str(&field_int(*node));
                }
                line.push('\n');
                deck.push_str(&line);
            }
        }
        if !case.shells.tris.is_empty() {
            deck.push_str(&format!("/SH3N/{part}\n"));
            for (index, tri) in case.shells.tris.iter().enumerate() {
                let mut line = field_int(index + 1);
                for node in tri {
                    line.push_str(&field_int(*node));
                }
                line.push('\n');
                deck.push_str(&line);
            }
        }
    }
    for (index, set) in base.node_sets.iter().enumerate() {
        deck.push_str(&format!("/GRNOD/NODE/{}/1\n", index + 1));
        deck.push_str(&format!("{}\n", set.name));
        for chunk in set.nodes.chunks(10) {
            let mut line = String::new();
            for node in chunk {
                line.push_str(&field_int(*node));
            }
            line.push('\n');
            deck.push_str(&line);
        }
    }
    write_extended_material(&mut deck, case);
    if let Some(part) = case.shell_part() {
        let prop = case.shell_prop.as_ref().expect("validated shell property");
        deck.push_str(&format!("/PROP/SHELL/{part}/1\n"));
        deck.push_str(&format!("{}\n", prop.name));
        deck.push_str(&format!(
            "{:>10}{:>10}{:>10}{:>10}{}{}{}{}{}{}\n",
            prop.ishell,
            prop.ismstr,
            prop.ish3n,
            prop.idrill,
            field_int(0),
            field_int(0),
            field_float(0.0),
            field_int(0),
            field_int(0),
            field_int(0),
        ));
        // Membrane, bending, and rotational hourglass plus damping are
        // unused by QEPH/QBAT; the card stays blank (guide defaults).
        deck.push_str(&format!(
            "{}{}{}{}{}\n",
            field_float(0.0),
            field_float(0.0),
            field_float(0.0),
            field_float(0.0),
            field_float(0.0),
        ));
        deck.push_str(&format!(
            "{:>10}{}{}{}{}{:>10}{:>10}{}\n",
            prop.n_thickness,
            field_int(0),
            field_float(prop.thickness_m),
            field_float(0.0),
            field_int(0),
            prop.ithick,
            prop.iplas,
            field_int(0),
        ));
    }
    if let Some(part) = case.solid_part() {
        deck.push_str(&format!("/PROP/SOLID/{part}/1\nelastic brick property\n"));
        // HEPH bricks + full geometric nonlinearities (best practice;
        // proven 1.6% vs EB on the coarse cantilever, Isolid=1 gave 41%).
        deck.push_str("        24         2\n");
    }
    if let Some(part) = case.shell_part() {
        deck.push_str(&format!("/PART/{part}\nshell part\n"));
        deck.push_str(&format!("{}{}\n", field_int(part), field_int(1)));
    }
    if let Some(part) = case.solid_part() {
        deck.push_str(&format!("/PART/{part}\nsolid part\n"));
        deck.push_str(&format!("{}{}\n", field_int(part), field_int(1)));
    }
    for (index, function) in base.functions.iter().enumerate() {
        deck.push_str(&format!("/FUNCT/{}\n", index + 1));
        deck.push_str(&format!("{}\n", function.name));
        for (x, y) in &function.points {
            deck.push_str(&format!("{}{}\n", field_float(*x), field_float(*y)));
        }
    }
    if let RadMaterial::Plastic(RadPlasticMaterial::Tabulated(table)) = &case.material {
        deck.push_str(&format!("/FUNCT/{}\n", base.functions.len() + 1));
        deck.push_str("LAW36 yield stress vs plastic strain\n");
        for (strain, stress) in &table.points {
            deck.push_str(&format!(
                "{}{}\n",
                field_float(*strain),
                field_float(*stress)
            ));
        }
    }
    for (index, bcs) in base.bcs.iter().enumerate() {
        deck.push_str(&format!("/BCS/{}/1\n", index + 1));
        deck.push_str(&format!("{}\n", bcs.name));
        // Proven against the engine + reference guide: one Trarot/Skew/grnd
        // line (Trarot packs the six lock flags as "TTT RRR", 1 = fixed).
        let trarot = format!(
            "{}{}{} {}{}{}",
            i32::from(bcs.lock[0]),
            i32::from(bcs.lock[1]),
            i32::from(bcs.lock[2]),
            i32::from(bcs.lock[3]),
            i32::from(bcs.lock[4]),
            i32::from(bcs.lock[5]),
        );
        deck.push_str(&format!(
            "{:>10}{}{}\n",
            trarot,
            field_int(0),
            field_int(bcs.set + 1)
        ));
    }
    for (index, cload) in base.cloads.iter().enumerate() {
        deck.push_str(&format!("/CLOAD/{}/1\n", index + 1));
        deck.push_str(&format!("{}\n", cload.name));
        deck.push_str(&format!(
            "{}{:>10}{}{}{}{}{}{}\n",
            field_int(cload.function + 1),
            direction_code(cload.direction),
            field_int(0),
            field_int(0),
            field_int(cload.set + 1),
            field_int(0),
            field_float(1.0),
            field_float(cload.force_scale_n),
        ));
    }
    if let Some(gravity) = &case.gravity {
        deck.push_str("/GRAV/1/1\n");
        deck.push_str(&format!("{}\n", gravity.name));
        deck.push_str(&format!(
            "{}{:>10}{}{}{}{}{}{}\n",
            field_int(gravity.function.map_or(0, |id| id + 1)),
            gravity.direction,
            field_int(0),
            field_int(0),
            field_int(gravity.set.map_or(0, |id| id + 1)),
            field_blank(),
            field_float(1.0),
            field_float(gravity.value_m_per_s2),
        ));
    }
    for (index, wall) in base.walls.iter().enumerate() {
        write_wall_block(&mut deck, index, wall);
    }
    let mut th_group = 0;
    write_th_block(
        &mut deck,
        &mut th_group,
        "NODE",
        "time history nodes",
        &case.th_node_vars,
        &base.th_nodes,
    );
    write_th_block(
        &mut deck,
        &mut th_group,
        "SHEL",
        "time history shells",
        &case.th_shell.vars,
        &case.th_shell.elems,
    );
    write_th_block(
        &mut deck,
        &mut th_group,
        "BRIC",
        "time history bricks",
        &case.th_brick.vars,
        &case.th_brick.elems,
    );
    if !base.walls.is_empty() {
        let wall_ids: Vec<usize> = (1..=base.walls.len()).collect();
        write_th_block(
            &mut deck,
            &mut th_group,
            "RWALL",
            "time history walls",
            &["FN".to_string()],
            &wall_ids,
        );
    }
    deck.push_str("/END\n");
    deck
}

fn write_extended_material(deck: &mut String, case: &RadExtendedCase) {
    match &case.material {
        RadMaterial::Elastic(material) => {
            deck.push_str("/MAT/LAW1/1/1\n");
            deck.push_str("linear elastic solid\n");
            // Proven against the engine: line 1 is density, line 2 is E/nu.
            deck.push_str(&format!("{}\n", field_float(material.density_kg_per_m3)));
            deck.push_str(&format!(
                "{}{}\n",
                field_float(material.youngs_modulus_pa),
                field_float(material.poisson_ratio)
            ));
        }
        RadMaterial::Plastic(RadPlasticMaterial::JohnsonCook(jc)) => {
            deck.push_str("/MAT/LAW2/1/1\n");
            deck.push_str("johnson-cook elasto-plastic metal\n");
            deck.push_str(&format!("{}\n", field_float(jc.density_kg_per_m3)));
            // Classic input (Iflag 0), total strain rate (VP 2), and the
            // guide default pressure cutoff.
            deck.push_str(&format!(
                "{}{}{}{}{}\n",
                field_float(jc.youngs_modulus_pa),
                field_float(jc.poisson_ratio),
                field_int(0),
                field_int(2),
                field_float(-1.0e20),
            ));
            deck.push_str(&format!(
                "{}{}{}{}{}\n",
                field_float(jc.yield_pa),
                field_float(jc.hardening_pa),
                field_float(jc.hardening_exp),
                field_float(jc.fail_plastic_strain),
                field_float(jc.max_stress_pa),
            ));
            // Rate block: ICC 1 and Fsmooth 1 are the guide defaults;
            // isotropic hardening (Chard 0); no thermal softening run.
            deck.push_str(&format!(
                "{}{}{}{}{}{}\n",
                field_float(jc.rate_coeff),
                field_float(jc.ref_strain_rate),
                field_int(1),
                field_int(1),
                field_float(1.0e20),
                field_float(0.0),
            ));
            // Isothermal steel thermal line (unused without /HEAT).
            deck.push_str(&format!(
                "{}{}{}{}{}\n",
                field_float(1.0),
                field_float(1793.0),
                field_float(jc.density_kg_per_m3 * 500.0),
                field_float(293.0),
                field_float(1793.0),
            ));
        }
        RadMaterial::Plastic(RadPlasticMaterial::Tabulated(table)) => {
            let curve = case.base.functions.len() + 1;
            deck.push_str("/MAT/LAW36/1/1\n");
            deck.push_str("tabulated isotropic hardening metal\n");
            deck.push_str(&format!("{}\n", field_float(table.density_kg_per_m3)));
            deck.push_str(&format!(
                "{}{}{}{}{}\n",
                field_float(table.youngs_modulus_pa),
                field_float(table.poisson_ratio),
                field_float(1.0e20),
                field_float(1.0e20),
                field_float(2.0e20),
            ));
            // One hardening curve, smoothed rates, full isotropic model,
            // guide-default filter and total-rate flag.
            deck.push_str(&format!(
                "{}{}{}{}{}{}{}\n",
                field_int(1),
                field_int(1),
                field_float(0.0),
                field_float(10_000.0),
                field_float(3.0e20),
                field_blank(),
                field_int(0),
            ));
            deck.push_str(&format!(
                "{}{}{}{}{}\n",
                field_int(0),
                field_float(1.0),
                field_int(0),
                field_float(0.0),
                field_float(0.0),
            ));
            deck.push_str(&format!("{}\n", field_int(curve)));
            deck.push_str(&format!("{}\n", field_float(1.0)));
            deck.push_str(&format!("{}\n", field_float(0.0)));
        }
    }
}

fn write_th_block(
    deck: &mut String,
    next_group: &mut usize,
    keyword: &str,
    title: &str,
    vars: &[String],
    ids: &[usize],
) {
    if vars.is_empty() || ids.is_empty() {
        return;
    }
    // Proven against the engine (error 79): TH group ids share one
    // namespace across all /TH/* types, so the counter is deck-global.
    for var_card in vars.chunks(10) {
        *next_group += 1;
        deck.push_str(&format!("/TH/{keyword}/{}\n", *next_group));
        deck.push_str(&format!("{title}\n"));
        let mut line = String::new();
        for var in var_card {
            line.push_str(&field_var(var));
        }
        line.push('\n');
        deck.push_str(&line);
        for id in ids {
            deck.push_str(&format!("{}{}\n", field_int(*id), field_int(0)));
        }
    }
}

/// Render the engine deck of an extended case.
///
/// Run controls live in the wrapped base case, so this forwards to
/// [`write_engine`].
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{shell_scordelis_lo_case, write_extended_engine};
/// let case = shell_scordelis_lo_case();
/// let deck = write_extended_engine(&case);
/// assert!(deck.contains("#RADIOSS ENGINE"));
/// ```
pub fn write_extended_engine(case: &RadExtendedCase) -> String {
    write_engine(&case.base)
}

/// Run the starter then the engine for an extended case.
///
/// Decks resolve from the wrapped base case name; see [`run_case`].
///
/// # Errors
///
/// Returns [`RadError`] for failed or numerically aborted stages, malformed
/// histories, or histories that do not cover the original requested stop time.
pub fn run_extended_case(
    decks_dir: &Path,
    exe: &Path,
    case: &RadExtendedCase,
) -> Result<RunOutput, RadError> {
    run_case(decks_dir, exe, &case.base)
}

/// Render the checked-in `probes.json` artifact of an extended case.
///
/// Records expected probe values, tolerances, and the producing
/// engine version. See [`probes_json`].
pub fn extended_probes_json(case: &RadExtendedCase, engine_version: &str) -> String {
    probes_json(&case.base, engine_version)
}

/// Time-history channel of a shell variable, e.g. `M1/SHEL/12`.
///
/// Element channels carry the `/TH` group so shell and brick ids never
/// collide; node channels keep the established `DZ/95` spelling. The
/// spelling is confirmed only up to the `th_to_csv` naming, which no
/// host engine can verify yet.
pub fn shel_channel(var: &str, shell: usize) -> String {
    format!("{var}/SHEL/{shell}")
}

/// Time-history channel of a brick variable, e.g. `SX/BRIC/4`.
///
/// Brick stress channels from `/TH/BRIC` use this group-qualified
/// spelling (see [`shel_channel`]).
pub fn bric_channel(var: &str, brick: usize) -> String {
    format!("{var}/BRIC/{brick}")
}

/// Quad mesh of the full Scordelis-Lo roof.
#[derive(Debug, Clone)]
pub struct ScordelisShellMesh {
    /// Node coordinates in metres, 0-based storage.
    pub nodes: Vec<[f64; 3]>,
    /// Quad connectivity, 1-based node ids, outward normals.
    pub quads: Vec<[usize; 4]>,
    /// Diaphragm ring at `z = 0`.
    pub end_lo: Vec<usize>,
    /// Diaphragm ring at `z = L`.
    pub end_hi: Vec<usize>,
    /// Free-edge midspan probe node (theta = +40 deg, z = L/2).
    pub probe_node: usize,
    /// Quad id nearest the apex midspan (theta = 0, z = L/2).
    pub probe_shell: usize,
}

/// Mesh the Scordelis-Lo roof with quads.
///
/// Covers theta in [-40, +40] degrees over `n_theta` divisions and the
/// length over `n_z` divisions. Both counts must be even so the probe
/// node and shell sit exactly on the free edge and apex midspan.
///
/// # Panics
///
/// Panics for odd or zero divisions.
pub fn scordelis_lo_shell_mesh(n_theta: usize, n_z: usize) -> ScordelisShellMesh {
    assert!(
        n_theta >= 2 && n_theta.is_multiple_of(2),
        "n_theta must be even"
    );
    assert!(n_z >= 2 && n_z.is_multiple_of(2), "n_z must be even");
    let (radius, length) = (25.0, 50.0);
    let half_angle = 40.0_f64.to_radians();
    let id = |iz: usize, it: usize| iz * (n_theta + 1) + it + 1;
    let mut nodes = Vec::with_capacity((n_theta + 1) * (n_z + 1));
    for iz in 0..=n_z {
        for it in 0..=n_theta {
            let theta = -half_angle + 2.0 * half_angle * it as f64 / n_theta as f64;
            nodes.push([
                radius * theta.sin(),
                radius * theta.cos(),
                length * iz as f64 / n_z as f64,
            ]);
        }
    }
    let mut quads = Vec::with_capacity(n_theta * n_z);
    for iz in 0..n_z {
        for it in 0..n_theta {
            // Winding (iz,it) -> (iz+1,it) -> (iz+1,it+1) -> (iz,it+1)
            // keeps the outward normal (dz cross dtheta points out).
            quads.push([
                id(iz, it),
                id(iz + 1, it),
                id(iz + 1, it + 1),
                id(iz, it + 1),
            ]);
        }
    }
    let mut end_lo = Vec::with_capacity(n_theta + 1);
    let mut end_hi = Vec::with_capacity(n_theta + 1);
    for it in 0..=n_theta {
        end_lo.push(id(0, it));
        end_hi.push(id(n_z, it));
    }
    ScordelisShellMesh {
        nodes,
        quads,
        end_lo,
        end_hi,
        probe_node: id(n_z / 2, n_theta),
        probe_shell: (n_z / 2 - 1) * n_theta + (n_theta / 2 - 1) + 1,
    }
}

/// Polar shell mesh of a clamped circular plate.
#[derive(Debug, Clone)]
pub struct ClampedPlateShellMesh {
    /// Node coordinates in metres, 0-based storage.
    pub nodes: Vec<[f64; 3]>,
    /// Quad connectivity between rings, 1-based node ids.
    pub quads: Vec<[usize; 4]>,
    /// Center fan triangles, 1-based node ids.
    pub tris: Vec<[usize; 3]>,
    /// Clamped rim nodes.
    pub rim: Vec<usize>,
    /// Center probe node.
    pub center_node: usize,
    /// Ring-1 probe quad id for shell resultants.
    pub probe_shell: usize,
    /// Centroid radius of the probe quad in metres.
    pub probe_radius_m: f64,
    /// Tributary area per node in m^2 (sums to the disk area).
    pub nodal_area_m2: Vec<f64>,
}

/// Mesh a disk with a center fan plus quad rings.
///
/// Ring `i` sits at `radius * i / n_rings` with `n_sectors` nodes; the
/// center fan covers the hub with triangles and every outer band with
/// quads. Tributary areas follow the half-station ring bands.
///
/// # Panics
///
/// Panics for fewer than two rings or three sectors.
pub fn clamped_plate_shell_mesh(
    radius_m: f64,
    n_rings: usize,
    n_sectors: usize,
) -> ClampedPlateShellMesh {
    assert!(n_rings >= 2, "plate mesh needs at least two rings");
    assert!(n_sectors >= 3, "plate mesh needs at least three sectors");
    let ring_radius = |i: usize| radius_m * i as f64 / n_rings as f64;
    // Node 1 is the center; ring i sector j follows in row-major order.
    let id = |i: usize, j: usize| 1 + (i - 1) * n_sectors + j + 1;
    let mut nodes = vec![[0.0, 0.0, 0.0]];
    for i in 1..=n_rings {
        for j in 0..n_sectors {
            let theta = 2.0 * std::f64::consts::PI * j as f64 / n_sectors as f64;
            nodes.push([
                ring_radius(i) * theta.cos(),
                ring_radius(i) * theta.sin(),
                0.0,
            ]);
        }
    }
    let mut tris = Vec::with_capacity(n_sectors);
    for j in 0..n_sectors {
        tris.push([1, id(1, j), id(1, (j + 1) % n_sectors)]);
    }
    let mut quads = Vec::with_capacity((n_rings - 1) * n_sectors);
    for i in 1..n_rings {
        for j in 0..n_sectors {
            quads.push([
                id(i, j),
                id(i, (j + 1) % n_sectors),
                id(i + 1, (j + 1) % n_sectors),
                id(i + 1, j),
            ]);
        }
    }
    let mut rim = Vec::with_capacity(n_sectors);
    for j in 0..n_sectors {
        rim.push(id(n_rings, j));
    }
    // Half-station bands: the center owns the hub disk, each ring owns
    // its band split over sectors, and the rim band closes at `radius`.
    let station = |i: usize| {
        // Outer half-station of ring i (i = 0 is the center hub edge).
        if i == 0 {
            ring_radius(1) / 2.0
        } else if i >= n_rings {
            radius_m
        } else {
            (ring_radius(i) + ring_radius(i + 1)) / 2.0
        }
    };
    let mut nodal_area_m2 = vec![0.0; nodes.len()];
    nodal_area_m2[0] = std::f64::consts::PI * station(0).powi(2);
    let mut inner = station(0);
    for i in 1..=n_rings {
        let outer = station(i);
        let band = std::f64::consts::PI * (outer.powi(2) - inner.powi(2)) / n_sectors as f64;
        for j in 0..n_sectors {
            nodal_area_m2[id(i, j) - 1] = band;
        }
        inner = outer;
    }
    ClampedPlateShellMesh {
        nodes,
        quads,
        tris,
        rim,
        center_node: 1,
        probe_shell: 1,
        probe_radius_m: (ring_radius(1) + ring_radius(2)) / 2.0,
        nodal_area_m2,
    }
}

/// Quasi-static ramp-hold curve for explicit gravity and pressure.
///
/// Ramps 0 to 1 over the first 80 percent of the stop time, then holds
/// full load so final-value probes read a settled state.
fn ramp_hold_function() -> RadFunction {
    RadFunction {
        name: "ramp 0 to 1 then hold".to_string(),
        points: vec![(0.0, 0.0), (0.8, 1.0), (1.0, 1.0)],
    }
}

/// Build the Scordelis-Lo shell comparison case.
///
/// A full-roof QEPH model (R = 25, L = 50, t = 0.25) with diaphragm
/// ends carries self weight through `/GRAV`. The free-edge midspan
/// probe uses a full-load nonlinear regression baseline (-0.2574).
/// The published MacNeal target (-0.3024) is a separate linear-accuracy
/// reference. Matching the nonlinear regression does not prove linear
/// accuracy or matching native parity. Controlled load-sensitivity runs
/// prevent attributing the difference to locking from this probe alone.
pub fn shell_scordelis_lo_case() -> RadExtendedCase {
    // 32x32 with a slow ramp; the hold average is the settled static.
    let mesh = scordelis_lo_shell_mesh(32, 32);
    // Density carries the 90 Pa self weight through rho * g * t.
    let density = 90.0 / (9.81 * 0.25);
    let base = OpenCourantCase {
        name: "shell_scordelis_lo".to_string(),
        title: "sonium_qa rad scordelis-lo roof quasi-static gravity".to_string(),
        mesh: RadMesh {
            nodes: mesh.nodes.clone(),
            bricks: Vec::new(),
        },
        material: RadElasticMaterial {
            youngs_modulus_pa: 4.32e8,
            poisson_ratio: 0.0,
            density_kg_per_m3: density,
        },
        node_sets: vec![
            RadNodeSet {
                name: "diaphragm z=0".to_string(),
                nodes: mesh.end_lo,
            },
            RadNodeSet {
                name: "diaphragm z=L".to_string(),
                nodes: mesh.end_hi,
            },
        ],
        bcs: vec![
            RadBoundaryCondition {
                name: "diaphragm z=0".to_string(),
                set: 0,
                lock: [true, true, false, false, false, false],
            },
            RadBoundaryCondition {
                name: "diaphragm z=L".to_string(),
                set: 1,
                lock: [true, true, false, false, false, false],
            },
        ],
        functions: vec![RadFunction {
            // First mode period is 1.19 s with higher-mode hash; the 5 s
            // ramp (Tr/T1 > 4) tracks quasi-statically and the 3 s hold
            // averages the residual corner transient to its center.
            name: "slow ramp 0 to 1 then hold".to_string(),
            points: vec![(0.0, 0.0), (5.0, 1.0), (8.0, 1.0)],
        }],
        cloads: Vec::new(),
        walls: Vec::new(),
        th_nodes: vec![mesh.probe_node],
        probes: vec![
            RadProbe {
                // Nonlinear engine regression only. A controlled original
                // QEPH32 full8s run at1% gravity gives normalized DY=-.305846,
                // while full gravity gives about-.2574. The published linear
                // target is-.3024; its accuracy and full-load native parity
                // require separate evidence. Do not label this difference
                // as shell locking or use regression agreement as validation.
                label: "free edge mid DY".to_string(),
                channel: format!("DY/{}", mesh.probe_node),
                expected: -0.2574,
                abs_tol: 5e-3,
                rel_tol: 0.02,
            },
            RadProbe {
                label: "apex shell THIC".to_string(),
                channel: shel_channel("THIC", mesh.probe_shell),
                expected: 0.25,
                abs_tol: 1e-9,
                rel_tol: 1e-6,
            },
        ],
        stop_time_s: 8.0,
        th_period_s: 0.02,
        anim_period_s: 0.8,
    };
    let mut case = RadExtendedCase::wrap_elastic(base);
    case.shells = RadShellMesh {
        quads: mesh.quads,
        tris: Vec::new(),
    };
    case.shell_prop = Some(RadShellProperty::qeph("roof section", 0.25));
    case.gravity = Some(RadGravity {
        name: "self weight".to_string(),
        direction: 'Y',
        value_m_per_s2: -9.81,
        function: Some(0),
        set: None,
    });
    case.th_shell = RadThRequest::shell_resultants(vec![mesh.probe_shell]);
    case
}

/// Clamped circular plate center deflection (Kirchhoff closed form).
fn clamped_plate_center_deflection(
    pressure_pa: f64,
    radius_m: f64,
    young_pa: f64,
    poisson: f64,
    thick_m: f64,
) -> f64 {
    let rigidity = young_pa * thick_m.powi(3) / (12.0 * (1.0 - poisson.powi(2)));
    -pressure_pa * radius_m.powi(4) / (64.0 * rigidity)
}

/// Clamped circular plate circumferential moment at a radius.
///
/// `M_t(r) = q/16 [(1+nu) a^2 - (1+3nu) r^2]` (Timoshenko plates,
/// axisymmetric bending). The probe quad local axis 1 runs along its
/// N1-N2 edge (circumferential), so `M1` reads this component.
fn clamped_plate_hoop_moment(
    pressure_pa: f64,
    radius_m: f64,
    poisson: f64,
    at_radius_m: f64,
) -> f64 {
    pressure_pa / 16.0
        * ((1.0 + poisson) * radius_m.powi(2) - (1.0 + 3.0 * poisson) * at_radius_m.powi(2))
}

/// Build the clamped circular plate shell comparison case.
///
/// A polar QEPH disk (a = 1, t = 0.02) with a clamped rim carries
/// uniform pressure as ring-wise nodal `/CLOAD`s; the center probe
/// replays the Kirchhoff closed form and the probe shell checks the
/// hoop moment plus constant thickness.
pub fn shell_clamped_plate_case() -> RadExtendedCase {
    let (radius, thick, young, poisson, pressure) = (1.0, 0.02, 200.0e9, 0.3, 1000.0);
    let (n_rings, n_sectors) = (8, 32);
    let mesh = clamped_plate_shell_mesh(radius, n_rings, n_sectors);
    // One node set per ring: tributary areas are uniform within a ring.
    let mut node_sets = vec![RadNodeSet {
        name: "rim".to_string(),
        nodes: mesh.rim.clone(),
    }];
    let mut cloads = Vec::new();
    let ring_nodes = |i: usize| {
        if i == 0 {
            vec![1]
        } else {
            (0..n_sectors)
                .map(|j| 1 + (i - 1) * n_sectors + j + 1)
                .collect()
        }
    };
    for i in 0..=n_rings {
        let nodes = ring_nodes(i);
        let area = mesh.nodal_area_m2[nodes[0] - 1];
        node_sets.push(RadNodeSet {
            name: format!("pressure ring {i}"),
            nodes: nodes.clone(),
        });
        cloads.push(RadConcentratedLoad {
            name: format!("pressure ring {i}"),
            set: node_sets.len() - 1,
            direction: 3,
            function: 0,
            force_scale_n: -pressure * area,
        });
    }
    let center = clamped_plate_center_deflection(pressure, radius, young, poisson, thick);
    let hoop = clamped_plate_hoop_moment(pressure, radius, poisson, mesh.probe_radius_m);
    let base = OpenCourantCase {
        name: "shell_clamped_plate".to_string(),
        title: "sonium_qa rad clamped plate quasi-static pressure".to_string(),
        mesh: RadMesh {
            nodes: mesh.nodes.clone(),
            bricks: Vec::new(),
        },
        material: RadElasticMaterial {
            youngs_modulus_pa: young,
            poisson_ratio: poisson,
            density_kg_per_m3: 7800.0,
        },
        node_sets,
        bcs: vec![RadBoundaryCondition {
            name: "clamp rim".to_string(),
            set: 0,
            lock: [true, true, true, true, true, true],
        }],
        functions: vec![ramp_hold_function()],
        cloads,
        walls: Vec::new(),
        th_nodes: vec![mesh.center_node],
        probes: vec![
            RadProbe {
                label: "center DZ".to_string(),
                channel: format!("DZ/{}", mesh.center_node),
                expected: center,
                abs_tol: 1e-9,
                rel_tol: 0.05,
            },
            RadProbe {
                label: "probe shell THIC".to_string(),
                channel: shel_channel("THIC", mesh.probe_shell),
                expected: thick,
                abs_tol: 1e-9,
                rel_tol: 1e-6,
            },
            RadProbe {
                label: "probe shell M1".to_string(),
                channel: shel_channel("M1", mesh.probe_shell),
                // M1 carries moment per unit length per thickness squared.
                expected: hoop / thick.powi(2),
                abs_tol: 1e3,
                rel_tol: 0.15,
            },
        ],
        stop_time_s: 1.0,
        th_period_s: 0.01,
        anim_period_s: 0.1,
    };
    let mut case = RadExtendedCase::wrap_elastic(base);
    case.shells = RadShellMesh {
        quads: mesh.quads,
        tris: mesh.tris,
    };
    case.shell_prop = Some(RadShellProperty::qeph("plate section", thick));
    case.th_shell = RadThRequest::shell_resultants(vec![mesh.probe_shell]);
    case
}

/// Elastic-perfectly plastic steel as a LAW36 material.
///
/// Matches the thick-cylinder autofrettage case inputs (E, nu, rho,
/// yield) so the plasticity lane can adopt it with a wrap call.
pub fn perfectly_plastic_steel_law36(
    youngs_modulus_pa: f64,
    poisson_ratio: f64,
    density_kg_per_m3: f64,
    yield_pa: f64,
) -> RadPlasticMaterial {
    RadPlasticMaterial::Tabulated(RadTabulatedHardening::perfectly_plastic(
        youngs_modulus_pa,
        poisson_ratio,
        density_kg_per_m3,
        yield_pa,
    ))
}

// =====================================================================
// Phase-4 lane 4B writer completion. Additive: nothing above this line
// changes behavior or writer output. P4C1: `OpenCourantCase` is frozen, so every
// new card arrives as a NEW type plus a NEW `write_*` function, bundled
// for deck splicing through `RadExtra` (`write_starter_extra` /
// `write_engine_extra`). Card layouts follow the Altair Radioss 2022
// Reference Guide block format (ten 10-character fields per line, p.41);
// each item cites its guide page plus the deck- or reader-grounded
// layout checks recorded in the worker report.
// =====================================================================

/// Render an interface `IBC` flag triplet (columns 8-10, guide p.252).
///
/// The reader layout is `%7s%1d%1d%1d` (hm_cfg_files `inter_type7.cfg`,
/// radioss2020 section), so the three axis flags sit at columns 8, 9,
/// and 10 with seven leading blanks.
fn field_ibc(ibc: [bool; 3]) -> String {
    let bit = |flag: bool| if flag { '1' } else { '0' };
    format!("       {}{}{}", bit(ibc[0]), bit(ibc[1]), bit(ibc[2]))
}

/// Render a signed flag in a 10-character integer field.
fn field_int32(value: i32) -> String {
    format!("{value:>10}")
}

/// Render a left-justified name clipped to 20 characters.
///
/// Vent and porous titles occupy the last 20 columns of their card
/// (AIRBAG1 porous layout, hm_cfg_files radioss140 `airbag1.cfg`).
fn field_name20(name: &str) -> String {
    let clipped: String = name.chars().take(20).collect();
    format!("{clipped:<20}")
}

/// Render a header/title/id-list block (ten ids per line).
///
/// Shared by `/SURF/PART`, `/GRNOD/SURF`, and `/LINE/SURF`, whose
/// formats are all "any number may be input, 10 per line" (guide
/// pp.2358, 2471, 2490).
fn write_id_list(deck: &mut String, header: &str, title: &str, ids: &[usize]) {
    deck.push_str(header);
    deck.push('\n');
    deck.push_str(title);
    deck.push('\n');
    for chunk in ids.chunks(10) {
        let mut line = String::new();
        for id in chunk {
            line.push_str(&field_int(*id));
        }
        line.push('\n');
        deck.push_str(&line);
    }
}

/// Part surface rendered as `/SURF/PART` (guide p.2490).
///
/// Collects the shell faces of the listed parts into one surface for
/// interfaces (`/INTER/TYPE2`, `/INTER/TYPE7`) and monitored volumes.
/// All segments must belong to 2D (shell/triangle) elements.
#[derive(Debug, Clone)]
pub struct RadSurfParts {
    /// Surface id written on the card.
    pub id: usize,
    /// Surface title written on the card.
    pub title: String,
    /// Part ids forming the surface (1-based, at least one).
    pub parts: Vec<usize>,
}

/// Node group collected from surfaces (`/GRNOD/SURF`, guide p.2358).
///
/// Supplies secondary nodes for `/INTER/TYPE2` and `/INTER/TYPE7`
/// from previously defined surfaces.
#[derive(Debug, Clone)]
pub struct RadGrnodSurfs {
    /// Node group id written on the card.
    pub id: usize,
    /// Group title written on the card.
    pub title: String,
    /// Surface ids forming the group (at least one).
    pub surfs: Vec<usize>,
}

/// Line group collected from surfaces (`/LINE/SURF`, guide p.2471).
///
/// Supplies the secondary and main lines for `/INTER/TYPE11` from
/// previously defined surfaces.
#[derive(Debug, Clone)]
pub struct RadLineSurfs {
    /// Line group id written on the card.
    pub id: usize,
    /// Line title written on the card.
    pub title: String,
    /// Surface ids forming the lines (at least one).
    pub surfs: Vec<usize>,
}

/// Part subset rendered as `/SUBSET`.
///
/// The bare header-plus-title form is deck-proven (miniqa `IDEL8`,
/// `RBPLI` decks); subsets group parts for `/TH/SUBS` output.
#[derive(Debug, Clone)]
pub struct RadSubset {
    /// Subset id written on the card.
    pub id: usize,
    /// Subset title written on the card.
    pub title: String,
}

/// Render one `/SURF/PART` block.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{RadSurfParts, write_surf_parts};
/// let mut deck = String::new();
/// write_surf_parts(&mut deck, &RadSurfParts { id: 4, title: "box skin".to_string(), parts: vec![1] });
/// assert!(deck.contains("/SURF/PART/4\n"));
/// ```
pub fn write_surf_parts(deck: &mut String, surf: &RadSurfParts) {
    write_id_list(
        deck,
        &format!("/SURF/PART/{}", surf.id),
        &surf.title,
        &surf.parts,
    );
}

/// Render one `/GRNOD/SURF` block.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{RadGrnodSurfs, write_grnod_surfs};
/// let mut deck = String::new();
/// write_grnod_surfs(&mut deck, &RadGrnodSurfs { id: 32, title: "contact nodes".to_string(), surfs: vec![4] });
/// assert!(deck.contains("/GRNOD/SURF/32\n"));
/// ```
pub fn write_grnod_surfs(deck: &mut String, grnod: &RadGrnodSurfs) {
    write_id_list(
        deck,
        &format!("/GRNOD/SURF/{}", grnod.id),
        &grnod.title,
        &grnod.surfs,
    );
}

/// Render one `/LINE/SURF` block.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{RadLineSurfs, write_line_surfs};
/// let mut deck = String::new();
/// write_line_surfs(&mut deck, &RadLineSurfs { id: 33, title: "contact edges".to_string(), surfs: vec![4] });
/// assert!(deck.contains("/LINE/SURF/33\n"));
/// ```
pub fn write_line_surfs(deck: &mut String, line: &RadLineSurfs) {
    write_id_list(
        deck,
        &format!("/LINE/SURF/{}", line.id),
        &line.title,
        &line.surfs,
    );
}

/// Render one `/SUBSET` block (header plus title).
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{RadSubset, write_subset};
/// let mut deck = String::new();
/// write_subset(&mut deck, &RadSubset { id: 1, title: "all parts".to_string() });
/// assert!(deck.contains("/SUBSET/1\nall parts\n"));
/// ```
pub fn write_subset(deck: &mut String, subset: &RadSubset) {
    deck.push_str(&format!("/SUBSET/{}\n", subset.id));
    deck.push_str(&format!("{}\n", subset.title));
}

/// Tied interface rendered as `/INTER/TYPE2` (guide p.205).
///
/// Connects secondary nodes to a main surface (coarse/fine mesh
/// ties, spotwelds). This writer covers the base card; the
/// rupture/spotweld extensions (`Spotflag` 20-22/25/27/28) are out of
/// scope and rejected by validation.
#[derive(Debug, Clone)]
pub struct RadInterType2 {
    /// Interface id written on the card.
    pub id: usize,
    /// Interface title written on the card.
    pub title: String,
    /// Secondary node group id (`grnd_IDs`).
    pub grnd_secondary: usize,
    /// Main surface id (`surf_IDm`).
    pub surf_main: usize,
    /// Ignore flag (0 = starter default, keep unprojected nodes).
    pub ignore: i32,
    /// Spotweld flag (0 = plain tied contact).
    pub spotflag: i32,
    /// Search level (0 = default).
    pub level: i32,
    /// Search flag (0 = default).
    pub isearch: i32,
    /// Deletion flag (0 = default).
    pub idel: i32,
    /// Search distance in metres (0 = automatic).
    pub dsearch_m: f64,
}

impl RadInterType2 {
    /// Plain tied interface between one group and one surface.
    pub fn basic(id: usize, title: &str, grnd_secondary: usize, surf_main: usize) -> Self {
        Self {
            id,
            title: title.to_string(),
            grnd_secondary,
            surf_main,
            ignore: 0,
            spotflag: 0,
            level: 0,
            isearch: 0,
            idel: 0,
            dsearch_m: 0.0,
        }
    }
}

/// General contact interface (`/INTER/TYPE7`, guide p.252).
///
/// Multi-usage impact interface between secondary nodes and a main
/// surface. Layout is the current (radioss2020) reader format: ten
/// ids with a blank sixth field, gap/stiffness lines, penalty line,
/// `IBC` line, and friction line. Curvature nodes (`Icurv` 1/2),
/// extra friction coefficients (`Ifric` > 0), thermal (`Ithe` > 1),
/// and admixed (`Iadm` 2) extensions are out of scope.
#[derive(Debug, Clone)]
pub struct RadInterType7 {
    /// Interface id written on the card.
    pub id: usize,
    /// Interface title written on the card.
    pub title: String,
    /// Secondary node group id.
    pub grnd_secondary: usize,
    /// Main surface id.
    pub surf_main: usize,
    /// Stiffness definition flag (0 = default).
    pub istf: i32,
    /// Thermal flag (0 = no heat transfer).
    pub ithe: i32,
    /// Gap flag (0 = constant gap).
    pub igap: i32,
    /// Airbag flag (0 = off).
    pub ibag: i32,
    /// Deletion flag (0 = default).
    pub idel: i32,
    /// Curvature flag (0 = off).
    pub icurv: i32,
    /// Admixture flag (0 = off).
    pub iadm: i32,
    /// Gap scale factor.
    pub fscalegap: f64,
    /// Maximum gap in metres.
    pub gapmax_m: f64,
    /// Maximum penetration force in N.
    pub fpenmax_n: f64,
    /// Tied-contact flag (0 = sliding).
    pub itied: i32,
    /// Minimum interface stiffness in N/m.
    pub stmin: f64,
    /// Maximum interface stiffness in N/m.
    pub stmax: f64,
    /// Mesh-size fraction for stiffness (0 = default).
    pub mesh_size_pct: f64,
    /// Minimum time step in seconds (0 = default).
    pub dtmin_s: f64,
    /// Gap-removal flag (0 = default).
    pub irem_gap: i32,
    /// Secondary-removal flag (0 = default).
    pub irem_i2: i32,
    /// Stiffness scale factor.
    pub stfac: f64,
    /// Coulomb friction coefficient.
    pub fric: f64,
    /// Minimum gap in metres.
    pub gapmin_m: f64,
    /// Activation start time in seconds.
    pub tstart_s: f64,
    /// Activation stop time in seconds.
    pub tstop_s: f64,
    /// Boundary deactivation flags per axis.
    pub ibc: [bool; 3],
    /// Inactivation flag (0 = active).
    pub inacti: i32,
    /// Stiffness damping.
    pub viss: f64,
    /// Friction damping.
    pub visf: f64,
    /// Sorting multiplier.
    pub bumult: f64,
    /// Friction model flag (0 = Coulomb).
    pub ifric: i32,
    /// Friction filter flag (0 = off).
    pub ifiltr: i32,
    /// Friction cutoff frequency in Hz.
    pub xfreq_hz: f64,
    /// Friction formulation flag (0 = default).
    pub iform: i32,
    /// Activation sensor id (0 = none).
    pub sens_id: usize,
    /// Friction function id (0 = none).
    pub fct_idf: usize,
    /// Friction function scale.
    pub ascalef: f64,
    /// Friction identifier (0 = none).
    pub fric_id: usize,
}

impl RadInterType7 {
    /// Frictionless sliding contact with default stiffness.
    ///
    /// Mirrors the deck-proven miniqa `TRITUBV4` values (`Stfac`
    /// 0.2, open stop time); the stop default `1e30` keeps the
    /// interface active through any run.
    pub fn basic(id: usize, title: &str, grnd_secondary: usize, surf_main: usize) -> Self {
        Self {
            id,
            title: title.to_string(),
            grnd_secondary,
            surf_main,
            istf: 0,
            ithe: 0,
            igap: 0,
            ibag: 0,
            idel: 0,
            icurv: 0,
            iadm: 0,
            fscalegap: 0.0,
            gapmax_m: 0.0,
            fpenmax_n: 0.0,
            itied: 0,
            stmin: 0.0,
            stmax: 0.0,
            mesh_size_pct: 0.0,
            dtmin_s: 0.0,
            irem_gap: 0,
            irem_i2: 0,
            stfac: 0.2,
            fric: 0.0,
            gapmin_m: 0.0,
            tstart_s: 0.0,
            tstop_s: 1.0e30,
            ibc: [false; 3],
            inacti: 0,
            viss: 0.0,
            visf: 0.0,
            bumult: 0.0,
            ifric: 0,
            ifiltr: 0,
            xfreq_hz: 0.0,
            iform: 0,
            sens_id: 0,
            fct_idf: 0,
            ascalef: 0.0,
            fric_id: 0,
        }
    }
}

/// Edge-to-edge contact interface (`/INTER/TYPE11`, guide p.281).
///
/// Impact between a secondary and a main line (shell edges, beams,
/// trusses, springs). Complements `/INTER/TYPE7`, which misses edge
/// contact. The thermal extension (`Ithe` > 0) is out of scope.
#[derive(Debug, Clone)]
pub struct RadInterType11 {
    /// Interface id written on the card.
    pub id: usize,
    /// Interface title written on the card.
    pub title: String,
    /// Secondary line group id.
    pub line_secondary: usize,
    /// Main line group id.
    pub line_main: usize,
    /// Stiffness definition flag (0 = default series stiffness).
    pub istf: i32,
    /// Thermal flag (0 = no heat transfer).
    pub ithe: i32,
    /// Gap flag (0 = constant gap).
    pub igap: i32,
    /// Gap-removal flag (0 = default).
    pub irem_gap: i32,
    /// Deletion flag (0 = default).
    pub idel: i32,
    /// Minimum interface stiffness in N/m.
    pub stmin: f64,
    /// Maximum interface stiffness in N/m.
    pub stmax: f64,
    /// Mesh-size fraction for stiffness (0 = default).
    pub mesh_size_pct: f64,
    /// Minimum time step in seconds (0 = default).
    pub dtmin_s: f64,
    /// Friction formulation flag (0 = default).
    pub iform: i32,
    /// Activation sensor id (0 = none).
    pub sens_id: usize,
    /// Stiffness scale factor.
    pub stfac: f64,
    /// Coulomb friction coefficient.
    pub fric: f64,
    /// Minimum gap in metres.
    pub gapmin_m: f64,
    /// Activation start time in seconds.
    pub tstart_s: f64,
    /// Activation stop time in seconds.
    pub tstop_s: f64,
    /// Boundary deactivation flags per axis.
    pub ibc: [bool; 3],
    /// Inactivation flag (0 = active).
    pub inacti: i32,
    /// Stiffness damping.
    pub viss: f64,
    /// Friction damping.
    pub visf: f64,
    /// Sorting multiplier.
    pub bumult: f64,
    /// Friction identifier (0 = none).
    pub fric_id: usize,
}

impl RadInterType11 {
    /// Frictionless edge contact with unit stiffness scale.
    ///
    /// Mirrors the deck-proven miniqa `TEST_002` values (`Stfac`
    /// 1, open stop time) without friction.
    pub fn basic(id: usize, title: &str, line_secondary: usize, line_main: usize) -> Self {
        Self {
            id,
            title: title.to_string(),
            line_secondary,
            line_main,
            istf: 0,
            ithe: 0,
            igap: 0,
            irem_gap: 0,
            idel: 0,
            stmin: 0.0,
            stmax: 0.0,
            mesh_size_pct: 0.0,
            dtmin_s: 0.0,
            iform: 0,
            sens_id: 0,
            stfac: 1.0,
            fric: 0.0,
            gapmin_m: 0.0,
            tstart_s: 0.0,
            tstop_s: 1.0e30,
            ibc: [false; 3],
            inacti: 0,
            viss: 0.0,
            visf: 0.0,
            bumult: 0.0,
            fric_id: 0,
        }
    }
}

/// Contact interface selected for a deck.
#[derive(Debug, Clone)]
pub enum RadInterface {
    /// Tied contact (`/INTER/TYPE2`).
    Type2(RadInterType2),
    /// General sliding contact (`/INTER/TYPE7`).
    Type7(RadInterType7),
    /// Edge-to-edge contact (`/INTER/TYPE11`).
    Type11(RadInterType11),
}

impl RadInterface {
    /// Interface id carried on the card header.
    pub fn id(&self) -> usize {
        match self {
            Self::Type2(inter) => inter.id,
            Self::Type7(inter) => inter.id,
            Self::Type11(inter) => inter.id,
        }
    }
}

/// Render one `/INTER/TYPE2` block.
///
/// The data line is `grnd surf Ignore Spotflag Level Isearch Idel`
/// plus a blank field and `Dsearch` (hm_cfg_files radioss2025
/// `inter_type2.cfg`: `%10d%10d%10s%10d%10d%10d%10d%10s%20lg`).
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{RadInterType2, write_inter_type2};
/// let mut deck = String::new();
/// write_inter_type2(&mut deck, &RadInterType2::basic(1, "tie", 4, 5));
/// assert!(deck.contains("/INTER/TYPE2/1\n"));
/// ```
pub fn write_inter_type2(deck: &mut String, inter: &RadInterType2) {
    deck.push_str(&format!("/INTER/TYPE2/{}\n", inter.id));
    deck.push_str(&format!("{}\n", inter.title));
    deck.push_str(&format!(
        "{}{}{:>10}{:>10}{:>10}{:>10}{:>10}{}{}\n",
        field_int(inter.grnd_secondary),
        field_int(inter.surf_main),
        inter.ignore,
        inter.spotflag,
        inter.level,
        inter.isearch,
        inter.idel,
        field_blank(),
        field_float(inter.dsearch_m),
    ));
}

/// Render one `/INTER/TYPE7` block.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{RadInterType7, write_inter_type7};
/// let mut deck = String::new();
/// write_inter_type7(&mut deck, &RadInterType7::basic(1, "contact", 2, 3));
/// assert!(deck.contains("/INTER/TYPE7/1\n"));
/// assert!(deck.contains("2.000000000000E-1"));
/// ```
pub fn write_inter_type7(deck: &mut String, inter: &RadInterType7) {
    deck.push_str(&format!("/INTER/TYPE7/{}\n", inter.id));
    deck.push_str(&format!("{}\n", inter.title));
    // Line 1: the sixth field (legacy Multimp) stays blank in the
    // current format (hm_cfg_files radioss2020 section).
    deck.push_str(&format!(
        "{}{}{}{}{}{}{}{}{}{}\n",
        field_int(inter.grnd_secondary),
        field_int(inter.surf_main),
        field_int32(inter.istf),
        field_int32(inter.ithe),
        field_int32(inter.igap),
        field_blank(),
        field_int32(inter.ibag),
        field_int32(inter.idel),
        field_int32(inter.icurv),
        field_int32(inter.iadm),
    ));
    // Line 2: `Itied` sits in field 9 per the radioss2020
    // `%20lg%20lg%20lg%20s%10d` layout.
    deck.push_str(&format!(
        "{}{}{}{}{}{}\n",
        field_float(inter.fscalegap),
        field_float(inter.gapmax_m),
        field_float(inter.fpenmax_n),
        field_blank(),
        field_blank(),
        field_int32(inter.itied),
    ));
    deck.push_str(&format!(
        "{}{}{}{}{}{}\n",
        field_float(inter.stmin),
        field_float(inter.stmax),
        field_float(inter.mesh_size_pct),
        field_float(inter.dtmin_s),
        field_int32(inter.irem_gap),
        field_int32(inter.irem_i2),
    ));
    deck.push_str(&format!(
        "{}{}{}{}{}\n",
        field_float(inter.stfac),
        field_float(inter.fric),
        field_float(inter.gapmin_m),
        field_float(inter.tstart_s),
        field_float(inter.tstop_s),
    ));
    // IBC line: `%7s%1d%1d%1d%20s%10d%20lg%20lg%20lg`.
    deck.push_str(&format!(
        "{}{}{}{}{}{}{}\n",
        field_ibc(inter.ibc),
        field_blank(),
        field_blank(),
        field_int32(inter.inacti),
        field_float(inter.viss),
        field_float(inter.visf),
        field_float(inter.bumult),
    ));
    deck.push_str(&format!(
        "{}{}{}{}{}{}{}{}\n",
        field_int32(inter.ifric),
        field_int32(inter.ifiltr),
        field_float(inter.xfreq_hz),
        field_int32(inter.iform),
        field_int(inter.sens_id),
        field_int(inter.fct_idf),
        field_float(inter.ascalef),
        field_int(inter.fric_id),
    ));
}

/// Render one `/INTER/TYPE11` block.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{RadInterType11, write_inter_type11};
/// let mut deck = String::new();
/// write_inter_type11(&mut deck, &RadInterType11::basic(4, "edges", 33, 33));
/// assert!(deck.contains("/INTER/TYPE11/4\n"));
/// ```
pub fn write_inter_type11(deck: &mut String, inter: &RadInterType11) {
    deck.push_str(&format!("/INTER/TYPE11/{}\n", inter.id));
    deck.push_str(&format!("{}\n", inter.title));
    deck.push_str(&format!(
        "{}{}{}{}{}{}{}{}\n",
        field_int(inter.line_secondary),
        field_int(inter.line_main),
        field_int32(inter.istf),
        field_int32(inter.ithe),
        field_int32(inter.igap),
        field_blank(),
        field_int32(inter.irem_gap),
        field_int32(inter.idel),
    ));
    deck.push_str(&format!(
        "{}{}{}{}{}{}\n",
        field_float(inter.stmin),
        field_float(inter.stmax),
        field_float(inter.mesh_size_pct),
        field_float(inter.dtmin_s),
        field_int32(inter.iform),
        field_int(inter.sens_id),
    ));
    deck.push_str(&format!(
        "{}{}{}{}{}\n",
        field_float(inter.stfac),
        field_float(inter.fric),
        field_float(inter.gapmin_m),
        field_float(inter.tstart_s),
        field_float(inter.tstop_s),
    ));
    deck.push_str(&format!(
        "{}{}{}{}{}{}{}\n",
        field_ibc(inter.ibc),
        field_blank(),
        field_blank(),
        field_int32(inter.inacti),
        field_float(inter.viss),
        field_float(inter.visf),
        field_float(inter.bumult),
    ));
    // `fric_ID` alone in field 10 (cols 91-100): nine blanks first.
    deck.push_str(&format!(
        "{}{}{}{}{}{}{}{}{}{}\n",
        field_blank(),
        field_blank(),
        field_blank(),
        field_blank(),
        field_blank(),
        field_blank(),
        field_blank(),
        field_blank(),
        field_blank(),
        field_int(inter.fric_id),
    ));
}

/// Render any contact interface block.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{RadInterType2, RadInterface, write_interface};
/// let mut deck = String::new();
/// write_interface(&mut deck, &RadInterface::Type2(RadInterType2::basic(1, "tie", 4, 5)));
/// assert!(deck.contains("/INTER/TYPE2/1\n"));
/// ```
pub fn write_interface(deck: &mut String, inter: &RadInterface) {
    match inter {
        RadInterface::Type2(typed) => write_inter_type2(deck, typed),
        RadInterface::Type7(typed) => write_inter_type7(deck, typed),
        RadInterface::Type11(typed) => write_inter_type11(deck, typed),
    }
}

/// Predefined airbag gas name (`/MAT/GAS/PREDEF`, guide p.1046).
#[derive(Debug, Clone)]
pub enum RadGasKind {
    /// Predefined air mixture.
    Air,
    /// Predefined nitrogen.
    Nitrogen,
    /// Predefined carbon dioxide.
    CarbonDioxide,
    /// Predefined water vapor.
    Water,
    /// Predefined hydrogen.
    Hydrogen,
    /// Any other predefined gas name from the guide table.
    Named(String),
}

impl RadGasKind {
    /// Gas name written on the card data line.
    pub fn name(&self) -> String {
        match self {
            Self::Air => "AIR".to_string(),
            Self::Nitrogen => "N2".to_string(),
            Self::CarbonDioxide => "CO2".to_string(),
            Self::Water => "H2O".to_string(),
            Self::Hydrogen => "H2".to_string(),
            Self::Named(name) => name.clone(),
        }
    }
}

/// Predefined gas rendered as `/MAT/GAS/PREDEF` (guide p.1046).
///
/// Supplies the initial and injected gases of `/MONVOL/AIRBAG1`
/// by predefined name (deck-proven miniqa `RBPLI` values).
#[derive(Debug, Clone)]
pub struct RadGasPredef {
    /// Material id written on the card.
    pub id: usize,
    /// Material title written on the card.
    pub title: String,
    /// Predefined gas selected.
    pub gas: RadGasKind,
}

/// Render one `/MAT/GAS/PREDEF` block.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{RadGasKind, RadGasPredef, write_gas_predef};
/// let mut deck = String::new();
/// write_gas_predef(&mut deck, &RadGasPredef { id: 4, title: "bag gas".to_string(), gas: RadGasKind::Air });
/// assert!(deck.contains("/MAT/GAS/PREDEF/4\n"));
/// assert!(deck.contains("\nAIR\n"));
/// ```
pub fn write_gas_predef(deck: &mut String, gas: &RadGasPredef) {
    deck.push_str(&format!("/MAT/GAS/PREDEF/{}\n", gas.id));
    deck.push_str(&format!("{}\n", gas.title));
    deck.push_str(&format!("{}\n", gas.gas.name()));
}

/// One gas of a `/PROP/INJECT2` mixture.
#[derive(Debug, Clone)]
pub struct RadInjectGas {
    /// Gas material id (`/MAT/GAS` card).
    pub mat: usize,
    /// Molar fraction in the mixture.
    pub molar_fraction: f64,
    /// Mass-flow function id (0 = none).
    pub function: usize,
}

/// Gas injector rendered as `/PROP/INJECT2` (guide p.1867).
///
/// Feeds a `/MONVOL/AIRBAG1` injector with a gas mixture; the mass
/// and temperature histories come from `/FUNCT` curves.
#[derive(Debug, Clone)]
pub struct RadInject2 {
    /// Injector property id written on the card.
    pub id: usize,
    /// Injector title written on the card.
    pub title: String,
    /// Flow flag (0 = default).
    pub iflow: i32,
    /// Injected-mass function id.
    pub mass_function: usize,
    /// Injected-temperature function id.
    pub temp_function: usize,
    /// Injected-mass scale (kg).
    pub scal_m: f64,
    /// Temperature scale (K).
    pub scal_t: f64,
    /// Time scale applied to both curves.
    pub ascal_t: f64,
    /// Mixture gases (at least one).
    pub gases: Vec<RadInjectGas>,
}

/// Render one `/PROP/INJECT2` block.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{RadInject2, RadInjectGas, write_inject2};
/// let mut deck = String::new();
/// write_inject2(&mut deck, &RadInject2 {
///     id: 10, title: "injector".to_string(), iflow: 0,
///     mass_function: 1, temp_function: 2,
///     scal_m: 1.0, scal_t: 0.0, ascal_t: 0.0,
///     gases: vec![RadInjectGas { mat: 4, molar_fraction: 1.0, function: 0 }],
/// });
/// assert!(deck.contains("/PROP/INJECT2/10\n"));
/// ```
pub fn write_inject2(deck: &mut String, inject: &RadInject2) {
    deck.push_str(&format!("/PROP/INJECT2/{}\n", inject.id));
    deck.push_str(&format!("{}\n", inject.title));
    deck.push_str(&format!(
        "{}{}\n",
        field_int(inject.gases.len()),
        field_int32(inject.iflow),
    ));
    deck.push_str(&format!(
        "{}{}{}{}{}\n",
        field_int(inject.mass_function),
        field_int(inject.temp_function),
        field_float(inject.scal_m),
        field_float(inject.scal_t),
        field_float(inject.ascal_t),
    ));
    for gas in &inject.gases {
        // `%10d %20lg%10d` with a blank second field (hm_cfg_files
        // radioss110 `prop_inject2.cfg`).
        deck.push_str(&format!(
            "{}{}{}{}\n",
            field_int(gas.mat),
            field_blank(),
            field_float(gas.molar_fraction),
            field_int(gas.function),
        ));
    }
}

/// Jetting direction functions of an airbag injector.
///
/// Read only when the injector sets `Ijet` to 1; the 100-column
/// shape (three function ids, one blank, three scales) mirrors the
/// deck-proven `JETV41B` jetting line read by the same starter
/// field names (`hm_read_monvol_type7.F`).
#[derive(Debug, Clone)]
pub struct RadJetting {
    /// Pressure function id.
    pub fct_pt: usize,
    /// Theta function id.
    pub fct_theta: usize,
    /// Delta function id.
    pub fct_delta: usize,
    /// Pressure scale.
    pub fscale_pt: f64,
    /// Theta scale.
    pub fscale_ptheta: f64,
    /// Delta scale.
    pub fscale_pdelta: f64,
}

fn write_jetting(deck: &mut String, jet: &RadJetting) {
    deck.push_str(&format!(
        "{}{}{}{}{}{}{}\n",
        field_int(jet.fct_pt),
        field_int(jet.fct_theta),
        field_int(jet.fct_delta),
        field_blank(),
        field_float(jet.fscale_pt),
        field_float(jet.fscale_ptheta),
        field_float(jet.fscale_pdelta),
    ));
}

/// Injector of a `/MONVOL/AIRBAG1` volume (guide p.1790).
#[derive(Debug, Clone)]
pub struct RadAirbagInjector {
    /// Injector property id (`/PROP/INJECT2` card).
    pub inject_id: usize,
    /// Fire sensor id (0 = none).
    pub sens_id: usize,
    /// Jetting flag (1 = read a [`RadJetting`] line).
    pub ijet: i32,
    /// Jetting direction node ids (0 = unused).
    pub nodes: [usize; 3],
    /// Jetting functions (required when `ijet` is 1).
    pub jetting: Option<RadJetting>,
}

/// Vent hole of a `/MONVOL/AIRBAG1` volume (guide p.1791).
#[derive(Debug, Clone)]
pub struct RadAirbagVent {
    /// Vent surface id.
    pub surf: usize,
    /// Vent formulation flag (2 = read the Chemkin line).
    pub iform: i32,
    /// Vent area in m^2.
    pub avent_m2: f64,
    /// Vent coefficient.
    pub bvent: f64,
    /// Vent title (clipped to 20 characters).
    pub title: String,
    /// Vent start time in seconds.
    pub tstart_s: f64,
    /// Vent stop time in seconds.
    pub tstop_s: f64,
    /// Pressure-deficit flag value.
    pub dpdef: f64,
    /// Pressure-deficit time in seconds.
    pub dtpdef_s: f64,
    /// Pressure-deficit flag id.
    pub idtpdef: usize,
    /// Vent area function ids (time, pressure, area).
    pub fcts: [usize; 3],
    /// Vent area function scales.
    pub fscales: [f64; 3],
    /// Second vent area function ids (primed row).
    pub fcts_p: [usize; 3],
    /// Second vent area function scales.
    pub fscales_p: [f64; 3],
    /// Chemkin vent function (read only when `iform` is 2).
    pub chemkin: Option<(usize, f64)>,
}

/// Porous fabric surface of `/MONVOL/AIRBAG1` (guide p.1791).
///
/// The card shapes are the hm_cfg_files radioss140 `airbag1.cfg`
/// layouts; `Iformps` 1 (as in the deck-proven miniqa `RBPLI`
/// volume) skips the leakage-coefficient line.
#[derive(Debug, Clone)]
pub struct RadAirbagPorous {
    /// Porous surface id.
    pub surf: usize,
    /// Porous formulation flag (0/1/2).
    pub iformps: i32,
    /// Blockage flag (0 = none).
    pub iblockage: i32,
    /// Surface title (clipped to 20 characters).
    pub title: String,
    /// Porosity start time in seconds.
    pub tstart_s: f64,
    /// Porosity stop time in seconds.
    pub tstop_s: f64,
    /// Pressure-deficit value.
    pub dpdef: f64,
    /// Pressure-deficit time in seconds.
    pub dtpdef_s: f64,
    /// Pressure-deficit flag id.
    pub idtpdef: usize,
    /// Leakage line `(Cps, Areaps, fct_cps, fct_aps, Fscale_cps,
    /// Fscale_aps)`; read only when `iformps` is 0.
    pub leakage: Option<(f64, f64, usize, usize, f64, f64)>,
    /// Chemkin line `(fct, Fscale)`; read only when `iformps` is 2.
    pub chemkin: Option<(usize, f64)>,
}

/// One-chamber airbag (`/MONVOL/AIRBAG1`, guide pp.1790-1792).
///
/// The leading lines are the hm_cfg_files radioss140 `airbag1.cfg`
/// layout: external surface plus heat coefficient, five area
/// scales, then `mat_ID Mu Pext T0 Iequi Ittf` (the guide
/// extraction drops the `Mu` column; the reader and the miniqa
/// `RBPLI` deck carry it).
#[derive(Debug, Clone)]
pub struct RadMonvolAirbag1 {
    /// Monitored-volume id written on the card.
    pub id: usize,
    /// Volume title written on the card.
    pub title: String,
    /// External surface id.
    pub surf_external: usize,
    /// Heat transfer coefficient.
    pub hconv: f64,
    /// Area scales (time, pressure, surface, area, deficit).
    pub ascale: [f64; 5],
    /// Initial-gas material id (`/MAT/GAS` card).
    pub mat_gas: usize,
    /// Gas viscosity.
    pub gas_mu: f64,
    /// External pressure in Pa.
    pub pext_pa: f64,
    /// Initial temperature in K.
    pub t0_k: f64,
    /// Equilibrium flag (0 = default).
    pub iequi: i32,
    /// Time-temperature flag (0 = default).
    pub ittf: i32,
    /// Gas injectors.
    pub injectors: Vec<RadAirbagInjector>,
    /// Vent holes.
    pub vents: Vec<RadAirbagVent>,
    /// Porous fabric surfaces.
    pub porous: Vec<RadAirbagPorous>,
}

fn write_airbag_vent(deck: &mut String, vent: &RadAirbagVent) {
    // Title in columns 81-100, mirroring the porous layout below.
    deck.push_str(&format!(
        "{}{}{}{}{}{}{}\n",
        field_int(vent.surf),
        field_int32(vent.iform),
        field_float(vent.avent_m2),
        field_float(vent.bvent),
        field_blank(),
        field_blank(),
        field_name20(&vent.title),
    ));
    deck.push_str(&format!(
        "{}{}{}{}{}{}\n",
        field_float(vent.tstart_s),
        field_float(vent.tstop_s),
        field_float(vent.dpdef),
        field_float(vent.dtpdef_s),
        field_blank(),
        field_int(vent.idtpdef),
    ));
    for (fcts, scales) in [(&vent.fcts, &vent.fscales), (&vent.fcts_p, &vent.fscales_p)] {
        deck.push_str(&format!(
            "{}{}{}{}{}{}{}\n",
            field_int(fcts[0]),
            field_int(fcts[1]),
            field_int(fcts[2]),
            field_blank(),
            field_float(scales[0]),
            field_float(scales[1]),
            field_float(scales[2]),
        ));
    }
    if let Some((fct, fscale)) = vent.chemkin {
        deck.push_str(&format!(
            "{}{}{}\n",
            field_int(fct),
            field_blank(),
            field_float(fscale),
        ));
    }
}

fn write_airbag_porous(deck: &mut String, porous: &RadAirbagPorous) {
    // Title in columns 81-100 (hm_cfg_files radioss140 airbag1.cfg:
    // `%10d%10d%10d` + 50 blanks + `%20s`).
    deck.push_str(&format!(
        "{}{}{}{}{}{}{}{}{}\n",
        field_int(porous.surf),
        field_int32(porous.iformps),
        field_int32(porous.iblockage),
        field_blank(),
        field_blank(),
        field_blank(),
        field_blank(),
        field_blank(),
        field_name20(&porous.title),
    ));
    deck.push_str(&format!(
        "{}{}{}{}{}{}\n",
        field_float(porous.tstart_s),
        field_float(porous.tstop_s),
        field_float(porous.dpdef),
        field_float(porous.dtpdef_s),
        field_blank(),
        field_int(porous.idtpdef),
    ));
    if let Some((cps, area, fct_c, fct_a, fscale_c, fscale_a)) = porous.leakage {
        deck.push_str(&format!(
            "{}{}{}{}{}{}\n",
            field_float(cps),
            field_float(area),
            field_int(fct_c),
            field_int(fct_a),
            field_float(fscale_c),
            field_float(fscale_a),
        ));
    }
    if let Some((fct, fscale)) = porous.chemkin {
        deck.push_str(&format!(
            "{}{}{}\n",
            field_int(fct),
            field_blank(),
            field_float(fscale),
        ));
    }
}

/// Render one `/MONVOL/AIRBAG1` block.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{RadMonvolAirbag1, write_monvol_airbag1};
/// let mut deck = String::new();
/// write_monvol_airbag1(&mut deck, &RadMonvolAirbag1 {
///     id: 1, title: "bag".to_string(), surf_external: 4, hconv: 0.0,
///     ascale: [0.0; 5], mat_gas: 2, gas_mu: 0.1, pext_pa: 101_325.0,
///     t0_k: 0.0, iequi: 0, ittf: 0,
///     injectors: Vec::new(), vents: Vec::new(), porous: Vec::new(),
/// });
/// assert!(deck.contains("/MONVOL/AIRBAG1/1\n"));
/// ```
pub fn write_monvol_airbag1(deck: &mut String, bag: &RadMonvolAirbag1) {
    deck.push_str(&format!("/MONVOL/AIRBAG1/{}\n", bag.id));
    deck.push_str(&format!("{}\n", bag.title));
    deck.push_str(&format!(
        "{}{}{}\n",
        field_int(bag.surf_external),
        field_blank(),
        field_float(bag.hconv),
    ));
    deck.push_str(&format!(
        "{}{}{}{}{}\n",
        field_float(bag.ascale[0]),
        field_float(bag.ascale[1]),
        field_float(bag.ascale[2]),
        field_float(bag.ascale[3]),
        field_float(bag.ascale[4]),
    ));
    deck.push_str(&format!(
        "{}{}{}{}{}{}{}\n",
        field_int(bag.mat_gas),
        field_blank(),
        field_float(bag.gas_mu),
        field_float(bag.pext_pa),
        field_float(bag.t0_k),
        field_int32(bag.iequi),
        field_int32(bag.ittf),
    ));
    deck.push_str(&format!("{}\n", field_int(bag.injectors.len())));
    for injector in &bag.injectors {
        deck.push_str(&format!(
            "{}{}{}{}{}{}\n",
            field_int(injector.inject_id),
            field_int(injector.sens_id),
            field_int32(injector.ijet),
            field_int(injector.nodes[0]),
            field_int(injector.nodes[1]),
            field_int(injector.nodes[2]),
        ));
        if let Some(jetting) = &injector.jetting {
            write_jetting(deck, jetting);
        }
    }
    deck.push_str(&format!(
        "{}{}\n",
        field_int(bag.vents.len()),
        field_int(bag.porous.len()),
    ));
    for vent in &bag.vents {
        write_airbag_vent(deck, vent);
    }
    for porous in &bag.porous {
        write_airbag_porous(deck, porous);
    }
}

/// Injector jet of a `/MONVOL/COMMU` volume.
///
/// The obsolete-COMMU layout below mirrors the deck-proven miniqa
/// `JETV41B` volumes byte for byte (field widths measured from the
/// deck); the starter still reads this shape (`read_monvol.F`
/// `ITYPE = 5`).
#[derive(Debug, Clone)]
pub struct RadCommuJet {
    /// Injected-gas ratio of specific heats.
    pub gamma: f64,
    /// Injected-gas heat-capacity coefficients.
    pub cp: [f64; 3],
    /// Injected-mass function id (0 = none).
    pub fct_mass: usize,
    /// Flow flag (0 = default).
    pub iflow: i32,
    /// Injected-mass scale.
    pub fscale_mass: f64,
    /// Injected-temperature function id (0 = none).
    pub fct_temp: usize,
    /// Injected-temperature scale.
    pub fscale_temp: f64,
    /// Fire sensor id (0 = none).
    pub sens_id: usize,
    /// Jetting flag (1 = read a [`RadJetting`] line).
    pub ijet: i32,
    /// Jetting direction node ids (0 = unused).
    pub nodes: [usize; 3],
    /// Jetting functions (required when `ijet` is 1).
    pub jetting: Option<RadJetting>,
}

/// Vent hole of a `/MONVOL/COMMU` volume.
#[derive(Debug, Clone)]
pub struct RadCommuVent {
    /// Vent surface id (0 = none).
    pub surf: usize,
    /// Vent area in m^2.
    pub avent_m2: f64,
    /// Vent coefficient.
    pub bvent: f64,
    /// Vent stop time in seconds.
    pub tstop_s: f64,
    /// Vent start time in seconds.
    pub tvent_s: f64,
    /// Pressure-deficit value.
    pub dpdef: f64,
    /// Pressure-deficit time in seconds.
    pub dtpdef_s: f64,
    /// Vent function id (0 = none).
    pub fct_v: usize,
    /// Vent function scale.
    pub fscale_v: f64,
    /// Pressure-deficit flag id.
    pub idtpdef: usize,
    /// Vent area function ids (time, pressure, area).
    pub fcts: [usize; 3],
    /// Vent area function scales.
    pub fscales: [f64; 3],
    /// Second vent area function ids (primed row).
    pub fcts_p: [usize; 3],
    /// Second vent area function scales.
    pub fscales_p: [f64; 3],
}

/// Chamber communication of a `/MONVOL/COMMU` volume.
#[derive(Debug, Clone)]
pub struct RadCommuBag {
    /// Connected volume id.
    pub bag: usize,
    /// Communication surface id (0 = none).
    pub surf_c: usize,
    /// Communication pressure deficit in Pa.
    pub dpcdef_pa: f64,
    /// Communication area in m^2.
    pub acom_m2: f64,
    /// Communication stop time in seconds.
    pub tcom_s: f64,
    /// Communication pressure-deficit time in seconds.
    pub dtpcdef_s: f64,
}

/// Multi-chamber airbag (`/MONVOL/COMMU`, miniqa header).
///
/// This is the obsolete inline-gas COMMU shape the miniqa corpus
/// actually contains (`JETV41B`, `Invers` 2019, still accepted by
/// the current starter); the modern `/MONVOL/COMMU1` with `/MAT/GAS`
/// is documented at guide p.1806.
#[derive(Debug, Clone)]
pub struct RadMonvolCommu {
    /// Monitored-volume id written on the card.
    pub id: usize,
    /// Volume title written on the card.
    pub title: String,
    /// External surface id.
    pub surf_external: usize,
    /// Area scales (time, pressure, surface, area, deficit).
    pub ascale: [f64; 5],
    /// Gas viscosity.
    pub gas_mu: f64,
    /// External pressure in Pa.
    pub pext_pa: f64,
    /// Initial temperature in K.
    pub t0_k: f64,
    /// Equilibrium flag (0 = default).
    pub iequi: i32,
    /// Time-temperature flag (0 = default).
    pub ittf: i32,
    /// Initial-gas ratio of specific heats.
    pub gamma_i: f64,
    /// Initial-gas heat-capacity coefficients.
    pub cp_i: [f64; 3],
    /// Injector jets.
    pub jets: Vec<RadCommuJet>,
    /// Vent holes.
    pub vents: Vec<RadCommuVent>,
    /// Connected chambers.
    pub bags: Vec<RadCommuBag>,
}

/// Monitored volume selected for a deck.
#[derive(Debug, Clone)]
pub enum RadMonvol {
    /// One-chamber airbag (`/MONVOL/AIRBAG1`).
    Airbag1(RadMonvolAirbag1),
    /// Multi-chamber airbag (`/MONVOL/COMMU`).
    Commu(RadMonvolCommu),
}

impl RadMonvol {
    /// Volume id carried on the card header.
    pub fn id(&self) -> usize {
        match self {
            Self::Airbag1(bag) => bag.id,
            Self::Commu(bag) => bag.id,
        }
    }
}

/// Render one `/MONVOL/COMMU` block.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{RadMonvolCommu, write_monvol_commu};
/// let mut deck = String::new();
/// write_monvol_commu(&mut deck, &RadMonvolCommu {
///     id: 1, title: "chamber".to_string(), surf_external: 3,
///     ascale: [0.0; 5], gas_mu: 0.1, pext_pa: 101.35, t0_k: 0.0,
///     iequi: 0, ittf: 0, gamma_i: 1.667, cp_i: [520.275, 0.0, 0.0],
///     jets: Vec::new(), vents: Vec::new(), bags: Vec::new(),
/// });
/// assert!(deck.contains("/MONVOL/COMMU/1\n"));
/// ```
pub fn write_monvol_commu(deck: &mut String, bag: &RadMonvolCommu) {
    deck.push_str(&format!("/MONVOL/COMMU/{}\n", bag.id));
    deck.push_str(&format!("{}\n", bag.title));
    deck.push_str(&format!("{}\n", field_int(bag.surf_external)));
    deck.push_str(&format!(
        "{}{}{}{}{}\n",
        field_float(bag.ascale[0]),
        field_float(bag.ascale[1]),
        field_float(bag.ascale[2]),
        field_float(bag.ascale[3]),
        field_float(bag.ascale[4]),
    ));
    // Obsolete layout has no `mat_ID`: one blank field, then
    // `Mu Pext T0 Iequi Ittf` (measured `JETV41B` bytes).
    deck.push_str(&format!(
        "{}{}{}{}{}{}{}\n",
        field_blank(),
        field_blank(),
        field_float(bag.gas_mu),
        field_float(bag.pext_pa),
        field_float(bag.t0_k),
        field_int32(bag.iequi),
        field_int32(bag.ittf),
    ));
    deck.push_str(&format!(
        "{}{}{}{}\n",
        field_float(bag.gamma_i),
        field_float(bag.cp_i[0]),
        field_float(bag.cp_i[1]),
        field_float(bag.cp_i[2]),
    ));
    deck.push_str(&format!("{}\n", field_int(bag.jets.len())));
    for jet in &bag.jets {
        deck.push_str(&format!(
            "{}{}{}{}\n",
            field_float(jet.gamma),
            field_float(jet.cp[0]),
            field_float(jet.cp[1]),
            field_float(jet.cp[2]),
        ));
        deck.push_str(&format!(
            "{}{}{}{}{}{}\n",
            field_int(jet.fct_mass),
            field_int32(jet.iflow),
            field_float(jet.fscale_mass),
            field_int(jet.fct_temp),
            field_float(jet.fscale_temp),
            field_int(jet.sens_id),
        ));
        deck.push_str(&format!(
            "{}{}{}{}\n",
            field_int32(jet.ijet),
            field_int(jet.nodes[0]),
            field_int(jet.nodes[1]),
            field_int(jet.nodes[2]),
        ));
        if let Some(jetting) = &jet.jetting {
            write_jetting(deck, jetting);
        }
    }
    deck.push_str(&format!("{}\n", field_int(bag.vents.len())));
    for vent in &bag.vents {
        deck.push_str(&format!(
            "{}{}{}{}\n",
            field_int(vent.surf),
            field_float(vent.avent_m2),
            field_float(vent.bvent),
            field_float(vent.tstop_s),
        ));
        deck.push_str(&format!(
            "{}{}{}{}{}{}\n",
            field_float(vent.tvent_s),
            field_float(vent.dpdef),
            field_float(vent.dtpdef_s),
            field_int(vent.fct_v),
            field_float(vent.fscale_v),
            field_int(vent.idtpdef),
        ));
        for (fcts, scales) in [(&vent.fcts, &vent.fscales), (&vent.fcts_p, &vent.fscales_p)] {
            deck.push_str(&format!(
                "{}{}{}{}{}{}{}\n",
                field_int(fcts[0]),
                field_int(fcts[1]),
                field_int(fcts[2]),
                field_blank(),
                field_float(scales[0]),
                field_float(scales[1]),
                field_float(scales[2]),
            ));
        }
    }
    deck.push_str(&format!("{}\n", field_int(bag.bags.len())));
    for link in &bag.bags {
        deck.push_str(&format!(
            "{}{}{}{}{}{}\n",
            field_int(link.bag),
            field_int(link.surf_c),
            field_float(link.dpcdef_pa),
            field_float(link.acom_m2),
            field_float(link.tcom_s),
            field_float(link.dtpcdef_s),
        ));
    }
}

/// Render any monitored-volume block.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{RadMonvol, RadMonvolCommu, write_monvol};
/// let mut deck = String::new();
/// write_monvol(&mut deck, &RadMonvol::Commu(RadMonvolCommu {
///     id: 2, title: "chamber 2".to_string(), surf_external: 4,
///     ascale: [0.0; 5], gas_mu: 0.1, pext_pa: 101.35, t0_k: 0.0,
///     iequi: 0, ittf: 0, gamma_i: 1.667, cp_i: [520.275, 0.0, 0.0],
///     jets: Vec::new(), vents: Vec::new(), bags: Vec::new(),
/// }));
/// assert!(deck.contains("/MONVOL/COMMU/2\n"));
/// ```
pub fn write_monvol(deck: &mut String, bag: &RadMonvol) {
    match bag {
        RadMonvol::Airbag1(typed) => write_monvol_airbag1(deck, typed),
        RadMonvol::Commu(typed) => write_monvol_commu(deck, typed),
    }
}

/// Rayleigh damping rendered as `/DAMP` short format (guide p.131).
///
/// Applies mass (`alpha`) and stiffness (`beta`) damping to every
/// degree of freedom of a node group. Stiffness damping is a time
/// and must stay below the model time step (guide p.133).
#[derive(Debug, Clone)]
pub struct RadDamping {
    /// Damping id written on the card.
    pub id: usize,
    /// Damping title written on the card.
    pub title: String,
    /// Mass damping coefficient in 1/s.
    pub alpha: f64,
    /// Stiffness damping coefficient in seconds.
    pub beta_s: f64,
    /// Damped node group id.
    pub grnd: usize,
    /// Skew system id (0 = global frame).
    pub skew: usize,
    /// Damping start time in seconds.
    pub tstart_s: f64,
    /// Damping stop time in seconds.
    pub tstop_s: f64,
}

/// Per-degree damping (`/DAMP` full format, guide p.131).
///
/// Non-isotropic Rayleigh damping with one mass/stiffness pair per
/// translational (`x/y/z`) and rotational (`xx/yy/zz`) axis.
#[derive(Debug, Clone)]
pub struct RadDampingFull {
    /// Damping id written on the card.
    pub id: usize,
    /// Damping title written on the card.
    pub title: String,
    /// Mass coefficients per axis in 1/s.
    pub alpha: [f64; 6],
    /// Stiffness coefficients per axis in seconds.
    pub beta_s: [f64; 6],
    /// Damped node group id.
    pub grnd: usize,
    /// Skew system id (0 = global frame).
    pub skew: usize,
    /// Damping start time in seconds.
    pub tstart_s: f64,
    /// Damping stop time in seconds.
    pub tstop_s: f64,
}

/// Engine damping override (`/DAMP`, guide p.2780).
///
/// Replaces the starter `/DAMP` coefficients of one damping id;
/// engine options use free format, so plain spacing applies.
#[derive(Debug, Clone)]
pub struct RadEngineDamping {
    /// Starter damping id to override.
    pub damp_id: usize,
    /// Mass damping coefficient in 1/s.
    pub alpha: f64,
    /// Stiffness damping coefficient in seconds.
    pub beta_s: f64,
}

/// Render one short-format `/DAMP` block.
///
/// The data line is `alpha beta grnd skew Tstart Tstop`
/// (`F20 F20 I10 I10 F20 F20`, deck-proven miniqa `DAMPC`).
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{RadDamping, write_damping};
/// let mut deck = String::new();
/// write_damping(&mut deck, &RadDamping {
///     id: 1, title: "bulk damping".to_string(), alpha: 1.0, beta_s: 0.0,
///     grnd: 2, skew: 0, tstart_s: 0.0, tstop_s: 1.0e30,
/// });
/// assert!(deck.contains("/DAMP/1\n"));
/// ```
pub fn write_damping(deck: &mut String, damp: &RadDamping) {
    deck.push_str(&format!("/DAMP/{}\n", damp.id));
    deck.push_str(&format!("{}\n", damp.title));
    deck.push_str(&format!(
        "{}{}{}{}{}{}\n",
        field_float(damp.alpha),
        field_float(damp.beta_s),
        field_int(damp.grnd),
        field_int(damp.skew),
        field_float(damp.tstart_s),
        field_float(damp.tstop_s),
    ));
}

/// Render one full-format `/DAMP` block.
///
/// Six coefficient lines follow the title; the group, skew, and
/// time window ride on the first (`x`) line (guide p.131).
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{RadDampingFull, write_damping_full};
/// let mut deck = String::new();
/// write_damping_full(&mut deck, &RadDampingFull {
///     id: 2, title: "axis damping".to_string(),
///     alpha: [1.0; 6], beta_s: [0.0; 6],
///     grnd: 3, skew: 0, tstart_s: 0.0, tstop_s: 1.0e30,
/// });
/// assert!(deck.contains("/DAMP/2\n"));
/// ```
pub fn write_damping_full(deck: &mut String, damp: &RadDampingFull) {
    deck.push_str(&format!("/DAMP/{}\n", damp.id));
    deck.push_str(&format!("{}\n", damp.title));
    deck.push_str(&format!(
        "{}{}{}{}{}{}\n",
        field_float(damp.alpha[0]),
        field_float(damp.beta_s[0]),
        field_int(damp.grnd),
        field_int(damp.skew),
        field_float(damp.tstart_s),
        field_float(damp.tstop_s),
    ));
    for axis in 1..6 {
        deck.push_str(&format!(
            "{}{}\n",
            field_float(damp.alpha[axis]),
            field_float(damp.beta_s[axis]),
        ));
    }
}

/// Render one engine `/DAMP` override block.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{RadEngineDamping, write_engine_damping};
/// let mut deck = String::new();
/// write_engine_damping(&mut deck, &RadEngineDamping { damp_id: 1, alpha: 40.0, beta_s: 0.0 });
/// assert!(deck.contains("/DAMP/1\n40"));
/// ```
pub fn write_engine_damping(deck: &mut String, damp: &RadEngineDamping) {
    deck.push_str(&format!("/DAMP/{}\n", damp.damp_id));
    deck.push_str(&format!("{} {}\n", damp.alpha, damp.beta_s));
}

/// Activation sensor (`/SENSOR/*`, guide pp.2274-2304).
///
/// Sensors fire airbags, imposed loads, and `/ACTIV` element sets.
/// Headers are `/SENSOR/TIME`, `/SENSOR/SENS`, `/SENSOR/DIST`,
/// `/SENSOR/AND`, and `/SENSOR/OR` (starter `hm_read_sensors.F`
/// accepts exactly these keys).
#[derive(Debug, Clone)]
pub enum RadSensor {
    /// Time-delay sensor (guide p.2304).
    Time {
        /// Sensor id written on the card.
        id: usize,
        /// Sensor title written on the card.
        title: String,
        /// Firing delay in seconds.
        tdelay_s: f64,
    },
    /// Sensor-driven sensor (guide p.2302).
    Sens {
        /// Sensor id written on the card.
        id: usize,
        /// Sensor title written on the card.
        title: String,
        /// Minimum activation duration in seconds.
        tdelay_s: f64,
        /// Activation sensor id.
        sens_on: usize,
        /// Deactivation sensor id (0 = none).
        sens_off: usize,
    },
    /// Distance sensor (guide p.2277).
    Dist {
        /// Sensor id written on the card.
        id: usize,
        /// Sensor title written on the card.
        title: String,
        /// Time delay in seconds.
        tdelay_s: f64,
        /// First node id.
        node_a: usize,
        /// Second node id.
        node_b: usize,
        /// Minimum distance in metres.
        dmin_m: f64,
        /// Maximum distance in metres.
        dmax_m: f64,
    },
    /// Boolean sensor over two sensors (guide p.2274).
    AndOr {
        /// Sensor id written on the card.
        id: usize,
        /// Sensor title written on the card.
        title: String,
        /// Time delay in seconds.
        tdelay_s: f64,
        /// First input sensor id.
        sens_a: usize,
        /// Second input sensor id.
        sens_b: usize,
        /// True for AND, false for OR.
        is_and: bool,
    },
}

impl RadSensor {
    /// Sensor id carried on the card header.
    pub fn id(&self) -> usize {
        match self {
            Self::Time { id, .. }
            | Self::Sens { id, .. }
            | Self::Dist { id, .. }
            | Self::AndOr { id, .. } => *id,
        }
    }
}

/// Render one `/SENSOR/*` block.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{RadSensor, write_sensor};
/// let mut deck = String::new();
/// write_sensor(&mut deck, &RadSensor::Time { id: 1, title: "fire".to_string(), tdelay_s: 0.0 });
/// assert!(deck.contains("/SENSOR/TIME/1\n"));
/// ```
pub fn write_sensor(deck: &mut String, sensor: &RadSensor) {
    match sensor {
        RadSensor::Time {
            id,
            title,
            tdelay_s,
        } => {
            deck.push_str(&format!("/SENSOR/TIME/{id}\n"));
            deck.push_str(&format!("{title}\n"));
            deck.push_str(&format!("{}\n", field_float(*tdelay_s)));
        }
        RadSensor::Sens {
            id,
            title,
            tdelay_s,
            sens_on,
            sens_off,
        } => {
            deck.push_str(&format!("/SENSOR/SENS/{id}\n"));
            deck.push_str(&format!("{title}\n"));
            deck.push_str(&format!("{}\n", field_float(*tdelay_s)));
            deck.push_str(&format!(
                "{}{}\n",
                field_int(*sens_on),
                field_int(*sens_off)
            ));
        }
        RadSensor::Dist {
            id,
            title,
            tdelay_s,
            node_a,
            node_b,
            dmin_m,
            dmax_m,
        } => {
            deck.push_str(&format!("/SENSOR/DIST/{id}\n"));
            deck.push_str(&format!("{title}\n"));
            deck.push_str(&format!("{}\n", field_float(*tdelay_s)));
            deck.push_str(&format!(
                "{}{}{}{}\n",
                field_int(*node_a),
                field_int(*node_b),
                field_float(*dmin_m),
                field_float(*dmax_m),
            ));
        }
        RadSensor::AndOr {
            id,
            title,
            tdelay_s,
            sens_a,
            sens_b,
            is_and,
        } => {
            let kind = if *is_and { "AND" } else { "OR" };
            deck.push_str(&format!("/SENSOR/{kind}/{id}\n"));
            deck.push_str(&format!("{title}\n"));
            deck.push_str(&format!("{}\n", field_float(*tdelay_s)));
            deck.push_str(&format!("{}{}\n", field_int(*sens_a), field_int(*sens_b)));
        }
    }
}

/// Submodel wrapper (`//SUBMODEL`, guide p.198).
///
/// Groups starter options under one id with numbering offsets;
/// zero offsets keep global numbering (deck-proven miniqa
/// `BOXBEAM_SUB` shape `//SUBMODEL/1` .. `//ENDSUB`).
#[derive(Debug, Clone)]
pub struct RadSubmodel {
    /// Submodel id written on the card.
    pub id: usize,
    /// Submodel title written on the card.
    pub title: String,
    /// Default offset for every option.
    pub off_def: i32,
    /// Node offset (defaults to `off_def`).
    pub off_nod: i32,
    /// Element offset (defaults to `off_def`).
    pub off_ele: i32,
    /// Part offset (defaults to `off_def`).
    pub off_part: i32,
    /// Material offset (defaults to `off_def`).
    pub off_mat: i32,
    /// Type offset (defaults to `off_def`).
    pub off_type: i32,
    /// Subset offset (defaults to `off_def`).
    pub off_sub: i32,
}

impl RadSubmodel {
    /// Transparent submodel with zero offsets.
    pub fn plain(id: usize, title: &str) -> Self {
        Self {
            id,
            title: title.to_string(),
            off_def: 0,
            off_nod: 0,
            off_ele: 0,
            off_part: 0,
            off_mat: 0,
            off_type: 0,
            off_sub: 0,
        }
    }
}

/// Open a `//SUBMODEL` section.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{RadSubmodel, write_submodel_open};
/// let mut deck = String::new();
/// write_submodel_open(&mut deck, &RadSubmodel::plain(1, "extra cards"));
/// assert!(deck.contains("//SUBMODEL/1\n"));
/// ```
pub fn write_submodel_open(deck: &mut String, sub: &RadSubmodel) {
    deck.push_str(&format!("//SUBMODEL/{}\n", sub.id));
    deck.push_str(&format!("{}\n", sub.title));
    deck.push_str(&format!(
        "{}{}{}{}{}{}{}\n",
        field_int32(sub.off_def),
        field_int32(sub.off_nod),
        field_int32(sub.off_ele),
        field_int32(sub.off_part),
        field_int32(sub.off_mat),
        field_int32(sub.off_type),
        field_int32(sub.off_sub),
    ));
}

/// Close a `//SUBMODEL` section (guide p.41).
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::write_submodel_close;
/// let mut deck = String::new();
/// write_submodel_close(&mut deck);
/// assert!(deck.contains("//ENDSUB\n"));
/// ```
pub fn write_submodel_close(deck: &mut String) {
    deck.push_str("//ENDSUB\n");
}

/// Element time-history family (`/TH/*`, guide p.2557).
///
/// Covers every element-object `/TH` header in the miniqa
/// inventory, including the `/TH/SHEL` and `/TH/BRIC` spellings
/// the base writer already emits (re-emitting them here under a
/// fresh group id is legal).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadThElemFamily {
    /// Nodal output (`/TH/NODE`).
    Node,
    /// Four-node shells, short spelling (`/TH/SHEL`).
    Shel,
    /// Three-node shells (`/TH/SH3N`).
    Sh3n,
    /// Four-node shells, long spelling (`/TH/SHELL`).
    Shell,
    /// Bricks, short spelling (`/TH/BRIC`).
    Bric,
    /// Bricks, long spelling (`/TH/BRICK`).
    Brick,
    /// Two-dimensional solids (`/TH/QUAD`).
    Quad,
    /// Trusses (`/TH/TRUSS`).
    Truss,
    /// Beams (`/TH/BEAM`).
    Beam,
    /// Springs (`/TH/SPRING`).
    Spring,
    /// SPH cells (`/TH/SPHCEL`).
    Sphcel,
}

impl RadThElemFamily {
    /// Header keyword of the family.
    pub fn keyword(self) -> &'static str {
        match self {
            Self::Node => "NODE",
            Self::Shel => "SHEL",
            Self::Sh3n => "SH3N",
            Self::Shell => "SHELL",
            Self::Bric => "BRIC",
            Self::Brick => "BRICK",
            Self::Quad => "QUAD",
            Self::Truss => "TRUSS",
            Self::Beam => "BEAM",
            Self::Spring => "SPRING",
            Self::Sphcel => "SPHCEL",
        }
    }

    /// Whether id lines carry a skew column.
    ///
    /// `NODE`, `SHEL`/`SHELL`, and `SH3N` use `Elid Skew_ID`
    /// lines; the remaining families use bare `Elid` lines (the
    /// second field is a name, and a zero there trips starter
    /// warning 100214, proven by spike against the vendor starter).
    pub fn has_skew(self) -> bool {
        matches!(self, Self::Node | Self::Shel | Self::Sh3n | Self::Shell)
    }
}

/// Element time-history request for one `/TH` family.
#[derive(Debug, Clone)]
pub struct RadThElemRequest {
    /// Output family selected.
    pub family: RadThElemFamily,
    /// Group title written on the card.
    pub title: String,
    /// Variable names (`DEF` selects the default set).
    pub vars: Vec<String>,
    /// Element ids recorded (1-based).
    pub elems: Vec<usize>,
}

/// Object time-history family (`/TH/*`, guide p.2557).
///
/// Covers every non-element `/TH` header in the miniqa inventory.
/// `/TH/MONV` is the short spelling of `/TH/MONVOL`; both appear
/// in the corpus and both are accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadThObjKind {
    /// Parts (`/TH/PART`).
    Part,
    /// Interfaces (`/TH/INTER`).
    Inter,
    /// Monitored volumes, short spelling (`/TH/MONV`).
    Monv,
    /// Monitored volumes (`/TH/MONVOL`).
    Monvol,
    /// Subsets (`/TH/SUBS`).
    Subs,
    /// Accelerometers (`/TH/ACCEL`).
    Accel,
    /// Rigid bodies (`/TH/RBODY`).
    Rbody,
    /// Sections (`/TH/SECTIO`).
    Sectio,
    /// Rigid walls (`/TH/RWALL`).
    Rwall,
}

impl RadThObjKind {
    /// Header keyword of the family.
    pub fn keyword(self) -> &'static str {
        match self {
            Self::Part => "PART",
            Self::Inter => "INTER",
            Self::Monv => "MONV",
            Self::Monvol => "MONVOL",
            Self::Subs => "SUBS",
            Self::Accel => "ACCEL",
            Self::Rbody => "RBODY",
            Self::Sectio => "SECTIO",
            Self::Rwall => "RWALL",
        }
    }
}

/// Object time-history request for one `/TH` family.
#[derive(Debug, Clone)]
pub struct RadThObjRequest {
    /// Output family selected.
    pub kind: RadThObjKind,
    /// Group title written on the card.
    pub title: String,
    /// Variable names (`DEF` selects the default set).
    pub vars: Vec<String>,
    /// Object ids recorded (1-based).
    pub ids: Vec<usize>,
}

/// Render element time-history blocks for one request.
///
/// Variables chunk into cards of ten; each card opens a fresh TH
/// group id from `next_group` (TH groups share one deck-global
/// namespace). Skew families emit `Elid Skew_ID` lines, name
/// families emit bare `Elid` lines.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{RadThElemFamily, RadThElemRequest, write_th_elems};
/// let mut deck = String::new();
/// let mut group = 4;
/// write_th_elems(&mut deck, &mut group, &RadThElemRequest {
///     family: RadThElemFamily::Truss, title: "truss".to_string(),
///     vars: vec!["DEF".to_string()], elems: vec![1],
/// });
/// assert!(deck.contains("/TH/TRUSS/5\n"));
/// assert_eq!(group, 5);
/// ```
pub fn write_th_elems(deck: &mut String, next_group: &mut usize, req: &RadThElemRequest) {
    if req.vars.is_empty() || req.elems.is_empty() {
        return;
    }
    for var_card in req.vars.chunks(10) {
        *next_group += 1;
        deck.push_str(&format!("/TH/{}/{}\n", req.family.keyword(), *next_group));
        deck.push_str(&format!("{}\n", req.title));
        let mut line = String::new();
        for var in var_card {
            line.push_str(&field_var(var));
        }
        line.push('\n');
        deck.push_str(&line);
        for elem in &req.elems {
            if req.family.has_skew() {
                deck.push_str(&format!("{}{}\n", field_int(*elem), field_int(0)));
            } else {
                deck.push_str(&format!("{}\n", field_int(*elem)));
            }
        }
    }
}

/// Render object time-history blocks for one request.
///
/// Id lines pack ten `Obj` ids per line (deck-proven miniqa
/// `/TH/PART`, `/TH/RWALL`, `/TH/MONVOL` shapes).
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{RadThObjKind, RadThObjRequest, write_th_objects};
/// let mut deck = String::new();
/// let mut group = 1;
/// write_th_objects(&mut deck, &mut group, &RadThObjRequest {
///     kind: RadThObjKind::Part, title: "parts".to_string(),
///     vars: vec!["DEF".to_string()], ids: vec![1, 2],
/// });
/// assert!(deck.contains("/TH/PART/2\n"));
/// assert_eq!(group, 2);
/// ```
pub fn write_th_objects(deck: &mut String, next_group: &mut usize, req: &RadThObjRequest) {
    if req.vars.is_empty() || req.ids.is_empty() {
        return;
    }
    for var_card in req.vars.chunks(10) {
        *next_group += 1;
        deck.push_str(&format!("/TH/{}/{}\n", req.kind.keyword(), *next_group));
        deck.push_str(&format!("{}\n", req.title));
        let mut line = String::new();
        for var in var_card {
            line.push_str(&field_var(var));
        }
        line.push('\n');
        deck.push_str(&line);
        for chunk in req.ids.chunks(10) {
            let mut id_line = String::new();
            for id in chunk {
                id_line.push_str(&field_int(*id));
            }
            id_line.push('\n');
            deck.push_str(&id_line);
        }
    }
}

/// Truss elements rendered as `/TRUSS/part_ID`.
///
/// Support card so `/TH/TRUSS` acceptance can reference a real
/// element; the two-node shape mirrors miniqa `BESHV44`.
#[derive(Debug, Clone)]
pub struct RadTrussElems {
    /// Part id written on the header.
    pub part: usize,
    /// Two-node connectivity per element (1-based node ids).
    pub elems: Vec<[usize; 2]>,
}

/// Truss property rendered as `/PROP/TRUSS` (miniqa `BESHV44`).
#[derive(Debug, Clone)]
pub struct RadTrussProp {
    /// Property id written on the card.
    pub id: usize,
    /// Property title written on the card.
    pub title: String,
    /// Cross-section area in m^2.
    pub area_m2: f64,
    /// Initial gap in metres (0 = none).
    pub gap_init_m: f64,
}

/// Beam elements rendered as `/BEAM/part_ID`.
///
/// Support card so `/TH/BEAM` acceptance can reference a real
/// element; the three-node shape mirrors miniqa `BESHV44`.
#[derive(Debug, Clone)]
pub struct RadBeamElems {
    /// Part id written on the header.
    pub part: usize,
    /// Three-node connectivity per element (1-based node ids).
    pub elems: Vec<[usize; 3]>,
}

/// Beam property rendered as `/PROP/BEAM` (miniqa `BESHV44`).
#[derive(Debug, Clone)]
pub struct RadBeamProp {
    /// Property id written on the card.
    pub id: usize,
    /// Property title written on the card.
    pub title: String,
    /// Small-strain flag (0 = default).
    pub ismstr: i32,
    /// Membrane damping factor.
    pub dm: f64,
    /// Flexural damping factor.
    pub df: f64,
    /// Cross-section area in m^2.
    pub area_m2: f64,
    /// Second moments of area in m^4 (`Iyy Izz Ixx`).
    pub inertia_m4: [f64; 3],
    /// Shear flag (0 = default).
    pub ishear: i32,
}

/// Spring elements rendered as `/SPRING/part_ID`.
///
/// Support card so `/TH/SPRING` acceptance can reference a real
/// element; the two-node shape mirrors miniqa `PLREG41B`.
#[derive(Debug, Clone)]
pub struct RadSpringElems {
    /// Part id written on the header.
    pub part: usize,
    /// Two-node connectivity per element (1-based node ids).
    pub elems: Vec<[usize; 2]>,
}

/// General spring property (`/PROP/TYPE13`, guide p.1490).
///
/// Support card for spring acceptance. Translational stiffness
/// applies to all three axes and rotational stiffness to all
/// three moments; every other coefficient is zero (linear
/// springs, no functions, SI units). Travel limits are +/-0.01 m
/// per axis, zero for rotations.
#[derive(Debug, Clone)]
pub struct RadSpringProp {
    /// Property id written on the card.
    pub id: usize,
    /// Property title written on the card.
    pub title: String,
    /// Spring mass in kg.
    pub mass_kg: f64,
    /// Spring inertia in kg m^2.
    pub inertia: f64,
    /// Translational stiffness in N/m.
    pub k_trans: f64,
    /// Rotational stiffness in N m/rad.
    pub k_rot: f64,
}

/// Extra part rendered as `/PART`.
///
/// The base writers own parts 1..N; this covers additional parts
/// (1D elements) with ids above the base range.
#[derive(Debug, Clone)]
pub struct RadExtraPart {
    /// Part id written on the card.
    pub id: usize,
    /// Part title written on the card.
    pub title: String,
    /// Property id referenced.
    pub prop: usize,
    /// Material id referenced (0 = none, for springs).
    pub mat: usize,
}

/// Render one `/TRUSS/part_ID` block.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{RadTrussElems, write_truss};
/// let mut deck = String::new();
/// write_truss(&mut deck, &RadTrussElems { part: 3, elems: vec![[1, 2]] });
/// assert!(deck.contains("/TRUSS/3\n"));
/// ```
pub fn write_truss(deck: &mut String, truss: &RadTrussElems) {
    deck.push_str(&format!("/TRUSS/{}\n", truss.part));
    for (index, elem) in truss.elems.iter().enumerate() {
        deck.push_str(&format!(
            "{}{}{}\n",
            field_int(index + 1),
            field_int(elem[0]),
            field_int(elem[1]),
        ));
    }
}

/// Render one `/PROP/TRUSS` block.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{RadTrussProp, write_truss_prop};
/// let mut deck = String::new();
/// write_truss_prop(&mut deck, &RadTrussProp { id: 3, title: "rod".to_string(), area_m2: 1.0e-4, gap_init_m: 0.0 });
/// assert!(deck.contains("/PROP/TRUSS/3\n"));
/// ```
pub fn write_truss_prop(deck: &mut String, prop: &RadTrussProp) {
    deck.push_str(&format!("/PROP/TRUSS/{}\n", prop.id));
    deck.push_str(&format!("{}\n", prop.title));
    deck.push_str(&format!(
        "{}{}\n",
        field_float(prop.area_m2),
        field_float(prop.gap_init_m),
    ));
}

/// Render one `/BEAM/part_ID` block.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{RadBeamElems, write_beam};
/// let mut deck = String::new();
/// write_beam(&mut deck, &RadBeamElems { part: 4, elems: vec![[1, 2, 3]] });
/// assert!(deck.contains("/BEAM/4\n"));
/// ```
pub fn write_beam(deck: &mut String, beam: &RadBeamElems) {
    deck.push_str(&format!("/BEAM/{}\n", beam.part));
    for (index, elem) in beam.elems.iter().enumerate() {
        deck.push_str(&format!(
            "{}{}{}{}\n",
            field_int(index + 1),
            field_int(elem[0]),
            field_int(elem[1]),
            field_int(elem[2]),
        ));
    }
}

/// Render one `/PROP/BEAM` block.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{RadBeamProp, write_beam_prop};
/// let mut deck = String::new();
/// write_beam_prop(&mut deck, &RadBeamProp {
///     id: 4, title: "beam".to_string(), ismstr: 0, dm: 0.0, df: 0.0,
///     area_m2: 1.0e-4, inertia_m4: [1.0e-9, 1.0e-9, 2.0e-9], ishear: 0,
/// });
/// assert!(deck.contains("/PROP/BEAM/4\n"));
/// ```
pub fn write_beam_prop(deck: &mut String, prop: &RadBeamProp) {
    deck.push_str(&format!("/PROP/BEAM/{}\n", prop.id));
    deck.push_str(&format!("{}\n", prop.title));
    // `Ismstr` sits in field 2 (`"          %10d"`, hm_cfg_files
    // radioss110 `prop_p3_beam.cfg`).
    deck.push_str(&format!("{}{}\n", field_blank(), field_int32(prop.ismstr)));
    deck.push_str(&format!(
        "{}{}\n",
        field_float(prop.dm),
        field_float(prop.df),
    ));
    deck.push_str(&format!(
        "{}{}{}{}\n",
        field_float(prop.area_m2),
        field_float(prop.inertia_m4[0]),
        field_float(prop.inertia_m4[1]),
        field_float(prop.inertia_m4[2]),
    ));
    // `OmegaDof` triplet (unused) plus `Ishear`, mirroring `BESHV44`.
    deck.push_str(&format!("   000 000{}\n", field_int32(prop.ishear)));
}

/// Render one `/SPRING/part_ID` block.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{RadSpringElems, write_spring};
/// let mut deck = String::new();
/// write_spring(&mut deck, &RadSpringElems { part: 5, elems: vec![[1, 2]] });
/// assert!(deck.contains("/SPRING/5\n"));
/// ```
pub fn write_spring(deck: &mut String, spring: &RadSpringElems) {
    deck.push_str(&format!("/SPRING/{}\n", spring.part));
    for (index, elem) in spring.elems.iter().enumerate() {
        deck.push_str(&format!(
            "{}{}{}\n",
            field_int(index + 1),
            field_int(elem[0]),
            field_int(elem[1]),
        ));
    }
}

/// Render one `/PROP/TYPE13` spring property block.
///
/// Twenty-six data lines per the newest reader layout
/// (hm_cfg_files radioss2018 `prop_p13_spr_beam.cfg`): mass and
/// inertia flags, six three-line loading indices (stiffness,
/// function ids plus travel limits, scaling), one rate-velocity
/// line, and six rate-coefficient lines.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{RadSpringProp, write_spring_prop};
/// let mut deck = String::new();
/// write_spring_prop(&mut deck, &RadSpringProp {
///     id: 5, title: "spring".to_string(), mass_kg: 1.0e-3,
///     inertia: 1.0e-3, k_trans: 10000.0, k_rot: 100000.0,
/// });
/// assert!(deck.contains("/PROP/TYPE13/5\n"));
/// ```
pub fn write_spring_prop(deck: &mut String, prop: &RadSpringProp) {
    deck.push_str(&format!("/PROP/TYPE13/{}\n", prop.id));
    deck.push_str(&format!("{}\n", prop.title));
    deck.push_str(&format!(
        "{}{}{}{}{}{}{}{}\n",
        field_float(prop.mass_kg),
        field_float(prop.inertia),
        field_int(0),
        field_int(0),
        field_int(0),
        field_int(0),
        field_int(0),
        field_int(0),
    ));
    // Function lines carry five ids, one blank field, then the
    // travel limits (`%10d%10d%10d%10d%10d %20lg%20lg`).
    for (stiffness, limits) in [
        (prop.k_trans, (-0.01, 0.01)),
        (prop.k_trans, (-0.01, 0.01)),
        (prop.k_trans, (-0.01, 0.01)),
        (prop.k_rot, (0.0, 0.0)),
        (prop.k_rot, (0.0, 0.0)),
        (prop.k_rot, (0.0, 0.0)),
    ] {
        deck.push_str(&format!(
            "{}{}{}{}{}\n",
            field_float(stiffness),
            field_float(0.0),
            field_float(0.0),
            field_float(0.0),
            field_float(0.0),
        ));
        deck.push_str(&format!(
            "{}{}{}{}{}{}{}{}\n",
            field_int(0),
            field_int(0),
            field_int(0),
            field_int(0),
            field_int(0),
            field_blank(),
            field_float(limits.0),
            field_float(limits.1),
        ));
        deck.push_str(&format!(
            "{}{}{}{}\n",
            field_float(0.0),
            field_float(0.0),
            field_float(0.0),
            field_float(0.0),
        ));
    }
    // Rate-velocity line plus six rate-coefficient lines.
    deck.push_str(&format!(
        "{}{}{}{}\n",
        field_float(0.0),
        field_float(0.0),
        field_float(0.0),
        field_int(0),
    ));
    for _ in 0..6 {
        deck.push_str(&format!(
            "{}{}{}{}\n",
            field_float(0.0),
            field_float(0.0),
            field_float(0.0),
            field_float(0.0),
        ));
    }
}

/// Render one extra `/PART` block.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{RadExtraPart, write_extra_part};
/// let mut deck = String::new();
/// write_extra_part(&mut deck, &RadExtraPart { id: 3, title: "rod".to_string(), prop: 3, mat: 1 });
/// assert!(deck.contains("/PART/3\n"));
/// ```
pub fn write_extra_part(deck: &mut String, part: &RadExtraPart) {
    deck.push_str(&format!("/PART/{}\n", part.id));
    deck.push_str(&format!("{}\n", part.title));
    deck.push_str(&format!(
        "{}{}\n",
        field_int(part.prop),
        field_int(part.mat),
    ));
}

/// State-file output (`/STATE/DT`, guide p.3051).
///
/// Engine option: writes `.sta` state files every `tfreq_s`
/// starting at `tstart_s` for the listed parts (empty with
/// `all_parts` selects `Keyword3 = ALL`).
#[derive(Debug, Clone)]
pub struct RadStateOutput {
    /// First state output time in seconds.
    pub tstart_s: f64,
    /// State output period in seconds.
    pub tfreq_s: f64,
    /// True for `ALL` parts, false for the part list.
    pub all_parts: bool,
    /// Part ids written (ignored when `all_parts` is set).
    pub parts: Vec<usize>,
}

/// State-file content flag (`/STATE/*`, guide pp.3046-3052).
///
/// Bare-header engine options selecting which blocks land in each
/// `.sta` file for the `/STATE/DT` parts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadStateFlag {
    /// Beam internal variables (`/STATE/BEAM/AUX`).
    BeamAux,
    /// Beam full state (`/STATE/BEAM/FULL`).
    BeamFull,
    /// Solid internal variables (`/STATE/BRICK/AUX/FULL`).
    BrickAux,
    /// Solid reference strain (`/STATE/BRICK/EREF`).
    BrickEref,
    /// Solid failure state (`/STATE/BRICK/FAIL`).
    BrickFail,
    /// Solid orthotropy (`/STATE/BRICK/ORTHO`).
    BrickOrtho,
    /// Solid strain (`/STATE/BRICK/STRAIN/FULL`).
    BrickStrain,
    /// Solid stress (`/STATE/BRICK/STRES/FULL`).
    BrickStress,
    /// Nodal boundary conditions (`/STATE/NODE/BCS`).
    NodeBcs,
    /// Nodal temperatures (`/STATE/NODE/TEMP`).
    NodeTemp,
    /// Nodal velocities (`/STATE/NODE/VEL`).
    NodeVel,
    /// Shell internal variables (`/STATE/SHELL/AUX/FULL`).
    ShellAux,
    /// Shell plastic strain (`/STATE/SHELL/EPSP/FULL`).
    ShellEpsp,
    /// Shell failure state (`/STATE/SHELL/FAIL`).
    ShellFail,
    /// Shell orthotropy (`/STATE/SHELL/ORTHL`).
    ShellOrthl,
    /// Shell strain (`/STATE/SHELL/STRAIN/FULL`).
    ShellStrain,
    /// Shell stress (`/STATE/SHELL/STRESS/FULL`).
    ShellStress,
    /// Spring state (`/STATE/SPRING/FULL`).
    SpringFull,
    /// Truss state (`/STATE/TRUSS/FULL`).
    TrussFull,
}

impl RadStateFlag {
    /// Engine header of the flag.
    pub fn header(self) -> &'static str {
        match self {
            Self::BeamAux => "/STATE/BEAM/AUX",
            Self::BeamFull => "/STATE/BEAM/FULL",
            Self::BrickAux => "/STATE/BRICK/AUX/FULL",
            Self::BrickEref => "/STATE/BRICK/EREF",
            Self::BrickFail => "/STATE/BRICK/FAIL",
            Self::BrickOrtho => "/STATE/BRICK/ORTHO",
            Self::BrickStrain => "/STATE/BRICK/STRAIN/FULL",
            Self::BrickStress => "/STATE/BRICK/STRES/FULL",
            Self::NodeBcs => "/STATE/NODE/BCS",
            Self::NodeTemp => "/STATE/NODE/TEMP",
            Self::NodeVel => "/STATE/NODE/VEL",
            Self::ShellAux => "/STATE/SHELL/AUX/FULL",
            Self::ShellEpsp => "/STATE/SHELL/EPSP/FULL",
            Self::ShellFail => "/STATE/SHELL/FAIL",
            Self::ShellOrthl => "/STATE/SHELL/ORTHL",
            Self::ShellStrain => "/STATE/SHELL/STRAIN/FULL",
            Self::ShellStress => "/STATE/SHELL/STRESS/FULL",
            Self::SpringFull => "/STATE/SPRING/FULL",
            Self::TrussFull => "/STATE/TRUSS/FULL",
        }
    }
}

/// Render one `/STATE/DT` engine block.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{RadStateOutput, write_state_dt};
/// let mut deck = String::new();
/// write_state_dt(&mut deck, &RadStateOutput { tstart_s: 0.0, tfreq_s: 0.1, all_parts: false, parts: vec![1, 2] });
/// assert!(deck.contains("/STATE/DT\n"));
/// ```
pub fn write_state_dt(deck: &mut String, state: &RadStateOutput) {
    if state.all_parts {
        deck.push_str("/STATE/DT/ALL\n");
    } else {
        deck.push_str("/STATE/DT\n");
    }
    deck.push_str(&format!("{} {}\n", state.tstart_s, state.tfreq_s));
    if !state.all_parts {
        let mut line = String::new();
        for part in &state.parts {
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(&part.to_string());
        }
        line.push('\n');
        deck.push_str(&line);
    }
}

/// Render one `/STATE/*` flag header.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{RadStateFlag, write_state_flag};
/// let mut deck = String::new();
/// write_state_flag(&mut deck, RadStateFlag::BrickStress);
/// assert!(deck.contains("/STATE/BRICK/STRES/FULL\n"));
/// ```
pub fn write_state_flag(deck: &mut String, flag: RadStateFlag) {
    deck.push_str(flag.header());
    deck.push('\n');
}

/// Extra starter and engine cards (Phase-4 lane 4B bundle).
///
/// P4C1 keeps `OpenCourantCase` frozen, so every 4B card family lives here
/// and renders through [`write_starter_extra`] (spliced before the
/// terminal `/END`) and [`write_engine_extra`] (appended to the
/// engine deck). All ids are absolute: extra ids must not collide
/// with base-writer ids (see [`RadExtraContext`]).
#[derive(Debug, Clone, Default)]
pub struct RadExtra {
    /// Part surfaces (`/SURF/PART`).
    pub surfs: Vec<RadSurfParts>,
    /// Surface node groups (`/GRNOD/SURF`).
    pub grnods: Vec<RadGrnodSurfs>,
    /// Surface line groups (`/LINE/SURF`).
    pub lines: Vec<RadLineSurfs>,
    /// Part subsets (`/SUBSET`).
    pub subsets: Vec<RadSubset>,
    /// Contact interfaces (`/INTER/TYPE*`).
    pub interfaces: Vec<RadInterface>,
    /// Airbag gases (`/MAT/GAS/PREDEF`).
    pub gases: Vec<RadGasPredef>,
    /// Airbag injectors (`/PROP/INJECT2`).
    pub injectors: Vec<RadInject2>,
    /// Monitored volumes (`/MONVOL/*`).
    pub monvols: Vec<RadMonvol>,
    /// Rayleigh damping cards (`/DAMP` short format).
    pub damps: Vec<RadDamping>,
    /// Rayleigh damping cards (`/DAMP` full format).
    pub full_damps: Vec<RadDampingFull>,
    /// Activation sensors (`/SENSOR/*`).
    pub sensors: Vec<RadSensor>,
    /// Optional submodel wrapping the whole extra block.
    pub submodel: Option<RadSubmodel>,
    /// Element time-history requests (`/TH/*`).
    pub th_elems: Vec<RadThElemRequest>,
    /// Object time-history requests (`/TH/*`).
    pub th_objs: Vec<RadThObjRequest>,
    /// Truss properties (`/PROP/TRUSS`).
    pub truss_props: Vec<RadTrussProp>,
    /// Beam properties (`/PROP/BEAM`).
    pub beam_props: Vec<RadBeamProp>,
    /// Spring properties (`/PROP/TYPE13`).
    pub spring_props: Vec<RadSpringProp>,
    /// Truss elements (`/TRUSS`).
    pub truss: Vec<RadTrussElems>,
    /// Beam elements (`/BEAM`).
    pub beams: Vec<RadBeamElems>,
    /// Spring elements (`/SPRING`).
    pub springs: Vec<RadSpringElems>,
    /// Extra parts (`/PART`, ids above the base range).
    pub parts: Vec<RadExtraPart>,
    /// State-file output (`/STATE/DT`, engine side).
    pub state: Option<RadStateOutput>,
    /// State-file content flags (`/STATE/*`, engine side).
    pub state_flags: Vec<RadStateFlag>,
    /// Engine damping overrides (`/DAMP`, engine side).
    pub engine_damps: Vec<RadEngineDamping>,
    /// Time-history format (`/TH/VERS/*`, engine side).
    pub th_vers: Option<u32>,
}

/// Base-deck counts validating a [`RadExtra`].
///
/// The base writers own `/GRNOD/NODE` ids `1..=num_node_sets`,
/// `/FUNCT` ids `1..=num_functions`, parts `1..=num_parts`, and
/// material id 1; extra cards reference or extend these ranges.
#[derive(Debug, Clone)]
pub struct RadExtraContext {
    /// Node count of the base mesh.
    pub num_nodes: usize,
    /// Node-set count (`/GRNOD/NODE` ids `1..=N`).
    pub num_node_sets: usize,
    /// Load-curve count (`/FUNCT` ids `1..=N`).
    pub num_functions: usize,
    /// Base part count (`/PART` ids `1..=N`).
    pub num_parts: usize,
    /// Base material count (always id 1).
    pub num_materials: usize,
    /// Wall count (`/RWALL` ids `1..=N`).
    pub num_walls: usize,
    /// Shell quad count.
    pub num_shells: usize,
    /// Shell triangle count.
    pub num_tris: usize,
    /// Brick count.
    pub num_bricks: usize,
}

impl RadExtraContext {
    /// Context of a basic [`OpenCourantCase`] deck.
    pub fn of_case(case: &OpenCourantCase) -> Self {
        Self {
            num_nodes: case.mesh.num_nodes(),
            num_node_sets: case.node_sets.len(),
            num_functions: case.functions.len(),
            num_parts: 1,
            num_materials: 1,
            num_walls: case.walls.len(),
            num_shells: 0,
            num_tris: 0,
            num_bricks: case.mesh.num_bricks(),
        }
    }

    /// Context of an extended [`RadExtendedCase`] deck.
    pub fn of_extended(case: &RadExtendedCase) -> Self {
        let mut parts = 0;
        if case.shell_part().is_some() {
            parts += 1;
        }
        if case.solid_part().is_some() {
            parts += 1;
        }
        let mut ctx = Self::of_case(&case.base);
        ctx.num_parts = parts;
        ctx.num_shells = case.shells.num_quads();
        ctx.num_tris = case.shells.num_tris();
        ctx
    }
}

fn check_ids_unique(ids: &[usize], what: &str) -> Result<(), RadError> {
    let mut seen = std::collections::BTreeSet::new();
    for id in ids {
        if *id == 0 || !seen.insert(*id) {
            return Err(RadError::Parse(format!(
                "{what} has a zero or duplicate id {id}"
            )));
        }
    }
    Ok(())
}

fn check_th_vars(vars: &[String], what: &str) -> Result<(), RadError> {
    if vars.is_empty() {
        return Err(RadError::Parse(format!(
            "{what} needs at least one variable"
        )));
    }
    for var in vars {
        if var.is_empty() || var.len() > 10 {
            return Err(RadError::Parse(format!(
                "{what} variable {var} must use 1..=10 characters"
            )));
        }
    }
    Ok(())
}

impl RadExtra {
    /// Node-group ids defined by this bundle (`/GRNOD/SURF`).
    fn extra_grnods(&self) -> Vec<usize> {
        self.grnods.iter().map(|group| group.id).collect()
    }

    /// Whether a node group id exists (base sets or extra).
    fn has_grnod(&self, ctx: &RadExtraContext, id: usize) -> bool {
        (1..=ctx.num_node_sets).contains(&id) || self.extra_grnods().contains(&id)
    }

    /// Whether a part id exists (base parts or extra).
    fn has_part(&self, ctx: &RadExtraContext, id: usize) -> bool {
        (1..=ctx.num_parts).contains(&id) || self.parts.iter().any(|part| part.id == id)
    }

    /// Whether a property id exists (base props or extra).
    fn has_prop(&self, ctx: &RadExtraContext, id: usize) -> bool {
        (1..=ctx.num_parts).contains(&id)
            || self.truss_props.iter().any(|prop| prop.id == id)
            || self.beam_props.iter().any(|prop| prop.id == id)
            || self.spring_props.iter().any(|prop| prop.id == id)
            || self.injectors.iter().any(|inject| inject.id == id)
    }

    /// Check every id, reference, and card range of the bundle.
    ///
    /// References resolve against `ctx` (base deck) plus this
    /// bundle. Dangling `/TH/SECTIO`, `/TH/RBODY`, and `/TH/ACCEL`
    /// ids pass: the starter only warns (id 257) on those, proven
    /// by spike, since sections, rigid bodies, and accelerometers
    /// belong to other lanes.
    ///
    /// # Errors
    ///
    /// Returns [`RadError::Parse`] for the first inconsistency found.
    pub fn validate(&self, ctx: &RadExtraContext) -> Result<(), RadError> {
        self.validate_with_surface_ids(ctx, &[])
    }

    fn validate_with_surface_ids(
        &self,
        ctx: &RadExtraContext,
        segment_surface_ids: &[usize],
    ) -> Result<(), RadError> {
        check_ids_unique(
            &self
                .surfs
                .iter()
                .map(|surf| surf.id)
                .chain(segment_surface_ids.iter().copied())
                .collect::<Vec<_>>(),
            "/SURF",
        )?;
        check_ids_unique(
            &self.grnods.iter().map(|group| group.id).collect::<Vec<_>>(),
            "/GRNOD/SURF",
        )?;
        check_ids_unique(
            &self.lines.iter().map(|line| line.id).collect::<Vec<_>>(),
            "/LINE/SURF",
        )?;
        check_ids_unique(
            &self
                .subsets
                .iter()
                .map(|subset| subset.id)
                .collect::<Vec<_>>(),
            "/SUBSET",
        )?;
        check_ids_unique(
            &self
                .interfaces
                .iter()
                .map(RadInterface::id)
                .collect::<Vec<_>>(),
            "/INTER",
        )?;
        check_ids_unique(
            &self.gases.iter().map(|gas| gas.id).collect::<Vec<_>>(),
            "/MAT/GAS/PREDEF",
        )?;
        check_ids_unique(
            &self
                .injectors
                .iter()
                .map(|inject| inject.id)
                .collect::<Vec<_>>(),
            "/PROP/INJECT2",
        )?;
        check_ids_unique(
            &self.monvols.iter().map(RadMonvol::id).collect::<Vec<_>>(),
            "/MONVOL",
        )?;
        check_ids_unique(
            &self.sensors.iter().map(RadSensor::id).collect::<Vec<_>>(),
            "/SENSOR",
        )?;
        let mut damp_ids: Vec<usize> = self.damps.iter().map(|damp| damp.id).collect();
        damp_ids.extend(self.full_damps.iter().map(|damp| damp.id));
        check_ids_unique(&damp_ids, "/DAMP")?;
        check_ids_unique(
            &self.parts.iter().map(|part| part.id).collect::<Vec<_>>(),
            "extra /PART",
        )?;
        let mut prop_ids: Vec<usize> = self.truss_props.iter().map(|prop| prop.id).collect();
        prop_ids.extend(self.beam_props.iter().map(|prop| prop.id));
        prop_ids.extend(self.spring_props.iter().map(|prop| prop.id));
        prop_ids.extend(self.injectors.iter().map(|inject| inject.id));
        check_ids_unique(&prop_ids, "extra /PROP")?;
        let surf_ids: Vec<usize> = self
            .surfs
            .iter()
            .map(|surf| surf.id)
            .chain(segment_surface_ids.iter().copied())
            .collect();
        let line_ids: Vec<usize> = self.lines.iter().map(|line| line.id).collect();
        let sensor_ids: Vec<usize> = self.sensors.iter().map(RadSensor::id).collect();
        let gas_ids: Vec<usize> = self.gases.iter().map(|gas| gas.id).collect();
        let injector_ids: Vec<usize> = self.injectors.iter().map(|inject| inject.id).collect();
        // Extra groups must not collide with base `/GRNOD/NODE` ids.
        for group in &self.grnods {
            if (1..=ctx.num_node_sets).contains(&group.id) {
                return Err(RadError::Parse(format!(
                    "/GRNOD/SURF/{} collides with a base node set",
                    group.id
                )));
            }
            if group.surfs.is_empty() {
                return Err(RadError::Parse(format!(
                    "/GRNOD/SURF/{} needs at least one surface",
                    group.id
                )));
            }
            for surf in &group.surfs {
                if !surf_ids.contains(surf) {
                    return Err(RadError::Parse(format!(
                        "/GRNOD/SURF/{} references missing surface {surf}",
                        group.id
                    )));
                }
            }
        }
        for line in &self.lines {
            if line.surfs.is_empty() {
                return Err(RadError::Parse(format!(
                    "/LINE/SURF/{} needs at least one surface",
                    line.id
                )));
            }
            for surf in &line.surfs {
                if !surf_ids.contains(surf) {
                    return Err(RadError::Parse(format!(
                        "/LINE/SURF/{} references missing surface {surf}",
                        line.id
                    )));
                }
            }
        }
        for surf in &self.surfs {
            if surf.parts.is_empty() {
                return Err(RadError::Parse(format!(
                    "/SURF/PART/{} needs at least one part",
                    surf.id
                )));
            }
            for part in &surf.parts {
                if !self.has_part(ctx, *part) {
                    return Err(RadError::Parse(format!(
                        "/SURF/PART/{} references missing part {part}",
                        surf.id
                    )));
                }
            }
        }
        for inter in &self.interfaces {
            match inter {
                RadInterface::Type2(typed) => {
                    if !self.has_grnod(ctx, typed.grnd_secondary) {
                        return Err(RadError::Parse(format!(
                            "/INTER/TYPE2/{} references missing group {}",
                            typed.id, typed.grnd_secondary
                        )));
                    }
                    if !surf_ids.contains(&typed.surf_main) {
                        return Err(RadError::Parse(format!(
                            "/INTER/TYPE2/{} references missing surface {}",
                            typed.id, typed.surf_main
                        )));
                    }
                    if matches!(typed.spotflag, 20..=22 | 25 | 27 | 28) {
                        return Err(RadError::Parse(format!(
                            "/INTER/TYPE2/{} Spotflag {} needs an extension card",
                            typed.id, typed.spotflag
                        )));
                    }
                    if !typed.dsearch_m.is_finite() {
                        return Err(RadError::Parse(format!(
                            "/INTER/TYPE2/{} needs a finite search distance",
                            typed.id
                        )));
                    }
                }
                RadInterface::Type7(typed) => {
                    if !self.has_grnod(ctx, typed.grnd_secondary) {
                        return Err(RadError::Parse(format!(
                            "/INTER/TYPE7/{} references missing group {}",
                            typed.id, typed.grnd_secondary
                        )));
                    }
                    if !surf_ids.contains(&typed.surf_main) {
                        return Err(RadError::Parse(format!(
                            "/INTER/TYPE7/{} references missing surface {}",
                            typed.id, typed.surf_main
                        )));
                    }
                    if typed.icurv == 1 || typed.icurv == 2 {
                        return Err(RadError::Parse(format!(
                            "/INTER/TYPE7/{} Icurv {} needs a curvature-nodes line",
                            typed.id, typed.icurv
                        )));
                    }
                    if typed.ifric > 0 || typed.ithe > 1 || typed.iadm == 2 {
                        return Err(RadError::Parse(format!(
                            "/INTER/TYPE7/{} needs an out-of-scope extension card",
                            typed.id
                        )));
                    }
                }
                RadInterface::Type11(typed) => {
                    if !line_ids.contains(&typed.line_secondary)
                        || !line_ids.contains(&typed.line_main)
                    {
                        return Err(RadError::Parse(format!(
                            "/INTER/TYPE11/{} references a missing line",
                            typed.id
                        )));
                    }
                    if typed.ithe > 0 {
                        return Err(RadError::Parse(format!(
                            "/INTER/TYPE11/{} Ithe {} needs a thermal extension",
                            typed.id, typed.ithe
                        )));
                    }
                }
            }
        }
        for gas in &self.gases {
            if gas.id <= ctx.num_materials {
                return Err(RadError::Parse(format!(
                    "/MAT/GAS/PREDEF/{} collides with a base material",
                    gas.id
                )));
            }
        }
        for inject in &self.injectors {
            if inject.gases.is_empty() {
                return Err(RadError::Parse(format!(
                    "/PROP/INJECT2/{} needs at least one gas",
                    inject.id
                )));
            }
            if inject.id <= ctx.num_parts {
                return Err(RadError::Parse(format!(
                    "/PROP/INJECT2/{} collides with a base property",
                    inject.id
                )));
            }
            for fun in [inject.mass_function, inject.temp_function] {
                if fun == 0 || fun > ctx.num_functions {
                    return Err(RadError::Parse(format!(
                        "/PROP/INJECT2/{} references missing function {fun}",
                        inject.id
                    )));
                }
            }
            for gas in &inject.gases {
                if !gas_ids.contains(&gas.mat) {
                    return Err(RadError::Parse(format!(
                        "/PROP/INJECT2/{} references missing gas {}",
                        inject.id, gas.mat
                    )));
                }
                if gas.function > ctx.num_functions {
                    return Err(RadError::Parse(format!(
                        "/PROP/INJECT2/{} references missing function {}",
                        inject.id, gas.function
                    )));
                }
            }
        }
        self.validate_monvols(&surf_ids, &gas_ids, &injector_ids, &sensor_ids, ctx)?;
        for damp in &self.damps {
            if !self.has_grnod(ctx, damp.grnd) {
                return Err(RadError::Parse(format!(
                    "/DAMP/{} references missing group {}",
                    damp.id, damp.grnd
                )));
            }
        }
        for damp in &self.full_damps {
            if !self.has_grnod(ctx, damp.grnd) {
                return Err(RadError::Parse(format!(
                    "/DAMP/{} references missing group {}",
                    damp.id, damp.grnd
                )));
            }
        }
        for sensor in &self.sensors {
            match sensor {
                RadSensor::Time { .. } => {}
                RadSensor::Sens {
                    id,
                    sens_on,
                    sens_off,
                    ..
                } => {
                    if !sensor_ids.contains(sens_on) {
                        return Err(RadError::Parse(format!(
                            "/SENSOR/SENS/{id} references missing sensor {sens_on}"
                        )));
                    }
                    if *sens_off > 0 && !sensor_ids.contains(sens_off) {
                        return Err(RadError::Parse(format!(
                            "/SENSOR/SENS/{id} references missing sensor {sens_off}"
                        )));
                    }
                }
                RadSensor::Dist {
                    id, node_a, node_b, ..
                } => {
                    for node in [node_a, node_b] {
                        if *node == 0 || *node > ctx.num_nodes {
                            return Err(RadError::Parse(format!(
                                "/SENSOR/DIST/{id} references missing node {node}"
                            )));
                        }
                    }
                }
                RadSensor::AndOr {
                    id, sens_a, sens_b, ..
                } => {
                    for sens in [sens_a, sens_b] {
                        if !sensor_ids.contains(sens) {
                            return Err(RadError::Parse(format!(
                                "/SENSOR/{id} references missing sensor {sens}"
                            )));
                        }
                    }
                }
            }
        }
        self.validate_1d(ctx)?;
        self.validate_th(ctx)?;
        if self.submodel.as_ref().is_some_and(|sub| sub.id == 0) {
            return Err(RadError::Parse("//SUBMODEL needs a nonzero id".to_string()));
        }
        if let Some(state) = &self.state {
            if !state.tstart_s.is_finite() || !state.tfreq_s.is_finite() || state.tfreq_s <= 0.0 {
                return Err(RadError::Parse(
                    "/STATE/DT needs a finite start and positive period".to_string(),
                ));
            }
            if !state.all_parts {
                if state.parts.is_empty() {
                    return Err(RadError::Parse("/STATE/DT needs parts or ALL".to_string()));
                }
                for part in &state.parts {
                    if !self.has_part(ctx, *part) {
                        return Err(RadError::Parse(format!(
                            "/STATE/DT references missing part {part}"
                        )));
                    }
                }
            }
        }
        for damp in &self.engine_damps {
            if !damp_ids.contains(&damp.damp_id) {
                return Err(RadError::Parse(format!(
                    "engine /DAMP/{} has no starter card",
                    damp.damp_id
                )));
            }
        }
        if let Some(vers) = self.th_vers
            && !matches!(vers, 41 | 51 | 2021)
        {
            return Err(RadError::Parse(format!(
                "/TH/VERS/{vers} must be 41, 51, or 2021"
            )));
        }
        Ok(())
    }
}

impl RadExtra {
    /// Validate monitored-volume references.
    fn validate_monvols(
        &self,
        surf_ids: &[usize],
        gas_ids: &[usize],
        injector_ids: &[usize],
        sensor_ids: &[usize],
        ctx: &RadExtraContext,
    ) -> Result<(), RadError> {
        for bag in &self.monvols {
            match bag {
                RadMonvol::Airbag1(typed) => {
                    if !surf_ids.contains(&typed.surf_external) {
                        return Err(RadError::Parse(format!(
                            "/MONVOL/AIRBAG1/{} references missing surface {}",
                            typed.id, typed.surf_external
                        )));
                    }
                    if !gas_ids.contains(&typed.mat_gas) {
                        return Err(RadError::Parse(format!(
                            "/MONVOL/AIRBAG1/{} references missing gas {}",
                            typed.id, typed.mat_gas
                        )));
                    }
                    for injector in &typed.injectors {
                        if !injector_ids.contains(&injector.inject_id) {
                            return Err(RadError::Parse(format!(
                                "/MONVOL/AIRBAG1/{} references missing injector {}",
                                typed.id, injector.inject_id
                            )));
                        }
                        if injector.sens_id > 0 && !sensor_ids.contains(&injector.sens_id) {
                            return Err(RadError::Parse(format!(
                                "/MONVOL/AIRBAG1/{} references missing sensor {}",
                                typed.id, injector.sens_id
                            )));
                        }
                        if (injector.ijet == 1) != injector.jetting.is_some() {
                            return Err(RadError::Parse(format!(
                                "/MONVOL/AIRBAG1/{} injector needs jetting exactly when Ijet is 1",
                                typed.id
                            )));
                        }
                        for node in injector.nodes {
                            if node > ctx.num_nodes {
                                return Err(RadError::Parse(format!(
                                    "/MONVOL/AIRBAG1/{} references missing node {node}",
                                    typed.id
                                )));
                            }
                        }
                    }
                    for vent in &typed.vents {
                        if !surf_ids.contains(&vent.surf) {
                            return Err(RadError::Parse(format!(
                                "/MONVOL/AIRBAG1/{} references missing vent surface {}",
                                typed.id, vent.surf
                            )));
                        }
                        if (vent.iform == 2) != vent.chemkin.is_some() {
                            return Err(RadError::Parse(format!(
                                "/MONVOL/AIRBAG1/{} vent needs a Chemkin line exactly when Iform is 2",
                                typed.id
                            )));
                        }
                        for fun in vent.fcts.iter().chain(vent.fcts_p.iter()) {
                            if *fun > ctx.num_functions {
                                return Err(RadError::Parse(format!(
                                    "/MONVOL/AIRBAG1/{} references missing function {fun}",
                                    typed.id
                                )));
                            }
                        }
                    }
                    for porous in &typed.porous {
                        if !surf_ids.contains(&porous.surf) {
                            return Err(RadError::Parse(format!(
                                "/MONVOL/AIRBAG1/{} references missing porous surface {}",
                                typed.id, porous.surf
                            )));
                        }
                        if (porous.iformps == 0) != porous.leakage.is_some() {
                            return Err(RadError::Parse(format!(
                                "/MONVOL/AIRBAG1/{} porous needs a leakage line exactly when Iformps is 0",
                                typed.id
                            )));
                        }
                        if (porous.iformps == 2) != porous.chemkin.is_some() {
                            return Err(RadError::Parse(format!(
                                "/MONVOL/AIRBAG1/{} porous needs a Chemkin line exactly when Iformps is 2",
                                typed.id
                            )));
                        }
                    }
                }
                RadMonvol::Commu(typed) => {
                    if !surf_ids.contains(&typed.surf_external) {
                        return Err(RadError::Parse(format!(
                            "/MONVOL/COMMU/{} references missing surface {}",
                            typed.id, typed.surf_external
                        )));
                    }
                    for jet in &typed.jets {
                        for fun in [jet.fct_mass, jet.fct_temp] {
                            if fun > ctx.num_functions {
                                return Err(RadError::Parse(format!(
                                    "/MONVOL/COMMU/{} references missing function {fun}",
                                    typed.id
                                )));
                            }
                        }
                        if jet.sens_id > 0 && !sensor_ids.contains(&jet.sens_id) {
                            return Err(RadError::Parse(format!(
                                "/MONVOL/COMMU/{} references missing sensor {}",
                                typed.id, jet.sens_id
                            )));
                        }
                        if (jet.ijet == 1) != jet.jetting.is_some() {
                            return Err(RadError::Parse(format!(
                                "/MONVOL/COMMU/{} jet needs jetting exactly when Ijet is 1",
                                typed.id
                            )));
                        }
                        for node in jet.nodes {
                            if node > ctx.num_nodes {
                                return Err(RadError::Parse(format!(
                                    "/MONVOL/COMMU/{} references missing node {node}",
                                    typed.id
                                )));
                            }
                        }
                    }
                    for vent in &typed.vents {
                        if vent.surf > 0 && !surf_ids.contains(&vent.surf) {
                            return Err(RadError::Parse(format!(
                                "/MONVOL/COMMU/{} references missing vent surface {}",
                                typed.id, vent.surf
                            )));
                        }
                        for fun in vent.fcts.iter().chain(vent.fcts_p.iter()) {
                            if *fun > ctx.num_functions {
                                return Err(RadError::Parse(format!(
                                    "/MONVOL/COMMU/{} references missing function {fun}",
                                    typed.id
                                )));
                            }
                        }
                    }
                    for link in &typed.bags {
                        if !self.monvols.iter().any(|bag| bag.id() == link.bag) {
                            return Err(RadError::Parse(format!(
                                "/MONVOL/COMMU/{} references missing volume {}",
                                typed.id, link.bag
                            )));
                        }
                        if link.surf_c > 0 && !surf_ids.contains(&link.surf_c) {
                            return Err(RadError::Parse(format!(
                                "/MONVOL/COMMU/{} references missing surface {}",
                                typed.id, link.surf_c
                            )));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Validate 1D support cards and extra parts.
    fn validate_1d(&self, ctx: &RadExtraContext) -> Result<(), RadError> {
        for part in &self.parts {
            if part.id <= ctx.num_parts {
                return Err(RadError::Parse(format!(
                    "extra /PART/{} collides with a base part",
                    part.id
                )));
            }
            if !self.has_prop(ctx, part.prop) {
                return Err(RadError::Parse(format!(
                    "extra /PART/{} references missing property {}",
                    part.id, part.prop
                )));
            }
            if part.mat > ctx.num_materials {
                return Err(RadError::Parse(format!(
                    "extra /PART/{} references missing material {}",
                    part.id, part.mat
                )));
            }
        }
        for prop in self
            .truss_props
            .iter()
            .map(|prop| prop.id)
            .chain(self.beam_props.iter().map(|prop| prop.id))
            .chain(self.spring_props.iter().map(|prop| prop.id))
        {
            if prop <= ctx.num_parts {
                return Err(RadError::Parse(format!(
                    "extra property {prop} collides with a base property"
                )));
            }
        }
        for truss in &self.truss {
            if !self.has_part(ctx, truss.part) {
                return Err(RadError::Parse(format!(
                    "/TRUSS/{} references missing part",
                    truss.part
                )));
            }
            for elem in &truss.elems {
                for node in elem {
                    if *node == 0 || *node > ctx.num_nodes {
                        return Err(RadError::Parse(format!(
                            "/TRUSS/{} references missing node {node}",
                            truss.part
                        )));
                    }
                }
            }
        }
        for beam in &self.beams {
            if !self.has_part(ctx, beam.part) {
                return Err(RadError::Parse(format!(
                    "/BEAM/{} references missing part",
                    beam.part
                )));
            }
            for elem in &beam.elems {
                for node in elem {
                    if *node == 0 || *node > ctx.num_nodes {
                        return Err(RadError::Parse(format!(
                            "/BEAM/{} references missing node {node}",
                            beam.part
                        )));
                    }
                }
            }
        }
        for spring in &self.springs {
            if !self.has_part(ctx, spring.part) {
                return Err(RadError::Parse(format!(
                    "/SPRING/{} references missing part",
                    spring.part
                )));
            }
            for elem in &spring.elems {
                for node in elem {
                    if *node == 0 || *node > ctx.num_nodes {
                        return Err(RadError::Parse(format!(
                            "/SPRING/{} references missing node {node}",
                            spring.part
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// Validate time-history requests against entity counts.
    fn validate_th(&self, ctx: &RadExtraContext) -> Result<(), RadError> {
        let num_truss: usize = self.truss.iter().map(|group| group.elems.len()).sum();
        let num_beams: usize = self.beams.iter().map(|group| group.elems.len()).sum();
        let num_springs: usize = self.springs.iter().map(|group| group.elems.len()).sum();
        for req in &self.th_elems {
            let what = format!("/TH/{}", req.family.keyword());
            check_th_vars(&req.vars, &what)?;
            if req.elems.is_empty() {
                return Err(RadError::Parse(format!(
                    "{what} needs at least one element"
                )));
            }
            // The starter errors (id 69) on missing TH elements, so
            // every request resolves against a real entity count.
            // `/QUAD` and `/SPHCEL` have no writers in this lane,
            // hence any request for them fails here by design.
            let count = match req.family {
                RadThElemFamily::Node => ctx.num_nodes,
                RadThElemFamily::Shel | RadThElemFamily::Shell => ctx.num_shells,
                RadThElemFamily::Sh3n => ctx.num_tris,
                RadThElemFamily::Bric | RadThElemFamily::Brick => ctx.num_bricks,
                RadThElemFamily::Quad | RadThElemFamily::Sphcel => 0,
                RadThElemFamily::Truss => num_truss,
                RadThElemFamily::Beam => num_beams,
                RadThElemFamily::Spring => num_springs,
            };
            for elem in &req.elems {
                if *elem == 0 || *elem > count {
                    return Err(RadError::Parse(format!(
                        "{what} references missing element {elem}"
                    )));
                }
            }
        }
        for req in &self.th_objs {
            let what = format!("/TH/{}", req.kind.keyword());
            check_th_vars(&req.vars, &what)?;
            if req.ids.is_empty() {
                return Err(RadError::Parse(format!("{what} needs at least one id")));
            }
            match req.kind {
                RadThObjKind::Part => {
                    for id in &req.ids {
                        if !self.has_part(ctx, *id) {
                            return Err(RadError::Parse(format!(
                                "{what} references missing part {id}"
                            )));
                        }
                    }
                }
                RadThObjKind::Inter => {
                    for id in &req.ids {
                        if !self.interfaces.iter().any(|inter| inter.id() == *id) {
                            return Err(RadError::Parse(format!(
                                "{what} references missing interface {id}"
                            )));
                        }
                    }
                }
                RadThObjKind::Monv | RadThObjKind::Monvol => {
                    for id in &req.ids {
                        if !self.monvols.iter().any(|bag| bag.id() == *id) {
                            return Err(RadError::Parse(format!(
                                "{what} references missing volume {id}"
                            )));
                        }
                    }
                }
                RadThObjKind::Subs => {
                    for id in &req.ids {
                        if !self.subsets.iter().any(|subset| subset.id == *id) {
                            return Err(RadError::Parse(format!(
                                "{what} references missing subset {id}"
                            )));
                        }
                    }
                }
                RadThObjKind::Rwall => {
                    for id in &req.ids {
                        if *id == 0 || *id > ctx.num_walls {
                            return Err(RadError::Parse(format!(
                                "{what} references missing wall {id}"
                            )));
                        }
                    }
                }
                // Sections, rigid bodies, and accelerometers belong
                // to other lanes; dangling ids only warn (id 257).
                RadThObjKind::Sectio | RadThObjKind::Rbody | RadThObjKind::Accel => {}
            }
        }
        Ok(())
    }
}

/// Count `/TH` groups emitted by [`write_starter`].
///
/// Mirrors the base writer exactly (one group per ten-variable
/// card): node output plus one wall group when walls exist.
pub fn starter_th_groups(case: &OpenCourantCase) -> usize {
    let mut groups = 0;
    if !case.th_nodes.is_empty() {
        groups += 1;
    }
    if !case.walls.is_empty() {
        groups += 1;
    }
    groups
}

/// Count `/TH` groups emitted by [`write_extended_starter`].
///
/// Mirrors the extended writer exactly: node, shell, brick, and
/// wall requests, one group per ten-variable card.
pub fn extended_starter_th_groups(case: &RadExtendedCase) -> usize {
    let mut groups = 0;
    if !case.th_node_vars.is_empty() && !case.base.th_nodes.is_empty() {
        groups += case.th_node_vars.chunks(10).len();
    }
    if !case.th_shell.vars.is_empty() && !case.th_shell.elems.is_empty() {
        groups += case.th_shell.vars.chunks(10).len();
    }
    if !case.th_brick.vars.is_empty() && !case.th_brick.elems.is_empty() {
        groups += case.th_brick.vars.chunks(10).len();
    }
    if !case.base.walls.is_empty() {
        groups += 1;
    }
    groups
}

/// Render the starter side of a [`RadExtra`] bundle.
///
/// Splice the result before the terminal `/END` of a base starter
/// deck; `first_th_group` continues the deck-global TH group
/// namespace (see [`starter_th_groups`]). When `extra.submodel`
/// is set, the whole block sits inside one `//SUBMODEL` section.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{RadDamping, RadExtra, write_starter_extra};
/// let mut extra = RadExtra::default();
/// extra.damps.push(RadDamping {
///     id: 1, title: "bulk".to_string(), alpha: 1.0, beta_s: 0.0,
///     grnd: 1, skew: 0, tstart_s: 0.0, tstop_s: 1.0e30,
/// });
/// let deck = write_starter_extra(&extra, 1);
/// assert!(deck.contains("/DAMP/1\n"));
/// ```
pub fn write_starter_extra(extra: &RadExtra, first_th_group: usize) -> String {
    let mut deck = String::new();
    if let Some(sub) = &extra.submodel {
        write_submodel_open(&mut deck, sub);
    }
    for surf in &extra.surfs {
        write_surf_parts(&mut deck, surf);
    }
    for grnod in &extra.grnods {
        write_grnod_surfs(&mut deck, grnod);
    }
    for line in &extra.lines {
        write_line_surfs(&mut deck, line);
    }
    for subset in &extra.subsets {
        write_subset(&mut deck, subset);
    }
    for inter in &extra.interfaces {
        write_interface(&mut deck, inter);
    }
    for gas in &extra.gases {
        write_gas_predef(&mut deck, gas);
    }
    for inject in &extra.injectors {
        write_inject2(&mut deck, inject);
    }
    for bag in &extra.monvols {
        write_monvol(&mut deck, bag);
    }
    for damp in &extra.damps {
        write_damping(&mut deck, damp);
    }
    for damp in &extra.full_damps {
        write_damping_full(&mut deck, damp);
    }
    for sensor in &extra.sensors {
        write_sensor(&mut deck, sensor);
    }
    for prop in &extra.truss_props {
        write_truss_prop(&mut deck, prop);
    }
    for prop in &extra.beam_props {
        write_beam_prop(&mut deck, prop);
    }
    for prop in &extra.spring_props {
        write_spring_prop(&mut deck, prop);
    }
    for part in &extra.parts {
        write_extra_part(&mut deck, part);
    }
    for truss in &extra.truss {
        write_truss(&mut deck, truss);
    }
    for beam in &extra.beams {
        write_beam(&mut deck, beam);
    }
    for spring in &extra.springs {
        write_spring(&mut deck, spring);
    }
    let mut group = first_th_group;
    for req in &extra.th_elems {
        write_th_elems(&mut deck, &mut group, req);
    }
    for req in &extra.th_objs {
        write_th_objects(&mut deck, &mut group, req);
    }
    if extra.submodel.is_some() {
        write_submodel_close(&mut deck);
    }
    deck
}

/// Render the engine side of a [`RadExtra`] bundle.
///
/// Appends `/STATE/DT`, `/STATE/*` flags, engine `/DAMP`
/// overrides, and `/TH/VERS/*` to an engine deck.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::{RadExtra, RadStateFlag, RadStateOutput, write_engine_extra};
/// let mut extra = RadExtra::default();
/// extra.state = Some(RadStateOutput { tstart_s: 0.0, tfreq_s: 0.1, all_parts: true, parts: Vec::new() });
/// extra.state_flags.push(RadStateFlag::NodeVel);
/// extra.th_vers = Some(41);
/// let deck = write_engine_extra(&extra);
/// assert!(deck.contains("/STATE/DT/ALL\n"));
/// assert!(deck.contains("/STATE/NODE/VEL\n"));
/// assert!(deck.contains("/TH/VERS/41\n"));
/// ```
pub fn write_engine_extra(extra: &RadExtra) -> String {
    let mut deck = String::new();
    if let Some(state) = &extra.state {
        write_state_dt(&mut deck, state);
    }
    for flag in &extra.state_flags {
        write_state_flag(&mut deck, *flag);
    }
    for damp in &extra.engine_damps {
        write_engine_damping(&mut deck, damp);
    }
    if let Some(vers) = extra.th_vers {
        deck.push_str(&format!("/TH/VERS/{vers}\n"));
    }
    deck
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brick_volume(mesh: &RadMesh, brick: &[usize; 8]) -> f64 {
        // Decompose the hex into five tets sharing the main diagonal.
        let p = |i: usize| mesh.nodes[i - 1];
        let tet = |a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]| {
            let (ab, ac, ad) = (
                [b[0] - a[0], b[1] - a[1], b[2] - a[2]],
                [c[0] - a[0], c[1] - a[1], c[2] - a[2]],
                [d[0] - a[0], d[1] - a[1], d[2] - a[2]],
            );
            ((ab[1] * ac[2] - ab[2] * ac[1]) * ad[0]
                + (ab[2] * ac[0] - ab[0] * ac[2]) * ad[1]
                + (ab[0] * ac[1] - ab[1] * ac[0]) * ad[2])
                / 6.0
        };
        let n: Vec<[f64; 3]> = brick.iter().map(|id| p(*id)).collect();
        tet(n[0], n[1], n[3], n[4]).abs()
            + tet(n[1], n[2], n[3], n[6]).abs()
            + tet(n[1], n[3], n[4], n[6]).abs()
            + tet(n[3], n[4], n[6], n[7]).abs()
            + tet(n[1], n[4], n[5], n[6]).abs()
    }

    #[test]
    fn cantilever_mesh_is_valid_and_indexed() {
        let (mesh, root, tip, tip_node) = cantilever_brick_mesh(4, 1, 1, 1.0, 0.1, 0.1);
        assert_eq!(mesh.num_nodes(), 5 * 2 * 2);
        assert_eq!(mesh.num_bricks(), 4);
        mesh.validate().unwrap();
        assert_eq!(root.len(), 4);
        assert_eq!(tip.len(), 4);
        assert!(tip.contains(&tip_node));
        for brick in &mesh.bricks {
            assert!(brick_volume(&mesh, brick) > 0.0);
        }
    }

    #[test]
    fn sphere_mesh_is_all_positive_volume() {
        let mesh = sphere_ball_mesh(8, 0.05);
        mesh.validate().unwrap();
        assert!(!mesh.bricks.is_empty());
        for brick in &mesh.bricks {
            assert!(
                brick_volume(&mesh, brick) > 1e-15,
                "inverted brick: {brick:?}"
            );
        }
        // Surface nodes sit on the sphere within projection tolerance.
        let mut surface = 0;
        for node in &mesh.nodes {
            let r = (node[0] * node[0] + node[1] * node[1] + (node[2] - 0.05).powi(2)).sqrt();
            assert!(r <= 0.05 + 1e-9, "node escapes the ball: {node:?}");
            if r > 0.049 {
                surface += 1;
            }
        }
        assert!(surface > 0, "projection must place nodes on the surface");
    }

    #[test]
    fn starter_and_engine_decks_carry_required_cards() {
        for case in [cantilever_case(), hertz_sphere_case()] {
            case.mesh.validate().unwrap();
            let starter = write_starter(&case);
            for card in [
                "#RADIOSS STARTER",
                "/BEGIN",
                "/UNIT/1",
                "/NODE",
                "/BRICK",
                "/MAT/LAW1/1/1",
                "/PROP/SOLID/1/1",
                "/PART/1",
                "/END",
            ] {
                assert!(starter.contains(card), "{} lacks {card}", case.name);
            }
            let engine = write_engine(&case);
            for card in [
                "#RADIOSS ENGINE",
                "/RUN/",
                "/TFILE",
                "/TH/TITLE",
                "/ANIM/DT",
            ] {
                assert!(engine.contains(card), "{} lacks {card}", case.name);
            }
            // Proven against the engine: no /STOP or /END terminator cards.
            assert!(!engine.contains("/STOP"), "{} carries /STOP", case.name);
            assert!(!engine.contains("/END"), "{} carries /END", case.name);
        }
        let starter = write_starter(&cantilever_case());
        assert!(starter.contains("/BCS/1"));
        assert!(starter.contains("/CLOAD/1/1"));
        let starter = write_starter(&hertz_sphere_case());
        assert!(starter.contains("/RWALL/PLANE/1/1"));
        assert!(starter.contains("/RWALL/PLANE/2/1"));
    }

    #[test]
    fn rwall_plane_card_order_matches_reference() {
        let starter = write_starter(&hertz_sphere_case());
        let lines: Vec<&str> = starter.lines().collect();
        for wall_id in [1, 2] {
            let header = format!("/RWALL/PLANE/{wall_id}/1");
            let at = lines
                .iter()
                .position(|line| *line == header)
                .unwrap_or_else(|| panic!("{starter} lacks {header}"));
            // Title, node_ID/Slide/grnds, Dsearch/fric, then M (fixed) or
            // Mass/V (moving), then M1. Six lines per wall block.
            assert!(lines.len() > at + 5, "truncated {header}");
            let node_slide: Vec<&str> = lines[at + 2].split_whitespace().collect();
            let mass_or_m: Vec<&str> = lines[at + 4].split_whitespace().collect();
            if wall_id == 1 {
                assert_eq!(node_slide[0], "0", "fixed wall needs node_ID 0");
                assert_eq!(mass_or_m.len(), 3, "fixed wall needs an M line");
            } else {
                assert_ne!(node_slide[0], "0", "moving wall needs a driver node");
                assert_eq!(mass_or_m.len(), 4, "moving wall needs Mass/VX0/VY0/VZ0");
                assert!(
                    mass_or_m[0].starts_with("0."),
                    "moving wall needs Mass 0 (imposed velocity)"
                );
            }
        }
    }

    #[test]
    fn wall_channels_normalize_past_wall_titles() {
        assert_eq!(
            normalize_channel(
                "\"time history walls                       1 fixed bottom plane                        FNZ\""
            ),
            "FNZ/RWALL/1"
        );
        assert_eq!(
            normalize_channel(
                "\"time history walls                       2 moving top plane                          FNX\""
            ),
            "FNX/RWALL/2"
        );
        assert_eq!(
            normalize_channel(
                "\"time history nodes                       247                                           DZ\""
            ),
            "DZ/247"
        );
    }

    #[test]
    fn time_history_csv_parses_and_interpolates() {
        let dir = std::env::temp_dir().join("sonium_qa_rad_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sampleT01.csv");
        std::fs::write(
            &path,
            "Time,DZ/5,FNZ/RWALL/1\n0.0,0.0,0.0\n1.0,0.002,40.0\n2.0,0.004,80.0\n",
        )
        .unwrap();
        let history = parse_t01(&path).unwrap();
        assert_eq!(history.times, vec![0.0, 1.0, 2.0]);
        assert!((history.final_value("DZ/5").unwrap() - 0.004).abs() < 1e-15);
        assert!((history.value_at("DZ/5", 0.5).unwrap() - 0.001).abs() < 1e-15);
        assert!(history.channel("missing").is_err());
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn impulse_channels_differentiate_to_force() {
        let times = vec![0.0, 1.0, 2.0, 3.0];
        let impulse = vec![0.0, 10.0, 40.0, 90.0];
        let force = impulse_to_force(&times, &impulse);
        assert!((force[1] - 20.0).abs() < 1e-12);
        assert!((force[2] - 40.0).abs() < 1e-12);
    }

    #[test]
    fn probe_comparator_accepts_and_rejects() {
        assert!(assert_probe_close(1.0, 1.0 + 1e-9, 1e-6, 1e-6, "close").is_ok());
        assert!(assert_probe_close(2.0, 1.0, 1e-6, 1e-6, "far").is_err());
        assert!(assert_probe_close(f64::NAN, 1.0, 1e-6, 1e-6, "nan").is_err());
    }

    fn quad_area(nodes: &[[f64; 3]], quad: &[usize; 4]) -> f64 {
        let (a, b, c, d) = (
            nodes[quad[0] - 1],
            nodes[quad[1] - 1],
            nodes[quad[2] - 1],
            nodes[quad[3] - 1],
        );
        let cross = |u: [f64; 3], v: [f64; 3]| {
            [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ]
        };
        let sub = |u: [f64; 3], v: [f64; 3]| [u[0] - v[0], u[1] - v[1], u[2] - v[2]];
        let norm = |u: [f64; 3]| (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        0.5 * (norm(cross(sub(b, a), sub(c, a))) + norm(cross(sub(c, a), sub(d, a))))
    }

    #[test]
    fn scordelis_mesh_has_outward_quads_and_probes() {
        let mesh = scordelis_lo_shell_mesh(8, 8);
        assert_eq!(mesh.nodes.len(), 81);
        assert_eq!(mesh.quads.len(), 64);
        // Free-edge probe at theta = +40 deg, z = 25.
        let probe = mesh.nodes[mesh.probe_node - 1];
        assert!((probe[0] - 25.0 * 40.0_f64.to_radians().sin()).abs() < 1e-9);
        assert!((probe[2] - 25.0).abs() < 1e-12);
        // Every quad has a positive area and an outward normal.
        for quad in &mesh.quads {
            assert!(quad_area(&mesh.nodes, quad) > 1.0);
            let (a, b, d) = (
                mesh.nodes[quad[0] - 1],
                mesh.nodes[quad[1] - 1],
                mesh.nodes[quad[3] - 1],
            );
            let edge1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let edge2 = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
            let normal = [
                edge1[1] * edge2[2] - edge1[2] * edge2[1],
                edge1[2] * edge2[0] - edge1[0] * edge2[2],
                edge1[0] * edge2[1] - edge1[1] * edge2[0],
            ];
            let radial = [a[0], a[1], 0.0];
            let dot = normal[0] * radial[0] + normal[1] * radial[1];
            assert!(dot > 0.0, "quad {quad:?} normal points inward");
        }
        // Probe shell straddles the apex midspan.
        assert!((1..=mesh.quads.len()).contains(&mesh.probe_shell));
        let quad = mesh.quads[mesh.probe_shell - 1];
        let cx: f64 = quad.iter().map(|n| mesh.nodes[n - 1][0]).sum::<f64>() / 4.0;
        let cz: f64 = quad.iter().map(|n| mesh.nodes[n - 1][2]).sum::<f64>() / 4.0;
        assert!(cx.abs() < 25.0 * 10.0_f64.to_radians());
        assert!((cz - 25.0).abs() < 50.0 / 8.0);
    }

    #[test]
    fn clamped_plate_mesh_tiles_the_disk() {
        let mesh = clamped_plate_shell_mesh(1.0, 8, 32);
        assert_eq!(mesh.nodes.len(), 1 + 8 * 32);
        assert_eq!(mesh.quads.len(), 7 * 32);
        assert_eq!(mesh.tris.len(), 32);
        let total: f64 = mesh.nodal_area_m2.iter().sum();
        assert!(
            (total - std::f64::consts::PI).abs() < 1e-9,
            "tributary areas sum to {total}"
        );
        for quad in &mesh.quads {
            assert!(quad_area(&mesh.nodes, quad) > 1e-8);
        }
        assert_eq!(mesh.center_node, 1);
        assert!((mesh.probe_radius_m - 0.1875).abs() < 1e-12);
    }

    #[test]
    fn shell_cases_validate_and_match_closed_forms() {
        for case in [shell_scordelis_lo_case(), shell_clamped_plate_case()] {
            case.validate().unwrap();
            assert!(!case.shells.is_empty());
            assert!(case.shell_prop.is_some());
        }
        let plate = shell_clamped_plate_case();
        let center = plate.base.probes[0].expected;
        let rigidity = 200.0e9 * 0.02_f64.powi(3) / (12.0 * (1.0 - 0.3_f64.powi(2)));
        let exact = -1000.0 / (64.0 * rigidity);
        assert!((center - exact).abs() < 1e-15);
        assert!((center + 1.066_406e-4).abs() < 1e-10);
        // Ring CLOADs integrate to the pressure resultant.
        let mut resultant = 0.0;
        for load in &plate.base.cloads {
            let set = &plate.base.node_sets[load.set];
            resultant += load.force_scale_n * set.nodes.len() as f64;
        }
        assert!(
            (resultant + 1000.0 * std::f64::consts::PI).abs() < 1e-6,
            "pressure resultant is {resultant}"
        );
        let roof = shell_scordelis_lo_case();
        // This full-load nonlinear regression guards engine drift. The
        // published-.3024 linear target requires a separate accuracy leg.
        // Regression agreement alone is not native/external validation.
        assert!((roof.base.probes[0].expected + 0.2574).abs() < 1e-15);
        assert!((roof.base.probes[0].rel_tol - 0.02).abs() < 1e-12);
        assert!(roof.gravity.is_some());
    }

    #[test]
    fn extended_starter_carries_guide_blocks() {
        let roof = shell_scordelis_lo_case();
        let starter = write_extended_starter(&roof);
        for card in [
            "#RADIOSS STARTER",
            "/NODE",
            "/SHELL/1",
            "/GRNOD/NODE/1/1",
            "/MAT/LAW1/1/1",
            "/PROP/SHELL/1/1",
            "/PART/1",
            "/FUNCT/1",
            "/BCS/1",
            "/GRAV/1/1",
            "/TH/NODE/1",
            "/TH/SHEL/2",
            "/END",
        ] {
            assert!(starter.contains(card), "roof starter lacks {card}");
        }
        assert!(!starter.contains("/BRICK"));
        assert!(!starter.contains("/TH/BRIC"));
        let plate = shell_clamped_plate_case();
        let starter = write_extended_starter(&plate);
        for card in ["/SHELL/1", "/SH3N/1", "/CLOAD/1/1", "/TH/SHEL/2"] {
            assert!(starter.contains(card), "plate starter lacks {card}");
        }
        // TH variable cards carry the requested names.
        assert!(starter.contains("M1"));
        assert!(starter.contains("THIC"));
        assert!(starter.contains("DX"));
    }

    #[test]
    fn plastic_materials_render_law_cards() {
        let base = cantilever_case();
        let mut law2 = RadExtendedCase::wrap_elastic(base.clone());
        law2.material = RadMaterial::Plastic(RadPlasticMaterial::JohnsonCook(
            RadJohnsonCook::perfectly_plastic(200.0e9, 0.3, 7800.0, 250.0e6),
        ));
        law2.th_brick = RadThRequest::brick_stress(vec![1]);
        law2.validate().unwrap();
        let starter = write_extended_starter(&law2);
        for card in ["/MAT/LAW2/1/1", "/BRICK/1", "/TH/BRIC/2", "/PART/1"] {
            assert!(starter.contains(card), "LAW2 starter lacks {card}");
        }
        assert!(starter.contains("SX"));
        assert!(!starter.contains("/PROP/SHELL"));

        let mut law36 = RadExtendedCase::wrap_elastic(base);
        law36.material =
            RadMaterial::Plastic(perfectly_plastic_steel_law36(200.0e9, 0.3, 7800.0, 250.0e6));
        law36.validate().unwrap();
        let starter = write_extended_starter(&law36);
        assert!(starter.contains("/MAT/LAW36/1/1"));
        // The hardening curve follows the user functions.
        assert!(starter.contains("/FUNCT/2"));
        assert!(starter.contains("LAW36 yield stress vs plastic strain"));
    }

    #[test]
    fn extended_case_validation_catches_gaps() {
        let mut case = shell_scordelis_lo_case();
        case.shell_prop = None;
        assert!(case.validate().is_err());
        let mut case = shell_scordelis_lo_case();
        case.th_shell.elems = vec![10_000];
        assert!(case.validate().is_err());
        let mut case = shell_scordelis_lo_case();
        case.gravity.as_mut().unwrap().direction = 'Q';
        assert!(case.validate().is_err());
        let cantilever = RadExtendedCase::wrap_elastic(cantilever_case());
        cantilever.validate().unwrap();
    }

    #[test]
    fn element_channels_are_group_qualified() {
        assert_eq!(shel_channel("M1", 12), "M1/SHEL/12");
        assert_eq!(bric_channel("SX", 4), "SX/BRIC/4");
    }

    #[test]
    fn channel_headers_normalize_to_probe_form() {
        // Observed `th_to_csv` spellings (engine prove-out, Oct 2026).
        assert_eq!(
            normalize_channel(
                "\"time history nodes                       95                                           DX         \""
            ),
            "DX/95"
        );
        assert_eq!(
            normalize_channel(
                "\"time history shells                       1                                           THIC         \""
            ),
            "THIC/SHEL/1"
        );
        assert_eq!(
            normalize_channel(
                "\"time history bricks                       25                                           SX         \""
            ),
            "SX/BRIC/25"
        );
        // Energy channels pass through with quotes stripped.
        assert_eq!(normalize_channel("\"INTERNAL ENERGY\""), "INTERNAL ENERGY");
        assert_eq!(normalize_channel("\"TIME STEP\""), "TIME STEP");
        assert_eq!(
            normalize_channel("\"PERCENTAGE ADDED MASS\""),
            "PERCENTAGE ADDED MASS"
        );
    }

    fn fake_exec_dir(tag: &str, files: &[&str]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sonium_qa_rad_pair_{tag}"));
        std::fs::create_dir_all(&dir).unwrap();
        for file in files {
            std::fs::write(dir.join(file), "fake").unwrap();
        }
        dir
    }

    #[test]
    fn engine_pair_prefers_double_precision_smp() {
        let dir = fake_exec_dir(
            "rank",
            &[
                "engine_linux64_gf_ompi_sp",
                "engine_linux64_gf_sp",
                "engine_linux64_gf",
                "starter_linux64_gf_sp",
                "starter_linux64_gf",
            ],
        );
        let (starter, engine) = resolve_engine_pair(&dir).unwrap();
        assert_eq!(starter.file_name().unwrap(), "starter_linux64_gf");
        assert_eq!(engine.file_name().unwrap(), "engine_linux64_gf");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn engine_pair_matches_file_suffix() {
        let dir = fake_exec_dir("suffix", &["starter_linux64_gf_sp", "engine_linux64_gf_sp"]);
        let (starter, engine) = resolve_engine_pair(&dir.join("starter_linux64_gf_sp")).unwrap();
        assert_eq!(engine.file_name().unwrap(), "engine_linux64_gf_sp");
        assert_eq!(starter.file_name().unwrap(), "starter_linux64_gf_sp");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn engine_pair_rejects_empty_directory() {
        let dir = fake_exec_dir("empty", &["unrelated.txt"]);
        assert!(resolve_engine_pair(&dir).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn engine_flavor_sees_install_path() {
        assert_eq!(
            engine_flavor(Path::new("/opt/OpenCourant/exec/engine_linux64_gf")),
            "OpenCourant"
        );
        assert_eq!(
            engine_flavor(Path::new("/opt/radioss/exec/engine_linux64_gf")),
            "OpenCourant"
        );
    }

    #[test]
    fn probes_payload_check_ignores_engine_stamp() {
        let stamped = probes_json(
            &cantilever_case(),
            "OpenRadioss linux64_gf CommitID 33e68517",
        );
        let plain = probes_json(&cantilever_case(), "none (prove-out deferred)");
        assert_probes_payload_eq(&stamped, &plain, "test");
    }

    #[test]
    #[should_panic(expected = "drifted")]
    fn probes_payload_check_catches_value_drift() {
        let mut case = cantilever_case();
        let stamped = probes_json(&case, "OpenRadioss linux64_gf CommitID 33e68517");
        case.probes[0].expected += 1.0;
        let drifted = probes_json(&case, "none (prove-out deferred)");
        assert_probes_payload_eq(&stamped, &drifted, "drifted");
    }

    // Phase-4 lane 4B tests: writer completion cards.

    fn block_lines(deck: &str, header: &str) -> Vec<String> {
        let lines: Vec<&str> = deck.lines().collect();
        let at = lines
            .iter()
            .position(|line| *line == header)
            .unwrap_or_else(|| panic!("deck lacks {header}"));
        lines[at..].iter().map(ToString::to_string).collect()
    }

    #[test]
    fn extra_groups_emit_ten_ids_per_line() {
        let mut deck = String::new();
        write_surf_parts(
            &mut deck,
            &RadSurfParts {
                id: 4,
                title: "skin".to_string(),
                parts: (1..=11).collect(),
            },
        );
        let lines = block_lines(&deck, "/SURF/PART/4");
        assert_eq!(lines[1], "skin");
        assert_eq!(lines[2].len(), 100);
        assert_eq!(lines[3].trim(), "11");
        let mut deck = String::new();
        write_grnod_surfs(
            &mut deck,
            &RadGrnodSurfs {
                id: 32,
                title: "nodes".to_string(),
                surfs: vec![4],
            },
        );
        assert!(deck.contains("/GRNOD/SURF/32\nnodes\n"));
        let mut deck = String::new();
        write_line_surfs(
            &mut deck,
            &RadLineSurfs {
                id: 33,
                title: "edges".to_string(),
                surfs: vec![4],
            },
        );
        assert!(deck.contains("/LINE/SURF/33\nedges\n"));
        let mut deck = String::new();
        write_subset(
            &mut deck,
            &RadSubset {
                id: 1,
                title: "all".to_string(),
            },
        );
        assert_eq!(deck, "/SUBSET/1\nall\n");
    }

    #[test]
    fn inter_type2_line_is_100_columns() {
        let mut deck = String::new();
        write_inter_type2(&mut deck, &RadInterType2::basic(1, "tie", 4, 5));
        let lines = block_lines(&deck, "/INTER/TYPE2/1");
        assert_eq!(lines.len(), 3);
        let data = &lines[2];
        assert_eq!(data.len(), 100);
        assert_eq!(&data[0..10], "         4");
        assert_eq!(&data[10..20], "         5");
        // `Dsearch` occupies the trailing F20 field (cols 81-100).
        assert_eq!(&data[80..100].trim().parse::<f64>().unwrap(), &0.0);
    }

    #[test]
    fn inter_type7_matches_reader_layout() {
        let mut deck = String::new();
        let mut inter = RadInterType7::basic(1, "contact", 2, 3);
        inter.ibc = [true, false, true];
        inter.itied = 1;
        inter.fric_id = 7;
        write_inter_type7(&mut deck, &inter);
        let lines = block_lines(&deck, "/INTER/TYPE7/1");
        assert_eq!(lines.len(), 8);
        // Line 1: ten ids, sixth (legacy Multimp) blank.
        assert_eq!(lines[2].len(), 100);
        assert_eq!(&lines[2][50..60], "          ");
        assert_eq!(&lines[2][60..70], "         0");
        // Line 2: Itied in field 9 (cols 81-90).
        assert_eq!(lines[3].len(), 90);
        assert_eq!(&lines[3][80..90], "         1");
        // Line 3: stiffness + removal flags.
        assert_eq!(lines[4].len(), 100);
        // Line 4: penalty window carries Stfac 0.2.
        assert!((lines[5][0..20].trim().parse::<f64>().unwrap() - 0.2).abs() < 1e-15);
        // Line 5: IBC digits at columns 8-10.
        assert_eq!(&lines[6][0..10], "       101");
        assert_eq!(lines[6].len(), 100);
        // Line 6: friction tail with fric_ID in field 10.
        assert_eq!(lines[7].len(), 100);
        assert_eq!(&lines[7][90..100], "         7");
    }

    #[test]
    fn inter_type11_fric_id_in_field_10() {
        let mut deck = String::new();
        let mut inter = RadInterType11::basic(4, "edges", 33, 33);
        inter.fric_id = 9;
        write_inter_type11(&mut deck, &inter);
        let lines = block_lines(&deck, "/INTER/TYPE11/4");
        assert_eq!(lines.len(), 7);
        assert_eq!(lines[2].len(), 80);
        assert_eq!(&lines[2][50..60], "          ");
        let last = &lines[6];
        assert_eq!(last.len(), 100);
        assert_eq!(&last[0..90], " ".repeat(90));
        assert_eq!(&last[90..100], "         9");
    }

    #[test]
    fn monvol_airbag1_matches_cfg_layout() {
        let mut deck = String::new();
        write_monvol_airbag1(
            &mut deck,
            &RadMonvolAirbag1 {
                id: 1,
                title: "bag".to_string(),
                surf_external: 4,
                hconv: 0.0,
                ascale: [0.0; 5],
                mat_gas: 2,
                gas_mu: 0.1,
                pext_pa: 101_325.0,
                t0_k: 293.0,
                iequi: 0,
                ittf: 0,
                injectors: vec![RadAirbagInjector {
                    inject_id: 10,
                    sens_id: 1,
                    ijet: 0,
                    nodes: [0; 3],
                    jetting: None,
                }],
                vents: Vec::new(),
                porous: vec![RadAirbagPorous {
                    surf: 4,
                    iformps: 1,
                    iblockage: 0,
                    title: "fabric".to_string(),
                    tstart_s: 0.0,
                    tstop_s: 1.0e30,
                    dpdef: 0.0,
                    dtpdef_s: 0.0,
                    idtpdef: 0,
                    leakage: None,
                    chemkin: None,
                }],
            },
        );
        let lines = block_lines(&deck, "/MONVOL/AIRBAG1/1");
        // surf line: I10 + blank + F20.
        assert_eq!(lines[2].len(), 40);
        // mat line: mat blank Mu Pext T0 Iequi Ittf.
        assert_eq!(lines[4].len(), 100);
        assert_eq!(&lines[4][0..10], "         2");
        assert!((lines[4][20..40].trim().parse::<f64>().unwrap() - 0.1).abs() < 1e-15);
        // Njet, injector, Nvent/Nps, porous pair.
        assert_eq!(lines[5].trim(), "1");
        assert_eq!(lines[6].len(), 60);
        assert_eq!(lines[7].len(), 20);
        assert_eq!(lines[8].len(), 100);
        assert_eq!(&lines[8][80..100], "fabric              ");
        assert_eq!(lines[9].len(), 100);
    }

    #[test]
    fn monvol_commu_mirrors_jetv41b() {
        let mut deck = String::new();
        write_monvol_commu(
            &mut deck,
            &RadMonvolCommu {
                id: 1,
                title: "chamber".to_string(),
                surf_external: 3,
                ascale: [0.0; 5],
                gas_mu: 0.1,
                pext_pa: 101.35,
                t0_k: 0.0,
                iequi: 0,
                ittf: 0,
                gamma_i: 1.667,
                cp_i: [520.275, 0.0, 0.0],
                jets: vec![RadCommuJet {
                    gamma: 1.667,
                    cp: [520.275, 0.0, 0.0],
                    fct_mass: 1,
                    iflow: 0,
                    fscale_mass: 1.0,
                    fct_temp: 2,
                    fscale_temp: 1.0,
                    sens_id: 0,
                    ijet: 1,
                    nodes: [4, 5, 0],
                    jetting: Some(RadJetting {
                        fct_pt: 6,
                        fct_theta: 7,
                        fct_delta: 8,
                        fscale_pt: 0.0,
                        fscale_ptheta: 0.0,
                        fscale_pdelta: 0.0,
                    }),
                }],
                vents: vec![RadCommuVent {
                    surf: 0,
                    avent_m2: 840.0,
                    bvent: 0.0,
                    tstop_s: 0.0,
                    tvent_s: 1.0e30,
                    dpdef: 0.0,
                    dtpdef_s: 0.0,
                    fct_v: 0,
                    fscale_v: 0.0,
                    idtpdef: 0,
                    fcts: [0; 3],
                    fscales: [0.0; 3],
                    fcts_p: [0; 3],
                    fscales_p: [0.0; 3],
                }],
                bags: vec![RadCommuBag {
                    bag: 2,
                    surf_c: 0,
                    dpcdef_pa: 0.0,
                    acom_m2: 3958.5,
                    tcom_s: 1.0e30,
                    dtpcdef_s: 0.0,
                }],
            },
        );
        let lines = block_lines(&deck, "/MONVOL/COMMU/1");
        // Mu line: blank mat field, then Mu/Pext/T0/Iequi/Ittf.
        assert_eq!(lines[4].len(), 100);
        assert_eq!(&lines[4][0..20], " ".repeat(20));
        assert!((lines[4][20..40].trim().parse::<f64>().unwrap() - 0.1).abs() < 1e-15);
        // Jetting line: three ids, one blank, three scales.
        assert_eq!(lines[10].len(), 100);
        assert_eq!(&lines[10][0..10], "         6");
        assert_eq!(&lines[10][30..40], "          ");
        // Vent function rows share the jetting shape.
        assert_eq!(lines[lines.len() - 5].len(), 100);
        // Bag link: two ids plus four floats.
        assert_eq!(lines[lines.len() - 1].len(), 100);
        let bag = &lines[lines.len() - 1];
        assert!((bag[40..60].trim().parse::<f64>().unwrap() - 3958.5).abs() < 1e-9);
    }

    #[test]
    fn damping_cards_match_guide_shapes() {
        let mut deck = String::new();
        write_damping(
            &mut deck,
            &RadDamping {
                id: 1,
                title: "bulk".to_string(),
                alpha: 1.0,
                beta_s: 0.0,
                grnd: 2,
                skew: 0,
                tstart_s: 0.0,
                tstop_s: 1.0e30,
            },
        );
        let lines = block_lines(&deck, "/DAMP/1");
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[2].len(), 100);
        let mut deck = String::new();
        write_damping_full(
            &mut deck,
            &RadDampingFull {
                id: 2,
                title: "axes".to_string(),
                alpha: [1.0; 6],
                beta_s: [0.0; 6],
                grnd: 3,
                skew: 0,
                tstart_s: 0.0,
                tstop_s: 1.0e30,
            },
        );
        let lines = block_lines(&deck, "/DAMP/2");
        assert_eq!(lines.len(), 8);
        assert_eq!(lines[2].len(), 100);
        for line in &lines[3..] {
            assert_eq!(line.len(), 40);
        }
    }

    #[test]
    fn sensors_emit_all_guide_headers() {
        let sensors = vec![
            RadSensor::Time {
                id: 1,
                title: "fire".to_string(),
                tdelay_s: 0.5,
            },
            RadSensor::Sens {
                id: 3,
                title: "follow".to_string(),
                tdelay_s: 0.5,
                sens_on: 1,
                sens_off: 0,
            },
            RadSensor::Dist {
                id: 6,
                title: "gap".to_string(),
                tdelay_s: 0.0,
                node_a: 1,
                node_b: 2,
                dmin_m: -1.0e30,
                dmax_m: 1.0e30,
            },
            RadSensor::AndOr {
                id: 7,
                title: "both".to_string(),
                tdelay_s: 0.0,
                sens_a: 1,
                sens_b: 3,
                is_and: true,
            },
            RadSensor::AndOr {
                id: 8,
                title: "either".to_string(),
                tdelay_s: 0.0,
                sens_a: 1,
                sens_b: 3,
                is_and: false,
            },
        ];
        let mut deck = String::new();
        for sensor in &sensors {
            write_sensor(&mut deck, sensor);
        }
        for header in [
            "/SENSOR/TIME/1",
            "/SENSOR/SENS/3",
            "/SENSOR/DIST/6",
            "/SENSOR/AND/7",
            "/SENSOR/OR/8",
        ] {
            assert!(deck.contains(&format!("{header}\n")), "deck lacks {header}");
        }
        let dist = block_lines(&deck, "/SENSOR/DIST/6");
        assert_eq!(dist[3].len(), 60);
    }

    #[test]
    fn submodel_wraps_with_offsets() {
        let mut deck = String::new();
        let mut sub = RadSubmodel::plain(1, "extra");
        sub.off_nod = 100;
        write_submodel_open(&mut deck, &sub);
        deck.push_str("/SENSOR/TIME/1\n");
        write_submodel_close(&mut deck);
        let lines: Vec<&str> = deck.lines().collect();
        assert_eq!(lines[0], "//SUBMODEL/1");
        assert_eq!(lines[2].len(), 70);
        assert!(lines[2].contains("100"));
        assert_eq!(lines[4], "//ENDSUB");
    }

    #[test]
    fn th_completion_covers_every_family() {
        let elem_families = [
            RadThElemFamily::Node,
            RadThElemFamily::Shel,
            RadThElemFamily::Sh3n,
            RadThElemFamily::Shell,
            RadThElemFamily::Bric,
            RadThElemFamily::Brick,
            RadThElemFamily::Quad,
            RadThElemFamily::Truss,
            RadThElemFamily::Beam,
            RadThElemFamily::Spring,
            RadThElemFamily::Sphcel,
        ];
        let mut deck = String::new();
        let mut group = 0;
        for family in elem_families {
            write_th_elems(
                &mut deck,
                &mut group,
                &RadThElemRequest {
                    family,
                    title: "probe".to_string(),
                    vars: vec!["DEF".to_string()],
                    elems: vec![1],
                },
            );
        }
        assert_eq!(group, elem_families.len());
        for family in elem_families {
            let header = format!("/TH/{}", family.keyword());
            assert!(deck.contains(&header), "deck lacks {header}");
        }
        // Skew families carry a zero skew column; name families use
        // bare ids (a zero there trips starter warning 100214).
        let node = block_lines(&deck, "/TH/NODE/1");
        assert_eq!(node[3].len(), 20);
        let spring = block_lines(&deck, "/TH/SPRING/10");
        assert_eq!(spring[3].len(), 10);
        let obj_kinds = [
            RadThObjKind::Part,
            RadThObjKind::Inter,
            RadThObjKind::Monv,
            RadThObjKind::Monvol,
            RadThObjKind::Subs,
            RadThObjKind::Accel,
            RadThObjKind::Rbody,
            RadThObjKind::Sectio,
            RadThObjKind::Rwall,
        ];
        for kind in obj_kinds {
            write_th_objects(
                &mut deck,
                &mut group,
                &RadThObjRequest {
                    kind,
                    title: "probe".to_string(),
                    vars: vec!["DEF".to_string()],
                    ids: (1..=11).collect(),
                },
            );
        }
        for kind in obj_kinds {
            let header = format!("/TH/{}", kind.keyword());
            assert!(deck.contains(&header), "deck lacks {header}");
        }
        // Eleven object ids wrap ten plus one.
        let part = block_lines(&deck, "/TH/PART/12");
        assert_eq!(part[3].len(), 100);
        assert_eq!(part[4].trim(), "11");
        // Eleven variables open a second group for the same request.
        let before = group;
        write_th_elems(
            &mut deck,
            &mut group,
            &RadThElemRequest {
                family: RadThElemFamily::Node,
                title: "wide".to_string(),
                vars: (0..11).map(|n| format!("V{n}")).collect(),
                elems: vec![1],
            },
        );
        assert_eq!(group, before + 2);
    }

    #[test]
    fn spring_prop_has_26_data_lines() {
        let mut deck = String::new();
        write_spring_prop(
            &mut deck,
            &RadSpringProp {
                id: 5,
                title: "spring".to_string(),
                mass_kg: 1.0e-3,
                inertia: 1.0e-3,
                k_trans: 10000.0,
                k_rot: 100000.0,
            },
        );
        let lines = block_lines(&deck, "/PROP/TYPE13/5");
        assert_eq!(lines.len(), 28);
        assert!((lines[3][0..20].trim().parse::<f64>().unwrap() - 10000.0).abs() < 1e-9);
        assert!((lines[12][0..20].trim().parse::<f64>().unwrap() - 100000.0).abs() < 1e-9);
        // Function lines carry a blank field before the limits.
        assert_eq!(lines[4].len(), 100);
        assert_eq!(&lines[4][50..60], "          ");
        assert_eq!(&lines[4][60..80].trim().parse::<f64>().unwrap(), &-0.01);
    }

    #[test]
    fn th_group_counters_mirror_writers() {
        for case in [cantilever_case(), hertz_sphere_case()] {
            let starter = write_starter(&case);
            let groups = starter
                .lines()
                .filter(|line| line.starts_with("/TH/"))
                .count();
            assert_eq!(starter_th_groups(&case), groups, "{}", case.name);
        }
        for case in [shell_scordelis_lo_case(), shell_clamped_plate_case()] {
            let starter = write_extended_starter(&case);
            let groups = starter
                .lines()
                .filter(|line| line.starts_with("/TH/"))
                .count();
            assert_eq!(
                extended_starter_th_groups(&case),
                groups,
                "{}",
                case.base.name
            );
        }
        let mut law2 = RadExtendedCase::wrap_elastic(cantilever_case());
        law2.th_brick = RadThRequest::brick_stress(vec![1]);
        let starter = write_extended_starter(&law2);
        let groups = starter
            .lines()
            .filter(|line| line.starts_with("/TH/"))
            .count();
        assert_eq!(extended_starter_th_groups(&law2), groups);
    }

    fn consistent_extra() -> (RadExtra, RadExtraContext) {
        let ctx = RadExtraContext {
            num_nodes: 20,
            num_node_sets: 2,
            num_functions: 2,
            num_parts: 2,
            num_materials: 1,
            num_walls: 1,
            num_shells: 6,
            num_tris: 1,
            num_bricks: 1,
        };
        let mut extra = RadExtra::default();
        extra.surfs.push(RadSurfParts {
            id: 4,
            title: "skin".to_string(),
            parts: vec![1],
        });
        extra.grnods.push(RadGrnodSurfs {
            id: 5,
            title: "contact".to_string(),
            surfs: vec![4],
        });
        extra.lines.push(RadLineSurfs {
            id: 6,
            title: "edges".to_string(),
            surfs: vec![4],
        });
        extra.subsets.push(RadSubset {
            id: 1,
            title: "all".to_string(),
        });
        extra
            .interfaces
            .push(RadInterface::Type2(RadInterType2::basic(1, "tie", 5, 4)));
        extra
            .interfaces
            .push(RadInterface::Type7(RadInterType7::basic(
                2, "contact", 5, 4,
            )));
        extra
            .interfaces
            .push(RadInterface::Type11(RadInterType11::basic(
                3, "edges", 6, 6,
            )));
        extra.gases.push(RadGasPredef {
            id: 2,
            title: "air".to_string(),
            gas: RadGasKind::Air,
        });
        extra.injectors.push(RadInject2 {
            id: 6,
            title: "injector".to_string(),
            iflow: 0,
            mass_function: 1,
            temp_function: 2,
            scal_m: 1.0,
            scal_t: 0.0,
            ascal_t: 0.0,
            gases: vec![RadInjectGas {
                mat: 2,
                molar_fraction: 1.0,
                function: 0,
            }],
        });
        extra.sensors.push(RadSensor::Time {
            id: 1,
            title: "fire".to_string(),
            tdelay_s: 0.0,
        });
        extra.monvols.push(RadMonvol::Airbag1(RadMonvolAirbag1 {
            id: 1,
            title: "bag".to_string(),
            surf_external: 4,
            hconv: 0.0,
            ascale: [0.0; 5],
            mat_gas: 2,
            gas_mu: 0.1,
            pext_pa: 101_325.0,
            t0_k: 0.0,
            iequi: 0,
            ittf: 0,
            injectors: vec![RadAirbagInjector {
                inject_id: 6,
                sens_id: 1,
                ijet: 0,
                nodes: [0; 3],
                jetting: None,
            }],
            vents: Vec::new(),
            porous: Vec::new(),
        }));
        extra.damps.push(RadDamping {
            id: 1,
            title: "bulk".to_string(),
            alpha: 1.0,
            beta_s: 0.0,
            grnd: 1,
            skew: 0,
            tstart_s: 0.0,
            tstop_s: 1.0e30,
        });
        extra.truss_props.push(RadTrussProp {
            id: 3,
            title: "rod".to_string(),
            area_m2: 1.0e-4,
            gap_init_m: 0.0,
        });
        extra.parts.push(RadExtraPart {
            id: 3,
            title: "rod".to_string(),
            prop: 3,
            mat: 1,
        });
        extra.truss.push(RadTrussElems {
            part: 3,
            elems: vec![[9, 10]],
        });
        extra.th_elems.push(RadThElemRequest {
            family: RadThElemFamily::Truss,
            title: "rod".to_string(),
            vars: vec!["DEF".to_string()],
            elems: vec![1],
        });
        extra.th_objs.push(RadThObjRequest {
            kind: RadThObjKind::Part,
            title: "parts".to_string(),
            vars: vec!["DEF".to_string()],
            ids: vec![1, 2, 3],
        });
        extra.state = Some(RadStateOutput {
            tstart_s: 0.0,
            tfreq_s: 0.1,
            all_parts: false,
            parts: vec![1, 2],
        });
        extra.state_flags.push(RadStateFlag::NodeVel);
        extra.engine_damps.push(RadEngineDamping {
            damp_id: 1,
            alpha: 2.0,
            beta_s: 0.0,
        });
        extra.th_vers = Some(41);
        (extra, ctx)
    }

    #[test]
    fn extra_validate_accepts_consistent_bundle() {
        let (extra, ctx) = consistent_extra();
        extra.validate(&ctx).unwrap();
        let deck = write_starter_extra(&extra, 4);
        for header in [
            "/SURF/PART/4",
            "/GRNOD/SURF/5",
            "/LINE/SURF/6",
            "/SUBSET/1",
            "/INTER/TYPE2/1",
            "/INTER/TYPE7/2",
            "/INTER/TYPE11/3",
            "/MAT/GAS/PREDEF/2",
            "/PROP/INJECT2/6",
            "/MONVOL/AIRBAG1/1",
            "/DAMP/1",
            "/SENSOR/TIME/1",
            "/PROP/TRUSS/3",
            "/PART/3",
            "/TRUSS/3",
            "/TH/TRUSS/5",
            "/TH/PART/6",
        ] {
            assert!(deck.contains(header), "extra deck lacks {header}");
        }
        let engine = write_engine_extra(&extra);
        for header in ["/STATE/DT\n", "/STATE/NODE/VEL", "/DAMP/1", "/TH/VERS/41"] {
            assert!(engine.contains(header), "engine extra lacks {header}");
        }
    }

    #[test]
    fn extra_validate_catches_gaps() {
        let (extra, ctx) = consistent_extra();
        let mut bad = extra.clone();
        bad.interfaces
            .push(RadInterface::Type2(RadInterType2::basic(
                9, "dangling", 5, 99,
            )));
        assert!(bad.validate(&ctx).is_err());
        let mut bad = extra.clone();
        bad.sensors.push(RadSensor::Dist {
            id: 9,
            title: "gap".to_string(),
            tdelay_s: 0.0,
            node_a: 1,
            node_b: 999,
            dmin_m: 0.0,
            dmax_m: 1.0,
        });
        assert!(bad.validate(&ctx).is_err());
        let mut bad = extra.clone();
        bad.th_elems.push(RadThElemRequest {
            family: RadThElemFamily::Quad,
            title: "quad".to_string(),
            vars: vec!["DEF".to_string()],
            elems: vec![1],
        });
        assert!(bad.validate(&ctx).is_err());
        let mut bad = extra.clone();
        bad.grnods.push(RadGrnodSurfs {
            id: 1,
            title: "collision".to_string(),
            surfs: vec![4],
        });
        assert!(bad.validate(&ctx).is_err());
        let mut bad = extra.clone();
        bad.th_vers = Some(44);
        assert!(bad.validate(&ctx).is_err());
        let mut bad = extra.clone();
        bad.state.as_mut().unwrap().parts = vec![99];
        assert!(bad.validate(&ctx).is_err());
        let mut bad = extra.clone();
        bad.engine_damps[0].damp_id = 99;
        assert!(bad.validate(&ctx).is_err());
        // Dangling section/rbody/accel TH ids only warn, so they pass.
        let mut ok = extra.clone();
        for kind in [
            RadThObjKind::Sectio,
            RadThObjKind::Rbody,
            RadThObjKind::Accel,
        ] {
            ok.th_objs.push(RadThObjRequest {
                kind,
                title: "dangling".to_string(),
                vars: vec!["DEF".to_string()],
                ids: vec![9],
            });
        }
        ok.validate(&ctx).unwrap();
    }

    #[test]
    fn starter_extra_honors_submodel_wrap() {
        let (mut extra, ctx) = consistent_extra();
        extra.validate(&ctx).unwrap();
        extra.submodel = Some(RadSubmodel::plain(1, "extra"));
        let deck = write_starter_extra(&extra, 4);
        assert!(deck.starts_with("//SUBMODEL/1\n"));
        assert!(deck.ends_with("//ENDSUB\n"));
    }

    #[test]
    fn th_to_csv_arg_prefers_bare_filename_in_dir() {
        let dir = Path::new("/scratch/sonium_rad_plasticity_thick_cylinder_1");
        let t01 = dir.join("plasticity_thick_cylinderT01");
        assert_eq!(
            th_to_csv_arg(&t01, dir),
            PathBuf::from("plasticity_thick_cylinderT01")
        );
        let elsewhere = Path::new("/other/placeT01");
        assert_eq!(th_to_csv_arg(elsewhere, dir), elsewhere.to_path_buf());
    }
    #[test]
    fn successful_process_exit_cannot_hide_numerical_abort() {
        let log = "MESSAGE ID:205\n** RUN KILLED: ENERGY ERROR LIMIT REACHED\nNORMAL TERMINATION\nUSER BREAK";
        let error = validate_stage_log("engine", Some(0), log).unwrap_err();
        match error {
            RadError::EngineFailed {
                stage,
                code,
                output,
            } => {
                assert_eq!(stage, "engine");
                assert_eq!(code, Some(0));
                assert!(output.contains("ENERGY ERROR LIMIT REACHED"));
            }
            other => panic!("wrong abort classification: {other:?}"),
        }
        assert!(
            validate_stage_log("engine", Some(0), "ENERGY ERROR:0.1%\nNORMAL TERMINATION").is_ok()
        );
    }

    #[test]
    fn history_must_cover_the_original_requested_physical_time() {
        let case = shell_scordelis_lo_case().base;
        let make = |times| TimeHistory {
            times,
            channels: BTreeMap::new(),
        };
        assert!(validate_history_extent(&make(vec![0.0, 0.02, 8.0]), &case).is_ok());
        assert!(validate_history_extent(&make(vec![0.0, 0.02, 7.99]), &case).is_ok());
        assert!(validate_history_extent(&make(vec![0.0, 0.005792]), &case).is_err());
        assert!(validate_history_extent(&make(vec![0.0, 4.0]), &case).is_err());
        assert!(validate_history_extent(&make(vec![5.5, 8.0]), &case).is_err());
        assert!(validate_history_extent(&make(vec![0.0, 9.0]), &case).is_err());
        assert!(validate_history_extent(&make(vec![0.0, f64::NAN]), &case).is_err());
        assert!(validate_history_extent(&make(vec![0.0, 6.0, 5.0, 8.0]), &case).is_err());
    }
}
