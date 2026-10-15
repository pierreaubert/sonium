//! Unit 4 bench arm: native explicit shell dynamics for C3 decks.
//!
//! Runs the four `/SHELL` C3 decks (clamped plate, open-hole tension,
//! Scordelis-Lo roof, twisted beam) through the native T3 CST/DKT
//! explicit kernels ([`ShellExplicitModel`]) and checks the deck
//! `probes.json` expectations. Quad decks split each quad along the
//! `0-2` diagonal; `/SH3N` triangles run directly. Deck cards map
//! one-to-one: `/NODE`, `/SHELL`, `/SH3N`, `/PROP/SHELL` thickness,
//! `/MAT/LAW1`, `/BCS` six-flag pins, `/CLOAD` and `/GRAV` through
//! `/FUNCT` curves, and the engine `/RUN` stop time.
//!
//! Sampling mirrors the engine estimand, not its damping: the engine
//! legs settle under numerical damping, while the native run is
//! undamped, so every probe averages its samples over the deck hold
//! window (last FUNCT plateau) instead of reading the phase-sensitive
//! final state. Element resultants use the engine output convention
//! (`F = N / t`, `M = M / t^2`, verified on the OHT Kt = 2.51 and the
//! clamped-plate Kirchhoff moment); thickness probes read the deck
//! thickness, which linear shells hold constant.
//!
//! This module is self-contained on public APIs so the bench keeps
//! only a marked dispatch arm: it parses decks, aggregates repeats,
//! and emits [`BenchRow`] rows exactly like the brick path.

// Rust guideline compliant 2026-02-21

use sonium_bench::oc_compare::{BenchRow, Side};
use sonium_fem::forms::NonlinearElementForm;
use sonium_fem::mesh::Point;
use sonium_fem::physics::solid::shell::explicit::{ShellExplicitModel, ShellExplicitState};
use sonium_fem::physics::solid::shell::probe::{quad_frame, quad_resultants};
use sonium_fem::physics::solid::shell::{
    CorotationalQuadShellForm, MembraneFormulation, QuadShellElement, ShellElement, ShellSection,
};
use sonium_qa::rad::assert_probe_close;
use sonium_qa::rad_parse::{CardKind, ElemType, LoadCard, RadDeck, TypedCard};
use std::collections::BTreeMap;
use std::path::Path;

/// Native explicit step budget, mirroring the brick arm cap.
const UNIT4_MAX_STEPS: u64 = 8_000_000;
/// Cap on sampled probe points per run; stride grows to fit.
const UNIT4_MAX_SAMPLES: usize = 100_000;
/// Gate on the native energy ledger, mirroring the brick arm.
const UNIT4_DRIFT_TOL: f64 = 0.03;
/// Fixed-step safety margin below the stability limit.
const UNIT4_DT_MARGIN: f64 = 0.5;
/// Tighter margin for corotational runs, whose tangent stiffens under
/// load past the linear power-iteration bound (support precedent).
const UNIT4_COROTATIONAL_DT_MARGIN: f64 = 0.4;
/// Power-iteration count for the elastic critical step.
const UNIT4_POWER_ITERATIONS: usize = 500;

/// True when the starter deck carries shell element cards.
pub fn starter_has_shells(starter: &Path) -> Result<bool, String> {
    let deck = RadDeck::parse_file(starter).map_err(|error| error.to_string())?;
    Ok(deck
        .cards
        .iter()
        .any(|card| matches!(card.kind, CardKind::Shell | CardKind::Sh3n)))
}

/// True when the starter deck carries an SI (kg/m/s) unit line.
fn has_si_units(starter_deck_text: &str) -> bool {
    starter_deck_text
        .lines()
        .any(|line| line.split_whitespace().collect::<Vec<_>>() == ["kg", "m", "s"])
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
fn grnod_ids(fields: &[Vec<String>]) -> Result<Vec<i64>, String> {
    let mut ids = Vec::new();
    for row in fields {
        for token in row {
            let id = parse_int(token, "GRNOD node")?;
            if id != 0 {
                ids.push(id);
            }
        }
    }
    Ok(ids)
}

/// Maps a node group to compact structural indices, skipping drivers.
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
        if let Some(index) = index_of.get(id) {
            indices.push(*index);
        }
    }
    Ok(indices)
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

/// One engine-verified probe expectation from `probes.json`.
struct ShellProbe {
    label: String,
    channel: String,
    expected: f64,
    abs_tol: f64,
    rel_tol: f64,
}

/// Reads probe expectations from a C3 `probes.json` file.
fn read_probe_file(path: &Path) -> Result<Vec<ShellProbe>, String> {
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
        out.push(ShellProbe {
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

/// Peak RSS of this process in bytes, via `/proc` on Linux.
fn self_peak_rss_bytes() -> u64 {
    #[cfg(target_os = "linux")]
    if let Ok(status) = std::fs::read_to_string("/proc/self/status")
        && let Some(rest) = status.lines().find_map(|line| line.strip_prefix("VmHWM:"))
        && let Ok(kib) = rest.trim().trim_end_matches("kB").trim().parse::<u64>()
    {
        return kib.saturating_mul(1024);
    }
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 {
        return 0;
    }
    let units: u64 = usage.ru_maxrss.max(0) as u64;
    #[cfg(target_os = "macos")]
    {
        units
    }
    #[cfg(not(target_os = "macos"))]
    {
        units.saturating_mul(1024)
    }
}

/// One timed repeat: wall time plus the correctness verdict.
struct ShellSample {
    wall_s: f64,
    cycles: u64,
    dof: u64,
    peak_rss_bytes: u64,
    correct: bool,
}

/// Aggregates repeats: median wall/cycles, worst RSS, AND verdict.
fn aggregate_samples(samples: &[ShellSample]) -> Result<ShellSample, String> {
    if samples.is_empty() {
        return Err("no timed repeats to aggregate".to_string());
    }
    let mut order: Vec<usize> = (0..samples.len()).collect();
    order.sort_by(|a, b| samples[*a].wall_s.total_cmp(&samples[*b].wall_s));
    let median = &samples[order[order.len() / 2]];
    Ok(ShellSample {
        wall_s: median.wall_s,
        cycles: median.cycles,
        dof: median.dof,
        peak_rss_bytes: samples
            .iter()
            .map(|sample| sample.peak_rss_bytes)
            .max()
            .unwrap_or(0),
        correct: samples.iter().all(|sample| sample.correct),
    })
}

/// One nodal concentrated load: nodes share axis, curve, and scale.
struct ShellCload {
    nodes: Vec<usize>,
    axis: usize,
    funct: i64,
    scale: f64,
}

/// One gravity load: acceleration scale on a node set along an axis.
struct ShellGravity {
    nodes: Vec<usize>,
    axis: usize,
    funct: i64,
    accel: f64,
}

/// Native explicit shell model parsed from a C3 deck pair.
struct ShellModel {
    coords: Vec<[f64; 3]>,
    quads: Vec<[usize; 4]>,
    quad_ids: Vec<i64>,
    tris: Vec<[usize; 3]>,
    tri_ids: Vec<i64>,
    index_of: BTreeMap<i64, usize>,
    quad_index_of: BTreeMap<i64, usize>,
    tri_index_of: BTreeMap<i64, usize>,
    section: ShellSection,
    pinned: Vec<bool>,
    cloads: Vec<ShellCload>,
    gravities: Vec<ShellGravity>,
    functs: BTreeMap<i64, Vec<[f64; 2]>>,
    stop: f64,
}

/// Reads E/nu/rho from the single `/MAT/LAW1` block.
fn parse_law1(deck: &RadDeck) -> Result<(f64, f64, f64), String> {
    let mut mats = Vec::new();
    for card in &deck.cards {
        if card.kind == CardKind::MatLaw1 {
            mats.push(card);
        }
    }
    let [mat] = mats.as_slice() else {
        return Err(format!(
            "shell arm needs one /MAT/LAW1 block, found {}",
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

/// Reads the thickness from the single `/PROP/SHELL` block.
///
/// The writer emits three data rows (formulation flags, blank
/// hourglass, `N 0 Thick 0 0 Ithick Iplas 0`), so the thickness is
/// the third token of the third row.
fn parse_shell_thickness(deck: &RadDeck) -> Result<f64, String> {
    let mut props = Vec::new();
    for card in &deck.cards {
        if card.kind == CardKind::PropShell {
            props.push(card);
        }
    }
    let [prop] = props.as_slice() else {
        return Err(format!(
            "shell arm needs one /PROP/SHELL block, found {}",
            props.len()
        ));
    };
    let row = prop
        .fields
        .get(2)
        .ok_or_else(|| "/PROP/SHELL block holds no thickness row".to_string())?;
    let thick = row
        .get(2)
        .ok_or_else(|| "/PROP/SHELL thickness row is short".to_string())?;
    let thick = parse_scalar(thick, "shell thickness")?;
    if thick <= 0.0 {
        return Err(format!("shell thickness {thick:.6e} is not positive"));
    }
    Ok(thick)
}

/// Parses `/BCS` blocks into per-DOF pins over 6 DOFs per node.
fn parse_shell_bcs(
    deck: &RadDeck,
    groups: &BTreeMap<i64, Vec<i64>>,
    index_of: &BTreeMap<i64, usize>,
) -> Result<Vec<bool>, String> {
    let mut pinned = vec![false; 6 * index_of.len()];
    for card in &deck.cards {
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
        let tra: Vec<char> = bcs.tra.chars().collect();
        let rot: Vec<char> = bcs.rot.chars().collect();
        if tra.len() != 3
            || rot.len() != 3
            || !tra
                .iter()
                .chain(rot.iter())
                .all(|flag| matches!(flag, '0' | '1'))
        {
            return Err(format!(
                "/BCS codes {:?}/{:?} are not two flag triples",
                bcs.tra, bcs.rot
            ));
        }
        let group = bcs
            .grnod
            .ok_or_else(|| "/BCS block references no group".to_string())?;
        for index in group_indices(groups, group, index_of, "/BCS")? {
            for axis in 0..3 {
                pinned[6 * index + axis] |= tra[axis] == '1';
                pinned[6 * index + 3 + axis] |= rot[axis] == '1';
            }
        }
    }
    Ok(pinned)
}

/// Parses `/CLOAD` blocks into nodal shell loads.
fn parse_shell_cloads(
    deck: &RadDeck,
    groups: &BTreeMap<i64, Vec<i64>>,
    index_of: &BTreeMap<i64, usize>,
) -> Result<Vec<ShellCload>, String> {
    let mut cloads = Vec::new();
    for card in &deck.cards {
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
            other => return Err(format!("/CLOAD direction {other} unsupported on shells")),
        };
        if cload.skew != 0 || cload.sens != 0 {
            return Err("/CLOAD with skew or sensor is not supported".to_string());
        }
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
        cloads.push(ShellCload {
            nodes,
            axis,
            funct: cload.funct,
            scale: fscale * magnitude,
        });
    }
    Ok(cloads)
}

/// Parses `/GRAV` blocks into mass-proportional shell loads.
///
/// Deck rows carry `[funct, dir, skew, sens, grnod, ax, fy]`: the
/// acceleration is `fy` on the group (`grnod` 0 covers the whole
/// model); a non-unit abscissa scale would retime the curve and is
/// rejected because no deck needs it.
fn parse_shell_gravity(
    deck: &RadDeck,
    groups: &BTreeMap<i64, Vec<i64>>,
    index_of: &BTreeMap<i64, usize>,
    n_nodes: usize,
) -> Result<Vec<ShellGravity>, String> {
    let mut gravities = Vec::new();
    for card in &deck.cards {
        if card.kind != CardKind::Grav {
            continue;
        }
        let TypedCard::Load(LoadCard::Grav(grav)) = card
            .typed
            .as_ref()
            .ok_or_else(|| "/GRAV block undecodable".to_string())?
        else {
            return Err("/GRAV block undecodable".to_string());
        };
        let axis = match grav.dir.to_uppercase().as_str() {
            "X" => 0,
            "Y" => 1,
            "Z" => 2,
            other => return Err(format!("/GRAV direction {other} unsupported on shells")),
        };
        if grav.skew != 0 || grav.sens != 0 {
            return Err("/GRAV with skew or sensor is not supported".to_string());
        }
        if grav.ax.is_some_and(|ax| ax != 1.0) {
            return Err("/GRAV with non-unit time scale is not supported".to_string());
        }
        if grav.extra_scale.is_some() {
            return Err("/GRAV with an eighth token is not supported".to_string());
        }
        let accel = grav
            .fy
            .ok_or_else(|| "/GRAV block holds no magnitude".to_string())?;
        if !accel.is_finite() {
            return Err("/GRAV magnitude is not finite".to_string());
        }
        let nodes = if grav.grnod == 0 {
            (0..n_nodes).collect()
        } else {
            group_indices(groups, grav.grnod, index_of, "/GRAV")?
        };
        if nodes.is_empty() {
            return Err("/GRAV group holds no structural nodes".to_string());
        }
        gravities.push(ShellGravity {
            nodes,
            axis,
            funct: grav.funct,
            accel,
        });
    }
    Ok(gravities)
}

/// Builds the native explicit shell model from a C3 deck pair.
fn parse_shell_model(starter_path: &Path, engine_path: &Path) -> Result<ShellModel, String> {
    let starter_text = std::fs::read_to_string(starter_path)
        .map_err(|error| format!("cannot read {}: {error}", starter_path.display()))?;
    if !has_si_units(&starter_text) {
        return Err(format!(
            "shell arm needs SI (kg/m/s) C3 decks: {} carries other units",
            starter_path.display()
        ));
    }
    let starter = RadDeck::parse_file(starter_path).map_err(|error| error.to_string())?;
    let engine = RadDeck::parse_file(engine_path).map_err(|error| error.to_string())?;
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
            CardKind::Title
            | CardKind::Begin
            | CardKind::End
            | CardKind::Tfile
            | CardKind::ThTitle
            | CardKind::AnimDt
            | CardKind::Print => {}
            _ => {
                return Err(format!(
                    "shell arm supports no engine card {} (deck {})",
                    card.kind.as_str(),
                    engine_path.display()
                ));
            }
        }
    }
    let stop = stop
        .filter(|stop| *stop > 0.0)
        .ok_or_else(|| "engine deck holds no positive /RUN stop time".to_string())?;

    let mut nodes: Vec<(i64, [f64; 3])> = Vec::new();
    let mut quads: Vec<([i64; 4], i64)> = Vec::new();
    let mut tris: Vec<([i64; 3], i64)> = Vec::new();
    let mut groups: BTreeMap<i64, Vec<i64>> = BTreeMap::new();
    let mut functs: BTreeMap<i64, Vec<[f64; 2]>> = BTreeMap::new();
    for card in &starter.cards {
        match card.kind {
            CardKind::Node => {
                let TypedCard::Nodes(entries) = card
                    .typed
                    .as_ref()
                    .ok_or_else(|| format!("{} /NODE block undecodable", starter_path.display()))?
                else {
                    return Err(format!(
                        "{} /NODE block undecodable",
                        starter_path.display()
                    ));
                };
                for node in entries {
                    nodes.push((node.id, [node.x, node.y, node.z]));
                }
            }
            CardKind::Shell | CardKind::Sh3n => {
                let TypedCard::Elems(block) = card.typed.as_ref().ok_or_else(|| {
                    format!("{} element block undecodable", starter_path.display())
                })?
                else {
                    return Err(format!(
                        "{} element block undecodable",
                        starter_path.display()
                    ));
                };
                for elem in &block.elems {
                    if elem.thick.is_some() {
                        return Err(format!(
                            "shell {} carries per-element thickness, need uniform section",
                            elem.id
                        ));
                    }
                    match block.etype {
                        ElemType::Shell => {
                            if elem.nodes.len() != 4 {
                                return Err(format!(
                                    "shell {} has {} nodes, need 4",
                                    elem.id,
                                    elem.nodes.len()
                                ));
                            }
                            let mut conn = [0i64; 4];
                            conn.copy_from_slice(&elem.nodes);
                            quads.push((conn, elem.id));
                        }
                        ElemType::Tri => {
                            if elem.nodes.len() != 3 {
                                return Err(format!(
                                    "shell {} has {} nodes, need 3",
                                    elem.id,
                                    elem.nodes.len()
                                ));
                            }
                            let mut conn = [0i64; 3];
                            conn.copy_from_slice(&elem.nodes);
                            tris.push((conn, elem.id));
                        }
                        other => {
                            return Err(format!("shell arm supports no {other:?} elements"));
                        }
                    }
                }
            }
            CardKind::GrnodNode => {
                let id = card
                    .id
                    .ok_or_else(|| "/GRNOD/NODE block misses its group id".to_string())?;
                groups.insert(id, grnod_ids(&card.fields)?);
            }
            CardKind::Funct => {
                let id = card
                    .id
                    .ok_or_else(|| "/FUNCT block misses its id".to_string())?;
                let TypedCard::Funct(funct) = card
                    .typed
                    .as_ref()
                    .ok_or_else(|| format!("/FUNCT {id} undecodable"))?
                else {
                    return Err(format!("/FUNCT {id} undecodable"));
                };
                functs.insert(id, funct.points.clone());
            }
            CardKind::MatLaw1
            | CardKind::PropShell
            | CardKind::Part
            | CardKind::Unit
            | CardKind::Title
            | CardKind::Begin
            | CardKind::End
            | CardKind::ThNode
            | CardKind::ThShel
            | CardKind::ThSh3n
            | CardKind::ThShell
            | CardKind::ThTitle => {}
            CardKind::Bcs | CardKind::Cload | CardKind::Grav => {}
            _ => {
                return Err(format!(
                    "shell arm supports no starter card {} (deck {})",
                    card.kind.as_str(),
                    starter_path.display()
                ));
            }
        }
    }
    if nodes.is_empty() || (quads.is_empty() && tris.is_empty()) {
        return Err("starter deck holds no mesh (need /NODE + /SHELL or /SH3N)".to_string());
    }
    let (youngs, poisson, density) = parse_law1(&starter)?;
    let thick = parse_shell_thickness(&starter)?;
    // Compact native numbering over shell-referenced nodes only.
    let mut referenced = std::collections::BTreeSet::new();
    for (conn, _) in &quads {
        for id in conn {
            referenced.insert(*id);
        }
    }
    for (conn, _) in &tris {
        for id in conn {
            referenced.insert(*id);
        }
    }
    let coords_of: BTreeMap<i64, [f64; 3]> = nodes.into_iter().collect();
    let mut index_of: BTreeMap<i64, usize> = BTreeMap::new();
    let mut coords: Vec<[f64; 3]> = Vec::new();
    for id in &referenced {
        let position = coords_of
            .get(id)
            .ok_or_else(|| format!("shell references missing node {id}"))?;
        index_of.insert(*id, coords.len());
        coords.push(*position);
    }
    let mut native_quads = Vec::with_capacity(quads.len());
    let mut quad_ids = Vec::with_capacity(quads.len());
    let mut quad_index_of: BTreeMap<i64, usize> = BTreeMap::new();
    for (conn, id) in &quads {
        if quad_index_of.insert(*id, native_quads.len()).is_some() {
            return Err(format!("/SHELL id {id} repeats in the deck"));
        }
        native_quads.push([
            index_of[&conn[0]],
            index_of[&conn[1]],
            index_of[&conn[2]],
            index_of[&conn[3]],
        ]);
        quad_ids.push(*id);
    }
    let mut native_tris = Vec::with_capacity(tris.len());
    let mut tri_ids = Vec::with_capacity(tris.len());
    let mut tri_index_of: BTreeMap<i64, usize> = BTreeMap::new();
    for (conn, id) in &tris {
        if tri_index_of.insert(*id, native_tris.len()).is_some() {
            return Err(format!("/SH3N id {id} repeats in the deck"));
        }
        native_tris.push([index_of[&conn[0]], index_of[&conn[1]], index_of[&conn[2]]]);
        tri_ids.push(*id);
    }
    let pinned = parse_shell_bcs(&starter, &groups, &index_of)?;
    let cloads = parse_shell_cloads(&starter, &groups, &index_of)?;
    let gravities = parse_shell_gravity(&starter, &groups, &index_of, coords.len())?;
    for cload in &cloads {
        if !functs.contains_key(&cload.funct) {
            return Err(format!("/CLOAD references missing /FUNCT {}", cload.funct));
        }
    }
    for gravity in &gravities {
        if !functs.contains_key(&gravity.funct) {
            return Err(format!("/GRAV references missing /FUNCT {}", gravity.funct));
        }
    }
    Ok(ShellModel {
        coords,
        quads: native_quads,
        quad_ids,
        tris: native_tris,
        tri_ids,
        index_of,
        quad_index_of,
        tri_index_of,
        section: ShellSection::new(youngs, poisson, thick, density),
        pinned,
        cloads,
        gravities,
        functs,
        stop,
    })
}

/// Shell element family of an element probe channel.
#[derive(Clone, Copy)]
enum ShellFamily {
    Quad,
    Tri,
}

/// Native shell probe target: a nodal displacement, an element
/// resultant component, or the section thickness.
#[derive(Clone, Copy)]
enum ShellProbeTarget {
    Disp {
        axis: usize,
        node: usize,
    },
    Force {
        comp: usize,
        family: ShellFamily,
        elem: usize,
    },
    Moment {
        comp: usize,
        family: ShellFamily,
        elem: usize,
    },
    // Thickness is uniform, so the target carries no element; the
    // resolver still verifies the named element exists.
    Thick,
}

/// Resolves a shell probe channel to a native target.
fn resolve_shell_probe(channel: &str, model: &ShellModel) -> Result<ShellProbeTarget, String> {
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
        return Ok(ShellProbeTarget::Disp { axis: rest.0, node });
    }
    for (prefix, comp) in [
        ("F1/", 0),
        ("F2/", 1),
        ("F12/", 2),
        ("M1/", 0),
        ("M2/", 1),
        ("M12/", 2),
    ] {
        if let Some(rest) = channel.strip_prefix(prefix) {
            let moment = prefix.starts_with('M');
            let (family, id) = if let Some(id) = rest.strip_prefix("SHEL/") {
                (ShellFamily::Quad, id)
            } else if let Some(id) = rest.strip_prefix("SH3N/") {
                (ShellFamily::Tri, id)
            } else {
                return Err(format!("probe channel {channel} names no shell family"));
            };
            let number: i64 = id
                .parse()
                .ok()
                .ok_or_else(|| format!("probe channel {channel} names no element"))?;
            let map = match family {
                ShellFamily::Quad => &model.quad_index_of,
                ShellFamily::Tri => &model.tri_index_of,
            };
            let elem = map
                .get(&number)
                .copied()
                .ok_or_else(|| format!("probe channel {channel} names a missing element"))?;
            if moment {
                return Ok(ShellProbeTarget::Moment { comp, family, elem });
            }
            return Ok(ShellProbeTarget::Force { comp, family, elem });
        }
    }
    if let Some(rest) = channel.strip_prefix("THIC/") {
        let (family, id) = if let Some(id) = rest.strip_prefix("SHEL/") {
            (ShellFamily::Quad, id)
        } else if let Some(id) = rest.strip_prefix("SH3N/") {
            (ShellFamily::Tri, id)
        } else {
            return Err(format!("probe channel {channel} names no shell family"));
        };
        let number: i64 = id
            .parse()
            .ok()
            .ok_or_else(|| format!("probe channel {channel} names no element"))?;
        let map = match family {
            ShellFamily::Quad => &model.quad_index_of,
            ShellFamily::Tri => &model.tri_index_of,
        };
        if !map.contains_key(&number) {
            return Err(format!("probe channel {channel} names a missing element"));
        }
        return Ok(ShellProbeTarget::Thick);
    }
    Err(format!(
        "probe channel {channel} unsupported (need DX/DY/DZ, F1/F2/F12, M1/M2/M12, or THIC per SHEL/SH3N)"
    ))
}

/// Hold-window start of a shell deck in seconds.
///
/// Each window covers the settled second half of the deck hold,
/// transcribed from the deck `/FUNCT` ramp end and stop: plate
/// ramps 0.8 s of 1.0 s, OHT 1 ms of 2 ms, twisted 20 ms of 60 ms
/// (two ~14 ms bending periods), Scordelis 5 s of 8 s. Unknown
/// decks fail closed so new holds get explicit review.
fn hold_start(deck: &str) -> Result<f64, String> {
    match deck {
        "shell_clamped_plate" => Ok(0.9),
        "shell_oht_elastic" => Ok(0.0015),
        "shell_twisted_beam" => Ok(0.032),
        "shell_scordelis_lo" => Ok(5.5),
        _ => Err(format!("shell deck {deck} has no reviewed hold window")),
    }
}

/// Unsigned triangle area from global corner coordinates.
fn tri_area(coords: &[[f64; 3]; 3]) -> f64 {
    let ax = coords[1][0] - coords[0][0];
    let ay = coords[1][1] - coords[0][1];
    let az = coords[1][2] - coords[0][2];
    let bx = coords[2][0] - coords[0][0];
    let by = coords[2][1] - coords[0][1];
    let bz = coords[2][2] - coords[0][2];
    let nx = ay * bz - az * by;
    let ny = az * bx - ax * bz;
    let nz = ax * by - ay * bx;
    0.5 * (nx * nx + ny * ny + nz * nz).sqrt()
}

/// T3 dynamics mesh built from the deck shells plus probe lookup.
struct ShellTriMesh {
    elements: Vec<ShellElement>,
    quad_children: Vec<[usize; 2]>,
    tri_child: Vec<usize>,
}

/// Splits quads along the 0-2 diagonal and keeps deck triangles.
fn build_tri_mesh(model: &ShellModel) -> Result<ShellTriMesh, String> {
    let point = |node: usize| {
        let xyz = model.coords[node];
        Point::new_3d(xyz[0], xyz[1], xyz[2])
    };
    let mut elements = Vec::new();
    let mut quad_children = Vec::with_capacity(model.quads.len());
    for (quad_id, conn) in model.quad_ids.iter().zip(model.quads.iter()) {
        let mut children = [0usize; 2];
        for (slot, local) in [[0, 1, 2], [0, 2, 3]].iter().enumerate() {
            let ids = [conn[local[0]], conn[local[1]], conn[local[2]]];
            let corners = [
                model.coords[ids[0]],
                model.coords[ids[1]],
                model.coords[ids[2]],
            ];
            if !tri_area(&corners).is_finite() || tri_area(&corners) <= 0.0 {
                return Err(format!(
                    "/SHELL {quad_id} splits into a degenerate triangle"
                ));
            }
            children[slot] = elements.len();
            elements.push(ShellElement::from_points(
                ids,
                &point(ids[0]),
                &point(ids[1]),
                &point(ids[2]),
                model.section,
            ));
        }
        quad_children.push(children);
    }
    let mut tri_child = Vec::with_capacity(model.tris.len());
    for (tri_id, conn) in model.tri_ids.iter().zip(model.tris.iter()) {
        let corners = [
            model.coords[conn[0]],
            model.coords[conn[1]],
            model.coords[conn[2]],
        ];
        if !tri_area(&corners).is_finite() || tri_area(&corners) <= 0.0 {
            return Err(format!("/SH3N {tri_id} is degenerate"));
        }
        tri_child.push(elements.len());
        elements.push(ShellElement::from_points(
            *conn,
            &point(conn[0]),
            &point(conn[1]),
            &point(conn[2]),
            model.section,
        ));
    }
    Ok(ShellTriMesh {
        elements,
        quad_children,
        tri_child,
    })
}

/// Assembles the external force vector at a time in N.
fn external_forces(model: &ShellModel, masses: &[f64], time: f64) -> Result<Vec<f64>, String> {
    let mut forces = vec![0.0; masses.len()];
    for cload in &model.cloads {
        let curve = model
            .functs
            .get(&cload.funct)
            .ok_or_else(|| format!("load curve {} went missing", cload.funct))?;
        let value = cload.scale * eval_funct(curve, time)?;
        for node in &cload.nodes {
            forces[6 * node + cload.axis] += value;
        }
    }
    for gravity in &model.gravities {
        let curve = model
            .functs
            .get(&gravity.funct)
            .ok_or_else(|| format!("load curve {} went missing", gravity.funct))?;
        let accel = gravity.accel * eval_funct(curve, time)?;
        for node in &gravity.nodes {
            forces[6 * node + gravity.axis] += masses[6 * node + gravity.axis] * accel;
        }
    }
    Ok(forces)
}

/// Samples one resultant probe from the current displacement state.
///
/// Membrane and bending resultants come from the child triangles
/// in the quad frame (single-triangle probes use the child frame,
/// which is the quad frame of one); outputs follow the engine
/// convention `F = N / t`, `M = M / t^2`.
fn sample_resultant(
    model: &ShellModel,
    mesh: &ShellTriMesh,
    target: ShellProbeTarget,
    u: &[f64],
) -> Result<f64, String> {
    let thick = model.section.thickness;
    let (family, elem, moment, comp) = match target {
        ShellProbeTarget::Force { comp, family, elem } => (family, elem, false, comp),
        ShellProbeTarget::Moment { comp, family, elem } => (family, elem, true, comp),
        _ => return Err("resultant sampler needs a force or moment target".to_string()),
    };
    let (membrane, bending) = match family {
        ShellFamily::Quad => {
            let conn = model.quads[elem];
            let frame = quad_frame(&[
                model.coords[conn[0]],
                model.coords[conn[1]],
                model.coords[conn[2]],
                model.coords[conn[3]],
            ])
            .map_err(|error| error.to_string())?;
            let children = mesh.quad_children[elem];
            quad_resultants(
                &model.section,
                &frame,
                [&mesh.elements[children[0]], &mesh.elements[children[1]]],
                u,
            )
            .map_err(|error| error.to_string())?
        }
        ShellFamily::Tri => {
            let conn = model.tris[elem];
            let frame = quad_frame(&[
                model.coords[conn[0]],
                model.coords[conn[1]],
                model.coords[conn[2]],
                model.coords[conn[2]],
            ])
            .map_err(|error| error.to_string())?;
            let child = mesh.tri_child[elem];
            quad_resultants(
                &model.section,
                &frame,
                [&mesh.elements[child], &mesh.elements[child]],
                u,
            )
            .map_err(|error| error.to_string())?
        }
    };
    if moment {
        Ok(bending[comp] / (thick * thick))
    } else {
        Ok(membrane[comp] / thick)
    }
}

/// Runs one native explicit shell repeat over the deck stop time.
fn shell_repeat(
    deck: &str,
    model: &ShellModel,
    probes: &[ShellProbe],
) -> Result<ShellSample, String> {
    let mesh = build_tri_mesh(model)?;
    let n_nodes = model.coords.len();
    let n_dofs = 6 * n_nodes;
    let mut dynamics = ShellExplicitModel::elastic(mesh.elements.clone(), n_dofs)
        .map_err(|error| format!("shell model build failed: {error}"))?;
    for (dof, pinned) in model.pinned.iter().enumerate() {
        if *pinned {
            dynamics.constrain_dof(dof);
        }
    }
    let masses = dynamics.masses().to_vec();
    let wave = dynamics.wave_dt();
    let critical = dynamics
        .elastic_critical_dt(UNIT4_POWER_ITERATIONS)
        .map_err(|error| format!("shell power iteration failed: {error}"))?;
    let dt_target = UNIT4_DT_MARGIN * wave.min(critical);
    if !dt_target.is_finite() || dt_target <= 0.0 {
        return Err("shell dt bound is not positive".to_string());
    }
    let estimated = (model.stop / dt_target).ceil() as u64;
    if estimated > UNIT4_MAX_STEPS {
        return Err(format!(
            "shell run needs about {estimated} steps (stop {}s, dt {dt_target:.3e}s), exceeding step budget {UNIT4_MAX_STEPS}",
            model.stop
        ));
    }
    let steps = estimated.max(1);
    let dt = model.stop / steps as f64;
    println!(
        "sonium mesh: {n_nodes} nodes, {} quads, {} tris, dt {dt:.3e}s, {steps} steps",
        model.quads.len(),
        model.tris.len()
    );
    let targets: Vec<ShellProbeTarget> = probes
        .iter()
        .map(|probe| resolve_shell_probe(&probe.channel, model))
        .collect::<Result<_, _>>()?;
    let mut times: Vec<f64> = Vec::new();
    let mut values: Vec<Vec<f64>> = vec![Vec::new(); probes.len()];

    let start = std::time::Instant::now();
    let mut state = ShellExplicitState::new(n_dofs);
    let mut force_prev = external_forces(model, &masses, 0.0)?;
    let mut u_prev = state.u.clone();
    let mut external_work = 0.0;
    for _ in 0..steps {
        let load = external_forces(model, &masses, state.time)?;
        dynamics
            .step(&mut state, &load, dt)
            .map_err(|error| format!("shell step failed: {error}"))?;
        let force = external_forces(model, &masses, state.time)?;
        for dof in 0..n_dofs {
            external_work += 0.5 * (force_prev[dof] + force[dof]) * (state.u[dof] - u_prev[dof]);
        }
        force_prev = force;
        u_prev.clone_from(&state.u);
        if !state.u.iter().chain(state.v.iter()).all(|v| v.is_finite()) {
            return Err("shell state became non-finite".to_string());
        }
        // Every step is sampled; the hold mean thins a posteriori so
        // short runs keep full resolution and long runs stay bounded.
        times.push(state.time);
        for (slot, target) in targets.iter().enumerate() {
            let value = match *target {
                ShellProbeTarget::Disp { axis, node } => state.u[6 * node + axis],
                ShellProbeTarget::Thick => model.section.thickness,
                force_or_moment => sample_resultant(model, &mesh, force_or_moment, &state.u)?,
            };
            values[slot].push(value);
        }
    }
    let wall_s = start.elapsed().as_secs_f64();
    println!(
        "sonium run: reached t = {:.6e}s in {steps} steps",
        state.time
    );
    let mut failures = check_hold_probes(deck, probes, &times, &values)?;
    for failure in &failures {
        println!("FAIL: {failure}");
    }
    let kinetic = dynamics.kinetic_energy(&state.v);
    let internal = dynamics
        .elastic_strain_energy(&state.u)
        .map_err(|error| format!("shell strain energy failed: {error}"))?;
    let energy_scale = kinetic + internal.abs();
    let drift_frac = if energy_scale > 0.0 {
        (kinetic + internal - external_work).abs() / energy_scale
    } else {
        f64::INFINITY
    };
    println!(
        "energy: KE {kinetic:.6e}J, IE {internal:.6e}J, Wext {external_work:.6e}J, drift {drift_frac:.3e} rel"
    );
    if !drift_frac.is_finite() || drift_frac > UNIT4_DRIFT_TOL {
        failures.push(format!(
            "energy drift {drift_frac:.3e} exceeds {UNIT4_DRIFT_TOL:.3e}"
        ));
    }
    // DOF convention matches the engine arm (`deck_dof` counts 3 per
    // deck node): rotations are integrated on both sides but excluded
    // from both counts, so the per-dof cycle cost compares fairly.
    let dof =
        u64::try_from(3 * n_nodes).map_err(|_| "shell dof count overflows u64".to_string())?;
    let sample = ShellSample {
        wall_s,
        cycles: steps,
        dof,
        peak_rss_bytes: self_peak_rss_bytes(),
        correct: failures.is_empty(),
    };
    println!(
        "sonium: wall {:.3}s, cycles {}, dof {dof}, correct {}",
        sample.wall_s, sample.cycles, sample.correct
    );
    Ok(sample)
}

/// Checks hold-window probe means against expectations.
fn check_hold_probes(
    deck: &str,
    probes: &[ShellProbe],
    times: &[f64],
    values: &[Vec<f64>],
) -> Result<Vec<String>, String> {
    let start_hold = hold_start(deck)?;
    // Thin the stride-free record down to the sample cap a posteriori
    // so short runs keep every step and long runs stay bounded.
    let thin = (times.len() / UNIT4_MAX_SAMPLES).max(1);
    let mut failures = Vec::new();
    for (probe, history) in probes.iter().zip(values.iter()) {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut count = 0usize;
        for (index, value) in history.iter().enumerate() {
            if index % thin != 0 || times[index] < start_hold {
                continue;
            }
            sum += value;
            sum_sq += value * value;
            count += 1;
        }
        if count == 0 {
            failures.push(format!("probe {} has no hold samples", probe.label));
            continue;
        }
        let mean = sum / count as f64;
        let std = ((sum_sq / count as f64 - mean * mean).max(0.0)).sqrt();
        println!(
            "probe {} [{}]: hold mean {mean:.6e} (std {std:.3e}, {count} samples), expected {:.6e}",
            probe.label, probe.channel, probe.expected
        );
        if let Err(error) = assert_probe_close(
            mean,
            probe.expected,
            probe.abs_tol,
            probe.rel_tol,
            &probe.label,
        ) {
            failures.push(error.to_string());
        }
    }
    Ok(failures)
}

/// True for decks whose deflection needs geometric nonlinearity.
///
/// Only the Scordelis-Lo roof qualifies: its 0.26 m hold deflection
/// matches its 0.25 m thickness, and the engine runs full geometric
/// nonlinearity (`Ismstr` 2), so the linear formulation is the wrong
/// physics there. The other shell decks deflect under 2% of their
/// thickness (verified linear regime) and keep the linear arm.
fn needs_geometric_nonlinearity(deck: &str) -> bool {
    deck == "shell_scordelis_lo"
}

/// Scatters corotational MITC4 quad forces to the global DOFs.
fn corotational_quad_forces(form: &CorotationalQuadShellForm, u: &[f64]) -> Vec<f64> {
    let mut forces = vec![0.0; form.num_global_dofs()];
    for elem in 0..form.num_elements() {
        for (dof, value) in form
            .local_to_global(elem)
            .iter()
            .zip(form.element_residual(elem, u))
        {
            forces[*dof] += value;
        }
    }
    forces
}

/// Builds warp-aware MITC4 quad elements for the deck quads.
fn build_quad_elements(model: &ShellModel) -> Vec<QuadShellElement> {
    let point = |node: usize| {
        let xyz = model.coords[node];
        Point::new_3d(xyz[0], xyz[1], xyz[2])
    };
    let mut elements = Vec::with_capacity(model.quads.len());
    for conn in &model.quads {
        let mut element = QuadShellElement::from_points_with_section(
            *conn,
            &point(conn[0]),
            &point(conn[1]),
            &point(conn[2]),
            &point(conn[3]),
            model.section,
        );
        // Warp-aware covariant membrane with Allman drilling; the
        // legacy flat facet diverges on warped quads.
        element.membrane = MembraneFormulation::WarpAware;
        elements.push(element);
    }
    elements
}

/// Runs one corotational shell repeat over the deck stop time.
///
/// Velocity Verlet on corotational MITC4 quad forces (warp-aware
/// membrane) with physical lumped mass, fixed dt below the
/// Gershgorin stability bound, and the same hold-window probe rule
/// as the linear arm. Quads match the deck with no split-diagonal
/// bias; the energy ledger compares kinetic plus exact corotated
/// elastic potential against external work.
fn shell_repeat_corotational(
    deck: &str,
    model: &ShellModel,
    probes: &[ShellProbe],
) -> Result<ShellSample, String> {
    if !model.tris.is_empty() {
        return Err("corotational quad arm needs an all-quad mesh".to_string());
    }
    // Area validation reuses the split-mesh builder (its T3 elements
    // are discarded; only the degenerate-geometry check matters).
    let _ = build_tri_mesh(model)?;
    let n_nodes = model.coords.len();
    let n_dofs = 6 * n_nodes;
    let elements = build_quad_elements(model);
    let form = CorotationalQuadShellForm::new(elements.clone(), n_dofs);
    let mut masses = vec![0.0; n_dofs];
    for element in &elements {
        let mass = element.mass_global();
        for (row, dof) in element.global_dofs().iter().enumerate() {
            masses[*dof] += mass[24 * row + row];
        }
    }
    if masses.iter().any(|mass| !mass.is_finite() || *mass <= 0.0) {
        return Err("corotational quad mesh has non-positive lumped mass".to_string());
    }
    // Gershgorin bound on M^-1/2 K M^-1/2 (initial config): the max
    // absolute row sum bounds lambda_max from above, so 2/sqrt(bound)
    // is a conservative critical step (support precedent).
    let mut row_sums = vec![0.0; n_dofs];
    for element in &elements {
        let stiffness = element.stiffness_global();
        let ids = element.global_dofs();
        for (row, global_row) in ids.iter().enumerate() {
            if model.pinned[*global_row] {
                continue;
            }
            for (column, global_column) in ids.iter().enumerate() {
                if !model.pinned[*global_column] {
                    row_sums[*global_row] += stiffness[24 * row + column].abs()
                        / (masses[*global_row] * masses[*global_column]).sqrt();
                }
            }
        }
    }
    let bound = row_sums.iter().copied().fold(0.0, f64::max);
    if !bound.is_finite() || bound <= 0.0 {
        return Err("corotational stability bound is not positive".to_string());
    }
    let dt_target = UNIT4_COROTATIONAL_DT_MARGIN * 2.0 / bound.sqrt();
    if !dt_target.is_finite() || dt_target <= 0.0 {
        return Err("shell dt bound is not positive".to_string());
    }
    let estimated = (model.stop / dt_target).ceil() as u64;
    if estimated > UNIT4_MAX_STEPS {
        return Err(format!(
            "shell run needs about {estimated} steps (stop {}s, dt {dt_target:.3e}s), exceeding step budget {UNIT4_MAX_STEPS}",
            model.stop
        ));
    }
    let steps = estimated.max(1);
    let dt = model.stop / steps as f64;
    println!(
        "sonium mesh: {n_nodes} nodes, {} quads, dt {dt:.3e}s, {steps} steps (corotational MITC4)",
        model.quads.len(),
    );
    let targets: Vec<ShellProbeTarget> = probes
        .iter()
        .map(|probe| resolve_shell_probe(&probe.channel, model))
        .collect::<Result<_, _>>()?;
    if targets.iter().any(|target| {
        matches!(
            target,
            ShellProbeTarget::Force { .. } | ShellProbeTarget::Moment { .. }
        )
    }) {
        return Err("resultant probes need the linear arm (total-displacement strain)".to_string());
    }
    let mut times: Vec<f64> = Vec::new();
    let mut values: Vec<Vec<f64>> = vec![Vec::new(); probes.len()];

    let start = std::time::Instant::now();
    let mut u = vec![0.0; n_dofs];
    let mut v = vec![0.0; n_dofs];
    let mut time = 0.0;
    let mut internal = corotational_quad_forces(&form, &u);
    let mut external = external_forces(model, &masses, 0.0)?;
    let mut external_work = 0.0;
    for _ in 0..steps {
        let u_prev = u.clone();
        for dof in 0..n_dofs {
            if !model.pinned[dof] {
                v[dof] += 0.5 * dt * (external[dof] - internal[dof]) / masses[dof];
                u[dof] += dt * v[dof];
            }
        }
        time += dt;
        let next_internal = corotational_quad_forces(&form, &u);
        let next_external = external_forces(model, &masses, time)?;
        for dof in 0..n_dofs {
            if !model.pinned[dof] {
                v[dof] += 0.5 * dt * (next_external[dof] - next_internal[dof]) / masses[dof];
                external_work +=
                    0.5 * (external[dof] + next_external[dof]) * (u[dof] - u_prev[dof]);
            }
        }
        internal = next_internal;
        external = next_external;
        if !u.iter().chain(v.iter()).all(|value| value.is_finite()) {
            return Err("shell state became non-finite".to_string());
        }
        times.push(time);
        for (slot, target) in targets.iter().enumerate() {
            let value = match *target {
                ShellProbeTarget::Disp { axis, node } => u[6 * node + axis],
                ShellProbeTarget::Thick => model.section.thickness,
                ShellProbeTarget::Force { .. } | ShellProbeTarget::Moment { .. } => {
                    return Err("resultant probes need the linear arm".to_string());
                }
            };
            values[slot].push(value);
        }
    }
    let wall_s = start.elapsed().as_secs_f64();
    println!("sonium run: reached t = {time:.6e}s in {steps} steps");
    let mut failures = check_hold_probes(deck, probes, &times, &values)?;
    for failure in &failures {
        println!("FAIL: {failure}");
    }
    let kinetic: f64 = v
        .iter()
        .zip(masses.iter())
        .map(|(velocity, mass)| 0.5 * mass * velocity * velocity)
        .sum();
    let potential: f64 = (0..form.num_elements())
        .map(|index| form.element_strain_energy(index, &u))
        .sum();
    if !potential.is_finite() {
        failures.push("corotational elastic potential is non-finite".to_string());
    }
    let energy_scale = kinetic + potential.abs() + external_work.abs();
    let drift_frac = if energy_scale > 0.0 {
        (kinetic + potential - external_work).abs() / energy_scale
    } else {
        f64::INFINITY
    };
    println!(
        "energy: KE {kinetic:.6e}J, PE {potential:.6e}J, Wext {external_work:.6e}J, drift {drift_frac:.3e} rel"
    );
    if !drift_frac.is_finite() || drift_frac > UNIT4_DRIFT_TOL {
        failures.push(format!(
            "energy drift {drift_frac:.3e} exceeds {UNIT4_DRIFT_TOL:.3e}"
        ));
    }
    // DOF convention matches the engine arm (`deck_dof` counts 3 per
    // deck node): rotations are integrated on both sides but excluded
    // from both counts, so the per-dof cycle cost compares fairly.
    let dof =
        u64::try_from(3 * n_nodes).map_err(|_| "shell dof count overflows u64".to_string())?;
    let sample = ShellSample {
        wall_s,
        cycles: steps,
        dof,
        peak_rss_bytes: self_peak_rss_bytes(),
        correct: failures.is_empty(),
    };
    println!(
        "sonium: wall {:.3}s, cycles {}, dof {dof}, correct {}",
        sample.wall_s, sample.cycles, sample.correct
    );
    Ok(sample)
}

/// Runs the native shell arm over warmup plus timed repeats.
///
/// The deck pair must carry `/SHELL` or `/SH3N` cards; brick decks
/// keep the established arm behind the bench dispatch.
///
/// # Errors
///
/// Returns a fail-closed reason for unreadable decks, unsupported
/// cards, unstable steps, non-finite states, or missing holds.
pub fn run_shell_arm(
    threads: usize,
    deck: &str,
    starter: &Path,
    engine: &Path,
    probes_path: &Path,
    repeats: u32,
    warmup: u32,
) -> Result<BenchRow, String> {
    let probes = read_probe_file(probes_path)?;
    let model = parse_shell_model(starter, engine)?;
    // Large-deflection decks integrate the corotational forces;
    // small-deflection decks keep the linear explicit model.
    let repeat = if needs_geometric_nonlinearity(deck) {
        shell_repeat_corotational
    } else {
        shell_repeat
    };
    for warmup_index in 0..warmup {
        println!("sonium {deck}: warmup {}/{}", warmup_index + 1, warmup);
        let _ = repeat(deck, &model, &probes)?;
    }
    let mut samples = Vec::new();
    for repeat_index in 0..repeats {
        println!("sonium {deck}: repeat {}/{}", repeat_index + 1, repeats);
        samples.push(repeat(deck, &model, &probes)?);
    }
    let total = aggregate_samples(&samples)?;
    BenchRow::measured(
        deck,
        Side::Sonium,
        "native",
        u32::try_from(threads.max(1)).map_err(|_| "thread count overflows u32".to_string())?,
        total.wall_s,
        total.cycles,
        total.dof,
        total.peak_rss_bytes,
        total.correct,
    )
    .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn deck_paths(deck: &str) -> (PathBuf, PathBuf, PathBuf) {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("decks")
            .join(deck);
        (
            dir.join(format!("{deck}_0000.rad")),
            dir.join(format!("{deck}_0001.rad")),
            dir.join("probes.json"),
        )
    }

    #[test]
    fn shell_detection_splits_brick_and_shell_decks() {
        let (starter, _, _) = deck_paths("shell_clamped_plate");
        assert!(starter_has_shells(&starter).unwrap());
        let (starter, _, _) = deck_paths("taylor_bar");
        assert!(!starter_has_shells(&starter).unwrap());
    }

    #[test]
    fn all_shell_decks_parse_with_resolving_probes() {
        for deck in [
            "shell_clamped_plate",
            "shell_oht_elastic",
            "shell_scordelis_lo",
            "shell_twisted_beam",
        ] {
            let (starter, engine, probes_path) = deck_paths(deck);
            let model = parse_shell_model(&starter, &engine)
                .unwrap_or_else(|error| panic!("{deck}: {error}"));
            assert!(!model.coords.is_empty(), "{deck}: no nodes");
            assert!(
                !model.quads.is_empty() || !model.tris.is_empty(),
                "{deck}: no shells"
            );
            assert!(model.stop > 0.0, "{deck}: no stop");
            let probes = read_probe_file(&probes_path).unwrap();
            for probe in &probes {
                resolve_shell_probe(&probe.channel, &model)
                    .unwrap_or_else(|error| panic!("{deck}: {error}"));
            }
            hold_start(deck).unwrap();
            build_tri_mesh(&model).unwrap_or_else(|error| panic!("{deck}: {error}"));
        }
    }

    #[test]
    fn unknown_shell_deck_has_no_hold_window() {
        assert!(hold_start("no_such_shell_deck").is_err());
    }

    #[test]
    fn corotational_dispatch_covers_only_the_roof() {
        assert!(needs_geometric_nonlinearity("shell_scordelis_lo"));
        for deck in [
            "shell_clamped_plate",
            "shell_oht_elastic",
            "shell_twisted_beam",
        ] {
            assert!(!needs_geometric_nonlinearity(deck));
        }
    }

    #[test]
    fn quad_corotational_matches_linear_at_small_displacement() {
        let (starter, engine, _) = deck_paths("shell_scordelis_lo");
        let model = parse_shell_model(&starter, &engine).unwrap();
        let elements = build_quad_elements(&model);
        let n_dofs = 6 * model.coords.len();
        let form = CorotationalQuadShellForm::new(elements.clone(), n_dofs);
        // Tiny quadratic field: corotational quad forces equal the
        // linear MITC4 map to first order.
        let mut u = vec![0.0; n_dofs];
        for (node, xyz) in model.coords.iter().enumerate() {
            u[6 * node] = 1.0e-9 * xyz[0] * xyz[0];
            u[6 * node + 1] = 1.0e-9 * xyz[0] * xyz[1];
            u[6 * node + 2] = 1.0e-9 * (xyz[0] * xyz[0] + xyz[1] * xyz[1]);
        }
        let nonlinear = corotational_quad_forces(&form, &u);
        let mut reference = vec![0.0; n_dofs];
        for element in &elements {
            let stiffness = element.stiffness_global();
            let ids = element.global_dofs();
            for (row, dof) in ids.iter().enumerate() {
                let mut sum = 0.0;
                for (col, source) in ids.iter().enumerate() {
                    sum += stiffness[row * 24 + col] * u[*source];
                }
                reference[*dof] += sum;
            }
        }
        let norm = reference.iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(norm > 0.0);
        for (actual, expected) in nonlinear.iter().zip(reference.iter()) {
            assert!(
                (actual - expected).abs() < 1.0e-6 * norm,
                "quad corotational/linear mismatch {actual:.3e} vs {expected:.3e}"
            );
        }
    }
}
