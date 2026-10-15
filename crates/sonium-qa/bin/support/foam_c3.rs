//! Native foam-cube replay arm for `qa-oc-bench` (Unit 3).
//!
//! Prescribed-motion material-point replays for the foam C3 decks
//! (`foam76_compression`, `foam76_tension`, `foam33_cube`). Every node of
//! these single-brick fixtures is kinematically driven (fixed left face,
//! imposed axial displacement on the right face, transverse motion pinned),
//! so the native arm replays the deck load path through the deck-selected
//! law ([`Samp76Quadratic`](sonium_fem::physics::solid::plasticity::samp76_quadratic::Samp76Quadratic)
//! for LAW76,
//! [`FoamPlas33`](sonium_fem::physics::solid::plasticity::foam_plas33::FoamPlas33)
//! for FOAM_PLAS) at a fixed substep density, booking midpoint stress work.
//! This mirrors the validated `qa-fem-foam-laws` replays (stress/work within
//! 0.2-0.5% of the harvested engine histories) but reads mesh, path, and
//! material from the C3 pair instead of the golden archives, and reports
//! bench cycles/dof.
//!
//! Equivalence basis: same mesh (one brick, 8 nodes, dof 24 both arms),
//! same material card and parameters, same imposed-displacement path and
//! stop time. What differs: the engine integrates full explicit dynamics
//! (5683-5820 cycles) while sonium replays the constrained material point
//! (fixed 30720 substeps); cycle counts differ and are normalized by
//! ns/dof/cycle, mirroring the own-dt-per-side precedent. Probe channels
//! match the engine TH names; verbatim `th_to_csv` brick/node headers are
//! whitespace-collapsed before matching so converter padding cannot skew.

// Rust guideline compliant 2026-02-21

use super::{C3Decks, Probe, RepeatSample, aggregate_repeats, read_probe_file, resolve_c3_decks};
use sonium_bench::oc_compare::{BenchRow, Side};
use sonium_fem::physics::solid::plasticity::foam_plas33::{
    FoamPlas33, FoamPlasIncrement, FoamPlasState,
};
use sonium_fem::physics::solid::plasticity::funct::Funct1D;
use sonium_fem::physics::solid::plasticity::samp76_quadratic::{
    Samp76Quadratic, Samp76QuadraticState,
};
use sonium_qa::rad::assert_probe_close;
use sonium_qa::rad_parse::{CardKind, DeckCard, ElemType, LoadCard, RadDeck, TypedCard};
use std::collections::BTreeMap;

/// Replay substeps per millisecond of load path.
///
/// Matches the validated `qa-fem-foam-laws` density (256 substeps per
/// 25 us TH cadence: 256 / 0.025 ms = 10240/ms), so the 3 ms fixtures
/// run a fixed 30720 substeps at a uniform 97.7 ns. Lowering this
/// coarsens the return solve and can breach probe tolerances; raising
/// it slows the bench row linearly with no accuracy need.
const SUBSTEPS_PER_MS: f64 = 10240.0;

/// Foam C3 deck ids served by this arm (`run_sonium_arm` routes here).
pub(super) const FOAM_DECKS: [&str; 3] = ["foam76_compression", "foam76_tension", "foam33_cube"];

/// Parsed foam-cube fixture: geometry, path, and selected law.
#[derive(Debug)]
struct FoamModel {
    side_m: f64,
    volume_m3: f64,
    path: Vec<[f64; 2]>,
    stop_s: f64,
    dof: u64,
    law: FoamLaw,
}

/// Deck-selected foam law with its parsed parameters.
#[derive(Debug)]
enum FoamLaw {
    Law76 {
        young: f64,
        poisson: f64,
        tension: f64,
        compression: f64,
        shear: f64,
        hardening: Funct1D,
    },
    Law33 {
        young: f64,
        a: f64,
        b: f64,
        c: f64,
        pressure0: f64,
        porosity: f64,
        gamma0: f64,
    },
}

/// Final replay observations checked against the C3 probes.
#[derive(Debug, Clone, Copy)]
struct FoamResult {
    work_j: f64,
    sxx_pa: f64,
    /// Equivalent plastic strain; `None` for LAW33 (the engine PLAS
    /// variable stays zero on that fixture while the native cap tally
    /// is a different quantity, so LAW33 decks carry no ep probe).
    ep: Option<f64>,
    dx_m: f64,
    substeps: u64,
}

/// Parses one finite f64 token with deck context.
fn scalar(token: &str, what: &str) -> Result<f64, String> {
    token
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite())
        .ok_or_else(|| format!("{what} token {token:?} is not a finite number"))
}

/// Parses one i64 token with deck context.
fn integer(token: &str, what: &str) -> Result<i64, String> {
    token
        .parse::<i64>()
        .ok()
        .ok_or_else(|| format!("{what} token {token:?} is not an integer"))
}

/// Reads a raw data row from a deck card.
fn card_row<'a>(card: &'a DeckCard, row: usize, what: &str) -> Result<&'a [String], String> {
    card.fields
        .get(row)
        .map(Vec::as_slice)
        .ok_or_else(|| format!("{} block holds no {what} row", card.kind.as_str()))
}

/// Collects cards of one family, failing closed on miscounts.
fn one_card<'a>(deck: &'a RadDeck, kind: CardKind, what: &str) -> Result<&'a DeckCard, String> {
    let mut found = None;
    for card in &deck.cards {
        if card.kind == kind {
            if found.is_some() {
                return Err(format!("foam arm needs one {what} block, found several"));
            }
            found = Some(card);
        }
    }
    found.ok_or_else(|| format!("foam arm needs one {what} block, found none"))
}

/// Flattens a `/GRNOD/NODE` block to deck node ids, skipping padding.
fn group_ids(card: &DeckCard) -> Result<Vec<i64>, String> {
    let mut ids = Vec::new();
    for row in &card.fields {
        for token in row {
            let id = integer(token, "GRNOD node")?;
            if id != 0 {
                ids.push(id);
            }
        }
    }
    Ok(ids)
}

/// Parses the `/MAT/LAW76` quadratic SAMP card and its yield curves.
///
/// Row layout mirrors `scripts/generate_foam76_deck.py`: density; E/nu;
/// tension/compression/shear funct ids; unit scales; plastic Poisson
/// ratio plus zeros; two zero rows; `IFORM/IQUAD/ICONV` flags. Strengths
/// are the first ordinates of `/FUNCT/2,3,4`; hardening is the
/// tension curve normalized by its initial strength. The shear curve
/// abscissa follows the engine convention (`sqrt(3)/2` times plastic
/// strain), so its last point sits at `sqrt(3)/4` for the 0.5 range.
fn parse_law76(
    card: &DeckCard,
    functs: &BTreeMap<i64, Vec<[f64; 2]>>,
) -> Result<(FoamLaw, f64), String> {
    let density = scalar(
        card_row(card, 0, "density")?
            .first()
            .map(String::as_str)
            .unwrap_or(""),
        "LAW76 density",
    )?;
    let moduli = card_row(card, 1, "moduli")?;
    if moduli.len() != 2 {
        return Err("/MAT/LAW76 moduli row needs E and nu".to_string());
    }
    let young = scalar(&moduli[0], "LAW76 E")?;
    let poisson = scalar(&moduli[1], "LAW76 nu")?;
    let refs = card_row(card, 2, "funct refs")?;
    if refs.len() != 3 {
        return Err("/MAT/LAW76 funct row needs three curve ids".to_string());
    }
    let curve_id = [
        integer(&refs[0], "LAW76 tension curve")?,
        integer(&refs[1], "LAW76 compression curve")?,
        integer(&refs[2], "LAW76 shear curve")?,
    ];
    if curve_id != [2, 3, 4] {
        return Err(format!(
            "foam arm needs LAW76 curves /FUNCT/2,3,4, found {curve_id:?}"
        ));
    }
    for (row, what) in [(3, "scale"), (5, "rate"), (6, "flag spill")] {
        for token in card_row(card, row, what)? {
            let value = scalar(token, &format!("LAW76 {what}"))?;
            let want = if row == 3 { 1.0 } else { 0.0 };
            if value != want {
                return Err(format!("foam arm needs LAW76 {what} {want}, found {value}"));
            }
        }
    }
    let plastic = card_row(card, 4, "plastic poisson")?;
    if plastic.len() != 5 {
        return Err("/MAT/LAW76 plastic row needs five tokens".to_string());
    }
    if scalar(&plastic[0], "LAW76 nu_p")? != 0.5 {
        return Err("foam arm needs isochoric LAW76 (nu_p 0.5)".to_string());
    }
    for token in &plastic[1..] {
        if scalar(token, "LAW76 plastic")? != 0.0 {
            return Err("foam arm needs zero LAW76 plastic extras".to_string());
        }
    }
    let flags = card_row(card, 7, "formulation flags")?;
    if flags.len() != 3 {
        return Err("/MAT/LAW76 flag row needs IFORM/IQUAD/ICONV".to_string());
    }
    let iform = integer(&flags[0], "LAW76 IFORM")?;
    let iquad = integer(&flags[1], "LAW76 IQUAD")?;
    let iconv = integer(&flags[2], "LAW76 ICONV")?;
    if (iform, iquad, iconv) != (0, 1, 1) {
        return Err(format!(
            "foam arm needs quadratic isochoric LAW76 (0/1/1), found {iform}/{iquad}/{iconv}"
        ));
    }
    let curves: Vec<&Vec<[f64; 2]>> = curve_id
        .iter()
        .map(|id| {
            functs
                .get(id)
                .ok_or_else(|| format!("LAW76 curve /FUNCT/{id} is missing"))
        })
        .collect::<Result<_, _>>()?;
    for (curve, id) in curves.iter().zip(curve_id) {
        if curve.len() != 2 || curve[0][0] != 0.0 {
            return Err(format!("/FUNCT/{id} must start at zero with two points"));
        }
    }
    let strengths = [curves[0][0][1], curves[1][0][1], curves[2][0][1]];
    if strengths
        .iter()
        .any(|strength| !strength.is_finite() || *strength <= 0.0)
    {
        return Err("LAW76 initial strengths must be positive".to_string());
    }
    let hardening_x: Vec<f64> = curves[0].iter().map(|point| point[0]).collect();
    let hardening_y: Vec<f64> = curves[0]
        .iter()
        .map(|point| point[1] / strengths[0])
        .collect();
    for (curve, strength, id) in curves
        .iter()
        .zip(strengths)
        .zip([2, 3, 4])
        .map(|((c, s), i)| (c, s, i))
    {
        for (point, (x, y)) in curve.iter().zip(hardening_x.iter().zip(hardening_y.iter())) {
            let expected_x = if id == 4 {
                x * 3.0_f64.sqrt() / 2.0
            } else {
                *x
            };
            if (point[0] - expected_x).abs() > 1e-12 * expected_x.abs().max(1.0)
                || (point[1] / strength - y).abs() > 1e-12
            {
                return Err(format!("/FUNCT/{id} hardening disagrees with /FUNCT/2"));
            }
        }
    }
    if young <= 0.0 || density <= 0.0 {
        return Err("LAW76 E and density must be positive".to_string());
    }
    let hardening = Funct1D::new(hardening_x, hardening_y)?;
    Ok((
        FoamLaw::Law76 {
            young,
            poisson,
            tension: strengths[0],
            compression: strengths[1],
            shear: strengths[2],
            hardening,
        },
        density,
    ))
}

/// Parses the plain `/MAT/FOAM_PLAS` LAW33 card.
///
/// Row layout mirrors `scripts/generate_foam33_deck.py`: density plus
/// zero; E plus two zeros (Ka = 0, no tabulated/rate multipliers) plus
/// unit scale; trapped-air P0/porosity/gamma0; A/B/C yield constants.
fn parse_foam_plas(card: &DeckCard) -> Result<(FoamLaw, f64), String> {
    let row0 = card_row(card, 0, "density")?;
    if row0.len() != 2 {
        return Err("/MAT/FOAM_PLAS density row needs two tokens".to_string());
    }
    let density = scalar(&row0[0], "LAW33 density")?;
    if scalar(&row0[1], "LAW33 density extra")? != 0.0 {
        return Err("foam arm needs zero LAW33 density extra".to_string());
    }
    let row1 = card_row(card, 1, "moduli")?;
    if row1.len() != 4 {
        return Err("/MAT/FOAM_PLAS moduli row needs four tokens".to_string());
    }
    let young = scalar(&row1[0], "LAW33 E")?;
    if scalar(&row1[1], "LAW33 Ka")? != 0.0 || scalar(&row1[2], "LAW33 multiplier")? != 0.0 {
        return Err("foam arm needs plain LAW33 (Ka 0, no multipliers)".to_string());
    }
    if scalar(&row1[3], "LAW33 scale")? != 1.0 {
        return Err("foam arm needs unit LAW33 scale".to_string());
    }
    let row2 = card_row(card, 2, "trapped air")?;
    if row2.len() != 3 {
        return Err("/MAT/FOAM_PLAS air row needs P0/porosity/gamma0".to_string());
    }
    let pressure0 = scalar(&row2[0], "LAW33 P0")?;
    let porosity = scalar(&row2[1], "LAW33 porosity")?;
    let gamma0 = scalar(&row2[2], "LAW33 gamma0")?;
    let row3 = card_row(card, 3, "yield")?;
    if row3.len() != 3 {
        return Err("/MAT/FOAM_PLAS yield row needs A/B/C".to_string());
    }
    let a = scalar(&row3[0], "LAW33 A")?;
    let b = scalar(&row3[1], "LAW33 B")?;
    let c = scalar(&row3[2], "LAW33 C")?;
    if young <= 0.0 || density <= 0.0 {
        return Err("LAW33 E and density must be positive".to_string());
    }
    Ok((
        FoamLaw::Law33 {
            young,
            a,
            b,
            c,
            pressure0,
            porosity,
            gamma0,
        },
        density,
    ))
}

/// Reads the engine `/RUN` stop time from raw deck text.
///
/// The engine file carries a bare `/DT/BRICK` card the typed parser
/// rejects, so only the stop line is consumed; every other engine card
/// (output cadence, timestep scale, print stride) is engine-side only
/// and the replay carries its own fixed discretization instead.
fn engine_stop(text: &str) -> Result<f64, String> {
    let lines: Vec<&str> = text.lines().map(str::trim).collect();
    for (index, line) in lines.iter().enumerate() {
        if line.starts_with("/RUN") {
            for following in lines.iter().skip(index + 1) {
                if following.is_empty() || following.starts_with('#') || following.starts_with('/')
                {
                    continue;
                }
                let token = following
                    .split_whitespace()
                    .next()
                    .ok_or_else(|| "engine /RUN block holds no stop time".to_string())?;
                let stop = scalar(token, "RUN stop")?;
                if stop > 0.0 {
                    return Ok(stop);
                }
                return Err("engine deck holds no positive /RUN stop time".to_string());
            }
        }
    }
    Err("engine deck holds no /RUN block".to_string())
}

/// Builds the native foam model from a C3 deck pair.
///
/// Verifies the full prescribed-motion fixture: one 8-node brick, two
/// node groups on the extreme faces, `/BCS` pinning (111 fixed face,
/// 011 transverse on the driven face), one axial `/IMPDISP` on the
/// driven group with unit scales, full-integration property, and a
/// load path covering `[0, stop]`. Any deviation fails closed: the
/// replay cannot represent unprescribed motion or rate effects.
fn parse_foam_model(decks: &C3Decks) -> Result<FoamModel, String> {
    let starter_text = std::fs::read_to_string(&decks.starter)
        .map_err(|error| format!("cannot read {}: {error}", decks.starter.display()))?;
    if !super::has_si_units(&starter_text) {
        return Err(format!(
            "sonium foam arm needs SI (kg/m/s) C3 decks: {} carries other units",
            decks.starter.display()
        ));
    }
    let starter = RadDeck::parse_file(&decks.starter).map_err(|error| error.to_string())?;
    for card in &starter.cards {
        match card.kind {
            CardKind::Begin
            | CardKind::End
            | CardKind::Node
            | CardKind::Part
            | CardKind::Brick
            | CardKind::PropSolid
            | CardKind::MatLaw76
            | CardKind::MatFoamPlas
            | CardKind::Funct
            | CardKind::GrnodNode
            | CardKind::Bcs
            | CardKind::Impdisp
            | CardKind::ThBric
            | CardKind::ThBrick
            | CardKind::ThNode => {}
            _ => {
                return Err(format!(
                    "sonium foam arm supports no {} card",
                    card.kind.as_str()
                ));
            }
        }
    }
    let node_card = one_card(&starter, CardKind::Node, "/NODE")?;
    let TypedCard::Nodes(nodes) = node_card
        .typed
        .as_ref()
        .ok_or_else(|| format!("{} /NODE block undecodable", decks.starter.display()))?
    else {
        return Err(format!(
            "{} /NODE block undecodable",
            decks.starter.display()
        ));
    };
    if nodes.len() != 8 {
        return Err(format!("foam arm needs 8 nodes, found {}", nodes.len()));
    }
    let mut coords = BTreeMap::new();
    for node in nodes {
        if !node.x.is_finite() || !node.y.is_finite() || !node.z.is_finite() {
            return Err(format!("node {} carries non-finite coords", node.id));
        }
        coords.insert(node.id, [node.x, node.y, node.z]);
    }
    let brick_card = one_card(&starter, CardKind::Brick, "/BRICK")?;
    let TypedCard::Elems(block) = brick_card
        .typed
        .as_ref()
        .ok_or_else(|| format!("{} /BRICK block undecodable", decks.starter.display()))?
    else {
        return Err(format!(
            "{} /BRICK block undecodable",
            decks.starter.display()
        ));
    };
    if block.etype != ElemType::Brick || block.elems.len() != 1 {
        return Err("foam arm needs one BRICK element".to_string());
    }
    if block.elems[0].nodes.len() != 8 {
        return Err("foam arm needs one 8-node brick".to_string());
    }
    let mut referenced = std::collections::BTreeSet::new();
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for id in &block.elems[0].nodes {
        let xyz = coords
            .get(id)
            .copied()
            .ok_or_else(|| format!("brick references missing node {id}"))?;
        referenced.insert(*id);
        for axis in 0..3 {
            min[axis] = min[axis].min(xyz[axis]);
            max[axis] = max[axis].max(xyz[axis]);
        }
    }
    let extent = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    if extent.iter().any(|edge| !edge.is_finite() || *edge <= 0.0) {
        return Err("foam brick has a non-positive edge".to_string());
    }
    let side = extent[0];
    for edge in &extent[1..] {
        if (edge - side).abs() > 1e-9 * side {
            return Err("foam arm needs a cube brick".to_string());
        }
    }
    let prop = one_card(&starter, CardKind::PropSolid, "/PROP/SOLID")?;
    let isolid = integer(
        card_row(prop, 0, "formulation")?
            .first()
            .map(String::as_str)
            .unwrap_or(""),
        "PROP Isolid",
    )?;
    if isolid != 18 {
        return Err(format!(
            "foam arm needs full-integration Isolid 18, found {isolid}"
        ));
    }
    let mut groups = BTreeMap::new();
    for card in &starter.cards {
        if card.kind != CardKind::GrnodNode {
            continue;
        }
        let id = card
            .id
            .ok_or_else(|| "/GRNOD/NODE block misses its group id".to_string())?;
        groups.insert(id, group_ids(card)?);
    }
    if groups.len() != 2 {
        return Err(format!(
            "foam arm needs two node groups, found {}",
            groups.len()
        ));
    }
    let mut functs = BTreeMap::new();
    for card in &starter.cards {
        if card.kind != CardKind::Funct {
            continue;
        }
        let id = card
            .id
            .ok_or_else(|| "/FUNCT block misses its id".to_string())?;
        let TypedCard::Funct(funct) = card.typed.as_ref().ok_or_else(|| {
            format!("/FUNCT {id} undecodable (need strictly increasing X Y pairs)")
        })?
        else {
            return Err(format!(
                "/FUNCT {id} undecodable (need strictly increasing X Y pairs)"
            ));
        };
        functs.insert(id, funct.points.clone());
    }
    let path = functs
        .get(&1)
        .ok_or_else(|| "foam arm needs load path /FUNCT/1".to_string())?
        .clone();
    if path.len() < 2 || path[0] != [0.0, 0.0] {
        return Err("/FUNCT/1 must start at (0, 0) with two points".to_string());
    }
    let mut fixed = None;
    let mut driven = None;
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
        let group = bcs
            .grnod
            .ok_or_else(|| "/BCS block references no group".to_string())?;
        if !groups.contains_key(&group) {
            return Err(format!("/BCS references missing group {group}"));
        }
        match bcs.tra.as_str() {
            "111" => {
                if fixed.replace(group).is_some() {
                    return Err("foam arm needs one fixed face".to_string());
                }
            }
            "011" => {
                if driven.replace(group).is_some() {
                    return Err("foam arm needs one driven face".to_string());
                }
            }
            code => return Err(format!("/BCS code {code} is not 111 or 011")),
        }
    }
    let (fixed, driven) = fixed
        .zip(driven)
        .ok_or_else(|| "foam arm needs 111 and 011 /BCS blocks".to_string())?;
    for (group, face, what) in [(fixed, min[0], "fixed"), (driven, max[0], "driven")] {
        let members = groups.get(&group).cloned().unwrap_or_default();
        if members.len() != 4 {
            return Err(format!("foam {what} face needs four nodes"));
        }
        for id in members {
            let xyz = coords
                .get(&id)
                .copied()
                .ok_or_else(|| format!("group {group} references missing node {id}"))?;
            if (xyz[0] - face).abs() > 1e-12 * side {
                return Err(format!("foam {what} face leaves its plane"));
            }
        }
    }
    let imp = one_card(&starter, CardKind::Impdisp, "/IMPDISP")?;
    let TypedCard::Load(LoadCard::Impdisp(imp)) = imp
        .typed
        .as_ref()
        .ok_or_else(|| "/IMPDISP block undecodable".to_string())?
    else {
        return Err("/IMPDISP block undecodable".to_string());
    };
    if imp.dir != "X" || imp.funct != 1 || imp.grnod != driven || imp.skew != 0 || imp.sens != 0 {
        return Err("foam arm needs axial /IMPDISP on the driven group via /FUNCT/1".to_string());
    }
    for (scale, what) in [(imp.ax, "abscissa"), (imp.fy, "ordinate")] {
        if scale.is_some_and(|value| value != 1.0) {
            return Err(format!("foam arm needs unit IMPDISP {what} scale"));
        }
    }
    let mut law76 = Vec::new();
    let mut law33 = Vec::new();
    for card in &starter.cards {
        match card.kind {
            CardKind::MatLaw76 => law76.push(card),
            CardKind::MatFoamPlas => law33.push(card),
            _ => {}
        }
    }
    if law76.len() + law33.len() != 1 {
        return Err("foam arm needs one /MAT block (LAW76 or FOAM_PLAS)".to_string());
    }
    let (law, _density) = if let [card] = law76.as_slice() {
        parse_law76(card, &functs)?
    } else if let [card] = law33.as_slice() {
        parse_foam_plas(card)?
    } else {
        return Err("foam arm needs one /MAT block (LAW76 or FOAM_PLAS)".to_string());
    };
    let engine_text = std::fs::read_to_string(&decks.engine)
        .map_err(|error| format!("cannot read {}: {error}", decks.engine.display()))?;
    let stop = engine_stop(&engine_text)?;
    let end = path.last().copied().unwrap_or([0.0, 0.0]);
    if (end[0] - stop).abs() > 1e-9 * stop {
        return Err("load path must end at the engine stop time".to_string());
    }
    let dof = u64::try_from(referenced.len())
        .ok()
        .and_then(|nodes| nodes.checked_mul(3))
        .ok_or_else(|| "node count overflows dof".to_string())?;
    Ok(FoamModel {
        side_m: side,
        volume_m3: side * side * side,
        path,
        stop_s: stop,
        dof,
        law,
    })
}

/// Replays the deck path through a LAW76 point, booking work.
fn replay_law76(
    model: &FoamModel,
    young: f64,
    poisson: f64,
    tension: f64,
    compression: f64,
    shear: f64,
    hardening: &Funct1D,
) -> Result<FoamResult, String> {
    let law = Samp76Quadratic::new(
        young,
        poisson,
        tension,
        compression,
        shear,
        hardening.clone(),
    )?;
    let mut state = Samp76QuadraticState::default();
    let mut work = 0.0;
    let mut substeps = 0u64;
    for segment in model.path.windows(2) {
        let (left, right) = (segment[0], segment[1]);
        let span = right[0] - left[0];
        if span <= 0.0 {
            return Err("/FUNCT/1 times must increase strictly".to_string());
        }
        let count = (span * 1000.0 * SUBSTEPS_PER_MS).round() as u64;
        if count == 0 {
            return Err("/FUNCT/1 segment is too short to substep".to_string());
        }
        for step in 1..=count {
            let fraction = step as f64 / count as f64;
            let previous = (step - 1) as f64 / count as f64;
            let j0 = 1.0 + (left[1] + (right[1] - left[1]) * previous) / model.side_m;
            let j1 = 1.0 + (left[1] + (right[1] - left[1]) * fraction) / model.side_m;
            if j0 <= 0.0 || j1 <= 0.0 {
                return Err("load path inverts the cube".to_string());
            }
            let old = state.stress[0];
            let update = law.step(&mut state, &[(j1 / j0).ln(), 0.0, 0.0, 0.0, 0.0, 0.0])?;
            // Fixed transverse dimensions: sigma A dx = sigma V0 dJ.
            work += 0.5 * (old + update.stress[0]) * model.volume_m3 * (j1 - j0);
            substeps += 1;
        }
    }
    if !work.is_finite() || state.stress.iter().any(|value| !value.is_finite()) {
        return Err("LAW76 replay leaves finite states".to_string());
    }
    Ok(FoamResult {
        work_j: work,
        sxx_pa: state.stress[0],
        ep: Some(state.plastic_strain),
        dx_m: model.path.last().copied().unwrap_or([0.0, 0.0])[1],
        substeps,
    })
}

/// Replays the deck path through a LAW33 point, booking work.
fn replay_law33(
    model: &FoamModel,
    young: f64,
    a: f64,
    b: f64,
    c: f64,
    pressure0: f64,
    porosity: f64,
    gamma0: f64,
) -> Result<FoamResult, String> {
    let law = FoamPlas33::new(young, a, b, c, pressure0, porosity, gamma0)?;
    let mut state = FoamPlasState::default();
    let mut work = 0.0;
    let mut substeps = 0u64;
    for segment in model.path.windows(2) {
        let (left, right) = (segment[0], segment[1]);
        let span = right[0] - left[0];
        if span <= 0.0 {
            return Err("/FUNCT/1 times must increase strictly".to_string());
        }
        let count = (span * 1000.0 * SUBSTEPS_PER_MS).round() as u64;
        if count == 0 {
            return Err("/FUNCT/1 segment is too short to substep".to_string());
        }
        for step in 1..=count {
            let fraction = step as f64 / count as f64;
            let previous = (step - 1) as f64 / count as f64;
            let j0 = 1.0 + (left[1] + (right[1] - left[1]) * previous) / model.side_m;
            let j1 = 1.0 + (left[1] + (right[1] - left[1]) * fraction) / model.side_m;
            if j0 <= 0.0 || j1 <= 0.0 {
                return Err("load path inverts the cube".to_string());
            }
            let old = state.stress[0];
            let update = law.step(
                &mut state,
                &FoamPlasIncrement {
                    strain: [(j1 / j0).ln(), 0.0, 0.0, 0.0, 0.0, 0.0],
                    volume_ratio: j1,
                },
            )?;
            work += 0.5 * (old + update.stress[0]) * model.volume_m3 * (j1 - j0);
            substeps += 1;
        }
    }
    if !work.is_finite() || state.stress.iter().any(|value| !value.is_finite()) {
        return Err("LAW33 replay leaves finite states".to_string());
    }
    Ok(FoamResult {
        work_j: work,
        sxx_pa: state.stress[0],
        ep: None,
        dx_m: model.path.last().copied().unwrap_or([0.0, 0.0])[1],
        substeps,
    })
}

/// Runs one untimed replay of the parsed fixture.
fn replay(model: &FoamModel) -> Result<FoamResult, String> {
    match &model.law {
        FoamLaw::Law76 {
            young,
            poisson,
            tension,
            compression,
            shear,
            hardening,
        } => replay_law76(
            model,
            *young,
            *poisson,
            *tension,
            *compression,
            *shear,
            hardening,
        ),
        FoamLaw::Law33 {
            young,
            a,
            b,
            c,
            pressure0,
            porosity,
            gamma0,
        } => replay_law33(model, *young, *a, *b, *c, *pressure0, *porosity, *gamma0),
    }
}

/// Checks C3 probes against replay observations.
///
/// Channels match the engine TH names after whitespace collapsing
/// (the converter pads brick/node titles). Both energy probes read the
/// booked boundary work: at the final zero-motion hold the kinetic
/// energy is zero and archived INTERNAL and EXTERNAL WORK agree to
/// 4e-6 relative.
fn check_foam_probes(probes: &[Probe], result: &FoamResult) -> Vec<String> {
    let mut failures = Vec::new();
    for probe in probes {
        let collapsed = probe
            .channel
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        let sampled = match collapsed.as_str() {
            "INTERNAL ENERGY" | "EXTERNAL WORK" => Ok(result.work_j),
            "cube 1 cube var 25" => Ok(result.sxx_pa),
            "cube 1 cube var 33" => result.ep.ok_or_else(|| {
                format!(
                    "probe {} needs an ep replay this law withholds",
                    probe.label
                )
            }),
            "face 2 moving_node var 35" => Ok(result.dx_m),
            _ => Err(format!("probe channel {} unsupported", probe.channel)),
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
            Err(error) => failures.push(error),
        }
    }
    failures
}

/// Runs one timed foam repeat: replay wall plus probe verdict.
fn foam_repeat(model: &FoamModel, probes: &[Probe]) -> Result<RepeatSample, String> {
    let start = std::time::Instant::now();
    let result = replay(model)?;
    let wall_s = start.elapsed().as_secs_f64();
    if wall_s <= 0.0 {
        return Err("foam replay measured non-positive wall time".to_string());
    }
    let failures = check_foam_probes(probes, &result);
    for failure in &failures {
        println!("FAIL: {failure}");
    }
    let rss = super::self_peak_rss_bytes();
    println!(
        "sonium foam: wall {wall_s:.3}s, cycles {}, dof {}, rss {rss} bytes, correct {}",
        result.substeps,
        model.dof,
        failures.is_empty()
    );
    Ok(RepeatSample {
        wall_s,
        cycles: result.substeps,
        dof: model.dof,
        peak_rss_bytes: rss,
        correct: failures.is_empty(),
    })
}

/// Runs the sonium arm for one foam C3 deck.
///
/// Parses once, replays warmups untimed, then times repeats and emits
/// the native row. The replay is serial by construction (each substep
/// commits state the next one reads); the thread setting is recorded
/// on the row so the matrix stays explicit.
pub(super) fn run_arm(
    deck: &str,
    threads: usize,
    repeats: u32,
    warmup: u32,
) -> Result<BenchRow, String> {
    // Safety: no other thread reads the environment concurrently at
    // this point (single-threaded startup before any spawn).
    unsafe {
        std::env::set_var("RAYON_NUM_THREADS", threads.max(1).to_string());
    }
    let decks = resolve_c3_decks(deck)?;
    let probes = read_probe_file(&decks.probes)?;
    let model = parse_foam_model(&decks)?;
    println!(
        "sonium foam mesh: 8 nodes, 1 brick, law {:?}, path {} corners to {:.3}s",
        match model.law {
            FoamLaw::Law76 { .. } => "LAW76",
            FoamLaw::Law33 { .. } => "LAW33",
        },
        model.path.len(),
        model.stop_s
    );
    for warmup_index in 0..warmup {
        println!("sonium {deck}: warmup {}/{warmup}", warmup_index + 1);
        let _ = foam_repeat(&model, &probes)?;
    }
    let mut samples = Vec::new();
    for repeat in 0..repeats {
        println!("sonium {deck}: repeat {}/{repeats}", repeat + 1);
        samples.push(foam_repeat(&model, &probes)?);
    }
    let total = aggregate_repeats(&samples)?;
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
