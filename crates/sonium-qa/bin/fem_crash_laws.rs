//! Phase-3 lane 3B G-leg (lane-owned): crash metal laws.
//!
//! Native-vs-harvested-golden comparison for LOIS/LOI{2,22,27,44,62,68}.
//! Offline: reads only `goldens/miniqa/**/*.json` plus the schema rules.
//! Exit 0 = PASS, exit 1 = FAIL with per-probe report on stdout.
//!
//! Probe design (single-point kinematic replay; no cross-lane runtimes):
//! deck geometry and material cards are transcribed at author time from
//! the vendor miniqa decks (reference-only), while every runtime input
//! comes from the harvested goldens. All replays run in deck units, so
//! no unit conversion enters any comparison.
//!
//! - L2a replays the LOI2 truss element: logarithmic strain from nodal
//!   DY over L0 = 200 drives the transcribed LAW2 (E = 210000, nu = 0,
//!   JC a = 20, b = 210, n = 1; deck SIG_MAX0/EPS_MAX = 0 means
//!   inactive). The replay drives rate-free (deck c = 1, ref = 20
//!   give factor 1 on this path). Compares axial stress (F/A), total
//!   internal energy, and plastic strain within golden tol or a 0.2%
//!   formulation band: the golden sags below its own F/A = a + b ep
//!   line increasingly (engine cycle integration), which no
//!   quasi-static replay reproduces point by point.
//! - L2b checks the beam/truss part-mass identity MASS(t) = rho A L(t)
//!   from nodal motion (rho = 8e-10, A = 400). L2c checks total mass
//!   and the energy ledger.
//! - L22a replays the LOI22 LAW22 brick uniaxially along the golden
//!   axial log path (lateral solve for free sides) with the
//!   transcribed card (E = 70000, JC 100/0/1, cap 100 fixed,
//!   Eps_dam = 0.1, Et = -2000, rupture 0.2). Compares SZ, energy
//!   density, PLAS, and lateral strain in golden-only regimes: ring
//!   mask (informational), pre-damage (golden tol), softening (bands:
//!   6.0 stress, 2.5% ep, 1% energy; softening amplifies onset
//!   seeds, gain 1 + |Et|/E_dam), freeze-transition (informational
//!   timing slop), frozen (bands: 1.0 stress, 0.5% ep, 1% energy).
//!   Energy rebases at each clean segment start so masked-burst work
//!   never leaks into judged increments. The LAW23 twin brick is a
//!   different material (outside lane scope) and is not replayed.
//! - L22b rebuilds the 5400 mass units from both brick volumes.
//!   L22c checks the energy ledger.
//! - L27a checks the part-IE sum against global internal energy.
//!   L27b checks mass conservation plus the ERODED verdicts. L27c/d
//!   drive both transcribed PLAS_BRIT cards uniaxially to their deck
//!   eps_f and require the native rupture latch (qualitative verdict
//!   consistency with the golden erosion; the poke kinematics that
//!   select which elements erode are not single-point replayable).
//! - L44a replays the LOI44 brick uniaxially along the golden axial log
//!   path (lateral solve for free sides) with the transcribed LAW44
//!   (E = 2.05e5, a = 180, b = 450, n = 0.5, chard = 1, sigmax = 300
//!   scaled, c = 400, p = 5, total rate) at the golden step times.
//!   Compares SZ, energy density, PLAS, and lateral strain. Stamps
//!   under the per-stamp oscillation mask (detrended golden residual
//!   above channel tol with live signal) are informational only;
//!   judged stamps must meet golden tol. L44b rebuilds the 7800 mass.
//!   L44c checks the ledger.
//! - L62a replays the LOI62 shear test: engineering shear from top DX
//!   over height 70 drives the transcribed LAW62 (nu = 0 fully
//!   compressible, two Ogden pairs, four Prony branches) at the golden
//!   step times. Compares rotation-invariant response (von Mises,
//!   pressure, energy density) at a documented 5%-of-peak formulation
//!   band: the deck runs total-Lagrange finite rotation (Ismstr = 10)
//!   beyond this lane's small-strain C1 scope, so raw components are
//!   not compared. Exact data checks cover OFF, PLAS = 0, TEMP = 0,
//!   and constant DENS. L62b rebuilds the mass. L62c checks the ledger.
//! - L68a checks the energy ledger, L68b mass constancy, L68c the
//!   impactor constant-velocity motion (DZ = t). LAW68 replay belongs
//!   to lane 3C (noted, not probed).
//!
//! Tolerance policy: series probes use each channel's own golden
//! abs/rel tolerance with the vendor near-zero skip (SCHEMA.md),
//! except L62a (formulation band above) and the ledgers. L2c/L22c/L44c
//! balance at 1% relative; L62c/L68a use a 5% band with worst/end
//! reported: the shear/impact work sums carry transient explicit
//! integration noise (1.05%/2.56% worst mid-test) while end-states
//! balance to 0.06%/0.93% with all other energy channels zero.

// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sonium_fem::physics::solid::plasticity::StressUpdateModel;
use sonium_fem::physics::solid::plasticity::crash::drive_uniaxial_stress;
use sonium_fem::physics::solid::plasticity::law_metals::Law2;
use sonium_fem::physics::solid::plasticity::law_metals::{
    BrittleDir, Law2Flow, Law2State, Law22, Law22State, Law27, Law27State, Law44, Law44Rate,
    Law44State, Law62, Law62State, MaxRateEffect,
};
use std::path::PathBuf;

/// Golden schema accepted by this leg (see `goldens/SCHEMA.md`).
const GOLDEN_SCHEMA: &str = "opencourant-golden-v1";

/// Vendor near-zero skip from SCHEMA.md (`Zero_tolerance`).
const ZERO_TOLERANCE: f64 = 5e-14;

/// Energy-ledger relative band (documented in the header).
const LEDGER_REL: f64 = 0.01;

/// Loose ledger band for shear/impact tests (documented in the header).
/// Measured worst ledger deviation is 2.56e-2 (LOI68); gate at ~1.5x.
const LEDGER_LOOSE_REL: f64 = 0.04;

/// LOI62 formulation band, fraction of series peak (documented above).
/// Measured worst judged ratio is 1.6e-2; gate at ~2x.
const FORMULATION_REL: f64 = 0.03;

/// LOI2 truss formulation band, relative (documented in the header).
const TRUSS_BAND_REL: f64 = 2e-3;

/// LOI22 softening-regime bands (documented in the header): stress
/// absolute floor plus relative bands for plastic strain and energy.
const SOFTEN_SZ_ABS: f64 = 6.0;
const SOFTEN_EP_REL: f64 = 2.5e-2;
const SOFTEN_IE_REL: f64 = 1e-2;

/// LOI22 frozen-regime bands (documented in the header).
const FROZEN_SZ_ABS: f64 = 1.0;
const FROZEN_EP_REL: f64 = 5e-3;
const FROZEN_IE_REL: f64 = 1e-2;

fn goldens_root() -> PathBuf {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        return PathBuf::from(&args[1]);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("goldens")
}

fn load_golden(root: &std::path::Path, case: &str) -> Value {
    let path = root.join("miniqa").join(format!("{case}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("G-leg needs harvested golden {path:?}: {err}"));
    let doc: Value = serde_json::from_str(&text).expect("golden must be valid JSON");
    assert_eq!(
        doc["schema"].as_str().unwrap_or(""),
        GOLDEN_SCHEMA,
        "golden schema mismatch for {case}"
    );
    assert_eq!(
        doc["meta"]["normal_termination"].as_bool(),
        Some(true),
        "golden {case} did not terminate normally"
    );
    doc
}

fn time_grid(doc: &Value) -> Vec<f64> {
    doc["time"]
        .as_array()
        .expect("golden needs a time grid")
        .iter()
        .map(channel_number)
        .collect()
}

fn channel_number(value: &Value) -> f64 {
    value.as_f64().expect("golden samples must be numbers")
}

fn channel(doc: &Value, name: &str) -> Vec<f64> {
    doc["channels"][name]["values"]
        .as_array()
        .unwrap_or_else(|| panic!("golden lacks channel {name}"))
        .iter()
        .map(channel_number)
        .collect()
}

fn channel_tol(doc: &Value, name: &str) -> (f64, f64) {
    let abs = doc["channels"][name]["abs_tol"].as_f64().unwrap_or(1e-12);
    let rel = doc["channels"][name]["rel_tol"].as_f64().unwrap_or(6.6e-4);
    (abs, rel)
}

/// Sample comparison with the vendor near-zero skip (SCHEMA.md).
fn within_tol(actual: f64, expected: f64, abs: f64, rel: f64) -> bool {
    if !actual.is_finite() || !expected.is_finite() {
        return false;
    }
    if actual.abs() < ZERO_TOLERANCE && expected.abs() < ZERO_TOLERANCE {
        return true;
    }
    if (actual - expected).abs() < 2.0 * ZERO_TOLERANCE {
        return true;
    }
    (actual - expected).abs() <= abs + rel * expected.abs()
}

/// Sample comparison with an extra relative formulation band.
fn within_band(actual: f64, expected: f64, abs: f64, rel: f64, band: f64) -> bool {
    within_tol(actual, expected, abs, rel)
        || (actual - expected).abs() <= band * expected.abs().max(1e-30)
}

/// Worst relative deviation of two series (for reporting).
fn max_rel_dev(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected.iter())
        .map(|(&a, &e)| {
            if e.abs() < ZERO_TOLERANCE {
                (a - e).abs()
            } else {
                (a - e).abs() / e.abs()
            }
        })
        .fold(0.0, f64::max)
}

/// Von Mises stress of a C1 tensor.
fn von_mises(s: &[f64; 6]) -> f64 {
    let d0 = s[0] - s[1];
    let d1 = s[1] - s[2];
    let d2 = s[2] - s[0];
    let shear = 6.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]);
    (0.5 * (d0 * d0 + d1 * d1 + d2 * d2 + shear)).sqrt()
}

/// Logarithmic axial strain from a nodal displacement over length.
fn log_strain(disp: f64, length: f64) -> f64 {
    (1.0 + disp / length).ln()
}

/// Per-stamp oscillation mask of a golden series (ringing bursts).
///
/// Detrended local oscillation: max residual of the series against the
/// chord over a ±3-stamp window exceeding half the channel tolerance
/// (detection margin), gated on a live window peak (bursts swinging
/// through zero still mask; strengthless tails carry only harvest
/// noise, never ringing). Masked stamps are informational only
/// (elastic ringing aliases at stamp rate and no constitutive law
/// reproduces it point by point); clean stamps carry the verdict.
fn ring_mask(series: &[f64], abs_tol: f64, rel_tol: f64) -> Vec<bool> {
    let mut mask = vec![false; series.len()];
    if series.len() < 2 {
        return mask;
    }
    let peak = series.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    let live = 0.01 * peak;
    for i in 0..series.len() {
        let lo = i.saturating_sub(3);
        let hi = (i + 3).min(series.len() - 1);
        let span = (hi - lo).max(1) as f64;
        let mut worst = 0.0f64;
        let mut win_peak = 0.0f64;
        for j in lo..=hi {
            let chord = series[lo] + (series[hi] - series[lo]) * ((j - lo) as f64) / span;
            worst = worst.max((series[j] - chord).abs());
            win_peak = win_peak.max(series[j].abs());
        }
        if win_peak <= live {
            continue;
        }
        mask[i] = worst > 0.5 * (abs_tol + rel_tol * series[i].abs());
    }
    mask
}

/// One uniaxial-stress substep with a lateral solve.
///
/// Solves the transverse-isotropic lateral strain for zero transverse
/// stress by numerical Newton (25 passes from the previous lateral
/// increment), commits, and returns the full stress plus the lateral
/// increment. Axial is XX (isotropic laws are axis-free). `dt`
/// selects the rated path; `None` drives rate-free.
fn uniaxial_substep<M>(
    law: &M,
    state: &mut M::State,
    d_axial: f64,
    dt: Option<f64>,
    guess: f64,
) -> ([f64; 6], f64)
where
    M: StressUpdateModel<StrainIncrement = [f64; 6]>,
{
    let trial = |state: &M::State, lat: f64| {
        let d_eps = [d_axial, lat, lat, 0.0, 0.0, 0.0];
        match dt {
            Some(dt) => law.update_with_rate(state, &d_eps, dt),
            None => law.update(state, &d_eps),
        }
    };
    let mut lat = guess;
    for _ in 0..25 {
        let update = trial(state, lat);
        let scale = update.stress[0].abs().max(1.0);
        if update.stress[1].abs() < 1e-9 * scale && update.stress[2].abs() < 1e-9 * scale {
            let stress = update.stress;
            law.commit(state, &update);
            return (stress, lat);
        }
        let h = 1.5e-8 * (1.0 + lat.abs());
        let ahead = trial(state, lat + h);
        let slope = (ahead.stress[1] - update.stress[1]) / h;
        if !slope.is_finite() || slope.abs() < 1e-30 {
            break;
        }
        lat -= update.stress[1] / slope;
        if !lat.is_finite() {
            break;
        }
    }
    let update = trial(state, lat);
    let stress = update.stress;
    law.commit(state, &update);
    (stress, lat)
}

fn main() {
    let root = goldens_root();
    let mut total = 0;
    let mut passed = 0;

    // LOI2 truss replay (L2a): log path, rate-free LAW2.
    total += 1;
    let loi2 = load_golden(&root, "LOIS/LOI2/BESH/input_44");
    let time2 = time_grid(&loi2);
    let dy35 = channel(&loi2, "BESHV31D00_th_nodes 35 ~2");
    let law2 = Law2::new(
        210_000.0,
        0.0,
        Law2Flow::JohnsonCook {
            a: 20.0,
            b: 210.0,
            n: 1.0,
            c: 1.0,
        },
        20.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0e30,
        MaxRateEffect::Fixed,
        1.0e30,
        0.0,
        1.0e30,
    );
    let mut s2 = Law2State::new(0.0);
    let (f_abs, f_rel) = channel_tol(&loi2, "BESHV31D00_th_Truss 1 BESHV31D00_th_Truss ~2");
    let (ie_abs, ie_rel) = channel_tol(&loi2, "BESHV31D00_th_Truss 1 BESHV31D00_th_Truss ~3");
    let (ep_abs, ep_rel) = channel_tol(&loi2, "BESHV31D00_th_Truss 1 BESHV31D00_th_Truss ~4");
    let force = channel(&loi2, "BESHV31D00_th_Truss 1 BESHV31D00_th_Truss ~2");
    let truss_ie = channel(&loi2, "BESHV31D00_th_Truss 1 BESHV31D00_th_Truss ~3");
    let truss_ep = channel(&loi2, "BESHV31D00_th_Truss 1 BESHV31D00_th_Truss ~4");
    let mut l2a = true;
    let mut energy = 0.0;
    let (mut worst_f, mut worst_ie, mut worst_ep) = (0.0f64, 0.0f64, 0.0f64);
    let mut prev_sig2 = 0.0;
    for i in 1..time2.len() {
        // Uniaxial STRESS replay (lateral free): substepped driver
        // per golden stamp along the log path (isotropic: axis free).
        // Substepping converges the coarse-stamp integration toward
        // the engine's per-cycle result; ep reads the accumulated
        // state (the driver's curve ep restarts every call).
        let step = log_strain(dy35[i], 200.0) - log_strain(dy35[i - 1], 200.0);
        // The driver's curve[0] is zeros (not the caller's state), so
        // the trapezoid restarts from the tracked previous stress.
        let prev = prev_sig2;
        let curve = drive_uniaxial_stress(&law2, &mut s2, step, 10);
        let sig = curve[10].stress_xx;
        let ep_state = s2.equiv_plastic_strain;
        // Truss internal energy is nodal work (trapezoid per substep).
        for j in 1..=10 {
            let left = if j == 1 { prev } else { curve[j - 1].stress_xx };
            energy += 0.5 * (left + curve[j].stress_xx) * (dy35[i] - dy35[i - 1]) / 200.0 / 10.0;
        }
        prev_sig2 = sig;
        let ok_f = within_band(sig, force[i] / 400.0, f_abs / 400.0, f_rel, TRUSS_BAND_REL);
        let ok_ie = within_band(
            energy * 80_000.0,
            truss_ie[i],
            ie_abs,
            ie_rel,
            TRUSS_BAND_REL,
        );
        let ok_ep = within_band(ep_state, truss_ep[i], ep_abs, ep_rel, TRUSS_BAND_REL);
        l2a = l2a && ok_f && ok_ie && ok_ep;
        worst_f = worst_f.max(max_rel_dev(&[sig], &[force[i] / 400.0]));
        worst_ie = worst_ie.max(max_rel_dev(&[energy * 80_000.0], &[truss_ie[i]]));
        worst_ep = worst_ep.max(max_rel_dev(&[ep_state], &[truss_ep[i]]));
        if !(ok_f && ok_ie && ok_ep) && std::env::var("GLEG_DEBUG").is_ok() {
            println!(
                "L2a first fail i={i} t={} sig={sig} vs {} ie={} vs {} ep={ep_state} vs {}",
                time2[i],
                force[i] / 400.0,
                energy * 80_000.0,
                truss_ie[i],
                truss_ep[i]
            );
        }
    }
    println!(
        "PROBE L2a truss-replay: worst f/ie/ep {worst_f:.3e}/{worst_ie:.3e}/{worst_ep:.3e} -> {}",
        if l2a { "PASS" } else { "FAIL" }
    );
    passed += usize::from(l2a);

    // L2b: beam/truss part-mass identity MASS(t) = rho A L(t).
    total += 1;
    let dy25 = channel(&loi2, "BESHV31D00_th_nodes 25 ~2");
    let beam_mass = channel(&loi2, "BEAM PROPERTY FOR F1: SIDE-MBR MASS ~1");
    let truss_mass = channel(&loi2, "BEAM PROPERTY FOR F1: SIDE-MBR MASS ~2");
    let (m_abs, m_rel) = channel_tol(&loi2, "BEAM PROPERTY FOR F1: SIDE-MBR MASS ~1");
    let mut l2b = beam_mass.len() == time2.len() && truss_mass.len() == time2.len();
    for i in 0..time2.len() {
        l2b =
            l2b && within_tol(
                8e-10 * 400.0 * (200.0 + dy25[i]),
                beam_mass[i],
                m_abs,
                m_rel,
            ) && within_tol(
                8e-10 * 400.0 * (200.0 + dy35[i]),
                truss_mass[i],
                m_abs,
                m_rel,
            );
    }
    println!(
        "PROBE L2b part-mass-identity: -> {}",
        if l2b { "PASS" } else { "FAIL" }
    );
    passed += usize::from(l2b);

    // L2c: total mass plus energy ledger.
    total += 1;
    let mass = channel(&loi2, "MASS");
    let (tm_abs, tm_rel) = channel_tol(&loi2, "MASS");
    let extw = channel(&loi2, "EXTERNAL WORK");
    let gl_ie = channel(&loi2, "INTERNAL ENERGY");
    let gl_ke = channel(&loi2, "KINETIC ENERGY");
    let mut l2c = within_tol(1.92e-4, mass[0], tm_abs, tm_rel);
    for i in 0..time2.len() {
        l2c = l2c && within_tol(mass[i], 1.92e-4, tm_abs, tm_rel);
        let sink = gl_ie[i] + gl_ke[i];
        l2c = l2c && (extw[i] - sink).abs() <= LEDGER_REL * extw[i].abs().max(1.0);
    }
    println!(
        "PROBE L2c mass-and-ledger: -> {}",
        if l2c { "PASS" } else { "FAIL" }
    );
    passed += usize::from(l2c);

    // LOI22 LAW22-brick replay (L22a): uniaxial-stress replay of the
    // axial log path with a lateral solve (free sides); the lateral
    // strain output validates the damage-coupled Poisson response.
    total += 1;
    let loi22 = load_golden(&root, "LOIS/LOI22/sol2223/version_100");
    let time22 = time_grid(&loi22);
    let base22 = Law2::new(
        70_000.0,
        0.3,
        Law2Flow::JohnsonCook {
            a: 100.0,
            b: 0.0,
            n: 1.0,
            c: 0.0,
        },
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        100.0,
        MaxRateEffect::Fixed,
        0.2,
        0.0,
        1.0e30,
    );
    let law22 = Law22::new(base22, 0.1, -2000.0);
    let mut s22 = Law22State::new(0.0);
    let dx1 = channel(&loi22, "Noeuds loi 22 1 ~1");
    let dx2 = channel(&loi22, "Noeuds loi 22 2 ~1");
    let dy2 = channel(&loi22, "Noeuds loi 22 2 ~2");
    let dy3 = channel(&loi22, "Noeuds loi 22 3 ~2");
    let dz5 = channel(&loi22, "Noeuds loi 22 5 ~3");
    let sz = channel(&loi22, "TH Brick selection 1 Brique loi 22 ~4");
    let ie22 = channel(&loi22, "TH Brick selection 1 Brique loi 22 ~8");
    let pl22 = channel(&loi22, "TH Brick selection 1 Brique loi 22 ~10");
    let (z_abs, z_rel) = channel_tol(&loi22, "TH Brick selection 1 Brique loi 22 ~4");
    let (e_abs, e_rel) = channel_tol(&loi22, "TH Brick selection 1 Brique loi 22 ~8");
    let (p_abs, p_rel) = channel_tol(&loi22, "TH Brick selection 1 Brique loi 22 ~10");
    let (nx1_abs, nx1_rel) = channel_tol(&loi22, "Noeuds loi 22 1 ~1");
    let (nx2_abs, nx2_rel) = channel_tol(&loi22, "Noeuds loi 22 2 ~1");
    let (ny2_abs, ny2_rel) = channel_tol(&loi22, "Noeuds loi 22 2 ~2");
    let (ny3_abs, ny3_rel) = channel_tol(&loi22, "Noeuds loi 22 3 ~2");
    // Golden-only regime boundaries: oscillation mask plus the last
    // PLAS rise (freeze plateau after that).
    let mask22 = ring_mask(&sz, z_abs, z_rel);
    let mut frozen22 = 0;
    for i in 1..time22.len() {
        if pl22[i] > pl22[i - 1] + 1e-9 {
            frozen22 = i;
        }
    }
    let mut l22a = true;
    let mut energy22 = 0.0;
    let (mut w_sz, mut w_ie, mut w_ep, mut w_lat) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let (mut r_sz, mut r_ie, mut r_ep, mut r_lat) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut ie_nat_pre = 0.0;
    let mut ie_gold_pre = ie22[0];
    let mut pre_armed = true;
    let mut prev_axial = 0.0;
    let mut prev_sig22 = [0.0; 6];
    let mut lat_total = 0.0;
    let mut lat_guess = 0.0;
    for i in 1..time22.len() {
        let axial = log_strain(dz5[i], 100.0);
        let d_axial = (axial - prev_axial) / 8.0;
        prev_axial = axial;
        for _ in 0..8 {
            let (stress, lat) = uniaxial_substep(&law22, &mut s22, d_axial, None, lat_guess);
            lat_guess = lat;
            lat_total += lat;
            let d_full = [d_axial, lat, lat, 0.0, 0.0, 0.0];
            for k in 0..6 {
                energy22 += 0.5 * (prev_sig22[k] + stress[k]) * d_full[k];
                prev_sig22[k] = stress[k];
            }
        }
        let ex = log_strain(dx2[i] - dx1[i], 100.0);
        let ey = log_strain(dy3[i] - dy2[i], 100.0);
        let lat_band = ((nx2_abs + nx2_rel * dx2[i].abs() + nx1_abs + nx1_rel * dx1[i].abs())
            + (ny3_abs + ny3_rel * dy3[i].abs() + ny2_abs + ny2_rel * dy2[i].abs()))
            / 2.0
            / 100.0;
        // Regime from golden channels only (deck Eps_dam = 0.1, cap =
        // 100): 0 ring, 1 pre-damage, 2 softening, 3 freeze
        // transition, 4 frozen. Pre-damage energy rebases at its
        // first clean stamp (ring-phase work offset); softening and
        // frozen compare absolute energy under their bands.
        let d_gold = (1.0 - sz[i] / 100.0).clamp(0.0, 1.0);
        let regime = if mask22[i] {
            0
        } else if i > frozen22 {
            4
        } else if pl22[i] <= 0.1 {
            1
        } else if d_gold > 0.9 {
            3
        } else {
            2
        };
        if regime == 0 || regime == 3 {
            pre_armed = true;
        } else if regime == 1 && pre_armed {
            ie_nat_pre = energy22;
            ie_gold_pre = ie22[i];
            pre_armed = false;
        }
        let ok_sz = match regime {
            1 => within_tol(s22.inner.stress[0], sz[i], z_abs, z_rel),
            2 => (s22.inner.stress[0] - sz[i]).abs() <= z_abs + z_rel * sz[i].abs() + SOFTEN_SZ_ABS,
            4 => (s22.inner.stress[0] - sz[i]).abs() <= z_abs + z_rel * sz[i].abs() + FROZEN_SZ_ABS,
            _ => true,
        };
        let ok_ie = match regime {
            1 => within_tol(energy22 - ie_nat_pre, ie22[i] - ie_gold_pre, e_abs, e_rel),
            2 => within_band(energy22, ie22[i], e_abs, e_rel, SOFTEN_IE_REL),
            4 => within_band(energy22, ie22[i], e_abs, e_rel, FROZEN_IE_REL),
            _ => true,
        };
        let ok_ep = match regime {
            1 => within_tol(s22.inner.equiv_plastic_strain, pl22[i], p_abs, p_rel),
            2 => within_band(
                s22.inner.equiv_plastic_strain,
                pl22[i],
                p_abs,
                p_rel,
                SOFTEN_EP_REL,
            ),
            4 => within_band(
                s22.inner.equiv_plastic_strain,
                pl22[i],
                p_abs,
                p_rel,
                FROZEN_EP_REL,
            ),
            _ => true,
        };
        let oks = [
            ok_sz,
            ok_ie,
            ok_ep,
            regime == 0
                || regime == 3
                || (lat_total - ex).abs() <= lat_band.max(1e-9) + 1e-6 * ex.abs().max(1.0),
            regime == 0
                || regime == 3
                || (lat_total - ey).abs() <= lat_band.max(1e-9) + 1e-6 * ey.abs().max(1.0),
        ];
        let judged = regime == 1 || regime == 2 || regime == 4;
        l22a = l22a && (!judged || oks.iter().all(|v| *v));
        if judged {
            let ie_pair = if regime == 1 {
                (energy22 - ie_nat_pre, ie22[i] - ie_gold_pre)
            } else {
                (energy22, ie22[i])
            };
            // Stress spans 100 to 0 across regimes (bands absolute
            // there), so its worst tracks absolute deviation.
            w_sz = w_sz.max((s22.inner.stress[0] - sz[i]).abs());
            w_ie = w_ie.max(max_rel_dev(&[ie_pair.0], &[ie_pair.1]));
            w_ep = w_ep.max(max_rel_dev(&[s22.inner.equiv_plastic_strain], &[pl22[i]]));
            w_lat = w_lat.max(max_rel_dev(&[lat_total], &[ex]));
        } else {
            r_sz = r_sz.max((s22.inner.stress[0] - sz[i]).abs());
            r_ie = r_ie.max(max_rel_dev(&[energy22], &[ie22[i]]));
            r_ep = r_ep.max(max_rel_dev(&[s22.inner.equiv_plastic_strain], &[pl22[i]]));
            r_lat = r_lat.max(max_rel_dev(&[lat_total], &[ex]));
        }
        if judged && !oks.iter().all(|v| *v) && std::env::var("GLEG_DEBUG").is_ok() {
            println!(
                "L22a fail i={i} t={:.3} oks={oks:?} sz={:.3} vs {:.3} ie={:.4} vs {:.4} ep={:.5} vs {:.5} lat={lat_total:.5} vs {ex:.5}/{ey:.5} d={:.4}",
                time22[i],
                s22.inner.stress[0],
                sz[i],
                energy22,
                ie22[i],
                s22.inner.equiv_plastic_strain,
                pl22[i],
                s22.damage
            );
        }
    }
    println!(
        "PROBE L22a brick-replay: masked {} frozen t={:.2} judged worst sz(abs)/ie/ep/lat {w_sz:.3e}/{w_ie:.3e}/{w_ep:.3e}/{w_lat:.3e} info worst {r_sz:.3e}/{r_ie:.3e}/{r_ep:.3e}/{r_lat:.3e} -> {}",
        mask22.iter().filter(|m| **m).count(),
        time22[frozen22.min(time22.len() - 1)],
        if l22a { "PASS" } else { "FAIL" }
    );
    passed += usize::from(l22a);

    // L22b: mass rebuild from both brick volumes. L22c: ledger.
    total += 1;
    let mass22 = channel(&loi22, "MASS");
    let (m22_abs, m22_rel) = channel_tol(&loi22, "MASS");
    let mut l22b = within_tol(5400.0, mass22[0], m22_abs, m22_rel);
    for i in 0..time22.len() {
        l22b = l22b && within_tol(mass22[i], 5400.0, m22_abs, m22_rel);
    }
    println!("PROBE L22b mass: -> {}", if l22b { "PASS" } else { "FAIL" });
    passed += usize::from(l22b);
    total += 1;
    let ext22 = channel(&loi22, "EXTERNAL WORK");
    let ie22g = channel(&loi22, "INTERNAL ENERGY");
    let ke22 = channel(&loi22, "KINETIC ENERGY");
    let mut l22c = true;
    for i in 0..time22.len() {
        let sink = ie22g[i] + ke22[i];
        l22c = l22c && (ext22[i] - sink).abs() <= LEDGER_REL * ext22[i].abs().max(1.0);
    }
    println!(
        "PROBE L22c ledger: -> {}",
        if l22c { "PASS" } else { "FAIL" }
    );
    passed += usize::from(l22c);

    // LOI27 poke test (L27a): part-IE sum against global IE.
    total += 1;
    let loi27 = load_golden(&root, "LOIS/LOI27/B_POKE_9");
    let time27 = time_grid(&loi27);
    let ie20 = channel(&loi27, "Part 20 IE");
    let ie21 = channel(&loi27, "part 21 IE");
    let ie27g = channel(&loi27, "INTERNAL ENERGY");
    let (g27_abs, g27_rel) = channel_tol(&loi27, "INTERNAL ENERGY");
    let mut l27a = ie20.len() == time27.len() && ie21.len() == time27.len();
    for i in 0..time27.len() {
        l27a = l27a && within_tol(ie20[i] + ie21[i], ie27g[i], g27_abs, g27_rel);
    }
    println!(
        "PROBE L27a part-ie-sum: -> {}",
        if l27a { "PASS" } else { "FAIL" }
    );
    passed += usize::from(l27a);

    // L27b: mass conservation plus erosion verdicts.
    total += 1;
    let mass27 = channel(&loi27, "MASS");
    let (m27_abs, m27_rel) = channel_tol(&loi27, "MASS");
    let er20 = channel(&loi27, "Part 20 ERODED");
    let er21 = channel(&loi27, "part 21 ERODED");
    let pm20 = channel(&loi27, "Part 20 MASS");
    let pm21 = channel(&loi27, "part 21 MASS");
    let (e_abs, e_rel) = channel_tol(&loi27, "Part 20 ERODED");
    let (pm_abs, pm_rel) = channel_tol(&loi27, "Part 20 MASS");
    let n27 = time27.len();
    let mut l27b = within_tol(1.68, mass27[0], m27_abs, m27_rel);
    for i in 0..n27 {
        l27b = l27b && within_tol(mass27[i], 1.68, m27_abs, m27_rel);
    }
    l27b = l27b
        && within_tol(er20[n27 - 1], 1.0, e_abs, e_rel)
        && within_tol(er21[n27 - 1], 1.0, e_abs, e_rel)
        && within_tol(pm20[n27 - 1], 0.0, pm_abs, pm_rel)
        && within_tol(pm21[n27 - 1], 0.0, pm_abs, pm_rel);
    println!(
        "PROBE L27b mass-and-erosion: -> {}",
        if l27b { "PASS" } else { "FAIL" }
    );
    passed += usize::from(l27b);

    // L27c/d: native rupture verdicts for both transcribed cards.
    for (id, e, a, b, n, cap, eps_t, eps_m, eps_f, target) in [
        (
            "L27c", 71_000.0, 290.0, 562.3, 0.63, 425.0, 0.14, 0.15, 0.151, 0.18,
        ),
        (
            "L27d", 16_600.0, 220.0, 420.0, 0.6, 270.0, 0.07, 0.08, 0.081, 0.13,
        ),
    ] {
        total += 1;
        let base = Law2::new(
            e,
            0.33,
            Law2Flow::JohnsonCook { a, b, n, c: 0.0 },
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            cap,
            MaxRateEffect::Fixed,
            1.0e30,
            0.0,
            1.0e30,
        );
        let dir = BrittleDir::new(eps_t, eps_m, 0.999, eps_f);
        let law = Law27::new(base, dir, dir);
        let mut state = Law27State::new(0.0);
        drive_uniaxial_stress(&law, &mut state, target, 320);
        let verdict = state.ruptured && von_mises(&state.inner.stress) < 1e-6 * cap;
        println!(
            "PROBE {id} native-rupture-verdict: ruptured={} -> {}",
            state.ruptured,
            if verdict { "PASS" } else { "FAIL" }
        );
        passed += usize::from(verdict);
    }

    // LOI44 brick replay (L44a): uniaxial-stress replay of the axial
    // log path at golden step times with a lateral solve (free sides).
    total += 1;
    let loi44 = load_golden(&root, "LOIS/LOI44/solides_8pt");
    let time44 = time_grid(&loi44);
    let law44 = Law44::new(
        205_000.0,
        0.3,
        180.0,
        450.0,
        0.5,
        1.0,
        300.0,
        400.0,
        5.0,
        MaxRateEffect::Scaled,
        Law44Rate::Total,
        1.0e30,
        1.0e20,
        2.0e20,
    );
    let mut s44 = Law44State::new();
    let n1x = channel(&loi44, "1 NODE 1 ~1");
    let n2x = channel(&loi44, "2 NODE 2 ~1");
    let n1y = channel(&loi44, "1 NODE 1 ~2");
    let n4y = channel(&loi44, "4 NODE 4 ~2");
    let n5z = channel(&loi44, "5 NODE 5 ~3");
    let sz44 = channel(&loi44, "1 TH-BRICK: 1 ~4");
    let ie44 = channel(&loi44, "1 TH-BRICK: 1 ~8");
    let pl44 = channel(&loi44, "1 TH-BRICK: 1 ~10");
    let (z4_abs, z4_rel) = channel_tol(&loi44, "1 TH-BRICK: 1 ~4");
    let (e4_abs, e4_rel) = channel_tol(&loi44, "1 TH-BRICK: 1 ~8");
    let (p4_abs, p4_rel) = channel_tol(&loi44, "1 TH-BRICK: 1 ~10");
    let (nx1_abs, nx1_rel) = channel_tol(&loi44, "1 NODE 1 ~1");
    let (nx2_abs, nx2_rel) = channel_tol(&loi44, "2 NODE 2 ~1");
    let (ny1_abs, ny1_rel) = channel_tol(&loi44, "1 NODE 1 ~2");
    let (ny4_abs, ny4_rel) = channel_tol(&loi44, "4 NODE 4 ~2");
    let mask44 = ring_mask(&sz44, z4_abs, z4_rel);
    let mut l44a = true;
    let mut energy44 = 0.0;
    let (mut w4z, mut w4e, mut w4p, mut w4l) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let (mut r4z, mut r4e, mut r4p, mut r4l) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut prev_axial44 = 0.0;
    let mut prev_sig44 = [0.0; 6];
    let mut lat44 = 0.0;
    let mut lat_guess44 = 0.0;
    for i in 1..time44.len() {
        let axial = log_strain(n5z[i], 100.0);
        let d_axial = (axial - prev_axial44) / 8.0;
        prev_axial44 = axial;
        let dt_sub = (time44[i] - time44[i - 1]) / 8.0;
        for _ in 0..8 {
            let (stress, lat) =
                uniaxial_substep(&law44, &mut s44, d_axial, Some(dt_sub), lat_guess44);
            lat_guess44 = lat;
            lat44 += lat;
            let d_full = [d_axial, lat, lat, 0.0, 0.0, 0.0];
            for k in 0..6 {
                energy44 += 0.5 * (prev_sig44[k] + stress[k]) * d_full[k];
                prev_sig44[k] = stress[k];
            }
        }
        let ex = log_strain(n2x[i] - n1x[i], 100.0);
        let ey = log_strain(n4y[i] - n1y[i], 100.0);
        let lat_band = ((nx2_abs + nx2_rel * n2x[i].abs() + nx1_abs + nx1_rel * n1x[i].abs())
            + (ny4_abs + ny4_rel * n4y[i].abs() + ny1_abs + ny1_rel * n1y[i].abs()))
            / 2.0
            / 100.0;
        let ok4 = [
            within_tol(s44.stress[0], sz44[i], z4_abs, z4_rel),
            within_tol(energy44, ie44[i], e4_abs, e4_rel),
            within_tol(s44.equiv_plastic_strain, pl44[i], p4_abs, p4_rel),
            (lat44 - ex).abs() <= lat_band.max(1e-9) + 1e-6 * ex.abs().max(1.0),
            (lat44 - ey).abs() <= lat_band.max(1e-9) + 1e-6 * ey.abs().max(1.0),
        ];
        let judged = !mask44[i];
        l44a = l44a && (!judged || ok4.iter().all(|v| *v));
        if judged {
            w4z = w4z.max(max_rel_dev(&[s44.stress[0]], &[sz44[i]]));
            w4e = w4e.max(max_rel_dev(&[energy44], &[ie44[i]]));
            w4p = w4p.max(max_rel_dev(&[s44.equiv_plastic_strain], &[pl44[i]]));
            w4l = w4l.max(max_rel_dev(&[lat44], &[ex]));
        } else {
            r4z = r4z.max(max_rel_dev(&[s44.stress[0]], &[sz44[i]]));
            r4e = r4e.max(max_rel_dev(&[energy44], &[ie44[i]]));
            r4p = r4p.max(max_rel_dev(&[s44.equiv_plastic_strain], &[pl44[i]]));
            r4l = r4l.max(max_rel_dev(&[lat44], &[ex]));
        }
        if judged && !ok4.iter().all(|v| *v) && std::env::var("GLEG_DEBUG").is_ok() {
            println!(
                "L44a fail i={i} t={:.3} ok={ok4:?} sz={:.3} vs {:.3} ie={:.4} vs {:.4} ep={:.5} vs {:.5} lat={lat44:.5} vs {ex:.5}/{ey:.5}",
                time44[i],
                s44.stress[0],
                sz44[i],
                energy44,
                ie44[i],
                s44.equiv_plastic_strain,
                pl44[i]
            );
        }
    }
    println!(
        "PROBE L44a brick-replay: masked {} judged worst sz/ie/ep/lat {w4z:.3e}/{w4e:.3e}/{w4p:.3e}/{w4l:.3e} ring worst {r4z:.3e}/{r4e:.3e}/{r4p:.3e}/{r4l:.3e} -> {}",
        mask44.iter().filter(|m| **m).count(),
        if l44a { "PASS" } else { "FAIL" }
    );
    passed += usize::from(l44a);

    // L44b: mass rebuild. L44c: ledger.
    total += 1;
    let mass44 = channel(&loi44, "MASS");
    let (m44_abs, m44_rel) = channel_tol(&loi44, "MASS");
    let mut l44b = within_tol(7800.0, mass44[0], m44_abs, m44_rel);
    for i in 0..time44.len() {
        l44b = l44b && within_tol(mass44[i], 7800.0, m44_abs, m44_rel);
    }
    println!("PROBE L44b mass: -> {}", if l44b { "PASS" } else { "FAIL" });
    passed += usize::from(l44b);
    total += 1;
    let ext44 = channel(&loi44, "EXTERNAL WORK");
    let ie44g = channel(&loi44, "INTERNAL ENERGY");
    let ke44 = channel(&loi44, "KINETIC ENERGY");
    let mut l44c = true;
    for i in 0..time44.len() {
        let sink = ie44g[i] + ke44[i];
        l44c = l44c && (ext44[i] - sink).abs() <= LEDGER_REL * ext44[i].abs().max(1.0);
    }
    println!(
        "PROBE L44c ledger: -> {}",
        if l44c { "PASS" } else { "FAIL" }
    );
    passed += usize::from(l44c);

    // LOI62 shear replay (L62a): formulation band on invariants.
    total += 1;
    let loi62 = load_golden(&root, "LOIS/LOI62/cisaillement");
    let time62 = time_grid(&loi62);
    let law62 = Law62::new(
        0.0,
        vec![(6.9119e-5, 30.4979), (1.20126e-7, -3.6961)],
        vec![
            (0.34391, 46.199),
            (0.10785, 14206.0),
            (0.070763, 199183.3),
            (0.077511, 8625495.0),
        ],
        1.0e30,
    );
    let mut s62 = Law62State::new(4);
    let dx62 = channel(&loi62, "th_nodes 1633 ~1");
    let g62: Vec<Vec<f64>> = (2..=7)
        .map(|k| channel(&loi62, &format!("th_3Dsolid 687 th_3Dsolid ~{k}")))
        .collect();
    let ie62 = channel(&loi62, "th_3Dsolid 687 th_3Dsolid ~8");
    let mut l62a = true;
    let mut energy62 = 0.0;
    let mut worst62 = 0.0f64;
    let peak_q = g62[0]
        .iter()
        .zip(g62[1].iter())
        .zip(g62[2].iter())
        .zip(g62[3].iter())
        .zip(g62[4].iter())
        .zip(g62[5].iter())
        .map(|(((((&a, &b), &c), &d), &e), &f)| von_mises(&[a, b, c, d, e, f]))
        .fold(0.0, f64::max);
    let peak_ie = ie62.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    let mut prev_sig62 = 0.0;
    for i in 1..time62.len() {
        let step = (dx62[i] - dx62[i - 1]) / 70.0 / 8.0;
        let dt_sub = (time62[i] - time62[i - 1]) / 8.0;
        for _ in 0..8 {
            let d_eps = [0.0, 0.0, 0.0, 0.0, 0.0, step];
            let update = law62.update_with_rate(&s62, &d_eps, dt_sub);
            law62.commit(&mut s62, &update);
            energy62 += 0.5 * (prev_sig62 + s62.stress[5]) * step;
            prev_sig62 = s62.stress[5];
        }
        let gold = [
            g62[0][i], g62[1][i], g62[2][i], g62[3][i], g62[4][i], g62[5][i],
        ];
        let q_gold = von_mises(&gold);
        let q_nat = von_mises(&s62.stress);
        let p_gold = -(gold[0] + gold[1] + gold[2]) / 3.0;
        let p_nat = -(s62.stress[0] + s62.stress[1] + s62.stress[2]) / 3.0;
        let ok_q = (q_nat - q_gold).abs() <= FORMULATION_REL * q_gold.abs().max(peak_q);
        let ok_p = (p_nat - p_gold).abs() <= FORMULATION_REL * p_gold.abs().max(peak_q);
        let ok_ie = (energy62 - ie62[i]).abs() <= FORMULATION_REL * ie62[i].abs().max(peak_ie);
        l62a = l62a && ok_q && ok_p && ok_ie;
        worst62 = worst62.max(max_rel_dev(
            &[q_nat, p_nat, energy62],
            &[q_gold, p_gold, ie62[i]],
        ));
        if !(ok_q && ok_p && ok_ie) && std::env::var("GLEG_DEBUG").is_ok() {
            println!(
                "L62a fail i={i} t={:.2} gam={:.4} q={q_nat:.4e} vs {q_gold:.4e} p={p_nat:.4e} vs {p_gold:.4e} ie={energy62:.4e} vs {:.4e} sxz={:.4e} vs {:.4e}",
                time62[i],
                dx62[i] / 70.0,
                ie62[i],
                s62.stress[5],
                gold[5]
            );
        }
    }
    // Exact data checks: OFF, zero PLAS/TEMP, constant DENS.
    let off62 = channel(&loi62, "th_3Dsolid 687 th_3Dsolid ~1");
    let pl62 = channel(&loi62, "th_3Dsolid 687 th_3Dsolid ~10");
    let te62 = channel(&loi62, "th_3Dsolid 687 th_3Dsolid ~11");
    let de62 = channel(&loi62, "th_3Dsolid 687 th_3Dsolid ~9");
    let (d_abs, d_rel) = channel_tol(&loi62, "th_3Dsolid 687 th_3Dsolid ~9");
    for i in 0..time62.len() {
        l62a = l62a
            && within_tol(off62[i], 1.0, 1e-12, 1e-12)
            && within_tol(pl62[i], 0.0, 1e-12, 1e-12)
            && within_tol(te62[i], 0.0, 1e-12, 1e-12)
            && within_tol(de62[i], 5.5e-8, d_abs, d_rel);
    }
    println!(
        "PROBE L62a shear-replay: worst invariant deviation {worst62:.3e} -> {}",
        if l62a { "PASS" } else { "FAIL" }
    );
    passed += usize::from(l62a);

    // L62b: mass rebuild. L62c: ledger.
    total += 1;
    let mass62 = channel(&loi62, "MASS");
    let (m62_abs, m62_rel) = channel_tol(&loi62, "MASS");
    let mut l62b = within_tol(5.5e-8 * 393_750.0, mass62[0], m62_abs, m62_rel);
    for i in 0..time62.len() {
        l62b = l62b && within_tol(mass62[i], 5.5e-8 * 393_750.0, m62_abs, m62_rel);
    }
    println!("PROBE L62b mass: -> {}", if l62b { "PASS" } else { "FAIL" });
    passed += usize::from(l62b);
    total += 1;
    let ext62 = channel(&loi62, "EXTERNAL WORK");
    let ie62g = channel(&loi62, "INTERNAL ENERGY");
    let ke62 = channel(&loi62, "KINETIC ENERGY");
    let mut l62c = true;
    let mut w62 = 0.0f64;
    for i in 0..time62.len() {
        let sink = ie62g[i] + ke62[i];
        let dev = (ext62[i] - sink).abs() / ext62[i].abs().max(1.0);
        w62 = w62.max(dev);
        l62c = l62c && dev <= LEDGER_LOOSE_REL;
    }
    let end62 = (ext62[time62.len() - 1] - ie62g[time62.len() - 1] - ke62[time62.len() - 1]).abs()
        / ext62[time62.len() - 1].abs().max(1.0);
    println!(
        "PROBE L62c ledger: worst {w62:.3e} end {end62:.3e} -> {}",
        if l62c { "PASS" } else { "FAIL" }
    );
    passed += usize::from(l62c);

    // LOI68 (L68a/b/c): ledger, mass, impactor motion. LAW68 material
    // replay belongs to lane 3C and is explicitly not probed here.
    total += 1;
    let loi68 = load_golden(&root, "LOIS/LOI68/cisaillement/test1/input_51");
    let time68 = time_grid(&loi68);
    let ext68 = channel(&loi68, "EXTERNAL WORK");
    let ie68g = channel(&loi68, "INTERNAL ENERGY");
    let ke68 = channel(&loi68, "KINETIC ENERGY");
    let he68 = channel(&loi68, "HOURGLASS ENERGY");
    let mut l68a = true;
    let mut w68 = 0.0f64;
    for i in 0..time68.len() {
        let sink = ie68g[i] + ke68[i] + he68[i];
        let dev = (ext68[i] - sink).abs() / ext68[i].abs().max(1.0);
        w68 = w68.max(dev);
        l68a = l68a && dev <= LEDGER_LOOSE_REL;
    }
    let end68 = (ext68[time68.len() - 1]
        - ie68g[time68.len() - 1]
        - ke68[time68.len() - 1]
        - he68[time68.len() - 1])
        .abs()
        / ext68[time68.len() - 1].abs().max(1.0);
    println!(
        "PROBE L68a ledger: worst {w68:.3e} end {end68:.3e} -> {}",
        if l68a { "PASS" } else { "FAIL" }
    );
    passed += usize::from(l68a);
    total += 1;
    let mass68 = channel(&loi68, "MASS");
    let (m68_abs, m68_rel) = channel_tol(&loi68, "MASS");
    let mut l68b = within_tol(0.0508186, mass68[0], m68_abs, m68_rel);
    for i in 0..time68.len() {
        l68b = l68b && within_tol(mass68[i], 0.0508186, m68_abs, m68_rel);
    }
    println!("PROBE L68b mass: -> {}", if l68b { "PASS" } else { "FAIL" });
    passed += usize::from(l68b);
    total += 1;
    let dz68 = channel(&loi68, "impacteur 15319 impacteur ~3");
    let vz68 = channel(&loi68, "impacteur 15319 impacteur ~6");
    let (z_abs, z_rel) = channel_tol(&loi68, "impacteur 15319 impacteur ~3");
    let (v_abs, v_rel) = channel_tol(&loi68, "impacteur 15319 impacteur ~6");
    let mut l68c = dz68.len() == time68.len();
    for i in 1..time68.len() {
        l68c = l68c
            && within_tol(dz68[i], time68[i], z_abs, z_rel)
            && within_tol(vz68[i], 1.0, v_abs, v_rel);
    }
    println!(
        "PROBE L68c impactor-motion: -> {}",
        if l68c { "PASS" } else { "FAIL" }
    );
    passed += usize::from(l68c);

    println!("G-leg qa-fem-crash-laws: {passed}/{total} probes passed");
    if passed == total {
        println!("G-LEG RESULT: PASS");
        println!("SUCCESS: qa-fem-crash-laws");
    } else {
        println!("G-LEG RESULT: FAIL");
        println!("FAILED: qa-fem-crash-laws");
        std::process::exit(1);
    }
}
