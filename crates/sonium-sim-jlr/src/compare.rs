//! Comparison of run results against the paper's published fixtures.
//!
//! Contains the pass/fail gates of the replication: profile deviation
//! against Table A.4 measurements, circulation number against Table 4.2
//! (experimental band and the authors' simulated values), and Reynolds
//! numbers against Table 4.1. Tolerances follow the paper's own claim:
//! simulated values inside experimental uncertainty, remaining deviation
//! below 8% (abstract and Section 4.2).

// Rust guideline compliant 2026-02-21

use serde::{Deserialize, Serialize};

use crate::case::CaseResult;

/// Deviation summary for one operating point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointComparison {
    /// Specific power input in kW/m³.
    pub specific_power_kw_m3: f64,
    /// Max relative profile deviation vs Table A.4 at matched radii.
    pub max_profile_deviation: f64,
    /// Mean relative profile deviation vs Table A.4.
    pub mean_profile_deviation: f64,
    /// Our circulation number.
    pub circulation_number: f64,
    /// Paper's simulated circulation number.
    pub paper_sim_circulation_number: f64,
    /// Relative deviation from the paper's simulated Ncirc.
    pub circulation_deviation_vs_paper_sim: f64,
    /// Whether Ncirc lies in the experimental band (value ± uncertainty).
    pub circulation_in_experimental_band: bool,
    /// Relative deviations of (ReN, ReAG, ReDT) vs Table 4.1.
    pub reynolds_deviation: [f64; 3],
    /// All replication gates passed for this point.
    pub passed: bool,
}

/// Paper reference values for one operating point, loaded from fixtures.
#[derive(Debug, Clone, Copy)]
pub struct PaperPoint {
    /// Specific power input in kW/m³.
    pub specific_power_kw_m3: f64,
    /// Measured annular velocities at r = 60.0, 62.5, …, 77.5 mm (Table A.4).
    pub measured_profile: [f64; 8],
    /// Paper's simulated circulation number (Table 4.2).
    pub n_circ_sim: f64,
    /// Experimental circulation number (Table 4.2).
    pub n_circ_exp: f64,
    /// Experimental circulation uncertainty (Table 4.2).
    pub n_circ_exp_uncertainty: f64,
    /// (ReN, ReAG, ReDT) from Table 4.1.
    pub reynolds: [f64; 3],
}

/// The paper's published values, transcribed in `fixtures/` (see
/// `fixtures/PROVENANCE.md`); embedded here so the QA binary is standalone.
pub const PAPER_POINTS: [PaperPoint; 3] = [
    PaperPoint {
        specific_power_kw_m3: 2.243,
        measured_profile: [0.424, 0.421, 0.412, 0.400, 0.395, 0.385, 0.380, 0.351],
        n_circ_sim: 3.07,
        n_circ_exp: 3.25,
        n_circ_exp_uncertainty: 0.47,
        reynolds: [1.25e5, 1.67e4, 6.09e4],
    },
    PaperPoint {
        specific_power_kw_m3: 4.038,
        measured_profile: [0.559, 0.532, 0.526, 0.510, 0.505, 0.487, 0.470, 0.454],
        n_circ_sim: 3.18,
        n_circ_exp: 3.42,
        n_circ_exp_uncertainty: 0.41,
        reynolds: [1.52e5, 2.10e4, 7.61e4],
    },
    PaperPoint {
        specific_power_kw_m3: 5.833,
        measured_profile: [0.613, 0.602, 0.584, 0.569, 0.564, 0.549, 0.538, 0.511],
        n_circ_sim: 3.16,
        n_circ_exp: 3.38,
        n_circ_exp_uncertainty: 0.37,
        reynolds: [1.71e5, 2.36e4, 8.56e4],
    },
];

/// Compares one result with the paper point of matching power input.
///
/// The profile gate uses the nearest extracted radius to each measured
/// radius (60.0–77.5 mm). `deviation_budget` is the paper's 8% claim.
///
/// # Panics
/// Panics if no paper point matches the result's power input.
#[must_use]
pub fn compare_point(result: &CaseResult, deviation_budget: f64) -> PointComparison {
    let paper = PAPER_POINTS
        .iter()
        .find(|p| (p.specific_power_kw_m3 - result.specific_power_kw_m3).abs() < 0.01)
        .expect("operating point must be one of the three published points");
    let mut max_dev = 0.0f64;
    let mut sum_dev = 0.0f64;
    for (i, &measured) in paper.measured_profile.iter().enumerate() {
        let r_target = 60.0 + 2.5 * i as f64;
        let (_, sim) = result
            .annular_profile
            .iter()
            .min_by(|a, b| (a.0 - r_target).abs().partial_cmp(&(b.0 - r_target).abs()).unwrap())
            .expect("profile must not be empty");
        let dev = ((sim - measured) / measured).abs();
        max_dev = max_dev.max(dev);
        sum_dev += dev;
    }
    let mean_dev = sum_dev / paper.measured_profile.len() as f64;
    let circ_dev = ((result.circulation_number - paper.n_circ_sim) / paper.n_circ_sim).abs();
    let in_band = (result.circulation_number - paper.n_circ_exp).abs()
        <= paper.n_circ_exp_uncertainty;
    let re = [
        result.re_nozzle,
        result.re_annular_gap,
        result.re_draft_tube,
    ];
    let re_dev = [
        ((re[0] - paper.reynolds[0]) / paper.reynolds[0]).abs(),
        ((re[1] - paper.reynolds[1]) / paper.reynolds[1]).abs(),
        ((re[2] - paper.reynolds[2]) / paper.reynolds[2]).abs(),
    ];
    PointComparison {
        specific_power_kw_m3: result.specific_power_kw_m3,
        max_profile_deviation: max_dev,
        mean_profile_deviation: mean_dev,
        circulation_number: result.circulation_number,
        paper_sim_circulation_number: paper.n_circ_sim,
        circulation_deviation_vs_paper_sim: circ_dev,
        circulation_in_experimental_band: in_band,
        reynolds_deviation: re_dev,
        passed: max_dev <= deviation_budget && circ_dev <= deviation_budget && in_band,
    }
}
