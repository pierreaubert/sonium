//! CEA-2034 measurement generation and file output.
//!
//! Generates 72 evaluation points (36 horizontal + 36 vertical at 1 m)
//! and writes per-angle measurement files to a `cea2034/` subdirectory.
//!
//! Also provides spinorama scoring (Listening Window, Early Reflections,
//! Sound Power, PIR) and preference score via `sonium_roomeq::cea2034`.

use std::io::Write as IoWrite;
use std::path::Path;

use sonium_compute::HostMatrix;

use crate::simulation::{SimulationProvenance, SimulationResult};

/// Number of angles per plane (every 10° from 0° to 350°).
pub const ANGLES_PER_PLANE: usize = 36;

/// Total number of evaluation points (horizontal + vertical).
pub const TOTAL_EVAL_POINTS: usize = 2 * ANGLES_PER_PLANE;

/// Named axes in Sonium's speaker-product coordinate frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeakerAxis {
    Right,
    Up,
    Forward,
}

/// Coordinate and ordering contract for the CEA-2034 angle files.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Cea2034MeasurementFrame {
    /// All samples lie one metre from this origin.
    pub origin: String,
    pub x_axis: SpeakerAxis,
    pub y_axis: SpeakerAxis,
    pub z_axis: SpeakerAxis,
    /// `H 0.txt` through `H 350.txt`: rotation about this axis.
    pub horizontal_rotation_axis: SpeakerAxis,
    /// `V 0.txt` through `V 350.txt`: rotation about this axis.
    pub vertical_rotation_axis: SpeakerAxis,
    pub radius_m: f64,
    pub angle_step_degrees: u16,
}

/// Versioned sidecar for a CEA-2034 text-file set.
///
/// The angle files deliberately retain the conventional, simple three-column
/// format.  This manifest carries the information that format cannot express,
/// so CEA measurements are not detached from the solved product contract.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Cea2034ArtifactManifest {
    pub schema_version: u32,
    pub frequencies_hz: Vec<f64>,
    pub measurement_frame: Cea2034MeasurementFrame,
    /// `None` is a legacy prescribed-velocity result.  `Some(n)` means every
    /// frequency retained valid dynamic evidence for exactly `n` drivers.
    pub driver_count: Option<usize>,
    pub provenance: SimulationProvenance,
}

const CEA2034_MANIFEST_SCHEMA_VERSION: u32 = 2;

fn cea2034_measurement_frame() -> Cea2034MeasurementFrame {
    Cea2034MeasurementFrame {
        origin: "baffle_center".into(),
        x_axis: SpeakerAxis::Right,
        y_axis: SpeakerAxis::Up,
        z_axis: SpeakerAxis::Forward,
        horizontal_rotation_axis: SpeakerAxis::Up,
        vertical_rotation_axis: SpeakerAxis::Right,
        radius_m: 1.0,
        angle_step_degrees: 10,
    }
}

/// Generate the 72 CEA-2034 evaluation points at 1 m from the baffle center.
///
/// Returns an array of 72 points: first 36 are horizontal plane (rotating
/// around y-axis), next 36 are vertical plane (rotating around x-axis).
///
/// Coordinate system: z = acoustic axis (forward), y = up, x = right.
pub fn generate_eval_points(baffle_center: [f64; 3]) -> Vec<[f64; 3]> {
    let mut points = Vec::with_capacity(TOTAL_EVAL_POINTS);

    // Horizontal plane: rotate around y-axis
    for i in 0..ANGLES_PER_PLANE {
        let angle_rad = (i as f64 * 10.0).to_radians();
        points.push([
            baffle_center[0] + angle_rad.sin(),
            baffle_center[1],
            baffle_center[2] + angle_rad.cos(),
        ]);
    }

    // Vertical plane: rotate around x-axis
    for i in 0..ANGLES_PER_PLANE {
        let angle_rad = (i as f64 * 10.0).to_radians();
        points.push([
            baffle_center[0],
            baffle_center[1] + angle_rad.sin(),
            baffle_center[2] + angle_rad.cos(),
        ]);
    }

    points
}

/// Write CEA-2034 measurement files.
///
/// Creates `{output_dir}/cea2034/H {angle}.txt` and `V {angle}.txt` files,
/// each containing `freq_hz  spl_db  phase_deg` rows.
pub fn write_cea2034_files(output_dir: &Path, result: &SimulationResult) -> std::io::Result<()> {
    validate_cea2034_result(result)
        .map_err(|message| std::io::Error::new(std::io::ErrorKind::InvalidInput, message))?;
    let cea_dir = output_dir.join("cea2034");
    std::fs::create_dir_all(&cea_dir)?;

    let n_freqs = result.frequencies.len();

    // Horizontal files: H 0.txt, H 10.txt, ..., H 350.txt
    for i in 0..ANGLES_PER_PLANE {
        let angle = i * 10;
        let path = cea_dir.join(format!("H {angle}.txt"));
        write_angle_file(
            &path,
            &result.frequencies,
            n_freqs,
            i,
            &result.spl,
            &result.phase,
        )?;
    }

    // Vertical files: V 0.txt, V 10.txt, ..., V 350.txt
    for i in 0..ANGLES_PER_PLANE {
        let angle = i * 10;
        let path = cea_dir.join(format!("V {angle}.txt"));
        let point_idx = ANGLES_PER_PLANE + i;
        write_angle_file(
            &path,
            &result.frequencies,
            n_freqs,
            point_idx,
            &result.spl,
            &result.phase,
        )?;
    }

    let driver_count = cea_dynamic_driver_count(result)
        .map_err(|message| std::io::Error::new(std::io::ErrorKind::InvalidInput, message))?;
    let manifest = Cea2034ArtifactManifest {
        schema_version: CEA2034_MANIFEST_SCHEMA_VERSION,
        frequencies_hz: result.frequencies.clone(),
        measurement_frame: cea2034_measurement_frame(),
        driver_count,
        provenance: result.provenance.clone(),
    };
    let manifest = serde_json::to_string_pretty(&manifest)
        .map_err(|error| std::io::Error::other(format!("serialize CEA-2034 manifest: {error}")))?;
    std::fs::write(cea_dir.join("manifest.json"), manifest)?;

    Ok(())
}

fn write_angle_file(
    path: &Path,
    frequencies: &[f64],
    n_freqs: usize,
    point_idx: usize,
    spl: &HostMatrix<f64>,
    phase: &HostMatrix<f64>,
) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    for fi in 0..n_freqs {
        let freq = frequencies[fi];
        let s = spl[(fi, point_idx)];
        let p_deg = phase[(fi, point_idx)].to_degrees();
        writeln!(f, "{freq:.2}\t{s:.4}\t{p_deg:.4}")?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Spinorama scoring
// ---------------------------------------------------------------------------

/// Spinorama-derived metrics from a simulation result.
#[cfg(feature = "cli")]
#[derive(Debug, Clone)]
pub struct SpinoramaMetrics {
    /// On-axis SPL curve (frequencies × SPL dB).
    pub on_axis: Vec<f64>,
    /// Listening Window SPL.
    pub listening_window: Vec<f64>,
    /// Early Reflections SPL.
    pub early_reflections: Vec<f64>,
    /// Sound Power SPL.
    pub sound_power: Vec<f64>,
    /// Predicted In-Room response.
    pub pir: Vec<f64>,
    /// CEA-2034 preference score.
    pub preference_score: f64,
}

/// One independently supplied CEA-2034 curve sample.
#[cfg(feature = "cli")]
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MeasuredSpinoramaPoint {
    pub frequency_hz: f64,
    pub on_axis_db: f64,
    pub listening_window_db: f64,
    pub early_reflections_db: f64,
    pub sound_power_db: f64,
    pub pir_db: f64,
}

/// RMSE and worst-case gates for the CEA-2034 curves and preference score.
#[cfg(feature = "cli")]
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SpinoramaReferenceTolerances {
    pub curve_rmse_db: f64,
    pub curve_max_db: f64,
    pub preference_score_max: f64,
}

#[cfg(feature = "cli")]
impl Default for SpinoramaReferenceTolerances {
    fn default() -> Self {
        Self {
            curve_rmse_db: 0.1,
            curve_max_db: 0.25,
            preference_score_max: 0.1,
        }
    }
}

#[cfg(feature = "cli")]
impl SpinoramaReferenceTolerances {
    fn validate(self) -> Result<(), String> {
        for (name, value) in [
            ("curve_rmse_db", self.curve_rmse_db),
            ("curve_max_db", self.curve_max_db),
            ("preference_score_max", self.preference_score_max),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(format!(
                    "spinorama tolerance {name} must be finite and non-negative"
                ));
            }
        }
        Ok(())
    }
}

/// Correlation statistics for one CEA-2034 curve.
#[cfg(feature = "cli")]
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SpinoramaMetricReport {
    pub samples: usize,
    pub rmse_db: f64,
    pub maximum_absolute_error_db: f64,
    pub rmse_tolerance_db: f64,
    pub maximum_tolerance_db: f64,
    pub passed: bool,
}

/// Complete spinorama reference-correlation report.
#[cfg(feature = "cli")]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SpinoramaReferenceReport {
    pub on_axis: SpinoramaMetricReport,
    pub listening_window: SpinoramaMetricReport,
    pub early_reflections: SpinoramaMetricReport,
    pub sound_power: SpinoramaMetricReport,
    pub pir: SpinoramaMetricReport,
    pub preference_score_absolute_error: f64,
    pub preference_score_tolerance: f64,
    pub preference_score_passed: bool,
    pub passed: bool,
}

/// Correlate computed CEA-2034 curves against an independent reference grid.
#[cfg(feature = "cli")]
pub fn correlate_spinorama_reference(
    frequencies_hz: &[f64],
    actual: &SpinoramaMetrics,
    reference: &[MeasuredSpinoramaPoint],
    reference_preference_score: f64,
    tolerances: SpinoramaReferenceTolerances,
) -> Result<SpinoramaReferenceReport, String> {
    if reference.is_empty() {
        return Err("spinorama reference must contain at least one frequency".into());
    }
    if frequencies_hz.len() != reference.len() {
        return Err("spinorama frequency and reference grids must have equal length".into());
    }
    tolerances.validate()?;
    if !actual.preference_score.is_finite() || !reference_preference_score.is_finite() {
        return Err("spinorama preference scores must be finite".into());
    }

    let actual_curves = [
        ("on-axis", actual.on_axis.as_slice()),
        ("listening-window", actual.listening_window.as_slice()),
        ("early-reflections", actual.early_reflections.as_slice()),
        ("sound-power", actual.sound_power.as_slice()),
        ("PIR", actual.pir.as_slice()),
    ];
    for (name, curve) in actual_curves {
        if curve.len() != reference.len() {
            return Err(format!(
                "spinorama {name} curve and reference grids must have equal length"
            ));
        }
        if curve.iter().any(|value| !value.is_finite()) {
            return Err(format!("spinorama {name} curve contains non-finite values"));
        }
    }

    let mut previous_frequency = None;
    for (index, (&frequency_hz, point)) in frequencies_hz.iter().zip(reference).enumerate() {
        if !frequency_hz.is_finite()
            || frequency_hz <= 0.0
            || previous_frequency.is_some_and(|previous| frequency_hz <= previous)
        {
            return Err(format!(
                "spinorama frequency grid is not finite, positive, and strictly increasing at index {index}"
            ));
        }
        if !same_frequency(frequency_hz, point.frequency_hz) {
            return Err(format!(
                "spinorama reference frequency does not match computed grid at index {index}"
            ));
        }
        if [
            point.on_axis_db,
            point.listening_window_db,
            point.early_reflections_db,
            point.sound_power_db,
            point.pir_db,
        ]
        .iter()
        .any(|value| !value.is_finite())
        {
            return Err(format!(
                "spinorama reference sample {index} contains a non-finite curve value"
            ));
        }
        previous_frequency = Some(frequency_hz);
    }

    let expected = [
        reference
            .iter()
            .map(|point| point.on_axis_db)
            .collect::<Vec<_>>(),
        reference
            .iter()
            .map(|point| point.listening_window_db)
            .collect::<Vec<_>>(),
        reference
            .iter()
            .map(|point| point.early_reflections_db)
            .collect::<Vec<_>>(),
        reference
            .iter()
            .map(|point| point.sound_power_db)
            .collect::<Vec<_>>(),
        reference
            .iter()
            .map(|point| point.pir_db)
            .collect::<Vec<_>>(),
    ];
    let reports: Vec<_> = actual_curves
        .iter()
        .zip(expected.iter())
        .map(|((_, curve), expected)| {
            metric_report(
                curve,
                expected,
                tolerances.curve_rmse_db,
                tolerances.curve_max_db,
            )
        })
        .collect();
    let preference_score_absolute_error =
        (actual.preference_score - reference_preference_score).abs();
    let preference_score_passed =
        preference_score_absolute_error <= tolerances.preference_score_max;
    Ok(SpinoramaReferenceReport {
        on_axis: reports[0],
        listening_window: reports[1],
        early_reflections: reports[2],
        sound_power: reports[3],
        pir: reports[4],
        preference_score_absolute_error,
        preference_score_tolerance: tolerances.preference_score_max,
        preference_score_passed,
        passed: reports.iter().all(|report| report.passed) && preference_score_passed,
    })
}

#[cfg(feature = "cli")]
fn same_frequency(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1.0e-9 * left.max(right).max(1.0)
}

#[cfg(feature = "cli")]
fn metric_report(
    actual: &[f64],
    expected: &[f64],
    rmse_tolerance_db: f64,
    maximum_tolerance_db: f64,
) -> SpinoramaMetricReport {
    let errors: Vec<_> = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .collect();
    let rmse_db =
        (errors.iter().map(|error| error * error).sum::<f64>() / errors.len() as f64).sqrt();
    let maximum_absolute_error_db = errors.iter().copied().fold(0.0, f64::max);
    SpinoramaMetricReport {
        samples: errors.len(),
        rmse_db,
        maximum_absolute_error_db,
        rmse_tolerance_db,
        maximum_tolerance_db,
        passed: rmse_db <= rmse_tolerance_db && maximum_absolute_error_db <= maximum_tolerance_db,
    }
}

/// Compute spinorama metrics from a simulation result.
///
/// Builds the `spl_h` and `spl_v` matrices and computes LW, ER, SP, PIR, and
/// the preference score via `sonium_roomeq::cea2034::score_peq`.
#[cfg(feature = "cli")]
pub fn compute_spinorama_metrics(result: &SimulationResult) -> Result<SpinoramaMetrics, String> {
    validate_cea2034_result(result)?;
    let freq_arr = result.frequencies.clone();
    let (spl_h, spl_v) = spl_planes_for_autoeq(result);
    let angles = autoeq_angle_labels();
    let spinorama_input = sonium_roomeq::cea2034::ValidatedSpinorama::new(
        &freq_arr,
        &angles,
        &spl_h,
        &angles,
        &spl_v,
        sonium_roomeq::cea2034::SpinoramaAngle {
            plane: sonium_roomeq::cea2034::SpinoramaPlane::Horizontal,
            degrees: 0.0,
        },
    )
    .map_err(|error| error.to_string())?;
    let idx = spinorama_index_groups();
    let weights = spinorama_sound_power_weights();
    let peq = vec![0.0f64; result.frequencies.len()];
    let intervals = sonium_roomeq::cea2034::octave_intervals(2, &freq_arr);
    let (spinorama, metrics) =
        sonium_roomeq::cea2034::score_peq(&spinorama_input, &idx, &intervals, &weights, &peq)
            .map_err(|error| error.to_string())?;

    let sound_power_row = idx.len() - 1;
    let pir_row = sound_power_row + 1;

    let output = SpinoramaMetrics {
        on_axis: spl_h[AUTOEQ_ON_AXIS_ROW].clone(),
        listening_window: spinorama[0].clone(),
        early_reflections: spinorama[1].clone(),
        sound_power: spinorama[sound_power_row].clone(),
        pir: spinorama[pir_row].clone(),
        preference_score: metrics.pref_score,
    };
    let curves_are_finite = output
        .on_axis
        .iter()
        .chain(&output.listening_window)
        .chain(&output.early_reflections)
        .chain(&output.sound_power)
        .chain(&output.pir)
        .all(|value| value.is_finite());
    if !curves_are_finite || !output.preference_score.is_finite() {
        return Err(
            "CEA-2034 scoring produced a non-finite result; the frequency grid is insufficient for preference scoring"
                .into(),
        );
    }
    Ok(output)
}

fn validate_cea2034_result(result: &SimulationResult) -> Result<(), String> {
    let n_freqs = result.frequencies.len();
    if n_freqs == 0 {
        return Err("simulation result has no frequencies".into());
    }
    if result.spl.nrows() != n_freqs || result.phase.nrows() != n_freqs {
        return Err("SPL/phase row counts must match the frequency count".into());
    }
    if result.spl.ncols() != TOTAL_EVAL_POINTS || result.phase.ncols() != TOTAL_EVAL_POINTS {
        return Err(format!(
            "CEA-2034 requires exactly {TOTAL_EVAL_POINTS} evaluation points"
        ));
    }
    let mut previous_frequency = None;
    for (index, &frequency) in result.frequencies.iter().enumerate() {
        if !frequency.is_finite() || frequency <= 0.0 {
            return Err(format!(
                "frequency at index {index} must be finite and positive"
            ));
        }
        if previous_frequency.is_some_and(|previous| frequency <= previous) {
            return Err("frequencies must be strictly increasing".into());
        }
        previous_frequency = Some(frequency);
        for point in 0..TOTAL_EVAL_POINTS {
            if !result.spl[(index, point)].is_finite() || !result.phase[(index, point)].is_finite()
            {
                return Err(format!(
                    "non-finite SPL/phase at frequency index {index}, point {point}"
                ));
            }
        }
    }
    cea_dynamic_driver_count(result)?;
    Ok(())
}

fn cea_dynamic_driver_count(result: &SimulationResult) -> Result<Option<usize>, String> {
    if result.driver_velocities_bem.len() != result.frequencies.len() {
        return Err("CEA-2034 driver-velocity evidence does not match the frequency grid".into());
    }
    let Some(first) = result.driver_velocities_bem.first() else {
        return Ok(None);
    };
    if first.is_empty() {
        if result
            .driver_velocities_bem
            .iter()
            .all(|velocities| velocities.is_empty())
        {
            return Ok(None);
        }
        return Err("CEA-2034 mixes missing and dynamic driver-velocity evidence".into());
    }
    let driver_count = first.len();
    if result.driver_velocities_bem.iter().any(|velocities| {
        velocities.len() != driver_count
            || velocities
                .iter()
                .any(|velocity| !velocity.re.is_finite() || !velocity.im.is_finite())
    }) {
        return Err("CEA-2034 has malformed dynamic driver-velocity evidence".into());
    }
    Ok(Some(driver_count))
}

#[cfg(feature = "cli")]
const AUTOEQ_ON_AXIS_ROW: usize = 17;
#[cfg(feature = "cli")]
const AUTOEQ_BACK_AXIS_ROW: usize = 35;
#[cfg(feature = "cli")]
const AUTOEQ_VERTICAL_OFFSET: usize = ANGLES_PER_PLANE;

#[cfg(feature = "cli")]
fn autoeq_angle_labels() -> Vec<f64> {
    (0..ANGLES_PER_PLANE)
        .map(|row| row as f64 * 10.0 - 170.0)
        .collect()
}

#[cfg(feature = "cli")]
fn spl_planes_for_autoeq(result: &SimulationResult) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n_freqs = result.frequencies.len();
    let mut spl_h = vec![vec![0.0f64; n_freqs]; ANGLES_PER_PLANE];
    let mut spl_v = vec![vec![0.0f64; n_freqs]; ANGLES_PER_PLANE];

    for i in 0..ANGLES_PER_PLANE {
        let row = measurement_angle_index_to_autoeq_row(i);
        for fi in 0..n_freqs {
            spl_h[row][fi] = result.spl[(fi, i)];
            spl_v[row][fi] = result.spl[(fi, ANGLES_PER_PLANE + i)];
        }
    }

    (spl_h, spl_v)
}

#[cfg(feature = "cli")]
fn spinorama_index_groups() -> Vec<Vec<usize>> {
    vec![
        // Listening Window: on-axis, ±10° V, ±10/20/30° H.
        vec![
            h(0),
            v(-10),
            v(10),
            h(-10),
            h(10),
            h(-20),
            h(20),
            h(-30),
            h(30),
        ],
        // score_peq overwrites row 1 with the aggregate ER curve after computing rows 2..=6.
        vec![h(0)],
        rows_v(&[-20, -30, -40]),
        rows_v(&[40, 50, 60]),
        rows_h(&[0, -10, 10, -20, 20, -30, 30]),
        rows_h(&[-40, 40, -50, 50, -60, 60, -70, 70, -80, 80]),
        rows_h(&[-90, 90, 180]),
        sound_power_rows(),
    ]
}

#[cfg(feature = "cli")]
fn sound_power_rows() -> Vec<usize> {
    let mut rows = Vec::with_capacity(70);
    rows.extend(0..ANGLES_PER_PLANE);
    rows.extend((0..ANGLES_PER_PLANE).filter_map(|row| {
        if row == AUTOEQ_ON_AXIS_ROW || row == AUTOEQ_BACK_AXIS_ROW {
            None
        } else {
            Some(AUTOEQ_VERTICAL_OFFSET + row)
        }
    }));
    rows
}

#[cfg(feature = "cli")]
fn spinorama_sound_power_weights() -> Vec<f64> {
    let mut weights = vec![0.0; TOTAL_EVAL_POINTS];
    for row in 0..ANGLES_PER_PLANE {
        let theta = autoeq_row_to_polar_angle(row);
        let ring_weight = spherical_slice_weight(theta);
        let samples_in_ring = if theta == 0 || theta == 180 { 1.0 } else { 4.0 };
        let point_weight = ring_weight / samples_in_ring;

        weights[row] = point_weight;
        if theta != 0 && theta != 180 {
            weights[AUTOEQ_VERTICAL_OFFSET + row] = point_weight;
        }
    }

    weights
}

#[cfg(feature = "cli")]
fn spherical_slice_weight(theta_deg: i32) -> f64 {
    let low = (theta_deg - 5).max(0) as f64;
    let high = (theta_deg + 5).min(180) as f64;
    low.to_radians().cos() - high.to_radians().cos()
}

#[cfg(feature = "cli")]
fn autoeq_row_to_polar_angle(row: usize) -> i32 {
    (row as i32 * 10 - 170).abs()
}

#[cfg(feature = "cli")]
fn measurement_angle_index_to_autoeq_row(index: usize) -> usize {
    let angle_deg = index as i32 * 10;
    let signed = if angle_deg <= 180 {
        angle_deg
    } else {
        angle_deg - 360
    };
    signed_angle_to_autoeq_row(signed)
}

#[cfg(feature = "cli")]
fn signed_angle_to_autoeq_row(angle_deg: i32) -> usize {
    debug_assert!((-170..=180).contains(&angle_deg));
    debug_assert_eq!(angle_deg % 10, 0);
    ((angle_deg + 170) / 10) as usize
}

#[cfg(feature = "cli")]
fn h(angle_deg: i32) -> usize {
    signed_angle_to_autoeq_row(angle_deg)
}

#[cfg(feature = "cli")]
fn v(angle_deg: i32) -> usize {
    AUTOEQ_VERTICAL_OFFSET + signed_angle_to_autoeq_row(angle_deg)
}

#[cfg(feature = "cli")]
fn rows_h(angles_deg: &[i32]) -> Vec<usize> {
    angles_deg.iter().copied().map(h).collect()
}

#[cfg(feature = "cli")]
fn rows_v(angles_deg: &[i32]) -> Vec<usize> {
    angles_deg.iter().copied().map(v).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_points_count() {
        let pts = generate_eval_points([0.0, 0.0, 0.14]);
        assert_eq!(pts.len(), TOTAL_EVAL_POINTS);
    }

    #[test]
    fn test_eval_points_at_1m() {
        let center = [0.0, 0.0, 0.14];
        let pts = generate_eval_points(center);
        for pt in &pts {
            let dx = pt[0] - center[0];
            let dy = pt[1] - center[1];
            let dz = pt[2] - center[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            assert!(
                (dist - 1.0).abs() < 1e-10,
                "point distance from center = {dist}"
            );
        }
    }

    #[test]
    fn test_on_axis_is_first_point() {
        let center = [0.0, 0.0, 0.14];
        let pts = generate_eval_points(center);
        // Horizontal 0°: should be on the acoustic axis (+z)
        assert!((pts[0][0] - center[0]).abs() < 1e-10);
        assert!((pts[0][1] - center[1]).abs() < 1e-10);
        assert!((pts[0][2] - center[2] - 1.0).abs() < 1e-10);
        // Vertical 0°: should also be on the acoustic axis
        assert!((pts[36][0] - center[0]).abs() < 1e-10);
        assert!((pts[36][1] - center[1]).abs() < 1e-10);
        assert!((pts[36][2] - center[2] - 1.0).abs() < 1e-10);
    }

    // Fixture fills the deprecated per-driver vectors alongside the matrix.
    #[allow(deprecated)]
    #[test]
    fn test_write_cea2034_files() {
        let mut spl = HostMatrix::<f64>::zeros(3, TOTAL_EVAL_POINTS);
        spl.row_mut(0).fill(80.0);
        spl.row_mut(1).fill(85.0);
        spl.row_mut(2).fill(75.0);
        let mut phase = HostMatrix::<f64>::zeros(3, TOTAL_EVAL_POINTS);
        phase.row_mut(0).fill(0.0);
        phase.row_mut(1).fill(-0.5);
        phase.row_mut(2).fill(-1.0);
        let mut result = SimulationResult {
            frequencies: vec![100.0, 1000.0, 10000.0],
            complex_pressure: HostMatrix::zeros(spl.nrows(), spl.ncols()),
            spl,
            phase,
            woofer_electrical_impedance_ohm: vec![None; 3],
            tweeter_electrical_impedance_ohm: vec![None; 3],
            driver_electrical_impedances_ohm: vec![vec![None; 2]; 3],
            radiation_matrices: vec![None; 3],
            radiation_matrix_diagnostics: vec![None; 3],
            thermal_driver_states: vec![[None, None]; 3],
            driver_thermal_states: vec![vec![None; 2]; 3],
            power_ledgers: vec![None; 3],
            product_reference_sweep: None,
            structural_responses: Vec::new(),
            driver_velocities_bem: vec![vec![]; 3],
            exact_residual_evaluations: 0,
            provenance: Default::default(),
        };
        result.driver_velocities_bem = vec![
            vec![
                num_complex::Complex64::new(0.01, 0.0),
                num_complex::Complex64::new(0.02, 0.0),
            ];
            3
        ];

        let dir = std::env::temp_dir().join("speaker_cea2034_test");
        let _ = std::fs::remove_dir_all(&dir);
        write_cea2034_files(&dir, &result).unwrap();

        // Check files exist
        let cea_dir = dir.join("cea2034");
        assert!(cea_dir.join("H 0.txt").exists());
        assert!(cea_dir.join("H 350.txt").exists());
        assert!(cea_dir.join("V 0.txt").exists());
        assert!(cea_dir.join("V 350.txt").exists());
        let manifest: Cea2034ArtifactManifest =
            serde_json::from_str(&std::fs::read_to_string(cea_dir.join("manifest.json")).unwrap())
                .unwrap();
        assert_eq!(manifest.schema_version, CEA2034_MANIFEST_SCHEMA_VERSION);
        assert_eq!(manifest.frequencies_hz, result.frequencies);
        assert_eq!(manifest.measurement_frame, cea2034_measurement_frame());
        assert_eq!(manifest.driver_count, Some(2));
        assert_eq!(manifest.provenance, result.provenance);

        // Check file content
        let content = std::fs::read_to_string(cea_dir.join("H 0.txt")).unwrap();
        let lines: Vec<&str> = content.trim().lines().collect();
        assert_eq!(lines.len(), 3); // 3 frequencies
        assert!(lines[0].starts_with("100.00"));
        assert!(lines[0].contains("80.0"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    // Fixture fills the deprecated per-driver vectors alongside the matrix.
    #[allow(deprecated)]
    #[test]
    fn test_write_cea2034_files_rejects_wrong_shape_and_non_finite_data() {
        let dir = std::env::temp_dir().join("speaker_cea2034_invalid_test");
        let _ = std::fs::remove_dir_all(&dir);

        let wrong_shape = SimulationResult {
            frequencies: vec![1000.0],
            spl: HostMatrix::<f64>::zeros(1, TOTAL_EVAL_POINTS - 1),
            phase: HostMatrix::<f64>::zeros(1, TOTAL_EVAL_POINTS - 1),
            complex_pressure: HostMatrix::zeros(1, TOTAL_EVAL_POINTS - 1),
            woofer_electrical_impedance_ohm: vec![None],
            tweeter_electrical_impedance_ohm: vec![None],
            driver_electrical_impedances_ohm: vec![vec![None; 2]],
            radiation_matrices: vec![None],
            radiation_matrix_diagnostics: vec![None],
            thermal_driver_states: vec![[None, None]],
            driver_thermal_states: vec![vec![None; 2]],
            power_ledgers: vec![None],
            product_reference_sweep: None,
            structural_responses: Vec::new(),
            driver_velocities_bem: vec![vec![]],
            exact_residual_evaluations: 0,
            provenance: Default::default(),
        };
        assert!(write_cea2034_files(&dir, &wrong_shape).is_err());

        let mut non_finite = SimulationResult {
            frequencies: vec![1000.0],
            spl: HostMatrix::<f64>::zeros(1, TOTAL_EVAL_POINTS),
            phase: HostMatrix::<f64>::zeros(1, TOTAL_EVAL_POINTS),
            complex_pressure: HostMatrix::zeros(1, TOTAL_EVAL_POINTS),
            woofer_electrical_impedance_ohm: vec![None],
            tweeter_electrical_impedance_ohm: vec![None],
            driver_electrical_impedances_ohm: vec![vec![None; 2]],
            radiation_matrices: vec![None],
            radiation_matrix_diagnostics: vec![None],
            thermal_driver_states: vec![[None, None]],
            driver_thermal_states: vec![vec![None; 2]],
            power_ledgers: vec![None],
            product_reference_sweep: None,
            structural_responses: Vec::new(),
            driver_velocities_bem: vec![vec![]],
            exact_residual_evaluations: 0,
            provenance: Default::default(),
        };
        non_finite.spl[(0, 0)] = f64::NAN;
        assert!(write_cea2034_files(&dir, &non_finite).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    // Fixture fills the deprecated per-driver vectors alongside the matrix.
    #[allow(deprecated)]
    #[test]
    fn test_write_cea2034_files_rejects_malformed_dynamic_driver_evidence() {
        let dir = std::env::temp_dir().join("speaker_cea2034_malformed_driver_test");
        let _ = std::fs::remove_dir_all(&dir);
        let malformed = SimulationResult {
            frequencies: vec![1000.0],
            spl: HostMatrix::<f64>::zeros(1, TOTAL_EVAL_POINTS),
            phase: HostMatrix::<f64>::zeros(1, TOTAL_EVAL_POINTS),
            complex_pressure: HostMatrix::zeros(1, TOTAL_EVAL_POINTS),
            woofer_electrical_impedance_ohm: vec![None],
            tweeter_electrical_impedance_ohm: vec![None],
            driver_electrical_impedances_ohm: vec![vec![None; 2]],
            radiation_matrices: vec![None],
            radiation_matrix_diagnostics: vec![None],
            thermal_driver_states: vec![[None, None]],
            driver_thermal_states: vec![vec![None; 2]],
            power_ledgers: vec![None],
            product_reference_sweep: None,
            structural_responses: Vec::new(),
            driver_velocities_bem: vec![vec![num_complex::Complex64::new(f64::NAN, 0.0)]],
            exact_residual_evaluations: 0,
            provenance: Default::default(),
        };
        assert!(write_cea2034_files(&dir, &malformed).is_err());
        assert!(!dir.join("cea2034").join("manifest.json").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_angle_mapping_row17_is_on_axis() {
        // Measurement index 0 = 0° → signed 0° → row (0+170)/10 = 17
        let angle_deg = 0_i32;
        let signed = angle_deg;
        let row = (signed + 170) / 10;
        assert_eq!(row, 17);
    }

    #[test]
    fn test_angle_mapping_coverage() {
        // Ensure all 36 measurement angles map to unique rows 0..36
        let mut rows = Vec::new();
        for i in 0..36 {
            let angle_deg = i * 10;
            let signed = if angle_deg <= 180 {
                angle_deg
            } else {
                angle_deg - 360
            };
            let row = ((signed + 170) / 10) as usize;
            assert!(row < 36, "row {row} out of range for angle {angle_deg}°");
            rows.push(row);
        }
        rows.sort();
        rows.dedup();
        assert_eq!(rows.len(), 36, "not all rows covered: {rows:?}");
    }

    #[cfg(feature = "cli")]
    #[test]
    fn test_spinorama_index_groups_match_cta_layout() {
        let idx = spinorama_index_groups();

        assert_eq!(
            idx[0],
            vec![
                h(0),
                v(-10),
                v(10),
                h(-10),
                h(10),
                h(-20),
                h(20),
                h(-30),
                h(30),
            ]
        );
        assert_eq!(idx[2], rows_v(&[-20, -30, -40]));
        assert_eq!(idx[3], rows_v(&[40, 50, 60]));
        assert_eq!(idx[4], rows_h(&[0, -10, 10, -20, 20, -30, 30]));
        assert_eq!(
            idx[5],
            rows_h(&[-40, 40, -50, 50, -60, 60, -70, 70, -80, 80])
        );
        assert_eq!(idx[6], rows_h(&[-90, 90, 180]));

        let sound_power = idx.last().unwrap();
        assert_eq!(sound_power.len(), 70);
        assert!(sound_power.contains(&h(0)));
        assert!(sound_power.contains(&h(180)));
        assert!(!sound_power.contains(&v(0)));
        assert!(!sound_power.contains(&v(180)));
    }

    #[cfg(feature = "cli")]
    #[test]
    fn test_sound_power_weights_are_spherical_slices() {
        let weights = spinorama_sound_power_weights();
        let sound_power = sound_power_rows();
        let total: f64 = sound_power.iter().map(|&row| weights[row]).sum();

        assert!((total - 2.0).abs() < 1e-12);
        assert!(weights[h(90)] > weights[h(10)]);
        assert_eq!(weights[v(0)], 0.0);
        assert_eq!(weights[v(180)], 0.0);
    }

    // Fixture fills the deprecated per-driver vectors alongside the matrix.
    #[allow(deprecated)]
    #[cfg(feature = "cli")]
    #[test]
    fn test_listening_window_includes_on_axis() {
        let frequencies = synthetic_spinorama_result().frequencies;
        let n_freqs = frequencies.len();
        let mut spl = HostMatrix::<f64>::zeros(n_freqs, TOTAL_EVAL_POINTS);
        for row in 0..n_freqs {
            spl.row_mut(row).fill(80.0);
            spl[(row, 0)] = 100.0;
        }
        let phase = HostMatrix::<f64>::zeros(frequencies.len(), TOTAL_EVAL_POINTS);
        let result = SimulationResult {
            frequencies,
            complex_pressure: HostMatrix::zeros(spl.nrows(), spl.ncols()),
            spl,
            phase,
            woofer_electrical_impedance_ohm: vec![None; 1],
            tweeter_electrical_impedance_ohm: vec![None; 1],
            driver_electrical_impedances_ohm: vec![vec![None; 2]; 1],
            radiation_matrices: vec![None; 1],
            radiation_matrix_diagnostics: vec![None; 1],
            thermal_driver_states: vec![[None, None]; 1],
            driver_thermal_states: vec![vec![None; 2]; 1],
            power_ledgers: vec![None; 1],
            product_reference_sweep: None,
            structural_responses: Vec::new(),
            driver_velocities_bem: vec![vec![]; n_freqs],
            exact_residual_evaluations: 0,
            provenance: Default::default(),
        };

        let metrics = compute_spinorama_metrics(&result).unwrap();
        let expected = rms_spl_db(&[100.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0]);

        assert_close(metrics.on_axis[0], 100.0, 1e-12);
        assert_close(metrics.listening_window[0], expected, 1e-12);
    }

    // Fixture fills the deprecated per-driver vectors alongside the matrix.
    #[allow(deprecated)]
    #[cfg(feature = "cli")]
    #[test]
    fn test_spinorama_metrics_rejects_insufficient_grid_instead_of_returning_nan() {
        let result = SimulationResult {
            frequencies: vec![1000.0],
            spl: HostMatrix::<f64>::zeros(1, TOTAL_EVAL_POINTS),
            phase: HostMatrix::<f64>::zeros(1, TOTAL_EVAL_POINTS),
            complex_pressure: HostMatrix::zeros(1, TOTAL_EVAL_POINTS),
            woofer_electrical_impedance_ohm: vec![None],
            tweeter_electrical_impedance_ohm: vec![None],
            driver_electrical_impedances_ohm: vec![vec![None; 2]],
            radiation_matrices: vec![None],
            radiation_matrix_diagnostics: vec![None],
            thermal_driver_states: vec![[None, None]],
            driver_thermal_states: vec![vec![None; 2]],
            power_ledgers: vec![None],
            product_reference_sweep: None,
            structural_responses: Vec::new(),
            driver_velocities_bem: vec![vec![]],
            exact_residual_evaluations: 0,
            provenance: Default::default(),
        };
        let error = compute_spinorama_metrics(&result).unwrap_err();
        assert!(error.contains("at least two"), "got: {error}");
    }

    #[cfg(feature = "cli")]
    #[test]
    fn test_compute_spinorama_metrics_matches_reference() {
        let result = synthetic_spinorama_result();
        let got = compute_spinorama_metrics(&result).unwrap();

        let freq = result.frequencies.clone();
        let (spl_h, spl_v) = spl_planes_for_autoeq(&result);
        let angles = autoeq_angle_labels();
        let spinorama_input = sonium_roomeq::cea2034::ValidatedSpinorama::new(
            &freq,
            &angles,
            &spl_h,
            &angles,
            &spl_v,
            sonium_roomeq::cea2034::SpinoramaAngle {
                plane: sonium_roomeq::cea2034::SpinoramaPlane::Horizontal,
                degrees: 0.0,
            },
        )
        .expect("valid labelled synthetic spinorama input");
        let idx = spinorama_index_groups();
        let weights = spinorama_sound_power_weights();
        let peq = vec![0.0f64; freq.len()];
        let intervals = sonium_roomeq::cea2034::octave_intervals(2, &freq);
        let (spinorama, expected_metrics) =
            sonium_roomeq::cea2034::score_peq(&spinorama_input, &idx, &intervals, &weights, &peq)
                .expect("valid synthetic spinorama input");

        assert_vec_close(&got.on_axis, &spl_h[AUTOEQ_ON_AXIS_ROW], 1e-12);
        assert_vec_close(&got.listening_window, &spinorama[0], 1e-12);
        assert_vec_close(&got.early_reflections, &spinorama[1], 1e-12);
        assert_vec_close(&got.sound_power, &spinorama[idx.len() - 1], 1e-12);
        assert_vec_close(&got.pir, &spinorama[idx.len()], 1e-12);
        assert_close(got.preference_score, expected_metrics.pref_score, 1e-12);
    }

    #[cfg(feature = "cli")]
    fn synthetic_reference_fixture() -> (Vec<f64>, SpinoramaMetrics, Vec<MeasuredSpinoramaPoint>) {
        let frequencies = vec![100.0, 200.0, 400.0];
        let metrics = SpinoramaMetrics {
            on_axis: vec![80.0, 81.0, 82.0],
            listening_window: vec![79.0, 80.0, 81.0],
            early_reflections: vec![78.0, 79.0, 80.0],
            sound_power: vec![77.0, 78.0, 79.0],
            pir: vec![76.0, 77.0, 78.0],
            preference_score: 4.0,
        };
        let reference = frequencies
            .iter()
            .enumerate()
            .map(|(index, &frequency_hz)| MeasuredSpinoramaPoint {
                frequency_hz,
                on_axis_db: metrics.on_axis[index],
                listening_window_db: metrics.listening_window[index],
                early_reflections_db: metrics.early_reflections[index],
                sound_power_db: metrics.sound_power[index],
                pir_db: metrics.pir[index],
            })
            .collect();
        (frequencies, metrics, reference)
    }

    #[cfg(feature = "cli")]
    #[test]
    fn spinorama_reference_correlation_reports_curve_and_score_mismatches() {
        let (frequencies, metrics, reference) = synthetic_reference_fixture();
        let tolerances = SpinoramaReferenceTolerances {
            curve_rmse_db: 1.0e-12,
            curve_max_db: 1.0e-12,
            preference_score_max: 1.0e-12,
        };
        let report =
            correlate_spinorama_reference(&frequencies, &metrics, &reference, 4.0, tolerances)
                .expect("matching spinorama reference should pass");
        assert!(report.passed);
        assert_eq!(report.on_axis.samples, frequencies.len());

        let mut mismatch = reference.clone();
        mismatch[1].pir_db += 0.1;
        let report =
            correlate_spinorama_reference(&frequencies, &metrics, &mismatch, 4.2, tolerances)
                .expect("mismatched spinorama reference should report");
        assert!(!report.passed);
        assert!(!report.pir.passed);
        assert!(!report.preference_score_passed);
    }

    #[cfg(feature = "cli")]
    #[test]
    fn spinorama_reference_correlation_rejects_malformed_inputs() {
        let (mut frequencies, metrics, reference) = synthetic_reference_fixture();
        frequencies[1] = frequencies[0];
        assert!(
            correlate_spinorama_reference(
                &frequencies,
                &metrics,
                &reference,
                4.0,
                SpinoramaReferenceTolerances::default(),
            )
            .is_err()
        );

        let (frequencies, metrics, mut reference) = synthetic_reference_fixture();
        reference[0].on_axis_db = f64::NAN;
        assert!(
            correlate_spinorama_reference(
                &frequencies,
                &metrics,
                &reference,
                4.0,
                SpinoramaReferenceTolerances::default(),
            )
            .is_err()
        );

        let mut invalid_tolerances = SpinoramaReferenceTolerances::default();
        invalid_tolerances.curve_max_db = -1.0;
        let (_, _, reference) = synthetic_reference_fixture();
        assert!(
            correlate_spinorama_reference(
                &frequencies,
                &metrics,
                &reference,
                4.0,
                invalid_tolerances,
            )
            .is_err()
        );
    }

    #[cfg(feature = "cli")]
    #[test]
    fn spinorama_reference_report_round_trips_through_serde() {
        let (_, metrics, reference) = synthetic_reference_fixture();
        let frequencies = reference
            .iter()
            .map(|point| point.frequency_hz)
            .collect::<Vec<_>>();
        let report = correlate_spinorama_reference(
            &frequencies,
            &metrics,
            &reference,
            4.0,
            SpinoramaReferenceTolerances::default(),
        )
        .expect("reference report should be valid");
        let encoded = serde_json::to_string(&report).expect("serialize spinorama report");
        let decoded: SpinoramaReferenceReport =
            serde_json::from_str(&encoded).expect("deserialize spinorama report");
        assert_eq!(decoded, report);
    }

    // Fixture fills the deprecated per-driver vectors alongside the matrix.
    #[allow(deprecated)]
    #[cfg(feature = "cli")]
    fn synthetic_spinorama_result() -> SimulationResult {
        let frequencies = vec![
            80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0,
            1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0,
            12500.0,
        ];
        let n_freqs = frequencies.len();
        let spl = HostMatrix::<f64>::from_fn(n_freqs, TOTAL_EVAL_POINTS, |fi, point| {
            let plane_loss = if point < ANGLES_PER_PLANE { 0.0 } else { 0.2 };
            84.0 + fi as f64 * 0.05 - point as f64 * 0.01 - plane_loss
        });
        let phase = HostMatrix::<f64>::zeros(n_freqs, TOTAL_EVAL_POINTS);

        SimulationResult {
            woofer_electrical_impedance_ohm: vec![None; frequencies.len()],
            tweeter_electrical_impedance_ohm: vec![None; frequencies.len()],
            driver_electrical_impedances_ohm: vec![vec![None; 2]; frequencies.len()],
            radiation_matrices: vec![None; frequencies.len()],
            radiation_matrix_diagnostics: vec![None; frequencies.len()],
            thermal_driver_states: vec![[None, None]; frequencies.len()],
            driver_thermal_states: vec![vec![None; 2]; frequencies.len()],
            power_ledgers: vec![None; frequencies.len()],
            product_reference_sweep: None,
            structural_responses: Vec::new(),
            driver_velocities_bem: vec![vec![]; frequencies.len()],
            exact_residual_evaluations: 0,
            provenance: Default::default(),
            frequencies,
            complex_pressure: HostMatrix::zeros(spl.nrows(), spl.ncols()),
            spl,
            phase,
        }
    }

    #[cfg(feature = "cli")]
    fn rms_spl_db(values: &[f64]) -> f64 {
        let mean_p2 = values
            .iter()
            .map(|spl| 10f64.powf((spl - 105.0) / 10.0))
            .sum::<f64>()
            / values.len() as f64;
        10.0 * mean_p2.log10() + 105.0
    }

    #[cfg(feature = "cli")]
    fn assert_vec_close(actual: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_close_msg(
                actual,
                expected,
                tolerance,
                &format!("index {i}: {actual} != {expected}"),
            );
        }
    }

    #[cfg(feature = "cli")]
    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert_close_msg(actual, expected, tolerance, "");
    }

    #[cfg(feature = "cli")]
    fn assert_close_msg(actual: f64, expected: f64, tolerance: f64, context: &str) {
        if actual.is_nan() && expected.is_nan() {
            return;
        }
        assert!(
            (actual - expected).abs() <= tolerance,
            "{context}: {actual} != {expected}"
        );
    }
}
