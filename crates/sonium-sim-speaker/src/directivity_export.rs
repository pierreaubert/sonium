//! Export of the BEM-computed complex far-field directivity as a
//! [`ComplexDirectivityGrid`] for the room models (`sonium-xem` /
//! `sonium-sim-room`).
//!
//! Frame convention. The speaker simulation frame is `x` = right, `y` = up,
//! `z` = forward (the acoustic axis, see
//! [`crate::model::LoudspeakerModel::acoustic_axis`]). The xem directivity
//! grid evaluates patterns in the room source's local frame, where `+Y` is
//! forward, `+Z` is up, `+X` is right, with `theta` the polar angle from `+Z`
//! and `phi` the azimuth from `+X` towards `+Y`. The export therefore maps a
//! speaker-frame direction `d = (x, y, z)` to grid angles
//!
//! ```text
//! theta = acos(y / r)        (polar from the speaker's up axis)
//! phi   = atan2(z, x)        (azimuth from right towards forward)
//! ```
//!
//! so the on-axis (baffle-normal) direction is `theta = 90°, phi = 90°`, which
//! is exactly the xem local forward direction. In the room, the source's
//! `orientation` must be set to the speaker's acoustic axis in world
//! coordinates; `mirrored()` then behaves as for any other grid.
//!
//! Normalization. Each frequency's pattern is divided by the complex on-axis
//! pressure sample, so the exported grid is dimensionless and exactly `1+0i`
//! on axis at every frequency — the same "pattern multiplier" role as the
//! magnitude [`sonium_xem::DirectivityGrid`] (whose cardioid peaks at 1.0).
//! The absolute level stays with the room source's `amplitude` (times its own
//! broadband spectrum), so a room source reaches comparable levels whether it
//! is fed an L0/L1 magnitude grid or this complex grid.
//!
//! Crossover semantics. The exported full-system pattern already contains the
//! speaker's crossover magnitude *and* phase and the complex multi-driver
//! interference (the BEM sim drives the typed drivers through their network).
//! A room source using this grid must therefore NOT also apply its own
//! `CrossoverFilter` — that would double-count the crossover. (Chained use is
//! not hard-blocked because cascading e.g. an additional room-EQ filter can be
//! legitimate; `sonium-sim-room` logs a warning for the combination.)
//!
//! Coverage. The measurement points must form a complete rectangular angular
//! grid (`n_vertical × n_horizontal`) at a uniform radius about the origin and
//! include the on-axis direction. CEA-2034-style arc sets (two circles) are
//! NOT sufficient; use [`spherical_eval_points`] to generate a suitable
//! full-sphere measurement set. Pole rows (theta = 0°/180°) are special:
//! azimuth is undefined there, so the samples on a pole row are averaged and
//! fill the entire row of the exported grid (the duplicated pole points that
//! [`spherical_eval_points`] emits are therefore expected, not an error).

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use sonium_compute::HostMatrix;
use sonium_xem::ComplexDirectivityGrid;

use crate::simulation::{SimulationError, SimulationProvenance, SimulationResult};

/// Angular tolerance (degrees) used when recovering the grid axes from the
/// evaluation-point coordinates.
const ANGLE_TOLERANCE_DEG: f64 = 1e-6;
/// Snap distance (degrees) from the poles. Azimuth is undefined at the poles
/// and unrecoverable from absolute coordinates (the horizontal offset of a
/// pole sample is smaller than the floating-point rounding of the absolute
/// position), so pole samples fill their entire elevation row.
const POLE_SNAP_DEG: f64 = 1e-9;
/// Relative radius spread tolerated across the measurement sphere.
const RADIUS_RELATIVE_TOLERANCE: f64 = 1e-6;

/// Complex directivity together with the product-simulation contract that
/// produced it.  The contained grid remains directly consumable by XEM while
/// callers that persist or promote an artifact retain backend, precision,
/// residual-gate, and dynamic-driver provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComplexDirectivityArtifact {
    #[serde(flatten)]
    pub grid: ComplexDirectivityGrid,
    pub frequencies_hz: Vec<f64>,
    /// `None` denotes a legacy prescribed-velocity result without dynamic
    /// typed-driver evidence; otherwise every frequency row had this count.
    pub driver_count: Option<usize>,
    pub provenance: SimulationProvenance,
}

fn dynamic_driver_count(result: &SimulationResult) -> Result<Option<usize>, SimulationError> {
    if result.driver_velocities_bem.len() != result.frequencies.len() {
        return Err(SimulationError::new(
            "directivity artifact driver-velocity evidence does not match the frequency grid",
        ));
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
        return Err(SimulationError::new(
            "directivity artifact mixes missing and dynamic driver-velocity evidence",
        ));
    }
    let driver_count = first.len();
    if result.driver_velocities_bem.iter().any(|velocities| {
        velocities.len() != driver_count
            || velocities
                .iter()
                .any(|velocity| !velocity.re.is_finite() || !velocity.im.is_finite())
    }) {
        return Err(SimulationError::new(
            "directivity artifact has malformed dynamic driver-velocity evidence",
        ));
    }
    Ok(Some(driver_count))
}

/// Export a provenance-preserving directivity product artifact.
pub fn export_complex_directivity_artifact(
    result: &SimulationResult,
    eval_points: &[[f64; 3]],
    origin: [f64; 3],
) -> Result<ComplexDirectivityArtifact, SimulationError> {
    let driver_count = dynamic_driver_count(result)?;
    let grid = export_complex_directivity(result, eval_points, origin)?;
    Ok(ComplexDirectivityArtifact {
        grid,
        frequencies_hz: result.frequencies.clone(),
        driver_count,
        provenance: result.provenance.clone(),
    })
}

/// Generate measurement points on a full-sphere angular grid about `origin`,
/// ordered row-major `[n_vertical][n_horizontal]` to match the grid pattern
/// layout.
///
/// Angles are in degrees and follow the grid convention in the speaker frame:
/// `theta` (vertical) is the polar angle from the up axis (`+y`), `phi`
/// (horizontal) is the azimuth from right (`+x`) towards forward (`+z`), so
/// the on-axis direction is `(90°, 90°)`. Include `90°` in both axes and
/// `0°`/`180°` in the vertical axis for full polar coverage.
pub fn spherical_eval_points(
    origin: [f64; 3],
    radius: f64,
    horizontal_angles_deg: &[f64],
    vertical_angles_deg: &[f64],
) -> Vec<[f64; 3]> {
    let mut points = Vec::with_capacity(horizontal_angles_deg.len() * vertical_angles_deg.len());
    for &theta_deg in vertical_angles_deg {
        let theta = theta_deg.to_radians();
        let (sin_theta, cos_theta) = theta.sin_cos();
        for &phi_deg in horizontal_angles_deg {
            let phi = phi_deg.to_radians();
            let (sin_phi, cos_phi) = phi.sin_cos();
            points.push([
                origin[0] + radius * sin_theta * cos_phi,
                origin[1] + radius * cos_theta,
                origin[2] + radius * sin_theta * sin_phi,
            ]);
        }
    }
    points
}

/// Build the [`ComplexDirectivityGrid`] from a finished simulation.
///
/// `eval_points` must be the exact points the simulation was run with; their
/// directions about `origin` (the acoustic origin, typically
/// [`crate::model::LoudspeakerModel::baffle_center`]) are recovered and
/// validated against the coverage rules documented in the module docs.
pub fn export_complex_directivity(
    result: &SimulationResult,
    eval_points: &[[f64; 3]],
    origin: [f64; 3],
) -> Result<ComplexDirectivityGrid, SimulationError> {
    let n_freqs = result.frequencies.len();
    if result.complex_pressure.shape() != (n_freqs, eval_points.len()) {
        return Err(SimulationError::new(format!(
            "complex pressure shape {:?} does not match {} frequencies x {} evaluation points",
            result.complex_pressure.shape(),
            n_freqs,
            eval_points.len()
        )));
    }
    if eval_points.len() < 4 {
        return Err(SimulationError::new(
            "directivity export requires a full-sphere angular grid of evaluation points",
        ));
    }

    // Recover (theta, phi, r) for every evaluation point in the speaker
    // frame. Pole samples are snapped to exactly 0°/180°; their azimuth is
    // physically undefined (and numerically unrecoverable from absolute
    // coordinates), so they fill their whole elevation row.
    let mut theta_deg = Vec::with_capacity(eval_points.len());
    let mut phi_deg = Vec::with_capacity(eval_points.len());
    let mut radius = Vec::with_capacity(eval_points.len());
    let mut is_pole = Vec::with_capacity(eval_points.len());
    for point in eval_points {
        let d = [
            point[0] - origin[0],
            point[1] - origin[1],
            point[2] - origin[2],
        ];
        let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if !r.is_finite() || r <= 0.0 {
            return Err(SimulationError::new(
                "evaluation point coincides with the directivity origin",
            ));
        }
        let mut theta = (d[1] / r).clamp(-1.0, 1.0).acos().to_degrees();
        let mut pole = false;
        if theta <= POLE_SNAP_DEG || 180.0 - theta <= POLE_SNAP_DEG {
            theta = if theta <= POLE_SNAP_DEG { 0.0 } else { 180.0 };
            pole = true;
        }
        theta_deg.push(theta);
        phi_deg.push(d[2].atan2(d[0]).to_degrees().rem_euclid(360.0));
        radius.push(r);
        is_pole.push(pole);
    }
    let mean_radius = radius.iter().sum::<f64>() / radius.len() as f64;
    let max_deviation = radius
        .iter()
        .map(|r| (r - mean_radius).abs())
        .fold(0.0_f64, f64::max);
    if max_deviation > RADIUS_RELATIVE_TOLERANCE * mean_radius {
        return Err(SimulationError::new(format!(
            "evaluation points must lie on a sphere about the origin: radius spread {max_deviation:.3e} m around mean {mean_radius:.3} m"
        )));
    }

    let vertical_angles = unique_axis(&theta_deg);
    // The horizontal axis comes from interior (non-pole) samples only.
    let horizontal_angles = unique_axis(
        &phi_deg
            .iter()
            .zip(&is_pole)
            .filter(|&(_, &pole)| !pole)
            .map(|(&phi, _)| phi)
            .collect::<Vec<_>>(),
    );
    let n_vertical = vertical_angles.len();
    let n_horizontal = horizontal_angles.len();
    if n_vertical < 2 || n_horizontal < 2 {
        return Err(SimulationError::new(
            "directivity export requires at least two vertical and two horizontal angle samples (CEA-2034 arc sets are not sufficient; use spherical_eval_points)",
        ));
    }

    // Map each interior point to its (vertical, horizontal) cell, requiring
    // full, duplicate-free coverage of the interior rows. Pole samples are
    // collected per pole row instead.
    let mut cell_of_point = vec![usize::MAX; eval_points.len()];
    let mut covered = vec![false; n_vertical * n_horizontal];
    let mut pole_points: Vec<Vec<usize>> = vec![Vec::new(); n_vertical];
    for (index, (&theta, &phi)) in theta_deg.iter().zip(&phi_deg).enumerate() {
        let v = axis_index(&vertical_angles, theta).ok_or_else(|| {
            SimulationError::new(format!(
                "evaluation point {index} has elevation {theta:.6}° outside the recovered vertical axis"
            ))
        })?;
        if is_pole[index] {
            pole_points[v].push(index);
            continue;
        }
        let h = axis_index(&horizontal_angles, phi).ok_or_else(|| {
            SimulationError::new(format!(
                "evaluation point {index} has azimuth {phi:.6}° outside the recovered horizontal axis"
            ))
        })?;
        let cell = v * n_horizontal + h;
        if covered[cell] {
            return Err(SimulationError::new(format!(
                "duplicate evaluation point at (theta={theta:.6}°, phi={phi:.6}°)"
            )));
        }
        covered[cell] = true;
        cell_of_point[index] = cell;
    }
    for (v, &theta) in vertical_angles.iter().enumerate() {
        if theta == 0.0 || theta == 180.0 {
            if pole_points[v].is_empty() {
                return Err(SimulationError::new(format!(
                    "pole row theta={theta}° has no evaluation samples"
                )));
            }
            continue;
        }
        for h in 0..n_horizontal {
            if !covered[v * n_horizontal + h] {
                return Err(SimulationError::new(format!(
                    "angular grid has no sample at (theta={theta:.6}°, phi={:.6}°)",
                    horizontal_angles[h]
                )));
            }
        }
    }

    // On-axis reference: theta=90°, phi=90° must be a sampled interior node.
    let on_axis_cell = axis_index(&vertical_angles, 90.0)
        .and_then(|v| axis_index(&horizontal_angles, 90.0).map(|h| v * n_horizontal + h))
        .ok_or_else(|| {
            SimulationError::new(
                "angular grid must include the on-axis direction (theta=90°, phi=90°) for normalization",
            )
        })?;
    let on_axis_point = cell_of_point
        .iter()
        .position(|&cell| cell == on_axis_cell)
        .ok_or_else(|| {
            SimulationError::new(
                "angular grid on-axis cell was not covered after validating the recovered axes",
            )
        })?;

    let patterns: Vec<HostMatrix<Complex64>> = (0..n_freqs)
        .map(|fi| {
            let reference = result.complex_pressure[[fi, on_axis_point]];
            if !reference.is_finite() || reference.norm() <= 0.0 {
                return Err(SimulationError::new(format!(
                    "on-axis pressure at {:.3} Hz is zero or non-finite; cannot normalize the directivity pattern",
                    result.frequencies[fi]
                )));
            }
            let mut pattern = HostMatrix::zeros(n_vertical, n_horizontal);
            for (point_index, &cell) in cell_of_point.iter().enumerate() {
                if cell != usize::MAX {
                    pattern[[cell / n_horizontal, cell % n_horizontal]] =
                        result.complex_pressure[[fi, point_index]] / reference;
                }
            }
            // Pole rows: fill every azimuth with the mean of that row's
            // (physically identical) samples.
            for (v, indices) in pole_points.iter().enumerate() {
                if indices.is_empty() {
                    continue;
                }
                let mean = indices
                    .iter()
                    .map(|&point_index| result.complex_pressure[[fi, point_index]])
                    .sum::<Complex64>()
                    / indices.len() as f64
                    / reference;
                for h in 0..n_horizontal {
                    pattern[[v, h]] = mean;
                }
            }
            Ok(pattern)
        })
        .collect::<Result<_, SimulationError>>()?;

    ComplexDirectivityGrid::try_new(
        horizontal_angles,
        vertical_angles,
        result.frequencies.clone(),
        patterns,
    )
    .map_err(|error| SimulationError::new(format!("invalid directivity grid: {error}")))
}

/// Sorted, deduplicated angle axis recovered from sample values.
fn unique_axis(values: &[f64]) -> Vec<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted.dedup_by(|a, b| (*a - *b).abs() <= ANGLE_TOLERANCE_DEG);
    sorted
}

fn axis_index(axis: &[f64], value: f64) -> Option<usize> {
    axis.iter()
        .position(|angle| (*angle - value).abs() <= ANGLE_TOLERANCE_DEG)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::LoudspeakerModel;
    use crate::simulation::{run_simulation, tests::tiny_model};
    use sonium_xem::{Directivity, Point3D, Source};

    fn export_test_model() -> LoudspeakerModel {
        let mut model = tiny_model();
        model.simulation.freq_start = 500.0;
        model.simulation.freq_end = 1000.0;
        model.simulation.freq_points = 2;
        model
    }

    fn export_axes() -> (Vec<f64>, Vec<f64>) {
        let horizontal: Vec<f64> = (0..8).map(|i| i as f64 * 45.0).collect();
        let vertical: Vec<f64> = (0..5).map(|i| i as f64 * 45.0).collect();
        (horizontal, vertical)
    }

    struct ExportFixture {
        result: SimulationResult,
        grid: ComplexDirectivityGrid,
        horizontal: Vec<f64>,
        vertical: Vec<f64>,
    }

    fn run_export_fixture() -> ExportFixture {
        let model = export_test_model();
        let origin = model.baffle_center();
        let (horizontal, vertical) = export_axes();
        let points = spherical_eval_points(origin, 1.0, &horizontal, &vertical);
        let result = run_simulation(&model, &points).expect("simulation should succeed");
        let grid =
            export_complex_directivity(&result, &points, origin).expect("export should succeed");
        ExportFixture {
            result,
            grid,
            horizontal,
            vertical,
        }
    }

    #[test]
    fn artifact_retains_dynamic_driver_and_simulation_provenance() {
        let origin = [0.0, 0.0, 0.14];
        let (horizontal, vertical) = export_axes();
        let points = spherical_eval_points(origin, 1.0, &horizontal, &vertical);
        let mut result = synthetic_result(2, points.len());
        result.driver_velocities_bem = result
            .frequencies
            .iter()
            .map(|_| {
                vec![
                    Complex64::new(1.0e-3, 0.0),
                    Complex64::new(0.5e-3, 0.25e-3),
                    Complex64::new(0.25e-3, -0.5e-3),
                ]
            })
            .collect();
        let provenance = result.provenance.clone();

        let artifact = export_complex_directivity_artifact(&result, &points, origin)
            .expect("artifact export should preserve valid dynamic evidence");
        assert_eq!(artifact.driver_count, Some(3));
        assert_eq!(artifact.frequencies_hz, result.frequencies);
        assert_eq!(artifact.provenance, provenance);
        assert_eq!(artifact.grid.frequencies(), result.frequencies.as_slice());

        let json = serde_json::to_string(&artifact).expect("artifact should serialize");
        assert!(json.contains("frequencies_hz"));
        assert!(json.contains("driver_count"));
        let decoded: ComplexDirectivityArtifact =
            serde_json::from_str(&json).expect("artifact should round-trip");
        assert_eq!(decoded, artifact);
        // The flattened representation remains compatible with the legacy grid
        // consumer, which ignores the additional provenance fields.
        let legacy_grid: ComplexDirectivityGrid =
            serde_json::from_str(&json).expect("legacy grid should read artifact");
        assert_eq!(legacy_grid.frequencies(), result.frequencies.as_slice());
    }

    #[test]
    fn artifact_rejects_mismatched_dynamic_driver_evidence() {
        let origin = [0.0, 0.0, 0.14];
        let (horizontal, vertical) = export_axes();
        let points = spherical_eval_points(origin, 1.0, &horizontal, &vertical);
        let mut result = synthetic_result(2, points.len());
        result.driver_velocities_bem = vec![
            vec![Complex64::new(1.0e-3, 0.0)],
            vec![Complex64::new(1.0e-3, 0.0), Complex64::new(1.0e-3, 0.0)],
        ];
        assert!(export_complex_directivity_artifact(&result, &points, origin).is_err());
    }

    #[test]
    fn spherical_eval_points_follow_grid_convention() {
        let origin = [0.0, 0.0, 0.14];
        let points = spherical_eval_points(origin, 1.0, &[90.0, 0.0], &[90.0, 0.0, 180.0]);
        // (theta=90°, phi=90°): forward (+z in the speaker frame).
        assert!((points[0][2] - 1.14).abs() < 1e-12 && points[0][0].abs() < 1e-12);
        // (theta=90°, phi=0°): right (+x).
        assert!((points[1][0] - 1.0).abs() < 1e-12 && (points[1][2] - 0.14).abs() < 1e-12);
        // (theta=0°): up (+y); (theta=180°): down.
        assert!((points[2][1] - 1.0).abs() < 1e-12);
        assert!((points[4][1] + 1.0).abs() < 1e-12);
    }

    #[test]
    fn export_reproduces_sim_pressures_on_grid_nodes() {
        let fixture = run_export_fixture();
        let grid = &fixture.grid;
        let assert_axis_matches = |recovered: &[f64], expected: &[f64]| {
            assert_eq!(recovered.len(), expected.len());
            for (&a, &b) in recovered.iter().zip(expected) {
                assert!((a - b).abs() < 1e-9, "recovered axis {a} vs expected {b}");
            }
        };
        assert_axis_matches(grid.horizontal_angles(), &fixture.horizontal);
        assert_axis_matches(grid.vertical_angles(), &fixture.vertical);
        assert_eq!(grid.frequencies(), fixture.result.frequencies.as_slice());

        let n_horizontal = fixture.horizontal.len();
        for (fi, &frequency) in fixture.result.frequencies.iter().enumerate() {
            for (vi, &theta) in fixture.vertical.iter().enumerate() {
                for (hi, &phi) in fixture.horizontal.iter().enumerate() {
                    let point_index = vi * n_horizontal + hi;
                    let node_value = grid.patterns()[fi][[vi, hi]];
                    // Exact at nodes: the exported value reproduces the sim's
                    // complex pressure up to the (shared) on-axis scale.
                    let interpolated =
                        grid.interpolate(theta.to_radians(), phi.to_radians(), frequency);
                    let scale = node_value.norm().max(1e-12);
                    assert!(
                        (interpolated - node_value).norm() < 1e-9 * scale,
                        "f={frequency} theta={theta} phi={phi}: interp={interpolated} node={node_value}"
                    );
                    // Cross-check against the raw pressure ratio: same
                    // magnitude and phase (the on-axis reference cancels).
                    let pressure_ratio = fixture.result.complex_pressure[[fi, point_index]]
                        / fixture.result.complex_pressure[[fi, on_axis_index(&fixture)]];
                    assert!(
                        (node_value - pressure_ratio).norm() < 1e-12,
                        "f={frequency} theta={theta} phi={phi}"
                    );
                }
            }
            // On-axis sample is exactly 1+0i by construction.
            let on_axis = grid.interpolate(90.0_f64.to_radians(), 90.0_f64.to_radians(), frequency);
            assert!((on_axis - Complex64::new(1.0, 0.0)).norm() < 1e-12);
        }

        // Between nodes: bilinear combination of complex corners is a convex
        // combination, so the value is finite and bounded by the corner norms.
        let f = fixture.result.frequencies[0];
        let mid = grid.interpolate(22.5_f64.to_radians(), 22.5_f64.to_radians(), f);
        assert!(mid.is_finite());
        let corner_norm_max = [0, 1]
            .iter()
            .flat_map(|&dv| [0, 1].iter().map(move |&dh| (dv, dh)))
            .map(|(dv, dh)| grid.patterns()[0][[dv, dh]].norm())
            .fold(0.0_f64, f64::max);
        assert!(mid.norm() <= corner_norm_max + 1e-12);
        // Between frequency samples: piecewise-linear, must stay finite.
        let f_mid = 0.5 * (fixture.result.frequencies[0] + fixture.result.frequencies[1]);
        assert!(
            grid.interpolate(90.0_f64.to_radians(), 45.0_f64.to_radians(), f_mid)
                .is_finite()
        );
    }

    fn on_axis_index(fixture: &ExportFixture) -> usize {
        // theta=90° is vertical index 2, phi=90° is horizontal index 2.
        2 * fixture.horizontal.len() + 2
    }

    /// Convention test: the exported grid, consumed through the room's
    /// `Source` machinery with the default orientation (world `+Y` forward,
    /// `+Z` up, `+X` right), must reproduce the sim's complex field direction
    /// at sample directions — this catches axis/origin/mirroring mistakes.
    #[test]
    fn exported_grid_matches_room_source_convention() {
        let fixture = run_export_fixture();
        let source = Source::new(
            Point3D::new(0.0, 0.0, 0.0),
            Directivity::ComplexGrid(fixture.grid.clone()),
            1.0,
        );
        let n_horizontal = fixture.horizontal.len();
        for (vi, &theta_deg) in fixture.vertical.iter().enumerate() {
            for (hi, &phi_deg) in fixture.horizontal.iter().enumerate() {
                let theta = theta_deg.to_radians();
                let phi = phi_deg.to_radians();
                // Room world point for grid direction (theta, phi) under the
                // default orientation: local == world, forward = +Y.
                let world = Point3D::new(
                    theta.sin() * phi.cos(),
                    theta.sin() * phi.sin(),
                    theta.cos(),
                );
                for (fi, &frequency) in fixture.result.frequencies.iter().enumerate() {
                    let actual = source.complex_amplitude_towards(&world, frequency);
                    let point_index = vi * n_horizontal + hi;
                    let expected = fixture.result.complex_pressure[[fi, point_index]]
                        / fixture.result.complex_pressure[[fi, on_axis_index(&fixture)]];
                    let pole = theta_deg <= 1e-9 || (180.0 - theta_deg) <= 1e-9;
                    // At the poles the room interpolator averages the
                    // (numerically identical) duplicated samples; node
                    // pressures agree to BEM accuracy, so compare loosely.
                    let tolerance = if pole { 1e-6 } else { 1e-9 };
                    assert!(
                        (actual - expected).norm() < tolerance,
                        "theta={theta_deg} phi={phi_deg} f={frequency}: source={actual} sim={expected}"
                    );
                }
            }
        }
        // Physical sanity: the baffled speaker radiates forward, so the
        // on-axis lobe must dominate the rear direction at 1 kHz.
        let rear = fixture.grid.interpolate(
            90.0_f64.to_radians(),
            270.0_f64.to_radians(),
            *fixture.result.frequencies.last().unwrap(),
        );
        assert!(
            rear.norm() < 1.0,
            "rear radiation {} should be below the normalized on-axis level",
            rear.norm()
        );
    }

    // Fixture fills the deprecated per-driver vectors alongside the matrix.
    #[allow(deprecated)]
    /// Synthetic result for coverage-validation tests (no BEM solve needed).
    fn synthetic_result(n_freqs: usize, n_points: usize) -> SimulationResult {
        SimulationResult {
            frequencies: (0..n_freqs).map(|i| 500.0 + 100.0 * i as f64).collect(),
            spl: HostMatrix::zeros(n_freqs, n_points),
            phase: HostMatrix::zeros(n_freqs, n_points),
            complex_pressure: HostMatrix::from_elem(n_freqs, n_points, Complex64::new(1.0, 0.0)),
            woofer_electrical_impedance_ohm: vec![None; n_freqs],
            tweeter_electrical_impedance_ohm: vec![None; n_freqs],
            driver_electrical_impedances_ohm: vec![vec![None; 2]; n_freqs],
            radiation_matrices: vec![None; n_freqs],
            radiation_matrix_diagnostics: vec![None; n_freqs],
            thermal_driver_states: vec![[None, None]; n_freqs],
            driver_thermal_states: vec![vec![None; 2]; n_freqs],
            power_ledgers: vec![None; n_freqs],
            product_reference_sweep: None,
            structural_responses: Vec::new(),
            driver_velocities_bem: vec![vec![]; n_freqs],
            exact_residual_evaluations: 0,
            provenance: Default::default(),
        }
    }

    #[test]
    fn export_rejects_arc_only_coverage() {
        let origin = [0.0, 0.0, 0.14];
        // CEA-2034-style horizontal arc only: no vertical coverage.
        let arc = spherical_eval_points(origin, 1.0, &[0.0, 90.0, 180.0, 270.0], &[90.0]);
        let result = synthetic_result(1, arc.len());
        assert!(export_complex_directivity(&result, &arc, origin).is_err());
    }

    #[test]
    fn export_rejects_grid_without_on_axis_sample() {
        let origin = [0.0, 0.0, 0.14];
        // Full rectangular grid but the vertical axis skips theta=90°.
        let horizontal: Vec<f64> = (0..8).map(|i| i as f64 * 45.0).collect();
        let vertical = vec![0.0, 45.0, 135.0, 180.0];
        let points = spherical_eval_points(origin, 1.0, &horizontal, &vertical);
        let result = synthetic_result(1, points.len());
        assert!(export_complex_directivity(&result, &points, origin).is_err());
    }

    #[test]
    fn export_rejects_non_spherical_points() {
        let origin = [0.0, 0.0, 0.14];
        let (horizontal, vertical) = export_axes();
        let mut points = spherical_eval_points(origin, 1.0, &horizontal, &vertical);
        points[3][2] += 0.5; // break the uniform radius
        let result = synthetic_result(1, points.len());
        assert!(export_complex_directivity(&result, &points, origin).is_err());
    }
}
