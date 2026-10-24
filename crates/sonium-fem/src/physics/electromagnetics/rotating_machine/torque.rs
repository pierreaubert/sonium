//! Electromagnetic torque from a solved `A_z` field.
//!
//! Two field-only evaluators share the polar Maxwell stress `B_r B_phi`:
//! Arkkio's annulus average (robust, the default diagnostic) and the
//! flux-linkage energy mean over an electrical period (the headline mean
//! torque for transient runs, where single-frame stress integrals on a
//! sliding band carry mesh noise).

// Rust guideline compliant 2026-02-21

use super::super::magnetostatics::MU_0;
use super::az::AzQuadraturePoint;

/// Arkkio torque over an air-gap annulus, in N m (sector value).
///
/// `T = L / (mu_0 (r_out - r_in)) * integral r B_r B_phi dA`, integrated
/// over the quadrature points of `state` whose radius lies in
/// `[r_in_m, r_out_m]`. The caller multiplies by the number of modelled
/// sectors. Returns `0.0` when the annulus contains no quadrature point
/// or is degenerate; check [`annulus_point_count`] when that matters.
pub fn arkkio_torque_nm(
    state: &[AzQuadraturePoint],
    r_in_m: f64,
    r_out_m: f64,
    stack_length_m: f64,
) -> f64 {
    if !(r_out_m > r_in_m && r_in_m >= 0.0 && stack_length_m > 0.0) {
        return 0.0;
    }
    let mut integral = 0.0;
    for point in state {
        let [x, y] = point.position_m;
        let radius = x.hypot(y);
        if radius < r_in_m || radius > r_out_m || radius <= 0.0 {
            continue;
        }
        let (cos_phi, sin_phi) = (x / radius, y / radius);
        let [bx, by] = point.flux_density_t;
        let b_r = bx * cos_phi + by * sin_phi;
        let b_phi = -bx * sin_phi + by * cos_phi;
        integral += radius * b_r * b_phi * point.integration_weight_m2;
    }
    stack_length_m / (MU_0 * (r_out_m - r_in_m)) * integral
}

/// Number of quadrature points of `state` inside `[r_in_m, r_out_m]`.
pub fn annulus_point_count(state: &[AzQuadraturePoint], r_in_m: f64, r_out_m: f64) -> usize {
    state
        .iter()
        .filter(|point| {
            let radius = point.position_m[0].hypot(point.position_m[1]);
            (r_in_m..=r_out_m).contains(&radius)
        })
        .count()
}

/// Mean torque from terminal flux linkages over one electrical period.
///
/// For imposed phase currents the converted energy per period is
/// `sum_phase oint i dpsi`, so with `p` pole pairs
/// `T_mean = p / (2 pi) * sum_phase sum_k i_k (psi_{k+1} - psi_k)`
/// (trapezoidal in `psi`, currents sampled at frame starts). Each slice
/// in `flux_linkages` and `currents` is one phase over the same uniform
/// frames of one electrical period, without a duplicated closing frame.
///
/// # Errors
///
/// Returns [`TorqueError`] when phase counts or frame counts disagree.
pub fn flux_linkage_mean_torque_nm(
    flux_linkages_wb_turn: &[&[f64]],
    currents_a: &[&[f64]],
    pole_pairs: usize,
) -> Result<f64, TorqueError> {
    if pole_pairs == 0 {
        return Err(TorqueError::InvalidPolePairs);
    }
    if flux_linkages_wb_turn.len() != currents_a.len() || flux_linkages_wb_turn.is_empty() {
        return Err(TorqueError::PhaseCountMismatch);
    }
    let frames = flux_linkages_wb_turn[0].len();
    if frames < 2
        || flux_linkages_wb_turn.iter().any(|psi| psi.len() != frames)
        || currents_a.iter().any(|current| current.len() != frames)
    {
        return Err(TorqueError::FrameCountMismatch);
    }
    let mut work = 0.0;
    for (psi, current) in flux_linkages_wb_turn.iter().zip(currents_a.iter()) {
        for frame in 0..frames {
            let next = (frame + 1) % frames;
            // Midpoint current against the linkage increment is second
            // order on a uniform period grid.
            let midpoint_current = 0.5 * (current[frame] + current[next]);
            work += midpoint_current * (psi[next] - psi[frame]);
        }
    }
    Ok(pole_pairs as f64 / (2.0 * std::f64::consts::PI) * work)
}

/// Terminal-work mean torque: `n_parallel * <sum_phases i * dpsi/dtheta_m>`.
///
/// The derivative is the full-DFT spectral derivative on the uniform,
/// endpoint-excluded mechanical-angle grid (the even-count Nyquist bin
/// contributes nothing, as its sine quadrature is not represented by
/// the samples). This is the reference solver's headline mean for
/// imposed-current runs; the discrete `sum i * delta(psi)` form in
/// [`flux_linkage_mean_torque_nm`] differs from it by aliasing at
/// coarse step counts.
///
/// # Errors
///
/// Returns [`TorqueError::FrameCountMismatch`] for mismatched series or
/// fewer than eight frames, and [`TorqueError::NonUniformAngles`] when
/// the angle grid is not uniform or does not span integer electrical
/// periods.
pub fn terminal_work_mean_nm(
    psi_wb_turn: &[&[f64]; 3],
    currents_branch_a: &[&[f64]; 3],
    mechanical_angle_rad: &[f64],
    pole_pairs: usize,
    n_parallel: f64,
) -> Result<f64, TorqueError> {
    let frames = mechanical_angle_rad.len();
    if frames < 8
        || psi_wb_turn.iter().any(|series| series.len() != frames)
        || currents_branch_a
            .iter()
            .any(|series| series.len() != frames)
    {
        return Err(TorqueError::FrameCountMismatch);
    }
    if pole_pairs == 0 {
        return Err(TorqueError::InvalidPolePairs);
    }
    let step = mechanical_angle_rad[1] - mechanical_angle_rad[0];
    if !step.is_finite()
        || step == 0.0
        || mechanical_angle_rad
            .windows(2)
            .any(|pair| (pair[1] - pair[0] - step).abs() > 1.0e-10 * step.abs() + 1.0e-14)
    {
        return Err(TorqueError::NonUniformAngles);
    }
    let periods = (frames as f64 * step * pole_pairs as f64 / (2.0 * std::f64::consts::PI)).abs();
    if (periods - periods.round()).abs() > 1.0e-9 || periods.round() < 1.0 {
        return Err(TorqueError::NonUniformAngles);
    }
    let mut mean = 0.0;
    for phase in 0..3 {
        let derivative = spectral_derivative(psi_wb_turn[phase], step);
        mean += currents_branch_a[phase]
            .iter()
            .zip(&derivative)
            .map(|(i, dpsi)| i * dpsi)
            .sum::<f64>()
            / frames as f64;
    }
    Ok(n_parallel * mean)
}

/// Space-vector (Clarke) mean torque `(3/2) p n_par <psi_a i_b - psi_b i_a>`.
///
/// The reference solver's fallback headline for non-current-drive runs.
/// Exact for sinusoidal quantities; a diagnostic next to
/// [`terminal_work_mean_nm`] otherwise.
pub fn space_vector_mean_nm(
    psi_wb_turn: &[&[f64]; 3],
    currents_branch_a: &[&[f64]; 3],
    pole_pairs: usize,
    n_parallel: f64,
) -> f64 {
    let frames = psi_wb_turn[0].len();
    if frames == 0 {
        return 0.0;
    }
    let s = 2.0 / 3.0;
    let kc = (3.0f64).sqrt() / 2.0;
    let mut sum = 0.0;
    for frame in 0..frames {
        let (pa, pb, pc) = (
            psi_wb_turn[0][frame],
            psi_wb_turn[1][frame],
            psi_wb_turn[2][frame],
        );
        let (ia, ib, ic) = (
            currents_branch_a[0][frame],
            currents_branch_a[1][frame],
            currents_branch_a[2][frame],
        );
        let psi_alpha = s * (pa - 0.5 * pb - 0.5 * pc);
        let psi_beta = s * kc * (pb - pc);
        let i_alpha = s * (ia - 0.5 * ib - 0.5 * ic);
        let i_beta = s * kc * (ib - ic);
        sum += psi_alpha * i_beta - psi_beta * i_alpha;
    }
    1.5 * pole_pairs as f64 * n_parallel * sum / frames as f64
}

/// Spectral derivative of a periodic series on a uniform grid.
fn spectral_derivative(values: &[f64], step: f64) -> Vec<f64> {
    let n = values.len();
    // DFT coefficients.
    let mut re = vec![0.0; n];
    let mut im = vec![0.0; n];
    for (k, (re_k, im_k)) in re.iter_mut().zip(&mut im).enumerate() {
        for (j, &value) in values.iter().enumerate() {
            let angle = -2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64;
            *re_k += value * angle.cos();
            *im_k += value * angle.sin();
        }
    }
    // Multiply by i*omega with fftfreq ordering; take real inverse DFT.
    let mut out = vec![0.0; n];
    for (j, value) in out.iter_mut().enumerate() {
        let mut acc = 0.0;
        for k in 0..n {
            let freq = if k <= n / 2 {
                k as f64
            } else {
                k as f64 - n as f64
            };
            let omega = 2.0 * std::f64::consts::PI * freq / (n as f64 * step);
            // (i * omega) * (re + i im) = -omega*im + i*omega*re
            let (dre, dim) = (-omega * im[k], omega * re[k]);
            let angle = 2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64;
            acc += dre * angle.cos() - dim * angle.sin();
        }
        *value = acc / n as f64;
    }
    out
}

/// Peak-to-peak ripple of a torque series, as `(mean, ripple_pp, pct)`.
///
/// The percentage is relative to `|mean|` and is `f64::INFINITY` for a
/// zero-mean series (cogging), matching how cogging must be reported:
/// absolute peak-to-peak, never a percentage.
pub fn torque_metrics(series_nm: &[f64]) -> TorqueMetrics {
    if series_nm.is_empty() {
        return TorqueMetrics {
            mean_nm: 0.0,
            ripple_pp_nm: 0.0,
            ripple_pct: 0.0,
        };
    }
    let mean = series_nm.iter().sum::<f64>() / series_nm.len() as f64;
    let max = series_nm.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min = series_nm.iter().copied().fold(f64::INFINITY, f64::min);
    let ripple = max - min;
    TorqueMetrics {
        mean_nm: mean,
        ripple_pp_nm: ripple,
        ripple_pct: if mean.abs() > 0.0 {
            100.0 * ripple / mean.abs()
        } else if ripple > 0.0 {
            f64::INFINITY
        } else {
            0.0
        },
    }
}

/// Mean and ripple summary of a torque series.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TorqueMetrics {
    /// Mean torque in N m.
    pub mean_nm: f64,
    /// Peak-to-peak ripple in N m.
    pub ripple_pp_nm: f64,
    /// Ripple as a percentage of `|mean|`.
    pub ripple_pct: f64,
}

/// Invalid flux-linkage torque input.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TorqueError {
    /// Pole-pair count was zero.
    #[error("pole-pair count must be positive")]
    InvalidPolePairs,
    /// Flux-linkage and current phase counts disagreed or were empty.
    #[error("flux-linkage and current phase counts must agree and be non-empty")]
    PhaseCountMismatch,
    /// Frame counts disagreed or numbered fewer than two.
    #[error("all phases need the same frame count, at least two")]
    FrameCountMismatch,
    /// Mechanical angles were not a uniform integer-period grid.
    #[error("mechanical angles must be uniform and span integer electrical periods")]
    NonUniformAngles,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn point(position: [f64; 2], b: [f64; 2], weight: f64) -> AzQuadraturePoint {
        AzQuadraturePoint {
            element_index: 0,
            quadrature_index: 0,
            marker: 0,
            position_m: position,
            integration_weight_m2: weight,
            vector_potential_wb_per_m: 0.0,
            flux_density_t: b,
            field_strength_a_per_m: [0.0, 0.0],
            reluctivity_m_per_h: 0.0,
            differential_reluctivity_m_per_h: 0.0,
            energy_density_j_per_m3: 0.0,
            coenergy_density_j_per_m3: 0.0,
            imposed_current_density_a_per_m2: 0.0,
            eddy_current_density_a_per_m2: 0.0,
        }
    }

    #[test]
    fn arkkio_of_pure_radial_field_is_zero() {
        // B purely radial at angle 0: B_phi = 0, so no torque.
        let state = vec![point([0.05, 0.0], [1.0, 0.0], 1.0e-6)];
        assert_eq!(arkkio_torque_nm(&state, 0.04, 0.06, 0.1), 0.0);
    }

    #[test]
    fn arkkio_matches_hand_value_for_tilted_field() {
        // At phi = 0 with B = [Br, Bphi]: T = L/(mu0 dr) * r Br Bphi w.
        let state = vec![point([0.05, 0.0], [0.8, 0.3], 2.0e-6)];
        let expected = 0.1 / (MU_0 * 0.02) * (0.05 * 0.8 * 0.3 * 2.0e-6);
        let actual = arkkio_torque_nm(&state, 0.04, 0.06, 0.1);
        assert!((actual - expected).abs() / expected < 1.0e-12);
    }

    #[test]
    fn flux_linkage_torque_matches_analytic_sinusoid() {
        // One phase: psi = Psi cos(theta_e), i = I sin(theta_e) over a
        // period. oint i dpsi = -pi * Psi * I, so T = -p Psi I / 2.
        let frames = 720;
        let psi: Vec<f64> = (0..frames)
            .map(|k| 0.4 * (2.0 * PI * k as f64 / frames as f64).cos())
            .collect();
        let current: Vec<f64> = (0..frames)
            .map(|k| 10.0 * (2.0 * PI * k as f64 / frames as f64).sin())
            .collect();
        let torque = flux_linkage_mean_torque_nm(&[&psi], &[&current], 7).unwrap();
        let expected = -7.0 * 0.4 * 10.0 / 2.0;
        assert!((torque - expected).abs() < 1.0e-3, "{torque} vs {expected}");
    }

    #[test]
    fn terminal_work_and_space_vector_match_analytic_sinusoid() {
        // psi_A = Psi cos(p*theta), 120-degree shifted phases; currents
        // pure q-axis: i_A = I sin(p*theta) etc. Converted power gives
        // T = -(3/2) p Psi I / ... evaluate against the closed form of
        // sum i dpsi/dtheta_m: per phase <I sin * (-p Psi sin)> = -p Psi I/2,
        // three phases => -(3/2) p Psi I.
        let frames = 360;
        let p = 7usize;
        let theta: Vec<f64> = (0..frames)
            .map(|k| 2.0 * PI * k as f64 / (p as f64 * frames as f64))
            .collect();
        let make = |shift: f64, sine: bool| -> Vec<f64> {
            theta
                .iter()
                .map(|t| {
                    let a = p as f64 * t + shift;
                    if sine { 10.0 * a.sin() } else { 0.4 * a.cos() }
                })
                .collect()
        };
        let psi = [
            make(0.0, false),
            make(-2.0 * PI / 3.0, false),
            make(2.0 * PI / 3.0, false),
        ];
        let cur = [
            make(0.0, true),
            make(-2.0 * PI / 3.0, true),
            make(2.0 * PI / 3.0, true),
        ];
        let psi_refs: [&[f64]; 3] = [&psi[0], &psi[1], &psi[2]];
        let cur_refs: [&[f64]; 3] = [&cur[0], &cur[1], &cur[2]];
        let expected = -1.5 * p as f64 * 0.4 * 10.0;
        let tw = terminal_work_mean_nm(&psi_refs, &cur_refs, &theta, p, 1.0).unwrap();
        assert!(
            (tw - expected).abs() < 1.0e-6,
            "terminal work {tw} vs {expected}"
        );
        let sv = space_vector_mean_nm(&psi_refs, &cur_refs, p, 1.0);
        // Space vector: (3/2) p <psi_alpha i_beta - psi_beta i_alpha>;
        // for this phasing it equals +(3/2) p Psi I /... sign follows
        // the psi-leading-i convention; magnitude must match.
        assert!(
            (sv.abs() - expected.abs()).abs() < 1.0e-6,
            "space vector {sv}"
        );
    }

    #[test]
    fn cogging_series_reports_infinite_percentage() {
        let metrics = torque_metrics(&[1.0, -1.0, 1.0, -1.0]);
        assert_eq!(metrics.mean_nm, 0.0);
        assert_eq!(metrics.ripple_pp_nm, 2.0);
        assert!(metrics.ripple_pct.is_infinite());
    }
}
