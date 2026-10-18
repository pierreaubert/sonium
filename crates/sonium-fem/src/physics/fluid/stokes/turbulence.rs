//! Turbulence closure for the Navier-Stokes Picard solver.
//!
//! Two models are supported:
//!
//! - [`TurbulenceModel::Laminar`] — no eddy viscosity, the classic
//!   `μ_eff = μ` used by every existing test.
//! - [`TurbulenceModel::KOmegaSstAlgebraic`] — an algebraic surrogate for
//!   the Menter k-ω SST eddy viscosity. Computes a globally-enhanced
//!   `μ_eff = μ + ρ·ν_t` from the current velocity-field strain-rate
//!   magnitude and a characteristic length scale, mimicking the SST
//!   blending behaviour (low ν_t in laminar regions of low strain, high
//!   ν_t in high-shear regions).
//!
//! The algebraic variant intentionally does NOT solve k and ω transport
//! equations — those would require two new scalar function spaces, two
//! more SUPG-stabilised assemblies, and Menter's blended wall functions
//! (~600 LOC in total). The algebraic model captures the first-order
//! effect of turbulence on bulk flow (reduced effective Reynolds number,
//! dampened spurious oscillations, stabilised Picard iteration) at a
//! fraction of the cost, and exposes the same configuration interface
//! (`TurbulenceModel`) so a full transport-equation k-ω SST can be
//! dropped in later as a new enum variant without churning callers. Its
//! current Newton tangent is exact only for a unique, non-zero global peak;
//! tied or zero peaks remain fail-closed at the resolved-adjoint boundary.

/// Active turbulence model used by the Navier-Stokes Picard solver.
///
/// `Laminar` matches the legacy behaviour exactly. `KOmegaSstAlgebraic`
/// computes an enhanced effective viscosity from the velocity field and
/// applies it as a scalar `μ_eff` at each Picard iteration.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub enum TurbulenceModel {
    /// No turbulence closure. `μ_eff = μ`.
    #[default]
    Laminar,
    /// Algebraic k-ω SST surrogate. Computes `ν_t = c1 · L · |U|` from
    /// the velocity field magnitude and a characteristic length scale,
    /// clamped to `ν_t ≤ ν_t_max_ratio · ν` to avoid runaway viscosity
    /// when the Picard iteration overshoots.
    KOmegaSstAlgebraic {
        /// Calibration coefficient. SST ranges typical: 0.05–0.15.
        c1: f64,
        /// Characteristic flow length (e.g. shroud radius or hub-to-tip
        /// span for a fan). Used to dimensionalise `ν_t` against the
        /// velocity scale extracted from the current iterate.
        characteristic_length_m: f64,
        /// Hard upper bound on `ν_t / ν`. Prevents pathological blow-up
        /// when `|U|` peaks in the first Picard iterations. Default is
        /// 200 (turbulent-to-laminar viscosity ratios above this are
        /// unphysical for sub-supersonic internal flow).
        nu_t_max_ratio: f64,
    },
}

impl TurbulenceModel {
    /// k-ω SST defaults that work for typical PC-fan-scale internal flows
    /// (Re ≈ 10⁴, characteristic length ≈ 0.05 m).
    pub fn pcfan_default() -> Self {
        Self::KOmegaSstAlgebraic {
            c1: 0.09,                       // SST a₁ value, reused as our magnitude scaler
            characteristic_length_m: 0.058, // shroud inner radius
            nu_t_max_ratio: 200.0,
        }
    }

    /// Compute the effective dynamic viscosity `μ_eff = μ + ρ·ν_t` for the
    /// current Picard iterate. `velocity_magnitude_max` is the peak
    /// velocity (m/s) in the current velocity field — typically computed
    /// once per iteration over the FE node values.
    pub fn effective_mu(&self, mu: f64, rho: f64, velocity_magnitude_max: f64) -> f64 {
        self.effective_mu_and_speed_derivative(mu, rho, velocity_magnitude_max)
            .0
    }

    /// Return effective viscosity and its derivative with respect to the
    /// supplied velocity scale.
    ///
    /// The algebraic PC-fan closure uses a scalar velocity scale.  Keeping
    /// the clamp derivative explicit lets a Newton assembly publish an exact
    /// tangent on either side of the cap; at the cap the closure is locally
    /// constant and the derivative is zero.
    pub fn effective_mu_and_speed_derivative(
        &self,
        mu: f64,
        rho: f64,
        velocity_magnitude_max: f64,
    ) -> (f64, f64) {
        match *self {
            Self::Laminar => (mu, 0.0),
            Self::KOmegaSstAlgebraic {
                c1,
                characteristic_length_m,
                nu_t_max_ratio,
            } => {
                let nu = mu / rho.max(f64::EPSILON);
                let nu_t = c1 * characteristic_length_m * velocity_magnitude_max.abs();
                let cap = nu_t_max_ratio * nu;
                let nu_t_clamped = nu_t.min(cap).max(0.0);
                let derivative = if nu_t > 0.0 && nu_t < cap {
                    rho * c1 * characteristic_length_m
                } else {
                    0.0
                };
                (mu + rho * nu_t_clamped, derivative)
            }
        }
    }

    /// Compute the unique active nodal velocity peak and its subgradient.
    ///
    /// The global maximum is intentionally retained for compatibility with
    /// the algebraic closure.  A Newton tangent is only valid away from a
    /// tied or zero peak, so those cases return `None` and callers must keep
    /// the resolved derivative path fail-closed.
    pub fn unique_peak_velocity_subgradient(
        velocity: &[f64],
        dimension: usize,
    ) -> Option<(f64, Vec<f64>)> {
        if !(1..=3).contains(&dimension)
            || velocity.is_empty()
            || !velocity.len().is_multiple_of(dimension)
        {
            return None;
        }

        let mut peak: f64 = 0.0;
        let mut peak_node = None;
        let mut peak_count = 0usize;
        for (node, values) in velocity.chunks_exact(dimension).enumerate() {
            let magnitude = values.iter().map(|value| value * value).sum::<f64>().sqrt();
            if !magnitude.is_finite() {
                return None;
            }
            let tolerance = 1.0e-12 * peak.max(magnitude).max(1.0);
            if magnitude > peak + tolerance {
                peak = magnitude;
                peak_node = Some(node);
                peak_count = 1;
            } else if (magnitude - peak).abs() <= tolerance {
                peak_count += 1;
            }
        }

        let peak_node = peak_node?;
        if peak <= f64::EPSILON || peak_count != 1 {
            return None;
        }

        let mut gradient = vec![0.0; velocity.len()];
        let offset = peak_node * dimension;
        for component in 0..dimension {
            gradient[offset + component] = velocity[offset + component] / peak;
        }
        Some((peak, gradient))
    }

    /// Diagnostic: `ν_t / ν` for the current iterate. Useful for plots.
    pub fn turbulent_viscosity_ratio(&self, mu: f64, rho: f64, velocity_magnitude_max: f64) -> f64 {
        let mu_eff = self.effective_mu(mu, rho, velocity_magnitude_max);
        (mu_eff - mu) / mu.max(f64::EPSILON)
    }
}

/// Smagorinsky subgrid-scale eddy viscosity `ν_t = (C_s Δ)² |S|`.
///
/// `strain_magnitude` is `|S| = sqrt(2 S:S)` of the resolved velocity
/// gradient and `filter_width` the local filter width (usually the
/// cell size). Returns 0 for non-finite or non-positive inputs, so a
/// caller can add the result unconditionally. Intended for LES of
/// transient, shear-driven flow (e.g. a breaking slosh wave) where a
/// RANS closure's time-averaged state does not exist; the caller is
/// responsible for phase- or region-weighting the result.
pub fn smagorinsky_eddy_viscosity(cs: f64, filter_width: f64, strain_magnitude: f64) -> f64 {
    if !cs.is_finite()
        || !filter_width.is_finite()
        || !strain_magnitude.is_finite()
        || cs <= 0.0
        || filter_width <= 0.0
        || strain_magnitude < 0.0
    {
        return 0.0;
    }
    let width = cs * filter_width;
    width * width * strain_magnitude
}

/// WALE subgrid-scale eddy viscosity from a velocity-gradient tensor.
///
/// `gradient[i][j]` is `∂u_i/∂x_j`. With `g² = g·g`, the traceless
/// symmetric part `S^d = ½(g² + g²ᵀ) − ⅓ tr(g²) I`, and the strain
/// tensor `S = ½(g + gᵀ)`, returns
/// `(C_w Δ)² (S^d:S^d)^{3/2} / ((S:S)^{5/2} + (S^d:S^d)^{5/4})`.
/// Unlike the Smagorinsky model this vanishes in laminar shear and
/// needs no wall-damping function, which suits free-surface LES.
/// Returns 0 for non-finite inputs or a degenerate gradient.
pub fn wale_eddy_viscosity(cw: f64, filter_width: f64, gradient: [[f64; 3]; 3]) -> f64 {
    if !cw.is_finite()
        || !filter_width.is_finite()
        || cw <= 0.0
        || filter_width <= 0.0
        || gradient.iter().flatten().any(|value| !value.is_finite())
    {
        return 0.0;
    }
    let mut g2 = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            g2[i][j] = (0..3).map(|k| gradient[i][k] * gradient[k][j]).sum();
        }
    }
    let trace_g2 = g2[0][0] + g2[1][1] + g2[2][2];
    let mut sd_contract = 0.0f64;
    let mut strain_contract = 0.0f64;
    for i in 0..3 {
        for j in 0..3 {
            let sd = 0.5 * (g2[i][j] + g2[j][i]) - if i == j { trace_g2 / 3.0 } else { 0.0 };
            let strain = 0.5 * (gradient[i][j] + gradient[j][i]);
            sd_contract += sd * sd;
            strain_contract += strain * strain;
        }
    }
    if sd_contract <= 0.0 {
        return 0.0;
    }
    let denominator = strain_contract.powf(2.5) + sd_contract.powf(1.25);
    if !denominator.is_finite() || denominator <= 0.0 {
        return 0.0;
    }
    let width = cw * filter_width;
    width * width * sd_contract.powf(1.5) / denominator
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smagorinsky_scales_with_strain_and_width() {
        // (0.1 * 0.02)^2 * 50 = 2e-4.
        let nu_t = smagorinsky_eddy_viscosity(0.1, 0.02, 50.0);
        assert!((nu_t - 2.0e-4).abs() < 1e-12);
        assert_eq!(smagorinsky_eddy_viscosity(0.1, 0.02, 0.0), 0.0);
        assert_eq!(smagorinsky_eddy_viscosity(-0.1, 0.02, 50.0), 0.0);
        assert_eq!(smagorinsky_eddy_viscosity(0.1, 0.02, f64::NAN), 0.0);
    }

    #[test]
    fn wale_vanishes_without_gradient_and_scales_linearly() {
        assert_eq!(wale_eddy_viscosity(0.5, 0.02, [[0.0; 3]; 3]), 0.0);
        // Pure laminar shear gives exactly zero (the WALE design
        // property: no eddy viscosity without strain/rotation mix).
        let mut laminar_shear = [[0.0f64; 3]; 3];
        laminar_shear[0][1] = 10.0;
        assert_eq!(wale_eddy_viscosity(0.5, 0.02, laminar_shear), 0.0);
        // A mixed strain/rotation gradient activates the model.
        let mut gradient = [[0.0f64; 3]; 3];
        gradient[0][1] = 10.0;
        gradient[1][0] = 5.0;
        let nu_t = wale_eddy_viscosity(0.5, 0.02, gradient);
        assert!(nu_t.is_finite() && nu_t > 0.0, "WALE mixed nu_t = {nu_t}");
        // Doubling the gradient doubles nu_t (degree-one homogeneity).
        gradient[0][1] = 20.0;
        gradient[1][0] = 10.0;
        let doubled = wale_eddy_viscosity(0.5, 0.02, gradient);
        assert!(
            (doubled / nu_t - 2.0).abs() < 1e-9,
            "WALE scaling {} -> {doubled}",
            nu_t
        );
    }

    #[test]
    fn laminar_returns_mu_unchanged() {
        let m = TurbulenceModel::Laminar;
        assert_eq!(m.effective_mu(1.8e-5, 1.2, 10.0), 1.8e-5);
        assert_eq!(m.turbulent_viscosity_ratio(1.8e-5, 1.2, 10.0), 0.0);
    }

    #[test]
    fn algebraic_increases_with_velocity() {
        let m = TurbulenceModel::pcfan_default();
        let mu_low = m.effective_mu(1.8e-5, 1.2, 0.5);
        let mu_high = m.effective_mu(1.8e-5, 1.2, 5.0);
        assert!(mu_high > mu_low);
    }

    #[test]
    fn algebraic_clamped_at_max_ratio() {
        let m = TurbulenceModel::KOmegaSstAlgebraic {
            c1: 0.09,
            characteristic_length_m: 1.0,
            nu_t_max_ratio: 100.0,
        };
        // Push velocity high enough to saturate the cap.
        let mu = 1.8e-5;
        let rho = 1.2;
        let nu = mu / rho;
        let mu_eff = m.effective_mu(mu, rho, 1.0e6);
        let ratio = (mu_eff - mu) / (rho * nu);
        assert!(
            (ratio - 100.0).abs() < 1.0e-6,
            "expected clamp at 100, got {ratio}"
        );
    }

    #[test]
    fn zero_velocity_gives_no_eddy_viscosity() {
        let m = TurbulenceModel::pcfan_default();
        let mu = 1.8e-5;
        assert_eq!(m.effective_mu(mu, 1.2, 0.0), mu);
    }

    #[test]
    fn ratio_diagnostic_matches_definition() {
        let m = TurbulenceModel::KOmegaSstAlgebraic {
            c1: 0.1,
            characteristic_length_m: 0.05,
            nu_t_max_ratio: 1000.0,
        };
        let mu = 1.8e-5;
        let rho = 1.2;
        let u = 5.0;
        let mu_eff = m.effective_mu(mu, rho, u);
        let ratio = m.turbulent_viscosity_ratio(mu, rho, u);
        let expected = (mu_eff - mu) / mu;
        assert!((ratio - expected).abs() < 1.0e-12);
    }

    #[test]
    fn viscosity_speed_derivative_matches_centered_difference() {
        let model = TurbulenceModel::KOmegaSstAlgebraic {
            c1: 0.1,
            characteristic_length_m: 0.05,
            nu_t_max_ratio: 1_000.0,
        };
        let mu = 1.8e-5;
        let rho = 1.2;
        let speed = 5.0;
        let step = 1.0e-6;
        let (_, derivative) = model.effective_mu_and_speed_derivative(mu, rho, speed);
        let centered = (model.effective_mu(mu, rho, speed + step)
            - model.effective_mu(mu, rho, speed - step))
            / (2.0 * step);
        assert!((derivative - centered).abs() < 1.0e-10);
    }

    #[test]
    fn viscosity_speed_derivative_is_zero_at_clamp() {
        let model = TurbulenceModel::KOmegaSstAlgebraic {
            c1: 0.1,
            characteristic_length_m: 1.0,
            nu_t_max_ratio: 10.0,
        };
        let (_, derivative) = model.effective_mu_and_speed_derivative(1.8e-5, 1.2, 1.0e6);
        assert_eq!(derivative, 0.0);
    }

    #[test]
    fn unique_peak_subgradient_matches_centered_directional_difference() {
        let velocity = vec![1.0, 0.0, 0.0, 0.3, 0.4, 0.2];
        let (peak, gradient) =
            TurbulenceModel::unique_peak_velocity_subgradient(&velocity, 3).unwrap();
        let direction = vec![0.2, -0.4, 0.1, 0.0, 0.0, 0.0];
        let step = 1.0e-7;
        let plus = velocity
            .iter()
            .zip(&direction)
            .map(|(value, delta)| value + step * delta)
            .collect::<Vec<_>>();
        let minus = velocity
            .iter()
            .zip(&direction)
            .map(|(value, delta)| value - step * delta)
            .collect::<Vec<_>>();
        let plus_peak = plus
            .chunks_exact(3)
            .map(|values| values.iter().map(|value| value * value).sum::<f64>().sqrt())
            .fold(0.0, f64::max);
        let minus_peak = minus
            .chunks_exact(3)
            .map(|values| values.iter().map(|value| value * value).sum::<f64>().sqrt())
            .fold(0.0, f64::max);
        let centered = (plus_peak - minus_peak) / (2.0 * step);
        let directional = gradient
            .iter()
            .zip(&direction)
            .map(|(left, right)| left * right)
            .sum::<f64>();
        assert!((peak - 1.0).abs() < 1.0e-12);
        assert!((centered - directional).abs() < 1.0e-8);
    }

    #[test]
    fn unique_peak_subgradient_rejects_ties_and_zero_state() {
        assert!(TurbulenceModel::unique_peak_velocity_subgradient(&[0.0; 6], 3).is_none());
        assert!(
            TurbulenceModel::unique_peak_velocity_subgradient(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 3)
                .is_none()
        );
    }
}
