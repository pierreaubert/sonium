//! Spline port profile for shape optimisation (PORT-OPTIM.md Step 1).
//!
//! [`SplinePortProfile`] represents a 2D axisymmetric meridian profile `r(x)`,
//! `x in [0, L]`, as a clamped piecewise-cubic through fixed axial knots.
//! Only radii vary (normalised variables `u in [0, 1]`); the bore envelope
//! `r(x) <= R0`, the straight start `r'(0) = 0`, and the slope/curvature
//! budgets are enforced at construction and re-checked on a dense grid.
//!
//! Two interpolants are available ([`ProfileSplineKind`]). The default is the
//! shape-preserving clamped monotone cubic Hermite ([`PchipSpline`]): it
//! satisfies all four profile constraints (`r(0) = r(L) = R0`, `r'(0) = 0`,
//! `r_min <= r <= R0`) by construction for every in-range control polygon.
//! The C² clamped cubic ([`ClampedCubicSpline`]) is kept for smooth dense
//! data, but it provably overshoots the bore between knots on dipping
//! polygons (measured +4e-5 m above `R0` on a gentle `N = 5` dip), so
//! validation rejects such C² candidates. That measured violation is plan
//! risk R5 and activates decision D2: the shape-preserving variant is the
//! default carrier for optimisation, not a silent substitution — both modes
//! are tested, including the C² rejection.
//!
//! This module must not depend on `sonium-sim-speaker` (dependency direction:
//! the BEM product layer sits above the modeling crate). Both splines here
//! are intentionally local rather than reusing the natural spline in
//! `sonium-sim-speaker/src/spline.rs`, which has the wrong end conditions
//! (`r'' = 0` instead of `r' = 0`).
//!
//! # Validated envelope
//!
//! Pure geometry: construction, dense-grid validation, and sampling. No flow
//! solve, no noise claim. Flow/noise claims live with the tier evidence behind
//! them (2D/meridian Tier 2 at the converged timestep, cross-checked by the
//! Tier-1 spline oracle); the frozen campaign winner is handed to downstream
//! consumers through [`super::handoff`] with its evidence links.
//!
//! # Inward-only default and opt-in flare
//!
//! Normalised variables `u in [0, 1]` map to `r = R0 - u·(R0 - r_min)`, so
//! every candidate from [`SplinePortProfile::from_normalized`] stays inside
//! the original bore (`r(x) <= R0`) by construction and the dense grid
//! re-audits it. The frozen campaign used this mapping, so it stays the
//! default. [`SplinePortProfile::from_normalized_flared`] is the explicit
//! opt-in flare mapping: every knot is free, `u = 0` is still the straight
//! bore, negative `u` widens toward a caller-supplied flare cap above `R0`,
//! and positive `u` still pinches toward `r_min`.
//!
//! # Equal-airflow reporting rule
//!
//! A narrower port at equal voltage trivially shows less flow-borne noise, so
//! winner-vs-straight comparisons are always reported at matched airflow (a
//! rescaled-drive verification run with an honestly measured residual) next to
//! the equal-voltage deltas. Success Criterion 2 judges the equal-airflow
//! levels only; see [`super::campaign::EqualAirflowDelta`].

use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use super::error::PortFlowError;

/// Number of dense-grid samples used to validate every candidate profile.
///
/// Covers the `[0, L]` interval inclusive, so 201 points (the plan requires
/// at least 200).
pub const PROFILE_VALIDATION_GRID_POINTS: usize = 201;

/// Default maximum wall slope `|r'(x)|` (dimensionless).
pub const DEFAULT_SHAPE_SLOPE_MAX: f64 = 0.5;

/// Default curvature budget `∫(r'')² dx` in m⁻³.
///
/// Generous on purpose: a smooth dip to `0.6·R0` over a 0.1 m port measures
/// O(1) m⁻³, while kinked control polygons reach O(100) m⁻³. The optimisation
/// campaign is expected to tighten this; construction rejects anything above
/// the configured budget.
pub const DEFAULT_SHAPE_CURVATURE_BUDGET_M3: f64 = 1.0e4;

/// Relative tolerance (fraction of `R0`) for dense-grid bound checks.
///
/// Covers spline-evaluation roundoff only; genuine cubic overshoot between
/// knots is orders of magnitude larger and still rejected.
const BOUND_TOL_REL: f64 = 1.0e-9;

/// Absolute tolerance for the clamped-slope audits (dimensionless).
const SLOPE_TOL_ABS: f64 = 1.0e-9;

/// Axial knot placement for [`SplinePortProfile`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KnotSpacing {
    /// Uniform knots `x_i = L·i/(N-1)`.
    Uniform,
    /// Chebyshev knots of the second kind `x_i = L/2·(1 - cos(π·i/(N-1)))`,
    /// clustered toward the inlet/outlet where end conditions matter most.
    Chebyshev,
}

/// Interpolant used by [`SplinePortProfile`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ProfileSplineKind {
    /// C¹ clamped monotone cubic Hermite ([`PchipSpline`]): Fritsch–Carlson
    /// interior derivatives with zero end slopes. Shape-preserving, so
    /// in-range knots imply an in-range profile. Default (plan decision D2).
    #[default]
    ShapePreserving,
    /// C² clamped cubic spline ([`ClampedCubicSpline`]) with zero end slopes.
    /// Faithful for smooth dense data; overshoots between knots on dipping
    /// polygons, and validation rejects those candidates.
    ClampedCubic,
}

/// Manufacturability limits enforced at profile construction.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ShapeLimits {
    /// Maximum wall slope `|r'(x)|` (dimensionless).
    pub slope_max: f64,
    /// Maximum curvature budget `∫(r'')² dx` in m⁻³.
    pub curvature_budget_m3: f64,
}

impl Default for ShapeLimits {
    fn default() -> Self {
        Self {
            slope_max: DEFAULT_SHAPE_SLOPE_MAX,
            curvature_budget_m3: DEFAULT_SHAPE_CURVATURE_BUDGET_M3,
        }
    }
}

/// Construction options for [`SplinePortProfile`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProfileOptions {
    /// Axial knot placement.
    pub spacing: KnotSpacing,
    /// Slope/curvature limits enforced at construction.
    pub limits: ShapeLimits,
    /// Interpolant (default [`ProfileSplineKind::ShapePreserving`]).
    pub kind: ProfileSplineKind,
}

impl Default for ProfileOptions {
    fn default() -> Self {
        Self {
            spacing: KnotSpacing::Uniform,
            limits: ShapeLimits::default(),
            kind: ProfileSplineKind::default(),
        }
    }
}

/// A clamped cubic spline through control points with prescribed end slopes.
///
/// Interpolates `(x_i, y_i)` with C² piecewise cubics and `S'(x_0) = d_start`,
/// `S'(x_{n-1}) = d_end`. Queries outside the knot range are clamped to the
/// nearest end. C² continuity can force overshoot between knots on dipping
/// polygons; [`SplinePortProfile`] validation rejects such candidates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClampedCubicSpline {
    xs: Vec<f64>,
    ys: Vec<f64>,
    /// Second derivatives at the knots (tridiagonal solve).
    moments: Vec<f64>,
}

impl ClampedCubicSpline {
    /// Fit a clamped cubic spline through `(x, y)` points with end slopes.
    ///
    /// Returns [`PortFlowError::InvalidConfig`] when fewer than 2 points are
    /// given, any coordinate or slope is non-finite, or knots are not
    /// strictly increasing.
    pub fn try_from_control_points(
        pts: &[[f64; 2]],
        slope_start: f64,
        slope_end: f64,
    ) -> Result<Self, PortFlowError> {
        if pts.len() < 2 {
            return Err(PortFlowError::InvalidConfig(
                "clamped spline needs at least 2 control points".to_string(),
            ));
        }
        for (index, point) in pts.iter().enumerate() {
            for (coordinate, value) in point.iter().enumerate() {
                if !value.is_finite() {
                    return Err(PortFlowError::InvalidConfig(format!(
                        "clamped spline control point {index} coordinate {coordinate} must be finite"
                    )));
                }
            }
        }
        if !slope_start.is_finite() || !slope_end.is_finite() {
            return Err(PortFlowError::InvalidConfig(
                "clamped spline end slopes must be finite".to_string(),
            ));
        }
        if let Some(index) = pts.windows(2).position(|pair| pair[1][0] <= pair[0][0]) {
            return Err(PortFlowError::InvalidConfig(format!(
                "clamped spline knots must be strictly increasing at index {}",
                index + 1
            )));
        }

        let n = pts.len();
        let xs: Vec<f64> = pts.iter().map(|p| p[0]).collect();
        let ys: Vec<f64> = pts.iter().map(|p| p[1]).collect();
        let h: Vec<f64> = xs.windows(2).map(|pair| pair[1] - pair[0]).collect();

        // Tridiagonal system for the moments M_i = S''(x_i):
        //   interior: h[i-1]·M[i-1] + 2(h[i-1]+h[i])·M[i] + h[i]·M[i+1]
        //             = 6·((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])
        //   clamped ends: 2·h[0]·M[0] + h[0]·M[1] = 6·((y[1]-y[0])/h[0] - d0)
        //                 h[n-2]·M[n-2] + 2·h[n-2]·M[n-1]
        //             = 6·(d1 - (y[n-1]-y[n-2])/h[n-2])
        // The same stencil covers n == 2 (single Hermite interval).
        let mut lower = vec![0.0; n];
        let mut diag = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut rhs = vec![0.0; n];
        diag[0] = 2.0 * h[0];
        upper[0] = h[0];
        rhs[0] = 6.0 * ((ys[1] - ys[0]) / h[0] - slope_start);
        for i in 1..n - 1 {
            lower[i] = h[i - 1];
            diag[i] = 2.0 * (h[i - 1] + h[i]);
            upper[i] = h[i];
            rhs[i] = 6.0 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1]);
        }
        lower[n - 1] = h[n - 2];
        diag[n - 1] = 2.0 * h[n - 2];
        rhs[n - 1] = 6.0 * (slope_end - (ys[n - 1] - ys[n - 2]) / h[n - 2]);

        // Thomas algorithm (diagonally dominant: no pivoting needed).
        for i in 1..n {
            let factor = lower[i] / diag[i - 1];
            diag[i] -= factor * upper[i - 1];
            rhs[i] -= factor * rhs[i - 1];
        }
        let mut moments = vec![0.0; n];
        moments[n - 1] = rhs[n - 1] / diag[n - 1];
        for i in (0..n - 1).rev() {
            moments[i] = (rhs[i] - upper[i] * moments[i + 1]) / diag[i];
        }

        Ok(Self { xs, ys, moments })
    }

    /// Interval index holding `x` (clamped to the end intervals outside).
    fn interval(&self, x: f64) -> usize {
        let n = self.xs.len();
        if x <= self.xs[0] {
            return 0;
        }
        if x >= self.xs[n - 1] {
            return n - 2;
        }
        let upper = self.xs.partition_point(|knot| *knot <= x);
        upper.saturating_sub(1).min(n - 2)
    }

    /// Evaluate the spline value, clamping `x` to the knot range.
    pub fn value(&self, x: f64) -> f64 {
        let x = x.clamp(self.xs[0], self.xs[self.xs.len() - 1]);
        let i = self.interval(x);
        let h = self.xs[i + 1] - self.xs[i];
        let a = (self.xs[i + 1] - x) / h;
        let b = (x - self.xs[i]) / h;
        a * self.ys[i]
            + b * self.ys[i + 1]
            + (h * h / 6.0)
                * ((a * a * a - a) * self.moments[i] + (b * b * b - b) * self.moments[i + 1])
    }

    /// Evaluate the spline first derivative, clamping `x` to the knot range.
    pub fn derivative(&self, x: f64) -> f64 {
        let x = x.clamp(self.xs[0], self.xs[self.xs.len() - 1]);
        let i = self.interval(x);
        let h = self.xs[i + 1] - self.xs[i];
        let a = (self.xs[i + 1] - x) / h;
        let b = (x - self.xs[i]) / h;
        (self.ys[i + 1] - self.ys[i]) / h
            - (h / 6.0)
                * ((3.0 * a * a - 1.0) * self.moments[i]
                    - (3.0 * b * b - 1.0) * self.moments[i + 1])
    }

    /// Evaluate the spline second derivative, clamping `x` to the knot range.
    pub fn second_derivative(&self, x: f64) -> f64 {
        let x = x.clamp(self.xs[0], self.xs[self.xs.len() - 1]);
        let i = self.interval(x);
        let h = self.xs[i + 1] - self.xs[i];
        let a = (self.xs[i + 1] - x) / h;
        let b = (x - self.xs[i]) / h;
        a * self.moments[i] + b * self.moments[i + 1]
    }
}

/// A clamped shape-preserving cubic Hermite interpolant (PCHIP ends at zero).
///
/// Interior knot derivatives follow Fritsch–Carlson: zero at local extrema
/// and flat neighbours, otherwise the weighted harmonic mean of the adjacent
/// secants. Both end slopes are clamped to exactly zero (the straight
/// start/exit the profile requires) instead of the textbook one-sided
/// estimate. Zero ends keep the Fritsch–Carlson monotonicity certificate
/// (`α + β ≤ 3` with `α = 0` on the end intervals), so every interval is
/// monotone and the interpolant provably stays within the control-polygon
/// range — no overshoot between knots for in-range data.
///
/// Queries outside the knot range are clamped to the nearest end.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PchipSpline {
    xs: Vec<f64>,
    ys: Vec<f64>,
    /// First derivatives at the knots (zero ends, Fritsch–Carlson interior).
    ds: Vec<f64>,
}

impl PchipSpline {
    /// Fit the clamped shape-preserving interpolant through `(x, y)` points.
    ///
    /// Returns [`PortFlowError::InvalidConfig`] when fewer than 2 points are
    /// given, any coordinate is non-finite, or knots are not strictly
    /// increasing.
    pub fn try_from_control_points(pts: &[[f64; 2]]) -> Result<Self, PortFlowError> {
        if pts.len() < 2 {
            return Err(PortFlowError::InvalidConfig(
                "shape-preserving spline needs at least 2 control points".to_string(),
            ));
        }
        for (index, point) in pts.iter().enumerate() {
            for (coordinate, value) in point.iter().enumerate() {
                if !value.is_finite() {
                    return Err(PortFlowError::InvalidConfig(format!(
                        "shape-preserving spline control point {index} coordinate {coordinate} must be finite"
                    )));
                }
            }
        }
        if let Some(index) = pts.windows(2).position(|pair| pair[1][0] <= pair[0][0]) {
            return Err(PortFlowError::InvalidConfig(format!(
                "shape-preserving spline knots must be strictly increasing at index {}",
                index + 1
            )));
        }

        let n = pts.len();
        let xs: Vec<f64> = pts.iter().map(|p| p[0]).collect();
        let ys: Vec<f64> = pts.iter().map(|p| p[1]).collect();
        let h: Vec<f64> = xs.windows(2).map(|pair| pair[1] - pair[0]).collect();
        let secant: Vec<f64> = ys
            .windows(2)
            .zip(h.iter())
            .map(|(pair, h)| (pair[1] - pair[0]) / h)
            .collect();

        // Clamped zero ends; Fritsch–Butland weighted harmonic mean inside.
        let mut ds = vec![0.0; n];
        for i in 1..n - 1 {
            let s0 = secant[i - 1];
            let s1 = secant[i];
            if s0 == 0.0 || s1 == 0.0 || s0.signum() != s1.signum() {
                ds[i] = 0.0;
            } else {
                let w0 = 2.0 * h[i] + h[i - 1];
                let w1 = h[i] + 2.0 * h[i - 1];
                ds[i] = (w0 + w1) / (w0 / s0 + w1 / s1);
            }
        }

        Ok(Self { xs, ys, ds })
    }

    /// Interval index holding `x` (clamped to the end intervals outside).
    fn interval(&self, x: f64) -> usize {
        let n = self.xs.len();
        if x <= self.xs[0] {
            return 0;
        }
        if x >= self.xs[n - 1] {
            return n - 2;
        }
        let upper = self.xs.partition_point(|knot| *knot <= x);
        upper.saturating_sub(1).min(n - 2)
    }

    /// Evaluate the interpolant value, clamping `x` to the knot range.
    pub fn value(&self, x: f64) -> f64 {
        let x = x.clamp(self.xs[0], self.xs[self.xs.len() - 1]);
        let i = self.interval(x);
        let h = self.xs[i + 1] - self.xs[i];
        let t = (x - self.xs[i]) / h;
        let t2 = t * t;
        let t3 = t2 * t;
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;
        self.ys[i] * h00 + h * self.ds[i] * h10 + self.ys[i + 1] * h01 + h * self.ds[i + 1] * h11
    }

    /// Evaluate the first derivative, clamping `x` to the knot range.
    pub fn derivative(&self, x: f64) -> f64 {
        let x = x.clamp(self.xs[0], self.xs[self.xs.len() - 1]);
        let i = self.interval(x);
        let h = self.xs[i + 1] - self.xs[i];
        let t = (x - self.xs[i]) / h;
        let t2 = t * t;
        let dh00 = 6.0 * t2 - 6.0 * t;
        let dh10 = 3.0 * t2 - 4.0 * t + 1.0;
        let dh01 = -6.0 * t2 + 6.0 * t;
        let dh11 = 3.0 * t2 - 2.0 * t;
        (self.ys[i] * dh00
            + h * self.ds[i] * dh10
            + self.ys[i + 1] * dh01
            + h * self.ds[i + 1] * dh11)
            / h
    }

    /// Evaluate the second derivative, clamping `x` to the knot range.
    ///
    /// Piecewise linear with jumps at interior knots (the interpolant is C¹);
    /// still the exact `r''(x)` the curvature budget integrates.
    pub fn second_derivative(&self, x: f64) -> f64 {
        let x = x.clamp(self.xs[0], self.xs[self.xs.len() - 1]);
        let i = self.interval(x);
        let h = self.xs[i + 1] - self.xs[i];
        let t = (x - self.xs[i]) / h;
        let d2h00 = 12.0 * t - 6.0;
        let d2h10 = 6.0 * t - 4.0;
        let d2h01 = -12.0 * t + 6.0;
        let d2h11 = 6.0 * t - 2.0;
        (self.ys[i] * d2h00
            + h * self.ds[i] * d2h10
            + self.ys[i + 1] * d2h01
            + h * self.ds[i + 1] * d2h11)
            / (h * h)
    }
}

/// The fitted interpolant backing a [`SplinePortProfile`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
enum ProfileInterpolant {
    Clamped(ClampedCubicSpline),
    ShapePreserving(PchipSpline),
}

impl ProfileInterpolant {
    fn value(&self, x: f64) -> f64 {
        match self {
            Self::Clamped(s) => s.value(x),
            Self::ShapePreserving(s) => s.value(x),
        }
    }

    fn derivative(&self, x: f64) -> f64 {
        match self {
            Self::Clamped(s) => s.derivative(x),
            Self::ShapePreserving(s) => s.derivative(x),
        }
    }

    fn second_derivative(&self, x: f64) -> f64 {
        match self {
            Self::Clamped(s) => s.second_derivative(x),
            Self::ShapePreserving(s) => s.second_derivative(x),
        }
    }
}

/// Axisymmetric port meridian profile `r(x)` on `[0, L]`.
///
/// Control-point axial positions are fixed (uniform or Chebyshev); only radii
/// vary, in normalised variables `u_i = (R0 - r_i)/(R0 - r_min) in [0, 1]`
/// with `u = 0` the straight bore. The inlet, the second knot (straight
/// start), and the outlet are pinned to `R0`, leaving `N - 3` free radii at
/// knots `2..N-1`.
///
/// Every constructor runs the full dense-grid validation (finite, ordered
/// knots, bounds, `r'(0) = 0`, `r <=` effective radius cap, slope limit,
/// curvature budget) and rejects the candidate with
/// [`PortFlowError::InvalidConfig`] before any solve can consume it. The
/// effective cap is `R0` for legacy profiles and the flare cap for profiles
/// built by [`SplinePortProfile::from_normalized_flared`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplinePortProfile {
    n_control_points: usize,
    length_m: f64,
    bore_radius_m: f64,
    throat_radius_m: f64,
    /// Flare cap above `R0` in m (`None` keeps the legacy `r <= R0` cap).
    #[serde(default)]
    flare_max_radius_m: Option<f64>,
    spacing: KnotSpacing,
    limits: ShapeLimits,
    kind: ProfileSplineKind,
    knots: Vec<f64>,
    radii: Vec<f64>,
    interpolant: ProfileInterpolant,
}

impl SplinePortProfile {
    /// Build fixed axial knots for `n` control points over `[0, length_m]`.
    fn build_knots(n: usize, length_m: f64, spacing: KnotSpacing) -> Vec<f64> {
        (0..n)
            .map(|i| match spacing {
                KnotSpacing::Uniform => length_m * i as f64 / (n - 1) as f64,
                KnotSpacing::Chebyshev => {
                    0.5 * length_m * (1.0 - (PI * i as f64 / (n - 1) as f64).cos())
                }
            })
            .collect()
    }

    /// Shared constructor: validate scalars, fit the interpolant, audit.
    fn from_radii_with(
        radii: Vec<f64>,
        length_m: f64,
        bore_radius_m: f64,
        throat_radius_m: f64,
        flare_max_radius_m: Option<f64>,
        options: ProfileOptions,
    ) -> Result<Self, PortFlowError> {
        let n = radii.len();
        if n < 3 {
            return Err(PortFlowError::InvalidConfig(format!(
                "spline port profile needs at least 3 control points, got {n}"
            )));
        }
        for (name, value) in [
            ("length_m", length_m),
            ("bore_radius_m", bore_radius_m),
            ("throat_radius_m", throat_radius_m),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(PortFlowError::InvalidConfig(format!(
                    "spline port profile {name} must be finite and positive, got {value}"
                )));
            }
        }
        if throat_radius_m > bore_radius_m {
            return Err(PortFlowError::InvalidConfig(format!(
                "spline port profile throat radius {throat_radius_m} exceeds bore radius {bore_radius_m}"
            )));
        }
        if let Some(flare_max) = flare_max_radius_m
            && (!flare_max.is_finite() || flare_max <= bore_radius_m)
        {
            return Err(PortFlowError::InvalidConfig(format!(
                "spline port profile flare cap {flare_max} must be finite and above bore radius {bore_radius_m}"
            )));
        }
        let max_radius_m = flare_max_radius_m.unwrap_or(bore_radius_m);
        if !options.limits.slope_max.is_finite() || options.limits.slope_max < 0.0 {
            return Err(PortFlowError::InvalidConfig(format!(
                "spline port profile slope_max must be finite and non-negative, got {}",
                options.limits.slope_max
            )));
        }
        if !options.limits.curvature_budget_m3.is_finite()
            || options.limits.curvature_budget_m3 < 0.0
        {
            return Err(PortFlowError::InvalidConfig(format!(
                "spline port profile curvature_budget_m3 must be finite and non-negative, got {}",
                options.limits.curvature_budget_m3
            )));
        }
        for (index, radius) in radii.iter().enumerate() {
            if !radius.is_finite() {
                return Err(PortFlowError::InvalidConfig(format!(
                    "spline port profile radius {index} must be finite"
                )));
            }
            if *radius < throat_radius_m || *radius > max_radius_m {
                return Err(PortFlowError::InvalidConfig(format!(
                    "spline port profile radius {index} = {radius} outside [r_min, r_max] = [{throat_radius_m}, {max_radius_m}]"
                )));
            }
        }

        let knots = Self::build_knots(n, length_m, options.spacing);
        let pts: Vec<[f64; 2]> = knots
            .iter()
            .zip(radii.iter())
            .map(|(x, r)| [*x, *r])
            .collect();
        // Both interpolants clamp the ends to zero slope: straight start
        // r'(0) = 0 and, by symmetry of the inlet/outlet requirement, a
        // straight exit r'(L) = 0.
        let interpolant = match options.kind {
            ProfileSplineKind::ShapePreserving => {
                ProfileInterpolant::ShapePreserving(PchipSpline::try_from_control_points(&pts)?)
            }
            ProfileSplineKind::ClampedCubic => ProfileInterpolant::Clamped(
                ClampedCubicSpline::try_from_control_points(&pts, 0.0, 0.0)?,
            ),
        };
        let profile = Self {
            n_control_points: n,
            length_m,
            bore_radius_m,
            throat_radius_m,
            flare_max_radius_m,
            spacing: options.spacing,
            limits: options.limits,
            kind: options.kind,
            knots,
            radii,
            interpolant,
        };
        profile.validate()?;
        Ok(profile)
    }

    /// Build a profile from normalised free radii `u in [0, 1]^(N-3)`.
    ///
    /// `u[i]` sets knot `i + 2` via `r = R0 - u·(R0 - r_min)`; knots 0, 1,
    /// and `N - 1` are pinned to `R0`. For `N = 3` there are no free radii
    /// and `u` must be empty (the straight tube).
    pub fn from_normalized(
        n_control_points: usize,
        length_m: f64,
        bore_radius_m: f64,
        throat_radius_m: f64,
        u: &[f64],
        options: ProfileOptions,
    ) -> Result<Self, PortFlowError> {
        if n_control_points < 3 {
            return Err(PortFlowError::InvalidConfig(format!(
                "spline port profile needs at least 3 control points, got {n_control_points}"
            )));
        }
        if u.len() != n_control_points - 3 {
            return Err(PortFlowError::InvalidConfig(format!(
                "spline port profile with N = {n_control_points} needs {} free radii, got {}",
                n_control_points - 3,
                u.len()
            )));
        }
        for (index, value) in u.iter().enumerate() {
            if !value.is_finite() || *value < 0.0 || *value > 1.0 {
                return Err(PortFlowError::InvalidConfig(format!(
                    "spline port profile u[{index}] = {value} outside [0, 1]"
                )));
            }
        }
        let span = bore_radius_m - throat_radius_m;
        let mut radii = vec![bore_radius_m; n_control_points];
        for (i, value) in u.iter().enumerate() {
            radii[i + 2] = bore_radius_m - value * span;
        }
        Self::from_radii_with(
            radii,
            length_m,
            bore_radius_m,
            throat_radius_m,
            None,
            options,
        )
    }

    /// Build a flared profile from signed normalised radii `u in [u_min, 1]^N`.
    ///
    /// Every knot is a free variable: `u[i]` sets knot `i` via the same
    /// linear map `r = R0 - u·(R0 - r_min)` as [`Self::from_normalized`], so
    /// `u = 0` is the straight bore, positive `u` pinches toward `r_min`, and
    /// negative `u` widens toward `flare_max_radius_m`. The lower bound is
    /// `u_min = (R0 - flare_max)/(R0 - r_min)` (negative, since the cap must
    /// exceed `R0`). Both mouths can therefore flare, unlike the legacy
    /// mapping which pins them to `R0`.
    ///
    /// # Errors
    ///
    /// Returns [`PortFlowError::InvalidConfig`] when `N < 3`, `u` does not
    /// hold one entry per knot, any `u` is non-finite or outside
    /// `[u_min, 1]`, the flare cap is not finite and above `R0`, or the
    /// resulting profile violates the shape limits.
    pub fn from_normalized_flared(
        n_control_points: usize,
        length_m: f64,
        bore_radius_m: f64,
        throat_radius_m: f64,
        flare_max_radius_m: f64,
        u: &[f64],
        options: ProfileOptions,
    ) -> Result<Self, PortFlowError> {
        if n_control_points < 3 {
            return Err(PortFlowError::InvalidConfig(format!(
                "spline port profile needs at least 3 control points, got {n_control_points}"
            )));
        }
        if u.len() != n_control_points {
            return Err(PortFlowError::InvalidConfig(format!(
                "flared spline port profile with N = {n_control_points} needs {n_control_points} free radii, got {}",
                u.len()
            )));
        }
        if !flare_max_radius_m.is_finite() || flare_max_radius_m <= bore_radius_m {
            return Err(PortFlowError::InvalidConfig(format!(
                "spline port profile flare cap {flare_max_radius_m} must be finite and above bore radius {bore_radius_m}"
            )));
        }
        let span = bore_radius_m - throat_radius_m;
        if !span.is_finite() || span <= 0.0 {
            return Err(PortFlowError::InvalidConfig(format!(
                "flared spline port profile needs throat radius {throat_radius_m} below bore radius {bore_radius_m}"
            )));
        }
        let u_min = (bore_radius_m - flare_max_radius_m) / span;
        for (index, value) in u.iter().enumerate() {
            if !value.is_finite() || *value < u_min || *value > 1.0 {
                return Err(PortFlowError::InvalidConfig(format!(
                    "spline port profile u[{index}] = {value} outside [{u_min}, 1]"
                )));
            }
        }
        let radii: Vec<f64> = u.iter().map(|value| bore_radius_m - value * span).collect();
        Self::from_radii_with(
            radii,
            length_m,
            bore_radius_m,
            throat_radius_m,
            Some(flare_max_radius_m),
            options,
        )
    }

    /// Build a profile from absolute knot radii (all pinned knots included).
    ///
    /// Every entry must lie in `[r_min, R0]`; use [`Self::from_normalized`]
    /// when the optimiser owns the free variables. The inlet/outlet pinning
    /// is the caller's responsibility here so validation meshes can probe
    /// arbitrary (legal) control polygons.
    pub fn try_from_radii(
        radii: &[f64],
        length_m: f64,
        bore_radius_m: f64,
        throat_radius_m: f64,
        options: ProfileOptions,
    ) -> Result<Self, PortFlowError> {
        Self::from_radii_with(
            radii.to_vec(),
            length_m,
            bore_radius_m,
            throat_radius_m,
            None,
            options,
        )
    }

    /// The straight reference tube (`u = 0` everywhere).
    pub fn straight(
        n_control_points: usize,
        length_m: f64,
        bore_radius_m: f64,
        throat_radius_m: f64,
        options: ProfileOptions,
    ) -> Result<Self, PortFlowError> {
        Self::from_normalized(
            n_control_points,
            length_m,
            bore_radius_m,
            throat_radius_m,
            &vec![0.0; n_control_points.saturating_sub(3)],
            options,
        )
    }

    /// Re-run the full dense-grid validation on this profile.
    ///
    /// Constructors call this; expose it so later steps can re-audit a
    /// deserialized profile before an expensive solve.
    pub fn validate(&self) -> Result<(), PortFlowError> {
        if self.knots.len() != self.n_control_points || self.radii.len() != self.n_control_points {
            return Err(PortFlowError::InvalidConfig(
                "spline port profile knot/radii length mismatch".to_string(),
            ));
        }
        if self.knots.windows(2).any(|pair| pair[1] <= pair[0]) {
            return Err(PortFlowError::InvalidConfig(
                "spline port profile knots must be strictly increasing".to_string(),
            ));
        }
        if (self.knots[0] - 0.0).abs() > 1.0e-12 * self.length_m.max(1.0e-12)
            || (self.knots[self.n_control_points - 1] - self.length_m).abs()
                > 1.0e-12 * self.length_m.max(1.0e-12)
        {
            return Err(PortFlowError::InvalidConfig(
                "spline port profile knots must span [0, L]".to_string(),
            ));
        }

        let tol = BOUND_TOL_REL * self.bore_radius_m;
        let grid = PROFILE_VALIDATION_GRID_POINTS;
        let mut max_radius = f64::NEG_INFINITY;
        let mut min_radius = f64::INFINITY;
        let mut max_slope = 0.0f64;
        for i in 0..grid {
            let x = self.length_m * i as f64 / (grid - 1) as f64;
            let r = self.interpolant.value(x);
            let slope = self.interpolant.derivative(x).abs();
            max_radius = max_radius.max(r);
            min_radius = min_radius.min(r);
            max_slope = max_slope.max(slope);
        }
        if max_radius > self.max_radius_m() + tol {
            return Err(PortFlowError::InvalidConfig(format!(
                "spline port profile overshoots the bore: max r = {max_radius} > r_max = {}",
                self.max_radius_m()
            )));
        }
        if min_radius < self.throat_radius_m - tol {
            return Err(PortFlowError::InvalidConfig(format!(
                "spline port profile undershoots the throat: min r = {min_radius} < r_min = {}",
                self.throat_radius_m
            )));
        }
        if max_slope > self.limits.slope_max + SLOPE_TOL_ABS {
            return Err(PortFlowError::InvalidConfig(format!(
                "spline port profile slope {max_slope} exceeds limit {}",
                self.limits.slope_max
            )));
        }
        let start_slope = self.interpolant.derivative(0.0).abs();
        if start_slope > SLOPE_TOL_ABS {
            return Err(PortFlowError::InvalidConfig(format!(
                "spline port profile start slope {start_slope} violates the straight-start r'(0) = 0"
            )));
        }
        let curvature = self.curvature_integral();
        if !curvature.is_finite() || curvature > self.limits.curvature_budget_m3 {
            return Err(PortFlowError::InvalidConfig(format!(
                "spline port profile curvature budget exceeded: {curvature} > {}",
                self.limits.curvature_budget_m3
            )));
        }
        Ok(())
    }

    /// Number of control points `N`.
    pub fn n_control_points(&self) -> usize {
        self.n_control_points
    }

    /// Port length `L` in m.
    pub fn length_m(&self) -> f64 {
        self.length_m
    }

    /// Straight-bore radius `R0` in m.
    pub fn bore_radius_m(&self) -> f64 {
        self.bore_radius_m
    }

    /// Minimum admissible radius `r_min` in m.
    pub fn throat_radius_m(&self) -> f64 {
        self.throat_radius_m
    }

    /// Flare cap above `R0` in m (`None` for a legacy inward-only profile).
    pub fn flare_max_radius_m(&self) -> Option<f64> {
        self.flare_max_radius_m
    }

    /// Effective maximum radius in m: the flare cap, or `R0` when unflared.
    pub fn max_radius_m(&self) -> f64 {
        self.flare_max_radius_m.unwrap_or(self.bore_radius_m)
    }

    /// Knot placement used by this profile.
    pub fn spacing(&self) -> KnotSpacing {
        self.spacing
    }

    /// Manufacturability limits enforced at construction.
    pub fn limits(&self) -> ShapeLimits {
        self.limits
    }

    /// Interpolant backing this profile.
    pub fn kind(&self) -> ProfileSplineKind {
        self.kind
    }

    /// Fixed axial knot positions in m.
    pub fn knots(&self) -> &[f64] {
        &self.knots
    }

    /// Absolute radii at the knots in m.
    pub fn radii(&self) -> &[f64] {
        &self.radii
    }

    /// Free radii in normalised variables.
    ///
    /// Legacy profiles return `u in [0, 1]` for knots `2..N-1`. Flared
    /// profiles return one signed entry per knot, with negative values
    /// above `R0` toward the flare cap (see
    /// [`Self::from_normalized_flared`]).
    pub fn normalized(&self) -> Vec<f64> {
        let span = self.bore_radius_m - self.throat_radius_m;
        if self.flare_max_radius_m.is_some() {
            return self
                .radii
                .iter()
                .map(|r| (self.bore_radius_m - r) / span)
                .collect();
        }
        self.radii
            .iter()
            .skip(2)
            .take(self.n_control_points.saturating_sub(3))
            .map(|r| (self.bore_radius_m - r) / span)
            .collect()
    }

    /// Meridian radius `r(x)` in m (`x` clamped to `[0, L]`).
    pub fn local_radius_m(&self, x: f64) -> f64 {
        self.interpolant.value(x)
    }

    /// Cross-sectional area `π·r(x)²` in m² (`x` clamped to `[0, L]`).
    pub fn area_m2(&self, x: f64) -> f64 {
        let r = self.local_radius_m(x);
        PI * r * r
    }

    /// Wall slope `r'(x)` (dimensionless, `x` clamped to `[0, L]`).
    pub fn slope(&self, x: f64) -> f64 {
        self.interpolant.derivative(x)
    }

    /// Wall curvature `r''(x)` in m⁻¹ (`x` clamped to `[0, L]`).
    ///
    /// Continuous for [`ProfileSplineKind::ClampedCubic`], piecewise linear
    /// with knot jumps for [`ProfileSplineKind::ShapePreserving`] (C¹).
    pub fn curvature(&self, x: f64) -> f64 {
        self.interpolant.second_derivative(x)
    }

    /// Curvature budget `∫₀ᴸ(r'')² dx` in m⁻³ (trapezoid on the dense grid).
    pub fn curvature_integral(&self) -> f64 {
        let grid = PROFILE_VALIDATION_GRID_POINTS;
        let dx = self.length_m / (grid - 1) as f64;
        let mut sum = 0.0;
        for i in 0..grid {
            let x = self.length_m * i as f64 / (grid - 1) as f64;
            let k = self.interpolant.second_derivative(x);
            let weight = if i == 0 || i == grid - 1 { 0.5 } else { 1.0 };
            sum += weight * k * k;
        }
        sum * dx
    }

    /// Sample `(x, r)` pairs on `n` uniform stations over `[0, L]`.
    ///
    /// # Panics
    ///
    /// Panics when `n < 2` (programming contract, not profile data).
    pub fn sampled_profile(&self, n: usize) -> Vec<[f64; 2]> {
        assert!(n >= 2, "sampled_profile needs at least 2 stations");
        (0..n)
            .map(|i| {
                let x = self.length_m * i as f64 / (n - 1) as f64;
                [x, self.local_radius_m(x)]
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const L: f64 = 0.1;
    const R0: f64 = 0.025;
    const RMIN: f64 = 0.015;

    fn opts(kind: ProfileSplineKind) -> ProfileOptions {
        ProfileOptions {
            kind,
            ..ProfileOptions::default()
        }
    }

    fn profile(n: usize, u: &[f64]) -> SplinePortProfile {
        SplinePortProfile::from_normalized(n, L, R0, RMIN, u, ProfileOptions::default())
            .expect("test profile must construct")
    }

    #[test]
    fn spline_passes_through_control_points() {
        let p = profile(5, &[0.25, 0.75]);
        for (x, r) in p.knots().iter().zip(p.radii().iter()) {
            assert!(
                (p.local_radius_m(*x) - r).abs() < 1.0e-12,
                "profile misses control point at x = {x}"
            );
        }
    }

    #[test]
    fn clamped_start_and_end_slopes_are_zero() {
        let p = profile(5, &[0.25, 0.75]);
        assert!(
            p.slope(0.0).abs() < 1.0e-12,
            "straight start violated: r'(0) = {}",
            p.slope(0.0)
        );
        assert!(
            p.slope(L).abs() < 1.0e-12,
            "straight exit violated: r'(L) = {}",
            p.slope(L)
        );
        assert!(
            (p.local_radius_m(0.0) - R0).abs() < 1.0e-12,
            "r(0) must equal R0"
        );
        assert!(
            (p.local_radius_m(L) - R0).abs() < 1.0e-12,
            "r(L) must equal R0"
        );
    }

    #[test]
    fn straight_profile_reproduces_bore_everywhere() {
        for kind in [
            ProfileSplineKind::ShapePreserving,
            ProfileSplineKind::ClampedCubic,
        ] {
            let p = SplinePortProfile::straight(5, L, R0, RMIN, opts(kind))
                .expect("straight profile must construct");
            for [x, r] in p.sampled_profile(PROFILE_VALIDATION_GRID_POINTS) {
                assert!(
                    (r - R0).abs() < 1.0e-12,
                    "{kind:?}: straight profile deviates at x = {x}: r = {r}"
                );
                assert!(
                    p.slope(x).abs() < 1.0e-12,
                    "{kind:?}: straight profile slope nonzero at x = {x}"
                );
                assert!(
                    (p.area_m2(x) - PI * R0 * R0).abs() < 1.0e-15,
                    "{kind:?}: straight profile area wrong at x = {x}"
                );
            }
            assert!(
                p.curvature_integral() < 1.0e-20,
                "{kind:?}: straight profile curvature must vanish"
            );
            assert_eq!(p.normalized(), vec![0.0; 2]);
        }
    }

    #[test]
    fn normalized_round_trip_recovers_free_radii() {
        let u = vec![0.2, 0.6];
        let p = profile(5, &u);
        let back = p.normalized();
        assert_eq!(back.len(), u.len());
        for (got, want) in back.iter().zip(u.iter()) {
            assert!((got - want).abs() < 1.0e-12, "u round trip failed");
        }
        // Absolute radii follow r = R0 - u·(R0 - r_min) at knots 2..N-1.
        for (i, want) in u.iter().enumerate() {
            let r = p.radii()[i + 2];
            let expected = R0 - want * (R0 - RMIN);
            assert!((r - expected).abs() < 1.0e-15, "radius mapping wrong");
        }
    }

    #[test]
    fn clamped_cubic_overshoot_is_rejected() {
        // Even a gentle dip overshoots the bore between knots under C²
        // interpolation; validation must reject the candidate.
        for radii in [
            vec![R0, R0, R0 - 0.25 * (R0 - RMIN), R0 - 0.75 * (R0 - RMIN), R0],
            vec![R0, R0, RMIN, R0, R0],
        ] {
            let err = SplinePortProfile::try_from_radii(
                &radii,
                L,
                R0,
                RMIN,
                opts(ProfileSplineKind::ClampedCubic),
            )
            .expect_err("C² overshoot must be rejected");
            assert!(
                matches!(err, PortFlowError::InvalidConfig(_)),
                "wrong error kind: {err}"
            );
            assert!(
                err.to_string().contains("overshoot") || err.to_string().contains("undershoot"),
                "rejection must come from the dense-grid bound arm: {err}"
            );
        }
    }

    #[test]
    fn shape_preserving_accepts_dipping_polygons() {
        // Polygons the C² mode rejects for overshoot construct fine by
        // default, with the dense grid provably inside [r_min, R0]. The
        // full-depth V needs a relaxed slope limit (its wall slope is 0.6,
        // above the default 0.5); bounds and slope are independent arms.
        let mid = R0 - 0.5 * (R0 - RMIN);
        let relaxed = ProfileOptions {
            limits: ShapeLimits {
                slope_max: 1.0,
                curvature_budget_m3: DEFAULT_SHAPE_CURVATURE_BUDGET_M3,
            },
            ..ProfileOptions::default()
        };
        for (radii, options) in [
            (
                vec![R0, R0, R0 - 0.25 * (R0 - RMIN), R0 - 0.5 * (R0 - RMIN), R0],
                ProfileOptions::default(),
            ),
            (vec![R0, R0, mid, R0, R0], ProfileOptions::default()),
            (vec![R0, R0, mid, mid, R0, R0], ProfileOptions::default()),
            (vec![R0, R0, RMIN, R0, R0], relaxed),
        ] {
            let p = SplinePortProfile::try_from_radii(&radii, L, R0, RMIN, options)
                .unwrap_or_else(|err| panic!("shape-preserving mode must accept {radii:?}: {err}"));
            for [x, r] in p.sampled_profile(PROFILE_VALIDATION_GRID_POINTS) {
                assert!(
                    (RMIN - 1.0e-12..=R0 + 1.0e-12).contains(&r),
                    "shape-preserving bound violated at x = {x}: r = {r}"
                );
            }
        }
    }

    #[test]
    fn clamped_cubic_overshoot_magnitude_documents_d2_trigger() {
        // Direct evidence for plan decision D2: the C² clamped spline
        // through a gentle dip polygon exceeds R0 between knots.
        let u = [0.25, 0.75];
        let knots = SplinePortProfile::build_knots(5, L, KnotSpacing::Uniform);
        let span = R0 - RMIN;
        let pts: Vec<[f64; 2]> = [R0, R0, R0 - u[0] * span, R0 - u[1] * span, R0]
            .iter()
            .zip(knots.iter())
            .map(|(r, x)| [*x, *r])
            .collect();
        let spline = ClampedCubicSpline::try_from_control_points(&pts, 0.0, 0.0)
            .expect("C² fit must succeed");
        let mut max_r = f64::NEG_INFINITY;
        for i in 0..PROFILE_VALIDATION_GRID_POINTS {
            let x = L * i as f64 / (PROFILE_VALIDATION_GRID_POINTS - 1) as f64;
            max_r = max_r.max(spline.value(x));
        }
        assert!(
            max_r > R0 + 1.0e-9,
            "expected measurable C² overshoot above R0, got max r = {max_r}"
        );
    }

    #[test]
    fn out_of_range_normalized_radii_are_rejected() {
        for bad in [vec![-0.1, 0.5], vec![0.5, 1.2], vec![f64::NAN, 0.5]] {
            assert!(
                SplinePortProfile::from_normalized(5, L, R0, RMIN, &bad, ProfileOptions::default())
                    .is_err(),
                "u = {bad:?} must be rejected"
            );
        }
        // Wrong free-variable count for N = 5 (needs exactly N - 3 = 2).
        assert!(
            SplinePortProfile::from_normalized(5, L, R0, RMIN, &[0.5], ProfileOptions::default())
                .is_err(),
            "wrong u length must be rejected"
        );
    }

    #[test]
    fn invalid_geometry_scalars_are_rejected() {
        // Fewer than 3 control points.
        assert!(
            SplinePortProfile::from_normalized(2, L, R0, RMIN, &[], ProfileOptions::default())
                .is_err()
        );
        // Non-positive length, throat above bore.
        assert!(
            SplinePortProfile::from_normalized(
                5,
                0.0,
                R0,
                RMIN,
                &[0.1, 0.2],
                ProfileOptions::default()
            )
            .is_err()
        );
        assert!(
            SplinePortProfile::from_normalized(
                5,
                L,
                R0,
                2.0 * R0,
                &[0.1, 0.2],
                ProfileOptions::default()
            )
            .is_err()
        );
        // Tight slope limit rejects a genuine dip.
        let tight = ProfileOptions {
            limits: ShapeLimits {
                slope_max: 1.0e-6,
                curvature_budget_m3: DEFAULT_SHAPE_CURVATURE_BUDGET_M3,
            },
            ..ProfileOptions::default()
        };
        assert!(
            SplinePortProfile::from_normalized(5, L, R0, RMIN, &[0.5, 0.5], tight).is_err(),
            "tight slope limit must reject a dipping profile"
        );
        // Vanishing curvature budget rejects any non-straight profile but
        // keeps the straight tube.
        let no_curvature = ProfileOptions {
            limits: ShapeLimits {
                slope_max: DEFAULT_SHAPE_SLOPE_MAX,
                curvature_budget_m3: 0.0,
            },
            ..ProfileOptions::default()
        };
        assert!(
            SplinePortProfile::from_normalized(5, L, R0, RMIN, &[0.5, 0.5], no_curvature).is_err(),
            "zero curvature budget must reject a dipping profile"
        );
        assert!(
            SplinePortProfile::straight(5, L, R0, RMIN, no_curvature).is_ok(),
            "straight tube must pass a zero curvature budget"
        );
    }

    #[test]
    fn control_point_counts_3_5_8_construct_and_evaluate() {
        for n in [3, 5, 8] {
            let u = vec![0.4; n - 3];
            let p =
                SplinePortProfile::from_normalized(n, L, R0, RMIN, &u, ProfileOptions::default())
                    .unwrap_or_else(|err| panic!("N = {n} must construct: {err}"));
            assert_eq!(p.n_control_points(), n);
            assert_eq!(p.knots().len(), n);
            let samples = p.sampled_profile(64);
            for [x, r] in &samples {
                assert!(
                    (RMIN - 1.0e-12..=R0 + 1.0e-12).contains(r),
                    "N = {n}: r({x}) = {r} outside [r_min, R0]"
                );
                assert!(p.area_m2(*x) > 0.0, "N = {n}: non-positive area");
                assert!(p.slope(*x).is_finite(), "N = {n}: non-finite slope");
                assert!(p.curvature(*x).is_finite(), "N = {n}: non-finite curvature");
            }
            // Chebyshev spacing builds the same-shaped contract. End
            // clustering shortens the exit transition, so this leg dips
            // shallower to respect the same default slope limit.
            let cheb_u = vec![0.1; n - 3];
            let cheb_options = ProfileOptions {
                spacing: KnotSpacing::Chebyshev,
                ..ProfileOptions::default()
            };
            let cheb = SplinePortProfile::from_normalized(n, L, R0, RMIN, &cheb_u, cheb_options)
                .unwrap_or_else(|err| panic!("N = {n} Chebyshev must construct: {err}"));
            assert!(
                (cheb.local_radius_m(0.0) - R0).abs() < 1.0e-12
                    && (cheb.local_radius_m(L) - R0).abs() < 1.0e-12,
                "N = {n}: Chebyshev end values must equal R0"
            );
            // C² mode constructs the straight tube at every count.
            let cubic =
                SplinePortProfile::straight(n, L, R0, RMIN, opts(ProfileSplineKind::ClampedCubic))
                    .unwrap_or_else(|err| panic!("N = {n} C² straight must construct: {err}"));
            assert!(
                (cubic.local_radius_m(0.5 * L) - R0).abs() < 1.0e-12,
                "N = {n}: C² straight tube must reproduce R0"
            );
        }
    }

    #[test]
    fn n3_has_no_free_radii_and_is_straight() {
        let p = profile(3, &[]);
        assert_eq!(p.normalized(), Vec::<f64>::new());
        for [x, r] in p.sampled_profile(32) {
            assert!(
                (r - R0).abs() < 1.0e-12,
                "N = 3 pins every knot to R0, got r({x}) = {r}"
            );
        }
    }

    #[test]
    fn clamped_spline_rejects_bad_input() {
        // Too few points.
        assert!(ClampedCubicSpline::try_from_control_points(&[[0.0, 1.0]], 0.0, 0.0).is_err());
        // Non-finite coordinate.
        assert!(
            ClampedCubicSpline::try_from_control_points(&[[0.0, 1.0], [1.0, f64::NAN]], 0.0, 0.0)
                .is_err()
        );
        // Non-increasing knots.
        assert!(
            ClampedCubicSpline::try_from_control_points(&[[0.0, 1.0], [0.0, 2.0]], 0.0, 0.0)
                .is_err()
        );
        // Non-finite end slope.
        assert!(
            ClampedCubicSpline::try_from_control_points(
                &[[0.0, 1.0], [1.0, 2.0]],
                f64::INFINITY,
                0.0
            )
            .is_err()
        );
        // Shape-preserving twin guards the same contract.
        assert!(PchipSpline::try_from_control_points(&[[0.0, 1.0]]).is_err());
        assert!(PchipSpline::try_from_control_points(&[[0.0, 1.0], [0.0, 2.0]]).is_err());
    }

    #[test]
    fn clamped_spline_matches_end_slopes_and_values() {
        // Cosine on [0, π]: clamped end slopes 0 reproduce the shape to
        // interpolation accuracy with enough knots.
        let n = 9;
        let pts: Vec<[f64; 2]> = (0..n)
            .map(|i| {
                let x = PI * i as f64 / (n - 1) as f64;
                [x, x.cos()]
            })
            .collect();
        let spline = ClampedCubicSpline::try_from_control_points(&pts, 0.0, 0.0)
            .expect("cosine spline must fit");
        assert!(
            spline.derivative(0.0).abs() < 1.0e-12,
            "start slope must be 0"
        );
        assert!(spline.derivative(PI).abs() < 1.0e-12, "end slope must be 0");
        for [x, y] in &pts {
            assert!(
                (spline.value(*x) - y).abs() < 1.0e-12,
                "spline misses knot at x = {x}"
            );
        }
        let mid = PI / 2.0;
        assert!(
            (spline.value(mid) - mid.cos()).abs() < 5.0e-4,
            "midpoint interpolation inaccurate: {}",
            spline.value(mid)
        );
        // Second derivative continuity across an interior knot.
        let knot = pts[4][0];
        let left = spline.second_derivative(knot - 1.0e-9);
        let right = spline.second_derivative(knot + 1.0e-9);
        assert!(
            (left - right).abs() < 1.0e-6,
            "C² continuity broken at x = {knot}"
        );
    }

    #[test]
    fn pchip_is_shape_preserving_and_clamped() {
        // Non-monotone dip polygon: every interval stays monotone, so the
        // interpolant never leaves the control-polygon range.
        let pts: Vec<[f64; 2]> = [
            R0,
            R0,
            R0 - 0.5 * (R0 - RMIN),
            RMIN,
            R0 - 0.5 * (R0 - RMIN),
            R0,
            R0,
        ]
        .iter()
        .enumerate()
        .map(|(i, r)| [L * i as f64 / 6.0, *r])
        .collect();
        let spline = PchipSpline::try_from_control_points(&pts).expect("PCHIP fit must succeed");
        assert!(
            spline.derivative(0.0).abs() == 0.0,
            "PCHIP start slope must be exactly 0"
        );
        assert!(
            spline.derivative(L).abs() == 0.0,
            "PCHIP end slope must be exactly 0"
        );
        for [x, y] in &pts {
            assert!(
                (spline.value(*x) - y).abs() < 1.0e-12,
                "PCHIP misses knot at x = {x}"
            );
        }
        for i in 0..401 {
            let x = L * i as f64 / 400.0;
            let v = spline.value(x);
            // 1e-12 covers Hermite-basis evaluation roundoff (~5e-18 seen);
            // genuine overshoot is orders of magnitude larger (cf. the C²
            // test above, which measures ~4e-5 on a gentler polygon).
            assert!(
                (RMIN - 1.0e-12..=R0 + 1.0e-12).contains(&v),
                "PCHIP left the control range at x = {x}: {v}"
            );
        }
        // C¹ continuity: one-sided derivatives agree at an interior knot.
        let knot = pts[3][0];
        let left = spline.derivative(knot - 1.0e-9);
        let right = spline.derivative(knot + 1.0e-9);
        assert!(
            (left - right).abs() < 1.0e-6,
            "C¹ continuity broken at x = {knot}"
        );
        // Monotone data stays monotone on the dense grid.
        let ramp: Vec<[f64; 2]> = (0..5).map(|i| [i as f64, (i * i) as f64]).collect();
        let ramp = PchipSpline::try_from_control_points(&ramp).expect("ramp must fit");
        let mut prev = f64::NEG_INFINITY;
        for i in 0..201 {
            let v = ramp.value(4.0 * i as f64 / 200.0);
            assert!(v >= prev - 1.0e-12, "monotone data must stay monotone");
            prev = v;
        }
    }

    #[test]
    fn flared_profile_flares_both_mouths_and_round_trips() {
        // Flare cap 1.4·R0: u = -1 maps a knot to the cap, u = 0 to R0.
        let flare_max = 1.4 * R0;
        let u = vec![-0.5, -0.25, 0.0, -0.25, -0.5];
        let p = SplinePortProfile::from_normalized_flared(
            5,
            L,
            R0,
            RMIN,
            flare_max,
            &u,
            ProfileOptions::default(),
        )
        .expect("gentle flare must construct");
        assert_eq!(p.flare_max_radius_m(), Some(flare_max));
        assert!(
            (p.local_radius_m(0.0) - 0.030).abs() < 1.0e-12,
            "interior mouth must flare above R0, got {}",
            p.local_radius_m(0.0)
        );
        assert!(
            (p.local_radius_m(L) - 0.030).abs() < 1.0e-12,
            "exterior mouth must flare above R0, got {}",
            p.local_radius_m(L)
        );
        assert!(
            (p.local_radius_m(0.5 * L) - R0).abs() < 1.0e-12,
            "throat knot must stay at R0 for u = 0"
        );
        let back = p.normalized();
        assert_eq!(back.len(), u.len(), "flare mode packs every knot");
        for (got, want) in back.iter().zip(u.iter()) {
            assert!((got - want).abs() < 1.0e-12, "flare u round trip failed");
        }
        p.validate().expect("flared profile must re-validate");
    }

    #[test]
    fn flared_profile_rejects_beyond_cap_and_bad_cap() {
        let flare_max = 1.4 * R0;
        assert!(
            SplinePortProfile::from_normalized_flared(
                5,
                L,
                R0,
                RMIN,
                flare_max,
                &[-1.1, 0.0, 0.0, 0.0, 0.0],
                ProfileOptions::default(),
            )
            .is_err(),
            "radius beyond the flare cap must be rejected"
        );
        assert!(
            SplinePortProfile::from_normalized_flared(
                5,
                L,
                R0,
                RMIN,
                R0,
                &[0.0; 5],
                ProfileOptions::default(),
            )
            .is_err(),
            "flare cap at or below R0 is not a flare and must be rejected"
        );
        assert!(
            SplinePortProfile::from_normalized_flared(
                5,
                L,
                R0,
                RMIN,
                flare_max,
                &[0.0; 2],
                ProfileOptions::default(),
            )
            .is_err(),
            "flare mode needs one variable per knot"
        );
        // The legacy inward-only constructor still rejects negative u.
        assert!(
            SplinePortProfile::from_normalized(
                5,
                L,
                R0,
                RMIN,
                &[-0.1, 0.0],
                ProfileOptions::default()
            )
            .is_err(),
            "legacy mapping must stay inward-only"
        );
    }

    #[test]
    fn profile_serializes_for_campaign_history() {
        let p = profile(5, &[0.3, 0.6]);
        let json = serde_json::to_string(&p).expect("profile must serialize");
        let back: SplinePortProfile =
            serde_json::from_str(&json).expect("profile must deserialize");
        assert_eq!(back, p);
        assert_eq!(back.kind(), ProfileSplineKind::ShapePreserving);
        back.validate().expect("deserialized profile must validate");
    }
}
