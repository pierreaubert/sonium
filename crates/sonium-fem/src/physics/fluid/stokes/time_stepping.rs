//! Time integration for unsteady Navier-Stokes
//!
//! Provides BDF (Backward Differentiation Formula) time steppers for
//! the transient incompressible Navier-Stokes equations:
//!
//! ```text
//! ρ ∂u/∂t + ρ(u·∇)u - μΔu + ∇p = f
//!                          div(u) = 0
//! ```
//!
//! # BDF1 (Implicit Euler)
//!
//! ```text
//! ρ(u^{n+1} - u^n)/Δt + ρ(u^{n+1}·∇)u^{n+1} - μΔu^{n+1} + ∇p^{n+1} = f^{n+1}
//! ```
//!
//! # BDF2 (Second-order backward differentiation)
//!
//! ```text
//! ρ(3u^{n+1} - 4u^n + u^{n-1})/(2Δt) + ρ(u*·∇)u^{n+1} - μΔu^{n+1} + ∇p^{n+1} = f^{n+1}
//! ```
//!
//! where `u* = 2u^n - u^{n-1}` is the extrapolated advection velocity.
//!
//! At each time step, a linearised (Oseen) system is solved using the
//! existing Picard machinery.

/// BDF time-stepping configuration.
#[derive(Debug, Clone)]
pub struct BdfConfig {
    /// Time step Δt \[s\].
    pub dt: f64,
    /// BDF order (1 or 2).
    pub order: usize,
    /// Number of time steps.
    pub num_steps: usize,
    /// Output interval (write VTK every N steps, 0 = never).
    pub output_interval: usize,
    /// Duration of the previously accepted step \[s\], when varying `dt`.
    ///
    /// `None` (the default) selects the constant-step coefficients.
    /// Callers that change `dt` between steps set this to the step size
    /// that produced the current state; the BDF2 coefficients and the
    /// advection extrapolation then use the true step ratio instead of
    /// silently assuming equal steps. Ignored for BDF1 and before the
    /// first completed step (no previous time level exists yet).
    pub previous_dt: Option<f64>,
}

impl BdfConfig {
    /// BDF1 (implicit Euler).
    pub fn bdf1(dt: f64, num_steps: usize) -> Self {
        Self {
            dt,
            order: 1,
            num_steps,
            output_interval: 0,
            previous_dt: None,
        }
    }

    /// BDF2 (second-order, recommended for accuracy).
    pub fn bdf2(dt: f64, num_steps: usize) -> Self {
        Self {
            dt,
            order: 2,
            num_steps,
            output_interval: 0,
            previous_dt: None,
        }
    }

    /// Set VTK output interval.
    pub fn with_output(mut self, interval: usize) -> Self {
        self.output_interval = interval;
        self
    }

    /// Step ratio `dt / previous_dt` for the step about to be taken.
    ///
    /// Returns 1.0 (the constant-step assumption) unless a valid
    /// previous step size is recorded and at least one step has been
    /// completed. Variable-step BDF2 stays zero-stable for growth
    /// ratios up to about 2.4; callers should ramp `dt` well below
    /// that (see [`AdaptiveStepController`]).
    pub fn step_ratio(&self, completed_steps: usize) -> f64 {
        if completed_steps == 0 {
            return 1.0;
        }
        match self.previous_dt {
            Some(previous)
                if previous.is_finite()
                    && previous > 0.0
                    && self.dt.is_finite()
                    && self.dt > 0.0 =>
            {
                self.dt / previous
            }
            _ => 1.0,
        }
    }
}

/// State for the BDF time stepper.
#[derive(Debug, Clone)]
pub struct BdfState {
    /// Current velocity DOFs (time level n).
    pub velocity: Vec<f64>,
    /// Previous velocity DOFs (time level n-1, for BDF2).
    pub velocity_prev: Vec<f64>,
    /// Current pressure DOFs.
    pub pressure: Vec<f64>,
    /// Current simulation time.
    pub time: f64,
    /// Current step number.
    pub step: usize,
}

impl BdfState {
    /// Create initial state from rest (zero velocity, zero pressure).
    pub fn from_rest(n_velocity_dofs: usize, n_pressure_dofs: usize) -> Self {
        Self {
            velocity: vec![0.0; n_velocity_dofs],
            velocity_prev: vec![0.0; n_velocity_dofs],
            pressure: vec![0.0; n_pressure_dofs],
            time: 0.0,
            step: 0,
        }
    }

    /// Create initial state from a steady solution.
    pub fn from_steady(velocity: Vec<f64>, pressure: Vec<f64>) -> Self {
        let vel_prev = velocity.clone();
        Self {
            velocity,
            velocity_prev: vel_prev,
            pressure,
            time: 0.0,
            step: 0,
        }
    }
}

/// BDF time derivative coefficients.
///
/// The time derivative is approximated as:
/// `∂u/∂t ≈ (a0 u^{n+1} + a1 u^n + a2 u^{n-1}) / Δt`
///
/// This adds `ρ a0 / Δt × M` to the system matrix and
/// `ρ / Δt × M × (-a1 u^n - a2 u^{n-1})` to the RHS.
pub struct BdfCoefficients {
    /// Coefficient for u^{n+1} (appears in the system matrix).
    pub a0: f64,
    /// Coefficient for u^n (appears in the RHS).
    pub a1: f64,
    /// Coefficient for u^{n-1} (appears in the RHS, zero for BDF1).
    pub a2: f64,
}

impl BdfCoefficients {
    /// BDF1 coefficients: (u^{n+1} - u^n) / Δt
    pub fn bdf1() -> Self {
        Self {
            a0: 1.0,
            a1: -1.0,
            a2: 0.0,
        }
    }

    /// BDF2 coefficients: (3u^{n+1} - 4u^n + u^{n-1}) / (2Δt)
    pub fn bdf2() -> Self {
        Self {
            a0: 1.5,
            a1: -2.0,
            a2: 0.5,
        }
    }

    /// Get coefficients for the given BDF order.
    pub fn for_order(order: usize) -> Self {
        match order {
            2 => Self::bdf2(),
            _ => Self::bdf1(),
        }
    }

    /// Variable-step BDF2 coefficients for step ratio `r = dt_new / dt_old`.
    ///
    /// Differentiating the quadratic through `u^{n-1}, u^n, u^{n+1}` at
    /// `t^{n+1}` gives `(a0 u^{n+1} + a1 u^n + a2 u^{n-1}) / dt_new` with
    /// `a0 = (1 + 2r) / (1 + r)`, `a1 = -(1 + r)`, `a2 = r^2 / (1 + r)`.
    /// Reduces to [`Self::bdf2`] at `r = 1`; a non-positive or non-finite
    /// ratio also falls back to the equal-step coefficients.
    pub fn bdf2_variable(step_ratio: f64) -> Self {
        if !step_ratio.is_finite() || step_ratio <= 0.0 {
            return Self::bdf2();
        }
        let r = step_ratio;
        Self {
            a0: (1.0 + 2.0 * r) / (1.0 + r),
            a1: -(1.0 + r),
            a2: r * r / (1.0 + r),
        }
    }

    /// Get coefficients for the given BDF order and step ratio.
    pub fn for_order_variable(order: usize, step_ratio: f64) -> Self {
        match order {
            2 => Self::bdf2_variable(step_ratio),
            _ => Self::bdf1(),
        }
    }
}

/// Extrapolate the advection velocity for the BDF scheme.
///
/// - BDF1: `u* = u^n` (first-order extrapolation)
/// - BDF2: `u* = 2u^n - u^{n-1}` (second-order extrapolation)
pub fn extrapolate_velocity(state: &BdfState, order: usize) -> Vec<f64> {
    match order {
        2 if state.step >= 1 => {
            // u* = 2 u^n - u^{n-1}
            state
                .velocity
                .iter()
                .zip(&state.velocity_prev)
                .map(|(&un, &unm1)| 2.0 * un - unm1)
                .collect()
        }
        _ => {
            // u* = u^n
            state.velocity.clone()
        }
    }
}

/// Extrapolate the advection velocity for a variable-step BDF scheme.
///
/// - BDF1: `u* = u^n`
/// - BDF2: linear extrapolation over the true step sizes,
///   `u* = (1 + r) u^n - r u^{n-1}` with `r = dt_new / dt_old`
///   (reduces to `2u^n - u^{n-1}` at `r = 1`).
pub fn extrapolate_velocity_variable(state: &BdfState, order: usize, step_ratio: f64) -> Vec<f64> {
    let r = if step_ratio.is_finite() && step_ratio > 0.0 {
        step_ratio
    } else {
        1.0
    };
    match order {
        2 if state.step >= 1 => state
            .velocity
            .iter()
            .zip(&state.velocity_prev)
            .map(|(&un, &unm1)| (1.0 + r) * un - r * unm1)
            .collect(),
        _ => state.velocity.clone(),
    }
}

/// CFL-based adaptive time-step controller.
///
/// The caller reports the peak speed and a characteristic length
/// (usually the smallest cell size) after each accepted step; the
/// controller returns the step size for the next step, clamped to
/// `[min_dt, max_dt]`. Growth per step is limited to `max_growth` so
/// the variable-step BDF2 ratio stays well inside its stability
/// range; shrinking is unlimited so a sudden jet is answered on the
/// very next step. A zero or non-finite reported speed relaxes the
/// step toward `max_dt` (still growth-limited).
#[derive(Debug, Clone)]
pub struct AdaptiveStepController {
    target_cfl: f64,
    min_dt: f64,
    max_dt: f64,
    max_growth: f64,
    current_dt: f64,
}

impl AdaptiveStepController {
    /// Create a controller starting from `initial_dt`.
    ///
    /// Non-finite or out-of-range inputs are clamped to sane values:
    /// `min_dt` floors at the smallest positive normal, `max_dt` is at
    /// least `min_dt`, `target_cfl` defaults to 0.5 if not positive,
    /// and `max_growth` is at least 1.0 (never grow).
    pub fn new(
        initial_dt: f64,
        target_cfl: f64,
        min_dt: f64,
        max_dt: f64,
        max_growth: f64,
    ) -> Self {
        let min_dt = if min_dt.is_finite() && min_dt > 0.0 {
            min_dt
        } else {
            f64::MIN_POSITIVE
        };
        let max_dt = if max_dt.is_finite() && max_dt >= min_dt {
            max_dt
        } else {
            min_dt
        };
        let target_cfl = if target_cfl.is_finite() && target_cfl > 0.0 {
            target_cfl
        } else {
            0.5
        };
        let max_growth = if max_growth.is_finite() && max_growth >= 1.0 {
            max_growth
        } else {
            1.0
        };
        let current_dt = if initial_dt.is_finite() {
            initial_dt.clamp(min_dt, max_dt)
        } else {
            min_dt
        };
        Self {
            target_cfl,
            min_dt,
            max_dt,
            max_growth,
            current_dt,
        }
    }

    /// Step size currently in force (the last value returned).
    pub fn current_dt(&self) -> f64 {
        self.current_dt
    }

    /// Compute and store the step size for the next step.
    pub fn dt_for_speed(&mut self, max_speed: f64, char_length: f64) -> f64 {
        let cfl_dt = if max_speed.is_finite()
            && max_speed > 0.0
            && char_length.is_finite()
            && char_length > 0.0
        {
            self.target_cfl * char_length / max_speed
        } else {
            self.max_dt
        };
        let bounded = cfl_dt.clamp(self.min_dt, self.max_dt);
        let growth_limited = bounded.min(self.current_dt * self.max_growth);
        self.current_dt = growth_limited.max(self.min_dt);
        self.current_dt
    }
}

/// Compute the BDF RHS contribution from previous time levels.
///
/// Returns `ρ/Δt × (−a1 u^n − a2 u^{n-1})` which is added to the
/// momentum RHS after multiplication by the mass matrix.
pub fn bdf_rhs_velocity(state: &BdfState, coeffs: &BdfCoefficients, rho: f64, dt: f64) -> Vec<f64> {
    let factor = rho / dt;
    state
        .velocity
        .iter()
        .zip(&state.velocity_prev)
        .map(|(&un, &unm1)| factor * (-coeffs.a1 * un - coeffs.a2 * unm1))
        .collect()
}

/// Monitor for tracking time-dependent quantities.
#[derive(Debug, Default)]
pub struct TimeMonitor {
    /// Time stamps.
    pub times: Vec<f64>,
    /// Drag coefficient history.
    pub drag: Vec<f64>,
    /// Lift coefficient history.
    pub lift: Vec<f64>,
    /// Kinetic energy: E_k = 0.5 ∫ |u|² dΩ
    pub kinetic_energy: Vec<f64>,
}

impl TimeMonitor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, time: f64, drag: f64, lift: f64, energy: f64) {
        self.times.push(time);
        self.drag.push(drag);
        self.lift.push(lift);
        self.kinetic_energy.push(energy);
    }

    /// Estimate Strouhal number from lift coefficient oscillation.
    ///
    /// St = f L / U where f is the dominant frequency.
    /// Uses zero-crossing counting for frequency estimation.
    pub fn strouhal_number(&self, char_length: f64, char_velocity: f64) -> Option<f64> {
        if self.lift.len() < 10 {
            return None;
        }

        // Count zero crossings in the last half of the signal
        let start = self.lift.len() / 2;
        let mut crossings = 0;
        for i in start + 1..self.lift.len() {
            if self.lift[i] * self.lift[i - 1] < 0.0 {
                crossings += 1;
            }
        }

        if crossings < 2 {
            return None;
        }

        let duration = self.times.last()? - self.times[start];
        let frequency = crossings as f64 / (2.0 * duration); // half-periods → full periods
        Some(frequency * char_length / char_velocity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bdf_coefficients() {
        let c1 = BdfCoefficients::bdf1();
        assert_eq!(c1.a0, 1.0);
        assert_eq!(c1.a1, -1.0);
        assert_eq!(c1.a2, 0.0);

        let c2 = BdfCoefficients::bdf2();
        assert_eq!(c2.a0, 1.5);
        assert_eq!(c2.a1, -2.0);
        assert_eq!(c2.a2, 0.5);
    }

    #[test]
    fn test_extrapolate_velocity_bdf1() {
        let state = BdfState {
            velocity: vec![1.0, 2.0],
            velocity_prev: vec![0.5, 1.5],
            pressure: vec![],
            time: 0.0,
            step: 0,
        };
        let u_star = extrapolate_velocity(&state, 1);
        assert_eq!(u_star, vec![1.0, 2.0]); // BDF1: u* = u^n
    }

    #[test]
    fn test_extrapolate_velocity_bdf2() {
        let state = BdfState {
            velocity: vec![1.0, 2.0],
            velocity_prev: vec![0.5, 1.5],
            pressure: vec![],
            time: 0.0,
            step: 1, // need at least 1 previous step for BDF2
        };
        let u_star = extrapolate_velocity(&state, 2);
        // u* = 2*u^n - u^{n-1} = 2*[1,2] - [0.5,1.5] = [1.5, 2.5]
        assert_eq!(u_star, vec![1.5, 2.5]);
    }

    #[test]
    fn test_bdf_rhs() {
        let state = BdfState {
            velocity: vec![1.0, 2.0],
            velocity_prev: vec![0.5, 1.5],
            pressure: vec![],
            time: 0.0,
            step: 1,
        };
        let coeffs = BdfCoefficients::bdf2();
        let rhs = bdf_rhs_velocity(&state, &coeffs, 1.0, 0.1);
        // rhs = (1/0.1) * (-(-2)*[1,2] - 0.5*[0.5,1.5])
        //     = 10 * (2*[1,2] - 0.5*[0.5,1.5])
        //     = 10 * ([2,4] - [0.25,0.75])
        //     = 10 * [1.75, 3.25]
        //     = [17.5, 32.5]
        assert!((rhs[0] - 17.5).abs() < 1e-10);
        assert!((rhs[1] - 32.5).abs() < 1e-10);
    }

    #[test]
    fn test_bdf2_variable_matches_constant_at_unit_ratio() {
        let variable = BdfCoefficients::bdf2_variable(1.0);
        let constant = BdfCoefficients::bdf2();
        assert_eq!(variable.a0, constant.a0);
        assert_eq!(variable.a1, constant.a1);
        assert_eq!(variable.a2, constant.a2);
    }

    #[test]
    fn test_bdf2_variable_coefficients() {
        // r = 2: a0 = 5/3, a1 = -3, a2 = 4/3, and the weights sum to
        // zero (constants are differentiated exactly).
        let c = BdfCoefficients::bdf2_variable(2.0);
        assert!((c.a0 - 5.0 / 3.0).abs() < 1e-12);
        assert!((c.a1 + 3.0).abs() < 1e-12);
        assert!((c.a2 - 4.0 / 3.0).abs() < 1e-12);
        assert!((c.a0 + c.a1 + c.a2).abs() < 1e-12);
        // Invalid ratios fall back to the equal-step coefficients.
        let fallback = BdfCoefficients::bdf2_variable(-1.0);
        assert_eq!(fallback.a0, BdfCoefficients::bdf2().a0);
    }

    #[test]
    fn test_extrapolate_velocity_variable() {
        let state = BdfState {
            velocity: vec![1.0, 2.0],
            velocity_prev: vec![0.5, 1.5],
            pressure: vec![],
            time: 0.0,
            step: 1,
        };
        // r = 2: u* = 3u^n - 2u^{n-1} = [2.0, 3.0].
        let u_star = extrapolate_velocity_variable(&state, 2, 2.0);
        assert_eq!(u_star, vec![2.0, 3.0]);
        // r = 1 reproduces the constant-step extrapolation.
        assert_eq!(
            extrapolate_velocity_variable(&state, 2, 1.0),
            vec![1.5, 2.5]
        );
    }

    #[test]
    fn test_step_ratio_requires_previous_dt_and_completed_step() {
        let mut config = BdfConfig::bdf2(0.02, 10);
        assert_eq!(config.step_ratio(3), 1.0);
        config.previous_dt = Some(0.01);
        assert_eq!(config.step_ratio(0), 1.0);
        assert!((config.step_ratio(3) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_adaptive_controller_cfl_bounds_and_growth() {
        let mut controller = AdaptiveStepController::new(0.01, 0.5, 0.001, 0.02, 1.25);
        // Peak speed 2 m/s over h = 0.02 m at CFL 0.5 gives dt = 0.005.
        assert!((controller.dt_for_speed(2.0, 0.02) - 0.005).abs() < 1e-12);
        // A sudden jet shrinks the step immediately, floored at min_dt.
        assert!((controller.dt_for_speed(100.0, 0.02) - 0.001).abs() < 1e-12);
        // Rest relaxes only by the growth factor per step, capped at max_dt.
        assert!((controller.dt_for_speed(0.0, 0.02) - 0.00125).abs() < 1e-12);
        for _ in 0..40 {
            controller.dt_for_speed(0.0, 0.02);
        }
        assert!((controller.current_dt() - 0.02).abs() < 1e-12);
    }

    #[test]
    fn test_strouhal_detection() {
        let mut monitor = TimeMonitor::new();
        let dt = 0.01;
        let freq = 5.0; // 5 Hz
        for i in 0..200 {
            let t = i as f64 * dt;
            let cl = (2.0 * std::f64::consts::PI * freq * t).sin();
            monitor.record(t, 0.0, cl, 0.0);
        }
        let st = monitor.strouhal_number(1.0, 1.0);
        assert!(st.is_some());
        let st = st.unwrap();
        // Should detect frequency ≈ 5 Hz → St ≈ 5
        assert!((st - freq).abs() < 1.0, "St = {st}, expected ~{freq}");
    }
}
