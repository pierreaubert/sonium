//! Automatic differentiation for FEM design optimisation.
//!
//! Two complementary modes are provided for the explicit parts of a
//! shape or topology objective (geometry, material interpolation,
//! and post-processing):
//!
//! - [`forward`]: dual numbers. One seeded evaluation yields a
//!   Jacobian-vector product. Best for few design variables or
//!   directional derivatives.
//! - [`reverse`]: scalar tape. One backward sweep yields the full
//!   gradient of a scalar objective. Best for many design variables.
//!
//! The PDE solve itself is not expanded into scalar operations.
//! Its derivatives use the implicit-function theorem instead: the
//! forward sensitivity solve for [`forward`] and the adjoint solve
//! for [`reverse`], bridged by [`solve_jvp`] and [`solve_vjp`].
//!
//! The adjoint bridge types [`DiffDesignVars`] and
//! [`DiffFemObjective`] are retained for objectives whose gradient
//! already comes from an adjoint solve.
//!
//! # Examples
//!
//! ```
//! use sonium_fem::optimization::autograd::{forward, reverse};
//!
//! let point = [1.3, 0.7];
//! let (_, forward_grad) = forward::gradient(
//!     |x| x[0] * x[0] * x[1] + x[1].sin(),
//!     &point,
//! );
//! let (_, reverse_grad) = reverse::gradient(
//!     |tape, x| {
//!         let product = tape.mul(tape.mul(x[0], x[0]), x[1]);
//!         tape.add(product, tape.sin(x[1]))
//!     },
//!     &point,
//! );
//! assert!((forward_grad[0] - reverse_grad[0]).abs() < 1e-14);
//! assert!((forward_grad[1] - reverse_grad[1]).abs() < 1e-14);
//! ```

// Rust guideline compliant 2026-02-21

pub mod forward;
pub mod reverse;

#[doc(inline)]
pub use forward::Dual;
#[doc(inline)]
pub use reverse::{Tape, Var};

use super::adjoint::compute_adjoint_sensitivity;

/// Differentiable design variables wrapper.
///
/// Wraps a vector of design parameters so gradients computed by
/// either differentiation mode or by an adjoint solve can be
/// stored alongside the primal values.
#[derive(Debug, Clone)]
pub struct DiffDesignVars {
    /// Current values.
    pub values: Vec<f64>,
    /// Accumulated gradients (populated after backward pass).
    pub gradients: Option<Vec<f64>>,
}

impl DiffDesignVars {
    /// Creates design variables with no recorded gradient.
    pub fn new(values: Vec<f64>) -> Self {
        Self {
            values,
            gradients: None,
        }
    }

    /// Returns the number of design variables.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns whether there are no design variables.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Seeds dual numbers along the given tangent direction.
    ///
    /// # Panics
    ///
    /// Panics if `direction` does not match the variable count.
    pub fn duals(&self, direction: &[f64]) -> Vec<Dual> {
        assert_eq!(
            self.values.len(),
            direction.len(),
            "direction length must match variable count"
        );
        self.values
            .iter()
            .zip(direction.iter())
            .map(|(&value, &tangent)| Dual::variable(value, tangent))
            .collect()
    }

    /// Records the variables as inputs on a reverse-mode tape.
    pub fn record_on(&self, tape: &Tape) -> Vec<Var> {
        self.values.iter().map(|&value| tape.var(value)).collect()
    }

    /// Stores a gradient computed in either mode.
    pub fn set_gradient(&mut self, gradient: Vec<f64>) {
        debug_assert_eq!(self.values.len(), gradient.len());
        self.gradients = Some(gradient);
    }
}

/// Differentiable FEM objective from forward and gradient closures.
///
/// The gradient closure normally wraps an adjoint solve, so this
/// type remains the bridge for PDE-constrained objectives whose
/// derivative is not a scalar tape expression.
pub struct DiffFemObjective<F, G> {
    /// Forward solve: design_vars → objective_value
    pub forward: F,
    /// Gradient: design_vars → dF/dp (via adjoint)
    pub gradient: G,
}

impl<F, G> DiffFemObjective<F, G>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    /// Creates an objective from forward and gradient closures.
    pub fn new(forward: F, gradient: G) -> Self {
        Self { forward, gradient }
    }

    /// Evaluates the objective at the stored variable values.
    pub fn evaluate(&self, vars: &DiffDesignVars) -> f64 {
        (self.forward)(&vars.values)
    }

    /// Computes gradients and stores them in the design variables.
    pub fn backward(&self, vars: &mut DiffDesignVars) {
        vars.gradients = Some((self.gradient)(&vars.values));
    }
}

/// Computes the forward sensitivity of a linear solve.
///
/// For `K(p) u = b(p)` and a design direction `v`, the state
/// tangent solves `K du = db_dir - dK_dir u`, where `dK_dir u` is
/// `(dK/dp · v) u` and `db_dir` is `(db/dp · v)`. The caller
/// supplies both directional products and a solver for `K`.
///
/// This is the forward-mode counterpart of the adjoint method:
/// one solve per design direction.
pub fn solve_jvp<S>(
    dk_dp_times_u_direction: &[f64],
    db_dp_direction: Option<&[f64]>,
    linear_solve: S,
) -> Vec<f64>
where
    S: Fn(&[f64]) -> Vec<f64>,
{
    let rhs: Vec<f64> = dk_dp_times_u_direction
        .iter()
        .enumerate()
        .map(|(index, dk_u)| db_dp_direction.map_or(0.0, |db| db[index]) - dk_u)
        .collect();
    linear_solve(&rhs)
}

/// Computes the reverse sensitivity of a linear solve.
///
/// For `K(p) u = b(p)` and a scalar objective `F(u, p)`, one
/// adjoint solve `Kᵀ λ = -∂F/∂u` yields every component of
/// `dF/dp`. Each entry of `residual_derivatives` must contain
/// `(∂K/∂pᵢ) u - ∂b/∂pᵢ`, and `explicit_derivatives` holds
/// `∂F/∂pᵢ` at fixed state. This delegates to
/// [`compute_adjoint_sensitivity`] for real systems; for complex
/// systems use the Hermitian variant in [`super::adjoint`].
///
/// This is the reverse-mode counterpart of [`solve_jvp`]: one
/// solve for the full gradient, independent of the design count.
pub fn solve_vjp<S>(
    df_du: &[f64],
    residual_derivatives: &[Vec<f64>],
    explicit_derivatives: &[f64],
    linear_solve_transpose: S,
) -> Vec<f64>
where
    S: Fn(&[f64]) -> Vec<f64>,
{
    assert_eq!(
        residual_derivatives.len(),
        explicit_derivatives.len(),
        "one explicit derivative is required per residual derivative"
    );
    // Residual form: dF/dp = explicit + λᵀ (dK/dp u - db/dp).
    // compute_adjoint_sensitivity already implements λᵀ (dK/dp u),
    // so fold the explicit part in afterwards.
    let result = compute_adjoint_sensitivity(df_du, residual_derivatives, linear_solve_transpose);
    result
        .sensitivities
        .iter()
        .zip(explicit_derivatives.iter())
        .map(|(sensitivity, explicit)| sensitivity + explicit)
        .collect()
}

/// Differentiable SIMP interpolation.
///
/// Maps density `rho` to effective stiffness ratio and derivative.
pub fn simp_interpolation(rho: f64, penalty: f64) -> (f64, f64) {
    let val = rho.powf(penalty);
    let deriv = penalty * rho.powf(penalty - 1.0);
    (val, deriv)
}

/// Applies the chain rule between matrix and design sensitivities.
///
/// Connects the adjoint-computed sensitivity with respect to the
/// assembled matrix to the design parameter sensitivity.
pub fn chain_rule_sensitivity(df_dk_elements: &[f64], dk_dp_elements: &[f64]) -> Vec<f64> {
    df_dk_elements
        .iter()
        .zip(dk_dp_elements)
        .map(|(&df_dk, &dk_dp)| df_dk * dk_dp)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diff_design_vars() {
        let vars = DiffDesignVars::new(vec![0.5, 0.8, 1.0]);
        assert_eq!(vars.len(), 3);
        assert!(vars.gradients.is_none());
    }

    #[test]
    fn test_diff_fem_objective() {
        let obj = DiffFemObjective::new(
            |p: &[f64]| p.iter().map(|x| x * x).sum::<f64>(),
            |p: &[f64]| p.iter().map(|x| 2.0 * x).collect(),
        );

        let mut vars = DiffDesignVars::new(vec![1.0, 2.0, 3.0]);
        let val = obj.evaluate(&vars);
        assert!((val - 14.0).abs() < 1e-14);

        obj.backward(&mut vars);
        let grad = vars.gradients.unwrap();
        assert!((grad[0] - 2.0).abs() < 1e-14);
        assert!((grad[1] - 4.0).abs() < 1e-14);
        assert!((grad[2] - 6.0).abs() < 1e-14);
    }

    #[test]
    fn test_chain_rule() {
        let df_dk = vec![1.0, 2.0, 3.0];
        let dk_dp = vec![0.5, 0.5, 0.5];
        let df_dp = chain_rule_sensitivity(&df_dk, &dk_dp);
        assert!((df_dp[0] - 0.5).abs() < 1e-14);
        assert!((df_dp[1] - 1.0).abs() < 1e-14);
        assert!((df_dp[2] - 1.5).abs() < 1e-14);
    }

    #[test]
    fn test_simp_interpolation() {
        let (val, deriv) = simp_interpolation(0.5, 3.0);
        assert!((val - 0.125).abs() < 1e-14);
        assert!((deriv - 0.75).abs() < 1e-14);
    }

    #[test]
    fn forward_and_reverse_gradients_agree() {
        // Shape-like objective: area-weighted compliance surrogate.
        // J(L, A) = 1/(L*A) + L*A, with analytic gradient known.
        let point = [0.2, 1.0e-3];
        let (_, forward_grad) = forward::gradient(
            |x| Dual::constant(1.0) / (x[0] * x[1]) + x[0] * x[1],
            &point,
        );
        let (_, reverse_grad) = reverse::gradient(
            |tape: &Tape, x| {
                let product = tape.mul(x[0], x[1]);
                let one = tape.constant(1.0);
                tape.add(tape.div(one, product), product)
            },
            &point,
        );
        for index in 0..2 {
            assert!(
                (forward_grad[index] - reverse_grad[index]).abs() < 1e-9,
                "forward={} reverse={}",
                forward_grad[index],
                reverse_grad[index]
            );
        }
        // dJ/dL = -1/(L^2 A) + A, dJ/dA = -1/(L A^2) + L.
        let expected_l = -1.0 / (0.2 * 0.2 * 1.0e-3) + 1.0e-3;
        let expected_a = -1.0 / (0.2 * 1.0e-3 * 1.0e-3) + 0.2;
        assert!((reverse_grad[0] - expected_l).abs() / expected_l.abs() < 1e-12);
        assert!((reverse_grad[1] - expected_a).abs() / expected_a.abs() < 1e-12);
    }

    #[test]
    fn solve_jvp_and_vjp_agree_on_small_system() {
        // K(p) = [[2+p, -1], [-1, 2]], b = [1, 0], F = u[0].
        // At p = 0: u = [2/3, 1/3], du/dp = -K^-1 (dK/dp) u.
        // dK/dp = [[1, 0], [0, 0]], (dK/dp) u = [2/3, 0].
        // K^-1 = (1/3)[[2, 1], [1, 2]], so du = [-(4/9), -(2/9)].
        // Forward mode gives du directly; reverse mode gives
        // dF/dp = du[0] = -4/9.
        let solve =
            |rhs: &[f64]| vec![(2.0 * rhs[0] + rhs[1]) / 3.0, (rhs[0] + 2.0 * rhs[1]) / 3.0];
        let dk_u = vec![2.0 / 3.0, 0.0];
        let du = solve_jvp(&dk_u, None, solve);
        assert!((du[0] - (-4.0 / 9.0)).abs() < 1e-14);
        assert!((du[1] - (-2.0 / 9.0)).abs() < 1e-14);

        let gradient = solve_vjp(&[1.0, 0.0], &[dk_u], &[0.0], solve);
        assert!((gradient[0] - (-4.0 / 9.0)).abs() < 1e-14);
        assert!((gradient[0] - du[0]).abs() < 1e-14);
    }

    #[test]
    fn both_modes_match_finite_difference() {
        // Non-trivial chain: exp, sqrt, division, and tanh.
        let plain = |x: &[f64]| (x[0] * x[1]).exp() / (x[0] + x[1]).sqrt() + (x[0] - x[1]).tanh();
        let point = [0.6, 0.9];
        let eps = 1e-6;
        let mut finite_difference = Vec::new();
        for axis in 0..2 {
            let mut plus = point;
            let mut minus = point;
            plus[axis] += eps;
            minus[axis] -= eps;
            finite_difference.push((plain(&plus) - plain(&minus)) / (2.0 * eps));
        }
        let (_, forward_grad) = forward::gradient(
            |x| (x[0] * x[1]).exp() / (x[0] + x[1]).sqrt() + (x[0] - x[1]).tanh(),
            &point,
        );
        let (_, reverse_grad) = reverse::gradient(
            |tape, x| {
                let ratio = tape.div(
                    tape.exp(tape.mul(x[0], x[1])),
                    tape.sqrt(tape.add(x[0], x[1])),
                );
                tape.add(ratio, tape.tanh(tape.sub(x[0], x[1])))
            },
            &point,
        );
        for axis in 0..2 {
            assert!(
                (forward_grad[axis] - finite_difference[axis]).abs() < 1e-6,
                "forward[{axis}]={} fd={}",
                forward_grad[axis],
                finite_difference[axis]
            );
            assert!(
                (reverse_grad[axis] - finite_difference[axis]).abs() < 1e-6,
                "reverse[{axis}]={} fd={}",
                reverse_grad[axis],
                finite_difference[axis]
            );
        }
    }

    #[test]
    fn design_vars_seed_both_modes() {
        let mut vars = DiffDesignVars::new(vec![2.0, 3.0]);
        let duals = vars.duals(&[1.0, 0.0]);
        assert_eq!(duals[0].derivative(), 1.0);
        assert_eq!(duals[1].derivative(), 0.0);

        let tape = Tape::new();
        let handles = vars.record_on(&tape);
        let product = tape.mul(handles[0], handles[1]);
        vars.set_gradient(tape.backward(product, &handles));
        assert_eq!(vars.gradients.unwrap(), vec![3.0, 2.0]);
    }
}
