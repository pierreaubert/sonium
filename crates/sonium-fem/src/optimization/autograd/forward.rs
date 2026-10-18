//! Forward-mode automatic differentiation with dual numbers.
//!
//! Each value carries its directional derivative alongside the primal,
//! so one evaluation computes a Jacobian-vector product. Seeding the
//! tangent with each basis vector in turn recovers a full Jacobian.
//!
//! Forward mode costs one evaluation per input direction. It suits
//! shape parametrisations with few design variables, directional
//! derivative checks, and Jacobian-vector products inside Krylov
//! methods. For many inputs and one output, reverse mode in
//! [`super::reverse`] is cheaper.

// Rust guideline compliant 2026-02-21

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Dual number carrying a value and one directional derivative.
///
/// The derivative is the directional derivative along the seeded
/// tangent direction. Constants have a zero derivative.
///
/// # Examples
///
/// ```
/// use sonium_fem::optimization::autograd::forward::Dual;
///
/// let x = Dual::variable(2.0, 1.0);
/// let y = x * x + Dual::constant(1.0);
/// assert_eq!(y.value(), 5.0);
/// assert_eq!(y.derivative(), 4.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual {
    val: f64,
    der: f64,
}

impl Dual {
    /// Creates a dual number from a value and a tangent.
    ///
    /// The tangent is the directional derivative seed for this value.
    pub fn new(val: f64, der: f64) -> Self {
        Self { val, der }
    }

    /// Creates a constant with a zero derivative.
    pub fn constant(val: f64) -> Self {
        Self { val, der: 0.0 }
    }

    /// Creates an input seeded with the given tangent.
    pub fn variable(val: f64, der: f64) -> Self {
        Self { val, der }
    }

    /// Returns the primal value.
    pub fn value(self) -> f64 {
        self.val
    }

    /// Returns the directional derivative.
    pub fn derivative(self) -> f64 {
        self.der
    }

    /// Raises the value to a constant power.
    pub fn powf(self, exponent: f64) -> Self {
        let val = self.val.powf(exponent);
        let der = if self.der == 0.0 {
            0.0
        } else {
            exponent * self.val.powf(exponent - 1.0) * self.der
        };
        Self { val, der }
    }

    /// Raises the value to a constant integer power.
    pub fn powi(self, exponent: i32) -> Self {
        let val = self.val.powi(exponent);
        let der = if self.der == 0.0 || exponent == 0 {
            0.0
        } else {
            f64::from(exponent) * self.val.powi(exponent - 1) * self.der
        };
        Self { val, der }
    }

    /// Computes the square root.
    pub fn sqrt(self) -> Self {
        let val = self.val.sqrt();
        let der = if self.der == 0.0 {
            0.0
        } else {
            self.der / (2.0 * val)
        };
        Self { val, der }
    }

    /// Computes the exponential.
    pub fn exp(self) -> Self {
        let val = self.val.exp();
        Self {
            val,
            der: val * self.der,
        }
    }

    /// Computes the natural logarithm.
    pub fn ln(self) -> Self {
        Self {
            val: self.val.ln(),
            der: self.der / self.val,
        }
    }

    /// Computes the sine.
    pub fn sin(self) -> Self {
        Self {
            val: self.val.sin(),
            der: self.val.cos() * self.der,
        }
    }

    /// Computes the cosine.
    pub fn cos(self) -> Self {
        Self {
            val: self.val.cos(),
            der: -self.val.sin() * self.der,
        }
    }

    /// Computes the hyperbolic tangent.
    pub fn tanh(self) -> Self {
        let val = self.val.tanh();
        Self {
            val,
            der: (1.0 - val * val) * self.der,
        }
    }

    /// Computes the absolute value.
    ///
    /// The derivative at zero follows the right-hand derivative.
    pub fn abs(self) -> Self {
        let sign = if self.val < 0.0 { -1.0 } else { 1.0 };
        Self {
            val: self.val.abs(),
            der: sign * self.der,
        }
    }
}

impl From<f64> for Dual {
    fn from(val: f64) -> Self {
        Self::constant(val)
    }
}

impl Add for Dual {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            val: self.val + rhs.val,
            der: self.der + rhs.der,
        }
    }
}

impl Add<f64> for Dual {
    type Output = Self;

    fn add(self, rhs: f64) -> Self {
        Self {
            val: self.val + rhs,
            der: self.der,
        }
    }
}

impl Add<Dual> for f64 {
    type Output = Dual;

    fn add(self, rhs: Dual) -> Dual {
        Dual {
            val: self + rhs.val,
            der: rhs.der,
        }
    }
}

impl AddAssign for Dual {
    fn add_assign(&mut self, rhs: Self) {
        self.val += rhs.val;
        self.der += rhs.der;
    }
}

impl AddAssign<f64> for Dual {
    fn add_assign(&mut self, rhs: f64) {
        self.val += rhs;
    }
}

impl Sub for Dual {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            val: self.val - rhs.val,
            der: self.der - rhs.der,
        }
    }
}

impl Sub<f64> for Dual {
    type Output = Self;

    fn sub(self, rhs: f64) -> Self {
        Self {
            val: self.val - rhs,
            der: self.der,
        }
    }
}

impl Sub<Dual> for f64 {
    type Output = Dual;

    fn sub(self, rhs: Dual) -> Dual {
        Dual {
            val: self - rhs.val,
            der: -rhs.der,
        }
    }
}

impl SubAssign for Dual {
    fn sub_assign(&mut self, rhs: Self) {
        self.val -= rhs.val;
        self.der -= rhs.der;
    }
}

impl SubAssign<f64> for Dual {
    fn sub_assign(&mut self, rhs: f64) {
        self.val -= rhs;
    }
}

impl Mul for Dual {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self {
            val: self.val * rhs.val,
            der: self.der * rhs.val + self.val * rhs.der,
        }
    }
}

impl Mul<f64> for Dual {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        Self {
            val: self.val * rhs,
            der: self.der * rhs,
        }
    }
}

impl Mul<Dual> for f64 {
    type Output = Dual;

    fn mul(self, rhs: Dual) -> Dual {
        Dual {
            val: self * rhs.val,
            der: self * rhs.der,
        }
    }
}

impl MulAssign for Dual {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl MulAssign<f64> for Dual {
    fn mul_assign(&mut self, rhs: f64) {
        self.val *= rhs;
        self.der *= rhs;
    }
}

impl Div for Dual {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        Self {
            val: self.val / rhs.val,
            der: (self.der * rhs.val - self.val * rhs.der) / (rhs.val * rhs.val),
        }
    }
}

impl Div<f64> for Dual {
    type Output = Self;

    fn div(self, rhs: f64) -> Self {
        Self {
            val: self.val / rhs,
            der: self.der / rhs,
        }
    }
}

impl Div<Dual> for f64 {
    type Output = Dual;

    fn div(self, rhs: Dual) -> Dual {
        Dual {
            val: self / rhs.val,
            der: -self * rhs.der / (rhs.val * rhs.val),
        }
    }
}

impl DivAssign for Dual {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl DivAssign<f64> for Dual {
    fn div_assign(&mut self, rhs: f64) {
        self.val /= rhs;
        self.der /= rhs;
    }
}

impl Neg for Dual {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            val: -self.val,
            der: -self.der,
        }
    }
}

/// Computes a Jacobian-vector product for a scalar function.
///
/// Seeds each input tangent from `direction` and returns the primal
/// value together with the directional derivative.
///
/// # Panics
///
/// Panics if `direction` does not have the same length as `point`.
pub fn jvp<F>(function: F, point: &[f64], direction: &[f64]) -> (f64, f64)
where
    F: Fn(&[Dual]) -> Dual,
{
    assert_eq!(
        point.len(),
        direction.len(),
        "direction length must match point length"
    );
    let seeded: Vec<Dual> = point
        .iter()
        .zip(direction.iter())
        .map(|(&val, &der)| Dual::variable(val, der))
        .collect();
    let output = function(&seeded);
    (output.value(), output.derivative())
}

/// Computes the gradient of a scalar function by forward mode.
///
/// Costs one seeded evaluation per input. Prefer
/// [`super::reverse::gradient`] when the input count is large.
///
/// # Panics
///
/// Panics if the function does not return a consistent primal when
/// seeded with different tangents. This indicates a bug in the caller.
pub fn gradient<F>(function: F, point: &[f64]) -> (f64, Vec<f64>)
where
    F: Fn(&[Dual]) -> Dual,
{
    let mut gradient = Vec::with_capacity(point.len());
    let mut primal: Option<f64> = None;
    for axis in 0..point.len() {
        let seeded: Vec<Dual> = point
            .iter()
            .enumerate()
            .map(|(index, &val)| Dual::variable(val, if index == axis { 1.0 } else { 0.0 }))
            .collect();
        let output = function(&seeded);
        match primal {
            None => primal = Some(output.value()),
            Some(expected) => assert_eq!(
                output.value(),
                expected,
                "forward gradient primal changed with the seed"
            ),
        }
        gradient.push(output.derivative());
    }
    (primal.unwrap_or(0.0), gradient)
}

/// Computes a Jacobian-vector product for a vector function.
pub fn jvp_vec<F>(function: F, point: &[f64], direction: &[f64]) -> (Vec<f64>, Vec<f64>)
where
    F: Fn(&[Dual]) -> Vec<Dual>,
{
    assert_eq!(
        point.len(),
        direction.len(),
        "direction length must match point length"
    );
    let seeded: Vec<Dual> = point
        .iter()
        .zip(direction.iter())
        .map(|(&val, &der)| Dual::variable(val, der))
        .collect();
    let output = function(&seeded);
    (
        output.iter().map(|dual| dual.value()).collect(),
        output.iter().map(|dual| dual.derivative()).collect(),
    )
}

/// Computes the Jacobian of a vector function by forward mode.
///
/// The result is row-major: `jacobian[row][column]` is the partial
/// derivative of output `row` with respect to input `column`. Costs
/// one seeded evaluation per input column.
pub fn jacobian<F>(function: F, point: &[f64]) -> (Vec<f64>, Vec<Vec<f64>>)
where
    F: Fn(&[Dual]) -> Vec<Dual>,
{
    let (primal, first_column) = jvp_vec(&function, point, &basis(point.len(), 0));
    let output_count = primal.len();
    let mut columns: Vec<Vec<f64>> = Vec::with_capacity(point.len());
    columns.push(first_column);
    for axis in 1..point.len() {
        let (_, column) = jvp_vec(&function, point, &basis(point.len(), axis));
        columns.push(column);
    }
    let mut jacobian = vec![vec![0.0; point.len()]; output_count];
    for (column_index, column) in columns.iter().enumerate() {
        for (row_index, &value) in column.iter().enumerate() {
            jacobian[row_index][column_index] = value;
        }
    }
    (primal, jacobian)
}

fn basis(len: usize, axis: usize) -> Vec<f64> {
    let mut direction = vec![0.0; len];
    if axis < len {
        direction[axis] = 1.0;
    }
    direction
}

#[cfg(test)]
mod tests {
    use super::*;

    fn polynomial(inputs: &[Dual]) -> Dual {
        inputs[0] * inputs[0] * inputs[1] + inputs[1].sin()
    }

    #[test]
    fn dual_arithmetic_matches_finite_difference() {
        let point = [1.3, 0.7];
        let (value, directional) = jvp(polynomial, &point, &[1.0, 0.0]);
        let expected_value = 1.3 * 1.3 * 0.7 + 0.7_f64.sin();
        assert!((value - expected_value).abs() < 1e-14);
        // d/dx (x^2 y + sin y) = 2 x y.
        assert!((directional - 2.0 * 1.3 * 0.7).abs() < 1e-14);
    }

    #[test]
    fn forward_gradient_matches_analytic() {
        let (_, gradient) = gradient(polynomial, &[1.3, 0.7]);
        assert!((gradient[0] - 2.0 * 1.3 * 0.7).abs() < 1e-14);
        assert!((gradient[1] - (1.3 * 1.3 + 0.7_f64.cos())).abs() < 1e-14);
    }

    #[test]
    fn forward_jacobian_matches_analytic() {
        let function = |inputs: &[Dual]| vec![inputs[0] * inputs[1], inputs[0] + inputs[1].exp()];
        let (primal, jacobian) = jacobian(function, &[2.0, 0.5]);
        assert!((primal[0] - 1.0).abs() < 1e-14);
        assert!((primal[1] - (2.0 + 0.5_f64.exp())).abs() < 1e-14);
        assert!((jacobian[0][0] - 0.5).abs() < 1e-14);
        assert!((jacobian[0][1] - 2.0).abs() < 1e-14);
        assert!((jacobian[1][0] - 1.0).abs() < 1e-14);
        assert!((jacobian[1][1] - 0.5_f64.exp()).abs() < 1e-14);
    }

    #[test]
    fn mixed_scalar_operations_propagate_tangent() {
        let x = Dual::variable(4.0, 1.0);
        let y = 2.0 * x.sqrt() + 8.0 / x;
        assert!((y.value() - 6.0).abs() < 1e-14);
        // d/dx (2 sqrt(x) + 8/x) = 1/sqrt(x) - 8/x^2 = 0.5 - 0.5 = 0.
        assert!(y.derivative().abs() < 1e-14);
    }
}
