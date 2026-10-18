//! Reverse-mode automatic differentiation on a scalar tape.
//!
//! Operations record their local partial derivatives on a Wengert
//! list during the forward pass. One backward sweep from a scalar
//! output then yields the gradient with respect to every input,
//! regardless of the input count.
//!
//! Reverse mode costs one backward sweep per output. It suits
//! shape and topology objectives with many design variables and
//! one scalar objective. For vector outputs it costs one sweep per
//! output row; prefer [`super::forward`] when inputs are few.

// Rust guideline compliant 2026-02-21

/// Handle to a value recorded on a [`Tape`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Var(usize);

impl Var {
    /// Returns the zero-based tape position of this value.
    pub fn index(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone)]
struct Node {
    value: f64,
    // (parent index, local partial d(node)/d(parent)) pairs.
    parents: Vec<(usize, f64)>,
}

/// Scalar reverse-mode tape that records differentiable operations.
///
/// Methods take a shared reference backed by interior mutability,
/// so calls nest naturally, as in `tape.add(tape.mul(x, x), y)`.
///
/// Build a graph with [`Tape::var`] and the arithmetic methods,
/// then call [`Tape::backward`] or the free [`gradient`] helper.
///
/// # Examples
///
/// ```
/// use sonium_fem::optimization::autograd::reverse::Tape;
///
/// let tape = Tape::new();
/// let x = tape.var(2.0);
/// let y = tape.var(3.0);
/// let z = tape.add(tape.mul(x, x), y);
/// let gradients = tape.backward(z, &[x, y]);
/// assert_eq!(tape.value(z), 7.0);
/// assert_eq!(gradients, vec![4.0, 1.0]);
/// ```
#[derive(Debug, Default)]
pub struct Tape {
    nodes: std::cell::RefCell<Vec<Node>>,
}

impl Tape {
    /// Creates an empty tape.
    pub fn new() -> Self {
        Self {
            nodes: std::cell::RefCell::new(Vec::new()),
        }
    }

    /// Records an input value and returns its handle.
    pub fn var(&self, value: f64) -> Var {
        self.push(value, Vec::new())
    }

    /// Records a constant value and returns its handle.
    ///
    /// Constants are recorded like inputs; their gradient is simply
    /// never queried by callers.
    pub fn constant(&self, value: f64) -> Var {
        self.push(value, Vec::new())
    }

    /// Returns the primal value recorded for `var`.
    ///
    /// # Panics
    ///
    /// Panics if `var` was recorded on a different tape.
    pub fn value(&self, var: Var) -> f64 {
        self.nodes.borrow()[var.0].value
    }

    /// Returns the number of recorded values.
    pub fn len(&self) -> usize {
        self.nodes.borrow().len()
    }

    /// Returns whether the tape has recorded no values.
    pub fn is_empty(&self) -> bool {
        self.nodes.borrow().is_empty()
    }

    fn push(&self, value: f64, parents: Vec<(usize, f64)>) -> Var {
        let mut nodes = self.nodes.borrow_mut();
        let var = Var(nodes.len());
        nodes.push(Node { value, parents });
        var
    }

    fn unary(&self, input: Var, value: f64, partial: f64) -> Var {
        self.push(value, vec![(input.0, partial)])
    }

    /// Records `lhs + rhs`.
    pub fn add(&self, lhs: Var, rhs: Var) -> Var {
        let value = self.value(lhs) + self.value(rhs);
        self.push(value, vec![(lhs.0, 1.0), (rhs.0, 1.0)])
    }

    /// Records `lhs - rhs`.
    pub fn sub(&self, lhs: Var, rhs: Var) -> Var {
        let value = self.value(lhs) - self.value(rhs);
        self.push(value, vec![(lhs.0, 1.0), (rhs.0, -1.0)])
    }

    /// Records `lhs * rhs`.
    pub fn mul(&self, lhs: Var, rhs: Var) -> Var {
        let lhs_value = self.value(lhs);
        let rhs_value = self.value(rhs);
        self.push(
            lhs_value * rhs_value,
            vec![(lhs.0, rhs_value), (rhs.0, lhs_value)],
        )
    }

    /// Records `lhs / rhs`.
    pub fn div(&self, lhs: Var, rhs: Var) -> Var {
        let lhs_value = self.value(lhs);
        let rhs_value = self.value(rhs);
        self.push(
            lhs_value / rhs_value,
            vec![
                (lhs.0, 1.0 / rhs_value),
                (rhs.0, -lhs_value / (rhs_value * rhs_value)),
            ],
        )
    }

    /// Records `-input`.
    pub fn neg(&self, input: Var) -> Var {
        let value = -self.value(input);
        self.unary(input, value, -1.0)
    }

    /// Records `input` raised to a constant power.
    pub fn powf(&self, input: Var, exponent: f64) -> Var {
        let base = self.value(input);
        let value = base.powf(exponent);
        let partial = if exponent == 0.0 {
            0.0
        } else {
            exponent * base.powf(exponent - 1.0)
        };
        self.unary(input, value, partial)
    }

    /// Records `input` raised to a constant integer power.
    pub fn powi(&self, input: Var, exponent: i32) -> Var {
        let base = self.value(input);
        let value = base.powi(exponent);
        let partial = if exponent == 0 {
            0.0
        } else {
            f64::from(exponent) * base.powi(exponent - 1)
        };
        self.unary(input, value, partial)
    }

    /// Records the square root of `input`.
    pub fn sqrt(&self, input: Var) -> Var {
        let value = self.value(input).sqrt();
        // Singular at zero and NaN for negative inputs, as in f64.
        self.unary(input, value, 0.5 / value)
    }

    /// Records the exponential of `input`.
    pub fn exp(&self, input: Var) -> Var {
        let value = self.value(input).exp();
        self.unary(input, value, value)
    }

    /// Records the natural logarithm of `input`.
    pub fn ln(&self, input: Var) -> Var {
        let base = self.value(input);
        self.unary(input, base.ln(), 1.0 / base)
    }

    /// Records the sine of `input`.
    pub fn sin(&self, input: Var) -> Var {
        let base = self.value(input);
        self.unary(input, base.sin(), base.cos())
    }

    /// Records the cosine of `input`.
    pub fn cos(&self, input: Var) -> Var {
        let base = self.value(input);
        self.unary(input, base.cos(), -base.sin())
    }

    /// Records the hyperbolic tangent of `input`.
    pub fn tanh(&self, input: Var) -> Var {
        let value = self.value(input).tanh();
        self.unary(input, value, 1.0 - value * value)
    }

    /// Records a unary operation with a caller-supplied derivative.
    ///
    /// The `forward` closure maps the primal input to the primal
    /// output, and `partial` is `d(output)/d(input)` at that point.
    /// This is the extension point for custom primitives whose
    /// derivative is known analytically, such as a linear-solve
    /// objective reduced through the adjoint method.
    pub fn unary_custom<F>(&self, input: Var, forward: F, partial: f64) -> Var
    where
        F: Fn(f64) -> f64,
    {
        let value = forward(self.value(input));
        self.unary(input, value, partial)
    }

    /// Records a node from explicit local partials.
    ///
    /// Each entry pairs a parent handle with `d(output)/d(parent)`.
    /// Use this to attach a precomputed sensitivity, for example
    /// one row of an adjoint gradient, to the tape.
    pub fn push_custom(&self, value: f64, partials: &[(Var, f64)]) -> Var {
        let parents = partials
            .iter()
            .map(|(parent, partial)| (parent.0, *partial))
            .collect();
        self.push(value, parents)
    }

    /// Runs reverse mode from `output` with unit seed.
    ///
    /// Returns the gradients with respect to `inputs` in order.
    /// One sweep yields every input gradient at once.
    ///
    /// # Panics
    ///
    /// Panics if `output` or any entry of `inputs` was recorded on
    /// a different tape.
    pub fn backward(&self, output: Var, inputs: &[Var]) -> Vec<f64> {
        let adjoints = self.backward_all(output, 1.0);
        inputs.iter().map(|input| adjoints[input.0]).collect()
    }

    /// Runs reverse mode and returns adjoints for every node.
    ///
    /// The seed is the cotangent `d(objective)/d(output)`, normally
    /// `1.0` when `output` is the scalar objective itself.
    ///
    /// # Panics
    ///
    /// Panics if `output` was recorded on a different tape.
    pub fn backward_all(&self, output: Var, seed: f64) -> Vec<f64> {
        // Snapshot the graph so no cell borrow is held during the
        // sweep; tapes for shape objectives are small.
        let nodes = self.nodes.borrow().clone();
        let mut adjoints = vec![0.0; nodes.len()];
        adjoints[output.0] = seed;
        for index in (0..=output.0).rev() {
            let adjoint = adjoints[index];
            if adjoint == 0.0 {
                continue;
            }
            for &(parent, partial) in &nodes[index].parents {
                adjoints[parent] += adjoint * partial;
            }
        }
        adjoints
    }
}

/// Computes the value and gradient of a scalar function.
///
/// The function receives a fresh tape and the input handles, and
/// returns the output handle. One backward sweep follows.
///
/// # Examples
///
/// ```
/// use sonium_fem::optimization::autograd::reverse::gradient;
///
/// let (value, grad) = gradient(|tape, x| tape.mul(x[0], x[1]), &[2.0, 3.0]);
/// assert_eq!(value, 6.0);
/// assert_eq!(grad, vec![3.0, 2.0]);
/// ```
pub fn gradient<F>(function: F, point: &[f64]) -> (f64, Vec<f64>)
where
    F: Fn(&Tape, &[Var]) -> Var,
{
    let tape = Tape::new();
    let inputs: Vec<Var> = point.iter().map(|&value| tape.var(value)).collect();
    let output = function(&tape, &inputs);
    let value = tape.value(output);
    (value, tape.backward(output, &inputs))
}

/// Computes a vector-Jacobian product for a vector function.
///
/// Seeds the backward sweep with `cotangent`, returning the primal
/// outputs and the gradient with respect to every input. This is
/// the reverse-mode counterpart of a forward Jacobian-vector
/// product and costs one sweep per call.
pub fn vjp<F>(function: F, point: &[f64], cotangent: &[f64]) -> (Vec<f64>, Vec<f64>)
where
    F: Fn(&Tape, &[Var]) -> Vec<Var>,
{
    let tape = Tape::new();
    let inputs: Vec<Var> = point.iter().map(|&value| tape.var(value)).collect();
    let outputs = function(&tape, &inputs);
    assert_eq!(
        outputs.len(),
        cotangent.len(),
        "cotangent length must match output length"
    );
    let primal: Vec<f64> = outputs.iter().map(|output| tape.value(*output)).collect();
    let mut gradient = vec![0.0; point.len()];
    // Accumulate one seeded sweep per output; the tape is reused
    // read-only, so no re-recording is needed.
    let mut combined = vec![0.0; tape.len()];
    for (output, &seed) in outputs.iter().zip(cotangent.iter()) {
        let adjoints = tape.backward_all(*output, seed);
        for (total, adjoint) in combined.iter_mut().zip(adjoints.iter()) {
            *total += *adjoint;
        }
    }
    for (slot, input) in gradient.iter_mut().zip(inputs.iter()) {
        *slot = combined[input.0];
    }
    (primal, gradient)
}

/// Computes the Jacobian of a vector function by reverse mode.
///
/// The result is row-major: `jacobian[row][column]` is the partial
/// derivative of output `row` with respect to input `column`. Costs
/// one backward sweep per output row. Prefer
/// [`super::forward::jacobian`] when outputs outnumber inputs.
pub fn jacobian<F>(function: F, point: &[f64]) -> (Vec<f64>, Vec<Vec<f64>>)
where
    F: Fn(&Tape, &[Var]) -> Vec<Var>,
{
    let tape = Tape::new();
    let inputs: Vec<Var> = point.iter().map(|&value| tape.var(value)).collect();
    let outputs = function(&tape, &inputs);
    let primal: Vec<f64> = outputs.iter().map(|output| tape.value(*output)).collect();
    let jacobian: Vec<Vec<f64>> = outputs
        .iter()
        .map(|output| tape.backward(*output, &inputs))
        .collect();
    (primal, jacobian)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reverse_gradient_matches_analytic() {
        // f(x, y) = x^2 * y + sin(y)
        let function = |tape: &Tape, x: &[Var]| {
            let x_squared = tape.mul(x[0], x[0]);
            let product = tape.mul(x_squared, x[1]);
            tape.add(product, tape.sin(x[1]))
        };
        let (value, gradient) = gradient(function, &[1.3, 0.7]);
        let expected_value = 1.3 * 1.3 * 0.7 + 0.7_f64.sin();
        assert!((value - expected_value).abs() < 1e-14);
        assert!((gradient[0] - 2.0 * 1.3 * 0.7).abs() < 1e-14);
        assert!((gradient[1] - (1.3 * 1.3 + 0.7_f64.cos())).abs() < 1e-14);
    }

    #[test]
    fn reverse_jacobian_matches_analytic() {
        let function =
            |tape: &Tape, x: &[Var]| vec![tape.mul(x[0], x[1]), tape.add(x[0], tape.exp(x[1]))];
        let (primal, jacobian) = jacobian(function, &[2.0, 0.5]);
        assert!((primal[0] - 1.0).abs() < 1e-14);
        assert!((jacobian[0][0] - 0.5).abs() < 1e-14);
        assert!((jacobian[0][1] - 2.0).abs() < 1e-14);
        assert!((jacobian[1][0] - 1.0).abs() < 1e-14);
        assert!((jacobian[1][1] - 0.5_f64.exp()).abs() < 1e-14);
    }

    #[test]
    fn division_and_pow_propagate_correctly() {
        // f(x) = x^3 / (x + 1) at x = 2: f = 8/3, f' = (3x^2(x+1)-x^3)/(x+1)^2 = 28/9.
        let function = |tape: &Tape, x: &[Var]| {
            let numerator = tape.powf(x[0], 3.0);
            let one = tape.constant(1.0);
            let denominator = tape.add(x[0], one);
            tape.div(numerator, denominator)
        };
        let (value, gradient) = gradient(function, &[2.0]);
        assert!((value - 8.0 / 3.0).abs() < 1e-14);
        assert!((gradient[0] - 28.0 / 9.0).abs() < 1e-14);
    }

    #[test]
    fn shared_subexpression_accumulates_adjoint() {
        // f(x) = (x + x) * (x + x) = 4x^2, f'(2) = 16.
        let function = |tape: &Tape, x: &[Var]| {
            let doubled = tape.add(x[0], x[0]);
            tape.mul(doubled, doubled)
        };
        let (_, gradient) = gradient(function, &[2.0]);
        assert!((gradient[0] - 16.0).abs() < 1e-14);
    }

    #[test]
    fn vjp_matches_seeded_gradient() {
        let function = |tape: &Tape, x: &[Var]| vec![tape.mul(x[0], x[1]), tape.add(x[0], x[1])];
        let (_, gradient) = vjp(function, &[2.0, 3.0], &[1.0, 10.0]);
        // d/dx = 1*3 + 10*1 = 13, d/dy = 1*2 + 10*1 = 12.
        assert_eq!(gradient, vec![13.0, 12.0]);
    }
}
