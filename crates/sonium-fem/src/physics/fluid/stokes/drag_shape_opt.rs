//! Minimum-drag shape optimisation for steady incompressible flow.
//!
//! The obstacle starts as the circular cylinder of the Schäfer–Turek
//! channel benchmark and its profile is deformed to minimise drag at
//! fixed cross-sectional area. This is the Navier–Stokes counterpart of
//! the Stokes benchmark in `examples/stokes_shape_opt.rs`: Pironneau
//! (1973) proved the Stokes optimum is an ogival body, and the same
//! area-constrained drag minimisation is applied here to the steady
//! incompressible Navier–Stokes equations at moderate Reynolds number,
//! with the cylinder flow demo's geometry and air properties.
//!
//! The profile uses even Fourier cosines,
//! `r(theta) = a0 + a1 cos(2 theta) + a2 cos(4 theta) + ...`, which
//! enforces fore-aft and top-bottom symmetry. Drag gradients are
//! forward finite differences of the drag functional; the fixed-area
//! problem is solved by projected-gradient descent with backtracking.
//!
//! # References
//!
//! - Pironneau, O. (1973). "On optimum profiles in Stokes flow."
//!   *J. Fluid Mech.* 59(1).
//! - Pironneau, O. (1974). "On optimum design in fluid mechanics."
//!   *J. Fluid Mech.* 64(1).
//! - Schäfer, M., & Turek, S. (1996). DFG flow-past-cylinder benchmark.

// Rust guideline compliant 2026-02-21
use std::cell::RefCell;
use std::f64::consts::PI;
use std::fmt;

use crate::basis::PolynomialDegree;
use crate::boundary::locate_dofs_geometrical;
use crate::mesh::Mesh;
use crate::mesh::generators::channel_with_cylinder_triangles;
use crate::spaces::{H1Space, VectorH1Space};

use super::navier_stokes::{
    NavierStokesConfig, compute_drag_lift, solve_picard_with_velocity_force_from_initial,
};

/// Air density at sea level, used for the demo fluid (kg/m³).
pub const AIR_DENSITY_KG_M3: f64 = 1.225;

/// Air dynamic viscosity at 15 °C, used for the demo fluid (Pa·s).
pub const AIR_VISCOSITY_PA_S: f64 = 1.81e-5;

/// Objective value returned for shapes that cannot be evaluated.
const INVALID_PENALTY: f64 = 1e10;

/// Configuration for the channel drag shape optimisation.
#[derive(Debug, Clone)]
pub struct DragShapeOptConfig {
    /// Channel length (m).
    pub channel_length: f64,
    /// Channel height (m).
    pub channel_height: f64,
    /// Obstacle centre x coordinate (m).
    pub center_x: f64,
    /// Obstacle centre y coordinate (m).
    pub center_y: f64,
    /// Initial cylinder radius (m).
    pub radius: f64,
    /// Reynolds number based on diameter and mean inlet velocity.
    pub reynolds: f64,
    /// Fluid density (kg/m³).
    pub density: f64,
    /// Fluid dynamic viscosity (Pa·s).
    pub viscosity: f64,
    /// Number of Fourier coefficients (`a0..`), at least 1.
    pub n_modes: usize,
    /// Background channel mesh divisions in x.
    pub nx: usize,
    /// Background channel mesh divisions in y.
    pub ny: usize,
    /// Finite-difference step for the drag gradient (m).
    pub fd_eps: f64,
    /// Maximum projected-gradient iterations.
    pub max_outer_iterations: usize,
    /// Maximum backtracking trials per gradient iteration.
    pub max_inner_iterations: usize,
    /// Maximum Picard iterations per flow solve.
    pub picard_iterations: usize,
    /// Picard convergence tolerance per flow solve.
    pub picard_tolerance: f64,
}

impl Default for DragShapeOptConfig {
    fn default() -> Self {
        Self {
            channel_length: 2.2,
            channel_height: 0.41,
            center_x: 0.2,
            center_y: 0.2,
            radius: 0.05,
            reynolds: 20.0,
            density: AIR_DENSITY_KG_M3,
            viscosity: AIR_VISCOSITY_PA_S,
            n_modes: 3,
            nx: 64,
            ny: 12,
            fd_eps: 1e-4,
            max_outer_iterations: 12,
            max_inner_iterations: 25,
            picard_iterations: 50,
            picard_tolerance: 1e-6,
        }
    }
}

impl DragShapeOptConfig {
    /// Mean inlet velocity implied by the Reynolds number (m/s).
    pub fn mean_velocity(&self) -> f64 {
        self.reynolds * self.viscosity / (self.density * 2.0 * self.radius)
    }

    /// Validate the configuration before any solve.
    ///
    /// # Errors
    ///
    /// Returns [`DragShapeOptError::InvalidConfig`] for non-finite or
    /// out-of-range values.
    pub fn validate(&self) -> Result<(), DragShapeOptError> {
        let positive = [
            self.channel_length,
            self.channel_height,
            self.radius,
            self.reynolds,
            self.density,
            self.viscosity,
            self.fd_eps,
        ];
        if positive.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || !self.center_x.is_finite()
            || !self.center_y.is_finite()
            || self.n_modes == 0
            || self.nx < 8
            || self.ny < 4
            || self.max_outer_iterations == 0
            || self.max_inner_iterations == 0
            || self.picard_iterations == 0
            || !(0.0..1.0).contains(&self.picard_tolerance)
        {
            return Err(DragShapeOptError::InvalidConfig);
        }
        if self.center_x - 3.0 * self.radius <= 0.0
            || self.center_x + 3.0 * self.radius >= self.channel_length
            || self.center_y - 3.0 * self.radius <= 0.0
            || self.center_y + 3.0 * self.radius >= self.channel_height
        {
            return Err(DragShapeOptError::InvalidConfig);
        }
        Ok(())
    }
}

/// Error returned by the drag shape optimisation.
#[derive(Debug, Clone, PartialEq)]
pub enum DragShapeOptError {
    /// The supplied configuration is invalid.
    InvalidConfig,
    /// A flow solve did not converge or produced a non-finite drag.
    FlowSolveFailed,
    /// The constrained optimiser failed.
    OptimiserFailed(String),
}

impl fmt::Display for DragShapeOptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig => write!(f, "invalid drag shape optimisation configuration"),
            Self::FlowSolveFailed => {
                write!(f, "Navier-Stokes flow solve did not converge")
            }
            Self::OptimiserFailed(message) => write!(f, "shape optimiser failed: {message}"),
        }
    }
}

impl std::error::Error for DragShapeOptError {}

/// Radial profile `r(theta) = a0 + a1 cos(2 theta) + a2 cos(4 theta) + ...`.
pub fn shape_radius(theta: f64, params: &[f64]) -> f64 {
    let mut r = params[0];
    for (k, &ak) in params.iter().enumerate().skip(1) {
        r += ak * (2.0 * k as f64 * theta).cos();
    }
    r
}

/// Cross-sectional area from Fourier coefficients.
///
/// `A = pi a0^2 + (pi/2) sum_{k>=1} ak^2`.
pub fn shape_area(params: &[f64]) -> f64 {
    let mut area = PI * params[0] * params[0];
    for &ak in &params[1..] {
        area += 0.5 * PI * ak * ak;
    }
    area
}

/// Gradient of [`shape_area`]: `2 pi a0` for `a0`, `pi ak` for `k >= 1`.
pub fn shape_area_gradient(params: &[f64]) -> Vec<f64> {
    let mut gradient = vec![2.0 * PI * params[0]];
    for &ak in &params[1..] {
        gradient.push(PI * ak);
    }
    gradient
}

/// Whether the profile stays positive and therefore non-self-intersecting.
pub fn shape_is_valid(params: &[f64]) -> bool {
    // 256 samples resolve every admitted Fourier mode around the profile.
    (0..256).all(|i| {
        let theta = 2.0 * PI * i as f64 / 256.0;
        shape_radius(theta, params) > 0.0
    })
}

/// One solved flow state on a deformed obstacle mesh.
#[derive(Debug, Clone)]
pub struct FlowEvaluation {
    /// Deformed fluid mesh.
    pub mesh: Mesh,
    /// Interleaved nodal velocity `[u0, v0, u1, v1, ...]` (m/s).
    pub velocity: Vec<f64>,
    /// Nodal pressure (Pa).
    pub pressure: Vec<f64>,
    /// Drag force per unit depth on the obstacle (N/m).
    pub drag: f64,
    /// Lift force per unit depth on the obstacle (N/m).
    pub lift: f64,
    /// Drag coefficient `2 F_D / (rho U^2 D)`.
    pub drag_coefficient: f64,
    /// Lift coefficient `2 F_L / (rho U^2 D)`.
    pub lift_coefficient: f64,
    /// Whether the Picard iteration converged.
    pub converged: bool,
    /// Picard iterations performed.
    pub picard_iterations: usize,
    /// Final Picard relative residual.
    pub picard_residual: f64,
}

/// One objective evaluation recorded during optimisation.
#[derive(Debug, Clone)]
pub struct DragEvaluation {
    /// Sequential evaluation index, starting at zero.
    pub evaluation: usize,
    /// Shape coefficients evaluated.
    pub params: Vec<f64>,
    /// Drag force per unit depth (N/m).
    pub drag: f64,
    /// Drag coefficient.
    pub drag_coefficient: f64,
}

/// Outcome of [`optimise_drag`].
#[derive(Debug, Clone)]
pub struct DragShapeOptResult {
    /// Initial (circular) coefficients.
    pub initial_params: Vec<f64>,
    /// Optimal coefficients.
    pub optimal_params: Vec<f64>,
    /// Solved flow on the initial cylinder.
    pub initial_flow: FlowEvaluation,
    /// Solved flow on the optimal body.
    pub optimal_flow: FlowEvaluation,
    /// Target cross-sectional area (m²).
    pub area_target: f64,
    /// Area of the optimal body (m²).
    pub optimal_area: f64,
    /// Optimal half-length over half-width.
    pub aspect_ratio: f64,
    /// Every finite drag evaluation, in evaluation order.
    pub history: Vec<DragEvaluation>,
    /// Whether the constrained optimiser reported convergence.
    pub converged: bool,
    /// Optimiser outer iterations reported by the solver.
    pub optimiser_iterations: usize,
    /// Objective evaluations reported by the solver.
    pub fun_evals: usize,
}

struct ReferenceMesh {
    mesh: Mesh,
    reference: Vec<(f64, f64)>,
    base_signed_areas: Vec<f64>,
    bc_dofs: Vec<usize>,
    bc_tolerance: f64,
    config: DragShapeOptConfig,
}

impl ReferenceMesh {
    fn new(config: &DragShapeOptConfig) -> Self {
        let mesh = channel_with_cylinder_triangles(
            0.0,
            config.channel_length,
            0.0,
            config.channel_height,
            config.nx,
            config.ny,
            config.center_x,
            config.center_y,
            config.radius,
        );
        let reference: Vec<(f64, f64)> = mesh.nodes.iter().map(|node| (node.x, node.y)).collect();
        let base_signed_areas = mesh
            .elements
            .iter()
            .map(|element| signed_area(&mesh, element.vertices()))
            .collect();

        let v_space = VectorH1Space::new(PolynomialDegree::P1, 2);
        let bc_tolerance = config.channel_length / config.nx as f64 * 0.6;
        let tol = bc_tolerance;
        let height = config.channel_height;
        let mut bc_dofs = locate_dofs_geometrical(&mesh, &v_space, |x, _y, _z| x < tol);
        bc_dofs.extend(locate_dofs_geometrical(&mesh, &v_space, |_x, y, _z| {
            y < tol || (height - y) < tol
        }));
        // Obstacle DOFs are the nodes on the reference circle; their DOF
        // indices stay valid on every deformed copy of this mesh.
        for (node, &(x, y)) in reference.iter().enumerate() {
            let distance = ((x - config.center_x).powi(2) + (y - config.center_y).powi(2)).sqrt();
            if (distance - config.radius).abs() < 1e-9 {
                bc_dofs.push(2 * node);
                bc_dofs.push(2 * node + 1);
            }
        }
        bc_dofs.sort_unstable();
        bc_dofs.dedup();

        Self {
            mesh,
            reference,
            base_signed_areas,
            bc_dofs,
            bc_tolerance,
            config: config.clone(),
        }
    }

    fn deform(&self, params: &[f64]) -> Mesh {
        let config = &self.config;
        // Blend length: three radii keeps wall and inlet nodes fixed while
        // the rings around the obstacle follow the profile smoothly.
        let blend = 3.0 * config.radius;
        let mut mesh = self.mesh.clone();
        for (node, &(x, y)) in self.reference.iter().enumerate() {
            let dx = x - config.center_x;
            let dy = y - config.center_y;
            let r = (dx * dx + dy * dy).sqrt();
            if r < 1e-15 {
                continue;
            }
            let s = (r - config.radius) / blend;
            if s >= 1.0 {
                continue;
            }
            // Cosine taper with zero value and slope at the blend edge.
            let weight = if s <= 0.0 {
                1.0
            } else {
                0.5 * (1.0 + (PI * s).cos())
            };
            let theta = dy.atan2(dx);
            let r_new = r + (shape_radius(theta, params) - config.radius) * weight;
            mesh.nodes[node].x = config.center_x + r_new * dx / r;
            mesh.nodes[node].y = config.center_y + r_new * dy / r;
        }
        mesh
    }

    fn deformed_is_valid(&self, mesh: &Mesh) -> bool {
        mesh.elements
            .iter()
            .zip(&self.base_signed_areas)
            .all(|(element, &base)| {
                let area = signed_area(mesh, element.vertices());
                area.is_finite() && area * base > 1e-24
            })
    }
}

fn signed_area(mesh: &Mesh, vertices: &[usize]) -> f64 {
    let [a, b, c] = [vertices[0], vertices[1], vertices[2]];
    let (x0, y0) = (mesh.nodes[a].x, mesh.nodes[a].y);
    let (x1, y1) = (mesh.nodes[b].x, mesh.nodes[b].y);
    let (x2, y2) = (mesh.nodes[c].x, mesh.nodes[c].y);
    0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
}

/// Solve the steady flow for one shape and integrate its drag.
///
/// # Errors
///
/// Returns [`DragShapeOptError::FlowSolveFailed`] when the Picard
/// iteration does not converge or the drag is not finite.
pub fn evaluate_flow(
    config: &DragShapeOptConfig,
    params: &[f64],
) -> Result<FlowEvaluation, DragShapeOptError> {
    config.validate()?;
    let reference = ReferenceMesh::new(config);
    evaluate_flow_on(&reference, params, None)
}

fn evaluate_flow_on(
    reference: &ReferenceMesh,
    params: &[f64],
    warm_start: Option<(&[f64], &[f64])>,
) -> Result<FlowEvaluation, DragShapeOptError> {
    let config = &reference.config;
    if params.len() != config.n_modes || !shape_is_valid(params) {
        return Err(DragShapeOptError::FlowSolveFailed);
    }
    let mesh = reference.deform(params);
    if !reference.deformed_is_valid(&mesh) {
        return Err(DragShapeOptError::FlowSolveFailed);
    }

    let v_space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let p_space = H1Space::new(PolynomialDegree::P1);
    let u_mean = config.mean_velocity();
    let u_max = 1.5 * u_mean;
    let height = config.channel_height;
    let tol = reference.bc_tolerance;
    let bc_fn = move |x: f64, y: f64, _z: f64| -> Vec<f64> {
        if x < tol {
            vec![4.0 * u_max * y * (height - y) / (height * height), 0.0]
        } else {
            vec![0.0, 0.0]
        }
    };
    let solver_config = NavierStokesConfig {
        mu: config.viscosity,
        rho: config.density,
        max_picard_iterations: config.picard_iterations,
        picard_tolerance: config.picard_tolerance,
        ..Default::default()
    };
    let result = solve_picard_with_velocity_force_from_initial(
        &mesh,
        &v_space,
        &p_space,
        &solver_config,
        &bc_fn,
        &reference.bc_dofs,
        |_, _| vec![0.0, 0.0],
        warm_start.map(|(velocity, _)| velocity),
        warm_start.map(|(_, pressure)| pressure),
    );
    // Boundary marker 4 is the obstacle surface in the channel generator.
    let (drag, lift) = compute_drag_lift(
        &mesh,
        &result.velocity,
        &result.pressure,
        config.viscosity,
        4,
    );
    let diameter = 2.0 * config.radius;
    let scale = config.density * u_mean * u_mean * diameter;
    let evaluation = FlowEvaluation {
        mesh,
        velocity: result.velocity,
        pressure: result.pressure,
        drag,
        lift,
        drag_coefficient: 2.0 * drag / scale,
        lift_coefficient: 2.0 * lift / scale,
        converged: result.converged,
        picard_iterations: result.iterations,
        picard_residual: result.residual,
    };
    if !evaluation.converged
        || !evaluation.drag.is_finite()
        || !evaluation.drag_coefficient.is_finite()
    {
        return Err(DragShapeOptError::FlowSolveFailed);
    }
    Ok(evaluation)
}

type WarmStart = RefCell<Option<(Vec<f64>, Vec<f64>)>>;

fn evaluate_drag(
    reference: &ReferenceMesh,
    params: &[f64],
    history: &RefCell<Vec<DragEvaluation>>,
    warm: &WarmStart,
    evals: &std::cell::Cell<usize>,
) -> f64 {
    evals.set(evals.get() + 1);
    let cached = warm.borrow().clone();
    let warm_ref = cached
        .as_ref()
        .map(|(velocity, pressure)| (velocity.as_slice(), pressure.as_slice()));
    match evaluate_flow_on(reference, params, warm_ref) {
        Ok(flow) => {
            *warm.borrow_mut() = Some((flow.velocity.clone(), flow.pressure.clone()));
            let mut history = history.borrow_mut();
            let evaluation = history.len();
            history.push(DragEvaluation {
                evaluation,
                params: params.to_vec(),
                drag: flow.drag,
                drag_coefficient: flow.drag_coefficient,
            });
            flow.drag
        }
        Err(_) => INVALID_PENALTY,
    }
}

fn fd_gradient(
    reference: &ReferenceMesh,
    params: &[f64],
    drag: f64,
    eps: f64,
    history: &RefCell<Vec<DragEvaluation>>,
    warm: &WarmStart,
    evals: &std::cell::Cell<usize>,
) -> Vec<f64> {
    let mut gradient = vec![0.0; params.len()];
    for i in 0..params.len() {
        let mut plus = params.to_vec();
        plus[i] += eps;
        let fp = evaluate_drag(reference, &plus, history, warm, evals);
        gradient[i] = if fp >= INVALID_PENALTY {
            0.0
        } else {
            (fp - drag) / eps
        };
    }
    gradient
}

/// Project coefficients onto the fixed-area manifold in closed form.
///
/// With `A = pi a0^2 + (pi/2) sum ak^2`, only `a0` needs adjustment.
fn project_area(params: &mut [f64], area_target: f64) -> bool {
    let harmonics: f64 = params[1..].iter().map(|ak| ak * ak).sum();
    let radicand = (area_target / PI) - 0.5 * harmonics;
    if radicand <= 0.0 {
        return false;
    }
    params[0] = radicand.sqrt();
    shape_is_valid(params)
}

/// Minimise drag at fixed area, starting from the circular cylinder.
///
/// Uses projected-gradient descent: a forward finite-difference drag
/// gradient is projected onto the area-constraint tangent, trial shapes
/// are projected back onto the fixed-area manifold in closed form, and
/// a backtracking line search accepts only drag-decreasing shapes.
/// This keeps the evaluation budget explicit — one gradient costs
/// `n_modes` flow solves — unlike a generic constrained optimiser
/// whose inner iterations would each need a full gradient.
///
/// # Errors
///
/// Returns [`DragShapeOptError::InvalidConfig`] for an invalid
/// configuration and [`DragShapeOptError::FlowSolveFailed`] when the
/// initial cylinder cannot be solved.
pub fn optimise_drag(config: &DragShapeOptConfig) -> Result<DragShapeOptResult, DragShapeOptError> {
    config.validate()?;
    let reference = ReferenceMesh::new(config);
    let mut initial_params = vec![0.0; config.n_modes];
    initial_params[0] = config.radius;
    let area_target = shape_area(&initial_params);

    let history = RefCell::new(Vec::new());
    let warm: WarmStart = RefCell::new(None);
    let evals = std::cell::Cell::new(0_usize);
    let initial_flow = evaluate_flow_on(&reference, &initial_params, None)?;
    *warm.borrow_mut() = Some((initial_flow.velocity.clone(), initial_flow.pressure.clone()));
    {
        let mut recorded = history.borrow_mut();
        recorded.push(DragEvaluation {
            evaluation: 0,
            params: initial_params.clone(),
            drag: initial_flow.drag,
            drag_coefficient: initial_flow.drag_coefficient,
        });
    }

    let mut params = initial_params.clone();
    let mut drag = initial_flow.drag;
    let mut converged = false;
    let mut iterations = 0_usize;
    // Initial trial step: a quarter radius in the largest coefficient.
    let initial_step = 0.25 * config.radius;
    for _ in 0..config.max_outer_iterations {
        iterations += 1;
        let gradient = fd_gradient(
            &reference,
            &params,
            drag,
            config.fd_eps,
            &history,
            &warm,
            &evals,
        );
        let area_gradient = shape_area_gradient(&params);
        let area_norm_sq: f64 = area_gradient.iter().map(|g| g * g).sum();
        let projection: f64 = gradient
            .iter()
            .zip(&area_gradient)
            .map(|(g, a)| g * a)
            .sum::<f64>()
            / area_norm_sq;
        let projected: Vec<f64> = gradient
            .iter()
            .zip(&area_gradient)
            .map(|(g, a)| g - projection * a)
            .collect();
        let max_component = projected.iter().map(|g| g.abs()).fold(0.0, f64::max);
        if max_component <= 0.0 {
            converged = true;
            break;
        }
        let direction: Vec<f64> = projected.iter().map(|g| -g / max_component).collect();
        let mut step = initial_step;
        let mut accepted = false;
        for _ in 0..config.max_inner_iterations {
            let mut trial: Vec<f64> = params
                .iter()
                .zip(&direction)
                .map(|(value, d)| value + step * d)
                .collect();
            if project_area(&mut trial, area_target) {
                let trial_drag = evaluate_drag(&reference, &trial, &history, &warm, &evals);
                if trial_drag < drag {
                    let relative_decrease = (drag - trial_drag) / drag;
                    params = trial;
                    drag = trial_drag;
                    accepted = true;
                    if relative_decrease < 1e-3 {
                        converged = true;
                    }
                    break;
                }
            }
            step *= 0.5;
        }
        if !accepted || converged {
            // No decreasing trial (or a negligible one) means the
            // projected gradient has stalled at this resolution.
            converged = true;
            break;
        }
    }

    let cached = warm.borrow().clone();
    let warm_ref = cached
        .as_ref()
        .map(|(velocity, pressure)| (velocity.as_slice(), pressure.as_slice()));
    let optimal_flow = evaluate_flow_on(&reference, &params, warm_ref)?;
    let half_length = shape_radius(0.0, &params);
    let half_width = shape_radius(PI / 2.0, &params);
    Ok(DragShapeOptResult {
        initial_params,
        optimal_params: params.clone(),
        initial_flow,
        optimal_flow,
        area_target,
        optimal_area: shape_area(&params),
        aspect_ratio: half_length / half_width,
        history: history.into_inner(),
        converged,
        optimiser_iterations: iterations,
        fun_evals: evals.get(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coarse_config() -> DragShapeOptConfig {
        DragShapeOptConfig {
            nx: 40,
            ny: 8,
            max_outer_iterations: 8,
            max_inner_iterations: 6,
            ..Default::default()
        }
    }

    #[test]
    fn area_formula_and_gradient_match_finite_differences() {
        let params = vec![0.05, 0.01, -0.005];
        let expected = PI * 0.05 * 0.05 + 0.5 * PI * (0.01 * 0.01 + 0.005 * 0.005);
        assert!((shape_area(&params) - expected).abs() < 1e-15);
        let gradient = shape_area_gradient(&params);
        for i in 0..params.len() {
            let mut plus = params.clone();
            let mut minus = params.clone();
            plus[i] += 1e-7;
            minus[i] -= 1e-7;
            let fd = (shape_area(&plus) - shape_area(&minus)) / 2e-7;
            assert!((gradient[i] - fd).abs() < 1e-6, "coefficient {i}");
        }
    }

    #[test]
    fn area_projection_restores_target_area() {
        let target = shape_area(&[0.05, 0.0, 0.0]);
        let mut params = vec![0.05, 0.02, 0.01];
        assert!(project_area(&mut params, target));
        assert!((shape_area(&params) - target).abs() < 1e-15);
    }

    #[test]
    fn deformation_fixes_far_field_and_keeps_elements_positive() {
        let config = coarse_config();
        let reference = ReferenceMesh::new(&config);
        let params = vec![0.046, 0.02, 0.0];
        let mesh = reference.deform(&params);
        assert!(reference.deformed_is_valid(&mesh));
        for (node, &(x, y)) in reference.reference.iter().enumerate() {
            let distance = ((x - config.center_x).powi(2) + (y - config.center_y).powi(2)).sqrt();
            if distance - config.radius >= 3.0 * config.radius {
                assert_eq!(mesh.nodes[node].x, x);
                assert_eq!(mesh.nodes[node].y, y);
            }
            if (distance - config.radius).abs() < 1e-9 {
                let theta = (y - config.center_y).atan2(x - config.center_x);
                let moved = ((mesh.nodes[node].x - config.center_x).powi(2)
                    + (mesh.nodes[node].y - config.center_y).powi(2))
                .sqrt();
                assert!((moved - shape_radius(theta, &params)).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn cylinder_flow_converges_with_positive_drag() {
        let config = coarse_config();
        let flow = evaluate_flow(&config, &[config.radius, 0.0, 0.0])
            .expect("cylinder flow should converge");
        assert!(flow.drag > 0.0);
        assert!(flow.drag_coefficient > 1.0);
    }

    #[test]
    fn optimisation_reduces_drag_at_fixed_area() {
        let config = coarse_config();
        let result = optimise_drag(&config).expect("optimisation should complete");
        assert!(
            result.optimal_flow.drag < result.initial_flow.drag,
            "optimal drag {} should beat cylinder {}",
            result.optimal_flow.drag,
            result.initial_flow.drag
        );
        assert!(
            (result.optimal_area - result.area_target).abs() < 0.05 * result.area_target,
            "area {} vs target {}",
            result.optimal_area,
            result.area_target
        );
        assert!(result.aspect_ratio > 1.2, "aspect {}", result.aspect_ratio);
    }
}
