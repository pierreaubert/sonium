//! Navier-Stokes extensions to the Stokes solver
//!
//! Adds the nonlinear convection term `(u·∇)u` to the Stokes equations,
//! giving the incompressible Navier-Stokes equations:
//!
//! ```text
//!   -μ Δu + (u·∇)u + ∇p = f    (momentum)
//!                  div(u) = 0    (incompressibility)
//! ```
//!
//! Two solution strategies are provided:
//!
//! 1. **Oseen linearisation** (Picard iteration): Replace `(u·∇)u` with
//!    `(u_k·∇)u_{k+1}` where `u_k` is from the previous iteration. This
//!    gives a linear system at each step. Simple but converges linearly.
//!
//! 2. **Newton linearisation**: Full tangent including `(δu·∇)u + (u·∇)δu`.
//!    Quadratic convergence near the solution but needs a good initial guess.
//!
//! # Reynolds number
//!
//! Re = ρ U L / μ. For Re < ~40 the flow is steady; for higher Re,
//! unsteady and eventually turbulent.

use crate::forms::{BilinearForm, LocalMatrix, LocalVector, MassForm, NonlinearForm};
use crate::quadrature::QuadratureRule;
use crate::spaces::BasisTabulation;

// ---------------------------------------------------------------------------
// Oseen convection form (for Picard iteration)
// ---------------------------------------------------------------------------

/// Oseen (linearised) convection form: `c(u,v; w) = ∫ (w·∇u)·v dΩ`
///
/// The advection velocity `w` is fixed from the previous iteration.
/// This is a standard **bilinear** form in (u, v) for given w.
///
/// Combined with the viscous form, the Oseen system per iteration is:
/// ```text
/// [μK + C(w) + S_u   B ] [u_{k+1}]   [f]
/// [B^T               S_p] [p_{k+1}] = [0]
/// ```
pub struct OseenConvectionForm<F: Fn([f64; 3]) -> Vec<f64> + Send + Sync> {
    /// Density ρ \[kg/m³\].
    pub rho: f64,
    /// Advection velocity w(x,y,z) from previous iteration.
    pub velocity_fn: F,
}

impl<F: Fn([f64; 3]) -> Vec<f64> + Send + Sync> OseenConvectionForm<F> {
    pub fn new(rho: f64, velocity_fn: F) -> Self {
        Self { rho, velocity_fn }
    }
}

impl<F: Fn([f64; 3]) -> Vec<f64> + Send + Sync> BilinearForm<f64> for OseenConvectionForm<F> {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let dim = test.spatial_dim;
        let n = test.n_dofs;
        let w = weight * test.det_j * self.rho;

        // Evaluate advection velocity at quadrature point
        let vel = (self.velocity_fn)(test.physical_coords);

        // c(u,v; w) = ∫ ρ (w·∇u_j)·v_i dΩ
        // For VectorH1, DOF i has component c_i = i % dim.
        // Only same-component pairs contribute: c_i == c_j.
        for i in 0..n {
            let ci = i % dim;
            let phi_i = test.vector_value(i, ci);
            for j in 0..n {
                let cj = j % dim;
                if ci != cj {
                    continue;
                }
                // w · ∇(φ_j e_{cj}) = Σ_d w_d * ∂φ_j/∂x_d
                let w_dot_grad_j: f64 = (0..dim).map(|d| vel[d] * trial.grad(j, d)).sum();
                local_matrix.add(i, j, phi_i * w_dot_grad_j * w);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Newton convection form (for full Newton iteration)
// ---------------------------------------------------------------------------

/// Full Newton convection form implementing [`NonlinearForm`].
///
/// Residual: R(u) = ∫ ρ (u·∇u)·v dΩ
/// Tangent:  J(u)[δu] = ∫ ρ ((δu·∇)u + (u·∇)δu)·v dΩ
///
/// The DOF layout assumes `VectorH1Space` with interleaved components:
/// DOF `i` has component `i % dim` and node index `i / dim`.
pub struct NavierStokesConvectionForm {
    /// Density ρ.
    pub rho: f64,
    /// Spatial dimension (2 or 3).
    pub dim: usize,
}

impl NavierStokesConvectionForm {
    pub fn new(rho: f64, dim: usize) -> Self {
        Self { rho, dim }
    }

    /// Reconstruct the velocity vector at a quadrature point from the local solution.
    #[allow(clippy::needless_range_loop)]
    fn eval_velocity(&self, tab: &BasisTabulation, local_sol: &[f64]) -> Vec<f64> {
        let dim = self.dim;
        let mut u = vec![0.0; dim];
        let n_nodes = tab.n_dofs / dim;
        for node in 0..n_nodes {
            for d in 0..dim {
                let dof = node * dim + d;
                if dof < local_sol.len() {
                    // vector_value(dof, d) gives φ_node for the active component
                    u[d] += tab.vector_value(dof, d) * local_sol[dof];
                }
            }
        }
        u
    }

    /// Reconstruct the velocity gradient ∂u_c/∂x_d at a quadrature point.
    #[allow(clippy::needless_range_loop)]
    fn eval_velocity_gradient(&self, tab: &BasisTabulation, local_sol: &[f64]) -> Vec<Vec<f64>> {
        let dim = self.dim;
        let n_nodes = tab.n_dofs / dim;
        let mut grad_u = vec![vec![0.0; dim]; dim]; // grad_u[c][d] = ∂u_c/∂x_d
        for node in 0..n_nodes {
            for c in 0..dim {
                let dof = node * dim + c;
                if dof < local_sol.len() {
                    for d in 0..dim {
                        // Use dof (not node) to correctly index VectorH1 gradients
                        grad_u[c][d] += tab.grad(dof, d) * local_sol[dof];
                    }
                }
            }
        }
        grad_u
    }
}

impl NonlinearForm<f64> for NavierStokesConvectionForm {
    fn element_residual(
        &self,
        test: &BasisTabulation,
        local_solution: &[f64],
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let dim = self.dim;
        let n = test.n_dofs;
        let w = weight * test.det_j * self.rho;

        let u = self.eval_velocity(test, local_solution);
        let grad_u = self.eval_velocity_gradient(test, local_solution);

        // R_i = ∫ ρ (u·∇u)_ci · v_i dΩ
        // (u·∇u)_c = Σ_d u_d ∂u_c/∂x_d
        for i in 0..n {
            let ci = i % dim;
            let phi_i = test.vector_value(i, ci);
            let convection_ci: f64 = (0..dim).map(|d| u[d] * grad_u[ci][d]).sum();
            local_vector.add(i, convection_ci * phi_i * w);
        }
    }

    fn element_tangent(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        local_solution: &[f64],
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let dim = self.dim;
        let n = test.n_dofs;
        let w = weight * test.det_j * self.rho;

        let u = self.eval_velocity(test, local_solution);
        let grad_u = self.eval_velocity_gradient(test, local_solution);

        // J_ij = ∫ ρ [((δu_j·∇)u)_ci + ((u·∇)δu_j)_ci] · v_i dΩ
        //
        // Term 1: (δu·∇)u → δu_j contributes to component cj.
        //   If cj == d, this gives: φ_j * ∂u_ci/∂x_d. Non-zero only for all ci.
        //
        // Term 2: (u·∇)δu → u·∇(φ_j e_cj).
        //   = Σ_d u_d * ∂φ_j/∂x_d * δ_{ci,cj}
        //   Only ci == cj contributes.

        for i in 0..n {
            let ci = i % dim;
            let phi_i = test.vector_value(i, ci);

            for j in 0..n {
                let cj = j % dim;
                // Use vector_value to correctly index into VectorH1 tabulation
                let phi_j = trial.vector_value(j, cj);

                let mut val = 0.0;

                // Term 1: (δu_j·∇)u — δu_j is in direction cj
                // Contribution to ci: φ_j * ∂u_ci/∂x_cj
                val += phi_j * grad_u[ci][cj];

                // Term 2: (u·∇)δu_j — only if ci == cj
                if ci == cj {
                    let u_dot_grad_phi_j: f64 = (0..dim).map(|d| u[d] * trial.grad(j, d)).sum();
                    val += u_dot_grad_phi_j;
                }

                local_matrix.add(i, j, phi_i * val * w);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Navier-Stokes solver (Picard + Newton)
// ---------------------------------------------------------------------------

/// Configuration for the Navier-Stokes solver.
#[derive(Debug, Clone)]
pub struct NavierStokesConfig {
    /// Dynamic viscosity μ \[Pa·s\].
    pub mu: f64,
    /// Density ρ \[kg/m³\].
    pub rho: f64,
    /// Maximum number of Picard (outer) iterations.
    pub max_picard_iterations: usize,
    /// Minimum Picard iterations before convergence may be accepted.
    pub min_picard_iterations: usize,
    /// Picard convergence tolerance (relative velocity change).
    pub picard_tolerance: f64,
    /// Picard under-relaxation factor for each nonlinear update.
    pub picard_relaxation: f64,
    /// Enable residual-based SUPG/PSPG/LSIC stabilisation.
    ///
    /// Keep enabled for equal-order velocity/pressure pairs such as P1-P1.
    /// Disable for inf-sup-stable Taylor-Hood pairs such as P2-P1, where the
    /// additional pressure stabilisation is unnecessary.
    pub use_residual_stabilisation: bool,
    /// Whether to switch to Newton after Picard converges to rough tolerance.
    pub use_newton_refinement: bool,
    /// Maximum Newton iterations (if enabled).
    pub max_newton_iterations: usize,
    /// Newton convergence tolerance.
    pub newton_tolerance: f64,
    /// Maximum iterations for each Picard linearized solve.
    pub linear_solver_max_iterations: usize,
    /// Relative residual tolerance for each Picard linearized solve.
    pub linear_solver_tolerance: f64,
    /// CPU preconditioner for iterative solves.
    pub linear_solver_cpu_preconditioner: LinearSolverCpuPreconditionerKind,
    /// Use sparse direct solver instead of GMRES+ILU.
    /// Recommended for saddle-point systems (default: true).
    pub use_direct_solver: bool,
    /// Turbulence closure. `Laminar` (default) preserves the legacy
    /// behaviour exactly. `KOmegaSstAlgebraic` adds an algebraic eddy
    /// viscosity to `μ_eff = μ + ρ·ν_t` per Picard iteration.
    pub turbulence: super::turbulence::TurbulenceModel,
    /// Optional k--omega eddy viscosity supplied by a checked segregated
    /// transport update. When present, this lagged value replaces the
    /// algebraic closure for the next momentum assembly.
    pub transported_eddy_viscosity: Option<f64>,
    /// Optional element-local k--omega eddy viscosity supplied by a checked
    /// segregated transport update. When present, this field replaces the
    /// scalar transported value in the viscous momentum block. The values
    /// are lagged for the current segregated step and are not differentiated
    /// by the Newton tangent APIs.
    pub transported_eddy_viscosity_per_element: Option<Vec<f64>>,
    /// Compute backend used for the per-iteration linear solve. `Cpu`
    /// (default) keeps the legacy in-house BLAS-1 + ILU path. `Cuda` and `Metal`
    /// route the linear system through `sonium_compute::Backend` —
    /// upload CSR + RHS to device, run `gmres_backend_preconditioned`
    /// with the polynomial Jacobi preconditioner, download the solution.
    /// CPU assembly is unchanged; only the linear-solve hot loop migrates.
    pub linear_solver_backend: LinearSolverBackend,
    /// Which GPU-friendly preconditioner the backend GMRES uses.
    /// Default is [`BackendPreconditionerKind::PolynomialJacobi`] with
    /// the existing `linear_solver_polynomial_*` knobs (preserves prior
    /// behaviour). [`BackendPreconditionerKind::Chebyshev`] is the
    /// stronger choice on SPD or near-SPD systems and uses the
    /// `linear_solver_chebyshev_*` knobs. Ignored on the CPU backend
    /// (which uses ILU).
    pub linear_solver_backend_preconditioner: BackendPreconditionerKind,
    /// Polynomial Jacobi preconditioner degree for the GPU/backend path.
    /// `1` (default) is plain point-Jacobi — `K` SpMVs per apply gets
    /// you a stronger preconditioner at the cost of `K-1` extra SpMVs
    /// per GMRES iteration. Typical values: 2–4. Used when
    /// `linear_solver_backend_preconditioner == PolynomialJacobi`.
    pub linear_solver_polynomial_degree: usize,
    /// Damping factor `ω` for the polynomial Jacobi preconditioner.
    /// `1.0` (default) is plain Jacobi-Richardson. `2/3` is the classical
    /// weighted-Jacobi smoother for diffusive SPD systems. For indefinite
    /// Stokes blocks, `0.5..0.8` is the practical range.
    pub linear_solver_polynomial_damping: f64,
    /// Polynomial degree for the pressure Schur solve inside backend
    /// Jacobi/Schur. Default `1` preserves diagonal Schur behavior.
    pub linear_solver_schur_polynomial_degree: usize,
    /// Damping for Schur polynomial correction steps. The first Schur
    /// step is always the undamped diagonal inverse; this only affects
    /// extra correction steps when degree > 1.
    pub linear_solver_schur_polynomial_damping: f64,
    /// Pressure-block solve used inside backend Jacobi/Schur.
    pub linear_solver_schur_preconditioner: BackendPressureSchurPreconditionerKind,
    /// Chebyshev degree for the pressure Schur solve inside backend
    /// Jacobi/Schur. Used when `linear_solver_schur_preconditioner`
    /// is `Chebyshev`.
    pub linear_solver_schur_chebyshev_degree: usize,
    /// Pressure Schur Chebyshev eigenvalue interval ratio.
    pub linear_solver_schur_chebyshev_eigenvalue_ratio: f64,
    /// Chebyshev preconditioner degree (number of 3-term-recurrence
    /// steps). `5` is the PETSc/HYPRE default. Used when
    /// `linear_solver_backend_preconditioner == Chebyshev`.
    pub linear_solver_chebyshev_degree: usize,
    /// Chebyshev eigenvalue-interval ratio `λ_max / λ_min`. `30.0`
    /// (default) matches PETSc/HYPRE. Smaller values target a tighter
    /// interval around the upper end of the spectrum (good as a
    /// multigrid smoother); larger values cover the whole spectrum at
    /// the cost of weaker damping per step.
    pub linear_solver_chebyshev_eigenvalue_ratio: f64,
    /// Working precision for the iterative linear solve.
    /// [`LinearSolverKind::F64`] (default) runs the entire Krylov in
    /// f64 — the legacy path. [`LinearSolverKind::MixedF32F64`] routes
    /// the inner Krylov through f32 (FGMRES-IR, Carson-Higham 2018);
    /// outer FGMRES still f64. Required to get any acceleration on
    /// Apple Silicon Metal, whose GPU compiler rejects `double`.
    pub linear_solver_kind: LinearSolverKind,
    /// Maximum **inner** GMRES iterations per outer FGMRES-IR step
    /// (only used when `linear_solver_kind == MixedF32F64`). Default 50.
    pub linear_solver_ir_inner_max_iterations: usize,
    /// Inner f32 GMRES relative-residual tolerance. Loose by design
    /// — the outer FGMRES refines whatever the inner gives. Default `1e-3`.
    pub linear_solver_ir_inner_tolerance: f32,
    /// Cache the BDF body-force RHS/Jacobian assemblies in the workspace
    /// instead of reassembling them every step. The closures only see
    /// spatial coordinates, so time-independent forces (including the
    /// zero force used when no body force is present) assemble
    /// identically each step; caching skips two full velocity-space
    /// assemblies per step after the first. Default `false` preserves
    /// the legacy reassemble-every-step behavior. Each reuse
    /// re-evaluates the closures at a few sample DOFs and reassembles
    /// on any mismatch, so a varying force only wastes the cache.
    pub cache_body_force_assembly: bool,
    /// Maximum consecutive BDF solves reusing a stale Schur-complement
    /// preconditioner setup. The Oseen matrix drifts slowly between
    /// steps, so rebuilding the BtDiagB product plus ILUT factors every
    /// step wastes setup work; lagging amortizes it. Default `0`
    /// preserves the legacy rebuild-on-any-change behavior. Each lagged
    /// reuse additionally requires the matrix change to sit below
    /// [`Self::linear_solver_setup_lag_tolerance`], and a lagged solve
    /// that fails to converge rebuilds fresh and retries once, so a
    /// stale setup can only waste one solve, never fail a step.
    pub linear_solver_setup_max_lag_steps: usize,
    /// Maximum relative matrix change (`||ΔA||_F / ||A||_F`) for one
    /// lagged preconditioner reuse. Only read when
    /// [`Self::linear_solver_setup_max_lag_steps`] is nonzero. Default
    /// `0.1`.
    pub linear_solver_setup_lag_tolerance: f64,
}

// Selection policy for the per-Picard-iteration linear solve lives in
// `crate::backend_selection`; this re-export keeps the historic
// `navier_stokes::` paths working. (r2factor split output: keep this shim
// layout if the facade is re-split.)
pub use crate::backend_selection::{
    BackendPreconditionerKind, BackendPressureSchurPreconditionerKind, LinearSolverBackend,
    LinearSolverCpuPreconditionerKind, LinearSolverKind,
};

impl Default for NavierStokesConfig {
    fn default() -> Self {
        Self {
            mu: 1.0,
            rho: 1.0,
            max_picard_iterations: 50,
            min_picard_iterations: 0,
            picard_tolerance: 1e-6,
            picard_relaxation: 1.0,
            use_residual_stabilisation: true,
            use_newton_refinement: false,
            max_newton_iterations: 10,
            newton_tolerance: 1e-10,
            linear_solver_max_iterations: 5000,
            linear_solver_tolerance: 1e-8,
            linear_solver_cpu_preconditioner: LinearSolverCpuPreconditionerKind::Ilu,
            use_direct_solver: true,
            turbulence: super::turbulence::TurbulenceModel::Laminar,
            transported_eddy_viscosity: None,
            transported_eddy_viscosity_per_element: None,
            linear_solver_backend: LinearSolverBackend::Auto,
            linear_solver_backend_preconditioner: BackendPreconditionerKind::PolynomialJacobi,
            linear_solver_polynomial_degree: 1,
            linear_solver_polynomial_damping: 1.0,
            linear_solver_schur_polynomial_degree: 1,
            linear_solver_schur_polynomial_damping: 1.0,
            linear_solver_schur_preconditioner: BackendPressureSchurPreconditionerKind::Polynomial,
            linear_solver_schur_chebyshev_degree: 5,
            linear_solver_schur_chebyshev_eigenvalue_ratio: 30.0,
            linear_solver_chebyshev_degree: 5,
            linear_solver_chebyshev_eigenvalue_ratio: 30.0,
            linear_solver_kind: LinearSolverKind::F64,
            linear_solver_ir_inner_max_iterations: 50,
            linear_solver_ir_inner_tolerance: 1.0e-3,
            cache_body_force_assembly: false,
            linear_solver_setup_max_lag_steps: 0,
            linear_solver_setup_lag_tolerance: 0.1,
        }
    }
}

impl NavierStokesConfig {
    /// Return the dynamic viscosity used by a lagged momentum assembly and
    /// its derivative with respect to the velocity-scale closure argument.
    pub fn effective_dynamic_viscosity_and_speed_derivative(
        &self,
        velocity_magnitude: f64,
    ) -> (f64, f64) {
        if let Some(eddy_viscosity) = self.transported_eddy_viscosity
            && eddy_viscosity.is_finite()
            && eddy_viscosity >= 0.0
        {
            return (self.mu + self.rho * eddy_viscosity, 0.0);
        }
        self.turbulence
            .effective_mu_and_speed_derivative(self.mu, self.rho, velocity_magnitude)
    }

    /// Config for a given Reynolds number: Re = ρ U L / μ.
    ///
    /// Sets μ = ρ U L / Re with ρ = 1, U = 1, L = 1 (nondimensional).
    pub fn from_reynolds(re: f64) -> Self {
        Self {
            mu: 1.0 / re,
            rho: 1.0,
            ..Default::default()
        }
    }
}

/// Assemble the element-local viscous block for a transported eddy-viscosity
/// field, validating the field before parallel FEM assembly starts.
///
/// `None` means that the configuration requests the legacy uniform viscous
/// block. A per-element transported field is intentionally treated as a
/// lagged coefficient: the exact coupled scalar/momentum Newton derivative
/// is not part of this API contract.
fn assemble_transported_viscous_triplet<VSpace: crate::spaces::FunctionSpace>(
    mesh: &crate::mesh::Mesh,
    v_space: &VSpace,
    config: &NavierStokesConfig,
    quad_order: usize,
) -> Result<Option<crate::assembly::TripletMatrix<f64>>, &'static str> {
    let Some(eddy_viscosities) = config.transported_eddy_viscosity_per_element.as_ref() else {
        return Ok(None);
    };

    if eddy_viscosities.len() != mesh.num_elements() {
        return Err("transported element eddy-viscosity field has the wrong length");
    }
    if !config.mu.is_finite() || !config.rho.is_finite() {
        return Err("molecular viscosity or density is not finite");
    }

    let forms = eddy_viscosities
        .iter()
        .map(|&eddy_viscosity| {
            if !eddy_viscosity.is_finite() || eddy_viscosity < 0.0 {
                return Err("transported element eddy viscosity must be finite and nonnegative");
            }
            let dynamic_viscosity = config.mu + config.rho * eddy_viscosity;
            if !dynamic_viscosity.is_finite() || dynamic_viscosity < 0.0 {
                return Err("transported element dynamic viscosity is invalid");
            }
            Ok(super::forms::StokesViscousForm::new(dynamic_viscosity))
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Some(crate::assembly::assemble_bilinear_per_element(
        &forms, v_space, v_space, mesh, quad_order,
    )))
}

#[cfg(test)]
mod transported_viscosity_tests {
    use super::*;

    #[test]
    fn transported_eddy_viscosity_overrides_algebraic_closure_for_one_step() {
        let mut config = NavierStokesConfig::default();
        config.mu = 1.8e-5;
        config.rho = 1.2;
        config.turbulence =
            crate::physics::fluid::stokes::turbulence::TurbulenceModel::pcfan_default();
        config.transported_eddy_viscosity = Some(2.5e-3);

        let (effective, derivative) = config.effective_dynamic_viscosity_and_speed_derivative(10.0);

        assert!((effective - (config.mu + config.rho * 2.5e-3)).abs() < 1.0e-15);
        assert_eq!(derivative, 0.0);
    }

    #[test]
    fn transported_element_viscosity_assembles_heterogeneous_momentum_block() {
        let mesh = crate::mesh::unit_cube_tetrahedra(1);
        let v_space = crate::spaces::VectorH1Space::new(crate::basis::PolynomialDegree::P1, 3);
        let mut config = NavierStokesConfig::default();
        config.mu = 1.0;
        config.rho = 1.0;
        let element_count = mesh.num_elements();
        let mut eddy_viscosities = vec![0.0; element_count];
        eddy_viscosities[0] = 1.0;
        config.transported_eddy_viscosity_per_element = Some(eddy_viscosities);

        let heterogeneous = assemble_transported_viscous_triplet(&mesh, &v_space, &config, 2)
            .expect("valid transported field")
            .expect("element field should select element assembly");
        let mean_dynamic_viscosity = 1.0 + 1.0 / element_count as f64;
        let mean_form = super::super::forms::StokesViscousForm::new(mean_dynamic_viscosity);
        let uniform = crate::assembly::assemble_bilinear(&mean_form, &v_space, &v_space, &mesh, 2);

        assert_eq!(heterogeneous.rows, uniform.rows);
        assert_eq!(heterogeneous.cols, uniform.cols);
        assert!(
            heterogeneous
                .values
                .iter()
                .zip(&uniform.values)
                .any(|(left, right)| (left - right).abs() > 1.0e-12)
        );
    }

    #[test]
    fn transported_element_viscosity_validates_shape_and_values() {
        let mesh = crate::mesh::unit_cube_tetrahedra(1);
        let v_space = crate::spaces::VectorH1Space::new(crate::basis::PolynomialDegree::P1, 3);
        let mut config = NavierStokesConfig::default();
        config.transported_eddy_viscosity_per_element = Some(vec![0.0]);
        assert!(assemble_transported_viscous_triplet(&mesh, &v_space, &config, 2).is_err());

        config.transported_eddy_viscosity_per_element = Some(vec![-1.0; mesh.num_elements()]);
        assert!(assemble_transported_viscous_triplet(&mesh, &v_space, &config, 2).is_err());
    }
}

// ---------------------------------------------------------------------------
// Picard iteration solver
// ---------------------------------------------------------------------------

/// Result of a Navier-Stokes solve.
#[derive(Debug)]
pub struct NavierStokesResult {
    /// Velocity DOF values (interleaved: [u0,v0, u1,v1, ...]).
    pub velocity: Vec<f64>,
    /// Pressure DOF values.
    pub pressure: Vec<f64>,
    /// Number of Picard iterations performed.
    pub iterations: usize,
    /// Final relative velocity change.
    pub residual: f64,
    /// Whether the iteration converged.
    pub converged: bool,
    /// Concrete backend used by the last linear solve.
    ///
    /// This is distinct from the requested backend in [`NavierStokesConfig`]:
    /// unavailable CUDA/Metal backends are explicitly reported as `Cpu` after
    /// the checked dispatcher fallback.
    pub resolved_linear_solver_backend: LinearSolverBackend,
    /// The last checked Oseen system and fixed-point update.
    ///
    /// This is intentionally labelled as a Picard/Oseen linearization. It is
    /// useful for diagnostics and continuation, but it is not the exact full
    /// Navier--Stokes Newton tangent.
    pub picard_linearization: Option<NavierStokesPicardLinearization>,
    /// The complete laminar Newton tangent and residual of the assembled
    /// Navier--Stokes map at the last Picard input, including active
    /// SUPG/PSPG/LSIC and velocity-dependent body-force terms.  Algebraic
    /// turbulence closures do not publish this field because their global
    /// peak-velocity derivative is not part of this contract.
    pub newton_linearization: Option<NavierStokesNewtonLinearization>,
}

/// Sparse linearization published by the Picard solver at its last update.
#[derive(Debug, Clone)]
pub struct NavierStokesPicardLinearization {
    /// Boundary-conditioned Oseen matrix used for the last linear solve.
    pub state_jacobian: sonium_solvers::CsrMatrix<f64>,
    /// Concatenated state `[velocity, pressure]` at which the published
    /// one-step Picard map was evaluated.
    pub state: Vec<f64>,
    /// Fixed-point update `state_new - state_old`.
    pub fixed_point_residual: Vec<f64>,
}

/// Boundary-conditioned Newton tangent and residual at one Navier--Stokes
/// map input state.
///
/// The velocity convection block is the analytic derivative of
/// `rho * (u · grad) u`, including both Newton terms.  The surrounding
/// viscous, incompressibility, stabilisation, and body-force blocks are
/// differentiated at the same state for the laminar contract.
#[derive(Debug, Clone)]
pub struct NavierStokesNewtonLinearization {
    /// Boundary-conditioned full `[velocity, pressure]` Newton tangent.
    pub state_jacobian: sonium_solvers::CsrMatrix<f64>,
    /// Boundary-conditioned nonlinear residual at `state`.
    pub state_residual: Vec<f64>,
    /// Concatenated map-input state `[velocity, pressure]`.
    pub state: Vec<f64>,
}

/// Error returned when a body-force design derivative cannot be assembled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyForceDesignDerivativeError {
    /// The callback did not return one row per force component or one column
    /// per declared design variable.
    InvalidShape,
    /// A callback returned a non-finite derivative.
    NonFiniteValue,
}

/// Failure while assembling the analytic residual derivative induced by
/// nodal mesh motion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshMotionDesignDerivativeError {
    /// The contraction currently targets production 3-D affine tetrahedra with
    /// P1 pressure and P1/P2 velocity. Higher-order and non-tetrahedral spaces
    /// need their own geometry maps before they can be admitted.
    UnsupportedDiscretisation,
    /// A mesh, state, or nodal-motion array has an inconsistent dimension.
    DimensionMismatch(&'static str),
    /// A mesh or supplied state contains a non-finite value.
    NonFiniteInput(&'static str),
    /// An element has a singular or otherwise invalid affine map.
    InvalidElement(usize),
    /// The coordinate derivative of the body force was not finite.
    NonFiniteBodyForceDerivative,
}

impl std::fmt::Display for MeshMotionDesignDerivativeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedDiscretisation => write!(
                formatter,
                "mesh-motion residual derivative requires 3-D affine tetrahedra with P1 pressure and P1/P2 velocity"
            ),
            Self::DimensionMismatch(field) => {
                write!(
                    formatter,
                    "mesh-motion residual derivative dimension mismatch: {field}"
                )
            }
            Self::NonFiniteInput(field) => {
                write!(
                    formatter,
                    "mesh-motion residual derivative has non-finite {field}"
                )
            }
            Self::InvalidElement(element) => {
                write!(
                    formatter,
                    "mesh-motion residual derivative invalid element {element}"
                )
            }
            Self::NonFiniteBodyForceDerivative => {
                write!(
                    formatter,
                    "mesh-motion residual derivative has non-finite body-force derivative"
                )
            }
        }
    }
}

impl std::error::Error for MeshMotionDesignDerivativeError {}

impl std::fmt::Display for BodyForceDesignDerivativeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidShape => write!(formatter, "invalid body-force design Jacobian shape"),
            Self::NonFiniteValue => {
                write!(
                    formatter,
                    "body-force design Jacobian contains a non-finite value"
                )
            }
        }
    }
}

impl std::error::Error for BodyForceDesignDerivativeError {}

/// Solve the steady incompressible Navier-Stokes equations using
/// Picard (Oseen) iteration.
///
/// At each iteration k:
/// 1. Assemble the stabilised Oseen system:
///    `[μK + C(u_k) + S_u, B; B^T, S_p]`
/// 2. Apply boundary conditions
/// 3. Solve the linear system for `(u_{k+1}, p_{k+1})`
/// 4. Check convergence: `‖u_{k+1} - u_k‖ / ‖u_{k+1}‖ < tol`
///
/// # Arguments
/// - `mesh`: the computational mesh
/// - `v_space`: velocity function space (VectorH1)
/// - `p_space`: pressure function space (H1 or DG)
/// - `config`: solver parameters (μ, ρ, tolerances)
/// - `bc_velocity_fn`: returns the prescribed velocity at boundary DOFs.
///   Called as `bc_velocity_fn(x, y, z) -> velocity components`.
/// - `bc_dofs`: indices of velocity DOFs with Dirichlet BCs
/// - `body_force_fn`: body force f(x,y,z) -> force components
pub fn solve_picard<VSpace, PSpace, BcFn, ForceFn>(
    mesh: &crate::mesh::Mesh,
    v_space: &VSpace,
    p_space: &PSpace,
    config: &NavierStokesConfig,
    bc_velocity_fn: &BcFn,
    bc_dofs: &[usize],
    body_force_fn: ForceFn,
) -> NavierStokesResult
where
    VSpace: crate::spaces::FunctionSpace,
    PSpace: crate::spaces::FunctionSpace,
    BcFn: Fn(f64, f64, f64) -> Vec<f64>,
    ForceFn: Fn([f64; 3]) -> Vec<f64> + Send + Sync + 'static,
{
    solve_picard_with_velocity_force(
        mesh,
        v_space,
        p_space,
        config,
        bc_velocity_fn,
        bc_dofs,
        move |coords, _velocity| body_force_fn(coords),
    )
}

/// Solve the steady incompressible Navier-Stokes equations using Picard
/// iteration, with a body force evaluated against the previous velocity
/// iterate at the same physical point.
pub fn solve_picard_with_velocity_force<VSpace, PSpace, BcFn, ForceFn>(
    mesh: &crate::mesh::Mesh,
    v_space: &VSpace,
    p_space: &PSpace,
    config: &NavierStokesConfig,
    bc_velocity_fn: &BcFn,
    bc_dofs: &[usize],
    body_force_fn: ForceFn,
) -> NavierStokesResult
where
    VSpace: crate::spaces::FunctionSpace,
    PSpace: crate::spaces::FunctionSpace,
    BcFn: Fn(f64, f64, f64) -> Vec<f64>,
    ForceFn: Fn([f64; 3], [f64; 3]) -> Vec<f64> + Send + Sync + 'static,
{
    solve_picard_with_velocity_force_from_initial(
        mesh,
        v_space,
        p_space,
        config,
        bc_velocity_fn,
        bc_dofs,
        body_force_fn,
        None,
        None,
    )
}

/// Picard solve with an optional velocity/pressure initial state.
///
/// Supplying the initial state is useful for fixed-point residual and
/// Jacobian probes: a one-iteration solve evaluates the production discrete
/// Picard map at a prescribed state without running a second converged CFD
/// solve.
pub fn solve_picard_with_velocity_force_from_initial<VSpace, PSpace, BcFn, ForceFn>(
    mesh: &crate::mesh::Mesh,
    v_space: &VSpace,
    p_space: &PSpace,
    config: &NavierStokesConfig,
    bc_velocity_fn: &BcFn,
    bc_dofs: &[usize],
    body_force_fn: ForceFn,
    initial_velocity: Option<&[f64]>,
    initial_pressure: Option<&[f64]>,
) -> NavierStokesResult
where
    VSpace: crate::spaces::FunctionSpace,
    PSpace: crate::spaces::FunctionSpace,
    BcFn: Fn(f64, f64, f64) -> Vec<f64>,
    ForceFn: Fn([f64; 3], [f64; 3]) -> Vec<f64> + Send + Sync + 'static,
{
    solve_picard_with_velocity_force_and_jacobian_from_initial(
        mesh,
        v_space,
        p_space,
        config,
        bc_velocity_fn,
        bc_dofs,
        body_force_fn,
        |_, _| [[0.0; 3]; 3],
        initial_velocity,
        initial_pressure,
    )
}

/// Picard solve with an optional velocity/pressure initial state and an
/// analytic velocity Jacobian for the body force.
///
/// The Jacobian callback returns `∂f_c/∂u_d` at the same point and local
/// velocity passed to `body_force_fn`.  Supplying it is required for the
/// complete laminar Newton observation; the historical wrapper above keeps
/// the zero-Jacobian behaviour for callers whose force is state independent.
#[expect(
    clippy::too_many_arguments,
    reason = "solver entry point threading spaces, force callbacks and Newton options; grouping would churn the public solver API"
)]
pub fn solve_picard_with_velocity_force_and_jacobian_from_initial<
    VSpace,
    PSpace,
    BcFn,
    ForceFn,
    ForceJacFn,
>(
    mesh: &crate::mesh::Mesh,
    v_space: &VSpace,
    p_space: &PSpace,
    config: &NavierStokesConfig,
    bc_velocity_fn: &BcFn,
    bc_dofs: &[usize],
    body_force_fn: ForceFn,
    body_force_jacobian_fn: ForceJacFn,
    initial_velocity: Option<&[f64]>,
    initial_pressure: Option<&[f64]>,
) -> NavierStokesResult
where
    VSpace: crate::spaces::FunctionSpace,
    PSpace: crate::spaces::FunctionSpace,
    BcFn: Fn(f64, f64, f64) -> Vec<f64>,
    ForceFn: Fn([f64; 3], [f64; 3]) -> Vec<f64> + Send + Sync + 'static,
    ForceJacFn: Fn([f64; 3], [f64; 3]) -> [[f64; 3]; 3] + Send + Sync + 'static,
{
    solve_picard_with_velocity_force_and_jacobian_with_policy(
        mesh,
        v_space,
        p_space,
        config,
        bc_velocity_fn,
        bc_dofs,
        body_force_fn,
        body_force_jacobian_fn,
        initial_velocity,
        initial_pressure,
        false,
    )
}

/// Assemble the constrained steady Newton system without solving a linear system.
///
/// Uses the same state mapping, quadrature, stabilization, force derivatives,
/// and pressure gauge as the Picard solver's Newton observation. Algebraic
/// turbulence requires a unique nonzero velocity peak. No solver backend is
/// dispatched by this function.
///
/// # Errors
///
/// Returns `None` for incorrect state dimensions, non-finite state entries,
/// out-of-range velocity boundary DOFs, or an unavailable turbulence tangent.
///
/// # Panics
///
/// Panics for non-nodal velocity spaces or boundary callbacks returning too
/// few velocity components, matching the solver's function-space contract.
#[allow(clippy::too_many_arguments)]
pub fn assemble_steady_newton_linearization<VSpace, PSpace, BcFn, ForceFn, ForceJacFn>(
    mesh: &crate::mesh::Mesh,
    v_space: &VSpace,
    p_space: &PSpace,
    config: &NavierStokesConfig,
    bc_velocity_fn: &BcFn,
    bc_dofs: &[usize],
    body_force_fn: ForceFn,
    body_force_jacobian_fn: ForceJacFn,
    initial_velocity: &[f64],
    initial_pressure: &[f64],
) -> Option<NavierStokesNewtonLinearization>
where
    VSpace: crate::spaces::FunctionSpace,
    PSpace: crate::spaces::FunctionSpace,
    BcFn: Fn(f64, f64, f64) -> Vec<f64>,
    ForceFn: Fn([f64; 3], [f64; 3]) -> Vec<f64> + Send + Sync + 'static,
    ForceJacFn: Fn([f64; 3], [f64; 3]) -> [[f64; 3]; 3] + Send + Sync + 'static,
{
    let n_v = v_space.num_global_dofs(mesh);
    if initial_velocity.len() != n_v
        || initial_pressure.len() != p_space.num_global_dofs(mesh)
        || initial_velocity
            .iter()
            .chain(initial_pressure)
            .any(|value| !value.is_finite())
        || bc_dofs.iter().any(|&dof| dof >= n_v)
    {
        return None;
    }
    let mut assembly_config = config.clone();
    assembly_config.max_picard_iterations = 1;
    solve_picard_with_velocity_force_and_jacobian_with_policy(
        mesh,
        v_space,
        p_space,
        &assembly_config,
        bc_velocity_fn,
        bc_dofs,
        body_force_fn,
        body_force_jacobian_fn,
        Some(initial_velocity),
        Some(initial_pressure),
        true,
    )
    .newton_linearization
}

#[allow(clippy::too_many_arguments)]
fn solve_picard_with_velocity_force_and_jacobian_with_policy<
    VSpace,
    PSpace,
    BcFn,
    ForceFn,
    ForceJacFn,
>(
    mesh: &crate::mesh::Mesh,
    v_space: &VSpace,
    p_space: &PSpace,
    config: &NavierStokesConfig,
    bc_velocity_fn: &BcFn,
    bc_dofs: &[usize],
    body_force_fn: ForceFn,
    body_force_jacobian_fn: ForceJacFn,
    initial_velocity: Option<&[f64]>,
    initial_pressure: Option<&[f64]>,
    assembly_only: bool,
) -> NavierStokesResult
where
    VSpace: crate::spaces::FunctionSpace,
    PSpace: crate::spaces::FunctionSpace,
    BcFn: Fn(f64, f64, f64) -> Vec<f64>,
    ForceFn: Fn([f64; 3], [f64; 3]) -> Vec<f64> + Send + Sync + 'static,
    ForceJacFn: Fn([f64; 3], [f64; 3]) -> [[f64; 3]; 3] + Send + Sync + 'static,
{
    let n_v = v_space.num_global_dofs(mesh);
    let n_p = p_space.num_global_dofs(mesh);
    let n_total = n_v + n_p;
    let dim = mesh.dimension;
    let picard_relaxation = config.picard_relaxation.clamp(1.0e-3, 1.0);
    let velocity_dof_coordinates = v_space
        .dof_coordinates(mesh)
        .expect("Navier-Stokes requires a nodal velocity space");
    assert_eq!(
        velocity_dof_coordinates.len(),
        n_v,
        "velocity space returned the wrong number of DOF coordinates"
    );
    // Constant pressure is a gauge only when its weak traction vanishes on
    // every free velocity test. Natural open boundaries fix its level.
    let divergence_system = crate::assembly::mixed::assemble_mixed(
        &super::forms::StokesViscousForm::new(0.0),
        &super::forms::StokesDivergenceForm,
        v_space,
        p_space,
        mesh,
        2,
    );
    let pin_pressure = pressure_constant_mode_is_free(&divergence_system.block_01, bc_dofs);
    let mesh_centroid = mesh.nodes.iter().fold([0.0_f64; 3], |mut sum, point| {
        sum[0] += point.x;
        sum[1] += point.y;
        sum[2] += point.z;
        sum
    });
    let mesh_centroid = mesh_centroid.map(|sum| sum / mesh.num_nodes().max(1) as f64);
    let pressure_pin = p_space
        .dof_coordinates(mesh)
        .and_then(|coordinates| {
            coordinates
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let distance_a: f64 = (0..dim)
                        .map(|direction| (a[direction] - mesh_centroid[direction]).powi(2))
                        .sum();
                    let distance_b: f64 = (0..dim)
                        .map(|direction| (b[direction] - mesh_centroid[direction]).powi(2))
                        .sum();
                    distance_a.total_cmp(&distance_b)
                })
                .map(|(dof, _)| dof)
        })
        .unwrap_or(0);

    let mut velocity = match initial_velocity {
        Some(values) => {
            assert_eq!(
                values.len(),
                n_v,
                "initial velocity has wrong number of Navier-Stokes DOFs"
            );
            values.to_vec()
        }
        None => vec![0.0; n_v],
    };
    let mut pressure = match initial_pressure {
        Some(values) => {
            assert_eq!(
                values.len(),
                n_p,
                "initial pressure has wrong number of Navier-Stokes DOFs"
            );
            values.to_vec()
        }
        None => vec![0.0; n_p],
    };
    if n_p > 0 && pin_pressure {
        // Keep the prescribed-state map on the same gauge slice as the
        // boundary-conditioned tangent.  Without this, perturbing the pinned
        // pressure DOF changes free residual rows even though its tangent
        // column is deliberately eliminated.
        pressure[pressure_pin] = 0.0;
    }

    // Set initial BC values
    for &dof in bc_dofs {
        assert!(dof < n_v, "velocity Dirichlet DOF {dof} is out of range");
        let point = velocity_dof_coordinates[dof];
        let comp = dof % v_space.num_components();
        let bc_val = bc_velocity_fn(point[0], point[1], point[2]);
        assert!(
            comp < bc_val.len(),
            "velocity boundary function returned too few components"
        );
        velocity[dof] = bc_val[comp];
    }

    // Viscous form is rebuilt each Picard iteration when the turbulence
    // model is non-laminar — it then depends on the current velocity
    // iterate via `μ_eff = μ + ρ·ν_t`. For laminar flow it stays at the
    // configured `config.mu`.
    let body_force_fn = std::sync::Arc::new(body_force_fn);
    let body_force_jacobian_fn = std::sync::Arc::new(body_force_jacobian_fn);
    let stabilisation_rho = config.rho.max(f64::EPSILON);

    let mut converged = false;
    let mut final_residual = 1.0;
    let mut last_linearization = None;
    let mut last_newton_linearization = None;
    let mut resolved_linear_solver_backend = LinearSolverBackend::Cpu;
    // The row claims an accelerator only when every linear solve ran on
    // it: any fallback taints the report permanently (last-wins would
    // let one accelerated solve whitewash many CPU fallbacks).
    let requested_backend = match config.linear_solver_backend {
        LinearSolverBackend::Auto => LinearSolverBackend::resolve_auto(),
        other => other,
    };
    let mut backend_fell_back = false;

    // The algebraic eddy-viscosity model uses a global peak velocity. Its exact

    // tangent is published only when the active peak is unique and non-zero;

    // ties and the zero state remain deliberately fail-closed.
    let publish_newton_linearization = config.use_newton_refinement || initial_velocity.is_some();
    // Per-iteration timing spans for performance work. Timers always run
    // (nanosecond cost); only the report line is gated.
    let profile_ns = std::env::var("SONIUM_NS_PROFILE").as_deref() == Ok("1");
    // BVH velocity topology, built once and shared: every Picard
    // iteration re-queries it against the current iterate instead of
    // rebuilding (and linear-scanning) a per-iteration interpolator.
    // Arc-shared because the assembly closures require 'static data.
    let velocity_topology = std::sync::Arc::new(super::workspace::ElementVelocityTopology::new(
        mesh, v_space,
    ));

    for iter in 0..config.max_picard_iterations {
        let iter_start = std::time::Instant::now();

        // Peak velocity magnitude across all FE nodes — feeds the
        // algebraic turbulence closure. Vertex DOFs only, matching the
        // historical node-velocity max exactly.
        let velocity_magnitude_max = (0..mesh.num_nodes())
            .map(|ni| {
                let mut sum = 0.0;
                for d in 0..dim {
                    let dof = ni * dim + d;
                    if dof < velocity.len() {
                        sum += velocity[dof] * velocity[dof];
                    }
                }
                sum.sqrt()
            })
            .fold(0.0_f64, f64::max);
        let mu_eff = config
            .effective_dynamic_viscosity_and_speed_derivative(velocity_magnitude_max)
            .0;
        let turbulence_mu_gradient = if matches!(
            config.turbulence,
            super::turbulence::TurbulenceModel::Laminar
        ) {
            None
        } else {
            super::turbulence::TurbulenceModel::unique_peak_velocity_subgradient(&velocity, dim)
                .and_then(|(peak, speed_gradient)| {
                    let (_, d_mu_d_speed) =
                        config.effective_dynamic_viscosity_and_speed_derivative(peak);
                    d_mu_d_speed.is_finite().then(|| {
                        speed_gradient
                            .into_iter()
                            .map(|value| value * d_mu_d_speed)
                            .collect::<Vec<_>>()
                    })
                })
        };
        let publish_this_newton = publish_newton_linearization
            && (matches!(
                config.turbulence,
                super::turbulence::TurbulenceModel::Laminar
            ) || turbulence_mu_gradient.is_some());
        let kinematic_viscosity = mu_eff / stabilisation_rho;
        let viscous = super::forms::StokesViscousForm::new(mu_eff);

        // Own a snapshot of the iterate: the assembly closures are
        // 'static, so the evaluator cannot borrow `velocity`.
        let velocity_field = std::sync::Arc::new(PicardVelocityField {
            topology: std::sync::Arc::clone(&velocity_topology),
            values: velocity.clone(),
            dim,
        });
        let convection_velocity_field = std::sync::Arc::clone(&velocity_field);
        let convection = OseenConvectionForm::new(config.rho, move |coords: [f64; 3]| {
            convection_velocity_field.velocity_components_at(coords)
        });
        let supg_velocity_field = std::sync::Arc::clone(&velocity_field);
        let supg = super::stabilisation::SupgConvectionForm {
            rho: config.rho,
            nu: kinematic_viscosity,
            velocity_fn: move |coords: [f64; 3]| supg_velocity_field.velocity_components_at(coords),
            dt: None,
        };
        let lsic_velocity_field = std::sync::Arc::clone(&velocity_field);
        let lsic = super::stabilisation::LsicForm {
            rho: config.rho,
            nu: kinematic_viscosity,
            velocity_magnitude_fn: move |coords: [f64; 3]| {
                lsic_velocity_field.velocity_magnitude_at(coords)
            },
            dt: None,
        };
        let pspg_velocity_field = std::sync::Arc::clone(&velocity_field);
        let pspg = super::stabilisation::PspgForm {
            rho: stabilisation_rho,
            nu: kinematic_viscosity,
            velocity_magnitude_fn: move |coords: [f64; 3]| {
                pspg_velocity_field.velocity_magnitude_at(coords)
            },
            dt: None,
        };
        let supg_pressure_velocity_field = std::sync::Arc::clone(&velocity_field);
        let supg_pressure = super::stabilisation::SupgPressureForm {
            nu: kinematic_viscosity,
            velocity_fn: move |coords: [f64; 3]| {
                supg_pressure_velocity_field.velocity_components_at(coords)
            },
            dt: None,
        };
        let pspg_convection_velocity_field = std::sync::Arc::clone(&velocity_field);
        let pspg_convection = super::stabilisation::PspgConvectionForm {
            nu: kinematic_viscosity,
            velocity_fn: move |coords: [f64; 3]| {
                pspg_convection_velocity_field.velocity_components_at(coords)
            },
            dt: None,
        };

        // Assemble: viscous + convection + stabilisation blocks
        let viscous_triplet = match assemble_transported_viscous_triplet(mesh, v_space, config, 2) {
            Ok(Some(triplet)) => triplet,
            Ok(None) => crate::assembly::assemble_bilinear(&viscous, v_space, v_space, mesh, 2),
            Err(_) => {
                // The public steady-solver result has no typed configuration
                // error channel. Invalid transported fields therefore fail
                // closed with the normal non-converged result.
                break;
            }
        };
        let convection_triplet =
            crate::assembly::assemble_bilinear(&convection, v_space, v_space, mesh, 2);
        let supg_triplet = crate::assembly::assemble_bilinear(&supg, v_space, v_space, mesh, 2);
        let lsic_triplet = crate::assembly::assemble_bilinear(&lsic, v_space, v_space, mesh, 2);
        let pspg_triplet = crate::assembly::assemble_bilinear(&pspg, p_space, p_space, mesh, 2);
        let supg_pressure_triplet = crate::assembly::mixed::assemble_bilinear_two_spaces(
            &supg_pressure,
            v_space,
            p_space,
            mesh,
            2,
        );
        let pspg_convection_triplet = crate::assembly::mixed::assemble_bilinear_two_spaces(
            &pspg_convection,
            p_space,
            v_space,
            mesh,
            2,
        );

        // Combined A = μK + C(u_k) + SUPG + LSIC
        let mut a00 = crate::assembly::TripletMatrix::new(n_v, n_v);
        for triplet in [&viscous_triplet, &convection_triplet] {
            for k in 0..triplet.nnz() {
                a00.push(triplet.rows[k], triplet.cols[k], triplet.values[k]);
            }
        }
        if config.use_residual_stabilisation {
            for triplet in [&supg_triplet, &lsic_triplet] {
                for k in 0..triplet.nnz() {
                    a00.push(triplet.rows[k], triplet.cols[k], triplet.values[k]);
                }
            }
        }

        // Stabilised pressure-pressure block.
        let mut a11 = crate::assembly::TripletMatrix::new(n_p, n_p);
        for k in 0..pspg_triplet.nnz() {
            a11.push(
                pspg_triplet.rows[k],
                pspg_triplet.cols[k],
                pspg_triplet.values[k],
            );
        }

        // Divergence coupling
        let system = &divergence_system;

        let forms_elapsed = iter_start.elapsed();
        let rest_start = std::time::Instant::now();
        // Build monolithic system manually with our combined a00
        let mut mono = crate::assembly::TripletMatrix::new(n_total, n_total);
        // Block (0,0): A = μK + C + SUPG + LSIC
        for k in 0..a00.nnz() {
            mono.push(a00.rows[k], a00.cols[k], a00.values[k]);
        }
        // Block (0,1): B (divergence coupling)
        for k in 0..system.block_01.nnz() {
            mono.push(
                system.block_01.rows[k],
                system.block_01.cols[k] + n_v,
                system.block_01.values[k],
            );
        }
        if config.use_residual_stabilisation {
            for k in 0..supg_pressure_triplet.nnz() {
                mono.push(
                    supg_pressure_triplet.rows[k],
                    supg_pressure_triplet.cols[k] + n_v,
                    supg_pressure_triplet.values[k],
                );
            }
        }
        // Block (1,0): B^T
        for k in 0..system.block_10.nnz() {
            mono.push(
                system.block_10.rows[k] + n_v,
                system.block_10.cols[k],
                system.block_10.values[k],
            );
        }
        if config.use_residual_stabilisation {
            for k in 0..pspg_convection_triplet.nnz() {
                mono.push(
                    pspg_convection_triplet.rows[k] + n_v,
                    pspg_convection_triplet.cols[k],
                    -pspg_convection_triplet.values[k],
                );
            }
        }
        // Block (1,1): PSPG pressure stabilisation
        // `block_10` represents `-∫ q div(u)` because it is the transpose
        // of `B[i,j] = -∫ p_j div(v_i)`.  The PSPG residual must therefore
        // carry the same global sign: `-τ ∫ grad(q)·grad(p)`.  Adding the
        // positive Laplacian here reverses the stabilised pressure response
        // and can drive outlet flow opposite to a prescribed inlet.
        if config.use_residual_stabilisation {
            for k in 0..a11.nnz() {
                mono.push(a11.rows[k] + n_v, a11.cols[k] + n_v, -a11.values[k]);
            }
        }

        // RHS: body force for velocity, zero for pressure
        let force_velocity_field = std::sync::Arc::clone(&velocity_field);
        let standard_force_body = std::sync::Arc::clone(&body_force_fn);
        let body_force = super::forms::StokesBodyForceForm::new(move |coords| {
            let local_velocity = force_velocity_field.velocity_at(coords);
            standard_force_body(coords, local_velocity)
        });
        let rhs_v = crate::assembly::assemble_linear(&body_force, v_space, mesh, 2);

        let supg_force_advection_field = std::sync::Arc::clone(&velocity_field);
        let supg_force_evaluation_field = std::sync::Arc::clone(&velocity_field);
        let supg_force_body = std::sync::Arc::clone(&body_force_fn);
        let supg_body_force = super::stabilisation::SupgBodyForceForm {
            nu: kinematic_viscosity,
            velocity_fn: move |coords: [f64; 3]| {
                supg_force_advection_field.velocity_components_at(coords)
            },
            force_fn: move |coords: [f64; 3]| {
                let local_velocity = supg_force_evaluation_field.velocity_at(coords);
                supg_force_body(coords, local_velocity)
            },
            dt: None,
        };
        let rhs_supg = crate::assembly::assemble_linear(&supg_body_force, v_space, mesh, 2);

        let pspg_force_magnitude_field = std::sync::Arc::clone(&velocity_field);
        let pspg_force_evaluation_field = std::sync::Arc::clone(&velocity_field);
        let pspg_force_body = std::sync::Arc::clone(&body_force_fn);
        let pspg_body_force = super::stabilisation::PspgBodyForceForm {
            rho: stabilisation_rho,
            nu: kinematic_viscosity,
            velocity_magnitude_fn: move |coords: [f64; 3]| {
                pspg_force_magnitude_field.velocity_magnitude_at(coords)
            },
            force_fn: move |coords: [f64; 3]| {
                let local_velocity = pspg_force_evaluation_field.velocity_at(coords);
                pspg_force_body(coords, local_velocity)
            },
            dt: None,
        };
        let rhs_pspg = crate::assembly::assemble_linear(&pspg_body_force, p_space, mesh, 2);
        let mut rhs = vec![0.0; n_total];
        for (i, &val) in rhs_v.iter().enumerate() {
            rhs[i] = val
                + if config.use_residual_stabilisation {
                    rhs_supg[i]
                } else {
                    0.0
                };
        }
        if config.use_residual_stabilisation {
            for (i, &val) in rhs_pspg.iter().enumerate() {
                rhs[n_v + i] = -val;
            }
        }

        // Build BC prescribed values
        let mut bc_values = vec![0.0; n_total];
        let mut is_bc = vec![false; n_total];
        for &dof in bc_dofs {
            assert!(dof < n_v, "velocity Dirichlet DOF {dof} is out of range");
            let point = velocity_dof_coordinates[dof];
            let comp = dof % v_space.num_components();
            let bc_val = bc_velocity_fn(point[0], point[1], point[2]);
            assert!(
                comp < bc_val.len(),
                "velocity boundary function returned too few components"
            );
            is_bc[dof] = true;
            bc_values[dof] = bc_val[comp];
        }
        if n_p > 0 && pin_pressure {
            // Only the closed-boundary constant-pressure nullspace needs a
            // gauge. Open natural tractions already fix the pressure level.
            // Pin the DOF nearest the domain centroid. Pinning an arbitrary
            // corner breaks reflection/rotation symmetries in finite precision
            // and can contaminate small lift coefficients.
            is_bc[n_v + pressure_pin] = true;
            bc_values[n_v + pressure_pin] = 0.0;
            // Taylor-Hood without pressure stabilisation has a structurally
            // zero (p,p) block. Insert the gauge diagonal before CSR
            // conversion so symmetric Dirichlet elimination can set it to 1.
            mono.push(n_v + pressure_pin, n_v + pressure_pin, 0.0);
        }

        // Assemble the complete laminar Newton residual/tangent.  This
        // reuses the same quadrature definitions as the Picard matrix, but
        // differentiates every active velocity-dependent term (convection,
        // SUPG/PSPG/LSIC, and the supplied body-force callback).
        if publish_this_newton {
            let (exact_residual, exact_tangent) = assemble_complete_newton_linearization(
                mesh,
                v_space,
                p_space,
                config,
                mu_eff,
                turbulence_mu_gradient.as_deref(),
                &velocity,
                &pressure,
                &*body_force_fn,
                &*body_force_jacobian_fn,
                2,
            );
            let mut newton_mono = crate::assembly::TripletMatrix::new(n_total, n_total);
            for k in 0..exact_tangent.nnz() {
                newton_mono.push(
                    exact_tangent.rows[k],
                    exact_tangent.cols[k],
                    exact_tangent.values[k],
                );
            }

            let mut newton_csr = newton_mono.to_csr();
            let mut newton_rhs = vec![0.0; n_total];
            apply_dirichlet_to_csr(&mut newton_csr, &mut newton_rhs, &is_bc, &bc_values);
            let mut state = Vec::with_capacity(n_total);
            state.extend_from_slice(&velocity);
            state.extend_from_slice(&pressure);
            let mut state_residual = exact_residual;
            for dof in is_bc
                .iter()
                .enumerate()
                .filter_map(|(dof, constrained)| constrained.then_some(dof))
            {
                state_residual[dof] = state[dof] - bc_values[dof];
            }
            last_newton_linearization = Some(NavierStokesNewtonLinearization {
                state_jacobian: newton_csr,
                state_residual,
                state,
            });
        }

        if assembly_only {
            let residual =
                last_newton_linearization
                    .as_ref()
                    .map_or(f64::INFINITY, |observation| {
                        observation
                            .state_residual
                            .iter()
                            .map(|value| value * value)
                            .sum::<f64>()
                            .sqrt()
                    });
            return NavierStokesResult {
                velocity,
                pressure,
                iterations: 0,
                residual,
                converged: false,
                resolved_linear_solver_backend: LinearSolverBackend::Cpu,
                picard_linearization: None,
                newton_linearization: last_newton_linearization,
            };
        }

        // Apply symmetric Dirichlet BCs
        let rest_elapsed = rest_start.elapsed();
        let merge_start = std::time::Instant::now();
        let mut csr = mono.to_csr();
        apply_dirichlet_to_csr(&mut csr, &mut rhs, &is_bc, &bc_values);
        let merge_elapsed = merge_start.elapsed();
        let assemble_elapsed = iter_start.elapsed();

        // Solve
        let b: Vec<f64> = rhs;
        let solve_start = std::time::Instant::now();
        let mut linear_iterations: usize = 0;

        let x: Vec<f64> = if config.use_direct_solver {
            // Opt-in execution policy for long steady QA runs. On the original
            // 11876-DOF fan matrix, strict-pivot supernodal LU is substantially
            // faster with the same independently checked linear residual. Its
            // own measured wave-size gate decides whether parallelism pays off.
            let factorization =
                if std::env::var("SONIUM_STEADY_SUPERNODAL_LU").as_deref() == Ok("1") {
                    sonium_solvers::direct::SparseLuFactorization::factorize_supernodal(&csr)
                } else {
                    sonium_solvers::direct::SparseLuFactorization::factorize(&csr)
                };
            match factorization.and_then(|lu| lu.solve(&b)) {
                Ok(solution) => solution,
                Err(_) => {
                    return NavierStokesResult {
                        velocity,
                        pressure,
                        iterations: iter,
                        residual: final_residual,
                        converged: false,
                        resolved_linear_solver_backend,
                        picard_linearization: last_linearization,
                        newton_linearization: last_newton_linearization.clone(),
                    };
                }
            }
        } else {
            let mut initial_guess = Vec::with_capacity(n_total);
            initial_guess.extend_from_slice(&velocity);
            initial_guess.extend_from_slice(&pressure);
            let x0: Vec<f64> = initial_guess;
            let schur_csr = (config.linear_solver_cpu_preconditioner
                == LinearSolverCpuPreconditionerKind::JacobiSchur
                || config.linear_solver_cpu_preconditioner
                    == LinearSolverCpuPreconditionerKind::IlutSchur
                || config.linear_solver_cpu_preconditioner
                    == LinearSolverCpuPreconditionerKind::IluSchur
                || config.linear_solver_backend_preconditioner
                    == BackendPreconditionerKind::JacobiSchur)
                .then(|| {
                    pressure_mass_schur_csr(
                        p_space,
                        mesh,
                        mu_eff,
                        pin_pressure.then_some(pressure_pin),
                    )
                });
            match solve_iterative_linear_system(
                config,
                &csr,
                &b,
                Some(&x0),
                n_v,
                schur_csr.as_ref(),
            ) {
                Some((x, backend, iterations)) => {
                    if backend != requested_backend {
                        backend_fell_back = true;
                    }
                    resolved_linear_solver_backend = if backend_fell_back {
                        LinearSolverBackend::Cpu
                    } else {
                        backend
                    };
                    linear_iterations = iterations;
                    x
                }
                None => {
                    return NavierStokesResult {
                        velocity,
                        pressure,
                        iterations: iter,
                        residual: final_residual,
                        converged: false,
                        resolved_linear_solver_backend,
                        picard_linearization: last_linearization,
                        newton_linearization: last_newton_linearization.clone(),
                    };
                }
            }
        };

        let solve_elapsed = solve_start.elapsed();
        if profile_ns {
            eprintln!(
                "NS_PROFILE picard={} assemble_ms={:.1} forms_ms={:.1} rest_ms={:.1} merge_ms={:.1} solve_ms={:.1} linear_iters={}",
                iter + 1,
                assemble_elapsed.as_secs_f64() * 1000.0,
                forms_elapsed.as_secs_f64() * 1000.0,
                rest_elapsed.as_secs_f64() * 1000.0,
                merge_elapsed.as_secs_f64() * 1000.0,
                solve_elapsed.as_secs_f64() * 1000.0,
                linear_iterations,
            );
        }

        // Extract velocity and pressure, then under-relax the nonlinear update.
        let solved_velocity: Vec<f64> = x.iter().take(n_v).copied().collect();
        let solved_pressure: Vec<f64> = x.iter().skip(n_v).copied().collect();
        let new_velocity = if picard_relaxation < 1.0 {
            velocity
                .iter()
                .zip(solved_velocity.iter())
                .map(|(previous, solved)| previous + picard_relaxation * (solved - previous))
                .collect()
        } else {
            solved_velocity
        };
        let new_pressure = if picard_relaxation < 1.0 {
            pressure
                .iter()
                .zip(solved_pressure.iter())
                .map(|(previous, solved)| previous + picard_relaxation * (solved - previous))
                .collect()
        } else {
            solved_pressure
        };

        let mut state = Vec::with_capacity(n_total);
        state.extend_from_slice(&velocity);
        state.extend_from_slice(&pressure);
        let mut fixed_point_residual = Vec::with_capacity(n_total);
        fixed_point_residual.extend(
            new_velocity
                .iter()
                .zip(&velocity)
                .map(|(next, previous)| next - previous),
        );
        fixed_point_residual.extend(
            new_pressure
                .iter()
                .zip(&pressure)
                .map(|(next, previous)| next - previous),
        );
        last_linearization = Some(NavierStokesPicardLinearization {
            state_jacobian: csr.clone(),
            state,
            fixed_point_residual,
        });

        // Check convergence
        let vel_diff_sq: f64 = new_velocity
            .iter()
            .zip(&velocity)
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let vel_norm_sq: f64 = new_velocity.iter().map(|v| v * v).sum();
        let rel_change = if vel_norm_sq > 1e-30 {
            vel_diff_sq.sqrt() / vel_norm_sq.sqrt()
        } else {
            vel_diff_sq.sqrt()
        };

        velocity = new_velocity;
        pressure = new_pressure;
        final_residual = rel_change;

        if config.max_picard_iterations <= 5
            || (iter + 1) % 5 == 0
            || rel_change < config.picard_tolerance
        {
            eprintln!(
                "  Picard iter {}: rel_change = {:.6e}",
                iter + 1,
                rel_change
            );
        }

        if iter + 1 >= config.min_picard_iterations && rel_change < config.picard_tolerance {
            converged = true;
            return NavierStokesResult {
                velocity,
                pressure,
                iterations: iter + 1,
                residual: final_residual,
                converged,
                resolved_linear_solver_backend,
                picard_linearization: last_linearization,
                newton_linearization: last_newton_linearization.clone(),
            };
        }
    }

    NavierStokesResult {
        velocity,
        pressure,
        iterations: config.max_picard_iterations,
        residual: final_residual,
        converged,
        resolved_linear_solver_backend,
        picard_linearization: last_linearization,
        newton_linearization: last_newton_linearization,
    }
}

/// Assemble complete steady Navier--Stokes residual and tangent
/// used by checked Newton observation.
///
/// The ordinary Picard path intentionally keeps stabilisation coefficients
/// frozen while solving its Oseen system. This helper evaluates the same
/// residual terms at a mapped input state and differentiates dependence
/// on velocity: convection, SUPG/PSPG/LSIC, velocity-dependent body force,
/// pressure couplings and, when supplied, algebraic turbulence viscosity
/// sensitivity. The latter is valid only for the unique-peak contract
/// admitted by [`super::turbulence::TurbulenceModel`].
#[allow(clippy::too_many_arguments)]
fn assemble_complete_newton_linearization<VSpace, PSpace, ForceFn, ForceJacFn>(
    mesh: &crate::mesh::Mesh,
    v_space: &VSpace,
    p_space: &PSpace,
    config: &NavierStokesConfig,
    viscous_mu: f64,
    viscous_mu_gradient: Option<&[f64]>,
    velocity: &[f64],
    pressure: &[f64],
    body_force_fn: &ForceFn,
    body_force_jacobian_fn: &ForceJacFn,
    quad_order: usize,
) -> (Vec<f64>, crate::assembly::TripletMatrix<f64>)
where
    VSpace: crate::spaces::FunctionSpace,
    PSpace: crate::spaces::FunctionSpace,
    ForceFn: Fn([f64; 3], [f64; 3]) -> Vec<f64>,
    ForceJacFn: Fn([f64; 3], [f64; 3]) -> [[f64; 3]; 3],
{
    let n_v = v_space.num_global_dofs(mesh);
    let n_p = p_space.num_global_dofs(mesh);
    let n_total = n_v + n_p;
    let dim = mesh.dimension;
    let mut residual = vec![0.0; n_total];
    let mut tangent = crate::assembly::TripletMatrix::new(n_total, n_total);
    let rho = config.rho;
    let viscous_mu_per_element = config.transported_eddy_viscosity_per_element.as_deref();
    for element_index in 0..mesh.num_elements() {
        let element = &mesh.elements[element_index];
        let element_viscous_mu = viscous_mu_per_element
            .map(|viscosities| viscosities[element_index])
            .unwrap_or(viscous_mu);
        let element_nu = element_viscous_mu / rho.max(f64::EPSILON);
        let velocity_dofs = v_space.local_to_global(mesh, element_index);
        let pressure_dofs = p_space.local_to_global(mesh, element_index);
        let local_velocity: Vec<_> = velocity_dofs.iter().map(|&dof| velocity[dof]).collect();
        let local_pressure: Vec<_> = pressure_dofs.iter().map(|&dof| pressure[dof]).collect();
        let quadrature = QuadratureRule::new(element.element_type, quad_order);

        for point in quadrature.iter() {
            let v_test =
                v_space.tabulate(mesh, element_index, point.xi(), point.eta(), point.zeta());
            let p_test =
                p_space.tabulate(mesh, element_index, point.xi(), point.eta(), point.zeta());
            let weight = point.weight * v_test.det_j;
            let u = velocity_at(&v_test, &local_velocity, dim);
            let grad_u = velocity_gradient_at(&v_test, &local_velocity, dim);
            let pressure_gradient = (0..dim)
                .map(|direction| {
                    local_pressure
                        .iter()
                        .enumerate()
                        .map(|(j, value)| p_test.grad(j, direction) * value)
                        .sum::<f64>()
                })
                .collect::<Vec<_>>();
            let pressure_value = local_pressure
                .iter()
                .enumerate()
                .map(|(j, value)| p_test.value(j) * value)
                .sum::<f64>();
            let velocity_magnitude = u.iter().map(|value| value * value).sum::<f64>().sqrt();
            let h = if dim == 2 {
                (2.0 * v_test.det_j).sqrt()
            } else {
                (6.0 * v_test.det_j).cbrt()
            };
            let tau =
                super::stabilisation::stabilisation_tau(velocity_magnitude, h, element_nu, None);
            let tau_lsic = tau * velocity_magnitude * h * 0.5;
            let force = body_force_fn(v_test.physical_coords, point_velocity(&u, dim));
            let force_jacobian =
                body_force_jacobian_fn(v_test.physical_coords, point_velocity(&u, dim));

            let div_u = (0..dim)
                .map(|component| grad_u[component][component])
                .sum::<f64>();
            for (local_row, &global_row) in velocity_dofs.iter().enumerate() {
                let component = local_row % dim;
                let test_value = v_test.vector_value(local_row, component);
                let test_gradient = (0..dim)
                    .map(|direction| v_test.grad(local_row, direction))
                    .collect::<Vec<_>>();
                let streamline_test = dot(&u, &test_gradient);
                let advected_component = dot(&u, &grad_u[component]);
                let divergence_test = test_gradient[component];
                let viscous_residual = grad_u[component]
                    .iter()
                    .zip(&test_gradient)
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
                let mut row_residual = element_viscous_mu * viscous_residual
                    + rho * advected_component * test_value
                    - test_value * force.get(component).copied().unwrap_or(0.0)
                    - pressure_value * divergence_test;

                if config.use_residual_stabilisation {
                    row_residual += rho * tau_lsic * divergence_test * div_u;
                    row_residual += rho * tau * streamline_test * advected_component;
                    row_residual += tau * streamline_test * pressure_gradient[component];
                    row_residual -=
                        tau * streamline_test * force.get(component).copied().unwrap_or(0.0);
                }
                residual[global_row] += row_residual * weight;

                for (local_column, &global_column) in velocity_dofs.iter().enumerate() {
                    let column_component = local_column % dim;
                    let delta_u = (0..dim)
                        .map(|direction| v_test.vector_value(local_column, direction))
                        .collect::<Vec<_>>();
                    let mut delta_grad = vec![vec![0.0; dim]; dim];
                    for (direction, slot) in delta_grad[column_component].iter_mut().enumerate() {
                        *slot = v_test.grad(local_column, direction);
                    }
                    let delta_advected =
                        dot(&delta_u, &grad_u[component]) + dot(&u, &delta_grad[component]);
                    let delta_streamline_test = dot(&delta_u, &test_gradient);
                    let delta_velocity_magnitude = if velocity_magnitude > f64::EPSILON {
                        dot(&u, &delta_u) / velocity_magnitude
                    } else {
                        0.0
                    };
                    let delta_mu = if viscous_mu_per_element.is_some() {
                        0.0
                    } else {
                        viscous_mu_gradient
                            .and_then(|gradient| gradient.get(global_column).copied())
                            .unwrap_or(0.0)
                    };
                    let delta_nu = delta_mu / rho.max(f64::EPSILON);
                    let delta_tau = tau_directional_derivative(tau, h, &u, &delta_u)
                        + tau_viscosity_directional_derivative(tau, h, element_nu, delta_nu);
                    let delta_tau_lsic =
                        h * 0.5 * (delta_tau * velocity_magnitude + tau * delta_velocity_magnitude);
                    let delta_divergence = delta_grad
                        .iter()
                        .enumerate()
                        .map(|(c, row)| row[c])
                        .sum::<f64>();
                    let delta_force = (0..dim)
                        .map(|output| {
                            (0..dim)
                                .map(|input| force_jacobian[output][input] * delta_u[input])
                                .sum::<f64>()
                        })
                        .collect::<Vec<_>>();
                    let force_component = force.get(component).copied().unwrap_or(0.0);
                    let delta_force_component = delta_force.get(component).copied().unwrap_or(0.0);
                    let mut value = element_viscous_mu
                        * dot(&delta_grad[component], &test_gradient)
                        + delta_mu * viscous_residual
                        + rho * delta_advected * test_value
                        - test_value * delta_force_component;

                    if config.use_residual_stabilisation {
                        value += rho
                            * (delta_tau_lsic * divergence_test * div_u
                                + tau_lsic * divergence_test * delta_divergence);
                        value += rho
                            * (delta_tau * streamline_test * advected_component
                                + tau * delta_streamline_test * advected_component
                                + tau * streamline_test * delta_advected);
                        value += delta_tau * streamline_test * pressure_gradient[component]
                            + tau * delta_streamline_test * pressure_gradient[component];
                        value -= delta_tau * streamline_test * force_component
                            + tau * delta_streamline_test * force_component
                            + tau * streamline_test * delta_force_component;
                    }
                    tangent.push(global_row, global_column, value * weight);
                }

                for (local_column, &global_column) in pressure_dofs.iter().enumerate() {
                    let pressure_basis = p_test.value(local_column);
                    let pressure_gradient_basis = (0..dim)
                        .map(|direction| p_test.grad(local_column, direction))
                        .collect::<Vec<_>>();
                    let mut value = -pressure_basis * divergence_test;
                    if config.use_residual_stabilisation {
                        value += tau * streamline_test * pressure_gradient_basis[component];
                    }
                    tangent.push(global_row, n_v + global_column, value * weight);
                }
            }

            for (local_row, &global_row) in pressure_dofs.iter().enumerate() {
                let test_value = p_test.value(local_row);
                let test_gradient = (0..dim)
                    .map(|direction| p_test.grad(local_row, direction))
                    .collect::<Vec<_>>();
                let grad_dot_advected = (0..dim)
                    .map(|component| test_gradient[component] * dot(&u, &grad_u[component]))
                    .sum::<f64>();
                let grad_dot_force = test_gradient
                    .iter()
                    .enumerate()
                    .map(|(component, value)| value * force.get(component).copied().unwrap_or(0.0))
                    .sum::<f64>();
                let mut row_residual = -test_value * div_u;
                if config.use_residual_stabilisation {
                    row_residual += -tau * grad_dot_advected
                        - tau / rho.max(f64::EPSILON)
                            * grad_dot_pressure(&test_gradient, &pressure_gradient)
                        + tau / rho.max(f64::EPSILON) * grad_dot_force;
                }
                residual[n_v + global_row] += row_residual * weight;

                for (local_column, &global_column) in velocity_dofs.iter().enumerate() {
                    let column_component = local_column % dim;
                    let delta_u = (0..dim)
                        .map(|direction| v_test.vector_value(local_column, direction))
                        .collect::<Vec<_>>();
                    let mut delta_grad = vec![vec![0.0; dim]; dim];
                    for (direction, slot) in delta_grad[column_component].iter_mut().enumerate() {
                        *slot = v_test.grad(local_column, direction);
                    }
                    let delta_velocity_magnitude = if velocity_magnitude > f64::EPSILON {
                        dot(&u, &delta_u) / velocity_magnitude
                    } else {
                        0.0
                    };
                    let delta_tau = tau_directional_derivative(tau, h, &u, &delta_u);
                    let delta_divergence = delta_grad
                        .iter()
                        .enumerate()
                        .map(|(c, row)| row[c])
                        .sum::<f64>();
                    let delta_advected = (0..dim)
                        .map(|component| {
                            test_gradient[component]
                                * (dot(&delta_u, &grad_u[component])
                                    + dot(&u, &delta_grad[component]))
                        })
                        .sum::<f64>();
                    let delta_force_dot = (0..dim)
                        .map(|output| {
                            test_gradient[output]
                                * (0..dim)
                                    .map(|input| force_jacobian[output][input] * delta_u[input])
                                    .sum::<f64>()
                        })
                        .sum::<f64>();
                    let mut value = -test_value * delta_divergence;
                    if config.use_residual_stabilisation {
                        value += -delta_tau * grad_dot_advected - tau * delta_advected;
                        value += -delta_tau / rho.max(f64::EPSILON)
                            * grad_dot_pressure(&test_gradient, &pressure_gradient);
                        value += (delta_tau * grad_dot_force + tau * delta_force_dot)
                            / rho.max(f64::EPSILON);
                        // Keep the local magnitude derivative explicit in the
                        // expression above; this variable documents that the
                        // state dependence is intentionally velocity-only.
                        let _ = delta_velocity_magnitude;
                    }
                    tangent.push(n_v + global_row, global_column, value * weight);
                }

                for (local_column, &global_column) in pressure_dofs.iter().enumerate() {
                    let pressure_gradient_basis = (0..dim)
                        .map(|direction| p_test.grad(local_column, direction))
                        .collect::<Vec<_>>();
                    let value = if config.use_residual_stabilisation {
                        -tau / rho.max(f64::EPSILON)
                            * grad_dot_pressure(&test_gradient, &pressure_gradient_basis)
                    } else {
                        0.0
                    };
                    tangent.push(n_v + global_row, n_v + global_column, value * weight);
                }
            }
        }
    }

    (residual, tangent)
}

fn velocity_at(tab: &BasisTabulation, local_solution: &[f64], dim: usize) -> Vec<f64> {
    (0..dim)
        .map(|component| {
            (0..tab.n_dofs)
                .map(|dof| tab.vector_value(dof, component) * local_solution[dof])
                .sum()
        })
        .collect()
}

fn velocity_gradient_at(
    tab: &BasisTabulation,
    local_solution: &[f64],
    dim: usize,
) -> Vec<Vec<f64>> {
    (0..dim)
        .map(|component| {
            (0..dim)
                .map(|direction| {
                    (0..tab.n_dofs)
                        .map(|dof| {
                            if dof % dim == component {
                                tab.grad(dof, direction) * local_solution[dof]
                            } else {
                                0.0
                            }
                        })
                        .sum()
                })
                .collect()
        })
        .collect()
}

fn point_velocity(values: &[f64], dim: usize) -> [f64; 3] {
    let mut result = [0.0; 3];
    result[..dim].copy_from_slice(&values[..dim]);
    result
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn grad_dot_pressure(test_gradient: &[f64], pressure_gradient: &[f64]) -> f64 {
    dot(test_gradient, pressure_gradient)
}

fn tau_directional_derivative(tau: f64, h: f64, velocity: &[f64], delta_velocity: &[f64]) -> f64 {
    if tau == 0.0 || h <= f64::EPSILON {
        return 0.0;
    }
    -4.0 * tau.powi(3) * dot(velocity, delta_velocity) / (h * h)
}

fn tau_viscosity_directional_derivative(
    tau: f64,
    h: f64,
    kinematic_viscosity: f64,
    delta_kinematic_viscosity: f64,
) -> f64 {
    if tau == 0.0 || h <= f64::EPSILON {
        return 0.0;
    }
    -36.0 * tau.powi(3) * kinematic_viscosity * delta_kinematic_viscosity / h.powi(4)
}

/// Exact state residual and tangent for one BDF Navier--Stokes step.
///
/// The cached transient solver intentionally assembles an Oseen matrix. This
/// helper is the checked Newton counterpart used when a caller needs to
/// publish a derivative for an adjoint: it adds the BDF mass block to the
/// complete steady residual/tangent, including the optional algebraic-
/// turbulence viscosity subgradient, and applies the same symmetric
/// Dirichlet and pressure-gauge conditioning as the transient solver.
///
/// `viscous_mu_gradient` is the global velocity-DOF derivative of the
/// effective viscosity. An element-local transported field may instead be
/// supplied through [`NavierStokesConfig::transported_eddy_viscosity_per_element`];
/// it is held fixed here and its scalar cross-derivative belongs in the
/// caller's coupled block. A non-laminar caller must provide it only when its
/// turbulence closure has admitted a valid differentiable branch (for the
/// algebraic PC-fan closure, a unique non-zero velocity peak).
#[allow(clippy::too_many_arguments)]
pub fn assemble_bdf_newton_linearization<VSpace, PSpace, BcFn, ForceFn, ForceJacFn>(
    mesh: &crate::mesh::Mesh,
    v_space: &VSpace,
    p_space: &PSpace,
    config: &NavierStokesConfig,
    bdf: &super::time_stepping::BdfConfig,
    state: &super::time_stepping::BdfState,
    bc_velocity_fn: &BcFn,
    bc_dofs: &[usize],
    body_force_fn: &ForceFn,
    body_force_jacobian_fn: &ForceJacFn,
    workspace: &super::workspace::BdfWorkspace,
    viscous_mu: f64,
    viscous_mu_gradient: Option<&[f64]>,
) -> Option<(Vec<f64>, sonium_solvers::CsrMatrix<f64>)>
where
    VSpace: crate::spaces::FunctionSpace,
    PSpace: crate::spaces::FunctionSpace,
    BcFn: Fn(f64, f64, f64) -> Vec<f64>,
    ForceFn: Fn([f64; 3]) -> Vec<f64>,
    ForceJacFn: Fn([f64; 3]) -> [[f64; 3]; 3],
{
    // The segregated transported field is intentionally lagged in the
    // momentum solve. Its scalar/momentum cross-derivative is not available
    // to this exact Newton tangent API, so do not publish a misleading
    // coupled Jacobian.
    let n_v = v_space.num_global_dofs(mesh);
    let n_p = p_space.num_global_dofs(mesh);
    let n_total = n_v.checked_add(n_p)?;
    if workspace.n_v != n_v
        || workspace.n_p != n_p
        || state.velocity.len() != n_v
        || state.velocity_prev.len() != n_v
        || state.pressure.len() != n_p
        || mesh.dimension != v_space.num_components()
        || !bdf.dt.is_finite()
        || bdf.dt <= 0.0
        || !config.rho.is_finite()
        || config.rho <= 0.0
        || !viscous_mu.is_finite()
        || viscous_mu <= 0.0
        || state
            .velocity
            .iter()
            .chain(&state.velocity_prev)
            .chain(&state.pressure)
            .any(|value| !value.is_finite())
    {
        return None;
    }
    if let Some(viscosities) = config.transported_eddy_viscosity_per_element.as_deref()
        && (viscosities.len() != mesh.num_elements()
            || viscosities.iter().any(|&eddy_viscosity| {
                let dynamic_viscosity = config.mu + config.rho * eddy_viscosity;
                !eddy_viscosity.is_finite()
                    || eddy_viscosity < 0.0
                    || !dynamic_viscosity.is_finite()
                    || dynamic_viscosity <= 0.0
            })
            || viscous_mu_gradient.is_some())
    {
        return None;
    }
    if let Some(gradient) = viscous_mu_gradient
        && (gradient.len() != n_v || gradient.iter().any(|value| !value.is_finite()))
    {
        return None;
    }
    if bc_dofs.iter().any(|&dof| dof >= n_v) {
        return None;
    }

    let body_force = |coords: [f64; 3], _velocity: [f64; 3]| body_force_fn(coords);
    let body_force_jacobian =
        |coords: [f64; 3], _velocity: [f64; 3]| body_force_jacobian_fn(coords);
    let (mut residual, mut tangent) = assemble_complete_newton_linearization(
        mesh,
        v_space,
        p_space,
        config,
        viscous_mu,
        viscous_mu_gradient,
        &state.velocity,
        &state.pressure,
        &body_force,
        &body_force_jacobian,
        workspace.quad_order,
    );

    let coefficients = super::time_stepping::BdfCoefficients::for_order(bdf.order);
    let mass = workspace.mass_triplet.to_csr();
    let mass_scale = config.rho * coefficients.a0 / bdf.dt;
    for entry in 0..workspace.mass_triplet.nnz() {
        tangent.push(
            workspace.mass_triplet.rows[entry],
            workspace.mass_triplet.cols[entry],
            mass_scale * workspace.mass_triplet.values[entry],
        );
    }
    let history_scale = config.rho / bdf.dt;
    let mut history_source = vec![0.0; n_v];
    for ((source, &current), &previous) in history_source
        .iter_mut()
        .zip(&state.velocity)
        .zip(&state.velocity_prev)
    {
        *source = history_scale * (-coefficients.a1 * current - coefficients.a2 * previous);
    }
    let mass_current = mass.matvec(&state.velocity);
    let mass_history = mass.matvec(&history_source);
    for row in 0..n_v {
        residual[row] += mass_scale * mass_current[row] - mass_history[row];
    }

    let mut is_bc = vec![false; n_total];
    let mut bc_values = vec![0.0; n_total];
    let components = v_space.num_components();
    for &dof in bc_dofs {
        is_bc[dof] = true;
        let point = workspace.velocity_dof_coordinates[dof];
        let values = bc_velocity_fn(point[0], point[1], point[2]);
        bc_values[dof] = values.get(dof % components).copied().unwrap_or(0.0);
    }
    if n_p > 0 {
        if workspace.pressure_pin >= n_p {
            return None;
        }
        is_bc[n_v + workspace.pressure_pin] = true;
        bc_values[n_v + workspace.pressure_pin] = workspace.pressure_pin_value;
        tangent.push(
            n_v + workspace.pressure_pin,
            n_v + workspace.pressure_pin,
            0.0,
        );
    }

    let mut jacobian = tangent.to_csr();
    let mut unused_rhs = vec![0.0; n_total];
    apply_dirichlet_to_csr(&mut jacobian, &mut unused_rhs, &is_bc, &bc_values);

    let mut state_vector = Vec::with_capacity(n_total);
    state_vector.extend_from_slice(&state.velocity);
    state_vector.extend_from_slice(&state.pressure);
    for (dof, constrained) in is_bc.iter().copied().enumerate() {
        if constrained {
            residual[dof] = state_vector[dof] - bc_values[dof];
        }
    }
    if residual.iter().any(|value| !value.is_finite())
        || jacobian.values.iter().any(|value| !value.is_finite())
    {
        return None;
    }
    Some((residual, jacobian))
}

/// Assemble the exact design derivative of the body-force contribution to a
/// steady Navier--Stokes residual at a prescribed state.
///
/// This helper deliberately differentiates only the body-force terms.  The
/// mesh, boundary conditions, stabilisation coefficients, and state are held
/// fixed; callers that change geometry or topology must provide the remaining
/// shape/material terms separately and must not promote this matrix as a full
/// shape derivative.  The callback returns a `3 × design_dimension` matrix
/// containing `∂f_component/∂design` at the quadrature point.
#[allow(clippy::too_many_arguments)]
pub fn assemble_body_force_design_residual<VSpace, PSpace, ForceDesignFn>(
    mesh: &crate::mesh::Mesh,
    v_space: &VSpace,
    p_space: &PSpace,
    config: &NavierStokesConfig,
    velocity: &[f64],
    design_dimension: usize,
    body_force_design_jacobian_fn: ForceDesignFn,
    quad_order: usize,
) -> Result<Vec<Vec<f64>>, BodyForceDesignDerivativeError>
where
    VSpace: crate::spaces::FunctionSpace,
    PSpace: crate::spaces::FunctionSpace,
    ForceDesignFn: Fn([f64; 3], [f64; 3]) -> Vec<Vec<f64>>,
{
    assemble_body_force_design_residual_with_viscosity(
        mesh,
        v_space,
        p_space,
        config,
        config.mu,
        velocity,
        design_dimension,
        body_force_design_jacobian_fn,
        quad_order,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn assemble_body_force_design_residual_with_viscosity<VSpace, PSpace, ForceDesignFn>(
    mesh: &crate::mesh::Mesh,
    v_space: &VSpace,
    p_space: &PSpace,
    config: &NavierStokesConfig,
    viscous_mu: f64,
    velocity: &[f64],
    design_dimension: usize,
    body_force_design_jacobian_fn: ForceDesignFn,
    quad_order: usize,
) -> Result<Vec<Vec<f64>>, BodyForceDesignDerivativeError>
where
    VSpace: crate::spaces::FunctionSpace,
    PSpace: crate::spaces::FunctionSpace,
    ForceDesignFn: Fn([f64; 3], [f64; 3]) -> Vec<Vec<f64>>,
{
    let n_v = v_space.num_global_dofs(mesh);
    let n_p = p_space.num_global_dofs(mesh);
    let n_total = n_v + n_p;
    let dim = mesh.dimension;
    if velocity.len() != n_v {
        return Err(BodyForceDesignDerivativeError::InvalidShape);
    }
    let mut residual_design = vec![vec![0.0; design_dimension]; n_total];
    let rho = config.rho.max(f64::EPSILON);
    let nu = viscous_mu / rho;

    for element_index in 0..mesh.num_elements() {
        let element = &mesh.elements[element_index];
        let velocity_dofs = v_space.local_to_global(mesh, element_index);
        let pressure_dofs = p_space.local_to_global(mesh, element_index);
        let local_velocity: Vec<f64> = velocity_dofs.iter().map(|&dof| velocity[dof]).collect();
        let quadrature = QuadratureRule::new(element.element_type, quad_order);

        for point in quadrature.iter() {
            let v_test =
                v_space.tabulate(mesh, element_index, point.xi(), point.eta(), point.zeta());
            let p_test =
                p_space.tabulate(mesh, element_index, point.xi(), point.eta(), point.zeta());
            let weight = point.weight * v_test.det_j;
            let state_velocity = point_velocity(&velocity_at(&v_test, &local_velocity, dim), dim);
            let velocity_magnitude = state_velocity
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            let h = if dim == 2 {
                (2.0 * v_test.det_j).sqrt()
            } else {
                (6.0 * v_test.det_j).cbrt()
            };
            let tau = super::stabilisation::stabilisation_tau(velocity_magnitude, h, nu, None);
            let design_force =
                body_force_design_jacobian_fn(v_test.physical_coords, state_velocity);
            if design_force.len() != dim
                || design_force.iter().any(|row| row.len() != design_dimension)
            {
                return Err(BodyForceDesignDerivativeError::InvalidShape);
            }
            if design_force
                .iter()
                .flat_map(|row| row.iter())
                .any(|value| !value.is_finite())
            {
                return Err(BodyForceDesignDerivativeError::NonFiniteValue);
            }

            for (local_row, &global_row) in velocity_dofs.iter().enumerate() {
                let component = local_row % dim;
                let test_value = v_test.vector_value(local_row, component);
                let streamline_test = if config.use_residual_stabilisation {
                    (0..dim)
                        .map(|direction| {
                            state_velocity[direction] * v_test.grad(local_row, direction)
                        })
                        .sum::<f64>()
                } else {
                    0.0
                };
                let coefficient = -test_value
                    - if config.use_residual_stabilisation {
                        tau * streamline_test
                    } else {
                        0.0
                    };
                for design_index in 0..design_dimension {
                    residual_design[global_row][design_index] +=
                        coefficient * design_force[component][design_index] * weight;
                }
            }

            if config.use_residual_stabilisation {
                for (local_row, &global_row) in pressure_dofs.iter().enumerate() {
                    for design_index in 0..design_dimension {
                        let gradient_force = (0..dim)
                            .map(|component| {
                                p_test.grad(local_row, component)
                                    * design_force[component][design_index]
                            })
                            .sum::<f64>();
                        residual_design[n_v + global_row][design_index] +=
                            tau / rho * gradient_force * weight;
                    }
                }
            }
        }
    }
    Ok(residual_design)
}

/// Assemble the volume residual derivative caused by an analytic nodal mesh
/// motion field.
///
/// This is the geometry counterpart to [`assemble_body_force_design_residual`]
/// for the resolved PC-fan production discretisation: three-dimensional
/// affine P1/P1 tetrahedra.  The state DOFs are held fixed while the affine
/// map, physical gradients, element measure, SUPG/PSPG/LSIC coefficients, and
/// coordinate-dependent body force are differentiated analytically.  Boundary
/// rows and prescribed boundary values remain the caller's responsibility,
/// just as for the body-force derivative helper.
///
/// The explicit P1/P1 admission is intentional.  Reusing this contraction for
/// higher-order or non-affine elements would silently omit geometry terms.
/// The algebraic turbulence closure is supported because its effective
/// viscosity is a function of the fixed state DOFs, not of moved
/// coordinates; its value is therefore constant in this mesh derivative.
#[allow(clippy::too_many_arguments)]
pub fn assemble_mesh_motion_design_residual<
    const DESIGN_DIM: usize,
    VSpace,
    PSpace,
    ForceFn,
    ForcePositionFn,
>(
    mesh: &crate::mesh::Mesh,
    v_space: &VSpace,
    p_space: &PSpace,
    config: &NavierStokesConfig,
    velocity: &[f64],
    pressure: &[f64],
    nodal_coordinate_jacobian: &[[[f64; 3]; DESIGN_DIM]],
    body_force_fn: ForceFn,
    body_force_position_jacobian_fn: ForcePositionFn,
) -> Result<Vec<Vec<f64>>, MeshMotionDesignDerivativeError>
where
    VSpace: crate::spaces::FunctionSpace,
    PSpace: crate::spaces::FunctionSpace,
    ForceFn: Fn([f64; 3], [f64; 3]) -> [f64; 3],
    ForcePositionFn: Fn([f64; 3], [f64; 3]) -> [[f64; 3]; 3],
{
    if mesh.dimension != 3 {
        return Err(MeshMotionDesignDerivativeError::UnsupportedDiscretisation);
    }
    let velocity_dimension = v_space.num_global_dofs(mesh);
    let pressure_dimension = p_space.num_global_dofs(mesh);
    if velocity.len() != velocity_dimension || pressure.len() != pressure_dimension {
        return Err(MeshMotionDesignDerivativeError::DimensionMismatch("state"));
    }
    if nodal_coordinate_jacobian.len() != mesh.nodes.len() {
        return Err(MeshMotionDesignDerivativeError::DimensionMismatch(
            "nodal coordinate Jacobian",
        ));
    }
    if velocity
        .iter()
        .chain(pressure)
        .any(|value| !value.is_finite())
    {
        return Err(MeshMotionDesignDerivativeError::NonFiniteInput("state"));
    }
    if mesh
        .nodes
        .iter()
        .any(|node| !node.x.is_finite() || !node.y.is_finite() || !node.z.is_finite())
    {
        return Err(MeshMotionDesignDerivativeError::NonFiniteInput(
            "mesh coordinates",
        ));
    }
    if nodal_coordinate_jacobian
        .iter()
        .flatten()
        .any(|value| value.iter().any(|component| !component.is_finite()))
    {
        return Err(MeshMotionDesignDerivativeError::NonFiniteInput(
            "nodal coordinate Jacobian",
        ));
    }

    let mut residual_design = vec![vec![0.0; DESIGN_DIM]; velocity_dimension + pressure_dimension];
    let rho = config.rho;
    let velocity_magnitude_max = velocity
        .as_chunks::<3>()
        .0
        .iter()
        .take(mesh.nodes.len())
        .map(|node_velocity| {
            node_velocity
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt()
        })
        .fold(0.0_f64, f64::max);
    let viscous_mu = config
        .effective_dynamic_viscosity_and_speed_derivative(velocity_magnitude_max)
        .0;
    let nu = viscous_mu / rho.max(f64::EPSILON);

    for (element_index, element) in mesh.elements.iter().enumerate() {
        if element.element_type != crate::mesh::ElementType::Tetrahedron || element.nodes.len() != 4
        {
            return Err(MeshMotionDesignDerivativeError::UnsupportedDiscretisation);
        }
        let velocity_dofs = v_space.local_to_global(mesh, element_index);
        let pressure_dofs = p_space.local_to_global(mesh, element_index);
        if (velocity_dofs.len() != 12 && velocity_dofs.len() != 30) || pressure_dofs.len() != 4 {
            return Err(MeshMotionDesignDerivativeError::UnsupportedDiscretisation);
        }

        let velocity_scalar_dofs = velocity_dofs.len() / 3;
        let velocity_is_p2 = velocity_scalar_dofs == 10;
        let mut coordinates = [[0.0; 3]; 4];
        for (local_node, &node_index) in element.nodes.iter().enumerate() {
            let node = mesh.nodes.get(node_index).ok_or(
                MeshMotionDesignDerivativeError::DimensionMismatch("element node"),
            )?;
            coordinates[local_node] = [node.x, node.y, node.z];
        }
        let jacobian = [
            [
                coordinates[1][0] - coordinates[0][0],
                coordinates[2][0] - coordinates[0][0],
                coordinates[3][0] - coordinates[0][0],
            ],
            [
                coordinates[1][1] - coordinates[0][1],
                coordinates[2][1] - coordinates[0][1],
                coordinates[3][1] - coordinates[0][1],
            ],
            [
                coordinates[1][2] - coordinates[0][2],
                coordinates[2][2] - coordinates[0][2],
                coordinates[3][2] - coordinates[0][2],
            ],
        ];
        let determinant = det3(jacobian);
        let Some(inverse) = inverse3(jacobian) else {
            return Err(MeshMotionDesignDerivativeError::InvalidElement(
                element_index,
            ));
        };
        let determinant_abs = determinant.abs();
        if !determinant_abs.is_finite() || determinant_abs <= f64::EPSILON {
            return Err(MeshMotionDesignDerivativeError::InvalidElement(
                element_index,
            ));
        }
        let inverse_transpose = transpose3(inverse);

        let mut local_velocity = [[0.0; 3]; 10];
        let mut local_pressure = [0.0; 4];
        for local_node in 0..velocity_scalar_dofs {
            for component in 0..3 {
                local_velocity[local_node][component] =
                    velocity[velocity_dofs[3 * local_node + component]];
            }
        }
        for local_node in 0..4 {
            local_pressure[local_node] = pressure[pressure_dofs[local_node]];
        }

        let quadrature = QuadratureRule::new(element.element_type, 2);
        for point in quadrature.iter() {
            let geometry_reference_values =
                crate::basis::p1_tetrahedron(point.xi(), point.eta(), point.zeta());
            let (reference_values, reference_gradients): (Vec<f64>, Vec<[f64; 3]>) =
                if velocity_is_p2 {
                    (
                        crate::basis::p2_tetrahedron(point.xi(), point.eta(), point.zeta())
                            .to_vec(),
                        crate::basis::p2_tetrahedron_grad(point.xi(), point.eta(), point.zeta())
                            .to_vec(),
                    )
                } else {
                    (
                        geometry_reference_values.to_vec(),
                        crate::basis::p1_tetrahedron_grad().to_vec(),
                    )
                };
            let pressure_reference_values = geometry_reference_values;
            let pressure_reference_gradients = crate::basis::p1_tetrahedron_grad();
            let pressure_gradients =
                pressure_reference_gradients.map(|gradient| mat_vec3(inverse_transpose, gradient));
            let mut gradients = vec![[0.0; 3]; velocity_scalar_dofs];
            for local_node in 0..velocity_scalar_dofs {
                gradients[local_node] =
                    mat_vec3(inverse_transpose, reference_gradients[local_node]);
            }
            let weight = point.weight * determinant_abs;

            let mut physical_coords = [0.0; 3];
            let mut state_velocity = [0.0; 3];
            let mut velocity_gradient = [[0.0; 3]; 3];
            let mut pressure_value = 0.0;
            let mut pressure_gradient = [0.0; 3];
            for local_node in 0..4 {
                for direction in 0..3 {
                    physical_coords[direction] +=
                        pressure_reference_values[local_node] * coordinates[local_node][direction];
                    state_velocity[direction] +=
                        reference_values[local_node] * local_velocity[local_node][direction];
                    pressure_gradient[direction] +=
                        local_pressure[local_node] * pressure_gradients[local_node][direction];
                }
                pressure_value +=
                    pressure_reference_values[local_node] * local_pressure[local_node];
                for component in 0..3 {
                    for direction in 0..3 {
                        velocity_gradient[component][direction] += local_velocity[local_node]
                            [component]
                            * gradients[local_node][direction];
                    }
                }
            }
            for local_node in 4..velocity_scalar_dofs {
                for direction in 0..3 {
                    state_velocity[direction] +=
                        reference_values[local_node] * local_velocity[local_node][direction];
                }
                for component in 0..3 {
                    for direction in 0..3 {
                        velocity_gradient[component][direction] += local_velocity[local_node]
                            [component]
                            * gradients[local_node][direction];
                    }
                }
            }

            let force = body_force_fn(physical_coords, state_velocity);
            let force_position_jacobian =
                body_force_position_jacobian_fn(physical_coords, state_velocity);
            if force.iter().any(|value| !value.is_finite())
                || force_position_jacobian
                    .iter()
                    .flatten()
                    .any(|value| !value.is_finite())
            {
                return Err(MeshMotionDesignDerivativeError::NonFiniteBodyForceDerivative);
            }

            let velocity_magnitude = state_velocity
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            let h = (6.0 * determinant_abs).cbrt();
            let tau = super::stabilisation::stabilisation_tau(velocity_magnitude, h, nu, None);
            let tau_lsic = tau * velocity_magnitude * h * 0.5;
            let divergence = (0..3)
                .map(|component| velocity_gradient[component][component])
                .sum::<f64>();

            for local_node in 0..velocity_scalar_dofs {
                for component in 0..3 {
                    let global_row = velocity_dofs[3 * local_node + component];
                    let test_value = reference_values[local_node];
                    let test_gradient = gradients[local_node];
                    let streamline_test = dot3(state_velocity, test_gradient);
                    let advected_component = dot3(state_velocity, velocity_gradient[component]);
                    let divergence_test = test_gradient[component];
                    let viscous_residual = dot3(velocity_gradient[component], test_gradient);
                    let mut row_residual = rho * advected_component * test_value
                        + viscous_mu * viscous_residual
                        - test_value * force[component]
                        - pressure_value * divergence_test;
                    if config.use_residual_stabilisation {
                        row_residual += rho * tau_lsic * divergence_test * divergence;
                        row_residual += rho * tau * streamline_test * advected_component;
                        row_residual += tau * streamline_test * pressure_gradient[component];
                        row_residual -= tau * streamline_test * force[component];
                    }

                    for design_index in 0..DESIGN_DIM {
                        let mut nodal_motion = [[0.0; 3]; 4];
                        for local_node_motion in 0..4 {
                            nodal_motion[local_node_motion] = nodal_coordinate_jacobian
                                [element.nodes[local_node_motion]][design_index];
                        }
                        let mut d_coordinates = [0.0; 3];
                        for local_node_motion in 0..4 {
                            for direction in 0..3 {
                                d_coordinates[direction] += pressure_reference_values
                                    [local_node_motion]
                                    * nodal_motion[local_node_motion][direction];
                            }
                        }
                        let mut d_jacobian = [[0.0; 3]; 3];
                        for row in 0..3 {
                            for column in 0..3 {
                                d_jacobian[row][column] =
                                    nodal_motion[column + 1][row] - nodal_motion[0][row];
                            }
                        }
                        let d_determinant = determinant
                            * (0..3)
                                .map(|index| {
                                    (0..3)
                                        .map(|other| {
                                            inverse[index][other] * d_jacobian[other][index]
                                        })
                                        .sum::<f64>()
                                })
                                .sum::<f64>();
                        let d_determinant_abs = determinant.signum() * d_determinant;
                        let d_weight = point.weight * d_determinant_abs;
                        let d_inverse_transpose = scale3(
                            mat_mul3(
                                mat_mul3(inverse_transpose, transpose3(d_jacobian)),
                                inverse_transpose,
                            ),
                            -1.0,
                        );
                        let mut d_gradients = vec![[0.0; 3]; velocity_scalar_dofs];
                        for local_node_motion in 0..velocity_scalar_dofs {
                            d_gradients[local_node_motion] = mat_vec3(
                                d_inverse_transpose,
                                reference_gradients[local_node_motion],
                            );
                        }

                        let mut d_pressure_gradients = [[0.0; 3]; 4];
                        for local_node_motion in 0..4 {
                            d_pressure_gradients[local_node_motion] = mat_vec3(
                                d_inverse_transpose,
                                pressure_reference_gradients[local_node_motion],
                            );
                        }
                        let mut d_velocity_gradient = [[0.0; 3]; 3];
                        let mut d_pressure_gradient = [0.0; 3];
                        for local_node_motion in 0..4 {
                            for direction in 0..3 {
                                d_pressure_gradient[direction] += local_pressure[local_node_motion]
                                    * d_pressure_gradients[local_node_motion][direction];
                            }
                            for component_motion in 0..3 {
                                for direction in 0..3 {
                                    d_velocity_gradient[component_motion][direction] +=
                                        local_velocity[local_node_motion][component_motion]
                                            * d_gradients[local_node_motion][direction];
                                }
                            }
                        }
                        for local_node_motion in 4..velocity_scalar_dofs {
                            for component_motion in 0..3 {
                                for direction in 0..3 {
                                    d_velocity_gradient[component_motion][direction] +=
                                        local_velocity[local_node_motion][component_motion]
                                            * d_gradients[local_node_motion][direction];
                                }
                            }
                        }

                        let d_force = mat_vec3(force_position_jacobian, d_coordinates);
                        let d_test_gradient = d_gradients[local_node];
                        let d_divergence = (0..3)
                            .map(|direction| d_velocity_gradient[direction][direction])
                            .sum::<f64>();
                        let d_streamline_test = dot3(state_velocity, d_test_gradient);
                        let d_advected_component =
                            dot3(state_velocity, d_velocity_gradient[component]);
                        let d_divergence_test = d_test_gradient[component];
                        let d_viscous_residual =
                            dot3(d_velocity_gradient[component], test_gradient)
                                + dot3(velocity_gradient[component], d_test_gradient);
                        let d_h = h * d_determinant_abs / (3.0 * determinant_abs);
                        let d_tau = tau_geometry_derivative(tau, h, velocity_magnitude, d_h, nu);
                        let d_tau_lsic = 0.5 * velocity_magnitude * (d_tau * h + tau * d_h);
                        let mut d_row_residual = rho * d_advected_component * test_value
                            + viscous_mu * d_viscous_residual
                            - test_value * d_force[component]
                            - pressure_value * d_divergence_test;
                        if config.use_residual_stabilisation {
                            d_row_residual += rho
                                * (d_tau_lsic * divergence_test * divergence
                                    + tau_lsic * d_divergence_test * divergence
                                    + tau_lsic * divergence_test * d_divergence);
                            d_row_residual += rho
                                * (d_tau * streamline_test * advected_component
                                    + tau * d_streamline_test * advected_component
                                    + tau * streamline_test * d_advected_component);
                            d_row_residual +=
                                d_tau * streamline_test * pressure_gradient[component]
                                    + tau * d_streamline_test * pressure_gradient[component]
                                    + tau * streamline_test * d_pressure_gradient[component];
                            d_row_residual -= d_tau * streamline_test * force[component]
                                + tau * d_streamline_test * force[component]
                                + tau * streamline_test * d_force[component];
                        }
                        residual_design[global_row][design_index] +=
                            d_row_residual * weight + row_residual * d_weight;

                        // The pressure rows contain incompressibility plus the
                        // PSPG residual.  Evaluate them once per design
                        // direction (the surrounding velocity-row loop is
                        // only used to share the geometry differential).
                        if local_node == 0 && component == 0 {
                            for pressure_node in 0..4 {
                                let pressure_row = pressure_dofs[pressure_node];
                                let pressure_test_value = pressure_reference_values[pressure_node];
                                let pressure_test_gradient = pressure_gradients[pressure_node];
                                let grad_dot_advected = (0..3)
                                    .map(|direction| {
                                        pressure_test_gradient[direction]
                                            * dot3(state_velocity, velocity_gradient[direction])
                                    })
                                    .sum::<f64>();
                                let grad_dot_force = dot3(pressure_test_gradient, force);
                                let mut pressure_residual = -pressure_test_value * divergence;
                                if config.use_residual_stabilisation {
                                    pressure_residual += -tau * grad_dot_advected
                                        - tau / rho.max(f64::EPSILON)
                                            * dot3(pressure_test_gradient, pressure_gradient)
                                        + tau / rho.max(f64::EPSILON) * grad_dot_force;
                                }

                                let d_pressure_test_gradient = d_pressure_gradients[pressure_node];
                                let d_grad_dot_advected = (0..3)
                                    .map(|direction| {
                                        d_pressure_test_gradient[direction]
                                            * dot3(state_velocity, velocity_gradient[direction])
                                            + pressure_test_gradient[direction]
                                                * dot3(
                                                    state_velocity,
                                                    d_velocity_gradient[direction],
                                                )
                                    })
                                    .sum::<f64>();
                                let d_grad_dot_pressure =
                                    dot3(d_pressure_test_gradient, pressure_gradient)
                                        + dot3(pressure_test_gradient, d_pressure_gradient);
                                let d_grad_dot_force = dot3(d_pressure_test_gradient, force)
                                    + dot3(pressure_test_gradient, d_force);
                                let mut d_pressure_residual = -pressure_test_value * d_divergence;
                                if config.use_residual_stabilisation {
                                    d_pressure_residual += -d_tau * grad_dot_advected
                                        - tau * d_grad_dot_advected
                                        - d_tau / rho.max(f64::EPSILON)
                                            * dot3(pressure_test_gradient, pressure_gradient)
                                        - tau / rho.max(f64::EPSILON) * d_grad_dot_pressure
                                        + d_tau / rho.max(f64::EPSILON) * grad_dot_force
                                        + tau / rho.max(f64::EPSILON) * d_grad_dot_force;
                                }
                                residual_design[velocity_dimension + pressure_row][design_index] +=
                                    d_pressure_residual * weight + pressure_residual * d_weight;
                            }
                        }
                    }
                }
            }
        }
    }

    if residual_design
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(MeshMotionDesignDerivativeError::NonFiniteInput(
            "assembled derivative",
        ));
    }
    Ok(residual_design)
}

fn dot3(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn transpose3(matrix: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [matrix[0][0], matrix[1][0], matrix[2][0]],
        [matrix[0][1], matrix[1][1], matrix[2][1]],
        [matrix[0][2], matrix[1][2], matrix[2][2]],
    ]
}

fn mat_vec3(matrix: [[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    [
        dot3(matrix[0], vector),
        dot3(matrix[1], vector),
        dot3(matrix[2], vector),
    ]
}

fn mat_mul3(left: [[f64; 3]; 3], right: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut result = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            result[row][column] = (0..3)
                .map(|index| left[row][index] * right[index][column])
                .sum();
        }
    }
    result
}

fn scale3(matrix: [[f64; 3]; 3], scale: f64) -> [[f64; 3]; 3] {
    let mut result = matrix;
    for row in &mut result {
        for value in row {
            *value *= scale;
        }
    }
    result
}

fn inverse3(matrix: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let determinant = det3(matrix);
    if !determinant.is_finite() || determinant.abs() <= f64::EPSILON {
        return None;
    }
    let inverse = [
        [
            matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1],
            matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2],
            matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1],
        ],
        [
            matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2],
            matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0],
            matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2],
        ],
        [
            matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0],
            matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1],
            matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0],
        ],
    ];
    Some(scale3(inverse, 1.0 / determinant))
}

fn tau_geometry_derivative(tau: f64, h: f64, velocity_magnitude: f64, d_h: f64, nu: f64) -> f64 {
    if tau == 0.0 || h <= f64::EPSILON {
        return 0.0;
    }
    let inverse_tau_squared =
        4.0 * velocity_magnitude * velocity_magnitude / (h * h) + 36.0 * nu * nu / (h * h * h * h);
    if inverse_tau_squared <= 1.0e-30 {
        0.0
    } else {
        tau.powi(3)
            * (4.0 * velocity_magnitude * velocity_magnitude / h.powi(3)
                + 72.0 * nu * nu / h.powi(5))
            * d_h
    }
}

fn apply_dirichlet_to_csr(
    csr: &mut sonium_solvers::CsrMatrix<f64>,
    rhs: &mut [f64],
    is_bc: &[bool],
    bc_values: &[f64],
) {
    assert_eq!(rhs.len(), is_bc.len());
    assert_eq!(rhs.len(), bc_values.len());
    for row in 0..rhs.len() {
        let row_start = csr.row_ptrs[row];
        let row_end = csr.row_ptrs[row + 1];
        if is_bc[row] {
            for j in row_start..row_end {
                csr.values[j] = if csr.col_indices[j] == row { 1.0 } else { 0.0 };
            }
            rhs[row] = bc_values[row];
        } else {
            for j in row_start..row_end {
                let col = csr.col_indices[j];
                if is_bc[col] {
                    rhs[row] -= csr.values[j] * bc_values[col];
                    csr.values[j] = 0.0;
                }
            }
        }
    }
}

/// Picard-iterate velocity evaluator backed by the BVH topology.
///
/// Thin adapter over [`super::workspace::ElementVelocityTopology`] so the
/// Oseen/stabilisation/body-force assembly closures share one evaluator
/// with the BDF path. Queries resolve in O(log n) instead of the
/// historical per-query linear mesh scan; a query outside the mesh fails
/// closed (assembly quadrature points are always interior). Fully owned
/// (`Arc` topology plus a per-iteration value snapshot) because the
/// assembly closures require `'static` data.
struct PicardVelocityField {
    topology: std::sync::Arc<super::workspace::ElementVelocityTopology>,
    values: Vec<f64>,
    dim: usize,
}

impl PicardVelocityField {
    fn velocity_at(&self, coords: [f64; 3]) -> [f64; 3] {
        self.topology
            .field(&self.values)
            .velocity_at(coords)
            .expect("Picard quadrature point was not located in its velocity mesh")
    }

    fn velocity_components_at(&self, coords: [f64; 3]) -> Vec<f64> {
        self.velocity_at(coords)[..self.dim].to_vec()
    }

    fn velocity_magnitude_at(&self, coords: [f64; 3]) -> f64 {
        let v = self.velocity_at(coords);
        (0..self.dim).map(|d| v[d] * v[d]).sum::<f64>().sqrt()
    }
}

fn pressure_constant_mode_is_free(
    gradient: &crate::assembly::TripletMatrix<f64>,
    bc_dofs: &[usize],
) -> bool {
    if gradient.num_cols == 0 {
        return false;
    }
    let mut constrained = vec![false; gradient.num_rows];
    for &dof in bc_dofs {
        assert!(
            dof < gradient.num_rows,
            "velocity boundary DOF out of range"
        );
        constrained[dof] = true;
    }
    let mut constant_force = vec![0.0; gradient.num_rows];
    let mut row_scale = vec![0.0; gradient.num_rows];
    for i in 0..gradient.nnz() {
        constant_force[gradient.rows[i]] += gradient.values[i];
        row_scale[gradient.rows[i]] += gradient.values[i].abs();
    }
    // Partition-of-unity cancellation accumulates many element triplets.
    // This relative roundoff bound detects a null mode, not a flow tolerance.
    constant_force
        .iter()
        .zip(row_scale)
        .zip(constrained)
        .all(|((&force, scale), fixed)| fixed || force.abs() <= 1e-12 * scale)
}

fn pressure_mass_schur_csr<PSpace>(
    p_space: &PSpace,
    mesh: &crate::mesh::Mesh,
    mu_eff: f64,
    pressure_pin: Option<usize>,
) -> sonium_solvers::CsrMatrix<f64>
where
    PSpace: crate::spaces::FunctionSpace,
{
    let mass = crate::assembly::assemble_bilinear(&MassForm, p_space, p_space, mesh, 2);
    let scale = mu_eff.max(f64::EPSILON).recip();
    let mut schur =
        crate::assembly::TripletMatrix::with_capacity(mass.num_rows, mass.num_cols, mass.nnz());
    for i in 0..mass.nnz() {
        schur.push(mass.rows[i], mass.cols[i], mass.values[i] * scale);
    }

    let mut csr = schur.to_csr();
    if let Some(pin) = pressure_pin {
        apply_local_dirichlet_to_csr(&mut csr, pin);
    }
    csr
}

fn apply_local_dirichlet_to_csr(csr: &mut sonium_solvers::CsrMatrix<f64>, dof: usize) {
    if dof >= csr.num_rows {
        return;
    }
    for row in 0..csr.num_rows {
        let row_start = csr.row_ptrs[row];
        let row_end = csr.row_ptrs[row + 1];
        if row == dof {
            for j in row_start..row_end {
                csr.values[j] = if csr.col_indices[j] == dof { 1.0 } else { 0.0 };
            }
        } else {
            for j in row_start..row_end {
                if csr.col_indices[j] == dof {
                    csr.values[j] = 0.0;
                }
            }
        }
    }
}

/// Dispatch the per-Picard linear solve onto either the legacy CPU
/// path (in-house BLAS-1 + ILU(0)) or a `sonium_compute::Backend` path
/// (BLAS-1 + SpMV via the trait, point-Jacobi preconditioner). Returns
/// `None` on solver failure.
///
/// CSR row pointers and column indices are converted from the Rust
/// `usize` representation to `i32` on the GPU path, matching cuSPARSE's
/// expected layout. The cost of the CSR upload is `O(nnz)` per Picard
/// iteration; for typical fan-CFD grids (10–100k DOFs, ~10× nnz) this
/// is sub-millisecond and dwarfed by the solve.
///
/// `Cuda` and `Metal` selections fall back to `Cpu` with a warning when
/// the corresponding `sonium-compute` feature is not built.
fn solve_iterative_linear_system(
    config: &NavierStokesConfig,
    csr: &sonium_solvers::CsrMatrix<f64>,
    b: &[f64],
    x0: Option<&[f64]>,
    velocity_block_size: usize,
    pressure_schur: Option<&sonium_solvers::CsrMatrix<f64>>,
) -> Option<(Vec<f64>, LinearSolverBackend, usize)> {
    let resolved = match config.linear_solver_backend {
        LinearSolverBackend::Auto => LinearSolverBackend::resolve_auto(),
        other => other,
    };
    match resolved {
        LinearSolverBackend::Auto => unreachable!("Auto resolved above"),
        LinearSolverBackend::Cpu => {
            solve_iterative_cpu(config, csr, b, x0, velocity_block_size, pressure_schur)
                .map(|(solution, _, iterations)| (solution, LinearSolverBackend::Cpu, iterations))
        }
        LinearSolverBackend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                let backend = match sonium_compute::cuda::CudaBackend::new_default() {
                    Ok(b) => b,
                    Err(err) => {
                        eprintln!(
                            "WARNING: CUDA backend init failed ({err:?}); falling back to CPU"
                        );
                        return solve_iterative_cpu(
                            config,
                            csr,
                            b,
                            x0,
                            velocity_block_size,
                            pressure_schur,
                        )
                        .map(|(solution, _, iterations)| {
                            (solution, LinearSolverBackend::Cpu, iterations)
                        });
                    }
                };
                solve_iterative_via_backend(
                    &backend,
                    config,
                    csr,
                    b,
                    x0,
                    velocity_block_size,
                    pressure_schur,
                )
                .map(|(solution, used_accelerator, iterations)| {
                    (
                        solution,
                        if used_accelerator {
                            LinearSolverBackend::Cuda
                        } else {
                            LinearSolverBackend::Cpu
                        },
                        iterations,
                    )
                })
            }
            #[cfg(not(feature = "cuda"))]
            {
                eprintln!(
                    "WARNING: requested CUDA backend but build does not include the `cuda` feature; falling back to CPU"
                );
                solve_iterative_cpu(config, csr, b, x0, velocity_block_size, pressure_schur).map(
                    |(solution, _, iterations)| (solution, LinearSolverBackend::Cpu, iterations),
                )
            }
        }
        LinearSolverBackend::Metal => {
            #[cfg(feature = "metal")]
            {
                let backend = match sonium_compute::metal::MetalBackend::new() {
                    Ok(b) => b,
                    Err(err) => {
                        eprintln!(
                            "WARNING: Metal backend init failed ({err:?}); falling back to CPU"
                        );
                        return solve_iterative_cpu(
                            config,
                            csr,
                            b,
                            x0,
                            velocity_block_size,
                            pressure_schur,
                        )
                        .map(|(solution, _, iterations)| {
                            (solution, LinearSolverBackend::Cpu, iterations)
                        });
                    }
                };
                solve_iterative_via_backend(
                    &backend,
                    config,
                    csr,
                    b,
                    x0,
                    velocity_block_size,
                    pressure_schur,
                )
                .map(|(solution, used_accelerator, iterations)| {
                    (
                        solution,
                        if used_accelerator {
                            LinearSolverBackend::Metal
                        } else {
                            LinearSolverBackend::Cpu
                        },
                        iterations,
                    )
                })
            }
            #[cfg(not(feature = "metal"))]
            {
                eprintln!(
                    "WARNING: requested Metal backend but build doesn't include `metal` feature; falling back to CPU"
                );
                solve_iterative_cpu(config, csr, b, x0, velocity_block_size, pressure_schur).map(
                    |(solution, _, iterations)| (solution, LinearSolverBackend::Cpu, iterations),
                )
            }
        }
    }
}

/// Legacy CPU iterative solve — exactly the code path that lived inline
/// in the Picard loop before the backend dispatch was added. Behaviour
/// preserved bit-for-bit (modulo the function-call boundary).
fn solve_iterative_cpu(
    config: &NavierStokesConfig,
    csr: &sonium_solvers::CsrMatrix<f64>,
    b: &[f64],
    x0: Option<&[f64]>,
    velocity_block_size: usize,
    pressure_schur: Option<&sonium_solvers::CsrMatrix<f64>>,
) -> Option<(Vec<f64>, bool, usize)> {
    // Validate the CSR structure once per solve: every Krylov iteration
    // applies the operator, and the shared borrow freezes the storage so
    // applies skip the O(nnz) re-validation.
    let operator = sonium_solvers::ValidatedCsr::new(csr);
    match config.linear_solver_cpu_preconditioner {
        LinearSolverCpuPreconditionerKind::Ilu => {
            let solve_config = sonium_solvers::GmresConfig {
                max_iterations: config.linear_solver_max_iterations.max(1),
                tolerance: config.linear_solver_tolerance,
                ..Default::default()
            };
            let setup_start = std::time::Instant::now();
            let preconditioner = sonium_solvers::IluPreconditioner::from_csr(csr);
            let setup_elapsed = setup_start.elapsed();
            let gmres_start = std::time::Instant::now();
            let result = sonium_solvers::gmres_preconditioned_with_guess(
                &operator,
                &preconditioner,
                b,
                x0,
                &solve_config,
            );
            let gmres_elapsed = gmres_start.elapsed();
            if std::env::var("SONIUM_NS_PROFILE").as_deref() == Ok("1") {
                eprintln!(
                    "NS_PROFILE linear ilu_setup_ms={:.1} gmres_ms={:.1} iters={}",
                    setup_elapsed.as_secs_f64() * 1000.0,
                    gmres_elapsed.as_secs_f64() * 1000.0,
                    result.iterations,
                );
            }
            let iterations = result.iterations;
            finite_converged_solution(result.x, result.converged)
                .map(|solution| (solution, false, iterations))
        }
        LinearSolverCpuPreconditionerKind::JacobiSchur
        | LinearSolverCpuPreconditionerKind::IlutSchur
        | LinearSolverCpuPreconditionerKind::IluSchur => {
            let pressure_schur = pressure_schur?;
            let block_config = match config.linear_solver_cpu_preconditioner {
                LinearSolverCpuPreconditionerKind::JacobiSchur => {
                    sonium_solvers::BlockPreconditionerConfig::Diagonal
                }
                LinearSolverCpuPreconditionerKind::IlutSchur => {
                    sonium_solvers::BlockPreconditionerConfig::Ilut(
                        sonium_solvers::IlutConfig::new(1.0e-3, 2, 1.0e-12),
                    )
                }
                LinearSolverCpuPreconditionerKind::IluSchur => {
                    sonium_solvers::BlockPreconditionerConfig::Ilu
                }
                LinearSolverCpuPreconditionerKind::Ilu => unreachable!(),
            };
            // The external Schur block is Laplacian-like, so it always
            // gets an ILUT subsolver (same drop/fill/pivot values as the
            // BDF cached path); point Jacobi on the Schur block leaves
            // an h^{-2} spread that costs tens of thousands of
            // iterations on fine meshes. If ILUT setup rejects the
            // Schur block, fall back to Jacobi rather than failing.
            let schur_ilut = sonium_solvers::BlockPreconditionerConfig::Ilut(
                sonium_solvers::IlutConfig::new(1.0e-3, 2, 1.0e-12),
            );
            let sp_config = sonium_solvers::SaddlePointConfig {
                block1_size: velocity_block_size,
                block1_config: block_config,
                block2_config: schur_ilut,
                schur_strategy: sonium_solvers::SchurStrategy::External,
                upper_triangular: true,
            };
            let mut fallback_config = sp_config.clone();
            fallback_config.block2_config = sonium_solvers::BlockPreconditionerConfig::Diagonal;
            let preconditioner =
                sonium_solvers::SaddlePointPreconditioner::<f64>::try_from_matrix_with_schur(
                    csr,
                    pressure_schur,
                    sp_config,
                )
                .or_else(|_| {
                    sonium_solvers::SaddlePointPreconditioner::<f64>::try_from_matrix_with_schur(
                        csr,
                        pressure_schur,
                        fallback_config,
                    )
                })
                .ok()?;
            let solve_config = sonium_solvers::FgmresConfig {
                max_iterations: config.linear_solver_max_iterations.max(1),
                restart: 50,
                tolerance: config.linear_solver_tolerance,
                print_interval: 0,
                exact_residual_check_interval: 0,
                convergence_check_interval: 1,
            };
            let result = sonium_solvers::fgmres_preconditioned(
                &operator,
                &preconditioner,
                b,
                x0,
                &solve_config,
            );
            let iterations = result.iterations;
            finite_converged_solution(result.x, result.converged)
                .map(|solution| (solution, false, iterations))
        }
    }
}

fn finite_converged_solution(x: Vec<f64>, converged: bool) -> Option<Vec<f64>> {
    (converged && x.iter().all(|value| value.is_finite())).then_some(x)
}

/// Backend-aware iterative solve. Uploads the CSR + RHS to the device,
/// runs `gmres_backend_preconditioned` with point-Jacobi, downloads
/// the solution. Generic over `B: Backend` so the same body services
/// CPU, CUDA, and Metal.
#[cfg_attr(
    not(any(feature = "cuda", feature = "metal")),
    allow(dead_code, reason = "only callable from cuda/metal-gated arms")
)]
fn solve_iterative_via_backend<B: sonium_compute::Backend>(
    backend: &B,
    config: &NavierStokesConfig,
    csr: &sonium_solvers::CsrMatrix<f64>,
    b: &[f64],
    x0: Option<&[f64]>,
    velocity_block_size: usize,
    pressure_schur: Option<&sonium_solvers::CsrMatrix<f64>>,
) -> Option<(Vec<f64>, bool, usize)> {
    use sonium_compute::gmres::{
        BackendCsr, BackendPreconditioner, GmresBackendConfig, gmres_backend_preconditioned,
    };
    use sonium_compute::gmres_ir::{BackendCsrF32, GmresIrConfig, gmres_ir_backend_preconditioned};
    use sonium_compute::preconditioner::{
        PressureSchurBackendKind, SaddlePointJacobiSchurBackend,
        SaddlePointJacobiSchurBackendConfig, embed_pressure_schur_with_velocity_identity,
    };
    use sonium_compute::{
        ChebyshevBackend, ChebyshevBackendF32, HostCsrRef, PolynomialJacobiBackend,
        PolynomialJacobiBackendF32,
    };
    use sonium_solvers::preconditioners::{AmgConfig, AmgSmoother, BackendAmgPreconditioner};

    let n = csr.num_rows;
    if n != csr.num_cols {
        eprintln!(
            "WARNING: backend GMRES requires square CSR (got {}×{}); falling back to CPU",
            csr.num_rows, csr.num_cols
        );
        return solve_iterative_cpu(config, csr, b, x0, velocity_block_size, pressure_schur);
    }

    // Convert usize indices to i32 (the cuSPARSE / Backend convention).
    // On 64-bit hosts a usize index could overflow i32 in pathological
    // cases (>2 G entries); guard explicitly and fall back to CPU.
    let row_ptr_host: Vec<i32> = match csr
        .row_ptrs
        .iter()
        .map(|&v| i32::try_from(v))
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(v) => v,
        Err(_) => {
            eprintln!(
                "WARNING: CSR row pointers overflow i32 ({} entries); falling back to CPU",
                csr.row_ptrs.last().copied().unwrap_or(0)
            );
            return solve_iterative_cpu(config, csr, b, x0, velocity_block_size, pressure_schur);
        }
    };
    let col_idx_host: Vec<i32> = match csr
        .col_indices
        .iter()
        .map(|&v| i32::try_from(v))
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(v) => v,
        Err(_) => {
            eprintln!("WARNING: CSR column indices overflow i32; falling back to CPU");
            return solve_iterative_cpu(config, csr, b, x0, velocity_block_size, pressure_schur);
        }
    };

    // Upload CSR matrix and Jacobi preconditioner.
    let device_csr = match BackendCsr::try_from_host(
        backend,
        n,
        csr.num_cols,
        &row_ptr_host,
        &col_idx_host,
        &csr.values,
    ) {
        Ok(csr) => csr,
        Err(err) => {
            eprintln!("WARNING: backend CSR upload failed ({err}); falling back to CPU");
            return solve_iterative_cpu(config, csr, b, x0, velocity_block_size, pressure_schur);
        }
    };
    // Upload RHS.
    let mut b_dev = match backend.raw_alloc::<f64>(n) {
        Ok(buf) => buf,
        Err(err) => {
            eprintln!("WARNING: backend RHS alloc failed ({err:?}); falling back to CPU");
            return solve_iterative_cpu(config, csr, b, x0, velocity_block_size, pressure_schur);
        }
    };
    if let Err(err) = backend.try_copy_h2d(b, &mut b_dev) {
        eprintln!("WARNING: backend RHS upload failed ({err}); falling back to CPU");
        return solve_iterative_cpu(config, csr, b, x0, velocity_block_size, pressure_schur);
    }

    let x0_dev = match x0 {
        Some(x) => match backend.raw_alloc::<f64>(n) {
            Ok(mut buf) => {
                if let Err(err) = backend.try_copy_h2d(x, &mut buf) {
                    eprintln!(
                        "WARNING: backend initial-guess upload failed ({err}); falling back to CPU"
                    );
                    return solve_iterative_cpu(
                        config,
                        csr,
                        b,
                        x0,
                        velocity_block_size,
                        pressure_schur,
                    )
                    .map(|(solution, _, iterations)| (solution, false, iterations));
                }
                Some(buf)
            }
            Err(err) => {
                eprintln!(
                    "WARNING: backend initial-guess allocation failed ({err:?}); falling back to CPU"
                );
                return solve_iterative_cpu(
                    config,
                    csr,
                    b,
                    x0,
                    velocity_block_size,
                    pressure_schur,
                )
                .map(|(solution, _, iterations)| (solution, false, iterations));
            }
        },
        None => None,
    };

    let mut out_dev = match backend.raw_alloc::<f64>(n) {
        Ok(buf) => buf,
        Err(err) => {
            eprintln!("WARNING: backend output allocation failed ({err:?}); falling back to CPU");
            return solve_iterative_cpu(config, csr, b, x0, velocity_block_size, pressure_schur);
        }
    };

    let cfg = GmresBackendConfig {
        max_iterations: config.linear_solver_max_iterations.max(1),
        tolerance: config.linear_solver_tolerance,
    };

    // Top-level branch on working precision. F64 is the legacy
    // backend GMRES; MixedF32F64 routes through FGMRES-IR (Carson-Higham
    // 2018) — outer FGMRES f64, inner Krylov f32, native MSL kernels on
    // Metal.
    let (converged, total_iterations, final_residual) = match config.linear_solver_kind {
        LinearSolverKind::F64 => {
            let status_result = match config.linear_solver_backend_preconditioner {
                BackendPreconditionerKind::PolynomialJacobi => {
                    let precond = match PolynomialJacobiBackend::try_from_csr_host(
                        backend,
                        &device_csr,
                        &row_ptr_host,
                        &col_idx_host,
                        &csr.values,
                        config.linear_solver_polynomial_degree.max(1),
                        config.linear_solver_polynomial_damping,
                    ) {
                        Ok(preconditioner) => preconditioner,
                        Err(err) => {
                            eprintln!(
                                "WARNING: backend polynomial preconditioner setup failed ({err}); falling back to CPU"
                            );
                            return solve_iterative_cpu(
                                config,
                                csr,
                                b,
                                x0,
                                velocity_block_size,
                                pressure_schur,
                            );
                        }
                    };
                    gmres_backend_preconditioned(
                        backend,
                        &device_csr,
                        &precond,
                        &b_dev,
                        x0_dev.as_ref(),
                        &mut out_dev,
                        &cfg,
                    )
                }
                BackendPreconditionerKind::Chebyshev => {
                    let precond = match ChebyshevBackend::try_from_csr_host(
                        backend,
                        &device_csr,
                        &row_ptr_host,
                        &col_idx_host,
                        &csr.values,
                        config.linear_solver_chebyshev_degree.max(1),
                        config.linear_solver_chebyshev_eigenvalue_ratio.max(1.0),
                    ) {
                        Ok(preconditioner) => preconditioner,
                        Err(err) => {
                            eprintln!(
                                "WARNING: backend Chebyshev preconditioner setup failed ({err}); falling back to CPU"
                            );
                            return solve_iterative_cpu(
                                config,
                                csr,
                                b,
                                x0,
                                velocity_block_size,
                                pressure_schur,
                            );
                        }
                    };
                    gmres_backend_preconditioned(
                        backend,
                        &device_csr,
                        &precond,
                        &b_dev,
                        x0_dev.as_ref(),
                        &mut out_dev,
                        &cfg,
                    )
                }
                BackendPreconditionerKind::JacobiSchur => {
                    let Some(pressure_schur) = pressure_schur else {
                        eprintln!(
                            "WARNING: backend Jacobi/Schur requested without Schur matrix; falling back to CPU"
                        );
                        return solve_iterative_cpu(
                            config,
                            csr,
                            b,
                            x0,
                            velocity_block_size,
                            pressure_schur,
                        );
                    };
                    let pressure_schur_ref = HostCsrRef::new(
                        pressure_schur.num_rows,
                        pressure_schur.num_cols,
                        &pressure_schur.row_ptrs,
                        &pressure_schur.col_indices,
                        &pressure_schur.values,
                    );
                    let mut pressure_amg_holder = None;
                    let pressure_kind = match config.linear_solver_schur_preconditioner {
                        BackendPressureSchurPreconditionerKind::Polynomial => {
                            PressureSchurBackendKind::Polynomial
                        }
                        BackendPressureSchurPreconditionerKind::Chebyshev => {
                            PressureSchurBackendKind::Chebyshev
                        }
                        BackendPressureSchurPreconditionerKind::Amg => {
                            let (row_ptr, col_idx, values) =
                                embed_pressure_schur_with_velocity_identity(
                                    n,
                                    velocity_block_size,
                                    pressure_schur_ref,
                                );
                            let matrix = sonium_solvers::CsrMatrix::from_raw_parts(
                                n, n, row_ptr, col_idx, values,
                            );
                            pressure_amg_holder = Some(BackendAmgPreconditioner::from_csr(
                                backend,
                                &matrix,
                                AmgConfig {
                                    smoother: AmgSmoother::Chebyshev,
                                    num_pre_smooth: 2,
                                    num_post_smooth: 1,
                                    ..AmgConfig::default()
                                },
                            ));
                            PressureSchurBackendKind::External
                        }
                    };
                    let pressure_preconditioner = pressure_amg_holder
                        .as_ref()
                        .map(|p| p as &dyn BackendPreconditioner<B, f64>);
                    let precond = match SaddlePointJacobiSchurBackend::try_from_csr_host_parts_with_pressure_preconditioner(
                        backend,
                        &device_csr,
                        &row_ptr_host,
                        &col_idx_host,
                        &csr.values,
                        pressure_schur_ref,
                        SaddlePointJacobiSchurBackendConfig {
                            velocity_block_size,
                            velocity_degree: config.linear_solver_polynomial_degree.max(1),
                            velocity_damping: config.linear_solver_polynomial_damping,
                            pressure_degree: config.linear_solver_schur_polynomial_degree.max(1),
                            pressure_damping: config.linear_solver_schur_polynomial_damping,
                            pressure_kind,
                            pressure_chebyshev_degree: config
                                .linear_solver_schur_chebyshev_degree
                                .max(1),
                            pressure_chebyshev_eigenvalue_ratio: config
                                .linear_solver_schur_chebyshev_eigenvalue_ratio
                                .max(1.0),
                        },
                        pressure_preconditioner,
                    ) {
                        Ok(preconditioner) => preconditioner,
                        Err(err) => {
                            eprintln!(
                                "WARNING: saddle-point backend preconditioner upload failed ({err}); falling back to CPU"
                            );
                            return solve_iterative_cpu(
                                config,
                                csr,
                                b,
                                x0,
                                velocity_block_size,
                                Some(pressure_schur),
                            );
                        }
                    };
                    gmres_backend_preconditioned(
                        backend,
                        &device_csr,
                        &precond,
                        &b_dev,
                        x0_dev.as_ref(),
                        &mut out_dev,
                        &cfg,
                    )
                }
            };
            let status = match status_result {
                Ok(s) => s,
                Err(err) => {
                    eprintln!("WARNING: backend GMRES failed ({err}); falling back to CPU");
                    return solve_iterative_cpu(
                        config,
                        csr,
                        b,
                        x0,
                        velocity_block_size,
                        pressure_schur,
                    );
                }
            };
            (status.converged, status.iterations, status.final_residual)
        }
        LinearSolverKind::MixedF32F64 => {
            // Build the f32 cast of A once — `nnz` floats, fast.
            let a_f32 = match BackendCsrF32::try_from_host_cast(
                backend,
                csr.num_rows,
                csr.num_cols,
                &row_ptr_host,
                &col_idx_host,
                &csr.values,
            ) {
                Ok(csr) => csr,
                Err(err) => {
                    eprintln!(
                        "WARNING: mixed-precision backend CSR upload failed ({err}); falling back to CPU"
                    );
                    return solve_iterative_cpu(
                        config,
                        csr,
                        b,
                        x0,
                        velocity_block_size,
                        pressure_schur,
                    );
                }
            };
            let values_f32: Vec<f32> = csr.values.iter().map(|&v| v as f32).collect();
            let ir_cfg = GmresIrConfig {
                max_outer_iterations: config.linear_solver_max_iterations.clamp(1, 200),
                outer_tolerance: config.linear_solver_tolerance,
                max_inner_iterations: config.linear_solver_ir_inner_max_iterations.max(1),
                inner_tolerance: config.linear_solver_ir_inner_tolerance,
                ..GmresIrConfig::default()
            };
            // Honour the configured preconditioner kind on the f32 inner.
            let status_result = match config.linear_solver_backend_preconditioner {
                BackendPreconditionerKind::PolynomialJacobi => {
                    let pc_f32 = match PolynomialJacobiBackendF32::try_from_csr_host(
                        backend,
                        &a_f32,
                        &row_ptr_host,
                        &col_idx_host,
                        &values_f32,
                        config.linear_solver_polynomial_degree.max(1),
                        config.linear_solver_polynomial_damping as f32,
                    ) {
                        Ok(preconditioner) => preconditioner,
                        Err(err) => {
                            eprintln!(
                                "WARNING: mixed-precision backend polynomial preconditioner setup failed ({err}); falling back to CPU"
                            );
                            return solve_iterative_cpu(
                                config,
                                csr,
                                b,
                                x0,
                                velocity_block_size,
                                pressure_schur,
                            );
                        }
                    };
                    gmres_ir_backend_preconditioned(
                        backend,
                        &device_csr,
                        &a_f32,
                        &pc_f32,
                        &b_dev,
                        x0_dev.as_ref(),
                        &mut out_dev,
                        &ir_cfg,
                    )
                }
                BackendPreconditionerKind::Chebyshev => {
                    let pc_f32 = match ChebyshevBackendF32::try_from_csr_host(
                        backend,
                        &a_f32,
                        &row_ptr_host,
                        &col_idx_host,
                        &values_f32,
                        config.linear_solver_chebyshev_degree.max(1),
                        config.linear_solver_chebyshev_eigenvalue_ratio.max(1.0) as f32,
                    ) {
                        Ok(preconditioner) => preconditioner,
                        Err(err) => {
                            eprintln!(
                                "WARNING: mixed-precision backend Chebyshev preconditioner setup failed ({err}); falling back to CPU"
                            );
                            return solve_iterative_cpu(
                                config,
                                csr,
                                b,
                                x0,
                                velocity_block_size,
                                pressure_schur,
                            );
                        }
                    };
                    gmres_ir_backend_preconditioned(
                        backend,
                        &device_csr,
                        &a_f32,
                        &pc_f32,
                        &b_dev,
                        x0_dev.as_ref(),
                        &mut out_dev,
                        &ir_cfg,
                    )
                }
                BackendPreconditionerKind::JacobiSchur => {
                    let Some(pressure_schur) = pressure_schur else {
                        eprintln!(
                            "WARNING: backend Jacobi/Schur requested without Schur matrix; falling back to CPU"
                        );
                        return solve_iterative_cpu(
                            config,
                            csr,
                            b,
                            x0,
                            velocity_block_size,
                            pressure_schur,
                        );
                    };
                    let pressure_kind = match config.linear_solver_schur_preconditioner {
                        BackendPressureSchurPreconditionerKind::Polynomial => {
                            PressureSchurBackendKind::Polynomial
                        }
                        BackendPressureSchurPreconditionerKind::Chebyshev => {
                            PressureSchurBackendKind::Chebyshev
                        }
                        BackendPressureSchurPreconditionerKind::Amg => {
                            eprintln!(
                                "WARNING: mixed-precision backend Schur/AMG is not available; falling back to CPU"
                            );
                            return solve_iterative_cpu(
                                config,
                                csr,
                                b,
                                x0,
                                velocity_block_size,
                                Some(pressure_schur),
                            );
                        }
                    };
                    let schur_values_f32: Vec<f32> =
                        pressure_schur.values.iter().map(|&v| v as f32).collect();
                    let pressure_schur_ref = HostCsrRef::new(
                        pressure_schur.num_rows,
                        pressure_schur.num_cols,
                        &pressure_schur.row_ptrs,
                        &pressure_schur.col_indices,
                        &schur_values_f32,
                    );
                    let pc_f32 = match SaddlePointJacobiSchurBackend::try_from_csr_host_parts_f32(
                        backend,
                        &a_f32,
                        &row_ptr_host,
                        &col_idx_host,
                        &values_f32,
                        pressure_schur_ref,
                        SaddlePointJacobiSchurBackendConfig {
                            velocity_block_size,
                            velocity_degree: config.linear_solver_polynomial_degree.max(1),
                            velocity_damping: config.linear_solver_polynomial_damping,
                            pressure_degree: config.linear_solver_schur_polynomial_degree.max(1),
                            pressure_damping: config.linear_solver_schur_polynomial_damping,
                            pressure_kind,
                            pressure_chebyshev_degree: config
                                .linear_solver_schur_chebyshev_degree
                                .max(1),
                            pressure_chebyshev_eigenvalue_ratio: config
                                .linear_solver_schur_chebyshev_eigenvalue_ratio
                                .max(1.0),
                        },
                    ) {
                        Ok(preconditioner) => preconditioner,
                        Err(err) => {
                            eprintln!(
                                "WARNING: mixed-precision backend saddle-point preconditioner setup failed ({err}); falling back to CPU"
                            );
                            return solve_iterative_cpu(
                                config,
                                csr,
                                b,
                                x0,
                                velocity_block_size,
                                Some(pressure_schur),
                            );
                        }
                    };
                    gmres_ir_backend_preconditioned(
                        backend,
                        &device_csr,
                        &a_f32,
                        &pc_f32,
                        &b_dev,
                        x0_dev.as_ref(),
                        &mut out_dev,
                        &ir_cfg,
                    )
                }
            };
            let status = match status_result {
                Ok(s) => s,
                Err(err) => {
                    eprintln!("WARNING: GMRES-IR failed ({err}); falling back to CPU");
                    return solve_iterative_cpu(
                        config,
                        csr,
                        b,
                        x0,
                        velocity_block_size,
                        pressure_schur,
                    );
                }
            };
            (
                status.converged,
                status.outer_iterations,
                status.final_outer_residual,
            )
        }
    };

    if !converged {
        eprintln!(
            "WARNING: backend solver did not converge ({} iterations, residual {:.3e}); falling back to CPU",
            total_iterations, final_residual
        );
        return solve_iterative_cpu(config, csr, b, x0, velocity_block_size, pressure_schur);
    }

    let mut x_host = vec![0.0_f64; n];
    if let Err(err) = backend.try_copy_d2h(&out_dev, &mut x_host) {
        eprintln!("WARNING: backend solution download failed ({err}); falling back to CPU");
        return solve_iterative_cpu(config, csr, b, x0, velocity_block_size, pressure_schur);
    }
    if !x_host.iter().all(|v| v.is_finite()) {
        return None;
    }
    Some((x_host, true, total_iterations))
}

fn det3(matrix: [[f64; 3]; 3]) -> f64 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}

// ---------------------------------------------------------------------------
// BDF time step (one step of unsteady NS)
// ---------------------------------------------------------------------------

/// Perform one BDF time step of the unsteady Navier-Stokes equations,
/// adding pressure-traction natural boundary conditions to the RHS.
///
/// Assembles and solves:
/// ```text
/// [ρ a₀/Δt M + μK + C(u*), B] [u^{n+1}]   [f + ρ/Δt M(-a₁u^n - a₂u^{n-1}) + t]
/// [B^T,                     0] [p^{n+1}] = [0                                ]
/// ```
///
/// where `u* = extrapolated velocity`, `a₀,a₁,a₂` are BDF coefficients, and
/// `t` is the pressure-traction load vector assembled from `tractions`
/// (empty slice = pure Dirichlet/Neumann-free step, identical to
/// [`solve_bdf_step`]).
///
/// Builds a [`super::workspace::BdfWorkspace`] internally and delegates to
/// [`solve_bdf_step_cached`]; callers stepping in a loop should build the
/// workspace once and call the cached variant directly.
///
/// Updates `state` in-place with the new velocity and pressure.
#[allow(clippy::too_many_arguments)]
pub fn solve_bdf_step_with_traction<VSpace, PSpace, BcFn>(
    mesh: &crate::mesh::Mesh,
    v_space: &VSpace,
    p_space: &PSpace,
    config: &NavierStokesConfig,
    bdf: &super::time_stepping::BdfConfig,
    state: &mut super::time_stepping::BdfState,
    bc_velocity_fn: &BcFn,
    bc_dofs: &[usize],
    tractions: &[super::traction::PressureTraction],
) -> bool
where
    VSpace: crate::spaces::FunctionSpace,
    PSpace: crate::spaces::FunctionSpace,
    BcFn: Fn(f64, f64, f64) -> Vec<f64>,
{
    let workspace = super::workspace::BdfWorkspace::new(mesh, v_space, p_space);
    solve_bdf_step_cached(
        mesh,
        v_space,
        p_space,
        config,
        bdf,
        state,
        bc_velocity_fn,
        bc_dofs,
        tractions,
        &workspace,
    )
}

struct BdfBodyForceJacobianForm<F> {
    jacobian: F,
}

impl<F> BilinearForm<f64> for BdfBodyForceJacobianForm<F>
where
    F: Fn([f64; 3]) -> [[f64; 3]; 3] + Send + Sync,
{
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let dimension = test.spatial_dim;
        let force_jacobian = (self.jacobian)(test.physical_coords);
        let weighted_jacobian = weight * test.det_j;
        for row in 0..test.n_dofs {
            let output = row % dimension;
            let test_value = test.vector_value(row, output);
            for column in 0..trial.n_dofs {
                let input = column % dimension;
                let trial_value = trial.vector_value(column, input);
                // The residual is A(u)u - f(u), hence the force tangent is
                // assembled with a negative sign on the system matrix.
                local_matrix.add(
                    row,
                    column,
                    -force_jacobian[output][input] * test_value * trial_value * weighted_jacobian,
                );
            }
        }
    }
}

/// Perform one BDF time step of the unsteady Navier-Stokes equations using a
/// pre-assembled [`super::workspace::BdfWorkspace`].
///
/// Identical numerics to [`solve_bdf_step_with_traction`], but the constant
/// mass / unit-μ viscous / divergence blocks are read from `workspace`
/// instead of being re-assembled; only the Oseen convection `C(u*)` is
/// assembled per step. The monolithic velocity block is
/// `mass_coeff·M + mu_step·K_unit + C(u*)` where `mu_step` is the effective
/// viscosity from `config.turbulence` (explicitly lagged: computed from the
/// previous step's peak velocity), matching the Picard steady solver.
///
/// Updates `state` in-place with the new velocity and pressure.
#[allow(clippy::too_many_arguments)]
pub fn solve_bdf_step_cached<VSpace, PSpace, BcFn>(
    mesh: &crate::mesh::Mesh,
    v_space: &VSpace,
    p_space: &PSpace,
    config: &NavierStokesConfig,
    bdf: &super::time_stepping::BdfConfig,
    state: &mut super::time_stepping::BdfState,
    bc_velocity_fn: &BcFn,
    bc_dofs: &[usize],
    tractions: &[super::traction::PressureTraction],
    workspace: &super::workspace::BdfWorkspace,
) -> bool
where
    VSpace: crate::spaces::FunctionSpace,
    PSpace: crate::spaces::FunctionSpace,
    BcFn: Fn(f64, f64, f64) -> Vec<f64>,
{
    solve_bdf_step_cached_with_force(
        mesh,
        v_space,
        p_space,
        config,
        bdf,
        state,
        bc_velocity_fn,
        bc_dofs,
        tractions,
        |_| vec![0.0; 3],
        |_| [[0.0; 3]; 3],
        workspace,
    )
}

/// Perform one cached BDF step with an affine body-force linearisation.
///
/// `body_force_jacobian_fn` returns `J = ∂f/∂u`.  The assembled system adds
/// `-J` to the left-hand side and `f` to the right-hand side.  This is exact
/// for affine forces such as the immersed rotating-wall penalty; non-affine
/// forces must be linearised by the caller rather than silently
/// finite-differenced here.
#[allow(clippy::too_many_arguments)]
pub fn solve_bdf_step_cached_with_force<VSpace, PSpace, BcFn, ForceFn, ForceJacFn>(
    mesh: &crate::mesh::Mesh,
    v_space: &VSpace,
    p_space: &PSpace,
    config: &NavierStokesConfig,
    bdf: &super::time_stepping::BdfConfig,
    state: &mut super::time_stepping::BdfState,
    bc_velocity_fn: &BcFn,
    bc_dofs: &[usize],
    tractions: &[super::traction::PressureTraction],
    body_force_fn: ForceFn,
    body_force_jacobian_fn: ForceJacFn,
    workspace: &super::workspace::BdfWorkspace,
) -> bool
where
    VSpace: crate::spaces::FunctionSpace,
    PSpace: crate::spaces::FunctionSpace,
    BcFn: Fn(f64, f64, f64) -> Vec<f64>,
    ForceFn: Fn([f64; 3]) -> Vec<f64> + Send + Sync + 'static,
    ForceJacFn: Fn([f64; 3]) -> [[f64; 3]; 3] + Send + Sync + 'static,
{
    use super::time_stepping::{BdfCoefficients, extrapolate_velocity};

    debug_assert_eq!(workspace.n_v, v_space.num_global_dofs(mesh));
    debug_assert_eq!(workspace.n_p, p_space.num_global_dofs(mesh));
    let progress = std::env::var("SONIUM_BDF_SOLVER_PROGRESS").as_deref() == Ok("1");
    let assembly_start = std::time::Instant::now();
    let n_v = workspace.n_v;
    let n_p = workspace.n_p;
    let n_total = n_v + n_p;
    let dim = mesh.dimension;
    let dt = bdf.dt;
    let coeffs = BdfCoefficients::for_order(bdf.order);

    // Extrapolated advection velocity, evaluated as the exact P1/P2 finite-
    // element field through the workspace-owned element locator.
    let u_star = extrapolate_velocity(state, bdf.order);
    let conv_triplet = assemble_bdf_convection(mesh, v_space, workspace, &u_star, config.rho, dim);

    let mass_triplet = &workspace.mass_triplet;
    let transported_viscous_triplet =
        match assemble_transported_viscous_triplet(mesh, v_space, config, workspace.quad_order) {
            Ok(triplet) => triplet,
            Err(_) => return false,
        };

    // Build monolithic system: block (0,0) = ρ a₀/Δt M + mu_step·K_unit + C(u*)
    let mass_coeff = config.rho * coeffs.a0 / dt;
    // Per-step viscosity scaling for the unit-μ viscous block: the effective
    // viscosity from the turbulence closure, computed from the previous
    // step's peak velocity (explicit lag — same semi-implicit philosophy as
    // the Oseen extrapolation). Mirrors the Picard `mu_eff` computation.
    let velocity_magnitude_max = state
        .velocity
        .chunks(dim)
        .map(|v| v.iter().map(|c| c * c).sum::<f64>().sqrt())
        .fold(0.0_f64, f64::max);
    let mu_step = config
        .effective_dynamic_viscosity_and_speed_derivative(velocity_magnitude_max)
        .0;
    // Viscous: either the lagged element-local transported block or the
    // legacy uniform unit-viscosity block scaled by `mu_step`.
    let use_transported = transported_viscous_triplet.is_some();
    let viscous_scale = if use_transported { 1.0 } else { mu_step };

    // RHS: velocity block = ρ/Δt M × (-a₁ u^n - a₂ u^{n-1}) + body_force
    let rho_over_dt = config.rho / dt;
    let bdf_source: Vec<f64> = state
        .velocity
        .iter()
        .zip(&state.velocity_prev)
        .map(|(&un, &unm1)| rho_over_dt * (-coeffs.a1 * un - coeffs.a2 * unm1))
        .collect();

    // Multiply BDF source by mass matrix: rhs_v = M × bdf_source. The CSR
    // conversion happens once at workspace setup, not every step.
    let rhs_v = workspace.mass_csr.matvec(&bdf_source);

    let mut rhs = vec![0.0; n_total];
    for (i, &val) in rhs_v.iter().enumerate() {
        if i < n_v {
            rhs[i] = val;
        }
    }

    // Sample fingerprint first: the miss path below moves the closures
    // into the assembly forms.
    let body_sample = config.cache_body_force_assembly.then(|| {
        body_force_sample_fingerprint(
            &body_force_fn,
            &body_force_jacobian_fn,
            &workspace.velocity_dof_coordinates,
        )
    });
    let cached_body = body_sample.and_then(|sample| {
        workspace
            .body_force_cache
            .borrow()
            .as_ref()
            .filter(|cached| cached.sample_fingerprint == sample)
            .map(|cached| (cached.rhs.clone(), cached.matrix.clone()))
    });
    let (body_force_rhs, body_force_matrix) = match cached_body {
        Some(hit) => hit,
        None => {
            let body_force = super::forms::StokesBodyForceForm::new(body_force_fn);
            let rhs =
                crate::assembly::assemble_linear(&body_force, v_space, mesh, workspace.quad_order);
            let body_force_jacobian = BdfBodyForceJacobianForm {
                jacobian: body_force_jacobian_fn,
            };
            let matrix = crate::assembly::assemble_bilinear(
                &body_force_jacobian,
                v_space,
                v_space,
                mesh,
                workspace.quad_order,
            );
            if let Some(sample) = body_sample {
                *workspace.body_force_cache.borrow_mut() =
                    Some(super::workspace::CachedBodyForceAssembly {
                        rhs: rhs.clone(),
                        matrix: matrix.clone(),
                        sample_fingerprint: sample,
                    });
            }
            (rhs, matrix)
        }
    };
    for (row, value) in body_force_rhs.iter().enumerate() {
        if row < n_v {
            rhs[row] += *value;
        }
    }
    // Pressure-traction natural BCs: adds ∫_Γ (−p·n)·v dΓ to the momentum RHS.
    super::traction::assemble_pressure_tractions_for_space(mesh, v_space, tractions, &mut rhs);

    // Build BC prescribed values. Coordinates come from the velocity space's
    // DOF layout, so P2 edge-midpoint DOFs get their boundary values too;
    // for P1 these are exactly the mesh node coordinates (Picard pattern).
    let mut bc_values = vec![0.0; n_total];
    let mut is_bc = vec![false; n_total];
    for &dof in bc_dofs {
        debug_assert!(dof < n_v, "velocity Dirichlet DOF {dof} is out of range");
        is_bc[dof] = true;
        let point = workspace.velocity_dof_coordinates[dof];
        let comp = dof % v_space.num_components();
        let bc_val = bc_velocity_fn(point[0], point[1], point[2]);
        bc_values[dof] = if comp < bc_val.len() {
            bc_val[comp]
        } else {
            0.0
        };
    }
    if n_p > 0 {
        is_bc[n_v + workspace.pressure_pin] = true;
        bc_values[n_v + workspace.pressure_pin] = workspace.pressure_pin_value;
    }

    // Fixed-pattern assembly: the merged monolithic pattern never changes
    // across steps, so values are recomputed in place instead of merging
    // and sorting tens of millions of triplets every step. A pattern
    // mismatch (or the first step) rebuilds the cache from scratch.
    let pin = if n_p > 0 {
        Some(n_v + workspace.pressure_pin)
    } else {
        None
    };
    let cached_fill = workspace
        .mono_pattern_cache
        .borrow()
        .as_ref()
        .and_then(|cache| {
            cache.try_fill(
                mass_coeff,
                viscous_scale,
                use_transported,
                transported_viscous_triplet.as_ref(),
                &conv_triplet,
                &body_force_matrix,
            )
        });
    let pattern_cached = cached_fill.is_some();
    let (values, value_fingerprints) = match cached_fill {
        Some(filled) => filled,
        None => {
            let cache = super::workspace::BdfMonoPatternCache::build(
                n_total,
                n_v,
                mass_triplet,
                &workspace.viscous_unit_triplet,
                transported_viscous_triplet.as_ref(),
                &workspace.block_01,
                &workspace.block_10,
                &conv_triplet,
                &body_force_matrix,
                pin,
            );
            let filled = cache
                .try_fill(
                    mass_coeff,
                    viscous_scale,
                    use_transported,
                    transported_viscous_triplet.as_ref(),
                    &conv_triplet,
                    &body_force_matrix,
                )
                .expect("fresh pattern cache matches current triplets");
            workspace.mono_pattern_cache.borrow_mut().replace(cache);
            filled
        }
    };
    let cached_pattern = workspace.mono_pattern_cache.borrow();
    let cache = cached_pattern.as_ref().expect("pattern cache built above");

    // Apply symmetric Dirichlet BCs: lift RHS, zero rows+cols, set diagonal=1
    let mut csr = cache.to_csr(values);
    for row in 0..n_total {
        let row_start = csr.row_ptrs[row];
        let row_end = csr.row_ptrs[row + 1];
        if is_bc[row] {
            for j in row_start..row_end {
                csr.values[j] = if csr.col_indices[j] == row { 1.0 } else { 0.0 };
            }
            rhs[row] = bc_values[row];
        } else {
            for j in row_start..row_end {
                let col = csr.col_indices[j];
                if is_bc[col] {
                    rhs[row] -= csr.values[j] * bc_values[col];
                    csr.values[j] = 0.0;
                }
            }
        }
    }

    // Backend (GPU) Schur product, memoized across same-matrix solves (the
    // coupled base/probe pair shares one). Pure function of the assembled
    // values, keyed on the pattern cache's dynamic fingerprints plus the
    // per-step scales.
    let backend_schur = backend_btdiagb_schur_cached(
        config,
        workspace,
        &csr,
        n_v,
        mass_coeff,
        viscous_scale,
        use_transported,
        value_fingerprints,
    );

    if progress {
        eprintln!(
            "BDF_ASSEMBLY_DONE assembly_ms={:.3} pattern_cached={pattern_cached}",
            assembly_start.elapsed().as_secs_f64() * 1000.0,
        );
    }

    // Solve
    let b: Vec<f64> = rhs;
    workspace.record_linear_system(&csr, &b);

    let mut initial_guess = Vec::with_capacity(n_total);
    initial_guess.extend_from_slice(&state.velocity);
    initial_guess.extend_from_slice(&state.pressure);
    let Some((x, telemetry)) = solve_cached_bdf_linear_system(
        config,
        workspace,
        &csr,
        &b,
        &initial_guess,
        n_v,
        backend_schur,
    ) else {
        return false;
    };
    workspace.record_linear_solve(telemetry, &b);

    // Advance state
    state.velocity_prev = state.velocity.clone();
    state.velocity = x.iter().take(n_v).copied().collect();
    state.pressure = x.iter().skip(n_v).copied().collect();
    state.time += dt;
    state.step += 1;
    true
}

/// Unsteady Schur product for backend (GPU/CPU-fallback) Jacobi-Schur
/// solves, memoized in the workspace across same-matrix solves.
///
/// The `BtDiagB` product is a pure function of the assembled monolithic
/// values, which are identified exactly by the fill's value fingerprints
/// plus the per-step scales. A key mismatch rebuilds from scratch, so
/// the memo can only save work, never serve a stale product.
#[allow(clippy::too_many_arguments)]
fn backend_btdiagb_schur_cached(
    config: &NavierStokesConfig,
    workspace: &super::workspace::BdfWorkspace,
    csr: &sonium_solvers::CsrMatrix<f64>,
    velocity_block_size: usize,
    mass_coeff: f64,
    viscous_scale: f64,
    use_transported: bool,
    fingerprints: super::workspace::ValueFingerprints,
) -> Option<sonium_solvers::CsrMatrix<f64>> {
    let resolved_backend = match config.linear_solver_backend {
        LinearSolverBackend::Auto => LinearSolverBackend::resolve_auto(),
        backend => backend,
    };
    // A GPU solve that fails setup falls back to the CPU path, which
    // needs the Schur product when the CPU preconditioner is a Schur
    // variant. Build it in that case so the fallback can succeed.
    let fallback_needs_schur = matches!(
        config.linear_solver_cpu_preconditioner,
        LinearSolverCpuPreconditionerKind::JacobiSchur
            | LinearSolverCpuPreconditionerKind::IlutSchur
            | LinearSolverCpuPreconditionerKind::IluSchur
    );
    if matches!(resolved_backend, LinearSolverBackend::Cpu)
        || (config.linear_solver_backend_preconditioner != BackendPreconditionerKind::JacobiSchur
            && !fallback_needs_schur)
    {
        return None;
    }
    let mut guard = workspace.bdf_schur_cache.borrow_mut();
    if let Some(entry) = guard.as_mut()
        && entry.matches(mass_coeff, viscous_scale, use_transported, fingerprints)
    {
        entry.hits += 1;
        return Some(entry.schur.clone());
    }
    let schur = sonium_solvers::btdiagb_schur_from_monolithic(csr, velocity_block_size);
    *guard = Some(super::workspace::BdfSchurCacheEntry {
        mass_coeff_bits: mass_coeff.to_bits(),
        viscous_scale_bits: viscous_scale.to_bits(),
        use_transported,
        value_fingerprints: fingerprints,
        schur: schur.clone(),
        hits: 0,
    });
    Some(schur)
}

fn solve_cached_bdf_linear_system(
    config: &NavierStokesConfig,
    workspace: &super::workspace::BdfWorkspace,
    csr: &sonium_solvers::CsrMatrix<f64>,
    b: &[f64],
    x0: &[f64],
    velocity_block_size: usize,
    backend_schur: Option<sonium_solvers::CsrMatrix<f64>>,
) -> Option<(Vec<f64>, super::workspace::BdfLinearSolveTelemetry)> {
    use super::workspace::{BdfCachedLinearSolver, BdfLinearSolveTelemetry, BdfLinearSolverKind};

    // Validate the CSR structure once per solve: every Krylov iteration
    // applies the operator, and the shared borrow freezes the workspace
    // storage so applies skip the O(nnz) re-validation.
    let operator = sonium_solvers::ValidatedCsr::new(csr);
    let progress = std::env::var("SONIUM_BDF_SOLVER_PROGRESS").as_deref() == Ok("1");
    if !config.use_direct_solver {
        let resolved_backend = match config.linear_solver_backend {
            LinearSolverBackend::Auto => LinearSolverBackend::resolve_auto(),
            backend => backend,
        };
        if !matches!(resolved_backend, LinearSolverBackend::Cpu) {
            // Unsteady BtDiagB Schur product, memoized across same-matrix
            // solves by the caller (a steady-Stokes Mp/mu matrix would
            // mis-scale the preconditioned Schur block by ~sigma/mu and
            // stall the backend Krylov solver on refined unsteady
            // systems).
            let pressure_schur = backend_schur;
            if progress {
                eprintln!(
                    "BDF_LINEAR_SETUP solver=backend-gmres rows={} nnz={} reused=false",
                    csr.num_rows,
                    csr.values.len()
                );
            }
            let (solution, actual_backend, iterations) = solve_iterative_linear_system(
                config,
                csr,
                b,
                Some(x0),
                velocity_block_size,
                pressure_schur.as_ref(),
            )?;
            let solver = if matches!(actual_backend, LinearSolverBackend::Cpu) {
                match config.linear_solver_cpu_preconditioner {
                    LinearSolverCpuPreconditionerKind::Ilu => BdfLinearSolverKind::GmresIlu,
                    LinearSolverCpuPreconditionerKind::JacobiSchur => {
                        BdfLinearSolverKind::FgmresJacobiSchur
                    }
                    LinearSolverCpuPreconditionerKind::IlutSchur => {
                        BdfLinearSolverKind::FgmresIlutSchur
                    }
                    LinearSolverCpuPreconditionerKind::IluSchur => {
                        BdfLinearSolverKind::FgmresIluSchur
                    }
                }
            } else {
                BdfLinearSolverKind::BackendFgmres
            };
            let true_relative_residual = true_relative_residual(csr, &solution, b);
            let converged = solution.iter().all(|value| value.is_finite())
                && true_relative_residual.is_finite()
                && true_relative_residual <= config.linear_solver_tolerance;
            if progress {
                let solver_name = match solver {
                    BdfLinearSolverKind::BackendFgmres => "backend-gmres",
                    BdfLinearSolverKind::GmresIlu => "gmres-ilu",
                    BdfLinearSolverKind::FgmresJacobiSchur => "fgmres-jacobi-schur",
                    BdfLinearSolverKind::FgmresIlutSchur => "fgmres-ilut-schur",
                    BdfLinearSolverKind::FgmresIluSchur => "fgmres-ilu-schur",
                    BdfLinearSolverKind::Direct => "direct",
                };
                eprintln!(
                    "BDF_LINEAR_SOLVE_DONE solver={solver_name} iterations={iterations} residual={true_relative_residual:.12e} converged={converged} reused=false"
                );
            }
            return converged.then_some((
                solution,
                BdfLinearSolveTelemetry {
                    solver,
                    resolved_linear_solver_backend: actual_backend,
                    iterations,
                    true_relative_residual,
                    setup_reused: false,
                    converged,
                },
            ));
        }
    }
    let (x, iterations, solver, setup_reused, solver_converged) = if config.use_direct_solver {
        let mut cache = workspace.linear_solver_cache.borrow_mut();
        let setup_reused = matches!(
            cache.as_ref(),
            Some(BdfCachedLinearSolver::Direct { matrix, .. })
                if csr_exactly_equal(matrix, csr)
        );
        if !setup_reused {
            if progress {
                eprintln!(
                    "BDF_LINEAR_SETUP solver=direct rows={} nnz={} reused=false",
                    csr.num_rows,
                    csr.values.len()
                );
            }
            let setup_start = std::time::Instant::now();
            let factorization =
                sonium_solvers::direct::SparseLuFactorization::factorize(csr).ok()?;
            if progress {
                eprintln!(
                    "BDF_LINEAR_SETUP_DONE solver=direct setup_ms={:.3}",
                    setup_start.elapsed().as_secs_f64() * 1000.0
                );
            }
            *cache = Some(BdfCachedLinearSolver::Direct {
                matrix: csr.clone(),
                factorization,
            });
        }
        let solution = match cache.as_ref()? {
            BdfCachedLinearSolver::Direct { factorization, .. } => factorization.solve(b).ok()?,
            BdfCachedLinearSolver::JacobiSchur { .. } | BdfCachedLinearSolver::Ilu { .. } => {
                unreachable!()
            }
        };
        (solution, 0, BdfLinearSolverKind::Direct, setup_reused, true)
    } else {
        match config.linear_solver_cpu_preconditioner {
            LinearSolverCpuPreconditionerKind::Ilu => {
                let mut cache = workspace.linear_solver_cache.borrow_mut();
                let setup_reused = matches!(
                    cache.as_ref(),
                    Some(BdfCachedLinearSolver::Ilu { matrix, .. })
                        if csr_exactly_equal(matrix, csr)
                );
                if !setup_reused {
                    if progress {
                        eprintln!(
                            "BDF_LINEAR_SETUP solver=gmres-ilu rows={} nnz={} reused=false",
                            csr.num_rows,
                            csr.values.len()
                        );
                    }
                    let setup_start = std::time::Instant::now();
                    let preconditioner = sonium_solvers::IluPreconditioner::from_csr(csr);
                    *cache = Some(BdfCachedLinearSolver::Ilu {
                        matrix: csr.clone(),
                        preconditioner,
                    });
                    if progress {
                        eprintln!(
                            "BDF_LINEAR_SETUP_DONE solver=gmres-ilu setup_ms={:.3}",
                            setup_start.elapsed().as_secs_f64() * 1000.0
                        );
                    }
                }
                let solve_config = sonium_solvers::GmresConfig {
                    max_iterations: config.linear_solver_max_iterations.max(1),
                    restart: 50,
                    tolerance: config.linear_solver_tolerance,
                    exact_residual_check_interval: 10,
                    print_interval: if progress { 100 } else { 0 },
                    ..Default::default()
                };
                let preconditioner = match cache.as_ref()? {
                    BdfCachedLinearSolver::Ilu { preconditioner, .. } => preconditioner,
                    BdfCachedLinearSolver::Direct { .. }
                    | BdfCachedLinearSolver::JacobiSchur { .. } => unreachable!(),
                };
                let solve_start = std::time::Instant::now();
                let result = sonium_solvers::gmres_preconditioned_with_guess(
                    &operator,
                    preconditioner,
                    b,
                    Some(x0),
                    &solve_config,
                );
                if progress {
                    eprintln!(
                        "BDF_LINEAR_SOLVE_DONE solver=gmres-ilu iterations={} residual={:.12e} estimated_residual={:.12e} backward_error={:.12e} converged={} reused={setup_reused} solve_ms={:.3}",
                        result.iterations,
                        result.residual,
                        result.estimated_residual,
                        result.backward_error,
                        result.converged,
                        solve_start.elapsed().as_secs_f64() * 1000.0,
                    );
                }
                (
                    result.x,
                    result.iterations,
                    BdfLinearSolverKind::GmresIlu,
                    setup_reused,
                    result.converged,
                )
            }
            LinearSolverCpuPreconditionerKind::JacobiSchur
            | LinearSolverCpuPreconditionerKind::IlutSchur
            | LinearSolverCpuPreconditionerKind::IluSchur => {
                let solver_tag = match config.linear_solver_cpu_preconditioner {
                    LinearSolverCpuPreconditionerKind::IlutSchur => "fgmres-ilut-schur",
                    LinearSolverCpuPreconditionerKind::IluSchur => "fgmres-ilu-schur",
                    _ => "fgmres-jacobi-schur",
                };
                let mut cache = workspace.linear_solver_cache.borrow_mut();
                let exact_reuse = matches!(
                    cache.as_ref(),
                    Some(BdfCachedLinearSolver::JacobiSchur { matrix, .. })
                        if csr_exactly_equal(matrix, csr)
                );
                // Lagged reuse: same-pattern matrix within tolerance and
                // lag budget left. The stored setup matrix is untouched by
                // lagged solves, so the tolerance always measures drift
                // from the last fresh setup, not step-to-step wobble.
                let mut lagged = false;
                if !exact_reuse
                    && config.linear_solver_setup_max_lag_steps > 0
                    && let Some(BdfCachedLinearSolver::JacobiSchur {
                        matrix, lag_steps, ..
                    }) = cache.as_ref()
                    && *lag_steps < config.linear_solver_setup_max_lag_steps
                    && csr_relative_change(matrix, csr)
                        .is_some_and(|change| change < config.linear_solver_setup_lag_tolerance)
                {
                    lagged = true;
                }
                if lagged
                    && let Some(BdfCachedLinearSolver::JacobiSchur { lag_steps, .. }) =
                        cache.as_mut()
                {
                    *lag_steps += 1;
                }
                let mut setup_reused = exact_reuse || lagged;
                let mut retried = false;
                let mut total_iterations = 0usize;
                let (x, iterations, solver, setup_reused, solver_converged) = loop {
                    if !setup_reused {
                        if progress {
                            eprintln!(
                                "BDF_LINEAR_SETUP solver={solver_tag} rows={} nnz={} reused=false",
                                csr.num_rows,
                                csr.values.len()
                            );
                        }
                        let block_config = match config.linear_solver_cpu_preconditioner {
                            LinearSolverCpuPreconditionerKind::JacobiSchur => {
                                sonium_solvers::BlockPreconditionerConfig::Diagonal
                            }
                            LinearSolverCpuPreconditionerKind::IlutSchur => {
                                sonium_solvers::BlockPreconditionerConfig::Ilut(
                                    sonium_solvers::IlutConfig::new(1.0e-3, 2, 1.0e-12),
                                )
                            }
                            LinearSolverCpuPreconditionerKind::IluSchur => {
                                sonium_solvers::BlockPreconditionerConfig::Ilu
                            }
                            LinearSolverCpuPreconditionerKind::Ilu => unreachable!(),
                        };
                        // BtDiagB assembles B^T*diag(F)^{-1}*B from the actual
                        // unsteady momentum block F = sigma*M + mu*K
                        // (sigma = rho*a0/dt). A steady-Stokes Mp/mu Schur
                        // approximation mis-scales the preconditioned Schur
                        // block by ~sigma/mu, splitting the spectrum across 4+
                        // orders of magnitude and stalling FGMRES on refined
                        // unsteady systems (speaker-port fine-3D: 5000
                        // iterations without reaching 1e-8). The assembled
                        // Schur block is Laplacian-like, so it always gets an
                        // ILUT subsolver (same drop/fill/pivot values as the
                        // IlutSchur velocity block); point Jacobi on a
                        // Laplacian leaves an h^{-2} spread that still costs
                        // thousands of iterations on fine meshes. If ILUT setup
                        // rejects the Schur block, fall back to Jacobi rather
                        // than failing the step.
                        let schur_ilut = sonium_solvers::BlockPreconditionerConfig::Ilut(
                            sonium_solvers::IlutConfig::new(1.0e-3, 2, 1.0e-12),
                        );
                        let sp_config = sonium_solvers::SaddlePointConfig {
                            block1_size: velocity_block_size,
                            block1_config: block_config,
                            block2_config: schur_ilut,
                            schur_strategy: sonium_solvers::SchurStrategy::BtDiagB,
                            upper_triangular: true,
                        };
                        let mut fallback_config = sp_config.clone();
                        fallback_config.block2_config =
                            sonium_solvers::BlockPreconditionerConfig::Diagonal;
                        let setup_start = std::time::Instant::now();
                        let preconditioner =
                            sonium_solvers::SaddlePointPreconditioner::<f64>::try_from_matrix(
                                csr, sp_config,
                            )
                            .or_else(|_| {
                                sonium_solvers::SaddlePointPreconditioner::<f64>::try_from_matrix(
                                    csr,
                                    fallback_config,
                                )
                            })
                            .ok()?;
                        *cache = Some(BdfCachedLinearSolver::JacobiSchur {
                            matrix: csr.clone(),
                            preconditioner,
                            lag_steps: 0,
                        });
                        if progress {
                            eprintln!(
                                "BDF_LINEAR_SETUP_DONE solver={solver_tag} setup_ms={:.3}",
                                setup_start.elapsed().as_secs_f64() * 1000.0
                            );
                        }
                    }
                    let preconditioner = match cache.as_ref()? {
                        BdfCachedLinearSolver::JacobiSchur { preconditioner, .. } => preconditioner,
                        BdfCachedLinearSolver::Direct { .. }
                        | BdfCachedLinearSolver::Ilu { .. } => {
                            unreachable!()
                        }
                    };
                    let solve_config = sonium_solvers::FgmresConfig {
                        max_iterations: config.linear_solver_max_iterations.max(1),
                        restart: 50,
                        tolerance: config.linear_solver_tolerance,
                        print_interval: if progress { 100 } else { 0 },
                        exact_residual_check_interval: 10,
                        convergence_check_interval: 1,
                    };
                    let solve_start = std::time::Instant::now();
                    let result = sonium_solvers::fgmres_preconditioned(
                        &operator,
                        preconditioner,
                        b,
                        Some(x0),
                        &solve_config,
                    );
                    if progress {
                        eprintln!(
                            "BDF_LINEAR_SOLVE_DONE solver={solver_tag} iterations={} residual={:.12e} estimated_residual={:.12e} backward_error={:.12e} converged={} reused={setup_reused} solve_ms={:.3}",
                            result.iterations,
                            result.residual,
                            result.estimated_residual,
                            result.backward_error,
                            result.converged,
                            solve_start.elapsed().as_secs_f64() * 1000.0,
                        );
                    }
                    total_iterations += result.iterations;
                    if result.converged || !lagged || retried {
                        let solver_kind = match config.linear_solver_cpu_preconditioner {
                            LinearSolverCpuPreconditionerKind::JacobiSchur => {
                                BdfLinearSolverKind::FgmresJacobiSchur
                            }
                            LinearSolverCpuPreconditionerKind::IlutSchur => {
                                BdfLinearSolverKind::FgmresIlutSchur
                            }
                            LinearSolverCpuPreconditionerKind::IluSchur => {
                                BdfLinearSolverKind::FgmresIluSchur
                            }
                            LinearSolverCpuPreconditionerKind::Ilu => unreachable!(),
                        };
                        break (
                            result.x,
                            total_iterations,
                            solver_kind,
                            setup_reused,
                            result.converged,
                        );
                    }
                    // Lagged setup failed: rebuild fresh and retry once, so a
                    // stale setup wastes one solve but never fails the step.
                    retried = true;
                    lagged = false;
                    setup_reused = false;
                };
                (x, iterations, solver, setup_reused, solver_converged)
            }
        }
    };

    let true_relative_residual = true_relative_residual(csr, &x, b);
    let finite = x.iter().all(|value| value.is_finite()) && true_relative_residual.is_finite();
    let converged =
        solver_converged && finite && true_relative_residual <= config.linear_solver_tolerance;
    let telemetry = BdfLinearSolveTelemetry {
        solver,
        resolved_linear_solver_backend: LinearSolverBackend::Cpu,
        iterations,
        true_relative_residual,
        setup_reused,
        converged,
    };
    converged.then_some((x, telemetry))
}

fn csr_exactly_equal(
    left: &sonium_solvers::CsrMatrix<f64>,
    right: &sonium_solvers::CsrMatrix<f64>,
) -> bool {
    left.num_rows == right.num_rows
        && left.num_cols == right.num_cols
        && left.row_ptrs == right.row_ptrs
        && left.col_indices == right.col_indices
        && left
            .values
            .iter()
            .zip(&right.values)
            .all(|(a, b)| a.to_bits() == b.to_bits())
}

/// Relative value change `||B - A||_F / ||A||_F` between two CSR
/// matrices, or `None` when the structure differs. Guards lagged
/// preconditioner reuse: only a small drift justifies skipping setup.
fn csr_relative_change(
    old: &sonium_solvers::CsrMatrix<f64>,
    new: &sonium_solvers::CsrMatrix<f64>,
) -> Option<f64> {
    if old.num_rows != new.num_rows
        || old.num_cols != new.num_cols
        || old.row_ptrs != new.row_ptrs
        || old.col_indices != new.col_indices
        || old.values.len() != new.values.len()
    {
        return None;
    }
    let mut diff_squares = 0.0;
    let mut old_squares = 0.0;
    for (a, b) in old.values.iter().zip(&new.values) {
        let delta = b - a;
        diff_squares += delta * delta;
        old_squares += a * a;
    }
    if old_squares <= 0.0 {
        return None;
    }
    Some((diff_squares / old_squares).sqrt())
}

fn true_relative_residual(matrix: &sonium_solvers::CsrMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
    if x.len() != matrix.num_cols || b.len() != matrix.num_rows {
        return f64::INFINITY;
    }
    let ax = matrix.matvec(x);
    let residual_norm = ax
        .iter()
        .zip(b)
        .map(|(a, rhs)| (rhs - a).powi(2))
        .sum::<f64>()
        .sqrt();
    let rhs_norm = b.iter().map(|value| value * value).sum::<f64>().sqrt();
    residual_norm / rhs_norm.max(f64::MIN_POSITIVE)
}

/// FNV-1a fingerprint of body-force closures at strided sample DOFs.
///
/// Guards the workspace body-force cache: a changed force (or a workspace
/// shared across problems) mismatches and triggers reassembly, so the
/// cache can only serve closures identical at every sample. Eight samples
/// keep the check to a handful of closure evaluations per step.
fn body_force_sample_fingerprint<ForceFn, ForceJacFn>(
    body_force_fn: &ForceFn,
    body_force_jacobian_fn: &ForceJacFn,
    dof_coordinates: &[[f64; 3]],
) -> u64
where
    ForceFn: Fn([f64; 3]) -> Vec<f64>,
    ForceJacFn: Fn([f64; 3]) -> [[f64; 3]; 3],
{
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    const SAMPLES: usize = 8;
    let mut hash = FNV_OFFSET;
    let mut push = |bits: u64| {
        for byte in bits.to_ne_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    };
    let count = dof_coordinates.len();
    if count == 0 {
        return hash;
    }
    let stride = count.div_ceil(SAMPLES).max(1);
    let mut sampled = 0usize;
    for point in dof_coordinates.iter().step_by(stride) {
        for value in body_force_fn(*point) {
            push(value.to_bits());
        }
        for row in body_force_jacobian_fn(*point) {
            for value in row {
                push(value.to_bits());
            }
        }
        sampled += 1;
        if sampled >= SAMPLES {
            break;
        }
    }
    push(sampled as u64);
    hash
}

fn assemble_bdf_convection<VSpace: crate::spaces::FunctionSpace>(
    mesh: &crate::mesh::Mesh,
    v_space: &VSpace,
    workspace: &super::workspace::BdfWorkspace,
    u_star: &[f64],
    rho: f64,
    dim: usize,
) -> crate::assembly::TripletMatrix<f64> {
    let velocity_field = workspace.element_velocity_topology.field(u_star);
    let convection = OseenConvectionForm::new(rho, |coords: [f64; 3]| {
        velocity_field
            .velocity_at(coords)
            .expect("BDF quadrature point was not located in its velocity mesh")[..dim]
            .to_vec()
    });
    crate::assembly::assemble_bilinear(&convection, v_space, v_space, mesh, workspace.quad_order)
}

/// Signed discrete energy channels for one accepted BDF velocity state.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct BdfStepAudit {
    /// Fluid kinetic energy at the beginning of the accepted step (J).
    pub fluid_kinetic_old_j: f64,
    /// Fluid kinetic energy at the end of the accepted step (J).
    pub fluid_kinetic_new_j: f64,
    /// Signed work of the discrete BDF mass term (J).
    pub bdf_mass_work_j: f64,
    /// BDF mass work not represented by the kinetic-energy change (J).
    pub bdf_numerical_j: f64,
    /// Nonnegative total molecular-plus-eddy viscous dissipation (J).
    pub viscous_dissipation_j: f64,
    /// Molecular-viscosity contribution to `viscous_dissipation_j` (J).
    pub molecular_viscous_dissipation_j: f64,
    /// Eddy-viscosity contribution to `viscous_dissipation_j` (J).
    pub eddy_viscous_dissipation_j: f64,
    /// Signed convection contribution; positive values export energy (J).
    pub convection_export_j: f64,
    /// Signed pressure-constraint work in the accepted algebraic state (J).
    pub pressure_work_j: f64,
    /// Signed total pressure-traction work on all audited caps (J).
    pub traction_work_j: f64,
    /// Signed pressure-traction work on the box cap (J).
    pub box_traction_work_j: f64,
    /// Signed pressure-traction work on the exterior cap (J).
    pub exterior_traction_work_j: f64,
    /// Signed work-equivalent residual of the accepted linear solve (J).
    pub solve_residual_j: f64,
    /// Effective dynamic viscosity used for this Oseen/BDF step (Pa·s).
    pub effective_viscosity_pa_s: f64,
}

fn triplet_bilinear(
    left: &[f64],
    matrix: &crate::assembly::TripletMatrix<f64>,
    right: &[f64],
) -> f64 {
    (0..matrix.nnz())
        .map(|k| left[matrix.rows[k]] * matrix.values[k] * right[matrix.cols[k]])
        .sum()
}

fn traction_vector<VSpace: crate::spaces::FunctionSpace>(
    mesh: &crate::mesh::Mesh,
    v_space: &VSpace,
    n_v: usize,
    tractions: &[super::traction::PressureTraction],
) -> Vec<f64> {
    let mut rhs = vec![0.0; n_v];
    super::traction::assemble_pressure_tractions_for_space(mesh, v_space, tractions, &mut rhs);
    rhs
}

/// Audit an accepted BDF state against the unmodified discrete momentum and
/// incompressibility equations. This performs no solve and is suitable for an
/// affine combination of trial solutions.
#[allow(clippy::too_many_arguments)]
pub fn audit_bdf_step<VSpace: crate::spaces::FunctionSpace>(
    mesh: &crate::mesh::Mesh,
    v_space: &VSpace,
    config: &NavierStokesConfig,
    bdf: &super::time_stepping::BdfConfig,
    previous: &super::time_stepping::BdfState,
    accepted: &super::time_stepping::BdfState,
    tractions: &[super::traction::PressureTraction],
    box_marker: i32,
    exterior_marker: i32,
    workspace: &super::workspace::BdfWorkspace,
) -> BdfStepAudit {
    use super::time_stepping::{BdfCoefficients, extrapolate_velocity};

    let progress = std::env::var("SONIUM_BDF_SOLVER_PROGRESS").as_deref() == Ok("1");
    let audit_start = std::time::Instant::now();
    let h = bdf.dt;
    let coeffs = BdfCoefficients::for_order(bdf.order);
    let u = &accepted.velocity;
    let un = &previous.velocity;
    let unm1 = &previous.velocity_prev;
    let u_star = extrapolate_velocity(previous, bdf.order);
    let convection = assemble_bdf_convection(
        mesh,
        v_space,
        workspace,
        &u_star,
        config.rho,
        mesh.dimension,
    );
    let kinetic = |velocity: &[f64]| {
        0.5 * config.rho * triplet_bilinear(velocity, &workspace.mass_triplet, velocity)
    };
    let derivative: Vec<f64> = u
        .iter()
        .zip(un)
        .zip(unm1)
        .map(|((&u1, &u0), &um1)| coeffs.a0 * u1 + coeffs.a1 * u0 + coeffs.a2 * um1)
        .collect();
    let bdf_mass_work = config.rho * triplet_bilinear(u, &workspace.mass_triplet, &derivative);
    let effective_viscosity = config
        .effective_dynamic_viscosity_and_speed_derivative(
            un.chunks(mesh.dimension)
                .map(|v| v.iter().map(|x| x * x).sum::<f64>().sqrt())
                .fold(0.0_f64, f64::max),
        )
        .0;
    let viscous_quadratic = h * triplet_bilinear(u, &workspace.viscous_unit_triplet, u);
    let molecular_viscous = config.mu * viscous_quadratic;
    let eddy_viscous = (effective_viscosity - config.mu) * viscous_quadratic;
    let viscous = molecular_viscous + eddy_viscous;
    let convection_work = h * triplet_bilinear(u, &convection, u);
    let pressure_work = h
        * (triplet_bilinear(u, &workspace.block_01, &accepted.pressure)
            + triplet_bilinear(&accepted.pressure, &workspace.block_10, u));
    let total_traction = traction_vector(mesh, v_space, workspace.n_v, tractions);
    let work = |force: &[f64]| h * u.iter().zip(force).map(|(a, b)| a * b).sum::<f64>();
    let traction_work = work(&total_traction);
    let marker_work = |marker| {
        let selected: Vec<_> = tractions
            .iter()
            .copied()
            .filter(|traction| traction.marker == marker)
            .collect();
        work(&traction_vector(mesh, v_space, workspace.n_v, &selected))
    };
    let old_kinetic = kinetic(un);
    let new_kinetic = kinetic(u);
    let audit = BdfStepAudit {
        fluid_kinetic_old_j: old_kinetic,
        fluid_kinetic_new_j: new_kinetic,
        bdf_mass_work_j: bdf_mass_work,
        bdf_numerical_j: bdf_mass_work - (new_kinetic - old_kinetic),
        viscous_dissipation_j: viscous,
        molecular_viscous_dissipation_j: molecular_viscous,
        eddy_viscous_dissipation_j: eddy_viscous,
        convection_export_j: convection_work,
        pressure_work_j: pressure_work,
        traction_work_j: traction_work,
        box_traction_work_j: marker_work(box_marker),
        exterior_traction_work_j: marker_work(exterior_marker),
        solve_residual_j: traction_work - bdf_mass_work - viscous - convection_work - pressure_work,
        effective_viscosity_pa_s: effective_viscosity,
    };
    if progress {
        eprintln!(
            "BDF_AUDIT_DONE audit_ms={:.3}",
            audit_start.elapsed().as_secs_f64() * 1000.0
        );
    }
    audit
}

/// Perform one BDF time step of the unsteady Navier-Stokes equations.
///
/// Backward-compatible wrapper around [`solve_bdf_step_with_traction`] with
/// no pressure-traction boundary conditions.
#[allow(clippy::too_many_arguments)]
pub fn solve_bdf_step<VSpace, PSpace, BcFn>(
    mesh: &crate::mesh::Mesh,
    v_space: &VSpace,
    p_space: &PSpace,
    config: &NavierStokesConfig,
    bdf: &super::time_stepping::BdfConfig,
    state: &mut super::time_stepping::BdfState,
    bc_velocity_fn: &BcFn,
    bc_dofs: &[usize],
) -> bool
where
    VSpace: crate::spaces::FunctionSpace,
    PSpace: crate::spaces::FunctionSpace,
    BcFn: Fn(f64, f64, f64) -> Vec<f64>,
{
    solve_bdf_step_with_traction(
        mesh,
        v_space,
        p_space,
        config,
        bdf,
        state,
        bc_velocity_fn,
        bc_dofs,
        &[],
    )
}

// ---------------------------------------------------------------------------
// Flow-specific post-processing
// ---------------------------------------------------------------------------

/// Compute vorticity ω = ∂v/∂x - ∂u/∂y (2D scalar vorticity).
///
/// `velocity` contains interleaved [u0, v0, u1, v1, ...] at mesh nodes.
/// Returns one vorticity value per element (piecewise constant approximation).
pub fn compute_vorticity_2d(mesh: &crate::mesh::Mesh, velocity: &[f64]) -> Vec<f64> {
    let mut vorticity = Vec::with_capacity(mesh.num_elements());

    for elem in &mesh.elements {
        let verts = elem.vertices();
        let n = verts.len();
        if n < 3 {
            vorticity.push(0.0);
            continue;
        }

        // Compute ∂v/∂x - ∂u/∂y using P1 gradient
        // For a triangle with vertices (x0,y0), (x1,y1), (x2,y2):
        // Gradient of a P1 field is constant within the element.
        let x: Vec<f64> = verts.iter().map(|&v| mesh.nodes[v].x).collect();
        let y: Vec<f64> = verts.iter().map(|&v| mesh.nodes[v].y).collect();

        let area2 = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]);
        if area2.abs() < 1e-30 {
            vorticity.push(0.0);
            continue;
        }

        // u-component at vertices
        let u: Vec<f64> = verts.iter().map(|&v| velocity[2 * v]).collect();
        // v-component at vertices
        let v: Vec<f64> = verts.iter().map(|&vi| velocity[2 * vi + 1]).collect();

        // P1 gradients on a triangle:
        //   ∂f/∂x = (f0(y1-y2) + f1(y2-y0) + f2(y0-y1)) / (2A)
        //   ∂f/∂y = (f0(x2-x1) + f1(x0-x2) + f2(x1-x0)) / (2A)
        let du_dy = (u[0] * (x[2] - x[1]) + u[1] * (x[0] - x[2]) + u[2] * (x[1] - x[0])) / area2;
        let dv_dx = (v[0] * (y[1] - y[2]) + v[1] * (y[2] - y[0]) + v[2] * (y[0] - y[1])) / area2;

        vorticity.push(dv_dx - du_dy);
    }

    vorticity
}

/// Compute force on a closed Dirichlet boundary from the discrete momentum
/// residual (Babuška-Miller variational force evaluation).
///
/// The lift functions are one on the selected boundary's velocity component
/// and zero on every other velocity DOF.  This avoids differentiating a P1
/// velocity directly at the wall, which is only zeroth-order accurate and can
/// badly pollute drag/lift even when the field solution is converging.
pub fn compute_drag_lift_variational<ForceFn>(
    mesh: &crate::mesh::Mesh,
    velocity: &[f64],
    pressure: &[f64],
    mu: f64,
    rho: f64,
    boundary_marker: i32,
    body_force_fn: ForceFn,
) -> (f64, f64)
where
    ForceFn: Fn([f64; 3]) -> Vec<f64>,
{
    use crate::spaces::FunctionSpace as _;

    assert_eq!(
        mesh.dimension, 2,
        "variational drag/lift requires a 2-D mesh"
    );
    assert_eq!(
        velocity.len(),
        2 * mesh.num_nodes(),
        "variational drag/lift currently requires a two-component P1 velocity"
    );
    assert_eq!(pressure.len(), mesh.num_nodes());
    assert!(mu.is_finite() && mu >= 0.0);
    assert!(rho.is_finite() && rho > 0.0);
    assert!(velocity.iter().all(|value| value.is_finite()));
    assert!(pressure.iter().all(|value| value.is_finite()));

    let mut boundary_nodes = vec![false; mesh.num_nodes()];
    for boundary in &mesh.boundaries {
        if boundary.marker == boundary_marker {
            for &node in &boundary.nodes {
                boundary_nodes[node] = true;
            }
        }
    }
    assert!(
        boundary_nodes.iter().any(|&selected| selected),
        "variational drag/lift boundary marker {boundary_marker} is absent"
    );

    let space = crate::spaces::H1Space::new(crate::basis::PolynomialDegree::P1);
    let mut fluid_reaction = [0.0_f64; 2];
    for (element_idx, element) in mesh.elements.iter().enumerate() {
        assert_eq!(
            element.element_type,
            crate::mesh::ElementType::Triangle,
            "variational drag/lift currently requires triangular elements"
        );
        let local_dofs = space.local_to_global(mesh, element_idx);
        if !local_dofs.iter().any(|&node| boundary_nodes[node]) {
            continue;
        }

        let quadrature = crate::quadrature::QuadratureRule::new(element.element_type, 3);
        for point in quadrature.iter() {
            let tabulation =
                space.tabulate(mesh, element_idx, point.xi(), point.eta(), point.zeta());
            let weight = point.weight * tabulation.det_j;
            let mut value = [0.0_f64; 2];
            let mut gradient = [[0.0_f64; 2]; 2];
            let mut pressure_value = 0.0;
            for (local, &global) in local_dofs.iter().enumerate() {
                let basis_value = tabulation.value(local);
                pressure_value += pressure[global] * basis_value;
                for component in 0..2 {
                    value[component] += velocity[2 * global + component] * basis_value;
                    for (direction, derivative) in gradient[component].iter_mut().enumerate() {
                        *derivative +=
                            velocity[2 * global + component] * tabulation.grad(local, direction);
                    }
                }
            }
            let force = body_force_fn(tabulation.physical_coords);
            assert!(force.len() >= 2, "body force must have two components");
            assert!(force[0].is_finite() && force[1].is_finite());

            for (local, &global) in local_dofs.iter().enumerate() {
                if !boundary_nodes[global] {
                    continue;
                }
                let basis_value = tabulation.value(local);
                for component in 0..2 {
                    let viscous: f64 = (0..2)
                        .map(|direction| {
                            mu * tabulation.grad(local, direction) * gradient[component][direction]
                        })
                        .sum();
                    let convection = rho
                        * basis_value
                        * (value[0] * gradient[component][0] + value[1] * gradient[component][1]);
                    let pressure_term = -pressure_value * tabulation.grad(local, component);
                    fluid_reaction[component] += (viscous + convection + pressure_term
                        - basis_value * force[component])
                        * weight;
                }
            }
        }
    }

    let force_on_solid = [-fluid_reaction[0], -fluid_reaction[1]];
    assert!(force_on_solid.iter().all(|value| value.is_finite()));
    (force_on_solid[0], force_on_solid[1])
}

/// Compute the force exerted by a 2-D P1/P2 velocity and P1 pressure solution
/// on a solid boundary.
///
/// With `n` pointing out of the fluid domain, the traction exerted on the
/// fluid is `σ n`.  By action/reaction, the force on the solid is:
///
/// `F_solid = -∫_Γ σ n dΓ`, where `σ = -pI + μ(∇u + ∇uᵀ)`.
///
/// Returns (drag, lift) = (F_x, F_y).
pub fn compute_drag_lift(
    mesh: &crate::mesh::Mesh,
    velocity: &[f64],
    pressure: &[f64],
    mu: f64,
    boundary_marker: i32,
) -> (f64, f64) {
    use crate::spaces::FunctionSpace as _;

    assert_eq!(
        mesh.dimension, 2,
        "drag/lift integration requires a 2-D mesh"
    );
    let p1_space = crate::spaces::H1Space::new(crate::basis::PolynomialDegree::P1);
    let p2_space = crate::spaces::H1Space::new(crate::basis::PolynomialDegree::P2);
    let velocity_space = if velocity.len() == 2 * p1_space.num_global_dofs(mesh) {
        &p1_space
    } else if velocity.len() == 2 * p2_space.num_global_dofs(mesh) {
        &p2_space
    } else {
        panic!(
            "velocity length {} is neither a two-component P1 nor P2 field",
            velocity.len()
        );
    };
    assert_eq!(
        pressure.len(),
        mesh.num_nodes(),
        "P1 pressure must contain one value per mesh node"
    );
    assert!(
        mu.is_finite() && mu >= 0.0,
        "dynamic viscosity must be finite and non-negative"
    );
    assert!(
        velocity.iter().all(|value| value.is_finite()),
        "velocity contains a non-finite value"
    );
    assert!(
        pressure.iter().all(|value| value.is_finite()),
        "pressure contains a non-finite value"
    );

    let mut drag = 0.0;
    let mut lift = 0.0;

    for boundary in &mesh.boundaries {
        if boundary.marker != boundary_marker {
            continue;
        }
        let nodes = &boundary.nodes;
        assert_eq!(
            nodes.len(),
            2,
            "drag/lift integration currently requires P1 boundary edges"
        );

        let p0 = &mesh.nodes[nodes[0]];
        let p1 = &mesh.nodes[nodes[1]];
        let dx = p1.x - p0.x;
        let dy = p1.y - p0.y;
        let len = (dx * dx + dy * dy).sqrt();
        assert!(
            len >= 1e-15,
            "drag/lift boundary contains a zero-length edge"
        );

        let element = mesh
            .elements
            .get(boundary.element_idx)
            .expect("drag/lift boundary has an invalid owning element");
        assert_eq!(
            element.element_type,
            crate::mesh::ElementType::Triangle,
            "drag/lift integration currently requires triangular elements"
        );
        let vertices = element.vertices();
        assert!(
            nodes.iter().all(|node| vertices.contains(node)),
            "drag/lift boundary edge does not belong to its owning element"
        );

        // Orient the normal geometrically rather than trusting boundary-node
        // ordering, which is not guaranteed by imported or generated meshes.
        let centroid_x = vertices.iter().map(|&node| mesh.nodes[node].x).sum::<f64>() / 3.0;
        let centroid_y = vertices.iter().map(|&node| mesh.nodes[node].y).sum::<f64>() / 3.0;
        let midpoint_x = 0.5 * (p0.x + p1.x);
        let midpoint_y = 0.5 * (p0.y + p1.y);
        let mut nx = dy / len;
        let mut ny = -dx / len;
        if nx * (midpoint_x - centroid_x) + ny * (midpoint_y - centroid_y) < 0.0 {
            nx = -nx;
            ny = -ny;
        }

        let [v0, v1, v2] = [vertices[0], vertices[1], vertices[2]];
        let [x0, x1, x2] = [mesh.nodes[v0].x, mesh.nodes[v1].x, mesh.nodes[v2].x];
        let [y0, y1, y2] = [mesh.nodes[v0].y, mesh.nodes[v1].y, mesh.nodes[v2].y];
        let area2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        assert!(
            area2.abs() >= 1.0e-30,
            "drag/lift owning element has zero area"
        );
        let local_edge_nodes = nodes
            .iter()
            .map(|node| {
                vertices
                    .iter()
                    .position(|vertex| vertex == node)
                    .expect("boundary node belongs to owning triangle")
            })
            .collect::<Vec<_>>();
        let edge = if local_edge_nodes[0] < local_edge_nodes[1] {
            (local_edge_nodes[0], local_edge_nodes[1])
        } else {
            (local_edge_nodes[1], local_edge_nodes[0])
        };
        let (xi, eta) = match edge {
            (0, 1) => (0.5, 0.0),
            (1, 2) => (0.5, 0.5),
            (0, 2) => (0.0, 0.5),
            _ => unreachable!("triangle boundary edge must join two distinct vertices"),
        };
        let tabulation = velocity_space.tabulate(mesh, boundary.element_idx, xi, eta, 0.0);
        let local_dofs = velocity_space.local_to_global(mesh, boundary.element_idx);
        let velocity_gradient = |component: usize, direction: usize| {
            local_dofs
                .iter()
                .enumerate()
                .map(|(local, &global)| {
                    velocity[2 * global + component] * tabulation.grad(local, direction)
                })
                .sum::<f64>()
        };
        let du_dx = velocity_gradient(0, 0);
        let du_dy = velocity_gradient(0, 1);
        let dv_dx = velocity_gradient(1, 0);
        let dv_dy = velocity_gradient(1, 1);
        let p_avg = 0.5 * (pressure[nodes[0]] + pressure[nodes[1]]);

        let sigma_xx = -p_avg + 2.0 * mu * du_dx;
        let sigma_xy = mu * (du_dy + dv_dx);
        let sigma_yy = -p_avg + 2.0 * mu * dv_dy;

        // Minus the outward-fluid traction gives the force on the solid.
        drag -= (sigma_xx * nx + sigma_xy * ny) * len;
        lift -= (sigma_xy * nx + sigma_yy * ny) * len;
    }

    assert!(
        drag.is_finite() && lift.is_finite(),
        "drag/lift integration produced a non-finite force"
    );
    (drag, lift)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembly;
    use crate::basis::PolynomialDegree;
    use crate::mesh::generators::{channel_with_cylinder_triangles, rectangular_mesh_triangles};
    use crate::mesh::{BoundaryType, ElementType, Mesh, Point, unit_square_triangles};
    use crate::spaces::{FunctionSpace, H1Space, VectorH1Space};
    use sonium_solvers::LinearOperator;

    #[test]
    fn pressure_gauge_respects_closed_and_natural_open_velocity_boundaries() {
        let mesh = crate::mesh::generators::box_mesh_tetrahedra_symmetric(
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2, 2, 2,
        );
        let velocity = VectorH1Space::new(PolynomialDegree::P1, 3);
        let pressure = H1Space::new(PolynomialDegree::P1);
        let system = crate::assembly::mixed::assemble_mixed(
            &super::super::forms::StokesViscousForm::new(0.0),
            &super::super::forms::StokesDivergenceForm,
            &velocity,
            &pressure,
            &mesh,
            2,
        );
        let coords = velocity.dof_coordinates(&mesh).unwrap();
        let side = coords
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                (p[0].abs() < 1e-12
                    || (p[0] - 1.0).abs() < 1e-12
                    || p[1].abs() < 1e-12
                    || (p[1] - 1.0).abs() < 1e-12)
                    .then_some(i)
            })
            .collect::<Vec<_>>();
        let closed = coords
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                p.iter()
                    .any(|x| x.abs() < 1e-12 || (*x - 1.0).abs() < 1e-12)
                    .then_some(i)
            })
            .collect::<Vec<_>>();
        assert!(!pressure_constant_mode_is_free(&system.block_01, &side));
        assert!(pressure_constant_mode_is_free(&system.block_01, &closed));
        let nv = velocity.num_global_dofs(&mesh);
        let np = pressure.num_global_dofs(&mesh);
        let observation = assemble_steady_newton_linearization(
            &mesh,
            &velocity,
            &pressure,
            &NavierStokesConfig::default(),
            &|_, _, _| vec![0.0; 3],
            &side,
            |_, _| vec![0.0; 3],
            |_, _| [[0.0; 3]; 3],
            &vec![0.0; nv],
            &vec![1.0; np],
        )
        .unwrap();
        // Uniform pressure has zero divergence/stabilization source. Its
        // nonzero natural-end traction remains in momentum, and its value
        // must not be overwritten by a fictitious interior pressure pin.
        assert!(observation.state[nv..].iter().all(|p| *p == 1.0));
        assert!(
            observation.state_residual[nv..]
                .iter()
                .all(|r| r.abs() < 1e-12)
        );
        assert!(
            observation.state_residual[..nv]
                .iter()
                .any(|r| r.abs() > 1e-3)
        );
    }

    #[test]
    fn transported_element_field_publishes_exact_fixed_history_bdf_tangent() {
        let mesh = crate::mesh::generators::box_mesh_tetrahedra_symmetric(
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2, 1, 1,
        );
        let velocity_space = VectorH1Space::new(PolynomialDegree::P1, 3);
        let pressure_space = H1Space::new(PolynomialDegree::P1);
        let mut config = NavierStokesConfig::default();
        config.mu = 0.2;
        config.rho = 1.0;
        let mut eddy_viscosities = vec![0.0; mesh.num_elements()];
        eddy_viscosities[0] = 0.5;
        config.transported_eddy_viscosity_per_element = Some(eddy_viscosities.clone());

        let bdf = crate::physics::fluid::stokes::time_stepping::BdfConfig::bdf2(0.1, 1);
        let state = crate::physics::fluid::stokes::time_stepping::BdfState::from_rest(
            velocity_space.num_global_dofs(&mesh),
            pressure_space.num_global_dofs(&mesh),
        );
        let workspace = crate::physics::fluid::stokes::workspace::BdfWorkspace::new(
            &mesh,
            &velocity_space,
            &pressure_space,
        );
        let exact = assemble_bdf_newton_linearization(
            &mesh,
            &velocity_space,
            &pressure_space,
            &config,
            &bdf,
            &state,
            &|_, _, _| vec![0.0; 3],
            &[],
            &|_| vec![0.0; 3],
            &|_| [[0.0; 3]; 3],
            &workspace,
            config.mu,
            None,
        )
        .expect("valid transported field should publish fixed-history Newton tangent");

        let mut uniform_config = config.clone();
        let mean_eddy_viscosity =
            eddy_viscosities.iter().sum::<f64>() / eddy_viscosities.len() as f64;
        uniform_config.transported_eddy_viscosity_per_element =
            Some(vec![mean_eddy_viscosity; mesh.num_elements()]);
        let uniform = assemble_bdf_newton_linearization(
            &mesh,
            &velocity_space,
            &pressure_space,
            &uniform_config,
            &bdf,
            &state,
            &|_, _, _| vec![0.0; 3],
            &[],
            &|_| vec![0.0; 3],
            &|_| [[0.0; 3]; 3],
            &workspace,
            uniform_config.mu,
            None,
        )
        .expect("uniform transported field should publish fixed-history tangent");

        assert_eq!(exact.0.len(), uniform.0.len());
        assert_eq!(exact.1.num_rows, uniform.1.num_rows);
        assert!(exact.1.values.iter().all(|value| value.is_finite()));
        assert!(
            exact
                .1
                .values
                .iter()
                .zip(&uniform.1.values)
                .any(|(left, right)| (left - right).abs() > 1.0e-12)
        );

        for invalid_field in [
            vec![0.0],
            vec![-1.0; mesh.num_elements()],
            vec![f64::NAN; mesh.num_elements()],
        ] {
            let mut invalid_config = config.clone();
            invalid_config.transported_eddy_viscosity_per_element = Some(invalid_field);
            assert!(
                assemble_bdf_newton_linearization(
                    &mesh,
                    &velocity_space,
                    &pressure_space,
                    &invalid_config,
                    &bdf,
                    &state,
                    &|_, _, _| vec![0.0; 3],
                    &[],
                    &|_| vec![0.0; 3],
                    &|_| [[0.0; 3]; 3],
                    &workspace,
                    invalid_config.mu,
                    None,
                )
                .is_none()
            );
        }

        let mut gradient_config = config.clone();
        gradient_config.transported_eddy_viscosity_per_element = None;
        for invalid_gradient in [
            vec![0.0],
            vec![f64::NAN; velocity_space.num_global_dofs(&mesh)],
        ] {
            assert!(
                assemble_bdf_newton_linearization(
                    &mesh,
                    &velocity_space,
                    &pressure_space,
                    &gradient_config,
                    &bdf,
                    &state,
                    &|_, _, _| vec![0.0; 3],
                    &[],
                    &|_| vec![0.0; 3],
                    &|_| [[0.0; 3]; 3],
                    &workspace,
                    gradient_config.mu,
                    Some(&invalid_gradient),
                )
                .is_none()
            );
        }

        let valid_gradient = vec![0.0; velocity_space.num_global_dofs(&mesh)];
        assert!(
            assemble_bdf_newton_linearization(
                &mesh,
                &velocity_space,
                &pressure_space,
                &config,
                &bdf,
                &state,
                &|_, _, _| vec![0.0; 3],
                &[],
                &|_| vec![0.0; 3],
                &|_| [[0.0; 3]; 3],
                &workspace,
                config.mu,
                Some(&valid_gradient),
            )
            .is_none()
        );
    }

    #[test]
    fn test_oseen_convection_form() {
        let mesh = unit_square_triangles(4);
        let space = VectorH1Space::new(PolynomialDegree::P1, 2);

        // Uniform velocity w = [1, 0]
        let form = OseenConvectionForm::new(1.0, |_| vec![1.0, 0.0]);
        let triplet = assembly::assemble_bilinear(&form, &space, &space, &mesh, 2);
        let csr = triplet.to_csr();

        // Convection matrix should be non-zero
        assert!(csr.nnz() > 0);
        // Row sums should not all be zero (asymmetric operator)
    }

    #[test]
    fn test_newton_convection_tangent_matches_centered_residual() {
        let mesh = unit_square_triangles(2);
        let space = VectorH1Space::new(PolynomialDegree::P1, 2);
        let form = NavierStokesConvectionForm::new(1.7, 2);
        let state: Vec<f64> = (0..space.num_global_dofs(&mesh))
            .map(|index| 0.13 + 0.017 * index as f64)
            .collect();
        let step = 1.0e-6;
        let (_, tangent) =
            crate::solver::nonlinear::assemble_nonlinear(&form, &space, &mesh, &state, 2);
        let tangent = tangent.to_csr();
        let mut maximum_relative_error: f64 = 0.0;

        for column in 0..state.len() {
            let mut plus = state.clone();
            let mut minus = state.clone();
            plus[column] += step;
            minus[column] -= step;
            let (plus_residual, _) =
                crate::solver::nonlinear::assemble_nonlinear(&form, &space, &mesh, &plus, 2);
            let (minus_residual, _) =
                crate::solver::nonlinear::assemble_nonlinear(&form, &space, &mesh, &minus, 2);
            for row in 0..state.len() {
                let finite_difference = (plus_residual[row] - minus_residual[row]) / (2.0 * step);
                let analytic = (tangent.row_ptrs[row]..tangent.row_ptrs[row + 1])
                    .find_map(|entry| {
                        (tangent.col_indices[entry] == column).then_some(tangent.values[entry])
                    })
                    .unwrap_or(0.0);
                let scale = analytic.abs().max(finite_difference.abs()).max(1.0);
                maximum_relative_error =
                    maximum_relative_error.max((analytic - finite_difference).abs() / scale);
            }
        }

        assert!(
            maximum_relative_error < 2.0e-7,
            "Newton convection tangent finite-difference error = {maximum_relative_error:e}"
        );
    }

    #[test]
    fn test_complete_laminar_newton_tangent_matches_stabilised_residual() {
        let mesh = unit_square_triangles(2);
        let v_space = VectorH1Space::new(PolynomialDegree::P1, 2);
        let p_space = H1Space::new(PolynomialDegree::P1);
        let tolerance = 1.0e-12;
        let bc_dofs = crate::boundary::locate_dofs_geometrical(&mesh, &v_space, |x, y, _| {
            x.abs() < tolerance
                || (x - 1.0).abs() < tolerance
                || y.abs() < tolerance
                || (y - 1.0).abs() < tolerance
        });
        let config = NavierStokesConfig {
            mu: 0.2,
            rho: 1.3,
            use_residual_stabilisation: true,
            use_direct_solver: true,
            max_picard_iterations: 1,
            min_picard_iterations: 1,
            picard_tolerance: 0.0,
            ..Default::default()
        };
        let body_force = |_: [f64; 3], velocity: [f64; 3]| {
            vec![0.4 - 0.2 * velocity[0], -0.3 + 0.15 * velocity[1]]
        };
        let body_force_jacobian =
            |_: [f64; 3], _: [f64; 3]| [[-0.2, 0.0, 0.0], [0.0, 0.15, 0.0], [0.0, 0.0, 0.0]];
        let n_v = v_space.num_global_dofs(&mesh);
        let n_p = p_space.num_global_dofs(&mesh);
        let pressure_pin = p_space
            .dof_coordinates(&mesh)
            .unwrap()
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| {
                let left_distance = (left[0] - 0.5).powi(2) + (left[1] - 0.5).powi(2);
                let right_distance = (right[0] - 0.5).powi(2) + (right[1] - 0.5).powi(2);
                left_distance.total_cmp(&right_distance)
            })
            .map(|(index, _)| index)
            .unwrap();
        let initial_state: Vec<f64> = (0..n_v + n_p)
            .map(|index| 0.08 + 0.013 * index as f64)
            .collect();

        let evaluate = |state: &[f64]| {
            solve_picard_with_velocity_force_and_jacobian_from_initial(
                &mesh,
                &v_space,
                &p_space,
                &config,
                &|_, _, _| vec![0.0, 0.0],
                &bc_dofs,
                body_force,
                body_force_jacobian,
                Some(&state[..n_v]),
                Some(&state[n_v..]),
            )
            .newton_linearization
            .expect("prescribed state must publish complete tangent")
        };
        let base = evaluate(&initial_state);
        let step = 1.0e-6;
        let mut maximum_relative_error: f64 = 0.0;
        let mut maximum_location = (0usize, 0usize, 0.0, 0.0);
        for column in 0..base.state.len() {
            if (column < n_v && bc_dofs.contains(&column)) || column == n_v + pressure_pin {
                continue;
            }
            let mut plus = base.state.clone();
            let mut minus = base.state.clone();
            plus[column] += step;
            minus[column] -= step;
            let plus_residual = evaluate(&plus).state_residual;
            let minus_residual = evaluate(&minus).state_residual;
            for row in 0..base.state.len() {
                let finite_difference = (plus_residual[row] - minus_residual[row]) / (2.0 * step);
                let analytic = (base.state_jacobian.row_ptrs[row]
                    ..base.state_jacobian.row_ptrs[row + 1])
                    .filter(|&entry| base.state_jacobian.col_indices[entry] == column)
                    .map(|entry| base.state_jacobian.values[entry])
                    .sum::<f64>();
                let scale = analytic.abs().max(finite_difference.abs()).max(1.0);
                let relative_error = (analytic - finite_difference).abs() / scale;
                if relative_error > maximum_relative_error {
                    maximum_relative_error = relative_error;
                    maximum_location = (row, column, analytic, finite_difference);
                }
            }
        }
        assert!(
            maximum_relative_error < 2.0e-6,
            "complete stabilised Newton tangent finite-difference error = {maximum_relative_error:e} at row={}, column={}, analytic={}, finite_difference={}",
            maximum_location.0,
            maximum_location.1,
            maximum_location.2,
            maximum_location.3,
        );
    }

    #[test]
    fn test_ns_config_reynolds() {
        let config = NavierStokesConfig::from_reynolds(100.0);
        assert!((config.mu - 0.01).abs() < 1e-10);
        assert!((config.rho - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cpu_krylov_result_fails_closed_on_nonconvergence() {
        assert!(finite_converged_solution(vec![1.0, 2.0], false).is_none());
        assert!(finite_converged_solution(vec![1.0, f64::NAN], true).is_none());
        assert_eq!(
            finite_converged_solution(vec![1.0, 2.0], true),
            Some(vec![1.0, 2.0])
        );
    }

    #[test]
    fn test_stabilised_picard_p1p1_zero_flow_runs() {
        let mesh = unit_square_triangles(2);
        let v_space = VectorH1Space::new(PolynomialDegree::P1, 2);
        let p_space = H1Space::new(PolynomialDegree::P1);
        let tolerance = 1.0e-12;
        let bc_dofs = crate::boundary::locate_dofs_geometrical(&mesh, &v_space, |x, y, _| {
            x.abs() < tolerance
                || (x - 1.0).abs() < tolerance
                || y.abs() < tolerance
                || (y - 1.0).abs() < tolerance
        });
        let config = NavierStokesConfig {
            max_picard_iterations: 2,
            picard_tolerance: 1.0e-12,
            use_direct_solver: true,
            ..Default::default()
        };

        let result = solve_picard(
            &mesh,
            &v_space,
            &p_space,
            &config,
            &|_, _, _| vec![0.0, 0.0],
            &bc_dofs,
            |_| vec![0.0, 0.0],
        );

        assert!(result.converged);
        assert!(result.velocity.iter().all(|value| value.is_finite()));
        assert!(result.pressure.iter().all(|value| value.is_finite()));
        let linearization = result
            .picard_linearization
            .as_ref()
            .expect("converged Picard solve should publish its last Oseen system");
        assert_eq!(
            linearization.state.len(),
            result.velocity.len() + result.pressure.len()
        );
        assert_eq!(
            linearization.fixed_point_residual.len(),
            linearization.state.len()
        );
        assert_eq!(
            linearization.state_jacobian.row_ptrs.len(),
            linearization.state_jacobian.num_rows + 1
        );
        assert!(
            linearization
                .state_jacobian
                .values
                .iter()
                .all(|value| value.is_finite())
        );

        let mut probe_config = config.clone();
        probe_config.max_picard_iterations = 1;
        probe_config.min_picard_iterations = 1;
        probe_config.picard_tolerance = 0.0;
        let probe = solve_picard_with_velocity_force_from_initial(
            &mesh,
            &v_space,
            &p_space,
            &probe_config,
            &|_, _, _| vec![0.0, 0.0],
            &bc_dofs,
            |_, _| vec![0.0, 0.0],
            Some(&result.velocity),
            Some(&result.pressure),
        );
        assert_eq!(probe.iterations, 1);
        assert!(probe.velocity.iter().all(|value| value.is_finite()));
        assert!(probe.pressure.iter().all(|value| value.is_finite()));
        let newton_linearization = probe
            .newton_linearization
            .as_ref()
            .expect("prescribed-state map should publish its Newton tangent");
        assert_eq!(
            newton_linearization.state.len(),
            newton_linearization.state_residual.len()
        );
        assert_eq!(
            newton_linearization.state.len(),
            newton_linearization.state_jacobian.num_rows
        );
        assert!(
            newton_linearization
                .state_jacobian
                .values
                .iter()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn test_stabilised_picard_p1p1_preserves_poiseuille_flow_direction() {
        let length = 4.0;
        let mesh = rectangular_mesh_triangles(0.0, length, 0.0, 1.0, 8, 4);
        let v_space = VectorH1Space::new(PolynomialDegree::P1, 2);
        let p_space = H1Space::new(PolynomialDegree::P1);
        let tolerance = 1.0e-12;
        let mut bc_dofs = crate::boundary::locate_dofs_geometrical(&mesh, &v_space, |x, _, _| {
            x.abs() < tolerance
        });
        bc_dofs.extend(crate::boundary::locate_dofs_geometrical(
            &mesh,
            &v_space,
            |_, y, _| y.abs() < tolerance || (y - 1.0).abs() < tolerance,
        ));
        bc_dofs.sort_unstable();
        bc_dofs.dedup();

        let config = NavierStokesConfig {
            mu: 0.1,
            rho: 1.0,
            max_picard_iterations: 100,
            picard_tolerance: 1.0e-6,
            picard_relaxation: 0.5,
            ..Default::default()
        };
        let result = solve_picard(
            &mesh,
            &v_space,
            &p_space,
            &config,
            &|_, y, _| vec![4.0 * y * (1.0 - y), 0.0],
            &bc_dofs,
            |_| vec![0.0, 0.0],
        );

        assert!(result.converged, "Poiseuille Picard solve did not converge");
        let outlet_center = mesh
            .nodes
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                ((a.x - length).powi(2) + (a.y - 0.5).powi(2))
                    .total_cmp(&((b.x - length).powi(2) + (b.y - 0.5).powi(2)))
            })
            .map(|(index, _)| index)
            .expect("mesh has an outlet-center node");
        let outlet_velocity = result.velocity[2 * outlet_center];
        assert!(
            outlet_velocity > 0.0,
            "pressure stabilisation reversed the outlet flow: u={outlet_velocity:.6}"
        );
    }

    #[test]
    fn test_stabilised_picard_is_consistent_for_affine_momentum_balance() {
        // Exact P1 solution of rho(u·grad)u - mu*Delta(u) + grad(p) = f:
        // u=(y,0), p=x, f=(1,0).  Every strong momentum residual is zero.
        // Residual-based SUPG/PSPG terms must therefore also cancel exactly.
        let mesh = unit_square_triangles(2);
        let v_space = VectorH1Space::new(PolynomialDegree::P1, 2);
        let p_space = H1Space::new(PolynomialDegree::P1);
        let tolerance = 1.0e-12;
        let bc_dofs = crate::boundary::locate_dofs_geometrical(&mesh, &v_space, |x, y, _| {
            x.abs() < tolerance
                || (x - 1.0).abs() < tolerance
                || y.abs() < tolerance
                || (y - 1.0).abs() < tolerance
        });
        let config = NavierStokesConfig {
            mu: 0.1,
            rho: 1.0,
            max_picard_iterations: 10,
            picard_tolerance: 1.0e-10,
            use_direct_solver: true,
            ..Default::default()
        };

        let result = solve_picard(
            &mesh,
            &v_space,
            &p_space,
            &config,
            &|_, y, _| vec![y, 0.0],
            &bc_dofs,
            |_| vec![1.0, 0.0],
        );

        assert!(
            result.converged,
            "affine manufactured solve did not converge"
        );
        for (node, point) in mesh.nodes.iter().enumerate() {
            assert!(
                (result.velocity[2 * node] - point.y).abs() < 1.0e-9,
                "u({},{})={} instead of {}",
                point.x,
                point.y,
                result.velocity[2 * node],
                point.y
            );
            assert!(result.velocity[2 * node + 1].abs() < 1.0e-9);
            // Pressure is only defined up to a constant. Compare against the
            // same numerical gauge instead of assuming the pinned DOF is at
            // x=0 (the solver intentionally pins near the centroid).
            let pressure_difference = result.pressure[node] - result.pressure[0];
            let exact_difference = point.x - mesh.nodes[0].x;
            assert!(
                (pressure_difference - exact_difference).abs() < 1.0e-9,
                "pressure difference at ({},{}) is {} instead of {}",
                point.x,
                point.y,
                pressure_difference,
                exact_difference
            );
        }
    }

    #[test]
    fn test_picard_applies_dirichlet_values_to_p2_edge_dofs() {
        let mesh = unit_square_triangles(1);
        let v_space = VectorH1Space::new(PolynomialDegree::P2, 2);
        let p_space = H1Space::new(PolynomialDegree::P1);
        let tolerance = 1.0e-12;
        let bc_dofs = crate::boundary::locate_dofs_geometrical(&mesh, &v_space, |x, y, _| {
            x.abs() < tolerance
                || (x - 1.0).abs() < tolerance
                || y.abs() < tolerance
                || (y - 1.0).abs() < tolerance
        });
        let config = NavierStokesConfig {
            mu: 0.1,
            rho: 1.0,
            max_picard_iterations: 10,
            picard_tolerance: 1.0e-10,
            use_direct_solver: true,
            ..Default::default()
        };

        let result = solve_picard(
            &mesh,
            &v_space,
            &p_space,
            &config,
            &|_, y, _| vec![y, 0.0],
            &bc_dofs,
            |_| vec![1.0, 0.0],
        );

        assert!(
            result.converged,
            "P2 affine manufactured solve did not converge"
        );
        let scalar_dofs = v_space.num_global_dofs(&mesh) / 2;
        for scalar_dof in 0..scalar_dofs {
            let point = v_space
                .dof_coordinate(&mesh, 2 * scalar_dof)
                .expect("P2 H1 DOF must be nodal");
            assert!(
                (result.velocity[2 * scalar_dof] - point[1]).abs() < 1.0e-9,
                "P2 u at {:?} is {} instead of {}",
                point,
                result.velocity[2 * scalar_dof],
                point[1]
            );
            assert!(result.velocity[2 * scalar_dof + 1].abs() < 1.0e-9);
        }
    }

    #[test]
    fn test_drag_lift_uses_outward_normal_and_full_viscous_stress() {
        // One fluid triangle above a horizontal solid boundary.  The boundary
        // node order is deliberately opposite to the element edge order so
        // force orientation cannot depend on imported boundary ordering.
        let mut mesh = Mesh::new(2);
        let n0 = mesh.add_node(Point::new_3d(0.0, 0.0, 0.0));
        let n1 = mesh.add_node(Point::new_3d(1.0, 0.0, 0.0));
        let n2 = mesh.add_node(Point::new_3d(0.0, 1.0, 0.0));
        let element = mesh.add_element(ElementType::Triangle, vec![n0, n1, n2]);
        mesh.add_boundary(vec![n1, n0], BoundaryType::Dirichlet, 7, element, 0);

        // u=(2y, 0), p=3, mu=0.5.  On the lower boundary the outward fluid
        // normal is (0,-1).  The force exerted by the fluid on the solid is
        // -sigma*n = (mu*2, -p) = (1,-3), integrated over a unit edge.
        let velocity = vec![0.0, 0.0, 0.0, 0.0, 2.0, 0.0];
        let pressure = vec![3.0; 3];
        let (drag, lift) = compute_drag_lift(&mesh, &velocity, &pressure, 0.5, 7);

        assert!((drag - 1.0).abs() < 1.0e-12, "drag={drag}");
        assert!((lift + 3.0).abs() < 1.0e-12, "lift={lift}");
    }

    #[test]
    fn test_drag_lift_evaluates_p2_velocity_gradient_at_boundary() {
        let mut mesh = Mesh::new(2);
        let n0 = mesh.add_node(Point::new_3d(0.0, 0.0, 0.0));
        let n1 = mesh.add_node(Point::new_3d(1.0, 0.0, 0.0));
        let n2 = mesh.add_node(Point::new_3d(0.0, 1.0, 0.0));
        let element = mesh.add_element(ElementType::Triangle, vec![n0, n1, n2]);
        mesh.add_boundary(vec![n0, n1], BoundaryType::Dirichlet, 7, element, 0);

        let space = VectorH1Space::new(PolynomialDegree::P2, 2);
        let mut velocity = vec![0.0; space.num_global_dofs(&mesh)];
        for scalar_dof in 0..velocity.len() / 2 {
            let point = space
                .dof_coordinate(&mesh, 2 * scalar_dof)
                .expect("P2 velocity DOF coordinate");
            velocity[2 * scalar_dof] = point[1] * point[1];
        }
        let pressure = vec![3.0; mesh.num_nodes()];

        // du/dy=0 on y=0 for u=y², so only pressure acts on this edge.
        let (drag, lift) = compute_drag_lift(&mesh, &velocity, &pressure, 0.5, 7);
        assert!(drag.abs() < 1.0e-12, "drag={drag}");
        assert!((lift + 3.0).abs() < 1.0e-12, "lift={lift}");
    }

    #[test]
    fn test_variational_drag_lift_recovers_affine_pressure_force() {
        let radius = 0.2;
        let mesh = channel_with_cylinder_triangles(-1.0, 1.0, -1.0, 1.0, 8, 8, 0.0, 0.0, radius);
        let velocity = vec![0.0; 2 * mesh.num_nodes()];
        let pressure: Vec<f64> = mesh.nodes.iter().map(|point| point.x).collect();

        // grad(p)=(1,0) is balanced by f=(1,0). The remaining weak residual
        // is the pressure traction on the polygonal obstacle. With N equal
        // circular chords its enclosed area is N*r²*sin(2π/N)/2.
        let (drag, lift) =
            compute_drag_lift_variational(&mesh, &velocity, &pressure, 0.0, 1.0, 4, |_| {
                vec![1.0, 0.0]
            });
        let segments = mesh
            .boundaries
            .iter()
            .filter(|boundary| boundary.marker == 4)
            .count();
        let polygon_area = 0.5
            * segments as f64
            * radius
            * radius
            * (std::f64::consts::TAU / segments as f64).sin();

        assert!((drag + polygon_area).abs() < 1.0e-10, "drag={drag}");
        assert!(lift.abs() < 1.0e-10, "lift={lift}");
    }

    #[test]
    fn test_picard_velocity_field_uses_element_local_barycentric_values() {
        let mut mesh = Mesh::new(3);
        let n0 = mesh.add_node(Point::new_3d(0.0, 0.0, 0.0));
        let n1 = mesh.add_node(Point::new_3d(1.0, 0.0, 0.0));
        let n2 = mesh.add_node(Point::new_3d(0.0, 1.0, 0.0));
        let n3 = mesh.add_node(Point::new_3d(0.0, 0.0, 1.0));
        mesh.add_element(ElementType::Tetrahedron, vec![n0, n1, n2, n3]);
        let v_space = VectorH1Space::new(PolynomialDegree::P1, 3);
        let topology = std::sync::Arc::new(
            crate::physics::fluid::stokes::workspace::ElementVelocityTopology::new(&mesh, &v_space),
        );
        let field = PicardVelocityField {
            topology,
            values: vec![
                0.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, //
                0.0, 2.0, 0.0, //
                0.0, 0.0, 3.0, //
            ],
            dim: 3,
        };

        let at = field.velocity_at([0.25, 0.25, 0.25]);

        assert!((at[0] - 0.25).abs() < 1.0e-12);
        assert!((at[1] - 0.50).abs() < 1.0e-12);
        assert!((at[2] - 0.75).abs() < 1.0e-12);
        let comps = field.velocity_components_at([0.25, 0.25, 0.25]);
        assert_eq!(comps.len(), 3);
        assert!(
            (field.velocity_magnitude_at([0.25, 0.25, 0.25]) - 0.25f64.hypot(0.50).hypot(0.75))
                .abs()
                < 1.0e-12
        );
    }

    #[test]
    fn test_vorticity_uniform_flow() {
        let mesh = unit_square_triangles(4);
        let n = mesh.num_nodes();
        // Uniform flow u = [1, 0] has zero vorticity
        let mut vel = vec![0.0; 2 * n];
        for i in 0..n {
            vel[2 * i] = 1.0; // u = 1
            vel[2 * i + 1] = 0.0; // v = 0
        }
        let omega = compute_vorticity_2d(&mesh, &vel);
        for &w in &omega {
            assert!(
                w.abs() < 1e-10,
                "Uniform flow should have zero vorticity: {w}"
            );
        }
    }

    /// Parity test: legacy CPU path (`solve_iterative_cpu`) and the new
    /// backend dispatch path (`solve_iterative_via_backend::<CpuBackend>`)
    /// must produce the same answer for the same linear system. Different
    /// preconditioners (ILU vs point-Jacobi) so iteration counts differ,
    /// but the converged solutions must agree to solver tolerance.
    #[test]
    fn test_iterative_dispatch_cpu_parity() {
        // Build A = 4 I − tridiag(−1, 0, −1) on n = 64 rows. Strictly
        // diagonally dominant; both ILU and Jacobi converge in a handful
        // of iters.
        let n = 64usize;
        let mut row_ptrs = Vec::with_capacity(n + 1);
        let mut col_indices: Vec<usize> = Vec::new();
        let mut values: Vec<f64> = Vec::new();
        row_ptrs.push(0);
        for i in 0..n {
            if i > 0 {
                col_indices.push(i - 1);
                values.push(-1.0);
            }
            col_indices.push(i);
            values.push(4.0);
            if i + 1 < n {
                col_indices.push(i + 1);
                values.push(-1.0);
            }
            row_ptrs.push(values.len());
        }
        let csr =
            sonium_solvers::CsrMatrix::<f64>::from_raw_parts(n, n, row_ptrs, col_indices, values);
        let b: Vec<f64> = vec![1.0_f64; n];

        let config = NavierStokesConfig {
            linear_solver_max_iterations: 200,
            linear_solver_tolerance: 1.0e-10,
            ..Default::default()
        };

        let (x_cpu, cpu_backend, _) = solve_iterative_cpu(&config, &csr, &b, None, n, None)
            .expect("legacy CPU GMRES must converge on diag-dominant system");
        assert!(!cpu_backend);

        let backend = sonium_compute::CpuBackend::new();
        let (x_dispatch, dispatched_backend, _) =
            solve_iterative_via_backend(&backend, &config, &csr, &b, None, n, None)
                .expect("backend GMRES must converge on diag-dominant system");
        assert!(dispatched_backend);

        assert_eq!(x_cpu.len(), n);
        assert_eq!(x_dispatch.len(), n);

        // Verify each solution residual is below tolerance, then check
        // they agree to looser numerical tolerance (different preconds).
        for xs in [&x_cpu, &x_dispatch] {
            let mut residual_norm_sq = 0.0_f64;
            let mut b_norm_sq = 0.0_f64;
            for i in 0..n {
                let mut row = 4.0 * xs[i];
                if i > 0 {
                    row -= xs[i - 1];
                }
                if i + 1 < n {
                    row -= xs[i + 1];
                }
                let r = 1.0 - row;
                residual_norm_sq += r * r;
                b_norm_sq += 1.0;
            }
            let rel = (residual_norm_sq / b_norm_sq).sqrt();
            assert!(rel < 1.0e-8, "residual {rel:.3e} exceeds tolerance");
        }

        for i in 0..n {
            let diff = (x_cpu[i] - x_dispatch[i]).abs();
            assert!(
                diff < 1.0e-6,
                "row {i}: cpu={} dispatch={} diff={diff:.3e}",
                x_cpu[i],
                x_dispatch[i]
            );
        }
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn requested_cuda_without_feature_reports_cpu_fallback() {
        let dense =
            sonium_solvers::dense::DenseMatrix::from_row_major_vec(2, 2, vec![2.0, 0.0, 0.0, 4.0]);
        let csr = sonium_solvers::CsrMatrix::from_dense(&dense, 1.0e-14);
        let b = [4.0, 8.0];
        let config = NavierStokesConfig {
            linear_solver_backend: LinearSolverBackend::Cuda,
            linear_solver_cpu_preconditioner: LinearSolverCpuPreconditionerKind::Ilu,
            linear_solver_max_iterations: 20,
            linear_solver_tolerance: 1.0e-10,
            ..Default::default()
        };

        let (solution, actual_backend, _) =
            solve_iterative_linear_system(&config, &csr, &b, None, 2, None)
                .expect("CPU fallback should solve the diagonal system");

        assert_eq!(actual_backend, LinearSolverBackend::Cpu);
        assert!((solution[0] - 2.0).abs() < 1.0e-10);
        assert!((solution[1] - 2.0).abs() < 1.0e-10);
    }

    #[test]
    fn test_cpu_jacobi_schur_preconditioner_solves_small_saddle_system() {
        let dense = sonium_solvers::dense::DenseMatrix::from_row_major_vec(
            4,
            4,
            vec![
                4.0_f64, 0.0, 1.0, 0.0, //
                0.0, 4.0, 0.0, 1.0, //
                1.0, 0.0, 0.1, 0.0, //
                0.0, 1.0, 0.0, 0.1,
            ],
        );
        let csr = sonium_solvers::CsrMatrix::from_dense(&dense, 1.0e-14);
        let schur_dense = sonium_solvers::dense::DenseMatrix::from_row_major_vec(
            2,
            2,
            vec![1.0_f64, 0.0, 0.0, 1.0],
        );
        let schur = sonium_solvers::CsrMatrix::from_dense(&schur_dense, 1.0e-14);
        let b: Vec<f64> = vec![1.0_f64, 2.0, 0.5, -0.25];
        let config = NavierStokesConfig {
            linear_solver_cpu_preconditioner: LinearSolverCpuPreconditionerKind::JacobiSchur,
            linear_solver_max_iterations: 100,
            linear_solver_tolerance: 1.0e-10,
            ..Default::default()
        };

        let (x, cpu_backend, _) = solve_iterative_cpu(&config, &csr, &b, None, 2, Some(&schur))
            .expect("Jacobi/Schur CPU preconditioned solve should return finite solution");
        assert!(!cpu_backend);
        let ax = csr.apply(&x);
        let residual = b
            .iter()
            .zip(ax.iter())
            .map(|(bi, axi)| bi - axi)
            .collect::<Vec<_>>();
        let residual_norm = residual.iter().map(|v| v * v).sum::<f64>().sqrt();

        assert!(
            residual_norm < 1.0e-8,
            "Jacobi/Schur residual too large: {residual_norm}"
        );
    }

    /// Jittered-grid Delaunay channel mesh with classified boundary markers.
    ///
    /// Returns `(mesh, inlet_marker, outlet_marker)` for the plane channel
    /// `[0, length] × [0, height]` on an `nx`×`ny` background grid. Interior
    /// nodes are perturbed deterministically and triangulated with Delaunay
    /// connectivity: a perfectly structured P1–P1 grid carries an exact
    /// checkerboard null mode in Bᵀ, which makes the monolithic BDF saddle
    /// matrix singular and the sparse LU factorization fail. Markers are
    /// classified by node coordinates: 1 = inlet (x = 0), 2 = outlet
    /// (x = length), 3 = walls (y = 0 and y = height).
    fn marked_channel_mesh(length: f64, height: f64, nx: usize, ny: usize) -> (Mesh, i32, i32) {
        let dx = length / nx as f64;
        let dy = height / ny as f64;
        let mut points = Vec::with_capacity((nx + 1) * (ny + 1));
        for j in 0..=ny {
            for i in 0..=nx {
                let interior = i > 0 && i < nx && j > 0 && j < ny;
                let (mut x, mut y) = (i as f64 * dx, j as f64 * dy);
                if interior {
                    // Deterministic hash jitter (±20% of a cell) so the test
                    // mesh is reproducible but unstructured.
                    let h1 = (i as u64)
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add((j as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
                    let h2 = (j as u64)
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add((i as u64).wrapping_mul(0x1656_67B1_9E37_79F9));
                    x += ((h1 >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.4 * dx;
                    y += ((h2 >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.4 * dy;
                }
                points.push((x, y));
            }
        }
        let triangles = crate::mesh::triangulate::triangulate_points_2d(&points);
        let mut mesh = Mesh::new(2);
        for &(x, y) in &points {
            mesh.add_node(Point::new_2d(x, y));
        }
        for triangle in triangles {
            mesh.add_element(ElementType::Triangle, triangle.to_vec());
        }
        mesh.detect_boundaries();
        let tol = 1.0e-9;
        mesh.set_boundary_condition(BoundaryType::Neumann, 1, |pts| {
            pts.iter().all(|p| p.x.abs() < tol)
        });
        mesh.set_boundary_condition(BoundaryType::Neumann, 2, |pts| {
            pts.iter().all(|p| (p.x - length).abs() < tol)
        });
        mesh.set_boundary_condition(BoundaryType::Dirichlet, 3, |pts| {
            pts.iter()
                .all(|p| p.y.abs() < tol || (p.y - height).abs() < tol)
        });
        (mesh, 1, 2)
    }

    /// Pressure-driven plane channel (Hagen–Poiseuille): inlet and outlet are
    /// free of Dirichlet conditions and the flow is driven purely by pressure
    /// tractions. Runs 80 BDF2 steps (Δt = 0.125, t_end = 10) to steady state,
    /// then compares the mid-height centerline velocity against the
    /// plane-Poiseuille value u_max = Δp·h²/(8·μ·L).
    #[test]
    #[ignore = "full 80-step Poiseuille validation; run via `just qa-nightly-regressions`"]
    fn pressure_driven_channel_reaches_poiseuille() {
        use crate::physics::fluid::stokes::time_stepping::{BdfConfig, BdfState};
        use crate::physics::fluid::stokes::traction::PressureTraction;

        let length = 2.0;
        let height = 0.1;
        let ny = 16;
        let (mesh, inlet, outlet) = marked_channel_mesh(length, height, 20, ny);
        let v_space = VectorH1Space::new(PolynomialDegree::P1, 2);
        let p_space = H1Space::new(PolynomialDegree::P1);
        let n_v = v_space.num_global_dofs(&mesh);
        let n_p = p_space.num_global_dofs(&mesh);

        // No-slip walls only; inlet/outlet stay traction-driven.
        let tolerance = 1.0e-9;
        let mut bc_dofs = crate::boundary::locate_dofs_geometrical(&mesh, &v_space, |_, y, _| {
            y.abs() < tolerance || (y - height).abs() < tolerance
        });
        bc_dofs.sort_unstable();
        bc_dofs.dedup();
        let bc_fn = |_: f64, _: f64, _: f64| vec![0.0, 0.0];

        let mu = 1.0e-3;
        let config = NavierStokesConfig {
            mu,
            rho: 1.0,
            use_residual_stabilisation: true,
            ..Default::default()
        };
        let n_steps = 80;
        let bdf = BdfConfig::bdf2(0.125, n_steps);
        let mut state = BdfState::from_rest(n_v, n_p);
        let tractions = [
            PressureTraction {
                marker: inlet,
                pressure_pa: 1.0,
            },
            PressureTraction {
                marker: outlet,
                pressure_pa: 0.0,
            },
        ];
        for step in 0..n_steps {
            assert!(
                solve_bdf_step_with_traction(
                    &mesh, &v_space, &p_space, &config, &bdf, &mut state, &bc_fn, &bc_dofs,
                    &tractions,
                ),
                "BDF step {step} failed"
            );
        }

        let dp = 1.0;
        let u_max_exact = dp * height * height / (8.0 * mu * length);
        // Sample u_x at interior nodes near the centreline, away from the
        // inlet/outlet where the profile is still developing.
        let dy = height / ny as f64;
        let u_max_num = mesh
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, p)| {
                (p.y - 0.5 * height).abs() < 0.3 * dy && p.x > 0.25 * length && p.x < 0.75 * length
            })
            .map(|(i, _)| state.velocity[2 * i])
            .fold(0.0_f64, f64::max);
        let rel = (u_max_num - u_max_exact).abs() / u_max_exact;
        assert!(
            rel < 0.05,
            "Poiseuille u_max: exact {u_max_exact}, got {u_max_num} (rel {rel:.4})"
        );
    }

    /// Taylor–Hood P2–P1 variant of [`pressure_driven_channel_reaches_poiseuille`]:
    /// verify that pressure traction on the inlet performs the exact work on
    /// the quadratic Poiseuille profile represented by the P2 velocity trace.
    #[test]
    fn pressure_driven_channel_reaches_poiseuille_p2_p1() {
        use crate::physics::fluid::stokes::traction::PressureTraction;

        let length = 2.0;
        let height = 0.1;
        let (mesh, inlet, _) = marked_channel_mesh(length, height, 20, 16);
        let v_space = VectorH1Space::new(PolynomialDegree::P2, 2);
        let p_space = H1Space::new(PolynomialDegree::P1);
        let n_v = v_space.num_global_dofs(&mesh);
        assert!(p_space.num_global_dofs(&mesh) > 0);

        let mu = 1.0e-3;
        let dp = 0.01;
        let u_max_exact = dp * height * height / (8.0 * mu * length);
        let dof_coords = v_space
            .dof_coordinates(&mesh)
            .expect("VectorH1Space is nodal");
        let poiseuille_velocity: Vec<f64> = dof_coords
            .iter()
            .enumerate()
            .map(|(dof, p)| {
                if dof % 2 == 0 {
                    let y = p[1];
                    4.0 * u_max_exact * y * (height - y) / (height * height)
                } else {
                    0.0
                }
            })
            .collect();
        let mut traction_rhs = vec![0.0; n_v];
        crate::physics::fluid::stokes::traction::assemble_pressure_tractions_for_space(
            &mesh,
            &v_space,
            &[PressureTraction {
                marker: inlet,
                pressure_pa: dp,
            }],
            &mut traction_rhs,
        );
        let work = traction_rhs
            .iter()
            .zip(&poiseuille_velocity)
            .map(|(force, velocity)| force * velocity)
            .sum::<f64>();
        let flux_exact = 2.0 * u_max_exact * height / 3.0;
        let work_exact = dp * flux_exact;
        let rel = (work - work_exact).abs() / work_exact;
        assert!(
            rel < 1.0e-12,
            "P2-P1 Poiseuille traction work: exact {work_exact}, got {work} (rel {rel:.4})"
        );
    }

    /// Cached vs uncached BDF stepping must agree: `BdfWorkspace` hoists the
    /// constant mass / unit-μ viscous / divergence assemblies out of the time
    /// loop, so `solve_bdf_step_cached` must reproduce `solve_bdf_step`
    /// step for step. Uses the jittered (unstructured) channel mesh: a
    /// structured P1-P1 grid carries a checkerboard null mode in Bᵀ that
    /// makes the BDF saddle matrix singular. The flow is driven by a moving
    /// top wall (walls-only Dirichlet); adding inlet Dirichlet on top of the
    /// wall BCs makes this P1-P1 saddle matrix exactly singular for the
    /// sparse LU (pre-existing quirk, also present without the cache).
    #[test]
    fn bdf_convection_uses_exact_element_velocity_not_nearest_support() {
        use crate::physics::fluid::stokes::workspace::BdfWorkspace;

        let mesh = crate::mesh::unit_square_triangles(1);
        let v_space = VectorH1Space::new(PolynomialDegree::P2, 2);
        let p_space = H1Space::new(PolynomialDegree::P1);
        let workspace = BdfWorkspace::new(&mesh, &v_space, &p_space);
        let affine = |p: [f64; 3]| {
            [
                1.0 + 2.0 * p[0] - 3.0 * p[1],
                -2.0 + 0.5 * p[0] + 4.0 * p[1],
            ]
        };
        let u_star: Vec<f64> = (0..v_space.num_global_dofs(&mesh))
            .map(|dof| affine(v_space.dof_coordinate(&mesh, dof).unwrap())[dof % 2])
            .collect();

        let actual = assemble_bdf_convection(&mesh, &v_space, &workspace, &u_star, 1.0, 2).to_csr();
        let exact = OseenConvectionForm::new(1.0, |point| affine(point).to_vec());
        let exact = crate::assembly::assemble_bilinear(
            &exact,
            &v_space,
            &v_space,
            &mesh,
            workspace.quad_order,
        )
        .to_csr();

        let support_coords: Vec<[f64; 3]> = workspace
            .velocity_dof_coordinates
            .chunks(2)
            .map(|pair| pair[0])
            .collect();
        let nearest = OseenConvectionForm::new(1.0, |point: [f64; 3]| {
            let index = support_coords
                .iter()
                .enumerate()
                .min_by(|(ia, a), (ib, b)| {
                    let da = (0..3)
                        .map(|axis| (a[axis] - point[axis]).powi(2))
                        .sum::<f64>();
                    let db = (0..3)
                        .map(|axis| (b[axis] - point[axis]).powi(2))
                        .sum::<f64>();
                    da.total_cmp(&db).then(ia.cmp(ib))
                })
                .unwrap()
                .0;
            u_star[index * 2..index * 2 + 2].to_vec()
        });
        let nearest = crate::assembly::assemble_bilinear(
            &nearest,
            &v_space,
            &v_space,
            &mesh,
            workspace.quad_order,
        )
        .to_csr();

        let dense = |matrix: &sonium_solvers::sparse::CsrMatrix<f64>| {
            let mut values = vec![0.0; matrix.num_rows * matrix.num_cols];
            for row in 0..matrix.num_rows {
                for (col, value) in matrix.row_entries(row) {
                    values[row * matrix.num_cols + col] = value;
                }
            }
            values
        };
        let actual = dense(&actual);
        let exact = dense(&exact);
        let nearest = dense(&nearest);
        let exact_error = actual
            .iter()
            .zip(&exact)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let legacy_difference = actual
            .iter()
            .zip(&nearest)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            exact_error < 1.0e-12,
            "exact element convection error {exact_error:e}"
        );
        assert!(
            legacy_difference > 1.0e-4,
            "fixture did not distinguish nearest-support convection: {legacy_difference:e}"
        );
    }

    #[test]
    fn bdf_energy_audit_closes_rest_state() {
        use crate::physics::fluid::stokes::time_stepping::{BdfConfig, BdfState};
        use crate::physics::fluid::stokes::workspace::BdfWorkspace;

        let mesh = crate::mesh::unit_square_triangles(2);
        let velocity_space = VectorH1Space::new(PolynomialDegree::P2, 2);
        let pressure_space = H1Space::new(PolynomialDegree::P1);
        let workspace = BdfWorkspace::new(&mesh, &velocity_space, &pressure_space);
        let state = BdfState::from_rest(workspace.n_v, workspace.n_p);
        let audit = audit_bdf_step(
            &mesh,
            &velocity_space,
            &NavierStokesConfig::default(),
            &BdfConfig::bdf2(1.0e-3, 1),
            &state,
            &state,
            &[],
            1,
            2,
            &workspace,
        );
        assert_eq!(audit.fluid_kinetic_old_j, 0.0);
        assert_eq!(audit.fluid_kinetic_new_j, 0.0);
        assert_eq!(audit.bdf_mass_work_j, 0.0);
        assert_eq!(audit.viscous_dissipation_j, 0.0);
        assert_eq!(audit.convection_export_j, 0.0);
        assert_eq!(audit.pressure_work_j, 0.0);
        assert_eq!(audit.traction_work_j, 0.0);
        assert_eq!(audit.solve_residual_j, 0.0);
    }

    #[test]
    fn cached_bdf_body_force_applies_rhs_and_analytic_jacobian() {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };

        use crate::physics::fluid::stokes::time_stepping::{BdfConfig, BdfState};
        use crate::physics::fluid::stokes::workspace::BdfWorkspace;

        let mesh = unit_square_triangles(2);
        let velocity_space = VectorH1Space::new(PolynomialDegree::P2, 2);
        let pressure_space = H1Space::new(PolynomialDegree::P1);
        let workspace = BdfWorkspace::new(&mesh, &velocity_space, &pressure_space);
        let n_v = workspace.n_v;
        let n_p = workspace.n_p;
        let boundary_tolerance = 1.0e-12;
        let mut bc_dofs =
            crate::boundary::locate_dofs_geometrical(&mesh, &velocity_space, |_, y, _| {
                y.abs() < boundary_tolerance || (y - 1.0).abs() < boundary_tolerance
            });
        bc_dofs.sort_unstable();
        bc_dofs.dedup();

        let config = NavierStokesConfig {
            mu: 1.0e-2,
            rho: 1.0,
            use_residual_stabilisation: true,
            ..NavierStokesConfig::default()
        };
        let bdf = BdfConfig::bdf1(0.1, 1);
        let bc_fn = |_: f64, _: f64, _: f64| vec![0.0, 0.0];
        let force = |_| vec![0.0, 1.0];

        let solve = |jacobian: Box<dyn Fn([f64; 3]) -> [[f64; 3]; 3] + Send + Sync>,
                     state: &mut BdfState| {
            assert!(solve_bdf_step_cached_with_force(
                &mesh,
                &velocity_space,
                &pressure_space,
                &config,
                &bdf,
                state,
                &bc_fn,
                &bc_dofs,
                &[],
                force,
                jacobian,
                &workspace,
            ));
        };

        let mut undamped = BdfState::from_rest(n_v, n_p);
        solve(Box::new(|_| [[0.0; 3]; 3]), &mut undamped);
        let undamped_peak = undamped
            .velocity
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        assert!(undamped_peak > 1.0e-8, "body force did not enter the RHS");

        let jacobian_calls = Arc::new(AtomicUsize::new(0));
        let calls = Arc::clone(&jacobian_calls);
        let mut damped = BdfState::from_rest(n_v, n_p);
        solve(
            Box::new(move |_| {
                calls.fetch_add(1, Ordering::Relaxed);
                [[0.0, 0.0, 0.0], [0.0, -10.0, 0.0], [0.0, 0.0, 0.0]]
            }),
            &mut damped,
        );
        let damped_peak = damped
            .velocity
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);

        assert!(jacobian_calls.load(Ordering::Relaxed) > 0);
        assert!(
            damped_peak < undamped_peak,
            "analytic damping Jacobian was not applied: undamped={undamped_peak}, damped={damped_peak}"
        );
    }

    #[test]
    fn bdf1_energy_audit_mass_work_has_step_integrated_units() {
        use crate::physics::fluid::stokes::time_stepping::{BdfConfig, BdfState};
        use crate::physics::fluid::stokes::workspace::BdfWorkspace;

        let mesh = crate::mesh::unit_square_triangles(1);
        let velocity_space = VectorH1Space::new(PolynomialDegree::P2, 2);
        let pressure_space = H1Space::new(PolynomialDegree::P1);
        let workspace = BdfWorkspace::new(&mesh, &velocity_space, &pressure_space);
        let mut previous = BdfState::from_rest(workspace.n_v, workspace.n_p);
        let mut accepted = previous.clone();
        for (i, value) in previous.velocity.iter_mut().enumerate() {
            *value = 0.01 * (i as f64 + 1.0);
        }
        for (i, value) in accepted.velocity.iter_mut().enumerate() {
            *value = previous.velocity[i] + 0.002 * ((i % 3) as f64 - 1.0);
        }

        let rho = 1.7;
        let config = NavierStokesConfig {
            rho,
            ..Default::default()
        };
        let audit_at = |dt| {
            audit_bdf_step(
                &mesh,
                &velocity_space,
                &config,
                &BdfConfig::bdf1(dt, 1),
                &previous,
                &accepted,
                &[],
                1,
                2,
                &workspace,
            )
        };
        let audit = audit_at(1.0e-3);
        let delta: Vec<_> = accepted
            .velocity
            .iter()
            .zip(&previous.velocity)
            .map(|(new, old)| new - old)
            .collect();
        let expected_mass_work =
            rho * triplet_bilinear(&accepted.velocity, &workspace.mass_triplet, &delta);
        let expected_numerical =
            0.5 * rho * triplet_bilinear(&delta, &workspace.mass_triplet, &delta);

        assert!((audit.bdf_mass_work_j - expected_mass_work).abs() < 1.0e-14);
        assert!((audit.bdf_numerical_j - expected_numerical).abs() < 1.0e-14);
        assert!(expected_numerical > 0.0);
        assert!(
            (audit_at(2.5e-4).bdf_mass_work_j - audit.bdf_mass_work_j).abs() < 1.0e-14,
            "the step-integrated BDF mass work must not retain the 1/dt from the momentum equation"
        );
    }

    #[test]
    fn cached_workspace_matches_uncached_steps() {
        use crate::physics::fluid::stokes::time_stepping::{BdfConfig, BdfState};
        use crate::physics::fluid::stokes::workspace::BdfWorkspace;

        let length = 1.0;
        let height = 0.1;
        let (mesh, _, _) = marked_channel_mesh(length, height, 12, 8);
        let v_space = VectorH1Space::new(PolynomialDegree::P1, 2);
        let p_space = H1Space::new(PolynomialDegree::P1);
        let n_v = v_space.num_global_dofs(&mesh);
        let n_p = p_space.num_global_dofs(&mesh);

        // Walls-only Dirichlet (moving top wall); inlet/outlet traction-free.
        let tolerance = 1.0e-9;
        let mut bc_dofs = crate::boundary::locate_dofs_geometrical(&mesh, &v_space, |_, y, _| {
            y.abs() < tolerance || (y - height).abs() < tolerance
        });
        bc_dofs.sort_unstable();
        bc_dofs.dedup();
        let bc_fn = move |_: f64, y: f64, _: f64| {
            if (y - height).abs() < tolerance {
                vec![1.0, 0.0]
            } else {
                vec![0.0, 0.0]
            }
        };

        let config = NavierStokesConfig {
            mu: 1.0e-3,
            rho: 1.0,
            ..Default::default()
        };
        let n_steps = 20;
        let bdf = BdfConfig::bdf2(0.05, n_steps);

        let mut state_a = BdfState::from_rest(n_v, n_p);
        for step in 0..n_steps {
            assert!(
                solve_bdf_step(
                    &mesh,
                    &v_space,
                    &p_space,
                    &config,
                    &bdf,
                    &mut state_a,
                    &bc_fn,
                    &bc_dofs,
                ),
                "uncached BDF step {step} failed"
            );
        }

        let ws = BdfWorkspace::new(&mesh, &v_space, &p_space);
        let mut state_b = BdfState::from_rest(n_v, n_p);
        for step in 0..n_steps {
            assert!(
                solve_bdf_step_cached(
                    &mesh,
                    &v_space,
                    &p_space,
                    &config,
                    &bdf,
                    &mut state_b,
                    &bc_fn,
                    &bc_dofs,
                    &[],
                    &ws,
                ),
                "cached BDF step {step} failed"
            );
        }

        // Sanity: the moving lid must have produced a nonzero flow field.
        assert!(
            state_a.velocity.iter().any(|v| v.abs() > 1.0e-3),
            "fixture produced a trivial zero flow"
        );
        for (a, b) in state_a.velocity.iter().zip(&state_b.velocity) {
            assert!((a - b).abs() < 1e-12, "velocity mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn cached_bdf_direct_and_jacobi_schur_match_and_reuse_affine_setup() {
        use crate::physics::fluid::stokes::time_stepping::{BdfConfig, BdfState};
        use crate::physics::fluid::stokes::traction::PressureTraction;
        use crate::physics::fluid::stokes::workspace::{BdfLinearSolverKind, BdfWorkspace};

        let (mesh, inlet, outlet) = marked_channel_mesh(1.0, 0.1, 8, 4);
        let v_space = VectorH1Space::new(PolynomialDegree::P2, 2);
        let p_space = H1Space::new(PolynomialDegree::P1);
        let n_v = v_space.num_global_dofs(&mesh);
        let n_p = p_space.num_global_dofs(&mesh);
        let mut bc_dofs = v_space.boundary_dofs(&mesh, 3);
        bc_dofs.sort_unstable();
        bc_dofs.dedup();
        let bc_fn = |_: f64, _: f64, _: f64| vec![0.0, 0.0];
        let bdf = BdfConfig::bdf2(1.0e-3, 2);
        let base = NavierStokesConfig {
            mu: 1.0e-3,
            rho: 1.0,
            use_residual_stabilisation: false,
            linear_solver_max_iterations: 5000,
            linear_solver_tolerance: 1.0e-10,
            linear_solver_backend: LinearSolverBackend::Cpu,
            ..Default::default()
        };
        let direct_config = NavierStokesConfig {
            use_direct_solver: true,
            ..base.clone()
        };
        let iterative_config = NavierStokesConfig {
            use_direct_solver: false,
            linear_solver_cpu_preconditioner: LinearSolverCpuPreconditionerKind::JacobiSchur,
            ..base
        };
        let direct_workspace = BdfWorkspace::new(&mesh, &v_space, &p_space);
        let iterative_workspace = BdfWorkspace::new(&mesh, &v_space, &p_space);
        let initial = BdfState::from_rest(n_v, n_p);

        let solve_probe =
            |config: &NavierStokesConfig, workspace: &BdfWorkspace, pressure_pa: f64| {
                let mut state = initial.clone();
                let tractions = [
                    PressureTraction {
                        marker: inlet,
                        pressure_pa,
                    },
                    PressureTraction {
                        marker: outlet,
                        pressure_pa: 0.0,
                    },
                ];
                assert!(
                    solve_bdf_step_cached(
                        &mesh, &v_space, &p_space, config, &bdf, &mut state, &bc_fn, &bc_dofs,
                        &tractions, workspace,
                    ),
                    "{} probe solve failed",
                    if config.use_direct_solver {
                        "direct"
                    } else {
                        "iterative"
                    }
                );
                (state, workspace.last_linear_solve_telemetry().unwrap())
            };

        let (direct, direct_first) = solve_probe(&direct_config, &direct_workspace, 1.0);
        let (_, direct_second) = solve_probe(&direct_config, &direct_workspace, 2.0);
        let (iterative, iterative_first) =
            solve_probe(&iterative_config, &iterative_workspace, 1.0);
        let (_, iterative_second) = solve_probe(&iterative_config, &iterative_workspace, 2.0);

        assert_eq!(direct_first.solver, BdfLinearSolverKind::Direct);
        assert_eq!(
            direct_first.resolved_linear_solver_backend,
            LinearSolverBackend::Cpu
        );
        assert!(!direct_first.setup_reused);
        assert!(direct_second.setup_reused);
        assert_eq!(
            iterative_first.solver,
            BdfLinearSolverKind::FgmresJacobiSchur
        );
        assert_eq!(
            iterative_first.resolved_linear_solver_backend,
            LinearSolverBackend::Cpu
        );
        assert!(!iterative_first.setup_reused);
        assert!(iterative_second.setup_reused);
        assert!(direct_first.true_relative_residual <= 1.0e-9);
        assert!(iterative_first.true_relative_residual <= 1.0e-9);

        let velocity_error = direct
            .velocity
            .iter()
            .zip(&iterative.velocity)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let velocity_scale = direct
            .velocity
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt()
            .max(f64::MIN_POSITIVE);
        assert!(velocity_error / velocity_scale <= 1.0e-7);

        let direct_mean = direct.pressure.iter().sum::<f64>() / n_p as f64;
        let iterative_mean = iterative.pressure.iter().sum::<f64>() / n_p as f64;
        let pressure_error = direct
            .pressure
            .iter()
            .zip(&iterative.pressure)
            .map(|(a, b)| ((a - direct_mean) - (b - iterative_mean)).powi(2))
            .sum::<f64>()
            .sqrt();
        let pressure_scale = direct
            .pressure
            .iter()
            .map(|value| (value - direct_mean).powi(2))
            .sum::<f64>()
            .sqrt()
            .max(f64::MIN_POSITIVE);
        assert!(pressure_error / pressure_scale <= 1.0e-6);
    }

    #[test]
    fn jacobi_schur_converges_mass_dominated_unsteady_step() {
        use crate::physics::fluid::stokes::time_stepping::{BdfConfig, BdfState};
        use crate::physics::fluid::stokes::traction::PressureTraction;
        use crate::physics::fluid::stokes::workspace::BdfWorkspace;

        // Port-flow regime (speaker-port 3D failure): the momentum block is
        // mass-dominated (rho/dt >> mu/h^2), so a steady-Stokes Mp/mu Schur
        // approximation is off by orders of magnitude and FGMRES stalls. The
        // Schur approximation must account for the unsteady mass term.
        let (mesh, inlet, outlet) = marked_channel_mesh(1.0, 0.1, 32, 16);
        let v_space = VectorH1Space::new(PolynomialDegree::P2, 2);
        let p_space = H1Space::new(PolynomialDegree::P1);
        let n_v = v_space.num_global_dofs(&mesh);
        let n_p = p_space.num_global_dofs(&mesh);
        let mut bc_dofs = v_space.boundary_dofs(&mesh, 3);
        bc_dofs.sort_unstable();
        bc_dofs.dedup();
        let bc_fn = |_: f64, _: f64, _: f64| vec![0.0, 0.0];
        let bdf = BdfConfig::bdf2(8.844870228e-5, 2);
        let base = NavierStokesConfig {
            mu: 1.85e-5,
            rho: 1.2,
            use_residual_stabilisation: false,
            linear_solver_max_iterations: 5000,
            linear_solver_tolerance: 1.0e-8,
            linear_solver_backend: LinearSolverBackend::Cpu,
            ..Default::default()
        };
        let direct_config = NavierStokesConfig {
            use_direct_solver: true,
            ..base.clone()
        };
        let iterative_config = NavierStokesConfig {
            use_direct_solver: false,
            linear_solver_cpu_preconditioner: LinearSolverCpuPreconditionerKind::JacobiSchur,
            ..base
        };
        let direct_workspace = BdfWorkspace::new(&mesh, &v_space, &p_space);
        let iterative_workspace = BdfWorkspace::new(&mesh, &v_space, &p_space);
        let tractions = [
            PressureTraction {
                marker: inlet,
                pressure_pa: 1.0,
            },
            PressureTraction {
                marker: outlet,
                pressure_pa: 0.0,
            },
        ];
        let mut direct = BdfState::from_rest(n_v, n_p);
        assert!(solve_bdf_step_cached(
            &mesh,
            &v_space,
            &p_space,
            &direct_config,
            &bdf,
            &mut direct,
            &bc_fn,
            &bc_dofs,
            &tractions,
            &direct_workspace,
        ));
        let mut iterative = BdfState::from_rest(n_v, n_p);
        assert!(solve_bdf_step_cached(
            &mesh,
            &v_space,
            &p_space,
            &iterative_config,
            &bdf,
            &mut iterative,
            &bc_fn,
            &bc_dofs,
            &tractions,
            &iterative_workspace,
        ));
        let telemetry = iterative_workspace
            .last_linear_solve_telemetry()
            .expect("telemetry recorded");
        assert!(telemetry.true_relative_residual <= 1.0e-8);
        // Regression bound for the speaker-port fine-3D stall: the
        // steady-Stokes Schur needed 19629 iterations here (5850 with
        // rescaling, 2049 with BtDiagB plus a Jacobi Schur subsolver).
        // BtDiagB plus an ILUT Schur subsolver converges in 68.
        const MAX_UNSTEADY_STALL_REGRESSION_ITERATIONS: usize = 500;
        assert!(
            telemetry.iterations <= MAX_UNSTEADY_STALL_REGRESSION_ITERATIONS,
            "unsteady Schur stall: {} iterations",
            telemetry.iterations
        );
        let velocity_error = direct
            .velocity
            .iter()
            .zip(&iterative.velocity)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let velocity_scale = direct
            .velocity
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt()
            .max(f64::MIN_POSITIVE);
        eprintln!(
            "REPRO_DIAG velocity_error={velocity_error:.6e} velocity_scale={velocity_scale:.6e}"
        );
        assert!(velocity_error / velocity_scale <= 1.0e-6);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_jacobi_schur_converges_mass_dominated_unsteady_step() {
        use crate::physics::fluid::stokes::time_stepping::{BdfConfig, BdfState};
        use crate::physics::fluid::stokes::traction::PressureTraction;
        use crate::physics::fluid::stokes::workspace::BdfWorkspace;

        // Skip-capable on hosts without an NVIDIA driver, following the
        // sonium-compute CUDA test convention.
        if sonium_compute::cuda::CudaBackend::new_default().is_err() {
            if std::env::var("SONIUM_REQUIRE_CUDA").is_ok() {
                panic!("CUDA required by SONIUM_REQUIRE_CUDA but unavailable");
            }
            eprintln!("skipping CUDA Jacobi/Schur test: no NVIDIA driver");
            return;
        }

        // Same mass-dominated port-flow fixture as the CPU regression test:
        // the backend must upload the unsteady BtDiagB Schur product, not a
        // steady-Stokes Mp/mu matrix, or backend GMRES stalls and the solve
        // falls back to CPU.
        let (mesh, inlet, outlet) = marked_channel_mesh(1.0, 0.1, 32, 16);
        let v_space = VectorH1Space::new(PolynomialDegree::P2, 2);
        let p_space = H1Space::new(PolynomialDegree::P1);
        let n_v = v_space.num_global_dofs(&mesh);
        let n_p = p_space.num_global_dofs(&mesh);
        let mut bc_dofs = v_space.boundary_dofs(&mesh, 3);
        bc_dofs.sort_unstable();
        bc_dofs.dedup();
        let bc_fn = |_: f64, _: f64, _: f64| vec![0.0, 0.0];
        let bdf = BdfConfig::bdf2(8.844870228e-5, 2);
        let base = NavierStokesConfig {
            mu: 1.85e-5,
            rho: 1.2,
            use_residual_stabilisation: false,
            use_direct_solver: false,
            linear_solver_max_iterations: 5000,
            // Tighter than the CPU test: the backend GMRES stopping
            // criterion leaves a larger forward error at 1e-8 residual on
            // this ill-conditioned saddle system.
            linear_solver_tolerance: 1.0e-10,
            linear_solver_backend: LinearSolverBackend::Cuda,
            linear_solver_backend_preconditioner: BackendPreconditionerKind::JacobiSchur,
            ..Default::default()
        };
        let direct_config = NavierStokesConfig {
            use_direct_solver: true,
            linear_solver_backend: LinearSolverBackend::Cpu,
            ..base.clone()
        };
        let mut direct = BdfState::from_rest(n_v, n_p);
        let direct_workspace = BdfWorkspace::new(&mesh, &v_space, &p_space);
        let tractions = [
            PressureTraction {
                marker: inlet,
                pressure_pa: 1.0,
            },
            PressureTraction {
                marker: outlet,
                pressure_pa: 0.0,
            },
        ];
        assert!(solve_bdf_step_cached(
            &mesh,
            &v_space,
            &p_space,
            &direct_config,
            &bdf,
            &mut direct,
            &bc_fn,
            &bc_dofs,
            &tractions,
            &direct_workspace,
        ));
        let mut iterative = BdfState::from_rest(n_v, n_p);
        let iterative_workspace = BdfWorkspace::new(&mesh, &v_space, &p_space);
        assert!(solve_bdf_step_cached(
            &mesh,
            &v_space,
            &p_space,
            &base,
            &bdf,
            &mut iterative,
            &bc_fn,
            &bc_dofs,
            &tractions,
            &iterative_workspace,
        ));
        let telemetry = iterative_workspace
            .last_linear_solve_telemetry()
            .expect("telemetry recorded");
        assert_eq!(
            telemetry.resolved_linear_solver_backend,
            LinearSolverBackend::Cuda,
            "backend solve fell back to CPU"
        );
        assert!(telemetry.true_relative_residual <= 1.0e-10);
        const MAX_CUDA_STALL_REGRESSION_ITERATIONS: usize = 3000;
        assert!(
            telemetry.iterations <= MAX_CUDA_STALL_REGRESSION_ITERATIONS,
            "unsteady Schur stall on CUDA: {} iterations",
            telemetry.iterations
        );
        let velocity_error = direct
            .velocity
            .iter()
            .zip(&iterative.velocity)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let velocity_scale = direct
            .velocity
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt()
            .max(f64::MIN_POSITIVE);
        eprintln!(
            "CUDA_DIAG iterations={} velocity_error={velocity_error:.6e} velocity_scale={velocity_scale:.6e}",
            telemetry.iterations
        );
        assert!(velocity_error / velocity_scale <= 1.0e-6);
    }

    /// Run the coupled-port base/probe solve pair (same BDF step, two
    /// pressure loadings) and return the workspace plus the affine-combined
    /// accepted state and combination factor, mirroring `run_coupled_port`.
    #[allow(clippy::too_many_arguments)]
    fn run_affine_probe_pair(
        mesh: &Mesh,
        v_space: &VectorH1Space,
        p_space: &H1Space,
        bdf: &crate::physics::fluid::stokes::time_stepping::BdfConfig,
        bc_fn: &impl Fn(f64, f64, f64) -> Vec<f64>,
        bc_dofs: &[usize],
        inlet: i32,
        outlet: i32,
        config: &NavierStokesConfig,
    ) -> (
        crate::physics::fluid::stokes::workspace::BdfWorkspace,
        Vec<f64>,
        f64,
    ) {
        use crate::physics::fluid::stokes::time_stepping::BdfState;
        use crate::physics::fluid::stokes::traction::PressureTraction;
        use crate::physics::fluid::stokes::workspace::BdfWorkspace;

        let n_v = v_space.num_global_dofs(mesh);
        let n_p = p_space.num_global_dofs(mesh);
        let workspace = BdfWorkspace::new(mesh, v_space, p_space);
        let previous = BdfState::from_rest(n_v, n_p);
        let solve_at_pressure = |pressure_pa: f64, trial: &mut BdfState| {
            let tractions = [
                PressureTraction {
                    marker: inlet,
                    pressure_pa,
                },
                PressureTraction {
                    marker: outlet,
                    pressure_pa: 0.0,
                },
            ];
            solve_bdf_step_cached(
                mesh, v_space, p_space, config, bdf, trial, bc_fn, bc_dofs, &tractions, &workspace,
            )
        };
        let mut flow_base = previous.clone();
        assert!(solve_at_pressure(1.0, &mut flow_base));
        let mut flow_probe = previous.clone();
        assert!(solve_at_pressure(2.0, &mut flow_probe));
        let factor = 0.25;
        let mut accepted = Vec::with_capacity(n_v + n_p);
        accepted.extend_from_slice(&flow_base.velocity);
        accepted.extend_from_slice(&flow_base.pressure);
        let probe_state: Vec<f64> = flow_probe
            .velocity
            .iter()
            .chain(&flow_probe.pressure)
            .copied()
            .collect();
        for (value, probe) in accepted.iter_mut().zip(&probe_state) {
            *value += factor * (*probe - *value);
        }
        (workspace, accepted, factor)
    }

    #[test]
    fn affine_probe_residual_available_after_cpu_solves() {
        use crate::physics::fluid::stokes::time_stepping::BdfConfig;

        let (mesh, inlet, outlet) = marked_channel_mesh(1.0, 0.1, 32, 16);
        let v_space = VectorH1Space::new(PolynomialDegree::P2, 2);
        let p_space = H1Space::new(PolynomialDegree::P1);
        let mut bc_dofs = v_space.boundary_dofs(&mesh, 3);
        bc_dofs.sort_unstable();
        bc_dofs.dedup();
        let bc_fn = |_: f64, _: f64, _: f64| vec![0.0, 0.0];
        let bdf = BdfConfig::bdf2(8.844870228e-5, 2);
        let config = NavierStokesConfig {
            mu: 1.85e-5,
            rho: 1.2,
            use_residual_stabilisation: false,
            use_direct_solver: false,
            linear_solver_max_iterations: 5000,
            linear_solver_tolerance: 1.0e-8,
            linear_solver_backend: LinearSolverBackend::Cpu,
            linear_solver_cpu_preconditioner: LinearSolverCpuPreconditionerKind::JacobiSchur,
            ..Default::default()
        };
        let (workspace, accepted, factor) = run_affine_probe_pair(
            &mesh, &v_space, &p_space, &bdf, &bc_fn, &bc_dofs, inlet, outlet, &config,
        );
        let residual = workspace
            .affine_true_relative_residual(&accepted, factor)
            .expect("affine residual available after CPU solves");
        assert!(residual <= 1.0e-6, "affine residual {residual:e}");
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn affine_probe_residual_available_after_cuda_solves() {
        use crate::physics::fluid::stokes::time_stepping::BdfConfig;

        if sonium_compute::cuda::CudaBackend::new_default().is_err() {
            if std::env::var("SONIUM_REQUIRE_CUDA").is_ok() {
                panic!("CUDA required by SONIUM_REQUIRE_CUDA but unavailable");
            }
            eprintln!("skipping CUDA affine test: no NVIDIA driver");
            return;
        }
        let (mesh, inlet, outlet) = marked_channel_mesh(1.0, 0.1, 32, 16);
        let v_space = VectorH1Space::new(PolynomialDegree::P2, 2);
        let p_space = H1Space::new(PolynomialDegree::P1);
        let mut bc_dofs = v_space.boundary_dofs(&mesh, 3);
        bc_dofs.sort_unstable();
        bc_dofs.dedup();
        let bc_fn = |_: f64, _: f64, _: f64| vec![0.0, 0.0];
        let bdf = BdfConfig::bdf2(8.844870228e-5, 2);
        let config = NavierStokesConfig {
            mu: 1.85e-5,
            rho: 1.2,
            use_residual_stabilisation: false,
            use_direct_solver: false,
            linear_solver_max_iterations: 5000,
            linear_solver_tolerance: 1.0e-10,
            linear_solver_backend: LinearSolverBackend::Cuda,
            linear_solver_backend_preconditioner: BackendPreconditionerKind::JacobiSchur,
            ..Default::default()
        };
        let (workspace, accepted, factor) = run_affine_probe_pair(
            &mesh, &v_space, &p_space, &bdf, &bc_fn, &bc_dofs, inlet, outlet, &config,
        );
        let residual = workspace
            .affine_true_relative_residual(&accepted, factor)
            .expect("affine residual available after CUDA solves");
        assert!(residual <= 1.0e-6, "affine residual {residual:e}");
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn backend_schur_cache_reuses_same_matrix_product() {
        use crate::physics::fluid::stokes::time_stepping::{BdfConfig, BdfState};
        use crate::physics::fluid::stokes::traction::PressureTraction;
        use crate::physics::fluid::stokes::workspace::BdfWorkspace;

        if sonium_compute::cuda::CudaBackend::new_default().is_err() {
            if std::env::var("SONIUM_REQUIRE_CUDA").is_ok() {
                panic!("CUDA required by SONIUM_REQUIRE_CUDA but unavailable");
            }
            eprintln!("skipping CUDA Schur-cache test: no NVIDIA driver");
            return;
        }
        let (mesh, inlet, outlet) = marked_channel_mesh(1.0, 0.1, 32, 16);
        let v_space = VectorH1Space::new(PolynomialDegree::P2, 2);
        let p_space = H1Space::new(PolynomialDegree::P1);
        let n_v = v_space.num_global_dofs(&mesh);
        let n_p = p_space.num_global_dofs(&mesh);
        let mut bc_dofs = v_space.boundary_dofs(&mesh, 3);
        bc_dofs.sort_unstable();
        bc_dofs.dedup();
        let bc_fn = |_: f64, _: f64, _: f64| vec![0.0, 0.0];
        let bdf = BdfConfig::bdf2(8.844870228e-5, 2);
        let config = NavierStokesConfig {
            mu: 1.85e-5,
            rho: 1.2,
            use_residual_stabilisation: false,
            use_direct_solver: false,
            linear_solver_max_iterations: 5000,
            linear_solver_tolerance: 1.0e-10,
            linear_solver_backend: LinearSolverBackend::Cuda,
            linear_solver_backend_preconditioner: BackendPreconditionerKind::JacobiSchur,
            ..Default::default()
        };
        let workspace = BdfWorkspace::new(&mesh, &v_space, &p_space);
        let solve_at_pressure = |pressure_pa: f64, trial: &mut BdfState| {
            let tractions = [
                PressureTraction {
                    marker: inlet,
                    pressure_pa,
                },
                PressureTraction {
                    marker: outlet,
                    pressure_pa: 0.0,
                },
            ];
            solve_bdf_step_cached(
                &mesh, &v_space, &p_space, &config, &bdf, trial, &bc_fn, &bc_dofs, &tractions,
                &workspace,
            )
        };
        let hits = || {
            workspace
                .bdf_schur_cache
                .borrow()
                .as_ref()
                .map(|entry| entry.hits)
        };
        // Same step, same state, two loadings: identical matrices, so the
        // second solve must reuse the memoized Schur product.
        let previous = BdfState::from_rest(n_v, n_p);
        let mut flow_base = previous.clone();
        assert!(solve_at_pressure(1.0, &mut flow_base));
        assert_eq!(hits(), Some(0));
        let mut flow_probe = previous.clone();
        assert!(solve_at_pressure(2.0, &mut flow_probe));
        assert_eq!(hits(), Some(1));
        // Changed advecting state: different convection block, so the memo
        // must miss and rebuild instead of serving a stale product.
        let mut perturbed = previous.clone();
        for value in perturbed.velocity.iter_mut().take(n_v.min(16)) {
            *value += 0.5;
        }
        assert!(solve_at_pressure(1.0, &mut perturbed));
        assert_eq!(hits(), Some(0));
    }

    #[test]
    fn body_force_cache_matches_reassembly_bitwise() {
        use crate::physics::fluid::stokes::time_stepping::{BdfConfig, BdfState};
        use crate::physics::fluid::stokes::traction::PressureTraction;
        use crate::physics::fluid::stokes::workspace::BdfWorkspace;

        let run_two_steps = |cache: bool| {
            let (mesh, inlet, outlet) = marked_channel_mesh(1.0, 0.1, 8, 4);
            let v_space = VectorH1Space::new(PolynomialDegree::P2, 2);
            let p_space = H1Space::new(PolynomialDegree::P1);
            let n_v = v_space.num_global_dofs(&mesh);
            let n_p = p_space.num_global_dofs(&mesh);
            let mut bc_dofs = v_space.boundary_dofs(&mesh, 3);
            bc_dofs.sort_unstable();
            bc_dofs.dedup();
            let bc_fn = |_: f64, _: f64, _: f64| vec![0.0, 0.0];
            let bdf = BdfConfig::bdf2(0.125, 2);
            let config = NavierStokesConfig {
                mu: 1.0e-3,
                rho: 1.0,
                use_residual_stabilisation: false,
                cache_body_force_assembly: cache,
                ..Default::default()
            };
            let workspace = BdfWorkspace::new(&mesh, &v_space, &p_space);
            let mut state = BdfState::from_rest(n_v, n_p);
            // Nonzero body force so the cached triplets carry live
            // values, not just zeros.
            let force = |_: [f64; 3]| vec![0.25, -0.125, 0.0];
            let jacobian = |_: [f64; 3]| [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]];
            let tractions = [
                PressureTraction {
                    marker: inlet,
                    pressure_pa: 1.0,
                },
                PressureTraction {
                    marker: outlet,
                    pressure_pa: 0.0,
                },
            ];
            for _ in 0..2 {
                assert!(solve_bdf_step_cached_with_force(
                    &mesh, &v_space, &p_space, &config, &bdf, &mut state, &bc_fn, &bc_dofs,
                    &tractions, force, jacobian, &workspace,
                ));
            }
            let populated = workspace.body_force_cache.borrow().is_some();
            (state.velocity, state.pressure, populated)
        };
        let (v_plain, p_plain, populated_plain) = run_two_steps(false);
        let (v_cached, p_cached, populated_cached) = run_two_steps(true);
        assert!(!populated_plain, "disabled cache must stay empty");
        assert!(populated_cached, "enabled cache must populate");
        assert_eq!(v_plain.len(), v_cached.len());
        assert_eq!(p_plain.len(), p_cached.len());
        for (plain, cached) in v_plain.iter().zip(&v_cached) {
            assert_eq!(
                plain.to_bits(),
                cached.to_bits(),
                "cached velocity differs from reassembled"
            );
        }
        for (plain, cached) in p_plain.iter().zip(&p_cached) {
            assert_eq!(
                plain.to_bits(),
                cached.to_bits(),
                "cached pressure differs from reassembled"
            );
        }
    }

    #[test]
    fn body_force_cache_reassembles_on_changed_force() {
        use crate::physics::fluid::stokes::time_stepping::{BdfConfig, BdfState};
        use crate::physics::fluid::stokes::traction::PressureTraction;
        use crate::physics::fluid::stokes::workspace::BdfWorkspace;

        let (mesh, inlet, outlet) = marked_channel_mesh(1.0, 0.1, 8, 4);
        let v_space = VectorH1Space::new(PolynomialDegree::P2, 2);
        let p_space = H1Space::new(PolynomialDegree::P1);
        let n_v = v_space.num_global_dofs(&mesh);
        let n_p = p_space.num_global_dofs(&mesh);
        let mut bc_dofs = v_space.boundary_dofs(&mesh, 3);
        bc_dofs.sort_unstable();
        bc_dofs.dedup();
        let bc_fn = |_: f64, _: f64, _: f64| vec![0.0, 0.0];
        let bdf = BdfConfig::bdf2(0.125, 1);
        let config = NavierStokesConfig {
            mu: 1.0e-3,
            rho: 1.0,
            use_residual_stabilisation: false,
            cache_body_force_assembly: true,
            ..Default::default()
        };
        let tractions = [
            PressureTraction {
                marker: inlet,
                pressure_pa: 1.0,
            },
            PressureTraction {
                marker: outlet,
                pressure_pa: 0.0,
            },
        ];
        fn zero_jacobian(_: [f64; 3]) -> [[f64; 3]; 3] {
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        }
        fn force_a(_: [f64; 3]) -> Vec<f64> {
            vec![0.25, 0.0, 0.0]
        }
        fn force_b(_: [f64; 3]) -> Vec<f64> {
            vec![-0.5, 0.125, 0.0]
        }
        let step =
            |force: fn([f64; 3]) -> Vec<f64>, state: &mut BdfState, workspace: &BdfWorkspace| {
                assert!(solve_bdf_step_cached_with_force(
                    &mesh,
                    &v_space,
                    &p_space,
                    &config,
                    &bdf,
                    state,
                    &bc_fn,
                    &bc_dofs,
                    &tractions,
                    force,
                    zero_jacobian,
                    workspace,
                ));
            };
        let fingerprint = |workspace: &BdfWorkspace| {
            workspace
                .body_force_cache
                .borrow()
                .as_ref()
                .map(|cached| cached.sample_fingerprint)
        };
        // Shared workspace, changed force: the sample fingerprint must
        // mismatch and trigger reassembly instead of serving stale triplets.
        let shared = BdfWorkspace::new(&mesh, &v_space, &p_space);
        let mut state_a = BdfState::from_rest(n_v, n_p);
        step(force_a, &mut state_a, &shared);
        let fingerprint_a = fingerprint(&shared).expect("cache populated");
        let mut state_b_shared = BdfState::from_rest(n_v, n_p);
        step(force_b, &mut state_b_shared, &shared);
        let fingerprint_b = fingerprint(&shared).expect("cache repopulated");
        assert_ne!(
            fingerprint_a, fingerprint_b,
            "changed force must produce a new sample fingerprint"
        );
        // The reassembled shared-workspace step must match a fresh
        // workspace that only ever saw force B.
        let fresh = BdfWorkspace::new(&mesh, &v_space, &p_space);
        let mut state_b_fresh = BdfState::from_rest(n_v, n_p);
        step(force_b, &mut state_b_fresh, &fresh);
        for (shared_value, fresh_value) in
            state_b_shared.velocity.iter().zip(&state_b_fresh.velocity)
        {
            assert_eq!(
                shared_value.to_bits(),
                fresh_value.to_bits(),
                "stale body force served after force change"
            );
        }
    }

    #[test]
    fn lagged_setup_reuses_across_changed_steps_and_matches_physics() {
        use crate::physics::fluid::stokes::time_stepping::{BdfConfig, BdfState};
        use crate::physics::fluid::stokes::traction::PressureTraction;
        use crate::physics::fluid::stokes::workspace::BdfWorkspace;

        fn relative_l2(a: &[f64], b: &[f64]) -> f64 {
            assert_eq!(a.len(), b.len());
            let mut diff = 0.0;
            let mut norm = 0.0;
            for (x, y) in a.iter().zip(b) {
                diff += (x - y) * (x - y);
                norm += x * x;
            }
            (diff / norm.max(f64::MIN_POSITIVE)).sqrt()
        }

        let run = |max_lag_steps: usize| {
            let (mesh, inlet, outlet) = marked_channel_mesh(1.0, 0.1, 8, 4);
            let v_space = VectorH1Space::new(PolynomialDegree::P2, 2);
            let p_space = H1Space::new(PolynomialDegree::P1);
            let n_v = v_space.num_global_dofs(&mesh);
            let n_p = p_space.num_global_dofs(&mesh);
            let mut bc_dofs = v_space.boundary_dofs(&mesh, 3);
            bc_dofs.sort_unstable();
            bc_dofs.dedup();
            let bc_fn = |_: f64, _: f64, _: f64| vec![0.0, 0.0];
            let bdf = BdfConfig::bdf2(0.125, 4);
            let config = NavierStokesConfig {
                mu: 1.0e-3,
                rho: 1.0,
                use_residual_stabilisation: false,
                use_direct_solver: false,
                linear_solver_backend: LinearSolverBackend::Cpu,
                linear_solver_cpu_preconditioner: LinearSolverCpuPreconditionerKind::JacobiSchur,
                linear_solver_max_iterations: 5000,
                linear_solver_tolerance: 1.0e-10,
                linear_solver_setup_max_lag_steps: max_lag_steps,
                ..Default::default()
            };
            let workspace = BdfWorkspace::new(&mesh, &v_space, &p_space);
            let mut state = BdfState::from_rest(n_v, n_p);
            let tractions = [
                PressureTraction {
                    marker: inlet,
                    pressure_pa: 1.0,
                },
                PressureTraction {
                    marker: outlet,
                    pressure_pa: 0.0,
                },
            ];
            let mut reused = Vec::new();
            for _ in 0..4 {
                assert!(solve_bdf_step_cached(
                    &mesh, &v_space, &p_space, &config, &bdf, &mut state, &bc_fn, &bc_dofs,
                    &tractions, &workspace,
                ));
                reused.push(
                    workspace
                        .last_linear_solve_telemetry()
                        .expect("telemetry recorded")
                        .setup_reused,
                );
            }
            (state.velocity, state.pressure, reused)
        };
        let (v_plain, p_plain, reused_plain) = run(0);
        // Lag disabled: the Oseen matrix changes every step, so every
        // step rebuilds.
        assert_eq!(reused_plain, vec![false, false, false, false]);
        let (v_lagged, p_lagged, reused_lagged) = run(10);
        // Lag enabled: one fresh setup, then lagged reuse (the
        // mass-dominated drift sits far below the default tolerance).
        assert_eq!(reused_lagged, vec![false, true, true, true]);
        // Same equations, same tolerance: only the Krylov path differs.
        assert!(
            relative_l2(&v_plain, &v_lagged) < 1.0e-8,
            "lagged velocity drifted: {}",
            relative_l2(&v_plain, &v_lagged)
        );
        assert!(
            relative_l2(&p_plain, &p_lagged) < 1.0e-8,
            "lagged pressure drifted: {}",
            relative_l2(&p_plain, &p_lagged)
        );
        // The lag cap binds: K=1 alternates fresh and lagged setups.
        let (_, _, reused_capped) = run(1);
        assert_eq!(reused_capped, vec![false, true, false, true]);
    }

    #[test]
    fn lagged_setup_retries_once_with_fresh_setup() {
        use crate::physics::fluid::stokes::time_stepping::{BdfConfig, BdfState};
        use crate::physics::fluid::stokes::traction::PressureTraction;
        use crate::physics::fluid::stokes::workspace::BdfWorkspace;

        let (mesh, inlet, outlet) = marked_channel_mesh(1.0, 0.1, 8, 4);
        let v_space = VectorH1Space::new(PolynomialDegree::P2, 2);
        let p_space = H1Space::new(PolynomialDegree::P1);
        let n_v = v_space.num_global_dofs(&mesh);
        let n_p = p_space.num_global_dofs(&mesh);
        let mut bc_dofs = v_space.boundary_dofs(&mesh, 3);
        bc_dofs.sort_unstable();
        bc_dofs.dedup();
        let bc_fn = |_: f64, _: f64, _: f64| vec![0.0, 0.0];
        let bdf = BdfConfig::bdf2(0.125, 2);
        let base = NavierStokesConfig {
            mu: 1.0e-3,
            rho: 1.0,
            use_residual_stabilisation: false,
            use_direct_solver: false,
            linear_solver_backend: LinearSolverBackend::Cpu,
            linear_solver_cpu_preconditioner: LinearSolverCpuPreconditionerKind::JacobiSchur,
            linear_solver_tolerance: 1.0e-10,
            linear_solver_setup_max_lag_steps: 10,
            // Force lag on step 2 regardless of drift size.
            linear_solver_setup_lag_tolerance: 1.0e30,
            ..Default::default()
        };
        let tractions_at = |pressure_pa: f64| {
            [
                PressureTraction {
                    marker: inlet,
                    pressure_pa,
                },
                PressureTraction {
                    marker: outlet,
                    pressure_pa: 0.0,
                },
            ]
        };
        // Step 2 shrinks dt 1000x after step 1 populates the cache.
        // The lag tolerance is forced wide open, so the lagged run
        // reuses a setup built for a mass scaling 1000x off while the
        // control rebuilds fresh. Both attempts get two Krylov cycles.
        let step_2_with = |max_lag_steps: usize| {
            let config = NavierStokesConfig {
                linear_solver_setup_max_lag_steps: max_lag_steps,
                ..base.clone()
            };
            let workspace = BdfWorkspace::new(&mesh, &v_space, &p_space);
            let mut state = BdfState::from_rest(n_v, n_p);
            let converge = NavierStokesConfig {
                linear_solver_max_iterations: 5000,
                ..config.clone()
            };
            assert!(solve_bdf_step_cached(
                &mesh,
                &v_space,
                &p_space,
                &converge,
                &bdf,
                &mut state,
                &bc_fn,
                &bc_dofs,
                &tractions_at(1.0),
                &workspace,
            ));
            let small_dt = BdfConfig::bdf2(0.125 / 1000.0, 2);
            let budgeted = NavierStokesConfig {
                linear_solver_max_iterations: 2,
                ..config
            };
            assert!(solve_bdf_step_cached(
                &mesh,
                &v_space,
                &p_space,
                &budgeted,
                &small_dt,
                &mut state,
                &bc_fn,
                &bc_dofs,
                &tractions_at(1.0),
                &workspace,
            ));
            workspace
                .last_linear_solve_telemetry()
                .expect("telemetry recorded")
        };
        let fresh_only = step_2_with(0);
        assert!(
            fresh_only.converged,
            "fresh setup handles the dt change within budget"
        );
        assert!(
            !fresh_only.setup_reused,
            "lag disabled rebuilds the changed matrix"
        );
        let lagged = step_2_with(10);
        // The stale attempt fails but the fresh retry rescues the step.
        assert!(lagged.converged);
        // Each attempt runs at most two 50-cycles (the arm's restart
        // with max_iterations = 2), so exceeding 100 proves the lagged
        // attempt ran to budget and the retry ran after it.
        assert!(
            lagged.iterations > 100,
            "retry must run a second attempt, got {} iterations",
            lagged.iterations
        );
        assert!(
            !lagged.setup_reused,
            "retry rebuilds fresh, so the final setup is not reused"
        );
    }

    #[test]
    fn mono_pattern_cache_matches_triplet_merge() {
        use crate::physics::fluid::stokes::workspace::{BdfMonoPatternCache, BdfWorkspace};

        let (mesh, _, _) = marked_channel_mesh(1.0, 0.1, 8, 4);
        let v_space = VectorH1Space::new(PolynomialDegree::P2, 2);
        let p_space = H1Space::new(PolynomialDegree::P1);
        let n_v = v_space.num_global_dofs(&mesh);
        let n_p = p_space.num_global_dofs(&mesh);
        let n_total = n_v + n_p;
        let workspace = BdfWorkspace::new(&mesh, &v_space, &p_space);
        // Deterministic nonzero advecting field so convection values are live.
        let u_star: Vec<f64> = (0..n_v).map(|i| 0.001 * ((i % 7) as f64)).collect();
        let conv = assemble_bdf_convection(&mesh, &v_space, &workspace, &u_star, 1.2, 2);
        let body = crate::assembly::assemble_bilinear(
            &BdfBodyForceJacobianForm {
                jacobian: |_| [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]],
            },
            &v_space,
            &v_space,
            &mesh,
            workspace.quad_order,
        );
        let pin = if n_p > 0 {
            Some(n_v + workspace.pressure_pin)
        } else {
            None
        };
        let mass_coeff = 20360.0;
        let viscous_scale = 1.85e-5;

        let cache = BdfMonoPatternCache::build(
            n_total,
            n_v,
            &workspace.mass_triplet,
            &workspace.viscous_unit_triplet,
            None,
            &workspace.block_01,
            &workspace.block_10,
            &conv,
            &body,
            pin,
        );
        let (values, _) = cache
            .try_fill(mass_coeff, viscous_scale, false, None, &conv, &body)
            .expect("cache matches current triplets");
        let fast = cache.to_csr(values);

        // Reference: legacy triplet merge with identical scales.
        let mut mono = crate::assembly::TripletMatrix::new(n_total, n_total);
        for k in 0..workspace.mass_triplet.nnz() {
            mono.push(
                workspace.mass_triplet.rows[k],
                workspace.mass_triplet.cols[k],
                mass_coeff * workspace.mass_triplet.values[k],
            );
        }
        for k in 0..workspace.viscous_unit_triplet.nnz() {
            mono.push(
                workspace.viscous_unit_triplet.rows[k],
                workspace.viscous_unit_triplet.cols[k],
                viscous_scale * workspace.viscous_unit_triplet.values[k],
            );
        }
        for k in 0..workspace.block_01.nnz() {
            mono.push(
                workspace.block_01.rows[k],
                workspace.block_01.cols[k] + n_v,
                workspace.block_01.values[k],
            );
        }
        for k in 0..workspace.block_10.nnz() {
            mono.push(
                workspace.block_10.rows[k] + n_v,
                workspace.block_10.cols[k],
                workspace.block_10.values[k],
            );
        }
        for k in 0..conv.nnz() {
            mono.push(conv.rows[k], conv.cols[k], conv.values[k]);
        }
        for k in 0..body.nnz() {
            mono.push(body.rows[k], body.cols[k], body.values[k]);
        }
        if let Some(dof) = pin {
            mono.push(dof, dof, 0.0);
        }
        let reference = mono.to_csr();

        assert_eq!(fast.row_ptrs, reference.row_ptrs);
        assert_eq!(fast.col_indices, reference.col_indices);
        let scale = reference
            .values
            .iter()
            .fold(0.0_f64, |max, value| max.max(value.abs()))
            .max(f64::MIN_POSITIVE);
        let max_diff = fast
            .values
            .iter()
            .zip(&reference.values)
            .fold(0.0_f64, |max, (a, b)| max.max((a - b).abs()));
        assert!(
            max_diff / scale <= 1.0e-12,
            "pattern-cache drift {max_diff:e} (scale {scale:e})"
        );

        // Fail-closed: reordered triplet positions must miss the cache even
        // though they describe the same matrix.
        let mut reordered = conv.clone();
        let last = reordered.rows.len() - 1;
        reordered.rows.swap(0, last);
        reordered.cols.swap(0, last);
        reordered.values.swap(0, last);
        assert!(
            cache
                .try_fill(mass_coeff, viscous_scale, false, None, &reordered, &body)
                .is_none(),
            "reordered positions must miss the pattern cache"
        );
        // ...while a cache rebuilt from the reordered triplets is exact.
        let rebuilt = BdfMonoPatternCache::build(
            n_total,
            n_v,
            &workspace.mass_triplet,
            &workspace.viscous_unit_triplet,
            None,
            &workspace.block_01,
            &workspace.block_10,
            &reordered,
            &body,
            pin,
        );
        let (rebuilt_values, _) = rebuilt
            .try_fill(mass_coeff, viscous_scale, false, None, &reordered, &body)
            .expect("rebuilt cache matches reordered triplets");
        let rebuilt_csr = rebuilt.to_csr(rebuilt_values);
        assert_eq!(rebuilt_csr.row_ptrs, reference.row_ptrs);
        assert_eq!(rebuilt_csr.col_indices, reference.col_indices);
        let rebuilt_diff = rebuilt_csr
            .values
            .iter()
            .zip(&reference.values)
            .fold(0.0_f64, |max, (a, b)| max.max((a - b).abs()));
        assert!(
            rebuilt_diff / scale <= 1.0e-12,
            "rebuilt-cache drift {rebuilt_diff:e} (scale {scale:e})"
        );

        // Value fingerprints track values, not just positions: the Schur
        // memo keys on these, so equal positions with different values
        // must produce different fingerprints.
        let mut rescaled = conv.clone();
        for value in &mut rescaled.values {
            *value *= 1.5;
        }
        let (_, base_fingerprints) = cache
            .try_fill(mass_coeff, viscous_scale, false, None, &conv, &body)
            .expect("base fill succeeds");
        let (_, rescaled_fingerprints) = cache
            .try_fill(mass_coeff, viscous_scale, false, None, &rescaled, &body)
            .expect("same positions fill");
        assert_ne!(
            base_fingerprints.convection, rescaled_fingerprints.convection,
            "value fingerprints must track values"
        );

        // Transported-viscosity mode: unit block replaced by the scattered
        // transported block, and mode switches miss the cache.
        let mut transported_config = NavierStokesConfig::default();
        transported_config.transported_eddy_viscosity_per_element =
            Some(vec![1.0e-4; mesh.num_elements()]);
        let transported = assemble_transported_viscous_triplet(
            &mesh,
            &v_space,
            &transported_config,
            workspace.quad_order,
        )
        .expect("transported assembly succeeds")
        .expect("transported block present");
        let transported_cache = BdfMonoPatternCache::build(
            n_total,
            n_v,
            &workspace.mass_triplet,
            &workspace.viscous_unit_triplet,
            Some(&transported),
            &workspace.block_01,
            &workspace.block_10,
            &conv,
            &body,
            pin,
        );
        let (transported_values, _) = transported_cache
            .try_fill(mass_coeff, 1.0, true, Some(&transported), &conv, &body)
            .expect("transported cache matches");
        let transported_csr = transported_cache.to_csr(transported_values);
        let mut transported_mono = crate::assembly::TripletMatrix::new(n_total, n_total);
        for k in 0..workspace.mass_triplet.nnz() {
            transported_mono.push(
                workspace.mass_triplet.rows[k],
                workspace.mass_triplet.cols[k],
                mass_coeff * workspace.mass_triplet.values[k],
            );
        }
        for k in 0..transported.nnz() {
            transported_mono.push(
                transported.rows[k],
                transported.cols[k],
                transported.values[k],
            );
        }
        for k in 0..workspace.block_01.nnz() {
            transported_mono.push(
                workspace.block_01.rows[k],
                workspace.block_01.cols[k] + n_v,
                workspace.block_01.values[k],
            );
        }
        for k in 0..workspace.block_10.nnz() {
            transported_mono.push(
                workspace.block_10.rows[k] + n_v,
                workspace.block_10.cols[k],
                workspace.block_10.values[k],
            );
        }
        for k in 0..conv.nnz() {
            transported_mono.push(conv.rows[k], conv.cols[k], conv.values[k]);
        }
        for k in 0..body.nnz() {
            transported_mono.push(body.rows[k], body.cols[k], body.values[k]);
        }
        if let Some(dof) = pin {
            transported_mono.push(dof, dof, 0.0);
        }
        let transported_reference = transported_mono.to_csr();
        assert_eq!(
            transported_csr.row_ptrs, transported_reference.row_ptrs,
            "transported pattern must match"
        );
        assert_eq!(
            transported_csr.col_indices, transported_reference.col_indices,
            "transported pattern must match"
        );
        let transported_scale = transported_reference
            .values
            .iter()
            .fold(0.0_f64, |max, value| max.max(value.abs()))
            .max(f64::MIN_POSITIVE);
        let transported_diff = transported_csr
            .values
            .iter()
            .zip(&transported_reference.values)
            .fold(0.0_f64, |max, (a, b)| max.max((a - b).abs()));
        assert!(
            transported_diff / transported_scale <= 1.0e-12,
            "transported drift {transported_diff:e}"
        );
        assert!(
            transported_cache
                .try_fill(mass_coeff, viscous_scale, false, None, &conv, &body)
                .is_none(),
            "viscosity-mode switch must miss the cache"
        );
        assert!(
            cache
                .try_fill(mass_coeff, 1.0, true, Some(&transported), &conv, &body)
                .is_none(),
            "viscosity-mode switch must miss the cache"
        );
    }

    /// Run the pressure-driven Task-2 channel fixture to steady state with
    /// `solve_bdf_step_cached` and return the peak velocity magnitude.
    ///
    /// Walls are Dirichlet no-slip (`bc_dofs`/`bc_fn` from the caller) and the
    /// flow is driven by pressure tractions on the caps — markers 1 (inlet)
    /// and 2 (outlet) as classified by [`marked_channel_mesh`]. This path
    /// factorizes fine on the jittered Delaunay mesh, unlike an all-Dirichlet
    /// inlet+walls BC set (pre-existing LU-singular quirk).
    #[allow(clippy::too_many_arguments)]
    fn run_channel_peak_velocity<BcFn>(
        mesh: &Mesh,
        v_space: &VectorH1Space,
        p_space: &H1Space,
        config: &NavierStokesConfig,
        bdf: &super::super::time_stepping::BdfConfig,
        bc_fn: &BcFn,
        bc_dofs: &[usize],
    ) -> f64
    where
        BcFn: Fn(f64, f64, f64) -> Vec<f64>,
    {
        use crate::physics::fluid::stokes::time_stepping::BdfState;
        use crate::physics::fluid::stokes::traction::PressureTraction;
        use crate::physics::fluid::stokes::workspace::BdfWorkspace;

        let n_v = v_space.num_global_dofs(mesh);
        let n_p = p_space.num_global_dofs(mesh);
        let workspace = BdfWorkspace::new(mesh, v_space, p_space);
        let mut state = BdfState::from_rest(n_v, n_p);
        let tractions = [
            PressureTraction {
                marker: 1,
                pressure_pa: 1.0,
            },
            PressureTraction {
                marker: 2,
                pressure_pa: 0.0,
            },
        ];
        for step in 0..bdf.num_steps {
            assert!(
                solve_bdf_step_cached(
                    mesh, v_space, p_space, config, bdf, &mut state, bc_fn, bc_dofs, &tractions,
                    &workspace,
                ),
                "cached BDF step {step} failed"
            );
        }
        state
            .velocity
            .chunks(mesh.dimension)
            .map(|v| v.iter().map(|c| c * c).sum::<f64>().sqrt())
            .fold(0.0_f64, f64::max)
    }

    /// BDF Newton assembly must add the consistent mass contribution and
    /// the BDF history source to the steady residual/tangent without changing
    /// the assembled steady terms.
    #[test]
    fn bdf_newton_linearization_adds_mass_and_history_terms() {
        use crate::physics::fluid::stokes::time_stepping::{BdfConfig, BdfState};
        use crate::physics::fluid::stokes::workspace::BdfWorkspace;

        let mesh = crate::mesh::unit_square_triangles(1);
        let velocity_space = VectorH1Space::new(PolynomialDegree::P1, 2);
        let pressure_space = H1Space::new(PolynomialDegree::P1);
        let workspace = BdfWorkspace::new(&mesh, &velocity_space, &pressure_space);
        let config = NavierStokesConfig {
            mu: 0.7,
            rho: 1.3,
            use_residual_stabilisation: false,
            ..NavierStokesConfig::default()
        };
        let bdf = BdfConfig::bdf2(0.2, 1);
        let state = BdfState {
            velocity: (0..workspace.n_v)
                .map(|index| 0.2 + 0.03 * index as f64)
                .collect(),
            velocity_prev: (0..workspace.n_v)
                .map(|index| -0.1 + 0.02 * index as f64)
                .collect(),
            pressure: (0..workspace.n_p)
                .map(|index| 0.1 - 0.04 * index as f64)
                .collect(),
            time: 0.4,
            step: 2,
        };
        let zero_force = |_coords: [f64; 3]| vec![0.0; 3];
        let zero_force_jacobian = |_coords: [f64; 3]| [[0.0; 3]; 3];
        let assemble = |state: &BdfState| {
            assemble_bdf_newton_linearization(
                &mesh,
                &velocity_space,
                &pressure_space,
                &config,
                &bdf,
                state,
                &|_, _, _| vec![0.0, 0.0],
                &[],
                &zero_force,
                &zero_force_jacobian,
                &workspace,
                config.mu,
                None,
            )
            .expect("finite BDF Newton assembly")
        };

        let (residual, jacobian) = assemble(&state);
        let mut previous = state.clone();
        previous.velocity_prev[0] += 0.37;
        let (previous_residual, previous_jacobian) = assemble(&previous);
        let mass = workspace.mass_triplet.to_csr();
        let mut changed_history = vec![0.0; workspace.n_v];
        changed_history[0] = 0.37;
        let expected_history_delta = mass
            .matvec(&changed_history)
            .into_iter()
            .map(|value| config.rho * 0.5 / bdf.dt * value)
            .chain(std::iter::repeat(0.0))
            .take(workspace.n_v + workspace.n_p)
            .collect::<Vec<_>>();
        assert!(
            residual
                .iter()
                .zip(&previous_residual)
                .zip(&expected_history_delta)
                .all(|((current, previous), expected)| {
                    (previous - current - expected).abs() < 1.0e-12
                })
        );
        let direction = vec![0.013; workspace.n_v + workspace.n_p];
        let tangent_difference = jacobian
            .matvec(&direction)
            .iter()
            .zip(previous_jacobian.matvec(&direction))
            .map(|(current, previous)| (current - previous).abs())
            .fold(0.0_f64, f64::max);
        assert!(tangent_difference < 1.0e-12);
    }

    #[test]
    fn bdf_algebraic_turbulence_increases_effective_viscosity() {
        use crate::physics::fluid::stokes::time_stepping::BdfConfig;

        // Channel fixture from Task 2, driven hard enough that
        // KOmegaSstAlgebraic engages.
        let length = 2.0;
        let height = 0.1;
        let (mesh, _, _) = marked_channel_mesh(length, height, 20, 16);
        let v_space = VectorH1Space::new(PolynomialDegree::P1, 2);
        let p_space = H1Space::new(PolynomialDegree::P1);

        // No-slip walls only; inlet/outlet stay traction-driven.
        let tolerance = 1.0e-9;
        let mut bc_dofs = crate::boundary::locate_dofs_geometrical(&mesh, &v_space, |_, y, _| {
            y.abs() < tolerance || (y - height).abs() < tolerance
        });
        bc_dofs.sort_unstable();
        bc_dofs.dedup();
        let bc_fn = |_: f64, _: f64, _: f64| vec![0.0, 0.0];

        let mut config = NavierStokesConfig {
            mu: 1.0e-3,
            rho: 1.0,
            use_residual_stabilisation: true,
            ..Default::default()
        };
        config.turbulence = super::super::turbulence::TurbulenceModel::KOmegaSstAlgebraic {
            c1: 0.09,
            characteristic_length_m: 0.05,
            nu_t_max_ratio: 100.0,
        };
        let n_steps = 80;
        let bdf = BdfConfig::bdf2(0.125, n_steps);

        let u_lam = run_channel_peak_velocity(
            &mesh,
            &v_space,
            &p_space,
            &NavierStokesConfig {
                turbulence: super::super::turbulence::TurbulenceModel::Laminar,
                ..config.clone()
            },
            &bdf,
            &bc_fn,
            &bc_dofs,
        );
        let u_turb =
            run_channel_peak_velocity(&mesh, &v_space, &p_space, &config, &bdf, &bc_fn, &bc_dofs);
        assert!(u_turb < 0.95 * u_lam, "laminar {u_lam}, turbulent {u_turb}");
    }

    #[test]
    fn mesh_motion_residual_derivative_supports_algebraic_turbulence() {
        let mut mesh = Mesh::new(3);
        mesh.nodes = vec![
            Point::new_3d(0.0, 0.0, 0.0),
            Point::new_3d(1.0, 0.0, 0.0),
            Point::new_3d(0.0, 1.0, 0.0),
            Point::new_3d(0.0, 0.0, 1.0),
        ];
        mesh.elements.push(crate::mesh::Element::new(
            ElementType::Tetrahedron,
            vec![0, 1, 2, 3],
            0,
        ));

        let velocity_space = VectorH1Space::new(PolynomialDegree::P1, 3);
        let pressure_space = H1Space::new(PolynomialDegree::P1);
        let config = NavierStokesConfig {
            mu: 0.23,
            rho: 1.7,
            use_residual_stabilisation: true,
            turbulence: super::super::turbulence::TurbulenceModel::KOmegaSstAlgebraic {
                c1: 0.09,
                characteristic_length_m: 0.5,
                nu_t_max_ratio: 100.0,
            },
            ..NavierStokesConfig::default()
        };
        let velocity = (0..velocity_space.num_global_dofs(&mesh))
            .map(|index| 0.17 + 0.031 * index as f64)
            .collect::<Vec<_>>();
        let pressure = (0..pressure_space.num_global_dofs(&mesh))
            .map(|index| -0.11 + 0.047 * index as f64)
            .collect::<Vec<_>>();
        let mut nodal_jacobian = vec![[[0.0; 3]; 1]; mesh.nodes.len()];
        nodal_jacobian[0][0][0] = 1.0;

        let analytic = assemble_mesh_motion_design_residual(
            &mesh,
            &velocity_space,
            &pressure_space,
            &config,
            &velocity,
            &pressure,
            &nodal_jacobian,
            |_coords, _velocity| [0.0; 3],
            |_coords, _velocity| [[0.0; 3]; 3],
        )
        .expect("algebraic turbulence mesh derivative should be admitted");

        let velocity_magnitude_max = velocity
            .chunks_exact(3)
            .take(mesh.nodes.len())
            .map(|node_velocity| {
                node_velocity
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt()
            })
            .fold(0.0_f64, f64::max);
        let viscous_mu =
            config
                .turbulence
                .effective_mu(config.mu, config.rho, velocity_magnitude_max);
        let step = 1.0e-7;
        let mut plus_mesh = mesh.clone();
        plus_mesh.nodes[0].x += step;
        let mut minus_mesh = mesh;
        minus_mesh.nodes[0].x -= step;
        let (plus_residual, _) = assemble_complete_newton_linearization(
            &plus_mesh,
            &velocity_space,
            &pressure_space,
            &config,
            viscous_mu,
            None,
            &velocity,
            &pressure,
            &|_coords, _velocity| vec![0.0; 3],
            &|_coords, _velocity| [[0.0; 3]; 3],
            2,
        );
        let (minus_residual, _) = assemble_complete_newton_linearization(
            &minus_mesh,
            &velocity_space,
            &pressure_space,
            &config,
            viscous_mu,
            None,
            &velocity,
            &pressure,
            &|_coords, _velocity| vec![0.0; 3],
            &|_coords, _velocity| [[0.0; 3]; 3],
            2,
        );

        for row in 0..analytic.len() {
            let centered = (plus_residual[row] - minus_residual[row]) / (2.0 * step);
            let scale = analytic[row][0].abs().max(centered.abs()).max(1.0);
            assert!(
                (analytic[row][0] - centered).abs() / scale < 2.0e-6,
                "row {row}: analytic={}, centered={centered}",
                analytic[row][0]
            );
        }
    }

    #[test]
    fn mesh_motion_residual_derivative_matches_centered_affine_tetrahedron_p2_velocity() {
        let mut mesh = Mesh::new(3);
        mesh.nodes = vec![
            Point::new_3d(0.0, 0.0, 0.0),
            Point::new_3d(1.0, 0.0, 0.0),
            Point::new_3d(0.0, 1.0, 0.0),
            Point::new_3d(0.0, 0.0, 1.0),
        ];
        mesh.elements.push(crate::mesh::Element::new(
            ElementType::Tetrahedron,
            vec![0, 1, 2, 3],
            0,
        ));
        let velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
        let pressure_space = H1Space::new(PolynomialDegree::P1);
        let config = NavierStokesConfig {
            mu: 0.23,
            rho: 1.7,
            use_residual_stabilisation: true,
            ..NavierStokesConfig::default()
        };
        let velocity = (0..velocity_space.num_global_dofs(&mesh))
            .map(|index| 0.17 + 0.031 * index as f64)
            .collect::<Vec<_>>();
        let pressure = (0..pressure_space.num_global_dofs(&mesh))
            .map(|index| -0.11 + 0.047 * index as f64)
            .collect::<Vec<_>>();
        let body_force = |coords: [f64; 3], _velocity: [f64; 3]| {
            [
                0.7 * coords[0] - 0.2 * coords[1],
                -0.1 * coords[0] + 0.4 * coords[2],
                0.3 * coords[1] + 0.5 * coords[2],
            ]
        };
        let body_force_position_jacobian = |_coords: [f64; 3], _velocity: [f64; 3]| {
            [[0.7, -0.2, 0.0], [-0.1, 0.0, 0.4], [0.0, 0.3, 0.5]]
        };
        let mut nodal_jacobian = vec![[[0.0; 3]; 1]; mesh.nodes.len()];
        nodal_jacobian[0][0][0] = 1.0;
        let analytic = assemble_mesh_motion_design_residual(
            &mesh,
            &velocity_space,
            &pressure_space,
            &config,
            &velocity,
            &pressure,
            &nodal_jacobian,
            body_force,
            body_force_position_jacobian,
        )
        .expect("analytic mesh-motion residual derivative");

        let step = 1.0e-7;
        let mut plus_mesh = mesh.clone();
        plus_mesh.nodes[0].x += step;
        let mut minus_mesh = mesh;
        minus_mesh.nodes[0].x -= step;
        let (plus_residual, _) = assemble_complete_newton_linearization(
            &plus_mesh,
            &velocity_space,
            &pressure_space,
            &config,
            config.mu,
            None,
            &velocity,
            &pressure,
            &|coords, velocity| body_force(coords, velocity).to_vec(),
            &|_coords, _velocity| [[0.0; 3]; 3],
            2,
        );
        let (minus_residual, _) = assemble_complete_newton_linearization(
            &minus_mesh,
            &velocity_space,
            &pressure_space,
            &config,
            config.mu,
            None,
            &velocity,
            &pressure,
            &|coords, velocity| body_force(coords, velocity).to_vec(),
            &|_coords, _velocity| [[0.0; 3]; 3],
            2,
        );
        for row in 0..analytic.len() {
            let centered = (plus_residual[row] - minus_residual[row]) / (2.0 * step);
            let scale = analytic[row][0].abs().max(centered.abs()).max(1.0);
            assert!(
                (analytic[row][0] - centered).abs() / scale < 2.0e-6,
                "row {row}: analytic={} centered={centered}",
                analytic[row][0]
            );
        }
    }
}
