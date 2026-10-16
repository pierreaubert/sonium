//! High-level BEM Solver API
//!
//! This module provides a unified, high-level interface for acoustic BEM simulations.
//! It integrates mesh generation, system assembly, linear solving, and post-processing.
//!
//! The Direct solver method works in both native and WASM modes. When the `native` feature
//! is enabled, it uses optimized BLAS/LAPACK. Otherwise, it falls back to a pure Rust
//! LU factorization implementation.
//!
//! # Example
//!
//! ```ignore
//! use sonium_bem::core::{BemSolver, BemProblem, IncidentField};
//!
//! // Create a rigid sphere scattering problem
//! let problem = BemProblem::rigid_sphere_scattering(
//!     0.1,        // radius
//!     1000.0,     // frequency
//!     343.0,      // speed of sound
//!     1.21,       // density
//! );
//!
//! // Configure solver
//! let solver = BemSolver::new()
//!     .with_mesh_refinement(3)
//!     .with_solver_method(SolverMethod::Direct);
//!
//! // Solve
//! let solution = solver.solve(&problem)?;
//!
//! // Evaluate field at a point
//! let pressure = solution.evaluate_pressure(&[0.0, 0.0, 0.2]);
//! ```

use num_complex::Complex64;
use sonium_solvers::dense::DenseMatrix;
use std::f64::consts::PI;

use crate::core::admission::{BemResourceBudget, BemResourceRequest};
use crate::core::assembly::common::count_dofs;
use crate::core::assembly::galerkin::assemble_galerkin_single_layer_with_global_dofs;
use crate::core::assembly::hmatrix::{
    apply_tbem_stabilized_exact, apply_tbem_stabilized_exact_hermitian,
    apply_tbem_stabilized_exact_higher_order, apply_tbem_stabilized_exact_higher_order_curved,
    apply_tbem_stabilized_exact_higher_order_curved_hermitian,
    apply_tbem_stabilized_exact_higher_order_hermitian,
    apply_tbem_stabilized_exact_higher_order_owner,
    apply_tbem_stabilized_exact_higher_order_owner_flat,
    apply_tbem_stabilized_exact_higher_order_owner_flat_hermitian,
    apply_tbem_stabilized_exact_higher_order_owner_hermitian,
};
use crate::core::assembly::slfmm::SlfmmSystem;
use crate::core::assembly::tbem::{
    TbemQuadraturePolicy, build_tbem_system_linear, build_tbem_system_quadratic_stabilized,
    build_tbem_system_stabilized, build_tbem_system_stabilized_density_derivative,
    build_tbem_system_stabilized_impedance_density_derivative,
    build_tbem_system_stabilized_impedance_speed_derivative,
    build_tbem_system_stabilized_speed_derivative,
    build_tbem_system_stabilized_wavenumber_derivative, tbem_wavenumber_objective_gradient,
    validate_tbem_quadrature_for_basis,
};
use crate::core::conventions::BemConvention;
use crate::core::incident::IncidentField;
use crate::core::integration::duffy::quad4_geometry;
use crate::core::integration::gauss::{gauss_legendre, triangle_quadrature};
use crate::core::io::vtu::write_quadratic_surface_pressure_vtu;
use crate::core::mesh::element::{
    basis_values, compute_element_area, compute_element_center, compute_element_normal,
};
use crate::core::mesh::generators::{generate_icosphere_mesh, generate_sphere_mesh};
use crate::core::mesh::{
    BemCacheInvalidation, BemReuseContext, BemReuseDecision, BemReusePolicy, MeshUpdatePlan,
};
use crate::core::postprocess::pressure::{
    FieldEvaluationError, FieldPoint, compute_scattered_field, compute_scattered_field_checked,
    compute_scattered_field_gradient_checked, compute_single_layer_field, compute_total_field,
    compute_total_field_checked, compute_total_field_gradient_checked,
    map_dual_basis_coefficients_checked,
};
use crate::core::types::{
    BasisContinuity, BasisOrder, BoundaryCondition, Element, ElementType, Mesh, PhysicsParams,
};
use sonium_solvers::direct::{lu_factorize, lu_solve};
use sonium_solvers::iterative::{
    BiCgstabConfig, BlockGmresConfig, GmresConfig, bicgstab, block_gmres_preconditioned, gmres,
    gmres_preconditioned,
};
use sonium_solvers::preconditioners::{DiagonalPreconditioner, HiluPreconditioner};
use sonium_solvers::traits::{LinearOperator, Preconditioner};

/// Solver method for the linear system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SolverMethod {
    /// Direct LU factorization (best for small problems)
    #[default]
    Direct,
    /// Conjugate Gradient Squared (iterative)
    Cgs,
    /// BiCGSTAB (iterative, more stable than CGS)
    BiCgStab,
    /// Restarted GMRES (iterative, supports left preconditioning)
    Gmres,
}

#[cfg(test)]
mod continuous_quadratic_solver_tests {
    use super::*;

    #[test]
    fn solver_continuous_quadratic_dispatches_unique_tri3_rows() {
        let element = Element {
            connectivity: vec![0, 1, 2],
            element_type: ElementType::Tri3,
            property: crate::core::types::ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0, 0.0, 1.0]; 3],
            center: [1.0 / 3.0, 1.0 / 3.0, 0.0],
            area: 0.5,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::default()]),
            group: 0,
            dof_addresses: Vec::new(),
        };
        let problem = BemProblem {
            mesh: Mesh {
                nodes: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                elements: vec![element],
                external_node_numbers: Vec::new(),
                external_element_numbers: Vec::new(),
                num_boundary_nodes: 3,
                num_evaluation_nodes: 0,
                num_boundary_elements: 1,
                num_evaluation_elements: 0,
                symmetry_planes: [0; 3],
                symmetry_coordinates: [0.0; 3],
                num_reflections: 0,
                node_to_elements: Vec::new(),
            },
            physics: PhysicsParams::new(100.0, 343.0, 1.21, false),
            incident_field: IncidentField::plane_wave_z(),
            bc_type: BoundaryConditionType::Rigid,
            use_burton_miller: true,
        };
        let solver = BemSolver::new().with_continuous_quadratic();
        let elements = solver.prepare_elements(&problem).expect("Tri3 map");
        assert_eq!(elements[0].dof_addresses, vec![0, 1, 2, 3, 4, 5]);

        let assembly = solver
            .assemble_system_stabilized(&elements, &problem.mesh.nodes, &problem.physics, 1.0, 0.2)
            .expect("continuous quadratic TBEM assembly");
        match assembly {
            AssemblyResult::Dense(matrix, rhs) => {
                assert_eq!(matrix.shape(), (6, 6));
                assert_eq!(rhs.len(), 6);
                assert!(
                    matrix
                        .as_slice()
                        .iter()
                        .all(|value| value.re.is_finite() && value.im.is_finite())
                );
            }
            _ => panic!("continuous quadratic solver must dispatch to dense TBEM"),
        }
    }
}

/// Preconditioner selection for iterative solves.
///
/// Caveats specific to the current Burton-Miller assembly (`bem-step1.md`
/// §1.2): the system operator is the *combined* `K' + iη·E` (or its dual),
/// not a pure first-kind V or W. The Calderón identity that motivates the
/// opposite-order preconditioner only delivers mesh-independent
/// conditioning when the system itself is first-kind, so on the current
/// Burton-Miller assembly the `Calderon*` variants buy correctness (they're
/// non-singular preconditioners) but the iteration-count win is empirical
/// rather than theoretical. The doc-mandated upgrade — pure V or pure W
/// system operator and the *dual* mass on a barycentric refinement — is
/// the prerequisite for the 5–20× iteration reduction the plan promises.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BemPreconditioner {
    /// No preconditioning.
    #[default]
    None,
    /// Pointwise Jacobi: 1 / A_ii from the assembled diagonal.
    Jacobi,
    /// Surface lumped mass: 1 / area_i (constant elements) or `Σ area/n`
    /// row-sum (linear elements). For uniform meshes with constant basis
    /// this collapses to a uniform diagonal scaling — the iteration count
    /// will not change versus the unpreconditioned solve. Useful as a
    /// no-op smoke test of the GMRES wiring.
    SurfaceMassLumped,
    /// Calderón opposite-order with single-layer V as dual operator:
    /// `P = M⁻¹ V M⁻¹`. Targets W-dominated first-kind systems. Cost: one
    /// dense `O(N²)` V assembly at solve time. Constant-element basis only
    /// — see `core::assembly::operators` for the panic guard.
    CalderonV,
    /// Calderón opposite-order with hyper-singular W as dual operator:
    /// `P = M⁻¹ W M⁻¹`. Targets V-dominated first-kind systems. Cost: one
    /// dense `O(N²)` W assembly at solve time. Constant-element basis only.
    CalderonW,
    /// Hierarchical ILU assembled from exact FMM near-field blocks.
    ///
    /// This option is valid for SLFMM/MLFMM GMRES solves. The far-field
    /// action remains matrix-free and the result is checked against the
    /// complete FMM operator. Other assembly methods reject it instead of
    /// silently falling back to a different preconditioner.
    HierarchicalIlu,
    /// Phase-1.2 dual mass on barycentric refinement. Builds the
    /// barycentric refinement of the surface mesh once, assembles the
    /// consistent P1 mass `M_refined` on the refined mesh (sparse SPD),
    /// and applies `M_dual⁻¹ ≈ R · M_refined⁻¹ · L` via inner CG, where
    /// `L` is the centroid-sample lift from primal-P0 to refined-P1
    /// and `R` is its adjoint restriction. Real-valued mass is applied
    /// to `Complex64` residuals by splitting into real and imaginary
    /// halves (the mass commutes with the imaginary unit).
    /// Constant-element basis only — see
    /// `crates/sonium-bem/src/core/assembly/surface_mass.rs`.
    SurfaceMassDualBarycentric,
}

/// Assembly method for the BEM matrix.
///
/// Empirical guidance from `qa-bem-assembly-crossover` (kD = 1, icosphere
/// meshes; see `bem-step1.md` §1.4). Heavy run measurements
/// (`CROSSOVER_HEAVY=1`):
///
///   subdiv |  DOFs | Tbem assembly+10·matvec | Hmatrix | compression
///   -------+-------+-------------------------+---------+------------
///        1 |    80 |            0.04s        |   0.04s |   1.000×
///        2 |   320 |            0.39s        |   0.39s |   0.983×
///        3 |  1280 |            6.20s        |   4.76s |   1.078×
///        4 |  5120 |           83.50s        |  66.31s |   1.518×
///
/// - **`Tbem`** (dense O(N²)) is competitive only up to a few hundred
///   DOFs; H-matrix compression ratio is essentially 1.0 in that range,
///   so the dense path's lower constant factor wins.
/// - **`Hmatrix`** is the recommended choice for moderate-large problems —
///   compression starts paying off at ~1k DOFs and produces sub-percent
///   matvec error against dense at all kD measured. Crossover (Hmatrix
///   first beats Tbem on assembly + 10 matvecs) lands at 1280 DOFs.
/// - **`Mlfmm`** matvec parity vs dense (`qa-bem-mlfmm-audit`) after the
///   §1.4 fix stack (octant coverage, GL quadrature stability,
///   triangle-Gauss source-element integration, tightened
///   `MIN_GEOMETRIC_SEPARATION = 3.0` near/far criterion, Kahan-summed M2L):
///   `kD = 1` → rel L² err ≈ `4.8 × 10⁻¹⁶` (machine precision);
///   `kD = 10` → rel L² err ≈ `4.0 × 10⁻⁴`;
///   `kD = 50` → rel L² err ≈ `7.2 × 10⁻⁴` (`MLFMM_AUDIT_HEAVY=1`,
///   ~5120 DOFs). All three PASS the doc-mandated `≤ 10⁻³` budget.
///   Trade-off: stricter near/far classification routes more cluster
///   pairs through dense near-field, so `Mlfmm` matvec wall-clock at
///   kD = 10 / 1280 DOFs is `~7×` slower than the loose-criterion path
///   that DEGRADED. Prefer `Mlfmm` when matvec accuracy matters and
///   problem size is large enough that the per-matvec saving over `Tbem`
///   covers the dense-near-field cost.
/// - **`Slfmm`** has only been validated on single-cluster (all-near)
///   parity (`qa-bem-fmm-solver-validation`); treat as preview-quality.
/// - **`Auto`** picks `Tbem` for `N < AUTO_TBEM_DOF_LIMIT` (1280 by
///   default, the first measured size at which `Hmatrix` strictly beats
///   `Tbem`) and `Hmatrix` at and above the threshold. With `Mlfmm` now
///   PASS-quality at all measured `kD`, a future revision may add a
///   secondary `Tbem`-vs-`Mlfmm` crossover measurement and route the
///   high-DOF tail into `Mlfmm` instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AssemblyMethod {
    /// Traditional BEM with O(N^2) dense matrix
    #[default]
    Tbem,
    /// Single-Level Fast Multipole Method
    Slfmm,
    /// Multi-Level Fast Multipole Method (PASS at the audit's `kD ∈ {1, 10}` —
    /// see enum-level docs for the trade-off vs `Hmatrix`).
    Mlfmm,
    /// Experimental directional MLFMM: prunes high-`kr` M2L plane-wave
    /// directions in cones while keeping the standard MLFMM as the oracle.
    DirectionalMlfmm,
    /// H-matrix with Adaptive Cross Approximation (ACA)
    Hmatrix,
    /// Pick `Tbem` or `Hmatrix` based on problem size — see
    /// `AUTO_TBEM_DOF_LIMIT` and `AssemblyMethod::resolve`.
    Auto,
}

/// Empirically-measured DOF threshold at and above which `AssemblyMethod::Auto`
/// uses `Hmatrix` instead of `Tbem`. See `qa-bem-assembly-crossover` (heavy
/// run, `CROSSOVER_HEAVY=1`) — the smallest measured DOF count at which
/// `Hmatrix` strictly beats `Tbem` on assembly + 10 matvecs.
pub const AUTO_TBEM_DOF_LIMIT: usize = 1280;

impl AssemblyMethod {
    /// Resolve `Auto` to a concrete method for the given DOF count. Other
    /// variants pass through unchanged. Useful when a caller wants to log
    /// the chosen path without re-implementing the threshold.
    pub fn resolve(self, num_dofs: usize) -> AssemblyMethod {
        match self {
            AssemblyMethod::Auto => {
                if num_dofs < AUTO_TBEM_DOF_LIMIT {
                    AssemblyMethod::Tbem
                } else {
                    AssemblyMethod::Hmatrix
                }
            }
            other => other,
        }
    }
}

/// Boundary condition type for the problem
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryConditionType {
    /// Rigid surface (zero normal velocity)
    Rigid,
    /// Soft surface (zero pressure)
    Soft,
    /// Impedance boundary condition
    Impedance,
}

/// Definition of a BEM problem
#[derive(Debug, Clone)]
pub struct BemProblem {
    /// Problem geometry mesh
    pub mesh: Mesh,
    /// Physical parameters
    pub physics: PhysicsParams,
    /// Incident field
    pub incident_field: IncidentField,
    /// Boundary condition type
    pub bc_type: BoundaryConditionType,
    /// Use Burton-Miller formulation (recommended for exterior problems)
    pub use_burton_miller: bool,
}

impl BemProblem {
    /// Admit a backend-owned BEM problem from a validated XEM description.
    ///
    /// XEM carries the stable field/convention contract while the mesh,
    /// medium, and source remain backend-specific.  This constructor binds
    /// those pieces before assembly, propagates the phasor convention into
    /// [`PhysicsParams`], and rejects malformed backend data as a typed
    /// admission error.  `pressure_field_id` is the field to pass later to
    /// [`BemSolution::to_xem_result`] or [`BemSolution::borrowed_xem_field`].
    pub fn from_xem(
        exchange: &sonium_xem::MultiphysicsProblem,
        pressure_field_id: &str,
        mesh: Mesh,
        physics: PhysicsParams,
        incident_field: IncidentField,
        bc_type: BoundaryConditionType,
        use_burton_miller: bool,
    ) -> Result<Self, crate::core::conventions::BemProblemError> {
        // BEM consumes the same partitioned exchange contract as FEM. Reject
        // cyclic dependencies before any mesh or dense-operator setup.
        let coupling_graph = sonium_xem::CouplingGraph::from_problem(exchange)?;
        let _partitioned_order = coupling_graph.partitioned_order()?;
        let convention = BemConvention::from_problem(exchange)?;
        BemConvention::pressure_field(exchange, pressure_field_id)?;
        let candidate = Self {
            mesh,
            physics: physics.with_convention(convention),
            incident_field,
            bc_type,
            use_burton_miller,
        };
        candidate
            .validate()
            .map_err(crate::core::conventions::BemProblemError::Backend)?;
        Ok(candidate)
    }

    /// Rebind this problem to a validated XEM convention before solving.
    ///
    /// The legacy constructors continue to use the canonical acoustic
    /// convention. Exchange-driven callers should use this method so the
    /// phasor sign is carried into `PhysicsParams` instead of being silently
    /// replaced by the BEM default at solver entry.
    pub fn with_exchange_convention(
        mut self,
        registry: &sonium_xem::ConventionRegistry,
    ) -> Result<Self, BemError> {
        let convention = BemConvention::from_exchange(registry).map_err(BemError::Convention)?;
        self.physics = self.physics.with_convention(convention);
        self.validate()?;
        Ok(self)
    }

    /// Validate the mesh and physical parameters before any assembly starts.
    ///
    /// This is intentionally explicit at the high-level boundary: malformed
    /// connectivity, non-finite geometry, and non-physical media must become
    /// typed errors rather than panics deep inside an integration kernel.
    pub fn validate(&self) -> Result<(), BemError> {
        if self.mesh.nodes.is_empty() || self.mesh.elements.is_empty() {
            return Err(BemError::InvalidMesh(
                "BEM problem requires at least one node and element".to_string(),
            ));
        }
        self.physics
            .validate()
            .map_err(|error| BemError::InvalidParameters(error.to_string()))?;
        self.incident_field
            .validate()
            .map_err(|error| BemError::InvalidParameters(format!("incident field: {error}")))?;
        for (node, coordinates) in self.mesh.nodes.iter().enumerate() {
            if coordinates.iter().any(|value| !value.is_finite()) {
                return Err(BemError::InvalidMesh(format!(
                    "node {node} contains a non-finite coordinate"
                )));
            }
        }
        // This tolerance applies uniformly to every panel. Derive it once so
        // full mesh admission stays linear instead of rescanning all nodes for
        // every element.
        let geometry_tolerance = 1e-9
            * self
                .mesh
                .nodes
                .iter()
                .flatten()
                .fold(1.0_f64, |scale, coordinate| scale.max(coordinate.abs()));
        for (element, panel) in self.mesh.elements.iter().enumerate() {
            if panel.connectivity.len() != panel.element_type.num_nodes() {
                return Err(BemError::InvalidMesh(format!(
                    "element {element} has {} connectivity entries, expected {}",
                    panel.connectivity.len(),
                    panel.element_type.num_nodes()
                )));
            }
            if let Some(&node) = panel
                .connectivity
                .iter()
                .find(|&&node| node >= self.mesh.nodes.len())
            {
                return Err(BemError::InvalidMesh(format!(
                    "element {element} references node {node} outside the mesh"
                )));
            }
            for (local_node, &node) in panel.connectivity.iter().enumerate() {
                if panel.connectivity[..local_node].contains(&node) {
                    return Err(BemError::InvalidMesh(format!(
                        "element {element} repeats connectivity node {node}"
                    )));
                }
            }
            if !panel.area.is_finite() || panel.area <= 0.0 {
                return Err(BemError::InvalidMesh(format!(
                    "element {element} has non-positive or non-finite area"
                )));
            }
            if panel
                .normal
                .iter()
                .chain(panel.center.iter())
                .any(|value| !value.is_finite())
            {
                return Err(BemError::InvalidMesh(format!(
                    "element {element} has non-finite normal or center"
                )));
            }

            let stored_normal_length = panel
                .normal
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            if (stored_normal_length - 1.0).abs() > 1e-8 {
                return Err(BemError::InvalidMesh(format!(
                    "element {element} normal has length {stored_normal_length}, expected a unit vector"
                )));
            }

            let expected_center = compute_element_center(&self.mesh.nodes, &panel.connectivity);
            if expected_center.iter().any(|value| !value.is_finite()) {
                return Err(BemError::InvalidMesh(format!(
                    "element {element} connectivity produces a non-finite center"
                )));
            }
            if panel
                .center
                .iter()
                .zip(expected_center)
                .any(|(stored, expected)| (stored - expected).abs() > geometry_tolerance)
            {
                return Err(BemError::InvalidMesh(format!(
                    "element {element} center does not match its connectivity"
                )));
            }

            let expected_normal =
                compute_element_normal(&self.mesh.nodes, &panel.connectivity, panel.element_type);
            if expected_normal.iter().any(|value| !value.is_finite()) {
                return Err(BemError::InvalidMesh(format!(
                    "element {element} connectivity produces a non-finite normal"
                )));
            }
            let normal_alignment = panel
                .normal
                .iter()
                .zip(expected_normal)
                .map(|(stored, expected)| stored * expected)
                .sum::<f64>();
            if normal_alignment < 1.0 - 1e-8 {
                return Err(BemError::InvalidMesh(format!(
                    "element {element} normal is inconsistent with its connectivity orientation"
                )));
            }

            let expected_area =
                compute_element_area(&self.mesh.nodes, &panel.connectivity, panel.element_type);
            if !expected_area.is_finite() || expected_area <= 0.0 {
                return Err(BemError::InvalidMesh(format!(
                    "element {element} connectivity produces a non-positive or non-finite area"
                )));
            }
            if (panel.area - expected_area).abs() > 1e-9 * expected_area.max(1.0) {
                return Err(BemError::InvalidMesh(format!(
                    "element {element} area does not match its connectivity geometry"
                )));
            }
        }
        Ok(())
    }

    /// Create a rigid sphere scattering problem with plane wave incidence
    ///
    /// # Arguments
    /// * `radius` - Sphere radius (m)
    /// * `frequency` - Excitation frequency (Hz)
    /// * `speed_of_sound` - Speed of sound (m/s)
    /// * `density` - Medium density (kg/m�)
    ///
    /// Prefer [`Self::try_rigid_sphere_scattering`] at an untrusted
    /// configuration boundary. The historical infallible constructor remains
    /// for compatibility with analytical examples.
    pub fn rigid_sphere_scattering(
        radius: f64,
        frequency: f64,
        speed_of_sound: f64,
        density: f64,
    ) -> Self {
        // Determine mesh resolution based on ka
        let k = 2.0 * PI * frequency / speed_of_sound;
        let ka = k * radius;

        // Rule of thumb: ~10 elements per wavelength
        // For a sphere, this translates to subdivision level
        let subdivisions = if ka < 1.0 {
            2 // Low frequency: coarse mesh ok
        } else if ka < 5.0 {
            3 // Medium frequency
        } else {
            4 // High frequency: need finer mesh
        };

        let mesh = generate_icosphere_mesh(radius, subdivisions);
        let physics = PhysicsParams::new(frequency, speed_of_sound, density, false);
        let incident_field = IncidentField::plane_wave_z();

        Self {
            mesh,
            physics,
            incident_field,
            bc_type: BoundaryConditionType::Rigid,
            use_burton_miller: true,
        }
    }

    /// Checked rigid-sphere problem constructor.
    ///
    /// Validates radius and medium parameters before mesh generation, so
    /// malformed input cannot reach geometry code that assumes positive
    /// dimensions.
    pub fn try_rigid_sphere_scattering(
        radius: f64,
        frequency: f64,
        speed_of_sound: f64,
        density: f64,
    ) -> Result<Self, BemError> {
        if !radius.is_finite() || radius <= 0.0 {
            return Err(BemError::InvalidParameters(
                "sphere radius must be finite and positive".into(),
            ));
        }
        let physics = PhysicsParams::try_new(frequency, speed_of_sound, density, false)
            .map_err(|error| BemError::InvalidParameters(error.to_string()))?;
        let ka = physics.wave_number * radius;
        let subdivisions = if ka < 1.0 {
            2
        } else if ka < 5.0 {
            3
        } else {
            4
        };
        let problem = Self {
            mesh: generate_icosphere_mesh(radius, subdivisions),
            physics,
            incident_field: IncidentField::plane_wave_z(),
            bc_type: BoundaryConditionType::Rigid,
            use_burton_miller: true,
        };
        problem.validate()?;
        Ok(problem)
    }

    /// Create a rigid sphere scattering problem with custom mesh resolution
    pub fn rigid_sphere_scattering_custom(
        radius: f64,
        frequency: f64,
        speed_of_sound: f64,
        density: f64,
        n_theta: usize,
        n_phi: usize,
    ) -> Self {
        let mesh = generate_sphere_mesh(radius, n_theta, n_phi);
        let physics = PhysicsParams::new(frequency, speed_of_sound, density, false);
        let incident_field = IncidentField::plane_wave_z();

        Self {
            mesh,
            physics,
            incident_field,
            bc_type: BoundaryConditionType::Rigid,
            use_burton_miller: true,
        }
    }

    /// Checked custom-resolution rigid-sphere problem constructor.
    pub fn try_rigid_sphere_scattering_custom(
        radius: f64,
        frequency: f64,
        speed_of_sound: f64,
        density: f64,
        n_theta: usize,
        n_phi: usize,
    ) -> Result<Self, BemError> {
        if !radius.is_finite() || radius <= 0.0 {
            return Err(BemError::InvalidParameters(
                "sphere radius must be finite and positive".into(),
            ));
        }
        if n_theta < 3 || n_phi < 3 {
            return Err(BemError::InvalidParameters(
                "sphere angular resolutions must be at least three".into(),
            ));
        }
        let physics = PhysicsParams::try_new(frequency, speed_of_sound, density, false)
            .map_err(|error| BemError::InvalidParameters(error.to_string()))?;
        let problem = Self {
            mesh: generate_sphere_mesh(radius, n_theta, n_phi),
            physics,
            incident_field: IncidentField::plane_wave_z(),
            bc_type: BoundaryConditionType::Rigid,
            use_burton_miller: true,
        };
        problem.validate()?;
        Ok(problem)
    }

    /// Set the incident field
    pub fn with_incident_field(mut self, field: IncidentField) -> Self {
        self.incident_field = field;
        self
    }

    /// Set the boundary condition type
    pub fn with_boundary_condition(mut self, bc_type: BoundaryConditionType) -> Self {
        self.bc_type = bc_type;
        self
    }

    /// Enable/disable Burton-Miller formulation
    pub fn with_burton_miller(mut self, use_bm: bool) -> Self {
        self.use_burton_miller = use_bm;
        self
    }

    /// Get the wave number times radius (ka)
    pub fn ka(&self) -> f64 {
        self.physics.wave_number * self.mesh_radius()
    }

    /// Estimate mesh radius from bounding box
    fn mesh_radius(&self) -> f64 {
        // Estimate radius from mesh nodes
        let mut max_r = 0.0f64;
        for i in 0..self.mesh.nodes.len() {
            let r = (self.mesh.nodes[i][0].powi(2)
                + self.mesh.nodes[i][1].powi(2)
                + self.mesh.nodes[i][2].powi(2))
            .sqrt();
            max_r = max_r.max(r);
        }
        max_r
    }
}

/// BEM solver configuration
/// Material parameter whose compressed operator derivative is approximated.
#[derive(Debug, Clone, Copy)]
enum HmatrixDerivativeParameter {
    SpeedOfSound,
    Density,
}

/// Physical parameter differentiated by a dense objective-gradient fallback.
///
/// Constant-panel TBEM uses the analytic derivative assemblers in `tbem.rs`.
/// Higher-order direct TBEM reuses its production matrix assembler at two
/// nearby material/frequency points instead; keeping this selector separate
/// avoids widening the compressed-operator derivative contract.
#[derive(Debug, Clone, Copy)]
enum DenseObjectiveGradientParameter {
    Wavenumber,
    SpeedOfSound,
    Density,
}

/// Refresh generated impedance admittances for a perturbed medium.
///
/// `BemProblem` represents its impedance boundary as `Y = 1/(rho*c)` in
/// [`BemSolver::prepare_elements`]. Compressed material brackets reuse the
/// prepared topology, so their plus/minus element copies must update `Y`
/// together with the perturbed `PhysicsParams`.
fn update_impedance_admittances(
    elements: &mut [Element],
    physics: &PhysicsParams,
    layered_surface: Option<&crate::materials::LayeredPorousSurface>,
    thin_shell_surface: Option<&crate::materials::ThinShellSurface>,
) -> Result<(), BemError> {
    let admittance = match (layered_surface, thin_shell_surface) {
        (Some(_), Some(_)) => {
            return Err(BemError::InvalidParameters(
                "layered porous and thin-shell surfaces are mutually exclusive".to_string(),
            ));
        }
        (Some(surface), None) => surface
            .try_admittance(
                physics.frequency,
                crate::materials::AmbientMedium::new(physics.density, physics.speed_of_sound),
            )
            .map_err(|error| BemError::InvalidParameters(error.to_string()))?,
        (None, Some(surface)) => surface
            .try_admittance(physics.frequency)
            .map_err(|error| BemError::InvalidParameters(error.to_string()))?,
        (None, None) => Complex64::new(1.0 / (physics.density * physics.speed_of_sound), 0.0),
    };
    for element in elements {
        if let BoundaryCondition::VelocityWithAdmittance {
            admittance: element_admittance,
            ..
        } = &mut element.boundary_condition
        {
            *element_admittance = admittance;
        }
    }
    Ok(())
}

type QuadraturePolicyFingerprint = Option<(
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    u64,
    u64,
)>;
type BemCacheConfiguration = (
    BoundaryConditionType,
    BasisOrder,
    BasisContinuity,
    u64,
    usize,
    usize,
    QuadraturePolicyFingerprint,
    Option<crate::materials::LayeredPorousSurface>,
    Option<crate::materials::ThinShellSurface>,
);

/// High-level BEM solver configuration.
#[derive(Debug, Clone)]
pub struct BemSolver {
    /// Linear solver method
    pub solver_method: SolverMethod,
    /// Matrix assembly method
    pub assembly_method: AssemblyMethod,
    /// Maximum iterations for iterative solvers
    pub max_iterations: usize,
    /// Tolerance for iterative solvers
    pub tolerance: f64,
    /// Verbose output
    pub verbose: bool,
    /// Burton-Miller β scale factor (default: 0.2 from complex Mie sweeps)
    pub beta_scale: f64,
    /// Basis function order (Constant or Linear)
    pub basis_order: BasisOrder,
    /// Continuity contract for the selected basis.
    ///
    /// The default is the historical panel-local layout. Continuous mode
    /// supports conforming linear/quadratic Tri3/Quad4 panels on CPU TBEM and
    /// H-matrix assembly.
    pub basis_continuity: BasisContinuity,
    /// Preconditioner for iterative solves (consumed by `SolverMethod::Gmres`
    /// on dense and H-matrix assemblies).
    pub preconditioner: BemPreconditioner,
    /// GMRES restart length (only used when `solver_method == Gmres`).
    pub gmres_restart: usize,
    /// GCRO-DR recycled subspace size used by `solve_frequency_sweep`.
    /// `0` keeps the older warm-start-only path.
    pub frequency_recycle_subspace_size: usize,
    /// ACA tolerance used by H-matrix far-field compression.
    pub hmatrix_aca_tolerance: f64,
    /// Maximum ACA rank per far-field block; zero selects an automatic limit.
    pub hmatrix_max_rank: usize,
    /// Number of bounded H-matrix accuracy retries after a failed solve.
    pub compressed_retry_count: usize,
    /// Maximum compressed-system DOFs for optional exact-TBEM residual
    /// certification. A value of zero disables the O(N²) reference pass.
    pub compressed_exact_residual_dof_limit: usize,
    /// Relative residual tolerance for the independent exact-TBEM check.
    pub compressed_exact_residual_tolerance: f64,
    /// Optional fail-fast host/device/work admission limits.
    pub resource_budget: Option<BemResourceBudget>,
    /// Optional adjacent-order quadrature admission policy.
    ///
    /// When present, the policy is checked before constant-panel and
    /// basis-aware linear/quadratic assembly, including compressed H/FMM
    /// admission where supported. Higher-order estimators cover every local
    /// basis coefficient but do not imply curved physical convergence.
    pub quadrature_policy: Option<TbemQuadraturePolicy>,
    /// Optional checked multilayer porous surface model for impedance
    /// boundaries. The model is evaluated at the problem frequency during
    /// element preparation, so frequency sweeps receive fresh admittance.
    pub layered_porous_surface: Option<crate::materials::LayeredPorousSurface>,
    /// Optional local thin-shell mass--spring--damper surface model for
    /// impedance boundaries. The model is evaluated at each problem
    /// frequency during element preparation.
    pub thin_shell_surface: Option<crate::materials::ThinShellSurface>,
}

impl Default for BemSolver {
    fn default() -> Self {
        Self {
            solver_method: SolverMethod::Direct,
            assembly_method: AssemblyMethod::Tbem,
            max_iterations: 1000,
            tolerance: 1e-8,
            verbose: false,
            beta_scale: 0.2,
            basis_order: BasisOrder::Constant,
            basis_continuity: BasisContinuity::Discontinuous,
            preconditioner: BemPreconditioner::None,
            gmres_restart: 50,
            frequency_recycle_subspace_size: 20,
            hmatrix_aca_tolerance: 1e-6,
            hmatrix_max_rank: 0,
            compressed_retry_count: 2,
            compressed_exact_residual_dof_limit: 0,
            compressed_exact_residual_tolerance: 1e-8,
            resource_budget: None,
            quadrature_policy: None,
            layered_porous_surface: None,
            thin_shell_surface: None,
        }
    }
}

impl BemSolver {
    /// Evaluate a fixed-geometry material derivative on a compressed
    /// H-matrix/SLFMM/MLFMM path. The value operators are assembled at a symmetric,
    /// scale-aware parameter bracket; the dense derivative builders supply
    /// the exact boundary-RHS term. This keeps the compressed operator path
    /// fail-closed and independently checkable until analytic speed/density
    /// ACA entry kernels are available.
    fn solve_compressed_parameter_objective_gradient(
        &self,
        problem: &BemProblem,
        forward_rhs: &[Complex64],
        objective_gradient: &[Complex64],
        parameter: HmatrixDerivativeParameter,
    ) -> Result<f64, BemError> {
        let elements = self.prepare_elements(problem)?;
        let forward = self
            .solve_rhs_many(problem, &[forward_rhs.to_vec()])?
            .into_iter()
            .next()
            .ok_or_else(|| {
                BemError::SolverFailed(
                    "forward material-gradient solve returned no solution".to_string(),
                )
            })?;
        let adjoint = self
            .solve_adjoint_many(problem, &[objective_gradient.to_vec()])?
            .into_iter()
            .next()
            .ok_or_else(|| {
                BemError::SolverFailed(
                    "adjoint material-gradient solve returned no solution".to_string(),
                )
            })?;

        let (base_parameter, derivative_rhs) = match parameter {
            HmatrixDerivativeParameter::SpeedOfSound => {
                let derivative = if problem.bc_type == BoundaryConditionType::Impedance {
                    build_tbem_system_stabilized_impedance_speed_derivative(
                        &elements,
                        &problem.mesh.nodes,
                        &problem.physics,
                        PhysicsParams::optimal_formulation_sign(problem.ka()),
                        if problem.use_burton_miller {
                            PhysicsParams::optimal_beta_scale(problem.ka())
                        } else {
                            0.0
                        },
                    )
                } else {
                    build_tbem_system_stabilized_speed_derivative(
                        &elements,
                        &problem.mesh.nodes,
                        &problem.physics,
                        PhysicsParams::optimal_formulation_sign(problem.ka()),
                        if problem.use_burton_miller {
                            PhysicsParams::optimal_beta_scale(problem.ka())
                        } else {
                            0.0
                        },
                    )
                };
                (problem.physics.speed_of_sound, derivative.rhs)
            }
            HmatrixDerivativeParameter::Density => {
                let derivative = if problem.bc_type == BoundaryConditionType::Impedance {
                    build_tbem_system_stabilized_impedance_density_derivative(
                        &elements,
                        &problem.mesh.nodes,
                        &problem.physics,
                        PhysicsParams::optimal_formulation_sign(problem.ka()),
                        if problem.use_burton_miller {
                            PhysicsParams::optimal_beta_scale(problem.ka())
                        } else {
                            0.0
                        },
                    )
                } else {
                    build_tbem_system_stabilized_density_derivative(
                        &elements,
                        &problem.mesh.nodes,
                        &problem.physics,
                        PhysicsParams::optimal_formulation_sign(problem.ka()),
                        if problem.use_burton_miller {
                            PhysicsParams::optimal_beta_scale(problem.ka())
                        } else {
                            0.0
                        },
                    )
                };
                (problem.physics.density, derivative.rhs)
            }
        };
        if !base_parameter.is_finite() || base_parameter <= 0.0 {
            return Err(BemError::InvalidParameters(
                "compressed material derivative requires a positive finite base parameter"
                    .to_string(),
            ));
        }
        let step = (base_parameter * 1.0e-5)
            .max(1.0e-8)
            .min(base_parameter * 0.25);
        let make_problem = |speed: f64, density: f64| -> Result<BemProblem, BemError> {
            let mut candidate = problem.clone();
            let mut physics = PhysicsParams::try_new(
                problem.physics.frequency,
                speed,
                density,
                problem.physics.tau < 0.0,
            )
            .map_err(|error| BemError::InvalidParameters(error.to_string()))?;
            physics.harmonic_factor = problem.physics.harmonic_factor;
            physics.pressure_factor = physics.density * physics.omega * physics.harmonic_factor;
            physics
                .validate()
                .map_err(|error| BemError::InvalidParameters(error.to_string()))?;
            candidate.physics = physics;
            Ok(candidate)
        };
        let (plus, minus) = match parameter {
            HmatrixDerivativeParameter::SpeedOfSound => (
                make_problem(
                    problem.physics.speed_of_sound + step,
                    problem.physics.density,
                )?,
                make_problem(
                    problem.physics.speed_of_sound - step,
                    problem.physics.density,
                )?,
            ),
            HmatrixDerivativeParameter::Density => (
                make_problem(
                    problem.physics.speed_of_sound,
                    problem.physics.density + step,
                )?,
                make_problem(
                    problem.physics.speed_of_sound,
                    problem.physics.density - step,
                )?,
            ),
        };
        let assemble_compressed_action =
            |candidate: &BemProblem| -> Result<Vec<Complex64>, BemError> {
                let ka = candidate.ka();
                let sign = PhysicsParams::optimal_formulation_sign(ka);
                let beta_scale = if candidate.use_burton_miller {
                    PhysicsParams::optimal_beta_scale(ka)
                } else {
                    0.0
                };
                let mut candidate_elements = elements.clone();
                if candidate.bc_type == BoundaryConditionType::Impedance {
                    update_impedance_admittances(
                        &mut candidate_elements,
                        &candidate.physics,
                        self.layered_porous_surface.as_ref(),
                        self.thin_shell_surface.as_ref(),
                    )?;
                }
                match self.assemble_system_stabilized(
                    &candidate_elements,
                    &candidate.mesh.nodes,
                    &candidate.physics,
                    sign,
                    beta_scale,
                )? {
                    AssemblyResult::Hmatrix(matrix, _) => Ok(matrix.matvec(forward.coefficients())),
                    AssemblyResult::Slfmm(system) => Ok(system.matvec(forward.coefficients())),
                    AssemblyResult::Mlfmm(system) => Ok(system.matvec(forward.coefficients())),
                    AssemblyResult::Dense(_, _) => Err(BemError::NotImplemented(
                        "compressed material derivative requires a compressed assembly".to_string(),
                    )),
                }
            };
        let plus_action = assemble_compressed_action(&plus)?;
        let minus_action = assemble_compressed_action(&minus)?;
        if derivative_rhs.len() != plus_action.len()
            || plus_action.len() != minus_action.len()
            || adjoint.coefficients().len() != plus_action.len()
        {
            return Err(BemError::InvalidParameters(
                "compressed material derivative dimension mismatch".to_string(),
            ));
        }
        let gradient = adjoint
            .coefficients()
            .iter()
            .zip(
                derivative_rhs
                    .iter()
                    .zip(plus_action.iter().zip(minus_action.iter())),
            )
            .map(|(adjoint, (rhs, (plus, minus)))| {
                adjoint.conj() * (*rhs - (*plus - *minus) / (2.0 * step))
            })
            .sum::<Complex64>();
        if !gradient.re.is_finite() {
            return Err(BemError::InvalidParameters(
                "compressed material objective gradient is non-finite".to_string(),
            ));
        }
        Ok(gradient.re)
    }

    /// Contract a dense higher-order TBEM operator derivative with a solved
    /// forward/adjoint pair.
    ///
    /// The constant-panel path has analytic derivative assemblers. The
    /// quadratic/linear dense path is assembled by the same production code as
    /// the value operator and differentiated with a centered two-sided stencil;
    /// this keeps basis, continuity, and quadrature policy handling identical to
    /// the solve path while avoiding a second higher-order derivative assembler.
    /// Evaluate an objective gradient on a higher-order compressed operator.
    ///
    /// H-matrix/FMM operators expose matrix actions but do not yet expose
    /// basis-aware parameter derivative entry kernels. Re-solving the two
    /// perturbed production operators gives a checked centered derivative of
    /// `Re(gᴴx)` without manufacturing a local-DOF derivative matrix. This is
    /// deliberately separate from the constant-panel analytic compressed path.
    fn solve_compressed_objective_gradient_finite_difference(
        &self,
        problem: &BemProblem,
        forward_rhs: &[Complex64],
        objective_gradient: &[Complex64],
        parameter: DenseObjectiveGradientParameter,
    ) -> Result<f64, BemError> {
        let elements = self.prepare_elements(problem)?;
        let num_dofs = count_dofs(&elements);
        if forward_rhs.len() != num_dofs || objective_gradient.len() != num_dofs {
            return Err(BemError::InvalidParameters(format!(
                "compressed objective-gradient RHS dimensions must equal {num_dofs}"
            )));
        }

        let base_value = match parameter {
            DenseObjectiveGradientParameter::Wavenumber => problem.physics.wave_number,
            DenseObjectiveGradientParameter::SpeedOfSound => problem.physics.speed_of_sound,
            DenseObjectiveGradientParameter::Density => problem.physics.density,
        };
        if !base_value.is_finite() || base_value <= 0.0 {
            return Err(BemError::InvalidParameters(
                "compressed objective-gradient parameter must be finite and positive".to_string(),
            ));
        }
        let step = 1.0e-5 * base_value.abs().max(1.0);

        let make_problem = |offset: f64| -> Result<BemProblem, BemError> {
            let target = base_value + offset;
            if !target.is_finite() || target <= 0.0 {
                return Err(BemError::InvalidParameters(
                    "compressed objective-gradient perturbation left physical range".to_string(),
                ));
            }
            let (frequency, speed_of_sound, density) = match parameter {
                DenseObjectiveGradientParameter::Wavenumber => (
                    target * problem.physics.speed_of_sound / (2.0 * PI),
                    problem.physics.speed_of_sound,
                    problem.physics.density,
                ),
                DenseObjectiveGradientParameter::SpeedOfSound => {
                    (problem.physics.frequency, target, problem.physics.density)
                }
                DenseObjectiveGradientParameter::Density => (
                    problem.physics.frequency,
                    problem.physics.speed_of_sound,
                    target,
                ),
            };
            let mut candidate = problem.clone();
            let mut physics = PhysicsParams::try_new(
                frequency,
                speed_of_sound,
                density,
                problem.physics.tau < 0.0,
            )
            .map_err(|error| BemError::InvalidParameters(error.to_string()))?;
            physics.harmonic_factor = problem.physics.harmonic_factor;
            physics.pressure_factor = physics.density * physics.omega * physics.harmonic_factor;
            physics
                .validate()
                .map_err(|error| BemError::InvalidParameters(error.to_string()))?;
            candidate.physics = physics;
            Ok(candidate)
        };

        let objective_at = |candidate: &BemProblem| -> Result<f64, BemError> {
            let solution = self
                .solve_rhs_many(candidate, &[forward_rhs.to_vec()])?
                .into_iter()
                .next()
                .ok_or_else(|| {
                    BemError::SolverFailed(
                        "compressed objective-gradient solve returned no solution".to_string(),
                    )
                })?;
            if solution.coefficients().len() != num_dofs {
                return Err(BemError::InvalidParameters(
                    "compressed objective-gradient solution dimension mismatch".to_string(),
                ));
            }
            let value = objective_gradient
                .iter()
                .zip(solution.coefficients())
                .map(|(gradient, coefficient)| (gradient.conj() * coefficient).re)
                .sum::<f64>();
            if !value.is_finite() {
                return Err(BemError::InvalidParameters(
                    "compressed objective-gradient objective is non-finite".to_string(),
                ));
            }
            Ok(value)
        };

        let plus = objective_at(&make_problem(step)?)?;
        let minus = objective_at(&make_problem(-step)?)?;
        let gradient = (plus - minus) / (2.0 * step);
        if !gradient.is_finite() {
            return Err(BemError::InvalidParameters(
                "compressed objective-gradient result is non-finite".to_string(),
            ));
        }
        Ok(gradient)
    }

    fn solve_dense_objective_gradient_finite_difference(
        &self,
        problem: &BemProblem,
        forward_rhs: &[Complex64],
        objective_gradient: &[Complex64],
        parameter: DenseObjectiveGradientParameter,
    ) -> Result<f64, BemError> {
        let elements = self.prepare_elements(problem)?;
        let num_dofs = count_dofs(&elements);
        if forward_rhs.len() != num_dofs || objective_gradient.len() != num_dofs {
            return Err(BemError::InvalidParameters(format!(
                "dense objective-gradient RHS dimensions must equal {num_dofs}"
            )));
        }

        let forward = self
            .solve_rhs_many(problem, &[forward_rhs.to_vec()])?
            .into_iter()
            .next()
            .ok_or_else(|| {
                BemError::SolverFailed("forward gradient solve returned no solution".to_string())
            })?;
        let adjoint = self
            .solve_adjoint_many(problem, &[objective_gradient.to_vec()])?
            .into_iter()
            .next()
            .ok_or_else(|| {
                BemError::SolverFailed("adjoint gradient solve returned no solution".to_string())
            })?;

        let base_value = match parameter {
            DenseObjectiveGradientParameter::Wavenumber => problem.physics.wave_number,
            DenseObjectiveGradientParameter::SpeedOfSound => problem.physics.speed_of_sound,
            DenseObjectiveGradientParameter::Density => problem.physics.density,
        };
        if !base_value.is_finite() || base_value <= 0.0 {
            return Err(BemError::InvalidParameters(
                "dense objective-gradient parameter must be finite and positive".to_string(),
            ));
        }
        let step = 1.0e-5 * base_value.abs().max(1.0);

        let assemble = |offset: f64| -> Result<(DenseMatrix<Complex64>, Vec<Complex64>), BemError> {
            let target = base_value + offset;
            if !target.is_finite() || target <= 0.0 {
                return Err(BemError::InvalidParameters(
                    "dense objective-gradient perturbation left the physical parameter range"
                        .to_string(),
                ));
            }
            let mut candidate = problem.clone();
            let (frequency, speed_of_sound, density) = match parameter {
                DenseObjectiveGradientParameter::Wavenumber => (
                    target * problem.physics.speed_of_sound / (2.0 * PI),
                    problem.physics.speed_of_sound,
                    problem.physics.density,
                ),
                DenseObjectiveGradientParameter::SpeedOfSound => {
                    (problem.physics.frequency, target, problem.physics.density)
                }
                DenseObjectiveGradientParameter::Density => (
                    problem.physics.frequency,
                    problem.physics.speed_of_sound,
                    target,
                ),
            };
            let mut physics = PhysicsParams::try_new(
                frequency,
                speed_of_sound,
                density,
                problem.physics.tau < 0.0,
            )
            .map_err(|error| BemError::InvalidParameters(error.to_string()))?;
            physics.harmonic_factor = problem.physics.harmonic_factor;
            physics.pressure_factor = physics.density * physics.omega * physics.harmonic_factor;
            physics
                .validate()
                .map_err(|error| BemError::InvalidParameters(error.to_string()))?;
            candidate.physics = physics;
            let candidate_elements = self.prepare_elements(&candidate)?;
            if count_dofs(&candidate_elements) != num_dofs {
                return Err(BemError::InvalidParameters(
                    "dense objective-gradient perturbation changed the DOF layout".to_string(),
                ));
            }
            let ka = candidate.ka();
            let sign = PhysicsParams::optimal_formulation_sign(ka);
            let beta_scale = if candidate.use_burton_miller {
                PhysicsParams::optimal_beta_scale(ka)
            } else {
                0.0
            };
            match self.assemble_system_stabilized(
                &candidate_elements,
                &candidate.mesh.nodes,
                &candidate.physics,
                sign,
                beta_scale,
            )? {
                AssemblyResult::Dense(matrix, rhs) => Ok((matrix, rhs)),
                AssemblyResult::Hmatrix(_, _)
                | AssemblyResult::Slfmm(_)
                | AssemblyResult::Mlfmm(_) => Err(BemError::NotImplemented(
                    "higher-order objective gradients require dense TBEM assembly".to_string(),
                )),
            }
        };

        let (plus_matrix, plus_rhs) = assemble(step)?;
        let (minus_matrix, minus_rhs) = assemble(-step)?;
        if plus_matrix.nrows() != num_dofs
            || plus_matrix.ncols() != num_dofs
            || minus_matrix.nrows() != num_dofs
            || minus_matrix.ncols() != num_dofs
            || plus_rhs.len() != num_dofs
            || minus_rhs.len() != num_dofs
            || adjoint.coefficients().len() != num_dofs
            || forward.coefficients().len() != num_dofs
        {
            return Err(BemError::InvalidParameters(
                "dense objective-gradient derivative dimension mismatch".to_string(),
            ));
        }

        let scale = 1.0 / (2.0 * step);
        let mut derivative_action = vec![Complex64::new(0.0, 0.0); num_dofs];
        for row in 0..num_dofs {
            for column in 0..num_dofs {
                derivative_action[row] += (plus_matrix[[row, column]]
                    - minus_matrix[[row, column]])
                    * scale
                    * forward.coefficients()[column];
            }
        }
        let gradient = adjoint
            .coefficients()
            .iter()
            .enumerate()
            .map(|(row, adjoint_value)| {
                let derivative_rhs = (plus_rhs[row] - minus_rhs[row]) * scale;
                adjoint_value.conj() * (derivative_rhs - derivative_action[row])
            })
            .sum::<Complex64>();
        if !gradient.re.is_finite() {
            return Err(BemError::InvalidParameters(
                "dense objective-gradient result is non-finite".to_string(),
            ));
        }
        Ok(gradient.re)
    }

    fn reuse_context(&self, problem: &BemProblem) -> Result<BemReuseContext, BemError> {
        let beta_scale = if problem.use_burton_miller {
            PhysicsParams::optimal_beta_scale(problem.ka())
        } else {
            f64::MIN_POSITIVE
        };
        BemReuseContext::capture(
            &problem.mesh,
            &problem.physics,
            PhysicsParams::convention(),
            self.assembly_method,
            self.solver_method,
            self.preconditioner,
            self.tolerance,
            beta_scale,
        )
        .map_err(|error| BemError::InvalidParameters(error.to_string()))
    }

    fn cache_configuration(&self, bc_type: BoundaryConditionType) -> BemCacheConfiguration {
        (
            bc_type,
            self.basis_order,
            self.basis_continuity,
            self.hmatrix_aca_tolerance.to_bits(),
            self.hmatrix_max_rank,
            self.compressed_retry_count,
            self.quadrature_policy.map(|policy| {
                (
                    policy.regular_coarse_order,
                    policy.regular_fine_order,
                    policy.singular_coarse.edge_gauss_order,
                    policy.singular_coarse.subelement_gauss_order,
                    policy.singular_coarse.edge_sections,
                    policy.singular_coarse.subtriangles_per_section,
                    policy.singular_fine.edge_gauss_order,
                    policy.singular_fine.subelement_gauss_order,
                    policy.singular_fine.edge_sections,
                    policy.singular_fine.subtriangles_per_section,
                    policy.absolute_tolerance.to_bits(),
                    policy.relative_tolerance.to_bits(),
                )
            }),
            self.layered_porous_surface.clone(),
            self.thin_shell_surface,
        )
    }

    fn cache_mesh_plan(previous: BemReuseContext, current: BemReuseContext) -> MeshUpdatePlan {
        let invalidation = |topology: bool| BemCacheInvalidation {
            rebuild_cluster_tree: true,
            rebuild_admissibility_buckets: true,
            rebuild_interpolation_operators: true,
            rebuild_operator: true,
            rebuild_preconditioner: true,
            discard_recycled_krylov: true,
            rebuild_dof_layout: topology,
        };
        if previous.mesh.topology_fingerprint != current.mesh.topology_fingerprint {
            MeshUpdatePlan::TopologyChanged {
                previous: previous.mesh,
                current: current.mesh,
                invalidation: invalidation(true),
            }
        } else if previous.mesh.geometry_fingerprint != current.mesh.geometry_fingerprint {
            MeshUpdatePlan::GeometryChanged {
                previous: previous.mesh,
                current: current.mesh,
                max_node_displacement: f64::INFINITY,
                invalidation: invalidation(false),
            }
        } else {
            MeshUpdatePlan::Unchanged {
                lineage: current.mesh,
                invalidation: BemCacheInvalidation {
                    rebuild_cluster_tree: false,
                    rebuild_admissibility_buckets: false,
                    rebuild_interpolation_operators: false,
                    rebuild_operator: false,
                    rebuild_preconditioner: false,
                    discard_recycled_krylov: false,
                    rebuild_dof_layout: false,
                },
            }
        }
    }

    /// Solve using a caller-owned persistent assembled-operator cache.
    /// Numerical operator entries are reused only for an exact
    /// frequency/context match; nearby-frequency decisions remain visible in
    /// [`BemOperatorCache::last_decision`], while those solves rebuild entries.
    pub fn solve_with_cache(
        &self,
        problem: &BemProblem,
        cache: &mut BemOperatorCache,
        policy: BemReusePolicy,
    ) -> Result<BemSolution, BemError> {
        self.validate()?;
        problem.validate()?;
        let elements = self.prepare_elements(problem)?;
        self.admit_resources(&elements, 1, 1)?;
        let context = self.reuse_context(problem)?;
        let decision = cache.context.map(|previous| {
            crate::core::mesh::assess_bem_cache_reuse(
                previous,
                context,
                Self::cache_mesh_plan(previous, context),
                policy,
            )
        });
        let decision = match decision {
            Some(Ok(decision)) => Some(decision),
            Some(Err(error)) => {
                cache.clear();
                return Err(BemError::InvalidParameters(error.to_string()));
            }
            None => None,
        };
        cache.last_decision = decision;
        let exact_context_match = decision.is_some_and(|decision| {
            decision.rejection.is_none()
                && cache.context.is_some_and(|previous| {
                    previous.frequency_hz.to_bits() == context.frequency_hz.to_bits()
                })
                && cache.configuration == Some(self.cache_configuration(problem.bc_type))
        });
        let global_sign = PhysicsParams::optimal_formulation_sign(problem.ka());
        let global_beta_scale = if problem.use_burton_miller {
            PhysicsParams::optimal_beta_scale(problem.ka())
        } else {
            0.0
        };
        let indirect_soft = problem.bc_type == BoundaryConditionType::Soft
            && self.basis_order == BasisOrder::Constant;
        let assembly_result = if exact_context_match {
            cache.assembly.as_ref().cloned().ok_or_else(|| {
                BemError::SolverFailed("cache context exists without operator".to_string())
            })?
        } else if indirect_soft {
            AssemblyResult::Dense(
                self.assemble_soft_dirichlet_matrix(problem, &elements)?,
                Vec::new(),
            )
        } else {
            self.assemble_system_stabilized(
                &elements,
                &problem.mesh.nodes,
                &problem.physics,
                global_sign,
                global_beta_scale,
            )?
        };
        let rhs = if indirect_soft {
            self.soft_dirichlet_rhs(problem, &elements)
        } else {
            let rhs = match &assembly_result {
                AssemblyResult::Dense(_, rhs) => rhs.clone(),
                AssemblyResult::Slfmm(system) => system.rhs.clone(),
                AssemblyResult::Mlfmm(system) => system.rhs.clone(),
                AssemblyResult::Hmatrix(_, rhs) => rhs.clone(),
            };
            self.add_incident_field_rhs_stabilized(
                rhs,
                &elements,
                &problem.mesh.nodes,
                &problem.incident_field,
                &problem.physics,
                problem.use_burton_miller,
                global_sign,
                global_beta_scale,
            )
        };
        let cached_operator = assembly_result.clone();
        let surface_pressure = match if indirect_soft {
            match assembly_result {
                AssemblyResult::Dense(matrix, _) => {
                    self.solve_dense_system(&matrix, &rhs, &BemPrecondImpl::Identity)
                }
                _ => Err(BemError::SolverFailed(
                    "soft-Dirichlet cache contains a non-dense operator".to_string(),
                )),
            }
        } else {
            self.solve_assembled_system(assembly_result, &rhs, &elements, problem)
        } {
            Ok(solution) => solution,
            Err(error) => {
                cache.clear();
                return Err(error);
            }
        };
        cache.context = Some(context);
        cache.assembly = Some(cached_operator);
        cache.indirect_soft = indirect_soft;
        cache.configuration = Some(self.cache_configuration(problem.bc_type));
        if exact_context_match {
            cache.reuse_hits = cache.reuse_hits.saturating_add(1);
        } else {
            cache.rebuilds = cache.rebuilds.saturating_add(1);
        }
        Ok(BemSolution {
            surface_pressure,
            elements,
            nodes: problem.mesh.nodes.clone(),
            incident_field: problem.incident_field.clone(),
            physics: problem.physics.clone(),
            basis_order: self.basis_order,
            representation: if indirect_soft {
                BemSolutionRepresentation::IndirectSingleLayer
            } else {
                BemSolutionRepresentation::DirectCauchy
            },
        })
    }

    /// Create a new solver with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate solver controls before assembly.
    pub fn validate(&self) -> Result<(), BemError> {
        if self.basis_continuity == BasisContinuity::Continuous
            && !matches!(self.basis_order, BasisOrder::Linear | BasisOrder::Quadratic)
        {
            return Err(BemError::NotImplemented(
                "continuous basis is currently implemented only for linear/quadratic Tri3/Quad4 panels"
                    .to_string(),
                ));
        }
        if let Some(surface) = &self.layered_porous_surface {
            surface
                .validate()
                .map_err(|error| BemError::InvalidParameters(error.to_string()))?;
        }
        if let Some(surface) = self.thin_shell_surface {
            surface
                .validate()
                .map_err(|error| BemError::InvalidParameters(error.to_string()))?;
        }
        if self.layered_porous_surface.is_some() && self.thin_shell_surface.is_some() {
            return Err(BemError::InvalidParameters(
                "layered porous and thin-shell surfaces are mutually exclusive".to_string(),
            ));
        }
        if let Some(policy) = self.quadrature_policy {
            if policy.regular_coarse_order == 0
                || policy.regular_fine_order <= policy.regular_coarse_order
            {
                return Err(BemError::InvalidParameters(
                    "quadrature regular orders must be positive and strictly adjacent".to_string(),
                ));
            }
            let valid_singular = |params: crate::core::integration::QuadratureParams| {
                params.edge_gauss_order > 0
                    && params.subelement_gauss_order > 0
                    && params.edge_sections > 0
                    && params.subtriangles_per_section > 0
            };
            if !valid_singular(policy.singular_coarse) || !valid_singular(policy.singular_fine) {
                return Err(BemError::InvalidParameters(
                    "quadrature singular orders must be positive".to_string(),
                ));
            }
            if !policy.absolute_tolerance.is_finite()
                || policy.absolute_tolerance < 0.0
                || !policy.relative_tolerance.is_finite()
                || policy.relative_tolerance < 0.0
            {
                return Err(BemError::InvalidParameters(
                    "quadrature tolerances must be finite and non-negative".to_string(),
                ));
            }
        }
        if self.max_iterations == 0 || self.gmres_restart == 0 {
            return Err(BemError::InvalidParameters(
                "max_iterations and gmres_restart must be non-zero".to_string(),
            ));
        }
        if !self.tolerance.is_finite() || self.tolerance <= 0.0 {
            return Err(BemError::InvalidParameters(
                "tolerance must be finite and positive".to_string(),
            ));
        }
        if !self.beta_scale.is_finite() || self.beta_scale < 0.0 {
            return Err(BemError::InvalidParameters(
                "beta_scale must be finite and non-negative".to_string(),
            ));
        }
        if !self.hmatrix_aca_tolerance.is_finite() || self.hmatrix_aca_tolerance <= 0.0 {
            return Err(BemError::InvalidParameters(
                "hmatrix_aca_tolerance must be finite and positive".to_string(),
            ));
        }
        if !self.compressed_exact_residual_tolerance.is_finite()
            || self.compressed_exact_residual_tolerance <= 0.0
        {
            return Err(BemError::InvalidParameters(
                "compressed_exact_residual_tolerance must be finite and positive".to_string(),
            ));
        }
        if let Some(budget) = self.resource_budget {
            budget.validate().map_err(BemError::ResourceAdmission)?;
        }
        Ok(())
    }

    /// Set the linear solver method
    pub fn with_solver_method(mut self, method: SolverMethod) -> Self {
        self.solver_method = method;
        self
    }

    /// Set the assembly method
    pub fn with_assembly_method(mut self, method: AssemblyMethod) -> Self {
        self.assembly_method = method;
        self
    }

    /// Set maximum iterations for iterative solvers
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Set tolerance for iterative solvers
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Configure H-matrix ACA accuracy and the optional per-block rank cap.
    ///
    /// A rank of zero keeps the size-based automatic cap. The values are
    /// checked by [`Self::validate`] before any assembly starts.
    pub fn with_hmatrix_accuracy(mut self, aca_tolerance: f64, max_rank: usize) -> Self {
        self.hmatrix_aca_tolerance = aca_tolerance;
        self.hmatrix_max_rank = max_rank;
        self
    }

    /// Set bounded retries that tighten H-matrix compression after failure.
    pub fn with_compressed_retries(mut self, retries: usize) -> Self {
        self.compressed_retry_count = retries;
        self
    }

    /// Enable bounded independent exact-TBEM residual certification for
    /// compressed H-matrix/SLFMM/MLFMM solves. Constant-panel layouts are
    /// supported on all compressed CPU backends; higher-order layouts are
    /// certified on H-matrix, SLFMM, MLFMM, and directional-MLFMM assembly
    /// when they fit the bounded matrix-free reference budget.
    ///
    /// The reference assembly is intentionally opt-in because it is O(N²)
    /// in memory/work. A zero `max_dofs` disables the certification pass.
    pub fn with_compressed_exact_residual_certification(
        mut self,
        max_dofs: usize,
        tolerance: f64,
    ) -> Self {
        self.compressed_exact_residual_dof_limit = max_dofs;
        self.compressed_exact_residual_tolerance = tolerance;
        self
    }

    /// Set fail-fast host/device/work limits for future solves.
    pub fn with_resource_budget(mut self, budget: BemResourceBudget) -> Self {
        self.resource_budget = Some(budget);
        self
    }

    /// Enable checked adjacent-order quadrature admission for the selected panel
    /// basis. The estimator runs before dense or compressed assembly and returns a
    /// typed [`BemError::InvalidParameters`] diagnostic when a configured pair
    /// fails its error budget.
    pub fn with_quadrature_policy(mut self, policy: TbemQuadraturePolicy) -> Self {
        self.quadrature_policy = Some(policy);
        self
    }

    /// Disable the optional quadrature admission gate.
    pub fn without_quadrature_policy(mut self) -> Self {
        self.quadrature_policy = None;
        self
    }

    /// Use a checked multilayer porous stack for impedance boundaries.
    ///
    /// The problem must select [`BoundaryConditionType::Impedance`]. The
    /// stack is validated at solver admission and its dimensional
    /// velocity/pressure admittance is recomputed for every problem
    /// frequency.
    pub fn with_layered_porous_surface(
        mut self,
        surface: crate::materials::LayeredPorousSurface,
    ) -> Self {
        self.layered_porous_surface = Some(surface);
        self
    }

    /// Remove a previously configured multilayer porous impedance model.
    pub fn without_layered_porous_surface(mut self) -> Self {
        self.layered_porous_surface = None;
        self
    }

    /// Use a checked local thin-shell mass--spring--damper impedance model.
    ///
    /// The problem must select [`BoundaryConditionType::Impedance`]. The
    /// shell's dimensional admittance is evaluated at every solve frequency.
    pub fn with_thin_shell_surface(mut self, surface: crate::materials::ThinShellSurface) -> Self {
        self.thin_shell_surface = Some(surface);
        self
    }

    /// Remove a previously configured thin-shell impedance model.
    pub fn without_thin_shell_surface(mut self) -> Self {
        self.thin_shell_surface = None;
        self
    }

    fn admit_resources(
        &self,
        elements: &[Element],
        rhs_count: usize,
        frequency_count: usize,
    ) -> Result<(), BemError> {
        let Some(budget) = self.resource_budget else {
            return Ok(());
        };
        let active_dofs = count_dofs(elements);
        let request = BemResourceRequest::new(
            self.assembly_method.resolve(active_dofs),
            active_dofs,
            rhs_count,
            frequency_count,
        )
        .with_hmatrix_rank(self.hmatrix_max_rank);
        budget
            .admit(request)
            .map(|_| ())
            .map_err(BemError::ResourceAdmission)
    }

    /// Enable verbose output
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set the basis function order (Constant or Linear)
    pub fn with_basis_order(mut self, basis: BasisOrder) -> Self {
        self.basis_order = basis;
        self
    }

    /// Select the global continuity contract for the basis.
    pub fn with_basis_continuity(mut self, continuity: BasisContinuity) -> Self {
        self.basis_continuity = continuity;
        self
    }

    /// Enable globally continuous quadratic Tri3/Quad4 unknowns.
    pub fn with_continuous_quadratic(self) -> Self {
        self.with_basis_order(BasisOrder::Quadratic)
            .with_basis_continuity(BasisContinuity::Continuous)
    }

    /// Enable globally continuous linear Tri3/Quad4 unknowns.
    pub fn with_continuous_linear(self) -> Self {
        self.with_basis_order(BasisOrder::Linear)
            .with_basis_continuity(BasisContinuity::Continuous)
    }

    /// Configure the preconditioner used by iterative solves. Only consumed
    /// when `solver_method == SolverMethod::Gmres`.
    pub fn with_preconditioner(mut self, p: BemPreconditioner) -> Self {
        self.preconditioner = p;
        self
    }

    /// Set the GMRES restart length (default 50).
    pub fn with_gmres_restart(mut self, restart: usize) -> Self {
        self.gmres_restart = restart.max(1);
        self
    }

    /// Set the GCRO-DR recycled subspace size used by `solve_frequency_sweep`.
    ///
    /// Use `0` for the old warm-start-only path. Values around 10-25 are the
    /// intended production range for smooth acoustic sweeps.
    pub fn with_frequency_recycle_subspace_size(mut self, size: usize) -> Self {
        self.frequency_recycle_subspace_size = size;
        self
    }

    /// Opt into the scalable single-node defaults used for larger problems.
    ///
    /// The crate default remains dense/direct because the always-on Mie
    /// accuracy gate is still the Stage-1 floor. This constructor makes the
    /// H-matrix/GMRES route explicit until its end-to-end solve accuracy is
    /// promoted to the same tier as dense TBEM.
    pub fn scalable_single_node() -> Self {
        Self::default()
            .with_solver_method(SolverMethod::Gmres)
            .with_assembly_method(AssemblyMethod::Auto)
    }

    /// Solve a BEM problem
    ///
    /// # Arguments
    /// * `problem` - The BEM problem definition
    ///
    /// # Returns
    /// Solution containing surface pressures and methods to evaluate fields
    pub fn solve(&self, problem: &BemProblem) -> Result<BemSolution, BemError> {
        self.validate()?;
        problem.validate()?;
        if self.verbose {
            log::info!(
                "Solving BEM problem: {} elements, ka = {:.3}",
                problem.mesh.elements.len(),
                problem.ka()
            );
        }

        let elements = self.prepare_elements(problem)?;
        self.admit_resources(&elements, 1, 1)?;

        if problem.bc_type == BoundaryConditionType::Soft {
            return self.solve_soft_dirichlet_indirect(problem, elements);
        }

        let ka_problem = problem.ka();
        let global_sign = PhysicsParams::optimal_formulation_sign(ka_problem);
        // A regular formulation must remove the Burton–Miller operator term
        // as well as its incident-field RHS contribution.  Passing a unit
        // scale here used to leave the stabilized matrix active while
        // selecting the regular RHS, which produced a physically inconsistent
        // formulation across TBEM and all compressed backends.
        let global_beta_scale = if problem.use_burton_miller {
            PhysicsParams::optimal_beta_scale(ka_problem)
        } else {
            0.0
        };

        if self.verbose {
            log::info!(
                "Stability parameters: sign = {}, beta_scale = {:.2}",
                global_sign,
                global_beta_scale
            );
        }

        // This value drives Auto dispatch and H-matrix retry rank caps.  It
        // must count global unknowns, not surface panels: higher-order Quad4
        // panels otherwise make retries choose rank 1–2 even when dozens of
        // basis DOFs are present, causing avoidable GMRES stagnation.
        let active_dofs = count_dofs(&elements);
        let adaptive_hmatrix = self.assembly_method.resolve(active_dofs) == AssemblyMethod::Hmatrix
            && self.compressed_retry_count > 0;
        // A compressed solve can stagnate on small, high-order systems even
        // after ACA tightening (the operator is still useful for matvecs, but
        // GMRES sees the discretization's poor conditioning).  Keep a
        // bounded, explicit dense safety net for discontinuous CPU H-matrices;
        // large systems remain fail-closed instead of silently materializing
        // an O(N²) fallback.
        let allow_small_hmatrix_dense_fallback = self.assembly_method.resolve(active_dofs)
            == AssemblyMethod::Hmatrix
            && self.basis_continuity == BasisContinuity::Discontinuous
            && active_dofs <= 256;
        let mut attempt_solver = self.clone();
        let mut retry = 0;
        let surface_pressure = loop {
            let assembly_result = attempt_solver.assemble_system_stabilized(
                &elements,
                &problem.mesh.nodes,
                &problem.physics,
                global_sign,
                global_beta_scale,
            )?;
            let rhs: Vec<Complex64> = match &assembly_result {
                AssemblyResult::Dense(_, rhs) => rhs.clone(),
                AssemblyResult::Slfmm(system) => system.rhs.to_vec(),
                AssemblyResult::Mlfmm(system) => system.rhs.to_vec(),
                AssemblyResult::Hmatrix(_, rhs) => rhs.clone(),
            };
            let rhs = attempt_solver.add_incident_field_rhs_stabilized(
                rhs,
                &elements,
                &problem.mesh.nodes,
                &problem.incident_field,
                &problem.physics,
                problem.use_burton_miller,
                global_sign,
                global_beta_scale,
            );
            let solved =
                attempt_solver.solve_assembled_system(assembly_result, &rhs, &elements, problem);
            match solved {
                Ok(solution) => break solution,
                Err(error)
                    if adaptive_hmatrix
                        && retry < attempt_solver.compressed_retry_count
                        && matches!(
                            &error,
                            BemError::ResidualCheckFailed { .. } | BemError::SolverFailed(_)
                        ) =>
                {
                    retry += 1;
                    attempt_solver.hmatrix_aca_tolerance =
                        (attempt_solver.hmatrix_aca_tolerance * 0.1).max(1e-12);
                    let current_rank = if attempt_solver.hmatrix_max_rank == 0 {
                        (active_dofs / 4).saturating_add(1)
                    } else {
                        attempt_solver.hmatrix_max_rank
                    };
                    attempt_solver.hmatrix_max_rank = current_rank
                        .saturating_mul(2)
                        .max(current_rank.saturating_add(1));
                    log::warn!(
                        "compressed BEM solve failed ({error}); retrying H-matrix assembly with ACA tolerance {:.3e} and rank {} ({}/{})",
                        attempt_solver.hmatrix_aca_tolerance,
                        attempt_solver.hmatrix_max_rank,
                        retry,
                        attempt_solver.compressed_retry_count
                    );
                }
                Err(error)
                    if allow_small_hmatrix_dense_fallback
                        && matches!(
                            &error,
                            BemError::ResidualCheckFailed { .. } | BemError::SolverFailed(_)
                        ) =>
                {
                    let mut dense_solver = attempt_solver.clone();
                    dense_solver.assembly_method = AssemblyMethod::Tbem;
                    dense_solver.solver_method = SolverMethod::Direct;
                    let dense_assembly = dense_solver.assemble_system_stabilized(
                        &elements,
                        &problem.mesh.nodes,
                        &problem.physics,
                        global_sign,
                        global_beta_scale,
                    )?;
                    log::warn!(
                        "small H-matrix solve failed ({error}); using bounded dense TBEM fallback for {active_dofs} DOFs"
                    );
                    break dense_solver.solve_assembled_system(
                        dense_assembly,
                        &rhs,
                        &elements,
                        problem,
                    )?;
                }
                Err(error) => return Err(error),
            }
        };

        if self.verbose {
            log::info!(
                "Solution complete. Max surface pressure: {:.6}",
                surface_pressure
                    .iter()
                    .map(|p| p.norm())
                    .fold(0.0f64, f64::max)
            );
        }

        Ok(BemSolution {
            surface_pressure,
            elements,
            nodes: problem.mesh.nodes.clone(),
            incident_field: problem.incident_field.clone(),
            physics: problem.physics.clone(),
            basis_order: self.basis_order,
            representation: BemSolutionRepresentation::DirectCauchy,
        })
    }

    /// Solve one validated mesh/physics problem for several incident fields.
    ///
    /// Dense/TBEM systems are assembled and LU-factorized once. Compressed
    /// systems use one block-GMRES operator path, so the returned solutions
    /// are ordered like `incident_fields` without silently re-solving each
    /// right-hand side independently.
    pub fn solve_many(
        &self,
        problem: &BemProblem,
        incident_fields: &[IncidentField],
    ) -> Result<Vec<BemSolution>, BemError> {
        self.validate()?;
        problem.validate()?;
        if incident_fields.is_empty() {
            return Ok(Vec::new());
        }

        let elements = self.prepare_elements(problem)?;
        self.admit_resources(&elements, incident_fields.len(), 1)?;

        if problem.bc_type == BoundaryConditionType::Soft {
            let matrix = self.assemble_soft_dirichlet_matrix(problem, &elements)?;
            let rhs_columns: Vec<Vec<Complex64>> = if self.basis_order == BasisOrder::Constant {
                incident_fields
                    .iter()
                    .map(|incident_field| {
                        Ok(self.soft_dirichlet_rhs_for_incident(
                            &problem.physics,
                            incident_field,
                            &elements,
                        ))
                    })
                    .collect::<Result<_, BemError>>()?
            } else {
                incident_fields
                    .iter()
                    .map(|incident_field| {
                        self.soft_dirichlet_rhs_higher_order_for_incident(
                            &problem.physics,
                            incident_field,
                            &problem.mesh.nodes,
                            &elements,
                        )
                    })
                    .collect::<Result<_, BemError>>()?
            };
            let num_dofs = matrix.nrows();
            let rhs_block = DenseMatrix::from_fn(num_dofs, rhs_columns.len(), |row, column| {
                rhs_columns[column][row]
            });
            let factor =
                lu_factorize(&matrix).map_err(|error| BemError::SolverFailed(error.to_string()))?;
            let solution_values = factor
                .solve_many(rhs_block.as_slice(), rhs_columns.len())
                .map_err(|error| BemError::SolverFailed(error.to_string()))?;
            let solution_block =
                DenseMatrix::from_row_major_vec(num_dofs, rhs_columns.len(), solution_values);
            let operator = DenseMatrixOperator(&matrix);
            for (column, rhs) in rhs_columns.iter().enumerate() {
                self.check_exact_residual(
                    &operator,
                    &solution_block.column_to_vec(column),
                    rhs,
                    "soft-dirichlet-block-rhs",
                )?;
            }
            return Ok((0..rhs_columns.len())
                .map(|column| BemSolution {
                    surface_pressure: solution_block.column_to_vec(column),
                    elements: elements.clone(),
                    nodes: problem.mesh.nodes.clone(),
                    incident_field: incident_fields[column].clone(),
                    physics: problem.physics.clone(),
                    basis_order: self.basis_order,
                    representation: BemSolutionRepresentation::IndirectSingleLayer,
                })
                .collect());
        }

        let ka_problem = problem.ka();
        let global_sign = PhysicsParams::optimal_formulation_sign(ka_problem);
        let global_beta_scale = if problem.use_burton_miller {
            PhysicsParams::optimal_beta_scale(ka_problem)
        } else {
            0.0
        };
        let assembly = self.assemble_system_stabilized(
            &elements,
            &problem.mesh.nodes,
            &problem.physics,
            global_sign,
            global_beta_scale,
        )?;

        let base_rhs = match &assembly {
            AssemblyResult::Dense(_, rhs) | AssemblyResult::Hmatrix(_, rhs) => rhs.clone(),
            AssemblyResult::Slfmm(system) => system.rhs.to_vec(),
            AssemblyResult::Mlfmm(system) => system.rhs.to_vec(),
        };
        let rhs_columns: Vec<Vec<Complex64>> = incident_fields
            .iter()
            .map(|incident_field| {
                self.add_incident_field_rhs_stabilized(
                    base_rhs.clone(),
                    &elements,
                    &problem.mesh.nodes,
                    incident_field,
                    &problem.physics,
                    problem.use_burton_miller,
                    global_sign,
                    global_beta_scale,
                )
            })
            .collect();
        let num_dofs = assembly.num_dofs();
        let num_rhs = rhs_columns.len();

        let solution_block = match assembly {
            AssemblyResult::Dense(matrix, _) => {
                let dense_diag: Vec<Complex64> =
                    (0..matrix.nrows()).map(|i| matrix[[i, i]]).collect();
                let precond = self.build_preconditioner(
                    &elements,
                    &problem.mesh.nodes,
                    &problem.physics,
                    Some(&dense_diag),
                    None,
                )?;
                if self.solver_method == SolverMethod::Direct {
                    let rhs_block =
                        DenseMatrix::from_fn(num_dofs, num_rhs, |row, rhs| rhs_columns[rhs][row]);
                    let factor = lu_factorize(&matrix)
                        .map_err(|error| BemError::SolverFailed(error.to_string()))?;
                    let solution = factor
                        .solve_many(rhs_block.as_slice(), num_rhs)
                        .map_err(|error| BemError::SolverFailed(error.to_string()))?;
                    let operator = DenseMatrixOperator(&matrix);
                    for rhs_index in 0..num_rhs {
                        self.check_exact_residual(
                            &operator,
                            &solution
                                .chunks(num_rhs)
                                .map(|row| row[rhs_index])
                                .collect::<Vec<_>>(),
                            &rhs_columns[rhs_index],
                            "dense-direct-multi-rhs",
                        )?;
                    }
                    DenseMatrix::from_row_major_vec(num_dofs, num_rhs, solution)
                } else if self.solver_method == SolverMethod::Gmres {
                    self.solve_compressed_many(
                        &DenseMatrixOperator(&matrix),
                        &precond,
                        &rhs_columns,
                        "dense-block-gmres",
                    )?
                } else {
                    let mut solution = DenseMatrix::zeros(num_dofs, num_rhs);
                    for (rhs_index, rhs) in rhs_columns.iter().enumerate() {
                        let column = self.solve_dense_system(&matrix, rhs, &precond)?;
                        solution.set_column(rhs_index, &column);
                    }
                    solution
                }
            }
            AssemblyResult::Slfmm(system) => {
                let precond = if self.preconditioner == BemPreconditioner::HierarchicalIlu {
                    let nearfield_matrix =
                        system.try_extract_near_field_matrix().map_err(|error| {
                            BemError::SolverFailed(format!("SLFMM near-field: {error}"))
                        })?;
                    let nearfield =
                        sonium_solvers::sparse::CsrMatrix::from_dense(&nearfield_matrix, 1.0e-15);
                    let hil = HiluPreconditioner::try_from_csr(&nearfield).map_err(|error| {
                        BemError::SolverFailed(format!("SLFMM hierarchical ILU: {error}"))
                    })?;
                    BemPrecondImpl::Hilu(Box::new(hil))
                } else {
                    BemPrecondImpl::Identity
                };
                self.solve_compressed_many(&system, &precond, &rhs_columns, "slfmm-block-gmres")?
            }
            AssemblyResult::Mlfmm(system) => {
                use crate::core::solver::MlfmmOperator;

                let precond = if self.preconditioner == BemPreconditioner::HierarchicalIlu {
                    let nearfield_matrix =
                        system.try_extract_near_field_matrix().map_err(|error| {
                            BemError::SolverFailed(format!("MLFMM near-field: {error}"))
                        })?;
                    let nearfield =
                        sonium_solvers::sparse::CsrMatrix::from_dense(&nearfield_matrix, 1.0e-15);
                    let hil = HiluPreconditioner::try_from_csr(&nearfield).map_err(|error| {
                        BemError::SolverFailed(format!("MLFMM hierarchical ILU: {error}"))
                    })?;
                    BemPrecondImpl::Hilu(Box::new(hil))
                } else {
                    self.build_preconditioner(
                        &elements,
                        &problem.mesh.nodes,
                        &problem.physics,
                        None,
                        None,
                    )?
                };
                let operator = MlfmmOperator::new(*system);
                self.solve_compressed_many(&operator, &precond, &rhs_columns, "mlfmm-block-gmres")?
            }
            AssemblyResult::Hmatrix(hmatrix, _) => {
                let h_diag = hmatrix_diagonal(&hmatrix)?;
                let precond = self.build_preconditioner(
                    &elements,
                    &problem.mesh.nodes,
                    &problem.physics,
                    None,
                    Some(&h_diag),
                )?;
                self.solve_compressed_many(&hmatrix, &precond, &rhs_columns, "hmatrix-block-gmres")?
            }
        };

        Ok(incident_fields
            .iter()
            .cloned()
            .enumerate()
            .map(|(rhs_index, incident_field)| BemSolution {
                surface_pressure: solution_block.column_to_vec(rhs_index),
                elements: elements.clone(),
                nodes: problem.mesh.nodes.clone(),
                incident_field,
                physics: problem.physics.clone(),
                basis_order: self.basis_order,
                representation: BemSolutionRepresentation::DirectCauchy,
            })
            .collect())
    }

    /// Solve arbitrary forward driver/unit-load RHS columns on one assembled
    /// BEM problem.
    ///
    /// The supplied columns are interpreted as `A X = B` loads in assembled
    /// DOF order. This is the forward counterpart to
    /// [`Self::solve_adjoint_many`] and is intended for driver unit loads and
    /// objective terms that are not represented by an [`IncidentField`].
    /// Dense direct solves reuse one LU factorization; GMRES and compressed
    /// paths retain the exact residual checks used by production solves.
    pub fn solve_rhs_many(
        &self,
        problem: &BemProblem,
        rhs_columns: &[Vec<Complex64>],
    ) -> Result<Vec<BemLoadSolution>, BemError> {
        self.validate()?;
        problem.validate()?;
        if rhs_columns.is_empty() {
            return Ok(Vec::new());
        }
        let elements = self.prepare_elements(problem)?;
        self.admit_resources(&elements, rhs_columns.len(), 1)?;
        let num_dofs = count_dofs(&elements);
        for (rhs_index, rhs) in rhs_columns.iter().enumerate() {
            if rhs.len() != num_dofs {
                return Err(BemError::InvalidParameters(format!(
                    "forward RHS {rhs_index} has {} entries, expected {num_dofs}",
                    rhs.len()
                )));
            }
            if rhs
                .iter()
                .any(|value| !value.re.is_finite() || !value.im.is_finite())
            {
                return Err(BemError::InvalidParameters(format!(
                    "forward RHS {rhs_index} contains non-finite values"
                )));
            }
        }

        let ka = problem.ka();
        let sign = PhysicsParams::optimal_formulation_sign(ka);
        let beta_scale = if problem.use_burton_miller {
            PhysicsParams::optimal_beta_scale(ka)
        } else {
            0.0
        };
        let assembly = self.assemble_system_stabilized(
            &elements,
            &problem.mesh.nodes,
            &problem.physics,
            sign,
            beta_scale,
        )?;
        let (solution_block, residuals) = match assembly {
            AssemblyResult::Dense(matrix, _) => {
                let operator = DenseMatrixOperator(&matrix);
                if self.solver_method == SolverMethod::Direct {
                    let rhs_block =
                        DenseMatrix::from_fn(num_dofs, rhs_columns.len(), |row, col| {
                            rhs_columns[col][row]
                        });
                    let factor = lu_factorize(&matrix)
                        .map_err(|error| BemError::SolverFailed(error.to_string()))?;
                    let solution = factor
                        .solve_many(rhs_block.as_slice(), rhs_columns.len())
                        .map_err(|error| BemError::SolverFailed(error.to_string()))?;
                    let solution =
                        DenseMatrix::from_row_major_vec(num_dofs, rhs_columns.len(), solution);
                    let residuals = exact_block_residuals(&operator, &solution, rhs_columns)?;
                    (solution, residuals)
                } else {
                    let dense_diag: Vec<Complex64> = (0..matrix.nrows())
                        .map(|index| matrix[[index, index]])
                        .collect();
                    let precond = self.build_preconditioner(
                        &elements,
                        &problem.mesh.nodes,
                        &problem.physics,
                        Some(&dense_diag),
                        None,
                    )?;
                    let solution = self.solve_compressed_many(
                        &operator,
                        &precond,
                        rhs_columns,
                        "dense-forward-load-block-gmres",
                    )?;
                    let residuals = exact_block_residuals(&operator, &solution, rhs_columns)?;
                    (solution, residuals)
                }
            }
            AssemblyResult::Slfmm(system) => {
                let nearfield_matrix = system.try_extract_near_field_matrix().map_err(|error| {
                    BemError::SolverFailed(format!("SLFMM near-field: {error}"))
                })?;
                let nearfield =
                    sonium_solvers::sparse::CsrMatrix::from_dense(&nearfield_matrix, 1.0e-15);
                let precond = if self.preconditioner == BemPreconditioner::HierarchicalIlu {
                    let hil = HiluPreconditioner::try_from_csr(&nearfield).map_err(|error| {
                        BemError::SolverFailed(format!("SLFMM hierarchical ILU: {error}"))
                    })?;
                    BemPrecondImpl::Hilu(Box::new(hil))
                } else {
                    let ilu = sonium_solvers::preconditioners::IluPreconditioner::try_from_csr(
                        &nearfield,
                    )
                    .map_err(|error| {
                        BemError::SolverFailed(format!("SLFMM near-field ILU: {error}"))
                    })?;
                    BemPrecondImpl::Ilu(ilu)
                };
                let operator = crate::core::solver::SlfmmOperator::new(system);
                let solution = self.solve_compressed_many(
                    &operator,
                    &precond,
                    rhs_columns,
                    "slfmm-forward-load-block-gmres",
                )?;
                let residuals = exact_block_residuals(&operator, &solution, rhs_columns)?;
                (solution, residuals)
            }
            AssemblyResult::Mlfmm(system) => {
                let precond = if self.preconditioner == BemPreconditioner::HierarchicalIlu {
                    let nearfield_matrix =
                        system.try_extract_near_field_matrix().map_err(|error| {
                            BemError::SolverFailed(format!("MLFMM near-field: {error}"))
                        })?;
                    let nearfield =
                        sonium_solvers::sparse::CsrMatrix::from_dense(&nearfield_matrix, 1.0e-15);
                    let hil = HiluPreconditioner::try_from_csr(&nearfield).map_err(|error| {
                        BemError::SolverFailed(format!("MLFMM hierarchical ILU: {error}"))
                    })?;
                    BemPrecondImpl::Hilu(Box::new(hil))
                } else if self.preconditioner == BemPreconditioner::None {
                    let nearfield_matrix =
                        system.try_extract_near_field_matrix().map_err(|error| {
                            BemError::SolverFailed(format!("MLFMM near-field: {error}"))
                        })?;
                    let nearfield =
                        sonium_solvers::sparse::CsrMatrix::from_dense(&nearfield_matrix, 1.0e-15);
                    let ilu = sonium_solvers::preconditioners::IluPreconditioner::try_from_csr(
                        &nearfield,
                    )
                    .map_err(|error| {
                        BemError::SolverFailed(format!("MLFMM adjoint near-field ILU: {error}"))
                    })?;
                    BemPrecondImpl::Ilu(ilu)
                } else {
                    self.build_preconditioner(
                        &elements,
                        &problem.mesh.nodes,
                        &problem.physics,
                        None,
                        None,
                    )?
                };
                let operator = crate::core::solver::MlfmmOperator::new(*system);
                let solution = self.solve_compressed_many(
                    &operator,
                    &precond,
                    rhs_columns,
                    "mlfmm-forward-load-block-gmres",
                )?;
                let residuals = exact_block_residuals(&operator, &solution, rhs_columns)?;
                (solution, residuals)
            }
            AssemblyResult::Hmatrix(hmatrix, _) => {
                let h_diag = hmatrix_diagonal(&hmatrix)?;
                let precond = self.build_preconditioner(
                    &elements,
                    &problem.mesh.nodes,
                    &problem.physics,
                    None,
                    Some(&h_diag),
                )?;
                let solution = self.solve_compressed_many(
                    &hmatrix,
                    &precond,
                    rhs_columns,
                    "hmatrix-forward-load-block-gmres",
                )?;
                let residuals = exact_block_residuals(&hmatrix, &solution, rhs_columns)?;
                (solution, residuals)
            }
        };
        let exact_reference_residuals = self.certify_compressed_batch(
            problem,
            &elements,
            sign,
            beta_scale,
            self.assembly_method.resolve(num_dofs),
            &solution_block,
            rhs_columns,
            false,
        )?;

        Ok(rhs_columns
            .iter()
            .enumerate()
            .map(|(rhs_index, _)| BemLoadSolution {
                coefficients: solution_block.column_to_vec(rhs_index),
                relative_residual: residuals[rhs_index],
                exact_reference_residual: exact_reference_residuals
                    .as_ref()
                    .and_then(|values| values.get(rhs_index).copied()),
                elements: elements.clone(),
                nodes: problem.mesh.nodes.clone(),
                physics: problem.physics.clone(),
                basis_order: self.basis_order,
            })
            .collect())
    }

    /// Alias for [`Self::solve_rhs_many`] emphasizing driver/unit-load use.
    pub fn solve_driver_loads(
        &self,
        problem: &BemProblem,
        rhs_columns: &[Vec<Complex64>],
    ) -> Result<Vec<BemLoadSolution>, BemError> {
        self.solve_rhs_many(problem, rhs_columns)
    }

    /// Solve a forward load and its Hermitian adjoint objective, then return
    /// the fixed-geometry wavenumber derivative of `Re(gᴴ x)`.
    ///
    /// The supplied `forward_rhs` is used for `A x = b`, while
    /// `objective_gradient` is used for `Aᴴ λ = g`.  The derivative follows
    /// the same frequency-to-wavenumber convention as
    /// [`build_tbem_system_stabilized_wavenumber_derivative`].  This checked
    /// high-level contract admits dense direct constant-panel TBEM through the
    /// analytic assembler and dense direct linear/quadratic TBEM through the
    /// production-assembly centered fallback. Compressed higher-order,
    /// geometry, interface, device, and distributed derivatives remain
    /// separate contracts.
    pub fn solve_wavenumber_objective_gradient(
        &self,
        problem: &BemProblem,
        forward_rhs: &[Complex64],
        objective_gradient: &[Complex64],
    ) -> Result<f64, BemError> {
        self.validate()?;
        problem.validate()?;
        let elements = self.prepare_elements(problem)?;
        let num_dofs = count_dofs(&elements);
        let resolved_assembly = self.assembly_method.resolve(num_dofs);
        let compressed_derivative = self.solver_method == SolverMethod::Gmres
            && matches!(
                resolved_assembly,
                AssemblyMethod::Hmatrix
                    | AssemblyMethod::Slfmm
                    | AssemblyMethod::Mlfmm
                    | AssemblyMethod::DirectionalMlfmm
            );
        let hmatrix_derivative =
            compressed_derivative && resolved_assembly == AssemblyMethod::Hmatrix;
        if self.solver_method == SolverMethod::Direct
            && resolved_assembly == AssemblyMethod::Tbem
            && (self.basis_order != BasisOrder::Constant
                || self.basis_continuity != BasisContinuity::Discontinuous)
        {
            return self.solve_dense_objective_gradient_finite_difference(
                problem,
                forward_rhs,
                objective_gradient,
                DenseObjectiveGradientParameter::Wavenumber,
            );
        }
        if (!compressed_derivative && self.solver_method != SolverMethod::Direct)
            || (!compressed_derivative && resolved_assembly != AssemblyMethod::Tbem)
            || (!matches!(
                resolved_assembly,
                AssemblyMethod::Hmatrix
                    | AssemblyMethod::Slfmm
                    | AssemblyMethod::Mlfmm
                    | AssemblyMethod::DirectionalMlfmm
            ) && (self.basis_order != BasisOrder::Constant
                || self.basis_continuity != BasisContinuity::Discontinuous))
        {
            return Err(BemError::NotImplemented(
                "wavenumber objective gradients currently require dense direct constant-panel TBEM"
                    .to_string(),
            ));
        }

        let forward = self
            .solve_rhs_many(problem, &[forward_rhs.to_vec()])?
            .into_iter()
            .next()
            .ok_or_else(|| {
                BemError::SolverFailed("forward gradient solve returned no solution".to_string())
            })?;
        let adjoint = self
            .solve_adjoint_many(problem, &[objective_gradient.to_vec()])?
            .into_iter()
            .next()
            .ok_or_else(|| {
                BemError::SolverFailed("adjoint gradient solve returned no solution".to_string())
            })?;

        let ka = problem.ka();
        let sign = PhysicsParams::optimal_formulation_sign(ka);
        let beta_scale = if problem.use_burton_miller {
            PhysicsParams::optimal_beta_scale(ka)
        } else {
            0.0
        };
        if hmatrix_derivative {
            let use_continuous_proxies =
                matches!(self.basis_order, BasisOrder::Linear | BasisOrder::Quadratic)
                    && self.basis_continuity == BasisContinuity::Continuous;
            let (continuous_proxies, continuous_element_proxies) = if use_continuous_proxies {
                let (proxies, element_proxies) = build_continuous_hmatrix_proxy_elements(
                    &elements,
                    &problem.mesh.nodes,
                    self.basis_order,
                );
                (Some(proxies), Some(element_proxies))
            } else {
                (None, None)
            };
            let derivative_elements = continuous_proxies.as_deref().unwrap_or(&elements);
            let clusters = if let Some(element_proxies) = continuous_element_proxies.as_deref() {
                let base_clusters = Self::build_hmatrix_clusters(&elements);
                expand_hmatrix_clusters_for_proxies(&base_clusters, element_proxies)
            } else {
                Self::build_hmatrix_clusters(&elements)
            };
            let max_rank = if self.hmatrix_max_rank == 0 {
                derivative_elements.len() / 4 + 1
            } else {
                self.hmatrix_max_rank
            };
            let derivative =
                crate::core::assembly::hmatrix::build_hmatrix_stabilized_wavenumber_derivative(
                    derivative_elements,
                    &problem.mesh.nodes,
                    &clusters,
                    &problem.physics,
                    self.hmatrix_aca_tolerance,
                    max_rank,
                    crate::core::assembly::hmatrix::AcaVariant::default(),
                    crate::core::assembly::hmatrix::LowRankPrecision::default(),
                    sign,
                    beta_scale,
                    6,
                )
                .map_err(|error| BemError::InvalidParameters(error.to_string()))?;
            let derivative_rhs = if self.basis_order == BasisOrder::Constant
                && self.basis_continuity == BasisContinuity::Discontinuous
            {
                build_tbem_system_stabilized_wavenumber_derivative(
                    &elements,
                    &problem.mesh.nodes,
                    &problem.physics,
                    sign,
                    beta_scale,
                )
                .rhs
            } else {
                // Higher-order public objective-gradient calls receive an
                // externally supplied fixed RHS, so its wavenumber
                // derivative is zero.  Keep the dimension in the global DOF
                // space while the H-matrix derivative handles the operator.
                vec![Complex64::new(0.0, 0.0); num_dofs]
            };
            let derivative_action = derivative.matvec(forward.coefficients());
            if derivative_action.len() != adjoint.coefficients().len()
                || derivative_rhs.len() != derivative_action.len()
            {
                return Err(BemError::InvalidParameters(
                    "H-matrix wavenumber derivative dimension mismatch".to_string(),
                ));
            }
            let gradient = adjoint
                .coefficients()
                .iter()
                .zip(derivative_rhs.iter().zip(derivative_action.iter()))
                .map(|(adjoint, (rhs, action))| adjoint.conj() * (*rhs - *action))
                .sum::<Complex64>();
            return Ok(gradient.re);
        }

        if compressed_derivative {
            if resolved_assembly == AssemblyMethod::Slfmm
                && (self.basis_order != BasisOrder::Constant
                    || self.basis_continuity == BasisContinuity::Discontinuous)
            {
                use crate::core::assembly::slfmm::{
                    build_slfmm_system_stabilized_checked,
                    build_slfmm_system_stabilized_with_basis_checked,
                    build_slfmm_wavenumber_derivative,
                    build_slfmm_wavenumber_derivative_with_basis_checked,
                };
                use crate::core::mesh::octree::Octree;
                use crate::core::types::Cluster;
                use std::collections::HashMap;

                let centers: Vec<[f64; 3]> =
                    elements.iter().map(|element| element.center).collect();
                let mut octree = Octree::build(&centers, 32, 8);
                octree.compute_interaction_lists(1.8);
                let leaf_indices = octree.leaves();
                let mut node_to_cluster = HashMap::with_capacity(leaf_indices.len());
                for (cluster_index, &node_index) in leaf_indices.iter().enumerate() {
                    node_to_cluster.insert(node_index, cluster_index);
                }
                let mut clusters = Vec::with_capacity(leaf_indices.len());
                for &node_index in &leaf_indices {
                    let node = &octree.nodes[node_index];
                    let mut cluster = Cluster::new(node.center);
                    cluster.element_indices = node.element_indices.clone();
                    cluster.num_elements = cluster.element_indices.len();
                    cluster.radius = node.radius();
                    cluster.near_clusters = node
                        .near_clusters
                        .iter()
                        .filter_map(|index| node_to_cluster.get(index).copied())
                        .collect();
                    cluster.far_clusters = node
                        .far_clusters
                        .iter()
                        .filter_map(|index| node_to_cluster.get(index).copied())
                        .collect();
                    clusters.push(cluster);
                }
                let characteristic_radius = clusters
                    .iter()
                    .map(|cluster| cluster.radius)
                    .fold(0.0_f64, f64::max);
                let ka = problem.physics.wave_number * characteristic_radius;
                let n_terms = (ka.ceil() as usize + 3).clamp(5, 20);
                let n_theta = 2 * n_terms;
                let n_phi = 2 * n_theta;
                let base = if self.basis_order == BasisOrder::Constant
                    && self.basis_continuity == BasisContinuity::Discontinuous
                {
                    build_slfmm_system_stabilized_checked(
                        &elements,
                        &problem.mesh.nodes,
                        &clusters,
                        &problem.physics,
                        n_theta,
                        n_phi,
                        n_terms,
                        sign,
                        beta_scale,
                    )
                    .map_err(|error| BemError::InvalidParameters(error.to_string()))?
                } else {
                    build_slfmm_system_stabilized_with_basis_checked(
                        &elements,
                        &problem.mesh.nodes,
                        &clusters,
                        &problem.physics,
                        self.basis_order,
                        n_theta,
                        n_phi,
                        n_terms,
                        sign,
                        beta_scale,
                    )
                    .map_err(|error| BemError::InvalidParameters(error.to_string()))?
                };
                let derivative = if self.basis_order == BasisOrder::Constant
                    && self.basis_continuity == BasisContinuity::Discontinuous
                {
                    build_slfmm_wavenumber_derivative(
                        &elements,
                        &problem.mesh.nodes,
                        &clusters,
                        &problem.physics,
                        n_theta,
                        n_phi,
                        n_terms,
                        sign,
                        beta_scale,
                    )
                    .map_err(BemError::InvalidParameters)?
                } else {
                    build_slfmm_wavenumber_derivative_with_basis_checked(
                        &elements,
                        &problem.mesh.nodes,
                        &clusters,
                        &problem.physics,
                        self.basis_order,
                        n_theta,
                        n_phi,
                        n_terms,
                        sign,
                        beta_scale,
                    )
                    .map_err(|error| BemError::InvalidParameters(error.to_string()))?
                };
                let derivative_action = derivative.apply(&base, forward.coefficients());
                let gradient = adjoint
                    .coefficients()
                    .iter()
                    .zip(derivative_action.iter())
                    .map(|(adjoint, action)| -adjoint.conj() * *action)
                    .sum::<Complex64>();
                if !gradient.re.is_finite() {
                    return Err(BemError::InvalidParameters(
                        "compressed SLFMM wavenumber derivative is non-finite".to_string(),
                    ));
                }
                return Ok(gradient.re);
            }

            if matches!(
                resolved_assembly,
                AssemblyMethod::Mlfmm | AssemblyMethod::DirectionalMlfmm
            ) && (self.basis_order != BasisOrder::Constant
                || self.basis_continuity == BasisContinuity::Discontinuous)
            {
                use crate::core::assembly::mlfmm::{
                    build_cluster_tree, build_directional_mlfmm_system_stabilized_checked,
                    build_directional_mlfmm_system_stabilized_with_basis_checked,
                    build_directional_mlfmm_wavenumber_derivative,
                    build_directional_mlfmm_wavenumber_derivative_with_basis_checked,
                    build_mlfmm_system_stabilized_checked,
                    build_mlfmm_system_stabilized_with_basis_checked,
                    build_mlfmm_wavenumber_derivative,
                    build_mlfmm_wavenumber_derivative_with_basis_checked,
                };
                let cluster_levels = build_cluster_tree(&elements, 32, &problem.physics);
                let base = if resolved_assembly == AssemblyMethod::DirectionalMlfmm {
                    if self.basis_order == BasisOrder::Constant {
                        build_directional_mlfmm_system_stabilized_checked(
                            &elements,
                            &problem.mesh.nodes,
                            cluster_levels.clone(),
                            &problem.physics,
                            sign,
                            beta_scale,
                            Default::default(),
                        )
                    } else {
                        build_directional_mlfmm_system_stabilized_with_basis_checked(
                            &elements,
                            &problem.mesh.nodes,
                            cluster_levels.clone(),
                            &problem.physics,
                            self.basis_order,
                            sign,
                            beta_scale,
                            Default::default(),
                        )
                    }
                } else {
                    if self.basis_order == BasisOrder::Constant {
                        build_mlfmm_system_stabilized_checked(
                            &elements,
                            &problem.mesh.nodes,
                            cluster_levels.clone(),
                            &problem.physics,
                            sign,
                            beta_scale,
                        )
                    } else {
                        build_mlfmm_system_stabilized_with_basis_checked(
                            &elements,
                            &problem.mesh.nodes,
                            cluster_levels.clone(),
                            &problem.physics,
                            self.basis_order,
                            sign,
                            beta_scale,
                        )
                    }
                }
                .map_err(|error| BemError::InvalidParameters(error.to_string()))?;
                let derivative = if resolved_assembly == AssemblyMethod::DirectionalMlfmm {
                    if self.basis_order == BasisOrder::Constant {
                        build_directional_mlfmm_wavenumber_derivative(
                            &elements,
                            &problem.mesh.nodes,
                            cluster_levels,
                            &problem.physics,
                            sign,
                            beta_scale,
                            Default::default(),
                        )
                    } else {
                        build_directional_mlfmm_wavenumber_derivative_with_basis_checked(
                            &elements,
                            &problem.mesh.nodes,
                            cluster_levels,
                            &problem.physics,
                            self.basis_order,
                            sign,
                            beta_scale,
                            Default::default(),
                        )
                    }
                } else {
                    if self.basis_order == BasisOrder::Constant {
                        build_mlfmm_wavenumber_derivative(
                            &elements,
                            &problem.mesh.nodes,
                            cluster_levels,
                            &problem.physics,
                            sign,
                            beta_scale,
                        )
                    } else {
                        build_mlfmm_wavenumber_derivative_with_basis_checked(
                            &elements,
                            &problem.mesh.nodes,
                            cluster_levels,
                            &problem.physics,
                            self.basis_order,
                            sign,
                            beta_scale,
                        )
                    }
                }
                .map_err(BemError::InvalidParameters)?;
                let derivative_action = derivative.apply(&base, forward.coefficients());
                let gradient = adjoint
                    .coefficients()
                    .iter()
                    .zip(derivative_action.iter())
                    .map(|(adjoint, action)| -adjoint.conj() * *action)
                    .sum::<Complex64>();
                if !gradient.re.is_finite() {
                    return Err(BemError::InvalidParameters(
                        "compressed MLFMM wavenumber derivative is non-finite".to_string(),
                    ));
                }
                return Ok(gradient.re);
            }

            // Keep the derivative on the same compressed production path as
            // the forward and adjoint solves.  The previous implementation
            // applied an independently assembled O(N²) dense derivative
            // action here, defeating the memory/CPU bound of FMM on larger
            // simulations.  Reusing the bounded centered stencil preserves
            // the selected backend's approximation and preconditioner.
            return self.solve_compressed_objective_gradient_finite_difference(
                problem,
                forward_rhs,
                objective_gradient,
                DenseObjectiveGradientParameter::Wavenumber,
            );
        }

        let derivative = build_tbem_system_stabilized_wavenumber_derivative(
            &elements,
            &problem.mesh.nodes,
            &problem.physics,
            sign,
            beta_scale,
        );
        tbem_wavenumber_objective_gradient(
            &derivative,
            forward.coefficients(),
            adjoint.coefficients(),
        )
        .map_err(|error| BemError::InvalidParameters(error.to_string()))
    }

    /// Assemble the incident Burton--Miller RHS derivative at every active DOF.
    ///
    /// Constant-panel callers historically used the element-center helpers in
    /// [`IncidentField`]. Higher-order TBEM uses one collocation point per
    /// local DOF instead; for continuous quadratic bases shared DOFs are
    /// evaluated once with the first panel's normal, matching
    /// `add_incident_field_rhs_stabilized`.
    /// Solve fixed-geometry wavenumber objective gradients for several
    /// frequencies. Each entry in `forward_rhs_batches` supplies the load
    /// for `A(f) x = b`, while the matching entry in
    /// `objective_gradient_batches` supplies the right-hand side for
    /// `A(f)^H lambda = g`. Returned gradients follow `frequencies` in order
    /// and use the same convention as
    /// [`Self::solve_wavenumber_objective_gradient`].
    pub fn solve_wavenumber_objective_gradient_many(
        &self,
        problem: &BemProblem,
        frequencies: &[f64],
        forward_rhs_batches: &[Vec<Complex64>],
        objective_gradient_batches: &[Vec<Complex64>],
    ) -> Result<Vec<f64>, BemError> {
        self.validate()?;
        problem.validate()?;
        if frequencies.is_empty()
            || frequencies
                .iter()
                .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
        {
            return Err(BemError::InvalidParameters(
                "wavenumber-gradient frequency batch requires finite, positive frequencies"
                    .to_string(),
            ));
        }
        if forward_rhs_batches.len() != frequencies.len()
            || objective_gradient_batches.len() != frequencies.len()
        {
            return Err(BemError::InvalidParameters(format!(
                "wavenumber-gradient batch has {} frequencies, {} forward RHS batches, and {} objective-gradient batches",
                frequencies.len(),
                forward_rhs_batches.len(),
                objective_gradient_batches.len(),
            )));
        }

        let mut gradients = Vec::with_capacity(frequencies.len());
        for (index, &frequency) in frequencies.iter().enumerate() {
            let mut at_frequency = problem.clone();
            let mut physics = PhysicsParams::try_new(
                frequency,
                problem.physics.speed_of_sound,
                problem.physics.density,
                problem.physics.tau < 0.0,
            )
            .map_err(|error| BemError::InvalidParameters(error.to_string()))?;
            physics.harmonic_factor = problem.physics.harmonic_factor;
            physics.pressure_factor = physics.density * physics.omega * physics.harmonic_factor;
            physics
                .validate()
                .map_err(|error| BemError::InvalidParameters(error.to_string()))?;
            at_frequency.physics = physics;

            let gradient = self.solve_wavenumber_objective_gradient(
                &at_frequency,
                &forward_rhs_batches[index],
                &objective_gradient_batches[index],
            )?;
            if !gradient.is_finite() {
                return Err(BemError::InvalidParameters(format!(
                    "wavenumber-gradient batch entry {index} is non-finite"
                )));
            }
            gradients.push(gradient);
        }
        Ok(gradients)
    }

    fn build_rhs_position_derivative(
        &self,
        elements: &[Element],
        nodes: &[[f64; 3]],
        incident_field: &IncidentField,
        physics: &PhysicsParams,
        sign: f64,
        beta_scale: f64,
        source_position: bool,
    ) -> Result<Vec<[Complex64; 3]>, BemError> {
        let num_dofs = count_dofs(elements);
        let zero = Complex64::new(0.0, 0.0);
        let mut derivative = vec![[zero; 3]; num_dofs];
        let mut seen = vec![false; num_dofs];
        let beta = physics.burton_miller_beta_scaled(beta_scale);
        let gamma = Complex64::new(physics.gamma(), 0.0);
        let tau = Complex64::new(physics.tau, 0.0);

        for element in elements
            .iter()
            .filter(|element| !element.property.is_evaluation())
        {
            let points = element.collocation_points(self.basis_order, nodes);
            let normals = if self.basis_continuity == BasisContinuity::Continuous {
                vec![element.normal; element.dof_addresses.len()]
            } else {
                element.collocation_normals(self.basis_order, nodes)
            };
            if points.len() != element.dof_addresses.len() || normals.len() != points.len() {
                return Err(BemError::InvalidParameters(
                    "position-derivative collocation/DOF dimension mismatch".to_string(),
                ));
            }

            for (local, &dof) in element.dof_addresses.iter().enumerate() {
                if dof >= num_dofs {
                    return Err(BemError::InvalidParameters(
                        "position-derivative DOF address out of range".to_string(),
                    ));
                }
                if self.basis_continuity == BasisContinuity::Continuous && seen[dof] {
                    continue;
                }
                seen[dof] = true;

                let point = [points[local]];
                let normal = [normals[local]];
                let pressure = if source_position {
                    incident_field.evaluate_pressure_source_position_derivative(&point, physics)
                } else {
                    incident_field.evaluate_pressure_point_position_derivative(&point, physics)
                };
                let normal_derivative = if source_position {
                    incident_field
                        .evaluate_normal_source_position_derivative(&point, &normal, physics)
                } else {
                    incident_field
                        .evaluate_normal_point_position_derivative(&point, &normal, physics)
                };
                let pressure = pressure[0];
                let normal_derivative = normal_derivative[0];
                for axis in 0..3 {
                    derivative[dof][axis] = if beta_scale == 0.0 {
                        -sign * gamma * pressure[axis]
                    } else {
                        -sign * (gamma * pressure[axis] + beta * tau * normal_derivative[axis])
                    };
                }
            }
        }
        Ok(derivative)
    }

    /// Evaluate a higher-order compressed point-source position gradient by
    /// centered production solves at nearby source positions.
    fn solve_higher_order_compressed_source_position_objective_gradient(
        &self,
        problem: &BemProblem,
        objective_gradient: &[Complex64],
    ) -> Result<[f64; 3], BemError> {
        let elements = self.prepare_elements(problem)?;
        let num_dofs = count_dofs(&elements);
        if objective_gradient.len() != num_dofs {
            return Err(BemError::InvalidParameters(format!(
                "source-position objective gradient length {} must equal {num_dofs}",
                objective_gradient.len()
            )));
        }
        let (position, strength) = match problem.incident_field {
            IncidentField::PointSource { position, strength } => (position, strength),
            _ => {
                return Err(BemError::NotImplemented(
                "source-position objective gradients require single point-source incident field"
                    .to_string(),
            ));
            }
        };
        let objective_at = |candidate_position: [f64; 3]| -> Result<f64, BemError> {
            let mut candidate = problem.clone();
            candidate.incident_field = IncidentField::PointSource {
                position: candidate_position,
                strength,
            };
            let solution = self.solve(&candidate)?;
            if solution.surface_pressure.len() != num_dofs {
                return Err(BemError::InvalidParameters(
                    "source-position perturbed solve changed the DOF layout".to_string(),
                ));
            }
            let value = objective_gradient
                .iter()
                .zip(&solution.surface_pressure)
                .map(|(gradient, coefficient)| (gradient.conj() * coefficient).re)
                .sum::<f64>();
            if !value.is_finite() {
                return Err(BemError::InvalidParameters(
                    "source-position objective value is non-finite".to_string(),
                ));
            }
            Ok(value)
        };
        let step = 1.0e-5;
        let mut gradient = [0.0; 3];
        for axis in 0..3 {
            let mut plus = position;
            plus[axis] += step;
            let mut minus = position;
            minus[axis] -= step;
            gradient[axis] = (objective_at(plus)? - objective_at(minus)?) / (2.0 * step);
        }
        if gradient.iter().any(|value| !value.is_finite()) {
            return Err(BemError::InvalidParameters(
                "higher-order compressed source-position objective gradient is non-finite"
                    .to_string(),
            ));
        }
        Ok(gradient)
    }

    /// Solve a forward load and its Hermitian-adjoint objective to obtain the
    /// fixed-geometry gradient with respect to a point-source position.
    ///
    /// Dense TBEM supports all admitted linear/quadratic collocation bases;
    /// the assembled operator is source-independent, so the gradient is the
    /// real part of `lambdaᴴ * d(rhs)/d(source_position)` for the incident
    /// point-source RHS generated by `problem`. Higher-order compressed paths
    /// use checked centered production solves; analytic compressed RHS
    /// contractions remain available for constant-panel operators.
    pub fn solve_source_position_objective_gradient(
        &self,
        problem: &BemProblem,
        objective_gradient: &[Complex64],
    ) -> Result<[f64; 3], BemError> {
        self.validate()?;
        problem.validate()?;
        let elements = self.prepare_elements(problem)?;
        let num_dofs = count_dofs(&elements);
        let resolved_assembly = self.assembly_method.resolve(num_dofs);
        let dense_source_derivative =
            self.solver_method == SolverMethod::Direct && resolved_assembly == AssemblyMethod::Tbem;
        let compressed_source_derivative = self.solver_method == SolverMethod::Gmres
            && matches!(
                resolved_assembly,
                AssemblyMethod::Hmatrix
                    | AssemblyMethod::Slfmm
                    | AssemblyMethod::Mlfmm
                    | AssemblyMethod::DirectionalMlfmm
            )
            && self.basis_order == BasisOrder::Constant
            && self.basis_continuity == BasisContinuity::Discontinuous;
        let higher_order_compressed_source_derivative = self.solver_method == SolverMethod::Gmres
            && matches!(
                resolved_assembly,
                AssemblyMethod::Hmatrix
                    | AssemblyMethod::Slfmm
                    | AssemblyMethod::Mlfmm
                    | AssemblyMethod::DirectionalMlfmm
            )
            && matches!(self.basis_order, BasisOrder::Linear | BasisOrder::Quadratic);
        if higher_order_compressed_source_derivative {
            return self.solve_higher_order_compressed_source_position_objective_gradient(
                problem,
                objective_gradient,
            );
        }
        if !dense_source_derivative && !compressed_source_derivative {
            return Err(BemError::NotImplemented(
                "source-position objective gradients require dense TBEM or constant-panel compressed GMRES"
                    .to_string(),
            ));
        }
        if !matches!(problem.incident_field, IncidentField::PointSource { .. }) {
            return Err(BemError::NotImplemented(
                "source-position objective gradients require a single point-source incident field"
                    .to_string(),
            ));
        }

        let adjoint = self
            .solve_adjoint_many(problem, &[objective_gradient.to_vec()])?
            .into_iter()
            .next()
            .ok_or_else(|| {
                BemError::SolverFailed(
                    "adjoint source-position solve returned no solution".to_string(),
                )
            })?;
        let beta_scale = if problem.use_burton_miller {
            PhysicsParams::optimal_beta_scale(problem.ka())
        } else {
            0.0
        };
        let global_sign = PhysicsParams::optimal_formulation_sign(problem.ka());
        let rhs_derivative = self.build_rhs_position_derivative(
            &elements,
            &problem.mesh.nodes,
            &problem.incident_field,
            &problem.physics,
            global_sign,
            beta_scale,
            true,
        )?;
        let adjoint_coefficients = adjoint.coefficients();
        let mut gradient = [0.0; 3];
        for (row, derivative) in rhs_derivative.iter().enumerate() {
            for axis in 0..3 {
                gradient[axis] += (adjoint_coefficients[row].conj() * derivative[axis]).re;
            }
        }
        if gradient.iter().any(|value| !value.is_finite()) {
            return Err(BemError::InvalidParameters(
                "source-position objective gradient is non-finite".to_string(),
            ));
        }
        Ok(gradient)
    }

    /// Solve a forward-load Hermitian-adjoint objective and return its
    /// derivative with respect to a global rigid translation of the surface.
    ///
    /// A global translation leaves the constant-panel TBEM operator unchanged:
    /// pairwise distances, panel normals, and panel areas are invariant. The
    /// only derivative is therefore the incident RHS evaluated at translated
    /// collocation points. Dense TBEM supports admitted linear/quadratic
    /// collocation bases; compressed higher-order operators use a checked
    /// centered production-solve fallback. Local/deforming geometry still
    /// requires a separate operator derivative and is rejected by the
    /// fail-closed APIs.
    fn solve_higher_order_compressed_geometry_translation_objective_gradient(
        &self,
        problem: &BemProblem,
        objective_gradient: &[Complex64],
    ) -> Result<[f64; 3], BemError> {
        let elements = self.prepare_elements(problem)?;
        let num_dofs = count_dofs(&elements);
        if objective_gradient.len() != num_dofs {
            return Err(BemError::InvalidParameters(format!(
                "higher-order translation objective gradient has {} entries; expected {num_dofs}",
                objective_gradient.len()
            )));
        }

        let objective_at = |candidate: &BemProblem| -> Result<f64, BemError> {
            let solution = self.solve(candidate)?;
            if solution.coefficients().len() != num_dofs {
                return Err(BemError::InvalidParameters(
                    "higher-order translation solve changed DOF layout".to_string(),
                ));
            }
            let value = objective_gradient
                .iter()
                .zip(solution.coefficients())
                .map(|(gradient, coefficient)| (gradient.conj() * coefficient).re)
                .sum::<f64>();
            if !value.is_finite() {
                return Err(BemError::InvalidParameters(
                    "higher-order translation objective is non-finite".to_string(),
                ));
            }
            Ok(value)
        };

        let step = 1.0e-5;
        let mut derivative = [0.0; 3];
        for axis in 0..3 {
            let mut plus = problem.clone();
            let mut minus = problem.clone();
            for node in &mut plus.mesh.nodes {
                node[axis] += step;
            }
            for node in &mut minus.mesh.nodes {
                node[axis] -= step;
            }
            for element in &mut plus.mesh.elements {
                element.center[axis] += step;
            }
            for element in &mut minus.mesh.elements {
                element.center[axis] -= step;
            }
            derivative[axis] = (objective_at(&plus)? - objective_at(&minus)?) / (2.0 * step);
        }
        if derivative.iter().any(|value| !value.is_finite()) {
            return Err(BemError::InvalidParameters(
                "higher-order translation objective gradient is non-finite".to_string(),
            ));
        }
        Ok(derivative)
    }

    /// Solve the objective gradient with respect to a global rigid surface
    /// translation. Dense constant-panel TBEM uses the analytic incident-RHS
    /// derivative; higher-order compressed paths use the checked centered
    /// production-solve fallback described above.
    pub fn solve_geometry_translation_objective_gradient(
        &self,
        problem: &BemProblem,
        objective_gradient: &[Complex64],
    ) -> Result<[f64; 3], BemError> {
        self.validate()?;
        problem.validate()?;
        let elements = self.prepare_elements(problem)?;
        let num_dofs = count_dofs(&elements);
        let resolved_assembly = self.assembly_method.resolve(num_dofs);
        let dense_translation =
            self.solver_method == SolverMethod::Direct && resolved_assembly == AssemblyMethod::Tbem;
        let compressed_method = self.solver_method == SolverMethod::Gmres
            && matches!(
                resolved_assembly,
                AssemblyMethod::Hmatrix
                    | AssemblyMethod::Slfmm
                    | AssemblyMethod::Mlfmm
                    | AssemblyMethod::DirectionalMlfmm
            );
        let compressed_translation = compressed_method
            && ((self.basis_order == BasisOrder::Constant
                && self.basis_continuity == BasisContinuity::Discontinuous)
                || matches!(self.basis_order, BasisOrder::Linear | BasisOrder::Quadratic));
        if !dense_translation && !compressed_translation {
            return Err(BemError::NotImplemented(
                "geometry-translation objective gradients require dense TBEM or constant-panel compressed GMRES"
                    .to_string(),
            ));
        }
        if compressed_method
            && matches!(self.basis_order, BasisOrder::Linear | BasisOrder::Quadratic)
        {
            return self.solve_higher_order_compressed_geometry_translation_objective_gradient(
                problem,
                objective_gradient,
            );
        }

        let adjoint = self
            .solve_adjoint_many(problem, &[objective_gradient.to_vec()])?
            .into_iter()
            .next()
            .ok_or_else(|| {
                BemError::SolverFailed(
                    "geometry-translation adjoint solve returned no solution".to_string(),
                )
            })?;
        let beta_scale = if problem.use_burton_miller {
            PhysicsParams::optimal_beta_scale(problem.ka())
        } else {
            0.0
        };
        let global_sign = PhysicsParams::optimal_formulation_sign(problem.ka());
        let rhs_derivative = self.build_rhs_position_derivative(
            &elements,
            &problem.mesh.nodes,
            &problem.incident_field,
            &problem.physics,
            global_sign,
            beta_scale,
            false,
        )?;
        let adjoint_coefficients = adjoint.coefficients();
        let mut gradient = [0.0; 3];
        for (row, derivative) in rhs_derivative.iter().enumerate() {
            for axis in 0..3 {
                gradient[axis] += (adjoint_coefficients[row].conj() * derivative[axis]).re;
            }
        }
        if gradient.iter().any(|value| !value.is_finite()) {
            return Err(BemError::InvalidParameters(
                "geometry-translation objective gradient is non-finite".to_string(),
            ));
        }
        Ok(gradient)
    }

    /// Solve a forward load and its Hermitian adjoint objective, then return the
    /// fixed-geometry sound-speed derivative of `Re(gᴴ x)` at fixed frequency.
    ///
    /// This is the material-independent acoustic-medium speed sensitivity.
    /// The sound-speed derivative is the wavenumber derivative composed with
    /// `dk/dc = -k/c` at fixed frequency. This keeps the same analytic
    /// operator/RHS dispatch for dense and compressed higher-order paths.
    pub fn solve_speed_of_sound_objective_gradient(
        &self,
        problem: &BemProblem,
        forward_rhs: &[Complex64],
        objective_gradient: &[Complex64],
    ) -> Result<f64, BemError> {
        self.validate()?;
        problem.validate()?;
        let elements = self.prepare_elements(problem)?;
        let num_dofs = count_dofs(&elements);
        let resolved_assembly = self.assembly_method.resolve(num_dofs);
        let compressed_speed_chain = self.solver_method == SolverMethod::Gmres
            && matches!(
                resolved_assembly,
                AssemblyMethod::Hmatrix
                    | AssemblyMethod::Slfmm
                    | AssemblyMethod::Mlfmm
                    | AssemblyMethod::DirectionalMlfmm
            )
            && problem.bc_type != BoundaryConditionType::Impedance;
        if compressed_speed_chain {
            let speed = problem.physics.speed_of_sound;
            let wave_number = problem.physics.wave_number;
            if !speed.is_finite() || speed <= 0.0 || !wave_number.is_finite() {
                return Err(BemError::InvalidParameters(
                    "sound-speed objective gradient requires finite positive medium parameters"
                        .to_string(),
                ));
            }
            let dk_dc = -wave_number / speed;
            let wavenumber_gradient =
                self.solve_wavenumber_objective_gradient(problem, forward_rhs, objective_gradient)?;
            let gradient = dk_dc * wavenumber_gradient;
            if !gradient.is_finite() {
                return Err(BemError::InvalidParameters(
                    "sound-speed objective gradient is non-finite".to_string(),
                ));
            }
            return Ok(gradient);
        }
        let compressed_speed_derivative = self.solver_method == SolverMethod::Gmres
            && matches!(
                resolved_assembly,
                AssemblyMethod::Hmatrix
                    | AssemblyMethod::Slfmm
                    | AssemblyMethod::Mlfmm
                    | AssemblyMethod::DirectionalMlfmm
            )
            && self.basis_order == BasisOrder::Constant
            && self.basis_continuity == BasisContinuity::Discontinuous;
        if self.solver_method == SolverMethod::Gmres
            && matches!(
                resolved_assembly,
                AssemblyMethod::Hmatrix
                    | AssemblyMethod::Slfmm
                    | AssemblyMethod::Mlfmm
                    | AssemblyMethod::DirectionalMlfmm
            )
            && matches!(self.basis_order, BasisOrder::Linear | BasisOrder::Quadratic)
        {
            return self.solve_compressed_objective_gradient_finite_difference(
                problem,
                forward_rhs,
                objective_gradient,
                DenseObjectiveGradientParameter::SpeedOfSound,
            );
        }
        if compressed_speed_derivative {
            return self.solve_compressed_parameter_objective_gradient(
                problem,
                forward_rhs,
                objective_gradient,
                HmatrixDerivativeParameter::SpeedOfSound,
            );
        }
        if self.solver_method == SolverMethod::Direct
            && resolved_assembly == AssemblyMethod::Tbem
            && (self.basis_order != BasisOrder::Constant
                || self.basis_continuity != BasisContinuity::Discontinuous)
        {
            return self.solve_dense_objective_gradient_finite_difference(
                problem,
                forward_rhs,
                objective_gradient,
                DenseObjectiveGradientParameter::SpeedOfSound,
            );
        }
        if self.solver_method != SolverMethod::Direct
            || resolved_assembly != AssemblyMethod::Tbem
            || self.basis_order != BasisOrder::Constant
            || self.basis_continuity != BasisContinuity::Discontinuous
        {
            return Err(BemError::NotImplemented(
                "sound-speed objective gradients currently require dense direct constant-panel TBEM"
                    .to_string(),
            ));
        }
        let forward = self
            .solve_rhs_many(problem, &[forward_rhs.to_vec()])?
            .into_iter()
            .next()
            .ok_or_else(|| {
                BemError::SolverFailed("forward gradient solve returned no solution".to_string())
            })?;
        let adjoint = self
            .solve_adjoint_many(problem, &[objective_gradient.to_vec()])?
            .into_iter()
            .next()
            .ok_or_else(|| {
                BemError::SolverFailed("adjoint gradient solve returned no solution".to_string())
            })?;
        let ka = problem.ka();
        let sign = PhysicsParams::optimal_formulation_sign(ka);
        let beta_scale = if problem.use_burton_miller {
            PhysicsParams::optimal_beta_scale(ka)
        } else {
            0.0
        };
        let derivative = build_tbem_system_stabilized_speed_derivative(
            &elements,
            &problem.mesh.nodes,
            &problem.physics,
            sign,
            beta_scale,
        );
        tbem_wavenumber_objective_gradient(
            &derivative,
            forward.coefficients(),
            adjoint.coefficients(),
        )
        .map_err(|error| BemError::InvalidParameters(error.to_string()))
    }

    /// Solve a forward-load Hermitian-adjoint objective and return the
    /// fixed-geometry medium-density derivative `Re(gᴴ x)` at fixed frequency
    /// and sound speed.
    ///
    /// Constant-panel TBEM uses the analytic density derivative assembler;
    /// dense direct linear/quadratic TBEM uses the production-assembly
    /// centered fallback. Compressed impedance boundaries synchronize
    /// `Y = 1/(rho*c)` in the operator bracket; moving-geometry,
    /// interface, device, and distributed derivative paths remain separate
    /// contracts.
    pub fn solve_density_objective_gradient(
        &self,
        problem: &BemProblem,
        forward_rhs: &[Complex64],
        objective_gradient: &[Complex64],
    ) -> Result<f64, BemError> {
        self.validate()?;
        problem.validate()?;
        let elements = self.prepare_elements(problem)?;
        let num_dofs = count_dofs(&elements);
        let resolved_assembly = self.assembly_method.resolve(num_dofs);
        let compressed_density_derivative = self.solver_method == SolverMethod::Gmres
            && matches!(
                resolved_assembly,
                AssemblyMethod::Hmatrix
                    | AssemblyMethod::Slfmm
                    | AssemblyMethod::Mlfmm
                    | AssemblyMethod::DirectionalMlfmm
            )
            && self.basis_order == BasisOrder::Constant
            && self.basis_continuity == BasisContinuity::Discontinuous;
        if self.solver_method == SolverMethod::Gmres
            && matches!(
                resolved_assembly,
                AssemblyMethod::Hmatrix
                    | AssemblyMethod::Slfmm
                    | AssemblyMethod::Mlfmm
                    | AssemblyMethod::DirectionalMlfmm
            )
            && matches!(self.basis_order, BasisOrder::Linear | BasisOrder::Quadratic)
        {
            return self.solve_compressed_objective_gradient_finite_difference(
                problem,
                forward_rhs,
                objective_gradient,
                DenseObjectiveGradientParameter::Density,
            );
        }
        if compressed_density_derivative {
            return self.solve_compressed_parameter_objective_gradient(
                problem,
                forward_rhs,
                objective_gradient,
                HmatrixDerivativeParameter::Density,
            );
        }
        if self.solver_method == SolverMethod::Direct
            && resolved_assembly == AssemblyMethod::Tbem
            && (self.basis_order != BasisOrder::Constant
                || self.basis_continuity != BasisContinuity::Discontinuous)
        {
            return self.solve_dense_objective_gradient_finite_difference(
                problem,
                forward_rhs,
                objective_gradient,
                DenseObjectiveGradientParameter::Density,
            );
        }
        if self.solver_method != SolverMethod::Direct
            || resolved_assembly != AssemblyMethod::Tbem
            || self.basis_order != BasisOrder::Constant
            || self.basis_continuity != BasisContinuity::Discontinuous
        {
            return Err(BemError::NotImplemented(
                "density objective gradients currently require dense direct constant-panel TBEM"
                    .to_string(),
            ));
        }

        let forward = self
            .solve_rhs_many(problem, &[forward_rhs.to_vec()])?
            .into_iter()
            .next()
            .ok_or_else(|| {
                BemError::SolverFailed("forward gradient solve returned no solution".to_string())
            })?;
        let adjoint = self
            .solve_adjoint_many(problem, &[objective_gradient.to_vec()])?
            .into_iter()
            .next()
            .ok_or_else(|| {
                BemError::SolverFailed("adjoint gradient solve returned no solution".to_string())
            })?;

        let ka = problem.ka();
        let sign = PhysicsParams::optimal_formulation_sign(ka);
        let beta_scale = if problem.use_burton_miller {
            PhysicsParams::optimal_beta_scale(ka)
        } else {
            0.0
        };
        let derivative = if problem.bc_type == BoundaryConditionType::Impedance {
            build_tbem_system_stabilized_impedance_density_derivative(
                &elements,
                &problem.mesh.nodes,
                &problem.physics,
                sign,
                beta_scale,
            )
        } else {
            build_tbem_system_stabilized_density_derivative(
                &elements,
                &problem.mesh.nodes,
                &problem.physics,
                sign,
                beta_scale,
            )
        };
        tbem_wavenumber_objective_gradient(
            &derivative,
            forward.coefficients(),
            adjoint.coefficients(),
        )
        .map_err(|error| BemError::InvalidParameters(error.to_string()))
    }

    /// Solve arbitrary forward driver/unit-load batches across frequencies.
    ///
    /// The outer result follows `frequencies`, while each inner vector keeps
    /// the caller's RHS-column order.  Physical parameters are rebuilt and
    /// validated for every frequency before the checked load solve runs.
    pub fn solve_rhs_frequency_sweep_many(
        &self,
        problem: &BemProblem,
        frequencies: &[f64],
        rhs_columns: &[Vec<Vec<Complex64>>],
    ) -> Result<Vec<Vec<BemLoadSolution>>, BemError> {
        self.validate()?;
        problem.validate()?;
        if frequencies.is_empty()
            || frequencies
                .iter()
                .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
        {
            return Err(BemError::InvalidParameters(
                "forward load frequency batch requires finite, positive frequencies".to_string(),
            ));
        }
        if frequencies.len() != rhs_columns.len() {
            return Err(BemError::InvalidParameters(format!(
                "forward load frequency batch has {} frequency entries but {} RHS batches",
                frequencies.len(),
                rhs_columns.len()
            )));
        }

        let is_internal = problem.physics.tau < 0.0;
        let mut batches = Vec::with_capacity(frequencies.len());
        for (&frequency, frequency_rhs) in frequencies.iter().zip(rhs_columns) {
            let mut at_frequency = problem.clone();
            let mut physics = PhysicsParams::try_new(
                frequency,
                problem.physics.speed_of_sound,
                problem.physics.density,
                is_internal,
            )
            .map_err(|error| BemError::InvalidParameters(error.to_string()))?;
            physics.harmonic_factor = problem.physics.harmonic_factor;
            physics.pressure_factor = physics.density * physics.omega * physics.harmonic_factor;
            physics
                .validate()
                .map_err(|error| BemError::InvalidParameters(error.to_string()))?;
            at_frequency.physics = physics;
            batches.push(self.solve_rhs_many(&at_frequency, frequency_rhs)?);
        }
        Ok(batches)
    }

    /// Solve several Hermitian-adjoint systems for one assembled BEM problem.
    ///
    /// The RHS columns are interpreted as independent loads for `Aᴴ Y = B`.
    /// Dense direct solves reuse one LU factorization; dense GMRES and the
    /// H/SLFMM/MLFMM paths use their exact Hermitian operator and preconditioner
    /// views with the same block residual certification as forward batches.
    pub fn solve_adjoint_many(
        &self,
        problem: &BemProblem,
        rhs_columns: &[Vec<Complex64>],
    ) -> Result<Vec<BemAdjointSolution>, BemError> {
        self.validate()?;
        problem.validate()?;
        if rhs_columns.is_empty() {
            return Ok(Vec::new());
        }

        let elements = self.prepare_elements(problem)?;
        self.admit_resources(&elements, rhs_columns.len(), 1)?;
        let num_dofs = count_dofs(&elements);
        for (rhs_index, rhs) in rhs_columns.iter().enumerate() {
            if rhs.len() != num_dofs {
                return Err(BemError::InvalidParameters(format!(
                    "adjoint RHS {rhs_index} has {} entries; expected {num_dofs}",
                    rhs.len()
                )));
            }
            if rhs
                .iter()
                .any(|value| !value.re.is_finite() || !value.im.is_finite())
            {
                return Err(BemError::InvalidParameters(format!(
                    "adjoint RHS {rhs_index} contains non-finite values"
                )));
            }
        }

        if problem.bc_type == BoundaryConditionType::Soft {
            return self.solve_soft_dirichlet_adjoint_many(problem, &elements, rhs_columns);
        }

        let ka_problem = problem.ka();
        let global_sign = PhysicsParams::optimal_formulation_sign(ka_problem);
        let global_beta_scale = if problem.use_burton_miller {
            PhysicsParams::optimal_beta_scale(ka_problem)
        } else {
            0.0
        };
        let assembly = self.assemble_system_stabilized(
            &elements,
            &problem.mesh.nodes,
            &problem.physics,
            global_sign,
            global_beta_scale,
        )?;

        let (solution_block, residuals) = match assembly {
            AssemblyResult::Dense(matrix, _) => {
                let mut hermitian = DenseMatrix::<Complex64>::zeros(num_dofs, num_dofs);
                for row in 0..num_dofs {
                    for column in 0..num_dofs {
                        hermitian[[row, column]] = matrix[[column, row]].conj();
                    }
                }
                let rhs_block = DenseMatrix::from_fn(num_dofs, rhs_columns.len(), |row, column| {
                    rhs_columns[column][row]
                });
                let operator = DenseMatrixOperator(&hermitian);
                if self.solver_method == SolverMethod::Direct {
                    let factorization = sonium_solvers::direct::lu_factorize(&hermitian)
                        .map_err(|error| BemError::SolverFailed(error.to_string()))?;
                    let solution = factorization
                        .solve_many(rhs_block.as_slice(), rhs_columns.len())
                        .map_err(|error| BemError::SolverFailed(error.to_string()))?;
                    let solution =
                        DenseMatrix::from_row_major_vec(num_dofs, rhs_columns.len(), solution);
                    let residuals = exact_block_residuals(&operator, &solution, rhs_columns)?;
                    (solution, residuals)
                } else if self.solver_method == SolverMethod::Gmres {
                    let precond = self.build_preconditioner(
                        &elements,
                        &problem.mesh.nodes,
                        &problem.physics,
                        None,
                        None,
                    )?;
                    if precond.hermitian_capability()
                        != sonium_solvers::traits::AdjointCapability::Exact
                    {
                        return Err(BemError::NotImplemented(
                            "configured BEM preconditioner has no exact Hermitian action"
                                .to_string(),
                        ));
                    }
                    let adjoint_precond = HermitianPreconditioner { inner: precond };
                    let solution = self.solve_compressed_many(
                        &operator,
                        &adjoint_precond,
                        rhs_columns,
                        "dense-adjoint-block-gmres",
                    )?;
                    let residuals = exact_block_residuals(&operator, &solution, rhs_columns)?;
                    (solution, residuals)
                } else {
                    return Err(BemError::NotImplemented(format!(
                        "adjoint multi-RHS requires Direct or Gmres; requested {:?}",
                        self.solver_method
                    )));
                }
            }
            AssemblyResult::Slfmm(system) => {
                let nearfield_matrix = system.try_extract_near_field_matrix().map_err(|error| {
                    BemError::SolverFailed(format!("SLFMM adjoint near-field: {error}"))
                })?;
                let nearfield =
                    sonium_solvers::sparse::CsrMatrix::from_dense(&nearfield_matrix, 1.0e-15);
                let preconditioner = if self.preconditioner == BemPreconditioner::HierarchicalIlu {
                    let hil = HiluPreconditioner::try_from_csr(&nearfield).map_err(|error| {
                        BemError::SolverFailed(format!("SLFMM adjoint hierarchical ILU: {error}"))
                    })?;
                    BemPrecondImpl::Hilu(Box::new(hil))
                } else {
                    let ilu = sonium_solvers::preconditioners::IluPreconditioner::try_from_csr(
                        &nearfield,
                    )
                    .map_err(|error| {
                        BemError::SolverFailed(format!("SLFMM adjoint near-field ILU: {error}"))
                    })?;
                    BemPrecondImpl::Ilu(ilu)
                };
                let operator = HermitianOperator {
                    inner: crate::core::solver::SlfmmOperator::new(system),
                };
                let precond = HermitianPreconditioner {
                    inner: preconditioner,
                };
                let solution = self.solve_compressed_many(
                    &operator,
                    &precond,
                    rhs_columns,
                    "slfmm-adjoint-block-gmres",
                )?;
                let residuals = exact_block_residuals(&operator, &solution, rhs_columns)?;
                (solution, residuals)
            }
            AssemblyResult::Mlfmm(system) => {
                let precond = if self.preconditioner == BemPreconditioner::HierarchicalIlu {
                    let nearfield_matrix =
                        system.try_extract_near_field_matrix().map_err(|error| {
                            BemError::SolverFailed(format!("MLFMM adjoint near-field: {error}"))
                        })?;
                    let nearfield =
                        sonium_solvers::sparse::CsrMatrix::from_dense(&nearfield_matrix, 1.0e-15);
                    let hil = HiluPreconditioner::try_from_csr(&nearfield).map_err(|error| {
                        BemError::SolverFailed(format!("MLFMM adjoint hierarchical ILU: {error}"))
                    })?;
                    BemPrecondImpl::Hilu(Box::new(hil))
                } else {
                    self.build_preconditioner(
                        &elements,
                        &problem.mesh.nodes,
                        &problem.physics,
                        None,
                        None,
                    )?
                };
                if precond.hermitian_capability()
                    != sonium_solvers::traits::AdjointCapability::Exact
                {
                    return Err(BemError::NotImplemented(
                        "configured BEM preconditioner has no exact Hermitian action".to_string(),
                    ));
                }
                let operator = HermitianOperator {
                    inner: crate::core::solver::MlfmmOperator::new(*system),
                };
                let adjoint_precond = HermitianPreconditioner { inner: precond };
                let solution = self.solve_compressed_many(
                    &operator,
                    &adjoint_precond,
                    rhs_columns,
                    "mlfmm-adjoint-block-gmres",
                )?;
                let residuals = exact_block_residuals(&operator, &solution, rhs_columns)?;
                (solution, residuals)
            }
            AssemblyResult::Hmatrix(hmatrix, _) => {
                let h_diag = hmatrix_diagonal(&hmatrix)?;
                let precond = self.build_preconditioner(
                    &elements,
                    &problem.mesh.nodes,
                    &problem.physics,
                    None,
                    Some(&h_diag),
                )?;
                if precond.hermitian_capability()
                    != sonium_solvers::traits::AdjointCapability::Exact
                {
                    return Err(BemError::NotImplemented(
                        "configured BEM preconditioner has no exact Hermitian action".to_string(),
                    ));
                }
                let operator = HermitianOperator { inner: hmatrix };
                let adjoint_precond = HermitianPreconditioner { inner: precond };
                let solution = self.solve_compressed_many(
                    &operator,
                    &adjoint_precond,
                    rhs_columns,
                    "hmatrix-adjoint-block-gmres",
                )?;
                let residuals = exact_block_residuals(&operator, &solution, rhs_columns)?;
                (solution, residuals)
            }
        };

        let exact_reference_residuals = self.certify_compressed_batch(
            problem,
            &elements,
            global_sign,
            global_beta_scale,
            self.assembly_method.resolve(num_dofs),
            &solution_block,
            rhs_columns,
            true,
        )?;
        let mut solutions = Vec::with_capacity(rhs_columns.len());
        for (rhs_index, residual) in residuals.iter().copied().enumerate() {
            let coefficients = solution_block.column_to_vec(rhs_index);
            solutions.push(BemAdjointSolution {
                coefficients,
                relative_residual: residual,
                exact_reference_residual: exact_reference_residuals
                    .as_ref()
                    .and_then(|values| values.get(rhs_index).copied()),
                elements: elements.clone(),
                nodes: problem.mesh.nodes.clone(),
                physics: problem.physics.clone(),
                basis_order: self.basis_order,
            });
        }
        Ok(solutions)
    }

    /// Solve arbitrary adjoint driver/unit-load RHS columns on one assembled
    /// BEM problem.
    ///
    /// This named counterpart to [`Self::solve_adjoint_many`] makes the
    /// driver/objective use case explicit while retaining the same exact
    /// Hermitian residual certificates and backend capability checks.
    pub fn solve_adjoint_driver_loads(
        &self,
        problem: &BemProblem,
        rhs_columns: &[Vec<Complex64>],
    ) -> Result<Vec<BemAdjointSolution>, BemError> {
        self.solve_adjoint_many(problem, rhs_columns)
    }

    /// Re-evaluate compressed solutions with the independent exact constant-
    /// panel TBEM action when the caller has enabled a bounded certification
    /// budget. The exact action is matrix-free, so this does not retain a
    /// second dense operator in memory, while still catching an inaccurate
    /// compressed solve before it reaches a caller.
    fn certify_compressed_batch(
        &self,
        problem: &BemProblem,
        elements: &[Element],
        sign: f64,
        beta_scale: f64,
        assembly_method: AssemblyMethod,
        solution: &DenseMatrix<Complex64>,
        rhs_columns: &[Vec<Complex64>],
        hermitian: bool,
    ) -> Result<Option<Vec<f64>>, BemError> {
        let num_dofs = count_dofs(elements);
        let higher_order = self.basis_order != BasisOrder::Constant;
        let supported_method = matches!(
            assembly_method,
            AssemblyMethod::Hmatrix
                | AssemblyMethod::Slfmm
                | AssemblyMethod::Mlfmm
                | AssemblyMethod::DirectionalMlfmm
        );
        let supported_layout = match self.basis_continuity {
            BasisContinuity::Discontinuous => true,
            BasisContinuity::Continuous => {
                higher_order
                    && matches!(
                        assembly_method,
                        AssemblyMethod::Hmatrix
                            | AssemblyMethod::Slfmm
                            | AssemblyMethod::Mlfmm
                            | AssemblyMethod::DirectionalMlfmm
                    )
            }
        };
        if self.compressed_exact_residual_dof_limit == 0
            || num_dofs > self.compressed_exact_residual_dof_limit
            || !supported_method
            || !matches!(
                self.basis_order,
                BasisOrder::Constant | BasisOrder::Linear | BasisOrder::Quadratic
            )
            || !supported_layout
            || !matches!(
                problem.bc_type,
                BoundaryConditionType::Rigid | BoundaryConditionType::Impedance
            )
        {
            return Ok(None);
        }

        let exact = ExactTbEmOperator {
            elements,
            nodes: &problem.mesh.nodes,
            physics: problem.physics.clone(),
            sign,
            beta_scale,
            num_dofs,
            higher_order,
            // Continuous higher-order MLFMM uses flat panel normals; its proxies
            // preserve that same contract.  SLFMM and H-matrix continuous paths
            // still retain geometric normals, as do all discontinuous FMM paths.
            curved_normals: higher_order
                && ((self.basis_continuity == BasisContinuity::Continuous
                    && matches!(
                        assembly_method,
                        AssemblyMethod::Hmatrix | AssemblyMethod::Slfmm
                    ))
                    || (self.basis_continuity == BasisContinuity::Discontinuous
                        && matches!(
                            assembly_method,
                            AssemblyMethod::Slfmm
                                | AssemblyMethod::Mlfmm
                                | AssemblyMethod::DirectionalMlfmm
                        ))),
            owner_rows: higher_order
                && self.basis_continuity == BasisContinuity::Continuous
                && matches!(
                    assembly_method,
                    AssemblyMethod::Mlfmm | AssemblyMethod::DirectionalMlfmm
                ),
            hermitian,
        };
        let residuals = exact_block_residuals(&exact, solution, rhs_columns)?;
        let tolerance = self.compressed_exact_residual_tolerance;
        if let Some((_index, &residual)) = residuals
            .iter()
            .enumerate()
            .find(|(_, residual)| !residual.is_finite() || **residual > tolerance)
        {
            return Err(BemError::ResidualCheckFailed {
                label: if hermitian {
                    format!("{assembly_method:?}-adjoint-independent-exact")
                } else {
                    format!("{assembly_method:?}-independent-exact")
                },
                residual,
                tolerance,
            });
        }
        Ok(Some(residuals))
    }

    fn solve_compressed_many<O: LinearOperator<Complex64>, P: Preconditioner<Complex64>>(
        &self,
        operator: &O,
        precond: &P,
        rhs_columns: &[Vec<Complex64>],
        label: &str,
    ) -> Result<DenseMatrix<Complex64>, BemError> {
        if self.solver_method != SolverMethod::Gmres {
            return Err(BemError::NotImplemented(format!(
                "block compressed solve requires SolverMethod::Gmres; requested {:?} for {label}",
                self.solver_method
            )));
        }
        let rhs_block = DenseMatrix::from_fn(operator.num_rows(), rhs_columns.len(), |row, rhs| {
            rhs_columns[rhs][row]
        });
        let solution = block_gmres_preconditioned(
            operator,
            &rhs_block,
            precond,
            &BlockGmresConfig {
                max_iterations: self.max_iterations,
                restart: self.gmres_restart,
                tolerance: self.tolerance,
                true_residual_check: true,
                exact_residual_check_interval: 0,
            },
        );
        if !solution.converged {
            return Err(BemError::SolverFailed(format!(
                "block GMRES did not converge for {label}: true residuals = {:?}",
                solution.true_residuals
            )));
        }
        for (rhs_index, rhs) in rhs_columns.iter().enumerate() {
            self.check_exact_residual(operator, &solution.x.column_to_vec(rhs_index), rhs, label)?;
        }
        Ok(solution.x)
    }

    fn solve_assembled_system(
        &self,
        assembly_result: AssemblyResult,
        rhs: &[Complex64],
        elements: &[Element],
        problem: &BemProblem,
    ) -> Result<Vec<Complex64>, BemError> {
        match assembly_result {
            AssemblyResult::Dense(matrix, _) => {
                let dense_diag: Vec<Complex64> =
                    (0..matrix.nrows()).map(|i| matrix[[i, i]]).collect();
                let precond = self.build_preconditioner(
                    elements,
                    &problem.mesh.nodes,
                    &problem.physics,
                    Some(&dense_diag),
                    None,
                )?;
                self.solve_dense_system(&matrix, rhs, &precond)
            }
            AssemblyResult::Slfmm(system) => self.solve_fmm_system(system, rhs),
            AssemblyResult::Mlfmm(system) => {
                let precond = if self.preconditioner == BemPreconditioner::HierarchicalIlu {
                    BemPrecondImpl::HierarchicalIlu
                } else {
                    self.build_preconditioner(
                        elements,
                        &problem.mesh.nodes,
                        &problem.physics,
                        None,
                        None,
                    )?
                };
                self.solve_mlfmm_system(*system, rhs, &precond)
            }
            AssemblyResult::Hmatrix(hmatrix, _) => {
                let h_diag = hmatrix_diagonal(&hmatrix)?;
                let precond = self.build_preconditioner(
                    elements,
                    &problem.mesh.nodes,
                    &problem.physics,
                    None,
                    Some(&h_diag),
                )?;
                self.solve_hmatrix_system(hmatrix, rhs, &precond)
            }
        }
    }

    fn assemble_soft_dirichlet_matrix(
        &self,
        problem: &BemProblem,
        elements: &[Element],
    ) -> Result<DenseMatrix<Complex64>, BemError> {
        if self.basis_order != BasisOrder::Constant {
            let global_dof_count = count_dofs(elements);
            return assemble_galerkin_single_layer_with_global_dofs(
                elements,
                &problem.mesh.nodes,
                &problem.physics,
                self.basis_order,
                self.soft_dirichlet_quadrature_order(),
                global_dof_count,
            )
            .map_err(|error| {
                BemError::InvalidParameters(format!(
                    "higher-order soft-Dirichlet single-layer assembly failed: {error}"
                ))
            });
        }
        use crate::core::integration::{regular_integration, singular_integration};

        let active_elements: Vec<&Element> = elements
            .iter()
            .filter(|elem| !elem.property.is_evaluation())
            .collect();
        let n = active_elements.len();
        let mut matrix = DenseMatrix::<Complex64>::zeros(n, n);
        let coords: Vec<DenseMatrix<f64>> = active_elements
            .iter()
            .map(|elem| solver_element_coords(elem, &problem.mesh.nodes))
            .collect();

        for (i, source) in active_elements.iter().enumerate() {
            for (j, field) in active_elements.iter().enumerate() {
                let integral = if i == j {
                    singular_integration(
                        &source.center,
                        &source.normal,
                        &coords[j],
                        field.element_type,
                        &problem.physics,
                        None,
                        0,
                        false,
                    )
                } else {
                    regular_integration(
                        &source.center,
                        &source.normal,
                        &coords[j],
                        field.element_type,
                        field.area,
                        &problem.physics,
                        None,
                        0,
                        false,
                    )
                };
                matrix[[i, j]] = integral.g_integral;
            }
        }
        Ok(matrix)
    }

    fn soft_dirichlet_quadrature_order(&self) -> usize {
        self.quadrature_policy
            .map(|policy| policy.regular_fine_order)
            .unwrap_or(6)
            .clamp(4, 16)
    }

    fn soft_dirichlet_rhs_higher_order(
        &self,
        problem: &BemProblem,
        elements: &[Element],
    ) -> Result<Vec<Complex64>, BemError> {
        self.soft_dirichlet_rhs_higher_order_for_incident(
            &problem.physics,
            &problem.incident_field,
            &problem.mesh.nodes,
            elements,
        )
    }

    fn soft_dirichlet_rhs_higher_order_for_incident(
        &self,
        physics: &PhysicsParams,
        incident_field: &IncidentField,
        nodes: &[[f64; 3]],
        elements: &[Element],
    ) -> Result<Vec<Complex64>, BemError> {
        let active = elements
            .iter()
            .filter(|element| !element.property.is_evaluation())
            .collect::<Vec<_>>();
        if active.is_empty() {
            return Err(BemError::InvalidParameters(
                "higher-order soft-Dirichlet projection requires active panels".to_string(),
            ));
        }
        let global_dof_count = count_dofs(elements);
        if global_dof_count == 0 {
            return Err(BemError::InvalidParameters(
                "higher-order soft-Dirichlet projection requires DOFs".to_string(),
            ));
        }
        let local_basis_count = match self.basis_order {
            BasisOrder::Linear => match active[0].element_type {
                ElementType::Tri3 => 3,
                ElementType::Quad4 => 4,
            },
            BasisOrder::Quadratic => match active[0].element_type {
                ElementType::Tri3 => 6,
                ElementType::Quad4 => 9,
            },
            BasisOrder::Constant => {
                return Err(BemError::InvalidParameters(
                    "higher-order soft-Dirichlet projection requires non-constant basis"
                        .to_string(),
                ));
            }
        };
        if active
            .iter()
            .any(|element| element.element_type != active[0].element_type)
        {
            return Err(BemError::InvalidParameters(
                "higher-order soft-Dirichlet projection rejects mixed Tri3/Quad4 surfaces"
                    .to_string(),
            ));
        }
        let order = self.soft_dirichlet_quadrature_order();
        let mut mass = DenseMatrix::<Complex64>::zeros(global_dof_count, global_dof_count);
        let mut load = vec![Complex64::new(0.0, 0.0); global_dof_count];

        for element in active {
            if element.dof_addresses.len() != local_basis_count {
                return Err(BemError::InvalidParameters(format!(
                    "higher-order soft-Dirichlet element has {} DOFs; expected {local_basis_count}",
                    element.dof_addresses.len()
                )));
            }
            let coords = solver_element_coords(element, nodes);
            let mut add_sample = |point: [f64; 3], weight: f64, values: &[f64]| {
                if !weight.is_finite()
                    || point.iter().any(|value| !value.is_finite())
                    || values.len() != local_basis_count
                    || values.iter().any(|value| !value.is_finite())
                {
                    return Err(BemError::InvalidParameters(
                        "higher-order soft-Dirichlet projection encountered non-finite quadrature data"
                            .to_string(),
                    ));
                }
                let pressure = incident_field.evaluate_pressure(&[point], physics)[0];
                if !pressure.re.is_finite() || !pressure.im.is_finite() {
                    return Err(BemError::InvalidParameters(
                        "higher-order soft-Dirichlet incident pressure is non-finite".to_string(),
                    ));
                }
                for (local_row, &row) in element.dof_addresses.iter().enumerate() {
                    if row >= global_dof_count {
                        return Err(BemError::InvalidParameters(
                            "higher-order soft-Dirichlet DOF is out of range".to_string(),
                        ));
                    }
                    load[row] -= pressure * (weight * values[local_row]);
                    for (local_column, &column) in element.dof_addresses.iter().enumerate() {
                        if column >= global_dof_count {
                            return Err(BemError::InvalidParameters(
                                "higher-order soft-Dirichlet DOF is out of range".to_string(),
                            ));
                        }
                        mass[[row, column]] +=
                            Complex64::new(weight * values[local_row] * values[local_column], 0.0);
                    }
                }
                Ok(())
            };

            match element.element_type {
                ElementType::Tri3 => {
                    let jacobian = 2.0 * element.area;
                    for (s, t, reference_weight) in triangle_quadrature(order) {
                        let vertices = [
                            [coords[[0, 0]], coords[[0, 1]], coords[[0, 2]]],
                            [coords[[1, 0]], coords[[1, 1]], coords[[1, 2]]],
                            [coords[[2, 0]], coords[[2, 1]], coords[[2, 2]]],
                        ];
                        let point = [
                            vertices[0][0]
                                + s * (vertices[1][0] - vertices[0][0])
                                + t * (vertices[2][0] - vertices[0][0]),
                            vertices[0][1]
                                + s * (vertices[1][1] - vertices[0][1])
                                + t * (vertices[2][1] - vertices[0][1]),
                            vertices[0][2]
                                + s * (vertices[1][2] - vertices[0][2])
                                + t * (vertices[2][2] - vertices[0][2]),
                        ];
                        let values = basis_values(ElementType::Tri3, self.basis_order, s, t);
                        add_sample(point, reference_weight * jacobian, &values)?;
                    }
                }
                ElementType::Quad4 => {
                    let (points, weights) = gauss_legendre(order);
                    for (&s, &weight_s) in points.iter().zip(&weights) {
                        for (&t, &weight_t) in points.iter().zip(&weights) {
                            let (point, _, jacobian) = quad4_geometry(&coords, s, t);
                            let values = basis_values(ElementType::Quad4, self.basis_order, s, t);
                            add_sample(point, weight_s * weight_t * jacobian, &values)?;
                        }
                    }
                }
            }
        }
        lu_solve(&mass, &load).map_err(|error| {
            BemError::SolverFailed(format!(
                "higher-order soft-Dirichlet incident projection failed: {error}"
            ))
        })
    }

    fn soft_dirichlet_rhs_for_incident(
        &self,
        physics: &PhysicsParams,
        incident_field: &IncidentField,
        elements: &[Element],
    ) -> Vec<Complex64> {
        elements
            .iter()
            .filter(|elem| !elem.property.is_evaluation())
            .map(|element| {
                let point: [[f64; 3]; 1] = [element.center];
                -incident_field.evaluate_pressure(&point, physics)[0]
            })
            .collect()
    }

    fn soft_dirichlet_rhs(&self, problem: &BemProblem, elements: &[Element]) -> Vec<Complex64> {
        self.soft_dirichlet_rhs_for_incident(&problem.physics, &problem.incident_field, elements)
    }

    fn solve_soft_dirichlet_adjoint_many(
        &self,
        problem: &BemProblem,
        elements: &[Element],
        rhs_columns: &[Vec<Complex64>],
    ) -> Result<Vec<BemAdjointSolution>, BemError> {
        let matrix = self.assemble_soft_dirichlet_matrix(problem, elements)?;
        let num_dofs = matrix.nrows();
        let mut hermitian = DenseMatrix::<Complex64>::zeros(num_dofs, num_dofs);
        for row in 0..num_dofs {
            for column in 0..num_dofs {
                hermitian[[row, column]] = matrix[[column, row]].conj();
            }
        }
        let rhs_block = DenseMatrix::from_fn(num_dofs, rhs_columns.len(), |row, column| {
            rhs_columns[column][row]
        });
        let factor =
            lu_factorize(&hermitian).map_err(|error| BemError::SolverFailed(error.to_string()))?;
        let values = factor
            .solve_many(rhs_block.as_slice(), rhs_columns.len())
            .map_err(|error| BemError::SolverFailed(error.to_string()))?;
        let solution = DenseMatrix::from_row_major_vec(num_dofs, rhs_columns.len(), values);
        let operator = DenseMatrixOperator(&hermitian);
        let residuals = exact_block_residuals(&operator, &solution, rhs_columns)?;
        Ok((0..rhs_columns.len())
            .map(|column| BemAdjointSolution {
                coefficients: solution.column_to_vec(column),
                relative_residual: residuals[column],
                exact_reference_residual: None,
                elements: elements.to_vec(),
                nodes: problem.mesh.nodes.clone(),
                physics: problem.physics.clone(),
                basis_order: self.basis_order,
            })
            .collect())
    }

    fn solve_soft_dirichlet_indirect(
        &self,
        problem: &BemProblem,
        elements: Vec<Element>,
    ) -> Result<BemSolution, BemError> {
        let matrix = self.assemble_soft_dirichlet_matrix(problem, &elements)?;
        let rhs = if self.basis_order == BasisOrder::Constant {
            self.soft_dirichlet_rhs(problem, &elements)
        } else {
            self.soft_dirichlet_rhs_higher_order(problem, &elements)?
        };
        let density = self.solve_dense_system(&matrix, &rhs, &BemPrecondImpl::Identity)?;

        Ok(BemSolution {
            surface_pressure: density,
            elements,
            nodes: problem.mesh.nodes.clone(),
            incident_field: problem.incident_field.clone(),
            physics: problem.physics.clone(),
            basis_order: self.basis_order,
            representation: BemSolutionRepresentation::IndirectSingleLayer,
        })
    }

    /// Prepare elements with boundary conditions
    fn prepare_elements(&self, problem: &BemProblem) -> Result<Vec<Element>, BemError> {
        let mut elements = problem.mesh.elements.clone();
        let continuous_map = if self.basis_continuity == BasisContinuity::Continuous
            && self.basis_order == BasisOrder::Quadratic
        {
            Some(
                problem
                    .mesh
                    .continuous_quadratic_dof_map()
                    .map_err(|error| BemError::InvalidMesh(error.to_string()))?,
            )
        } else {
            None
        };

        let thin_shell_admittance = if problem.bc_type == BoundaryConditionType::Impedance {
            self.thin_shell_surface
                .map(|surface| {
                    surface
                        .try_admittance(problem.physics.frequency)
                        .map_err(|error| BemError::InvalidParameters(error.to_string()))
                })
                .transpose()?
        } else if self.thin_shell_surface.is_some() {
            return Err(BemError::InvalidParameters(
                "impedance surface model requires impedance boundary".to_string(),
            ));
        } else {
            None
        };
        let layered_admittance = if problem.bc_type == BoundaryConditionType::Impedance {
            self.layered_porous_surface
                .as_ref()
                .map(|surface| {
                    surface
                        .try_admittance(
                            problem.physics.frequency,
                            crate::materials::AmbientMedium::new(
                                problem.physics.density,
                                problem.physics.speed_of_sound,
                            ),
                        )
                        .map_err(|error| BemError::InvalidParameters(error.to_string()))
                })
                .transpose()?
        } else if self.layered_porous_surface.is_some() {
            return Err(BemError::InvalidParameters(
                "layered porous surface requires an impedance boundary".to_string(),
            ));
        } else {
            None
        };

        // Set boundary conditions based on problem type
        let bc = match problem.bc_type {
            BoundaryConditionType::Rigid => {
                // Zero normal velocity (Neumann BC)
                BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)])
            }
            BoundaryConditionType::Soft => {
                // Zero pressure (Dirichlet BC)
                BoundaryCondition::Pressure(vec![Complex64::new(0.0, 0.0)])
            }
            BoundaryConditionType::Impedance => {
                // Default impedance (plane wave), unless a checked porous
                // surface model was configured on the solver.
                let z0 = problem.physics.density * problem.physics.speed_of_sound;
                BoundaryCondition::VelocityWithAdmittance {
                    velocity: vec![Complex64::new(0.0, 0.0)],
                    admittance: layered_admittance
                        .or(thin_shell_admittance)
                        .unwrap_or_else(|| Complex64::new(1.0 / z0, 0.0)),
                }
            }
        };

        // Assign BC and DOF addresses based on basis order
        if let Some(map) = &continuous_map {
            for (element_index, elem) in elements.iter_mut().enumerate() {
                elem.boundary_condition = bc.clone();
                elem.dof_addresses = map.element_dofs(element_index).unwrap_or(&[]).to_vec();
            }
            return Ok(elements);
        }

        if self.basis_continuity == BasisContinuity::Continuous
            && self.basis_order == BasisOrder::Linear
        {
            for elem in &mut elements {
                elem.boundary_condition = bc.clone();
                elem.dof_addresses = elem.connectivity.clone();
            }
            return Ok(elements);
        }

        let mut dof_counter = 0usize;
        for elem in elements.iter_mut() {
            elem.boundary_condition = bc.clone();
            let num_dofs = elem.num_dofs_for_basis(self.basis_order);
            elem.dof_addresses = (dof_counter..dof_counter + num_dofs).collect();
            dof_counter += num_dofs;
        }
        Ok(elements)
    }

    /// Build the leaf clusters used by the production H-matrix assembly.
    ///
    /// Keeping this construction in one helper is important for derivative
    /// operators: the value and derivative H-matrices must have identical
    /// near/far topology for their actions to be comparable.
    fn build_hmatrix_clusters(elements: &[Element]) -> Vec<crate::core::types::Cluster> {
        use crate::core::mesh::octree::Octree;
        use crate::core::types::Cluster;
        use std::collections::HashMap;

        let centers: Vec<[f64; 3]> = elements.iter().map(|element| element.center).collect();
        let mut octree = Octree::build(&centers, 32, 8);
        octree.compute_interaction_lists(1.2);

        let leaf_indices = octree.leaves();
        let mut node_to_cluster = HashMap::with_capacity(leaf_indices.len());
        for (cluster_index, &node_index) in leaf_indices.iter().enumerate() {
            node_to_cluster.insert(node_index, cluster_index);
        }

        leaf_indices
            .iter()
            .map(|&node_index| {
                let node = &octree.nodes[node_index];
                let mut cluster = Cluster::new(node.center);
                cluster.element_indices = node.element_indices.clone();
                cluster.num_elements = node.element_indices.len();
                cluster.radius = node.radius();
                cluster.near_clusters = node
                    .near_clusters
                    .iter()
                    .filter_map(|index| node_to_cluster.get(index).copied())
                    .collect();
                cluster.far_clusters = node
                    .far_clusters
                    .iter()
                    .filter_map(|index| node_to_cluster.get(index).copied())
                    .collect();
                cluster
            })
            .collect()
    }

    /// Assemble the BEM system matrix with stability parameters
    fn assemble_system_stabilized(
        &self,
        elements: &[Element],
        nodes: &[[f64; 3]],
        physics: &PhysicsParams,
        sign: f64,
        beta_scale: f64,
    ) -> Result<AssemblyResult, BemError> {
        let num_active_dofs = count_dofs(elements);
        let resolved_method = self.assembly_method.resolve(num_active_dofs);
        if self.verbose && self.assembly_method == AssemblyMethod::Auto {
            log::info!(
                "AssemblyMethod::Auto resolved to {:?} for {} DOFs (threshold {})",
                resolved_method,
                num_active_dofs,
                AUTO_TBEM_DOF_LIMIT
            );
        }
        let higher_order = matches!(self.basis_order, BasisOrder::Linear | BasisOrder::Quadratic);
        if let Some(policy) = self.quadrature_policy {
            validate_tbem_quadrature_for_basis(elements, nodes, physics, self.basis_order, &policy)
                .map_err(|error| {
                    BemError::InvalidParameters(format!(
                        "TBEM quadrature admission rejected assembly: {error}"
                    ))
                })?;
        }
        if higher_order
            && self.basis_continuity == BasisContinuity::Continuous
            && !matches!(
                resolved_method,
                AssemblyMethod::Tbem
                    | AssemblyMethod::Hmatrix
                    | AssemblyMethod::Slfmm
                    | AssemblyMethod::Mlfmm
                    | AssemblyMethod::DirectionalMlfmm
            )
        {
            return Err(BemError::NotImplemented(
            "continuous linear/quadratic accelerated BEM currently supports CPU TBEM, H-matrix, and FMM"
                    .to_string(),
            ));
        }
        if higher_order
            && self.basis_continuity == BasisContinuity::Discontinuous
            && !matches!(
                resolved_method,
                AssemblyMethod::Tbem
                    | AssemblyMethod::Hmatrix
                    | AssemblyMethod::Slfmm
                    | AssemblyMethod::Mlfmm
                    | AssemblyMethod::DirectionalMlfmm
            )
        {
            return Err(BemError::NotImplemented(format!(
                "discontinuous linear/quadratic accelerated BEM currently supports TBEM, H-matrix, and FMM; \
                 {resolved_method:?} does not provide a basis-aware DOF contract"
            )));
        }
        match resolved_method {
            AssemblyMethod::Auto => unreachable!("resolve() never returns Auto"),
            AssemblyMethod::Tbem => {
                use crate::core::assembly::tbem::build_tbem_system_stabilized;
                let system = match self.basis_order {
                    BasisOrder::Constant => {
                        build_tbem_system_stabilized(elements, nodes, physics, sign, beta_scale)
                    }
                    BasisOrder::Linear => {
                        use crate::core::assembly::tbem::build_tbem_system_linear_stabilized;
                        build_tbem_system_linear_stabilized(
                            elements, nodes, physics, sign, beta_scale,
                        )
                    }
                    BasisOrder::Quadratic => {
                        if self.basis_continuity == BasisContinuity::Continuous {
                            use crate::core::assembly::tbem::build_tbem_system_continuous_quadratic_stabilized;
                            build_tbem_system_continuous_quadratic_stabilized(
                                elements, nodes, physics, sign, beta_scale,
                            )
                        } else {
                            use crate::core::assembly::tbem::build_tbem_system_quadratic_stabilized;
                            build_tbem_system_quadratic_stabilized(
                                elements, nodes, physics, sign, beta_scale,
                            )
                        }
                    }
                };
                Ok(AssemblyResult::Dense(system.matrix, system.rhs))
            }
            AssemblyMethod::Slfmm => {
                use crate::core::mesh::octree::Octree;
                use crate::core::types::Cluster;
                use std::collections::HashMap;

                // Build octree from element centers
                let centers: Vec<[f64; 3]> = elements.iter().map(|e| e.center).collect();

                // Target ~32 elements per leaf cluster, max depth 8
                let max_per_leaf = 32;
                let max_depth = 8;
                let mut octree = Octree::build(&centers, max_per_leaf, max_depth);

                // Compute near/far interaction lists
                // separation_ratio 1.8: clusters are "far" if distance > 1.8*(r_i+r_j)
                // Lower ratios (e.g. 1.2) are too aggressive for the plane-wave approximation
                // and produce >20% far-field error; 1.8 keeps error below ~5%.
                octree.compute_interaction_lists(1.8);

                // Get leaf node indices and build mapping from node index to cluster index
                let leaf_indices = octree.leaves();
                let mut node_to_cluster: HashMap<usize, usize> = HashMap::new();
                for (cluster_idx, &node_idx) in leaf_indices.iter().enumerate() {
                    node_to_cluster.insert(node_idx, cluster_idx);
                }

                // Convert octree leaves to Cluster structs
                let mut clusters: Vec<Cluster> = Vec::with_capacity(leaf_indices.len());
                for &node_idx in &leaf_indices {
                    let node = &octree.nodes[node_idx];
                    let center = node.center;
                    let mut cluster = Cluster::new(center);
                    cluster.element_indices = node.element_indices.clone();
                    cluster.num_elements = node.element_indices.len();
                    // Radius = half-diagonal of the bounding box
                    cluster.radius = node.radius();

                    // Map near/far node indices to cluster indices
                    cluster.near_clusters = node
                        .near_clusters
                        .iter()
                        .filter_map(|ni| node_to_cluster.get(ni).copied())
                        .collect();
                    cluster.far_clusters = node
                        .far_clusters
                        .iter()
                        .filter_map(|ni| node_to_cluster.get(ni).copied())
                        .collect();

                    clusters.push(cluster);
                }

                // Estimate characteristic radius for FMM parameter selection
                let characteristic_radius =
                    clusters.iter().map(|c| c.radius).fold(0.0f64, f64::max);

                // FMM expansion parameters based on wavenumber * cluster radius
                let ka = physics.wave_number * characteristic_radius;
                let n_terms = (ka.ceil() as usize + 3).clamp(5, 20);
                let n_theta = 2 * n_terms;
                let n_phi = 2 * n_theta;

                let system = if higher_order {
            use crate::core::assembly::slfmm::build_slfmm_system_stabilized_with_basis_checked;
            build_slfmm_system_stabilized_with_basis_checked(
                elements,
                nodes,
                &clusters,
                physics,
                self.basis_order,
                n_theta,
                n_phi,
                n_terms,
                sign,
                beta_scale,
            )
        } else {
            use crate::core::assembly::slfmm::build_slfmm_system_stabilized_checked;
            build_slfmm_system_stabilized_checked(
                elements, nodes, &clusters, physics, n_theta, n_phi, n_terms, sign, beta_scale,
            )
        }
        .map_err(|error| {
            BemError::InvalidParameters(format!("SLFMM admission rejected: {error}"))
        })?;
                Ok(AssemblyResult::Slfmm(system))
            }
            AssemblyMethod::Mlfmm | AssemblyMethod::DirectionalMlfmm => {
                use crate::core::assembly::mlfmm::build_cluster_tree;

                // Build hierarchical cluster tree from elements
                // Target ~32 elements per leaf cluster
                let target_per_leaf = 32;
                let cluster_levels = build_cluster_tree(elements, target_per_leaf, physics);

                let system = if higher_order {
            if matches!(self.assembly_method, AssemblyMethod::DirectionalMlfmm) {
                use crate::core::assembly::mlfmm::
                    build_directional_mlfmm_system_stabilized_with_basis_checked;
                build_directional_mlfmm_system_stabilized_with_basis_checked(
                    elements,
                    nodes,
                    cluster_levels,
                    physics,
                    self.basis_order,
                    sign,
                    beta_scale,
                    Default::default(),
                )
            } else {
                use crate::core::assembly::mlfmm::build_mlfmm_system_stabilized_with_basis_checked;
                build_mlfmm_system_stabilized_with_basis_checked(
                    elements,
                    nodes,
                    cluster_levels,
                    physics,
                    self.basis_order,
                    sign,
                    beta_scale,
                )
            }
        } else if matches!(self.assembly_method, AssemblyMethod::DirectionalMlfmm) {
            use crate::core::assembly::mlfmm::build_directional_mlfmm_system_stabilized_checked;
            build_directional_mlfmm_system_stabilized_checked(
                elements,
                nodes,
                cluster_levels,
                physics,
                sign,
                beta_scale,
                Default::default(),
            )
        } else {
            use crate::core::assembly::mlfmm::build_mlfmm_system_stabilized_checked;
            build_mlfmm_system_stabilized_checked(
                elements,
                nodes,
                cluster_levels,
                physics,
                sign,
                beta_scale,
            )
        }
            .map_err(|error| {
                BemError::InvalidParameters(format!(
                    "MLFMM constant-panel admission rejected: {error}"
                ))
            })?;
                Ok(AssemblyResult::Mlfmm(Box::new(system)))
            }
            AssemblyMethod::Hmatrix => {
                use crate::core::assembly::hmatrix::build_hmatrix_stabilized;
                use crate::core::mesh::octree::Octree;
                use crate::core::types::Cluster;
                use std::collections::HashMap;

                let use_continuous_proxies =
                    higher_order && self.basis_continuity == BasisContinuity::Continuous;
                let (continuous_proxies, continuous_element_proxies) = if use_continuous_proxies {
                    build_continuous_hmatrix_proxy_elements(elements, nodes, self.basis_order)
                } else {
                    (Vec::new(), Vec::new())
                };
                let hmatrix_elements = if use_continuous_proxies {
                    continuous_proxies.as_slice()
                } else {
                    elements
                };

                // Build octree from element centers
                let centers: Vec<[f64; 3]> = hmatrix_elements.iter().map(|e| e.center).collect();

                // Target ~32 elements per leaf cluster, max depth 8
                let max_per_leaf = 32;
                let max_depth = 8;
                let mut octree = Octree::build(&centers, max_per_leaf, max_depth);

                // Compute near/far interaction lists
                octree.compute_interaction_lists(1.2);

                // Get leaf node indices and build mapping from node index to cluster index
                let leaf_indices = octree.leaves();
                let mut node_to_cluster: HashMap<usize, usize> = HashMap::new();
                for (cluster_idx, &node_idx) in leaf_indices.iter().enumerate() {
                    node_to_cluster.insert(node_idx, cluster_idx);
                }

                // Convert octree leaves to Cluster structs
                let mut clusters: Vec<Cluster> = Vec::with_capacity(leaf_indices.len());
                for &node_idx in &leaf_indices {
                    let node = &octree.nodes[node_idx];
                    let center = node.center;
                    let mut cluster = Cluster::new(center);
                    cluster.element_indices = node.element_indices.clone();
                    cluster.num_elements = node.element_indices.len();
                    cluster.radius = node.radius();

                    cluster.near_clusters = node
                        .near_clusters
                        .iter()
                        .filter_map(|ni| node_to_cluster.get(ni).copied())
                        .collect();
                    cluster.far_clusters = node
                        .far_clusters
                        .iter()
                        .filter_map(|ni| node_to_cluster.get(ni).copied())
                        .collect();

                    clusters.push(cluster);
                }

                let clusters = if use_continuous_proxies {
                    let base_clusters = Self::build_hmatrix_clusters(elements);
                    expand_hmatrix_clusters_for_proxies(&base_clusters, &continuous_element_proxies)
                } else {
                    clusters
                };

                // Build H-matrix with ACA
                let aca_tolerance = self.hmatrix_aca_tolerance;
                let max_rank = if self.hmatrix_max_rank == 0 {
                    count_dofs(elements) / 4 + 1
                } else {
                    self.hmatrix_max_rank
                };
                let hmatrix = build_hmatrix_stabilized(
                    hmatrix_elements,
                    nodes,
                    &clusters,
                    physics,
                    aca_tolerance,
                    max_rank,
                    sign,
                    beta_scale,
                );

                // Build RHS using TBEM (the RHS only needs element-level contributions)
                let tbem_system = match self.basis_order {
                    BasisOrder::Constant => {
                        build_tbem_system_stabilized(elements, nodes, physics, sign, beta_scale)
                    }
                    BasisOrder::Linear => {
                        let beta = physics.burton_miller_beta_scaled(beta_scale);
                        build_tbem_system_linear(elements, nodes, physics, beta)
                    }
                    BasisOrder::Quadratic => {
                        // The H-matrix operator is basis-aware, but the incident/RHS
                        // contribution remains assembled by the trusted TBEM path.
                        build_tbem_system_quadratic_stabilized(
                            elements, nodes, physics, sign, beta_scale,
                        )
                    }
                };

                Ok(AssemblyResult::Hmatrix(hmatrix, tbem_system.rhs))
            }
        }
    }

    /// Add incident field contribution to RHS (constant elements only)
    #[allow(dead_code)]
    fn add_incident_field_rhs(
        &self,
        mut rhs: Vec<Complex64>,
        elements: &[Element],
        incident_field: &IncidentField,
        physics: &PhysicsParams,
        use_burton_miller: bool,
    ) -> Vec<Complex64> {
        match self.basis_order {
            BasisOrder::Constant => {
                let centers: Vec<[f64; 3]> = elements.iter().map(|e| e.center).collect();
                let normals: Vec<[f64; 3]> = elements.iter().map(|e| e.normal).collect();

                let incident_rhs = if use_burton_miller {
                    let beta = physics.burton_miller_beta_scaled(self.beta_scale);
                    incident_field.compute_rhs_with_beta(&centers, &normals, physics, beta)
                } else {
                    incident_field.compute_rhs(&centers, &normals, physics, false)
                };

                for (r, inc) in rhs.iter_mut().zip(incident_rhs.iter()) {
                    *r += *inc;
                }
            }
            BasisOrder::Linear | BasisOrder::Quadratic => {
                // Linear basis RHS computation requires mesh nodes.
                // Use add_incident_field_rhs_with_nodes() instead (called by solve()).
                unreachable!(
                    "Linear basis RHS computation requires mesh nodes; use solve() which calls the correct path"
                );
            }
        }

        rhs
    }

    /// Add incident field contribution to RHS (with mesh nodes for linear elements)
    fn add_incident_field_rhs_stabilized(
        &self,
        mut rhs: Vec<Complex64>,
        elements: &[Element],
        nodes: &[[f64; 3]],
        incident_field: &IncidentField,
        physics: &PhysicsParams,
        use_burton_miller: bool,
        sign: f64,
        beta_scale: f64,
    ) -> Vec<Complex64> {
        match self.basis_order {
            BasisOrder::Constant => {
                let centers: Vec<[f64; 3]> = elements.iter().map(|e| e.center).collect();
                let normals_arr: Vec<[f64; 3]> = elements.iter().map(|elem| elem.normal).collect();

                let incident_rhs = if use_burton_miller {
                    let beta = physics.burton_miller_beta_scaled(beta_scale);
                    incident_field.compute_rhs_with_beta(&centers, &normals_arr, physics, beta)
                } else {
                    incident_field.compute_rhs(&centers, &normals_arr, physics, false)
                };

                // RHS: compute_rhs_with_beta returns -(p_inc + β*τ*∂p_inc/∂n)
                // Stabilized: sign * incident_rhs
                let sign_c = Complex64::new(sign, 0.0);
                for (r, inc) in rhs.iter_mut().zip(incident_rhs.iter()) {
                    *r += *inc * sign_c;
                }
            }
            BasisOrder::Linear | BasisOrder::Quadratic => {
                if self.basis_continuity == BasisContinuity::Continuous {
                    let beta = physics.burton_miller_beta_scaled(beta_scale);
                    let gamma = Complex64::new(physics.gamma(), 0.0);
                    let tau = Complex64::new(physics.tau, 0.0);
                    let mut seen = vec![false; rhs.len()];
                    for element in elements.iter().filter(|e| !e.property.is_evaluation()) {
                        let collocation_points =
                            element.collocation_points(self.basis_order, nodes);
                        for (local, &dof) in element.dof_addresses.iter().enumerate() {
                            if dof >= seen.len() || seen[dof] {
                                continue;
                            }
                            seen[dof] = true;
                            let point_arr: [[f64; 3]; 1] = [collocation_points[local]];
                            let normal_arr: [[f64; 3]; 1] = [element.normal];
                            let p_inc = incident_field.evaluate_pressure(&point_arr, physics);
                            let dpdn = incident_field.evaluate_normal_derivative(
                                &point_arr,
                                &normal_arr,
                                physics,
                            );
                            rhs[dof] += if use_burton_miller {
                                -sign * (gamma * p_inc[0] + beta * tau * dpdn[0])
                            } else {
                                -sign * gamma * p_inc[0]
                            };
                        }
                    }
                    return rhs;
                }
                let basis = self.basis_order;
                let beta = physics.burton_miller_beta_scaled(beta_scale);
                let gamma = Complex64::new(physics.gamma(), 0.0);
                let tau = Complex64::new(physics.tau, 0.0);

                for elem in elements.iter() {
                    if elem.property.is_evaluation() {
                        continue;
                    }

                    let coll_points = elem.collocation_points(basis, nodes);
                    let coll_normals = vec![elem.normal; elem.dof_addresses.len()];

                    for (c, dof) in elem.dof_addresses.iter().enumerate() {
                        let point = &coll_points[c];
                        let normal = &coll_normals[c];

                        // Evaluate incident field at this collocation point
                        let point_arr: [[f64; 3]; 1] = [*point];
                        let normal_arr: [[f64; 3]; 1] = [*normal];

                        let p_inc = incident_field.evaluate_pressure(&point_arr, physics);
                        let dpdn = incident_field.evaluate_normal_derivative(
                            &point_arr,
                            &normal_arr,
                            physics,
                        );

                        // Burton-Miller RHS: -(γ*p_inc + β*τ*∂p_inc/∂n)
                        // Apply formulation sign
                        if use_burton_miller {
                            rhs[*dof] += -sign * (gamma * p_inc[0] + beta * tau * dpdn[0]);
                        } else {
                            rhs[*dof] += -sign * gamma * p_inc[0];
                        }
                    }
                }
            }
        }

        rhs
    }

    /// Solve dense linear system
    fn solve_dense_system(
        &self,
        matrix: &sonium_solvers::dense::DenseMatrix<Complex64>,
        rhs: &[Complex64],
        precond: &BemPrecondImpl,
    ) -> Result<Vec<Complex64>, BemError> {
        match self.solver_method {
            SolverMethod::Direct => {
                let solution =
                    lu_solve(matrix, rhs).map_err(|e| BemError::SolverFailed(e.to_string()))?;
                self.check_exact_residual(
                    &DenseMatrixOperator(matrix),
                    &solution,
                    rhs,
                    "dense-direct",
                )?;
                Ok(solution)
            }
            SolverMethod::Cgs | SolverMethod::BiCgStab => {
                let config = BiCgstabConfig {
                    max_iterations: self.max_iterations,
                    tolerance: self.tolerance,
                    print_interval: 0,
                };

                match bicgstab(&DenseMatrixOperator(matrix), rhs, &config) {
                    sol if sol.converged => {
                        self.check_exact_residual(
                            &DenseMatrixOperator(matrix),
                            &sol.x,
                            rhs,
                            "dense",
                        )?;
                        Ok(sol.x)
                    }
                    sol => Err(BemError::SolverFailed(format!(
                        "BiCGSTAB did not converge: residual = {}",
                        sol.residual
                    ))),
                }
            }
            SolverMethod::Gmres => {
                self.run_gmres(&DenseMatrixOperator(matrix), precond, rhs, "dense")
            }
        }
    }

    /// Solve FMM system using matrix-vector products
    fn solve_fmm_system(
        &self,
        system: SlfmmSystem,
        rhs: &[Complex64],
    ) -> Result<Vec<Complex64>, BemError> {
        match self.solver_method {
            SolverMethod::Direct => {
                if system.num_dofs <= 2000 {
                    let matrix = system
                        .try_extract_near_field_matrix()
                        .map_err(|error| BemError::SolverFailed(error.to_string()))?;
                    let solution = lu_solve(&matrix, rhs)
                        .map_err(|e| BemError::SolverFailed(e.to_string()))?;
                    self.check_exact_residual(
                        &DenseMatrixOperator(&matrix),
                        &solution,
                        rhs,
                        "slfmm-direct-near-field",
                    )?;
                    Ok(solution)
                } else {
                    Err(BemError::NotImplemented(
                        "Direct solver not available for large FMM problems".to_string(),
                    ))
                }
            }
            SolverMethod::Cgs | SolverMethod::BiCgStab => {
                let config = BiCgstabConfig {
                    max_iterations: self.max_iterations,
                    tolerance: self.tolerance,
                    print_interval: 0,
                };

                match bicgstab(&system, rhs, &config) {
                    sol if sol.converged => {
                        self.check_exact_residual(&system, &sol.x, rhs, "slfmm")?;
                        Ok(sol.x)
                    }
                    sol => Err(BemError::SolverFailed(format!(
                        "BiCGSTAB did not converge: residual = {}",
                        sol.residual
                    ))),
                }
            }
            SolverMethod::Gmres => {
                let config = GmresConfig {
                    max_iterations: self.max_iterations,
                    restart: self.gmres_restart,
                    tolerance: self.tolerance,
                    print_interval: 0,
                    convergence_check_interval: 1,
                    exact_residual_check_interval: 0,
                    min_restart: None,
                    max_restart: None,
                };
                if self.preconditioner == BemPreconditioner::HierarchicalIlu {
                    let solution = crate::core::solver::try_gmres_solve_fmm_batched_with_hilu(
                        &system, rhs, &config,
                    )
                    .map_err(|error| {
                        BemError::SolverFailed(format!("SLFMM hierarchical ILU GMRES: {error}"))
                    })?;
                    if !solution.converged {
                        return Err(BemError::SolverFailed(format!(
                            "GMRES SLFMM hierarchical ILU did not converge: residual = {}",
                            solution.residual
                        )));
                    }
                    return Ok(solution.x);
                }
                let nearfield_matrix = system.try_extract_near_field_matrix().map_err(|error| {
                    BemError::SolverFailed(format!("SLFMM near-field: {error}"))
                })?;
                let nearfield =
                    sonium_solvers::sparse::CsrMatrix::from_dense(&nearfield_matrix, 1.0e-15);
                let precond =
                    sonium_solvers::preconditioners::IluPreconditioner::try_from_csr(&nearfield)
                        .map_err(|error| {
                            BemError::SolverFailed(format!("SLFMM near-field ILU: {error}"))
                        })?;
                match sonium_solvers::iterative::gmres_preconditioned(
                    &system, &precond, rhs, &config,
                ) {
                    sol if sol.converged => {
                        self.check_exact_residual(&system, &sol.x, rhs, "slfmm-ilu")?;
                        Ok(sol.x)
                    }
                    sol => Err(BemError::SolverFailed(format!(
                        "GMRES SLFMM near-field ILU did not converge: residual = {}",
                        sol.residual
                    ))),
                }
            }
        }
    }

    /// Solve MLFMM system using iterative solver
    fn solve_mlfmm_system(
        &self,
        system: crate::core::assembly::mlfmm::MlfmmSystem,
        rhs: &[Complex64],
        precond: &BemPrecondImpl,
    ) -> Result<Vec<Complex64>, BemError> {
        use crate::core::solver::MlfmmOperator;

        // Match the batched MLFMM path: the default GMRES solve uses an ILU
        // built from the exact near-field block, while explicit preconditioners
        // remain under the caller's control.
        let default_nearfield_precond = if matches!(self.solver_method, SolverMethod::Gmres)
            && self.preconditioner != BemPreconditioner::HierarchicalIlu
            && matches!(precond, BemPrecondImpl::Identity)
        {
            let nearfield_matrix = system
                .try_extract_near_field_matrix()
                .map_err(|error| BemError::SolverFailed(format!("MLFMM near-field: {error}")))?;
            let nearfield =
                sonium_solvers::sparse::CsrMatrix::from_dense(&nearfield_matrix, 1.0e-15);
            let ilu = sonium_solvers::preconditioners::IluPreconditioner::try_from_csr(&nearfield)
                .map_err(|error| {
                    BemError::SolverFailed(format!("MLFMM near-field ILU: {error}"))
                })?;
            Some(BemPrecondImpl::Ilu(ilu))
        } else {
            None
        };

        if self.solver_method == SolverMethod::Gmres
            && self.preconditioner == BemPreconditioner::HierarchicalIlu
        {
            let config = GmresConfig {
                max_iterations: self.max_iterations,
                restart: self.gmres_restart,
                tolerance: self.tolerance,
                print_interval: 0,
                convergence_check_interval: 1,
                exact_residual_check_interval: 0,
                min_restart: None,
                max_restart: None,
            };
            let solution =
                crate::core::solver::try_gmres_solve_mlfmm_batched_with_hilu(&system, rhs, &config)
                    .map_err(|error| {
                        BemError::SolverFailed(format!("MLFMM hierarchical ILU GMRES: {error}"))
                    })?;
            if !solution.converged {
                return Err(BemError::SolverFailed(format!(
                    "GMRES MLFMM hierarchical ILU did not converge: residual = {}",
                    solution.residual
                )));
            }
            return Ok(solution.x);
        }

        let operator = MlfmmOperator::new(system);

        match self.solver_method {
            SolverMethod::Direct => Err(BemError::NotImplemented(
                "Direct solver not available for MLFMM problems; use BiCgStab or Cgs".to_string(),
            )),
            SolverMethod::Cgs | SolverMethod::BiCgStab => {
                let config = BiCgstabConfig {
                    max_iterations: self.max_iterations,
                    tolerance: self.tolerance,
                    print_interval: 0,
                };

                match bicgstab(&operator, rhs, &config) {
                    sol if sol.converged => {
                        self.check_exact_residual(&operator, &sol.x, rhs, "mlfmm")?;
                        Ok(sol.x)
                    }
                    sol => Err(BemError::SolverFailed(format!(
                        "BiCGSTAB did not converge for MLFMM: residual = {}",
                        sol.residual
                    ))),
                }
            }
            SolverMethod::Gmres => {
                if let Some(default_precond) = default_nearfield_precond.as_ref() {
                    self.run_gmres(&operator, default_precond, rhs, "mlfmm-ilu")
                } else {
                    self.run_gmres(&operator, precond, rhs, "mlfmm")
                }
            }
        }
    }

    /// Solve H-matrix system using iterative solver
    fn solve_hmatrix_system(
        &self,
        hmatrix: crate::core::assembly::hmatrix::HMatrix,
        rhs: &[Complex64],
        precond: &BemPrecondImpl,
    ) -> Result<Vec<Complex64>, BemError> {
        if matches!(precond, BemPrecondImpl::HierarchicalIlu) {
            return Err(BemError::NotImplemented(
                "HierarchicalIlu requires SLFMM or MLFMM assembly".to_string(),
            ));
        }
        match self.solver_method {
            SolverMethod::Direct => Err(BemError::NotImplemented(
                "Direct solver not available for H-matrix problems; use BiCgStab or Cgs"
                    .to_string(),
            )),
            SolverMethod::Cgs | SolverMethod::BiCgStab => {
                let config = BiCgstabConfig {
                    max_iterations: self.max_iterations,
                    tolerance: self.tolerance,
                    print_interval: 0,
                };

                match bicgstab(&hmatrix, rhs, &config) {
                    sol if sol.converged => {
                        self.check_exact_residual(&hmatrix, &sol.x, rhs, "hmatrix")?;
                        Ok(sol.x)
                    }
                    sol => Err(BemError::SolverFailed(format!(
                        "BiCGSTAB did not converge for H-matrix: residual = {}",
                        sol.residual
                    ))),
                }
            }
            SolverMethod::Gmres => self.run_gmres(&hmatrix, precond, rhs, "hmatrix"),
        }
    }

    /// Solve a BEM problem across multiple frequencies with solution recycling.
    ///
    /// At each frequency, the BEM system is reassembled (the mesh stays fixed)
    /// and the converged solution from the previous frequency is used as the
    /// initial guess for the iterative solver. This dramatically reduces total
    /// computation time for closely spaced frequencies.
    ///
    /// # Arguments
    /// * `problem` - The BEM problem definition (frequency will be overridden)
    /// * `frequencies` - Frequencies to solve at (Hz)
    ///
    /// # Returns
    /// Vector of solutions, one per frequency
    fn frequency_sweep_admittances(
        &self,
        problem: &BemProblem,
        frequencies: &[f64],
    ) -> Result<Option<Vec<Complex64>>, BemError> {
        if problem.bc_type != BoundaryConditionType::Impedance {
            if self.layered_porous_surface.is_some() || self.thin_shell_surface.is_some() {
                return Err(BemError::InvalidParameters(
                    "frequency-sweep surface model requires an impedance problem".to_string(),
                ));
            }
            return Ok(None);
        }
        let (layered, thin_shell) = match (
            self.layered_porous_surface.as_ref(),
            self.thin_shell_surface.as_ref(),
        ) {
            (None, None) => return Ok(None),
            (Some(_), Some(_)) => {
                return Err(BemError::InvalidParameters(
                    "layered porous and thin-shell surfaces are mutually exclusive".to_string(),
                ));
            }
            pair => pair,
        };

        let medium = crate::materials::AmbientMedium::new(
            problem.physics.density,
            problem.physics.speed_of_sound,
        );
        frequencies
            .iter()
            .map(|&frequency| {
                let admittance = if let Some(surface) = layered {
                    surface
                        .try_admittance(frequency, medium)
                        .map_err(|error| BemError::InvalidParameters(error.to_string()))?
                } else if let Some(surface) = thin_shell {
                    surface
                        .try_admittance(frequency)
                        .map_err(|error| BemError::InvalidParameters(error.to_string()))?
                } else {
                    unreachable!("validated surface model pair")
                };
                Ok(admittance)
            })
            .collect::<Result<Vec<_>, BemError>>()
            .map(Some)
    }

    /// Solve the problem at each requested frequency using recycled Krylov
    /// iterates and the configured assembly backend.
    pub fn solve_frequency_sweep(
        &self,
        problem: &BemProblem,
        frequencies: &[f64],
    ) -> Result<Vec<BemSolution>, BemError> {
        use crate::core::solver::frequency_sweep::{
            FrequencySweepConfig, frequency_sweep_with_physics_and_admittances,
        };

        self.validate()?;
        problem.validate()?;
        if frequencies.is_empty()
            || frequencies
                .iter()
                .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
        {
            return Err(BemError::InvalidParameters(
                "frequency sweep requires finite, positive frequencies".to_string(),
            ));
        }

        let elements = self.prepare_elements(problem)?;

        let config = FrequencySweepConfig {
            frequencies: frequencies.to_vec(),
            tolerance: self.tolerance,
            max_iterations: self.max_iterations,
            recycle_solution: true,
            gmres_restart: 30,
            beta_scale: self.beta_scale,
            recycle_subspace_size: self.frequency_recycle_subspace_size,
            max_recycle_relative_frequency_delta: 0.1,
        };

        let sweep_admittances = self.frequency_sweep_admittances(problem, frequencies)?;
        let sweep_result = frequency_sweep_with_physics_and_admittances(
            &elements,
            &problem.mesh.nodes,
            &problem.incident_field,
            &config,
            self.assembly_method,
            &problem.physics,
            sweep_admittances.as_deref(),
        )?;

        // Convert sweep results to BemSolution instances
        let mut solutions = Vec::with_capacity(frequencies.len());
        let is_internal = problem.physics.tau < 0.0;
        for (idx, solution) in sweep_result.solutions.into_iter().enumerate() {
            let freq = frequencies[idx];
            let mut physics = PhysicsParams::try_new(
                freq,
                problem.physics.speed_of_sound,
                problem.physics.density,
                is_internal,
            )
            .map_err(|error| BemError::InvalidParameters(error.to_string()))?;
            physics.harmonic_factor = problem.physics.harmonic_factor;
            physics.pressure_factor = physics.density * physics.omega * physics.harmonic_factor;
            physics
                .validate()
                .map_err(|error| BemError::InvalidParameters(error.to_string()))?;

            let solution_elements = if let Some(admittances) = sweep_admittances.as_deref() {
                let admittance = admittances[idx];
                let mut updated = elements.clone();
                for element in &mut updated {
                    if let BoundaryCondition::VelocityWithAdmittance {
                        admittance: current,
                        ..
                    } = &mut element.boundary_condition
                    {
                        *current = admittance;
                    }
                }
                updated
            } else {
                elements.clone()
            };

            solutions.push(BemSolution {
                surface_pressure: solution,
                elements: solution_elements,
                nodes: problem.mesh.nodes.clone(),
                incident_field: problem.incident_field.clone(),
                physics,
                basis_order: self.basis_order,
                representation: BemSolutionRepresentation::DirectCauchy,
            });
        }

        Ok(solutions)
    }

    /// Solve a frequency × incident-field batch while sharing one assembled
    /// operator and block-RHS solve per frequency.
    ///
    /// The outer result dimension follows `frequencies`; each inner vector is
    /// ordered like `incident_fields`. This is the forward multi-load
    /// counterpart to [`Self::solve_many`]: it avoids the accidental
    /// field-by-field reassembly that a caller would incur by nesting scalar
    /// frequency sweeps. Exact residual checks remain those of `solve_many`
    /// for every RHS at every frequency.
    pub fn solve_frequency_sweep_many(
        &self,
        problem: &BemProblem,
        frequencies: &[f64],
        incident_fields: &[IncidentField],
    ) -> Result<Vec<Vec<BemSolution>>, BemError> {
        self.validate()?;
        problem.validate()?;
        if frequencies.is_empty()
            || frequencies
                .iter()
                .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
        {
            return Err(BemError::InvalidParameters(
                "frequency batch requires finite, positive frequencies".to_string(),
            ));
        }
        if incident_fields.is_empty() {
            return Ok((0..frequencies.len()).map(|_| Vec::new()).collect());
        }

        let is_internal = problem.physics.tau < 0.0;
        let mut batches = Vec::with_capacity(frequencies.len());
        for &frequency in frequencies {
            let mut at_frequency = problem.clone();
            let mut physics = PhysicsParams::try_new(
                frequency,
                problem.physics.speed_of_sound,
                problem.physics.density,
                is_internal,
            )
            .map_err(|error| BemError::InvalidParameters(error.to_string()))?;
            physics.harmonic_factor = problem.physics.harmonic_factor;
            physics.pressure_factor = physics.density * physics.omega * physics.harmonic_factor;
            physics
                .validate()
                .map_err(|error| BemError::InvalidParameters(error.to_string()))?;
            at_frequency.physics = physics;
            batches.push(self.solve_many(&at_frequency, incident_fields)?);
        }
        Ok(batches)
    }

    /// Solve several Hermitian-adjoint RHS batches across multiple frequencies.
    ///
    /// The outer result follows `frequencies`; each inner vector contains the
    /// checked adjoint solutions for the corresponding frequency and RHS
    /// order.  Each frequency is assembled with its own physical parameters,
    /// and [`BemAdjointSolution::relative_residual`] is recomputed against the
    /// exact Hermitian operator before the result is returned.  This mirrors
    /// [`Self::solve_frequency_sweep_many`] for objective/driver adjoints
    /// without making callers reassemble one frequency at a time.
    ///
    /// The current implementation intentionally keeps the per-frequency
    /// admission and solver contracts of [`Self::solve_adjoint_many`].  It does
    /// not silently recycle a forward factorization or a non-adjoint
    /// preconditioner; configured backends return their typed capability
    /// errors when an exact Hermitian action is unavailable.
    pub fn solve_adjoint_frequency_sweep_many(
        &self,
        problem: &BemProblem,
        frequencies: &[f64],
        rhs_columns: &[Vec<Vec<Complex64>>],
    ) -> Result<Vec<Vec<BemAdjointSolution>>, BemError> {
        self.validate()?;
        problem.validate()?;
        if frequencies.is_empty()
            || frequencies
                .iter()
                .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
        {
            return Err(BemError::InvalidParameters(
                "adjoint frequency batch requires finite, positive frequencies".to_string(),
            ));
        }
        if frequencies.len() != rhs_columns.len() {
            return Err(BemError::InvalidParameters(format!(
                "adjoint frequency batch has {} frequency entries but {} RHS batches",
                frequencies.len(),
                rhs_columns.len()
            )));
        }

        let is_internal = problem.physics.tau < 0.0;
        let mut batches = Vec::with_capacity(frequencies.len());
        for (&frequency, frequency_rhs) in frequencies.iter().zip(rhs_columns) {
            let mut at_frequency = problem.clone();
            let mut physics = PhysicsParams::try_new(
                frequency,
                problem.physics.speed_of_sound,
                problem.physics.density,
                is_internal,
            )
            .map_err(|error| BemError::InvalidParameters(error.to_string()))?;
            physics.harmonic_factor = problem.physics.harmonic_factor;
            physics.pressure_factor = physics.density * physics.omega * physics.harmonic_factor;
            physics
                .validate()
                .map_err(|error| BemError::InvalidParameters(error.to_string()))?;
            at_frequency.physics = physics;
            batches.push(self.solve_adjoint_many(&at_frequency, frequency_rhs)?);
        }
        Ok(batches)
    }
}

/// Expand continuous higher-order panels into one incidence proxy per shared
/// global DOF.  The H-matrix stores and accumulates these repeated global
/// indices, so a shared trace coefficient receives every panel contribution
/// while the solver still operates on the compact global vector.
fn build_continuous_hmatrix_proxy_elements(
    elements: &[Element],
    nodes: &[[f64; 3]],
    basis: BasisOrder,
) -> (Vec<Element>, Vec<Vec<usize>>) {
    let mut proxies = Vec::new();
    let mut element_proxies = vec![Vec::new(); elements.len()];

    for (element_index, element) in elements.iter().enumerate() {
        if element.property.is_evaluation() {
            continue;
        }
        let collocation_points = element.collocation_points(basis, nodes);
        let collocation_normals = element.collocation_normals(basis, nodes);
        for (local, &dof) in element.dof_addresses.iter().enumerate() {
            let mut proxy = element.clone();
            proxy.dof_addresses = vec![dof];
            proxy.center = collocation_points
                .get(local)
                .copied()
                .unwrap_or(element.center);
            proxy.normal = collocation_normals
                .get(local)
                .copied()
                .unwrap_or(element.normal);
            proxy.group = crate::core::assembly::hmatrix::encode_shared_dof_proxy_group(
                element.dof_addresses.len(),
                local,
            );
            element_proxies[element_index].push(proxies.len());
            proxies.push(proxy);
        }
    }

    (proxies, element_proxies)
}

fn expand_hmatrix_clusters_for_proxies(
    clusters: &[crate::core::types::Cluster],
    element_proxies: &[Vec<usize>],
) -> Vec<crate::core::types::Cluster> {
    clusters
        .iter()
        .enumerate()
        .map(|(cluster_index, cluster)| {
            let mut expanded = cluster.clone();
            expanded.element_indices = cluster
                .element_indices
                .iter()
                .flat_map(|&element_index| element_proxies[element_index].iter().copied())
                .collect();
            expanded.num_elements = expanded.element_indices.len();
            expanded.dofs_per_element = 1;
            if !expanded.near_clusters.contains(&cluster_index) {
                expanded.near_clusters.push(cluster_index);
            }
            expanded
        })
        .collect()
}

/// Certified solution of a high-level BEM Hermitian-adjoint system.
#[derive(Debug, Clone)]
pub struct BemAdjointSolution {
    coefficients: Vec<Complex64>,
    relative_residual: f64,
    exact_reference_residual: Option<f64>,
    elements: Vec<Element>,
    nodes: Vec<[f64; 3]>,
    physics: PhysicsParams,
    basis_order: BasisOrder,
}

impl BemAdjointSolution {
    /// Return the adjoint coefficients in the assembled operator DOF order.
    pub fn coefficients(&self) -> &[Complex64] {
        &self.coefficients
    }

    /// Return the independently recomputed relative residual of `Aᴴ y = b`.
    pub fn relative_residual(&self) -> f64 {
        self.relative_residual
    }

    /// Return the optional residual against an independently evaluated exact
    /// constant-panel TBEM action. `None` means certification was disabled or
    /// the solve was outside its configured bounded scope.
    pub fn exact_reference_residual(&self) -> Option<f64> {
        self.exact_reference_residual
    }

    /// Return the number of adjoint degrees of freedom.
    pub fn num_dofs(&self) -> usize {
        self.coefficients.len()
    }

    /// Evaluate the adjoint scattered pressure at one exterior point.
    pub fn evaluate_scattered_pressure(&self, point: &[f64; 3]) -> Complex64 {
        compute_scattered_field(
            &[*point],
            &self.elements,
            &self.nodes,
            &self.coefficients,
            None,
            &self.physics,
            self.basis_order,
        )
        .into_iter()
        .next()
        .unwrap_or_default()
    }

    /// Evaluate adjoint scattered pressure at multiple exterior points.
    pub fn evaluate_scattered_pressure_field(&self, points: &[[f64; 3]]) -> Vec<Complex64> {
        compute_scattered_field(
            points,
            &self.elements,
            &self.nodes,
            &self.coefficients,
            None,
            &self.physics,
            self.basis_order,
        )
    }

    /// Checked scattered-pressure field evaluation for adjoint coefficients.
    pub fn evaluate_scattered_pressure_field_checked(
        &self,
        points: &[[f64; 3]],
    ) -> Result<Vec<Complex64>, FieldEvaluationError> {
        compute_scattered_field_checked(
            points,
            &self.elements,
            &self.nodes,
            &self.coefficients,
            None,
            &self.physics,
            self.basis_order,
        )
    }

    /// Evaluate an adjoint field after converting dual DOFs to primal traces.
    ///
    /// Higher-order Hermitian-adjoint solves expose Euclidean-dual
    /// coefficients. Use this opt-in method when the returned field is meant
    /// to represent the physical adjoint trace; it applies the surface L²
    /// Riesz map before evaluating the usual exterior representation. The
    /// historical [`Self::evaluate_scattered_pressure_field_checked`] method
    /// remains available for algebraic probes and compatibility.
    pub fn evaluate_scattered_pressure_field_dual_basis_checked(
        &self,
        points: &[[f64; 3]],
    ) -> Result<Vec<Complex64>, FieldEvaluationError> {
        let primal_coefficients = map_dual_basis_coefficients_checked(
            &self.coefficients,
            &self.elements,
            &self.nodes,
            &self.physics,
            self.basis_order,
        )?;
        compute_scattered_field_checked(
            points,
            &self.elements,
            &self.nodes,
            &primal_coefficients,
            None,
            &self.physics,
            self.basis_order,
        )
    }
}

/// Certified forward solution for an arbitrary driver/unit-load RHS.
#[derive(Debug, Clone)]
pub struct BemLoadSolution {
    coefficients: Vec<Complex64>,
    relative_residual: f64,
    exact_reference_residual: Option<f64>,
    elements: Vec<Element>,
    nodes: Vec<[f64; 3]>,
    physics: PhysicsParams,
    basis_order: BasisOrder,
}

impl BemLoadSolution {
    /// Return forward coefficients in assembled operator DOF order.
    pub fn coefficients(&self) -> &[Complex64] {
        &self.coefficients
    }

    /// Return independently recomputed relative residual `A x = b`.
    pub fn relative_residual(&self) -> f64 {
        self.relative_residual
    }

    /// Return the optional residual against an independently evaluated exact
    /// constant-panel TBEM action. `None` means certification was disabled or
    /// the solve was outside its configured bounded scope.
    pub fn exact_reference_residual(&self) -> Option<f64> {
        self.exact_reference_residual
    }

    /// Return number of forward degrees of freedom.
    pub fn num_dofs(&self) -> usize {
        self.coefficients.len()
    }

    /// Evaluate scattered pressure for one exterior point.
    pub fn evaluate_scattered_pressure(&self, point: &[f64; 3]) -> Complex64 {
        compute_scattered_field(
            &[*point],
            &self.elements,
            &self.nodes,
            &self.coefficients,
            None,
            &self.physics,
            self.basis_order,
        )
        .into_iter()
        .next()
        .unwrap_or_default()
    }

    /// Evaluate scattered pressure at multiple exterior points.
    pub fn evaluate_scattered_pressure_field(&self, points: &[[f64; 3]]) -> Vec<Complex64> {
        compute_scattered_field(
            points,
            &self.elements,
            &self.nodes,
            &self.coefficients,
            None,
            &self.physics,
            self.basis_order,
        )
    }

    /// Checked scattered-pressure field evaluation for driver/load solutions.
    pub fn evaluate_scattered_pressure_field_checked(
        &self,
        points: &[[f64; 3]],
    ) -> Result<Vec<Complex64>, FieldEvaluationError> {
        compute_scattered_field_checked(
            points,
            &self.elements,
            &self.nodes,
            &self.coefficients,
            None,
            &self.physics,
            self.basis_order,
        )
    }
}

/// Boundary representation used by a solved BEM system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BemSolutionRepresentation {
    /// Direct formulation storing physical Cauchy pressure data in
    /// `surface_pressure` for the common Neumann/impedance paths.
    DirectCauchy,
    /// Indirect Dirichlet formulation storing a single-layer density.
    IndirectSingleLayer,
}

/// Solution of a BEM problem
#[derive(Debug, Clone)]
pub struct BemSolution {
    /// Surface pressure at each element (for constant) or at each DOF (for linear)
    pub surface_pressure: Vec<Complex64>,
    /// Elements used in the solution
    pub elements: Vec<Element>,
    /// Node coordinates
    pub nodes: Vec<[f64; 3]>,
    /// Incident field used
    pub incident_field: IncidentField,
    /// Physics parameters
    pub physics: PhysicsParams,
    /// Basis function order used
    pub basis_order: BasisOrder,
    /// Representation used to interpret the stored surface unknowns.
    pub representation: BemSolutionRepresentation,
}

impl BemSolution {
    /// Return the solved surface unknowns in assembled operator DOF order.
    ///
    /// Direct-Cauchy solutions store surface pressure values and indirect
    /// solutions store single-layer density values; both are the coefficient
    /// vector acted on by the assembled BEM operator.
    pub fn coefficients(&self) -> &[Complex64] {
        &self.surface_pressure
    }

    /// Evaluate total pressure at a single point
    pub fn evaluate_pressure(&self, point: &[f64; 3]) -> Complex64 {
        let eval_points: [[f64; 3]; 1] = [*point];

        match self.representation {
            BemSolutionRepresentation::DirectCauchy => {
                let field_points = compute_total_field(
                    &eval_points,
                    &self.elements,
                    &self.nodes,
                    &self.surface_pressure,
                    None,
                    &self.incident_field,
                    &self.physics,
                    self.basis_order,
                );

                field_points[0].p_total
            }
            BemSolutionRepresentation::IndirectSingleLayer => {
                let p_incident = self
                    .incident_field
                    .evaluate_pressure(&eval_points, &self.physics);
                let p_scattered = compute_single_layer_field(
                    &eval_points,
                    &self.elements,
                    &self.nodes,
                    &self.surface_pressure,
                    &self.physics,
                    self.basis_order,
                );
                p_incident[0] + p_scattered[0]
            }
        }
    }

    /// Build a metadata-only XEM result envelope for the solved surface
    /// pressure field. The `mem://` reference identifies this solution's
    /// allocation to a caller-owned writer; no pressure values are cloned.
    pub fn to_xem_result(
        &self,
        problem: &sonium_xem::MultiphysicsProblem,
        field_id: &str,
    ) -> Result<sonium_xem::MultiphysicsResult, BemResultError> {
        let field = validate_xem_pressure_field(problem, field_id, self.surface_pressure.len())?;
        let byte_length = self
            .surface_pressure
            .len()
            .checked_mul(std::mem::size_of::<Complex64>())
            .and_then(|size| u64::try_from(size).ok())
            .ok_or_else(|| BemResultError::LengthMismatch {
                field_id: field.id.clone(),
                expected: field.layout.element_count().unwrap_or(usize::MAX),
                actual: self.surface_pressure.len(),
            })?;
        Ok(sonium_xem::MultiphysicsResult::from_problem_field(
            problem,
            field_id,
            sonium_xem::DataReference {
                expression: None,
                uri: Some(format!("mem://sonium/bem/{}/{}", problem.id, field_id)),
                byte_offset: 0,
                byte_length,
            },
        )?)
    }

    /// Borrow the solved surface pressure as a checked, allocation-free XEM
    /// field view. The returned view remains valid for the lifetime of this
    /// solution and is suitable for a direct solver/consumer hand-off.
    pub fn borrowed_xem_field<'a>(
        &'a self,
        problem: &'a sonium_xem::MultiphysicsProblem,
        field_id: &str,
    ) -> Result<sonium_xem::BorrowedFieldView<'a, Complex64>, BemResultError> {
        let field = validate_xem_pressure_field(problem, field_id, self.surface_pressure.len())?;
        Ok(sonium_xem::BorrowedFieldView::new(
            &field.layout,
            &self.surface_pressure,
        )?)
    }

    /// Evaluate total pressure at multiple points
    pub fn evaluate_pressure_field(&self, points: &[[f64; 3]]) -> Vec<FieldPoint> {
        match self.representation {
            BemSolutionRepresentation::DirectCauchy => compute_total_field(
                points,
                &self.elements,
                &self.nodes,
                &self.surface_pressure,
                None,
                &self.incident_field,
                &self.physics,
                self.basis_order,
            ),
            BemSolutionRepresentation::IndirectSingleLayer => {
                let p_incident = self.incident_field.evaluate_pressure(points, &self.physics);
                let p_scattered = compute_single_layer_field(
                    points,
                    &self.elements,
                    &self.nodes,
                    &self.surface_pressure,
                    &self.physics,
                    self.basis_order,
                );
                points
                    .iter()
                    .enumerate()
                    .map(|(i, &position)| FieldPoint::new(position, p_incident[i], p_scattered[i]))
                    .collect()
            }
        }
    }

    /// Evaluate total pressure at multiple points with fail-closed validation.
    ///
    /// Unlike [`Self::evaluate_pressure_field`], this entry point validates
    /// geometry, basis/DOF layout, physics, evaluation coordinates, incident
    /// samples, and computed totals before returning any field values. The
    /// infallible method remains available for trusted legacy callers.
    pub fn evaluate_pressure_field_checked(
        &self,
        points: &[[f64; 3]],
    ) -> Result<Vec<FieldPoint>, FieldEvaluationError> {
        match self.representation {
            BemSolutionRepresentation::DirectCauchy => compute_total_field_checked(
                points,
                &self.elements,
                &self.nodes,
                &self.surface_pressure,
                None,
                &self.incident_field,
                &self.physics,
                self.basis_order,
            ),
            BemSolutionRepresentation::IndirectSingleLayer => {
                let scattered = compute_scattered_field_checked(
                    points,
                    &self.elements,
                    &self.nodes,
                    &self.surface_pressure,
                    None,
                    &self.physics,
                    self.basis_order,
                )?;
                let incident = self.incident_field.evaluate_pressure(points, &self.physics);
                let mut fields = Vec::with_capacity(points.len());
                for (index, ((position, incident), scattered)) in points
                    .iter()
                    .zip(incident.iter())
                    .zip(scattered.iter())
                    .enumerate()
                {
                    if !incident.re.is_finite()
                        || !incident.im.is_finite()
                        || !scattered.re.is_finite()
                        || !scattered.im.is_finite()
                    {
                        return Err(FieldEvaluationError::NonFiniteResult { point: index });
                    }
                    let total = *incident + *scattered;
                    if !total.re.is_finite() || !total.im.is_finite() {
                        return Err(FieldEvaluationError::NonFiniteResult { point: index });
                    }
                    fields.push(FieldPoint::new(*position, *incident, *scattered));
                }
                Ok(fields)
            }
        }
    }

    /// Evaluate the total pressure gradient at multiple exterior points with
    /// fail-closed geometry, basis, and finite-value validation.
    ///
    /// Each returned vector is ordered as `[∂p/∂x, ∂p/∂y, ∂p/∂z]`. The
    /// boundary representation uses analytic Green-function derivatives;
    /// this is therefore a physical field derivative rather than a finite
    /// difference of repeated field evaluations.
    pub fn evaluate_pressure_gradient_field_checked(
        &self,
        points: &[[f64; 3]],
    ) -> Result<Vec<[Complex64; 3]>, FieldEvaluationError> {
        match self.representation {
            BemSolutionRepresentation::DirectCauchy => compute_total_field_gradient_checked(
                points,
                &self.elements,
                &self.nodes,
                &self.surface_pressure,
                None,
                &self.incident_field,
                &self.physics,
                self.basis_order,
            ),
            BemSolutionRepresentation::IndirectSingleLayer => {
                let scattered = compute_scattered_field_gradient_checked(
                    points,
                    &self.elements,
                    &self.nodes,
                    &self.surface_pressure,
                    None,
                    &self.physics,
                    self.basis_order,
                )?;
                let incident = self
                    .incident_field
                    .evaluate_pressure_point_position_derivative(points, &self.physics);
                let mut total = Vec::with_capacity(points.len());
                for (point, (incident, scattered)) in
                    incident.iter().zip(scattered.iter()).enumerate()
                {
                    let mut value = [Complex64::new(0.0, 0.0); 3];
                    for axis in 0..3 {
                        value[axis] = incident[axis] + scattered[axis];
                    }
                    if value
                        .iter()
                        .any(|component| !component.re.is_finite() || !component.im.is_finite())
                    {
                        return Err(FieldEvaluationError::NonFiniteResult { point });
                    }
                    total.push(value);
                }
                Ok(total)
            }
        }
    }

    /// Evaluate the total pressure gradient at one exterior point.
    pub fn evaluate_pressure_gradient_checked(
        &self,
        point: &[f64; 3],
    ) -> Result<[Complex64; 3], FieldEvaluationError> {
        let values = self.evaluate_pressure_gradient_field_checked(std::slice::from_ref(point))?;
        values
            .into_iter()
            .next()
            .ok_or(FieldEvaluationError::NonFiniteResult { point: 0 })
    }

    /// Get max surface pressure magnitude
    pub fn max_surface_pressure(&self) -> f64 {
        self.physical_surface_pressure_values()
            .into_iter()
            .map(|p| p.norm())
            .fold(0.0f64, f64::max)
    }

    /// Get mean surface pressure magnitude
    pub fn mean_surface_pressure(&self) -> f64 {
        let pressure = self.physical_surface_pressure_values();
        let sum: f64 = pressure.iter().map(|p| p.norm()).sum();
        sum / pressure.len().max(1) as f64
    }

    /// Number of DOFs in the solution
    pub fn num_dofs(&self) -> usize {
        self.surface_pressure.len()
    }

    fn physical_surface_pressure_values(&self) -> Vec<Complex64> {
        match self.basis_order {
            BasisOrder::Constant => self
                .elements
                .iter()
                .filter(|e| !e.property.is_evaluation())
                .enumerate()
                .map(|(dof, elem)| match &elem.boundary_condition {
                    BoundaryCondition::Pressure(values) => {
                        values.iter().copied().sum::<Complex64>() / values.len().max(1) as f64
                    }
                    _ => self.surface_pressure[dof],
                })
                .collect(),
            BasisOrder::Linear | BasisOrder::Quadratic => self.surface_pressure.to_vec(),
        }
    }

    /// Export the surface mesh and pressure field to a `.vtu` for ParaView.
    ///
    /// For constant-basis solutions, surface pressure is written as
    /// per-cell (`Pressure_*`) data. Linear-basis solutions store three
    /// discontinuous element-local values per Tri3 panel; export projects
    /// those values onto shared mesh nodes by arithmetic averaging before
    /// writing point data.
    pub fn write_vtu(&self, mesh: &Mesh, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        use sonium_xem::io::vtu::VtuWriter;
        use sonium_xem::pressure_to_spl;

        if self.basis_order == BasisOrder::Quadratic {
            return write_quadratic_surface_pressure_vtu(
                mesh,
                &self.elements,
                &self.surface_pressure,
                path,
            );
        }

        let mut w = VtuWriter::new(mesh);
        match self.basis_order {
            BasisOrder::Constant => {
                let surface_cells: Vec<&Element> = mesh
                    .elements
                    .iter()
                    .filter(|e| !e.property.is_evaluation())
                    .collect();
                let n = surface_cells.len();
                let p = &self.surface_pressure;
                if p.len() != n {
                    return Err(std::io::Error::other(format!(
                        "BemSolution::write_vtu: pressure DOFs ({}) != surface cells ({})",
                        p.len(),
                        n
                    )));
                }
                let mag: Vec<f64> = p.iter().map(|v| v.norm()).collect();
                let real: Vec<f64> = p.iter().map(|v| v.re).collect();
                let imag: Vec<f64> = p.iter().map(|v| v.im).collect();
                let spl: Vec<f64> = p.iter().map(|&v| pressure_to_spl(v)).collect();
                w.add_cell_scalar("Pressure_magnitude", &mag);
                w.add_cell_scalar("Pressure_real", &real);
                w.add_cell_scalar("Pressure_imag", &imag);
                w.add_cell_scalar("Pressure_SPL_dB", &spl);
            }
            BasisOrder::Linear => {
                let n_nodes = mesh.nodes.len();
                if self.nodes.len() != n_nodes {
                    return Err(std::io::Error::other(format!(
                        "BemSolution::write_vtu: solution nodes ({}) != mesh nodes ({n_nodes})",
                        self.nodes.len()
                    )));
                }
                let mut nodal_pressure = vec![Complex64::new(0.0, 0.0); n_nodes];
                let mut contribution_count = vec![0usize; n_nodes];
                for element in self
                    .elements
                    .iter()
                    .filter(|element| !element.property.is_evaluation())
                {
                    if element.connectivity.len() != element.dof_addresses.len() {
                        return Err(std::io::Error::other(format!(
                            "BemSolution::write_vtu: linear element has {} nodes but {} DOFs",
                            element.connectivity.len(),
                            element.dof_addresses.len()
                        )));
                    }
                    for (&node, &dof) in element
                        .connectivity
                        .iter()
                        .zip(element.dof_addresses.iter())
                    {
                        let pressure = self.surface_pressure.get(dof).ok_or_else(|| {
                            std::io::Error::other(format!(
                                "BemSolution::write_vtu: element DOF {dof} is outside pressure vector of length {}",
                                self.surface_pressure.len()
                            ))
                        })?;
                        let node_pressure = nodal_pressure.get_mut(node).ok_or_else(|| {
                            std::io::Error::other(format!(
                                "BemSolution::write_vtu: element node {node} is outside mesh node range 0..{n_nodes}"
                            ))
                        })?;
                        *node_pressure += *pressure;
                        contribution_count[node] += 1;
                    }
                }
                for (node, (pressure, count)) in nodal_pressure
                    .iter_mut()
                    .zip(contribution_count.iter())
                    .enumerate()
                {
                    if *count == 0 {
                        return Err(std::io::Error::other(format!(
                            "BemSolution::write_vtu: mesh node {node} has no linear surface DOF"
                        )));
                    }
                    *pressure /= *count as f64;
                }
                w.add_point_scalar_complex("Pressure", &nodal_pressure);
                let spl: Vec<f64> = nodal_pressure
                    .iter()
                    .map(|&value| pressure_to_spl(value))
                    .collect();
                w.add_point_scalar("Pressure_SPL_dB", &spl);
            }
            BasisOrder::Quadratic => unreachable!("quadratic export handled above"),
        }
        w.write(path)
    }
}

fn validate_xem_pressure_field<'a>(
    problem: &'a sonium_xem::MultiphysicsProblem,
    field_id: &str,
    actual: usize,
) -> Result<&'a sonium_xem::FieldSpec, BemResultError> {
    problem.validate()?;
    let field = match problem.acoustic_complex_pressure_field(field_id) {
        Ok(field) => field,
        Err(sonium_xem::IrValidationError::MissingReference { .. }) => {
            return Err(BemResultError::MissingField(field_id.to_owned()));
        }
        Err(error) => {
            return Err(BemResultError::IncompatibleField {
                field_id: field_id.to_owned(),
                reason: error.to_string(),
            });
        }
    };
    let expected = field.layout.element_count()?;
    if expected != actual {
        return Err(BemResultError::LengthMismatch {
            field_id: field_id.to_owned(),
            expected,
            actual,
        });
    }
    Ok(field)
}

/// Failure while adapting a solved BEM field to XEM.
#[derive(Debug, thiserror::Error)]
pub enum BemResultError {
    /// The source problem or result metadata failed XEM validation.
    #[error("invalid XEM result metadata: {0}")]
    InvalidProblem(#[from] sonium_xem::IrValidationError),
    /// The requested field is absent from the XEM problem.
    #[error("XEM field `{0}` is not declared")]
    MissingField(String),
    /// The declared field cannot represent BEM complex surface pressure.
    #[error("incompatible XEM field `{field_id}`: {reason}")]
    IncompatibleField {
        /// Field identifier supplied by the exchange document.
        field_id: String,
        /// Why the declared scalar/layout is incompatible.
        reason: String,
    },
    /// The declared tensor extent does not match the solved pressure vector.
    #[error("XEM field `{field_id}` expects {expected} values, got {actual}")]
    LengthMismatch {
        /// Field identifier supplied by the exchange document.
        field_id: String,
        /// Number of values declared by the tensor layout.
        expected: usize,
        /// Number of values produced by the BEM solve.
        actual: usize,
    },
}

/// BEM solver errors
#[derive(Debug, Clone)]
pub enum BemError {
    /// Feature not yet implemented
    NotImplemented(String),
    /// Linear solver failed
    SolverFailed(String),
    /// Invalid mesh
    InvalidMesh(String),
    /// Invalid parameters
    InvalidParameters(String),
    /// Exchange convention could not be adapted to the BEM sign contract.
    Convention(crate::core::conventions::BemConventionError),
    /// A checked resource admission rejected the request before allocation.
    ResourceAdmission(crate::core::admission::BemAdmissionError),
    /// The reported Krylov convergence did not survive an unpreconditioned
    /// post-solve application of the assembled operator.
    ResidualCheckFailed {
        /// Assembly/operator path used for the check.
        label: String,
        /// Relative residual measured after the solve.
        residual: f64,
        /// Maximum permitted residual.
        tolerance: f64,
    },
    /// A compressed or fast operator exceeded its independently assembled
    /// reference-action error budget.
    OperatorAccuracyFailed {
        /// Assembly/operator path used for the check.
        label: String,
        /// Diagnostic from the backend-independent contract evaluator.
        reason: String,
    },
}

impl std::fmt::Display for BemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BemError::NotImplemented(msg) => write!(f, "Not implemented: {}", msg),
            BemError::SolverFailed(msg) => write!(f, "Solver failed: {}", msg),
            BemError::InvalidMesh(msg) => write!(f, "Invalid mesh: {}", msg),
            BemError::InvalidParameters(msg) => write!(f, "Invalid parameters: {}", msg),
            BemError::Convention(error) => write!(f, "invalid exchange convention: {error}"),
            BemError::ResourceAdmission(error) => write!(f, "resource admission failed: {error}"),
            BemError::ResidualCheckFailed {
                label,
                residual,
                tolerance,
            } => write!(
                f,
                "exact residual check failed for {label}: {residual:.3e} > {tolerance:.3e}"
            ),
            BemError::OperatorAccuracyFailed { label, reason } => {
                write!(f, "operator-accuracy check failed for {label}: {reason}")
            }
        }
    }
}

impl std::error::Error for BemError {}

/// Result of system assembly (dense matrix or FMM system)
#[derive(Clone)]
enum AssemblyResult {
    /// Dense matrix for TBEM
    Dense(
        sonium_solvers::dense::DenseMatrix<Complex64>,
        Vec<Complex64>,
    ),
    /// SLFMM system for iterative solvers
    Slfmm(SlfmmSystem),
    /// MLFMM system for iterative solvers
    Mlfmm(Box<crate::core::assembly::mlfmm::MlfmmSystem>),
    /// H-matrix with ACA-compressed far-field blocks
    Hmatrix(crate::core::assembly::hmatrix::HMatrix, Vec<Complex64>),
}

/// Caller-owned persistent BEM operator cache.
///
/// The cache stores the assembled numerical operator, never an incident-field
/// RHS.  Reusing it is therefore safe across source/load changes when the
/// [`BemReuseContext`] is unchanged.  A cache miss or rejected reuse decision
/// rebuilds the operator and replaces the previous entry.  The cache is
/// intentionally caller-owned so optimization/design loops can choose its
/// lifetime and memory budget explicitly.
#[derive(Default)]
pub struct BemOperatorCache {
    context: Option<BemReuseContext>,
    assembly: Option<AssemblyResult>,
    indirect_soft: bool,
    configuration: Option<BemCacheConfiguration>,
    last_decision: Option<BemReuseDecision>,
    reuse_hits: u64,
    rebuilds: u64,
}

impl BemOperatorCache {
    /// Construct an empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Remove the assembled operator and all reuse telemetry.
    pub fn clear(&mut self) {
        self.context = None;
        self.assembly = None;
        self.indirect_soft = false;
        self.configuration = None;
        self.last_decision = None;
        self.reuse_hits = 0;
        self.rebuilds = 0;
    }

    /// Whether an assembled operator is currently resident.
    pub fn is_populated(&self) -> bool {
        self.assembly.is_some()
    }

    /// Number of solves that reused the resident operator.
    pub fn reuse_hits(&self) -> u64 {
        self.reuse_hits
    }

    /// Number of solves that rebuilt and replaced the resident operator.
    pub fn rebuilds(&self) -> u64 {
        self.rebuilds
    }

    /// Most recent conservative lineage/configuration decision.
    pub fn last_decision(&self) -> Option<BemReuseDecision> {
        self.last_decision
    }
}

impl AssemblyResult {
    #[allow(dead_code)]
    fn num_dofs(&self) -> usize {
        match self {
            AssemblyResult::Dense(m, _) => m.nrows(),
            AssemblyResult::Slfmm(s) => s.num_dofs,
            AssemblyResult::Mlfmm(s) => s.num_dofs,
            AssemblyResult::Hmatrix(h, _) => h.num_dofs,
        }
    }
}

struct DenseMatrixOperator<'a>(&'a sonium_solvers::dense::DenseMatrix<Complex64>);

impl<'a> LinearOperator<Complex64> for DenseMatrixOperator<'a> {
    fn num_rows(&self) -> usize {
        self.0.nrows()
    }

    fn num_cols(&self) -> usize {
        self.0.ncols()
    }

    fn apply(&self, x: &[Complex64]) -> Vec<Complex64> {
        self.0.matvec(x)
    }

    fn apply_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        self.0.matvec_into(x, y);
    }

    fn apply_block(
        &self,
        x: &sonium_solvers::dense::DenseMatrix<Complex64>,
        y: &mut sonium_solvers::dense::DenseMatrix<Complex64>,
    ) {
        // One row-major GEMM (A n×n · X n×k → Y n×k) instead of k matvecs.
        self.0.matmul_into(x, y);
    }

    fn apply_transpose(&self, x: &[Complex64]) -> Vec<Complex64> {
        let mut y = vec![Complex64::new(0.0, 0.0); self.0.ncols()];
        self.0.matvec_transpose_into(x, &mut y);
        y
    }

    fn apply_transpose_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        self.0.matvec_transpose_into(x, y);
    }

    fn transpose_capability(&self) -> sonium_solvers::traits::AdjointCapability {
        sonium_solvers::traits::AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> sonium_solvers::traits::AdjointCapability {
        sonium_solvers::traits::AdjointCapability::Exact
    }
}

/// Linear-operator view of an exact Hermitian-transposed action.
/// Matrix-free exact constant-panel TBEM reference used by bounded
/// compressed-solve certification. Errors are represented as non-finite
/// actions so the surrounding residual contract fails closed instead of
/// panicking through the `LinearOperator` trait.
struct ExactTbEmOperator<'a> {
    elements: &'a [Element],
    nodes: &'a [[f64; 3]],
    physics: PhysicsParams,
    sign: f64,
    beta_scale: f64,
    num_dofs: usize,
    higher_order: bool,
    curved_normals: bool,
    owner_rows: bool,
    hermitian: bool,
}

impl LinearOperator<Complex64> for ExactTbEmOperator<'_> {
    fn num_rows(&self) -> usize {
        self.num_dofs
    }

    fn num_cols(&self) -> usize {
        self.num_dofs
    }

    fn apply(&self, input: &[Complex64]) -> Vec<Complex64> {
        let action = if self.higher_order {
            if self.owner_rows && self.hermitian {
                let apply = if self.curved_normals {
                    apply_tbem_stabilized_exact_higher_order_owner_hermitian
                } else {
                    apply_tbem_stabilized_exact_higher_order_owner_flat_hermitian
                };
                apply(
                    self.elements,
                    self.nodes,
                    &self.physics,
                    self.sign,
                    self.beta_scale,
                    input,
                )
            } else if self.owner_rows {
                let apply = if self.curved_normals {
                    apply_tbem_stabilized_exact_higher_order_owner
                } else {
                    apply_tbem_stabilized_exact_higher_order_owner_flat
                };
                apply(
                    self.elements,
                    self.nodes,
                    &self.physics,
                    self.sign,
                    self.beta_scale,
                    input,
                )
            } else if self.curved_normals && self.hermitian {
                apply_tbem_stabilized_exact_higher_order_curved_hermitian(
                    self.elements,
                    self.nodes,
                    &self.physics,
                    self.sign,
                    self.beta_scale,
                    input,
                )
            } else if self.curved_normals {
                apply_tbem_stabilized_exact_higher_order_curved(
                    self.elements,
                    self.nodes,
                    &self.physics,
                    self.sign,
                    self.beta_scale,
                    input,
                )
            } else if self.hermitian {
                apply_tbem_stabilized_exact_higher_order_hermitian(
                    self.elements,
                    self.nodes,
                    &self.physics,
                    self.sign,
                    self.beta_scale,
                    input,
                )
            } else {
                apply_tbem_stabilized_exact_higher_order(
                    self.elements,
                    self.nodes,
                    &self.physics,
                    self.sign,
                    self.beta_scale,
                    input,
                )
            }
        } else if self.hermitian {
            apply_tbem_stabilized_exact_hermitian(
                self.elements,
                self.nodes,
                &self.physics,
                self.sign,
                self.beta_scale,
                input,
            )
        } else {
            apply_tbem_stabilized_exact(
                self.elements,
                self.nodes,
                &self.physics,
                self.sign,
                self.beta_scale,
                input,
            )
        };
        action.unwrap_or_else(|_| vec![Complex64::new(f64::NAN, f64::NAN); self.num_dofs])
    }

    fn apply_into(&self, input: &[Complex64], output: &mut [Complex64]) {
        output.copy_from_slice(&self.apply(input));
    }

    fn apply_transpose_into(&self, _input: &[Complex64], output: &mut [Complex64]) {
        output.fill(Complex64::new(f64::NAN, f64::NAN));
    }
}

struct HermitianOperator<O> {
    inner: O,
}

impl<O> LinearOperator<Complex64> for HermitianOperator<O>
where
    O: LinearOperator<Complex64>,
{
    fn num_rows(&self) -> usize {
        self.inner.num_cols()
    }

    fn num_cols(&self) -> usize {
        self.inner.num_rows()
    }

    fn apply_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        y.copy_from_slice(&self.inner.apply_hermitian(x));
    }

    fn apply_transpose_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        let conjugated_input: Vec<_> = x.iter().map(|value| value.conj()).collect();
        let applied = self.inner.apply(&conjugated_input);
        for (output, value) in y.iter_mut().zip(applied) {
            *output = value.conj();
        }
    }

    fn transpose_capability(&self) -> sonium_solvers::traits::AdjointCapability {
        sonium_solvers::traits::AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> sonium_solvers::traits::AdjointCapability {
        sonium_solvers::traits::AdjointCapability::Exact
    }
}

/// Preconditioner view matching a Hermitian-transposed operator.
struct HermitianPreconditioner<P> {
    inner: P,
}

impl<P> Preconditioner<Complex64> for HermitianPreconditioner<P>
where
    P: Preconditioner<Complex64>,
{
    fn apply(&self, x: &[Complex64]) -> Vec<Complex64> {
        self.inner.apply_hermitian(x)
    }

    fn apply_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        self.inner.apply_hermitian_into(x, y);
    }

    fn apply_transpose_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        self.inner.apply_into(x, y);
    }

    fn transpose_capability(&self) -> sonium_solvers::traits::AdjointCapability {
        sonium_solvers::traits::AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> sonium_solvers::traits::AdjointCapability {
        sonium_solvers::traits::AdjointCapability::Exact
    }
}

/// Per-row diagonal of the lumped surface mass matrix.
///
/// Sums each panel's contribution so a node shared by multiple panels
/// accumulates correctly. For constant elements (1 DOF/panel) this is
/// `diag[dof] = area_T`. For linear elements (3 DOFs/triangle) every panel
/// adds `area_T / 3` to each of its three nodes — the standard row-sum
/// (nodal-quadrature) lumping of the consistent surface mass.
fn surface_mass_lumped_diagonal(elements: &[Element]) -> Vec<Complex64> {
    let total_dofs = elements
        .iter()
        .filter(|e| !e.property.is_evaluation())
        .flat_map(|e| e.dof_addresses.iter().copied())
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);
    let mut diag = vec![Complex64::new(0.0, 0.0); total_dofs];
    for elem in elements.iter().filter(|e| !e.property.is_evaluation()) {
        let n_local = elem.dof_addresses.len().max(1) as f64;
        let contrib = Complex64::new(elem.area / n_local, 0.0);
        for &dof in &elem.dof_addresses {
            if dof < total_dofs {
                diag[dof] += contrib;
            }
        }
    }
    // Defensive: any DOF that never received a contribution (would mean a
    // stranded element-less DOF, not expected on well-formed meshes) gets 1.0
    // so the inverse stays finite.
    for entry in diag.iter_mut() {
        if entry.norm() < 1e-30 {
            *entry = Complex64::new(1.0, 0.0);
        }
    }
    diag
}

fn solver_element_coords(elem: &Element, nodes: &[[f64; 3]]) -> DenseMatrix<f64> {
    let mut coords = DenseMatrix::<f64>::zeros(elem.connectivity.len(), 3);
    for (local, &node) in elem.connectivity.iter().enumerate() {
        for dim in 0..3 {
            coords[[local, dim]] = nodes[node][dim];
        }
    }
    coords
}

/// Extract the per-DOF diagonal of an assembled H-matrix from its dense
/// self-blocks. Missing DOFs (not in any self-block) fill with 1.0.
fn hmatrix_diagonal(
    h: &crate::core::assembly::hmatrix::HMatrix,
) -> Result<Vec<Complex64>, BemError> {
    let mut diag: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); h.num_dofs];
    let mut filled = vec![false; h.num_dofs];

    for block in &h.dense_blocks {
        if block.source_cluster != block.field_cluster {
            continue;
        }
        let dofs = h
            .cluster_dof_indices
            .get(block.source_cluster)
            .ok_or_else(|| {
                BemError::SolverFailed(format!(
                    "hmatrix diagonal references missing cluster {}",
                    block.source_cluster
                ))
            })?;
        for (local, &global) in dofs.iter().enumerate() {
            if local >= block.matrix.nrows() || local >= block.matrix.ncols() {
                return Err(BemError::SolverFailed(format!(
                    "hmatrix self-block {} has shape {}x{}, missing diagonal {local}",
                    block.source_cluster,
                    block.matrix.nrows(),
                    block.matrix.ncols()
                )));
            }
            if global >= h.num_dofs {
                return Err(BemError::SolverFailed(format!(
                    "hmatrix cluster {} maps local {local} to global {global}, outside {} DOFs",
                    block.source_cluster, h.num_dofs
                )));
            }
            let value = block.matrix[[local, local]];
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(BemError::SolverFailed(format!(
                    "hmatrix diagonal entry {global} is non-finite"
                )));
            }
            diag[global] = value;
            filled[global] = true;
        }
    }

    let missing = filled.iter().filter(|filled| !**filled).count();
    if missing != 0 {
        return Err(BemError::SolverFailed(format!(
            "hmatrix diagonal is missing {missing}/{} DOFs",
            h.num_dofs
        )));
    }
    Ok(diag)
}

/// Calderón opposite-order preconditioner: `P r = M⁻¹ X (M⁻¹ r)`.
struct CalderonPreconditioner {
    inv_mass_diag: Vec<Complex64>,
    op: DenseMatrix<Complex64>,
}

impl CalderonPreconditioner {
    fn new(mass_diag: &[Complex64], op: DenseMatrix<Complex64>) -> Self {
        let inv_mass_diag: Vec<Complex64> = mass_diag
            .iter()
            .map(|&d| {
                if d.norm() > 1e-20 {
                    Complex64::new(1.0, 0.0) / d
                } else {
                    Complex64::new(1.0, 0.0)
                }
            })
            .collect();
        Self { inv_mass_diag, op }
    }
}

impl Preconditioner<Complex64> for CalderonPreconditioner {
    fn apply(&self, r: &[Complex64]) -> Vec<Complex64> {
        let mut z: Vec<Complex64> = r
            .iter()
            .zip(self.inv_mass_diag.iter())
            .map(|(&ri, &mi)| ri * mi)
            .collect();
        let xz = self.op.matvec(&z);
        for (zi, (&xi, &mi)) in z.iter_mut().zip(xz.iter().zip(self.inv_mass_diag.iter())) {
            *zi = xi * mi;
        }
        z
    }
}

/// Internal dispatch type — keeps `gmres_preconditioned` instantiated on a
/// concrete (Sized) preconditioner without wrapping in a trait object.
enum BemPrecondImpl {
    Identity,
    Diagonal(DiagonalPreconditioner<Complex64>),
    Ilu(sonium_solvers::preconditioners::IluPreconditioner<Complex64>),
    /// Marker used until an FMM near-field matrix is available.
    HierarchicalIlu,
    Hilu(Box<HiluPreconditioner<Complex64>>),
    Calderon(CalderonPreconditioner),
    DualMass(OwnedDualMass),
}

impl Preconditioner<Complex64> for BemPrecondImpl {
    fn apply(&self, r: &[Complex64]) -> Vec<Complex64> {
        match self {
            BemPrecondImpl::Identity => r.to_vec(),
            BemPrecondImpl::Diagonal(d) => d.apply(r),
            BemPrecondImpl::Ilu(d) => d.apply(r),
            BemPrecondImpl::HierarchicalIlu => r.to_vec(),
            BemPrecondImpl::Hilu(d) => d.apply(r),
            BemPrecondImpl::Calderon(c) => c.apply(r),
            BemPrecondImpl::DualMass(d) => d.apply(r),
        }
    }

    fn transpose_capability(&self) -> sonium_solvers::traits::AdjointCapability {
        match self {
            BemPrecondImpl::Identity
            | BemPrecondImpl::Diagonal(_)
            | BemPrecondImpl::Ilu(_)
            | BemPrecondImpl::HierarchicalIlu
            | BemPrecondImpl::Hilu(_)
            | BemPrecondImpl::DualMass(_) => sonium_solvers::traits::AdjointCapability::Exact,
            BemPrecondImpl::Calderon(_) => sonium_solvers::traits::AdjointCapability::Unsupported,
        }
    }

    fn hermitian_capability(&self) -> sonium_solvers::traits::AdjointCapability {
        self.transpose_capability()
    }

    fn apply_transpose_into(&self, r: &[Complex64], y: &mut [Complex64]) {
        match self {
            BemPrecondImpl::Identity => y.copy_from_slice(r),
            BemPrecondImpl::Diagonal(d) => d.apply_transpose_into(r, y),
            BemPrecondImpl::Ilu(d) => d.apply_transpose_into(r, y),
            BemPrecondImpl::HierarchicalIlu => y.copy_from_slice(r),
            BemPrecondImpl::Hilu(d) => d.apply_transpose_into(r, y),
            BemPrecondImpl::DualMass(d) => y.copy_from_slice(&d.apply(r)),
            BemPrecondImpl::Calderon(c) => y.copy_from_slice(&c.apply(r)),
        }
    }

    fn apply_hermitian_into(&self, r: &[Complex64], y: &mut [Complex64]) {
        match self {
            BemPrecondImpl::Identity => y.copy_from_slice(r),
            BemPrecondImpl::Diagonal(d) => d.apply_hermitian_into(r, y),
            BemPrecondImpl::Ilu(d) => d.apply_hermitian_into(r, y),
            BemPrecondImpl::HierarchicalIlu => y.copy_from_slice(r),
            BemPrecondImpl::Hilu(d) => d.apply_hermitian_into(r, y),
            BemPrecondImpl::DualMass(d) => y.copy_from_slice(&d.apply(r)),
            BemPrecondImpl::Calderon(c) => y.copy_from_slice(&c.apply(r)),
        }
    }
}

/// Owned dual-mass-on-barycentric preconditioner state.
///
/// Holds the assembled refined-mesh P1 mass matrix and the centroid
/// offset needed to lift / restrict between primal-P0 (BEM DOFs) and
/// refined-P1 (centroid sample). Real-valued by construction —
/// `apply()` splits the `Complex64` residual into real and imaginary
/// halves, applies the real preconditioner to each (the L²-mass
/// matrix commutes with the imaginary unit), and recombines.
struct OwnedDualMass {
    mass: sonium_solvers::sparse::CsrMatrix<f64>,
    centroid_offset: usize,
    n_refined: usize,
    n_primal: usize,
    inner_cg_max_iter: usize,
    inner_cg_tol: f64,
}

impl OwnedDualMass {
    fn apply_real(&self, r: &[f64]) -> Vec<f64> {
        assert_eq!(
            r.len(),
            self.n_primal,
            "SurfaceMassDualBarycentric residual length must match active constant-panel DOFs"
        );
        // Lift primal-P0 → refined-P1 (centroid sample, zeros elsewhere).
        let mut lifted = vec![0.0_f64; self.n_refined];
        for k in 0..self.n_primal {
            lifted[self.centroid_offset + k] = r[k];
        }
        // Solve M_refined · y = lifted via inner CG.
        let cfg = sonium_solvers::iterative::CgConfig::<f64> {
            max_iterations: self.inner_cg_max_iter,
            tolerance: self.inner_cg_tol,
            print_interval: 0,
        };
        let sol = sonium_solvers::iterative::cg(&self.mass, &lifted, &cfg);
        // Restrict refined-P1 → primal-P0 (centroid sample of solution).
        let mut out = vec![0.0_f64; self.n_primal];
        for (k, out_k) in out.iter_mut().enumerate().take(self.n_primal) {
            *out_k = sol.x[self.centroid_offset + k];
        }
        out
    }
}

impl Preconditioner<Complex64> for OwnedDualMass {
    fn apply(&self, r: &[Complex64]) -> Vec<Complex64> {
        assert_eq!(
            r.len(),
            self.n_primal,
            "SurfaceMassDualBarycentric residual length must match active constant-panel DOFs"
        );
        // The L² mass matrix is real and symmetric. For a complex
        // residual `r = re + i·im`, the application is
        // `M⁻¹ r = M⁻¹ re + i · M⁻¹ im` — two independent real solves.
        let re: Vec<f64> = r.iter().map(|c| c.re).collect();
        let im: Vec<f64> = r.iter().map(|c| c.im).collect();
        let y_re = self.apply_real(&re);
        let y_im = self.apply_real(&im);
        let mut out = vec![Complex64::new(0.0, 0.0); r.len()];
        for k in 0..r.len() {
            out[k] = Complex64::new(y_re[k], y_im[k]);
        }
        out
    }
}

impl BemSolver {
    /// Build the per-solve preconditioner instance from the configured kind
    /// and the assembled state.
    fn build_preconditioner(
        &self,
        elements: &[Element],
        nodes: &[[f64; 3]],
        physics: &PhysicsParams,
        dense_diag: Option<&[Complex64]>,
        h_diag: Option<&[Complex64]>,
    ) -> Result<BemPrecondImpl, BemError> {
        let preconditioner = match self.preconditioner {
            BemPreconditioner::None => BemPrecondImpl::Identity,
            BemPreconditioner::Jacobi => {
                let diag_owned: Vec<Complex64>;
                let diag: &[Complex64] = if let Some(d) = dense_diag {
                    d
                } else if let Some(d) = h_diag {
                    d
                } else {
                    log::warn!("Jacobi requested but no diagonal available; using identity");
                    // The assembled operator is indexed by global DOF
                    // addresses.  Summing local incidences over-counts
                    // continuous/shared higher-order traces (for example,
                    // six Quad4 panels expose 54 local slots but only 26
                    // global quadratic DOFs), which leaves Jacobi with an
                    // incompatible preconditioner dimension.
                    let n_dofs = count_dofs(elements);
                    diag_owned = vec![Complex64::new(1.0, 0.0); n_dofs];
                    &diag_owned
                };
                BemPrecondImpl::Diagonal(DiagonalPreconditioner::from_diagonal(diag))
            }
            BemPreconditioner::SurfaceMassLumped => {
                let diag = surface_mass_lumped_diagonal(elements);
                BemPrecondImpl::Diagonal(DiagonalPreconditioner::from_diagonal(&diag))
            }
            BemPreconditioner::CalderonV => {
                let mass = surface_mass_lumped_diagonal(elements);
                let v = crate::core::assembly::operators::assemble_single_layer(
                    elements, nodes, physics,
                );
                BemPrecondImpl::Calderon(CalderonPreconditioner::new(&mass, v))
            }
            BemPreconditioner::CalderonW => {
                let mass = surface_mass_lumped_diagonal(elements);
                let w = crate::core::assembly::operators::assemble_hypersingular(
                    elements, nodes, physics,
                );
                BemPrecondImpl::Calderon(CalderonPreconditioner::new(&mass, w))
            }
            // FMM near-field extraction is deferred until the assembled
            // operator is available in the FMM solve path. Reaching this
            // builder means the selected assembly is not an FMM path.
            BemPreconditioner::HierarchicalIlu => {
                return Err(BemError::NotImplemented(
                    "HierarchicalIlu requires SLFMM or MLFMM assembly".to_string(),
                ));
            }
            BemPreconditioner::SurfaceMassDualBarycentric => {
                use crate::core::assembly::assemble_p1_surface_mass;
                use crate::core::mesh::refine_from_elements;
                validate_dual_mass_barycentric_supported(elements)?;
                let refined = refine_from_elements(elements, nodes);
                let centroid_offset = refined.n_orig_nodes + refined.n_edge_nodes;
                let n_refined = refined.n_refined_nodes();
                let n_primal = refined.primal_to_children.len();
                let mass = assemble_p1_surface_mass(&refined);
                BemPrecondImpl::DualMass(OwnedDualMass {
                    mass,
                    centroid_offset,
                    n_refined,
                    n_primal,
                    inner_cg_max_iter: 50,
                    inner_cg_tol: 1e-8,
                })
            }
        };
        Ok(preconditioner)
    }

    /// Run GMRES on a generic `LinearOperator` with the supplied preconditioner.
    fn run_gmres<O: LinearOperator<Complex64>>(
        &self,
        operator: &O,
        precond: &BemPrecondImpl,
        rhs: &[Complex64],
        label: &str,
    ) -> Result<Vec<Complex64>, BemError> {
        let config = GmresConfig::<f64> {
            max_iterations: self.max_iterations,
            restart: self.gmres_restart,
            tolerance: self.tolerance,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 0,
            min_restart: None,
            max_restart: None,
        };
        let sol = match precond {
            BemPrecondImpl::Identity => gmres(operator, rhs, &config),
            _ => gmres_preconditioned(operator, precond, rhs, &config),
        };
        if self.verbose {
            log::info!(
                "GMRES ({label}): converged={}, iters={}, residual={:.3e}",
                sol.converged,
                sol.iterations,
                sol.residual
            );
        }
        if sol.converged {
            self.check_exact_residual(operator, &sol.x, rhs, label)?;
            Ok(sol.x)
        } else {
            Err(BemError::SolverFailed(format!(
                "GMRES ({label}) did not converge: iters={}, residual={:.3e}",
                sol.iterations, sol.residual
            )))
        }
    }

    fn check_exact_residual<O: LinearOperator<Complex64>>(
        &self,
        operator: &O,
        solution: &[Complex64],
        rhs: &[Complex64],
        label: &str,
    ) -> Result<(), BemError> {
        let residual = relative_operator_residual(operator, solution, rhs)?;
        let residual_tolerance = (self.tolerance * 20.0).max(1e-10);
        if residual > residual_tolerance {
            return Err(BemError::ResidualCheckFailed {
                label: label.to_string(),
                residual,
                tolerance: residual_tolerance,
            });
        }
        Ok(())
    }
}

fn relative_operator_residual<O: LinearOperator<Complex64>>(
    operator: &O,
    solution: &[Complex64],
    rhs: &[Complex64],
) -> Result<f64, BemError> {
    if solution.len() != rhs.len() {
        return Err(BemError::SolverFailed(format!(
            "solution has {} entries for a {}-entry RHS",
            solution.len(),
            rhs.len()
        )));
    }
    let applied = operator.apply(solution);
    if applied.len() != rhs.len() {
        return Err(BemError::SolverFailed(format!(
            "operator returned {} entries for a {}-entry RHS",
            applied.len(),
            rhs.len()
        )));
    }
    let residual_squared: f64 = applied
        .iter()
        .zip(rhs)
        .map(|(lhs, rhs)| (*lhs - *rhs).norm_sqr())
        .sum();
    let rhs_squared: f64 = rhs.iter().map(|value| value.norm_sqr()).sum();
    if !residual_squared.is_finite() || !rhs_squared.is_finite() {
        return Err(BemError::SolverFailed(
            "exact operator residual is non-finite".to_string(),
        ));
    }
    // Keep the residual scale-relative even for small physical RHS values.
    // The frequency-sweep checker uses the same floor; a unit floor here
    // would turn a relative contract into an absolute one whenever
    // `||b|| < 1`, allowing poorly scaled solves to pass silently.
    Ok(residual_squared.sqrt() / rhs_squared.sqrt().max(f64::MIN_POSITIVE))
}

fn exact_block_residuals<O: LinearOperator<Complex64>>(
    operator: &O,
    solution: &DenseMatrix<Complex64>,
    rhs_columns: &[Vec<Complex64>],
) -> Result<Vec<f64>, BemError> {
    if solution.ncols() != rhs_columns.len() {
        return Err(BemError::SolverFailed(format!(
            "solution has {} columns for {} RHS columns",
            solution.ncols(),
            rhs_columns.len()
        )));
    }
    rhs_columns
        .iter()
        .enumerate()
        .map(|(rhs_index, rhs)| {
            relative_operator_residual(operator, &solution.column_to_vec(rhs_index), rhs)
        })
        .collect()
}

fn validate_dual_mass_barycentric_supported(elements: &[Element]) -> Result<(), BemError> {
    for (idx, elem) in elements.iter().enumerate() {
        if elem.property.is_evaluation() {
            continue;
        }
        if !matches!(elem.element_type, ElementType::Tri3) {
            return Err(BemError::InvalidParameters(format!(
                "SurfaceMassDualBarycentric supports only active Tri3 panels; element {idx} has {:?}",
                elem.element_type
            )));
        }
        if elem.dof_addresses.len() != 1 {
            return Err(BemError::InvalidParameters(format!(
                "SurfaceMassDualBarycentric supports only constant P0 panels; element {idx} carries {} DOFs",
                elem.dof_addresses.len()
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::core::types::ElementProperty;

    #[test]
    fn continuous_curved_quad4_hmatrix_proxies_carry_collocation_normals() {
        let nodes = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.4],
            [0.0, 1.0, 0.0],
        ];
        let element = Element {
            connectivity: vec![0, 1, 2, 3],
            element_type: ElementType::Quad4,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0; 3]; 4],
            center: [0.5, 0.5, 0.1],
            area: 1.0,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
            group: 0,
            dof_addresses: vec![0, 1, 2, 3],
        };
        let expected = element.collocation_normals(BasisOrder::Linear, &nodes);
        let (proxies, _) = build_continuous_hmatrix_proxy_elements(
            std::slice::from_ref(&element),
            &nodes,
            BasisOrder::Linear,
        );
        assert_eq!(proxies.len(), 4);
        assert!((expected[0][2] - expected[1][2]).abs() > 1.0e-6);
        for (proxy, normal) in proxies.iter().zip(expected) {
            let error = (0..3)
                .map(|axis| (proxy.normal[axis] - normal[axis]).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                error < 1.0e-12,
                "curved continuous H-matrix proxy normal error={error}"
            );
        }
    }

    #[test]
    fn higher_order_compressed_wavenumber_objective_gradient_matches_central_difference() {
        let problem = BemProblem {
            mesh: crate::core::mesh::generate_cubed_sphere_quad_mesh(0.1, 1),
            physics: PhysicsParams::new(120.0, 343.0, 1.21, false),
            incident_field: IncidentField::plane_wave_z(),
            bc_type: BoundaryConditionType::Rigid,
            use_burton_miller: true,
        };
        let solver = BemSolver::new()
            .with_basis_order(BasisOrder::Quadratic)
            .with_basis_continuity(BasisContinuity::Continuous)
            .with_assembly_method(AssemblyMethod::Hmatrix)
            .with_solver_method(SolverMethod::Gmres);
        let elements = solver
            .prepare_elements(&problem)
            .expect("prepared quadratic H-matrix elements");
        let dimension = count_dofs(&elements);
        let rhs = (0..dimension)
            .map(|index| Complex64::new(0.7 + index as f64 / 17.0, -0.2 + index as f64 / 23.0))
            .collect::<Vec<_>>();
        let objective_gradient = (0..dimension)
            .map(|index| Complex64::new(0.3 - index as f64 / 19.0, 0.05 + index as f64 / 29.0))
            .collect::<Vec<_>>();

        let analytic = solver
            .solve_wavenumber_objective_gradient(&problem, &rhs, &objective_gradient)
            .expect("quadratic compressed wavenumber objective gradient");
        let objective_at = |wavenumber: f64| {
            let mut candidate = problem.clone();
            let frequency = wavenumber * problem.physics.speed_of_sound / (2.0 * PI);
            let mut physics = PhysicsParams::try_new(
                frequency,
                problem.physics.speed_of_sound,
                problem.physics.density,
                problem.physics.tau < 0.0,
            )
            .expect("finite perturbed physics");
            physics.harmonic_factor = problem.physics.harmonic_factor;
            physics.pressure_factor = physics.density * physics.omega * physics.harmonic_factor;
            candidate.physics = physics;
            let solution = solver
                .solve_rhs_many(&candidate, &[rhs.clone()])
                .expect("quadratic compressed perturbed solve")
                .pop()
                .expect("quadratic compressed perturbed solution");
            objective_gradient
                .iter()
                .zip(solution.coefficients())
                .map(|(gradient, coefficient)| (gradient.conj() * coefficient).re)
                .sum::<f64>()
        };
        let step = 1.0e-4;
        let finite_difference = (objective_at(problem.physics.wave_number + step)
            - objective_at(problem.physics.wave_number - step))
            / (2.0 * step);
        let scale = analytic.abs().max(finite_difference.abs()).max(1.0);
        assert!(
            (analytic - finite_difference).abs() / scale < 5.0e-3,
            "quadratic compressed wavenumber gradient: analytic={analytic}, finite_difference={finite_difference}"
        );
        let speed = solver
            .solve_speed_of_sound_objective_gradient(&problem, &rhs, &objective_gradient)
            .expect("quadratic compressed speed objective gradient");
        let expected_speed =
            -problem.physics.wave_number / problem.physics.speed_of_sound * analytic;
        let speed_scale = expected_speed.abs().max(speed.abs()).max(1.0);
        assert!(
            (speed - expected_speed).abs() / speed_scale < 1.0e-12,
            "quadratic compressed speed chain rule: speed={speed}, expected={expected_speed}"
        );
        let density = solver
            .solve_density_objective_gradient(&problem, &rhs, &objective_gradient)
            .expect("quadratic compressed density objective gradient");
        assert!(speed.is_finite());
        assert!(density.is_finite());
    }

    use super::*;
    use crate::core::contracts::{
        check_adjoint_contract, check_block_rhs_contract, require_operator_accuracy,
    };

    #[test]
    fn test_bem_problem_creation() {
        let problem = BemProblem::rigid_sphere_scattering(0.1, 1000.0, 343.0, 1.21);

        assert!(!problem.mesh.elements.is_empty());
        assert!(problem.mesh.nodes.len() > 0);
        assert!(problem.ka() > 0.0);
    }

    #[test]
    fn hmatrix_diagonal_rejects_malformed_self_block_without_panicking() {
        let malformed = crate::core::assembly::hmatrix::HMatrix {
            num_dofs: 1,
            dense_blocks: vec![crate::core::assembly::hmatrix::DenseBlock {
                source_cluster: 0,
                field_cluster: 0,
                matrix: DenseMatrix::zeros(0, 0),
            }],
            low_rank_blocks: Vec::new(),
            cluster_dof_indices: vec![vec![0]],
            partitions: Vec::new(),
            transpose_dense_order: Vec::new(),
            transpose_low_rank_order: Vec::new(),
            transpose_partitions: Vec::new(),
            memory_bytes: 0,
            compression_ratio: 1.0,
        };

        assert!(matches!(
            hmatrix_diagonal(&malformed),
            Err(BemError::SolverFailed(_))
        ));
    }

    #[test]
    fn checked_sphere_constructors_reject_invalid_geometry_and_medium() {
        assert!(matches!(
            BemProblem::try_rigid_sphere_scattering(0.0, 1000.0, 343.0, 1.21),
            Err(BemError::InvalidParameters(message)) if message.contains("radius")
        ));
        assert!(matches!(
            BemProblem::try_rigid_sphere_scattering(0.1, f64::NAN, 343.0, 1.21),
            Err(BemError::InvalidParameters(message)) if message.contains("frequency")
        ));
        assert!(matches!(
            BemProblem::try_rigid_sphere_scattering_custom(0.1, 1000.0, 343.0, 1.21, 2, 8),
            Err(BemError::InvalidParameters(message)) if message.contains("angular")
        ));
        let checked =
            BemProblem::try_rigid_sphere_scattering_custom(0.1, 1000.0, 343.0, 1.21, 4, 8)
                .expect("valid checked sphere setup");
        assert!(checked.validate().is_ok());
    }

    #[test]
    fn bem_problem_admission_rejects_inconsistent_panel_geometry() {
        let base = BemProblem::rigid_sphere_scattering_custom(0.1, 1_000.0, 343.0, 1.21, 4, 8);
        assert!(base.validate().is_ok());

        let mut bad_center = base.clone();
        bad_center.mesh.elements[0].center[0] += 0.01;
        assert!(
            matches!(bad_center.validate(), Err(BemError::InvalidMesh(message)) if message.contains("center"))
        );

        let mut bad_normal = base.clone();
        bad_normal.mesh.elements[0].normal = [0.0, 0.0, 0.5];
        assert!(
            matches!(bad_normal.validate(), Err(BemError::InvalidMesh(message)) if message.contains("normal"))
        );

        let mut bad_area = base;
        let triangle = bad_area
            .mesh
            .elements
            .iter_mut()
            .find(|element| element.element_type == ElementType::Tri3)
            .expect("sphere fixture contains triangles");
        triangle.area *= 1.5;
        assert!(
            matches!(bad_area.validate(), Err(BemError::InvalidMesh(message)) if message.contains("area"))
        );

        let mut overflowing_center =
            BemProblem::rigid_sphere_scattering_custom(0.1, 1_000.0, 343.0, 1.21, 4, 8);
        for &node in &overflowing_center.mesh.elements[0].connectivity {
            overflowing_center.mesh.nodes[node] = [f64::MAX; 3];
        }
        assert!(
            matches!(overflowing_center.validate(), Err(BemError::InvalidMesh(message)) if message.contains("non-finite center"))
        );

        let mut quad = BemProblem::rigid_sphere_scattering_custom(0.1, 1_000.0, 343.0, 1.21, 4, 8);
        let connectivity = vec![0, 1, 2, 3];
        let center = compute_element_center(&quad.mesh.nodes, &connectivity);
        let normal = compute_element_normal(&quad.mesh.nodes, &connectivity, ElementType::Quad4);
        let area = compute_element_area(&quad.mesh.nodes, &connectivity, ElementType::Quad4);
        let panel = &mut quad.mesh.elements[0];
        panel.element_type = ElementType::Quad4;
        panel.connectivity = connectivity;
        panel.center = center;
        panel.normal = normal;
        panel.area = area;
        assert!(quad.validate().is_ok());
        quad.mesh.elements[0].area *= 1.5;
        assert!(
            matches!(quad.validate(), Err(BemError::InvalidMesh(message)) if message.contains("area"))
        );
    }

    #[test]
    fn xem_admission_propagates_convention_and_rejects_field_contract_errors() {
        let mut exchange = sonium_xem::MultiphysicsProblem {
            id: "bem-admission".into(),
            domains: vec![sonium_xem::Domain {
                id: "acoustics".into(),
                label: "Acoustics".into(),
                kind: sonium_xem::DomainKind::Acoustics,
            }],
            fields: vec![sonium_xem::FieldSpec {
                id: "pressure".into(),
                name: "surface pressure".into(),
                domain_id: "acoustics".into(),
                kind: sonium_xem::FieldKind::ComplexScalar,
                units: Some("Pa".into()),
                layout: sonium_xem::TensorLayout {
                    scalar_type: sonium_xem::ScalarType::ComplexF64,
                    shape: vec![1],
                    strides: vec![1],
                    ..Default::default()
                },
            }],
            ..Default::default()
        };
        exchange.conventions.harmonic_time = sonium_xem::HarmonicConvention::ExpPlusIwt;
        let base = BemProblem::rigid_sphere_scattering_custom(0.1, 1000.0, 343.0, 1.21, 4, 8);
        let admitted = BemProblem::from_xem(
            &exchange,
            "pressure",
            base.mesh.clone(),
            base.physics.clone(),
            base.incident_field.clone(),
            base.bc_type,
            base.use_burton_miller,
        )
        .expect("valid XEM metadata and backend inputs should be admitted");
        assert_eq!(admitted.physics.harmonic_factor, -1.0);

        exchange.fields[0].units = Some("m/s".into());
        assert!(matches!(
            BemProblem::from_xem(
                &exchange,
                "pressure",
                base.mesh.clone(),
                base.physics.clone(),
                base.incident_field.clone(),
                base.bc_type,
                base.use_burton_miller,
            ),
            Err(crate::core::conventions::BemProblemError::Field { .. })
        ));

        exchange.fields[0].units = Some("Pa".into());
        exchange.fields.push(sonium_xem::FieldSpec {
            id: "temperature".into(),
            name: "temperature".into(),
            domain_id: "acoustics".into(),
            kind: sonium_xem::FieldKind::Scalar,
            units: Some("K".into()),
            layout: sonium_xem::TensorLayout {
                shape: vec![1],
                strides: vec![1],
                ..Default::default()
            },
        });
        exchange.couplings = vec![
            sonium_xem::Coupling {
                id: "pressure-temperature".into(),
                source_field_id: "pressure".into(),
                target_field_id: "temperature".into(),
                kind: sonium_xem::CouplingKind::Partitioned,
                conservation: Some("energy".into()),
            },
            sonium_xem::Coupling {
                id: "temperature-pressure".into(),
                source_field_id: "temperature".into(),
                target_field_id: "pressure".into(),
                kind: sonium_xem::CouplingKind::Partitioned,
                conservation: Some("energy".into()),
            },
        ];
        assert!(matches!(
            BemProblem::from_xem(
                &exchange,
                "pressure",
                base.mesh.clone(),
                base.physics.clone(),
                base.incident_field.clone(),
                base.bc_type,
                base.use_burton_miller,
            ),
            Err(crate::core::conventions::BemProblemError::CouplingGraph(
                sonium_xem::CouplingGraphError::Cycle(_)
            ))
        ));

        let mut invalid_mesh = base.mesh;
        invalid_mesh.elements.clear();
        assert!(matches!(
            BemProblem::from_xem(
                &sonium_xem::MultiphysicsProblem {
                    fields: vec![sonium_xem::FieldSpec {
                        units: Some("Pa".into()),
                        ..exchange.fields[0].clone()
                    }],
                    couplings: vec![],
                    ..exchange
                },
                "pressure",
                invalid_mesh,
                PhysicsParams::new(1000.0, 343.0, 1.21, false),
                IncidentField::plane_wave_z(),
                BoundaryConditionType::Rigid,
                true,
            ),
            Err(crate::core::conventions::BemProblemError::Backend(
                BemError::InvalidMesh(_)
            ))
        ));
    }

    #[test]
    fn solved_surface_pressure_adapts_to_xem_without_copying() {
        let exchange = sonium_xem::MultiphysicsProblem {
            id: "bem-case".into(),
            domains: vec![sonium_xem::Domain {
                id: "acoustics".into(),
                label: "Acoustics".into(),
                kind: sonium_xem::DomainKind::Acoustics,
            }],
            fields: vec![sonium_xem::FieldSpec {
                id: "pressure".into(),
                name: "surface pressure".into(),
                domain_id: "acoustics".into(),
                kind: sonium_xem::FieldKind::ComplexScalar,
                units: Some("Pa".into()),
                layout: sonium_xem::TensorLayout {
                    scalar_type: sonium_xem::ScalarType::ComplexF64,
                    shape: vec![2],
                    strides: vec![1],
                    ..Default::default()
                },
            }],
            ..Default::default()
        };
        let solution = BemSolution {
            surface_pressure: vec![Complex64::new(1.0, 2.0), Complex64::new(-1.0, 0.5)],
            elements: Vec::new(),
            nodes: Vec::new(),
            incident_field: IncidentField::plane_wave_z(),
            physics: PhysicsParams::new(1000.0, 343.0, 1.21, false),
            basis_order: BasisOrder::Constant,
            representation: BemSolutionRepresentation::DirectCauchy,
        };
        let result = solution
            .to_xem_result(&exchange, "pressure")
            .expect("surface field metadata should adapt");
        assert_eq!(
            result.fields[0].data.byte_length,
            2 * std::mem::size_of::<Complex64>() as u64
        );
        let view = solution
            .borrowed_xem_field(&exchange, "pressure")
            .expect("surface values should be borrowed");
        assert_eq!(view.values().as_ptr(), solution.surface_pressure.as_ptr());
    }

    #[test]
    fn exchange_convention_is_applied_before_bem_entry() {
        let mut registry = sonium_xem::ConventionRegistry::default();
        registry.harmonic_time = sonium_xem::HarmonicConvention::ExpPlusIwt;
        let problem = BemProblem::rigid_sphere_scattering(0.1, 1000.0, 343.0, 1.21)
            .with_exchange_convention(&registry)
            .expect("the explicit exchange convention is valid");

        assert_eq!(problem.physics.harmonic_factor, -1.0);
        assert!(matches!(
            problem.with_exchange_convention(&sonium_xem::ConventionRegistry {
                outward_normals: false,
                ..registry
            }),
            Err(BemError::Convention(_))
        ));
    }

    #[test]
    fn test_bem_problem_validation_rejects_nonfinite_mesh() {
        let mut problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
        problem.mesh.nodes[0][0] = f64::NAN;
        assert!(matches!(
            problem.validate(),
            Err(BemError::InvalidMesh(message)) if message.contains("non-finite")
        ));

        let mut duplicate =
            BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
        duplicate.mesh.elements[0].connectivity[1] = duplicate.mesh.elements[0].connectivity[0];
        assert!(matches!(
            duplicate.validate(),
            Err(BemError::InvalidMesh(message)) if message.contains("repeats connectivity node")
        ));

        let error = BemSolver::new()
            .solve_frequency_sweep(
                &BemProblem::rigid_sphere_scattering(0.1, 100.0, 343.0, 1.21),
                &[],
            )
            .expect_err("empty frequency sweeps must be rejected");
        assert!(matches!(error, BemError::InvalidParameters(_)));
    }

    #[test]
    fn bem_problem_validation_rejects_tampered_physics() {
        let mut problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
        problem.physics.wave_length = f64::NAN;
        assert!(matches!(
            problem.validate(),
            Err(BemError::InvalidParameters(message)) if message.contains("wave_length")
        ));
    }

    #[test]
    fn test_bem_solver_creation() {
        let solver = BemSolver::new()
            .with_solver_method(SolverMethod::Direct)
            .with_assembly_method(AssemblyMethod::Tbem)
            .with_verbose(false);

        assert_eq!(solver.solver_method, SolverMethod::Direct);
        assert_eq!(solver.assembly_method, AssemblyMethod::Tbem);
    }

    #[test]
    fn resource_budget_rejects_before_operator_assembly() {
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
        let solver = BemSolver::new().with_resource_budget(BemResourceBudget::new(1, 0, 1));
        let error = solver
            .solve(&problem)
            .expect_err("one byte cannot admit a BEM operator");
        assert!(matches!(
            error,
            BemError::ResourceAdmission(crate::core::admission::BemAdmissionError::LimitExceeded {
                resource: crate::core::admission::BemResource::HostMemory,
                ..
            })
        ));
    }

    #[test]
    fn resource_budget_counts_multi_rhs_work() {
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
        let solver =
            BemSolver::new().with_resource_budget(BemResourceBudget::new(usize::MAX, 0, 1));
        let fields = [
            problem.incident_field.clone(),
            problem.incident_field.clone(),
        ];
        let error = solver
            .solve_many(&problem, &fields)
            .expect_err("two RHS must exceed one active work slot");
        assert!(matches!(
            error,
            BemError::ResourceAdmission(crate::core::admission::BemAdmissionError::LimitExceeded {
                resource: crate::core::admission::BemResource::ActiveWork,
                requested: 2,
                limit: 1,
            })
        ));
    }

    #[test]
    fn exact_residual_rejects_malformed_solution_and_nonfinite_operator() {
        let mut matrix = DenseMatrix::<Complex64>::zeros(1, 1);
        matrix[[0, 0]] = Complex64::new(1.0, 0.0);
        let rhs = [Complex64::new(1.0, 0.0)];

        let length_error = relative_operator_residual(&DenseMatrixOperator(&matrix), &[], &rhs)
            .expect_err("a malformed solution must be rejected before matvec");
        assert!(
            matches!(length_error, BemError::SolverFailed(message) if message.contains("solution has 0 entries"))
        );

        matrix[[0, 0]] = Complex64::new(f64::NAN, 0.0);
        let nonfinite_error = relative_operator_residual(
            &DenseMatrixOperator(&matrix),
            &[Complex64::new(1.0, 0.0)],
            &rhs,
        )
        .expect_err("a non-finite exact residual must be rejected");
        assert!(
            matches!(nonfinite_error, BemError::SolverFailed(message) if message.contains("non-finite"))
        );
    }

    #[test]
    fn exact_residual_preserves_small_rhs_relative_scale() {
        let mut matrix = DenseMatrix::<Complex64>::zeros(1, 1);
        matrix[[0, 0]] = Complex64::new(1.0, 0.0);
        let rhs = [Complex64::new(1.0e-12, 0.0)];
        let solution = [Complex64::new(2.0e-12, 0.0)];

        let residual = relative_operator_residual(&DenseMatrixOperator(&matrix), &solution, &rhs)
            .expect("finite dimensions and values");
        assert!((residual - 1.0).abs() < 1.0e-12, "residual={residual}");
    }

    #[test]
    fn test_hmatrix_accuracy_controls_are_validated_and_copied_for_retries() {
        let solver = BemSolver::new()
            .with_hmatrix_accuracy(1e-8, 12)
            .with_compressed_retries(3);
        assert_eq!(solver.hmatrix_aca_tolerance, 1e-8);
        assert_eq!(solver.hmatrix_max_rank, 12);
        assert_eq!(solver.compressed_retry_count, 3);
        assert!(solver.validate().is_ok());

        let invalid = BemSolver::new().with_hmatrix_accuracy(f64::NAN, 0);
        assert!(matches!(
            invalid.validate(),
            Err(BemError::InvalidParameters(message)) if message.contains("hmatrix_aca_tolerance")
        ));
    }

    #[test]
    fn test_bem_solver_scalable_single_node_is_explicit() {
        let solver = BemSolver::scalable_single_node();
        assert_eq!(solver.solver_method, SolverMethod::Gmres);
        assert_eq!(solver.assembly_method, AssemblyMethod::Auto);
        assert!(solver.frequency_recycle_subspace_size > 0);
    }

    #[test]
    fn test_hmatrix_solve_matches_dense_tbem_on_small_mie() {
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
        let ka = problem.ka();
        let sign = PhysicsParams::optimal_formulation_sign(ka);
        let beta_scale = PhysicsParams::optimal_beta_scale(ka);
        let elements = BemSolver::new().prepare_elements(&problem).unwrap();
        let dense_system = BemSolver::new()
            .with_assembly_method(AssemblyMethod::Tbem)
            .assemble_system_stabilized(
                &elements,
                &problem.mesh.nodes,
                &problem.physics,
                sign,
                beta_scale,
            )
            .expect("dense assembly");
        let hmatrix_system = BemSolver::new()
            .with_assembly_method(AssemblyMethod::Hmatrix)
            .assemble_system_stabilized(
                &elements,
                &problem.mesh.nodes,
                &problem.physics,
                sign,
                beta_scale,
            )
            .expect("H-matrix assembly");
        let x: Vec<Complex64> = (0..elements.len())
            .map(|i| Complex64::new((0.13 * i as f64).sin(), (0.07 * i as f64).cos()))
            .collect();
        if let (AssemblyResult::Dense(matrix, _), AssemblyResult::Hmatrix(hmatrix, _)) =
            (&dense_system, &hmatrix_system)
        {
            let y: Vec<Complex64> = (0..elements.len())
                .map(|i| Complex64::new((0.19 * i as f64).cos(), (0.11 * i as f64).sin()))
                .collect();
            let z: Vec<Complex64> = (0..elements.len())
                .map(|i| Complex64::new((0.23 * i as f64).sin(), (0.17 * i as f64).cos()))
                .collect();
            let dense_operator = DenseMatrixOperator(matrix);

            let dense_adjoint = check_adjoint_contract(&dense_operator, &x, &y)
                .expect("assembled dense operator has valid adjoint dimensions");
            assert!(
                dense_adjoint.relative_error < 1e-12,
                "dense TBEM Hermitian contract failed: {dense_adjoint:?}"
            );
            let hmatrix_adjoint = check_adjoint_contract(hmatrix, &x, &y)
                .expect("assembled H-matrix has valid adjoint dimensions");
            assert!(
                hmatrix_adjoint.relative_error < 1e-12,
                "H-matrix Hermitian contract failed: {hmatrix_adjoint:?}"
            );

            let dense_block_error =
                check_block_rhs_contract(&dense_operator, &[x.clone(), y.clone(), z.clone()])
                    .expect("dense block dimensions are valid");
            assert!(
                dense_block_error < 1e-12,
                "dense TBEM block-RHS contract failed: {dense_block_error:.3e}"
            );
            let hmatrix_block_error =
                check_block_rhs_contract(hmatrix, &[x.clone(), y.clone(), z.clone()])
                    .expect("H-matrix block dimensions are valid");
            assert!(
                hmatrix_block_error < 1e-12,
                "H-matrix block-RHS contract failed: {hmatrix_block_error:.3e}"
            );

            let accuracy = require_operator_accuracy(
                hmatrix,
                &dense_operator,
                &[x.clone(), y.clone(), z.clone()],
                1e-6,
            )
            .expect("the H-matrix must stay within the small-fixture dense error budget");
            assert_eq!(accuracy.probes, 3);

            let y_dense = matrix.matvec(&x);
            let y_h = hmatrix.matvec(&x);
            let diff: f64 = y_dense
                .iter()
                .zip(y_h.iter())
                .map(|(a, b)| (a - b).norm_sqr())
                .sum::<f64>()
                .sqrt();
            let reference = y_dense
                .iter()
                .map(|a| a.norm_sqr())
                .sum::<f64>()
                .sqrt()
                .max(1e-30);
            assert!(
                diff / reference < 1e-6,
                "H-matrix matvec must match dense TBEM on the small Mie fixture: rel={:.3e}",
                diff / reference
            );
        } else {
            panic!("unexpected assembly variants");
        }

        let dense = BemSolver::new()
            .with_solver_method(SolverMethod::Direct)
            .with_assembly_method(AssemblyMethod::Tbem)
            .solve(&problem)
            .expect("dense TBEM solve");
        let hmatrix = BemSolver::new()
            .with_solver_method(SolverMethod::Gmres)
            .with_assembly_method(AssemblyMethod::Hmatrix)
            .with_max_iterations(500)
            .with_tolerance(1e-8)
            .solve(&problem)
            .expect("H-matrix GMRES solve");

        let diff: f64 = dense
            .surface_pressure
            .iter()
            .zip(hmatrix.surface_pressure.iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let reference: f64 = dense
            .surface_pressure
            .iter()
            .map(|a| a.norm_sqr())
            .sum::<f64>()
            .sqrt()
            .max(1e-30);
        assert!(
            diff / reference < 1e-6,
            "H-matrix solve must match dense TBEM on the small Mie fixture: rel={:.3e}",
            diff / reference
        );
    }

    #[test]
    fn test_assembly_method_auto_resolve() {
        // Tbem at the threshold and below; Hmatrix above.
        assert_eq!(
            AssemblyMethod::Auto.resolve(0),
            AssemblyMethod::Tbem,
            "Auto should pick Tbem on empty input"
        );
        assert_eq!(
            AssemblyMethod::Auto.resolve(AUTO_TBEM_DOF_LIMIT - 1),
            AssemblyMethod::Tbem,
            "Auto should pick Tbem one DOF below the threshold"
        );
        assert_eq!(
            AssemblyMethod::Auto.resolve(AUTO_TBEM_DOF_LIMIT),
            AssemblyMethod::Hmatrix,
            "Auto should pick Hmatrix at the threshold (Hmatrix wins at 1280 DOFs empirically)"
        );
        // Concrete variants pass through unchanged.
        for method in [
            AssemblyMethod::Tbem,
            AssemblyMethod::Hmatrix,
            AssemblyMethod::Slfmm,
            AssemblyMethod::Mlfmm,
            AssemblyMethod::DirectionalMlfmm,
        ] {
            assert_eq!(
                method.resolve(123),
                method,
                "Non-Auto variant {method:?} must pass through resolve()"
            );
        }
    }

    // Direct solver tests work in both native and WASM modes
    #[test]
    fn test_bem_solver_small_problem() {
        // Very small problem for quick test
        let problem = BemProblem::rigid_sphere_scattering_custom(
            0.1,   // radius
            100.0, // very low frequency for quick test
            343.0, 1.21, 4, // coarse mesh
            8,
        );

        let solver = BemSolver::new();
        let result = solver.solve(&problem);

        assert!(result.is_ok());
        let solution = result.unwrap();
        assert!(solution.num_dofs() > 0);
        assert!(solution.max_surface_pressure() > 0.0);
    }

    #[test]
    fn test_dense_multi_rhs_matches_independent_solves() {
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
        let incident_fields = vec![
            IncidentField::plane_wave_z(),
            IncidentField::plane_wave_neg_z(),
        ];
        let solver = BemSolver::new();
        let batch = solver
            .solve_many(&problem, &incident_fields)
            .expect("dense multi-RHS solve");
        assert_eq!(batch.len(), incident_fields.len());

        for (field, batched_solution) in incident_fields.iter().zip(batch) {
            let independent = solver
                .solve(&problem.clone().with_incident_field(field.clone()))
                .expect("independent dense solve");
            assert_eq!(
                batched_solution.surface_pressure.len(),
                independent.surface_pressure.len()
            );
            let maximum_difference = batched_solution
                .surface_pressure
                .iter()
                .zip(independent.surface_pressure.iter())
                .map(|(batched, independent)| (*batched - *independent).norm())
                .fold(0.0, f64::max);
            assert!(
                maximum_difference < 1e-10,
                "batch mismatch {maximum_difference:e}"
            );
        }
    }

    #[test]
    fn test_wavenumber_objective_gradient_matches_frequency_difference() {
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
        let solver = BemSolver::new()
            .with_assembly_method(AssemblyMethod::Tbem)
            .with_solver_method(SolverMethod::Direct);
        let elements = solver
            .prepare_elements(&problem)
            .expect("prepared elements");
        let dimension = count_dofs(&elements);
        let forward_rhs = (0..dimension)
            .map(|index| Complex64::new(0.2 + index as f64 / 17.0, -0.1 + index as f64 / 23.0))
            .collect::<Vec<_>>();
        let objective_gradient = (0..dimension)
            .map(|index| Complex64::new(0.3 - index as f64 / 19.0, 0.05 + index as f64 / 29.0))
            .collect::<Vec<_>>();
        let analytic = solver
            .solve_wavenumber_objective_gradient(&problem, &forward_rhs, &objective_gradient)
            .expect("analytic wavenumber objective gradient");

        let delta_k = 1.0e-4;
        let delta_frequency = delta_k * problem.physics.speed_of_sound / (2.0 * PI);
        let at_frequency = |frequency: f64| {
            let mut candidate = problem.clone();
            let mut physics = PhysicsParams::try_new(
                frequency,
                problem.physics.speed_of_sound,
                problem.physics.density,
                problem.physics.tau < 0.0,
            )
            .expect("finite perturbed physics");
            physics.harmonic_factor = problem.physics.harmonic_factor;
            physics.pressure_factor = physics.density * physics.omega * physics.harmonic_factor;
            physics.validate().expect("validated perturbed physics");
            candidate.physics = physics;
            candidate
        };
        let objective = |candidate: &BemProblem| {
            let solution = solver
                .solve_rhs_many(candidate, &[forward_rhs.clone()])
                .expect("finite perturbed solve")
                .pop()
                .expect("one perturbed solution");
            objective_gradient
                .iter()
                .zip(solution.coefficients())
                .map(|(gradient, coefficient)| (gradient.conj() * coefficient).re)
                .sum::<f64>()
        };
        let finite_difference =
            (objective(&at_frequency(problem.physics.frequency + delta_frequency))
                - objective(&at_frequency(problem.physics.frequency - delta_frequency)))
                / (2.0 * delta_k);
        let scale = analytic.abs().max(finite_difference.abs()).max(1.0);
        assert!(
            (analytic - finite_difference).abs() / scale < 2.0e-5,
            "wavenumber objective gradient mismatch: analytic={analytic:e}, finite_difference={finite_difference:e}"
        );
    }

    #[test]
    fn wavenumber_objective_gradient_accepts_hmatrix_and_higher_order() {
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
        let elements = BemSolver::new()
            .prepare_elements(&problem)
            .expect("prepared elements");
        let dimension = count_dofs(&elements);
        let rhs = vec![Complex64::new(1.0, 0.0); dimension];

        let compressed = BemSolver::new()
            .with_assembly_method(AssemblyMethod::Hmatrix)
            .with_solver_method(SolverMethod::Gmres);
        let analytic = compressed
            .solve_wavenumber_objective_gradient(&problem, &rhs, &rhs)
            .expect("H-matrix wavenumber objective gradient");
        assert!(analytic.is_finite());
        let delta_k = 1.0e-4;
        let at_wavenumber = |wavenumber: f64| {
            let mut candidate = problem.clone();
            let frequency = wavenumber * problem.physics.speed_of_sound / (2.0 * PI);
            let mut physics = PhysicsParams::try_new(
                frequency,
                problem.physics.speed_of_sound,
                problem.physics.density,
                problem.physics.tau < 0.0,
            )
            .expect("finite perturbed physics");
            physics.harmonic_factor = problem.physics.harmonic_factor;
            physics.pressure_factor = physics.density * physics.omega * physics.harmonic_factor;
            candidate.physics = physics;
            candidate
        };
        let objective = |candidate: &BemProblem| {
            let solution = compressed
                .solve_rhs_many(candidate, std::slice::from_ref(&rhs))
                .expect("finite H-matrix perturbed solve")
                .pop()
                .expect("one H-matrix perturbed solution");
            rhs.iter()
                .zip(solution.coefficients())
                .map(|(gradient, coefficient)| (gradient.conj() * coefficient).re)
                .sum::<f64>()
        };
        let finite_difference = (objective(&at_wavenumber(problem.physics.wave_number + delta_k))
            - objective(&at_wavenumber(problem.physics.wave_number - delta_k)))
            / (2.0 * delta_k);
        let scale = analytic.abs().max(finite_difference.abs()).max(1.0);
        assert!(
            (analytic - finite_difference).abs() / scale < 5.0e-4,
            "H-matrix wavenumber gradient mismatch: analytic={analytic:e}, finite_difference={finite_difference:e}"
        );

        let higher_order = BemSolver::new()
            .with_basis_order(BasisOrder::Linear)
            .with_assembly_method(AssemblyMethod::Tbem)
            .with_solver_method(SolverMethod::Direct);
        let higher_elements = higher_order
            .prepare_elements(&problem)
            .expect("prepared higher-order elements");
        let higher_rhs = vec![Complex64::new(0.7, -0.2); count_dofs(&higher_elements)];
        let higher_analytic = higher_order
            .solve_wavenumber_objective_gradient(&problem, &higher_rhs, &higher_rhs)
            .expect("higher-order wavenumber objective gradient");
        assert!(higher_analytic.is_finite());
    }

    #[test]
    fn wavenumber_objective_gradient_accepts_fmm_backends() {
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
        let elements = BemSolver::new()
            .prepare_elements(&problem)
            .expect("prepared elements");
        let dimension = count_dofs(&elements);
        let rhs = (0..dimension)
            .map(|index| Complex64::new(0.7 + index as f64 / 13.0, -0.2 + index as f64 / 17.0))
            .collect::<Vec<_>>();
        let delta_k = 1.0e-4;

        for assembly in [
            AssemblyMethod::Slfmm,
            AssemblyMethod::Mlfmm,
            AssemblyMethod::DirectionalMlfmm,
        ] {
            let solver = BemSolver::new()
                .with_assembly_method(assembly)
                .with_solver_method(SolverMethod::Gmres);
            let analytic = solver
                .solve_wavenumber_objective_gradient(&problem, &rhs, &rhs)
                .unwrap_or_else(|error| panic!("{assembly:?} wavenumber gradient: {error}"));
            assert!(analytic.is_finite(), "{assembly:?} gradient is non-finite");

            let at_wavenumber = |wavenumber: f64| {
                let mut candidate = problem.clone();
                let frequency = wavenumber * problem.physics.speed_of_sound / (2.0 * PI);
                let mut physics = PhysicsParams::try_new(
                    frequency,
                    problem.physics.speed_of_sound,
                    problem.physics.density,
                    problem.physics.tau < 0.0,
                )
                .expect("finite perturbed physics");
                physics.harmonic_factor = problem.physics.harmonic_factor;
                physics.pressure_factor = physics.density * physics.omega * physics.harmonic_factor;
                candidate.physics = physics;
                candidate
            };
            let objective = |candidate: &BemProblem| {
                let solution = solver
                    .solve_rhs_many(candidate, std::slice::from_ref(&rhs))
                    .unwrap_or_else(|error| panic!("{assembly:?} perturbed solve: {error}"))
                    .pop()
                    .expect("one perturbed solution");
                rhs.iter()
                    .zip(solution.coefficients())
                    .map(|(gradient, coefficient)| (gradient.conj() * coefficient).re)
                    .sum::<f64>()
            };
            let finite_difference =
                (objective(&at_wavenumber(problem.physics.wave_number + delta_k))
                    - objective(&at_wavenumber(problem.physics.wave_number - delta_k)))
                    / (2.0 * delta_k);
            let scale = analytic.abs().max(finite_difference.abs()).max(1.0);
            assert!(
                (analytic - finite_difference).abs() / scale < 5.0e-2,
                "{assembly:?} wavenumber gradient mismatch: analytic={analytic:e}, finite_difference={finite_difference:e}"
            );
        }
    }

    #[test]
    fn higher_order_wavenumber_objective_gradient_accepts_fmm_backends() {
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 3, 4);
        let template = BemSolver::new().with_basis_order(BasisOrder::Quadratic);
        let elements = template
            .prepare_elements(&problem)
            .expect("prepared quadratic elements");
        let dimension = count_dofs(&elements);
        let rhs = (0..dimension)
            .map(|index| Complex64::new(0.7 + index as f64 / 13.0, -0.2 + index as f64 / 17.0))
            .collect::<Vec<_>>();
        let delta_k = 1.0e-4;

        for assembly in [
            AssemblyMethod::Hmatrix,
            AssemblyMethod::Slfmm,
            AssemblyMethod::Mlfmm,
            AssemblyMethod::DirectionalMlfmm,
        ] {
            let solver = BemSolver::new()
                .with_basis_order(BasisOrder::Quadratic)
                .with_basis_continuity(BasisContinuity::Discontinuous)
                .with_assembly_method(assembly)
                .with_solver_method(SolverMethod::Gmres);
            let analytic = solver
                .solve_wavenumber_objective_gradient(&problem, &rhs, &rhs)
                .unwrap_or_else(|error| {
                    panic!("{assembly:?} higher-order wavenumber gradient: {error}")
                });
            assert!(
                analytic.is_finite(),
                "{assembly:?} higher-order gradient is non-finite"
            );

            let at_wavenumber = |wavenumber: f64| {
                let mut candidate = problem.clone();
                let frequency = wavenumber * problem.physics.speed_of_sound / (2.0 * PI);
                let mut physics = PhysicsParams::try_new(
                    frequency,
                    problem.physics.speed_of_sound,
                    problem.physics.density,
                    problem.physics.tau < 0.0,
                )
                .expect("finite perturbed physics");
                physics.harmonic_factor = problem.physics.harmonic_factor;
                physics.pressure_factor = physics.density * physics.omega * physics.harmonic_factor;
                candidate.physics = physics;
                candidate
            };
            let objective = |candidate: &BemProblem| {
                let solution = solver
                    .solve_rhs_many(candidate, std::slice::from_ref(&rhs))
                    .unwrap_or_else(|error| panic!("{assembly:?} perturbed solve: {error}"))
                    .pop()
                    .expect("one perturbed solution");
                rhs.iter()
                    .zip(solution.coefficients())
                    .map(|(gradient, coefficient)| (gradient.conj() * coefficient).re)
                    .sum::<f64>()
            };
            let finite_difference =
                (objective(&at_wavenumber(problem.physics.wave_number + delta_k))
                    - objective(&at_wavenumber(problem.physics.wave_number - delta_k)))
                    / (2.0 * delta_k);
            let scale = analytic.abs().max(finite_difference.abs()).max(1.0);
            assert!(
                (analytic - finite_difference).abs() / scale < 5.0e-2,
                "{assembly:?} higher-order wavenumber gradient mismatch: analytic={analytic:e}, finite_difference={finite_difference:e}"
            );
        }
    }

    #[test]
    fn wavenumber_objective_gradient_frequency_batch_matches_scalar_solves() {
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
        let solver = BemSolver::new()
            .with_assembly_method(AssemblyMethod::Tbem)
            .with_solver_method(SolverMethod::Direct);
        let elements = solver
            .prepare_elements(&problem)
            .expect("prepared elements");
        let dimension = count_dofs(&elements);
        let frequencies = [100.0, 120.0];
        let make_rhs = |offset: usize| {
            (0..dimension)
                .map(|index| {
                    Complex64::new(
                        0.2 + (index + offset) as f64 / 17.0,
                        -0.1 + (index + 2 * offset) as f64 / 23.0,
                    )
                })
                .collect::<Vec<_>>()
        };
        let forward_rhs_batches = vec![make_rhs(0), make_rhs(3)];
        let objective_gradient_batches = vec![make_rhs(5), make_rhs(7)];

        let gradients = solver
            .solve_wavenumber_objective_gradient_many(
                &problem,
                &frequencies,
                &forward_rhs_batches,
                &objective_gradient_batches,
            )
            .expect("frequency-batched wavenumber gradients");
        assert_eq!(gradients.len(), frequencies.len());

        for (index, &frequency) in frequencies.iter().enumerate() {
            let mut candidate = problem.clone();
            let mut physics = PhysicsParams::try_new(
                frequency,
                problem.physics.speed_of_sound,
                problem.physics.density,
                problem.physics.tau < 0.0,
            )
            .expect("finite candidate physics");
            physics.harmonic_factor = problem.physics.harmonic_factor;
            physics.pressure_factor = physics.density * physics.omega * physics.harmonic_factor;
            candidate.physics = physics;
            let scalar = solver
                .solve_wavenumber_objective_gradient(
                    &candidate,
                    &forward_rhs_batches[index],
                    &objective_gradient_batches[index],
                )
                .expect("scalar wavenumber gradient");
            assert!(
                (gradients[index] - scalar).abs() <= 1.0e-12,
                "frequency {frequency} batch/scalar mismatch: {} vs {scalar}",
                gradients[index]
            );
        }
    }

    #[test]
    fn wavenumber_objective_gradient_frequency_batch_rejects_bad_shape() {
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 3, 4);
        let solver = BemSolver::new();
        let error = solver
            .solve_wavenumber_objective_gradient_many(
                &problem,
                &[100.0, 120.0],
                &[Vec::new()],
                &[Vec::new(), Vec::new()],
            )
            .expect_err("mismatched gradient batch cardinalities");
        assert!(
            error.to_string().contains("wavenumber-gradient batch"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_source_position_objective_gradient_matches_central_difference() {
        let position = [0.45, -0.3, 0.4];
        let strength = Complex64::new(0.8, -0.2);
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8)
            .with_incident_field(IncidentField::PointSource { position, strength });
        let solver = BemSolver::new()
            .with_assembly_method(AssemblyMethod::Tbem)
            .with_solver_method(SolverMethod::Direct);
        let elements = solver
            .prepare_elements(&problem)
            .expect("prepared elements");
        let dimension = count_dofs(&elements);
        let objective_gradient = (0..dimension)
            .map(|index| Complex64::new(0.3 - index as f64 / 19.0, 0.05 + index as f64 / 29.0))
            .collect::<Vec<_>>();
        let analytic = solver
            .solve_source_position_objective_gradient(&problem, &objective_gradient)
            .expect("analytic source-position objective gradient");

        let delta = 1.0e-5;
        let objective = |source_position: [f64; 3]| {
            let candidate = problem
                .clone()
                .with_incident_field(IncidentField::PointSource {
                    position: source_position,
                    strength,
                });
            let solution = solver.solve(&candidate).expect("source-position solve");
            objective_gradient
                .iter()
                .zip(&solution.surface_pressure)
                .map(|(gradient, coefficient)| (gradient.conj() * coefficient).re)
                .sum::<f64>()
        };
        for axis in 0..3 {
            let mut plus = position;
            plus[axis] += delta;
            let mut minus = position;
            minus[axis] -= delta;
            let finite_difference = (objective(plus) - objective(minus)) / (2.0 * delta);
            let scale = analytic[axis].abs().max(finite_difference.abs()).max(1.0);
            assert!(
                (analytic[axis] - finite_difference).abs() / scale < 2.0e-5,
                "source-position gradient axis {axis}: analytic={} finite_difference={}",
                analytic[axis],
                finite_difference
            );
        }
    }

    #[test]
    fn higher_order_source_position_objective_gradient_matches_central_difference() {
        let source_position = [0.45, -0.3, 0.4];
        let source_strength = Complex64::new(0.8, -0.2);
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8)
            .with_incident_field(IncidentField::PointSource {
                position: source_position,
                strength: source_strength,
            });
        let solver = BemSolver::new()
            .with_assembly_method(AssemblyMethod::Tbem)
            .with_solver_method(SolverMethod::Direct)
            .with_basis_order(BasisOrder::Quadratic);
        let elements = solver
            .prepare_elements(&problem)
            .expect("prepared quadratic elements");
        let dimension = count_dofs(&elements);
        let objective_gradient = (0..dimension)
            .map(|index| Complex64::new(0.3 - index as f64 / 19.0, 0.05 + index as f64 / 29.0))
            .collect::<Vec<_>>();
        let analytic = solver
            .solve_source_position_objective_gradient(&problem, &objective_gradient)
            .expect("quadratic source-position objective gradient");
        let objective = |candidate: &BemProblem| {
            let solution = solver
                .solve(candidate)
                .expect("quadratic source-position solve");
            objective_gradient
                .iter()
                .zip(&solution.surface_pressure)
                .map(|(gradient, coefficient)| (gradient.conj() * coefficient).re)
                .sum::<f64>()
        };
        let delta = 1.0e-5;
        for axis in 0..3 {
            let mut plus = problem.clone();
            let mut minus = problem.clone();
            if let IncidentField::PointSource { position, .. } = &mut plus.incident_field {
                position[axis] += delta;
            }
            if let IncidentField::PointSource { position, .. } = &mut minus.incident_field {
                position[axis] -= delta;
            }
            let finite_difference = (objective(&plus) - objective(&minus)) / (2.0 * delta);
            let scale = analytic[axis].abs().max(finite_difference.abs()).max(1.0);
            assert!(
                (analytic[axis] - finite_difference).abs() / scale < 2.0e-4,
                "quadratic source-position gradient axis {axis} mismatch: analytic={:e}, finite_difference={finite_difference:e}",
                analytic[axis]
            );
        }
    }

    #[test]
    fn continuous_quadratic_geometry_translation_gradient_matches_central_difference() {
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
        let solver = BemSolver::new()
            .with_assembly_method(AssemblyMethod::Tbem)
            .with_solver_method(SolverMethod::Direct)
            .with_continuous_quadratic();
        let elements = solver
            .prepare_elements(&problem)
            .expect("prepared continuous elements");
        let dimension = count_dofs(&elements);
        let objective_gradient = (0..dimension)
            .map(|index| Complex64::new(0.3 - index as f64 / 19.0, 0.05 + index as f64 / 29.0))
            .collect::<Vec<_>>();
        let analytic = solver
            .solve_geometry_translation_objective_gradient(&problem, &objective_gradient)
            .expect("continuous quadratic translation objective gradient");
        let objective = |candidate: &BemProblem| {
            let solution = solver
                .solve(candidate)
                .expect("continuous quadratic translation solve");
            objective_gradient
                .iter()
                .zip(&solution.surface_pressure)
                .map(|(gradient, coefficient)| (gradient.conj() * coefficient).re)
                .sum::<f64>()
        };
        let delta = 1.0e-5;
        for axis in 0..3 {
            let mut plus = problem.clone();
            let mut minus = problem.clone();
            for node in plus.mesh.nodes.iter_mut() {
                node[axis] += delta;
            }
            for node in minus.mesh.nodes.iter_mut() {
                node[axis] -= delta;
            }
            for element in plus.mesh.elements.iter_mut() {
                element.center[axis] += delta;
            }
            for element in minus.mesh.elements.iter_mut() {
                element.center[axis] -= delta;
            }
            let finite_difference = (objective(&plus) - objective(&minus)) / (2.0 * delta);
            let scale = analytic[axis].abs().max(finite_difference.abs()).max(1.0);
            assert!(
                (analytic[axis] - finite_difference).abs() / scale < 2.0e-4,
                "continuous quadratic translation gradient axis {axis} mismatch: analytic={:e}, finite_difference={finite_difference:e}",
                analytic[axis]
            );
        }
    }

    #[test]
    fn higher_order_compressed_source_position_objective_gradient_matches_central_difference() {
        let source_position = [0.45, -0.3, 0.4];
        let source_strength = Complex64::new(0.8, -0.2);
        let problem = BemProblem {
            mesh: crate::core::mesh::generate_cubed_sphere_quad_mesh(0.1, 1),
            physics: PhysicsParams::new(120.0, 343.0, 1.21, false),
            incident_field: IncidentField::PointSource {
                position: source_position,
                strength: source_strength,
            },
            bc_type: BoundaryConditionType::Rigid,
            use_burton_miller: true,
        };
        let solver = BemSolver::new()
            .with_basis_order(BasisOrder::Quadratic)
            .with_basis_continuity(BasisContinuity::Continuous)
            .with_assembly_method(AssemblyMethod::Hmatrix)
            .with_solver_method(SolverMethod::Gmres);
        let elements = solver
            .prepare_elements(&problem)
            .expect("prepared continuous quadratic elements");
        let dimension = count_dofs(&elements);
        let objective_gradient = (0..dimension)
            .map(|index| Complex64::new(0.3 - index as f64 / 19.0, 0.05 + index as f64 / 29.0))
            .collect::<Vec<_>>();
        let analytic = solver
            .solve_source_position_objective_gradient(&problem, &objective_gradient)
            .expect("higher-order compressed source-position objective gradient");
        let objective = |position: [f64; 3]| {
            let candidate = problem
                .clone()
                .with_incident_field(IncidentField::PointSource {
                    position,
                    strength: source_strength,
                });
            let solution = solver
                .solve(&candidate)
                .expect("compressed source-position solve");
            objective_gradient
                .iter()
                .zip(&solution.surface_pressure)
                .map(|(gradient, coefficient)| (gradient.conj() * coefficient).re)
                .sum::<f64>()
        };
        for axis in 0..3 {
            let mut plus = source_position;
            plus[axis] += 1.0e-5;
            let mut minus = source_position;
            minus[axis] -= 1.0e-5;
            let finite_difference = (objective(plus) - objective(minus)) / (2.0e-5);
            let scale = analytic[axis].abs().max(finite_difference.abs()).max(1.0);
            assert!(
                (analytic[axis] - finite_difference).abs() / scale < 5.0e-3,
                "higher-order compressed source-position axis {axis}: analytic={}, finite_difference={finite_difference}",
                analytic[axis]
            );
        }
    }

    #[test]
    fn source_position_objective_gradient_rejects_unsupported_inputs() {
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
        let solver = BemSolver::new();
        let elements = solver
            .prepare_elements(&problem)
            .expect("prepared elements");
        let rhs = vec![Complex64::new(1.0, 0.0); count_dofs(&elements)];
        assert!(matches!(
            solver.solve_source_position_objective_gradient(&problem, &rhs),
            Err(BemError::NotImplemented(message)) if message.contains("point-source")
        ));

        let point_source = problem.with_incident_field(IncidentField::PointSource {
            position: [0.45, -0.3, 0.4],
            strength: Complex64::new(0.8, -0.2),
        });
        let compressed = BemSolver::new()
            .with_assembly_method(AssemblyMethod::Hmatrix)
            .with_solver_method(SolverMethod::Gmres);
        assert!(matches!(
        compressed.solve_source_position_objective_gradient(&point_source, &rhs),
        Ok(gradient) if gradient.iter().all(|value| value.is_finite())
        ));
    }

    #[test]
    fn geometry_translation_objective_gradient_matches_central_difference() {
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
        let solver = BemSolver::new()
            .with_assembly_method(AssemblyMethod::Tbem)
            .with_solver_method(SolverMethod::Direct);
        let elements = solver
            .prepare_elements(&problem)
            .expect("prepared elements");
        let dimension = count_dofs(&elements);
        let objective_gradient = (0..dimension)
            .map(|index| Complex64::new(0.3 - index as f64 / 19.0, 0.05 + index as f64 / 29.0))
            .collect::<Vec<_>>();
        let analytic = solver
            .solve_geometry_translation_objective_gradient(&problem, &objective_gradient)
            .expect("analytic geometry-translation objective gradient");
        let objective = |translation: [f64; 3]| {
            let mut candidate = problem.clone();
            for node in &mut candidate.mesh.nodes {
                for axis in 0..3 {
                    node[axis] += translation[axis];
                }
            }
            for element in &mut candidate.mesh.elements {
                for axis in 0..3 {
                    element.center[axis] += translation[axis];
                }
            }
            let solution = solver.solve(&candidate).expect("translated solve");
            objective_gradient
                .iter()
                .zip(&solution.surface_pressure)
                .map(|(gradient, coefficient)| (gradient.conj() * coefficient).re)
                .sum::<f64>()
        };
        let delta = 1.0e-5;
        for axis in 0..3 {
            let mut plus = [0.0; 3];
            plus[axis] += delta;
            let mut minus = [0.0; 3];
            minus[axis] -= delta;
            let finite_difference = (objective(plus) - objective(minus)) / (2.0 * delta);
            let scale = analytic[axis].abs().max(finite_difference.abs()).max(1.0);
            assert!(
                (analytic[axis] - finite_difference).abs() / scale < 2.0e-5,
                "geometry-translation gradient axis {axis}: analytic={}, finite_difference={}",
                analytic[axis],
                finite_difference
            );
        }
    }

    #[test]
    fn geometry_translation_objective_gradient_accepts_compressed_backend() {
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
        let solver = BemSolver::new()
            .with_assembly_method(AssemblyMethod::Hmatrix)
            .with_solver_method(SolverMethod::Gmres);
        let elements = BemSolver::new()
            .prepare_elements(&problem)
            .expect("prepared elements");
        let objective_gradient = vec![Complex64::new(1.0, 0.0); count_dofs(&elements)];
        let gradient = solver
            .solve_geometry_translation_objective_gradient(&problem, &objective_gradient)
            .expect("H-matrix geometry-translation objective gradient");
        assert!(gradient.iter().all(|value| value.is_finite()));

        let dense = BemSolver::new()
            .with_assembly_method(AssemblyMethod::Tbem)
            .with_solver_method(SolverMethod::Direct);
        assert!(matches!(
            dense.solve_geometry_translation_objective_gradient(
                &problem,
                &[Complex64::new(1.0, 0.0)]
            ),
            Err(BemError::InvalidParameters(_))
        ));
    }

    #[test]
    fn higher_order_compressed_geometry_translation_gradient_matches_central_difference() {
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
        let solver = BemSolver::new()
            .with_assembly_method(AssemblyMethod::Hmatrix)
            .with_solver_method(SolverMethod::Gmres)
            .with_basis_order(BasisOrder::Linear)
            .with_max_iterations(500)
            .with_tolerance(1.0e-7)
            .with_compressed_retries(0);
        let elements = solver
            .prepare_elements(&problem)
            .expect("prepared linear elements");
        let dimension = count_dofs(&elements);
        let objective_gradient = (0..dimension)
            .map(|index| Complex64::new(0.3 - index as f64 / 19.0, 0.05 + index as f64 / 29.0))
            .collect::<Vec<_>>();
        let analytic = solver
            .solve_geometry_translation_objective_gradient(&problem, &objective_gradient)
            .expect("linear compressed translation objective gradient");
        let objective = |candidate: &BemProblem| {
            let solution = solver.solve(candidate).expect("linear compressed solve");
            objective_gradient
                .iter()
                .zip(&solution.surface_pressure)
                .map(|(gradient, coefficient)| (gradient.conj() * coefficient).re)
                .sum::<f64>()
        };
        let delta = 1.0e-5;
        for axis in 0..3 {
            let mut plus = problem.clone();
            let mut minus = problem.clone();
            for node in &mut plus.mesh.nodes {
                node[axis] += delta;
            }
            for node in &mut minus.mesh.nodes {
                node[axis] -= delta;
            }
            for element in &mut plus.mesh.elements {
                element.center[axis] += delta;
            }
            for element in &mut minus.mesh.elements {
                element.center[axis] -= delta;
            }
            let finite_difference = (objective(&plus) - objective(&minus)) / (2.0 * delta);
            let scale = analytic[axis].abs().max(finite_difference.abs()).max(1.0);
            assert!(
                (analytic[axis] - finite_difference).abs() / scale < 1.0e-10,
                "linear compressed translation gradient axis {axis}: analytic={}, finite_difference={finite_difference}",
                analytic[axis]
            );
        }
    }

    #[test]
    fn slfmm_scalar_gmres_uses_nearfield_ilu_and_certifies_residual() {
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
        let solver = BemSolver::new()
            .with_assembly_method(AssemblyMethod::Slfmm)
            .with_solver_method(SolverMethod::Gmres);
        let solution = solver.solve(&problem).expect("SLFMM near-field ILU solve");
        assert!(
            solution
                .surface_pressure
                .iter()
                .all(|value| value.re.is_finite() && value.im.is_finite())
        );
    }

    #[test]
    fn persistent_operator_cache_reuses_exact_operator_and_rebuilds_frequency() {
        let base = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
        let solver = BemSolver::new()
            .with_assembly_method(AssemblyMethod::Tbem)
            .with_solver_method(SolverMethod::Direct);
        let mut cache = BemOperatorCache::new();
        let _first = solver
            .solve_with_cache(&base, &mut cache, BemReusePolicy::default())
            .expect("initial cached solve");
        assert!(cache.is_populated());
        assert_eq!(cache.rebuilds(), 1);
        assert_eq!(cache.reuse_hits(), 0);

        let second_problem = base
            .clone()
            .with_incident_field(IncidentField::plane_wave_neg_z());
        let cached_second = solver
            .solve_with_cache(&second_problem, &mut cache, BemReusePolicy::default())
            .expect("same-operator cached solve");
        let independent_second = solver.solve(&second_problem).expect("independent solve");
        let difference = cached_second
            .surface_pressure
            .iter()
            .zip(&independent_second.surface_pressure)
            .map(|(cached, independent)| (*cached - *independent).norm())
            .fold(0.0, f64::max);
        assert!(
            difference < 1.0e-10,
            "cached solve mismatch: {difference:e}"
        );
        assert_eq!(cache.rebuilds(), 1);
        assert_eq!(cache.reuse_hits(), 1);
        assert!(
            cache
                .last_decision()
                .is_some_and(|decision| decision.rejection.is_none())
        );

        let mut changed_frequency = base.clone();
        let mut physics =
            PhysicsParams::try_new(110.0, 343.0, 1.21, false).expect("valid frequency");
        physics.harmonic_factor = base.physics.harmonic_factor;
        physics.pressure_factor = physics.density * physics.omega * physics.harmonic_factor;
        changed_frequency.physics = physics;
        solver
            .solve_with_cache(&changed_frequency, &mut cache, BemReusePolicy::default())
            .expect("frequency rebuild");
        assert_eq!(cache.rebuilds(), 2);
        assert_eq!(cache.reuse_hits(), 1);
    }

    #[test]
    fn soft_dirichlet_solve_many_matches_independent_solves() {
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8)
            .with_boundary_condition(BoundaryConditionType::Soft);
        let fields = vec![
            IncidentField::plane_wave_z(),
            IncidentField::plane_wave_neg_z(),
        ];
        let solver = BemSolver::new()
            .with_assembly_method(AssemblyMethod::Tbem)
            .with_solver_method(SolverMethod::Direct);
        let batch = solver
            .solve_many(&problem, &fields)
            .expect("soft block solve");
        assert_eq!(batch.len(), fields.len());
        for (field, batched) in fields.iter().zip(batch) {
            let independent = solver
                .solve(&problem.clone().with_incident_field(field.clone()))
                .expect("independent soft solve");
            let difference = batched
                .surface_pressure
                .iter()
                .zip(&independent.surface_pressure)
                .map(|(batched, independent)| (*batched - *independent).norm())
                .fold(0.0, f64::max);
            assert!(difference < 1.0e-10, "soft batch mismatch: {difference:e}");
        }
    }

    #[test]
    fn soft_dirichlet_adjoint_many_certifies_hermitian_residuals() {
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8)
            .with_boundary_condition(BoundaryConditionType::Soft);
        let solver = BemSolver::new()
            .with_assembly_method(AssemblyMethod::Tbem)
            .with_solver_method(SolverMethod::Direct);
        let elements = solver.prepare_elements(&problem).expect("soft elements");
        let rhs = vec![
            (0..count_dofs(&elements))
                .map(|index| Complex64::new(0.2 + index as f64 / 31.0, -0.1))
                .collect::<Vec<_>>(),
            (0..count_dofs(&elements))
                .map(|index| Complex64::new(-0.3, 0.15 + index as f64 / 47.0))
                .collect::<Vec<_>>(),
        ];
        let solutions = solver
            .solve_adjoint_many(&problem, &rhs)
            .expect("soft Hermitian-adjoint block solve");
        assert_eq!(solutions.len(), rhs.len());
        assert!(
            solutions
                .iter()
                .all(|solution| solution.relative_residual() <= 1.0e-10)
        );
    }

    #[test]
    fn persistent_operator_cache_reuses_hmatrix_operator_and_rebuilds_frequency() {
        let base = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
        let solver = BemSolver::new()
            .with_assembly_method(AssemblyMethod::Hmatrix)
            .with_solver_method(SolverMethod::Gmres)
            .with_max_iterations(500)
            .with_tolerance(1.0e-7)
            .with_compressed_retries(0);
        let mut cache = BemOperatorCache::new();
        solver
            .solve_with_cache(&base, &mut cache, BemReusePolicy::default())
            .expect("initial H-matrix cached solve");
        assert_eq!(cache.rebuilds(), 1);
        assert_eq!(cache.reuse_hits(), 0);

        let changed_load = base
            .clone()
            .with_incident_field(IncidentField::plane_wave_neg_z());
        let cached = solver
            .solve_with_cache(&changed_load, &mut cache, BemReusePolicy::default())
            .expect("same H-matrix operator cached solve");
        let independent = solver
            .solve(&changed_load)
            .expect("independent H-matrix solve");
        let difference = cached
            .surface_pressure
            .iter()
            .zip(&independent.surface_pressure)
            .map(|(cached, independent)| (*cached - *independent).norm())
            .fold(0.0, f64::max);
        assert!(
            difference <= 1.0e-10,
            "H-matrix cached solve mismatch: {difference:e}"
        );
        assert_eq!(cache.rebuilds(), 1);
        assert_eq!(cache.reuse_hits(), 1);

        let mut changed_frequency = changed_load.clone();
        let mut physics = PhysicsParams::try_new(110.0, 343.0, 1.21, false)
            .expect("valid changed H-matrix frequency");
        physics.harmonic_factor = base.physics.harmonic_factor;
        physics.pressure_factor = physics.density * physics.omega * physics.harmonic_factor;
        changed_frequency.physics = physics;
        solver
            .solve_with_cache(&changed_frequency, &mut cache, BemReusePolicy::default())
            .expect("changed H-matrix frequency rebuild");
        assert_eq!(cache.rebuilds(), 2);
        assert_eq!(cache.reuse_hits(), 1);
    }

    #[test]
    fn persistent_operator_cache_reuses_fmm_operators_and_rebuilds_frequency() {
        let base = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);

        for assembly_method in [
            AssemblyMethod::Slfmm,
            AssemblyMethod::Mlfmm,
            AssemblyMethod::DirectionalMlfmm,
        ] {
            let solver = BemSolver::new()
                .with_assembly_method(assembly_method)
                .with_solver_method(SolverMethod::Gmres)
                .with_max_iterations(500)
                .with_tolerance(1.0e-7)
                .with_compressed_retries(0);
            let mut cache = BemOperatorCache::new();

            solver
                .solve_with_cache(&base, &mut cache, BemReusePolicy::default())
                .unwrap_or_else(|error| {
                    panic!("initial {assembly_method:?} cached solve: {error}")
                });
            assert_eq!(cache.rebuilds(), 1, "{assembly_method:?} initial rebuild");
            assert_eq!(cache.reuse_hits(), 0, "{assembly_method:?} initial reuse");

            let changed_load = base
                .clone()
                .with_incident_field(IncidentField::plane_wave_neg_z());
            let cached = solver
                .solve_with_cache(&changed_load, &mut cache, BemReusePolicy::default())
                .unwrap_or_else(|error| panic!("cached {assembly_method:?} load: {error}"));
            let independent = solver
                .solve(&changed_load)
                .unwrap_or_else(|error| panic!("independent {assembly_method:?} load: {error}"));
            let difference = cached
                .surface_pressure
                .iter()
                .zip(&independent.surface_pressure)
                .map(|(cached, independent)| (*cached - *independent).norm())
                .fold(0.0, f64::max);
            assert!(
                difference <= 1.0e-9,
                "{assembly_method:?} cached solve mismatch: {difference:e}"
            );
            assert_eq!(
                cache.rebuilds(),
                1,
                "{assembly_method:?} rebuild after reuse"
            );
            assert_eq!(cache.reuse_hits(), 1, "{assembly_method:?} reuse hit");
            assert!(
                cache
                    .last_decision()
                    .is_some_and(|decision| decision.rejection.is_none())
            );

            let mut changed_frequency = changed_load;
            let mut physics = PhysicsParams::try_new(110.0, 343.0, 1.21, false)
                .expect("valid changed FMM frequency");
            physics.harmonic_factor = base.physics.harmonic_factor;
            physics.pressure_factor = physics.density * physics.omega * physics.harmonic_factor;
            changed_frequency.physics = physics;
            solver
                .solve_with_cache(&changed_frequency, &mut cache, BemReusePolicy::default())
                .unwrap_or_else(|error| panic!("changed {assembly_method:?} frequency: {error}"));
            assert_eq!(cache.rebuilds(), 2, "{assembly_method:?} frequency rebuild");
            assert_eq!(
                cache.reuse_hits(),
                1,
                "{assembly_method:?} frequency reuse count"
            );
        }
    }

    #[test]
    fn persistent_operator_cache_reuses_soft_dirichlet_operator_and_tracks_boundary_type() {
        let base = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8)
            .with_boundary_condition(BoundaryConditionType::Soft);
        let solver = BemSolver::new()
            .with_assembly_method(AssemblyMethod::Tbem)
            .with_solver_method(SolverMethod::Direct);
        let mut cache = BemOperatorCache::new();

        solver
            .solve_with_cache(&base, &mut cache, BemReusePolicy::default())
            .expect("initial soft cached solve");
        let changed_load = base
            .clone()
            .with_incident_field(IncidentField::plane_wave_neg_z());
        let cached = solver
            .solve_with_cache(&changed_load, &mut cache, BemReusePolicy::default())
            .expect("same soft operator cached solve");
        let independent = solver.solve(&changed_load).expect("independent soft solve");
        let difference = cached
            .surface_pressure
            .iter()
            .zip(&independent.surface_pressure)
            .map(|(cached, independent)| (*cached - *independent).norm())
            .fold(0.0, f64::max);
        assert!(
            difference < 1.0e-10,
            "soft cached solve mismatch: {difference:e}"
        );
        assert_eq!(cache.rebuilds(), 1);
        assert_eq!(cache.reuse_hits(), 1);

        let rigid = changed_load.with_boundary_condition(BoundaryConditionType::Rigid);
        solver
            .solve_with_cache(&rigid, &mut cache, BemReusePolicy::default())
            .expect("boundary-type change rebuild");
        assert_eq!(cache.rebuilds(), 2);
        assert_eq!(cache.reuse_hits(), 1);
    }

    #[test]
    fn persistent_operator_cache_rechecks_quadrature_policy_on_configuration_change() {
        use crate::core::integration::QuadratureParams;

        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
        let permissive = TbemQuadraturePolicy {
            regular_coarse_order: 3,
            regular_fine_order: 4,
            singular_coarse: QuadratureParams::default(),
            singular_fine: QuadratureParams::high_accuracy(),
            absolute_tolerance: 1.0e6,
            relative_tolerance: 1.0e6,
        };
        let strict = TbemQuadraturePolicy {
            absolute_tolerance: 0.0,
            relative_tolerance: 0.0,
            ..permissive
        };
        let mut cache = BemOperatorCache::new();
        BemSolver::new()
            .with_quadrature_policy(permissive)
            .solve_with_cache(&problem, &mut cache, BemReusePolicy::default())
            .expect("permissive cached solve");

        let strict_error = BemSolver::new()
            .with_quadrature_policy(strict)
            .solve_with_cache(&problem, &mut cache, BemReusePolicy::default())
            .expect_err("strict policy must not reuse an unchecked cached operator");
        assert!(strict_error.to_string().contains("quadrature admission"));
    }

    #[test]
    fn test_speed_of_sound_objective_gradient_matches_central_difference() {
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
        let solver = BemSolver::new()
            .with_assembly_method(AssemblyMethod::Tbem)
            .with_solver_method(SolverMethod::Direct);
        let elements = solver
            .prepare_elements(&problem)
            .expect("prepared elements");
        let dimension = count_dofs(&elements);
        let forward_rhs = (0..dimension)
            .map(|index| Complex64::new(0.2 + index as f64 / 17.0, -0.1 + index as f64 / 23.0))
            .collect::<Vec<_>>();
        let objective_gradient = (0..dimension)
            .map(|index| Complex64::new(0.3 - index as f64 / 19.0, 0.05 + index as f64 / 29.0))
            .collect::<Vec<_>>();
        let analytic = solver
            .solve_speed_of_sound_objective_gradient(&problem, &forward_rhs, &objective_gradient)
            .expect("analytic sound-speed objective gradient");
        let delta = 1.0e-2;
        let at_speed = |speed_of_sound: f64| {
            let mut candidate = problem.clone();
            let mut physics = PhysicsParams::try_new(
                problem.physics.frequency,
                speed_of_sound,
                problem.physics.density,
                problem.physics.tau < 0.0,
            )
            .expect("finite perturbed physics");
            physics.harmonic_factor = problem.physics.harmonic_factor;
            physics.pressure_factor = physics.density * physics.omega * physics.harmonic_factor;
            physics.validate().expect("validated perturbed physics");
            candidate.physics = physics;
            candidate
        };
        let objective = |candidate: &BemProblem| {
            let solution = solver
                .solve_rhs_many(candidate, &[forward_rhs.clone()])
                .expect("finite perturbed solve")
                .pop()
                .expect("one perturbed solution");
            objective_gradient
                .iter()
                .zip(solution.coefficients())
                .map(|(gradient, coefficient)| (gradient.conj() * coefficient).re)
                .sum::<f64>()
        };
        let finite_difference = (objective(&at_speed(problem.physics.speed_of_sound + delta))
            - objective(&at_speed(problem.physics.speed_of_sound - delta)))
            / (2.0 * delta);
        let scale = analytic.abs().max(finite_difference.abs()).max(1.0);
        assert!(
            (analytic - finite_difference).abs() / scale < 2.0e-5,
            "sound-speed objective gradient mismatch: analytic={analytic:e}, finite_difference={finite_difference:e}"
        );
    }

    #[test]
    fn compressed_material_objective_gradients_match_central_difference() {
        let speed_problem =
            BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
        let density_problem = speed_problem.clone();
        let mut solver = BemSolver::new()
            .with_assembly_method(AssemblyMethod::Hmatrix)
            .with_solver_method(SolverMethod::Gmres);
        solver.hmatrix_aca_tolerance = 1.0e-10;
        solver.hmatrix_max_rank = 64;
        let elements = solver
            .prepare_elements(&speed_problem)
            .expect("prepared elements");
        let dimension = count_dofs(&elements);
        let forward_rhs = (0..dimension)
            .map(|index| Complex64::new(0.2 + index as f64 / 17.0, -0.1 + index as f64 / 23.0))
            .collect::<Vec<_>>();
        let objective_gradient = (0..dimension)
            .map(|index| Complex64::new(0.3 - index as f64 / 19.0, 0.05 + index as f64 / 29.0))
            .collect::<Vec<_>>();
        let objective = |candidate: &BemProblem| {
            let solution = solver
                .solve_rhs_many(candidate, std::slice::from_ref(&forward_rhs))
                .expect("finite perturbed H-matrix solve")
                .pop()
                .expect("one perturbed solution");
            objective_gradient
                .iter()
                .zip(solution.coefficients())
                .map(|(gradient, coefficient)| (gradient.conj() * coefficient).re)
                .sum::<f64>()
        };
        let with_medium = |base: &BemProblem, speed: f64, density: f64| {
            let mut candidate = base.clone();
            let mut physics = PhysicsParams::try_new(
                base.physics.frequency,
                speed,
                density,
                base.physics.tau < 0.0,
            )
            .expect("valid perturbed physics");
            physics.harmonic_factor = base.physics.harmonic_factor;
            physics.pressure_factor = physics.density * physics.omega * physics.harmonic_factor;
            physics.validate().expect("validated perturbed physics");
            candidate.physics = physics;
            candidate
        };

        let speed_gradient = solver
            .solve_speed_of_sound_objective_gradient(
                &speed_problem,
                &forward_rhs,
                &objective_gradient,
            )
            .expect("compressed sound-speed objective gradient");
        let speed_step = speed_problem.physics.speed_of_sound * 1.0e-4;
        let speed_finite_difference = (objective(&with_medium(
            &speed_problem,
            speed_problem.physics.speed_of_sound + speed_step,
            speed_problem.physics.density,
        )) - objective(&with_medium(
            &speed_problem,
            speed_problem.physics.speed_of_sound - speed_step,
            speed_problem.physics.density,
        ))) / (2.0 * speed_step);
        let speed_scale = speed_gradient
            .abs()
            .max(speed_finite_difference.abs())
            .max(1.0);
        assert!(
            (speed_gradient - speed_finite_difference).abs() / speed_scale < 2.0e-3,
            "compressed sound-speed gradient mismatch: analytic={speed_gradient:e}, finite_difference={speed_finite_difference:e}"
        );

        let density_gradient = solver
            .solve_density_objective_gradient(&density_problem, &forward_rhs, &objective_gradient)
            .expect("compressed density objective gradient");
        let density_step = density_problem.physics.density * 1.0e-4;
        let density_finite_difference = (objective(&with_medium(
            &density_problem,
            density_problem.physics.speed_of_sound,
            density_problem.physics.density + density_step,
        )) - objective(&with_medium(
            &density_problem,
            density_problem.physics.speed_of_sound,
            density_problem.physics.density - density_step,
        ))) / (2.0 * density_step);
        let density_scale = density_gradient
            .abs()
            .max(density_finite_difference.abs())
            .max(1.0);
        assert!(
            (density_gradient - density_finite_difference).abs() / density_scale < 2.0e-3,
            "compressed density gradient mismatch: analytic={density_gradient:e}, finite_difference={density_finite_difference:e}"
        );
    }

    #[test]
    fn compressed_material_objective_gradients_accept_fmm_backends() {
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
        let reference = BemSolver::new()
            .prepare_elements(&problem)
            .expect("prepared elements");
        let dimension = count_dofs(&reference);
        let forward_rhs = vec![Complex64::new(1.0, 0.0); dimension];
        let objective_gradient = vec![Complex64::new(0.7, -0.2); dimension];
        let at_medium = |base: &BemProblem, speed: f64, density: f64| {
            let mut candidate = base.clone();
            let mut physics = PhysicsParams::try_new(
                base.physics.frequency,
                speed,
                density,
                base.physics.tau < 0.0,
            )
            .expect("valid perturbed physics");
            physics.harmonic_factor = base.physics.harmonic_factor;
            physics.pressure_factor = physics.density * physics.omega * physics.harmonic_factor;
            physics.validate().expect("validated perturbed physics");
            candidate.physics = physics;
            candidate
        };
        for assembly in [
            AssemblyMethod::Slfmm,
            AssemblyMethod::Mlfmm,
            AssemblyMethod::DirectionalMlfmm,
        ] {
            let solver = BemSolver::new()
                .with_assembly_method(assembly)
                .with_solver_method(SolverMethod::Gmres);
            let objective = |candidate: &BemProblem| {
                let solution = solver
                    .solve_rhs_many(candidate, std::slice::from_ref(&forward_rhs))
                    .expect("FMM perturbed solve")
                    .pop()
                    .expect("one FMM perturbed solution");
                objective_gradient
                    .iter()
                    .zip(solution.coefficients())
                    .map(|(gradient, coefficient)| (gradient.conj() * coefficient).re)
                    .sum::<f64>()
            };
            let speed = solver
                .solve_speed_of_sound_objective_gradient(
                    &problem,
                    &forward_rhs,
                    &objective_gradient,
                )
                .unwrap_or_else(|error| panic!("{assembly:?} sound-speed gradient: {error}"));
            let density = solver
                .solve_density_objective_gradient(&problem, &forward_rhs, &objective_gradient)
                .unwrap_or_else(|error| panic!("{assembly:?} density gradient: {error}"));
            assert!(speed.is_finite(), "{assembly:?} speed gradient");
            assert!(density.is_finite(), "{assembly:?} density gradient");
            let speed_step = problem.physics.speed_of_sound * 1.0e-4;
            let speed_fd = (objective(&at_medium(
                &problem,
                problem.physics.speed_of_sound + speed_step,
                problem.physics.density,
            )) - objective(&at_medium(
                &problem,
                problem.physics.speed_of_sound - speed_step,
                problem.physics.density,
            ))) / (2.0 * speed_step);
            let density_step = problem.physics.density * 1.0e-4;
            let density_fd = (objective(&at_medium(
                &problem,
                problem.physics.speed_of_sound,
                problem.physics.density + density_step,
            )) - objective(&at_medium(
                &problem,
                problem.physics.speed_of_sound,
                problem.physics.density - density_step,
            ))) / (2.0 * density_step);
            let speed_scale = speed.abs().max(speed_fd.abs()).max(1.0);
            let density_scale = density.abs().max(density_fd.abs()).max(1.0);
            assert!(
                (speed - speed_fd).abs() / speed_scale < 5.0e-2,
                "{assembly:?} sound-speed gradient mismatch: analytic={speed:e}, finite_difference={speed_fd:e}"
            );
            assert!(
                (density - density_fd).abs() / density_scale < 5.0e-2,
                "{assembly:?} density gradient mismatch: analytic={density:e}, finite_difference={density_fd:e}"
            );
        }
    }

    #[test]
    fn higher_order_compressed_material_objective_gradients_accept_fmm_backends() {
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 3, 4);
        let panel_count = problem
            .mesh
            .elements
            .iter()
            .filter(|element| !element.property.is_evaluation())
            .count();
        let local_basis_count = 6;
        let dimension = panel_count * local_basis_count;
        let forward_rhs = (0..dimension)
            .map(|index| Complex64::new(0.2 + index as f64 / 17.0, -0.1 + index as f64 / 23.0))
            .collect::<Vec<_>>();
        let objective_gradient = (0..dimension)
            .map(|index| Complex64::new(0.3 - index as f64 / 19.0, 0.05 + index as f64 / 29.0))
            .collect::<Vec<_>>();

        for assembly in [
            AssemblyMethod::Hmatrix,
            AssemblyMethod::Slfmm,
            AssemblyMethod::Mlfmm,
            AssemblyMethod::DirectionalMlfmm,
        ] {
            let solver = BemSolver::new()
                .with_basis_order(BasisOrder::Quadratic)
                .with_basis_continuity(BasisContinuity::Discontinuous)
                .with_assembly_method(assembly)
                .with_solver_method(SolverMethod::Gmres);
            let speed = solver
                .solve_speed_of_sound_objective_gradient(
                    &problem,
                    &forward_rhs,
                    &objective_gradient,
                )
                .unwrap_or_else(|error| {
                    panic!("{assembly:?} higher-order speed gradient: {error}")
                });
            let density = solver
                .solve_density_objective_gradient(&problem, &forward_rhs, &objective_gradient)
                .unwrap_or_else(|error| {
                    panic!("{assembly:?} higher-order density gradient: {error}")
                });
            assert!(
                speed.is_finite(),
                "{assembly:?} higher-order speed gradient"
            );
            assert!(
                density.is_finite(),
                "{assembly:?} higher-order density gradient"
            );
        }
    }

    #[test]
    fn test_density_objective_gradient_matches_central_difference() {
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8)
            .with_boundary_condition(BoundaryConditionType::Impedance);
        let solver = BemSolver::new()
            .with_assembly_method(AssemblyMethod::Tbem)
            .with_solver_method(SolverMethod::Direct);
        let elements = solver
            .prepare_elements(&problem)
            .expect("prepared elements");
        let dimension = count_dofs(&elements);
        let forward_rhs = (0..dimension)
            .map(|index| Complex64::new(0.2 + index as f64 / 17.0, -0.1 + index as f64 / 23.0))
            .collect::<Vec<_>>();
        let objective_gradient = (0..dimension)
            .map(|index| Complex64::new(0.3 - index as f64 / 19.0, 0.05 + index as f64 / 29.0))
            .collect::<Vec<_>>();
        let analytic = solver
            .solve_density_objective_gradient(&problem, &forward_rhs, &objective_gradient)
            .expect("analytic density objective gradient");
        let delta = 1.0e-4;
        let at_density = |density: f64| {
            let mut candidate = problem.clone();
            let mut physics = PhysicsParams::try_new(
                problem.physics.frequency,
                problem.physics.speed_of_sound,
                density,
                problem.physics.tau < 0.0,
            )
            .expect("finite perturbed physics");
            physics.harmonic_factor = problem.physics.harmonic_factor;
            physics.pressure_factor = physics.density * physics.omega * physics.harmonic_factor;
            physics.validate().expect("validated perturbed physics");
            candidate.physics = physics;
            candidate
        };
        let objective = |candidate: &BemProblem| {
            let solution = solver
                .solve_rhs_many(candidate, &[forward_rhs.clone()])
                .expect("finite perturbed solve")
                .pop()
                .expect("one perturbed solution");
            objective_gradient
                .iter()
                .zip(solution.coefficients())
                .map(|(gradient, coefficient)| (gradient.conj() * coefficient).re)
                .sum::<f64>()
        };
        let finite_difference = (objective(&at_density(problem.physics.density + delta))
            - objective(&at_density(problem.physics.density - delta)))
            / (2.0 * delta);
        let scale = analytic.abs().max(finite_difference.abs()).max(1.0);
        assert!(
            (analytic - finite_difference).abs() / scale < 2.0e-5,
            "density objective gradient mismatch: analytic={analytic:e}, finite_difference={finite_difference:e}"
        );
    }

    #[test]
    fn density_objective_gradient_accepts_hmatrix_and_higher_order() {
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
        let solver = BemSolver::new()
            .with_assembly_method(AssemblyMethod::Hmatrix)
            .with_solver_method(SolverMethod::Gmres);
        let elements = BemSolver::new()
            .prepare_elements(&problem)
            .expect("prepared elements");
        let rhs = vec![Complex64::new(1.0, 0.0); count_dofs(&elements)];
        let analytic = solver
            .solve_density_objective_gradient(&problem, &rhs, &rhs)
            .expect("H-matrix density objective gradient");
        assert!(analytic.is_finite());
        let higher_order = BemSolver::new()
            .with_basis_order(BasisOrder::Linear)
            .with_assembly_method(AssemblyMethod::Tbem)
            .with_solver_method(SolverMethod::Direct);
        let higher_elements = higher_order
            .prepare_elements(&problem)
            .expect("prepared higher-order elements");
        let higher_rhs = vec![Complex64::new(0.7, -0.2); count_dofs(&higher_elements)];
        let density = higher_order
            .solve_density_objective_gradient(&problem, &higher_rhs, &higher_rhs)
            .expect("higher-order density objective gradient");
        let speed = higher_order
            .solve_speed_of_sound_objective_gradient(&problem, &higher_rhs, &higher_rhs)
            .expect("higher-order sound-speed objective gradient");
        assert!(density.is_finite());
        assert!(speed.is_finite());
        let impedance_problem = problem
            .clone()
            .with_boundary_condition(BoundaryConditionType::Impedance);
        let impedance_density = solver
            .solve_density_objective_gradient(&impedance_problem, &rhs, &rhs)
            .expect("compressed impedance density objective gradient");
        let impedance_speed = solver
            .solve_speed_of_sound_objective_gradient(&impedance_problem, &rhs, &rhs)
            .expect("compressed impedance speed objective gradient");
        assert!(impedance_density.is_finite());
        assert!(impedance_speed.is_finite());
    }

    #[test]
    fn compressed_impedance_material_gradients_match_centered_reassembly() {
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 3, 6)
            .with_boundary_condition(BoundaryConditionType::Impedance);
        let solver = BemSolver::new()
            .with_assembly_method(AssemblyMethod::Hmatrix)
            .with_solver_method(SolverMethod::Gmres)
            .with_max_iterations(100)
            .with_tolerance(1.0e-7);
        let elements = solver
            .prepare_elements(&problem)
            .expect("prepared elements");
        let rhs = vec![Complex64::new(1.0, 0.0); count_dofs(&elements)];
        let speed_gradient = solver
            .solve_speed_of_sound_objective_gradient(&problem, &rhs, &rhs)
            .expect("compressed impedance speed gradient");
        let density_gradient = solver
            .solve_density_objective_gradient(&problem, &rhs, &rhs)
            .expect("compressed impedance density gradient");

        let objective = |candidate: &BemProblem| {
            let solution = solver
                .solve_rhs_many(candidate, &[rhs.clone()])
                .expect("perturbed impedance solve")
                .pop()
                .expect("one perturbed solution");
            rhs.iter()
                .zip(solution.coefficients())
                .map(|(gradient, coefficient)| (gradient.conj() * coefficient).re)
                .sum::<f64>()
        };
        let perturb = |speed: f64, density: f64| {
            let mut candidate = problem.clone();
            let mut physics = PhysicsParams::try_new(
                problem.physics.frequency,
                speed,
                density,
                problem.physics.tau < 0.0,
            )
            .expect("finite perturbed physics");
            physics.harmonic_factor = problem.physics.harmonic_factor;
            physics.pressure_factor = physics.density * physics.omega * physics.harmonic_factor;
            candidate.physics = physics;
            candidate
        };
        let speed_step = 1.0e-3;
        let density_step = 1.0e-5;
        let speed_fd = (objective(&perturb(
            problem.physics.speed_of_sound + speed_step,
            problem.physics.density,
        )) - objective(&perturb(
            problem.physics.speed_of_sound - speed_step,
            problem.physics.density,
        ))) / (2.0 * speed_step);
        let density_fd = (objective(&perturb(
            problem.physics.speed_of_sound,
            problem.physics.density + density_step,
        )) - objective(&perturb(
            problem.physics.speed_of_sound,
            problem.physics.density - density_step,
        ))) / (2.0 * density_step);
        assert!(
            (speed_gradient - speed_fd).abs() / speed_gradient.abs().max(speed_fd.abs()).max(1.0)
                < 3.0e-4,
            "speed gradient mismatch: analytic={speed_gradient:e}, finite_difference={speed_fd:e}"
        );
        assert!(
            (density_gradient - density_fd).abs()
                / density_gradient.abs().max(density_fd.abs()).max(1.0)
                < 3.0e-4,
            "density gradient mismatch: analytic={density_gradient:e}, finite_difference={density_fd:e}"
        );
    }

    #[test]
    fn compressed_impedance_material_gradients_accept_fmm_backends() {
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 3, 6)
            .with_boundary_condition(BoundaryConditionType::Impedance);
        let elements = BemSolver::new()
            .prepare_elements(&problem)
            .expect("prepared impedance elements");
        let rhs = vec![Complex64::new(1.0, 0.0); count_dofs(&elements)];
        for assembly in [AssemblyMethod::Slfmm, AssemblyMethod::Mlfmm] {
            let solver = BemSolver::new()
                .with_assembly_method(assembly)
                .with_solver_method(SolverMethod::Gmres)
                .with_max_iterations(100)
                .with_tolerance(1.0e-7);
            let speed = solver
                .solve_speed_of_sound_objective_gradient(&problem, &rhs, &rhs)
                .unwrap_or_else(|error| panic!("{assembly:?} impedance speed gradient: {error}"));
            let density = solver
                .solve_density_objective_gradient(&problem, &rhs, &rhs)
                .unwrap_or_else(|error| panic!("{assembly:?} impedance density gradient: {error}"));
            assert!(
                speed.is_finite(),
                "{assembly:?} speed gradient must be finite"
            );
            assert!(
                density.is_finite(),
                "{assembly:?} density gradient must be finite"
            );
        }
    }

    #[test]
    fn test_frequency_multi_rhs_batch_matches_scalar_frequency_load_solves() {
        let mut problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
        problem.physics.harmonic_factor = -1.0;
        problem.physics.pressure_factor = -problem.physics.density * problem.physics.omega;
        problem
            .physics
            .validate()
            .expect("alternate validated phasor sign");
        let frequencies = [100.0, 160.0];
        let incident_fields = [
            IncidentField::plane_wave_z(),
            IncidentField::plane_wave_neg_z(),
        ];
        let solver = BemSolver::new();

        let batch = solver
            .solve_frequency_sweep_many(&problem, &frequencies, &incident_fields)
            .expect("frequency × RHS batch solve");
        assert_eq!(batch.len(), frequencies.len());

        for (&frequency, solutions) in frequencies.iter().zip(&batch) {
            assert_eq!(solutions.len(), incident_fields.len());
            for (field, batched_solution) in incident_fields.iter().zip(solutions) {
                let mut scalar_problem = problem.clone().with_incident_field(field.clone());
                let mut scalar_physics = PhysicsParams::try_new(
                    frequency,
                    problem.physics.speed_of_sound,
                    problem.physics.density,
                    false,
                )
                .expect("valid scalar frequency physics");
                scalar_physics.harmonic_factor = problem.physics.harmonic_factor;
                scalar_physics.pressure_factor =
                    scalar_physics.density * scalar_physics.omega * scalar_physics.harmonic_factor;
                scalar_problem.physics = scalar_physics;
                let scalar = solver
                    .solve(&scalar_problem)
                    .expect("scalar frequency solve");
                let maximum_difference = batched_solution
                    .surface_pressure
                    .iter()
                    .zip(&scalar.surface_pressure)
                    .map(|(batched, scalar)| (*batched - *scalar).norm())
                    .fold(0.0, f64::max);
                assert!(
                    maximum_difference < 1e-10,
                    "{frequency} Hz block batch mismatch {maximum_difference:e}"
                );
                assert_eq!(
                    batched_solution.physics.harmonic_factor, problem.physics.harmonic_factor,
                    "frequency batch must preserve the input phasor sign"
                );
            }
        }
    }

    #[test]
    fn scalar_frequency_sweep_preserves_input_phasor_sign_and_domain() {
        let mut problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
        problem.physics.harmonic_factor = -1.0;
        problem.physics.pressure_factor = -problem.physics.density * problem.physics.omega;
        problem.physics.tau = -1.0;
        problem
            .physics
            .validate()
            .expect("alternate internal phasor is valid");
        let solutions = BemSolver::new()
            .solve_frequency_sweep(&problem, &[100.0, 105.0])
            .expect("frequency sweep should accept a validated convention");
        assert!(solutions.iter().all(|solution| {
            solution.physics.harmonic_factor == -1.0 && solution.physics.tau == -1.0
        }));
    }

    #[test]
    fn frequency_multi_rhs_batch_rejects_invalid_frequency_and_preserves_empty_rhs_behavior() {
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
        let solver = BemSolver::new();
        assert!(matches!(
            solver.solve_frequency_sweep_many(
                &problem,
                &[100.0, f64::NAN],
                &[IncidentField::plane_wave_z()],
            ),
            Err(BemError::InvalidParameters(message)) if message.contains("frequency batch")
        ));
        let empty = solver
            .solve_frequency_sweep_many(&problem, &[100.0], &[])
            .expect("empty RHS batch preserves the frequency grid");
        assert_eq!(empty.len(), 1);
        assert!(empty[0].is_empty());
    }

    #[test]
    fn test_iterative_multi_rhs_matches_independent_solves() {
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
        let incident_fields = vec![
            IncidentField::plane_wave_z(),
            IncidentField::plane_wave_neg_z(),
        ];
        let solver = BemSolver::new()
            .with_solver_method(SolverMethod::Gmres)
            .with_max_iterations(500)
            .with_tolerance(1e-7);
        let batch = solver
            .solve_many(&problem, &incident_fields)
            .expect("iterative multi-RHS solve");
        assert_eq!(batch.len(), incident_fields.len());

        for (field, batched_solution) in incident_fields.iter().zip(batch) {
            let independent = solver
                .solve(&problem.clone().with_incident_field(field.clone()))
                .expect("independent iterative solve");
            let maximum_difference = batched_solution
                .surface_pressure
                .iter()
                .zip(independent.surface_pressure.iter())
                .map(|(batched, independent)| (*batched - *independent).norm())
                .fold(0.0, f64::max);
            assert!(
                maximum_difference < 1e-7,
                "iterative batch mismatch {maximum_difference:e}"
            );
        }
    }

    #[test]
    fn test_hmatrix_multi_rhs_uses_block_operator_path() {
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
        let incident_fields = vec![
            IncidentField::plane_wave_z(),
            IncidentField::plane_wave_neg_z(),
        ];
        let solver = BemSolver::new()
            .with_solver_method(SolverMethod::Gmres)
            .with_assembly_method(AssemblyMethod::Hmatrix)
            .with_max_iterations(500)
            .with_tolerance(1e-7)
            .with_compressed_retries(0);
        let batch = solver
            .solve_many(&problem, &incident_fields)
            .expect("H-matrix block multi-RHS solve");
        assert_eq!(batch.len(), incident_fields.len());

        for (field, batched_solution) in incident_fields.iter().zip(batch) {
            let independent = solver
                .solve(&problem.clone().with_incident_field(field.clone()))
                .expect("independent H-matrix solve");
            let maximum_difference = batched_solution
                .surface_pressure
                .iter()
                .zip(independent.surface_pressure.iter())
                .map(|(batched, independent)| (*batched - *independent).norm())
                .fold(0.0, f64::max);
            assert!(
                maximum_difference < 1e-6,
                "H-matrix block mismatch {maximum_difference:e}"
            );
        }
    }

    #[test]
    fn test_compressed_multi_rhs_rejects_non_gmres_without_fallback() {
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
        let solver = BemSolver::new().with_assembly_method(AssemblyMethod::Hmatrix);
        let error = solver
            .solve_many(
                &problem,
                &[
                    IncidentField::plane_wave_z(),
                    IncidentField::plane_wave_neg_z(),
                ],
            )
            .expect_err("compressed multi-RHS must not hide a scalar fallback");
        assert!(matches!(
            error,
            BemError::NotImplemented(message) if message.contains("block compressed solve")
        ));
    }

    #[test]
    fn test_bem_solver_dual_mass_barycentric_matches_unpreconditioned() {
        // Phase 1.2 dual-mass-on-barycentric end-to-end gate. Solve a
        // small Mie sphere with `BemPreconditioner::None` and again with
        // `BemPreconditioner::SurfaceMassDualBarycentric`; both must
        // converge to the same surface pressure (silent-failure detector
        // the spec calls out: "parity test against unpreconditioned
        // solve on small Mie cases must reach identical answer").
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);

        let baseline = BemSolver::new()
            .with_solver_method(SolverMethod::Gmres)
            .with_preconditioner(BemPreconditioner::None)
            .with_max_iterations(2000)
            .with_tolerance(1e-8)
            .solve(&problem)
            .expect("baseline GMRES solve");

        let dual = BemSolver::new()
            .with_solver_method(SolverMethod::Gmres)
            .with_preconditioner(BemPreconditioner::SurfaceMassDualBarycentric)
            .with_max_iterations(2000)
            .with_tolerance(1e-8)
            .solve(&problem)
            .expect("dual-mass GMRES solve");

        // Both must produce the same physical answer to round-off.
        // Use the surface pressure max as a single-number proxy — this
        // is what the existing CalderonV/W parity already gates.
        let p_base = baseline.max_surface_pressure();
        let p_dual = dual.max_surface_pressure();
        let rel = (p_base - p_dual).abs() / p_base.max(1e-30);
        assert!(
            rel < 1e-5,
            "DualMassBarycentric must reach the same |p|_max as unpreconditioned: \
             baseline {p_base:.6}, dual {p_dual:.6}, rel {rel:.3e}"
        );
    }

    #[test]
    fn test_dual_mass_barycentric_rejects_linear_basis() {
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
        let solver = BemSolver::new()
            .with_basis_order(BasisOrder::Linear)
            .with_preconditioner(BemPreconditioner::SurfaceMassDualBarycentric);
        let elements = solver.prepare_elements(&problem).unwrap();

        let error = match solver.build_preconditioner(
            &elements,
            &problem.mesh.nodes,
            &problem.physics,
            None,
            None,
        ) {
            Ok(_) => panic!("linear basis must be rejected by the typed contract"),
            Err(error) => error,
        };
        assert!(matches!(error, BemError::InvalidParameters(_)));
    }

    #[test]
    fn test_field_evaluation() {
        // Very small problem
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);

        let solver = BemSolver::new();
        let solution = solver.solve(&problem).unwrap();

        // Evaluate at a point outside the sphere
        let p = solution.evaluate_pressure(&[0.0, 0.0, 0.2]);

        // Should have some pressure (incident + scattered)
        assert!(p.norm() > 0.0);
    }

    #[test]
    fn test_bem_solver_linear_elements() {
        // Small problem with linear elements
        let problem = BemProblem::rigid_sphere_scattering_custom(
            0.1,   // radius
            100.0, // very low frequency for quick test
            343.0, 1.21, 4, // coarse mesh
            8,
        );

        let solver = BemSolver::new().with_basis_order(BasisOrder::Linear);
        let result = solver.solve(&problem);

        assert!(result.is_ok(), "Linear solver failed: {:?}", result.err());
        let solution = result.unwrap();
        // Linear: 3 DOFs per element (all Tri3)
        let n_elements = problem.mesh.elements.len();
        assert_eq!(solution.num_dofs(), n_elements * 3);
        assert!(solution.max_surface_pressure() > 0.0);
    }

    #[test]
    fn test_linear_basis_hmatrix_and_fmm_support() {
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);

        let hmatrix_solution = BemSolver::new()
            .with_basis_order(BasisOrder::Linear)
            .with_assembly_method(AssemblyMethod::Hmatrix)
            .with_solver_method(SolverMethod::Cgs)
            .solve(&problem)
            .expect("basis-aware H-matrix linear solve");
        assert_eq!(hmatrix_solution.num_dofs(), problem.mesh.elements.len() * 3);
        assert!(hmatrix_solution.max_surface_pressure().is_finite());

        for assembly_method in [
            AssemblyMethod::Slfmm,
            AssemblyMethod::Mlfmm,
            AssemblyMethod::DirectionalMlfmm,
        ] {
            let result = BemSolver::new()
                .with_basis_order(BasisOrder::Linear)
                .with_assembly_method(assembly_method)
                .with_solver_method(SolverMethod::Gmres)
                .solve(&problem);

            let solution = result.unwrap_or_else(|error| {
                panic!("{assembly_method:?} basis-aware linear solve failed: {error}")
            });
            assert_eq!(
                solution.num_dofs(),
                problem.mesh.elements.len() * 3,
                "{assembly_method:?} linear solve returned the wrong DOF count"
            );
            assert!(
                solution.max_surface_pressure().is_finite(),
                "{assembly_method:?} linear solve returned a non-finite pressure"
            );
        }
    }

    #[test]
    fn test_linear_vtu_export_maps_discontinuous_dofs_to_shared_nodes() {
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
        let solution = BemSolver::new()
            .with_basis_order(BasisOrder::Linear)
            .solve(&problem)
            .expect("linear solve");
        assert_ne!(
            solution.num_dofs(),
            problem.mesh.nodes.len(),
            "fixture must exercise discontinuous element-local P1 DOFs"
        );

        let path = std::env::temp_dir().join(format!(
            "sonium-bem-linear-export-{}.vtu",
            std::process::id()
        ));
        let result = solution.write_vtu(&problem.mesh, &path);
        let _ = std::fs::remove_file(&path);

        assert!(
            result.is_ok(),
            "linear VTU export must project discontinuous element-local values onto shared nodes: {result:?}"
        );
    }

    #[test]
    fn test_linear_field_evaluation() {
        // Small problem with linear elements
        let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);

        let solver = BemSolver::new().with_basis_order(BasisOrder::Linear);
        let solution = solver.solve(&problem).unwrap();

        // Evaluate at a point outside the sphere
        let p = solution.evaluate_pressure(&[0.0, 0.0, 0.2]);

        // Should have some pressure (incident + scattered)
        assert!(p.norm() > 0.0);
    }

    #[test]
    fn test_basis_order_default_is_constant() {
        let solver = BemSolver::new();
        assert_eq!(solver.basis_order, BasisOrder::Constant);
    }

    #[test]
    fn test_indirect_dirichlet_matches_direct_for_soft_sphere() {
        // The indirect single-layer formulation for Dirichlet (soft) BC
        // must give the same physical answer as the direct formulation.
        let problem_direct =
            BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8)
                .with_boundary_condition(BoundaryConditionType::Soft);

        let solver = BemSolver::new().with_solver_method(SolverMethod::Direct);

        // Direct solve (goes through the main solve path with Soft BC)
        let sol_direct = solver.solve(&problem_direct).unwrap();

        // Indirect solve (explicitly call the indirect path)
        let elements = solver.prepare_elements(&problem_direct).unwrap();
        let sol_indirect = solver
            .solve_soft_dirichlet_indirect(&problem_direct, elements)
            .unwrap();

        // Compare surface pressure magnitude at each element
        for (i, (d, ind)) in sol_direct
            .surface_pressure
            .iter()
            .zip(sol_indirect.surface_pressure.iter())
            .enumerate()
        {
            let diff = (d - ind).norm();
            let ref_val = d.norm().max(1e-30);
            let rel = diff / ref_val;
            assert!(
                rel < 1e-3,
                "Direct vs indirect Dirichlet disagree at elem {i}: direct={d}, indirect={ind}, rel={rel:.3e}"
            );
        }
    }
}

#[test]
fn higher_order_indirect_dirichlet_projects_galerkin_rhs() {
    let base = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 3, 4)
        .with_boundary_condition(BoundaryConditionType::Soft);
    let points = [[0.0, 0.0, 0.14], [0.08, 0.02, 0.12], [-0.04, 0.07, 0.13]];

    for (basis, continuity) in [
        (BasisOrder::Linear, BasisContinuity::Discontinuous),
        (BasisOrder::Quadratic, BasisContinuity::Discontinuous),
        (BasisOrder::Quadratic, BasisContinuity::Continuous),
    ] {
        let solver = BemSolver::new()
            .with_basis_order(basis)
            .with_basis_continuity(continuity)
            .with_solver_method(SolverMethod::Direct);
        let indirect = solver
            .solve(&base)
            .expect("higher-order indirect soft solve");
        assert_eq!(
            indirect.num_dofs(),
            solver
                .prepare_elements(&base)
                .expect("prepared elements")
                .iter()
                .filter(|element| !element.property.is_evaluation())
                .flat_map(|element| element.dof_addresses.iter().copied())
                .max()
                .map_or(0, |dof| dof + 1)
        );
        assert!(
            indirect
                .surface_pressure
                .iter()
                .all(|value| value.re.is_finite() && value.im.is_finite())
        );
        let fields = indirect
            .evaluate_pressure_field_checked(&points)
            .expect("higher-order indirect field evaluation");
        assert!(fields.iter().all(|field| field.p_total.norm().is_finite()));
        assert!(indirect.max_surface_pressure().is_finite());
    }
}

#[test]
fn test_dense_adjoint_multi_rhs_matches_direct_reference_and_rejects_bad_rhs() {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
    let solver = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::Direct);
    let elements = solver
        .prepare_elements(&problem)
        .expect("prepared elements");
    let num_dofs = count_dofs(&elements);
    let rhs_columns = (0..2)
        .map(|column| {
            (0..num_dofs)
                .map(|row| {
                    Complex64::new(
                        0.2 + (row + column) as f64 / 17.0,
                        -(((row + 2 * column) % 5) as f64) / 23.0,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let batch = solver
        .solve_adjoint_many(&problem, &rhs_columns)
        .expect("dense adjoint batch");
    assert_eq!(batch.len(), rhs_columns.len());
    assert!(batch.iter().all(|solution| {
        solution.num_dofs() == num_dofs && solution.relative_residual() <= 1.0e-10
    }));

    let ka = problem.ka();
    let assembly = solver
        .assemble_system_stabilized(
            &elements,
            &problem.mesh.nodes,
            &problem.physics,
            PhysicsParams::optimal_formulation_sign(ka),
            PhysicsParams::optimal_beta_scale(ka),
        )
        .expect("reference assembly");
    let AssemblyResult::Dense(matrix, _) = assembly else {
        panic!("Tbem must assemble a dense matrix");
    };
    let mut hermitian = DenseMatrix::<Complex64>::zeros(num_dofs, num_dofs);
    for row in 0..num_dofs {
        for column in 0..num_dofs {
            hermitian[[row, column]] = matrix[[column, row]].conj();
        }
    }
    for (solution, rhs) in batch.iter().zip(&rhs_columns) {
        let reference =
            sonium_solvers::direct::lu_solve(&hermitian, rhs).expect("reference adjoint solve");
        let maximum_difference = solution
            .coefficients()
            .iter()
            .zip(reference)
            .map(|(actual, expected)| (*actual - expected).norm())
            .fold(0.0, f64::max);
        assert!(
            maximum_difference <= 1.0e-10,
            "adjoint batch mismatch {maximum_difference:e}"
        );
    }

    let eval_points = [[0.0, 0.0, 0.25], [0.12, -0.03, 0.22]];
    let field = batch[0].evaluate_scattered_pressure_field(&eval_points);
    assert_eq!(field.len(), eval_points.len());
    assert!(
        field
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    assert_eq!(
        batch[0].evaluate_scattered_pressure(&eval_points[0]),
        field[0]
    );

    assert!(solver.solve_adjoint_many(&problem, &[]).unwrap().is_empty());
    assert!(
        solver
            .solve_adjoint_many(&problem, &[vec![Complex64::default(); num_dofs - 1]])
            .is_err()
    );
    assert!(
        solver
            .solve_adjoint_many(&problem, &[vec![Complex64::new(f64::NAN, 0.0); num_dofs]],)
            .is_err()
    );
}

#[test]
fn test_adjoint_driver_loads_alias_matches_generic_batch_contract() {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 3, 6);
    let solver = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::Direct);
    let num_dofs = count_dofs(
        &solver
            .prepare_elements(&problem)
            .expect("prepared elements"),
    );
    let rhs_columns = (0..2)
        .map(|column| {
            (0..num_dofs)
                .map(|row| {
                    Complex64::new(
                        0.13 + (row + column) as f64 / 17.0,
                        -0.04 + (row + 2 * column) as f64 / 23.0,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let named = solver
        .solve_adjoint_driver_loads(&problem, &rhs_columns)
        .expect("named adjoint driver loads");
    let generic = solver
        .solve_adjoint_many(&problem, &rhs_columns)
        .expect("generic adjoint batch");
    assert_eq!(named.len(), rhs_columns.len());
    assert_eq!(named.len(), generic.len());
    for (named_solution, generic_solution) in named.iter().zip(&generic) {
        assert!(named_solution.relative_residual() <= 1.0e-10);
        assert!(generic_solution.relative_residual() <= 1.0e-10);
        let error = named_solution
            .coefficients()
            .iter()
            .zip(generic_solution.coefficients())
            .map(|(actual, expected)| (*actual - *expected).norm())
            .fold(0.0, f64::max);
        assert!(error <= 1.0e-12, "named adjoint alias error {error:.3e}");
    }
}

#[test]
fn test_hmatrix_adjoint_multi_rhs_reports_exact_residuals() {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
    let solver = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Hmatrix)
        .with_solver_method(SolverMethod::Gmres)
        .with_max_iterations(500)
        .with_tolerance(1.0e-7)
        .with_compressed_retries(0);
    let elements = solver
        .prepare_elements(&problem)
        .expect("prepared elements");
    let num_dofs = count_dofs(&elements);
    let rhs_columns = (0..2)
        .map(|column| {
            (0..num_dofs)
                .map(|row| {
                    Complex64::new(
                        0.3 + (row + column) as f64 / 19.0,
                        (row + 3 * column) as f64 / 31.0,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let batch = solver
        .solve_adjoint_many(&problem, &rhs_columns)
        .expect("H-matrix adjoint batch");
    assert_eq!(batch.len(), rhs_columns.len());
    assert!(
        batch
            .iter()
            .all(|solution| solution.relative_residual() <= 1.0e-6)
    );
}

#[test]
fn test_adjoint_frequency_sweep_many_matches_individual_solves_and_rejects_bad_shape() {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
    let solver = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::Direct);
    let num_dofs = count_dofs(
        &solver
            .prepare_elements(&problem)
            .expect("prepared elements"),
    );
    let make_rhs = |offset: usize| {
        (0..2)
            .map(|column| {
                (0..num_dofs)
                    .map(|row| {
                        Complex64::new(
                            0.1 + (row + column + offset) as f64 / 23.0,
                            -(((row + 2 * column + offset) % 7) as f64) / 29.0,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    let frequencies = [100.0, 120.0];
    let rhs_batches = vec![make_rhs(0), make_rhs(3)];

    let batches = solver
        .solve_adjoint_frequency_sweep_many(&problem, &frequencies, &rhs_batches)
        .expect("frequency adjoint batches");
    assert_eq!(batches.len(), frequencies.len());
    assert!(batches.iter().all(|batch| {
        batch.len() == 2
            && batch
                .iter()
                .all(|solution| solution.relative_residual() <= 1.0e-10)
    }));

    for (frequency_index, (&frequency, rhs_columns)) in
        frequencies.iter().zip(&rhs_batches).enumerate()
    {
        let mut at_frequency = problem.clone();
        let mut physics = PhysicsParams::try_new(
            frequency,
            problem.physics.speed_of_sound,
            problem.physics.density,
            problem.physics.tau < 0.0,
        )
        .expect("valid frequency physics");
        physics.harmonic_factor = problem.physics.harmonic_factor;
        physics.pressure_factor = physics.density * physics.omega * physics.harmonic_factor;
        at_frequency.physics = physics;
        let individual = solver
            .solve_adjoint_many(&at_frequency, rhs_columns)
            .expect("individual frequency adjoint");
        for (batch_solution, individual_solution) in batches[frequency_index].iter().zip(individual)
        {
            let coefficient_error = batch_solution
                .coefficients()
                .iter()
                .zip(individual_solution.coefficients())
                .map(|(actual, expected)| (*actual - *expected).norm())
                .fold(0.0, f64::max);
            assert!(
                coefficient_error <= 1.0e-12,
                "coefficient error {coefficient_error}"
            );
        }
    }

    assert!(matches!(
        solver.solve_adjoint_frequency_sweep_many(&problem, &[100.0], &rhs_batches),
        Err(BemError::InvalidParameters(message)) if message.contains("RHS batches")
    ));
    assert!(matches!(
        solver.solve_adjoint_frequency_sweep_many(&problem, &[f64::NAN, 120.0], &rhs_batches),
        Err(BemError::InvalidParameters(message)) if message.contains("finite")
    ));
}

#[test]
fn test_forward_driver_load_many_compressed_backends() {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
    let direct = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::Direct);
    let dimension = direct
        .solve(&problem)
        .expect("dense reference solve")
        .surface_pressure
        .len();
    let rhs_columns = (0..2)
        .map(|column| {
            (0..dimension)
                .map(|row| {
                    Complex64::new(
                        0.15 + (row + column) as f64 / 29.0,
                        -(((row + 2 * column) % 6) as f64) / 31.0,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let dense = direct
        .solve_rhs_many(&problem, &rhs_columns)
        .expect("dense driver loads");
    for assembly_method in [
        AssemblyMethod::Hmatrix,
        AssemblyMethod::Slfmm,
        AssemblyMethod::Mlfmm,
    ] {
        let solver = BemSolver::new()
            .with_assembly_method(assembly_method)
            .with_solver_method(SolverMethod::Gmres)
            .with_max_iterations(500)
            .with_tolerance(1.0e-7)
            .with_compressed_retries(0);
        let compressed = solver
            .solve_rhs_many(&problem, &rhs_columns)
            .unwrap_or_else(|error| panic!("{assembly_method:?} forward loads: {error}"));
        assert_eq!(compressed.len(), rhs_columns.len());
        for (actual, reference) in compressed.iter().zip(dense.iter()) {
            assert!(actual.relative_residual() <= 1.0e-6);
            assert!(
                actual
                    .coefficients()
                    .iter()
                    .all(|value| value.re.is_finite() && value.im.is_finite())
            );
            let coefficient_error = actual
                .coefficients()
                .iter()
                .zip(reference.coefficients())
                .map(|(actual, reference)| (*actual - *reference).norm())
                .fold(0.0, f64::max);
            assert!(
                coefficient_error <= 5.0e-5,
                "{assembly_method:?} coefficient error {coefficient_error:e}"
            );
            let actual_field =
                actual.evaluate_scattered_pressure_field(&[[0.0, 0.0, 0.3], [0.2, -0.1, 0.4]]);
            let reference_field =
                reference.evaluate_scattered_pressure_field(&[[0.0, 0.0, 0.3], [0.2, -0.1, 0.4]]);
            let field_error = actual_field
                .iter()
                .zip(reference_field.iter())
                .map(|(actual, reference)| (*actual - *reference).norm())
                .fold(0.0, f64::max);
            assert!(
                field_error <= 5.0e-5,
                "{assembly_method:?} field error {field_error:e}"
            );
        }
    }
}

#[test]
fn test_forward_frequency_sweep_many_compressed_backend_dispatch() {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
    let direct = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::Direct);
    let dimension = direct
        .solve(&problem)
        .expect("dense reference solve")
        .surface_pressure
        .len();
    let frequencies = [100.0, 110.0];
    let rhs_columns = (0..2)
        .map(|column| {
            (0..dimension)
                .map(|row| {
                    Complex64::new(
                        0.15 + (row + column) as f64 / 29.0,
                        -(((row + 2 * column) % 6) as f64) / 31.0,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let rhs_batches = vec![rhs_columns.clone(), rhs_columns.clone()];
    let reference = direct
        .solve_rhs_frequency_sweep_many(&problem, &frequencies, &rhs_batches)
        .expect("dense frequency-batch reference solve");

    for assembly_method in [
        AssemblyMethod::Hmatrix,
        AssemblyMethod::Slfmm,
        AssemblyMethod::Mlfmm,
    ] {
        let solver = BemSolver::new()
            .with_assembly_method(assembly_method)
            .with_solver_method(SolverMethod::Gmres)
            .with_max_iterations(500)
            .with_tolerance(1.0e-7)
            .with_compressed_retries(0);
        let batches = solver
            .solve_rhs_frequency_sweep_many(&problem, &frequencies, &rhs_batches)
            .unwrap_or_else(|error| {
                panic!("{assembly_method:?} forward frequency batch failed: {error}")
            });
        assert_eq!(batches.len(), frequencies.len());
        for (batch, reference_batch) in batches.iter().zip(reference.iter()) {
            assert_eq!(batch.len(), rhs_columns.len());
            for (solution, reference_solution) in batch.iter().zip(reference_batch.iter()) {
                assert!(
                    solution.relative_residual() <= 1.0e-6,
                    "{assembly_method:?} residual {}",
                    solution.relative_residual()
                );
                let coefficient_error = solution
                    .coefficients()
                    .iter()
                    .zip(reference_solution.coefficients())
                    .map(|(actual, expected)| (*actual - *expected).norm())
                    .fold(0.0, f64::max);
                assert!(
                    coefficient_error <= 5.0e-5,
                    "{assembly_method:?} coefficient error {coefficient_error:e}"
                );
            }
        }
    }
}

#[test]
fn test_adjoint_frequency_sweep_many_compressed_backend_dispatch() {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
    let dimension = count_dofs(
        &BemSolver::new()
            .prepare_elements(&problem)
            .expect("prepared elements"),
    );
    let rhs = (0..dimension)
        .map(|row| Complex64::new(0.25 + row as f64 / 37.0, -((row % 5) as f64) / 41.0))
        .collect::<Vec<_>>();
    let rhs_batches = vec![vec![rhs]];

    for assembly_method in [
        AssemblyMethod::Hmatrix,
        AssemblyMethod::Slfmm,
        AssemblyMethod::Mlfmm,
    ] {
        let solver = BemSolver::new()
            .with_assembly_method(assembly_method)
            .with_solver_method(SolverMethod::Gmres)
            .with_max_iterations(500)
            .with_tolerance(1.0e-7)
            .with_compressed_retries(0);
        let batches = solver
            .solve_adjoint_frequency_sweep_many(&problem, &[100.0], &rhs_batches)
            .unwrap_or_else(|error| {
                panic!("{assembly_method:?} compressed adjoint batch failed: {error}")
            });
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 1);
        assert!(
            batches[0][0].relative_residual() <= 1.0e-6,
            "{assembly_method:?} residual {}",
            batches[0][0].relative_residual()
        );
    }
}

#[test]
fn test_forward_driver_load_many_matches_dense_reference_and_rejects_bad_rhs() {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
    let solver = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::Direct);
    let elements = solver
        .prepare_elements(&problem)
        .expect("prepared elements");
    let num_dofs = count_dofs(&elements);
    let rhs_columns = (0..2)
        .map(|column| {
            (0..num_dofs)
                .map(|row| {
                    Complex64::new(
                        0.15 + (row + column) as f64 / 29.0,
                        -(((row + 2 * column) % 6) as f64) / 31.0,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let solutions = solver
        .solve_rhs_many(&problem, &rhs_columns)
        .expect("forward driver loads");
    assert_eq!(solutions.len(), rhs_columns.len());
    assert!(solutions.iter().all(|solution| {
        solution.num_dofs() == num_dofs && solution.relative_residual() <= 1.0e-10
    }));

    let ka = problem.ka();
    let AssemblyResult::Dense(matrix, _) = solver
        .assemble_system_stabilized(
            &elements,
            &problem.mesh.nodes,
            &problem.physics,
            PhysicsParams::optimal_formulation_sign(ka),
            PhysicsParams::optimal_beta_scale(ka),
        )
        .expect("dense reference assembly")
    else {
        panic!("expected dense reference assembly");
    };
    let factor = lu_factorize(&matrix).expect("reference factorization");
    for (solution, rhs) in solutions.iter().zip(&rhs_columns) {
        let expected = factor.solve(rhs).expect("reference solve");
        let coefficient_error = solution
            .coefficients()
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (*actual - expected).norm())
            .fold(0.0, f64::max);
        assert!(
            coefficient_error <= 1.0e-12,
            "coefficient error {coefficient_error}"
        );
    }
    let field = solutions[0].evaluate_scattered_pressure_field(&[[0.0, 0.0, 0.3]]);
    assert_eq!(field.len(), 1);
    assert!(field[0].re.is_finite() && field[0].im.is_finite());
    assert_eq!(
        solver
            .solve_driver_loads(&problem, &rhs_columns)
            .expect("driver-load alias")
            .len(),
        rhs_columns.len()
    );
    let rhs_frequency_batches = vec![rhs_columns.clone(), rhs_columns.clone()];
    let frequency_batches = solver
        .solve_rhs_frequency_sweep_many(&problem, &[100.0, 120.0], &rhs_frequency_batches)
        .expect("frequency driver-load batches");
    assert_eq!(frequency_batches.len(), 2);
    assert!(frequency_batches.iter().all(|batch| {
        batch.len() == rhs_columns.len()
            && batch
                .iter()
                .all(|solution| solution.relative_residual() <= 1.0e-10)
    }));

    assert!(matches!(
        solver.solve_rhs_many(&problem, &[vec![Complex64::default(); num_dofs - 1]]),
        Err(BemError::InvalidParameters(message)) if message.contains("forward RHS")
    ));
    assert!(matches!(
        solver.solve_rhs_many(&problem, &[vec![Complex64::new(f64::NAN, 0.0); num_dofs]]),
        Err(BemError::InvalidParameters(message)) if message.contains("non-finite")
    ));
    assert!(matches!(
        solver.solve_rhs_frequency_sweep_many(&problem, &[100.0], &rhs_frequency_batches),
        Err(BemError::InvalidParameters(message)) if message.contains("RHS batches")
    ));
}

#[test]
fn solver_quadrature_policy_is_enforced_before_dense_and_compressed_assembly() {
    use crate::core::integration::QuadratureParams;

    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 100.0, 343.0, 1.21, 4, 8);
    let permissive = TbemQuadraturePolicy {
        regular_coarse_order: 3,
        regular_fine_order: 4,
        singular_coarse: QuadratureParams::default(),
        singular_fine: QuadratureParams::high_accuracy(),
        absolute_tolerance: 1.0e6,
        relative_tolerance: 1.0e6,
    };
    let dense = BemSolver::new()
        .with_quadrature_policy(permissive)
        .solve(&problem)
        .expect("permissive quadrature policy should admit dense assembly");
    assert!(dense.surface_pressure.iter().all(|value| value.is_finite()));

    let strict = TbemQuadraturePolicy {
        regular_coarse_order: 1,
        regular_fine_order: 2,
        singular_coarse: QuadratureParams::default(),
        singular_fine: QuadratureParams::default(),
        absolute_tolerance: 0.0,
        relative_tolerance: 0.0,
    };
    let dense_error = BemSolver::new()
        .with_quadrature_policy(strict)
        .solve(&problem)
        .expect_err("zero-budget quadrature policy must reject dense assembly");
    assert!(dense_error.to_string().contains("quadrature admission"));

    for assembly_method in [
        AssemblyMethod::Hmatrix,
        AssemblyMethod::Slfmm,
        AssemblyMethod::Mlfmm,
    ] {
        let compressed_error = BemSolver::new()
            .with_assembly_method(assembly_method)
            .with_solver_method(SolverMethod::Gmres)
            .with_quadrature_policy(strict)
            .solve(&problem)
            .expect_err("zero-budget quadrature policy must reject compressed assembly");
        assert!(
            compressed_error
                .to_string()
                .contains("quadrature admission")
        );
    }

    let invalid_policy = BemSolver::new().with_quadrature_policy(TbemQuadraturePolicy {
        regular_coarse_order: 0,
        ..permissive
    });
    assert!(matches!(
        invalid_policy.validate(),
        Err(BemError::InvalidParameters(message)) if message.contains("regular orders")
    ));

    let higher_order = BemSolver::new()
        .with_basis_order(BasisOrder::Quadratic)
        .with_quadrature_policy(permissive);
    higher_order
        .validate()
        .expect("higher-order basis-weighted quadrature admission is supported");
}

#[test]
fn hierarchical_ilu_preconditioner_fails_closed_without_an_fmm_operator() {
    let solver = BemSolver::new().with_preconditioner(BemPreconditioner::HierarchicalIlu);
    let error = solver
        .build_preconditioner(
            &[],
            &[],
            &PhysicsParams::new(100.0, 343.0, 1.21, false),
            None,
            None,
        )
        .err()
        .expect("hierarchical ILU must not silently degrade to identity");
    assert!(matches!(
        error,
        BemError::NotImplemented(message) if message.contains("requires SLFMM or MLFMM")
    ));
}
