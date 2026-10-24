//! Planar `A_z` magnetoquasistatics for rotating machines.
//!
//! Solves, per unit stack depth on a 2D cross-section,
//!
//! ```text
//! div(nu(|B|) grad A_z) = -J_z + div(H_c x z) - sigma * dA_z/dt,
//! B = curl(A_z * z) = [dA_z/dy, -dA_z/dx].
//! ```
//!
//! The weak residual at a quadrature point is
//! `H(B) . curl(v z) - J_z v + sigma (A_z - A_z_prev)/dt v`, with the
//! consistent material tangent from [`AzMaterial::h_and_tangent`]. The
//! transient term is backward Euler; it is absent unless a
//! [`TransientStep`] is supplied. Isolated solid conductors additionally
//! need a zero-net-current constraint per body, which the winding layer
//! adds on top of this field problem.

// Rust guideline compliant 2026-02-21

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use sonium_solvers::{
    ForcingTerm, InexactNewtonConfig, LinearSolveBackend, LinearSolveConfig, LinearSolveFallback,
    LinearSolverType, LinearSystemAssumption, NewtonConfig, NewtonError, TrustRegionConfig,
    newton_raphson_inexact, solve_real, solve_with_backend_and_assumption,
};

use crate::assembly::TripletMatrix;
use crate::basis::PolynomialDegree;
use crate::forms::{BilinearForm, LinearForm, LocalMatrix, LocalVector, WeightedLaplacianForm};
use crate::mesh::Mesh;
use crate::quadrature::QuadratureRule;
use crate::spaces::{BasisTabulation, FunctionSpace, H1Space};

use super::materials::AzMaterial;

/// Spatially varying axial current density `J_z(x, y)` in A/m^2.
///
/// Used by manufactured-solution tests; production motor problems use
/// per-marker constants through
/// [`AzProblem::current_density_a_per_m2`].
pub type CurrentDensityFn = Arc<dyn Fn(f64, f64) -> f64 + Send + Sync>;

/// Backward-Euler state for the `sigma * dA_z/dt` eddy term.
#[derive(Debug, Clone)]
pub struct TransientStep {
    /// Vector potential at the previous time level.
    pub previous_potential: Vec<f64>,
    /// Time step in seconds.
    pub dt_s: f64,
    /// Per-block axial field offset `E0` in V/m: the initial guess on
    /// input to [`solve_az`], and the solved value when a step is
    /// passed to [`AzProblem::quadrature_state`] after a solve. Empty
    /// means zero for every block.
    pub conductor_e0_v_per_m: Vec<f64>,
}

/// A solid conductor body with a prescribed net axial current.
///
/// In a transient step the body carries one extra unknown, the uniform
/// axial electric field offset `E0`, with current density
/// `J = sigma * (E0 - dA_z/dt)` and the constraint
/// `integral_body J dA = target_current_a` (zero for floating bodies
/// such as magnets and shafts). Conductivity comes from each marker's
/// [`AzMaterial`]. Blocks are inactive in magnetostatic solves.
#[derive(Debug, Clone, PartialEq)]
pub struct ConductorBlock {
    /// Cell markers forming the body.
    pub markers: Vec<i32>,
    /// Prescribed net axial current in amperes.
    pub target_current_a: f64,
}

/// A planar `A_z` problem on a marker-tagged 2D mesh.
pub struct AzProblem {
    /// Cross-section mesh in metres; every cell marker needs a material.
    pub mesh: Mesh,
    /// Region materials by cell marker.
    pub materials: BTreeMap<i32, AzMaterial>,
    /// Imposed axial current density by cell marker, in A/m^2.
    pub current_density_a_per_m2: BTreeMap<i32, f64>,
    /// Optional spatially varying current density; overrides the map.
    pub current_density_fn: Option<CurrentDensityFn>,
    /// H1 polynomial degree; the mesh connectivity must match it.
    pub degree: PolynomialDegree,
    /// Quadrature order used for every element.
    pub quadrature_order: usize,
    /// Stack length in metres; scales energies, flux linkage and torque.
    pub stack_length_m: f64,
    /// Solid-conductor current constraints (transient solves only).
    pub conductor_blocks: Vec<ConductorBlock>,
    /// Per-element irreversible-demagnetisation factor in `[0, 1]`,
    /// scaling magnet coercivity element by element (on top of each
    /// [`super::materials::MagnetLaw`]'s own factor). Empty means
    /// every element at full strength; otherwise one entry per mesh
    /// element. Ignored on non-magnet elements.
    pub element_magnet_scale: Vec<f64>,
}

impl AzProblem {
    /// Validates materials, sources and discretisation parameters.
    ///
    /// # Errors
    ///
    /// Returns [`AzError`] for a missing material, non-finite source,
    /// non-positive stack length, or zero quadrature order.
    pub fn validate(&self) -> Result<(), AzError> {
        if self.quadrature_order == 0 {
            return Err(AzError::InvalidQuadratureOrder);
        }
        if !self.stack_length_m.is_finite() || self.stack_length_m <= 0.0 {
            return Err(AzError::InvalidStackLength);
        }
        if self.mesh.dimension != 2 {
            return Err(AzError::InvalidMeshDimension {
                actual: self.mesh.dimension,
            });
        }
        for (element_index, element) in self.mesh.elements.iter().enumerate() {
            if !self.materials.contains_key(&element.marker) {
                return Err(AzError::MissingMaterial {
                    element_index,
                    marker: element.marker,
                });
            }
        }
        for (&marker, &source) in &self.current_density_a_per_m2 {
            if !source.is_finite() {
                return Err(AzError::InvalidCurrentDensity { marker });
            }
        }
        if !self.element_magnet_scale.is_empty() {
            if self.element_magnet_scale.len() != self.mesh.num_elements() {
                return Err(AzError::DimensionMismatch {
                    expected: self.mesh.num_elements(),
                    actual: self.element_magnet_scale.len(),
                });
            }
            if self
                .element_magnet_scale
                .iter()
                .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
            {
                return Err(AzError::InvalidMagnetScale);
            }
        }
        let mut seen_markers: std::collections::HashSet<i32> = std::collections::HashSet::new();
        for block in &self.conductor_blocks {
            if block.markers.is_empty() || !block.target_current_a.is_finite() {
                return Err(AzError::InvalidConductorBlock);
            }
            for marker in &block.markers {
                if !seen_markers.insert(*marker) {
                    return Err(AzError::InvalidConductorBlock);
                }
            }
        }
        Ok(())
    }

    /// Number of global `A_z` degrees of freedom.
    pub fn num_dofs(&self) -> usize {
        H1Space::new(self.degree).num_global_dofs(&self.mesh)
    }

    /// Whether conductor blocks add `E0` unknowns for this step.
    pub fn blocks_active(&self, transient: Option<&TransientStep>) -> bool {
        transient.is_some() && !self.conductor_blocks.is_empty()
    }

    /// Total unknowns (`A_z` DOFs plus active conductor `E0` fields).
    pub fn num_unknowns(&self, transient: Option<&TransientStep>) -> usize {
        self.num_dofs()
            + if self.blocks_active(transient) {
                self.conductor_blocks.len()
            } else {
                0
            }
    }

    fn block_of_marker(&self) -> std::collections::HashMap<i32, usize> {
        let mut map = std::collections::HashMap::new();
        for (index, block) in self.conductor_blocks.iter().enumerate() {
            for marker in &block.markers {
                map.insert(*marker, index);
            }
        }
        map
    }

    /// Whether every material is affine, so one linear solve is exact.
    pub fn is_linear(&self) -> bool {
        self.materials.values().all(AzMaterial::is_linear)
    }

    fn current_density(&self, marker: i32, position: [f64; 2]) -> f64 {
        if let Some(function) = &self.current_density_fn {
            function(position[0], position[1])
        } else {
            self.current_density_a_per_m2
                .get(&marker)
                .copied()
                .unwrap_or(0.0)
        }
    }

    /// Assembles the nonlinear residual and exact tangent at `potential`.
    ///
    /// `load_factor` in `[0, 1]` scales imposed currents and magnet
    /// coercivity for continuation. The eddy term is never scaled.
    ///
    /// # Errors
    ///
    /// Returns [`AzError`] on validation failure or a dimension mismatch.
    pub fn assemble(
        &self,
        potential: &[f64],
        load_factor: f64,
        transient: Option<&TransientStep>,
    ) -> Result<(Vec<f64>, TripletMatrix<f64>), AzError> {
        self.assemble_impl(potential, load_factor, transient, true)
    }

    /// Assembles only the residual (the chord-Newton inner loop).
    ///
    /// # Errors
    ///
    /// Returns [`AzError`] on validation failure or a dimension mismatch.
    pub fn assemble_residual(
        &self,
        potential: &[f64],
        load_factor: f64,
        transient: Option<&TransientStep>,
    ) -> Result<Vec<f64>, AzError> {
        Ok(self
            .assemble_impl(potential, load_factor, transient, false)?
            .0)
    }

    fn assemble_impl(
        &self,
        potential: &[f64],
        load_factor: f64,
        transient: Option<&TransientStep>,
        want_tangent: bool,
    ) -> Result<(Vec<f64>, TripletMatrix<f64>), AzError> {
        self.validate()?;
        let num_dofs = self.num_dofs();
        let num_unknowns = self.num_unknowns(transient);
        if potential.len() != num_unknowns {
            return Err(AzError::DimensionMismatch {
                expected: num_unknowns,
                actual: potential.len(),
            });
        }
        if !load_factor.is_finite() || !(0.0..=1.0).contains(&load_factor) {
            return Err(AzError::InvalidLoadFactor);
        }
        if let Some(step) = transient {
            if step.previous_potential.len() != num_dofs {
                return Err(AzError::DimensionMismatch {
                    expected: num_dofs,
                    actual: step.previous_potential.len(),
                });
            }
            if !step.dt_s.is_finite() || step.dt_s <= 0.0 {
                return Err(AzError::InvalidTimeStep);
            }
        }
        let space = H1Space::new(self.degree);
        // Parallel assembly over element chunks (rayon is always on in
        // this workspace): each chunk accumulates a private residual
        // and triplet list, merged afterwards in chunk order, so the
        // result is deterministic. Serial for small problems.
        let num_elements = self.mesh.num_elements();
        let blocks_active = self.blocks_active(transient);
        let block_of = self.block_of_marker();
        let block_dt = transient.map_or(0.0, |step| step.dt_s);
        let assemble_range = |range: &[usize]| -> (Vec<f64>, TripletMatrix<f64>) {
            let mut residual = vec![0.0; num_unknowns];
            let mut tangent = TripletMatrix::new(num_unknowns, num_unknowns);
            for &element_index in range {
                let element = &self.mesh.elements[element_index];
                let material = &self.materials[&element.marker];
                let sigma = material.conductivity_s_per_m();
                let sigma_over_dt = transient.map_or(0.0, |step| sigma / step.dt_s);
                let block_index = if blocks_active {
                    block_of.get(&element.marker).copied()
                } else {
                    None
                };
                let e0 = block_index.map_or(0.0, |block| potential[num_dofs + block]);
                let dofs = space.local_to_global(&self.mesh, element_index);
                let quadrature = QuadratureRule::new(element.element_type, self.quadrature_order);
                for point in quadrature.iter() {
                    let tab = space.tabulate(
                        &self.mesh,
                        element_index,
                        point.xi(),
                        point.eta(),
                        point.zeta(),
                    );
                    let position = [tab.physical_coords[0], tab.physical_coords[1]];
                    let source = load_factor * self.current_density(element.marker, position);
                    let weight = tab.det_j * point.weight;
                    // B = [dA/dy, -dA/dx]; basis vector per local DOF.
                    let basis_b: Vec<[f64; 2]> = (0..dofs.len())
                        .map(|local| [tab.grad(local, 1), -tab.grad(local, 0)])
                        .collect();
                    let mut b = [0.0; 2];
                    let mut a = 0.0;
                    let mut a_previous = 0.0;
                    for (local, &dof) in dofs.iter().enumerate() {
                        b[0] += potential[dof] * basis_b[local][0];
                        b[1] += potential[dof] * basis_b[local][1];
                        a += potential[dof] * tab.value(local);
                        if let Some(step) = transient {
                            a_previous += step.previous_potential[dof] * tab.value(local);
                        }
                    }
                    let eddy_density = sigma_over_dt * (a - a_previous) - sigma * e0;
                    // Soft laws ignore the load factor; for magnets it
                    // scales coercivity only, so the per-element demag
                    // factor composes with continuation here.
                    let magnet_scale = self
                        .element_magnet_scale
                        .get(element_index)
                        .copied()
                        .unwrap_or(1.0);
                    let (h, constitutive_tangent) =
                        material.h_and_tangent(b, load_factor * magnet_scale);
                    if let Some(block) = block_index {
                        // dt-scaled net-current row: dt*(int J - I_target),
                        // accumulated here without the constant term.
                        residual[num_dofs + block] +=
                            (sigma * e0 * block_dt - sigma * (a - a_previous)) * weight;
                        if want_tangent {
                            for (local_j, &global_j) in dofs.iter().enumerate() {
                                tangent.push(
                                    num_dofs + block,
                                    global_j,
                                    -sigma * tab.value(local_j) * weight,
                                );
                            }
                            tangent.push(
                                num_dofs + block,
                                num_dofs + block,
                                sigma * block_dt * weight,
                            );
                        }
                    }
                    for (local_i, &global_i) in dofs.iter().enumerate() {
                        let delta_b_i = basis_b[local_i];
                        residual[global_i] += (h[0] * delta_b_i[0] + h[1] * delta_b_i[1]
                            - source * tab.value(local_i)
                            + eddy_density * tab.value(local_i))
                            * weight;
                        if want_tangent && let Some(block) = block_index {
                            tangent.push(
                                global_i,
                                num_dofs + block,
                                -sigma * tab.value(local_i) * weight,
                            );
                        }
                        if !want_tangent {
                            continue;
                        }
                        for (local_j, &global_j) in dofs.iter().enumerate() {
                            let delta_b_j = basis_b[local_j];
                            let db_h = [
                                constitutive_tangent[0][0] * delta_b_j[0]
                                    + constitutive_tangent[0][1] * delta_b_j[1],
                                constitutive_tangent[1][0] * delta_b_j[0]
                                    + constitutive_tangent[1][1] * delta_b_j[1],
                            ];
                            tangent.push(
                                global_i,
                                global_j,
                                (delta_b_i[0] * db_h[0]
                                    + delta_b_i[1] * db_h[1]
                                    + sigma_over_dt * tab.value(local_i) * tab.value(local_j))
                                    * weight,
                            );
                        }
                    }
                }
            }
            (residual, tangent)
        };
        let indices: Vec<usize> = (0..num_elements).collect();
        let mut residual = vec![0.0; num_unknowns];
        let mut tangent = TripletMatrix::new(num_unknowns, num_unknowns);
        if num_elements < 1024 {
            let (r, t) = assemble_range(&indices);
            residual = r;
            tangent = t;
        } else {
            use rayon::prelude::*;
            let chunk = (num_elements / (4 * rayon::current_num_threads().max(1))).max(64);
            let parts: Vec<(Vec<f64>, TripletMatrix<f64>)> =
                indices.par_chunks(chunk).map(assemble_range).collect();
            for (part_residual, part_tangent) in parts {
                for (acc, part) in residual.iter_mut().zip(part_residual) {
                    *acc += part;
                }
                tangent.rows.extend(part_tangent.rows);
                tangent.cols.extend(part_tangent.cols);
                tangent.values.extend(part_tangent.values);
            }
        }
        if blocks_active {
            for (block, conductor) in self.conductor_blocks.iter().enumerate() {
                residual[num_dofs + block] -= block_dt * conductor.target_current_a;
            }
        }
        Ok((residual, tangent))
    }

    /// Evaluates the solved field at every quadrature point.
    ///
    /// # Errors
    ///
    /// Returns [`AzError`] on validation failure or a dimension mismatch.
    pub fn quadrature_state(
        &self,
        potential: &[f64],
        transient: Option<&TransientStep>,
    ) -> Result<Vec<AzQuadraturePoint>, AzError> {
        self.validate()?;
        if potential.len() != self.num_dofs() {
            return Err(AzError::DimensionMismatch {
                expected: self.num_dofs(),
                actual: potential.len(),
            });
        }
        let space = H1Space::new(self.degree);
        let block_of = self.block_of_marker();
        let mut state = Vec::new();
        for element_index in 0..self.mesh.num_elements() {
            let element = &self.mesh.elements[element_index];
            let material = &self.materials[&element.marker];
            let dofs = space.local_to_global(&self.mesh, element_index);
            let quadrature = QuadratureRule::new(element.element_type, self.quadrature_order);
            for (quadrature_index, point) in quadrature.iter().enumerate() {
                let tab = space.tabulate(
                    &self.mesh,
                    element_index,
                    point.xi(),
                    point.eta(),
                    point.zeta(),
                );
                let mut b = [0.0; 2];
                let mut a = 0.0;
                let mut a_previous = 0.0;
                for (local, &dof) in dofs.iter().enumerate() {
                    b[0] += potential[dof] * tab.grad(local, 1);
                    b[1] -= potential[dof] * tab.grad(local, 0);
                    a += potential[dof] * tab.value(local);
                    if let Some(step) = transient {
                        a_previous += step.previous_potential[dof] * tab.value(local);
                    }
                }
                let position = [tab.physical_coords[0], tab.physical_coords[1]];
                let imposed = self.current_density(element.marker, position);
                let eddy = transient.map_or(0.0, |step| {
                    let e0 = block_of
                        .get(&element.marker)
                        .and_then(|block| step.conductor_e0_v_per_m.get(*block))
                        .copied()
                        .unwrap_or(0.0);
                    material.conductivity_s_per_m() * (e0 - (a - a_previous) / step.dt_s)
                });
                let magnet_scale = self
                    .element_magnet_scale
                    .get(element_index)
                    .copied()
                    .unwrap_or(1.0);
                let (h, _) = material.h_and_tangent(b, magnet_scale);
                let magnitude = b[0].hypot(b[1]);
                state.push(AzQuadraturePoint {
                    element_index,
                    quadrature_index,
                    marker: element.marker,
                    position_m: position,
                    integration_weight_m2: tab.det_j * point.weight,
                    vector_potential_wb_per_m: a,
                    flux_density_t: b,
                    field_strength_a_per_m: h,
                    reluctivity_m_per_h: material.reluctivity(magnitude),
                    differential_reluctivity_m_per_h: material.differential_reluctivity(magnitude),
                    energy_density_j_per_m3: material.energy_density_j_per_m3(b),
                    coenergy_density_j_per_m3: material.coenergy_density_j_per_m3(b),
                    imposed_current_density_a_per_m2: imposed,
                    eddy_current_density_a_per_m2: eddy,
                });
            }
        }
        Ok(state)
    }

    /// Total magnetic energy and coenergy over the stack, in joules.
    pub fn energy_report(&self, state: &[AzQuadraturePoint]) -> AzEnergyReport {
        let (mut energy, mut coenergy) = (0.0, 0.0);
        for point in state {
            let volume = point.integration_weight_m2 * self.stack_length_m;
            energy += point.energy_density_j_per_m3 * volume;
            coenergy += point.coenergy_density_j_per_m3 * volume;
        }
        AzEnergyReport {
            energy_j: energy,
            coenergy_j: coenergy,
        }
    }

    /// Flux linkage of one stranded coil region, in weber-turns.
    ///
    /// `psi = turns * L_stack * mean(A_z over the region)`. A phase
    /// winding sums this over its go regions and subtracts its return
    /// regions (or, equivalently, weights each region by the sign of its
    /// imposed current density).
    ///
    /// # Errors
    ///
    /// Returns [`AzError::MissingCoilRegion`] when no quadrature point
    /// carries `coil_marker`, and [`AzError::InvalidTurns`] for
    /// non-finite turns.
    pub fn coil_flux_linkage_wb_turn(
        &self,
        state: &[AzQuadraturePoint],
        coil_marker: i32,
        turns: f64,
    ) -> Result<f64, AzError> {
        if !turns.is_finite() {
            return Err(AzError::InvalidTurns);
        }
        let (mut integral, mut area) = (0.0, 0.0);
        for point in state.iter().filter(|point| point.marker == coil_marker) {
            integral += point.vector_potential_wb_per_m * point.integration_weight_m2;
            area += point.integration_weight_m2;
        }
        if area <= 0.0 {
            return Err(AzError::MissingCoilRegion { coil_marker });
        }
        Ok(turns * self.stack_length_m * integral / area)
    }
}

/// Solved field values at one quadrature point.
#[derive(Debug, Clone)]
pub struct AzQuadraturePoint {
    /// Owning element index.
    pub element_index: usize,
    /// Quadrature point index inside the element.
    pub quadrature_index: usize,
    /// Cell marker of the owning element.
    pub marker: i32,
    /// Physical position `[x, y]` in metres.
    pub position_m: [f64; 2],
    /// Integration measure `detJ * weight` in m^2 (per unit depth).
    pub integration_weight_m2: f64,
    /// Vector potential `A_z` in Wb/m.
    pub vector_potential_wb_per_m: f64,
    /// Flux density `[B_x, B_y]` in tesla.
    pub flux_density_t: [f64; 2],
    /// Field strength `[H_x, H_y]` in A/m.
    pub field_strength_a_per_m: [f64; 2],
    /// Secant reluctivity in m/H.
    pub reluctivity_m_per_h: f64,
    /// Differential reluctivity in m/H.
    pub differential_reluctivity_m_per_h: f64,
    /// Magnetic energy density in J/m^3.
    pub energy_density_j_per_m3: f64,
    /// Magnetic coenergy density in J/m^3.
    pub coenergy_density_j_per_m3: f64,
    /// Imposed axial current density in A/m^2.
    pub imposed_current_density_a_per_m2: f64,
    /// Transient eddy current density `-sigma dA_z/dt` in A/m^2.
    pub eddy_current_density_a_per_m2: f64,
}

/// Stack-integrated magnetic energy and coenergy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AzEnergyReport {
    /// Magnetic energy in joules.
    pub energy_j: f64,
    /// Magnetic coenergy in joules.
    pub coenergy_j: f64,
}

/// Solver controls for [`solve_az`].
#[derive(Debug, Clone)]
pub struct AzSolveConfig {
    /// Source-continuation steps for the nonlinear path (ignored when linear).
    pub continuation_steps: usize,
    /// Newton controls for the nonlinear path.
    pub newton: NewtonConfig,
    /// Inner linear solver backend.
    pub backend: AzLinearBackend,
    /// Iterative-solver controls (used by the iterative backends).
    ///
    /// For GMRES strategies, `max_iterations` follows the host
    /// facade's convention of counting restart cycles; the CUDA
    /// backend counts total Arnoldi steps, so the CUDA dispatch
    /// multiplies the budget by `restart` to give both backends
    /// the same Krylov work for one config.
    pub inner_solver: LinearSolveConfig,
    /// Inexact-Newton forcing for the iterative backends.
    ///
    /// When `true`, the nonlinear solve runs trust-region
    /// inexact Newton with relative Eisenstat–Walker-style forcing
    /// (`η = clamp(0.1·sqrt(‖R_k‖/‖R_0‖), floor, 0.1)`): early
    /// Newton steps solve their tangent systems loosely and the
    /// inner tolerance tightens with the nonlinear residual,
    /// never below `inner_solver.tolerance` (which then acts as
    /// the forcing floor, not a per-step demand). A failed inner
    /// solve retries the step at a looser forcing (up to 0.5)
    /// with an escalated Krylov budget rather than failing Newton. This is the practical route
    /// for Krylov backends on motor tangents, whose cold
    /// unsaturated first system resists tight solves while loose
    /// decades are cheap. Ignored by the direct backend.
    pub inexact_newton: bool,
}

/// Inner linear solver of an `A_z` solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AzLinearBackend {
    /// Sparse direct factorization (symbolic analysis reused across the
    /// Newton iterations of one solve). Matches the reference solver's
    /// `spsolve` and is immune to the cancellation noise that sliver
    /// elements inject into iterative-solver validation checks.
    #[default]
    DirectSparse,
    /// Krylov iteration per [`AzSolveConfig::inner_solver`].
    Iterative,
    /// Krylov iteration on the CUDA backend per
    /// [`AzSolveConfig::inner_solver`]. Admitted strategies are the
    /// solvers facade's native CUDA paths (CG/GMRES with
    /// identity/Jacobi/AMG), plus `GmresIlu`, which is routed
    /// through the flexible session API with a resident device
    /// ILU(0) (see `solve_cuda_fgmres_ilu`). The solve fails
    /// rather than fall back to CPU, and result provenance is
    /// checked, so a successful solve with this backend really
    /// ran on the GPU. Requires the workspace `cuda` feature at
    /// runtime.
    IterativeCuda,
}

impl Default for AzSolveConfig {
    fn default() -> Self {
        Self {
            continuation_steps: 4,
            // The reference Newton drives |R|/|f| below 1e-7. A 1e-9
            // default sits under the residual floor that sliver
            // elements impose on motor meshes (~3e-9 on GEO_30MM), so
            // warm-started frame solves stall in the line search a
            // factor of three from the tolerance.
            newton: NewtonConfig {
                max_iterations: 40,
                absolute_tolerance: 1.0e-7,
                relative_tolerance: 1.0e-7,
                line_search: sonium_solvers::LineSearch::backtracking(),
                ..NewtonConfig::default()
            },
            backend: AzLinearBackend::DirectSparse,
            inner_solver: LinearSolveConfig {
                solver: LinearSolverType::GmresIlu,
                max_iterations: 2_000,
                tolerance: 1.0e-9,
                ..LinearSolveConfig::default()
            },
            inexact_newton: false,
        }
    }
}

/// Stateful linear solver for one [`solve_az`] call.
struct LinearSolver {
    backend: AzLinearBackend,
    iterative: LinearSolveConfig,
    /// Eisenstat–Walker forcing tolerance for the next iterative
    /// solve (`None` keeps `iterative.tolerance`).
    tolerance_override: Option<f64>,
    /// Escalated Krylov budget for safeguard retries (`None` keeps
    /// `iterative.max_iterations`).
    budget_override: Option<usize>,
    indefinite: bool,
    symbolic: Option<sonium_solvers::SparseSymbolic>,
    factor: Option<sonium_solvers::SparseFactorization<f64>>,
}

impl LinearSolver {
    fn new(config: &AzSolveConfig, indefinite: bool) -> Self {
        Self {
            backend: config.backend,
            iterative: config.inner_solver.clone(),
            tolerance_override: None,
            budget_override: None,
            indefinite,
            symbolic: None,
            factor: None,
        }
    }

    /// Iterative controls with any forcing-tolerance override applied.
    fn effective_iterative(&self) -> LinearSolveConfig {
        let mut config = self.iterative.clone();
        if let Some(tolerance) = self.tolerance_override {
            config.tolerance = tolerance;
        }
        if let Some(budget) = self.budget_override {
            config.max_iterations = budget;
        }
        config
    }

    fn direct_config(&self) -> sonium_solvers::SparseDirectConfig {
        // Without conductor constraints the A_z tangent is SPD by
        // construction (symmetric reluctivity tangent); Cholesky
        // carries roughly half the fill of LU and ignores the
        // rounding-level asymmetry sliver elements inject. The
        // conductor-bordered system is a symmetric saddle point and
        // takes the indefinite (LDL) path.
        sonium_solvers::SparseDirectConfig {
            matrix_class: if self.indefinite {
                sonium_solvers::MatrixClass::SymmetricIndefinite
            } else {
                sonium_solvers::MatrixClass::SymmetricPositiveDefinite
            },
            ..sonium_solvers::SparseDirectConfig::default()
        }
    }

    /// Factorizes `matrix` and caches the factorization for chord steps.
    fn factorize(&mut self, matrix: &sonium_solvers::CsrMatrix<f64>) -> Result<(), String> {
        let config = self.direct_config();
        if self.symbolic.is_none() {
            self.symbolic = sonium_solvers::analyze_sparse(matrix, &config).ok();
        }
        self.factor = Some(
            sonium_solvers::factorize_sparse(matrix, &config, self.symbolic.as_ref())
                .or_else(|_| sonium_solvers::factorize_sparse(matrix, &config, None))
                .map_err(|error| error.to_string())?,
        );
        Ok(())
    }

    /// Solves with the cached factorization (chord-Newton inner step).
    fn solve_factored(&self, rhs: &[f64]) -> Result<Vec<f64>, String> {
        self.factor
            .as_ref()
            .ok_or_else(|| "no cached factorization".to_string())?
            .solve(rhs)
            .map_err(|error| error.to_string())
    }

    fn solve(
        &mut self,
        matrix: &sonium_solvers::CsrMatrix<f64>,
        rhs: &[f64],
    ) -> Result<Vec<f64>, String> {
        let mut result = self.solve_inner(matrix, rhs);
        // Inexact-Newton safeguard: some tangents have a Krylov
        // convergence wall just above the forcing demand (measured:
        // 1e-3 in 73 iterations but 1e-4 in 19674 on one GEO_30MM
        // tangent; another reached only 0.093 in 30000 iterations),
        // so a failed inner solve retries the step at a looser
        // forcing with an escalated budget instead of failing
        // Newton outright. The line search still judges the actual
        // residual decrease, so a looser step stays valid.
        if result.is_err() {
            for _ in 0..3 {
                let Some(current) = self.tolerance_override else {
                    break;
                };
                let looser = (current * 4.0).min(0.5);
                if looser <= current {
                    break;
                }
                self.tolerance_override = Some(looser);
                let budget = self
                    .budget_override
                    .unwrap_or(self.iterative.max_iterations)
                    .saturating_mul(5);
                self.budget_override = Some(budget);
                result = self.solve_inner(matrix, rhs);
                if result.is_ok() {
                    break;
                }
            }
            self.budget_override = None;
        }
        result
    }

    fn solve_inner(
        &mut self,
        matrix: &sonium_solvers::CsrMatrix<f64>,
        rhs: &[f64],
    ) -> Result<Vec<f64>, String> {
        match self.backend {
            AzLinearBackend::Iterative => {
                let sym = symmetrized(matrix);
                let config = self.effective_iterative();
                solve_real(&sym, rhs, &config)
                    .map(|solution| solution.x)
                    .map_err(|error| error.to_string())
            }
            AzLinearBackend::IterativeCuda => {
                // Without conductor blocks the A_z tangent is SPD by
                // construction; declare it so large-system SPD
                // validation does not fail closed on the sliver
                // rounding the direct backend tolerates.
                let assumption = if self.indefinite {
                    LinearSystemAssumption::General
                } else {
                    LinearSystemAssumption::PositiveDefinite
                };
                // Host GMRES counts max_iterations in restart cycles,
                // the device backend in total Arnoldi steps; scale so
                // both backends budget the same Krylov work (see
                // AzSolveConfig::inner_solver).
                let mut device_config = self.effective_iterative();
                if matches!(
                    device_config.solver,
                    LinearSolverType::Gmres
                        | LinearSolverType::GmresJacobi
                        | LinearSolverType::GmresAmg
                        | LinearSolverType::GmresIlu
                ) {
                    device_config.max_iterations = device_config
                        .max_iterations
                        .saturating_mul(device_config.restart.max(1));
                }
                let sym = symmetrized(matrix);
                // The one-shot facade admits no ILU strategy on CUDA;
                // route GmresIlu through the flexible session API
                // with a resident device ILU(0) instead.
                if device_config.solver == LinearSolverType::GmresIlu {
                    match solve_cuda_fgmres_ilu(&sym, rhs, &device_config) {
                        Ok(solution) => return Ok(solution),
                        Err(CudaIluError::Solve(error)) => return Err(error),
                        Err(CudaIluError::Setup(_)) => {
                            // ILU(0) cannot factor this tangent (pivot
                            // breakdown occurs on some saturated
                            // iterates); fall through to the one-shot
                            // path as plain GMRES for this step.
                            device_config.solver = LinearSolverType::Gmres;
                        }
                    }
                }
                let result = solve_with_backend_and_assumption(
                    &sym,
                    rhs,
                    &device_config,
                    LinearSolveBackend::Cuda,
                    LinearSolveFallback::Reject,
                    assumption,
                )
                .map_err(|error| error.to_string())?;
                if result.backend.actual != LinearSolveBackend::Cuda {
                    return Err(format!(
                        "CUDA backend requested but solve executed on {:?}",
                        result.backend.actual
                    ));
                }
                Ok(result.x)
            }
            AzLinearBackend::DirectSparse => {
                self.factorize(matrix)?;
                self.solve_factored(rhs)
            }
        }
    }
}

/// Result of one [`solve_az`] call.
#[derive(Debug, Clone)]
pub struct AzSolution {
    /// Nodal/edge `A_z` values in Wb/m, Dirichlet DOFs exactly zero.
    pub vector_potential_wb_per_m: Vec<f64>,
    /// Quadrature-point field state.
    pub quadrature_state: Vec<AzQuadraturePoint>,
    /// Newton iterations per continuation step (one `0` when linear).
    pub continuation_iterations: Vec<usize>,
    /// Final free-DOF residual norm (linear path: post-solve residual).
    pub final_residual: f64,
    /// Solved per-block `E0` in V/m (empty without active blocks).
    pub conductor_e0_v_per_m: Vec<f64>,
    /// Stack energy and coenergy.
    pub energy: AzEnergyReport,
}

/// Solves an `A_z` problem with homogeneous Dirichlet DOFs (`A_z = 0`).
///
/// Affine problems take one direct linear solve of the tangent system at
/// `A_z = 0` (exact, and the fast path transient loops rely on).
/// Nonlinear iron takes source continuation with inexact Newton, warm
/// started from `initial` when supplied.
///
/// # Errors
///
/// Returns [`AzError`] on validation failure, an out-of-range Dirichlet
/// DOF, or a failed inner/Newton solve.
pub fn solve_az(
    problem: &AzProblem,
    dirichlet_dofs: &[usize],
    config: &AzSolveConfig,
    initial: Option<&[f64]>,
    transient: Option<&TransientStep>,
) -> Result<AzSolution, AzError> {
    problem.validate()?;
    let num_dofs = problem.num_dofs();
    let constrained: HashSet<usize> = dirichlet_dofs.iter().copied().collect();
    if constrained.iter().any(|&dof| dof >= num_dofs) {
        return Err(AzError::InvalidDirichletDof);
    }
    let mut potential = match initial {
        Some(values) => {
            if values.len() != num_dofs {
                return Err(AzError::DimensionMismatch {
                    expected: num_dofs,
                    actual: values.len(),
                });
            }
            values.to_vec()
        }
        None => transient.map_or_else(
            || vec![0.0; num_dofs],
            |step| step.previous_potential.clone(),
        ),
    };
    for &dof in &constrained {
        potential[dof] = 0.0;
    }
    // Extended unknown vector: [A_z DOFs; conductor E0 fields]. The
    // public initial guess covers the FEM part; E0 starts from the
    // step's stored value (previous frame) or zero.
    let blocks_active = problem.blocks_active(transient);
    if blocks_active {
        let mut extended = vec![0.0; problem.num_unknowns(transient)];
        extended[..num_dofs].copy_from_slice(&potential);
        if let Some(step) = transient {
            for (offset, value) in step.conductor_e0_v_per_m.iter().enumerate() {
                if num_dofs + offset < extended.len() {
                    extended[num_dofs + offset] = *value;
                }
            }
        }
        potential = extended;
    }
    let num_unknowns = potential.len();
    let solver = std::cell::RefCell::new(LinearSolver::new(config, blocks_active));
    let (continuation_iterations, final_residual) = if problem.is_linear() {
        let zero = vec![0.0; num_unknowns];
        let (residual_at_zero, mut tangent) = problem.assemble(&zero, 1.0, transient)?;
        apply_constraints(&mut tangent, &constrained);
        let mut rhs: Vec<f64> = residual_at_zero.iter().map(|value| -value).collect();
        // The affine residual is K*A - f + K*A_prev_eddy terms already
        // contained in residual_at_zero, so K*A = -residual(0) is exact.
        // A warm start only matters to iterative inner solvers; solve_real
        // starts from zero, so no correction is applied here.
        for &dof in &constrained {
            rhs[dof] = 0.0;
        }
        potential = solver
            .borrow_mut()
            .solve(&tangent.to_csr(), &rhs)
            .map_err(AzError::LinearSolve)?;
        for &dof in &constrained {
            potential[dof] = 0.0;
        }
        let (residual, _) = problem.assemble(&potential, 1.0, transient)?;
        (vec![0], free_residual_norm(&residual, &constrained))
    } else {
        if config.continuation_steps == 0 {
            return Err(AzError::InvalidContinuationSteps);
        }
        let mut iterations = Vec::with_capacity(config.continuation_steps);
        let mut residual_norm = f64::INFINITY;
        if config.backend == AzLinearBackend::DirectSparse {
            // Chord Newton: one Cholesky factorization serves many
            // residual evaluations; refactorize only on stall. Direct
            // factorisation dominates cost on motor-sized systems.
            for step in 1..=config.continuation_steps {
                let load_factor = step as f64 / config.continuation_steps as f64;
                let (count, norm) = chord_newton_step(
                    problem,
                    &mut potential,
                    load_factor,
                    transient,
                    &constrained,
                    config,
                    &solver,
                )?;
                iterations.push(count);
                residual_norm = norm;
            }
            (iterations, residual_norm)
        } else {
            for step in 1..=config.continuation_steps {
                let load_factor = step as f64 / config.continuation_steps as f64;
                let residual_fn = |trial: &[f64]| {
                    let (mut residual, _) = problem
                        .assemble(trial, load_factor, transient)
                        .expect("validated A_z assembly");
                    for &dof in &constrained {
                        residual[dof] = 0.0;
                    }
                    residual
                };
                let solve_step = |trial: &[f64], rhs: &[f64]| -> Result<Vec<f64>, String> {
                    let (_, mut tangent) = problem
                        .assemble(trial, load_factor, transient)
                        .map_err(|error| error.to_string())?;
                    apply_constraints(&mut tangent, &constrained);
                    let mut constrained_rhs = rhs.to_vec();
                    for &dof in &constrained {
                        constrained_rhs[dof] = 0.0;
                    }
                    solver
                        .borrow_mut()
                        .solve(&tangent.to_csr(), &constrained_rhs)
                };
                let result = if config.inexact_newton {
                    // Relative Eisenstat-Walker-style forcing, computed
                    // here from the Newton RHS norms: eta =
                    // clamp(0.1 * sqrt(||R_k|| / ||R_0||), floor, 0.1).
                    // (The facade's built-in forcing keys off the
                    // absolute residual norm, which for this problem's
                    // scaling keeps eta at its cap until the final
                    // steps; Newton then crawls and can wander into
                    // tangents no Krylov method will solve.) Steps are
                    // additionally trust-region clipped: loose inexact
                    // steps far from the root otherwise overshoot into
                    // saturation states whose tangents defeat the
                    // Krylov stack entirely (measured repeatedly).
                    let first_rhs_norm = std::cell::Cell::new(0.0f64);
                    let floor = config.inner_solver.tolerance;
                    let inexact = InexactNewtonConfig {
                        newton: config.newton.clone(),
                        forcing: ForcingTerm::None,
                        trust_region: Some(TrustRegionConfig::default()),
                        ..InexactNewtonConfig::default()
                    };
                    newton_raphson_inexact(
                        &potential,
                        residual_fn,
                        |trial, rhs, _eta| {
                            let norm = rhs.iter().map(|value| value * value).sum::<f64>().sqrt();
                            if first_rhs_norm.get() == 0.0 {
                                first_rhs_norm.set(norm.max(1.0e-300));
                            }
                            let relative = norm / first_rhs_norm.get();
                            let eta = (0.1 * relative.sqrt()).clamp(floor, 0.1);
                            solver.borrow_mut().tolerance_override = Some(eta);
                            solve_step(trial, rhs)
                        },
                        &inexact,
                    )
                } else {
                    sonium_solvers::newton_raphson(
                        &potential,
                        residual_fn,
                        solve_step,
                        &config.newton,
                    )
                }?;
                potential = result.x;
                for &dof in &constrained {
                    potential[dof] = 0.0;
                }
                iterations.push(result.iterations);
                residual_norm = result.residual;
            }
            (iterations, residual_norm)
        }
    };
    let conductor_e0: Vec<f64> = if blocks_active {
        potential[num_dofs..].to_vec()
    } else {
        Vec::new()
    };
    let fem_potential: Vec<f64> = potential[..num_dofs].to_vec();
    let state_step;
    let state_transient = match transient {
        Some(step) if blocks_active => {
            state_step = TransientStep {
                previous_potential: step.previous_potential.clone(),
                dt_s: step.dt_s,
                conductor_e0_v_per_m: conductor_e0.clone(),
            };
            Some(&state_step)
        }
        other => other,
    };
    let quadrature_state = problem.quadrature_state(&fem_potential, state_transient)?;
    let energy = problem.energy_report(&quadrature_state);
    Ok(AzSolution {
        vector_potential_wb_per_m: fem_potential,
        quadrature_state,
        continuation_iterations,
        final_residual,
        conductor_e0_v_per_m: conductor_e0,
        energy,
    })
}

/// One continuation step of chord Newton with a reused factorization.
///
/// Iterates `x += lambda * dx`, `J_frozen dx = -R(x)`, with backtracking
/// on the free residual norm. The tangent is refactorized when an
/// accepted step reduces the residual by less than half, or when
/// backtracking fails entirely. Convergence uses the
/// Newton config's absolute tolerance, or its relative tolerance
/// against the step's initial residual.
fn chord_newton_step(
    problem: &AzProblem,
    potential: &mut [f64],
    load_factor: f64,
    transient: Option<&TransientStep>,
    constrained: &HashSet<usize>,
    config: &AzSolveConfig,
    solver: &std::cell::RefCell<LinearSolver>,
) -> Result<(usize, f64), AzError> {
    let constrained_residual = |trial: &[f64]| -> Result<Vec<f64>, AzError> {
        let mut residual = problem.assemble_residual(trial, load_factor, transient)?;
        for &dof in constrained {
            residual[dof] = 0.0;
        }
        Ok(residual)
    };
    let mut residual = constrained_residual(potential)?;
    let mut norm = free_residual_norm(&residual, constrained);
    let initial_norm = norm.max(1.0e-300);
    let tolerance = config
        .newton
        .absolute_tolerance
        .max(config.newton.relative_tolerance * initial_norm);
    let mut factorized = false;
    let mut stall = 0usize;
    let mut fresh_factor = false;
    for iteration in 0..config.newton.max_iterations {
        if norm <= tolerance {
            return Ok((iteration, norm));
        }
        if !factorized {
            let (_, mut tangent) = problem.assemble(potential, load_factor, transient)?;
            apply_constraints(&mut tangent, constrained);
            solver
                .borrow_mut()
                .factorize(&tangent.to_csr())
                .map_err(AzError::LinearSolve)?;
            factorized = true;
            fresh_factor = true;
        }
        let rhs: Vec<f64> = residual.iter().map(|value| -value).collect();
        let direction = solver
            .borrow()
            .solve_factored(&rhs)
            .map_err(AzError::LinearSolve)?;
        // Backtracking line search on the residual norm.
        let mut lambda = 1.0;
        let mut accepted: Option<(Vec<f64>, Vec<f64>, f64)> = None;
        for _ in 0..12 {
            let mut trial: Vec<f64> = potential
                .iter()
                .zip(&direction)
                .map(|(x, dx)| x + lambda * dx)
                .collect();
            for &dof in constrained {
                trial[dof] = 0.0;
            }
            let trial_residual = constrained_residual(&trial)?;
            let trial_norm = free_residual_norm(&trial_residual, constrained);
            if trial_norm < norm || trial_norm <= tolerance {
                accepted = Some((trial, trial_residual, trial_norm));
                break;
            }
            lambda *= 0.5;
        }
        let Some((trial, trial_residual, trial_norm)) = accepted else {
            if fresh_factor {
                return Err(AzError::Newton(NewtonError::LineSearchFailed {
                    iteration,
                    halvings: 12,
                    residual: norm,
                }));
            }
            factorized = false;
            continue;
        };
        if trial_norm > 0.5 * norm {
            stall += 1;
            if stall >= 1 {
                factorized = false;
                stall = 0;
            }
        } else {
            stall = 0;
        }
        fresh_factor = false;
        potential.copy_from_slice(&trial);
        residual = trial_residual;
        norm = trial_norm;
    }
    if norm <= tolerance {
        return Ok((config.newton.max_iterations, norm));
    }
    Err(AzError::Newton(NewtonError::NotConverged(
        config.newton.max_iterations,
        norm,
    )))
}

fn free_residual_norm(residual: &[f64], constrained: &HashSet<usize>) -> f64 {
    residual
        .iter()
        .enumerate()
        .filter(|(dof, _)| !constrained.contains(dof))
        .map(|(_, value)| value * value)
        .sum::<f64>()
        .sqrt()
}

/// Prepares the device ILU(0) preconditioner for a flexible session.
///
/// Uploads the *compatibility* host factors: the same
/// `IluPreconditioner::from_csr` policy the CPU `GmresIlu` route
/// trusts, including its silent pivot regularization. The checked
/// device setup rejects saturated tangents whose elimination
/// regularizes a handful of pivots, and the historical workaround
/// (refactorizing a diagonally shifted copy) produced visibly
/// weaker factors — measured on frozen fine-mesh tangents, the
/// host factors converge in ~10 iterations where shifted checked
/// factors stall above 0.6. Threshold ILU variants (ILUT/ILUTP,
/// HILU, COLAMD reorderings, equilibrated factors) measured no
/// better on these tangents, so ILU(0) with host-identical policy
/// is the single device route.
#[cfg(feature = "cuda")]
fn prepare_cuda_ilu(
    backend: &sonium_compute::CudaBackend,
    canonical: &sonium_solvers::CsrMatrix<f64>,
) -> sonium_compute::BackendResult<sonium_solvers::preconditioners::CudaIluPreconditioner<f64>> {
    use sonium_solvers::preconditioners::{CudaIluPreconditioner, IluPreconditioner};

    let host = IluPreconditioner::from_csr(canonical);
    CudaIluPreconditioner::try_from_host(backend, &host)
}

/// Failure stage of [`solve_cuda_fgmres_ilu`].
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
enum CudaIluError {
    /// Backend/session/ILU-factorization setup failed; the caller
    /// may retry with another strategy for this step.
    Setup(String),
    /// The device solve itself failed (including non-convergence).
    Solve(String),
}

/// One-shot CUDA FGMRES with a resident ILU(0) preconditioner.
///
/// Mirrors the facade's one-shot CUDA solve (fresh backend and pool
/// per call) but prepares a host ILU(0) factorization as a native
/// device preconditioner through the flexible session API, which
/// the config-only one-shot entry point does not admit (see
/// [`prepare_cuda_ilu`] for the factor policy). Solve-stage
/// failures (including non-convergence) are not retried here — the
/// inexact-Newton safeguard in the caller owns tolerance/budget
/// retries. Provenance is checked exactly like the one-shot path.
#[cfg(feature = "cuda")]
fn solve_cuda_fgmres_ilu(
    matrix: &sonium_solvers::CsrMatrix<f64>,
    rhs: &[f64],
    config: &LinearSolveConfig,
) -> Result<Vec<f64>, CudaIluError> {
    use sonium_compute::{CudaBackend, MemoryPool, MemoryPoolConfig};
    use sonium_solvers::iterative::DeviceInitialGuess;
    use sonium_solvers::{BackendKrylovConfig, CudaLinearSolveSession};

    let setup = |error: &dyn std::fmt::Display| CudaIluError::Setup(error.to_string());
    let backend = CudaBackend::new(0).map_err(|error| setup(&error))?;
    let pool =
        MemoryPool::new(backend, MemoryPoolConfig::default()).map_err(|error| setup(&error))?;
    let controls = BackendKrylovConfig {
        max_iterations: config.max_iterations,
        restart: config.restart,
        tolerance: config.tolerance,
        require_native: true,
        exact_residual_check_interval: config.exact_residual_check_interval,
    };
    let mut session =
        CudaLinearSolveSession::try_new_flexible(&pool, matrix, &controls, |backend, canonical| {
            prepare_cuda_ilu(backend, canonical)
        })
        .map_err(|error| setup(&error))?;
    let result = session
        .solve_host(rhs, DeviceInitialGuess::Zero)
        .map_err(|error| CudaIluError::Solve(error.to_string()))?;
    if result.backend.actual != LinearSolveBackend::Cuda {
        return Err(CudaIluError::Solve(format!(
            "CUDA backend requested but solve executed on {:?}",
            result.backend.actual
        )));
    }
    Ok(result.x)
}

/// Non-CUDA builds have no device ILU path; fail with the reason
/// instead of silently changing strategy.
#[cfg(not(feature = "cuda"))]
fn solve_cuda_fgmres_ilu(
    _matrix: &sonium_solvers::CsrMatrix<f64>,
    _rhs: &[f64],
    _config: &LinearSolveConfig,
) -> Result<Vec<f64>, CudaIluError> {
    Err(CudaIluError::Solve(
        "GmresIlu on the CUDA backend requires the cuda feature".to_string(),
    ))
}

/// `(K + K^T) / 2` of an assembled tangent.
///
/// The A_z tangent is symmetric by construction (symmetric
/// constitutive tangent, symmetric saddle border), but parallel
/// triplet accumulation sums the (i, j) and (j, i) duplicates in
/// different orders; on sliver-bearing meshes with entries near
/// 1e9 cancelling, the residue asymmetry exceeds the exact
/// symmetry validation that iterative solvers apply. Averaging
/// removes that accumulation noise and nothing else. The direct
/// backend does not need it (it reads one triangle).
fn symmetrized(matrix: &sonium_solvers::CsrMatrix<f64>) -> sonium_solvers::CsrMatrix<f64> {
    let mut triplets = TripletMatrix::new(matrix.num_rows, matrix.num_cols);
    for row in 0..matrix.num_rows {
        for index in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            let col = matrix.col_indices[index];
            let half = 0.5 * matrix.values[index];
            triplets.push(row, col, half);
            triplets.push(col, row, half);
        }
    }
    triplets.to_csr()
}

fn apply_constraints(matrix: &mut TripletMatrix<f64>, constrained: &HashSet<usize>) {
    for index in 0..matrix.nnz() {
        if constrained.contains(&matrix.rows[index]) || constrained.contains(&matrix.cols[index]) {
            matrix.values[index] = 0.0;
        }
    }
    for &dof in constrained {
        matrix.push(dof, dof, 1.0);
    }
}

/// Failure of an `A_z` problem assembly or solve.
#[derive(Debug, thiserror::Error)]
pub enum AzError {
    /// Quadrature order was zero.
    #[error("quadrature order must be positive")]
    InvalidQuadratureOrder,
    /// Stack length was non-positive or non-finite.
    #[error("stack length must be finite and positive")]
    InvalidStackLength,
    /// Mesh was not two-dimensional.
    #[error("A_z problems need a 2D mesh, got dimension {actual}")]
    InvalidMeshDimension {
        /// Actual mesh dimension.
        actual: usize,
    },
    /// A cell marker had no material.
    #[error("cell {element_index} has no A_z material for marker {marker}")]
    MissingMaterial {
        /// Element index of the unmapped cell.
        element_index: usize,
        /// Unmapped cell marker.
        marker: i32,
    },
    /// A current density was non-finite.
    #[error("current density for marker {marker} must be finite")]
    InvalidCurrentDensity {
        /// Marker carrying the invalid density.
        marker: i32,
    },
    /// A potential vector had the wrong length.
    #[error("potential has {actual} DOFs; expected {expected}")]
    DimensionMismatch {
        /// Expected DOF count.
        expected: usize,
        /// Actual DOF count.
        actual: usize,
    },
    /// Continuation load factor lay outside `[0, 1]`.
    #[error("continuation load factor must lie in [0, 1]")]
    InvalidLoadFactor,
    /// Continuation had no steps.
    #[error("continuation requires at least one load step")]
    InvalidContinuationSteps,
    /// Time step was non-positive or non-finite.
    #[error("time step must be finite and positive")]
    InvalidTimeStep,
    /// A Dirichlet DOF lay outside the problem.
    #[error("Dirichlet DOF is outside the problem")]
    InvalidDirichletDof,
    /// Turns were non-finite.
    #[error("turns must be finite")]
    InvalidTurns,
    /// No quadrature point carried the requested coil marker.
    #[error("mesh has no quadrature points in coil marker {coil_marker}")]
    MissingCoilRegion {
        /// Requested coil marker.
        coil_marker: i32,
    },
    /// A conductor block was empty, duplicated a marker, or non-finite.
    #[error("conductor blocks need unique non-empty markers and finite target currents")]
    InvalidConductorBlock,
    /// A per-element magnet demagnetisation factor lay outside `[0, 1]`.
    #[error("element magnet scale must lie in [0, 1]")]
    InvalidMagnetScale,
    /// A material was invalid.
    #[error(transparent)]
    Material(#[from] super::materials::AzMaterialError),
    /// The Newton solve failed.
    #[error(transparent)]
    Newton(#[from] NewtonError),
    /// The inner linear solve failed.
    #[error("linear solve failed: {0}")]
    LinearSolve(String),
}

/// Reluctivity-weighted Laplacian for `A_z`: `integral nu grad A . grad v`.
///
/// This is the form-level entry point for affine problems assembled
/// through the generic region assemblers; [`AzProblem`] is the nonlinear
/// entry point.
#[derive(Debug, Clone, Copy)]
pub struct ReluctivityForm {
    /// Reluctivity `nu` in m/H.
    pub reluctivity_m_per_h: f64,
}

impl BilinearForm<f64> for ReluctivityForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        WeightedLaplacianForm {
            coefficient: self.reluctivity_m_per_h,
        }
        .integrate_point(test, trial, weight, local_matrix);
    }
}

/// Imposed-current load form: `l(v) = integral J_z v`.
#[derive(Debug, Clone, Copy)]
pub struct AxialCurrentForm {
    /// Current density `J_z` in A/m^2.
    pub current_density_a_per_m2: f64,
}

impl LinearForm<f64> for AxialCurrentForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let w = weight * test.det_j * self.current_density_a_per_m2;
        for i in 0..test.n_dofs {
            local_vector.add(i, test.value(i) * w);
        }
    }
}

/// Magnet load form: `l(v) = integral H_c . curl(v z)`.
#[derive(Debug, Clone, Copy)]
pub struct MagnetCoercivityForm {
    /// Coercivity `[H_cx, H_cy]` in A/m; see [`super::materials::MagnetLaw`].
    pub coercivity_a_per_m: [f64; 2],
}

impl LinearForm<f64> for MagnetCoercivityForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let w = weight * test.det_j;
        for i in 0..test.n_dofs {
            // curl(v z) basis vector is [dN/dy, -dN/dx].
            let integrand = self.coercivity_a_per_m[0] * test.grad(i, 1)
                - self.coercivity_a_per_m[1] * test.grad(i, 0);
            local_vector.add(i, integrand * w);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::rectangular_mesh_triangles;
    use crate::physics::electromagnetics::magnetostatics::BhCurve;
    use crate::physics::electromagnetics::rotating_machine::materials::MagnetLaw;
    use std::f64::consts::PI;

    fn marked(mesh: &mut Mesh, marker: i32) {
        for element in &mut mesh.elements {
            element.marker = marker;
        }
    }

    fn boundary_dofs(mesh: &Mesh, extent: [f64; 4]) -> Vec<usize> {
        let [x0, x1, y0, y1] = extent;
        mesh.nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| {
                (node.x - x0).abs() < 1.0e-12
                    || (node.x - x1).abs() < 1.0e-12
                    || (node.y - y0).abs() < 1.0e-12
                    || (node.y - y1).abs() < 1.0e-12
            })
            .map(|(index, _)| index)
            .collect()
    }

    fn steel_curve() -> BhCurve {
        BhCurve::new(vec![
            (0.0, 0.0),
            (0.5, 100.0),
            (1.0, 300.0),
            (1.4, 1_000.0),
            (1.7, 8_000.0),
            (2.0, 80_000.0),
        ])
        .unwrap()
    }

    #[test]
    fn tangent_matches_directional_finite_difference_with_magnet_and_eddy() {
        let mut mesh = rectangular_mesh_triangles(0.0, 0.03, 0.0, 0.02, 3, 2);
        for element in &mut mesh.elements {
            let centre_x = element
                .vertices()
                .iter()
                .map(|node| mesh.nodes[*node].x)
                .sum::<f64>()
                / 3.0;
            element.marker = if centre_x < 0.015 { 1 } else { 2 };
        }
        let problem = AzProblem {
            mesh,
            materials: BTreeMap::from([
                (1, AzMaterial::nonlinear(steel_curve())),
                (
                    2,
                    AzMaterial::magnet(MagnetLaw::new(1.05, [0.2, 1.1]).unwrap())
                        .unwrap()
                        .with_conductivity(6.0e5)
                        .unwrap(),
                ),
            ]),
            current_density_a_per_m2: BTreeMap::from([(1, 1.0e6)]),
            current_density_fn: None,
            degree: PolynomialDegree::P1,
            quadrature_order: 2,
            stack_length_m: 0.05,
            conductor_blocks: Vec::new(),
            element_magnet_scale: Vec::new(),
        };
        let num_dofs = problem.num_dofs();
        let potential: Vec<f64> = (0..num_dofs).map(|i| 1.0e-4 * (i as f64 + 1.0)).collect();
        let direction: Vec<f64> = (0..num_dofs).map(|i| 0.3 - 0.02 * i as f64).collect();
        let transient = TransientStep {
            previous_potential: potential.iter().map(|value| value * 0.5).collect(),
            dt_s: 1.0e-4,
            conductor_e0_v_per_m: Vec::new(),
        };
        let (residual, tangent) = problem.assemble(&potential, 1.0, Some(&transient)).unwrap();
        let epsilon = 1.0e-7;
        let perturbed: Vec<f64> = potential
            .iter()
            .zip(&direction)
            .map(|(a, d)| a + epsilon * d)
            .collect();
        let (next, _) = problem.assemble(&perturbed, 1.0, Some(&transient)).unwrap();
        let mut tangent_direction = vec![0.0; num_dofs];
        for index in 0..tangent.nnz() {
            tangent_direction[tangent.rows[index]] +=
                tangent.values[index] * direction[tangent.cols[index]];
        }
        let error = next
            .iter()
            .zip(&residual)
            .zip(&tangent_direction)
            .map(|((n, b), exact)| ((n - b) / epsilon - exact).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale = tangent_direction.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            error / scale < 1.0e-5,
            "relative tangent error {}",
            error / scale
        );
    }

    /// Manufactured solution `A = sin(pi x) sin(pi y)` for `-nu lap A = J`
    /// on the unit square; returns the nodal L2 error against `sin sin`.
    fn mms_error(n: usize, degree: PolynomialDegree) -> f64 {
        let nu = 2.0;
        let mut mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, n, n);
        if degree == PolynomialDegree::P2 {
            mesh = mesh.to_p2();
        }
        marked(&mut mesh, 1);
        let dirichlet = boundary_dofs(&mesh, [0.0, 1.0, 0.0, 1.0]);
        let problem = AzProblem {
            mesh,
            materials: BTreeMap::from([(1, AzMaterial::linear_permeability(1.0 / nu).unwrap())]),
            current_density_a_per_m2: BTreeMap::new(),
            current_density_fn: Some(Arc::new(move |x: f64, y: f64| {
                nu * 2.0 * PI * PI * (PI * x).sin() * (PI * y).sin()
            })),
            degree,
            quadrature_order: 4,
            stack_length_m: 1.0,
            conductor_blocks: Vec::new(),
            element_magnet_scale: Vec::new(),
        };
        assert_eq!(problem.num_dofs(), problem.mesh.num_nodes());
        let solution = solve_az(&problem, &dirichlet, &AzSolveConfig::default(), None, None)
            .expect("MMS solve");
        let mut squared = 0.0;
        for (node, value) in problem
            .mesh
            .nodes
            .iter()
            .zip(&solution.vector_potential_wb_per_m)
        {
            let exact = (PI * node.x).sin() * (PI * node.y).sin();
            squared += (value - exact).powi(2);
        }
        (squared / problem.mesh.num_nodes() as f64).sqrt()
    }

    #[test]
    fn mms_convergence_rates_p1_and_p2() {
        let p1_coarse = mms_error(8, PolynomialDegree::P1);
        let p1_fine = mms_error(16, PolynomialDegree::P1);
        let p1_rate = (p1_coarse / p1_fine).log2();
        assert!(
            p1_rate > 1.8,
            "P1 rate {p1_rate} (errors {p1_coarse}, {p1_fine})"
        );
        let p2_coarse = mms_error(8, PolynomialDegree::P2);
        let p2_fine = mms_error(16, PolynomialDegree::P2);
        let p2_rate = (p2_coarse / p2_fine).log2();
        assert!(
            p2_rate > 2.6,
            "P2 rate {p2_rate} (errors {p2_coarse}, {p2_fine})"
        );
        assert!(p2_fine < p1_fine / 10.0, "P2 must be far more accurate");
    }

    #[test]
    fn conductor_block_enforces_net_current_and_decays() {
        // Left half: solid conductor block. Right half: air with an
        // imposed current in step 1 only. The block must carry exactly
        // its target net current every step; after the source switches
        // off, its eddy loss must decay monotonically.
        let mut mesh = rectangular_mesh_triangles(0.0, 0.02, 0.0, 0.01, 16, 8);
        for element in &mut mesh.elements {
            let centre_x = element
                .vertices()
                .iter()
                .map(|node| mesh.nodes[*node].x)
                .sum::<f64>()
                / 3.0;
            element.marker = if centre_x < 0.01 { 1 } else { 2 };
        }
        let dirichlet = boundary_dofs(&mesh, [0.0, 0.02, 0.0, 0.01]);
        let num_dofs = mesh.num_nodes();
        let make_problem = |mesh: Mesh, source_j: f64| AzProblem {
            mesh,
            materials: BTreeMap::from([
                (1, AzMaterial::air().with_conductivity(5.8e7).unwrap()),
                (2, AzMaterial::air()),
            ]),
            current_density_a_per_m2: BTreeMap::from([(2, source_j)]),
            current_density_fn: None,
            degree: PolynomialDegree::P1,
            quadrature_order: 2,
            stack_length_m: 1.0,
            conductor_blocks: vec![ConductorBlock {
                markers: vec![1],
                target_current_a: 0.0,
            }],
            element_magnet_scale: Vec::new(),
        };
        let block_current = |state: &[AzQuadraturePoint]| -> f64 {
            state
                .iter()
                .filter(|point| point.marker == 1)
                .map(|point| point.eddy_current_density_a_per_m2 * point.integration_weight_m2)
                .sum()
        };
        let block_loss = |state: &[AzQuadraturePoint]| -> f64 {
            state
                .iter()
                .filter(|point| point.marker == 1)
                .map(|point| {
                    point.eddy_current_density_a_per_m2.powi(2) / 5.8e7
                        * point.integration_weight_m2
                })
                .sum()
        };
        let dt = 2.0e-6;
        let mut previous = vec![0.0; num_dofs];
        let mut losses: Vec<f64> = Vec::new();
        for step_index in 0..6 {
            let problem = make_problem(mesh.clone(), if step_index == 0 { 2.0e7 } else { 0.0 });
            let step = TransientStep {
                previous_potential: previous.clone(),
                dt_s: dt,
                conductor_e0_v_per_m: Vec::new(),
            };
            let solution = solve_az(
                &problem,
                &dirichlet,
                &AzSolveConfig::default(),
                Some(&previous),
                Some(&step),
            )
            .expect("eddy step");
            let net = block_current(&solution.quadrature_state);
            assert!(
                net.abs() < 1.0e-6,
                "step {step_index}: net block current {net:e} A"
            );
            assert_eq!(solution.conductor_e0_v_per_m.len(), 1);
            losses.push(block_loss(&solution.quadrature_state));
            previous = solution.vector_potential_wb_per_m;
        }
        assert!(losses[0] > 0.0, "induced loss in driven step");
        for pair in losses.windows(2).skip(1) {
            assert!(
                pair[1] < pair[0],
                "loss must decay after source off: {losses:?}"
            );
        }
    }

    #[test]
    fn slab_current_recovers_parabolic_potential() {
        // J uniform, A = 0 at x = 0 and x = 1, natural at y bounds:
        // A(x) = J x (1 - x) / (2 nu), centre value J / (8 nu).
        let (jz, nu) = (3.0e6, 1.0 / (4.0 * PI * 1.0e-7 * 1000.0));
        let mut mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.25, 20, 2);
        marked(&mut mesh, 1);
        let dirichlet: Vec<usize> = mesh
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.x.abs() < 1.0e-12 || (node.x - 1.0).abs() < 1.0e-12)
            .map(|(index, _)| index)
            .collect();
        let problem = AzProblem {
            mesh,
            materials: BTreeMap::from([(1, AzMaterial::linear_mu_r(1000.0).unwrap())]),
            current_density_a_per_m2: BTreeMap::from([(1, jz)]),
            current_density_fn: None,
            degree: PolynomialDegree::P1,
            quadrature_order: 2,
            stack_length_m: 1.0,
            conductor_blocks: Vec::new(),
            element_magnet_scale: Vec::new(),
        };
        let solution = solve_az(&problem, &dirichlet, &AzSolveConfig::default(), None, None)
            .expect("slab solve");
        let exact_centre = jz / (8.0 * nu);
        let centre = problem
            .mesh
            .nodes
            .iter()
            .zip(&solution.vector_potential_wb_per_m)
            .filter(|(node, _)| (node.x - 0.5).abs() < 1.0e-12)
            .map(|(_, value)| *value)
            .fold(0.0_f64, f64::max);
        assert!(
            (centre - exact_centre).abs() / exact_centre < 0.01,
            "centre {centre} vs exact {exact_centre}"
        );
    }
}
