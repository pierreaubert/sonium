//! Explicit P1/P2 discontinuous-Galerkin compressible Navier--Stokes solver.

use std::collections::HashMap;

use crate::basis::PolynomialDegree;
use crate::mesh::{ElementType, Mesh};
use crate::parallelism::ParallelPolicy;
use crate::quadrature::QuadratureRule;
use crate::spaces::dg::{DGSpace, InteriorFacet, build_interior_facets, element_diameter};
use crate::spaces::{BasisTabulation, FunctionSpace};
use rayon::prelude::*;

use super::{ConservativeState, IdealGas, ViscosityModel, hllc_flux, rusanov_flux};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompressibleBoundaryCondition {
    SlipWall,
    NoSlipAdiabatic {
        wall_velocity: [f64; 3],
    },
    NoSlipIsothermal {
        wall_velocity: [f64; 3],
        temperature: f64,
    },
    FarField(ConservativeState),
    SupersonicState(ConservativeState),
    SubsonicTotalInlet {
        total_pressure: f64,
        total_temperature: f64,
        direction: [f64; 3],
    },
    StaticPressureOutlet {
        pressure: f64,
    },
}

#[derive(Debug, Clone)]
pub struct CompressibleDgConfig {
    pub gas: IdealGas,
    pub viscosity: ViscosityModel,
    pub prandtl: f64,
    pub cfl: f64,
    pub sipg_penalty: f64,
    pub density_floor: f64,
    pub pressure_floor: f64,
    pub troubled_cell_relaxation: f64,
    pub maximum_step_retries: usize,
    pub boundaries: HashMap<i32, CompressibleBoundaryCondition>,
    /// Persson--Peraire modal smoothness threshold on `log10` of the
    /// highest-degree modal energy fraction of the density. When finite,
    /// it replaces the jump/deviation sensor as the primary troubled-cell
    /// criterion (a strong cell-average jump above 0.5 remains as a safety
    /// net). `f64::INFINITY` (default) disables it.
    pub persson_threshold: f64,
    /// Troubled-cell sensor threshold: an element is limited only when the
    /// relative jump of density or pressure between its cell average and a
    /// facet neighbour's cell average exceeds this value. Smooth elements
    /// keep their full polynomial slopes. The default 0.03 was calibrated
    /// on the Mach 1 cylinder (best stagnation-pressure accuracy of the
    /// swept values 0.03/0.1/0.2/0.5, with no positivity retries); set
    /// `f64::INFINITY` to disable the sensor and limit every element
    /// (legacy behaviour).
    pub troubled_cell_threshold: f64,
    /// Element polynomial degree of the DG space. `P1` (default) preserves
    /// the legacy solver exactly; `P2` uses quadratic Lagrange nodes
    /// (vertices plus edge midpoints) with a quadrature-computed mass
    /// inverse. `P3` is rejected: its face/bubble node layout is not wired
    /// into this solver's state layout.
    pub degree: PolynomialDegree,
    /// Numerical flux for the inviscid interface term.
    pub flux: CompressibleFluxKind,
}

/// Interface flux selection for the compressible DG operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompressibleFluxKind {
    /// Rusanov (local Lax--Friedrichs): one scalar dissipation for all
    /// waves. Legacy default.
    #[default]
    Rusanov,
    /// HLLC: restores the contact wave; falls back to Rusanov whenever the
    /// star-region state would be nonphysical.
    Hllc,
}

impl Default for CompressibleDgConfig {
    fn default() -> Self {
        Self {
            gas: IdealGas::AIR,
            viscosity: ViscosityModel::AIR,
            prandtl: 0.72,
            cfl: 0.25,
            sipg_penalty: 4.0,
            density_floor: 1.0e-12,
            pressure_floor: 1.0e-12,
            troubled_cell_relaxation: 0.05,
            maximum_step_retries: 8,
            boundaries: HashMap::new(),
            troubled_cell_threshold: 0.03,
            persson_threshold: f64::INFINITY,
            degree: PolynomialDegree::P1,
            flux: CompressibleFluxKind::Rusanov,
        }
    }
}

impl CompressibleDgConfig {
    pub fn validate(&self) -> Result<(), CompressibleDgError> {
        if !self.gas.gamma.is_finite()
            || self.gas.gamma <= 1.0
            || !self.gas.gas_constant.is_finite()
            || self.gas.gas_constant <= 0.0
            || !self.viscosity.is_valid()
            || !self.prandtl.is_finite()
            || self.prandtl <= 0.0
            || !self.cfl.is_finite()
            || self.cfl <= 0.0
            || !self.sipg_penalty.is_finite()
            || self.sipg_penalty <= 0.0
            || !self.density_floor.is_finite()
            || self.density_floor <= 0.0
            || !self.pressure_floor.is_finite()
            || self.pressure_floor <= 0.0
            || !self.troubled_cell_relaxation.is_finite()
            || self.troubled_cell_relaxation < 0.0
            || self.maximum_step_retries == 0
            || self.degree == PolynomialDegree::P3
            || self.troubled_cell_threshold.is_nan()
            || self.troubled_cell_threshold <= 0.0
            || self.persson_threshold.is_nan()
        {
            return Err(CompressibleDgError::InvalidConfiguration);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressibleDgError {
    InvalidConfiguration,
    UnsupportedMesh,
    StateSizeMismatch,
    MissingBoundaryCondition,
    NonPhysicalCellAverage,
    NonPhysicalQuadratureState,
    StepRetriesExhausted,
}

/// Element-local DG conservative state. Values are stored for every local
/// Lagrange node of every element (P1: vertices; P2: vertices plus edge
/// midpoints, in `ElementType::local_edges` order); neighboring elements
/// never share entries.
#[derive(Debug, Clone, PartialEq)]
pub struct CompressibleDgState {
    pub dimension: usize,
    pub values: Vec<ConservativeState>,
    element_offsets: Vec<usize>,
    /// Polynomial degree the local node layout was built for.
    pub degree: PolynomialDegree,
    /// Quadrature weights of the local basis over the reference element
    /// (∫φᵢ dΩ / V), one per local node. The cell average is the weighted
    /// sum with these weights; for P2 the vertex weights are zero on
    /// triangles and negative on tetrahedra, so an arithmetic mean of the
    /// nodal values would not be the cell average.
    average_weights: Vec<f64>,
}

/// Local Lagrange-node count for one simplex element at `degree`.
fn local_node_count(element_type: ElementType, degree: PolynomialDegree) -> usize {
    match degree {
        PolynomialDegree::P1 => element_type.num_nodes_p1(),
        PolynomialDegree::P2 => element_type.num_nodes_p2(),
        PolynomialDegree::P3 => element_type.num_nodes_p3(),
    }
}

/// Physical positions of the local Lagrange nodes of one element: vertices
/// first, then (P2) edge midpoints in `local_edges` order. Geometry is
/// affine (P1-mapped), so midpoints are exact.
fn local_node_positions(mesh: &Mesh, element: usize, degree: PolynomialDegree) -> Vec<[f64; 3]> {
    let cell = &mesh.elements[element];
    let vertices = cell.vertices();
    let mut positions: Vec<[f64; 3]> = vertices
        .iter()
        .map(|&node| {
            let point = &mesh.nodes[node];
            [point.x, point.y, point.z]
        })
        .collect();
    if degree == PolynomialDegree::P2 {
        for &(a, b) in cell.element_type.local_edges() {
            let (pa, pb) = (positions[a], positions[b]);
            positions.push([
                0.5 * (pa[0] + pb[0]),
                0.5 * (pa[1] + pb[1]),
                0.5 * (pa[2] + pb[2]),
            ]);
        }
    }
    positions
}

/// Reference-element average weights ∫φᵢ/V per local node.
fn average_weights(element_type: ElementType, degree: PolynomialDegree) -> Vec<f64> {
    match (element_type, degree) {
        (ElementType::Triangle, PolynomialDegree::P1) => vec![1.0 / 3.0; 3],
        (ElementType::Triangle, PolynomialDegree::P2) => {
            vec![0.0, 0.0, 0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        }
        (ElementType::Tetrahedron, PolynomialDegree::P1) => vec![0.25; 4],
        (ElementType::Tetrahedron, PolynomialDegree::P2) => {
            let mut weights = vec![-1.0 / 20.0; 4];
            weights.extend([0.2; 6]);
            weights
        }
        _ => Vec::new(),
    }
}

impl CompressibleDgState {
    pub fn from_function<F>(mesh: &Mesh, initial: F) -> Result<Self, CompressibleDgError>
    where
        F: FnMut([f64; 3]) -> ConservativeState,
    {
        Self::from_function_degree(mesh, PolynomialDegree::P1, initial)
    }

    /// Build a state at an explicit polynomial degree by sampling the
    /// initial field at the element-local Lagrange nodes.
    pub fn from_function_degree<F>(
        mesh: &Mesh,
        degree: PolynomialDegree,
        mut initial: F,
    ) -> Result<Self, CompressibleDgError>
    where
        F: FnMut([f64; 3]) -> ConservativeState,
    {
        validate_simplex_mesh(mesh)?;
        if degree == PolynomialDegree::P3 {
            return Err(CompressibleDgError::UnsupportedMesh);
        }
        let mut values = Vec::new();
        let mut element_offsets = Vec::with_capacity(mesh.num_elements() + 1);
        element_offsets.push(0);
        for element in 0..mesh.num_elements() {
            for position in local_node_positions(mesh, element, degree) {
                let mut state = initial(position);
                if mesh.dimension == 2 {
                    state.momentum[2] = 0.0;
                }
                values.push(state);
            }
            element_offsets.push(values.len());
        }
        let weights = if mesh.num_elements() > 0 {
            average_weights(mesh.elements[0].element_type, degree)
        } else {
            Vec::new()
        };
        Ok(Self {
            dimension: mesh.dimension,
            values,
            element_offsets,
            degree,
            average_weights: weights,
        })
    }

    pub fn num_equations(&self) -> usize {
        self.dimension + 2
    }

    pub fn element(&self, element: usize) -> &[ConservativeState] {
        &self.values[self.element_offsets[element]..self.element_offsets[element + 1]]
    }

    fn element_mut(&mut self, element: usize) -> &mut [ConservativeState] {
        let start = self.element_offsets[element];
        let end = self.element_offsets[element + 1];
        &mut self.values[start..end]
    }

    pub fn cell_average(&self, element: usize) -> ConservativeState {
        let cell = self.element(element);
        if self.average_weights.len() != cell.len() {
            return average_states(cell);
        }
        let mut average = [0.0; 5];
        for (state, weight) in cell.iter().zip(&self.average_weights) {
            for (target, value) in average.iter_mut().zip(state.as_array()) {
                *target += weight * value;
            }
        }
        ConservativeState::from_array(average)
    }

    pub fn validate_layout(&self, mesh: &Mesh) -> Result<(), CompressibleDgError> {
        if self.dimension != mesh.dimension
            || self.element_offsets.len() != mesh.num_elements() + 1
            || self.element_offsets.last().copied() != Some(self.values.len())
            || mesh.elements.iter().enumerate().any(|(element, cell)| {
                self.element_offsets[element + 1] - self.element_offsets[element]
                    != local_node_count(cell.element_type, self.degree)
            })
        {
            return Err(CompressibleDgError::StateSizeMismatch);
        }
        Ok(())
    }

    fn add_scaled(&self, derivative: &[ConservativeState], scale: f64) -> Self {
        let mut result = self.clone();
        for (state, rate) in result.values.iter_mut().zip(derivative) {
            *state = combine_states(*state, 1.0, *rate, scale);
            if self.dimension == 2 {
                state.momentum[2] = 0.0;
            }
        }
        result
    }

    fn convex_combination(&self, a: f64, other: &Self, b: f64) -> Self {
        let mut result = self.clone();
        for ((out, left), right) in result
            .values
            .iter_mut()
            .zip(&self.values)
            .zip(&other.values)
        {
            *out = combine_states(*left, a, *right, b);
            if self.dimension == 2 {
                out.momentum[2] = 0.0;
            }
        }
        result
    }
}

pub struct CompressibleDgOperator<'a> {
    mesh: &'a Mesh,
    space: DGSpace,
    pub config: CompressibleDgConfig,
    /// Per-element inverse mass matrices (row-major, local-node sized),
    /// computed once by quadrature. At P1 this reproduces the closed-form
    /// simplex inverse the solver used previously.
    mass_inverses: Vec<Vec<f64>>,
    /// Interior facet list, built once. The residual and the limiter both
    /// consumed a fresh `build_interior_facets` (a hash-map pass over the
    /// whole mesh) on every SSPRK3 stage before this cache existed.
    interior_facets: Vec<InteriorFacet>,
    /// Per-element interior-facet indices (ascending), for the
    /// element-centric parallel facet gather.
    element_facets: Vec<Vec<usize>>,
    /// Per-element boundary-face indices into `mesh.boundaries`
    /// (ascending), for the element-centric parallel boundary gather.
    element_boundary_faces: Vec<Vec<usize>>,
}

/// Volume quadrature order for the operator at `degree`.
fn volume_quadrature_order(degree: PolynomialDegree) -> usize {
    match degree {
        PolynomialDegree::P1 => 3,
        _ => 5,
    }
}

/// Invert a small dense matrix by Gauss--Jordan elimination with partial
/// pivoting. Returns `None` for (near-)singular input.
fn invert_small(matrix: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut a: Vec<Vec<f64>> = (0..n)
        .map(|row| matrix[row * n..(row + 1) * n].to_vec())
        .collect();
    let mut inv: Vec<Vec<f64>> = (0..n)
        .map(|row| {
            (0..n)
                .map(|col| if row == col { 1.0 } else { 0.0 })
                .collect()
        })
        .collect();
    for col in 0..n {
        let pivot = (col..n).max_by(|&r1, &r2| {
            a[r1][col]
                .abs()
                .partial_cmp(&a[r2][col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        if a[pivot][col].abs() < 1.0e-300 {
            return None;
        }
        a.swap(col, pivot);
        inv.swap(col, pivot);
        let scale = a[col][col];
        for value in a[col].iter_mut().chain(inv[col].iter_mut()) {
            *value /= scale;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            if factor == 0.0 {
                continue;
            }
            for c in 0..n {
                a[row][c] -= factor * a[col][c];
                inv[row][c] -= factor * inv[col][c];
            }
        }
    }
    Some(inv.into_iter().flatten().collect())
}

impl<'a> CompressibleDgOperator<'a> {
    pub fn new(mesh: &'a Mesh, config: CompressibleDgConfig) -> Result<Self, CompressibleDgError> {
        validate_simplex_mesh(mesh)?;
        config.validate()?;
        let space = DGSpace::new(config.degree);
        // Precompute per-element inverse mass matrices by quadrature:
        // M_ij = sum_q w_q detJ_q phi_i(q) phi_j(q).
        let mut mass_inverses = Vec::with_capacity(mesh.num_elements());
        for element in 0..mesh.num_elements() {
            let n = local_node_count(mesh.elements[element].element_type, config.degree);
            let rule = QuadratureRule::new(
                mesh.elements[element].element_type,
                volume_quadrature_order(config.degree),
            );
            let mut mass = vec![0.0; n * n];
            for point in rule.iter() {
                let tab = space.tabulate(
                    mesh,
                    element,
                    point.coords[0],
                    point.coords[1],
                    point.coords[2],
                );
                let weight = point.weight * tab.det_j;
                for i in 0..n {
                    for j in 0..n {
                        mass[i * n + j] += weight * tab.values[i] * tab.values[j];
                    }
                }
            }
            let inverse = invert_small(&mass, n).ok_or(CompressibleDgError::UnsupportedMesh)?;
            mass_inverses.push(inverse);
        }
        let interior_facets = build_interior_facets(mesh);
        let mut element_facets: Vec<Vec<usize>> = vec![Vec::new(); mesh.num_elements()];
        for (index, facet) in interior_facets.iter().enumerate() {
            element_facets[facet.elem_plus].push(index);
            element_facets[facet.elem_minus].push(index);
        }
        let mut element_boundary_faces: Vec<Vec<usize>> = vec![Vec::new(); mesh.num_elements()];
        for (index, face) in mesh.boundaries.iter().enumerate() {
            element_boundary_faces[face.element_idx].push(index);
        }
        Ok(Self {
            mesh,
            space,
            config,
            mass_inverses,
            interior_facets,
            element_facets,
            element_boundary_faces,
        })
    }

    /// Map `work` over all elements, in parallel when the crate policy
    /// allows. Results are collected in element order, and every element
    /// block is computed with the same arithmetic order as the serial
    /// loops this replaces, so parallel and serial results agree bitwise
    /// (the `SONIUM_SERIAL=1` escape hatch selects the serial path).
    fn map_elements<T, F>(&self, work: F) -> Vec<T>
    where
        T: Send,
        F: Fn(usize) -> T + Sync + Send,
    {
        let count = self.mesh.num_elements();
        if ParallelPolicy::from_env().decide(count, 8) {
            (0..count).into_par_iter().map(work).collect()
        } else {
            (0..count).map(work).collect()
        }
    }

    fn check_state_degree(&self, state: &CompressibleDgState) -> Result<(), CompressibleDgError> {
        state.validate_layout(self.mesh)?;
        if state.degree != self.config.degree {
            return Err(CompressibleDgError::StateSizeMismatch);
        }
        Ok(())
    }

    /// Evaluate the semi-discrete `dU/dt` including Euler volume/facet fluxes,
    /// Newtonian stress, Fourier heat flux, and SIPG jump stabilization.
    /// Basis values of `element`'s local space at a physical point, via its
    /// barycentric coordinates. At P1 this returns the barycentric weights
    /// themselves; at P2 it evaluates the quadratic Lagrange basis.
    fn facet_basis(&self, element: usize, point: [f64; 3]) -> Vec<f64> {
        let lambda = barycentric_coordinates(self.mesh, element, point);
        let (xi, eta, zeta) = if self.mesh.dimension == 2 {
            (lambda[1], lambda[2], 0.0)
        } else {
            (lambda[1], lambda[2], lambda[3])
        };
        self.space
            .tabulate(self.mesh, element, xi, eta, zeta)
            .values
    }

    pub fn spatial_operator(
        &self,
        state: &CompressibleDgState,
    ) -> Result<Vec<ConservativeState>, CompressibleDgError> {
        self.check_state_degree(state)?;
        let mut integrated = vec![[0.0; 5]; state.values.len()];
        // Element-block stages: each element's contribution is computed
        // independently (the layout a future GPU kernel-per-element port
        // will reuse), then merged in element order.
        for stage in [
            Self::volume_block
                as fn(
                    &Self,
                    &CompressibleDgState,
                    usize,
                ) -> Result<Vec<[f64; 5]>, CompressibleDgError>,
            Self::facet_block,
            Self::boundary_block,
        ] {
            let blocks = self.map_elements(|element| stage(self, state, element));
            for (element, block) in blocks.into_iter().enumerate() {
                let block = block?;
                let offset = state.element_offsets[element];
                for (local, value) in block.into_iter().enumerate() {
                    for equation in 0..5 {
                        integrated[offset + local][equation] += value[equation];
                    }
                }
            }
        }
        Ok(self.apply_inverse_mass(&integrated))
    }

    pub fn stable_time_step(
        &self,
        state: &CompressibleDgState,
    ) -> Result<f64, CompressibleDgError> {
        self.check_state_degree(state)?;
        let locals = self.map_elements(|element| {
            let average = state.cell_average(element);
            if !average.is_physical(
                self.config.gas,
                self.config.density_floor,
                self.config.pressure_floor,
            ) {
                return Err(CompressibleDgError::NonPhysicalCellAverage);
            }
            let h = element_min_altitude(self.mesh, element);
            let velocity = average.velocity();
            let speed = norm3(velocity)
                + self
                    .config
                    .gas
                    .speed_of_sound(average.pressure(self.config.gas), average.density);
            let temperature = self
                .config
                .gas
                .temperature(average.pressure(self.config.gas), average.density);
            let mu = self.config.viscosity.dynamic_viscosity(temperature);
            let conductivity = self.config.viscosity.thermal_conductivity(
                temperature,
                self.config.gas,
                self.config.prandtl,
            );
            let diffusivity = (mu / average.density)
                .max(conductivity / (average.density * self.config.gas.specific_heat_cp()));
            let convective = h / speed.max(f64::MIN_POSITIVE);
            let viscous = 0.5 * h * h / diffusivity.max(f64::MIN_POSITIVE);
            // DG spectral radius grows like (2p + 1); scale the P1-tuned
            // CFL by 3 / (2p + 1) so `cfl` keeps its meaning across degrees.
            let degree_factor = 3.0 / (2.0 * self.config.degree.degree() as f64 + 1.0);
            Ok(self.config.cfl * degree_factor * convective.min(viscous))
        });
        let mut dt = f64::INFINITY;
        for local in locals {
            dt = dt.min(local?);
        }
        Ok(dt)
    }

    pub fn conserved_totals(&self, state: &CompressibleDgState) -> [f64; 5] {
        let mut totals = [0.0; 5];
        for element in 0..self.mesh.num_elements() {
            let volume = cell_volume(self.mesh, element);
            let average = state.cell_average(element).as_array();
            for equation in 0..5 {
                totals[equation] += volume * average[equation];
            }
        }
        totals
    }

    fn volume_block(
        &self,
        state: &CompressibleDgState,
        element: usize,
    ) -> Result<Vec<[f64; 5]>, CompressibleDgError> {
        let cell = state.element(element);
        let mut residual = vec![[0.0; 5]; cell.len()];
        let gradients = primitive_gradients(
            cell,
            &self.space.tabulate(self.mesh, element, 0.25, 0.25, 0.25),
            self.config.gas,
        );
        let rule = QuadratureRule::new(
            self.mesh.elements[element].element_type,
            volume_quadrature_order(self.config.degree),
        );
        for point in rule.iter() {
            let tab = self.space.tabulate(
                self.mesh,
                element,
                point.coords[0],
                point.coords[1],
                point.coords[2],
            );
            let value = interpolate_with_basis(cell, &tab.values);
            if !value.is_physical(
                self.config.gas,
                self.config.density_floor,
                self.config.pressure_floor,
            ) {
                return Err(CompressibleDgError::NonPhysicalQuadratureState);
            }
            let viscous = viscous_flux_tensor(
                value,
                gradients,
                self.config.gas,
                self.config.viscosity,
                self.config.prandtl,
            );
            let weight = point.weight * tab.det_j;
            for (local, res) in residual.iter_mut().enumerate().take(tab.n_dofs) {
                for direction in 0..self.mesh.dimension {
                    let mut normal = [0.0; 3];
                    normal[direction] = 1.0;
                    let inviscid = value.euler_flux(self.config.gas, normal);
                    for equation in 0..5 {
                        res[equation] += tab.grad(local, direction)
                            * (inviscid[equation] - viscous[direction][equation])
                            * weight;
                    }
                }
            }
        }
        Ok(residual)
    }

    /// Element-centric interior-facet gather: each element accumulates
    /// its own facet contributions (sign -1 as facet plus side, +1 as
    /// minus side), in ascending facet order, matching the accumulation
    /// order of the former facet-major loop exactly.
    fn facet_block(
        &self,
        state: &CompressibleDgState,
        element: usize,
    ) -> Result<Vec<[f64; 5]>, CompressibleDgError> {
        let cell = state.element(element);
        let mut residual = vec![[0.0; 5]; cell.len()];
        let own_gradients = primitive_gradients(
            cell,
            &self.space.tabulate(self.mesh, element, 0.25, 0.25, 0.25),
            self.config.gas,
        );
        for &facet_index in &self.element_facets[element] {
            let facet = &self.interior_facets[facet_index];
            let is_plus = facet.elem_plus == element;
            let other_element = if is_plus {
                facet.elem_minus
            } else {
                facet.elem_plus
            };
            let other = state.element(other_element);
            let other_gradients = primitive_gradients(
                other,
                &self
                    .space
                    .tabulate(self.mesh, other_element, 0.25, 0.25, 0.25),
                self.config.gas,
            );
            let plus_centroid = element_centroid(self.mesh, facet.elem_plus);
            let minus_centroid = element_centroid(self.mesh, facet.elem_minus);
            let mut normal = facet.normal;
            if dot3(normal, sub3(minus_centroid, plus_centroid)) < 0.0 {
                normal = scale3(normal, -1.0);
            }
            let (plus, minus) = if is_plus {
                (cell, other)
            } else {
                (other, cell)
            };
            let (plus_gradients, minus_gradients) = if is_plus {
                (own_gradients, other_gradients)
            } else {
                (other_gradients, own_gradients)
            };
            let h = 0.5
                * (element_diameter(self.mesh, facet.elem_plus)
                    + element_diameter(self.mesh, facet.elem_minus));
            for (point, weight) in
                facet_quadrature(self.mesh, &facet.shared_nodes, self.config.degree)
            {
                let plus_basis = self.facet_basis(facet.elem_plus, point);
                let minus_basis = self.facet_basis(facet.elem_minus, point);
                let left = interpolate_with_basis(plus, &plus_basis);
                let right = interpolate_with_basis(minus, &minus_basis);
                let flux =
                    self.numerical_flux(left, right, plus_gradients, minus_gradients, normal, h)?;
                let (basis, sign) = if is_plus {
                    (&plus_basis, -1.0)
                } else {
                    (&minus_basis, 1.0)
                };
                for local in 0..cell.len() {
                    for equation in 0..5 {
                        residual[local][equation] += sign * basis[local] * flux[equation] * weight;
                    }
                }
            }
        }
        Ok(residual)
    }

    fn boundary_block(
        &self,
        state: &CompressibleDgState,
        element: usize,
    ) -> Result<Vec<[f64; 5]>, CompressibleDgError> {
        let cell = state.element(element);
        let mut residual = vec![[0.0; 5]; cell.len()];
        if self.element_boundary_faces[element].is_empty() {
            return Ok(residual);
        }
        let gradients = primitive_gradients(
            cell,
            &self.space.tabulate(self.mesh, element, 0.25, 0.25, 0.25),
            self.config.gas,
        );
        for &face_index in &self.element_boundary_faces[element] {
            let face = &self.mesh.boundaries[face_index];
            let condition = self
                .config
                .boundaries
                .get(&face.marker)
                .copied()
                .ok_or(CompressibleDgError::MissingBoundaryCondition)?;
            let mut normal = facet_normal(self.mesh, &face.nodes);
            let face_centroid = points_centroid(self.mesh, &face.nodes);
            if dot3(
                normal,
                sub3(face_centroid, element_centroid(self.mesh, element)),
            ) < 0.0
            {
                normal = scale3(normal, -1.0);
            }
            let h = element_diameter(self.mesh, element);
            for (point, weight) in facet_quadrature(self.mesh, &face.nodes, self.config.degree) {
                let basis = self.facet_basis(element, point);
                let interior = interpolate_with_basis(cell, &basis);
                let exterior = boundary_state(condition, interior, normal, self.config.gas)?;
                let flux =
                    self.numerical_flux(interior, exterior, gradients, gradients, normal, h)?;
                for local in 0..cell.len() {
                    for equation in 0..5 {
                        residual[local][equation] -= basis[local] * flux[equation] * weight;
                    }
                }
            }
        }
        Ok(residual)
    }

    fn numerical_flux(
        &self,
        left: ConservativeState,
        right: ConservativeState,
        left_gradients: PrimitiveGradients,
        right_gradients: PrimitiveGradients,
        normal: [f64; 3],
        h: f64,
    ) -> Result<[f64; 5], CompressibleDgError> {
        for state in [left, right] {
            if !state.is_physical(
                self.config.gas,
                self.config.density_floor,
                self.config.pressure_floor,
            ) {
                return Err(CompressibleDgError::NonPhysicalQuadratureState);
            }
        }
        let inviscid = match self.config.flux {
            CompressibleFluxKind::Rusanov => rusanov_flux(left, right, self.config.gas, normal),
            CompressibleFluxKind::Hllc => hllc_flux(left, right, self.config.gas, normal)
                .unwrap_or_else(|| rusanov_flux(left, right, self.config.gas, normal)),
        };
        let left_viscous = viscous_flux_tensor(
            left,
            left_gradients,
            self.config.gas,
            self.config.viscosity,
            self.config.prandtl,
        );
        let right_viscous = viscous_flux_tensor(
            right,
            right_gradients,
            self.config.gas,
            self.config.viscosity,
            self.config.prandtl,
        );
        let left_temperature = self
            .config
            .gas
            .temperature(left.pressure(self.config.gas), left.density);
        let right_temperature = self
            .config
            .gas
            .temperature(right.pressure(self.config.gas), right.density);
        let maximum_diffusivity = (self.config.viscosity.dynamic_viscosity(left_temperature)
            / left.density)
            .max(self.config.viscosity.dynamic_viscosity(right_temperature) / right.density);
        let penalty = self.config.sipg_penalty * maximum_diffusivity / h.max(f64::MIN_POSITIVE);
        let left_array = left.as_array();
        let right_array = right.as_array();
        let mut flux = inviscid;
        for equation in 0..5 {
            let viscous_normal = (0..self.mesh.dimension)
                .map(|direction| {
                    0.5 * (left_viscous[direction][equation] + right_viscous[direction][equation])
                        * normal[direction]
                })
                .sum::<f64>();
            flux[equation] = inviscid[equation] - viscous_normal
                + penalty * (left_array[equation] - right_array[equation]);
        }
        Ok(flux)
    }

    fn apply_inverse_mass(&self, integrated: &[[f64; 5]]) -> Vec<ConservativeState> {
        // Element offsets, identical layout to the state's.
        let mut offsets = Vec::with_capacity(self.mesh.num_elements());
        let mut running = 0usize;
        for element in 0..self.mesh.num_elements() {
            offsets.push(running);
            running +=
                local_node_count(self.mesh.elements[element].element_type, self.config.degree);
        }
        let blocks = self.map_elements(|element| {
            let n = local_node_count(self.mesh.elements[element].element_type, self.config.degree);
            let offset = offsets[element];
            let inverse = &self.mass_inverses[element];
            let mut block = vec![ConservativeState::from_array([0.0; 5]); n];
            for equation in 0..5 {
                for local in 0..n {
                    let value: f64 = (0..n)
                        .map(|other| {
                            inverse[local * n + other] * integrated[offset + other][equation]
                        })
                        .sum();
                    let mut array = block[local].as_array();
                    array[equation] = value;
                    block[local] = ConservativeState::from_array(array);
                }
            }
            block
        });
        blocks.into_iter().flatten().collect()
    }
}

#[cfg(feature = "cuda")]
impl<'a> CompressibleDgOperator<'a> {
    /// Stage this operator as flat device tables for
    /// `sonium_compute::cuda::dg`.
    ///
    /// Every table is produced by the same CPU helpers the operator
    /// itself uses (quadrature tabulations, facet quadrature, Persson
    /// matrices), so the device kernels consume identical numbers.
    /// Boundary-condition parameters are validated here with the same
    /// checks `boundary_state` applies per evaluation.
    pub(crate) fn build_cuda_problem(
        &self,
        state: &CompressibleDgState,
    ) -> Result<sonium_compute::cuda::dg::CudaDgProblem, CompressibleDgError> {
        use sonium_compute::cuda::dg::{CudaDgProblem, DgParams};
        self.check_state_degree(state)?;
        let mesh = self.mesh;
        let degree = self.config.degree;
        let element_type = mesh.elements[0].element_type;
        let n = local_node_count(element_type, degree);
        let dim = mesh.dimension;
        let average_weights = if state.average_weights.len() == n {
            state.average_weights.clone()
        } else {
            vec![1.0 / n as f64; n]
        };
        let (flux_kind, sensor_mode, persson) = {
            let flux_kind = match self.config.flux {
                CompressibleFluxKind::Rusanov => 0,
                CompressibleFluxKind::Hllc => 1,
            };
            if self.config.persson_threshold.is_finite() {
                match persson_matrices(element_type, degree) {
                    Some(matrices) => (flux_kind, 0, Some(matrices)),
                    None => (flux_kind, 2, None),
                }
            } else if self.config.troubled_cell_threshold.is_finite() {
                (flux_kind, 1, None)
            } else {
                (flux_kind, 2, None)
            }
        };
        let (visc_kind, visc_a, visc_b, visc_c) = match self.config.viscosity {
            ViscosityModel::Constant(mu) => (0, mu, 0.0, 0.0),
            ViscosityModel::Sutherland {
                reference_viscosity,
                reference_temperature,
                sutherland_temperature,
            } => (
                1,
                reference_viscosity,
                reference_temperature,
                sutherland_temperature,
            ),
        };
        let params = DgParams {
            dim: dim as i32,
            n: n as i32,
            flux_kind,
            visc_kind,
            sensor_mode,
            persson_highest: persson
                .as_ref()
                .map_or(0, |matrices| matrices.highest_start as i32),
            gamma: self.config.gas.gamma,
            gas_constant: self.config.gas.gas_constant,
            prandtl: self.config.prandtl,
            visc_a,
            visc_b,
            visc_c,
            density_floor: self.config.density_floor,
            pressure_floor: self.config.pressure_floor,
            sipg_penalty: self.config.sipg_penalty,
            cfl_scaled: self.config.cfl * 3.0 / (2.0 * degree.degree() as f64 + 1.0),
            relaxation: self.config.troubled_cell_relaxation,
            threshold: self.config.troubled_cell_threshold,
            persson_threshold: self.config.persson_threshold,
        };
        // Volume quadrature tables: basis values are reference-based
        // (shared), gradients and weights are per element.
        let rule = QuadratureRule::new(element_type, volume_quadrature_order(degree));
        let points: Vec<_> = rule.iter().collect();
        let nq_vol = points.len();
        let mut vol_basis = Vec::with_capacity(nq_vol * n);
        for point in &points {
            let tab =
                self.space
                    .tabulate(mesh, 0, point.coords[0], point.coords[1], point.coords[2]);
            vol_basis.extend_from_slice(&tab.values[..n]);
        }
        let mut vol_grad = Vec::with_capacity(mesh.num_elements() * nq_vol * n * 3);
        let mut vol_wdet = Vec::with_capacity(mesh.num_elements() * nq_vol);
        let mut cent_grad = Vec::with_capacity(mesh.num_elements() * n * 3);
        for element in 0..mesh.num_elements() {
            for point in &points {
                let tab = self.space.tabulate(
                    mesh,
                    element,
                    point.coords[0],
                    point.coords[1],
                    point.coords[2],
                );
                for local in 0..n {
                    for direction in 0..3 {
                        vol_grad.push(if direction < tab.spatial_dim {
                            tab.grad(local, direction)
                        } else {
                            0.0
                        });
                    }
                }
                vol_wdet.push(point.weight * tab.det_j);
            }
            let tab = self.space.tabulate(mesh, element, 0.25, 0.25, 0.25);
            for local in 0..n {
                for direction in 0..3 {
                    cent_grad.push(if direction < tab.spatial_dim {
                        tab.grad(local, direction)
                    } else {
                        0.0
                    });
                }
            }
        }
        // Interior-facet CSR in the element-centric gather order.
        let mut facet_elem_offsets = vec![0i32];
        let mut facet_neighbor = Vec::new();
        let mut facet_is_plus = Vec::new();
        let mut facet_normals = Vec::new();
        let mut facet_h = Vec::new();
        let mut facet_quad_offsets = vec![0i32];
        let mut facet_basis_own = Vec::new();
        let mut facet_basis_other = Vec::new();
        let mut facet_weights = Vec::new();
        for element in 0..mesh.num_elements() {
            for &facet_index in &self.element_facets[element] {
                let facet = &self.interior_facets[facet_index];
                let is_plus = facet.elem_plus == element;
                let other = if is_plus {
                    facet.elem_minus
                } else {
                    facet.elem_plus
                };
                let plus_centroid = element_centroid(mesh, facet.elem_plus);
                let minus_centroid = element_centroid(mesh, facet.elem_minus);
                let mut normal = facet.normal;
                if dot3(normal, sub3(minus_centroid, plus_centroid)) < 0.0 {
                    normal = scale3(normal, -1.0);
                }
                facet_neighbor.push(other as i32);
                facet_is_plus.push(i32::from(is_plus));
                facet_normals.extend_from_slice(&normal);
                facet_h.push(
                    0.5 * (element_diameter(mesh, facet.elem_plus)
                        + element_diameter(mesh, facet.elem_minus)),
                );
                for (point, weight) in facet_quadrature(mesh, &facet.shared_nodes, degree) {
                    let plus_basis = self.facet_basis(facet.elem_plus, point);
                    let minus_basis = self.facet_basis(facet.elem_minus, point);
                    let (own, other_basis) = if is_plus {
                        (&plus_basis, &minus_basis)
                    } else {
                        (&minus_basis, &plus_basis)
                    };
                    facet_basis_own.extend_from_slice(&own[..n]);
                    facet_basis_other.extend_from_slice(&other_basis[..n]);
                    facet_weights.push(weight);
                }
                facet_quad_offsets.push(facet_weights.len() as i32);
            }
            facet_elem_offsets.push(facet_neighbor.len() as i32);
        }
        // Boundary-face CSR with conditions resolved to kernel kinds.
        let mut bnd_elem_offsets = vec![0i32];
        let mut bnd_kind = Vec::new();
        let mut bnd_params = Vec::new();
        let mut bnd_normal = Vec::new();
        let mut bnd_h = Vec::new();
        let mut bnd_quad_offsets = vec![0i32];
        let mut bnd_basis = Vec::new();
        let mut bnd_weights = Vec::new();
        for element in 0..mesh.num_elements() {
            for &face_index in &self.element_boundary_faces[element] {
                let face = &mesh.boundaries[face_index];
                let condition = self
                    .config
                    .boundaries
                    .get(&face.marker)
                    .copied()
                    .ok_or(CompressibleDgError::MissingBoundaryCondition)?;
                let (kind, params8) = encode_boundary_condition(condition)?;
                let mut normal = facet_normal(mesh, &face.nodes);
                let face_centroid = points_centroid(mesh, &face.nodes);
                if dot3(normal, sub3(face_centroid, element_centroid(mesh, element))) < 0.0 {
                    normal = scale3(normal, -1.0);
                }
                bnd_kind.push(kind);
                bnd_params.extend_from_slice(&params8);
                bnd_normal.extend_from_slice(&normal);
                bnd_h.push(element_diameter(mesh, element));
                for (point, weight) in facet_quadrature(mesh, &face.nodes, degree) {
                    bnd_basis.extend_from_slice(&self.facet_basis(element, point)[..n]);
                    bnd_weights.push(weight);
                }
                bnd_quad_offsets.push(bnd_weights.len() as i32);
            }
            bnd_elem_offsets.push(bnd_kind.len() as i32);
        }
        // Limiter neighbour CSR, built exactly as the CPU limiter does.
        let mut neighbours: Vec<Vec<(usize, [f64; 3])>> = vec![Vec::new(); mesh.num_elements()];
        for facet in &self.interior_facets {
            let mut normal = facet.normal;
            if dot3(
                normal,
                sub3(
                    element_centroid(mesh, facet.elem_minus),
                    element_centroid(mesh, facet.elem_plus),
                ),
            ) < 0.0
            {
                normal = scale3(normal, -1.0);
            }
            neighbours[facet.elem_plus].push((facet.elem_minus, normal));
            neighbours[facet.elem_minus].push((facet.elem_plus, scale3(normal, -1.0)));
        }
        let mut lim_offsets = vec![0i32];
        let mut lim_neighbor = Vec::new();
        let mut lim_normal = Vec::new();
        for element_neighbours in &neighbours {
            for &(neighbour, normal) in element_neighbours {
                lim_neighbor.push(neighbour as i32);
                lim_normal.extend_from_slice(&normal);
            }
            lim_offsets.push(lim_neighbor.len() as i32);
        }
        Ok(CudaDgProblem {
            params,
            num_elements: mesh.num_elements(),
            nq_vol,
            vol_basis,
            vol_grad,
            vol_wdet,
            cent_grad,
            mass_inv: self.mass_inverses.concat(),
            avg_weights: average_weights,
            elem_min_altitude: (0..mesh.num_elements())
                .map(|element| element_min_altitude(mesh, element))
                .collect(),
            elem_volume: (0..mesh.num_elements())
                .map(|element| cell_volume(mesh, element))
                .collect(),
            facet_elem_offsets,
            facet_neighbor,
            facet_is_plus,
            facet_normal: facet_normals,
            facet_h,
            facet_quad_offsets,
            facet_basis_own,
            facet_basis_other,
            facet_weights,
            bnd_elem_offsets,
            bnd_kind,
            bnd_params,
            bnd_normal,
            bnd_h,
            bnd_quad_offsets,
            bnd_basis,
            bnd_weights,
            lim_offsets,
            lim_neighbor,
            lim_normal,
            persson_vinv: persson
                .as_ref()
                .map_or_else(Vec::new, |matrices| matrices.vandermonde_inverse.clone()),
            persson_gram: persson
                .as_ref()
                .map_or_else(Vec::new, |matrices| matrices.gram.clone()),
        })
    }
}

/// Encode a boundary condition as `(kind, params[8])` for the device
/// kernels, applying the same validation `boundary_state` performs.
#[cfg(feature = "cuda")]
fn encode_boundary_condition(
    condition: CompressibleBoundaryCondition,
) -> Result<(i32, [f64; 8]), CompressibleDgError> {
    let mut params = [0.0; 8];
    let kind = match condition {
        CompressibleBoundaryCondition::SlipWall => 0,
        CompressibleBoundaryCondition::NoSlipAdiabatic { wall_velocity } => {
            params[..3].copy_from_slice(&wall_velocity);
            1
        }
        CompressibleBoundaryCondition::NoSlipIsothermal {
            wall_velocity,
            temperature,
        } => {
            if !temperature.is_finite() || temperature <= 0.0 {
                return Err(CompressibleDgError::InvalidConfiguration);
            }
            params[..3].copy_from_slice(&wall_velocity);
            params[3] = temperature;
            2
        }
        CompressibleBoundaryCondition::FarField(far_field) => {
            params[..5].copy_from_slice(&far_field.as_array());
            3
        }
        CompressibleBoundaryCondition::SupersonicState(prescribed) => {
            params[..5].copy_from_slice(&prescribed.as_array());
            4
        }
        CompressibleBoundaryCondition::SubsonicTotalInlet {
            total_pressure,
            total_temperature,
            direction,
        } => {
            if total_pressure <= 0.0 || total_temperature <= 0.0 || norm3(direction) <= f64::EPSILON
            {
                return Err(CompressibleDgError::InvalidConfiguration);
            }
            params[0] = total_pressure;
            params[1] = total_temperature;
            params[2..5].copy_from_slice(&direction);
            5
        }
        CompressibleBoundaryCondition::StaticPressureOutlet { pressure } => {
            if pressure <= 0.0 || !pressure.is_finite() {
                return Err(CompressibleDgError::InvalidConfiguration);
            }
            params[0] = pressure;
            6
        }
    };
    Ok((kind, params))
}

pub struct CompressibleDgSolver<'a> {
    pub operator: CompressibleDgOperator<'a>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompressibleStepReport {
    pub accepted_time_step: f64,
    pub retries: usize,
    pub minimum_density: f64,
    pub minimum_pressure: f64,
    pub conserved_before: [f64; 5],
    pub conserved_after: [f64; 5],
}

impl<'a> CompressibleDgSolver<'a> {
    pub fn new(mesh: &'a Mesh, config: CompressibleDgConfig) -> Result<Self, CompressibleDgError> {
        Ok(Self {
            operator: CompressibleDgOperator::new(mesh, config)?,
        })
    }

    /// Advance one SSPRK3 step. A nonphysical cell average rejects the trial
    /// and halves `dt`, up to the configured retry limit.
    pub fn step(
        &self,
        state: &mut CompressibleDgState,
        requested_time_step: f64,
    ) -> Result<CompressibleStepReport, CompressibleDgError> {
        self.operator.check_state_degree(state)?;
        let before = self.operator.conserved_totals(state);
        let original = state.clone();
        let stable = self.operator.stable_time_step(state)?;
        let mut dt = requested_time_step.min(stable);
        for retry in 0..self.operator.config.maximum_step_retries {
            match self.ssprk3_trial(&original, dt) {
                Ok(candidate) => {
                    let (minimum_density, minimum_pressure) =
                        minimum_thermodynamics(&candidate, self.operator.config.gas);
                    *state = candidate;
                    return Ok(CompressibleStepReport {
                        accepted_time_step: dt,
                        retries: retry,
                        minimum_density,
                        minimum_pressure,
                        conserved_before: before,
                        conserved_after: self.operator.conserved_totals(state),
                    });
                }
                Err(CompressibleDgError::NonPhysicalCellAverage)
                | Err(CompressibleDgError::NonPhysicalQuadratureState) => dt *= 0.5,
                Err(error) => return Err(error),
            }
        }
        Err(CompressibleDgError::StepRetriesExhausted)
    }

    fn ssprk3_trial(
        &self,
        initial: &CompressibleDgState,
        dt: f64,
    ) -> Result<CompressibleDgState, CompressibleDgError> {
        let rate0 = self.operator.spatial_operator(initial)?;
        let mut stage1 = initial.add_scaled(&rate0, dt);
        self.limit(&mut stage1)?;

        let rate1 = self.operator.spatial_operator(&stage1)?;
        let euler1 = stage1.add_scaled(&rate1, dt);
        let mut stage2 = initial.convex_combination(0.75, &euler1, 0.25);
        self.limit(&mut stage2)?;

        let rate2 = self.operator.spatial_operator(&stage2)?;
        let euler2 = stage2.add_scaled(&rate2, dt);
        let mut stage3 = initial.convex_combination(1.0 / 3.0, &euler2, 2.0 / 3.0);
        self.limit(&mut stage3)?;
        Ok(stage3)
    }

    fn limit(&self, state: &mut CompressibleDgState) -> Result<(), CompressibleDgError> {
        apply_characteristic_vertex_limiter(
            self.operator.mesh,
            state,
            self.operator.config.gas,
            self.operator.config.troubled_cell_relaxation,
            self.operator.config.troubled_cell_threshold,
            self.operator.config.persson_threshold,
            &self.operator.interior_facets,
        );
        apply_zhang_shu_positivity(
            state,
            self.operator.config.gas,
            self.operator.config.density_floor,
            self.operator.config.pressure_floor,
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct PrimitiveGradients {
    velocity: [[f64; 3]; 3],
    temperature: [f64; 3],
}

fn primitive_gradients(
    cell: &[ConservativeState],
    tab: &BasisTabulation,
    gas: IdealGas,
) -> PrimitiveGradients {
    let mut velocity = [[0.0; 3]; 3];
    let mut temperature = [0.0; 3];
    for (local, state) in cell.iter().enumerate() {
        let nodal_velocity = state.velocity();
        let nodal_temperature = gas.temperature(state.pressure(gas), state.density);
        for direction in 0..tab.spatial_dim {
            for component in 0..3 {
                velocity[component][direction] +=
                    nodal_velocity[component] * tab.grad(local, direction);
            }
            temperature[direction] += nodal_temperature * tab.grad(local, direction);
        }
    }
    PrimitiveGradients {
        velocity,
        temperature,
    }
}

fn viscous_flux_tensor(
    state: ConservativeState,
    gradients: PrimitiveGradients,
    gas: IdealGas,
    viscosity: ViscosityModel,
    prandtl: f64,
) -> [[f64; 5]; 3] {
    let pressure = state.pressure(gas);
    let temperature = gas.temperature(pressure, state.density);
    let mu = viscosity.dynamic_viscosity(temperature);
    let conductivity = viscosity.thermal_conductivity(temperature, gas, prandtl);
    let velocity = state.velocity();
    let divergence = gradients.velocity[0][0] + gradients.velocity[1][1] + gradients.velocity[2][2];
    let mut stress = [[0.0; 3]; 3];
    for (component, row) in stress.iter_mut().enumerate() {
        for (direction, slot) in row.iter_mut().enumerate() {
            *slot = mu
                * (gradients.velocity[component][direction]
                    + gradients.velocity[direction][component]
                    - if component == direction {
                        2.0 * divergence / 3.0
                    } else {
                        0.0
                    });
        }
    }
    let mut flux = [[0.0; 5]; 3];
    for direction in 0..3 {
        flux[direction][1] = stress[0][direction];
        flux[direction][2] = stress[1][direction];
        flux[direction][3] = stress[2][direction];
        flux[direction][4] = velocity
            .iter()
            .enumerate()
            .map(|(component, value)| value * stress[component][direction])
            .sum::<f64>()
            + conductivity * gradients.temperature[direction];
    }
    flux
}

fn boundary_state(
    condition: CompressibleBoundaryCondition,
    interior: ConservativeState,
    normal: [f64; 3],
    gas: IdealGas,
) -> Result<ConservativeState, CompressibleDgError> {
    let pressure = interior.pressure(gas);
    let velocity = interior.velocity();
    let state = match condition {
        CompressibleBoundaryCondition::SlipWall => {
            let normal_velocity = dot3(velocity, normal);
            let reflected = sub3(velocity, scale3(normal, 2.0 * normal_velocity));
            ConservativeState::from_primitive(interior.density, reflected, pressure, gas)
        }
        CompressibleBoundaryCondition::NoSlipAdiabatic { wall_velocity } => {
            let reflected = sub3(scale3(wall_velocity, 2.0), velocity);
            ConservativeState::from_primitive(interior.density, reflected, pressure, gas)
        }
        CompressibleBoundaryCondition::NoSlipIsothermal {
            wall_velocity,
            temperature,
        } => {
            if !temperature.is_finite() || temperature <= 0.0 {
                return Err(CompressibleDgError::InvalidConfiguration);
            }
            let interior_temperature = gas.temperature(pressure, interior.density);
            let ghost_temperature = (2.0 * temperature - interior_temperature).max(1.0e-12);
            let density = pressure / (gas.gas_constant * ghost_temperature);
            let reflected = sub3(scale3(wall_velocity, 2.0), velocity);
            ConservativeState::from_primitive(density, reflected, pressure, gas)
        }
        CompressibleBoundaryCondition::FarField(far_field) => {
            characteristic_far_field_state(interior, far_field, normal, gas)
        }
        CompressibleBoundaryCondition::SupersonicState(prescribed) => prescribed,
        CompressibleBoundaryCondition::SubsonicTotalInlet {
            total_pressure,
            total_temperature,
            direction,
        } => {
            let direction_norm = norm3(direction);
            if total_pressure <= 0.0 || total_temperature <= 0.0 || direction_norm <= f64::EPSILON {
                return Err(CompressibleDgError::InvalidConfiguration);
            }
            let unit_direction = scale3(direction, 1.0 / direction_norm);
            let speed = norm3(velocity);
            let sound = gas.speed_of_sound(pressure, interior.density);
            let mach = (speed / sound).clamp(0.0, 0.999);
            let factor = 1.0 + 0.5 * (gas.gamma - 1.0) * mach * mach;
            let static_temperature = total_temperature / factor;
            let static_pressure = total_pressure / factor.powf(gas.gamma / (gas.gamma - 1.0));
            let density = static_pressure / (gas.gas_constant * static_temperature);
            let inlet_speed = mach * (gas.gamma * gas.gas_constant * static_temperature).sqrt();
            ConservativeState::from_primitive(
                density,
                scale3(unit_direction, inlet_speed),
                static_pressure,
                gas,
            )
        }
        CompressibleBoundaryCondition::StaticPressureOutlet { pressure } => {
            if pressure <= 0.0 || !pressure.is_finite() {
                return Err(CompressibleDgError::InvalidConfiguration);
            }
            ConservativeState::from_primitive(interior.density, velocity, pressure, gas)
        }
    };
    Ok(state)
}

fn characteristic_far_field_state(
    interior: ConservativeState,
    far_field: ConservativeState,
    normal: [f64; 3],
    gas: IdealGas,
) -> ConservativeState {
    let ui = interior.velocity();
    let uf = far_field.velocity();
    let pi = interior.pressure(gas);
    let pf = far_field.pressure(gas);
    let ci = gas.speed_of_sound(pi, interior.density);
    let cf = gas.speed_of_sound(pf, far_field.density);
    let uni = dot3(ui, normal);
    if uni >= ci {
        return interior;
    }
    if uni <= -ci {
        return far_field;
    }
    let unf = dot3(uf, normal);
    let r_plus = uni + 2.0 * ci / (gas.gamma - 1.0);
    let r_minus = unf - 2.0 * cf / (gas.gamma - 1.0);
    let normal_velocity = 0.5 * (r_plus + r_minus);
    let sound_speed = 0.25 * (gas.gamma - 1.0) * (r_plus - r_minus);
    let upstream = if normal_velocity < 0.0 {
        far_field
    } else {
        interior
    };
    let entropy = upstream.pressure(gas) / upstream.density.powf(gas.gamma);
    let density = (sound_speed * sound_speed / (gas.gamma * entropy))
        .max(f64::MIN_POSITIVE)
        .powf(1.0 / (gas.gamma - 1.0));
    let pressure = entropy * density.powf(gas.gamma);
    let upstream_velocity = upstream.velocity();
    let tangential = sub3(
        upstream_velocity,
        scale3(normal, dot3(upstream_velocity, normal)),
    );
    ConservativeState::from_primitive(
        density,
        add3(tangential, scale3(normal, normal_velocity)),
        pressure,
        gas,
    )
}

/// Monomial exponents up to `degree` in `dim` variables, ordered by total
/// degree (constant, linears, then quadratics). The count equals the
/// simplex Lagrange-node count at that degree.
fn monomial_exponents(dim: usize, degree: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    for total in 0..=degree {
        // Generate compositions of `total` into `dim` parts.
        fn compositions(
            dim: usize,
            total: usize,
            current: &mut Vec<usize>,
            out: &mut Vec<Vec<usize>>,
        ) {
            if current.len() == dim - 1 {
                let mut entry = current.clone();
                entry.push(total - current.iter().sum::<usize>());
                out.push(entry);
                return;
            }
            let used: usize = current.iter().sum();
            for value in 0..=(total - used) {
                current.push(value);
                compositions(dim, total, current, out);
                current.pop();
            }
        }
        compositions(dim, total, &mut Vec::new(), &mut result);
    }
    result
}

fn monomial_value(exponents: &[usize], point: [f64; 3]) -> f64 {
    exponents
        .iter()
        .enumerate()
        .map(|(axis, &power)| point[axis].powi(power as i32))
        .product()
}

/// Precomputed Persson--Peraire data for one element type and degree:
/// the nodal-to-modal (monomial) transform and the modal Gram matrix on
/// the reference element. Both energy terms scale with the same Jacobian
/// determinant on affine elements, so their ratio is mesh-independent.
struct PerssonMatrices {
    vandermonde_inverse: Vec<f64>,
    gram: Vec<f64>,
    n: usize,
    highest_start: usize,
}

fn persson_matrices(
    element_type: ElementType,
    degree: PolynomialDegree,
) -> Option<PerssonMatrices> {
    let dim = element_type.dimension();
    let degree_value = degree.degree();
    let exponents = monomial_exponents(dim, degree_value);
    let n = exponents.len();
    if n != local_node_count(element_type, degree) {
        return None;
    }
    let corners: &[[f64; 3]] = if dim == 2 {
        &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    } else {
        &[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    };
    let mut nodes: Vec<[f64; 3]> = corners[..element_type.num_vertices()].to_vec();
    if degree == PolynomialDegree::P2 {
        for &(a, b) in element_type.local_edges() {
            nodes.push(std::array::from_fn(|axis| {
                0.5 * (corners[a][axis] + corners[b][axis])
            }));
        }
    }
    if nodes.len() != n {
        return None;
    }
    let mut vandermonde = vec![0.0; n * n];
    for (node, point) in nodes.iter().enumerate() {
        for (mode, exponents) in exponents.iter().enumerate() {
            vandermonde[node * n + mode] = monomial_value(exponents, *point);
        }
    }
    let vandermonde_inverse = invert_small(&vandermonde, n)?;
    let rule = QuadratureRule::new(element_type, 5);
    let mut gram = vec![0.0; n * n];
    for point in rule.iter() {
        let coords = [point.coords[0], point.coords[1], point.coords[2]];
        let values: Vec<f64> = exponents
            .iter()
            .map(|exponents| monomial_value(exponents, coords))
            .collect();
        for row in 0..n {
            for col in 0..n {
                gram[row * n + col] += point.weight * values[row] * values[col];
            }
        }
    }
    let highest_start = exponents
        .iter()
        .filter(|exponents| exponents.iter().sum::<usize>() < degree_value)
        .count();
    Some(PerssonMatrices {
        vandermonde_inverse,
        gram,
        n,
        highest_start,
    })
}

/// Persson--Peraire smoothness indicator for one element's density:
/// `log10` of the fraction of modal energy in the highest-degree modes.
/// Smooth (polynomial-degree < p) fields give `-inf`; discontinuities
/// give values near 0.
fn persson_indicator(density: &[f64], matrices: &PerssonMatrices) -> f64 {
    let n = matrices.n;
    let coefficients: Vec<f64> = (0..n)
        .map(|mode| {
            (0..n)
                .map(|node| matrices.vandermonde_inverse[mode * n + node] * density[node])
                .sum()
        })
        .collect();
    let energy = |range: std::ops::Range<usize>| -> f64 {
        let mut total = 0.0;
        for row in range.clone() {
            for col in range.clone() {
                total += coefficients[row] * matrices.gram[row * n + col] * coefficients[col];
            }
        }
        total
    };
    let total = energy(0..n);
    if total <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let highest = energy(matrices.highest_start..n).max(0.0);
    if highest <= 0.0 {
        return f64::NEG_INFINITY;
    }
    (highest / total).log10()
}

/// Troubled-cell sensor with two criteria, either of which flags the
/// element for limiting:
/// 1. the relative density/pressure jump of the cell average to any facet
///    neighbour exceeds `threshold` (captured discontinuities), or
/// 2. any local nodal density/pressure deviates from the cell average by
///    more than `threshold` (within-element oscillation at a shock foot,
///    which the cell-average jump alone misses: a Sod run with only
///    criterion 1 exhausts its positivity retries).
///
/// Smooth gradients shrink both measures with the mesh; discontinuities
/// keep them O(1).
fn is_troubled_cell(
    cell: &[ConservativeState],
    average: ConservativeState,
    neighbors: &[(usize, [f64; 3])],
    averages: &[ConservativeState],
    gas: IdealGas,
    threshold: f64,
) -> bool {
    let pressure = average.pressure(gas);
    let density_scale = average.density.abs().max(1.0e-300);
    let pressure_scale = pressure.abs().max(1.0e-300);
    let within_element = cell.iter().any(|nodal| {
        (nodal.density - average.density).abs() / density_scale > threshold
            || (nodal.pressure(gas) - pressure).abs() / pressure_scale > threshold
    });
    if within_element {
        return true;
    }
    neighbors.iter().any(|&(neighbor, _)| {
        let other = averages[neighbor];
        let density_jump = (other.density - average.density).abs() / density_scale;
        let pressure_jump = (other.pressure(gas) - pressure).abs() / pressure_scale;
        density_jump > threshold || pressure_jump > threshold
    })
}

#[allow(clippy::too_many_arguments)]
fn apply_characteristic_vertex_limiter(
    mesh: &Mesh,
    state: &mut CompressibleDgState,
    gas: IdealGas,
    relaxation: f64,
    threshold: f64,
    persson_threshold: f64,
    interior_facets: &[InteriorFacet],
) {
    let original = state.clone();
    let averages = (0..mesh.num_elements())
        .map(|element| original.cell_average(element))
        .collect::<Vec<_>>();
    let persson_matrices = if persson_threshold.is_finite() && mesh.num_elements() > 0 {
        persson_matrices(mesh.elements[0].element_type, state.degree)
    } else {
        None
    };
    let mut neighbors: Vec<Vec<(usize, [f64; 3])>> = vec![Vec::new(); mesh.num_elements()];
    for facet in interior_facets {
        let mut normal = facet.normal;
        if dot3(
            normal,
            sub3(
                element_centroid(mesh, facet.elem_minus),
                element_centroid(mesh, facet.elem_plus),
            ),
        ) < 0.0
        {
            normal = scale3(normal, -1.0);
        }
        neighbors[facet.elem_plus].push((facet.elem_minus, normal));
        neighbors[facet.elem_minus].push((facet.elem_plus, scale3(normal, -1.0)));
    }
    // Per-element limiter factors, computed in parallel (each element
    // reads only the frozen original state), then applied in order.
    let element_theta = |element: usize| -> Option<f64> {
        if neighbors[element].is_empty() {
            return None;
        }
        let average = averages[element];
        // Troubled-cell decision. With a finite Persson--Peraire
        // threshold it is the primary criterion, backed by a strong-jump
        // safety net (relative average jump above 0.5); otherwise the
        // jump/deviation sensor decides, and an infinite threshold there
        // disables sensing entirely (legacy limit-everything).
        let troubled = if persson_threshold.is_finite() {
            let density: Vec<f64> = original
                .element(element)
                .iter()
                .map(|nodal| nodal.density)
                .collect();
            let smooth = persson_matrices
                .as_ref()
                .is_some_and(|matrices| persson_indicator(&density, matrices) <= persson_threshold);
            let strong_jump = neighbors[element].iter().any(|&(neighbor, _)| {
                let other = averages[neighbor];
                (other.density - average.density).abs() / average.density.abs().max(1.0e-300) > 0.5
                    || (other.pressure(gas) - average.pressure(gas)).abs()
                        / average.pressure(gas).abs().max(1.0e-300)
                        > 0.5
            });
            !smooth || strong_jump
        } else if threshold.is_finite() {
            is_troubled_cell(
                original.element(element),
                average,
                &neighbors[element],
                &averages,
                gas,
                threshold,
            )
        } else {
            true
        };
        if !troubled {
            return None;
        }
        let mut theta = 1.0_f64;
        for &(neighbor, normal) in &neighbors[element] {
            let center_characteristics = characteristic_variables(average, normal, gas);
            let neighbor_characteristics =
                characteristic_variables(averages[neighbor], normal, gas);
            for nodal in original.element(element) {
                let nodal_characteristics = characteristic_variables(*nodal, normal, gas);
                for wave in 0..5 {
                    let delta_neighbor =
                        neighbor_characteristics[wave] - center_characteristics[wave];
                    let margin = relaxation * delta_neighbor.abs().max(1.0e-12);
                    let lower =
                        center_characteristics[wave].min(neighbor_characteristics[wave]) - margin;
                    let upper =
                        center_characteristics[wave].max(neighbor_characteristics[wave]) + margin;
                    let delta = nodal_characteristics[wave] - center_characteristics[wave];
                    if delta > 0.0 {
                        theta = theta.min((upper - center_characteristics[wave]) / delta);
                    } else if delta < 0.0 {
                        theta = theta.min((lower - center_characteristics[wave]) / delta);
                    }
                }
            }
        }
        Some(theta)
    };
    let thetas: Vec<Option<f64>> = if ParallelPolicy::from_env().decide(mesh.num_elements(), 8) {
        (0..mesh.num_elements())
            .into_par_iter()
            .map(&element_theta)
            .collect()
    } else {
        (0..mesh.num_elements()).map(&element_theta).collect()
    };
    for (element, theta) in thetas.into_iter().enumerate() {
        let Some(theta) = theta else { continue };
        let average = averages[element];
        let average_array = average.as_array();
        for nodal in state.element_mut(element) {
            let values = nodal.as_array();
            let mut limited = [0.0; 5];
            for equation in 0..5 {
                limited[equation] = average_array[equation]
                    + theta.clamp(0.0, 1.0) * (values[equation] - average_array[equation]);
            }
            *nodal = ConservativeState::from_array(limited);
        }
    }
}

fn apply_zhang_shu_positivity(
    state: &mut CompressibleDgState,
    gas: IdealGas,
    density_floor: f64,
    pressure_floor: f64,
) -> Result<(), CompressibleDgError> {
    // Per-element scaling factors in parallel; application in order.
    let count = state.element_offsets.len() - 1;
    let compute_theta = |element: usize| -> Result<f64, CompressibleDgError> {
        let average = state.cell_average(element);
        if !average.is_physical(gas, density_floor, pressure_floor) {
            return Err(CompressibleDgError::NonPhysicalCellAverage);
        }
        let average_array = average.as_array();
        let mut theta = 1.0_f64;
        for nodal in state.element(element) {
            if nodal.density < density_floor {
                theta = theta
                    .min((average.density - density_floor) / (average.density - nodal.density));
            }
        }
        for nodal in state.element(element) {
            let nodal_array = nodal.as_array();
            let candidate = |factor: f64| {
                let mut values = [0.0; 5];
                for equation in 0..5 {
                    values[equation] = average_array[equation]
                        + factor * (nodal_array[equation] - average_array[equation]);
                }
                ConservativeState::from_array(values)
            };
            if candidate(theta).pressure(gas) < pressure_floor {
                let mut lower = 0.0;
                let mut upper = theta;
                for _ in 0..60 {
                    let midpoint = 0.5 * (lower + upper);
                    if candidate(midpoint).pressure(gas) >= pressure_floor {
                        lower = midpoint;
                    } else {
                        upper = midpoint;
                    }
                }
                theta = theta.min(lower);
            }
        }
        Ok(theta)
    };
    let thetas: Vec<Result<f64, CompressibleDgError>> =
        if ParallelPolicy::from_env().decide(count, 8) {
            (0..count).into_par_iter().map(&compute_theta).collect()
        } else {
            (0..count).map(&compute_theta).collect()
        };
    for (element, theta) in thetas.into_iter().enumerate() {
        let theta = theta?;
        let average = state.cell_average(element);
        let average_array = average.as_array();
        for nodal in state.element_mut(element) {
            let values = nodal.as_array();
            let mut limited = [0.0; 5];
            for equation in 0..5 {
                limited[equation] = average_array[equation]
                    + theta.clamp(0.0, 1.0) * (values[equation] - average_array[equation]);
            }
            *nodal = ConservativeState::from_array(limited);
        }
    }
    Ok(())
}

fn characteristic_variables(state: ConservativeState, normal: [f64; 3], gas: IdealGas) -> [f64; 5] {
    let velocity = state.velocity();
    let pressure = state.pressure(gas);
    let sound = gas.speed_of_sound(pressure, state.density);
    let normal_velocity = dot3(velocity, normal);
    let tangent1 = normalized_tangent(normal);
    let tangent2 = cross3(normal, tangent1);
    [
        pressure - state.density * sound * normal_velocity,
        state.density - pressure / (sound * sound),
        pressure + state.density * sound * normal_velocity,
        dot3(velocity, tangent1),
        dot3(velocity, tangent2),
    ]
}

fn normalized_tangent(normal: [f64; 3]) -> [f64; 3] {
    let seed = if normal[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let tangent = cross3(normal, seed);
    scale3(tangent, 1.0 / norm3(tangent))
}

fn validate_simplex_mesh(mesh: &Mesh) -> Result<(), CompressibleDgError> {
    let expected = match mesh.dimension {
        2 => ElementType::Triangle,
        3 => ElementType::Tetrahedron,
        _ => return Err(CompressibleDgError::UnsupportedMesh),
    };
    if mesh.elements.is_empty()
        || mesh
            .elements
            .iter()
            .any(|element| element.element_type != expected)
    {
        return Err(CompressibleDgError::UnsupportedMesh);
    }
    Ok(())
}

fn average_states(states: &[ConservativeState]) -> ConservativeState {
    let mut sum = [0.0; 5];
    for state in states {
        for (target, value) in sum.iter_mut().zip(state.as_array()) {
            *target += value;
        }
    }
    for value in &mut sum {
        *value /= states.len() as f64;
    }
    ConservativeState::from_array(sum)
}

fn combine_states(
    left: ConservativeState,
    left_scale: f64,
    right: ConservativeState,
    right_scale: f64,
) -> ConservativeState {
    let left = left.as_array();
    let right = right.as_array();
    let mut result = [0.0; 5];
    for equation in 0..5 {
        result[equation] = left_scale * left[equation] + right_scale * right[equation];
    }
    ConservativeState::from_array(result)
}

fn interpolate_with_basis(states: &[ConservativeState], basis: &[f64]) -> ConservativeState {
    let mut result = [0.0; 5];
    for (state, shape) in states.iter().zip(basis) {
        for (target, value) in result.iter_mut().zip(state.as_array()) {
            *target += shape * value;
        }
    }
    ConservativeState::from_array(result)
}

fn minimum_thermodynamics(state: &CompressibleDgState, gas: IdealGas) -> (f64, f64) {
    state.values.iter().fold(
        (f64::INFINITY, f64::INFINITY),
        |(minimum_density, minimum_pressure), value| {
            (
                minimum_density.min(value.density),
                minimum_pressure.min(value.pressure(gas)),
            )
        },
    )
}

/// Minimum altitude of a simplex element: `2A / longest edge` in 2D,
/// `3V / largest face area` in 3D. The stable time step must use this,
/// not the maximum-edge diameter: on sliver triangles (which graded
/// body-fitted meshes produce near curved surfaces) the diameter
/// overestimates the stability length by an order of magnitude, and the
/// stepper then pays a fixed number of positivity retries on every step
/// until halving reaches the altitude-based step.
fn element_min_altitude(mesh: &Mesh, element: usize) -> f64 {
    let volume = cell_volume(mesh, element);
    let cell = &mesh.elements[element];
    let vertices = cell.vertices();
    if mesh.dimension == 2 {
        let mut longest = 0.0f64;
        for i in 0..vertices.len() {
            let a = point3(mesh, vertices[i]);
            let b = point3(mesh, vertices[(i + 1) % vertices.len()]);
            longest = longest.max(norm3(sub3(b, a)));
        }
        if longest > 0.0 {
            2.0 * volume / longest
        } else {
            element_diameter(mesh, element)
        }
    } else {
        // Largest triangular face area of the tetrahedron.
        let mut largest = 0.0f64;
        for face in 0..vertices.len() {
            let others: Vec<usize> = (0..vertices.len()).filter(|&i| i != face).collect();
            let a = point3(mesh, vertices[others[0]]);
            let b = point3(mesh, vertices[others[1]]);
            let c = point3(mesh, vertices[others[2]]);
            let cross = cross3(sub3(b, a), sub3(c, a));
            largest = largest.max(0.5 * norm3(cross));
        }
        if largest > 0.0 {
            3.0 * volume / largest
        } else {
            element_diameter(mesh, element)
        }
    }
}

fn cell_volume(mesh: &Mesh, element: usize) -> f64 {
    let rule = QuadratureRule::new(mesh.elements[element].element_type, 1);
    let space = DGSpace::new(PolynomialDegree::P1);
    rule.iter()
        .map(|point| {
            point.weight
                * space
                    .tabulate(
                        mesh,
                        element,
                        point.coords[0],
                        point.coords[1],
                        point.coords[2],
                    )
                    .det_j
        })
        .sum()
}

fn element_centroid(mesh: &Mesh, element: usize) -> [f64; 3] {
    points_centroid(mesh, mesh.elements[element].vertices())
}

fn points_centroid(mesh: &Mesh, nodes: &[usize]) -> [f64; 3] {
    let mut result = [0.0; 3];
    for &node in nodes {
        result[0] += mesh.nodes[node].x;
        result[1] += mesh.nodes[node].y;
        result[2] += mesh.nodes[node].z;
    }
    scale3(result, 1.0 / nodes.len() as f64)
}

fn barycentric_coordinates(mesh: &Mesh, element: usize, point: [f64; 3]) -> Vec<f64> {
    let vertices = mesh.elements[element].vertices();
    if mesh.dimension == 2 {
        let a = point3(mesh, vertices[0]);
        let b = point3(mesh, vertices[1]);
        let c = point3(mesh, vertices[2]);
        let denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
        let l0 =
            ((b[1] - c[1]) * (point[0] - c[0]) + (c[0] - b[0]) * (point[1] - c[1])) / denominator;
        let l1 =
            ((c[1] - a[1]) * (point[0] - c[0]) + (a[0] - c[0]) * (point[1] - c[1])) / denominator;
        vec![l0, l1, 1.0 - l0 - l1]
    } else {
        let a = point3(mesh, vertices[0]);
        let matrix = [
            sub3(point3(mesh, vertices[1]), a),
            sub3(point3(mesh, vertices[2]), a),
            sub3(point3(mesh, vertices[3]), a),
        ];
        let rhs = sub3(point, a);
        let determinant = dot3(matrix[0], cross3(matrix[1], matrix[2]));
        let l1 = dot3(rhs, cross3(matrix[1], matrix[2])) / determinant;
        let l2 = dot3(matrix[0], cross3(rhs, matrix[2])) / determinant;
        let l3 = dot3(matrix[0], cross3(matrix[1], rhs)) / determinant;
        vec![1.0 - l1 - l2 - l3, l1, l2, l3]
    }
}

fn facet_quadrature(
    mesh: &Mesh,
    nodes: &[usize],
    degree: PolynomialDegree,
) -> Vec<([f64; 3], f64)> {
    let area = facet_measure(mesh, nodes);
    if nodes.len() == 2 {
        let a = point3(mesh, nodes[0]);
        let b = point3(mesh, nodes[1]);
        if degree == PolynomialDegree::P2 {
            // 3-point Gauss--Legendre (degree 5) for quadratic traces.
            let offset = (3.0_f64 / 5.0).sqrt();
            return [
                (0.5 * (1.0 - offset), 5.0 / 18.0),
                (0.5, 8.0 / 18.0),
                (0.5 * (1.0 + offset), 5.0 / 18.0),
            ]
            .into_iter()
            .map(|(s, w)| (add3(scale3(a, 1.0 - s), scale3(b, s)), w * area))
            .collect();
        }
        let offset = 1.0 / 3.0_f64.sqrt();
        [0.5 * (1.0 - offset), 0.5 * (1.0 + offset)]
            .into_iter()
            .map(|s| (add3(scale3(a, 1.0 - s), scale3(b, s)), 0.5 * area))
            .collect()
    } else {
        const RULE: [([f64; 3], f64); 7] = [
            ([1.0 / 3.0; 3], 0.225),
            (
                [
                    0.059_715_871_789_770,
                    0.470_142_064_105_115,
                    0.470_142_064_105_115,
                ],
                0.132_394_152_788_506,
            ),
            (
                [
                    0.470_142_064_105_115,
                    0.059_715_871_789_770,
                    0.470_142_064_105_115,
                ],
                0.132_394_152_788_506,
            ),
            (
                [
                    0.470_142_064_105_115,
                    0.470_142_064_105_115,
                    0.059_715_871_789_770,
                ],
                0.132_394_152_788_506,
            ),
            (
                [
                    0.797_426_985_353_087,
                    0.101_286_507_323_456,
                    0.101_286_507_323_456,
                ],
                0.125_939_180_544_827,
            ),
            (
                [
                    0.101_286_507_323_456,
                    0.797_426_985_353_087,
                    0.101_286_507_323_456,
                ],
                0.125_939_180_544_827,
            ),
            (
                [
                    0.101_286_507_323_456,
                    0.101_286_507_323_456,
                    0.797_426_985_353_087,
                ],
                0.125_939_180_544_827,
            ),
        ];
        let points = [
            point3(mesh, nodes[0]),
            point3(mesh, nodes[1]),
            point3(mesh, nodes[2]),
        ];
        RULE.into_iter()
            .map(|(basis, weight)| {
                let point = add3(
                    add3(scale3(points[0], basis[0]), scale3(points[1], basis[1])),
                    scale3(points[2], basis[2]),
                );
                (point, weight * area)
            })
            .collect()
    }
}

fn facet_measure(mesh: &Mesh, nodes: &[usize]) -> f64 {
    if nodes.len() == 2 {
        norm3(sub3(point3(mesh, nodes[1]), point3(mesh, nodes[0])))
    } else {
        0.5 * norm3(cross3(
            sub3(point3(mesh, nodes[1]), point3(mesh, nodes[0])),
            sub3(point3(mesh, nodes[2]), point3(mesh, nodes[0])),
        ))
    }
}

fn facet_normal(mesh: &Mesh, nodes: &[usize]) -> [f64; 3] {
    let raw = if nodes.len() == 2 {
        let edge = sub3(point3(mesh, nodes[1]), point3(mesh, nodes[0]));
        [edge[1], -edge[0], 0.0]
    } else {
        cross3(
            sub3(point3(mesh, nodes[1]), point3(mesh, nodes[0])),
            sub3(point3(mesh, nodes[2]), point3(mesh, nodes[0])),
        )
    };
    scale3(raw, 1.0 / norm3(raw))
}

fn point3(mesh: &Mesh, node: usize) -> [f64; 3] {
    [mesh.nodes[node].x, mesh.nodes[node].y, mesh.nodes[node].z]
}

fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale3(a: [f64; 3], scale: f64) -> [f64; 3] {
    [scale * a[0], scale * a[1], scale * a[2]]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{unit_cube_tetrahedra, unit_square_triangles};

    fn uniform_state() -> ConservativeState {
        ConservativeState::from_primitive(1.2, [30.0, 5.0, 0.0], 101_325.0, IdealGas::AIR)
    }

    fn config_with_far_field(mesh: &Mesh, state: ConservativeState) -> CompressibleDgConfig {
        let mut config = CompressibleDgConfig::default();
        config.viscosity = ViscosityModel::Constant(1.8e-5);
        for face in &mesh.boundaries {
            config
                .boundaries
                .insert(face.marker, CompressibleBoundaryCondition::FarField(state));
        }
        config
    }

    #[test]
    fn uniform_flow_residual_vanishes_on_triangles() {
        let mesh = unit_square_triangles(2);
        let reference = uniform_state();
        let state = CompressibleDgState::from_function(&mesh, |_| reference).unwrap();
        let operator =
            CompressibleDgOperator::new(&mesh, config_with_far_field(&mesh, reference)).unwrap();
        let residual = operator.spatial_operator(&state).unwrap();
        let maximum = residual
            .iter()
            .flat_map(|value| value.as_array())
            .map(f64::abs)
            .fold(0.0, f64::max);
        assert!(maximum < 1.0e-5, "maximum residual={maximum:e}");
    }

    #[test]
    fn uniform_flow_residual_vanishes_on_tetrahedra() {
        let mesh = unit_cube_tetrahedra(1);
        let reference = uniform_state();
        let state = CompressibleDgState::from_function(&mesh, |_| reference).unwrap();
        let operator =
            CompressibleDgOperator::new(&mesh, config_with_far_field(&mesh, reference)).unwrap();
        let residual = operator.spatial_operator(&state).unwrap();
        let maximum = residual
            .iter()
            .flat_map(|value| value.as_array())
            .map(f64::abs)
            .fold(0.0, f64::max);
        assert!(maximum < 1.0e-5, "maximum residual={maximum:e}");
    }

    fn config_p2_far_field(mesh: &Mesh, state: ConservativeState) -> CompressibleDgConfig {
        let mut config = config_with_far_field(mesh, state);
        config.degree = PolynomialDegree::P2;
        config
    }

    #[test]
    fn p2_uniform_flow_residual_vanishes_on_triangles_and_tetrahedra() {
        let reference = uniform_state();
        let tri = unit_square_triangles(2);
        let state =
            CompressibleDgState::from_function_degree(&tri, PolynomialDegree::P2, |_| reference)
                .unwrap();
        assert_eq!(state.element(0).len(), 6);
        let operator =
            CompressibleDgOperator::new(&tri, config_p2_far_field(&tri, reference)).unwrap();
        let residual = operator.spatial_operator(&state).unwrap();
        let maximum = residual
            .iter()
            .flat_map(|value| value.as_array())
            .map(f64::abs)
            .fold(0.0, f64::max);
        assert!(maximum < 1.0e-5, "P2 triangle residual={maximum:e}");

        let tet = unit_cube_tetrahedra(1);
        let state =
            CompressibleDgState::from_function_degree(&tet, PolynomialDegree::P2, |_| reference)
                .unwrap();
        assert_eq!(state.element(0).len(), 10);
        let operator =
            CompressibleDgOperator::new(&tet, config_p2_far_field(&tet, reference)).unwrap();
        let residual = operator.spatial_operator(&state).unwrap();
        let maximum = residual
            .iter()
            .flat_map(|value| value.as_array())
            .map(f64::abs)
            .fold(0.0, f64::max);
        assert!(maximum < 1.0e-5, "P2 tetrahedron residual={maximum:e}");
    }

    #[test]
    fn p2_cell_average_matches_linear_field_at_centroid() {
        // Density is linear in space, so its cell average equals the value
        // at the centroid. This pins the P2 average weights (zero/negative
        // at vertices) and the edge-midpoint node layout.
        let field = |point: [f64; 3]| ConservativeState {
            density: 1.0 + 0.3 * point[0] - 0.2 * point[1] + 0.1 * point[2],
            momentum: [0.0; 3],
            total_energy: 2.5,
        };
        let tri = unit_square_triangles(2);
        let state =
            CompressibleDgState::from_function_degree(&tri, PolynomialDegree::P2, field).unwrap();
        for element in 0..tri.num_elements() {
            let centroid = element_centroid(&tri, element);
            let expected = field(centroid).density;
            let average = state.cell_average(element);
            assert!(
                (average.density - expected).abs() < 1.0e-12,
                "triangle {element}: average {} vs centroid {expected}",
                average.density
            );
        }
        let tet = unit_cube_tetrahedra(1);
        let state =
            CompressibleDgState::from_function_degree(&tet, PolynomialDegree::P2, field).unwrap();
        for element in 0..tet.num_elements() {
            let centroid = element_centroid(&tet, element);
            let expected = field(centroid).density;
            let average = state.cell_average(element);
            assert!(
                (average.density - expected).abs() < 1.0e-12,
                "tetrahedron {element}: average {} vs centroid {expected}",
                average.density
            );
        }
        // Interpolation at a non-node point pins the edge-node ordering
        // shared by the state layout and the tabulated P2 basis: a permuted
        // edge order would still pass the average checks above.
        let space = DGSpace::new(PolynomialDegree::P2);
        let tri_state =
            CompressibleDgState::from_function_degree(&tri, PolynomialDegree::P2, field).unwrap();
        for (mesh, state) in [(&tri, &tri_state), (&tet, &state)] {
            let tab = space.tabulate(mesh, 0, 0.2, 0.3, 0.1);
            let interpolated = interpolate_with_basis(state.element(0), &tab.values);
            let expected = field(tab.physical_coords).density;
            assert!(
                (interpolated.density - expected).abs() < 1.0e-12,
                "P2 interpolation {} vs exact {expected}",
                interpolated.density
            );
        }
    }

    #[test]
    fn p2_ssprk3_advances_physical_uniform_state() {
        let mesh = unit_square_triangles(1);
        let reference = uniform_state();
        let mut state =
            CompressibleDgState::from_function_degree(&mesh, PolynomialDegree::P2, |_| reference)
                .unwrap();
        let solver =
            CompressibleDgSolver::new(&mesh, config_p2_far_field(&mesh, reference)).unwrap();
        let report = solver.step(&mut state, 1.0e-5).unwrap();
        assert!(report.minimum_density > 0.0);
        assert!(report.minimum_pressure > 0.0);
    }

    #[test]
    fn troubled_cell_sensor_leaves_smooth_field_unlimited() {
        // Linear density ramp: neighbour cell-average jumps are ~1%, far
        // below the 0.5 threshold, so no element may be touched.
        let mesh = unit_square_triangles(4);
        let gas = IdealGas {
            gamma: 1.4,
            gas_constant: 1.0,
        };
        let mut state =
            CompressibleDgState::from_function_degree(&mesh, PolynomialDegree::P2, |point| {
                ConservativeState {
                    density: 1.0 + 0.1 * point[0],
                    momentum: [0.0; 3],
                    total_energy: 2.5,
                }
            })
            .unwrap();
        let before = state.clone();
        apply_characteristic_vertex_limiter(
            &mesh,
            &mut state,
            gas,
            0.5,
            0.5,
            f64::INFINITY,
            &build_interior_facets(&mesh),
        );
        assert_eq!(state, before, "smooth field must not be limited");
        // With the sensor disabled (infinite threshold) the same field IS
        // flattened somewhere: this is the legacy behaviour being gated.
        let mut legacy = before.clone();
        apply_characteristic_vertex_limiter(
            &mesh,
            &mut legacy,
            gas,
            0.5,
            f64::INFINITY,
            f64::INFINITY,
            &build_interior_facets(&mesh),
        );
        assert_ne!(legacy, before, "legacy limiter should alter the ramp");
    }

    #[test]
    fn troubled_cell_sensor_limits_only_the_shocked_element() {
        // Density ramp on the left, jump down at x = 0.5. The element
        // straddling the jump is troubled; a far-upstream element is not.
        let mesh = unit_square_triangles(4);
        let gas = IdealGas {
            gamma: 1.4,
            gas_constant: 1.0,
        };
        let field = |point: [f64; 3]| ConservativeState {
            density: 1.0 + 0.4 * point[0] - if point[0] > 0.5 { 0.8 } else { 0.0 },
            momentum: [0.0; 3],
            total_energy: 2.5,
        };
        let mut state =
            CompressibleDgState::from_function_degree(&mesh, PolynomialDegree::P2, field).unwrap();
        let before = state.clone();
        apply_characteristic_vertex_limiter(
            &mesh,
            &mut state,
            gas,
            0.05,
            0.1,
            f64::INFINITY,
            &build_interior_facets(&mesh),
        );
        // Find an element whose centroid is far upstream (smooth) and one
        // adjacent to the jump.
        let mut smooth_changed = false;
        let mut shocked_changed = false;
        for element in 0..mesh.num_elements() {
            let centroid = element_centroid(&mesh, element);
            let changed = state.element(element) != before.element(element);
            if centroid[0] < 0.2 {
                smooth_changed |= changed;
            }
            if (centroid[0] - 0.45).abs() < 0.08 {
                shocked_changed |= changed;
            }
        }
        assert!(
            !smooth_changed,
            "smooth upstream element must stay unlimited"
        );
        assert!(shocked_changed, "element at the jump must be limited");
    }

    #[test]
    fn persson_indicator_separates_smooth_from_discontinuous() {
        let mesh = unit_square_triangles(4);
        let matrices = persson_matrices(mesh.elements[0].element_type, PolynomialDegree::P2)
            .expect("P2 Persson matrices");
        let gas = IdealGas {
            gamma: 1.4,
            gas_constant: 1.0,
        };
        let _ = gas;
        let indicator_of = |field: &dyn Fn([f64; 3]) -> f64| -> f64 {
            let state =
                CompressibleDgState::from_function_degree(&mesh, PolynomialDegree::P2, |point| {
                    ConservativeState {
                        density: field(point),
                        momentum: [0.0; 3],
                        total_energy: 2.5,
                    }
                })
                .unwrap();
            // Worst (largest) indicator across elements.
            (0..mesh.num_elements())
                .map(|element| {
                    let density: Vec<f64> = state
                        .element(element)
                        .iter()
                        .map(|nodal| nodal.density)
                        .collect();
                    persson_indicator(&density, &matrices)
                })
                .fold(f64::NEG_INFINITY, f64::max)
        };
        let linear = indicator_of(&|point| 1.0 + 0.1 * point[0] - 0.05 * point[1]);
        let quadratic = indicator_of(&|point| 1.0 + 0.3 * point[0] * point[0]);
        let step = indicator_of(&|point| if point[0] < 0.5 { 1.0 } else { 0.3 });
        assert!(
            linear < -8.0,
            "linear field indicator {linear} should be tiny"
        );
        assert!(
            step > -1.5,
            "step field indicator {step} should approach 0 (all energy in top modes)"
        );
        assert!(
            quadratic > linear && quadratic < step,
            "quadratic indicator {quadratic} should sit between linear {linear} and step {step}"
        );
    }

    #[test]
    fn persson_mode_leaves_smooth_ramp_unlimited() {
        let mesh = unit_square_triangles(4);
        let gas = IdealGas {
            gamma: 1.4,
            gas_constant: 1.0,
        };
        let mut state =
            CompressibleDgState::from_function_degree(&mesh, PolynomialDegree::P2, |point| {
                ConservativeState {
                    density: 1.0 + 0.4 * point[0] * point[0],
                    momentum: [0.0; 3],
                    total_energy: 2.5,
                }
            })
            .unwrap();
        let before = state.clone();
        // PP threshold -1 flags only near-discontinuous elements; the
        // legacy jump sensor stays out of the decision in PP mode.
        apply_characteristic_vertex_limiter(
            &mesh,
            &mut state,
            gas,
            0.05,
            0.03,
            -1.0,
            &build_interior_facets(&mesh),
        );
        assert_eq!(
            state, before,
            "smooth quadratic field must not be limited in PP mode"
        );
    }

    #[test]
    fn p2_state_rejected_by_p1_operator() {
        let mesh = unit_square_triangles(1);
        let reference = uniform_state();
        let state =
            CompressibleDgState::from_function_degree(&mesh, PolynomialDegree::P2, |_| reference)
                .unwrap();
        let operator =
            CompressibleDgOperator::new(&mesh, config_with_far_field(&mesh, reference)).unwrap();
        assert!(operator.spatial_operator(&state).is_err());
    }

    #[test]
    fn positivity_scaling_preserves_cell_average() {
        let mesh = unit_square_triangles(1);
        let gas = IdealGas::AIR;
        let reference = ConservativeState::from_primitive(1.0, [0.0; 3], 1.0, gas);
        let mut state = CompressibleDgState::from_function(&mesh, |_| reference).unwrap();
        let before = state.cell_average(0);
        state.element_mut(0)[0].density = -0.1;
        // Compensate another node so the average remains physical.
        state.element_mut(0)[1].density += 1.1;
        let perturbed_average = state.cell_average(0);
        apply_zhang_shu_positivity(&mut state, gas, 1.0e-10, 1.0e-10).unwrap();
        let after = state.cell_average(0);
        assert!((after.density - perturbed_average.density).abs() < 1.0e-12);
        assert!(state.element(0).iter().all(|value| value.density > 0.0));
        assert!((before.total_energy - after.total_energy).abs() < 1.0e-12);
    }

    #[test]
    fn ssprk3_advances_physical_uniform_state() {
        let mesh = unit_square_triangles(1);
        let reference = uniform_state();
        let mut state = CompressibleDgState::from_function(&mesh, |_| reference).unwrap();
        let solver =
            CompressibleDgSolver::new(&mesh, config_with_far_field(&mesh, reference)).unwrap();
        let report = solver.step(&mut state, 1.0e-5).unwrap();
        assert!(report.minimum_density > 0.0);
        assert!(report.minimum_pressure > 0.0);
        assert_eq!(report.retries, 0);
    }

    #[test]
    fn wall_and_characteristic_boundary_states_are_physical() {
        let gas = IdealGas::AIR;
        let interior = uniform_state();
        let normal = [1.0, 0.0, 0.0];
        let slip = boundary_state(
            CompressibleBoundaryCondition::SlipWall,
            interior,
            normal,
            gas,
        )
        .unwrap();
        assert!((slip.velocity()[0] + interior.velocity()[0]).abs() < 1.0e-12);
        let far = boundary_state(
            CompressibleBoundaryCondition::FarField(interior),
            interior,
            normal,
            gas,
        )
        .unwrap();
        assert!(far.is_physical(gas, 1.0e-12, 1.0e-12));
    }
}
