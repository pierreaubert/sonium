//! Rotor centrifugal stress: 2-D plane-stress linear elasticity on
//! the motor cross-section, mirroring the welded (perfectly bonded)
//! solve of the reference's `simulation/mechanical/rotor_stress.py`.
//!
//! The load is the centrifugal body force `f = rho * omega^2 * r`
//! plus an optional per-element thermal eigenstrain `alpha * dT`.
//! The rotor is held by NOTHING — the load is self-equilibrated —
//! so the three rigid-body modes are removed by a bordered system
//! `[[K, R], [R^T, 0]]` with an orthonormal rigid-mode basis `R`,
//! never by pinning nodes (a pin invents a reaction and a stress
//! concentration exactly where an engineer would read a result).
//! The border is scaled by `sqrt(mean(diag K))` so the multiplier
//! rows are not orders below the stiffness rows, as in the
//! reference. `rigid_residual` measures the part of the load the
//! multipliers had to absorb; on a healthy full rotor it is
//! rounding noise.
//!
//! Elements are constant-strain triangles on the corner nodes of
//! the EM mesh, so stress is elementwise constant: bridge stress
//! concentrations converge more slowly than the reference's default
//! P2 field. The stiffness does not depend on speed, so one
//! factorisation serves every `omega` — that is what makes the
//! limit-speed search (SF = 1 by false position on the
//! `1/SF = a + b * omega^2` line, the reference's model) cheap.
//!
//! Scope versus the reference: no contact pairs, no interference
//! fits, no cyclic-sector ties — one welded body, orthotropic
//! cylindrical materials included (a hoop-wound sleeve).

// Rust guideline compliant 2026-02-21

use std::collections::HashMap;

use sonium_solvers::TripletMatrix;

use crate::mesh::Mesh;

/// Mechanical material of one part, plane stress.
#[derive(Debug, Clone, Copy)]
pub enum MechMaterial {
    /// Isotropic part (laminated iron in-plane, shaft, magnets).
    Isotropic {
        /// Young's modulus in Pa.
        youngs_modulus_pa: f64,
        /// Poisson ratio.
        poisson_ratio: f64,
        /// Density in kg/m^3.
        density_kg_per_m3: f64,
        /// Thermal expansion coefficient in 1/K.
        thermal_expansion_per_k: f64,
        /// Yield (or allowable) stress in Pa for the safety factor.
        yield_pa: f64,
    },
    /// Orthotropic part in cylindrical axes — axis 1 tangential
    /// (hoop fibre direction), axis 2 radial.
    OrthotropicCylindrical {
        /// Tangential (fibre) Young's modulus in Pa.
        e_tangential_pa: f64,
        /// Radial (transverse) Young's modulus in Pa.
        e_radial_pa: f64,
        /// Poisson ratio nu_12 (tangential load, radial strain).
        poisson_12: f64,
        /// In-plane shear modulus in Pa.
        shear_modulus_pa: f64,
        /// Density in kg/m^3.
        density_kg_per_m3: f64,
        /// Tangential thermal expansion in 1/K.
        alpha_tangential_per_k: f64,
        /// Radial thermal expansion in 1/K.
        alpha_radial_per_k: f64,
        /// Yield (or allowable) stress in Pa for the safety factor.
        yield_pa: f64,
    },
}

impl MechMaterial {
    /// Density in kg/m^3.
    pub fn density_kg_per_m3(&self) -> f64 {
        match self {
            Self::Isotropic {
                density_kg_per_m3, ..
            }
            | Self::OrthotropicCylindrical {
                density_kg_per_m3, ..
            } => *density_kg_per_m3,
        }
    }

    /// Yield stress in Pa.
    pub fn yield_pa(&self) -> f64 {
        match self {
            Self::Isotropic { yield_pa, .. } | Self::OrthotropicCylindrical { yield_pa, .. } => {
                *yield_pa
            }
        }
    }

    /// Plane-stress stiffness in global axes for an element whose
    /// centroid sits at polar angle `phi_rad`.
    fn stiffness(&self, phi_rad: f64) -> [[f64; 3]; 3] {
        match *self {
            Self::Isotropic {
                youngs_modulus_pa: e,
                poisson_ratio: nu,
                ..
            } => {
                let f = e / (1.0 - nu * nu);
                [
                    [f, f * nu, 0.0],
                    [f * nu, f, 0.0],
                    [0.0, 0.0, 0.5 * e / (1.0 + nu)],
                ]
            }
            Self::OrthotropicCylindrical {
                e_tangential_pa: e1,
                e_radial_pa: e2,
                poisson_12: nu12,
                shear_modulus_pa: g12,
                ..
            } => {
                let nu21 = nu12 * e2 / e1;
                let d = 1.0 - nu12 * nu21;
                let c_mat = [
                    [e1 / d, nu12 * e2 / d, 0.0],
                    [nu12 * e2 / d, e2 / d, 0.0],
                    [0.0, 0.0, g12],
                ];
                // Material axis 1 is tangential: its angle from +x is
                // phi + pi/2. Classical lamination rotation.
                rotate_stiffness(c_mat, phi_rad + std::f64::consts::FRAC_PI_2)
            }
        }
    }

    /// Thermal eigenstrain (engineering Voigt, global axes) for a
    /// temperature rise `delta_t_k` at polar angle `phi_rad`.
    fn eigenstrain(&self, phi_rad: f64, delta_t_k: f64) -> [f64; 3] {
        match *self {
            Self::Isotropic {
                thermal_expansion_per_k: alpha,
                ..
            } => [alpha * delta_t_k, alpha * delta_t_k, 0.0],
            Self::OrthotropicCylindrical {
                alpha_tangential_per_k: alpha1,
                alpha_radial_per_k: alpha2,
                ..
            } => {
                // Diagonal strain tensor in (tangential, radial)
                // axes, rotated to (x, y); axis 1 at phi + pi/2.
                let theta = phi_rad + std::f64::consts::FRAC_PI_2;
                let (c, s) = (theta.cos(), theta.sin());
                let (e1, e2) = (alpha1 * delta_t_k, alpha2 * delta_t_k);
                [
                    c * c * e1 + s * s * e2,
                    s * s * e1 + c * c * e2,
                    2.0 * c * s * (e1 - e2),
                ]
            }
        }
    }

    fn validate(&self) -> Result<(), MechError> {
        let finite_positive = |value: f64| value.is_finite() && value > 0.0;
        let ok = match *self {
            Self::Isotropic {
                youngs_modulus_pa,
                poisson_ratio,
                density_kg_per_m3,
                thermal_expansion_per_k,
                yield_pa,
            } => {
                finite_positive(youngs_modulus_pa)
                    && finite_positive(density_kg_per_m3)
                    && finite_positive(yield_pa)
                    && thermal_expansion_per_k.is_finite()
                    && poisson_ratio.is_finite()
                    && (-1.0..0.5).contains(&poisson_ratio)
            }
            Self::OrthotropicCylindrical {
                e_tangential_pa,
                e_radial_pa,
                poisson_12,
                shear_modulus_pa,
                density_kg_per_m3,
                alpha_tangential_per_k,
                alpha_radial_per_k,
                yield_pa,
            } => {
                finite_positive(e_tangential_pa)
                    && finite_positive(e_radial_pa)
                    && finite_positive(shear_modulus_pa)
                    && finite_positive(density_kg_per_m3)
                    && finite_positive(yield_pa)
                    && poisson_12.is_finite()
                    && alpha_tangential_per_k.is_finite()
                    && alpha_radial_per_k.is_finite()
                    && (poisson_12 * poisson_12 * e_radial_pa / e_tangential_pa) < 1.0
            }
        };
        if ok {
            Ok(())
        } else {
            Err(MechError::InvalidMaterial)
        }
    }
}

/// Rotates a plane-stress stiffness from material axes to global
/// axes (classical lamination theory, engineering shear).
fn rotate_stiffness(c_mat: [[f64; 3]; 3], theta_rad: f64) -> [[f64; 3]; 3] {
    let (c, s) = (theta_rad.cos(), theta_rad.sin());
    let t = [
        [c * c, s * s, 2.0 * c * s],
        [s * s, c * c, -2.0 * c * s],
        [-c * s, c * s, c * c - s * s],
    ];
    // T is orthogonal-like with inverse = T(-theta) transposed
    // conventions aside: build Cbar = T^-1 C T^-T with T^-1 = T(-θ)
    // formed by the same formula at -theta.
    let (cm, sm) = (c, -s);
    let t_inv = [
        [cm * cm, sm * sm, 2.0 * cm * sm],
        [sm * sm, cm * cm, -2.0 * cm * sm],
        [-cm * sm, cm * sm, cm * cm - sm * sm],
    ];
    let _ = t;
    let mut tmp = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            tmp[i][j] = (0..3).map(|k| t_inv[i][k] * c_mat[k][j]).sum();
        }
    }
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = (0..3).map(|k| tmp[i][k] * t_inv[j][k]).sum();
        }
    }
    out
}

/// A rotor-stress problem on a triangular cross-section mesh.
#[derive(Debug, Clone)]
pub struct RotorStressProblem {
    /// Material by cell marker.
    pub materials: std::collections::BTreeMap<i32, MechMaterial>,
    /// Cell markers to drop (air, coils, stator on a full-machine mesh).
    pub drop_markers: Vec<i32>,
    /// Temperature rise per element in K (thermal eigenstrain);
    /// empty = isothermal.
    pub delta_t_k: Vec<f64>,
}

/// Result of one rotor-stress solve.
#[derive(Debug, Clone)]
pub struct RotorStressResult {
    /// Nodal displacement at every input node in metres (`NaN` for
    /// nodes outside the kept domain).
    pub displacement_m: Vec<[f64; 2]>,
    /// Element stress `[sigma_x, sigma_y, tau_xy]` in Pa, per kept
    /// element slot (same order as `kept_elements`).
    pub stress_pa: Vec<[f64; 3]>,
    /// Element von Mises stress in Pa, per kept element slot.
    pub von_mises_pa: Vec<f64>,
    /// Input-mesh element indices of the kept slots.
    pub kept_elements: Vec<usize>,
    /// Worst safety factor `yield / von Mises` per marker.
    pub safety_factor_by_marker: Vec<(i32, f64)>,
    /// Overall minimum safety factor.
    pub safety_factor_min: f64,
    /// Largest radial displacement component at any kept node, metres.
    pub max_radial_growth_m: f64,
    /// Load share absorbed by the rigid-mode multipliers (rounding
    /// noise on a self-equilibrated rotor).
    pub rigid_residual: f64,
}

/// One triangle's cached geometry and constitutive data.
struct MechTriangle {
    nodes: [usize; 3],
    grads: [[f64; 2]; 3],
    area_m2: f64,
    stiffness: [[f64; 3]; 3],
    eigenstrain: [f64; 3],
    density_kg_per_m3: f64,
}

/// Factorised rotor-stress solver: build once, solve at any speed.
pub struct RotorStressSolver {
    triangles: Vec<MechTriangle>,
    kept_elements: Vec<usize>,
    markers: Vec<i32>,
    yields_pa: Vec<f64>,
    /// Corner-node renumbering: input node -> slot.
    remap: HashMap<usize, usize>,
    node_positions: Vec<[f64; 2]>,
    num_input_nodes: usize,
    factor: sonium_solvers::SparseFactorization<f64>,
    /// Centrifugal load at omega = 1 rad/s.
    f_rot_unit: Vec<f64>,
    /// Eigenstrain load (speed independent).
    f_eigen: Vec<f64>,
}

/// Degree-2 3-point rule (barycentric permutations of 2/3, 1/6, 1/6).
const QUAD_BARY: [[f64; 3]; 3] = [
    [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
    [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
    [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
];

impl RotorStressSolver {
    /// Assembles and factorises the bordered stiffness.
    ///
    /// # Errors
    ///
    /// Returns [`MechError`] for invalid materials, a kept element
    /// without a material, a non-triangular or degenerate element,
    /// an empty domain, or a failed factorisation.
    pub fn new(mesh: &Mesh, problem: &RotorStressProblem) -> Result<Self, MechError> {
        for material in problem.materials.values() {
            material.validate()?;
        }
        let num_elements = mesh.num_elements();
        if !problem.delta_t_k.is_empty() && problem.delta_t_k.len() != num_elements {
            return Err(MechError::ElementDataMismatch);
        }
        let mut triangles: Vec<MechTriangle> = Vec::new();
        let mut kept_elements: Vec<usize> = Vec::new();
        let mut markers: Vec<i32> = Vec::new();
        let mut yields_pa: Vec<f64> = Vec::new();
        let mut remap: HashMap<usize, usize> = HashMap::new();
        for (element_index, element) in mesh.elements.iter().enumerate() {
            if problem.drop_markers.contains(&element.marker) {
                continue;
            }
            let material = problem
                .materials
                .get(&element.marker)
                .ok_or(MechError::MissingMaterial { element_index })?;
            let vertices = element.vertices();
            if vertices.len() != 3 {
                return Err(MechError::NonTriangularElement { element_index });
            }
            let corners = [vertices[0], vertices[1], vertices[2]];
            let p: [[f64; 2]; 3] = corners.map(|node| {
                let point = mesh.nodes[node];
                [point.x, point.y]
            });
            let det = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1])
                - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]);
            if det.abs() <= 1.0e-300 {
                return Err(MechError::DegenerateElement { element_index });
            }
            let grads = [
                [(p[1][1] - p[2][1]) / det, (p[2][0] - p[1][0]) / det],
                [(p[2][1] - p[0][1]) / det, (p[0][0] - p[2][0]) / det],
                [(p[0][1] - p[1][1]) / det, (p[1][0] - p[0][0]) / det],
            ];
            let centroid = [
                (p[0][0] + p[1][0] + p[2][0]) / 3.0,
                (p[0][1] + p[1][1] + p[2][1]) / 3.0,
            ];
            let phi = centroid[1].atan2(centroid[0]);
            let delta_t = problem.delta_t_k.get(element_index).copied().unwrap_or(0.0);
            for &node in &corners {
                let next = remap.len();
                remap.entry(node).or_insert(next);
            }
            triangles.push(MechTriangle {
                nodes: corners,
                grads,
                area_m2: 0.5 * det.abs(),
                stiffness: material.stiffness(phi),
                eigenstrain: material.eigenstrain(phi, delta_t),
                density_kg_per_m3: material.density_kg_per_m3(),
            });
            kept_elements.push(element_index);
            markers.push(element.marker);
            yields_pa.push(material.yield_pa());
        }
        if triangles.is_empty() {
            return Err(MechError::EmptyDomain);
        }
        let num_nodes = remap.len();
        let num_dofs = 2 * num_nodes;
        let num_unknowns = num_dofs + 3;
        let mut node_positions = vec![[0.0; 2]; num_nodes];
        for (&node, &slot) in &remap {
            let point = mesh.nodes[node];
            node_positions[slot] = [point.x, point.y];
        }

        // Stiffness (unit thickness; thickness cancels between K and
        // the body force, exactly as in the thermal twin) and loads.
        let mut stiffness = TripletMatrix::new(num_unknowns, num_unknowns);
        let mut f_rot_unit = vec![0.0; num_unknowns];
        let mut f_eigen = vec![0.0; num_unknowns];
        let mut diag_sum = 0.0;
        for tri in &triangles {
            let dofs: [usize; 6] = {
                let n = tri.nodes.map(|node| remap[&node]);
                [
                    2 * n[0],
                    2 * n[0] + 1,
                    2 * n[1],
                    2 * n[1] + 1,
                    2 * n[2],
                    2 * n[2] + 1,
                ]
            };
            // B (3x6): [d/dx 0; 0 d/dy; d/dy d/dx] per node.
            let b = |node: usize, row: usize, comp: usize| -> f64 {
                match (row, comp) {
                    (0, 0) => tri.grads[node][0],
                    (1, 1) => tri.grads[node][1],
                    (2, 0) => tri.grads[node][1],
                    (2, 1) => tri.grads[node][0],
                    _ => 0.0,
                }
            };
            for i in 0..3 {
                for ci in 0..2 {
                    // Eigenstrain load: int B^T C eps0 dA.
                    let bt_c_eps: f64 = (0..3)
                        .map(|row| {
                            b(i, row, ci)
                                * (0..3)
                                    .map(|k| tri.stiffness[row][k] * tri.eigenstrain[k])
                                    .sum::<f64>()
                        })
                        .sum();
                    f_eigen[dofs[2 * i + ci]] += bt_c_eps * tri.area_m2;
                    for j in 0..3 {
                        for cj in 0..2 {
                            let value: f64 = (0..3)
                                .map(|r| {
                                    b(i, r, ci)
                                        * (0..3)
                                            .map(|k| tri.stiffness[r][k] * b(j, k, cj))
                                            .sum::<f64>()
                                })
                                .sum();
                            stiffness.push(dofs[2 * i + ci], dofs[2 * j + cj], value * tri.area_m2);
                            if dofs[2 * i + ci] == dofs[2 * j + cj] {
                                diag_sum += value * tri.area_m2;
                            }
                        }
                    }
                }
            }
            // Centrifugal load at omega = 1: int rho * x * phi_i dA.
            let corners: [[f64; 2]; 3] = tri.nodes.map(|node| {
                let point = mesh.nodes[node];
                [point.x, point.y]
            });
            for bary in &QUAD_BARY {
                let position = [
                    bary[0] * corners[0][0] + bary[1] * corners[1][0] + bary[2] * corners[2][0],
                    bary[0] * corners[0][1] + bary[1] * corners[1][1] + bary[2] * corners[2][1],
                ];
                let weight = tri.area_m2 / 3.0;
                for i in 0..3 {
                    f_rot_unit[dofs[2 * i]] +=
                        tri.density_kg_per_m3 * position[0] * bary[i] * weight;
                    f_rot_unit[dofs[2 * i + 1]] +=
                        tri.density_kg_per_m3 * position[1] * bary[i] * weight;
                }
            }
        }

        // Orthonormal rigid-mode basis (Gram-Schmidt on 3 columns).
        let mut modes: Vec<Vec<f64>> = vec![
            (0..num_nodes).flat_map(|_| [1.0, 0.0]).collect(),
            (0..num_nodes).flat_map(|_| [0.0, 1.0]).collect(),
            node_positions.iter().flat_map(|p| [-p[1], p[0]]).collect(),
        ];
        for i in 0..3 {
            for j in 0..i {
                let proj: f64 = modes[i].iter().zip(&modes[j]).map(|(a, b)| a * b).sum();
                let (left, right) = modes.split_at_mut(i);
                for (value, basis) in right[0].iter_mut().zip(&left[j]) {
                    *value -= proj * basis;
                }
            }
            let norm = modes[i].iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm <= 0.0 {
                return Err(MechError::EmptyDomain);
            }
            for value in &mut modes[i] {
                *value /= norm;
            }
        }
        let border_scale = (diag_sum / num_dofs as f64).abs().sqrt();
        for (mode_index, mode) in modes.iter().enumerate() {
            for (dof, &value) in mode.iter().enumerate() {
                if value != 0.0 {
                    stiffness.push(dof, num_dofs + mode_index, border_scale * value);
                    stiffness.push(num_dofs + mode_index, dof, border_scale * value);
                }
            }
        }
        let config = sonium_solvers::SparseDirectConfig {
            matrix_class: sonium_solvers::MatrixClass::SymmetricIndefinite,
            ..sonium_solvers::SparseDirectConfig::default()
        };
        let factor = sonium_solvers::factorize_sparse(&stiffness.to_csr(), &config, None)
            .map_err(|error| MechError::LinearSolve(error.to_string()))?;
        Ok(Self {
            triangles,
            kept_elements,
            markers,
            yields_pa,
            remap,
            node_positions,
            num_input_nodes: mesh.num_nodes(),
            factor,
            f_rot_unit,
            f_eigen,
        })
    }

    /// Solves at angular speed `omega_rad_s`.
    ///
    /// # Errors
    ///
    /// Returns [`MechError`] when the back-solve fails.
    pub fn solve(&self, omega_rad_s: f64) -> Result<RotorStressResult, MechError> {
        let omega2 = omega_rad_s * omega_rad_s;
        let rhs: Vec<f64> = self
            .f_rot_unit
            .iter()
            .zip(&self.f_eigen)
            .map(|(rot, eig)| rot * omega2 + eig)
            .collect();
        let solution = self
            .factor
            .solve(&rhs)
            .map_err(|error| MechError::LinearSolve(error.to_string()))?;

        let mut displacement_m = vec![[f64::NAN; 2]; self.num_input_nodes];
        for (&node, &slot) in &self.remap {
            displacement_m[node] = [solution[2 * slot], solution[2 * slot + 1]];
        }
        let mut max_radial_growth_m = 0.0f64;
        for (slot, position) in self.node_positions.iter().enumerate() {
            let radius = position[0].hypot(position[1]);
            if radius > 0.0 {
                let radial = (solution[2 * slot] * position[0]
                    + solution[2 * slot + 1] * position[1])
                    / radius;
                max_radial_growth_m = max_radial_growth_m.max(radial);
            }
        }

        let mut stress_pa: Vec<[f64; 3]> = Vec::with_capacity(self.triangles.len());
        let mut von_mises_pa: Vec<f64> = Vec::with_capacity(self.triangles.len());
        let mut worst: HashMap<i32, f64> = HashMap::new();
        for (slot, tri) in self.triangles.iter().enumerate() {
            let n = tri.nodes.map(|node| self.remap[&node]);
            let u: [[f64; 2]; 3] = n.map(|s| [solution[2 * s], solution[2 * s + 1]]);
            let mut strain = [0.0; 3];
            for (grad, disp) in tri.grads.iter().zip(&u) {
                strain[0] += grad[0] * disp[0];
                strain[1] += grad[1] * disp[1];
                strain[2] += grad[1] * disp[0] + grad[0] * disp[1];
            }
            let mechanical = [
                strain[0] - tri.eigenstrain[0],
                strain[1] - tri.eigenstrain[1],
                strain[2] - tri.eigenstrain[2],
            ];
            let sigma: [f64; 3] =
                std::array::from_fn(|r| (0..3).map(|k| tri.stiffness[r][k] * mechanical[k]).sum());
            let von_mises = (sigma[0] * sigma[0] - sigma[0] * sigma[1]
                + sigma[1] * sigma[1]
                + 3.0 * sigma[2] * sigma[2])
                .sqrt();
            stress_pa.push(sigma);
            von_mises_pa.push(von_mises);
            let sf = if von_mises > 0.0 {
                self.yields_pa[slot] / von_mises
            } else {
                f64::INFINITY
            };
            let entry = worst.entry(self.markers[slot]).or_insert(f64::INFINITY);
            *entry = entry.min(sf);
        }
        let mut safety_factor_by_marker: Vec<(i32, f64)> = worst.into_iter().collect();
        safety_factor_by_marker.sort_by_key(|entry| entry.0);
        let safety_factor_min = safety_factor_by_marker
            .iter()
            .map(|entry| entry.1)
            .fold(f64::INFINITY, f64::min);
        Ok(RotorStressResult {
            displacement_m,
            stress_pa,
            von_mises_pa,
            kept_elements: self.kept_elements.clone(),
            safety_factor_by_marker,
            safety_factor_min,
            max_radial_growth_m,
            rigid_residual: self.rigid_residual(&rhs),
        })
    }

    /// `|R^T f|_inf / |f|_2` of the physical load.
    fn rigid_residual(&self, rhs: &[f64]) -> f64 {
        let num_dofs = 2 * self.node_positions.len();
        let modes = self.rigid_modes();
        let f_norm = rhs[..num_dofs].iter().map(|v| v * v).sum::<f64>().sqrt();
        if f_norm <= 0.0 {
            return 0.0;
        }
        modes
            .iter()
            .map(|mode| {
                mode.iter()
                    .zip(&rhs[..num_dofs])
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
                    .abs()
            })
            .fold(0.0, f64::max)
            / f_norm
    }

    fn rigid_modes(&self) -> Vec<Vec<f64>> {
        // The stored border rows were orthonormal modes; rebuild
        // them deterministically from the node positions.
        let num_nodes = self.node_positions.len();
        let mut modes: Vec<Vec<f64>> = vec![
            (0..num_nodes).flat_map(|_| [1.0, 0.0]).collect(),
            (0..num_nodes).flat_map(|_| [0.0, 1.0]).collect(),
            self.node_positions
                .iter()
                .flat_map(|p| [-p[1], p[0]])
                .collect(),
        ];
        for i in 0..3 {
            for j in 0..i {
                let proj: f64 = modes[i].iter().zip(&modes[j]).map(|(a, b)| a * b).sum();
                let (left, right) = modes.split_at_mut(i);
                for (value, basis) in right[0].iter_mut().zip(&left[j]) {
                    *value -= proj * basis;
                }
            }
            let norm = modes[i]
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt()
                .max(1.0e-300);
            for value in &mut modes[i] {
                *value /= norm;
            }
        }
        modes
    }

    /// Limit speed where the minimum safety factor reaches 1.
    ///
    /// Mirrors the reference search at showcase fidelity: the pure
    /// `omega^2` estimate from `rpm0`, one verification solve there,
    /// then false position on the `1/SF = a + b * omega^2` line
    /// through the two solves. Returns `None` when the safety factor
    /// does not fall with speed (no centrifugal limit in range).
    ///
    /// # Errors
    ///
    /// Propagates solve failures.
    pub fn limit_speed_rpm(&self, rpm0: f64) -> Result<Option<f64>, MechError> {
        let to_omega = |rpm: f64| rpm * 2.0 * std::f64::consts::PI / 60.0;
        let sf0 = self.solve(to_omega(rpm0))?.safety_factor_min;
        if !sf0.is_finite() || sf0 <= 0.0 {
            return Ok(None);
        }
        if (sf0 - 1.0).abs() < 1.0e-3 {
            return Ok(Some(rpm0));
        }
        let rpm1 = rpm0 * sf0.sqrt();
        let sf1 = self.solve(to_omega(rpm1))?.safety_factor_min;
        if !sf1.is_finite() {
            return Ok(None);
        }
        // 1/SF = a + b * rpm^2 through (rpm0, sf0), (rpm1, sf1).
        let (x0, y0) = (rpm0 * rpm0, 1.0 / sf0);
        let (x1, y1) = (rpm1 * rpm1, 1.0 / sf1);
        if (x1 - x0).abs() <= 0.0 || y1 <= y0 {
            return Ok(None);
        }
        let b = (y1 - y0) / (x1 - x0);
        let a = y0 - b * x0;
        let x = (1.0 - a) / b;
        if x > 0.0 && x.is_finite() {
            Ok(Some(x.sqrt()))
        } else {
            Ok(None)
        }
    }
}

/// Failure of a rotor-stress solve.
#[derive(Debug, thiserror::Error)]
pub enum MechError {
    /// A material field was non-finite or non-physical.
    #[error("mechanical material data must be finite and physical")]
    InvalidMaterial,
    /// Per-element arrays did not match the element count.
    #[error("per-element mechanical data does not match the mesh element count")]
    ElementDataMismatch,
    /// A kept element's marker had no material.
    #[error("element {element_index} has no mechanical material")]
    MissingMaterial {
        /// Offending element index.
        element_index: usize,
    },
    /// An element was not a triangle.
    #[error("rotor stress needs triangles; element {element_index} is not one")]
    NonTriangularElement {
        /// Offending element index.
        element_index: usize,
    },
    /// An element had zero area.
    #[error("rotor-stress element {element_index} is degenerate")]
    DegenerateElement {
        /// Offending element index.
        element_index: usize,
    },
    /// Every element was dropped.
    #[error("no elements left after dropping markers")]
    EmptyDomain,
    /// The sparse direct solve failed.
    #[error("rotor-stress linear solve failed: {0}")]
    LinearSolve(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{ElementType, Point};

    const STEEL: MechMaterial = MechMaterial::Isotropic {
        youngs_modulus_pa: 200.0e9,
        poisson_ratio: 0.3,
        density_kg_per_m3: 7_800.0,
        thermal_expansion_per_k: 12.0e-6,
        yield_pa: 400.0e6,
    };

    /// Polar fan mesh: solid disk when `r_in = 0` (centre node fan),
    /// annulus otherwise.
    fn polar_mesh(r_in: f64, r_out: f64, rings: usize, sectors: usize) -> Mesh {
        let mut mesh = Mesh::new(2);
        let solid = r_in <= 0.0;
        if solid {
            mesh.add_node(Point::new_2d(0.0, 0.0));
        }
        let first_ring = if solid { 1 } else { 0 };
        for ring in first_ring..=rings {
            let radius = r_in + (r_out - r_in) * ring as f64 / rings as f64;
            for sector in 0..sectors {
                let angle = 2.0 * std::f64::consts::PI * sector as f64 / sectors as f64;
                mesh.add_node(Point::new_2d(radius * angle.cos(), radius * angle.sin()));
            }
        }
        let offset = usize::from(solid);
        let node =
            |ring: usize, sector: usize| offset + (ring - first_ring) * sectors + sector % sectors;
        if solid {
            for sector in 0..sectors {
                mesh.add_element_with_marker(
                    ElementType::Triangle,
                    vec![0, node(1, sector + 1), node(1, sector)],
                    1,
                );
            }
        }
        for ring in first_ring.max(1)..rings {
            for sector in 0..sectors {
                let (a, b) = (node(ring, sector), node(ring, sector + 1));
                let (c, d) = (node(ring + 1, sector), node(ring + 1, sector + 1));
                mesh.add_element_with_marker(ElementType::Triangle, vec![a, d, c], 1);
                mesh.add_element_with_marker(ElementType::Triangle, vec![a, b, d], 1);
            }
        }
        mesh
    }

    fn problem(mesh: &Mesh) -> RotorStressProblem {
        let _ = mesh;
        RotorStressProblem {
            materials: [(1, STEEL)].into_iter().collect(),
            drop_markers: Vec::new(),
            delta_t_k: Vec::new(),
        }
    }

    /// Hoop stress and centroid radius of the element whose centroid
    /// is nearest radius `r` on the +x axis (sigma_theta = sigma_y
    /// there). The radius comes back so the exact solution is
    /// evaluated where the constant element stress actually sits.
    fn hoop_near(mesh: &Mesh, result: &RotorStressResult, radius: f64) -> (f64, f64) {
        let mut best: Option<(f64, f64, f64)> = None;
        for (slot, &element_index) in result.kept_elements.iter().enumerate() {
            let element = &mesh.elements[element_index];
            let centroid: [f64; 2] = {
                let mut c = [0.0; 2];
                for &node in element.vertices() {
                    c[0] += mesh.nodes[node].x / 3.0;
                    c[1] += mesh.nodes[node].y / 3.0;
                }
                c
            };
            let distance = (centroid[0] - radius).abs() + centroid[1].abs();
            if best.is_none_or(|(d, _, _)| distance < d) {
                best = Some((
                    distance,
                    centroid[0].hypot(centroid[1]),
                    result.stress_pa[slot][1],
                ));
            }
        }
        let (_, r, hoop) = best.expect("elements present");
        (r, hoop)
    }

    #[test]
    fn solid_disk_matches_lame_solution() {
        // Free solid disk: sigma_r = (3+nu)/8 rho w^2 (R^2 - r^2),
        // sigma_theta(0) = (3+nu)/8 rho w^2 R^2,
        // sigma_theta(R) = (1-nu)/4 rho w^2 R^2.
        let (radius, omega) = (0.05, 1_000.0);
        let mesh = polar_mesh(0.0, radius, 24, 48);
        let solver = RotorStressSolver::new(&mesh, &problem(&mesh)).expect("solver");
        let result = solver.solve(omega).expect("solve");
        let rho_w2_r2 = 7_800.0 * omega * omega * radius * radius;
        let (_, centre) = hoop_near(&mesh, &result, radius / 24.0);
        let exact_centre = (3.0 + 0.3) / 8.0 * rho_w2_r2;
        assert!(
            (centre - exact_centre).abs() / exact_centre < 0.03,
            "centre hoop {centre:.4e} vs exact {exact_centre:.4e}"
        );
        let (r_rim, rim) = hoop_near(&mesh, &result, radius * 23.0 / 24.0);
        let exact_rim = {
            let r = r_rim;
            (3.0 + 0.3) / 8.0 * 7_800.0 * omega * omega * radius * radius
                - (1.0 + 3.0 * 0.3) / 8.0 * 7_800.0 * omega * omega * r * r
        };
        assert!(
            (rim - exact_rim).abs() / exact_rim < 0.05,
            "rim hoop {rim:.4e} vs exact {exact_rim:.4e}"
        );
        assert!(
            result.rigid_residual < 1.0e-10,
            "centrifugal load not self-equilibrated: {}",
            result.rigid_residual
        );
    }

    #[test]
    fn annulus_bore_hoop_stress_matches_lame() {
        // Free annulus closed form (Timoshenko):
        //   sigma_t(r) = (3+nu)/8 rho w^2 [a^2+b^2 + a^2 b^2/r^2
        //                - ((1+3nu)/(3+nu)) r^2].
        // The bore-adjacent CST ring carries the full free-surface
        // hoop level, so it is compared against sigma_t(a); a
        // mid-annulus element is compared at its own centroid.
        let (a, b, omega) = (0.01, 0.03, 2_000.0);
        let mesh = polar_mesh(a, b, 48, 96);
        let solver = RotorStressSolver::new(&mesh, &problem(&mesh)).expect("solver");
        let result = solver.solve(omega).expect("solve");
        let nu = 0.3;
        let exact = |r: f64| {
            (3.0 + nu) / 8.0
                * 7_800.0
                * omega
                * omega
                * (a * a + b * b + a * a * b * b / (r * r) - (1.0 + 3.0 * nu) / (3.0 + nu) * r * r)
        };
        let (_, bore) = hoop_near(&mesh, &result, a);
        assert!(
            (bore - exact(a)).abs() / exact(a) < 0.04,
            "bore hoop {bore:.4e} vs exact {:.4e}",
            exact(a)
        );
        // CST stress converges O(h): 6.3 % at 24x48, 3.3 % here.
        let (r_mid, mid) = hoop_near(&mesh, &result, 0.5 * (a + b));
        assert!(
            (mid - exact(r_mid)).abs() / exact(r_mid) < 0.04,
            "mid hoop {mid:.4e} vs exact {:.4e} at r={r_mid:.4}",
            exact(r_mid)
        );
    }

    #[test]
    fn uniform_temperature_rise_is_stress_free_expansion() {
        // A free body at uniform dT expands without stress:
        // u_r(r) = alpha * dT * r exactly (representable in P1).
        let mesh = polar_mesh(0.0, 0.02, 8, 16);
        let mut prob = problem(&mesh);
        prob.delta_t_k = vec![50.0; mesh.num_elements()];
        let solver = RotorStressSolver::new(&mesh, &prob).expect("solver");
        let result = solver.solve(0.0).expect("solve");
        let max_vm = result.von_mises_pa.iter().copied().fold(0.0, f64::max);
        let scale = 200.0e9 * 12.0e-6 * 50.0;
        assert!(max_vm < 1.0e-6 * scale, "spurious thermal stress {max_vm}");
        let expected_growth = 12.0e-6 * 50.0 * 0.02;
        assert!(
            (result.max_radial_growth_m - expected_growth).abs()
                < 1.0e-9 + 1.0e-6 * expected_growth,
            "growth {} vs {expected_growth}",
            result.max_radial_growth_m
        );
    }

    #[test]
    fn safety_factor_scales_and_limit_speed_brackets() {
        // Pure centrifugal: SF ~ 1/omega^2 exactly, so the limit
        // speed from rpm0 is rpm0 * sqrt(SF0) (up to the a-term fit).
        let mesh = polar_mesh(0.005, 0.02, 10, 24);
        let solver = RotorStressSolver::new(&mesh, &problem(&mesh)).expect("solver");
        let rpm0 = 10_000.0;
        let omega0 = rpm0 * 2.0 * std::f64::consts::PI / 60.0;
        let sf0 = solver.solve(omega0).expect("solve").safety_factor_min;
        let limit = solver
            .limit_speed_rpm(rpm0)
            .expect("search")
            .expect("a limit exists");
        let expected = rpm0 * sf0.sqrt();
        assert!(
            (limit - expected).abs() / expected < 0.01,
            "limit {limit} vs omega^2 expectation {expected}"
        );
        let sf_at_limit = solver
            .solve(limit * 2.0 * std::f64::consts::PI / 60.0)
            .expect("solve")
            .safety_factor_min;
        assert!(
            (sf_at_limit - 1.0).abs() < 0.02,
            "SF at reported limit is {sf_at_limit}"
        );
    }

    #[test]
    fn invalid_problems_are_rejected() {
        let mesh = polar_mesh(0.0, 0.02, 2, 4);
        let mut prob = problem(&mesh);
        prob.materials.clear();
        assert!(RotorStressSolver::new(&mesh, &prob).is_err());
        let mut bad = problem(&mesh);
        bad.materials.insert(
            1,
            MechMaterial::Isotropic {
                youngs_modulus_pa: -1.0,
                poisson_ratio: 0.3,
                density_kg_per_m3: 1.0,
                thermal_expansion_per_k: 0.0,
                yield_pa: 1.0,
            },
        );
        assert!(RotorStressSolver::new(&mesh, &bad).is_err());
    }
}
