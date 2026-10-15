//! Elastoplastic shell form with through-thickness integration
//!
//! Extends the corotational shell with J2 plasticity. Each element has
//! multiple through-thickness integration points, each tracking its own
//! [`PlasticState`]. The constitutive response at each point comes from
//! [`return_mapping_plane_stress`] instead of a fixed elastic matrix.
//!
//! # Through-thickness integration
//!
//! 5-point Simpson rule on ζ ∈ [-1, 1] (physical z = ζ * t/2):
//!
//! ```text
//!   ζ:    -1     -0.5     0      0.5     1
//!   w:    1/6     4/6    2/6     4/6    1/6    (Simpson weights sum to 2)
//! ```
//!
//! Strain at ζ: `ε(ζ) = ε_membrane + (ζ * t/2) * κ`
//!
//! Resultant forces and moments:
//!   `N = (t/2) Σ w_i σ(ζ_i)`,  `M = (t/2)² Σ w_i ζ_i σ(ζ_i)`

use super::dkt;
use super::element::ShellElement;
use super::nonlinear::CorotationalShellForm;
use crate::physics::solid::plasticity::{
    J2Material, PlasticState, return_mapping_3d, return_mapping_plane_stress, update_plastic_state,
};

/// Number of through-thickness integration points (Simpson 5-point).
const N_THICKNESS: usize = 5;

/// Simpson 5-point rule on [-1, 1]: (ζ, weight).
const THICKNESS_POINTS: [(f64, f64); N_THICKNESS] = [
    (-1.0, 1.0 / 6.0),
    (-0.5, 4.0 / 6.0),
    (0.0, 2.0 / 6.0),
    (0.5, 4.0 / 6.0),
    (1.0, 1.0 / 6.0),
];

type Matrix3 = [f64; 9];
type Vector3 = [f64; 3];
type ThroughThicknessIntegration = (Matrix3, Matrix3, Matrix3, Vector3, Vector3);

/// Corotational shell form with J2 elastoplasticity.
///
/// Implements [`NonlinearElementForm<f64>`](crate::forms::NonlinearElementForm).
/// The `element_residual` and `element_tangent` methods use the committed
/// plastic state (read-only). Call [`commit_state`](Self::commit_state) after
/// each converged load step to update the plastic state.
pub struct ElastoplasticShellForm {
    /// Underlying corotational form (handles geometry/rotation).
    corot: CorotationalShellForm,
    /// J2 material for the plastic response.
    material: J2Material,
    /// Committed plastic state: `[n_elements][N_THICKNESS]`.
    states: Vec<[PlasticState; N_THICKNESS]>,
    /// Committed total strain (3D Voigt) per point: `[n_elements][N_THICKNESS]`.
    ///
    /// The return mapping takes strain increments, so trials subtract this.
    committed_strain: Vec<[[f64; 6]; N_THICKNESS]>,
}

impl ElastoplasticShellForm {
    pub fn new(elements: Vec<ShellElement>, n_dofs: usize, material: J2Material) -> Self {
        let n_elems = elements.len();
        let states = (0..n_elems)
            .map(|_| std::array::from_fn(|_| PlasticState::new_3d()))
            .collect();
        let committed_strain = vec![[[0.0; 6]; N_THICKNESS]; n_elems];
        Self {
            corot: CorotationalShellForm::new(elements, n_dofs),
            material,
            states,
            committed_strain,
        }
    }

    /// Update all plastic states after a converged Newton step.
    ///
    /// Recomputes the strain at each through-thickness point from the converged
    /// displacement and performs the 3D return mapping (with plane-stress
    /// iteration on ε_zz) to update stress and accumulated plastic strain.
    pub fn commit_state(&mut self, global_solution: &[f64]) {
        for elem_idx in 0..self.corot.elements.len() {
            let (eps_m, kappa) = self.element_strains(elem_idx, global_solution);
            let t = self.corot.elements[elem_idx].section.thickness;
            let half_t = t / 2.0;
            let nu = self.material.poisson_ratio;

            for (ip, &(zeta, _)) in THICKNESS_POINTS.iter().enumerate() {
                let z = zeta * half_t;
                let total = [
                    eps_m[0] + z * kappa[0],
                    eps_m[1] + z * kappa[1],
                    eps_m[2] + z * kappa[2],
                ];
                // The return mapping takes strain increments.
                let committed = self.committed_strain[elem_idx][ip];
                let d_inplane = [
                    total[0] - committed[0],
                    total[1] - committed[1],
                    total[2] - committed[5],
                ];

                // Iterate on the out-of-plane increment for sigma_zz = 0.
                let mut d_zz = -nu / (1.0 - nu) * (d_inplane[0] + d_inplane[1]);
                for _ in 0..20 {
                    let strain_inc = [d_inplane[0], d_inplane[1], d_zz, 0.0, 0.0, d_inplane[2]];
                    let update =
                        return_mapping_3d(&self.material, &self.states[elem_idx][ip], &strain_inc);
                    let sigma_zz = update.stress[2];
                    if sigma_zz.abs() < 1e-12 {
                        break;
                    }
                    let c_zz = update.tangent[2 * 6 + 2];
                    if c_zz.abs() < 1e-30 {
                        break;
                    }
                    d_zz -= sigma_zz / c_zz;
                }

                // Final 3D return mapping with converged out-of-plane increment.
                let strain_inc = [d_inplane[0], d_inplane[1], d_zz, 0.0, 0.0, d_inplane[2]];
                let update =
                    return_mapping_3d(&self.material, &self.states[elem_idx][ip], &strain_inc);
                update_plastic_state(&mut self.states[elem_idx][ip], &update, &self.material);
                self.committed_strain[elem_idx][ip] =
                    [total[0], total[1], committed[2] + d_zz, 0.0, 0.0, total[2]];
            }
        }
    }

    /// Number of elements currently yielding (at least one through-thickness point).
    pub fn yielding_count(&self) -> usize {
        self.states
            .iter()
            .filter(|s| s.iter().any(|p| p.yielding))
            .count()
    }

    /// Maximum equivalent plastic strain across all elements and points.
    pub fn max_equiv_plastic_strain(&self) -> f64 {
        self.states
            .iter()
            .flat_map(|s| s.iter())
            .map(|p| p.equiv_plastic_strain)
            .fold(0.0f64, f64::max)
    }

    /// Compute membrane strain and curvature for an element in the corotated local frame.
    ///
    /// Returns `(ε_m, κ)` each as `[xx, yy, xy]`.
    fn element_strains(&self, elem_idx: usize, global_solution: &[f64]) -> ([f64; 3], [f64; 3]) {
        let elem = &self.corot.elements[elem_idx];
        let dofs = elem.global_dofs();

        // Global displacements
        let mut u_global = [0.0; 18];
        for (i, &d) in dofs.iter().enumerate() {
            u_global[i] = global_solution[d];
        }

        // Rotate to corotated local frame
        let def_pts = self.corot.deformed_points(elem_idx, global_solution);
        let rot_def = CorotationalShellForm::deformed_rotation(&def_pts);
        let t18 = CorotationalShellForm::build_rotation_18x18(&rot_def);

        let mut d_local = [0.0; 18];
        for i in 0..18 {
            for j in 0..18 {
                d_local[i] += t18[i * 18 + j] * u_global[j];
            }
        }

        // Extract membrane DOFs: [u1, v1, u2, v2, u3, v3]
        let d_m = [
            d_local[0],
            d_local[1],
            d_local[6],
            d_local[7],
            d_local[12],
            d_local[13],
        ];
        // Extract bending DOFs: [w1, θx1, θy1, w2, θx2, θy2, w3, θx3, θy3]
        let d_b = [
            d_local[2],
            d_local[3],
            d_local[4],
            d_local[8],
            d_local[9],
            d_local[10],
            d_local[14],
            d_local[15],
            d_local[16],
        ];

        // CST membrane strain: ε_m = B_m * d_m
        let b_m = elem.cst_b_matrix();
        let mut eps_m = [0.0; 3];
        for i in 0..3 {
            for j in 0..6 {
                eps_m[i] += b_m[i * 6 + j] * d_m[j];
            }
        }

        // DKT curvature at centroid: κ = B_b(1/3, 1/3) * d_b
        let b_b = dkt::dkt_b_matrix_at(&elem.local_coords, elem.area, 1.0 / 3.0, 1.0 / 3.0);
        let mut kappa = [0.0; 3];
        for i in 0..3 {
            for j in 0..9 {
                kappa[i] += b_b[i * 9 + j] * d_b[j];
            }
        }

        (eps_m, kappa)
    }

    /// Compute the through-thickness integrated constitutive matrices.
    ///
    /// Returns `(D_mm, D_mb, D_bb, N, M)`:
    /// - `D_mm` (3×3): membrane tangent
    /// - `D_mb` (3×3): membrane-bending coupling tangent
    /// - `D_bb` (3×3): bending tangent
    /// - `N` (3): resultant membrane force
    /// - `M` (3): resultant bending moment
    fn through_thickness_integration(
        &self,
        elem_idx: usize,
        eps_m: &[f64; 3],
        kappa: &[f64; 3],
    ) -> ThroughThicknessIntegration {
        let t = self.corot.elements[elem_idx].section.thickness;
        let half_t = t / 2.0;

        let mut d_mm = [0.0; 9];
        let mut d_mb = [0.0; 9];
        let mut d_bb = [0.0; 9];
        let mut n_res = [0.0; 3];
        let mut m_res = [0.0; 3];

        for (ip, &(zeta, w)) in THICKNESS_POINTS.iter().enumerate() {
            let z = zeta * half_t;

            // Strain increment at this thickness point (trial from committed).
            let committed = self.committed_strain[elem_idx][ip];
            let strain_inc = [
                eps_m[0] + z * kappa[0] - committed[0],
                eps_m[1] + z * kappa[1] - committed[1],
                eps_m[2] + z * kappa[2] - committed[5],
            ];

            // Return mapping (trial from committed state)
            let update = return_mapping_plane_stress(
                &self.material,
                &self.states[elem_idx][ip],
                &strain_inc,
            );

            // Integration weight: w * (t/2) for the dz measure
            let wt = w * half_t;

            for i in 0..3 {
                n_res[i] += wt * update.stress[i];
                m_res[i] += wt * z * update.stress[i];
            }

            for i in 0..3 {
                for j in 0..3 {
                    let c = update.tangent[i * 3 + j];
                    d_mm[i * 3 + j] += wt * c;
                    d_mb[i * 3 + j] += wt * z * c;
                    d_bb[i * 3 + j] += wt * z * z * c;
                }
            }
        }

        (d_mm, d_mb, d_bb, n_res, m_res)
    }
}

impl crate::forms::NonlinearElementForm<f64> for ElastoplasticShellForm {
    fn num_elements(&self) -> usize {
        self.corot.elements.len()
    }

    fn num_local_dofs(&self, _elem_idx: usize) -> usize {
        18
    }

    fn local_to_global(&self, elem_idx: usize) -> Vec<usize> {
        self.corot.elements[elem_idx].global_dofs().to_vec()
    }

    fn num_global_dofs(&self) -> usize {
        self.corot.n_dofs
    }

    fn element_residual(&self, elem_idx: usize, global_solution: &[f64]) -> Vec<f64> {
        let elem = &self.corot.elements[elem_idx];
        let dofs = elem.global_dofs();

        let mut u_global = [0.0; 18];
        for (i, &d) in dofs.iter().enumerate() {
            u_global[i] = global_solution[d];
        }

        let (eps_m, _) = self.element_strains(elem_idx, global_solution);
        let a = elem.area;
        let b_m = elem.cst_b_matrix();

        let def_pts = self.corot.deformed_points(elem_idx, global_solution);
        let rot_def = CorotationalShellForm::deformed_rotation(&def_pts);
        let t18 = CorotationalShellForm::build_rotation_18x18(&rot_def);

        let mut d_local = [0.0; 18];
        for i in 0..18 {
            for j in 0..18 {
                d_local[i] += t18[i * 18 + j] * u_global[j];
            }
        }

        // Extract bending DOFs for curvature at Gauss points
        let d_b = [
            d_local[2],
            d_local[3],
            d_local[4],
            d_local[8],
            d_local[9],
            d_local[10],
            d_local[14],
            d_local[15],
            d_local[16],
        ];

        let mem_pos = [0usize, 1, 6, 7, 12, 13];
        let bend_pos = [2usize, 3, 4, 8, 9, 10, 14, 15, 16];

        let mut f_local = [0.0; 18];

        // Membrane: constant B_m, single-point through-thickness
        let kappa_c = eval_curvature(&elem.local_coords, elem.area, 1.0 / 3.0, 1.0 / 3.0, &d_b);
        let (_, _, _, n_res, _) = self.through_thickness_integration(elem_idx, &eps_m, &kappa_c);
        for (mi, &gi) in mem_pos.iter().enumerate() {
            for k in 0..3 {
                f_local[gi] += b_m[k * 6 + mi] * n_res[k] * a;
            }
        }

        // Bending: 3-point Gauss quadrature over the triangle
        let gauss_pts: [(f64, f64, f64); 3] = [
            (1.0 / 6.0, 1.0 / 6.0, 1.0 / 3.0),
            (2.0 / 3.0, 1.0 / 6.0, 1.0 / 3.0),
            (1.0 / 6.0, 2.0 / 3.0, 1.0 / 3.0),
        ];
        for &(xi, eta, w_gp) in &gauss_pts {
            let b_b = dkt::dkt_b_matrix_at(&elem.local_coords, a, xi, eta);
            let kappa = eval_curvature(&elem.local_coords, a, xi, eta, &d_b);
            let (_, _, _, _, m_gp) = self.through_thickness_integration(elem_idx, &eps_m, &kappa);
            for (bi, &gi) in bend_pos.iter().enumerate() {
                for k in 0..3 {
                    f_local[gi] += w_gp * a * b_b[k * 9 + bi] * m_gp[k];
                }
            }
        }

        // Drilling
        let drill = elem.section.membrane_stiffness() * a * 1e-6;
        for i in 0..3 {
            let dof = i * 6 + 5;
            f_local[dof] += drill * d_local[dof];
        }

        // Project to global with the frame variation: a bare transpose
        // rotation omits (dq/du) terms and yields nonconservative forces.
        self.corot
            .project_local_forces(elem_idx, global_solution, &f_local)
    }

    fn element_tangent(&self, elem_idx: usize, global_solution: &[f64]) -> Vec<f64> {
        let dofs = self.corot.elements[elem_idx].global_dofs();
        if dofs.iter().all(|&dof| global_solution[dof] == 0.0)
            && self.states[elem_idx]
                .iter()
                .all(|state| state.equiv_plastic_strain == 0.0)
        {
            return crate::forms::NonlinearElementForm::element_tangent(
                &self.corot,
                elem_idx,
                global_solution,
            );
        }
        let elem = &self.corot.elements[elem_idx];
        let a = elem.area;

        let (eps_m, _) = self.element_strains(elem_idx, global_solution);
        let b_m = elem.cst_b_matrix();

        let def_pts = self.corot.deformed_points(elem_idx, global_solution);
        let rot_def = CorotationalShellForm::deformed_rotation(&def_pts);
        let t18 = CorotationalShellForm::build_rotation_18x18(&rot_def);

        let mut d_local = [0.0; 18];
        {
            let dofs = elem.global_dofs();
            let mut u_global = [0.0; 18];
            for (i, &d) in dofs.iter().enumerate() {
                u_global[i] = global_solution[d];
            }
            for i in 0..18 {
                for j in 0..18 {
                    d_local[i] += t18[i * 18 + j] * u_global[j];
                }
            }
        }
        let d_b = [
            d_local[2],
            d_local[3],
            d_local[4],
            d_local[8],
            d_local[9],
            d_local[10],
            d_local[14],
            d_local[15],
            d_local[16],
        ];

        let mem_pos = [0usize, 1, 6, 7, 12, 13];
        let bend_pos = [2usize, 3, 4, 8, 9, 10, 14, 15, 16];

        let mut k_local = vec![0.0; 18 * 18];

        // Membrane: constant B_m, centroid through-thickness integration
        let kappa_c = eval_curvature(&elem.local_coords, a, 1.0 / 3.0, 1.0 / 3.0, &d_b);
        let (d_mm, d_mb, _, _, _) = self.through_thickness_integration(elem_idx, &eps_m, &kappa_c);
        add_btdb(
            &mut k_local,
            18,
            &mem_pos,
            &mem_pos,
            &b_m,
            3,
            6,
            &d_mm,
            &b_m,
            3,
            6,
            a,
        );

        // Membrane-bending coupling: B_m^T D_mb B_b with 3-pt Gauss for B_b
        let gauss_pts: [(f64, f64, f64); 3] = [
            (1.0 / 6.0, 1.0 / 6.0, 1.0 / 3.0),
            (2.0 / 3.0, 1.0 / 6.0, 1.0 / 3.0),
            (1.0 / 6.0, 2.0 / 3.0, 1.0 / 3.0),
        ];
        for &(xi, eta, w_gp) in &gauss_pts {
            let b_b = dkt::dkt_b_matrix_at(&elem.local_coords, a, xi, eta);
            add_btdb(
                &mut k_local,
                18,
                &mem_pos,
                &bend_pos,
                &b_m,
                3,
                6,
                &d_mb,
                &b_b,
                3,
                9,
                w_gp * a,
            );
            let d_mb_t = transpose_3x3(&d_mb);
            add_btdb(
                &mut k_local,
                18,
                &bend_pos,
                &mem_pos,
                &b_b,
                3,
                9,
                &d_mb_t,
                &b_m,
                3,
                6,
                w_gp * a,
            );
        }

        // Bending-bending: B_b^T D_bb B_b with 3-pt Gauss for B_b
        for &(xi, eta, w_gp) in &gauss_pts {
            let b_b = dkt::dkt_b_matrix_at(&elem.local_coords, a, xi, eta);
            let kappa = eval_curvature(&elem.local_coords, a, xi, eta, &d_b);
            let (_, _, d_bb, _, _) = self.through_thickness_integration(elem_idx, &eps_m, &kappa);
            add_btdb(
                &mut k_local,
                18,
                &bend_pos,
                &bend_pos,
                &b_b,
                3,
                9,
                &d_bb,
                &b_b,
                3,
                9,
                w_gp * a,
            );
        }

        // Drilling
        let drill = elem.section.membrane_stiffness() * a * 1e-6;
        for i in 0..3 {
            let dof = i * 6 + 5;
            k_local[dof * 18 + dof] = drill;
        }

        // Transform to global
        let mut temp = vec![0.0; 18 * 18];
        for i in 0..18 {
            for j in 0..18 {
                for p in 0..18 {
                    temp[i * 18 + j] += k_local[i * 18 + p] * t18[p * 18 + j];
                }
            }
        }
        let mut k_global = vec![0.0; 18 * 18];
        for i in 0..18 {
            for j in 0..18 {
                for p in 0..18 {
                    k_global[i * 18 + j] += t18[p * 18 + i] * temp[p * 18 + j];
                }
            }
        }
        k_global
    }
}

/// Add `scale * B1^T * D * B2` into matrix `k` at positions `(rows[i], cols[j])`.
#[allow(clippy::too_many_arguments)]
fn add_btdb(
    k: &mut [f64],
    k_dim: usize,
    rows: &[usize],
    cols: &[usize],
    b1: &[f64],
    n_stress: usize,
    n_dof1: usize,
    d: &[f64; 9],
    b2: &[f64],
    _n_stress2: usize,
    n_dof2: usize,
    scale: f64,
) {
    // db2 = D * B2 (n_stress × n_dof2)
    let mut db2 = vec![0.0; n_stress * n_dof2];
    for i in 0..n_stress {
        for j in 0..n_dof2 {
            for p in 0..n_stress {
                db2[i * n_dof2 + j] += d[i * 3 + p] * b2[p * n_dof2 + j];
            }
        }
    }

    // K += scale * B1^T * (D * B2)
    for (li, &gi) in rows.iter().enumerate() {
        for (lj, &gj) in cols.iter().enumerate() {
            let mut val = 0.0;
            for p in 0..n_stress {
                val += b1[p * n_dof1 + li] * db2[p * n_dof2 + lj];
            }
            k[gi * k_dim + gj] += scale * val;
        }
    }
}

/// Evaluate DKT curvature at a point from bending DOFs.
fn eval_curvature(
    coords: &[(f64, f64); 3],
    area: f64,
    xi: f64,
    eta: f64,
    d_b: &[f64; 9],
) -> [f64; 3] {
    let b_b = dkt::dkt_b_matrix_at(coords, area, xi, eta);
    let mut kappa = [0.0; 3];
    for i in 0..3 {
        for j in 0..9 {
            kappa[i] += b_b[i * 9 + j] * d_b[j];
        }
    }
    kappa
}

fn transpose_3x3(m: &[f64; 9]) -> [f64; 9] {
    [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forms::NonlinearElementForm;
    use crate::mesh::Point;
    use crate::physics::solid::plasticity::HardeningLaw;
    use crate::physics::solid::shell::ShellSection;

    fn test_elements() -> (Vec<ShellElement>, usize) {
        let p0 = Point::new_3d(0.0, 0.0, 0.0);
        let p1 = Point::new_3d(1.0, 0.0, 0.0);
        let p2 = Point::new_3d(0.5, 0.8, 0.0);
        let section = ShellSection::new(200e9, 0.3, 0.002, 7800.0);
        let elements = vec![ShellElement::from_points([0, 1, 2], &p0, &p1, &p2, section)];
        (elements, 18)
    }

    #[test]
    fn test_zero_displacement_zero_residual() {
        let (elements, n_dofs) = test_elements();
        let mat = J2Material::mild_steel();
        let form = ElastoplasticShellForm::new(elements, n_dofs, mat);
        let u = vec![0.0; n_dofs];

        let r = form.element_residual(0, &u);
        for (i, &ri) in r.iter().enumerate() {
            assert!(ri.abs() < 1e-6, "Residual[{i}] = {ri}");
        }
    }

    #[test]
    fn test_elastic_tangent_matches_corotational() {
        // For elastic (small) loading, elastoplastic tangent should match
        // the purely elastic corotational tangent.
        let (elements, n_dofs) = test_elements();
        let e = 200e9;
        let nu = 0.3;
        let mat = J2Material::new(e, nu, 1e20, HardeningLaw::Perfect); // very high yield → stays elastic
        let form_ep = ElastoplasticShellForm::new(elements.clone(), n_dofs, mat);
        let form_el = CorotationalShellForm::new(elements, n_dofs);
        let u = vec![0.0; n_dofs];

        let k_ep = form_ep.element_tangent(0, &u);
        let k_el = form_el.element_tangent(0, &u);

        for i in 0..18 {
            for j in 0..18 {
                let diff = (k_ep[i * 18 + j] - k_el[i * 18 + j]).abs();
                let scale = k_el[i * 18 + i].abs().max(1.0);
                assert!(
                    diff / scale < 0.05,
                    "K_ep[{i},{j}]={:.4e} vs K_el[{i},{j}]={:.4e} (diff/scale={:.4e})",
                    k_ep[i * 18 + j],
                    k_el[i * 18 + j],
                    diff / scale,
                );
            }
        }
    }

    #[test]
    fn test_virgin_tangent_is_exact_corotational_jacobian() {
        // The virgin-state tangent delegates to the corotational Jacobian,
        // which is genuinely nonsymmetric (the rotated-local force is not an
        // exact energy gradient), so symmetry must NOT be asserted here:
        // symmetrising it corrupts the Newton direction into an ascent
        // direction in membrane-dominated states. Pin the delegation exactly.
        let (elements, n_dofs) = test_elements();
        let mat = J2Material::mild_steel();
        let form = ElastoplasticShellForm::new(elements.clone(), n_dofs, mat);
        let corot = CorotationalShellForm::new(elements, n_dofs);
        let u = vec![0.0; n_dofs];

        let k_ep = form.element_tangent(0, &u);
        let k_el = corot.element_tangent(0, &u);
        assert_eq!(k_ep.len(), k_el.len());
        for (i, (&a, &b)) in k_ep.iter().zip(k_el.iter()).enumerate() {
            assert!(
                a == b,
                "virgin tangent entry {i} not delegated: {a:e} vs corot {b:e}"
            );
        }
    }

    #[test]
    fn test_virgin_residual_matches_corotational_tilted() {
        // Virgin-state residual must match the corotational residual on
        // tilted elements: the global projection needs the frame
        // variation, and a bare transpose rotation only agrees on flat
        // elements (where the variation vanishes).
        let section = ShellSection::new(200e9, 0.3, 0.002, 7800.0);
        let angle = 30.0f64.to_radians();
        let (c, s) = (angle.cos(), angle.sin());
        let rot = |x: f64, y: f64| Point::new_3d(c * x, y, -s * x);
        let elements = vec![ShellElement::from_points(
            [0, 1, 2],
            &rot(0.0, 0.0),
            &rot(1.0, 0.0),
            &rot(0.0, 1.0),
            section,
        )];
        let mat = J2Material::new(200e9, 0.3, 1e20, HardeningLaw::Perfect);
        let form_ep = ElastoplasticShellForm::new(elements.clone(), 18, mat);
        let form_el = CorotationalShellForm::new(elements, 18);
        let mut u = vec![0.0; 18];
        u[6] = 1e-9 * c;
        u[8] = 2e-9;
        u[14] = 3e-9;
        let r_ep = form_ep.element_residual(0, &u);
        let r_el = form_el.element_residual(0, &u);
        let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let diff: Vec<f64> = r_ep.iter().zip(r_el.iter()).map(|(a, b)| a - b).collect();
        let rel = norm(&diff) / norm(&r_el);
        assert!(rel < 1e-6, "virgin residual mismatch tilted: {rel:e}");
    }

    #[test]
    fn test_commit_state_accumulates_plastic_strain() {
        let (elements, n_dofs) = test_elements();
        let mat = J2Material::new(200e9, 0.3, 100.0, HardeningLaw::Perfect); // very low yield
        let mut form = ElastoplasticShellForm::new(elements, n_dofs, mat);

        // Apply large membrane strain
        let mut u = vec![0.0; n_dofs];
        u[0] = 0.1; // large u_x on node 0

        form.commit_state(&u);

        assert!(form.max_equiv_plastic_strain() > 0.0);
        assert!(form.yielding_count() > 0);
    }
}
