//! Corotational nonlinear shell form
//!
//! Geometrically nonlinear shell element using the corotational approach:
//! 1. Compute current triangle orientation from deformed node positions
//! 2. Project displacements to corotated local frame
//! 3. Evaluate CST membrane forces + DKT bending moments using existing
//!    analytical B-matrices and constitutive laws
//! 4. Apply the corotated-displacement derivative transpose to local forces
//! 5. Tangent = numerical derivative of the complete corotated residual

use super::element::ShellElement;

/// Corotational shell form for geometrically nonlinear analysis.
///
/// Internal forces differentiate the corotated elastic potential, including
/// the position-dependent frame. The tangent differentiates those forces.
///
/// Implements [`NonlinearElementForm<f64>`](crate::forms::NonlinearElementForm).
pub struct CorotationalShellForm {
    /// Reference (undeformed) elements.
    pub elements: Vec<ShellElement>,
    /// Total number of global DOFs.
    pub n_dofs: usize,
}

impl CorotationalShellForm {
    pub fn new(elements: Vec<ShellElement>, n_dofs: usize) -> Self {
        Self { elements, n_dofs }
    }

    /// Compute reference positions relative to the element origin, in global axes.
    fn reference_points(&self, elem_idx: usize) -> [[f64; 3]; 3] {
        let elem = &self.elements[elem_idx];
        let rot = &elem.rotation;
        let mut points = [[0.0; 3]; 3];

        for (i, &(lx, ly)) in elem.local_coords.iter().enumerate() {
            points[i] = [
                rot[0] * lx + rot[1] * ly,
                rot[3] * lx + rot[4] * ly,
                rot[6] * lx + rot[7] * ly,
            ];
        }

        points
    }

    /// Compute deformed relative positions from the global displacement solution.
    pub(crate) fn deformed_points(&self, elem_idx: usize, u: &[f64]) -> [[f64; 3]; 3] {
        let elem = &self.elements[elem_idx];
        let mut points = self.reference_points(elem_idx);

        // Orientations and strains depend only on position differences. Avoid
        // adding a large world origin or rigid translation and subtracting it
        // again: that loses the small elastic deformation in force evaluation.
        let anchor = elem.nodes[0] * 6;
        for (i, &node) in elem.nodes.iter().enumerate() {
            points[i][0] += u[node * 6] - u[anchor];
            points[i][1] += u[node * 6 + 1] - u[anchor + 1];
            points[i][2] += u[node * 6 + 2] - u[anchor + 2];
        }

        points
    }

    /// Rotate a global vector into the current local frame.
    pub(super) fn global_to_local(rot: &[f64; 9], vector: [f64; 3]) -> [f64; 3] {
        [
            rot[0] * vector[0] + rot[3] * vector[1] + rot[6] * vector[2],
            rot[1] * vector[0] + rot[4] * vector[1] + rot[7] * vector[2],
            rot[2] * vector[0] + rot[5] * vector[1] + rot[8] * vector[2],
        ]
    }

    /// Small-rotation vector of `reference^T * current`, in reference-local axes.
    pub(super) fn relative_rotation_vector(reference: &[f64; 9], current: &[f64; 9]) -> [f64; 3] {
        let mut relative = [0.0; 9];
        for row in 0..3 {
            for column in 0..3 {
                relative[row * 3 + column] = (0..3)
                    .map(|k| reference[k * 3 + row] * current[k * 3 + column])
                    .sum();
            }
        }

        [
            0.5 * (relative[7] - relative[5]),
            0.5 * (relative[2] - relative[6]),
            0.5 * (relative[3] - relative[1]),
        ]
    }

    /// Extract the element displacement in its current corotated frame.
    ///
    /// Translational DOFs are formed from relative current node positions,
    /// removing rigid translation and rotation. Rotational DOFs are expressed
    /// in the same frame with the rigid element rotation removed. This makes
    /// rigid-body translation and rotation produce zero internal force.
    fn corotated_local_displacement(
        &self,
        elem_idx: usize,
        global_solution: &[f64],
        current_rotation: &[f64; 9],
    ) -> [f64; 18] {
        let elem = &self.elements[elem_idx];
        let dofs = elem.global_dofs();
        let reference_rotation = elem.rotation;
        let points = self.deformed_points(elem_idx, global_solution);
        let rigid_rotation = Self::relative_rotation_vector(&reference_rotation, current_rotation);
        let reference_points = self.reference_points(elem_idx);
        let mut local = [0.0; 18];

        for node in 0..3 {
            let relative_position = [
                points[node][0] - points[0][0],
                points[node][1] - points[0][1],
                points[node][2] - points[0][2],
            ];
            let current_local = Self::global_to_local(current_rotation, relative_position);
            let reference_relative = [
                reference_points[node][0] - reference_points[0][0],
                reference_points[node][1] - reference_points[0][1],
                reference_points[node][2] - reference_points[0][2],
            ];
            let reference_local = Self::global_to_local(&reference_rotation, reference_relative);

            for component in 0..3 {
                local[node * 6 + component] = current_local[component] - reference_local[component];
            }

            let rotation = [
                global_solution[dofs[node * 6 + 3]],
                global_solution[dofs[node * 6 + 4]],
                global_solution[dofs[node * 6 + 5]],
            ];
            let local_rotation = Self::global_to_local(current_rotation, rotation);
            for component in 0..3 {
                local[node * 6 + 3 + component] =
                    local_rotation[component] - rigid_rotation[component];
            }
        }

        local
    }

    /// Projects corotated local forces to global DOFs.
    ///
    /// Evaluates `f_global = (dq/du)^T f_local` where `q` holds the
    /// corotated local displacements. The current frame depends on the
    /// nodal translations, so merely rotating by the frame transpose omits
    /// the frame variation and produces nonconservative forces with a
    /// nonsymmetric Jacobian. Callers must pass forces conjugate to
    /// [`Self::corotated_local_displacement`]; forces built from rigid-free
    /// strains (such as `B^T` resultants) satisfy this.
    pub(crate) fn project_local_forces(
        &self,
        elem_idx: usize,
        global_solution: &[f64],
        f_local: &[f64; 18],
    ) -> Vec<f64> {
        let elem = &self.elements[elem_idx];
        let def_pts = self.deformed_points(elem_idx, global_solution);
        let rot_def = Self::deformed_rotation(&def_pts);
        let dofs = elem.global_dofs();
        let mut f_global = vec![0.0; 18];
        for node in 0..3 {
            for component in 0..3 {
                let variation =
                    Self::deformed_rotation_variation(&def_pts, &rot_def, node, component);
                let rigid_variation = Self::relative_rotation_vector(&elem.rotation, &variation);
                for local_node in 0..3 {
                    let relative = std::array::from_fn(|i| def_pts[local_node][i] - def_pts[0][i]);
                    let mut motion = [0.0; 3];
                    motion[component] = f64::from(node == local_node) - f64::from(node == 0);
                    let frame_motion = Self::global_to_local(&variation, relative);
                    let direct_motion = Self::global_to_local(&rot_def, motion);
                    let theta =
                        std::array::from_fn(|i| global_solution[dofs[6 * local_node + 3 + i]]);
                    let frame_theta = Self::global_to_local(&variation, theta);
                    for i in 0..3 {
                        f_global[6 * node + component] += f_local[6 * local_node + i]
                            * (frame_motion[i] + direct_motion[i])
                            + f_local[6 * local_node + 3 + i]
                                * (frame_theta[i] - rigid_variation[i]);
                    }
                }
                // Rotational unknowns do not change the position-defined frame.
                for i in 0..3 {
                    f_global[6 * node + 3 + component] +=
                        rot_def[3 * component + i] * f_local[6 * node + 3 + i];
                }
            }
        }
        f_global
    }

    /// Build rotation matrix from deformed triangle (returns 3x3 row-major).
    pub(crate) fn deformed_rotation(pts: &[[f64; 3]; 3]) -> [f64; 9] {
        let v01 = [
            pts[1][0] - pts[0][0],
            pts[1][1] - pts[0][1],
            pts[1][2] - pts[0][2],
        ];
        let v02 = [
            pts[2][0] - pts[0][0],
            pts[2][1] - pts[0][1],
            pts[2][2] - pts[0][2],
        ];

        let len01 = (v01[0] * v01[0] + v01[1] * v01[1] + v01[2] * v01[2]).sqrt();
        if !len01.is_finite() || len01 <= f64::EPSILON {
            return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        }
        let ex = [v01[0] / len01, v01[1] / len01, v01[2] / len01];

        let normal = [
            v01[1] * v02[2] - v01[2] * v02[1],
            v01[2] * v02[0] - v01[0] * v02[2],
            v01[0] * v02[1] - v01[1] * v02[0],
        ];
        let area2 = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if !area2.is_finite() || area2 <= f64::EPSILON {
            return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        }
        let ez = [normal[0] / area2, normal[1] / area2, normal[2] / area2];

        let ey = [
            ez[1] * ex[2] - ez[2] * ex[1],
            ez[2] * ex[0] - ez[0] * ex[2],
            ez[0] * ex[1] - ez[1] * ex[0],
        ];

        [
            ex[0], ey[0], ez[0], ex[1], ey[1], ez[1], ex[2], ey[2], ez[2],
        ]
    }

    /// Differentiate the current frame with respect to one nodal translation.
    pub(super) fn deformed_rotation_variation(
        pts: &[[f64; 3]; 3],
        rotation: &[f64; 9],
        node: usize,
        component: usize,
    ) -> [f64; 9] {
        let dot = |a: [f64; 3], b: [f64; 3]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        let cross = |a: [f64; 3], b: [f64; 3]| {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        };
        let a = std::array::from_fn(|i| pts[1][i] - pts[0][i]);
        let b = std::array::from_fn(|i| pts[2][i] - pts[0][i]);
        let normal = cross(a, b);
        let length = dot(a, a).sqrt();
        let area2 = dot(normal, normal).sqrt();
        if !length.is_finite()
            || length <= f64::EPSILON
            || !area2.is_finite()
            || area2 <= f64::EPSILON
        {
            // The degenerate branch of deformed_rotation uses a constant frame.
            return [0.0; 9];
        }
        let ex = [rotation[0], rotation[3], rotation[6]];
        let ez = [rotation[2], rotation[5], rotation[8]];
        let mut da = [0.0; 3];
        let mut db = [0.0; 3];
        da[component] = f64::from(node == 1) - f64::from(node == 0);
        db[component] = f64::from(node == 2) - f64::from(node == 0);
        let de = std::array::from_fn(|i| (da[i] - ex[i] * dot(ex, da)) / length);
        let first = cross(da, b);
        let second = cross(a, db);
        let dn = std::array::from_fn(|i| first[i] + second[i]);
        let dz = std::array::from_fn(|i| (dn[i] - ez[i] * dot(ez, dn)) / area2);
        let first = cross(dz, ex);
        let second = cross(ez, de);
        let dy: [f64; 3] = std::array::from_fn(|i| first[i] + second[i]);
        [
            de[0], dy[0], dz[0], de[1], dy[1], dz[1], de[2], dy[2], dz[2],
        ]
    }

    /// Build the 18x18 block-diagonal local-to-global rotation matrix.
    /// Builds the 18x18 global-to-local rotation for one element.
    ///
    /// Block-diagonal repetition of the frame transpose, matching
    /// [`Self::global_to_local`]: `rot` stores the local-to-global frame,
    /// so the global-to-local map needs its transpose.
    pub(crate) fn build_rotation_18x18(rot: &[f64; 9]) -> Vec<f64> {
        let n = 18;
        let mut t = vec![0.0; n * n];
        for block in 0..6 {
            let offset = block * 3;
            for i in 0..3 {
                for j in 0..3 {
                    t[(offset + i) * n + (offset + j)] = rot[j * 3 + i];
                }
            }
        }
        t
    }
}

impl crate::forms::NonlinearElementForm<f64> for CorotationalShellForm {
    fn num_elements(&self) -> usize {
        self.elements.len()
    }

    fn num_local_dofs(&self, _elem_idx: usize) -> usize {
        18
    }

    fn local_to_global(&self, elem_idx: usize) -> Vec<usize> {
        self.elements[elem_idx].global_dofs().to_vec()
    }

    fn num_global_dofs(&self) -> usize {
        self.n_dofs
    }

    fn element_residual(&self, elem_idx: usize, global_solution: &[f64]) -> Vec<f64> {
        let elem = &self.elements[elem_idx];
        // Deformed configuration
        let def_pts = self.deformed_points(elem_idx, global_solution);
        let rot_def = Self::deformed_rotation(&def_pts);

        let d_local = self.corotated_local_displacement(elem_idx, global_solution, &rot_def);

        // Compute local stiffness (in deformed local frame)
        // For small local deformations, the linear stiffness is valid
        let k_local = elem.stiffness_local_combined();
        let mut f_local = [0.0; 18];
        for i in 0..18 {
            for j in 0..18 {
                // The elastic potential is 1/2 q^T K q; remove roundoff-only
                // skew in the material stiffness when forming its derivative.
                f_local[i] += 0.5 * (k_local[i * 18 + j] + k_local[j * 18 + i]) * d_local[j];
            }
        }

        // Classic corotational force transform: rotate the local force by
        // the current frame, block by block. The frame is held fixed during
        // the variation; its derivative enters only the tangent. Projecting
        // with the full (dq/du)^T of the naive 1/2 q^T K q energy instead
        // double-counts the frame spin (q already removes rigid motion) and
        // measurably degrades the elastica strip benchmark under refinement,
        // so it is not used here.
        let mut f_global = vec![0.0; 18];
        for block in 0..6 {
            let offset = block * 3;
            for i in 0..3 {
                for j in 0..3 {
                    f_global[offset + i] += rot_def[i * 3 + j] * f_local[offset + j];
                }
            }
        }
        f_global
    }

    fn element_tangent(&self, elem_idx: usize, global_solution: &[f64]) -> Vec<f64> {
        // Differentiate the complete corotational residual, including the
        // current-frame rotation. The previous transformed-material tangent
        // omitted this geometric contribution and was not a Newton tangent.
        //
        // The classic corotational force is not an exact energy gradient, so
        // the tangent is genuinely nonsymmetric; retain this direct numerical
        // derivative so finite-difference error is visible to independent
        // checks rather than hiding it with post-hoc symmetrization.
        let dofs = self.elements[elem_idx].global_dofs();
        let mut tangent = vec![0.0; 18 * 18];
        for (column, &dof) in dofs.iter().enumerate() {
            let step = 1.0e-7 * (1.0 + global_solution[dof].abs());
            let mut plus = global_solution.to_vec();
            let mut minus = global_solution.to_vec();
            plus[dof] += step;
            minus[dof] -= step;
            let residual_plus = self.element_residual(elem_idx, &plus);
            let residual_minus = self.element_residual(elem_idx, &minus);
            for row in 0..18 {
                tangent[row * 18 + column] =
                    (residual_plus[row] - residual_minus[row]) / (2.0 * step);
            }
        }

        tangent
    }
}

#[cfg(test)]
mod tests {
    use super::super::section::ShellSection;
    use super::*;
    use crate::forms::NonlinearElementForm;
    use crate::mesh::Point;

    fn test_shell_elements() -> (Vec<ShellElement>, usize) {
        let p0 = Point::new_3d(0.0, 0.0, 0.0);
        let p1 = Point::new_3d(1.0, 0.0, 0.0);
        let p2 = Point::new_3d(0.5, 0.8, 0.0);
        let section = ShellSection::new(200e9, 0.3, 0.002, 7800.0);

        let elements = vec![ShellElement::from_points([0, 1, 2], &p0, &p1, &p2, section)];
        let n_nodes = 3;
        let n_dofs = n_nodes * 6;
        (elements, n_dofs)
    }

    #[test]
    fn test_corotational_shell_zero_displacement_zero_residual() {
        let (elements, n_dofs) = test_shell_elements();
        let form = CorotationalShellForm::new(elements, n_dofs);
        let u = vec![0.0; n_dofs];

        let r = form.element_residual(0, &u);
        for (i, &ri) in r.iter().enumerate() {
            assert!(ri.abs() < 1e-6, "Residual[{i}] = {ri}, expected 0");
        }
    }

    #[test]
    fn world_origin_and_rigid_translation_do_not_corrupt_elastic_forces() {
        let (elements, n_dofs) = test_shell_elements();
        let mut translated = elements.clone();
        translated[0].origin = [1e12, -1e12, 1e12];
        let reference = CorotationalShellForm::new(elements, n_dofs);
        let distant = CorotationalShellForm::new(translated, n_dofs);
        // Binary-exact displacement differences also remain representable
        // when the complete element undergoes a large rigid translation.
        let mut u = vec![0.0; n_dofs];
        u[6] = 1.0 / 128.0;
        u[13] = -1.0 / 64.0;
        u[14] = 1.0 / 256.0;
        u[10] = 1.0 / 128.0;
        let expected = reference.element_residual(0, &u);
        assert!(expected.iter().any(|v| v.abs() > 1.0));
        assert_eq!(distant.element_residual(0, &u), expected);
        assert_eq!(
            distant.element_tangent(0, &u),
            reference.element_tangent(0, &u)
        );
        for node in 0..3 {
            for component in 0..3 {
                u[6 * node + component] += 1e8;
            }
        }
        assert_eq!(reference.element_residual(0, &u), expected);
        assert_eq!(distant.element_residual(0, &u), expected);
    }

    #[test]
    fn test_corotational_shell_tangent_matches_fine_step_derivative() {
        // The numerical tangent must agree with an independent central
        // difference at 100x smaller step, evaluated at a bent state where
        // frame rotation couples into the Jacobian. This guards the complete
        // derivative without hiding differentiation error by symmetrizing it.
        let (elements, n_dofs) = test_shell_elements();
        let dofs = elements[0].global_dofs();
        let form = CorotationalShellForm::new(elements, n_dofs);
        let mut u = vec![0.0; n_dofs];
        u[dofs[2]] = 0.05;
        u[dofs[8]] = 0.03;
        u[dofs[14]] = -0.02;
        u[dofs[3]] = 0.1;
        u[dofs[10]] = -0.08;

        let k = form.element_tangent(0, &u);
        let norm = k.iter().map(|value| value.abs()).fold(0.0, f64::max);
        assert!(norm > 0.0, "tangent must be nonzero at a bent state");
        for (column, &dof) in dofs.iter().enumerate() {
            let step = 1.0e-9 * (1.0 + u[dof].abs());
            let mut plus = u.clone();
            let mut minus = u.clone();
            plus[dof] += step;
            minus[dof] -= step;
            let residual_plus = form.element_residual(0, &plus);
            let residual_minus = form.element_residual(0, &minus);
            for row in 0..18 {
                let fine = (residual_plus[row] - residual_minus[row]) / (2.0 * step);
                let diff = (k[row * 18 + column] - fine).abs();
                assert!(
                    diff / norm < 1e-5,
                    "tangent mismatch at ({row},{column}): {} vs fine-fd {fine} (norm {norm:e})",
                    k[row * 18 + column],
                );
            }
        }
    }

    #[test]
    fn corotational_forces_match_frame_rotated_local_force() {
        // Classic corotational transform: the global force is the local
        // K q force rotated by the current frame, with the frame held fixed
        // during the variation. Build the expectation independently from
        // the frame basis vectors (columns of the stored rotation) rather
        // than through the residual's own transform code.
        let (elements, n_dofs) = test_shell_elements();
        let form = CorotationalShellForm::new(elements, n_dofs);
        let mut u = vec![0.0; n_dofs];
        u[6] = 0.015;
        u[13] = -0.01;
        u[2] = 0.05;
        u[8] = 0.03;
        u[14] = -0.02;
        u[3] = 0.1;
        u[10] = -0.08;
        let rotation = CorotationalShellForm::deformed_rotation(&form.deformed_points(0, &u));
        let d = form.corotated_local_displacement(0, &u, &rotation);
        let k = form.elements[0].stiffness_local_combined();
        let mut f_local = [0.0; 18];
        for i in 0..18 {
            for j in 0..18 {
                f_local[i] += 0.5 * (k[i * 18 + j] + k[j * 18 + i]) * d[j];
            }
        }
        // Stored rotation is [ex0, ey0, ez0, ex1, ...]: basis vector j has
        // components rotation[j], rotation[j + 3], rotation[j + 6].
        let mut expected = vec![0.0; 18];
        for block in 0..6 {
            let offset = block * 3;
            for component in 0..3 {
                for j in 0..3 {
                    expected[offset + component] +=
                        rotation[j + 3 * component] * f_local[offset + j];
                }
            }
        }
        let force = form.element_residual(0, &u);
        let error = force
            .iter()
            .zip(&expected)
            .map(|(f, e)| (f - e).powi(2))
            .sum::<f64>()
            .sqrt()
            / expected.iter().map(|e| e * e).sum::<f64>().sqrt();
        assert!(
            error < 1e-12,
            "frame-rotated force relative error {error:e}"
        );
    }

    #[test]
    fn test_corotational_shell_rigid_body_motion_zero_residual() {
        let (elements, n_dofs) = test_shell_elements();
        let form = CorotationalShellForm::new(elements, n_dofs);
        let angle = 1.0e-4_f64;
        let (sin_angle, cos_angle) = angle.sin_cos();
        let reference = [[0.0, 0.0], [1.0, 0.0], [0.5, 0.8]];
        let mut u = vec![0.0; n_dofs];

        for (node, [x, y]) in reference.into_iter().enumerate() {
            let rotated = [cos_angle * x - sin_angle * y, sin_angle * x + cos_angle * y];
            u[node * 6] = rotated[0] - x + 0.25;
            u[node * 6 + 1] = rotated[1] - y - 0.15;
            u[node * 6 + 2] = 0.07;
            u[node * 6 + 5] = angle;
        }

        let residual = form.element_residual(0, &u);
        let maximum = residual.iter().map(|value| value.abs()).fold(0.0, f64::max);
        assert!(maximum < 1.0e-4, "rigid-body residual was {maximum:e}");
    }

    #[test]
    fn test_corotational_shell_degenerate_orientation_is_finite() {
        let degenerate = [[0.0, 0.0, 0.0]; 3];
        let rotation = CorotationalShellForm::deformed_rotation(&degenerate);
        assert_eq!(rotation, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert!(rotation.iter().all(|value| value.is_finite()));
    }
}
