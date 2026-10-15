//! Enhanced assumed strain bricks for bending and incompressibility.
//!
//! A nine-mode incompatible enhancement of the deformation gradient relieves
//! shear and volumetric locking on H8 bricks. Enhanced parameters equilibrate
//! locally per residual evaluation, so the element stays stateless; the
//! condensed tangent is a central-difference Jacobian of the residual.

// Rust guideline compliant 2026-02-21

use super::hex::{
    HexMesh, dense_solve, hex_jacobian, hex_quadrature, hex_shape_derivatives, physical_gradients,
};
use super::materials::HyperelasticMaterial;

/// EAS incompatible modes as natural-coordinate 3x3 matrices.
///
/// Modes enhance `F11/F22/F33` diagonally plus symmetric shear pairs, the
/// Q1/E9 pattern with zero mean over the reference cube.
fn enhancement_modes(natural: &[f64; 3]) -> [[f64; 9]; 9] {
    let (xi, eta, zeta) = (natural[0], natural[1], natural[2]);
    let mut modes = [[0.0; 9]; 9];
    modes[0][0] = xi;
    modes[1][4] = eta;
    modes[2][8] = zeta;
    modes[3][1] = xi;
    modes[3][3] = xi;
    modes[4][1] = eta;
    modes[4][3] = eta;
    modes[5][5] = eta;
    modes[5][7] = eta;
    modes[6][5] = zeta;
    modes[6][7] = zeta;
    modes[7][2] = xi;
    modes[7][6] = xi;
    modes[8][2] = zeta;
    modes[8][6] = zeta;
    modes
}

/// Row-major 3x3 product `C = A B`.
fn mat3(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut out = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            out[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
        }
    }
    out
}

/// Applies the isotropic 6x6 stiffness to a Voigt vector.
///
/// Ordering is `[exx, eyy, ezz, gxy, gyz, gzx]` with engineering
/// shears, matching the bench `linear_hex_stiffness` convention.
fn isotropic_c_times(lambda: f64, mu: f64, strain: &[f64; 6]) -> [f64; 6] {
    let lame2 = lambda + 2.0 * mu;
    [
        lame2 * strain[0] + lambda * (strain[1] + strain[2]),
        lame2 * strain[1] + lambda * (strain[0] + strain[2]),
        lame2 * strain[2] + lambda * (strain[0] + strain[1]),
        mu * strain[3],
        mu * strain[4],
        mu * strain[5],
    ]
}

/// Compatible stiffness, enhancement coupling, and enhancement Gramian.
///
/// Returns `(Kuu, Kua, D)` for one brick at the reference state: the
/// standard Voigt `B^T C B` stiffness, the 24x9 displacement-to-mode
/// coupling, and the 9x9 symmetric positive-definite mode Gramian.
/// Condensation `Kuu - Kua D^{-1} Kua^T` is the statically condensed
/// EAS stiffness; separating assembly from the solve keeps the patch
/// check (`D^{-1} Kua^T u = 0` under constant strain) testable.
fn eas_linear_system(
    lambda: f64,
    mu: f64,
    coords: &[[f64; 3]; 8],
) -> ([f64; 576], [[f64; 9]; 24], [f64; 81]) {
    assert!(
        lambda.is_finite() && mu.is_finite(),
        "EAS moduli must be finite"
    );
    let center = [0.0, 0.0, 0.0];
    let center_derivatives = hex_shape_derivatives(&center);
    let (_, j0_det, j0_inv) = hex_jacobian(coords, &center_derivatives);
    assert!(j0_det > 0.0, "EAS brick needs a positive centroid Jacobian");
    let mut kuu = [0.0; 576];
    let mut kua = [[0.0; 9]; 24];
    let mut d = [0.0; 81];
    for (natural, weight) in hex_quadrature() {
        let derivatives = hex_shape_derivatives(&natural);
        let (_, det, inv) = hex_jacobian(coords, &derivatives);
        assert!(det > 0.0, "EAS brick needs positive Gauss Jacobians");
        let grad = physical_gradients(&inv, &derivatives);
        let det_w = weight * det;
        // Compatible B rows per node, same layout as the bench
        // linear stiffness (engineering shears).
        let mut b = [[0.0; 24]; 6];
        for (node, g) in grad.iter().enumerate() {
            b[0][3 * node] = g[0];
            b[1][3 * node + 1] = g[1];
            b[2][3 * node + 2] = g[2];
            b[3][3 * node] = g[1];
            b[3][3 * node + 1] = g[0];
            b[4][3 * node + 1] = g[2];
            b[4][3 * node + 2] = g[1];
            b[5][3 * node] = g[2];
            b[5][3 * node + 2] = g[0];
        }
        // Push-forward at the reference state (F0 = I): G = (j0/j) M J0^{-1},
        // symmetrized into Voigt rows with engineering shears.
        let modes = enhancement_modes(&natural);
        let scale = j0_det / det;
        let mut g = [[0.0; 9]; 6];
        for (k, mode) in modes.iter().enumerate() {
            let pushed = mat3(mode, &j0_inv);
            let row = [
                pushed[0],
                pushed[4],
                pushed[8],
                pushed[1] + pushed[3],
                pushed[5] + pushed[7],
                pushed[2] + pushed[6],
            ];
            for (entry, value) in g.iter_mut().zip(row.iter()) {
                entry[k] = scale * value;
            }
        }
        for row in 0..24 {
            let column = [
                b[0][row], b[1][row], b[2][row], b[3][row], b[4][row], b[5][row],
            ];
            let cb = isotropic_c_times(lambda, mu, &column);
            for col in 0..24 {
                kuu[row * 24 + col] += det_w
                    * (b[0][col] * cb[0]
                        + b[1][col] * cb[1]
                        + b[2][col] * cb[2]
                        + b[3][col] * cb[3]
                        + b[4][col] * cb[4]
                        + b[5][col] * cb[5]);
            }
            for k in 0..9 {
                kua[row][k] += det_w
                    * (g[0][k] * cb[0]
                        + g[1][k] * cb[1]
                        + g[2][k] * cb[2]
                        + g[3][k] * cb[3]
                        + g[4][k] * cb[4]
                        + g[5][k] * cb[5]);
            }
        }
        for k in 0..9 {
            let column = [g[0][k], g[1][k], g[2][k], g[3][k], g[4][k], g[5][k]];
            let cg = isotropic_c_times(lambda, mu, &column);
            for l in 0..9 {
                d[k * 9 + l] += det_w
                    * (g[0][l] * cg[0]
                        + g[1][l] * cg[1]
                        + g[2][l] * cg[2]
                        + g[3][l] * cg[3]
                        + g[4][l] * cg[4]
                        + g[5][l] * cg[5]);
            }
        }
    }
    (kuu, kua, d)
}

/// Condensed EAS linear stiffness for one brick.
///
/// Statically condenses the nine incompatible modes at the reference
/// state, giving the locking-free small-strain stiffness that the
/// explicit loop precomputes once per element (same `dgemv` cost as
/// the locked linear stiffness). `coords` follows [`HEX_NODES`](super::hex::HEX_NODES)
/// order and rows use interleaved `[ux, uy, uz]` DOFs.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::physics::solid::nonlinear_solid::{eas_linear_stiffness, HexMesh};
///
/// let mesh = HexMesh::unit_cube(1.0);
/// let stiff = eas_linear_stiffness(1.0e6, 0.5e6, &mesh.element_coords(0));
/// // Rigid x-translation carries no force.
/// let shift = [1.0f64; 24];
/// for row in 0..24 {
///     let force: f64 = (0..24).map(|col| stiff[row * 24 + col] * shift[col]).sum();
///     assert!(force.abs() < 1.0e-6);
/// }
/// ```
///
/// # Panics
///
/// Panics for non-finite moduli, a non-positive Jacobian (inverted
/// or degenerate brick), or a singular enhancement Gramian (only on
/// degenerate geometry: `D` is positive definite otherwise).
#[must_use]
pub fn eas_linear_stiffness(lambda: f64, mu: f64, coords: &[[f64; 3]; 8]) -> [f64; 576] {
    let (mut kuu, kua, d) = eas_linear_system(lambda, mu, coords);
    // Correction Kua D^{-1} Kua^T, one 9x9 solve per DOF column.
    for col in 0..24 {
        let mut rhs = [0.0; 9];
        rhs.copy_from_slice(&kua[col]);
        let mut matrix = d.to_vec();
        dense_solve(&mut matrix, &mut rhs, 9);
        for row in 0..24 {
            let mut correction = 0.0;
            for k in 0..9 {
                correction += kua[row][k] * rhs[k];
            }
            kuu[row * 24 + col] -= correction;
        }
    }
    kuu
}

/// Double contraction `A : B` of two 3x3 matrices.
fn contract(a: &[f64], b: &[f64; 9]) -> f64 {
    let mut sum = 0.0;
    for i in 0..9 {
        sum += a[i] * b[i];
    }
    sum
}

/// Per-quadrature data shared by the local and global passes.
struct QuadraturePoint {
    /// Reference weight `w detJ0`.
    weight: f64,
    /// Physical shape gradients per node.
    grad: [[f64; 3]; 8],
    /// Compatible deformation gradient.
    f: [f64; 9],
    /// Push-forward enhancement per mode.
    g: [[f64; 9]; 9],
}

/// EAS hexahedral form with nine incompatible modes.
///
/// Implements [`NonlinearElementForm<f64>`](crate::forms::NonlinearElementForm)
/// with interleaved displacement DOFs (`[ux, uy, uz]` per node).
#[derive(Debug)]
pub struct EasHexForm<M: HyperelasticMaterial> {
    /// Base hyperelastic material evaluated at the enhanced gradient.
    pub material: M,
    /// Hexahedral mesh.
    pub mesh: HexMesh,
    /// Total number of global DOFs.
    pub n_dofs: usize,
    /// Finite-difference step for the tangent stiffness.
    pub fd_step: f64,
    /// Local Newton relative step tolerance.
    pub local_tolerance: f64,
    /// Local Newton iteration cap.
    pub local_max_iterations: usize,
}

impl<M: HyperelasticMaterial> EasHexForm<M> {
    /// Build an EAS form over a hex mesh.
    #[must_use]
    pub fn new(material: M, mesh: HexMesh) -> Self {
        assert_eq!(material.dim(), 3, "EAS bricks need a 3D material");
        let n_dofs = 3 * mesh.num_nodes();
        Self {
            material,
            mesh,
            n_dofs,
            fd_step: 1.0e-8,
            local_tolerance: 1.0e-12,
            local_max_iterations: 25,
        }
    }

    /// Global DOF indices for one element.
    fn element_dofs(&self, elem_idx: usize) -> [usize; 24] {
        let mut dofs = [0_usize; 24];
        for a in 0..8 {
            let node = self.mesh.conn[elem_idx][a];
            dofs[a * 3] = node * 3;
            dofs[a * 3 + 1] = node * 3 + 1;
            dofs[a * 3 + 2] = node * 3 + 2;
        }
        dofs
    }

    /// Compatible deformation gradient from nodal displacements.
    fn compatible_f(&self, dofs: &[usize; 24], u: &[f64], grad: &[[f64; 3]; 8]) -> [f64; 9] {
        let mut f = [0.0; 9];
        for i in 0..3 {
            f[i * 3 + i] = 1.0;
        }
        for a in 0..8 {
            for i in 0..3 {
                let u_a = u[dofs[a * 3 + i]];
                for col in 0..3 {
                    f[i * 3 + col] += u_a * grad[a][col];
                }
            }
        }
        f
    }

    /// Quadrature data with push-forward enhancement operators.
    fn quadrature_data(&self, elem_idx: usize, u: &[f64]) -> [QuadraturePoint; 8] {
        let coords = self.mesh.element_coords(elem_idx);
        let dofs = self.element_dofs(elem_idx);
        // Centroid map and deformation gradient drive the push-forward.
        let center = [0.0, 0.0, 0.0];
        let center_derivatives = hex_shape_derivatives(&center);
        let (_, j0_det, j0_inv) = hex_jacobian(&coords, &center_derivatives);
        let center_grad = physical_gradients(&j0_inv, &center_derivatives);
        let f0 = self.compatible_f(&dofs, u, &center_grad);

        let quadrature = hex_quadrature();
        std::array::from_fn(|qp| {
            let (natural, weight) = quadrature[qp];
            let derivatives = hex_shape_derivatives(&natural);
            let (_, det, inv) = hex_jacobian(&coords, &derivatives);
            let grad = physical_gradients(&inv, &derivatives);
            let f = self.compatible_f(&dofs, u, &grad);
            let modes = enhancement_modes(&natural);
            let scale = j0_det / det;
            let mut g = [[0.0; 9]; 9];
            for k in 0..9 {
                // G = (j0 / j) F0 M J0^{-1}.
                let pushed = mat3(&mat3(&f0, &modes[k]), &j0_inv);
                for i in 0..9 {
                    g[k][i] = scale * pushed[i];
                }
            }
            QuadraturePoint {
                weight: weight * det,
                grad,
                f,
                g,
            }
        })
    }

    /// Enhanced deformation gradient at one quadrature point.
    fn enhanced_f(&self, point: &QuadraturePoint, alpha: &[f64; 9]) -> [f64; 9] {
        let mut f = point.f;
        for (k, mode) in point.g.iter().enumerate() {
            for (entry, increment) in f.iter_mut().zip(mode.iter()) {
                *entry += alpha[k] * increment;
            }
        }
        f
    }

    /// Local Newton solve for the enhanced parameters.
    ///
    /// Stops on the Newton step size, which tracks the parameter error
    /// without depending on the residual scale (`local_tolerance` is a
    /// relative step tolerance).
    fn equilibrate(&self, points: &[QuadraturePoint; 8]) -> [f64; 9] {
        let mut alpha = [0.0; 9];
        for _ in 0..self.local_max_iterations {
            let mut residual = [0.0; 9];
            let mut stiffness = [0.0; 81];
            for point in points {
                let f_enh = self.enhanced_f(point, &alpha);
                let p = self.material.stress(&f_enh);
                let a = self.material.tangent(&f_enh);
                for (entry, mode) in residual.iter_mut().zip(point.g.iter()) {
                    *entry += point.weight * contract(&p, mode);
                }
                // D_kl = sum G_k : A : G_l over quadrature points.
                for k in 0..9 {
                    for l in 0..9 {
                        let mut d = 0.0;
                        for ij in 0..9 {
                            for kl in 0..9 {
                                d += point.g[k][ij] * a[ij * 9 + kl] * point.g[l][kl];
                            }
                        }
                        stiffness[k * 9 + l] += point.weight * d;
                    }
                }
            }
            let mut step = residual;
            for value in &mut step {
                *value = -*value;
            }
            let mut matrix = stiffness.to_vec();
            dense_solve(&mut matrix, &mut step, 9);
            let mut worst: f64 = 0.0;
            let mut scale: f64 = 1.0;
            for k in 0..9 {
                alpha[k] += step[k];
                worst = worst.max(step[k].abs());
                scale = scale.max(alpha[k].abs());
            }
            if worst < self.local_tolerance * scale {
                break;
            }
        }
        alpha
    }

    /// Internal forces for fixed enhanced parameters.
    fn forces_for_alpha(&self, points: &[QuadraturePoint; 8], alpha: &[f64; 9]) -> [f64; 24] {
        let mut residual = [0.0; 24];
        for point in points {
            let f_enh = self.enhanced_f(point, alpha);
            let p = self.material.stress(&f_enh);
            for a in 0..8 {
                for i in 0..3 {
                    let mut r = 0.0;
                    for col in 0..3 {
                        r += p[i * 3 + col] * point.grad[a][col];
                    }
                    residual[a * 3 + i] += r * point.weight;
                }
            }
        }
        residual
    }

    /// Internal force residual with locally equilibrated enhancement.
    fn internal_forces(&self, elem_idx: usize, u: &[f64]) -> [f64; 24] {
        let points = self.quadrature_data(elem_idx, u);
        let alpha = self.equilibrate(&points);
        self.forces_for_alpha(&points, &alpha)
    }
}

impl<M: HyperelasticMaterial> crate::forms::NonlinearElementForm<f64> for EasHexForm<M> {
    fn num_elements(&self) -> usize {
        self.mesh.conn.len()
    }

    fn num_local_dofs(&self, _elem_idx: usize) -> usize {
        24
    }

    fn local_to_global(&self, elem_idx: usize) -> Vec<usize> {
        self.element_dofs(elem_idx).to_vec()
    }

    fn num_global_dofs(&self) -> usize {
        self.n_dofs
    }

    fn element_residual(&self, elem_idx: usize, global_solution: &[f64]) -> Vec<f64> {
        self.internal_forces(elem_idx, global_solution).to_vec()
    }

    fn element_tangent(&self, elem_idx: usize, global_solution: &[f64]) -> Vec<f64> {
        let dofs = self.element_dofs(elem_idx);
        let step = self.fd_step;
        let mut tangent = vec![0.0; 576];
        let mut perturbed = global_solution.to_vec();
        for (col, &dof) in dofs.iter().enumerate() {
            let base = global_solution[dof];
            let h = step * (1.0 + base.abs());
            perturbed[dof] = base + h;
            let plus = self.internal_forces(elem_idx, &perturbed);
            perturbed[dof] = base - h;
            let minus = self.internal_forces(elem_idx, &perturbed);
            perturbed[dof] = base;
            for row in 0..24 {
                tangent[row * 24 + col] = (plus[row] - minus[row]) / (2.0 * h);
            }
        }
        tangent
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forms::NonlinearElementForm;
    use crate::physics::solid::nonlinear_solid::NeoHookean;

    fn cube_form() -> EasHexForm<NeoHookean> {
        let material = NeoHookean::from_young(1.0e6, 0.3, 3);
        EasHexForm::new(material, HexMesh::unit_cube(1.0))
    }

    #[test]
    fn homogeneous_deformation_needs_no_enhancement() {
        let form = cube_form();
        let mut u = vec![0.0; 24];
        for node in 0..8 {
            let coords = form.mesh.coords[node];
            u[node * 3] = 0.02 * coords[0] + 0.01 * coords[1];
            u[node * 3 + 1] = -0.01 * coords[0] + 0.03 * coords[2];
            u[node * 3 + 2] = 0.01 * coords[1] - 0.02 * coords[2];
        }
        let points = form.quadrature_data(0, &u);
        let alpha = form.equilibrate(&points);
        for (k, &a) in alpha.iter().enumerate() {
            assert!(a.abs() < 1.0e-12, "alpha[{k}] = {a}");
        }
    }

    #[test]
    fn bending_relaxes_parasitic_shear() {
        let form = cube_form();
        // Pure-bending displacement: u_x varies quadratically through height.
        let mut u = vec![0.0; 24];
        for node in 0..8 {
            let [x, y, _] = form.mesh.coords[node];
            u[node * 3] = 0.05 * (x * x - y * y);
            u[node * 3 + 1] = -0.1 * x * y;
        }
        let eas = form.element_residual(0, &u);
        let base = crate::physics::solid::nonlinear_solid::q1_element_residual(
            &form.material,
            &form.mesh,
            0,
            &u,
        );
        let eas_norm: f64 = eas.iter().map(|r| r * r).sum::<f64>().sqrt();
        let base_norm: f64 = base.iter().map(|r| r * r).sum::<f64>().sqrt();
        assert!(
            eas_norm < base_norm,
            "EAS {eas_norm} should relax Q1 {base_norm}"
        );
    }

    #[test]
    fn linear_cure_is_symmetric_and_self_equilibrated() {
        let mesh = HexMesh::brick(1, 1, 1, [0.0, 0.0, 0.0], [0.1, 0.025, 0.025]);
        let stiff = eas_linear_stiffness(115.4e9, 76.9e9, &mesh.element_coords(0));
        let scale: f64 = stiff.iter().map(|k| k.abs()).sum::<f64>().max(1.0e-300);
        for row in 0..24 {
            for col in 0..24 {
                let skew = (stiff[row * 24 + col] - stiff[col * 24 + row]).abs() / scale;
                assert!(skew < 1.0e-14, "skew[{row},{col}] = {skew:.3e}");
            }
        }
        // Rigid translations and infinitesimal rotations carry no force.
        // Stiffness rows run in connectivity-slot order, so the modes
        // sample the slot-ordered element coordinates.
        let slots = mesh.element_coords(0);
        let mut modes = [[0.0; 24]; 6];
        for node in 0..8 {
            let [x, y, z] = slots[node];
            modes[0][3 * node] = 1.0;
            modes[1][3 * node + 1] = 1.0;
            modes[2][3 * node + 2] = 1.0;
            modes[3][3 * node + 1] = -z;
            modes[3][3 * node + 2] = y;
            modes[4][3 * node] = z;
            modes[4][3 * node + 2] = -x;
            modes[5][3 * node] = -y;
            modes[5][3 * node + 1] = x;
        }
        for (m, mode) in modes.iter().enumerate() {
            for row in 0..24 {
                let force: f64 = (0..24).map(|col| stiff[row * 24 + col] * mode[col]).sum();
                assert!(
                    force.abs() / scale < 1.0e-13,
                    "rigid mode {m} row {row}: {force:.3e}"
                );
            }
        }
    }

    #[test]
    fn linear_cure_enhancement_rests_under_constant_strain() {
        // Patch check on a deck-shaped brick: constant strain needs no
        // enhancement, so the condensed parameters vanish.
        let mesh = HexMesh::brick(1, 1, 1, [0.0, 0.0, 0.0], [0.1, 0.025, 0.025]);
        let (_, kua, d) = eas_linear_system(115.4e9, 76.9e9, &mesh.element_coords(0));
        let mut u = [0.0; 24];
        let slots = mesh.element_coords(0);
        for node in 0..8 {
            let [x, y, z] = slots[node];
            u[3 * node] = 0.02 * x + 0.01 * y + 0.005 * z;
            u[3 * node + 1] = 0.01 * x + 0.03 * y - 0.004 * z;
            u[3 * node + 2] = 0.005 * x - 0.004 * y + 0.025 * z;
        }
        let mut rhs = [0.0; 9];
        for k in 0..9 {
            rhs[k] = (0..24).map(|row| kua[row][k] * u[row]).sum();
        }
        let mut matrix = d.to_vec();
        dense_solve(&mut matrix, &mut rhs, 9);
        let alpha: f64 = rhs.iter().map(|a| a.abs()).sum();
        // Enhancement strains sit at roundoff next to the imposed state.
        let state: f64 = u.iter().map(|v| v.abs()).sum::<f64>().max(1.0e-300);
        assert!(
            alpha / state < 1.0e-12,
            "alpha {alpha:.3e} over state {state:.3e}"
        );
    }

    #[test]
    fn linear_cure_matches_nonlinear_eas_at_small_strain() {
        let material = NeoHookean::from_young(1.0e6, 0.3, 3);
        let mesh = HexMesh::brick(1, 1, 1, [0.0, 0.0, 0.0], [0.1, 0.025, 0.025]);
        let form = EasHexForm::new(material, mesh.clone());
        let stiff = eas_linear_stiffness(
            form.material.lambda,
            form.material.mu,
            &mesh.element_coords(0),
        );
        // Deterministic small state: strains ~1e-7, so the Neo-Hookean
        // second-order remainder sits ~1e-14 below the linear term.
        // Slot order feeds the stiffness rows; global order feeds the form.
        let slot_u: Vec<f64> = (0..24)
            .map(|i| 1.0e-8 * ((i * 37 % 11) as f64 - 5.0))
            .collect();
        let mut u = vec![0.0; 3 * mesh.num_nodes()];
        for (slot, node) in mesh.conn[0].iter().enumerate() {
            u[3 * node] = slot_u[3 * slot];
            u[3 * node + 1] = slot_u[3 * slot + 1];
            u[3 * node + 2] = slot_u[3 * slot + 2];
        }
        let expected = form.element_residual(0, &u);
        let scale: f64 = expected.iter().map(|r| r.abs()).sum::<f64>().max(1.0e-300);
        for row in 0..24 {
            let linear: f64 = (0..24).map(|col| stiff[row * 24 + col] * slot_u[col]).sum();
            assert!(
                (linear - expected[row]).abs() / scale < 1.0e-6,
                "row {row}: linear {linear:.6e} vs EAS {:.6e}",
                expected[row]
            );
        }
    }

    #[test]
    fn linear_cure_unlocks_cantilever_tip_deflection() {
        // Static replica of the C3 cantilever deck: 10x2x2 bricks over
        // 1.0 x 0.05 x 0.05 m, steel, root clamped, 100 N tip load in z.
        let mesh = HexMesh::brick(10, 2, 2, [0.0, -0.025, -0.025], [1.0, 0.025, 0.025]);
        assert_eq!(mesh.num_nodes(), 99);
        assert_eq!(mesh.conn.len(), 40);
        let young = 200.0e9;
        let poisson = 0.3;
        let lambda = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
        let mu = young / (2.0 * (1.0 + poisson));
        let locked: Vec<[f64; 576]> = (0..mesh.conn.len())
            .map(|elem| q1_voigt_stiffness(lambda, mu, &mesh.element_coords(elem)))
            .collect();
        let cured: Vec<[f64; 576]> = (0..mesh.conn.len())
            .map(|elem| eas_linear_stiffness(lambda, mu, &mesh.element_coords(elem)))
            .collect();
        // Tip load matches the deck CLOAD: 11.111 N on each of the 9
        // tip-face nodes, totaling 100 N.
        let mut load = vec![0.0; 3 * mesh.num_nodes()];
        let mut tip_center = 0;
        for (node, coords) in mesh.coords.iter().enumerate() {
            if (coords[0] - 1.0).abs() < 1.0e-12 {
                load[3 * node + 2] += 100.0 / 9.0;
                if coords[1].abs() < 1.0e-12 && coords[2].abs() < 1.0e-12 {
                    tip_center = node;
                }
            }
        }
        let pinned: Vec<bool> = mesh
            .coords
            .iter()
            .map(|coords| coords[0].abs() < 1.0e-12)
            .collect();
        let locked_tip = static_tip(&mesh, &locked, &load, &pinned, tip_center);
        let cured_tip = static_tip(&mesh, &cured, &load, &pinned, tip_center);
        // Beam theory FL^3/3EI = 3.2e-4 m, the deck probe value.
        let theory = 100.0 * 1.0f64.powi(3) / (3.0 * young * 0.05 * 0.05f64.powi(3) / 12.0);
        assert!((theory - 3.2e-4).abs() < 1.0e-12, "theory {theory:.6e}");
        assert!(
            locked_tip / theory < 0.5,
            "locked Q1 tip {locked_tip:.6e} should show shear locking"
        );
        assert!(
            (cured_tip - theory).abs() / theory < 0.03,
            "cured tip {cured_tip:.6e} vs theory {theory:.6e}"
        );
    }

    /// Locked Voigt `B^T C B` stiffness, the explicit loop's linear reference.
    fn q1_voigt_stiffness(lambda: f64, mu: f64, coords: &[[f64; 3]; 8]) -> [f64; 576] {
        let mut stiff = [0.0; 576];
        for (natural, weight) in hex_quadrature() {
            let derivatives = hex_shape_derivatives(&natural);
            let (_, det, inv) = hex_jacobian(coords, &derivatives);
            let grad = physical_gradients(&inv, &derivatives);
            let det_w = weight * det;
            let mut b = [[0.0; 24]; 6];
            for (node, g) in grad.iter().enumerate() {
                b[0][3 * node] = g[0];
                b[1][3 * node + 1] = g[1];
                b[2][3 * node + 2] = g[2];
                b[3][3 * node] = g[1];
                b[3][3 * node + 1] = g[0];
                b[4][3 * node + 1] = g[2];
                b[4][3 * node + 2] = g[1];
                b[5][3 * node] = g[2];
                b[5][3 * node + 2] = g[0];
            }
            for row in 0..24 {
                let column = [
                    b[0][row], b[1][row], b[2][row], b[3][row], b[4][row], b[5][row],
                ];
                let cb = isotropic_c_times(lambda, mu, &column);
                for col in 0..24 {
                    stiff[row * 24 + col] += det_w
                        * (b[0][col] * cb[0]
                            + b[1][col] * cb[1]
                            + b[2][col] * cb[2]
                            + b[3][col] * cb[3]
                            + b[4][col] * cb[4]
                            + b[5][col] * cb[5]);
                }
            }
        }
        stiff
    }

    /// Solves the clamped static problem and returns the tip-center z displacement.
    fn static_tip(
        mesh: &HexMesh,
        elem_stiff: &[[f64; 576]],
        load: &[f64],
        pinned: &[bool],
        tip_center: usize,
    ) -> f64 {
        let dofs = 3 * mesh.num_nodes();
        let mut matrix = vec![0.0; dofs * dofs];
        for (elem, conn) in mesh.conn.iter().enumerate() {
            let stiff = &elem_stiff[elem];
            let mut dof = [0usize; 24];
            for (slot, node) in conn.iter().enumerate() {
                dof[3 * slot] = 3 * node;
                dof[3 * slot + 1] = 3 * node + 1;
                dof[3 * slot + 2] = 3 * node + 2;
            }
            for row in 0..24 {
                for col in 0..24 {
                    matrix[dof[row] * dofs + dof[col]] += stiff[row * 24 + col];
                }
            }
        }
        let mut rhs = load.to_vec();
        for (node, pinned) in pinned.iter().enumerate() {
            if !pinned {
                continue;
            }
            for axis in 0..3 {
                let row = 3 * node + axis;
                for col in 0..dofs {
                    matrix[row * dofs + col] = 0.0;
                    matrix[col * dofs + row] = 0.0;
                }
                matrix[row * dofs + row] = 1.0;
                rhs[row] = 0.0;
            }
        }
        dense_solve(&mut matrix, &mut rhs, dofs);
        rhs[3 * tip_center + 2]
    }

    #[test]
    fn tangent_is_consistent_with_residual() {
        let form = cube_form();
        let state: Vec<f64> = (0..24)
            .map(|i| 0.008 * ((i * 37 % 11) as f64 - 5.0))
            .collect();
        let direction: Vec<f64> = (0..24)
            .map(|i| 0.04 * ((i * 53 % 7) as f64 - 3.0))
            .collect();
        let report = crate::validation::check_tangent_consistency(
            |current| form.element_residual(0, current),
            |current, tangent_direction| {
                let tangent = form.element_tangent(0, current);
                (0..current.len())
                    .map(|row| {
                        (0..current.len())
                            .map(|column| tangent[row * 24 + column] * tangent_direction[column])
                            .sum()
                    })
                    .collect()
            },
            &state,
            &direction,
            &[1.0e-3, 5.0e-4, 2.5e-4],
        )
        .expect("eas residual and tangent dimensions must agree");
        assert!(report.observed_order > 1.8, "{report:?}");
    }
}
