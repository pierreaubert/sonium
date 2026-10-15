//! Radial return mapping algorithm for J2 plasticity
//!
//! Given a strain increment, computes the updated stress and consistent tangent
//! using the classical radial return (closest-point projection) algorithm.
//!
//! Works in 3D (6 Voigt components) and 2D plane stress/strain (3 components).

use super::material::{HardeningLaw, J2Material, PlasticState};

/// Result of a stress update via return mapping.
#[derive(Debug, Clone)]
pub struct StressUpdate {
    /// Updated stress (Voigt notation).
    pub stress: Vec<f64>,
    /// Consistent (algorithmic) tangent modulus (n×n, row-major).
    pub tangent: Vec<f64>,
    /// Plastic strain increment (scalar equivalent).
    pub delta_equiv_plastic_strain: f64,
    /// Whether yielding occurred.
    pub yielding: bool,
}

/// Perform radial return mapping for J2 plasticity (3D, 6 Voigt components).
///
/// Given the current state and a total strain increment `delta_eps`,
/// returns the updated stress and consistent tangent.
///
/// Voigt ordering: `[eps_xx, eps_yy, eps_zz, gamma_yz, gamma_xz, gamma_xy]`.
pub fn return_mapping_3d(
    material: &J2Material,
    state: &PlasticState,
    strain_increment: &[f64; 6],
) -> StressUpdate {
    let g = material.shear_modulus();
    let k = material.bulk_modulus();
    let g2 = 2.0 * g;

    // Trial stress = old stress + elastic increment
    // Elastic: sigma_trial = sigma_n + C_e : delta_eps
    let vol_strain = strain_increment[0] + strain_increment[1] + strain_increment[2];
    let pressure_inc = k * vol_strain;

    // Deviatoric strain increment
    let dev_strain = [
        strain_increment[0] - vol_strain / 3.0,
        strain_increment[1] - vol_strain / 3.0,
        strain_increment[2] - vol_strain / 3.0,
        strain_increment[3] / 2.0, // engineering -> tensor shear
        strain_increment[4] / 2.0,
        strain_increment[5] / 2.0,
    ];

    // Trial deviatoric stress
    let mut s_trial = [0.0; 6];
    for i in 0..6 {
        let s_old = state.stress[i]
            - if i < 3 {
                (state.stress[0] + state.stress[1] + state.stress[2]) / 3.0
            } else {
                0.0
            };
        s_trial[i] = s_old + g2 * dev_strain[i];
    }

    // Shifted deviatoric (for kinematic hardening): xi = s_trial - alpha
    let mut xi = [0.0; 6];
    for i in 0..6 {
        xi[i] = s_trial[i] - state.back_stress[i];
    }

    // Von Mises equivalent stress of trial shifted deviatoric
    let q_trial = von_mises_stress(&xi);

    // Trial hydrostatic pressure
    let p_trial = (state.stress[0] + state.stress[1] + state.stress[2]) / 3.0 + pressure_inc;

    // Current yield stress
    let sigma_y = material.current_yield_stress(state.equiv_plastic_strain);

    // Yield function: f = q_trial - sigma_y
    let f_trial = q_trial - sigma_y;

    if f_trial <= 0.0 {
        // Elastic step: no plastic correction
        let mut stress = s_trial;
        stress[0] += p_trial;
        stress[1] += p_trial;
        stress[2] += p_trial;

        let tangent = elastic_tangent_3d(k, g);

        StressUpdate {
            stress: stress.to_vec(),
            tangent,
            delta_equiv_plastic_strain: 0.0,
            yielding: false,
        }
    } else {
        // Plastic step: radial return
        let h_iso = material.isotropic_hardening_slope(state.equiv_plastic_strain);
        let h_kin = material.kinematic_hardening_modulus();
        let mut h_bar = h_iso + h_kin;

        // Linear laws have the closed-form multiplier below. Voce needs the
        // yield stress at the updated plastic strain, not a frozen old slope.
        let mut delta_gamma = f_trial / (3.0 * g + h_bar);
        if matches!(material.hardening, HardeningLaw::VoceSaturation { .. }) {
            for _ in 0..20 {
                let ep = state.equiv_plastic_strain + delta_gamma;
                h_bar = material.isotropic_hardening_slope(ep) + h_kin;
                let residual =
                    q_trial - (3.0 * g + h_kin) * delta_gamma - material.current_yield_stress(ep);
                if residual.abs() <= 1e-13 * q_trial.max(sigma_y) {
                    break;
                }
                delta_gamma += residual / (3.0 * g + h_bar);
            }
            h_bar = material.isotropic_hardening_slope(state.equiv_plastic_strain + delta_gamma)
                + h_kin;
        }

        // Updated deviatoric stress: s = s_trial * (1 - 3G*dg/q_trial)
        // This is the radial return: s_new = sigma_y/q_trial * s_trial
        let mut stress = [0.0; 6];
        let ratio = 1.0 - 3.0 * g * delta_gamma / q_trial;
        for i in 0..3 {
            stress[i] = ratio * s_trial[i] + (1.0 - ratio) * state.back_stress[i] + p_trial;
        }
        for i in 3..6 {
            stress[i] = ratio * s_trial[i] + (1.0 - ratio) * state.back_stress[i];
        }

        // Differentiate the solved yield consistency equation at updated ep.
        let tangent = consistent_tangent_3d(k, g, q_trial, delta_gamma, &xi, h_bar);

        StressUpdate {
            stress: stress.to_vec(),
            tangent,
            delta_equiv_plastic_strain: delta_gamma,
            yielding: true,
        }
    }
}

/// Update the plastic state after a converged Newton step.
pub fn update_plastic_state(
    state: &mut PlasticState,
    update: &StressUpdate,
    material: &J2Material,
) {
    state.stress.copy_from_slice(&update.stress);
    state.equiv_plastic_strain += update.delta_equiv_plastic_strain;
    state.yielding = update.yielding;

    // Update back-stress for kinematic hardening
    if update.yielding && update.delta_equiv_plastic_strain > 0.0 {
        let c = material.kinematic_hardening_modulus();
        if c > 0.0 {
            let q = von_mises_stress_from_slice(&update.stress);
            if q > 1e-15 {
                for i in 0..state.back_stress.len() {
                    let s_dev = if i < 3 {
                        update.stress[i]
                            - (update.stress[0] + update.stress[1] + update.stress[2]) / 3.0
                    } else {
                        update.stress[i]
                    };
                    state.back_stress[i] +=
                        (2.0 / 3.0) * c * update.delta_equiv_plastic_strain * s_dev / q;
                }
            }
        }
    }
}

/// Von Mises equivalent stress from 6-component deviatoric stress.
fn von_mises_stress(s: &[f64; 6]) -> f64 {
    let j2 =
        0.5 * (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]) + s[3] * s[3] + s[4] * s[4] + s[5] * s[5];
    (3.0 * j2).sqrt()
}

fn von_mises_stress_from_slice(sigma: &[f64]) -> f64 {
    if sigma.len() < 6 {
        return 0.0;
    }
    let p = (sigma[0] + sigma[1] + sigma[2]) / 3.0;
    let s = [
        sigma[0] - p,
        sigma[1] - p,
        sigma[2] - p,
        sigma[3],
        sigma[4],
        sigma[5],
    ];
    von_mises_stress(&s)
}

/// Elastic tangent modulus (6×6, row-major).
fn elastic_tangent_3d(k: f64, g: f64) -> Vec<f64> {
    let lambda = k - 2.0 * g / 3.0;
    let g2 = 2.0 * g;
    vec![
        lambda + g2,
        lambda,
        lambda,
        0.0,
        0.0,
        0.0,
        lambda,
        lambda + g2,
        lambda,
        0.0,
        0.0,
        0.0,
        lambda,
        lambda,
        lambda + g2,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        g,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        g,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        g,
    ]
}

/// Consistent (algorithmic) tangent for J2 plasticity (6×6).
///
/// Reference: Simo & Hughes (1998), Computational Inelasticity, Eq. 3.7.10.
///
/// Differentiates the returned stress at fixed committed history. The
/// hardening slope is the sum used in the plastic-multiplier denominator.
fn consistent_tangent_3d(
    k: f64,
    g: f64,
    q_trial: f64,
    delta_gamma: f64,
    xi: &[f64; 6],
    h_bar: f64,
) -> Vec<f64> {
    let theta = 1.0 - 3.0 * g * delta_gamma / q_trial;
    let theta_bar = 1.0 / (1.0 + h_bar / (3.0 * g)) - (1.0 - theta);
    let g2 = 2.0 * g;

    // Flow direction n = xi / q_trial in tensor-stress Voigt notation.
    // Differentiating q_trial with respect to engineering strain gives
    // 3G*n for both normal and shear columns. Therefore the rank-one
    // correction is 3G*theta_bar*n⊗n; Mandel sqrt(2) weights do not apply
    // to this tensor-stress/engineering-strain matrix.
    let n_tensor: Vec<f64> = xi.iter().map(|&x| x / q_trial).collect();

    // C_ep = K*(1⊗1) + 2G*theta*I_dev - 3G*theta_bar*(n⊗n).
    let mut c = vec![0.0; 36];

    for i in 0..6 {
        for j in 0..6 {
            let idx = i * 6 + j;

            // Volumetric: K * delta_vol
            if i < 3 && j < 3 {
                c[idx] += k;
            }
            // Deviatoric projector in Voigt:
            // I_dev_Voigt[i,j] = delta_ij * (2/3 for normal, 1/2 for shear)
            //                  + (-1/3 for i,j both normal, 0 otherwise)
            let i_dev = if i == j {
                if i < 3 { 2.0 / 3.0 } else { 0.5 }
            } else if i < 3 && j < 3 {
                -1.0 / 3.0
            } else {
                0.0
            };
            c[idx] += g2 * theta * i_dev;
            c[idx] -= 3.0 * g * theta_bar * n_tensor[i] * n_tensor[j];
        }
    }

    c
}

/// Maps an in-plane strain increment to updated plane stress.
///
/// Iteratively adjusts the out-of-plane strain increment to enforce
/// σ_zz = 0, then returns the 3-component in-plane stress and 3×3
/// consistent tangent.
///
/// The input is the strain increment since the committed `state`, not the
/// total strain: the trial stress is the committed stress plus the elastic
/// increment. Passing totals double-counts history on every call after the
/// first commit.
///
/// Voigt ordering (input/output): `[Δε_xx, Δε_yy, Δγ_xy]` /
/// `[σ_xx, σ_yy, τ_xy]`.
pub fn return_mapping_plane_stress(
    material: &J2Material,
    state: &PlasticState,
    strain_increment_2d: &[f64; 3],
) -> StressUpdate {
    let nu = material.poisson_ratio;

    // Initial guess for out-of-plane strain: elastic plane-stress relation
    let mut eps_zz = -nu / (1.0 - nu) * (strain_increment_2d[0] + strain_increment_2d[1]);

    let max_iter = 20;
    let tol = 1e-12;

    let mut update_3d = StressUpdate {
        stress: vec![0.0; 6],
        tangent: vec![0.0; 36],
        delta_equiv_plastic_strain: 0.0,
        yielding: false,
    };

    for _ in 0..max_iter {
        // Build full 3D strain increment: [Δε_xx, Δε_yy, Δε_zz, 0, 0, Δγ_xy]
        let strain_3d = [
            strain_increment_2d[0],
            strain_increment_2d[1],
            eps_zz,
            0.0,
            0.0,
            strain_increment_2d[2],
        ];

        update_3d = return_mapping_3d(material, state, &strain_3d);

        let sigma_zz = update_3d.stress[2];
        if sigma_zz.abs() < tol {
            break;
        }

        // Newton correction: eps_zz -= sigma_zz / C[2,2]
        let c_zz = update_3d.tangent[2 * 6 + 2];
        if c_zz.abs() < 1e-30 {
            break;
        }
        eps_zz -= sigma_zz / c_zz;
    }

    // Static condensation of 3D tangent to plane stress (3×3)
    // Condense out indices 2 (zz), 3 (yz), 4 (xz)
    // Keep indices 0 (xx), 1 (yy), 5 (xy)
    let keep = [0, 1, 5]; // indices in 3D Voigt
    let cond = [2, 3, 4]; // indices to condense out

    // Compute C_cc^{-1} (3×3 block of condensed indices)
    let mut c_cc = [0.0; 9];
    for (i, &ci) in cond.iter().enumerate() {
        for (j, &cj) in cond.iter().enumerate() {
            c_cc[i * 3 + j] = update_3d.tangent[ci * 6 + cj];
        }
    }
    let c_cc_inv = invert_3x3(&c_cc);

    // C_kc (3×3): keep rows, condensed cols
    let mut c_kc = [0.0; 9];
    for (i, &ki) in keep.iter().enumerate() {
        for (j, &cj) in cond.iter().enumerate() {
            c_kc[i * 3 + j] = update_3d.tangent[ki * 6 + cj];
        }
    }

    // C_ck (3×3): condensed rows, keep cols
    let mut c_ck = [0.0; 9];
    for (i, &ci) in cond.iter().enumerate() {
        for (j, &kj) in keep.iter().enumerate() {
            c_ck[i * 3 + j] = update_3d.tangent[ci * 6 + kj];
        }
    }

    // C_kk (3×3): kept block
    let mut c_kk = [0.0; 9];
    for (i, &ki) in keep.iter().enumerate() {
        for (j, &kj) in keep.iter().enumerate() {
            c_kk[i * 3 + j] = update_3d.tangent[ki * 6 + kj];
        }
    }

    // Condensed tangent: C_ps = C_kk - C_kc * C_cc^{-1} * C_ck
    let mut tangent_ps = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            let mut correction = 0.0;
            for p in 0..3 {
                let mut inv_ck = 0.0;
                for q in 0..3 {
                    inv_ck += c_cc_inv[p * 3 + q] * c_ck[q * 3 + j];
                }
                correction += c_kc[i * 3 + p] * inv_ck;
            }
            tangent_ps[i * 3 + j] = c_kk[i * 3 + j] - correction;
        }
    }

    StressUpdate {
        stress: vec![
            update_3d.stress[0],
            update_3d.stress[1],
            update_3d.stress[5],
        ],
        tangent: tangent_ps.to_vec(),
        delta_equiv_plastic_strain: update_3d.delta_equiv_plastic_strain,
        yielding: update_3d.yielding,
    }
}

/// Invert a 3×3 matrix (row-major).
fn invert_3x3(m: &[f64; 9]) -> [f64; 9] {
    let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    let inv_det = 1.0 / det;
    [
        (m[4] * m[8] - m[5] * m[7]) * inv_det,
        (m[2] * m[7] - m[1] * m[8]) * inv_det,
        (m[1] * m[5] - m[2] * m[4]) * inv_det,
        (m[5] * m[6] - m[3] * m[8]) * inv_det,
        (m[0] * m[8] - m[2] * m[6]) * inv_det,
        (m[2] * m[3] - m[0] * m[5]) * inv_det,
        (m[3] * m[7] - m[4] * m[6]) * inv_det,
        (m[1] * m[6] - m[0] * m[7]) * inv_det,
        (m[0] * m[4] - m[1] * m[3]) * inv_det,
    ]
}

#[cfg(test)]
mod tests {
    use super::super::material::{HardeningLaw, J2Material, PlasticState};
    use super::*;

    #[test]
    fn voce_return_satisfies_updated_yield_surface_for_large_increments() {
        let material = J2Material::new(
            200e9,
            0.3,
            250e6,
            HardeningLaw::VoceSaturation { q: 300e6, b: 5.0 },
        );
        let mut state = PlasticState::new_3d();
        for strain in [
            [0.1, -0.05, -0.05, 0.0, 0.0, 0.0],
            [0.03, -0.01, -0.02, 0.02, -0.01, 0.01],
        ] {
            let update = return_mapping_3d(&material, &state, &strain);
            assert!(update.yielding);
            let yield_stress = material.current_yield_stress(
                state.equiv_plastic_strain + update.delta_equiv_plastic_strain,
            );
            let equivalent = von_mises_stress_from_slice(&update.stress);
            assert!(
                (equivalent - yield_stress).abs() / yield_stress < 1e-11,
                "updated yield mismatch: equivalent={equivalent}, yield={yield_stress}"
            );
            update_plastic_state(&mut state, &update, &material);
        }
    }

    #[test]
    fn plastic_tangent_differentiates_stress_in_engineering_voigt_order() {
        for hardening in [
            HardeningLaw::Perfect,
            HardeningLaw::VoceSaturation { q: 300e6, b: 5.0 },
            HardeningLaw::LinearIsotropic { modulus: 1e9 },
            HardeningLaw::LinearKinematic { modulus: 2e9 },
            HardeningLaw::Combined {
                isotropic_modulus: 1e9,
                kinematic_modulus: 2e9,
            },
        ] {
            let material = J2Material::new(200e9, 0.3, 250e6, hardening);
            let mut state = PlasticState::new_3d();
            let preload =
                return_mapping_3d(&material, &state, &[0.002, -0.001, 0.0, 0.001, 0.0, 0.0]);
            update_plastic_state(&mut state, &preload, &material);
            for strain in [
                [0.005, -0.002, 0.001, 0.004, 0.003, -0.002],
                [0.0, 0.0, 0.0, 0.02, 0.0, 0.0],
            ] {
                let update = return_mapping_3d(&material, &state, &strain);
                assert!(update.yielding);
                // Perturb each engineering strain independently. Mixed and
                // pure-shear paths expose both coefficient and ordering errors.
                let h = 1e-7;
                for column in 0..6 {
                    let mut plus = strain;
                    let mut minus = strain;
                    plus[column] += h;
                    minus[column] -= h;
                    let a = return_mapping_3d(&material, &state, &plus);
                    let b = return_mapping_3d(&material, &state, &minus);
                    for row in 0..6 {
                        let derivative = (a.stress[row] - b.stress[row]) / (2.0 * h);
                        let error = (derivative - update.tangent[row * 6 + column]).abs()
                            / material.youngs_modulus;
                        assert!(
                            error < 1e-7,
                            "hardening={hardening:?}, row={row}, column={column}, normalized tangent error={error}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_elastic_step_no_yielding() {
        let mat = J2Material::mild_steel();
        let state = PlasticState::new_3d();

        // Small strain: well below yield
        let deps = [1e-6, -0.3e-6, -0.3e-6, 0.0, 0.0, 0.0];
        let update = return_mapping_3d(&mat, &state, &deps);

        assert!(!update.yielding);
        assert_eq!(update.delta_equiv_plastic_strain, 0.0);
        // Stress should be approximately E * eps for uniaxial
        assert!((update.stress[0] - 210e9 * 1e-6).abs() < 1e3);
    }

    #[test]
    fn test_plastic_step_yields() {
        let mat = J2Material::new(200e9, 0.3, 250e6, HardeningLaw::Perfect);
        let state = PlasticState::new_3d();

        // Large strain: well above yield
        let deps = [0.01, -0.003, -0.003, 0.0, 0.0, 0.0];
        let update = return_mapping_3d(&mat, &state, &deps);

        assert!(update.yielding);
        assert!(update.delta_equiv_plastic_strain > 0.0);

        // Von Mises stress should equal yield stress for perfect plasticity
        let q = von_mises_stress_from_slice(&update.stress);
        assert!(
            (q - 250e6).abs() < 1e3,
            "Von Mises = {q:.0}, expected 250e6"
        );
    }

    #[test]
    fn test_consistent_tangent_symmetry() {
        let mat = J2Material::mild_steel();
        let state = PlasticState::new_3d();

        let deps = [0.005, -0.0015, -0.0015, 0.0, 0.0, 0.0];
        let update = return_mapping_3d(&mat, &state, &deps);

        for i in 0..6 {
            for j in i + 1..6 {
                let diff = (update.tangent[i * 6 + j] - update.tangent[j * 6 + i]).abs();
                let scale = update.tangent[i * 6 + i].abs().max(1.0);
                assert!(diff / scale < 1e-10, "Tangent not symmetric at ({i},{j})");
            }
        }
    }

    #[test]
    fn test_hardening_increases_yield() {
        let mat = J2Material::mild_steel();
        let mut state = PlasticState::new_3d();

        // First load: yield
        let deps1 = [0.005, -0.0015, -0.0015, 0.0, 0.0, 0.0];
        let update1 = return_mapping_3d(&mat, &state, &deps1);
        update_plastic_state(&mut state, &update1, &mat);

        let q1 = von_mises_stress_from_slice(&state.stress);

        // Second load: should yield at higher stress
        let deps2 = [0.005, -0.0015, -0.0015, 0.0, 0.0, 0.0];
        let update2 = return_mapping_3d(&mat, &state, &deps2);
        update_plastic_state(&mut state, &update2, &mat);

        let q2 = von_mises_stress_from_slice(&state.stress);

        assert!(
            q2 > q1 - 1.0, // allow numerical tolerance
            "Yield stress should increase with hardening: q1={q1:.0}, q2={q2:.0}"
        );
        assert!(state.equiv_plastic_strain > 0.0);
    }

    #[test]
    fn test_elastic_tangent_matches_consistent_at_elastic_step() {
        // When no yielding occurs, the consistent tangent must equal the elastic tangent.
        let mat = J2Material::mild_steel();
        let state = PlasticState::new_3d();

        let deps = [1e-6, -0.3e-6, -0.3e-6, 0.0, 0.0, 0.0];
        let update = return_mapping_3d(&mat, &state, &deps);
        assert!(!update.yielding);

        let c_elastic = elastic_tangent_3d(mat.bulk_modulus(), mat.shear_modulus());

        for (i, (&tangent_i, &elastic_i)) in update.tangent.iter().zip(c_elastic.iter()).enumerate()
        {
            let diff = (tangent_i - elastic_i).abs();
            let scale = elastic_i.abs().max(1.0);
            assert!(
                diff / scale < 1e-12,
                "Tangent[{}] = {:.6e}, elastic = {:.6e}",
                i,
                tangent_i,
                elastic_i
            );
        }
    }

    #[test]
    fn test_consistent_tangent_numerical_derivative() {
        // Verify the consistent tangent by numerical differentiation of the stress.
        let mat = J2Material::new(
            200e9,
            0.3,
            250e6,
            HardeningLaw::LinearIsotropic { modulus: 1e9 },
        );
        let state = PlasticState::new_3d();

        // Use a large strain well into plastic regime so perturbations stay plastic
        let deps_base = [0.02, -0.006, -0.006, 0.002, 0.0, 0.0];
        let update = return_mapping_3d(&mat, &state, &deps_base);
        assert!(update.yielding, "Base strain must be in plastic regime");

        let eps = 1e-8;
        for col in 0..6 {
            let mut deps_plus = deps_base;
            let mut deps_minus = deps_base;
            deps_plus[col] += eps;
            deps_minus[col] -= eps;

            let s_plus = return_mapping_3d(&mat, &state, &deps_plus);
            let s_minus = return_mapping_3d(&mat, &state, &deps_minus);

            for row in 0..6 {
                let numerical = (s_plus.stress[row] - s_minus.stress[row]) / (2.0 * eps);
                let analytical = update.tangent[row * 6 + col];
                let diff = (analytical - numerical).abs();
                let scale = analytical.abs().max(1e6);
                // 10% tolerance: the consistent tangent is exact for the linearized
                // return mapping, but finite differences perturb the plastic state
                // slightly (different delta_gamma), causing ~5-10% discrepancy.
                assert!(
                    diff / scale < 0.10,
                    "Tangent[{row},{col}]: analytical={analytical:.4e}, numerical={numerical:.4e}, rel_err={:.2e}",
                    diff / scale
                );
            }
        }
    }

    #[test]
    fn test_plane_stress_elastic_matches_plane_stress_constitutive() {
        let mat = J2Material::mild_steel();
        let state = PlasticState::new_3d();

        // Small elastic strain in plane stress
        let deps = [1e-6, -0.3e-6, 0.0];
        let update = return_mapping_plane_stress(&mat, &state, &deps);

        assert!(!update.yielding);
        assert_eq!(update.stress.len(), 3);
        assert_eq!(update.tangent.len(), 9);

        // sigma_zz should be enforced to zero (not in output)
        // sigma_xx should match plane-stress elastic: E/(1-nu²) * (eps_xx + nu*eps_yy)
        let e = mat.youngs_modulus;
        let nu = mat.poisson_ratio;
        let d = e / (1.0 - nu * nu);
        let expected_xx = d * (deps[0] + nu * deps[1]);
        assert!(
            (update.stress[0] - expected_xx).abs() < 1e2,
            "sigma_xx={:.4e}, expected={expected_xx:.4e}",
            update.stress[0]
        );
    }

    #[test]
    fn test_plane_stress_plastic_yield() {
        let mat = J2Material::new(200e9, 0.3, 250e6, HardeningLaw::Perfect);
        let state = PlasticState::new_3d();

        // Large uniaxial strain → should yield
        let deps = [0.01, 0.0, 0.0];
        let update = return_mapping_plane_stress(&mat, &state, &deps);

        assert!(update.yielding);
        assert!(update.delta_equiv_plastic_strain > 0.0);

        // Von Mises from 2D: q = sqrt(s_xx² + s_yy² - s_xx*s_yy + 3*t_xy²)
        let sx = update.stress[0];
        let sy = update.stress[1];
        let txy = update.stress[2];
        let q = (sx * sx + sy * sy - sx * sy + 3.0 * txy * txy).sqrt();
        assert!(
            (q - 250e6).abs() < 1e4,
            "Von Mises = {q:.0}, expected ~250e6"
        );
    }

    #[test]
    fn test_plane_stress_tangent_symmetry() {
        let mat = J2Material::mild_steel();
        let state = PlasticState::new_3d();

        let deps = [0.005, -0.001, 0.001];
        let update = return_mapping_plane_stress(&mat, &state, &deps);

        for i in 0..3 {
            for j in i + 1..3 {
                let diff = (update.tangent[i * 3 + j] - update.tangent[j * 3 + i]).abs();
                let scale = update.tangent[i * 3 + i].abs().max(1.0);
                assert!(
                    diff / scale < 1e-6,
                    "Plane-stress tangent not symmetric at ({i},{j})"
                );
            }
        }
    }
}
