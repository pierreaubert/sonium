//! Layered-shell patch: CLT equivalence, strip solve, J2, thickness.
//!
//! 1. A cross-ply [0/90] layered section reproduces the composites CLT A/B/D
//!    matrices exactly through the C1 ply interface.
//! 2. A cantilever strip of MITC4 quads with layered resultants matches
//!    Euler-Bernoulli tip deflection using D11 from the same section.
//! 3. A J2 layered section yields under large membrane strain and accumulates
//!    plastic strain on commit.
//! 4. Membrane-thickness coupling thins a stretched strip element.
//!
//! Reference: Jones, R. M., "Mechanics of Composite Materials", 2nd ed.,
//! classical lamination theory.

use sonium_fem::mesh::Point;
use sonium_fem::physics::solid::composites::{Laminate, OrthotropicPly};
use sonium_fem::physics::solid::plasticity::{HardeningLaw, J2Material};
use sonium_fem::physics::solid::shell::layered::{
    AnyLayerModel, AnyLayerState, ElasticIsotropicModel, Layer, LayeredSection, SmallStrainJ2Model,
};
use sonium_fem::physics::solid::shell::{
    QuadShellElement, ShellSection, ThicknessField, ThicknessRule, add_quad_shell_point_load,
};
use sonium_solvers::direct::SparseLuFactorization;

fn cross_ply() -> Laminate {
    Laminate {
        plies: vec![
            OrthotropicPly {
                e1: 140e9,
                e2: 10e9,
                g12: 5e9,
                nu12: 0.25,
                thickness: 0.001,
                angle_rad: 0.0,
            },
            OrthotropicPly {
                e1: 140e9,
                e2: 10e9,
                g12: 5e9,
                nu12: 0.25,
                thickness: 0.001,
                angle_rad: std::f64::consts::FRAC_PI_2,
            },
        ],
    }
}

fn cantilever_tip(n_x: usize, d_star_11: f64) -> (f64, f64) {
    // Strip L = 2, width 0.1 along Y, clamped root, tip load P = 10.
    let length = 2.0;
    let width = 0.1;
    let load = 10.0;
    let n_y = 2;
    let mut points = Vec::new();
    for ix in 0..=n_x {
        for iy in 0..=n_y {
            points.push(Point::new_3d(
                length * ix as f64 / n_x as f64,
                width * iy as f64 / n_y as f64,
                0.0,
            ));
        }
    }
    let node = |ix: usize, iy: usize| -> usize { ix * (n_y + 1) + iy };
    let section = ShellSection::new(140e9, 0.25, 0.002, 1600.0);
    let mut elements = Vec::with_capacity(n_x * n_y);
    for ix in 0..n_x {
        for iy in 0..n_y {
            let q = [
                node(ix, iy),
                node(ix + 1, iy),
                node(ix + 1, iy + 1),
                node(ix, iy + 1),
            ];
            elements.push(QuadShellElement::from_points_with_section(
                q,
                &points[q[0]],
                &points[q[1]],
                &points[q[2]],
                &points[q[3]],
                section,
            ));
        }
    }
    // Rebuild stiffness with layered resultants (evaluated at zero strain).
    let laminate = cross_ply();
    let layered = LayeredSection::from_laminate(&laminate, &[1600.0, 1600.0]);
    let res = layered.resultants(&[0.0; 3], &[0.0; 3]);
    let shear = 5.0 / 6.0 * 5e9 * layered.total_thickness();
    let n_dofs = points.len() * 6;
    let mut triplet = sonium_fem::assembly::TripletMatrix::new(n_dofs, n_dofs);
    for elem in &elements {
        let k = elem.stiffness_global_general(&res.d_mm, &res.d_mb, &res.d_bb, shear);
        let dofs = elem.global_dofs();
        for (i, &gi) in dofs.iter().enumerate() {
            for (j, &gj) in dofs.iter().enumerate() {
                let val = k[i * 24 + j];
                if val.abs() > 1e-15 {
                    triplet.push(gi, gj, val);
                }
            }
        }
    }
    let mut csr = triplet.to_csr();
    let mut rhs = vec![0.0; n_dofs];
    add_quad_shell_point_load(&mut rhs, node(n_x, 1), [0.0, 0.0, -load, 0.0, 0.0, 0.0]);
    for iy in 0..=n_y {
        for d in 0..6 {
            let dof = node(0, iy) * 6 + d;
            let range = csr.row_range(dof);
            for i in range {
                csr.values[i] = 0.0;
            }
            let diag = csr
                .row_entries(dof)
                .position(|(j, _)| j == dof)
                .expect("shell CSR rows must carry a diagonal");
            csr.values[csr.row_ptrs[dof] + diag] = 1.0;
            rhs[dof] = 0.0;
        }
    }
    let lu = SparseLuFactorization::factorize(&csr).expect("sparse LU must factorize");
    let x = lu.solve(&rhs).expect("sparse LU must solve");
    let tip_w = x[node(n_x, 1) * 6 + 2];
    // Euler-Bernoulli with the coupling-corrected D*11 per unit width.
    let exact = -load * length.powi(3) / (3.0 * d_star_11 * width);
    (tip_w, exact)
}

/// Effective bending stiffness D* = D - B A^-1 B for free membrane.
fn effective_d_star(a: &[f64; 9], b: &[f64; 9], d: &[f64; 9]) -> [f64; 9] {
    let det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6])
        + a[2] * (a[3] * a[7] - a[4] * a[6]);
    let inv_det = 1.0 / det;
    let ainv = [
        (a[4] * a[8] - a[5] * a[7]) * inv_det,
        (a[2] * a[7] - a[1] * a[8]) * inv_det,
        (a[1] * a[5] - a[2] * a[4]) * inv_det,
        (a[5] * a[6] - a[3] * a[8]) * inv_det,
        (a[0] * a[8] - a[2] * a[6]) * inv_det,
        (a[2] * a[3] - a[0] * a[5]) * inv_det,
        (a[3] * a[7] - a[4] * a[6]) * inv_det,
        (a[1] * a[6] - a[0] * a[7]) * inv_det,
        (a[0] * a[4] - a[1] * a[3]) * inv_det,
    ];
    let mut out = *d;
    for i in 0..3 {
        for j in 0..3 {
            let mut correction = 0.0;
            for p in 0..3 {
                for q in 0..3 {
                    correction += b[i * 3 + p] * ainv[p * 3 + q] * b[j * 3 + q];
                }
            }
            out[i * 3 + j] -= correction;
        }
    }
    out
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        // 1. CLT equivalence through the C1 interface.
        let laminate = cross_ply();
        let abd = laminate.stiffness_abd();
        let layered = LayeredSection::from_laminate(&laminate, &[1600.0, 1600.0]);
        let res = layered.resultants(&[1e-3, 2e-4, 1e-4], &[0.5, -0.25, 0.1]);
        let mut max_err = 0.0f64;
        for i in 0..3 {
            for j in 0..3 {
                max_err = max_err
                    .max((res.d_mm[i * 3 + j] - abd.a[i][j]).abs() / abd.a[i][j].abs().max(1.0));
                max_err = max_err
                    .max((res.d_mb[i * 3 + j] - abd.b[i][j]).abs() / abd.b[i][j].abs().max(1.0));
                max_err = max_err
                    .max((res.d_bb[i * 3 + j] - abd.d[i][j]).abs() / abd.d[i][j].abs().max(1.0));
            }
        }
        println!("Layered patch: CLT A/B/D max rel err = {max_err:.3e}");
        assert!(max_err < 1e-12, "layered section must match CLT exactly");

        // 2. Cantilever strip with layered resultants vs beam theory.
        let d_star = effective_d_star(&res.d_mm, &res.d_mb, &res.d_bb);
        let (tip_w, exact) = cantilever_tip(20, d_star[0]);
        let err = (tip_w - exact).abs() / exact.abs();
        println!("Layered strip: tip w = {tip_w:.6e}, beam theory {exact:.6e} (err {err:.3e})");
        // Measured strip error = 3.86e-4 vs beam theory; gate at ~5x.
        assert!(err < 2.0e-3, "layered strip misses beam theory");

        // 3. J2 layered section yields and accumulates plastic strain.
        let layers = vec![Layer {
            thickness: 0.01,
            density: 7800.0,
            model: AnyLayerModel::J2(SmallStrainJ2Model {
                material: J2Material::new(200e9, 0.3, 250e6, HardeningLaw::Perfect),
            }),
        }];
        let mut j2 = LayeredSection::new(layers, AnyLayerState::virgin);
        j2.commit(&[0.05, 0.0, 0.0], &[0.0; 3]);
        println!(
            "Layered J2: yielded points = {}, max ep = {:.3e}",
            j2.yielded_point_count(),
            j2.max_equiv_plastic_strain()
        );
        assert!(j2.yielded_point_count() > 0, "large strain must yield");
        assert!(j2.max_equiv_plastic_strain() > 0.01);

        // 4. Thickness coupling on a stretched element.
        let mut field = ThicknessField::new(vec![0.01], ThicknessRule::plane_stress(0.3));
        field.commit_element(0, &[1e-2, -0.3e-2, 0.0]);
        let t = field.current(0);
        let expected = 0.01f64 * (-0.3f64 / 0.7f64 * 0.7e-2f64).exp();
        println!("Thickness: t = {t:.6e}, expected {expected:.6e}");
        assert!((t - expected).abs() / expected < 1e-12);
        assert!(t < 0.01);

        // Elastic C1 smoke: single isotropic ply matches Q11 * t.
        let elastic = Layer {
            thickness: 0.01,
            density: 7800.0,
            model: AnyLayerModel::Elastic(ElasticIsotropicModel {
                youngs_modulus: 200e9,
                poisson_ratio: 0.3,
            }),
        };
        let section = LayeredSection::new(vec![elastic], AnyLayerState::virgin);
        let r = section.resultants(&[1e-3, 0.0, 0.0], &[0.0; 3]);
        let q11 = 200e9 / (1.0 - 0.09);
        assert!((r.d_mm[0] - q11 * 0.01).abs() / (q11 * 0.01) < 1e-12);
    });
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
