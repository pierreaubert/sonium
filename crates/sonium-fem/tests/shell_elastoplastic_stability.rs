//! Stability of elastoplastic explicit shells on a curved mesh.
//!
//! Regression test: the elastoplastic residual once double-counted stress
//! history (total strain passed to an incremental return mapping) and
//! rotated local displacements with a non-transposed frame. Both errors
//! vanish on flat single-element checks, so this test integrates the
//! tilted arch mesh explicitly and requires a bounded response.

use sonium_fem::mesh::Point;
use sonium_fem::physics::solid::plasticity::{HardeningLaw, J2Material};
use sonium_fem::physics::solid::shell::explicit::{ShellExplicitModel, ShellExplicitState};
use sonium_fem::physics::solid::shell::{ShellElement, ShellSection};

fn arch_mesh() -> (Vec<ShellElement>, usize, usize, usize) {
    let span = 1.0f64;
    let rise = 0.09f64;
    let width = 0.14f64;
    let n_along = 24usize;
    let n_across = 3usize;
    let radius = ((span / 2.0).powi(2) + rise.powi(2)) / (2.0 * rise);
    let phi_max = (span / 2.0 / radius).asin();
    let section = ShellSection::new(200.0e9, 0.3, 0.004, 7800.0);
    let node_id = |i: usize, j: usize| i * (n_across + 1) + j;
    let point = |i: usize, j: usize| {
        let phi = -phi_max + 2.0 * phi_max * i as f64 / n_along as f64;
        Point::new_3d(
            radius * phi.sin(),
            -width / 2.0 + width * j as f64 / n_across as f64,
            radius * phi.cos() - (radius - rise),
        )
    };
    let n_dofs = (n_along + 1) * (n_across + 1) * 6;
    let mut elements: Vec<ShellElement> = Vec::new();
    for i in 0..n_along {
        for j in 0..n_across {
            let a = node_id(i, j);
            let b = node_id(i + 1, j);
            let c = node_id(i + 1, j + 1);
            let d = node_id(i, j + 1);
            for (nodes, pa, pb, pc) in [
                ([a, b, c], point(i, j), point(i + 1, j), point(i + 1, j + 1)),
                ([a, c, d], point(i, j), point(i + 1, j + 1), point(i, j + 1)),
            ] {
                elements.push(ShellElement::from_points(nodes, &pa, &pb, &pc, section));
            }
        }
    }
    (elements, n_dofs, n_along, n_across)
}

#[test]
fn elastoplastic_explicit_stays_bounded() {
    let (elements, n_dofs, n_along, n_across) = arch_mesh();
    let node_id = |i: usize, j: usize| i * (n_across + 1) + j;
    let material = J2Material::new(
        200.0e9,
        0.3,
        250.0e6,
        HardeningLaw::LinearIsotropic { modulus: 0.0 },
    );
    let dt = 0.85
        * ShellExplicitModel::elastic(elements.clone(), n_dofs)
            .unwrap()
            .wave_dt();
    let mut model = ShellExplicitModel::elastoplastic(elements, n_dofs, material).unwrap();
    for j in 0..=n_across {
        model.fix_node(node_id(0, j));
        model.fix_node(node_id(n_along, j));
    }
    let mut state = ShellExplicitState::new(n_dofs);
    for j in 0..=n_across {
        state.u[node_id(n_along / 2, j) * 6 + 2] = 1e-9;
    }
    let zero_force = vec![0.0f64; n_dofs];
    for _ in 1..=10000 {
        model.step(&mut state, &zero_force, dt).unwrap();
    }
    let max_u = state.u.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    assert!(
        max_u < 1e-6,
        "elastoplastic explicit response grew to {max_u:e} from a 1e-9 pluck"
    );
}
