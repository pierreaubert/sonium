//! Hexahedral H1 boundary traces using the owning element's geometry map.

// Rust guideline compliant 2026-02-21
use super::{PreparedRobinC64, constrain_boundary, invalid};
use crate::assembly::TripletMatrix;
use crate::basis::{Jacobian, PolynomialDegree, evaluate_shape};
use crate::boundary::FemSystem;
use crate::mesh::{ElementType, Mesh};
use crate::physics::acoustics::pressure::{
    HelmholtzForm,
    device_hex::{prepare_hex_helmholtz, tensor_order},
};
use crate::quadrature::gauss_legendre_1d;
use crate::spaces::{FunctionSpace, H1Space};
use num_complex::Complex64;
use sonium_compute::BackendResult;
use std::collections::HashSet;

const VERTICES: [[f64; 3]; 8] = [
    [-1., -1., -1.],
    [1., -1., -1.],
    [1., 1., -1.],
    [-1., 1., -1.],
    [-1., -1., 1.],
    [1., -1., 1.],
    [1., 1., 1.],
    [-1., 1., 1.],
];

/// Prepare Q1/Q2/Q3 hex Robin action, source and prescribed-pressure coupling.
///
/// Integrates alpha u v and g v on marked quadrilateral faces using the owning
/// element's full geometry map. Boundary records must contain all 4/9/16 nodes
/// of the geometry face, with the four corners first. Node orientation in a record may differ from the owning face order.
/// Quadrature points per axis are explicit (1–8); curved maps and variable
/// callbacks require convergence checks. This is host setup for retained device
/// matrix actions, not GPU numeric boundary assembly.
///
/// # Errors
/// Rejects invalid volume inputs, stale face ownership, duplicate marked faces,
/// inconsistent boundary nodes, degenerate surface metrics, non-finite callback
/// values and out-of-range prescribed DOFs.
pub fn prepare_hex_robin(
    mesh: &Mesh,
    space: &H1Space,
    marker: i32,
    alpha: &dyn Fn(f64, f64, f64) -> Complex64,
    source: &dyn Fn(f64, f64, f64) -> Complex64,
    quadrature_points: usize,
    dirichlet: &[usize],
) -> BackendResult<PreparedRobinC64> {
    let volume = prepare_hex_helmholtz(
        mesh,
        space,
        &HelmholtzForm::from_real(0.0),
        quadrature_points,
    )?;
    let n = volume.dimension();
    if dirichlet.iter().any(|&i| i >= n) {
        return Err(invalid("Dirichlet DOF out of bounds"));
    }
    let mut system = FemSystem::new(TripletMatrix::new(n, n), vec![Complex64::default(); n]);
    visit_hex_quadrature(
        mesh,
        space,
        marker,
        quadrature_points,
        |point, dofs, basis, weight| {
            let alpha = alpha(point[0], point[1], point[2]);
            let g = source(point[0], point[1], point[2]);
            if ![alpha.re, alpha.im, g.re, g.im]
                .into_iter()
                .all(f64::is_finite)
            {
                return Err(invalid("non-finite Robin callback"));
            }
            for (i, &row) in dofs.iter().enumerate() {
                if basis[i] == 0.0 {
                    continue;
                }
                system.rhs[row] += g * (basis[i] * weight);
                for (j, &col) in dofs.iter().enumerate() {
                    if basis[j] != 0.0 {
                        system
                            .matrix
                            .push(row, col, alpha * (basis[i] * basis[j] * weight));
                    }
                }
            }
            Ok(())
        },
    )?;
    constrain_boundary(system, dirichlet)
}

// Both callers validate volume geometry/space before entering this shared traversal.
pub(super) fn visit_hex_quadrature(
    mesh: &Mesh,
    space: &H1Space,
    marker: i32,
    quadrature_points: usize,
    mut visit: impl FnMut([f64; 3], &[usize], &[f64], f64) -> BackendResult<()>,
) -> BackendResult<()> {
    let quadrature = gauss_legendre_1d(quadrature_points);
    let mut seen = HashSet::new();
    for boundary in mesh.boundaries.iter().filter(|b| b.marker == marker) {
        let element = mesh
            .elements
            .get(boundary.element_idx)
            .ok_or_else(|| invalid("missing boundary owner"))?;
        let face = ElementType::Hexahedron
            .local_faces()
            .get(boundary.local_idx)
            .ok_or_else(|| invalid("invalid local hex face"))?;
        if !seen.insert((boundary.element_idx, boundary.local_idx)) {
            return Err(invalid("duplicate marked hex face"));
        }
        let mut corners: Vec<_> = face.iter().map(|&i| element.nodes[i]).collect();
        corners.sort_unstable();
        let mut supplied = boundary
            .nodes
            .get(..4)
            .ok_or_else(|| invalid("hex boundary needs four corners"))?
            .to_vec();
        supplied.sort_unstable();
        if corners != supplied {
            return Err(invalid("boundary corners do not match owning hex face"));
        }
        let axis = (0..3)
            .find(|&axis| {
                face.iter()
                    .all(|&i| VERTICES[i][axis] == VERTICES[face[0]][axis])
            })
            .ok_or_else(|| invalid("invalid reference face"))?;
        let fixed = VERTICES[face[0]][axis];
        let tangents: Vec<_> = (0..3).filter(|&i| i != axis).collect();
        let geometry_degree = match element.nodes.len() {
            8 => PolynomialDegree::P1,
            27 => PolynomialDegree::P2,
            64 => PolynomialDegree::P3,
            _ => return Err(invalid("unsupported hex geometry")),
        };
        {
            let order = tensor_order(geometry_degree)?;
            let size = geometry_degree.degree() + 1;
            if boundary.nodes.len() != size * size {
                return Err(invalid("boundary node count does not match geometry order"));
            }
            let plane = if fixed < 0.0 { 0 } else { size - 1 };
            let mut expected: Vec<_> = order
                .iter()
                .enumerate()
                .filter_map(|(slot, &local)| {
                    let ijk = [slot % size, slot / size % size, slot / (size * size)];
                    (ijk[axis] == plane).then_some(element.nodes[local])
                })
                .collect();
            expected.sort_unstable();
            let mut supplied = boundary.nodes.clone();
            supplied.sort_unstable();
            if expected != supplied {
                return Err(invalid(
                    "boundary high-order nodes do not match geometry face",
                ));
            }
        }
        let coordinates: Vec<_> = element
            .nodes
            .iter()
            .map(|&i| {
                let p = &mesh.nodes[i];
                [p.x, p.y, p.z]
            })
            .collect();
        let dofs = space.local_to_global(mesh, boundary.element_idx);
        for u in &quadrature {
            for v in &quadrature {
                let mut reference = [0.0; 3];
                reference[axis] = fixed;
                reference[tangents[0]] = u.xi();
                reference[tangents[1]] = v.xi();
                let [xi, eta, zeta] = reference;
                let geometry =
                    evaluate_shape(ElementType::Hexahedron, geometry_degree, xi, eta, zeta);
                let jac = Jacobian::from_3d(&geometry.gradients, 3, &coordinates);
                let a: [f64; 3] = std::array::from_fn(|i| jac.matrix[i][tangents[0]]);
                let b: [f64; 3] = std::array::from_fn(|i| jac.matrix[i][tangents[1]]);
                let cross = [
                    a[1] * b[2] - a[2] * b[1],
                    a[2] * b[0] - a[0] * b[2],
                    a[0] * b[1] - a[1] * b[0],
                ];
                let area = cross.iter().map(|v| v * v).sum::<f64>().sqrt();
                if !area.is_finite() || area <= 0.0 {
                    return Err(invalid("degenerate hex boundary metric"));
                }
                let point: [f64; 3] = std::array::from_fn(|axis| {
                    geometry
                        .values
                        .iter()
                        .zip(&coordinates)
                        .map(|(shape, p)| shape * p[axis])
                        .sum()
                });
                if point.iter().any(|v| !v.is_finite()) {
                    return Err(invalid("non-finite boundary point"));
                }
                let shape = evaluate_shape(ElementType::Hexahedron, space.degree, xi, eta, zeta);
                let weight = u.weight * v.weight * area;
                visit(point, &dofs, &shape.values, weight)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{Point, unit_cube_hexahedra};

    fn curved(degree: PolynomialDegree, beta: f64) -> Mesh {
        let base = unit_cube_hexahedra(1);
        let space = H1Space::new(degree);
        let mut mesh = Mesh::new(3);
        for [x, y, z] in space.dof_coordinates(&base).unwrap() {
            mesh.add_node(Point::new_3d(x, y, z + beta * x * x));
        }
        mesh.add_element(ElementType::Hexahedron, space.local_to_global(&base, 0));
        mesh.boundaries = base.boundaries.clone();
        crate::physics::acoustics::pressure::device_hex::populate_hex_test_boundaries(
            &mut mesh, degree,
        );
        for b in &mut mesh.boundaries {
            b.marker = if b.local_idx == 1 { 7 } else { 0 };
        }
        mesh
    }
    fn close(a: Complex64, b: Complex64) {
        assert!((a - b).norm() < 2e-11 * (1.0 + b.norm()), "{a} != {b}");
    }

    #[test]
    fn curved_hex_robin_matches_surface_area_and_trace_energy() {
        let alpha = Complex64::new(1.3, 0.4);
        let g = Complex64::new(0.8, -0.2);
        let beta = 0.2_f64;
        let c = 2.0 * beta;
        let area = 0.5 * (1.0 + c * c).sqrt() + c.asinh() / (2.0 * c);
        // Independent one-dimensional integral of x² sqrt(1+4 beta² x²).
        let x_energy: f64 = gauss_legendre_1d(20)
            .iter()
            .map(|p| {
                let x = (p.xi() + 1.0) * 0.5;
                p.weight * 0.5 * x * x * (1.0 + c * c * x * x).sqrt()
            })
            .sum();
        for geometry in [PolynomialDegree::P2, PolynomialDegree::P3] {
            let mesh = curved(geometry, beta);
            for degree in [PolynomialDegree::P1, geometry] {
                let space = H1Space::new(degree);
                let prepared =
                    prepare_hex_robin(&mesh, &space, 7, &|_, _, _| alpha, &|_, _, _| g, 8, &[])
                        .unwrap();
                close(prepared.rhs.iter().sum(), g * area);
                close(prepared.matrix.values.iter().sum(), alpha * area);
                assert!(prepared.dirichlet_coupling.values.is_empty());
                let coordinates = space.dof_coordinates(&mesh).unwrap();
                let mut energy = Complex64::default();
                for row in 0..coordinates.len() {
                    for j in prepared.matrix.row_ptrs[row]..prepared.matrix.row_ptrs[row + 1] {
                        energy += prepared.matrix.values[j]
                            * coordinates[row][0]
                            * coordinates[prepared.matrix.col_indices[j]][0];
                    }
                }
                close(energy, alpha * x_energy);
            }
        }
    }

    #[test]
    fn shared_hex_faces_preserve_polynomial_load_and_constraint_split() {
        let mut mesh = unit_cube_hexahedra(2);
        for b in &mut mesh.boundaries {
            b.marker = if b.nodes.iter().all(|&i| mesh.nodes[i].z == 1.0) {
                7
            } else {
                0
            };
        }
        for degree in [
            PolynomialDegree::P1,
            PolynomialDegree::P2,
            PolynomialDegree::P3,
        ] {
            let space = H1Space::new(degree);
            let full = prepare_hex_robin(
                &mesh,
                &space,
                7,
                &|_, _, _| Complex64::new(1.0, 0.0),
                &|x, y, _| Complex64::new(x + y, 0.0),
                6,
                &[],
            )
            .unwrap();
            close(full.matrix.values.iter().sum(), Complex64::new(1.0, 0.0));
            close(full.rhs.iter().sum(), Complex64::new(1.0, 0.0));
            let coords = space.dof_coordinates(&mesh).unwrap();
            let weighted_load: Complex64 = full
                .rhs
                .iter()
                .zip(&coords)
                .map(|(b, p)| *b * (p[0] + p[1]))
                .sum();
            close(weighted_load, Complex64::new(7.0 / 6.0, 0.0));
            let dofs: Vec<_> = coords
                .iter()
                .enumerate()
                .filter_map(|(i, p)| (p[0] == 0.0).then_some(i))
                .collect();
            let constrained = prepare_hex_robin(
                &mesh,
                &space,
                7,
                &|_, _, _| Complex64::new(1.0, 0.0),
                &|x, y, _| Complex64::new(x + y, 0.0),
                6,
                &dofs,
            )
            .unwrap();
            assert!(!constrained.dirichlet_coupling.values.is_empty());
            let x: Vec<_> = (0..coords.len())
                .map(|i| Complex64::new((i as f64).sin(), 0.3))
                .collect();
            let apply = |matrix: &sonium_solvers::CsrMatrix<Complex64>, row: usize| -> Complex64 {
                (matrix.row_ptrs[row]..matrix.row_ptrs[row + 1])
                    .map(|j| matrix.values[j] * x[matrix.col_indices[j]])
                    .sum()
            };
            for row in 0..coords.len() {
                if dofs.contains(&row) {
                    close(constrained.rhs[row], Complex64::default());
                    close(apply(&constrained.matrix, row), Complex64::default());
                    close(
                        apply(&constrained.dirichlet_coupling, row),
                        Complex64::default(),
                    );
                } else {
                    close(constrained.rhs[row], full.rhs[row]);
                    close(
                        apply(&constrained.matrix, row)
                            + apply(&constrained.dirichlet_coupling, row),
                        apply(&full.matrix, row),
                    );
                }
            }
        }
    }

    #[test]
    fn explicit_hex_face_nodes_preserve_orientation_independence() {
        for degree in [PolynomialDegree::P2, PolynomialDegree::P3] {
            let mut mesh = curved(degree, 0.2);
            let space = H1Space::new(degree);
            let baseline = prepare_hex_robin(
                &mesh,
                &space,
                7,
                &|_, _, _| Complex64::new(1.0, 0.3),
                &|_, _, _| Complex64::new(0.2, 0.1),
                6,
                &[],
            )
            .unwrap();
            let top = mesh.boundaries.iter_mut().find(|b| b.marker == 7).unwrap();
            top.nodes[..4].reverse();
            top.nodes[4..].reverse();
            let prepared = prepare_hex_robin(
                &mesh,
                &space,
                7,
                &|_, _, _| Complex64::new(1.0, 0.3),
                &|_, _, _| Complex64::new(0.2, 0.1),
                6,
                &[],
            )
            .unwrap();
            assert_eq!(prepared.matrix.values, baseline.matrix.values);
            assert_eq!(prepared.rhs, baseline.rhs);
            let top = mesh.boundaries.iter_mut().find(|b| b.marker == 7).unwrap();
            top.nodes[4] = mesh.elements[0].nodes[0];
            assert!(
                prepare_hex_robin(
                    &mesh,
                    &space,
                    7,
                    &|_, _, _| Complex64::new(1.0, 0.0),
                    &|_, _, _| Complex64::default(),
                    6,
                    &[]
                )
                .is_err()
            );
        }
    }

    #[test]
    fn invalid_hex_boundary_data_is_rejected() {
        let space = H1Space::new(PolynomialDegree::P2);
        let mesh = curved(PolynomialDegree::P2, 0.2);
        let alpha = |_: f64, _: f64, _: f64| Complex64::new(1.0, 0.0);
        let zero = |_: f64, _: f64, _: f64| Complex64::default();
        prepare_hex_robin(&mesh, &space, 7, &alpha, &zero, 4, &[]).unwrap();
        assert!(
            prepare_hex_robin(
                &mesh,
                &space,
                7,
                &|_, _, _| Complex64::new(f64::NAN, 0.0),
                &zero,
                4,
                &[]
            )
            .is_err()
        );
        for mode in 0..4 {
            let mut bad = mesh.clone();
            let index = bad.boundaries.iter().position(|b| b.marker == 7).unwrap();
            match mode {
                0 => bad.boundaries[index].local_idx = 0,
                1 => bad.boundaries[index].element_idx = usize::MAX,
                2 => bad.boundaries.push(bad.boundaries[index].clone()),
                _ => bad.boundaries[index].nodes.truncate(4),
            }
            assert!(prepare_hex_robin(&bad, &space, 7, &alpha, &zero, 4, &[]).is_err());
        }
    }
}
