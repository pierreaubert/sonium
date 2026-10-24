//! Prepare hexahedral Helmholtz volume data for resident partial assembly.
//!
//! The H1 solution basis and geometry may have different orders. Geometry is
//! evaluated at every quadrature point, including explicit curved Q2/Q3 maps.
//! Prescribed-pressure DOFs use `try_with_dirichlet` and `lift_rhs` on the
//! prepared operator. Facet operators, sources and preconditioners remain separate.

// Rust guideline compliant 2026-02-21
#[cfg(test)]
mod aggregation_tests;

#[cfg(all(test, feature = "cuda"))]
mod native_lor;

use super::HelmholtzForm;
use crate::basis::{Jacobian, PolynomialDegree, evaluate_shape};
use crate::mesh::{ElementType, Mesh};
use crate::quadrature::gauss_legendre_1d;
use crate::spaces::{FunctionSpace, H1Space};
use num_complex::Complex64;
use sonium_compute::fem_hex::HexPartialC64;
use sonium_compute::{BackendError, BackendResult};

fn invalid(reason: impl Into<String>) -> BackendError {
    BackendError::InvalidArgument {
        operation: "prepare_hex_helmholtz",
        reason: reason.into(),
    }
}

// Derive the permutation from FEM's canonical nodal basis rather than duplicate
// its edge/face ordering. This is setup work, independent of mesh size.
pub(super) fn tensor_order(degree: PolynomialDegree) -> BackendResult<Vec<usize>> {
    let n = degree.degree() + 1;
    let coordinate = |i: usize| -1.0 + 2.0 * i as f64 / (n - 1) as f64;
    let mut order = Vec::with_capacity(n * n * n);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let shape = evaluate_shape(
                    ElementType::Hexahedron,
                    degree,
                    coordinate(i),
                    coordinate(j),
                    coordinate(k),
                );
                let node = shape
                    .values
                    .iter()
                    .position(|v| (*v - 1.0).abs() < 1e-12)
                    .ok_or_else(|| invalid("missing tensor nodal basis"))?;
                if order.contains(&node)
                    || shape
                        .values
                        .iter()
                        .enumerate()
                        .any(|(index, v)| index != node && v.abs() > 1e-12)
                {
                    return Err(invalid("basis is not a tensor nodal permutation"));
                }
                order.push(node);
            }
        }
    }
    Ok(order)
}

/// Prepare Q1/Q2/Q3 Helmholtz volume data without assembling element matrices.
///
/// `quadrature_points` is the number of Gauss points per axis (1–8); use at
/// least degree + 1 for affine constant-coefficient mass integration. Curved
/// geometry requires quadrature convergence checks. Upload the result once with
/// `CudaHexPartialC64::try_new` and retain it across Krylov actions.
///
/// # Errors
/// Rejects invalid topology/DOF maps, non-3D or non-hex meshes, unsupported
/// geometry counts, non-finite coefficients/geometry, degenerate sampled
/// Jacobians, sampled orientation changes, and kernel index overflow.
/// Uniformly reversed orientation is supported through the absolute measure.
/// Sampled checks do not certify geometry validity between quadrature points.
pub fn prepare_hex_helmholtz(
    mesh: &Mesh,
    space: &H1Space,
    form: &HelmholtzForm,
    quadrature_points: usize,
) -> BackendResult<HexPartialC64> {
    prepare_hex_helmholtz_impl(mesh, space, form, quadrature_points, None)
}

/// Prepare volume data and physical coordinates in matching tensor quadrature order.
///
/// Points follow element order, then x-fastest quadrature within each element.
/// Coordinates use the full geometry basis, including curved Q2/Q3 maps.
/// Upload them once for a device material producer; no material law is evaluated.
///
/// # Errors
/// Uses the same geometry, space, quadrature and coefficient checks as
/// `prepare_hex_helmholtz`.
pub fn prepare_hex_helmholtz_with_points(
    mesh: &Mesh,
    space: &H1Space,
    form: &HelmholtzForm,
    quadrature_points: usize,
) -> BackendResult<(HexPartialC64, Vec<[f64; 3]>)> {
    let mut points = Vec::new();
    let volume =
        prepare_hex_helmholtz_impl(mesh, space, form, quadrature_points, Some(&mut points))?;
    Ok((volume, points))
}

fn prepare_hex_helmholtz_impl(
    mesh: &Mesh,
    space: &H1Space,
    form: &HelmholtzForm,
    quadrature_points: usize,
    mut points: Option<&mut Vec<[f64; 3]>>,
) -> BackendResult<HexPartialC64> {
    if mesh.dimension != 3 || mesh.elements.is_empty() || !(1..=8).contains(&quadrature_points) {
        return Err(invalid(
            "requires nonempty 3D mesh and 1–8 quadrature points per axis",
        ));
    }
    if mesh
        .elements
        .iter()
        .any(|e| e.element_type != ElementType::Hexahedron || !matches!(e.nodes.len(), 8 | 27 | 64))
    {
        return Err(invalid("requires hexahedral Q1/Q2/Q3 geometry"));
    }
    mesh.validate_connectivity()
        .map_err(|errors| invalid(format!("invalid mesh: {errors:?}")))?;
    space.validate(mesh).map_err(|e| invalid(e.to_string()))?;
    let dimension = space.num_global_dofs(mesh);
    if dimension > u32::MAX as usize {
        return Err(invalid("global DOF count exceeds kernel indices"));
    }
    let mass = -form.wavenumber * form.wavenumber;
    if !mass.re.is_finite() || !mass.im.is_finite() {
        return Err(invalid("non-finite Helmholtz coefficient"));
    }
    let n = space.degree.degree() + 1;
    let order = tensor_order(space.degree)?;
    let quadrature = gauss_legendre_1d(quadrature_points);
    let mut basis = Vec::with_capacity(n * quadrature_points);
    let mut derivative = Vec::with_capacity(n * quadrature_points);
    for point in &quadrature {
        let shape = evaluate_shape(
            ElementType::Hexahedron,
            space.degree,
            point.xi(),
            -1.0,
            -1.0,
        );
        for &node in &order[..n] {
            basis.push(shape.values[node]);
            derivative.push(shape.gradients[3 * node]);
        }
    }
    let mut nodes = Vec::new();
    let mut data = Vec::new();
    for (element_index, element) in mesh.elements.iter().enumerate() {
        let dofs = space.local_to_global(mesh, element_index);
        nodes.extend(order.iter().map(|&i| dofs[i] as u32));
        let geometry_degree = match element.nodes.len() {
            8 => PolynomialDegree::P1,
            27 => PolynomialDegree::P2,
            64 => PolynomialDegree::P3,
            _ => return Err(invalid("unsupported geometry")),
        };
        let coordinates: Vec<_> = element
            .nodes
            .iter()
            .map(|&i| {
                let p = &mesh.nodes[i];
                [p.x, p.y, p.z]
            })
            .collect();
        if coordinates.iter().flatten().any(|v| !v.is_finite()) {
            return Err(invalid("non-finite geometry"));
        }
        let mut orientation = None;
        // The compute tensor contract is x fastest, unlike gauss_hexahedron.
        for z in &quadrature {
            for y in &quadrature {
                for x in &quadrature {
                    let shape = evaluate_shape(
                        ElementType::Hexahedron,
                        geometry_degree,
                        x.xi(),
                        y.xi(),
                        z.xi(),
                    );
                    let jac = Jacobian::from_3d(&shape.gradients, 3, &coordinates);
                    let scale = jac
                        .matrix
                        .iter()
                        .flatten()
                        .fold(0.0_f64, |a, &b| a.max(b.abs()));
                    if !jac.det.is_finite()
                        || jac.det.abs() <= 64.0 * f64::EPSILON * scale.powi(3)
                        || jac.inverse.iter().any(|v| !v.is_finite())
                    {
                        return Err(invalid(format!(
                            "degenerate Jacobian in element {element_index}"
                        )));
                    }
                    let sign = jac.det.is_sign_positive();
                    if orientation.is_some_and(|previous| previous != sign) {
                        return Err(invalid(format!(
                            "folded sampled geometry in element {element_index}"
                        )));
                    }
                    orientation = Some(sign);
                    if let Some(points) = points.as_deref_mut() {
                        let mut position = [0.0; 3];
                        for (weight, coordinate) in shape.values.iter().zip(&coordinates) {
                            for axis in 0..3 {
                                position[axis] += weight * coordinate[axis];
                            }
                        }
                        if position.iter().any(|v| !v.is_finite()) {
                            return Err(invalid("non-finite quadrature position"));
                        }
                        points.push(position);
                    }
                    let weight = x.weight * y.weight * z.weight * jac.det.abs();
                    let metric = |i: usize, j: usize| -> Complex64 {
                        Complex64::new(
                            weight
                                * (0..3)
                                    .map(|k| jac.inverse[3 * i + k] * jac.inverse[3 * j + k])
                                    .sum::<f64>(),
                            0.0,
                        )
                    };
                    data.push([
                        mass * weight,
                        metric(0, 0),
                        metric(1, 1),
                        metric(2, 2),
                        metric(0, 1),
                        metric(0, 2),
                        metric(1, 2),
                    ]);
                }
            }
        }
    }
    HexPartialC64::try_new(
        dimension,
        n,
        quadrature_points,
        nodes,
        basis,
        derivative,
        data,
    )
}

/// Assemble a constrained shifted-Laplacian matrix for retained device AMG setup.
///
/// Returns K + `mass_shift` M with the same quadrature, H1 numbering and
/// symmetric Dirichlet elimination as the partial-assembly volume operator.
/// A common damping choice is `mass_shift = -k² + i β k²`, β > 0. The caller
/// selects the shift and AMG configuration for the workload; convergence must
/// be checked on the original Helmholtz operator, not the shifted matrix.
/// This is explicit host setup. Upload once using the CUDA complex-double AMG
/// coarse-LU constructor and retain it alongside a device Krylov session.
///
/// # Errors
/// Rejects invalid volume inputs, quadrature, constraints, non-finite shift or
/// checked FEM assembly failures.
pub fn prepare_hex_shifted_laplacian(
    mesh: &Mesh,
    space: &H1Space,
    mass_shift: Complex64,
    quadrature_points: usize,
    dirichlet: &[usize],
) -> BackendResult<sonium_solvers::CsrMatrix<Complex64>> {
    // Match the volume producer's admission, including sampled geometry checks.
    let form = HelmholtzForm::new((-mass_shift).sqrt());
    let prepared = prepare_hex_helmholtz(mesh, space, &form, quadrature_points)?;
    let n = prepared.dimension();
    if dirichlet.iter().any(|&i| i >= n) {
        return Err(invalid("preconditioner Dirichlet DOF out of range"));
    }
    let mut mask = vec![false; n];
    for &i in dirichlet {
        mask[i] = true;
    }
    let matrix =
        crate::assembly::assemble_bilinear_checked(&form, space, space, mesh, quadrature_points)
            .map_err(|e| invalid(e.to_string()))?
            .to_csr();
    let mut entries = Vec::with_capacity(matrix.values.len());
    for row in 0..n {
        if mask[row] {
            entries.push((row, row, Complex64::new(1.0, 0.0)));
        } else {
            for j in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
                let col = matrix.col_indices[j];
                if !mask[col] {
                    entries.push((row, col, matrix.values[j]));
                }
            }
        }
    }
    sonium_solvers::CsrMatrix::try_from_triplets(n, n, entries).map_err(|e| invalid(e.to_string()))
}

/// Assemble a low-order refined shifted operator on the high-order nodal grid.
///
/// Splits each Q1/Q2/Q3 hex into degree-cubed trilinear subcells. Global node
/// indices are the original H1 DOFs, so a device AMG hierarchy built from this
/// matrix acts directly on high-order vectors without transfer or reordering.
/// The original high-order system operator must still be used for residuals
/// and the outer solve. This is an auxiliary preconditioning matrix, not an
/// equivalent discretization. Curved geometry is approximated by the subcells;
/// each subcell is checked through the Q1 producer and integrated with two Gauss
/// points per axis. Geometry and matrix construction are host setup.
///
/// # Errors
/// Rejects unsupported spaces/geometry, missing nodal coordinates, invalid
/// subcells, non-finite shifts or out-of-range prescribed DOFs.
pub fn prepare_hex_lor_shifted_laplacian(
    mesh: &Mesh,
    space: &H1Space,
    mass_shift: Complex64,
    dirichlet: &[usize],
) -> BackendResult<sonium_solvers::CsrMatrix<Complex64>> {
    let degree = space.degree.degree();
    // Validate the same geometry/numbering contract as partial assembly.
    let volume = prepare_hex_helmholtz(mesh, space, &HelmholtzForm::from_real(0.0), degree + 1)?;
    let coordinates = space
        .dof_coordinates(mesh)
        .ok_or_else(|| invalid("missing high-order nodal coordinates"))?;
    if coordinates.len() != volume.dimension() {
        return Err(invalid("auxiliary nodal grid dimension mismatch"));
    }
    let mut refined = Mesh::new(3);
    for [x, y, z] in coordinates {
        refined.add_node(crate::mesh::Point::new_3d(x, y, z));
    }
    let order = tensor_order(space.degree)?;
    let width = degree + 1;
    for element in 0..mesh.elements.len() {
        let dofs = space.local_to_global(mesh, element);
        let node = |i, j, k| dofs[order[i + width * (j + width * k)]];
        for k in 0..degree {
            for j in 0..degree {
                for i in 0..degree {
                    refined.add_element(
                        ElementType::Hexahedron,
                        vec![
                            node(i, j, k),
                            node(i + 1, j, k),
                            node(i + 1, j + 1, k),
                            node(i, j + 1, k),
                            node(i, j, k + 1),
                            node(i + 1, j, k + 1),
                            node(i + 1, j + 1, k + 1),
                            node(i, j + 1, k + 1),
                        ],
                    );
                }
            }
        }
    }
    prepare_hex_shifted_laplacian(
        &refined,
        &H1Space::new(PolynomialDegree::P1),
        mass_shift,
        2,
        dirichlet,
    )
}

// Construct complete geometry traces for the curved integration fixtures.
#[cfg(test)]
pub(super) fn populate_hex_test_boundaries(mesh: &mut Mesh, degree: PolynomialDegree) {
    let order = tensor_order(degree).unwrap();
    let size = degree.degree() + 1;
    let coords = |slot: usize| [slot % size, slot / size % size, slot / (size * size)];
    for boundary in &mut mesh.boundaries {
        let element = &mesh.elements[boundary.element_idx];
        let face = &ElementType::Hexahedron.local_faces()[boundary.local_idx];
        let corners: Vec<_> = face
            .iter()
            .map(|local| coords(order.iter().position(|i| i == local).unwrap()))
            .collect();
        let axis = (0..3)
            .find(|&axis| corners.iter().all(|p| p[axis] == corners[0][axis]))
            .unwrap();
        boundary.nodes = face.iter().map(|&i| element.nodes[i]).collect();
        for (slot, &local) in order.iter().enumerate() {
            if coords(slot)[axis] == corners[0][axis]
                && !boundary.nodes.contains(&element.nodes[local])
            {
                boundary.nodes.push(element.nodes[local]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembly::assemble_bilinear_checked;
    use crate::mesh::{Point, unit_cube_hexahedra};
    use sonium_compute::Transpose;

    #[test]
    fn low_order_refined_operator_matches_independent_uniform_mesh() {
        let mesh = unit_cube_hexahedra(2);
        let shift = Complex64::new(3.0, 0.7);
        for degree in [
            PolynomialDegree::P1,
            PolynomialDegree::P2,
            PolynomialDegree::P3,
        ] {
            let space = H1Space::new(degree);
            let coordinates = space.dof_coordinates(&mesh).unwrap();
            let n = coordinates.len();
            let reference_mesh = unit_cube_hexahedra(2 * degree.degree());
            let reference_space = H1Space::new(PolynomialDegree::P1);
            let permutation: Vec<_> = coordinates
                .iter()
                .map(|p| {
                    reference_mesh
                        .nodes
                        .iter()
                        .position(|q| {
                            (p[0] - q.x).abs() < 1e-12
                                && (p[1] - q.y).abs() < 1e-12
                                && (p[2] - q.z).abs() < 1e-12
                        })
                        .unwrap()
                })
                .collect();
            let mut sorted = permutation.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..n).collect::<Vec<_>>());
            for constrained in [vec![], vec![0, n / 2, n - 1]] {
                let actual =
                    prepare_hex_lor_shifted_laplacian(&mesh, &space, shift, &constrained).unwrap();
                let mapped_constraints: Vec<_> =
                    constrained.iter().map(|&i| permutation[i]).collect();
                let expected = prepare_hex_shifted_laplacian(
                    &reference_mesh,
                    &reference_space,
                    shift,
                    2,
                    &mapped_constraints,
                )
                .unwrap();
                assert_eq!(actual.num_rows, n);
                let x: Vec<_> = (0..n)
                    .map(|i| Complex64::new((i as f64).sin(), 0.2))
                    .collect();
                let mut mapped_x = vec![Complex64::default(); n];
                for (i, &j) in permutation.iter().enumerate() {
                    mapped_x[j] = x[i];
                }
                let apply_row = |a: &sonium_solvers::CsrMatrix<Complex64>,
                                 values: &[Complex64],
                                 row: usize|
                 -> Complex64 {
                    (a.row_ptrs[row]..a.row_ptrs[row + 1])
                        .map(|j| a.values[j] * values[a.col_indices[j]])
                        .sum()
                };
                for (row, &mapped) in permutation.iter().enumerate() {
                    let a = apply_row(&actual, &x, row);
                    let e = apply_row(&expected, &mapped_x, mapped);
                    assert!((a - e).norm() < 2e-12 * (1.0 + e.norm()));
                }
                if constrained.is_empty() {
                    let integral: Complex64 = actual.values.iter().sum();
                    assert!((integral - shift).norm() < 2e-12);
                }
            }
            assert!(prepare_hex_lor_shifted_laplacian(&mesh, &space, shift, &[n]).is_err());
        }
    }

    fn compare(mesh: &Mesh, degree: PolynomialDegree, q: usize) {
        let space = H1Space::new(degree);
        let form = HelmholtzForm::new(Complex64::new(2.3, -0.4));
        let op = prepare_hex_helmholtz(mesh, &space, &form, q).unwrap();
        let csr = assemble_bilinear_checked(&form, &space, &space, mesh, q)
            .unwrap()
            .to_csr();
        assert_eq!(op.dimension(), csr.num_rows);
        for (row, actual) in op.diagonal().iter().enumerate() {
            let expected: Complex64 = (csr.row_ptrs[row]..csr.row_ptrs[row + 1])
                .filter(|&i| csr.col_indices[i] == row)
                .map(|i| csr.values[i])
                .sum();
            assert!((*actual - expected).norm() < 2e-11 * (1.0 + expected.norm()));
        }

        let x: Vec<_> = (0..op.dimension())
            .map(|i| Complex64::new((i as f64 * 0.7).sin(), (i as f64 * 0.3).cos()))
            .collect();
        for action in [Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans] {
            let mut expected = vec![Complex64::default(); x.len()];
            for row in 0..csr.num_rows {
                for offset in csr.row_ptrs[row]..csr.row_ptrs[row + 1] {
                    let col = csr.col_indices[offset];
                    let v = csr.values[offset];
                    match action {
                        Transpose::NoTrans => expected[row] += v * x[col],
                        Transpose::Trans => expected[col] += v * x[row],
                        Transpose::ConjTrans => expected[col] += v.conj() * x[row],
                    }
                }
            }
            let mut actual = vec![Complex64::default(); x.len()];
            op.apply(&x, &mut actual, action).unwrap();
            for (a, b) in actual.iter().zip(expected) {
                assert!(
                    (*a - b).norm() < 2e-11 * (1.0 + b.norm()),
                    "{degree:?} {action:?}: {a} != {b}"
                );
            }
        }
    }

    #[test]
    fn hex_volume_matches_fem_on_shared_warped_elements() {
        let mut mesh = unit_cube_hexahedra(2);
        for p in &mut mesh.nodes {
            p.x = 1.3 * p.x + 0.2 * p.y + 0.13 * p.y * p.z;
            p.y = 0.8 * p.y + 0.1 * p.z;
            p.z *= 1.7;
        }
        for degree in [
            PolynomialDegree::P1,
            PolynomialDegree::P2,
            PolynomialDegree::P3,
        ] {
            compare(&mesh, degree, degree.degree() + 2);
        }
        for p in &mut mesh.nodes {
            p.x = -p.x;
        }
        compare(&mesh, PolynomialDegree::P3, 5);
    }

    fn curved_mesh(degree: PolynomialDegree) -> Mesh {
        let base = unit_cube_hexahedra(1);
        let space = H1Space::new(degree);
        let points = space.dof_coordinates(&base).unwrap();
        let mut mesh = Mesh::new(3);
        for [x, y, z] in points {
            mesh.add_node(Point::new_3d(
                x + 0.12 * x * (1.0 - x) * y,
                y + 0.1 * z * (1.0 - z) * x,
                z + 0.08 * y * (1.0 - y) * x,
            ));
        }
        mesh.add_element(ElementType::Hexahedron, space.local_to_global(&base, 0));
        mesh.boundaries = base.boundaries.clone();
        populate_hex_test_boundaries(&mut mesh, degree);
        mesh
    }

    #[test]
    fn curved_q2_q3_geometry_matches_fem_quadrature() {
        for degree in [PolynomialDegree::P2, PolynomialDegree::P3] {
            let mesh = curved_mesh(degree);
            compare(&mesh, degree, 5);
            compare(&mesh, PolynomialDegree::P1, 5);
            let (_, points) = prepare_hex_helmholtz_with_points(
                &mesh,
                &H1Space::new(PolynomialDegree::P1),
                &HelmholtzForm::from_real(1.0),
                5,
            )
            .unwrap();
            let q = gauss_legendre_1d(5);
            assert_eq!(points.len(), 125);
            for (i, point) in points.iter().enumerate() {
                let x = (q[i % 5].xi() + 1.0) * 0.5;
                let y = (q[i / 5 % 5].xi() + 1.0) * 0.5;
                let z = (q[i / 25].xi() + 1.0) * 0.5;
                let expected = [
                    x + 0.12 * x * (1.0 - x) * y,
                    y + 0.1 * z * (1.0 - z) * x,
                    z + 0.08 * y * (1.0 - y) * x,
                ];
                for (a, e) in point.iter().zip(expected) {
                    assert!((*a - e).abs() < 3e-13);
                }
            }
        }
    }

    #[test]
    fn invalid_hex_inputs_are_rejected() {
        let base = unit_cube_hexahedra(1);
        let space = H1Space::new(PolynomialDegree::P2);
        let form = HelmholtzForm::from_real(2.0);
        for q in [0, 9] {
            assert!(prepare_hex_helmholtz(&base, &space, &form, q).is_err());
        }
        for mode in 0..6 {
            let mut mesh = base.clone();
            match mode {
                0 => mesh.nodes[0].x = f64::NAN,
                1 => {
                    for p in &mut mesh.nodes {
                        p.z = 0.0;
                    }
                }
                2 => mesh.elements[0].nodes[0] = usize::MAX,
                3 => mesh.elements[0].nodes[0] = mesh.elements[0].nodes[1],
                4 => mesh.nodes[0].x = 2.0,
                _ => mesh.dimension = 2,
            }
            assert!(prepare_hex_helmholtz(&mesh, &space, &form, 3).is_err());
        }
        assert!(
            prepare_hex_helmholtz(&base, &space, &HelmholtzForm::from_real(f64::INFINITY), 3)
                .is_err()
        );
    }
    #[test]
    fn hex_dirichlet_matches_fem_boundary_elimination() {
        use crate::boundary::{FemSystem, try_apply_dirichlet_values_complex};
        let mesh = unit_cube_hexahedra(2);
        for degree in [
            PolynomialDegree::P1,
            PolynomialDegree::P2,
            PolynomialDegree::P3,
        ] {
            let space = H1Space::new(degree);
            let marker = mesh.boundaries[0].marker;
            let dofs = space.boundary_dofs(&mesh, marker);
            let coordinates = space.dof_coordinates(&mesh).unwrap();
            let size = coordinates.len();
            assert!(!dofs.is_empty() && dofs.len() < size);
            let form = HelmholtzForm::new(Complex64::new(2.3, -0.4));
            let op = prepare_hex_helmholtz(&mesh, &space, &form, 5)
                .unwrap()
                .try_with_dirichlet(&dofs)
                .unwrap();
            let matrix = assemble_bilinear_checked(&form, &space, &space, &mesh, 5).unwrap();
            let b: Vec<_> = (0..size)
                .map(|i| Complex64::new(i as f64 * 0.1, -0.2))
                .collect();
            for shift in [0.0, 0.7] {
                let value = |x: f64, y: f64, z: f64| Complex64::new(0.2 + x + shift, y - 0.3 * z);
                let g: Vec<_> = coordinates
                    .iter()
                    .map(|&[x, y, z]| value(x, y, z))
                    .collect();
                let mut system = FemSystem::new(matrix.clone(), b.clone());
                try_apply_dirichlet_values_complex(
                    &mut system,
                    &mesh,
                    &space,
                    marker,
                    &value,
                    None,
                )
                .unwrap();
                let mut lifted = vec![Complex64::default(); size];
                op.lift_rhs(&b, &g, &mut lifted).unwrap();
                for (a, b) in lifted.iter().zip(&system.rhs) {
                    assert!((*a - *b).norm() < 2e-11 * (1.0 + b.norm()));
                }
                let csr = system.matrix.to_csr();
                for action in [Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans] {
                    let mut expected = vec![Complex64::default(); size];
                    for row in 0..size {
                        for offset in csr.row_ptrs[row]..csr.row_ptrs[row + 1] {
                            let col = csr.col_indices[offset];
                            let v = csr.values[offset];
                            match action {
                                Transpose::NoTrans => expected[row] += v * b[col],
                                Transpose::Trans => expected[col] += v * b[row],
                                Transpose::ConjTrans => expected[col] += v.conj() * b[row],
                            }
                        }
                    }
                    let mut actual = vec![Complex64::default(); size];
                    op.apply(&b, &mut actual, action).unwrap();
                    for (a, b) in actual.iter().zip(expected) {
                        assert!((*a - b).norm() < 2e-11 * (1.0 + b.norm()));
                    }
                }
                for (row, actual) in op.diagonal().iter().enumerate() {
                    let expected: Complex64 = (csr.row_ptrs[row]..csr.row_ptrs[row + 1])
                        .filter(|&i| csr.col_indices[i] == row)
                        .map(|i| csr.values[i])
                        .sum();
                    assert!((*actual - expected).norm() < 2e-11 * (1.0 + expected.norm()));
                }
            }
        }
    }

    #[test]
    fn shifted_amg_fgmres_solves_original_indefinite_hex_system() {
        use sonium_compute::{Backend, CpuBackend, DeviceCsr, MemoryPool, MemoryPoolConfig};
        use sonium_solvers::iterative::{
            BackendKrylovConfig, DeviceGmresSession, DeviceInitialGuess,
        };
        use sonium_solvers::{AmgConfig, AmgSmoother, BackendAmgPreconditioner};
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 26)).unwrap();
        for degree in [PolynomialDegree::P2, PolynomialDegree::P3] {
            let mesh = curved_mesh(degree);
            let space = H1Space::new(degree);
            let form = HelmholtzForm::from_real(5.0);
            let volume = prepare_hex_helmholtz(&mesh, &space, &form, 5).unwrap();
            let n = volume.dimension();
            let dofs = [0, 8, n - 1];
            let constrained = volume.clone().try_with_dirichlet(&dofs).unwrap();
            let a =
                prepare_hex_shifted_laplacian(&mesh, &space, Complex64::new(-25.0, 0.0), 5, &dofs)
                    .unwrap();
            let p =
                prepare_hex_shifted_laplacian(&mesh, &space, Complex64::new(-25.0, 25.0), 5, &dofs)
                    .unwrap();
            let operator = DeviceCsr::try_from_host(
                pool.backend(),
                n,
                n,
                &a.row_ptrs.iter().map(|&i| i as i32).collect::<Vec<_>>(),
                &a.col_indices.iter().map(|&i| i as i32).collect::<Vec<_>>(),
                &a.values,
            )
            .unwrap();
            let preconditioner = BackendAmgPreconditioner::<CpuBackend, Complex64>::try_from_csr(
                pool.backend(),
                &p,
                AmgConfig {
                    coarse_size: 4,
                    smoother: AmgSmoother::Jacobi,
                    jacobi_weight: 0.6,
                    ..Default::default()
                },
            )
            .unwrap();
            assert!(preconditioner.diagnostics().num_levels > 1);
            let mut session = DeviceGmresSession::try_new_flexible(&pool, n).unwrap();
            let mut rhs = pool.try_alloc::<Complex64>(n).unwrap();
            let config = BackendKrylovConfig {
                max_iterations: 300,
                restart: n,
                tolerance: 1e-11,
                require_native: true,
                exact_residual_check_interval: 0,
            };
            for source in 0..3 {
                let exact: Vec<_> = (0..n)
                    .map(|i| Complex64::new((i as f64 + source as f64).sin(), 0.2))
                    .collect();
                let mut b = vec![Complex64::default(); n];
                volume.apply(&exact, &mut b, Transpose::NoTrans).unwrap();
                let mut lifted = b.clone();
                constrained.lift_rhs(&b, &exact, &mut lifted).unwrap();
                pool.backend().try_copy_h2d(&lifted, rhs.raw_mut()).unwrap();
                let initial = if source == 0 {
                    DeviceInitialGuess::Zero
                } else {
                    DeviceInitialGuess::PreviousSolution
                };
                let status = session
                    .solve(
                        &operator,
                        &preconditioner,
                        rhs.raw(),
                        initial,
                        &config,
                        None,
                    )
                    .unwrap();
                assert!(status.converged, "{degree:?}: {status:?}");
                let mut actual = vec![Complex64::default(); n];
                pool.backend()
                    .try_copy_d2h(session.solution().unwrap(), &mut actual)
                    .unwrap();
                for (a, b) in actual.iter().zip(&exact) {
                    assert!((*a - *b).norm() < 1e-8);
                }
                // Certify against the independently prepared original volume operator.
                constrained
                    .apply(&actual, &mut b, Transpose::NoTrans)
                    .unwrap();
                let residual = b
                    .iter()
                    .zip(&lifted)
                    .map(|(a, b)| (*a - *b).norm_sqr())
                    .sum::<f64>()
                    .sqrt();
                let norm = lifted.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
                assert!(residual < 2e-10 * norm);
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn native_hex_persistent_solve() {
        use super::super::device_boundary::prepare_hex_robin;
        use sonium_compute::cuda::{CudaBackend, fem_hex::CudaHexPartialC64};
        use sonium_compute::operator_sum::DeviceSum;
        use sonium_compute::{
            AdjointDeviceCsr, Backend, DeviceOperator, HostCsrRef, MemoryPool, MemoryPoolConfig,
            PersistentDeviceCsr,
        };
        use sonium_solvers::iterative::{
            BackendKrylovConfig, DeviceGmresSession, DeviceInitialGuess,
        };
        use sonium_solvers::{AmgConfig, AmgSmoother, BackendAmgPreconditioner};
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                assert!(
                    std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                    "CUDA required: {e}"
                );
                eprintln!("SKIP FEM hex persistent solve: {e}");
                return;
            }
        };
        let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 26)).unwrap();
        for degree in [PolynomialDegree::P2, PolynomialDegree::P3] {
            let mesh = curved_mesh(degree);
            let space = H1Space::new(degree);
            // Solve the original indefinite operator; only the preconditioner is shifted.
            let form = HelmholtzForm::from_real(5.0);
            let prepared = prepare_hex_helmholtz(&mesh, &space, &form, 5).unwrap();
            let dofs = [0, 8, prepared.dimension() - 1];
            let prepared = prepared.try_with_dirichlet(&dofs).unwrap();
            let volume = CudaHexPartialC64::try_new(pool.clone(), &prepared, 1).unwrap();
            let boundary = prepare_hex_robin(
                &mesh,
                &space,
                mesh.boundaries[0].marker,
                &|_, _, _| Complex64::new(0.4, 0.3),
                &|_, _, _| Complex64::default(),
                5,
                &dofs,
            )
            .unwrap();
            let matrix = &boundary.matrix;
            let device_facet = AdjointDeviceCsr::try_from_host_csr_ref(
                pool.backend(),
                HostCsrRef::try_new(
                    matrix.num_rows,
                    matrix.num_cols,
                    &matrix.row_ptrs,
                    &matrix.col_indices,
                    &matrix.values,
                )
                .unwrap(),
            )
            .unwrap();
            let facet = device_facet.try_prebake(pool.backend()).unwrap();
            let coupling = &boundary.dirichlet_coupling;
            let coupling_device = PersistentDeviceCsr::try_from_host_csr_ref(
                pool.backend(),
                HostCsrRef::try_new(
                    coupling.num_rows,
                    coupling.num_cols,
                    &coupling.row_ptrs,
                    &coupling.col_indices,
                    &coupling.values,
                )
                .unwrap(),
            )
            .unwrap();
            let op = DeviceSum::try_new(&pool, &volume, &facet, &[]).unwrap();
            let shifted =
                prepare_hex_shifted_laplacian(&mesh, &space, Complex64::new(-25.0, 25.0), 5, &dofs)
                    .unwrap();
            let mut terms = Vec::new();
            for row in 0..shifted.num_rows {
                for j in shifted.row_ptrs[row]..shifted.row_ptrs[row + 1] {
                    terms.push((row, shifted.col_indices[j], shifted.values[j]));
                }
                for j in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
                    terms.push((row, matrix.col_indices[j], matrix.values[j]));
                }
            }
            let shifted =
                sonium_solvers::CsrMatrix::from_triplets(shifted.num_rows, shifted.num_cols, terms);
            let preconditioner = BackendAmgPreconditioner::<CudaBackend,Complex64>::try_from_csr_with_cuda_coarse_lu(pool.backend(), &shifted, AmgConfig {
                coarse_size: 4, smoother: AmgSmoother::Jacobi, jacobi_weight: 0.6, ..Default::default()
            }).unwrap();
            assert!(preconditioner.diagnostics().has_coarse_direct_solver);
            let csr = assemble_bilinear_checked(&form, &space, &space, &mesh, 5)
                .unwrap()
                .to_csr();
            let n = prepared.dimension();
            let mut session = DeviceGmresSession::try_new_flexible(&pool, n).unwrap();
            let mut rhs = pool.try_alloc::<Complex64>(n).unwrap();
            let mut original_rhs = pool.try_alloc::<Complex64>(n).unwrap();
            let mut prescribed = pool.try_alloc::<Complex64>(n).unwrap();
            let mut boundary_lift = pool.try_alloc::<Complex64>(n).unwrap();
            let config = BackendKrylovConfig {
                max_iterations: 300,
                restart: n,
                tolerance: 1e-11,
                require_native: true,
                exact_residual_check_interval: 0,
            };
            for source in 0..3 {
                let exact: Vec<_> = (0..n)
                    .map(|i| Complex64::new((i as f64 + source as f64).sin(), 0.2))
                    .collect();
                let mut b = vec![Complex64::default(); n];
                for (row, entry) in b.iter_mut().enumerate() {
                    for j in csr.row_ptrs[row]..csr.row_ptrs[row + 1] {
                        *entry += csr.values[j] * exact[csr.col_indices[j]];
                    }
                }
                for (row, value) in b.iter_mut().enumerate() {
                    for boundary_matrix in [matrix, coupling] {
                        for j in boundary_matrix.row_ptrs[row]..boundary_matrix.row_ptrs[row + 1] {
                            *value +=
                                boundary_matrix.values[j] * exact[boundary_matrix.col_indices[j]];
                        }
                    }
                }
                pool.backend()
                    .try_copy_h2d(&b, original_rhs.raw_mut())
                    .unwrap();
                pool.backend()
                    .try_copy_h2d(&exact, prescribed.raw_mut())
                    .unwrap();
                volume
                    .try_lift_rhs_colmajor(
                        pool.backend(),
                        1,
                        original_rhs.raw(),
                        prescribed.raw(),
                        rhs.raw_mut(),
                    )
                    .unwrap();
                coupling_device
                    .try_apply(pool.backend(), prescribed.raw(), boundary_lift.raw_mut())
                    .unwrap();
                pool.backend()
                    .try_axpy(
                        Complex64::new(-1.0, 0.0),
                        boundary_lift.raw(),
                        rhs.raw_mut(),
                    )
                    .unwrap();
                let guess = if source == 0 {
                    DeviceInitialGuess::Zero
                } else {
                    DeviceInitialGuess::PreviousSolution
                };
                let status = session
                    .solve(&op, &preconditioner, rhs.raw(), guess, &config, None)
                    .unwrap();
                assert!(status.converged);
                let mut actual = vec![Complex64::default(); n];
                pool.backend()
                    .try_copy_d2h(session.solution().unwrap(), &mut actual)
                    .unwrap();
                for (a, b) in actual.iter().zip(exact) {
                    assert!((*a - b).norm() < 1e-8);
                }
            }
        }
    }
}
