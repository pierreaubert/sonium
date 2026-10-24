//! Prepare affine tetrahedral acoustic volume operators for device execution.
//!
//! The prepared operator uses FEM's nodal ordering and represents `K - k² M`.
//! Natural homogeneous Neumann conditions need no extra action. Use the prepared
//! operator's `try_with_dirichlet` with H1 boundary DOFs for prescribed pressure,
//! then lift each RHS before solving. Impedance, radiation and sources are separate. Preparing
//! a volume operator does not admit a complete workload through generic dispatch.

// Rust guideline compliant 2026-02-21
use super::HelmholtzForm;
use crate::basis::PolynomialDegree;
use crate::mesh::{ElementType, Mesh};
use crate::spaces::{FunctionSpace, H1Space};
use num_complex::Complex64;
use sonium_compute::fem_tet::{TetP1AcousticC64, TetP2AcousticC64};
use sonium_compute::{BackendError, BackendResult};

fn invalid(reason: impl Into<String>) -> BackendError {
    BackendError::InvalidArgument {
        operation: "prepare_tet_helmholtz",
        reason: reason.into(),
    }
}

/// Prepare a complex Helmholtz volume operator from FEM mesh and space inputs.
///
/// Retain the result for repeated actions. With CUDA enabled, upload it once
/// using `sonium_compute::cuda::fem_tet::CudaTetP1C64::try_new` and pass that
/// operator to a retained device Krylov session. The assembled preconditioning
/// operator and boundary terms remain separate.
///
/// # Errors
/// Rejects non-3D meshes, non-P1 spaces, non-affine tetrahedra, invalid nodal
/// numbering, non-finite coefficients, degenerate geometry and index overflow.
pub fn prepare_tet_p1_helmholtz(
    mesh: &Mesh,
    space: &H1Space,
    form: &HelmholtzForm,
) -> BackendResult<TetP1AcousticC64> {
    if mesh.dimension != 3 || space.degree != PolynomialDegree::P1 {
        return Err(invalid("requires a 3D H1 P1 space"));
    }
    if mesh
        .elements
        .iter()
        .any(|element| element.element_type != ElementType::Tetrahedron || element.nodes.len() != 4)
    {
        return Err(invalid("requires affine four-node tetrahedra"));
    }
    space
        .validate(mesh)
        .map_err(|error| invalid(error.to_string()))?;
    if space.num_global_dofs(mesh) != mesh.nodes.len() {
        return Err(invalid("requires one nodal degree of freedom per vertex"));
    }
    let vertices: Vec<_> = mesh.nodes.iter().map(|p| [p.x, p.y, p.z]).collect();
    let mut elements = Vec::with_capacity(mesh.elements.len());
    for (index, element) in mesh.elements.iter().enumerate() {
        if space.local_to_global(mesh, index) != element.nodes {
            return Err(invalid(
                "space ordering differs from tetrahedron vertex ordering",
            ));
        }
        elements.push([
            element.nodes[0],
            element.nodes[1],
            element.nodes[2],
            element.nodes[3],
        ]);
    }
    let diffusion = vec![Complex64::new(1.0, 0.0); elements.len()];
    let mass = vec![-form.wavenumber * form.wavenumber; elements.len()];
    TetP1AcousticC64::try_new(&vertices, &elements, &diffusion, &mass)
}

/// Prepare a P2 Helmholtz volume operator on affine tetrahedra.
///
/// Uses H1's canonical global vertex/edge numbering and midpoint coordinates.
/// Upload once with `CudaTetP2C64::try_new` when CUDA is enabled. Dirichlet sets
/// must include the prescribed edge DOFs; use `H1Space::boundary_dofs`.
///
/// # Errors
/// Rejects non-P2 spaces, non-3D or non-affine tetrahedra, inconsistent DOF
/// layouts, non-finite coefficients, invalid geometry and index overflow.
pub fn prepare_tet_p2_helmholtz(
    mesh: &Mesh,
    space: &H1Space,
    form: &HelmholtzForm,
) -> BackendResult<TetP2AcousticC64> {
    if mesh.dimension != 3 || space.degree != PolynomialDegree::P2 {
        return Err(invalid("requires a 3D H1 P2 space"));
    }
    if mesh
        .elements
        .iter()
        .any(|e| e.element_type != ElementType::Tetrahedron || e.nodes.len() != 4)
    {
        return Err(invalid(
            "P2 device operator currently requires four-node affine geometry",
        ));
    }
    space.validate(mesh).map_err(|e| invalid(e.to_string()))?;
    let coordinates = space
        .dof_coordinates(mesh)
        .ok_or_else(|| invalid("missing P2 DOF coordinates"))?;
    if coordinates.len() != space.num_global_dofs(mesh) {
        return Err(invalid("P2 coordinate/DOF mismatch"));
    }
    let elements: Vec<[usize; 10]> = (0..mesh.elements.len())
        .map(|index| {
            let dofs = space.local_to_global(mesh, index);
            if dofs.get(..4) != Some(mesh.elements[index].nodes.as_slice()) {
                return Err(invalid("P2 vertex ordering differs from geometry"));
            }
            dofs.try_into()
                .map_err(|_| invalid("expected ten P2 tetrahedral DOFs"))
        })
        .collect::<BackendResult<_>>()?;
    TetP2AcousticC64::try_new(
        &coordinates,
        &elements,
        &vec![Complex64::new(1.0, 0.0); elements.len()],
        &vec![-form.wavenumber * form.wavenumber; elements.len()],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembly::assemble_bilinear_checked;
    use crate::mesh::{unit_cube_tetrahedra, unit_square_triangles};
    use sonium_compute::Transpose;

    #[test]
    fn tet_volume_matches_fem_quadrature_for_all_actions() {
        let mut mesh = unit_cube_tetrahedra(2);
        // A skew affine map exercises physical gradient transforms.
        for p in &mut mesh.nodes {
            p.x = 1.3 * p.x + 0.2 * p.y;
            p.y = 0.8 * p.y + 0.1 * p.z;
            p.z *= 1.7;
        }
        let space = H1Space::new(PolynomialDegree::P1);
        for k in [Complex64::new(0.0, 0.0), Complex64::new(3.2, -0.4)] {
            let form = HelmholtzForm::new(k);
            let op = prepare_tet_p1_helmholtz(&mesh, &space, &form).unwrap();
            let csr = assemble_bilinear_checked(&form, &space, &space, &mesh, 4)
                .unwrap()
                .to_csr();
            let (pattern, values) = op
                .assemble_csr(Complex64::new(1., 0.), Complex64::new(1., 0.))
                .unwrap();
            assert_eq!(pattern.row_ptrs().len(), csr.num_rows + 1);
            for row in 0..csr.num_rows {
                for j in pattern.row_ptrs()[row] as usize..pattern.row_ptrs()[row + 1] as usize {
                    let col = pattern.columns()[j] as usize;
                    let expected: Complex64 = (csr.row_ptrs[row]..csr.row_ptrs[row + 1])
                        .filter(|&k| csr.col_indices[k] == col)
                        .map(|k| csr.values[k])
                        .sum();
                    assert!((values[j] - expected).norm() < 2e-12 * (1. + expected.norm()));
                }
            }
            let stiffness =
                assemble_bilinear_checked(&HelmholtzForm::from_real(0.), &space, &space, &mesh, 4)
                    .unwrap()
                    .to_csr();
            let ds = Complex64::new(1.2, 0.1);
            let ms = Complex64::new(1., 0.5);
            let (shifted_pattern, shifted) = op.assemble_csr(ds, ms).unwrap();
            assert_eq!(shifted_pattern.row_ptrs(), pattern.row_ptrs());
            assert_eq!(shifted_pattern.columns(), pattern.columns());
            for row in 0..csr.num_rows {
                for j in pattern.row_ptrs()[row] as usize..pattern.row_ptrs()[row + 1] as usize {
                    let col = pattern.columns()[j] as usize;
                    let k_entry: Complex64 = (stiffness.row_ptrs[row]..stiffness.row_ptrs[row + 1])
                        .filter(|&j| stiffness.col_indices[j] == col)
                        .map(|j| stiffness.values[j])
                        .sum();
                    let expected = ds * k_entry + ms * (values[j] - k_entry);
                    assert!((shifted[j] - expected).norm() < 2e-12 * (1. + expected.norm()));
                }
            }
            let diagonal = op.diagonal();
            for (row, actual) in diagonal.iter().enumerate() {
                let expected: Complex64 = (csr.row_ptrs[row]..csr.row_ptrs[row + 1])
                    .filter(|&j| csr.col_indices[j] == row)
                    .map(|j| csr.values[j])
                    .sum();
                assert!((*actual - expected).norm() < 2e-12 * (1.0 + expected.norm()));
            }
            let x: Vec<_> = (0..mesh.nodes.len())
                .map(|i| Complex64::new((i as f64 * 0.7).sin(), (i as f64 * 0.3).cos()))
                .collect();
            for action in [Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans] {
                let mut expected = vec![Complex64::default(); x.len()];
                for row in 0..csr.num_rows {
                    for offset in csr.row_ptrs[row]..csr.row_ptrs[row + 1] {
                        let col = csr.col_indices[offset];
                        let value = csr.values[offset];
                        match action {
                            Transpose::NoTrans => expected[row] += value * x[col],
                            Transpose::Trans => expected[col] += value * x[row],
                            Transpose::ConjTrans => expected[col] += value.conj() * x[row],
                        }
                    }
                }
                let mut actual = vec![Complex64::default(); x.len()];
                op.apply(&x, &mut actual, action).unwrap();
                for (a, b) in actual.iter().zip(&expected) {
                    assert!((*a - *b).norm() < 2e-12 * (1.0 + b.norm()), "{a} != {b}");
                }
            }
        }
    }

    #[test]
    fn dirichlet_actions_and_lifting_match_fem_boundary_elimination() {
        use crate::boundary::{FemSystem, try_apply_dirichlet_values_complex};
        let mesh = unit_cube_tetrahedra(2);
        let space = H1Space::new(PolynomialDegree::P1);
        let marker = mesh.boundaries[0].marker;
        let dofs = space.boundary_dofs(&mesh, marker);
        assert!(!dofs.is_empty() && dofs.len() < mesh.nodes.len());
        let form = HelmholtzForm::new(Complex64::new(2.3, -0.4));
        let volume = prepare_tet_p1_helmholtz(&mesh, &space, &form).unwrap();
        let op = volume.try_with_dirichlet(&dofs).unwrap();
        let matrix = assemble_bilinear_checked(&form, &space, &space, &mesh, 4).unwrap();
        let b: Vec<_> = (0..mesh.nodes.len())
            .map(|i| Complex64::new(i as f64 * 0.1, -0.2))
            .collect();
        for shift in [0.0, 0.7] {
            let value = |x: f64, y: f64, z: f64| Complex64::new(0.2 + x + shift, y - 0.3 * z);
            // Free-node values deliberately nonzero: lifting must ignore them.
            let prescribed: Vec<_> = mesh.nodes.iter().map(|p| value(p.x, p.y, p.z)).collect();
            let mut system = FemSystem::new(matrix.clone(), b.clone());
            try_apply_dirichlet_values_complex(&mut system, &mesh, &space, marker, &value, None)
                .unwrap();
            let mut lifted = vec![Complex64::default(); b.len()];
            op.lift_rhs(&b, &prescribed, &mut lifted).unwrap();
            for (a, expected) in lifted.iter().zip(&system.rhs) {
                assert!((*a - *expected).norm() < 2e-12 * (1.0 + expected.norm()));
            }
            let csr = system.matrix.to_csr();
            let (pattern, values) = op
                .assemble_csr(Complex64::new(1., 0.), Complex64::new(1., 0.))
                .unwrap();
            for row in 0..csr.num_rows {
                for j in pattern.row_ptrs()[row] as usize..pattern.row_ptrs()[row + 1] as usize {
                    let col = pattern.columns()[j] as usize;
                    let expected: Complex64 = (csr.row_ptrs[row]..csr.row_ptrs[row + 1])
                        .filter(|&k| csr.col_indices[k] == col)
                        .map(|k| csr.values[k])
                        .sum();
                    assert!((values[j] - expected).norm() < 2e-12 * (1. + expected.norm()));
                }
            }

            for action in [Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans] {
                let mut expected = vec![Complex64::default(); b.len()];
                for row in 0..csr.num_rows {
                    for j in csr.row_ptrs[row]..csr.row_ptrs[row + 1] {
                        let col = csr.col_indices[j];
                        match action {
                            Transpose::NoTrans => expected[row] += csr.values[j] * b[col],
                            Transpose::Trans => expected[col] += csr.values[j] * b[row],
                            Transpose::ConjTrans => expected[col] += csr.values[j].conj() * b[row],
                        }
                    }
                }
                let mut actual = vec![Complex64::default(); b.len()];
                op.apply(&b, &mut actual, action).unwrap();
                for (a, expected) in actual.iter().zip(expected) {
                    assert!((*a - expected).norm() < 2e-12 * (1.0 + expected.norm()));
                }
            }
            let diagonal = op.diagonal();
            for &i in &dofs {
                assert_eq!(diagonal[i], Complex64::new(1.0, 0.0));
            }
        }
    }

    #[test]
    fn p2_volume_matches_fem_quadrature_for_all_actions() {
        let mut mesh = unit_cube_tetrahedra(2);
        // A skew affine map exercises physical gradient transforms.
        for p in &mut mesh.nodes {
            p.x = 1.3 * p.x + 0.2 * p.y;
            p.y = 0.8 * p.y + 0.1 * p.z;
            p.z *= 1.7;
        }
        mesh.elements[0].nodes.swap(0, 1);
        let space = H1Space::new(PolynomialDegree::P2);
        for k in [Complex64::new(0.0, 0.0), Complex64::new(3.2, -0.4)] {
            let form = HelmholtzForm::new(k);
            let op = prepare_tet_p2_helmholtz(&mesh, &space, &form).unwrap();
            let csr = assemble_bilinear_checked(&form, &space, &space, &mesh, 4)
                .unwrap()
                .to_csr();
            let (pattern, values) = op
                .assemble_csr(Complex64::new(1., 0.), Complex64::new(1., 0.))
                .unwrap();
            assert_eq!(pattern.row_ptrs().len(), csr.num_rows + 1);
            for row in 0..csr.num_rows {
                for j in pattern.row_ptrs()[row] as usize..pattern.row_ptrs()[row + 1] as usize {
                    let col = pattern.columns()[j] as usize;
                    let expected: Complex64 = (csr.row_ptrs[row]..csr.row_ptrs[row + 1])
                        .filter(|&k| csr.col_indices[k] == col)
                        .map(|k| csr.values[k])
                        .sum();
                    assert!((values[j] - expected).norm() < 2e-12 * (1. + expected.norm()));
                }
            }
            let stiffness =
                assemble_bilinear_checked(&HelmholtzForm::from_real(0.), &space, &space, &mesh, 4)
                    .unwrap()
                    .to_csr();
            let ds = Complex64::new(1.2, 0.1);
            let ms = Complex64::new(1., 0.5);
            let (shifted_pattern, shifted) = op.assemble_csr(ds, ms).unwrap();
            assert_eq!(shifted_pattern.row_ptrs(), pattern.row_ptrs());
            assert_eq!(shifted_pattern.columns(), pattern.columns());
            for row in 0..csr.num_rows {
                for j in pattern.row_ptrs()[row] as usize..pattern.row_ptrs()[row + 1] as usize {
                    let col = pattern.columns()[j] as usize;
                    let k_entry: Complex64 = (stiffness.row_ptrs[row]..stiffness.row_ptrs[row + 1])
                        .filter(|&j| stiffness.col_indices[j] == col)
                        .map(|j| stiffness.values[j])
                        .sum();
                    let expected = ds * k_entry + ms * (values[j] - k_entry);
                    assert!((shifted[j] - expected).norm() < 2e-12 * (1. + expected.norm()));
                }
            }
            let diagonal = op.diagonal();
            for (row, actual) in diagonal.iter().enumerate() {
                let expected: Complex64 = (csr.row_ptrs[row]..csr.row_ptrs[row + 1])
                    .filter(|&j| csr.col_indices[j] == row)
                    .map(|j| csr.values[j])
                    .sum();
                assert!((*actual - expected).norm() < 2e-12 * (1.0 + expected.norm()));
            }
            let x: Vec<_> = (0..space.num_global_dofs(&mesh))
                .map(|i| Complex64::new((i as f64 * 0.7).sin(), (i as f64 * 0.3).cos()))
                .collect();
            for action in [Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans] {
                let mut expected = vec![Complex64::default(); x.len()];
                for row in 0..csr.num_rows {
                    for offset in csr.row_ptrs[row]..csr.row_ptrs[row + 1] {
                        let col = csr.col_indices[offset];
                        let value = csr.values[offset];
                        match action {
                            Transpose::NoTrans => expected[row] += value * x[col],
                            Transpose::Trans => expected[col] += value * x[row],
                            Transpose::ConjTrans => expected[col] += value.conj() * x[row],
                        }
                    }
                }
                let mut actual = vec![Complex64::default(); x.len()];
                op.apply(&x, &mut actual, action).unwrap();
                for (a, b) in actual.iter().zip(&expected) {
                    assert!((*a - *b).norm() < 2e-12 * (1.0 + b.norm()), "{a} != {b}");
                }
            }
        }
    }

    #[test]
    fn p2_dirichlet_actions_and_lifting_match_fem_boundary_elimination() {
        use crate::boundary::{FemSystem, try_apply_dirichlet_values_complex};
        let mesh = unit_cube_tetrahedra(2);
        let space = H1Space::new(PolynomialDegree::P2);
        let marker = mesh.boundaries[0].marker;
        let dofs = space.boundary_dofs(&mesh, marker);
        assert!(!dofs.is_empty() && dofs.len() < space.num_global_dofs(&mesh));
        let form = HelmholtzForm::new(Complex64::new(2.3, -0.4));
        let volume = prepare_tet_p2_helmholtz(&mesh, &space, &form).unwrap();
        let op = volume.try_with_dirichlet(&dofs).unwrap();
        let matrix = assemble_bilinear_checked(&form, &space, &space, &mesh, 4).unwrap();
        let b: Vec<_> = (0..space.num_global_dofs(&mesh))
            .map(|i| Complex64::new(i as f64 * 0.1, -0.2))
            .collect();
        for shift in [0.0, 0.7] {
            let value = |x: f64, y: f64, z: f64| Complex64::new(0.2 + x + shift, y - 0.3 * z);
            // Free-node values deliberately nonzero: lifting must ignore them.
            let prescribed: Vec<_> = space
                .dof_coordinates(&mesh)
                .unwrap()
                .iter()
                .map(|p| value(p[0], p[1], p[2]))
                .collect();
            let mut system = FemSystem::new(matrix.clone(), b.clone());
            try_apply_dirichlet_values_complex(&mut system, &mesh, &space, marker, &value, None)
                .unwrap();
            let mut lifted = vec![Complex64::default(); b.len()];
            op.lift_rhs(&b, &prescribed, &mut lifted).unwrap();
            for (a, expected) in lifted.iter().zip(&system.rhs) {
                assert!((*a - *expected).norm() < 2e-12 * (1.0 + expected.norm()));
            }
            let csr = system.matrix.to_csr();
            let (pattern, values) = op
                .assemble_csr(Complex64::new(1., 0.), Complex64::new(1., 0.))
                .unwrap();
            for row in 0..csr.num_rows {
                for j in pattern.row_ptrs()[row] as usize..pattern.row_ptrs()[row + 1] as usize {
                    let col = pattern.columns()[j] as usize;
                    let expected: Complex64 = (csr.row_ptrs[row]..csr.row_ptrs[row + 1])
                        .filter(|&k| csr.col_indices[k] == col)
                        .map(|k| csr.values[k])
                        .sum();
                    assert!((values[j] - expected).norm() < 2e-12 * (1. + expected.norm()));
                }
            }

            for action in [Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans] {
                let mut expected = vec![Complex64::default(); b.len()];
                for row in 0..csr.num_rows {
                    for j in csr.row_ptrs[row]..csr.row_ptrs[row + 1] {
                        let col = csr.col_indices[j];
                        match action {
                            Transpose::NoTrans => expected[row] += csr.values[j] * b[col],
                            Transpose::Trans => expected[col] += csr.values[j] * b[row],
                            Transpose::ConjTrans => expected[col] += csr.values[j].conj() * b[row],
                        }
                    }
                }
                let mut actual = vec![Complex64::default(); b.len()];
                op.apply(&b, &mut actual, action).unwrap();
                for (a, expected) in actual.iter().zip(expected) {
                    assert!((*a - expected).norm() < 2e-12 * (1.0 + expected.norm()));
                }
            }
            let diagonal = op.diagonal();
            for &i in &dofs {
                assert_eq!(diagonal[i], Complex64::new(1.0, 0.0));
            }
        }
    }

    #[test]
    fn rejects_unsupported_or_invalid_fem_inputs() {
        let mesh = unit_cube_tetrahedra(1);
        let p1 = H1Space::new(PolynomialDegree::P1);
        let form = HelmholtzForm::from_real(2.0);
        assert!(
            prepare_tet_p1_helmholtz(&mesh, &H1Space::new(PolynomialDegree::P2), &form).is_err()
        );
        assert!(prepare_tet_p1_helmholtz(&unit_square_triangles(1), &p1, &form).is_err());
        let mut malformed = mesh.clone();
        malformed.elements[0].nodes[0] = mesh.nodes.len();
        assert!(prepare_tet_p1_helmholtz(&malformed, &p1, &form).is_err());
        let mut curved = mesh.clone();
        curved.elements[0].nodes.push(0);
        assert!(prepare_tet_p1_helmholtz(&curved, &p1, &form).is_err());
        assert!(prepare_tet_p1_helmholtz(&mesh, &p1, &HelmholtzForm::from_real(f64::NAN)).is_err());
    }
    #[cfg(feature = "cuda")]
    #[test]
    fn native_tet_volume_persistent_solve() {
        use sonium_compute::cuda::{CudaBackend, fem_tet::CudaTetP1C64};
        use sonium_compute::{Backend, MemoryPool, MemoryPoolConfig, PointJacobiBackend};
        use sonium_solvers::iterative::{
            BackendKrylovConfig, DeviceGmresSession, DeviceInitialGuess,
        };
        let backend = match CudaBackend::new(0) {
            Ok(backend) => backend,
            Err(error) => {
                assert!(
                    std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                    "CUDA required: {error}"
                );
                eprintln!("SKIP FEM tetrahedral solve: {error}");
                return;
            }
        };
        let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 26)).unwrap();
        let mesh = unit_cube_tetrahedra(2);
        let space = H1Space::new(PolynomialDegree::P1);
        // Use lossy acoustics with nonzero prescribed pressure on one marked face.
        let form = HelmholtzForm::new(Complex64::new(1.0, -1.0));
        let dofs = space.boundary_dofs(&mesh, mesh.boundaries[0].marker);
        let prepared = prepare_tet_p1_helmholtz(&mesh, &space, &form)
            .unwrap()
            .try_with_dirichlet(&dofs)
            .unwrap();
        let op = CudaTetP1C64::try_new(pool.clone(), &prepared).unwrap();
        let csr = assemble_bilinear_checked(&form, &space, &space, &mesh, 4)
            .unwrap()
            .to_csr();
        let n = mesh.nodes.len();
        let diagonal = prepared.diagonal();
        let preconditioner =
            PointJacobiBackend::try_from_diagonal(pool.backend(), &diagonal).unwrap();
        let mut session = DeviceGmresSession::try_new(&pool, n).unwrap();
        let mut rhs = pool.try_alloc::<Complex64>(n).unwrap();
        let mut original_rhs = pool.try_alloc::<Complex64>(n).unwrap();
        let mut prescribed = pool.try_alloc::<Complex64>(n).unwrap();
        let config = BackendKrylovConfig {
            max_iterations: 200,
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
            pool.backend()
                .try_copy_h2d(&b, original_rhs.raw_mut())
                .unwrap();
            pool.backend()
                .try_copy_h2d(&exact, prescribed.raw_mut())
                .unwrap();
            op.try_lift_rhs_colmajor(
                pool.backend(),
                1,
                original_rhs.raw(),
                prescribed.raw(),
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
            for (a, x) in actual.iter().zip(exact) {
                assert!((*a - x).norm() < 1e-8);
            }
        }
    }
    #[cfg(feature = "cuda")]
    #[test]
    fn native_tet_p2_persistent_solve() {
        use sonium_compute::cuda::{CudaBackend, fem_tet::CudaTetP2C64};
        use sonium_compute::{Backend, MemoryPool, MemoryPoolConfig, PointJacobiBackend};
        use sonium_solvers::iterative::{
            BackendKrylovConfig, DeviceGmresSession, DeviceInitialGuess,
        };
        let backend = match CudaBackend::new(0) {
            Ok(backend) => backend,
            Err(error) => {
                assert!(
                    std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                    "CUDA required: {error}"
                );
                eprintln!("SKIP FEM tetrahedral solve: {error}");
                return;
            }
        };
        let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 26)).unwrap();
        let mesh = unit_cube_tetrahedra(2);
        let space = H1Space::new(PolynomialDegree::P2);
        // Use lossy acoustics with nonzero prescribed pressure on one marked face.
        let form = HelmholtzForm::new(Complex64::new(1.0, -1.0));
        let dofs = space.boundary_dofs(&mesh, mesh.boundaries[0].marker);
        let prepared = prepare_tet_p2_helmholtz(&mesh, &space, &form)
            .unwrap()
            .try_with_dirichlet(&dofs)
            .unwrap();
        let op = CudaTetP2C64::try_new(pool.clone(), &prepared).unwrap();
        let csr = assemble_bilinear_checked(&form, &space, &space, &mesh, 4)
            .unwrap()
            .to_csr();
        let n = space.num_global_dofs(&mesh);
        let diagonal = prepared.diagonal();
        let preconditioner =
            PointJacobiBackend::try_from_diagonal(pool.backend(), &diagonal).unwrap();
        let mut session = DeviceGmresSession::try_new(&pool, n).unwrap();
        let mut rhs = pool.try_alloc::<Complex64>(n).unwrap();
        let mut original_rhs = pool.try_alloc::<Complex64>(n).unwrap();
        let mut prescribed = pool.try_alloc::<Complex64>(n).unwrap();
        let config = BackendKrylovConfig {
            max_iterations: 200,
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
            pool.backend()
                .try_copy_h2d(&b, original_rhs.raw_mut())
                .unwrap();
            pool.backend()
                .try_copy_h2d(&exact, prescribed.raw_mut())
                .unwrap();
            op.try_lift_rhs_colmajor(
                pool.backend(),
                1,
                original_rhs.raw(),
                prescribed.raw(),
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
            for (a, x) in actual.iter().zip(exact) {
                assert!((*a - x).norm() < 1e-8);
            }
        }
    }
}
