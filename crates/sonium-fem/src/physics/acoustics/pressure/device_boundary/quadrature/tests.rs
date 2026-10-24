//! Numeric boundary maps checked against direct quadrature and retained CUDA CSR.

// Rust guideline compliant 2026-02-21
use super::*;
use crate::basis::PolynomialDegree;
use crate::mesh::{ElementType, Point, unit_cube_hexahedra};
use crate::physics::acoustics::pressure::device_boundary::prepare_hex_robin;
use crate::physics::acoustics::pressure::device_hex::populate_hex_test_boundaries;
use crate::spaces::FunctionSpace;

fn alpha(x: f64, y: f64, z: f64) -> Complex64 {
    Complex64::new(0.7 + x, 0.3 + y * z)
}
fn source(x: f64, y: f64, z: f64) -> Complex64 {
    Complex64::new(0.2 + x * y, -z)
}
fn close(a: Complex64, b: Complex64) {
    assert!((a - b).norm() < 3e-12 * (1.0 + b.norm()), "{a} != {b}");
}
fn apply(matrix: &CsrMatrix<Complex64>, x: &[Complex64]) -> Vec<Complex64> {
    (0..matrix.num_rows)
        .map(|row| {
            (matrix.row_ptrs[row]..matrix.row_ptrs[row + 1])
                .map(|j| matrix.values[j] * x[matrix.col_indices[j]])
                .sum()
        })
        .collect()
}
fn marked_mesh(degree: PolynomialDegree, curved: bool) -> Mesh {
    let base = unit_cube_hexahedra(2);
    let mut mesh = if curved {
        let space = H1Space::new(degree);
        let mut mesh = Mesh::new(3);
        for [x, y, z] in space.dof_coordinates(&base).unwrap() {
            mesh.add_node(Point::new_3d(x, y, z + 0.2 * x * x));
        }
        for i in 0..base.elements.len() {
            mesh.add_element(ElementType::Hexahedron, space.local_to_global(&base, i));
        }
        mesh.boundaries = base.boundaries.clone();
        populate_hex_test_boundaries(&mut mesh, degree);
        mesh
    } else {
        base
    };
    for face in &mut mesh.boundaries {
        face.marker = if face.nodes.iter().all(|&i| mesh.nodes[i].x == 0.0) {
            1
        } else {
            2
        };
    }
    mesh
}

#[test]
fn fixed_boundary_maps_match_direct_assembly_with_constraints_and_changed_data() {
    for degree in [
        PolynomialDegree::P1,
        PolynomialDegree::P2,
        PolynomialDegree::P3,
    ] {
        for curved in [false, true] {
            let mesh = marked_mesh(degree, curved);
            let space = H1Space::new(degree);
            let n = space.dof_coordinates(&mesh).unwrap().len();
            for dirichlet in [vec![], space.boundary_dofs(&mesh, 1), (0..n).collect()] {
                let plan = prepare_hex_robin_quadrature(&mesh, &space, 2, 4, &dirichlet).unwrap();
                for scale in [0.0, 1.0, -0.3] {
                    let a: Vec<_> = plan
                        .points()
                        .iter()
                        .map(|p| scale * alpha(p[0], p[1], p[2]))
                        .collect();
                    let g: Vec<_> = plan
                        .points()
                        .iter()
                        .map(|p| scale * source(p[0], p[1], p[2]))
                        .collect();
                    let reference = prepare_hex_robin(
                        &mesh,
                        &space,
                        2,
                        &|x, y, z| scale * alpha(x, y, z),
                        &|x, y, z| scale * source(x, y, z),
                        4,
                        &dirichlet,
                    )
                    .unwrap();
                    let input: Vec<_> = (0..n)
                        .map(|i| Complex64::new((0.17 * i as f64).sin(), (0.13 * i as f64).cos()))
                        .collect();
                    for (map, expected) in [
                        (plan.action(), &reference.matrix),
                        (plan.coupling(), &reference.dirichlet_coupling),
                    ] {
                        let mut assembled = map.template().clone();
                        assembled.values = map.values().try_assemble(&a).unwrap();
                        for (got, want) in apply(&assembled, &input)
                            .into_iter()
                            .zip(apply(expected, &input))
                        {
                            close(got, want);
                        }
                    }
                    for (got, want) in plan
                        .load()
                        .try_assemble(&g)
                        .unwrap()
                        .into_iter()
                        .zip(reference.rhs)
                    {
                        close(got, want);
                    }
                }
            }
            assert!(prepare_hex_robin_quadrature(&mesh, &space, 999, 4, &[]).is_err());
            assert!(prepare_hex_robin_quadrature(&mesh, &space, 2, 4, &[n]).is_err());
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn native_boundary_updates_feed_retained_sparse_operators_without_host_staging() {
    use sonium_compute::cuda::{CudaBackend, quadrature_assembly::CudaQuadratureAssembly};
    use sonium_compute::{Backend, MemoryPool, MemoryPoolConfig, PersistentDeviceCsr};
    let backend = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            assert!(
                std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                "CUDA required: {e}"
            );
            eprintln!("SKIP CUDA numeric boundary: {e}");
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 28)).unwrap();
    let b = pool.backend();
    for degree in [PolynomialDegree::P2, PolynomialDegree::P3] {
        let mesh = marked_mesh(degree, true);
        let space = H1Space::new(degree);
        let dirichlet = space.boundary_dofs(&mesh, 1);
        let plan = prepare_hex_robin_quadrature(&mesh, &space, 2, 4, &dirichlet).unwrap();
        let n = plan.load().output_count();
        let input: Vec<_> = (0..n)
            .map(|i| Complex64::new((0.13 * i as f64).sin(), -0.3))
            .collect();
        let mut x = pool.try_alloc::<Complex64>(n).unwrap();
        let mut y = pool.try_alloc::<Complex64>(n).unwrap();
        let mut samples = pool.try_alloc::<Complex64>(plan.points().len()).unwrap();
        b.try_copy_h2d(&input, x.raw_mut()).unwrap();
        for (map, is_coupling) in [(plan.action(), false), (plan.coupling(), true)] {
            let device = CudaQuadratureAssembly::try_new(pool.clone(), map.values()).unwrap();
            assert!(!map.template().values.is_empty());
            let mut sparse =
                PersistentDeviceCsr::try_from_host_csr_ref(b, map.template().as_host_csr_ref())
                    .unwrap();
            let mut values = pool
                .try_alloc::<Complex64>(map.values().output_count())
                .unwrap();
            for scale in [1.0, 0.0, -0.4] {
                let coefficients: Vec<_> = plan
                    .points()
                    .iter()
                    .map(|p| scale * alpha(p[0], p[1], p[2]))
                    .collect();
                b.try_copy_h2d(&coefficients, samples.raw_mut()).unwrap();
                let before = b.runtime_stats();
                device
                    .try_assemble(samples.raw(), values.raw_mut())
                    .unwrap();
                sparse.try_copy_values_from_device(b, values.raw()).unwrap();
                sparse.try_apply(b, x.raw(), y.raw_mut()).unwrap();
                let after = b.runtime_stats();
                assert_eq!(after.allocation_operations, before.allocation_operations);
                assert_eq!(after.transfer.h2d_bytes, before.transfer.h2d_bytes);
                assert_eq!(after.transfer.d2h_bytes, before.transfer.d2h_bytes);
                let mut actual = vec![Complex64::default(); n];
                b.try_copy_d2h(y.raw(), &mut actual).unwrap();
                let reference = prepare_hex_robin(
                    &mesh,
                    &space,
                    2,
                    &|x, y, z| scale * alpha(x, y, z),
                    &source,
                    4,
                    &dirichlet,
                )
                .unwrap();
                let matrix = if is_coupling {
                    &reference.dirichlet_coupling
                } else {
                    &reference.matrix
                };
                for (got, want) in actual.into_iter().zip(apply(matrix, &input)) {
                    close(got, want);
                }
            }
        }
        let load = CudaQuadratureAssembly::try_new(pool.clone(), plan.load()).unwrap();
        let coefficients: Vec<_> = plan
            .points()
            .iter()
            .map(|p| source(p[0], p[1], p[2]))
            .collect();
        b.try_copy_h2d(&coefficients, samples.raw_mut()).unwrap();
        load.try_assemble(samples.raw(), y.raw_mut()).unwrap();
        let mut actual = vec![Complex64::default(); n];
        b.try_copy_d2h(y.raw(), &mut actual).unwrap();
        let expected = prepare_hex_robin(&mesh, &space, 2, &alpha, &source, 4, &dirichlet).unwrap();
        for (got, want) in actual.into_iter().zip(expected.rhs) {
            close(got, want);
        }
    }
}
