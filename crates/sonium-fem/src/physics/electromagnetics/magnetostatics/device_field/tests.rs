//! Polynomial field accuracy, admission and publication failure tests.

// Rust guideline compliant 2026-02-21
use super::*;
use crate::mesh::{generators::rectangular_mesh_triangles, unit_cube_tetrahedra};
use sonium_compute::{CudaFailureInjectionPoint, MemoryPoolConfig};

fn pool() -> Option<MemoryPool<CudaBackend>> {
    match CudaBackend::new_default() {
        Ok(b) => Some(MemoryPool::new(b, MemoryPoolConfig::with_cap(1 << 28)).unwrap()),
        Err(e) => {
            assert!(
                std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                "{e}"
            );
            None
        }
    }
}
#[test]
fn native_magnetic_centroids_polynomial_accuracy() {
    let Some(pool) = pool() else { return };
    for (dim, mesh) in [
        (2, rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 2, 2)),
        (3, unit_cube_tetrahedra(2)),
    ] {
        for degree in [PolynomialDegree::P1, PolynomialDegree::P2] {
            let space = H1Space::new(degree);
            let mu: Vec<_> = (0..mesh.elements.len())
                .map(|e| 2.0 + 0.1 * e as f64)
                .collect();
            let br = vec![[0.4, 0.2, 0.1]; mesh.elements.len()];
            let mut evaluator =
                CudaMagneticCentroidField::try_new(&pool, &mesh, &space, &mu, &br).unwrap();
            assert!(evaluator.field().is_none());
            let coords = space.dof_coordinates(&mesh).unwrap();
            let mut potential = pool.try_alloc(coords.len()).unwrap();
            for (i, amplitude) in [1.0, 2.0, 0.5].into_iter().enumerate() {
                let values: Vec<_> = coords
                    .iter()
                    .map(|p| {
                        amplitude
                            * if degree == PolynomialDegree::P1 {
                                p[0] - p[1] + p[2]
                            } else {
                                p[0] * p[0] - p[1] * p[1] + p[2] * p[2]
                            }
                    })
                    .collect();
                pool.backend()
                    .try_copy_h2d(&values, potential.raw_mut())
                    .unwrap();
                let before = pool.backend().begin_runtime_stats();
                evaluator.evaluate(potential.raw()).unwrap();
                let stats = pool.backend().runtime_stats_since(before);
                assert_eq!(stats.transfer.h2d_bytes, 0);
                assert_eq!(stats.transfer.d2h_bytes, 8);
                if i > 0 {
                    assert_eq!(stats.allocation_operations, 0);
                }
                let mut field = vec![0.0; dim * mesh.elements.len()];
                pool.backend()
                    .try_copy_d2h(evaluator.field().unwrap(), &mut field)
                    .unwrap();
                for (e, elem) in mesh.elements.iter().enumerate() {
                    let mut p = [0.0; 3];
                    for &node in elem.vertices() {
                        let node = mesh.nodes[node];
                        for (d, value) in [node.x, node.y, node.z].into_iter().enumerate() {
                            p[d] += value / elem.vertices().len() as f64;
                        }
                    }
                    for d in 0..dim {
                        let gradient = amplitude
                            * if d == 1 { -1.0 } else { 1.0 }
                            * if degree == PolynomialDegree::P1 {
                                1.0
                            } else {
                                2.0 * p[d]
                            };
                        let expected = br[e][d] - mu[e] * gradient;
                        assert!((field[e * dim + d] - expected).abs() < 1e-12);
                    }
                }
            }
        }
    }
}

#[test]
fn native_magnetic_centroid_rejection_and_recovery() {
    let Some(pool) = pool() else { return };
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 2, 2);
    let space = H1Space::new(PolynomialDegree::P1);
    let mu = vec![2.0; mesh.elements.len()];
    let br = vec![[0.4, 0.0, 0.0]; mesh.elements.len()];
    let mut degenerate = mesh.clone();
    for node in &mut degenerate.nodes {
        node.y = 0.0;
    }
    let mut malformed = mesh.clone();
    malformed.elements[0].nodes[0] = usize::MAX;
    for (m, s, u, b) in [
        (
            &mesh,
            H1Space::new(PolynomialDegree::P3),
            mu.clone(),
            br.clone(),
        ),
        (&mesh, space.clone(), vec![0.0; mu.len()], br.clone()),
        (&mesh, space.clone(), vec![f64::NAN; mu.len()], br.clone()),
        (&mesh, space.clone(), vec![2.0], br.clone()),
        (
            &mesh,
            space.clone(),
            mu.clone(),
            vec![[f64::NAN; 3]; br.len()],
        ),
        (&degenerate, space.clone(), mu.clone(), br.clone()),
        (&malformed, space.clone(), mu.clone(), br.clone()),
    ] {
        let before = pool.backend().begin_runtime_stats();
        assert!(CudaMagneticCentroidField::try_new(&pool, m, &s, &u, &b).is_err());
        let stats = pool.backend().runtime_stats_since(before);
        assert_eq!(
            stats.allocation_operations
                + stats.kernel_launches
                + stats.transfer.h2d_bytes
                + stats.transfer.d2h_bytes,
            0
        );
    }
    let mut evaluator = CudaMagneticCentroidField::try_new(&pool, &mesh, &space, &mu, &br).unwrap();
    let coords = space.dof_coordinates(&mesh).unwrap();
    let mut potential = pool.try_alloc(coords.len()).unwrap();
    pool.backend()
        .try_copy_h2d(
            &coords.iter().map(|p| p[0]).collect::<Vec<_>>(),
            potential.raw_mut(),
        )
        .unwrap();
    evaluator.evaluate(potential.raw()).unwrap();
    assert!(evaluator.field().is_some());
    let short = pool.try_alloc::<f64>(1).unwrap();
    let foreign_pool = MemoryPool::new(
        CudaBackend::new_default().unwrap(),
        MemoryPoolConfig::with_cap(1 << 20),
    )
    .unwrap();
    let foreign = foreign_pool.try_alloc::<f64>(coords.len()).unwrap();
    for input in [short.raw(), foreign.raw()] {
        let before = pool.backend().begin_runtime_stats();
        assert!(evaluator.evaluate(input).is_err());
        assert!(evaluator.field().is_none());
        let stats = pool.backend().runtime_stats_since(before);
        assert_eq!(
            stats.allocation_operations
                + stats.kernel_launches
                + stats.transfer.h2d_bytes
                + stats.transfer.d2h_bytes,
            0
        );
    }
    pool.backend()
        .inject_failure(CudaFailureInjectionPoint::Launch);
    assert!(evaluator.evaluate(potential.raw()).is_err());
    assert!(evaluator.field().is_none());
    pool.backend().recover_after_error().unwrap();
    evaluator.evaluate(potential.raw()).unwrap();
    let mut changed_br = pool.try_alloc(mesh.elements.len() * 2).unwrap();
    pool.backend()
        .try_copy_h2d(&vec![0.8; changed_br.len()], changed_br.raw_mut())
        .unwrap();
    pool.backend()
        .inject_failure(CudaFailureInjectionPoint::TransferD2D);
    assert!(evaluator.try_update_remanence(changed_br.raw()).is_err());
    assert!(evaluator.field().is_none());
    pool.backend().recover_after_error().unwrap();
    assert!(evaluator.evaluate(potential.raw()).is_err());
    evaluator.try_update_remanence(changed_br.raw()).unwrap();
    evaluator.evaluate(potential.raw()).unwrap();
    let mut output = vec![0.0; changed_br.len()];
    pool.backend()
        .try_copy_d2h(evaluator.field().unwrap(), &mut output)
        .unwrap();
    for b in output.chunks_exact(2) {
        assert!((b[0] + 1.2).abs() < 1e-12 && (b[1] - 0.8).abs() < 1e-12);
    }
    pool.backend()
        .try_copy_h2d(&vec![f64::NAN; coords.len()], potential.raw_mut())
        .unwrap();
    assert!(evaluator.evaluate(potential.raw()).is_err());
    assert!(evaluator.field().is_none());
}
