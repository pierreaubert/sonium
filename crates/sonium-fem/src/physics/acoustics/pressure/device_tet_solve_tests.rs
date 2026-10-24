//! Independent FEM assembly checks for retained tetrahedral solves and updates.

// Rust guideline compliant 2026-02-21
use super::*;
use crate::{
    assembly::assemble_bilinear_checked, basis::PolynomialDegree, mesh::unit_cube_tetrahedra,
    spaces::FunctionSpace,
};
use sonium_compute::{Backend, CudaFailureInjectionPoint, MemoryPoolConfig, Transpose};
use sonium_solvers::AmgSmoother;

fn exercise<const N: usize>(
    mut session: CudaTetPressureSession<'_, N>,
    mesh: &Mesh,
    space: &H1Space,
) {
    let backend = session.pool.backend();
    let n = space.num_global_dofs(mesh);
    let controls = BackendKrylovConfig {
        max_iterations: 800,
        restart: 60,
        tolerance: 1e-10,
        require_native: true,
        exact_residual_check_interval: 0,
    };
    let coordinates = space.dof_coordinates(mesh).unwrap();
    let mut rhs = backend.raw_alloc::<Complex64>(n).unwrap();
    let mut prescribed = backend.raw_alloc::<Complex64>(n).unwrap();
    let mut action = backend.raw_alloc::<Complex64>(n).unwrap();
    assert!(!session.fused_jacobi_enabled());
    for pass in 0..3 {
        if pass == 1 {
            assert!(session.solution().is_some());
            let before = backend.begin_runtime_stats();
            session.set_fused_jacobi_enabled(true);
            assert!(session.fused_jacobi_enabled());
            assert!(session.solution().is_some());
            let stats = backend.runtime_stats_since(before);
            assert_eq!(stats.allocation_operations, 0);
            assert_eq!(stats.kernel_launches + stats.library_operations, 0);
            assert_eq!(
                stats.transfer.h2d_bytes + stats.transfer.d2h_bytes + stats.transfer.d2d_bytes,
                0
            );
        }
        let k = Complex64::new(4.0 + 0.1 * pass as f64, -0.02 * pass as f64);
        let physical = assemble_bilinear_checked(&HelmholtzForm::new(k), space, space, mesh, 4)
            .unwrap()
            .to_csr();
        let exact: Vec<_> = coordinates
            .iter()
            .map(|p| Complex64::new((p[0] + 0.3 * p[1] + pass as f64).sin(), 0.2 + p[2] * 0.1))
            .collect();
        let b: Vec<_> = (0..n)
            .map(|row| {
                (physical.row_ptrs[row]..physical.row_ptrs[row + 1])
                    .map(|j| physical.values[j] * exact[physical.col_indices[j]])
                    .sum::<Complex64>()
            })
            .collect();
        backend.try_copy_h2d(&b, &mut rhs).unwrap();
        backend.try_copy_h2d(&exact, &mut prescribed).unwrap();
        let before = backend.begin_runtime_stats();
        session.try_update_wavenumber(k).unwrap();
        let update = backend.runtime_stats_since(before);
        assert_eq!(update.allocation_operations, 0);
        assert_eq!(update.transfer.h2d_bytes, 0);
        assert_eq!(update.transfer.d2h_bytes, 0);
        if session.mode == TetPressureOperator::Element {
            assert_eq!(update.kernel_launches, 0);
            assert_eq!(update.library_operations, 0);
            assert_eq!(update.transfer.d2d_bytes, 0);
        }
        assert!(session.solution().is_none());
        let before = backend.begin_runtime_stats();
        let status = session
            .solve(
                &rhs,
                &prescribed,
                if pass == 0 {
                    DeviceInitialGuess::Zero
                } else {
                    DeviceInitialGuess::PreviousSolution
                },
                &controls,
                None,
            )
            .unwrap();
        let stats = backend.runtime_stats_since(before);
        assert!(
            status.converged,
            "N={N} mode={:?}: {status:?}",
            session.mode
        );
        // Krylov control coefficients/scalars still cross the host boundary.
        if pass > 0 {
            assert_eq!(
                stats.allocation_operations, 0,
                "warm solve allocations: {stats:?}"
            );
        }
        let mut actual = vec![Complex64::default(); n];
        backend
            .try_copy_d2h(session.solution().unwrap(), &mut actual)
            .unwrap();
        let error = actual
            .iter()
            .zip(&exact)
            .map(|(a, b)| (*a - *b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(error < 1e-7 * (n as f64).sqrt(), "solution error {error}");
        // Check the updated Hermitian action against independently constrained CSR.
        session
            .volume
            .element()
            .try_apply_action(backend, 1, &prescribed, &mut action, Transpose::ConjTrans)
            .unwrap();
        backend.try_copy_d2h(&action, &mut actual).unwrap();
        let constrained: std::collections::HashSet<_> =
            space.boundary_dofs(mesh, 1).into_iter().collect();
        let mut expected = vec![Complex64::default(); n];
        for row in 0..n {
            if constrained.contains(&row) {
                expected[row] = exact[row];
                continue;
            }
            for j in physical.row_ptrs[row]..physical.row_ptrs[row + 1] {
                let column = physical.col_indices[j];
                if !constrained.contains(&column) {
                    expected[column] += physical.values[j].conj() * exact[row];
                }
            }
        }
        for (a, b) in actual.iter().zip(expected) {
            assert!((*a - b).norm() < 1e-10 * (1.0 + b.norm()));
        }
        eprintln!(
            "tet session N={N} mode={:?} pass={pass} iterations={} allocations={} error={error}",
            session.mode, status.iterations, stats.allocation_operations
        );
    }
    let amg = AmgConfig {
        coarse_size: 16,
        smoother: AmgSmoother::Jacobi,
        jacobi_weight: 0.6,
        ..Default::default()
    };
    session
        .try_rebuild_preconditioner(Complex64::new(18.0, 18.0), amg.clone())
        .unwrap();
    assert!(session.fused_jacobi_enabled());
    let before = backend.begin_runtime_stats();
    assert!(
        session
            .try_refresh_preconditioner(Complex64::new(18.001, 18.001), amg.clone())
            .unwrap()
    );
    assert!(session.fused_jacobi_enabled());
    let refresh = backend.runtime_stats_since(before);
    assert_eq!(refresh.allocation_operations, 0);
    eprintln!(
        "tet coarse refresh N={N} mode={:?} allocations={} h2d={} d2h={}",
        session.mode,
        refresh.allocation_operations,
        refresh.transfer.h2d_bytes,
        refresh.transfer.d2h_bytes
    );
    assert!(session.solution().is_none());
    assert!(
        session
            .solve(
                &rhs,
                &prescribed,
                DeviceInitialGuess::PreviousSolution,
                &controls,
                None
            )
            .unwrap()
            .converged
    );
    assert!(
        session
            .try_rebuild_preconditioner(Complex64::new(f64::NAN, 0.0), amg)
            .is_err()
    );
    assert!(session.fused_jacobi_enabled());
    assert!(session.solution().is_none());
    assert!(
        session
            .solve(&rhs, &prescribed, DeviceInitialGuess::Zero, &controls, None)
            .unwrap()
            .converged
    );
    if session.mode == TetPressureOperator::Csr {
        backend.inject_failure(CudaFailureInjectionPoint::TransferD2D);
        assert!(
            session
                .try_update_wavenumber(Complex64::new(4.0, 0.0))
                .is_err()
        );
        assert!(session.solution().is_none());
        backend.recover_after_error().unwrap();
        assert!(
            session
                .solve(&rhs, &prescribed, DeviceInitialGuess::Zero, &controls, None)
                .is_err()
        );
    } else {
        // No copy occurs during element-mode frequency updates. Exercise a real
        // element launch failure instead, then recover using the retained state.
        backend.inject_failure(CudaFailureInjectionPoint::Launch);
        assert!(
            session
                .solve(&rhs, &prescribed, DeviceInitialGuess::Zero, &controls, None)
                .is_err()
        );
        assert!(session.solution().is_none());
        backend.recover_after_error().unwrap();
    }
    session
        .try_update_wavenumber(Complex64::new(4.2, -0.04))
        .unwrap();
    assert!(
        session
            .solve(&rhs, &prescribed, DeviceInitialGuess::Zero, &controls, None)
            .unwrap()
            .converged
    );
}

#[test]
fn native_tet_sessions_p1_p2_csr_element_frequency_and_failure() {
    let backend = match CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => {
            assert!(
                std::env::var("SONIUM_REQUIRE_CUDA").is_err(),
                "CUDA required: {e}"
            );
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 29)).unwrap();
    let mut mesh = unit_cube_tetrahedra(3);
    for face in &mut mesh.boundaries {
        face.marker = if face.nodes.iter().all(|&i| mesh.nodes[i].x == 0.0) {
            1
        } else {
            2
        };
    }
    for mode in [TetPressureOperator::Csr, TetPressureOperator::Element] {
        for degree in [PolynomialDegree::P1, PolynomialDegree::P2] {
            let space = H1Space::new(degree);
            let config = TetPressureConfig {
                dirichlet: space.boundary_dofs(&mesh, 1),
                mass_shift: Complex64::new(16.0, 16.0),
                operator: mode,
                amg: AmgConfig {
                    coarse_size: 16,
                    smoother: AmgSmoother::Jacobi,
                    jacobi_weight: 0.6,
                    ..Default::default()
                },
            };
            let form = HelmholtzForm::from_real(4.0);
            match degree {
                PolynomialDegree::P1 => exercise(
                    CudaTetP1PressureSession::try_new(&pool, &mesh, &space, &form, &config)
                        .unwrap(),
                    &mesh,
                    &space,
                ),
                PolynomialDegree::P2 => exercise(
                    CudaTetP2PressureSession::try_new(&pool, &mesh, &space, &form, &config)
                        .unwrap(),
                    &mesh,
                    &space,
                ),
                _ => unreachable!(),
            }
        }
    }
}

#[test]
fn native_tet_p1_robin_session_frequency_lifting_and_reuse() {
    use crate::boundary::{FemSystem, try_apply_robin_complex};
    let backend = match CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => {
            assert!(
                std::env::var("SONIUM_REQUIRE_CUDA").is_err(),
                "CUDA required: {e}"
            );
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 29)).unwrap();
    let backend = pool.backend();
    let mut mesh = unit_cube_tetrahedra(3);
    for face in &mut mesh.boundaries {
        face.marker = if face.nodes.iter().all(|&i| mesh.nodes[i].x == 0.0) {
            1
        } else {
            2
        };
    }
    let space = H1Space::new(PolynomialDegree::P1);
    let n = space.num_global_dofs(&mesh);
    let controls = BackendKrylovConfig {
        max_iterations: 800,
        restart: 60,
        tolerance: 1e-10,
        require_native: true,
        exact_residual_check_interval: 0,
    };
    let source = |x: f64, y: f64, z: f64| Complex64::new(0.3 + x, y - z);
    for scale in [0.0, 1.0] {
        let alpha = |x: f64, y: f64, z: f64| Complex64::new(0.7 + x, 0.2 + y + z) * scale;
        for mode in [TetPressureOperator::Csr, TetPressureOperator::Element] {
            let config = TetPressureConfig {
                dirichlet: space.boundary_dofs(&mesh, 1),
                mass_shift: Complex64::new(16.0, 16.0),
                operator: mode,
                amg: AmgConfig {
                    coarse_size: 16,
                    smoother: AmgSmoother::Jacobi,
                    jacobi_weight: 0.6,
                    ..Default::default()
                },
            };
            let boundary = TetRobinBoundary {
                marker: 2,
                alpha: &alpha,
                source: &source,
            };
            let mut session = CudaTetP1PressureSession::try_new_with_boundaries(
                &pool,
                &mesh,
                &space,
                &HelmholtzForm::from_real(4.0),
                &config,
                &[boundary],
            )
            .unwrap();
            let mut rhs = backend.raw_alloc::<Complex64>(n).unwrap();
            let mut prescribed = backend.raw_alloc::<Complex64>(n).unwrap();
            let points = session.boundary_samples().unwrap().0.to_vec();
            let mut alpha_device = backend.raw_alloc::<Complex64>(points.len()).unwrap();
            let mut source_device = backend.raw_alloc::<Complex64>(points.len()).unwrap();
            backend
                .try_copy_h2d(
                    &points
                        .iter()
                        .map(|p| alpha(p[0], p[1], p[2]))
                        .collect::<Vec<_>>(),
                    &mut alpha_device,
                )
                .unwrap();
            backend
                .try_copy_h2d(
                    &points
                        .iter()
                        .map(|p| source(p[0], p[1], p[2]))
                        .collect::<Vec<_>>(),
                    &mut source_device,
                )
                .unwrap();
            for pass in 0..3 {
                let k = Complex64::new(4.0 + pass as f64 * 0.1, -0.02 * pass as f64);
                let matrix =
                    assemble_bilinear_checked(&HelmholtzForm::new(k), &space, &space, &mesh, 4)
                        .unwrap();
                let mut full = FemSystem::new(matrix, vec![Complex64::default(); n]);
                try_apply_robin_complex(&mut full, &mesh, 2, &alpha, &source).unwrap();
                let matrix = full.matrix.to_csr();
                let exact: Vec<_> = mesh
                    .nodes
                    .iter()
                    .map(|p| Complex64::new((p.x + p.y + pass as f64).sin(), 0.2 + p.z))
                    .collect();
                let load: Vec<_> = (0..n)
                    .map(|row| {
                        (matrix.row_ptrs[row]..matrix.row_ptrs[row + 1])
                            .map(|j| matrix.values[j] * exact[matrix.col_indices[j]])
                            .sum::<Complex64>()
                            - full.rhs[row]
                    })
                    .collect();
                backend.try_copy_h2d(&load, &mut rhs).unwrap();
                backend.try_copy_h2d(&exact, &mut prescribed).unwrap();
                let before = backend.begin_runtime_stats();
                session.try_update_wavenumber(k).unwrap();
                session
                    .try_update_boundary(&alpha_device, &source_device)
                    .unwrap();
                let stats = backend.runtime_stats_since(before);
                assert_eq!(stats.allocation_operations, 0);
                assert_eq!(stats.transfer.h2d_bytes, 0);
                assert_eq!(stats.transfer.d2h_bytes, 0);
                let before = backend.begin_runtime_stats();
                let status = session
                    .solve(
                        &rhs,
                        &prescribed,
                        if pass == 0 {
                            DeviceInitialGuess::Zero
                        } else {
                            DeviceInitialGuess::PreviousSolution
                        },
                        &controls,
                        None,
                    )
                    .unwrap();
                let stats = backend.runtime_stats_since(before);
                assert!(status.converged, "scale={scale} mode={mode:?}: {status:?}");
                if pass > 0 {
                    assert_eq!(stats.allocation_operations, 0, "{stats:?}");
                }
                let mut actual = vec![Complex64::default(); n];
                backend
                    .try_copy_d2h(session.solution().unwrap(), &mut actual)
                    .unwrap();
                let error = actual
                    .iter()
                    .zip(&exact)
                    .map(|(a, b)| (*a - *b).norm_sqr())
                    .sum::<f64>()
                    .sqrt();
                assert!(error < 1e-7 * (n as f64).sqrt(), "{error}");
                eprintln!(
                    "tet Robin scale={scale} mode={mode:?} pass={pass} iterations={} allocations={} error={error}",
                    status.iterations, stats.allocation_operations
                );
            }
            session
                .try_rebuild_preconditioner(Complex64::new(18.0, 18.0), config.amg.clone())
                .unwrap();
            assert!(session.solution().is_none());
            assert!(
                session
                    .solve(&rhs, &prescribed, DeviceInitialGuess::Zero, &controls, None)
                    .unwrap()
                    .converged
            );
            let bad = TetRobinBoundary {
                marker: 99,
                alpha: &alpha,
                source: &source,
            };
            assert!(
                CudaTetP1PressureSession::try_new_with_boundaries(
                    &pool,
                    &mesh,
                    &space,
                    &HelmholtzForm::from_real(4.0),
                    &config,
                    &[bad]
                )
                .is_err()
            );
            let duplicates = [
                TetRobinBoundary {
                    marker: 2,
                    alpha: &alpha,
                    source: &source,
                },
                TetRobinBoundary {
                    marker: 2,
                    alpha: &alpha,
                    source: &source,
                },
            ];
            assert!(
                CudaTetP1PressureSession::try_new_with_boundaries(
                    &pool,
                    &mesh,
                    &space,
                    &HelmholtzForm::from_real(4.0),
                    &config,
                    &duplicates
                )
                .is_err()
            );
        }
    }
}

#[test]
fn native_tet_p2_robin_session_shared_edges_and_reuse() {
    use crate::physics::acoustics::pressure::device_boundary::prepare_tet_p2_robin;
    let backend = match CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => {
            assert!(
                std::env::var("SONIUM_REQUIRE_CUDA").is_err(),
                "CUDA required: {e}"
            );
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 29)).unwrap();
    let backend = pool.backend();
    let mut mesh = unit_cube_tetrahedra(2);
    for face in &mut mesh.boundaries {
        face.marker = if face.nodes.iter().all(|&i| mesh.nodes[i].x == 0.0) {
            1
        } else {
            2
        };
    }
    let space = H1Space::new(PolynomialDegree::P2);
    let coordinates = space.dof_coordinates(&mesh).unwrap();
    let n = coordinates.len();
    let alpha = |x: f64, y: f64, z: f64| Complex64::new(0.7 + x, 0.2 + y + z);
    let source = |x: f64, y: f64, z: f64| Complex64::new(0.3 + x, y - z);
    let controls = BackendKrylovConfig {
        max_iterations: 800,
        restart: 60,
        tolerance: 1e-10,
        require_native: true,
        exact_residual_check_interval: 0,
    };
    for mode in [TetPressureOperator::Csr, TetPressureOperator::Element] {
        let config = TetPressureConfig {
            dirichlet: space.boundary_dofs(&mesh, 1),
            mass_shift: Complex64::new(16.0, 16.0),
            operator: mode,
            amg: AmgConfig {
                coarse_size: 16,
                smoother: AmgSmoother::Jacobi,
                jacobi_weight: 0.6,
                ..Default::default()
            },
        };
        assert!(config.dirichlet.iter().any(|&d| d >= mesh.nodes.len()));
        let mut session = CudaTetP2PressureSession::try_new_with_boundaries(
            &pool,
            &mesh,
            &space,
            &HelmholtzForm::from_real(4.0),
            &config,
            &[TetRobinBoundary {
                marker: 2,
                alpha: &alpha,
                source: &source,
            }],
        )
        .unwrap();
        let mut rhs = backend.raw_alloc::<Complex64>(n).unwrap();
        let mut prescribed = backend.raw_alloc::<Complex64>(n).unwrap();
        let points = session.boundary_samples().unwrap().0.to_vec();
        let mut alpha_device = backend.raw_alloc::<Complex64>(points.len()).unwrap();
        let mut source_device = backend.raw_alloc::<Complex64>(points.len()).unwrap();
        for pass in 0..3 {
            let changed_alpha = |x, y, z| alpha(x, y, z) * pass as f64;
            let changed_source = |x, y, z| source(x, y, z) * (1.0 + 0.2 * pass as f64);
            let surface =
                prepare_tet_p2_robin(&mesh, &space, 2, &changed_alpha, &changed_source, &[])
                    .unwrap();
            let av: Vec<_> = points
                .iter()
                .map(|p| changed_alpha(p[0], p[1], p[2]))
                .collect();
            let gv: Vec<_> = points
                .iter()
                .map(|p| changed_source(p[0], p[1], p[2]))
                .collect();
            backend.try_copy_h2d(&av, &mut alpha_device).unwrap();
            backend.try_copy_h2d(&gv, &mut source_device).unwrap();
            let k = Complex64::new(4.0 + pass as f64 * 0.1, -pass as f64 * 0.02);
            let physical =
                assemble_bilinear_checked(&HelmholtzForm::new(k), &space, &space, &mesh, 4)
                    .unwrap()
                    .to_csr();
            let exact: Vec<_> = coordinates
                .iter()
                .map(|p| Complex64::new(p[0] * p[0] + p[1] + pass as f64, 0.2 + p[2] * p[1]))
                .collect();
            let b: Vec<_> = (0..n)
                .map(|r| {
                    [&physical, &surface.matrix]
                        .iter()
                        .map(|a| {
                            (a.row_ptrs[r]..a.row_ptrs[r + 1])
                                .map(|j| a.values[j] * exact[a.col_indices[j]])
                                .sum::<Complex64>()
                        })
                        .sum::<Complex64>()
                        - surface.rhs[r]
                })
                .collect();
            backend.try_copy_h2d(&b, &mut rhs).unwrap();
            backend.try_copy_h2d(&exact, &mut prescribed).unwrap();
            let before = backend.begin_runtime_stats();
            session.try_update_wavenumber(k).unwrap();
            session
                .try_update_boundary(&alpha_device, &source_device)
                .unwrap();
            let stats = backend.runtime_stats_since(before);
            assert_eq!(stats.allocation_operations, 0);
            assert_eq!(stats.transfer.h2d_bytes, 0);
            assert_eq!(stats.transfer.d2h_bytes, 0);
            let before = backend.begin_runtime_stats();
            let status = session
                .solve(
                    &rhs,
                    &prescribed,
                    if pass == 0 {
                        DeviceInitialGuess::Zero
                    } else {
                        DeviceInitialGuess::PreviousSolution
                    },
                    &controls,
                    None,
                )
                .unwrap();
            let stats = backend.runtime_stats_since(before);
            assert!(status.converged, "{status:?}");
            if pass > 0 {
                assert_eq!(stats.allocation_operations, 0, "{stats:?}");
            }
            let mut actual = vec![Complex64::default(); n];
            backend
                .try_copy_d2h(session.solution().unwrap(), &mut actual)
                .unwrap();
            let error = actual
                .iter()
                .zip(&exact)
                .map(|(a, b)| (*a - *b).norm_sqr())
                .sum::<f64>()
                .sqrt();
            assert!(error < 1e-7 * (n as f64).sqrt(), "{error}");
            eprintln!(
                "tet P2 Robin mode={mode:?} pass={pass} iterations={} allocations={} error={error}",
                status.iterations, stats.allocation_operations
            );
        }
        backend.inject_failure(CudaFailureInjectionPoint::TransferD2D);
        assert!(
            session
                .try_update_boundary(&alpha_device, &source_device)
                .is_err()
        );
        assert!(session.solution().is_none());
        backend.recover_after_error().unwrap();
        session
            .try_update_wavenumber(Complex64::new(4.2, -0.04))
            .unwrap();
        assert!(
            session
                .solve(&rhs, &prescribed, DeviceInitialGuess::Zero, &controls, None)
                .is_err()
        );
        session
            .try_update_boundary(&alpha_device, &source_device)
            .unwrap();
        session
            .try_rebuild_preconditioner(Complex64::new(18.0, 18.0), config.amg)
            .unwrap();
        assert!(session.solution().is_none());
        assert!(
            session
                .solve(&rhs, &prescribed, DeviceInitialGuess::Zero, &controls, None)
                .unwrap()
                .converged
        );
    }
}

#[test]
fn native_tet_element_storage_omits_physical_csr() {
    let backend = match CudaBackend::new_default() {
        Ok(b) => b,
        Err(e) => {
            assert!(
                std::env::var("SONIUM_REQUIRE_CUDA").is_err(),
                "CUDA required: {e}"
            );
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 29)).unwrap();
    let mesh = unit_cube_tetrahedra(4);
    let space = H1Space::new(PolynomialDegree::P2);
    let host = prepare_tet_p2_helmholtz(&mesh, &space, &HelmholtzForm::from_real(1.0)).unwrap();
    let backend = pool.backend();
    let before = backend.memory_stats().live_allocation_bytes;
    let element = volume::TetVolume::prepare(&pool, &host, TetPressureOperator::Element).unwrap();
    let after_element = backend.memory_stats().live_allocation_bytes;
    let csr = volume::TetVolume::prepare(&pool, &host, TetPressureOperator::Csr).unwrap();
    let after_csr = backend.memory_stats().live_allocation_bytes;
    let element_bytes = after_element - before;
    let csr_bytes = after_csr - after_element;
    assert!(matches!(element, volume::TetVolume::Element(_)));
    assert!(matches!(csr, volume::TetVolume::Csr(_)));
    assert!(
        element_bytes > 0 && csr_bytes > element_bytes,
        "element={element_bytes}, csr={csr_bytes}"
    );
    // Both stay alive during measurement: pool reuse cannot hide an allocation.
    eprintln!(
        "tet physical P2 dofs={} element_bytes={element_bytes} csr_bytes={csr_bytes}",
        host.dimension()
    );
    drop((element, csr));
}
