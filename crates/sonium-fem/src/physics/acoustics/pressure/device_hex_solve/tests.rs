//! Independent assembled-system checks for the public retained solve API.

// Rust guideline compliant 2026-02-21
use super::*;
use crate::assembly::assemble_bilinear_checked;
use crate::basis::PolynomialDegree;
use crate::mesh::unit_cube_hexahedra;
use crate::spaces::FunctionSpace;
use sonium_compute::MemoryPoolConfig;
use sonium_solvers::AmgSmoother;

fn apply(matrix: &CsrMatrix<Complex64>, x: &[Complex64]) -> Vec<Complex64> {
    (0..matrix.num_rows)
        .map(|row| {
            (matrix.row_ptrs[row]..matrix.row_ptrs[row + 1])
                .map(|j| matrix.values[j] * x[matrix.col_indices[j]])
                .sum()
        })
        .collect()
}

struct MaterialHelmholtz {
    wavenumber: Complex64,
    mass_factor: f64,
    spatial: bool,
}
impl MaterialHelmholtz {
    fn samples(&self, [x, y, z]: [f64; 3]) -> (Complex64, Complex64) {
        if self.spatial {
            (
                Complex64::new(1.0 + 0.03 * (x + y + z), 0.01 * y),
                self.mass_factor * Complex64::new(1.0 + 0.02 * x, -0.01 * z),
            )
        } else {
            (
                Complex64::new(1.0, 0.0),
                Complex64::new(self.mass_factor, 0.0),
            )
        }
    }
}
impl crate::forms::BilinearForm<Complex64> for MaterialHelmholtz {
    fn integrate_point(
        &self,
        test: &crate::spaces::BasisTabulation,
        trial: &crate::spaces::BasisTabulation,
        weight: f64,
        matrix: &mut crate::forms::LocalMatrix<Complex64>,
    ) {
        let (diffusion, mass) = self.samples(test.physical_coords);
        let k2 = self.wavenumber * self.wavenumber;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                let gradient: f64 = (0..test.spatial_dim)
                    .map(|d| test.grad(i, d) * trial.grad(j, d))
                    .sum();
                matrix.add(
                    i,
                    j,
                    weight
                        * test.det_j
                        * (diffusion * gradient - k2 * mass * test.value(i) * trial.value(j)),
                );
            }
        }
    }
}

#[test]
fn native_session_boundaries_warm_start_and_failed_publication() {
    let backend = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            assert!(
                std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                "CUDA required: {e}"
            );
            eprintln!("SKIP CUDA pressure session: {e}");
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 29)).unwrap();
    let b = pool.backend();
    let mut mesh = unit_cube_hexahedra(2);
    for face in &mut mesh.boundaries {
        face.marker = if face.nodes.iter().all(|&i| mesh.nodes[i].x == 0.0) {
            1
        } else if face.nodes.iter().all(|&i| mesh.nodes[i].y == 0.0) {
            2
        } else {
            3
        };
    }
    let alpha = |x: f64, y: f64, z: f64| Complex64::new(0.7 + x, 0.3 + y + z);
    let zero = |_: f64, _: f64, _: f64| Complex64::default();
    let source = |x: f64, y: f64, z: f64| Complex64::new(0.3 + y, x - z);
    let form = HelmholtzForm::from_real(5.0);
    let controls = BackendKrylovConfig {
        max_iterations: 600,
        restart: 60,
        tolerance: 1e-10,
        require_native: true,
        exact_residual_check_interval: 0,
    };
    for degree in [PolynomialDegree::P2, PolynomialDegree::P3] {
        let space = H1Space::new(degree);
        let n = space.dof_coordinates(&mesh).unwrap().len();
        let dirichlet = space.boundary_dofs(&mesh, 1);
        assert!(!dirichlet.is_empty());
        let setup = HexPressureConfig {
            quadrature_points: 4,
            dirichlet: dirichlet.clone(),
            mass_shift: Complex64::new(25.0, 25.0),
            amg: AmgConfig {
                coarse_size: 16,
                smoother: AmgSmoother::Jacobi,
                jacobi_weight: 0.6,
                ..Default::default()
            },
        };
        // Exercise absent facets, pure Neumann loads, and multiple Robin markers.
        for mode in 0..3 {
            let boundaries = match mode {
                0 => vec![],
                1 => vec![HexRobinBoundary {
                    marker: 2,
                    alpha: &zero,
                    source: &source,
                }],
                _ => vec![
                    HexRobinBoundary {
                        marker: 2,
                        alpha: &alpha,
                        source: &source,
                    },
                    HexRobinBoundary {
                        marker: 3,
                        alpha: &alpha,
                        source: &source,
                    },
                ],
            };
            let mut original = assemble_bilinear_checked(&form, &space, &space, &mesh, 4)
                .unwrap()
                .to_csr();
            let mut triplets = Vec::new();
            entries(&original, &mut triplets);
            let mut load = vec![Complex64::default(); n];
            for boundary in &boundaries {
                let full = prepare_hex_robin(
                    &mesh,
                    &space,
                    boundary.marker,
                    boundary.alpha,
                    boundary.source,
                    4,
                    &[],
                )
                .unwrap();
                entries(&full.matrix, &mut triplets);
                for (a, v) in load.iter_mut().zip(full.rhs) {
                    *a += v;
                }
            }
            original = CsrMatrix::try_from_triplets(n, n, triplets).unwrap();
            let mut session =
                CudaHexPressureSession::try_new(&pool, &mesh, &space, &form, &setup, &boundaries)
                    .unwrap();
            let mut rhs = pool.try_alloc::<Complex64>(n).unwrap();
            let mut prescribed = pool.try_alloc::<Complex64>(n).unwrap();
            for source_index in 0..3 {
                if source_index == 1 {
                    assert!(!session.small_jacobi_enabled());
                    assert!(session.solution().is_some());
                    let before = b.begin_runtime_stats();
                    session.set_small_jacobi_enabled(true);
                    assert!(session.small_jacobi_enabled());
                    assert!(session.solution().is_some());
                    let stats = b.runtime_stats_since(before);
                    assert_eq!(stats.allocation_operations, 0);
                    assert_eq!(stats.kernel_launches + stats.library_operations, 0);
                    assert_eq!(stats.transfer.h2d_bytes + stats.transfer.d2h_bytes, 0);
                }
                if source_index == 2 {
                    assert!(!session.fused_jacobi_enabled());
                    assert!(session.solution().is_some());
                    let before = b.begin_runtime_stats();
                    session.set_small_jacobi_enabled(false);
                    session.set_fused_jacobi_enabled(true);
                    assert!(session.fused_jacobi_enabled());
                    assert!(session.solution().is_some());
                    let stats = b.runtime_stats_since(before);
                    assert_eq!(stats.allocation_operations, 0);
                    assert_eq!(stats.kernel_launches + stats.library_operations, 0);
                    assert_eq!(
                        stats.transfer.h2d_bytes
                            + stats.transfer.d2h_bytes
                            + stats.transfer.d2d_bytes,
                        0
                    );
                }
                let exact: Vec<_> = (0..n)
                    .map(|i| {
                        Complex64::new(
                            (0.13 * i as f64 + source_index as f64).sin(),
                            (0.07 * i as f64).cos(),
                        )
                    })
                    .collect();
                let full_rhs = apply(&original, &exact);
                let volume_rhs: Vec<_> = full_rhs.iter().zip(&load).map(|(a, f)| *a - *f).collect();
                let mut g = vec![Complex64::new(f64::NAN, f64::NAN); n];
                for &i in &dirichlet {
                    g[i] = exact[i];
                }
                b.try_copy_h2d(&volume_rhs, rhs.raw_mut()).unwrap();
                b.try_copy_h2d(&g, prescribed.raw_mut()).unwrap();
                let before = b.runtime_stats();
                let status = session
                    .solve(
                        rhs.raw(),
                        prescribed.raw(),
                        if source_index == 0 {
                            DeviceInitialGuess::Zero
                        } else {
                            DeviceInitialGuess::PreviousSolution
                        },
                        &controls,
                        None,
                    )
                    .unwrap();
                assert!(
                    status.converged,
                    "degree={degree:?}, mode={mode}: {status:?}"
                );
                if source_index > 0 {
                    assert_eq!(
                        before.allocation_operations,
                        b.runtime_stats().allocation_operations
                    );
                }
                let mut actual = vec![Complex64::default(); n];
                b.try_copy_d2h(session.solution().unwrap(), &mut actual)
                    .unwrap();
                let action = apply(&original, &actual);
                let mut residual = 0.0;
                let mut norm = 0.0;
                for i in 0..n {
                    if dirichlet.contains(&i) {
                        assert!((actual[i] - exact[i]).norm() < 1e-8);
                    } else {
                        residual += (action[i] - full_rhs[i]).norm_sqr();
                        norm += full_rhs[i].norm_sqr();
                    }
                }
                assert!((residual / norm).sqrt() < 1e-8);
            }
            let short = pool.try_alloc::<Complex64>(n - 1).unwrap();
            assert!(
                session
                    .solve(
                        short.raw(),
                        prescribed.raw(),
                        DeviceInitialGuess::Zero,
                        &controls,
                        None
                    )
                    .is_err()
            );
            assert!(session.solution().is_none());
            assert!(
                session
                    .solve(
                        rhs.raw(),
                        prescribed.raw(),
                        DeviceInitialGuess::PreviousSolution,
                        &controls,
                        None
                    )
                    .is_err()
            );
            assert!(
                session
                    .solve(
                        rhs.raw(),
                        prescribed.raw(),
                        DeviceInitialGuess::Zero,
                        &controls,
                        None
                    )
                    .unwrap()
                    .converged
            );
            let cancelled = CancellationToken::new();
            cancelled.cancel();
            assert!(
                session
                    .solve(
                        rhs.raw(),
                        prescribed.raw(),
                        DeviceInitialGuess::PreviousSolution,
                        &controls,
                        Some(&cancelled)
                    )
                    .is_err()
            );
            assert!(session.solution().is_none());
        }
        for markers in [[2, 2], [2, 999]] {
            let boundaries: Vec<_> = markers
                .into_iter()
                .map(|marker| HexRobinBoundary {
                    marker,
                    alpha: &alpha,
                    source: &source,
                })
                .collect();
            assert!(
                CudaHexPressureSession::try_new(&pool, &mesh, &space, &form, &setup, &boundaries)
                    .is_err()
            );
        }
    }
}

#[test]
fn native_frequency_and_boundary_updates_certify_changed_system() {
    let backend = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            assert!(
                std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                "CUDA required: {e}"
            );
            eprintln!("SKIP CUDA boundary-update solve: {e}");
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 29)).unwrap();
    let b = pool.backend();
    let mut mesh = unit_cube_hexahedra(2);
    for point in &mut mesh.nodes {
        point.z += 0.15 * point.x * point.y;
    }
    for face in &mut mesh.boundaries {
        face.marker = if face.nodes.iter().all(|&i| mesh.nodes[i].y == 0.0) {
            2
        } else {
            3
        };
    }
    let zero = |_: f64, _: f64, _: f64| Complex64::default();
    let initial = [
        HexRobinBoundary {
            marker: 2,
            alpha: &zero,
            source: &zero,
        },
        HexRobinBoundary {
            marker: 3,
            alpha: &zero,
            source: &zero,
        },
    ];
    let alpha =
        |marker: i32, p: [f64; 3]| Complex64::new(0.2 * marker as f64 + p[0], 0.3 + p[1] * p[2]);
    let source = |marker: i32, p: [f64; 3]| Complex64::new(p[0] * p[1], p[2] - marker as f64 * 0.1);
    let form = HelmholtzForm::from_real(0.0);
    let controls = BackendKrylovConfig {
        max_iterations: 600,
        restart: 60,
        tolerance: 1e-10,
        require_native: true,
        exact_residual_check_interval: 0,
    };
    for (degree, aggregation) in [
        PolynomialDegree::P1,
        PolynomialDegree::P2,
        PolynomialDegree::P3,
    ]
    .into_iter()
    .flat_map(|degree| [(degree, false), (degree, true)])
    {
        let space = H1Space::new(degree);
        let coordinates = space.dof_coordinates(&mesh).unwrap();
        let n = coordinates.len();
        let dirichlet: Vec<_> = coordinates
            .iter()
            .enumerate()
            .filter_map(|(i, p)| (p[0] == 0.0).then_some(i))
            .collect();
        let setup = HexPressureConfig {
            quadrature_points: 4,
            dirichlet: dirichlet.clone(),
            mass_shift: Complex64::new(25.0, 25.0),
            amg: AmgConfig {
                coarse_size: 16,
                smoother: AmgSmoother::Jacobi,
                jacobi_weight: 0.6,
                ..if aggregation {
                    AmgConfig::for_aggregation()
                } else {
                    AmgConfig::default()
                }
            },
        };
        let mut session = CudaHexPressureSession::try_new_updatable(
            &pool, &mesh, &space, &form, &setup, &initial,
        )
        .unwrap();
        assert!(!session.small_jacobi_enabled());
        session.set_small_jacobi_enabled(true);
        assert!(!session.fused_jacobi_enabled());
        session.set_fused_jacobi_enabled(true);
        let (points, markers) = session.boundary_samples().unwrap();
        let metadata: Vec<_> = points
            .iter()
            .copied()
            .zip(markers.iter().copied())
            .collect();
        assert!(metadata.iter().any(|(_, m)| *m == 2) && metadata.iter().any(|(_, m)| *m == 3));
        let mut a = pool.try_alloc::<Complex64>(metadata.len()).unwrap();
        let mut f = pool.try_alloc::<Complex64>(metadata.len()).unwrap();
        let mut rhs = pool.try_alloc::<Complex64>(n).unwrap();
        let mut g = pool.try_alloc::<Complex64>(n).unwrap();
        let frequencies = [
            (1.0, Complex64::default()),
            (0.0, Complex64::new(4.0, 0.2)),
            (1.4, Complex64::new(6.0, -0.1)),
            (0.8, Complex64::default()),
        ];
        let canonical =
            prepare_hex_helmholtz(&mesh, &space, &HelmholtzForm::from_real(1.0), 4).unwrap();
        let mut volume_data = pool
            .try_alloc::<Complex64>(canonical.quadrature_data().len())
            .unwrap();
        let volume_points = session.volume_samples().unwrap().to_vec();
        assert_eq!(volume_points.len() * 7, canonical.quadrature_data().len());
        let mut material_d = pool.try_alloc::<Complex64>(volume_points.len()).unwrap();
        let mut material_m = pool.try_alloc::<Complex64>(volume_points.len()).unwrap();
        for (update, (scale, wavenumber)) in frequencies.into_iter().enumerate() {
            let mass_factor: f64 = if update == 2 { 1.2 } else { 1.0 };
            let material = MaterialHelmholtz {
                wavenumber,
                mass_factor,
                spatial: update == 2,
            };
            let (ds, ms): (Vec<_>, Vec<_>) =
                volume_points.iter().map(|&p| material.samples(p)).unzip();
            b.try_copy_h2d(&ds, material_d.raw_mut()).unwrap();
            b.try_copy_h2d(&ms, material_m.raw_mut()).unwrap();
            let mut data = canonical.quadrature_data().to_vec();
            for point in data.as_chunks_mut::<7>().0 {
                point[0] *= mass_factor;
            }
            b.try_copy_h2d(&data, volume_data.raw_mut()).unwrap();
            let coefficients: Vec<_> = metadata.iter().map(|&(p, m)| scale * alpha(m, p)).collect();
            let sources: Vec<_> = metadata
                .iter()
                .map(|&(p, m)| (scale + 0.2) * source(m, p))
                .collect();
            b.try_copy_h2d(&coefficients, a.raw_mut()).unwrap();
            b.try_copy_h2d(&sources, f.raw_mut()).unwrap();
            let before = b.runtime_stats();
            if update == 1 || update == 2 {
                session
                    .try_update_material_samples(material_d.raw(), material_m.raw())
                    .unwrap();
            } else {
                session.try_update_volume_data(volume_data.raw()).unwrap();
            }
            session.try_update_wavenumber(wavenumber).unwrap();
            assert_eq!(session.wavenumber(), wavenumber);
            session.try_update_boundary(a.raw(), f.raw()).unwrap();
            let after = b.runtime_stats();
            assert_eq!(after.allocation_operations, before.allocation_operations);
            assert_eq!(after.transfer.h2d_bytes, before.transfer.h2d_bytes);
            assert_eq!(after.transfer.d2h_bytes, before.transfer.d2h_bytes);
            assert_eq!(
                after.kernel_launches,
                before.kernel_launches + if update == 1 || update == 2 { 4 } else { 3 }
            );
            assert!(session.solution().is_none());
            if update == 2 {
                let auxiliary = prepare_hex_lor_shifted_laplacian(
                    &mesh,
                    &space,
                    Complex64::new(36.0, 36.0),
                    &dirichlet,
                )
                .unwrap();
                session
                    .try_rebuild_preconditioner(&auxiliary, setup.amg.clone())
                    .unwrap();
                assert!(session.small_jacobi_enabled());
                assert!(session.fused_jacobi_enabled());
                let changed_auxiliary = prepare_hex_lor_shifted_laplacian(
                    &mesh,
                    &space,
                    Complex64::new(36.001, 36.001),
                    &dirichlet,
                )
                .unwrap();
                let before = b.begin_runtime_stats();
                assert!(
                    session
                        .try_refresh_preconditioner(&changed_auxiliary, setup.amg.clone())
                        .unwrap()
                );
                assert!(session.small_jacobi_enabled());
                assert!(session.fused_jacobi_enabled());
                let stats = b.runtime_stats_since(before);
                assert_eq!(stats.allocation_operations, 0);
                assert_eq!(stats.transfer.d2h_bytes, 4);
                eprintln!(
                    "hex coarse refresh {degree:?} aggregation={aggregation}: allocations={} h2d={} d2h={}",
                    stats.allocation_operations, stats.transfer.h2d_bytes, stats.transfer.d2h_bytes
                );
                let checkpoint = session.preconditioner.to_checkpoint_bytes().unwrap();
                let before = b.begin_runtime_stats();
                let different = AmgConfig {
                    coarse_size: n,
                    ..setup.amg.clone()
                };
                assert!(
                    !session
                        .try_refresh_preconditioner(&changed_auxiliary, different)
                        .unwrap()
                );
                let stats = b.runtime_stats_since(before);
                assert_eq!(stats.allocation_operations, 0);
                assert_eq!(stats.transfer.h2d_bytes + stats.transfer.d2h_bytes, 0);
                assert_eq!(
                    checkpoint,
                    session.preconditioner.to_checkpoint_bytes().unwrap()
                );
                b.inject_preconditioner_failure(
                    sonium_compute::PreconditionerFailureInjectionPoint::CoarseSolve,
                );
                assert!(
                    session
                        .try_refresh_preconditioner(&changed_auxiliary, setup.amg.clone())
                        .is_err()
                );
                assert!(session.small_jacobi_enabled());
                assert!(session.fused_jacobi_enabled());
                assert!(session.solution().is_none());
                assert!(session.preconditioner.to_checkpoint_bytes().is_err());
                b.recover_after_error().unwrap();
                assert!(
                    session
                        .try_refresh_preconditioner(&changed_auxiliary, setup.amg.clone())
                        .unwrap()
                );
                assert_eq!(session.wavenumber(), wavenumber);
            }
            let volume = assemble_bilinear_checked(&material, &space, &space, &mesh, 4)
                .unwrap()
                .to_csr();
            let mut terms = Vec::new();
            entries(&volume, &mut terms);
            let mut load = vec![Complex64::default(); n];
            for marker in [2, 3] {
                let boundary = prepare_hex_robin(
                    &mesh,
                    &space,
                    marker,
                    &|x, y, z| scale * alpha(marker, [x, y, z]),
                    &|x, y, z| (scale + 0.2) * source(marker, [x, y, z]),
                    4,
                    &[],
                )
                .unwrap();
                entries(&boundary.matrix, &mut terms);
                for (a, f) in load.iter_mut().zip(boundary.rhs) {
                    *a += f;
                }
            }
            let original = CsrMatrix::try_from_triplets(n, n, terms).unwrap();
            let exact: Vec<_> = (0..n)
                .map(|i| Complex64::new((0.13 * i as f64 + scale).sin(), (0.07 * i as f64).cos()))
                .collect();
            let full_rhs = apply(&original, &exact);
            let volume_rhs: Vec<_> = full_rhs.iter().zip(&load).map(|(b, f)| *b - *f).collect();
            let mut prescribed = vec![Complex64::new(f64::NAN, 0.0); n];
            for &i in &dirichlet {
                prescribed[i] = exact[i];
            }
            b.try_copy_h2d(&volume_rhs, rhs.raw_mut()).unwrap();
            b.try_copy_h2d(&prescribed, g.raw_mut()).unwrap();
            let before = b.runtime_stats();
            let status = session
                .solve(
                    rhs.raw(),
                    g.raw(),
                    if update == 0 {
                        DeviceInitialGuess::Zero
                    } else {
                        DeviceInitialGuess::PreviousSolution
                    },
                    &controls,
                    None,
                )
                .unwrap();
            assert!(
                status.converged,
                "degree={degree:?}, update={update}: {status:?}"
            );
            eprintln!(
                "frequency sweep {degree:?} aggregation={aggregation}: k={wavenumber}, iterations={}, residual={}",
                status.iterations, status.true_residual
            );
            if update > 0 {
                assert_eq!(
                    b.runtime_stats().allocation_operations,
                    before.allocation_operations
                );
            }
            let mut actual = vec![Complex64::default(); n];
            b.try_copy_d2h(session.solution().unwrap(), &mut actual)
                .unwrap();
            let action = apply(&original, &actual);
            let mut residual = 0.0;
            let mut norm = 0.0;
            for i in 0..n {
                if dirichlet.contains(&i) {
                    assert!((actual[i] - exact[i]).norm() < 1e-8);
                } else {
                    residual += (action[i] - full_rhs[i]).norm_sqr();
                    norm += full_rhs[i].norm_sqr();
                }
            }
            assert!((residual / norm).sqrt() < 1e-8);
            for invalid in [Complex64::new(f64::NAN, 0.0), Complex64::new(f64::MAX, 0.0)] {
                assert!(session.try_update_wavenumber(invalid).is_err());
                assert_eq!(session.wavenumber(), wavenumber);
                assert!(session.solution().is_some());
            }
            let wrong_size =
                CsrMatrix::try_from_triplets(1, 1, vec![(0, 0, Complex64::new(1.0, 0.0))]).unwrap();
            assert!(
                session
                    .try_rebuild_preconditioner(&wrong_size, setup.amg.clone())
                    .is_err()
            );
            assert!(session.solution().is_some());
        }
        b.inject_failure(sonium_compute::CudaFailureInjectionPoint::TransferD2D);
        assert!(session.try_update_volume_data(volume_data.raw()).is_err());
        assert!(session.solution().is_none());
        b.recover_after_error().unwrap();
        assert!(
            session
                .solve(
                    rhs.raw(),
                    g.raw(),
                    DeviceInitialGuess::Zero,
                    &controls,
                    None
                )
                .is_err()
        );
        session.try_update_volume_data(volume_data.raw()).unwrap();
        assert!(
            session
                .solve(
                    rhs.raw(),
                    g.raw(),
                    DeviceInitialGuess::PreviousSolution,
                    &controls,
                    None
                )
                .is_err()
        );
        let bad_source = pool.try_alloc::<Complex64>(metadata.len() - 1).unwrap();
        assert!(
            session
                .try_update_boundary(a.raw(), bad_source.raw())
                .is_err()
        );
        assert!(session.solution().is_none());
        assert!(
            session
                .solve(
                    rhs.raw(),
                    g.raw(),
                    DeviceInitialGuess::Zero,
                    &controls,
                    None
                )
                .is_err()
        );
        session.try_update_boundary(a.raw(), f.raw()).unwrap();
        assert!(
            session
                .solve(
                    rhs.raw(),
                    g.raw(),
                    DeviceInitialGuess::PreviousSolution,
                    &controls,
                    None
                )
                .is_err()
        );
        assert!(
            session
                .solve(
                    rhs.raw(),
                    g.raw(),
                    DeviceInitialGuess::Zero,
                    &controls,
                    None
                )
                .unwrap()
                .converged
        );
    }
}
