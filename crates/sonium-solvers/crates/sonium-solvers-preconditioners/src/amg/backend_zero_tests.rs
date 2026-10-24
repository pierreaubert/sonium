//! Independent cycle reference for zero-start smoothing and W/F revisits.

// Rust guideline compliant 2026-02-21
use super::*;
use num_complex::Complex64 as C;
use sonium_compute::{Backend, CpuBackend, DevicePreconditioner, MemoryPool, MemoryPoolConfig};

fn action(matrix: &CsrMatrix<C>, x: &[C]) -> Vec<C> {
    (0..matrix.num_rows)
        .map(|row| {
            (matrix.row_ptrs[row]..matrix.row_ptrs[row + 1])
                .map(|j| matrix.values[j] * x[matrix.col_indices[j]])
                .sum()
        })
        .collect()
}

fn relax(host: &AmgPreconditioner<C>, level: usize, x: &mut [C], b: &[C], sweeps: usize) {
    let matrix = &host.levels[level].matrix;
    if host.config.smoother == AmgSmoother::Chebyshev {
        // Independent scalar recurrence: always form A*x, even for zero x.
        // This catches accidental restarts after the optimized first sweep.
        let upper = host.levels[level].lambda_max.unwrap();
        let lower = upper / 30.0;
        let center = (upper + lower) / 2.0;
        let radius = (upper - lower) / 2.0;
        let sigma = center / radius;
        let mut rho = 1.0 / sigma;
        let mut direction = vec![C::default(); x.len()];
        for sweep in 0..sweeps {
            let (old_weight, residual_weight) = if sweep == 0 {
                (0.0, 1.0 / center)
            } else {
                let next = 1.0 / (2.0 * sigma - rho);
                let weights = (next * rho, 2.0 * next / radius);
                rho = next;
                weights
            };
            let ax = action(matrix, x);
            for row in 0..x.len() {
                direction[row] = old_weight * direction[row]
                    + residual_weight * host.levels[level].diag_inv[row] * (b[row] - ax[row]);
                x[row] += direction[row];
            }
        }
        return;
    }
    for _ in 0..sweeps {
        let ax = action(matrix, x);
        for row in 0..x.len() {
            let delta = if host.config.smoother == AmgSmoother::L1Jacobi {
                let norm: f64 = (matrix.row_ptrs[row]..matrix.row_ptrs[row + 1])
                    .map(|j| matrix.values[j].norm())
                    .sum();
                (b[row] - ax[row]) / norm
            } else {
                host.config.jacobi_weight * host.levels[level].diag_inv[row] * (b[row] - ax[row])
            };
            x[row] += delta;
        }
    }
}

fn reference(host: &AmgPreconditioner<C>, level: usize, x: &mut [C], b: &[C], cycle: AmgCycle) {
    let entry = &host.levels[level];
    let Some(p) = entry.prolongation.as_ref() else {
        // The tested backend hierarchy has no attached direct coarse solver.
        relax(host, level, x, b, BACKEND_AMG_COARSE_SWEEPS);
        return;
    };
    relax(host, level, x, b, host.config.num_pre_smooth);
    let residual: Vec<_> = b
        .iter()
        .zip(action(&entry.matrix, x))
        .map(|(b, ax)| b - ax)
        .collect();
    let coarse_rhs = action(entry.restriction.as_ref().unwrap(), &residual);
    let mut correction = vec![C::default(); coarse_rhs.len()];
    reference(host, level + 1, &mut correction, &coarse_rhs, cycle);
    match cycle {
        AmgCycle::VCycle => {}
        AmgCycle::WCycle => reference(
            host,
            level + 1,
            &mut correction,
            &coarse_rhs,
            AmgCycle::WCycle,
        ),
        AmgCycle::FCycle => reference(
            host,
            level + 1,
            &mut correction,
            &coarse_rhs,
            AmgCycle::VCycle,
        ),
    }
    for (x, delta) in x.iter_mut().zip(action(p, &correction)) {
        *x += delta;
    }
    relax(host, level, x, b, host.config.num_post_smooth);
}

fn check<B: Backend>(pool: &MemoryPool<B>) {
    let n = 64;
    let mut entries = Vec::new();
    for i in 0..n {
        entries.push((i, i, C::new(3.2 + 0.01 * i as f64, 0.2)));
        if i + 1 < n {
            entries.extend([(i, i + 1, C::new(-1.0, 0.1)), (i + 1, i, C::new(-0.9, 0.1))]);
        }
    }
    let matrix = CsrMatrix::from_triplets(n, n, entries);
    let backend = pool.backend();
    let mut rhs = pool.try_alloc::<C>(n * 2).unwrap();
    let mut out = pool.try_alloc::<C>(n * 2).unwrap();
    for smoother in [
        AmgSmoother::Jacobi,
        AmgSmoother::L1Jacobi,
        AmgSmoother::Chebyshev,
    ] {
        for cycle in [AmgCycle::VCycle, AmgCycle::WCycle, AmgCycle::FCycle] {
            for pre in [0, 1, 3] {
                let host = AmgPreconditioner::try_from_csr(
                    &matrix,
                    AmgConfig {
                        smoother,
                        cycle,
                        num_pre_smooth: pre,
                        num_post_smooth: 2,
                        coarse_size: 2,
                        jacobi_weight: 0.6,
                        ..Default::default()
                    },
                )
                .unwrap();
                assert!(host.levels.len() >= 3);
                let mut device =
                    BackendAmgPreconditioner::try_from_host_amg(backend, &host).unwrap();
                assert!(!device.diagnostics().has_coarse_direct_solver);
                for pass in 0..4 {
                    device.set_small_jacobi_enabled(pass == 2);
                    device.set_fused_jacobi_enabled(pass == 3);
                    // Profiling must preserve the independent V/W/F reference,
                    // numeric workspace and transfer behavior, then disable cleanly.
                    device.set_phase_profiling(pass == 1);
                    if pass == 1 {
                        assert!(device.phase_profile().iter().all(|p| p.calls == 0));
                    } else {
                        assert!(device.phase_profile().is_empty());
                    }
                    let values: Vec<_> = (0..n * 2)
                        .map(|i| C::new(((i + pass) as f64).sin(), 0.2 * pass as f64))
                        .collect();
                    backend.try_copy_h2d(&values, rhs.raw_mut()).unwrap();
                    backend
                        .try_copy_h2d(&vec![C::new(f64::NAN, f64::NAN); n * 2], out.raw_mut())
                        .unwrap();
                    let before = backend.transfer_stats();
                    let memory_before = backend.memory_stats();
                    device
                        .try_apply_colmajor(backend, 2, rhs.raw(), out.raw_mut())
                        .unwrap();
                    let after = backend.transfer_stats();
                    assert_eq!(before.h2d_bytes, after.h2d_bytes);
                    assert_eq!(before.d2h_bytes, after.d2h_bytes);
                    if pass > 0 {
                        assert_eq!(memory_before, backend.memory_stats());
                    }
                    if pass == 1 {
                        let profile = device.phase_profile();
                        assert_eq!(profile.len(), host.levels.len() * 6);
                        for entry in profile {
                            let coarsest = entry.level + 1 == host.levels.len();
                            let active = coarsest == (entry.phase == BackendAmgPhase::CoarseSolve);
                            if active {
                                let visits_per_column = match cycle {
                                    AmgCycle::VCycle => 1,
                                    AmgCycle::WCycle => 1_u64 << entry.level,
                                    AmgCycle::FCycle => entry.level as u64 + 1,
                                };
                                assert_eq!(entry.calls, 2 * visits_per_column, "{entry:?}");
                                assert!(entry.elapsed_ms.is_finite() && entry.elapsed_ms >= 0.0);
                            } else {
                                assert_eq!(entry.calls, 0, "{entry:?}");
                                assert_eq!(entry.elapsed_ms, 0.0);
                            }
                        }
                    } else {
                        assert!(device.phase_profile().is_empty());
                    }
                    let mut actual = vec![C::default(); n * 2];
                    backend.try_copy_d2h(out.raw(), &mut actual).unwrap();
                    for column in 0..2 {
                        let mut expected = vec![C::default(); n];
                        reference(
                            &host,
                            0,
                            &mut expected,
                            &values[column * n..(column + 1) * n],
                            cycle,
                        );
                        for (a, e) in actual[column * n..(column + 1) * n].iter().zip(expected) {
                            assert!(
                                (*a - e).norm() < 3e-12 * (1.0 + e.norm()),
                                "{smoother:?}/{cycle:?}/pre={pre}: {a} != {e}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn backend_zero_start_matches_independent_cycles() {
    let pool = MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(1 << 25)).unwrap();
    check(&pool);
}

#[cfg(feature = "cuda")]
#[test]
fn native_zero_start_matches_independent_cycles() {
    let backend = match sonium_compute::CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            assert!(
                std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                "CUDA required: {error}"
            );
            eprintln!("SKIP CUDA zero-start cycles: {error}");
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 25)).unwrap();
    check(&pool);
}

#[cfg(feature = "cuda")]
#[test]
fn native_small_jacobi_cycles_all_scalars() {
    let backend = match sonium_compute::CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            assert!(
                std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                "{error}"
            );
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 27)).unwrap();
    let backend = pool.backend();
    macro_rules! check_scalar {
        ($t:ty, $tol:expr) => {{
            let n = 513;
            let scalar = |v| <$t as num_traits::FromPrimitive>::from_f64(v).unwrap();
            let mut entries = Vec::new();
            for i in 0..n {
                entries.push((i, i, scalar(3.0)));
                if i + 1 < n {
                    entries.extend([(i, i + 1, scalar(-1.0)), (i + 1, i, scalar(-1.0))]);
                }
            }
            let matrix = CsrMatrix::from_triplets(n, n, entries);
            let mut rhs = pool.try_alloc::<$t>(n).unwrap();
            let mut out = pool.try_alloc::<$t>(n).unwrap();
            backend
                .try_copy_h2d(
                    &(0..n).map(|i| scalar((i as f64).sin())).collect::<Vec<_>>(),
                    rhs.raw_mut(),
                )
                .unwrap();
            for cycle in [AmgCycle::VCycle, AmgCycle::WCycle, AmgCycle::FCycle] {
                let mut amg = BackendAmgPreconditioner::try_from_csr(
                    backend,
                    &matrix,
                    AmgConfig {
                        cycle,
                        smoother: AmgSmoother::Jacobi,
                        num_pre_smooth: 3,
                        num_post_smooth: 2,
                        coarse_size: 8,
                        jacobi_weight: 0.6,
                        ..Default::default()
                    },
                )
                .unwrap();
                assert!(amg.diagnostics().level_dofs.iter().any(|&n| n <= 256));
                let mut reference = vec![<$t>::default(); n];
                let mut launches = Vec::new();
                for enabled in [false, true, false] {
                    amg.set_small_jacobi_enabled(enabled);
                    amg.try_apply(backend, rhs.raw(), out.raw_mut()).unwrap();
                    let before = backend.begin_runtime_stats();
                    amg.try_apply(backend, rhs.raw(), out.raw_mut()).unwrap();
                    let stats = backend.runtime_stats_since(before);
                    assert_eq!(stats.allocation_operations, 0);
                    assert_eq!(stats.transfer.h2d_bytes, 0);
                    assert_eq!(stats.transfer.d2h_bytes, 0);
                    launches.push(stats.kernel_launches + stats.library_operations);
                    let mut actual = vec![<$t>::default(); n];
                    backend.try_copy_d2h(out.raw(), &mut actual).unwrap();
                    if launches.len() == 1 {
                        reference = actual;
                    } else {
                        for (a, r) in actual.iter().zip(&reference) {
                            assert!(
                                <$t as sonium_compute::DeviceScalar>::norm_sqr(&(*a - *r))
                                    < $tol * $tol
                            );
                        }
                    }
                }
                assert!(launches[1] < launches[0], "{cycle:?}: {launches:?}");
                assert_eq!(launches[0], launches[2]);
            }
        }};
    }
    check_scalar!(f32, 2e-6);
    check_scalar!(f64, 2e-13);
    check_scalar!(num_complex::Complex32, 2e-6);
    check_scalar!(num_complex::Complex64, 2e-13);
}
