//! Numeric updates preserve persistent sparse actions and their device storage.

// Rust guideline compliant 2026-02-21
use num_complex::{Complex32, Complex64};
use num_traits::{FromPrimitive, ToPrimitive};
use sonium_compute::{
    Backend, CpuBackend, DeviceScalar, LowerPrecision, MemoryPool, MemoryPoolConfig,
    PersistentDeviceCsr,
};

fn combine<B: Backend, T: DeviceScalar>(
    pool: &MemoryPool<B>,
    value: T,
    tolerance: f64,
    allocations: impl Fn() -> usize,
) {
    let backend = pool.backend();
    let n = 17;
    let mut matrix =
        PersistentDeviceCsr::try_from_host_diagonal(backend, &vec![T::zero(); n]).unwrap();
    let mut base = pool.try_alloc::<T>(n).unwrap();
    let mut term = pool.try_alloc::<T>(n).unwrap();
    let mut x = pool.try_alloc::<T>(n).unwrap();
    let mut y = pool.try_alloc::<T>(n).unwrap();
    backend
        .try_copy_h2d(&vec![T::one(); n], x.raw_mut())
        .unwrap();
    backend
        .try_copy_h2d(&vec![value; n], base.raw_mut())
        .unwrap();
    backend
        .try_copy_h2d(&vec![T::one(); n], term.raw_mut())
        .unwrap();
    matrix.try_apply(backend, x.raw(), y.raw_mut()).unwrap();
    for alpha in [T::zero(), -value, value, T::one()] {
        let before = allocations();
        let transfers = backend.transfer_stats();
        matrix
            .try_combine_values(
                backend,
                base.raw(),
                &[(alpha, term.raw()), (value, term.raw())],
            )
            .unwrap();
        assert_eq!(allocations(), before);
        let after = backend.transfer_stats();
        assert_eq!(after.h2d_bytes, transfers.h2d_bytes);
        assert_eq!(after.d2h_bytes, transfers.d2h_bytes);
        matrix.try_apply(backend, x.raw(), y.raw_mut()).unwrap();
        let mut actual = vec![T::zero(); n];
        backend.try_copy_d2h(y.raw(), &mut actual).unwrap();
        let expected = value + alpha + value;
        assert!(
            actual
                .iter()
                .all(|v| (*v - expected).magnitude().to_f64().unwrap() < tolerance)
        );
    }
    let mut previous = vec![T::zero(); n];
    backend.try_copy_d2h(y.raw(), &mut previous).unwrap();
    let short = pool.try_alloc::<T>(n - 1).unwrap();
    let nan = T::from_real(T::Real::from_f64(f64::NAN).unwrap());
    for terms in [
        vec![(T::one(), term.raw()), (T::one(), short.raw())],
        vec![(nan, term.raw())],
    ] {
        assert!(
            matrix
                .try_combine_values(backend, base.raw(), &terms)
                .is_err()
        );
        matrix.try_apply(backend, x.raw(), y.raw_mut()).unwrap();
        let mut actual = vec![T::zero(); n];
        backend.try_copy_d2h(y.raw(), &mut actual).unwrap();
        assert!(
            actual
                .iter()
                .zip(&previous)
                .all(|(a, b)| (*a - *b).magnitude().to_f64().unwrap() == 0.)
        );
    }
    // Empty terms replace all values, including a previously nonzero matrix.
    backend
        .try_copy_h2d(&vec![T::zero(); n], base.raw_mut())
        .unwrap();
    matrix.try_combine_values(backend, base.raw(), &[]).unwrap();
    matrix.try_apply(backend, x.raw(), y.raw_mut()).unwrap();
    let mut actual = vec![T::one(); n];
    backend.try_copy_d2h(y.raw(), &mut actual).unwrap();
    assert!(actual.iter().all(|v| v.magnitude().to_f64().unwrap() == 0.));
}

fn promote<B: Backend, T: LowerPrecision>(
    pool: &MemoryPool<B>,
    value: T,
    allocations: impl Fn() -> usize,
    native_transfers: bool,
) {
    let backend = pool.backend();
    let mut matrix = PersistentDeviceCsr::try_from_host_diagonal(backend, &[T::zero(); 3]).unwrap();
    let mut lower = pool.try_alloc::<T::Lower>(3).unwrap();
    backend
        .try_copy_h2d(&[value.cast_to_lower(); 3], lower.raw_mut())
        .unwrap();
    let before = allocations();
    let transfer = backend.transfer_stats();
    matrix
        .try_cast_values_from_lower(backend, lower.raw())
        .unwrap();
    assert_eq!(allocations(), before);
    if native_transfers {
        assert_eq!(backend.transfer_stats().h2d_bytes, transfer.h2d_bytes);
        assert_eq!(backend.transfer_stats().d2h_bytes, transfer.d2h_bytes);
    }
    let mut actual = [T::zero(); 3];
    backend
        .try_copy_d2h(&matrix.matrix().values, &mut actual)
        .unwrap();
    assert!(actual.iter().all(|a| {
        (*a - T::cast_from_lower(value.cast_to_lower()))
            .magnitude()
            .to_f64()
            .unwrap()
            == 0.
    }));
    let bad = pool.try_alloc::<T::Lower>(2).unwrap();
    assert!(
        matrix
            .try_cast_values_from_lower(backend, bad.raw())
            .is_err()
    );
}

fn exercise<B: Backend>(
    pool: &MemoryPool<B>,
    allocations: impl Fn() -> usize + Copy,
    native_transfers: bool,
) {
    combine(pool, 0.375_f32, 1e-6, allocations);
    combine(pool, 0.375_f64, 1e-14, allocations);
    combine(pool, Complex32::new(0.375, -0.5), 1e-6, allocations);
    combine(pool, Complex64::new(0.375, -0.5), 1e-14, allocations);
    promote(pool, 1.25_f64, allocations, native_transfers);
    promote(
        pool,
        Complex64::new(1.25, -0.75),
        allocations,
        native_transfers,
    );
}

#[test]
fn cpu_persistent_numeric_updates() {
    let pool = MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(1 << 20)).unwrap();
    exercise(&pool, || pool.bytes_in_use(), false);
}

#[cfg(feature = "cuda")]
#[test]
fn native_persistent_numeric_updates() {
    let backend = match sonium_compute::CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            assert!(std::env::var_os("SONIUM_REQUIRE_CUDA").is_none(), "{error}");
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
    exercise(
        &pool,
        || pool.backend().runtime_stats().allocation_operations,
        true,
    );
}
