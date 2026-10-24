//! CUDA views transfer only selected elements and preserve surrounding storage.

// Rust guideline compliant 2026-02-21
#![cfg(feature = "cuda")]
use sonium_compute::{Backend, BufferElement, BufferView, CudaBackend, CudaFailureInjectionPoint};

fn backend() -> Option<CudaBackend> {
    match CudaBackend::new(0) {
        Ok(backend) => Some(backend),
        Err(error) => {
            assert!(
                std::env::var_os("SONIUM_REQUIRE_CUDA").is_none(),
                "CUDA required: {error}"
            );
            None
        }
    }
}

fn check<T: BufferElement + PartialEq + std::fmt::Debug>(
    backend: &CudaBackend,
    value: impl Fn(usize) -> T,
) {
    let original: Vec<_> = (0..64).map(&value).collect();
    let mut device = backend.raw_alloc::<T>(64).unwrap();
    for (offset, len, stride) in [(4, 3, 1), (1, 5, 3), (63, 1, 1), (64, 0, 1)] {
        backend.try_copy_h2d(&original, &mut device).unwrap();
        let view = BufferView::new(64, offset, len, stride).unwrap();
        let update: Vec<_> = (300..300 + len).map(&value).collect();
        let mut actual = vec![value(999); len];
        let before = backend.runtime_stats();
        backend
            .try_copy_view_h2d(&update, &mut device, view)
            .unwrap();
        backend
            .try_copy_view_d2h(&device, view, &mut actual)
            .unwrap();
        let after = backend.runtime_stats();
        assert_eq!(after.allocation_operations, before.allocation_operations);
        assert_eq!(
            after.transfer.h2d_bytes - before.transfer.h2d_bytes,
            std::mem::size_of_val(update.as_slice())
        );
        assert_eq!(
            after.transfer.d2h_bytes - before.transfer.d2h_bytes,
            std::mem::size_of_val(update.as_slice())
        );
        assert_eq!(actual, update);
        let mut full = original.clone();
        backend.try_copy_d2h(&device, &mut full).unwrap();
        let mut expected = original.clone();
        for (i, v) in update.into_iter().enumerate() {
            expected[offset + i * stride] = v;
        }
        assert_eq!(full, expected);
    }
}

#[test]
fn cuda_selected_transfers_cover_layouts_and_scalar_sizes() {
    let Some(backend) = backend() else {
        return;
    };
    check(&backend, |i| i as f32);
    check(&backend, |i| i as f64);
    check(&backend, |i| {
        num_complex::Complex32::new(i as f32, -(i as f32))
    });
    check(&backend, |i| {
        num_complex::Complex64::new(i as f64, -(i as f64))
    });
    check(&backend, |i| i as i32);
}

#[test]
fn cuda_view_failures_preserve_destinations() {
    let Some(backend) = backend() else {
        return;
    };
    let mut device = backend.raw_alloc::<f64>(8).unwrap();
    backend.try_copy_h2d(&[7.0; 8], &mut device).unwrap();
    let view = BufferView::new(8, 2, 3, 1).unwrap();
    assert!(
        backend
            .try_copy_view_h2d(&[1.0; 2], &mut device, view)
            .is_err()
    );
    let mut host = [9.0; 2];
    assert!(backend.try_copy_view_d2h(&device, view, &mut host).is_err());
    assert_eq!(host, [9.0; 2]);
    backend.inject_failure(CudaFailureInjectionPoint::TransferH2D);
    assert!(
        backend
            .try_copy_view_h2d(&[1.0; 3], &mut device, view)
            .is_err()
    );
    backend.recover_after_error().unwrap();
    let mut full = [0.0; 8];
    backend.try_copy_d2h(&device, &mut full).unwrap();
    assert_eq!(full, [7.0; 8]);
    backend.inject_failure(CudaFailureInjectionPoint::TransferD2H);
    let mut host = [9.0; 3];
    assert!(backend.try_copy_view_d2h(&device, view, &mut host).is_err());
    assert_eq!(host, [9.0; 3]);
    backend.recover_after_error().unwrap();
    let other = CudaBackend::new(0).unwrap();
    assert!(
        other
            .try_copy_view_h2d(&[1.0; 3], &mut device, view)
            .is_err()
    );
    assert!(other.try_copy_view_d2h(&device, view, &mut host).is_err());
    backend.try_copy_d2h(&device, &mut full).unwrap();
    assert_eq!(full, [7.0; 8]);
}
