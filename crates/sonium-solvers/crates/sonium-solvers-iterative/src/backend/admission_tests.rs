//! Runtime dependency admission without requiring a CUDA device.

// Rust guideline compliant 2026-02-21
use super::*;
use sonium_compute::{CapabilityState, CpuBackend, ScalarCapabilities};

fn check<T: DeviceScalar>() {
    for method in [
        BackendKrylovMethod::Gmres,
        BackendKrylovMethod::Fgmres,
        BackendKrylovMethod::BlockGmres,
        BackendKrylovMethod::BlockFgmres,
    ] {
        let block = matches!(
            method,
            BackendKrylovMethod::BlockGmres | BackendKrylovMethod::BlockFgmres
        );
        for field in 0..if block { 7 } else { 5 } {
            for state in [
                CapabilityState::Unsupported,
                CapabilityState::HostStagedFallback,
                CapabilityState::HostSharedFallback,
                CapabilityState::Native,
                CapabilityState::VendorNative,
            ] {
                let mut caps = CpuBackend.capabilities();
                caps.backend_name = "test-runtime";
                let target = match field {
                    0 if block => &mut caps.csr_spmm,
                    0 => &mut caps.csr_spmv,
                    1 => &mut caps.blas1,
                    2 => &mut caps.axpy_many,
                    3 => &mut caps.multi_dot,
                    4 => &mut caps.reduction.fast,
                    5 => &mut caps.multi_dot_strided,
                    6 => &mut caps.basis_axpy_strided,
                    _ => unreachable!(),
                };
                *target = ScalarCapabilities::splat(state);
                let cell = capability_from_report::<T>(&caps, method, scalar_kind::<T>());
                assert_eq!(
                    cell.native_execution,
                    state.is_native(),
                    "{method:?} field={field}"
                );
                assert_eq!(
                    validate_capability_cell::<T>(&cell, true).is_ok(),
                    state.is_native()
                );
                assert_eq!(
                    validate_capability_cell::<T>(&cell, false).is_ok(),
                    state.is_available()
                );
            }
        }
        // Missing unrelated kernels must not reject a CSR Krylov solve.
        let mut caps = CpuBackend.capabilities();
        caps.gather4_weighted = ScalarCapabilities::splat(CapabilityState::Unsupported);
        caps.batched_local_contract =
            ScalarCapabilities::splat(CapabilityState::HostStagedFallback);
        let cell = capability_from_report::<T>(&caps, method, scalar_kind::<T>());
        validate_capability_cell::<T>(&cell, true).unwrap();
    }
}

#[test]
fn runtime_dependency_admission_all_scalars() {
    check::<f32>();
    check::<f64>();
    check::<num_complex::Complex32>();
    check::<num_complex::Complex64>();
}

#[test]
fn public_preflight_is_read_only() {
    let backend = CpuBackend;
    let before = backend.transfer_stats();
    validate_backend_krylov::<_, f64>(&backend, BackendKrylovMethod::Gmres, true).unwrap();
    assert_eq!(before, backend.transfer_stats());
}
