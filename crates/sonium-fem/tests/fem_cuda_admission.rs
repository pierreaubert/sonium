use sonium_fem::{
    FemAction, FemActualBackend, FemAdmissionError, FemAssemblyMode, FemBackend, FemBackendRequest,
    FemElement, FemOrder, FemPhysics, FemScalar, FemSpace, FemWorkload, admit_fem_cuda,
    admit_fem_workload,
};

fn workload() -> FemWorkload {
    FemWorkload::native_cuda(FemScalar::F64, FemAction::Forward)
}

fn assert_rejected(workload: FemWorkload) {
    let error = admit_fem_cuda(workload).expect_err("unsupported CUDA workload must be rejected");
    let FemAdmissionError::Unsupported {
        requested_backend,
        actual_backend,
        fallback,
        fallback_reason,
        ..
    } = error
    else {
        panic!("unsupported workload returned a different error");
    };
    assert_eq!(requested_backend, FemBackend::Cuda);
    assert_eq!(actual_backend, FemActualBackend::NotExecuted);
    assert_eq!(fallback, "none");
    assert!(fallback_reason.contains("rejected before assembly"));
}

#[test]
fn unsupported_element_rejects_before_assembly() {
    let unsupported = FemWorkload {
        element: FemElement::Tetrahedron,
        ..workload()
    };
    let error = admit_fem_cuda(unsupported).expect_err("tetrahedron CUDA must be rejected");
    let FemAdmissionError::Unsupported {
        requested_backend,
        actual_backend,
        fallback,
        fallback_reason,
        ..
    } = error
    else {
        panic!("unsupported element returned a different error");
    };
    assert_eq!(requested_backend, FemBackend::Cuda);
    assert_eq!(actual_backend, FemActualBackend::NotExecuted);
    assert_eq!(fallback, "none");
    assert!(fallback_reason.contains("before assembly"));
}

#[test]
fn unsupported_physics_and_order_reject_without_dispatch() {
    let other_physics = FemWorkload {
        physics: FemPhysics::Other,
        ..workload()
    };
    assert_rejected(other_physics);

    let higher_order = FemWorkload {
        order: FemOrder::P2OrHigher,
        ..workload()
    };
    assert_rejected(higher_order);

    for scalar in [FemScalar::F32, FemScalar::Complex32] {
        assert_rejected(FemWorkload {
            scalar,
            ..workload()
        });
    }
    for element in [
        FemElement::Quad,
        FemElement::Tetrahedron,
        FemElement::Hexahedron,
    ] {
        assert_rejected(FemWorkload {
            element,
            ..workload()
        });
    }
    for space in [FemSpace::Hcurl, FemSpace::Hdiv, FemSpace::Dg] {
        assert_rejected(FemWorkload {
            space,
            ..workload()
        });
    }
    for mode in [FemAssemblyMode::PartialAssembly] {
        assert_rejected(FemWorkload { mode, ..workload() });
    }
}

#[test]
fn p2_helmholtz_is_admitted_as_an_explicit_bounded_cell() {
    let admission = admit_fem_cuda(FemWorkload::native_cuda_p2(
        FemScalar::Complex64,
        FemAction::Hermitian,
    ))
    .expect("P2/Tri3 Helmholtz CUDA cell should be admitted");
    assert_eq!(admission.provenance.actual_backend, FemActualBackend::Cuda);
    assert_eq!(admission.provenance.fallback, "none");
    assert_eq!(admission.workload.order, FemOrder::P2);
}

#[test]
fn fallback_provenance_records_requested_and_actual_backend() {
    let unsupported = FemWorkload {
        scalar: FemScalar::Complex32,
        ..workload()
    };
    let admission = admit_fem_workload(unsupported, FemBackendRequest::with_cpu_fallback())
        .expect("explicit CPU fallback should be admitted");
    assert_eq!(admission.provenance.requested_backend, FemBackend::Cuda);
    assert_eq!(admission.provenance.actual_backend, FemActualBackend::Cpu);
    assert_eq!(admission.provenance.fallback, "cpu");
    assert!(
        admission
            .provenance
            .fallback_reason
            .contains("CUDA workload unsupported")
    );
}

#[test]
fn tetrahedral_pressure_rows_match_bounded_public_admission() {
    use sonium_fem::backend::{FemCapabilityStatus, fem_cuda_capability_rows};
    let rows = fem_cuda_capability_rows();
    for order in [FemOrder::P1, FemOrder::P2] {
        for action in [
            FemAction::Forward,
            FemAction::Transpose,
            FemAction::Hermitian,
        ] {
            let cell = FemWorkload::native_cuda_tet(order, action);
            let admission = admit_fem_cuda(cell).unwrap();
            assert_eq!(admission.provenance.actual_backend, FemActualBackend::Cuda);
            let matches: Vec<_> = rows.iter().filter(|row| row.id == cell.id()).collect();
            assert_eq!(matches.len(), 1);
            assert_eq!(matches[0].status, FemCapabilityStatus::Native);
            assert!(matches[0].dispatch);
            assert!(
                matches[0]
                    .evidence
                    .as_ref()
                    .unwrap()
                    .contains("host tetrahedral")
            );
        }
    }
    let base = FemWorkload::native_cuda_tet(FemOrder::P1, FemAction::Forward);
    for rejected in [
        FemWorkload {
            scalar: FemScalar::F64,
            ..base
        },
        FemWorkload {
            scalar: FemScalar::Complex32,
            ..base
        },
        FemWorkload {
            order: FemOrder::P2OrHigher,
            ..base
        },
        FemWorkload {
            mode: FemAssemblyMode::MatrixFree,
            ..base
        },
        FemWorkload {
            mode: FemAssemblyMode::PartialAssembly,
            ..base
        },
        FemWorkload {
            space: FemSpace::Hcurl,
            ..base
        },
    ] {
        assert_rejected(rejected);
    }
}

#[test]
fn complex_triangle_transpose_has_native_assembled_rows() {
    use sonium_fem::backend::{FemCapabilityStatus, fem_cuda_capability_rows};
    let rows = fem_cuda_capability_rows();
    for workload in [
        FemWorkload::native_cuda(FemScalar::Complex64, FemAction::Transpose),
        FemWorkload::native_cuda_p2(FemScalar::Complex64, FemAction::Transpose),
    ] {
        let admission = admit_fem_cuda(workload).unwrap();
        assert_eq!(admission.provenance.actual_backend, FemActualBackend::Cuda);
        assert_eq!(admission.provenance.fallback, "none");
        let matching: Vec<_> = rows.iter().filter(|row| row.id == workload.id()).collect();
        assert_eq!(matching.len(), 1);
        assert_eq!(matching[0].status, FemCapabilityStatus::Native);
        assert!(matching[0].dispatch);
        assert!(
            matching[0]
                .evidence
                .as_deref()
                .unwrap()
                .contains("retained CUDA")
        );
    }
}

#[test]
fn hexahedral_pressure_rows_require_explicit_order_and_assembled_complex64() {
    use sonium_fem::backend::{FemCapabilityStatus, fem_cuda_capability_rows};
    let rows = fem_cuda_capability_rows();
    for order in [FemOrder::P1, FemOrder::P2, FemOrder::P3] {
        for action in [
            FemAction::Forward,
            FemAction::Transpose,
            FemAction::Hermitian,
        ] {
            let workload = FemWorkload::native_cuda_hex(order, action);
            let admission = admit_fem_cuda(workload).unwrap();
            assert_eq!(admission.provenance.actual_backend, FemActualBackend::Cuda);
            assert_eq!(admission.provenance.fallback, "none");
            let matching: Vec<_> = rows.iter().filter(|row| row.id == workload.id()).collect();
            assert_eq!(matching.len(), 1);
            assert_eq!(matching[0].status, FemCapabilityStatus::Native);
            assert!(matching[0].dispatch);
            assert!(
                matching[0]
                    .evidence
                    .as_deref()
                    .unwrap()
                    .contains("LOR auxiliary AMG")
            );
        }
    }
    let base = FemWorkload::native_cuda_hex(FemOrder::P3, FemAction::Forward);
    for rejected in [
        FemWorkload {
            scalar: FemScalar::F64,
            ..base
        },
        FemWorkload {
            scalar: FemScalar::F32,
            ..base
        },
        FemWorkload {
            scalar: FemScalar::Complex32,
            ..base
        },
        FemWorkload {
            order: FemOrder::P2OrHigher,
            ..base
        },
        FemWorkload {
            mode: FemAssemblyMode::MatrixFree,
            ..base
        },
        FemWorkload {
            mode: FemAssemblyMode::PartialAssembly,
            ..base
        },
        FemWorkload {
            space: FemSpace::Hdiv,
            ..base
        },
        FemWorkload {
            element: FemElement::Tri3,
            ..base
        },
        FemWorkload {
            element: FemElement::Tetrahedron,
            ..base
        },
        FemWorkload {
            physics: FemPhysics::Other,
            ..base
        },
    ] {
        assert_rejected(rejected);
    }
    // The explicit cubic order must round-trip independently of the old bucket.
    let value = serde_json::to_value(base).unwrap();
    assert_eq!(value["order"], "p3");
    assert_eq!(serde_json::from_value::<FemWorkload>(value).unwrap(), base);
    assert_ne!(FemOrder::P3, FemOrder::P2OrHigher);
}

#[test]
fn electrostatic_potential_admission_is_explicit_and_bounded() {
    for element in [FemElement::Tri3, FemElement::Tetrahedron] {
        for order in [FemOrder::P1, FemOrder::P2] {
            for action in [
                FemAction::Forward,
                FemAction::Transpose,
                FemAction::Hermitian,
            ] {
                let workload =
                    FemWorkload::native_cuda_electrostatic_potential(element, order, action);
                assert_eq!(workload.physics, FemPhysics::ElectrostaticPotential);
                admit_fem_cuda(workload).unwrap();
                for scalar in [FemScalar::F32, FemScalar::Complex32, FemScalar::Complex64] {
                    assert_rejected(FemWorkload { scalar, ..workload });
                }
                for mode in [
                    FemAssemblyMode::PartialAssembly,
                    FemAssemblyMode::MatrixFree,
                ] {
                    assert_rejected(FemWorkload { mode, ..workload });
                }
                for space in [FemSpace::Hcurl, FemSpace::Hdiv, FemSpace::Dg] {
                    assert_rejected(FemWorkload { space, ..workload });
                }
                for order in [FemOrder::P3, FemOrder::P2OrHigher] {
                    assert_rejected(FemWorkload { order, ..workload });
                }
                for element in [FemElement::Quad, FemElement::Hexahedron] {
                    assert_rejected(FemWorkload {
                        element,
                        ..workload
                    });
                }
            }
        }
    }
}
