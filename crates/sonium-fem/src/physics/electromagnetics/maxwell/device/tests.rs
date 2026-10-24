//! Physical Maxwell and auxiliary-space contracts.

// Rust guideline compliant 2026-02-21
use super::*;
use crate::mesh::{unit_cube_tetrahedra, unit_square_triangles};
use crate::quadrature::QuadratureRule;
use crate::spaces::FunctionSpace;
use sonium_compute::{CudaFailureInjectionPoint, MemoryPoolConfig};
use sonium_solvers::iterative::{BackendKrylovConfig, DeviceInitialGuess};
use std::f64::consts::PI;

fn config() -> MaxwellCudaConfig {
    MaxwellCudaConfig {
        curl_coefficient: 1.0,
        mass_coefficient: C64::new(-1.0, 0.25),
        auxiliary_mass: 1.0,
    }
}

// A transverse field plus an irrotational mode, both with zero PEC trace.
fn exact(x: f64, y: f64) -> [f64; 2] {
    [
        (PI * y).sin() + 0.2 * PI * (PI * x).cos() * (PI * y).sin(),
        0.2 * PI * (PI * x).sin() * (PI * y).cos(),
    ]
}

fn current(x: f64, y: f64, _: f64) -> [C64; 3] {
    let e = exact(x, y);
    let mass = config().mass_coefficient;
    let i = C64::new(0.0, 1.0);
    [
        i * (PI * PI * (PI * y).sin() + mass * e[0]),
        i * mass * e[1],
        C64::default(),
    ]
}

fn exact_3d(x: f64, y: f64, z: f64) -> [f64; 3] {
    let (sx, cx) = (PI * x).sin_cos();
    let (sy, cy) = (PI * y).sin_cos();
    let (sz, cz) = (PI * z).sin_cos();
    [
        sy * sz + 0.2 * PI * cx * sy * sz,
        0.2 * PI * sx * cy * sz,
        0.2 * PI * sx * sy * cz,
    ]
}

fn current_for(dim: usize, action: FemAction, x: f64, y: f64, z: f64) -> [C64; 3] {
    let mut load = if dim == 2 {
        current(x, y, z)
    } else {
        // The transverse field is independent of x and divergence-free, so
        // curl curl E = 2*pi^2*(sin(pi*y)*sin(pi*z),0,0). The gradient has zero curl.
        let e = exact_3d(x, y, z);
        let mut load = e.map(|v| C64::new(0.0, 1.0) * config().mass_coefficient * v);
        load[0] += C64::new(0.0, 2.0 * PI * PI * (PI * y).sin() * (PI * z).sin());
        load
    };
    if action == FemAction::Hermitian {
        // J=i*f: the adjoint load is i*conj(f), hence -conj(J).
        for value in &mut load {
            *value = -value.conj();
        }
    }
    load
}

fn field_error(mesh: &Mesh, coefficients: &[C64]) -> f64 {
    let space = HcurlSpace::new();
    let mut error = 0.0;
    let mut norm = 0.0;
    for element in 0..mesh.elements.len() {
        let dofs = space.local_to_global(mesh, element);
        for q in QuadratureRule::new(mesh.elements[element].element_type, 7).iter() {
            let basis = space
                .try_tabulate(mesh, element, q.xi(), q.eta(), q.zeta())
                .unwrap();
            let [x, y, z] = basis.physical_coords;
            let reference = if mesh.dimension == 3 {
                exact_3d(x, y, z)
            } else {
                let [ex, ey] = exact(x, y);
                [ex, ey, 0.0]
            };
            for (d, &expected) in reference.iter().take(mesh.dimension).enumerate() {
                let actual: C64 = dofs
                    .iter()
                    .enumerate()
                    .map(|(j, &global)| coefficients[global] * basis.vector_value(j, d))
                    .sum();
                error += q.weight * basis.det_j * (actual - expected).norm_sqr();
                norm += q.weight * basis.det_j * expected * expected;
            }
        }
    }
    (error / norm).sqrt()
}

#[test]
fn auxiliary_maps_preserve_exact_sequence_and_pec() {
    let mut reversed = unit_cube_tetrahedra(3);
    reverse_node_numbering(&mut reversed);
    for mesh in [unit_square_triangles(4), unit_cube_tetrahedra(3), reversed] {
        check_maps(&mesh);
    }
    let mut nonmanifold = unit_cube_tetrahedra(2);
    nonmanifold.elements.push(nonmanifold.elements[0].clone());
    nonmanifold.elements.push(nonmanifold.elements[0].clone());
    assert!(prepare(&nonmanifold, config()).is_err());
}

fn reverse_node_numbering(mesh: &mut Mesh) {
    mesh.nodes.reverse();
    let last = mesh.nodes.len() - 1;
    for element in &mut mesh.elements {
        for node in &mut element.nodes {
            *node = last - *node;
        }
    }
    for face in &mut mesh.boundaries {
        for node in &mut face.nodes {
            *node = last - *node;
        }
    }
}

fn check_maps(mesh: &Mesh) {
    let prepared = prepare(mesh, config()).unwrap();
    let topology = MeshTopology::try_build(mesh).unwrap();
    for (edge, &(a, b)) in topology.edges.iter().enumerate() {
        let a = &mesh.nodes[a];
        let b = &mesh.nodes[b];
        let on_surface = [(a.x, b.x), (a.y, b.y), (a.z, b.z)]
            .iter()
            .take(mesh.dimension)
            .any(|&(a, b)| (a == 0.0 && b == 0.0) || (a == 1.0 && b == 1.0));
        assert_eq!(
            prepared.pec.contains(&edge),
            on_surface,
            "incorrect PEC edge {edge}"
        );
    }
    let edge = HcurlSpace::new();
    let curl = assemble_bilinear_checked(
        &MaxwellForm::from_coefficients(1.0, C64::default()),
        &edge,
        &edge,
        mesh,
        4,
    )
    .unwrap()
    .to_csr();
    let mass = assemble_bilinear_checked(
        &MaxwellForm::from_coefficients(0.0, C64::new(config().auxiliary_mass, 0.0)),
        &edge,
        &edge,
        mesh,
        4,
    )
    .unwrap()
    .to_csr();
    let phi: Vec<_> = (0..prepared.scalar.num_rows)
        .map(|j| C64::new((j as f64).sin(), (j as f64).cos()))
        .collect();
    let gradient = prepared.gradient.try_matvec(&phi).unwrap();
    let curl_gradient = curl.try_matvec(&gradient).unwrap();
    assert!(curl_gradient.iter().all(|v| v.norm() < 2e-12));
    let projected = prepared
        .gradient
        .try_matvec_hermitian(&mass.try_matvec(&gradient).unwrap())
        .unwrap();
    let expected = prepared.scalar.try_matvec(&phi).unwrap();
    for (a, b) in projected.iter().zip(expected) {
        assert!((*a - b).norm() < 2e-12);
    }
    for &edge in &prepared.pec {
        assert_eq!(
            prepared.gradient.row_ptrs[edge],
            prepared.gradient.row_ptrs[edge + 1]
        );
        assert_eq!(
            prepared.interpolation.row_ptrs[edge],
            prepared.interpolation.row_ptrs[edge + 1]
        );
        assert_eq!(prepared.diagonal[edge], C64::new(1.0, 0.0));
    }
    for beta in [0.0, -1.0, f64::NAN] {
        assert!(
            prepare(
                mesh,
                MaxwellCudaConfig {
                    auxiliary_mass: beta,
                    ..config()
                }
            )
            .is_err()
        );
    }
    assert!(prepare(&unit_square_triangles(1), config()).is_err());
}

#[test]
fn native_maxwell_auxiliary_retains_physical_solves() {
    run_physical(2, FemAction::Forward);
}

#[test]
fn native_maxwell_tetrahedron_actions_retain_physical_solves() {
    for action in [
        FemAction::Forward,
        FemAction::Transpose,
        FemAction::Hermitian,
    ] {
        run_physical(3, action);
    }
}

fn run_physical(dim: usize, action: FemAction) {
    let backend = match CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            assert!(
                std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                "CUDA required: {error}"
            );
            eprintln!("SKIP Maxwell CUDA: {error}");
            return;
        }
    };
    let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 29)).unwrap();
    let backend = pool.backend();
    let mut previous_error = f64::INFINITY;
    let sizes = if dim == 2 { [4, 8, 12] } else { [2, 3, 4] };
    for subdivisions in sizes {
        let mut mesh = if dim == 2 {
            unit_square_triangles(subdivisions)
        } else {
            unit_cube_tetrahedra(subdivisions)
        };
        if dim == 3 && subdivisions == 3 {
            reverse_node_numbering(&mut mesh);
        }
        let prepared = prepare(&mesh, config()).unwrap();
        let space = HcurlSpace::new();
        let a = assemble_bilinear_checked(
            &MaxwellForm::from_coefficients(
                config().curl_coefficient,
                if action == FemAction::Hermitian {
                    config().mass_coefficient.conj()
                } else {
                    config().mass_coefficient
                },
            ),
            &space,
            &space,
            &mesh,
            4,
        )
        .unwrap();
        let rhs = assemble_linear_checked(
            &MaxwellCurrentSourceForm::new(1.0, |x, y, z| current_for(dim, action, x, y, z)),
            &space,
            &mesh,
            6,
        )
        .unwrap();
        let mut reference = FemSystem::new(a, rhs);
        try_apply_homogeneous_dirichlet_complex_dofs(&mut reference, &prepared.pec).unwrap();
        let csr = reference.matrix.to_csr();
        let cpu = sonium_solvers::lu_solve(&csr.to_dense(), &reference.rhs).unwrap();
        let mut session = try_prepare_cuda_action(&pool, &mesh, config(), action, |x, y, z| {
            current_for(dim, action, x, y, z)
        })
        .unwrap();
        let n = reference.rhs.len();
        let mut device_rhs = backend.raw_alloc::<C64>(n).unwrap();
        let controls = BackendKrylovConfig {
            max_iterations: 2000,
            restart: 60,
            tolerance: 1e-10,
            require_native: true,
            exact_residual_check_interval: 0,
        };
        let baseline = if action == FemAction::Forward {
            let mut diagonal_baseline = CudaFemSolveSession::try_assemble(
                &pool,
                FemWorkload {
                    element: if dim == 2 {
                        FemElement::Tri3
                    } else {
                        FemElement::Tetrahedron
                    },
                    ..FemWorkload::native_cuda_maxwell(FemScalar::Complex64, action)
                },
                FemDeviceMethod::Fgmres,
                || {
                    let mut system =
                        FemSystem::new(reference.matrix.clone(), reference.rhs.clone());
                    system.dirichlet_dofs.clone_from(&reference.dirichlet_dofs);
                    Ok(system)
                },
                |_| PointJacobiBackend::try_from_diagonal(backend, &prepared.diagonal),
            )
            .unwrap();
            let baseline = diagonal_baseline
                .solve(DeviceInitialGuess::Zero, &controls)
                .unwrap();
            if dim == 2 {
                assert!(baseline.converged, "diagonal baseline {baseline:?}");
            }
            Some(baseline)
        } else {
            None
        };
        for (pass, amplitude) in [1.0, -0.5, 1.5].into_iter().enumerate() {
            let rhs: Vec<_> = reference.rhs.iter().map(|v| amplitude * v).collect();
            backend.try_copy_h2d(&rhs, &mut device_rhs).unwrap();
            session.try_update_rhs(&device_rhs).unwrap();
            let before = backend.begin_runtime_stats();
            let status = session
                .solve(
                    if pass == 0 {
                        DeviceInitialGuess::Zero
                    } else {
                        DeviceInitialGuess::PreviousSolution
                    },
                    &controls,
                )
                .unwrap();
            let stats = backend.runtime_stats_since(before);
            assert!(status.converged, "mesh={subdivisions} {status:?}");
            if pass > 0 {
                assert_eq!(stats.allocation_operations, 0, "{stats:?}");
            }
            let mut actual = vec![C64::default(); n];
            backend
                .try_copy_d2h(session.solution().unwrap(), &mut actual)
                .unwrap();
            let relative = actual
                .iter()
                .zip(&cpu)
                .map(|(a, b)| (*a / amplitude - b).norm_sqr())
                .sum::<f64>()
                / cpu.iter().map(|v| v.norm_sqr()).sum::<f64>();
            assert!(relative.sqrt() < 2e-7, "CPU mismatch {}", relative.sqrt());
            let ax = csr.try_matvec(&actual).unwrap();
            let residual = (ax
                .iter()
                .zip(&rhs)
                .map(|(a, b)| (*a - b).norm_sqr())
                .sum::<f64>()
                / rhs.iter().map(|v| v.norm_sqr()).sum::<f64>())
            .sqrt();
            assert!(residual < 2e-10, "{residual}");
            if pass == 0 {
                let error = field_error(&mesh, &actual);
                assert!(
                    error < (if dim == 2 { 0.4 } else { 0.8 }) && error < previous_error * 0.85,
                    "L2={error} previous={previous_error}"
                );
                previous_error = error;
                eprintln!(
                    "MAXWELL_CUDA dim={dim} action={action:?} mesh={subdivisions} edges={n} iterations={} jacobi_iterations={:?} residual={residual:.3e} field_l2={error:.6e}",
                    status.iterations,
                    baseline.as_ref().map(|b| (b.iterations, b.converged))
                );
                if subdivisions == sizes[2]
                    && let Some(baseline) = &baseline
                {
                    assert!(
                        status.iterations < baseline.iterations,
                        "auxiliary correction did not reduce iteration count"
                    );
                }
            }
        }
        // Check the preconditioner separately: no bulk transfers or allocations,
        // and failure cannot overwrite a caller's last successful result.
        let preconditioner = MaxwellAuxiliaryCuda::upload(backend, prepared).unwrap();
        let mut out = backend.raw_alloc::<C64>(n).unwrap();
        preconditioner
            .try_apply(backend, &device_rhs, &mut out)
            .unwrap();
        let before = backend.begin_runtime_stats();
        preconditioner
            .try_apply(backend, &device_rhs, &mut out)
            .unwrap();
        let stats = backend.runtime_stats_since(before);
        assert_eq!(stats.allocation_operations, 0);
        assert_eq!(stats.transfer.h2d_bytes, 0);
        assert_eq!(stats.transfer.d2h_bytes, 0);
        let sentinel = vec![C64::new(17.0, -3.0); n];
        backend.try_copy_h2d(&sentinel, &mut out).unwrap();
        backend.inject_failure(CudaFailureInjectionPoint::Launch);
        assert!(
            preconditioner
                .try_apply(backend, &device_rhs, &mut out)
                .is_err()
        );
        backend.clear_failure();
        backend.recover_after_error().unwrap();
        let mut actual = vec![C64::default(); n];
        backend.try_copy_d2h(&out, &mut actual).unwrap();
        assert_eq!(actual, sentinel);
        preconditioner
            .try_apply(backend, &device_rhs, &mut out)
            .unwrap();
    }
}
