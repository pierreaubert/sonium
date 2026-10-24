//! Host preparation of retained acoustic boundary contributions.
//!
//! Upload the matrices once and compose the action with a volume operator using
//! `sonium_compute::operator_sum::DeviceSum`. Boundary quadrature and callbacks
//! are host setup. Retained quadrature maps additionally support GPU numeric
//! matrix/load assembly from resident coefficient samples.

// Rust guideline compliant 2026-02-21
use super::{HelmholtzForm, device_tet::prepare_tet_p1_helmholtz};
use crate::assembly::TripletMatrix;
use crate::boundary::{FemSystem, try_apply_robin_complex};
use crate::mesh::Mesh;
use crate::spaces::H1Space;
use num_complex::Complex64;
use sonium_compute::{BackendError, BackendResult};
use sonium_solvers::CsrMatrix;

mod hex;
pub use hex::prepare_hex_robin;
mod tet;
mod tet_quadrature;
#[doc(inline)]
pub use tet::prepare_tet_p2_robin;
#[doc(inline)]
pub use tet_quadrature::prepare_tet_robin_quadratures;
mod quadrature;
#[doc(inline)]
pub use quadrature::{
    BoundaryMatrixMap, HexRobinQuadrature, RobinQuadrature, prepare_hex_robin_quadrature,
    prepare_hex_robin_quadratures,
};

/// Separate Robin action, source and prescribed-pressure coupling matrices.
///
/// For boundary matrix B, load f, constrained projector Q and P = I-Q:
/// - `matrix` is P B P, with zero constrained rows (no identity contribution).
/// - `dirichlet_coupling` is P B Q.
/// - `rhs` is P f.
///
/// Add `rhs - dirichlet_coupling*g` to the volume's lifted RHS. Compose `matrix`
/// with the constrained volume operator, which supplies the identity rows.
#[derive(Debug)]
pub struct PreparedRobinC64 {
    /// Robin action with prescribed rows and columns removed.
    pub matrix: CsrMatrix<Complex64>,
    /// Coupling from prescribed values into free equations.
    pub dirichlet_coupling: CsrMatrix<Complex64>,
    /// Natural boundary load with zero prescribed entries.
    pub rhs: Vec<Complex64>,
}

fn invalid(reason: impl Into<String>) -> BackendError {
    BackendError::InvalidArgument {
        operation: "prepare_robin",
        reason: reason.into(),
    }
}

/// Prepare a P1 tetrahedral Robin/impedance boundary for retained device actions.
///
/// Uses the existing checked FEM quadrature for ∂u/∂n + alpha u = g. Coefficients
/// may vary spatially; changing them requires rebuilding this host preparation.
/// `dirichlet` must match the prescribed set used by the volume operator.
///
/// # Errors
/// Rejects invalid volume geometry/space, connectivity, constrained DOFs,
/// unsupported boundary topology and non-finite callback values.
pub fn prepare_tet_p1_robin(
    mesh: &Mesh,
    space: &H1Space,
    marker: i32,
    alpha: &dyn Fn(f64, f64, f64) -> Complex64,
    source: &dyn Fn(f64, f64, f64) -> Complex64,
    dirichlet: &[usize],
) -> BackendResult<PreparedRobinC64> {
    mesh.validate_connectivity()
        .map_err(|e| invalid(format!("invalid mesh: {e:?}")))?;
    let volume = prepare_tet_p1_helmholtz(mesh, space, &HelmholtzForm::from_real(0.0))?;
    let n = volume.dimension();
    if dirichlet.iter().any(|&i| i >= n) {
        return Err(invalid("Dirichlet DOF out of bounds"));
    }
    let mut system = FemSystem::new(TripletMatrix::new(n, n), vec![Complex64::default(); n]);
    try_apply_robin_complex(&mut system, mesh, marker, alpha, source)
        .map_err(|e| invalid(e.to_string()))?;
    constrain_boundary(system, dirichlet)
}

fn constrain_boundary(
    mut system: FemSystem<Complex64>,
    dirichlet: &[usize],
) -> BackendResult<PreparedRobinC64> {
    let n = system.rhs.len();
    if dirichlet.iter().any(|&i| i >= n) {
        return Err(invalid("Dirichlet DOF out of bounds"));
    }
    let mut mask = vec![false; n];
    for &i in dirichlet {
        mask[i] = true;
    }
    let boundary = system.matrix.to_csr();
    let mut action = Vec::new();
    let mut coupling = Vec::new();
    for (row, load) in system.rhs.iter_mut().enumerate() {
        if mask[row] {
            *load = Complex64::default();
            continue;
        }
        for j in boundary.row_ptrs[row]..boundary.row_ptrs[row + 1] {
            let col = boundary.col_indices[j];
            let entry = (row, col, boundary.values[j]);
            if mask[col] {
                coupling.push(entry);
            } else {
                action.push(entry);
            }
        }
    }
    Ok(PreparedRobinC64 {
        matrix: CsrMatrix::try_from_triplets(n, n, action).map_err(|e| invalid(e.to_string()))?,
        dirichlet_coupling: CsrMatrix::try_from_triplets(n, n, coupling)
            .map_err(|e| invalid(e.to_string()))?,
        rhs: system.rhs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembly::assemble_bilinear_checked;
    use crate::basis::PolynomialDegree;
    use crate::boundary::try_apply_dirichlet_values_complex;
    use crate::mesh::unit_cube_tetrahedra;
    use crate::spaces::FunctionSpace;
    use sonium_compute::operator_sum::DeviceSum;
    use sonium_compute::{
        AdjointDeviceCsr, Backend, CpuBackend, DeviceOperator, HostCsrRef, MemoryPool,
        MemoryPoolConfig,
    };

    fn marked_mesh() -> Mesh {
        let mut mesh = unit_cube_tetrahedra(2);
        for boundary in &mut mesh.boundaries {
            boundary.marker = if boundary.nodes.iter().all(|&i| mesh.nodes[i].x == 0.0) {
                1
            } else {
                2
            };
        }
        mesh
    }

    #[test]
    fn robin_sum_and_lifting_match_full_fem_elimination() {
        let mesh = marked_mesh();
        let space = H1Space::new(PolynomialDegree::P1);
        let marker = 1;
        let dofs = space.boundary_dofs(&mesh, marker);
        // Use Robin on a neighboring marked face, including its intersections
        // with the prescribed face, to exercise the coupling term P B Q.
        let robin_marker = mesh
            .boundaries
            .iter()
            .find(|b| b.marker != marker)
            .unwrap()
            .marker;
        let alpha = |x: f64, y: f64, z: f64| Complex64::new(0.7 + x, 0.2 + y + z);
        let source = |x: f64, y: f64, z: f64| Complex64::new(0.3 + y, x - z);
        let form = HelmholtzForm::new(Complex64::new(3.0, -0.2));
        let volume = prepare_tet_p1_helmholtz(&mesh, &space, &form)
            .unwrap()
            .try_with_dirichlet(&dofs)
            .unwrap();
        let boundary =
            prepare_tet_p1_robin(&mesh, &space, robin_marker, &alpha, &source, &dofs).unwrap();
        let n = volume.dimension();
        let (pattern, values) = volume
            .assemble_csr(Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0))
            .unwrap();
        let rows: Vec<_> = pattern.row_ptrs().iter().map(|&v| v as usize).collect();
        let cols: Vec<_> = pattern.columns().iter().map(|&v| v as usize).collect();
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 24)).unwrap();
        let a = AdjointDeviceCsr::try_from_host_csr_ref(
            pool.backend(),
            HostCsrRef::try_new(n, n, &rows, &cols, &values).unwrap(),
        )
        .unwrap();
        let b = &boundary.matrix;
        let b = AdjointDeviceCsr::try_from_host_csr_ref(
            pool.backend(),
            HostCsrRef::try_new(n, n, &b.row_ptrs, &b.col_indices, &b.values).unwrap(),
        )
        .unwrap();
        let a = a.try_prebake(pool.backend()).unwrap();
        let b = b.try_prebake(pool.backend()).unwrap();
        let sum = DeviceSum::try_new(&pool, &a, &b, &[]).unwrap();
        let load = vec![Complex64::new(0.1, -0.2); n];
        let mut device_x = pool.try_alloc(n).unwrap();
        let mut device_y = pool.try_alloc(n).unwrap();
        for shift in [0.0, 0.6] {
            let g = |x: f64, y: f64, z: f64| Complex64::new(x + y + shift, z - 0.2);
            let prescribed: Vec<_> = mesh.nodes.iter().map(|p| g(p.x, p.y, p.z)).collect();
            let matrix = assemble_bilinear_checked(&form, &space, &space, &mesh, 4).unwrap();
            let mut full = FemSystem::new(matrix, load.clone());
            try_apply_robin_complex(&mut full, &mesh, robin_marker, &alpha, &source).unwrap();
            try_apply_dirichlet_values_complex(&mut full, &mesh, &space, marker, &g, None).unwrap();
            let mut lifted = vec![Complex64::default(); n];
            volume.lift_rhs(&load, &prescribed, &mut lifted).unwrap();
            let coupling = &boundary.dirichlet_coupling;
            for (row, v) in lifted.iter_mut().enumerate() {
                *v += boundary.rhs[row];
                for j in coupling.row_ptrs[row]..coupling.row_ptrs[row + 1] {
                    *v -= coupling.values[j] * prescribed[coupling.col_indices[j]];
                }
            }
            for (a, b) in lifted.iter().zip(&full.rhs) {
                assert!((*a - *b).norm() < 1e-11);
            }
            pool.backend()
                .try_copy_h2d(&prescribed, device_x.raw_mut())
                .unwrap();
            sum.try_apply(pool.backend(), device_x.raw(), device_y.raw_mut())
                .unwrap();
            let mut actual = vec![Complex64::default(); n];
            pool.backend()
                .try_copy_d2h(device_y.raw(), &mut actual)
                .unwrap();
            let csr = full.matrix.to_csr();
            for (row, &v) in actual.iter().enumerate() {
                let expected: Complex64 = (csr.row_ptrs[row]..csr.row_ptrs[row + 1])
                    .map(|j| csr.values[j] * prescribed[csr.col_indices[j]])
                    .sum();
                assert!((v - expected).norm() < 1e-11);
            }
        }
    }
    #[cfg(feature = "cuda")]
    #[test]
    fn native_robin_volume_sum_persistent_solve() {
        use sonium_compute::Transpose;
        use sonium_compute::cuda::{CudaBackend, fem_tet::CudaTetP1C64};
        use sonium_solvers::iterative::{
            BackendKrylovConfig, DeviceGmresSession, DeviceInitialGuess,
        };
        use sonium_solvers::{AmgConfig, AmgSmoother, BackendAmgPreconditioner};
        let backend = match CudaBackend::new_default() {
            Ok(b) => b,
            Err(e) => {
                assert!(
                    std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                    "CUDA required: {e}"
                );
                eprintln!("SKIP native Robin sum solve: {e}");
                return;
            }
        };
        let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 26)).unwrap();
        let mesh = marked_mesh();
        let space = H1Space::new(PolynomialDegree::P1);
        let marker = 1;
        let dofs = space.boundary_dofs(&mesh, marker);
        let robin_marker = mesh
            .boundaries
            .iter()
            .find(|b| b.marker != marker)
            .unwrap()
            .marker;
        let boundary = prepare_tet_p1_robin(
            &mesh,
            &space,
            robin_marker,
            &|_, _, _| Complex64::new(0.7, 0.4),
            &|_, _, _| Complex64::default(),
            &dofs,
        )
        .unwrap();
        let host = prepare_tet_p1_helmholtz(&mesh, &space, &HelmholtzForm::from_real(3.0))
            .unwrap()
            .try_with_dirichlet(&dofs)
            .unwrap();
        let n = host.dimension();
        let volume = CudaTetP1C64::try_new(pool.clone(), &host).unwrap();
        let matrix = &boundary.matrix;
        let device = AdjointDeviceCsr::try_from_host_csr_ref(
            pool.backend(),
            HostCsrRef::try_new(n, n, &matrix.row_ptrs, &matrix.col_indices, &matrix.values)
                .unwrap(),
        )
        .unwrap();
        let facet = device.try_prebake(pool.backend()).unwrap();
        let sum = DeviceSum::try_new(&pool, &volume, &facet, &[]).unwrap();
        let shifted = prepare_tet_p1_helmholtz(
            &mesh,
            &space,
            &HelmholtzForm::new(Complex64::new(9.0, -9.0).sqrt()),
        )
        .unwrap()
        .try_with_dirichlet(&dofs)
        .unwrap();
        let (pattern, values) = shifted
            .assemble_csr(Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0))
            .unwrap();
        let mut entries = Vec::new();
        for row in 0..n {
            for j in pattern.row_ptrs()[row] as usize..pattern.row_ptrs()[row + 1] as usize {
                entries.push((row, pattern.columns()[j] as usize, values[j]));
            }
            for j in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
                entries.push((row, matrix.col_indices[j], matrix.values[j]));
            }
        }
        let p = CsrMatrix::from_triplets(n, n, entries);
        let preconditioner =
            BackendAmgPreconditioner::<CudaBackend, Complex64>::try_from_csr_with_cuda_coarse_lu(
                pool.backend(),
                &p,
                AmgConfig {
                    coarse_size: 4,
                    smoother: AmgSmoother::Jacobi,
                    jacobi_weight: 0.6,
                    ..Default::default()
                },
            )
            .unwrap();
        let mut session = DeviceGmresSession::try_new_flexible(&pool, n).unwrap();
        let mut rhs = pool.try_alloc(n).unwrap();
        let config = BackendKrylovConfig {
            max_iterations: 200,
            restart: n,
            tolerance: 1e-11,
            require_native: true,
            exact_residual_check_interval: 0,
        };
        for source in 0..3 {
            let exact: Vec<_> = (0..n)
                .map(|i| Complex64::new((i as f64 + source as f64).sin(), 0.2))
                .collect();
            let mut b = vec![Complex64::default(); n];
            host.apply(&exact, &mut b, Transpose::NoTrans).unwrap();
            for (row, v) in b.iter_mut().enumerate() {
                for j in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
                    *v += matrix.values[j] * exact[matrix.col_indices[j]];
                }
            }
            pool.backend().try_copy_h2d(&b, rhs.raw_mut()).unwrap();
            let guess = if source == 0 {
                DeviceInitialGuess::Zero
            } else {
                DeviceInitialGuess::PreviousSolution
            };
            assert!(
                session
                    .solve(&sum, &preconditioner, rhs.raw(), guess, &config, None)
                    .unwrap()
                    .converged
            );
            let mut actual = vec![Complex64::default(); n];
            pool.backend()
                .try_copy_d2h(session.solution().unwrap(), &mut actual)
                .unwrap();
            for (a, b) in actual.iter().zip(exact) {
                assert!((*a - b).norm() < 1e-8);
            }
        }
    }
}
