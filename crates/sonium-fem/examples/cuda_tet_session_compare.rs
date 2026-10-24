//! Compare complete retained tetrahedral CSR and element solves with Robin updates.
//!
//! Run with `--features cuda --example cuda_tet_session_compare -- 4 8`.
//! JSON lines contain synchronized wall times; no CUDA failure is skipped.
//! Host manufactured loads and coefficient uploads precede the measured regions.
//! Set `SONIUM_FEM_BENCH_FUSED_JACOBI=1` to enable the fused auxiliary smoother,
//! and `SONIUM_FEM_BENCH_CSR_PREPROCESS=1` to retain cuSPARSE analysis. Both
//! options default to zero and are recorded in the JSON benchmark header.

// Rust guideline compliant 2026-02-21
use anyhow::{Result, ensure};
use num_complex::Complex64;
use serde_json::json;
use sonium_compute::{Backend, CudaBackend, MemoryPool, MemoryPoolConfig};
use sonium_fem::physics::acoustics::pressure::{
    HelmholtzForm,
    device_boundary::{prepare_tet_p1_robin, prepare_tet_p2_robin},
    device_tet_solve::{
        CudaTetP1PressureSession, CudaTetP2PressureSession, CudaTetPressureSession,
        TetPressureConfig, TetPressureOperator, TetRobinBoundary,
    },
};
use sonium_fem::{
    assembly::assemble_bilinear_checked,
    basis::PolynomialDegree,
    mesh::{Mesh, unit_cube_tetrahedra},
    spaces::{FunctionSpace, H1Space},
};
use sonium_solvers::iterative::{BackendKrylovConfig, DeviceInitialGuess};
use sonium_solvers::{AmgConfig, AmgSmoother, CsrMatrix};
use std::time::Instant;

fn apply(matrix: &CsrMatrix<Complex64>, x: &[Complex64]) -> Vec<Complex64> {
    (0..matrix.num_rows)
        .map(|r| {
            (matrix.row_ptrs[r]..matrix.row_ptrs[r + 1])
                .map(|j| matrix.values[j] * x[matrix.col_indices[j]])
                .sum()
        })
        .collect()
}
fn norm(x: &[Complex64]) -> f64 {
    x.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt()
}
fn summary(samples: &[f64]) -> serde_json::Value {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    json!({"median_ms":sorted[sorted.len()/2],"min_ms":sorted[0],"max_ms":sorted[sorted.len()-1],"samples_ms":samples})
}
fn run<const N: usize>(
    mut session: CudaTetPressureSession<'_, N>,
    pool: &MemoryPool<CudaBackend>,
    mesh: &Mesh,
    space: &H1Space,
    subdivisions: usize,
    mode: TetPressureOperator,
    setup_ms: f64,
    fused_jacobi: bool,
) -> Result<()> {
    session.set_fused_jacobi_enabled(fused_jacobi);
    let backend = pool.backend();
    let coordinates = space
        .dof_coordinates(mesh)
        .expect("validated H1 coordinates");
    let n = coordinates.len();
    let points = session
        .boundary_samples()
        .expect("prepared Robin boundary")
        .0
        .to_vec();
    let mut alpha = pool.try_alloc::<Complex64>(points.len())?;
    let mut source = pool.try_alloc::<Complex64>(points.len())?;
    let mut rhs = pool.try_alloc::<Complex64>(n)?;
    let mut prescribed = pool.try_alloc::<Complex64>(n)?;
    let controls = BackendKrylovConfig {
        max_iterations: 800,
        restart: 60,
        tolerance: 1e-9,
        require_native: true,
        exact_residual_check_interval: 0,
    };
    let constrained = space.boundary_dofs(mesh, 1);
    let mut mask = vec![false; n];
    for &d in &constrained {
        mask[d] = true;
    }
    for step in 0..3 {
        let k = Complex64::new(4.0 + step as f64 * 0.1, -step as f64 * 0.02);
        let coefficient = |x: f64, y: f64, z: f64| {
            Complex64::new(0.7 + x, 0.2 + y + z) * (1.0 + 0.1 * step as f64)
        };
        let load =
            |x: f64, y: f64, z: f64| Complex64::new(0.3 + x, y - z) * (1.0 + 0.2 * step as f64);
        let prepare = if N == 4 {
            prepare_tet_p1_robin
        } else {
            prepare_tet_p2_robin
        };
        let boundary = prepare(mesh, space, 2, &coefficient, &load, &[])?;
        let physical =
            assemble_bilinear_checked(&HelmholtzForm::new(k), space, space, mesh, 4)?.to_csr();
        let exact: Vec<_> = coordinates
            .iter()
            .map(|p| Complex64::new((p[0] + p[1] + step as f64).sin(), 0.2 + p[2] * p[1]))
            .collect();
        let ax = apply(&physical, &exact);
        let bx = apply(&boundary.matrix, &exact);
        let b: Vec<_> = ax
            .iter()
            .zip(&bx)
            .zip(&boundary.rhs)
            .map(|((a, b), f)| *a + *b - *f)
            .collect();
        backend.try_copy_h2d(&b, rhs.raw_mut())?;
        backend.try_copy_h2d(&exact, prescribed.raw_mut())?;
        backend.try_copy_h2d(
            &points
                .iter()
                .map(|p| coefficient(p[0], p[1], p[2]))
                .collect::<Vec<_>>(),
            alpha.raw_mut(),
        )?;
        backend.try_copy_h2d(
            &points
                .iter()
                .map(|p| load(p[0], p[1], p[2]))
                .collect::<Vec<_>>(),
            source.raw_mut(),
        )?;
        backend.try_synchronize()?;
        let before = backend.begin_runtime_stats();
        let start = Instant::now();
        session.try_update_wavenumber(k)?;
        session.try_update_boundary(alpha.raw(), source.raw())?;
        backend.try_synchronize()?;
        let update_ms = start.elapsed().as_secs_f64() * 1e3;
        let update = backend.runtime_stats_since(before);
        ensure!(
            update.allocation_operations == 0
                && update.transfer.h2d_bytes == 0
                && update.transfer.d2h_bytes == 0,
            "numeric update not resident: {update:?}"
        );
        let warm = session.solve(
            rhs.raw(),
            prescribed.raw(),
            DeviceInitialGuess::Zero,
            &controls,
            None,
        )?;
        ensure!(warm.converged, "warm solve failed: {warm:?}");
        backend.try_synchronize()?;
        let before = backend.begin_runtime_stats();
        let mut times = Vec::new();
        let mut iterations = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            let status = session.solve(
                rhs.raw(),
                prescribed.raw(),
                DeviceInitialGuess::Zero,
                &controls,
                None,
            )?;
            backend.try_synchronize()?;
            times.push(start.elapsed().as_secs_f64() * 1e3);
            ensure!(status.converged, "timed solve failed: {status:?}");
            iterations.push(status.iterations);
        }
        let stats = backend.runtime_stats_since(before);
        ensure!(
            stats.allocation_operations == 0,
            "warm solves allocated: {stats:?}"
        );
        let mut actual = vec![Complex64::default(); n];
        backend.try_copy_d2h(session.solution().expect("certified solution"), &mut actual)?;
        let ax = apply(&physical, &actual);
        let bx = apply(&boundary.matrix, &actual);
        let residual: Vec<_> = (0..n)
            .map(|r| {
                if mask[r] {
                    actual[r] - exact[r]
                } else {
                    ax[r] + bx[r] - b[r] - boundary.rhs[r]
                }
            })
            .collect();
        let denominator: Vec<_> = (0..n)
            .map(|r| {
                if mask[r] {
                    exact[r]
                } else {
                    b[r] + boundary.rhs[r]
                }
            })
            .collect();
        let residual = norm(&residual) / norm(&denominator);
        let error = norm(
            &actual
                .iter()
                .zip(&exact)
                .map(|(a, b)| *a - *b)
                .collect::<Vec<_>>(),
        ) / norm(&exact);
        ensure!(
            residual < 2e-9 && error < 1e-6,
            "independent residual {residual}, error {error}"
        );
        println!(
            "{}",
            json!({"fused_jacobi":session.fused_jacobi_enabled(),"subdivisions":subdivisions,"degree":space.degree.degree(),"dofs":n,"elements":mesh.elements.len(),"operator":format!("{mode:?}"),"step":step,"setup_ms":setup_ms,"numeric_update_ms":update_ms,"solve":summary(&times),"iterations":iterations,"relative_residual":residual,"relative_solution_error":error,"warm_allocations":stats.allocation_operations,"solve_h2d_bytes":stats.transfer.h2d_bytes,"solve_d2h_bytes":stats.transfer.d2h_bytes,"update_launches":update.kernel_launches,"update_library_operations":update.library_operations})
        );
    }
    Ok(())
}
fn env_flag(name: &str) -> Result<bool> {
    match std::env::var(name) {
        Ok(value) if value == "1" => Ok(true),
        Ok(value) if value == "0" => Ok(false),
        Ok(_) => anyhow::bail!("{name} must be 0 or 1"),
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(error) => Err(error.into()),
    }
}

fn main() -> Result<()> {
    let fused_jacobi = env_flag("SONIUM_FEM_BENCH_FUSED_JACOBI")?;
    let csr_preprocess = env_flag("SONIUM_FEM_BENCH_CSR_PREPROCESS")?;
    let sizes = std::env::args()
        .skip(1)
        .map(|s| s.parse::<usize>())
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let sizes = if sizes.is_empty() { vec![4, 8] } else { sizes };
    ensure!(
        sizes.iter().all(|&n| n > 0 && n <= 32),
        "subdivisions must be 1..=32"
    );
    let pool = MemoryPool::new(
        CudaBackend::new(0)?.try_with_csr_preprocessing(csr_preprocess)?,
        MemoryPoolConfig::with_cap(8usize << 30),
    )?;
    println!(
        "{}",
        json!({"benchmark":"retained_tet_session","fused_jacobi":fused_jacobi,"csr_preprocessing":csr_preprocess,"scalar":"complex64","tolerance":1e-9,"samples":5,"warmups_per_frequency":1,"initial_guess":"zero","amg_shift":[16.0,16.0],"smoother":"Jacobi","restart":60,"host_preparation_and_input_uploads_timed":false,"setup_includes":"session geometry, boundary maps, auxiliary AMG, JIT and initial numeric assembly","setup_order":"CSR before Element; context and driver caches shared","memory_note":"element mode omits physical CSR storage and assembly maps; auxiliary AMG remains assembled"})
    );
    for size in sizes {
        let mut mesh = unit_cube_tetrahedra(size);
        for f in &mut mesh.boundaries {
            f.marker = if f.nodes.iter().all(|&d| mesh.nodes[d].x == 0.0) {
                1
            } else {
                2
            };
        }
        for degree in [PolynomialDegree::P1, PolynomialDegree::P2] {
            let space = H1Space::new(degree);
            for mode in [TetPressureOperator::Csr, TetPressureOperator::Element] {
                eprintln!("retained tet {degree:?} {mode:?} subdivisions={size}");
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
                let coefficient = |x: f64, y: f64, z: f64| Complex64::new(0.7 + x, 0.2 + y + z);
                let load = |x: f64, y: f64, z: f64| Complex64::new(0.3 + x, y - z);
                let boundaries = [TetRobinBoundary {
                    marker: 2,
                    alpha: &coefficient,
                    source: &load,
                }];
                let start = Instant::now();
                if degree == PolynomialDegree::P1 {
                    let session = CudaTetP1PressureSession::try_new_with_boundaries(
                        &pool,
                        &mesh,
                        &space,
                        &HelmholtzForm::from_real(4.0),
                        &config,
                        &boundaries,
                    )?;
                    pool.backend().try_synchronize()?;
                    run(
                        session,
                        &pool,
                        &mesh,
                        &space,
                        size,
                        mode,
                        start.elapsed().as_secs_f64() * 1e3,
                        fused_jacobi,
                    )?;
                } else {
                    let session = CudaTetP2PressureSession::try_new_with_boundaries(
                        &pool,
                        &mesh,
                        &space,
                        &HelmholtzForm::from_real(4.0),
                        &config,
                        &boundaries,
                    )?;
                    pool.backend().try_synchronize()?;
                    run(
                        session,
                        &pool,
                        &mesh,
                        &space,
                        size,
                        mode,
                        start.elapsed().as_secs_f64() * 1e3,
                        fused_jacobi,
                    )?;
                }
            }
        }
    }
    Ok(())
}
