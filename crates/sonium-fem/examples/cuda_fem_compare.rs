//! Compare retained CUDA FEM actions and solves against cuSPARSE.
//!
//! Run with `--features cuda --example cuda_fem_compare -- <subdivisions> ...`.
//! Outputs JSON lines. Requires a GPU; failures never become skipped benchmarks.
//! Uses affine unit cubes, complex-double Helmholtz k=5 and natural Neumann BCs.
//! Both operators use the same independently assembled shifted-AMG preconditioner.

// Rust guideline compliant 2026-02-21
use anyhow::{Result, ensure};
use num_complex::Complex64;
use serde_json::json;
use sonium_compute::cuda::{
    CudaBackend,
    fem_hex::CudaHexPartialC64,
    fem_tet::{CudaTetP1C64, CudaTetP2C64},
};
use sonium_compute::{
    Backend, DeviceOperator, HostCsrRef, MemoryPool, MemoryPoolConfig, PersistentDeviceCsr,
};
use sonium_fem::assembly::assemble_bilinear_checked;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::{unit_cube_hexahedra, unit_cube_tetrahedra};
use sonium_fem::physics::acoustics::pressure::{
    HelmholtzForm,
    device_hex::{prepare_hex_helmholtz, prepare_hex_lor_shifted_laplacian},
    device_tet::{prepare_tet_p1_helmholtz, prepare_tet_p2_helmholtz},
};
use sonium_fem::spaces::H1Space;
use sonium_solvers::iterative::{BackendKrylovConfig, DeviceGmresSession, DeviceInitialGuess};
use sonium_solvers::{
    AmgConfig, AmgCycle, AmgInterpolation, AmgSmoother, BackendAmgPreconditioner, CsrMatrix,
};
use std::time::Instant;

#[path = "support/cuda_profile.rs"]
mod profile;

type Operator = dyn DeviceOperator<CudaBackend, Complex64>;

fn apply_host(a: &CsrMatrix<Complex64>, x: &[Complex64]) -> Vec<Complex64> {
    (0..a.num_rows)
        .map(|row| {
            (a.row_ptrs[row]..a.row_ptrs[row + 1])
                .map(|j| a.values[j] * x[a.col_indices[j]])
                .sum()
        })
        .collect()
}

fn relative_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    let numerator: f64 = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (*a - *b).norm_sqr())
        .sum();
    let denominator: f64 = expected.iter().map(|b| b.norm_sqr()).sum();
    (numerator / denominator).sqrt()
}

fn summary(samples: &[f64]) -> serde_json::Value {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    json!({"min_ms": sorted[0], "median_ms": sorted[sorted.len()/2],
        "max_ms": sorted[sorted.len()-1], "samples_ms": samples})
}

fn run(
    pool: &MemoryPool<CudaBackend>,
    subdivisions: usize,
    family: &str,
    degree: PolynomialDegree,
    amg: &AmgConfig,
    use_lor: bool,
    profile_solve: bool,
    small_jacobi: bool,
) -> Result<bool> {
    eprintln!("benchmark {family} {degree:?}, subdivisions={subdivisions}, AMG={amg:?}");
    let backend = pool.backend();
    let mesh = if family == "tet" {
        unit_cube_tetrahedra(subdivisions)
    } else {
        unit_cube_hexahedra(subdivisions)
    };
    let space = H1Space::new(degree);
    let form = HelmholtzForm::from_real(5.0);
    // P2 tetrahedral quadrature needs degree-four exactness; hex takes points/axis.
    let q = if family == "tet" {
        4
    } else {
        degree.degree() + 1
    };
    let start = Instant::now();
    let matrix = assemble_bilinear_checked(&form, &space, &space, &mesh, q)?.to_csr();
    let host_assembly_ms = start.elapsed().as_secs_f64() * 1e3;
    let n = matrix.num_rows;
    let start = Instant::now();
    let csr = PersistentDeviceCsr::try_from_host_csr_ref(
        backend,
        HostCsrRef::try_new(n, n, &matrix.row_ptrs, &matrix.col_indices, &matrix.values)?,
    )?;
    backend.try_synchronize()?;
    let csr_setup_ms = start.elapsed().as_secs_f64() * 1e3;
    let start = Instant::now();
    let specialized: Box<Operator> = match (family, degree) {
        ("tet", PolynomialDegree::P1) => Box::new(CudaTetP1C64::try_new(
            pool.clone(),
            &prepare_tet_p1_helmholtz(&mesh, &space, &form)?,
        )?),
        ("tet", PolynomialDegree::P2) => Box::new(CudaTetP2C64::try_new(
            pool.clone(),
            &prepare_tet_p2_helmholtz(&mesh, &space, &form)?,
        )?),
        _ => Box::new(CudaHexPartialC64::try_new(
            pool.clone(),
            &prepare_hex_helmholtz(&mesh, &space, &form, q)?,
            1,
        )?),
    };
    backend.try_synchronize()?;
    let specialized_setup_ms = start.elapsed().as_secs_f64() * 1e3;
    let exact: Vec<_> = (0..n)
        .map(|i| Complex64::new((0.13 * i as f64).sin(), (0.07 * i as f64).cos()))
        .collect();
    let b = apply_host(&matrix, &exact);
    let mut x = pool.try_alloc::<Complex64>(n)?;
    let mut y = pool.try_alloc::<Complex64>(n)?;
    let mut rhs = pool.try_alloc::<Complex64>(n)?;
    backend.try_copy_h2d(&exact, x.raw_mut())?;
    backend.try_copy_h2d(&b, rhs.raw_mut())?;
    let operators: [(&str, &Operator); 2] =
        [("cusparse", &csr), ("specialized", specialized.as_ref())];
    let mut action_results = Vec::new();
    for (name, op) in operators {
        op.try_apply(backend, x.raw(), y.raw_mut())?;
        backend.try_synchronize()?;
        let mut actual = vec![Complex64::default(); n];
        backend.try_copy_d2h(y.raw(), &mut actual)?;
        let error = relative_error(&actual, &b);
        ensure!(
            error < 1e-11,
            "{family} {degree:?} {name} action error {error}"
        );
        let before = backend.runtime_stats();
        let mut samples = Vec::new();
        // Batch 100 launches, synchronize at each batch boundary, report per-action wall time.
        for _ in 0..7 {
            let start = Instant::now();
            for _ in 0..100 {
                op.try_apply(backend, x.raw(), y.raw_mut())?;
            }
            backend.try_synchronize()?;
            samples.push(start.elapsed().as_secs_f64() * 10.0);
        }
        let after = backend.runtime_stats();
        ensure!(
            after.allocation_operations == before.allocation_operations,
            "warm action allocated"
        );
        ensure!(
            after.transfer.h2d_bytes == before.transfer.h2d_bytes
                && after.transfer.d2h_bytes == before.transfer.d2h_bytes,
            "warm action transferred"
        );
        action_results.push(
            json!({"operator":name,"timing":summary(&samples),"relative_error":error,
            "kernel_launches":after.kernel_launches-before.kernel_launches,
            "library_operations":after.library_operations-before.library_operations}),
        );
    }
    let start = Instant::now();
    // Keep the auxiliary operator coercive; convergence still needs validation.
    // This shifts only the preconditioner; the solved operator remains K - 25 M.
    let shift = Complex64::new(25.0, 25.0);
    let shifted = if use_lor && family == "hex" {
        prepare_hex_lor_shifted_laplacian(&mesh, &space, shift, &[])?
    } else {
        assemble_bilinear_checked(
            &HelmholtzForm::new((-shift).sqrt()),
            &space,
            &space,
            &mesh,
            q,
        )?
        .to_csr()
    };
    let mut preconditioner =
        BackendAmgPreconditioner::<CudaBackend, Complex64>::try_from_csr_with_cuda_coarse_lu(
            backend,
            &shifted,
            amg.clone(),
        )?;
    preconditioner.set_small_jacobi_enabled(small_jacobi);
    backend.try_synchronize()?;
    let preconditioner_setup_ms = start.elapsed().as_secs_f64() * 1e3;
    // Verify the selected preconditioner independently of Krylov control traffic.
    sonium_compute::DevicePreconditioner::try_apply(
        &preconditioner,
        backend,
        rhs.raw(),
        y.raw_mut(),
    )?;
    backend.try_synchronize()?;
    let preconditioner_before = backend.runtime_stats();
    let start = Instant::now();
    sonium_compute::DevicePreconditioner::try_apply(
        &preconditioner,
        backend,
        rhs.raw(),
        y.raw_mut(),
    )?;
    backend.try_synchronize()?;
    let preconditioner_apply_ms = start.elapsed().as_secs_f64() * 1e3;
    let preconditioner_after = backend.runtime_stats();
    ensure!(
        preconditioner_after.allocation_operations == preconditioner_before.allocation_operations,
        "warm preconditioner allocated"
    );
    ensure!(
        preconditioner_after.transfer.h2d_bytes == preconditioner_before.transfer.h2d_bytes
            && preconditioner_after.transfer.d2h_bytes == preconditioner_before.transfer.d2h_bytes,
        "preconditioner staged data on the host"
    );
    ensure!(
        sonium_compute::DevicePreconditioner::diagnostics(&preconditioner).fallback_count == 0,
        "preconditioner substituted a fallback smoother"
    );
    let config = BackendKrylovConfig {
        max_iterations: 600,
        restart: 60,
        tolerance: 1e-9,
        require_native: true,
        exact_residual_check_interval: 0,
    };
    let mut solve_results = Vec::new();
    let mut converged = true;
    for (name, op) in operators {
        let mut session = DeviceGmresSession::try_new_flexible(pool, n)?;
        let warm = session.solve(
            op,
            &preconditioner,
            rhs.raw(),
            DeviceInitialGuess::Zero,
            &config,
            None,
        )?;
        eprintln!(
            "{name} warm solve: converged={}, iterations={}, residual={}",
            warm.converged, warm.iterations, warm.true_residual
        );
        if !warm.converged {
            converged = false;
            solve_results.push(json!({"operator":name,"converged":false,
                "iterations":warm.iterations,"solver_true_residual":warm.true_residual,
                "stopping_reason":format!("{:?}",warm.stopping_reason)}));
            continue;
        }
        backend.try_synchronize()?;
        let before = backend.runtime_stats();
        let mut samples = Vec::new();
        let mut iterations = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            let status = session.solve(
                op,
                &preconditioner,
                rhs.raw(),
                DeviceInitialGuess::Zero,
                &config,
                None,
            )?;
            backend.try_synchronize()?;
            samples.push(start.elapsed().as_secs_f64() * 1e3);
            ensure!(status.converged, "timed solve failed: {status:?}");
            iterations.push(status.iterations);
        }
        let after = backend.runtime_stats();
        let mut actual = vec![Complex64::default(); n];
        backend.try_copy_d2h(
            session.solution().expect("certified converged solve"),
            &mut actual,
        )?;
        let residual = relative_error(&apply_host(&matrix, &actual), &b);
        ensure!(residual < 2e-9, "independent residual failed: {residual}");
        let profile = if profile_solve {
            preconditioner.set_phase_profiling(true);
            let measured_operator = profile::ProfileOperator::new(op);
            let measured_preconditioner = profile::ProfilePreconditioner::new(&preconditioner);
            backend.try_synchronize()?;
            let before = backend.runtime_stats();
            let start = Instant::now();
            let status = session.solve(
                &measured_operator,
                &measured_preconditioner,
                rhs.raw(),
                DeviceInitialGuess::Zero,
                &config,
                None,
            )?;
            backend.try_synchronize()?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
            let after = backend.runtime_stats();
            ensure!(
                status.converged && status.iterations == warm.iterations,
                "instrumentation changed solve convergence: {status:?}"
            );
            ensure!(
                after.allocation_operations == before.allocation_operations,
                "instrumented warm solve allocated"
            );
            backend.try_copy_d2h(
                session.solution().expect("certified profiled solve"),
                &mut actual,
            )?;
            let checked_residual = relative_error(&apply_host(&matrix, &actual), &b);
            ensure!(
                checked_residual < 2e-9,
                "profiled independent residual {checked_residual}"
            );
            let mut report = measured_preconditioner.report(&measured_operator, elapsed_ms);
            report["amg_levels"] = json!(preconditioner.phase_profile().iter().map(|p| json!({
                "level":p.level,"phase":format!("{:?}",p.phase),"calls":p.calls,"synchronized_wall_ms":p.elapsed_ms
            })).collect::<Vec<_>>());
            report["iterations"] = json!(status.iterations);
            report["true_relative_residual"] = json!(checked_residual);
            report["warm_allocations"] =
                json!(after.allocation_operations - before.allocation_operations);
            report["h2d_bytes"] = json!(after.transfer.h2d_bytes - before.transfer.h2d_bytes);
            report["d2h_bytes"] = json!(after.transfer.d2h_bytes - before.transfer.d2h_bytes);
            preconditioner.set_phase_profiling(false);
            Some(report)
        } else {
            None
        };
        solve_results.push(
            json!({"operator":name,"converged":true,"timing":summary(&samples),"iterations":iterations,
            "synchronized_profile":profile,
            "true_relative_residual":residual,"solution_error":relative_error(&actual,&exact),
            "kernel_launches":after.kernel_launches-before.kernel_launches,
            "library_operations":after.library_operations-before.library_operations,
            "warm_allocations":after.allocation_operations-before.allocation_operations,
            "h2d_bytes":after.transfer.h2d_bytes-before.transfer.h2d_bytes,
            "d2h_bytes":after.transfer.d2h_bytes-before.transfer.d2h_bytes}),
        );
    }
    let hierarchy = preconditioner.diagnostics();
    println!(
        "{}",
        json!({"family":family,"degree":degree.degree(),"subdivisions":subdivisions,
        "dofs":n,"elements":mesh.elements.len(),"nnz":matrix.values.len(),"quadrature":q,
        "host_assembly_ms":host_assembly_ms,"csr_setup_ms":csr_setup_ms,
        "specialized_setup_ms":specialized_setup_ms,"preconditioner_setup_ms":preconditioner_setup_ms,
        "amg_config":format!("{amg:?}"),"small_jacobi":small_jacobi,
        "amg_level_dofs":hierarchy.level_dofs,"amg_level_nnz":hierarchy.level_nnz,
        "amg_operator_complexity":hierarchy.operator_complexity,
        "preconditioner_apply_ms":preconditioner_apply_ms,
        "preconditioner_kernel_launches":preconditioner_after.kernel_launches-preconditioner_before.kernel_launches,
        "preconditioner_library_operations":preconditioner_after.library_operations-preconditioner_before.library_operations,
        "preconditioner_warm_allocations":preconditioner_after.allocation_operations-preconditioner_before.allocation_operations,
        "preconditioner_warm_h2d_bytes":preconditioner_after.transfer.h2d_bytes-preconditioner_before.transfer.h2d_bytes,
        "preconditioner_warm_d2h_bytes":preconditioner_after.transfer.d2h_bytes-preconditioner_before.transfer.d2h_bytes,
        "actions":action_results,"solves":solve_results})
    );
    Ok(converged)
}

fn main() -> Result<()> {
    let sizes: Vec<usize> = std::env::args()
        .skip(1)
        .map(|s| s.parse())
        .collect::<std::result::Result<_, _>>()?;
    let sizes = if sizes.is_empty() {
        vec![2, 4, 8]
    } else {
        sizes
    };
    ensure!(
        sizes.iter().all(|&n| n > 0 && n <= 64),
        "subdivisions must be 1..=64"
    );
    let start = Instant::now();
    let pool = MemoryPool::new(
        CudaBackend::new(0)?,
        MemoryPoolConfig::with_cap(8usize << 30),
    )?;
    println!(
        "{}",
        json!({"context_setup_ms":start.elapsed().as_secs_f64()*1e3,
        "preconditioner_mass_shift":[25.0,25.0],"k":5.0,"scalar":"complex64","boundary":"natural_neumann","tolerance":1e-9,
        "action_repetitions":100,"action_samples":7,"solve_samples":5,"solve_initial_guess":"zero"})
    );
    let use_lor = match std::env::var("SONIUM_FEM_BENCH_PRECONDITIONER").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("assembled") => false,
        Ok("lor") => true,
        _ => anyhow::bail!("SONIUM_FEM_BENCH_PRECONDITIONER must be assembled or lor"),
    };
    let smoother = match std::env::var("SONIUM_FEM_BENCH_SMOOTHER").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("jacobi") => AmgSmoother::Jacobi,
        Ok("l1-jacobi") => AmgSmoother::L1Jacobi,
        Ok("multicolor-gs") => AmgSmoother::MulticolorGaussSeidel,
        _ => anyhow::bail!("SONIUM_FEM_BENCH_SMOOTHER must be jacobi, l1-jacobi or multicolor-gs"),
    };
    let amg_profile = std::env::var("SONIUM_FEM_BENCH_AMG").unwrap_or_else(|_| "baseline".into());
    let mut amg = AmgConfig {
        coarse_size: 16,
        smoother,
        jacobi_weight: 0.6,
        ..Default::default()
    };
    match amg_profile.as_str() {
        "baseline" => {}
        "extended" => amg.interpolation = AmgInterpolation::ExtendedI,
        "two-sweeps" => {
            amg.num_pre_smooth = 2;
            amg.num_post_smooth = 2;
        }
        "w-cycle" => amg.cycle = AmgCycle::WCycle,
        "aggregation" => {
            amg = AmgConfig {
                coarse_size: 16,
                smoother,
                jacobi_weight: 0.6,
                ..AmgConfig::for_aggregation()
            }
        }
        _ => anyhow::bail!(
            "SONIUM_FEM_BENCH_AMG must be baseline, extended, two-sweeps, w-cycle or aggregation"
        ),
    }
    let family_filter = std::env::var("SONIUM_FEM_BENCH_FAMILY").unwrap_or_else(|_| "all".into());
    ensure!(
        matches!(family_filter.as_str(), "all" | "tet" | "hex"),
        "SONIUM_FEM_BENCH_FAMILY must be all, tet or hex"
    );
    let profile_solve = match std::env::var("SONIUM_FEM_BENCH_PROFILE").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("off") => false,
        Ok("sync") => true,
        _ => anyhow::bail!("SONIUM_FEM_BENCH_PROFILE must be off or sync"),
    };
    println!(
        "{}",
        json!({"amg_profile":amg_profile,"family_filter":family_filter,"amg_config":format!("{amg:?}"),"hex_low_order_refined":use_lor,"amg_smoother":format!("{smoother:?}"),"restart":60,"max_iterations":600,"synchronized_profile":profile_solve})
    );
    let small_jacobi = match std::env::var("SONIUM_FEM_BENCH_SMALL_JACOBI").as_deref() {
        Ok("on") => true,
        Ok("off") | Err(_) => false,
        _ => anyhow::bail!("SONIUM_FEM_BENCH_SMALL_JACOBI must be on or off"),
    };
    let mut converged = true;
    for size in sizes {
        for (family, degree) in [
            ("tet", PolynomialDegree::P1),
            ("tet", PolynomialDegree::P2),
            ("hex", PolynomialDegree::P2),
            ("hex", PolynomialDegree::P3),
        ] {
            if family_filter != "all" && family_filter != family {
                continue;
            }
            converged &= run(
                &pool,
                size,
                family,
                degree,
                &amg,
                use_lor,
                profile_solve,
                small_jacobi,
            )?;
        }
    }
    ensure!(
        converged,
        "one or more benchmark solves did not converge; see JSON rows"
    );
    Ok(())
}
