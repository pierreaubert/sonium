//! Retained CUDA linear state for host-assembled BDF momentum systems.
//!
//! This cache preserves context, CSR descriptors and Krylov vectors. Identical
//! matrices and controls reuse setup. Polynomial and Chebyshev value updates
//! retain preconditioner storage, as do Schur updates with polynomial or
//! Chebyshev pressure actions. External pressure AMG is retained when its
//! pressure matrix is unchanged; changed values refresh matching hierarchy storage.
//! Hierarchy structure changes rebuild the device preconditioner. Host AMG setup remains.
//! RHS, initial state and final state still cross the host boundary each call.

// Rust guideline compliant 2026-02-21
use super::navier_stokes::{
    BackendPreconditionerKind, BackendPressureSchurPreconditionerKind, NavierStokesConfig,
};
use super::workspace::BdfWorkspace;
use sonium_compute::preconditioner::{
    PressureSchurBackendKind, SaddlePointJacobiSchurBackend, SaddlePointJacobiSchurBackendConfig,
};
use sonium_compute::{
    Backend, Buffer, ChebyshevBackend, CudaBackend, DevicePreconditioner, MemoryPool,
    MemoryPoolConfig, PersistentDeviceCsr, PolynomialJacobiBackend,
};
use sonium_solvers::CsrMatrix;
use sonium_solvers::iterative::{
    BackendKrylovConfig, BackendKrylovWorkspace, try_fgmres_backend_preconditioned,
};
mod pressure;
use pressure::PressureAmg;

enum Preconditioner {
    Polynomial(Box<PolynomialJacobiBackend<'static, CudaBackend, f64>>),
    Chebyshev(Box<ChebyshevBackend<'static, CudaBackend, f64>>),
    Schur {
        action: Box<SaddlePointJacobiSchurBackend<'static, CudaBackend, f64>>,
        pressure: Option<PressureAmg>,
    },
}

impl Preconditioner {
    fn as_ref(&self) -> &dyn DevicePreconditioner<CudaBackend, f64> {
        match self {
            Self::Polynomial(p) => p.as_ref(),
            Self::Chebyshev(p) => p.as_ref(),
            Self::Schur { action, .. } => action.as_ref(),
        }
    }

    // Called only after the cache has checked the complete retained pattern.
    fn refresh(
        &mut self,
        backend: &CudaBackend,
        operator: &PersistentDeviceCsr<CudaBackend, f64>,
        matrix: &CsrMatrix<f64>,
        schur: Option<&CsrMatrix<f64>>,
        pressure_changed: bool,
        nv: usize,
    ) -> Result<bool, String> {
        match self {
            Self::Polynomial(p) => p
                .try_update_numeric(backend, &operator.matrix().values, &diagonal(matrix))
                .map_err(|e| e.to_string())?,
            Self::Chebyshev(p) => p
                .try_update_from_csr_host(
                    backend,
                    &operator.matrix().values,
                    &indices_i32(&matrix.row_ptrs)?,
                    &indices_i32(&matrix.col_indices)?,
                    &matrix.values,
                )
                .map_err(|e| e.to_string())?,
            Self::Schur { action, pressure } => {
                let schur = schur.ok_or("Schur refresh requires a pressure matrix")?;
                if let Some(pressure) = pressure {
                    if pressure_changed && !pressure.refresh(backend, matrix.num_rows, nv, schur)? {
                        return Ok(false);
                    }
                    action
                        .try_update_numeric_host_with_pressure(
                            backend,
                            matrix.as_host_csr_ref(),
                            schur.as_host_csr_ref(),
                            Box::new(pressure.clone()),
                        )
                        .map_err(|e| e.to_string())?;
                } else {
                    action
                        .try_update_numeric_host(
                            backend,
                            matrix.as_host_csr_ref(),
                            schur.as_host_csr_ref(),
                        )
                        .map_err(|e| e.to_string())?;
                }
            }
        }
        Ok(true)
    }
}

fn diagonal(matrix: &CsrMatrix<f64>) -> Vec<f64> {
    (0..matrix.num_rows)
        .map(|r| {
            (matrix.row_ptrs[r]..matrix.row_ptrs[r + 1])
                .filter(|&j| matrix.col_indices[j] == r)
                .map(|j| matrix.values[j])
                .sum()
        })
        .collect()
}

fn indices_i32(indices: &[usize]) -> Result<Vec<i32>, String> {
    indices
        .iter()
        .map(|&v| i32::try_from(v).map_err(|e| e.to_string()))
        .collect()
}

pub(crate) struct BdfCudaSolver {
    matrix: CsrMatrix<f64>,
    schur: Option<CsrMatrix<f64>>,
    key: [u64; 12],
    operator: PersistentDeviceCsr<CudaBackend, f64>,
    preconditioner: Preconditioner,
    iteration: BackendKrylovWorkspace<CudaBackend, f64>,
    rhs: Buffer<f64, CudaBackend>,
    guess: Buffer<f64, CudaBackend>,
    solution: Buffer<f64, CudaBackend>,
    // Full device preconditioner constructions; numeric AMG refresh still builds a host hierarchy.
    pub(crate) setup_builds: usize,
    pub(crate) matrix_uploads: usize,
    pub(crate) last_allocations: usize,
    pub(crate) pool: MemoryPool<CudaBackend>,
}

// Exact setup controls, including floating-point bit patterns. Iteration budget
// and tolerance affect the solve, not the retained preconditioner construction.
fn key(config: &NavierStokesConfig, nv: usize) -> [u64; 12] {
    [
        nv as u64,
        config.linear_solver_backend_preconditioner as u64,
        config.linear_solver_polynomial_degree as u64,
        config.linear_solver_polynomial_damping.to_bits(),
        config.linear_solver_schur_preconditioner as u64,
        config.linear_solver_bdf_pressure_only_amg as u64,
        config.linear_solver_schur_polynomial_degree as u64,
        config.linear_solver_schur_polynomial_damping.to_bits(),
        config.linear_solver_schur_chebyshev_degree as u64,
        config
            .linear_solver_schur_chebyshev_eigenvalue_ratio
            .to_bits(),
        config.linear_solver_chebyshev_degree as u64,
        config.linear_solver_chebyshev_eigenvalue_ratio.to_bits(),
    ]
}

fn same_pattern(a: &CsrMatrix<f64>, b: &CsrMatrix<f64>) -> bool {
    a.num_rows == b.num_rows
        && a.num_cols == b.num_cols
        && a.row_ptrs == b.row_ptrs
        && a.col_indices == b.col_indices
}
fn same_matrix(a: &CsrMatrix<f64>, b: &CsrMatrix<f64>) -> bool {
    same_pattern(a, b) && a.values == b.values
}
fn same_schur(a: Option<&CsrMatrix<f64>>, b: Option<&CsrMatrix<f64>>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => same_matrix(a, b),
        (None, None) => true,
        _ => false,
    }
}

fn prepare(
    pool: &MemoryPool<CudaBackend>,
    matrix: &CsrMatrix<f64>,
    schur: Option<&CsrMatrix<f64>>,
    config: &NavierStokesConfig,
    nv: usize,
) -> Result<Preconditioner, String> {
    let backend = pool.backend();
    let failure = |e: sonium_compute::BackendError| e.to_string();
    match config.linear_solver_backend_preconditioner {
        BackendPreconditionerKind::JacobiSchur => {
            let schur = schur.ok_or("Jacobi-Schur requires a pressure matrix")?;
            let kind = match config.linear_solver_schur_preconditioner {
                BackendPressureSchurPreconditionerKind::Chebyshev => {
                    PressureSchurBackendKind::Chebyshev
                }
                _ => PressureSchurBackendKind::Polynomial,
            };
            let mut action = SaddlePointJacobiSchurBackend::try_from_host_csr(
                backend,
                matrix.as_host_csr_ref(),
                schur.as_host_csr_ref(),
                SaddlePointJacobiSchurBackendConfig {
                    velocity_block_size: nv,
                    velocity_degree: config.linear_solver_polynomial_degree.max(1),
                    velocity_damping: config.linear_solver_polynomial_damping,
                    pressure_degree: config.linear_solver_schur_polynomial_degree.max(1),
                    pressure_damping: config.linear_solver_schur_polynomial_damping,
                    pressure_kind: kind,
                    pressure_chebyshev_degree: config.linear_solver_schur_chebyshev_degree.max(1),
                    pressure_chebyshev_eigenvalue_ratio: config
                        .linear_solver_schur_chebyshev_eigenvalue_ratio
                        .max(1.0),
                },
            )
            .map_err(failure)?;
            let pressure = if config.linear_solver_schur_preconditioner
                == BackendPressureSchurPreconditionerKind::Amg
            {
                let pressure = PressureAmg::new(
                    pool,
                    matrix.num_rows,
                    nv,
                    schur,
                    config.linear_solver_bdf_pressure_only_amg,
                )?;
                action = action.with_owned_pressure_preconditioner(Box::new(pressure.clone()));
                Some(pressure)
            } else {
                None
            };
            Ok(Preconditioner::Schur {
                action: Box::new(action),
                pressure,
            })
        }
        BackendPreconditionerKind::PolynomialJacobi | BackendPreconditionerKind::Chebyshev => {
            let owned =
                PersistentDeviceCsr::try_from_host_csr_ref(backend, matrix.as_host_csr_ref())
                    .map_err(failure)?;
            if config.linear_solver_backend_preconditioner
                == BackendPreconditionerKind::PolynomialJacobi
            {
                Ok(Preconditioner::Polynomial(Box::new(
                    PolynomialJacobiBackend::try_from_owned_csr(
                        backend,
                        owned,
                        &diagonal(matrix),
                        config.linear_solver_polynomial_degree.max(1),
                        config.linear_solver_polynomial_damping,
                    )
                    .map_err(failure)?,
                )))
            } else {
                Ok(Preconditioner::Chebyshev(Box::new(
                    ChebyshevBackend::try_from_owned_csr_host(
                        backend,
                        owned,
                        &indices_i32(&matrix.row_ptrs)?,
                        &indices_i32(&matrix.col_indices)?,
                        &matrix.values,
                        config.linear_solver_chebyshev_degree.max(1),
                        config.linear_solver_chebyshev_eigenvalue_ratio.max(1.0),
                    )
                    .map_err(failure)?,
                )))
            }
        }
    }
}

impl BdfCudaSolver {
    fn new(
        matrix: &CsrMatrix<f64>,
        schur: Option<&CsrMatrix<f64>>,
        config: &NavierStokesConfig,
        nv: usize,
    ) -> Result<Self, String> {
        let backend = CudaBackend::new(0).map_err(|e| e.to_string())?;
        let pool =
            MemoryPool::new(backend, MemoryPoolConfig::default()).map_err(|e| e.to_string())?;
        let operator =
            PersistentDeviceCsr::try_from_host_csr_ref(pool.backend(), matrix.as_host_csr_ref())
                .map_err(|e| e.to_string())?;
        let preconditioner = prepare(&pool, matrix, schur, config, nv)?;
        let n = matrix.num_rows;
        Ok(Self {
            matrix: matrix.clone(),
            schur: schur.cloned(),
            key: key(config, nv),
            operator,
            preconditioner,
            iteration: BackendKrylovWorkspace::new(n),
            rhs: pool.try_alloc(n).map_err(|e| e.to_string())?,
            guess: pool.try_alloc(n).map_err(|e| e.to_string())?,
            solution: pool.try_alloc(n).map_err(|e| e.to_string())?,
            setup_builds: 1,
            matrix_uploads: 1,
            last_allocations: 0,
            pool,
        })
    }

    fn solve(
        &mut self,
        matrix: &CsrMatrix<f64>,
        schur: Option<&CsrMatrix<f64>>,
        config: &NavierStokesConfig,
        nv: usize,
        rhs: &[f64],
        guess: &[f64],
    ) -> Result<(Vec<f64>, usize, bool), String> {
        let backend = self.pool.backend();
        let before = backend.begin_runtime_stats();
        let same_values = same_matrix(&self.matrix, matrix);
        let reused =
            same_values && self.key == key(config, nv) && same_schur(self.schur.as_ref(), schur);
        if !same_values {
            self.operator
                .try_copy_values_from_host(backend, &matrix.values)
                .map_err(|e| e.to_string())?;
            self.matrix_uploads += 1;
        }
        if !reused {
            let refreshable_schur = !matches!(self.preconditioner, Preconditioner::Schur { .. })
                || matches!((self.schur.as_ref(), schur), (Some(a), Some(b)) if same_pattern(a, b));
            let refreshed = self.key == key(config, nv)
                && refreshable_schur
                && self.preconditioner.refresh(
                    backend,
                    &self.operator,
                    matrix,
                    schur,
                    !same_schur(self.schur.as_ref(), schur),
                    nv,
                )?;
            if !refreshed {
                self.preconditioner = prepare(&self.pool, matrix, schur, config, nv)?;
                self.setup_builds += 1;
            }
            self.key = key(config, nv);
            self.schur = schur.cloned();
        }
        if !same_values {
            self.matrix = matrix.clone();
        }
        backend
            .try_copy_h2d(rhs, self.rhs.raw_mut())
            .map_err(|e| e.to_string())?;
        backend
            .try_copy_h2d(guess, self.guess.raw_mut())
            .map_err(|e| e.to_string())?;
        let status = try_fgmres_backend_preconditioned(
            backend,
            &self.pool,
            &mut self.iteration,
            &self.operator,
            self.preconditioner.as_ref(),
            self.rhs.raw(),
            Some(self.guess.raw()),
            self.solution.raw_mut(),
            &BackendKrylovConfig {
                max_iterations: config.linear_solver_max_iterations.max(1),
                restart: config.linear_solver_max_iterations.max(1),
                tolerance: config.linear_solver_tolerance,
                require_native: true,
                exact_residual_check_interval: 0,
            },
            None,
        )
        .map_err(|e| e.to_string())?;
        if !status.converged {
            return Err(format!(
                "CUDA BDF solve did not converge after {} iterations",
                status.iterations
            ));
        }
        let mut output = vec![0.0; matrix.num_rows];
        backend
            .try_copy_d2h(self.solution.raw(), &mut output)
            .map_err(|e| e.to_string())?;
        self.last_allocations = backend.runtime_stats_since(before).allocation_operations;
        Ok((output, status.iterations, reused))
    }
}

/// A failed update/solve drops the entire device cache before any caller recovery.
pub(super) fn solve_cached(
    workspace: &BdfWorkspace,
    config: &NavierStokesConfig,
    matrix: &CsrMatrix<f64>,
    rhs: &[f64],
    guess: &[f64],
    nv: usize,
    schur: Option<&CsrMatrix<f64>>,
) -> Result<(Vec<f64>, usize, bool), String> {
    let mut slot = workspace.cuda_solver_cache.borrow_mut();
    let previous = slot.take();
    let existing = previous
        .as_ref()
        .is_some_and(|solver| same_pattern(&solver.matrix, matrix));
    let mut solver = match previous {
        Some(solver) if same_pattern(&solver.matrix, matrix) => solver,
        _ => BdfCudaSolver::new(matrix, schur, config, nv)?,
    };
    let mut result = solver.solve(matrix, schur, config, nv, rhs, guess)?;
    result.2 &= existing;
    *slot = Some(solver);
    Ok(result)
}

#[cfg(test)]
mod tests;
