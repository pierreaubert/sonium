//! Persistent PCG with independent device residual certification.

// Rust guideline compliant 2026-02-21
use super::DeviceInitialGuess;
use num_traits::{Float, Zero};
use sonium_compute::pool::Buffer;
use sonium_compute::{
    Backend, DeviceBufferLike, DeviceOperator, DevicePreconditioner, DeviceScalar, MemoryPool,
    OomError, PcgBackendConfig, PcgBackendError, PcgBackendWorkspace, ReductionMode,
    pcg_backend_preconditioned,
};

/// Validate primitive admission for a CSR PCG solve before preparation.
///
/// Checks CSR SpMV, BLAS1 and fast reductions without allocating or launching
/// device work. PCG does not require fused multi-dot or block projection kernels.
/// Caller-supplied operators/preconditioners must separately validate their own
/// dependencies and positive-definiteness. This does not inspect their actions.
///
/// # Errors
/// Rejects unsupported primitives and, when requested, non-native execution.
pub fn validate_backend_pcg<B: Backend, T: DeviceScalar>(
    backend: &B,
    require_native: bool,
) -> Result<(), PcgBackendError> {
    validate_pcg_report::<T>(&backend.capabilities(), require_native)
}

fn validate_pcg_report<T: DeviceScalar>(
    caps: &sonium_compute::BackendCapabilities,
    require_native: bool,
) -> Result<(), PcgBackendError> {
    for (operation, state) in [
        ("pcg_csr_spmv", caps.csr_spmv.state_for::<T>()),
        ("pcg_blas1", caps.blas1.state_for::<T>()),
        ("pcg_fast_reduction", caps.reduction.fast.state_for::<T>()),
    ] {
        if !state.is_available() || (require_native && !state.is_native()) {
            return Err(sonium_compute::BackendError::Unsupported {
                backend: caps.backend_name,
                operation,
                scalar: std::any::type_name::<T>(),
            }
            .into());
        }
    }
    Ok(())
}

/// Caller certification required by preconditioned conjugate gradients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevicePcgAssumption {
    /// Operator and stationary preconditioner are both Hermitian positive definite.
    ///
    /// For real scalars this means symmetric positive definite. The session
    /// cannot prove this property for an arbitrary device operator. A small
    /// residual does not establish it. Helmholtz and indefinite saddle-point
    /// systems must use an appropriate nonsymmetric solver instead.
    PositiveDefinite,
}

/// Independently certified result of a persistent PCG solve.
#[derive(Debug, Clone)]
pub struct DevicePcgStatus<R> {
    /// Number of PCG iterations performed.
    pub iterations: usize,
    /// Residual reported by the recurrence, before independent certification.
    pub recurrence_residual: R,
    /// Recomputed `||A*x-b||/||b||`, or absolute residual when `b` is zero.
    pub true_residual: R,
    /// Whether the independent residual satisfies the requested tolerance.
    pub converged: bool,
}

/// Retains PCG vectors and publishes only independently certified solutions.
///
/// Operators, preconditioners, and right-hand sides remain caller-owned device
/// objects. Matrix values may change between solves at a fixed dimension. Use
/// a persistent sparse operator to retain its descriptors. This session retains
/// all its own device allocations and performs no host vector transfers.
/// Operator/preconditioner setup and fallback behavior remain their contracts.
pub struct DevicePcgSession<'a, B: Backend, T: DeviceScalar> {
    pool: &'a MemoryPool<B>,
    workspace: PcgBackendWorkspace<B, T>,
    solution: Buffer<T, B>,
    candidate: Buffer<T, B>,
    residual: Buffer<T, B>,
    valid: bool,
}

impl<B: Backend, T: DeviceScalar> std::fmt::Debug for DevicePcgSession<'_, B, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DevicePcgSession")
            .field("dimension", &self.workspace.dimension())
            .field("has_solution", &self.valid)
            .finish_non_exhaustive()
    }
}

impl<'a, B: Backend, T: DeviceScalar> DevicePcgSession<'a, B, T> {
    /// Allocate solution and certification buffers under the supplied pool quota.
    ///
    /// The caller certifies the operator and preconditioner satisfy `assumption`
    /// for every solve. PCG workspace is allocated lazily on first use.
    ///
    /// # Errors
    /// Returns allocation errors; partial allocations are automatically refunded.
    pub fn try_new(
        pool: &'a MemoryPool<B>,
        dimension: usize,
        _assumption: DevicePcgAssumption,
    ) -> Result<Self, OomError> {
        Ok(Self {
            pool,
            workspace: PcgBackendWorkspace::new(dimension),
            solution: pool.try_alloc(dimension)?,
            candidate: pool.try_alloc(dimension)?,
            residual: pool.try_alloc(dimension)?,
            valid: false,
        })
    }

    /// Borrow the independently certified solution from the most recent solve.
    #[must_use]
    pub fn solution(&self) -> Option<&B::Buffer<T>> {
        self.valid.then(|| self.solution.raw())
    }

    /// Solve with a stationary positive-definite preconditioner and certify the result.
    ///
    /// A zero RHS produces the zero solution, irrespective of the previous guess.
    /// Nonconvergence or any error invalidates published solution access. No
    /// partial iterate is exposed. Both configurations and device vectors are
    /// checked before starting the iteration. Small reductions reach the host.
    ///
    /// # Errors
    /// Returns errors for invalid dimensions/configuration, missing previous
    /// solutions, PCG breakdown, allocation failures, or backend failures.
    /// Backend failures retain their typed context-recovery requirements.
    pub fn solve<A, P>(
        &mut self,
        operator: &A,
        preconditioner: &P,
        rhs: &B::Buffer<T>,
        initial: DeviceInitialGuess,
        config: &PcgBackendConfig<T>,
    ) -> Result<DevicePcgStatus<T::Real>, PcgBackendError>
    where
        A: DeviceOperator<B, T> + ?Sized,
        P: DevicePreconditioner<B, T> + ?Sized,
    {
        let previous_valid = std::mem::replace(&mut self.valid, false);
        let n = self.workspace.dimension();
        if n == 0 || operator.n() != n || rhs.len() != n {
            return Err(PcgBackendError::InvalidArgument {
                reason:
                    "PCG session requires matching nonzero operator, RHS and session dimensions"
                        .into(),
            });
        }
        if config.max_iterations == 0
            || !config.tolerance.is_finite()
            || config.tolerance <= T::Real::zero()
        {
            return Err(PcgBackendError::InvalidArgument {
                reason:
                    "PCG session requires positive iteration limit and finite positive tolerance"
                        .into(),
            });
        }
        if initial == DeviceInitialGuess::PreviousSolution && !previous_valid {
            return Err(PcgBackendError::InvalidArgument {
                reason: "no certified previous PCG solution".into(),
            });
        }
        let backend = self.pool.backend();
        let rhs_norm = backend.try_nrm2_with_mode::<T>(ReductionMode::Fast, rhs)?;
        let (iterations, recurrence_residual) = if rhs_norm == T::Real::zero() {
            // Copy the zero RHS instead of multiplying possibly failed scratch
            // by zero (IEEE arithmetic would preserve NaNs).
            backend.try_copy_d2d(rhs, self.candidate.raw_mut())?;
            (0, T::Real::zero())
        } else {
            let guess =
                (initial == DeviceInitialGuess::PreviousSolution).then(|| self.solution.raw());
            let status = pcg_backend_preconditioned(
                backend,
                self.pool,
                &mut self.workspace,
                operator,
                preconditioner,
                rhs,
                guess,
                self.candidate.raw_mut(),
                config,
            )?;
            (status.iterations, status.final_residual)
        };
        operator.try_apply(backend, self.candidate.raw(), self.residual.raw_mut())?;
        backend.try_axpy(-T::one(), rhs, self.residual.raw_mut())?;
        let residual_norm =
            backend.try_nrm2_with_mode::<T>(ReductionMode::Fast, self.residual.raw())?;
        let true_residual = if rhs_norm == T::Real::zero() {
            residual_norm
        } else {
            residual_norm / rhs_norm
        };
        let converged = true_residual.is_finite() && true_residual <= config.tolerance;
        if converged {
            std::mem::swap(&mut self.solution, &mut self.candidate);
            self.valid = true;
        }
        Ok(DevicePcgStatus {
            iterations,
            recurrence_residual,
            true_residual,
            converged,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::{Complex32, Complex64};
    use num_traits::FromPrimitive;
    use sonium_compute::{
        CpuBackend, DeviceCsr, HostCsrRef, IdentityDevicePreconditioner, MemoryPoolConfig,
        PersistentDeviceCsr,
    };
    use sonium_solvers_preconditioners::{AmgConfig, AmgSmoother, BackendAmgPreconditioner};
    use sonium_solvers_sparse::CsrMatrix;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn scalar_contract<T: DeviceScalar>(
        values: &[T],
        rhs_host: &[T],
        expected: &[T],
        tolerance: f64,
    ) {
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let backend = pool.backend();
        let matrix =
            DeviceCsr::try_from_host(backend, 2, 2, &[0, 2, 4], &[0, 1, 0, 1], values).unwrap();
        let mut rhs = pool.try_alloc::<T>(2).unwrap();
        backend.try_copy_h2d(rhs_host, rhs.raw_mut()).unwrap();
        let mut session =
            DevicePcgSession::try_new(&pool, 2, DevicePcgAssumption::PositiveDefinite).unwrap();
        let config = PcgBackendConfig {
            max_iterations: 20,
            tolerance: T::Real::from_f64(tolerance).unwrap(),
        };
        let first = session
            .solve(
                &matrix,
                &IdentityDevicePreconditioner,
                rhs.raw(),
                DeviceInitialGuess::Zero,
                &config,
            )
            .unwrap();
        assert!(first.converged);
        let mut observed = vec![T::zero(); 2];
        backend
            .try_copy_d2h(session.solution().unwrap(), &mut observed)
            .unwrap();
        for (&x, &reference) in observed.iter().zip(expected) {
            assert!((x - reference).magnitude() < T::Real::from_f64(tolerance * 10.0).unwrap());
        }
        let peak = pool.high_water_bytes();
        let repeated = session
            .solve(
                &matrix,
                &IdentityDevicePreconditioner,
                rhs.raw(),
                DeviceInitialGuess::PreviousSolution,
                &config,
            )
            .unwrap();
        assert!(repeated.converged);
        assert_eq!(repeated.iterations, 0);
        assert_eq!(pool.high_water_bytes(), peak);
        // A previous failed operation may leave scratch non-finite. A zero
        // RHS must overwrite it without evaluating NaN * 0.
        backend
            .try_copy_h2d(
                &[T::from_real(T::Real::nan()); 2],
                session.candidate.raw_mut(),
            )
            .unwrap();
        backend
            .try_copy_h2d(&[T::zero(), T::zero()], rhs.raw_mut())
            .unwrap();
        let zero = session
            .solve(
                &matrix,
                &IdentityDevicePreconditioner,
                rhs.raw(),
                DeviceInitialGuess::PreviousSolution,
                &config,
            )
            .unwrap();
        assert!(zero.converged);
        backend
            .try_copy_d2h(session.solution().unwrap(), &mut observed)
            .unwrap();
        assert!(observed.iter().all(|v| v.magnitude() == T::Real::zero()));
        backend.try_copy_h2d(rhs_host, rhs.raw_mut()).unwrap();
        let short = PcgBackendConfig {
            max_iterations: 1,
            ..config
        };
        let incomplete = session
            .solve(
                &matrix,
                &IdentityDevicePreconditioner,
                rhs.raw(),
                DeviceInitialGuess::Zero,
                &short,
            )
            .unwrap();
        assert!(!incomplete.converged);
        assert!(session.solution().is_none());
        assert!(
            session
                .solve(
                    &matrix,
                    &IdentityDevicePreconditioner,
                    rhs.raw(),
                    DeviceInitialGuess::PreviousSolution,
                    &short
                )
                .is_err()
        );
        drop(session);
        drop(rhs);
        assert_eq!(pool.bytes_in_use(), 0);
    }

    #[test]
    fn all_scalar_families_certify_and_reuse_solutions() {
        scalar_contract(&[4.0_f64, 1.0, 1.0, 3.0], &[6.0, 7.0], &[1.0, 2.0], 1e-12);
        scalar_contract(&[4.0_f32, 1.0, 1.0, 3.0], &[6.0, 7.0], &[1.0, 2.0], 1e-5);
        scalar_contract(
            &[
                Complex64::new(4.0, 0.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(0.0, -1.0),
                Complex64::new(3.0, 0.0),
            ],
            &[Complex64::new(5.0, 6.0), Complex64::new(7.0, -4.0)],
            &[Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)],
            1e-12,
        );
        scalar_contract(
            &[
                Complex32::new(4.0, 0.0),
                Complex32::new(0.0, 1.0),
                Complex32::new(0.0, -1.0),
                Complex32::new(3.0, 0.0),
            ],
            &[Complex32::new(5.0, 6.0), Complex32::new(7.0, -4.0)],
            &[Complex32::new(1.0, 1.0), Complex32::new(2.0, -1.0)],
            1e-5,
        );
    }

    struct ChangedOperator(AtomicUsize);
    impl DeviceOperator<CpuBackend, f64> for ChangedOperator {
        fn n(&self) -> usize {
            2
        }
        fn apply(
            &self,
            backend: &CpuBackend,
            x: &<CpuBackend as Backend>::Buffer<f64>,
            y: &mut <CpuBackend as Backend>::Buffer<f64>,
        ) {
            backend.copy_d2d(x, y);
            if self.0.fetch_add(1, Ordering::Relaxed) > 0 {
                backend.scale(2.0, y);
            }
        }
    }

    #[test]
    fn recurrence_convergence_cannot_bypass_true_residual() {
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let mut session =
            DevicePcgSession::try_new(&pool, 2, DevicePcgAssumption::PositiveDefinite).unwrap();
        let mut rhs = pool.try_alloc::<f64>(2).unwrap();
        pool.backend()
            .try_copy_h2d(&[1.0, 2.0], rhs.raw_mut())
            .unwrap();
        let status = session
            .solve(
                &ChangedOperator(AtomicUsize::new(0)),
                &IdentityDevicePreconditioner,
                rhs.raw(),
                DeviceInitialGuess::Zero,
                &PcgBackendConfig::default(),
            )
            .unwrap();
        assert!(status.recurrence_residual < 1e-12);
        assert!(status.true_residual > 0.9);
        assert!(!status.converged);
        assert!(session.solution().is_none());
    }

    fn amg_repeated<B: Backend>(backend: B) {
        let n = 64;
        let mut triplets = Vec::new();
        for i in 0..n {
            triplets.push((i, i, 2.0_f64));
            if i > 0 {
                triplets.push((i, i - 1, -1.0));
            }
            if i + 1 < n {
                triplets.push((i, i + 1, -1.0));
            }
        }
        let host = CsrMatrix::from_triplets(n, n, triplets);
        let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 26)).unwrap();
        let backend = pool.backend();
        let operator = PersistentDeviceCsr::try_from_host_csr_ref(
            backend,
            HostCsrRef::try_new(n, n, &host.row_ptrs, &host.col_indices, &host.values).unwrap(),
        )
        .unwrap();
        let amg = BackendAmgPreconditioner::try_from_csr(
            backend,
            &host,
            AmgConfig {
                coarse_size: 8,
                smoother: AmgSmoother::Jacobi,
                num_pre_smooth: 2,
                num_post_smooth: 2,
                jacobi_weight: 0.6,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(amg.num_levels() > 1);
        let mut rhs = pool.try_alloc::<f64>(n).unwrap();
        let mut session =
            DevicePcgSession::try_new(&pool, n, DevicePcgAssumption::PositiveDefinite).unwrap();
        let config = PcgBackendConfig {
            max_iterations: 200,
            tolerance: 1e-10,
        };
        for scale in [1.0, -2.0, 0.5] {
            let mut source = vec![0.0; n];
            source[0] = scale;
            source[n - 1] = scale;
            backend.try_copy_h2d(&source, rhs.raw_mut()).unwrap();
            let initial = if session.solution().is_some() {
                DeviceInitialGuess::PreviousSolution
            } else {
                DeviceInitialGuess::Zero
            };
            let status = session
                .solve(&operator, &amg, rhs.raw(), initial, &config)
                .unwrap();
            assert!(status.converged, "{status:?}");
            let mut observed = vec![0.0; n];
            backend
                .try_copy_d2h(session.solution().unwrap(), &mut observed)
                .unwrap();
            assert!(observed.iter().all(|x| (x - scale).abs() < 1e-7));
        }
        assert_eq!(amg.diagnostics().runtime.setup_count, 1);
        assert_eq!(amg.diagnostics().runtime.fallback_count, 0);
    }

    #[test]
    fn cpu_amg_repeated_poisson_solves() {
        amg_repeated(CpuBackend);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_amg_repeated_poisson_solves() {
        match sonium_compute::CudaBackend::new(0) {
            Ok(backend) => amg_repeated(backend),
            Err(error) => {
                assert!(
                    std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                    "CUDA required: {error}"
                );
                eprintln!("SKIP cuda_amg_repeated_poisson_solves: {error}");
            }
        }
    }

    #[test]
    fn constructor_refunds_partial_allocations() {
        let pool = MemoryPool::new(CpuBackend, MemoryPoolConfig::with_cap(32)).unwrap();
        assert!(
            DevicePcgSession::<_, f64>::try_new(&pool, 2, DevicePcgAssumption::PositiveDefinite)
                .is_err()
        );
        assert_eq!(pool.bytes_in_use(), 0);
    }

    fn check_pcg_admission<T: DeviceScalar>() {
        use sonium_compute::{CapabilityState, ScalarCapabilities};
        for field in 0..3 {
            for state in [
                CapabilityState::Unsupported,
                CapabilityState::HostStagedFallback,
                CapabilityState::HostSharedFallback,
                CapabilityState::Native,
                CapabilityState::VendorNative,
            ] {
                let mut caps = CpuBackend.capabilities();
                let (target, operation) = match field {
                    0 => (&mut caps.csr_spmv, "pcg_csr_spmv"),
                    1 => (&mut caps.blas1, "pcg_blas1"),
                    _ => (&mut caps.reduction.fast, "pcg_fast_reduction"),
                };
                *target = ScalarCapabilities::splat(state);
                let strict = validate_pcg_report::<T>(&caps, true);
                assert_eq!(strict.is_ok(), state.is_native());
                if let Err(PcgBackendError::Backend {
                    source: sonium_compute::BackendError::Unsupported { operation: actual, .. },
                }) = strict {
                    assert_eq!(actual, operation);
                }
                assert_eq!(validate_pcg_report::<T>(&caps, false).is_ok(), state.is_available());
            }
        }
        let mut caps = CpuBackend.capabilities();
        let unavailable = ScalarCapabilities::splat(CapabilityState::Unsupported);
        caps.multi_dot = unavailable;
        caps.multi_dot_strided = unavailable;
        caps.basis_axpy_strided = unavailable;
        caps.csr_spmm = unavailable;
        caps.axpy_many = unavailable;
        validate_pcg_report::<T>(&caps, true).unwrap();
    }

    #[test]
    fn pcg_admission_all_scalars() {
        check_pcg_admission::<f32>();
        check_pcg_admission::<f64>();
        check_pcg_admission::<Complex32>();
        check_pcg_admission::<Complex64>();
        let backend = CpuBackend;
        let before = backend.transfer_stats();
        validate_backend_pcg::<_, f64>(&backend, true).unwrap();
        assert_eq!(before, backend.transfer_stats());
    }
}
