//! Retained complex dense LU with device-input numeric refresh.

// Rust guideline compliant 2026-02-21
use super::*;

impl CudaBackend {
    /// Factor a host column-major matrix and retain its CUDA LU factors.
    ///
    /// # Errors
    /// Rejects empty, malformed, non-finite or singular matrices and CUDA failures.
    pub fn dense_lu_c64_from_column_major(
        &self,
        n: usize,
        values: &[Complex64],
    ) -> Result<CudaDenseLuC64, OomError> {
        self.validate_lu_c64_shape(n, values.len())?;
        if values
            .iter()
            .any(|v| !v.re.is_finite() || !v.im.is_finite())
        {
            return Err(OomError::Invalid("dense LU: non-finite matrix".into()));
        }
        let mut factors = self.raw_alloc::<Complex64>(values.len())?;
        self.try_copy_h2d(values, &mut factors)
            .map_err(backend_error_to_oom)?;
        self.factor_owned_c64(n, factors)
    }

    /// Factor a resident column-major matrix without host matrix staging.
    ///
    /// The input is preserved. Factors, pivots and workspace remain allocated
    /// for repeated solves and same-size numeric refreshes.
    ///
    /// # Errors
    /// Rejects invalid sizes, foreign streams, non-finite or singular matrices,
    /// allocation failures and CUDA failures.
    pub fn dense_lu_c64_from_device(
        &self,
        n: usize,
        values: &CudaBuffer<Complex64>,
    ) -> Result<CudaDenseLuC64, OomError> {
        self.validate_lu_c64_shape(n, values.len)?;
        self.validate_lu_buffer(values)?;
        self.validate_lu_c64_finite(values)?;
        let mut factors = self.raw_alloc::<Complex64>(values.len)?;
        self.try_copy_d2d(values, &mut factors)
            .map_err(backend_error_to_oom)?;
        self.factor_owned_c64(n, factors)
    }

    /// Refactor a resident matrix using retained factorization storage.
    ///
    /// Invalid input leaves the old factors usable. Once copying begins, any
    /// failure invalidates the factors until a successful refresh. No matrix
    /// data crosses the host boundary; only validation/status scalars do.
    ///
    /// # Errors
    /// Rejects sizes, foreign streams, non-finite or singular matrices and CUDA
    /// failures. A backend failure may also require backend recovery.
    pub fn dense_lu_c64_refactor_device(
        &self,
        lu: &mut CudaDenseLuC64,
        values: &CudaBuffer<Complex64>,
    ) -> Result<(), OomError> {
        self.validate_lu_c64_shape(lu.n, values.len)?;
        self.validate_lu_buffer(&lu.factors)?;
        self.validate_lu_buffer(values)?;
        self.validate_lu_c64_finite(values)?;
        lu.valid = false;
        self.try_copy_d2d(values, &mut lu.factors)
            .map_err(backend_error_to_oom)?;
        self.factor_c64_in_place(lu)
    }

    /// Refactor a host matrix using retained complex-double LU storage.
    ///
    /// Validation preserves existing factors. Once upload begins, errors invalidate
    /// them until a complete refresh succeeds. Factor, pivot and vendor workspace
    /// buffers retain their allocations; the host matrix is explicitly uploaded.
    ///
    /// # Errors
    /// Rejects invalid dimensions, non-finite values, foreign factors, singular
    /// matrices and backend failures. Backend recovery may be required before retry.
    pub fn dense_lu_c64_refactor_column_major(
        &self,
        lu: &mut CudaDenseLuC64,
        values: &[Complex64],
    ) -> Result<(), OomError> {
        self.validate_lu_c64_shape(lu.n, values.len())?;
        self.validate_lu_buffer(&lu.factors)?;
        if values.iter().any(|v| !v.re.is_finite() || !v.im.is_finite()) {
            return Err(OomError::Invalid("dense LU: non-finite matrix".into()));
        }
        lu.valid = false;
        self.try_copy_h2d(values, &mut lu.factors)
            .map_err(backend_error_to_oom)?;
        self.factor_c64_in_place(lu)
    }

    fn validate_lu_c64_shape(&self, n: usize, len: usize) -> Result<(), OomError> {
        self.check_pending_error().map_err(backend_error_to_oom)?;
        if n == 0 || i32::try_from(n).is_err() || n.checked_mul(n) != Some(len) {
            return Err(OomError::Invalid(
                "dense LU: invalid matrix dimensions".into(),
            ));
        }
        Ok(())
    }

    fn validate_lu_buffer<T: BufferElement>(&self, buffer: &CudaBuffer<T>) -> Result<(), OomError> {
        if buffer
            .raw()
            .is_none_or(|raw| !Arc::ptr_eq(raw.stream(), &self.stream))
        {
            return Err(OomError::Invalid(
                "dense LU: empty buffer or foreign CUDA stream".into(),
            ));
        }
        Ok(())
    }

    fn validate_lu_c64_finite(&self, values: &CudaBuffer<Complex64>) -> Result<(), OomError> {
        if self
            .try_all_finite_native(values)
            .map_err(backend_error_to_oom)?
            != Some(true)
        {
            return Err(OomError::Invalid(
                "dense LU: finite validation failed".into(),
            ));
        }
        Ok(())
    }

    fn validate_lu_c64(&self, lu: &CudaDenseLuC64) -> Result<(), OomError> {
        self.check_pending_error().map_err(backend_error_to_oom)?;
        self.validate_lu_buffer(&lu.factors)?;
        if !lu.valid {
            return Err(OomError::Invalid(
                "dense LU: factors require successful numeric refresh".into(),
            ));
        }
        Ok(())
    }

    fn factor_owned_c64(
        &self,
        n: usize,
        mut factors: CudaBuffer<Complex64>,
    ) -> Result<CudaDenseLuC64, OomError> {
        let n_i32 = i32::try_from(n)
            .map_err(|_| OomError::Invalid("dense LU: dimension overflow".into()))?;
        let mut lwork = 0_i32;
        let (a_ptr, guard) = buf_ptr_mut(&mut factors, &self.stream)
            .ok_or_else(|| OomError::Invalid("dense LU: empty matrix".into()))?;
        self.runtime_stats.record_library_operation();
        // SAFETY: the owned matrix has n*n complex elements on this stream;
        // n fits the vendor integer type and the guard retains its allocation.
        let status = unsafe {
            cusolver_sys::cusolverDnZgetrf_bufferSize(
                self.solver,
                n_i32,
                n_i32,
                a_ptr as *mut cusolver_sys::cuDoubleComplex,
                n_i32,
                &mut lwork,
            )
        };
        check_cusolver("cusolverDnZgetrf_bufferSize", status)?;
        drop(guard);
        let workspace_len = usize::try_from(lwork)
            .map_err(|_| OomError::Invalid("dense LU: negative workspace size".into()))?;
        let mut lu = CudaDenseLuC64 {
            n,
            factors,
            ipiv: self.raw_alloc::<i32>(n)?,
            info: Mutex::new(self.raw_alloc::<i32>(1)?),
            workspace: self.raw_alloc::<Complex64>(workspace_len.max(1))?,
            valid: false,
        };
        self.factor_c64_in_place(&mut lu)?;
        Ok(lu)
    }

    fn factor_c64_in_place(&self, lu: &mut CudaDenseLuC64) -> Result<(), OomError> {
        lu.valid = false;
        self.try_custom_kernel_preflight()
            .map_err(backend_error_to_oom)?;
        let n = i32::try_from(lu.n)
            .map_err(|_| OomError::Invalid("dense LU: dimension overflow".into()))?;
        let mut info = lu
            .info
            .lock()
            .map_err(|_| OomError::Invalid("dense LU: poisoned status lock".into()))?;
        let (a, a_guard) = buf_ptr_mut(&mut lu.factors, &self.stream)
            .ok_or_else(|| OomError::Invalid("dense LU: empty factors".into()))?;
        let (work, work_guard) = buf_ptr_mut(&mut lu.workspace, &self.stream)
            .ok_or_else(|| OomError::Invalid("dense LU: empty workspace".into()))?;
        let (pivots, pivots_guard) = buf_ptr_mut(&mut lu.ipiv, &self.stream)
            .ok_or_else(|| OomError::Invalid("dense LU: empty pivots".into()))?;
        let (status_ptr, status_guard) = buf_ptr_mut(&mut info, &self.stream)
            .ok_or_else(|| OomError::Invalid("dense LU: empty status".into()))?;
        self.runtime_stats.record_library_operation();
        // SAFETY: private factorization storage was allocated for this dimension
        // and vendor workspace query. All pointers use this stream and remain
        // guarded until submission completes; status download waits for execution.
        let status = unsafe {
            cusolver_sys::cusolverDnZgetrf(
                self.solver,
                n,
                n,
                a as *mut cusolver_sys::cuDoubleComplex,
                n,
                work as *mut cusolver_sys::cuDoubleComplex,
                pivots as *mut i32,
                status_ptr as *mut i32,
            )
        };
        check_cusolver("cusolverDnZgetrf", status)?;
        drop((a_guard, work_guard, pivots_guard, status_guard));
        self.check_dense_lu_info(&info, "cusolverDnZgetrf")?;
        lu.valid = true;
        Ok(())
    }

    /// Solve `op(A) * out = rhs` for one `Complex64` right-hand side while
    /// retaining the LU factors and the right-hand side on the device.
    pub fn dense_lu_c64_solve(
        &self,
        lu: &CudaDenseLuC64,
        rhs: &CudaBuffer<Complex64>,
        out: &mut CudaBuffer<Complex64>,
        operation: Transpose,
    ) -> Result<(), OomError> {
        self.validate_lu_c64(lu)?;
        self.validate_lu_buffer(rhs)?;
        self.validate_lu_buffer(out)?;
        self.try_custom_kernel_preflight()
            .map_err(backend_error_to_oom)?;
        if rhs.len != lu.n || out.len != lu.n {
            return Err(OomError::Invalid(format!(
                "dense LU solve: rhs/out length mismatch, n={}, rhs={}, out={}",
                lu.n, rhs.len, out.len
            )));
        }
        self.try_copy_d2d(rhs, out).map_err(backend_error_to_oom)?;

        let n_i32 = i32::try_from(lu.n).map_err(|_| {
            OomError::Invalid(format!("dense LU solve: n={} exceeds i32::MAX", lu.n))
        })?;
        let (a_ptr, _a_guard) = buf_ptr(&lu.factors, &self.stream)
            .ok_or_else(|| OomError::Invalid("dense LU solve: empty factor matrix".into()))?;
        let (ipiv_ptr, _ipiv_guard) = buf_ptr(&lu.ipiv, &self.stream)
            .ok_or_else(|| OomError::Invalid("dense LU solve: empty pivot buffer".into()))?;
        let (b_ptr, _b_guard) = buf_ptr_mut(out, &self.stream)
            .ok_or_else(|| OomError::Invalid("dense LU solve: empty output buffer".into()))?;
        let mut info = lu
            .info
            .lock()
            .map_err(|_| OomError::Invalid("dense LU solve: info mutex poisoned".into()))?;
        let (info_ptr, _info_guard) = buf_ptr_mut(&mut info, &self.stream)
            .ok_or_else(|| OomError::Invalid("dense LU solve: empty info buffer".into()))?;
        let trans = match operation {
            Transpose::NoTrans => cusolver_sys::cublasOperation_t::CUBLAS_OP_N,
            Transpose::Trans => cusolver_sys::cublasOperation_t::CUBLAS_OP_T,
            Transpose::ConjTrans => cusolver_sys::cublasOperation_t::CUBLAS_OP_C,
        };

        self.runtime_stats.record_library_operation();
        // SAFETY: all buffers share this stream, dimensions and factor validity
        // are checked, and the pointer guards live through the cuSOLVER call.
        let st = unsafe {
            cusolver_sys::cusolverDnZgetrs(
                self.solver,
                trans,
                n_i32,
                1,
                a_ptr as *const cusolver_sys::cuDoubleComplex,
                n_i32,
                ipiv_ptr as *const i32,
                b_ptr as *mut cusolver_sys::cuDoubleComplex,
                n_i32,
                info_ptr as *mut i32,
            )
        };
        check_cusolver("cusolverDnZgetrs", st)?;
        drop((_a_guard, _ipiv_guard, _b_guard, _info_guard));
        Ok(())
    }
}
