//! Native incident-field evaluation over retained BEM collocation geometry.

// Rust guideline compliant 2026-02-21
use super::*;

impl CudaBackend {
    /// Whether native incident evaluation and its finite-data validation are available.
    pub fn has_bem_incident_rhs_c64(&self) -> bool {
        self.bem_kernels.is_some() && self.validation_kernels.is_some()
    }

    /// Evaluate incident pressure and normal derivative into a resident BEM RHS.
    ///
    /// `points` stores six f64 values per row: position followed by normal.
    /// Each source stores six f64 values: direction/position, complex amplitude
    /// (real, imaginary), and kind (0 for plane wave, 1 for point source).
    /// Only the first `source_count` records are evaluated; remaining capacity
    /// must still be finite. The result is `-(gamma*p + beta_tau*dp/dn)`.
    /// Point contributions at radius <= 1e-10 are zero, matching the CPU API.
    /// Empty source lists produce a zero RHS. Sources are accumulated in order.
    ///
    /// # Errors
    /// Rejects invalid sizes, scalar parameters, kinds, non-finite data, foreign
    /// streams and backend failures. Outputs may change on failure and must not
    /// be published until this method succeeds. Validation downloads flags only.
    #[allow(clippy::too_many_arguments)]
    pub fn try_bem_incident_rhs_c64_device(
        &self,
        points: &CudaBuffer<f64>,
        sources: &CudaBuffer<f64>,
        source_count: usize,
        wave_number: f64,
        harmonic_factor: f64,
        gamma: f64,
        beta_tau: Complex64,
        rhs: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        const OP: &str = "bem_incident_rhs_c64";
        self.check_pending_error()?;
        let invalid = |reason: &str| BackendError::InvalidArgument {
            operation: OP,
            reason: reason.into(),
        };
        if rhs.len.checked_mul(6) != Some(points.len)
            || !sources.len.is_multiple_of(6)
            || source_count > sources.len / 6
        {
            return Err(invalid("invalid geometry, source count or RHS shape"));
        }
        if ![
            wave_number,
            harmonic_factor,
            gamma,
            beta_tau.re,
            beta_tau.im,
        ]
        .into_iter()
        .all(f64::is_finite)
        {
            return Err(invalid("non-finite physics parameters"));
        }
        for raw in [points.raw(), sources.raw(), rhs.raw()]
            .into_iter()
            .flatten()
        {
            if !Arc::ptr_eq(raw.stream(), &self.stream) {
                return Err(invalid("foreign CUDA stream"));
            }
        }
        if rhs.len == 0 {
            return Ok(());
        }
        if !self.has_bem_incident_rhs_c64() {
            return Err(BackendError::Unsupported {
                backend: "cuda",
                operation: OP,
                scalar: "Complex64",
            });
        }
        let rows = u32::try_from(rhs.len).map_err(|_| invalid("row count exceeds launch range"))?;
        if self.try_all_finite_native(points)? != Some(true)
            || self.try_all_finite_native(sources)? != Some(true)
        {
            return Err(invalid("non-finite geometry or source data"));
        }
        let kernels = self.bem_kernels.as_ref().ok_or(BackendError::Unsupported {
            backend: "cuda",
            operation: OP,
            scalar: "Complex64",
        })?;
        let n = rhs.len as u64;
        let count = source_count as u64;
        let points_raw = points.raw().ok_or_else(|| invalid("missing geometry"))?;
        // Empty source lists never dereference this valid dummy pointer.
        let sources_raw = sources.raw().unwrap_or(points_raw);
        self.try_custom_kernel_preflight()?;
        self.record_custom_kernel_launch();
        // SAFETY: dimensions bound every six-value record and output row;
        // pointers share this stream, and all scalars match the kernel ABI.
        unsafe {
            self.stream
                .launch_builder(&kernels.incident_rhs_c64)
                .arg(points_raw)
                .arg(sources_raw)
                .arg(rhs.raw_mut().ok_or_else(|| invalid("missing RHS"))?)
                .arg(&n)
                .arg(&count)
                .arg(&wave_number)
                .arg(&harmonic_factor)
                .arg(&gamma)
                .arg(&beta_tau.re)
                .arg(&beta_tau.im)
                .launch(LaunchConfig {
                    grid_dim: (rows.div_ceil(128), 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|error| self.custom_kernel_error(OP, format!("{error:?}")))?;
        }
        if self.try_all_finite_native(rhs)? != Some(true) {
            return Err(invalid("invalid source kind or non-finite incident RHS"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MemoryPool, MemoryPoolConfig};

    #[test]
    fn incident_admission_rejects_missing_modules_without_transfers() {
        for missing_validation in [false, true] {
            let mut backend = match CudaBackend::new_default() {
                Ok(backend) => backend,
                Err(error) => {
                    assert!(
                        std::env::var_os("SONIUM_REQUIRE_CUDA").is_none(),
                        "CUDA required: {error}"
                    );
                    return;
                }
            };
            assert!(backend.has_bem_incident_rhs_c64());
            if missing_validation {
                backend.validation_kernels = None;
            } else {
                backend.bem_kernels = None;
            }
            assert!(!backend.has_bem_incident_rhs_c64());
            let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
            let points = pool.try_alloc::<f64>(6).unwrap();
            let sources = pool.try_alloc::<f64>(6).unwrap();
            let mut rhs = pool.try_alloc::<Complex64>(1).unwrap();
            let sentinel = Complex64::new(3.0, -2.0);
            pool.backend()
                .try_copy_h2d(&[sentinel], rhs.raw_mut())
                .unwrap();
            let before = pool.backend().begin_runtime_stats();
            let error = pool
                .backend()
                .try_bem_incident_rhs_c64_device(
                    points.raw(),
                    sources.raw(),
                    1,
                    1.0,
                    1.0,
                    1.0,
                    Complex64::new(0.0, 0.2),
                    rhs.raw_mut(),
                )
                .unwrap_err();
            assert!(matches!(error, BackendError::Unsupported { .. }));
            let stats = pool.backend().runtime_stats_since(before);
            assert_eq!(stats.allocation_operations, 0);
            assert_eq!(stats.transfer.h2d_bytes, 0);
            assert_eq!(stats.transfer.d2h_bytes, 0);
            assert_eq!(stats.kernel_launches, 0);
            let mut actual = [Complex64::new(0.0, 0.0)];
            pool.backend().try_copy_d2h(rhs.raw(), &mut actual).unwrap();
            assert_eq!(actual, [sentinel]);
        }
    }
}
