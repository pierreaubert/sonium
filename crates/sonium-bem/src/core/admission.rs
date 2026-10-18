//! Checked resource admission for high-level BEM solves.
//!
//! Assembly is allowed to allocate large, long-lived operators.  A caller
//! therefore needs a decision before the first pair loop starts, rather than
//! discovering an out-of-memory condition after a dense matrix or an FMM
//! interaction structure has already been allocated.  The estimates here are
//! deliberately conservative: an admission that passes is bounded by the
//! configured limit, while an overly conservative estimate is reported as a
//! typed rejection instead of risking an unbounded allocation.

use thiserror::Error;

use crate::core::bem_solver::AssemblyMethod;

const COMPLEX64_BYTES: usize = std::mem::size_of::<num_complex::Complex64>();
const POINTER_BYTES: usize = std::mem::size_of::<usize>();
/// Default H-matrix rank cap used when the caller sets no explicit cap.
///
/// Admission charges compressed operators at this capped rank instead of
/// the previous `n/4` default, whose `n * rank` workspace grew
/// quadratically and refused otherwise feasible compressed problems.
pub const DEFAULT_MAX_RANK: usize = 64;

/// Resource limits for one high-level BEM work item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BemResourceBudget {
    /// Maximum host-resident bytes reserved by the operator, right-hand sides,
    /// and assembly scratch space.
    pub max_host_bytes: usize,
    /// Maximum device-resident bytes reserved by an accelerator path.  The
    /// current high-level CPU path requests zero device bytes, but keeping the
    /// field here makes device admission explicit for future GPU dispatch.
    pub max_device_bytes: usize,
    /// Maximum number of simultaneous frequency/RHS work items admitted by a
    /// caller using this budget.
    pub max_active_work: usize,
}

impl BemResourceBudget {
    /// Construct a budget in bytes and simultaneous work items.
    pub const fn new(
        max_host_bytes: usize,
        max_device_bytes: usize,
        max_active_work: usize,
    ) -> Self {
        Self {
            max_host_bytes,
            max_device_bytes,
            max_active_work,
        }
    }

    /// Validate that the budget can admit at least one work item.
    pub fn validate(self) -> Result<(), BemAdmissionError> {
        if self.max_host_bytes == 0 {
            return Err(BemAdmissionError::InvalidBudget {
                resource: BemResource::HostMemory,
                limit: self.max_host_bytes,
            });
        }
        if self.max_active_work == 0 {
            return Err(BemAdmissionError::InvalidBudget {
                resource: BemResource::ActiveWork,
                limit: self.max_active_work,
            });
        }
        Ok(())
    }
}

/// A workload to be admitted before assembly starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BemResourceRequest {
    /// Assembly/operator family used for the estimate.
    pub method: AssemblyMethod,
    /// Number of active unknowns in the operator.
    pub active_dofs: usize,
    /// Number of right-hand sides retained concurrently.
    pub rhs_count: usize,
    /// Number of frequency work items retained concurrently.
    pub frequency_count: usize,
    /// Optional H-matrix rank cap.
    pub max_rank: usize,
    /// Device bytes needed by a backend-specific assembly path.
    pub device_bytes: usize,
}

impl BemResourceRequest {
    /// Construct a request with no device allocation and automatic H-matrix rank.
    pub const fn new(
        method: AssemblyMethod,
        active_dofs: usize,
        rhs_count: usize,
        frequency_count: usize,
    ) -> Self {
        Self {
            method,
            active_dofs,
            rhs_count,
            frequency_count,
            max_rank: 0,
            device_bytes: 0,
        }
    }

    /// Set the H-matrix rank cap used by the request.
    pub const fn with_hmatrix_rank(mut self, max_rank: usize) -> Self {
        self.max_rank = max_rank;
        self
    }

    /// Set device-resident bytes required by an accelerator path.
    pub const fn with_device_bytes(mut self, device_bytes: usize) -> Self {
        self.device_bytes = device_bytes;
        self
    }

    /// Return a conservative peak allocation estimate.
    pub fn estimate(self) -> Result<BemResourceEstimate, BemAdmissionError> {
        if self.active_dofs == 0 {
            return Err(BemAdmissionError::InvalidRequest {
                reason: "active_dofs must be greater than zero",
            });
        }
        if self.rhs_count == 0 {
            return Err(BemAdmissionError::InvalidRequest {
                reason: "rhs_count must be greater than zero",
            });
        }
        if self.frequency_count == 0 {
            return Err(BemAdmissionError::InvalidRequest {
                reason: "frequency_count must be greater than zero",
            });
        }

        let n = self.active_dofs;
        // Staged operator bound: only the dense method is charged the
        // full n^2 operator. Compressed/FMM methods are charged their
        // capped-rank factor storage (computed below once the rank cap
        // is known), so a feasible compressed problem is not refused by
        // a dense bound it will never allocate.
        let dense_operator_bytes = if matches!(self.method, AssemblyMethod::Tbem) {
            checked_product(
                &[n, n, COMPLEX64_BYTES],
                BemResource::HostMemory,
                "dense operator size",
            )?
        } else {
            0
        };
        let rhs_bytes = checked_product(
            &[n, self.rhs_count, COMPLEX64_BYTES],
            BemResource::HostMemory,
            "right-hand-side size",
        )?;
        let frequency_rhs = checked_product(
            &[self.frequency_count, self.rhs_count],
            BemResource::ActiveWork,
            "frequency/RHS work count",
        )?;

        // Linear scratch covers cluster metadata, expansion vectors, and
        // Krylov workspaces. The actual H-matrix memory_bytes value is
        // checked again after construction where available; admission
        // here is the staged pre-construction bound.
        let scratch_per_dof = match self.method {
            AssemblyMethod::Tbem => 4 * COMPLEX64_BYTES + POINTER_BYTES,
            AssemblyMethod::Hmatrix => 16 * COMPLEX64_BYTES + 4 * POINTER_BYTES,
            AssemblyMethod::Slfmm => 32 * COMPLEX64_BYTES + 8 * POINTER_BYTES,
            AssemblyMethod::Mlfmm | AssemblyMethod::DirectionalMlfmm => {
                64 * COMPLEX64_BYTES + 12 * POINTER_BYTES
            }
            AssemblyMethod::Auto => 16 * COMPLEX64_BYTES + 4 * POINTER_BYTES,
        };
        let base_scratch_bytes = checked_product(
            &[n, scratch_per_dof],
            BemResource::HostMemory,
            "assembly scratch size",
        )?;
        let rank = if self.max_rank == 0 {
            DEFAULT_MAX_RANK
        } else {
            self.max_rank
        }
        .clamp(1, n);
        let compressed_operator_bytes = if matches!(self.method, AssemblyMethod::Tbem) {
            dense_operator_bytes
        } else {
            checked_sum(
                &[
                    checked_product(
                        &[n, rank, 2 * COMPLEX64_BYTES],
                        BemResource::HostMemory,
                        "compressed operator factor size",
                    )?,
                    checked_product(
                        &[n, COMPLEX64_BYTES],
                        BemResource::HostMemory,
                        "compressed operator diagonal size",
                    )?,
                ],
                BemResource::HostMemory,
                "compressed operator size",
            )?
        };
        let rank_bytes = if matches!(self.method, AssemblyMethod::Hmatrix) {
            checked_product(
                &[n, rank, 2 * COMPLEX64_BYTES],
                BemResource::HostMemory,
                "H-matrix rank workspace",
            )?
        } else {
            0
        };
        let scratch_bytes = checked_sum(
            &[base_scratch_bytes, rank_bytes],
            BemResource::HostMemory,
            "total assembly scratch size",
        )?;
        let per_work_rhs = checked_product(
            &[rhs_bytes, self.frequency_count],
            BemResource::HostMemory,
            "frequency right-hand-side size",
        )?;
        let host_bytes = checked_sum(
            &[compressed_operator_bytes, per_work_rhs, scratch_bytes],
            BemResource::HostMemory,
            "peak host allocation",
        )?;

        Ok(BemResourceEstimate {
            host_bytes,
            device_bytes: self.device_bytes,
            active_work: frequency_rhs,
            operator_bytes: compressed_operator_bytes,
            rhs_bytes: per_work_rhs,
            scratch_bytes,
        })
    }
}

/// Resources reserved by a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BemResourceEstimate {
    /// Conservative peak host-resident allocation.
    pub host_bytes: usize,
    /// Device-resident allocation requested by the caller.
    pub device_bytes: usize,
    /// Concurrent frequency/RHS work items.
    pub active_work: usize,
    /// Conservative operator allocation.
    pub operator_bytes: usize,
    /// Right-hand-side allocation, including concurrent frequencies.
    pub rhs_bytes: usize,
    /// Assembly and solver scratch allocation.
    pub scratch_bytes: usize,
}

/// Resource classes used in typed admission errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BemResource {
    /// Host RAM.
    HostMemory,
    /// Accelerator/device memory.
    DeviceMemory,
    /// Concurrent frequency/RHS work slots.
    ActiveWork,
}

impl std::fmt::Display for BemResource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::HostMemory => "host memory",
            Self::DeviceMemory => "device memory",
            Self::ActiveWork => "active work",
        })
    }
}

/// Failure before a BEM operator is allocated.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BemAdmissionError {
    /// A zero budget cannot admit work.
    #[error("invalid {resource} budget: limit is zero")]
    InvalidBudget {
        /// Resource with the invalid limit.
        resource: BemResource,
        /// Configured limit.
        limit: usize,
    },
    /// The request contains a zero-sized workload dimension.
    #[error("invalid BEM resource request: {reason}")]
    InvalidRequest {
        /// Explanation of the invalid dimension.
        reason: &'static str,
    },
    /// Checked arithmetic overflowed while estimating a resource.
    #[error("{resource} estimate overflowed while computing {operation}")]
    ArithmeticOverflow {
        /// Resource whose estimate overflowed.
        resource: BemResource,
        /// Checked operation being evaluated.
        operation: &'static str,
    },
    /// The estimate exceeds a configured budget.
    #[error("BEM {resource} admission rejected: requested {requested} bytes/work, limit {limit}")]
    LimitExceeded {
        /// Resource over the configured limit.
        resource: BemResource,
        /// Conservative requested amount.
        requested: usize,
        /// Configured resource limit.
        limit: usize,
    },
}

impl BemResourceBudget {
    /// Validate a request and return its estimate if it fits the budget.
    pub fn admit(
        self,
        request: BemResourceRequest,
    ) -> Result<BemResourceEstimate, BemAdmissionError> {
        self.validate()?;
        let estimate = request.estimate()?;
        if estimate.host_bytes > self.max_host_bytes {
            return Err(BemAdmissionError::LimitExceeded {
                resource: BemResource::HostMemory,
                requested: estimate.host_bytes,
                limit: self.max_host_bytes,
            });
        }
        if estimate.device_bytes > self.max_device_bytes {
            return Err(BemAdmissionError::LimitExceeded {
                resource: BemResource::DeviceMemory,
                requested: estimate.device_bytes,
                limit: self.max_device_bytes,
            });
        }
        if estimate.active_work > self.max_active_work {
            return Err(BemAdmissionError::LimitExceeded {
                resource: BemResource::ActiveWork,
                requested: estimate.active_work,
                limit: self.max_active_work,
            });
        }
        Ok(estimate)
    }
}

fn checked_product(
    values: &[usize],
    resource: BemResource,
    operation: &'static str,
) -> Result<usize, BemAdmissionError> {
    values.iter().try_fold(1usize, |value, factor| {
        value
            .checked_mul(*factor)
            .ok_or(BemAdmissionError::ArithmeticOverflow {
                resource,
                operation,
            })
    })
}

fn checked_sum(
    values: &[usize],
    resource: BemResource,
    operation: &'static str,
) -> Result<usize, BemAdmissionError> {
    values.iter().try_fold(0usize, |value, addend| {
        value
            .checked_add(*addend)
            .ok_or(BemAdmissionError::ArithmeticOverflow {
                resource,
                operation,
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_estimate_is_checked_before_allocation() {
        let request = BemResourceRequest::new(AssemblyMethod::Tbem, 10, 2, 3);
        let estimate = request.estimate().expect("valid estimate");
        assert_eq!(estimate.operator_bytes, 10 * 10 * COMPLEX64_BYTES);
        assert_eq!(estimate.active_work, 6);
        BemResourceBudget::new(estimate.host_bytes, 0, 6)
            .admit(request)
            .expect("estimate fits exact host/work budget");
    }

    #[test]
    fn over_budget_and_overflow_are_typed() {
        let request = BemResourceRequest::new(AssemblyMethod::Hmatrix, 100, 1, 1);
        assert!(matches!(
            BemResourceBudget::new(1, 0, 1).admit(request),
            Err(BemAdmissionError::LimitExceeded {
                resource: BemResource::HostMemory,
                ..
            })
        ));
        assert!(matches!(
            BemResourceRequest::new(AssemblyMethod::Tbem, usize::MAX, 2, 1).estimate(),
            Err(BemAdmissionError::ArithmeticOverflow { .. })
        ));
    }

    #[test]
    fn compressed_large_problem_is_not_charged_dense_memory() {
        // At n = 100k the old bound charged 16*n^2 = 160 GB for every
        // method, refusing compressed problems that fit in a few GB.
        let request = BemResourceRequest::new(AssemblyMethod::Hmatrix, 100_000, 1, 1);
        let estimate = request.estimate().expect("valid estimate");
        assert!(
            estimate.operator_bytes < 1_000_000_000,
            "compressed operator estimate {} bytes",
            estimate.operator_bytes
        );
        BemResourceBudget::new(8 * 1024 * 1024 * 1024, 0, 1)
            .admit(request)
            .expect("compressed problem fits an 8 GiB budget");
    }

    #[test]
    fn dense_large_problem_is_still_refused() {
        let request = BemResourceRequest::new(AssemblyMethod::Tbem, 100_000, 1, 1);
        assert!(matches!(
            BemResourceBudget::new(8 * 1024 * 1024 * 1024, 0, 1).admit(request),
            Err(BemAdmissionError::LimitExceeded {
                resource: BemResource::HostMemory,
                ..
            })
        ));
    }

    #[test]
    fn zero_work_and_zero_budget_do_not_pass() {
        assert!(matches!(
            BemResourceBudget::new(1, 0, 1).admit(BemResourceRequest::new(
                AssemblyMethod::Tbem,
                0,
                1,
                1,
            )),
            Err(BemAdmissionError::InvalidRequest { .. })
        ));
        assert!(matches!(
            BemResourceBudget::new(0, 0, 1).validate(),
            Err(BemAdmissionError::InvalidBudget {
                resource: BemResource::HostMemory,
                ..
            })
        ));
    }
}
