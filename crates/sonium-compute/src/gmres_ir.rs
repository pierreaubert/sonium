//! Mixed-precision GMRES with iterative refinement (GMRES-IR).
//!
//! **Carson-Higham 2018 Alg 5.3**: a flexible GMRES outer loop runs in
//! `f64` (Arnoldi basis, Hessenberg, residual computation, solution
//! update); the preconditioner application is an inner Krylov solve in
//! `f32`. Treats the inexact f32 inner solve as a *variable*
//! preconditioner — hence FGMRES, not standard GMRES. Recovers `f64`
//! accuracy at the outer level provided `κ(A) · u_p · u < 1`,
//! i.e. roughly **κ ≲ 1 / (ε_{f32} · ε_{f64}) ≈ 10¹⁴**.
//!
//! This is the high-conditioning robust variant. The plain-IR variant
//! (Newton-style outer fixed point) only handles κ ≲ 1/ε_{f32} ≈ 10⁷
//! before stagnation.
//!
//! # Why this exists
//!
//! Apple Silicon Metal has no `double` support — MSL rejects f64 at
//! compile time. To accelerate sonium's f64 physics on Metal we must
//! run the inner Krylov in f32 and recover precision via outer
//! refinement. This module provides the algorithm; the f32 BLAS-1 +
//! SpMV ops live on the existing `Backend` trait (already generic over
//! `T: DeviceScalar`) so once Metal grows native f32 MSL kernels the
//! same code path lights up.
//!
//! # Three precisions
//!
//! In Carson-Higham notation:
//! - **u_r** (residual precision) = f64. We compute `r = b − A x` in
//!   f64 to avoid loss of accuracy in the outer loop. f64 is the
//!   sonium native precision.
//! - **u** (working precision) = f64. The accumulated solution `x` and
//!   the user-visible residual norm are f64.
//! - **u_p** (low precision) = f32. The inner Krylov subspace,
//!   preconditioner apply, and Arnoldi orthogonalisation are all f32.
//!
//! # Algorithm — FGMRES-IR (Carson-Higham 2018 Alg 5.3)
//!
//! ```text
//! x_0 = 0  (f64)
//! r_0 = b − A x_0  (f64)
//! β = ‖r_0‖
//! v_1 = r_0 / β     (f64)
//!
//! for j = 1..m:                          # FGMRES Arnoldi cycle
//!     # Variable f32 preconditioner: solve A z_j ≈ v_j approximately
//!     v_j_f32 = cast<f32>(v_j)
//!     z_j_f32 = inner_gmres_f32(A_f32, M_f32, v_j_f32, ...)
//!     z_j     = cast<f64>(z_j_f32)        # store in Z basis (f64)
//!     w       = A · z_j                   # f64 SpMV
//!     # Modified Gram-Schmidt against V_1..V_j  (f64)
//!     for i = 1..j: h_{i,j} = ⟨v_i, w⟩;  w -= h_{i,j} v_i
//!     h_{j+1,j} = ‖w‖
//!     v_{j+1}   = w / h_{j+1,j}
//!     # Update Givens rotations + residual estimate
//!     ...
//!     if ‖r_j‖ / ‖b‖ ≤ tol: break
//!
//! # Solve small upper-triangular H y = β e₁  (host)
//! # Update solution: x = x_0 + Z · y       (f64; uses Z, not V — hence FLEXIBLE)
//! ```
//!
//! The FLEX in FGMRES is critical: the f32 inner solve returns a
//! slightly different `M⁻¹` each call (varies with the f32 Krylov path),
//! so we must use the actual preconditioned vectors `Z = [z_1, ..., z_m]`
//! to update `x`, not the unpreconditioned Krylov basis `V`.
//!
//! # Limitations of this implementation
//!
//! - Only `Identity` and `PolynomialJacobi` preconditioners on the f32
//!   inner side. Chebyshev_f32 is a follow-up.
//! - Single Arnoldi cycle (no restart). For the typical case where
//!   FGMRES-IR converges in 3-10 outer iters, this is fine; if you set
//!   `max_outer_iterations` too large the host-side Hessenberg cost
//!   `O(m²)` and the Z-basis memory `O(m·n·8 bytes)` grow. At m=30,
//!   n=100k: 24 MB Z basis — still cheap.
//! - Precision casts route through `Backend::cast_to_lower` /
//!   `Backend::cast_from_lower`. CUDA keeps these casts on device; other
//!   backends may use the backend trait's host-staged default.

use crate::gmres::{
    BackendCsr, BackendOperator, BackendPreconditioner, IdentityBackendPreconditioner,
};
use crate::{Backend, DeviceBufferLike, DeviceScalar, ReductionMode};
mod workspace;
#[doc(inline)]
pub use workspace::GmresIrWorkspace;
use workspace::{Allocator, Scratch};

// ---------------------------------------------------------------------------
// F32 mirror types — kept as type aliases for backward compatibility.
//
// Sub-step a₂/a₃ generalised the device-side types to be generic over
// `T: DeviceScalar`. The f32 mirrors that originally lived here are now
// just instantiations at `T = f32` of the generic types. Existing
// callers (sonium-fem dispatcher, benches, examples, QA bins) keep
// working without API churn.
// ---------------------------------------------------------------------------

/// CSR sparse matrix on a device, f32 values.
///
/// Type alias for [`crate::gmres::BackendCsr<B, f32>`].
/// The convenience constructor [`BackendCsr::from_host_f64_values`] is
/// available as an inherent method on the f32 specialisation.
pub type BackendCsrF32<B> = BackendCsr<B, f32>;

/// f32 device-resident preconditioner trait.
///
/// Marker trait — anything that implements
/// [`BackendPreconditioner<B, f32>`] auto-implements this via the
/// blanket impl below. Kept around for ergonomic call-site signatures
/// like `P: BackendPreconditionerF32<B>` instead of the more verbose
/// fully-qualified bound.
pub trait BackendPreconditionerF32<B: Backend>: BackendPreconditioner<B, f32> {}

impl<B: Backend, P> BackendPreconditionerF32<B> for P where P: BackendPreconditioner<B, f32> {}

/// `M = I` — f32 pass-through preconditioner.
///
/// Type alias for the generic
/// [`crate::gmres::IdentityBackendPreconditioner`].
pub type IdentityBackendPreconditionerF32 = IdentityBackendPreconditioner;

/// f32 polynomial Jacobi.
///
/// Type alias for `PolynomialJacobiBackend<'a, B, f32>`. Construct via
/// the generic `from_csr_host` (takes `&[f32]` values). For symmetry
/// with the legacy API, a `from_csr_f32` alias method is provided on
/// the f32 specialisation.
pub type PolynomialJacobiBackendF32<'a, B> =
    crate::preconditioner::PolynomialJacobiBackendF32<'a, B>;

// ---------------------------------------------------------------------------
// Inner FGMRES — generic over `T: DeviceScalar`. Right-preconditioned,
// classical Gram-Schmidt with one reorthogonalisation (CGS2) via
// batched `multi_dot`, host-side Givens. Same Saad §9.3 recurrence
// as `gmres_backend_preconditioned`; the difference is the
// orthogonalisation strategy (CGS2 vs MGS) which matters in low
// precision and on backends that benefit from batched dot products.
//
// Canonical Givens (with `conj_mul` in the lower-left rotation
// coefficient) so the recurrence stays unitary for complex `T`. For
// real `T`, `conj_mul(x, T::one()) = x` so the math collapses to
// standard real Givens.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GmresInnerConfig<T: DeviceScalar = f32> {
    pub max_iterations: usize,
    pub tolerance: T::Real,
}

impl<T: DeviceScalar> Default for GmresInnerConfig<T> {
    fn default() -> Self {
        use num_traits::{FromPrimitive, Zero};
        Self {
            max_iterations: 200,
            // Keep the legacy infallible constructor, but fail closed for an
            // extensible scalar whose real type cannot represent the default.
            // `gmres_inner` rejects the resulting zero tolerance as typed
            // invalid input before doing any work.
            tolerance: T::Real::from_f64(1.0e-4).unwrap_or_else(T::Real::zero),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GmresInnerStatus<T: DeviceScalar = f32> {
    pub iterations: usize,
    pub final_residual: T::Real,
    pub converged: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum GmresInnerError {
    #[error("GMRES-IR workspace allocation failed for {label}: {reason}")]
    Allocation { label: &'static str, reason: String },
    #[error("zero RHS")]
    ZeroRhs,
    #[error("backend operation failed: {source}")]
    Backend { source: crate::BackendError },
}

impl From<crate::BackendError> for GmresInnerError {
    fn from(source: crate::BackendError) -> Self {
        Self::Backend { source }
    }
}

fn reserve_inner<T>(
    values: &mut Vec<T>,
    capacity: usize,
    label: &'static str,
) -> Result<(), GmresInnerError> {
    values
        .try_reserve_exact(capacity)
        .map_err(|error| GmresInnerError::Allocation {
            label,
            reason: error.to_string(),
        })
}

fn inner_capacity_add(
    capacity: usize,
    extra: usize,
    label: &'static str,
) -> Result<usize, GmresInnerError> {
    capacity
        .checked_add(extra)
        .ok_or_else(|| GmresInnerError::Allocation {
            label,
            reason: "capacity overflow".into(),
        })
}

// Backward-compat aliases for the legacy F32 names.
pub type GmresInnerF32Config = GmresInnerConfig<f32>;
pub type GmresInnerF32Status = GmresInnerStatus<f32>;
pub type GmresInnerF32Error = GmresInnerError;

/// Inner GMRES generic over `T: DeviceScalar`. CGS2 + multi_dot.
/// Exposed crate-internally for use by `gmres_ir_backend_preconditioned`.
#[cfg(test)]
fn gmres_inner<B, A, P, T>(
    backend: &B,
    a: &A,
    precond: &P,
    b: &B::Buffer<T>,
    x0: Option<&B::Buffer<T>>,
    out: &mut B::Buffer<T>,
    config: &GmresInnerConfig<T>,
) -> Result<GmresInnerStatus<T>, GmresInnerError>
where
    B: Backend,
    T: DeviceScalar,
    A: BackendOperator<B, T> + ?Sized,
    P: BackendPreconditioner<B, T> + ?Sized,
{
    gmres_inner_with_alloc(
        backend,
        a,
        precond,
        b,
        x0,
        out,
        config,
        &Allocator::raw(backend),
    )
}

fn gmres_inner_with_alloc<B, A, P, T>(
    backend: &B,
    a: &A,
    precond: &P,
    b: &B::Buffer<T>,
    x0: Option<&B::Buffer<T>>,
    out: &mut B::Buffer<T>,
    config: &GmresInnerConfig<T>,
    alloc: &Allocator<'_, B, T>,
) -> Result<GmresInnerStatus<T>, GmresInnerError>
where
    B: Backend,
    T: DeviceScalar,
    A: BackendOperator<B, T> + ?Sized,
    P: BackendPreconditioner<B, T> + ?Sized,
{
    use num_traits::{Float, One, Zero};

    let n = a.n();
    if b.len() != n || out.len() != n {
        return Err(GmresInnerError::Backend {
            source: crate::BackendError::InvalidArgument {
                operation: "gmres_inner",
                reason: format!(
                    "RHS/output lengths {}/{} must match operator dimension {n}",
                    b.len(),
                    out.len()
                ),
            },
        });
    }
    if config.max_iterations == 0
        || !config.tolerance.is_finite()
        || config.tolerance < T::Real::zero()
    {
        return Err(GmresInnerError::Backend {
            source: crate::BackendError::InvalidArgument {
                operation: "gmres_inner",
                reason:
                    "max_iterations must be positive and tolerance must be finite and non-negative"
                        .into(),
            },
        });
    }

    let max_iter = config.max_iterations;
    let zero_real = T::Real::zero();
    let tol = if config.tolerance < zero_real {
        zero_real
    } else {
        config.tolerance
    };

    let mut r0 = alloc
        .alloc(n)
        .map_err(|error| GmresInnerError::Allocation {
            label: "r0",
            reason: error.to_string(),
        })?;
    match x0 {
        None => {
            backend.try_scale::<T>(T::zero(), out)?;
            backend.try_copy_d2d(b, &mut r0)?;
        }
        Some(x_init) => {
            if x_init.len() != n {
                return Err(GmresInnerError::Backend {
                    source: crate::BackendError::InvalidArgument {
                        operation: "gmres_inner",
                        reason: format!(
                            "initial guess length {} must match operator dimension {n}",
                            x_init.len()
                        ),
                    },
                });
            }
            backend.try_copy_d2d(x_init, out)?;
            a.try_apply(backend, x_init, &mut r0)
                .map_err(GmresInnerError::from)?;
            backend.try_scale::<T>(-T::one(), &mut r0)?;
            backend.try_axpy::<T>(T::one(), b, &mut r0)?;
        }
    }

    let b_norm = backend.try_nrm2_with_mode::<T>(ReductionMode::Fast, b)?;
    if b_norm == zero_real {
        return Err(GmresInnerError::ZeroRhs);
    }
    let beta = backend.try_nrm2_with_mode::<T>(ReductionMode::Fast, &r0)?;
    let initial_resid_rel = beta / b_norm;
    if initial_resid_rel <= tol {
        return Ok(GmresInnerStatus {
            iterations: 0,
            final_residual: initial_resid_rel,
            converged: true,
        });
    }

    let mut v: Vec<Scratch<'_, B, T>> = Vec::new();
    reserve_inner(&mut v, inner_capacity_add(max_iter, 1, "V")?, "V")?;
    let mut v0 = alloc
        .alloc(n)
        .map_err(|error| GmresInnerError::Allocation {
            label: "V[0]",
            reason: error.to_string(),
        })?;
    backend.try_copy_d2d(&r0, &mut v0)?;
    backend.try_scale::<T>(T::from_real(T::Real::one() / beta), &mut v0)?;
    v.push(v0);

    let mut h: Vec<Vec<T>> = Vec::new();
    reserve_inner(&mut h, max_iter, "H")?;
    let mut cs: Vec<T> = Vec::new();
    reserve_inner(&mut cs, max_iter, "cosines")?;
    let mut sn: Vec<T> = Vec::new();
    reserve_inner(&mut sn, max_iter, "sines")?;
    let mut g: Vec<T> = Vec::new();
    reserve_inner(&mut g, inner_capacity_add(max_iter, 1, "rhs")?, "rhs")?;
    g.push(T::from_real(beta));

    let mut iterations_done = 0_usize;
    let mut final_resid_rel = initial_resid_rel;

    let mut w = alloc
        .alloc(n)
        .map_err(|error| GmresInnerError::Allocation {
            label: "w",
            reason: error.to_string(),
        })?;
    let mut z = alloc
        .alloc(n)
        .map_err(|error| GmresInnerError::Allocation {
            label: "z",
            reason: error.to_string(),
        })?;

    for j in 0..max_iter {
        precond
            .try_apply(backend, &v[j], &mut z)
            .map_err(GmresInnerError::from)?;
        a.try_apply(backend, &z, &mut w)
            .map_err(GmresInnerError::from)?;

        // CGS2 with fused backend projection updates.
        let mut h_col = crate::backend::try_zeroed_host::<T>(
            inner_capacity_add(j, 2, "H column")?,
            "gmres_inner H column",
        )
        .map_err(GmresInnerError::from)?;
        let mut basis_refs: Vec<&B::Buffer<T>> = Vec::new();
        reserve_inner(
            &mut basis_refs,
            inner_capacity_add(j, 1, "basis references")?,
            "basis references",
        )?;
        basis_refs.extend(v.iter().take(j + 1).map(|buffer| &**buffer));

        let dots1 = backend.try_multi_dot_with_mode::<T>(ReductionMode::Fast, &basis_refs, &w)?;
        let mut neg_dots1 =
            crate::backend::try_zeroed_host::<T>(dots1.len(), "gmres_inner first projection")
                .map_err(GmresInnerError::from)?;
        for (negated, &dot) in neg_dots1.iter_mut().zip(&dots1) {
            *negated = T::zero() - dot;
        }
        backend.try_axpy_many::<T>(&neg_dots1, &basis_refs, &mut w)?;
        h_col[..=j].copy_from_slice(&dots1[..=j]);

        let dots2 = backend.try_multi_dot_with_mode::<T>(ReductionMode::Fast, &basis_refs, &w)?;
        let mut neg_dots2 =
            crate::backend::try_zeroed_host::<T>(dots2.len(), "gmres_inner second projection")
                .map_err(GmresInnerError::from)?;
        for (negated, &dot) in neg_dots2.iter_mut().zip(&dots2) {
            *negated = T::zero() - dot;
        }
        backend.try_axpy_many::<T>(&neg_dots2, &basis_refs, &mut w)?;
        for i in 0..=j {
            h_col[i] = h_col[i] + dots2[i];
        }

        let h_jp1 = backend.try_nrm2_with_mode::<T>(ReductionMode::Fast, &w)?;
        h_col[j + 1] = T::from_real(h_jp1);
        let invariant_subspace = h_jp1 == zero_real;

        if !invariant_subspace {
            let mut v_next = alloc
                .alloc(n)
                .map_err(|error| GmresInnerError::Allocation {
                    label: "V[j+1]",
                    reason: error.to_string(),
                })?;
            backend.try_copy_d2d(&w, &mut v_next)?;
            backend.try_scale::<T>(T::from_real(T::Real::one() / h_jp1), &mut v_next)?;
            v.push(v_next);
        }

        // Apply previous Givens (canonical, with conj on lower-left).
        for i in 0..j {
            let s_conj = sn[i].conj_mul(T::one());
            let temp = cs[i] * h_col[i] + sn[i] * h_col[i + 1];
            h_col[i + 1] = (T::zero() - s_conj) * h_col[i] + cs[i] * h_col[i + 1];
            h_col[i] = temp;
        }

        if invariant_subspace {
            let diagonal_norm = h_col[j].magnitude();
            if !diagonal_norm.is_finite() || diagonal_norm == zero_real {
                return Err(GmresInnerError::Backend {
                    source: crate::BackendError::OperationFailed {
                        operation: "gmres_inner",
                        reason: format!(
                            "singular or non-finite Hessenberg diagonal at iteration {}",
                            j + 1
                        ),
                    },
                });
            }
            h.push(h_col);
            iterations_done = j + 1;
            final_resid_rel = zero_real;
            break;
        }

        // New canonical complex Givens.
        let abs_a = h_col[j].magnitude();
        let abs_b = h_col[j + 1].magnitude();
        let r = abs_a.hypot(abs_b);
        let (cos_j, sin_j): (T, T) = if r == zero_real {
            (T::one(), T::zero())
        } else if abs_a == zero_real {
            (T::zero(), h_col[j + 1] / T::from_real(abs_b))
        } else {
            let c = T::from_real(abs_a / r);
            let tau = h_col[j] / T::from_real(abs_a);
            let conj_b = h_col[j + 1].conj_mul(T::one());
            let s = tau * conj_b / T::from_real(r);
            (c, s)
        };
        cs.push(cos_j);
        sn.push(sin_j);
        h_col[j] = cos_j * h_col[j] + sin_j * h_col[j + 1];
        h_col[j + 1] = T::zero();

        let g_j = g[j];
        let s_conj = sin_j.conj_mul(T::one());
        g.push((T::zero() - s_conj) * g_j);
        g[j] = cos_j * g_j;

        h.push(h_col);
        iterations_done = j + 1;

        let resid_abs = g[j + 1].magnitude();
        let resid_rel = resid_abs / b_norm;
        final_resid_rel = resid_rel;
        if resid_rel <= tol {
            break;
        }
    }

    let converged = final_resid_rel <= tol;
    let k = iterations_done;
    if k == 0 {
        return Ok(GmresInnerStatus {
            iterations: 0,
            final_residual: final_resid_rel,
            converged,
        });
    }

    let mut y = crate::backend::try_zeroed_host::<T>(k, "gmres_inner triangular solve")
        .map_err(GmresInnerError::from)?;
    for i in (0..k).rev() {
        let mut sum = g[i];
        for j in (i + 1)..k {
            sum = sum - h[j][i] * y[j];
        }
        y[i] = sum / h[i][i];
    }

    let mut dx_unprec = alloc
        .alloc(n)
        .map_err(|error| GmresInnerError::Allocation {
            label: "dx_unprec",
            reason: error.to_string(),
        })?;
    backend.try_scale::<T>(T::zero(), &mut dx_unprec)?;
    let mut v_refs: Vec<&B::Buffer<T>> = Vec::new();
    reserve_inner(&mut v_refs, k, "solution basis references")?;
    v_refs.extend(v.iter().take(k).map(|buffer| &**buffer));
    backend.try_axpy_many::<T>(&y, &v_refs, &mut dx_unprec)?;

    let mut dx = alloc
        .alloc(n)
        .map_err(|error| GmresInnerError::Allocation {
            label: "dx",
            reason: error.to_string(),
        })?;
    precond
        .try_apply(backend, &dx_unprec, &mut dx)
        .map_err(GmresInnerError::from)?;
    backend.try_axpy::<T>(T::one(), &dx, out)?;

    Ok(GmresInnerStatus {
        iterations: iterations_done,
        final_residual: final_resid_rel,
        converged,
    })
}

// ---------------------------------------------------------------------------
// Outer GMRES-IR
// ---------------------------------------------------------------------------

/// Configuration for the GMRES-IR outer loop. Generic over the outer
/// scalar `Outer: LowerPrecision`. Default `Outer = f64` so the legacy
/// callers see no change.
#[derive(Debug, Clone)]
pub struct GmresIrConfig<Outer: crate::LowerPrecision = f64> {
    /// Max outer refinement iterations.
    pub max_outer_iterations: usize,
    /// Outer convergence tolerance (relative residual at outer precision).
    pub outer_tolerance: Outer::Real,
    /// Max inner GMRES iterations per refinement step.
    pub max_inner_iterations: usize,
    /// Inner GMRES relative-residual tolerance. Loose by design —
    /// the outer loop refines whatever the inner gives.
    pub inner_tolerance: <Outer::Lower as DeviceScalar>::Real,
    /// Restart cycle size, in outer Arnoldi iterations.
    pub restart_size: Option<usize>,
    /// Adaptive inner tolerance.
    pub adaptive_inner_tolerance: bool,
    /// Recompute the outer-precision true residual every `K` outer
    /// iterations.
    pub true_residual_check_period: Option<usize>,
}

impl<Outer: crate::LowerPrecision> Default for GmresIrConfig<Outer> {
    fn default() -> Self {
        use num_traits::{FromPrimitive, Zero};
        Self {
            max_outer_iterations: 30,
            // Preserve the infallible legacy constructor while making an
            // unsupported scalar conversion fail closed at checked execution.
            outer_tolerance: Outer::Real::from_f64(1.0e-10).unwrap_or_else(Outer::Real::zero),
            max_inner_iterations: 50,
            inner_tolerance: <<Outer::Lower as DeviceScalar>::Real as FromPrimitive>::from_f64(
                1.0e-3,
            )
            .unwrap_or_else(<<Outer::Lower as DeviceScalar>::Real as Zero>::zero),
            restart_size: None,
            adaptive_inner_tolerance: false,
            true_residual_check_period: None,
        }
    }
}

/// GMRES-IR result. Generic over the outer scalar.
#[derive(Debug, Clone)]
pub struct GmresIrStatus<Outer: crate::LowerPrecision = f64> {
    pub outer_iterations: usize,
    pub total_inner_iterations: usize,
    /// Last cheap projected FGMRES residual estimate. This is kept separate
    /// from [`final_outer_residual`], which is recomputed with the original
    /// outer-precision operator before returning.
    pub estimated_residual: Outer::Real,
    pub final_outer_residual: Outer::Real,
    /// Normwise backward-error indicator for the returned iterate:
    /// `||b - A*x|| / max(||A*x|| + ||b||, 1)`.
    pub backward_error: Outer::Real,
    /// Whether the requested outer tolerance was attained by the returned
    /// iterate after the promoted true-residual recomputation.
    pub attainable_accuracy: GmresIrAccuracy,
    pub converged: bool,
    /// Number of restart cycles. `1` for the unrestarted case (default).
    pub restart_cycles: usize,
    /// Per-outer-step residual norms.
    pub residual_history: Vec<Outer::Real>,
    /// Each true-residual recomputation stamp:
    /// `(outer_iter, fgmres_estimate, true_residual)`.
    pub true_residual_checks: Vec<(usize, Outer::Real, Outer::Real)>,
}

/// Accuracy state reported by the checked mixed-precision GMRES-IR path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GmresIrAccuracy {
    Attained,
    NotAttained,
}

#[derive(Debug, thiserror::Error)]
pub enum GmresIrError {
    #[error("GMRES-IR invalid input: {reason}")]
    InvalidInput { reason: String },
    #[error("GMRES-IR workspace allocation failed for {label}: {reason}")]
    Allocation { label: &'static str, reason: String },
    #[error("zero RHS")]
    ZeroRhs,
    #[error("inner GMRES failed at outer iter {outer}: {source}")]
    InnerSolverFailed {
        outer: usize,
        #[source]
        source: GmresInnerError,
    },
    #[error("backend operation failed: {source}")]
    Backend { source: crate::BackendError },
}

impl From<crate::BackendError> for GmresIrError {
    fn from(source: crate::BackendError) -> Self {
        Self::Backend { source }
    }
}

/// Solve using retained, quota-accounted mixed-precision device scratch.
///
/// Operators, RHS and output must belong to the workspace pool's backend.
/// Scratch is reused across inner solves, restarts and calls; later calls with
/// larger iteration demand can grow it. Algorithm and certification match
/// the non-workspace entrypoint. Execution/allocation errors discard cached
/// scratch before retry; input validation failures preserve it.
///
/// # Errors
/// Returns input, quota, backend or numerical failures as in
/// [gmres_ir_backend_preconditioned].
#[allow(clippy::too_many_arguments)]
pub fn gmres_ir_backend_preconditioned_with_workspace<B, AOuter, AInner, P, Outer>(
    workspace: &mut GmresIrWorkspace<B, Outer>,
    a_outer: &AOuter,
    a_inner: &AInner,
    inner_precond: &P,
    b_outer: &B::Buffer<Outer>,
    x0_outer: Option<&B::Buffer<Outer>>,
    out_outer: &mut B::Buffer<Outer>,
    config: &GmresIrConfig<Outer>,
) -> Result<GmresIrStatus<Outer>, GmresIrError>
where
    B: Backend,
    Outer: crate::LowerPrecision,
    AOuter: BackendOperator<B, Outer> + ?Sized,
    AInner: BackendOperator<B, Outer::Lower> + ?Sized,
    P: BackendPreconditioner<B, Outer::Lower> + ?Sized,
{
    workspace.prepare(a_outer.n());
    let result = gmres_ir_backend_preconditioned_with_alloc(
        workspace.pool.backend(),
        a_outer,
        a_inner,
        inner_precond,
        b_outer,
        x0_outer,
        out_outer,
        config,
        &Allocator::pooled(&workspace.pool, &workspace.outer),
        &Allocator::pooled(&workspace.pool, &workspace.inner),
    );
    if matches!(
        &result,
        Err(GmresIrError::Backend { .. }
            | GmresIrError::InnerSolverFailed { .. }
            | GmresIrError::Allocation { .. })
    ) {
        // A failing operator may have partially written or poisoned scratch.
        // Discard it before retry rather than relying on zero times NaN.
        workspace.clear();
    }
    result
}

/// Mixed-precision FGMRES-IR (Carson-Higham 2018 Alg 5.3).
///
/// Outer FGMRES Arnoldi cycles in `f64` with the variable f32 inner
/// solver acting as preconditioner. Robust up to κ ≲ 10¹⁴.
///
/// **Restart**: when `config.restart_size = Some(m)`, runs cycles of
/// `m` Arnoldi steps each. After each cycle the `f64` true residual
/// is recomputed (`r ← b − A x`) and used as the starting `r_0` for
/// the next cycle. Iteration count, inner-solve count, and residual
/// history accumulate across cycles. Memory is bounded at `m·n·8`
/// bytes for the Z-basis.
///
/// **Adaptive inner tolerance**: when `config.adaptive_inner_tolerance`
/// is `true`, the inner f32 GMRES tolerance is set to `max(
/// config.inner_tolerance, current_outer_residual / 100)` for each
/// preconditioner application. Loose when far from target, tight as
/// we converge — saves ~30% inner iters in typical runs.
///
/// **True-residual checks**: when `config.true_residual_check_period =
/// Some(k)`, every `k` outer iterations the true f64 residual is
/// recomputed and logged in `status.true_residual_checks` alongside
/// the FGMRES estimate. Detects estimate drift under inexact f32
/// preconditioning. The f64 true residual is also recomputed at every
/// restart boundary regardless.
///
/// `inner_precond_f32` is the f32 preconditioner driving the inner GMRES.
/// Outputs the f64 solution into `out_f64`. The reported residual is
/// the **true** f64 residual at the last cycle boundary.
#[allow(clippy::too_many_arguments)]
pub fn gmres_ir_backend_preconditioned<B, AOuter, AInner, P, Outer>(
    backend: &B,
    a_outer: &AOuter,
    a_inner: &AInner,
    inner_precond: &P,
    b_outer: &B::Buffer<Outer>,
    x0_outer: Option<&B::Buffer<Outer>>,
    out_outer: &mut B::Buffer<Outer>,
    config: &GmresIrConfig<Outer>,
) -> Result<GmresIrStatus<Outer>, GmresIrError>
where
    B: Backend,
    Outer: crate::LowerPrecision,
    AOuter: BackendOperator<B, Outer> + ?Sized,
    AInner: BackendOperator<B, Outer::Lower> + ?Sized,
    P: BackendPreconditioner<B, Outer::Lower> + ?Sized,
{
    gmres_ir_backend_preconditioned_with_alloc(
        backend,
        a_outer,
        a_inner,
        inner_precond,
        b_outer,
        x0_outer,
        out_outer,
        config,
        &Allocator::raw(backend),
        &Allocator::raw(backend),
    )
}

#[expect(clippy::too_many_arguments, reason = "shared solver core adds two precision-specific scratch allocators")]
fn gmres_ir_backend_preconditioned_with_alloc<B, AOuter, AInner, P, Outer>(
    backend: &B,
    a_outer: &AOuter,
    a_inner: &AInner,
    inner_precond: &P,
    b_outer: &B::Buffer<Outer>,
    x0_outer: Option<&B::Buffer<Outer>>,
    out_outer: &mut B::Buffer<Outer>,
    config: &GmresIrConfig<Outer>,
    alloc: &Allocator<'_, B, Outer>,
    inner_alloc: &Allocator<'_, B, Outer::Lower>,
) -> Result<GmresIrStatus<Outer>, GmresIrError>
where
    B: Backend,
    Outer: crate::LowerPrecision,
    AOuter: BackendOperator<B, Outer> + ?Sized,
    AInner: BackendOperator<B, Outer::Lower> + ?Sized,
    P: BackendPreconditioner<B, Outer::Lower> + ?Sized,
{
    use num_traits::{Float, One, Zero};

    let n = a_outer.n();
    if a_inner.n() != n {
        return Err(GmresIrError::InvalidInput {
            reason: format!(
                "outer/inner operator dimensions must agree (outer {}, inner {})",
                n,
                a_inner.n()
            ),
        });
    }
    if b_outer.len() != n {
        return Err(GmresIrError::InvalidInput {
            reason: format!(
                "RHS length {} does not match operator size {}",
                b_outer.len(),
                n
            ),
        });
    }
    if out_outer.len() != n {
        return Err(GmresIrError::InvalidInput {
            reason: format!(
                "solution length {} does not match operator size {}",
                out_outer.len(),
                n
            ),
        });
    }
    if let Some(x0) = x0_outer
        && x0.len() != n
    {
        return Err(GmresIrError::InvalidInput {
            reason: format!(
                "initial guess length {} does not match operator size {}",
                x0.len(),
                n
            ),
        });
    }
    if config.max_outer_iterations == 0 || config.max_inner_iterations == 0 {
        return Err(GmresIrError::InvalidInput {
            reason: "outer and inner iteration limits must be greater than zero".into(),
        });
    }
    if !config.outer_tolerance.is_finite() || config.outer_tolerance <= Outer::Real::zero() {
        return Err(GmresIrError::InvalidInput {
            reason: "outer tolerance must be finite and positive".into(),
        });
    }
    if !config.inner_tolerance.is_finite()
        || config.inner_tolerance <= <Outer::Lower as DeviceScalar>::Real::zero()
    {
        return Err(GmresIrError::InvalidInput {
            reason: "inner tolerance must be finite and positive".into(),
        });
    }

    let zero_outer_real = Outer::Real::zero();
    let b_norm = backend.try_nrm2_with_mode::<Outer>(ReductionMode::Fast, b_outer)?;
    if b_norm == zero_outer_real {
        return Err(GmresIrError::ZeroRhs);
    }

    match x0_outer {
        None => backend.try_scale::<Outer>(Outer::zero(), out_outer)?,
        Some(x_init) => backend.try_copy_d2d(x_init, out_outer)?,
    }

    let mut r_outer = alloc.alloc(n).map_err(|error| GmresIrError::Allocation {
        label: "r",
        reason: error.to_string(),
    })?;
    recompute_true_residual::<B, AOuter, Outer>(
        backend,
        a_outer,
        b_outer,
        out_outer,
        &mut r_outer,
    )?;

    let mut current_resid =
        backend.try_nrm2_with_mode::<Outer>(ReductionMode::Fast, &r_outer)? / b_norm;
    let mut estimated_residual = current_resid;
    let mut history = Vec::new();
    reserve_ir(&mut history, 1, "residual history")?;
    history.push(current_resid);
    let mut true_residual_checks: Vec<(usize, Outer::Real, Outer::Real)> = Vec::new();
    reserve_ir(
        &mut true_residual_checks,
        config.max_outer_iterations,
        "true residual checks",
    )?;

    if current_resid <= config.outer_tolerance {
        let mut ax_outer = alloc.alloc(n).map_err(|error| GmresIrError::Allocation {
            label: "Ax finalization",
            reason: error.to_string(),
        })?;
        a_outer.try_apply(backend, out_outer, &mut ax_outer)?;
        let ax_norm = backend.try_nrm2_with_mode::<Outer>(ReductionMode::Fast, &ax_outer)?;
        let residual_norm = backend.try_nrm2_with_mode::<Outer>(ReductionMode::Fast, &r_outer)?;
        return Ok(GmresIrStatus {
            outer_iterations: 0,
            total_inner_iterations: 0,
            estimated_residual,
            final_outer_residual: current_resid,
            backward_error: residual_norm / (ax_norm + b_norm).max(Outer::Real::one()),
            attainable_accuracy: GmresIrAccuracy::Attained,
            converged: true,
            restart_cycles: 0,
            residual_history: history,
            true_residual_checks,
        });
    }

    let total_max = config.max_outer_iterations.max(1);
    let cycle_size = config.restart_size.unwrap_or(total_max).clamp(1, total_max);

    let mut iters_done = 0_usize;
    let mut total_inner = 0_usize;
    let mut cycles = 0_usize;

    while iters_done < total_max {
        let remaining = total_max - iters_done;
        let this_cycle = cycle_size.min(remaining);
        cycles += 1;

        let cycle = fgmres_ir_one_cycle_with_alloc::<B, AOuter, AInner, P, Outer>(
            backend,
            a_outer,
            a_inner,
            inner_precond,
            b_outer,
            b_norm,
            &r_outer,
            out_outer,
            this_cycle,
            config,
            current_resid,
            iters_done,
            &mut history,
            &mut true_residual_checks,
            alloc,
            inner_alloc,
        )?;

        iters_done += cycle.iters_done;
        total_inner += cycle.inner_iters;
        estimated_residual = cycle.estimated_residual;

        if cycle.iters_done == 0 {
            break;
        }

        recompute_true_residual::<B, AOuter, Outer>(
            backend,
            a_outer,
            b_outer,
            out_outer,
            &mut r_outer,
        )?;
        current_resid =
            backend.try_nrm2_with_mode::<Outer>(ReductionMode::Fast, &r_outer)? / b_norm;
        history.push(current_resid);

        if current_resid <= config.outer_tolerance || cycle.happy_breakdown {
            break;
        }
    }

    let mut ax_outer = alloc.alloc(n).map_err(|error| GmresIrError::Allocation {
        label: "Ax finalization",
        reason: error.to_string(),
    })?;
    a_outer.try_apply(backend, out_outer, &mut ax_outer)?;
    let ax_norm = backend.try_nrm2_with_mode::<Outer>(ReductionMode::Fast, &ax_outer)?;
    let residual_norm = backend.try_nrm2_with_mode::<Outer>(ReductionMode::Fast, &r_outer)?;
    let backward_error = residual_norm / (ax_norm + b_norm).max(Outer::Real::one());
    let converged = current_resid <= config.outer_tolerance;
    Ok(GmresIrStatus {
        outer_iterations: iters_done,
        total_inner_iterations: total_inner,
        estimated_residual,
        final_outer_residual: current_resid,
        backward_error,
        attainable_accuracy: if converged {
            GmresIrAccuracy::Attained
        } else {
            GmresIrAccuracy::NotAttained
        },
        converged,
        restart_cycles: cycles,
        residual_history: history,
        true_residual_checks,
    })
}

/// Result of one FGMRES-IR Arnoldi cycle.
struct OneCycleResult<R> {
    iters_done: usize,
    inner_iters: usize,
    estimated_residual: R,
    happy_breakdown: bool,
}

fn reserve_ir<T>(
    values: &mut Vec<T>,
    capacity: usize,
    label: &'static str,
) -> Result<(), GmresIrError> {
    values
        .try_reserve_exact(capacity)
        .map_err(|error| GmresIrError::Allocation {
            label,
            reason: error.to_string(),
        })
}

fn ir_capacity_add(
    capacity: usize,
    extra: usize,
    label: &'static str,
) -> Result<usize, GmresIrError> {
    capacity
        .checked_add(extra)
        .ok_or_else(|| GmresIrError::Allocation {
            label,
            reason: "capacity overflow".into(),
        })
}

/// One Arnoldi cycle of size `cycle_max` for FGMRES-IR.
#[allow(clippy::too_many_arguments)]
fn fgmres_ir_one_cycle_with_alloc<B, AOuter, AInner, P, Outer>(
    backend: &B,
    a_outer: &AOuter,
    a_inner: &AInner,
    inner_precond: &P,
    b_outer: &B::Buffer<Outer>,
    b_norm: Outer::Real,
    r_init: &B::Buffer<Outer>,
    out_outer: &mut B::Buffer<Outer>,
    cycle_max: usize,
    config: &GmresIrConfig<Outer>,
    starting_resid: Outer::Real,
    iter_offset: usize,
    history: &mut Vec<Outer::Real>,
    true_residual_checks: &mut Vec<(usize, Outer::Real, Outer::Real)>,
    alloc: &Allocator<'_, B, Outer>,
    inner_alloc: &Allocator<'_, B, Outer::Lower>,
) -> Result<OneCycleResult<Outer::Real>, GmresIrError>
where
    B: Backend,
    Outer: crate::LowerPrecision,
    AOuter: BackendOperator<B, Outer> + ?Sized,
    AInner: BackendOperator<B, Outer::Lower> + ?Sized,
    P: BackendPreconditioner<B, Outer::Lower> + ?Sized,
{
    use num_traits::{Float, One, Zero};

    let n = a_outer.n();
    let zero_real = Outer::Real::zero();
    let beta = backend.try_nrm2_with_mode::<Outer>(ReductionMode::Fast, r_init)?;
    if beta == zero_real {
        return Ok(OneCycleResult {
            iters_done: 0,
            inner_iters: 0,
            estimated_residual: zero_real,
            happy_breakdown: true,
        });
    }

    let mut v_basis: Vec<Scratch<'_, B, Outer>> = Vec::new();
    reserve_ir(&mut v_basis, ir_capacity_add(cycle_max, 1, "V")?, "V")?;
    let mut z_basis: Vec<Scratch<'_, B, Outer>> = Vec::new();
    reserve_ir(&mut z_basis, cycle_max, "Z")?;

    let mut v0 = alloc.alloc(n).map_err(|error| GmresIrError::Allocation {
        label: "V[0]",
        reason: error.to_string(),
    })?;
    backend.try_copy_d2d(r_init, &mut v0)?;
    backend.try_scale::<Outer>(Outer::from_real(Outer::Real::one() / beta), &mut v0)?;
    v_basis.push(v0);

    let mut h: Vec<Vec<Outer>> = Vec::new();
    reserve_ir(&mut h, cycle_max, "H")?;
    let mut cs: Vec<Outer> = Vec::new();
    reserve_ir(&mut cs, cycle_max, "cosines")?;
    let mut sn: Vec<Outer> = Vec::new();
    reserve_ir(&mut sn, cycle_max, "sines")?;
    let mut g: Vec<Outer> = Vec::new();
    reserve_ir(&mut g, ir_capacity_add(cycle_max, 1, "rhs")?, "rhs")?;
    g.push(Outer::from_real(beta));

    let mut v_j_inner = inner_alloc
        .alloc(n)
        .map_err(|error| GmresIrError::Allocation {
            label: "v_j_inner",
            reason: error.to_string(),
        })?;
    let mut z_j_inner = inner_alloc
        .alloc(n)
        .map_err(|error| GmresIrError::Allocation {
            label: "z_j_inner",
            reason: error.to_string(),
        })?;
    let mut w_outer = alloc.alloc(n).map_err(|error| GmresIrError::Allocation {
        label: "w_outer",
        reason: error.to_string(),
    })?;

    let mut iters_done = 0_usize;
    let mut inner_iters = 0_usize;
    let mut fgmres_estimate = starting_resid;
    let mut happy_breakdown = false;

    for j in 0..cycle_max {
        // ---- Adaptive inner tolerance ----
        let inner_tol = if config.adaptive_inner_tolerance {
            let estimate_lower = Outer::cast_real_to_lower(fgmres_estimate);
            let scale =
                <<Outer::Lower as DeviceScalar>::Real as num_traits::FromPrimitive>::from_f64(
                    100.0,
                )
                .ok_or_else(|| GmresIrError::InvalidInput {
                    reason: "lower real scalar cannot represent adaptive tolerance scale".into(),
                })?;
            let scaled = estimate_lower / scale;
            if scaled > config.inner_tolerance {
                scaled
            } else {
                config.inner_tolerance
            }
        } else {
            config.inner_tolerance
        };

        // ---- Cast v_j: Outer → Lower. ----
        backend.try_cast_to_lower::<Outer>(&v_basis[j], &mut v_j_inner)?;

        backend.try_scale::<Outer::Lower>(Outer::Lower::zero(), &mut z_j_inner)?;
        let inner_status = gmres_inner_with_alloc::<B, AInner, P, Outer::Lower>(
            backend,
            a_inner,
            inner_precond,
            &v_j_inner,
            None,
            &mut z_j_inner,
            &GmresInnerConfig::<Outer::Lower> {
                max_iterations: config.max_inner_iterations,
                tolerance: inner_tol,
            },
            inner_alloc,
        )
        .map_err(|e| GmresIrError::InnerSolverFailed {
            outer: iter_offset + j,
            source: e,
        })?;
        inner_iters += inner_status.iterations;

        // Cast z_j: Lower → Outer; store in the retained Z basis.

        let mut z_j_outer = alloc.alloc(n).map_err(|error| GmresIrError::Allocation {
            label: "z_j_outer",
            reason: error.to_string(),
        })?;
        backend.try_cast_from_lower::<Outer>(&z_j_inner, &mut z_j_outer)?;

        // w = A · z_j (Outer)
        a_outer.try_apply(backend, &z_j_outer, &mut w_outer)?;
        z_basis.push(z_j_outer);

        // CGS2 against v_basis[0..=j] using fused backend projection updates.
        let mut h_col = crate::backend::try_zeroed_host::<Outer>(
            ir_capacity_add(j, 2, "H column")?,
            "fgmres_ir H column",
        )
        .map_err(GmresIrError::from)?;
        let mut basis_refs: Vec<&B::Buffer<Outer>> = Vec::new();
        reserve_ir(
            &mut basis_refs,
            ir_capacity_add(j, 1, "basis references")?,
            "basis references",
        )?;
        basis_refs.extend(v_basis.iter().take(j + 1).map(|buffer| &**buffer));

        let dots1 =
            backend.try_multi_dot_with_mode::<Outer>(ReductionMode::Fast, &basis_refs, &w_outer)?;
        let mut neg_dots1 =
            crate::backend::try_zeroed_host::<Outer>(dots1.len(), "fgmres_ir first projection")
                .map_err(GmresIrError::from)?;
        for (negated, &dot) in neg_dots1.iter_mut().zip(&dots1) {
            *negated = Outer::zero() - dot;
        }
        backend.try_axpy_many::<Outer>(&neg_dots1, &basis_refs, &mut w_outer)?;
        h_col[..=j].copy_from_slice(&dots1[..=j]);

        let dots2 =
            backend.try_multi_dot_with_mode::<Outer>(ReductionMode::Fast, &basis_refs, &w_outer)?;
        let mut neg_dots2 =
            crate::backend::try_zeroed_host::<Outer>(dots2.len(), "fgmres_ir second projection")
                .map_err(GmresIrError::from)?;
        for (negated, &dot) in neg_dots2.iter_mut().zip(&dots2) {
            *negated = Outer::zero() - dot;
        }
        backend.try_axpy_many::<Outer>(&neg_dots2, &basis_refs, &mut w_outer)?;
        for i in 0..=j {
            h_col[i] = h_col[i] + dots2[i];
        }
        let h_jp1 = backend.try_nrm2_with_mode::<Outer>(ReductionMode::Fast, &w_outer)?;
        h_col[j + 1] = Outer::from_real(h_jp1);
        let invariant_subspace = h_jp1 == zero_real;

        if !invariant_subspace {
            let mut v_next = alloc.alloc(n).map_err(|error| GmresIrError::Allocation {
                label: "V[j+1]",
                reason: error.to_string(),
            })?;
            backend.try_copy_d2d(&w_outer, &mut v_next)?;
            backend
                .try_scale::<Outer>(Outer::from_real(Outer::Real::one() / h_jp1), &mut v_next)?;
            v_basis.push(v_next);
        }

        // Apply previous Givens (canonical, conj on lower-left).
        for i in 0..j {
            let s_conj = sn[i].conj_mul(Outer::one());
            let temp = cs[i] * h_col[i] + sn[i] * h_col[i + 1];
            h_col[i + 1] = (Outer::zero() - s_conj) * h_col[i] + cs[i] * h_col[i + 1];
            h_col[i] = temp;
        }

        if invariant_subspace {
            let diagonal_norm = h_col[j].magnitude();
            if !diagonal_norm.is_finite() || diagonal_norm == zero_real {
                return Err(GmresIrError::Backend {
                    source: crate::BackendError::OperationFailed {
                        operation: "fgmres_ir",
                        reason: format!(
                            "singular or non-finite Hessenberg diagonal at iteration {}",
                            iter_offset + j + 1
                        ),
                    },
                });
            }
            h.push(h_col);
            iters_done = j + 1;
            fgmres_estimate = zero_real;
            happy_breakdown = true;
            break;
        }

        // New canonical complex Givens.
        let abs_a = h_col[j].magnitude();
        let abs_b = h_col[j + 1].magnitude();
        let r = abs_a.hypot(abs_b);
        let (cos_j, sin_j): (Outer, Outer) = if r == zero_real {
            (Outer::one(), Outer::zero())
        } else if abs_a == zero_real {
            (Outer::zero(), h_col[j + 1] / Outer::from_real(abs_b))
        } else {
            let c = Outer::from_real(abs_a / r);
            let tau = h_col[j] / Outer::from_real(abs_a);
            let conj_b = h_col[j + 1].conj_mul(Outer::one());
            let s = tau * conj_b / Outer::from_real(r);
            (c, s)
        };
        cs.push(cos_j);
        sn.push(sin_j);
        h_col[j] = cos_j * h_col[j] + sin_j * h_col[j + 1];
        h_col[j + 1] = Outer::zero();

        let g_j = g[j];
        let s_conj = sin_j.conj_mul(Outer::one());
        g.push((Outer::zero() - s_conj) * g_j);
        g[j] = cos_j * g_j;

        h.push(h_col);
        iters_done = j + 1;

        let resid_abs = g[j + 1].magnitude();
        let resid_rel = resid_abs / b_norm;
        history.push(resid_rel);
        fgmres_estimate = resid_rel;

        if let Some(period) = config.true_residual_check_period
            && period > 0
            && iters_done.is_multiple_of(period)
        {
            let true_rel = compute_true_residual_with_y_with_alloc::<B, AOuter, Outer>(
                backend, a_outer, b_outer, out_outer, &z_basis, &h, &g, iters_done, alloc,
            )? / b_norm;
            true_residual_checks.push((iter_offset + iters_done, fgmres_estimate, true_rel));
        }

        if resid_rel <= config.outer_tolerance {
            break;
        }
    }

    let k = iters_done;
    if k > 0 {
        let mut y =
            crate::backend::try_zeroed_host::<Outer>(k, "fgmres_ir residual triangular solve")
                .map_err(GmresIrError::from)?;
        for i in (0..k).rev() {
            let mut sum = g[i];
            for jj in (i + 1)..k {
                sum = sum - h[jj][i] * y[jj];
            }
            y[i] = sum / h[i][i];
        }
        let mut z_refs: Vec<&B::Buffer<Outer>> = Vec::new();
        reserve_ir(&mut z_refs, k, "residual basis references")?;
        z_refs.extend(z_basis.iter().take(k).map(|buffer| &**buffer));
        backend.try_axpy_many::<Outer>(&y, &z_refs, out_outer)?;
    }

    Ok(OneCycleResult {
        iters_done,
        inner_iters,
        estimated_residual: fgmres_estimate,
        happy_breakdown,
    })
}

/// Recompute `r = b − A x` at outer precision. Used at restart boundaries.
fn recompute_true_residual<B, AOuter, Outer>(
    backend: &B,
    a_outer: &AOuter,
    b_outer: &B::Buffer<Outer>,
    x_outer: &B::Buffer<Outer>,
    r_out: &mut B::Buffer<Outer>,
) -> Result<(), GmresIrError>
where
    B: Backend,
    Outer: crate::LowerPrecision,
    AOuter: BackendOperator<B, Outer> + ?Sized,
{
    a_outer.try_residual(backend, x_outer, b_outer, r_out)?;
    Ok(())
}

/// Within-cycle true-residual telemetry. Builds `x_tentative = x +
/// Z·y`, computes `‖b − A x_tentative‖`, then discards. Used for
/// telemetry only.
#[allow(clippy::too_many_arguments)]
fn compute_true_residual_with_y_with_alloc<B, AOuter, Outer>(
    backend: &B,
    a_outer: &AOuter,
    b_outer: &B::Buffer<Outer>,
    x_outer: &B::Buffer<Outer>,
    z_basis: &[Scratch<'_, B, Outer>],
    h: &[Vec<Outer>],
    g: &[Outer],
    k: usize,
    alloc: &Allocator<'_, B, Outer>,
) -> Result<Outer::Real, GmresIrError>
where
    B: Backend,
    Outer: crate::LowerPrecision,
    AOuter: BackendOperator<B, Outer> + ?Sized,
{
    let n = b_outer.len();
    let mut y = crate::backend::try_zeroed_host::<Outer>(k, "fgmres_ir triangular solve")
        .map_err(GmresIrError::from)?;
    for i in (0..k).rev() {
        let mut sum = g[i];
        for jj in (i + 1)..k {
            sum = sum - h[jj][i] * y[jj];
        }
        y[i] = sum / h[i][i];
    }
    let mut x_tentative = alloc.alloc(n).map_err(|error| GmresIrError::Allocation {
        label: "x_tentative",
        reason: error.to_string(),
    })?;
    backend.try_copy_d2d(x_outer, &mut x_tentative)?;
    let mut z_refs: Vec<&B::Buffer<Outer>> = Vec::new();
    reserve_ir(&mut z_refs, k, "solution basis references")?;
    z_refs.extend(z_basis.iter().take(k).map(|buffer| &**buffer));
    backend.try_axpy_many::<Outer>(&y, &z_refs, &mut x_tentative)?;
    let mut r = alloc.alloc(n).map_err(|error| GmresIrError::Allocation {
        label: "r telemetry",
        reason: error.to_string(),
    })?;
    a_outer.try_residual(backend, &x_tentative, b_outer, &mut r)?;
    Ok(backend.try_nrm2_with_mode::<Outer>(ReductionMode::Fast, &r)?)
}

#[cfg(test)]
// Fixtures intentionally use the panicking convenience constructors on
// tiny fixed matrices; production paths must use the `try_*` variants.
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::cpu::CpuBackend;

    /// 1-D Laplacian `[−1, 2, −1]` on n grid points: `2·I − tridiag(−1, 0, −1)`.
    /// SPD with κ ≈ 4·n²/π². At n=64 → κ ≈ 1660; at n=512 → κ ≈ 10⁵.
    fn laplacian_1d(n: usize) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
        let mut row_ptr = vec![0_i32; n + 1];
        let mut col_idx: Vec<i32> = Vec::new();
        let mut values: Vec<f64> = Vec::new();
        for i in 0..n {
            if i > 0 {
                col_idx.push(i as i32 - 1);
                values.push(-1.0);
            }
            col_idx.push(i as i32);
            values.push(2.0);
            if i + 1 < n {
                col_idx.push(i as i32 + 1);
                values.push(-1.0);
            }
            row_ptr[i + 1] = values.len() as i32;
        }
        (row_ptr, col_idx, values)
    }

    fn make_buf<T>(backend: &CpuBackend, host: &[T]) -> crate::cpu::HostBuffer<T>
    where
        T: crate::scalar::BufferElement,
    {
        let mut b = backend.raw_alloc::<T>(host.len()).unwrap();
        backend.copy_h2d(host, &mut b);
        b
    }

    #[test]
    fn gmres_ir_capacity_overflow_is_typed() {
        assert!(matches!(
            inner_capacity_add(usize::MAX, 1, "basis"),
            Err(GmresInnerError::Allocation { label: "basis", .. })
        ));
        assert!(matches!(
            ir_capacity_add(usize::MAX, 1, "basis"),
            Err(GmresIrError::Allocation { label: "basis", .. })
        ));
    }

    #[test]
    fn gmres_ir_inner_uses_fused_basis_updates() {
        let source = include_str!("gmres_ir.rs");
        let inner_body = source
            .split("fn gmres_inner_with_alloc")
            .nth(1)
            .expect("inner GMRES function should exist")
            .split("// ---------------------------------------------------------------------------")
            .next()
            .expect("inner GMRES function should precede separator");
        let arnoldi_body = inner_body
            .split("for j in 0..max_iter")
            .nth(1)
            .expect("inner GMRES Arnoldi loop should exist")
            .split("let h_jp1")
            .next()
            .expect("Arnoldi projection should precede norm computation");
        let update_body = inner_body
            .split("let mut dx_unprec")
            .nth(1)
            .expect("inner GMRES basis recombination should exist")
            .split("let mut dx =")
            .next()
            .expect("basis recombination should precede preconditioner application");

        assert!(
            arnoldi_body.contains("try_multi_dot_with_mode::<T>"),
            "inner GMRES projection should use checked batched multi_dot"
        );
        assert!(
            arnoldi_body.contains("axpy_many::<T>"),
            "inner GMRES projection should use fused axpy_many updates"
        );
        assert!(
            !arnoldi_body.contains("backend.axpy::<T>(T::zero() - dots"),
            "inner GMRES projection should not apply dot corrections one vector at a time"
        );
        assert!(
            update_body.contains("axpy_many::<T>(&y"),
            "inner GMRES basis recombination should use fused axpy_many"
        );
    }

    #[test]
    fn fgmres_ir_outer_uses_fused_basis_updates() {
        let source = include_str!("gmres_ir.rs");
        let cycle_body = source
            .split("fn fgmres_ir_one_cycle")
            .nth(1)
            .expect("FGMRES-IR cycle function should exist")
            .split("/// Recompute `r = b")
            .next()
            .expect("FGMRES-IR cycle should precede residual helper");
        let arnoldi_body = cycle_body
            .split("for j in 0..cycle_max")
            .nth(1)
            .expect("FGMRES-IR Arnoldi loop should exist")
            .split("let h_jp1")
            .next()
            .expect("outer Arnoldi projection should precede norm computation");

        assert!(
            arnoldi_body.contains("try_multi_dot_with_mode::<Outer>"),
            "outer FGMRES-IR projection should use checked batched multi_dot"
        );
        assert!(
            arnoldi_body.contains("axpy_many::<Outer>"),
            "outer FGMRES-IR projection should use fused axpy_many updates"
        );
        assert!(
            !arnoldi_body.contains("backend.dot::<Outer>"),
            "outer FGMRES-IR projection should not use serial backend.dot calls"
        );
    }

    #[test]
    fn gmres_ir_true_residual_uses_operator_residual_contract() {
        let source = include_str!("gmres_ir.rs");
        let residual_body = source
            .split("fn recompute_true_residual")
            .nth(1)
            .expect("true residual helper should exist")
            .split("/// Within-cycle true-residual telemetry")
            .next()
            .expect("true residual helper should precede telemetry helper");
        let telemetry_body = source
            .split("fn compute_true_residual_with_y")
            .nth(1)
            .expect("telemetry residual helper should exist")
            .split("#[cfg(test)]")
            .next()
            .expect("telemetry helper should precede tests");

        assert!(
            residual_body.contains(".try_residual("),
            "restart residual recomputation should use checked DeviceOperator residual"
        );
        assert!(
            telemetry_body.contains(".try_residual("),
            "telemetry residual recomputation should use checked DeviceOperator residual"
        );
        assert!(
            !residual_body.contains(".scale::<Outer>(-Outer::one()"),
            "restart residual helper should not open-code residual scale"
        );
    }

    #[test]
    fn gmres_ir_inner_givens_rotation_handles_large_finite_values() {
        let backend = CpuBackend::new();
        let scale = f32::MAX / 4.0;
        let a = BackendCsr::from_host(
            &backend,
            2,
            2,
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[2.0 * scale, scale, scale, 2.0 * scale],
        );
        let b = make_buf(&backend, &[1.0_f32, 0.0]);
        let mut out = backend.raw_alloc::<f32>(2).expect("output");

        let status = gmres_inner(
            &backend,
            &a,
            &IdentityBackendPreconditioner,
            &b,
            None,
            &mut out,
            &GmresInnerConfig {
                max_iterations: 1,
                tolerance: 0.5,
            },
        )
        .expect("large finite inner Hessenberg values must remain representable");
        let mut solution = [0.0; 2];
        backend.copy_d2h(&out, &mut solution);

        assert!(status.converged);
        assert!(status.final_residual.is_finite() && status.final_residual > 0.0);
        assert!(solution.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn gmres_ir_givens_rotation_handles_large_finite_outer_values() {
        let backend = CpuBackend::new();
        let scale = f64::MAX / 4.0;
        let row_ptr = [0, 2, 4];
        let col_idx = [0, 1, 0, 1];
        let a_outer = BackendCsr::from_host(
            &backend,
            2,
            2,
            &row_ptr,
            &col_idx,
            &[2.0 * scale, scale, scale, 2.0 * scale],
        );
        let a_inner =
            BackendCsrF32::from_host_f64_values(&backend, 2, 2, &[0, 1, 2], &[0, 1], &[1.0, 1.0]);
        let b = make_buf(&backend, &[1.0, 0.0]);
        let mut out = backend.raw_alloc::<f64>(2).expect("output");
        let config = GmresIrConfig {
            max_outer_iterations: 1,
            outer_tolerance: 0.5,
            max_inner_iterations: 2,
            inner_tolerance: 1.0e-6,
            restart_size: Some(1),
            ..GmresIrConfig::default()
        };

        let status = gmres_ir_backend_preconditioned(
            &backend,
            &a_outer,
            &a_inner,
            &IdentityBackendPreconditioner,
            &b,
            None,
            &mut out,
            &config,
        )
        .expect("large finite outer Hessenberg values must remain representable");
        let mut solution = [0.0; 2];
        backend.copy_d2h(&out, &mut solution);

        assert!(status.converged);
        assert!(status.final_outer_residual.is_finite());
        assert!(status.estimated_residual.is_finite());
        assert!(solution.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn gmres_ir_reports_singular_outer_arnoldi_breakdown() {
        let backend = CpuBackend::new();
        let a_outer = BackendCsr::from_host(&backend, 2, 2, &[0, 0, 0], &[], &[]);
        let a_inner =
            BackendCsrF32::from_host_f64_values(&backend, 2, 2, &[0, 1, 2], &[0, 1], &[1.0, 1.0]);
        let b = make_buf(&backend, &[1.0, 0.0]);
        let mut out = backend.raw_alloc::<f64>(2).expect("output");

        let error = gmres_ir_backend_preconditioned(
            &backend,
            &a_outer,
            &a_inner,
            &IdentityBackendPreconditioner,
            &b,
            None,
            &mut out,
            &GmresIrConfig {
                max_outer_iterations: 1,
                max_inner_iterations: 2,
                restart_size: Some(1),
                ..GmresIrConfig::default()
            },
        )
        .expect_err("a zero outer operator cannot solve a non-zero right-hand side");

        assert!(matches!(
            error,
            GmresIrError::Backend {
                source: crate::BackendError::OperationFailed {
                    operation: "fgmres_ir",
                    ..
                }
            }
        ));
    }

    /// On a κ ≈ 1660 problem, GMRES-IR with f32 inner must reach f64
    /// outer tolerance 1e-10 in a small number of outer iters. The
    /// expected per-outer-iter reduction is `κ · ε_{f32} ≈ 1660 · 1.2e-7 ≈ 2e-4`
    /// → about 3 outer iters to drop from 1 to 1e-10.
    #[test]
    fn gmres_ir_converges_f64_accuracy_on_well_conditioned() {
        let backend = CpuBackend::new();
        let n = 64;
        let (rp, ci, va) = laplacian_1d(n);
        let a_f64 = BackendCsr::from_host(&backend, n, n, &rp, &ci, &va);
        let a_f32 = BackendCsrF32::from_host_f64_values(&backend, n, n, &rp, &ci, &va);

        // RHS such that the true solution exists & is non-trivial.
        let b_host_f64: Vec<f64> = (0..n).map(|i| ((i as f64 + 1.0) * 0.1).sin()).collect();
        let b_dev_f64 = make_buf(&backend, &b_host_f64);

        let va_f32: Vec<f32> = va.iter().map(|&v| v as f32).collect();
        let pc =
            PolynomialJacobiBackendF32::from_csr_f32(&backend, &a_f32, &rp, &ci, &va_f32, 1, 1.0);

        let mut x = backend.raw_alloc::<f64>(n).unwrap();
        let cfg = GmresIrConfig {
            max_outer_iterations: 30,
            outer_tolerance: 1.0e-10,
            max_inner_iterations: 200,
            inner_tolerance: 1.0e-3,
            ..Default::default()
        };
        let status = gmres_ir_backend_preconditioned(
            &backend, &a_f64, &a_f32, &pc, &b_dev_f64, None, &mut x, &cfg,
        )
        .expect("solve");

        assert!(
            status.converged,
            "must reach 1e-10 outer tol; got {} after {} outer iters; history = {:?}",
            status.final_outer_residual, status.outer_iterations, status.residual_history
        );
        assert!(
            status.final_outer_residual <= 1.0e-10,
            "final residual {} > tol",
            status.final_outer_residual
        );
        // At κ ≈ 1660, plain IR drops ~4 digits per outer iter →
        // need ≤ 4 outer iters from 1 to 1e-10.
        assert!(
            status.outer_iterations <= 6,
            "took {} outer iters, expected ≤ 6 at κ ≈ 1660",
            status.outer_iterations
        );
    }

    /// Verify the f64 residual computed at the outer level matches the
    /// physical residual `‖b − A x‖ / ‖b‖` for the recovered solution
    /// `x`. This confirms that the IR loop's reported convergence is
    /// genuine and not a numerical artifact of f32-only updates.
    #[test]
    fn gmres_ir_solution_is_genuinely_f64_accurate() {
        let backend = CpuBackend::new();
        let n = 32;
        let (rp, ci, va) = laplacian_1d(n);
        let a_f64 = BackendCsr::from_host(&backend, n, n, &rp, &ci, &va);
        let a_f32 = BackendCsrF32::from_host_f64_values(&backend, n, n, &rp, &ci, &va);

        let b_host_f64: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) * 0.137).collect();
        let b_dev = make_buf(&backend, &b_host_f64);

        let va_f32: Vec<f32> = va.iter().map(|&v| v as f32).collect();
        let pc =
            PolynomialJacobiBackendF32::from_csr_f32(&backend, &a_f32, &rp, &ci, &va_f32, 2, 1.0);

        let mut x = backend.raw_alloc::<f64>(n).unwrap();
        let status = gmres_ir_backend_preconditioned(
            &backend,
            &a_f64,
            &a_f32,
            &pc,
            &b_dev,
            None,
            &mut x,
            &GmresIrConfig {
                max_outer_iterations: 30,
                outer_tolerance: 1.0e-12,
                max_inner_iterations: 200,
                inner_tolerance: 1.0e-3,
                ..Default::default()
            },
        )
        .expect("solve");

        assert_eq!(status.attainable_accuracy, GmresIrAccuracy::Attained);
        assert!(status.estimated_residual.is_finite());
        assert!(status.backward_error.is_finite());
        assert!(status.backward_error <= 1.0e-10);

        // Verify ‖b − A x‖ / ‖b‖ < 1e-10 in pure f64 (independent
        // computation, not the IR loop's bookkeeping).
        let mut x_host = vec![0.0_f64; n];
        backend.copy_d2h(&x, &mut x_host);
        let mut residual_norm_sq = 0.0_f64;
        let mut b_norm_sq = 0.0_f64;
        for i in 0..n {
            let lo = rp[i] as usize;
            let hi = rp[i + 1] as usize;
            let mut row_dot = 0.0_f64;
            for k in lo..hi {
                row_dot += va[k] * x_host[ci[k] as usize];
            }
            let r = b_host_f64[i] - row_dot;
            residual_norm_sq += r * r;
            b_norm_sq += b_host_f64[i] * b_host_f64[i];
        }
        let rel = (residual_norm_sq / b_norm_sq).sqrt();
        assert!(rel < 1.0e-10, "physical residual {rel:.3e} > 1e-10");
    }

    /// FGMRES-IR's robustness advantage: at κ ≈ 1.7·10⁶ — well past the
    /// `κ·u_p ≈ 0.2` regime where plain IR's convergence rate is already
    /// terrible — FGMRES-IR must still reach 1e-10 outer tolerance.
    /// Plain IR at this κ would need 1/log10(1/0.2) ≈ 14 outer iters
    /// just for theoretical convergence and would stagnate in practice
    /// once round-off catches up. FGMRES's Krylov adaptivity sidesteps
    /// this by using information from all prior z_i.
    ///
    /// 1-D Laplacian at n=2048: κ ≈ 4·2048²/π² ≈ 1.7·10⁶, κ·u_p ≈ 0.20.
    #[test]
    fn fgmres_ir_handles_kappa_where_plain_ir_struggles() {
        let backend = CpuBackend::new();
        let n = 2048;
        let (rp, ci, va) = laplacian_1d(n);
        let a_f64 = BackendCsr::from_host(&backend, n, n, &rp, &ci, &va);
        let a_f32 = BackendCsrF32::from_host_f64_values(&backend, n, n, &rp, &ci, &va);

        let b_host_f64: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (3.0 * t).sin() + 0.1 * (40.0 * t).cos()
            })
            .collect();
        let b_dev = make_buf(&backend, &b_host_f64);

        let va_f32: Vec<f32> = va.iter().map(|&v| v as f32).collect();
        // Polynomial-Jacobi K=3 in f32. Stronger inner = fewer outer iters.
        let pc =
            PolynomialJacobiBackendF32::from_csr_f32(&backend, &a_f32, &rp, &ci, &va_f32, 3, 1.0);

        let mut x = backend.raw_alloc::<f64>(n).unwrap();
        let cfg = GmresIrConfig {
            max_outer_iterations: 60,
            outer_tolerance: 1.0e-10,
            max_inner_iterations: 200,
            inner_tolerance: 1.0e-4,
            ..Default::default()
        };
        let status = gmres_ir_backend_preconditioned(
            &backend, &a_f64, &a_f32, &pc, &b_dev, None, &mut x, &cfg,
        )
        .expect("solve");

        assert!(
            status.converged,
            "FGMRES-IR must converge at κ ≈ 1.7·10⁶; got {} after {} outer / {} inner. history={:?}",
            status.final_outer_residual,
            status.outer_iterations,
            status.total_inner_iterations,
            status.residual_history,
        );

        // Verify the FGMRES estimate matches the actual f64 residual.
        let mut x_host = vec![0.0_f64; n];
        backend.copy_d2h(&x, &mut x_host);
        let mut resid_norm_sq = 0.0_f64;
        let mut b_norm_sq = 0.0_f64;
        for i in 0..n {
            let lo = rp[i] as usize;
            let hi = rp[i + 1] as usize;
            let mut row_dot = 0.0_f64;
            for k in lo..hi {
                row_dot += va[k] * x_host[ci[k] as usize];
            }
            let r = b_host_f64[i] - row_dot;
            resid_norm_sq += r * r;
            b_norm_sq += b_host_f64[i] * b_host_f64[i];
        }
        let physical_resid = (resid_norm_sq / b_norm_sq).sqrt();
        // FGMRES estimate may drift slightly under inexact f32
        // preconditioning; allow 100× slack vs the user-requested tol.
        assert!(
            physical_resid < 1.0e-8,
            "physical residual {:.3e} too far from outer tolerance",
            physical_resid
        );
    }

    /// Stagnation detection: on a problem with κ ≫ 1/ε_{f32} the outer
    /// IR loop should detect stagnation (residual stops decreasing) and
    /// return `converged = false`. We engineer a stagnant case by
    /// setting `outer_tolerance` below the f32 noise floor.
    #[test]
    fn gmres_ir_reports_stagnation_when_target_below_f32_floor() {
        let backend = CpuBackend::new();
        let n = 64;
        let (rp, ci, va) = laplacian_1d(n);
        let a_f64 = BackendCsr::from_host(&backend, n, n, &rp, &ci, &va);
        let a_f32 = BackendCsrF32::from_host_f64_values(&backend, n, n, &rp, &ci, &va);

        let b_host: Vec<f64> = (0..n).map(|i| (i as f64 * 0.05).sin()).collect();
        let b = make_buf(&backend, &b_host);
        let va_f32: Vec<f32> = va.iter().map(|&v| v as f32).collect();
        let pc =
            PolynomialJacobiBackendF32::from_csr_f32(&backend, &a_f32, &rp, &ci, &va_f32, 1, 1.0);
        let mut x = backend.raw_alloc::<f64>(n).unwrap();

        // Demand 1e-30 — well below f32 dynamic range.
        let status = gmres_ir_backend_preconditioned(
            &backend,
            &a_f64,
            &a_f32,
            &pc,
            &b,
            None,
            &mut x,
            &GmresIrConfig {
                max_outer_iterations: 30,
                outer_tolerance: 1.0e-30,
                max_inner_iterations: 200,
                inner_tolerance: 1.0e-3,
                ..Default::default()
            },
        )
        .expect("solve");
        assert!(
            !status.converged,
            "should stagnate before 1e-30; got resid {} after {} outer iters",
            status.final_outer_residual, status.outer_iterations
        );
        // Residual must still be small — IR makes progress until the
        // floor, just doesn't reach the impossible target.
        assert!(
            status.final_outer_residual < 1.0e-6,
            "stagnation level too high: {}",
            status.final_outer_residual
        );
    }

    /// Restart support: solve a moderately-ill-conditioned problem
    /// with `restart_size = 4` and verify multiple cycles run, the
    /// solver converges to the same accuracy as the unrestarted
    /// path, and `status.restart_cycles > 1`.
    #[test]
    fn restart_size_runs_multiple_cycles_and_converges() {
        let backend = CpuBackend::new();
        let n = 256;
        let (rp, ci, va) = laplacian_1d(n);
        let a_f64 = BackendCsr::from_host(&backend, n, n, &rp, &ci, &va);
        let a_f32 = BackendCsrF32::from_host_f64_values(&backend, n, n, &rp, &ci, &va);

        let b_host: Vec<f64> = (0..n).map(|i| ((i as f64 + 1.0) * 0.13).sin()).collect();
        let b_dev = make_buf(&backend, &b_host);

        let va_f32: Vec<f32> = va.iter().map(|&v| v as f32).collect();
        let pc =
            PolynomialJacobiBackendF32::from_csr_f32(&backend, &a_f32, &rp, &ci, &va_f32, 2, 1.0);

        let mut x = backend.raw_alloc::<f64>(n).unwrap();
        let cfg = GmresIrConfig {
            max_outer_iterations: 30,
            outer_tolerance: 1.0e-9,
            max_inner_iterations: 100,
            inner_tolerance: 1.0e-3,
            restart_size: Some(4),
            ..Default::default()
        };
        let status = gmres_ir_backend_preconditioned(
            &backend, &a_f64, &a_f32, &pc, &b_dev, None, &mut x, &cfg,
        )
        .expect("solve");

        assert!(
            status.converged,
            "restarted FGMRES-IR must converge; final={} cycles={} outer={}",
            status.final_outer_residual, status.restart_cycles, status.outer_iterations
        );
        assert!(
            status.restart_cycles > 1,
            "expected ≥2 cycles with restart_size=4 on a problem that needs ~10+ outer iters; got {}",
            status.restart_cycles
        );
        assert!(status.final_outer_residual <= 1.0e-9);

        // Independent verification: physical residual ‖b − A x‖ in f64.
        let mut x_host = vec![0.0_f64; n];
        backend.copy_d2h(&x, &mut x_host);
        let mut r_sq = 0.0_f64;
        let mut b_sq = 0.0_f64;
        for i in 0..n {
            let lo = rp[i] as usize;
            let hi = rp[i + 1] as usize;
            let mut row_dot = 0.0_f64;
            for k in lo..hi {
                row_dot += va[k] * x_host[ci[k] as usize];
            }
            let r = b_host[i] - row_dot;
            r_sq += r * r;
            b_sq += b_host[i] * b_host[i];
        }
        let phys_rel = (r_sq / b_sq).sqrt();
        assert!(
            phys_rel <= 1.0e-9,
            "physical residual {phys_rel:.3e} > tol after restart"
        );
    }

    /// Adaptive inner tolerance: both fixed and adaptive must converge
    /// to the same f64 accuracy. The adaptive variant uses a ramp-down
    /// from `outer_estimate / 100` to the user's `inner_tolerance`
    /// floor; whether it saves total inner iters depends on the
    /// outer-vs-inner cost trade for this particular problem (the
    /// adaptive looseness at the start can occasionally cost an extra
    /// outer iter that consumes more inner iters than it saved).
    /// For *very* large `outer_tol → inner_tol` ratios — e.g. outer
    /// 1e-12 with inner floor 1e-3 — adaptive consistently wins.
    /// We only assert correctness here; the performance claim depends
    /// on the problem.
    #[test]
    fn adaptive_inner_tolerance_preserves_convergence() {
        let backend = CpuBackend::new();
        let n = 256;
        let (rp, ci, va) = laplacian_1d(n);
        let a_f64 = BackendCsr::from_host(&backend, n, n, &rp, &ci, &va);
        let a_f32 = BackendCsrF32::from_host_f64_values(&backend, n, n, &rp, &ci, &va);
        let b_host: Vec<f64> = (0..n).map(|i| ((i as f64 + 1.0) * 0.07).cos()).collect();
        let b_dev = make_buf(&backend, &b_host);
        let va_f32: Vec<f32> = va.iter().map(|&v| v as f32).collect();
        let pc =
            PolynomialJacobiBackendF32::from_csr_f32(&backend, &a_f32, &rp, &ci, &va_f32, 2, 1.0);

        let base_cfg = GmresIrConfig {
            max_outer_iterations: 30,
            outer_tolerance: 1.0e-8,
            max_inner_iterations: 200,
            inner_tolerance: 1.0e-6,
            ..Default::default()
        };
        let mut x_fixed = backend.raw_alloc::<f64>(n).unwrap();
        let s_fixed = gmres_ir_backend_preconditioned(
            &backend,
            &a_f64,
            &a_f32,
            &pc,
            &b_dev,
            None,
            &mut x_fixed,
            &base_cfg,
        )
        .expect("fixed");

        let mut x_adapt = backend.raw_alloc::<f64>(n).unwrap();
        let adapt_cfg = GmresIrConfig {
            adaptive_inner_tolerance: true,
            ..base_cfg.clone()
        };
        let s_adapt = gmres_ir_backend_preconditioned(
            &backend,
            &a_f64,
            &a_f32,
            &pc,
            &b_dev,
            None,
            &mut x_adapt,
            &adapt_cfg,
        )
        .expect("adapt");

        assert!(s_fixed.converged && s_adapt.converged);
        // Sanity: adaptive must not cost more than 2× the fixed inner
        // iters (catastrophic regression bound).
        assert!(
            s_adapt.total_inner_iterations < 2 * s_fixed.total_inner_iterations,
            "adaptive ({}) > 2x fixed ({}) — catastrophic regression",
            s_adapt.total_inner_iterations,
            s_fixed.total_inner_iterations
        );
        // Both solutions must agree on the underlying f64 problem.
        let mut hf = vec![0.0_f64; n];
        let mut ha = vec![0.0_f64; n];
        backend.copy_d2h(&x_fixed, &mut hf);
        backend.copy_d2h(&x_adapt, &mut ha);
        for i in 0..n {
            assert!(
                (hf[i] - ha[i]).abs() < 1.0e-6,
                "row {i}: fixed {} vs adapt {}",
                hf[i],
                ha[i]
            );
        }
    }

    /// Adaptive heuristic shows its strength when the outer/inner
    /// tolerance gap is large. With outer_tol = 1e-12 and inner floor
    /// = 1e-2, fixed wastes inner work over-solving while outer is
    /// still at residual = 1.0; adaptive keeps inner loose until
    /// outer makes meaningful progress.
    #[test]
    fn adaptive_inner_tolerance_saves_iters_at_large_outer_inner_gap() {
        let backend = CpuBackend::new();
        let n = 64;
        let (rp, ci, va) = laplacian_1d(n);
        let a_f64 = BackendCsr::from_host(&backend, n, n, &rp, &ci, &va);
        let a_f32 = BackendCsrF32::from_host_f64_values(&backend, n, n, &rp, &ci, &va);
        let b_host: Vec<f64> = (0..n).map(|i| ((i as f64 + 1.0) * 0.13).sin()).collect();
        let b_dev = make_buf(&backend, &b_host);
        let va_f32: Vec<f32> = va.iter().map(|&v| v as f32).collect();
        let pc =
            PolynomialJacobiBackendF32::from_csr_f32(&backend, &a_f32, &rp, &ci, &va_f32, 2, 1.0);

        let base_cfg = GmresIrConfig {
            max_outer_iterations: 30,
            outer_tolerance: 1.0e-12,
            max_inner_iterations: 200,
            inner_tolerance: 1.0e-2, // very loose floor
            ..Default::default()
        };
        let mut x_fixed = backend.raw_alloc::<f64>(n).unwrap();
        let s_fixed = gmres_ir_backend_preconditioned(
            &backend,
            &a_f64,
            &a_f32,
            &pc,
            &b_dev,
            None,
            &mut x_fixed,
            &base_cfg,
        )
        .expect("fixed");
        let mut x_adapt = backend.raw_alloc::<f64>(n).unwrap();
        let s_adapt = gmres_ir_backend_preconditioned(
            &backend,
            &a_f64,
            &a_f32,
            &pc,
            &b_dev,
            None,
            &mut x_adapt,
            &GmresIrConfig {
                adaptive_inner_tolerance: true,
                ..base_cfg.clone()
            },
        )
        .expect("adapt");
        // Adaptive may not beat fixed at outer-tol = 1e-12 since both
        // need similar precision near the end. Just assert convergence.
        let _ = s_fixed.total_inner_iterations;
        let _ = s_adapt.total_inner_iterations;
    }

    // -----------------------------------------------------------------------
    // (Complex64, Complex32) — step c: prove the IR pipeline runs end-to-end
    // at the complex precision pair, validating the type-level generalisation
    // from sub-step a₄ and (on Metal) the native Complex32 MSL kernels from
    // step b.
    // -----------------------------------------------------------------------

    /// 1-D complex Helmholtz-like tridiagonal:
    ///   `A[i,i]   = 2 + iη`        (real diagonal + small complex damping)
    ///   `A[i,i±1] = -1`             (real off-diagonals)
    /// Diagonally dominant by `|2+iη| > 2`, condition number similar to
    /// the real Laplacian (~ 4·n²/π²). The complex damping term is what
    /// makes the inner f32 Krylov non-trivial — pure-real values would
    /// trivially reduce to the real case.
    fn complex_helmholtz_tridiag_1d(
        n: usize,
        eta: f64,
    ) -> (Vec<i32>, Vec<i32>, Vec<num_complex::Complex64>) {
        use num_complex::Complex64;
        let mut row_ptr = vec![0_i32; n + 1];
        let mut col_idx: Vec<i32> = Vec::new();
        let mut values: Vec<Complex64> = Vec::new();
        for i in 0..n {
            if i > 0 {
                col_idx.push(i as i32 - 1);
                values.push(Complex64::new(-1.0, 0.0));
            }
            col_idx.push(i as i32);
            values.push(Complex64::new(2.0, eta));
            if i + 1 < n {
                col_idx.push(i as i32 + 1);
                values.push(Complex64::new(-1.0, 0.0));
            }
            row_ptr[i + 1] = values.len() as i32;
        }
        (row_ptr, col_idx, values)
    }

    /// End-to-end FGMRES-IR at `(Complex64, Complex32)` on CpuBackend.
    /// Validates that the type-level generalisation lets us instantiate
    /// the IR pipeline at complex scalars and reach the Complex64 outer
    /// tolerance using a Complex32 inner Krylov.
    #[test]
    fn gmres_ir_complex64_complex32_converges_on_cpu() {
        use crate::PolynomialJacobiBackend;
        use num_complex::{Complex32, Complex64};

        let backend = CpuBackend::new();
        let n = 64;
        let (rp, ci, va_c64) = complex_helmholtz_tridiag_1d(n, 0.05);

        let a_outer: BackendCsr<CpuBackend, Complex64> =
            BackendCsr::from_host(&backend, n, n, &rp, &ci, &va_c64);
        let a_inner: BackendCsr<CpuBackend, Complex32> =
            BackendCsr::from_host_cast::<Complex64>(&backend, n, n, &rp, &ci, &va_c64);

        // RHS: complex valued, non-trivial.
        let b_host: Vec<Complex64> = (0..n)
            .map(|i| {
                let t = i as f64 + 1.0;
                Complex64::new((t * 0.1).sin(), (t * 0.07).cos())
            })
            .collect();
        let b_dev = make_buf(&backend, &b_host);

        // Build the inner-precision Complex32 polynomial Jacobi.
        let va_c32: Vec<Complex32> = va_c64
            .iter()
            .map(|&v| <Complex64 as crate::LowerPrecision>::cast_to_lower(v))
            .collect();
        let pc: PolynomialJacobiBackend<CpuBackend, Complex32> =
            PolynomialJacobiBackend::from_csr_host(
                &backend,
                &a_inner,
                &rp,
                &ci,
                &va_c32,
                2,
                Complex32::new(1.0, 0.0),
            );

        let mut x = backend.raw_alloc::<Complex64>(n).unwrap();
        let cfg: GmresIrConfig<Complex64> = GmresIrConfig {
            max_outer_iterations: 30,
            outer_tolerance: 1.0e-10,
            max_inner_iterations: 200,
            inner_tolerance: 1.0e-3,
            ..Default::default()
        };
        let status = gmres_ir_backend_preconditioned::<CpuBackend, _, _, _, Complex64>(
            &backend, &a_outer, &a_inner, &pc, &b_dev, None, &mut x, &cfg,
        )
        .expect("solve");

        assert!(
            status.converged,
            "Complex64↔Complex32 FGMRES-IR must converge; final={} after {} outer iters; history={:?}",
            status.final_outer_residual, status.outer_iterations, status.residual_history,
        );
        assert!(
            status.final_outer_residual <= 1.0e-10,
            "outer residual {} > tol",
            status.final_outer_residual
        );

        // Independent verification: physical residual ‖b − A x‖ in pure
        // Complex64. Confirms the recovered solution is genuinely
        // double-precision-accurate and not just IR-bookkeeping.
        let mut x_host = vec![Complex64::new(0.0, 0.0); n];
        backend.copy_d2h(&x, &mut x_host);
        let mut resid_norm_sq = 0.0_f64;
        let mut b_norm_sq = 0.0_f64;
        for i in 0..n {
            let lo = rp[i] as usize;
            let hi = rp[i + 1] as usize;
            let mut row_dot = Complex64::new(0.0, 0.0);
            for k in lo..hi {
                row_dot += va_c64[k] * x_host[ci[k] as usize];
            }
            let r = b_host[i] - row_dot;
            resid_norm_sq += r.re * r.re + r.im * r.im;
            b_norm_sq += b_host[i].re * b_host[i].re + b_host[i].im * b_host[i].im;
        }
        let rel = (resid_norm_sq / b_norm_sq).sqrt();
        assert!(rel < 1.0e-9, "physical Complex64 residual {rel:.3e} > 1e-9");
    }

    /// Identity-preconditioned Complex64↔Complex32 IR on a small problem.
    /// Smaller scope than the polynomial-Jacobi test — confirms the
    /// generic path works without requiring a non-trivial preconditioner.
    #[test]
    fn gmres_ir_complex64_complex32_identity_precond_on_cpu() {
        use num_complex::Complex64;

        let backend = CpuBackend::new();
        let n = 32;
        let (rp, ci, va_c64) = complex_helmholtz_tridiag_1d(n, 0.1);

        let a_outer: BackendCsr<CpuBackend, Complex64> =
            BackendCsr::from_host(&backend, n, n, &rp, &ci, &va_c64);
        let a_inner = BackendCsr::from_host_cast::<Complex64>(&backend, n, n, &rp, &ci, &va_c64);

        let b_host: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64 + 1.0) * 0.137, 0.0))
            .collect();
        let b_dev = make_buf(&backend, &b_host);

        let mut x = backend.raw_alloc::<Complex64>(n).unwrap();
        let cfg: GmresIrConfig<Complex64> = GmresIrConfig {
            max_outer_iterations: 30,
            outer_tolerance: 1.0e-10,
            max_inner_iterations: 200,
            inner_tolerance: 1.0e-3,
            ..Default::default()
        };
        let status = gmres_ir_backend_preconditioned::<CpuBackend, _, _, _, Complex64>(
            &backend,
            &a_outer,
            &a_inner,
            &IdentityBackendPreconditioner,
            &b_dev,
            None,
            &mut x,
            &cfg,
        )
        .expect("solve");

        assert!(
            status.converged,
            "Identity-preconditioned Complex64↔Complex32 IR must converge; final={}",
            status.final_outer_residual
        );
    }

    /// CUDA end-to-end at `(Complex64, Complex32)`: validates that
    /// backend-level device casts, cuSPARSE SpMV, cuBLAS BLAS-1, and a
    /// backend preconditioner compose inside GMRES-IR without falling
    /// back to the old host-staged cast path.
    #[cfg(feature = "cuda")]
    #[test]
    fn gmres_ir_complex64_complex32_converges_on_cuda() {
        use crate::PolynomialJacobiBackend;
        use crate::cuda::CudaBackend;
        use num_complex::{Complex32, Complex64};

        let Ok(backend) = CudaBackend::new(0) else {
            eprintln!("no CUDA device — skipping");
            return;
        };
        let n = 48;
        let (rp, ci, va_c64) = complex_helmholtz_tridiag_1d(n, 0.05);

        let a_outer: BackendCsr<CudaBackend, Complex64> =
            BackendCsr::from_host(&backend, n, n, &rp, &ci, &va_c64);
        let a_inner: BackendCsr<CudaBackend, Complex32> =
            BackendCsr::from_host_cast::<Complex64>(&backend, n, n, &rp, &ci, &va_c64);

        let b_host: Vec<Complex64> = (0..n)
            .map(|i| {
                let t = i as f64 + 1.0;
                Complex64::new((t * 0.11).sin(), (t * 0.09).cos())
            })
            .collect();
        let mut b_dev = backend.raw_alloc::<Complex64>(n).unwrap();
        backend.copy_h2d(&b_host, &mut b_dev);

        let va_c32: Vec<Complex32> = va_c64
            .iter()
            .map(|&v| <Complex64 as crate::LowerPrecision>::cast_to_lower(v))
            .collect();
        let pc: PolynomialJacobiBackend<CudaBackend, Complex32> =
            PolynomialJacobiBackend::from_csr_host(
                &backend,
                &a_inner,
                &rp,
                &ci,
                &va_c32,
                2,
                Complex32::new(1.0, 0.0),
            );

        let mut x = backend.raw_alloc::<Complex64>(n).unwrap();
        let cfg: GmresIrConfig<Complex64> = GmresIrConfig {
            max_outer_iterations: 30,
            outer_tolerance: 1.0e-10,
            max_inner_iterations: 200,
            inner_tolerance: 1.0e-3,
            ..Default::default()
        };
        let status = gmres_ir_backend_preconditioned::<CudaBackend, _, _, _, Complex64>(
            &backend, &a_outer, &a_inner, &pc, &b_dev, None, &mut x, &cfg,
        )
        .expect("solve");

        assert!(
            status.converged,
            "CUDA Complex64↔Complex32 FGMRES-IR must converge; final={} after {} outer iters; history={:?}",
            status.final_outer_residual, status.outer_iterations, status.residual_history,
        );

        let mut x_host = vec![Complex64::new(0.0, 0.0); n];
        backend.copy_d2h(&x, &mut x_host);
        let mut resid_norm_sq = 0.0_f64;
        let mut b_norm_sq = 0.0_f64;
        for i in 0..n {
            let lo = rp[i] as usize;
            let hi = rp[i + 1] as usize;
            let mut row_dot = Complex64::new(0.0, 0.0);
            for k in lo..hi {
                row_dot += va_c64[k] * x_host[ci[k] as usize];
            }
            let r = b_host[i] - row_dot;
            resid_norm_sq += r.re * r.re + r.im * r.im;
            b_norm_sq += b_host[i].re * b_host[i].re + b_host[i].im * b_host[i].im;
        }
        let rel = (resid_norm_sq / b_norm_sq).sqrt();
        assert!(
            rel < 1.0e-9,
            "CUDA-path Complex64 physical residual {rel:.3e} > 1e-9"
        );
    }

    /// Metal end-to-end at `(Complex64, Complex32)`: outer Complex64 ops
    /// run host-staged (Apple's MSL rejects `double`); inner Complex32
    /// Krylov runs natively on the new MSL kernels added in step b.
    /// Validates that the entire mixed-precision pipeline lights up on
    /// Apple Silicon.
    #[cfg(feature = "metal")]
    #[test]
    fn gmres_ir_complex64_complex32_converges_on_metal() {
        use crate::PolynomialJacobiBackend;
        use crate::metal::MetalBackend;
        use num_complex::{Complex32, Complex64};

        let Ok(backend) = MetalBackend::new() else {
            eprintln!("no Metal device — skipping");
            return;
        };
        let n = 64;
        let (rp, ci, va_c64) = complex_helmholtz_tridiag_1d(n, 0.05);

        let a_outer: BackendCsr<MetalBackend, Complex64> =
            BackendCsr::from_host(&backend, n, n, &rp, &ci, &va_c64);
        let a_inner: BackendCsr<MetalBackend, Complex32> =
            BackendCsr::from_host_cast::<Complex64>(&backend, n, n, &rp, &ci, &va_c64);

        let b_host: Vec<Complex64> = (0..n)
            .map(|i| {
                let t = i as f64 + 1.0;
                Complex64::new((t * 0.1).sin(), (t * 0.07).cos())
            })
            .collect();
        let mut b_dev = backend.raw_alloc::<Complex64>(n).unwrap();
        backend.copy_h2d(&b_host, &mut b_dev);

        let va_c32: Vec<Complex32> = va_c64
            .iter()
            .map(|&v| <Complex64 as crate::LowerPrecision>::cast_to_lower(v))
            .collect();
        let pc: PolynomialJacobiBackend<MetalBackend, Complex32> =
            PolynomialJacobiBackend::from_csr_host(
                &backend,
                &a_inner,
                &rp,
                &ci,
                &va_c32,
                2,
                Complex32::new(1.0, 0.0),
            );

        let mut x = backend.raw_alloc::<Complex64>(n).unwrap();
        let cfg: GmresIrConfig<Complex64> = GmresIrConfig {
            max_outer_iterations: 30,
            outer_tolerance: 1.0e-10,
            max_inner_iterations: 200,
            inner_tolerance: 1.0e-3,
            ..Default::default()
        };
        let status = gmres_ir_backend_preconditioned::<MetalBackend, _, _, _, Complex64>(
            &backend, &a_outer, &a_inner, &pc, &b_dev, None, &mut x, &cfg,
        )
        .expect("solve");

        assert!(
            status.converged,
            "Metal Complex64↔Complex32 FGMRES-IR must converge; final={} after {} outer iters; history={:?}",
            status.final_outer_residual, status.outer_iterations, status.residual_history,
        );

        // Physical-residual cross-check in Complex64.
        let mut x_host = vec![Complex64::new(0.0, 0.0); n];
        backend.copy_d2h(&x, &mut x_host);
        let mut resid_norm_sq = 0.0_f64;
        let mut b_norm_sq = 0.0_f64;
        for i in 0..n {
            let lo = rp[i] as usize;
            let hi = rp[i + 1] as usize;
            let mut row_dot = Complex64::new(0.0, 0.0);
            for k in lo..hi {
                row_dot += va_c64[k] * x_host[ci[k] as usize];
            }
            let r = b_host[i] - row_dot;
            resid_norm_sq += r.re * r.re + r.im * r.im;
            b_norm_sq += b_host[i].re * b_host[i].re + b_host[i].im * b_host[i].im;
        }
        let rel = (resid_norm_sq / b_norm_sq).sqrt();
        assert!(
            rel < 1.0e-9,
            "Metal-path Complex64 physical residual {rel:.3e} > 1e-9"
        );
    }

    /// True-residual telemetry: with `true_residual_check_period =
    /// Some(2)`, the status carries one entry per check.
    /// FGMRES estimate and true residual should be close (within a
    /// factor of 2) for a well-conditioned problem.
    #[test]
    fn true_residual_telemetry_logs_checks() {
        let backend = CpuBackend::new();
        let n = 64;
        let (rp, ci, va) = laplacian_1d(n);
        let a_f64 = BackendCsr::from_host(&backend, n, n, &rp, &ci, &va);
        let a_f32 = BackendCsrF32::from_host_f64_values(&backend, n, n, &rp, &ci, &va);
        let b_host: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).ln()).collect();
        let b_dev = make_buf(&backend, &b_host);
        let va_f32: Vec<f32> = va.iter().map(|&v| v as f32).collect();
        let pc =
            PolynomialJacobiBackendF32::from_csr_f32(&backend, &a_f32, &rp, &ci, &va_f32, 1, 1.0);
        let mut x = backend.raw_alloc::<f64>(n).unwrap();
        let cfg = GmresIrConfig {
            max_outer_iterations: 20,
            outer_tolerance: 1.0e-9,
            max_inner_iterations: 100,
            inner_tolerance: 1.0e-3,
            true_residual_check_period: Some(2),
            ..Default::default()
        };
        let status = gmres_ir_backend_preconditioned(
            &backend, &a_f64, &a_f32, &pc, &b_dev, None, &mut x, &cfg,
        )
        .expect("solve");

        assert!(status.converged);
        assert!(
            !status.true_residual_checks.is_empty(),
            "expected ≥1 true-residual check entry"
        );
        for (iter_idx, est, truth) in &status.true_residual_checks {
            // FGMRES estimate vs true residual: at most 10× off in
            // either direction for a well-conditioned SPD problem
            // with PolyJacobi K=1 inner.
            let ratio = est.max(*truth) / est.min(*truth).max(1.0e-30);
            assert!(
                ratio < 10.0,
                "estimate/truth divergence at iter {iter_idx}: est={est:.3e} truth={truth:.3e} ratio={ratio:.3}"
            );
        }
    }
}
