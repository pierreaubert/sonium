//! Block preconditioner for saddle point systems
//!
//! Provides preconditioning for systems of the form:
//!
//! ```text
//! [A  B ] [x] = [f]
//! [C  D ] [y]   [g]
//! ```
//!
//! Where A is typically SPD (for H1), D is SPD or zero (for pressure/Lagrange multiplier),
//! and B, C are coupling matrices.
//!
//! # Preconditioner Strategies
//!
//! - **Block diagonal**: `diag(A^{-1}, S^{-1})` - simple, parallel
//! - **Upper triangular**: `[A^{-1}, -A^{-1}*B*S^{-1}; 0, S^{-1}]`
//! - **Lower triangular**: `[A^{-1}, 0; -S^{-1}*C*A^{-1}, S^{-1}]`
//!
//! # Schur complement approximations
//!
//! - `Diagonal`: use only D (when D is nonzero)
//! - `External`: user-supplied approximation (e.g., pressure mass matrix for Stokes)
//! - `BtDiagB`: `B^T * diag(A)^{-1} * B`, assembled explicitly; the A-block
//!   solve still uses the full `block1_config` preconditioner
//! - `Simple`: classic SIMPLE approximation — diagonal scaling `diag(A)^{-1}`
//!   for the A-block solve and `B^T * diag(A)^{-1} * B` for the Schur complement
//! - `Pcd`: pressure convection-diffusion (Elman/Silvester),
//!   `S^{-1} ≈ M_p^{-1} F_p A_p^{-1}` with user-supplied pressure operators
//! - `Lsc`: least-squares commutator, `S^{-1} ≈ Q^{-1} M Q^{-1}` with
//!   `Q = B^T diag(M_u)^{-1} B`, `M = B^T diag(M_u)^{-1} A diag(M_u)^{-1} B`

use crate::{AmgConfig, AmgPreconditioner, IluPreconditioner, IlutConfig, IlutPreconditioner};
use num_traits::{Float, FromPrimitive, Zero};
use sonium_solvers_core::traits::{
    AdjointCapability, ComplexField, LinearOperator, Preconditioner, SolverError,
};
use sonium_solvers_direct::SparseCholeskyFactorization;
use sonium_solvers_sparse::CsrMatrix;

const SADDLE_SPD_EXACT_VALIDATION_MAX_ROWS: usize = 1_024;

mod saddle_point_checkpoint;
pub use saddle_point_checkpoint::SADDLE_POINT_CHECKPOINT_FORMAT_VERSION;

/// Caller-provided or independently checked definiteness assumption for one
/// of the SPD-only saddle-point blocks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SaddlePointBlockAssumption {
    /// No SPD/HPD claim is available.
    #[default]
    General,
    /// The caller declares that the block is SPD/HPD when it is too large for
    /// the bounded independent validation path.
    PositiveDefinite,
}

/// Definiteness assumptions supplied for the two saddle-point blocks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SaddlePointBlockAssumptions {
    pub block1: SaddlePointBlockAssumption,
    pub block2: SaddlePointBlockAssumption,
}

/// Evidence collected while checking the blocks selected for saddle-point
/// preconditioning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaddlePointBlockProvenance {
    NotRequired,
    IndependentlyValidated,
    CallerDeclared,
}

/// Result of checked saddle-point block assumption validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SaddlePointBlockValidation {
    pub block1: SaddlePointBlockProvenance,
    pub block2: SaddlePointBlockProvenance,
}

/// Configuration for saddle point block preconditioner.
#[derive(Debug, Clone)]
pub struct SaddlePointConfig {
    /// Number of DOFs in the (1,1) block (A).
    pub block1_size: usize,
    /// Preconditioner for block (1,1) - typically AMG or ILU.
    pub block1_config: BlockPreconditionerConfig,
    /// Preconditioner for block (2,2) or Schur complement - typically ILU or diagonal.
    pub block2_config: BlockPreconditionerConfig,
    /// Schur complement approximation strategy.
    pub schur_strategy: SchurStrategy,
    /// Whether to use upper or lower triangular approximation.
    pub upper_triangular: bool,
}

impl Default for SaddlePointConfig {
    fn default() -> Self {
        Self {
            block1_size: 0,
            block1_config: BlockPreconditionerConfig::Amg(AmgConfig::default()),
            block2_config: BlockPreconditionerConfig::Ilut(IlutConfig::default()),
            schur_strategy: SchurStrategy::Diagonal,
            upper_triangular: true,
        }
    }
}

/// Configuration for individual blocks.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum BlockPreconditionerConfig {
    /// Algebraic Multigrid (good for SPD A blocks).
    Amg(AmgConfig),
    /// ILUT factorization.
    Ilut(IlutConfig),
    /// ILU(0) factorization with default settings (pattern of the block,
    /// no fill, no pivoting search). Cheaper setup than ILUT; suites
    /// mass-dominated velocity blocks where the diagonal is strong.
    Ilu,
    /// Diagonal (Jacobi).
    Diagonal,
    /// No preconditioning (identity).
    None,
}

/// Strategy for approximating the Schur complement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchurStrategy {
    /// Use only D (block diagonal preconditioner).
    Diagonal,
    /// User-supplied external Schur complement approximation.
    /// Use `from_matrix_with_schur()` to provide the matrix.
    External,
    /// Use `S ≈ B^T * diag(A)^{-1} * B`, assembled explicitly as a sparse
    /// matrix and factored with the `block2_config` preconditioner.
    ///
    /// The A-block solve still uses the full `block1_config` preconditioner,
    /// which distinguishes this from `Simple`. This is the standard
    /// "diagonal Schur complement" approximation for Stokes-type problems
    /// where `C = B^T` and A is well approximated by its diagonal
    /// (e.g., small viscosity time steps). Note the transpose `B^T` is used
    /// (not the conjugate transpose), matching the `C = B^T` coupling.
    BtDiagB,
    /// Classic SIMPLE approximation: both the A-block solve and the Schur
    /// complement use the diagonal of A.
    ///
    /// The (1,1) block preconditioner is replaced by diagonal scaling
    /// `diag(A)^{-1}` (regardless of `block1_config`), and the Schur
    /// complement is the explicitly assembled `S ≈ B^T * diag(A)^{-1} * B`,
    /// factored with the `block2_config` preconditioner. Cheaper per
    /// iteration than `BtDiagB` but typically needs more iterations.
    Simple,
    /// PCD (pressure convection-diffusion) Schur approximation
    /// (Elman/Silvester): `S^{-1} ≈ M_p^{-1} F_p A_p^{-1}`, where `F_p` is
    /// the pressure convection-diffusion operator (applied as a matvec),
    /// `A_p` the pressure Poisson operator and `M_p` the pressure mass
    /// matrix. This is the ordering consistent with the commutator
    /// `F B ≈ B F_p` for the `[A B; B^T 0]` block convention: for Stokes
    /// (`F_p = ν A_p`) it recovers the exact scaling `S ≈ (1/ν) M_p`.
    /// Requires the pressure-space operators via `from_matrix_with_pcd()`;
    /// the `A_p` and `M_p` solves use the `block2_config` preconditioner.
    Pcd,
    /// LSC (least-squares commutator) Schur approximation:
    /// `S^{-1} ≈ Q^{-1} M Q^{-1}` with `Q = B^T W B` and
    /// `M = B^T W A W B` assembled explicitly, `W = diag(M_u)^{-1}` the
    /// inverse (lumped) velocity mass diagonal. Requires the mass diagonal
    /// via `from_matrix_with_lsc()`; the `Q` solve uses the `block2_config`
    /// preconditioner.
    Lsc,
}

/// User-supplied pressure-space operators for the PCD Schur approximation
/// ([`SchurStrategy::Pcd`]). All blocks are `m × m` where `m` is the (2,2)
/// block size. The preconditioner is physics-agnostic: discretizing these
/// operators consistently with the saddle point system is the caller's
/// responsibility.
#[derive(Debug, Clone)]
pub struct PcdBlocks<T: ComplexField> {
    /// Pressure convection-diffusion operator `F_p` (matvec only).
    pub fp: CsrMatrix<T>,
    /// Pressure Poisson operator `A_p` (factored with the `block2_config`
    /// preconditioner).
    pub ap: CsrMatrix<T>,
    /// Pressure mass matrix `M_p`.
    pub mp: CsrMatrix<T>,
}

/// User-supplied blocks for the LSC Schur approximation
/// ([`SchurStrategy::Lsc`]).
#[derive(Debug, Clone)]
pub struct LscBlocks<T: ComplexField> {
    /// Inverse of the (lumped) velocity mass matrix diagonal,
    /// `diag(M_u)^{-1}`, one entry per (1,1)-block DOF.
    pub velocity_mass_diag_inv: Vec<T>,
}

/// Additional operators for the composite Schur inverse applications
/// (PCD/LSC). For the single-preconditioner strategies this is `None` and
/// `block2` alone approximates `S^{-1}`.
enum SchurExtra<T: ComplexField> {
    /// PCD: `S^{-1} ≈ M_p^{-1} F_p A_p^{-1}`; the `A_p` solve lives in
    /// `block2`.
    Pcd {
        /// Pressure convection-diffusion operator (matvec only).
        fp: CsrMatrix<T>,
        /// Preconditioner for the pressure mass matrix solve.
        mp_inv: Box<dyn Preconditioner<T> + Send + Sync>,
    },
    /// LSC: `S^{-1} ≈ Q^{-1} M Q^{-1}` with `Q = B^T W B`; the `Q` solve
    /// lives in `block2`, this is the explicitly assembled middle operator
    /// `M = B^T W A W B`.
    Lsc { middle: CsrMatrix<T> },
}

/// Block preconditioner for 2×2 saddle point systems.
///
/// For upper triangular mode, applies:
///
/// ```text
/// P^{-1} r = [A^{-1} (r1 - B * S^{-1} r2)]
///            [S^{-1} r2                     ]
/// ```
///
/// For lower triangular mode, applies:
///
/// ```text
/// P^{-1} r = [A^{-1} r1                     ]
///            [S^{-1} (r2 - C * A^{-1} r1)   ]
/// ```
///
/// For diagonal mode: `diag(A^{-1}, S^{-1})`
pub struct SaddlePointPreconditioner<T: ComplexField> {
    /// Original monolithic operator retained for portable checkpointing.
    source_matrix: CsrMatrix<T>,
    config: SaddlePointConfig,
    block1: Box<dyn Preconditioner<T> + Send + Sync>,
    /// Primary Schur solver: the full `S^{-1}` approximation for
    /// Diagonal/External/BtDiagB/Simple; the `A_p` solve for PCD; the
    /// `Q = B^T W B` solve for LSC.
    block2: Box<dyn Preconditioner<T> + Send + Sync>,
    /// Composite Schur operators for PCD/LSC; `None` otherwise.
    schur_extra: Option<SchurExtra<T>>,
    /// Off-diagonal block B (rows 0..b1, cols b1..n) for triangular application
    coupling_b: Option<CsrMatrix<T>>,
    /// Off-diagonal block C (rows b1..n, cols 0..b1) for triangular application
    coupling_c: Option<CsrMatrix<T>>,
}

impl<T: ComplexField> SaddlePointPreconditioner<T> {
    /// Checked variant of [`Self::from_matrix`].
    pub fn try_from_matrix(
        matrix: &CsrMatrix<T>,
        config: SaddlePointConfig,
    ) -> Result<Self, SolverError> {
        Self::try_from_matrix_with_assumptions(
            matrix,
            config,
            SaddlePointBlockAssumptions::default(),
        )
    }

    /// Checked saddle-point setup with explicit assumptions for SPD-only
    /// block preconditioners.
    pub fn try_from_matrix_with_assumptions(
        matrix: &CsrMatrix<T>,
        config: SaddlePointConfig,
        assumptions: SaddlePointBlockAssumptions,
    ) -> Result<Self, SolverError> {
        Self::validate_checked_block_assumptions(matrix, &config, assumptions, None)?;
        if matches!(
            config.schur_strategy,
            SchurStrategy::Pcd | SchurStrategy::Lsc
        ) {
            return Err(SolverError::OperationFailed {
                operation: "saddle-point preconditioner setup",
                reason: format!(
                    "SchurStrategy::{:?} requires specialized block data",
                    config.schur_strategy
                ),
            });
        }
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Self::from_matrix(matrix, config)
        }))
        .map_err(|_| SolverError::OperationFailed {
            operation: "saddle-point preconditioner setup",
            reason: "legacy setup panicked after checked validation".into(),
        })
    }

    /// Validate the extracted A and Schur blocks before a checked setup.
    ///
    /// AMG is only constructed for a block that passes the bounded exact
    /// SPD/HPD check or carries an explicit caller declaration. Large blocks
    /// still receive cheap symmetry, finite-value, and positive-diagonal
    /// checks, but the declaration is the documented trust boundary for
    /// definiteness beyond the exact-validation limit.
    pub fn validate_checked_block_assumptions(
        matrix: &CsrMatrix<T>,
        config: &SaddlePointConfig,
        assumptions: SaddlePointBlockAssumptions,
        schur_approx: Option<&CsrMatrix<T>>,
    ) -> Result<SaddlePointBlockValidation, SolverError> {
        validate_checked_setup_inputs(matrix, config.block1_size, schur_approx)?;
        if matches!(
            config.schur_strategy,
            SchurStrategy::Pcd | SchurStrategy::Lsc
        ) {
            return Err(SolverError::OperationFailed {
                operation: "saddle-point preconditioner setup",
                reason: format!(
                    "SchurStrategy::{:?} requires specialized block data",
                    config.schur_strategy
                ),
            });
        }
        validate_checked_block_setups(matrix, config, assumptions, schur_approx)
    }

    /// Create a new saddle point preconditioner from a 2×2 block matrix.
    ///
    /// Extracts blocks from the monolithic matrix. For `SchurStrategy::Diagonal`,
    /// the (2,2) block D is used directly as the Schur complement approximation.
    /// For `SchurStrategy::BtDiagB` and `SchurStrategy::Simple`, the Schur
    /// complement `S ≈ B^T * diag(A)^{-1} * B` is assembled explicitly and
    /// factored with the `block2_config` preconditioner; `Simple` additionally
    /// replaces the A-block solve by diagonal scaling.
    pub fn from_matrix(matrix: &CsrMatrix<T>, config: SaddlePointConfig) -> Self {
        let n = matrix.num_rows;
        let b1 = config.block1_size;
        let m = n - b1;

        // Extract block (1,1) = A
        let block1_mat = extract_block(matrix, 0, 0, b1, b1);
        let block1_precond = match config.schur_strategy {
            // SIMPLE approximates the A-block solve by diagonal scaling.
            SchurStrategy::Simple => Box::new(crate::DiagonalPreconditioner::from_csr(&block1_mat))
                as Box<dyn Preconditioner<T> + Send + Sync>,
            _ => build_preconditioner(&block1_mat, &config.block1_config),
        };

        // Extract coupling blocks for triangular modes and Schur assembly
        let (coupling_b, coupling_c) = if config.schur_strategy != SchurStrategy::Diagonal {
            let b = extract_block(matrix, 0, b1, b1, m);
            let c = extract_block(matrix, b1, 0, m, b1);
            (Some(b), Some(c))
        } else {
            (None, None)
        };

        // Block (2,2) preconditioner: exact D for Diagonal/External, explicit
        // B^T * diag(A)^{-1} * B for BtDiagB/Simple.
        let block2_precond = match config.schur_strategy {
            SchurStrategy::BtDiagB | SchurStrategy::Simple => {
                let b = coupling_b
                    .as_ref()
                    .expect("coupling block B required for Schur assembly");
                let schur = form_bt_diag_b(&block1_mat, b);
                build_preconditioner(&schur, &config.block2_config)
            }
            SchurStrategy::Pcd => panic!(
                "SchurStrategy::Pcd requires pressure-space operators; \
                 use SaddlePointPreconditioner::from_matrix_with_pcd"
            ),
            SchurStrategy::Lsc => panic!(
                "SchurStrategy::Lsc requires the velocity mass diagonal; \
                 use SaddlePointPreconditioner::from_matrix_with_lsc"
            ),
            _ => {
                let block2_mat = extract_block(matrix, b1, b1, m, m);
                build_preconditioner(&block2_mat, &config.block2_config)
            }
        };

        Self {
            source_matrix: matrix.clone(),
            config,
            block1: block1_precond,
            block2: block2_precond,
            schur_extra: None,
            coupling_b,
            coupling_c,
        }
    }

    /// Create a saddle point preconditioner with an external Schur complement approximation.
    ///
    /// For Stokes: the Schur complement S = B^T A^{-1} B is approximated by
    /// `(1/mu) * M_p` where `M_p` is the pressure mass matrix. Pass `schur_approx = (1/mu) * M_p`.
    ///
    /// The monolithic matrix is still needed to extract the A block and coupling blocks B, C.
    pub fn from_matrix_with_schur(
        matrix: &CsrMatrix<T>,
        schur_approx: &CsrMatrix<T>,
        mut config: SaddlePointConfig,
    ) -> Self {
        config.schur_strategy = SchurStrategy::External;
        let n = matrix.num_rows;
        let b1 = config.block1_size;
        let m = n - b1;

        // Extract block (1,1) = A
        let block1_mat = extract_block(matrix, 0, 0, b1, b1);
        let block1_precond = build_preconditioner(&block1_mat, &config.block1_config);

        // Build Schur complement preconditioner from external approximation
        let block2_precond = build_preconditioner(schur_approx, &config.block2_config);

        // Extract coupling blocks for triangular application
        let coupling_b = Some(extract_block(matrix, 0, b1, b1, m));
        let coupling_c = Some(extract_block(matrix, b1, 0, m, b1));

        Self {
            source_matrix: matrix.clone(),
            config,
            block1: block1_precond,
            block2: block2_precond,
            schur_extra: None,
            coupling_b,
            coupling_c,
        }
    }

    /// Checked variant of [`Self::from_matrix_with_schur`].
    pub fn try_from_matrix_with_schur(
        matrix: &CsrMatrix<T>,
        schur_approx: &CsrMatrix<T>,
        config: SaddlePointConfig,
    ) -> Result<Self, SolverError> {
        Self::try_from_matrix_with_schur_and_assumptions(
            matrix,
            schur_approx,
            config,
            SaddlePointBlockAssumptions::default(),
        )
    }

    /// Checked external-Schur setup with explicit assumptions for SPD-only
    /// block preconditioners.
    pub fn try_from_matrix_with_schur_and_assumptions(
        matrix: &CsrMatrix<T>,
        schur_approx: &CsrMatrix<T>,
        config: SaddlePointConfig,
        assumptions: SaddlePointBlockAssumptions,
    ) -> Result<Self, SolverError> {
        let mut config = config;
        config.schur_strategy = SchurStrategy::External;
        Self::validate_checked_block_assumptions(matrix, &config, assumptions, Some(schur_approx))?;
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Self::from_matrix_with_schur(matrix, schur_approx, config)
        }))
        .map_err(|_| SolverError::OperationFailed {
            operation: "saddle-point external Schur setup",
            reason: "legacy setup panicked after checked validation".into(),
        })
    }

    /// Create a saddle point preconditioner with the PCD (pressure
    /// convection-diffusion) Schur complement approximation.
    ///
    /// The Schur complement inverse is approximated as
    /// `S^{-1} ≈ M_p^{-1} F_p A_p^{-1}` (Elman/Silvester): per application,
    /// one solve with `A_p`, one matvec with `F_p`, and one solve with
    /// `M_p`. The `A_p` and `M_p` solves use the `block2_config`
    /// preconditioner; the A-block solve uses `block1_config`.
    pub fn from_matrix_with_pcd(
        matrix: &CsrMatrix<T>,
        blocks: &PcdBlocks<T>,
        mut config: SaddlePointConfig,
    ) -> Self {
        config.schur_strategy = SchurStrategy::Pcd;
        let n = matrix.num_rows;
        let b1 = config.block1_size;
        let m = n - b1;
        assert_eq!(
            (blocks.fp.num_rows, blocks.fp.num_cols),
            (m, m),
            "PCD F_p block size mismatch"
        );
        assert_eq!(
            (blocks.ap.num_rows, blocks.ap.num_cols),
            (m, m),
            "PCD A_p block size mismatch"
        );
        assert_eq!(
            (blocks.mp.num_rows, blocks.mp.num_cols),
            (m, m),
            "PCD M_p block size mismatch"
        );
        // Validate-once: the owned `fp` clone below is applied unchecked
        // on every Schur-complement application.
        crate::amg::util::validate_owned_structure(&blocks.fp)
            .expect("PCD setup requires valid F_p CSR storage");

        // Extract block (1,1) = A
        let block1_mat = extract_block(matrix, 0, 0, b1, b1);
        let block1_precond = build_preconditioner(&block1_mat, &config.block1_config);

        let ap_inv = build_preconditioner(&blocks.ap, &config.block2_config);
        let mp_inv = build_preconditioner(&blocks.mp, &config.block2_config);

        // Extract coupling blocks for triangular application
        let coupling_b = Some(extract_block(matrix, 0, b1, b1, m));
        let coupling_c = Some(extract_block(matrix, b1, 0, m, b1));

        Self {
            source_matrix: matrix.clone(),
            config,
            block1: block1_precond,
            block2: ap_inv,
            schur_extra: Some(SchurExtra::Pcd {
                fp: blocks.fp.clone(),
                mp_inv,
            }),
            coupling_b,
            coupling_c,
        }
    }

    /// Create a saddle point preconditioner with the LSC (least-squares
    /// commutator) Schur complement approximation.
    ///
    /// With `W = diag(M_u)^{-1}` the inverse velocity mass diagonal, the
    /// Schur complement inverse is approximated as
    /// `S^{-1} ≈ Q^{-1} M Q^{-1}` where `Q = B^T W B` and
    /// `M = B^T W A W B` are assembled explicitly at construction. Per
    /// application: two solves with `Q` (using the `block2_config`
    /// preconditioner) and one matvec with `M`.
    pub fn from_matrix_with_lsc(
        matrix: &CsrMatrix<T>,
        blocks: &LscBlocks<T>,
        mut config: SaddlePointConfig,
    ) -> Self {
        config.schur_strategy = SchurStrategy::Lsc;
        let n = matrix.num_rows;
        let b1 = config.block1_size;
        let m = n - b1;
        assert_eq!(
            blocks.velocity_mass_diag_inv.len(),
            b1,
            "LSC velocity mass diagonal length mismatch"
        );

        // Extract block (1,1) = A (the velocity block F)
        let block1_mat = extract_block(matrix, 0, 0, b1, b1);
        let block1_precond = build_preconditioner(&block1_mat, &config.block1_config);

        // Extract coupling blocks for triangular application
        let coupling_b = extract_block(matrix, 0, b1, b1, m);
        let coupling_c = Some(extract_block(matrix, b1, 0, m, b1));

        // Y = W * B: scale each row of B by the inverse mass diagonal.
        let w = &blocks.velocity_mass_diag_inv;
        let mut y_values = coupling_b.values.clone();
        for (i, &wi) in w[..coupling_b.num_rows].iter().enumerate() {
            for slot in &mut y_values[coupling_b.row_ptrs[i]..coupling_b.row_ptrs[i + 1]] {
                *slot *= wi;
            }
        }
        let y = CsrMatrix::from_raw_parts(
            coupling_b.num_rows,
            coupling_b.num_cols,
            coupling_b.row_ptrs.clone(),
            coupling_b.col_indices.clone(),
            y_values,
        );

        // Q = B^T W B = B^T Y;  M = B^T W A W B = Y^T A Y.
        let q = transpose_csr(&coupling_b).matmul(&y);
        let middle = transpose_csr(&y).matmul(&block1_mat).matmul(&y);
        let q_inv = build_preconditioner(&q, &config.block2_config);

        Self {
            source_matrix: matrix.clone(),
            config,
            block1: block1_precond,
            block2: q_inv,
            schur_extra: Some(SchurExtra::Lsc { middle }),
            coupling_b: Some(coupling_b),
            coupling_c,
        }
    }

    /// Get references to the sub-preconditioners.
    pub fn block1_preconditioner(&self) -> &(dyn Preconditioner<T> + Send + Sync) {
        &*self.block1
    }

    pub fn block2_preconditioner(&self) -> &(dyn Preconditioner<T> + Send + Sync) {
        &*self.block2
    }

    /// Report whether this saddle-point approximation exposes exact adjoint
    /// actions.  CSR coupling operators have exact transpose/Hermitian
    /// matvecs; composite Schur strategies additionally require exact
    /// adjoints from each constituent preconditioner.
    fn adjoint_capability(&self, hermitian: bool) -> AdjointCapability {
        let capability = |preconditioner: &(dyn Preconditioner<T> + Send + Sync)| {
            if hermitian {
                preconditioner.hermitian_capability()
            } else {
                preconditioner.transpose_capability()
            }
        };

        if capability(&*self.block1) != AdjointCapability::Exact {
            return AdjointCapability::Unsupported;
        }

        match self.config.schur_strategy {
            SchurStrategy::Diagonal
            | SchurStrategy::External
            | SchurStrategy::BtDiagB
            | SchurStrategy::Simple => {
                if capability(&*self.block2) == AdjointCapability::Exact {
                    AdjointCapability::Exact
                } else {
                    AdjointCapability::Unsupported
                }
            }
            SchurStrategy::Pcd => {
                let Some(SchurExtra::Pcd { mp_inv, .. }) = &self.schur_extra else {
                    return AdjointCapability::Unsupported;
                };
                if capability(&*self.block2) == AdjointCapability::Exact
                    && capability(&**mp_inv) == AdjointCapability::Exact
                {
                    AdjointCapability::Exact
                } else {
                    AdjointCapability::Unsupported
                }
            }
            SchurStrategy::Lsc => {
                if !matches!(self.schur_extra, Some(SchurExtra::Lsc { .. })) {
                    return AdjointCapability::Unsupported;
                }
                if capability(&*self.block2) == AdjointCapability::Exact {
                    AdjointCapability::Exact
                } else {
                    AdjointCapability::Unsupported
                }
            }
        }
    }
}

impl<T: ComplexField + Send + Sync> SaddlePointPreconditioner<T> {
    /// Apply the Schur complement inverse approximation to a (2,2)-block
    /// residual.
    fn apply_schur(&self, r2: &[T], y2: &mut [T]) {
        match &self.schur_extra {
            None => self.block2.apply_into(r2, y2),
            Some(SchurExtra::Pcd { fp, mp_inv }) => {
                // y2 = M_p^{-1} (F_p (A_p^{-1} r2))
                let t = self.block2.apply(r2);
                let mut s = vec![T::zero(); fp.num_rows];
                // SAFETY: `fp` was validated once at setup
                // (`from_matrix_with_pcd`) and sits behind a private field
                // with no post-setup mutation path. Dimensions are
                // asserted inside.
                unsafe { fp.matvec_into_buf_unchecked(&t, &mut s) };
                mp_inv.apply_into(&s, y2);
            }
            Some(SchurExtra::Lsc { middle }) => {
                // y2 = Q^{-1} (M (Q^{-1} r2))
                let t = self.block2.apply(r2);
                let mut s = vec![T::zero(); middle.num_rows];
                // SAFETY: `middle` is a setup-assembled SpGEMM product,
                // hence structurally valid by construction, and sits
                // behind a private field with no post-setup mutation
                // path. Dimensions are asserted inside.
                unsafe { middle.matvec_into_buf_unchecked(&t, &mut s) };
                self.block2.apply_into(&s, y2);
            }
        }
    }

    /// Apply the transpose of the Schur-complement inverse approximation.
    fn apply_schur_transpose(&self, r2: &[T], y2: &mut [T]) {
        match &self.schur_extra {
            None => self.block2.apply_transpose_into(r2, y2),
            Some(SchurExtra::Pcd { fp, mp_inv }) => {
                // (M_p^{-1} F_p A_p^{-1})^T
                //     = A_p^{-T} F_p^T M_p^{-T}.
                let t = mp_inv.apply_transpose(r2);
                let s = fp.matvec_transpose(&t);
                self.block2.apply_transpose_into(&s, y2);
            }
            Some(SchurExtra::Lsc { middle }) => {
                // (Q^{-1} M Q^{-1})^T = Q^{-T} M^T Q^{-T}.
                let t = self.block2.apply_transpose(r2);
                let s = middle.matvec_transpose(&t);
                self.block2.apply_transpose_into(&s, y2);
            }
        }
    }

    /// Apply the Hermitian adjoint of the Schur-complement inverse
    /// approximation.
    fn apply_schur_hermitian(&self, r2: &[T], y2: &mut [T]) {
        match &self.schur_extra {
            None => self.block2.apply_hermitian_into(r2, y2),
            Some(SchurExtra::Pcd { fp, mp_inv }) => {
                // (M_p^{-1} F_p A_p^{-1})^H
                //     = A_p^{-H} F_p^H M_p^{-H}.
                let t = mp_inv.apply_hermitian(r2);
                let s = fp.matvec_hermitian(&t);
                self.block2.apply_hermitian_into(&s, y2);
            }
            Some(SchurExtra::Lsc { middle }) => {
                // (Q^{-1} M Q^{-1})^H = Q^{-H} M^H Q^{-H}.
                let t = self.block2.apply_hermitian(r2);
                let s = middle.matvec_hermitian(&t);
                self.block2.apply_hermitian_into(&s, y2);
            }
        }
    }
}

impl<T: ComplexField + Send + Sync> Preconditioner<T> for SaddlePointPreconditioner<T> {
    fn apply(&self, r: &[T]) -> Vec<T> {
        let mut y = vec![T::zero(); r.len()];
        self.apply_into(r, &mut y);
        y
    }

    fn apply_into(&self, r: &[T], y: &mut [T]) {
        let n = r.len();
        let b1 = self.config.block1_size;
        assert_eq!(y.len(), n, "saddle-point output length mismatch");
        assert!(
            b1 <= n,
            "saddle-point block1_size exceeds input vector length"
        );

        let (r1, r2) = r.split_at(b1);
        let (y1, y2) = y.split_at_mut(b1);

        match self.config.schur_strategy {
            SchurStrategy::Diagonal => {
                // Block diagonal: P^{-1} = diag(A^{-1}, D^{-1})
                self.block1.apply_into(r1, y1);
                self.block2.apply_into(r2, y2);
            }
            SchurStrategy::External
            | SchurStrategy::Simple
            | SchurStrategy::BtDiagB
            | SchurStrategy::Pcd
            | SchurStrategy::Lsc => {
                if self.config.upper_triangular {
                    // Upper triangular:
                    // y2 = S^{-1} * r2
                    // y1 = A^{-1} * (r1 - B * y2)
                    self.apply_schur(r2, y2);
                    if let Some(ref b) = self.coupling_b {
                        let mut b_y2 = vec![T::zero(); b.num_rows()];
                        // SAFETY: coupling blocks are `extract_block`
                        // products — monotone row pointers and bounded
                        // columns by construction — kept behind a private
                        // field with no post-setup mutation path.
                        // Dimensions are asserted inside.
                        unsafe { b.matvec_into_buf_unchecked(y2, &mut b_y2) };
                        let mut r1_mod = r1.to_vec();
                        for (dst, correction) in r1_mod.iter_mut().zip(b_y2) {
                            *dst -= correction;
                        }
                        self.block1.apply_into(&r1_mod, y1);
                    } else {
                        self.block1.apply_into(r1, y1);
                    }
                } else {
                    // Lower triangular:
                    // y1 = A^{-1} * r1
                    // y2 = S^{-1} * (r2 - C * y1)
                    self.block1.apply_into(r1, y1);
                    if let Some(ref c) = self.coupling_c {
                        let mut c_y1 = vec![T::zero(); c.num_rows()];
                        // SAFETY: see the `coupling_b` site above —
                        // `extract_block` output behind a private field
                        // with no post-setup mutation path.
                        unsafe { c.matvec_into_buf_unchecked(y1, &mut c_y1) };
                        let mut r2_mod = r2.to_vec();
                        for (dst, correction) in r2_mod.iter_mut().zip(c_y1) {
                            *dst -= correction;
                        }
                        self.apply_schur(&r2_mod, y2);
                    } else {
                        self.apply_schur(r2, y2);
                    }
                }
            }
        }
    }

    fn transpose_capability(&self) -> AdjointCapability {
        self.adjoint_capability(false)
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        self.adjoint_capability(true)
    }

    fn apply_transpose(&self, r: &[T]) -> Vec<T> {
        assert_eq!(
            self.transpose_capability(),
            AdjointCapability::Exact,
            "saddle-point preconditioner transpose requires exact block and Schur adjoints"
        );
        let b1 = self.config.block1_size;
        assert!(
            b1 <= r.len(),
            "saddle-point block size exceeds input length"
        );
        let (r1, r2) = r.split_at(b1);
        let mut y = vec![T::zero(); r.len()];
        let (y1, y2) = y.split_at_mut(b1);
        if self.config.schur_strategy == SchurStrategy::Diagonal {
            self.block1.apply_transpose_into(r1, y1);
            self.block2.apply_transpose_into(r2, y2);
            return y;
        }

        if self.config.upper_triangular {
            // P^{-T}: y1 = A^{-T} r1,
            //        y2 = S^{-T} (r2 - B^T y1).
            self.block1.apply_transpose_into(r1, y1);
            if let Some(ref b) = self.coupling_b {
                let b_t_y1 = b.matvec_transpose(y1);
                let mut r2_mod = r2.to_vec();
                for (dst, correction) in r2_mod.iter_mut().zip(b_t_y1) {
                    *dst -= correction;
                }
                self.apply_schur_transpose(&r2_mod, y2);
            } else {
                self.apply_schur_transpose(r2, y2);
            }
        } else {
            // P^{-T}: y2 = S^{-T} r2,
            //        y1 = A^{-T} (r1 - C^T y2).
            self.apply_schur_transpose(r2, y2);
            if let Some(ref c) = self.coupling_c {
                let c_t_y2 = c.matvec_transpose(y2);
                let mut r1_mod = r1.to_vec();
                for (dst, correction) in r1_mod.iter_mut().zip(c_t_y2) {
                    *dst -= correction;
                }
                self.block1.apply_transpose_into(&r1_mod, y1);
            } else {
                self.block1.apply_transpose_into(r1, y1);
            }
        }
        y
    }

    fn apply_hermitian(&self, r: &[T]) -> Vec<T> {
        assert_eq!(
            self.hermitian_capability(),
            AdjointCapability::Exact,
            "saddle-point preconditioner Hermitian action requires exact block and Schur adjoints"
        );
        let b1 = self.config.block1_size;
        assert!(
            b1 <= r.len(),
            "saddle-point block size exceeds input length"
        );
        let (r1, r2) = r.split_at(b1);
        let mut y = vec![T::zero(); r.len()];
        let (y1, y2) = y.split_at_mut(b1);
        if self.config.schur_strategy == SchurStrategy::Diagonal {
            self.block1.apply_hermitian_into(r1, y1);
            self.block2.apply_hermitian_into(r2, y2);
            return y;
        }

        if self.config.upper_triangular {
            // P^{-H}: y1 = A^{-H} r1,
            //        y2 = S^{-H} (r2 - B^H y1).
            self.block1.apply_hermitian_into(r1, y1);
            if let Some(ref b) = self.coupling_b {
                let b_h_y1 = b.matvec_hermitian(y1);
                let mut r2_mod = r2.to_vec();
                for (dst, correction) in r2_mod.iter_mut().zip(b_h_y1) {
                    *dst -= correction;
                }
                self.apply_schur_hermitian(&r2_mod, y2);
            } else {
                self.apply_schur_hermitian(r2, y2);
            }
        } else {
            // P^{-H}: y2 = S^{-H} r2,
            //        y1 = A^{-H} (r1 - C^H y2).
            self.apply_schur_hermitian(r2, y2);
            if let Some(ref c) = self.coupling_c {
                let c_h_y2 = c.matvec_hermitian(y2);
                let mut r1_mod = r1.to_vec();
                for (dst, correction) in r1_mod.iter_mut().zip(c_h_y2) {
                    *dst -= correction;
                }
                self.block1.apply_hermitian_into(&r1_mod, y1);
            } else {
                self.block1.apply_hermitian_into(r1, y1);
            }
        }
        y
    }
}

/// Extract a block from a CSR matrix.
fn extract_block<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
) -> CsrMatrix<T> {
    if nrows == 0 || ncols == 0 {
        return CsrMatrix::new(0, 0);
    }

    let row_ptrs = &matrix.row_ptrs;
    let col_indices = &matrix.col_indices;
    let values = &matrix.values;

    let mut new_row_ptrs = vec![0usize; nrows + 1];
    let mut new_col_indices: Vec<usize> = Vec::new();
    let mut new_values: Vec<T> = Vec::new();

    for i in 0..nrows {
        let global_row = row_start + i;
        let start = row_ptrs[global_row];
        let end = row_ptrs[global_row + 1];

        for j in start..end {
            let global_col = col_indices[j];
            if global_col >= col_start && global_col < col_start + ncols {
                new_col_indices.push(global_col - col_start);
                new_values.push(values[j]);
            }
        }
        new_row_ptrs[i + 1] = new_values.len();
    }

    CsrMatrix::from_raw_parts(nrows, ncols, new_row_ptrs, new_col_indices, new_values)
}

/// Transpose a CSR matrix (counting sort by column index).
fn transpose_csr<T: ComplexField>(matrix: &CsrMatrix<T>) -> CsrMatrix<T> {
    let nrows = matrix.num_rows;
    let ncols = matrix.num_cols;
    let nnz = matrix.nnz();

    let mut row_ptrs = vec![0usize; ncols + 1];
    for &col in &matrix.col_indices {
        row_ptrs[col + 1] += 1;
    }
    for i in 0..ncols {
        row_ptrs[i + 1] += row_ptrs[i];
    }

    let mut col_indices = vec![0usize; nnz];
    let mut values = vec![T::zero(); nnz];
    let mut next = row_ptrs[..ncols].to_vec();
    for i in 0..nrows {
        for idx in matrix.row_ptrs[i]..matrix.row_ptrs[i + 1] {
            let col = matrix.col_indices[idx];
            let dest = next[col];
            col_indices[dest] = i;
            values[dest] = matrix.values[idx];
            next[col] += 1;
        }
    }

    CsrMatrix::from_raw_parts(ncols, nrows, row_ptrs, col_indices, values)
}

/// Assemble `S = B^T * diag(A)^{-1} * B` explicitly as a sparse matrix.
///
/// Rows of B are scaled by the inverse of the corresponding diagonal entry
/// of A; rows whose diagonal entry is exactly zero are zeroed out (the
/// inverse is undefined there, and dropping the row is the standard
/// safeguard for Dirichlet-constrained DOFs). The plain transpose `B^T` is
/// used (not `B^H`), matching the `C = B^T` coupling of symmetric saddle
/// point systems.
fn form_bt_diag_b<T: ComplexField>(a: &CsrMatrix<T>, b: &CsrMatrix<T>) -> CsrMatrix<T> {
    let diag = a.diagonal();

    // W * B: scale each row of B by the inverse diagonal of A.
    let mut wb_values = b.values.clone();
    for i in 0..b.num_rows {
        let d = diag.get(i).copied().unwrap_or_else(T::zero);
        let w = if d.norm_sqr() > T::Real::zero() {
            d.inv()
        } else {
            T::zero()
        };
        for slot in &mut wb_values[b.row_ptrs[i]..b.row_ptrs[i + 1]] {
            *slot *= w;
        }
    }
    let wb = CsrMatrix::from_raw_parts(
        b.num_rows,
        b.num_cols,
        b.row_ptrs.clone(),
        b.col_indices.clone(),
        wb_values,
    );

    // S = B^T * (W * B) via Gustavson SpGEMM.
    let schur = transpose_csr(b).matmul(&wb);
    repair_schur_zero_diagonal(schur)
}

/// Replace structurally zero diagonal entries of an assembled Schur
/// approximation with a unit diagonal.
///
/// Constrained pressure rows (for example a pinned pressure DOF whose
/// coupling column was zeroed by symmetric Dirichlet elimination) produce
/// empty product rows. The true Schur complement carries an identity row
/// there, so a unit diagonal matches its structure and keeps the block
/// factorizable for ILUT-style subsolvers instead of forcing a fabricated
/// pivot or a singular block.
fn repair_schur_zero_diagonal<T: ComplexField>(schur: CsrMatrix<T>) -> CsrMatrix<T> {
    let diagonal = schur.diagonal();
    if diagonal
        .iter()
        .all(|entry| entry.norm_sqr() > T::Real::zero())
    {
        return schur;
    }
    let mut triplets = Vec::with_capacity(schur.nnz() + schur.num_rows);
    for row in 0..schur.num_rows {
        for index in schur.row_range(row) {
            triplets.push((row, schur.col_indices[index], schur.values[index]));
        }
        if diagonal[row].norm_sqr() <= T::Real::zero() {
            triplets.push((row, row, T::one()));
        }
    }
    CsrMatrix::from_triplets(schur.num_rows, schur.num_cols, triplets)
}

/// Assemble the `BtDiagB` Schur approximation from a monolithic matrix.
///
/// Extracts the leading `block1_size` velocity block `A` and the coupling
/// block `B` from `matrix` and returns `B^T * diag(A)^{-1} * B` with the
/// same constrained-row repair [`SchurStrategy::BtDiagB`] applies
/// internally. Backend (GPU) saddle-point solvers upload this product as
/// their pressure Schur matrix so unsteady systems get the correctly
/// scaled approximation instead of a steady-Stokes mass matrix.
///
/// # Panics
///
/// Panics when `block1_size` exceeds the matrix dimensions.
pub fn btdiagb_schur_from_monolithic<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    block1_size: usize,
) -> CsrMatrix<T> {
    let n = matrix.num_rows;
    let m = n - block1_size;
    let block1 = extract_block(matrix, 0, 0, block1_size, block1_size);
    let coupling_b = extract_block(matrix, 0, block1_size, block1_size, m);
    form_bt_diag_b(&block1, &coupling_b)
}

/// Build a preconditioner from configuration.
fn build_preconditioner<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    config: &BlockPreconditionerConfig,
) -> Box<dyn Preconditioner<T> + Send + Sync> {
    match config {
        BlockPreconditionerConfig::Amg(amg_config) => {
            if matrix.num_rows == 0 || matrix.nnz() == 0 {
                Box::new(sonium_solvers_core::traits::IdentityPreconditioner)
            } else {
                Box::new(AmgPreconditioner::from_csr(matrix, amg_config.clone()))
            }
        }
        BlockPreconditionerConfig::Ilut(ilut_config) => {
            if matrix.num_rows == 0 || matrix.nnz() == 0 {
                Box::new(sonium_solvers_core::traits::IdentityPreconditioner)
            } else {
                Box::new(IlutPreconditioner::from_csr_with_config(
                    matrix,
                    ilut_config,
                ))
            }
        }
        BlockPreconditionerConfig::Ilu => {
            if matrix.num_rows == 0 || matrix.nnz() == 0 {
                Box::new(sonium_solvers_core::traits::IdentityPreconditioner)
            } else {
                Box::new(IluPreconditioner::from_csr(matrix))
            }
        }
        BlockPreconditionerConfig::Diagonal => {
            Box::new(crate::DiagonalPreconditioner::from_csr(matrix))
        }
        BlockPreconditionerConfig::None => {
            Box::new(sonium_solvers_core::traits::IdentityPreconditioner)
        }
    }
}

fn validate_checked_setup_inputs<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    block1_size: usize,
    schur_approx: Option<&CsrMatrix<T>>,
) -> Result<(), SolverError> {
    matrix.validate_structure()?;
    if let Some((index, _)) = matrix
        .values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.norm().is_finite())
    {
        return Err(SolverError::OperationFailed {
            operation: "saddle-point preconditioner setup",
            reason: format!("matrix contains a non-finite value at storage index {index}"),
        });
    }
    if matrix.num_rows != matrix.num_cols {
        return Err(SolverError::OperationFailed {
            operation: "saddle-point preconditioner setup",
            reason: format!(
                "matrix must be square, got {}x{}",
                matrix.num_rows, matrix.num_cols
            ),
        });
    }
    if block1_size == 0 || block1_size >= matrix.num_rows {
        return Err(SolverError::OperationFailed {
            operation: "saddle-point preconditioner setup",
            reason: format!(
                "block1_size must split both fields, got {} for {} dofs",
                block1_size, matrix.num_rows
            ),
        });
    }
    if let Some(schur_approx) = schur_approx {
        schur_approx.validate_structure()?;
        if let Some((index, _)) = schur_approx
            .values
            .iter()
            .enumerate()
            .find(|(_, value)| !value.norm().is_finite())
        {
            return Err(SolverError::OperationFailed {
                operation: "saddle-point external Schur setup",
                reason: format!(
                    "Schur approximation contains a non-finite value at storage index {index}"
                ),
            });
        }
        let block2_size = matrix.num_rows - block1_size;
        if schur_approx.num_rows != block2_size || schur_approx.num_cols != block2_size {
            return Err(SolverError::OperationFailed {
                operation: "saddle-point external Schur setup",
                reason: format!(
                    "Schur approximation shape is {}x{}, expected {}x{}",
                    schur_approx.num_rows, schur_approx.num_cols, block2_size, block2_size
                ),
            });
        }
    }
    Ok(())
}

/// Preflight the block-local setup paths used by the checked constructors.
///
/// The compatibility `from_matrix` constructors intentionally retain
/// best-effort ILUT regularization. A checked saddle setup must not hide that
/// diagnostic, so ILUT blocks are constructed through their checked API before
/// the legacy-compatible object is assembled.
fn validate_checked_block_setups<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    config: &SaddlePointConfig,
    assumptions: SaddlePointBlockAssumptions,
    schur_approx: Option<&CsrMatrix<T>>,
) -> Result<SaddlePointBlockValidation, SolverError> {
    let block1_size = config.block1_size;
    let block2_size = matrix.num_rows - block1_size;
    let block1 = extract_block(matrix, 0, 0, block1_size, block1_size);

    let block1_provenance = if config.schur_strategy != SchurStrategy::Simple {
        validate_checked_block_preconditioner(
            &block1,
            &config.block1_config,
            assumptions.block1,
            "saddle-point A block",
        )?
    } else {
        SaddlePointBlockProvenance::NotRequired
    };

    let block2 = match config.schur_strategy {
        SchurStrategy::External => schur_approx
            .ok_or_else(|| SolverError::OperationFailed {
                operation: "saddle-point external Schur setup",
                reason: "missing Schur approximation".into(),
            })?
            .clone(),
        SchurStrategy::BtDiagB | SchurStrategy::Simple => {
            let coupling_b = extract_block(matrix, 0, block1_size, block1_size, block2_size);
            form_bt_diag_b(&block1, &coupling_b)
        }
        SchurStrategy::Diagonal => {
            extract_block(matrix, block1_size, block1_size, block2_size, block2_size)
        }
        SchurStrategy::Pcd | SchurStrategy::Lsc => unreachable!(
            "specialized saddle-point strategies are rejected before checked block setup"
        ),
    };

    let block2_provenance = validate_checked_block_preconditioner(
        &block2,
        &config.block2_config,
        assumptions.block2,
        "saddle-point Schur block",
    )?;

    Ok(SaddlePointBlockValidation {
        block1: block1_provenance,
        block2: block2_provenance,
    })
}

fn validate_checked_block_preconditioner<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    config: &BlockPreconditionerConfig,
    assumption: SaddlePointBlockAssumption,
    operation: &'static str,
) -> Result<SaddlePointBlockProvenance, SolverError> {
    if matrix.num_rows == 0 || matrix.nnz() == 0 {
        return Ok(SaddlePointBlockProvenance::NotRequired);
    }

    match config {
        BlockPreconditionerConfig::Ilut(ilut_config) => {
            { IlutPreconditioner::try_from_csr_with_config(matrix, ilut_config).map(|_| ()) }
                .map_err(|error| SolverError::OperationFailed {
                    operation,
                    reason: error.to_string(),
                })
                .map(|_| SaddlePointBlockProvenance::NotRequired)
        }
        BlockPreconditionerConfig::Ilu => { IluPreconditioner::try_from_csr(matrix).map(|_| ()) }
            .map_err(|error| SolverError::OperationFailed {
                operation,
                reason: error.to_string(),
            })
            .map(|_| SaddlePointBlockProvenance::NotRequired),
        BlockPreconditionerConfig::Amg(_) => {
            validate_checked_spd_block(matrix, assumption, operation)
        }
        BlockPreconditionerConfig::Diagonal | BlockPreconditionerConfig::None => {
            Ok(SaddlePointBlockProvenance::NotRequired)
        }
    }
}

fn validate_checked_spd_block<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    assumption: SaddlePointBlockAssumption,
    operation: &'static str,
) -> Result<SaddlePointBlockProvenance, SolverError> {
    let zero = T::Real::zero();
    let one = T::Real::from_f64(1.0).unwrap();
    for row in 0..matrix.num_rows {
        let diagonal = (matrix.row_ptrs[row]..matrix.row_ptrs[row + 1])
            .find(|&index| matrix.col_indices[index] == row)
            .map(|index| matrix.values[index]);
        let Some(diagonal) = diagonal else {
            return Err(SolverError::OperationFailed {
                operation,
                reason: format!("SPD/HPD block row {row} has no diagonal"),
            });
        };
        if !diagonal.re().is_finite()
            || !diagonal.im().is_finite()
            || diagonal.re() <= zero
            || diagonal.im().abs() > T::Real::from_f64(1e-12).unwrap() * diagonal.norm().max(one)
        {
            return Err(SolverError::OperationFailed {
                operation,
                reason: format!("SPD/HPD block diagonal at row {row} is not real and positive"),
            });
        }

        for index in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            let col = matrix.col_indices[index];
            let value = matrix.values[index];
            if !value.re().is_finite() || !value.im().is_finite() {
                return Err(SolverError::OperationFailed {
                    operation,
                    reason: format!("SPD/HPD block contains a non-finite value at ({row},{col})"),
                });
            }
            let counterpart = (matrix.row_ptrs[col]..matrix.row_ptrs[col + 1])
                .find(|&candidate| matrix.col_indices[candidate] == row)
                .map(|candidate| matrix.values[candidate]);
            let Some(counterpart) = counterpart else {
                return Err(SolverError::OperationFailed {
                    operation,
                    reason: format!(
                        "SPD/HPD block lacks the symmetric/Hermitian counterpart for ({row},{col})"
                    ),
                });
            };
            let scale = value.norm().max(counterpart.norm()).max(one);
            if (value - counterpart.conj()).norm() > T::Real::from_f64(1e-10).unwrap() * scale {
                return Err(SolverError::OperationFailed {
                    operation,
                    reason: format!(
                        "SPD/HPD block entries ({row},{col}) and ({col},{row}) disagree"
                    ),
                });
            }
        }
    }

    if matrix.num_rows <= SADDLE_SPD_EXACT_VALIDATION_MAX_ROWS {
        SparseCholeskyFactorization::factorize_hermitian(matrix).map_err(|error| {
            SolverError::OperationFailed {
                operation,
                reason: format!("SPD/HPD block is not positive definite: {error}"),
            }
        })?;
        Ok(SaddlePointBlockProvenance::IndependentlyValidated)
    } else if assumption == SaddlePointBlockAssumption::PositiveDefinite {
        Ok(SaddlePointBlockProvenance::CallerDeclared)
    } else {
        Err(SolverError::OperationFailed {
            operation,
            reason: format!(
                "large SPD/HPD block with {} rows requires SaddlePointBlockAssumption::PositiveDefinite",
                matrix.num_rows
            ),
        })
    }
}

/// Utility to check if a matrix has saddle point structure.
///
/// Returns true when the (2,2) block is numerically zero (no stored entry
/// above a 1e-12 absolute tolerance) and at least one of the off-diagonal
/// coupling blocks B, C contains a nonzero entry.
pub fn has_saddle_point_structure(matrix: &CsrMatrix<f64>, block1_size: usize) -> bool {
    let n = matrix.num_rows;
    let m = n - block1_size;

    if block1_size == 0 || m == 0 || n != matrix.num_cols {
        return false;
    }

    let row_ptrs = &matrix.row_ptrs;
    let col_indices = &matrix.col_indices;
    let values = &matrix.values;

    // The (2,2) block must be (numerically) zero.
    const ZERO_BLOCK_TOL: f64 = 1e-12;
    for i in block1_size..n {
        let start = row_ptrs[i];
        let end = row_ptrs[i + 1];
        for j in start..end {
            if col_indices[j] >= block1_size && values[j].abs() > ZERO_BLOCK_TOL {
                return false;
            }
        }
    }

    let mut has_b = false;
    let mut has_c = false;

    for i in 0..block1_size {
        let start = row_ptrs[i];
        let end = row_ptrs[i + 1];
        for &col in &col_indices[start..end] {
            if col >= block1_size {
                has_b = true;
                break;
            }
        }
        if has_b {
            break;
        }
    }

    for i in block1_size..n {
        let start = row_ptrs[i];
        let end = row_ptrs[i + 1];
        for &col in &col_indices[start..end] {
            if col < block1_size {
                has_c = true;
                break;
            }
        }
        if has_c {
            break;
        }
    }

    has_b || has_c
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_solvers_sparse::CsrMatrix;

    #[test]
    fn test_block_extraction() {
        let dense = sonium_solvers_core::array![
            [1.0, 2.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 6.0],
            [0.0, 0.0, 7.0, 8.0]
        ];
        let matrix = CsrMatrix::from_dense(&dense, 1e-12);

        let block = extract_block(&matrix, 2, 2, 2, 2);
        assert_eq!(block.num_rows, 2);
        assert_eq!(block.num_cols, 2);
    }

    #[test]
    fn test_saddle_point_detection() {
        let dense = sonium_solvers_core::array![
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ];
        let saddle = CsrMatrix::from_dense(&dense, 1e-12);

        assert!(has_saddle_point_structure(&saddle, 2));
    }

    #[test]
    fn test_saddle_point_detection_zero_block() {
        // Nonzero (2,2) block: NOT a saddle point system, even with coupling.
        let with_d = sonium_solvers_core::array![
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ];
        let m = CsrMatrix::from_dense(&with_d, 1e-12);
        assert!(!has_saddle_point_structure(&m, 2));

        // Zero (2,2) block but no off-diagonal coupling: NOT saddle point.
        let no_coupling = sonium_solvers_core::array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ];
        let m = CsrMatrix::from_dense(&no_coupling, 1e-12);
        assert!(!has_saddle_point_structure(&m, 2));
    }

    #[test]
    fn test_bt_diag_b_assembly() {
        // A = [[4, 1], [1, 4]], B = [[1, 0], [0, 2]]
        // W = diag(A)^{-1} = diag(1/4, 1/4)
        // W*B = [[0.25, 0], [0, 0.5]]
        // S = B^T * W * B = [[0.25, 0], [0, 1.0]]
        let a = CsrMatrix::from_dense(
            &sonium_solvers_core::array![[4.0_f64, 1.0], [1.0, 4.0]],
            1e-12,
        );
        let b = CsrMatrix::from_dense(
            &sonium_solvers_core::array![[1.0_f64, 0.0], [0.0, 2.0]],
            1e-12,
        );

        let s = form_bt_diag_b(&a, &b);
        assert_eq!(s.num_rows, 2);
        assert_eq!(s.num_cols, 2);
        assert_eq!(s.nnz(), 2, "S should be diagonal here");
        let diag = s.diagonal();
        assert!((diag[0] - 0.25).abs() < 1e-12);
        assert!((diag[1] - 1.0).abs() < 1e-12);
    }

    /// Build a small Stokes-type saddle point system [A B; B^T 0] with
    /// b1 = 6 velocity DOFs and m = 2 pressure DOFs.
    fn stokes_test_system() -> (CsrMatrix<f64>, CsrMatrix<f64>, CsrMatrix<f64>, Vec<f64>) {
        // A: SPD tridiagonal (diag 4, off-diag -1), so diag(A) != A and the
        // Schur approximations are genuinely approximate.
        let a_dense = sonium_solvers_core::array![
            [4.0_f64, -1.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 4.0, -1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 4.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 4.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0, 4.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, -1.0, 4.0]
        ];
        let b_dense = sonium_solvers_core::array![
            [1.0_f64, 0.0],
            [0.0, 1.0],
            [-1.0, 0.5],
            [0.0, -1.0],
            [0.5, 0.0],
            [1.0, -0.5]
        ];
        let a = CsrMatrix::from_dense(&a_dense, 1e-12);
        let b = CsrMatrix::from_dense(&b_dense, 1e-12);

        // Monolithic 8x8 [A B; B^T 0]
        let full = sonium_solvers_core::array![
            [4.0_f64, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [-1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, -1.0, 4.0, -1.0, 0.0, 0.0, -1.0, 0.5],
            [0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0, -1.0, 4.0, 1.0, -0.5],
            [1.0, 0.0, -1.0, 0.0, 0.5, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.5, -1.0, 0.0, -0.5, 0.0, 0.0]
        ];
        let matrix = CsrMatrix::from_dense(&full, 1e-12);

        let rhs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5, -0.5];
        (matrix, a, b, rhs)
    }

    /// Nearly-exact ILUT (low drop tolerance, generous fill) so block solves
    /// are effectively exact for these tiny systems.
    fn nearly_exact_ilut() -> IlutConfig {
        IlutConfig::new(1e-14, 20, 1e-14)
    }

    #[test]
    fn test_stokes_gmres_schur_strategies() {
        use sonium_solvers_iterative::{GmresConfig, gmres, gmres_preconditioned};

        let (matrix, a, b, rhs) = stokes_test_system();
        assert!(has_saddle_point_structure(&matrix, 6));

        // Exact Schur complement S = B^T A^{-1} B, formed by solving
        // A X = B column-wise with tight GMRES (A is 6x6 SPD: exact in <=6).
        let tight = GmresConfig {
            max_iterations: 10,
            restart: 10,
            tolerance: 1e-13,
            ..Default::default()
        };
        let mut x_cols = Vec::new();
        for j in 0..2 {
            let col: Vec<f64> = (0..6).map(|i| b.get(i, j)).collect();
            let sol = gmres(&a, &col, &tight);
            assert!(sol.converged, "exact Schur column solve must converge");
            x_cols.push(sol.x);
        }
        let s = |k: usize, j: usize| -> f64 { (0..6).map(|i| b.get(i, k) * x_cols[j][i]).sum() };
        let schur_exact = CsrMatrix::from_dense(
            &sonium_solvers_core::array![[s(0, 0), s(0, 1)], [s(1, 0), s(1, 1)]],
            1e-12,
        );

        let gmres_config = GmresConfig {
            max_iterations: 60,
            restart: 60,
            tolerance: 1e-10,
            ..Default::default()
        };

        // 1) Exact Schur via External strategy.
        let exact_config = SaddlePointConfig {
            block1_size: 6,
            block1_config: BlockPreconditionerConfig::Ilut(nearly_exact_ilut()),
            block2_config: BlockPreconditionerConfig::Ilut(nearly_exact_ilut()),
            schur_strategy: SchurStrategy::External,
            upper_triangular: true,
        };
        let exact_precond = SaddlePointPreconditioner::<f64>::from_matrix_with_schur(
            &matrix,
            &schur_exact,
            exact_config,
        );
        let sol_exact = gmres_preconditioned(&matrix, &exact_precond, &rhs, &gmres_config);
        assert!(sol_exact.converged, "exact Schur GMRES must converge");

        // 2) BtDiagB Schur.
        let btdiagb_config = SaddlePointConfig {
            block1_size: 6,
            block1_config: BlockPreconditionerConfig::Ilut(nearly_exact_ilut()),
            block2_config: BlockPreconditionerConfig::Ilut(nearly_exact_ilut()),
            schur_strategy: SchurStrategy::BtDiagB,
            upper_triangular: true,
        };
        let btdiagb_precond =
            SaddlePointPreconditioner::<f64>::from_matrix(&matrix, btdiagb_config);
        let sol_btdiagb = gmres_preconditioned(&matrix, &btdiagb_precond, &rhs, &gmres_config);
        assert!(sol_btdiagb.converged, "BtDiagB GMRES must converge");

        // 3) SIMPLE Schur.
        let simple_config = SaddlePointConfig {
            block1_size: 6,
            block1_config: BlockPreconditionerConfig::Ilut(nearly_exact_ilut()),
            block2_config: BlockPreconditionerConfig::Ilut(nearly_exact_ilut()),
            schur_strategy: SchurStrategy::Simple,
            upper_triangular: true,
        };
        let simple_precond = SaddlePointPreconditioner::<f64>::from_matrix(&matrix, simple_config);
        let sol_simple = gmres_preconditioned(&matrix, &simple_precond, &rhs, &gmres_config);
        assert!(sol_simple.converged, "SIMPLE GMRES must converge");

        // Exact block-triangular preconditioning gives a minimal polynomial
        // of degree <= 2; approximate Schurs need more iterations but must
        // still converge quickly on this well-conditioned system.
        assert!(
            sol_exact.iterations <= sol_btdiagb.iterations,
            "exact ({}) should need no more iterations than BtDiagB ({})",
            sol_exact.iterations,
            sol_btdiagb.iterations
        );
        assert!(
            sol_btdiagb.iterations <= sol_simple.iterations,
            "BtDiagB ({}) should need no more iterations than SIMPLE ({})",
            sol_btdiagb.iterations,
            sol_simple.iterations
        );
        assert!(
            sol_simple.iterations <= 30,
            "SIMPLE took {} iterations on a tiny well-conditioned system",
            sol_simple.iterations
        );

        // All solutions must agree (same linear system).
        for i in 0..8 {
            assert!(
                (sol_exact.x[i] - sol_btdiagb.x[i]).abs() < 1e-6,
                "exact vs BtDiagB solution mismatch at {i}"
            );
            assert!(
                (sol_exact.x[i] - sol_simple.x[i]).abs() < 1e-6,
                "exact vs SIMPLE solution mismatch at {i}"
            );
        }
    }

    #[test]
    fn test_block_diagonal_preconditioner() {
        // Simple 4x4 saddle point: [A B; B^T 0]
        let dense = sonium_solvers_core::array![
            [4.0_f64, 1.0, 1.0, 0.0],
            [1.0, 4.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ];
        let matrix = CsrMatrix::from_dense(&dense, 1e-12);

        let config = SaddlePointConfig {
            block1_size: 2,
            block1_config: BlockPreconditionerConfig::Diagonal,
            block2_config: BlockPreconditionerConfig::None,
            schur_strategy: SchurStrategy::Diagonal,
            upper_triangular: true,
        };

        let precond = SaddlePointPreconditioner::<f64>::from_matrix(&matrix, config);
        let r = sonium_solvers_core::array![1.0_f64, 1.0, 1.0, 1.0];
        let result = precond.apply(&r);
        assert_eq!(result.len(), 4);
        // Block 1 (diagonal): [1/4, 1/4], Block 2 (identity): [1, 1]
        assert!((result[0] - 0.25).abs() < 1e-10);
        assert!((result[1] - 0.25).abs() < 1e-10);
        assert!((result[2] - 1.0).abs() < 1e-10);
        assert!((result[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_upper_triangular_preconditioner() {
        // [A B; B^T 0] with external Schur = M_p
        let dense =
            sonium_solvers_core::array![[4.0_f64, 0.0, 1.0], [0.0, 4.0, 1.0], [1.0, 1.0, 0.0]];
        let matrix = CsrMatrix::from_dense(&dense, 1e-12);

        // External Schur: identity (for simplicity)
        let schur = CsrMatrix::from_dense(&sonium_solvers_core::array![[1.0_f64]], 1e-12);

        let config = SaddlePointConfig {
            block1_size: 2,
            block1_config: BlockPreconditionerConfig::Diagonal,
            block2_config: BlockPreconditionerConfig::Diagonal,
            schur_strategy: SchurStrategy::External,
            upper_triangular: true,
        };

        let precond =
            SaddlePointPreconditioner::<f64>::from_matrix_with_schur(&matrix, &schur, config);
        let r = sonium_solvers_core::array![1.0_f64, 1.0, 1.0];
        let result = precond.apply(&r);
        assert_eq!(result.len(), 3);
        // y2 = S^{-1} * r2 = 1.0
        // y1 = A^{-1} * (r1 - B * y2) = diag(1/4) * ([1,1] - [1,1]*1) = [0, 0]
        assert!(result[0].abs() < 1e-10);
        assert!(result[1].abs() < 1e-10);
        assert!((result[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn pcd_setup_rejects_structurally_invalid_fp_up_front() {
        // Validate-once: the owned `fp` clone is applied unchecked on
        // every Schur-complement application, so malformed storage must
        // panic here instead of at the first apply.
        let dense =
            sonium_solvers_core::array![[4.0_f64, 0.0, 1.0], [0.0, 4.0, 1.0], [1.0, 1.0, 0.0]];
        let matrix = CsrMatrix::from_dense(&dense, 1e-12);
        let mut fp = CsrMatrix::from_dense(&sonium_solvers_core::array![[2.0_f64]], 1e-12);
        fp.row_ptrs[1] = 2;
        let blocks = PcdBlocks {
            fp,
            ap: CsrMatrix::from_dense(&sonium_solvers_core::array![[3.0_f64]], 1e-12),
            mp: CsrMatrix::from_dense(&sonium_solvers_core::array![[4.0_f64]], 1e-12),
        };
        let config = SaddlePointConfig {
            block1_size: 2,
            block1_config: BlockPreconditionerConfig::Diagonal,
            block2_config: BlockPreconditionerConfig::Diagonal,
            schur_strategy: SchurStrategy::Pcd,
            upper_triangular: true,
        };
        assert!(
            std::panic::catch_unwind(|| {
                SaddlePointPreconditioner::<f64>::from_matrix_with_pcd(&matrix, &blocks, config)
            })
            .is_err()
        );
    }

    #[test]
    fn test_pcd_apply_structure() {
        // [A B; B^T 0] with A = 4I (2x2), B = [1; 1].
        // Scalar PCD blocks: F_p = [2], A_p = [3], M_p = [4] with exact
        // (diagonal) inner solves: S^{-1} = M_p^{-1} F_p A_p^{-1} = 2/12 = 1/6.
        let dense =
            sonium_solvers_core::array![[4.0_f64, 0.0, 1.0], [0.0, 4.0, 1.0], [1.0, 1.0, 0.0]];
        let matrix = CsrMatrix::from_dense(&dense, 1e-12);
        let blocks = PcdBlocks {
            fp: CsrMatrix::from_dense(&sonium_solvers_core::array![[2.0_f64]], 1e-12),
            ap: CsrMatrix::from_dense(&sonium_solvers_core::array![[3.0_f64]], 1e-12),
            mp: CsrMatrix::from_dense(&sonium_solvers_core::array![[4.0_f64]], 1e-12),
        };
        let config = SaddlePointConfig {
            block1_size: 2,
            block1_config: BlockPreconditionerConfig::Diagonal,
            block2_config: BlockPreconditionerConfig::Diagonal,
            schur_strategy: SchurStrategy::Pcd,
            upper_triangular: true,
        };
        let precond =
            SaddlePointPreconditioner::<f64>::from_matrix_with_pcd(&matrix, &blocks, config);
        let r = sonium_solvers_core::array![1.0_f64, 1.0, 1.0];
        let result = precond.apply(&r);
        // y2 = (1/6) * 1; y1 = diag(1/4) * ([1,1] - [1,1]*y2) = 5/24 each.
        assert!((result[2] - 1.0 / 6.0).abs() < 1e-12, "y2 = {}", result[2]);
        assert!(
            (result[0] - 5.0 / 24.0).abs() < 1e-12,
            "y1[0] = {}",
            result[0]
        );
        assert!(
            (result[1] - 5.0 / 24.0).abs() < 1e-12,
            "y1[1] = {}",
            result[1]
        );
    }

    #[test]
    fn test_lsc_apply_structure() {
        // Same 3x3 system. W = I, F = A = 4I:
        // Q = B^T W B = [2], M = B^T W F W B = [8].
        // Exact (diagonal) Q solve: S^{-1} = Q^{-1} M Q^{-1} = 8/4 = 2.
        let dense =
            sonium_solvers_core::array![[4.0_f64, 0.0, 1.0], [0.0, 4.0, 1.0], [1.0, 1.0, 0.0]];
        let matrix = CsrMatrix::from_dense(&dense, 1e-12);
        let blocks = LscBlocks {
            velocity_mass_diag_inv: vec![1.0, 1.0],
        };
        let config = SaddlePointConfig {
            block1_size: 2,
            block1_config: BlockPreconditionerConfig::Diagonal,
            block2_config: BlockPreconditionerConfig::Diagonal,
            schur_strategy: SchurStrategy::Lsc,
            upper_triangular: true,
        };
        let precond =
            SaddlePointPreconditioner::<f64>::from_matrix_with_lsc(&matrix, &blocks, config);
        let r = sonium_solvers_core::array![1.0_f64, 1.0, 1.0];
        let result = precond.apply(&r);
        // y2 = 2 * 1; y1 = diag(1/4) * ([1,1] - [1,1]*2) = -1/4 each.
        assert!((result[2] - 2.0).abs() < 1e-12, "y2 = {}", result[2]);
        assert!((result[0] + 0.25).abs() < 1e-12, "y1[0] = {}", result[0]);
        assert!((result[1] + 0.25).abs() < 1e-12, "y1[1] = {}", result[1]);
    }

    /// Tridiagonal CSR matrix with constant bands.
    fn tridiag(n: usize, lower: f64, diag: f64, upper: f64) -> CsrMatrix<f64> {
        let mut triplets = Vec::new();
        for i in 0..n {
            if i > 0 {
                triplets.push((i, i - 1, lower));
            }
            triplets.push((i, i, diag));
            if i + 1 < n {
                triplets.push((i, i + 1, upper));
            }
        }
        CsrMatrix::from_triplets(n, n, triplets)
    }

    /// 1D Stokes-type system [F B; B^T 0] with `n_u` velocity nodes and
    /// `n_u - 1` pressure DOFs; B is the finite-difference gradient
    /// (`B[p,p] = 1`, `B[p+1,p] = -1`), so neighboring pressure columns of B
    /// overlap and `B^T B` is Laplacian-like — the regime where diagonal
    /// Schur approximations degrade with the mesh size.
    fn stokes_1d(n_u: usize) -> (CsrMatrix<f64>, CsrMatrix<f64>, CsrMatrix<f64>) {
        let n_p = n_u - 1;
        let f = tridiag(n_u, -1.0, 3.0, -1.0);
        let mut b_triplets = Vec::new();
        for p in 0..n_p {
            b_triplets.push((p, p, 1.0));
            b_triplets.push((p + 1, p, -1.0));
        }
        let b = CsrMatrix::from_triplets(n_u, n_p, b_triplets);

        let mut full_triplets = Vec::new();
        for i in 0..n_u {
            for idx in f.row_range(i) {
                full_triplets.push((i, f.col_indices[idx], f.values[idx]));
            }
            for idx in b.row_range(i) {
                let j = b.col_indices[idx];
                let v = b.values[idx];
                full_triplets.push((i, n_u + j, v));
                full_triplets.push((n_u + j, i, v));
            }
        }
        let full = CsrMatrix::from_triplets(n_u + n_p, n_u + n_p, full_triplets);
        (full, f, b)
    }

    #[test]
    fn test_pcd_lsc_gmres_beat_diagonal_schur() {
        use sonium_solvers_iterative::{GmresConfig, gmres_preconditioned};

        let n_u = 17;
        let n_p = n_u - 1;
        let (matrix, _f, _b) = stokes_1d(n_u);
        assert!(has_saddle_point_structure(&matrix, n_u));

        let rhs: Vec<f64> = (0..n_u + n_p)
            .map(|i| (i as f64 * 0.9).sin() + 0.5)
            .collect();
        let gmres_config = GmresConfig {
            max_iterations: 200,
            restart: n_u + n_p,
            tolerance: 1e-8,
            ..Default::default()
        };
        // Nearly-exact inner solves for both blocks: the comparison isolates
        // the quality of the Schur complement approximation itself.
        let make_config = |strategy: SchurStrategy| SaddlePointConfig {
            block1_size: n_u,
            block1_config: BlockPreconditionerConfig::Ilut(nearly_exact_ilut()),
            block2_config: BlockPreconditionerConfig::Ilut(nearly_exact_ilut()),
            schur_strategy: strategy,
            upper_triangular: true,
        };

        // Diagonal Schur baseline (B^T diag(A)^{-1} B: Laplacian-like here).
        let btdiagb = SaddlePointPreconditioner::<f64>::from_matrix(
            &matrix,
            make_config(SchurStrategy::BtDiagB),
        );
        let sol_btdiagb = gmres_preconditioned(&matrix, &btdiagb, &rhs, &gmres_config);
        assert!(sol_btdiagb.converged, "BtDiagB GMRES must converge");

        // PCD: the velocity block F = -Δ + I (reaction-diffusion), so the
        // commutator-consistent pressure operator is F_p = A_p + M_p.
        let ap = tridiag(n_p, -1.0, 2.0, -1.0);
        let fp = tridiag(n_p, -1.0, 3.0, -1.0);
        let pcd_blocks = PcdBlocks {
            fp,
            ap,
            mp: CsrMatrix::identity(n_p),
        };
        let pcd = SaddlePointPreconditioner::<f64>::from_matrix_with_pcd(
            &matrix,
            &pcd_blocks,
            make_config(SchurStrategy::Pcd),
        );
        let sol_pcd = gmres_preconditioned(&matrix, &pcd, &rhs, &gmres_config);
        assert!(sol_pcd.converged, "PCD GMRES must converge");

        // LSC with lumped velocity mass M_u = I.
        let lsc_blocks = LscBlocks {
            velocity_mass_diag_inv: vec![1.0; n_u],
        };
        let lsc = SaddlePointPreconditioner::<f64>::from_matrix_with_lsc(
            &matrix,
            &lsc_blocks,
            make_config(SchurStrategy::Lsc),
        );
        let sol_lsc = gmres_preconditioned(&matrix, &lsc, &rhs, &gmres_config);
        assert!(sol_lsc.converged, "LSC GMRES must converge");

        eprintln!(
            "Stokes GMRES iterations: BtDiagB = {}, PCD = {}, LSC = {}",
            sol_btdiagb.iterations, sol_pcd.iterations, sol_lsc.iterations
        );
        assert!(
            sol_pcd.iterations < sol_btdiagb.iterations,
            "PCD ({}) should need fewer iterations than BtDiagB ({})",
            sol_pcd.iterations,
            sol_btdiagb.iterations
        );
        assert!(
            sol_lsc.iterations < sol_btdiagb.iterations,
            "LSC ({}) should need fewer iterations than BtDiagB ({})",
            sol_lsc.iterations,
            sol_btdiagb.iterations
        );

        // All solutions must agree (same linear system).
        for i in 0..n_u + n_p {
            assert!(
                (sol_pcd.x[i] - sol_btdiagb.x[i]).abs() < 1e-5,
                "PCD vs BtDiagB solution mismatch at {i}"
            );
            assert!(
                (sol_lsc.x[i] - sol_btdiagb.x[i]).abs() < 1e-5,
                "LSC vs BtDiagB solution mismatch at {i}"
            );
        }
    }

    #[test]
    fn test_ilu_velocity_block_beats_diagonal_block() {
        use sonium_solvers_iterative::{GmresConfig, gmres_preconditioned};

        let n_u = 17;
        let n_p = n_u - 1;
        let (matrix, _f, _b) = stokes_1d(n_u);
        assert!(has_saddle_point_structure(&matrix, n_u));

        let rhs: Vec<f64> = (0..n_u + n_p)
            .map(|i| (i as f64 * 0.9).sin() + 0.5)
            .collect();
        let gmres_config = GmresConfig {
            max_iterations: 400,
            restart: n_u + n_p,
            tolerance: 1e-8,
            ..Default::default()
        };
        // Same BtDiagB Schur approximation on both sides: the comparison
        // isolates the velocity-block solver (ILU(0) vs Jacobi).
        let make_config = |block1: BlockPreconditionerConfig| SaddlePointConfig {
            block1_size: n_u,
            block1_config: block1,
            block2_config: BlockPreconditionerConfig::Diagonal,
            schur_strategy: SchurStrategy::BtDiagB,
            upper_triangular: true,
        };

        let diag = SaddlePointPreconditioner::<f64>::from_matrix(
            &matrix,
            make_config(BlockPreconditionerConfig::Diagonal),
        );
        let sol_diag = gmres_preconditioned(&matrix, &diag, &rhs, &gmres_config);
        assert!(sol_diag.converged, "diagonal-block GMRES must converge");

        let ilu = SaddlePointPreconditioner::<f64>::from_matrix(
            &matrix,
            make_config(BlockPreconditionerConfig::Ilu),
        );
        let sol_ilu = gmres_preconditioned(&matrix, &ilu, &rhs, &gmres_config);
        assert!(sol_ilu.converged, "ILU(0)-block GMRES must converge");

        eprintln!(
            "Stokes GMRES iterations: diagonal block = {}, ILU(0) block = {}",
            sol_diag.iterations, sol_ilu.iterations
        );
        assert!(
            sol_ilu.iterations < sol_diag.iterations,
            "ILU(0) block ({}) should need fewer iterations than diagonal block ({})",
            sol_ilu.iterations,
            sol_diag.iterations
        );
        for i in 0..n_u + n_p {
            assert!(
                (sol_ilu.x[i] - sol_diag.x[i]).abs() < 1e-5,
                "ILU(0) vs diagonal solution mismatch at {i}"
            );
        }
    }
}
