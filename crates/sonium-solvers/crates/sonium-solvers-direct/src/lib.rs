//! Direct solvers for linear systems
//!
//! - [`lu_solve`]: Dense LU decomposition with partial pivoting
//! - [`SparseLuFactorization`]: Sparse LU with COLAMD column ordering and
//!   partial pivoting
//! - [`sparse_cholesky::SparseCholeskyFactorization`]: Simplicial sparse
//!   Cholesky for SPD/HPD matrices (nested-dissection ordering, reusable
//!   symbolic analysis), with a true supernodal (BLAS-3) numeric path in
//!   [`sparse_cholesky_supernodal`] sharing the same symbolic phase, factor
//!   layout, and triangular solves.
//! - [`MultifrontalBlrFactorization`]: Multifrontal sparse LU with
//!   Block-Low-Rank (BLR) or HODLR compression of off-diagonal frontal
//!   panels, optional half-precision low-rank storage, and configurable
//!   pivoting (partial / rook / Bunch-Kaufman).
//! - [`MultifrontalLdltFactorization`]: Multifrontal sparse LDLᵀ / LDLᴴ for
//!   symmetric / Hermitian indefinite matrices — Bunch-Kaufman 1×1/2×2
//!   pivoting, one-triangle front storage, and the same delayed-pivoting
//!   machinery as the multifrontal LU path.

mod api;
pub(crate) mod banded_schur;
mod cholesky;
mod dense_qr;
mod expm;
#[cfg(feature = "cuda")]
pub mod gpu;
pub(crate) mod lu;
pub mod multifrontal_blr;
pub mod multifrontal_ldlt;
pub mod saddle_schur;
pub mod sparse_cholesky;
pub mod sparse_cholesky_supernodal;
pub mod sparse_lu;
mod sparse_supernodal;

pub use api::{
    DenseConfiguredFactorization, DenseDirectConfig, DenseDirectError, DenseDirectMethod,
    DenseFactorization, DenseRefinementResult, DirectFactorization, DirectSolveResult, MatrixClass,
    Ordering, SparseConfiguredFactorization, SparseConfiguredRefinementResult, SparseDirectConfig,
    SparseDirectError, SparseDirectMethod, SparseFactorization, SparseSymbolic, analyze_sparse,
    dense_backward_error, factorize_dense, factorize_dense_configured, factorize_sparse,
    factorize_sparse_configured,
};
pub use cholesky::{
    CholeskyError, CholeskyFactorization, cholesky_factorize, cholesky_factorize_hermitian,
    cholesky_factorize_in_place, cholesky_factorize_in_place_hermitian,
    cholesky_factorize_in_place_unchecked, cholesky_factorize_in_place_unchecked_hermitian,
    cholesky_factorize_unchecked, cholesky_factorize_unchecked_hermitian,
};
pub use dense_qr::{
    DenseSvd, QrError, QrFactorization, QrWorkspace, dense_svd, qr_factorize_with_workspace,
};
pub use expm::{ExpmError, expm};
#[cfg(feature = "cuda")]
pub use gpu::{
    GpuCholError, GpuDenseCholF64, GpuDenseLuF64, GpuDenseRhsF64, GpuLuError,
    cholesky_factorize_gpu, cholesky_solve_gpu, cholesky_solve_many_gpu, lu_factorize_gpu,
    lu_rhs_to_gpu, lu_solve_gpu, lu_solve_many_gpu, lu_solve_resident_gpu,
};
pub use lu::{
    LuError, LuFactorization, LuWorkspace, MixedLuSolution, lu_factorize, lu_factorize_in_place,
    lu_factorize_with_workspace, lu_solve, lu_solve_conj_transpose, lu_solve_mixed,
    lu_solve_transpose,
};
pub use multifrontal_blr::{
    BlrConfig, BlrError, IrConfig, IrError, IrResult, IterativeRefinement, LowRankPrecision,
    MixedPrecisionRefinement, MultifrontalBlrFactorization, MultifrontalSymbolic, PanelKind,
    PivotStrategy, UpdateOrder,
};
pub use multifrontal_ldlt::{
    LdltBlrConfig, LdltConfig, MatrixInertia, MultifrontalLdltFactorization,
};
pub use saddle_schur::{
    SaddleBlocks, SaddlePointSchurSolver, SaddleSchurError, saddle_blocks_from_monolithic,
};
pub use sparse_cholesky::{
    CholeskyOrdering, SPARSE_CHOLESKY_CHECKPOINT_FORMAT_VERSION, SparseCholeskyError,
    SparseCholeskyFactorization, SparseCholeskySymbolic,
};
pub use sparse_cholesky_supernodal::SupernodeMap;
pub use sparse_lu::{
    SPARSE_LU_CHECKPOINT_FORMAT_VERSION, SparseLuBackend, SparseLuError, SparseLuFactorization,
    SparseLuSymbolic, SparseRefinementProvenance, SparseRefinementResult,
    SparseRefinementStopReason,
};
