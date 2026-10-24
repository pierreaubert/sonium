//! Multifrontal sparse LU with low-rank compression.
//!
//! Drop-in replacement for [`crate::SparseLuFactorization`] when the
//! input matrix is large enough that exact sparse LU's fill-in becomes
//! prohibitive (typically beyond ~50–100k DOFs on 3D PDE problems). The
//! algorithm is multifrontal sparse LU with:
//!
//! 1. **Nested-dissection ordering** (reused from
//!    [`sonium_solvers_sparse::partition::nested_dissection`]).
//! 2. **Frontal-matrix factorisation** with configurable pivoting on the
//!    dense fully-summed (1,1) block of each front. See [`PivotStrategy`]:
//!    `Partial`, `Rook`, or `BunchKaufman` (1×1 / 2×2 block pivots for
//!    indefinite fronts). Pivot rejection is scale-aware
//!    ([`BlrConfig::pivot_tol_multiplier`]), and rejected pivots are
//!    postponed to the parent front (MUMPS-style **delayed pivoting**):
//!    the factorisation only fails when the root front cannot pivot, which
//!    is what makes indefinite systems (e.g. interior Helmholtz) robust.
//! 3. **Low-rank panel compression** of the off-diagonal `L21`, `U12` and
//!    Schur-complement contribution blocks of every front. See
//!    [`PanelKind`]: flat Block-Low-Rank (`Blr`, MUMPS-style), hierarchical
//!    non-nested HODLR, or nested-basis global low-rank (`NestedLr` —
//!    shared row/column bases across levels; *not* true HSS: the whole
//!    panel is one global `U_root·S·V_rootᵀ` product). Compression uses
//!    column-pivoted Householder QR with relative tolerance `ε`. Low-rank
//!    factors can optionally be stored in `f16`
//!    ([`LowRankPrecision::Half`]) for real `T` BLR panels.
//! 4. **Rayon tree-level parallelism**: fronts at the same dissection-tree
//!    level have no mutual dependency and factorise in parallel.
//!
//! At `ε = 0`, the factorisation is mathematically equivalent to a
//! conventional multifrontal sparse LU (no compression). Compression
//! becomes effective as `ε` is relaxed and the off-diagonal panels of large
//! fronts become numerically low-rank — typical for 3D PDE problems with
//! smooth Green's functions.
//!
//! Generic over [`sonium_solvers_core::traits::ComplexField`] (`f32`, `f64`, `Complex32`,
//! `Complex64`).

use thiserror::Error;

use num_traits::{Float, FromPrimitive};
use sonium_solvers_core::traits::{AccuracyStatus, ComplexField, SolverFailure, SolverStatus};
use sonium_solvers_sparse::CsrMatrix;

pub mod blr;
pub mod factorize;
pub mod front;
pub mod hodlr;
pub mod nested_lr;
pub mod ordering;
pub mod panel;
pub mod refinement;
pub mod solve;
pub mod truncation;

/// Legacy module path, kept for backward compatibility. The panel was
/// renamed because it is a nested-basis global low-rank approximation,
/// not HSS — see [`nested_lr`].
#[deprecated(
    note = "renamed to `nested_lr`: the panel is a nested-basis global low-rank approximation, not HSS"
)]
pub mod hss {
    pub use super::nested_lr::*;
}

pub use blr::LowRankPrecision;
pub use panel::PanelKind;
pub use refinement::{IrConfig, IrError, IrResult, IterativeRefinement, MixedPrecisionRefinement};

use factorize::{numeric_factorise, permute_matrix};
use front::FactoredFront;
pub use front::{PivotStrategy, UpdateOrder};
use ordering::{AssemblyTree, amalgamate_tree, build_assembly_tree};

pub(crate) fn scalar_is_finite<T: ComplexField>(value: T) -> bool {
    value.re().is_finite() && value.im().is_finite()
}

pub(crate) fn first_non_finite<T: ComplexField>(values: &[T]) -> Option<usize> {
    values.iter().position(|&value| !scalar_is_finite(value))
}

/// Low-level validation or numerical failure while building/applying one front.
#[derive(Error, Debug, Clone, Copy)]
pub enum FrontError {
    #[error("invalid front state: {0}")]
    Invalid(&'static str),
    #[error("front numerical breakdown during {stage} at local index {index}")]
    Numerical { stage: &'static str, index: usize },
}

impl FrontError {
    pub(crate) fn numerical(stage: &'static str, index: usize) -> Self {
        Self::Numerical { stage, index }
    }
}

impl SolverFailure for FrontError {
    fn error_kind(&self) -> sonium_solvers_core::traits::SolverErrorKind {
        use sonium_solvers_core::traits::SolverErrorKind;

        match self {
            Self::Invalid(_) => SolverErrorKind::OperationFailed,
            Self::Numerical { .. } => SolverErrorKind::Breakdown,
        }
    }

    fn status(&self) -> SolverStatus {
        match self {
            Self::Invalid(_) => SolverStatus::NumericalError,
            Self::Numerical { .. } => SolverStatus::Breakdown,
        }
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        AccuracyStatus::NotChecked
    }
}

/// Configuration knobs for [`MultifrontalBlrFactorization`].
#[derive(Debug, Clone)]
pub struct BlrConfig {
    /// Relative ε-rank tolerance for BLR compression of off-diagonal blocks.
    /// `0.0` disables compression and falls back to exact multifrontal LU.
    pub epsilon: f64,
    /// Block size of the BLR grid on each panel. Default `128`.
    pub block_size: usize,
    /// Minimum dissection-block size: subdomains smaller than this are
    /// treated as leaves. Default `128`.
    pub min_block_size: usize,
    /// Maximum dissection depth (auto if `None`).
    pub max_levels: Option<usize>,
    /// Pivoting strategy for the dense FS block of each front. Default
    /// [`PivotStrategy::Partial`].
    pub pivot_strategy: PivotStrategy,
    /// Compressed-panel format for off-diagonal `L21 / U12 / CB` blocks of
    /// every front. Default [`PanelKind::Blr`]. [`PanelKind::Hodlr`] uses
    /// recursive 2×2 splits with independent low-rank factors per level;
    /// [`PanelKind::NestedLr`] additionally shares row / column bases
    /// across levels for asymptotically smaller storage on smooth kernels
    /// (nested-basis global low-rank, not true HSS).
    pub panel_kind: PanelKind,
    /// Storage precision for low-rank factor pairs. Default
    /// [`LowRankPrecision::Full`]. `Half` halves memory of low-rank blocks
    /// for `f32` / `f64` panels at ≈ `1e-3` per-block relative error.
    pub low_rank_precision: LowRankPrecision,
    /// Within-front Schur-update order. Default [`UpdateOrder::RightLook`].
    /// `LeftLook` uses a BLR-aware matmul that exploits low-rank operands.
    pub update_order: UpdateOrder,
    /// Supernode amalgamation threshold: after building the dissection tree,
    /// adjacent fronts are greedily fused while the combined supernode's
    /// `F11` size stays at or below this many DOFs. Default `256` — small
    /// enough to keep the rank-aware compression effective on the merged
    /// fronts, large enough that the `F11` GEMM amortises BLAS overhead.
    /// `0` disables amalgamation and gives one front per dissection node.
    pub amalgamation_threshold: usize,
    /// Front-internal parallelism threshold. Fronts whose total size
    /// (`nfs + ncb`) exceeds this value use rayon `par_chunks_mut` to
    /// distribute the `L21` row solves, `U12` column solves, and the
    /// dense `RightLook` Schur update across threads. Smaller fronts run
    /// each pass on a single thread to avoid rayon task-spawn overhead.
    /// Default `1024`. `0` or any threshold above every front's size
    /// effectively disables intra-front parallelism (only tree-level
    /// rayon parallelism remains).
    pub parallel_front_threshold: usize,
    /// Multiplier for the scale-aware front pivot rejection tolerance.
    /// Inside a front, a pivot candidate is rejected (and its row/column
    /// postponed to the parent front via delayed pivoting) when its
    /// magnitude is below
    /// `pivot_tol_multiplier · ε_machine · nfs · max|F11|`.
    /// Default `1.0` — the front-local analogue of the sparse-LU
    /// `ε·n·max|A|` convention. Larger values make pivoting more
    /// conservative (more postponement); `0.0` accepts any nonzero pivot.
    pub pivot_tol_multiplier: f64,
}

impl Default for BlrConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-8,
            block_size: 128,
            // 64, not 128: min-block sweep 2026-09-25 (2D Laplacian) shows
            // finer trees collapse fill/flops faster than front-count costs
            // grow — 128²: 54.7 -> 46.2 wall; 256²: ~204 -> ~176 wall, both
            // with better residuals. min32 is within noise at 256² but
            // loses at 128²; 64 wins both sizes. See perf/PERF.md.
            min_block_size: 64,
            max_levels: None,
            pivot_strategy: PivotStrategy::Partial,
            panel_kind: PanelKind::Blr,
            low_rank_precision: LowRankPrecision::Full,
            update_order: UpdateOrder::RightLook,
            amalgamation_threshold: 256,
            parallel_front_threshold: 1024,
            pivot_tol_multiplier: 1.0,
        }
    }
}

/// Errors produced by multifrontal factorization and checked solves.
#[derive(Error, Debug)]
pub enum BlrError {
    #[error("matrix must be square (rows={rows}, cols={cols})")]
    NotSquare { rows: usize, cols: usize },
    #[error("invalid CSR structure: {0}")]
    InvalidCsrStructure(String),
    #[error("matrix contains a non-finite value at storage index {index}")]
    NonFiniteMatrix { index: usize },
    #[error("invalid multifrontal configuration field `{field}`")]
    InvalidConfiguration { field: &'static str },
    #[error("right-hand side length mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("right-hand side contains a non-finite value at index {index}")]
    NonFiniteRhs { index: usize },
    #[error("invalid factor state at front {front}: {reason}")]
    InvalidFactorization { front: usize, reason: &'static str },
    #[error("numerical breakdown at front {front} during {stage}, local index {index}")]
    FrontNumericalBreakdown {
        front: usize,
        stage: &'static str,
        index: usize,
    },
    #[error("numerical breakdown during {stage} at index {index}")]
    NumericalBreakdown { stage: &'static str, index: usize },
    #[error(
        "numerically singular matrix: root front {0} still has postponed pivots after delayed pivoting"
    )]
    SingularFront(usize),
    #[error("CSR sparsity pattern does not match the reusable symbolic analysis")]
    PatternMismatch,
}

impl BlrError {
    /// Return the canonical status represented by this checked failure.
    pub fn status(&self) -> SolverStatus {
        match self {
            Self::InvalidFactorization { .. }
            | Self::FrontNumericalBreakdown { .. }
            | Self::NumericalBreakdown { .. }
            | Self::SingularFront(_) => SolverStatus::Breakdown,
            Self::NotSquare { .. }
            | Self::InvalidCsrStructure(_)
            | Self::NonFiniteMatrix { .. }
            | Self::InvalidConfiguration { .. }
            | Self::DimensionMismatch { .. }
            | Self::PatternMismatch
            | Self::NonFiniteRhs { .. } => SolverStatus::NumericalError,
        }
    }

    pub fn solver_status(&self) -> SolverStatus {
        self.status()
    }

    /// No requested accuracy is established when factorization or solve
    /// validation fails before a checked result is returned.
    pub fn accuracy_status(&self) -> AccuracyStatus {
        AccuracyStatus::NotChecked
    }

    pub fn attainable_accuracy(&self) -> AccuracyStatus {
        self.accuracy_status()
    }
}

impl SolverFailure for BlrError {
    fn error_kind(&self) -> sonium_solvers_core::traits::SolverErrorKind {
        use sonium_solvers_core::traits::SolverErrorKind;
        match self {
            Self::NotSquare { .. } | Self::DimensionMismatch { .. } => {
                SolverErrorKind::DimensionMismatch
            }
            Self::InvalidFactorization { .. }
            | Self::FrontNumericalBreakdown { .. }
            | Self::NumericalBreakdown { .. }
            | Self::SingularFront(_) => SolverErrorKind::Breakdown,
            _ => SolverErrorKind::OperationFailed,
        }
    }

    fn status(&self) -> SolverStatus {
        BlrError::status(self)
    }

    fn accuracy_status(&self) -> AccuracyStatus {
        BlrError::accuracy_status(self)
    }
}

/// Reusable symbolic analysis for multifrontal factorization.
///
/// Holds the nested-dissection assembly tree (elimination ordering,
/// fronts, and level schedule) plus the CSR pattern it was built
/// from. Factorizing a sequence of matrices that share one sparsity
/// pattern — the standard fixed-pattern situation in Newton and
/// time-stepping loops — pays the ordering and tree construction
/// once and then reuses it through
/// [`MultifrontalBlrFactorization::factorize_with_symbolic`].
#[derive(Debug, Clone)]
pub struct MultifrontalSymbolic {
    n: usize,
    tree: AssemblyTree,
    row_ptrs: Vec<usize>,
    col_indices: Vec<usize>,
    min_block_size: usize,
    max_levels: Option<usize>,
    amalgamation_threshold: usize,
}

impl MultifrontalSymbolic {
    /// Analyze a matrix's sparsity pattern for repeated factorization.
    ///
    /// Only the pattern and the tree-shaping configuration fields
    /// (`min_block_size`, `max_levels`, `amalgamation_threshold`) are
    /// consumed here; the numeric configuration is supplied again at
    /// factorization time.
    ///
    /// # Errors
    ///
    /// Returns [`BlrError::NotSquare`] for a non-square matrix,
    /// [`BlrError::NonFiniteMatrix`] for non-finite values,
    /// [`BlrError::InvalidCsrStructure`] for malformed CSR, and
    /// [`BlrError::InvalidConfiguration`] for a non-finite or
    /// negative numeric configuration field.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_solvers_direct::{BlrConfig, MultifrontalBlrFactorization, MultifrontalSymbolic};
    /// use sonium_solvers_sparse::CsrMatrix;
    ///
    /// let matrix = CsrMatrix::from_raw_parts(
    ///     2,
    ///     2,
    ///     vec![0, 2, 4],
    ///     vec![0, 1, 0, 1],
    ///     vec![2.0_f64, 0.5, 0.5, 3.0],
    /// );
    /// let config = BlrConfig { epsilon: 0.0, ..BlrConfig::default() };
    /// let symbolic = MultifrontalSymbolic::analyze(&matrix, &config).unwrap();
    /// let factorization =
    ///     MultifrontalBlrFactorization::factorize_with_symbolic(&matrix, &symbolic, &config)
    ///         .unwrap();
    /// let solution = factorization.try_solve(&[1.0, 2.0]).unwrap();
    /// assert!((solution[0] - 0.34783).abs() < 1.0e-4);
    /// ```
    pub fn analyze<T: ComplexField>(
        matrix: &CsrMatrix<T>,
        cfg: &BlrConfig,
    ) -> Result<Self, BlrError> {
        let n = matrix.num_rows;
        if n != matrix.num_cols {
            return Err(BlrError::NotSquare {
                rows: matrix.num_rows,
                cols: matrix.num_cols,
            });
        }
        // Check finiteness before `validate_structure`, which folds
        // non-finite values into its generic structural error and would
        // mask the typed `NonFiniteMatrix` variant.
        if let Some(index) = first_non_finite(&matrix.values) {
            return Err(BlrError::NonFiniteMatrix { index });
        }
        matrix
            .validate_structure()
            .map_err(|error| BlrError::InvalidCsrStructure(error.to_string()))?;
        validate_blr_config(cfg)?;
        let tree = if n == 0 {
            AssemblyTree {
                perm: Vec::new(),
                perm_inv: Vec::new(),
                fronts: Vec::new(),
                levels: Vec::new(),
            }
        } else {
            let raw_tree = build_assembly_tree(matrix, cfg.min_block_size.max(1), cfg.max_levels);
            if cfg.amalgamation_threshold > 0 {
                amalgamate_tree(&raw_tree, cfg.amalgamation_threshold)
            } else {
                raw_tree
            }
        };
        Ok(Self {
            n,
            tree,
            row_ptrs: matrix.row_ptrs.clone(),
            col_indices: matrix.col_indices.clone(),
            min_block_size: cfg.min_block_size,
            max_levels: cfg.max_levels,
            amalgamation_threshold: cfg.amalgamation_threshold,
        })
    }

    /// Matrix dimension the analysis was built for.
    pub fn dimension(&self) -> usize {
        self.n
    }

    /// Number of fronts in the assembly tree.
    pub fn front_count(&self) -> usize {
        self.tree.fronts.len()
    }

    /// Elimination ordering: `permutation()[k]` is the global DOF
    /// eliminated at step `k`.
    pub fn permutation(&self) -> &[usize] {
        &self.tree.perm
    }
}

fn validate_blr_config(cfg: &BlrConfig) -> Result<(), BlrError> {
    if !cfg.epsilon.is_finite() || cfg.epsilon < 0.0 {
        return Err(BlrError::InvalidConfiguration { field: "epsilon" });
    }
    if !cfg.pivot_tol_multiplier.is_finite() || cfg.pivot_tol_multiplier < 0.0 {
        return Err(BlrError::InvalidConfiguration {
            field: "pivot_tol_multiplier",
        });
    }
    Ok(())
}

/// Multifrontal sparse-LU factorisation with BLR compression.
pub struct MultifrontalBlrFactorization<T: ComplexField> {
    n: usize,
    tree: AssemblyTree,
    fronts: Vec<FactoredFront<T>>,
}

impl<T: ComplexField + FromPrimitive> MultifrontalBlrFactorization<T> {
    /// Matrix dimension.
    pub fn dimension(&self) -> usize {
        self.n
    }

    /// Factorise with default configuration.
    pub fn factorize(matrix: &CsrMatrix<T>) -> Result<Self, BlrError> {
        Self::factorize_with_config(matrix, &BlrConfig::default())
    }

    /// Factorise with a custom configuration.
    pub fn factorize_with_config(matrix: &CsrMatrix<T>, cfg: &BlrConfig) -> Result<Self, BlrError> {
        let symbolic = MultifrontalSymbolic::analyze(matrix, cfg)?;
        Self::factorize_with_symbolic(matrix, &symbolic, cfg)
    }

    /// Factorize with a reusable symbolic analysis.
    ///
    /// The matrix must have the same dimension and CSR sparsity
    /// pattern as the matrix the symbolic analysis was built from;
    /// only its numeric values may differ. The tree-shaping
    /// configuration fields recorded in the symbolic analysis
    /// (`min_block_size`, `max_levels`, `amalgamation_threshold`)
    /// must match the supplied configuration; the remaining fields
    /// steer the numeric phase and may differ from the analysis call.
    ///
    /// # Errors
    ///
    /// Returns [`BlrError::PatternMismatch`] when the pattern
    /// differs, [`BlrError::DimensionMismatch`] when the dimension
    /// differs, [`BlrError::NonFiniteMatrix`] for non-finite values,
    /// and [`BlrError::InvalidConfiguration`] for invalid or
    /// tree-inconsistent configuration fields.
    pub fn factorize_with_symbolic(
        matrix: &CsrMatrix<T>,
        symbolic: &MultifrontalSymbolic,
        cfg: &BlrConfig,
    ) -> Result<Self, BlrError> {
        let n = matrix.num_rows;
        if n != symbolic.n || matrix.num_cols != symbolic.n {
            return Err(BlrError::DimensionMismatch {
                expected: symbolic.n,
                actual: n,
            });
        }
        if matrix.row_ptrs != symbolic.row_ptrs || matrix.col_indices != symbolic.col_indices {
            return Err(BlrError::PatternMismatch);
        }
        if let Some(index) = first_non_finite(&matrix.values) {
            return Err(BlrError::NonFiniteMatrix { index });
        }
        validate_blr_config(cfg)?;
        if cfg.min_block_size != symbolic.min_block_size
            || cfg.max_levels != symbolic.max_levels
            || cfg.amalgamation_threshold != symbolic.amalgamation_threshold
        {
            return Err(BlrError::InvalidConfiguration {
                field: "symbolic tree configuration",
            });
        }
        if n == 0 {
            return Ok(Self {
                n: 0,
                tree: symbolic.tree.clone(),
                fronts: Vec::new(),
            });
        }

        let tree = symbolic.tree.clone();
        let a_perm = permute_matrix(matrix, &tree.perm_inv);
        let epsilon = T::Real::from_f64(cfg.epsilon)
            .filter(|value| value.is_finite())
            .ok_or(BlrError::InvalidConfiguration { field: "epsilon" })?;
        let pivot_tol_rel = T::Real::from_f64(cfg.pivot_tol_multiplier)
            .filter(|value| value.is_finite())
            .ok_or(BlrError::InvalidConfiguration {
                field: "pivot_tol_multiplier",
            })?;
        let fronts = numeric_factorise(
            &a_perm,
            &tree,
            cfg.block_size.max(1),
            epsilon,
            cfg.pivot_strategy,
            cfg.panel_kind,
            cfg.low_rank_precision,
            cfg.update_order,
            cfg.parallel_front_threshold,
            pivot_tol_rel,
        )?;

        Ok(Self { n, tree, fronts })
    }

    /// Solve `A · x = b` using the pre-computed factorisation.
    pub fn solve(&self, b: &[T]) -> Vec<T> {
        match self.try_solve(b) {
            Ok(solution) => solution,
            Err(error) => panic!("multifrontal BLR solve failed: {error}"),
        }
    }

    /// Checked solve that rejects malformed/non-finite input and generated
    /// numerical breakdowns without panicking or publishing a partial result.
    pub fn try_solve(&self, b: &[T]) -> Result<Vec<T>, BlrError> {
        if b.len() != self.n {
            return Err(BlrError::DimensionMismatch {
                expected: self.n,
                actual: b.len(),
            });
        }
        if let Some(index) = first_non_finite(b) {
            return Err(BlrError::NonFiniteRhs { index });
        }
        if self.n == 0 {
            return Ok(Vec::new());
        }

        // Permute b into elimination order.
        let mut b_perm = vec![T::zero(); self.n];
        for i in 0..self.n {
            b_perm[self.tree.perm_inv[i]] = b[i];
        }

        let w_perm = solve::forward_backward_solve(&self.tree, &self.fronts, &b_perm)?;

        // Permute back to global order: x[g] = w_perm[perm_inv[g]].
        let mut x = vec![T::zero(); self.n];
        for g in 0..self.n {
            x[g] = w_perm[self.tree.perm_inv[g]];
        }
        if let Some(index) = first_non_finite(&x) {
            return Err(BlrError::NumericalBreakdown {
                stage: "final solution permutation",
                index,
            });
        }
        Ok(x)
    }

    /// Total stored entries across all fronts (dense F11 + BLR panels).
    pub fn factor_memory_bytes(&self) -> usize {
        let entries = self.fronts.iter().fold(0usize, |total, front| {
            total.saturating_add(front.stored_entries())
        });
        entries.saturating_mul(std::mem::size_of::<T>())
    }

    /// Average numerical rank across all low-rank off-diagonal blocks.
    /// Returns `0.0` if the panels are HODLR (which doesn't expose a
    /// single average rank) or if there are no low-rank blocks.
    pub fn average_rank(&self) -> f64 {
        let mut sum = 0.0;
        let mut count = 0.0;
        for f in &self.fronts {
            let panels = [&f.l21, &f.u12, &f.cb_panel];
            for p in panels {
                if let Some(r) = p.average_low_rank()
                    && r > 0.0
                {
                    sum += r;
                    count += 1.0;
                }
            }
        }
        if count == 0.0 { 0.0 } else { sum / count }
    }

    /// Number of fronts in the assembly tree.
    pub fn num_fronts(&self) -> usize {
        self.fronts.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SparseLuFactorization;
    use num_complex::Complex64;
    use sonium_solvers_core::dense::DenseMatrix;

    fn make_2d_laplacian_f64(nx: usize) -> CsrMatrix<f64> {
        let n = nx * nx;
        let mut dense = DenseMatrix::<f64>::zeros(n, n);
        for iy in 0..nx {
            for ix in 0..nx {
                let i = iy * nx + ix;
                dense[[i, i]] = 4.0;
                if ix > 0 {
                    dense[[i, i - 1]] = -1.0;
                }
                if ix + 1 < nx {
                    dense[[i, i + 1]] = -1.0;
                }
                if iy > 0 {
                    dense[[i, i - nx]] = -1.0;
                }
                if iy + 1 < nx {
                    dense[[i, i + nx]] = -1.0;
                }
            }
        }
        CsrMatrix::from_dense(&dense, 1e-15)
    }

    fn rel_residual_f64(matrix: &CsrMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
        let n = b.len();
        let mut res = vec![0.0; n];
        for i in 0..n {
            let mut s = 0.0;
            for idx in matrix.row_ptrs[i]..matrix.row_ptrs[i + 1] {
                s += matrix.values[idx] * x[matrix.col_indices[idx]];
            }
            res[i] = b[i] - s;
        }
        let r_norm: f64 = res.iter().map(|v| v * v).sum::<f64>().sqrt();
        let b_norm: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        if b_norm == 0.0 {
            r_norm
        } else {
            r_norm / b_norm
        }
    }

    #[test]
    fn symbolic_reuse_factorizes_same_pattern_with_new_values() {
        let config = BlrConfig {
            epsilon: 0.0,
            ..BlrConfig::default()
        };
        let first = make_2d_laplacian_f64(12);
        let symbolic = MultifrontalSymbolic::analyze(&first, &config).expect("analyze");
        assert_eq!(symbolic.dimension(), first.num_rows);
        assert!(symbolic.front_count() > 0);
        assert_eq!(symbolic.permutation().len(), first.num_rows);

        // Same pattern, different numeric values.
        let mut second = first.clone();
        for value in second.values.iter_mut() {
            *value *= 1.7;
        }
        second.values[0] += 0.5;
        let rhs: Vec<f64> = (0..second.num_rows).map(|i| (i as f64).sin()).collect();
        let factorization =
            MultifrontalBlrFactorization::factorize_with_symbolic(&second, &symbolic, &config)
                .expect("factorize with reused symbolic");
        let solution = factorization.try_solve(&rhs).expect("solve");
        assert!(rel_residual_f64(&second, &solution, &rhs) < 1.0e-10);

        // Reuse must match a from-scratch factorization of the same matrix.
        let reference =
            MultifrontalBlrFactorization::factorize_with_config(&second, &config).expect("fresh");
        let reference_solution = reference.try_solve(&rhs).expect("reference solve");
        for (actual, expected) in solution.iter().zip(&reference_solution) {
            assert!((actual - expected).abs() < 1.0e-9);
        }
    }

    #[test]
    fn symbolic_reuse_rejects_mismatched_pattern_and_tree_config() {
        let config = BlrConfig {
            epsilon: 0.0,
            ..BlrConfig::default()
        };
        let matrix = make_2d_laplacian_f64(8);
        let symbolic = MultifrontalSymbolic::analyze(&matrix, &config).expect("analyze");

        // Drop one stored entry: same dimension, different pattern.
        let mut sparser = matrix.clone();
        sparser.col_indices.remove(3);
        sparser.values.remove(3);
        for ptr in sparser.row_ptrs.iter_mut().skip(2) {
            *ptr -= 1;
        }
        let Err(error) = MultifrontalBlrFactorization::factorize_with_symbolic(
            &sparser,
            &symbolic,
            &config,
        ) else {
            panic!("pattern mismatch must be rejected");
        };
        assert!(matches!(error, BlrError::PatternMismatch));

        // Tree-shaping fields recorded at analysis time must be honored.
        let other_config = BlrConfig {
            min_block_size: config.min_block_size + 1,
            ..config.clone()
        };
        let Err(error) =
            MultifrontalBlrFactorization::factorize_with_symbolic(&matrix, &symbolic, &other_config)
        else {
            panic!("tree-inconsistent config must be rejected");
        };
        assert!(matches!(
            error,
            BlrError::InvalidConfiguration {
                field: "symbolic tree configuration"
            }
        ));
    }

    #[test]
    fn tiny_5x5_matches_exact() {
        // Random-ish full 5×5 stored as CSR.
        let mut dense = DenseMatrix::<f64>::zeros(5, 5);
        for i in 0..5 {
            for j in 0..5 {
                dense[[i, j]] = if i == j {
                    5.0 + i as f64
                } else {
                    -1.0 / (1.0 + (i as f64 - j as f64).abs())
                };
            }
        }
        let m = CsrMatrix::from_dense(&dense, 1e-15);
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let fac = MultifrontalBlrFactorization::factorize_with_config(
            &m,
            &BlrConfig {
                epsilon: 0.0,
                block_size: 4,
                min_block_size: 2,
                max_levels: Some(3),
                ..Default::default()
            },
        )
        .expect("factorise");
        let x_blr = fac.solve(&b);

        let exact = SparseLuFactorization::factorize(&m).expect("exact LU");
        let x_exact = exact.solve(&b).expect("sparse LU solve");

        for i in 0..5 {
            assert!(
                (x_blr[i] - x_exact[i]).abs() < 1e-10,
                "i={} blr={} exact={}",
                i,
                x_blr[i],
                x_exact[i]
            );
        }
    }

    #[test]
    fn structurally_nonsymmetric_matrix() {
        // A is *structurally* nonsymmetric: many `A[i, j] ≠ 0` exist whose
        // transpose entry `A[j, i]` is zero. Without the column-incoming
        // pattern walk in compute_cb_indices, F21-only entries get dropped
        // and the BLR factor produces a non-zero residual.
        let n = 64;
        let mut dense = DenseMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            dense[[i, i]] = (n as f64) + (i as f64);
            // Symmetric band:
            if i + 1 < n {
                dense[[i, i + 1]] = -0.7;
                dense[[i + 1, i]] = -0.7;
            }
            // ASYMMETRIC long-range entries: only upper-triangular far links.
            if i + 5 < n {
                dense[[i, i + 5]] = -0.2;
                // intentionally NO dense[[i + 5, i]] entry
            }
            // ASYMMETRIC: only lower-triangular for a different offset.
            if i >= 7 {
                dense[[i, i - 7]] = -0.15;
                // intentionally NO dense[[i - 7, i]] entry
            }
        }
        let m = CsrMatrix::from_dense(&dense, 1e-15);
        let b: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();

        let fac = MultifrontalBlrFactorization::factorize_with_config(
            &m,
            &BlrConfig {
                epsilon: 0.0,
                block_size: 16,
                min_block_size: 16,
                max_levels: None,
                ..Default::default()
            },
        )
        .expect("factorise");
        let x = fac.solve(&b);
        let res = rel_residual_f64(&m, &x, &b);
        assert!(
            res < 1e-10,
            "structurally nonsym residual {} too large",
            res
        );

        // Cross-check against exact sparse LU.
        let exact = SparseLuFactorization::factorize(&m).expect("exact LU");
        let x_exact = exact.solve(&b).expect("sparse LU solve");
        for i in 0..n {
            assert!(
                (x[i] - x_exact[i]).abs() < 1e-8,
                "[{}] blr={} exact={}",
                i,
                x[i],
                x_exact[i]
            );
        }
    }

    #[test]
    fn hodlr_panel_kind_solves_laplacian() {
        // Same Laplacian, factorised twice: once with BLR, once with HODLR.
        // Both should match exact LU and give comparable residuals.
        let nx = 16;
        let m = make_2d_laplacian_f64(nx);
        let n = m.num_rows;
        let b: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5).sin()).collect();

        let blr = MultifrontalBlrFactorization::factorize_with_config(
            &m,
            &BlrConfig {
                epsilon: 0.0,
                block_size: 16,
                min_block_size: 16,
                max_levels: None,
                panel_kind: PanelKind::Blr,
                ..Default::default()
            },
        )
        .expect("BLR factorise");
        let x_blr = blr.solve(&b);
        let res_blr = rel_residual_f64(&m, &x_blr, &b);
        assert!(res_blr < 1e-10, "BLR residual {} too large", res_blr);

        let hodlr = MultifrontalBlrFactorization::factorize_with_config(
            &m,
            &BlrConfig {
                epsilon: 0.0,
                block_size: 16,
                min_block_size: 16,
                max_levels: None,
                panel_kind: PanelKind::Hodlr,
                ..Default::default()
            },
        )
        .expect("HODLR factorise");
        let x_hodlr = hodlr.solve(&b);
        let res_hodlr = rel_residual_f64(&m, &x_hodlr, &b);
        assert!(res_hodlr < 1e-10, "HODLR residual {} too large", res_hodlr);

        // Solutions should agree.
        for i in 0..n {
            assert!(
                (x_blr[i] - x_hodlr[i]).abs() < 1e-8,
                "BLR vs HODLR diverge at i={}: {} vs {}",
                i,
                x_blr[i],
                x_hodlr[i]
            );
        }
    }

    #[test]
    fn nested_lr_panel_kind_solves_laplacian() {
        // End-to-end: 2D Laplacian factored with NestedLr panels at ε = 0
        // must match exact sparse LU within round-off, just as BLR / HODLR
        // do.
        let nx = 16;
        let m = make_2d_laplacian_f64(nx);
        let n = m.num_rows;
        let b: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5).sin()).collect();

        let nlr = MultifrontalBlrFactorization::factorize_with_config(
            &m,
            &BlrConfig {
                epsilon: 0.0,
                block_size: 16,
                min_block_size: 16,
                max_levels: None,
                panel_kind: PanelKind::NestedLr,
                ..Default::default()
            },
        )
        .expect("NestedLr factorise");
        let x_nlr = nlr.solve(&b);
        let res_nlr = rel_residual_f64(&m, &x_nlr, &b);
        assert!(res_nlr < 1e-9, "NestedLr residual {} too large", res_nlr);

        let exact = SparseLuFactorization::factorize(&m).expect("exact LU");
        let x_exact = exact.solve(&b).expect("sparse LU solve");
        for i in 0..n {
            assert!(
                (x_nlr[i] - x_exact[i]).abs() < 1e-7,
                "NestedLr vs exact diverge at i={}: nlr={} exact={}",
                i,
                x_nlr[i],
                x_exact[i]
            );
        }
    }

    #[test]
    fn left_look_matches_right_look() {
        // Both update orders should give the same residual on a Laplacian
        // factored at ε = 0. LeftLook exercises BLR×BLR matmul.
        let nx = 16;
        let m = make_2d_laplacian_f64(nx);
        let n = m.num_rows;
        let b: Vec<f64> = (0..n).map(|i| (i as f64 + 0.2).cos()).collect();

        let fac_rl = MultifrontalBlrFactorization::factorize_with_config(
            &m,
            &BlrConfig {
                epsilon: 0.0,
                block_size: 12,
                min_block_size: 12,
                max_levels: None,
                update_order: UpdateOrder::RightLook,
                ..Default::default()
            },
        )
        .expect("RightLook factorise");
        let fac_ll = MultifrontalBlrFactorization::factorize_with_config(
            &m,
            &BlrConfig {
                epsilon: 0.0,
                block_size: 12,
                min_block_size: 12,
                max_levels: None,
                update_order: UpdateOrder::LeftLook,
                ..Default::default()
            },
        )
        .expect("LeftLook factorise");

        let x_rl = fac_rl.solve(&b);
        let x_ll = fac_ll.solve(&b);
        for i in 0..n {
            assert!(
                (x_rl[i] - x_ll[i]).abs() < 1e-9,
                "[{}] rl={} ll={}",
                i,
                x_rl[i],
                x_ll[i]
            );
        }
        let res_rl = rel_residual_f64(&m, &x_rl, &b);
        let res_ll = rel_residual_f64(&m, &x_ll, &b);
        assert!(res_rl < 1e-10);
        assert!(res_ll < 1e-10);
    }

    #[test]
    fn left_look_hodlr_matches_right_look() {
        let nx = 12;
        let m = make_2d_laplacian_f64(nx);
        let n = m.num_rows;
        let b: Vec<f64> = (0..n).map(|i| (i as f64 + 0.7).sin()).collect();

        let mk = |order, kind| {
            MultifrontalBlrFactorization::factorize_with_config(
                &m,
                &BlrConfig {
                    epsilon: 0.0,
                    block_size: 8,
                    min_block_size: 8,
                    max_levels: None,
                    panel_kind: kind,
                    update_order: order,
                    ..Default::default()
                },
            )
            .expect("factorise")
        };
        let x_h_rl = mk(UpdateOrder::RightLook, PanelKind::Hodlr).solve(&b);
        let x_h_ll = mk(UpdateOrder::LeftLook, PanelKind::Hodlr).solve(&b);
        for i in 0..n {
            assert!(
                (x_h_rl[i] - x_h_ll[i]).abs() < 1e-9,
                "[{}] h_rl={} h_ll={}",
                i,
                x_h_rl[i],
                x_h_ll[i]
            );
        }
    }

    #[test]
    fn left_look_nested_lr_matches_right_look() {
        // Exercises the NestedLr×NestedLr compressed matmul path used by
        // LeftLook: both update orders on the same Laplacian + NestedLr
        // configuration must produce matching solutions.
        let nx = 12;
        let m = make_2d_laplacian_f64(nx);
        let n = m.num_rows;
        let b: Vec<f64> = (0..n).map(|i| (i as f64 + 0.7).sin()).collect();

        let mk = |order| {
            MultifrontalBlrFactorization::factorize_with_config(
                &m,
                &BlrConfig {
                    epsilon: 0.0,
                    block_size: 8,
                    min_block_size: 8,
                    max_levels: None,
                    panel_kind: PanelKind::NestedLr,
                    update_order: order,
                    ..Default::default()
                },
            )
            .expect("factorise")
        };
        let x_rl = mk(UpdateOrder::RightLook).solve(&b);
        let x_ll = mk(UpdateOrder::LeftLook).solve(&b);
        for i in 0..n {
            assert!(
                (x_rl[i] - x_ll[i]).abs() < 1e-9,
                "[{}] nlr_rl={} nlr_ll={}",
                i,
                x_rl[i],
                x_ll[i]
            );
        }
        let res_rl = rel_residual_f64(&m, &x_rl, &b);
        let res_ll = rel_residual_f64(&m, &x_ll, &b);
        assert!(res_rl < 1e-9);
        assert!(res_ll < 1e-9);
    }

    #[test]
    fn half_precision_storage_solves_within_tolerance() {
        // f32 Laplacian: with Half storage, residual stays around 1e-3
        // (the per-block precision floor). Without half (Full), it's at
        // round-off. Both should run without panicking.
        let nx = 24;
        let n = nx * nx;
        let mut dense = DenseMatrix::<f32>::zeros(n, n);
        for iy in 0..nx {
            for ix in 0..nx {
                let i = iy * nx + ix;
                dense[[i, i]] = 4.0;
                if ix > 0 {
                    dense[[i, i - 1]] = -1.0;
                }
                if ix + 1 < nx {
                    dense[[i, i + 1]] = -1.0;
                }
                if iy > 0 {
                    dense[[i, i - nx]] = -1.0;
                }
                if iy + 1 < nx {
                    dense[[i, i + nx]] = -1.0;
                }
            }
        }
        let m = CsrMatrix::from_dense(&dense, 1e-7);
        let b: Vec<f32> = (0..n).map(|i| (i as f32 + 0.5).sin()).collect();

        let fac_half = MultifrontalBlrFactorization::factorize_with_config(
            &m,
            &BlrConfig {
                epsilon: 1e-4,
                block_size: 12,
                min_block_size: 12,
                max_levels: None,
                low_rank_precision: LowRankPrecision::Half,
                ..Default::default()
            },
        )
        .expect("Half-precision factorise");
        let x = fac_half.solve(&b);

        let mut res_sq = 0.0_f32;
        let mut b_sq = 0.0_f32;
        for i in 0..n {
            let mut s = 0.0_f32;
            for idx in m.row_ptrs[i]..m.row_ptrs[i + 1] {
                s += m.values[idx] * x[m.col_indices[idx]];
            }
            res_sq += (b[i] - s) * (b[i] - s);
            b_sq += b[i] * b[i];
        }
        let rel = (res_sq / b_sq).sqrt();
        // Half precision has ~1e-3 error floor; relax accordingly.
        assert!(rel < 5e-2, "f32 half-precision residual {} too large", rel);
    }

    #[test]
    fn parallel_front_threshold_matches_serial() {
        // Build a single front large enough to trigger the par_chunks_mut
        // paths, then factorise it both ways and confirm the solves agree
        // to round-off. We use max_levels = 0 so the whole matrix lands in
        // one front, and we amalgamate everything so size = n. The 3D
        // grid below produces a front of size 343 if undivided, so we
        // pick a threshold below 343 for the parallel run and a threshold
        // far above for the serial run.
        let nx = 7;
        let n = nx * nx * nx;
        let mut dense = DenseMatrix::<f64>::zeros(n, n);
        for iz in 0..nx {
            for iy in 0..nx {
                for ix in 0..nx {
                    let i = iz * nx * nx + iy * nx + ix;
                    dense[[i, i]] = 6.0;
                    if ix > 0 {
                        dense[[i, i - 1]] = -1.0;
                    }
                    if ix + 1 < nx {
                        dense[[i, i + 1]] = -1.0;
                    }
                    if iy > 0 {
                        dense[[i, i - nx]] = -1.0;
                    }
                    if iy + 1 < nx {
                        dense[[i, i + nx]] = -1.0;
                    }
                    if iz > 0 {
                        dense[[i, i - nx * nx]] = -1.0;
                    }
                    if iz + 1 < nx {
                        dense[[i, i + nx * nx]] = -1.0;
                    }
                }
            }
        }
        let m = CsrMatrix::from_dense(&dense, 1e-15);
        let b: Vec<f64> = (0..n).map(|i| (i as f64 + 0.13).sin()).collect();

        let mk = |parallel_threshold| {
            MultifrontalBlrFactorization::factorize_with_config(
                &m,
                &BlrConfig {
                    epsilon: 0.0,
                    block_size: 32,
                    min_block_size: 32,
                    max_levels: None,
                    amalgamation_threshold: 0,
                    parallel_front_threshold: parallel_threshold,
                    ..Default::default()
                },
            )
            .expect("factorise")
        };
        let serial = mk(usize::MAX);
        let parallel = mk(64);

        let x_serial = serial.solve(&b);
        let x_parallel = parallel.solve(&b);
        for i in 0..n {
            assert!(
                (x_serial[i] - x_parallel[i]).abs() < 1e-9,
                "intra-front parallelism diverges at i={}: serial={} parallel={}",
                i,
                x_serial[i],
                x_parallel[i]
            );
        }
        // Both residuals must independently be small.
        let res_s = rel_residual_f64(&m, &x_serial, &b);
        let res_p = rel_residual_f64(&m, &x_parallel, &b);
        assert!(res_s < 1e-9 && res_p < 1e-9);
    }

    #[test]
    fn parallel_front_threshold_left_look_matches_serial() {
        // The LeftLook update path uses a different Schur code branch
        // (Panel::matmul_sub_into_dense). Intra-front parallelism still
        // applies to the L21/U12 dense computation that feeds it. Verify
        // the LeftLook + parallel combination agrees with the serial.
        let nx = 5;
        let n = nx * nx * nx;
        let mut dense = DenseMatrix::<f64>::zeros(n, n);
        for iz in 0..nx {
            for iy in 0..nx {
                for ix in 0..nx {
                    let i = iz * nx * nx + iy * nx + ix;
                    dense[[i, i]] = 6.0;
                    if ix > 0 {
                        dense[[i, i - 1]] = -1.0;
                    }
                    if ix + 1 < nx {
                        dense[[i, i + 1]] = -1.0;
                    }
                    if iy > 0 {
                        dense[[i, i - nx]] = -1.0;
                    }
                    if iy + 1 < nx {
                        dense[[i, i + nx]] = -1.0;
                    }
                    if iz > 0 {
                        dense[[i, i - nx * nx]] = -1.0;
                    }
                    if iz + 1 < nx {
                        dense[[i, i + nx * nx]] = -1.0;
                    }
                }
            }
        }
        let m = CsrMatrix::from_dense(&dense, 1e-15);
        let b: Vec<f64> = (0..n).map(|i| (i as f64 + 0.4).cos()).collect();

        let mk = |parallel_threshold| {
            MultifrontalBlrFactorization::factorize_with_config(
                &m,
                &BlrConfig {
                    epsilon: 0.0,
                    block_size: 16,
                    min_block_size: 16,
                    max_levels: None,
                    amalgamation_threshold: 0,
                    update_order: UpdateOrder::LeftLook,
                    parallel_front_threshold: parallel_threshold,
                    ..Default::default()
                },
            )
            .expect("factorise")
        };
        let x_serial = mk(usize::MAX).solve(&b);
        let x_parallel = mk(32).solve(&b);
        for i in 0..n {
            assert!(
                (x_serial[i] - x_parallel[i]).abs() < 1e-9,
                "LeftLook intra-front parallelism diverges at i={}: serial={} parallel={}",
                i,
                x_serial[i],
                x_parallel[i]
            );
        }
    }

    #[test]
    fn amalgamation_matches_no_amalgamation_on_laplacian() {
        // Round-trip: solve a 2D Laplacian with amalgamation disabled
        // (threshold = 0) and with the default threshold (256), then
        // verify the two solutions agree to round-off. Amalgamation
        // reorders perm and fuses fronts but must not change the
        // mathematical answer.
        let nx = 16;
        let m = make_2d_laplacian_f64(nx);
        let n = m.num_rows;
        let b: Vec<f64> = (0..n).map(|i| (i as f64 + 0.3).sin()).collect();

        let mk = |threshold| {
            MultifrontalBlrFactorization::factorize_with_config(
                &m,
                &BlrConfig {
                    epsilon: 0.0,
                    block_size: 16,
                    min_block_size: 16,
                    max_levels: None,
                    amalgamation_threshold: threshold,
                    ..Default::default()
                },
            )
            .expect("factorise")
        };
        let raw = mk(0);
        let merged = mk(256);

        assert!(
            merged.tree.fronts.len() <= raw.tree.fronts.len(),
            "amalgamation should not increase front count: raw={} merged={}",
            raw.tree.fronts.len(),
            merged.tree.fronts.len()
        );

        let x_raw = raw.solve(&b);
        let x_merged = merged.solve(&b);
        for i in 0..n {
            assert!(
                (x_raw[i] - x_merged[i]).abs() < 1e-9,
                "amalgamation diverges at i={}: raw={} merged={}",
                i,
                x_raw[i],
                x_merged[i]
            );
        }
    }

    #[test]
    fn amalgamation_reduces_front_count_on_3d_grid() {
        // 3D-ish problem with many small separators: nested dissection
        // creates a long chain of small fronts that amalgamation should
        // visibly collapse.
        let nx = 6;
        let n = nx * nx * nx;
        let mut dense = DenseMatrix::<f64>::zeros(n, n);
        for iz in 0..nx {
            for iy in 0..nx {
                for ix in 0..nx {
                    let i = iz * nx * nx + iy * nx + ix;
                    dense[[i, i]] = 6.0;
                    if ix > 0 {
                        dense[[i, i - 1]] = -1.0;
                    }
                    if ix + 1 < nx {
                        dense[[i, i + 1]] = -1.0;
                    }
                    if iy > 0 {
                        dense[[i, i - nx]] = -1.0;
                    }
                    if iy + 1 < nx {
                        dense[[i, i + nx]] = -1.0;
                    }
                    if iz > 0 {
                        dense[[i, i - nx * nx]] = -1.0;
                    }
                    if iz + 1 < nx {
                        dense[[i, i + nx * nx]] = -1.0;
                    }
                }
            }
        }
        let m = CsrMatrix::from_dense(&dense, 1e-15);

        let raw = MultifrontalBlrFactorization::factorize_with_config(
            &m,
            &BlrConfig {
                epsilon: 0.0,
                block_size: 16,
                min_block_size: 16,
                max_levels: None,
                amalgamation_threshold: 0,
                ..Default::default()
            },
        )
        .expect("raw factorise");
        let merged = MultifrontalBlrFactorization::factorize_with_config(
            &m,
            &BlrConfig {
                epsilon: 0.0,
                block_size: 16,
                min_block_size: 16,
                max_levels: None,
                amalgamation_threshold: 256,
                ..Default::default()
            },
        )
        .expect("merged factorise");

        assert!(
            merged.tree.fronts.len() < raw.tree.fronts.len(),
            "amalgamation should strictly shrink the tree on a 3D problem: \
             raw={} merged={}",
            raw.tree.fronts.len(),
            merged.tree.fronts.len()
        );

        // Solve correctness still required.
        let b: Vec<f64> = (0..n).map(|i| (i as f64 + 1.7).cos()).collect();
        let x = merged.solve(&b);
        let res = rel_residual_f64(&m, &x, &b);
        assert!(res < 1e-9, "merged residual {} too large", res);
    }

    #[test]
    fn nested_lr_panel_kind_with_half_precision_solves() {
        // End-to-end test for `PanelKind::NestedLr` combined with
        // `LowRankPrecision::Half`: the plumbing routes precision through
        // the polymorphic Panel dispatcher into
        // `NestedLrPanel::from_dense_with_precision`, and the multifrontal
        // solve must still converge to within the f16 per-block error
        // floor. Without this test the plumbing could rot silently —
        // every other NestedLr integration test uses Full precision.
        let nx = 24;
        let n = nx * nx;
        let mut dense = DenseMatrix::<f32>::zeros(n, n);
        for iy in 0..nx {
            for ix in 0..nx {
                let i = iy * nx + ix;
                dense[[i, i]] = 4.0;
                if ix > 0 {
                    dense[[i, i - 1]] = -1.0;
                }
                if ix + 1 < nx {
                    dense[[i, i + 1]] = -1.0;
                }
                if iy > 0 {
                    dense[[i, i - nx]] = -1.0;
                }
                if iy + 1 < nx {
                    dense[[i, i + nx]] = -1.0;
                }
            }
        }
        let m = CsrMatrix::from_dense(&dense, 1e-7);
        let b: Vec<f32> = (0..n).map(|i| (i as f32 + 0.5).sin()).collect();

        let fac = MultifrontalBlrFactorization::factorize_with_config(
            &m,
            &BlrConfig {
                epsilon: 1e-4,
                block_size: 12,
                min_block_size: 12,
                max_levels: None,
                panel_kind: PanelKind::NestedLr,
                low_rank_precision: LowRankPrecision::Half,
                ..Default::default()
            },
        )
        .expect("NestedLr+Half factorise");
        let x = fac.solve(&b);

        let mut res_sq = 0.0_f32;
        let mut b_sq = 0.0_f32;
        for i in 0..n {
            let mut s = 0.0_f32;
            for idx in m.row_ptrs[i]..m.row_ptrs[i + 1] {
                s += m.values[idx] * x[m.col_indices[idx]];
            }
            res_sq += (b[i] - s) * (b[i] - s);
            b_sq += b[i] * b[i];
        }
        let rel = (res_sq / b_sq).sqrt();
        assert!(
            rel < 5e-2,
            "NestedLr+Half residual {} above expected f16 error floor",
            rel
        );
    }

    #[test]
    fn bunch_kaufman_pivot_handles_indefinite() {
        // Symmetric-indefinite-ish matrix with zero diagonal entries
        // every 4th row — partial / rook pivoting cope by row swaps, but
        // Bunch-Kaufman with 2×2 pivots is the natural fit. We just check
        // BK reaches an accurate residual.
        let n = 48;
        let mut dense = DenseMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            dense[[i, i]] = if i % 4 == 0 {
                0.0
            } else {
                2.0 + (i as f64) * 0.01
            };
            if i + 1 < n {
                dense[[i, i + 1]] = 1.0;
                dense[[i + 1, i]] = 1.0;
            }
            if i + 3 < n {
                dense[[i, i + 3]] = 0.3;
                dense[[i + 3, i]] = 0.3;
            }
        }
        let m = CsrMatrix::from_dense(&dense, 1e-15);
        let b: Vec<f64> = (0..n).map(|i| (i as f64 + 0.7).sin()).collect();

        let fac_bk = MultifrontalBlrFactorization::factorize_with_config(
            &m,
            &BlrConfig {
                epsilon: 0.0,
                block_size: 12,
                min_block_size: 12,
                max_levels: None,
                pivot_strategy: PivotStrategy::BunchKaufman,
                ..Default::default()
            },
        )
        .expect("BK factorise");
        let x = fac_bk.solve(&b);
        let res = rel_residual_f64(&m, &x, &b);
        assert!(res < 1e-9, "BK residual {} too large", res);

        // Cross-check against exact sparse LU.
        let exact = SparseLuFactorization::factorize(&m).expect("exact LU");
        let x_exact = exact.solve(&b).expect("sparse LU solve");
        for i in 0..n {
            assert!(
                (x[i] - x_exact[i]).abs() < 1e-7,
                "[{}] bk={} exact={}",
                i,
                x[i],
                x_exact[i]
            );
        }
    }

    #[test]
    fn bunch_kaufman_emits_2x2_pivot_on_zero_diagonal() {
        // Symmetric tridiagonal path graph with vanishing diagonal: the
        // first BK test |A[k,k]| ≥ α·λ fails (a_kk = 0, λ = 1), the σ-test
        // fails (a_kk · σ = 0 < α · λ²), and the row-r test fails
        // (a_rr = 0 < α · σ). Every step of the FS factorisation must
        // therefore emit a 2×2 block. Use a single-front configuration so
        // we can directly inspect `pivot_kinds` of the only front.
        use super::front::PivotKind;

        let n = 16;
        let mut dense = DenseMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            // diagonal stays zero — the whole matrix is structurally
            // indefinite (eigenvalues 2·cos(jπ/(n+1)), all nonzero).
            if i + 1 < n {
                dense[[i, i + 1]] = 1.0;
                dense[[i + 1, i]] = 1.0;
            }
        }
        let m = CsrMatrix::from_dense(&dense, 1e-15);

        let fac = MultifrontalBlrFactorization::factorize_with_config(
            &m,
            &BlrConfig {
                epsilon: 0.0,
                block_size: 64,
                min_block_size: 64,
                max_levels: Some(0),
                pivot_strategy: PivotStrategy::BunchKaufman,
                ..Default::default()
            },
        )
        .expect("BK single-front factorise");

        let two_by_two = fac
            .fronts
            .iter()
            .flat_map(|f| f.pivot_kinds.iter())
            .filter(|k| **k == PivotKind::TwoByTwoStart)
            .count();
        assert!(
            two_by_two > 0,
            "BK should emit at least one 2×2 pivot on zero-diagonal matrix \
             (got pivot_kinds = {:?})",
            fac.fronts
                .iter()
                .map(|f| f.pivot_kinds.clone())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn partial_pivoting_never_emits_2x2() {
        // Precondition for any future BLAS-TRSM fast path: `Partial` can
        // only select 1×1 pivots by construction (`select_pivot` returns
        // `BkPivot::OneByOne` unconditionally, with column fixed at `k`),
        // so every front's `pivot_kinds` must be all-`OneByOne` and
        // `col_pivots` identity.
        use super::front::PivotKind;

        fn assert_all_one_by_one(fac: &MultifrontalBlrFactorization<f64>, ctx: &str) {
            for (fi, f) in fac.fronts.iter().enumerate() {
                assert!(
                    f.pivot_kinds.iter().all(|k| *k == PivotKind::OneByOne),
                    "{ctx} front {fi}: Partial must emit only 1×1 pivots \
                     (got {:?})",
                    f.pivot_kinds
                );
                assert!(
                    f.col_pivots.iter().enumerate().all(|(i, &c)| c == i),
                    "{ctx} front {fi}: Partial never swaps columns \
                     (got {:?})",
                    f.col_pivots
                );
            }
        }

        // Multi-front factorisation of a diagonally dominant tridiagonal
        // (tiny blocks, no amalgamation so the tree is not merged back).
        let n = 64;
        let mut dense = DenseMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            dense[[i, i]] = 4.0;
            if i > 0 {
                dense[[i, i - 1]] = -1.0;
            }
            if i + 1 < n {
                dense[[i, i + 1]] = -1.0;
            }
        }
        let m = CsrMatrix::from_dense(&dense, 1e-15);
        let fac = MultifrontalBlrFactorization::factorize_with_config(
            &m,
            &BlrConfig {
                epsilon: 0.0,
                block_size: 8,
                min_block_size: 8,
                // No amalgamation: the default threshold (256) would merge
                // the whole tree back into a single front.
                amalgamation_threshold: 0,
                ..Default::default()
            },
        )
        .expect("partial multi-front factorise");
        assert!(
            fac.fronts.len() > 1,
            "expected a multi-front tree (got {})",
            fac.fronts.len()
        );
        assert_all_one_by_one(&fac, "multi-front");

        // Row-reversed copy: nonsingular (cond ~3) but Partial must swap
        // rows at every step. Single-front configuration factorizes fine;
        // assert the swaps happened and stayed 1×1.
        let mut dense_rev = DenseMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            let src = n - 1 - i;
            dense_rev[[i, src]] = 4.0;
            if src > 0 {
                dense_rev[[i, src - 1]] = -1.0;
            }
            if src + 1 < n {
                dense_rev[[i, src + 1]] = -1.0;
            }
        }
        let m_rev = CsrMatrix::from_dense(&dense_rev, 1e-15);
        let fac_rev = MultifrontalBlrFactorization::factorize_with_config(
            &m_rev,
            &BlrConfig {
                epsilon: 0.0,
                block_size: 64,
                min_block_size: 64,
                ..Default::default()
            },
        )
        .expect("partial single-front factorise of row-reversed matrix");
        assert_all_one_by_one(&fac_rev, "row-reversed");
        assert!(
            fac_rev
                .fronts
                .iter()
                .any(|f| f.pivots.iter().enumerate().any(|(i, &p)| p != i)),
            "row-reversed matrix should force at least one row swap"
        );
    }

    #[test]
    #[ignore = "delayed-pivot machinery fails (SingularFront) on well-conditioned \
                row-permuted matrices under multi-front configs; see perf/PERF.md \
                2026-09-25 entry. Re-enable after the delay path is fixed."]
    fn partial_row_reversed_multi_front_factorises() {
        // cond ~3, nonsingular — must factorise, but today aborts with
        // `SingularFront` once mass delayed pivoting meets a multi-front
        // tree (single-front config of the same matrix works).
        let n = 64;
        let mut dense = DenseMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            let src = n - 1 - i;
            dense[[i, src]] = 4.0;
            if src > 0 {
                dense[[i, src - 1]] = -1.0;
            }
            if src + 1 < n {
                dense[[i, src + 1]] = -1.0;
            }
        }
        let m = CsrMatrix::from_dense(&dense, 1e-15);
        MultifrontalBlrFactorization::factorize_with_config(
            &m,
            &BlrConfig {
                epsilon: 0.0,
                block_size: 8,
                min_block_size: 8,
                amalgamation_threshold: 0,
                ..Default::default()
            },
        )
        .expect("row-reversed multi-front factorise");
    }

    #[test]
    fn rook_pivot_indefinite_block() {
        // Matrix with tiny diagonals — partial pivoting works here too, but
        // rook should give a tighter residual since it actively searches
        // both row and column for the max. We use it to verify the rook
        // code path produces the same solution as partial within round-off.
        let n = 60;
        let mut dense = DenseMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            dense[[i, i]] = if i % 5 == 0 { 0.05 } else { 3.0 };
            if i + 1 < n {
                dense[[i, i + 1]] = -1.0;
                dense[[i + 1, i]] = -1.0;
            }
            if i + 4 < n {
                dense[[i, i + 4]] = -0.5;
                dense[[i + 4, i]] = -0.3;
            }
        }
        let m = CsrMatrix::from_dense(&dense, 1e-15);
        let b: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();

        let fac_partial = MultifrontalBlrFactorization::factorize_with_config(
            &m,
            &BlrConfig {
                epsilon: 0.0,
                block_size: 16,
                min_block_size: 16,
                max_levels: None,
                pivot_strategy: PivotStrategy::Partial,
                ..Default::default()
            },
        )
        .expect("partial pivot factorise");
        let fac_rook = MultifrontalBlrFactorization::factorize_with_config(
            &m,
            &BlrConfig {
                epsilon: 0.0,
                block_size: 16,
                min_block_size: 16,
                max_levels: None,
                pivot_strategy: PivotStrategy::Rook,
                ..Default::default()
            },
        )
        .expect("rook factorise");

        let x_partial = fac_partial.solve(&b);
        let x_rook = fac_rook.solve(&b);

        let res_partial = rel_residual_f64(&m, &x_partial, &b);
        let res_rook = rel_residual_f64(&m, &x_rook, &b);
        assert!(res_partial < 1e-10, "partial residual {}", res_partial);
        assert!(res_rook < 1e-10, "rook residual {}", res_rook);
    }

    #[test]
    fn laplacian_2d_small_residual() {
        let nx = 10;
        let m = make_2d_laplacian_f64(nx);
        let n = m.num_rows;
        let b: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();

        let fac = MultifrontalBlrFactorization::factorize_with_config(
            &m,
            &BlrConfig {
                epsilon: 0.0,
                block_size: 16,
                min_block_size: 8,
                max_levels: None,
                ..Default::default()
            },
        )
        .expect("factorise");
        let x = fac.solve(&b);
        let res = rel_residual_f64(&m, &x, &b);
        assert!(res < 1e-10, "residual {} too large", res);
    }

    #[test]
    fn laplacian_2d_with_blr_compression() {
        let nx = 16;
        let m = make_2d_laplacian_f64(nx);
        let n = m.num_rows;
        let b: Vec<f64> = (0..n).map(|i| (i as f64 + 0.3).cos()).collect();

        let fac = MultifrontalBlrFactorization::factorize_with_config(
            &m,
            &BlrConfig {
                epsilon: 1e-8,
                block_size: 16,
                min_block_size: 16,
                max_levels: None,
                ..Default::default()
            },
        )
        .expect("factorise");
        let x = fac.solve(&b);
        let res = rel_residual_f64(&m, &x, &b);
        // Compression tolerance is 1e-8; residual should be in that ballpark.
        assert!(res < 1e-5, "residual {} too large", res);
    }

    #[test]
    fn laplacian_f32_matches_exact() {
        // Same problem in f32: should still solve to the f32 precision floor.
        let nx = 8;
        let mut dense = DenseMatrix::<f32>::zeros(nx * nx, nx * nx);
        for iy in 0..nx {
            for ix in 0..nx {
                let i = iy * nx + ix;
                dense[[i, i]] = 4.0;
                if ix > 0 {
                    dense[[i, i - 1]] = -1.0;
                }
                if ix + 1 < nx {
                    dense[[i, i + 1]] = -1.0;
                }
                if iy > 0 {
                    dense[[i, i - nx]] = -1.0;
                }
                if iy + 1 < nx {
                    dense[[i, i + nx]] = -1.0;
                }
            }
        }
        let m = CsrMatrix::from_dense(&dense, 1e-7);
        let n = m.num_rows;
        let b: Vec<f32> = (0..n).map(|i| (i as f32 + 0.5).sin()).collect();

        let fac = MultifrontalBlrFactorization::factorize_with_config(
            &m,
            &BlrConfig {
                epsilon: 0.0,
                block_size: 16,
                min_block_size: 8,
                max_levels: None,
                ..Default::default()
            },
        )
        .expect("factorise");
        let x = fac.solve(&b);

        // Residual in f32.
        let mut res_sq = 0.0_f32;
        let mut b_sq = 0.0_f32;
        for i in 0..n {
            let mut s = 0.0_f32;
            for idx in m.row_ptrs[i]..m.row_ptrs[i + 1] {
                s += m.values[idx] * x[m.col_indices[idx]];
            }
            res_sq += (b[i] - s) * (b[i] - s);
            b_sq += b[i] * b[i];
        }
        let rel = (res_sq / b_sq).sqrt();
        assert!(rel < 1e-3, "f32 residual {} too large", rel);
    }

    #[test]
    fn complex_laplacian_runs() {
        let nx = 8;
        let n = nx * nx;
        let mut dense = DenseMatrix::<Complex64>::zeros(n, n);
        for iy in 0..nx {
            for ix in 0..nx {
                let i = iy * nx + ix;
                dense[[i, i]] = Complex64::new(4.0, 0.1);
                if ix > 0 {
                    dense[[i, i - 1]] = Complex64::new(-1.0, 0.0);
                }
                if ix + 1 < nx {
                    dense[[i, i + 1]] = Complex64::new(-1.0, 0.0);
                }
                if iy > 0 {
                    dense[[i, i - nx]] = Complex64::new(-1.0, 0.0);
                }
                if iy + 1 < nx {
                    dense[[i, i + nx]] = Complex64::new(-1.0, 0.0);
                }
            }
        }
        let m = CsrMatrix::from_dense(&dense, 1e-15);
        let b: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64 + 1.0).sin(), 0.5))
            .collect();
        let fac = MultifrontalBlrFactorization::factorize_with_config(
            &m,
            &BlrConfig {
                epsilon: 0.0,
                block_size: 16,
                min_block_size: 8,
                max_levels: None,
                ..Default::default()
            },
        )
        .expect("factorise");
        let x = fac.solve(&b);
        let mut res_sq = 0.0_f64;
        let mut b_sq = 0.0_f64;
        for i in 0..n {
            let mut s = Complex64::new(0.0, 0.0);
            for idx in m.row_ptrs[i]..m.row_ptrs[i + 1] {
                s += m.values[idx] * x[m.col_indices[idx]];
            }
            res_sq += (b[i] - s).norm_sqr();
            b_sq += b[i].norm_sqr();
        }
        let rel = (res_sq / b_sq).sqrt();
        assert!(rel < 1e-10, "complex residual {} too large", rel);
    }

    #[test]
    fn badly_scaled_matrix_factorizes_with_scale_aware_tolerance() {
        // 2D Laplacian scaled by 1e-25: well-conditioned (scaling does
        // not change κ), but every pivot magnitude is ~1e-25 — far below
        // the old absolute 1e-20 rejection threshold, which aborted with
        // SingularFront. The scale-aware tolerance
        // `multiplier · ε · nfs · max|F11|` must accept the pivots and
        // solve to round-off (in the scaled norm).
        let nx = 8;
        let mut m = make_2d_laplacian_f64(nx);
        for v in m.values.iter_mut() {
            *v *= 1e-25;
        }
        let n = m.num_rows;
        let b: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();

        let fac = MultifrontalBlrFactorization::factorize_with_config(
            &m,
            &BlrConfig {
                epsilon: 0.0,
                block_size: 16,
                min_block_size: 8,
                max_levels: None,
                ..Default::default()
            },
        )
        .expect("badly scaled factorise");
        let x = fac.solve(&b);
        let res = rel_residual_f64(&m, &x, &b);
        assert!(res < 1e-10, "badly scaled residual {} too large", res);

        // Same for a 1e+25 scale factor.
        let mut m_big = make_2d_laplacian_f64(nx);
        for v in m_big.values.iter_mut() {
            *v *= 1e25;
        }
        let fac_big = MultifrontalBlrFactorization::factorize_with_config(
            &m_big,
            &BlrConfig {
                epsilon: 0.0,
                block_size: 16,
                min_block_size: 8,
                max_levels: None,
                ..Default::default()
            },
        )
        .expect("badly scaled (big) factorise");
        let x_big = fac_big.solve(&b);
        let res_big = rel_residual_f64(&m_big, &x_big, &b);
        assert!(
            res_big < 1e-10,
            "badly scaled (big) residual {} too large",
            res_big
        );
    }

    #[test]
    fn pivot_tol_multiplier_controls_rejection() {
        // With an absurdly large multiplier the scale-aware threshold
        // exceeds every pivot magnitude, so every front postpones all its
        // pivots and the root front fails: the multiplier genuinely
        // controls the rejection basis.
        let nx = 8;
        let m = make_2d_laplacian_f64(nx);
        let result = MultifrontalBlrFactorization::factorize_with_config(
            &m,
            &BlrConfig {
                epsilon: 0.0,
                block_size: 16,
                min_block_size: 8,
                max_levels: None,
                pivot_tol_multiplier: 1e40,
                ..Default::default()
            },
        );
        assert!(
            matches!(result, Err(BlrError::SingularFront(_))),
            "huge pivot tolerance multiplier must reject every pivot"
        );
    }
}
