//! Deterministic row partitioning for the Phase-2 distributed BEM shell.
//!
//! This module deliberately stops at the MPI boundary: it creates the stable
//! rank ownership plan used by a distributed matvec/GMRES backend, but it does
//! not perform network communication. Keeping the partitioner in the core crate
//! lets single-node tests validate ownership, ordering, and DOF coverage before
//! an `rsmpi` backend is wired in.

use crate::core::bem_solver::BemError;
use crate::core::types::Element;
use num_traits::{Float, Zero};
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::iterative::{BlockGmresConfig, BlockGmresSolution, GmresConfig, GmresSolution};
use sonium_solvers::traits::{AdjointCapability, ComplexField, LinearOperator, Preconditioner};
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

/// Space-partitioning strategy for distributed row ownership.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionStrategy {
    /// Sort active elements by a 3D Morton key and split the ordered rows into
    /// contiguous rank slabs. This mirrors the H-matrix row-cluster partition
    /// and keeps neighboring panels mostly rank-local.
    Morton,
}

/// Configuration for a distributed BEM run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DistributedBemConfig {
    /// Number of MPI ranks participating in the solve.
    pub num_ranks: usize,
    /// Current rank. Must be in `0..num_ranks`.
    pub local_rank: usize,
    /// Row ownership strategy.
    pub strategy: PartitionStrategy,
}

impl DistributedBemConfig {
    /// Return a valid one-rank configuration for local parity runs.
    pub fn single_rank() -> Self {
        Self {
            num_ranks: 1,
            local_rank: 0,
            strategy: PartitionStrategy::Morton,
        }
    }

    /// Validate rank counts and local-rank bounds.
    pub fn validate(self) -> Result<(), BemError> {
        if self.num_ranks == 0 {
            return Err(BemError::InvalidParameters(
                "distributed BEM requires at least one rank".to_string(),
            ));
        }
        if self.local_rank >= self.num_ranks {
            return Err(BemError::InvalidParameters(format!(
                "local_rank {} is outside 0..{}",
                self.local_rank, self.num_ranks
            )));
        }
        Ok(())
    }
}

/// Rows and DOFs owned by one distributed rank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankPartition {
    /// MPI rank that owns this partition.
    pub rank: usize,
    /// Active BEM element rows assigned to `rank`, in Morton order.
    pub element_indices: Vec<usize>,
    /// Global BEM DOF rows owned by `rank`.
    pub dof_indices: Vec<usize>,
}

/// Complete deterministic ownership plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DistributedRowPlan {
    /// One row partition per rank.
    pub partitions: Vec<RankPartition>,
    /// Number of non-evaluation elements assigned across all ranks.
    pub num_active_elements: usize,
    /// Number of active global DOF rows assigned across all ranks.
    pub num_dofs: usize,
}

impl DistributedRowPlan {
    /// Validate the complete ownership layout before a transport or solver
    /// sees any caller-owned buffers.
    pub fn validate(&self) -> Result<(), BemError> {
        self.validate_vector_layout()
    }

    /// Return the partition owned by `rank`, if present.
    pub fn local(&self, rank: usize) -> Option<&RankPartition> {
        self.partitions.iter().find(|p| p.rank == rank)
    }

    /// Return the rank that owns one global DOF row.
    ///
    /// Ownership is deliberately kept separate from transport: this checked
    /// lookup is the deterministic metadata a future MPI/remote backend can
    /// use to route a gathered value. It never performs communication.
    pub fn owner_of_dof(&self, dof: usize) -> Result<usize, BemError> {
        self.validate_vector_layout()?;
        if dof >= self.num_dofs {
            return Err(BemError::InvalidParameters(format!(
                "distributed DOF {dof} outside global range 0..{}",
                self.num_dofs
            )));
        }
        self.partitions
            .iter()
            .find(|partition| partition.dof_indices.contains(&dof))
            .map(|partition| partition.rank)
            .ok_or_else(|| {
                BemError::InvalidParameters(format!("distributed DOF {dof} has no owning rank"))
            })
    }

    /// Return the rank that owns a pressure or normal-velocity transmission row.
    ///
    /// Rows `[0, interface_dofs)` are pressure rows and rows
    /// `[interface_dofs, 2 * interface_dofs)` are matching velocity rows.
    /// This is a local ownership query only; callers still exchange values
    /// through their chosen transport layer.
    pub fn owner_of_transmission_row(
        &self,
        row: usize,
        interface_dofs: usize,
    ) -> Result<usize, BemError> {
        if interface_dofs == 0 || self.num_dofs != interface_dofs {
            return Err(BemError::InvalidParameters(format!(
                "transmission row ownership expects {interface_dofs} planned DOFs, found {}",
                self.num_dofs
            )));
        }
        let total_rows = interface_dofs.checked_mul(2).ok_or_else(|| {
            BemError::InvalidParameters("transmission row count overflows usize".to_string())
        })?;
        if row >= total_rows {
            return Err(BemError::InvalidParameters(format!(
                "transmission row {row} outside global range 0..{total_rows}"
            )));
        }
        self.owner_of_dof(row % interface_dofs)
    }

    /// Scatter a global vector into deterministic per-rank owned slices.
    ///
    /// The returned outer vector follows `partitions` order and each inner
    /// vector follows that partition's `dof_indices` order. This prepares
    /// caller-owned buffers for communication, but performs no communication.
    pub fn scatter_dof_values<T: Clone>(
        &self,
        global_values: &[T],
    ) -> Result<Vec<Vec<T>>, BemError> {
        self.validate_vector_layout()?;
        if global_values.len() != self.num_dofs {
            return Err(BemError::InvalidParameters(format!(
                "distributed scatter expects {} values, found {}",
                self.num_dofs,
                global_values.len()
            )));
        }
        Ok(self
            .partitions
            .iter()
            .map(|partition| {
                partition
                    .dof_indices
                    .iter()
                    .map(|&dof| global_values[dof].clone())
                    .collect()
            })
            .collect())
    }

    /// Gather deterministic per-rank owned slices into one global vector.
    ///
    /// The input outer vector must follow `partitions` order. Shape, duplicate
    /// ownership, and missing rows are checked before returning a value so a
    /// transport backend cannot silently publish a partial Krylov vector.
    pub fn gather_dof_values<T: Clone>(&self, local_values: &[Vec<T>]) -> Result<Vec<T>, BemError> {
        self.validate_vector_layout()?;
        if local_values.len() != self.partitions.len() {
            return Err(BemError::InvalidParameters(format!(
                "distributed gather expects {} rank buffers, found {}",
                self.partitions.len(),
                local_values.len()
            )));
        }
        let mut gathered = vec![None; self.num_dofs];
        for (partition, values) in self.partitions.iter().zip(local_values) {
            if values.len() != partition.dof_indices.len() {
                return Err(BemError::InvalidParameters(format!(
                    "rank {} gather buffer expects {} values, found {}",
                    partition.rank,
                    partition.dof_indices.len(),
                    values.len()
                )));
            }
            for (&dof, value) in partition.dof_indices.iter().zip(values) {
                let slot = gathered.get_mut(dof).ok_or_else(|| {
                    BemError::InvalidParameters(format!(
                        "rank {} gather DOF {dof} outside global range 0..{}",
                        partition.rank, self.num_dofs
                    ))
                })?;
                if slot.is_some() {
                    return Err(BemError::InvalidParameters(format!(
                        "distributed gather received duplicate DOF {dof}"
                    )));
                }
                *slot = Some(value.clone());
            }
        }
        gathered
            .into_iter()
            .enumerate()
            .map(|(dof, value)| {
                value.ok_or_else(|| {
                    BemError::InvalidParameters(format!("distributed gather missing DOF {dof}"))
                })
            })
            .collect()
    }

    fn validate_vector_layout(&self) -> Result<(), BemError> {
        let mut ranks = HashSet::with_capacity(self.partitions.len());
        let mut dofs = HashSet::with_capacity(self.num_dofs);
        for partition in &self.partitions {
            if !ranks.insert(partition.rank) {
                return Err(BemError::InvalidParameters(format!(
                    "duplicate distributed row partition for rank {}",
                    partition.rank
                )));
            }
            for &dof in &partition.dof_indices {
                if dof >= self.num_dofs {
                    return Err(BemError::InvalidParameters(format!(
                        "rank {} DOF {dof} outside global range 0..{}",
                        partition.rank, self.num_dofs
                    )));
                }
                if !dofs.insert(dof) {
                    return Err(BemError::InvalidParameters(format!(
                        "distributed row plan repeats DOF {dof}"
                    )));
                }
            }
        }
        if dofs.len() != self.num_dofs {
            return Err(BemError::InvalidParameters(format!(
                "distributed row plan owns {} of {} global DOFs",
                dofs.len(),
                self.num_dofs
            )));
        }
        Ok(())
    }

    /// Expand owned interface DOFs into pressure/velocity transmission rows.
    ///
    /// The returned vectors preserve each partition's deterministic DOF order:
    /// pressure rows come first, followed by matching normal-velocity rows.
    /// This is an ownership/action contract only; it performs no MPI
    /// communication and does not solve a distributed system.
    pub fn transmission_trace_rows(
        &self,
        interface_dofs: usize,
    ) -> Result<Vec<Vec<usize>>, BemError> {
        if interface_dofs == 0 || self.num_dofs != interface_dofs {
            return Err(BemError::InvalidParameters(format!(
                "transmission row expansion expects {interface_dofs} planned DOFs, found {}",
                self.num_dofs
            )));
        }
        let mut rows_by_rank = Vec::with_capacity(self.partitions.len());
        let mut seen = HashSet::new();
        let mut seen_ranks = HashSet::new();
        for partition in &self.partitions {
            if !seen_ranks.insert(partition.rank) {
                return Err(BemError::InvalidParameters(format!(
                    "duplicate distributed row partition for rank {}",
                    partition.rank
                )));
            }
            let mut rows = Vec::with_capacity(partition.dof_indices.len() * 2);
            for &dof in &partition.dof_indices {
                if dof >= interface_dofs || !seen.insert(dof) {
                    return Err(BemError::InvalidParameters(format!(
                        "invalid or duplicate transmission DOF {dof} in rank {}",
                        partition.rank
                    )));
                }
                rows.push(dof);
            }
            rows.extend(
                partition
                    .dof_indices
                    .iter()
                    .map(|&dof| interface_dofs + dof),
            );
            rows_by_rank.push(rows);
        }
        if seen.len() != interface_dofs {
            return Err(BemError::InvalidParameters(format!(
                "transmission row expansion covers {} of {interface_dofs} DOFs",
                seen.len()
            )));
        }
        Ok(rows_by_rank)
    }
}

/// Collective exchange used by a distributed matrix-free BEM action.
///
/// A rank supplies its owned vector slice and receives the complete global
/// vector needed by a non-local BEM operator.  An MPI implementation can map
/// this directly to an `allgatherv`; the core crate deliberately keeps the
/// transport dependency-free so deterministic loopback and shared-memory
/// implementations remain available on every supported target.
pub trait DistributedVectorExchange<T: ComplexField>: Send + Sync {
    /// Exchange one rank's owned values and return the global vector.
    fn all_gather(
        &self,
        plan: &DistributedRowPlan,
        local_rank: usize,
        local_values: &[T],
    ) -> Result<Vec<T>, BemError>;

    /// Exchange several rank-local right-hand sides in one checked call.
    ///
    /// The default implementation preserves compatibility with scalar-only
    /// transports by gathering each column through [`Self::all_gather`].
    /// Network transports can override this with a native all-gatherv for
    /// better multi-RHS throughput without changing the operator contract.
    fn all_gather_block(
        &self,
        plan: &DistributedRowPlan,
        local_rank: usize,
        local_values: &DenseMatrix<T>,
    ) -> Result<DenseMatrix<T>, BemError> {
        plan.validate()?;
        let partition = plan.local(local_rank).ok_or_else(|| {
            BemError::InvalidParameters(format!(
                "distributed exchange rank {local_rank} has no partition"
            ))
        })?;
        if local_values.nrows() != partition.dof_indices.len() {
            return Err(BemError::InvalidParameters(format!(
                "distributed exchange rank {local_rank} expects {} block rows, found {}",
                partition.dof_indices.len(),
                local_values.nrows()
            )));
        }
        if local_values
            .as_slice()
            .iter()
            .any(|value| !value.norm().is_finite())
        {
            return Err(BemError::InvalidParameters(
                "distributed exchange received non-finite block values".to_string(),
            ));
        }

        let mut gathered = DenseMatrix::zeros(plan.num_dofs, local_values.ncols());
        for column in 0..local_values.ncols() {
            let global = self.all_gather(plan, local_rank, &local_values.column_to_vec(column))?;
            if global.len() != plan.num_dofs {
                return Err(BemError::InvalidParameters(format!(
                    "distributed exchange returned {} values for block column {}, expected {}",
                    global.len(),
                    column,
                    plan.num_dofs
                )));
            }
            gathered.set_column(column, &global);
        }
        Ok(gathered)
    }
}

/// Deterministic shared-memory exchange useful for serial execution and QA.
///
/// The backing vector represents the most recent global state.  Each call
/// updates the caller's owned entries before returning a snapshot, which
/// gives a faithful in-process model of rank-local all-gather semantics while
/// still allowing a real MPI transport to be supplied by an application.
#[derive(Debug, Clone)]
pub struct InProcessVectorExchange<T> {
    values: Arc<Mutex<Vec<T>>>,
}

impl<T> InProcessVectorExchange<T> {
    /// Create an exchange initialized with a complete global vector.
    pub fn new(global_values: Vec<T>) -> Self {
        Self {
            values: Arc::new(Mutex::new(global_values)),
        }
    }

    /// Return a consistent snapshot of the backing global vector.
    pub fn snapshot(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.values
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }
}

impl<T: ComplexField> DistributedVectorExchange<T> for InProcessVectorExchange<T> {
    fn all_gather(
        &self,
        plan: &DistributedRowPlan,
        local_rank: usize,
        local_values: &[T],
    ) -> Result<Vec<T>, BemError> {
        plan.validate()?;
        let partition = plan.local(local_rank).ok_or_else(|| {
            BemError::InvalidParameters(format!(
                "distributed exchange rank {local_rank} has no partition"
            ))
        })?;
        if local_values.len() != partition.dof_indices.len() {
            return Err(BemError::InvalidParameters(format!(
                "distributed exchange rank {local_rank} expects {} values, found {}",
                partition.dof_indices.len(),
                local_values.len()
            )));
        }
        if local_values.iter().any(|value| !value.norm().is_finite()) {
            return Err(BemError::InvalidParameters(
                "distributed exchange received non-finite values".to_string(),
            ));
        }

        let mut values = self
            .values
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if values.len() != plan.num_dofs {
            return Err(BemError::InvalidParameters(format!(
                "distributed exchange backing vector expects {} values, found {}",
                plan.num_dofs,
                values.len()
            )));
        }
        for (&dof, value) in partition.dof_indices.iter().zip(local_values) {
            values[dof] = *value;
        }
        Ok(values.clone())
    }
}

/// Operator that evaluates only a selected set of global rows.
///
/// Unlike a full [`LinearOperator`] action, a row-block action lets a
/// distributed rank evaluate exactly the rows it owns, so operator work
/// and memory scale with the local partition instead of replicating the
/// global action on every rank.
pub trait RowBlockOperator<T: ComplexField>: Send + Sync {
    /// Evaluates `rows` of the operator against the global input vector.
    ///
    /// # Errors
    /// Returns [`BemError`] when a row cannot be evaluated.
    fn apply_rows(&self, rows: &[usize], global_values: &[T]) -> Result<Vec<T>, BemError>;

    /// Number of global input columns the operator expects.
    fn num_cols(&self) -> usize;
}

/// Rank-local operator view backed by a [`RowBlockOperator`].
///
/// The input is all-gathered through the exchange, but only the rows
/// owned by `local_rank` are evaluated. Prefer this over
/// [`DistributedLinearOperator`] (which evaluates the full global action
/// on every rank) whenever the operator can apply selected rows.
pub struct DistributedRowOperator<'a, T, O: ?Sized, X: ?Sized> {
    plan: &'a DistributedRowPlan,
    local_rank: usize,
    operator: &'a O,
    exchange: &'a X,
    local_dimension: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<'a, T, O: ?Sized, X: ?Sized> DistributedRowOperator<'a, T, O, X>
where
    T: ComplexField,
    O: RowBlockOperator<T>,
    X: DistributedVectorExchange<T>,
{
    /// Validates dimensions and constructs a rank-local row operator.
    ///
    /// # Errors
    /// Returns [`BemError`] for an invalid plan, unknown rank, or an
    /// operator whose column count differs from the plan's DOF count.
    pub fn try_new(
        plan: &'a DistributedRowPlan,
        local_rank: usize,
        operator: &'a O,
        exchange: &'a X,
    ) -> Result<Self, BemError> {
        plan.validate()?;
        let partition = plan.local(local_rank).ok_or_else(|| {
            BemError::InvalidParameters(format!(
                "distributed row operator rank {local_rank} has no partition"
            ))
        })?;
        if operator.num_cols() != plan.num_dofs {
            return Err(BemError::InvalidParameters(format!(
                "distributed row operator expects {} global columns, found {}",
                plan.num_dofs,
                operator.num_cols()
            )));
        }
        Ok(Self {
            plan,
            local_rank,
            operator,
            exchange,
            local_dimension: partition.dof_indices.len(),
            _marker: std::marker::PhantomData,
        })
    }

    /// Number of rows owned by this rank.
    pub fn local_dimension(&self) -> usize {
        self.local_dimension
    }

    /// Checked rank-local action evaluating only this rank's rows.
    ///
    /// # Errors
    /// Returns [`BemError`] on dimension mismatch, non-finite values,
    /// exchange failure, or operator failure.
    pub fn try_apply_local(&self, local_values: &[T]) -> Result<Vec<T>, BemError> {
        if local_values.len() != self.local_dimension {
            return Err(BemError::InvalidParameters(format!(
                "distributed row operator rank {} expects {} values, found {}",
                self.local_rank,
                self.local_dimension,
                local_values.len()
            )));
        }
        if local_values.iter().any(|value| !value.norm().is_finite()) {
            return Err(BemError::InvalidParameters(
                "distributed row operator received non-finite values".to_string(),
            ));
        }
        let global_values = self
            .exchange
            .all_gather(self.plan, self.local_rank, local_values)?;
        if global_values.len() != self.plan.num_dofs {
            return Err(BemError::InvalidParameters(format!(
                "distributed exchange returned {} values, expected {}",
                global_values.len(),
                self.plan.num_dofs
            )));
        }
        let partition = self.plan.local(self.local_rank).ok_or_else(|| {
            BemError::InvalidParameters(format!(
                "distributed row operator rank {} has no partition",
                self.local_rank
            ))
        })?;
        let local_action = self
            .operator
            .apply_rows(&partition.dof_indices, &global_values)?;
        if local_action.len() != self.local_dimension {
            return Err(BemError::InvalidParameters(format!(
                "distributed row operator returned {} rows, expected {}",
                local_action.len(),
                self.local_dimension
            )));
        }
        if local_action.iter().any(|value| !value.norm().is_finite()) {
            return Err(BemError::SolverFailed(
                "distributed row operator produced non-finite output".to_string(),
            ));
        }
        Ok(local_action)
    }
}

/// A rank-local matrix-free view of a global BEM operator.
///
/// Every action performs a checked all-gather, evaluates the global operator,
/// and returns only rows owned by `local_rank`.  This is the missing adapter
/// between deterministic ownership metadata and a distributed Krylov method;
/// applications provide the transport (MPI, shared memory, or another
/// collective implementation) without changing the BEM operator itself.
/// Operators that can evaluate selected rows should use
/// [`DistributedRowOperator`] instead: this adapter replicates the global
/// action on every rank and does not scale in work or memory.
pub struct DistributedLinearOperator<'a, T, O: ?Sized, X: ?Sized> {
    plan: &'a DistributedRowPlan,
    local_rank: usize,
    operator: &'a O,
    exchange: &'a X,
    last_error: Mutex<Option<String>>,
    local_dimension: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<'a, T, O: ?Sized, X: ?Sized> DistributedLinearOperator<'a, T, O, X>
where
    T: ComplexField,
    O: LinearOperator<T>,
    X: DistributedVectorExchange<T>,
{
    /// Validate dimensions and construct a rank-local operator view.
    pub fn try_new(
        plan: &'a DistributedRowPlan,
        local_rank: usize,
        operator: &'a O,
        exchange: &'a X,
    ) -> Result<Self, BemError> {
        plan.validate()?;
        let partition = plan.local(local_rank).ok_or_else(|| {
            BemError::InvalidParameters(format!(
                "distributed operator rank {local_rank} has no partition"
            ))
        })?;
        if operator.num_rows() != plan.num_dofs || operator.num_cols() != plan.num_dofs {
            return Err(BemError::InvalidParameters(format!(
                "distributed operator expects {}x{} global operator, found {}x{}",
                plan.num_dofs,
                plan.num_dofs,
                operator.num_rows(),
                operator.num_cols()
            )));
        }
        Ok(Self {
            plan,
            local_rank,
            operator,
            exchange,
            last_error: Mutex::new(None),
            local_dimension: partition.dof_indices.len(),
            _marker: std::marker::PhantomData,
        })
    }

    /// Number of rows owned by this rank.
    pub fn local_dimension(&self) -> usize {
        self.local_dimension
    }

    /// Checked rank-local action.
    pub fn try_apply_local(&self, local_values: &[T]) -> Result<Vec<T>, BemError> {
        if local_values.len() != self.local_dimension {
            return Err(BemError::InvalidParameters(format!(
                "distributed operator rank {} expects {} values, found {}",
                self.local_rank,
                self.local_dimension,
                local_values.len()
            )));
        }
        if local_values.iter().any(|value| !value.norm().is_finite()) {
            return Err(BemError::InvalidParameters(
                "distributed operator received non-finite values".to_string(),
            ));
        }
        let global_values = self
            .exchange
            .all_gather(self.plan, self.local_rank, local_values)?;
        if global_values.len() != self.plan.num_dofs {
            return Err(BemError::InvalidParameters(format!(
                "distributed exchange returned {} values, expected {}",
                global_values.len(),
                self.plan.num_dofs
            )));
        }
        let global_action = self
            .operator
            .try_apply(&global_values)
            .map_err(|error| BemError::SolverFailed(error.to_string()))?;
        if global_action.len() != self.plan.num_dofs {
            return Err(BemError::InvalidParameters(format!(
                "distributed operator returned {} rows, expected {}",
                global_action.len(),
                self.plan.num_dofs
            )));
        }
        let partition = self.plan.local(self.local_rank).ok_or_else(|| {
            BemError::InvalidParameters(format!(
                "distributed operator rank {} has no partition",
                self.local_rank
            ))
        })?;
        let local_action = partition
            .dof_indices
            .iter()
            .map(|&dof| global_action[dof])
            .collect::<Vec<_>>();
        if local_action.iter().any(|value| !value.norm().is_finite()) {
            return Err(BemError::SolverFailed(
                "distributed operator produced non-finite output".to_string(),
            ));
        }
        Ok(local_action)
    }

    /// Checked rank-local block action for multiple right-hand sides.
    pub fn try_apply_block_local(
        &self,
        local_values: &DenseMatrix<T>,
    ) -> Result<DenseMatrix<T>, BemError> {
        if local_values.nrows() != self.local_dimension {
            return Err(BemError::InvalidParameters(format!(
                "distributed operator rank {} expects {} block rows, found {}",
                self.local_rank,
                self.local_dimension,
                local_values.nrows()
            )));
        }
        if local_values
            .as_slice()
            .iter()
            .any(|value| !value.norm().is_finite())
        {
            return Err(BemError::InvalidParameters(
                "distributed operator block received non-finite values".to_string(),
            ));
        }

        let global_values =
            self.exchange
                .all_gather_block(self.plan, self.local_rank, local_values)?;
        if global_values.nrows() != self.plan.num_dofs {
            return Err(BemError::InvalidParameters(format!(
                "distributed exchange returned {} block rows, expected {}",
                global_values.nrows(),
                self.plan.num_dofs
            )));
        }

        let mut global_action = DenseMatrix::zeros(self.plan.num_dofs, local_values.ncols());
        self.operator
            .try_apply_block(&global_values, &mut global_action)
            .map_err(|error| {
                BemError::SolverFailed(format!("distributed block action: {error}"))
            })?;

        let partition = self.plan.local(self.local_rank).ok_or_else(|| {
            BemError::InvalidParameters(format!(
                "distributed operator rank {} has no partition",
                self.local_rank
            ))
        })?;
        let mut local_action = DenseMatrix::zeros(self.local_dimension, local_values.ncols());
        for (local_row, &global_row) in partition.dof_indices.iter().enumerate() {
            for column in 0..local_values.ncols() {
                local_action[[local_row, column]] = global_action[[global_row, column]];
            }
        }
        if local_action
            .as_slice()
            .iter()
            .any(|value| !value.norm().is_finite())
        {
            return Err(BemError::SolverFailed(
                "distributed block operator produced non-finite output".to_string(),
            ));
        }
        Ok(local_action)
    }

    fn try_apply_other_local<F>(
        &self,
        local_values: &[T],
        operation: &str,
        apply: F,
    ) -> Result<Vec<T>, BemError>
    where
        F: FnOnce(&O, &[T]) -> Result<Vec<T>, sonium_solvers::traits::SolverError>,
    {
        if local_values.len() != self.local_dimension {
            return Err(BemError::InvalidParameters(format!(
                "distributed operator rank {} expects {} values, found {}",
                self.local_rank,
                self.local_dimension,
                local_values.len()
            )));
        }
        if local_values.iter().any(|value| !value.norm().is_finite()) {
            return Err(BemError::InvalidParameters(format!(
                "distributed operator {operation} received non-finite values"
            )));
        }
        let global_values = self
            .exchange
            .all_gather(self.plan, self.local_rank, local_values)?;
        if global_values.len() != self.plan.num_dofs {
            return Err(BemError::InvalidParameters(format!(
                "distributed exchange returned {} values, expected {}",
                global_values.len(),
                self.plan.num_dofs
            )));
        }
        let global_action = apply(self.operator, &global_values)
            .map_err(|error| BemError::SolverFailed(format!("distributed {operation}: {error}")))?;
        if global_action.len() != self.plan.num_dofs {
            return Err(BemError::InvalidParameters(format!(
                "distributed {operation} returned {} rows, expected {}",
                global_action.len(),
                self.plan.num_dofs
            )));
        }
        let partition = self.plan.local(self.local_rank).ok_or_else(|| {
            BemError::InvalidParameters(format!(
                "distributed operator rank {} has no partition",
                self.local_rank
            ))
        })?;
        let local_action = partition
            .dof_indices
            .iter()
            .map(|&dof| global_action[dof])
            .collect::<Vec<_>>();
        if local_action.iter().any(|value| !value.norm().is_finite()) {
            return Err(BemError::SolverFailed(format!(
                "distributed {operation} produced non-finite output"
            )));
        }
        Ok(local_action)
    }

    /// Checked algebraic-transpose rank-local action.
    pub fn try_apply_transpose_local(&self, local_values: &[T]) -> Result<Vec<T>, BemError> {
        self.try_apply_other_local(local_values, "transpose", |operator, values| {
            operator.try_apply_transpose(values)
        })
    }

    /// Checked Hermitian-adjoint rank-local action.
    pub fn try_apply_hermitian_local(&self, local_values: &[T]) -> Result<Vec<T>, BemError> {
        self.try_apply_other_local(local_values, "Hermitian", |operator, values| {
            operator.try_apply_hermitian(values)
        })
    }

    fn try_apply_other_block_local<F>(
        &self,
        local_values: &DenseMatrix<T>,
        operation: &str,
        apply: F,
    ) -> Result<DenseMatrix<T>, BemError>
    where
        F: FnOnce(
            &O,
            &DenseMatrix<T>,
            &mut DenseMatrix<T>,
        ) -> Result<(), sonium_solvers::traits::SolverError>,
    {
        if local_values.nrows() != self.local_dimension {
            return Err(BemError::InvalidParameters(format!(
                "distributed operator rank {} expects {} block rows, found {}",
                self.local_rank,
                self.local_dimension,
                local_values.nrows()
            )));
        }
        if local_values
            .as_slice()
            .iter()
            .any(|value| !value.norm().is_finite())
        {
            return Err(BemError::InvalidParameters(format!(
                "distributed operator {operation} block received non-finite values"
            )));
        }

        let global_values =
            self.exchange
                .all_gather_block(self.plan, self.local_rank, local_values)?;
        if global_values.nrows() != self.plan.num_dofs {
            return Err(BemError::InvalidParameters(format!(
                "distributed exchange returned {} block rows for {operation}, expected {}",
                global_values.nrows(),
                self.plan.num_dofs
            )));
        }
        let mut global_action = DenseMatrix::zeros(self.plan.num_dofs, local_values.ncols());
        apply(self.operator, &global_values, &mut global_action).map_err(|error| {
            BemError::SolverFailed(format!("distributed {operation} block action: {error}"))
        })?;

        let partition = self.plan.local(self.local_rank).ok_or_else(|| {
            BemError::InvalidParameters(format!(
                "distributed operator rank {} has no partition",
                self.local_rank
            ))
        })?;
        let mut local_action = DenseMatrix::zeros(self.local_dimension, local_values.ncols());
        for (local_row, &global_row) in partition.dof_indices.iter().enumerate() {
            for column in 0..local_values.ncols() {
                local_action[[local_row, column]] = global_action[[global_row, column]];
            }
        }
        if local_action
            .as_slice()
            .iter()
            .any(|value| !value.norm().is_finite())
        {
            return Err(BemError::SolverFailed(format!(
                "distributed {operation} block produced non-finite output"
            )));
        }
        Ok(local_action)
    }

    /// Checked algebraic-transpose block action for multiple right-hand sides.
    pub fn try_apply_transpose_block_local(
        &self,
        local_values: &DenseMatrix<T>,
    ) -> Result<DenseMatrix<T>, BemError> {
        self.try_apply_other_block_local(local_values, "transpose", |operator, values, output| {
            operator.try_apply_transpose_block(values, output)
        })
    }

    /// Checked Hermitian-adjoint block action for multiple right-hand sides.
    pub fn try_apply_hermitian_block_local(
        &self,
        local_values: &DenseMatrix<T>,
    ) -> Result<DenseMatrix<T>, BemError> {
        self.try_apply_other_block_local(local_values, "Hermitian", |operator, values, output| {
            operator.try_apply_hermitian_block(values, output)
        })
    }

    /// Recompute a relative residual over the complete distributed vector.
    ///
    /// Krylov methods operate on rank-owned rows, so their local residual can
    /// be small even when a remote row is still inaccurate.  This checked
    /// post-solve certificate gathers the candidate and RHS, evaluates the
    /// original global operator, and uses `||A x - b|| / max(||b||, eps)`.
    /// The candidate is gathered once more before returning so a loopback or
    /// MPI transport retains the solution rather than the temporary RHS.
    pub fn try_global_relative_residual(
        &self,
        local_solution: &[T],
        local_rhs: &[T],
    ) -> Result<T::Real, BemError> {
        self.validate_local_vector(local_solution, "solution")?;
        self.validate_local_vector(local_rhs, "RHS")?;

        let global_solution =
            self.exchange
                .all_gather(self.plan, self.local_rank, local_solution)?;
        let global_rhs = self
            .exchange
            .all_gather(self.plan, self.local_rank, local_rhs)?;
        // all_gather updates this rank's owned entries.  Restore the candidate
        // before any subsequent Krylov action uses the shared transport.
        self.exchange
            .all_gather(self.plan, self.local_rank, local_solution)?;

        if global_solution.len() != self.plan.num_dofs || global_rhs.len() != self.plan.num_dofs {
            return Err(BemError::InvalidParameters(
                "distributed residual gather returned an invalid global length".to_string(),
            ));
        }
        if global_solution
            .iter()
            .chain(global_rhs.iter())
            .any(|value| !value.norm().is_finite())
        {
            return Err(BemError::InvalidParameters(
                "distributed residual gather returned non-finite values".to_string(),
            ));
        }
        let global_action = self.operator.try_apply(&global_solution).map_err(|error| {
            BemError::SolverFailed(format!("distributed residual action: {error}"))
        })?;
        if global_action.len() != self.plan.num_dofs {
            return Err(BemError::InvalidParameters(format!(
                "distributed residual action returned {} rows, expected {}",
                global_action.len(),
                self.plan.num_dofs
            )));
        }
        checked_relative_residual(&global_action, &global_rhs)
    }

    /// Recompute relative residuals for every column of a distributed block.
    /// See [`Self::try_global_relative_residual`] for the scalar contract.
    pub fn try_global_relative_residual_block(
        &self,
        local_solution: &DenseMatrix<T>,
        local_rhs: &DenseMatrix<T>,
    ) -> Result<Vec<T::Real>, BemError> {
        if local_solution.nrows() != self.local_dimension
            || local_rhs.nrows() != self.local_dimension
            || local_solution.ncols() != local_rhs.ncols()
        {
            return Err(BemError::InvalidParameters(format!(
                "distributed residual block expects {}x{} solution/RHS, found {}x{} and {}x{}",
                self.local_dimension,
                local_rhs.ncols(),
                local_solution.nrows(),
                local_solution.ncols(),
                local_rhs.nrows(),
                local_rhs.ncols()
            )));
        }
        if local_solution
            .as_slice()
            .iter()
            .chain(local_rhs.as_slice())
            .any(|value| !value.norm().is_finite())
        {
            return Err(BemError::InvalidParameters(
                "distributed residual block received non-finite values".to_string(),
            ));
        }

        let global_solution =
            self.exchange
                .all_gather_block(self.plan, self.local_rank, local_solution)?;
        let global_rhs = self
            .exchange
            .all_gather_block(self.plan, self.local_rank, local_rhs)?;
        self.exchange
            .all_gather_block(self.plan, self.local_rank, local_solution)?;
        if global_solution.nrows() != self.plan.num_dofs
            || global_rhs.nrows() != self.plan.num_dofs
            || global_solution.ncols() != local_solution.ncols()
            || global_rhs.ncols() != local_rhs.ncols()
        {
            return Err(BemError::InvalidParameters(
                "distributed residual block gather returned an invalid shape".to_string(),
            ));
        }

        if global_solution
            .as_slice()
            .iter()
            .chain(global_rhs.as_slice())
            .any(|value| !value.norm().is_finite())
        {
            return Err(BemError::InvalidParameters(
                "distributed residual block gather returned non-finite values".to_string(),
            ));
        }
        let mut global_action = DenseMatrix::zeros(self.plan.num_dofs, local_solution.ncols());
        self.operator
            .try_apply_block(&global_solution, &mut global_action)
            .map_err(|error| {
                BemError::SolverFailed(format!("distributed residual block action: {error}"))
            })?;
        (0..global_solution.ncols())
            .map(|column| {
                let action = global_action.column_to_vec(column);
                let rhs = global_rhs.column_to_vec(column);
                checked_relative_residual(&action, &rhs)
            })
            .collect()
    }

    fn validate_local_vector(&self, values: &[T], label: &str) -> Result<(), BemError> {
        if values.len() != self.local_dimension {
            return Err(BemError::InvalidParameters(format!(
                "distributed residual {label} expects {} values, found {}",
                self.local_dimension,
                values.len()
            )));
        }
        if values.iter().any(|value| !value.norm().is_finite()) {
            return Err(BemError::InvalidParameters(format!(
                "distributed residual {label} received non-finite values"
            )));
        }
        Ok(())
    }

    fn record_error(&self, error: BemError) {
        let mut slot = self
            .last_error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if slot.is_none() {
            *slot = Some(error.to_string());
        }
    }

    fn take_error(&self) -> Option<String> {
        self.last_error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
    }
}

fn checked_relative_residual<T: ComplexField>(
    action: &[T],
    rhs: &[T],
) -> Result<T::Real, BemError> {
    if action.len() != rhs.len() {
        return Err(BemError::InvalidParameters(format!(
            "distributed residual action/RHS length mismatch: {} versus {}",
            action.len(),
            rhs.len()
        )));
    }
    if action
        .iter()
        .chain(rhs.iter())
        .any(|value| !value.norm().is_finite())
    {
        return Err(BemError::SolverFailed(
            "distributed residual action or RHS is non-finite".to_string(),
        ));
    }
    let residual_norm = action
        .iter()
        .zip(rhs)
        .map(|(a, b)| *a - *b)
        .collect::<Vec<_>>();
    let residual_norm = T::vec_norm(&residual_norm);
    let rhs_norm = T::vec_norm(rhs).max(T::Real::min_positive_value());
    let relative = residual_norm / rhs_norm;
    if !relative.is_finite() {
        return Err(BemError::SolverFailed(
            "distributed residual certificate is non-finite".to_string(),
        ));
    }
    Ok(relative)
}

impl<T, O: ?Sized, X: ?Sized> LinearOperator<T> for DistributedLinearOperator<'_, T, O, X>
where
    T: ComplexField,
    O: LinearOperator<T>,
    X: DistributedVectorExchange<T>,
{
    fn num_rows(&self) -> usize {
        self.local_dimension
    }

    fn num_cols(&self) -> usize {
        self.local_dimension
    }

    fn apply_into(&self, x: &[T], y: &mut [T]) {
        match self.try_apply_local(x) {
            Ok(values) if values.len() == y.len() => y.copy_from_slice(&values),
            Ok(values) => {
                self.record_error(BemError::InvalidParameters(format!(
                    "distributed operator output has {} values, destination has {}",
                    values.len(),
                    y.len()
                )));
                panic!("distributed operator output dimension mismatch");
            }
            Err(error) => {
                self.record_error(error);
                panic!("distributed operator action failed");
            }
        }
    }

    fn apply_block(&self, x: &DenseMatrix<T>, y: &mut DenseMatrix<T>) {
        match self.try_apply_block_local(x) {
            Ok(values) if values.nrows() == y.nrows() && values.ncols() == y.ncols() => {
                y.as_mut_slice().copy_from_slice(values.as_slice());
            }
            Ok(values) => {
                self.record_error(BemError::InvalidParameters(format!(
                    "distributed block operator output is {}x{}, destination is {}x{}",
                    values.nrows(),
                    values.ncols(),
                    y.nrows(),
                    y.ncols()
                )));
                panic!("distributed block operator output dimension mismatch");
            }
            Err(error) => {
                self.record_error(error);
                panic!("distributed block operator action failed");
            }
        }
    }

    fn apply_transpose_into(&self, _x: &[T], _y: &mut [T]) {
        match self.try_apply_transpose_local(_x) {
            Ok(values) if values.len() == _y.len() => _y.copy_from_slice(&values),
            Ok(_) | Err(_) => panic!("distributed transpose action failed"),
        }
    }

    fn apply_transpose_block(&self, x: &DenseMatrix<T>, y: &mut DenseMatrix<T>) {
        match self.try_apply_transpose_block_local(x) {
            Ok(values) if values.nrows() == y.nrows() && values.ncols() == y.ncols() => {
                y.as_mut_slice().copy_from_slice(values.as_slice());
            }
            Ok(values) => {
                self.record_error(BemError::InvalidParameters(format!(
                    "distributed transpose block output is {}x{}, destination is {}x{}",
                    values.nrows(),
                    values.ncols(),
                    y.nrows(),
                    y.ncols()
                )));
                panic!("distributed transpose block output dimension mismatch");
            }
            Err(error) => {
                self.record_error(error);
                panic!("distributed transpose block action failed");
            }
        }
    }

    fn apply_hermitian_block(&self, x: &DenseMatrix<T>, y: &mut DenseMatrix<T>) {
        match self.try_apply_hermitian_block_local(x) {
            Ok(values) if values.nrows() == y.nrows() && values.ncols() == y.ncols() => {
                y.as_mut_slice().copy_from_slice(values.as_slice());
            }
            Ok(values) => {
                self.record_error(BemError::InvalidParameters(format!(
                    "distributed Hermitian block output is {}x{}, destination is {}x{}",
                    values.nrows(),
                    values.ncols(),
                    y.nrows(),
                    y.ncols()
                )));
                panic!("distributed Hermitian block output dimension mismatch");
            }
            Err(error) => {
                self.record_error(error);
                panic!("distributed Hermitian block action failed");
            }
        }
    }

    fn transpose_capability(&self) -> AdjointCapability {
        self.operator.transpose_capability()
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        self.operator.hermitian_capability()
    }
}

/// Solve a rank-local distributed BEM system with checked matrix-free GMRES.
pub fn try_gmres_distributed<T, O, X>(
    plan: &DistributedRowPlan,
    local_rank: usize,
    operator: &O,
    exchange: &X,
    local_rhs: &[T],
    config: &GmresConfig<T::Real>,
) -> Result<GmresSolution<T>, BemError>
where
    T: ComplexField,
    O: LinearOperator<T> + ?Sized,
    X: DistributedVectorExchange<T> + ?Sized,
{
    let distributed_operator =
        DistributedLinearOperator::try_new(plan, local_rank, operator, exchange)?;
    if local_rhs.len() != distributed_operator.local_dimension() {
        return Err(BemError::InvalidParameters(format!(
            "distributed GMRES RHS expects {} values, found {}",
            distributed_operator.local_dimension(),
            local_rhs.len()
        )));
    }
    let result = sonium_solvers::iterative::try_gmres(&distributed_operator, local_rhs, config);
    if let Some(error) = distributed_operator.take_error() {
        return Err(BemError::SolverFailed(format!(
            "distributed GMRES action failed: {error}"
        )));
    }
    let mut solution = result
        .map_err(|error| BemError::SolverFailed(format!("distributed GMRES failed: {error}")))?;
    let global_residual =
        distributed_operator.try_global_relative_residual(&solution.x, local_rhs)?;
    if solution.converged && global_residual > config.tolerance {
        return Err(BemError::SolverFailed(format!(
            "distributed GMRES local convergence masked global residual {global_residual:?} > tolerance {tolerance:?}",
            tolerance = config.tolerance
        )));
    }
    solution.residual = global_residual;
    Ok(solution)
}

/// Solve a rank-local distributed BEM system with checked matrix-free block
/// GMRES. Each local RHS column is solved together while the exchange gathers
/// the complete global vector required by the matrix-free operator.
///
/// The returned block solution retains the iterative solver's true residuals;
/// transport and preconditioner failures crossing the infallible solver traits
/// are converted to [`BemError`] before the result is published.
pub fn try_block_gmres_distributed<T, O, X>(
    plan: &DistributedRowPlan,
    local_rank: usize,
    operator: &O,
    exchange: &X,
    local_rhs: &DenseMatrix<T>,
    config: &BlockGmresConfig<T::Real>,
) -> Result<BlockGmresSolution<T>, BemError>
where
    T: ComplexField,
    O: LinearOperator<T> + ?Sized,
    X: DistributedVectorExchange<T> + ?Sized,
{
    let distributed_operator =
        DistributedLinearOperator::try_new(plan, local_rank, operator, exchange)?;
    if local_rhs.nrows() != distributed_operator.local_dimension() {
        return Err(BemError::InvalidParameters(format!(
            "distributed block GMRES RHS expects {} rows, found {}",
            distributed_operator.local_dimension(),
            local_rhs.nrows()
        )));
    }
    let result =
        sonium_solvers::iterative::try_block_gmres(&distributed_operator, local_rhs, config);
    if let Some(error) = distributed_operator.take_error() {
        return Err(BemError::SolverFailed(format!(
            "distributed block GMRES action failed: {error}"
        )));
    }
    let mut solution = result.map_err(|error| {
        BemError::SolverFailed(format!("distributed block GMRES failed: {error}"))
    })?;
    let global_residuals =
        distributed_operator.try_global_relative_residual_block(&solution.x, local_rhs)?;
    if solution.converged
        && global_residuals
            .iter()
            .any(|residual| *residual > config.tolerance)
    {
        let maximum = global_residuals
            .iter()
            .copied()
            .fold(T::Real::zero(), T::Real::max);
        return Err(BemError::SolverFailed(format!(
            "distributed block GMRES local convergence masked global residual {maximum:?} > tolerance {tolerance:?}",
            tolerance = config.tolerance
        )));
    }
    solution.true_residuals = global_residuals;
    Ok(solution)
}

/// Rank-local wrapper for a preconditioner that already operates on owned
/// rows.  It preserves checked solver failures across the infallible
/// `Preconditioner::apply_into` trait boundary.
pub struct DistributedLocalPreconditioner<'a, T, P: ?Sized> {
    preconditioner: &'a P,
    dimension: usize,
    last_error: Mutex<Option<String>>,
    _marker: std::marker::PhantomData<T>,
}

impl<'a, T, P: ?Sized> DistributedLocalPreconditioner<'a, T, P>
where
    T: ComplexField,
    P: Preconditioner<T>,
{
    /// Construct a checked rank-local preconditioner view.
    pub fn try_new(preconditioner: &'a P, dimension: usize) -> Result<Self, BemError> {
        if dimension == 0 {
            return Err(BemError::InvalidParameters(
                "distributed preconditioner requires non-empty rank".to_string(),
            ));
        }
        Ok(Self {
            preconditioner,
            dimension,
            last_error: Mutex::new(None),
            _marker: std::marker::PhantomData,
        })
    }

    fn record_error(&self, error: sonium_solvers::traits::SolverError) {
        let mut slot = self
            .last_error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if slot.is_none() {
            *slot = Some(error.to_string());
        }
    }

    fn take_error(&self) -> Option<String> {
        self.last_error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
    }
}

impl<T, P: ?Sized> Preconditioner<T> for DistributedLocalPreconditioner<'_, T, P>
where
    T: ComplexField,
    P: Preconditioner<T>,
{
    fn apply_into(&self, r: &[T], y: &mut [T]) {
        if r.len() != self.dimension || y.len() != self.dimension {
            self.record_error(sonium_solvers::traits::SolverError::DimensionMismatch {
                expected: self.dimension,
                got: r.len().max(y.len()),
            });
            panic!("distributed preconditioner dimension mismatch");
        }
        match self.preconditioner.try_apply(r) {
            Ok(values) if values.len() == y.len() => y.copy_from_slice(&values),
            Ok(values) => {
                self.record_error(sonium_solvers::traits::SolverError::DimensionMismatch {
                    expected: y.len(),
                    got: values.len(),
                });
                panic!("distributed preconditioner output dimension mismatch");
            }
            Err(error) => {
                self.record_error(error);
                panic!("distributed preconditioner action failed");
            }
        }
    }
}

/// Solve a rank-local distributed BEM system with a local preconditioner.
pub fn try_gmres_distributed_preconditioned<T, O, X, P>(
    plan: &DistributedRowPlan,
    local_rank: usize,
    operator: &O,
    preconditioner: &P,
    exchange: &X,
    local_rhs: &[T],
    config: &GmresConfig<T::Real>,
) -> Result<GmresSolution<T>, BemError>
where
    T: ComplexField,
    O: LinearOperator<T> + ?Sized,
    X: DistributedVectorExchange<T> + ?Sized,
    P: Preconditioner<T> + ?Sized,
{
    let distributed_operator =
        DistributedLinearOperator::try_new(plan, local_rank, operator, exchange)?;
    if local_rhs.len() != distributed_operator.local_dimension() {
        return Err(BemError::InvalidParameters(format!(
            "distributed GMRES RHS expects {} values, found {}",
            distributed_operator.local_dimension(),
            local_rhs.len()
        )));
    }
    let distributed_preconditioner = DistributedLocalPreconditioner::try_new(
        preconditioner,
        distributed_operator.local_dimension(),
    )?;
    let result = sonium_solvers::iterative::try_gmres_preconditioned(
        &distributed_operator,
        &distributed_preconditioner,
        local_rhs,
        config,
    );
    if let Some(error) = distributed_operator.take_error() {
        return Err(BemError::SolverFailed(format!(
            "distributed GMRES action failed: {error}"
        )));
    }
    if let Some(error) = distributed_preconditioner.take_error() {
        return Err(BemError::SolverFailed(format!(
            "distributed GMRES preconditioner failed: {error}"
        )));
    }
    let mut solution = result
        .map_err(|error| BemError::SolverFailed(format!("distributed GMRES failed: {error}")))?;
    let global_residual =
        distributed_operator.try_global_relative_residual(&solution.x, local_rhs)?;
    if solution.converged && global_residual > config.tolerance {
        return Err(BemError::SolverFailed(format!(
            "distributed GMRES local convergence masked global residual {global_residual:?} > tolerance {tolerance:?}",
            tolerance = config.tolerance
        )));
    }
    solution.residual = global_residual;
    Ok(solution)
}

/// Solve a rank-local distributed BEM block system with a checked local
/// preconditioner. The preconditioner is applied independently to each owned
/// row column while operator actions continue to use transport-backed global
/// gathers.
pub fn try_block_gmres_distributed_preconditioned<T, O, X, P>(
    plan: &DistributedRowPlan,
    local_rank: usize,
    operator: &O,
    preconditioner: &P,
    exchange: &X,
    local_rhs: &DenseMatrix<T>,
    config: &BlockGmresConfig<T::Real>,
) -> Result<BlockGmresSolution<T>, BemError>
where
    T: ComplexField,
    O: LinearOperator<T> + ?Sized,
    X: DistributedVectorExchange<T> + ?Sized,
    P: Preconditioner<T> + ?Sized,
{
    let distributed_operator =
        DistributedLinearOperator::try_new(plan, local_rank, operator, exchange)?;
    if local_rhs.nrows() != distributed_operator.local_dimension() {
        return Err(BemError::InvalidParameters(format!(
            "distributed block GMRES RHS expects {} rows, found {}",
            distributed_operator.local_dimension(),
            local_rhs.nrows()
        )));
    }
    let distributed_preconditioner = DistributedLocalPreconditioner::try_new(
        preconditioner,
        distributed_operator.local_dimension(),
    )?;
    let result = sonium_solvers::iterative::try_block_gmres_preconditioned(
        &distributed_operator,
        local_rhs,
        &distributed_preconditioner,
        None,
        config,
    );
    if let Some(error) = distributed_operator.take_error() {
        return Err(BemError::SolverFailed(format!(
            "distributed block GMRES action failed: {error}"
        )));
    }
    if let Some(error) = distributed_preconditioner.take_error() {
        return Err(BemError::SolverFailed(format!(
            "distributed block GMRES preconditioner failed: {error}"
        )));
    }
    let mut solution = result.map_err(|error| {
        BemError::SolverFailed(format!("distributed block GMRES failed: {error}"))
    })?;
    let global_residuals =
        distributed_operator.try_global_relative_residual_block(&solution.x, local_rhs)?;
    if solution.converged
        && global_residuals
            .iter()
            .any(|residual| *residual > config.tolerance)
    {
        let maximum = global_residuals
            .iter()
            .copied()
            .fold(T::Real::zero(), T::Real::max);
        return Err(BemError::SolverFailed(format!(
            "distributed block GMRES local convergence masked global residual {maximum:?} > tolerance {tolerance:?}",
            tolerance = config.tolerance
        )));
    }
    solution.true_residuals = global_residuals;
    Ok(solution)
}

/// Build a deterministic active-row ownership plan for `num_ranks` ranks.
///
/// The legacy entry point infers the global DOF cardinality from the largest
/// active element address.  Inferred layouts are still checked for dense
/// zero-based numbering and complete ownership; callers that already own an
/// explicit topology cardinality should prefer
/// [`plan_distributed_rows_with_dof_count`].
pub fn plan_distributed_rows(
    elements: &[Element],
    num_ranks: usize,
) -> Result<DistributedRowPlan, BemError> {
    let inferred_dof_count = elements
        .iter()
        .filter(|element| !element.property.is_evaluation())
        .flat_map(|element| element.dof_addresses.iter().copied())
        .max()
        .map_or(0, |dof| dof.saturating_add(1));
    plan_distributed_rows_with_dof_count(elements, num_ranks, inferred_dof_count)
}

/// Build a deterministic active-row ownership plan with an explicit global
/// DOF cardinality.
///
/// This is the checked entry point for continuous linear/quadratic meshes,
/// whose shared element incidences cannot be reduced to a panel count.  Every
/// active global DOF must occur exactly once in the ownership union and all
/// element-local addresses must be unique, dense, and zero-based.  These
/// invariants make the resulting pressure/velocity row expansion safe for
/// compressed forward and transpose actions.
pub fn plan_distributed_rows_with_dof_count(
    elements: &[Element],
    num_ranks: usize,
    global_dof_count: usize,
) -> Result<DistributedRowPlan, BemError> {
    if num_ranks == 0 {
        return Err(BemError::InvalidParameters(
            "distributed BEM requires at least one rank".to_string(),
        ));
    }

    let active: Vec<(usize, &Element)> = elements
        .iter()
        .enumerate()
        .filter(|(_, e)| !e.property.is_evaluation())
        .collect();
    if active.is_empty() {
        if global_dof_count != 0 {
            return Err(BemError::InvalidParameters(format!(
                "distributed row plan has no active elements but expects {global_dof_count} DOFs"
            )));
        }
        return Ok(DistributedRowPlan {
            partitions: (0..num_ranks)
                .map(|rank| RankPartition {
                    rank,
                    element_indices: Vec::new(),
                    dof_indices: Vec::new(),
                })
                .collect(),
            num_active_elements: 0,
            num_dofs: 0,
        });
    }
    if global_dof_count == 0 {
        return Err(BemError::InvalidParameters(
            "distributed active rows require at least one global DOF".to_string(),
        ));
    }

    for (element_index, element) in &active {
        let mut local_dofs = HashSet::with_capacity(element.dof_addresses.len());
        for &dof in &element.dof_addresses {
            if dof >= global_dof_count {
                return Err(BemError::InvalidParameters(format!(
                    "element {element_index} DOF {dof} is outside global range 0..{global_dof_count}"
                )));
            }
            if !local_dofs.insert(dof) {
                return Err(BemError::InvalidParameters(format!(
                    "element {element_index} repeats global DOF {dof}"
                )));
            }
        }
    }

    let bounds = active_bounds(&active)?;

    let mut ordered: Vec<(u64, usize, Vec<usize>)> = active
        .iter()
        .map(|(idx, elem)| {
            (
                morton_key(elem.center, bounds),
                *idx,
                elem.dof_addresses.clone(),
            )
        })
        .collect();
    ordered.sort_by_key(|(key, idx, _)| (*key, *idx));

    let mut partitions = Vec::with_capacity(num_ranks);
    let mut owned_dofs = HashSet::new();
    let n = ordered.len();
    for rank in 0..num_ranks {
        let start = rank * n / num_ranks;
        let end = (rank + 1) * n / num_ranks;
        let mut element_indices = Vec::with_capacity(end.saturating_sub(start));
        let mut dof_indices = Vec::new();
        for (_, elem_idx, dofs) in &ordered[start..end] {
            element_indices.push(*elem_idx);
            for &dof in dofs {
                // A shared continuous DOF belongs to the first rank that
                // encounters it in deterministic Morton/element order.
                // Subsequent element owners still retain their element row,
                // but do not duplicate the global row in the communication
                // ownership list.
                if owned_dofs.insert(dof) {
                    dof_indices.push(dof);
                }
            }
        }
        partitions.push(RankPartition {
            rank,
            element_indices,
            dof_indices,
        });
    }

    if owned_dofs.len() != global_dof_count
        || (0..global_dof_count).any(|dof| !owned_dofs.contains(&dof))
    {
        return Err(BemError::InvalidParameters(format!(
            "distributed row plan owns {} of {global_dof_count} dense global DOFs",
            owned_dofs.len()
        )));
    }
    Ok(DistributedRowPlan {
        partitions,
        num_active_elements: n,
        num_dofs: global_dof_count,
    })
}

fn active_bounds(active: &[(usize, &Element)]) -> Result<([f64; 3], [f64; 3]), BemError> {
    if active.is_empty() {
        return Ok(([0.0; 3], [1.0; 3]));
    }
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for (_, elem) in active {
        for axis in 0..3 {
            let x = elem.center[axis];
            if !x.is_finite() {
                return Err(BemError::InvalidMesh(
                    "distributed partitioning requires finite element centers".to_string(),
                ));
            }
            lo[axis] = lo[axis].min(x);
            hi[axis] = hi[axis].max(x);
        }
    }
    Ok((lo, hi))
}

fn morton_key(p: [f64; 3], bounds: ([f64; 3], [f64; 3])) -> u64 {
    let (lo, hi) = bounds;
    let mut q = [0_u32; 3];
    for axis in 0..3 {
        let span = (hi[axis] - lo[axis]).max(1.0e-30);
        let t = ((p[axis] - lo[axis]) / span).clamp(0.0, 1.0);
        q[axis] = (t * ((1_u32 << 21) - 1) as f64).round() as u32;
    }
    interleave3(q[0], q[1], q[2])
}

fn interleave3(x: u32, y: u32, z: u32) -> u64 {
    let mut key = 0_u64;
    for bit in 0..21 {
        key |= (((x >> bit) & 1) as u64) << (3 * bit);
        key |= (((y >> bit) & 1) as u64) << (3 * bit + 1);
        key |= (((z >> bit) & 1) as u64) << (3 * bit + 2);
    }
    key
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::mesh::generators::generate_icosphere_mesh;
    use num_complex::Complex64;

    fn active_elements() -> Vec<Element> {
        let mut elements = generate_icosphere_mesh(1.0, 2).elements;
        for (i, elem) in elements.iter_mut().enumerate() {
            elem.dof_addresses = vec![i];
            elem.boundary_condition =
                crate::core::types::BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
        }
        elements
    }

    #[test]
    fn distributed_plan_covers_each_active_row_once() {
        let elements = active_elements();
        let plan = plan_distributed_rows(&elements, 4).unwrap();
        assert_eq!(plan.num_active_elements, elements.len());
        assert_eq!(plan.partitions.len(), 4);

        let mut rows: Vec<usize> = plan
            .partitions
            .iter()
            .flat_map(|p| p.element_indices.iter().copied())
            .collect();
        rows.sort_unstable();
        assert_eq!(rows, (0..elements.len()).collect::<Vec<_>>());
    }

    #[test]
    fn distributed_plan_is_balanced_and_deterministic() {
        let elements = active_elements();
        let a = plan_distributed_rows(&elements, 3).unwrap();
        let b = plan_distributed_rows(&elements, 3).unwrap();
        assert_eq!(a, b);

        let sizes: Vec<usize> = a
            .partitions
            .iter()
            .map(|p| p.element_indices.len())
            .collect();
        let min = *sizes.iter().min().unwrap();
        let max = *sizes.iter().max().unwrap();
        assert!(max - min <= 1, "rank row imbalance > 1: {sizes:?}");
    }

    #[test]
    fn distributed_config_rejects_invalid_rank() {
        let cfg = DistributedBemConfig {
            num_ranks: 2,
            local_rank: 2,
            strategy: PartitionStrategy::Morton,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn shared_dofs_are_owned_by_exactly_one_rank() {
        let mut elements = active_elements();
        for (index, element) in elements.iter_mut().enumerate() {
            element.dof_addresses = vec![index, index + 1];
        }
        let plan = plan_distributed_rows(&elements, 3).expect("shared-DOF plan");
        let mut owned = plan
            .partitions
            .iter()
            .flat_map(|partition| partition.dof_indices.iter().copied())
            .collect::<Vec<_>>();
        owned.sort_unstable();
        let expected = (0..=elements.len()).collect::<Vec<_>>();
        assert_eq!(owned, expected);
        assert_eq!(plan.num_dofs, expected.len());
    }

    #[test]
    fn transmission_trace_rows_cover_both_blocks_once() {
        let elements = active_elements();
        let plan = plan_distributed_rows(&elements, 4).expect("row plan");
        let rows = plan
            .transmission_trace_rows(plan.num_dofs)
            .expect("two-trace row expansion");
        let mut flattened = rows.into_iter().flatten().collect::<Vec<_>>();
        flattened.sort_unstable();
        assert_eq!(flattened, (0..2 * plan.num_dofs).collect::<Vec<_>>());
    }

    #[test]
    fn distributed_vector_scatter_and_gather_round_trip() {
        let elements = active_elements();
        let plan = plan_distributed_rows(&elements, 4).expect("row plan");
        let global = (0..plan.num_dofs)
            .map(|dof| format!("value-{dof}"))
            .collect::<Vec<_>>();

        let local = plan.scatter_dof_values(&global).expect("scatter");
        assert_eq!(local.len(), 4);
        assert_eq!(plan.gather_dof_values(&local).expect("gather"), global);
        for dof in 0..plan.num_dofs {
            let owner = plan.owner_of_dof(dof).expect("owner");
            assert_eq!(
                local[plan
                    .partitions
                    .iter()
                    .position(|p| p.rank == owner)
                    .unwrap()]
                .len(),
                plan.local(owner).unwrap().dof_indices.len()
            );
        }
    }

    #[test]
    fn distributed_vector_layout_rejects_malformed_packets() {
        let elements = active_elements();
        let plan = plan_distributed_rows(&elements, 3).expect("row plan");
        let global = vec![0_u32; plan.num_dofs];
        assert!(
            plan.scatter_dof_values(&global[..plan.num_dofs - 1])
                .is_err()
        );
        assert!(plan.gather_dof_values::<u32>(&[]).is_err());

        let mut local = plan.scatter_dof_values(&global).expect("scatter");
        local[0].pop();
        assert!(plan.gather_dof_values(&local).is_err());

        let mut malformed = plan.clone();
        malformed.partitions[1].dof_indices[0] = malformed.partitions[0].dof_indices[0];
        assert!(malformed.owner_of_dof(0).is_err());
    }

    #[test]
    fn row_operator_evaluates_only_owned_rows() {
        struct CountingRows {
            matrix: Vec<Vec<Complex64>>,
            requested: Mutex<Vec<usize>>,
        }
        impl RowBlockOperator<Complex64> for CountingRows {
            fn apply_rows(
                &self,
                rows: &[usize],
                global_values: &[Complex64],
            ) -> Result<Vec<Complex64>, BemError> {
                self.requested.lock().unwrap().extend_from_slice(rows);
                Ok(rows
                    .iter()
                    .map(|&row| {
                        self.matrix[row]
                            .iter()
                            .zip(global_values)
                            .map(|(a, x)| a * x)
                            .sum()
                    })
                    .collect())
            }
            fn num_cols(&self) -> usize {
                self.matrix.len()
            }
        }

        let elements = active_elements();
        let plan = plan_distributed_rows(&elements, 3).expect("row plan");
        let n = plan.num_dofs;
        let matrix: Vec<Vec<Complex64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| Complex64::new((i * 10 + j) as f64, 0.0))
                    .collect()
            })
            .collect();
        let global_x: Vec<Complex64> = (0..n)
            .map(|j| Complex64::new(1.0 + j as f64, 0.0))
            .collect();
        let expected: Vec<Complex64> = matrix
            .iter()
            .map(|row| row.iter().zip(&global_x).map(|(a, x)| a * x).sum())
            .collect();

        let operator = CountingRows {
            matrix,
            requested: Mutex::new(Vec::new()),
        };
        let exchange = InProcessVectorExchange::new(global_x.clone());
        let local_slices = plan.scatter_dof_values(&global_x).expect("scatter");
        for rank in 0..3 {
            let partition = plan.local(rank).unwrap();
            let local_op =
                DistributedRowOperator::try_new(&plan, rank, &operator, &exchange).unwrap();
            let result = local_op
                .try_apply_local(&local_slices[rank])
                .expect("row-local apply");
            let expected_local: Vec<Complex64> = partition
                .dof_indices
                .iter()
                .map(|&dof| expected[dof])
                .collect();
            assert_eq!(result, expected_local, "rank {rank}");
        }
        let requested = operator.requested.lock().unwrap().clone();
        assert_eq!(
            requested.len(),
            n,
            "each row evaluated exactly once: {requested:?}"
        );
    }

    #[test]
    fn transmission_row_owner_maps_pressure_and_velocity_rows() {
        let elements = active_elements();
        let plan = plan_distributed_rows(&elements, 4).expect("row plan");
        for dof in 0..plan.num_dofs {
            let owner = plan.owner_of_dof(dof).expect("DOF owner");
            assert_eq!(
                plan.owner_of_transmission_row(dof, plan.num_dofs)
                    .expect("pressure row owner"),
                owner
            );
            assert_eq!(
                plan.owner_of_transmission_row(plan.num_dofs + dof, plan.num_dofs)
                    .expect("velocity row owner"),
                owner
            );
        }
        assert!(
            plan.owner_of_transmission_row(2 * plan.num_dofs, plan.num_dofs)
                .is_err()
        );
    }
}
