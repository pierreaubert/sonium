//! Generic boundary condition application via FunctionSpace trait
//!
//! Provides boundary condition operations that work with any [`FunctionSpace`],
//! independent of the physics domain. The legacy BC modules in this crate
//! operate on `HelmholtzProblem` / `HelmholtzMatrix` directly; these generic
//! functions operate on [`FemSystem`] instead.

use crate::assembly::TripletMatrix;
use crate::basis::PolynomialDegree;
use crate::mesh::Mesh;
use crate::quadrature::{gauss_legendre_1d, gauss_triangle};
use crate::spaces::{FieldType, FunctionSpace, VectorH1Space};
use num_complex::Complex64;
use num_traits::{Float, Zero};
use sonium_solvers::ComplexField;
use std::collections::{HashMap, HashSet};

/// A generic FEM system that boundary conditions can modify.
///
/// Replaces `HelmholtzProblem` as the BC target. Generic over the scalar type.
pub struct FemSystem<T> {
    /// System matrix in triplet format
    pub matrix: TripletMatrix<T>,
    /// Right-hand side vector
    pub rhs: Vec<T>,
    /// DOFs with Dirichlet conditions applied
    pub dirichlet_dofs: HashSet<usize>,
}

/// Structural validation failure for a generic FEM system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FemSystemError {
    /// Triplet dimensions do not agree with the RHS length.
    DimensionMismatch { matrix_rows: usize, rhs_len: usize },
    /// COO row, column, and value arrays have different lengths.
    TripletLengthMismatch {
        rows: usize,
        cols: usize,
        values: usize,
    },
    /// A triplet references a row or column outside the declared matrix.
    MatrixIndexOutOfBounds {
        triplet: usize,
        row: usize,
        col: usize,
        rows: usize,
        cols: usize,
    },
    /// The candidate solution has the wrong number of entries.
    SolutionLengthMismatch { expected: usize, actual: usize },
    /// Absolute or relative tolerance is negative or non-finite.
    InvalidTolerance,
    /// A residual or right-hand-side norm is non-finite.
    NonFiniteResidual,
}

impl std::fmt::Display for FemSystemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch {
                matrix_rows,
                rhs_len,
            } => {
                write!(
                    f,
                    "matrix has {matrix_rows} rows but RHS has {rhs_len} entries"
                )
            }
            Self::TripletLengthMismatch { rows, cols, values } => write!(
                f,
                "COO arrays have inconsistent lengths: rows={rows}, cols={cols}, values={values}"
            ),
            Self::MatrixIndexOutOfBounds {
                triplet,
                row,
                col,
                rows,
                cols,
            } => write!(
                f,
                "triplet {triplet} ({row}, {col}) outside {rows}x{cols} matrix"
            ),
            Self::SolutionLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "solution has {actual} entries but matrix expects {expected}"
                )
            }
            Self::InvalidTolerance => {
                write!(f, "residual tolerances must be finite and non-negative")
            }
            Self::NonFiniteResidual => write!(f, "residual or RHS norm is non-finite"),
        }
    }
}

impl std::error::Error for FemSystemError {}

/// Typed failure while applying a generic boundary condition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoundaryConditionError {
    /// The target sparse system is structurally invalid.
    InvalidSystem(FemSystemError),
    /// A boundary space returned a DOF outside the target system.
    DofOutOfBounds { dof: usize, num_dofs: usize },
    /// A non-nodal space did not provide the coordinate needed by the value function.
    MissingDofCoordinate { dof: usize },
    /// A non-nodal space needs the compatibility DOF-to-node callback.
    MissingDofToNode { dof: usize },
    /// The compatibility DOF-to-node callback returned a missing mesh node.
    NodeOutOfBounds { dof: usize, node: usize },
    /// A prescribed value was not finite.
    NonFiniteValue { dof: usize },
    /// A natural/Robin callback returned a non-finite value.
    NonFiniteBoundaryValue { boundary: usize },
    /// A boundary node or face index cannot be addressed by the target system.
    BoundaryNodeOutOfBounds {
        boundary: usize,
        node: usize,
        num_dofs: usize,
    },
    /// A boundary references a node absent from the mesh coordinate array.
    MeshNodeOutOfBounds {
        boundary: usize,
        node: usize,
        num_nodes: usize,
    },
    /// A boundary has an unsupported node count for the P1 natural-boundary contract.
    UnsupportedBoundary { boundary: usize, nodes: usize },
    /// A boundary edge/face has non-finite or degenerate geometry.
    InvalidBoundaryGeometry { boundary: usize },
    /// A vector-valued natural-boundary callback returned too few components.
    BoundaryValueDimension {
        boundary: usize,
        expected: usize,
        actual: usize,
    },
    /// The boundary corner topology could not be mapped to a higher-order trace.
    BoundaryTopologyMismatch { boundary: usize },
    /// The requested vector natural-boundary space/order is not supported.
    UnsupportedVectorSpace { components: usize, degree: usize },
}

impl std::fmt::Display for BoundaryConditionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSystem(error) => write!(f, "invalid FEM system: {error}"),
            Self::DofOutOfBounds { dof, num_dofs } => {
                write!(f, "boundary DOF {dof} is outside system size {num_dofs}")
            }
            Self::MissingDofCoordinate { dof } => {
                write!(
                    f,
                    "no physical coordinate is available for boundary DOF {dof}"
                )
            }
            Self::MissingDofToNode { dof } => {
                write!(f, "non-nodal boundary DOF {dof} requires a DOF-to-node map")
            }
            Self::NodeOutOfBounds { dof, node } => {
                write!(f, "boundary DOF {dof} maps to missing mesh node {node}")
            }
            Self::NonFiniteValue { dof } => {
                write!(f, "prescribed boundary value for DOF {dof} is non-finite")
            }
            Self::NonFiniteBoundaryValue { boundary } => {
                write!(
                    f,
                    "natural-boundary callback on boundary {boundary} is non-finite"
                )
            }
            Self::BoundaryNodeOutOfBounds {
                boundary,
                node,
                num_dofs,
            } => write!(
                f,
                "boundary {boundary} node {node} is outside the P1 system size {num_dofs}"
            ),
            Self::MeshNodeOutOfBounds {
                boundary,
                node,
                num_nodes,
            } => write!(
                f,
                "boundary {boundary} node {node} is outside the mesh size {num_nodes}"
            ),
            Self::UnsupportedBoundary { boundary, nodes } => write!(
                f,
                "boundary {boundary} has {nodes} nodes; only 2-node edges and 3-node faces are supported"
            ),
            Self::InvalidBoundaryGeometry { boundary } => {
                write!(
                    f,
                    "boundary {boundary} has non-finite or degenerate geometry"
                )
            }
            Self::BoundaryValueDimension {
                boundary,
                expected,
                actual,
            } => write!(
                f,
                "boundary {boundary} traction has {actual} components; expected at least {expected}"
            ),
            Self::BoundaryTopologyMismatch { boundary } => write!(
                f,
                "boundary {boundary} cannot be mapped to the requested higher-order trace"
            ),
            Self::UnsupportedVectorSpace { components, degree } => write!(
                f,
                "vector natural boundary space with {components} components and degree {degree} is unsupported"
            ),
        }
    }
}

impl std::error::Error for BoundaryConditionError {}

impl From<FemSystemError> for BoundaryConditionError {
    fn from(error: FemSystemError) -> Self {
        Self::InvalidSystem(error)
    }
}

impl<T: Copy + Default + std::ops::AddAssign + Send + Sync> FemSystem<T> {
    /// Create a new system from a matrix and RHS.
    pub fn new(matrix: TripletMatrix<T>, rhs: Vec<T>) -> Self {
        Self {
            matrix,
            rhs,
            dirichlet_dofs: HashSet::new(),
        }
    }

    /// Construct and validate a generic FEM system.
    pub fn try_new(matrix: TripletMatrix<T>, rhs: Vec<T>) -> Result<Self, FemSystemError> {
        let system = Self::new(matrix, rhs);
        system.validate()?;
        Ok(system)
    }

    /// Validate matrix dimensions and all sparse indices.
    pub fn validate(&self) -> Result<(), FemSystemError> {
        if self.matrix.rows.len() != self.matrix.cols.len()
            || self.matrix.rows.len() != self.matrix.values.len()
        {
            return Err(FemSystemError::TripletLengthMismatch {
                rows: self.matrix.rows.len(),
                cols: self.matrix.cols.len(),
                values: self.matrix.values.len(),
            });
        }
        if self.matrix.num_rows != self.rhs.len() {
            return Err(FemSystemError::DimensionMismatch {
                matrix_rows: self.matrix.num_rows,
                rhs_len: self.rhs.len(),
            });
        }
        for (triplet, (&row, &col)) in self
            .matrix
            .rows
            .iter()
            .zip(self.matrix.cols.iter())
            .enumerate()
        {
            if row >= self.matrix.num_rows || col >= self.matrix.num_cols {
                return Err(FemSystemError::MatrixIndexOutOfBounds {
                    triplet,
                    row,
                    col,
                    rows: self.matrix.num_rows,
                    cols: self.matrix.num_cols,
                });
            }
        }
        Ok(())
    }

    /// Number of DOFs.
    pub fn num_dofs(&self) -> usize {
        self.rhs.len()
    }
}

/// True residual and convergence status for a candidate FEM solution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidualReport<R> {
    /// Euclidean norm of `rhs - matrix * solution`.
    pub absolute_residual: R,
    /// Absolute residual divided by the RHS norm (or absolute residual for zero RHS).
    pub relative_residual: R,
    /// Euclidean norm of the RHS used for relative scaling.
    pub rhs_norm: R,
    /// Whether the candidate satisfies the supplied mixed tolerance gate.
    pub converged: bool,
}

impl<T: ComplexField + Default> FemSystem<T> {
    /// Compute the true residual of a candidate solution against this FEM system.
    ///
    /// This deliberately evaluates the triplet matrix directly rather than
    /// reusing an iterative solver's estimated residual. It therefore remains
    /// valid after preconditioning, row/column boundary elimination, or a
    /// solver implementation that reports a recurrence estimate.
    pub fn residual_report(
        &self,
        solution: &[T],
        absolute_tolerance: T::Real,
        relative_tolerance: T::Real,
    ) -> Result<ResidualReport<T::Real>, FemSystemError> {
        self.validate()?;
        if solution.len() != self.matrix.num_cols {
            return Err(FemSystemError::SolutionLengthMismatch {
                expected: self.matrix.num_cols,
                actual: solution.len(),
            });
        }
        if !absolute_tolerance.is_finite()
            || !relative_tolerance.is_finite()
            || absolute_tolerance < T::Real::zero()
            || relative_tolerance < T::Real::zero()
        {
            return Err(FemSystemError::InvalidTolerance);
        }

        let mut ax = vec![T::zero(); self.matrix.num_rows];
        for ((&row, &col), &value) in self
            .matrix
            .rows
            .iter()
            .zip(self.matrix.cols.iter())
            .zip(self.matrix.values.iter())
        {
            ax[row] += value * solution[col];
        }

        let residual: Vec<T> = self
            .rhs
            .iter()
            .zip(ax.iter())
            .map(|(&rhs, &computed)| rhs - computed)
            .collect();
        let absolute_residual = T::vec_norm(&residual);
        let rhs_norm = T::vec_norm(&self.rhs);
        if !absolute_residual.is_finite() || !rhs_norm.is_finite() {
            return Err(FemSystemError::NonFiniteResidual);
        }

        let relative_residual = if rhs_norm > T::Real::zero() {
            absolute_residual / rhs_norm
        } else {
            absolute_residual
        };
        if !relative_residual.is_finite() {
            return Err(FemSystemError::NonFiniteResidual);
        }

        Ok(ResidualReport {
            absolute_residual,
            relative_residual,
            rhs_norm,
            converged: absolute_residual <= absolute_tolerance + relative_tolerance * rhs_norm,
        })
    }
}

/// Apply homogeneous Dirichlet BCs (u = 0) to a `FemSystem<f64>`.
///
/// For all DOFs on the given boundary:
/// 1. Corrects RHS for column elimination: `rhs[i] -= A[i,j] * 0` (no-op for homogeneous)
/// 2. Removes all triplets in constrained rows AND constrained columns
/// 3. Adds diagonal identity entries `(dof, dof, 1.0)`
/// 4. Sets `rhs[dof] = 0`
pub fn apply_homogeneous_dirichlet<S>(
    system: &mut FemSystem<f64>,
    mesh: &Mesh,
    space: &S,
    boundary_marker: i32,
) where
    S: FunctionSpace,
{
    if let Err(error) = try_apply_homogeneous_dirichlet(system, mesh, space, boundary_marker) {
        panic!("failed to apply homogeneous Dirichlet values: {error}");
    }
}

/// Checked variant of [`apply_homogeneous_dirichlet`] with transactional validation.
///
/// The system and every boundary DOF are validated before any matrix or RHS
/// mutation. This keeps malformed function-space topology from becoming a
/// deep indexing panic in the legacy infallible wrapper.
pub fn try_apply_homogeneous_dirichlet<S>(
    system: &mut FemSystem<f64>,
    mesh: &Mesh,
    space: &S,
    boundary_marker: i32,
) -> Result<(), BoundaryConditionError>
where
    S: FunctionSpace,
{
    let dofs = space.boundary_dofs(mesh, boundary_marker);
    try_apply_homogeneous_dirichlet_dofs(system, &dofs)
}

/// Apply Dirichlet BCs with prescribed values to a `FemSystem<f64>`.
///
/// Nodal spaces use [`FunctionSpace::dof_coordinates`], including higher-order
/// and vector-valued H1 spaces. `dof_to_node` is only a compatibility fallback.
///
/// Steps:
/// 1. Correct RHS for column elimination: `rhs[i] -= A[i,j] * g_j`
/// 2. Remove constrained row AND column entries
/// 3. Add diagonal identity `(dof, dof, 1.0)`
/// 4. Set `rhs[dof] = g(x, y, z)`
pub fn apply_dirichlet_values<S>(
    system: &mut FemSystem<f64>,
    mesh: &Mesh,
    space: &S,
    boundary_marker: i32,
    value_fn: &dyn Fn(f64, f64, f64) -> f64,
    dof_to_node: Option<&dyn Fn(usize) -> usize>,
) where
    S: FunctionSpace,
{
    // Keep the historical infallible API for existing applications. New code
    // should call `try_apply_dirichlet_values` so malformed spaces and systems
    // become typed errors before any matrix/RHS mutation occurs.
    if let Err(error) =
        try_apply_dirichlet_values(system, mesh, space, boundary_marker, value_fn, dof_to_node)
    {
        panic!("failed to apply Dirichlet values: {error}");
    }
}

/// Checked variant of [`apply_dirichlet_values`] with transactional validation.
///
/// All DOFs, coordinates, and prescribed values are checked before the matrix
/// or RHS is changed. This makes malformed topology and non-finite callbacks
/// explicit failures at the FEM boundary instead of deep indexing panics.
pub fn try_apply_dirichlet_values<S>(
    system: &mut FemSystem<f64>,
    mesh: &Mesh,
    space: &S,
    boundary_marker: i32,
    value_fn: &dyn Fn(f64, f64, f64) -> f64,
    dof_to_node: Option<&dyn Fn(usize) -> usize>,
) -> Result<(), BoundaryConditionError>
where
    S: FunctionSpace,
{
    system.validate()?;
    let dofs = space.boundary_dofs(mesh, boundary_marker);
    let dof_set: HashSet<usize> = dofs.iter().copied().collect();
    let space_coordinates = space.dof_coordinates(mesh);

    let dof_coords = |dof: usize| -> Result<(f64, f64, f64), BoundaryConditionError> {
        if dof >= system.matrix.num_cols || dof >= system.rhs.len() {
            return Err(BoundaryConditionError::DofOutOfBounds {
                dof,
                num_dofs: system.rhs.len().min(system.matrix.num_cols),
            });
        }
        if let Some(coordinates) = space_coordinates.as_ref() {
            let point = coordinates
                .get(dof)
                .ok_or(BoundaryConditionError::MissingDofCoordinate { dof })?;
            return Ok((point[0], point[1], point[2]));
        }

        let node_idx = dof_to_node
            .map(|map| map(dof))
            .ok_or(BoundaryConditionError::MissingDofToNode { dof })?;
        let point = mesh
            .nodes
            .get(node_idx)
            .ok_or(BoundaryConditionError::NodeOutOfBounds {
                dof,
                node: node_idx,
            })?;
        Ok((point.x, point.y, point.z))
    };

    let prescribed: Vec<(usize, f64)> = dofs
        .iter()
        .copied()
        .map(|dof| {
            let (x, y, z) = dof_coords(dof)?;
            let value = value_fn(x, y, z);
            if !value.is_finite() {
                return Err(BoundaryConditionError::NonFiniteValue { dof });
            }
            Ok((dof, value))
        })
        .collect::<Result<_, BoundaryConditionError>>()?;
    let prescribed_by_dof: HashMap<usize, f64> = prescribed.iter().copied().collect();

    // First pass: correct RHS for column elimination
    for i in 0..system.matrix.nnz() {
        let row = system.matrix.rows[i];
        let col = system.matrix.cols[i];
        if !dof_set.contains(&row) && dof_set.contains(&col) {
            system.rhs[row] -= system.matrix.values[i] * prescribed_by_dof[&col];
        }
    }

    // Second pass: filter out constrained rows AND constrained columns
    let mut new_rows = Vec::with_capacity(system.matrix.nnz());
    let mut new_cols = Vec::with_capacity(system.matrix.nnz());
    let mut new_vals = Vec::with_capacity(system.matrix.nnz());

    for i in 0..system.matrix.nnz() {
        let row = system.matrix.rows[i];
        let col = system.matrix.cols[i];
        if !dof_set.contains(&row) && !dof_set.contains(&col) {
            new_rows.push(row);
            new_cols.push(col);
            new_vals.push(system.matrix.values[i]);
        }
    }

    // Add diagonal entries for constrained DOFs
    for &dof in &dofs {
        new_rows.push(dof);
        new_cols.push(dof);
        new_vals.push(1.0);

        system.rhs[dof] = prescribed_by_dof[&dof];
        system.dirichlet_dofs.insert(dof);
    }

    system.matrix.rows = new_rows;
    system.matrix.cols = new_cols;
    system.matrix.values = new_vals;
    Ok(())
}

/// Apply non-homogeneous complex Dirichlet values to a checked FEM system.
///
/// This is the complex counterpart of [`try_apply_dirichlet_values`]. The
/// prescribed column contribution is moved to the unconstrained RHS before
/// constrained rows and columns are removed. Coordinates and callback values
/// are validated before any system mutation.
pub fn apply_dirichlet_values_complex<S>(
    system: &mut FemSystem<Complex64>,
    mesh: &Mesh,
    space: &S,
    boundary_marker: i32,
    value_fn: &dyn Fn(f64, f64, f64) -> Complex64,
    dof_to_node: Option<&dyn Fn(usize) -> usize>,
) where
    S: FunctionSpace,
{
    if let Err(error) = try_apply_dirichlet_values_complex(
        system,
        mesh,
        space,
        boundary_marker,
        value_fn,
        dof_to_node,
    ) {
        panic!("failed apply complex Dirichlet values: {error}");
    }
}

/// Checked, transactional complex-valued Dirichlet elimination.
pub fn try_apply_dirichlet_values_complex<S>(
    system: &mut FemSystem<Complex64>,
    mesh: &Mesh,
    space: &S,
    boundary_marker: i32,
    value_fn: &dyn Fn(f64, f64, f64) -> Complex64,
    dof_to_node: Option<&dyn Fn(usize) -> usize>,
) -> Result<(), BoundaryConditionError>
where
    S: FunctionSpace,
{
    system.validate()?;
    let mut dofs = space.boundary_dofs(mesh, boundary_marker);
    dofs.sort_unstable();
    dofs.dedup();
    let dof_set: HashSet<usize> = dofs.iter().copied().collect();
    let space_coordinates = space.dof_coordinates(mesh);

    let coordinate_for_dof = |dof: usize| -> Result<(f64, f64, f64), BoundaryConditionError> {
        if dof >= system.matrix.num_cols || dof >= system.rhs.len() {
            return Err(BoundaryConditionError::DofOutOfBounds {
                dof,
                num_dofs: system.rhs.len().min(system.matrix.num_cols),
            });
        }
        if let Some(coordinates) = space_coordinates.as_ref() {
            let point = coordinates
                .get(dof)
                .ok_or(BoundaryConditionError::MissingDofCoordinate { dof })?;
            if point.iter().any(|value| !value.is_finite()) {
                return Err(BoundaryConditionError::InvalidBoundaryGeometry { boundary: 0 });
            }
            return Ok((point[0], point[1], point[2]));
        }
        let node = dof_to_node
            .map(|map| map(dof))
            .ok_or(BoundaryConditionError::MissingDofToNode { dof })?;
        let point = mesh
            .nodes
            .get(node)
            .ok_or(BoundaryConditionError::NodeOutOfBounds { dof, node })?;
        if ![point.x, point.y, point.z].into_iter().all(f64::is_finite) {
            return Err(BoundaryConditionError::InvalidBoundaryGeometry { boundary: 0 });
        }
        Ok((point.x, point.y, point.z))
    };

    let prescribed: Vec<(usize, Complex64)> = dofs
        .iter()
        .map(|&dof| {
            let (x, y, z) = coordinate_for_dof(dof)?;
            let value = value_fn(x, y, z);
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(BoundaryConditionError::NonFiniteValue { dof });
            }
            Ok((dof, value))
        })
        .collect::<Result<_, BoundaryConditionError>>()?;
    let prescribed_by_dof: HashMap<usize, Complex64> = prescribed.iter().copied().collect();

    for i in 0..system.matrix.nnz() {
        let row = system.matrix.rows[i];
        let col = system.matrix.cols[i];
        if !dof_set.contains(&row) && dof_set.contains(&col) {
            system.rhs[row] -= system.matrix.values[i] * prescribed_by_dof[&col];
        }
    }

    let mut new_rows = Vec::with_capacity(system.matrix.nnz() + dofs.len());
    let mut new_cols = Vec::with_capacity(system.matrix.nnz() + dofs.len());
    let mut new_values = Vec::with_capacity(system.matrix.nnz() + dofs.len());
    for i in 0..system.matrix.nnz() {
        let row = system.matrix.rows[i];
        let col = system.matrix.cols[i];
        if !dof_set.contains(&row) && !dof_set.contains(&col) {
            new_rows.push(row);
            new_cols.push(col);
            new_values.push(system.matrix.values[i]);
        }
    }
    for &dof in &dofs {
        new_rows.push(dof);
        new_cols.push(dof);
        new_values.push(Complex64::new(1.0, 0.0));
        system.rhs[dof] = prescribed_by_dof[&dof];
        system.dirichlet_dofs.insert(dof);
    }
    system.matrix.rows = new_rows;
    system.matrix.cols = new_cols;
    system.matrix.values = new_values;
    Ok(())
}

/// Apply homogeneous Dirichlet conditions to an explicit DOF set.
pub fn apply_homogeneous_dirichlet_dofs(system: &mut FemSystem<f64>, dofs: &[usize]) {
    if let Err(error) = try_apply_homogeneous_dirichlet_dofs(system, dofs) {
        panic!("failed to apply homogeneous Dirichlet values: {error}");
    }
}

/// Checked variant of [`apply_homogeneous_dirichlet_dofs`].
///
/// All DOFs are checked against both matrix dimensions before the system is
/// changed. Duplicate DOFs are coalesced so each constrained row receives one
/// identity entry.
pub fn try_apply_homogeneous_dirichlet_dofs(
    system: &mut FemSystem<f64>,
    dofs: &[usize],
) -> Result<(), BoundaryConditionError> {
    apply_homogeneous_dirichlet_dofs_checked(system, dofs, 1.0, 0.0)
}

/// Apply homogeneous complex Dirichlet conditions to explicit DOFs.
///
/// Duplicate DOFs are coalesced before symmetric row and column elimination.
///
/// # Errors
/// Rejects malformed systems and out-of-range DOFs before changing the system.
pub fn try_apply_homogeneous_dirichlet_complex_dofs(
    system: &mut FemSystem<Complex64>,
    dofs: &[usize],
) -> Result<(), BoundaryConditionError> {
    apply_homogeneous_dirichlet_dofs_checked(
        system,
        dofs,
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
}

fn apply_homogeneous_dirichlet_dofs_checked<T>(
    system: &mut FemSystem<T>,
    dofs: &[usize],
    identity: T,
    zero: T,
) -> Result<(), BoundaryConditionError>
where
    T: Copy + Default + std::ops::AddAssign + Send + Sync,
{
    system.validate()?;
    let mut unique_dofs = dofs.to_vec();
    unique_dofs.sort_unstable();
    unique_dofs.dedup();
    for &dof in &unique_dofs {
        if dof >= system.matrix.num_rows || dof >= system.matrix.num_cols {
            return Err(BoundaryConditionError::DofOutOfBounds {
                dof,
                num_dofs: system.rhs.len().min(system.matrix.num_cols),
            });
        }
    }

    let dof_set: HashSet<usize> = unique_dofs.iter().copied().collect();
    let mut new_rows = Vec::with_capacity(system.matrix.nnz());
    let mut new_cols = Vec::with_capacity(system.matrix.nnz());
    let mut new_vals = Vec::with_capacity(system.matrix.nnz());

    for i in 0..system.matrix.nnz() {
        let row = system.matrix.rows[i];
        let col = system.matrix.cols[i];
        if !dof_set.contains(&row) && !dof_set.contains(&col) {
            new_rows.push(row);
            new_cols.push(col);
            new_vals.push(system.matrix.values[i]);
        }
    }

    for &dof in &unique_dofs {
        new_rows.push(dof);
        new_cols.push(dof);
        new_vals.push(identity);
        system.rhs[dof] = zero;
        system.dirichlet_dofs.insert(dof);
    }

    system.matrix.rows = new_rows;
    system.matrix.cols = new_cols;
    system.matrix.values = new_vals;
    Ok(())
}

/// Shell boundary condition flavor for 6-DOF-per-node shell elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellBoundaryKind {
    /// Fix all `[u, v, w, theta_x, theta_y, theta_z]`.
    Clamped,
    /// Fix translational DOFs `[u, v, w]`, leave rotations free.
    SimplySupported,
    /// Fix an explicit local DOF mask.
    Selective([bool; 6]),
}

/// Apply 6-DOF shell boundary conditions to nodes on a mesh boundary marker.
pub fn apply_shell_dirichlet(
    system: &mut FemSystem<f64>,
    mesh: &Mesh,
    boundary_marker: i32,
    kind: ShellBoundaryKind,
) {
    let mask = match kind {
        ShellBoundaryKind::Clamped => [true; 6],
        ShellBoundaryKind::SimplySupported => [true, true, true, false, false, false],
        ShellBoundaryKind::Selective(mask) => mask,
    };
    let mut nodes = HashSet::new();
    for boundary in &mesh.boundaries {
        if boundary.marker == boundary_marker {
            for &node in &boundary.nodes {
                nodes.insert(node);
            }
        }
    }
    let mut dofs = Vec::new();
    for node in nodes {
        for (local, &fix) in mask.iter().enumerate() {
            if fix {
                dofs.push(node * 6 + local);
            }
        }
    }
    dofs.sort_unstable();
    dofs.dedup();
    apply_homogeneous_dirichlet_dofs(system, &dofs);
}

// ---------------------------------------------------------------------------
// Generic Neumann BC: adds ∫_Γ h φ_i dΓ to the RHS
// ---------------------------------------------------------------------------

/// Apply Neumann boundary conditions to a generic `FemSystem<Complex64>`.
///
/// Adds `∫_Γ h(x,y,z) φ_i dΓ` to `system.rhs` for each boundary
/// edge/face with the given `marker`. Works with any physics domain —
/// the flux function `h` is caller-defined.
///
/// **Limitation**: This function assumes DOF index == node index (H1 P1 scalar spaces).
/// For P2/P3 elements, boundary edge DOFs exist at non-node indices and this function
/// will give incorrect results. For VectorH1 spaces, each node has multiple DOFs and this
/// function is not applicable. For higher-order or vector spaces, use a
/// `FunctionSpace::boundary_dofs()` based approach instead.
/// Checked P1 scalar Neumann application with transactional validation.
pub fn try_apply_neumann_complex(
    system: &mut FemSystem<Complex64>,
    mesh: &Mesh,
    marker: i32,
    flux_fn: &dyn Fn(f64, f64, f64) -> Complex64,
) -> Result<(), BoundaryConditionError> {
    try_apply_neumann_p1(system, mesh, marker, flux_fn, |value| {
        value.re.is_finite() && value.im.is_finite()
    })
}

/// Checked real counterpart of [`apply_neumann_real`].
pub fn try_apply_neumann_real(
    system: &mut FemSystem<f64>,
    mesh: &Mesh,
    marker: i32,
    flux_fn: &dyn Fn(f64, f64, f64) -> f64,
) -> Result<(), BoundaryConditionError> {
    try_apply_neumann_p1(system, mesh, marker, flux_fn, f64::is_finite)
}

fn try_apply_neumann_p1<T>(
    system: &mut FemSystem<T>,
    mesh: &Mesh,
    marker: i32,
    flux_fn: &dyn Fn(f64, f64, f64) -> T,
    is_finite: impl Fn(T) -> bool,
) -> Result<(), BoundaryConditionError>
where
    T: Copy + Default + std::ops::AddAssign + std::ops::Mul<f64, Output = T> + Send + Sync,
{
    system.validate()?;
    let num_dofs = system.rhs.len().min(system.matrix.num_cols);
    let mut rhs_delta = vec![T::default(); system.rhs.len()];
    let edge_quadrature = gauss_legendre_1d(2);
    let face_quadrature = gauss_triangle(3);

    for (boundary_index, boundary) in mesh.boundaries.iter().enumerate() {
        if boundary.marker != marker {
            continue;
        }
        if !matches!(boundary.nodes.len(), 2 | 3) {
            return Err(BoundaryConditionError::UnsupportedBoundary {
                boundary: boundary_index,
                nodes: boundary.nodes.len(),
            });
        }
        for &node in &boundary.nodes {
            if node >= num_dofs {
                return Err(BoundaryConditionError::BoundaryNodeOutOfBounds {
                    boundary: boundary_index,
                    node,
                    num_dofs,
                });
            }
            if node >= mesh.nodes.len() {
                return Err(BoundaryConditionError::MeshNodeOutOfBounds {
                    boundary: boundary_index,
                    node,
                    num_nodes: mesh.nodes.len(),
                });
            }
        }

        let points: Vec<[f64; 3]> = boundary
            .nodes
            .iter()
            .map(|&node| {
                let point = &mesh.nodes[node];
                [point.x, point.y, point.z]
            })
            .collect();
        if points.iter().flatten().any(|value| !value.is_finite()) {
            return Err(BoundaryConditionError::InvalidBoundaryGeometry {
                boundary: boundary_index,
            });
        }

        if points.len() == 2 {
            let delta = [
                points[1][0] - points[0][0],
                points[1][1] - points[0][1],
                points[1][2] - points[0][2],
            ];
            let jacobian = delta.iter().map(|value| value * value).sum::<f64>().sqrt();
            if !jacobian.is_finite() || jacobian <= 0.0 {
                return Err(BoundaryConditionError::InvalidBoundaryGeometry {
                    boundary: boundary_index,
                });
            }
            for quadrature_point in &edge_quadrature {
                let t = 0.5 * (quadrature_point.xi() + 1.0);
                let x = points[0][0] + t * delta[0];
                let y = points[0][1] + t * delta[1];
                let z = points[0][2] + t * delta[2];
                let flux = flux_fn(x, y, z);
                if !is_finite(flux) {
                    return Err(BoundaryConditionError::NonFiniteBoundaryValue {
                        boundary: boundary_index,
                    });
                }
                let weight = 0.5 * quadrature_point.weight * jacobian;
                rhs_delta[boundary.nodes[0]] += flux * ((1.0 - t) * weight);
                rhs_delta[boundary.nodes[1]] += flux * (t * weight);
            }
        } else {
            let first = [
                points[1][0] - points[0][0],
                points[1][1] - points[0][1],
                points[1][2] - points[0][2],
            ];
            let second = [
                points[2][0] - points[0][0],
                points[2][1] - points[0][1],
                points[2][2] - points[0][2],
            ];
            let cross = [
                first[1] * second[2] - first[2] * second[1],
                first[2] * second[0] - first[0] * second[2],
                first[0] * second[1] - first[1] * second[0],
            ];
            let jacobian = cross.iter().map(|value| value * value).sum::<f64>().sqrt();
            if !jacobian.is_finite() || jacobian <= 0.0 {
                return Err(BoundaryConditionError::InvalidBoundaryGeometry {
                    boundary: boundary_index,
                });
            }
            for quadrature_point in &face_quadrature {
                let shape = [
                    1.0 - quadrature_point.xi() - quadrature_point.eta(),
                    quadrature_point.xi(),
                    quadrature_point.eta(),
                ];
                let x = shape[0] * points[0][0] + shape[1] * points[1][0] + shape[2] * points[2][0];
                let y = shape[0] * points[0][1] + shape[1] * points[1][1] + shape[2] * points[2][1];
                let z = shape[0] * points[0][2] + shape[1] * points[1][2] + shape[2] * points[2][2];
                let flux = flux_fn(x, y, z);
                if !is_finite(flux) {
                    return Err(BoundaryConditionError::NonFiniteBoundaryValue {
                        boundary: boundary_index,
                    });
                }
                let weighted_flux = flux * (quadrature_point.weight * jacobian);
                for (local, &node) in boundary.nodes.iter().enumerate() {
                    rhs_delta[node] += weighted_flux * shape[local];
                }
            }
        }
    }

    for (target, delta) in system.rhs.iter_mut().zip(rhs_delta) {
        *target += delta;
    }
    Ok(())
}

pub fn apply_neumann_complex(
    system: &mut FemSystem<Complex64>,
    mesh: &Mesh,
    marker: i32,
    flux_fn: &dyn Fn(f64, f64, f64) -> Complex64,
) {
    if let Err(error) = try_apply_neumann_complex(system, mesh, marker, flux_fn) {
        panic!("failed to apply checked Neumann values: {error}");
    }
}

#[allow(dead_code)]
fn apply_neumann_complex_unchecked(
    system: &mut FemSystem<Complex64>,
    mesh: &Mesh,
    marker: i32,
    flux_fn: &dyn Fn(f64, f64, f64) -> Complex64,
) {
    let quad_1d = gauss_legendre_1d(2);

    for boundary in &mesh.boundaries {
        if boundary.marker != marker {
            continue;
        }
        let nodes = &boundary.nodes;

        if nodes.len() == 2 {
            let p0 = &mesh.nodes[nodes[0]];
            let p1 = &mesh.nodes[nodes[1]];
            let dx = p1.x - p0.x;
            let dy = p1.y - p0.y;
            let dz = p1.z - p0.z;
            let edge_len = (dx * dx + dy * dy + dz * dz).sqrt();

            for qp in &quad_1d {
                let t = 0.5 * (qp.xi() + 1.0);
                let w = 0.5 * qp.weight * edge_len;
                let x = p0.x + t * dx;
                let y = p0.y + t * dy;
                let z = p0.z + t * dz;
                let h = flux_fn(x, y, z);
                let n0 = 1.0 - t;
                let n1 = t;
                system.rhs[nodes[0]] += h * n0 * w;
                system.rhs[nodes[1]] += h * n1 * w;
            }
        } else if nodes.len() == 3 {
            // P1 triangle face (3D): centroid quadrature
            let p0 = &mesh.nodes[nodes[0]];
            let p1 = &mesh.nodes[nodes[1]];
            let p2 = &mesh.nodes[nodes[2]];
            let v1 = [p1.x - p0.x, p1.y - p0.y, p1.z - p0.z];
            let v2 = [p2.x - p0.x, p2.y - p0.y, p2.z - p0.z];
            let cx = v1[1] * v2[2] - v1[2] * v2[1];
            let cy = v1[2] * v2[0] - v1[0] * v2[2];
            let cz = v1[0] * v2[1] - v1[1] * v2[0];
            let area = 0.5 * (cx * cx + cy * cy + cz * cz).sqrt();
            let xc = (p0.x + p1.x + p2.x) / 3.0;
            let yc = (p0.y + p1.y + p2.y) / 3.0;
            let zc = (p0.z + p1.z + p2.z) / 3.0;
            let h = flux_fn(xc, yc, zc);
            let w = area / 3.0;
            for &node in nodes {
                system.rhs[node] += h * w;
            }
        }
    }
}

/// Apply a spatially varying scalar Neumann flux to a real P1 FEM system.
///
/// Adds `∫_Γ h(x,y,z) φ_i dΓ` on marked boundary edges or triangular faces.
/// The flux is evaluated at quadrature points, so localized or angularly
/// varying heat/mass fluxes do not need to be collapsed to one constant per
/// boundary. The scalar P1 contract requires DOF index equal to node index.
pub fn apply_neumann_real(
    system: &mut FemSystem<f64>,
    mesh: &Mesh,
    marker: i32,
    flux_fn: &dyn Fn(f64, f64, f64) -> f64,
) {
    if let Err(error) = try_apply_neumann_real(system, mesh, marker, flux_fn) {
        panic!("failed to apply checked Neumann values: {error}");
    }
}

#[allow(dead_code)]
fn apply_neumann_real_unchecked(
    system: &mut FemSystem<f64>,
    mesh: &Mesh,
    marker: i32,
    flux_fn: &dyn Fn(f64, f64, f64) -> f64,
) {
    let edge_quadrature = gauss_legendre_1d(3);
    let face_quadrature = gauss_triangle(3);

    for boundary in &mesh.boundaries {
        if boundary.marker != marker {
            continue;
        }
        let nodes = &boundary.nodes;
        if nodes.len() == 2 {
            let p0 = &mesh.nodes[nodes[0]];
            let p1 = &mesh.nodes[nodes[1]];
            let delta = [p1.x - p0.x, p1.y - p0.y, p1.z - p0.z];
            let edge_length =
                (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
            for point in &edge_quadrature {
                let coordinate = 0.5 * (point.xi() + 1.0);
                let weight = 0.5 * point.weight * edge_length;
                let x = p0.x + coordinate * delta[0];
                let y = p0.y + coordinate * delta[1];
                let z = p0.z + coordinate * delta[2];
                let flux = flux_fn(x, y, z);
                system.rhs[nodes[0]] += flux * (1.0 - coordinate) * weight;
                system.rhs[nodes[1]] += flux * coordinate * weight;
            }
        } else if nodes.len() == 3 {
            let p0 = &mesh.nodes[nodes[0]];
            let p1 = &mesh.nodes[nodes[1]];
            let p2 = &mesh.nodes[nodes[2]];
            let first = [p1.x - p0.x, p1.y - p0.y, p1.z - p0.z];
            let second = [p2.x - p0.x, p2.y - p0.y, p2.z - p0.z];
            let cross = [
                first[1] * second[2] - first[2] * second[1],
                first[2] * second[0] - first[0] * second[2],
                first[0] * second[1] - first[1] * second[0],
            ];
            let surface_jacobian =
                (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
            for point in &face_quadrature {
                let shape = [1.0 - point.xi() - point.eta(), point.xi(), point.eta()];
                let x = shape[0] * p0.x + shape[1] * p1.x + shape[2] * p2.x;
                let y = shape[0] * p0.y + shape[1] * p1.y + shape[2] * p2.y;
                let z = shape[0] * p0.z + shape[1] * p1.z + shape[2] * p2.z;
                let weighted_flux = flux_fn(x, y, z) * point.weight * surface_jacobian;
                for (node, value) in nodes.iter().zip(shape) {
                    system.rhs[*node] += weighted_flux * value;
                }
            }
        }
    }
}

/// Apply vector traction (Neumann) boundary conditions to a real vector FEM system.
///
/// Adds `∫_Γ t(x) · φ_i dΓ` to `system.rhs` for each boundary edge/face with
/// the given marker. This is the natural boundary condition for linear
/// elasticity, `σ · n = t`.
///
/// The current implementation targets `VectorH1Space` with interleaved DOF
/// ordering `[node0_x, node0_y, node1_x, ...]`. P1 and P2 traces are supported
/// on boundary edges in 2D and triangular boundary faces in 3D.
pub fn try_apply_traction_vector(
    system: &mut FemSystem<f64>,
    mesh: &Mesh,
    space: &VectorH1Space,
    boundary_marker: i32,
    traction_fn: &dyn Fn(f64, f64, f64) -> Vec<f64>,
) -> Result<(), BoundaryConditionError> {
    system.validate()?;
    let components = space.num_components();
    let expected_dofs = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        space.num_global_dofs(mesh)
    })) {
        Ok(expected_dofs) => expected_dofs,
        Err(_) => {
            let boundary = mesh
                .boundaries
                .iter()
                .position(|candidate| candidate.marker == boundary_marker)
                .unwrap_or(0);
            return Err(BoundaryConditionError::BoundaryTopologyMismatch { boundary });
        }
    };
    if system.rhs.len() != expected_dofs || system.matrix.num_cols != expected_dofs {
        return Err(BoundaryConditionError::InvalidSystem(
            FemSystemError::DimensionMismatch {
                matrix_rows: system.matrix.num_cols,
                rhs_len: expected_dofs,
            },
        ));
    }
    let degree = space.degree();
    if !matches!(degree, PolynomialDegree::P1 | PolynomialDegree::P2)
        || components == 0
        || space.field_type() != FieldType::Vector
    {
        return Err(BoundaryConditionError::UnsupportedVectorSpace {
            components,
            degree: degree.degree(),
        });
    }

    // The existing VectorH1 P2 numbering is the range-based implicit-geometry
    // numbering. Explicit P2 H1 geometry is scalar-space-only until vector
    // trace ownership is promoted through FunctionSpace.
    if degree == PolynomialDegree::P2
        && mesh
            .elements
            .iter()
            .any(|element| element.nodes.len() == element.element_type.num_nodes_p2())
    {
        return Err(BoundaryConditionError::UnsupportedVectorSpace {
            components,
            degree: degree.degree(),
        });
    }

    if degree == PolynomialDegree::P2 {
        let boundary_edge_is_owned = |a: usize, b: usize| {
            mesh.elements.iter().any(|element| {
                element
                    .element_type
                    .local_edges()
                    .iter()
                    .any(|&(local_a, local_b)| {
                        let (Some(&u), Some(&v)) =
                            (element.nodes.get(local_a), element.nodes.get(local_b))
                        else {
                            return false;
                        };
                        (u == a && v == b) || (u == b && v == a)
                    })
            })
        };
        for (boundary_index, boundary) in mesh.boundaries.iter().enumerate() {
            if !matches!(boundary.nodes.len(), 2 | 3) {
                continue;
            }
            if let Some(&node) = boundary
                .nodes
                .iter()
                .find(|&&node| node >= mesh.nodes.len())
            {
                return Err(BoundaryConditionError::MeshNodeOutOfBounds {
                    boundary: boundary_index,
                    node,
                    num_nodes: mesh.nodes.len(),
                });
            }
            let Some(owner) = mesh.elements.get(boundary.element_idx) else {
                return Err(BoundaryConditionError::BoundaryTopologyMismatch {
                    boundary: boundary_index,
                });
            };
            let entity_matches = if mesh.dimension == 2 {
                owner
                    .element_type
                    .local_edges()
                    .get(boundary.local_idx)
                    .map(|&(local_a, local_b)| {
                        [local_a, local_b].into_iter().all(|local_node| {
                            owner.nodes.get(local_node).is_some()
                                && boundary.nodes.contains(&owner.nodes[local_node])
                        })
                    })
                    .unwrap_or(false)
            } else {
                owner
                    .element_type
                    .local_faces()
                    .get(boundary.local_idx)
                    .map(|face| {
                        face.iter().all(|&local_node| {
                            owner.nodes.get(local_node).is_some()
                                && boundary.nodes.contains(&owner.nodes[local_node])
                        })
                    })
                    .unwrap_or(false)
            };
            if !entity_matches {
                return Err(BoundaryConditionError::BoundaryTopologyMismatch {
                    boundary: boundary_index,
                });
            }
            let edges = if boundary.nodes.len() == 2 {
                vec![(boundary.nodes[0], boundary.nodes[1])]
            } else {
                vec![
                    (boundary.nodes[0], boundary.nodes[1]),
                    (boundary.nodes[1], boundary.nodes[2]),
                    (boundary.nodes[2], boundary.nodes[0]),
                ]
            };
            if edges
                .into_iter()
                .any(|(a, b)| !boundary_edge_is_owned(a, b))
            {
                return Err(BoundaryConditionError::BoundaryTopologyMismatch {
                    boundary: boundary_index,
                });
            }
        }
    }
    let topology = if degree == PolynomialDegree::P2 {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::mesh::topology::MeshTopology::build(mesh)
        })) {
            Ok(topology) => Some(topology),
            Err(_) => {
                let boundary = mesh
                    .boundaries
                    .iter()
                    .position(|candidate| candidate.marker == boundary_marker)
                    .unwrap_or(0);
                return Err(BoundaryConditionError::BoundaryTopologyMismatch { boundary });
            }
        }
    } else {
        None
    };
    let edge_quadrature = gauss_legendre_1d(if degree == PolynomialDegree::P1 { 2 } else { 3 });
    let face_quadrature = gauss_triangle(if degree == PolynomialDegree::P1 { 1 } else { 3 });
    let mut rhs_delta = vec![0.0; expected_dofs];

    let mut add_trace = |scalar_dofs: &[usize], shapes: &[f64], weight: f64, traction: &[f64]| {
        for (local, &scalar_dof) in scalar_dofs.iter().enumerate() {
            for (component, &value) in traction.iter().take(components).enumerate() {
                let dof = scalar_dof
                    .checked_mul(components)
                    .and_then(|base| base.checked_add(component))
                    .ok_or(BoundaryConditionError::DofOutOfBounds {
                        dof: usize::MAX,
                        num_dofs: expected_dofs,
                    })?;
                if dof >= expected_dofs {
                    return Err(BoundaryConditionError::DofOutOfBounds {
                        dof,
                        num_dofs: expected_dofs,
                    });
                }
                rhs_delta[dof] += value * shapes[local] * weight;
            }
        }
        Ok::<(), BoundaryConditionError>(())
    };

    for (boundary_index, boundary) in mesh.boundaries.iter().enumerate() {
        if boundary.marker != boundary_marker {
            continue;
        }
        if !matches!(boundary.nodes.len(), 2 | 3) {
            return Err(BoundaryConditionError::UnsupportedBoundary {
                boundary: boundary_index,
                nodes: boundary.nodes.len(),
            });
        }
        for &node in &boundary.nodes {
            if node >= expected_dofs / components {
                return Err(BoundaryConditionError::BoundaryNodeOutOfBounds {
                    boundary: boundary_index,
                    node,
                    num_dofs: expected_dofs,
                });
            }
            if node >= mesh.nodes.len() {
                return Err(BoundaryConditionError::MeshNodeOutOfBounds {
                    boundary: boundary_index,
                    node,
                    num_nodes: mesh.nodes.len(),
                });
            }
        }

        let points: Vec<[f64; 3]> = boundary
            .nodes
            .iter()
            .map(|&node| {
                let point = &mesh.nodes[node];
                [point.x, point.y, point.z]
            })
            .collect();
        if points.iter().flatten().any(|value| !value.is_finite()) {
            return Err(BoundaryConditionError::InvalidBoundaryGeometry {
                boundary: boundary_index,
            });
        }

        if points.len() == 2 {
            let delta = [
                points[1][0] - points[0][0],
                points[1][1] - points[0][1],
                points[1][2] - points[0][2],
            ];
            let jacobian = delta.iter().map(|value| value * value).sum::<f64>().sqrt();
            if !jacobian.is_finite() || jacobian <= 0.0 {
                return Err(BoundaryConditionError::InvalidBoundaryGeometry {
                    boundary: boundary_index,
                });
            }
            let mut scalar_dofs = vec![boundary.nodes[0], boundary.nodes[1]];
            let edge = if boundary.nodes[0] < boundary.nodes[1] {
                (boundary.nodes[0], boundary.nodes[1])
            } else {
                (boundary.nodes[1], boundary.nodes[0])
            };
            let edge_index = topology
                .as_ref()
                .and_then(|topology| topology.edge_map.get(&edge).copied());
            if degree == PolynomialDegree::P2 {
                scalar_dofs.push(
                    mesh.num_nodes()
                        + edge_index.ok_or(BoundaryConditionError::BoundaryTopologyMismatch {
                            boundary: boundary_index,
                        })?,
                );
            }
            for quadrature_point in &edge_quadrature {
                let t = 0.5 * (quadrature_point.xi() + 1.0);
                let x = points[0][0] + t * delta[0];
                let y = points[0][1] + t * delta[1];
                let z = points[0][2] + t * delta[2];
                let traction = traction_fn(x, y, z);
                if traction.len() < components {
                    return Err(BoundaryConditionError::BoundaryValueDimension {
                        boundary: boundary_index,
                        expected: components,
                        actual: traction.len(),
                    });
                }
                if traction[..components]
                    .iter()
                    .any(|value| !value.is_finite())
                {
                    return Err(BoundaryConditionError::NonFiniteBoundaryValue {
                        boundary: boundary_index,
                    });
                }
                let shapes = if degree == PolynomialDegree::P1 {
                    vec![1.0 - t, t]
                } else {
                    vec![
                        (1.0 - t) * (1.0 - 2.0 * t),
                        t * (2.0 * t - 1.0),
                        4.0 * t * (1.0 - t),
                    ]
                };
                add_trace(
                    &scalar_dofs,
                    &shapes,
                    0.5 * quadrature_point.weight * jacobian,
                    &traction,
                )?;
            }
        } else {
            let first = [
                points[1][0] - points[0][0],
                points[1][1] - points[0][1],
                points[1][2] - points[0][2],
            ];
            let second = [
                points[2][0] - points[0][0],
                points[2][1] - points[0][1],
                points[2][2] - points[0][2],
            ];
            let cross = [
                first[1] * second[2] - first[2] * second[1],
                first[2] * second[0] - first[0] * second[2],
                first[0] * second[1] - first[1] * second[0],
            ];
            let jacobian = cross.iter().map(|value| value * value).sum::<f64>().sqrt();
            if !jacobian.is_finite() || jacobian <= 0.0 {
                return Err(BoundaryConditionError::InvalidBoundaryGeometry {
                    boundary: boundary_index,
                });
            }
            let mut scalar_dofs = boundary.nodes.clone();
            if degree == PolynomialDegree::P2 {
                for &(a, b) in &[
                    (boundary.nodes[0], boundary.nodes[1]),
                    (boundary.nodes[1], boundary.nodes[2]),
                    (boundary.nodes[2], boundary.nodes[0]),
                ] {
                    let edge = if a < b { (a, b) } else { (b, a) };
                    let edge_index = topology
                        .as_ref()
                        .and_then(|topology| topology.edge_map.get(&edge).copied())
                        .ok_or(BoundaryConditionError::BoundaryTopologyMismatch {
                            boundary: boundary_index,
                        })?;
                    scalar_dofs.push(mesh.num_nodes() + edge_index);
                }
            }
            for quadrature_point in &face_quadrature {
                let l1 = quadrature_point.xi();
                let l2 = quadrature_point.eta();
                let l0 = 1.0 - l1 - l2;
                let x = l0 * points[0][0] + l1 * points[1][0] + l2 * points[2][0];
                let y = l0 * points[0][1] + l1 * points[1][1] + l2 * points[2][1];
                let z = l0 * points[0][2] + l1 * points[1][2] + l2 * points[2][2];
                let traction = traction_fn(x, y, z);
                if traction.len() < components {
                    return Err(BoundaryConditionError::BoundaryValueDimension {
                        boundary: boundary_index,
                        expected: components,
                        actual: traction.len(),
                    });
                }
                if traction[..components]
                    .iter()
                    .any(|value| !value.is_finite())
                {
                    return Err(BoundaryConditionError::NonFiniteBoundaryValue {
                        boundary: boundary_index,
                    });
                }
                let shapes = if degree == PolynomialDegree::P1 {
                    vec![l0, l1, l2]
                } else {
                    vec![
                        l0 * (2.0 * l0 - 1.0),
                        l1 * (2.0 * l1 - 1.0),
                        l2 * (2.0 * l2 - 1.0),
                        4.0 * l0 * l1,
                        4.0 * l1 * l2,
                        4.0 * l2 * l0,
                    ]
                };
                add_trace(
                    &scalar_dofs,
                    &shapes,
                    quadrature_point.weight * jacobian,
                    &traction,
                )?;
            }
        }
    }

    for (target, delta) in system.rhs.iter_mut().zip(rhs_delta) {
        *target += delta;
    }
    Ok(())
}

/// Apply vector traction through the checked, transactional path.
pub fn apply_traction_vector(
    system: &mut FemSystem<f64>,
    mesh: &Mesh,
    space: &VectorH1Space,
    boundary_marker: i32,
    traction_fn: &dyn Fn(f64, f64, f64) -> Vec<f64>,
) {
    if let Err(error) = try_apply_traction_vector(system, mesh, space, boundary_marker, traction_fn)
    {
        panic!("failed to apply checked vector traction: {error}");
    }
}

#[allow(dead_code)]
fn apply_traction_vector_legacy_unchecked(
    system: &mut FemSystem<f64>,
    mesh: &Mesh,
    space: &VectorH1Space,
    boundary_marker: i32,
    traction_fn: &dyn Fn(f64, f64, f64) -> Vec<f64>,
) {
    assert_eq!(
        space.field_type(),
        FieldType::Vector,
        "traction BCs require a vector H1 space"
    );
    let components = space.num_components();
    assert_eq!(
        system.rhs.len(),
        space.num_global_dofs(mesh),
        "system RHS length must match vector space DOF count"
    );

    let degree = space.degree();
    assert!(
        matches!(degree, PolynomialDegree::P1 | PolynomialDegree::P2),
        "traction BCs support P1 and P2 vector H1 traces"
    );
    let quad_1d = gauss_legendre_1d(if degree == PolynomialDegree::P1 { 2 } else { 3 });
    let topology =
        (degree == PolynomialDegree::P2).then(|| crate::mesh::topology::MeshTopology::build(mesh));

    for boundary in &mesh.boundaries {
        if boundary.marker != boundary_marker {
            continue;
        }

        let nodes = &boundary.nodes;
        if nodes.len() == 2 {
            let p0 = &mesh.nodes[nodes[0]];
            let p1 = &mesh.nodes[nodes[1]];
            let dx = p1.x - p0.x;
            let dy = p1.y - p0.y;
            let dz = p1.z - p0.z;
            let edge_len = (dx * dx + dy * dy + dz * dz).sqrt();

            for qp in &quad_1d {
                let t = 0.5 * (qp.xi() + 1.0);
                let w = 0.5 * qp.weight * edge_len;
                let x = p0.x + t * dx;
                let y = p0.y + t * dy;
                let z = p0.z + t * dz;
                let traction = traction_fn(x, y, z);
                assert!(
                    traction.len() >= components,
                    "traction vector must have at least {components} components"
                );

                let mut scalar_dofs = vec![nodes[0], nodes[1]];
                let mut shapes = vec![1.0 - t, t];
                if let Some(topology) = topology.as_ref() {
                    let edge = if nodes[0] < nodes[1] {
                        (nodes[0], nodes[1])
                    } else {
                        (nodes[1], nodes[0])
                    };
                    let edge_index = topology.edge_map[&edge];
                    scalar_dofs.push(mesh.num_nodes() + edge_index);
                    shapes = vec![
                        (1.0 - t) * (1.0 - 2.0 * t),
                        t * (2.0 * t - 1.0),
                        4.0 * t * (1.0 - t),
                    ];
                }
                for (a, &scalar_dof) in scalar_dofs.iter().enumerate() {
                    for (c, &tc) in traction.iter().enumerate().take(components) {
                        let dof = scalar_dof * components + c;
                        system.rhs[dof] += tc * shapes[a] * w;
                    }
                }
            }
        } else if nodes.len() == 3 {
            let p0 = &mesh.nodes[nodes[0]];
            let p1 = &mesh.nodes[nodes[1]];
            let p2 = &mesh.nodes[nodes[2]];
            let v1 = [p1.x - p0.x, p1.y - p0.y, p1.z - p0.z];
            let v2 = [p2.x - p0.x, p2.y - p0.y, p2.z - p0.z];
            let cx = v1[1] * v2[2] - v1[2] * v2[1];
            let cy = v1[2] * v2[0] - v1[0] * v2[2];
            let cz = v1[0] * v2[1] - v1[1] * v2[0];
            let surface_jacobian = (cx * cx + cy * cy + cz * cz).sqrt();
            let face_quadrature =
                gauss_triangle(if degree == PolynomialDegree::P1 { 1 } else { 3 });
            for qp in &face_quadrature {
                let l1 = qp.xi();
                let l2 = qp.eta();
                let l0 = 1.0 - l1 - l2;
                let x = l0 * p0.x + l1 * p1.x + l2 * p2.x;
                let y = l0 * p0.y + l1 * p1.y + l2 * p2.y;
                let z = l0 * p0.z + l1 * p1.z + l2 * p2.z;
                let traction = traction_fn(x, y, z);
                assert!(
                    traction.len() >= components,
                    "traction vector must have at least {components} components"
                );

                let mut scalar_dofs = vec![nodes[0], nodes[1], nodes[2]];
                let mut shapes = vec![l0, l1, l2];
                if let Some(topology) = topology.as_ref() {
                    for (a, b) in [
                        (nodes[0], nodes[1]),
                        (nodes[1], nodes[2]),
                        (nodes[2], nodes[0]),
                    ] {
                        let edge = if a < b { (a, b) } else { (b, a) };
                        scalar_dofs.push(mesh.num_nodes() + topology.edge_map[&edge]);
                    }
                    shapes = vec![
                        l0 * (2.0 * l0 - 1.0),
                        l1 * (2.0 * l1 - 1.0),
                        l2 * (2.0 * l2 - 1.0),
                        4.0 * l0 * l1,
                        4.0 * l1 * l2,
                        4.0 * l2 * l0,
                    ];
                }
                let w = qp.weight * surface_jacobian;
                for (a, &scalar_dof) in scalar_dofs.iter().enumerate() {
                    for (c, &tc) in traction.iter().enumerate().take(components) {
                        let dof = scalar_dof * components + c;
                        system.rhs[dof] += tc * shapes[a] * w;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Generic Robin BC: adds α ∫_Γ φ_i φ_j dΓ to matrix + ∫_Γ g φ_i dΓ to RHS
// ---------------------------------------------------------------------------

/// Apply Robin boundary conditions to a generic `FemSystem<Complex64>`.
///
/// The Robin BC is: `∂u/∂n + α u = g`.
///
/// Adds `α ∫_Γ φ_i φ_j dΓ` to the system matrix and `∫_Γ g φ_i dΓ`
/// to the RHS for each boundary edge/face with the given `marker`.
///
/// **Limitation**: Same as `apply_neumann_complex` — this function assumes
/// DOF index == node index (H1 P1 scalar spaces). For higher-order or vector
/// spaces, a different approach is needed.
///
/// For uniform α and g = 0 (absorbing/impedance BCs), prefer the
/// PrebakedAssembler boundary coefficient approach which is more efficient.
/// Checked P1 scalar Robin application with transactional validation.
pub fn try_apply_robin_complex(
    system: &mut FemSystem<Complex64>,
    mesh: &Mesh,
    marker: i32,
    alpha_fn: &dyn Fn(f64, f64, f64) -> Complex64,
    g_fn: &dyn Fn(f64, f64, f64) -> Complex64,
) -> Result<(), BoundaryConditionError> {
    system.validate()?;
    let num_dofs = system.rhs.len().min(system.matrix.num_cols);
    let mut rhs_delta = vec![Complex64::new(0.0, 0.0); system.rhs.len()];
    let mut matrix_terms = Vec::new();
    let edge_quadrature = gauss_legendre_1d(2);

    for (boundary_index, boundary) in mesh.boundaries.iter().enumerate() {
        if boundary.marker != marker {
            continue;
        }
        if !matches!(boundary.nodes.len(), 2 | 3) {
            return Err(BoundaryConditionError::UnsupportedBoundary {
                boundary: boundary_index,
                nodes: boundary.nodes.len(),
            });
        }
        for &node in &boundary.nodes {
            if node >= num_dofs {
                return Err(BoundaryConditionError::BoundaryNodeOutOfBounds {
                    boundary: boundary_index,
                    node,
                    num_dofs,
                });
            }
            if node >= mesh.nodes.len() {
                return Err(BoundaryConditionError::MeshNodeOutOfBounds {
                    boundary: boundary_index,
                    node,
                    num_nodes: mesh.nodes.len(),
                });
            }
        }
        let points: Vec<[f64; 3]> = boundary
            .nodes
            .iter()
            .map(|&node| {
                let point = &mesh.nodes[node];
                [point.x, point.y, point.z]
            })
            .collect();
        if points.iter().flatten().any(|value| !value.is_finite()) {
            return Err(BoundaryConditionError::InvalidBoundaryGeometry {
                boundary: boundary_index,
            });
        }

        if points.len() == 2 {
            let delta = [
                points[1][0] - points[0][0],
                points[1][1] - points[0][1],
                points[1][2] - points[0][2],
            ];
            let jacobian = delta.iter().map(|value| value * value).sum::<f64>().sqrt();
            if !jacobian.is_finite() || jacobian <= 0.0 {
                return Err(BoundaryConditionError::InvalidBoundaryGeometry {
                    boundary: boundary_index,
                });
            }
            for quadrature_point in &edge_quadrature {
                let t = 0.5 * (quadrature_point.xi() + 1.0);
                let x = points[0][0] + t * delta[0];
                let y = points[0][1] + t * delta[1];
                let z = points[0][2] + t * delta[2];
                let alpha = alpha_fn(x, y, z);
                let g = g_fn(x, y, z);
                if [alpha.re, alpha.im, g.re, g.im]
                    .iter()
                    .any(|value| !value.is_finite())
                {
                    return Err(BoundaryConditionError::NonFiniteBoundaryValue {
                        boundary: boundary_index,
                    });
                }
                let weight = 0.5 * quadrature_point.weight * jacobian;
                let shapes = [1.0 - t, t];
                for (local_i, &shape_i) in shapes.iter().enumerate() {
                    rhs_delta[boundary.nodes[local_i]] += g * (shape_i * weight);
                    for (local_j, &shape_j) in shapes.iter().enumerate() {
                        matrix_terms.push((
                            boundary.nodes[local_i],
                            boundary.nodes[local_j],
                            alpha * (shape_i * shape_j * weight),
                        ));
                    }
                }
            }
        } else {
            let first = [
                points[1][0] - points[0][0],
                points[1][1] - points[0][1],
                points[1][2] - points[0][2],
            ];
            let second = [
                points[2][0] - points[0][0],
                points[2][1] - points[0][1],
                points[2][2] - points[0][2],
            ];
            let cross = [
                first[1] * second[2] - first[2] * second[1],
                first[2] * second[0] - first[0] * second[2],
                first[0] * second[1] - first[1] * second[0],
            ];
            let area = 0.5 * cross.iter().map(|value| value * value).sum::<f64>().sqrt();
            if !area.is_finite() || area <= 0.0 {
                return Err(BoundaryConditionError::InvalidBoundaryGeometry {
                    boundary: boundary_index,
                });
            }
            let centroid = [
                (points[0][0] + points[1][0] + points[2][0]) / 3.0,
                (points[0][1] + points[1][1] + points[2][1]) / 3.0,
                (points[0][2] + points[1][2] + points[2][2]) / 3.0,
            ];
            let alpha = alpha_fn(centroid[0], centroid[1], centroid[2]);
            let g = g_fn(centroid[0], centroid[1], centroid[2]);
            if [alpha.re, alpha.im, g.re, g.im]
                .iter()
                .any(|value| !value.is_finite())
            {
                return Err(BoundaryConditionError::NonFiniteBoundaryValue {
                    boundary: boundary_index,
                });
            }
            let factor = area / 12.0;
            for (local_i, &row) in boundary.nodes.iter().enumerate() {
                rhs_delta[row] += g * (area / 3.0);
                for (local_j, &column) in boundary.nodes.iter().enumerate() {
                    let mass = if local_i == local_j { 2.0 } else { 1.0 };
                    matrix_terms.push((row, column, alpha * (factor * mass)));
                }
            }
        }
    }

    for (target, delta) in system.rhs.iter_mut().zip(rhs_delta) {
        *target += delta;
    }
    for (row, column, value) in matrix_terms {
        system.matrix.push(row, column, value);
    }
    Ok(())
}

pub fn apply_robin_complex(
    system: &mut FemSystem<Complex64>,
    mesh: &Mesh,
    marker: i32,
    alpha_fn: &dyn Fn(f64, f64, f64) -> Complex64,
    g_fn: &dyn Fn(f64, f64, f64) -> Complex64,
) {
    if let Err(error) = try_apply_robin_complex(system, mesh, marker, alpha_fn, g_fn) {
        panic!("failed to apply checked Robin values: {error}");
    }
}

#[allow(dead_code)]
fn apply_robin_complex_unchecked(
    system: &mut FemSystem<Complex64>,
    mesh: &Mesh,
    marker: i32,
    alpha_fn: &dyn Fn(f64, f64, f64) -> Complex64,
    g_fn: &dyn Fn(f64, f64, f64) -> Complex64,
) {
    let quad_1d = gauss_legendre_1d(2);

    for boundary in &mesh.boundaries {
        if boundary.marker != marker {
            continue;
        }
        let nodes = &boundary.nodes;

        if nodes.len() == 2 {
            let p0 = &mesh.nodes[nodes[0]];
            let p1 = &mesh.nodes[nodes[1]];
            let dx = p1.x - p0.x;
            let dy = p1.y - p0.y;
            let dz = p1.z - p0.z;
            let edge_len = (dx * dx + dy * dy + dz * dz).sqrt();

            for qp in &quad_1d {
                let t = 0.5 * (qp.xi() + 1.0);
                let w = 0.5 * qp.weight * edge_len;
                let x = p0.x + t * dx;
                let y = p0.y + t * dy;
                let z = p0.z + t * dz;
                let alpha = alpha_fn(x, y, z);
                let g = g_fn(x, y, z);
                let shapes = [1.0 - t, t];

                // Matrix contribution: α φ_i φ_j
                for (i, &ni) in shapes.iter().enumerate() {
                    for (j, &nj) in shapes.iter().enumerate() {
                        system.matrix.push(nodes[i], nodes[j], alpha * ni * nj * w);
                    }
                    // RHS contribution: g φ_i
                    system.rhs[nodes[i]] += g * ni * w;
                }
            }
        } else if nodes.len() == 3 {
            // P1 triangle face: centroid quadrature (1-point)
            let p0 = &mesh.nodes[nodes[0]];
            let p1 = &mesh.nodes[nodes[1]];
            let p2 = &mesh.nodes[nodes[2]];
            let v1 = [p1.x - p0.x, p1.y - p0.y, p1.z - p0.z];
            let v2 = [p2.x - p0.x, p2.y - p0.y, p2.z - p0.z];
            let cx = v1[1] * v2[2] - v1[2] * v2[1];
            let cy = v1[2] * v2[0] - v1[0] * v2[2];
            let cz = v1[0] * v2[1] - v1[1] * v2[0];
            let area = 0.5 * (cx * cx + cy * cy + cz * cz).sqrt();
            let xc = (p0.x + p1.x + p2.x) / 3.0;
            let yc = (p0.y + p1.y + p2.y) / 3.0;
            let zc = (p0.z + p1.z + p2.z) / 3.0;
            let alpha = alpha_fn(xc, yc, zc);
            let g = g_fn(xc, yc, zc);

            // P1 triangle mass: (Area/12) [2 1 1; 1 2 1; 1 1 2]
            let factor = area / 12.0;
            for i in 0..3 {
                for j in 0..3 {
                    let m = if i == j { 2.0 * factor } else { factor };
                    system.matrix.push(nodes[i], nodes[j], alpha * m);
                }
                system.rhs[nodes[i]] += g * area / 3.0;
            }
        }
    }
}

/// Apply homogeneous Dirichlet BCs to a `FemSystem<Complex64>`.
///
/// Same logic as [`apply_homogeneous_dirichlet`] but for complex-valued systems.
pub fn apply_homogeneous_dirichlet_complex<S>(
    system: &mut FemSystem<Complex64>,
    mesh: &Mesh,
    space: &S,
    boundary_marker: i32,
) where
    S: FunctionSpace,
{
    if let Err(error) =
        try_apply_homogeneous_dirichlet_complex(system, mesh, space, boundary_marker)
    {
        panic!("failed to apply homogeneous Dirichlet values: {error}");
    }
}

/// Checked variant of [`apply_homogeneous_dirichlet_complex`].
pub fn try_apply_homogeneous_dirichlet_complex<S>(
    system: &mut FemSystem<Complex64>,
    mesh: &Mesh,
    space: &S,
    boundary_marker: i32,
) -> Result<(), BoundaryConditionError>
where
    S: FunctionSpace,
{
    let dofs = space.boundary_dofs(mesh, boundary_marker);
    apply_homogeneous_dirichlet_dofs_checked(
        system,
        &dofs,
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembly::TripletMatrix;
    use crate::basis::PolynomialDegree;
    use crate::mesh::{
        Mesh, unit_cube_hexahedra, unit_cube_tetrahedra, unit_square_quads, unit_square_triangles,
    };
    use crate::spaces::H1Space;

    /// Build a simple 3-DOF system: 1D Laplacian on [0, 1] with 3 nodes.
    /// K = [[ 1, -1,  0],
    ///      [-1,  2, -1],
    ///      [ 0, -1,  1]]
    /// rhs = [0, 1, 0]
    fn simple_3dof_system() -> FemSystem<f64> {
        let mut matrix = TripletMatrix::new(3, 3);
        matrix.push(0, 0, 1.0);
        matrix.push(0, 1, -1.0);
        matrix.push(1, 0, -1.0);
        matrix.push(1, 1, 2.0);
        matrix.push(1, 2, -1.0);
        matrix.push(2, 1, -1.0);
        matrix.push(2, 2, 1.0);
        let rhs = vec![0.0, 1.0, 0.0];
        FemSystem::new(matrix, rhs)
    }

    #[test]
    fn apply_dirichlet_values_preserves_linear_field_for_all_h1_elements() {
        let cases: [(&str, fn(usize) -> Mesh); 4] = [
            ("triangle", unit_square_triangles),
            ("quadrilateral", unit_square_quads),
            ("tetrahedron", unit_cube_tetrahedra),
            ("hexahedron", unit_cube_hexahedra),
        ];

        for (element, make_mesh) in cases {
            for degree in [PolynomialDegree::P1, PolynomialDegree::P2] {
                let mut mesh = make_mesh(2);
                // Use marker zero consistently across generators; some generators
                // assign side-specific markers during construction.
                mesh.detect_boundaries();
                let space = H1Space::new(degree);
                let coordinates = space
                    .dof_coordinates(&mesh)
                    .expect("H1 spaces must expose coordinates for every nodal DOF");
                let boundary_dofs = space.boundary_dofs(&mesh, 0);
                let n = space.num_global_dofs(&mesh);
                assert_eq!(coordinates.len(), n, "{element} {degree:?}");
                assert!(!boundary_dofs.is_empty(), "{element} {degree:?}");
                assert!(boundary_dofs.len() < n, "{element} {degree:?}");
                if degree == PolynomialDegree::P2 {
                    assert!(
                        boundary_dofs.iter().any(|&dof| dof >= mesh.num_nodes()),
                        "{element} P2 fixture must constrain a higher-order DOF"
                    );
                }

                let exact: Vec<f64> = coordinates
                    .iter()
                    .map(|point| 1.0 + point[0] + 2.0 * point[1] + 3.0 * point[2])
                    .collect();
                let mut matrix = TripletMatrix::new(n, n);
                let mut rhs = vec![0.0; n];
                for row in 0..n {
                    matrix.push(row, row, 2.0);
                    rhs[row] += 2.0 * exact[row];
                    if row > 0 {
                        matrix.push(row, row - 1, -0.25);
                        rhs[row] -= 0.25 * exact[row - 1];
                    }
                    if row + 1 < n {
                        matrix.push(row, row + 1, -0.25);
                        rhs[row] -= 0.25 * exact[row + 1];
                    }
                }

                let mut system = FemSystem::new(matrix, rhs);
                apply_dirichlet_values(
                    &mut system,
                    &mesh,
                    &space,
                    0,
                    &|x, y, z| 1.0 + x + 2.0 * y + 3.0 * z,
                    None,
                );

                let mut applied = vec![0.0; n];
                for entry in 0..system.matrix.nnz() {
                    applied[system.matrix.rows[entry]] +=
                        system.matrix.values[entry] * exact[system.matrix.cols[entry]];
                }
                for row in 0..n {
                    assert!(
                        (applied[row] - system.rhs[row]).abs() < 1e-12,
                        "{element} {degree:?} row {row}: transformed system no longer preserves the analytical linear field"
                    );
                }
                for dof in boundary_dofs {
                    assert!(system.dirichlet_dofs.contains(&dof));
                    assert!((system.rhs[dof] - exact[dof]).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn test_apply_homogeneous_dirichlet_produces_nonsingular_system() {
        let mut system = simple_3dof_system();

        // Manually create a boundary DOF set (DOFs 0 and 2)
        let constrained = HashSet::from([0usize, 2]);
        let dof_list: Vec<usize> = constrained.iter().copied().collect();

        // Apply manually (since we don't have a mesh/space here)
        {
            let mut new_rows = Vec::new();
            let mut new_cols = Vec::new();
            let mut new_vals = Vec::new();

            for i in 0..system.matrix.nnz() {
                let row = system.matrix.rows[i];
                let col = system.matrix.cols[i];
                if !constrained.contains(&row) && !constrained.contains(&col) {
                    new_rows.push(row);
                    new_cols.push(col);
                    new_vals.push(system.matrix.values[i]);
                }
            }

            for &dof in &dof_list {
                new_rows.push(dof);
                new_cols.push(dof);
                new_vals.push(1.0);
                system.rhs[dof] = 0.0;
            }

            system.matrix.rows = new_rows;
            system.matrix.cols = new_cols;
            system.matrix.values = new_vals;
        }

        let csr = system.matrix.to_csr();

        // Row 0: should be [1, 0, 0] (identity for Dirichlet)
        let row0: Vec<(usize, f64)> = csr.row_entries(0).collect();
        assert_eq!(row0.len(), 1);
        assert_eq!(row0[0], (0, 1.0));

        // Row 1: should be [0, 2, 0] (interior, columns 0 and 2 zeroed)
        let row1: Vec<(usize, f64)> = csr.row_entries(1).collect();
        assert_eq!(row1.len(), 1);
        assert_eq!(row1[0], (1, 2.0));

        // Row 2: should be [0, 0, 1] (identity for Dirichlet)
        let row2: Vec<(usize, f64)> = csr.row_entries(2).collect();
        assert_eq!(row2.len(), 1);
        assert_eq!(row2[0], (2, 1.0));
    }

    #[test]
    fn test_apply_dirichlet_values_preserves_interior_solution() {
        let mut system = simple_3dof_system();

        // Prescribe u(0) = 3.0 at DOF 0, u(1) = 5.0 at DOF 2
        let constrained = HashSet::from([0usize, 2]);
        let prescribed: std::collections::HashMap<usize, f64> =
            [(0, 3.0), (2, 5.0)].into_iter().collect();

        // RHS correction for column elimination
        for i in 0..system.matrix.nnz() {
            let row = system.matrix.rows[i];
            let col = system.matrix.cols[i];
            if !constrained.contains(&row) && constrained.contains(&col) {
                system.rhs[row] -= system.matrix.values[i] * prescribed[&col];
            }
        }

        // Filter out constrained rows and columns
        let mut new_rows = Vec::new();
        let mut new_cols = Vec::new();
        let mut new_vals = Vec::new();

        for i in 0..system.matrix.nnz() {
            let row = system.matrix.rows[i];
            let col = system.matrix.cols[i];
            if !constrained.contains(&row) && !constrained.contains(&col) {
                new_rows.push(row);
                new_cols.push(col);
                new_vals.push(system.matrix.values[i]);
            }
        }

        for &dof in &[0, 2] {
            new_rows.push(dof);
            new_cols.push(dof);
            new_vals.push(1.0);
            system.rhs[dof] = prescribed[&dof];
        }

        system.matrix.rows = new_rows;
        system.matrix.cols = new_cols;
        system.matrix.values = new_vals;

        let csr = system.matrix.to_csr();

        // Row 1 (interior): K[1,0]*3 + K[1,2]*5 was subtracted from rhs[1]
        // Original rhs[1] = 1.0, K[1,0] = -1, K[1,2] = -1
        // Corrected rhs[1] = 1.0 - (-1)*3 - (-1)*5 = 1.0 + 3 + 5 = 9.0
        assert!(
            (system.rhs[1] - 9.0).abs() < 1e-14,
            "rhs[1] = {}",
            system.rhs[1]
        );

        // Row 1 should only have diagonal entry (2.0)
        let row1: Vec<(usize, f64)> = csr.row_entries(1).collect();
        assert_eq!(row1.len(), 1);
        assert_eq!(row1[0], (1, 2.0));

        // Solution: x[1] = 9.0 / 2.0 = 4.5
        // Which is correct: with u(0)=3, u(1)=4.5, u(2)=5, the interior equation
        // -3 + 2*4.5 - 5 = -3 + 9 - 5 = 1.0 ✓
    }

    #[test]
    fn test_apply_traction_vector_bottom_edge_resultant() {
        let mut mesh = unit_square_triangles(1);
        mesh.detect_boundaries();
        mesh.set_boundary_condition(crate::mesh::BoundaryType::Neumann, 3, |points| {
            points.iter().all(|point| point.y.abs() < 1e-12)
        });
        let space = VectorH1Space::new(PolynomialDegree::P1, 2);
        let n = space.num_global_dofs(&mesh);
        let matrix = TripletMatrix::new(n, n);
        let mut system = FemSystem::new(matrix, vec![0.0; n]);

        apply_traction_vector(&mut system, &mesh, &space, 3, &|_, _, _| vec![2.0, 3.0]);

        let total_x: f64 = system.rhs.iter().step_by(2).sum();
        let total_y: f64 = system.rhs.iter().skip(1).step_by(2).sum();
        assert!((total_x - 2.0).abs() < 1e-12, "x resultant = {total_x}");
        assert!((total_y - 3.0).abs() < 1e-12, "y resultant = {total_y}");
    }

    #[test]
    fn p2_traction_trace_preserves_edge_and_face_resultants() {
        let cases: [(Mesh, usize, [f64; 3]); 2] = [
            (unit_square_triangles(1), 2, [2.0, 3.0, 0.0]),
            (unit_cube_tetrahedra(1), 3, [2.0, 3.0, 5.0]),
        ];

        for (mut mesh, components, traction) in cases {
            mesh.detect_boundaries();
            let space = VectorH1Space::new(PolynomialDegree::P2, components);
            let n = space.num_global_dofs(&mesh);
            let mut system = FemSystem::new(TripletMatrix::new(n, n), vec![0.0; n]);
            apply_traction_vector(&mut system, &mesh, &space, 0, &|_, _, _| {
                traction[..components].to_vec()
            });

            let boundary_measure = if components == 2 { 4.0 } else { 6.0 };
            for (component, &value) in traction.iter().enumerate().take(components) {
                let resultant: f64 = system.rhs.iter().skip(component).step_by(components).sum();
                assert!(
                    (resultant - value * boundary_measure).abs() < 1e-12,
                    "P2 {components}D component {component} resultant {resultant}"
                );
            }
            assert!(
                system.rhs[mesh.num_nodes() * components..]
                    .iter()
                    .any(|value| value.abs() > 1e-12),
                "P2 traction must load edge-midpoint DOFs"
            );
        }
    }

    #[test]
    fn test_apply_neumann_constant_flux_resultant() {
        let mut mesh = unit_square_triangles(1);
        mesh.detect_boundaries();
        let n = mesh.num_nodes();
        let matrix = TripletMatrix::new(n, n);
        let mut system = FemSystem::new(matrix, vec![Complex64::new(0.0, 0.0); n]);

        apply_neumann_complex(&mut system, &mesh, 0, &|_, _, _| Complex64::new(1.0, 0.0));

        let total: Complex64 = system.rhs.iter().sum();
        // Perimeter of unit square = 4.0; constant flux 1.0 → total = 4.0
        assert!(
            (total.re - 4.0).abs() < 1e-12,
            "real resultant = {}",
            total.re
        );
        assert!(total.im.abs() < 1e-12, "imag resultant = {}", total.im);
    }

    #[test]
    fn spatial_real_neumann_flux_integrates_edges_and_faces() {
        let mut square = unit_square_triangles(2);
        square.detect_boundaries();
        let mut square_system =
            FemSystem::new(TripletMatrix::new(9, 9), vec![0.0; square.num_nodes()]);
        apply_neumann_real(&mut square_system, &square, 0, &|x, _, _| x);
        assert!((square_system.rhs.iter().sum::<f64>() - 2.0).abs() < 1.0e-13);

        let mut cube = unit_cube_tetrahedra(2);
        cube.detect_boundaries();
        let mut cube_system = FemSystem::new(
            TripletMatrix::new(cube.num_nodes(), cube.num_nodes()),
            vec![0.0; cube.num_nodes()],
        );
        apply_neumann_real(&mut cube_system, &cube, 0, &|x, y, z| x + y + z);
        assert!((cube_system.rhs.iter().sum::<f64>() - 9.0).abs() < 1.0e-12);
    }

    #[test]
    fn test_apply_robin_constant_alpha_zero_g() {
        let mut mesh = unit_square_triangles(1);
        mesh.detect_boundaries();
        let n = mesh.num_nodes();
        let matrix = TripletMatrix::new(n, n);
        let mut system = FemSystem::new(matrix, vec![Complex64::new(0.0, 0.0); n]);

        let alpha = Complex64::new(1.0, 0.0);
        apply_robin_complex(&mut system, &mesh, 0, &|_, _, _| alpha, &|_, _, _| {
            Complex64::new(0.0, 0.0)
        });

        let csr = system.matrix.to_csr();
        // Each corner node touches 2 edges → 2 diagonal contributions of 1/3 each
        for i in 0..n {
            let diag: Complex64 = csr.get(i, i);
            assert!(
                (diag.re - 2.0 / 3.0).abs() < 1e-12,
                "diag[{i}] = {}",
                diag.re
            );
        }
        // RHS should be zero since g = 0
        for i in 0..n {
            assert!(system.rhs[i].norm() < 1e-12, "rhs[{i}] = {}", system.rhs[i]);
        }
    }

    #[test]
    fn test_apply_robin_zero_alpha_constant_g() {
        let mut mesh = unit_square_triangles(1);
        mesh.detect_boundaries();
        let n = mesh.num_nodes();
        let matrix = TripletMatrix::new(n, n);
        let mut system = FemSystem::new(matrix, vec![Complex64::new(0.0, 0.0); n]);

        let g = Complex64::new(2.0, 1.0);
        apply_robin_complex(
            &mut system,
            &mesh,
            0,
            &|_, _, _| Complex64::new(0.0, 0.0),
            &|_, _, _| g,
        );

        // Matrix entries should all be zero since alpha = 0
        let csr = system.matrix.to_csr();
        assert!(
            csr.values.iter().all(|v| v.norm() < 1e-12),
            "matrix values should be zero"
        );

        // RHS should equal Neumann with flux g: each node gets g * 1.0 (2 edges * 0.5 each)
        for i in 0..n {
            assert!(
                (system.rhs[i].re - 2.0).abs() < 1e-12,
                "rhs[{i}].re = {}",
                system.rhs[i].re
            );
            assert!(
                (system.rhs[i].im - 1.0).abs() < 1e-12,
                "rhs[{i}].im = {}",
                system.rhs[i].im
            );
        }
    }

    #[test]
    fn fem_system_checked_constructor_rejects_bad_dimensions() {
        let matrix = TripletMatrix::new(2, 2);
        let error = match FemSystem::try_new(matrix, vec![0.0]) {
            Ok(_) => panic!("dimension mismatch must be rejected"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            FemSystemError::DimensionMismatch {
                matrix_rows: 2,
                rhs_len: 1
            }
        ));
    }

    #[test]
    fn checked_dirichlet_application_rejects_bad_dofs_without_mutating() {
        let mut mesh = unit_square_triangles(1);
        mesh.detect_boundaries();
        let marker = mesh
            .boundaries
            .first()
            .expect("unit square has a boundary")
            .marker;
        let space = H1Space::new(PolynomialDegree::P1);
        let mut system = FemSystem::new(TripletMatrix::new(1, 1), vec![7.0]);
        let original_rhs = system.rhs.clone();
        let error =
            try_apply_dirichlet_values(&mut system, &mesh, &space, marker, &|_, _, _| 1.0, None)
                .expect_err("boundary DOFs must be checked against the target system");
        assert!(matches!(
            error,
            BoundaryConditionError::DofOutOfBounds { .. }
        ));
        assert_eq!(system.rhs, original_rhs);
        assert!(system.matrix.rows.is_empty());
    }

    #[test]
    fn checked_dirichlet_application_rejects_nonfinite_values() {
        let mut mesh = unit_square_triangles(1);
        mesh.detect_boundaries();
        let marker = mesh
            .boundaries
            .first()
            .expect("unit square has a boundary")
            .marker;
        let space = H1Space::new(PolynomialDegree::P1);
        let n = space.num_global_dofs(&mesh);
        let mut system = FemSystem::new(TripletMatrix::new(n, n), vec![0.0; n]);
        let original_rhs = system.rhs.clone();
        let error = try_apply_dirichlet_values(
            &mut system,
            &mesh,
            &space,
            marker,
            &|_, _, _| f64::NAN,
            None,
        )
        .expect_err("non-finite prescribed values must be rejected");
        assert!(matches!(
            error,
            BoundaryConditionError::NonFiniteValue { .. }
        ));
        assert_eq!(system.rhs, original_rhs);
        assert!(system.dirichlet_dofs.is_empty());
    }

    #[test]
    fn checked_homogeneous_dirichlet_rejects_bad_dofs_without_mutating() {
        let mut system = simple_3dof_system();
        let original_matrix = (
            system.matrix.rows.clone(),
            system.matrix.cols.clone(),
            system.matrix.values.clone(),
        );
        let original_rhs = system.rhs.clone();
        let error = try_apply_homogeneous_dirichlet_dofs(&mut system, &[0, 3])
            .expect_err("out-of-range homogeneous DOFs must be rejected");

        assert_eq!(
            error,
            BoundaryConditionError::DofOutOfBounds {
                dof: 3,
                num_dofs: 3,
            }
        );
        assert_eq!(
            (system.matrix.rows, system.matrix.cols, system.matrix.values),
            original_matrix
        );
        assert_eq!(system.rhs, original_rhs);
        assert!(system.dirichlet_dofs.is_empty());
    }

    #[test]
    fn checked_homogeneous_dirichlet_coalesces_duplicate_dofs() {
        let mut system = simple_3dof_system();
        try_apply_homogeneous_dirichlet_dofs(&mut system, &[2, 0, 2])
            .expect("valid homogeneous DOFs should be applied");

        let csr = system.matrix.to_csr();
        assert_eq!(csr.row_entries(0).collect::<Vec<_>>(), vec![(0, 1.0)]);
        assert_eq!(csr.row_entries(1).collect::<Vec<_>>(), vec![(1, 2.0)]);
        assert_eq!(csr.row_entries(2).collect::<Vec<_>>(), vec![(2, 1.0)]);
        assert_eq!(system.rhs, vec![0.0, 1.0, 0.0]);
        assert_eq!(system.dirichlet_dofs, HashSet::from([0, 2]));
    }

    #[test]
    fn checked_neumann_preserves_resultant_and_rejects_nonfinite_transactionally() {
        let mut mesh = unit_square_triangles(1);
        mesh.detect_boundaries();
        let n = mesh.num_nodes();
        let mut system = FemSystem::new(TripletMatrix::new(n, n), vec![0.0; n]);

        try_apply_neumann_real(&mut system, &mesh, 0, &|_, _, _| 1.0)
            .expect("constant checked Neumann flux should apply");
        assert!((system.rhs.iter().sum::<f64>() - 4.0).abs() < 1.0e-12);

        let original_rhs = system.rhs.clone();
        let original_nnz = system.matrix.nnz();
        let error = try_apply_neumann_real(&mut system, &mesh, 0, &|_, _, _| f64::NAN)
            .expect_err("non-finite natural-boundary callbacks must be rejected");
        assert!(matches!(
            error,
            BoundaryConditionError::NonFiniteBoundaryValue { .. }
        ));
        assert_eq!(system.rhs, original_rhs);
        assert_eq!(system.matrix.nnz(), original_nnz);
    }

    #[test]
    fn checked_robin_rejects_bad_nodes_and_degenerate_geometry_without_mutating() {
        let mut mesh = unit_square_triangles(1);
        mesh.detect_boundaries();
        let n = mesh.num_nodes();
        let mut system =
            FemSystem::new(TripletMatrix::new(n, n), vec![Complex64::new(0.0, 0.0); n]);
        let original_rhs = system.rhs.clone();
        let original_nnz = system.matrix.nnz();
        let error = try_apply_robin_complex(
            &mut system,
            &mesh,
            0,
            &|_, _, _| Complex64::new(f64::NAN, 0.0),
            &|_, _, _| Complex64::new(0.0, 0.0),
        )
        .expect_err("non-finite Robin coefficients must be rejected");
        assert!(matches!(
            error,
            BoundaryConditionError::NonFiniteBoundaryValue { .. }
        ));
        assert_eq!(system.rhs, original_rhs);
        assert_eq!(system.matrix.nnz(), original_nnz);

        let mut degenerate = mesh.clone();
        let first = degenerate.boundaries[0].nodes[0];
        let second = degenerate.boundaries[0].nodes[1];
        degenerate.nodes[second].x = degenerate.nodes[first].x;
        degenerate.nodes[second].y = degenerate.nodes[first].y;
        degenerate.nodes[second].z = degenerate.nodes[first].z;
        let error = try_apply_neumann_real(
            &mut FemSystem::new(TripletMatrix::new(n, n), vec![0.0; n]),
            &degenerate,
            0,
            &|_, _, _| 1.0,
        )
        .expect_err("degenerate natural-boundary geometry must be rejected");
        assert!(matches!(
            error,
            BoundaryConditionError::InvalidBoundaryGeometry { .. }
        ));

        let mut malformed = mesh.clone();
        malformed.boundaries[0].nodes[1] = malformed.nodes.len();
        let mut mesh_index_system = FemSystem::new(
            TripletMatrix::new(n + 1, n + 1),
            vec![Complex64::new(0.0, 0.0); n + 1],
        );
        let error = try_apply_robin_complex(
            &mut mesh_index_system,
            &malformed,
            0,
            &|_, _, _| Complex64::new(1.0, 0.0),
            &|_, _, _| Complex64::new(0.0, 0.0),
        )
        .expect_err("mesh-missing natural-boundary nodes must be rejected");
        assert!(matches!(
            error,
            BoundaryConditionError::MeshNodeOutOfBounds { .. }
        ));

        let error = try_apply_robin_complex(
            &mut system,
            &malformed,
            0,
            &|_, _, _| Complex64::new(1.0, 0.0),
            &|_, _, _| Complex64::new(0.0, 0.0),
        )
        .expect_err("out-of-range natural-boundary nodes must be rejected");
        assert!(matches!(
            error,
            BoundaryConditionError::BoundaryNodeOutOfBounds { .. }
        ));
        assert_eq!(system.rhs, original_rhs);
        assert_eq!(system.matrix.nnz(), original_nnz);
    }

    #[test]
    fn legacy_natural_boundary_wrappers_route_invalid_inputs_through_checked_paths() {
        let mut mesh = unit_square_triangles(1);
        mesh.detect_boundaries();
        let mut real_system = FemSystem::new(TripletMatrix::new(1, 1), vec![0.0]);
        let real_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            apply_neumann_real(&mut real_system, &mesh, 0, &|_, _, _| 1.0);
        }));
        assert!(real_result.is_err());
        assert_eq!(real_system.rhs, vec![0.0]);
        assert_eq!(real_system.matrix.nnz(), 0);

        let mut complex_system = FemSystem::new(
            TripletMatrix::new(mesh.num_nodes(), mesh.num_nodes()),
            vec![Complex64::new(0.0, 0.0); mesh.num_nodes()],
        );
        let complex_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            apply_robin_complex(
                &mut complex_system,
                &mesh,
                0,
                &|_, _, _| Complex64::new(f64::NAN, 0.0),
                &|_, _, _| Complex64::new(0.0, 0.0),
            );
        }));
        assert!(complex_result.is_err());
        assert!(complex_system.rhs.iter().all(|value| value.norm() == 0.0));
        assert_eq!(complex_system.matrix.nnz(), 0);
    }

    #[test]
    fn residual_report_uses_true_matrix_residual() {
        let mut system = simple_3dof_system();
        system.rhs = vec![-0.5, 1.0, -0.5];
        let report = system
            .residual_report(&[1.0, 1.5, 1.0], 1e-12, 1e-12)
            .expect("valid solution should produce a residual report");

        assert!(report.absolute_residual < 1e-12);
        assert!(report.relative_residual < 1e-12);
        assert!((report.rhs_norm - 1.5_f64.sqrt()).abs() < 1e-12);
        assert!(report.converged);
    }

    #[test]
    fn residual_report_rejects_non_converged_and_malformed_candidates() {
        let system = simple_3dof_system();
        let report = system
            .residual_report(&[0.0, 0.0, 0.0], 1e-12, 1e-12)
            .expect("zero candidate still has a well-defined residual");
        assert!((report.absolute_residual - 1.0).abs() < 1e-12);
        assert!((report.relative_residual - 1.0).abs() < 1e-12);
        assert!(!report.converged);

        assert!(matches!(
            system.residual_report(&[0.0, 0.0], 1e-12, 1e-12),
            Err(FemSystemError::SolutionLengthMismatch {
                expected: 3,
                actual: 2
            })
        ));
        assert!(matches!(
            system.residual_report(&[1.0, 1.5, 1.0], -1.0, 1e-12),
            Err(FemSystemError::InvalidTolerance)
        ));
    }

    #[test]
    fn checked_vector_traction_preserves_p1_edge_resultants() {
        let mut mesh = unit_square_triangles(1);
        mesh.detect_boundaries();
        for boundary in &mut mesh.boundaries {
            boundary.marker = 3;
        }
        let space = VectorH1Space::new(PolynomialDegree::P1, 2);
        let n = space.num_global_dofs(&mesh);
        let mut system = FemSystem::new(TripletMatrix::new(n, n), vec![0.0; n]);

        try_apply_traction_vector(&mut system, &mesh, &space, 3, &|_, _, _| vec![2.0, 3.0])
            .expect("constant P1 vector traction should apply");

        let total_x: f64 = system.rhs.iter().step_by(2).sum();
        let total_y: f64 = system.rhs.iter().skip(1).step_by(2).sum();
        assert!((total_x - 8.0).abs() < 1.0e-12);
        assert!((total_y - 12.0).abs() < 1.0e-12);
    }

    #[test]
    fn checked_vector_traction_preserves_p2_face_resultants_and_edge_dofs() {
        let mut mesh = unit_cube_tetrahedra(1);
        mesh.detect_boundaries();
        for boundary in &mut mesh.boundaries {
            boundary.marker = 7;
        }
        let space = VectorH1Space::new(PolynomialDegree::P2, 3);
        let n = space.num_global_dofs(&mesh);
        let mut system = FemSystem::new(TripletMatrix::new(n, n), vec![0.0; n]);

        try_apply_traction_vector(&mut system, &mesh, &space, 7, &|_, _, _| {
            vec![2.0, 3.0, 5.0]
        })
        .expect("constant P2 vector traction should apply");

        let expected_measure = 6.0;
        for (component, value) in [2.0, 3.0, 5.0].into_iter().enumerate() {
            let resultant: f64 = system.rhs.iter().skip(component).step_by(3).sum();
            assert!((resultant - value * expected_measure).abs() < 1.0e-11);
        }
        assert!(
            system.rhs[mesh.num_nodes() * 3..]
                .iter()
                .any(|value| value.abs() > 1.0e-12)
        );
    }

    #[test]
    fn checked_vector_traction_is_transactional_on_bad_callbacks_and_geometry() {
        let mut mesh = unit_square_triangles(1);
        mesh.detect_boundaries();
        for boundary in &mut mesh.boundaries {
            boundary.marker = 3;
        }
        let space = VectorH1Space::new(PolynomialDegree::P1, 2);
        let n = space.num_global_dofs(&mesh);
        let mut system = FemSystem::new(TripletMatrix::new(n, n), vec![0.0; n]);
        let original = system.rhs.clone();

        let error = try_apply_traction_vector(&mut system, &mesh, &space, 3, &|_, _, _| vec![1.0])
            .expect_err("short vector callback must be rejected");
        assert!(matches!(
            error,
            BoundaryConditionError::BoundaryValueDimension {
                expected: 2,
                actual: 1,
                ..
            }
        ));
        assert_eq!(system.rhs, original);

        let error = try_apply_traction_vector(&mut system, &mesh, &space, 3, &|_, _, _| {
            vec![f64::NAN, 0.0]
        })
        .expect_err("non-finite vector callback must be rejected");
        assert!(matches!(
            error,
            BoundaryConditionError::NonFiniteBoundaryValue { .. }
        ));
        assert_eq!(system.rhs, original);

        mesh.nodes[0].x = f64::NAN;
        let error =
            try_apply_traction_vector(&mut system, &mesh, &space, 3, &|_, _, _| vec![1.0, 1.0])
                .expect_err("non-finite boundary geometry must be rejected");
        assert!(matches!(
            error,
            BoundaryConditionError::InvalidBoundaryGeometry { .. }
        ));
        assert_eq!(system.rhs, original);
    }

    #[test]
    fn checked_vector_traction_rejects_missing_p2_edge_topology_without_mutation() {
        let mut mesh = unit_square_triangles(1);
        mesh.detect_boundaries();
        let space = VectorH1Space::new(PolynomialDegree::P2, 2);
        let n = space.num_global_dofs(&mesh);
        let last_node = mesh.num_nodes() - 1;
        let boundary = mesh
            .boundaries
            .iter_mut()
            .find(|boundary| boundary.nodes.len() == 2)
            .expect("unit square should have an edge boundary");
        boundary.marker = 9;
        boundary.nodes = vec![0, last_node];
        let mut system = FemSystem::new(TripletMatrix::new(n, n), vec![0.0; n]);
        let original = system.rhs.clone();

        let error =
            try_apply_traction_vector(&mut system, &mesh, &space, 9, &|_, _, _| vec![1.0, 1.0])
                .expect_err("unowned P2 boundary edge must be rejected");
        assert!(matches!(
            error,
            BoundaryConditionError::BoundaryTopologyMismatch { .. }
        ));
        assert_eq!(system.rhs, original);
    }
}
