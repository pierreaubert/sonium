//! Traditional BEM (O(N²) dense matrix) assembly
//!
//! Uses the Burton-Miller formulation to avoid spurious resonances.

use num_complex::Complex64;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::traits::{AdjointCapability, LinearOperator};

use crate::core::assembly::common::{count_dofs, get_bc_type_and_value, get_element_coords};
use crate::core::assembly::hmatrix::{
    AcaVariant, HMatrix, HMatrixOperatorError, LowRankPrecision,
    build_hmatrix_with_variant_precision_and_beta,
};
use crate::core::integration::regular::regular_integration_linear_fixed_order;
use crate::core::integration::{
    QuadratureParams, regular_integration, regular_integration_checked, regular_integration_linear,
    regular_integration_linear_with_normals, regular_integration_wavenumber_derivative,
    singular_integration, singular_integration_checked, singular_integration_linear,
    singular_integration_linear_curved, singular_integration_linear_with_params,
    singular_integration_wavenumber_derivative,
};
use crate::core::parallel::{
    BEM_QUADRATURE_PAIRS_PER_THREAD, gated_enumerate_filter_map, gated_map, use_parallel,
};
use crate::core::types::{
    BasisOrder, BoundaryCondition, Element, PhysicsParams, QuadratureConvergenceError,
};

/// Adjacent-order quadrature policy used by the checked TBEM assembly path.
///
/// The historical builders remain available for compatibility, but they are
/// infallible and therefore cannot report an under-resolved panel.  Callers
/// that need an explicit production accuracy gate can pass this policy to
/// [`build_tbem_system_checked`].  Both regular and self-element pairs are
/// checked before the existing parallel assembly is started.
#[derive(Debug, Clone, Copy)]
pub struct TbemQuadraturePolicy {
    /// Coarse regular-panel Gauss order.
    pub regular_coarse_order: usize,
    /// Fine regular-panel Gauss order.
    pub regular_fine_order: usize,
    /// Coarse self-element quadrature parameters.
    pub singular_coarse: QuadratureParams,
    /// Fine self-element quadrature parameters.
    pub singular_fine: QuadratureParams,
    /// Absolute error budget applied to every kernel channel.
    pub absolute_tolerance: f64,
    /// Relative error budget applied to every kernel channel.
    pub relative_tolerance: f64,
}

impl Default for TbemQuadraturePolicy {
    fn default() -> Self {
        Self {
            regular_coarse_order: 3,
            regular_fine_order: 4,
            singular_coarse: QuadratureParams::default(),
            singular_fine: QuadratureParams::high_accuracy(),
            absolute_tolerance: 1.0e-6,
            relative_tolerance: 1.0e-4,
        }
    }
}

/// Pair location attached to a checked quadrature failure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TbemQuadraturePairError {
    /// Source/collocation element index.
    pub source_element: usize,
    /// Field/integration element index.
    pub field_element: usize,
    /// Underlying convergence failure.
    pub error: QuadratureConvergenceError,
}

impl std::fmt::Display for TbemQuadraturePairError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TBEM quadrature pair ({}, {}) failed: {}",
            self.source_element, self.field_element, self.error
        )
    }
}

impl std::error::Error for TbemQuadraturePairError {}

/// Size mismatch between two [`TbemSystem`]s passed to
/// [`TbemSystem::try_merge`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MergeSizeError {
    /// `num_dofs` of the merge target.
    pub left: usize,
    /// `num_dofs` of the merged-in system.
    pub right: usize,
}

impl std::fmt::Display for MergeSizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TbemSystem size mismatch: {} DOFs vs {} DOFs",
            self.left, self.right
        )
    }
}

impl std::error::Error for MergeSizeError {}

/// A TBEM row with no owning source element.
///
/// `count_dofs` sizes the system from the maximum non-evaluation DOF index,
/// so a gapped DOF numbering leaves rows no element writes to. Assembling
/// those rows as silent zeros yields a singular system with no diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MissingSourceDof {
    /// Row (== DOF index) with no source element.
    pub dof: usize,
}

impl std::fmt::Display for MissingSourceDof {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TBEM assembly: DOF {} has no source element (row would be a silent zero)",
            self.dof
        )
    }
}

impl std::error::Error for MissingSourceDof {}

/// Result of TBEM assembly
pub struct TbemSystem {
    /// Coefficient matrix (dense, row-major)
    pub matrix: DenseMatrix<Complex64>,
    /// Right-hand side vector
    pub rhs: Vec<Complex64>,
    /// Number of DOFs
    pub num_dofs: usize,
}

impl TbemSystem {
    /// Create a new system with given number of DOFs
    pub fn new(num_dofs: usize) -> Self {
        Self {
            matrix: DenseMatrix::zeros(num_dofs, num_dofs),
            rhs: vec![Complex64::new(0.0, 0.0); num_dofs],
            num_dofs,
        }
    }

    /// Merge another system into self by element-wise addition.
    ///
    /// # Panics
    ///
    /// Panics if the two systems have different sizes. Prefer
    /// [`try_merge`](Self::try_merge), which returns a
    /// [`MergeSizeError`] instead, in all library and application code.
    pub fn merge(&mut self, other: Self) {
        self.try_merge(other)
            .expect("TbemSystem::merge requires equal num_dofs on both systems");
    }

    /// Merge another system into self by element-wise addition, checked.
    ///
    /// Returns [`MergeSizeError`] instead of panicking when the two systems
    /// have different sizes.
    pub fn try_merge(&mut self, other: Self) -> Result<(), MergeSizeError> {
        if self.num_dofs != other.num_dofs {
            return Err(MergeSizeError {
                left: self.num_dofs,
                right: other.num_dofs,
            });
        }
        for (a, b) in self.rhs.iter_mut().zip(other.rhs.iter()) {
            *a += *b;
        }
        for (a, b) in self
            .matrix
            .as_mut_slice()
            .iter_mut()
            .zip(other.matrix.as_slice().iter())
        {
            *a += *b;
        }
        Ok(())
    }
}

/// Errors returned by checked TBEM derivative contractions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TbemDerivativeError {
    /// A supplied vector does not match the derivative operator dimension.
    DimensionMismatch {
        /// Operand name used in the diagnostic.
        operand: &'static str,
        /// Expected number of entries.
        expected: usize,
        /// Supplied number of entries.
        actual: usize,
    },
    /// A supplied coefficient or derivative entry is not finite.
    NonFinite {
        /// Operand name used in the diagnostic.
        operand: &'static str,
        /// Flat entry index.
        index: usize,
    },
}

impl std::fmt::Display for TbemDerivativeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch {
                operand,
                expected,
                actual,
            } => write!(
                f,
                "TBEM derivative {operand} has {actual} entries, expected {expected}"
            ),
            Self::NonFinite { operand, index } => {
                write!(f, "TBEM derivative {operand} entry {index} is not finite")
            }
        }
    }
}

impl std::error::Error for TbemDerivativeError {}

/// Contract an adjoint, a fixed-geometry wavenumber derivative, and a forward
/// solution into a real scalar objective gradient.
///
/// If `A(k)x(k)=b(k)` and `A(k)ᴴλ=g`, this returns
/// `Re(λᴴ(db/dk − dA/dk·x))`.  The derivative system is produced by
/// [`build_tbem_system_stabilized_wavenumber_derivative`].
pub fn tbem_wavenumber_objective_gradient(
    derivative_system: &TbemSystem,
    forward_coefficients: &[Complex64],
    adjoint_coefficients: &[Complex64],
) -> Result<f64, TbemDerivativeError> {
    let expected = derivative_system.num_dofs;
    for (operand, actual) in [
        ("forward coefficients", forward_coefficients.len()),
        ("adjoint coefficients", adjoint_coefficients.len()),
        ("derivative RHS", derivative_system.rhs.len()),
    ] {
        if actual != expected {
            return Err(TbemDerivativeError::DimensionMismatch {
                operand,
                expected,
                actual,
            });
        }
    }

    let is_finite = |value: Complex64| value.re.is_finite() && value.im.is_finite();
    for (operand, values) in [
        ("forward coefficients", forward_coefficients),
        ("adjoint coefficients", adjoint_coefficients),
        ("derivative RHS", derivative_system.rhs.as_slice()),
        ("derivative matrix", derivative_system.matrix.as_slice()),
    ] {
        if let Some(index) = values.iter().position(|&value| !is_finite(value)) {
            return Err(TbemDerivativeError::NonFinite { operand, index });
        }
    }

    let mut gradient = Complex64::new(0.0, 0.0);
    for (row, &adj) in adjoint_coefficients.iter().enumerate() {
        let mut derivative_action = derivative_system.rhs[row];
        for (col, &fwd) in forward_coefficients.iter().enumerate() {
            derivative_action -= derivative_system.matrix[[row, col]] * fwd;
        }
        gradient += adj.conj() * derivative_action;
    }
    Ok(gradient.re)
}

/// Dense stabilized exterior boundary operator with normal velocity kept as
/// an explicit input block.
///
/// It represents the all-Neumann constant-panel problem as
/// `A p - Q v = 0`, where `p` is the panel-pressure vector and `v` is one
/// outward normal velocity per non-evaluation element.  Keeping `Q` as an
/// apply operation avoids baking one prescribed velocity pattern into the
/// system right-hand side and is the dense reference building block used by
/// monolithic FEM–BEM assemblies.
#[derive(Debug, Clone)]
pub struct StabilizedTbemVelocityOperator {
    pressure_matrix: DenseMatrix<Complex64>,
    elements: Vec<Element>,
    nodes: Vec<[f64; 3]>,
    physics: PhysicsParams,
    sign: f64,
    beta: Complex64,
    velocity_element_indices: Vec<usize>,
}

/// Errors returned while constructing or applying a raw velocity block.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TbemVelocityOperatorError {
    /// The supplied velocity vector does not cover every non-evaluation panel.
    #[error("velocity vector length {got} does not match boundary element count {expected}")]
    VelocityLength {
        /// Number of entries supplied by the caller.
        got: usize,
        /// Number of active boundary panels required by the operator.
        expected: usize,
    },
    /// A velocity sample was NaN or infinite.
    #[error("velocity entry {index} is not finite")]
    NonFiniteVelocity {
        /// Index of the non-finite velocity sample.
        index: usize,
    },
    /// The pressure system does not have one DOF per active boundary panel.
    #[error("TBEM pressure DOF count {got} does not match boundary element count {expected}")]
    DofLayout {
        /// Number of pressure DOFs in the assembled system.
        got: usize,
        /// Number of active boundary panels required by the operator.
        expected: usize,
    },
    /// A compressed H-matrix input vector has the wrong dimension.
    #[error("H-matrix input length {got} does not match operator dimension {expected}")]
    HMatrixInputLength {
        /// Number of values supplied by the caller.
        got: usize,
        /// Number of values required by the operator.
        expected: usize,
    },
    /// A compressed H-matrix output buffer has the wrong dimension.
    #[error("H-matrix output length {got} does not match operator dimension {expected}")]
    HMatrixOutputLength {
        /// Number of values in the generated or supplied output.
        got: usize,
        /// Number of values required by the operator.
        expected: usize,
    },
    /// A compressed H-matrix input or generated output is non-finite.
    #[error("H-matrix value at flat index {index} is not finite")]
    HMatrixNonFinite {
        /// Flat index of the invalid value.
        index: usize,
    },
    /// A compressed-operator construction parameter is invalid.
    #[error("invalid H-matrix velocity-operator parameter: {name}")]
    InvalidParameter {
        /// Name of the invalid construction parameter.
        name: &'static str,
    },
}

impl StabilizedTbemVelocityOperator {
    /// Assemble a stabilized all-Neumann pressure operator with explicit
    /// Burton–Miller coupling.  Evaluation elements are retained in the mesh
    /// but do not consume a velocity entry.
    pub fn new(
        elements: &[Element],
        nodes: &[[f64; 3]],
        physics: &PhysicsParams,
        sign: f64,
        beta: Complex64,
    ) -> Result<Self, TbemVelocityOperatorError> {
        let mut normalized = elements.to_vec();
        let mut velocity_element_indices = Vec::new();
        for (index, element) in normalized.iter_mut().enumerate() {
            if element.property.is_evaluation() {
                continue;
            }
            element.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::default()]);
            velocity_element_indices.push(index);
        }
        let system =
            build_tbem_system_stabilized_with_beta(&normalized, nodes, physics, sign, beta);
        if system.num_dofs != velocity_element_indices.len() {
            return Err(TbemVelocityOperatorError::DofLayout {
                got: system.num_dofs,
                expected: velocity_element_indices.len(),
            });
        }
        Ok(Self {
            pressure_matrix: system.matrix,
            elements: normalized,
            nodes: nodes.to_vec(),
            physics: physics.clone(),
            sign,
            beta,
            velocity_element_indices,
        })
    }

    /// Number of pressure unknowns and explicit normal-velocity entries.
    pub fn len(&self) -> usize {
        self.velocity_element_indices.len()
    }

    /// Returns whether this operator has no active boundary-velocity entries.
    pub fn is_empty(&self) -> bool {
        self.velocity_element_indices.is_empty()
    }

    /// Element indices corresponding to entries accepted by [`Self::velocity_rhs`].
    pub fn velocity_element_indices(&self) -> &[usize] {
        &self.velocity_element_indices
    }

    /// Stabilized pressure block `A` in `A p - Q v = 0`.
    pub fn pressure_matrix(&self) -> &DenseMatrix<Complex64> {
        &self.pressure_matrix
    }

    /// Assemble `Q v` for an arbitrary boundary velocity vector in the same
    /// element order as [`Self::velocity_element_indices`].
    pub fn velocity_rhs(
        &self,
        velocity: &[Complex64],
    ) -> Result<Vec<Complex64>, TbemVelocityOperatorError> {
        if velocity.len() != self.velocity_element_indices.len() {
            return Err(TbemVelocityOperatorError::VelocityLength {
                got: velocity.len(),
                expected: self.velocity_element_indices.len(),
            });
        }
        if let Some((index, _)) = velocity
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(TbemVelocityOperatorError::NonFiniteVelocity { index });
        }
        let mut elements = self.elements.clone();
        for (&element_index, &value) in self.velocity_element_indices.iter().zip(velocity) {
            elements[element_index].boundary_condition = BoundaryCondition::Velocity(vec![value]);
        }
        Ok(build_tbem_rhs_stabilized_with_beta(
            &elements,
            &self.nodes,
            &self.physics,
            self.sign,
            self.beta,
        ))
    }

    /// Assemble `Q V` for several boundary-velocity right-hand sides.
    ///
    /// The input is stored as `n_velocity × n_rhs`, with one velocity
    /// pattern per column.  This keeps the validated velocity contract in a
    /// single API and lets monolithic product assembly submit all port and
    /// driver columns together.  The current dense reference implementation
    /// still reuses the established TBEM RHS builder for each column; the
    /// same shape is consumed by compressed H/FMM backends.
    pub fn velocity_rhs_block(
        &self,
        velocity: &DenseMatrix<Complex64>,
    ) -> Result<DenseMatrix<Complex64>, TbemVelocityOperatorError> {
        if velocity.nrows() != self.velocity_element_indices.len() {
            return Err(TbemVelocityOperatorError::HMatrixInputLength {
                got: velocity.nrows(),
                expected: self.velocity_element_indices.len(),
            });
        }
        if let Some(index) = velocity
            .as_slice()
            .iter()
            .position(|value| !value.is_finite())
        {
            return Err(TbemVelocityOperatorError::HMatrixNonFinite { index });
        }

        let mut result = DenseMatrix::zeros(self.pressure_matrix.nrows(), velocity.ncols());
        let mut column = vec![Complex64::default(); velocity.nrows()];
        for rhs_index in 0..velocity.ncols() {
            for row in 0..velocity.nrows() {
                column[row] = velocity[[row, rhs_index]];
            }
            let rhs = self.velocity_rhs(&column)?;
            for (row, value) in rhs.into_iter().enumerate() {
                result[[row, rhs_index]] = value;
            }
        }
        Ok(result)
    }
}

/// Compressed stabilized exterior velocity operator.
///
/// The H-matrix supplied to [`Self::new`] must be assembled in pressure mode:
/// every active field element uses [`BoundaryCondition::Pressure`]. Its
/// entries contain the pressure-mode Burton--Miller operator, including the
/// pressure trace jump. The velocity right-hand-side block is then
///
/// ```text
/// Q = -i * omega * rho * H_pressure
///     + sign * i * omega * rho * beta * tau * I.
/// ```
///
/// This is the same constant-panel block produced by
/// [`build_tbem_rhs_stabilized_with_beta`], but the non-local part is applied
/// through the H-matrix's dense near-field and low-rank far-field blocks.
/// The operator accepts one velocity per active, non-evaluation panel.
pub struct HMatrixVelocityOperator {
    pressure_matrix: HMatrix,
    pressure_scale: Complex64,
    diagonal_correction: Complex64,
    velocity_element_indices: Vec<usize>,
}

impl HMatrixVelocityOperator {
    /// Wrap a pressure-mode H-matrix as the stabilized velocity RHS block.
    pub fn new(
        pressure_matrix: HMatrix,
        elements: &[Element],
        physics: &PhysicsParams,
        sign: f64,
        beta: Complex64,
    ) -> Result<Self, TbemVelocityOperatorError> {
        if !sign.is_finite() {
            return Err(TbemVelocityOperatorError::InvalidParameter { name: "sign" });
        }
        if !beta.is_finite() {
            return Err(TbemVelocityOperatorError::InvalidParameter { name: "beta" });
        }

        let mut velocity_element_indices = Vec::new();
        for (element_index, element) in elements.iter().enumerate() {
            if element.property.is_evaluation() {
                continue;
            }
            if element.dof_addresses.len() != 1 {
                return Err(TbemVelocityOperatorError::DofLayout {
                    got: element.dof_addresses.len(),
                    expected: 1,
                });
            }
            let expected_dof = velocity_element_indices.len();
            if element.dof_addresses[0] != expected_dof {
                return Err(TbemVelocityOperatorError::DofLayout {
                    got: element.dof_addresses[0],
                    expected: expected_dof,
                });
            }
            velocity_element_indices.push(element_index);
        }

        if pressure_matrix.num_dofs != velocity_element_indices.len() {
            return Err(TbemVelocityOperatorError::DofLayout {
                got: pressure_matrix.num_dofs,
                expected: velocity_element_indices.len(),
            });
        }

        let tau = Complex64::new(physics.tau, 0.0);
        let pressure_scale = Complex64::new(0.0, -physics.pressure_factor);
        let diagonal_correction = -sign * pressure_scale * beta * tau;
        if !pressure_scale.is_finite() || !diagonal_correction.is_finite() {
            return Err(TbemVelocityOperatorError::InvalidParameter {
                name: "physics, sign, or beta",
            });
        }

        Ok(Self {
            pressure_matrix,
            pressure_scale,
            diagonal_correction,
            velocity_element_indices,
        })
    }

    /// Build a pressure-mode H-matrix and wrap it as a velocity operator.
    pub fn build(
        elements: &[Element],
        nodes: &[[f64; 3]],
        clusters: &[crate::core::types::Cluster],
        physics: &PhysicsParams,
        aca_tolerance: f64,
        max_rank: usize,
        sign: f64,
        beta: Complex64,
    ) -> Result<Self, TbemVelocityOperatorError> {
        if !aca_tolerance.is_finite() || aca_tolerance <= 0.0 {
            return Err(TbemVelocityOperatorError::InvalidParameter {
                name: "aca_tolerance",
            });
        }
        if max_rank == 0 {
            return Err(TbemVelocityOperatorError::InvalidParameter { name: "max_rank" });
        }
        if clusters.is_empty() {
            return Err(TbemVelocityOperatorError::InvalidParameter { name: "clusters" });
        }
        if clusters.iter().any(|cluster| {
            cluster
                .element_indices
                .iter()
                .any(|&index| index >= elements.len())
                || cluster
                    .near_clusters
                    .iter()
                    .chain(cluster.far_clusters.iter())
                    .any(|&index| index >= clusters.len())
        }) {
            return Err(TbemVelocityOperatorError::InvalidParameter {
                name: "cluster references",
            });
        }

        let mut pressure_elements = elements.to_vec();
        for element in &mut pressure_elements {
            if !element.property.is_evaluation() {
                element.boundary_condition =
                    BoundaryCondition::Pressure(vec![Complex64::default()]);
            }
        }
        let pressure_matrix = build_hmatrix_with_variant_precision_and_beta(
            &pressure_elements,
            nodes,
            clusters,
            physics,
            aca_tolerance,
            max_rank,
            AcaVariant::default(),
            LowRankPrecision::default(),
            sign,
            beta,
        );
        Self::new(pressure_matrix, &pressure_elements, physics, sign, beta)
    }

    /// Number of pressure unknowns and explicit velocity entries.
    pub fn len(&self) -> usize {
        self.velocity_element_indices.len()
    }

    /// Returns whether this operator has no active velocity entries.
    pub fn is_empty(&self) -> bool {
        self.velocity_element_indices.is_empty()
    }

    /// Element indices corresponding to entries accepted by [`Self::apply`].
    pub fn velocity_element_indices(&self) -> &[usize] {
        &self.velocity_element_indices
    }

    /// Underlying pressure-mode H-matrix.
    pub fn pressure_matrix(&self) -> &HMatrix {
        &self.pressure_matrix
    }

    /// Apply the compressed velocity RHS block.
    pub fn apply(
        &self,
        velocity: &[Complex64],
    ) -> Result<Vec<Complex64>, TbemVelocityOperatorError> {
        self.validate_velocity(velocity)?;
        let mut result = self
            .pressure_matrix
            .matvec_checked(velocity)
            .map_err(Self::map_hmatrix_error)?;
        for (value, &velocity_value) in result.iter_mut().zip(velocity) {
            *value = self.pressure_scale * *value + self.diagonal_correction * velocity_value;
        }
        self.validate_output(&result)?;
        Ok(result)
    }

    /// Apply the algebraic transpose of the velocity RHS block.
    pub fn apply_transpose(
        &self,
        velocity: &[Complex64],
    ) -> Result<Vec<Complex64>, TbemVelocityOperatorError> {
        self.validate_velocity(velocity)?;
        let mut result = self
            .pressure_matrix
            .matvec_transpose_checked(velocity)
            .map_err(Self::map_hmatrix_error)?;
        for (value, &velocity_value) in result.iter_mut().zip(velocity) {
            *value = self.pressure_scale * *value + self.diagonal_correction * velocity_value;
        }
        self.validate_output(&result)?;
        Ok(result)
    }

    /// Apply the Hermitian transpose of the velocity RHS block.
    pub fn apply_hermitian(
        &self,
        velocity: &[Complex64],
    ) -> Result<Vec<Complex64>, TbemVelocityOperatorError> {
        self.validate_velocity(velocity)?;
        let mut result = self
            .pressure_matrix
            .matvec_hermitian_checked(velocity)
            .map_err(Self::map_hmatrix_error)?;
        for (value, &velocity_value) in result.iter_mut().zip(velocity) {
            *value = self.pressure_scale.conj() * *value
                + self.diagonal_correction.conj() * velocity_value;
        }
        self.validate_output(&result)?;
        Ok(result)
    }

    /// Apply the velocity RHS block to several right-hand sides.
    pub fn apply_block(
        &self,
        velocity: &DenseMatrix<Complex64>,
    ) -> Result<DenseMatrix<Complex64>, TbemVelocityOperatorError> {
        if velocity.nrows() != self.len() {
            return Err(TbemVelocityOperatorError::HMatrixInputLength {
                got: velocity.nrows(),
                expected: self.len(),
            });
        }
        if let Some(index) = velocity
            .as_slice()
            .iter()
            .position(|value| !value.is_finite())
        {
            return Err(TbemVelocityOperatorError::HMatrixNonFinite { index });
        }
        let mut result = self.pressure_matrix.matvec_block(velocity);
        for row in 0..result.nrows() {
            for column in 0..result.ncols() {
                result[[row, column]] = self.pressure_scale * result[[row, column]]
                    + self.diagonal_correction * velocity[[row, column]];
            }
        }
        if result.nrows() != self.len() {
            return Err(TbemVelocityOperatorError::HMatrixOutputLength {
                got: result.nrows(),
                expected: self.len(),
            });
        }
        if let Some(index) = result
            .as_slice()
            .iter()
            .position(|value| !value.is_finite())
        {
            return Err(TbemVelocityOperatorError::HMatrixNonFinite { index });
        }
        Ok(result)
    }

    fn validate_velocity(&self, velocity: &[Complex64]) -> Result<(), TbemVelocityOperatorError> {
        if velocity.len() != self.len() {
            return Err(TbemVelocityOperatorError::VelocityLength {
                got: velocity.len(),
                expected: self.len(),
            });
        }
        if let Some(index) = velocity.iter().position(|value| !value.is_finite()) {
            return Err(TbemVelocityOperatorError::NonFiniteVelocity { index });
        }
        Ok(())
    }

    fn validate_output(&self, output: &[Complex64]) -> Result<(), TbemVelocityOperatorError> {
        if output.len() != self.len() {
            return Err(TbemVelocityOperatorError::HMatrixOutputLength {
                got: output.len(),
                expected: self.len(),
            });
        }
        if let Some(index) = output.iter().position(|value| !value.is_finite()) {
            return Err(TbemVelocityOperatorError::HMatrixNonFinite { index });
        }
        Ok(())
    }

    fn map_hmatrix_error(error: HMatrixOperatorError) -> TbemVelocityOperatorError {
        match error {
            HMatrixOperatorError::InputLength { expected, actual } => {
                TbemVelocityOperatorError::HMatrixInputLength {
                    got: actual,
                    expected,
                }
            }
            HMatrixOperatorError::OutputLength { expected, actual } => {
                TbemVelocityOperatorError::HMatrixOutputLength {
                    got: actual,
                    expected,
                }
            }
            HMatrixOperatorError::BlockInputRows { expected, actual } => {
                TbemVelocityOperatorError::HMatrixInputLength {
                    got: actual,
                    expected,
                }
            }
            HMatrixOperatorError::BlockOutputRows { expected, actual } => {
                TbemVelocityOperatorError::HMatrixOutputLength {
                    got: actual,
                    expected,
                }
            }
            HMatrixOperatorError::BlockColumnCount { input, output } => {
                TbemVelocityOperatorError::HMatrixOutputLength {
                    got: output,
                    expected: input,
                }
            }
            HMatrixOperatorError::NonFiniteValue { index } => {
                TbemVelocityOperatorError::HMatrixNonFinite { index }
            }
        }
    }
}

impl LinearOperator<Complex64> for HMatrixVelocityOperator {
    fn num_rows(&self) -> usize {
        self.len()
    }

    fn num_cols(&self) -> usize {
        self.len()
    }

    fn apply(&self, x: &[Complex64]) -> Vec<Complex64> {
        HMatrixVelocityOperator::apply(self, x).expect("validated H-matrix velocity operator input")
    }

    fn apply_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        assert_eq!(y.len(), self.len(), "velocity operator output length");
        let result = HMatrixVelocityOperator::apply(self, x)
            .expect("validated H-matrix velocity operator input");
        y.copy_from_slice(&result);
    }

    fn apply_block(&self, x: &DenseMatrix<Complex64>, y: &mut DenseMatrix<Complex64>) {
        assert_eq!(x.nrows(), self.len(), "velocity operator block input rows");
        assert_eq!(y.nrows(), self.len(), "velocity operator block output rows");
        assert_eq!(x.ncols(), y.ncols(), "velocity operator block columns");
        let result = HMatrixVelocityOperator::apply_block(self, x)
            .expect("validated H-matrix velocity operator block input");
        y.as_mut_slice().copy_from_slice(result.as_slice());
    }

    fn apply_transpose(&self, x: &[Complex64]) -> Vec<Complex64> {
        HMatrixVelocityOperator::apply_transpose(self, x)
            .expect("validated H-matrix velocity operator input")
    }

    fn apply_transpose_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        assert_eq!(y.len(), self.len(), "velocity operator output length");
        let result = HMatrixVelocityOperator::apply_transpose(self, x)
            .expect("validated H-matrix velocity operator input");
        y.copy_from_slice(&result);
    }

    fn apply_hermitian(&self, x: &[Complex64]) -> Vec<Complex64> {
        HMatrixVelocityOperator::apply_hermitian(self, x)
            .expect("validated H-matrix velocity operator input")
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }
}

/// Build the TBEM system matrix and RHS vector
///
/// This implements NC_BuildSystemTBEM from the C++ code.
/// Uses Burton-Miller formulation: (γτG + βH^T) for the system matrix.
///
/// # Arguments
/// * `elements` - Vector of mesh elements
/// * `nodes` - Node coordinates (num_nodes × 3)
/// * `physics` - Physics parameters (wave number, etc.)
///
/// # Returns
/// TbemSystem with assembled matrix and RHS
pub fn build_tbem_system(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
) -> TbemSystem {
    build_tbem_system_with_beta(elements, nodes, physics, physics.burton_miller_beta())
}

/// Build a TBEM system after enforcing adjacent-order quadrature convergence.
///
/// This is the checked production entry point for constant-panel TBEM.  It
/// validates every active source/field pair (including self interactions) and
/// returns the pair location when any kernel channel exceeds the mixed
/// absolute/relative budget.  The validated system is then assembled through
/// the same parallel implementation as [`build_tbem_system_with_beta`], so
/// callers can opt into a typed accuracy failure without changing matrix
/// layout or Burton--Miller conventions.
pub fn build_tbem_system_checked(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    policy: TbemQuadraturePolicy,
) -> Result<TbemSystem, TbemQuadraturePairError> {
    validate_tbem_quadrature(elements, nodes, physics, &policy)?;
    Ok(build_tbem_system(elements, nodes, physics))
}

/// Validate all active TBEM element pairs using an adjacent-order estimate.
///
/// This helper is useful when a caller wants to run the gate separately from
/// assembly (for example, to cache the result before a frequency sweep).
fn validate_tbem_quadrature_constant(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    policy: &TbemQuadraturePolicy,
) -> Result<(), TbemQuadraturePairError> {
    let all_coords: Vec<DenseMatrix<f64>> = elements
        .iter()
        .map(|element| get_element_coords(element, nodes))
        .collect();

    for (source_index, source) in elements.iter().enumerate() {
        if source.property.is_evaluation() {
            continue;
        }
        for (field_index, field) in elements.iter().enumerate() {
            if field.property.is_evaluation() {
                continue;
            }
            let result = if source_index == field_index {
                singular_integration_checked(
                    &source.center,
                    &source.normal,
                    &all_coords[field_index],
                    field.element_type,
                    physics,
                    None,
                    0,
                    false,
                    &policy.singular_coarse,
                    &policy.singular_fine,
                    None,
                    policy.absolute_tolerance,
                    policy.relative_tolerance,
                )
                .map(|_| ())
            } else {
                regular_integration_checked(
                    &source.center,
                    &source.normal,
                    &all_coords[field_index],
                    field.element_type,
                    physics,
                    policy.regular_coarse_order,
                    policy.regular_fine_order,
                    policy.absolute_tolerance,
                    policy.relative_tolerance,
                )
                .map(|_| ())
            };
            if let Err(error) = result {
                return Err(TbemQuadraturePairError {
                    source_element: source_index,
                    field_element: field_index,
                    error,
                });
            }
        }
    }
    Ok(())
}

/// Validate all active TBEM element pairs using the selected panel basis.
///
/// Constant panels retain the historical scalar estimator. Linear and
/// quadratic panels use basis-weighted regular and singular evaluators so
/// every coefficient is covered by the admission budget. Continuous quadratic
/// numbering does not change these local panel integrals.
pub fn validate_tbem_quadrature_for_basis(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    basis: BasisOrder,
    policy: &TbemQuadraturePolicy,
) -> Result<(), TbemQuadraturePairError> {
    if basis == BasisOrder::Constant {
        return validate_tbem_quadrature_constant(elements, nodes, physics, policy);
    }

    let all_coords: Vec<DenseMatrix<f64>> = elements
        .iter()
        .map(|element| get_element_coords(element, nodes))
        .collect();
    for (source_index, source) in elements.iter().enumerate() {
        if source.property.is_evaluation() {
            continue;
        }
        for (field_index, field) in elements.iter().enumerate() {
            if field.property.is_evaluation() {
                continue;
            }
            if let Err(error) = validate_higher_order_quadrature_pair(
                source,
                field,
                &all_coords[field_index],
                nodes,
                physics,
                basis,
                source_index == field_index,
                policy,
            ) {
                return Err(TbemQuadraturePairError {
                    source_element: source_index,
                    field_element: field_index,
                    error,
                });
            }
        }
    }
    Ok(())
}

/// Backwards-compatible constant-panel admission entry point.
pub fn validate_tbem_quadrature(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    policy: &TbemQuadraturePolicy,
) -> Result<(), TbemQuadraturePairError> {
    validate_tbem_quadrature_for_basis(elements, nodes, physics, BasisOrder::Constant, policy)
}

fn validate_higher_order_quadrature_pair(
    source: &Element,
    field: &Element,
    field_coords: &DenseMatrix<f64>,
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    basis: BasisOrder,
    self_pair: bool,
    policy: &TbemQuadraturePolicy,
) -> Result<(), QuadratureConvergenceError> {
    if !policy.absolute_tolerance.is_finite()
        || !policy.relative_tolerance.is_finite()
        || policy.absolute_tolerance < 0.0
        || policy.relative_tolerance < 0.0
    {
        return Err(QuadratureConvergenceError::InvalidTolerance {
            absolute_tolerance: policy.absolute_tolerance,
            relative_tolerance: policy.relative_tolerance,
        });
    }
    if policy.regular_coarse_order == 0
        || policy.regular_fine_order <= policy.regular_coarse_order
        || policy.regular_fine_order > crate::core::integration::gauss::N_GAUORDER
    {
        return Err(QuadratureConvergenceError::InvalidOrder(
            crate::core::types::QuadratureOrderError {
                coarse_order: policy.regular_coarse_order,
                fine_order: policy.regular_fine_order,
            },
        ));
    }
    let valid_singular = |params: QuadratureParams| {
        (1..=crate::core::integration::gauss::N_GAUORDER).contains(&params.edge_gauss_order)
            && (1..=crate::core::integration::gauss::N_GAUORDER)
                .contains(&params.subelement_gauss_order)
            && params.edge_sections > 0
            && params.subtriangles_per_section > 0
    };
    if !valid_singular(policy.singular_coarse) || !valid_singular(policy.singular_fine) {
        return Err(QuadratureConvergenceError::InvalidOrder(
            crate::core::types::QuadratureOrderError {
                coarse_order: policy.singular_coarse.subelement_gauss_order,
                fine_order: policy.singular_fine.subelement_gauss_order,
            },
        ));
    }

    let num_basis = field.num_dofs_for_basis(basis);
    let source_points = source.collocation_points(basis, nodes);
    for source_point in source_points {
        let source_normal = source.normal;
        let (coarse, fine) = if self_pair {
            (
                singular_integration_linear_with_params(
                    &source_point,
                    &source_normal,
                    field_coords,
                    field.element_type,
                    physics,
                    num_basis,
                    &policy.singular_coarse,
                    None,
                ),
                singular_integration_linear_with_params(
                    &source_point,
                    &source_normal,
                    field_coords,
                    field.element_type,
                    physics,
                    num_basis,
                    &policy.singular_fine,
                    None,
                ),
            )
        } else {
            (
                regular_integration_linear_fixed_order(
                    &source_point,
                    &source_normal,
                    field_coords,
                    field.element_type,
                    physics,
                    num_basis,
                    policy.regular_coarse_order,
                ),
                regular_integration_linear_fixed_order(
                    &source_point,
                    &source_normal,
                    field_coords,
                    field.element_type,
                    physics,
                    num_basis,
                    policy.regular_fine_order,
                ),
            )
        };

        let mut max_absolute: f64 = 0.0;
        let mut max_relative: f64 = 0.0;
        let mut converged = true;
        for (coarse, fine) in coarse.basis_results.iter().zip(fine.basis_results.iter()) {
            let error = coarse.error_estimate(fine);
            max_absolute = max_absolute.max(error.max_absolute);
            max_relative = max_relative.max(error.max_relative);
            converged &= error.is_converged(
                coarse,
                fine,
                policy.absolute_tolerance,
                policy.relative_tolerance,
            );
        }
        if !max_absolute.is_finite() || !max_relative.is_finite() || !converged {
            return Err(QuadratureConvergenceError::NotConverged {
                max_absolute,
                max_relative,
                absolute_tolerance: policy.absolute_tolerance,
                relative_tolerance: policy.relative_tolerance,
            });
        }
    }
    Ok(())
}

/// Build TBEM system with mesh-adaptive Burton-Miller coupling
///
/// Uses β = i/(k + k_ref) where k_ref = 1/element_size for better conditioning
/// at low-to-mid frequencies (ka < 2). This avoids the 1/k divergence of the
/// traditional β = i/k formulation.
///
/// # Arguments
/// * `elements` - Vector of mesh elements
/// * `nodes` - Node coordinates (num_nodes × 3)
/// * `physics` - Physics parameters (wave number, etc.)
/// * `avg_element_size` - Average element edge length or sqrt(area)
pub fn build_tbem_system_bounded(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    avg_element_size: f64,
) -> TbemSystem {
    let beta = physics.burton_miller_beta_optimal(avg_element_size);
    build_tbem_system_with_beta(elements, nodes, physics, beta)
}

/// Build TBEM system with scaled Burton-Miller coupling
///
/// Uses β = scale × i/k for improved numerical conditioning.
/// Testing shows scale = 2.0 gives significantly better accuracy than the
/// standard β = i/k, especially at low to medium frequencies (ka < 2).
///
/// # Arguments
/// * `elements` - Vector of mesh elements
/// * `nodes` - Node coordinates (num_nodes × 3)
/// * `physics` - Physics parameters (wave number, etc.)
/// * `scale` - Beta scaling factor (recommended: 2.0)
pub fn build_tbem_system_scaled(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    scale: f64,
) -> TbemSystem {
    let beta = physics.burton_miller_beta_scaled(scale);
    build_tbem_system_with_beta(elements, nodes, physics, beta)
}

/// Build TBEM system with custom Burton-Miller coupling parameter
pub fn build_tbem_system_with_beta(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    beta: Complex64,
) -> TbemSystem {
    build_tbem_system_with_beta_parallel(elements, nodes, physics, beta)
}

/// Checked variant of [`build_tbem_system_with_beta`].
///
/// The parallel assembler sizes rows from `count_dofs` (maximum
/// non-evaluation DOF index + 1) and silently leaves a zero row — a singular
/// system with no diagnostic — for any DOF index no source element owns
/// (e.g. gapped DOF numbering). This entry point rejects that input with a
/// [`MissingSourceDof`] error before any kernel evaluation begins. The legacy
/// infallible builder is unchanged for compatibility.
pub fn try_build_tbem_system_with_beta(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    beta: Complex64,
) -> Result<TbemSystem, MissingSourceDof> {
    let num_dofs = count_dofs(elements);
    let mut owned = vec![false; num_dofs];
    for source_elem in elements.iter() {
        if source_elem.property.is_evaluation() {
            continue;
        }
        let source_dof = source_elem.dof_addresses[0];
        if source_dof < num_dofs {
            owned[source_dof] = true;
        }
    }
    if let Some(dof) = owned.iter().position(|&has| !has) {
        return Err(MissingSourceDof { dof });
    }
    Ok(build_tbem_system_with_beta_parallel(
        elements, nodes, physics, beta,
    ))
}

/// Check if BC values are non-zero
fn has_nonzero_bc(values: &[Complex64]) -> bool {
    values.iter().any(|v| v.norm() > 1e-15)
}

/// Add diagonal free terms from jump conditions
///
/// Burton-Miller formulation for exterior problems:
/// - Velocity BC: CBIE jump → -γ/2 on diagonal; HBIE jump → -(β·τ/2)·(iωρ·v) on RHS
/// - Pressure BC: HBIE jump → +β·τ/2 on diagonal; CBIE jump → -(γ/2)·p on RHS
fn add_free_terms(
    system: &mut TbemSystem,
    source_dof: usize,
    bc_type: i32,
    bc_value: &[Complex64],
    gamma: Complex64,
    tau: Complex64,
    beta: Complex64,
    i_omega_rho: Complex64,
    admittance: Option<Complex64>,
) {
    let avg_bc = bc_value.iter().sum::<Complex64>() / bc_value.len() as f64;

    match bc_type {
        0 => {
            // CBIE jump: -γ/2 on diagonal for unknown p
            system.matrix[[source_dof, source_dof]] -= gamma * 0.5;
            if let Some(admittance) = admittance {
                system.matrix[[source_dof, source_dof]] -=
                    admittance * i_omega_rho * beta * tau * 0.5;
            }
            // HBIE jump: velocity enters via ∂p/∂n = iωρ·v_n
            system.rhs[source_dof] += avg_bc * i_omega_rho * beta * tau * 0.5;
        }
        1 => {
            // Pressure BC (Dirichlet): unknown is velocity v_n
            // HBIE jump for K^T acting on unknown v: +β·τ/2 on the diagonal
            system.matrix[[source_dof, source_dof]] += beta * tau * 0.5;
            // RHS: CBIE free term c·p_known = (-γ/2)·p_known moved to RHS
            system.rhs[source_dof] -= avg_bc * gamma * 0.5;
        }
        _ => {}
    }
}

/// Build TBEM system in parallel using rayon (with custom beta)
pub fn build_tbem_system_with_beta_parallel(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    beta: Complex64,
) -> TbemSystem {
    use rayon::prelude::*;

    let num_dofs = count_dofs(elements);

    let gamma = Complex64::new(physics.gamma(), 0.0);
    let tau = Complex64::new(physics.tau, 0.0);
    let i_omega_rho = Complex64::new(0.0, physics.pressure_factor);

    let all_coords: Vec<DenseMatrix<f64>> = elements
        .iter()
        .map(|e| get_element_coords(e, nodes))
        .collect();

    let mut system = TbemSystem::new(num_dofs);
    if num_dofs == 0 {
        return system;
    }

    let mut source_elements_by_dof = vec![None; num_dofs];
    for (iel, source_elem) in elements.iter().enumerate() {
        if source_elem.property.is_evaluation() {
            continue;
        }

        let source_dof = source_elem.dof_addresses[0];
        assert!(
            source_dof < num_dofs,
            "TBEM source DOF {source_dof} is outside the {num_dofs}-DOF system"
        );
        assert!(
            source_elements_by_dof[source_dof]
                .replace((iel, source_elem))
                .is_none(),
            "TBEM constant-panel assembly requires a unique source element for DOF {source_dof}"
        );
    }

    // Row assembly (rayon above the machine-adaptive gate, serial for small
    // systems). The row closure is shared by both paths so assembled rows
    // are identical.
    let assemble_row = |source_dof: usize,
                        local_row: &mut [Complex64],
                        rhs_slot: &mut Complex64| {
        let Some(&(iel, source_elem)) = source_elements_by_dof[source_dof].as_ref() else {
            return;
        };

        let source_point = &source_elem.center;
        let source_normal = &source_elem.normal;
        let source_dof = source_elem.dof_addresses[0];
        let (bc_type, bc_value) = get_bc_type_and_value(&source_elem.boundary_condition);

        let mut local_rhs = Complex64::new(0.0, 0.0);

        let avg_bc = bc_value.iter().sum::<Complex64>() / bc_value.len() as f64;
        match bc_type {
            0 => {
                local_row[source_dof] -= gamma * 0.5;
                if let Some(admittance) = source_elem.boundary_condition.admittance() {
                    local_row[source_dof] -= admittance * i_omega_rho * beta * tau * 0.5;
                }
                local_rhs += avg_bc * i_omega_rho * beta * tau * 0.5;
            }
            1 => {
                local_row[source_dof] += beta * tau * 0.5;
                local_rhs -= avg_bc * gamma * 0.5;
            }
            _ => {}
        }

        for (jel, field_elem) in elements.iter().enumerate() {
            if field_elem.property.is_evaluation() {
                continue;
            }

            let element_coords = &all_coords[jel];
            let field_dof = field_elem.dof_addresses[0];
            let (field_bc_type, field_bc_values) =
                get_bc_type_and_value(&field_elem.boundary_condition);
            let compute_rhs = has_nonzero_bc(&field_bc_values);

            let result = if jel == iel {
                singular_integration(
                    source_point,
                    source_normal,
                    element_coords,
                    field_elem.element_type,
                    physics,
                    if compute_rhs {
                        Some(&field_bc_values)
                    } else {
                        None
                    },
                    field_bc_type,
                    compute_rhs,
                )
            } else {
                regular_integration(
                    source_point,
                    source_normal,
                    element_coords,
                    field_elem.element_type,
                    field_elem.area,
                    physics,
                    if compute_rhs {
                        Some(&field_bc_values)
                    } else {
                        None
                    },
                    field_bc_type,
                    compute_rhs,
                )
            };

            let mut coeff = match field_bc_type {
                0 => result.dg_dn_integral * gamma * tau + result.d2g_dnxdny_integral * beta,
                1 => -(result.g_integral * gamma * tau + result.dg_dnx_integral * beta),
                _ => Complex64::new(0.0, 0.0),
            };
            if field_bc_type == 0
                && let Some(admittance) = field_elem.boundary_condition.admittance()
            {
                coeff -= (result.g_integral * gamma * tau + result.dg_dnx_integral * beta)
                    * admittance
                    * i_omega_rho;
            }

            local_row[field_dof] += coeff;

            if compute_rhs {
                let avg_field_bc =
                    field_bc_values.iter().sum::<Complex64>() / field_bc_values.len() as f64;
                let rhs_contrib = match field_bc_type {
                    0 => {
                        (result.g_integral * gamma * tau + result.dg_dnx_integral * beta)
                            * avg_field_bc
                            * i_omega_rho
                    }
                    1 => {
                        (result.dg_dn_integral * gamma * tau + result.d2g_dnxdny_integral * beta)
                            * avg_field_bc
                    }
                    _ => Complex64::new(0.0, 0.0),
                };
                local_rhs += rhs_contrib;
            }
        }

        *rhs_slot = local_rhs;
    };
    if use_parallel(
        num_dofs.saturating_mul(num_dofs),
        BEM_QUADRATURE_PAIRS_PER_THREAD,
    ) {
        system
            .matrix
            .as_mut_slice()
            .par_chunks_mut(num_dofs)
            .zip(system.rhs.par_iter_mut())
            .enumerate()
            .for_each(|(source_dof, (local_row, rhs_slot))| {
                assemble_row(source_dof, local_row, rhs_slot);
            });
    } else {
        for (source_dof, (local_row, rhs_slot)) in system
            .matrix
            .as_mut_slice()
            .chunks_mut(num_dofs)
            .zip(system.rhs.iter_mut())
            .enumerate()
        {
            assemble_row(source_dof, local_row, rhs_slot);
        }
    }

    system
}

/// Build TBEM system in parallel using rayon
pub fn build_tbem_system_parallel(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
) -> TbemSystem {
    let beta = physics.burton_miller_beta();
    build_tbem_system_with_beta_parallel(elements, nodes, physics, beta)
}

/// Build TBEM system for discontinuous linear elements
///
/// Each element has N_nodes DOFs (3 for Tri3, 4 for Quad4).
/// Collocation points are at interior Gauss points, not at vertices or centroid.
///
/// The system matrix is (3*N_elements) x (3*N_elements) for triangular meshes.
///
/// # Arguments
/// * `elements` - Vector of mesh elements (with dof_addresses set for linear)
/// * `nodes` - Node coordinates
/// * `physics` - Physics parameters
/// * `beta` - Burton-Miller coupling parameter
pub fn build_tbem_system_linear(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    beta: num_complex::Complex64,
) -> TbemSystem {
    let num_dofs = count_dofs(elements);
    let mut system = TbemSystem::new(num_dofs);

    let gamma = Complex64::new(physics.gamma(), 0.0);
    let tau = Complex64::new(physics.tau, 0.0);
    let i_omega_rho = Complex64::new(0.0, physics.pressure_factor);

    let basis = BasisOrder::Linear;

    let all_coords: Vec<DenseMatrix<f64>> = elements
        .iter()
        .map(|e| get_element_coords(e, nodes))
        .collect();

    // Loop over source elements
    for (iel, source_elem) in elements.iter().enumerate() {
        if source_elem.property.is_evaluation() {
            continue;
        }

        let num_source_dofs = source_elem.num_dofs_for_basis(basis);
        let collocation_points = source_elem.collocation_points(basis, nodes);
        // Tri3 geometry is affine/flat even when the pressure basis is P1.
        // Smoothed vertex normals without a matching curved mapping make the
        // hypersingular operator inconsistent with its jump term.
        // Match accelerated higher-order proxy geometry on curved panels.
        let curved_source = source_elem
            .node_normals
            .iter()
            .any(|normal| normal.iter().any(|value| value.abs() > 1e-15));
        let collocation_normals = if curved_source {
            source_elem.collocation_normals(basis, nodes)
        } else {
            vec![source_elem.normal; num_source_dofs]
        };

        let (bc_type, bc_value) = get_bc_type_and_value(&source_elem.boundary_condition);
        let bc_admittance = source_elem.boundary_condition.admittance();

        // For each collocation point within the source element
        for c in 0..num_source_dofs {
            let source_point = &collocation_points[c];
            let source_normal = &collocation_normals[c];
            let source_dof = source_elem.dof_addresses[c];

            // Add diagonal free terms (jump conditions)
            add_free_terms(
                &mut system,
                source_dof,
                bc_type,
                &bc_value,
                gamma,
                tau,
                beta,
                i_omega_rho,
                bc_admittance,
            );

            // Loop over field elements
            for (jel, field_elem) in elements.iter().enumerate() {
                if field_elem.property.is_evaluation() {
                    continue;
                }

                let element_coords = &all_coords[jel];
                let num_field_dofs = field_elem.num_dofs_for_basis(basis);
                let (field_bc_type, field_bc_values) =
                    get_bc_type_and_value(&field_elem.boundary_condition);
                let compute_rhs = has_nonzero_bc(&field_bc_values);

                // Compute integrals weighted by basis functions
                let linear_result = if jel == iel {
                    // Singular integration (self-element)
                    singular_integration_linear(
                        source_point,
                        source_normal,
                        element_coords,
                        field_elem.element_type,
                        physics,
                        num_field_dofs,
                    )
                } else {
                    // Regular integration
                    regular_integration_linear(
                        source_point,
                        source_normal,
                        element_coords,
                        field_elem.element_type,
                        field_elem.area,
                        physics,
                        num_field_dofs,
                    )
                };

                // Assemble contributions for each field basis function
                for b in 0..num_field_dofs {
                    let field_dof = field_elem.dof_addresses[b];
                    let result = &linear_result.basis_results[b];

                    let mut coeff = match field_bc_type {
                        0 => {
                            // Velocity BC: K' + βH
                            result.dg_dn_integral * gamma * tau + result.d2g_dnxdny_integral * beta
                        }
                        1 => {
                            // Pressure BC: -(G + βH^T)
                            -(result.g_integral * gamma * tau + result.dg_dnx_integral * beta)
                        }
                        _ => Complex64::new(0.0, 0.0),
                    };
                    if field_bc_type == 0
                        && let Some(admittance) = field_elem.boundary_condition.admittance()
                    {
                        coeff -= (result.g_integral * gamma * tau + result.dg_dnx_integral * beta)
                            * admittance
                            * i_omega_rho;
                    }

                    system.matrix[[source_dof, field_dof]] += coeff;

                    // Accumulate RHS contribution from prescribed boundary conditions
                    if compute_rhs {
                        let bc_val = field_bc_values
                            .get(b)
                            .copied()
                            .unwrap_or(Complex64::new(0.0, 0.0));
                        let rhs_contrib = match field_bc_type {
                            0 => {
                                // Velocity BC: ∂p/∂n = iωρ·v
                                (result.g_integral * gamma * tau + result.dg_dnx_integral * beta)
                                    * bc_val
                                    * i_omega_rho
                            }
                            1 => {
                                // Pressure BC: known p enters directly
                                -(result.dg_dn_integral * gamma * tau
                                    + result.d2g_dnxdny_integral * beta)
                                    * bc_val
                            }
                            _ => Complex64::new(0.0, 0.0),
                        };
                        system.rhs[source_dof] += rhs_contrib;
                    }
                }
            }
        }
    }

    system
}

/// Return the average combined-operator row-sum diagnostic without mutating it.
///
/// This function retains its historical name for API compatibility. The old
/// implementation forced every row sum to zero, applying the Laplace identity
/// `W₀[1] = 0` to the finite-frequency combined Helmholtz operator. At nonzero
/// wavenumber the constant mode is not a null mode; the mutation therefore made
/// valid scattering matrices nearly singular. It is intentionally a no-op.
///
/// # Arguments
/// * `system` - The assembled TBEM system (read-only; retained as `&mut`
///   for API compatibility but never mutated)
///
/// # Returns
/// The average row sum magnitude (for diagnostics)
///
/// # Deprecated
///
/// Kept for API compatibility: it is a diagnostic-only no-op. New code must
/// use [`build_tbem_system_corrected`], which reports the same diagnostic
/// without the misleading "correction" name.
#[deprecated(note = "diagnostic-only no-op; use build_tbem_system_corrected")]
pub fn apply_row_sum_correction(system: &mut TbemSystem) -> f64 {
    average_row_sum(system)
}

/// Average combined-operator row-sum magnitude (diagnostic only).
fn average_row_sum(system: &TbemSystem) -> f64 {
    let n = system.num_dofs;
    if n == 0 {
        return 0.0;
    }
    let mut total_row_sum = Complex64::new(0.0, 0.0);

    for i in 0..n {
        let mut row_sum = Complex64::new(0.0, 0.0);
        for j in 0..n {
            row_sum += system.matrix[[i, j]];
        }
        total_row_sum += row_sum;
    }

    total_row_sum.norm() / n as f64
}

/// Build a TBEM system and report its row-sum diagnostic.
///
/// The returned system is unmodified.
pub fn build_tbem_system_corrected(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
) -> (TbemSystem, f64) {
    let system = build_tbem_system(elements, nodes, physics);
    let correction = average_row_sum(&system);
    (system, correction)
}

/// Build a constant-panel TBEM system with stabilized Burton-Miller formulation.
///
/// Uses a sign parameter and beta scaling for improved conditioning.
/// Constant panels use their flat geometric normal consistently for the source,
/// field kernels, surface measure, and jump terms.  Interpolated vertex normals
/// belong to the discontinuous-linear/curved path; mixing them into a flat-panel
/// operator breaks consistency and makes the Mie surface error plateau under
/// refinement.
pub fn build_tbem_system_stabilized(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
) -> TbemSystem {
    let beta = physics.burton_miller_beta_scaled(beta_scale);
    build_tbem_system_stabilized_with_beta(elements, nodes, physics, sign, beta)
}

/// Assemble the fixed-geometry constant-panel TBEM derivative with respect to
/// acoustic wavenumber `k`.
///
/// Kernel phase, all four operator channels, the Burton--Miller coupling, and
/// the pressure/velocity conversion factor are differentiated analytically.
/// Geometry, panel normals, boundary admittances, formulation sign, and the
/// supplied `beta_scale` are held fixed.  The returned RHS derivative is useful
/// for incident/load sensitivity contracts; callers that only need the matrix
/// derivative may ignore it.  This is intentionally a constant-panel CPU
/// contract; higher-order, moving-geometry, and material derivatives remain
/// separate APIs.
pub fn build_tbem_system_stabilized_wavenumber_derivative(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
) -> TbemSystem {
    build_tbem_system_stabilized_parameter_derivative(
        elements,
        nodes,
        physics,
        sign,
        beta_scale,
        1.0,
        physics.pressure_factor_wavenumber_derivative(),
        0.0,
    )
}

/// Assemble the fixed-geometry constant-panel TBEM derivative with respect
/// to sound speed at fixed frequency and density.
///
/// This uses `dk/dc = -k/c` while keeping the pressure/velocity conversion
/// factor fixed because `iρω` does not depend on sound speed at fixed
/// frequency. It is a dense CPU constant-panel contract; higher-order,
/// moving-geometry, material-interface, and compressed/device derivatives
/// remain separate APIs.
pub fn build_tbem_system_stabilized_speed_derivative(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
) -> TbemSystem {
    build_tbem_system_stabilized_parameter_derivative(
        elements,
        nodes,
        physics,
        sign,
        beta_scale,
        -physics.wave_number / physics.speed_of_sound,
        0.0,
        0.0,
    )
}

/// Assemble the fixed-geometry constant-panel TBEM derivative with respect
/// to medium density at fixed frequency and sound speed.
///
/// The acoustic kernels and wavenumber are density-independent.  Density
/// enters the pressure/normal-velocity conversion factor as `i * omega *
/// density * harmonic_factor`, so the derivative uses
/// `d(pressure_factor)/d(rho) = pressure_factor / rho`.  This is a dense CPU
/// constant-panel contract; higher-order, moving-geometry, interface,
/// compressed, and device derivative paths remain separate APIs.
pub fn build_tbem_system_stabilized_density_derivative(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
) -> TbemSystem {
    build_tbem_system_stabilized_parameter_derivative(
        elements,
        nodes,
        physics,
        sign,
        beta_scale,
        0.0,
        physics.pressure_factor / physics.density,
        0.0,
    )
}

/// Assemble the fixed-geometry density derivative for the impedance boundary
/// condition generated by [`crate::core::bem_solver::BoundaryConditionType::Impedance`].  That
/// condition uses `Y = 1 / (rho * c)`, so its admittance derivative is
/// included alongside the pressure/velocity conversion derivative.
pub fn build_tbem_system_stabilized_impedance_density_derivative(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
) -> TbemSystem {
    build_tbem_system_stabilized_parameter_derivative(
        elements,
        nodes,
        physics,
        sign,
        beta_scale,
        0.0,
        physics.pressure_factor / physics.density,
        -1.0 / physics.density,
    )
}

/// Assemble the fixed-geometry TBEM derivative with respect to sound speed
/// for an impedance boundary condition whose admittance is
/// `Y = 1 / (rho * c)`.
///
/// The derivative includes both `dk/dc = -k/c` in the Burton--Miller kernel
/// and `dY/dc = -Y/c` in the impedance coupling. Geometry, density, boundary
/// values, formulation sign, and `beta_scale` remain fixed. This is a checked
/// constant-panel CPU contract; higher-order impedance derivatives remain a
/// separate path.
pub fn build_tbem_system_stabilized_impedance_speed_derivative(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
) -> TbemSystem {
    build_tbem_system_stabilized_parameter_derivative(
        elements,
        nodes,
        physics,
        sign,
        beta_scale,
        -physics.wave_number / physics.speed_of_sound,
        0.0,
        -1.0 / physics.speed_of_sound,
    )
}

fn build_tbem_system_stabilized_parameter_derivative(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
    wavenumber_scale: f64,
    pressure_factor_derivative: f64,
    admittance_density_scale: f64,
) -> TbemSystem {
    let num_dofs = count_dofs(elements);
    let mut system = TbemSystem::new(num_dofs);
    if num_dofs == 0 {
        return system;
    }
    let gamma = Complex64::new(physics.gamma(), 0.0);
    let tau = Complex64::new(physics.tau, 0.0);
    let beta = physics.burton_miller_beta_scaled(beta_scale);
    let beta_derivative =
        physics.burton_miller_beta_wavenumber_derivative() * beta_scale * wavenumber_scale;
    let i_omega_rho = Complex64::new(0.0, physics.pressure_factor);
    let i_omega_rho_derivative = Complex64::new(0.0, pressure_factor_derivative);
    let all_coords: Vec<DenseMatrix<f64>> = elements
        .iter()
        .map(|element| get_element_coords(element, nodes))
        .collect();

    for (iel, source_elem) in elements.iter().enumerate() {
        if source_elem.property.is_evaluation() {
            continue;
        }
        let source_dof = source_elem.dof_addresses[0];
        let (source_bc_type, source_bc_values) =
            get_bc_type_and_value(&source_elem.boundary_condition);
        let source_admittance = source_elem.boundary_condition.admittance();
        let source_admittance_derivative =
            source_admittance.map(|admittance| admittance * admittance_density_scale);
        let average_source_bc =
            source_bc_values.iter().sum::<Complex64>() / source_bc_values.len() as f64;
        let beta_term_derivative = i_omega_rho_derivative * beta + i_omega_rho * beta_derivative;
        match source_bc_type {
            0 => {
                if let Some(admittance) = source_admittance {
                    system.matrix[[source_dof, source_dof]] -=
                        sign * admittance * beta_term_derivative * tau * 0.5;
                }
                if let Some(admittance_derivative) = source_admittance_derivative {
                    system.matrix[[source_dof, source_dof]] -=
                        sign * admittance_derivative * i_omega_rho * beta * tau * 0.5;
                }
                system.rhs[source_dof] +=
                    sign * average_source_bc * beta_term_derivative * tau * 0.5;
            }
            1 => {
                system.matrix[[source_dof, source_dof]] += sign * beta_derivative * tau * 0.5;
            }
            _ => {}
        }

        for (jel, field_elem) in elements.iter().enumerate() {
            if field_elem.property.is_evaluation() {
                continue;
            }
            let field_dof = field_elem.dof_addresses[0];
            let (field_bc_type, field_bc_values) =
                get_bc_type_and_value(&field_elem.boundary_condition);
            let derivative = if jel == iel {
                singular_integration_wavenumber_derivative(
                    &source_elem.center,
                    &source_elem.normal,
                    &all_coords[jel],
                    field_elem.element_type,
                    physics,
                    None,
                    field_bc_type,
                    false,
                    None,
                )
            } else {
                regular_integration_wavenumber_derivative(
                    &source_elem.center,
                    &source_elem.normal,
                    &all_coords[jel],
                    field_elem.element_type,
                    field_elem.area,
                    physics,
                    None,
                    field_bc_type,
                    false,
                    None,
                )
            };
            let base = if jel == iel {
                singular_integration(
                    &source_elem.center,
                    &source_elem.normal,
                    &all_coords[jel],
                    field_elem.element_type,
                    physics,
                    None,
                    field_bc_type,
                    false,
                )
            } else {
                regular_integration(
                    &source_elem.center,
                    &source_elem.normal,
                    &all_coords[jel],
                    field_elem.element_type,
                    field_elem.area,
                    physics,
                    None,
                    field_bc_type,
                    false,
                )
            };
            let d_cb = derivative.dg_dn_integral * gamma * tau * wavenumber_scale
                + derivative.d2g_dnxdny_integral * beta * wavenumber_scale
                + base.d2g_dnxdny_integral * beta_derivative;
            let d_hb = derivative.g_integral * gamma * tau * wavenumber_scale
                + derivative.dg_dnx_integral * beta * wavenumber_scale
                + base.dg_dnx_integral * beta_derivative;
            let field_admittance = field_elem.boundary_condition.admittance();
            let field_admittance_derivative =
                field_admittance.map(|admittance| admittance * admittance_density_scale);
            let coefficient = match field_bc_type {
                0 => {
                    let mut value = sign * d_cb;
                    if let Some(admittance) = field_admittance {
                        let h_b = base.g_integral * gamma * tau + base.dg_dnx_integral * beta;
                        value -=
                            sign * admittance * (d_hb * i_omega_rho + h_b * i_omega_rho_derivative);
                        if let Some(admittance_derivative) = field_admittance_derivative {
                            value -= sign * admittance_derivative * h_b * i_omega_rho;
                        }
                    }
                    value
                }
                1 => -sign * d_hb,
                _ => Complex64::new(0.0, 0.0),
            };
            system.matrix[[source_dof, field_dof]] += coefficient;

            if has_nonzero_bc(&field_bc_values) {
                let average_field_bc =
                    field_bc_values.iter().sum::<Complex64>() / field_bc_values.len() as f64;
                let rhs_derivative = match field_bc_type {
                    0 => {
                        let h_b = base.g_integral * gamma * tau + base.dg_dnx_integral * beta;
                        sign * average_field_bc
                            * (d_hb * i_omega_rho + h_b * i_omega_rho_derivative)
                    }
                    1 => sign * average_field_bc * d_cb,
                    _ => Complex64::new(0.0, 0.0),
                };
                system.rhs[source_dof] += rhs_derivative;
            }
        }
    }
    system
}

/// Assemble only the right-hand side of the stabilized constant-panel TBEM
/// system without allocating its dense matrix.
///
/// This is intended for H-matrix and other matrix-free solve paths. It is
/// algebraically identical to the RHS produced by
/// [`build_tbem_system_stabilized`] while using O(N) memory.
pub fn build_tbem_rhs_stabilized(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
) -> Vec<Complex64> {
    let beta = physics.burton_miller_beta_scaled(beta_scale);
    build_tbem_rhs_stabilized_with_beta(elements, nodes, physics, sign, beta)
}

/// RHS-only stabilized TBEM assembly with an explicit Burton-Miller beta.
pub fn build_tbem_rhs_stabilized_with_beta(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    sign: f64,
    beta: Complex64,
) -> Vec<Complex64> {
    let num_dofs = count_dofs(elements);
    let gamma = Complex64::new(physics.gamma(), 0.0);
    let tau = Complex64::new(physics.tau, 0.0);
    let i_omega_rho = Complex64::new(0.0, physics.pressure_factor);
    let all_coords: Vec<DenseMatrix<f64>> = elements
        .iter()
        .map(|element| get_element_coords(element, nodes))
        .collect();

    // RHS row contributions (rayon above the machine-adaptive gate, serial
    // for small systems). Work is source × field kernel evaluations.
    let rows: Vec<(usize, Complex64)> = gated_enumerate_filter_map(
        elements,
        num_dofs.saturating_mul(elements.len()),
        BEM_QUADRATURE_PAIRS_PER_THREAD,
        |source_index, source_element| {
            if source_element.property.is_evaluation() {
                return None;
            }

            let source_dof = source_element.dof_addresses[0];
            let (source_bc_type, source_bc_values) =
                get_bc_type_and_value(&source_element.boundary_condition);
            let average_source_bc =
                source_bc_values.iter().sum::<Complex64>() / source_bc_values.len() as f64;
            let mut local_rhs = match source_bc_type {
                0 => average_source_bc * i_omega_rho * beta * tau * 0.5 * sign,
                1 => -average_source_bc * gamma * 0.5 * sign,
                _ => Complex64::new(0.0, 0.0),
            };

            for (field_index, field_element) in elements.iter().enumerate() {
                if field_element.property.is_evaluation() {
                    continue;
                }
                let (field_bc_type, field_bc_values) =
                    get_bc_type_and_value(&field_element.boundary_condition);
                if !has_nonzero_bc(&field_bc_values) {
                    continue;
                }
                let element_coords = &all_coords[field_index];
                let integration = if field_index == source_index {
                    singular_integration(
                        &source_element.center,
                        &source_element.normal,
                        element_coords,
                        field_element.element_type,
                        physics,
                        Some(&field_bc_values),
                        field_bc_type,
                        true,
                    )
                } else {
                    regular_integration(
                        &source_element.center,
                        &source_element.normal,
                        element_coords,
                        field_element.element_type,
                        field_element.area,
                        physics,
                        Some(&field_bc_values),
                        field_bc_type,
                        true,
                    )
                };
                local_rhs += stabilized_constant_rhs_contribution(
                    &integration,
                    field_bc_type,
                    &field_bc_values,
                    sign,
                    gamma,
                    tau,
                    beta,
                    i_omega_rho,
                );
            }
            Some((source_dof, local_rhs))
        },
    );

    let mut rhs = vec![Complex64::new(0.0, 0.0); num_dofs];
    for (dof, value) in rows {
        rhs[dof] += value;
    }
    rhs
}

/// Build TBEM system with stabilized formulation and explicit Burton-Miller coupling.
///
/// This is useful for interior problems that intentionally use a non-zero
/// Burton-Miller coupling even though `PhysicsParams::burton_miller_beta_scaled`
/// returns zero for the default interior formulation.
pub fn build_tbem_system_stabilized_with_beta(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    sign: f64,
    beta: Complex64,
) -> TbemSystem {
    use rayon::prelude::*;

    let num_dofs = count_dofs(elements);

    let gamma = Complex64::new(physics.gamma(), 0.0);
    let tau = Complex64::new(physics.tau, 0.0);
    let i_omega_rho = Complex64::new(0.0, physics.pressure_factor);

    let all_coords: Vec<DenseMatrix<f64>> = elements
        .iter()
        .map(|e| get_element_coords(e, nodes))
        .collect();

    // Fold per-source contributions (rayon above the machine-adaptive gate,
    // serial for small systems). The fold closure is shared by both paths so
    // the assembled system is identical.
    let fold_source = |mut sys: TbemSystem, (iel, source_elem): (usize, &Element)| {
        if source_elem.property.is_evaluation() {
            return sys;
        }

        let source_point = &source_elem.center;
        let source_normal = &source_elem.normal;
        let source_dof = source_elem.dof_addresses[0];

        let (bc_type, bc_value) = get_bc_type_and_value(&source_elem.boundary_condition);
        let bc_admittance = source_elem.boundary_condition.admittance();

        let mut local_row = vec![Complex64::new(0.0, 0.0); num_dofs];
        let mut local_rhs = Complex64::new(0.0, 0.0);

        let avg_bc = bc_value.iter().sum::<Complex64>() / bc_value.len() as f64;
        match bc_type {
            0 => {
                local_row[source_dof] -= sign * (gamma * 0.5);
                if let Some(admittance) = bc_admittance {
                    local_row[source_dof] -= sign * admittance * i_omega_rho * beta * tau * 0.5;
                }
                local_rhs += avg_bc * i_omega_rho * beta * tau * 0.5 * sign;
            }
            1 => {
                local_row[source_dof] += sign * (beta * tau * 0.5);
                local_rhs -= avg_bc * gamma * 0.5 * sign;
            }
            _ => {}
        }

        for (jel, field_elem) in elements.iter().enumerate() {
            if field_elem.property.is_evaluation() {
                continue;
            }

            let element_coords = &all_coords[jel];
            let field_dof = field_elem.dof_addresses[0];
            let (field_bc_type, field_bc_values) =
                get_bc_type_and_value(&field_elem.boundary_condition);
            let compute_rhs = has_nonzero_bc(&field_bc_values);

            let result = if jel == iel {
                singular_integration(
                    source_point,
                    source_normal,
                    element_coords,
                    field_elem.element_type,
                    physics,
                    if compute_rhs {
                        Some(&field_bc_values)
                    } else {
                        None
                    },
                    field_bc_type,
                    compute_rhs,
                )
            } else {
                regular_integration(
                    source_point,
                    source_normal,
                    element_coords,
                    field_elem.element_type,
                    field_elem.area,
                    physics,
                    if compute_rhs {
                        Some(&field_bc_values)
                    } else {
                        None
                    },
                    field_bc_type,
                    compute_rhs,
                )
            };

            let mut coeff = match field_bc_type {
                0 => {
                    sign * (result.dg_dn_integral * gamma * tau + result.d2g_dnxdny_integral * beta)
                }
                1 => -sign * (result.g_integral * gamma * tau + result.dg_dnx_integral * beta),
                _ => Complex64::new(0.0, 0.0),
            };
            if field_bc_type == 0
                && let Some(admittance) = field_elem.boundary_condition.admittance()
            {
                coeff -= sign
                    * (result.g_integral * gamma * tau + result.dg_dnx_integral * beta)
                    * admittance
                    * i_omega_rho;
            }

            local_row[field_dof] += coeff;

            if compute_rhs {
                local_rhs += stabilized_constant_rhs_contribution(
                    &result,
                    field_bc_type,
                    &field_bc_values,
                    sign,
                    gamma,
                    tau,
                    beta,
                    i_omega_rho,
                );
            }
        }

        let row = sys.matrix.row_mut(source_dof);
        for (j, v) in local_row.into_iter().enumerate() {
            row[j] += v;
        }
        sys.rhs[source_dof] += local_rhs;
        sys
    };
    if use_parallel(
        num_dofs.saturating_mul(elements.len()),
        BEM_QUADRATURE_PAIRS_PER_THREAD,
    ) {
        elements
            .par_iter()
            .enumerate()
            .fold(|| TbemSystem::new(num_dofs), &fold_source)
            .reduce(
                || TbemSystem::new(num_dofs),
                |mut a, b| {
                    a.merge(b);
                    a
                },
            )
    } else {
        elements
            .iter()
            .enumerate()
            .fold(TbemSystem::new(num_dofs), &fold_source)
    }
}

/// One evaluated row of a partial stabilized TBEM system.
pub struct TbemPartialRow {
    /// Row DOF (the row element's first DOF address).
    pub row_dof: usize,
    /// Matrix entries as `(column_dof, value)` pairs. Row-local jump terms
    /// (full-row evaluations only) appear as separate pairs, so a column DOF
    /// can occur twice; callers accumulate by summation.
    pub entries: Vec<(usize, Complex64)>,
    /// RHS value accumulated from the evaluated columns. Full-row
    /// evaluations also include the row-local jump term, making this the
    /// complete row RHS; column-restricted evaluations contain only the
    /// contributions tied to the requested columns.
    pub rhs: Complex64,
}

/// Evaluate selected rows of the stabilized constant-panel TBEM system.
///
/// Partial-evaluation counterpart of [`build_tbem_system_stabilized`]: each
/// requested row is assembled with the same quadrature rules and
/// Burton-Miller coupling, optionally restricted to a subset of field
/// (column) elements. Geometry-sensitivity codes use it to re-evaluate only
/// the matrix rows/columns touched by a shape perturbation instead of
/// re-assembling the full O(N²) operator.
///
/// * `rows` — indices of the collocation (row) elements to evaluate; they
///   must be boundary (non-evaluation) elements.
/// * `columns` — `None` evaluates complete rows (all field elements plus the
///   geometry-independent row-local jump terms); `Some(cols)` evaluates only
///   the listed field elements and skips the row-local terms, so the
///   difference of two restricted evaluations at different geometries is
///   exactly the geometry delta of those columns.
pub fn eval_tbem_rows_stabilized(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
    rows: &[usize],
    columns: Option<&[usize]>,
) -> Vec<TbemPartialRow> {
    let beta = physics.burton_miller_beta_scaled(beta_scale);
    let gamma = Complex64::new(physics.gamma(), 0.0);
    let tau = Complex64::new(physics.tau, 0.0);
    let i_omega_rho = Complex64::new(0.0, physics.pressure_factor);
    let full_rows = columns.is_none();

    // Precompute coordinates for every field element a row may touch.
    let field_indices: Vec<usize> = match columns {
        Some(cols) => cols.to_vec(),
        None => (0..elements.len()).collect(),
    };
    let field_coords: Vec<DenseMatrix<f64>> = field_indices
        .iter()
        .map(|&index| get_element_coords(&elements[index], nodes))
        .collect();

    // Requested-row evaluation (rayon above the machine-adaptive gate,
    // serial for small row × field counts). Work is rows × field kernel
    // evaluations.
    let row_work = rows.len().saturating_mul(field_indices.len());
    gated_map(
        rows,
        row_work,
        BEM_QUADRATURE_PAIRS_PER_THREAD,
        |&row_index| {
            let source_elem = &elements[row_index];
            debug_assert!(!source_elem.property.is_evaluation());
            let source_point = &source_elem.center;
            let source_normal = &source_elem.normal;
            let source_dof = source_elem.dof_addresses[0];

            let (bc_type, bc_value) = get_bc_type_and_value(&source_elem.boundary_condition);
            let bc_admittance = source_elem.boundary_condition.admittance();

            let mut entries = Vec::with_capacity(field_indices.len() + 1);
            let mut local_rhs = Complex64::new(0.0, 0.0);

            if full_rows {
                let avg_bc = bc_value.iter().sum::<Complex64>() / bc_value.len() as f64;
                let mut local_diag = Complex64::new(0.0, 0.0);
                match bc_type {
                    0 => {
                        local_diag -= sign * (gamma * 0.5);
                        if let Some(admittance) = bc_admittance {
                            local_diag -= sign * admittance * i_omega_rho * beta * tau * 0.5;
                        }
                        local_rhs += avg_bc * i_omega_rho * beta * tau * 0.5 * sign;
                    }
                    1 => {
                        local_diag += sign * (beta * tau * 0.5);
                        local_rhs -= avg_bc * gamma * 0.5 * sign;
                    }
                    _ => {}
                }
                entries.push((source_dof, local_diag));
            }

            for (&field_index, element_coords) in field_indices.iter().zip(&field_coords) {
                let field_elem = &elements[field_index];
                if field_elem.property.is_evaluation() {
                    continue;
                }
                let field_dof = field_elem.dof_addresses[0];
                let (field_bc_type, field_bc_values) =
                    get_bc_type_and_value(&field_elem.boundary_condition);
                let compute_rhs = has_nonzero_bc(&field_bc_values);

                let result = if field_index == row_index {
                    singular_integration(
                        source_point,
                        source_normal,
                        element_coords,
                        field_elem.element_type,
                        physics,
                        if compute_rhs {
                            Some(&field_bc_values)
                        } else {
                            None
                        },
                        field_bc_type,
                        compute_rhs,
                    )
                } else {
                    regular_integration(
                        source_point,
                        source_normal,
                        element_coords,
                        field_elem.element_type,
                        field_elem.area,
                        physics,
                        if compute_rhs {
                            Some(&field_bc_values)
                        } else {
                            None
                        },
                        field_bc_type,
                        compute_rhs,
                    )
                };

                let mut coeff = match field_bc_type {
                    0 => {
                        sign * (result.dg_dn_integral * gamma * tau
                            + result.d2g_dnxdny_integral * beta)
                    }
                    1 => -sign * (result.g_integral * gamma * tau + result.dg_dnx_integral * beta),
                    _ => Complex64::new(0.0, 0.0),
                };
                if field_bc_type == 0
                    && let Some(admittance) = field_elem.boundary_condition.admittance()
                {
                    coeff -= sign
                        * (result.g_integral * gamma * tau + result.dg_dnx_integral * beta)
                        * admittance
                        * i_omega_rho;
                }

                entries.push((field_dof, coeff));

                if compute_rhs {
                    local_rhs += stabilized_constant_rhs_contribution(
                        &result,
                        field_bc_type,
                        &field_bc_values,
                        sign,
                        gamma,
                        tau,
                        beta,
                        i_omega_rho,
                    );
                }
            }

            TbemPartialRow {
                row_dof: source_dof,
                entries,
                rhs: local_rhs,
            }
        },
    )
}

/// Stabilised constant-panel TBEM assembly with `sign`, routing the
/// far-field pair tail through
/// [`sonium_compute::bem_cpu::regular_pair_batch`] (NEON 2-pair SIMD on
/// aarch64, scalar elsewhere).
///
/// Numerically equivalent to [`build_tbem_system_stabilized_with_beta`] for
/// zero nonzero field BC values and pair distances safely outside the
/// quasi-singular regime. For pairs where either precondition fails, this
/// function falls back to scalar flat-panel integration. The far-field tail
/// uses the 4-pt Tri3 rule
/// (matching the GPU kernels) instead of the canonical 13-pt rule used by
/// `regular_integration` — the residual quadrature delta is well
/// below the iterative solver tolerance (1e-9 per matrix entry vs 1e-6 GMRES
/// target), but the assembly is not bit-equal so this is exposed as a
/// separate function rather than replacing the canonical reference (which
/// the MLFMM parity tests pin to bit-level equality).
pub fn build_tbem_system_stabilized_with_beta_simd(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    sign: f64,
    beta: Complex64,
) -> TbemSystem {
    use crate::core::integration::regular::QUASI_SINGULAR_THRESHOLD;
    use rayon::prelude::*;
    use sonium_compute::bem_cpu::{RegularPairInF64, regular_pair_batch};

    let num_dofs = count_dofs(elements);

    let gamma = Complex64::new(physics.gamma(), 0.0);
    let tau = Complex64::new(physics.tau, 0.0);
    let i_omega_rho = Complex64::new(0.0, physics.pressure_factor);

    let all_coords: Vec<DenseMatrix<f64>> = elements
        .iter()
        .map(|e| get_element_coords(e, nodes))
        .collect();

    // Fold per-source contributions (rayon above the machine-adaptive gate,
    // serial for small systems). The fold closure is shared by both paths so
    // the assembled system is identical.
    let fold_source = |mut sys: TbemSystem, (iel, source_elem): (usize, &Element)| {
        if source_elem.property.is_evaluation() {
            return sys;
        }

        let source_point = source_elem.center;
        let source_normal = source_elem.normal;
        let source_dof = source_elem.dof_addresses[0];

        let (bc_type, bc_value) = get_bc_type_and_value(&source_elem.boundary_condition);
        let bc_admittance = source_elem.boundary_condition.admittance();

        let mut local_row = vec![Complex64::new(0.0, 0.0); num_dofs];
        let mut local_rhs = Complex64::new(0.0, 0.0);

        // Diagonal free terms — identical to scalar path.
        let avg_bc = bc_value.iter().sum::<Complex64>() / bc_value.len() as f64;
        match bc_type {
            0 => {
                local_row[source_dof] -= sign * (gamma * 0.5);
                if let Some(admittance) = bc_admittance {
                    local_row[source_dof] -= sign * admittance * i_omega_rho * beta * tau * 0.5;
                }
                local_rhs += avg_bc * i_omega_rho * beta * tau * 0.5 * sign;
            }
            1 => {
                local_row[source_dof] += sign * (beta * tau * 0.5);
                local_rhs -= avg_bc * gamma * 0.5 * sign;
            }
            _ => {}
        }

        // Per-row metadata kept in lockstep with the batch input vec.
        struct BatchMeta {
            field_dof: usize,
            field_bc_type: i32,
            admittance: Option<Complex64>,
        }

        let mut batch_inputs: Vec<RegularPairInF64> = Vec::new();
        let mut batch_meta: Vec<BatchMeta> = Vec::new();

        for (jel, field_elem) in elements.iter().enumerate() {
            if field_elem.property.is_evaluation() {
                continue;
            }

            let element_coords = &all_coords[jel];
            let field_dof = field_elem.dof_addresses[0];
            let (field_bc_type, field_bc_values) =
                get_bc_type_and_value(&field_elem.boundary_condition);
            let compute_rhs = has_nonzero_bc(&field_bc_values);

            // SELF: singular branch stays scalar.
            if jel == iel {
                let result = singular_integration(
                    &source_point,
                    &source_normal,
                    element_coords,
                    field_elem.element_type,
                    physics,
                    if compute_rhs {
                        Some(&field_bc_values)
                    } else {
                        None
                    },
                    field_bc_type,
                    compute_rhs,
                );
                scatter_coeff(
                    &mut local_row,
                    &mut local_rhs,
                    field_dof,
                    field_bc_type,
                    field_elem.boundary_condition.admittance(),
                    &result,
                    sign,
                    gamma,
                    tau,
                    beta,
                    i_omega_rho,
                    &field_bc_values,
                );
                continue;
            }

            // Batch-eligibility: far-field Tri3 with no BC-driven RHS.
            let center = &field_elem.center;
            let dx = center[0] - source_point[0];
            let dy = center[1] - source_point[1];
            let dz = center[2] - source_point[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            let elem_size = field_elem.area.sqrt().max(1e-30);
            let ratio = dist / elem_size;
            let tri3 = matches!(
                field_elem.element_type,
                crate::core::types::ElementType::Tri3
            );
            let batch_ok = tri3 && !compute_rhs && ratio >= QUASI_SINGULAR_THRESHOLD;

            if !batch_ok {
                let result = regular_integration(
                    &source_point,
                    &source_normal,
                    element_coords,
                    field_elem.element_type,
                    field_elem.area,
                    physics,
                    if compute_rhs {
                        Some(&field_bc_values)
                    } else {
                        None
                    },
                    field_bc_type,
                    compute_rhs,
                );
                scatter_coeff(
                    &mut local_row,
                    &mut local_rhs,
                    field_dof,
                    field_bc_type,
                    field_elem.boundary_condition.admittance(),
                    &result,
                    sign,
                    gamma,
                    tau,
                    beta,
                    i_omega_rho,
                    &field_bc_values,
                );
                continue;
            }

            // Tri3 has 3 vertices at rows 0..3 of element_coords (3 × 3).
            debug_assert_eq!(element_coords.nrows(), 3);
            batch_inputs.push(RegularPairInF64 {
                source_point,
                source_normal,
                v0: [
                    element_coords[[0, 0]],
                    element_coords[[0, 1]],
                    element_coords[[0, 2]],
                ],
                v1: [
                    element_coords[[1, 0]],
                    element_coords[[1, 1]],
                    element_coords[[1, 2]],
                ],
                v2: [
                    element_coords[[2, 0]],
                    element_coords[[2, 1]],
                    element_coords[[2, 2]],
                ],
            });
            batch_meta.push(BatchMeta {
                field_dof,
                field_bc_type,
                admittance: field_elem.boundary_condition.admittance(),
            });
        }

        // Run the SIMD batch over the far-field flat tail for this source row.
        let batch_out =
            regular_pair_batch(&batch_inputs, physics.wave_number, physics.harmonic_factor);

        for (out, meta) in batch_out.iter().zip(batch_meta.iter()) {
            let (g, h, ht, e) = *out;
            // Coefficient: same stabilised formula as the scalar path. The
            // batch zeros rhs_contribution; pairs that need a nonzero RHS
            // contribution have already been routed scalar above.
            let mut coeff = match meta.field_bc_type {
                0 => sign * (h * gamma * tau + e * beta),
                1 => -sign * (g * gamma * tau + ht * beta),
                _ => Complex64::new(0.0, 0.0),
            };
            if meta.field_bc_type == 0
                && let Some(admittance) = meta.admittance
            {
                coeff -= sign * (g * gamma * tau + ht * beta) * admittance * i_omega_rho;
            }
            local_row[meta.field_dof] += coeff;
        }

        let row = sys.matrix.row_mut(source_dof);
        for (j, v) in local_row.into_iter().enumerate() {
            row[j] += v;
        }
        sys.rhs[source_dof] += local_rhs;
        sys
    };
    if use_parallel(
        num_dofs.saturating_mul(elements.len()),
        BEM_QUADRATURE_PAIRS_PER_THREAD,
    ) {
        elements
            .par_iter()
            .enumerate()
            .fold(|| TbemSystem::new(num_dofs), &fold_source)
            .reduce(
                || TbemSystem::new(num_dofs),
                |mut a, b| {
                    a.merge(b);
                    a
                },
            )
    } else {
        elements
            .iter()
            .enumerate()
            .fold(TbemSystem::new(num_dofs), &fold_source)
    }
}

/// Stabilised coefficient scatter shared between the scalar and the
/// non-batch fallback paths of
/// [`build_tbem_system_stabilized_with_beta_simd`].
#[inline]
#[allow(clippy::too_many_arguments)]
fn stabilized_constant_rhs_contribution(
    result: &crate::core::types::IntegrationResult,
    field_bc_type: i32,
    field_bc_values: &[Complex64],
    sign: f64,
    gamma: Complex64,
    tau: Complex64,
    beta: Complex64,
    i_omega_rho: Complex64,
) -> Complex64 {
    if !has_nonzero_bc(field_bc_values) {
        return Complex64::new(0.0, 0.0);
    }
    let average_bc = field_bc_values.iter().sum::<Complex64>() / field_bc_values.len() as f64;
    match field_bc_type {
        0 => {
            sign * (result.g_integral * gamma * tau + result.dg_dnx_integral * beta)
                * average_bc
                * i_omega_rho
        }
        1 => {
            sign * (result.dg_dn_integral * gamma * tau + result.d2g_dnxdny_integral * beta)
                * average_bc
        }
        _ => Complex64::new(0.0, 0.0),
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn scatter_coeff(
    local_row: &mut [Complex64],
    local_rhs: &mut Complex64,
    field_dof: usize,
    field_bc_type: i32,
    admittance: Option<Complex64>,
    result: &crate::core::types::IntegrationResult,
    sign: f64,
    gamma: Complex64,
    tau: Complex64,
    beta: Complex64,
    i_omega_rho: Complex64,
    field_bc_values: &[Complex64],
) {
    let mut coeff = match field_bc_type {
        0 => sign * (result.dg_dn_integral * gamma * tau + result.d2g_dnxdny_integral * beta),
        1 => -sign * (result.g_integral * gamma * tau + result.dg_dnx_integral * beta),
        _ => Complex64::new(0.0, 0.0),
    };
    if field_bc_type == 0
        && let Some(admittance) = admittance
    {
        coeff -= sign
            * (result.g_integral * gamma * tau + result.dg_dnx_integral * beta)
            * admittance
            * i_omega_rho;
    }
    local_row[field_dof] += coeff;
    *local_rhs += stabilized_constant_rhs_contribution(
        result,
        field_bc_type,
        field_bc_values,
        sign,
        gamma,
        tau,
        beta,
        i_omega_rho,
    );
}

/// Build TBEM system for linear elements with stabilized formulation and curved normals
fn build_tbem_system_discontinuous_stabilized(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
    basis: BasisOrder,
) -> TbemSystem {
    let num_dofs = count_dofs(elements);
    let mut system = TbemSystem::new(num_dofs);

    let gamma = Complex64::new(physics.gamma(), 0.0);
    let tau = Complex64::new(physics.tau, 0.0);
    let beta = physics.burton_miller_beta_scaled(beta_scale);
    let i_omega_rho = Complex64::new(0.0, physics.pressure_factor);

    let all_coords: Vec<DenseMatrix<f64>> = elements
        .iter()
        .map(|e| get_element_coords(e, nodes))
        .collect();

    for source_elem in elements.iter() {
        if source_elem.property.is_evaluation() {
            continue;
        }

        let num_source_dofs = source_elem.num_dofs_for_basis(basis);
        let collocation_points = source_elem.collocation_points(basis, nodes);
        // Linear pressure interpolation does not make an affine Tri3 panel
        // geometrically curved. Keep collocation normals consistent with the
        // flat mapping and the panel jump term.
        let collocation_normals = vec![source_elem.normal; num_source_dofs];

        let (bc_type, bc_value) = get_bc_type_and_value(&source_elem.boundary_condition);
        let bc_admittance = source_elem.boundary_condition.admittance();

        for c in 0..num_source_dofs {
            let source_point = &collocation_points[c];
            let source_normal = &collocation_normals[c];
            let source_dof = source_elem.dof_addresses[c];

            // Add diagonal free terms with sign
            let avg_bc = bc_value.iter().sum::<Complex64>() / bc_value.len() as f64;
            match bc_type {
                0 => {
                    system.matrix[[source_dof, source_dof]] -= sign * (gamma * 0.5);
                    if let Some(admittance) = bc_admittance {
                        system.matrix[[source_dof, source_dof]] -=
                            sign * admittance * i_omega_rho * beta * tau * 0.5;
                    }
                    // HBIE jump: velocity enters via ∂p/∂n = iωρ·v_n
                    system.rhs[source_dof] += avg_bc * i_omega_rho * beta * tau * 0.5 * sign;
                }
                1 => {
                    system.matrix[[source_dof, source_dof]] += sign * (beta * tau * 0.5);
                    system.rhs[source_dof] -= avg_bc * gamma * 0.5 * sign;
                }
                _ => {}
            }

            for (jel, field_elem) in elements.iter().enumerate() {
                if field_elem.property.is_evaluation() {
                    continue;
                }

                let element_coords = &all_coords[jel];
                let num_field_dofs = field_elem.num_dofs_for_basis(basis);
                let (field_bc_type, field_bc_values) =
                    get_bc_type_and_value(&field_elem.boundary_condition);
                let compute_rhs = has_nonzero_bc(&field_bc_values);

                // Field geometry is also affine Tri3; using interpolated
                // vertex normals here would describe a different surface.
                let curved_field = field_elem
                    .node_normals
                    .iter()
                    .any(|normal| normal.iter().any(|value| value.abs() > 1e-15));
                let nn = curved_field.then_some(field_elem.node_normals.as_slice());

                let is_self = std::ptr::eq(source_elem, field_elem);

                let linear_result = if is_self {
                    if curved_field {
                        singular_integration_linear_curved(
                            source_point,
                            source_normal,
                            element_coords,
                            field_elem.element_type,
                            physics,
                            num_field_dofs,
                            nn,
                        )
                    } else {
                        singular_integration_linear(
                            source_point,
                            source_normal,
                            element_coords,
                            field_elem.element_type,
                            physics,
                            num_field_dofs,
                        )
                    }
                } else if curved_field {
                    regular_integration_linear_with_normals(
                        source_point,
                        source_normal,
                        element_coords,
                        field_elem.element_type,
                        field_elem.area,
                        physics,
                        num_field_dofs,
                        nn,
                    )
                } else {
                    regular_integration_linear(
                        source_point,
                        source_normal,
                        element_coords,
                        field_elem.element_type,
                        field_elem.area,
                        physics,
                        num_field_dofs,
                    )
                };

                for b in 0..num_field_dofs {
                    let field_dof = field_elem.dof_addresses[b];
                    let result = &linear_result.basis_results[b];

                    let mut coeff = match field_bc_type {
                        0 => {
                            sign * (result.dg_dn_integral * gamma * tau
                                + result.d2g_dnxdny_integral * beta)
                        }
                        1 => {
                            -sign
                                * (result.g_integral * gamma * tau + result.dg_dnx_integral * beta)
                        }
                        _ => Complex64::new(0.0, 0.0),
                    };
                    if field_bc_type == 0
                        && let Some(admittance) = field_elem.boundary_condition.admittance()
                    {
                        coeff -= sign
                            * (result.g_integral * gamma * tau + result.dg_dnx_integral * beta)
                            * admittance
                            * i_omega_rho;
                    }

                    system.matrix[[source_dof, field_dof]] += coeff;

                    if compute_rhs {
                        let bc_val = field_bc_values
                            .get(b)
                            .copied()
                            .unwrap_or(Complex64::new(0.0, 0.0));
                        let rhs_contrib = match field_bc_type {
                            0 => {
                                // Velocity BC: ∂p/∂n = iωρ·v
                                sign * (result.g_integral * gamma * tau
                                    + result.dg_dnx_integral * beta)
                                    * bc_val
                                    * i_omega_rho
                            }
                            1 => {
                                -sign
                                    * (result.dg_dn_integral * gamma * tau
                                        + result.d2g_dnxdny_integral * beta)
                                    * bc_val
                            }
                            _ => Complex64::new(0.0, 0.0),
                        };
                        system.rhs[source_dof] += rhs_contrib;
                    }
                }
            }
        }
    }

    system
}

/// Build the stabilized discontinuous-linear TBEM system.
pub fn build_tbem_system_linear_stabilized(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
) -> TbemSystem {
    build_tbem_system_discontinuous_stabilized(
        elements,
        nodes,
        physics,
        sign,
        beta_scale,
        BasisOrder::Linear,
    )
}

/// Build the stabilized discontinuous-quadratic TBEM system on Tri3/Quad4
/// geometric panels.
pub fn build_tbem_system_quadratic_stabilized(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
) -> TbemSystem {
    build_tbem_system_discontinuous_stabilized(
        elements,
        nodes,
        physics,
        sign,
        beta_scale,
        BasisOrder::Quadratic,
    )
}

// ============================================================================
/// Build a CPU TBEM system for globally continuous quadratic Tri3 unknowns.
///
/// The six local quadratic basis functions are assembled into the shared
/// addresses supplied by [`ContinuousQuadraticDofMap`](crate::core::mesh::ContinuousQuadraticDofMap).
/// Collocation rows use the first panel/local basis slot that introduced each
/// global DOF. Those points are the existing interior quadratic collocation
/// points, so self-panel singular quadrature never collocates at a triangle
/// corner or edge. This is a continuous-unknown collocation slice, not a
/// Galerkin or NAFEMS-equivalence implementation.
pub fn build_tbem_system_continuous_quadratic_stabilized(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
) -> TbemSystem {
    let num_dofs = count_dofs(elements);
    let mut system = TbemSystem::new(num_dofs);
    if num_dofs == 0 {
        return system;
    }

    let all_coords: Vec<DenseMatrix<f64>> = elements
        .iter()
        .map(|element| get_element_coords(element, nodes))
        .collect();
    let mut owners: Vec<Option<(usize, usize)>> = vec![None; num_dofs];
    for (element_index, element) in elements.iter().enumerate() {
        if element.property.is_evaluation() {
            continue;
        }
        let collocation_points = element.collocation_points(BasisOrder::Quadratic, nodes);
        for (local, &dof) in element.dof_addresses.iter().enumerate() {
            if dof < num_dofs && local < collocation_points.len() {
                owners[dof].get_or_insert((element_index, local));
            }
        }
    }

    let gamma = Complex64::new(physics.gamma(), 0.0);
    let tau = Complex64::new(physics.tau, 0.0);
    let beta = physics.burton_miller_beta_scaled(beta_scale);
    let i_omega_rho = Complex64::new(0.0, physics.pressure_factor);

    for (source_dof, owner) in owners.into_iter().enumerate() {
        let Some((source_index, source_local)) = owner else {
            continue;
        };
        let source_element = &elements[source_index];
        let source_points = source_element.collocation_points(BasisOrder::Quadratic, nodes);
        let source_point = &source_points[source_local];
        // The continuous-Q2 collocation assembly uses the affine panel
        // Jacobian in its singular charts.  Keep the same panel normal for
        // the jump term and every interaction; interpolating generated
        // sphere vertex normals would mix curved normals with an affine
        // Jacobian and breaks refinement convergence.
        let source_normal = &source_element.normal;
        let (source_bc_type, source_bc_values) =
            get_bc_type_and_value(&source_element.boundary_condition);
        let source_admittance = source_element.boundary_condition.admittance();
        let avg_bc =
            source_bc_values.iter().sum::<Complex64>() / source_bc_values.len().max(1) as f64;

        match source_bc_type {
            0 => {
                system.matrix[[source_dof, source_dof]] -= sign * gamma * 0.5;
                if let Some(admittance) = source_admittance {
                    system.matrix[[source_dof, source_dof]] -=
                        sign * admittance * i_omega_rho * beta * tau * 0.5;
                }
                system.rhs[source_dof] += avg_bc * i_omega_rho * beta * tau * 0.5 * sign;
            }
            1 => {
                system.matrix[[source_dof, source_dof]] += sign * beta * tau * 0.5;
                system.rhs[source_dof] -= avg_bc * gamma * 0.5 * sign;
            }
            _ => {}
        }

        for (field_index, field_element) in elements.iter().enumerate() {
            if field_element.property.is_evaluation() {
                continue;
            }
            let (field_bc_type, field_bc_values) =
                get_bc_type_and_value(&field_element.boundary_condition);
            let compute_rhs = has_nonzero_bc(&field_bc_values);
            let linear_result = if field_index == source_index {
                singular_integration_linear(
                    source_point,
                    source_normal,
                    &all_coords[field_index],
                    field_element.element_type,
                    physics,
                    field_element.dof_addresses.len(),
                )
            } else {
                regular_integration_linear(
                    source_point,
                    source_normal,
                    &all_coords[field_index],
                    field_element.element_type,
                    field_element.area,
                    physics,
                    field_element.dof_addresses.len(),
                )
            };

            for (local_basis, &field_dof) in field_element.dof_addresses.iter().enumerate() {
                let result = &linear_result.basis_results[local_basis];
                let mut coeff = match field_bc_type {
                    0 => {
                        sign * (result.dg_dn_integral * gamma * tau
                            + result.d2g_dnxdny_integral * beta)
                    }
                    1 => -sign * (result.g_integral * gamma * tau + result.dg_dnx_integral * beta),
                    _ => Complex64::new(0.0, 0.0),
                };
                if field_bc_type == 0
                    && let Some(admittance) = field_element.boundary_condition.admittance()
                {
                    coeff -= sign
                        * (result.g_integral * gamma * tau + result.dg_dnx_integral * beta)
                        * admittance
                        * i_omega_rho;
                }
                system.matrix[[source_dof, field_dof]] += coeff;

                if compute_rhs {
                    let bc_value = field_bc_values
                        .get(local_basis)
                        .copied()
                        .or_else(|| field_bc_values.first().copied())
                        .unwrap_or_default();
                    let rhs_contribution = match field_bc_type {
                        0 => {
                            sign * (result.g_integral * gamma * tau + result.dg_dnx_integral * beta)
                                * bc_value
                                * i_omega_rho
                        }
                        1 => {
                            -sign
                                * (result.dg_dn_integral * gamma * tau
                                    + result.d2g_dnxdny_integral * beta)
                                * bc_value
                        }
                        _ => Complex64::new(0.0, 0.0),
                    };
                    system.rhs[source_dof] += rhs_contribution;
                }
            }
        }
    }
    system
}

// Metal-backed assembly path (Phase 1.3 of bem-step1.md)
// ============================================================================

#[cfg(feature = "metal")]
mod metal_path {
    use super::*;
    use crate::core::integration::regular::QUASI_SINGULAR_THRESHOLD;
    use crate::core::types::IntegrationResult;
    use sonium_compute::{BemRegularPairIn, BemRegularPairOut, MetalBackend};

    /// Result of pair classification for Metal dispatch routing.
    #[derive(Debug, Clone, Copy)]
    enum PairRoute {
        /// Source-element diagonal — routes to `singular_integration` on CPU.
        Self_,
        /// Near-field pair (`distance / sqrt(area) < QUASI_SINGULAR_THRESHOLD`)
        /// — needs adaptive subdivision and stays on CPU through
        /// `regular_integration`.
        CpuRegular,
        /// Far-field pair — handled by the GPU `bem_regular_pair_c32_batch`
        /// kernel under fixed 4-point Gauss-tri quadrature.
        Gpu(usize), // index into the flat GPU output vector
    }

    /// Build the dense BEM system matrix using the Metal backend for the
    /// `N²-N` far-field regular-pair tail. Self-element pairs and near-field
    /// pairs (where the source point sits inside the
    /// `QUASI_SINGULAR_THRESHOLD = 3.0` zone of the field element) stay on
    /// the existing CPU path with adaptive subdivision / Duffy quadrature.
    ///
    /// Functionally equivalent to `build_tbem_system_with_beta_parallel`
    /// when the GPU returns the same per-pair `(G, H, H^T, E)` integrals as
    /// the CPU `regular_integration` (verified by
    /// `crates/sonium-compute/tests/parity.rs`'s
    /// `metal_bem_regular_pair_matches_cpu` and the smoke test on this
    /// module). Returns the same `TbemSystem` shape.
    ///
    /// Constant basis only (matching the production `regular_integration`
    /// API consumed by the GPU kernel). Linear-basis assembly stays on the
    /// dedicated `regular_integration_linear` CPU paths.
    pub fn build_tbem_system_with_beta_metal(
        elements: &[Element],
        nodes: &[[f64; 3]],
        physics: &PhysicsParams,
        beta: Complex64,
        backend: &MetalBackend,
    ) -> TbemSystem {
        let num_dofs = count_dofs(elements);
        let mut system = TbemSystem::new(num_dofs);

        let gamma = Complex64::new(physics.gamma(), 0.0);
        let tau = Complex64::new(physics.tau, 0.0);
        let i_omega_rho = Complex64::new(0.0, physics.pressure_factor);

        let all_coords: Vec<DenseMatrix<f64>> = elements
            .iter()
            .map(|e| get_element_coords(e, nodes))
            .collect();

        let n_elem = elements.len();

        // Pre-classify every (iel, jel) pair. Building the GPU input vector
        // up front lets us launch one kernel for the entire far-field tail.
        let mut routes: Vec<Vec<PairRoute>> = vec![vec![PairRoute::Self_; n_elem]; n_elem];
        let mut gpu_pairs: Vec<BemRegularPairIn> = Vec::new();

        for (iel, src) in elements.iter().enumerate() {
            if src.property.is_evaluation() {
                continue;
            }
            let sp = &src.center;
            let sn = &src.normal;
            for (jel, fld) in elements.iter().enumerate() {
                if fld.property.is_evaluation() {
                    continue;
                }
                if iel == jel {
                    routes[iel][jel] = PairRoute::Self_;
                    continue;
                }
                let center = &fld.center;
                let dx = center[0] - sp[0];
                let dy = center[1] - sp[1];
                let dz = center[2] - sp[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                let elem_size = fld.area.sqrt().max(1e-30);
                let ratio = dist / elem_size;
                if ratio < QUASI_SINGULAR_THRESHOLD {
                    routes[iel][jel] = PairRoute::CpuRegular;
                } else {
                    let coords = &all_coords[jel];
                    debug_assert_eq!(coords.nrows(), 3, "Tri3 only on the GPU path");
                    let pair = BemRegularPairIn::from_triangle(
                        [sp[0] as f32, sp[1] as f32, sp[2] as f32],
                        [sn[0] as f32, sn[1] as f32, sn[2] as f32],
                        [
                            coords[[0, 0]] as f32,
                            coords[[0, 1]] as f32,
                            coords[[0, 2]] as f32,
                        ],
                        [
                            coords[[1, 0]] as f32,
                            coords[[1, 1]] as f32,
                            coords[[1, 2]] as f32,
                        ],
                        [
                            coords[[2, 0]] as f32,
                            coords[[2, 1]] as f32,
                            coords[[2, 2]] as f32,
                        ],
                    );
                    let gpu_idx = gpu_pairs.len();
                    gpu_pairs.push(pair);
                    routes[iel][jel] = PairRoute::Gpu(gpu_idx);
                }
            }
        }

        // Dispatch the GPU batch (one kernel call for the entire far-field
        // tail). Empty input is a no-op.
        let gpu_out: Vec<BemRegularPairOut> = backend.bem_regular_pair_c32_batch(
            &gpu_pairs,
            physics.wave_number as f32,
            physics.harmonic_factor as f32,
        );

        // Scatter results into the system matrix with BC weighting. We
        // reuse the per-pair logic of `build_tbem_system_with_beta_parallel`
        // verbatim — only the `IntegrationResult` source changes (GPU vs
        // CPU `regular_integration` vs CPU `singular_integration`).
        for (iel, src) in elements.iter().enumerate() {
            if src.property.is_evaluation() {
                continue;
            }
            let source_point = &src.center;
            let source_normal = &src.normal;
            let source_dof = src.dof_addresses[0];
            let (bc_type, bc_value) = get_bc_type_and_value(&src.boundary_condition);

            // Free terms — same as CPU parallel path.
            let avg_bc = bc_value.iter().sum::<Complex64>() / bc_value.len() as f64;
            match bc_type {
                0 => {
                    system.matrix[[source_dof, source_dof]] -= gamma * 0.5;
                    if let Some(admittance) = src.boundary_condition.admittance() {
                        system.matrix[[source_dof, source_dof]] -=
                            admittance * i_omega_rho * beta * tau * 0.5;
                    }
                    system.rhs[source_dof] += avg_bc * i_omega_rho * beta * tau * 0.5;
                }
                1 => {
                    system.matrix[[source_dof, source_dof]] += beta * tau * 0.5;
                    system.rhs[source_dof] -= avg_bc * gamma * 0.5;
                }
                _ => {}
            }

            for (jel, fld) in elements.iter().enumerate() {
                if fld.property.is_evaluation() {
                    continue;
                }
                let element_coords = &all_coords[jel];
                let field_dof = fld.dof_addresses[0];
                let (field_bc_type, field_bc_values) =
                    get_bc_type_and_value(&fld.boundary_condition);
                let compute_rhs = has_nonzero_bc(&field_bc_values);
                let bc_for_call = compute_rhs.then_some(field_bc_values.as_slice());

                let result = match routes[iel][jel] {
                    PairRoute::Self_ => crate::core::integration::singular_integration(
                        source_point,
                        source_normal,
                        element_coords,
                        fld.element_type,
                        physics,
                        bc_for_call,
                        field_bc_type,
                        compute_rhs,
                    ),
                    PairRoute::CpuRegular => crate::core::integration::regular_integration(
                        source_point,
                        source_normal,
                        element_coords,
                        fld.element_type,
                        fld.area,
                        physics,
                        bc_for_call,
                        field_bc_type,
                        compute_rhs,
                    ),
                    PairRoute::Gpu(gpu_idx) => {
                        let g = &gpu_out[gpu_idx];
                        IntegrationResult {
                            g_integral: Complex64::new(g.g.re as f64, g.g.im as f64),
                            dg_dn_integral: Complex64::new(g.h.re as f64, g.h.im as f64),
                            dg_dnx_integral: Complex64::new(g.ht.re as f64, g.ht.im as f64),
                            d2g_dnxdny_integral: Complex64::new(g.e.re as f64, g.e.im as f64),
                            rhs_contribution: Complex64::new(0.0, 0.0),
                        }
                    }
                };

                let mut coeff = match field_bc_type {
                    0 => result.dg_dn_integral * gamma * tau + result.d2g_dnxdny_integral * beta,
                    1 => -(result.g_integral * gamma * tau + result.dg_dnx_integral * beta),
                    _ => Complex64::new(0.0, 0.0),
                };
                if field_bc_type == 0
                    && let Some(admittance) = fld.boundary_condition.admittance()
                {
                    coeff -= (result.g_integral * gamma * tau + result.dg_dnx_integral * beta)
                        * admittance
                        * i_omega_rho;
                }
                system.matrix[[source_dof, field_dof]] += coeff;

                if compute_rhs {
                    let avg_field_bc =
                        field_bc_values.iter().sum::<Complex64>() / field_bc_values.len() as f64;
                    let rhs_contrib = match field_bc_type {
                        0 => {
                            (result.g_integral * gamma * tau + result.dg_dnx_integral * beta)
                                * avg_field_bc
                                * i_omega_rho
                        }
                        1 => {
                            (result.dg_dn_integral * gamma * tau
                                + result.d2g_dnxdny_integral * beta)
                                * avg_field_bc
                        }
                        _ => Complex64::new(0.0, 0.0),
                    };
                    system.rhs[source_dof] += rhs_contrib;
                }
            }
        }

        system
    }

    /// Stabilised constant-panel TBEM system assembly with `sign`
    /// (interior/exterior), routing the far-field regular-pair tail
    /// through the Metal `bem_regular_pair_c32_batch` kernel.
    ///
    /// Mirrors `super::build_tbem_system_stabilized_with_beta` (the CPU
    /// `par_iter` version) but classifies pairs the way
    /// `build_tbem_system_with_beta_metal` does, then assembles the matrix
    /// from the GPU output. Self elements and near-field pairs (within
    /// `QUASI_SINGULAR_THRESHOLD`) stay on the CPU through flat-panel
    /// singular/regular integration, matching the stabilised CPU path.
    ///
    /// f32 Metal output is promoted to `Complex64` on readback; downstream
    /// BLR factor / GMRES run at their declared f64 precision.
    pub fn build_tbem_system_stabilized_with_beta_metal(
        elements: &[Element],
        nodes: &[[f64; 3]],
        physics: &PhysicsParams,
        sign: f64,
        beta: Complex64,
        backend: &MetalBackend,
    ) -> TbemSystem {
        use rayon::prelude::*;

        let num_dofs = count_dofs(elements);

        let gamma = Complex64::new(physics.gamma(), 0.0);
        let tau = Complex64::new(physics.tau, 0.0);
        let i_omega_rho = Complex64::new(0.0, physics.pressure_factor);

        let all_coords: Vec<DenseMatrix<f64>> = elements
            .iter()
            .map(|e| get_element_coords(e, nodes))
            .collect();

        let n_elem = elements.len();

        // Pre-classify every (iel, jel) pair and build the GPU input batch.
        // Same classification rule as `build_tbem_system_with_beta_metal`.
        let mut routes: Vec<Vec<PairRoute>> = vec![vec![PairRoute::Self_; n_elem]; n_elem];
        let mut gpu_pairs: Vec<BemRegularPairIn> = Vec::new();

        for (iel, src) in elements.iter().enumerate() {
            if src.property.is_evaluation() {
                continue;
            }
            let src_normal = src.normal;
            let sp = &src.center;
            for (jel, fld) in elements.iter().enumerate() {
                if fld.property.is_evaluation() {
                    continue;
                }
                if iel == jel {
                    routes[iel][jel] = PairRoute::Self_;
                    continue;
                }
                let center = &fld.center;
                let dx = center[0] - sp[0];
                let dy = center[1] - sp[1];
                let dz = center[2] - sp[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                let elem_size = fld.area.sqrt().max(1e-30);
                let ratio = dist / elem_size;
                if ratio < QUASI_SINGULAR_THRESHOLD {
                    routes[iel][jel] = PairRoute::CpuRegular;
                } else {
                    let coords = &all_coords[jel];
                    debug_assert_eq!(coords.nrows(), 3, "Tri3 only on the GPU path");
                    let pair = BemRegularPairIn::from_triangle(
                        [sp[0] as f32, sp[1] as f32, sp[2] as f32],
                        [
                            src_normal[0] as f32,
                            src_normal[1] as f32,
                            src_normal[2] as f32,
                        ],
                        [
                            coords[[0, 0]] as f32,
                            coords[[0, 1]] as f32,
                            coords[[0, 2]] as f32,
                        ],
                        [
                            coords[[1, 0]] as f32,
                            coords[[1, 1]] as f32,
                            coords[[1, 2]] as f32,
                        ],
                        [
                            coords[[2, 0]] as f32,
                            coords[[2, 1]] as f32,
                            coords[[2, 2]] as f32,
                        ],
                    );
                    let gpu_idx = gpu_pairs.len();
                    gpu_pairs.push(pair);
                    routes[iel][jel] = PairRoute::Gpu(gpu_idx);
                }
            }
        }

        let gpu_out: Vec<BemRegularPairOut> = backend.bem_regular_pair_c32_batch(
            &gpu_pairs,
            physics.wave_number as f32,
            physics.harmonic_factor as f32,
        );

        // Parallel scatter using the stabilised coefficient formulas
        // (sign-aware and flat-panel consistent). Each non-evaluation source
        // element owns a unique row under constant basis, so the per-row
        // buffer + single-mutex pattern is race-free and matches the CPU
        // `build_tbem_system_stabilized_with_beta`.
        // Fold per-source contributions (rayon above the machine-adaptive
        // gate, serial for small systems). The fold closure is shared by
        // both paths so the assembled system is identical.
        let fold_source = |mut sys: TbemSystem, (iel, src): (usize, &Element)| {
            if src.property.is_evaluation() {
                return sys;
            }
            let source_point = &src.center;
            let source_normal = src.normal;
            let source_dof = src.dof_addresses[0];
            let (bc_type, bc_value) = get_bc_type_and_value(&src.boundary_condition);
            let bc_admittance = src.boundary_condition.admittance();

            let mut local_row = vec![Complex64::new(0.0, 0.0); num_dofs];
            let mut local_rhs = Complex64::new(0.0, 0.0);

            let avg_bc = bc_value.iter().sum::<Complex64>() / bc_value.len() as f64;
            match bc_type {
                0 => {
                    local_row[source_dof] -= sign * (gamma * 0.5);
                    if let Some(admittance) = bc_admittance {
                        local_row[source_dof] -= sign * admittance * i_omega_rho * beta * tau * 0.5;
                    }
                    local_rhs += avg_bc * i_omega_rho * beta * tau * 0.5 * sign;
                }
                1 => {
                    local_row[source_dof] += sign * (beta * tau * 0.5);
                    local_rhs -= avg_bc * gamma * 0.5 * sign;
                }
                _ => {}
            }

            for (jel, fld) in elements.iter().enumerate() {
                if fld.property.is_evaluation() {
                    continue;
                }
                let element_coords = &all_coords[jel];
                let field_dof = fld.dof_addresses[0];
                let (field_bc_type, field_bc_values) =
                    get_bc_type_and_value(&fld.boundary_condition);
                let compute_rhs = has_nonzero_bc(&field_bc_values);
                let bc_for_call = compute_rhs.then_some(field_bc_values.as_slice());

                let result = match routes[iel][jel] {
                    PairRoute::Self_ => crate::core::integration::singular_integration(
                        source_point,
                        &source_normal,
                        element_coords,
                        fld.element_type,
                        physics,
                        bc_for_call,
                        field_bc_type,
                        compute_rhs,
                    ),
                    PairRoute::CpuRegular => crate::core::integration::regular_integration(
                        source_point,
                        &source_normal,
                        element_coords,
                        fld.element_type,
                        fld.area,
                        physics,
                        bc_for_call,
                        field_bc_type,
                        compute_rhs,
                    ),
                    PairRoute::Gpu(gpu_idx) => {
                        let g = &gpu_out[gpu_idx];
                        IntegrationResult {
                            g_integral: Complex64::new(g.g.re as f64, g.g.im as f64),
                            dg_dn_integral: Complex64::new(g.h.re as f64, g.h.im as f64),
                            dg_dnx_integral: Complex64::new(g.ht.re as f64, g.ht.im as f64),
                            d2g_dnxdny_integral: Complex64::new(g.e.re as f64, g.e.im as f64),
                            rhs_contribution: Complex64::new(0.0, 0.0),
                        }
                    }
                };

                let mut coeff = match field_bc_type {
                    0 => {
                        sign * (result.dg_dn_integral * gamma * tau
                            + result.d2g_dnxdny_integral * beta)
                    }
                    1 => -sign * (result.g_integral * gamma * tau + result.dg_dnx_integral * beta),
                    _ => Complex64::new(0.0, 0.0),
                };
                if field_bc_type == 0
                    && let Some(admittance) = fld.boundary_condition.admittance()
                {
                    coeff -= sign
                        * (result.g_integral * gamma * tau + result.dg_dnx_integral * beta)
                        * admittance
                        * i_omega_rho;
                }
                local_row[field_dof] += coeff;

                if compute_rhs {
                    local_rhs += stabilized_constant_rhs_contribution(
                        &result,
                        field_bc_type,
                        &field_bc_values,
                        sign,
                        gamma,
                        tau,
                        beta,
                        i_omega_rho,
                    );
                }
            }

            let row = sys.matrix.row_mut(source_dof);
            for (j, v) in local_row.into_iter().enumerate() {
                row[j] += v;
            }
            sys.rhs[source_dof] += local_rhs;
            sys
        };
        if use_parallel(
            num_dofs.saturating_mul(elements.len()),
            BEM_QUADRATURE_PAIRS_PER_THREAD,
        ) {
            elements
                .par_iter()
                .enumerate()
                .fold(|| TbemSystem::new(num_dofs), &fold_source)
                .reduce(
                    || TbemSystem::new(num_dofs),
                    |mut a, b| {
                        a.merge(b);
                        a
                    },
                )
        } else {
            elements
                .iter()
                .enumerate()
                .fold(TbemSystem::new(num_dofs), &fold_source)
        }
    }
}

#[cfg(feature = "metal")]
pub use metal_path::{
    build_tbem_system_stabilized_with_beta_metal, build_tbem_system_with_beta_metal,
};

#[cfg(feature = "cuda")]
mod cuda_path {
    use super::*;
    use crate::core::integration::gauss::{gauss_legendre, quad_quadrature, triangle_quadrature};
    use crate::core::integration::regular::QUASI_SINGULAR_THRESHOLD;
    use crate::core::integration::singular::{QuadratureParams, generate_subelements};
    use crate::core::mesh::element::{basis_values, cross_product};
    use crate::core::types::IntegrationResult;
    use sonium_compute::{
        Backend, BackendError, BemQuadraturePairC64, BemQuadratureSampleC64, BemRegularPairInF64,
        BemRegularPairOutF64, BemRegularPairOutRealF64, CudaBackend, CudaBuffer, DeviceBufferLike,
    };

    mod device {
        include!("tbem_device.rs");
    }
    pub use device::{CudaTbemGmresSession, CudaTbemDeviceSystem, CudaTbemSolveError, CudaTbemSolveReport, CudaTbemSolveSession};

    /// Typed failures from the CUDA constant-panel assembly seam.
    #[derive(Debug, thiserror::Error)]
    pub enum CudaTbemAssemblyError {
        /// The checked CUDA pair kernel currently accepts affine Tri3 panels
        /// with one constant DOF per element.
        #[error(
            "CUDA TBEM regular-pair assembly supports Tri3 constant panels only; element {element} is {element_type} with {dofs} DOFs"
        )]
        UnsupportedElement {
            /// Element index in the supplied mesh.
            element: usize,
            /// Element type name.
            element_type: &'static str,
            /// Number of local DOF addresses.
            dofs: usize,
        },
        /// The mesh, physics, or coupling parameter violates the checked
        /// CUDA assembly boundary.
        #[error("invalid CUDA TBEM input: {reason}")]
        InvalidInput {
            /// Human-readable validation reason.
            reason: String,
        },
        /// Backend validation, allocation, launch, transfer, or
        /// synchronization failed before the system was published.
        #[error("CUDA TBEM backend failure: {0}")]
        Backend(#[from] BackendError),
    }

    #[derive(Debug, Clone, Copy)]
    enum PairRoute {
        Self_,
        Gpu(usize),
        GpuQuadrature(usize),
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct CudaTbemPhysicsKey {
        wave_number: f64,
        harmonic_factor: f64,
        pressure_factor: f64,
        gamma: f64,
        tau: f64,
        beta: Complex64,
    }

    fn cuda_tbem_physics_key(physics: &PhysicsParams, beta: Complex64) -> CudaTbemPhysicsKey {
        CudaTbemPhysicsKey {
            wave_number: physics.wave_number,
            harmonic_factor: physics.harmonic_factor,
            pressure_factor: physics.pressure_factor,
            gamma: physics.gamma(),
            tau: physics.tau,
            beta,
        }
    }

    /// Device-resident geometry for repeated CUDA TBEM frequency assembly.
    ///
    /// The mesh topology, pair routes, regular-pair descriptors, and
    /// quadrature charts are uploaded once.  Frequency-dependent kernel
    /// parameters and boundary values remain explicit inputs to
    /// [`Self::assemble`].  Each singular chart uses its field panel's
    /// production `for_ka` policy at construction. Callers must rebuild the
    /// workspace when a frequency step changes that quadrature schedule.
    pub struct CudaTbemAssemblyWorkspace {
        device_system: Option<CudaTbemDeviceSystem>,
        quadrature_schedule: Vec<[usize; 4]>,
        geometry_key: u64,
        all_coords: Vec<DenseMatrix<f64>>,
        routes: Vec<Vec<PairRoute>>,
        regular_pairs: CudaBuffer<BemRegularPairInF64>,
        far_pair_channels: CudaBuffer<BemRegularPairInF64>,
        retains_far_channels: bool,
        quadrature_pairs: CudaBuffer<BemQuadraturePairC64>,
        quadrature_samples: CudaBuffer<BemQuadratureSampleC64>,
        regular_output: CudaBuffer<BemRegularPairOutF64>,
        regular_real_output: CudaBuffer<BemRegularPairOutRealF64>,
        quadrature_output: CudaBuffer<BemRegularPairOutF64>,
        regular_derivative_output: CudaBuffer<BemRegularPairOutF64>,
        regular_source_x_derivative_output: CudaBuffer<BemRegularPairOutF64>,
        regular_source_y_derivative_output: CudaBuffer<BemRegularPairOutF64>,
        quadrature_derivative_output: CudaBuffer<BemRegularPairOutF64>,
        last_regular_output: Vec<BemRegularPairOutF64>,
        last_quadrature_output: Vec<BemRegularPairOutF64>,
        last_physics_key: Option<CudaTbemPhysicsKey>,
    }

    impl CudaTbemAssemblyWorkspace {
        /// Build and upload all frequency-invariant CUDA TBEM descriptors.
        pub fn new(
            elements: &[Element],
            nodes: &[[f64; 3]],
            physics: &PhysicsParams,
            backend: &CudaBackend,
        ) -> Result<Self, CudaTbemAssemblyError> {
            Self::build(elements, nodes, physics, backend, false)
        }

        /// Build resident far-pair channels in addition to Helmholtz quadrature charts.
        ///
        /// Use this constructor for the four-point real Laplace and source-position
        /// derivative methods. Normal Helmholtz assembly avoids these extra buffers
        /// and evaluates far panels with the CPU's higher-order quadrature rule.
        ///
        /// # Errors
        /// Returns an error for unsupported panels, invalid input, or CUDA
        /// allocation and transfer failures before publishing the workspace.
        pub fn new_with_regular_pair_channels(
            elements: &[Element],
            nodes: &[[f64; 3]],
            physics: &PhysicsParams,
            backend: &CudaBackend,
        ) -> Result<Self, CudaTbemAssemblyError> {
            Self::build(elements, nodes, physics, backend, true)
        }

        fn build(
            elements: &[Element],
            nodes: &[[f64; 3]],
            physics: &PhysicsParams,
            backend: &CudaBackend,
            retain_far_channels: bool,
        ) -> Result<Self, CudaTbemAssemblyError> {
            validate_cuda_tbem_inputs(elements, nodes, physics, Complex64::default())?;
            let all_coords: Vec<DenseMatrix<f64>> = elements
                .iter()
                .map(|element| get_element_coords(element, nodes))
                .collect();
            let n_elem = elements.len();
            let mut routes = vec![vec![PairRoute::Self_; n_elem]; n_elem];
            let mut regular_pairs = Vec::new();
            let mut far_pair_channels = Vec::new();
            let mut quadrature_pairs = Vec::new();
            let mut quadrature_samples = Vec::new();
            // The CPU minimum far rule is order four (13 points), whereas
            // the compute regular-pair primitive deliberately uses four points.
            // Share the CPU chart once per field panel across far source pairs.
            let mut far_chart_ranges: Vec<Option<(u64, u64)>> = vec![None; n_elem];
            for (iel, src) in elements.iter().enumerate() {
                if src.property.is_evaluation() {
                    continue;
                }
                let source_point = src.center;
                let source_normal = src.normal;
                for (jel, fld) in elements.iter().enumerate() {
                    if fld.property.is_evaluation() {
                        continue;
                    }
                    if iel == jel {
                        let sample_offset =
                            u64::try_from(quadrature_samples.len()).map_err(|_| {
                                CudaTbemAssemblyError::InvalidInput {
                                    reason: "CUDA quadrature sample count exceeds u64".into(),
                                }
                            })?;
                        append_singular_tri3_quadrature_with_params(
                            &all_coords[jel],
                            QuadratureParams::for_ka(
                                physics.wave_number * fld.area.sqrt().max(1.0e-10),
                            ),
                            &mut quadrature_samples,
                        )?;
                        let sample_count = u64::try_from(
                            quadrature_samples
                                .len()
                                .checked_sub(sample_offset as usize)
                                .ok_or_else(|| CudaTbemAssemblyError::InvalidInput {
                                    reason: "CUDA singular quadrature sample range underflowed"
                                        .into(),
                                })?,
                        )
                        .map_err(|_| {
                            CudaTbemAssemblyError::InvalidInput {
                                reason: "CUDA quadrature sample count exceeds u64".into(),
                            }
                        })?;
                        let pair_index = quadrature_pairs.len();
                        quadrature_pairs.push(BemQuadraturePairC64::new(
                            source_point,
                            source_normal,
                            sample_offset,
                            sample_count,
                        ));
                        routes[iel][jel] = PairRoute::GpuQuadrature(pair_index);
                        continue;
                    }
                    let dx = fld.center[0] - source_point[0];
                    let dy = fld.center[1] - source_point[1];
                    let dz = fld.center[2] - source_point[2];
                    let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                    let ratio = distance / fld.area.sqrt().max(1.0e-30);
                    if retain_far_channels && ratio >= QUASI_SINGULAR_THRESHOLD {
                        let coords = &all_coords[jel];
                        far_pair_channels.push(BemRegularPairInF64::from_triangle(
                            source_point,
                            source_normal,
                            [coords[[0, 0]], coords[[0, 1]], coords[[0, 2]]],
                            [coords[[1, 0]], coords[[1, 1]], coords[[1, 2]]],
                            [coords[[2, 0]], coords[[2, 1]], coords[[2, 2]]],
                        ));
                    }
                    if ratio.is_finite() {
                        let cached = (ratio >= QUASI_SINGULAR_THRESHOLD)
                            .then_some(far_chart_ranges[jel])
                            .flatten();
                        let (sample_offset, sample_count) = if let Some(range) = cached {
                            range
                        } else {
                            let start = quadrature_samples.len();
                            append_regular_tri3_quadrature(
                                source_point,
                                &all_coords[jel],
                                fld.area,
                                &mut quadrature_samples,
                            )?;
                            let range = (
                                u64::try_from(start).map_err(|_| {
                                    CudaTbemAssemblyError::InvalidInput {
                                        reason: "CUDA quadrature offset exceeds u64".into(),
                                    }
                                })?,
                                u64::try_from(quadrature_samples.len() - start).map_err(|_| {
                                    CudaTbemAssemblyError::InvalidInput {
                                        reason: "CUDA quadrature sample count exceeds u64".into(),
                                    }
                                })?,
                            );
                            if ratio >= QUASI_SINGULAR_THRESHOLD {
                                far_chart_ranges[jel] = Some(range);
                            }
                            range
                        };
                        let pair_index = quadrature_pairs.len();
                        quadrature_pairs.push(BemQuadraturePairC64::new(
                            source_point,
                            source_normal,
                            sample_offset,
                            sample_count,
                        ));
                        routes[iel][jel] = PairRoute::GpuQuadrature(pair_index);
                    } else {
                        let coords = &all_coords[jel];
                        let pair_index = regular_pairs.len();
                        regular_pairs.push(BemRegularPairInF64::from_triangle(
                            source_point,
                            source_normal,
                            [coords[[0, 0]], coords[[0, 1]], coords[[0, 2]]],
                            [coords[[1, 0]], coords[[1, 1]], coords[[1, 2]]],
                            [coords[[2, 0]], coords[[2, 1]], coords[[2, 2]]],
                        ));
                        routes[iel][jel] = PairRoute::Gpu(pair_index);
                    }
                }
            }

            let mut regular_pairs_device = backend
                .raw_alloc::<BemRegularPairInF64>(regular_pairs.len())
                .map_err(BackendError::from)
                .map_err(CudaTbemAssemblyError::Backend)?;
            if !regular_pairs.is_empty() {
                backend.try_copy_h2d(&regular_pairs, &mut regular_pairs_device)?;
            }
            let mut far_pair_channels_device = backend
                .raw_alloc::<BemRegularPairInF64>(far_pair_channels.len())
                .map_err(BackendError::from)
                .map_err(CudaTbemAssemblyError::Backend)?;
            if !far_pair_channels.is_empty() {
                backend.try_copy_h2d(&far_pair_channels, &mut far_pair_channels_device)?;
            }
            let mut quadrature_pairs_device = backend
                .raw_alloc::<BemQuadraturePairC64>(quadrature_pairs.len())
                .map_err(BackendError::from)
                .map_err(CudaTbemAssemblyError::Backend)?;
            if !quadrature_pairs.is_empty() {
                backend.try_copy_h2d(&quadrature_pairs, &mut quadrature_pairs_device)?;
            }
            let mut quadrature_samples_device = backend
                .raw_alloc::<BemQuadratureSampleC64>(quadrature_samples.len())
                .map_err(BackendError::from)
                .map_err(CudaTbemAssemblyError::Backend)?;
            if !quadrature_samples.is_empty() {
                backend.try_copy_h2d(&quadrature_samples, &mut quadrature_samples_device)?;
            }
            let regular_output = backend
                .raw_alloc::<BemRegularPairOutF64>(regular_pairs.len())
                .map_err(BackendError::from)
                .map_err(CudaTbemAssemblyError::Backend)?;
            let regular_real_output = backend
                .raw_alloc::<BemRegularPairOutRealF64>(far_pair_channels.len())
                .map_err(BackendError::from)
                .map_err(CudaTbemAssemblyError::Backend)?;
            let quadrature_output = backend
                .raw_alloc::<BemRegularPairOutF64>(quadrature_pairs.len())
                .map_err(BackendError::from)
                .map_err(CudaTbemAssemblyError::Backend)?;
            let regular_derivative_output = backend
                .raw_alloc::<BemRegularPairOutF64>(regular_pairs.len())
                .map_err(BackendError::from)
                .map_err(CudaTbemAssemblyError::Backend)?;
            let regular_source_x_derivative_output = backend
                .raw_alloc::<BemRegularPairOutF64>(far_pair_channels.len())
                .map_err(BackendError::from)
                .map_err(CudaTbemAssemblyError::Backend)?;
            let regular_source_y_derivative_output = backend
                .raw_alloc::<BemRegularPairOutF64>(far_pair_channels.len())
                .map_err(BackendError::from)
                .map_err(CudaTbemAssemblyError::Backend)?;
            let quadrature_derivative_output = backend
                .raw_alloc::<BemRegularPairOutF64>(quadrature_pairs.len())
                .map_err(BackendError::from)
                .map_err(CudaTbemAssemblyError::Backend)?;

            Ok(Self {
                device_system: None,
                quadrature_schedule: device::quadrature_schedule(elements, physics),
                geometry_key: cuda_tbem_geometry_key(elements, nodes),
                all_coords,
                routes,
                regular_pairs: regular_pairs_device,
                far_pair_channels: far_pair_channels_device,
                retains_far_channels: retain_far_channels,
                quadrature_pairs: quadrature_pairs_device,
                quadrature_samples: quadrature_samples_device,
                regular_output,
                regular_real_output,
                quadrature_output,
                regular_derivative_output,
                regular_source_x_derivative_output,
                regular_source_y_derivative_output,
                quadrature_derivative_output,
                last_regular_output: Vec::new(),
                last_quadrature_output: Vec::new(),
                last_physics_key: None,
            })
        }

        /// The mesh identity captured by this workspace.
        pub fn geometry_key(&self) -> u64 {
            self.geometry_key
        }

        /// Assemble the resident far-panel regular pairs through the native
        /// real-valued `f64` Laplace seam.
        ///
        /// Requires [`Self::new_with_regular_pair_channels`].
        ///
        /// This deliberately returns scalar channels instead of embedding a
        /// real projection into `TbemSystem`'s Complex64 matrix.  The method
        /// is therefore useful for static/Laplace callers and capability
        /// evidence, while the Helmholtz `assemble` method above remains the
        /// only path that publishes a complex TBEM system.
        pub fn assemble_real_f64(
            &mut self,
            elements: &[Element],
            nodes: &[[f64; 3]],
            backend: &CudaBackend,
        ) -> Result<Vec<BemRegularPairOutRealF64>, CudaTbemAssemblyError> {
            if !self.retains_far_channels {
                return Err(CudaTbemAssemblyError::InvalidInput {
                    reason: "far-pair channels require new_with_regular_pair_channels".into(),
                });
            }
            let current_key = cuda_tbem_geometry_key(elements, nodes);
            if current_key != self.geometry_key {
                return Err(CudaTbemAssemblyError::InvalidInput {
                    reason: format!(
                        "resident CUDA TBEM geometry cache is stale (expected key {}, got {})",
                        self.geometry_key, current_key
                    ),
                });
            }
            let mut output =
                vec![BemRegularPairOutRealF64::default(); self.far_pair_channels.len()];
            if self.far_pair_channels.is_empty() {
                return Ok(output);
            }
            backend.try_bem_regular_pair_real_f64_device(
                &self.far_pair_channels,
                &mut self.regular_real_output,
            )?;
            backend.try_synchronize()?;
            backend.try_copy_d2h(&self.regular_real_output, &mut output)?;
            for (index, value) in output.iter().enumerate() {
                if [value.g, value.h, value.ht, value.e]
                    .into_iter()
                    .any(|component| !component.is_finite())
                {
                    return Err(CudaTbemAssemblyError::Backend(
                        BackendError::OperationFailed {
                            operation: "cuda_regular_pair_real_f64_output",
                            reason: format!("pair {index} contains a non-finite value"),
                        },
                    ));
                }
            }
            Ok(output)
        }

        /// Assemble the native d/d(source-point.x) derivative of resident
        /// far-panel regular pairs.
        ///
        /// Requires [`Self::new_with_regular_pair_channels`].
        ///
        /// The field-panel geometry, source normal, and four-point rule are
        /// retained on the device.  This method publishes only the regular
        /// pair channels because near/singular and element-Jacobian
        /// shape-calculus derivatives are separate, explicit capability
        /// cells.  A prior value assembly is not required: the derivative
        /// kernel consumes the resident geometry and current physics only.
        pub fn assemble_source_x_derivative(
            &mut self,
            elements: &[Element],
            nodes: &[[f64; 3]],
            physics: &PhysicsParams,
            backend: &CudaBackend,
        ) -> Result<Vec<BemRegularPairOutF64>, CudaTbemAssemblyError> {
            if !self.retains_far_channels {
                return Err(CudaTbemAssemblyError::InvalidInput {
                    reason: "far-pair channels require new_with_regular_pair_channels".into(),
                });
            }
            let current_key = cuda_tbem_geometry_key(elements, nodes);
            if current_key != self.geometry_key {
                return Err(CudaTbemAssemblyError::InvalidInput {
                    reason: format!(
                        "resident CUDA TBEM geometry cache is stale (expected key {}, got {})",
                        self.geometry_key, current_key
                    ),
                });
            }
            if !physics.wave_number.is_finite() || !physics.harmonic_factor.is_finite() {
                return Err(CudaTbemAssemblyError::InvalidInput {
                    reason: "source-position derivative physics must be finite".into(),
                });
            }
            let mut output = vec![BemRegularPairOutF64::default(); self.far_pair_channels.len()];
            if self.far_pair_channels.is_empty() {
                return Ok(output);
            }
            backend.try_bem_regular_pair_c64_source_x_derivative_device(
                &self.far_pair_channels,
                &mut self.regular_source_x_derivative_output,
                physics.wave_number,
                physics.harmonic_factor,
            )?;
            backend.try_synchronize()?;
            backend.try_copy_d2h(&self.regular_source_x_derivative_output, &mut output)?;
            validate_cuda_pair_outputs(&output, "cuda_regular_pair_source_x_derivative_output")?;
            Ok(output)
        }

        /// Publish the resident source-x derivative output without launching
        /// another kernel.  This is used by checked callers to verify that a
        /// pre-launch failure leaves the previously published device output
        /// untouched before a retry.
        pub fn read_source_x_derivative_output(
            &self,
            backend: &CudaBackend,
        ) -> Result<Vec<BemRegularPairOutF64>, CudaTbemAssemblyError> {
            if !self.retains_far_channels {
                return Err(CudaTbemAssemblyError::InvalidInput {
                    reason: "far-pair channels require new_with_regular_pair_channels".into(),
                });
            }
            let mut output = vec![BemRegularPairOutF64::default(); self.far_pair_channels.len()];
            if self.far_pair_channels.is_empty() {
                return Ok(output);
            }
            backend.try_copy_d2h(&self.regular_source_x_derivative_output, &mut output)?;
            validate_cuda_pair_outputs(&output, "cuda_regular_pair_source_x_derivative_output")?;
            Ok(output)
        }

        /// Assemble the native d/d(source-point.y) derivative of resident
        /// far-panel regular pairs.  The field-panel geometry and pair
        /// descriptors from [`Self::new_with_regular_pair_channels`] remain
        /// device-resident; only the checked publication
        /// crosses back to the host.
        pub fn assemble_source_y_derivative(
            &mut self,
            elements: &[Element],
            nodes: &[[f64; 3]],
            physics: &PhysicsParams,
            backend: &CudaBackend,
        ) -> Result<Vec<BemRegularPairOutF64>, CudaTbemAssemblyError> {
            if !self.retains_far_channels {
                return Err(CudaTbemAssemblyError::InvalidInput {
                    reason: "far-pair channels require new_with_regular_pair_channels".into(),
                });
            }
            let current_key = cuda_tbem_geometry_key(elements, nodes);
            if current_key != self.geometry_key {
                return Err(CudaTbemAssemblyError::InvalidInput {
                    reason: format!(
                        "resident CUDA TBEM geometry cache is stale (expected key {}, got {})",
                        self.geometry_key, current_key
                    ),
                });
            }
            if !physics.wave_number.is_finite() || !physics.harmonic_factor.is_finite() {
                return Err(CudaTbemAssemblyError::InvalidInput {
                    reason: "source-position derivative physics must be finite".into(),
                });
            }
            let mut output = vec![BemRegularPairOutF64::default(); self.far_pair_channels.len()];
            if self.far_pair_channels.is_empty() {
                return Ok(output);
            }
            backend.try_bem_regular_pair_c64_source_y_derivative_device(
                &self.far_pair_channels,
                &mut self.regular_source_y_derivative_output,
                physics.wave_number,
                physics.harmonic_factor,
            )?;
            backend.try_synchronize()?;
            backend.try_copy_d2h(&self.regular_source_y_derivative_output, &mut output)?;
            validate_cuda_pair_outputs(&output, "cuda_regular_pair_source_y_derivative_output")?;
            Ok(output)
        }

        /// Read the most recently published source-y derivative output.  This
        /// is used after an injected launch failure to prove transactional
        /// destination preservation without re-running the derivative.
        pub fn read_source_y_derivative_output(
            &self,
            backend: &CudaBackend,
        ) -> Result<Vec<BemRegularPairOutF64>, CudaTbemAssemblyError> {
            if !self.retains_far_channels {
                return Err(CudaTbemAssemblyError::InvalidInput {
                    reason: "far-pair channels require new_with_regular_pair_channels".into(),
                });
            }
            let mut output = vec![BemRegularPairOutF64::default(); self.far_pair_channels.len()];
            if self.far_pair_channels.is_empty() {
                return Ok(output);
            }
            backend.try_copy_d2h(&self.regular_source_y_derivative_output, &mut output)?;
            validate_cuda_pair_outputs(&output, "cuda_regular_pair_source_y_derivative_output")?;
            Ok(output)
        }

        /// Assemble one frequency using the resident pair geometry.
        pub fn assemble(
            &mut self,
            elements: &[Element],
            nodes: &[[f64; 3]],
            physics: &PhysicsParams,
            beta: Complex64,
            backend: &CudaBackend,
        ) -> Result<TbemSystem, CudaTbemAssemblyError> {
            let num_dofs = validate_cuda_tbem_inputs(elements, nodes, physics, beta)?;
            let current_key = cuda_tbem_geometry_key(elements, nodes);
            if current_key != self.geometry_key {
                return Err(CudaTbemAssemblyError::InvalidInput {
                    reason: format!(
                        "resident CUDA TBEM geometry cache is stale (expected key {}, got {})",
                        self.geometry_key, current_key
                    ),
                });
            }
            let mut system = TbemSystem::new(num_dofs);
            let mut gpu_out = vec![BemRegularPairOutF64::default(); self.regular_pairs.len()];
            let mut gpu_quadrature_out =
                vec![BemRegularPairOutF64::default(); self.quadrature_pairs.len()];
            if !self.regular_pairs.is_empty() {
                backend.try_bem_regular_pair_c64_device(
                    &self.regular_pairs,
                    &mut self.regular_output,
                    physics.wave_number,
                    physics.harmonic_factor,
                )?;
            }
            if !self.quadrature_pairs.is_empty() {
                backend.try_bem_quadrature_pair_c64_device(
                    &self.quadrature_pairs,
                    &self.quadrature_samples,
                    &mut self.quadrature_output,
                    physics.wave_number,
                    physics.harmonic_factor,
                )?;
            }
            if !self.regular_pairs.is_empty() || !self.quadrature_pairs.is_empty() {
                backend.try_synchronize()?;
            }
            if !self.regular_pairs.is_empty() {
                backend.try_copy_d2h(&self.regular_output, &mut gpu_out)?;
            }
            if !self.quadrature_pairs.is_empty() {
                backend.try_copy_d2h(&self.quadrature_output, &mut gpu_quadrature_out)?;
            }
            validate_cuda_pair_outputs(&gpu_out, "cuda_regular_pair_output")?;
            validate_cuda_pair_outputs(&gpu_quadrature_out, "cuda_quadrature_output")?;
            self.last_regular_output = gpu_out.clone();
            self.last_quadrature_output = gpu_quadrature_out.clone();
            self.last_physics_key = Some(cuda_tbem_physics_key(physics, beta));

            let gamma = Complex64::new(physics.gamma(), 0.0);
            let tau = Complex64::new(physics.tau, 0.0);
            let i_omega_rho = Complex64::new(0.0, physics.pressure_factor);
            for (iel, src) in elements.iter().enumerate() {
                if src.property.is_evaluation() {
                    continue;
                }
                let source_point = &src.center;
                let source_normal = &src.normal;
                let source_dof = src.dof_addresses[0];
                let (bc_type, bc_value) = get_bc_type_and_value(&src.boundary_condition);
                let avg_bc = bc_value.iter().sum::<Complex64>() / bc_value.len() as f64;
                match bc_type {
                    0 => {
                        system.matrix[[source_dof, source_dof]] -= gamma * 0.5;
                        if let Some(admittance) = src.boundary_condition.admittance() {
                            system.matrix[[source_dof, source_dof]] -=
                                admittance * i_omega_rho * beta * tau * 0.5;
                        }
                        system.rhs[source_dof] += avg_bc * i_omega_rho * beta * tau * 0.5;
                    }
                    1 => {
                        system.matrix[[source_dof, source_dof]] += beta * tau * 0.5;
                        system.rhs[source_dof] -= avg_bc * gamma * 0.5;
                    }
                    _ => {}
                }

                for (jel, fld) in elements.iter().enumerate() {
                    if fld.property.is_evaluation() {
                        continue;
                    }
                    let element_coords = &self.all_coords[jel];
                    let field_dof = fld.dof_addresses[0];
                    let (field_bc_type, field_bc_values) =
                        get_bc_type_and_value(&fld.boundary_condition);
                    let compute_rhs = has_nonzero_bc(&field_bc_values);
                    let result = match self.routes[iel][jel] {
                        PairRoute::Self_ => crate::core::integration::singular_integration(
                            source_point,
                            source_normal,
                            element_coords,
                            fld.element_type,
                            physics,
                            compute_rhs.then_some(field_bc_values.as_slice()),
                            field_bc_type,
                            compute_rhs,
                        ),
                        PairRoute::Gpu(gpu_idx) => {
                            let g = &gpu_out[gpu_idx];
                            IntegrationResult {
                                g_integral: g.g,
                                dg_dn_integral: g.h,
                                dg_dnx_integral: g.ht,
                                d2g_dnxdny_integral: g.e,
                                rhs_contribution: Complex64::default(),
                            }
                        }
                        PairRoute::GpuQuadrature(quadrature_idx) => {
                            let g = &gpu_quadrature_out[quadrature_idx];
                            IntegrationResult {
                                g_integral: g.g,
                                dg_dn_integral: g.h,
                                dg_dnx_integral: g.ht,
                                d2g_dnxdny_integral: g.e,
                                rhs_contribution: Complex64::default(),
                            }
                        }
                    };
                    let mut coeff = match field_bc_type {
                        0 => {
                            result.dg_dn_integral * gamma * tau + result.d2g_dnxdny_integral * beta
                        }
                        1 => -(result.g_integral * gamma * tau + result.dg_dnx_integral * beta),
                        _ => Complex64::default(),
                    };
                    if field_bc_type == 0
                        && let Some(admittance) = fld.boundary_condition.admittance()
                    {
                        coeff -= (result.g_integral * gamma * tau + result.dg_dnx_integral * beta)
                            * admittance
                            * i_omega_rho;
                    }
                    system.matrix[[source_dof, field_dof]] += coeff;
                    if compute_rhs {
                        let avg_field_bc = field_bc_values.iter().sum::<Complex64>()
                            / field_bc_values.len() as f64;
                        let rhs_contrib = match field_bc_type {
                            0 => {
                                (result.g_integral * gamma * tau + result.dg_dnx_integral * beta)
                                    * avg_field_bc
                                    * i_omega_rho
                            }
                            1 => {
                                (result.dg_dn_integral * gamma * tau
                                    + result.d2g_dnxdny_integral * beta)
                                    * avg_field_bc
                            }
                            _ => Complex64::default(),
                        };
                        system.rhs[source_dof] += rhs_contrib;
                    }
                }
            }
            Ok(system)
        }

        /// Assemble the fixed-geometry constant-panel TBEM derivative with
        /// respect to wavenumber using the resident CUDA pair charts.
        ///
        /// [`Self::assemble`] must have succeeded immediately before this
        /// method.  The base pair values are retained by that call so the
        /// derivative includes both the device-computed channel derivatives
        /// and the explicit Burton--Miller/pressure-factor derivatives on the
        /// host.  No host finite-difference or CPU integration fallback is
        /// used for the pair channels.
        pub fn assemble_wavenumber_derivative(
            &mut self,
            elements: &[Element],
            nodes: &[[f64; 3]],
            physics: &PhysicsParams,
            beta: Complex64,
            beta_derivative: Complex64,
            pressure_factor_derivative: f64,
            sign: f64,
            backend: &CudaBackend,
        ) -> Result<TbemSystem, CudaTbemAssemblyError> {
            let num_dofs = validate_cuda_tbem_inputs(elements, nodes, physics, beta)?;
            let current_key = cuda_tbem_geometry_key(elements, nodes);
            if current_key != self.geometry_key {
                return Err(CudaTbemAssemblyError::InvalidInput {
                    reason: format!(
                        "resident CUDA TBEM geometry cache is stale (expected key {}, got {})",
                        self.geometry_key, current_key
                    ),
                });
            }
            if self.last_physics_key != Some(cuda_tbem_physics_key(physics, beta))
                || self.last_regular_output.len() != self.regular_pairs.len()
                || self.last_quadrature_output.len() != self.quadrature_pairs.len()
            {
                return Err(CudaTbemAssemblyError::InvalidInput {
                    reason: "CUDA TBEM derivative requires the matching preceding value assembly (including frequency physics)".into(),
                });
            }
            if !sign.is_finite()
                || !pressure_factor_derivative.is_finite()
                || !beta_derivative.is_finite()
            {
                return Err(CudaTbemAssemblyError::InvalidInput {
                    reason: "CUDA TBEM derivative parameters must be finite".into(),
                });
            }

            let mut regular_derivative =
                vec![BemRegularPairOutF64::default(); self.regular_pairs.len()];
            let mut quadrature_derivative =
                vec![BemRegularPairOutF64::default(); self.quadrature_pairs.len()];
            if !self.regular_pairs.is_empty() {
                backend.try_bem_regular_pair_c64_derivative_device(
                    &self.regular_pairs,
                    &mut self.regular_derivative_output,
                    physics.wave_number,
                    physics.harmonic_factor,
                )?;
            }
            if !self.quadrature_pairs.is_empty() {
                backend.try_bem_quadrature_pair_c64_derivative_device(
                    &self.quadrature_pairs,
                    &self.quadrature_samples,
                    &mut self.quadrature_derivative_output,
                    physics.wave_number,
                    physics.harmonic_factor,
                )?;
            }
            if !self.regular_pairs.is_empty() || !self.quadrature_pairs.is_empty() {
                backend.try_synchronize()?;
            }
            if !self.regular_pairs.is_empty() {
                backend.try_copy_d2h(&self.regular_derivative_output, &mut regular_derivative)?;
            }
            if !self.quadrature_pairs.is_empty() {
                backend.try_copy_d2h(
                    &self.quadrature_derivative_output,
                    &mut quadrature_derivative,
                )?;
            }
            validate_cuda_pair_outputs(
                &regular_derivative,
                "cuda_regular_pair_wavenumber_derivative_output",
            )?;
            validate_cuda_pair_outputs(
                &quadrature_derivative,
                "cuda_quadrature_pair_wavenumber_derivative_output",
            )?;

            let mut system = TbemSystem::new(num_dofs);
            let gamma = Complex64::new(physics.gamma(), 0.0);
            let tau = Complex64::new(physics.tau, 0.0);
            let i_omega_rho = Complex64::new(0.0, physics.pressure_factor);
            let i_omega_rho_derivative = Complex64::new(0.0, pressure_factor_derivative);
            let beta_term_derivative =
                i_omega_rho_derivative * beta + i_omega_rho * beta_derivative;

            for (iel, src) in elements.iter().enumerate() {
                if src.property.is_evaluation() {
                    continue;
                }
                let source_dof = src.dof_addresses[0];
                let (source_bc_type, source_bc_values) =
                    get_bc_type_and_value(&src.boundary_condition);
                let average_source_bc =
                    source_bc_values.iter().sum::<Complex64>() / source_bc_values.len() as f64;
                match source_bc_type {
                    0 => {
                        if let Some(admittance) = src.boundary_condition.admittance() {
                            system.matrix[[source_dof, source_dof]] -=
                                sign * admittance * beta_term_derivative * tau * 0.5;
                        }
                        system.rhs[source_dof] +=
                            sign * average_source_bc * beta_term_derivative * tau * 0.5;
                    }
                    1 => {
                        system.matrix[[source_dof, source_dof]] +=
                            sign * beta_derivative * tau * 0.5;
                    }
                    _ => {}
                }

                for (jel, fld) in elements.iter().enumerate() {
                    if fld.property.is_evaluation() {
                        continue;
                    }
                    let field_dof = fld.dof_addresses[0];
                    let (field_bc_type, field_bc_values) =
                        get_bc_type_and_value(&fld.boundary_condition);
                    let (base, derivative) = match self.routes[iel][jel] {
                        PairRoute::Gpu(index) => (
                            self.last_regular_output.get(index).ok_or_else(|| {
                                CudaTbemAssemblyError::InvalidInput {
                                    reason: "resident regular-pair output index is stale".into(),
                                }
                            })?,
                            regular_derivative.get(index).ok_or_else(|| {
                                CudaTbemAssemblyError::InvalidInput {
                                    reason: "resident regular-pair derivative index is stale"
                                        .into(),
                                }
                            })?,
                        ),
                        PairRoute::GpuQuadrature(index) => (
                            self.last_quadrature_output.get(index).ok_or_else(|| {
                                CudaTbemAssemblyError::InvalidInput {
                                    reason: "resident quadrature output index is stale".into(),
                                }
                            })?,
                            quadrature_derivative.get(index).ok_or_else(|| {
                                CudaTbemAssemblyError::InvalidInput {
                                    reason: "resident quadrature derivative index is stale".into(),
                                }
                            })?,
                        ),
                        PairRoute::Self_ => {
                            return Err(CudaTbemAssemblyError::InvalidInput {
                                reason:
                                    "CUDA TBEM derivative encountered an unaccelerated self route"
                                        .into(),
                            });
                        }
                    };
                    let d_cb =
                        derivative.h * gamma * tau + derivative.e * beta + base.e * beta_derivative;
                    let d_hb = derivative.g * gamma * tau
                        + derivative.ht * beta
                        + base.ht * beta_derivative;
                    let h_b = base.g * gamma * tau + base.ht * beta;
                    let coefficient = match field_bc_type {
                        0 => {
                            let mut value = sign * d_cb;
                            if let Some(admittance) = fld.boundary_condition.admittance() {
                                value -= sign
                                    * admittance
                                    * (d_hb * i_omega_rho + h_b * i_omega_rho_derivative);
                            }
                            value
                        }
                        1 => -sign * d_hb,
                        _ => Complex64::default(),
                    };
                    system.matrix[[source_dof, field_dof]] += coefficient;

                    if has_nonzero_bc(&field_bc_values) {
                        let average_field_bc = field_bc_values.iter().sum::<Complex64>()
                            / field_bc_values.len() as f64;
                        let rhs_derivative = match field_bc_type {
                            0 => {
                                sign * average_field_bc
                                    * (d_hb * i_omega_rho + h_b * i_omega_rho_derivative)
                            }
                            1 => sign * average_field_bc * d_cb,
                            _ => Complex64::default(),
                        };
                        system.rhs[source_dof] += rhs_derivative;
                    }
                }
            }
            Ok(system)
        }
    }

    fn validate_cuda_pair_outputs(
        values: &[BemRegularPairOutF64],
        operation: &'static str,
    ) -> Result<(), CudaTbemAssemblyError> {
        for (index, value) in values.iter().enumerate() {
            if [
                value.g.re,
                value.g.im,
                value.h.re,
                value.h.im,
                value.ht.re,
                value.ht.im,
                value.e.re,
                value.e.im,
            ]
            .iter()
            .any(|component| !component.is_finite())
            {
                return Err(CudaTbemAssemblyError::Backend(
                    BackendError::OperationFailed {
                        operation,
                        reason: format!("pair {index} contains a non-finite value"),
                    },
                ));
            }
        }
        Ok(())
    }

    fn cuda_tbem_geometry_key(elements: &[Element], nodes: &[[f64; 3]]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        nodes.len().hash(&mut hasher);
        for node in nodes {
            for coordinate in node {
                coordinate.to_bits().hash(&mut hasher);
            }
        }
        elements.len().hash(&mut hasher);
        for element in elements {
            element.connectivity.hash(&mut hasher);
            match element.element_type {
                crate::core::types::ElementType::Tri3 => 3_u8.hash(&mut hasher),
                crate::core::types::ElementType::Quad4 => 4_u8.hash(&mut hasher),
            }
            (element.property as u8).hash(&mut hasher);
            for value in element
                .normal
                .iter()
                .chain(element.center.iter())
                .chain(std::iter::once(&element.area))
            {
                value.to_bits().hash(&mut hasher);
            }
            for normal in &element.node_normals {
                for value in normal {
                    value.to_bits().hash(&mut hasher);
                }
            }
            element.group.hash(&mut hasher);
            element.dof_addresses.hash(&mut hasher);
        }
        hasher.finish()
    }

    pub(super) fn validate_cuda_tbem_inputs(
        elements: &[Element],
        nodes: &[[f64; 3]],
        physics: &PhysicsParams,
        beta: Complex64,
    ) -> Result<usize, CudaTbemAssemblyError> {
        physics
            .validate()
            .map_err(|error| CudaTbemAssemblyError::InvalidInput {
                reason: error.to_string(),
            })?;
        if !beta.is_finite() {
            return Err(CudaTbemAssemblyError::InvalidInput {
                reason: "Burton-Miller beta must be finite".into(),
            });
        }

        let mut max_dof = None;
        let mut owners = std::collections::HashSet::new();
        let mut active_elements = 0usize;
        for (element_index, element) in elements.iter().enumerate() {
            if element.connectivity.len() != element.element_type.num_nodes() {
                return Err(CudaTbemAssemblyError::InvalidInput {
                    reason: format!(
                        "element {element_index} has {} connectivity entries, expected {}",
                        element.connectivity.len(),
                        element.element_type.num_nodes()
                    ),
                });
            }
            for (local, &node) in element.connectivity.iter().enumerate() {
                let Some(coordinates) = nodes.get(node) else {
                    return Err(CudaTbemAssemblyError::InvalidInput {
                        reason: format!(
                            "element {element_index} connectivity {local} references node {node}, but the mesh has {} nodes",
                            nodes.len()
                        ),
                    });
                };
                if coordinates.iter().any(|value| !value.is_finite()) {
                    return Err(CudaTbemAssemblyError::InvalidInput {
                        reason: format!("node {node} contains NaN or infinity"),
                    });
                }
            }
            if element.property.is_evaluation() {
                continue;
            }
            active_elements += 1;
            if element.center.iter().any(|value| !value.is_finite())
                || element.normal.iter().any(|value| !value.is_finite())
                || !element.area.is_finite()
                || element.area <= 0.0
            {
                return Err(CudaTbemAssemblyError::InvalidInput {
                    reason: format!(
                        "element {element_index} has non-finite or non-positive geometry"
                    ),
                });
            }
            let normal_squared = element
                .normal
                .iter()
                .map(|value| value * value)
                .sum::<f64>();
            if !normal_squared.is_finite() || normal_squared.sqrt() < 1.0e-15 {
                return Err(CudaTbemAssemblyError::InvalidInput {
                    reason: format!("element {element_index} has a degenerate surface normal"),
                });
            }
            if element.element_type == crate::core::types::ElementType::Tri3 {
                // The length check above makes these three indexed accesses
                // infallible without introducing a panic into the checked
                // accelerator admission path.
                let node0 = element.connectivity[0];
                let node1 = element.connectivity[1];
                let node2 = element.connectivity[2];
                let edge1 = [
                    nodes[node1][0] - nodes[node0][0],
                    nodes[node1][1] - nodes[node0][1],
                    nodes[node1][2] - nodes[node0][2],
                ];
                let edge2 = [
                    nodes[node2][0] - nodes[node0][0],
                    nodes[node2][1] - nodes[node0][1],
                    nodes[node2][2] - nodes[node0][2],
                ];
                let jacobian = cross_product(edge1, edge2)
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt();
                if !jacobian.is_finite() || jacobian < 1.0e-15 {
                    return Err(CudaTbemAssemblyError::InvalidInput {
                        reason: format!("element {element_index} has degenerate Tri3 vertices"),
                    });
                }
            }
            let (_, bc_values) = get_bc_type_and_value(&element.boundary_condition);
            if bc_values.is_empty() || bc_values.iter().any(|value| !value.is_finite()) {
                return Err(CudaTbemAssemblyError::InvalidInput {
                    reason: format!(
                        "element {element_index} has an empty or non-finite boundary condition"
                    ),
                });
            }
            if element.element_type != crate::core::types::ElementType::Tri3
                || element.dof_addresses.len() != 1
            {
                return Err(CudaTbemAssemblyError::UnsupportedElement {
                    element: element_index,
                    element_type: match element.element_type {
                        crate::core::types::ElementType::Tri3 => "Tri3",
                        crate::core::types::ElementType::Quad4 => "Quad4",
                    },
                    dofs: element.dof_addresses.len(),
                });
            }
            let dof = element.dof_addresses[0];
            if !owners.insert(dof) {
                return Err(CudaTbemAssemblyError::InvalidInput {
                    reason: format!("active element {element_index} reuses global DOF {dof}"),
                });
            }
            max_dof = Some(max_dof.map_or(dof, |current: usize| current.max(dof)));
        }
        if active_elements == 0 {
            return Err(CudaTbemAssemblyError::InvalidInput {
                reason: "CUDA TBEM assembly requires at least one active surface element".into(),
            });
        }
        match max_dof {
            Some(dof) => dof
                .checked_add(1)
                .ok_or_else(|| CudaTbemAssemblyError::InvalidInput {
                    reason: "global DOF index overflows the system dimension".into(),
                }),
            None => Ok(0),
        }
    }

    fn tri_geometry(coords: &DenseMatrix<f64>, xi: f64, eta: f64) -> ([f64; 3], f64, [f64; 3]) {
        let shape = [1.0 - xi - eta, xi, eta];
        let mut position = [0.0; 3];
        for (node, weight) in shape.into_iter().enumerate() {
            for axis in 0..3 {
                position[axis] += weight * coords[[node, axis]];
            }
        }
        let edge1 = [
            coords[[1, 0]] - coords[[0, 0]],
            coords[[1, 1]] - coords[[0, 1]],
            coords[[1, 2]] - coords[[0, 2]],
        ];
        let edge2 = [
            coords[[2, 0]] - coords[[0, 0]],
            coords[[2, 1]] - coords[[0, 1]],
            coords[[2, 2]] - coords[[0, 2]],
        ];
        let raw_normal = cross_product(edge1, edge2);
        let jacobian = raw_normal
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        let normal = if jacobian > 1.0e-15 {
            raw_normal.map(|value| value / jacobian)
        } else {
            [0.0; 3]
        };
        (position, jacobian, normal)
    }

    /// Build a resident CUDA quadrature batch for one production Quad4
    /// geometry with its discontinuous quadratic (nine-function) panel basis.
    ///
    /// This is intentionally a pair-level adapter: the checked constant-panel
    /// TBEM matrix builder remains Tri3-only, while callers that already own a
    /// production Quad4 form can submit these descriptors to
    /// [`sonium_compute::CudaBemQuadratureOperatorC64`].  Geometry, Jacobians,
    /// normals, and basis values are taken from the production element map;
    /// the CUDA kernel evaluates and accumulates all Green-function channels.
    pub fn build_quad4_cuda_regular_quadrature(
        element: &Element,
        nodes: &[[f64; 3]],
        source_point: [f64; 3],
        source_normal: [f64; 3],
        gauss_order: usize,
    ) -> Result<(Vec<BemQuadraturePairC64>, Vec<BemQuadratureSampleC64>), CudaTbemAssemblyError>
    {
        if element.element_type != crate::core::types::ElementType::Quad4 {
            return Err(CudaTbemAssemblyError::InvalidInput {
                reason: "Quad4 CUDA quadrature adapter requires a Quad4 element".into(),
            });
        }
        if element.connectivity.len() != 4 {
            return Err(CudaTbemAssemblyError::InvalidInput {
                reason: format!(
                    "Quad4 CUDA quadrature adapter received {} connectivity entries",
                    element.connectivity.len()
                ),
            });
        }
        if !(1..=crate::core::integration::gauss::N_GAUORDER).contains(&gauss_order) {
            return Err(CudaTbemAssemblyError::InvalidInput {
                reason: format!("Quad4 CUDA quadrature order {gauss_order} is outside 1..=80"),
            });
        }
        if source_point
            .iter()
            .chain(source_normal.iter())
            .any(|value| !value.is_finite())
        {
            return Err(CudaTbemAssemblyError::InvalidInput {
                reason: "Quad4 CUDA quadrature source geometry is non-finite".into(),
            });
        }
        let source_normal_norm = source_normal
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if !source_normal_norm.is_finite() || source_normal_norm < 1.0e-15 {
            return Err(CudaTbemAssemblyError::InvalidInput {
                reason: "Quad4 CUDA quadrature source normal is degenerate".into(),
            });
        }
        if element.connectivity.iter().any(|&node| {
            nodes
                .get(node)
                .is_none_or(|coordinates| coordinates.iter().any(|value| !value.is_finite()))
        }) {
            return Err(CudaTbemAssemblyError::InvalidInput {
                reason: "Quad4 CUDA quadrature connectivity references missing or non-finite nodes"
                    .into(),
            });
        }

        let mut geometry = Vec::new();
        for (xi, eta, quadrature_weight) in quad_quadrature(gauss_order) {
            let shape = element.shape_functions(xi, eta);
            let (shape_ds, shape_dt) = element.shape_derivatives(xi, eta);
            let mut position = [0.0; 3];
            let mut tangent_s = [0.0; 3];
            let mut tangent_t = [0.0; 3];
            for (local, &node_index) in element.connectivity.iter().enumerate() {
                let coordinates = nodes[node_index];
                for axis in 0..3 {
                    position[axis] += shape[local] * coordinates[axis];
                    tangent_s[axis] += shape_ds[local] * coordinates[axis];
                    tangent_t[axis] += shape_dt[local] * coordinates[axis];
                }
            }
            let raw_normal = cross_product(tangent_s, tangent_t);
            let jacobian = raw_normal
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            if !jacobian.is_finite() || jacobian < 1.0e-15 {
                return Err(CudaTbemAssemblyError::InvalidInput {
                    reason: "Quad4 CUDA quadrature geometry has a degenerate Jacobian".into(),
                });
            }
            let field_normal = raw_normal.map(|value| value / jacobian);
            let basis = basis_values(
                crate::core::types::ElementType::Quad4,
                BasisOrder::Quadratic,
                xi,
                eta,
            );
            geometry.push((position, field_normal, quadrature_weight * jacobian, basis));
        }

        let samples_per_pair = geometry.len();
        if samples_per_pair == 0 {
            return Err(CudaTbemAssemblyError::InvalidInput {
                reason: "Quad4 CUDA quadrature generated no samples".into(),
            });
        }
        let mut pairs = Vec::with_capacity(9);
        let mut samples = Vec::with_capacity(9 * samples_per_pair);
        for basis_index in 0..9 {
            let sample_offset =
                u64::try_from(samples.len()).map_err(|_| CudaTbemAssemblyError::InvalidInput {
                    reason: "Quad4 CUDA quadrature sample count exceeds u64".into(),
                })?;
            for &(position, field_normal, weight, ref basis) in &geometry {
                samples.push(BemQuadratureSampleC64::full(
                    position,
                    field_normal,
                    weight,
                    basis[basis_index],
                ));
            }
            pairs.push(BemQuadraturePairC64::new(
                source_point,
                source_normal,
                sample_offset,
                u64::try_from(samples_per_pair).map_err(|_| {
                    CudaTbemAssemblyError::InvalidInput {
                        reason: "Quad4 CUDA quadrature sample count exceeds u64".into(),
                    }
                })?,
            ));
        }
        Ok((pairs, samples))
    }

    /// Build the production corrected Duffy/edge chart used by the native
    /// quadrature kernel for a constant Tri3 self pair.  This small public
    /// adapter lets higher-order evidence compare its regular path beside the
    /// existing singular-corrected production path without copying chart
    /// construction into a QA binary.
    pub fn build_tri3_cuda_singular_quadrature(
        element: &Element,
        nodes: &[[f64; 3]],
        physics: &PhysicsParams,
    ) -> Result<(BemQuadraturePairC64, Vec<BemQuadratureSampleC64>), CudaTbemAssemblyError> {
        if element.element_type != crate::core::types::ElementType::Tri3
            || element.connectivity.len() != 3
        {
            return Err(CudaTbemAssemblyError::InvalidInput {
                reason: "CUDA singular quadrature adapter requires a Tri3 element".into(),
            });
        }
        let coords = get_element_coords(element, nodes);
        let mut samples = Vec::new();
        append_singular_tri3_quadrature(&coords, element.area, physics, &mut samples)?;
        let sample_count =
            u64::try_from(samples.len()).map_err(|_| CudaTbemAssemblyError::InvalidInput {
                reason: "CUDA singular quadrature sample count exceeds u64".into(),
            })?;
        Ok((
            BemQuadraturePairC64::new(element.center, element.normal, 0, sample_count),
            samples,
        ))
    }

    fn append_regular_tri3_quadrature(
        source_point: [f64; 3],
        coords: &DenseMatrix<f64>,
        area: f64,
        samples: &mut Vec<BemQuadratureSampleC64>,
    ) -> Result<(), CudaTbemAssemblyError> {
        let before = samples.len();
        for subelement in generate_subelements(
            &source_point,
            coords,
            crate::core::types::ElementType::Tri3,
            area,
        ) {
            for (xi, eta, weight) in triangle_quadrature(subelement.gauss_order) {
                let (mapped_xi, mapped_eta, mapped_weight) = if let Some(vertices) =
                    subelement.tri_vertices
                {
                    let l0 = 1.0 - xi - eta;
                    let mapped_xi = vertices[0].0 * l0 + vertices[1].0 * xi + vertices[2].0 * eta;
                    let mapped_eta = vertices[0].1 * l0 + vertices[1].1 * xi + vertices[2].1 * eta;
                    let dx1 = vertices[1].0 - vertices[0].0;
                    let dy1 = vertices[1].1 - vertices[0].1;
                    let dx2 = vertices[2].0 - vertices[0].0;
                    let dy2 = vertices[2].1 - vertices[0].1;
                    let determinant = (dx1 * dy2 - dx2 * dy1).abs();
                    (mapped_xi, mapped_eta, weight * determinant)
                } else {
                    (
                        subelement.xi_center + xi * subelement.factor,
                        subelement.eta_center + eta * subelement.factor,
                        weight * subelement.factor * subelement.factor,
                    )
                };
                let (position, jacobian, field_normal) =
                    tri_geometry(coords, mapped_xi, mapped_eta);
                if !jacobian.is_finite() || jacobian < 1.0e-15 {
                    return Err(CudaTbemAssemblyError::InvalidInput {
                        reason: "degenerate Tri3 geometry in CUDA quadrature chart".into(),
                    });
                }
                samples.push(BemQuadratureSampleC64::full(
                    position,
                    field_normal,
                    mapped_weight * jacobian,
                    1.0,
                ));
            }
        }
        if samples.len() == before {
            return Err(CudaTbemAssemblyError::InvalidInput {
                reason: "regular CUDA quadrature chart generated no samples".into(),
            });
        }
        Ok(())
    }

    fn append_singular_tri3_quadrature_with_params(
        coords: &DenseMatrix<f64>,
        quadrature: QuadratureParams,
        samples: &mut Vec<BemQuadratureSampleC64>,
    ) -> Result<(), CudaTbemAssemblyError> {
        const CSI6: [f64; 6] = [0.0, 1.0, 0.0, 0.5, 0.5, 0.0];
        const ETA6: [f64; 6] = [0.0, 0.0, 1.0, 0.0, 0.5, 0.5];
        let (edge_points, edge_weights) = gauss_legendre(quadrature.edge_gauss_order);
        let before = samples.len();

        for edge in 0..3 {
            let next = (edge + 1) % 3;
            let difference = [
                coords[[next, 0]] - coords[[edge, 0]],
                coords[[next, 1]] - coords[[edge, 1]],
                coords[[next, 2]] - coords[[edge, 2]],
            ];
            let length = difference
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            if !length.is_finite() || length < 1.0e-15 {
                return Err(CudaTbemAssemblyError::InvalidInput {
                    reason: "degenerate edge in CUDA singular quadrature chart".into(),
                });
            }
            let edge_direction = difference.map(|value| value / length);
            let section_scale = length / (2.0 * quadrature.edge_sections as f64);
            for section in 0..quadrature.edge_sections {
                let section_mid =
                    -1.0 + (section as f64 + 0.5) * (2.0 / quadrature.edge_sections as f64);
                for (&point, &weight) in edge_points.iter().zip(&edge_weights) {
                    let local = section_mid + point / quadrature.edge_sections as f64;
                    let position = [
                        coords[[edge, 0]] + difference[0] * (local + 1.0) / 2.0,
                        coords[[edge, 1]] + difference[1] * (local + 1.0) / 2.0,
                        coords[[edge, 2]] + difference[2] * (local + 1.0) / 2.0,
                    ];
                    samples.push(BemQuadratureSampleC64::hypersingular_edge(
                        position,
                        edge_direction,
                        weight * section_scale,
                        1.0,
                    ));
                }
            }
        }

        let (sub_points, sub_weights) = gauss_legendre(quadrature.subelement_gauss_order);
        for edge in 0..3 {
            let next = (edge + 1) % 3;
            let ig2 = edge + 3;
            for subtriangle in 0..quadrature.subtriangles_per_section {
                let area_scale = 1.0 / 24.0 / quadrature.subtriangles_per_section as f64;
                let (ssub, tsub) = if subtriangle == 0 {
                    (
                        [1.0 / 3.0, CSI6[edge], CSI6[ig2]],
                        [1.0 / 3.0, ETA6[edge], ETA6[ig2]],
                    )
                } else {
                    (
                        [1.0 / 3.0, CSI6[ig2], CSI6[next]],
                        [1.0 / 3.0, ETA6[ig2], ETA6[next]],
                    )
                };
                for (i, &sga) in sub_points.iter().enumerate() {
                    for (j, &tga) in sub_points.iter().enumerate() {
                        let xi = 0.5 * (1.0 - sga) * ssub[0]
                            + 0.25 * (1.0 + sga) * ((1.0 - tga) * ssub[1] + (1.0 + tga) * ssub[2]);
                        let eta = 0.5 * (1.0 - sga) * tsub[0]
                            + 0.25 * (1.0 + sga) * ((1.0 - tga) * tsub[1] + (1.0 + tga) * tsub[2]);
                        let (position, jacobian, field_normal) = tri_geometry(coords, xi, eta);
                        if !jacobian.is_finite() || jacobian < 1.0e-15 {
                            return Err(CudaTbemAssemblyError::InvalidInput {
                                reason: "degenerate Tri3 geometry in CUDA Duffy chart".into(),
                            });
                        }
                        let weight =
                            sub_weights[i] * sub_weights[j] * (1.0 + sga) * area_scale * jacobian;
                        samples.push(BemQuadratureSampleC64::singular_bulk(
                            position,
                            field_normal,
                            weight,
                            1.0,
                        ));
                    }
                }
            }
        }
        if samples.len() == before {
            return Err(CudaTbemAssemblyError::InvalidInput {
                reason: "singular CUDA quadrature chart generated no samples".into(),
            });
        }
        Ok(())
    }

    fn append_singular_tri3_quadrature(
        coords: &DenseMatrix<f64>,
        area: f64,
        physics: &PhysicsParams,
        samples: &mut Vec<BemQuadratureSampleC64>,
    ) -> Result<(), CudaTbemAssemblyError> {
        let element_size = area.sqrt().max(1.0e-10);
        let quadrature = QuadratureParams::for_ka(physics.wave_number * element_size);
        append_singular_tri3_quadrature_with_params(coords, quadrature, samples)
    }

    /// Build the dense BEM system matrix using CUDA for regular pairs and the
    /// device-quadrature path for quasi-singular and self pairs.
    pub fn build_tbem_system_with_beta_cuda_checked(
        elements: &[Element],
        nodes: &[[f64; 3]],
        physics: &PhysicsParams,
        beta: Complex64,
        backend: &CudaBackend,
    ) -> Result<TbemSystem, CudaTbemAssemblyError> {
        let num_dofs = validate_cuda_tbem_inputs(elements, nodes, physics, beta)?;
        let mut system = TbemSystem::new(num_dofs);

        let gamma = Complex64::new(physics.gamma(), 0.0);
        let tau = Complex64::new(physics.tau, 0.0);
        let i_omega_rho = Complex64::new(0.0, physics.pressure_factor);

        let all_coords: Vec<DenseMatrix<f64>> = elements
            .iter()
            .map(|e| get_element_coords(e, nodes))
            .collect();
        let n_elem = elements.len();
        let mut routes: Vec<Vec<PairRoute>> = vec![vec![PairRoute::Self_; n_elem]; n_elem];
        let mut gpu_pairs: Vec<BemRegularPairInF64> = Vec::new();
        let mut gpu_quadrature_pairs: Vec<BemQuadraturePairC64> = Vec::new();
        let mut gpu_quadrature_samples: Vec<BemQuadratureSampleC64> = Vec::new();

        for (iel, src) in elements.iter().enumerate() {
            if src.property.is_evaluation() {
                continue;
            }
            let sp = &src.center;
            let sn = &src.normal;
            for (jel, fld) in elements.iter().enumerate() {
                if fld.property.is_evaluation() {
                    continue;
                }
                if iel == jel {
                    let sample_offset =
                        u64::try_from(gpu_quadrature_samples.len()).map_err(|_| {
                            CudaTbemAssemblyError::InvalidInput {
                                reason: "CUDA quadrature sample count exceeds u64".into(),
                            }
                        })?;
                    append_singular_tri3_quadrature(
                        &all_coords[jel],
                        fld.area,
                        physics,
                        &mut gpu_quadrature_samples,
                    )?;
                    let sample_count = u64::try_from(
                        gpu_quadrature_samples
                            .len()
                            .checked_sub(sample_offset as usize)
                            .ok_or_else(|| CudaTbemAssemblyError::InvalidInput {
                                reason: "CUDA singular quadrature sample range underflowed".into(),
                            })?,
                    )
                    .map_err(|_| CudaTbemAssemblyError::InvalidInput {
                        reason: "CUDA quadrature sample count exceeds u64".into(),
                    })?;
                    let pair_index = gpu_quadrature_pairs.len();
                    gpu_quadrature_pairs.push(BemQuadraturePairC64::new(
                        *sp,
                        *sn,
                        sample_offset,
                        sample_count,
                    ));
                    routes[iel][jel] = PairRoute::GpuQuadrature(pair_index);
                    continue;
                }
                let center = &fld.center;
                let dx = center[0] - sp[0];
                let dy = center[1] - sp[1];
                let dz = center[2] - sp[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                let ratio = dist / fld.area.sqrt().max(1e-30);
                if ratio.is_finite() {
                    let sample_offset =
                        u64::try_from(gpu_quadrature_samples.len()).map_err(|_| {
                            CudaTbemAssemblyError::InvalidInput {
                                reason: "CUDA quadrature sample count exceeds u64".into(),
                            }
                        })?;
                    append_regular_tri3_quadrature(
                        *sp,
                        &all_coords[jel],
                        fld.area,
                        &mut gpu_quadrature_samples,
                    )?;
                    let sample_count = u64::try_from(
                        gpu_quadrature_samples
                            .len()
                            .checked_sub(sample_offset as usize)
                            .ok_or_else(|| CudaTbemAssemblyError::InvalidInput {
                                reason: "CUDA regular quadrature sample range underflowed".into(),
                            })?,
                    )
                    .map_err(|_| CudaTbemAssemblyError::InvalidInput {
                        reason: "CUDA quadrature sample count exceeds u64".into(),
                    })?;
                    let pair_index = gpu_quadrature_pairs.len();
                    gpu_quadrature_pairs.push(BemQuadraturePairC64::new(
                        *sp,
                        *sn,
                        sample_offset,
                        sample_count,
                    ));
                    routes[iel][jel] = PairRoute::GpuQuadrature(pair_index);
                } else {
                    let coords = &all_coords[jel];
                    debug_assert_eq!(coords.nrows(), 3, "Tri3 only on the CUDA path");
                    let gpu_idx = gpu_pairs.len();
                    gpu_pairs.push(BemRegularPairInF64::from_triangle(
                        *sp,
                        *sn,
                        [coords[[0, 0]], coords[[0, 1]], coords[[0, 2]]],
                        [coords[[1, 0]], coords[[1, 1]], coords[[1, 2]]],
                        [coords[[2, 0]], coords[[2, 1]], coords[[2, 2]]],
                    ));
                    routes[iel][jel] = PairRoute::Gpu(gpu_idx);
                }
            }
        }

        let gpu_out: Vec<BemRegularPairOutF64> = backend.try_bem_regular_pair_c64_batch(
            &gpu_pairs,
            physics.wave_number,
            physics.harmonic_factor,
        )?;
        let gpu_quadrature_out = backend.try_bem_quadrature_pair_c64_batch(
            &gpu_quadrature_pairs,
            &gpu_quadrature_samples,
            physics.wave_number,
            physics.harmonic_factor,
        )?;

        for (iel, src) in elements.iter().enumerate() {
            if src.property.is_evaluation() {
                continue;
            }
            let source_point = &src.center;
            let source_normal = &src.normal;
            let source_dof = src.dof_addresses[0];
            let (bc_type, bc_value) = get_bc_type_and_value(&src.boundary_condition);

            let avg_bc = bc_value.iter().sum::<Complex64>() / bc_value.len() as f64;
            match bc_type {
                0 => {
                    system.matrix[[source_dof, source_dof]] -= gamma * 0.5;
                    if let Some(admittance) = src.boundary_condition.admittance() {
                        system.matrix[[source_dof, source_dof]] -=
                            admittance * i_omega_rho * beta * tau * 0.5;
                    }
                    system.rhs[source_dof] += avg_bc * i_omega_rho * beta * tau * 0.5;
                }
                1 => {
                    system.matrix[[source_dof, source_dof]] += beta * tau * 0.5;
                    system.rhs[source_dof] -= avg_bc * gamma * 0.5;
                }
                _ => {}
            }

            for (jel, fld) in elements.iter().enumerate() {
                if fld.property.is_evaluation() {
                    continue;
                }
                let element_coords = &all_coords[jel];
                let field_dof = fld.dof_addresses[0];
                let (field_bc_type, field_bc_values) =
                    get_bc_type_and_value(&fld.boundary_condition);
                let compute_rhs = has_nonzero_bc(&field_bc_values);
                let bc_for_call = compute_rhs.then_some(field_bc_values.as_slice());

                let result = match routes[iel][jel] {
                    PairRoute::Self_ => crate::core::integration::singular_integration(
                        source_point,
                        source_normal,
                        element_coords,
                        fld.element_type,
                        physics,
                        bc_for_call,
                        field_bc_type,
                        compute_rhs,
                    ),
                    PairRoute::Gpu(gpu_idx) => {
                        let g = &gpu_out[gpu_idx];
                        IntegrationResult {
                            g_integral: g.g,
                            dg_dn_integral: g.h,
                            dg_dnx_integral: g.ht,
                            d2g_dnxdny_integral: g.e,
                            rhs_contribution: Complex64::new(0.0, 0.0),
                        }
                    }
                    PairRoute::GpuQuadrature(quadrature_idx) => {
                        let g = &gpu_quadrature_out[quadrature_idx];
                        IntegrationResult {
                            g_integral: g.g,
                            dg_dn_integral: g.h,
                            dg_dnx_integral: g.ht,
                            d2g_dnxdny_integral: g.e,
                            rhs_contribution: Complex64::new(0.0, 0.0),
                        }
                    }
                };

                let mut coeff = match field_bc_type {
                    0 => result.dg_dn_integral * gamma * tau + result.d2g_dnxdny_integral * beta,
                    1 => -(result.g_integral * gamma * tau + result.dg_dnx_integral * beta),
                    _ => Complex64::new(0.0, 0.0),
                };
                if field_bc_type == 0
                    && let Some(admittance) = fld.boundary_condition.admittance()
                {
                    coeff -= (result.g_integral * gamma * tau + result.dg_dnx_integral * beta)
                        * admittance
                        * i_omega_rho;
                }
                system.matrix[[source_dof, field_dof]] += coeff;

                if compute_rhs {
                    let avg_field_bc =
                        field_bc_values.iter().sum::<Complex64>() / field_bc_values.len() as f64;
                    let rhs_contrib = match field_bc_type {
                        0 => {
                            (result.g_integral * gamma * tau + result.dg_dnx_integral * beta)
                                * avg_field_bc
                                * i_omega_rho
                        }
                        1 => {
                            (result.dg_dn_integral * gamma * tau
                                + result.d2g_dnxdny_integral * beta)
                                * avg_field_bc
                        }
                        _ => Complex64::new(0.0, 0.0),
                    };
                    system.rhs[source_dof] += rhs_contrib;
                }
            }
        }

        Ok(system)
    }

    /// Compatibility wrapper for the historical infallible CUDA entry point.
    /// The checked function is the accelerator contract; this wrapper fails
    /// loudly rather than silently falling back to CPU assembly.
    pub fn build_tbem_system_with_beta_cuda(
        elements: &[Element],
        nodes: &[[f64; 3]],
        physics: &PhysicsParams,
        beta: Complex64,
        backend: &CudaBackend,
    ) -> TbemSystem {
        build_tbem_system_with_beta_cuda_checked(elements, nodes, physics, beta, backend)
            .unwrap_or_else(|error| panic!("checked CUDA TBEM assembly failed: {error}"))
    }
}

#[cfg(feature = "cuda")]
pub use cuda_path::{
    CudaTbemGmresSession, CudaTbemAssemblyError, CudaTbemAssemblyWorkspace, CudaTbemDeviceSystem, CudaTbemSolveError, CudaTbemSolveReport, CudaTbemSolveSession, build_quad4_cuda_regular_quadrature,
    build_tbem_system_with_beta_cuda, build_tbem_system_with_beta_cuda_checked,
    build_tri3_cuda_singular_quadrature,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{ElementProperty, ElementType};

    fn make_simple_mesh() -> (Vec<Element>, Vec<[f64; 3]>) {
        // Simple 2-element mesh for testing
        let nodes: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [1.5, 1.0, 0.0],
        ];

        let elem0 = Element {
            connectivity: vec![0, 1, 2],
            element_type: ElementType::Tri3,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0; 3]; 3],
            center: [0.5, 1.0 / 3.0, 0.0],
            area: 0.5,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(1.0, 0.0)]),
            group: 0,
            dof_addresses: vec![0],
        };

        let elem1 = Element {
            connectivity: vec![1, 3, 2],
            element_type: ElementType::Tri3,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0; 3]; 3],
            center: [1.0, 2.0 / 3.0, 0.0],
            area: 0.5,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
            group: 0,
            dof_addresses: vec![1],
        };

        (vec![elem0, elem1], nodes)
    }

    #[test]
    fn test_build_tbem_system() {
        let (elements, nodes) = make_simple_mesh();
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);

        let system = build_tbem_system(&elements, &nodes, &physics);

        assert_eq!(system.num_dofs, 2);
        assert_eq!(system.matrix.shape(), (2, 2));
        assert_eq!(system.rhs.len(), 2);

        // Matrix should have non-zero diagonal entries
        assert!(system.matrix[[0, 0]].norm() > 1e-15);
        assert!(system.matrix[[1, 1]].norm() > 1e-15);
    }

    #[test]
    fn checked_tbem_assembly_enforces_pairwise_quadrature_budget() {
        let (elements, nodes) = make_simple_mesh();
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);

        let permissive = TbemQuadraturePolicy {
            absolute_tolerance: 1.0e6,
            relative_tolerance: 1.0e6,
            ..TbemQuadraturePolicy::default()
        };
        let checked = build_tbem_system_checked(&elements, &nodes, &physics, permissive)
            .expect("a permissive pairwise gate should allow this reference mesh");
        assert_eq!(checked.num_dofs, 2);

        let strict = TbemQuadraturePolicy {
            absolute_tolerance: 0.0,
            relative_tolerance: 0.0,
            ..TbemQuadraturePolicy::default()
        };
        let error = match build_tbem_system_checked(&elements, &nodes, &physics, strict) {
            Ok(_) => panic!("zero budget must reject a non-identical adjacent estimate"),
            Err(error) => error,
        };
        assert!(error.source_element < elements.len());
        assert!(error.field_element < elements.len());
    }

    #[test]
    fn test_stabilized_rhs_only_matches_full_system() {
        let (mut elements, nodes) = make_simple_mesh();
        elements[1].boundary_condition = BoundaryCondition::VelocityWithAdmittance {
            velocity: vec![Complex64::new(0.2, -0.1)],
            admittance: Complex64::new(0.01, -0.02),
        };
        let physics = PhysicsParams::new(800.0, 343.0, 1.21, false);
        let sign = 1.0;
        let beta_scale = 4.0;
        let full = build_tbem_system_stabilized(&elements, &nodes, &physics, sign, beta_scale);
        let rhs = build_tbem_rhs_stabilized(&elements, &nodes, &physics, sign, beta_scale);
        assert_eq!(rhs.len(), full.rhs.len());
        for (index, (actual, expected)) in rhs.iter().zip(&full.rhs).enumerate() {
            assert!(
                (*actual - *expected).norm() < 1e-12,
                "RHS mismatch at {index}: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn stabilized_velocity_operator_matches_prescribed_velocity_system() {
        let (elements, nodes) = make_simple_mesh();
        let physics = PhysicsParams::new(800.0, 343.0, 1.21, false);
        let sign = 1.0;
        let beta = physics.burton_miller_beta_scaled(4.0);
        let operator = StabilizedTbemVelocityOperator::new(&elements, &nodes, &physics, sign, beta)
            .expect("constant-panel velocity operator");
        let velocity = [Complex64::new(0.2, -0.1), Complex64::new(-0.05, 0.3)];
        let rhs = operator.velocity_rhs(&velocity).expect("velocity RHS");

        let mut prescribed = elements.clone();
        for (element, &value) in prescribed.iter_mut().zip(&velocity) {
            element.boundary_condition = BoundaryCondition::Velocity(vec![value]);
        }
        let full =
            build_tbem_system_stabilized_with_beta(&prescribed, &nodes, &physics, sign, beta);

        assert_eq!(operator.len(), velocity.len());
        assert_eq!(operator.velocity_element_indices(), &[0, 1]);
        for (actual, expected) in operator
            .pressure_matrix()
            .as_slice()
            .iter()
            .zip(full.matrix.as_slice())
        {
            assert!((*actual - *expected).norm() < 1.0e-12);
        }
        for (actual, expected) in rhs.iter().zip(&full.rhs) {
            assert!((*actual - *expected).norm() < 1.0e-12);
        }
        assert!(matches!(
            operator.velocity_rhs(&velocity[..1]),
            Err(TbemVelocityOperatorError::VelocityLength { .. })
        ));
    }

    #[test]
    fn stabilized_velocity_operator_block_matches_individual_rhs_columns() {
        let (elements, nodes) = make_simple_mesh();
        let physics = PhysicsParams::new(800.0, 343.0, 1.21, false);
        let operator = StabilizedTbemVelocityOperator::new(
            &elements,
            &nodes,
            &physics,
            1.0,
            physics.burton_miller_beta_scaled(4.0),
        )
        .expect("constant-panel velocity operator");
        let input = DenseMatrix::from_row_major_vec(
            operator.len(),
            2,
            vec![
                Complex64::new(0.2, -0.1),
                Complex64::new(-0.05, 0.3),
                Complex64::new(-0.4, 0.2),
                Complex64::new(0.15, -0.35),
            ],
        );

        let block = operator
            .velocity_rhs_block(&input)
            .expect("batched velocity RHS");
        for column in 0..input.ncols() {
            let velocity = (0..input.nrows())
                .map(|row| input[[row, column]])
                .collect::<Vec<_>>();
            let expected = operator
                .velocity_rhs(&velocity)
                .expect("single velocity RHS");
            for (row, value) in expected.iter().enumerate() {
                assert!((block[[row, column]] - *value).norm() < 1.0e-12);
            }
        }

        let malformed = DenseMatrix::zeros(operator.len() - 1, 1);
        assert!(matches!(
            operator.velocity_rhs_block(&malformed),
            Err(TbemVelocityOperatorError::HMatrixInputLength { .. })
        ));
    }

    #[test]
    fn test_eval_tbem_rows_stabilized_matches_full_assembly() {
        // Three elements so restricted-column subsets exercise off-diagonal
        // entries, singular self terms, and velocity/admittance BCs.
        let (mut elements, nodes) = make_simple_mesh();
        elements[1].boundary_condition = BoundaryCondition::VelocityWithAdmittance {
            velocity: vec![Complex64::new(0.2, -0.1)],
            admittance: Complex64::new(0.01, -0.02),
        };
        elements.push(Element {
            connectivity: vec![0, 2, 3],
            element_type: ElementType::Tri3,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0; 3]; 3],
            center: [2.0 / 3.0, 2.0 / 3.0, 0.0],
            area: 0.5,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
            group: 0,
            dof_addresses: vec![2],
        });
        let physics = PhysicsParams::new(800.0, 343.0, 1.21, false);
        let sign = 1.0;
        let beta_scale = 4.0;
        let full = build_tbem_system_stabilized(&elements, &nodes, &physics, sign, beta_scale);
        let num_dofs = full.num_dofs;

        // Full-row partial evaluation must reproduce the complete system.
        let rows = eval_tbem_rows_stabilized(
            &elements,
            &nodes,
            &physics,
            sign,
            beta_scale,
            &[0, 1, 2],
            None,
        );
        assert_eq!(rows.len(), num_dofs);
        for partial_row in &rows {
            let mut accumulated = vec![Complex64::new(0.0, 0.0); num_dofs];
            for &(column, value) in &partial_row.entries {
                accumulated[column] += value;
            }
            for (column, &value) in accumulated.iter().enumerate() {
                assert!(
                    (value - full.matrix[[partial_row.row_dof, column]]).norm() < 1e-12,
                    "matrix mismatch at ({}, {column}): {} != {}",
                    partial_row.row_dof,
                    value,
                    full.matrix[[partial_row.row_dof, column]]
                );
            }
            assert!(
                (partial_row.rhs - full.rhs[partial_row.row_dof]).norm() < 1e-12,
                "RHS mismatch at row {}",
                partial_row.row_dof
            );
        }

        // Column-restricted evaluation: off-diagonal entries match the full
        // matrix; skipping the row-local jump terms shows up only on the
        // diagonal, and the restricted RHS drops the row-local term.
        let subset: Vec<usize> = (0..num_dofs).collect();
        let restricted = eval_tbem_rows_stabilized(
            &elements,
            &nodes,
            &physics,
            sign,
            beta_scale,
            &[0, 2],
            Some(&subset),
        );
        for partial_row in &restricted {
            let row = partial_row.row_dof;
            for &(column, value) in &partial_row.entries {
                if column != row {
                    assert!(
                        (value - full.matrix[[row, column]]).norm() < 1e-12,
                        "restricted entry mismatch at ({row}, {column})"
                    );
                }
            }
            // The full row equals the restricted row plus the row-local jump terms.
            let full_row = rows
                .iter()
                .find(|candidate| candidate.row_dof == row)
                .unwrap();
            let mut restricted_accumulated = vec![Complex64::new(0.0, 0.0); num_dofs];
            for &(column, value) in &partial_row.entries {
                restricted_accumulated[column] += value;
            }
            let mut full_accumulated = vec![Complex64::new(0.0, 0.0); num_dofs];
            for &(column, value) in &full_row.entries {
                full_accumulated[column] += value;
            }
            for column in 0..num_dofs {
                let local_term = full_accumulated[column] - restricted_accumulated[column];
                if column == row {
                    assert!(
                        local_term.norm() > 1e-6,
                        "row-local jump term should survive on the diagonal of row {row}"
                    );
                } else {
                    assert!(
                        local_term.norm() < 1e-12,
                        "unexpected row-local term off diagonal at ({row}, {column})"
                    );
                }
            }
            // Row 0 has a non-zero velocity BC, so its restricted RHS drops
            // the non-zero row-local term; row 2's local term is zero.
            let rhs_drop = (full_row.rhs - partial_row.rhs).norm();
            if row == 0 {
                assert!(rhs_drop > 1e-6, "row 0 should drop its local RHS term");
            } else {
                assert!(rhs_drop < 1e-12, "row {row} has no local RHS term");
            }
        }
    }

    #[test]
    fn test_stabilized_sign_one_matches_legacy_velocity_and_pressure_system() {
        let (mut elements, nodes) = make_simple_mesh();
        elements[0].boundary_condition =
            BoundaryCondition::Pressure(vec![Complex64::new(0.7, 0.2)]);
        elements[1].boundary_condition = BoundaryCondition::VelocityWithAdmittance {
            velocity: vec![Complex64::new(0.2, -0.1)],
            admittance: Complex64::new(0.01, -0.02),
        };
        let physics = PhysicsParams::new(800.0, 343.0, 1.21, false);
        let beta_scale = 4.0;
        let beta = physics.burton_miller_beta_scaled(beta_scale);
        let legacy = build_tbem_system_with_beta(&elements, &nodes, &physics, beta);
        let stabilized =
            build_tbem_system_stabilized_with_beta(&elements, &nodes, &physics, 1.0, beta);

        for row in 0..legacy.matrix.nrows() {
            for col in 0..legacy.matrix.ncols() {
                assert!(
                    (legacy.matrix[[row, col]] - stabilized.matrix[[row, col]]).norm() < 1e-12,
                    "matrix mismatch at ({row}, {col}): {} != {}",
                    legacy.matrix[[row, col]],
                    stabilized.matrix[[row, col]]
                );
            }
            assert!(
                (legacy.rhs[row] - stabilized.rhs[row]).norm() < 1e-12,
                "RHS mismatch at {row}: {} != {}",
                legacy.rhs[row],
                stabilized.rhs[row]
            );
        }
    }

    #[test]
    fn test_velocity_with_admittance_changes_tbem_matrix() {
        let (mut elements, nodes) = make_simple_mesh();
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);

        for element in &mut elements {
            element.boundary_condition =
                BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
        }
        let rigid = build_tbem_system(&elements, &nodes, &physics);

        elements[0].boundary_condition = BoundaryCondition::VelocityWithAdmittance {
            velocity: vec![Complex64::new(0.0, 0.0)],
            admittance: Complex64::new(0.15, -0.02),
        };
        let admittance = build_tbem_system(&elements, &nodes, &physics);

        let delta: f64 = admittance
            .matrix
            .as_slice()
            .iter()
            .zip(rigid.matrix.as_slice().iter())
            .map(|(a, b)| (*a - *b).norm())
            .sum();
        assert!(
            delta > 1e-10,
            "VelocityWithAdmittance must modify the assembled TBEM matrix"
        );
        assert!(
            admittance.rhs.iter().all(|v| v.norm() < 1e-10),
            "zero prescribed velocity with admittance should not inject a RHS"
        );
    }

    #[test]
    fn test_admittance_free_term_affects_diagonal() {
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
        let gamma = Complex64::new(physics.gamma(), 0.0);
        let tau = Complex64::new(physics.tau, 0.0);
        let beta = physics.burton_miller_beta();
        let i_omega_rho = Complex64::new(0.0, physics.pressure_factor);
        let y = Complex64::new(0.2, 0.05);
        let mut system = TbemSystem::new(1);

        add_free_terms(
            &mut system,
            0,
            0,
            &[Complex64::new(0.0, 0.0)],
            gamma,
            tau,
            beta,
            i_omega_rho,
            Some(y),
        );

        let expected = -gamma * 0.5 - y * i_omega_rho * beta * tau * 0.5;
        assert!(
            (system.matrix[[0, 0]] - expected).norm() < 1e-12,
            "admittance jump term was not added to the TBEM diagonal"
        );
    }

    #[test]
    fn test_transfer_admittance_changes_tbem_matrix() {
        let (mut elements, nodes) = make_simple_mesh();
        elements[0].boundary_condition =
            BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
        let rigid = build_tbem_system(&elements, &nodes, &physics);

        elements[0].boundary_condition = BoundaryCondition::TransferAdmittance {
            admittance: Complex64::new(0.1, 0.0),
        };
        let with_transfer = build_tbem_system(&elements, &nodes, &physics);

        let max_delta = with_transfer
            .matrix
            .as_slice()
            .iter()
            .zip(rigid.matrix.as_slice().iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f64::max);
        assert!(
            max_delta > 1e-12,
            "TransferAdmittance must modify the TBEM matrix"
        );
    }

    /// 8-element mesh with a mix of near-adjacent and well-separated pairs,
    /// so `build_tbem_system_stabilized_with_beta_simd` exercises both its
    /// scalar fallback (near-field) and its `regular_pair_batch` path
    /// (far-field).
    fn make_mixed_distance_mesh() -> (Vec<Element>, Vec<[f64; 3]>) {
        // Cluster A near the origin (4 elements), cluster B at x=8 (4
        // elements). Within a cluster, neighbour distance is ~1 element
        // size → ratio < 3, scalar. Cross-cluster distance is ~8 element
        // sizes → ratio > 3, batch.
        let mut nodes: Vec<[f64; 3]> = Vec::new();
        let mut elements: Vec<Element> = Vec::new();

        for cluster in 0..2 {
            let xshift = (cluster as f64) * 8.0;
            for j in 0..4 {
                let yshift = (j as f64) * 1.5;
                let base = nodes.len();
                nodes.push([xshift + 0.0, yshift + 0.0, 0.0]);
                nodes.push([xshift + 1.0, yshift + 0.0, 0.0]);
                nodes.push([xshift + 0.5, yshift + 1.0, 0.0]);
                let dof_idx = cluster * 4 + j;
                elements.push(Element {
                    connectivity: vec![base, base + 1, base + 2],
                    element_type: ElementType::Tri3,
                    property: ElementProperty::Surface,
                    normal: [0.0, 0.0, 1.0],
                    node_normals: vec![[0.0; 3]; 3],
                    center: [xshift + 0.5, yshift + 1.0 / 3.0, 0.0],
                    area: 0.5,
                    boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
                    group: 0,
                    dof_addresses: vec![dof_idx],
                });
            }
        }
        (elements, nodes)
    }

    #[test]
    fn build_tbem_simd_matches_scalar_within_quadrature_delta() {
        let (elements, nodes) = make_mixed_distance_mesh();
        let physics = PhysicsParams::new(40.0, 343.0, 1.21, false);
        let beta = Complex64::new(0.0, 1.0 / physics.wave_number);

        let scalar =
            build_tbem_system_stabilized_with_beta(&elements, &nodes, &physics, -1.0, beta);
        let simd =
            build_tbem_system_stabilized_with_beta_simd(&elements, &nodes, &physics, -1.0, beta);

        assert_eq!(scalar.matrix.shape(), simd.matrix.shape());
        assert_eq!(scalar.rhs.len(), simd.rhs.len());

        // The two paths use identical math except that the batch tail runs
        // 4-pt Tri3 quadrature where the scalar path runs 13-pt. The error
        // gap at ratio ≥ 3 is bounded by ~(0.5/3)^9 ≈ 1.4e-7 per entry; the
        // accumulated row norm stays well under 1e-5. We allow 1e-4 to
        // remain robust to mesh refinement of the test fixture without
        // masking algorithmic regressions.
        let m_diff: f64 = scalar
            .matrix
            .as_slice()
            .iter()
            .zip(simd.matrix.as_slice().iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let m_ref: f64 = scalar
            .matrix
            .as_slice()
            .iter()
            .map(|v| v.norm_sqr())
            .sum::<f64>()
            .sqrt();
        let m_rel = m_diff / m_ref.max(1e-30);
        assert!(
            m_rel < 1e-4,
            "stabilised-simd matrix vs scalar relative diff {m_rel:.3e} > 1e-4 \
             — algorithmic regression beyond the 13-pt → 4-pt quadrature delta"
        );

        // RHS is identical (no incident contribution from BC=0 elements).
        for (a, b) in scalar.rhs.iter().zip(simd.rhs.iter()) {
            assert!((a - b).norm() < 1e-12, "rhs entries diverge: {a} vs {b}");
        }
    }

    #[test]
    fn wavenumber_derivative_matches_central_difference_for_constant_panels() {
        let (mut elements, nodes) = make_simple_mesh();
        elements[1].boundary_condition = BoundaryCondition::VelocityWithAdmittance {
            velocity: vec![Complex64::new(0.2, -0.1)],
            admittance: Complex64::new(0.01, -0.02),
        };
        let physics = PhysicsParams::new(800.0, 343.0, 1.21, false);
        let sign = 1.0;
        let beta_scale = 2.0;
        let dk = 1.0e-5;
        let df = dk * physics.speed_of_sound / (2.0 * std::f64::consts::PI);
        let plus = PhysicsParams::new(
            physics.frequency + df,
            physics.speed_of_sound,
            physics.density,
            false,
        );
        let minus = PhysicsParams::new(
            physics.frequency - df,
            physics.speed_of_sound,
            physics.density,
            false,
        );

        let derivative = build_tbem_system_stabilized_wavenumber_derivative(
            &elements, &nodes, &physics, sign, beta_scale,
        );
        let plus_system = build_tbem_system_stabilized(&elements, &nodes, &plus, sign, beta_scale);
        let minus_system =
            build_tbem_system_stabilized(&elements, &nodes, &minus, sign, beta_scale);

        let mut max_matrix_error: f64 = 0.0;
        for ((&p, &m), &d) in plus_system
            .matrix
            .as_slice()
            .iter()
            .zip(minus_system.matrix.as_slice())
            .zip(derivative.matrix.as_slice())
        {
            max_matrix_error = max_matrix_error.max(((p - m) / (2.0 * dk) - d).norm());
        }
        let mut max_rhs_error: f64 = 0.0;
        for ((&p, &m), &d) in plus_system
            .rhs
            .iter()
            .zip(minus_system.rhs.iter())
            .zip(derivative.rhs.iter())
        {
            max_rhs_error = max_rhs_error.max(((p - m) / (2.0 * dk) - d).norm());
        }
        assert!(
            max_matrix_error < 2.0e-5,
            "matrix derivative error: {max_matrix_error:e}"
        );
        assert!(
            max_rhs_error < 2.0e-5,
            "RHS derivative error: {max_rhs_error:e}"
        );
    }

    #[test]
    fn density_derivative_matches_central_difference_for_constant_panels() {
        let (mut elements, nodes) = make_simple_mesh();
        elements[1].boundary_condition = BoundaryCondition::VelocityWithAdmittance {
            velocity: vec![Complex64::new(0.2, -0.1)],
            admittance: Complex64::new(0.01, -0.02),
        };
        let physics = PhysicsParams::new(800.0, 343.0, 1.21, false);
        let delta = 1.0e-5;
        let plus_physics = PhysicsParams::new(
            physics.frequency,
            physics.speed_of_sound,
            physics.density + delta,
            false,
        );
        let minus_physics = PhysicsParams::new(
            physics.frequency,
            physics.speed_of_sound,
            physics.density - delta,
            false,
        );
        let plus = build_tbem_system_stabilized(&elements, &nodes, &plus_physics, 1.0, 1.0);
        let minus = build_tbem_system_stabilized(&elements, &nodes, &minus_physics, 1.0, 1.0);
        let derivative =
            build_tbem_system_stabilized_density_derivative(&elements, &nodes, &physics, 1.0, 1.0);
        assert!(
            derivative
                .matrix
                .as_slice()
                .iter()
                .any(|value| value.norm() > 1.0e-8),
            "density derivative must be nonzero for an admittance boundary"
        );
        let mut max_matrix_error = 0.0_f64;
        for ((p, m), d) in plus
            .matrix
            .as_slice()
            .iter()
            .zip(minus.matrix.as_slice())
            .zip(derivative.matrix.as_slice())
        {
            max_matrix_error = max_matrix_error.max(((p - m) / (2.0 * delta) - d).norm());
        }
        let mut max_rhs_error = 0.0_f64;
        for ((p, m), d) in plus
            .rhs
            .iter()
            .zip(minus.rhs.iter())
            .zip(derivative.rhs.iter())
        {
            max_rhs_error = max_rhs_error.max(((p - m) / (2.0 * delta) - d).norm());
        }
        assert!(
            max_matrix_error < 2.0e-5,
            "density matrix derivative error: {max_matrix_error:e}"
        );
        assert!(
            max_rhs_error < 2.0e-5,
            "density RHS derivative error: {max_rhs_error:e}"
        );
    }

    #[test]
    fn objective_wavenumber_gradient_contract_checks_dimensions() {
        let mut derivative = TbemSystem::new(2);
        derivative.matrix[[0, 0]] = Complex64::new(1.0, 2.0);
        derivative.matrix[[0, 1]] = Complex64::new(-0.5, 0.25);
        derivative.matrix[[1, 0]] = Complex64::new(0.75, -1.0);
        derivative.matrix[[1, 1]] = Complex64::new(2.0, -0.5);
        derivative.rhs = vec![Complex64::new(0.25, -0.5), Complex64::new(-1.0, 0.75)];
        let forward = [Complex64::new(0.5, -0.25), Complex64::new(-0.75, 0.5)];
        let adjoint = [Complex64::new(1.0, 0.25), Complex64::new(-0.5, 1.5)];
        let expected = adjoint
            .iter()
            .enumerate()
            .map(|(row, &lambda)| {
                let action = derivative.rhs[row]
                    - (0..2)
                        .map(|col| derivative.matrix[[row, col]] * forward[col])
                        .sum::<Complex64>();
                (lambda.conj() * action).re
            })
            .sum::<f64>();
        let actual = tbem_wavenumber_objective_gradient(&derivative, &forward, &adjoint)
            .expect("matching vectors must be accepted");
        assert!((actual - expected).abs() < 1.0e-14);
        assert!(matches!(
            tbem_wavenumber_objective_gradient(&derivative, &[forward[0]], &adjoint),
            Err(TbemDerivativeError::DimensionMismatch { .. })
        ));
        let non_finite = [Complex64::new(f64::NAN, 0.0), forward[1]];
        assert!(matches!(
            tbem_wavenumber_objective_gradient(&derivative, &non_finite, &adjoint),
            Err(TbemDerivativeError::NonFinite { .. })
        ));
    }

    #[test]
    fn impedance_speed_derivative_matches_central_difference_for_constant_panels() {
        let (mut elements, nodes) = make_simple_mesh();
        let physics = PhysicsParams::new(800.0, 343.0, 1.21, false);
        let delta = 1.0e-5;
        let velocity = vec![Complex64::new(0.2, -0.1)];
        let admittance = |speed: f64| Complex64::new(1.0 / (physics.density * speed), 0.0);
        elements[1].boundary_condition = BoundaryCondition::VelocityWithAdmittance {
            velocity: velocity.clone(),
            admittance: admittance(physics.speed_of_sound),
        };
        let mut plus_elements = elements.clone();
        plus_elements[1].boundary_condition = BoundaryCondition::VelocityWithAdmittance {
            velocity: velocity.clone(),
            admittance: admittance(physics.speed_of_sound + delta),
        };
        let mut minus_elements = elements.clone();
        minus_elements[1].boundary_condition = BoundaryCondition::VelocityWithAdmittance {
            velocity,
            admittance: admittance(physics.speed_of_sound - delta),
        };
        let plus_physics = PhysicsParams::new(
            physics.frequency,
            physics.speed_of_sound + delta,
            physics.density,
            false,
        );
        let minus_physics = PhysicsParams::new(
            physics.frequency,
            physics.speed_of_sound - delta,
            physics.density,
            false,
        );
        let plus = build_tbem_system_stabilized(&plus_elements, &nodes, &plus_physics, 1.0, 1.0);
        let minus = build_tbem_system_stabilized(&minus_elements, &nodes, &minus_physics, 1.0, 1.0);
        let derivative = build_tbem_system_stabilized_impedance_speed_derivative(
            &elements, &nodes, &physics, 1.0, 1.0,
        );

        let max_matrix_error = plus
            .matrix
            .as_slice()
            .iter()
            .zip(minus.matrix.as_slice())
            .zip(derivative.matrix.as_slice())
            .map(|((&p, &m), &d)| ((p - m) / (2.0 * delta) - d).norm())
            .fold(0.0_f64, f64::max);
        let max_rhs_error = plus
            .rhs
            .iter()
            .zip(minus.rhs.iter())
            .zip(derivative.rhs.iter())
            .map(|((&p, &m), &d)| ((p - m) / (2.0 * delta) - d).norm())
            .fold(0.0_f64, f64::max);
        assert!(
            max_matrix_error < 2.0e-5,
            "impedance speed matrix derivative error: {max_matrix_error:e}"
        );
        assert!(
            max_rhs_error < 2.0e-5,
            "impedance speed RHS derivative error: {max_rhs_error:e}"
        );
    }

    #[test]
    fn test_count_dofs() {
        let (elements, _) = make_simple_mesh();
        assert_eq!(count_dofs(&elements), 2);
    }

    #[test]
    fn test_build_tbem_system_with_beta_small_mesh_uses_parallel_path() {
        // Regression guard: build_tbem_system_with_beta must work for tiny
        // meshes even after the dead serial path is removed.
        let (elements, nodes) = make_simple_mesh();
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
        let beta = physics.burton_miller_beta_scaled(2.0);

        let system = build_tbem_system_with_beta(&elements, &nodes, &physics, beta);

        assert_eq!(system.num_dofs, 2);
        assert!(system.matrix[[0, 0]].norm() > 1e-15);
        assert!(system.matrix[[1, 1]].norm() > 1e-15);
    }

    #[test]
    fn test_tbem_parallel_assembly_is_thread_count_independent() {
        let (elements, nodes) = make_simple_mesh();
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
        let beta = physics.burton_miller_beta_scaled(2.0);

        let assemble = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| build_tbem_system_with_beta_parallel(&elements, &nodes, &physics, beta))
        };
        let serial = assemble(1);
        let parallel = assemble(4);

        assert_eq!(serial.matrix.as_slice(), parallel.matrix.as_slice());
        assert_eq!(serial.rhs, parallel.rhs);
    }

    #[test]
    #[should_panic(expected = "unique source element for DOF 0")]
    fn test_tbem_parallel_assembly_rejects_duplicate_source_dofs() {
        let (mut elements, nodes) = make_simple_mesh();
        elements[1].dof_addresses[0] = 0;
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);

        let _ = build_tbem_system_with_beta_parallel(
            &elements,
            &nodes,
            &physics,
            physics.burton_miller_beta(),
        );
    }

    #[test]
    fn test_row_sum_correction_preserves_solution_accuracy() {
        // A finite-frequency Helmholtz combined operator does not annihilate
        // constants. The legacy correction must therefore remain a no-op;
        // forcing its row sums to zero made the matrix nearly singular.
        // Asserted through the supported builder (the legacy
        // `apply_row_sum_correction` is deprecated; the QA binary
        // `bem_row_sum_correction` pins its no-op contract directly).
        use crate::core::mesh::generators::generate_icosphere_mesh;

        // Closed surface (icosphere subdiv-0: 20 triangles) where row-sum
        // correction is physically meaningful.
        let mesh = generate_icosphere_mesh(1.0, 0);
        let mut elements = mesh.elements.clone();
        for (i, e) in elements.iter_mut().enumerate() {
            e.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
            e.dof_addresses = vec![i];
        }

        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);

        let plain = build_tbem_system(&elements, &mesh.nodes, &physics);
        let (system, diagnostic) = build_tbem_system_corrected(&elements, &mesh.nodes, &physics);
        assert!(diagnostic.is_finite());
        assert_eq!(system.matrix.as_slice(), plain.matrix.as_slice());
    }

    #[test]
    fn test_try_merge_rejects_size_mismatch() {
        let mut a = TbemSystem::new(2);
        let b = TbemSystem::new(3);
        assert_eq!(a.try_merge(b), Err(MergeSizeError { left: 2, right: 3 }));
        // Equal sizes merge element-wise.
        let mut c = TbemSystem::new(2);
        c.rhs[0] = Complex64::new(1.0, 0.0);
        let mut d = TbemSystem::new(2);
        d.rhs[0] = Complex64::new(2.0, 0.0);
        d.matrix[[1, 1]] = Complex64::new(0.5, 0.0);
        c.try_merge(d).expect("equal-size merge");
        assert_eq!(c.rhs[0], Complex64::new(3.0, 0.0));
        assert_eq!(c.matrix[[1, 1]], Complex64::new(0.5, 0.0));
    }

    #[test]
    fn test_try_build_rejects_dof_without_source_element() {
        // Gapped DOF numbering: max DOF index is 2 but no element owns DOF 1.
        // The legacy builder would leave a silent zero row; the checked
        // builder must report it before any kernel evaluation.
        let (mut elements, nodes) = make_simple_mesh();
        elements[1].dof_addresses = vec![2];
        let physics = PhysicsParams::new(800.0, 343.0, 1.21, false);
        let beta = physics.burton_miller_beta();
        assert_eq!(
            try_build_tbem_system_with_beta(&elements, &nodes, &physics, beta).err(),
            Some(MissingSourceDof { dof: 1 })
        );
        // Contiguous numbering assembles fine through the same entry point.
        let (elements_ok, _) = make_simple_mesh();
        assert!(try_build_tbem_system_with_beta(&elements_ok, &nodes, &physics, beta).is_ok());
    }

    #[test]
    fn test_get_element_coords() {
        let (elements, nodes) = make_simple_mesh();
        let coords = get_element_coords(&elements[0], &nodes);

        assert_eq!(coords.shape(), (3, 3));
        assert!((coords[[0, 0]] - 0.0).abs() < 1e-10); // node 0
        assert!((coords[[1, 0]] - 1.0).abs() < 1e-10); // node 1
    }

    fn make_simple_mesh_linear() -> (Vec<Element>, Vec<[f64; 3]>) {
        // Same 2-element mesh but with linear DOF addresses (3 per element)
        let nodes: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [1.5, 1.0, 0.0],
        ];

        let elem0 = Element {
            connectivity: vec![0, 1, 2],
            element_type: ElementType::Tri3,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0; 3]; 3],
            center: [0.5, 1.0 / 3.0, 0.0],
            area: 0.5,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
            group: 0,
            dof_addresses: vec![0, 1, 2], // 3 DOFs for linear
        };

        let elem1 = Element {
            connectivity: vec![1, 3, 2],
            element_type: ElementType::Tri3,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0; 3]; 3],
            center: [1.0, 2.0 / 3.0, 0.0],
            area: 0.5,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
            group: 0,
            dof_addresses: vec![3, 4, 5], // 3 DOFs for linear
        };

        (vec![elem0, elem1], nodes)
    }

    #[test]
    fn test_build_tbem_system_linear() {
        let (elements, nodes) = make_simple_mesh_linear();
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
        let beta = physics.burton_miller_beta();

        let system = build_tbem_system_linear(&elements, &nodes, &physics, beta);

        // 2 elements * 3 DOFs each = 6 total DOFs
        assert_eq!(system.num_dofs, 6);
        assert_eq!(system.matrix.shape(), (6, 6));
        assert_eq!(system.rhs.len(), 6);

        // Matrix should have non-zero diagonal entries
        for i in 0..6 {
            assert!(
                system.matrix[[i, i]].norm() > 1e-15,
                "Diagonal [{i},{i}] was zero"
            );
        }
    }

    #[test]
    fn test_linear_rhs_nonzero_for_velocity_bc() {
        // A mesh with nonzero prescribed velocity BC must produce a nonzero RHS
        // in the linear assembler (previously the BC values were silently dropped).
        let nodes: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

        let elem = Element {
            connectivity: vec![0, 1, 2],
            element_type: ElementType::Tri3,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0; 3]; 3],
            center: [1.0 / 3.0, 1.0 / 3.0, 0.0],
            area: 0.5,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(1.0, 0.0)]),
            group: 0,
            dof_addresses: vec![0, 1, 2],
        };

        let elements = vec![elem];
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
        let beta = physics.burton_miller_beta();

        let system = build_tbem_system_linear(&elements, &nodes, &physics, beta);

        // With nonzero prescribed velocity, the RHS must be nonzero.
        let rhs_norm: f64 = system.rhs.iter().map(|v| v.norm()).sum();
        assert!(
            rhs_norm > 1e-10,
            "RHS is zero for nonzero Velocity BC (norm = {rhs_norm})"
        );
    }

    #[test]
    fn test_count_dofs_linear() {
        let (elements, _) = make_simple_mesh_linear();
        assert_eq!(count_dofs(&elements), 6);
    }

    #[test]
    fn test_near_block_beta_not_double_scaled() {
        // compute_near_block_stabilized receives `beta` already scaled by
        // physics.burton_miller_beta_scaled(beta_scale).  It must not apply
        // beta_scale a second time.  With beta_scale = 2.0 the bug gives
        // beta_eff = 4.0 * i/k instead of 2.0 * i/k.
        let (elements, nodes) = make_simple_mesh();
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
        let beta_scale = 2.0;
        let beta = physics.burton_miller_beta_scaled(beta_scale);

        let all_coords: Vec<DenseMatrix<f64>> = elements
            .iter()
            .map(|e| get_element_coords(e, &nodes))
            .collect();

        let gamma = Complex64::new(physics.gamma(), 0.0);
        let tau = Complex64::new(physics.tau, 0.0);
        let sign = 1.0;

        // Build the near-field block with the (possibly buggy) function.
        // Use is_self=false so no free terms are added — we only want to
        // compare the pure integral contribution where beta appears.
        let block = crate::core::assembly::common::compute_near_block_stabilized(
            &elements,
            &all_coords,
            &[0],
            &[1],
            &physics,
            gamma,
            tau,
            beta,
            false,
            1.0, // sign
        );

        // Build the same pair using the dense path (which uses beta correctly)
        let dense_system =
            build_tbem_system_stabilized_with_beta(&elements, &nodes, &physics, sign, beta);

        // The off-diagonal entry [0,1] must match between the two paths
        assert!(
            (block[[0, 0]] - dense_system.matrix[[0, 1]]).norm() < 1e-12,
            "compute_near_block_stabilized double-scales beta: got {} expected {}",
            block[[0, 0]],
            dense_system.matrix[[0, 1]]
        );
    }

    /// Phase 1.3 smoke test: `build_tbem_system_with_beta_metal` must
    /// produce a system matrix and RHS that match the existing parallel
    /// CPU path within the GPU's f32 precision floor on every entry.
    /// Skips when Metal is not initialisable on the host (CI / non-Apple).
    #[test]
    #[cfg(feature = "metal")]
    fn test_build_tbem_system_with_beta_metal_matches_cpu() {
        use crate::core::mesh::generators::generate_icosphere_mesh;
        use sonium_compute::MetalBackend;

        let backend = match MetalBackend::new() {
            Ok(b) => b,
            Err(err) => {
                eprintln!("Metal unavailable, skipping tbem-metal parity test: {err}");
                return;
            }
        };

        // Subdivision-2 icosphere: 320 elements — small enough to make the
        // parity comparison cheap, large enough that most pairs land in the
        // far-field GPU bucket (so the test actually exercises the kernel).
        let mesh = generate_icosphere_mesh(1.0, 2);
        let mut elements = mesh.elements.clone();
        for (i, e) in elements.iter_mut().enumerate() {
            e.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
            e.dof_addresses = vec![i];
        }
        // Set element 0 to a non-zero velocity so the RHS is non-trivial.
        elements[0].boundary_condition =
            BoundaryCondition::Velocity(vec![Complex64::new(1.0, 0.0)]);

        let physics = PhysicsParams::new(343.0 / (2.0 * std::f64::consts::PI), 343.0, 1.21, false);
        let beta = physics.burton_miller_beta();

        let cpu = build_tbem_system_with_beta_parallel(&elements, &mesh.nodes, &physics, beta);
        let gpu =
            build_tbem_system_with_beta_metal(&elements, &mesh.nodes, &physics, beta, &backend);

        assert_eq!(
            cpu.matrix.shape(),
            gpu.matrix.shape(),
            "matrix dims must agree"
        );
        assert_eq!(cpu.rhs.len(), gpu.rhs.len(), "rhs lens must agree");

        // f32 GPU vs f64 CPU on a Helmholtz kernel: per-pair ~1e-6 relative
        // error, accumulated across N≈320 entries per row, condition number
        // bounded by gamma·tau, beta. 1e-3 is the empirically-defensible
        // upper bound that fails fast on a real algorithmic regression
        // without flagging f32 round-off noise.
        let m_diff: f64 = cpu
            .matrix
            .as_slice()
            .iter()
            .zip(gpu.matrix.as_slice().iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let m_ref: f64 = cpu
            .matrix
            .as_slice()
            .iter()
            .map(|v| v.norm_sqr())
            .sum::<f64>()
            .sqrt();
        let m_rel = m_diff / m_ref.max(1e-30);

        let r_diff: f64 = cpu
            .rhs
            .iter()
            .zip(gpu.rhs.iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let r_ref: f64 = cpu.rhs.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
        let r_rel = r_diff / r_ref.max(1e-30);

        assert!(
            m_rel < 1e-3,
            "Metal vs CPU matrix relative diff {m_rel:.3e} > 1e-3"
        );
        assert!(
            r_rel < 1e-3,
            "Metal vs CPU RHS relative diff {r_rel:.3e} > 1e-3"
        );
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_build_tbem_system_with_beta_cuda_matches_cpu() {
        use crate::core::mesh::generators::generate_icosphere_mesh;
        use sonium_compute::CudaBackend;

        let backend = match CudaBackend::new_default() {
            Ok(b) => b,
            Err(err) => {
                if std::env::var("SONIUM_REQUIRE_CUDA")
                    .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE"))
                {
                    panic!("CUDA required by SONIUM_REQUIRE_CUDA=1: {err}");
                }
                eprintln!("CUDA unavailable, skipping tbem-cuda parity test: {err}");
                return;
            }
        };
        if !backend.has_bem_regular_pair_c64() {
            if std::env::var("SONIUM_REQUIRE_CUDA")
                .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE"))
            {
                panic!("CUDA Complex64 BEM regular-pair kernel is unavailable");
            }
            eprintln!(
                "CUDA Complex64 BEM regular-pair kernel unavailable, skipping tbem-cuda parity test"
            );
            return;
        }

        let mesh = generate_icosphere_mesh(1.0, 2);
        let mut elements = mesh.elements.clone();
        for (i, e) in elements.iter_mut().enumerate() {
            e.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
            e.dof_addresses = vec![i];
        }
        elements[0].boundary_condition =
            BoundaryCondition::Velocity(vec![Complex64::new(1.0, 0.0)]);

        let physics = PhysicsParams::new(343.0 / (2.0 * std::f64::consts::PI), 343.0, 1.21, false);
        let beta = physics.burton_miller_beta();

        backend.reset_runtime_stats();
        let cpu = build_tbem_system_with_beta_parallel(&elements, &mesh.nodes, &physics, beta);
        let gpu = build_tbem_system_with_beta_cuda_checked(
            &elements,
            &mesh.nodes,
            &physics,
            beta,
            &backend,
        )
        .expect("checked CUDA TBEM assembly");

        assert_eq!(
            cpu.matrix.shape(),
            gpu.matrix.shape(),
            "matrix dims must agree"
        );
        assert_eq!(cpu.rhs.len(), gpu.rhs.len(), "rhs lens must agree");

        let m_diff: f64 = cpu
            .matrix
            .as_slice()
            .iter()
            .zip(gpu.matrix.as_slice().iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let m_ref: f64 = cpu
            .matrix
            .as_slice()
            .iter()
            .map(|v| v.norm_sqr())
            .sum::<f64>()
            .sqrt();
        let m_rel = m_diff / m_ref.max(1e-30);

        let r_diff: f64 = cpu
            .rhs
            .iter()
            .zip(gpu.rhs.iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let r_ref: f64 = cpu.rhs.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
        let r_rel = r_diff / r_ref.max(1e-30);

        assert!(
            m_rel < 1e-9,
            "CUDA vs CPU matrix relative diff {m_rel:.3e} > 1e-9"
        );
        assert!(
            r_rel < 1e-9,
            "CUDA vs CPU RHS relative diff {r_rel:.3e} > 1e-9"
        );

        let stats = backend.runtime_stats();
        assert!(
            stats.kernel_launches >= 1,
            "CUDA TBEM must launch a pair kernel"
        );
        assert!(stats.transfer.h2d_operations >= 1);
        assert!(stats.transfer.d2h_operations >= 1);

        // Exercise the assembled forward, algebraic-transpose, and Hermitian
        // products on the same 3-D fixture.  These products are computed
        // independently here instead of delegating to a BEM operator wrapper,
        // so a wrapper can neither hide an orientation nor conjugation error.
        let n = cpu.matrix.shape().0;
        let probe: Vec<Complex64> = (0..n)
            .map(|index| Complex64::new(0.25 + index as f64 * 0.003, -0.1 + index as f64 * 0.002))
            .collect();
        let product = |matrix: &DenseMatrix<Complex64>,
                       action: fn(Complex64, Complex64) -> Complex64| {
            (0..n)
                .map(|row| {
                    (0..n)
                        .map(|column| action(matrix[[row, column]], probe[column]))
                        .sum::<Complex64>()
                })
                .collect::<Vec<_>>()
        };
        let forward_cpu = product(&cpu.matrix, |entry, value| entry * value);
        let forward_gpu = product(&gpu.matrix, |entry, value| entry * value);
        let transpose_cpu = (0..n)
            .map(|row| {
                (0..n)
                    .map(|column| cpu.matrix[[column, row]] * probe[column])
                    .sum::<Complex64>()
            })
            .collect::<Vec<_>>();
        let transpose_gpu = (0..n)
            .map(|row| {
                (0..n)
                    .map(|column| gpu.matrix[[column, row]] * probe[column])
                    .sum::<Complex64>()
            })
            .collect::<Vec<_>>();
        let hermitian_cpu = (0..n)
            .map(|row| {
                (0..n)
                    .map(|column| cpu.matrix[[column, row]].conj() * probe[column])
                    .sum::<Complex64>()
            })
            .collect::<Vec<_>>();
        let hermitian_gpu = (0..n)
            .map(|row| {
                (0..n)
                    .map(|column| gpu.matrix[[column, row]].conj() * probe[column])
                    .sum::<Complex64>()
            })
            .collect::<Vec<_>>();
        for (label, expected, actual) in [
            ("forward", forward_cpu, forward_gpu),
            ("transpose", transpose_cpu, transpose_gpu),
            ("hermitian", hermitian_cpu, hermitian_gpu),
        ] {
            let reference = expected
                .iter()
                .map(|value| value.norm_sqr())
                .sum::<f64>()
                .sqrt();
            let error = expected
                .iter()
                .zip(actual.iter())
                .map(|(left, right)| (*left - *right).norm_sqr())
                .sum::<f64>()
                .sqrt()
                / reference.max(1.0e-30);
            assert!(
                error < 1.0e-3,
                "CUDA {label} product relative diff {error:.3e}"
            );
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_tbem_admission_rejects_unsupported_and_invalid_inputs_without_backend_work() {
        use crate::core::types::ElementType;

        let (mut elements, mut nodes) = make_simple_mesh();
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
        let beta = physics.burton_miller_beta();

        elements[0].element_type = ElementType::Quad4;
        elements[0].connectivity.push(3);
        let unsupported =
            super::cuda_path::validate_cuda_tbem_inputs(&elements, &nodes, &physics, beta)
                .expect_err("CUDA admission must reject Quad4 explicitly");
        assert!(matches!(
            unsupported,
            super::cuda_path::CudaTbemAssemblyError::UnsupportedElement {
                element: 0,
                element_type: "Quad4",
                dofs: 1,
            }
        ));

        nodes[1][0] = f64::NAN;
        let invalid =
            super::cuda_path::validate_cuda_tbem_inputs(&elements[1..], &nodes, &physics, beta)
                .expect_err("non-finite geometry must be rejected before CUDA work");
        assert!(matches!(
            invalid,
            super::cuda_path::CudaTbemAssemblyError::InvalidInput { .. }
        ));
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_tbem_launch_failure_is_typed_and_releases_transient_buffers() {
        use sonium_compute::{CudaBackend, CudaFailureInjectionPoint};

        let backend = match CudaBackend::new_default() {
            Ok(backend) => backend,
            Err(error) => {
                if std::env::var("SONIUM_REQUIRE_CUDA")
                    .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE"))
                {
                    panic!("CUDA required by SONIUM_REQUIRE_CUDA=1: {error}");
                }
                eprintln!("CUDA unavailable, skipping injected TBEM failure test: {error}");
                return;
            }
        };
        if !backend.has_bem_regular_pair_c64() {
            if std::env::var("SONIUM_REQUIRE_CUDA")
                .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE"))
            {
                panic!("CUDA Complex64 BEM regular-pair kernel is unavailable");
            }
            eprintln!("CUDA Complex64 BEM regular-pair kernel unavailable, skipping");
            return;
        }

        let (elements, nodes) = make_simple_mesh();
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
        let beta = physics.burton_miller_beta();
        let baseline = backend.live_allocation_bytes();
        backend.inject_failure(CudaFailureInjectionPoint::Launch);

        let error = match build_tbem_system_with_beta_cuda_checked(
            &elements, &nodes, &physics, beta, &backend,
        ) {
            Ok(_) => panic!("injected CUDA launch failure must abort before publication"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            CudaTbemAssemblyError::Backend(sonium_compute::BackendError::Launch {
                operation: "cuda_launch",
                ..
            })
        ));
        backend
            .recover_after_error()
            .expect("recoverable launch failure must clear CUDA error state");
        assert_eq!(
            backend.live_allocation_bytes(),
            baseline,
            "transient CUDA assembly buffers must be released after a failed publication"
        );
    }
}
