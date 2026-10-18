//! Matrix-free operator for FEM systems
//!
//! Computes `y = A*x` on-the-fly by re-assembling element contributions
//! without forming the global matrix. This trades increased FLOPs per
//! iteration for dramatically reduced memory usage.
//!
//! # Usage
//!
//! ```ignore
//! let op = MatrixFreeOperator::new(&form, &space, &mesh, quad_order);
//! let y = op.apply(&x);
//! // Use with CG: sonium_solvers::cg(&op, &rhs, &config)
//! ```

use crate::forms::{BilinearForm, LocalMatrix};
use crate::mesh::{ElementType, Mesh};
use crate::quadrature::QuadratureRule;
use crate::spaces::{FunctionSpace, SpaceValidationError};
use sonium_solvers::traits::{AdjointCapability, LinearOperator};
use std::collections::HashMap;

/// Failure while admitting or applying a matrix-free FEM operator.
#[derive(Debug, Clone, PartialEq)]
pub enum MatrixFreeError {
    /// The mesh/function-space DOF contract is invalid.
    InvalidSpace(SpaceValidationError),
    /// Matrix-free integration requires a positive quadrature order.
    InvalidQuadratureOrder {
        /// Requested quadrature order.
        order: usize,
    },
    /// A local map references a nonexistent global DOF.
    InvalidDofIndex {
        /// Element containing the invalid map.
        element: usize,
        /// Invalid global DOF index.
        dof: usize,
    },
    /// An input or output vector has the wrong global dimension.
    DimensionMismatch {
        /// Required operator dimension.
        expected: usize,
        /// Supplied vector dimension.
        actual: usize,
    },
}

impl std::fmt::Display for MatrixFreeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSpace(error) => write!(f, "invalid matrix-free function space: {error}"),
            Self::InvalidQuadratureOrder { order } => {
                write!(f, "quadrature order must be positive (got {order})")
            }
            Self::InvalidDofIndex { element, dof } => {
                write!(
                    f,
                    "element {element} references global DOF {dof} outside operator"
                )
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "vector dimension {actual} does not match operator dimension {expected}"
                )
            }
        }
    }
}

impl std::error::Error for MatrixFreeError {}

/// Matrix-free operator that computes `y = A*x` by element-level assembly.
///
/// Quadrature rules are cached per [`ElementType`] at construction. Dense
/// element matrices are integrated once, lazily on first use, and cached:
/// the borrowed form, space, and mesh are immutable for the operator's
/// lifetime, so re-integrating them on every application (the previous
/// behaviour) repeated identical work in every Krylov iteration. The
/// cache trades `sum_e (local_dofs_e)^2` storage for that repeated
/// quadrature and basis tabulation; it is still not high-order partial
/// assembly / sum factorization.
pub struct MatrixFreeOperator<'a, T, F, S>
where
    T: Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T> + Send + Sync,
    F: BilinearForm<T>,
    S: FunctionSpace,
{
    form: &'a F,
    space: &'a S,
    mesh: &'a Mesh,
    n_dofs: usize,
    validation: Result<(), MatrixFreeError>,
    quad_rules: HashMap<ElementType, QuadratureRule>,
    element_matrices: std::sync::OnceLock<Vec<LocalMatrix<T>>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T, F, S> MatrixFreeOperator<'a, T, F, S>
where
    T: Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T> + Send + Sync,
    F: BilinearForm<T>,
    S: FunctionSpace,
{
    pub fn new(form: &'a F, space: &'a S, mesh: &'a Mesh, quad_order: usize) -> Self {
        let n_dofs = space.num_global_dofs(mesh);
        let validation = Self::validate_setup(space, mesh, quad_order);
        let mut quad_rules = HashMap::new();
        for elem in &mesh.elements {
            quad_rules
                .entry(elem.element_type)
                .or_insert_with(|| QuadratureRule::new(elem.element_type, quad_order));
        }
        Self {
            form,
            space,
            mesh,
            n_dofs,
            validation,
            quad_rules,
            element_matrices: std::sync::OnceLock::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Construct a matrix-free operator after validating topology and
    /// quadrature admission. Untrusted meshes should use this boundary.
    pub fn try_new(
        form: &'a F,
        space: &'a S,
        mesh: &'a Mesh,
        quad_order: usize,
    ) -> Result<Self, MatrixFreeError> {
        let operator = Self::new(form, space, mesh, quad_order);
        operator.validation.clone().map(|()| operator)
    }

    fn validate_setup(space: &S, mesh: &Mesh, quad_order: usize) -> Result<(), MatrixFreeError> {
        if quad_order == 0 {
            return Err(MatrixFreeError::InvalidQuadratureOrder { order: quad_order });
        }
        space.validate(mesh).map_err(MatrixFreeError::InvalidSpace)
    }

    /// Validate the topology and quadrature configuration retained by this
    /// operator.
    pub fn validate(&self) -> Result<(), MatrixFreeError> {
        self.validation.clone()
    }

    pub fn num_dofs(&self) -> usize {
        self.n_dofs
    }

    /// Number of cached quadrature rules (one per distinct mesh element type).
    pub fn num_cached_quadrature_rules(&self) -> usize {
        self.quad_rules.len()
    }

    fn quad_rule(&self, elem_type: ElementType) -> &QuadratureRule {
        self.quad_rules
            .get(&elem_type)
            .unwrap_or_else(|| panic!("no cached quadrature rule for element type {elem_type:?}"))
    }

    /// Rebuild the dense element matrix for `elem_idx` into reused `scratch`.
    fn assemble_element_matrix(&self, elem_idx: usize, scratch: &mut LocalMatrix<T>) {
        let elem = &self.mesh.elements[elem_idx];
        let dofs = self.space.local_to_global(self.mesh, elem_idx);
        if scratch.dim != dofs.len() {
            *scratch = LocalMatrix::zeros(dofs.len());
        } else {
            scratch.values.fill(T::default());
        }
        let quad = self.quad_rule(elem.element_type);
        for qp in quad.iter() {
            let tab = self
                .space
                .tabulate(self.mesh, elem_idx, qp.xi(), qp.eta(), qp.zeta());
            self.form.integrate_point(&tab, &tab, qp.weight, scratch);
        }
    }

    /// Number of element matrices integrated and cached so far.
    pub fn num_cached_element_matrices(&self) -> usize {
        self.element_matrices.get().map_or(0, Vec::len)
    }

    /// Returns the cached dense element matrices, integrating them once.
    fn element_matrices(&self) -> &[LocalMatrix<T>] {
        self.element_matrices.get_or_init(|| {
            let mut scratch = LocalMatrix::zeros(0);
            (0..self.mesh.num_elements())
                .map(|elem_idx| {
                    self.assemble_element_matrix(elem_idx, &mut scratch);
                    scratch.clone()
                })
                .collect()
        })
    }

    /// Compute y = A*x element-by-element without forming the global matrix.
    /// Panics on invalid setup or dimensions; use `try_apply_vec` for errors.
    pub fn apply_vec(&self, x: &[T]) -> Vec<T> {
        self.try_apply_vec(x)
            .expect("invalid matrix-free application")
    }

    /// Checked matrix-free application.
    pub fn try_apply_vec(&self, x: &[T]) -> Result<Vec<T>, MatrixFreeError> {
        self.validate()?;
        if x.len() != self.n_dofs {
            return Err(MatrixFreeError::DimensionMismatch {
                expected: self.n_dofs,
                actual: x.len(),
            });
        }
        self.compute_apply_vec(x)
    }

    fn compute_apply_vec(&self, x: &[T]) -> Result<Vec<T>, MatrixFreeError> {
        let mut y = vec![T::default(); self.n_dofs];
        let matrices = self.element_matrices();

        for (elem_idx, local_mat) in matrices.iter().enumerate() {
            let dofs = self.space.local_to_global(self.mesh, elem_idx);

            // y += K_e * x_e (element matvec, scatter to global)
            for (i, &gi) in dofs.iter().enumerate() {
                if gi >= self.n_dofs {
                    return Err(MatrixFreeError::InvalidDofIndex {
                        element: elem_idx,
                        dof: gi,
                    });
                }
                let mut sum = T::default();
                for (j, &gj) in dofs.iter().enumerate() {
                    if gj >= self.n_dofs {
                        return Err(MatrixFreeError::InvalidDofIndex {
                            element: elem_idx,
                            dof: gj,
                        });
                    }
                    let k_ij = local_mat.get(i, j);
                    // sum += k_ij * x[gj]
                    // Need to multiply ComplexField values; use the trait
                    sum += k_ij * x[gj];
                }
                y[gi] += sum;
            }
        }
        Ok(y)
    }

    /// Apply the transposed element operator without forming a global matrix.
    fn apply_transpose_vec(&self, x: &[T]) -> Vec<T> {
        self.try_apply_transpose_vec(x)
            .expect("invalid matrix-free transpose application")
    }

    /// Checked transposed matrix-free application.
    pub fn try_apply_transpose_vec(&self, x: &[T]) -> Result<Vec<T>, MatrixFreeError> {
        self.validate()?;
        if x.len() != self.n_dofs {
            return Err(MatrixFreeError::DimensionMismatch {
                expected: self.n_dofs,
                actual: x.len(),
            });
        }
        self.compute_apply_transpose_vec(x)
    }

    fn compute_apply_transpose_vec(&self, x: &[T]) -> Result<Vec<T>, MatrixFreeError> {
        let mut y = vec![T::default(); self.n_dofs];
        let matrices = self.element_matrices();

        for (elem_idx, local_mat) in matrices.iter().enumerate() {
            let dofs = self.space.local_to_global(self.mesh, elem_idx);

            for (i, &gi) in dofs.iter().enumerate() {
                if gi >= self.n_dofs {
                    return Err(MatrixFreeError::InvalidDofIndex {
                        element: elem_idx,
                        dof: gi,
                    });
                }
                let mut sum = T::default();
                for (j, &gj) in dofs.iter().enumerate() {
                    if gj >= self.n_dofs {
                        return Err(MatrixFreeError::InvalidDofIndex {
                            element: elem_idx,
                            dof: gj,
                        });
                    }
                    sum += local_mat.get(j, i) * x[gj];
                }
                y[gi] += sum;
            }
        }
        Ok(y)
    }

    /// Extract the diagonal of the operator (for Jacobi preconditioning).
    /// Panics on invalid setup; use `try_diagonal` for errors.
    pub fn diagonal(&self) -> Vec<T> {
        self.try_diagonal().expect("invalid matrix-free diagonal")
    }

    /// Checked diagonal extraction for Jacobi preconditioning.
    pub fn try_diagonal(&self) -> Result<Vec<T>, MatrixFreeError> {
        self.validate()?;
        let mut diag = vec![T::default(); self.n_dofs];
        let matrices = self.element_matrices();

        for (elem_idx, local_mat) in matrices.iter().enumerate() {
            let dofs = self.space.local_to_global(self.mesh, elem_idx);

            for (i, &gi) in dofs.iter().enumerate() {
                if gi >= self.n_dofs {
                    return Err(MatrixFreeError::InvalidDofIndex {
                        element: elem_idx,
                        dof: gi,
                    });
                }
                diag[gi] += local_mat.get(i, i);
            }
        }
        Ok(diag)
    }
}

/// `LinearOperator<f64>` implementation for use with iterative solvers.
impl<F, S> LinearOperator<f64> for MatrixFreeOperator<'_, f64, F, S>
where
    F: BilinearForm<f64>,
    S: FunctionSpace,
{
    fn apply_into(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(
            y.len(),
            self.n_dofs,
            "matrix-free output dimension mismatch"
        );
        let result = self.apply_vec(x);
        y.copy_from_slice(&result);
    }

    fn apply_transpose_into(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(
            y.len(),
            self.n_dofs,
            "matrix-free transpose output dimension mismatch"
        );
        let result = self.apply_transpose_vec(x);
        y.copy_from_slice(&result);
    }

    fn transpose_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        AdjointCapability::Exact
    }

    fn num_rows(&self) -> usize {
        self.n_dofs
    }

    fn num_cols(&self) -> usize {
        self.n_dofs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::PolynomialDegree;
    use crate::forms::{BilinearForm, LaplacianForm, LocalMatrix};
    use crate::mesh::unit_square_triangles;
    use crate::spaces::{BasisTabulation, H1Space};

    struct NonsymmetricForm;

    impl BilinearForm<f64> for NonsymmetricForm {
        fn integrate_point(
            &self,
            _test: &BasisTabulation,
            _trial: &BasisTabulation,
            _weight: f64,
            local_matrix: &mut LocalMatrix<f64>,
        ) {
            for i in 0..local_matrix.dim {
                local_matrix.add(i, i, (i + 1) as f64);
                if i + 1 < local_matrix.dim {
                    local_matrix.add(i, i + 1, 0.5);
                }
            }
        }
    }

    #[test]
    fn test_matrix_free_matches_assembled() {
        let mesh = unit_square_triangles(4);
        let space = H1Space::new(PolynomialDegree::P1);
        let n = space.num_global_dofs(&mesh);

        // Matrix-free operator
        let op = MatrixFreeOperator::new(&LaplacianForm, &space, &mesh, 1);

        // Assembled matrix
        let triplet = crate::assembly::assemble_bilinear(&LaplacianForm, &space, &space, &mesh, 1);
        let csr = triplet.to_csr();

        // Compare A*x for a random-ish vector
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
        let y_mf = op.apply_vec(&x);

        let y_asm = csr.matvec(&x);

        for i in 0..n {
            assert!(
                (y_mf[i] - y_asm[i]).abs() < 1e-10,
                "Mismatch at DOF {i}: mf={}, asm={}",
                y_mf[i],
                y_asm[i]
            );
        }
    }

    #[test]
    fn test_diagonal_positive() {
        let mesh = unit_square_triangles(4);
        let space = H1Space::new(PolynomialDegree::P1);

        let op = MatrixFreeOperator::new(&LaplacianForm, &space, &mesh, 1);
        let diag = op.diagonal();

        for (i, &d) in diag.iter().enumerate() {
            assert!(
                d >= 0.0,
                "Diagonal[{i}] = {d} should be non-negative for Laplacian"
            );
        }
    }

    #[test]
    fn matrix_free_transpose_handles_nonsymmetric_forms() {
        let mesh = unit_square_triangles(2);
        let space = H1Space::new(PolynomialDegree::P1);
        let op = MatrixFreeOperator::new(&NonsymmetricForm, &space, &mesh, 1);
        let n = op.num_rows();
        let x: Vec<f64> = (0..n).map(|i| 0.25 + i as f64 * 0.1).collect();
        let y: Vec<f64> = (0..n).map(|i| 1.0 - i as f64 * 0.07).collect();

        let a_y = op.apply_vec(&y);
        let at_x = op.try_apply_transpose(&x).unwrap();
        let lhs: f64 = x.iter().zip(&a_y).map(|(x_i, a_i)| x_i * a_i).sum();
        let rhs: f64 = at_x.iter().zip(&y).map(|(at_i, y_i)| at_i * y_i).sum();

        assert_eq!(op.transpose_capability(), AdjointCapability::Exact);
        assert_eq!(op.hermitian_capability(), AdjointCapability::Exact);
        assert!(
            (lhs - rhs).abs() < 1e-10,
            "transpose inner-product mismatch: lhs={lhs}, rhs={rhs}"
        );
    }

    #[test]
    fn repeated_applies_reuse_cached_element_matrices() {
        let mesh = unit_square_triangles(4);
        let space = H1Space::new(PolynomialDegree::P1);
        let op = MatrixFreeOperator::new(&LaplacianForm, &space, &mesh, 1);
        let n = op.num_dofs();
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();

        assert_eq!(op.num_cached_element_matrices(), 0);
        let first = op.apply_vec(&x);
        assert_eq!(op.num_cached_element_matrices(), mesh.num_elements());
        let second = op.apply_vec(&x);
        assert_eq!(first, second);

        let triplet = crate::assembly::assemble_bilinear(&LaplacianForm, &space, &space, &mesh, 1);
        let assembled = triplet.to_csr().matvec(&x);
        for (index, (actual, expected)) in first.iter().zip(assembled.iter()).enumerate() {
            assert!((actual - expected).abs() < 1e-10, "dof {index}");
        }
    }

    #[test]
    fn checked_matrix_free_admission_rejects_bad_setup_and_vectors() {
        let mesh = unit_square_triangles(1);
        let space = H1Space::new(PolynomialDegree::P1);
        assert!(matches!(
            MatrixFreeOperator::try_new(&LaplacianForm, &space, &mesh, 0),
            Err(MatrixFreeError::InvalidQuadratureOrder { order: 0 })
        ));
        let operator = MatrixFreeOperator::try_new(&LaplacianForm, &space, &mesh, 1).unwrap();
        assert!(matches!(
            operator.try_apply_vec(&[]),
            Err(MatrixFreeError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            operator.try_diagonal(),
            Ok(diagonal) if diagonal.len() == operator.num_dofs()
        ));
    }

    #[test]
    fn checked_matrix_free_admission_rejects_malformed_mesh() {
        let mut mesh = unit_square_triangles(1);
        mesh.elements[0].nodes.truncate(2);
        let space = H1Space::new(PolynomialDegree::P1);
        assert!(matches!(
            MatrixFreeOperator::try_new(&LaplacianForm, &space, &mesh, 1),
            Err(MatrixFreeError::InvalidSpace(_))
        ));
    }

    #[test]
    fn convenience_operations_never_substitute_zeros_for_errors() {
        use std::panic::{AssertUnwindSafe, catch_unwind};
        let mesh = unit_square_triangles(1);
        let space = H1Space::new(PolynomialDegree::P1);
        let invalid = MatrixFreeOperator::new(&LaplacianForm, &space, &mesh, 0);
        let x = vec![1.0; invalid.num_dofs()];
        assert!(catch_unwind(|| invalid.apply_vec(&x)).is_err());
        assert!(catch_unwind(|| invalid.apply_transpose_vec(&x)).is_err());
        assert!(catch_unwind(|| invalid.diagonal()).is_err());
        assert!(invalid.try_apply(&x).is_err());
        assert!(invalid.try_apply_transpose(&x).is_err());
        assert!(invalid.try_apply_hermitian(&x).is_err());

        let valid = MatrixFreeOperator::new(&LaplacianForm, &space, &mesh, 1);
        assert!(catch_unwind(|| valid.apply_vec(&[])).is_err());
        assert!(catch_unwind(|| valid.apply_transpose_vec(&[])).is_err());
        for length in [0, valid.num_dofs() - 1, valid.num_dofs() + 1] {
            let mut output = vec![42.0; length];
            assert!(catch_unwind(AssertUnwindSafe(|| valid.apply_into(&x, &mut output))).is_err());
            assert_eq!(output, vec![42.0; length]);
            assert!(
                catch_unwind(AssertUnwindSafe(
                    || valid.apply_transpose_into(&x, &mut output)
                ))
                .is_err()
            );
            assert_eq!(output, vec![42.0; length]);
        }
        let mut output = vec![42.0; valid.num_dofs()];
        assert!(catch_unwind(AssertUnwindSafe(|| invalid.apply_into(&x, &mut output))).is_err());
        assert_eq!(output, vec![42.0; valid.num_dofs()]);
    }

    #[test]
    fn cached_quadrature_path_matches_assembled_operator() {
        // The per-element-type rule cache must reproduce the assembled
        // operator exactly, and repeated applications must reuse it
        // deterministically.
        let mesh = unit_square_triangles(4);
        let space = H1Space::new(PolynomialDegree::P1);
        let op = MatrixFreeOperator::new(&LaplacianForm, &space, &mesh, 2);
        assert_eq!(op.num_cached_quadrature_rules(), 1);

        let n = op.num_dofs();
        let triplet = crate::assembly::assemble_bilinear(&LaplacianForm, &space, &space, &mesh, 2);
        let csr = triplet.to_csr();

        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
        let y_mf = op.apply_vec(&x);
        let y_asm = csr.matvec(&x);
        for i in 0..n {
            assert!(
                (y_mf[i] - y_asm[i]).abs() < 1e-10,
                "Mismatch at DOF {i}: mf={}, asm={}",
                y_mf[i],
                y_asm[i]
            );
        }
        assert_eq!(y_mf, op.apply_vec(&x));
    }
}
