//! Generic finite element assembly over variational forms
//!
//! Assembles any [`BilinearForm`] or [`LinearForm`] into global sparse matrices
//! and vectors by looping over mesh elements in parallel.

use crate::forms::{
    BilinearForm, ElementForm, LinearForm, LocalMatrix, LocalVector, NonlinearElementForm,
};
use crate::mesh::{ElementType, Mesh};
use crate::parallelism::{ELEMENTS_PER_THREAD, collect_indexed, merge_element_triplets};
use crate::quadrature::QuadratureRule;
use crate::spaces::{FunctionSpace, SpaceValidationError};

use super::triplet::TripletMatrix;

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Process-wide cache of quadrature rules keyed by `(ElementType, order)`.
///
/// Rules are tiny, immutable after construction, and `Clone`, so a single
/// global table is cheaper and sound: no `thread_local` lifetime laundering,
/// no `UnsafeCell` aliasing risk under rayon worker churn.
static QUAD_CACHE: OnceLock<Mutex<HashMap<(ElementType, usize), QuadratureRule>>> = OnceLock::new();

fn quad_cache() -> &'static Mutex<HashMap<(ElementType, usize), QuadratureRule>> {
    QUAD_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Get a cached QuadratureRule for the given element type and order.
/// Avoids repeated allocation of the same quadrature rule for all elements of the same type.
///
/// Returns an owned clone; the cached entry is only held under the lock
/// while cloning, so callers never alias cache storage.
fn cached_quadrature_rule(elem_type: ElementType, quad_order: usize) -> QuadratureRule {
    let cache = quad_cache();
    let mut guard = cache.lock().expect("quadrature rule cache poisoned");
    guard
        .entry((elem_type, quad_order))
        .or_insert_with(|| QuadratureRule::new(elem_type, quad_order))
        .clone()
}

/// Assemble a bilinear form into a [`TripletMatrix`].
///
/// Loops over all mesh elements (in parallel if the `parallel` feature is enabled),
/// tabulates basis functions at each quadrature point, calls the form's integrand,
/// and scatters the element matrix into global triplets.
///
/// # Arguments
///
/// - `form`: the bilinear form to integrate
/// - `test_space` / `trial_space`: function spaces (same for Galerkin methods)
/// - `mesh`: the computational mesh
/// - `quad_order`: quadrature order (use [`crate::quadrature::required_order_for_stiffness`] etc.)
pub fn assemble_bilinear<T, F, S>(
    form: &F,
    test_space: &S,
    trial_space: &S,
    mesh: &Mesh,
    quad_order: usize,
) -> TripletMatrix<T>
where
    T: Copy + Default + std::ops::AddAssign + Send + Sync,
    F: BilinearForm<T>,
    S: FunctionSpace,
{
    let n_test = test_space.num_global_dofs(mesh);
    let n_trial = trial_space.num_global_dofs(mesh);
    let n_elems = mesh.num_elements();

    // Collect per-element triplets (rayon above the machine-adaptive gate,
    // serial for small meshes)
    let element_triplets: Vec<_> = collect_indexed(n_elems, ELEMENTS_PER_THREAD, |elem_idx| {
        assemble_element_bilinear(form, test_space, trial_space, mesh, elem_idx, quad_order)
    });

    // Merge into global triplet matrix
    let total_entries: usize = element_triplets.iter().map(|t| t.len()).sum();
    merge_element_triplets(element_triplets, n_test, n_trial, total_entries)
}

/// Reusable CSR sparsity pattern with an element-to-CSR scatter map.
///
/// Building the pattern once (unique `(row, col)` pairs per element,
/// sorted into CSR order) lets repeated assemblies of different forms
/// or coefficients skip the triplet materialization and the sort inside
/// `TripletMatrix::to_csr` entirely: element integration runs in
/// parallel as before, and the scatter is a direct indexed add per
/// local entry through the precomputed positions.
pub struct ElementScatter {
    n_rows: usize,
    n_cols: usize,
    row_ptrs: Vec<usize>,
    col_indices: Vec<usize>,
    /// CSR position of every local `(test, trial)` entry, grouped by
    /// element in the same nested order `assemble_element_bilinear`
    /// emits local entries.
    positions: Vec<usize>,
    /// Start of each element's slice in `positions`.
    element_offsets: Vec<usize>,
}

impl ElementScatter {
    /// Builds the shared sparsity pattern for a test/trial space pair.
    pub fn build<S>(test_space: &S, trial_space: &S, mesh: &Mesh) -> Self
    where
        S: FunctionSpace,
    {
        let n_rows = test_space.num_global_dofs(mesh);
        let n_cols = trial_space.num_global_dofs(mesh);
        let n_elems = mesh.num_elements();

        let mut element_dofs: Vec<(Vec<usize>, Vec<usize>)> = Vec::with_capacity(n_elems);
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        for elem_idx in 0..n_elems {
            let test_dofs = test_space.local_to_global(mesh, elem_idx);
            let trial_dofs = trial_space.local_to_global(mesh, elem_idx);
            for &row in &test_dofs {
                for &col in &trial_dofs {
                    pairs.push((row, col));
                }
            }
            element_dofs.push((test_dofs, trial_dofs));
        }
        pairs.sort_unstable();
        pairs.dedup();

        let mut row_ptrs = vec![0usize; n_rows + 1];
        let mut col_indices = Vec::with_capacity(pairs.len());
        for &(row, col) in &pairs {
            row_ptrs[row + 1] += 1;
            col_indices.push(col);
        }
        for row in 0..n_rows {
            row_ptrs[row + 1] += row_ptrs[row];
        }
        debug_assert_eq!(col_indices.len(), pairs.len());

        // Map each element's local entries to CSR positions by binary
        // search inside the row's column range (columns are sorted).
        let mut positions = Vec::with_capacity(pairs.len() * 2);
        let mut element_offsets = Vec::with_capacity(n_elems + 1);
        for (test_dofs, trial_dofs) in &element_dofs {
            element_offsets.push(positions.len());
            for &row in test_dofs {
                let row_range = row_ptrs[row]..row_ptrs[row + 1];
                for &col in trial_dofs {
                    let offset = col_indices[row_range.clone()]
                        .binary_search(&col)
                        .expect("element entry missing from sparsity pattern");
                    positions.push(row_range.start + offset);
                }
            }
        }
        element_offsets.push(positions.len());

        Self {
            n_rows,
            n_cols,
            row_ptrs,
            col_indices,
            positions,
            element_offsets,
        }
    }

    /// Number of structural non-zeros in the shared pattern.
    pub fn nnz(&self) -> usize {
        self.col_indices.len()
    }

    /// Assembles a form directly into CSR through the scatter map.
    pub fn assemble<T, F, S>(
        &self,
        form: &F,
        test_space: &S,
        trial_space: &S,
        mesh: &Mesh,
        quad_order: usize,
    ) -> sonium_solvers::CsrMatrix<T>
    where
        T: Copy + Default + std::ops::AddAssign + Send + Sync + sonium_solvers::ComplexField,
        F: BilinearForm<T>,
        S: FunctionSpace,
    {
        let mut matrix = sonium_solvers::CsrMatrix::from_raw_parts(
            self.n_rows,
            self.n_cols,
            self.row_ptrs.clone(),
            self.col_indices.clone(),
            vec![T::default(); self.nnz()],
        );
        self.assemble_into(form, test_space, trial_space, mesh, quad_order, &mut matrix);
        matrix
    }

    /// Assembles into an existing CSR matrix with this pattern.
    ///
    /// The output's sparsity buffers are reused untouched; only its
    /// values are zeroed and re-scattered.
    ///
    /// # Panics
    /// Panics when `out` does not carry this scatter's sparsity pattern.
    pub fn assemble_into<T, F, S>(
        &self,
        form: &F,
        test_space: &S,
        trial_space: &S,
        mesh: &Mesh,
        quad_order: usize,
        out: &mut sonium_solvers::CsrMatrix<T>,
    ) where
        T: Copy + Default + std::ops::AddAssign + Send + Sync + sonium_solvers::ComplexField,
        F: BilinearForm<T>,
        S: FunctionSpace,
    {
        assert_eq!(out.num_rows, self.n_rows, "scatter row count mismatch");
        assert_eq!(out.num_cols, self.n_cols, "scatter column count mismatch");
        assert_eq!(out.row_ptrs, self.row_ptrs, "scatter row pattern mismatch");
        assert_eq!(
            out.col_indices, self.col_indices,
            "scatter column pattern mismatch"
        );
        out.values.fill(T::default());

        let element_triplets: Vec<_> =
            collect_indexed(mesh.num_elements(), ELEMENTS_PER_THREAD, |elem_idx| {
                assemble_element_bilinear(form, test_space, trial_space, mesh, elem_idx, quad_order)
            });
        for (elem_idx, triplets) in element_triplets.iter().enumerate() {
            let slice =
                &self.positions[self.element_offsets[elem_idx]..self.element_offsets[elem_idx + 1]];
            debug_assert_eq!(slice.len(), triplets.len());
            for (&position, &(_, _, value)) in slice.iter().zip(triplets.iter()) {
                out.values[position] += value;
            }
        }
    }
}

/// Assemble a bilinear form that may vary by element.
///
/// This is the assembly seam for staggered material fields: each element gets
/// its own form instance, while the basis, quadrature, and global scatter
/// remain identical to [`assemble_bilinear`].
pub fn assemble_bilinear_per_element<T, F, S>(
    forms: &[F],
    test_space: &S,
    trial_space: &S,
    mesh: &Mesh,
    quad_order: usize,
) -> TripletMatrix<T>
where
    T: Copy + Default + std::ops::AddAssign + Send + Sync,
    F: BilinearForm<T>,
    S: FunctionSpace,
{
    assert_eq!(
        forms.len(),
        mesh.num_elements(),
        "one bilinear form is required for every mesh element"
    );
    let n_test = test_space.num_global_dofs(mesh);
    let n_trial = trial_space.num_global_dofs(mesh);
    let element_triplets: Vec<_> =
        collect_indexed(mesh.num_elements(), ELEMENTS_PER_THREAD, |elem_idx| {
            assemble_element_bilinear(
                &forms[elem_idx],
                test_space,
                trial_space,
                mesh,
                elem_idx,
                quad_order,
            )
        });

    let total_entries: usize = element_triplets.iter().map(|triplets| triplets.len()).sum();
    merge_element_triplets(element_triplets, n_test, n_trial, total_entries)
}

/// Checked variant of [`assemble_bilinear`].
///
/// The legacy function is intentionally infallible for hot inner loops. This
/// entry point validates global DOF numbering and element-local cardinality
/// before any parallel tabulation begins, allowing applications to turn bad
/// topology into a typed failure at their input boundary.
pub fn assemble_bilinear_checked<T, F, S>(
    form: &F,
    test_space: &S,
    trial_space: &S,
    mesh: &Mesh,
    quad_order: usize,
) -> Result<TripletMatrix<T>, SpaceValidationError>
where
    T: Copy + Default + std::ops::AddAssign + Send + Sync,
    F: BilinearForm<T>,
    S: FunctionSpace,
{
    test_space.validate(mesh)?;
    trial_space.validate(mesh)?;
    Ok(assemble_bilinear(
        form,
        test_space,
        trial_space,
        mesh,
        quad_order,
    ))
}

/// Assemble a single element's contribution to a bilinear form.
pub(super) fn assemble_element_bilinear<T, F, S>(
    form: &F,
    test_space: &S,
    trial_space: &S,
    mesh: &Mesh,
    elem_idx: usize,
    quad_order: usize,
) -> Vec<(usize, usize, T)>
where
    T: Copy + Default + std::ops::AddAssign,
    F: BilinearForm<T>,
    S: FunctionSpace,
{
    let elem = &mesh.elements[elem_idx];
    let elem_type = elem.element_type;

    let test_dofs = test_space.local_to_global(mesh, elem_idx);
    let trial_dofs = trial_space.local_to_global(mesh, elem_idx);
    let n_test = test_dofs.len();
    let n_trial = trial_dofs.len();

    let quad = cached_quadrature_rule(elem_type, quad_order);
    let mut local_mat = LocalMatrix::zeros(n_test.max(n_trial));

    for qp in quad.iter() {
        let test_tab = test_space.tabulate(mesh, elem_idx, qp.xi(), qp.eta(), qp.zeta());
        let trial_tab = trial_space.tabulate(mesh, elem_idx, qp.xi(), qp.eta(), qp.zeta());

        form.integrate_point(&test_tab, &trial_tab, qp.weight, &mut local_mat);
    }

    // Scatter to global triplets
    let mut triplets = Vec::with_capacity(n_test * n_trial);
    for (i, &test_dof) in test_dofs.iter().enumerate() {
        for (j, &trial_dof) in trial_dofs.iter().enumerate() {
            let val = local_mat.get(i, j);
            triplets.push((test_dof, trial_dof, val));
        }
    }

    triplets
}

/// Assemble a linear form into a global vector.
///
/// Returns a dense vector of size `test_space.num_global_dofs(mesh)`.
pub fn assemble_linear<T, F, S>(form: &F, test_space: &S, mesh: &Mesh, quad_order: usize) -> Vec<T>
where
    T: Copy + Default + std::ops::AddAssign + Send + Sync,
    F: LinearForm<T>,
    S: FunctionSpace,
{
    let n_dofs = test_space.num_global_dofs(mesh);
    let n_elems = mesh.num_elements();

    // Collect per-element contributions (rayon above the machine-adaptive
    // gate, serial for small meshes)
    let element_contribs: Vec<_> = collect_indexed(n_elems, ELEMENTS_PER_THREAD, |elem_idx| {
        assemble_element_linear(form, test_space, mesh, elem_idx, quad_order)
    });

    // Scatter into global vector
    let mut global = vec![T::default(); n_dofs];
    for contribs in element_contribs {
        for (dof, val) in contribs {
            global[dof] += val;
        }
    }

    global
}

/// Assemble a linear form that may vary by element.
pub fn assemble_linear_per_element<T, F, S>(
    forms: &[F],
    test_space: &S,
    mesh: &Mesh,
    quad_order: usize,
) -> Vec<T>
where
    T: Copy + Default + std::ops::AddAssign + Send + Sync,
    F: LinearForm<T>,
    S: FunctionSpace,
{
    assert_eq!(
        forms.len(),
        mesh.num_elements(),
        "one linear form is required for every mesh element"
    );
    let n_dofs = test_space.num_global_dofs(mesh);
    let element_contribs: Vec<_> =
        collect_indexed(mesh.num_elements(), ELEMENTS_PER_THREAD, |elem_idx| {
            assemble_element_linear(&forms[elem_idx], test_space, mesh, elem_idx, quad_order)
        });

    let mut global = vec![T::default(); n_dofs];
    for contribs in element_contribs {
        for (dof, value) in contribs {
            global[dof] += value;
        }
    }
    global
}

/// Checked variant of [`assemble_linear`].
///
/// The legacy assembler remains infallible for compatibility and hot-loop
/// callers. This entry point validates the complete global DOF map before
/// parallel tabulation, so malformed topology is reported instead of causing
/// an out-of-bounds scatter panic.
pub fn assemble_linear_checked<T, F, S>(
    form: &F,
    test_space: &S,
    mesh: &Mesh,
    quad_order: usize,
) -> Result<Vec<T>, SpaceValidationError>
where
    T: Copy + Default + std::ops::AddAssign + Send + Sync,
    F: LinearForm<T>,
    S: FunctionSpace,
{
    test_space.validate(mesh)?;
    Ok(assemble_linear(form, test_space, mesh, quad_order))
}

/// Assemble a single element's contribution to a linear form.
pub(super) fn assemble_element_linear<T, F, S>(
    form: &F,
    test_space: &S,
    mesh: &Mesh,
    elem_idx: usize,
    quad_order: usize,
) -> Vec<(usize, T)>
where
    T: Copy + Default + std::ops::AddAssign,
    F: LinearForm<T>,
    S: FunctionSpace,
{
    let elem = &mesh.elements[elem_idx];
    let elem_type = elem.element_type;

    let test_dofs = test_space.local_to_global(mesh, elem_idx);
    let n_test = test_dofs.len();

    let quad = cached_quadrature_rule(elem_type, quad_order);
    let mut local_vec = LocalVector::zeros(n_test);

    for qp in quad.iter() {
        let test_tab = test_space.tabulate(mesh, elem_idx, qp.xi(), qp.eta(), qp.zeta());
        form.integrate_point(&test_tab, qp.weight, &mut local_vec);
    }

    // Scatter to global
    test_dofs
        .iter()
        .zip(local_vec.values.iter())
        .map(|(&dof, &val)| (dof, val))
        .collect()
}

// ---------------------------------------------------------------------------
// Element-level assembly (beam, shell, analytical elements)
// ---------------------------------------------------------------------------

/// Assemble an [`ElementForm`] into a [`TripletMatrix`].
///
/// Unlike [`assemble_bilinear`], this calls the form once per element (no
/// quadrature loop). The form itself is responsible for integration — it may
/// use analytical formulas, specialised quadrature, or any other strategy.
pub fn assemble_element_form<T, F>(form: &F, n_global_dofs: usize) -> TripletMatrix<T>
where
    T: Copy + Default + std::ops::AddAssign + Send + Sync,
    F: ElementForm<T>,
{
    let n_elems = form.num_elements();

    let element_triplets: Vec<Vec<(usize, usize, T)>> =
        collect_indexed(n_elems, ELEMENTS_PER_THREAD, |e| {
            scatter_element_matrix(form, e)
        });

    let total: usize = element_triplets.iter().map(|t| t.len()).sum();
    merge_element_triplets(element_triplets, n_global_dofs, n_global_dofs, total)
}

fn scatter_element_matrix<T, F>(form: &F, elem_idx: usize) -> Vec<(usize, usize, T)>
where
    T: Copy + Default + std::ops::AddAssign,
    F: ElementForm<T>,
{
    let dofs = form.local_to_global(elem_idx);
    let n = dofs.len();
    let k = form.element_matrix(elem_idx);
    let mut triplets = Vec::with_capacity(n * n);
    for (i, &gi) in dofs.iter().enumerate() {
        for (j, &gj) in dofs.iter().enumerate() {
            triplets.push((gi, gj, k[i * n + j]));
        }
    }
    triplets
}

/// Assemble residual and tangent from a [`NonlinearElementForm`].
///
/// Returns `(residual_vector, tangent_matrix)` evaluated at the given global
/// solution.
pub fn assemble_nonlinear_element_form<T, F>(
    form: &F,
    global_solution: &[T],
) -> (Vec<T>, TripletMatrix<T>)
where
    T: Copy + Default + std::ops::AddAssign + Send + Sync,
    F: NonlinearElementForm<T>,
{
    let n_dofs = form.num_global_dofs();
    let n_elems = form.num_elements();

    let per_elem: Vec<_> = collect_indexed(n_elems, ELEMENTS_PER_THREAD, |e| {
        let dofs = form.local_to_global(e);
        let r = form.element_residual(e, global_solution);
        let k = form.element_tangent(e, global_solution);
        (dofs, r, k)
    });

    let mut residual = vec![T::default(); n_dofs];
    let total_nnz: usize = per_elem.iter().map(|(d, _, _)| d.len() * d.len()).sum();
    let mut tangent = TripletMatrix::with_capacity(n_dofs, n_dofs, total_nnz);

    for (dofs, r, k) in per_elem {
        let n = dofs.len();
        for (i, &gi) in dofs.iter().enumerate() {
            residual[gi] += r[i];
            for (j, &gj) in dofs.iter().enumerate() {
                tangent.push(gi, gj, k[i * n + j]);
            }
        }
    }

    (residual, tangent)
}

// ---------------------------------------------------------------------------
// Boundary surface assembly (physics-agnostic)
// ---------------------------------------------------------------------------

/// Assemble a boundary mass matrix ∫_Γ φ_i φ_j dΓ for a specific boundary
/// marker, returning a generic [`TripletMatrix<f64>`].
///
/// This is the physics-agnostic replacement for the legacy
/// `assemble_boundary_mass` (which returns a `MassMatrix`). It supports
/// P1 edges (2D) and P1 triangle faces (3D).
pub fn assemble_boundary_mass_generic(mesh: &Mesh, marker: i32) -> TripletMatrix<f64> {
    let n_dofs = mesh.num_nodes();
    let mut matrix = TripletMatrix::new(n_dofs, n_dofs);

    for boundary in &mesh.boundaries {
        if boundary.marker != marker {
            continue;
        }
        let nodes = &boundary.nodes;

        match nodes.len() {
            2 => {
                // P1 edge: M_loc = (L/6) [2 1; 1 2]
                let p0 = &mesh.nodes[nodes[0]];
                let p1 = &mesh.nodes[nodes[1]];
                let dx = p1.x - p0.x;
                let dy = p1.y - p0.y;
                let dz = p1.z - p0.z;
                let len = (dx * dx + dy * dy + dz * dz).sqrt();
                let factor = len / 6.0;
                for i in 0..2 {
                    for j in 0..2 {
                        let val = if i == j { 2.0 * factor } else { factor };
                        matrix.push(nodes[i], nodes[j], val);
                    }
                }
            }
            3 => {
                // P1 triangle face: M_loc = (Area/12) [2 1 1; 1 2 1; 1 1 2]
                let p0 = &mesh.nodes[nodes[0]];
                let p1 = &mesh.nodes[nodes[1]];
                let p2 = &mesh.nodes[nodes[2]];
                let v1 = [p1.x - p0.x, p1.y - p0.y, p1.z - p0.z];
                let v2 = [p2.x - p0.x, p2.y - p0.y, p2.z - p0.z];
                let cx = v1[1] * v2[2] - v1[2] * v2[1];
                let cy = v1[2] * v2[0] - v1[0] * v2[2];
                let cz = v1[0] * v2[1] - v1[1] * v2[0];
                let area = 0.5 * (cx * cx + cy * cy + cz * cz).sqrt();
                let factor = area / 12.0;
                for i in 0..3 {
                    for j in 0..3 {
                        let val = if i == j { 2.0 * factor } else { factor };
                        matrix.push(nodes[i], nodes[j], val);
                    }
                }
            }
            4 => {
                // Quad face: split into triangles 0-1-2 and 0-2-3
                for &tri_nodes in &[
                    [nodes[0], nodes[1], nodes[2]],
                    [nodes[0], nodes[2], nodes[3]],
                ] {
                    let tri_coords: Vec<[f64; 3]> = tri_nodes
                        .iter()
                        .map(|&n| [mesh.nodes[n].x, mesh.nodes[n].y, mesh.nodes[n].z])
                        .collect();
                    let v1 = [
                        tri_coords[1][0] - tri_coords[0][0],
                        tri_coords[1][1] - tri_coords[0][1],
                        tri_coords[1][2] - tri_coords[0][2],
                    ];
                    let v2 = [
                        tri_coords[2][0] - tri_coords[0][0],
                        tri_coords[2][1] - tri_coords[0][1],
                        tri_coords[2][2] - tri_coords[0][2],
                    ];
                    let cx = v1[1] * v2[2] - v1[2] * v2[1];
                    let cy = v1[2] * v2[0] - v1[0] * v2[2];
                    let cz = v1[0] * v2[1] - v1[1] * v2[0];
                    let area = 0.5 * (cx * cx + cy * cy + cz * cz).sqrt();
                    let factor = area / 12.0;
                    for i in 0..3 {
                        for j in 0..3 {
                            let val = if i == j { 2.0 * factor } else { factor };
                            matrix.push(tri_nodes[i], tri_nodes[j], val);
                        }
                    }
                }
            }
            _ => {} // skip unsupported face types
        }
    }

    matrix
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::PolynomialDegree;
    use crate::forms::{LaplacianForm, MassForm, SourceForm};
    use crate::mesh::unit_square_triangles;
    use crate::spaces::H1Space;

    #[test]
    fn test_assemble_bilinear_laplacian() {
        let mesh = unit_square_triangles(4);
        let space = H1Space::new(PolynomialDegree::P1);

        let triplet = assemble_bilinear(&LaplacianForm, &space, &space, &mesh, 1);

        assert_eq!(triplet.num_rows, mesh.num_nodes());
        assert_eq!(triplet.num_cols, mesh.num_nodes());
        assert!(triplet.nnz() > 0);

        // Convert to CSR and check row sums approximately zero (Laplacian on a closed domain)
        let csr = triplet.to_csr();
        for i in 0..csr.num_rows {
            let row_sum: f64 = csr.row_entries(i).map(|(_, v)| v).sum();
            assert!(
                row_sum.abs() < 1e-10,
                "Row {i} sum = {row_sum}, expected ~0"
            );
        }
    }

    #[test]
    fn element_scatter_matches_triplet_assembly_and_reuses_pattern() {
        let mesh = unit_square_triangles(4);
        let space = H1Space::new(PolynomialDegree::P1);
        let scatter = ElementScatter::build(&space, &space, &mesh);

        let direct = scatter.assemble(&LaplacianForm, &space, &space, &mesh, 1);
        let reference = assemble_bilinear(&LaplacianForm, &space, &space, &mesh, 1).to_csr();
        assert_eq!(direct.row_ptrs, reference.row_ptrs);
        assert_eq!(direct.col_indices, reference.col_indices);
        for (actual, expected) in direct.values.iter().zip(reference.values.iter()) {
            assert!((actual - expected).abs() < 1e-12);
        }

        // Reassembly with a different form reuses the same pattern.
        let mass = scatter.assemble(&MassForm, &space, &space, &mesh, 2);
        let mass_reference = assemble_bilinear(&MassForm, &space, &space, &mesh, 2).to_csr();
        assert_eq!(mass.row_ptrs, direct.row_ptrs);
        for (actual, expected) in mass.values.iter().zip(mass_reference.values.iter()) {
            assert!((actual - expected).abs() < 1e-12);
        }
        assert_eq!(scatter.nnz(), direct.col_indices.len());
    }

    #[test]
    fn test_assemble_bilinear_mass() {
        let mesh = unit_square_triangles(4);
        let space = H1Space::new(PolynomialDegree::P1);

        let triplet = assemble_bilinear(&MassForm, &space, &space, &mesh, 2);

        let csr = triplet.to_csr();
        // Total mass = integral of 1 over the unit square = 1.0
        let total_mass: f64 = (0..csr.num_rows)
            .map(|i| csr.row_entries(i).map(|(_, v)| v).sum::<f64>())
            .sum();
        assert!(
            (total_mass - 1.0).abs() < 1e-10,
            "Total mass = {total_mass}, expected 1.0"
        );
    }

    #[test]
    fn checked_linear_assembly_matches_legacy_path() {
        let mesh = unit_square_triangles(2);
        let space = H1Space::new(PolynomialDegree::P1);
        let form = SourceForm {
            source_fn: |_, _, _| 1.0,
        };

        let legacy = assemble_linear(&form, &space, &mesh, 2);
        let checked = assemble_linear_checked(&form, &space, &mesh, 2)
            .expect("valid H1 topology should pass checked assembly");

        assert_eq!(checked, legacy);
        assert!((checked.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn per_element_assembly_scales_local_sources_and_preserves_mass() {
        let mesh = unit_square_triangles(2);
        let space = H1Space::new(PolynomialDegree::P1);
        let forms: Vec<_> = (0..mesh.num_elements())
            .map(|element| SourceForm {
                source_fn: move |_, _, _| if element % 2 == 0 { 1.0 } else { 2.0 },
            })
            .collect();
        let rhs = assemble_linear_per_element(&forms, &space, &mesh, 2);
        assert!((rhs.iter().sum::<f64>() - 1.5).abs() < 1e-10);

        let mass_forms: Vec<_> = (0..mesh.num_elements()).map(|_| MassForm).collect();
        let per_element = assemble_bilinear_per_element(&mass_forms, &space, &space, &mesh, 2);
        let legacy = assemble_bilinear(&MassForm, &space, &space, &mesh, 2);
        assert_eq!(per_element.to_csr().values, legacy.to_csr().values);
    }

    /// Restores a process env var on drop. Only merge path selection reads
    /// these keys, and both paths are exactly correct, so concurrent tests
    /// are unaffected beyond which (correct) path they take.
    struct EnvVarGuard {
        key: &'static str,
        prev: Option<std::ffi::OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let prev = std::env::var_os(key);
            // SAFETY: only merge path selection reads these keys, and both
            // the serial and parallel merge paths are exactly correct, so a
            // concurrent test can at most take the other correct path.
            unsafe {
                std::env::set_var(key, value);
            }
            Self { key, prev }
        }

        fn remove(key: &'static str) -> Self {
            let prev = std::env::var_os(key);
            // SAFETY: see `set` — path selection only, both paths correct.
            unsafe {
                std::env::remove_var(key);
            }
            Self { key, prev }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            // SAFETY: restores the pre-test value; see `set`.
            unsafe {
                match &self.prev {
                    Some(value) => std::env::set_var(self.key, value),
                    None => std::env::remove_var(self.key),
                }
            }
        }
    }

    #[test]
    fn triplet_merge_parallel_matches_serial_order() {
        // Synthetic per-element lists with duplicate positions: exact
        // push-order equality implies identical duplicate summation in CSR.
        let element_triplets: Vec<Vec<(usize, usize, f64)>> = (0..64)
            .map(|e| {
                vec![
                    (e % 8, (3 * e) % 8, e as f64),
                    (e % 8, (3 * e) % 8, 0.5),
                    ((3 * e) % 8, e % 8, -(e as f64)),
                ]
            })
            .collect();
        let total: usize = element_triplets.iter().map(Vec::len).sum();

        let serial = {
            let _force_serial = EnvVarGuard::set("SONIUM_SERIAL", "1");
            crate::parallelism::merge_element_triplets(element_triplets.clone(), 8, 8, total)
        };
        let parallel = {
            let _threshold = EnvVarGuard::set("SONIUM_FEM_PAR_THRESHOLD", "1");
            let _no_serial = EnvVarGuard::remove("SONIUM_SERIAL");
            crate::parallelism::merge_element_triplets(element_triplets, 8, 8, total)
        };
        // On single-threaded machines the "parallel" side also runs serially;
        // either way the push order must be identical.
        assert_eq!(serial.rows, parallel.rows);
        assert_eq!(serial.cols, parallel.cols);
        assert_eq!(serial.values, parallel.values);
        assert_eq!(serial.to_csr().values, parallel.to_csr().values);
    }
}
