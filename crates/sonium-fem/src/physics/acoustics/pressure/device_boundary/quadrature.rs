//! Fixed boundary quadrature maps for device numeric coefficient updates.

// Rust guideline compliant 2026-02-21
use super::{hex::visit_hex_quadrature, invalid};
use crate::mesh::Mesh;
use crate::physics::acoustics::pressure::{HelmholtzForm, device_hex::prepare_hex_helmholtz};
use crate::spaces::H1Space;
use num_complex::Complex64;
use sonium_compute::{BackendResult, quadrature_assembly::QuadratureAssemblyMap};
use sonium_solvers::CsrMatrix;
use std::collections::BTreeMap;

/// Fixed CSR structure and a map producing its numeric values from samples.
#[derive(Debug)]
pub struct BoundaryMatrixMap {
    template: CsrMatrix<Complex64>,
    values: QuadratureAssemblyMap,
}
impl BoundaryMatrixMap {
    /// Borrow CSR structure with zero placeholder values, including structural zeros.
    pub fn template(&self) -> &CsrMatrix<Complex64> {
        &self.template
    }
    /// Borrow the map whose output order matches the template's CSR values.
    pub fn values(&self) -> &QuadratureAssemblyMap {
        &self.values
    }
}

/// Geometry and constraint maps for repeated marked Robin boundary updates.
///
/// Upload the three maps through `CudaQuadratureAssembly`; supply alpha to the
/// matrix maps and source samples to the load map. Device outputs update retained
/// CSR operators through `try_copy_values_from_device`. Setup remains on the host.
/// Geometry, topology and constraints must remain fixed. Sampling arbitrary Rust
/// callbacks is host work; resident material laws can produce samples on CUDA.
#[derive(Debug)]
pub struct RobinQuadrature {
    pub(super) points: Vec<[f64; 3]>,
    pub(super) markers: Vec<i32>,
    pub(super) action: BoundaryMatrixMap,
    pub(super) coupling: BoundaryMatrixMap,
    pub(super) load: QuadratureAssemblyMap,
}
/// Compatibility name for hex boundary maps.
pub type HexRobinQuadrature = RobinQuadrature;

impl RobinQuadrature {
    /// Quadrature point coordinates in the coefficient buffers' required order.
    pub fn points(&self) -> &[[f64; 3]] {
        &self.points
    }
    /// Boundary marker corresponding to each quadrature point.
    pub fn markers(&self) -> &[i32] {
        &self.markers
    }
    /// Boundary matrix P B P, with no constrained identity contribution.
    pub fn action(&self) -> &BoundaryMatrixMap {
        &self.action
    }
    /// Boundary matrix P B Q used for nonzero prescribed-pressure lifting.
    pub fn coupling(&self) -> &BoundaryMatrixMap {
        &self.coupling
    }
    /// Map producing P f; its output length is the full H1 dimension.
    pub fn load(&self) -> &QuadratureAssemblyMap {
        &self.load
    }
}

type MatrixTerms = BTreeMap<(usize, usize), Vec<(usize, f64)>>;

#[cfg(test)]
mod tests;

pub(super) fn matrix_map(
    n: usize,
    samples: usize,
    terms: MatrixTerms,
) -> BackendResult<BoundaryMatrixMap> {
    // Build with nonzero placeholders so canonicalization preserves the pattern,
    // then zero values. Future zero alpha must not remove structural slots.
    let mut template = CsrMatrix::try_from_triplets(
        n,
        n,
        terms
            .keys()
            .map(|&(row, col)| (row, col, Complex64::new(1.0, 0.0)))
            .collect(),
    )
    .map_err(|e| invalid(e.to_string()))?;
    template.values.fill(Complex64::default());
    let mut reductions = Vec::new();
    for (slot, (_, entries)) in terms.into_iter().enumerate() {
        reductions.extend(
            entries
                .into_iter()
                .map(|(sample, weight)| (slot, sample, weight)),
        );
    }
    Ok(BoundaryMatrixMap {
        values: QuadratureAssemblyMap::try_new(samples, template.values.len(), &reductions)?,
        template,
    })
}

/// Prepare fixed Q1/Q2/Q3 hex boundary maps for GPU numeric assembly.
///
/// Uses the checked owning-element geometry and full solution trace. Storage
/// scales with quadrature samples times squared trace DOFs; include this mapping
/// cost in memory planning. This explicit map is a baseline for future compression.
///
/// # Errors
/// Rejects invalid geometry/space, face records, absent markers, invalid
/// quadrature/constraints, non-finite weights and index overflow.
pub fn prepare_hex_robin_quadrature(
    mesh: &Mesh,
    space: &H1Space,
    marker: i32,
    quadrature_points: usize,
    dirichlet: &[usize],
) -> BackendResult<RobinQuadrature> {
    prepare_hex_robin_quadratures(mesh, space, &[marker], quadrature_points, dirichlet)
}

/// Prepare a combined map for distinct marked Robin boundaries.
///
/// Samples are grouped in the supplied marker order. Shared global matrix
/// entries accumulate contributions from all selected faces.
///
/// # Errors
/// Rejects empty, repeated or absent markers and the same invalid geometry,
/// quadrature, constraints and index limits as `prepare_hex_robin_quadrature`.
pub fn prepare_hex_robin_quadratures(
    mesh: &Mesh,
    space: &H1Space,
    markers: &[i32],
    quadrature_points: usize,
    dirichlet: &[usize],
) -> BackendResult<RobinQuadrature> {
    let mut seen = std::collections::HashSet::new();
    if markers.is_empty()
        || markers.iter().any(|&marker| {
            !seen.insert(marker) || !mesh.boundaries.iter().any(|b| b.marker == marker)
        })
    {
        return Err(invalid("empty, duplicate or absent boundary markers"));
    }
    let mut owners = std::collections::HashSet::new();
    if mesh
        .boundaries
        .iter()
        .filter(|b| seen.contains(&b.marker))
        .any(|b| !owners.insert((b.element_idx, b.local_idx)))
    {
        return Err(invalid("duplicate face across selected boundary markers"));
    }
    let n = prepare_hex_helmholtz(
        mesh,
        space,
        &HelmholtzForm::from_real(0.0),
        quadrature_points,
    )?
    .dimension();
    if dirichlet.iter().any(|&i| i >= n) {
        return Err(invalid("Dirichlet DOF out of bounds"));
    }
    let mut mask = vec![false; n];
    for &i in dirichlet {
        mask[i] = true;
    }
    let mut points = Vec::new();
    let mut sample_markers = Vec::new();
    let mut action = MatrixTerms::new();
    let mut coupling = MatrixTerms::new();
    let mut load = Vec::new();
    for &marker in markers {
        visit_hex_quadrature(
            mesh,
            space,
            marker,
            quadrature_points,
            |point, dofs, basis, weight| {
                let sample = points.len();
                points.push(point);
                sample_markers.push(marker);
                for (i, &row) in dofs.iter().enumerate() {
                    if mask[row] || basis[i] == 0.0 {
                        continue;
                    }
                    load.push((row, sample, basis[i] * weight));
                    for (j, &col) in dofs.iter().enumerate() {
                        if basis[j] == 0.0 {
                            continue;
                        }
                        let target = if mask[col] {
                            &mut coupling
                        } else {
                            &mut action
                        };
                        target
                            .entry((row, col))
                            .or_default()
                            .push((sample, basis[i] * basis[j] * weight));
                    }
                }
                Ok(())
            },
        )?;
    }
    if points.is_empty() {
        return Err(invalid("marked boundary has no quadrature samples"));
    }
    let samples = points.len();
    Ok(RobinQuadrature {
        points,
        markers: sample_markers,
        action: matrix_map(n, samples, action)?,
        coupling: matrix_map(n, samples, coupling)?,
        load: QuadratureAssemblyMap::try_new(samples, n, &load)?,
    })
}
