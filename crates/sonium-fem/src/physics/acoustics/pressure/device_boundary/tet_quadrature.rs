//! Affine P1/P2 trace maps for resident boundary numeric assembly.

// Rust guideline compliant 2026-02-21
use super::{
    invalid,
    quadrature::{RobinQuadrature, matrix_map},
    tet::prepare_p2_trace,
};
use crate::physics::acoustics::pressure::{HelmholtzForm, device_tet::prepare_tet_p1_helmholtz};
use crate::{basis::PolynomialDegree, mesh::Mesh, quadrature::gauss_triangle, spaces::H1Space};
use sonium_compute::{BackendResult, quadrature_assembly::QuadratureAssemblyMap};
use std::collections::{BTreeMap, HashSet};

/// Prepare fixed affine P1/P2 tetrahedral boundary maps for native numeric updates.
///
/// Samples follow marker order: P1 face centroids, P2 degree-four quadrature. The maps
/// retain geometry/constraints while alpha and source values may change on CUDA.
/// Storage scales with sample count times squared trace size; setup is host work.
///
/// # Errors
/// Rejects invalid geometry, unsupported spaces, empty/duplicate/missing markers,
/// repeated faces, invalid constraints and non-finite or oversized map weights.
pub fn prepare_tet_robin_quadratures(
    mesh: &Mesh,
    space: &H1Space,
    markers: &[i32],
    dirichlet: &[usize],
) -> BackendResult<RobinQuadrature> {
    mesh.validate_connectivity()
        .map_err(|e| invalid(format!("invalid mesh: {e:?}")))?;
    let promoted;
    let trace = match space.degree {
        PolynomialDegree::P1 => {
            prepare_tet_p1_helmholtz(mesh, space, &HelmholtzForm::from_real(0.0))?;
            mesh
        }
        PolynomialDegree::P2 => {
            promoted = prepare_p2_trace(mesh, space)?;
            &promoted
        }
        _ => return Err(invalid("tetrahedral boundary maps require P1/P2")),
    };
    let mut seen = HashSet::new();
    if markers.is_empty()
        || markers
            .iter()
            .any(|&m| !seen.insert(m) || !trace.boundaries.iter().any(|f| f.marker == m))
    {
        return Err(invalid("empty, duplicate or missing boundary markers"));
    }
    let mut faces = HashSet::new();
    for face in trace.boundaries.iter().filter(|f| seen.contains(&f.marker)) {
        if face.nodes.len()
            != if space.degree == PolynomialDegree::P1 {
                3
            } else {
                6
            }
        {
            return Err(invalid("invalid triangular trace size"));
        }
        let mut vertices = face.nodes[..3].to_vec();
        vertices.sort_unstable();
        if !faces.insert(vertices) {
            return Err(invalid("duplicate selected tetrahedral face"));
        }
    }
    let n = trace.nodes.len();
    if dirichlet.iter().any(|&d| d >= n) {
        return Err(invalid("Dirichlet DOF out of bounds"));
    }
    let mut mask = vec![false; n];
    for &d in dirichlet {
        mask[d] = true;
    }
    let mut points = Vec::new();
    let mut sample_markers = Vec::new();
    let mut action = BTreeMap::new();
    let mut coupling = BTreeMap::new();
    let mut load = Vec::new();
    for &marker in markers {
        for face in trace.boundaries.iter().filter(|f| f.marker == marker) {
            let expected = if space.degree == PolynomialDegree::P1 {
                3
            } else {
                6
            };
            if face.nodes.len() != expected {
                return Err(invalid("invalid triangular trace size"));
            }
            let p = face.nodes[..3]
                .iter()
                .map(|&d| trace.nodes[d])
                .collect::<Vec<_>>();
            let u = [p[1].x - p[0].x, p[1].y - p[0].y, p[1].z - p[0].z];
            let v = [p[2].x - p[0].x, p[2].y - p[0].y, p[2].z - p[0].z];
            let cross = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            let jac = cross.iter().map(|v| v * v).sum::<f64>().sqrt();
            if !jac.is_finite() || jac <= 0.0 {
                return Err(invalid("invalid boundary Jacobian"));
            }
            if expected == 3 {
                // Preserve the checked P1 contract: facewise centroid coefficients
                // multiplied by the exact constant-coefficient triangle mass/load.
                let sample = points.len();
                points.push([
                    (p[0].x + p[1].x + p[2].x) / 3.0,
                    (p[0].y + p[1].y + p[2].y) / 3.0,
                    (p[0].z + p[1].z + p[2].z) / 3.0,
                ]);
                sample_markers.push(marker);
                for (i, &r) in face.nodes.iter().enumerate() {
                    if mask[r] {
                        continue;
                    }
                    load.push((r, sample, jac / 6.0));
                    for (j, &c) in face.nodes.iter().enumerate() {
                        let target = if mask[c] { &mut coupling } else { &mut action };
                        target
                            .entry((r, c))
                            .or_insert_with(Vec::new)
                            .push((sample, jac / 24.0 * if i == j { 2.0 } else { 1.0 }));
                    }
                }
                continue;
            }
            for q in gauss_triangle(4) {
                let (l0, l1, l2) = (1.0 - q.xi() - q.eta(), q.xi(), q.eta());
                let basis = if expected == 3 {
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
                let sample = points.len();
                points.push([
                    l0 * p[0].x + l1 * p[1].x + l2 * p[2].x,
                    l0 * p[0].y + l1 * p[1].y + l2 * p[2].y,
                    l0 * p[0].z + l1 * p[1].z + l2 * p[2].z,
                ]);
                sample_markers.push(marker);
                let weight = jac * q.weight;
                for (i, &r) in face.nodes.iter().enumerate() {
                    if mask[r] {
                        continue;
                    }
                    load.push((r, sample, basis[i] * weight));
                    for (j, &c) in face.nodes.iter().enumerate() {
                        let target = if mask[c] { &mut coupling } else { &mut action };
                        target
                            .entry((r, c))
                            .or_insert_with(Vec::new)
                            .push((sample, basis[i] * basis[j] * weight));
                    }
                }
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::unit_cube_tetrahedra;
    use crate::physics::acoustics::pressure::device_boundary::{
        prepare_tet_p1_robin, prepare_tet_p2_robin,
    };
    use crate::spaces::FunctionSpace;
    use num_complex::Complex64;
    use sonium_solvers::CsrMatrix;

    fn compare(a: &CsrMatrix<Complex64>, b: &CsrMatrix<Complex64>) {
        for r in 0..a.num_rows {
            let mut difference = BTreeMap::new();
            for (m, sign) in [(a, 1.0), (b, -1.0)] {
                for j in m.row_ptrs[r]..m.row_ptrs[r + 1] {
                    *difference
                        .entry(m.col_indices[j])
                        .or_insert(Complex64::default()) += m.values[j] * sign;
                }
            }
            assert!(difference.values().all(|v| v.norm() < 1e-11));
        }
    }
    #[test]
    fn tet_maps_match_checked_boundary_assembly_and_keep_zero_slots() {
        let mut mesh = unit_cube_tetrahedra(2);
        for f in &mut mesh.boundaries {
            f.marker = if f.nodes.iter().all(|&d| mesh.nodes[d].x == 0.0) {
                1
            } else {
                2
            };
        }
        for degree in [PolynomialDegree::P1, PolynomialDegree::P2] {
            let space = H1Space::new(degree);
            let constrained = space.boundary_dofs(&mesh, 1);
            let map = prepare_tet_robin_quadratures(&mesh, &space, &[2], &constrained).unwrap();
            for scale in [0.0, 1.0, 2.0] {
                let alpha = |x: f64, y: f64, z: f64| Complex64::new(0.7 + x, 0.2 + y + z) * scale;
                let source = |x: f64, y: f64, z: f64| Complex64::new(x + y, z - 0.3);
                let prepare = if degree == PolynomialDegree::P1 {
                    prepare_tet_p1_robin
                } else {
                    prepare_tet_p2_robin
                };
                let reference = prepare(&mesh, &space, 2, &alpha, &source, &constrained).unwrap();
                let av: Vec<_> = map
                    .points()
                    .iter()
                    .map(|p| alpha(p[0], p[1], p[2]))
                    .collect();
                let gv: Vec<_> = map
                    .points()
                    .iter()
                    .map(|p| source(p[0], p[1], p[2]))
                    .collect();
                let mut a = map.action().template().clone();
                a.values = map.action().values().try_assemble(&av).unwrap();
                let mut c = map.coupling().template().clone();
                c.values = map.coupling().values().try_assemble(&av).unwrap();
                compare(&a, &reference.matrix);
                compare(&c, &reference.dirichlet_coupling);
                let load = map.load().try_assemble(&gv).unwrap();
                assert!(
                    load.iter()
                        .zip(&reference.rhs)
                        .all(|(a, b)| (*a - *b).norm() < 1e-11)
                );
                assert!(!a.values.is_empty());
            }
            assert!(prepare_tet_robin_quadratures(&mesh, &space, &[2, 2], &constrained).is_err());
            assert!(prepare_tet_robin_quadratures(&mesh, &space, &[99], &constrained).is_err());
        }
    }
}
