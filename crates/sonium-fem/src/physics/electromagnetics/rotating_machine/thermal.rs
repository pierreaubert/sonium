//! Steady-state 2-D heat conduction on the motor cross-section: the
//! thermal twin of the electromagnetic `A_z` solve, mirroring the
//! reference solver's `simulation/thermal_solver_2d.py`.
//!
//! Solves `-div(K grad T) = q` with P1 triangles on the corner nodes
//! of the EM mesh (so it accepts the P2-promoted mesh directly),
//! per-element P0 conductivity and heat source, and Robin films on
//! cooled circular surfaces. `K` is a tensor in the local
//! cylindrical frame,
//!
//! ```text
//! K = k_r * r_hat r_hat^T + k_t * (I - r_hat r_hat^T),
//! ```
//!
//! evaluated at the quadrature points of a 3-point rule — a hoop-wound
//! retaining sleeve conducts very differently through its thickness
//! than along its fibres, and averaging the two is not a small error.
//! Isotropic regions pass `k_r = k_t` and the form collapses exactly.
//!
//! The temperature field is independent of stack length (stiffness
//! and load scale together); reported watts are per-unit-depth
//! integrals times [`ThermalProblem::stack_length_m`]. The heat
//! budget is part of the result: every surface and sink reports its
//! watts, so the caller's balance closes or is seen not to.
//!
//! Scope versus the reference: the mesh here is conforming and
//! welded, so the reference's island bridging and slip-line tie do
//! not exist — the air gap is meshed elements carrying the effective
//! gap conductivity, and lumped one-temperature external bodies
//! (the reference's `lumped_nodes`) are not modelled.

// Rust guideline compliant 2026-02-21

use std::collections::HashMap;

use sonium_solvers::TripletMatrix;

use crate::mesh::Mesh;

/// A cooled circular surface with a Robin film.
#[derive(Debug, Clone)]
pub struct RobinSurface {
    /// Report name of the surface.
    pub name: String,
    /// Radius of the cooled surface in metres.
    pub radius_m: f64,
    /// Radial tolerance for matching boundary facets, in metres.
    pub radius_tol_m: f64,
    /// Film coefficient in W/(m^2 K).
    pub h_w_per_m2k: f64,
    /// Sink temperature in °C.
    pub t_sink_c: f64,
}

/// A lumped volumetric sink spread over tagged elements.
///
/// Models a heat path that leaves along the third dimension (the
/// exposed shaft stubs): a whole-machine conductance `G` spread over
/// the tagged elements' volume. On a symmetry wedge the wedge's
/// share is `G / symmetry_mult`.
#[derive(Debug, Clone)]
pub struct VolumeSink {
    /// Report name of the sink.
    pub name: String,
    /// Cell markers carrying the sink.
    pub markers: Vec<i32>,
    /// Whole-machine conductance in W/K.
    pub g_w_per_k: f64,
    /// Sink temperature in °C.
    pub t_sink_c: f64,
    /// Wedge multiplicity of the mesh (1 for a full cross-section).
    pub symmetry_mult: f64,
}

/// A steady thermal problem on a triangular cross-section mesh.
#[derive(Debug, Clone)]
pub struct ThermalProblem {
    /// Tangential (circumferential) conductivity per element, W/(m K).
    pub k_tangential_w_per_mk: Vec<f64>,
    /// Radial conductivity per element; `None` = isotropic.
    pub k_radial_w_per_mk: Option<Vec<f64>>,
    /// Volumetric heat source per element in W/m^3.
    pub heat_source_w_per_m3: Vec<f64>,
    /// Cell markers to drop from the thermal domain (far-field air).
    pub drop_markers: Vec<i32>,
    /// Per-element keep mask, applied on top of `drop_markers`
    /// (empty keeps every non-dropped element). This is how a
    /// caller drops the far-field part of a marker that also covers
    /// kept regions — the motor mesh tags gap air, slot pockets and
    /// the outer far field with one air marker.
    pub element_keep: Vec<bool>,
    /// Cooled Robin surfaces by radius.
    pub surfaces: Vec<RobinSurface>,
    /// Lumped volumetric sinks.
    pub volume_sinks: Vec<VolumeSink>,
    /// Stack length in metres; scales the reported watts only.
    pub stack_length_m: f64,
}

/// Heat budget of one Robin surface.
#[derive(Debug, Clone)]
pub struct SurfaceReport {
    /// Surface name.
    pub name: String,
    /// Heat leaving through the film in watts.
    pub heat_removed_w: f64,
    /// Facet area in m^2 (perimeter times stack length).
    pub area_m2: f64,
    /// Area-mean wall temperature in °C (`None` without facets).
    pub t_mean_c: Option<f64>,
    /// Matched boundary facets.
    pub num_facets: usize,
}

/// Heat budget of one volume sink.
#[derive(Debug, Clone)]
pub struct SinkReport {
    /// Sink name.
    pub name: String,
    /// Heat removed by the sink in watts (mesh share).
    pub heat_removed_w: f64,
    /// Volume-mean temperature of the tagged elements in °C.
    pub t_mean_c: Option<f64>,
}

/// Solved steady temperature field and its heat budget.
#[derive(Debug, Clone)]
pub struct ThermalResult {
    /// Nodal temperature in °C at every input-mesh node; nodes used
    /// only by dropped elements (or P2 edge nodes) are `NaN`.
    pub temperature_c: Vec<f64>,
    /// Per-surface heat budgets.
    pub surfaces: Vec<SurfaceReport>,
    /// Per-sink heat budgets.
    pub sinks: Vec<SinkReport>,
    /// Heat generated in the solved domain in watts.
    pub heat_generated_w: f64,
    /// Generated minus removed heat in watts (solve closure error).
    pub heat_residual_w: f64,
}

/// Corner nodes and geometry helpers for one triangle.
struct Triangle {
    nodes: [usize; 3],
    /// P1 gradients `grad(phi_i)` (constant per element).
    grads: [[f64; 2]; 3],
    area_m2: f64,
}

fn triangle_geometry(mesh: &Mesh, nodes: [usize; 3]) -> Option<Triangle> {
    let p: [[f64; 2]; 3] = nodes.map(|node| {
        let point = mesh.nodes[node];
        [point.x, point.y]
    });
    let det = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]);
    if det.abs() <= 1.0e-300 {
        return None;
    }
    // grad phi_0 = (p1 - p2)^perp / det etc. with (x,y)^perp = (y,-x)
    // rotated consistently for the signed determinant.
    let grads = [
        [(p[1][1] - p[2][1]) / det, (p[2][0] - p[1][0]) / det],
        [(p[2][1] - p[0][1]) / det, (p[0][0] - p[2][0]) / det],
        [(p[0][1] - p[1][1]) / det, (p[1][0] - p[0][0]) / det],
    ];
    Some(Triangle {
        nodes,
        grads,
        area_m2: 0.5 * det.abs(),
    })
}

/// 3-point rule on the reference triangle: the permutations of
/// barycentric `(2/3, 1/6, 1/6)`, degree-2 exact. This is the rule
/// scikit-fem assembles `ElementTriP1` forms with, so the
/// non-polynomial tensor orientation `r_hat(x)` is sampled at the
/// same points as the reference and same-mesh results agree to
/// solver precision rather than to quadrature-rule choice.
const QUAD_BARY: [[f64; 3]; 3] = [
    [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
    [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
    [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
];

/// Solves a steady thermal problem.
///
/// # Errors
///
/// Returns [`ThermalError`] for inconsistent per-element data, a
/// mesh with no kept elements, a problem with no sink or surface
/// (singular), or a failed linear solve.
pub fn solve_steady_thermal(
    mesh: &Mesh,
    problem: &ThermalProblem,
) -> Result<ThermalResult, ThermalError> {
    let num_elements = mesh.num_elements();
    if problem.k_tangential_w_per_mk.len() != num_elements
        || problem.heat_source_w_per_m3.len() != num_elements
        || problem
            .k_radial_w_per_mk
            .as_ref()
            .is_some_and(|k| k.len() != num_elements)
    {
        return Err(ThermalError::ElementDataMismatch);
    }
    if !problem.stack_length_m.is_finite() || problem.stack_length_m <= 0.0 {
        return Err(ThermalError::InvalidStackLength);
    }
    let length = problem.stack_length_m;

    // Kept elements and the corner-node submesh renumbering.
    let mut kept: Vec<usize> = Vec::new();
    let mut remap: HashMap<usize, usize> = HashMap::new();
    let mut geometry: Vec<Triangle> = Vec::new();
    for (element_index, element) in mesh.elements.iter().enumerate() {
        if problem.drop_markers.contains(&element.marker)
            || problem
                .element_keep
                .get(element_index)
                .is_some_and(|keep| !keep)
        {
            continue;
        }
        let vertices = element.vertices();
        if vertices.len() != 3 {
            return Err(ThermalError::NonTriangularElement { element_index });
        }
        let corners = [vertices[0], vertices[1], vertices[2]];
        let Some(tri) = triangle_geometry(mesh, corners) else {
            return Err(ThermalError::DegenerateElement { element_index });
        };
        for &node in &corners {
            let next = remap.len();
            remap.entry(node).or_insert(next);
        }
        kept.push(element_index);
        geometry.push(tri);
    }
    if kept.is_empty() {
        return Err(ThermalError::EmptyDomain);
    }
    if problem.surfaces.is_empty() && problem.volume_sinks.is_empty() {
        return Err(ThermalError::NoHeatPath);
    }
    let num_unknowns = remap.len();
    let local = |tri: &Triangle| tri.nodes.map(|node| remap[&node]);

    // Volume-sink coefficient g [W/(K m^3)] per kept element.
    let mut sink_g: Vec<f64> = vec![0.0; kept.len()];
    let mut sink_rhs_factor: Vec<f64> = vec![0.0; kept.len()];
    let mut sink_areas: Vec<Vec<usize>> = Vec::with_capacity(problem.volume_sinks.len());
    for sink in &problem.volume_sinks {
        let members: Vec<usize> = kept
            .iter()
            .enumerate()
            .filter(|pair| sink.markers.contains(&mesh.elements[*pair.1].marker))
            .map(|(slot, _)| slot)
            .collect();
        let area: f64 = members.iter().map(|&slot| geometry[slot].area_m2).sum();
        if area > 0.0 {
            let mult = if sink.symmetry_mult > 0.0 {
                sink.symmetry_mult
            } else {
                1.0
            };
            let g = sink.g_w_per_k / (area * length * mult);
            for &slot in &members {
                sink_g[slot] += g;
                sink_rhs_factor[slot] += g * sink.t_sink_c;
            }
        }
        sink_areas.push(members);
    }

    let mut stiffness = TripletMatrix::new(num_unknowns, num_unknowns);
    let mut rhs = vec![0.0; num_unknowns];
    let mut heat_generated_w = 0.0;
    for (slot, &element_index) in kept.iter().enumerate() {
        let tri = &geometry[slot];
        let dofs = local(tri);
        let k_t = problem.k_tangential_w_per_mk[element_index];
        let k_r = problem
            .k_radial_w_per_mk
            .as_ref()
            .map_or(k_t, |k| k[element_index]);
        let q = problem.heat_source_w_per_m3[element_index];
        heat_generated_w += q * tri.area_m2 * length;
        let corners: [[f64; 2]; 3] = tri.nodes.map(|node| {
            let point = mesh.nodes[node];
            [point.x, point.y]
        });
        for bary in &QUAD_BARY {
            let weight = tri.area_m2 / 3.0;
            let position = [
                bary[0] * corners[0][0] + bary[1] * corners[1][0] + bary[2] * corners[2][0],
                bary[0] * corners[0][1] + bary[1] * corners[1][1] + bary[2] * corners[2][1],
            ];
            // Cylindrical-frame tensor k_t I + (k_r - k_t) r_hat r_hat^T.
            let radius = position[0].hypot(position[1]);
            let tensor = if radius > 1.0e-300 {
                let r_hat = [position[0] / radius, position[1] / radius];
                let delta = k_r - k_t;
                [
                    [
                        k_t + delta * r_hat[0] * r_hat[0],
                        delta * r_hat[0] * r_hat[1],
                    ],
                    [
                        delta * r_hat[1] * r_hat[0],
                        k_t + delta * r_hat[1] * r_hat[1],
                    ],
                ]
            } else {
                [[k_t, 0.0], [0.0, k_t]]
            };
            for i in 0..3 {
                rhs[dofs[i]] += (q + sink_rhs_factor[slot]) * bary[i] * weight;
                for j in 0..3 {
                    let kg = [
                        tensor[0][0] * tri.grads[j][0] + tensor[0][1] * tri.grads[j][1],
                        tensor[1][0] * tri.grads[j][0] + tensor[1][1] * tri.grads[j][1],
                    ];
                    let conduction = tri.grads[i][0] * kg[0] + tri.grads[i][1] * kg[1];
                    stiffness.push(
                        dofs[i],
                        dofs[j],
                        (conduction + sink_g[slot] * bary[i] * bary[j]) * weight,
                    );
                }
            }
        }
    }

    // Boundary facets of the kept submesh (edges used exactly once).
    let mut edge_count: HashMap<(usize, usize), (usize, [usize; 2])> = HashMap::new();
    for tri in &geometry {
        for (a, b) in [(0, 1), (1, 2), (2, 0)] {
            let edge = [tri.nodes[a], tri.nodes[b]];
            let key = (edge[0].min(edge[1]), edge[0].max(edge[1]));
            edge_count
                .entry(key)
                .and_modify(|entry| entry.0 += 1)
                .or_insert((1, edge));
        }
    }
    let boundary_edges: Vec<[usize; 2]> = edge_count
        .values()
        .filter(|entry| entry.0 == 1)
        .map(|entry| entry.1)
        .collect();

    // Robin films: 2-point Gauss on each matched facet.
    let gauss = [0.5 - 0.5 / 3.0f64.sqrt(), 0.5 + 0.5 / 3.0f64.sqrt()];
    let mut surface_facets: Vec<Vec<[usize; 2]>> = Vec::with_capacity(problem.surfaces.len());
    for surface in &problem.surfaces {
        let facets: Vec<[usize; 2]> = boundary_edges
            .iter()
            .copied()
            .filter(|edge| {
                edge.iter().all(|&node| {
                    let point = mesh.nodes[node];
                    (point.x.hypot(point.y) - surface.radius_m).abs() <= surface.radius_tol_m
                })
            })
            .collect();
        for edge in &facets {
            let dofs = edge.map(|node| remap[&node]);
            let pa = mesh.nodes[edge[0]];
            let pb = mesh.nodes[edge[1]];
            let edge_len = (pa.x - pb.x).hypot(pa.y - pb.y);
            for &xi in &gauss {
                let shape = [1.0 - xi, xi];
                let weight = 0.5 * edge_len;
                for i in 0..2 {
                    rhs[dofs[i]] += surface.h_w_per_m2k * surface.t_sink_c * shape[i] * weight;
                    for j in 0..2 {
                        stiffness.push(
                            dofs[i],
                            dofs[j],
                            surface.h_w_per_m2k * shape[i] * shape[j] * weight,
                        );
                    }
                }
            }
        }
        surface_facets.push(facets);
    }

    let config = sonium_solvers::SparseDirectConfig {
        matrix_class: sonium_solvers::MatrixClass::SymmetricPositiveDefinite,
        ..sonium_solvers::SparseDirectConfig::default()
    };
    let matrix = stiffness.to_csr();
    let factor = sonium_solvers::factorize_sparse(&matrix, &config, None)
        .map_err(|error| ThermalError::LinearSolve(error.to_string()))?;
    let solution = factor
        .solve(&rhs)
        .map_err(|error| ThermalError::LinearSolve(error.to_string()))?;

    // Temperature back on the input nodes.
    let mut temperature_c = vec![f64::NAN; mesh.num_nodes()];
    for (&node, &slot) in &remap {
        temperature_c[node] = solution[slot];
    }
    let node_t = |node: usize| solution[remap[&node]];

    // Surface budgets from the same facet integrals the solve used.
    let mut surfaces = Vec::with_capacity(problem.surfaces.len());
    let mut removed_w = 0.0;
    for (surface, facets) in problem.surfaces.iter().zip(&surface_facets) {
        let (mut heat, mut perimeter, mut t_integral) = (0.0, 0.0, 0.0);
        for edge in facets {
            let pa = mesh.nodes[edge[0]];
            let pb = mesh.nodes[edge[1]];
            let edge_len = (pa.x - pb.x).hypot(pa.y - pb.y);
            perimeter += edge_len;
            for &xi in &gauss {
                let t = (1.0 - xi) * node_t(edge[0]) + xi * node_t(edge[1]);
                heat += surface.h_w_per_m2k * (t - surface.t_sink_c) * 0.5 * edge_len;
                t_integral += t * 0.5 * edge_len;
            }
        }
        removed_w += heat * length;
        surfaces.push(SurfaceReport {
            name: surface.name.clone(),
            heat_removed_w: heat * length,
            area_m2: perimeter * length,
            t_mean_c: (perimeter > 0.0).then_some(t_integral / perimeter),
            num_facets: facets.len(),
        });
    }
    // Sink budgets: integral g (T - T_sink) over the tagged elements.
    let mut sinks = Vec::with_capacity(problem.volume_sinks.len());
    for (sink, members) in problem.volume_sinks.iter().zip(&sink_areas) {
        let area: f64 = members.iter().map(|&slot| geometry[slot].area_m2).sum();
        if area <= 0.0 {
            sinks.push(SinkReport {
                name: sink.name.clone(),
                heat_removed_w: 0.0,
                t_mean_c: None,
            });
            continue;
        }
        let mult = if sink.symmetry_mult > 0.0 {
            sink.symmetry_mult
        } else {
            1.0
        };
        let g = sink.g_w_per_k / (area * length * mult);
        let (mut heat, mut t_integral) = (0.0, 0.0);
        for &slot in members {
            let tri = &geometry[slot];
            for bary in &QUAD_BARY {
                let t: f64 = (0..3).map(|i| bary[i] * node_t(tri.nodes[i])).sum();
                heat += g * (t - sink.t_sink_c) * tri.area_m2 / 3.0;
                t_integral += t * tri.area_m2 / 3.0;
            }
        }
        removed_w += heat * length;
        sinks.push(SinkReport {
            name: sink.name.clone(),
            heat_removed_w: heat * length,
            t_mean_c: Some(t_integral / area),
        });
    }

    Ok(ThermalResult {
        temperature_c,
        surfaces,
        sinks,
        heat_generated_w,
        heat_residual_w: heat_generated_w - removed_w,
    })
}

/// Volume-mean temperature over the elements of one marker.
///
/// Uses corner-node temperatures; elements touching nodes the
/// thermal solve did not reach (dropped neighbours cannot occur —
/// the marker's own elements define the mean) contribute their
/// P1-exact element mean.
pub fn marker_mean_temperature_c(mesh: &Mesh, temperature_c: &[f64], marker: i32) -> Option<f64> {
    let (mut integral, mut area) = (0.0, 0.0);
    for element in mesh.elements.iter().filter(|e| e.marker == marker) {
        let vertices = element.vertices();
        if vertices.len() != 3 {
            continue;
        }
        let corners = [vertices[0], vertices[1], vertices[2]];
        let p: [[f64; 2]; 3] = corners.map(|node| {
            let point = mesh.nodes[node];
            [point.x, point.y]
        });
        let element_area = 0.5
            * ((p[1][0] - p[0][0]) * (p[2][1] - p[0][1])
                - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]))
                .abs();
        let mean =
            (temperature_c[corners[0]] + temperature_c[corners[1]] + temperature_c[corners[2]])
                / 3.0;
        if mean.is_finite() {
            integral += mean * element_area;
            area += element_area;
        }
    }
    (area > 0.0).then_some(integral / area)
}

/// One loss region: total watts spread uniformly over its volume.
#[derive(Debug, Clone)]
pub struct RegionHeat {
    /// Cell markers forming the region.
    pub markers: Vec<i32>,
    /// Total dissipation in the region, in watts (whole stack).
    pub watts: f64,
}

/// Corner-triangle areas of every mesh element in m^2 (0 for
/// non-triangular elements).
pub fn element_areas_m2(mesh: &Mesh) -> Vec<f64> {
    mesh.elements
        .iter()
        .map(|element| {
            let vertices = element.vertices();
            if vertices.len() != 3 {
                return 0.0;
            }
            let p: [[f64; 2]; 3] = [
                [mesh.nodes[vertices[0]].x, mesh.nodes[vertices[0]].y],
                [mesh.nodes[vertices[1]].x, mesh.nodes[vertices[1]].y],
                [mesh.nodes[vertices[2]].x, mesh.nodes[vertices[2]].y],
            ];
            0.5 * ((p[1][0] - p[0][0]) * (p[2][1] - p[0][1])
                - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]))
                .abs()
        })
        .collect()
}

/// Element centroids of the corner triangles.
pub fn element_centroids_m(mesh: &Mesh) -> Vec<[f64; 2]> {
    mesh.elements
        .iter()
        .map(|element| {
            let vertices = element.vertices();
            let mut centroid = [0.0; 2];
            for &node in vertices {
                centroid[0] += mesh.nodes[node].x;
                centroid[1] += mesh.nodes[node].y;
            }
            let n = vertices.len().max(1) as f64;
            [centroid[0] / n, centroid[1] / n]
        })
        .collect()
}

/// Per-element volumetric heat source `q` [W/m^3] from region loss
/// totals: each region's watts spread uniformly over its elements'
/// volume (`area * stack_length_m`). This is the EM→thermal loss
/// hand-off the coupled loop uses: iron loss into the laminations,
/// copper loss into the wires, eddy losses into magnets and shaft.
/// Elements in no region get zero; overlapping regions add.
pub fn heat_sources_from_regions(
    mesh: &Mesh,
    regions: &[RegionHeat],
    stack_length_m: f64,
) -> Vec<f64> {
    let areas = element_areas_m2(mesh);
    let mut q = vec![0.0; mesh.num_elements()];
    for region in regions {
        let volume: f64 = mesh
            .elements
            .iter()
            .enumerate()
            .filter(|(_, element)| region.markers.contains(&element.marker))
            .map(|(index, _)| areas[index] * stack_length_m)
            .sum();
        if volume <= 0.0 {
            continue;
        }
        let density = region.watts / volume;
        for (index, element) in mesh.elements.iter().enumerate() {
            if region.markers.contains(&element.marker) {
                q[index] += density;
            }
        }
    }
    q
}

/// Failure of a steady thermal solve.
#[derive(Debug, thiserror::Error)]
pub enum ThermalError {
    /// Per-element arrays did not match the element count.
    #[error("per-element thermal data does not match the mesh element count")]
    ElementDataMismatch,
    /// Stack length was non-positive or non-finite.
    #[error("stack length must be finite and positive")]
    InvalidStackLength,
    /// An element was not a triangle.
    #[error("thermal solve needs triangles; element {element_index} is not one")]
    NonTriangularElement {
        /// Offending element index.
        element_index: usize,
    },
    /// An element had zero area.
    #[error("thermal mesh element {element_index} is degenerate")]
    DegenerateElement {
        /// Offending element index.
        element_index: usize,
    },
    /// Every element was dropped.
    #[error("no solid elements left after dropping the air domains")]
    EmptyDomain,
    /// No Robin surface and no volume sink: the problem is singular.
    #[error("a steady thermal problem needs at least one cooled surface or sink")]
    NoHeatPath,
    /// The sparse direct solve failed.
    #[error("thermal linear solve failed: {0}")]
    LinearSolve(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{ElementType, Point};

    /// Structured polar annulus mesh between `r_in` and `r_out`.
    fn annulus(r_in: f64, r_out: f64, rings: usize, sectors: usize) -> Mesh {
        let mut mesh = Mesh::new(2);
        for ring in 0..=rings {
            let radius = r_in + (r_out - r_in) * ring as f64 / rings as f64;
            for sector in 0..sectors {
                let angle = 2.0 * std::f64::consts::PI * sector as f64 / sectors as f64;
                mesh.add_node(Point::new_2d(radius * angle.cos(), radius * angle.sin()));
            }
        }
        let node = |ring: usize, sector: usize| ring * sectors + sector % sectors;
        for ring in 0..rings {
            for sector in 0..sectors {
                let (a, b) = (node(ring, sector), node(ring, sector + 1));
                let (c, d) = (node(ring + 1, sector), node(ring + 1, sector + 1));
                mesh.add_element_with_marker(ElementType::Triangle, vec![a, d, c], 1);
                mesh.add_element_with_marker(ElementType::Triangle, vec![a, b, d], 1);
            }
        }
        mesh
    }

    fn problem(mesh: &Mesh, k_t: f64, k_r: Option<f64>, q: f64) -> ThermalProblem {
        problem_h(mesh, k_t, k_r, q, 1.0e7)
    }

    fn problem_h(mesh: &Mesh, k_t: f64, k_r: Option<f64>, q: f64, h: f64) -> ThermalProblem {
        let n = mesh.num_elements();
        ThermalProblem {
            k_tangential_w_per_mk: vec![k_t; n],
            k_radial_w_per_mk: k_r.map(|k| vec![k; n]),
            heat_source_w_per_m3: vec![q; n],
            drop_markers: Vec::new(),
            element_keep: Vec::new(),
            surfaces: vec![
                RobinSurface {
                    name: "inner".to_string(),
                    radius_m: 0.01,
                    radius_tol_m: 1.0e-9,
                    h_w_per_m2k: h,
                    t_sink_c: 20.0,
                },
                RobinSurface {
                    name: "outer".to_string(),
                    radius_m: 0.02,
                    radius_tol_m: 1.0e-9,
                    h_w_per_m2k: h,
                    t_sink_c: 100.0,
                },
            ],
            volume_sinks: Vec::new(),
            stack_length_m: 0.01,
        }
    }

    #[test]
    fn radial_conduction_matches_log_profile() {
        // Huge films make the Robin surfaces near-Dirichlet; with no
        // source the exact field is T = 20 + 80 ln(r/r_in)/ln(2).
        let mesh = annulus(0.01, 0.02, 40, 24);
        let result =
            solve_steady_thermal(&mesh, &problem(&mesh, 5.0, None, 0.0)).expect("thermal solve");
        let mid = mesh
            .nodes
            .iter()
            .position(|node| (node.x.hypot(node.y) - 0.015).abs() < 1.0e-9)
            .expect("mid-radius node");
        let exact = 20.0 + 80.0 * (1.5f64).ln() / (2.0f64).ln();
        let got = result.temperature_c[mid];
        assert!(
            (got - exact).abs() < 0.35,
            "mid-radius temperature {got} vs exact {exact}"
        );
    }

    #[test]
    fn radial_heat_flow_sees_only_k_radial() {
        // Purely radial heat flow is carried by k_r alone. With
        // films in series the exact flux is
        // Q = dT / (1/(h A_in) + ln(2)/(2 pi L k_r) + 1/(h A_out)).
        // Multiplying k_t by ten must not move it (beyond
        // triangulation noise: the discrete field is not exactly
        // radial), while the flux tracks k_r through the series sum.
        // A moderate h keeps the film temperature drop far above
        // that noise; at near-Dirichlet h the film integral would
        // amplify millikelvin field noise into the flux. The sector
        // count is high because a polygonal annulus is thinner at
        // chord midpoints than at nodes, which inflates the radial
        // conductance by O((pi/sectors)^2) on a coarse fan.
        let mesh = annulus(0.01, 0.02, 20, 96);
        let solve = |k_t: f64, k_r: f64| {
            solve_steady_thermal(&mesh, &problem_h(&mesh, k_t, Some(k_r), 0.0, 2.0e3))
                .expect("thermal solve")
        };
        let flux = |result: &ThermalResult| result.surfaces[0].heat_removed_w.abs();
        let exact = |k_r: f64| {
            let (length, h) = (0.01, 2.0e3);
            let a_in = 2.0 * std::f64::consts::PI * 0.01 * length;
            let a_out = 2.0 * std::f64::consts::PI * 0.02 * length;
            80.0 / (1.0 / (h * a_in)
                + (2.0f64).ln() / (2.0 * std::f64::consts::PI * length * k_r)
                + 1.0 / (h * a_out))
        };
        let base = solve(5.0, 0.8);
        assert!(
            (flux(&base) - exact(0.8)).abs() / exact(0.8) < 0.01,
            "radial flux {} vs exact {}",
            flux(&base),
            exact(0.8)
        );
        // Continuum radial flow ignores k_t entirely; on triangles
        // the P1 gradient of a radial nodal field is not exactly
        // radial inside an element, so at an extreme 62:1 anisotropy
        // a percent-level discrete coupling remains. What must not
        // happen is the tensor axes being swapped: that is pinned by
        // the swapped solve below matching exact(k_r = 50).
        let spun = solve(50.0, 0.8);
        let rel = (flux(&spun) - flux(&base)).abs() / flux(&base);
        assert!(
            rel < 0.03,
            "k_t x10 moved the radial flux by {rel:.2e} relative"
        );
        let swapped = solve(0.8, 50.0);
        assert!(
            (flux(&swapped) - exact(50.0)).abs() / exact(50.0) < 0.01,
            "swapped-axes flux {} vs exact {} (tensor orientation)",
            flux(&swapped),
            exact(50.0)
        );
        let stiff = solve(5.0, 4.0);
        assert!(
            (flux(&stiff) - exact(4.0)).abs() / exact(4.0) < 0.01,
            "stiff flux {} vs exact {}",
            flux(&stiff),
            exact(4.0)
        );
    }

    #[test]
    fn heat_budget_closes_with_internal_generation() {
        let mesh = annulus(0.01, 0.02, 16, 16);
        let result =
            solve_steady_thermal(&mesh, &problem(&mesh, 5.0, None, 2.0e7)).expect("thermal solve");
        assert!(result.heat_generated_w > 0.0);
        assert!(
            result.heat_residual_w.abs() < 1.0e-6 * result.heat_generated_w,
            "budget residual {} W of {} W generated",
            result.heat_residual_w,
            result.heat_generated_w
        );
        // Surface identity: removed == h * A * (t_mean - t_sink).
        for (surface, report) in problem(&mesh, 5.0, None, 0.0)
            .surfaces
            .iter()
            .zip(&result.surfaces)
        {
            let t_mean = report.t_mean_c.expect("facets matched");
            let identity = surface.h_w_per_m2k * report.area_m2 * (t_mean - surface.t_sink_c);
            assert!(
                (identity - report.heat_removed_w).abs()
                    < 1.0e-9 * report.heat_removed_w.abs().max(1.0),
                "surface {} identity broken: {identity} vs {}",
                surface.name,
                report.heat_removed_w
            );
        }
    }

    #[test]
    fn invalid_problems_are_rejected() {
        let mesh = annulus(0.01, 0.02, 2, 4);
        let mut bad = problem(&mesh, 5.0, None, 0.0);
        bad.k_tangential_w_per_mk.pop();
        assert!(solve_steady_thermal(&mesh, &bad).is_err());
        let mut no_path = problem(&mesh, 5.0, None, 0.0);
        no_path.surfaces.clear();
        assert!(solve_steady_thermal(&mesh, &no_path).is_err());
    }
}
