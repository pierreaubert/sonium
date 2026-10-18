//! Mesh generators for common domains
//!
//! Provides functions to create structured meshes for rectangles, boxes, and circles.

use super::types::{BoundaryType, Element, ElementType, Mesh, Point};
use super::validate::{MeshValidationError, validate_mesh_admission};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Admission failure from the `try_*` structured-grid generators.
///
/// A zero cell count would divide the span by zero and fill the mesh with
/// infinite or NaN coordinates; non-finite extents propagate NaNs through
/// every node. Both are rejected here so untrusted browser/Python input has
/// a fallible entry point. The infallible convenience wrappers panic with
/// the same message instead of emitting a garbage mesh.
///
/// Counts are additionally bounded by an output byte budget
/// ([`StructuredGridError::MAX_GENERATOR_BYTES`]): zero/non-finite checks
/// alone still admit `usize::MAX` counts, which pass admission and then hang
/// or abort in allocation (observed: SIGABRT on a 6 GiB `Vec` growth from a
/// single hostile `nx`). The budget rejects absurd counts before any
/// allocation while admitting every plausible workload (a 100³ box is ~512
/// MiB under the conservative per-shape estimates).
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum StructuredGridError {
    /// A cell count of zero was supplied for the named axis.
    #[error("structured generator requires at least one cell on axis {axis}, got {actual}")]
    ZeroResolution { axis: &'static str, actual: usize },
    /// A non-finite extent was supplied for the named axis.
    #[error("structured generator extents must be finite on axis {axis}, got [{min}, {max}]")]
    NonFiniteExtent {
        axis: &'static str,
        min: f64,
        max: f64,
    },
    /// The implied output exceeds the generator byte budget (hostile or
    /// mistaken counts). `cells` is the checked cell product (`u64::MAX`
    /// when the product itself overflows `u64`); `max_cells` is the budget
    /// for this shape.
    #[error("structured generator {shape} needs {cells} cells, over budget {max_cells}")]
    CellBudgetExceeded {
        shape: &'static str,
        cells: u64,
        max_cells: u64,
    },
    /// A strict-inequality extent requirement failed (empty span or
    /// non-positive radius): the generator would divide by zero and emit
    /// infinite coordinates, or build a coincident degenerate mesh.
    #[error("structured generator extents degenerate on axis {axis}: [{min}, {max}]")]
    DegenerateExtent {
        axis: &'static str,
        min: f64,
        max: f64,
    },
    /// A binary-mask input is empty or ragged: counts are derived from the
    /// mask itself, so shape violations are admission failures, not panics.
    #[error("binary mask invalid: {detail}")]
    InvalidMask { detail: &'static str },
}

impl StructuredGridError {
    /// Total output budget for one generator call (1 GiB of estimated mesh
    /// storage). Per-shape byte-per-cell estimates are deliberately
    /// conservative upper bounds, so the budget errs toward admitting real
    /// workloads and only rejects absurd counts.
    pub const MAX_GENERATOR_BYTES: u64 = 1 << 30;
    /// Conservative bytes per background cell for 2D triangle/quad meshes
    /// (~2 elements/cell, node + connectivity storage).
    pub const BYTES_PER_CELL_2D: u64 = 128;
    /// Conservative bytes per background cell for 3D tet/hex meshes
    /// (up to 6 elements/cell).
    pub const BYTES_PER_CELL_3D: u64 = 512;
    /// Conservative bytes per icosphere base-triangle × radial-layer unit
    /// for spherical shells (prism split into tetrahedra plus the layer
    /// nodes amortized over small layer counts).
    pub const BYTES_PER_SHELL_UNIT: u64 = 1024;

    /// Reject a checked cell product that exceeds the byte budget for a
    /// shape. `cells` uses `u64::MAX` as the overflow sentinel: a product
    /// that does not fit `u64` is unconditionally over budget.
    pub fn check_cell_budget(
        shape: &'static str,
        cells: u64,
        bytes_per_cell: u64,
    ) -> Result<(), Self> {
        let max_cells = Self::MAX_GENERATOR_BYTES / bytes_per_cell;
        if cells > max_cells {
            return Err(Self::CellBudgetExceeded {
                shape,
                cells,
                max_cells,
            });
        }
        Ok(())
    }

    /// Checked product of count factors: any overflow saturates to
    /// `u64::MAX`, which unconditionally fails the budget check.
    pub fn checked_cells(factors: &[usize]) -> u64 {
        let mut cells = 1u64;
        for &f in factors {
            cells = cells.saturating_mul(f as u64);
            if cells == u64::MAX {
                break;
            }
        }
        cells
    }

    /// Conservative element multiplier for `levels` of conforming boundary
    /// refinement: each level can quadruple the selected elements, so the
    /// worst case is `4^levels` over the base cells. `levels >= 32`
    /// saturates (`4^32` exceeds `u64`), unconditionally failing the budget.
    pub fn refinement_multiplier(levels: usize) -> u64 {
        if levels >= 32 {
            u64::MAX
        } else {
            4u64.pow(levels as u32)
        }
    }

    fn check_2d(
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
        nx: usize,
        ny: usize,
    ) -> Result<(), Self> {
        if nx == 0 {
            return Err(Self::ZeroResolution {
                axis: "x",
                actual: nx,
            });
        }
        if ny == 0 {
            return Err(Self::ZeroResolution {
                axis: "y",
                actual: ny,
            });
        }
        Self::check_cell_budget(
            "structured 2D grid",
            Self::checked_cells(&[nx, ny]),
            Self::BYTES_PER_CELL_2D,
        )?;
        if !x_min.is_finite() || !x_max.is_finite() {
            return Err(Self::NonFiniteExtent {
                axis: "x",
                min: x_min,
                max: x_max,
            });
        }
        if !y_min.is_finite() || !y_max.is_finite() {
            return Err(Self::NonFiniteExtent {
                axis: "y",
                min: y_min,
                max: y_max,
            });
        }
        Ok(())
    }

    fn check_3d(
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
        z_min: f64,
        z_max: f64,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> Result<(), Self> {
        Self::check_2d(x_min, x_max, y_min, y_max, nx, ny)?;
        if nz == 0 {
            return Err(Self::ZeroResolution {
                axis: "z",
                actual: nz,
            });
        }
        // check_2d already bounded nx*ny; extend to the 3D product here so
        // the 3D budget (not the looser 2D one) governs boxes.
        Self::check_cell_budget(
            "structured 3D grid",
            Self::checked_cells(&[nx, ny, nz]),
            Self::BYTES_PER_CELL_3D,
        )?;
        if !z_min.is_finite() || !z_max.is_finite() {
            return Err(Self::NonFiniteExtent {
                axis: "z",
                min: z_min,
                max: z_max,
            });
        }
        Ok(())
    }
}

/// Generate a rectangular mesh with triangular elements
pub fn rectangular_mesh_triangles(
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    nx: usize,
    ny: usize,
) -> Mesh {
    try_rectangular_mesh_triangles(x_min, x_max, y_min, y_max, nx, ny)
        .unwrap_or_else(|error| panic!("{error}"))
}

/// Fallible [`rectangular_mesh_triangles`] for externally supplied
/// counts/extents: zero counts and non-finite extents return
/// [`StructuredGridError`] instead of emitting infinite/NaN nodes.
pub fn try_rectangular_mesh_triangles(
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    nx: usize,
    ny: usize,
) -> Result<Mesh, StructuredGridError> {
    StructuredGridError::check_2d(x_min, x_max, y_min, y_max, nx, ny)?;
    let mut mesh = Mesh::new(2);

    let dx = (x_max - x_min) / nx as f64;
    let dy = (y_max - y_min) / ny as f64;

    // Create nodes
    for j in 0..=ny {
        for i in 0..=nx {
            let x = x_min + i as f64 * dx;
            let y = y_min + j as f64 * dy;
            mesh.add_node(Point::new_2d(x, y));
        }
    }

    // Create triangular elements (2 triangles per cell)
    for j in 0..ny {
        for i in 0..nx {
            let n00 = j * (nx + 1) + i;
            let n10 = n00 + 1;
            let n01 = n00 + (nx + 1);
            let n11 = n01 + 1;

            // Two triangles per cell
            mesh.add_element(ElementType::Triangle, vec![n00, n10, n11]);
            mesh.add_element(ElementType::Triangle, vec![n00, n11, n01]);
        }
    }

    // Detect boundaries
    mesh.detect_boundaries();

    // Set boundary conditions based on position
    let tol = 1e-10;
    mesh.set_boundary_condition(BoundaryType::Dirichlet, 1, |points| {
        points.iter().all(|p| (p.x - x_min).abs() < tol)
    });
    mesh.set_boundary_condition(BoundaryType::Dirichlet, 2, |points| {
        points.iter().all(|p| (p.x - x_max).abs() < tol)
    });
    mesh.set_boundary_condition(BoundaryType::Neumann, 3, |points| {
        points.iter().all(|p| (p.y - y_min).abs() < tol)
    });
    mesh.set_boundary_condition(BoundaryType::Neumann, 4, |points| {
        points.iter().all(|p| (p.y - y_max).abs() < tol)
    });

    Ok(mesh)
}

/// Mesh a planar implicit region on a structured Cartesian background grid.
///
/// A cell is retained when its center satisfies `inside`. Retained cells are
/// split into two triangles, unused background nodes are removed, and every
/// exposed edge receives the requested Dirichlet boundary marker. This gives
/// a deliberately simple, deterministic mesh for moving-domain studies and
/// pixel/level-set regions where a boundary-conforming CAD mesh is unnecessary.
///
/// Panics on empty spans, zero counts or over-budget grids; prefer
/// [`try_implicit_region_mesh_triangles`] when the inputs come from outside
/// the call site.
#[allow(clippy::too_many_arguments)]
pub fn implicit_region_mesh_triangles<F>(
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    nx: usize,
    ny: usize,
    boundary_marker: i32,
    inside: F,
) -> Mesh
where
    F: Fn(f64, f64) -> bool,
{
    assert!(
        x_min < x_max && y_min < y_max,
        "region bounds must be nonempty"
    );
    assert!(nx > 0 && ny > 0, "region grid resolution must be positive");
    // Fail-fast budget gate for the infallible entry (untrusted input must
    // use try_implicit_region_mesh_triangles): hostile counts panic here
    // instead of hanging in the allocation below.
    StructuredGridError::check_cell_budget(
        "implicit region grid",
        StructuredGridError::checked_cells(&[nx, ny]),
        StructuredGridError::BYTES_PER_CELL_2D,
    )
    .expect("implicit region grid over cell budget; use try_implicit_region_mesh_triangles");

    let dx = (x_max - x_min) / nx as f64;
    let dy = (y_max - y_min) / ny as f64;
    let mut retained_cells = Vec::new();
    let mut used_nodes = vec![false; (nx + 1) * (ny + 1)];
    for j in 0..ny {
        for i in 0..nx {
            let x = x_min + (i as f64 + 0.5) * dx;
            let y = y_min + (j as f64 + 0.5) * dy;
            if inside(x, y) {
                let n00 = j * (nx + 1) + i;
                let cell = [n00, n00 + 1, n00 + nx + 1, n00 + nx + 2];
                for node in cell {
                    used_nodes[node] = true;
                }
                retained_cells.push(cell);
            }
        }
    }

    let mut mesh = Mesh::new(2);
    let mut compact = vec![usize::MAX; used_nodes.len()];
    for (background, used) in used_nodes.into_iter().enumerate() {
        if used {
            let i = background % (nx + 1);
            let j = background / (nx + 1);
            compact[background] =
                mesh.add_node(Point::new_2d(x_min + i as f64 * dx, y_min + j as f64 * dy));
        }
    }
    for [n00, n10, n01, n11] in retained_cells {
        mesh.add_element(
            ElementType::Triangle,
            vec![compact[n00], compact[n10], compact[n11]],
        );
        mesh.add_element(
            ElementType::Triangle,
            vec![compact[n00], compact[n11], compact[n01]],
        );
    }
    mesh.detect_boundaries();
    for boundary in &mut mesh.boundaries {
        boundary.boundary_type = BoundaryType::Dirichlet;
        boundary.marker = boundary_marker;
    }
    mesh
}

/// Fallible [`implicit_region_mesh_triangles`] for externally supplied
/// counts/extents: zero counts, empty spans, non-finite extents and
/// over-budget grids return [`StructuredGridError`] instead of panicking,
/// hanging, or emitting infinite-coordinate meshes.
#[allow(clippy::too_many_arguments)]
pub fn try_implicit_region_mesh_triangles<F>(
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    nx: usize,
    ny: usize,
    boundary_marker: i32,
    inside: F,
) -> Result<Mesh, StructuredGridError>
where
    F: Fn(f64, f64) -> bool,
{
    if nx == 0 {
        return Err(StructuredGridError::ZeroResolution {
            axis: "x",
            actual: nx,
        });
    }
    if ny == 0 {
        return Err(StructuredGridError::ZeroResolution {
            axis: "y",
            actual: ny,
        });
    }
    if !x_min.is_finite() || !x_max.is_finite() {
        return Err(StructuredGridError::NonFiniteExtent {
            axis: "x",
            min: x_min,
            max: x_max,
        });
    }
    if !y_min.is_finite() || !y_max.is_finite() {
        return Err(StructuredGridError::NonFiniteExtent {
            axis: "y",
            min: y_min,
            max: y_max,
        });
    }
    if x_min >= x_max {
        return Err(StructuredGridError::DegenerateExtent {
            axis: "x",
            min: x_min,
            max: x_max,
        });
    }
    if y_min >= y_max {
        return Err(StructuredGridError::DegenerateExtent {
            axis: "y",
            min: y_min,
            max: y_max,
        });
    }
    StructuredGridError::check_cell_budget(
        "implicit region grid",
        StructuredGridError::checked_cells(&[nx, ny]),
        StructuredGridError::BYTES_PER_CELL_2D,
    )?;
    Ok(implicit_region_mesh_triangles(
        x_min,
        x_max,
        y_min,
        y_max,
        nx,
        ny,
        boundary_marker,
        inside,
    ))
}

/// Options for boundary-fitted implicit-region meshing.
#[derive(Debug, Clone, Copy)]
pub struct FittedImplicitRegionConfig {
    /// Cartesian background-grid cells in x.
    pub nx: usize,
    /// Cartesian background-grid cells in y.
    pub ny: usize,
    /// Number of conforming refinements of boundary-adjacent elements.
    pub boundary_refinement_levels: usize,
    /// Maximum Newton projections used to place boundary nodes on `level_set = 0`.
    pub projection_iterations: usize,
}

impl FittedImplicitRegionConfig {
    pub fn new(nx: usize, ny: usize) -> Self {
        Self {
            nx,
            ny,
            boundary_refinement_levels: 1,
            projection_iterations: 12,
        }
    }
}

/// Mesh a smooth planar level-set region with a fitted, locally refined boundary.
///
/// Negative level-set values are inside. The deterministic Cartesian cut-cell
/// mesh is refined only next to exposed edges; boundary vertices are then
/// projected to the zero contour with a safeguarded finite-difference Newton
/// step. This is the geometry-convergent counterpart to
/// [`implicit_region_mesh_triangles`], whose staircase semantics remain useful
/// for exact pixel regions.
#[allow(clippy::too_many_arguments)]
pub fn fitted_implicit_region_mesh_triangles<F>(
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    boundary_marker: i32,
    config: FittedImplicitRegionConfig,
    level_set: F,
) -> Mesh
where
    F: Fn(f64, f64) -> f64,
{
    assert!(
        config.projection_iterations > 0,
        "boundary projection needs at least one iteration"
    );
    // Fail-fast budget gate for the infallible entry: refinement levels
    // multiply the background cells (up to 4^levels), so the gate must
    // account for them — the inner background-grid check alone does not.
    // Untrusted input must use try_fitted_implicit_region_mesh_triangles.
    {
        let base = StructuredGridError::checked_cells(&[config.nx, config.ny]);
        let mult = StructuredGridError::refinement_multiplier(config.boundary_refinement_levels);
        let cells = base.saturating_mul(mult);
        StructuredGridError::check_cell_budget(
            "fitted implicit grid",
            cells,
            StructuredGridError::BYTES_PER_CELL_2D,
        )
        .expect(
            "fitted implicit grid over cell budget; use try_fitted_implicit_region_mesh_triangles",
        );
    }
    let mut mesh = try_implicit_region_mesh_triangles(
        x_min,
        x_max,
        y_min,
        y_max,
        config.nx,
        config.ny,
        boundary_marker,
        |x, y| level_set(x, y) <= 0.0,
    )
    .unwrap_or_else(|error| panic!("fitted implicit background grid invalid: {error}"));

    for _ in 0..config.boundary_refinement_levels {
        let mut boundary_elements: Vec<_> = mesh
            .boundaries
            .iter()
            .filter(|boundary| boundary.marker == boundary_marker)
            .map(|boundary| boundary.element_idx)
            .collect();
        boundary_elements.sort_unstable();
        boundary_elements.dedup();
        if boundary_elements.is_empty() {
            break;
        }
        refine_boundary_triangles_conforming(&mut mesh, &boundary_elements, boundary_marker);
    }

    project_level_set_boundary(
        &mut mesh,
        boundary_marker,
        x_max - x_min,
        y_max - y_min,
        config,
        &level_set,
    );
    mesh
}

/// Fallible [`fitted_implicit_region_mesh_triangles`] for externally
/// supplied configs: zero counts and over-budget background grids —
/// including the refinement-level multiplier — return
/// [`StructuredGridError`] instead of panicking or hanging. Extent
/// admission is enforced by the inner
/// [`try_implicit_region_mesh_triangles`] call.
pub fn try_fitted_implicit_region_mesh_triangles<F>(
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    boundary_marker: i32,
    config: FittedImplicitRegionConfig,
    level_set: F,
) -> Result<Mesh, StructuredGridError>
where
    F: Fn(f64, f64) -> f64,
{
    let base = StructuredGridError::checked_cells(&[config.nx, config.ny]);
    let mult = StructuredGridError::refinement_multiplier(config.boundary_refinement_levels);
    let cells = base.saturating_mul(mult);
    StructuredGridError::check_cell_budget(
        "fitted implicit grid",
        cells,
        StructuredGridError::BYTES_PER_CELL_2D,
    )?;
    Ok(fitted_implicit_region_mesh_triangles(
        x_min,
        x_max,
        y_min,
        y_max,
        boundary_marker,
        config,
        level_set,
    ))
}

fn refine_boundary_triangles_conforming(
    mesh: &mut Mesh,
    selected_elements: &[usize],
    boundary_marker: i32,
) {
    let selected: HashSet<_> = selected_elements.iter().copied().collect();
    let edge_key = |a: usize, b: usize| if a < b { (a, b) } else { (b, a) };
    let mut split_edges = HashSet::new();
    for &element_index in &selected {
        let vertices = mesh.elements[element_index].vertices();
        split_edges.insert(edge_key(vertices[0], vertices[1]));
        split_edges.insert(edge_key(vertices[1], vertices[2]));
        split_edges.insert(edge_key(vertices[2], vertices[0]));
    }

    let mut midpoint = HashMap::new();
    let mut ordered_edges: Vec<_> = split_edges.iter().copied().collect();
    ordered_edges.sort_unstable();
    for (first, second) in ordered_edges {
        let a = mesh.nodes[first];
        let b = mesh.nodes[second];
        let node = mesh.add_node(Point::new_2d(0.5 * (a.x + b.x), 0.5 * (a.y + b.y)));
        midpoint.insert((first, second), node);
    }

    let original = std::mem::take(&mut mesh.elements);
    let mut next_id = original
        .iter()
        .map(|element| element.id)
        .max()
        .unwrap_or(0)
        .saturating_add(1);
    let mut refined = Vec::with_capacity(original.len() + 3 * selected.len());
    for (element_index, element) in original.into_iter().enumerate() {
        assert_eq!(element.element_type, ElementType::Triangle);
        let v = element.vertices();
        let mids = [
            midpoint.get(&edge_key(v[0], v[1])).copied(),
            midpoint.get(&edge_key(v[1], v[2])).copied(),
            midpoint.get(&edge_key(v[2], v[0])).copied(),
        ];
        let children: Vec<[usize; 3]> = match mids {
            [None, None, None] => {
                refined.push(element);
                continue;
            }
            [Some(m01), None, None] => vec![[v[0], m01, v[2]], [m01, v[1], v[2]]],
            [None, Some(m12), None] => vec![[v[1], m12, v[0]], [m12, v[2], v[0]]],
            [None, None, Some(m20)] => vec![[v[2], m20, v[1]], [m20, v[0], v[1]]],
            [Some(m01), Some(m12), None] => {
                vec![[m01, v[1], m12], [v[0], m01, v[2]], [m01, m12, v[2]]]
            }
            [None, Some(m12), Some(m20)] => {
                vec![[m12, v[2], m20], [v[1], m12, v[0]], [m12, m20, v[0]]]
            }
            [Some(m01), None, Some(m20)] => {
                vec![[m20, v[0], m01], [v[2], m20, v[1]], [m20, m01, v[1]]]
            }
            [Some(m01), Some(m12), Some(m20)] => vec![
                [v[0], m01, m20],
                [m01, v[1], m12],
                [m20, m12, v[2]],
                [m01, m12, m20],
            ],
        };
        for nodes in children {
            let mut child = Element::new(ElementType::Triangle, nodes.to_vec(), next_id);
            next_id = next_id.saturating_add(1);
            child.parent_id = Some(element.id);
            child.level = element.level + 1;
            child.marker = element.marker;
            refined.push(child);
        }
        debug_assert!(selected.contains(&element_index) || mids.iter().any(Option::is_some));
    }
    mesh.elements = refined;
    mesh.detect_boundaries();
    for boundary in &mut mesh.boundaries {
        boundary.boundary_type = BoundaryType::Dirichlet;
        boundary.marker = boundary_marker;
    }
    mesh.invalidate_caches();
}

fn project_level_set_boundary<F>(
    mesh: &mut Mesh,
    boundary_marker: i32,
    width: f64,
    height: f64,
    config: FittedImplicitRegionConfig,
    level_set: &F,
) where
    F: Fn(f64, f64) -> f64,
{
    let mut boundary_nodes: Vec<_> = mesh
        .boundaries
        .iter()
        .filter(|boundary| boundary.marker == boundary_marker)
        .flat_map(|boundary| boundary.nodes.iter().copied())
        .collect();
    boundary_nodes.sort_unstable();
    boundary_nodes.dedup();
    let scale = width.min(height);
    let derivative_step = scale * 1.0e-6;
    let local_spacing = (width / config.nx as f64).min(height / config.ny as f64)
        / 2.0_f64.powi(config.boundary_refinement_levels as i32);
    let maximum_step = 0.45 * local_spacing;
    let tolerance = 64.0 * f64::EPSILON * scale.max(1.0);
    let original = mesh.nodes.clone();

    for &node in &boundary_nodes {
        let mut x = mesh.nodes[node].x;
        let mut y = mesh.nodes[node].y;
        for _ in 0..config.projection_iterations {
            let value = level_set(x, y);
            if value.abs() <= tolerance {
                break;
            }
            let gx = (level_set(x + derivative_step, y) - level_set(x - derivative_step, y))
                / (2.0 * derivative_step);
            let gy = (level_set(x, y + derivative_step) - level_set(x, y - derivative_step))
                / (2.0 * derivative_step);
            let gradient_squared = gx * gx + gy * gy;
            if !gradient_squared.is_finite() || gradient_squared <= f64::EPSILON {
                break;
            }
            let mut dx = -value * gx / gradient_squared;
            let mut dy = -value * gy / gradient_squared;
            let step = (dx * dx + dy * dy).sqrt();
            if step > maximum_step {
                dx *= maximum_step / step;
                dy *= maximum_step / step;
            }
            x += dx;
            y += dy;
        }
        mesh.nodes[node].x = x;
        mesh.nodes[node].y = y;
    }

    // Safeguard against a projection crossing the opposite edge of a thin
    // cut cell. A global line search preserves conformity and orientation.
    let projected = mesh.nodes.clone();
    let mut blend = 1.0;
    while blend > 1.0 / 1024.0 {
        for &node in &boundary_nodes {
            mesh.nodes[node].x = original[node].x + blend * (projected[node].x - original[node].x);
            mesh.nodes[node].y = original[node].y + blend * (projected[node].y - original[node].y);
        }
        let valid = mesh.elements.iter().all(|element| {
            let [a, b, c] = [element.nodes[0], element.nodes[1], element.nodes[2]];
            let pa = &mesh.nodes[a];
            let pb = &mesh.nodes[b];
            let pc = &mesh.nodes[c];
            (pb.x - pa.x) * (pc.y - pa.y) - (pb.y - pa.y) * (pc.x - pa.x)
                > 128.0 * f64::EPSILON * scale * scale
        });
        if valid {
            break;
        }
        blend *= 0.5;
    }
}

/// Smooth a binary image mask through a bilinear half-value contour, fit the
/// mesh boundary to it, and locally refine boundary cells.
///
/// Use [`binary_mask_mesh_triangles`] when exact pixel staircasing is desired.
#[allow(clippy::too_many_arguments)]
pub fn fitted_binary_mask_mesh_triangles(
    mask: &[Vec<bool>],
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    boundary_marker: i32,
    boundary_refinement_levels: usize,
) -> Mesh {
    assert!(!mask.is_empty(), "binary mask must have at least one row");
    let ny = mask.len();
    let nx = mask[0].len();
    assert!(nx > 0, "binary mask must have at least one column");
    assert!(
        mask.iter().all(|row| row.len() == nx),
        "binary mask rows must have equal length"
    );
    let sample = |i: isize, j: isize| -> f64 {
        if i < 0 || j < 0 || i >= nx as isize || j >= ny as isize {
            1.0
        } else if mask[j as usize][i as usize] {
            -1.0
        } else {
            1.0
        }
    };
    let level_set = |x: f64, y: f64| {
        let px = (x - x_min) / (x_max - x_min) * nx as f64 - 0.5;
        let py = (y - y_min) / (y_max - y_min) * ny as f64 - 0.5;
        let i = px.floor() as isize;
        let j = py.floor() as isize;
        let tx = px - i as f64;
        let ty = py - j as f64;
        (1.0 - tx) * (1.0 - ty) * sample(i, j)
            + tx * (1.0 - ty) * sample(i + 1, j)
            + (1.0 - tx) * ty * sample(i, j + 1)
            + tx * ty * sample(i + 1, j + 1)
    };
    // Checked doubling: `2 * nx` wraps to a tiny mesh in release for
    // hostile mask dimensions instead of failing. Unreachable through the
    // try_ twin (mask dims are budgeted there); panics fast on direct
    // hostile calls.
    let nx2 = nx.checked_mul(2).expect(
        "fitted binary mask doubling overflowed; use try_fitted_binary_mask_mesh_triangles",
    );
    let ny2 = ny.checked_mul(2).expect(
        "fitted binary mask doubling overflowed; use try_fitted_binary_mask_mesh_triangles",
    );
    try_fitted_implicit_region_mesh_triangles(
        x_min,
        x_max,
        y_min,
        y_max,
        boundary_marker,
        FittedImplicitRegionConfig {
            nx: nx2,
            ny: ny2,
            boundary_refinement_levels,
            projection_iterations: 12,
        },
        level_set,
    )
    .unwrap_or_else(|error| panic!("fitted binary-mask grid invalid: {error}"))
}

/// Fallible [`fitted_binary_mask_mesh_triangles`] for externally supplied
/// masks: empty/ragged masks and over-budget refined grids — including the
/// 2x sampling doubling and the refinement-level multiplier — return
/// [`StructuredGridError`] instead of panicking or hanging.
pub fn try_fitted_binary_mask_mesh_triangles(
    mask: &[Vec<bool>],
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    boundary_marker: i32,
    boundary_refinement_levels: usize,
) -> Result<Mesh, StructuredGridError> {
    if mask.is_empty() {
        return Err(StructuredGridError::InvalidMask {
            detail: "at least one row required",
        });
    }
    let ny = mask.len();
    let nx = mask[0].len();
    if nx == 0 {
        return Err(StructuredGridError::InvalidMask {
            detail: "at least one column required",
        });
    }
    if !mask.iter().all(|row| row.len() == nx) {
        return Err(StructuredGridError::InvalidMask {
            detail: "rows must have equal length",
        });
    }
    let base = StructuredGridError::checked_cells(&[nx, ny]);
    let doubled = base.saturating_mul(4);
    let mult = StructuredGridError::refinement_multiplier(boundary_refinement_levels);
    let cells = doubled.saturating_mul(mult);
    StructuredGridError::check_cell_budget(
        "fitted binary-mask grid",
        cells,
        StructuredGridError::BYTES_PER_CELL_2D,
    )?;
    Ok(fitted_binary_mask_mesh_triangles(
        mask,
        x_min,
        x_max,
        y_min,
        y_max,
        boundary_marker,
        boundary_refinement_levels,
    ))
}

/// Mesh the union of two disks for moving-domain eigenvalue studies.
pub fn two_disk_union_mesh_triangles(
    center_1: (f64, f64),
    center_2: (f64, f64),
    radius_1: f64,
    radius_2: f64,
    nx: usize,
    ny: usize,
    boundary_marker: i32,
) -> Mesh {
    assert!(
        radius_1 > 0.0 && radius_2 > 0.0,
        "disk radii must be positive"
    );
    let x_min = (center_1.0 - radius_1).min(center_2.0 - radius_2);
    let x_max = (center_1.0 + radius_1).max(center_2.0 + radius_2);
    let y_min = (center_1.1 - radius_1).min(center_2.1 - radius_2);
    let y_max = (center_1.1 + radius_1).max(center_2.1 + radius_2);
    try_implicit_region_mesh_triangles(
        x_min,
        x_max,
        y_min,
        y_max,
        nx,
        ny,
        boundary_marker,
        move |x, y| {
            (x - center_1.0).powi(2) + (y - center_1.1).powi(2) <= radius_1.powi(2)
                || (x - center_2.0).powi(2) + (y - center_2.1).powi(2) <= radius_2.powi(2)
        },
    )
    .unwrap_or_else(|error| panic!("two-disk background grid invalid: {error}"))
}

/// Fallible [`two_disk_union_mesh_triangles`] for externally supplied
/// geometry/counts: non-finite centers, non-positive radii and over-budget
/// grids return [`StructuredGridError`] instead of panicking or hanging.
/// Extent admission for the derived bounding box happens in
/// [`try_implicit_region_mesh_triangles`].
pub fn try_two_disk_union_mesh_triangles(
    center_1: (f64, f64),
    center_2: (f64, f64),
    radius_1: f64,
    radius_2: f64,
    nx: usize,
    ny: usize,
    boundary_marker: i32,
) -> Result<Mesh, StructuredGridError> {
    for (name, c) in [("center_1", center_1), ("center_2", center_2)] {
        if !c.0.is_finite() || !c.1.is_finite() {
            return Err(StructuredGridError::NonFiniteExtent {
                axis: name,
                min: c.0,
                max: c.1,
            });
        }
    }
    for (name, r) in [("radius_1", radius_1), ("radius_2", radius_2)] {
        if !r.is_finite() {
            return Err(StructuredGridError::NonFiniteExtent {
                axis: name,
                min: r,
                max: r,
            });
        }
        if r <= 0.0 {
            return Err(StructuredGridError::DegenerateExtent {
                axis: name,
                min: r,
                max: r,
            });
        }
    }
    if nx == 0 {
        return Err(StructuredGridError::ZeroResolution {
            axis: "x",
            actual: nx,
        });
    }
    if ny == 0 {
        return Err(StructuredGridError::ZeroResolution {
            axis: "y",
            actual: ny,
        });
    }
    StructuredGridError::check_cell_budget(
        "two-disk grid",
        StructuredGridError::checked_cells(&[nx, ny]),
        StructuredGridError::BYTES_PER_CELL_2D,
    )?;
    Ok(two_disk_union_mesh_triangles(
        center_1,
        center_2,
        radius_1,
        radius_2,
        nx,
        ny,
        boundary_marker,
    ))
}

/// Mesh a binary image mask as a pixel-exact planar region.
///
/// Rows are ordered from `y_min` to `y_max`; `true` pixels belong to the
/// domain. The result uses two P1 triangles per active pixel, compacts unused
/// nodes, and marks all interfaces with `boundary_marker`. Internal false
/// islands therefore become correctly tagged holes.
pub fn binary_mask_mesh_triangles(
    mask: &[Vec<bool>],
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    boundary_marker: i32,
) -> Mesh {
    assert!(!mask.is_empty(), "binary mask must have at least one row");
    let ny = mask.len();
    let nx = mask[0].len();
    assert!(nx > 0, "binary mask must have at least one column");
    assert!(
        mask.iter().all(|row| row.len() == nx),
        "binary mask rows must have equal length"
    );
    try_implicit_region_mesh_triangles(
        x_min,
        x_max,
        y_min,
        y_max,
        nx,
        ny,
        boundary_marker,
        |x, y| {
            let i = (((x - x_min) / (x_max - x_min)) * nx as f64)
                .floor()
                .clamp(0.0, (nx - 1) as f64) as usize;
            let j = (((y - y_min) / (y_max - y_min)) * ny as f64)
                .floor()
                .clamp(0.0, (ny - 1) as f64) as usize;
            mask[j][i]
        },
    )
    .unwrap_or_else(|error| panic!("binary-mask background grid invalid: {error}"))
}

/// Fallible [`binary_mask_mesh_triangles`] for externally supplied masks:
/// empty/ragged masks, non-finite extents and over-budget grids return
/// [`StructuredGridError`] instead of panicking or hanging. Counts are
/// derived from the mask itself, so the budget bounds rows × cols.
pub fn try_binary_mask_mesh_triangles(
    mask: &[Vec<bool>],
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    boundary_marker: i32,
) -> Result<Mesh, StructuredGridError> {
    if mask.is_empty() {
        return Err(StructuredGridError::InvalidMask {
            detail: "at least one row required",
        });
    }
    let ny = mask.len();
    let nx = mask[0].len();
    if nx == 0 {
        return Err(StructuredGridError::InvalidMask {
            detail: "at least one column required",
        });
    }
    if !mask.iter().all(|row| row.len() == nx) {
        return Err(StructuredGridError::InvalidMask {
            detail: "rows must have equal length",
        });
    }
    if !x_min.is_finite() || !x_max.is_finite() {
        return Err(StructuredGridError::NonFiniteExtent {
            axis: "x",
            min: x_min,
            max: x_max,
        });
    }
    if !y_min.is_finite() || !y_max.is_finite() {
        return Err(StructuredGridError::NonFiniteExtent {
            axis: "y",
            min: y_min,
            max: y_max,
        });
    }
    StructuredGridError::check_cell_budget(
        "binary mask grid",
        StructuredGridError::checked_cells(&[nx, ny]),
        StructuredGridError::BYTES_PER_CELL_2D,
    )?;
    Ok(binary_mask_mesh_triangles(
        mask,
        x_min,
        x_max,
        y_min,
        y_max,
        boundary_marker,
    ))
}

/// Generate a rectangular mesh with quadrilateral elements
pub fn rectangular_mesh_quads(
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    nx: usize,
    ny: usize,
) -> Mesh {
    try_rectangular_mesh_quads(x_min, x_max, y_min, y_max, nx, ny)
        .unwrap_or_else(|error| panic!("{error}"))
}

/// Fallible [`rectangular_mesh_quads`] for externally supplied
/// counts/extents: zero counts and non-finite extents return
/// [`StructuredGridError`] instead of emitting infinite/NaN nodes.
pub fn try_rectangular_mesh_quads(
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    nx: usize,
    ny: usize,
) -> Result<Mesh, StructuredGridError> {
    StructuredGridError::check_2d(x_min, x_max, y_min, y_max, nx, ny)?;
    let mut mesh = Mesh::new(2);

    let dx = (x_max - x_min) / nx as f64;
    let dy = (y_max - y_min) / ny as f64;

    // Create nodes
    for j in 0..=ny {
        for i in 0..=nx {
            let x = x_min + i as f64 * dx;
            let y = y_min + j as f64 * dy;
            mesh.add_node(Point::new_2d(x, y));
        }
    }

    // Create quadrilateral elements
    for j in 0..ny {
        for i in 0..nx {
            let n00 = j * (nx + 1) + i;
            let n10 = n00 + 1;
            let n01 = n00 + (nx + 1);
            let n11 = n01 + 1;

            mesh.add_element(ElementType::Quadrilateral, vec![n00, n10, n11, n01]);
        }
    }

    mesh.detect_boundaries();
    Ok(mesh)
}

/// Generate a box mesh with tetrahedral elements using the 24-tet
/// **barycentric subdivision** of each cube. Each cube cell is split via
/// its centroid plus the centroid of each of its 6 faces, yielding
/// `4 × 6 = 24` tets per cube. Unlike the standard `box_mesh_tetrahedra`
/// (Kuhn 6-tet decomposition with a single fixed diagonal), this
/// decomposition is invariant under the full octahedral symmetry group
/// of the cube (48 elements, including all 90° rotations about each
/// axis). Use this when the physical problem has rotational or mirror
/// symmetry that the Kuhn diagonal would break — e.g. an axisymmetric
/// fan where a Kuhn-meshed pressure field would visibly tilt along the
/// `(0,0,0)→(1,1,1)` diagonal of every cell.
///
/// Mesh size grows by 4× elements vs Kuhn (24 vs 6 per cube), and
/// `(nx+1)(ny+1)(nz+1)` corner nodes plus face-centre and cube-centre
/// nodes; the wall-clock cost scales accordingly. Prefer Kuhn when
/// rotational symmetry is not relevant.
#[allow(clippy::too_many_arguments)]
pub fn box_mesh_tetrahedra_symmetric(
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    z_min: f64,
    z_max: f64,
    nx: usize,
    ny: usize,
    nz: usize,
) -> Mesh {
    try_box_mesh_tetrahedra_symmetric(x_min, x_max, y_min, y_max, z_min, z_max, nx, ny, nz)
        .unwrap_or_else(|error| panic!("{error}"))
}

/// Fallible symmetric [`box_mesh_tetrahedra_symmetric`] for externally
/// supplied counts/extents: zero counts and non-finite extents return
/// [`StructuredGridError`] instead of emitting infinite/NaN nodes.
#[allow(clippy::too_many_arguments)]
pub fn try_box_mesh_tetrahedra_symmetric(
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    z_min: f64,
    z_max: f64,
    nx: usize,
    ny: usize,
    nz: usize,
) -> Result<Mesh, StructuredGridError> {
    StructuredGridError::check_3d(x_min, x_max, y_min, y_max, z_min, z_max, nx, ny, nz)?;
    let mut mesh = Mesh::new(3);

    let dx = (x_max - x_min) / nx as f64;
    let dy = (y_max - y_min) / ny as f64;
    let dz = (z_max - z_min) / nz as f64;

    // 1. Corner nodes (shared between 8 cubes).
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                let x = x_min + i as f64 * dx;
                let y = y_min + j as f64 * dy;
                let z = z_min + k as f64 * dz;
                mesh.add_node(Point::new_3d(x, y, z));
            }
        }
    }
    let corner =
        |i: usize, j: usize, k: usize| -> usize { k * (ny + 1) * (nx + 1) + j * (nx + 1) + i };
    let n_corners = (nx + 1) * (ny + 1) * (nz + 1);

    // 2. Shared face-centre nodes. Adjacent cubes must use the same DOF
    //    on their common face to preserve continuity and interior topology.
    let face_centres_start = n_corners;
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..=nx {
                mesh.add_node(Point::new_3d(
                    x_min + i as f64 * dx,
                    y_min + (j as f64 + 0.5) * dy,
                    z_min + (k as f64 + 0.5) * dz,
                ));
            }
        }
    }
    let y_faces_start = mesh.num_nodes();
    for k in 0..nz {
        for j in 0..=ny {
            for i in 0..nx {
                mesh.add_node(Point::new_3d(
                    x_min + (i as f64 + 0.5) * dx,
                    y_min + j as f64 * dy,
                    z_min + (k as f64 + 0.5) * dz,
                ));
            }
        }
    }
    let z_faces_start = mesh.num_nodes();
    for k in 0..=nz {
        for j in 0..ny {
            for i in 0..nx {
                mesh.add_node(Point::new_3d(
                    x_min + (i as f64 + 0.5) * dx,
                    y_min + (j as f64 + 0.5) * dy,
                    z_min + k as f64 * dz,
                ));
            }
        }
    }
    let face_centre = |i: usize, j: usize, k: usize, face: usize| -> usize {
        match face {
            0 | 1 => face_centres_start + (k * ny + j) * (nx + 1) + i + face,
            2 | 3 => y_faces_start + (k * (ny + 1) + j + face - 2) * nx + i,
            4 | 5 => z_faces_start + ((k + face - 4) * ny + j) * nx + i,
            _ => unreachable!("cube has six faces"),
        }
    };

    // 3. Cube-centroid nodes — one per cube.
    let centroids_start = mesh.num_nodes();
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let xc = x_min + (i as f64 + 0.5) * dx;
                let yc = y_min + (j as f64 + 0.5) * dy;
                let zc = z_min + (k as f64 + 0.5) * dz;
                mesh.add_node(Point::new_3d(xc, yc, zc));
            }
        }
    }
    let centroid =
        |i: usize, j: usize, k: usize| -> usize { centroids_start + k * ny * nx + j * nx + i };

    // 4. Tetrahedral elements — for each cube, each face is split into 4
    //    triangles via its centre, and each triangle + the cube centroid
    //    forms one tet.
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let n000 = corner(i, j, k);
                let n100 = corner(i + 1, j, k);
                let n010 = corner(i, j + 1, k);
                let n110 = corner(i + 1, j + 1, k);
                let n001 = corner(i, j, k + 1);
                let n101 = corner(i + 1, j, k + 1);
                let n011 = corner(i, j + 1, k + 1);
                let n111 = corner(i + 1, j + 1, k + 1);
                let cc = centroid(i, j, k);

                // For each face: 4 corner pairs traversed CCW (as seen
                // from the outside), forming 4 triangles with the face
                // centre. Each triangle + the cube centroid = one tet.
                let faces: [(usize, [(usize, usize); 4]); 6] = [
                    (
                        face_centre(i, j, k, 0),
                        [(n000, n010), (n010, n011), (n011, n001), (n001, n000)],
                    ), // −x
                    (
                        face_centre(i, j, k, 1),
                        [(n100, n101), (n101, n111), (n111, n110), (n110, n100)],
                    ), // +x
                    (
                        face_centre(i, j, k, 2),
                        [(n000, n001), (n001, n101), (n101, n100), (n100, n000)],
                    ), // −y
                    (
                        face_centre(i, j, k, 3),
                        [(n010, n110), (n110, n111), (n111, n011), (n011, n010)],
                    ), // +y
                    (
                        face_centre(i, j, k, 4),
                        [(n000, n100), (n100, n110), (n110, n010), (n010, n000)],
                    ), // −z
                    (
                        face_centre(i, j, k, 5),
                        [(n001, n011), (n011, n111), (n111, n101), (n101, n001)],
                    ), // +z
                ];

                for (fc, edges) in faces {
                    for (a, b) in edges {
                        mesh.add_element(ElementType::Tetrahedron, vec![cc, fc, a, b]);
                    }
                }
            }
        }
    }

    mesh.detect_boundaries();
    Ok(mesh)
}

/// Generate a box mesh with tetrahedral elements (Kuhn 6-tet decomposition)
#[allow(clippy::too_many_arguments)]
pub fn box_mesh_tetrahedra(
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    z_min: f64,
    z_max: f64,
    nx: usize,
    ny: usize,
    nz: usize,
) -> Mesh {
    try_box_mesh_tetrahedra(x_min, x_max, y_min, y_max, z_min, z_max, nx, ny, nz)
        .unwrap_or_else(|error| panic!("{error}"))
}

/// Fallible Kuhn [`box_mesh_tetrahedra`] for externally supplied
/// counts/extents: zero counts and non-finite extents return
/// [`StructuredGridError`] instead of emitting infinite/NaN nodes.
#[allow(clippy::too_many_arguments)]
pub fn try_box_mesh_tetrahedra(
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    z_min: f64,
    z_max: f64,
    nx: usize,
    ny: usize,
    nz: usize,
) -> Result<Mesh, StructuredGridError> {
    StructuredGridError::check_3d(x_min, x_max, y_min, y_max, z_min, z_max, nx, ny, nz)?;
    let mut mesh = Mesh::new(3);

    let dx = (x_max - x_min) / nx as f64;
    let dy = (y_max - y_min) / ny as f64;
    let dz = (z_max - z_min) / nz as f64;

    // Create nodes
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                let x = x_min + i as f64 * dx;
                let y = y_min + j as f64 * dy;
                let z = z_min + k as f64 * dz;
                mesh.add_node(Point::new_3d(x, y, z));
            }
        }
    }

    // Node indexing function
    let node_idx =
        |i: usize, j: usize, k: usize| -> usize { k * (ny + 1) * (nx + 1) + j * (nx + 1) + i };

    // Create tetrahedral elements (6 tetrahedra per cube)
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                // 8 corners of the cube
                let n000 = node_idx(i, j, k);
                let n100 = node_idx(i + 1, j, k);
                let n010 = node_idx(i, j + 1, k);
                let n110 = node_idx(i + 1, j + 1, k);
                let n001 = node_idx(i, j, k + 1);
                let n101 = node_idx(i + 1, j, k + 1);
                let n011 = node_idx(i, j + 1, k + 1);
                let n111 = node_idx(i + 1, j + 1, k + 1);

                // Divide cube into 6 tetrahedra (Kuhn triangulation)
                mesh.add_element(ElementType::Tetrahedron, vec![n000, n100, n110, n111]);
                mesh.add_element(ElementType::Tetrahedron, vec![n000, n110, n010, n111]);
                mesh.add_element(ElementType::Tetrahedron, vec![n000, n010, n011, n111]);
                mesh.add_element(ElementType::Tetrahedron, vec![n000, n011, n001, n111]);
                mesh.add_element(ElementType::Tetrahedron, vec![n000, n001, n101, n111]);
                mesh.add_element(ElementType::Tetrahedron, vec![n000, n101, n100, n111]);
            }
        }
    }

    mesh.detect_boundaries();
    Ok(mesh)
}

/// Generate a box mesh with hexahedral elements
#[allow(clippy::too_many_arguments)]
pub fn box_mesh_hexahedra(
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    z_min: f64,
    z_max: f64,
    nx: usize,
    ny: usize,
    nz: usize,
) -> Mesh {
    try_box_mesh_hexahedra(x_min, x_max, y_min, y_max, z_min, z_max, nx, ny, nz)
        .unwrap_or_else(|error| panic!("{error}"))
}

/// Fallible [`box_mesh_hexahedra`] for externally supplied counts/extents:
/// zero counts and non-finite extents return [`StructuredGridError`]
/// instead of emitting infinite/NaN nodes.
#[allow(clippy::too_many_arguments)]
pub fn try_box_mesh_hexahedra(
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    z_min: f64,
    z_max: f64,
    nx: usize,
    ny: usize,
    nz: usize,
) -> Result<Mesh, StructuredGridError> {
    StructuredGridError::check_3d(x_min, x_max, y_min, y_max, z_min, z_max, nx, ny, nz)?;
    let mut mesh = Mesh::new(3);

    let dx = (x_max - x_min) / nx as f64;
    let dy = (y_max - y_min) / ny as f64;
    let dz = (z_max - z_min) / nz as f64;

    // Create nodes
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                let x = x_min + i as f64 * dx;
                let y = y_min + j as f64 * dy;
                let z = z_min + k as f64 * dz;
                mesh.add_node(Point::new_3d(x, y, z));
            }
        }
    }

    // Node indexing function
    let node_idx =
        |i: usize, j: usize, k: usize| -> usize { k * (ny + 1) * (nx + 1) + j * (nx + 1) + i };

    // Create hexahedral elements
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let n000 = node_idx(i, j, k);
                let n100 = node_idx(i + 1, j, k);
                let n010 = node_idx(i, j + 1, k);
                let n110 = node_idx(i + 1, j + 1, k);
                let n001 = node_idx(i, j, k + 1);
                let n101 = node_idx(i + 1, j, k + 1);
                let n011 = node_idx(i, j + 1, k + 1);
                let n111 = node_idx(i + 1, j + 1, k + 1);

                mesh.add_element(
                    ElementType::Hexahedron,
                    vec![n000, n100, n110, n010, n001, n101, n111, n011],
                );
            }
        }
    }

    mesh.detect_boundaries();
    Ok(mesh)
}

/// Convex hull surface mesh from a 3D point cloud.
///
/// Runs the in-tree quickhull on `points`, then builds a 3D triangle surface
/// `Mesh` with one `Element::Triangle` per hull face. The mesh dimension is set
/// to 3 because the triangles live in 3D space, even though each element is
/// topologically 2D.
///
/// Only the vertices that appear in at least one hull face are materialised as
/// mesh nodes — interior points dropped by the hull algorithm do not show up
/// here. Returns the underlying quickhull error verbatim when fewer than four
/// non-degenerate points are supplied.
///
/// The triangle winding matches quickhull's output (outward normals). No
/// boundary faces are emitted: a closed hull surface has none, and partial
/// hulls (where quickhull fails) are surfaced through the error path instead.
pub fn convex_hull_surface(points: &[Point]) -> Result<Mesh, crate::convexhull::ConvexHullError> {
    let vertices: Vec<crate::convexhull::Vertex> = points
        .iter()
        .map(|p| crate::convexhull::Vertex::new(p.x, p.y, p.z))
        .collect();
    let hull = crate::convexhull::ConvexHull3D::build(&vertices)?;

    // Materialise only vertices referenced by some face. This drops interior
    // points and also any vertex quickhull dropped (e.g. duplicates).
    let mut node_map: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut mesh = Mesh::new(3);
    for face in hull.faces() {
        for &v in &face.indices() {
            node_map.entry(v).or_insert_with(|| {
                mesh.add_node(Point::new_3d(
                    hull.vertices()[v].x,
                    hull.vertices()[v].y,
                    hull.vertices()[v].z,
                ))
            });
        }
    }
    for face in hull.faces() {
        let [a, b, c] = face.indices();
        mesh.add_element(
            ElementType::Triangle,
            vec![node_map[&a], node_map[&b], node_map[&c]],
        );
    }
    Ok(mesh)
}

/// Generate a circular mesh with triangular elements
pub fn circular_mesh_triangles(
    center_x: f64,
    center_y: f64,
    radius: f64,
    n_radial: usize,
    n_angular: usize,
) -> Mesh {
    // Fail-fast budget gate for the infallible entry: untrusted counts must
    // use try_circular_mesh_triangles. Hostile counts panic here instead of
    // hanging in the layer loops below.
    StructuredGridError::check_cell_budget(
        "circular mesh",
        StructuredGridError::checked_cells(&[n_radial, n_angular]),
        StructuredGridError::BYTES_PER_CELL_2D,
    )
    .expect("circular mesh over cell budget; use try_circular_mesh_triangles");
    let mut mesh = Mesh::new(2);

    // Center node
    mesh.add_node(Point::new_2d(center_x, center_y));

    // Radial layers of nodes
    for r in 1..=n_radial {
        let rad = radius * (r as f64) / (n_radial as f64);
        for a in 0..n_angular {
            let theta = 2.0 * std::f64::consts::PI * (a as f64) / (n_angular as f64);
            let x = center_x + rad * theta.cos();
            let y = center_y + rad * theta.sin();
            mesh.add_node(Point::new_2d(x, y));
        }
    }

    // Inner ring (triangles from center)
    for a in 0..n_angular {
        let n1 = 1 + a;
        let n2 = 1 + (a + 1) % n_angular;
        mesh.add_element(ElementType::Triangle, vec![0, n1, n2]);
    }

    // Outer rings (quadrilaterals split into triangles)
    for r in 1..n_radial {
        let offset_inner = 1 + (r - 1) * n_angular;
        let offset_outer = 1 + r * n_angular;
        for a in 0..n_angular {
            let n00 = offset_inner + a;
            let n10 = offset_inner + (a + 1) % n_angular;
            let n01 = offset_outer + a;
            let n11 = offset_outer + (a + 1) % n_angular;

            mesh.add_element(ElementType::Triangle, vec![n00, n10, n11]);
            mesh.add_element(ElementType::Triangle, vec![n00, n11, n01]);
        }
    }

    mesh.detect_boundaries();

    // Set outer boundary
    let tol = radius * 0.001;
    mesh.set_boundary_condition(BoundaryType::Dirichlet, 1, |points| {
        let avg_r = points
            .iter()
            .map(|p| ((p.x - center_x).powi(2) + (p.y - center_y).powi(2)).sqrt())
            .sum::<f64>()
            / points.len() as f64;
        (avg_r - radius).abs() < tol
    });

    mesh
}

/// Fallible [`circular_mesh_triangles`] for externally supplied
/// geometry/counts: zero counts, non-finite center/radius and over-budget
/// grids return [`StructuredGridError`] instead of dividing by zero,
/// indexing `% 0`, hanging, or emitting NaN nodes.
pub fn try_circular_mesh_triangles(
    center_x: f64,
    center_y: f64,
    radius: f64,
    n_radial: usize,
    n_angular: usize,
) -> Result<Mesh, StructuredGridError> {
    if n_radial == 0 {
        return Err(StructuredGridError::ZeroResolution {
            axis: "radial",
            actual: n_radial,
        });
    }
    if n_angular == 0 {
        return Err(StructuredGridError::ZeroResolution {
            axis: "angular",
            actual: n_angular,
        });
    }
    if !center_x.is_finite() || !center_y.is_finite() {
        return Err(StructuredGridError::NonFiniteExtent {
            axis: "center",
            min: center_x,
            max: center_y,
        });
    }
    if !radius.is_finite() {
        return Err(StructuredGridError::NonFiniteExtent {
            axis: "radius",
            min: radius,
            max: radius,
        });
    }
    StructuredGridError::check_cell_budget(
        "circular mesh",
        StructuredGridError::checked_cells(&[n_radial, n_angular]),
        StructuredGridError::BYTES_PER_CELL_2D,
    )?;
    Ok(circular_mesh_triangles(
        center_x, center_y, radius, n_radial, n_angular,
    ))
}

/// Generate an annular (ring) mesh with triangular elements
pub fn annular_mesh_triangles(
    center_x: f64,
    center_y: f64,
    inner_radius: f64,
    outer_radius: f64,
    n_radial: usize,
    n_angular: usize,
) -> Mesh {
    // Fail-fast budget gate for the infallible entry: untrusted counts must
    // use try_annular_mesh_triangles.
    StructuredGridError::check_cell_budget(
        "annular mesh",
        StructuredGridError::checked_cells(&[n_radial, n_angular]),
        StructuredGridError::BYTES_PER_CELL_2D,
    )
    .expect("annular mesh over cell budget; use try_annular_mesh_triangles");
    let mut mesh = Mesh::new(2);

    // Create nodes in radial layers from inner to outer radius
    let dr = (outer_radius - inner_radius) / (n_radial as f64);

    for r in 0..=n_radial {
        let rad = inner_radius + r as f64 * dr;
        for a in 0..n_angular {
            let theta = 2.0 * std::f64::consts::PI * (a as f64) / (n_angular as f64);
            let x = center_x + rad * theta.cos();
            let y = center_y + rad * theta.sin();
            mesh.add_node(Point::new_2d(x, y));
        }
    }

    // Create triangular elements (2 per quad cell)
    for r in 0..n_radial {
        let offset_inner = r * n_angular;
        let offset_outer = (r + 1) * n_angular;
        for a in 0..n_angular {
            let n00 = offset_inner + a;
            let n10 = offset_inner + (a + 1) % n_angular;
            let n01 = offset_outer + a;
            let n11 = offset_outer + (a + 1) % n_angular;

            mesh.add_element(ElementType::Triangle, vec![n00, n10, n11]);
            mesh.add_element(ElementType::Triangle, vec![n00, n11, n01]);
        }
    }

    mesh.detect_boundaries();

    // Set inner boundary (obstacle) - tag 1
    let tol = inner_radius * 0.01;
    mesh.set_boundary_condition(BoundaryType::Dirichlet, 1, |points| {
        let avg_r = points
            .iter()
            .map(|p| ((p.x - center_x).powi(2) + (p.y - center_y).powi(2)).sqrt())
            .sum::<f64>()
            / points.len() as f64;
        (avg_r - inner_radius).abs() < tol
    });

    // Set outer boundary (far-field) - tag 2
    let tol_outer = outer_radius * 0.01;
    mesh.set_boundary_condition(BoundaryType::Dirichlet, 2, |points| {
        let avg_r = points
            .iter()
            .map(|p| ((p.x - center_x).powi(2) + (p.y - center_y).powi(2)).sqrt())
            .sum::<f64>()
            / points.len() as f64;
        (avg_r - outer_radius).abs() < tol_outer
    });

    mesh
}

/// Fallible [`annular_mesh_triangles`] for externally supplied
/// geometry/counts: zero counts, non-finite center/radii and over-budget
/// grids return [`StructuredGridError`] instead of dividing by zero,
/// indexing `% 0`, hanging, or emitting NaN nodes.
pub fn try_annular_mesh_triangles(
    center_x: f64,
    center_y: f64,
    inner_radius: f64,
    outer_radius: f64,
    n_radial: usize,
    n_angular: usize,
) -> Result<Mesh, StructuredGridError> {
    if n_radial == 0 {
        return Err(StructuredGridError::ZeroResolution {
            axis: "radial",
            actual: n_radial,
        });
    }
    if n_angular == 0 {
        return Err(StructuredGridError::ZeroResolution {
            axis: "angular",
            actual: n_angular,
        });
    }
    if !center_x.is_finite() || !center_y.is_finite() {
        return Err(StructuredGridError::NonFiniteExtent {
            axis: "center",
            min: center_x,
            max: center_y,
        });
    }
    for (name, r) in [
        ("inner_radius", inner_radius),
        ("outer_radius", outer_radius),
    ] {
        if !r.is_finite() {
            return Err(StructuredGridError::NonFiniteExtent {
                axis: name,
                min: r,
                max: r,
            });
        }
    }
    StructuredGridError::check_cell_budget(
        "annular mesh",
        StructuredGridError::checked_cells(&[n_radial, n_angular]),
        StructuredGridError::BYTES_PER_CELL_2D,
    )?;
    Ok(annular_mesh_triangles(
        center_x,
        center_y,
        inner_radius,
        outer_radius,
        n_radial,
        n_angular,
    ))
}

/// Generate a spherical shell mesh using extruded icosphere (Tetrahedra)
///
/// Creates a 3D volume mesh between two concentric spheres.
/// Based on subdivisions of an icosahedron, extruded radially.
pub fn spherical_shell_mesh_tetrahedra(
    center_x: f64,
    center_y: f64,
    center_z: f64,
    inner_radius: f64,
    outer_radius: f64,
    subdivisions: usize,
    n_layers: usize,
) -> Mesh {
    // Fail-fast budget gate for the infallible entry: icosphere subdivision
    // quadruples base triangles per level. Untrusted counts must use
    // try_spherical_shell_mesh_tetrahedra.
    {
        let base = StructuredGridError::refinement_multiplier(subdivisions);
        let cells = base.saturating_mul(n_layers as u64);
        StructuredGridError::check_cell_budget(
            "spherical shell",
            cells,
            StructuredGridError::BYTES_PER_SHELL_UNIT,
        )
        .expect("spherical shell over cell budget; use try_spherical_shell_mesh_tetrahedra");
    }
    let mut mesh = Mesh::new(3);

    // 1. Generate unit icosphere surface (nodes and triangles)
    let (base_nodes, base_tris) = generate_icosphere(subdivisions);
    let num_surface_nodes = base_nodes.len();

    // 2. Create nodes for all layers
    let dr = (outer_radius - inner_radius) / n_layers as f64;
    for l in 0..=n_layers {
        let r = inner_radius + l as f64 * dr;
        for p in &base_nodes {
            mesh.add_node(Point::new_3d(
                center_x + p.x * r,
                center_y + p.y * r,
                center_z + p.z * r,
            ));
        }
    }

    // 3. Create elements connecting layers
    for l in 0..n_layers {
        let offset_lower = l * num_surface_nodes;
        let offset_upper = (l + 1) * num_surface_nodes;

        for tri in &base_tris {
            let u0 = offset_lower + tri[0];
            let u1 = offset_lower + tri[1];
            let u2 = offset_lower + tri[2];

            let v0 = offset_upper + tri[0];
            let v1 = offset_upper + tri[1];
            let v2 = offset_upper + tri[2];

            // Split prism (u0,u1,u2)-(v0,v1,v2) into 3 tetrahedra
            // We use a consistent split based on sorting the base indices
            split_prism_consistent(&mut mesh, [u0, u1, u2], [v0, v1, v2]);
        }
    }

    mesh.detect_boundaries();

    // Set BCs
    let tol = inner_radius * 0.01;
    mesh.set_boundary_condition(BoundaryType::Dirichlet, 1, |pts| {
        // Inner
        pts[0].distance(&Point::new_3d(center_x, center_y, center_z)) < inner_radius + tol
    });
    mesh.set_boundary_condition(BoundaryType::Dirichlet, 2, |pts| {
        // Outer
        pts[0].distance(&Point::new_3d(center_x, center_y, center_z)) > outer_radius - tol
    });

    mesh
}

/// Fallible [`spherical_shell_mesh_tetrahedra`] for externally supplied
/// geometry/counts: zero layers, non-finite center/radii and over-budget
/// subdivision/layer products return [`StructuredGridError`] instead of
/// dividing by zero, hanging in subdivision, or emitting NaN nodes.
/// (`subdivisions == 0` is admitted: it yields the base icosahedron.)
pub fn try_spherical_shell_mesh_tetrahedra(
    center_x: f64,
    center_y: f64,
    center_z: f64,
    inner_radius: f64,
    outer_radius: f64,
    subdivisions: usize,
    n_layers: usize,
) -> Result<Mesh, StructuredGridError> {
    if n_layers == 0 {
        return Err(StructuredGridError::ZeroResolution {
            axis: "layers",
            actual: n_layers,
        });
    }
    if !center_x.is_finite() || !center_y.is_finite() || !center_z.is_finite() {
        return Err(StructuredGridError::NonFiniteExtent {
            axis: "center",
            min: center_x,
            max: center_z,
        });
    }
    for (name, r) in [
        ("inner_radius", inner_radius),
        ("outer_radius", outer_radius),
    ] {
        if !r.is_finite() {
            return Err(StructuredGridError::NonFiniteExtent {
                axis: name,
                min: r,
                max: r,
            });
        }
    }
    {
        let base = StructuredGridError::refinement_multiplier(subdivisions);
        let cells = base.saturating_mul(n_layers as u64);
        StructuredGridError::check_cell_budget(
            "spherical shell",
            cells,
            StructuredGridError::BYTES_PER_SHELL_UNIT,
        )?;
    }
    Ok(spherical_shell_mesh_tetrahedra(
        center_x,
        center_y,
        center_z,
        inner_radius,
        outer_radius,
        subdivisions,
        n_layers,
    ))
}

/// Failure from [`try_layered_spherical_shell_mesh_tetrahedra`].
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum LayeredShellError {
    /// Fewer than the required inner + outer radii were supplied.
    #[error("at least inner and outer radii are required, got {actual}")]
    TooFewRadii { actual: usize },
    /// One radial subdivision count is required per material region.
    #[error(
        "one radial subdivision count is required per material region: {regions} region(s), {actual} count(s)"
    )]
    SubdivisionCountMismatch { regions: usize, actual: usize },
    /// Radii must be positive and strictly increasing.
    #[error("spherical-shell radii must be positive and strictly increasing")]
    NonIncreasingRadii,
    /// Every material region needs at least one radial subdivision.
    #[error("every material region needs at least one radial subdivision")]
    EmptyRadialSubdivision,
    /// The shell center must be finite (NaN would propagate into every node).
    #[error("layered-shell center must be finite")]
    NonFiniteCenter,
    /// The implied output (base icosphere times radial layers) exceeds the
    /// generator byte budget. See [`StructuredGridError::MAX_GENERATOR_BYTES`].
    #[error("layered shell needs {cells} cells, over budget {max_cells}")]
    BudgetExceeded { cells: u64, max_cells: u64 },
}

/// Generate a conforming, region-marked multilayer spherical shell mesh.
///
/// `radii` contains the inner radius, every material-interface radius, and
/// the outer radius. `radial_subdivisions[i]` is the number of conforming
/// radial prism layers in material region `i`. Tetrahedra in that region are
/// assigned marker `i + 1`; interface nodes are shared by both neighboring
/// regions. Boundary marker 1 identifies the inner sphere and marker 2 the
/// outer sphere.
///
/// Panics on malformed `radii` / `radial_subdivisions`; prefer
/// [`try_layered_spherical_shell_mesh_tetrahedra`] when the inputs come from
/// outside the call site.
pub fn layered_spherical_shell_mesh_tetrahedra(
    center_x: f64,
    center_y: f64,
    center_z: f64,
    radii: &[f64],
    subdivisions: usize,
    radial_subdivisions: &[usize],
) -> Mesh {
    try_layered_spherical_shell_mesh_tetrahedra(
        center_x,
        center_y,
        center_z,
        radii,
        subdivisions,
        radial_subdivisions,
    )
    .unwrap_or_else(|error| panic!("{error}"))
}

/// Fallible [`layered_spherical_shell_mesh_tetrahedra`] for externally
/// supplied radii: empty or non-increasing `radii` and mismatched/zero
/// `radial_subdivisions` return [`LayeredShellError`] instead of panicking.
pub fn try_layered_spherical_shell_mesh_tetrahedra(
    center_x: f64,
    center_y: f64,
    center_z: f64,
    radii: &[f64],
    subdivisions: usize,
    radial_subdivisions: &[usize],
) -> Result<Mesh, LayeredShellError> {
    if radii.len() < 2 {
        return Err(LayeredShellError::TooFewRadii {
            actual: radii.len(),
        });
    }
    if radial_subdivisions.len() != radii.len() - 1 {
        return Err(LayeredShellError::SubdivisionCountMismatch {
            regions: radii.len() - 1,
            actual: radial_subdivisions.len(),
        });
    }
    if !radii
        .windows(2)
        .all(|pair| pair[0] > 0.0 && pair[0] < pair[1])
    {
        return Err(LayeredShellError::NonIncreasingRadii);
    }
    if !radial_subdivisions.iter().all(|&count| count > 0) {
        return Err(LayeredShellError::EmptyRadialSubdivision);
    }
    if !center_x.is_finite() || !center_y.is_finite() || !center_z.is_finite() {
        return Err(LayeredShellError::NonFiniteCenter);
    }
    // Icosphere subdivision quadruples the base triangles per level, so the
    // prism count scales as 4^subdivisions times the radial layers. Bound it
    // before generating anything: hostile counts hang in subdivision or abort
    // in layer allocation.
    {
        let base = StructuredGridError::refinement_multiplier(subdivisions);
        let layers: u64 = radial_subdivisions
            .iter()
            .fold(0u64, |sum, &c| sum.saturating_add(c as u64));
        let cells = base.saturating_mul(layers);
        let max_cells =
            StructuredGridError::MAX_GENERATOR_BYTES / StructuredGridError::BYTES_PER_SHELL_UNIT;
        if cells > max_cells {
            return Err(LayeredShellError::BudgetExceeded { cells, max_cells });
        }
    }

    let (base_nodes, base_tris) = generate_icosphere(subdivisions);
    let num_surface_nodes = base_nodes.len();
    let mut layer_radii = vec![radii[0]];
    let mut layer_regions = Vec::new();
    for (region, (&inner, &count)) in radii.iter().zip(radial_subdivisions).enumerate() {
        let outer = radii[region + 1];
        for step in 1..=count {
            layer_radii.push(inner + (outer - inner) * step as f64 / count as f64);
            layer_regions.push(region);
        }
    }

    let mut mesh = Mesh::new(3);
    for &radius in &layer_radii {
        for point in &base_nodes {
            mesh.add_node(Point::new_3d(
                center_x + point.x * radius,
                center_y + point.y * radius,
                center_z + point.z * radius,
            ));
        }
    }

    for (radial_layer, &region) in layer_regions.iter().enumerate() {
        let lower_offset = radial_layer * num_surface_nodes;
        let upper_offset = (radial_layer + 1) * num_surface_nodes;
        for triangle in &base_tris {
            let first_new_element = mesh.elements.len();
            split_prism_consistent(
                &mut mesh,
                [
                    lower_offset + triangle[0],
                    lower_offset + triangle[1],
                    lower_offset + triangle[2],
                ],
                [
                    upper_offset + triangle[0],
                    upper_offset + triangle[1],
                    upper_offset + triangle[2],
                ],
            );
            for element in &mut mesh.elements[first_new_element..] {
                element.marker = (region + 1) as i32;
            }
        }
    }

    mesh.detect_boundaries();
    let center = Point::new_3d(center_x, center_y, center_z);
    let tolerance = 1.0e-10 * radii.last().copied().unwrap().max(1.0);
    mesh.set_boundary_condition(BoundaryType::Dirichlet, 1, |points| {
        points
            .iter()
            .all(|point| (point.distance(&center) - radii[0]).abs() <= tolerance)
    });
    let outer_radius = *radii.last().unwrap();
    mesh.set_boundary_condition(BoundaryType::Neumann, 2, |points| {
        points
            .iter()
            .all(|point| (point.distance(&center) - outer_radius).abs() <= tolerance)
    });
    Ok(mesh)
}

/// Helper: split a prism into 3 tetrahedra consistently
fn split_prism_consistent(mesh: &mut Mesh, lower: [usize; 3], upper: [usize; 3]) {
    // Sort base indices to define a local canonical order
    let mut perm = [0, 1, 2];
    perm.sort_by_key(|&i| lower[i]);

    let p0 = perm[0];
    let p1 = perm[1];
    let p2 = perm[2];

    // Canonical vertices
    let v0 = lower[p0];
    let v1 = lower[p1];
    let v2 = lower[p2];
    let v3 = upper[p0];
    let v4 = upper[p1];
    let v5 = upper[p2];

    // Standard split for 0<1<2
    // T1: 0-1-2-5 (base tri + top-max)
    mesh.add_element(ElementType::Tetrahedron, vec![v0, v1, v2, v5]);
    // T2: 0-1-5-4
    mesh.add_element(ElementType::Tetrahedron, vec![v0, v1, v5, v4]);
    // T3: 0-4-5-3
    mesh.add_element(ElementType::Tetrahedron, vec![v0, v4, v5, v3]);
}

/// Helper: generate unit icosphere
fn generate_icosphere(subdivisions: usize) -> (Vec<Point>, Vec<[usize; 3]>) {
    // Basic Icosahedron
    let t = (1.0 + 5.0f64.sqrt()) / 2.0;
    let mut nodes = vec![
        Point::new_3d(-1.0, t, 0.0),
        Point::new_3d(1.0, t, 0.0),
        Point::new_3d(-1.0, -t, 0.0),
        Point::new_3d(1.0, -t, 0.0),
        Point::new_3d(0.0, -1.0, t),
        Point::new_3d(0.0, 1.0, t),
        Point::new_3d(0.0, -1.0, -t),
        Point::new_3d(0.0, 1.0, -t),
        Point::new_3d(t, 0.0, -1.0),
        Point::new_3d(t, 0.0, 1.0),
        Point::new_3d(-t, 0.0, -1.0),
        Point::new_3d(-t, 0.0, 1.0),
    ];

    // Normalize
    for p in &mut nodes {
        let len = (p.x * p.x + p.y * p.y + p.z * p.z).sqrt();
        p.x /= len;
        p.y /= len;
        p.z /= len;
    }

    let mut tris = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];

    // Subdivide
    for _ in 0..subdivisions {
        let mut new_tris = Vec::new();
        let mut mid_cache = HashMap::new();

        for tri in tris {
            let v0 = tri[0];
            let v1 = tri[1];
            let v2 = tri[2];

            // Get midpoints (and add to nodes if new)
            // Note: can't use closure easily with mutable borrows, using helper function pattern
            // But we need to define helper inside or outside.
            // Since this is a standalone function, we can put logic in loop.

            let a = get_middle_point(v0, v1, &mut nodes, &mut mid_cache);
            let b = get_middle_point(v1, v2, &mut nodes, &mut mid_cache);
            let c = get_middle_point(v2, v0, &mut nodes, &mut mid_cache);

            new_tris.push([v0, a, c]);
            new_tris.push([v1, b, a]);
            new_tris.push([v2, c, b]);
            new_tris.push([a, b, c]);
        }
        tris = new_tris;
    }

    (nodes, tris)
}

fn get_middle_point(
    p1: usize,
    p2: usize,
    nodes: &mut Vec<Point>,
    cache: &mut HashMap<(usize, usize), usize>,
) -> usize {
    let key = if p1 < p2 { (p1, p2) } else { (p2, p1) };
    if let Some(&idx) = cache.get(&key) {
        return idx;
    }

    let pt1 = nodes[p1];
    let pt2 = nodes[p2];
    let mut middle = Point::new_3d(
        (pt1.x + pt2.x) / 2.0,
        (pt1.y + pt2.y) / 2.0,
        (pt1.z + pt2.z) / 2.0,
    );

    let len = (middle.x * middle.x + middle.y * middle.y + middle.z * middle.z).sqrt();
    middle.x /= len;
    middle.y /= len;
    middle.z /= len;

    let idx = nodes.len();
    nodes.push(middle);
    cache.insert(key, idx);
    idx
}

/// Generate a unit square mesh with triangles
pub fn unit_square_triangles(n: usize) -> Mesh {
    rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, n, n)
}

/// Fallible [`unit_square_triangles`]: zero/over-budget `n` returns
/// [`StructuredGridError`] instead of panicking or hanging.
pub fn try_unit_square_triangles(n: usize) -> Result<Mesh, StructuredGridError> {
    try_rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, n, n)
}

/// Generate a unit square mesh with quadrilaterals
pub fn unit_square_quads(n: usize) -> Mesh {
    rectangular_mesh_quads(0.0, 1.0, 0.0, 1.0, n, n)
}

/// Fallible [`unit_square_quads`]: zero/over-budget `n` returns
/// [`StructuredGridError`] instead of panicking or hanging.
pub fn try_unit_square_quads(n: usize) -> Result<Mesh, StructuredGridError> {
    try_rectangular_mesh_quads(0.0, 1.0, 0.0, 1.0, n, n)
}

/// Generate a unit cube mesh with tetrahedra
pub fn unit_cube_tetrahedra(n: usize) -> Mesh {
    box_mesh_tetrahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, n, n, n)
}

/// Fallible [`unit_cube_tetrahedra`]: zero/over-budget `n` returns
/// [`StructuredGridError`] instead of panicking or hanging.
pub fn try_unit_cube_tetrahedra(n: usize) -> Result<Mesh, StructuredGridError> {
    try_box_mesh_tetrahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, n, n, n)
}

/// Generate a unit cube mesh with hexahedra
pub fn unit_cube_hexahedra(n: usize) -> Mesh {
    box_mesh_hexahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, n, n, n)
}

/// Fallible [`unit_cube_hexahedra`]: zero/over-budget `n` returns
/// [`StructuredGridError`] instead of panicking or hanging.
pub fn try_unit_cube_hexahedra(n: usize) -> Result<Mesh, StructuredGridError> {
    try_box_mesh_hexahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, n, n, n)
}

/// Generate a backward-facing step mesh with triangular elements.
///
/// Creates a channel with a sudden expansion (step):
/// ```text
///     inlet_h
///   ┌──────┐
///   │      │  step_x
///   │      └──────────────────────┐
///   │                             │ channel_h
///   └─────────────────────────────┘
///   0      step_x            x_max
/// ```
///
/// Boundary markers:
/// - 1: inlet (x = 0, y ∈ [step_h, inlet_h])
/// - 2: outlet (x = x_max)
/// - 3: walls (top, bottom, step face)
///
/// Mapping produced by [`compact_unused_nodes_checked`].
///
/// `old_to_new[old]` is `None` when the old node was orphaned. Callers that
/// own nodal state can use this mapping to compact their vectors explicitly,
/// rather than losing track of DOFs after a coarsening operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeCompaction {
    /// New index for every original node, or `None` for an orphaned node.
    pub old_to_new: Vec<Option<usize>>,
    /// Number of orphaned nodes removed from the mesh.
    pub removed_nodes: usize,
}

/// Failure while transferring a nodal vector through [`NodeCompaction`].
///
/// The compaction mapping is public so that it can be serialized alongside a
/// remesh operation. These errors therefore validate it again rather than
/// assuming it was produced by [`compact_unused_nodes_checked`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeCompactionTransferError {
    /// The state vector does not have one value per pre-compaction node.
    ValueLengthMismatch {
        /// Number of supplied nodal values.
        values: usize,
        /// Number of source nodes expected by the mapping.
        old_nodes: usize,
    },
    /// A public mapping has two source nodes targeting one compacted node.
    DuplicateTarget {
        /// Source node whose mapping conflicts with an earlier source node.
        old_node: usize,
        /// Duplicate target node index.
        target_node: usize,
    },
    /// A public mapping leaves a compacted node without a source value.
    MissingTarget {
        /// Target compacted node index without a mapped source node.
        target_node: usize,
    },
    /// A public mapping cannot represent its target-vector length.
    TargetIndexOverflow,
}

impl fmt::Display for NodeCompactionTransferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ValueLengthMismatch { values, old_nodes } => write!(
                f,
                "nodal state has {values} values but compaction requires {old_nodes}"
            ),
            Self::DuplicateTarget {
                old_node,
                target_node,
            } => write!(
                f,
                "compaction maps source node {old_node} more than once to target node {target_node}"
            ),
            Self::MissingTarget { target_node } => write!(
                f,
                "compaction target node {target_node} has no source value"
            ),
            Self::TargetIndexOverflow => write!(f, "compaction target index overflows usize"),
        }
    }
}

impl std::error::Error for NodeCompactionTransferError {}

impl NodeCompaction {
    /// Remap a nodal state, adjoint, or design vector through this compaction.
    ///
    /// Values belonging to orphaned nodes are deliberately dropped. Retained
    /// values preserve their exact source-node association; the method does no
    /// interpolation or averaging. It therefore provides the conservative
    /// bookkeeping step needed immediately after coarsening/compaction, while
    /// geometric transfer between distinct meshes remains a separate policy.
    pub fn try_remap_nodal_values<T: Clone>(
        &self,
        values: &[T],
    ) -> Result<Vec<T>, NodeCompactionTransferError> {
        if values.len() != self.old_to_new.len() {
            return Err(NodeCompactionTransferError::ValueLengthMismatch {
                values: values.len(),
                old_nodes: self.old_to_new.len(),
            });
        }

        let target_len = match self.old_to_new.iter().flatten().copied().max() {
            Some(maximum) => maximum
                .checked_add(1)
                .ok_or(NodeCompactionTransferError::TargetIndexOverflow)?,
            None => 0,
        };
        let mut remapped: Vec<Option<T>> =
            std::iter::repeat_with(|| None).take(target_len).collect();
        for (old_node, target_node) in self.old_to_new.iter().enumerate() {
            let Some(target_node) = *target_node else {
                continue;
            };
            if remapped[target_node].is_some() {
                return Err(NodeCompactionTransferError::DuplicateTarget {
                    old_node,
                    target_node,
                });
            }
            remapped[target_node] = Some(values[old_node].clone());
        }
        remapped
            .into_iter()
            .enumerate()
            .map(|(target_node, value)| {
                value.ok_or(NodeCompactionTransferError::MissingTarget { target_node })
            })
            .collect()
    }
}

/// Remove orphaned nodes while preserving all element and boundary metadata.
///
/// The operation is transactional: malformed public mesh vectors leave the
/// mesh unchanged. Unlike the historical [`compact_mesh`] helper, this maps
/// every P1/P2/Q2 node, retains element IDs/refinement provenance, rewrites
/// every boundary node, and invalidates all derived caches.
pub fn compact_unused_nodes_checked(
    mesh: &mut Mesh,
) -> Result<NodeCompaction, MeshValidationError> {
    validate_mesh_admission(mesh)?;

    let mut used = vec![false; mesh.num_nodes()];
    for element in &mesh.elements {
        for &node in &element.nodes {
            used[node] = true;
        }
    }
    for boundary in &mesh.boundaries {
        for &node in &boundary.nodes {
            used[node] = true;
        }
    }

    let mut old_to_new = vec![None; mesh.num_nodes()];
    let mut nodes = Vec::with_capacity(mesh.num_nodes());
    for (old, point) in mesh.nodes.iter().copied().enumerate() {
        if used[old] {
            old_to_new[old] = Some(nodes.len());
            nodes.push(point);
        }
    }

    let mut candidate = mesh.clone();
    candidate.nodes = nodes;
    for element in &mut candidate.elements {
        for node in &mut element.nodes {
            *node = old_to_new[*node].expect("validated element node is marked used");
        }
    }
    for boundary in &mut candidate.boundaries {
        for node in &mut boundary.nodes {
            *node = old_to_new[*node].expect("validated boundary node is marked used");
        }
    }
    candidate.invalidate_caches();
    validate_mesh_admission(&candidate)?;

    let removed_nodes = mesh.num_nodes() - candidate.num_nodes();
    *mesh = candidate;
    Ok(NodeCompaction {
        old_to_new,
        removed_nodes,
    })
}

/// Remove orphaned nodes (not referenced by any element) and compact the mesh.
///
/// This compatibility helper returns a clone with contiguous node indices. New
/// stateful callers should prefer [`compact_unused_nodes_checked`] so their
/// nodal vectors can be remapped using its explicit index map.
///
/// Panics on malformed public mesh vectors; use
/// [`compact_unused_nodes_checked`] directly for deserialized or externally
/// mutated meshes.
pub fn compact_mesh(mesh: &Mesh) -> Mesh {
    let mut compacted = mesh.clone();
    compact_unused_nodes_checked(&mut compacted)
        .unwrap_or_else(|error| panic!("compact_mesh requires a valid mesh: {error}"));
    compacted
}

pub fn backward_facing_step_triangles(
    x_max: f64,
    channel_h: f64,
    step_x: f64,
    step_h: f64,
    nx: usize,
    ny: usize,
) -> Mesh {
    // Generate a full rectangular mesh [0, x_max] × [0, channel_h]
    // then remove elements in the step region [0, step_x] × [0, step_h]
    let full = rectangular_mesh_triangles(0.0, x_max, 0.0, channel_h, nx, ny);

    let mut mesh = Mesh::new(2);
    for node in &full.nodes {
        mesh.add_node(*node);
    }

    for elem in &full.elements {
        let verts = elem.vertices();
        let cx: f64 = verts.iter().map(|&v| full.nodes[v].x).sum::<f64>() / verts.len() as f64;
        let cy: f64 = verts.iter().map(|&v| full.nodes[v].y).sum::<f64>() / verts.len() as f64;

        // Keep elements outside the step block
        let in_step = cx < step_x && cy < step_h;
        if !in_step {
            mesh.add_element(elem.element_type, elem.vertices().to_vec());
        }
    }

    // Compact: remove orphaned nodes from element removal
    let mut mesh = compact_mesh(&mesh);
    mesh.detect_boundaries();

    // Re-tag boundaries
    let tol = x_max / nx as f64 * 0.5;
    for boundary in &mut mesh.boundaries {
        let nodes = &boundary.nodes;
        if nodes.len() < 2 {
            continue;
        }
        let avg_x: f64 = nodes.iter().map(|&n| mesh.nodes[n].x).sum::<f64>() / nodes.len() as f64;
        let avg_y: f64 = nodes.iter().map(|&n| mesh.nodes[n].y).sum::<f64>() / nodes.len() as f64;

        if (avg_x).abs() < tol && avg_y > step_h - tol {
            boundary.marker = 1; // inlet
        } else if (avg_x - x_max).abs() < tol {
            boundary.marker = 2; // outlet
        } else {
            boundary.marker = 3; // walls (top, bottom, step face)
        }
    }

    mesh
}

/// Fallible [`backward_facing_step_triangles`] for externally supplied
/// counts/geometry: zero counts, non-finite dimensions and over-budget
/// grids return [`StructuredGridError`] instead of panicking or hanging.
/// Step placement is semantic (no asserts); degenerate steps yield finite
/// degenerate meshes.
pub fn try_backward_facing_step_triangles(
    x_max: f64,
    channel_h: f64,
    step_x: f64,
    step_h: f64,
    nx: usize,
    ny: usize,
) -> Result<Mesh, StructuredGridError> {
    if nx == 0 {
        return Err(StructuredGridError::ZeroResolution {
            axis: "x",
            actual: nx,
        });
    }
    if ny == 0 {
        return Err(StructuredGridError::ZeroResolution {
            axis: "y",
            actual: ny,
        });
    }
    for (name, v) in [
        ("x_max", x_max),
        ("channel_h", channel_h),
        ("step_x", step_x),
        ("step_h", step_h),
    ] {
        if !v.is_finite() {
            return Err(StructuredGridError::NonFiniteExtent {
                axis: name,
                min: v,
                max: v,
            });
        }
    }
    StructuredGridError::check_cell_budget(
        "backward-facing-step grid",
        StructuredGridError::checked_cells(&[nx, ny]),
        StructuredGridError::BYTES_PER_CELL_2D,
    )?;
    Ok(backward_facing_step_triangles(
        x_max, channel_h, step_x, step_h, nx, ny,
    ))
}

/// Generate a rectangular channel mesh with a circular hole (cylinder obstacle).
///
/// Creates a structured triangular mesh over `[x_min, x_max] × [y_min, y_max]`
/// and removes all elements whose centroid lies inside the cylinder. Boundary
/// markers:
/// - 1: inlet (x = x_min)
/// - 2: outlet (x = x_max)
/// - 3: walls (y = y_min, y = y_max)
/// - 4: cylinder surface (edges adjacent to removed elements)
///
/// This is suitable for the Schäfer-Turek DFG flow-past-cylinder benchmark.
#[allow(clippy::too_many_arguments)]
pub fn channel_with_cylinder_triangles(
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    nx: usize,
    ny: usize,
    cyl_x: f64,
    cyl_y: f64,
    cyl_radius: f64,
) -> Mesh {
    assert!(
        nx > 0 && ny > 0,
        "channel mesh subdivisions must be non-zero"
    );
    assert!(
        x_min < x_max && y_min < y_max,
        "channel bounds must have positive extent"
    );
    assert!(
        cyl_radius.is_finite() && cyl_radius > 0.0,
        "cylinder radius must be finite and positive"
    );
    assert!(
        cyl_x - cyl_radius > x_min
            && cyl_x + cyl_radius < x_max
            && cyl_y - cyl_radius > y_min
            && cyl_y + cyl_radius < y_max,
        "cylinder must lie strictly inside the channel"
    );

    // Seed an unstructured Delaunay mesh with the background Cartesian grid
    // and an explicit polygon on the requested circle.  The old implementation
    // deleted structured triangles by centroid, producing a staircase obstacle
    // whose effective diameter and force normals depended strongly on nx/ny.
    // Fail-fast budget gate for the infallible entry: the segment count
    // below scales with the background resolution, so hostile counts hang
    // in triangulation. Untrusted input must use
    // try_channel_with_cylinder_triangles.
    StructuredGridError::check_cell_budget(
        "channel grid",
        StructuredGridError::checked_cells(&[nx, ny]),
        StructuredGridError::BYTES_PER_CELL_2D,
    )
    .expect("channel grid over cell budget; use try_channel_with_cylinder_triangles");
    let dx = (x_max - x_min) / nx as f64;
    let dy = (y_max - y_min) / ny as f64;
    let background_h = dx.min(dy);
    let cylinder_clearance = (2.5 * background_h).min(
        0.8 * (cyl_x - cyl_radius - x_min)
            .min(x_max - cyl_x - cyl_radius)
            .min(cyl_y - cyl_radius - y_min)
            .min(y_max - cyl_y - cyl_radius),
    );
    let clearance_radius = cyl_radius + cylinder_clearance;
    let mut points = Vec::with_capacity((nx + 1) * (ny + 1) + 64);
    for j in 0..=ny {
        for i in 0..=nx {
            // A small reflection-symmetric row perturbation removes the
            // cocircular ambiguity of every Cartesian cell. Without it,
            // Delaunay tie-breaking selects unrelated diagonals above and
            // below a centered obstacle and creates a large spurious lift.
            let row_phase = std::f64::consts::TAU * j as f64 / ny as f64;
            let column_window = (std::f64::consts::PI * i as f64 / nx as f64).sin();
            let x = x_min + i as f64 * dx + 0.05 * dx * column_window * row_phase.cos();
            let y = y_min + j as f64 * dy;
            let distance = ((x - cyl_x).powi(2) + (y - cyl_y).powi(2)).sqrt();
            if distance > clearance_radius {
                points.push((x, y));
            }
        }
    }

    // Resolve the curved boundary at twice the background linear resolution,
    // with a floor that keeps even coarse smoke meshes recognisably circular.
    let circumference = std::f64::consts::TAU * cyl_radius;
    let cylinder_segments = ((circumference / (0.5 * background_h)).ceil() as usize).max(32);
    // Boundary-layer rings provide well-shaped, nearly symmetric elements for
    // wall gradients and tractions without refining the entire long channel.
    for layer in 0..=4 {
        let radius = cyl_radius + cylinder_clearance * layer as f64 / 4.0;
        let angle_offset = if layer % 2 == 0 {
            0.0
        } else {
            std::f64::consts::PI / cylinder_segments as f64
        };
        for segment in 0..cylinder_segments {
            let angle =
                std::f64::consts::TAU * segment as f64 / cylinder_segments as f64 + angle_offset;
            points.push((cyl_x + radius * angle.cos(), cyl_y + radius * angle.sin()));
        }
    }

    let triangles = super::triangulate_points_2d(&points);
    let mut mesh = Mesh::new(2);
    for &(x, y) in &points {
        mesh.add_node(Point::new_2d(x, y));
    }
    for triangle in triangles {
        let cx = (points[triangle[0]].0 + points[triangle[1]].0 + points[triangle[2]].0) / 3.0;
        let cy = (points[triangle[0]].1 + points[triangle[1]].1 + points[triangle[2]].1) / 3.0;
        let centroid_radius = ((cx - cyl_x).powi(2) + (cy - cyl_y).powi(2)).sqrt();
        if centroid_radius >= cyl_radius {
            mesh.add_element(ElementType::Triangle, triangle.to_vec());
        }
    }

    // Compact: remove orphaned nodes from element removal
    let mut mesh = compact_mesh(&mesh);
    mesh.detect_boundaries();

    // Re-tag boundaries with specific markers
    let tol = 0.25 * background_h;
    for boundary in &mut mesh.boundaries {
        let nodes = &boundary.nodes;
        if nodes.len() < 2 {
            continue;
        }
        let avg_x: f64 = nodes.iter().map(|&n| mesh.nodes[n].x).sum::<f64>() / nodes.len() as f64;
        let avg_y: f64 = nodes.iter().map(|&n| mesh.nodes[n].y).sum::<f64>() / nodes.len() as f64;

        if (avg_x - x_min).abs() < tol {
            boundary.marker = 1; // inlet
        } else if (avg_x - x_max).abs() < tol {
            boundary.marker = 2; // outlet
        } else if (avg_y - y_min).abs() < tol || (avg_y - y_max).abs() < tol {
            boundary.marker = 3; // walls
        } else {
            // Interior boundary → cylinder surface
            let dist = ((avg_x - cyl_x).powi(2) + (avg_y - cyl_y).powi(2)).sqrt();
            if dist < cyl_radius * 2.0 {
                boundary.marker = 4; // cylinder
            }
        }
    }

    mesh
}

/// NACA 0012 airfoil polygon (unit chord, leading edge at origin, chord
/// along +x), ordered counter-clockwise starting at the trailing edge.
/// Uses the closed trailing edge (last coefficient -0.1036). A
/// finite-thickness trailing-edge face (0.0025 chords) was tried and
/// rejected: its face altitude, not the aerodynamics, set the global
/// explicit time step. The cosine sampling here only defines the shape
/// polygon for inside/distance tests; mesh seed stations are graded
/// separately in the generator.
pub fn naca0012_polygon(n_per_surface: usize) -> Vec<(f64, f64)> {
    let thickness = |x: f64| -> f64 {
        5.0 * 0.12
            * (0.2969 * x.sqrt() - 0.1260 * x - 0.3516 * x * x + 0.2843 * x * x * x
                - 0.1036 * x * x * x * x)
    };
    let n = n_per_surface.max(16);
    let station = |i: usize| -> f64 {
        0.5 * (1.0 - (std::f64::consts::PI * i as f64 / n as f64).cos())
    };
    let mut polygon = Vec::with_capacity(2 * n + 1);
    // Upper surface: trailing edge to leading edge.
    for i in (0..=n).rev() {
        let x = station(i);
        polygon.push((x, thickness(x)));
    }
    // Lower surface: leading edge back to trailing edge.
    for i in 1..n {
        let x = station(i);
        polygon.push((x, -thickness(x)));
    }
    polygon
}

/// Point-in-polygon test (even-odd ray casting) for a simple polygon.
fn point_in_polygon(point: (f64, f64), polygon: &[(f64, f64)]) -> bool {
    let mut inside = false;
    let mut j = polygon.len() - 1;
    for i in 0..polygon.len() {
        let (xi, yi) = polygon[i];
        let (xj, yj) = polygon[j];
        if (yi > point.1) != (yj > point.1) {
            let x_cross = xi + (point.1 - yi) * (xj - xi) / (yj - yi);
            if point.0 < x_cross {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

fn point_segment_distance(point: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (abx, aby) = (b.0 - a.0, b.1 - a.1);
    let length_sq = abx * abx + aby * aby;
    let t = if length_sq > 0.0 {
        (((point.0 - a.0) * abx + (point.1 - a.1) * aby) / length_sq).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let (dx, dy) = (point.0 - (a.0 + t * abx), point.1 - (a.1 + t * aby));
    (dx * dx + dy * dy).sqrt()
}

fn polygon_distance(point: (f64, f64), polygon: &[(f64, f64)]) -> f64 {
    let mut best = f64::INFINITY;
    for i in 0..polygon.len() {
        let a = polygon[i];
        let b = polygon[(i + 1) % polygon.len()];
        best = best.min(point_segment_distance(point, a, b));
    }
    best
}

/// Channel mesh with a body-fitted NACA 0012 airfoil, following the
/// construction of [`channel_with_cylinder_triangles`]: seeded Cartesian
/// background points, a densified body polygon with offset near-body
/// layers, Delaunay triangulation, and removal of elements inside the
/// body. Boundary markers match the cylinder convention: 1 inlet,
/// 2 outlet, 3 top/bottom, 4 airfoil surface. The airfoil occupies
/// `[chord_x, chord_x + chord] x [chord_y - 0.06 chord, chord_y + 0.06 chord]`
/// at zero geometric incidence; incidence is imposed through the
/// freestream direction by the solver driver.
#[allow(clippy::too_many_arguments)]
pub fn channel_with_airfoil_triangles(
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    nx: usize,
    ny: usize,
    chord_x: f64,
    chord_y: f64,
    chord: f64,
) -> Mesh {
    assert!(nx > 0 && ny > 0, "channel mesh subdivisions must be non-zero");
    assert!(x_min < x_max && y_min < y_max, "channel bounds must have positive extent");
    assert!(chord.is_finite() && chord > 0.0, "chord must be finite and positive");
    assert!(
        chord_x > x_min && chord_x + chord < x_max && chord_y - 0.06 * chord > y_min && chord_y + 0.06 * chord < y_max,
        "airfoil must lie strictly inside the channel"
    );
    StructuredGridError::check_cell_budget(
        "channel grid",
        StructuredGridError::checked_cells(&[nx, ny]),
        StructuredGridError::BYTES_PER_CELL_2D,
    )
    .expect("channel grid over cell budget");

    let dx = (x_max - x_min) / nx as f64;
    let dy = (y_max - y_min) / ny as f64;
    let background_h = dx.min(dy);
    let clearance = (2.0 * background_h).min(0.3 * chord);

    // Body polygon in physical coordinates (fine sampling for accurate
    // inside/distance tests).
    let polygon: Vec<(f64, f64)> = naca0012_polygon(200)
        .into_iter()
        .map(|(x, y)| (chord_x + chord * x, chord_y + chord * y))
        .collect();
    let body_centroid = (chord_x + 0.4 * chord, chord_y);

    // Graded surface seed stations per surface, marched from the leading
    // edge with geometrically growing spacing. Cosine sampling of the
    // shape polygon must NOT be used as the seed set: its leading/trailing
    // edge spacing is orders of magnitude below the wall-normal layer
    // distance, which produces aspect-ratio slivers whose minimum
    // altitude (and hence stable time step) collapses.
    let thickness_at = |x: f64| -> f64 {
        5.0 * 0.12
            * (0.2969 * x.sqrt() - 0.1260 * x - 0.3516 * x * x + 0.2843 * x * x * x
                - 0.1036 * x * x * x * x)
    };
    let ds_le = (0.12 * background_h).max(0.004 * chord);
    let ds_te = 0.5 * background_h;
    let mut surface_seeds: Vec<(f64, f64, f64)> = Vec::new(); // (x, y, local spacing)
    for sign in [1.0f64, -1.0] {
        let mut x = 0.0f64;
        let mut ds = ds_le;
        let mut first = true;
        loop {
            if sign < 0.0 && first {
                // Leading-edge point is shared with the upper surface.
                first = false;
                x = (x + ds).min(1.0);
                ds = (ds * 1.18).min(ds_te);
                continue;
            }
            first = false;
            surface_seeds.push((
                chord_x + chord * x,
                chord_y + sign * chord * thickness_at(x),
                ds * chord,
            ));
            if x >= 1.0 {
                break;
            }
            x = (x + ds).min(1.0);
            ds = (ds * 1.18).min(ds_te);
        }
    }

    // Outward normals at seed stations from the fine polygon: use the
    // analytic surface tangent of the thickness distribution.
    let seed_normals: Vec<(f64, f64)> = surface_seeds
        .iter()
        .map(|&(x, y, _)| {
            let xc = ((x - chord_x) / chord).clamp(0.0, 1.0);
            // d(yt)/dx by central difference on the unit-chord shape.
            let h = 1.0e-4;
            let slope = (thickness_at((xc + h).min(1.0)) - thickness_at((xc - h).max(0.0)))
                / ((xc + h).min(1.0) - (xc - h).max(0.0));
            let sign = if y >= chord_y { 1.0 } else { -1.0 };
            // Outward normal is (-slope, sign) on BOTH surfaces: the
            // lower surface is y = -yt(x), whose outward normal also has
            // x-component -slope. Using -slope*sign instead points the
            // lower leading-edge offsets downstream into the body and
            // breaks the nose symmetry (measured as spurious lift).
            let mut normal: (f64, f64) = (-slope, sign);
            let length = (normal.0.powi(2) + normal.1.powi(2)).sqrt().max(1.0e-300);
            normal.0 /= length;
            normal.1 /= length;
            let outward = (x - body_centroid.0) * normal.0 + (y - body_centroid.1) * normal.1;
            if outward < 0.0 && (xc - 0.4).abs() > 0.05 {
                normal = (-normal.0, -normal.1);
            }
            normal
        })
        .collect();

    let mut points: Vec<(f64, f64)> = Vec::with_capacity((nx + 1) * (ny + 1) + 512);
    for j in 0..=ny {
        for i in 0..=nx {
            // Same symmetry-breaking row perturbation as the cylinder mesh.
            let row_phase = std::f64::consts::TAU * j as f64 / ny as f64;
            let column_window = (std::f64::consts::PI * i as f64 / nx as f64).sin();
            let x = x_min + i as f64 * dx + 0.05 * dx * column_window * row_phase.cos();
            let y = y_min + j as f64 * dy;
            if !point_in_polygon((x, y), &polygon) && polygon_distance((x, y), &polygon) > clearance {
                points.push((x, y));
            }
        }
    }
    // Surface seeds plus three offset layers whose distance scales with
    // the local surface spacing (graded, aspect-ratio aware).
    for (point, _) in surface_seeds.iter().map(|&(x, y, ds)| ((x, y), ds)) {
        points.push(point);
    }
    for growth in [0.75, 2.0, 4.5] {
        for ((x, y, ds), normal) in surface_seeds.iter().zip(&seed_normals) {
            let offset = (ds * growth).min(clearance);
            points.push((x + offset * normal.0, y + offset * normal.1));
        }
    }

    let triangles = super::triangulate_points_2d(&points);
    let mut mesh = Mesh::new(2);
    for &(x, y) in &points {
        mesh.add_node(Point::new_2d(x, y));
    }
    for triangle in triangles {
        let cx = (points[triangle[0]].0 + points[triangle[1]].0 + points[triangle[2]].0) / 3.0;
        let cy = (points[triangle[0]].1 + points[triangle[1]].1 + points[triangle[2]].1) / 3.0;
        if !point_in_polygon((cx, cy), &polygon) {
            mesh.add_element(ElementType::Triangle, triangle.to_vec());
        }
    }
    let mut mesh = compact_mesh(&mesh);
    mesh.detect_boundaries();
    // Re-tag: faces on the domain box keep 1/2/3, everything else is the
    // airfoil (marker 4).
    let tol = 0.25 * background_h;
    for boundary in &mut mesh.boundaries {
        if boundary.nodes.len() < 2 {
            continue;
        }
        let avg_x: f64 =
            boundary.nodes.iter().map(|&n| mesh.nodes[n].x).sum::<f64>() / boundary.nodes.len() as f64;
        let avg_y: f64 =
            boundary.nodes.iter().map(|&n| mesh.nodes[n].y).sum::<f64>() / boundary.nodes.len() as f64;
        if (avg_x - x_min).abs() < tol {
            boundary.marker = 1;
        } else if (avg_x - x_max).abs() < tol {
            boundary.marker = 2;
        } else if (avg_y - y_min).abs() < tol || (avg_y - y_max).abs() < tol {
            boundary.marker = 3;
        } else {
            boundary.marker = 4;
        }
    }
    mesh
}

/// Fallible [`channel_with_cylinder_triangles`] for externally supplied
/// counts/geometry: zero counts, empty/non-finite spans, a non-positive
/// radius and over-budget grids return [`StructuredGridError`] instead of
/// panicking, dividing by zero, or hanging in triangulation. Cylinder
/// containment stays a documented panic (geometry, not counts).
#[allow(clippy::too_many_arguments)]
pub fn try_channel_with_cylinder_triangles(
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    nx: usize,
    ny: usize,
    cyl_x: f64,
    cyl_y: f64,
    cyl_radius: f64,
) -> Result<Mesh, StructuredGridError> {
    if nx == 0 {
        return Err(StructuredGridError::ZeroResolution {
            axis: "x",
            actual: nx,
        });
    }
    if ny == 0 {
        return Err(StructuredGridError::ZeroResolution {
            axis: "y",
            actual: ny,
        });
    }
    if !x_min.is_finite() || !x_max.is_finite() {
        return Err(StructuredGridError::NonFiniteExtent {
            axis: "x",
            min: x_min,
            max: x_max,
        });
    }
    if !y_min.is_finite() || !y_max.is_finite() {
        return Err(StructuredGridError::NonFiniteExtent {
            axis: "y",
            min: y_min,
            max: y_max,
        });
    }
    if x_min >= x_max {
        return Err(StructuredGridError::DegenerateExtent {
            axis: "x",
            min: x_min,
            max: x_max,
        });
    }
    if y_min >= y_max {
        return Err(StructuredGridError::DegenerateExtent {
            axis: "y",
            min: y_min,
            max: y_max,
        });
    }
    if !cyl_x.is_finite() || !cyl_y.is_finite() {
        return Err(StructuredGridError::NonFiniteExtent {
            axis: "cylinder center",
            min: cyl_x,
            max: cyl_y,
        });
    }
    if !cyl_radius.is_finite() {
        return Err(StructuredGridError::NonFiniteExtent {
            axis: "cylinder radius",
            min: cyl_radius,
            max: cyl_radius,
        });
    }
    if cyl_radius <= 0.0 {
        return Err(StructuredGridError::DegenerateExtent {
            axis: "cylinder radius",
            min: cyl_radius,
            max: cyl_radius,
        });
    }
    StructuredGridError::check_cell_budget(
        "channel grid",
        StructuredGridError::checked_cells(&[nx, ny]),
        StructuredGridError::BYTES_PER_CELL_2D,
    )?;
    Ok(channel_with_cylinder_triangles(
        x_min, x_max, y_min, y_max, nx, ny, cyl_x, cyl_y, cyl_radius,
    ))
}

// ---------------------------------------------------------------------------
// Duct profile mesh for axisymmetric acoustics
// ---------------------------------------------------------------------------

/// Generate a 2D triangular mesh for an axisymmetric duct in the (r, z) plane.
///
/// The mesh x-coordinate represents radial distance r, and y-coordinate
/// represents axial distance z. At each axial station, nodes span from
/// `r_inner(z)` to `r_outer(z)`.
///
/// # Arguments
///
/// * `z_min`, `z_max` — axial extent of the duct
/// * `r_inner` — closure `|z| -> r` for the inner boundary (use `|_| 0.0` for axis)
/// * `r_outer` — closure `|z| -> r` for the outer boundary (duct wall or domain edge)
/// * `nz` — number of elements in the axial direction
/// * `nr` — number of elements in the radial direction
///
/// # Boundary tags
///
/// * 1 — inlet (z = z_min)
/// * 2 — outlet (z = z_max)
/// * 3 — inner boundary (r = r_inner, may be axis at r=0)
/// * 4 — outer boundary (r = r_outer, typically the duct wall)
///
/// # Panics
///
/// Panics if `nz < 1`, `nr < 1`, or `z_min >= z_max`.
pub fn duct_profile_mesh<Fi: Fn(f64) -> f64, Fo: Fn(f64) -> f64>(
    z_min: f64,
    z_max: f64,
    r_inner: Fi,
    r_outer: Fo,
    nz: usize,
    nr: usize,
) -> Mesh {
    assert!(
        nz >= 1 && nr >= 1,
        "need at least 1 element in each direction"
    );
    assert!(z_min < z_max, "z_min must be less than z_max");
    // Fail-fast budget gate for the infallible entry: untrusted counts must
    // use try_duct_profile_mesh. Hostile counts panic here instead of
    // hanging in the profile loops below.
    StructuredGridError::check_cell_budget(
        "duct profile grid",
        StructuredGridError::checked_cells(&[nz, nr]),
        StructuredGridError::BYTES_PER_CELL_2D,
    )
    .expect("duct profile grid over cell budget; use try_duct_profile_mesh");

    let mut mesh = Mesh::new(2);

    let dz = (z_max - z_min) / nz as f64;

    for j in 0..=nz {
        let z = z_min + j as f64 * dz;
        let ri = r_inner(z);
        let ro = r_outer(z);
        let dr = (ro - ri) / nr as f64;
        for i in 0..=nr {
            let r = ri + i as f64 * dr;
            mesh.add_node(Point::new_2d(r, z));
        }
    }

    // Create triangular elements (2 per quad cell)
    let stride = nr + 1;
    for j in 0..nz {
        for i in 0..nr {
            let n00 = j * stride + i;
            let n10 = j * stride + (i + 1);
            let n01 = (j + 1) * stride + i;
            let n11 = (j + 1) * stride + (i + 1);

            mesh.add_element(ElementType::Triangle, vec![n00, n10, n11]);
            mesh.add_element(ElementType::Triangle, vec![n00, n11, n01]);
        }
    }

    mesh.detect_boundaries();

    let tol_z = dz * 0.1;
    let tol_r = (r_outer(z_min) - r_inner(z_min)) / nr as f64 * 0.1;

    // Tag 1: inlet (z = z_min)
    mesh.set_boundary_condition(BoundaryType::Neumann, 1, |points| {
        points.iter().all(|p| (p.y - z_min).abs() < tol_z)
    });

    // Tag 2: outlet (z = z_max)
    mesh.set_boundary_condition(BoundaryType::Neumann, 2, |points| {
        points.iter().all(|p| (p.y - z_max).abs() < tol_z)
    });

    // Tag 3: inner boundary (r = r_inner)
    {
        let r_inner_ref = &r_inner;
        mesh.set_boundary_condition(BoundaryType::Neumann, 3, |points| {
            points.iter().all(|p| {
                let expected = r_inner_ref(p.y);
                (p.x - expected).abs() < tol_r
            })
        });
    }

    // Tag 4: outer boundary (r = r_outer)
    {
        let r_outer_ref = &r_outer;
        mesh.set_boundary_condition(BoundaryType::Neumann, 4, |points| {
            points.iter().all(|p| {
                let expected = r_outer_ref(p.y);
                (p.x - expected).abs() < tol_r
            })
        });
    }

    mesh
}

/// Fallible [`duct_profile_mesh`] for externally supplied counts/extents:
/// zero counts, empty/non-finite z spans and over-budget grids return
/// [`StructuredGridError`] instead of panicking or hanging. The profile
/// closures' outputs are evaluated per station and cannot be admitted
/// statically; degenerate profiles yield finite degenerate rows.
pub fn try_duct_profile_mesh<Fi: Fn(f64) -> f64, Fo: Fn(f64) -> f64>(
    z_min: f64,
    z_max: f64,
    r_inner: Fi,
    r_outer: Fo,
    nz: usize,
    nr: usize,
) -> Result<Mesh, StructuredGridError> {
    if nz == 0 {
        return Err(StructuredGridError::ZeroResolution {
            axis: "z",
            actual: nz,
        });
    }
    if nr == 0 {
        return Err(StructuredGridError::ZeroResolution {
            axis: "r",
            actual: nr,
        });
    }
    if !z_min.is_finite() || !z_max.is_finite() {
        return Err(StructuredGridError::NonFiniteExtent {
            axis: "z",
            min: z_min,
            max: z_max,
        });
    }
    if z_min >= z_max {
        return Err(StructuredGridError::DegenerateExtent {
            axis: "z",
            min: z_min,
            max: z_max,
        });
    }
    StructuredGridError::check_cell_budget(
        "duct profile grid",
        StructuredGridError::checked_cells(&[nz, nr]),
        StructuredGridError::BYTES_PER_CELL_2D,
    )?;
    Ok(duct_profile_mesh(z_min, z_max, r_inner, r_outer, nz, nr))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::p2::promote_to_p2;

    #[test]
    fn test_rectangular_mesh_triangles() {
        let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 2, 2);

        // 3x3 = 9 nodes
        assert_eq!(mesh.num_nodes(), 9);
        // 2x2 cells x 2 triangles = 8 triangles
        assert_eq!(mesh.num_elements(), 8);
    }

    #[test]
    fn checked_node_compaction_preserves_p2_connectivity_and_boundary_metadata() {
        let base = unit_square_triangles(1);
        let mut mesh = promote_to_p2(&base);
        // Put an orphan before all active nodes to verify that every P2 and
        // boundary index is actually remapped, not merely retained by chance.
        mesh.nodes.insert(0, Point::new_2d(-1.0, -1.0));
        for element in &mut mesh.elements {
            for node in &mut element.nodes {
                *node += 1;
            }
            element.parent_id = Some(element.id + 100);
            element.level = 3;
        }
        for boundary in &mut mesh.boundaries {
            for node in &mut boundary.nodes {
                *node += 1;
            }
        }
        mesh.invalidate_caches();
        mesh.validate_connectivity().unwrap();
        let before_elements = mesh.elements.clone();
        let before_boundaries = mesh.boundaries.clone();
        let nodal_state: Vec<_> = (0..mesh.num_nodes() as i32).collect();

        let result = compact_unused_nodes_checked(&mut mesh).unwrap();

        assert_eq!(result.removed_nodes, 1);
        assert_eq!(result.old_to_new[0], None);
        for (old, new) in result.old_to_new.iter().enumerate().skip(1) {
            assert_eq!(*new, Some(old - 1));
        }
        assert!(mesh.elements.iter().all(|element| element.is_p2()));
        for (before, after) in before_elements.iter().zip(&mesh.elements) {
            assert_eq!(after.id, before.id);
            assert_eq!(after.parent_id, before.parent_id);
            assert_eq!(after.level, before.level);
            assert_eq!(after.marker, before.marker);
            let expected: Vec<_> = before
                .nodes
                .iter()
                .map(|&node| result.old_to_new[node].unwrap())
                .collect();
            assert_eq!(after.nodes, expected);
        }
        for (before, after) in before_boundaries.iter().zip(&mesh.boundaries) {
            assert_eq!(after.boundary_type, before.boundary_type);
            assert_eq!(after.marker, before.marker);
            assert_eq!(after.element_idx, before.element_idx);
            assert_eq!(after.local_idx, before.local_idx);
            let expected: Vec<_> = before
                .nodes
                .iter()
                .map(|&node| result.old_to_new[node].unwrap())
                .collect();
            assert_eq!(after.nodes, expected);
        }
        assert!(mesh.node_to_elements.is_none());
        assert!(mesh.topology.is_none());
        assert!(mesh.element_neighbors.is_none());
        assert!(mesh.validate_connectivity().is_ok());
        assert_eq!(
            result.try_remap_nodal_values(&nodal_state).unwrap(),
            (1..nodal_state.len() as i32).collect::<Vec<_>>(),
            "the orphaned state is dropped while every retained P2 value keeps its node association"
        );
    }

    #[test]
    fn node_compaction_state_transfer_rejects_malformed_public_mappings() {
        assert!(matches!(
            NodeCompaction {
                old_to_new: vec![Some(0)],
                removed_nodes: 0,
            }
            .try_remap_nodal_values::<f64>(&[]),
            Err(NodeCompactionTransferError::ValueLengthMismatch { .. })
        ));
        assert!(matches!(
            NodeCompaction {
                old_to_new: vec![Some(0), Some(0)],
                removed_nodes: 0,
            }
            .try_remap_nodal_values(&[1.0, 2.0]),
            Err(NodeCompactionTransferError::DuplicateTarget {
                old_node: 1,
                target_node: 0,
            })
        ));
        assert!(matches!(
            NodeCompaction {
                old_to_new: vec![Some(1)],
                removed_nodes: 0,
            }
            .try_remap_nodal_values(&[1.0]),
            Err(NodeCompactionTransferError::MissingTarget { target_node: 0 })
        ));
    }

    #[test]
    fn checked_node_compaction_rejects_invalid_mesh_without_mutation() {
        let mut mesh = unit_square_triangles(1);
        mesh.elements[0].nodes[0] = mesh.num_nodes();
        let nodes = mesh.nodes.clone();
        let elements = mesh.elements.clone();
        let boundaries = mesh.boundaries.clone();

        assert!(matches!(
            compact_unused_nodes_checked(&mut mesh),
            Err(MeshValidationError::InvalidMesh { .. })
        ));
        assert_eq!(mesh.nodes, nodes);
        assert_eq!(mesh.elements.len(), elements.len());
        assert_eq!(mesh.elements[0].nodes, elements[0].nodes);
        assert_eq!(mesh.boundaries.len(), boundaries.len());
        assert_eq!(mesh.boundaries[0].nodes, boundaries[0].nodes);
    }

    #[test]
    fn test_rectangular_mesh_quads() {
        let mesh = rectangular_mesh_quads(0.0, 1.0, 0.0, 1.0, 3, 3);

        assert_eq!(mesh.num_nodes(), 16);
        assert_eq!(mesh.num_elements(), 9);
    }

    #[test]
    fn test_box_mesh_tetrahedra() {
        let mesh = box_mesh_tetrahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2, 2, 2);

        // 3x3x3 = 27 nodes
        assert_eq!(mesh.num_nodes(), 27);
        // 2x2x2 cubes x 6 tetrahedra = 48 tetrahedra
        assert_eq!(mesh.num_elements(), 48);
    }

    #[test]
    fn test_circular_mesh() {
        let mesh = circular_mesh_triangles(0.0, 0.0, 1.0, 3, 8);

        // Center + 3 rings of 8 nodes = 1 + 24 = 25 nodes
        assert_eq!(mesh.num_nodes(), 25);
    }

    #[test]
    fn test_annular_mesh() {
        let mesh = annular_mesh_triangles(0.0, 0.0, 0.5, 2.0, 4, 16);

        // (n_radial + 1) rings of n_angular nodes = 5 * 16 = 80 nodes
        assert_eq!(mesh.num_nodes(), 80);

        // n_radial * n_angular * 2 triangles = 4 * 16 * 2 = 128 elements
        assert_eq!(mesh.num_elements(), 128);

        // Should have boundaries on both inner and outer circles
        assert!(!mesh.boundaries.is_empty());
    }

    #[test]
    fn test_unit_square() {
        let mesh = unit_square_triangles(4);
        assert_eq!(mesh.num_nodes(), 25);
        assert_eq!(mesh.num_elements(), 32);
    }

    #[test]
    fn test_boundary_detection() {
        let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 1.0, 2, 2);

        // Should have 8 boundary edges (2 per side of square)
        assert_eq!(mesh.boundaries.len(), 8);
    }

    fn mean_circle_boundary_error(mesh: &Mesh, marker: i32, radius: f64) -> f64 {
        let mut nodes: Vec<_> = mesh
            .boundaries
            .iter()
            .filter(|boundary| boundary.marker == marker)
            .flat_map(|boundary| boundary.nodes.iter().copied())
            .collect();
        nodes.sort_unstable();
        nodes.dedup();
        nodes
            .iter()
            .map(|&node| {
                let point = &mesh.nodes[node];
                ((point.x * point.x + point.y * point.y).sqrt() - radius).abs()
            })
            .sum::<f64>()
            / nodes.len() as f64
    }

    fn has_hanging_node(mesh: &Mesh) -> bool {
        mesh.elements.iter().any(|element| {
            let vertices = element.vertices();
            [
                (vertices[0], vertices[1]),
                (vertices[1], vertices[2]),
                (vertices[2], vertices[0]),
            ]
            .into_iter()
            .any(|(first, second)| {
                let a = mesh.nodes[first];
                let b = mesh.nodes[second];
                let dx = b.x - a.x;
                let dy = b.y - a.y;
                let length_squared = dx * dx + dy * dy;
                mesh.nodes.iter().enumerate().any(|(node, point)| {
                    if node == first || node == second {
                        return false;
                    }
                    let px = point.x - a.x;
                    let py = point.y - a.y;
                    let cross = dx * py - dy * px;
                    let projection = dx * px + dy * py;
                    cross.abs() <= 1.0e-12 * length_squared.sqrt()
                        && projection > 1.0e-12 * length_squared
                        && projection < (1.0 - 1.0e-12) * length_squared
                })
            })
        })
    }

    #[test]
    fn fitted_implicit_circle_reduces_staircase_error_and_refines_locally() {
        let radius = 0.8;
        let level_set = |x: f64, y: f64| x * x + y * y - radius * radius;
        let staircase = implicit_region_mesh_triangles(-1.0, 1.0, -1.0, 1.0, 24, 24, 7, |x, y| {
            level_set(x, y) <= 0.0
        });
        let fitted = fitted_implicit_region_mesh_triangles(
            -1.0,
            1.0,
            -1.0,
            1.0,
            7,
            FittedImplicitRegionConfig::new(24, 24),
            level_set,
        );
        let staircase_error = mean_circle_boundary_error(&staircase, 7, radius);
        let fitted_error = mean_circle_boundary_error(&fitted, 7, radius);
        assert!(
            fitted_error < 0.45 * staircase_error,
            "fitted error {fitted_error} versus staircase {staircase_error}"
        );
        assert!(fitted.num_elements() > staircase.num_elements());
        assert!(fitted.num_elements() < 4 * staircase.num_elements());
        assert!(!has_hanging_node(&fitted));
        assert!(
            fitted
                .boundaries
                .iter()
                .all(|boundary| boundary.marker == 7)
        );
    }

    #[test]
    fn fitted_binary_mask_improves_smooth_disk_geometry() {
        let n = 20;
        let radius = 0.62;
        let mask: Vec<Vec<bool>> = (0..n)
            .map(|j| {
                (0..n)
                    .map(|i| {
                        let x = -1.0 + 2.0 * (i as f64 + 0.5) / n as f64;
                        let y = -1.0 + 2.0 * (j as f64 + 0.5) / n as f64;
                        x * x + y * y <= radius * radius
                    })
                    .collect()
            })
            .collect();
        let pixel = binary_mask_mesh_triangles(&mask, -1.0, 1.0, -1.0, 1.0, 3);
        let fitted = fitted_binary_mask_mesh_triangles(&mask, -1.0, 1.0, -1.0, 1.0, 3, 1);
        let pixel_error = mean_circle_boundary_error(&pixel, 3, radius);
        let fitted_error = mean_circle_boundary_error(&fitted, 3, radius);
        assert!(
            fitted_error < pixel_error,
            "fitted image error {fitted_error} versus pixel staircase {pixel_error}"
        );
    }

    #[test]
    fn channel_with_cylinder_is_body_fitted() {
        let center = (0.2_f64, 0.2_f64);
        let radius = 0.05_f64;
        let mesh =
            channel_with_cylinder_triangles(0.0, 2.2, 0.0, 0.41, 44, 8, center.0, center.1, radius);
        let cylinder_edges: Vec<_> = mesh
            .boundaries
            .iter()
            .filter(|boundary| boundary.marker == 4)
            .collect();

        assert!(
            cylinder_edges.len() >= 12,
            "cylinder needs enough fitted edges, got {}",
            cylinder_edges.len()
        );
        for boundary in cylinder_edges {
            assert_eq!(boundary.nodes.len(), 2);
            assert!(boundary.element_idx < mesh.elements.len());
            for &node in &boundary.nodes {
                let point = mesh.nodes[node];
                let node_radius =
                    ((point.x - center.0).powi(2) + (point.y - center.1).powi(2)).sqrt();
                assert!(
                    (node_radius - radius).abs() < 1.0e-12,
                    "cylinder node radius {node_radius} is not the requested {radius}"
                );
            }
        }
    }

    #[test]
    fn centered_channel_cylinder_mesh_preserves_reflection_symmetry() {
        use std::collections::HashSet;

        let height = 1.0;
        let mesh =
            channel_with_cylinder_triangles(0.0, 3.0, 0.0, height, 24, 8, 0.75, 0.5 * height, 0.2);
        let key = |points: &[(f64, f64); 3]| {
            let mut quantized =
                points.map(|(x, y)| ((x * 1.0e10).round() as i64, (y * 1.0e10).round() as i64));
            quantized.sort_unstable();
            quantized
        };
        let element_keys: HashSet<_> = mesh
            .elements
            .iter()
            .map(|element| {
                let vertices = element.vertices();
                key(&[
                    (mesh.nodes[vertices[0]].x, mesh.nodes[vertices[0]].y),
                    (mesh.nodes[vertices[1]].x, mesh.nodes[vertices[1]].y),
                    (mesh.nodes[vertices[2]].x, mesh.nodes[vertices[2]].y),
                ])
            })
            .collect();

        for element in &mesh.elements {
            let vertices = element.vertices();
            let reflected = key(&[
                (
                    mesh.nodes[vertices[0]].x,
                    height - mesh.nodes[vertices[0]].y,
                ),
                (
                    mesh.nodes[vertices[1]].x,
                    height - mesh.nodes[vertices[1]].y,
                ),
                (
                    mesh.nodes[vertices[2]].x,
                    height - mesh.nodes[vertices[2]].y,
                ),
            ]);
            assert!(
                element_keys.contains(&reflected),
                "triangulation has no reflected counterpart for {reflected:?}"
            );
        }
    }

    #[test]
    fn test_spherical_shell_mesh() {
        // Icosphere subdiv 0 = 12 nodes, 20 tris
        // 2 layers
        // Total nodes = 12 * 3 = 36 nodes
        // Total elems = 2 layers * 20 tris * 3 tets = 120 tets
        let mesh = spherical_shell_mesh_tetrahedra(0.0, 0.0, 0.0, 1.0, 2.0, 0, 2);

        assert_eq!(mesh.num_nodes(), 36);
        assert_eq!(mesh.num_elements(), 120);
        assert_eq!(mesh.dimension, 3);

        // Check boundaries
        // Outer + Inner = 20 + 20 = 40 boundary faces
        assert_eq!(mesh.boundaries.len(), 40);
    }

    #[test]
    fn symmetric_box_has_24_tets_per_cube() {
        // 2×2×2 cubes → 8 cells × 24 tets/cell = 192 tetrahedra.
        let mesh = box_mesh_tetrahedra_symmetric(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 2, 2, 2);
        assert_eq!(mesh.num_elements(), 192);
        assert_eq!(mesh.dimension, 3);
    }

    #[test]
    fn symmetric_box_shares_interior_faces() {
        // Unequal axis counts exercise all three shared-face index layouts.
        let (nx, ny, nz) = (2, 3, 4);
        let mesh = box_mesh_tetrahedra_symmetric(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, nx, ny, nz);
        let expected_nodes = (nx + 1) * (ny + 1) * (nz + 1)
            + (nx + 1) * ny * nz
            + nx * (ny + 1) * nz
            + nx * ny * (nz + 1)
            + nx * ny * nz;
        assert_eq!(mesh.num_nodes(), expected_nodes);
        assert_eq!(mesh.boundaries.len(), 8 * (nx * ny + nx * nz + ny * nz));
        for face in &mesh.boundaries {
            assert!(
                (0..3).any(|axis| [0.0, 1.0].iter().any(|&plane| {
                    face.nodes.iter().all(|&node| {
                        let p = &mesh.nodes[node];
                        ([p.x, p.y, p.z][axis] - plane).abs() < 1e-12
                    })
                })),
                "interior face incorrectly classified as boundary: {:?}",
                face.nodes
            );
        }
    }

    #[test]
    fn symmetric_box_is_invariant_under_90_deg_z_rotation() {
        // Build a symmetric box centred on origin and verify that
        // rotating every node by +90° about Z maps the node set onto
        // itself (modulo permutation). This is the very symmetry the
        // Kuhn decomposition breaks.
        let mesh = box_mesh_tetrahedra_symmetric(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 2, 2, 2);
        let nodes: Vec<(f64, f64, f64)> = mesh.nodes.iter().map(|p| (p.x, p.y, p.z)).collect();
        // Rotate +90° about Z: (x, y, z) → (-y, x, z).
        let rotated: Vec<(f64, f64, f64)> = nodes.iter().map(|&(x, y, z)| (-y, x, z)).collect();
        // Every rotated node must coincide with some original node
        // within float tolerance.
        let tol = 1.0e-9;
        for r in &rotated {
            let found = nodes.iter().any(|n| {
                (n.0 - r.0).abs() < tol && (n.1 - r.1).abs() < tol && (n.2 - r.2).abs() < tol
            });
            assert!(found, "rotated node {r:?} has no match in original mesh");
        }
    }

    #[test]
    fn convex_hull_surface_tetrahedron_has_four_faces() {
        let pts = vec![
            Point::new_3d(0.0, 0.0, 0.0),
            Point::new_3d(1.0, 0.0, 0.0),
            Point::new_3d(0.0, 1.0, 0.0),
            Point::new_3d(0.0, 0.0, 1.0),
        ];
        let mesh = convex_hull_surface(&pts).expect("tet hull builds");
        assert_eq!(mesh.dimension, 3);
        assert_eq!(mesh.num_nodes(), 4);
        assert_eq!(mesh.num_elements(), 4);
        assert!(
            mesh.elements
                .iter()
                .all(|e| e.element_type == ElementType::Triangle)
        );
    }

    #[test]
    fn convex_hull_surface_drops_interior_point() {
        // 4 tet corners + 1 interior point. The interior point must not
        // appear in any hull face.
        let pts = vec![
            Point::new_3d(0.0, 0.0, 0.0),
            Point::new_3d(2.0, 0.0, 0.0),
            Point::new_3d(0.0, 2.0, 0.0),
            Point::new_3d(0.0, 0.0, 2.0),
            Point::new_3d(0.25, 0.25, 0.25), // strictly interior
        ];
        let mesh = convex_hull_surface(&pts).expect("hull builds");
        assert_eq!(mesh.num_nodes(), 4);
        assert_eq!(mesh.num_elements(), 4);
    }

    #[test]
    fn two_disk_union_mesh_is_compact_and_preserves_area() {
        let radius = 0.75;
        let separation = 0.9;
        let nx = 120;
        let ny = 80;
        let mesh = two_disk_union_mesh_triangles(
            (-separation / 2.0, 0.0),
            (separation / 2.0, 0.0),
            radius,
            radius,
            nx,
            ny,
            7,
        );

        assert!(!mesh.elements.is_empty());
        assert!(mesh.boundaries.iter().all(|boundary| boundary.marker == 7));
        let mut referenced = vec![false; mesh.num_nodes()];
        for element in &mesh.elements {
            for &node in &element.nodes {
                referenced[node] = true;
            }
        }
        assert!(referenced.into_iter().all(|value| value));

        let dx = (separation + 2.0 * radius) / nx as f64;
        let dy = (2.0 * radius) / ny as f64;
        let actual_area = mesh.num_elements() as f64 * dx * dy / 2.0;
        let overlap = 2.0 * radius.powi(2) * (separation / (2.0 * radius)).acos()
            - 0.5 * separation * (4.0 * radius.powi(2) - separation.powi(2)).sqrt();
        let exact_area = 2.0 * std::f64::consts::PI * radius.powi(2) - overlap;
        assert!((actual_area - exact_area).abs() / exact_area < 0.01);
    }

    #[test]
    fn binary_mask_mesh_preserves_pixels_and_internal_holes() {
        let mask = vec![
            vec![false, true, true, true, false],
            vec![true, true, true, true, true],
            vec![true, true, false, true, true],
            vec![true, true, true, true, true],
        ];
        let active_pixels = mask.iter().flatten().filter(|&&active| active).count();
        let mesh = binary_mask_mesh_triangles(&mask, 0.0, 5.0, 0.0, 4.0, 9);

        assert_eq!(mesh.num_elements(), 2 * active_pixels);
        assert!(mesh.boundaries.iter().all(|boundary| boundary.marker == 9));
        let hole_edges = mesh
            .boundaries
            .iter()
            .filter(|boundary| {
                boundary.nodes.iter().all(|&node| {
                    let point = &mesh.nodes[node];
                    (2.0..=3.0).contains(&point.x) && (2.0..=3.0).contains(&point.y)
                })
            })
            .count();
        assert_eq!(hole_edges, 4);
    }

    #[test]
    fn layered_spherical_shell_shares_marked_material_interfaces() {
        let mesh = layered_spherical_shell_mesh_tetrahedra(
            0.0,
            0.0,
            0.0,
            &[1.0, 2.0, 4.0, 6.0],
            1,
            &[2, 2, 2],
        );
        assert!(
            mesh.elements
                .iter()
                .all(|element| (1..=3).contains(&element.marker))
        );
        assert!(mesh.boundaries.iter().any(|boundary| boundary.marker == 1));
        assert!(mesh.boundaries.iter().any(|boundary| boundary.marker == 2));

        for (radius, neighboring_markers) in [(2.0, [1, 2]), (4.0, [2, 3])] {
            for (node_index, _node) in mesh.nodes.iter().enumerate().filter(|(_, node)| {
                (node.distance(&Point::new_3d(0.0, 0.0, 0.0)) - radius).abs() < 1.0e-12
            }) {
                let markers: HashSet<_> = mesh
                    .elements
                    .iter()
                    .filter(|element| element.nodes.contains(&node_index))
                    .map(|element| element.marker)
                    .collect();
                assert_eq!(markers, neighboring_markers.into_iter().collect());
            }
        }
    }
}
