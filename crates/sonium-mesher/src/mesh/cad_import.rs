//! Mesh-exchange CAD import for named surface selections.
//!
//! The pure-Rust core never links a CAD kernel. This module accepts only
//! triangulated exchange geometry (STL and OBJ), scales it by an explicit
//! caller-supplied factor, assigns every triangle to one named physical
//! group, and reports manifoldness and small-feature defeaturing actions.
//! Parametric STEP geometry must be tessellated by an outer tool first;
//! this module fails closed on any other input instead of guessing.

// Rust guideline compliant 2026-02-21

use super::types::{PhysicalGroupRegistry, Point};
use std::collections::HashMap;

/// Errors returned by mesh-exchange import.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum CadImportError {
    /// The byte payload is not recognised STL or OBJ geometry.
    #[error("unrecognised mesh-exchange payload: expected STL or OBJ triangles")]
    UnrecognisedFormat,
    /// A vertex or index failed validation.
    #[error("invalid imported geometry: {0}")]
    InvalidGeometry(String),
    /// The explicit unit scale was not finite and positive.
    #[error("unit scale must be finite and positive, got {0}")]
    InvalidUnitScale(f64),
    /// The named selection could not be registered.
    #[error("invalid selection name: {0}")]
    InvalidSelection(String),
    /// The defeaturing threshold was not finite and non-negative.
    #[error("defeature area threshold must be finite and non-negative, got {0}")]
    InvalidDefeatureThreshold(f64),
}

/// How a small imported feature is handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefeatureOp {
    /// Drop triangles at or below the area threshold.
    Remove,
    /// Keep triangles and list them in the report.
    Flag,
    /// Keep triangles and mark them for mesh-around sizing.
    MeshAround,
}

/// One defeatured feature found during import.
#[derive(Debug, Clone, PartialEq)]
pub struct DefeaturedFeature {
    /// Triangle index in the imported ordering.
    pub triangle: usize,
    /// Triangle area in square metres after scaling.
    pub area_m2: f64,
    /// Handling applied to the feature.
    pub op: DefeatureOp,
}

/// Manifoldness and defeaturing evidence for an imported surface.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportReport {
    /// Triangles read before defeaturing.
    pub triangles_read: usize,
    /// Triangles kept after defeaturing.
    pub triangles_kept: usize,
    /// Edges shared by exactly two triangles.
    pub manifold_edges: usize,
    /// Edges shared by exactly one triangle.
    pub boundary_edges: usize,
    /// Edges shared by more than two triangles.
    pub nonmanifold_edges: usize,
    /// Small features found at or below the defeature threshold.
    pub defeatured: Vec<DefeaturedFeature>,
}

impl ImportReport {
    /// Returns true when every edge is shared by exactly two triangles.
    #[must_use]
    pub fn is_watertight(&self) -> bool {
        self.boundary_edges == 0 && self.nonmanifold_edges == 0 && self.triangles_kept > 0
    }
}

/// A triangulated surface with one named selection.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportedSurface {
    /// Scaled vertex positions in metres.
    pub nodes: Vec<Point>,
    /// Triangle vertex indices into `nodes`.
    pub triangles: Vec<[usize; 3]>,
    /// Named selections; the imported surface registers dimension 2.
    pub groups: PhysicalGroupRegistry,
    /// Manifoldness and defeaturing evidence.
    pub report: ImportReport,
}

/// Imports STL or OBJ bytes as a named surface selection.
///
/// `unit_scale_m` is the explicit metres-per-source-unit factor (for
/// example `0.001` for millimetre STL). It is required and validated;
/// no unit is ever guessed from the payload.
///
/// # Errors
///
/// Returns [`CadImportError`] for unrecognised payloads, non-finite or
/// out-of-range geometry, an invalid scale, or an invalid selection.
pub fn import_surface(
    bytes: &[u8],
    selection_name: &str,
    unit_scale_m: f64,
    defeature_op: DefeatureOp,
    defeature_area_m2: f64,
) -> Result<ImportedSurface, CadImportError> {
    if !unit_scale_m.is_finite() || unit_scale_m <= 0.0 {
        return Err(CadImportError::InvalidUnitScale(unit_scale_m));
    }
    if !defeature_area_m2.is_finite() || defeature_area_m2 < 0.0 {
        return Err(CadImportError::InvalidDefeatureThreshold(defeature_area_m2));
    }
    if selection_name.trim().is_empty() {
        return Err(CadImportError::InvalidSelection("empty name".to_owned()));
    }
    let (nodes, triangles) = if looks_like_obj(bytes) {
        parse_obj(bytes, unit_scale_m)?
    } else {
        parse_stl(bytes, unit_scale_m)?
    };
    finish_import(nodes, triangles, selection_name, defeature_op, defeature_area_m2)
}

fn finish_import(
    nodes: Vec<Point>,
    triangles: Vec<[usize; 3]>,
    selection_name: &str,
    defeature_op: DefeatureOp,
    defeature_area_m2: f64,
) -> Result<ImportedSurface, CadImportError> {
    if triangles.is_empty() {
        return Err(CadImportError::InvalidGeometry("no triangles".to_owned()));
    }
    for point in &nodes {
        if !point.x.is_finite() || !point.y.is_finite() || !point.z.is_finite() {
            return Err(CadImportError::InvalidGeometry("non-finite vertex".to_owned()));
        }
    }
    let mut groups = PhysicalGroupRegistry::default();
    groups
        .register(2, 1, selection_name.to_owned())
        .map_err(|error| CadImportError::InvalidSelection(error.to_string()))?;
    let mut kept = Vec::with_capacity(triangles.len());
    let mut defeatured = Vec::new();
    for (index, triangle) in triangles.iter().enumerate() {
        let area = triangle_area(&nodes, triangle);
        if area <= defeature_area_m2 {
            defeatured.push(DefeaturedFeature { triangle: index, area_m2: area, op: defeature_op });
            if defeature_op == DefeatureOp::Remove {
                continue;
            }
        }
        kept.push(*triangle);
    }
    if kept.is_empty() {
        return Err(CadImportError::InvalidGeometry(
            "defeaturing removed every triangle".to_owned(),
        ));
    }
    let (manifold_edges, boundary_edges, nonmanifold_edges) = edge_counts(&kept);
    Ok(ImportedSurface {
        nodes,
        triangles: kept.clone(),
        groups,
        report: ImportReport {
            triangles_read: triangles.len(),
            triangles_kept: kept.len(),
            manifold_edges,
            boundary_edges,
            nonmanifold_edges,
            defeatured,
        },
    })
}

fn triangle_area(nodes: &[Point], triangle: &[usize; 3]) -> f64 {
    let a = nodes[triangle[0]];
    let b = nodes[triangle[1]];
    let c = nodes[triangle[2]];
    let ux = b.x - a.x;
    let uy = b.y - a.y;
    let uz = b.z - a.z;
    let vx = c.x - a.x;
    let vy = c.y - a.y;
    let vz = c.z - a.z;
    let cx = uy * vz - uz * vy;
    let cy = uz * vx - ux * vz;
    let cz = ux * vy - uy * vx;
    0.5 * (cx * cx + cy * cy + cz * cz).sqrt()
}

fn edge_counts(triangles: &[[usize; 3]]) -> (usize, usize, usize) {
    let mut counts: HashMap<(usize, usize), usize> = HashMap::new();
    for triangle in triangles {
        for edge in 0..3 {
            let a = triangle[edge];
            let b = triangle[(edge + 1) % 3];
            let key = if a < b { (a, b) } else { (b, a) };
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    let mut manifold = 0usize;
    let mut boundary = 0usize;
    let mut nonmanifold = 0usize;
    for count in counts.values() {
        match count {
            1 => boundary += 1,
            2 => manifold += 1,
            _ => nonmanifold += 1,
        }
    }
    (manifold, boundary, nonmanifold)
}

fn looks_like_obj(bytes: &[u8]) -> bool {
    let head = &bytes[..bytes.len().min(512)];
    let text = String::from_utf8_lossy(head);
    text.lines().any(|line| {
        let line = line.trim_start();
        line.starts_with("v ") || line.starts_with("f ")
    })
}

fn parse_obj(bytes: &[u8], scale: f64) -> Result<(Vec<Point>, Vec<[usize; 3]>), CadImportError> {
    let text = core::str::from_utf8(bytes)
        .map_err(|_| CadImportError::InvalidGeometry("OBJ payload is not UTF-8".to_owned()))?;
    let mut nodes: Vec<Point> = Vec::new();
    let mut triangles = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("v ") {
            let fields: Vec<&str> = rest.split_whitespace().collect();
            if fields.len() < 3 {
                return Err(CadImportError::InvalidGeometry("OBJ vertex needs 3 fields".to_owned()));
            }
            let mut coords = [0.0f64; 3];
            for (slot, field) in coords.iter_mut().zip(fields.iter()) {
                *slot = field
                    .parse::<f64>()
                    .map_err(|_| CadImportError::InvalidGeometry("bad OBJ vertex".to_owned()))?
                    * scale;
            }
            nodes.push(Point { x: coords[0], y: coords[1], z: coords[2] });
        } else if let Some(rest) = line.strip_prefix("f ") {
            let fields: Vec<&str> = rest.split_whitespace().collect();
            if fields.len() < 3 {
                return Err(CadImportError::InvalidGeometry("OBJ face needs 3 fields".to_owned()));
            }
            let mut indices = Vec::with_capacity(fields.len());
            for field in fields {
                let head = field.split('/').next().unwrap_or_default();
                let raw = head
                    .parse::<i64>()
                    .map_err(|_| CadImportError::InvalidGeometry("bad OBJ index".to_owned()))?;
                let index = if raw > 0 {
                    (raw - 1) as usize
                } else if raw < 0 {
                    (nodes.len() as i64 + raw) as usize
                } else {
                    return Err(CadImportError::InvalidGeometry("OBJ index 0".to_owned()));
                };
                if index >= nodes.len() {
                    return Err(CadImportError::InvalidGeometry("OBJ index out of range".to_owned()));
                }
                indices.push(index);
            }
            for fan in 1..indices.len() - 1 {
                triangles.push([indices[0], indices[fan], indices[fan + 1]]);
            }
        }
    }
    Ok((nodes, triangles))
}

fn parse_stl(bytes: &[u8], scale: f64) -> Result<(Vec<Point>, Vec<[usize; 3]>), CadImportError> {
    // Binary STL: 80-byte header, u32 count, then 50 bytes per facet.
    if bytes.len() >= 84 {
        let count = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
        if count > 0 && bytes.len() == 84 + count * 50 {
            let mut nodes = Vec::with_capacity(count * 3);
            let mut triangles = Vec::with_capacity(count);
            let mut weld: HashMap<(u64, u64, u64), usize> = HashMap::new();
            for facet in 0..count {
                let base = 84 + facet * 50 + 12;
                let mut tri = [0usize; 3];
                for (corner, slot) in tri.iter_mut().enumerate() {
                    let offset = base + corner * 12;
                    let mut coords = [0.0f64; 3];
                    for (axis, value) in coords.iter_mut().enumerate() {
                        let bits = u32::from_le_bytes([
                            bytes[offset + axis * 4],
                            bytes[offset + axis * 4 + 1],
                            bytes[offset + axis * 4 + 2],
                            bytes[offset + axis * 4 + 3],
                        ]);
                        *value = f64::from(f32::from_bits(bits)) * scale;
                    }
                    let key = (coords[0].to_bits(), coords[1].to_bits(), coords[2].to_bits());
                    let index = *weld.entry(key).or_insert_with(|| {
                        nodes.push(Point { x: coords[0], y: coords[1], z: coords[2] });
                        nodes.len() - 1
                    });
                    *slot = index;
                }
                triangles.push(tri);
            }
            return Ok((nodes, triangles));
        }
    }
    // ASCII STL fallback.
    let text = core::str::from_utf8(bytes).map_err(|_| CadImportError::UnrecognisedFormat)?;
    if !text.contains("facet") {
        return Err(CadImportError::UnrecognisedFormat);
    }
    let mut nodes: Vec<Point> = Vec::new();
    let mut weld: HashMap<(u64, u64, u64), usize> = HashMap::new();
    let mut triangles = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("vertex ") {
            let fields: Vec<&str> = rest.split_whitespace().collect();
            if fields.len() != 3 {
                return Err(CadImportError::InvalidGeometry("bad STL vertex".to_owned()));
            }
            let mut coords = [0.0f64; 3];
            for (slot, field) in coords.iter_mut().zip(fields.iter()) {
                *slot = field
                    .parse::<f64>()
                    .map_err(|_| CadImportError::InvalidGeometry("bad STL vertex".to_owned()))?
                    * scale;
            }
            let key = (coords[0].to_bits(), coords[1].to_bits(), coords[2].to_bits());
            let index = *weld.entry(key).or_insert_with(|| {
                nodes.push(Point { x: coords[0], y: coords[1], z: coords[2] });
                nodes.len() - 1
            });
            current.push(index);
            if current.len() == 3 {
                triangles.push([current[0], current[1], current[2]]);
                current.clear();
            }
        }
    }
    Ok((nodes, triangles))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tetra_obj() -> Vec<u8> {
        "v 0 0 0\nv 1000 0 0\nv 0 1000 0\nv 0 0 1000\nf 1 3 2\nf 1 2 4\nf 2 3 4\nf 3 1 4\n"
            .as_bytes()
            .to_vec()
    }

    #[test]
    fn obj_import_scales_and_is_watertight() {
        let surface = import_surface(&tetra_obj(), "basket", 0.001, DefeatureOp::Flag, 0.0)
            .expect("tetrahedron imports");
        assert_eq!(surface.report.triangles_kept, 4);
        assert!(surface.report.is_watertight());
        assert!(surface.groups.by_name(2, "basket").is_some());
        assert!((surface.nodes[1].x - 1.0).abs() < 1e-12);
    }

    #[test]
    fn defeature_remove_drops_small_triangles() {
        let bytes = "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 0\nv 0.001 0 0\nv 0 0.001 0\nf 1 2 3\nf 4 5 6\n"
            .as_bytes()
            .to_vec();
        let surface = import_surface(&bytes, "part", 1.0, DefeatureOp::Remove, 0.01)
            .expect("import succeeds");
        assert_eq!(surface.report.triangles_read, 2);
        assert_eq!(surface.report.triangles_kept, 1);
        assert_eq!(surface.report.defeatured.len(), 1);
    }

    #[test]
    fn invalid_scale_fails_closed() {
        assert!(matches!(
            import_surface(&tetra_obj(), "x", 0.0, DefeatureOp::Flag, 0.0),
            Err(CadImportError::InvalidUnitScale(_))
        ));
        assert!(matches!(
            import_surface(b"junk", "x", 1.0, DefeatureOp::Flag, 0.0),
            Err(CadImportError::UnrecognisedFormat)
        ));
    }
}
