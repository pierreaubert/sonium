//! Native constrained-Delaunay meshing of planar region sets.
//!
//! Motor cross-sections touch along whole edges (magnets in rotor
//! pockets, coils in slot voids), which defeats surface-with-holes
//! meshing in external tools. This module planarizes the region
//! contours (vertex snapping, segment splitting at vertices and
//! crossings), builds a Delaunay triangulation, recovers every contour
//! segment by edge flips (splitting a segment at its midpoint when its
//! crossed quads are all non-convex), classifies triangles by centroid,
//! and refines by longest-edge midpoint insertion against a size field.
//! The result is deterministic and needs no external mesher.

// Rust guideline compliant 2026-02-21

use std::collections::{HashMap, HashSet};

use crate::mesh::triangulate::triangulate_points_2d;
use crate::mesh::{ElementType, Mesh, Point};

/// One material region polygon, in metres.
#[derive(Debug, Clone, PartialEq)]
pub struct RegionPolygon {
    /// Mesh cell marker assigned to triangles inside the region.
    pub marker: i32,
    /// Outer contour (either orientation; closing repeat optional).
    pub exterior: Vec<(f64, f64)>,
    /// Hole contours.
    pub holes: Vec<Vec<(f64, f64)>>,
}

/// Controls for [`triangulate_regions`].
#[derive(Debug, Clone, Copy)]
pub struct PlanarMeshOptions {
    /// Coordinate quantization grid in metres (vertex welding).
    pub quantize_m: f64,
    /// Maximum snap distance onto a segment, in metres.
    pub snap_tolerance_m: f64,
    /// Hard cap on generated points.
    pub max_points: usize,
    /// Longest-edge refinement rounds.
    pub refinement_rounds: usize,
    /// Chord-error tolerance for contour decimation, in metres
    /// (0 disables). Exported contours are usually far denser than
    /// the size field (dense cocircular chains), which forces flat
    /// Delaunay triangles no refinement can cure; decimation
    /// removes mid-contour vertices whose shortcut stays within
    /// this tolerance of the original contour. Corners and
    /// junctions are never removed (see `decimate_segments`).
    pub decimate_tolerance_m: f64,
    /// Minimum-angle target of the Ruppert-style quality loop, in
    /// degrees (0 disables). After size refinement, circumcenters
    /// of worse triangles are inserted (encroached constraint
    /// segments split at midpoints instead) until the target is
    /// met, the point cap binds, or no insertion is accepted.
    pub quality_min_angle_deg: f64,
    /// Circumradius of a circle whose boundary segments are never
    /// split by decimation shortcuts or the quality loop (the
    /// sliding-band ring: splitting it would break the two-half
    /// weld and integer-shift rotation).
    pub protect_radius_m: Option<f64>,
}

impl Default for PlanarMeshOptions {
    fn default() -> Self {
        Self {
            quantize_m: 1.0e-7,
            snap_tolerance_m: 2.0e-6,
            max_points: 400_000,
            refinement_rounds: 24,
            decimate_tolerance_m: 3.0e-6,
            quality_min_angle_deg: 18.0,
            protect_radius_m: None,
        }
    }
}

/// Failure of planar region meshing.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum PlanarMeshError {
    /// A contour had fewer than three distinct points.
    #[error("a contour needs at least three distinct points")]
    DegenerateContour,
    /// A coordinate was non-finite.
    #[error("contour coordinates must be finite")]
    NonFiniteCoordinate,
    /// Constraint recovery exceeded its split budget.
    #[error("constraint recovery failed after {splits} segment splits")]
    RecoveryFailed {
        /// Segment splits attempted.
        splits: usize,
    },
    /// The point cap was reached before the size field was satisfied.
    #[error("point cap {max_points} reached during refinement")]
    PointCapReached {
        /// The configured cap.
        max_points: usize,
    },
}

type V = (f64, f64);

fn sub(a: V, b: V) -> V {
    (a.0 - b.0, a.1 - b.1)
}

fn cross(a: V, b: V) -> f64 {
    a.0 * b.1 - a.1 * b.0
}

fn orient(a: V, b: V, c: V) -> f64 {
    cross(sub(b, a), sub(c, a))
}

fn dist(a: V, b: V) -> f64 {
    (a.0 - b.0).hypot(a.1 - b.1)
}

/// Proper (interior-to-interior) segment intersection.
fn segments_cross(p1: V, p2: V, p3: V, p4: V) -> bool {
    let d1 = orient(p3, p4, p1);
    let d2 = orient(p3, p4, p2);
    let d3 = orient(p1, p2, p3);
    let d4 = orient(p1, p2, p4);
    ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
        && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
}

/// Intersection point of two segments known to cross properly.
fn segment_intersection(p1: V, p2: V, p3: V, p4: V) -> V {
    let r = sub(p2, p1);
    let s = sub(p4, p3);
    let denominator = cross(r, s);
    let t = cross(sub(p3, p1), s) / denominator;
    (p1.0 + t * r.0, p1.1 + t * r.1)
}

/// Even-odd point-in-polygon test; edges count as inside is false.
fn point_in_ring(point: V, ring: &[V]) -> bool {
    let mut inside = false;
    let n = ring.len();
    for i in 0..n {
        let (x1, y1) = ring[i];
        let (x2, y2) = ring[(i + 1) % n];
        if (y1 > point.1) != (y2 > point.1) {
            let x_intersect = x1 + (point.1 - y1) / (y2 - y1) * (x2 - x1);
            if point.0 < x_intersect {
                inside = !inside;
            }
        }
    }
    inside
}

fn point_in_region(
    point: V,
    region: &RegionPolygon,
    cleaned: &[(Vec<V>, Vec<Vec<V>>)],
    index: usize,
) -> bool {
    let _ = region;
    let (exterior, holes) = &cleaned[index];
    point_in_ring(point, exterior) && !holes.iter().any(|hole| point_in_ring(point, hole))
}

fn clean_ring(ring: &[V], quantize_m: f64) -> Result<Vec<V>, PlanarMeshError> {
    let mut out: Vec<V> = Vec::with_capacity(ring.len());
    for &p in ring {
        if !p.0.is_finite() || !p.1.is_finite() {
            return Err(PlanarMeshError::NonFiniteCoordinate);
        }
        let q = quantize_point(p, quantize_m);
        if out.last() != Some(&q) {
            out.push(q);
        }
    }
    if out.len() > 1 && out.first() == out.last() {
        out.pop();
    }
    if out.len() < 3 {
        return Err(PlanarMeshError::DegenerateContour);
    }
    Ok(out)
}

fn quantize_point(p: V, quantize_m: f64) -> V {
    (
        (p.0 / quantize_m).round() * quantize_m,
        (p.1 / quantize_m).round() * quantize_m,
    )
}

#[derive(Default)]
struct PointIndex {
    map: HashMap<(i64, i64), u32>,
    points: Vec<V>,
}

impl PointIndex {
    fn insert(&mut self, p: V, quantize_m: f64) -> u32 {
        let q = quantize_point(p, quantize_m);
        let key = (
            (q.0 / quantize_m).round() as i64,
            (q.1 / quantize_m).round() as i64,
        );
        *self.map.entry(key).or_insert_with(|| {
            self.points.push(q);
            (self.points.len() - 1) as u32
        })
    }
}

/// Splits segments at near-lying vertices and proper crossings.
///
/// A vertex within `snap_tolerance_m` of a segment interior becomes a
/// segment endpoint (the segment kinks by at most the snap distance).
/// Returns the split segment list over `index.points`.
fn planarize(
    index: &mut PointIndex,
    raw_segments: &[(u32, u32)],
    quantize_m: f64,
    snap_tolerance_m: f64,
) -> Vec<(u32, u32)> {
    let mut segments: Vec<(u32, u32)> = raw_segments.to_vec();
    for _round in 0..6 {
        let mut splits: HashMap<usize, Vec<(f64, u32)>> = HashMap::new();
        // Vertices on (or snapped onto) segment interiors.
        for (seg_index, &(a, b)) in segments.iter().enumerate() {
            let pa = index.points[a as usize];
            let pb = index.points[b as usize];
            let ab = sub(pb, pa);
            let len2 = ab.0 * ab.0 + ab.1 * ab.1;
            if len2 == 0.0 {
                continue;
            }
            for (p_index, &p) in index.points.iter().enumerate() {
                let p_index = p_index as u32;
                if p_index == a || p_index == b {
                    continue;
                }
                let t = (sub(p, pa).0 * ab.0 + sub(p, pa).1 * ab.1) / len2;
                if !(1.0e-9..1.0 - 1.0e-9).contains(&t) {
                    continue;
                }
                let projected = (pa.0 + t * ab.0, pa.1 + t * ab.1);
                if dist(p, projected) <= snap_tolerance_m {
                    splits.entry(seg_index).or_default().push((t, p_index));
                }
            }
        }
        // Proper crossings become new vertices.
        let mut crossing_points: Vec<(usize, usize, V)> = Vec::new();
        for (i, &(a, b)) in segments.iter().enumerate() {
            let (pa, pb) = (index.points[a as usize], index.points[b as usize]);
            for (j, &(c, d)) in segments.iter().enumerate().skip(i + 1) {
                if a == c || a == d || b == c || b == d {
                    continue;
                }
                let (pc, pd) = (index.points[c as usize], index.points[d as usize]);
                if pa.0.max(pb.0) < pc.0.min(pd.0)
                    || pc.0.max(pd.0) < pa.0.min(pb.0)
                    || pa.1.max(pb.1) < pc.1.min(pd.1)
                    || pc.1.max(pd.1) < pa.1.min(pb.1)
                {
                    continue;
                }
                if segments_cross(pa, pb, pc, pd) {
                    crossing_points.push((i, j, segment_intersection(pa, pb, pc, pd)));
                }
            }
        }
        if splits.is_empty() && crossing_points.is_empty() {
            break;
        }
        for (i, j, point) in crossing_points {
            let id = index.insert(point, quantize_m);
            for seg_index in [i, j] {
                let (a, b) = segments[seg_index];
                let (pa, pb) = (index.points[a as usize], index.points[b as usize]);
                let ab = sub(pb, pa);
                let len2 = ab.0 * ab.0 + ab.1 * ab.1;
                let t = if len2 > 0.0 {
                    (sub(point, pa).0 * ab.0 + sub(point, pa).1 * ab.1) / len2
                } else {
                    0.5
                };
                splits.entry(seg_index).or_default().push((t, id));
            }
        }
        let mut next: Vec<(u32, u32)> = Vec::with_capacity(segments.len() + splits.len());
        for (seg_index, &(a, b)) in segments.iter().enumerate() {
            if let Some(mut on_segment) = splits.remove(&seg_index) {
                on_segment.sort_by(|x, y| x.0.total_cmp(&y.0));
                on_segment.dedup_by_key(|entry| entry.1);
                let mut previous = a;
                for (_, id) in on_segment {
                    if id != previous && id != b {
                        next.push((previous, id));
                        previous = id;
                    }
                }
                next.push((previous, b));
            } else {
                next.push((a, b));
            }
        }
        segments = next;
    }
    let mut unique: Vec<(u32, u32)> = segments
        .into_iter()
        .filter(|(a, b)| a != b)
        .map(|(a, b)| if a <= b { (a, b) } else { (b, a) })
        .collect();
    unique.sort_unstable();
    unique.dedup();
    unique
}

/// Removes redundant mid-contour vertices from the segment set.
///
/// A vertex is removable when it has exactly two distinct segment
/// neighbours (so junctions, T-points and crossing points are
/// structurally preserved), both incident segments are shorter
/// than `0.55 * size(vertex)` (the chain is far denser than the
/// mesh wants to be there), and the shortcut chord stays within
/// `tolerance_m` of the vertex (so genuine corners, whose chord
/// error is a sizable fraction of their leg length, are kept).
/// Removals in one pass are pairwise non-adjacent; passes repeat
/// until stable. Dead points are compacted out of the index.
///
/// Why: Delaunay quality is bounded by the input point set. A
/// contour chain sampled at 90 µm feeding a 900 µm size field
/// forces flat boundary-layer triangles (measured: 96 % of
/// elements below 10° on a plain dense-ring disk), and no amount
/// of interior refinement repairs them, because refinement keeps
/// every input vertex. Decimation restores the spacing the size
/// field asks for, at a bounded geometric deviation.
fn decimate_segments(
    index: &mut PointIndex,
    segments: &[(u32, u32)],
    size_m: &dyn Fn(f64, f64) -> f64,
    tolerance_m: f64,
    quantize_m: f64,
    protect_radius_m: Option<f64>,
) -> Vec<(u32, u32)> {
    let mut segments: Vec<(u32, u32)> = segments.to_vec();
    let mut live: Vec<bool> = vec![true; index.points.len()];
    for _pass in 0..24 {
        // Distinct-neighbour sets per vertex over live segments.
        let mut neighbours: HashMap<u32, Vec<u32>> = HashMap::new();
        for &(a, b) in &segments {
            neighbours.entry(a).or_default().push(b);
            neighbours.entry(b).or_default().push(a);
        }
        let mut removed: HashSet<u32> = HashSet::new();
        let mut shortcuts: Vec<(u32, u32)> = Vec::new();
        // Index order, never hash order: removal is greedy with
        // neighbour blocking, so scan order decides which chain
        // vertices survive — the mesh must be reproducible.
        let mut scan: Vec<u32> = neighbours.keys().copied().collect();
        scan.sort_unstable();
        for v in scan {
            let adjacent = &neighbours[&v];
            if adjacent.len() != 2 || removed.contains(&v) {
                continue;
            }
            let (a, b) = (adjacent[0], adjacent[1]);
            if a == b || removed.contains(&a) || removed.contains(&b) {
                continue;
            }
            let pv = index.points[v as usize];
            let size = size_m(pv.0, pv.1);
            if size <= 0.0 {
                continue;
            }
            let pa = index.points[a as usize];
            let pb = index.points[b as usize];
            // Protected-circle (slip-ring) vertices are never
            // decimated: at coarse gap sizes the size test below
            // would otherwise eat the uniform ring.
            if segment_protected(pa, pv, protect_radius_m, quantize_m)
                || segment_protected(pv, pb, protect_radius_m, quantize_m)
            {
                continue;
            }
            if dist(pv, pa) >= 0.55 * size || dist(pv, pb) >= 0.55 * size {
                continue;
            }
            // Distance from v to the chord a-b.
            let ab = sub(pb, pa);
            let len_ab = dist(pa, pb);
            if len_ab <= 0.0 {
                continue;
            }
            let chord_error = cross(ab, sub(pv, pa)).abs() / len_ab;
            if chord_error > tolerance_m {
                continue;
            }
            removed.insert(v);
            live[v as usize] = false;
            shortcuts.push((a, b));
        }
        if removed.is_empty() {
            break;
        }
        segments.retain(|(a, b)| !removed.contains(a) && !removed.contains(b));
        segments.extend(shortcuts);
    }
    if live.iter().all(|&alive| alive) {
        return segments;
    }
    // Compact dead points out of the index; quantized coordinates
    // re-insert to identical keys, so welding is preserved.
    let mut new_index = PointIndex::default();
    let mut remap: HashMap<u32, u32> = HashMap::new();
    for (old, &point) in index.points.iter().enumerate() {
        if live[old] {
            let new_id = new_index.insert(point, quantize_m);
            remap.insert(old as u32, new_id);
        }
    }
    let segments: Vec<(u32, u32)> = segments
        .into_iter()
        .filter_map(|(a, b)| Some((*remap.get(&a)?, *remap.get(&b)?)))
        .collect();
    *index = new_index;
    segments
}

/// Circumcenter of a non-degenerate triangle.
fn circumcenter(a: V, b: V, c: V) -> Option<V> {
    let d = 2.0 * cross(sub(b, a), sub(c, a));
    if d.abs() <= 1.0e-300 {
        return None;
    }
    let a2 = a.0 * a.0 + a.1 * a.1;
    let b2 = b.0 * b.0 + b.1 * b.1;
    let c2 = c.0 * c.0 + c.1 * c.1;
    Some((
        ((b2 - a2) * (c.1 - a.1) - (c2 - a2) * (b.1 - a.1)) / d,
        ((c2 - a2) * (b.0 - a.0) - (b2 - a2) * (c.0 - a.0)) / d,
    ))
}

/// Smallest angle of a triangle, in degrees, and the vertex index
/// (0..3 into the argument order) where it sits.
fn min_angle_deg(a: V, b: V, c: V) -> (f64, usize) {
    let angle = |p: V, q: V, r: V| -> f64 {
        let u = sub(q, p);
        let v = sub(r, p);
        cross(u, v).abs().atan2(u.0 * v.0 + u.1 * v.1).to_degrees()
    };
    let angles = [angle(a, b, c), angle(b, c, a), angle(c, a, b)];
    let mut worst = 0usize;
    for i in 1..3 {
        if angles[i] < angles[worst] {
            worst = i;
        }
    }
    (angles[worst], worst)
}

/// Vertices where incident constraint segments meet at < 60°.
///
/// Quality insertion cannot fix a bad triangle whose small angle
/// sits at such a corner (Ruppert's classical non-termination
/// case); those triangles are exempt from the quality target.
fn sharp_vertices(segments: &[(u32, u32)], points: &[V]) -> HashSet<u32> {
    let mut directions: HashMap<u32, Vec<V>> = HashMap::new();
    for &(a, b) in segments {
        directions
            .entry(a)
            .or_default()
            .push(sub(points[b as usize], points[a as usize]));
        directions
            .entry(b)
            .or_default()
            .push(sub(points[a as usize], points[b as usize]));
    }
    let mut sharp = HashSet::new();
    for (vertex, dirs) in directions {
        let mut is_sharp = false;
        for i in 0..dirs.len() {
            for j in (i + 1)..dirs.len() {
                let (u, v) = (dirs[i], dirs[j]);
                let angle = cross(u, v).abs().atan2(u.0 * v.0 + u.1 * v.1).to_degrees();
                if angle < 60.0 {
                    is_sharp = true;
                }
            }
        }
        if is_sharp {
            sharp.insert(vertex);
        }
    }
    sharp
}

/// Whether a segment lies on the protected circle (both endpoints
/// within quantization tolerance of its radius).
fn segment_protected(a: V, b: V, protect_radius_m: Option<f64>, quantize_m: f64) -> bool {
    match protect_radius_m {
        Some(radius) => {
            (a.0.hypot(a.1) - radius).abs() <= 4.0 * quantize_m
                && (b.0.hypot(b.1) - radius).abs() <= 4.0 * quantize_m
        }
        None => false,
    }
}

fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a <= b { (a, b) } else { (b, a) }
}

struct Triangulation {
    points: Vec<V>,
    tris: Vec<[u32; 3]>,
    edge_tris: HashMap<(u32, u32), Vec<u32>>,
    vertex_tris: HashMap<u32, Vec<u32>>,
}

impl Triangulation {
    fn new(points: Vec<V>) -> Self {
        let raw = triangulate_points_2d(&points);
        let tris: Vec<[u32; 3]> = raw
            .into_iter()
            .map(|tri| [tri[0] as u32, tri[1] as u32, tri[2] as u32])
            .collect();
        let mut mesh = Self {
            points,
            tris,
            edge_tris: HashMap::new(),
            vertex_tris: HashMap::new(),
        };
        mesh.rebuild_maps();
        mesh
    }

    fn rebuild_maps(&mut self) {
        self.edge_tris.clear();
        self.vertex_tris.clear();
        for (tri_index, tri) in self.tris.iter().enumerate() {
            for vertex in tri {
                self.vertex_tris
                    .entry(*vertex)
                    .or_default()
                    .push(tri_index as u32);
            }
            for k in 0..3 {
                self.edge_tris
                    .entry(edge_key(tri[k], tri[(k + 1) % 3]))
                    .or_default()
                    .push(tri_index as u32);
            }
        }
    }

    fn has_edge(&self, a: u32, b: u32) -> bool {
        self.edge_tris.contains_key(&edge_key(a, b))
    }

    fn opposite(&self, tri_index: u32, edge: (u32, u32)) -> Option<u32> {
        self.tris[tri_index as usize]
            .into_iter()
            .find(|vertex| *vertex != edge.0 && *vertex != edge.1)
    }

    /// Ordered mesh edges crossed by segment `a-b`, found by walking.
    fn crossed_edges(&self, a: u32, b: u32) -> Option<Vec<(u32, u32)>> {
        let pa = self.points[a as usize];
        let pb = self.points[b as usize];
        // Start triangle: incident to `a`, exited across the opposite edge.
        let incident = self.vertex_tris.get(&a)?;
        let mut current: Option<u32> = None;
        let mut entry: Option<(u32, u32)> = None;
        for &tri_index in incident {
            let tri = self.tris[tri_index as usize];
            let others: Vec<u32> = tri.into_iter().filter(|v| *v != a).collect();
            if others.len() != 2 {
                continue;
            }
            if others.contains(&b) {
                return Some(Vec::new());
            }
            let (p, q) = (
                self.points[others[0] as usize],
                self.points[others[1] as usize],
            );
            if segments_cross(pa, pb, p, q) {
                current = Some(tri_index);
                entry = None;
                break;
            }
        }
        let mut current = current?;
        let mut crossed = Vec::new();
        for _step in 0..self.tris.len() + 1 {
            let tri = self.tris[current as usize];
            if tri.contains(&b) {
                return Some(crossed);
            }
            let mut advanced = false;
            for k in 0..3 {
                let edge = edge_key(tri[k], tri[(k + 1) % 3]);
                if Some(edge) == entry || edge.0 == a || edge.1 == a {
                    continue;
                }
                let (p, q) = (self.points[edge.0 as usize], self.points[edge.1 as usize]);
                if segments_cross(pa, pb, p, q) {
                    crossed.push(edge);
                    let neighbours = self.edge_tris.get(&edge)?;
                    let next = neighbours.iter().find(|t| **t != current)?;
                    entry = Some(edge);
                    current = *next;
                    advanced = true;
                    break;
                }
            }
            if !advanced {
                return None;
            }
        }
        None
    }

    /// Flips interior `edge`; returns false when its quad is not convex.
    fn flip(&mut self, edge: (u32, u32)) -> bool {
        let Some(owners) = self.edge_tris.get(&edge).cloned() else {
            return false;
        };
        if owners.len() != 2 {
            return false;
        }
        let (Some(p), Some(q)) = (
            self.opposite(owners[0], edge),
            self.opposite(owners[1], edge),
        ) else {
            return false;
        };
        let (pp, pq) = (self.points[p as usize], self.points[q as usize]);
        let (pu, pv) = (self.points[edge.0 as usize], self.points[edge.1 as usize]);
        if !segments_cross(pu, pv, pp, pq) {
            return false;
        }
        let orient_tri = |x: u32, y: u32, z: u32, points: &[V]| -> [u32; 3] {
            if orient(points[x as usize], points[y as usize], points[z as usize]) > 0.0 {
                [x, y, z]
            } else {
                [x, z, y]
            }
        };
        self.tris[owners[0] as usize] = orient_tri(edge.0, p, q, &self.points);
        self.tris[owners[1] as usize] = orient_tri(edge.1, q, p, &self.points);
        // Local map repair. The diagonal changes, two outer edges swap
        // owners, and the vertex incidence of the quad corners changes.
        let (u, v) = edge;
        self.edge_tris.remove(&edge);
        self.edge_tris.insert(edge_key(p, q), owners.clone());
        replace_owner(&mut self.edge_tris, edge_key(v, p), owners[0], owners[1]);
        replace_owner(&mut self.edge_tris, edge_key(u, q), owners[1], owners[0]);
        remove_incidence(&mut self.vertex_tris, v, owners[0]);
        remove_incidence(&mut self.vertex_tris, u, owners[1]);
        self.vertex_tris.entry(q).or_default().push(owners[0]);
        self.vertex_tris.entry(p).or_default().push(owners[1]);
        true
    }
}

fn replace_owner(map: &mut HashMap<(u32, u32), Vec<u32>>, edge: (u32, u32), old: u32, new: u32) {
    if let Some(owners) = map.get_mut(&edge) {
        for owner in owners.iter_mut() {
            if *owner == old {
                *owner = new;
            }
        }
    }
}

fn remove_incidence(map: &mut HashMap<u32, Vec<u32>>, vertex: u32, tri: u32) {
    if let Some(list) = map.get_mut(&vertex) {
        list.retain(|entry| *entry != tri);
    }
}

/// Recovers all constraint segments by flips, splitting when stuck.
///
/// On a split the triangulation restarts from the Delaunay of the
/// enlarged point set, with the split segment replaced by its halves.
fn constrained_triangulation(
    index: &mut PointIndex,
    segments: &mut Vec<(u32, u32)>,
    quantize_m: f64,
) -> Result<Triangulation, PlanarMeshError> {
    let mut splits = 0usize;
    'restart: loop {
        let mut mesh = Triangulation::new(index.points.clone());
        let mut recovered: HashSet<(u32, u32)> = HashSet::new();
        let work: Vec<(u32, u32)> = segments.clone();
        let mut cursor = 0;
        while cursor < work.len() {
            let (a, b) = work[cursor];
            cursor += 1;
            if mesh.has_edge(a, b) {
                recovered.insert(edge_key(a, b));
                continue;
            }
            let mut guard = 0usize;
            loop {
                guard += 1;
                if guard > mesh.tris.len() + 16 {
                    break;
                }
                if mesh.has_edge(a, b) {
                    break;
                }
                let Some(crossed) = mesh.crossed_edges(a, b) else {
                    break;
                };
                if crossed.is_empty() {
                    break;
                }
                let mut flipped = false;
                for edge in crossed {
                    if recovered.contains(&edge) {
                        continue;
                    }
                    // Never create an edge that destroys a recovered one:
                    // the flipped quad's new diagonal must not be forced
                    // to cross a recovered constraint. The convexity test
                    // inside flip plus the recovered-edge skip covers the
                    // cases this mesher commits to.
                    if mesh.flip(edge) {
                        flipped = true;
                        break;
                    }
                }
                if !flipped {
                    break;
                }
            }
            if mesh.has_edge(a, b) {
                recovered.insert(edge_key(a, b));
                continue;
            }
            // Stuck: split the segment at its midpoint and restart.
            splits += 1;
            if splits > 2 * segments.len() + 64 {
                return Err(PlanarMeshError::RecoveryFailed { splits });
            }
            let pa = index.points[a as usize];
            let pb = index.points[b as usize];
            let mid = index.insert(((pa.0 + pb.0) / 2.0, (pa.1 + pb.1) / 2.0), quantize_m);
            if mid == a || mid == b {
                return Err(PlanarMeshError::RecoveryFailed { splits });
            }
            segments.retain(|s| edge_key(s.0, s.1) != edge_key(a, b));
            segments.push(edge_key(a, mid));
            segments.push(edge_key(mid, b));
            continue 'restart;
        }
        return Ok(mesh);
    }
}

/// Meshes a cover polygon subdivided into priority-ordered regions.
///
/// `cover` bounds the meshed area (its marker is the fallback).
/// `regions` are tested in order at each triangle centroid; the first
/// containing region wins, so list embedded parts (coils, magnets)
/// before the bodies containing them. `size_m(x, y)` is the target edge
/// length used by longest-edge refinement.
///
/// # Errors
///
/// Returns [`PlanarMeshError`] for degenerate contours, non-finite
/// coordinates, failed constraint recovery, or an exhausted point cap.
pub fn triangulate_regions(
    cover: &RegionPolygon,
    regions: &[RegionPolygon],
    size_m: &dyn Fn(f64, f64) -> f64,
    options: &PlanarMeshOptions,
) -> Result<Mesh, PlanarMeshError> {
    let quantize = options.quantize_m;
    let clean = |region: &RegionPolygon| -> Result<(Vec<V>, Vec<Vec<V>>), PlanarMeshError> {
        let exterior = clean_ring(&region.exterior, quantize)?;
        let holes = region
            .holes
            .iter()
            .map(|hole| clean_ring(hole, quantize))
            .collect::<Result<Vec<_>, _>>()?;
        Ok((exterior, holes))
    };
    let cover_clean = clean(cover)?;
    let mut cleaned: Vec<(Vec<V>, Vec<Vec<V>>)> = Vec::with_capacity(regions.len());
    for region in regions {
        cleaned.push(clean(region)?);
    }

    let mut index = PointIndex::default();
    let mut raw_segments: Vec<(u32, u32)> = Vec::new();
    let add_ring = |ring: &[V], index: &mut PointIndex, out: &mut Vec<(u32, u32)>| {
        let ids: Vec<u32> = ring.iter().map(|&p| index.insert(p, quantize)).collect();
        for k in 0..ids.len() {
            out.push((ids[k], ids[(k + 1) % ids.len()]));
        }
    };
    add_ring(&cover_clean.0, &mut index, &mut raw_segments);
    for hole in &cover_clean.1 {
        add_ring(hole, &mut index, &mut raw_segments);
    }
    for (exterior, holes) in &cleaned {
        add_ring(exterior, &mut index, &mut raw_segments);
        for hole in holes {
            add_ring(hole, &mut index, &mut raw_segments);
        }
    }
    let mut segments = planarize(
        &mut index,
        &raw_segments,
        quantize,
        options.snap_tolerance_m,
    );
    if options.decimate_tolerance_m > 0.0 {
        segments = decimate_segments(
            &mut index,
            &segments,
            size_m,
            options.decimate_tolerance_m,
            quantize,
            options.protect_radius_m,
        );
        // Shortcuts can in principle cross a neighbouring contour
        // (bounded by the chord tolerance, but cheap to recheck):
        // run the planarizer again so crossings/splits are exact.
        segments = planarize(&mut index, &segments, quantize, options.snap_tolerance_m);
    }
    let mut tri = constrained_triangulation(&mut index, &mut segments, quantize)?;

    // Region bounding boxes keep centroid classification near-linear.
    let bbox = |ring: &[V]| -> (V, V) {
        ring.iter().fold(
            (
                (f64::INFINITY, f64::INFINITY),
                (f64::NEG_INFINITY, f64::NEG_INFINITY),
            ),
            |(lo, hi), p| {
                (
                    (lo.0.min(p.0), lo.1.min(p.1)),
                    (hi.0.max(p.0), hi.1.max(p.1)),
                )
            },
        )
    };
    let region_boxes: Vec<(V, V)> = cleaned.iter().map(|(ext, _)| bbox(ext)).collect();

    // Longest-edge refinement against the size field. New points are
    // mesh-edge midpoints; constraint segments hit by a midpoint are
    // split there before the next constrained rebuild.
    let classify = |tri: &Triangulation| -> Vec<Option<i32>> {
        tri.tris
            .iter()
            .map(|t| {
                let centroid = (
                    (tri.points[t[0] as usize].0
                        + tri.points[t[1] as usize].0
                        + tri.points[t[2] as usize].0)
                        / 3.0,
                    (tri.points[t[0] as usize].1
                        + tri.points[t[1] as usize].1
                        + tri.points[t[2] as usize].1)
                        / 3.0,
                );
                let in_cover = point_in_ring(centroid, &cover_clean.0)
                    && !cover_clean
                        .1
                        .iter()
                        .any(|hole| point_in_ring(centroid, hole));
                if !in_cover {
                    return None;
                }
                for (region_index, region) in regions.iter().enumerate() {
                    let (lo, hi) = region_boxes[region_index];
                    if centroid.0 < lo.0
                        || centroid.0 > hi.0
                        || centroid.1 < lo.1
                        || centroid.1 > hi.1
                    {
                        continue;
                    }
                    if point_in_region(centroid, region, &cleaned, region_index) {
                        return Some(region.marker);
                    }
                }
                Some(cover.marker)
            })
            .collect()
    };
    for _round in 0..options.refinement_rounds {
        let markers = classify(&tri);
        let mut added_points = 0usize;
        let mut split_edges: Vec<(u32, u32)> = Vec::new();
        for (tri_index, t) in tri.tris.iter().enumerate() {
            if markers[tri_index].is_none() {
                continue;
            }
            for k in 0..3 {
                let (u, v) = (t[k], t[(k + 1) % 3]);
                let (pu, pv) = (tri.points[u as usize], tri.points[v as usize]);
                let mid = ((pu.0 + pv.0) / 2.0, (pu.1 + pv.1) / 2.0);
                let target = size_m(mid.0, mid.1);
                if target > 0.0 && dist(pu, pv) > 1.6 * target {
                    split_edges.push(edge_key(u, v));
                }
            }
        }
        if split_edges.is_empty() {
            break;
        }
        split_edges.sort_unstable();
        split_edges.dedup();
        if index.points.len() + split_edges.len() > options.max_points {
            return Err(PlanarMeshError::PointCapReached {
                max_points: options.max_points,
            });
        }
        let constraint_set: HashSet<(u32, u32)> =
            segments.iter().map(|s| edge_key(s.0, s.1)).collect();
        for (u, v) in &split_edges {
            let (pu, pv) = (tri.points[*u as usize], tri.points[*v as usize]);
            let before = index.points.len();
            let mid = index.insert(((pu.0 + pv.0) / 2.0, (pu.1 + pv.1) / 2.0), quantize);
            added_points += index.points.len() - before;
            if constraint_set.contains(&edge_key(*u, *v)) && mid != *u && mid != *v {
                segments.retain(|s| edge_key(s.0, s.1) != edge_key(*u, *v));
                segments.push(edge_key(*u, mid));
                segments.push(edge_key(mid, *v));
            }
        }
        if added_points == 0 {
            break;
        }
        tri = constrained_triangulation(&mut index, &mut segments, quantize)?;
    }

    // Ruppert-style quality loop: circumcenters of triangles below
    // the minimum-angle target become Steiner points; a candidate
    // that encroaches a constraint segment (lies in its diametral
    // circle) splits that segment at its midpoint instead.
    // Protected (slip-ring) segments are never split, and
    // candidates encroaching one are dropped. Bad triangles at
    // sharp input corners are exempt — insertion cannot fix them
    // and would not terminate. Each accepted round rebuilds the
    // constrained triangulation, like the size loop above.
    if options.quality_min_angle_deg > 0.0 {
        let sharp = sharp_vertices(&segments, &index.points);
        for _round in 0..64 {
            let markers_now = classify(&tri);
            let mut bad: Vec<(usize, f64)> = Vec::new();
            for (tri_index, t) in tri.tris.iter().enumerate() {
                if markers_now[tri_index].is_none() {
                    continue;
                }
                let (angle, worst) = min_angle_deg(
                    tri.points[t[0] as usize],
                    tri.points[t[1] as usize],
                    tri.points[t[2] as usize],
                );
                if angle < options.quality_min_angle_deg && !sharp.contains(&t[worst]) {
                    bad.push((tri_index, angle));
                }
            }
            if bad.is_empty() {
                break;
            }
            bad.sort_by(|x, y| x.1.total_cmp(&y.1));
            let mut accepted: Vec<V> = Vec::new();
            let mut split_segments: Vec<(u32, u32)> = Vec::new();
            for &(tri_index, _) in bad.iter().take(4096) {
                let t = tri.tris[tri_index];
                let (pa, pb, pc) = (
                    tri.points[t[0] as usize],
                    tri.points[t[1] as usize],
                    tri.points[t[2] as usize],
                );
                let Some(centre) = circumcenter(pa, pb, pc) else {
                    continue;
                };
                let in_cover = point_in_ring(centre, &cover_clean.0)
                    && !cover_clean.1.iter().any(|hole| point_in_ring(centre, hole));
                if !in_cover {
                    continue;
                }
                let radius = dist(centre, pa);
                // Encroachment: centre inside a segment's diametral
                // circle. Only segments within reach can qualify.
                let mut encroached: Vec<(u32, u32)> = Vec::new();
                let mut hits_protected = false;
                for &(a, b) in &segments {
                    let (sa, sb) = (index.points[a as usize], index.points[b as usize]);
                    let mid = ((sa.0 + sb.0) / 2.0, (sa.1 + sb.1) / 2.0);
                    if dist(centre, mid) > dist(sa, sb) / 2.0 + radius {
                        continue;
                    }
                    let dot = sub(sa, centre).0 * sub(sb, centre).0
                        + sub(sa, centre).1 * sub(sb, centre).1;
                    if dot <= 0.0 {
                        if segment_protected(sa, sb, options.protect_radius_m, quantize) {
                            hits_protected = true;
                            break;
                        }
                        encroached.push(edge_key(a, b));
                    }
                }
                if hits_protected {
                    continue;
                }
                if !encroached.is_empty() {
                    split_segments.extend(encroached);
                    continue;
                }
                // Spacing guard against this round's accepted set:
                // two circumcenters closer than half the smaller
                // circumradius would manufacture a new sliver.
                if accepted
                    .iter()
                    .any(|&other| dist(centre, other) < 0.5 * radius)
                {
                    continue;
                }
                accepted.push(centre);
            }
            split_segments.sort_unstable();
            split_segments.dedup();
            let new_points = accepted.len() + split_segments.len();
            if new_points == 0 || index.points.len() + new_points > options.max_points {
                break;
            }
            for point in &accepted {
                index.insert(*point, quantize);
            }
            if !split_segments.is_empty() {
                let constraint_set: HashSet<(u32, u32)> =
                    segments.iter().map(|s| edge_key(s.0, s.1)).collect();
                for &(u, v) in &split_segments {
                    if !constraint_set.contains(&edge_key(u, v)) {
                        continue;
                    }
                    let (pu, pv) = (index.points[u as usize], index.points[v as usize]);
                    let mid = index.insert(((pu.0 + pv.0) / 2.0, (pu.1 + pv.1) / 2.0), quantize);
                    if mid != u && mid != v {
                        segments.retain(|s| edge_key(s.0, s.1) != edge_key(u, v));
                        segments.push(edge_key(u, mid));
                        segments.push(edge_key(mid, v));
                    }
                }
            }
            tri = constrained_triangulation(&mut index, &mut segments, quantize)?;
        }
    }

    let markers = classify(&tri);
    // Compact to used points in first-use order.
    let mut remap: HashMap<u32, usize> = HashMap::new();
    let mut nodes: Vec<V> = Vec::new();
    let mut elements: Vec<([usize; 3], i32)> = Vec::new();
    for (tri_index, t) in tri.tris.iter().enumerate() {
        let Some(marker) = markers[tri_index] else {
            continue;
        };
        let mut ids = [0usize; 3];
        for (slot, vertex) in t.iter().enumerate() {
            let next = nodes.len();
            let id = *remap.entry(*vertex).or_insert_with(|| {
                nodes.push(tri.points[*vertex as usize]);
                next
            });
            ids[slot] = id;
        }
        if orient(nodes[ids[0]], nodes[ids[1]], nodes[ids[2]]) < 0.0 {
            ids.swap(1, 2);
        }
        elements.push((ids, marker));
    }
    let mut mesh = Mesh::new(2);
    for &(x, y) in &nodes {
        mesh.add_node(Point::new_2d(x, y));
    }
    for (ids, marker) in elements {
        mesh.add_element_with_marker(ElementType::Triangle, ids.to_vec(), marker);
    }
    mesh.detect_boundaries();
    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn marker_areas(mesh: &Mesh) -> HashMap<i32, f64> {
        let mut areas: HashMap<i32, f64> = HashMap::new();
        for element in &mesh.elements {
            let a = mesh.nodes[element.nodes[0]];
            let b = mesh.nodes[element.nodes[1]];
            let c = mesh.nodes[element.nodes[2]];
            let area = 0.5 * ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)).abs();
            *areas.entry(element.marker).or_default() += area;
        }
        areas
    }

    fn rect(marker: i32, x0: f64, y0: f64, x1: f64, y1: f64) -> RegionPolygon {
        RegionPolygon {
            marker,
            exterior: vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
            holes: Vec::new(),
        }
    }

    #[test]
    fn touching_rectangles_mesh_conformally_with_exact_areas() {
        // Two regions touching along a full edge plus a T-junction:
        // left square, right square split horizontally at y = 0.5.
        let cover = rect(0, 0.0, 0.0, 2.0, 1.0);
        let regions = [
            rect(1, 0.0, 0.0, 1.0, 1.0),
            rect(2, 1.0, 0.0, 2.0, 0.5),
            rect(3, 1.0, 0.5, 2.0, 1.0),
        ];
        let mesh =
            triangulate_regions(&cover, &regions, &|_, _| 0.2, &PlanarMeshOptions::default())
                .expect("touching rectangles mesh");
        let areas = marker_areas(&mesh);
        assert!((areas[&1] - 1.0).abs() < 1.0e-9, "{areas:?}");
        assert!((areas[&2] - 0.5).abs() < 1.0e-9);
        assert!((areas[&3] - 0.5).abs() < 1.0e-9);
        // Conformal: every interior edge has exactly two owners.
        let mut edge_owners: HashMap<(usize, usize), usize> = HashMap::new();
        for element in &mesh.elements {
            for k in 0..3 {
                let (u, v) = (element.nodes[k], element.nodes[(k + 1) % 3]);
                let key = if u <= v { (u, v) } else { (v, u) };
                *edge_owners.entry(key).or_default() += 1;
            }
        }
        assert!(edge_owners.values().all(|count| *count <= 2));
        assert!(mesh.num_nodes() > 20, "refinement happened");
    }

    #[test]
    fn annulus_with_embedded_rectangles_keeps_ring_vertices() {
        let ring: Vec<(f64, f64)> = (0..84)
            .map(|k| {
                let a = 2.0 * std::f64::consts::PI * k as f64 / 84.0;
                (0.01 * a.cos(), 0.01 * a.sin())
            })
            .collect();
        let outer: Vec<(f64, f64)> = (0..84)
            .map(|k| {
                let a = 2.0 * std::f64::consts::PI * k as f64 / 84.0;
                (0.02 * a.cos(), 0.02 * a.sin())
            })
            .collect();
        let cover = RegionPolygon {
            marker: 0,
            exterior: outer,
            holes: vec![ring],
        };
        let regions = [rect(5, -0.002, 0.012, 0.002, 0.018)];
        let mesh = triangulate_regions(
            &cover,
            &regions,
            &|x, y| {
                let r = x.hypot(y);
                if (r - 0.01).abs() < 8.0e-4 {
                    2.0e-4
                } else {
                    1.5e-3
                }
            },
            &PlanarMeshOptions::default(),
        )
        .expect("annulus mesh");
        let ring_nodes = mesh
            .nodes
            .iter()
            .filter(|node| (node.x.hypot(node.y) - 0.01).abs() < 2.0e-7)
            .count();
        assert!(ring_nodes >= 84, "ring nodes {ring_nodes}");
        let areas = marker_areas(&mesh);
        assert!((areas[&5] - 0.004 * 0.006).abs() < 1.0e-9, "{areas:?}");
    }
}
