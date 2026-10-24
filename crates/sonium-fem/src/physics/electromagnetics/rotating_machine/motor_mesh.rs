//! Two-half sliding-band motor mesh built on the native planar mesher.
//!
//! The stator half (iron, coils, gap air down to the mid-gap ring,
//! outer air) and the rotor half (iron, magnets, shaft, gap air up to
//! the ring) are meshed independently with the same uniform ring
//! discretization, then welded at the ring. Rotor rotation by an
//! integer number of ring pitches is exact: rotor node coordinates
//! rotate rigidly and rotor elements re-reference ring nodes shifted
//! by that integer, so no element ever deforms and no interpolation
//! error enters the sliding interface.

// Rust guideline compliant 2026-02-21

use std::collections::HashMap;

use crate::mesh::{ElementType, Mesh, Point};

use super::planar_mesh::{PlanarMeshError, PlanarMeshOptions, RegionPolygon, triangulate_regions};

/// Stator-iron marker.
pub const MARKER_STATOR: i32 = 1;
/// Rotor-iron marker.
pub const MARKER_ROTOR: i32 = 2;
/// Shaft marker.
pub const MARKER_SHAFT: i32 = 3;
/// Air marker (gap, slot voids, outer region).
pub const MARKER_AIR: i32 = 4;
/// Coil marker base: wire `w` (in [`MotorPolygons::coils`] order)
/// uses `MARKER_COIL_BASE + w`; its slot is `MotorMesh::coil_slot_of_wire[w]`.
pub const MARKER_COIL_BASE: i32 = 100;
/// Magnet marker base: magnet `m` uses `MARKER_MAGNET_BASE + m`.
pub const MARKER_MAGNET_BASE: i32 = 200;

/// Region polygons of one motor cross-section, in metres.
///
/// Markers on the incoming polygons are ignored; the builder assigns
/// the `MARKER_*` convention above.
#[derive(Debug, Clone)]
pub struct MotorPolygons {
    /// Stator iron (exterior plus bore/slot hole).
    pub stator: RegionPolygon,
    /// Rotor iron (exterior plus shaft bore hole).
    pub rotor: RegionPolygon,
    /// Shaft (exterior; an optional bore hole is filled as shaft).
    pub shaft: RegionPolygon,
    /// Magnet polygons in magnet order.
    pub magnets: Vec<RegionPolygon>,
    /// Coil wire polygons with their slot index.
    pub coils: Vec<(usize, RegionPolygon)>,
    /// Mid-gap (slip) radius in metres.
    pub mid_gap_radius_m: f64,
    /// Outer air boundary radius in metres.
    pub outer_radius_m: f64,
}

/// Mesh-size controls for [`build_motor_mesh`], all in metres.
#[derive(Debug, Clone, Copy)]
pub struct MotorMeshOptions {
    /// Uniform ring segments; must divide the transient step plan.
    pub n_ring: usize,
    /// Outer-boundary polygon segments.
    pub n_outer: usize,
    /// Target edge length in the air-gap band.
    pub gap_size_m: f64,
    /// Target edge length in iron and coils.
    pub iron_size_m: f64,
    /// Target edge length far from the machine.
    pub outer_size_m: f64,
    /// Radial half-width of the fine gap band around the slip radius.
    pub gap_band_half_width_m: f64,
    /// Planar mesher controls.
    pub planar: PlanarMeshOptions,
}

impl Default for MotorMeshOptions {
    fn default() -> Self {
        Self {
            n_ring: 840,
            n_outer: 180,
            gap_size_m: 8.0e-5,
            iron_size_m: 9.0e-4,
            outer_size_m: 2.0e-3,
            gap_band_half_width_m: 3.0e-4,
            planar: PlanarMeshOptions::default(),
        }
    }
}

/// One ring reference inside a rotor element (for shift remapping).
#[derive(Debug, Clone, Copy)]
struct RingRef {
    element: usize,
    local: usize,
    ring_position: usize,
}

/// Welded two-half motor mesh with integer-shift rotor rotation.
#[derive(Debug, Clone)]
pub struct MotorMesh {
    /// Combined mesh at the current rotor shift.
    pub mesh: Mesh,
    /// Ring DOF node ids in counter-clockwise angle order.
    pub ring_nodes: Vec<usize>,
    /// Node ids rotating rigidly with the rotor (excluding ring nodes).
    pub rotor_nodes: Vec<usize>,
    /// Combined-mesh indices of rotor-half elements.
    pub rotor_elements: Vec<usize>,
    /// Mid-gap radius in metres.
    pub mid_gap_radius_m: f64,
    /// Slot index of each coil wire, in wire-marker order.
    pub coil_slot_of_wire: Vec<usize>,
    base_nodes: Vec<Point>,
    ring_refs: Vec<RingRef>,
    ring_pitches: usize,
    current_shift: i64,
}

fn circle(radius_m: f64, segments: usize) -> Vec<(f64, f64)> {
    (0..segments)
        .map(|k| {
            let angle = 2.0 * std::f64::consts::PI * k as f64 / segments as f64;
            (radius_m * angle.cos(), radius_m * angle.sin())
        })
        .collect()
}

fn with_marker(polygon: &RegionPolygon, marker: i32) -> RegionPolygon {
    RegionPolygon {
        marker,
        exterior: polygon.exterior.clone(),
        holes: polygon.holes.clone(),
    }
}

/// Builds the welded P1 motor mesh at rotor shift zero.
///
/// # Errors
///
/// Propagates [`PlanarMeshError`] from either half, and returns
/// [`PlanarMeshError::DegenerateContour`] when a radius or ring count
/// is invalid or the ring weld finds unmatched ring nodes.
pub fn build_motor_mesh(
    polygons: &MotorPolygons,
    options: &MotorMeshOptions,
) -> Result<MotorMesh, PlanarMeshError> {
    if options.n_ring < 12
        || polygons.mid_gap_radius_m <= 0.0
        || polygons.outer_radius_m <= polygons.mid_gap_radius_m
    {
        return Err(PlanarMeshError::DegenerateContour);
    }
    // Refinement must never split ring segments, or the ring stops
    // being uniform and integer-shift rotation loses exactness.
    let ring_segment_m =
        2.0 * std::f64::consts::PI * polygons.mid_gap_radius_m / options.n_ring as f64;
    if options.gap_size_m * 1.6 <= ring_segment_m {
        return Err(PlanarMeshError::DegenerateContour);
    }
    let r_mid = polygons.mid_gap_radius_m;
    // The slip ring must never be split by decimation or quality
    // refinement in either half: both halves weld on its vertices.
    let mut planar = options.planar;
    planar.protect_radius_m = Some(r_mid);
    let ring = circle(r_mid, options.n_ring);
    let size = |x: f64, y: f64| -> f64 {
        let r = x.hypot(y);
        if (r - r_mid).abs() <= options.gap_band_half_width_m {
            options.gap_size_m
        } else if r <= polygons.outer_radius_m * 0.85 {
            options.iron_size_m
        } else {
            options.outer_size_m
        }
    };

    // Stator half: annulus cover (outer disc minus ring disc).
    let stator_cover = RegionPolygon {
        marker: MARKER_AIR,
        exterior: circle(polygons.outer_radius_m, options.n_outer),
        holes: vec![ring.clone()],
    };
    let mut stator_regions: Vec<RegionPolygon> = polygons
        .coils
        .iter()
        .enumerate()
        .map(|(wire, (_slot, coil))| with_marker(coil, MARKER_COIL_BASE + wire as i32))
        .collect();
    stator_regions.push(with_marker(&polygons.stator, MARKER_STATOR));
    let stator_mesh = triangulate_regions(&stator_cover, &stator_regions, &size, &planar)?;

    // Rotor half: ring disc cover.
    let rotor_cover = RegionPolygon {
        marker: MARKER_AIR,
        exterior: ring,
        holes: Vec::new(),
    };
    let mut rotor_regions: Vec<RegionPolygon> = polygons
        .magnets
        .iter()
        .enumerate()
        .map(|(index, magnet)| with_marker(magnet, MARKER_MAGNET_BASE + index as i32))
        .collect();
    rotor_regions.push(with_marker(&polygons.rotor, MARKER_ROTOR));
    rotor_regions.push(with_marker(&polygons.shaft, MARKER_SHAFT));
    for hole in &polygons.shaft.holes {
        // Fill shaft bore holes as shaft material (solid-shaft model).
        rotor_regions.push(RegionPolygon {
            marker: MARKER_SHAFT,
            exterior: hole.clone(),
            holes: Vec::new(),
        });
    }
    let rotor_mesh = triangulate_regions(&rotor_cover, &rotor_regions, &size, &planar)?;

    let mut motor = weld_halves(stator_mesh, rotor_mesh, r_mid, options)?;
    motor.coil_slot_of_wire = polygons.coils.iter().map(|(slot, _)| *slot).collect();
    Ok(motor)
}

fn ring_key(point: &Point, quantize_m: f64) -> (i64, i64) {
    (
        (point.x / quantize_m).round() as i64,
        (point.y / quantize_m).round() as i64,
    )
}

fn weld_halves(
    stator: Mesh,
    rotor: Mesh,
    r_mid: f64,
    options: &MotorMeshOptions,
) -> Result<MotorMesh, PlanarMeshError> {
    let quantize = options.planar.quantize_m;
    let on_ring = |point: &Point| (point.x.hypot(point.y) - r_mid).abs() <= 4.0 * quantize;
    let mut combined = Mesh::new(2);
    for node in &stator.nodes {
        combined.add_node(*node);
    }
    let mut ring_by_key: HashMap<(i64, i64), usize> = HashMap::new();
    let mut ring_nodes: Vec<usize> = Vec::new();
    for (index, node) in stator.nodes.iter().enumerate() {
        if on_ring(node) {
            ring_by_key.insert(ring_key(node, quantize), index);
            ring_nodes.push(index);
        }
    }
    for element in &stator.elements {
        combined.add_element_with_marker(
            ElementType::Triangle,
            element.nodes.clone(),
            element.marker,
        );
    }
    // Rotor nodes: ring nodes weld to stator counterparts.
    let mut rotor_map: Vec<usize> = Vec::with_capacity(rotor.nodes.len());
    let mut rotor_nodes: Vec<usize> = Vec::new();
    for node in &rotor.nodes {
        if on_ring(node) {
            let Some(&welded) = ring_by_key.get(&ring_key(node, quantize)) else {
                return Err(PlanarMeshError::DegenerateContour);
            };
            rotor_map.push(welded);
        } else {
            let id = combined.add_node(*node);
            rotor_map.push(id);
            rotor_nodes.push(id);
        }
    }
    let mut rotor_elements: Vec<usize> = Vec::with_capacity(rotor.elements.len());
    for element in &rotor.elements {
        let nodes: Vec<usize> = element.nodes.iter().map(|n| rotor_map[*n]).collect();
        let id = combined.add_element_with_marker(ElementType::Triangle, nodes, element.marker);
        rotor_elements.push(id);
    }
    combined.detect_boundaries();

    ring_nodes.sort_by(|&a, &b| {
        let pa = combined.nodes[a];
        let pb = combined.nodes[b];
        pa.y.atan2(pa.x)
            .partial_cmp(&pb.y.atan2(pb.x))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let ring_position: HashMap<usize, usize> = ring_nodes
        .iter()
        .enumerate()
        .map(|(position, node)| (*node, position))
        .collect();
    let mut ring_refs = Vec::new();
    for &element_index in &rotor_elements {
        for (local, node) in combined.elements[element_index].nodes.iter().enumerate() {
            if let Some(&position) = ring_position.get(node) {
                ring_refs.push(RingRef {
                    element: element_index,
                    local,
                    ring_position: position,
                });
            }
        }
    }
    Ok(MotorMesh {
        base_nodes: combined.nodes.clone(),
        mesh: combined,
        ring_nodes,
        rotor_nodes,
        rotor_elements,
        mid_gap_radius_m: r_mid,
        coil_slot_of_wire: Vec::new(),
        ring_refs,
        ring_pitches: options.n_ring,
        current_shift: 0,
    })
}

impl MotorMesh {
    /// Current rotor shift in ring pitches.
    pub fn rotor_shift(&self) -> i64 {
        self.current_shift
    }

    /// Rotor mechanical angle for `shift` ring pitches, in degrees.
    pub fn shift_angle_deg(&self, shift: i64) -> f64 {
        shift as f64 * 360.0 / self.ring_pitches as f64
    }

    /// Number of ring DOFs per ring pitch (1 for P1, 2 after P2 promotion).
    fn dofs_per_pitch(&self) -> usize {
        self.ring_nodes.len() / self.ring_pitches.max(1)
    }

    /// Rotates the rotor by `shift` ring pitches (exact, rigid).
    ///
    /// Rotor node coordinates rotate from their base positions and
    /// rotor elements re-reference ring nodes shifted by the same
    /// angle. Callers must choose step plans whose mechanical step is
    /// an integer number of pitches.
    pub fn set_rotor_shift(&mut self, shift: i64) {
        let angle = shift as f64 * 2.0 * std::f64::consts::PI / self.ring_pitches as f64;
        let (cos_a, sin_a) = (angle.cos(), angle.sin());
        for &node in &self.rotor_nodes {
            let base = self.base_nodes[node];
            self.mesh.nodes[node].x = base.x * cos_a - base.y * sin_a;
            self.mesh.nodes[node].y = base.x * sin_a + base.y * cos_a;
        }
        let count = self.ring_nodes.len() as i64;
        let dof_shift = shift * self.dofs_per_pitch() as i64;
        for reference in &self.ring_refs {
            let position = (reference.ring_position as i64 + dof_shift).rem_euclid(count);
            self.mesh.elements[reference.element].nodes[reference.local] =
                self.ring_nodes[position as usize];
        }
        self.current_shift = shift;
    }

    /// Promotes the welded mesh to P2 and rebuilds the ring tables.
    ///
    /// Must be called at shift zero. After promotion the ring carries
    /// `2 * n_ring` DOFs (vertices plus edge midpoints) and
    /// [`Self::set_rotor_shift`] keeps working in ring pitches.
    ///
    /// # Errors
    ///
    /// Returns [`PlanarMeshError::DegenerateContour`] when called at a
    /// nonzero shift or when the promoted ring count is unexpected.
    pub fn promote_p2(&mut self, quantize_m: f64) -> Result<(), PlanarMeshError> {
        if self.current_shift != 0 {
            return Err(PlanarMeshError::DegenerateContour);
        }
        let promoted = self.mesh.to_p2();
        if promoted.elements.len() != self.mesh.elements.len() {
            return Err(PlanarMeshError::DegenerateContour);
        }
        let r_mid = self.mid_gap_radius_m;
        let on_ring = |point: &Point| (point.x.hypot(point.y) - r_mid).abs() <= 4.0 * quantize_m;
        let mut ring_nodes: Vec<usize> = promoted
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| on_ring(node))
            .map(|(index, _)| index)
            .collect();
        if ring_nodes.len() != 2 * self.ring_pitches {
            return Err(PlanarMeshError::DegenerateContour);
        }
        ring_nodes.sort_by(|&a, &b| {
            let pa = promoted.nodes[a];
            let pb = promoted.nodes[b];
            pa.y.atan2(pa.x)
                .partial_cmp(&pb.y.atan2(pb.x))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let ring_set: std::collections::HashSet<usize> = ring_nodes.iter().copied().collect();
        let rotor_element_set: std::collections::HashSet<usize> =
            self.rotor_elements.iter().copied().collect();
        let mut in_stator: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut in_rotor: Vec<usize> = Vec::new();
        let mut ring_refs: Vec<RingRef> = Vec::new();
        let ring_position: HashMap<usize, usize> = ring_nodes
            .iter()
            .enumerate()
            .map(|(position, node)| (*node, position))
            .collect();
        for (element_index, element) in promoted.elements.iter().enumerate() {
            if rotor_element_set.contains(&element_index) {
                for (local, node) in element.nodes.iter().enumerate() {
                    if let Some(&position) = ring_position.get(node) {
                        ring_refs.push(RingRef {
                            element: element_index,
                            local,
                            ring_position: position,
                        });
                    } else {
                        in_rotor.push(*node);
                    }
                }
            } else {
                in_stator.extend(element.nodes.iter().copied());
            }
        }
        in_rotor.sort_unstable();
        in_rotor.dedup();
        let rotor_nodes: Vec<usize> = in_rotor
            .into_iter()
            .filter(|node| !in_stator.contains(node) && !ring_set.contains(node))
            .collect();
        self.base_nodes = promoted.nodes.clone();
        self.mesh = promoted;
        self.ring_nodes = ring_nodes;
        self.rotor_nodes = rotor_nodes;
        self.ring_refs = ring_refs;
        Ok(())
    }
}
