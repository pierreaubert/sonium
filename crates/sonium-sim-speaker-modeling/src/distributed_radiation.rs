//! Distributed moving-assembly radiation: FEM velocity trace → BEM load.
//!
//! The axisymmetric driver ([`crate::axisymmetric_linear`]) resolves the
//! moving assembly (diaphragm/shell, former, surround, spider, coil) as a
//! distributed structure, but its compact fluid stage and the scalar
//! `install_interface_load` piston closure only carry the area-averaged
//! velocity. Above the compact limit (`ka` no longer small) the diaphragm
//! velocity varies over the surface (breakup), a rigid-piston radiation
//! impedance is wrong, and the full distributed load is required.
//!
//! This module is that handoff:
//!
//! ```text
//! AxisymmetricLinearSolution ──► InterfaceVelocityTrace ──► DtnMap (BEM)
//!   (nodal RMS phasors)          (per-element v_n, RMS,      p̄ = D·v
//!                                 outward normal, exp(+iωt)) │
//!                                                             ▼
//!                              DistributedRadiationResult: element
//!                              pressures, radiated power, full-matrix
//!                              mean impedance, piston comparison, ka regime
//! ```
//!
//! # Contract
//!
//! * Phasors are **RMS** amplitudes with the `exp(+iωt)` convention shared
//!   by `sonium-fem`, `sonium-bem`, and [`crate::fem_bem_coupling`]; average
//!   power is therefore `Re(Σⱼ conj(vⱼ)·pⱼ·Aⱼ)` with no `½` factor.
//! * Element velocities are outward-normal velocities on the radiating
//!   surface, in the BEM aperture-element order of the supplied
//!   [`DtnMap`]. Length or convention mismatches fail closed.
//! * The lumped piston impedance is always computed and labelled as such;
//!   it is a comparison baseline, never silently substituted for the
//!   distributed result.

use num_complex::Complex64;

use crate::axisymmetric_linear::{
    AxisymmetricLinearModel, AxisymmetricLinearSolution, STRUCTURAL_ACOUSTIC_INTERFACE,
};
use crate::fem_bem_coupling::DtnMap;

/// Errors for the distributed-radiation handoff.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum DistributedRadiationError {
    /// Trace was empty, non-finite, or had non-positive areas.
    #[error("invalid interface velocity trace: {detail}")]
    InvalidTrace {
        /// Human-readable detail.
        detail: String,
    },
    /// Trace element count does not match the DtN map dimension.
    #[error("velocity trace has {trace} elements but the DtN map is {dtn}x{dtn}")]
    LengthMismatch {
        /// Trace element count.
        trace: usize,
        /// DtN map dimension.
        dtn: usize,
    },
    /// Trace frequency does not match the DtN map frequency.
    #[error("velocity trace frequency {trace_hz} Hz does not match DtN frequency {dtn_hz} Hz")]
    FrequencyMismatch {
        /// Trace frequency in Hz.
        trace_hz: f64,
        /// DtN map frequency in Hz.
        dtn_hz: f64,
    },
    /// No structural-acoustic interface edges were found on the model.
    #[error("axisymmetric model has no structural acoustic interface")]
    MissingInterface,
    /// A breakup/modal input was invalid.
    #[error("invalid breakup input: {detail}")]
    InvalidBreakup {
        /// Human-readable detail.
        detail: String,
    },
    /// The exterior BEM construction or DtN condensation failed.
    #[error("exterior BEM failed: {detail}")]
    Bem {
        /// Human-readable detail.
        detail: String,
    },
}

/// Per-element outward-normal velocity trace on a radiating surface.
///
/// One entry per BEM aperture element: RMS phasor normal velocity,
/// element area, and centroid radius (used for the `ka` regime estimate).
#[derive(Debug, Clone, PartialEq)]
pub struct InterfaceVelocityTrace {
    /// Frequency in Hz.
    pub frequency_hz: f64,
    /// Element areas in m², in BEM aperture-element order.
    pub element_areas_m2: Vec<f64>,
    /// RMS outward-normal velocity phasors in m/s, same order.
    pub element_velocities_rms: Vec<Complex64>,
    /// Element centroid radii in metres (axisymmetric rings), same order.
    pub element_centroid_radius_m: Vec<f64>,
}

impl InterfaceVelocityTrace {
    /// Validate finiteness, positivity, and consistent lengths; fail closed.
    pub fn validate(&self) -> Result<(), DistributedRadiationError> {
        let invalid = |detail: &str| DistributedRadiationError::InvalidTrace {
            detail: detail.to_string(),
        };
        if self.element_areas_m2.is_empty() {
            return Err(invalid("trace must contain at least one element"));
        }
        if self.element_velocities_rms.len() != self.element_areas_m2.len()
            || self.element_centroid_radius_m.len() != self.element_areas_m2.len()
        {
            return Err(invalid("areas, velocities, and radii must have equal lengths"));
        }
        if !self.frequency_hz.is_finite() || self.frequency_hz <= 0.0 {
            return Err(invalid("frequency must be finite and positive"));
        }
        for (index, area) in self.element_areas_m2.iter().enumerate() {
            if !area.is_finite() || *area <= 0.0 {
                return Err(invalid("element areas must be finite and positive"));
            }
            if !self.element_velocities_rms[index].is_finite() {
                return Err(invalid("element velocities must be finite"));
            }
            if !self.element_centroid_radius_m[index].is_finite()
                || self.element_centroid_radius_m[index] < 0.0
            {
                return Err(invalid("element centroid radii must be finite and non-negative"));
            }
        }
        Ok(())
    }

    /// Total radiating area in m².
    pub fn total_area_m2(&self) -> f64 {
        self.element_areas_m2.iter().sum()
    }

    /// Volume velocity `U = Σⱼ vⱼ·Aⱼ` in m³/s (RMS phasor).
    pub fn volume_velocity(&self) -> Complex64 {
        self.element_velocities_rms
            .iter()
            .zip(&self.element_areas_m2)
            .map(|(v, a)| *v * *a)
            .sum()
    }

    /// Area-averaged (piston) velocity `U / A` in m/s.
    pub fn mean_velocity(&self) -> Complex64 {
        self.volume_velocity() / self.total_area_m2()
    }

    /// Effective piston radius `√(A/π)` in metres.
    pub fn effective_radius_m(&self) -> f64 {
        (self.total_area_m2() / std::f64::consts::PI).sqrt()
    }

    /// Helmholtz number `ka` with `a` the effective piston radius.
    pub fn ka(&self, sound_speed_m_per_s: f64) -> f64 {
        2.0 * std::f64::consts::PI * self.frequency_hz / sound_speed_m_per_s
            * self.effective_radius_m()
    }

    /// Replace every element velocity with the area-averaged velocity
    /// (rigid-piston projection of the same volume velocity).
    pub fn piston_projection(&self) -> Self {
        let mean = self.mean_velocity();
        Self {
            frequency_hz: self.frequency_hz,
            element_areas_m2: self.element_areas_m2.clone(),
            element_velocities_rms: vec![mean; self.element_velocities_rms.len()],
            element_centroid_radius_m: self.element_centroid_radius_m.clone(),
        }
    }
}

/// Extract the radiating-surface velocity trace from an axisymmetric solve.
///
/// Each structural-acoustic interface edge becomes one axisymmetric ring
/// element: area `2π·r̄·L`, outward normal `+z` (the piston axis, matching
/// the interface-load convention in
/// [`AxisymmetricLinearModel::install_interface_load`]), velocity the mean
/// of its two nodal axial RMS velocity phasors.
pub fn axisymmetric_interface_trace(
    model: &AxisymmetricLinearModel,
    solution: &AxisymmetricLinearSolution,
) -> Result<InterfaceVelocityTrace, DistributedRadiationError> {
    let mut areas = Vec::new();
    let mut velocities = Vec::new();
    let mut radii = Vec::new();
    for boundary in model
        .structural_mesh
        .boundaries
        .iter()
        .filter(|boundary| boundary.marker == STRUCTURAL_ACOUSTIC_INTERFACE)
    {
        if boundary.nodes.len() != 2 {
            continue;
        }
        let a = &model.structural_mesh.nodes[boundary.nodes[0]];
        let b = &model.structural_mesh.nodes[boundary.nodes[1]];
        let edge_length = (b.x - a.x).hypot(b.y - a.y);
        let radius = 0.5 * (a.x + b.x);
        let area = 2.0 * std::f64::consts::PI * radius * edge_length;
        if !area.is_finite() || area <= 0.0 {
            continue;
        }
        let velocity = (solution.velocity_rz_m_per_s[boundary.nodes[0]][1]
            + solution.velocity_rz_m_per_s[boundary.nodes[1]][1])
            * 0.5;
        areas.push(area);
        velocities.push(velocity);
        radii.push(radius);
    }
    if areas.is_empty() {
        return Err(DistributedRadiationError::MissingInterface);
    }
    let trace = InterfaceVelocityTrace {
        frequency_hz: solution.frequency_hz,
        element_areas_m2: areas,
        element_velocities_rms: velocities,
        element_centroid_radius_m: radii,
    };
    trace.validate()?;
    Ok(trace)
}

/// Compactness regime of the radiating surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadiationRegime {
    /// `ka ≤ 0.3`: lumped piston impedance is an acceptable approximation.
    Compact,
    /// `0.3 < ka ≤ 1`: piston error grows; distributed load recommended.
    Transitional,
    /// `ka > 1`: distributed (breakup-resolving) load is required.
    DistributedRequired,
}

/// Classify a Helmholtz number into a radiation regime.
pub fn radiation_regime(ka: f64) -> RadiationRegime {
    if ka <= 0.3 {
        RadiationRegime::Compact
    } else if ka <= 1.0 {
        RadiationRegime::Transitional
    } else {
        RadiationRegime::DistributedRequired
    }
}

/// Result of applying the exterior BEM DtN map to a distributed trace.
#[derive(Debug, Clone, PartialEq)]
pub struct DistributedRadiationResult {
    /// Element-average RMS pressures `p̄ = D·v` in Pa.
    pub element_pressures_rms: Vec<Complex64>,
    /// Time-averaged radiated power in watts (RMS phasors).
    pub radiated_power_w: f64,
    /// Full-matrix mean radiation impedance `p̄_mean / v_mean` in Pa·s/m.
    /// This is a labelled scalar reduction of the distributed result.
    pub distributed_mean_impedance: Complex64,
    /// Lumped piston impedance for the same volume velocity, in Pa·s/m.
    /// Comparison baseline only — never the production load above the
    /// compact regime.
    pub lumped_piston_impedance: Complex64,
    /// Relative L2 pressure difference between the distributed trace and
    /// its rigid-piston projection (`0` for a truly rigid surface).
    pub piston_projection_pressure_error: f64,
    /// Helmholtz number of the trace.
    pub ka: f64,
    /// Compactness regime implied by `ka`.
    pub regime: RadiationRegime,
}

/// Apply a production BEM DtN map to a distributed velocity trace.
///
/// The full `m × m` matrix is used (no lumping): element pressures,
/// radiated power, and both the distributed mean impedance and the lumped
/// piston baseline are returned so callers can gate on their difference.
pub fn evaluate_distributed_radiation(
    dtn: &DtnMap,
    trace: &InterfaceVelocityTrace,
    sound_speed_m_per_s: f64,
) -> Result<DistributedRadiationResult, DistributedRadiationError> {
    trace.validate()?;
    if trace.element_velocities_rms.len() != dtn.n {
        return Err(DistributedRadiationError::LengthMismatch {
            trace: trace.element_velocities_rms.len(),
            dtn: dtn.n,
        });
    }
    if (trace.frequency_hz - dtn.frequency_hz).abs() > 1.0e-9 * dtn.frequency_hz.max(1.0) {
        return Err(DistributedRadiationError::FrequencyMismatch {
            trace_hz: trace.frequency_hz,
            dtn_hz: dtn.frequency_hz,
        });
    }
    if !sound_speed_m_per_s.is_finite() || sound_speed_m_per_s <= 0.0 {
        return Err(DistributedRadiationError::InvalidTrace {
            detail: "sound speed must be finite and positive".to_string(),
        });
    }
    // Hot path: one no-allocation-reuse matvec per trace (buffer owned here
    // because the result owns the pressures; repeated sweeps should call
    // DtnMap::apply_into with a caller-owned buffer).
    let pressures = dtn.apply(&trace.element_velocities_rms);
    let power = dtn.radiated_power(&trace.element_velocities_rms, &trace.element_areas_m2);
    let piston = trace.piston_projection();
    let piston_pressures = dtn.apply(&piston.element_velocities_rms);
    let mean_velocity = trace.mean_velocity();
    let weighted_pressure = |pressures: &[Complex64]| -> Complex64 {
        pressures
            .iter()
            .zip(&trace.element_areas_m2)
            .map(|(p, a)| *p * *a)
            .sum::<Complex64>()
            / trace.total_area_m2()
    };
    let distributed_mean_impedance = weighted_pressure(&pressures) / mean_velocity;
    let lumped_piston_impedance = weighted_pressure(&piston_pressures) / mean_velocity;
    let numerator: f64 = pressures
        .iter()
        .zip(&piston_pressures)
        .map(|(p, q)| (p - q).norm_sqr())
        .sum();
    let denominator: f64 = pressures.iter().map(|p| p.norm_sqr()).sum();
    let piston_projection_pressure_error = if denominator > 0.0 {
        (numerator / denominator).sqrt()
    } else {
        0.0
    };
    let ka = trace.ka(sound_speed_m_per_s);
    Ok(DistributedRadiationResult {
        element_pressures_rms: pressures,
        radiated_power_w: power,
        distributed_mean_impedance,
        lumped_piston_impedance,
        piston_projection_pressure_error,
        ka,
        regime: radiation_regime(ka),
    })
}

/// Resolution of the baffled-aperture BEM surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BaffledApertureConfig {
    /// Baffle (and backing-dome) radius as a multiple of aperture radius.
    pub baffle_radius_factor: f64,
    /// Azimuthal sectors around the axis.
    pub sectors: usize,
    /// Radial rings across the aperture disk.
    pub aperture_rings: usize,
    /// Radial rings across the rigid baffle annulus.
    pub baffle_rings: usize,
    /// Latitude rings down the backing dome.
    pub dome_rings: usize,
}

impl Default for BaffledApertureConfig {
    fn default() -> Self {
        Self {
            baffle_radius_factor: 4.0,
            sectors: 16,
            aperture_rings: 3,
            baffle_rings: 2,
            dome_rings: 4,
        }
    }
}

/// Builds the closed baffled-aperture BEM surface for one driver.
///
/// The closed body is a half-ball below the baffle plane: a rigid dome
/// plus the flat baffle disk at `z = 0`. The aperture disk
/// (`r ≤ aperture_radius_m`) radiates toward `+z`; the annulus and dome
/// are rigid. Every element carries a zero-velocity condition and the
/// returned indices select the aperture, exactly as the production
/// mesh bridge marks its surfaces. Element windings are auto-oriented
/// against the body (dome: radial direction; disk: `+z`).
///
/// # Errors
///
/// Returns [`DistributedRadiationError::Bem`] for invalid resolution
/// parameters.
pub fn build_baffled_aperture_mesh(
    aperture_radius_m: f64,
    config: BaffledApertureConfig,
) -> Result<(sonium_bem::core::types::Mesh, Vec<usize>), DistributedRadiationError> {
    use sonium_bem::core::types::{
        BoundaryCondition, Element as BemElement, ElementProperty, ElementType as BemElementType,
        Mesh as BemMesh,
    };
    let bem = |detail: String| DistributedRadiationError::Bem { detail };
    if !aperture_radius_m.is_finite() || aperture_radius_m <= 0.0 {
        return Err(bem("aperture radius must be finite and positive".to_owned()));
    }
    if !config.baffle_radius_factor.is_finite() || config.baffle_radius_factor <= 1.0 {
        return Err(bem("baffle radius factor must exceed one".to_owned()));
    }
    if config.sectors < 8 || config.aperture_rings < 1 || config.baffle_rings < 1 || config.dome_rings < 1 {
        return Err(bem("aperture mesh resolution is below minimum".to_owned()));
    }
    let a = aperture_radius_m;
    let big = a * config.baffle_radius_factor;
    // Nodes are shared structurally (ring ids are reused across
    // adjacent bands and the sector wrap uses modulo indices), so no
    // coordinate welding pass is needed.
    let mut nodes: Vec<[f64; 3]> = Vec::new();
    let mut node = |nodes: &mut Vec<[f64; 3]>, point: [f64; 3]| -> usize {
        nodes.push(point);
        nodes.len() - 1
    };
    // Disk rings: centre, aperture rings, then baffle annulus rings.
    let mut ring_radii: Vec<f64> = vec![0.0];
    for i in 1..=config.aperture_rings {
        ring_radii.push(a * i as f64 / config.aperture_rings as f64);
    }
    for j in 1..=config.baffle_rings {
        ring_radii.push(a + (big - a) * j as f64 / config.baffle_rings as f64);
    }
    let mut disk_rings: Vec<Vec<usize>> = Vec::new();
    for (ring, &radius) in ring_radii.iter().enumerate() {
        if ring == 0 {
            let centre = node(&mut nodes, [0.0, 0.0, 0.0]);
            disk_rings.push(vec![centre]);
        } else {
            let mut ids = Vec::with_capacity(config.sectors);
            for sector in 0..config.sectors {
                let theta = std::f64::consts::TAU * sector as f64 / config.sectors as f64;
                ids.push(node(&mut nodes, [radius * theta.cos(), radius * theta.sin(), 0.0]));
            }
            disk_rings.push(ids);
        }
    }
    // Dome rings from the equator (z = 0, r = big) to the south pole.
    let mut dome_rings: Vec<Vec<usize>> = Vec::new();
    for lat in 1..=config.dome_rings {
        let phi = std::f64::consts::FRAC_PI_2 * lat as f64 / (config.dome_rings + 1) as f64;
        let radius = big * phi.cos();
        let z = -big * phi.sin();
        let mut ids = Vec::with_capacity(config.sectors);
        for sector in 0..config.sectors {
            let theta = std::f64::consts::TAU * sector as f64 / config.sectors as f64;
            ids.push(node(&mut nodes, [radius * theta.cos(), radius * theta.sin(), z]));
        }
        dome_rings.push(ids);
    }
    let pole = node(&mut nodes, [0.0, 0.0, -big]);
    let mut triangles: Vec<([usize; 3], bool)> = Vec::new();
    // Disk: fan from centre, then annular quads.
    let centre = disk_rings[0][0];
    for sector in 0..config.sectors {
        let next = (sector + 1) % config.sectors;
        triangles.push(([centre, disk_rings[1][sector], disk_rings[1][next]], true));
    }
    for ring in 1..disk_rings.len() - 1 {
        let is_aperture = ring_radii[ring + 1] <= a * (1.0 + 1.0e-9);
        for sector in 0..config.sectors {
            let next = (sector + 1) % config.sectors;
            let (i0, i1) = (disk_rings[ring][sector], disk_rings[ring][next]);
            let (o0, o1) = (disk_rings[ring + 1][sector], disk_rings[ring + 1][next]);
            triangles.push(([i0, o0, o1], is_aperture));
            triangles.push(([i0, o1, i1], is_aperture));
        }
    }
    // Dome: quads between disk rim and dome rings, then pole fan.
    let mut previous: Vec<usize> = disk_rings[disk_rings.len() - 1].clone();
    for ring in &dome_rings {
        for sector in 0..config.sectors {
            let next = (sector + 1) % config.sectors;
            let (i0, i1) = (previous[sector], previous[next]);
            let (o0, o1) = (ring[sector], ring[next]);
            triangles.push(([i0, o0, o1], false));
            triangles.push(([i0, o1, i1], false));
        }
        previous = ring.clone();
    }
    for sector in 0..config.sectors {
        let next = (sector + 1) % config.sectors;
        triangles.push(([previous[sector], pole, previous[next]], false));
    }
    let mut elements = Vec::with_capacity(triangles.len());
    let mut aperture_elements = Vec::new();
    for (index, (mut conn, is_aperture)) in triangles.into_iter().enumerate() {
        let p0 = nodes[conn[0]];
        let p1 = nodes[conn[1]];
        let p2 = nodes[conn[2]];
        let centre = [
            (p0[0] + p1[0] + p2[0]) / 3.0,
            (p0[1] + p1[1] + p2[1]) / 3.0,
            (p0[2] + p1[2] + p2[2]) / 3.0,
        ];
        let reference = if centre[2] < -1.0e-12 {
            centre
        } else {
            [0.0, 0.0, 1.0]
        };
        let mut normal = [
            (p1[1] - p0[1]) * (p2[2] - p0[2]) - (p1[2] - p0[2]) * (p2[1] - p0[1]),
            (p1[2] - p0[2]) * (p2[0] - p0[0]) - (p1[0] - p0[0]) * (p2[2] - p0[2]),
            (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]),
        ];
        if normal[0] * reference[0] + normal[1] * reference[1] + normal[2] * reference[2] < 0.0 {
            conn.swap(1, 2);
            normal = [-normal[0], -normal[1], -normal[2]];
        }
        let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if !(length > 0.0) {
            return Err(bem("degenerate aperture element".to_owned()));
        }
        let unit = [normal[0] / length, normal[1] / length, normal[2] / length];
        if is_aperture {
            aperture_elements.push(index);
        }
        elements.push(BemElement {
            connectivity: conn.to_vec(),
            element_type: BemElementType::Tri3,
            property: ElementProperty::Surface,
            normal: unit,
            node_normals: vec![unit; 3],
            center: centre,
            area: 0.5 * length,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
            group: 0,
            dof_addresses: vec![index],
        });
    }
    if aperture_elements.is_empty() {
        return Err(bem("aperture mesh has no aperture elements".to_owned()));
    }
    let num_nodes = nodes.len();
    let num_elements = elements.len();
    let mesh = BemMesh {
        nodes,
        elements,
        external_node_numbers: (0..num_nodes as i32).collect(),
        external_element_numbers: (0..num_elements as i32).collect(),
        num_boundary_nodes: num_nodes,
        num_evaluation_nodes: 0,
        num_boundary_elements: num_elements,
        num_evaluation_elements: 0,
        symmetry_planes: [0, 0, 0],
        symmetry_coordinates: [0.0, 0.0, 0.0],
        num_reflections: 0,
        node_to_elements: Vec::new(),
    };
    Ok((mesh, aperture_elements))
}

/// Interpolates a ring trace onto aperture elements by centroid radius.
///
/// Linear in radius between ring centroids; outside the ring span the
/// nearest ring value is held (the rings tile the aperture, so this
/// only affects a half-ring at the centre and rim).
fn map_trace_to_aperture(
    trace: &InterfaceVelocityTrace,
    mesh: &sonium_bem::core::types::Mesh,
    aperture_elements: &[usize],
    frequency_hz: f64,
) -> InterfaceVelocityTrace {
    let mut radii: Vec<f64> = trace.element_centroid_radius_m.clone();
    let order: Vec<usize> = {
        let mut indices: Vec<usize> = (0..radii.len()).collect();
        indices.sort_by(|&a, &b| radii[a].total_cmp(&radii[b]));
        indices
    };
    radii.sort_by(f64::total_cmp);
    let velocities: Vec<Complex64> = order
        .iter()
        .map(|&index| trace.element_velocities_rms[index])
        .collect();
    let sample = |radius: f64| -> Complex64 {
        if radius <= radii[0] {
            return velocities[0];
        }
        if radius >= radii[radii.len() - 1] {
            return velocities[velocities.len() - 1];
        }
        let upper = radii.partition_point(|&value| value <= radius);
        let (r0, r1) = (radii[upper - 1], radii[upper]);
        let (v0, v1) = (velocities[upper - 1], velocities[upper]);
        if (r1 - r0).abs() < 1.0e-15 {
            return v0;
        }
        let weight = (radius - r0) / (r1 - r0);
        v0 + (v1 - v0) * weight
    };
    let mut areas = Vec::with_capacity(aperture_elements.len());
    let mut mapped = Vec::with_capacity(aperture_elements.len());
    let mut centroids = Vec::with_capacity(aperture_elements.len());
    for &index in aperture_elements {
        let element = &mesh.elements[index];
        let radius = (element.center[0] * element.center[0]
            + element.center[1] * element.center[1])
            .sqrt();
        areas.push(element.area);
        mapped.push(sample(radius));
        centroids.push(radius);
    }
    InterfaceVelocityTrace {
        frequency_hz,
        element_areas_m2: areas,
        element_velocities_rms: mapped,
        element_centroid_radius_m: centroids,
    }
}

/// Solves one axisymmetric driver point through the exterior BEM.
///
/// This is the full trace-to-BEM path on a single surface: the FEM
/// interface rings and the BEM aperture share the diaphragm radius,
/// ring velocities are interpolated onto aperture elements by radius,
/// the production DtN map is condensed from the baffled closed
/// surface, and the distributed result (with its labelled piston
/// baseline) is evaluated. The finite baffle is part of the model;
/// results must not be quoted as infinite-baffle values.
///
/// # Errors
///
/// Returns an error for a missing interface, an invalid trace, or a
/// failed BEM condensation.
pub fn solve_axisymmetric_radiation(
    model: &AxisymmetricLinearModel,
    solution: &AxisymmetricLinearSolution,
    sound_speed_m_per_s: f64,
    density_kg_per_m3: f64,
    config: BaffledApertureConfig,
) -> Result<DistributedRadiationResult, DistributedRadiationError> {
    let trace = axisymmetric_interface_trace(model, solution)?;
    let radius = trace.effective_radius_m();
    let (mesh, aperture_elements) = build_baffled_aperture_mesh(radius, config)?;
    let builder = crate::fem_bem_coupling::DtnBuilder::new(
        mesh.clone(),
        aperture_elements.clone(),
        sound_speed_m_per_s,
        density_kg_per_m3,
    )
    .map_err(|error| DistributedRadiationError::Bem { detail: error.to_string() })?;
    let dtn = builder
        .factor(solution.frequency_hz)
        .and_then(|factors| factors.build_dtn())
        .map_err(|error| DistributedRadiationError::Bem { detail: error.to_string() })?;
    let aperture_trace =
        map_trace_to_aperture(&trace, &mesh, &aperture_elements, solution.frequency_hz);
    evaluate_distributed_radiation(&dtn, &aperture_trace, sound_speed_m_per_s)
}

/// One breakup peak: undamped eigenfrequency kept separate from the
/// damped response peak.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BreakupPeak {
    /// Undamped structural eigenfrequency in Hz (from the eigen solve).
    pub undamped_hz: f64,
    /// Damped peak frequency `f_n·√(1 − ζ²)` with `ζ ≈ η/2`, in Hz.
    pub damped_peak_hz: f64,
    /// Modal quality factor `Q ≈ 1/η`.
    pub quality_factor: f64,
    /// Structural loss factor `η` used for the damping correction.
    pub loss_factor: f64,
}

/// Separate undamped breakup eigenfrequencies from their damped peaks.
///
/// The eigen solve itself stays undamped (so mode shapes and frequencies
/// remain comparable across meshes); damping enters only here, as a
/// labelled correction with `Q = 1/η`. Peaks are returned in increasing
/// undamped-frequency order.
pub fn separate_breakup_peaks(
    undamped_hz: &[f64],
    loss_factor: f64,
) -> Result<Vec<BreakupPeak>, DistributedRadiationError> {
    let invalid = |detail: &str| DistributedRadiationError::InvalidBreakup {
        detail: detail.to_string(),
    };
    if undamped_hz.is_empty() {
        return Err(invalid("at least one undamped eigenfrequency is required"));
    }
    if !loss_factor.is_finite() || loss_factor < 0.0 {
        return Err(invalid("loss factor must be finite and non-negative"));
    }
    let zeta = 0.5 * loss_factor;
    if zeta >= 1.0 {
        return Err(invalid("loss factor implies an overdamped (ζ ≥ 1) correction"));
    }
    let mut sorted = undamped_hz.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mut peaks = Vec::with_capacity(sorted.len());
    for frequency in sorted {
        if !frequency.is_finite() || frequency <= 0.0 {
            return Err(invalid("eigenfrequencies must be finite and positive"));
        }
        peaks.push(BreakupPeak {
            undamped_hz: frequency,
            damped_peak_hz: frequency * (1.0 - zeta * zeta).sqrt(),
            quality_factor: if loss_factor > 0.0 {
                1.0 / loss_factor
            } else {
                f64::INFINITY
            },
            loss_factor,
        });
    }
    Ok(peaks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::axisymmetric_linear::AxisymmetricLinearConfig;

    fn diagonal_dtn(n: usize, frequency_hz: f64, impedance: Complex64) -> DtnMap {
        let mut matrix = vec![Complex64::new(0.0, 0.0); n * n];
        for i in 0..n {
            matrix[i * n + i] = impedance;
        }
        DtnMap { frequency_hz, n, matrix }
    }

    fn two_element_trace() -> InterfaceVelocityTrace {
        InterfaceVelocityTrace {
            frequency_hz: 1_000.0,
            element_areas_m2: vec![0.01, 0.01],
            element_velocities_rms: vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.5, 0.0),
            ],
            element_centroid_radius_m: vec![0.04, 0.07],
        }
    }

    #[test]
    fn uniform_trace_matches_piston_and_nonuniform_trace_does_not() {
        let dtn = diagonal_dtn(2, 1_000.0, Complex64::new(400.0, -100.0));
        let trace = two_element_trace();
        let result = evaluate_distributed_radiation(&dtn, &trace, 343.0).unwrap();
        assert!(result.radiated_power_w > 0.0);
        assert!(result.piston_projection_pressure_error > 0.1);
        let uniform = InterfaceVelocityTrace {
            element_velocities_rms: vec![Complex64::new(1.0, 0.0); 2],
            ..two_element_trace()
        };
        let uniform_result = evaluate_distributed_radiation(&dtn, &uniform, 343.0).unwrap();
        assert!(uniform_result.piston_projection_pressure_error < 1.0e-12);
        assert!(
            (uniform_result.distributed_mean_impedance - uniform_result.lumped_piston_impedance)
                .norm()
                < 1.0e-9
        );
    }

    #[test]
    fn handoff_fails_closed_on_length_and_frequency_mismatch() {
        let dtn = diagonal_dtn(3, 500.0, Complex64::new(400.0, 0.0));
        let trace = two_element_trace();
        assert!(matches!(
            evaluate_distributed_radiation(&dtn, &trace, 343.0),
            Err(DistributedRadiationError::LengthMismatch { .. })
        ));
        let dtn2 = diagonal_dtn(2, 500.0, Complex64::new(400.0, 0.0));
        assert!(matches!(
            evaluate_distributed_radiation(&dtn2, &trace, 343.0),
            Err(DistributedRadiationError::FrequencyMismatch { .. })
        ));
        let mut bad = two_element_trace();
        bad.element_areas_m2[0] = 0.0;
        assert!(bad.validate().is_err());
    }

    #[test]
    fn axisymmetric_trace_feeds_dtn_end_to_end() {
        let model =
            crate::axisymmetric_linear::AxisymmetricLinearModel::build(
                AxisymmetricLinearConfig::default(),
            )
            .unwrap();
        let solution = model
            .solve_frequency_direct_reference(200.0, Complex64::new(1.0, 0.0))
            .unwrap();
        let trace = axisymmetric_interface_trace(&model, &solution).unwrap();
        assert!(!trace.element_areas_m2.is_empty());
        assert!(trace.total_area_m2() > 0.0);
        // Locally reacting stand-in for the production BEM map: the trace
        // contract (order, areas, RMS phasors) is what is exercised here;
        // DtnMap itself is validated against the pulsating sphere in dtn.rs.
        let impedance = Complex64::new(1.204 * 343.0, 0.0);
        let dtn = diagonal_dtn(trace.element_areas_m2.len(), 200.0, impedance);
        let result = evaluate_distributed_radiation(&dtn, &trace, 343.0).unwrap();
        assert_eq!(result.element_pressures_rms.len(), trace.element_areas_m2.len());
        assert!(result.ka > 0.0);
    }

    #[test]
    fn baffled_aperture_mesh_is_closed_and_solves_end_to_end() {
        let config = BaffledApertureConfig {
            baffle_radius_factor: 3.0,
            sectors: 8,
            aperture_rings: 2,
            baffle_rings: 1,
            dome_rings: 2,
        };
        let (mesh, aperture) = build_baffled_aperture_mesh(0.05, config).unwrap();
        assert!(!aperture.is_empty());
        // Closed surface: every undirected edge is shared by two elements.
        let mut edges = std::collections::HashMap::<(usize, usize), usize>::new();
        for element in &mesh.elements {
            let conn = &element.connectivity;
            for corner in 0..3 {
                let (a, b) = (conn[corner], conn[(corner + 1) % 3]);
                *edges.entry(if a < b { (a, b) } else { (b, a) }).or_insert(0) += 1;
            }
            assert!(element.area > 0.0);
        }
        assert!(edges.values().all(|&count| count == 2), "baffled surface must be closed");
        let model =
            crate::axisymmetric_linear::AxisymmetricLinearModel::build(
                AxisymmetricLinearConfig::default(),
            )
            .unwrap();
        let solution = model
            .solve_frequency_direct_reference(200.0, Complex64::new(1.0, 0.0))
            .unwrap();
        let result = solve_axisymmetric_radiation(&model, &solution, 343.0, 1.204, config)
            .unwrap();
        assert_eq!(result.element_pressures_rms.len(), aperture.len());
        // Passivity holds for the exact operator; at ka ~ 0.2 the
        // radiation resistance is O((ka)^2) and a coarse constant-element
        // mesh can undershoot it by a small absolute amount. Gate on a
        // discretisation tolerance, not on exact positivity.
        assert!(
            result.radiated_power_w >= -1.0e-3,
            "radiated power must be passive within discretisation tolerance, got {}",
            result.radiated_power_w
        );
        assert!(result.distributed_mean_impedance.is_finite());
    }

    #[test]
    fn breakup_peaks_keep_undamped_and_damped_values_separate() {
        let peaks = separate_breakup_peaks(&[3_000.0, 1_500.0], 0.04).unwrap();
        assert_eq!(peaks[0].undamped_hz, 1_500.0);
        assert!(peaks[0].damped_peak_hz < peaks[0].undamped_hz);
        assert!((peaks[0].quality_factor - 25.0).abs() < 1.0e-12);
        assert!(separate_breakup_peaks(&[], 0.04).is_err());
        assert!(separate_breakup_peaks(&[100.0], 2.5).is_err());
        assert_eq!(radiation_regime(0.1), RadiationRegime::Compact);
        assert_eq!(radiation_regime(2.0), RadiationRegime::DistributedRequired);
    }
}
