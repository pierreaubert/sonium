//! Resolved axisymmetric enclosure, port, baffle, and exterior acoustics.

use crate::assembly::{PrebakedAssembler, TripletMatrix, assemble_bilinear};
use crate::basis::PolynomialDegree;
use crate::boundary::{PmlDirection, PmlRegion};
use crate::forms::{BilinearForm, LocalMatrix};
use crate::mesh::{Mesh, Point};
use crate::spaces::{BasisTabulation, H1Space};
use num_complex::Complex64;
use std::collections::BTreeMap;
use std::f64::consts::PI;

/// Axisymmetric Helmholtz form with coordinate-stretched PML coefficients.
#[derive(Debug, Clone)]
pub struct AxisymmetricPmlHelmholtzForm {
    pub wavenumber: Complex64,
    pub regions: Vec<PmlRegion>,
}

impl AxisymmetricPmlHelmholtzForm {
    pub fn new(wavenumber: Complex64, regions: Vec<PmlRegion>) -> Self {
        Self {
            wavenumber,
            regions,
        }
    }
}

impl BilinearForm<Complex64> for AxisymmetricPmlHelmholtzForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<Complex64>,
    ) {
        let point = Point::new_2d(test.physical_coords[0], test.physical_coords[1]);
        let radius = point.x.abs().max(1.0e-12);
        let volume_weight = 2.0 * PI * radius * test.det_j * weight;
        let (stiffness, mass) = pml_coefficients(&point, &self.regions);
        let k_squared = self.wavenumber * self.wavenumber;
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                let gradient = stiffness[0] * test.grad(i, 0) * trial.grad(j, 0)
                    + stiffness[1] * test.grad(i, 1) * trial.grad(j, 1);
                let value = gradient - k_squared * mass * test.value(i) * trial.value(j);
                local_matrix.add(i, j, value * volume_weight);
            }
        }
    }
}

/// Four-real-component affine assembler for a frequency-scaled axisymmetric PML.
///
/// Components are ordered as `Re(Kc), Im(Kc), Re(Mc), Im(Mc)`, allowing
/// `Kc - k² Mc` to be reconstructed with complex coefficients
/// `[1, i, -k², -i k²]`.
pub type AxisymmetricPmlAssembler = PrebakedAssembler<4>;

#[derive(Debug, Clone, Copy)]
enum AxisymmetricPmlComponent {
    StiffnessReal,
    StiffnessImaginary,
    MassReal,
    MassImaginary,
}

#[derive(Debug, Clone)]
struct AxisymmetricPmlComponentForm {
    component: AxisymmetricPmlComponent,
    regions: Vec<PmlRegion>,
}

impl BilinearForm<f64> for AxisymmetricPmlComponentForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        let point = Point::new_2d(test.physical_coords[0], test.physical_coords[1]);
        let radius = point.x.abs().max(1.0e-12);
        let volume_weight = 2.0 * PI * radius * test.det_j * weight;
        let (stiffness, mass) = pml_coefficients(&point, &self.regions);
        for i in 0..test.n_dofs {
            for j in 0..trial.n_dofs {
                let value = match self.component {
                    AxisymmetricPmlComponent::StiffnessReal => {
                        stiffness[0].re * test.grad(i, 0) * trial.grad(j, 0)
                            + stiffness[1].re * test.grad(i, 1) * trial.grad(j, 1)
                    }
                    AxisymmetricPmlComponent::StiffnessImaginary => {
                        stiffness[0].im * test.grad(i, 0) * trial.grad(j, 0)
                            + stiffness[1].im * test.grad(i, 1) * trial.grad(j, 1)
                    }
                    AxisymmetricPmlComponent::MassReal => mass.re * test.value(i) * trial.value(j),
                    AxisymmetricPmlComponent::MassImaginary => {
                        mass.im * test.value(i) * trial.value(j)
                    }
                };
                local_matrix.add(i, j, value * volume_weight);
            }
        }
    }
}

/// Prebake the affine `Kc - k² Mc` operator for a frequency-scaled PML.
///
/// `regions` must use a frequency-independent stretch (for example regions
/// returned by [`positive_iomega_axisymmetric_exterior_pml`] at any positive
/// reference wavenumber). Robin boundary mass matrices are retained as
/// separately weighted components.
pub fn build_axisymmetric_pml_assembler(
    mesh: &Mesh,
    regions: Vec<PmlRegion>,
    quad_order: usize,
    robin_markers: &[i32],
) -> AxisymmetricPmlAssembler {
    let space = H1Space::new(PolynomialDegree::P1);
    let component = |component| AxisymmetricPmlComponentForm {
        component,
        regions: regions.clone(),
    };
    let components = [
        assemble_bilinear(
            &component(AxisymmetricPmlComponent::StiffnessReal),
            &space,
            &space,
            mesh,
            quad_order,
        ),
        assemble_bilinear(
            &component(AxisymmetricPmlComponent::StiffnessImaginary),
            &space,
            &space,
            mesh,
            quad_order,
        ),
        assemble_bilinear(
            &component(AxisymmetricPmlComponent::MassReal),
            &space,
            &space,
            mesh,
            quad_order,
        ),
        assemble_bilinear(
            &component(AxisymmetricPmlComponent::MassImaginary),
            &space,
            &space,
            mesh,
            quad_order,
        ),
    ];
    let boundary_components: Vec<_> = robin_markers
        .iter()
        .map(|&marker| (marker as usize, axisymmetric_robin_triplet(mesh, marker)))
        .collect();
    PrebakedAssembler::from_triplets(components, &boundary_components)
}

/// Unit axisymmetric boundary mass matrix for a Robin coefficient.
pub fn axisymmetric_robin_triplet(mesh: &Mesh, marker: i32) -> TripletMatrix<f64> {
    let mut matrix = TripletMatrix::new(mesh.num_nodes(), mesh.num_nodes());
    for boundary in mesh.boundaries.iter().filter(|face| face.marker == marker) {
        let [n0, n1] = [boundary.nodes[0], boundary.nodes[1]];
        let p0 = &mesh.nodes[n0];
        let p1 = &mesh.nodes[n1];
        let length = (p1.x - p0.x).hypot(p1.y - p0.y);
        let radial_mass = [
            [3.0 * p0.x + p1.x, p0.x + p1.x],
            [p0.x + p1.x, p0.x + 3.0 * p1.x],
        ];
        for (i, row) in radial_mass.iter().enumerate() {
            for (j, coefficient) in row.iter().enumerate() {
                matrix.push(
                    boundary.nodes[i],
                    boundary.nodes[j],
                    2.0 * PI * length * coefficient / 12.0,
                );
            }
        }
    }
    matrix
}

fn pml_coefficients(point: &Point, regions: &[PmlRegion]) -> ([Complex64; 2], Complex64) {
    for region in regions {
        if region.contains(point) {
            return (
                [
                    region.stiffness_coefficient_x(point),
                    region.stiffness_coefficient_y(point),
                ],
                region.mass_coefficient(point),
            );
        }
    }
    ([Complex64::new(1.0, 0.0); 2], Complex64::new(1.0, 0.0))
}

/// Add `alpha ∫_Gamma p q 2*pi*r ds` to an axisymmetric matrix.
pub fn add_axisymmetric_robin(
    matrix: &mut TripletMatrix<Complex64>,
    mesh: &Mesh,
    marker: i32,
    alpha: Complex64,
) {
    for boundary in mesh.boundaries.iter().filter(|face| face.marker == marker) {
        let [n0, n1] = [boundary.nodes[0], boundary.nodes[1]];
        let p0 = &mesh.nodes[n0];
        let p1 = &mesh.nodes[n1];
        let length = (p1.x - p0.x).hypot(p1.y - p0.y);
        let radial_mass = [
            [3.0 * p0.x + p1.x, p0.x + p1.x],
            [p0.x + p1.x, p0.x + 3.0 * p1.x],
        ];
        for (i, row) in radial_mass.iter().enumerate() {
            for (j, coefficient) in row.iter().enumerate() {
                matrix.push(
                    boundary.nodes[i],
                    boundary.nodes[j],
                    alpha * 2.0 * PI * length * coefficient / 12.0,
                );
            }
        }
    }
}

/// Add a uniform normal-derivative load `g = ∂p/∂n` to an axisymmetric RHS.
pub fn add_axisymmetric_neumann(
    rhs: &mut [Complex64],
    mesh: &Mesh,
    marker: i32,
    normal_derivative: Complex64,
) {
    let gauss = [
        (-1.0_f64 / 3.0_f64.sqrt(), 1.0),
        (1.0_f64 / 3.0_f64.sqrt(), 1.0),
    ];
    for boundary in mesh.boundaries.iter().filter(|face| face.marker == marker) {
        let [n0, n1] = [boundary.nodes[0], boundary.nodes[1]];
        let p0 = &mesh.nodes[n0];
        let p1 = &mesh.nodes[n1];
        let length = (p1.x - p0.x).hypot(p1.y - p0.y);
        for (xi, weight) in gauss {
            let shape = [0.5 * (1.0 - xi), 0.5 * (1.0 + xi)];
            let radius = shape[0] * p0.x + shape[1] * p1.x;
            let measure = 2.0 * PI * radius * 0.5 * length * weight;
            rhs[n0] += normal_derivative * shape[0] * measure;
            rhs[n1] += normal_derivative * shape[1] * measure;
        }
    }
}

/// Kirchhoff representation evaluated from an internal marker interface.
///
/// The inner marker identifies the physical exterior and the outer marker the
/// PML. Pressure and its physical-side normal derivative are reconstructed
/// from the P1 solution; azimuthal rings are integrated explicitly.
pub fn axisymmetric_kirchhoff_far_field(
    mesh: &Mesh,
    physical_marker: i32,
    pml_marker: i32,
    pressure: &[Complex64],
    wavenumber: f64,
    observation_xyz_m: [f64; 3],
) -> Result<Complex64, ResolvedAcousticsError> {
    if pressure.len() != mesh.num_nodes() {
        return Err(ResolvedAcousticsError::PressureLengthMismatch);
    }
    let interfaces = marker_interfaces(mesh, physical_marker, pml_marker);
    if interfaces.is_empty() {
        return Err(ResolvedAcousticsError::MissingKirchhoffInterface);
    }
    let mut result = Complex64::default();
    let gauss = [
        (-1.0_f64 / 3.0_f64.sqrt(), 1.0),
        (1.0_f64 / 3.0_f64.sqrt(), 1.0),
    ];
    let azimuthal_count = 48;
    for interface in interfaces {
        let element = &mesh.elements[interface.physical_element];
        let nodes = element.vertices();
        let gradients = triangle_gradients(mesh, nodes);
        let pressure_gradient = [
            (0..3)
                .map(|local| pressure[nodes[local]] * gradients[local][0])
                .sum::<Complex64>(),
            (0..3)
                .map(|local| pressure[nodes[local]] * gradients[local][1])
                .sum::<Complex64>(),
        ];
        let p0 = &mesh.nodes[interface.edge[0]];
        let p1 = &mesh.nodes[interface.edge[1]];
        let length = (p1.x - p0.x).hypot(p1.y - p0.y);
        for (xi, line_weight) in gauss {
            let shape = [0.5 * (1.0 - xi), 0.5 * (1.0 + xi)];
            let radius = shape[0] * p0.x + shape[1] * p1.x;
            let axial = shape[0] * p0.y + shape[1] * p1.y;
            let p_surface =
                shape[0] * pressure[interface.edge[0]] + shape[1] * pressure[interface.edge[1]];
            let dp_dn = pressure_gradient[0] * interface.normal_rz[0]
                + pressure_gradient[1] * interface.normal_rz[1];
            let ring_weight =
                radius * 0.5 * length * line_weight * 2.0 * PI / azimuthal_count as f64;
            for azimuthal in 0..azimuthal_count {
                let phi = 2.0 * PI * (azimuthal as f64 + 0.5) / azimuthal_count as f64;
                let source = [radius * phi.cos(), radius * phi.sin(), axial];
                let normal = [
                    interface.normal_rz[0] * phi.cos(),
                    interface.normal_rz[0] * phi.sin(),
                    interface.normal_rz[1],
                ];
                let delta = [
                    source[0] - observation_xyz_m[0],
                    source[1] - observation_xyz_m[1],
                    source[2] - observation_xyz_m[2],
                ];
                let distance =
                    (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
                if distance <= 1.0e-12 {
                    return Err(ResolvedAcousticsError::ObservationOnSurface);
                }
                let green =
                    Complex64::new(0.0, -wavenumber * distance).exp() / (4.0 * PI * distance);
                let radial_derivative = Complex64::new(-1.0 / distance, -wavenumber) * green;
                let d_distance_dn =
                    (delta[0] * normal[0] + delta[1] * normal[1] + delta[2] * normal[2]) / distance;
                let dg_dn = radial_derivative * d_distance_dn;
                result += (green * dp_dn - p_surface * dg_dn) * ring_weight;
            }
        }
    }
    Ok(result)
}

/// Active acoustic power crossing the physical/PML interface.
///
/// Pressure and velocity are RMS phasors using `e^(+iωt)`, so the outward
/// intensity is `Re(p u_n*)` and `u_n = i (∂p/∂n)/(ωρ)`.
pub fn axisymmetric_interface_power(
    mesh: &Mesh,
    physical_marker: i32,
    pml_marker: i32,
    pressure: &[Complex64],
    omega_rad_per_s: f64,
    density_kg_per_m3: f64,
) -> Result<f64, ResolvedAcousticsError> {
    if pressure.len() != mesh.num_nodes() {
        return Err(ResolvedAcousticsError::PressureLengthMismatch);
    }
    if !omega_rad_per_s.is_finite()
        || omega_rad_per_s <= 0.0
        || !density_kg_per_m3.is_finite()
        || density_kg_per_m3 <= 0.0
    {
        return Err(ResolvedAcousticsError::InvalidMediumOrFrequency);
    }
    let interfaces = marker_interfaces(mesh, physical_marker, pml_marker);
    if interfaces.is_empty() {
        return Err(ResolvedAcousticsError::MissingKirchhoffInterface);
    }
    let gauss = [
        (-1.0_f64 / 3.0_f64.sqrt(), 1.0),
        (1.0_f64 / 3.0_f64.sqrt(), 1.0),
    ];
    let mut power = 0.0;
    for interface in interfaces {
        let element = &mesh.elements[interface.physical_element];
        let nodes = element.vertices();
        let gradients = triangle_gradients(mesh, nodes);
        let pressure_gradient = [
            (0..3)
                .map(|local| pressure[nodes[local]] * gradients[local][0])
                .sum::<Complex64>(),
            (0..3)
                .map(|local| pressure[nodes[local]] * gradients[local][1])
                .sum::<Complex64>(),
        ];
        let dp_dn = pressure_gradient[0] * interface.normal_rz[0]
            + pressure_gradient[1] * interface.normal_rz[1];
        let velocity_normal =
            Complex64::new(0.0, 1.0) * dp_dn / (omega_rad_per_s * density_kg_per_m3);
        let p0 = &mesh.nodes[interface.edge[0]];
        let p1 = &mesh.nodes[interface.edge[1]];
        let length = (p1.x - p0.x).hypot(p1.y - p0.y);
        for (xi, line_weight) in gauss {
            let shape = [0.5 * (1.0 - xi), 0.5 * (1.0 + xi)];
            let radius = shape[0] * p0.x + shape[1] * p1.x;
            let p_surface =
                shape[0] * pressure[interface.edge[0]] + shape[1] * pressure[interface.edge[1]];
            power += (p_surface * velocity_normal.conj()).re
                * 2.0
                * PI
                * radius
                * 0.5
                * length
                * line_weight;
        }
    }
    Ok(power)
}

/// Active acoustic power crossing a physical/PML interface, recovered from
/// the physical-domain weak-form reaction.
///
/// This is the discrete counterpart of integrating `Re(p u_n*)` over the
/// interface.  It is preferable to differentiating a P1 pressure field on
/// one side of an interior face: that gradient is discontinuous, whereas the
/// Galerkin reaction is the flux that the assembled physical domain applies
/// to its PML neighbour.
pub fn axisymmetric_interface_reaction_power(
    mesh: &Mesh,
    physical_marker: i32,
    pml_marker: i32,
    pressure: &[Complex64],
    omega_rad_per_s: f64,
    density_kg_per_m3: f64,
    speed_of_sound_m_per_s: f64,
) -> Result<f64, ResolvedAcousticsError> {
    if pressure.len() != mesh.num_nodes() {
        return Err(ResolvedAcousticsError::PressureLengthMismatch);
    }
    if !omega_rad_per_s.is_finite()
        || omega_rad_per_s <= 0.0
        || !density_kg_per_m3.is_finite()
        || density_kg_per_m3 <= 0.0
        || !speed_of_sound_m_per_s.is_finite()
        || speed_of_sound_m_per_s <= 0.0
    {
        return Err(ResolvedAcousticsError::InvalidMediumOrFrequency);
    }
    let interfaces = marker_interfaces(mesh, physical_marker, pml_marker);
    if interfaces.is_empty() {
        return Err(ResolvedAcousticsError::MissingKirchhoffInterface);
    }
    let interface_nodes: std::collections::BTreeSet<_> = interfaces
        .iter()
        .flat_map(|interface| interface.edge)
        .collect();
    let wavenumber_squared = (omega_rad_per_s / speed_of_sound_m_per_s).powi(2);
    let mut physical_action = vec![Complex64::default(); mesh.num_nodes()];
    for element in mesh
        .elements
        .iter()
        .filter(|element| element.marker == physical_marker)
    {
        let nodes = element.vertices();
        let points = [
            &mesh.nodes[nodes[0]],
            &mesh.nodes[nodes[1]],
            &mesh.nodes[nodes[2]],
        ];
        let twice_area = ((points[1].x - points[0].x) * (points[2].y - points[0].y)
            - (points[2].x - points[0].x) * (points[1].y - points[0].y))
            .abs();
        let area = 0.5 * twice_area;
        let radius_mean = (points[0].x + points[1].x + points[2].x) / 3.0;
        let gradients = triangle_gradients(mesh, nodes);
        for local_i in 0..3 {
            for local_j in 0..3 {
                let stiffness = radius_mean
                    * (gradients[local_i][0] * gradients[local_j][0]
                        + gradients[local_i][1] * gradients[local_j][1]);
                let mass_radius = if local_i == local_j {
                    area * (points[local_i].x / 10.0
                        + (radius_mean * 3.0 - points[local_i].x) / 30.0)
                } else {
                    let other = 3 - local_i - local_j;
                    area * (2.0 * points[local_i].x + 2.0 * points[local_j].x + points[other].x)
                        / 60.0
                };
                let value = 2.0 * PI * (area * stiffness - wavenumber_squared * mass_radius);
                physical_action[nodes[local_i]] += value * pressure[nodes[local_j]];
            }
        }
    }
    let reaction_work: Complex64 = interface_nodes
        .into_iter()
        .map(|node| pressure[node].conj() * physical_action[node])
        .sum();
    Ok(-reaction_work.im / (omega_rad_per_s * density_kg_per_m3))
}

#[derive(Debug, Clone)]
struct MarkerInterface {
    edge: [usize; 2],
    physical_element: usize,
    normal_rz: [f64; 2],
}

fn marker_interfaces(mesh: &Mesh, physical: i32, pml: i32) -> Vec<MarkerInterface> {
    let mut owners: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
    for (element_index, element) in mesh.elements.iter().enumerate() {
        for (a, b) in element.edges() {
            owners
                .entry((a.min(b), a.max(b)))
                .or_default()
                .push(element_index);
        }
    }
    owners
        .into_iter()
        .filter_map(|((a, b), elements)| {
            if elements.len() != 2 {
                return None;
            }
            let first = &mesh.elements[elements[0]];
            let second = &mesh.elements[elements[1]];
            let (physical_element, pml_element) =
                if first.marker == physical && second.marker == pml {
                    (elements[0], elements[1])
                } else if second.marker == physical && first.marker == pml {
                    (elements[1], elements[0])
                } else {
                    return None;
                };
            let pc = centroid(mesh, mesh.elements[physical_element].vertices());
            let qc = centroid(mesh, mesh.elements[pml_element].vertices());
            let length = (qc[0] - pc[0]).hypot(qc[1] - pc[1]);
            Some(MarkerInterface {
                edge: [a, b],
                physical_element,
                normal_rz: [(qc[0] - pc[0]) / length, (qc[1] - pc[1]) / length],
            })
        })
        .collect()
}

fn centroid(mesh: &Mesh, nodes: &[usize]) -> [f64; 2] {
    [
        nodes.iter().map(|node| mesh.nodes[*node].x).sum::<f64>() / nodes.len() as f64,
        nodes.iter().map(|node| mesh.nodes[*node].y).sum::<f64>() / nodes.len() as f64,
    ]
}

fn triangle_gradients(mesh: &Mesh, nodes: &[usize]) -> [[f64; 2]; 3] {
    let a = &mesh.nodes[nodes[0]];
    let b = &mesh.nodes[nodes[1]];
    let c = &mesh.nodes[nodes[2]];
    let twice_area = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
    [
        [(b.y - c.y) / twice_area, (c.x - b.x) / twice_area],
        [(c.y - a.y) / twice_area, (a.x - c.x) / twice_area],
        [(a.y - b.y) / twice_area, (b.x - a.x) / twice_area],
    ]
}

pub fn helmholtz_resonance_frequency_hz(
    sound_speed_m_per_s: f64,
    cavity_volume_m3: f64,
    port_area_m2: f64,
    effective_port_length_m: f64,
) -> f64 {
    sound_speed_m_per_s / (2.0 * PI)
        * (port_area_m2 / (cavity_volume_m3 * effective_port_length_m)).sqrt()
}

pub fn rigid_rectangular_cavity_mode_hz(
    sound_speed_m_per_s: f64,
    dimensions_m: [f64; 3],
    mode: [usize; 3],
) -> f64 {
    0.5 * sound_speed_m_per_s
        * ((mode[0] as f64 / dimensions_m[0]).powi(2)
            + (mode[1] as f64 / dimensions_m[1]).powi(2)
            + (mode[2] as f64 / dimensions_m[2]).powi(2))
        .sqrt()
}

pub fn hard_circular_duct_cutoff_hz(sound_speed_m_per_s: f64, radius_m: f64) -> f64 {
    // First zero of J1' is 1.8411837813; 1.8412 is the shared duct constant.
    1.8412 * sound_speed_m_per_s / (2.0 * PI * radius_m)
}

pub fn pulsating_sphere_pressure_rms(
    radius_m: f64,
    radial_velocity_rms_m_per_s: Complex64,
    observation_radius_m: f64,
    frequency_hz: f64,
    density_kg_per_m3: f64,
    sound_speed_m_per_s: f64,
) -> Complex64 {
    let omega = 2.0 * PI * frequency_hz;
    let k = omega / sound_speed_m_per_s;
    let phase = Complex64::new(0.0, -k * (observation_radius_m - radius_m)).exp();
    Complex64::new(0.0, density_kg_per_m3 * omega)
        * radial_velocity_rms_m_per_s
        * radius_m.powi(2)
        * phase
        / (observation_radius_m * Complex64::new(1.0, k * radius_m))
}

/// Build e^(+iωt)-compatible radial/top PML regions. Existing PML stretching
/// is conjugated by supplying a negative wavenumber.
pub fn positive_iomega_axisymmetric_pml(
    radial_start_m: f64,
    radial_thickness_m: f64,
    axial_start_m: f64,
    axial_thickness_m: f64,
    wavenumber: f64,
    sigma_max: f64,
) -> Vec<PmlRegion> {
    let dimensional_sigma_max = sigma_max * wavenumber.abs();
    vec![
        PmlRegion::new(
            PmlDirection::XY,
            [radial_start_m, axial_start_m, f64::NEG_INFINITY],
            [
                radial_start_m + radial_thickness_m,
                axial_start_m + axial_thickness_m,
                f64::INFINITY,
            ],
            dimensional_sigma_max,
            -wavenumber,
        ),
        PmlRegion::x_positive(
            radial_start_m,
            radial_thickness_m,
            dimensional_sigma_max,
            -wavenumber,
        ),
        PmlRegion::y_positive(
            axial_start_m,
            axial_thickness_m,
            dimensional_sigma_max,
            -wavenumber,
        ),
    ]
}

/// Build radial, upper, and lower PML regions for a finite-baffle exterior.
/// Corner regions are listed first so both coordinate stretches are applied.
#[allow(clippy::too_many_arguments)]
pub fn positive_iomega_axisymmetric_exterior_pml(
    radial_start_m: f64,
    radial_thickness_m: f64,
    upper_start_m: f64,
    upper_thickness_m: f64,
    lower_start_m: f64,
    lower_thickness_m: f64,
    wavenumber: f64,
    sigma_max: f64,
) -> Vec<PmlRegion> {
    // `sigma_max` is the dimensionless peak stretch. Convert it to the
    // dimensional profile used by PmlRegion so sigma/k remains independent
    // of frequency. The signed -k preserves the e^(+i omega t) convention.
    let dimensional_sigma_max = sigma_max * wavenumber.abs();
    let outer_radius = radial_start_m + radial_thickness_m;
    vec![
        PmlRegion::new(
            PmlDirection::XY,
            [radial_start_m, upper_start_m, f64::NEG_INFINITY],
            [
                outer_radius,
                upper_start_m + upper_thickness_m,
                f64::INFINITY,
            ],
            dimensional_sigma_max,
            -wavenumber,
        ),
        PmlRegion::new(
            PmlDirection::XY,
            [radial_start_m, lower_start_m, f64::NEG_INFINITY],
            [
                outer_radius,
                lower_start_m - lower_thickness_m,
                f64::INFINITY,
            ],
            dimensional_sigma_max,
            -wavenumber,
        ),
        PmlRegion::x_positive(
            radial_start_m,
            radial_thickness_m,
            dimensional_sigma_max,
            -wavenumber,
        ),
        PmlRegion::y_positive(
            upper_start_m,
            upper_thickness_m,
            dimensional_sigma_max,
            -wavenumber,
        ),
        PmlRegion::y_negative(
            lower_start_m,
            lower_thickness_m,
            dimensional_sigma_max,
            -wavenumber,
        ),
    ]
}

#[derive(Debug, thiserror::Error)]
pub enum ResolvedAcousticsError {
    #[error("pressure length does not match mesh nodes")]
    PressureLengthMismatch,
    #[error("physical/PML marker interface is missing")]
    MissingKirchhoffInterface,
    #[error("far-field observation lies on the integration surface")]
    ObservationOnSurface,
    #[error("frequency and medium properties must be finite and positive")]
    InvalidMediumOrFrequency,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frequency_scaled_pml_keeps_coordinate_stretch_fixed() {
        let low = positive_iomega_axisymmetric_pml(1.0, 0.5, 1.0, 0.5, 2.0, 3.0);
        let high = positive_iomega_axisymmetric_pml(1.0, 0.5, 1.0, 0.5, 8.0, 3.0);
        let point = Point::new_2d(1.5, 1.25);
        let low_stretch = low[1].stretching_x(point.x);
        let high_stretch = high[1].stretching_x(point.x);
        assert!((low_stretch - high_stretch).norm() < 1.0e-12);
        assert!((low_stretch - Complex64::new(1.0, -3.0)).norm() < 1.0e-12);
    }

    #[test]
    fn analytical_cavity_duct_and_helmholtz_references_have_correct_scaling() {
        let c = 343.0;
        assert!(
            (rigid_rectangular_cavity_mode_hz(c, [1.0, 0.5, 0.25], [1, 0, 0]) - 171.5).abs()
                < 1.0e-12
        );
        assert!(hard_circular_duct_cutoff_hz(c, 0.02) > 5_000.0);
        let f1 = helmholtz_resonance_frequency_hz(c, 0.02, 1.0e-3, 0.1);
        let f2 = helmholtz_resonance_frequency_hz(c, 0.08, 1.0e-3, 0.1);
        assert!((f1 / f2 - 2.0).abs() < 1.0e-12);
    }

    #[test]
    fn pulsating_sphere_has_inverse_radius_far_field_limit() {
        let p1 = pulsating_sphere_pressure_rms(
            0.02,
            Complex64::new(0.01, 0.0),
            1.0,
            1_000.0,
            1.2,
            343.0,
        );
        let p2 = pulsating_sphere_pressure_rms(
            0.02,
            Complex64::new(0.01, 0.0),
            2.0,
            1_000.0,
            1.2,
            343.0,
        );
        assert!((p1.norm() / p2.norm() - 2.0).abs() < 1.0e-12);
    }
}
