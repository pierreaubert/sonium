use num_complex::Complex64;

use sonium_bem::core::assembly::tbem::build_tbem_system_continuous_quadratic_stabilized;
use sonium_bem::core::mesh::element::basis_values;
use sonium_bem::core::postprocess::pressure::compute_single_layer_field;
use sonium_bem::core::types::{
    BoundaryCondition, Element, ElementProperty, ElementType, Mesh, PhysicsParams,
};

fn triangle(connectivity: Vec<usize>, center: [f64; 3]) -> Element {
    Element {
        connectivity,
        element_type: ElementType::Tri3,
        property: ElementProperty::Surface,
        normal: [0.0, 0.0, 1.0],
        node_normals: vec![[0.0, 0.0, 1.0]; 3],
        center,
        area: 0.5,
        boundary_condition: BoundaryCondition::Velocity(vec![Complex64::default()]),
        group: 0,
        dof_addresses: Vec::new(),
    }
}

fn two_triangle_mesh() -> Mesh {
    Mesh {
        nodes: vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        elements: vec![
            triangle(vec![0, 1, 2], [1.0 / 3.0, 1.0 / 3.0, 0.0]),
            triangle(vec![1, 3, 2], [2.0 / 3.0, 2.0 / 3.0, 0.0]),
        ],
        external_node_numbers: Vec::new(),
        external_element_numbers: Vec::new(),
        num_boundary_nodes: 4,
        num_evaluation_nodes: 0,
        num_boundary_elements: 2,
        num_evaluation_elements: 0,
        symmetry_planes: [0; 3],
        symmetry_coordinates: [0.0; 3],
        num_reflections: 0,
        node_to_elements: Vec::new(),
    }
}

#[test]
fn quadratic_trace_reproduces_an_affine_field_continuously() {
    let mesh = two_triangle_mesh();
    let map = mesh.continuous_quadratic_dof_map().unwrap();
    let mut values = vec![Complex64::new(f64::NAN, 0.0); map.num_dofs()];
    let local_points = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.0, 0.5),
    ];

    for (element_index, element) in mesh.elements.iter().enumerate() {
        let dofs = map.element_dofs(element_index).unwrap();
        for (local, &(s, t)) in local_points.iter().enumerate() {
            let point = element.local_to_global(s, t, &mesh.nodes);
            let value = Complex64::new(point[0] + 2.0 * point[1], 0.0);
            if values[dofs[local]].re.is_nan() {
                values[dofs[local]] = value;
            } else {
                assert!((values[dofs[local]] - value).norm() < 1.0e-12);
            }
        }
    }

    let first = &mesh.elements[0];
    let second = &mesh.elements[1];
    let trace = |_element: &Element, element_index: usize, s: f64, t: f64| {
        let basis = basis_values(
            ElementType::Tri3,
            sonium_bem::core::types::BasisOrder::Quadratic,
            s,
            t,
        );
        basis
            .iter()
            .enumerate()
            .map(|(local, &weight)| {
                values[map.element_dofs(element_index).unwrap()[local]] * weight
            })
            .sum::<Complex64>()
    };

    // The shared edge is (1,2): (s,t)=(1/2,1/2) on the first triangle and
    // (s,t)=(0,1/2) on the oppositely oriented second triangle.
    let first_trace = trace(first, 0, 0.5, 0.5);
    let second_trace = trace(second, 1, 0.0, 0.5);
    assert!((first_trace - second_trace).norm() < 1.0e-12);
    assert!((first_trace.re - 1.5).abs() < 1.0e-12);
}

#[test]
fn continuous_quadratic_assembly_has_unique_global_rows() {
    let mesh = two_triangle_mesh();
    let map = mesh.continuous_quadratic_dof_map().unwrap();
    let mut elements = mesh.elements.clone();
    for (index, element) in elements.iter_mut().enumerate() {
        element.dof_addresses = map.element_dofs(index).unwrap().to_vec();
    }
    let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
    let system = build_tbem_system_continuous_quadratic_stabilized(
        &elements,
        &mesh.nodes,
        &physics,
        1.0,
        0.2,
    );

    assert_eq!(system.matrix.shape(), (map.num_dofs(), map.num_dofs()));
    assert_eq!(system.rhs.len(), map.num_dofs());
    for row in 0..map.num_dofs() {
        for column in 0..map.num_dofs() {
            let value = system.matrix[[row, column]];
            assert!(value.re.is_finite() && value.im.is_finite());
        }
        assert!(system.matrix[[row, row]].norm() > 0.0);
    }
}

#[test]
fn shared_quadratic_field_evaluation_preserves_constant_trace() {
    let mesh = two_triangle_mesh();
    let map = mesh.continuous_quadratic_dof_map().unwrap();
    let mut continuous_elements = mesh.elements.clone();
    for (index, element) in continuous_elements.iter_mut().enumerate() {
        element.dof_addresses = map.element_dofs(index).unwrap().to_vec();
    }
    let density = vec![Complex64::new(1.0, 0.0); map.num_dofs()];
    let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
    let evaluation_point = [[0.4, 0.4, 1.5]];
    let continuous = compute_single_layer_field(
        &evaluation_point,
        &continuous_elements,
        &mesh.nodes,
        &density,
        &physics,
        sonium_bem::core::types::BasisOrder::Quadratic,
    )[0];

    let mut discontinuous_elements = mesh.elements.clone();
    let mut discontinuous_dof = 0;
    for element in &mut discontinuous_elements {
        element.dof_addresses = (discontinuous_dof..discontinuous_dof + 6).collect();
        discontinuous_dof += 6;
    }
    let discontinuous = compute_single_layer_field(
        &evaluation_point,
        &discontinuous_elements,
        &mesh.nodes,
        &vec![Complex64::new(1.0, 0.0); discontinuous_dof],
        &physics,
        sonium_bem::core::types::BasisOrder::Quadratic,
    )[0];
    assert!((continuous - discontinuous).norm() < 1.0e-12);
}
