use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferResponsibility,
};
use sonium_fem::mesh::{
    MeshTopology, MeshlangTransferError, apply_hcurl_order1_state_transfer,
    transfer_hcurl_order1_state_checked, unit_square_triangles,
};
use sonium_fem::spaces::HcurlSpace;
use std::collections::BTreeMap;

fn edge_id(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Curve, index as u64)
}

fn edge_moment(mesh: &sonium_fem::mesh::Mesh, edge: (usize, usize)) -> f64 {
    let p0 = mesh.nodes[edge.0];
    let p1 = mesh.nodes[edge.1];
    let dx = p1.x - p0.x;
    let dy = p1.y - p0.y;
    dx * (1.0 - 0.5 * (p0.y + p1.y)) + dy * (2.0 + 0.5 * (p0.x + p1.x))
}

pub fn meshlang_hcurl_bridge_transfers_curved_target_edges() {
    let source_mesh = unit_square_triangles(1);
    let mut target_mesh = source_mesh.clone().to_p2();
    let curved_midpoint = target_mesh.elements[0].nodes[3];
    target_mesh.nodes[curved_midpoint].y += 0.08;

    let source_space = HcurlSpace::new();
    let target_space = HcurlSpace::new();
    let source_topology = MeshTopology::build(&source_mesh);
    let target_topology = MeshTopology::build(&target_mesh);
    let source_values = source_topology
        .edges
        .iter()
        .map(|&edge| edge_moment(&source_mesh, edge))
        .collect::<Vec<_>>();
    let expected = transfer_hcurl_order1_state_checked(
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &source_values,
    )
    .expect("curved target transfer reference");

    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: target_topology
            .edges
            .iter()
            .enumerate()
            .map(|(index, _)| IrStateTransferEntry {
                entity: edge_id(index),
                action: IrStateTransferAction::Interpolate,
            })
            .collect(),
    };
    let source_entities = source_topology
        .edges
        .iter()
        .enumerate()
        .map(|(index, _)| (edge_id(index), index))
        .collect::<BTreeMap<_, _>>();
    let target_entities = target_topology
        .edges
        .iter()
        .enumerate()
        .map(|(index, _)| (edge_id(index), index))
        .collect::<BTreeMap<_, _>>();

    let transferred = apply_hcurl_order1_state_transfer::<&'static str, _>(
        &plan,
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &source_values,
        &source_entities,
        &target_entities,
        |_, _, _| Err("unexpected callback"),
    )
    .expect("MeshLang curved H(curl) bridge");

    assert_eq!(transferred.len(), expected.len());
    for (index, value) in expected.iter().enumerate() {
        let actual = transferred[&edge_id(index)];
        assert!(
            (actual - value).abs() < 2.0e-9,
            "edge {index}: {actual} != {value}"
        );
    }
}

#[test]
fn meshlang_hcurl_bridge_transfers_curved_target_edges_test() {
    meshlang_hcurl_bridge_transfers_curved_target_edges();
}

#[test]
fn meshlang_hcurl_bridge_rejects_curved_conservative_action() {
    let mesh = unit_square_triangles(1);
    let topology = MeshTopology::build(&mesh);
    let edge = edge_id(0);
    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Conservative,
        entries: vec![IrStateTransferEntry {
            entity: edge,
            action: IrStateTransferAction::Conservative,
        }],
    };
    let entities = BTreeMap::from([(edge, 0usize)]);
    let error = apply_hcurl_order1_state_transfer::<&'static str, _>(
        &plan,
        &mesh,
        &mesh,
        &HcurlSpace::new(),
        &HcurlSpace::new(),
        &vec![0.0; topology.edges.len()],
        &entities,
        &entities,
        |_, _, _| Ok(0.0),
    )
    .expect_err("scalar conservative action must remain rejected for H(curl)");
    assert!(matches!(
        error,
        MeshlangTransferError::UnsupportedAction {
            entity,
            action: IrStateTransferAction::Conservative,
        } if entity == edge
    ));
}
