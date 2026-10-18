//! Mesh types and generators for FEM
//!
//! This module re-exports mesh functionality from sonium-mesher.

pub use sonium_mesher::mesh as _mesh;
pub use sonium_mesher::partition;

// Re-export all mesh types
pub use sonium_mesher::mesh::{
    BoundaryFace, BoundaryType, Element, ElementType, Face, FaceKey, Mesh, MeshTopology,
    PhysicalGroup, PhysicalGroupError, PhysicalGroupRegistry, Point, PolynomialDegree, generators,
    submesh, topology, triangulate, types,
};

// Re-export generators
pub use sonium_mesher::mesh::generators::{
    annular_mesh_triangles, backward_facing_step_triangles, box_mesh_hexahedra,
    box_mesh_tetrahedra, box_mesh_tetrahedra_symmetric, channel_with_airfoil_triangles,
    channel_with_cylinder_triangles, circular_mesh_triangles, duct_profile_mesh,
    layered_spherical_shell_mesh_tetrahedra, naca0012_polygon, rectangular_mesh_quads,
    rectangular_mesh_triangles, spherical_shell_mesh_tetrahedra, unit_cube_hexahedra,
    unit_cube_tetrahedra, unit_square_quads, unit_square_triangles,
};

// Re-export partition
pub use sonium_mesher::partition::{
    ElementGraph, Partition, PartitionError, PartitionGhostLayout, PartitionMesh,
    PartitionObjective, PartitionOptions, build_element_graph, partition_mesh,
    partition_mesh_adjacency, partition_mesh_bisection, partition_mesh_round_robin,
    refine_partition_fm, try_build_element_graph, try_partition_mesh,
};

// Re-export tetrahedralize
pub use sonium_mesher::mesh::tetrahedralize::{
    SurfaceFacet, TetrahedralizationResult, TetrahedralizeConfig, tetrahedralize_box,
    tetrahedralize_box_with_result, tetrahedralize_points, tetrahedralize_points_with_result,
    tetrahedralize_surface, tetrahedralize_surface_with_facets,
    tetrahedralize_surface_with_facets_result,
};

pub mod boundary_mesh;
#[cfg(feature = "meshlang")]
pub mod meshlang_transfer;
#[cfg(feature = "meshlang")]
pub use meshlang_transfer::{
    MeshlangTransferError, PartitionedNodalState, apply_complex_hcurl_order1_state_transfer,
    apply_complex_hcurl_order1_state_transfer_with_refinement_chain,
    apply_complex_hcurl_order2_triangle_state_transfer,
    apply_complex_hcurl_order2_triangle_state_transfer_with_refinement_chain,
    apply_complex_hdiv_rt0_state_transfer,
    apply_complex_hdiv_rt0_state_transfer_with_refinement_chain,
    apply_complex_hdiv_rt1_state_transfer,
    apply_complex_hdiv_rt1_state_transfer_with_refinement_chain,
    apply_complex_nodal_state_transfer, apply_h1_mixed_state_transfer,
    apply_h1_mixed_state_transfer_with_refinement_chain,
    apply_h1_mixed_state_transfer_with_refinement_history, apply_h1_p2_state_transfer,
    apply_h1_p2_state_transfer_with_refinement_chain,
    apply_h1_p2_state_transfer_with_refinement_history, apply_h1_p3_state_transfer,
    apply_h1_p3_state_transfer_with_refinement_chain,
    apply_h1_p3_state_transfer_with_refinement_history, apply_hcurl_mixed_state_transfer,
    apply_hcurl_order1_state_transfer, apply_hcurl_order1_state_transfer_with_refinement_chain,
    apply_hcurl_order2_triangle_state_transfer,
    apply_hcurl_order2_triangle_state_transfer_with_refinement_chain,
    apply_hdiv_rt0_state_transfer, apply_hdiv_rt0_state_transfer_with_refinement_chain,
    apply_hdiv_rt1_state_transfer, apply_hdiv_rt1_state_transfer_with_refinement_chain,
    apply_nodal_state_transfer, apply_nodal_state_transfer_with_refinement_chain,
    apply_nodal_state_transfer_with_refinement_history,
    apply_partitioned_nodal_state_transfer_with_refinement_history,
    synchronize_partitioned_nodal_state,
};
#[cfg(feature = "mshio")]
pub use sonium_mesher::mesh::gmsh;
#[cfg(feature = "meshlang")]
pub use sonium_mesher::mesh::{mesh_from_mlg_file, mesh_from_mlg_str};
pub mod interpolation;
pub use interpolation::{
    InterpolationData, InterpolationError, RefinementTransferChainReport,
    RefinementTransferDirection, RefinementTransferReport, create_interpolation_data_checked,
    interpolate_complex_field_adjoint_checked, interpolate_complex_field_checked,
    interpolate_complex_field_conservative_checked, interpolate_field_adjoint_checked,
    interpolate_field_checked, interpolate_field_conservative_checked, interpolate_scalar_at_point,
    interpolate_scalar_at_point_checked, transfer_design_field_checked,
    transfer_h1_state_adjoint_checked, transfer_h1_state_checked,
    transfer_h1_state_with_refinement_chain_and_mpc_checked,
    transfer_h1_state_with_refinement_chain_checked,
    transfer_h1_state_with_refinement_history_and_mpc_checked,
    transfer_h1_state_with_refinement_history_checked, transfer_hcurl_mixed_triangle_state_checked,
    transfer_hcurl_order1_state_checked,
    transfer_hcurl_order1_state_with_refinement_chain_and_mpc_checked,
    transfer_hcurl_order1_state_with_refinement_chain_checked,
    transfer_hcurl_order1_state_with_refinement_history_and_mpc_checked,
    transfer_hcurl_order1_state_with_refinement_history_checked,
    transfer_hcurl_order2_triangle_state_checked,
    transfer_hcurl_order2_triangle_state_with_refinement_chain_checked,
    transfer_hcurl_order2_triangle_state_with_refinement_history_checked,
    transfer_hdiv_rt0_state_checked, transfer_hdiv_rt0_state_with_refinement_chain_and_mpc_checked,
    transfer_hdiv_rt0_state_with_refinement_chain_checked,
    transfer_hdiv_rt0_state_with_refinement_history_and_mpc_checked,
    transfer_hdiv_rt0_state_with_refinement_history_checked, transfer_hdiv_rt1_state_checked,
    transfer_hdiv_rt1_state_with_refinement_chain_and_mpc_checked,
    transfer_hdiv_rt1_state_with_refinement_chain_checked,
    transfer_hdiv_rt1_state_with_refinement_history_and_mpc_checked,
    transfer_hdiv_rt1_state_with_refinement_history_checked,
    validate_refinement_transfer_chain_checked, validate_refinement_transfer_lineage_checked,
};
pub mod motion;
pub use motion::{MeshMotion, MeshMotionError, MeshMotionReport};
pub mod panel_transfer;
pub use panel_transfer::{
    Panel, PanelSet, PanelStateTransferError, PanelStateTransferOptions, PanelStateTransferResult,
    transfer_panel_scalar_state,
};
pub mod state_transfer;
pub use state_transfer::{
    CfdStateTransfer, CfdStateTransferError, CfdStateTransferReport, KOmegaBdfState,
    KOmegaBdfStateAdjointTransfer, KOmegaBdfStateAdjointTransferChain,
    KOmegaBdfStateAdjointTransferChainReport, KOmegaBdfStateTransfer, KOmegaBdfStateTransferChain,
    KOmegaBdfStateTransferChainReport, KOmegaBdfStateTransferReport, KOmegaStateTransferError,
    P2P1StateAdjointTransfer, P2P1StateAdjointTransferChain, P2P1StateAdjointTransferChainReport,
    P2P1StateTransfer, P2P1StateTransferChain, P2P1StateTransferChainReport,
    P2P1StateTransferReport, transfer_komega_bdf_state, transfer_komega_bdf_state_adjoint,
    transfer_komega_bdf_state_adjoint_chain, transfer_komega_bdf_state_chain,
    transfer_p1_cfd_state, transfer_p2p1_bdf_state, transfer_p2p1_bdf_state_adjoint,
    transfer_p2p1_bdf_state_adjoint_chain, transfer_p2p1_bdf_state_chain, transfer_p2p1_state,
    transfer_p2p1_state_adjoint,
};

// Mesh refinement — re-exported from sonium-mesher
pub use sonium_mesher::mesh::refinement;
pub use sonium_mesher::mesh::refinement::*;
pub use sonium_mesher::mesh::validate::{
    ManifoldReport, MeshValidationError, RepairOptions, RepairReport, check_manifold,
    check_orientation, find_degenerate_elements, find_duplicate_nodes, merge_duplicate_nodes,
    remove_degenerate_elements, repair, try_check_manifold, try_check_orientation,
    try_find_degenerate_elements, try_find_duplicate_nodes, try_merge_duplicate_nodes,
    try_remove_degenerate_elements, try_repair,
};

pub mod c_core;
pub use c_core::*;
pub mod shell_surface;
pub use shell_surface::*;
pub mod two_cylinder;
pub use two_cylinder::*;
