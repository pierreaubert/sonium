//! Mesh types and generators
//!
//! This module provides mesh data structures and generators for 2D and 3D finite element analysis.

pub mod cad_import;
pub mod generators;
#[cfg(feature = "mshio")]
pub mod gmsh;
#[cfg(feature = "mshio")]
pub mod gmsh_write;
#[cfg(feature = "meshlang")]
pub mod meshlang_bridge;
pub mod motion;
pub mod p2;
pub mod parallelism;
pub mod ply;
pub mod quality;
pub mod refinement;
pub mod stl;
pub mod submesh;
pub mod tetrahedralize;
pub mod topology;
pub mod triangulate;
pub mod types;
pub mod validate;

pub use generators::*;
#[cfg(feature = "meshlang")]
pub use meshlang_bridge::{mesh_from_mlg_file, mesh_from_mlg_str};
pub use motion::{
    HarmonicMotionOptions, HarmonicMotionReport, MeshMotion, MeshMotionError, MeshMotionReport,
};
pub use p2::{
    HEX27_GMSH_TO_FEM, QUA9_GMSH_TO_FEM, TET10_GMSH_TO_FEM, TRI6_GMSH_TO_FEM, gmsh_to_fem_p2_perm,
    promote_to_p2, reorder_gmsh_to_fem_p2,
};
pub use parallelism::{
    ParallelPolicy, QUICKHULL_ITEMS_PER_THREAD, TOPOLOGY_ITEMS_PER_THREAD, use_parallel,
};
pub use submesh::*;
pub use topology::{Face, FaceKey, MeshTopology};
pub use triangulate::*;
pub use types::*;
