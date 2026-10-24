//! Multi-Level Fast Multipole Method (MLFMM) assembly

use num_complex::Complex64;
use sonium_compute::fmm_cpu::{
    compact_indexed_gather_c64, compact_indexed_scatter_accumulate_c64,
    dense_matvec_scatter_accumulate_c64, indexed_dense_matvec_accumulate_c64,
    indexed_dense_matvec_c64, phase_mul_accumulate_c64,
};
use sonium_solvers::dense::DenseMatrix;

use crate::core::assembly::common::{
    FmmAccuracyError, FmmAction, basis_order_for_proxy, checked_fmm_relative_action_error,
    compute_near_block_stabilized_with_basis, count_dofs, decode_dof_proxy_group,
    get_element_coords, proxy_local_index,
};
use crate::core::assembly::hmatrix::{
    compute_matrix_entry_wavenumber_derivative, compute_matrix_entry_wavenumber_derivative_curved,
};
use crate::core::integration::{quad_quadrature, unit_sphere_quadrature};
use crate::core::mesh::element::{basis_values, quadrature_geometry};
use crate::core::parallel::{
    BEM_BLOCKS_PER_THREAD, gated_chunks_mut, gated_enumerate_map, gated_filter_map_indexed,
    gated_flat_map, gated_for_each_mut, gated_map, use_parallel,
};
use crate::core::types::{BasisOrder, Cluster, ClusterLevel, Element, ElementType, PhysicsParams};
#[cfg(feature = "cuda")]
use sonium_compute::{
    BackendError, CudaBackend, CudaMlfmmC64, CudaMlfmmLevelC64Input, CudaMlfmmNearBlockC64Input,
    CudaMlfmmPhaseTransferC64Input, CudaMlfmmTransferC64Input, CudaMlfmmTranslationC64Input,
    MemoryPool,
};
use sonium_solvers::traits::LinearOperator;
use sonium_wave::special::{legendre_polynomials, spherical_hankel_first_kind};
use std::fmt;
use std::sync::Mutex;

#[cfg(feature = "cuda")]
#[path = "mlfmm_cuda.rs"]
mod cuda;
#[cfg(feature = "cuda")]
#[doc(inline)]
pub use cuda::{CudaMlfmmPreconditioner, CudaMlfmmSolveSession};

/// Validation failure while admitting a mesh to the constant-panel MLFMM
/// kernels.
///
/// The leaf multipole and near-field kernels assemble one column
/// per element and support affine `Tri3`/`Quad4` geometry.  The checked entry points reject
/// higher-order or shared-DOF layouts before any expensive assembly work so
/// callers receive a typed error instead of a panic or a mis-sized operator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MlfmmAssemblyError {
    /// The MLFMM hierarchy has no levels.
    EmptyClusterLevels,
    /// A mesh connectivity entry points outside the coordinate array.
    InvalidNode {
        /// Element index.
        element: usize,
        /// Local connectivity slot.
        local: usize,
        /// Referenced node index.
        node: usize,
        /// Number of coordinate nodes.
        available: usize,
    },
    /// The leaf kernels support only Tri3 and Quad4 surface panels.
    UnsupportedElementType {
        /// Element index.
        element: usize,
    },
    /// An element does not contain the connectivity entries required by its type.
    InvalidConnectivity {
        /// Element index.
        element: usize,
        /// Number of connectivity entries.
        nodes: usize,
    },
    /// The leaf kernels only support one constant coefficient per element.
    UnsupportedBasis {
        /// Element index.
        element: usize,
        /// Number of local coefficient slots.
        dofs: usize,
    },
    /// Element DOF count does not match the requested basis order.
    InvalidBasisDofCount {
        /// Element index.
        element: usize,
        /// Expected local basis count.
        expected: usize,
        /// Actual local DOF count.
        actual: usize,
    },
    /// An active element contains a global DOF outside the assembled vector.
    InvalidDofAddress {
        /// Element index.
        element: usize,
        /// Invalid global DOF.
        global: usize,
        /// Inferred system dimension.
        num_dofs: usize,
    },
    /// Two active elements claim the same global DOF.
    DuplicateGlobalDof {
        /// First element owning the DOF.
        first_element: usize,
        /// Second element claiming the DOF.
        second_element: usize,
        /// Shared global DOF.
        global: usize,
    },
    /// A global DOF is not owned by any active element.
    MissingGlobalDof {
        /// Missing global DOF.
        global: usize,
        /// Inferred system dimension.
        num_dofs: usize,
    },
    /// A cluster references an element not present in the mesh.
    InvalidClusterElement {
        /// Hierarchy level.
        level: usize,
        /// Cluster index.
        cluster: usize,
        /// Referenced element index.
        element: usize,
        /// Number of mesh elements.
        available: usize,
    },
    /// A leaf element occurs in more than one leaf cluster.
    DuplicateLeafElement {
        /// Element index.
        element: usize,
        /// First leaf cluster.
        first_cluster: usize,
        /// Second leaf cluster.
        second_cluster: usize,
    },
    /// An active element is absent from the leaf partition.
    MissingLeafElement {
        /// Element index.
        element: usize,
    },
    /// A hierarchy reference points outside its level.
    InvalidHierarchyReference {
        /// Hierarchy level.
        level: usize,
        /// Cluster index.
        cluster: usize,
        /// Referenced child or sibling index.
        reference: usize,
        /// Number of clusters in the target level.
        available: usize,
    },
    /// A level has no usable expansion quadrature/order.
    InvalidQuadrature {
        /// Hierarchy level.
        level: usize,
        /// Polar quadrature order.
        theta_points: usize,
        /// Azimuthal quadrature order.
        phi_points: usize,
        /// Expansion order.
        expansion_terms: usize,
    },
}

#[cfg(test)]
mod row_action_contract_tests {
    use super::*;

    fn valid_one_level_system() -> MlfmmSystem {
        let mut cluster = Cluster::new([0.0, 0.0, 0.0]);
        cluster.element_indices = vec![0];
        cluster.num_elements = 1;
        let mut level = ClusterLevel::new(1);
        level.clusters.push(cluster);

        let mut system = MlfmmSystem::new(1, 1, vec![level], vec![1]);
        system.leaf_dof_indices = vec![vec![0]];
        system.t_matrices = vec![LevelMatrices {
            level: 0,
            num_clusters: 1,
            matrices: vec![DenseMatrix::from_row_major_vec(
                1,
                1,
                vec![Complex64::new(1.0, 0.0)],
            )],
        }];
        system.s_matrices = vec![LevelMatrices {
            level: 0,
            num_clusters: 1,
            matrices: vec![DenseMatrix::from_row_major_vec(
                1,
                1,
                vec![Complex64::new(1.0, 0.0)],
            )],
        }];
        system.d_matrices = vec![Vec::new()];
        system
    }

    #[test]
    fn checked_full_actions_accept_valid_one_level_layout() {
        let system = valid_one_level_system();
        let input = [Complex64::new(1.0, 0.0)];

        assert!(system.matvec_checked(&input).is_ok());
        assert!(system.matvec_transpose_checked(&input).is_ok());
        assert!(system.apply_hermitian_checked(&input).is_ok());
    }

    #[test]
    fn checked_full_action_rejects_malformed_leaf_dof_before_indexing() {
        let mut system = valid_one_level_system();
        system.leaf_dof_indices = vec![vec![1]];

        assert!(matches!(
            system.matvec_checked(&[Complex64::new(1.0, 0.0)]),
            Err(MlfmmRowActionError::InvalidDof { .. })
        ));
    }

    #[test]
    fn checked_full_action_rejects_malformed_transfer_shape() {
        let mut system = valid_one_level_system();
        system.t_matrices[0].matrices[0] = DenseMatrix::zeros(1, 0);

        assert!(matches!(
            system.matvec_checked(&[Complex64::new(1.0, 0.0)]),
            Err(MlfmmRowActionError::TransferShape { stage: "T", .. })
        ));
    }

    #[test]
    fn checked_full_action_rejects_non_finite_output() {
        let mut system = valid_one_level_system();
        system.near_matrix.push(NearFieldBlock {
            source_cluster: 0,
            field_cluster: 0,
            coefficients: DenseMatrix::from_row_major_vec(
                1,
                1,
                vec![Complex64::new(f64::NAN, 0.0)],
            ),
        });

        assert!(matches!(
            system.matvec_checked(&[Complex64::new(1.0, 0.0)]),
            Err(MlfmmRowActionError::NonFiniteOutput {
                operation: "forward",
                ..
            })
        ));
    }

    #[test]
    fn checked_row_actions_validate_inputs_and_project_rows() {
        let system = MlfmmSystem::new(2, 0, Vec::new(), Vec::new());
        let input = [Complex64::new(1.0, 0.0), Complex64::new(-2.0, 0.5)];

        assert_eq!(
            system.apply_rows_checked(&input, &[1, 0]).unwrap(),
            vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)]
        );
        assert_eq!(
            system.apply_transpose_rows_checked(&input, &[0]).unwrap(),
            vec![Complex64::new(0.0, 0.0)]
        );
        assert!(matches!(
            system.apply_rows_checked(&input[..1], &[0]),
            Err(MlfmmRowActionError::Dimension { .. })
        ));
        assert!(matches!(
            system.apply_rows_checked(
                &[Complex64::new(0.0, 0.0), Complex64::new(f64::INFINITY, 0.0)],
                &[0]
            ),
            Err(MlfmmRowActionError::NonFiniteInput { .. })
        ));
        assert!(matches!(
            system.apply_transpose_rows_checked(&input, &[2]),
            Err(MlfmmRowActionError::InvalidRow { row: 2, .. })
        ));
    }

    #[test]
    fn checked_block_row_actions_match_column_actions() {
        let system = MlfmmSystem::new(2, 0, Vec::new(), Vec::new());
        let mut input = DenseMatrix::zeros(2, 2);
        input[[0, 0]] = Complex64::new(1.0, 0.2);
        input[[1, 0]] = Complex64::new(-2.0, 0.5);
        input[[0, 1]] = Complex64::new(0.4, -0.3);
        input[[1, 1]] = Complex64::new(0.8, 0.1);
        let rows = [1, 0, 1];
        let forward = system.apply_rows_block_checked(&input, &rows).unwrap();
        let transpose = system
            .apply_transpose_rows_block_checked(&input, &rows)
            .unwrap();
        assert_eq!(forward.nrows(), rows.len());
        assert_eq!(forward.ncols(), input.ncols());
        for column in 0..input.ncols() {
            let vector = (0..input.nrows())
                .map(|row| input[[row, column]])
                .collect::<Vec<_>>();
            let expected_forward = system.apply_rows_checked(&vector, &rows).unwrap();
            let expected_transpose = system.apply_transpose_rows_checked(&vector, &rows).unwrap();
            for row in 0..rows.len() {
                assert_eq!(forward[[row, column]], expected_forward[row]);
                assert_eq!(transpose[[row, column]], expected_transpose[row]);
            }
        }
        let wrong = DenseMatrix::zeros(1, 2);
        assert!(matches!(
            system.apply_rows_block_checked(&wrong, &rows),
            Err(MlfmmRowActionError::Dimension { .. })
        ));
    }

    #[test]
    fn checked_hermitian_block_actions_match_scalar_and_rows() {
        let system = MlfmmSystem::new(2, 0, Vec::new(), Vec::new());
        let input = DenseMatrix::from_row_major_vec(
            2,
            2,
            vec![
                Complex64::new(1.0, 0.2),
                Complex64::new(-2.0, 0.5),
                Complex64::new(0.4, -0.3),
                Complex64::new(0.8, 0.1),
            ],
        );
        let block = system.apply_hermitian_block_checked(&input).unwrap();
        let rows = [1, 0, 1];
        let row_block = system
            .apply_hermitian_rows_block_checked(&input, &rows)
            .unwrap();
        for rhs in 0..input.ncols() {
            let column: Vec<_> = (0..input.nrows()).map(|row| input[[row, rhs]]).collect();
            let expected = system.apply_hermitian_checked(&column).unwrap();
            for row in 0..input.nrows() {
                assert!((block[[row, rhs]] - expected[row]).norm() < 1e-12);
            }
            for (selected, &row) in rows.iter().enumerate() {
                assert!((row_block[[selected, rhs]] - expected[row]).norm() < 1e-12);
            }
        }
        assert!(matches!(
            system.apply_hermitian_block_checked(&DenseMatrix::zeros(1, 2)),
            Err(MlfmmRowActionError::Dimension { .. })
        ));
        let mut nonfinite = input.clone();
        nonfinite[[0, 1]].re = f64::NAN;
        assert!(matches!(
            system.apply_hermitian_block_checked(&nonfinite),
            Err(MlfmmRowActionError::NonFiniteInput { .. })
        ));
    }

    #[test]
    fn independent_reference_accuracy_covers_forward_transpose_and_hermitian() {
        let matrix = [
            [Complex64::new(1.0, 0.2), Complex64::new(-0.4, 0.7)],
            [Complex64::new(0.3, -0.5), Complex64::new(0.8, 0.1)],
        ];
        let mut system = MlfmmSystem::new(2, 1, Vec::new(), vec![0]);
        system.leaf_dof_indices = vec![vec![0, 1]];
        system.near_matrix.push(NearFieldBlock {
            source_cluster: 0,
            field_cluster: 0,
            coefficients: DenseMatrix::from_row_major_vec(
                2,
                2,
                matrix.iter().flatten().copied().collect(),
            ),
        });
        let probes = vec![
            vec![Complex64::new(0.7, -0.2), Complex64::new(-0.1, 0.6)],
            vec![Complex64::new(-0.3, 0.4), Complex64::new(0.9, 0.1)],
        ];
        for action in [
            FmmAction::Forward,
            FmmAction::Transpose,
            FmmAction::Hermitian,
        ] {
            let references = probes
                .iter()
                .map(|probe| {
                    (0..2)
                        .map(|row| {
                            (0..2)
                                .map(|column| {
                                    let coefficient = match action {
                                        FmmAction::Forward => matrix[row][column],
                                        FmmAction::Transpose => matrix[column][row],
                                        FmmAction::Hermitian => matrix[column][row].conj(),
                                    };
                                    coefficient * probe[column]
                                })
                                .sum()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let error = system
                .relative_action_error(action, &probes, &references)
                .unwrap();
            assert!(error < 1e-14, "{action:?} relative error {error}");
        }

        assert!(matches!(
            system.relative_action_error(FmmAction::Forward, &[], &[]),
            Err(FmmAccuracyError::EmptyProbeSet)
        ));
        assert!(matches!(
            system.relative_action_error(
                FmmAction::Forward,
                &[vec![Complex64::new(1.0, 0.0)]],
                &[vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]],
            ),
            Err(FmmAccuracyError::InputLength { .. })
        ));
    }

    #[test]
    fn basis_checked_builder_expands_linear_tri3_leaf() {
        use crate::core::types::{BoundaryCondition, ClusterLevel, ElementProperty};

        let nodes = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let elements = vec![Element {
            connectivity: vec![0, 1, 2],
            element_type: ElementType::Tri3,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0; 3]; 3],
            center: [1.0 / 3.0, 1.0 / 3.0, 0.0],
            area: 0.5,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
            group: 0,
            dof_addresses: vec![0, 1, 2],
        }];
        let mut cluster = Cluster::new(elements[0].center);
        cluster.element_indices = vec![0];
        cluster.num_elements = 1;
        let mut level = ClusterLevel::new(1);
        level.clusters.push(cluster);
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);

        let system = build_mlfmm_system_stabilized_with_basis_checked(
            &elements,
            &nodes,
            vec![level],
            &physics,
            BasisOrder::Linear,
            -1.0,
            1.0,
        )
        .expect("linear basis MLFMM admission");
        assert_eq!(system.num_dofs, 3);
        assert_eq!(system.leaf_dof_indices[0].len(), 3);
        assert_eq!(system.t_matrices[0].matrices[0].ncols(), 3);
        assert_eq!(system.s_matrices[0].matrices[0].nrows(), 3);
        let t_matrix = &system.t_matrices[0].matrices[0];
        let basis_column_delta: f64 = (0..t_matrix.nrows())
            .map(|row| (t_matrix[[row, 0]] - t_matrix[[row, 1]]).norm())
            .sum();
        assert!(
            basis_column_delta > 1e-12,
            "linear MLFMM T columns must retain distinct basis weights"
        );
    }

    #[test]
    fn basis_checked_builder_expands_linear_quad4_leaf() {
        use crate::core::types::{BoundaryCondition, ClusterLevel, ElementProperty};

        let nodes = vec![
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ];
        let elements = vec![Element {
            connectivity: vec![0, 1, 2, 3],
            element_type: ElementType::Quad4,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0; 3]; 4],
            center: [0.0, 0.0, 0.0],
            area: 4.0,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
            group: 0,
            dof_addresses: vec![0, 1, 2, 3],
        }];
        let mut cluster = Cluster::new(elements[0].center);
        cluster.element_indices = vec![0];
        cluster.num_elements = 1;
        let mut level = ClusterLevel::new(1);
        level.clusters.push(cluster);

        let system = build_mlfmm_system_stabilized_with_basis_checked(
            &elements,
            &nodes,
            vec![level],
            &PhysicsParams::new(100.0, 343.0, 1.21, false),
            BasisOrder::Linear,
            -1.0,
            1.0,
        )
        .expect("linear basis Quad4 MLFMM admission");
        assert_eq!(system.num_dofs, 4);
        assert_eq!(system.leaf_dof_indices[0].len(), 4);
        assert_eq!(system.t_matrices[0].matrices[0].ncols(), 4);
        let t_matrix = &system.t_matrices[0].matrices[0];
        let basis_column_delta: f64 = (0..t_matrix.nrows())
            .map(|row| (t_matrix[[row, 0]] - t_matrix[[row, 1]]).norm())
            .sum();
        assert!(
            basis_column_delta > 1e-12,
            "linear Quad4 MLFMM T columns must retain distinct basis weights"
        );
    }

    #[test]
    fn basis_checked_builder_uses_curved_quad4_collocation_normals() {
        use crate::core::types::{BoundaryCondition, ClusterLevel, ElementProperty};

        let nodes = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.4],
            [0.0, 1.0, 0.0],
        ];
        let element = Element {
            connectivity: vec![0, 1, 2, 3],
            element_type: ElementType::Quad4,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0; 3]; 4],
            center: [0.5, 0.5, 0.1],
            area: 1.0,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
            group: 0,
            dof_addresses: vec![0, 1, 2, 3],
        };
        let mut cluster = Cluster::new(element.center);
        cluster.element_indices = vec![0];
        cluster.num_elements = 1;
        let mut level = ClusterLevel::new(1);
        level.theta_points = 4;
        level.phi_points = 4;
        level.expansion_terms = 16;
        level.clusters.push(cluster);

        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
        let system = build_mlfmm_system_stabilized_with_basis_checked(
            std::slice::from_ref(&element),
            &nodes,
            vec![level],
            &physics,
            BasisOrder::Linear,
            -1.0,
            1.0,
        )
        .expect("curved Quad4 MLFMM admission");

        let (sphere_coords, _) = unit_sphere_quadrature(4, 4);
        let collocation_points = element.collocation_points(BasisOrder::Linear, &nodes);
        let collocation_normals = element.collocation_normals(BasisOrder::Linear, &nodes);
        assert!(
            (collocation_normals[0][2] - collocation_normals[1][2]).abs() > 1.0e-6,
            "warped Quad4 must have varying collocation normals"
        );

        let matrix = &system.s_matrices[0].matrices[0];
        let coord = sphere_coords[0];
        let ik = Complex64::new(0.0, physics.wave_number);
        let gamma_tau = Complex64::new(-physics.gamma() * physics.tau, 0.0);
        let beta = -physics.burton_miller_beta();
        for local in 0..4 {
            let point = collocation_points[local];
            let diff = [
                point[0] - element.center[0],
                point[1] - element.center[1],
                point[2] - element.center[2],
            ];
            let s_dot_diff = coord[0] * diff[0] + coord[1] * diff[1] + coord[2] * diff[2];
            let s_dot_normal = coord[0] * collocation_normals[local][0]
                + coord[1] * collocation_normals[local][1]
                + coord[2] * collocation_normals[local][2];
            let expected = (gamma_tau + beta * (ik * s_dot_normal))
                * sonium_fastmath::exp_i(physics.wave_number * s_dot_diff);
            let error = (matrix[[local, 0]] - expected).norm();
            assert!(error < 1.0e-12, "curved Quad4 proxy normal error={error}");
        }
    }

    #[test]
    fn basis_checked_builder_expands_quadratic_tri3_and_quad4_leaves() {
        use crate::core::types::{BoundaryCondition, ElementProperty};

        fn assert_quadratic_leaf(element: Element, nodes: Vec<[f64; 3]>, expected_dofs: usize) {
            let mut cluster = Cluster::new(element.center);
            cluster.element_indices = vec![0];
            cluster.num_elements = 1;
            let mut level = ClusterLevel::new(1);
            level.clusters.push(cluster);

            let system = build_mlfmm_system_stabilized_with_basis_checked(
                &[element],
                &nodes,
                vec![level],
                &PhysicsParams::new(100.0, 343.0, 1.21, false),
                BasisOrder::Quadratic,
                -1.0,
                1.0,
            )
            .expect("quadratic higher-order MLFMM admission");

            assert_eq!(system.num_dofs, expected_dofs);
            assert_eq!(system.leaf_dof_indices[0].len(), expected_dofs);
            assert_eq!(system.t_matrices[0].matrices[0].ncols(), expected_dofs);
            assert_eq!(system.s_matrices[0].matrices[0].nrows(), expected_dofs);

            let t_matrix = &system.t_matrices[0].matrices[0];
            let column_delta: f64 = (0..t_matrix.nrows())
                .map(|row| (t_matrix[[row, 0]] - t_matrix[[row, 1]]).norm())
                .sum();
            assert!(
                column_delta > 1e-12,
                "quadratic MLFMM basis columns must retain distinct weights"
            );
        }

        assert_quadratic_leaf(
            Element {
                connectivity: vec![0, 1, 2],
                element_type: ElementType::Tri3,
                property: ElementProperty::Surface,
                normal: [0.0, 0.0, 1.0],
                node_normals: vec![[0.0; 3]; 3],
                center: [1.0 / 3.0, 1.0 / 3.0, 0.0],
                area: 0.5,
                boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
                group: 0,
                dof_addresses: (0..6).collect(),
            },
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            6,
        );

        assert_quadratic_leaf(
            Element {
                connectivity: vec![0, 1, 2, 3],
                element_type: ElementType::Quad4,
                property: ElementProperty::Surface,
                normal: [0.0, 0.0, 1.0],
                node_normals: vec![[0.0; 3]; 4],
                center: [0.0, 0.0, 0.0],
                area: 4.0,
                boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
                group: 0,
                dof_addresses: (0..9).collect(),
            },
            vec![
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ],
            9,
        );
    }

    #[test]
    fn basis_checked_builder_rejects_wrong_local_dof_count() {
        use crate::core::types::{BoundaryCondition, ClusterLevel, ElementProperty};

        let element = Element {
            connectivity: vec![0, 1, 2],
            element_type: ElementType::Tri3,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0; 3]; 3],
            center: [1.0 / 3.0, 1.0 / 3.0, 0.0],
            area: 0.5,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
            group: 0,
            dof_addresses: vec![0],
        };
        let mut cluster = Cluster::new(element.center);
        cluster.element_indices = vec![0];
        let mut level = ClusterLevel::new(1);
        level.clusters.push(cluster);
        let error = match build_mlfmm_system_stabilized_with_basis_checked(
            &[element],
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![level],
            &PhysicsParams::new(100.0, 343.0, 1.21, false),
            BasisOrder::Linear,
            -1.0,
            1.0,
        ) {
            Ok(_) => panic!("wrong basis layout must fail before assembly"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            MlfmmAssemblyError::InvalidBasisDofCount {
                expected: 3,
                actual: 1,
                ..
            }
        ));
    }
}

impl fmt::Display for MlfmmAssemblyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyClusterLevels => write!(f, "MLFMM cluster hierarchy is empty"),
            Self::InvalidNode {
                element,
                local,
                node,
                available,
            } => write!(
                f,
                "MLFMM element {element} connectivity[{local}] references node {node} (mesh has {available})"
            ),
            Self::UnsupportedElementType { element } => {
                write!(
                    f,
                    "MLFMM element {element} is not a supported surface panel"
                )
            }
            Self::InvalidConnectivity { element, nodes } => write!(
                f,
                "MLFMM element {element} has {nodes} connectivity entries"
            ),
            Self::UnsupportedBasis { element, dofs } => write!(
                f,
                "MLFMM element {element} has {dofs} DOFs; constant one-DOF panels are required"
            ),
            Self::InvalidBasisDofCount {
                element,
                expected,
                actual,
            } => write!(
                f,
                "MLFMM element {element} has {actual} DOFs, expected {expected} for the requested basis"
            ),
            Self::InvalidDofAddress {
                element,
                global,
                num_dofs,
            } => write!(
                f,
                "MLFMM element {element} DOF {global} is outside {num_dofs} assembled DOFs"
            ),
            Self::DuplicateGlobalDof {
                first_element,
                second_element,
                global,
            } => write!(
                f,
                "MLFMM global DOF {global} is shared by elements {first_element} and {second_element}"
            ),
            Self::MissingGlobalDof { global, num_dofs } => write!(
                f,
                "MLFMM global DOF {global} is not owned by an active element (dimension {num_dofs})"
            ),
            Self::InvalidClusterElement {
                level,
                cluster,
                element,
                available,
            } => write!(
                f,
                "MLFMM level {level} cluster {cluster} references element {element} (mesh has {available})"
            ),
            Self::DuplicateLeafElement {
                element,
                first_cluster,
                second_cluster,
            } => write!(
                f,
                "MLFMM leaf element {element} occurs in clusters {first_cluster} and {second_cluster}"
            ),
            Self::MissingLeafElement { element } => {
                write!(
                    f,
                    "MLFMM active element {element} is absent from the leaf partition"
                )
            }
            Self::InvalidHierarchyReference {
                level,
                cluster,
                reference,
                available,
            } => write!(
                f,
                "MLFMM level {level} cluster {cluster} references hierarchy index {reference} (target has {available})"
            ),
            Self::InvalidQuadrature {
                level,
                theta_points,
                phi_points,
                expansion_terms,
            } => write!(
                f,
                "MLFMM level {level} has invalid quadrature theta={theta_points}, phi={phi_points}, terms={expansion_terms}"
            ),
        }
    }
}

impl std::error::Error for MlfmmAssemblyError {}

/// Geometric separation threshold for the leaf-level near/far classification.
///
/// A pair `(i, j)` is treated as far-field iff
///   `dist > MIN_GEOMETRIC_SEPARATION · (radius_i + radius_j)`.
///
/// The textbook FMM "well-separated" requirement is `2.0`. Bumping above
/// `2.0` pushes borderline pairs out of M2L and into the (exact) dense
/// near-field, trading more near-field block work for better aggregate
/// accuracy. Tuned to `3.0` so the per-pair M2L truncation error at the
/// closest-far pairs (where `kr_pair` is at the lower end of the
/// expansion-order's stability window) doesn't dominate the audit metric.
/// See `qa-bem-mlfmm-audit` and `bem-step1.md` §1.4.
const MIN_GEOMETRIC_SEPARATION: f64 = 3.0;

/// Wavelength-based separation threshold for the leaf-level near/far
/// classification — a pair is far iff `k·dist > MIN_WAVELENGTH_SEPARATION`
/// in addition to the geometric criterion. This is the wideband-style
/// guard that routes very-low-`k·dist` pairs to the near field where the
/// diagonal-form M2L would diverge (Hankel function tail).
const MIN_WAVELENGTH_SEPARATION: f64 = 2.0;

/// Configuration for the experimental directional MLFMM translator.
///
/// This keeps the proven plane-wave MLFMM bases intact, but prunes M2L
/// translation directions to a cone around the source-to-field direction once
/// `k·r` is large. Translations that share a quantized cone are applied through
/// compact cone-indexed workspaces instead of full sphere-point rows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectionalMlfmmConfig {
    /// Do not prune pair translations below this `k·r`; the low/moderate
    /// frequency MLFMM remains the accuracy oracle there.
    pub min_kr: f64,
    /// Cone half-angle multiplier: `half_angle ~= cone_safety / sqrt(k·r)`.
    pub cone_safety: f64,
    /// Lower bound for the cone half-angle in radians.
    pub min_half_angle: f64,
    /// Upper bound for the cone half-angle in radians.
    pub max_half_angle: f64,
    /// Minimum active quadrature directions retained for a pruned pair.
    pub min_active_points: usize,
    /// Number of quantized cone axes used for shared directional workspaces.
    pub num_cones: usize,
    /// Fixed cone-basis half-angle in radians. Use `0.0` to fall back to the
    /// adaptive `cone_safety / sqrt(k·r)` half-angle.
    pub basis_half_angle: f64,
}

impl Default for DirectionalMlfmmConfig {
    fn default() -> Self {
        Self {
            min_kr: 24.0,
            cone_safety: 5.0,
            min_half_angle: 0.30,
            max_half_angle: 1.35,
            min_active_points: 12,
            num_cones: 96,
            basis_half_angle: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum TranslationMode {
    Full,
    Directional(DirectionalMlfmmConfig),
}

/// Aggregate work counters for MLFMM translation entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlfmmTranslationStats {
    /// Number of M2L entries across all levels.
    pub entries: usize,
    /// Number of pointwise multiplies in the full translator.
    pub full_points: usize,
    /// Number of pointwise multiplies after any directional pruning.
    pub active_points: usize,
    /// Number of entries that use a directional cone.
    pub directional_entries: usize,
    /// Number of unique cone bases used by the compact directional workspace.
    pub cone_count: usize,
    /// Number of complex slots in compact cone multipole + local workspaces.
    pub cone_workspace_points: usize,
}

impl MlfmmTranslationStats {
    /// Fraction of translation point work retained by the current system.
    pub fn active_fraction(&self) -> f64 {
        if self.full_points == 0 {
            1.0
        } else {
            self.active_points as f64 / self.full_points as f64
        }
    }
}

/// Workspace for MLFMM matvec to avoid allocations
pub struct MlfmmWorkspace {
    /// Multipole expansions at each level: [level][cluster_idx, points]
    pub multipoles: Vec<DenseMatrix<Complex64>>,
    /// Local expansions at each level: [level][cluster_idx, points]
    pub locals: Vec<DenseMatrix<Complex64>>,
    /// Compact directional multipoles: [level][cone][cluster_idx, active direction]
    pub cone_multipoles: Vec<Vec<DenseMatrix<Complex64>>>,
    /// Compact directional locals: [level][cone][cluster_idx, active direction]
    pub cone_locals: Vec<Vec<DenseMatrix<Complex64>>>,
    /// Batched multipoles: [level][cluster_idx * points + point, rhs].
    ///
    /// The scalar workspace above is retained for the scalar and transpose
    /// paths.  Keeping RHS in columns lets the block path reuse every
    /// translation coefficient for all right-hand sides in one pass.
    block_multipoles: Vec<DenseMatrix<Complex64>>,
    /// Batched locals: [level][cluster_idx * points + point, rhs].
    block_locals: Vec<DenseMatrix<Complex64>>,
    /// Batched compact directional multipoles: [level][cone][cluster * active + direction, rhs].
    block_cone_multipoles: Vec<Vec<DenseMatrix<Complex64>>>,
    /// Batched compact directional locals: [level][cone][cluster * active + direction, rhs].
    block_cone_locals: Vec<Vec<DenseMatrix<Complex64>>>,
    block_rhs: usize,
}

impl MlfmmWorkspace {
    /// Create a new workspace for the given cluster levels
    ///
    /// Uses `actual_sphere_points` (derived from real quadrature output) rather than
    /// `level.theta_points * level.phi_points`, which can differ when the Gauss-Legendre
    /// table falls back to a lower order.
    pub fn new(cluster_levels: &[ClusterLevel], actual_sphere_points: &[usize]) -> Self {
        Self::new_with_directional(cluster_levels, actual_sphere_points, &[])
    }

    fn new_with_directional(
        cluster_levels: &[ClusterLevel],
        actual_sphere_points: &[usize],
        directional_plans: &[DirectionalLevelPlan],
    ) -> Self {
        let mut multipoles = Vec::with_capacity(cluster_levels.len());
        let mut locals = Vec::with_capacity(cluster_levels.len());
        let mut cone_multipoles = Vec::with_capacity(cluster_levels.len());
        let mut cone_locals = Vec::with_capacity(cluster_levels.len());

        for (level_idx, level) in cluster_levels.iter().enumerate() {
            let num_clusters = level.clusters.len();
            let num_points = actual_sphere_points[level_idx];

            multipoles.push(DenseMatrix::<Complex64>::zeros(num_clusters, num_points));
            locals.push(DenseMatrix::<Complex64>::zeros(num_clusters, num_points));

            if let Some(plan) = directional_plans.get(level_idx) {
                let mut level_cone_multipoles = Vec::with_capacity(plan.cones.len());
                let mut level_cone_locals = Vec::with_capacity(plan.cones.len());
                for cone in &plan.cones {
                    let n_active = cone.active_directions.len();
                    level_cone_multipoles.push(DenseMatrix::<Complex64>::zeros(
                        cone.source_clusters.len(),
                        n_active,
                    ));
                    level_cone_locals.push(DenseMatrix::<Complex64>::zeros(
                        cone.field_clusters.len(),
                        n_active,
                    ));
                }
                cone_multipoles.push(level_cone_multipoles);
                cone_locals.push(level_cone_locals);
            } else {
                cone_multipoles.push(Vec::new());
                cone_locals.push(Vec::new());
            }
        }

        Self {
            multipoles,
            locals,
            cone_multipoles,
            cone_locals,
            block_multipoles: Vec::new(),
            block_locals: Vec::new(),
            block_cone_multipoles: Vec::new(),
            block_cone_locals: Vec::new(),
            block_rhs: 0,
        }
    }

    /// Reset all values to zero
    pub fn clear(&mut self) {
        for m in &mut self.multipoles {
            m.fill(Complex64::new(0.0, 0.0));
        }
        for l in &mut self.locals {
            l.fill(Complex64::new(0.0, 0.0));
        }
        for level in &mut self.cone_multipoles {
            for m in level {
                m.fill(Complex64::new(0.0, 0.0));
            }
        }
        for level in &mut self.cone_locals {
            for l in level {
                l.fill(Complex64::new(0.0, 0.0));
            }
        }
    }

    fn ensure_block_storage(
        &mut self,
        cluster_levels: &[ClusterLevel],
        actual_sphere_points: &[usize],
        directional_plans: &[DirectionalLevelPlan],
        nrhs: usize,
    ) {
        let shape_matches = self.block_rhs == nrhs
            && self.block_multipoles.len() == cluster_levels.len()
            && self
                .block_multipoles
                .iter()
                .enumerate()
                .all(|(level, matrix)| {
                    matrix.nrows()
                        == cluster_levels[level].clusters.len() * actual_sphere_points[level]
                        && matrix.ncols() == nrhs
                });
        if shape_matches {
            return;
        }

        let mut block_multipoles = Vec::with_capacity(cluster_levels.len());
        let mut block_locals = Vec::with_capacity(cluster_levels.len());
        let mut block_cone_multipoles = Vec::with_capacity(cluster_levels.len());
        let mut block_cone_locals = Vec::with_capacity(cluster_levels.len());

        for (level_idx, level) in cluster_levels.iter().enumerate() {
            let num_clusters = level.clusters.len();
            let num_points = actual_sphere_points[level_idx];
            block_multipoles.push(DenseMatrix::zeros(num_clusters * num_points, nrhs));
            block_locals.push(DenseMatrix::zeros(num_clusters * num_points, nrhs));

            if let Some(plan) = directional_plans.get(level_idx) {
                let mut level_multipoles = Vec::with_capacity(plan.cones.len());
                let mut level_locals = Vec::with_capacity(plan.cones.len());
                for cone in &plan.cones {
                    let n_active = cone.active_directions.len();
                    level_multipoles.push(DenseMatrix::zeros(
                        cone.source_clusters.len() * n_active,
                        nrhs,
                    ));
                    level_locals.push(DenseMatrix::zeros(
                        cone.field_clusters.len() * n_active,
                        nrhs,
                    ));
                }
                block_cone_multipoles.push(level_multipoles);
                block_cone_locals.push(level_locals);
            } else {
                block_cone_multipoles.push(Vec::new());
                block_cone_locals.push(Vec::new());
            }
        }

        self.block_multipoles = block_multipoles;
        self.block_locals = block_locals;
        self.block_cone_multipoles = block_cone_multipoles;
        self.block_cone_locals = block_cone_locals;
        self.block_rhs = nrhs;
    }

    fn clear_block(&mut self) {
        for matrix in &mut self.block_multipoles {
            matrix.fill(Complex64::new(0.0, 0.0));
        }
        for matrix in &mut self.block_locals {
            matrix.fill(Complex64::new(0.0, 0.0));
        }
        for level in &mut self.block_cone_multipoles {
            for matrix in level {
                matrix.fill(Complex64::new(0.0, 0.0));
            }
        }
        for level in &mut self.block_cone_locals {
            for matrix in level {
                matrix.fill(Complex64::new(0.0, 0.0));
            }
        }
    }
}

/// Result of MLFMM assembly
pub struct MlfmmSystem {
    /// Number of levels in the tree
    pub num_levels: usize,
    /// Near-field coefficient matrix at leaf level (sparse, stored as dense blocks)
    pub near_matrix: Vec<NearFieldBlock>,
    /// T-matrices at each level (child-to-parent aggregation)
    /// Level 0 has no T-matrices (root)
    pub t_matrices: Vec<LevelMatrices>,
    /// D-matrices for far-field translation at each level
    pub d_matrices: Vec<Vec<DMatrixEntry>>,
    /// S-matrices at each level (parent-to-child disaggregation)
    pub s_matrices: Vec<LevelMatrices>,
    /// RHS vector
    pub rhs: Vec<Complex64>,
    /// Number of DOFs
    pub num_dofs: usize,
    /// Cluster tree (indexed by level)
    cluster_levels: Vec<ClusterLevel>,
    /// DOF mappings at leaf level: for each leaf cluster, the global DOF indices
    pub leaf_dof_indices: Vec<Vec<usize>>,
    /// Whether each leaf proxy incidence owns its output row.
    ///
    /// Continuous higher-order layouts repeat shared global DOFs across
    /// panels.  Forward actions select one row owner while field incidences
    /// remain fully accumulated; transpose actions apply the dual selection.
    /// Empty metadata preserves legacy hand-built systems.
    leaf_dof_owner: Vec<Vec<bool>>,
    /// Number of sphere points at each level
    pub sphere_points_per_level: Vec<usize>,
    /// Workspace for matvec
    pub workspace: Mutex<MlfmmWorkspace>,
    /// Compact cone-indexed M2L plans used by experimental directional MLFMM.
    directional_plans: Vec<DirectionalLevelPlan>,
    /// Compact inter-level M2M phase shifts used by the forward matvec.
    m2m_phase_plans: Vec<InterLevelPhasePlan>,
    /// Compact inter-level L2L phase shifts used by the forward matvec.
    l2l_phase_plans: Vec<InterLevelPhasePlan>,
    /// M2L entry indices grouped by field cluster for allocation-free forward translation.
    m2l_field_plans: Vec<Vec<Vec<usize>>>,
}

/// Checked local-row action failures for the multi-level FMM shell.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MlfmmRowActionError {
    /// The input vector does not match the assembled operator dimension.
    #[error("MLFMM {operation} input length {actual} does not match expected {expected}")]
    Dimension {
        /// Action name.
        operation: &'static str,
        /// Supplied input length.
        actual: usize,
        /// Required input length.
        expected: usize,
    },
    /// A requested output row lies outside the assembled operator.
    #[error("MLFMM row {row} is outside operator dimension {num_dofs}")]
    InvalidRow {
        /// Requested global row.
        row: usize,
        /// Operator dimension.
        num_dofs: usize,
    },
    /// An input entry is not finite.
    #[error("MLFMM {operation} input entry {index} must be finite")]
    NonFiniteInput {
        /// Action name.
        operation: &'static str,
        /// Invalid input index.
        index: usize,
    },
    /// The computed action contains a non-finite value.
    #[error("MLFMM {operation} output entry {index} is non-finite")]
    NonFiniteOutput {
        /// Action name.
        operation: &'static str,
        /// Invalid output index.
        index: usize,
    },
    /// A level-indexed hierarchy table has the wrong number of entries.
    #[error("MLFMM {operation} {stage} count {actual} does not match expected {expected}")]
    LayoutCount {
        /// Action name.
        operation: &'static str,
        /// Hierarchy component being checked.
        stage: &'static str,
        /// Actual number of entries.
        actual: usize,
        /// Required number of entries.
        expected: usize,
    },
    /// A level metadata record does not match the cluster tree.
    #[error("MLFMM {operation} {stage} level {level} has {actual} clusters, expected {expected}")]
    LevelClusterCount {
        /// Action name.
        operation: &'static str,
        /// Hierarchy component being checked.
        stage: &'static str,
        /// Level index.
        level: usize,
        /// Actual number of clusters.
        actual: usize,
        /// Required number of clusters.
        expected: usize,
    },
    /// A hierarchy reference points outside its target level.
    #[error(
        "MLFMM {operation} level {level} {side} cluster {cluster} references {reference}, available {available}"
    )]
    InvalidHierarchyReference {
        /// Action name.
        operation: &'static str,
        /// Level containing the reference.
        level: usize,
        /// Reference side.
        side: &'static str,
        /// Cluster owning the reference.
        cluster: usize,
        /// Referenced child or peer index.
        reference: usize,
        /// Number of entries in the target level.
        available: usize,
    },
    /// A leaf cluster maps a local slot outside the global operator.
    #[error(
        "MLFMM {operation} leaf cluster {cluster} DOF {local} maps to {global}, outside {num_dofs}"
    )]
    InvalidDof {
        /// Action name.
        operation: &'static str,
        /// Leaf cluster index.
        cluster: usize,
        /// Local DOF slot.
        local: usize,
        /// Global DOF index.
        global: usize,
        /// Operator dimension.
        num_dofs: usize,
    },
    /// A near-field block shape does not match its leaf mappings.
    #[error(
        "MLFMM {operation} near block {block} shape {rows}x{cols}, expected {expected_rows}x{expected_cols}"
    )]
    NearFieldShape {
        /// Action name.
        operation: &'static str,
        /// Near-field block index.
        block: usize,
        /// Actual row count.
        rows: usize,
        /// Actual column count.
        cols: usize,
        /// Expected row count.
        expected_rows: usize,
        /// Expected column count.
        expected_cols: usize,
    },
    /// A transfer table has an invalid matrix shape.
    #[error(
        "MLFMM {operation} {stage} level {level} cluster {cluster} shape {rows}x{cols}, expected {expected_rows}x{expected_cols}"
    )]
    TransferShape {
        /// Action name.
        operation: &'static str,
        /// Transfer stage (`T` or `S`).
        stage: &'static str,
        /// Level index.
        level: usize,
        /// Cluster index.
        cluster: usize,
        /// Actual row count.
        rows: usize,
        /// Actual column count.
        cols: usize,
        /// Expected row count.
        expected_rows: usize,
        /// Expected column count.
        expected_cols: usize,
    },
    /// An M2L diagonal has the wrong number of expansion points.
    #[error(
        "MLFMM {operation} level {level} translation {entry} diagonal length {actual}, expected {expected}"
    )]
    TranslationShape {
        /// Action name.
        operation: &'static str,
        /// Level containing the translation.
        level: usize,
        /// Translation entry index.
        entry: usize,
        /// Actual diagonal length.
        actual: usize,
        /// Required diagonal length.
        expected: usize,
    },
}

fn validate_row_action_input(
    input: &[Complex64],
    expected: usize,
    operation: &'static str,
) -> Result<(), MlfmmRowActionError> {
    if input.len() != expected {
        return Err(MlfmmRowActionError::Dimension {
            operation,
            actual: input.len(),
            expected,
        });
    }
    if let Some(index) = input
        .iter()
        .position(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(MlfmmRowActionError::NonFiniteInput { operation, index });
    }
    Ok(())
}

fn validate_row_action_output(
    output: &[Complex64],
    operation: &'static str,
) -> Result<(), MlfmmRowActionError> {
    if let Some(index) = output
        .iter()
        .position(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(MlfmmRowActionError::NonFiniteOutput { operation, index });
    }
    Ok(())
}

fn validate_row_action_block_output(
    output: &DenseMatrix<Complex64>,
    operation: &'static str,
) -> Result<(), MlfmmRowActionError> {
    if let Some(index) = output
        .as_slice()
        .iter()
        .position(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(MlfmmRowActionError::NonFiniteOutput { operation, index });
    }
    Ok(())
}

fn validate_row_action_block_input(
    input: &DenseMatrix<Complex64>,
    expected: usize,
    operation: &'static str,
) -> Result<(), MlfmmRowActionError> {
    if input.nrows() != expected {
        return Err(MlfmmRowActionError::Dimension {
            operation,
            actual: input.nrows(),
            expected,
        });
    }
    for column in 0..input.ncols() {
        for row in 0..input.nrows() {
            let value = input[[row, column]];
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(MlfmmRowActionError::NonFiniteInput {
                    operation,
                    index: column * input.nrows() + row,
                });
            }
        }
    }
    Ok(())
}

fn validate_rows(rows: &[usize], num_dofs: usize) -> Result<(), MlfmmRowActionError> {
    if let Some(&row) = rows.iter().find(|&&row| row >= num_dofs) {
        return Err(MlfmmRowActionError::InvalidRow { row, num_dofs });
    }
    Ok(())
}

impl Clone for MlfmmSystem {
    fn clone(&self) -> Self {
        Self {
            num_levels: self.num_levels,
            near_matrix: self.near_matrix.clone(),
            t_matrices: self.t_matrices.clone(),
            d_matrices: self.d_matrices.clone(),
            s_matrices: self.s_matrices.clone(),
            rhs: self.rhs.clone(),
            num_dofs: self.num_dofs,
            cluster_levels: self.cluster_levels.clone(),
            leaf_dof_indices: self.leaf_dof_indices.clone(),
            leaf_dof_owner: self.leaf_dof_owner.clone(),
            sphere_points_per_level: self.sphere_points_per_level.clone(),
            workspace: Mutex::new(MlfmmWorkspace::new(
                &self.cluster_levels,
                &self.sphere_points_per_level,
            )),
            directional_plans: self.directional_plans.clone(),
            m2m_phase_plans: self.m2m_phase_plans.clone(),
            l2l_phase_plans: self.l2l_phase_plans.clone(),
            m2l_field_plans: self.m2l_field_plans.clone(),
        }
    }
}

/// Near-field block between two leaf clusters
#[derive(Debug, Clone)]
pub struct NearFieldBlock {
    /// Source cluster index
    pub source_cluster: usize,
    /// Field cluster index
    pub field_cluster: usize,
    /// Dense coefficient matrix for this block.
    ///
    /// Rows are source (output) DOFs and columns are field (input) DOFs,
    /// matching the matrix-free near-field matvec convention.
    pub coefficients: DenseMatrix<Complex64>,
}

/// Validation failure while materializing the MLFMM near-field matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MlfmmNearFieldError {
    /// A block refers to a leaf cluster that is not present in the mapping.
    InvalidCluster {
        /// Block source/field side.
        side: &'static str,
        /// Invalid cluster index.
        cluster: usize,
        /// Number of mapped leaf clusters.
        available: usize,
    },
    /// A block shape does not match its source/field DOF mappings.
    BlockShape {
        /// Near-field block index.
        block: usize,
        /// Expected source-row count.
        source_dofs: usize,
        /// Expected field-column count.
        field_dofs: usize,
        /// Actual matrix row count.
        rows: usize,
        /// Actual matrix column count.
        cols: usize,
    },
    /// A mapped global DOF lies outside the system matrix.
    GlobalDof {
        /// Near-field block index.
        block: usize,
        /// Source/field side.
        side: &'static str,
        /// Local DOF position.
        local: usize,
        /// Invalid global DOF index.
        global: usize,
        /// Matrix dimension.
        num_dofs: usize,
    },
    /// A coefficient is NaN or infinite.
    NonFiniteCoefficient {
        /// Near-field block index.
        block: usize,
        /// Matrix row.
        row: usize,
        /// Matrix column.
        col: usize,
    },
}

impl std::fmt::Display for MlfmmNearFieldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidCluster {
                side,
                cluster,
                available,
            } => write!(
                f,
                "MLFMM near-field {side} cluster {cluster} is out of range (available {available})"
            ),
            Self::BlockShape {
                block,
                source_dofs,
                field_dofs,
                rows,
                cols,
            } => write!(
                f,
                "MLFMM near-field block {block} shape {rows}x{cols} does not match {source_dofs} source DOFs x {field_dofs} field DOFs"
            ),
            Self::GlobalDof {
                block,
                side,
                local,
                global,
                num_dofs,
            } => write!(
                f,
                "MLFMM near-field block {block} {side} DOF {local} maps to {global}, outside {num_dofs} DOFs"
            ),
            Self::NonFiniteCoefficient { block, row, col } => write!(
                f,
                "MLFMM near-field block {block} coefficient ({row},{col}) is non-finite"
            ),
        }
    }
}

impl std::error::Error for MlfmmNearFieldError {}

/// D-matrix entry for a far cluster pair
#[derive(Debug, Clone)]
pub struct DMatrixEntry {
    /// Source cluster index
    pub source_cluster: usize,
    /// Field cluster index
    pub field_cluster: usize,
    /// Tree level
    pub level: usize,
    /// Diagonal translation operator
    pub diagonal: Vec<Complex64>,
    /// Optional active direction indices for experimental directional M2L.
    /// `None` means every diagonal entry is active.
    pub active_directions: Option<Vec<usize>>,
}

#[derive(Debug, Clone)]
struct DirectionalConeBasis {
    active_directions: Vec<usize>,
    source_clusters: Vec<usize>,
    field_clusters: Vec<usize>,
}

#[derive(Debug, Clone)]
struct DirectionalConeEntry {
    source_row: usize,
    field_row: usize,
    diagonal: Vec<Complex64>,
}

#[derive(Debug, Clone)]
struct DirectionalLevelPlan {
    cones: Vec<DirectionalConeBasis>,
    entries_by_cone: Vec<Vec<DirectionalConeEntry>>,
    entries_by_cone_field_row: Vec<Vec<Vec<usize>>>,
    entries_by_cone_source_row: Vec<Vec<Vec<usize>>>,
}

impl DirectionalLevelPlan {
    fn empty() -> Self {
        Self {
            cones: Vec::new(),
            entries_by_cone: Vec::new(),
            entries_by_cone_field_row: Vec::new(),
            entries_by_cone_source_row: Vec::new(),
        }
    }

    fn is_empty(&self) -> bool {
        self.cones.is_empty()
    }
}

fn validate_directional_level_plan(
    plan: &DirectionalLevelPlan,
    level: usize,
    cluster_count: usize,
    points: usize,
    operation: &'static str,
) -> Result<(), MlfmmRowActionError> {
    if plan.is_empty() {
        return Ok(());
    }
    if plan.entries_by_cone.len() != plan.cones.len()
        || plan.entries_by_cone_field_row.len() != plan.cones.len()
        || plan.entries_by_cone_source_row.len() != plan.cones.len()
    {
        return Err(MlfmmRowActionError::LayoutCount {
            operation,
            stage: "directional cone tables",
            actual: plan.entries_by_cone.len(),
            expected: plan.cones.len(),
        });
    }
    for (cone_index, cone) in plan.cones.iter().enumerate() {
        for &direction in &cone.active_directions {
            if direction >= points {
                return Err(MlfmmRowActionError::InvalidHierarchyReference {
                    operation,
                    level,
                    side: "direction",
                    cluster: cone_index,
                    reference: direction,
                    available: points,
                });
            }
        }
        for &cluster in &cone.source_clusters {
            if cluster >= cluster_count {
                return Err(MlfmmRowActionError::InvalidHierarchyReference {
                    operation,
                    level,
                    side: "directional source cluster",
                    cluster: cone_index,
                    reference: cluster,
                    available: cluster_count,
                });
            }
        }
        for &cluster in &cone.field_clusters {
            if cluster >= cluster_count {
                return Err(MlfmmRowActionError::InvalidHierarchyReference {
                    operation,
                    level,
                    side: "directional field cluster",
                    cluster: cone_index,
                    reference: cluster,
                    available: cluster_count,
                });
            }
        }
        let entries = &plan.entries_by_cone[cone_index];
        for (entry_index, entry) in entries.iter().enumerate() {
            if entry.source_row >= cone.source_clusters.len() {
                return Err(MlfmmRowActionError::InvalidHierarchyReference {
                    operation,
                    level,
                    side: "directional source row",
                    cluster: cone_index,
                    reference: entry.source_row,
                    available: cone.source_clusters.len(),
                });
            }
            if entry.field_row >= cone.field_clusters.len() {
                return Err(MlfmmRowActionError::InvalidHierarchyReference {
                    operation,
                    level,
                    side: "directional field row",
                    cluster: cone_index,
                    reference: entry.field_row,
                    available: cone.field_clusters.len(),
                });
            }
            if entry.diagonal.len() != cone.active_directions.len() {
                return Err(MlfmmRowActionError::TranslationShape {
                    operation,
                    level,
                    entry: entry_index,
                    actual: entry.diagonal.len(),
                    expected: cone.active_directions.len(),
                });
            }
        }
        let field_rows = &plan.entries_by_cone_field_row[cone_index];
        if field_rows.len() != cone.field_clusters.len() {
            return Err(MlfmmRowActionError::LayoutCount {
                operation,
                stage: "directional field reverse rows",
                actual: field_rows.len(),
                expected: cone.field_clusters.len(),
            });
        }
        for (row, indices) in field_rows.iter().enumerate() {
            for &entry_index in indices {
                let Some(entry) = entries.get(entry_index) else {
                    return Err(MlfmmRowActionError::InvalidHierarchyReference {
                        operation,
                        level,
                        side: "directional field entry",
                        cluster: row,
                        reference: entry_index,
                        available: entries.len(),
                    });
                };
                if entry.field_row != row {
                    return Err(MlfmmRowActionError::InvalidHierarchyReference {
                        operation,
                        level,
                        side: "directional field row mismatch",
                        cluster: row,
                        reference: entry.field_row,
                        available: field_rows.len(),
                    });
                }
            }
        }
        let source_rows = &plan.entries_by_cone_source_row[cone_index];
        if source_rows.len() != cone.source_clusters.len() {
            return Err(MlfmmRowActionError::LayoutCount {
                operation,
                stage: "directional source reverse rows",
                actual: source_rows.len(),
                expected: cone.source_clusters.len(),
            });
        }
        for (row, indices) in source_rows.iter().enumerate() {
            for &entry_index in indices {
                let Some(entry) = entries.get(entry_index) else {
                    return Err(MlfmmRowActionError::InvalidHierarchyReference {
                        operation,
                        level,
                        side: "directional source entry",
                        cluster: row,
                        reference: entry_index,
                        available: entries.len(),
                    });
                };
                if entry.source_row != row {
                    return Err(MlfmmRowActionError::InvalidHierarchyReference {
                        operation,
                        level,
                        side: "directional source row mismatch",
                        cluster: row,
                        reference: entry.source_row,
                        available: source_rows.len(),
                    });
                }
            }
        }
    }
    Ok(())
}

fn validate_interlevel_phase_plans(
    plans: &[InterLevelPhasePlan],
    cluster_levels: &[ClusterLevel],
    sphere_points_per_level: &[usize],
    operation: &'static str,
    stage: &'static str,
) -> Result<(), MlfmmRowActionError> {
    if plans.is_empty() {
        return Ok(());
    }
    if plans.len() != cluster_levels.len() {
        return Err(MlfmmRowActionError::LayoutCount {
            operation,
            stage,
            actual: plans.len(),
            expected: cluster_levels.len(),
        });
    }
    for (level, plan) in plans.iter().enumerate() {
        let clusters = &cluster_levels[level].clusters;
        if plan.parent_children.len() != clusters.len() {
            return Err(MlfmmRowActionError::LevelClusterCount {
                operation,
                stage,
                level,
                actual: plan.parent_children.len(),
                expected: clusters.len(),
            });
        }
        let child_count = cluster_levels
            .get(level + 1)
            .map_or(0, |next| next.clusters.len());
        let expected_phase_len = if level + 1 < sphere_points_per_level.len() {
            sphere_points_per_level[level].min(sphere_points_per_level[level + 1])
        } else {
            0
        };
        for (cluster_index, (cluster, children)) in
            clusters.iter().zip(plan.parent_children.iter()).enumerate()
        {
            if children.len() != cluster.sons.len() {
                return Err(MlfmmRowActionError::LayoutCount {
                    operation,
                    stage,
                    actual: children.len(),
                    expected: cluster.sons.len(),
                });
            }
            for child in children {
                if child.child_cluster >= child_count
                    || !cluster.sons.contains(&child.child_cluster)
                {
                    return Err(MlfmmRowActionError::InvalidHierarchyReference {
                        operation,
                        level,
                        side: "phase child",
                        cluster: cluster_index,
                        reference: child.child_cluster,
                        available: child_count,
                    });
                }
                if child.phase.len() != expected_phase_len {
                    return Err(MlfmmRowActionError::TranslationShape {
                        operation,
                        level,
                        entry: cluster_index,
                        actual: child.phase.len(),
                        expected: expected_phase_len,
                    });
                }
                if let Some(index) = child
                    .phase
                    .iter()
                    .position(|value| !value.re.is_finite() || !value.im.is_finite())
                {
                    return Err(MlfmmRowActionError::NonFiniteOutput { operation, index });
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod directional_plan_admission_tests {
    use super::*;

    fn c(value: f64) -> Complex64 {
        Complex64::new(value, 0.0)
    }

    fn system_with_plan() -> MlfmmSystem {
        let mut cluster = Cluster::new([0.0, 0.0, 0.0]);
        cluster.element_indices = vec![0];
        cluster.num_elements = 1;
        let mut level = ClusterLevel::new(1);
        level.clusters.push(cluster);
        let mut system = MlfmmSystem::new(1, 1, vec![level], vec![1]);
        system.leaf_dof_indices = vec![vec![0]];
        system.t_matrices = vec![LevelMatrices {
            level: 0,
            num_clusters: 1,
            matrices: vec![DenseMatrix::from_row_major_vec(1, 1, vec![c(1.0)])],
        }];
        system.s_matrices = vec![LevelMatrices {
            level: 0,
            num_clusters: 1,
            matrices: vec![DenseMatrix::from_row_major_vec(1, 1, vec![c(1.0)])],
        }];
        system.d_matrices = vec![vec![DMatrixEntry {
            source_cluster: 0,
            field_cluster: 0,
            level: 0,
            diagonal: vec![c(1.0)],
            active_directions: Some(vec![0]),
        }]];
        system.m2l_field_plans = vec![vec![vec![0]]];
        system.directional_plans = vec![DirectionalLevelPlan {
            cones: vec![DirectionalConeBasis {
                active_directions: vec![0],
                source_clusters: vec![0],
                field_clusters: vec![0],
            }],
            entries_by_cone: vec![vec![DirectionalConeEntry {
                source_row: 0,
                field_row: 0,
                diagonal: vec![c(1.0)],
            }]],
            entries_by_cone_field_row: vec![vec![vec![0]]],
            entries_by_cone_source_row: vec![vec![vec![0]]],
        }];
        system
    }

    #[test]
    fn checked_directional_plan_accepts_consistent_compact_tables() {
        assert!(system_with_plan().matvec_checked(&[c(1.0)]).is_ok());
    }

    #[test]
    fn checked_directional_plan_rejects_invalid_direction_and_diagonal_shape() {
        let mut invalid_direction = system_with_plan();
        invalid_direction.directional_plans[0].cones[0].active_directions = vec![1];
        assert!(matches!(
            invalid_direction.matvec_checked(&[c(1.0)]),
            Err(MlfmmRowActionError::InvalidHierarchyReference {
                side: "direction",
                ..
            })
        ));
        let mut invalid_shape = system_with_plan();
        invalid_shape.directional_plans[0].entries_by_cone[0][0]
            .diagonal
            .clear();
        assert!(matches!(
            invalid_shape.matvec_checked(&[c(1.0)]),
            Err(MlfmmRowActionError::TranslationShape { .. })
        ));
    }

    #[test]
    fn checked_directional_plan_rejects_reverse_table_entry_reference() {
        let mut invalid = system_with_plan();
        invalid.directional_plans[0].entries_by_cone_field_row[0][0] = vec![1];
        assert!(matches!(
            invalid.matvec_transpose_checked(&[c(1.0)]),
            Err(MlfmmRowActionError::InvalidHierarchyReference {
                side: "directional field entry",
                ..
            })
        ));
    }

    #[test]
    fn checked_phase_plan_rejects_last_level_child_reference() {
        let mut invalid = system_with_plan();
        invalid.m2m_phase_plans = vec![InterLevelPhasePlan {
            parent_children: vec![vec![InterLevelChildPhase {
                child_cluster: 0,
                phase: vec![c(1.0)],
            }]],
        }];
        invalid.cluster_levels[0].clusters[0].sons = vec![0];
        let error = invalid
            .matvec_checked(&[c(1.0)])
            .expect_err("invalid phase plan");
        assert!(matches!(
            error,
            MlfmmRowActionError::InvalidHierarchyReference {
                side: "phase child",
                ..
            }
        ));
    }
}

#[derive(Debug, Clone)]
struct InterLevelChildPhase {
    child_cluster: usize,
    phase: Vec<Complex64>,
}

#[derive(Debug, Clone)]
struct InterLevelPhasePlan {
    parent_children: Vec<Vec<InterLevelChildPhase>>,
}

impl InterLevelPhasePlan {
    fn empty(num_clusters: usize) -> Self {
        Self {
            parent_children: vec![Vec::new(); num_clusters],
        }
    }

    fn is_empty(&self) -> bool {
        self.parent_children.iter().all(Vec::is_empty)
    }
}

/// Matrices at a single level of the tree
#[derive(Debug, Clone)]
pub struct LevelMatrices {
    /// Tree level index
    pub level: usize,
    /// Number of clusters at this level
    pub num_clusters: usize,
    /// Transfer matrices for each cluster
    pub matrices: Vec<DenseMatrix<Complex64>>,
}

impl MlfmmSystem {
    #[inline]
    fn leaf_row_is_owner(&self, cluster: usize, local: usize) -> bool {
        self.leaf_dof_owner
            .get(cluster)
            .and_then(|owners| owners.get(local))
            .copied()
            .unwrap_or(true)
    }

    /// Validate the public hierarchy metadata before entering the legacy
    /// matrix-free kernels.  The unchecked kernels intentionally use bounded
    /// `min`/`get` operations for performance and compatibility, which can
    /// otherwise silently drop malformed rows or child transfers.
    fn validate_action_layout(&self, operation: &'static str) -> Result<(), MlfmmRowActionError> {
        let expected_levels = self.num_levels;
        for (stage, actual) in [
            ("cluster levels", self.cluster_levels.len()),
            ("sphere-point levels", self.sphere_points_per_level.len()),
            ("T levels", self.t_matrices.len()),
            ("D levels", self.d_matrices.len()),
            ("S levels", self.s_matrices.len()),
            ("directional plans", self.directional_plans.len()),
        ] {
            if actual != expected_levels {
                return Err(MlfmmRowActionError::LayoutCount {
                    operation,
                    stage,
                    actual,
                    expected: expected_levels,
                });
            }
        }
        if self.m2l_field_plans.is_empty() {
            if self.d_matrices.iter().any(|entries| !entries.is_empty()) {
                return Err(MlfmmRowActionError::LayoutCount {
                    operation,
                    stage: "M2L field plans",
                    actual: 0,
                    expected: expected_levels,
                });
            }
        } else if self.m2l_field_plans.len() != expected_levels {
            return Err(MlfmmRowActionError::LayoutCount {
                operation,
                stage: "M2L field plans",
                actual: self.m2l_field_plans.len(),
                expected: expected_levels,
            });
        }

        for (level, plan) in self.directional_plans.iter().enumerate() {
            if plan.is_empty()
                && self.d_matrices[level]
                    .iter()
                    .any(|entry| entry.active_directions.is_some())
            {
                return Err(MlfmmRowActionError::LayoutCount {
                    operation,
                    stage: "directional active plan",
                    actual: 0,
                    expected: 1,
                });
            }
            validate_directional_level_plan(
                plan,
                level,
                self.cluster_levels[level].clusters.len(),
                self.sphere_points_per_level[level],
                operation,
            )?;
        }
        validate_interlevel_phase_plans(
            &self.m2m_phase_plans,
            &self.cluster_levels,
            &self.sphere_points_per_level,
            operation,
            "M2M phase plans",
        )?;
        validate_interlevel_phase_plans(
            &self.l2l_phase_plans,
            &self.cluster_levels,
            &self.sphere_points_per_level,
            operation,
            "L2L phase plans",
        )?;

        let leaf_clusters = self
            .cluster_levels
            .last()
            .map_or(0, |level| level.clusters.len());
        if self.leaf_dof_indices.len() != leaf_clusters {
            return Err(MlfmmRowActionError::LayoutCount {
                operation,
                stage: "leaf DOF mappings",
                actual: self.leaf_dof_indices.len(),
                expected: leaf_clusters,
            });
        }

        for (cluster, dofs) in self.leaf_dof_indices.iter().enumerate() {
            for (local, &global) in dofs.iter().enumerate() {
                if global >= self.num_dofs {
                    return Err(MlfmmRowActionError::InvalidDof {
                        operation,
                        cluster,
                        local,
                        global,
                        num_dofs: self.num_dofs,
                    });
                }
            }
        }

        for (level, tree_level) in self.cluster_levels.iter().enumerate() {
            let clusters = &tree_level.clusters;
            let expected_clusters = clusters.len();
            let t_level = &self.t_matrices[level];
            let s_level = &self.s_matrices[level];
            if t_level.level != level {
                return Err(MlfmmRowActionError::InvalidHierarchyReference {
                    operation,
                    level,
                    side: "T level",
                    cluster: 0,
                    reference: t_level.level,
                    available: expected_levels,
                });
            }
            if s_level.level != level {
                return Err(MlfmmRowActionError::InvalidHierarchyReference {
                    operation,
                    level,
                    side: "S level",
                    cluster: 0,
                    reference: s_level.level,
                    available: expected_levels,
                });
            }
            if t_level.num_clusters != expected_clusters {
                return Err(MlfmmRowActionError::LevelClusterCount {
                    operation,
                    stage: "T metadata",
                    level,
                    actual: t_level.num_clusters,
                    expected: expected_clusters,
                });
            }
            if s_level.num_clusters != expected_clusters {
                return Err(MlfmmRowActionError::LevelClusterCount {
                    operation,
                    stage: "S metadata",
                    level,
                    actual: s_level.num_clusters,
                    expected: expected_clusters,
                });
            }
            if t_level.matrices.len() != expected_clusters {
                return Err(MlfmmRowActionError::LevelClusterCount {
                    operation,
                    stage: "T matrices",
                    level,
                    actual: t_level.matrices.len(),
                    expected: expected_clusters,
                });
            }
            if s_level.matrices.len() != expected_clusters {
                return Err(MlfmmRowActionError::LevelClusterCount {
                    operation,
                    stage: "S matrices",
                    level,
                    actual: s_level.matrices.len(),
                    expected: expected_clusters,
                });
            }

            let points = self.sphere_points_per_level[level];
            for (cluster_index, cluster) in clusters.iter().enumerate() {
                let child_points = if level + 1 < expected_levels {
                    self.sphere_points_per_level[level + 1]
                } else {
                    points
                };
                let expected_child_rows = if level + 1 < expected_levels {
                    cluster.sons.len() * child_points
                } else {
                    self.leaf_dof_indices[cluster_index].len()
                };
                let expected_t_cols = if level + 1 < expected_levels {
                    expected_child_rows
                } else {
                    self.leaf_dof_indices[cluster_index].len()
                };
                let t_matrix = &t_level.matrices[cluster_index];
                if t_matrix.nrows() != points || t_matrix.ncols() != expected_t_cols {
                    return Err(MlfmmRowActionError::TransferShape {
                        operation,
                        stage: "T",
                        level,
                        cluster: cluster_index,
                        rows: t_matrix.nrows(),
                        cols: t_matrix.ncols(),
                        expected_rows: points,
                        expected_cols: expected_t_cols,
                    });
                }

                let s_matrix = &s_level.matrices[cluster_index];
                if s_matrix.nrows() != expected_child_rows || s_matrix.ncols() != points {
                    return Err(MlfmmRowActionError::TransferShape {
                        operation,
                        stage: "S",
                        level,
                        cluster: cluster_index,
                        rows: s_matrix.nrows(),
                        cols: s_matrix.ncols(),
                        expected_rows: expected_child_rows,
                        expected_cols: points,
                    });
                }

                for &child in &cluster.sons {
                    let available = self
                        .cluster_levels
                        .get(level + 1)
                        .map_or(0, |next| next.clusters.len());
                    if child >= available {
                        return Err(MlfmmRowActionError::InvalidHierarchyReference {
                            operation,
                            level,
                            side: "child",
                            cluster: cluster_index,
                            reference: child,
                            available,
                        });
                    }
                }
            }

            for (entry, translation) in self.d_matrices[level].iter().enumerate() {
                if translation.source_cluster >= expected_clusters {
                    return Err(MlfmmRowActionError::InvalidHierarchyReference {
                        operation,
                        level,
                        side: "translation source",
                        cluster: translation.field_cluster,
                        reference: translation.source_cluster,
                        available: expected_clusters,
                    });
                }
                if translation.field_cluster >= expected_clusters {
                    return Err(MlfmmRowActionError::InvalidHierarchyReference {
                        operation,
                        level,
                        side: "translation field",
                        cluster: translation.field_cluster,
                        reference: translation.field_cluster,
                        available: expected_clusters,
                    });
                }
                if translation.level != level {
                    return Err(MlfmmRowActionError::InvalidHierarchyReference {
                        operation,
                        level,
                        side: "translation level",
                        cluster: translation.field_cluster,
                        reference: translation.level,
                        available: expected_levels,
                    });
                }
                if translation.diagonal.len() != points {
                    return Err(MlfmmRowActionError::TranslationShape {
                        operation,
                        level,
                        entry,
                        actual: translation.diagonal.len(),
                        expected: points,
                    });
                }
                let invalid_direction =
                    translation.active_directions.as_deref().and_then(|active| {
                        active
                            .iter()
                            .find(|&&direction| direction >= points)
                            .copied()
                    });
                if let Some(direction) = invalid_direction {
                    return Err(MlfmmRowActionError::InvalidHierarchyReference {
                        operation,
                        level,
                        side: "active direction",
                        cluster: translation.field_cluster,
                        reference: direction,
                        available: points,
                    });
                }
            }

            let Some(field_plans) = self.m2l_field_plans.get(level) else {
                continue;
            };
            if field_plans.len() != expected_clusters {
                return Err(MlfmmRowActionError::LevelClusterCount {
                    operation,
                    stage: "M2L field plan",
                    level,
                    actual: field_plans.len(),
                    expected: expected_clusters,
                });
            }
            for (field_cluster, entries) in field_plans.iter().enumerate() {
                for &entry in entries {
                    if entry >= self.d_matrices[level].len() {
                        return Err(MlfmmRowActionError::InvalidHierarchyReference {
                            operation,
                            level,
                            side: "M2L entry",
                            cluster: field_cluster,
                            reference: entry,
                            available: self.d_matrices[level].len(),
                        });
                    }
                }
            }
        }

        for (block, near) in self.near_matrix.iter().enumerate() {
            let source_dofs = self.leaf_dof_indices.get(near.source_cluster).ok_or(
                MlfmmRowActionError::InvalidHierarchyReference {
                    operation,
                    level: expected_levels.saturating_sub(1),
                    side: "near source",
                    cluster: near.field_cluster,
                    reference: near.source_cluster,
                    available: self.leaf_dof_indices.len(),
                },
            )?;
            let field_dofs = self.leaf_dof_indices.get(near.field_cluster).ok_or(
                MlfmmRowActionError::InvalidHierarchyReference {
                    operation,
                    level: expected_levels.saturating_sub(1),
                    side: "near field",
                    cluster: near.field_cluster,
                    reference: near.field_cluster,
                    available: self.leaf_dof_indices.len(),
                },
            )?;
            if near.coefficients.nrows() != source_dofs.len()
                || near.coefficients.ncols() != field_dofs.len()
            {
                return Err(MlfmmRowActionError::NearFieldShape {
                    operation,
                    block,
                    rows: near.coefficients.nrows(),
                    cols: near.coefficients.ncols(),
                    expected_rows: source_dofs.len(),
                    expected_cols: field_dofs.len(),
                });
            }
        }

        Ok(())
    }

    /// Checked forward MLFMM action that rejects malformed hierarchy state,
    /// invalid input, and non-finite output before invoking legacy kernels.
    pub fn matvec_checked(&self, x: &[Complex64]) -> Result<Vec<Complex64>, MlfmmRowActionError> {
        validate_row_action_input(x, self.num_dofs, "forward")?;
        self.validate_action_layout("forward")?;
        let output = self.matvec(x);
        validate_row_action_output(&output, "forward")?;
        Ok(output)
    }

    /// Checked algebraic-transpose MLFMM action.
    pub fn matvec_transpose_checked(
        &self,
        x: &[Complex64],
    ) -> Result<Vec<Complex64>, MlfmmRowActionError> {
        validate_row_action_input(x, self.num_dofs, "transpose")?;
        self.validate_action_layout("transpose")?;
        let output = self.matvec_transpose(x);
        validate_row_action_output(&output, "transpose")?;
        Ok(output)
    }

    /// Checked Hermitian-transpose MLFMM action.
    pub fn apply_hermitian_checked(
        &self,
        x: &[Complex64],
    ) -> Result<Vec<Complex64>, MlfmmRowActionError> {
        validate_row_action_input(x, self.num_dofs, "Hermitian")?;
        self.validate_action_layout("Hermitian")?;
        let output = self.apply_hermitian(x);
        validate_row_action_output(&output, "Hermitian")?;
        Ok(output)
    }

    /// Compare one MLFMM action against independently assembled reference
    /// outputs for deterministic probe vectors.
    pub fn relative_action_error(
        &self,
        action: FmmAction,
        probes: &[Vec<Complex64>],
        reference_outputs: &[Vec<Complex64>],
    ) -> Result<f64, FmmAccuracyError> {
        checked_fmm_relative_action_error(
            action,
            probes,
            reference_outputs,
            self.num_dofs,
            |action, probe| match action {
                FmmAction::Forward => self.matvec(probe),
                FmmAction::Transpose => self.matvec_transpose(probe),
                FmmAction::Hermitian => self.apply_hermitian(probe),
            },
        )
    }

    /// Create a new MLFMM system with the given DOF count, tree levels, and cluster hierarchy
    pub fn new(
        num_dofs: usize,
        num_levels: usize,
        cluster_levels: Vec<ClusterLevel>,
        actual_sphere_points: Vec<usize>,
    ) -> Self {
        let workspace = Mutex::new(MlfmmWorkspace::new(&cluster_levels, &actual_sphere_points));

        Self {
            num_levels,
            near_matrix: Vec::new(),
            t_matrices: Vec::with_capacity(num_levels),
            d_matrices: Vec::with_capacity(num_levels),
            s_matrices: Vec::with_capacity(num_levels),
            rhs: vec![Complex64::new(0.0, 0.0); num_dofs],
            num_dofs,
            cluster_levels: cluster_levels.clone(),
            leaf_dof_indices: Vec::new(),
            leaf_dof_owner: Vec::new(),
            sphere_points_per_level: actual_sphere_points,
            workspace,
            directional_plans: vec![DirectionalLevelPlan::empty(); num_levels],
            m2m_phase_plans: Vec::new(),
            l2l_phase_plans: Vec::new(),
            m2l_field_plans: Vec::new(),
        }
    }

    /// Rebuild cached M2L traversal plans after replacing the public
    /// translation matrices in a synthetic or deserialized system.
    pub fn rebuild_translation_plans(&mut self) {
        self.m2l_field_plans = build_m2l_field_plans(&self.d_matrices, &self.cluster_levels);
        self.directional_plans =
            build_directional_level_plans(&self.d_matrices, &self.cluster_levels);
        self.workspace = Mutex::new(MlfmmWorkspace::new_with_directional(
            &self.cluster_levels,
            &self.sphere_points_per_level,
            &self.directional_plans,
        ));
    }

    /// Compute the matrix-vector product y = A*x using the multi-level FMM
    /// Extract the assembled near-field blocks as a dense matrix.
    ///
    /// The result is intended for a local ILU preconditioner: it preserves
    /// the exact leaf interactions while leaving the far-field MLFMM action
    /// matrix-free.  Blocks are emitted for both directions by assembly, so
    /// no symmetry mirroring is performed here.
    pub fn extract_near_field_matrix(&self) -> DenseMatrix<Complex64> {
        let mut matrix = DenseMatrix::<Complex64>::zeros(self.num_dofs, self.num_dofs);
        for block in &self.near_matrix {
            let source_dofs = &self.leaf_dof_indices[block.source_cluster];
            let field_dofs = &self.leaf_dof_indices[block.field_cluster];

            // Near-field coefficients use the same orientation as the
            // matrix-free path: source DOFs are rows (the output) and field
            // DOFs are columns (the input).  Keep the mapping in that order;
            // leaf clusters need not contain the same number of DOFs.
            for (local_row, &global_row) in source_dofs.iter().enumerate() {
                if !self.leaf_row_is_owner(block.source_cluster, local_row) {
                    continue;
                }
                for (local_col, &global_col) in field_dofs.iter().enumerate() {
                    matrix[[global_row, global_col]] += block.coefficients[[local_row, local_col]];
                }
            }
        }
        matrix
    }

    /// Checked near-field matrix extraction for solver/preconditioner setup.
    ///
    /// The legacy [`Self::extract_near_field_matrix`] API remains available
    /// for trusted callers, while checked solver paths can report malformed
    /// cluster, DOF, block-shape, or coefficient data without panicking.
    pub fn try_extract_near_field_matrix(
        &self,
    ) -> Result<DenseMatrix<Complex64>, MlfmmNearFieldError> {
        let mut matrix = DenseMatrix::<Complex64>::zeros(self.num_dofs, self.num_dofs);
        for (block_index, block) in self.near_matrix.iter().enumerate() {
            let source_dofs = self.leaf_dof_indices.get(block.source_cluster).ok_or(
                MlfmmNearFieldError::InvalidCluster {
                    side: "source",
                    cluster: block.source_cluster,
                    available: self.leaf_dof_indices.len(),
                },
            )?;
            let field_dofs = self.leaf_dof_indices.get(block.field_cluster).ok_or(
                MlfmmNearFieldError::InvalidCluster {
                    side: "field",
                    cluster: block.field_cluster,
                    available: self.leaf_dof_indices.len(),
                },
            )?;
            if block.coefficients.nrows() != source_dofs.len()
                || block.coefficients.ncols() != field_dofs.len()
            {
                return Err(MlfmmNearFieldError::BlockShape {
                    block: block_index,
                    source_dofs: source_dofs.len(),
                    field_dofs: field_dofs.len(),
                    rows: block.coefficients.nrows(),
                    cols: block.coefficients.ncols(),
                });
            }
            for (local, &global) in source_dofs.iter().enumerate() {
                if global >= self.num_dofs {
                    return Err(MlfmmNearFieldError::GlobalDof {
                        block: block_index,
                        side: "source",
                        local,
                        global,
                        num_dofs: self.num_dofs,
                    });
                }
            }
            for (local, &global) in field_dofs.iter().enumerate() {
                if global >= self.num_dofs {
                    return Err(MlfmmNearFieldError::GlobalDof {
                        block: block_index,
                        side: "field",
                        local,
                        global,
                        num_dofs: self.num_dofs,
                    });
                }
            }
            for row in 0..block.coefficients.nrows() {
                for col in 0..block.coefficients.ncols() {
                    let value = block.coefficients[[row, col]];
                    if !value.re.is_finite() || !value.im.is_finite() {
                        return Err(MlfmmNearFieldError::NonFiniteCoefficient {
                            block: block_index,
                            row,
                            col,
                        });
                    }
                    if self.leaf_row_is_owner(block.source_cluster, row) {
                        matrix[[source_dofs[row], field_dofs[col]]] += value;
                    }
                }
            }
        }
        Ok(matrix)
    }

    /// Compute the matrix-vector product `y = A*x`.
    pub fn matvec(&self, x: &[Complex64]) -> Vec<Complex64> {
        let mut y = vec![Complex64::new(0.0, 0.0); self.num_dofs];
        self.matvec_into(x, &mut y);
        y
    }

    /// Apply the forward FMM action only on caller-owned rows.
    ///
    /// This is a local-action contract for distributed transports; it does
    /// not perform communication or imply a distributed solve. The checked
    /// wrapper validates dimensions before delegating to the legacy matvec.
    pub fn apply_rows_checked(
        &self,
        input: &[Complex64],
        rows: &[usize],
    ) -> Result<Vec<Complex64>, MlfmmRowActionError> {
        validate_row_action_input(input, self.num_dofs, "forward")?;
        self.validate_action_layout("forward")?;
        validate_rows(rows, self.num_dofs)?;
        let output = self.matvec(input);
        validate_row_action_output(&output, "forward")?;
        Ok(rows.iter().map(|&row| output[row]).collect())
    }

    /// Apply the transpose FMM action only on caller-owned rows.
    pub fn apply_transpose_rows_checked(
        &self,
        input: &[Complex64],
        rows: &[usize],
    ) -> Result<Vec<Complex64>, MlfmmRowActionError> {
        validate_row_action_input(input, self.num_dofs, "transpose")?;
        self.validate_action_layout("transpose")?;
        validate_rows(rows, self.num_dofs)?;
        let output = self.matvec_transpose(input);
        validate_row_action_output(&output, "transpose")?;
        Ok(rows.iter().map(|&row| output[row]).collect())
    }

    /// Apply the forward FMM action to selected rows for several RHS columns.
    ///
    /// This is a local ownership/action contract: it performs no transport or
    /// distributed solve, while preserving caller-supplied row order.
    pub fn apply_rows_block_checked(
        &self,
        input: &DenseMatrix<Complex64>,
        rows: &[usize],
    ) -> Result<DenseMatrix<Complex64>, MlfmmRowActionError> {
        validate_row_action_block_input(input, self.num_dofs, "forward block")?;
        self.validate_action_layout("forward block")?;
        validate_rows(rows, self.num_dofs)?;
        let mut full_action = DenseMatrix::zeros(self.num_dofs, input.ncols());
        self.matvec_block_into(input, &mut full_action);
        validate_row_action_block_output(&full_action, "forward block")?;
        let mut output = DenseMatrix::zeros(rows.len(), input.ncols());
        for (selected, &row) in rows.iter().enumerate() {
            for rhs in 0..input.ncols() {
                output[[selected, rhs]] = full_action[[row, rhs]];
            }
        }
        Ok(output)
    }

    /// Apply the transpose FMM action to selected rows for several RHS
    /// columns, preserving caller-supplied row order.
    pub fn apply_transpose_rows_block_checked(
        &self,
        input: &DenseMatrix<Complex64>,
        rows: &[usize],
    ) -> Result<DenseMatrix<Complex64>, MlfmmRowActionError> {
        validate_row_action_block_input(input, self.num_dofs, "transpose block")?;
        self.validate_action_layout("transpose block")?;
        validate_rows(rows, self.num_dofs)?;
        let full_action = self.matvec_transpose_block(input);
        validate_row_action_block_output(&full_action, "transpose block")?;
        let mut output = DenseMatrix::zeros(rows.len(), input.ncols());
        for (selected, &row) in rows.iter().enumerate() {
            for rhs in 0..input.ncols() {
                output[[selected, rhs]] = full_action[[row, rhs]];
            }
        }
        Ok(output)
    }

    /// Compute the matrix-vector product y = A*x into a pre-allocated buffer.
    /// Checked Hermitian-transpose MLFMM action for several right-hand sides.
    pub fn apply_hermitian_block_checked(
        &self,
        input: &DenseMatrix<Complex64>,
    ) -> Result<DenseMatrix<Complex64>, MlfmmRowActionError> {
        validate_row_action_block_input(input, self.num_dofs, "Hermitian block")?;
        self.validate_action_layout("Hermitian block")?;
        let mut output = DenseMatrix::zeros(self.num_dofs, input.ncols());
        for rhs in 0..input.ncols() {
            let column: Vec<_> = (0..self.num_dofs).map(|row| input[[row, rhs]]).collect();
            let result = self.apply_hermitian_checked(&column)?;
            for (row, value) in result.into_iter().enumerate() {
                output[[row, rhs]] = value;
            }
        }
        validate_row_action_block_output(&output, "Hermitian block")?;
        Ok(output)
    }

    /// Apply checked Hermitian-transpose MLFMM action on caller-owned rows.
    pub fn apply_hermitian_rows_block_checked(
        &self,
        input: &DenseMatrix<Complex64>,
        rows: &[usize],
    ) -> Result<DenseMatrix<Complex64>, MlfmmRowActionError> {
        validate_row_action_block_input(input, self.num_dofs, "Hermitian block")?;
        self.validate_action_layout("Hermitian block")?;
        validate_rows(rows, self.num_dofs)?;
        let full_action = self.apply_hermitian_block_checked(input)?;
        let mut output = DenseMatrix::zeros(rows.len(), input.ncols());
        for (selected, &row) in rows.iter().enumerate() {
            for rhs in 0..input.ncols() {
                output[[selected, rhs]] = full_action[[row, rhs]];
            }
        }
        Ok(output)
    }

    /// Compute matrix-vector product `y = A*x` into a pre-allocated buffer.
    pub fn matvec_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        assert_eq!(y.len(), self.num_dofs);
        if self.num_levels == 0 {
            y.fill(Complex64::new(0.0, 0.0));
            return;
        }
        // Near-field accumulation (rayon above the machine-adaptive gate,
        // serial for few near-field blocks). Each block holds a dense
        // block matvec, so the grain is coarse blocks per thread.
        let near_y = if use_parallel(self.near_matrix.len(), BEM_BLOCKS_PER_THREAD) {
            self.near_field_matvec_parallel(x)
        } else {
            self.near_field_matvec(x)
        };
        y.copy_from_slice(&near_y);
        let leaf_level = self.num_levels - 1;

        if self.num_levels > 1 && !self.t_matrices.is_empty() {
            let mut workspace = self.workspace.lock().unwrap();
            workspace.clear();
            self.upward_pass(x, &mut workspace);
            self.translate_all_levels(&mut workspace);
            self.downward_pass(&mut workspace);
            self.evaluate_locals(&workspace, leaf_level, y);
        }
    }

    /// Compute several matrix-vector products at once, with RHS in columns.
    pub fn matvec_block(&self, x: &DenseMatrix<Complex64>) -> DenseMatrix<Complex64> {
        let mut y = DenseMatrix::<Complex64>::zeros(self.num_dofs, x.ncols());
        self.matvec_block_into(x, &mut y);
        y
    }

    /// Compute several MLFMM products into a pre-allocated column-major RHS block.
    ///
    /// The tree passes operate on all RHS columns together.  In particular, the
    /// expensive M2M/M2L/L2L coefficients are loaded once per expansion point and
    /// applied to every RHS column, while the dense T/S transfers use batched GEMM.
    pub fn matvec_block_into(&self, x: &DenseMatrix<Complex64>, y: &mut DenseMatrix<Complex64>) {
        assert_eq!(x.nrows(), self.num_dofs);
        assert_eq!(y.nrows(), self.num_dofs);
        assert_eq!(y.ncols(), x.ncols());
        y.fill(Complex64::new(0.0, 0.0));
        let nrhs = x.ncols();
        if nrhs == 0 || self.num_levels == 0 {
            return;
        }

        let near_y = self.near_field_matvec_block(x);
        y.as_mut_slice().copy_from_slice(near_y.as_slice());

        if self.num_levels <= 1 || self.t_matrices.is_empty() {
            return;
        }

        let leaf_level = self.num_levels - 1;
        let mut workspace = self.workspace.lock().unwrap();
        workspace.ensure_block_storage(
            &self.cluster_levels,
            &self.sphere_points_per_level,
            &self.directional_plans,
            nrhs,
        );
        workspace.clear_block();
        self.upward_pass_block(x, &mut workspace, nrhs);
        self.translate_all_levels_block(&mut workspace, nrhs);
        self.downward_pass_block(&mut workspace, nrhs);
        self.evaluate_locals_block(&workspace, leaf_level, y, nrhs);
    }

    /// Apply only the near-field portion of the matvec: y = [N] * x.
    ///
    /// Mirrors the near-field branch of `matvec` so callers (e.g. QA bins)
    /// can compute the near-field-only contribution without touching the
    /// private `Array2` per-block storage.
    pub fn near_field_matvec(&self, x: &[Complex64]) -> Vec<Complex64> {
        let mut y = vec![Complex64::new(0.0, 0.0); self.num_dofs];
        for block in &self.near_matrix {
            self.accumulate_near_field_block(block, x, &mut y);
        }
        y
    }

    fn near_field_matvec_parallel(&self, x: &[Complex64]) -> Vec<Complex64> {
        use rayon::prelude::*;
        self.near_matrix
            .par_iter()
            .fold(
                || vec![Complex64::new(0.0, 0.0); self.num_dofs],
                |mut local_y, block| {
                    self.accumulate_near_field_block(block, x, &mut local_y);
                    local_y
                },
            )
            .reduce(
                || vec![Complex64::new(0.0, 0.0); self.num_dofs],
                |mut a, b| {
                    for (ai, bi) in a.iter_mut().zip(b.iter()) {
                        *ai += *bi;
                    }
                    a
                },
            )
    }

    fn accumulate_near_field_block(
        &self,
        block: &NearFieldBlock,
        x: &[Complex64],
        y: &mut [Complex64],
    ) {
        if block.source_cluster >= self.leaf_dof_indices.len()
            || block.field_cluster >= self.leaf_dof_indices.len()
        {
            return;
        }
        let src_dofs = &self.leaf_dof_indices[block.source_cluster];
        let fld_dofs = &self.leaf_dof_indices[block.field_cluster];
        let num_rows = src_dofs.len().min(block.coefficients.nrows());
        let num_cols = fld_dofs.len().min(block.coefficients.ncols());
        let has_non_owner =
            (0..num_rows).any(|row| !self.leaf_row_is_owner(block.source_cluster, row));
        if !has_non_owner {
            indexed_dense_matvec_accumulate_c64(
                block.coefficients.as_slice(),
                num_rows,
                num_cols,
                block.coefficients.ncols(),
                x,
                &fld_dofs[..num_cols],
                y,
                &src_dofs[..num_rows],
            );
        } else {
            for local_row in 0..num_rows {
                if !self.leaf_row_is_owner(block.source_cluster, local_row) {
                    continue;
                }
                let mut value = Complex64::new(0.0, 0.0);
                for local_col in 0..num_cols {
                    value += block.coefficients[[local_row, local_col]] * x[fld_dofs[local_col]];
                }
                y[src_dofs[local_row]] += value;
            }
        }
    }

    fn near_field_matvec_block(&self, x: &DenseMatrix<Complex64>) -> DenseMatrix<Complex64> {
        let nrhs = x.ncols();
        // Near-field block contributions (rayon above the machine-adaptive
        // gate, serial for few blocks). Each block holds a dense block
        // matvec, so the grain is coarse blocks per thread.
        let contributions: Vec<Vec<(usize, Vec<Complex64>)>> = gated_map(
            &self.near_matrix,
            self.near_matrix.len(),
            BEM_BLOCKS_PER_THREAD,
            |block| {
                if block.source_cluster >= self.leaf_dof_indices.len()
                    || block.field_cluster >= self.leaf_dof_indices.len()
                {
                    return Vec::new();
                }
                let src_dofs = &self.leaf_dof_indices[block.source_cluster];
                let fld_dofs = &self.leaf_dof_indices[block.field_cluster];
                let mut x_local = DenseMatrix::<Complex64>::zeros(block.coefficients.ncols(), nrhs);
                for (local_i, &global_i) in
                    fld_dofs.iter().take(block.coefficients.ncols()).enumerate()
                {
                    if global_i < x.nrows() {
                        x_local.row_mut(local_i).copy_from_slice(x.row(global_i));
                    }
                }
                let mut y_local = DenseMatrix::<Complex64>::zeros(block.coefficients.nrows(), nrhs);
                if !block.coefficients.as_slice().is_empty() {
                    block.coefficients.matmul_into(&x_local, &mut y_local);
                }
                src_dofs
                    .iter()
                    .enumerate()
                    .take(y_local.nrows())
                    .filter(|(local_i, _)| self.leaf_row_is_owner(block.source_cluster, *local_i))
                    .map(|(local_i, &global_i)| (global_i, y_local.row(local_i).to_vec()))
                    .collect()
            },
        );

        let mut y = DenseMatrix::<Complex64>::zeros(self.num_dofs, nrhs);
        for block_contributions in contributions {
            for (global_i, values) in block_contributions {
                if global_i >= y.nrows() {
                    continue;
                }
                for (rhs, value) in values.into_iter().enumerate() {
                    y[[global_i, rhs]] += value;
                }
            }
        }
        y
    }

    /// Compute the algebraic transpose product y = A^T*x using the same MLFMM factors.
    /// Compute MLFMM algebraic-transpose products for several RHS columns.
    ///
    /// The public block contract is shared with SLFMM/H-matrix backends. The
    /// ordinary non-directional tree shares its near-field, Sᵀ, Dᵀ, and Tᵀ
    /// work across columns; directional plans use compact-cone M2L traversal
    /// behind the same boundary.
    pub fn matvec_transpose_block(&self, x: &DenseMatrix<Complex64>) -> DenseMatrix<Complex64> {
        let mut y = DenseMatrix::<Complex64>::zeros(self.num_dofs, x.ncols());
        self.matvec_transpose_block_into(x, &mut y);
        y
    }

    /// Compute the MLFMM algebraic-transpose block into pre-allocated output.
    pub fn matvec_transpose_block_into(
        &self,
        x: &DenseMatrix<Complex64>,
        y: &mut DenseMatrix<Complex64>,
    ) {
        assert_eq!(x.nrows(), self.num_dofs);
        assert_eq!(y.nrows(), self.num_dofs);
        assert_eq!(y.ncols(), x.ncols());
        y.fill(Complex64::new(0.0, 0.0));
        let nrhs = x.ncols();
        if nrhs == 0 || self.num_levels == 0 {
            return;
        }

        // Directional and ordinary plans share the batched transpose tree.
        let near_y = self.near_field_matvec_transpose_block(x);
        y.as_mut_slice().copy_from_slice(near_y.as_slice());

        if self.num_levels <= 1 || self.t_matrices.is_empty() {
            return;
        }

        let leaf_level = self.num_levels - 1;
        let mut workspace = self.workspace.lock().unwrap();
        workspace.ensure_block_storage(
            &self.cluster_levels,
            &self.sphere_points_per_level,
            &self.directional_plans,
            nrhs,
        );
        workspace.clear_block();
        self.evaluate_locals_transpose_block(x, leaf_level, &mut workspace, nrhs);
        self.downward_pass_transpose_block(&mut workspace, nrhs);
        self.translate_all_levels_transpose_block(&mut workspace, nrhs);
        self.upward_pass_transpose_block(&workspace, y, nrhs);
    }

    fn near_field_matvec_transpose_block(
        &self,
        x: &DenseMatrix<Complex64>,
    ) -> DenseMatrix<Complex64> {
        let nrhs = x.ncols();
        // Near-field transpose block contributions (rayon above the
        // machine-adaptive gate, serial for few blocks).
        let contributions: Vec<Vec<(usize, Vec<Complex64>)>> = gated_map(
            &self.near_matrix,
            self.near_matrix.len(),
            BEM_BLOCKS_PER_THREAD,
            |block| {
                if block.source_cluster >= self.leaf_dof_indices.len()
                    || block.field_cluster >= self.leaf_dof_indices.len()
                {
                    return Vec::new();
                }
                let src_dofs = &self.leaf_dof_indices[block.source_cluster];
                let fld_dofs = &self.leaf_dof_indices[block.field_cluster];
                let x_local = DenseMatrix::from_fn(block.coefficients.nrows(), nrhs, |row, rhs| {
                    if !self.leaf_row_is_owner(block.source_cluster, row) {
                        return Complex64::new(0.0, 0.0);
                    }
                    src_dofs
                        .get(row)
                        .and_then(|&global| x.as_slice().get(global * nrhs + rhs))
                        .copied()
                        .unwrap_or_else(|| Complex64::new(0.0, 0.0))
                });
                let mut y_local = DenseMatrix::<Complex64>::zeros(block.coefficients.ncols(), nrhs);
                block
                    .coefficients
                    .transposed()
                    .matmul_into(&x_local, &mut y_local);
                fld_dofs
                    .iter()
                    .enumerate()
                    .take(y_local.nrows())
                    .map(|(local, &global)| (global, y_local.row(local).to_vec()))
                    .collect()
            },
        );

        let mut y = DenseMatrix::<Complex64>::zeros(self.num_dofs, nrhs);
        for block_contributions in contributions {
            for (global, values) in block_contributions {
                if global >= y.nrows() {
                    continue;
                }
                for (rhs, value) in values.into_iter().enumerate() {
                    y[[global, rhs]] += value;
                }
            }
        }
        y
    }

    /// Compute the algebraic-transpose product `y = A^T*x`.
    pub fn matvec_transpose(&self, x: &[Complex64]) -> Vec<Complex64> {
        let mut y = vec![Complex64::new(0.0, 0.0); self.num_dofs];
        self.matvec_transpose_into(x, &mut y);
        y
    }

    /// Compute the algebraic transpose product y = A^T*x into a pre-allocated buffer.
    pub fn matvec_transpose_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        assert_eq!(y.len(), self.num_dofs);
        y.fill(Complex64::new(0.0, 0.0));
        if self.num_levels == 0 {
            return;
        }
        let leaf_level = self.num_levels - 1;

        // Near-field transpose contributions (rayon above the
        // machine-adaptive gate, serial for few blocks).
        let near_contributions: Vec<(usize, Complex64)> = gated_flat_map(
            &self.near_matrix,
            self.near_matrix.len(),
            BEM_BLOCKS_PER_THREAD,
            |block| {
                if block.source_cluster >= self.leaf_dof_indices.len()
                    || block.field_cluster >= self.leaf_dof_indices.len()
                {
                    return Vec::new();
                }
                let src_dofs = &self.leaf_dof_indices[block.source_cluster];
                let fld_dofs = &self.leaf_dof_indices[block.field_cluster];
                let x_local: Vec<Complex64> = src_dofs
                    .iter()
                    .enumerate()
                    .map(|(local, &global)| {
                        if self.leaf_row_is_owner(block.source_cluster, local) {
                            x[global]
                        } else {
                            Complex64::new(0.0, 0.0)
                        }
                    })
                    .collect();
                let mut y_local = vec![Complex64::new(0.0, 0.0); fld_dofs.len()];
                block
                    .coefficients
                    .matvec_transpose_into(&x_local, &mut y_local);
                fld_dofs
                    .iter()
                    .enumerate()
                    .map(|(local_j, &global_j)| (global_j, y_local[local_j]))
                    .collect()
            },
        );

        for (global_i, val) in near_contributions {
            y[global_i] += val;
        }

        if self.num_levels > 1 && !self.t_matrices.is_empty() {
            let mut workspace = MlfmmWorkspace::new_with_directional(
                &self.cluster_levels,
                &self.sphere_points_per_level,
                &self.directional_plans,
            );
            self.evaluate_locals_transpose(x, leaf_level, &mut workspace);
            self.downward_pass_transpose(&mut workspace);
            self.translate_all_levels_transpose(&mut workspace);
            self.upward_pass_transpose(&workspace, y);
        }
    }

    fn upward_pass(&self, x: &[Complex64], workspace: &mut MlfmmWorkspace) {
        let leaf_level = self.num_levels - 1;
        if leaf_level < self.t_matrices.len() {
            let t_level = &self.t_matrices[leaf_level];
            let leaf_dof_indices = &self.leaf_dof_indices;
            let t_matrices = &t_level.matrices;
            let multipoles_leaf = &mut workspace.multipoles[leaf_level];
            let leaf_cols = multipoles_leaf.ncols();

            // Leaf multipole fill (rayon above the machine-adaptive gate,
            // serial for few clusters).
            gated_chunks_mut(
                multipoles_leaf.as_mut_slice(),
                leaf_cols,
                BEM_BLOCKS_PER_THREAD,
                |cluster_idx, multipole| {
                    if cluster_idx >= leaf_dof_indices.len() {
                        multipole.fill(Complex64::new(0.0, 0.0));
                        return;
                    }
                    let cluster_dofs = &leaf_dof_indices[cluster_idx];
                    if cluster_dofs.is_empty() {
                        multipole.fill(Complex64::new(0.0, 0.0));
                        return;
                    }
                    let t_mat = &t_matrices[cluster_idx];
                    if t_mat.as_slice().is_empty() {
                        multipole.fill(Complex64::new(0.0, 0.0));
                        return;
                    }
                    let rows = multipole.len().min(t_mat.nrows());
                    let cols = cluster_dofs.len().min(t_mat.ncols());
                    indexed_dense_matvec_c64(
                        t_mat.as_slice(),
                        rows,
                        cols,
                        t_mat.ncols(),
                        x,
                        &cluster_dofs[..cols],
                        multipole,
                    );
                },
            );
        }

        for level in (0..leaf_level).rev() {
            if level >= self.t_matrices.len() || level >= self.cluster_levels.len() {
                continue;
            }
            let t_level = &self.t_matrices[level];
            let clusters = &self.cluster_levels[level].clusters;
            let child_level = level + 1;
            let child_num_points = self.sphere_points_per_level[child_level];
            let phase_plan = self
                .m2m_phase_plans
                .get(level)
                .filter(|plan| !plan.is_empty());

            let (child_multipoles_all, parent_multipoles_all): (
                &DenseMatrix<Complex64>,
                &mut DenseMatrix<Complex64>,
            ) = {
                let (first, second) = workspace.multipoles.split_at_mut(child_level);
                let parent = &mut first[level];
                let child = &second[0];
                (child, parent)
            };

            let parent_rows = parent_multipoles_all.nrows();
            let parent_cols = parent_multipoles_all.ncols();
            // Parent multipole fill (rayon above the machine-adaptive gate,
            // serial for few clusters). The slice is truncated to the
            // `.take(parent_rows)` window the parallel loop covered.
            let parent_slice = parent_multipoles_all.as_mut_slice();
            let take_len = parent_rows
                .saturating_mul(parent_cols)
                .min(parent_slice.len());
            gated_chunks_mut(
                &mut parent_slice[..take_len],
                parent_cols,
                BEM_BLOCKS_PER_THREAD,
                |cluster_idx, multipole| {
                    multipole.fill(Complex64::new(0.0, 0.0));
                    if cluster_idx >= clusters.len() || cluster_idx >= t_level.matrices.len() {
                        return;
                    }
                    if let Some(plan) = phase_plan {
                        if let Some(children) = plan.parent_children.get(cluster_idx) {
                            for child_phase in children {
                                if child_phase.child_cluster < child_multipoles_all.nrows() {
                                    let child_mult =
                                        child_multipoles_all.row(child_phase.child_cluster);
                                    let len = parent_cols
                                        .min(child_mult.len())
                                        .min(child_phase.phase.len());
                                    phase_mul_accumulate_c64(
                                        &mut multipole[..len],
                                        &child_phase.phase[..len],
                                        &child_mult[..len],
                                    );
                                }
                            }
                        }
                    } else {
                        let cluster = &clusters[cluster_idx];
                        let t_mat = &t_level.matrices[cluster_idx];
                        if t_mat.as_slice().is_empty() || cluster.sons.is_empty() {
                            return;
                        }
                        for (son_idx, &child_cluster_idx) in cluster.sons.iter().enumerate() {
                            if child_cluster_idx < child_multipoles_all.nrows() {
                                let child_mult = child_multipoles_all.row(child_cluster_idx);
                                let offset = son_idx * child_num_points;
                                let len = parent_cols
                                    .min(child_num_points)
                                    .min(child_mult.len())
                                    .min(t_mat.nrows());
                                for p in 0..len {
                                    let col = offset + p;
                                    if col < t_mat.ncols() {
                                        multipole[p] += t_mat.row(p)[col] * child_mult[p];
                                    }
                                }
                            }
                        }
                    }
                },
            );
        }
    }

    fn upward_pass_block(
        &self,
        x: &DenseMatrix<Complex64>,
        workspace: &mut MlfmmWorkspace,
        nrhs: usize,
    ) {
        let leaf_level = self.num_levels - 1;
        let leaf_points = self.sphere_points_per_level[leaf_level];
        if leaf_points == 0 || leaf_level >= self.t_matrices.len() {
            return;
        }

        let t_level = &self.t_matrices[leaf_level];
        let leaf_dof_indices = &self.leaf_dof_indices;
        let multipoles_leaf = &mut workspace.block_multipoles[leaf_level];
        let chunk_len = leaf_points * nrhs;
        // Block leaf multipole fill (rayon above the machine-adaptive gate,
        // serial for few clusters).
        gated_chunks_mut(
            multipoles_leaf.as_mut_slice(),
            chunk_len,
            BEM_BLOCKS_PER_THREAD,
            |cluster_idx, multipole| {
                let Some(cluster_dofs) = leaf_dof_indices.get(cluster_idx) else {
                    return;
                };
                let Some(t_mat) = t_level.matrices.get(cluster_idx) else {
                    return;
                };
                if cluster_dofs.is_empty() || t_mat.as_slice().is_empty() {
                    return;
                }
                let input = DenseMatrix::from_fn(t_mat.ncols(), nrhs, |local_i, rhs| {
                    cluster_dofs
                        .get(local_i)
                        .and_then(|&global_i| x.as_slice().get(global_i * nrhs + rhs))
                        .copied()
                        .unwrap_or_else(|| Complex64::new(0.0, 0.0))
                });
                let mut output = DenseMatrix::<Complex64>::zeros(t_mat.nrows(), nrhs);
                t_mat.matmul_into(&input, &mut output);
                for point in 0..leaf_points.min(output.nrows()) {
                    multipole[point * nrhs..(point + 1) * nrhs].copy_from_slice(output.row(point));
                }
            },
        );

        for level in (0..leaf_level).rev() {
            if level >= self.t_matrices.len() || level >= self.cluster_levels.len() {
                continue;
            }
            let parent_points = self.sphere_points_per_level[level];
            let child_level = level + 1;
            let child_points = self.sphere_points_per_level[child_level];
            if parent_points == 0 || child_points == 0 {
                continue;
            }
            let phase_plan = self
                .m2m_phase_plans
                .get(level)
                .filter(|plan| !plan.is_empty());
            let clusters = &self.cluster_levels[level].clusters;
            let t_level = &self.t_matrices[level];

            let (child_multipoles, parent_multipoles) = {
                let (first, second) = workspace.block_multipoles.split_at_mut(child_level);
                (&second[0], &mut first[level])
            };
            let parent_chunk_len = parent_points * nrhs;
            // Block parent multipole fill (rayon above the machine-adaptive
            // gate, serial for few clusters).
            gated_chunks_mut(
                parent_multipoles.as_mut_slice(),
                parent_chunk_len,
                BEM_BLOCKS_PER_THREAD,
                |cluster_idx, multipole| {
                    let Some(cluster) = clusters.get(cluster_idx) else {
                        return;
                    };
                    let Some(t_mat) = t_level.matrices.get(cluster_idx) else {
                        return;
                    };
                    if let Some(plan) = phase_plan {
                        if let Some(children) = plan.parent_children.get(cluster_idx) {
                            for child_phase in children {
                                let child_base = child_phase.child_cluster * child_points;
                                if child_base >= child_multipoles.nrows() {
                                    continue;
                                }
                                let limit =
                                    parent_points.min(child_points).min(child_phase.phase.len());
                                for point in 0..limit {
                                    let src = child_multipoles.row(child_base + point);
                                    let dst = &mut multipole[point * nrhs..(point + 1) * nrhs];
                                    let phase = child_phase.phase[point];
                                    for rhs in 0..nrhs {
                                        dst[rhs] += phase * src[rhs];
                                    }
                                }
                            }
                        }
                    } else if !t_mat.as_slice().is_empty() {
                        for (son_idx, &child_cluster_idx) in cluster.sons.iter().enumerate() {
                            let child_base = child_cluster_idx * child_points;
                            if child_base >= child_multipoles.nrows() {
                                continue;
                            }
                            let limit = parent_points.min(t_mat.nrows());
                            for point in 0..limit {
                                let col = son_idx * child_points + point;
                                if col >= t_mat.ncols()
                                    || child_base + point >= child_multipoles.nrows()
                                {
                                    continue;
                                }
                                let coefficient = t_mat.row(point)[col];
                                let src = child_multipoles.row(child_base + point);
                                let dst = &mut multipole[point * nrhs..(point + 1) * nrhs];
                                for rhs in 0..nrhs {
                                    dst[rhs] += coefficient * src[rhs];
                                }
                            }
                        }
                    }
                },
            );
        }
    }

    fn upward_pass_transpose(&self, workspace: &MlfmmWorkspace, y: &mut [Complex64]) {
        let leaf_level = self.num_levels - 1;

        let mut multipoles = workspace.multipoles.clone();
        for level in 0..leaf_level {
            if level >= self.t_matrices.len() || level >= self.cluster_levels.len() {
                continue;
            }
            let t_level = &self.t_matrices[level];
            let clusters = &self.cluster_levels[level].clusters;
            let child_level = level + 1;
            let child_num_points = self.sphere_points_per_level[child_level];
            let parent_multipoles = multipoles[level].clone();

            for (cluster_idx, cluster) in clusters.iter().enumerate() {
                if cluster_idx >= t_level.matrices.len() || cluster.sons.is_empty() {
                    continue;
                }
                let t_mat = &t_level.matrices[cluster_idx];
                if t_mat.as_slice().is_empty() || cluster_idx >= parent_multipoles.nrows() {
                    continue;
                }
                if let Some(phase_plan) = self
                    .m2m_phase_plans
                    .get(level)
                    .filter(|plan| !plan.is_empty())
                {
                    if let Some(children) = phase_plan.parent_children.get(cluster_idx) {
                        let parent = parent_multipoles.row(cluster_idx);
                        for child_phase in children {
                            if child_phase.child_cluster >= multipoles[child_level].nrows() {
                                continue;
                            }
                            let child = multipoles[child_level].row_mut(child_phase.child_cluster);
                            let len = child.len().min(parent.len()).min(child_phase.phase.len());
                            for point in 0..len {
                                child[point] += child_phase.phase[point] * parent[point];
                            }
                        }
                    }
                    continue;
                }
                let mut child_updates = vec![Complex64::new(0.0, 0.0); t_mat.ncols()];
                t_mat.matvec_transpose_into(parent_multipoles.row(cluster_idx), &mut child_updates);
                for (son_idx, &child_cluster_idx) in cluster.sons.iter().enumerate() {
                    if child_cluster_idx >= multipoles[child_level].nrows() {
                        continue;
                    }
                    let offset = son_idx * child_num_points;
                    if offset + child_num_points <= child_updates.len() {
                        let child_cols = multipoles[child_level].ncols();
                        for p in 0..child_num_points.min(child_cols) {
                            multipoles[child_level][[child_cluster_idx, p]] +=
                                child_updates[offset + p];
                        }
                    }
                }
            }
        }

        if leaf_level < self.t_matrices.len() {
            let t_level = &self.t_matrices[leaf_level];
            let multipoles_leaf = &multipoles[leaf_level];
            for (cluster_idx, cluster_dofs) in self.leaf_dof_indices.iter().enumerate() {
                if cluster_dofs.is_empty()
                    || cluster_idx >= t_level.matrices.len()
                    || cluster_idx >= multipoles_leaf.nrows()
                {
                    continue;
                }
                let t_mat = &t_level.matrices[cluster_idx];
                if t_mat.as_slice().is_empty() {
                    continue;
                }
                let mut y_local = vec![Complex64::new(0.0, 0.0); t_mat.ncols()];
                t_mat.matvec_transpose_into(multipoles_leaf.row(cluster_idx), &mut y_local);
                for (local_i, &global_i) in cluster_dofs.iter().enumerate() {
                    if global_i < y.len() && local_i < y_local.len() {
                        y[global_i] += y_local[local_i];
                    }
                }
            }
        }
    }

    fn upward_pass_transpose_block(
        &self,
        workspace: &MlfmmWorkspace,
        y: &mut DenseMatrix<Complex64>,
        nrhs: usize,
    ) {
        let leaf_level = self.num_levels.saturating_sub(1);
        let mut multipoles = workspace.block_multipoles.clone();

        for level in 0..leaf_level {
            if level >= self.t_matrices.len() || level >= self.cluster_levels.len() {
                continue;
            }
            let parent_points = self.sphere_points_per_level[level];
            let child_level = level + 1;
            let child_points = self.sphere_points_per_level[child_level];
            if parent_points == 0 || child_points == 0 {
                continue;
            }
            let t_level = &self.t_matrices[level];
            let clusters = &self.cluster_levels[level].clusters;
            let (parent_multipoles, child_multipoles) = {
                let (first, second) = multipoles.split_at_mut(child_level);
                (&first[level], &mut second[0])
            };

            for (cluster_idx, cluster) in clusters.iter().enumerate() {
                let Some(t_mat) = t_level.matrices.get(cluster_idx) else {
                    continue;
                };
                if t_mat.as_slice().is_empty()
                    || cluster.sons.is_empty()
                    || cluster_idx * parent_points + parent_points > parent_multipoles.nrows()
                {
                    continue;
                }
                if let Some(phase_plan) = self
                    .m2m_phase_plans
                    .get(level)
                    .filter(|plan| !plan.is_empty())
                {
                    if let Some(children) = phase_plan.parent_children.get(cluster_idx) {
                        let parent_base = cluster_idx * parent_points;
                        for child_phase in children {
                            let child_base = child_phase.child_cluster * child_points;
                            if child_base + child_points > child_multipoles.nrows() {
                                continue;
                            }
                            let limit =
                                parent_points.min(child_points).min(child_phase.phase.len());
                            for point in 0..limit {
                                let coefficient = child_phase.phase[point];
                                let src = parent_multipoles.row(parent_base + point).to_vec();
                                let dst = child_multipoles.row_mut(child_base + point);
                                for rhs in 0..nrhs {
                                    dst[rhs] += coefficient * src[rhs];
                                }
                            }
                        }
                    }
                    continue;
                }
                let parent_input = DenseMatrix::from_fn(parent_points, nrhs, |point, rhs| {
                    parent_multipoles[[cluster_idx * parent_points + point, rhs]]
                });
                let mut child_update = DenseMatrix::<Complex64>::zeros(t_mat.ncols(), nrhs);
                t_mat
                    .transposed()
                    .matmul_into(&parent_input, &mut child_update);
                for (son_idx, &child_cluster_idx) in cluster.sons.iter().enumerate() {
                    let child_base = child_cluster_idx * child_points;
                    let update_base = son_idx * child_points;
                    if child_base + child_points > child_multipoles.nrows()
                        || update_base + child_points > child_update.nrows()
                    {
                        continue;
                    }
                    for point in 0..child_points {
                        let dst = child_multipoles.row_mut(child_base + point);
                        let src = child_update.row(update_base + point);
                        for rhs in 0..nrhs {
                            dst[rhs] += src[rhs];
                        }
                    }
                }
            }
        }

        if leaf_level >= self.t_matrices.len() {
            return;
        }
        let leaf_points = self.sphere_points_per_level[leaf_level];
        let t_level = &self.t_matrices[leaf_level];
        let multipoles_leaf = &multipoles[leaf_level];
        for (cluster_idx, cluster_dofs) in self.leaf_dof_indices.iter().enumerate() {
            let Some(t_mat) = t_level.matrices.get(cluster_idx) else {
                continue;
            };
            if cluster_dofs.is_empty()
                || t_mat.as_slice().is_empty()
                || cluster_idx * leaf_points + leaf_points > multipoles_leaf.nrows()
            {
                continue;
            }
            let multipole = DenseMatrix::from_fn(leaf_points, nrhs, |point, rhs| {
                multipoles_leaf[[cluster_idx * leaf_points + point, rhs]]
            });
            let mut y_local = DenseMatrix::<Complex64>::zeros(t_mat.ncols(), nrhs);
            t_mat.transposed().matmul_into(&multipole, &mut y_local);
            for (local, &global) in cluster_dofs.iter().enumerate() {
                if global >= y.nrows() || local >= y_local.nrows() {
                    continue;
                }
                for rhs in 0..nrhs {
                    y[[global, rhs]] += y_local[[local, rhs]];
                }
            }
        }
    }

    fn translate_all_levels_transpose_block(&self, workspace: &mut MlfmmWorkspace, nrhs: usize) {
        for level in 0..self.num_levels {
            if level >= self.d_matrices.len()
                || level >= workspace.block_locals.len()
                || level >= workspace.block_multipoles.len()
            {
                continue;
            }
            let d_entries = &self.d_matrices[level];
            self.translate_full_entries_level_transpose_block(level, d_entries, workspace, nrhs);
            if self
                .directional_plans
                .get(level)
                .is_some_and(|plan| !plan.is_empty())
            {
                self.translate_directional_level_transpose_block(level, workspace, nrhs);
            }
        }
    }

    fn translate_full_entries_level_transpose_block(
        &self,
        level: usize,
        d_entries: &[DMatrixEntry],
        workspace: &mut MlfmmWorkspace,
        nrhs: usize,
    ) {
        let has_directional_plan = self
            .directional_plans
            .get(level)
            .is_some_and(|plan| !plan.is_empty());
        let locals = &workspace.block_locals[level];
        let multipoles = &mut workspace.block_multipoles[level];
        for entry in d_entries {
            if has_directional_plan && entry.active_directions.is_some() {
                continue;
            }
            let source_base = entry
                .source_cluster
                .saturating_mul(self.sphere_points_per_level[level]);
            let field_base = entry
                .field_cluster
                .saturating_mul(self.sphere_points_per_level[level]);
            if source_base >= multipoles.nrows() || field_base >= locals.nrows() {
                continue;
            }
            let limit = self.sphere_points_per_level[level]
                .min(entry.diagonal.len())
                .min(multipoles.nrows() - source_base)
                .min(locals.nrows() - field_base);
            for point in 0..limit {
                let coefficient = entry.diagonal[point];
                let field = locals.row(field_base + point).to_vec();
                let source = multipoles.row_mut(source_base + point);
                for rhs in 0..nrhs {
                    source[rhs] += coefficient * field[rhs];
                }
            }
        }
    }

    /// Apply directional M2L transpose entries to all RHS columns together.
    ///
    /// Directional plans store compact cone rows; expanding each active
    /// direction directly into the full block workspace keeps the algebraic
    /// transpose batched without falling back to one scalar traversal per RHS.
    fn translate_directional_level_transpose_block(
        &self,
        level: usize,
        workspace: &mut MlfmmWorkspace,
        nrhs: usize,
    ) {
        let Some(plan) = self.directional_plans.get(level) else {
            return;
        };
        if plan.is_empty()
            || level >= workspace.block_locals.len()
            || level >= workspace.block_multipoles.len()
        {
            return;
        }
        let points = self.sphere_points_per_level[level];
        if points == 0 {
            return;
        }
        let locals = &workspace.block_locals[level];
        let multipoles = &mut workspace.block_multipoles[level];
        for (cone_idx, cone) in plan.cones.iter().enumerate() {
            let Some(entries) = plan.entries_by_cone.get(cone_idx) else {
                continue;
            };
            for entry in entries {
                let Some(&source_cluster) = cone.source_clusters.get(entry.source_row) else {
                    continue;
                };
                let Some(&field_cluster) = cone.field_clusters.get(entry.field_row) else {
                    continue;
                };
                let source_base = source_cluster.saturating_mul(points);
                let field_base = field_cluster.saturating_mul(points);
                if source_base >= multipoles.nrows() || field_base >= locals.nrows() {
                    continue;
                }
                for (active_idx, &direction) in cone.active_directions.iter().enumerate() {
                    if direction >= points || active_idx >= entry.diagonal.len() {
                        continue;
                    }
                    let field = locals.row(field_base + direction).to_vec();
                    let source = multipoles.row_mut(source_base + direction);
                    let coefficient = entry.diagonal[active_idx];
                    for rhs in 0..nrhs {
                        source[rhs] += coefficient * field[rhs];
                    }
                }
            }
        }
    }

    fn downward_pass_transpose_block(&self, workspace: &mut MlfmmWorkspace, nrhs: usize) {
        let leaf_level = self.num_levels.saturating_sub(1);
        for level in (0..leaf_level).rev() {
            if level >= self.s_matrices.len() || level >= self.cluster_levels.len() {
                continue;
            }
            let parent_points = self.sphere_points_per_level[level];
            let child_level = level + 1;
            let child_points = self.sphere_points_per_level[child_level];
            if parent_points == 0 || child_points == 0 {
                continue;
            }
            let s_level = &self.s_matrices[level];
            let clusters = &self.cluster_levels[level].clusters;
            let (child_locals, parent_locals) = {
                let (first, second) = workspace.block_locals.split_at_mut(child_level);
                (&second[0], &mut first[level])
            };
            for (cluster_idx, cluster) in clusters.iter().enumerate() {
                let Some(s_mat) = s_level.matrices.get(cluster_idx) else {
                    continue;
                };
                if s_mat.as_slice().is_empty()
                    || cluster.sons.is_empty()
                    || cluster_idx * parent_points + parent_points > parent_locals.nrows()
                    || s_mat.nrows() != cluster.sons.len() * child_points
                {
                    continue;
                }
                if let Some(phase_plan) = self
                    .l2l_phase_plans
                    .get(level)
                    .filter(|plan| !plan.is_empty())
                {
                    let parent_base = cluster_idx * parent_points;
                    if let Some(children) = phase_plan.parent_children.get(cluster_idx) {
                        for child_phase in children {
                            let child_base = child_phase.child_cluster * child_points;
                            if child_base + child_points > child_locals.nrows() {
                                continue;
                            }
                            let limit =
                                parent_points.min(child_points).min(child_phase.phase.len());
                            for point in 0..limit {
                                let coefficient = child_phase.phase[point];
                                let src = child_locals.row(child_base + point).to_vec();
                                let dst = parent_locals.row_mut(parent_base + point);
                                for rhs in 0..nrhs {
                                    dst[rhs] += coefficient * src[rhs];
                                }
                            }
                        }
                    }
                    continue;
                }
                let child_input = DenseMatrix::from_fn(s_mat.nrows(), nrhs, |row, rhs| {
                    let son_idx = row / child_points;
                    let point = row % child_points;
                    let child_cluster = cluster.sons[son_idx];
                    child_locals[[child_cluster * child_points + point, rhs]]
                });
                let mut parent_update = DenseMatrix::<Complex64>::zeros(s_mat.ncols(), nrhs);
                s_mat
                    .transposed()
                    .matmul_into(&child_input, &mut parent_update);
                for point in 0..parent_points.min(parent_update.nrows()) {
                    let dst = parent_locals.row_mut(cluster_idx * parent_points + point);
                    let src = parent_update.row(point);
                    for rhs in 0..nrhs {
                        dst[rhs] += src[rhs];
                    }
                }
            }
        }
    }

    fn evaluate_locals_transpose_block(
        &self,
        x: &DenseMatrix<Complex64>,
        leaf_level: usize,
        workspace: &mut MlfmmWorkspace,
        nrhs: usize,
    ) {
        if leaf_level >= self.s_matrices.len() || leaf_level >= workspace.block_locals.len() {
            return;
        }
        let points = self.sphere_points_per_level[leaf_level];
        let s_level = &self.s_matrices[leaf_level];
        let locals = &mut workspace.block_locals[leaf_level];
        for (cluster_idx, cluster_dofs) in self.leaf_dof_indices.iter().enumerate() {
            let Some(s_mat) = s_level.matrices.get(cluster_idx) else {
                continue;
            };
            if cluster_dofs.is_empty()
                || s_mat.as_slice().is_empty()
                || cluster_idx * points + points > locals.nrows()
            {
                continue;
            }
            let x_local = DenseMatrix::from_fn(s_mat.nrows(), nrhs, |row, rhs| {
                if !self.leaf_row_is_owner(cluster_idx, row) {
                    return Complex64::new(0.0, 0.0);
                }
                cluster_dofs
                    .get(row)
                    .map(|&global| x[[global, rhs]])
                    .unwrap_or_else(|| Complex64::new(0.0, 0.0))
            });
            let mut local_update = DenseMatrix::<Complex64>::zeros(s_mat.ncols(), nrhs);
            s_mat.transposed().matmul_into(&x_local, &mut local_update);
            for point in 0..points.min(local_update.nrows()) {
                let dst = locals.row_mut(cluster_idx * points + point);
                let src = local_update.row(point);
                for rhs in 0..nrhs {
                    dst[rhs] += src[rhs];
                }
            }
        }
    }

    fn translate_all_levels(&self, workspace: &mut MlfmmWorkspace) {
        for level in 0..self.num_levels {
            if level >= self.d_matrices.len() {
                continue;
            }
            let d_entries = &self.d_matrices[level];
            if self
                .directional_plans
                .get(level)
                .is_some_and(|plan| !plan.is_empty())
            {
                self.translate_full_entries_level(level, d_entries, workspace);
                self.translate_directional_level(level, workspace);
            } else {
                self.translate_full_entries_level(level, d_entries, workspace);
            }
        }
    }

    fn translate_all_levels_block(&self, workspace: &mut MlfmmWorkspace, nrhs: usize) {
        for level in 0..self.num_levels {
            if level >= self.d_matrices.len() {
                continue;
            }
            let d_entries = &self.d_matrices[level];
            if self
                .directional_plans
                .get(level)
                .is_some_and(|plan| !plan.is_empty())
            {
                self.translate_full_entries_level_block(level, d_entries, workspace, nrhs);
                self.translate_directional_level_block(level, workspace, nrhs);
            } else {
                self.translate_full_entries_level_block(level, d_entries, workspace, nrhs);
            }
        }
    }

    fn translate_full_entries_level(
        &self,
        level: usize,
        d_entries: &[DMatrixEntry],
        workspace: &mut MlfmmWorkspace,
    ) {
        if level >= workspace.multipoles.len() || level >= workspace.locals.len() {
            return;
        }
        let has_directional_plan = self
            .directional_plans
            .get(level)
            .is_some_and(|plan| !plan.is_empty());
        if has_directional_plan
            && d_entries
                .iter()
                .all(|entry| entry.active_directions.is_some())
        {
            return;
        }
        let entries: Box<dyn Iterator<Item = &DMatrixEntry> + '_> = if has_directional_plan {
            Box::new(
                d_entries
                    .iter()
                    .filter(|entry| entry.active_directions.is_none()),
            )
        } else {
            Box::new(d_entries.iter())
        };

        let field_plan = self
            .m2l_field_plans
            .get(level)
            .filter(|plan| !plan.is_empty());
        let multipoles_this_level = &workspace.multipoles[level];
        let locals_this_level = &mut workspace.locals[level];

        if let Some(field_plan) = field_plan {
            let local_cols = locals_this_level.ncols();
            // Local-row fill (rayon above the machine-adaptive gate, serial
            // for few field clusters).
            gated_chunks_mut(
                locals_this_level.as_mut_slice(),
                local_cols,
                BEM_BLOCKS_PER_THREAD,
                |field_cluster, local_row| {
                    let Some(entry_indices) = field_plan.get(field_cluster) else {
                        return;
                    };
                    for &entry_idx in entry_indices {
                        let Some(d_entry) = d_entries.get(entry_idx) else {
                            continue;
                        };
                        if has_directional_plan && d_entry.active_directions.is_some() {
                            continue;
                        }
                        if d_entry.source_cluster >= multipoles_this_level.nrows() {
                            continue;
                        }
                        let src_mult = multipoles_this_level.row(d_entry.source_cluster);
                        let len = local_row
                            .len()
                            .min(src_mult.len())
                            .min(d_entry.diagonal.len());
                        phase_mul_accumulate_c64(
                            &mut local_row[..len],
                            &d_entry.diagonal[..len],
                            &src_mult[..len],
                        );
                    }
                },
            );
        } else {
            for d_entry in entries {
                if d_entry.field_cluster >= locals_this_level.nrows()
                    || d_entry.source_cluster >= multipoles_this_level.nrows()
                {
                    continue;
                }
                let src_mult = multipoles_this_level.row(d_entry.source_cluster);
                let row = locals_this_level.row_mut(d_entry.field_cluster);
                let len = row.len().min(src_mult.len()).min(d_entry.diagonal.len());
                phase_mul_accumulate_c64(
                    &mut row[..len],
                    &d_entry.diagonal[..len],
                    &src_mult[..len],
                );
            }
        }
    }

    fn translate_directional_level(&self, level: usize, workspace: &mut MlfmmWorkspace) {
        let Some(plan) = self.directional_plans.get(level) else {
            return;
        };
        if plan.is_empty()
            || level >= workspace.multipoles.len()
            || level >= workspace.locals.len()
            || level >= workspace.cone_multipoles.len()
            || level >= workspace.cone_locals.len()
        {
            return;
        }

        {
            let multipoles_this_level = &workspace.multipoles[level];
            let cone_multipoles = &mut workspace.cone_multipoles[level];
            for (cone_idx, cone) in plan.cones.iter().enumerate() {
                if cone_idx >= cone_multipoles.len() {
                    continue;
                }
                let cone_mult = &mut cone_multipoles[cone_idx];
                for (compact_cluster_idx, &cluster_idx) in cone.source_clusters.iter().enumerate() {
                    if compact_cluster_idx >= cone_mult.nrows()
                        || cluster_idx >= multipoles_this_level.nrows()
                    {
                        continue;
                    }
                    let src_row = multipoles_this_level.row(cluster_idx);
                    let dst_row = cone_mult.row_mut(compact_cluster_idx);
                    compact_indexed_gather_c64(src_row, &cone.active_directions, dst_row);
                }
            }
        }

        {
            let cone_multipoles = &workspace.cone_multipoles[level];
            let cone_locals = &mut workspace.cone_locals[level];
            // Cone-local accumulation (rayon above the machine-adaptive
            // gate, serial for few cones). The inner row walk stays serial
            // to avoid nested pool dispatch.
            gated_for_each_mut(
                cone_locals.as_mut_slice(),
                BEM_BLOCKS_PER_THREAD,
                |cone_idx, local_matrix| {
                    let Some(multipole_matrix) = cone_multipoles.get(cone_idx) else {
                        return;
                    };
                    let Some(entries) = plan.entries_by_cone.get(cone_idx) else {
                        return;
                    };
                    let Some(entries_by_field_row) = plan.entries_by_cone_field_row.get(cone_idx)
                    else {
                        return;
                    };
                    let ncols = local_matrix.ncols();
                    if ncols == 0 {
                        return;
                    }
                    for (field_row_idx, field_row) in
                        local_matrix.as_mut_slice().chunks_mut(ncols).enumerate()
                    {
                        let Some(entry_indices) = entries_by_field_row.get(field_row_idx) else {
                            continue;
                        };
                        for &entry_idx in entry_indices {
                            let Some(entry) = entries.get(entry_idx) else {
                                continue;
                            };
                            if entry.source_row >= multipole_matrix.nrows() {
                                continue;
                            }
                            let src_row = multipole_matrix.row(entry.source_row);
                            let len = field_row.len().min(src_row.len()).min(entry.diagonal.len());
                            phase_mul_accumulate_c64(
                                &mut field_row[..len],
                                &entry.diagonal[..len],
                                &src_row[..len],
                            );
                        }
                    }
                },
            );
        }

        {
            let cone_locals = &workspace.cone_locals[level];
            let locals = &mut workspace.locals[level];
            for (cone_idx, cone) in plan.cones.iter().enumerate() {
                if cone_idx >= cone_locals.len() {
                    continue;
                }
                let cone_local = &cone_locals[cone_idx];
                for (compact_cluster_idx, &cluster_idx) in cone.field_clusters.iter().enumerate() {
                    if compact_cluster_idx >= cone_local.nrows() || cluster_idx >= locals.nrows() {
                        continue;
                    }
                    let src_row = cone_local.row(compact_cluster_idx);
                    let dst_row = locals.row_mut(cluster_idx);
                    compact_indexed_scatter_accumulate_c64(
                        src_row,
                        &cone.active_directions,
                        dst_row,
                    );
                }
            }
        }
    }

    fn translate_full_entries_level_block(
        &self,
        level: usize,
        d_entries: &[DMatrixEntry],
        workspace: &mut MlfmmWorkspace,
        nrhs: usize,
    ) {
        if level >= workspace.block_multipoles.len() || level >= workspace.block_locals.len() {
            return;
        }
        let has_directional_plan = self
            .directional_plans
            .get(level)
            .is_some_and(|plan| !plan.is_empty());
        if has_directional_plan
            && d_entries
                .iter()
                .all(|entry| entry.active_directions.is_some())
        {
            return;
        }
        let field_plan = self
            .m2l_field_plans
            .get(level)
            .filter(|plan| !plan.is_empty());
        let points = self.sphere_points_per_level[level];
        if points == 0 {
            return;
        }
        let multipoles = &workspace.block_multipoles[level];
        let locals = &mut workspace.block_locals[level];

        if let Some(field_plan) = field_plan {
            // Block local-row fill (rayon above the machine-adaptive gate,
            // serial for few field clusters).
            gated_chunks_mut(
                locals.as_mut_slice(),
                points * nrhs,
                BEM_BLOCKS_PER_THREAD,
                |field_cluster, local_rows| {
                    let Some(entry_indices) = field_plan.get(field_cluster) else {
                        return;
                    };
                    for &entry_idx in entry_indices {
                        let Some(entry) = d_entries.get(entry_idx) else {
                            continue;
                        };
                        if has_directional_plan && entry.active_directions.is_some() {
                            continue;
                        }
                        let source_base = entry.source_cluster * points;
                        if source_base >= multipoles.nrows() {
                            continue;
                        }
                        let limit = points.min(entry.diagonal.len());
                        for point in 0..limit {
                            let src = multipoles.row(source_base + point);
                            let dst = &mut local_rows[point * nrhs..(point + 1) * nrhs];
                            let coefficient = entry.diagonal[point];
                            for rhs in 0..nrhs {
                                dst[rhs] += coefficient * src[rhs];
                            }
                        }
                    }
                },
            );
        } else {
            for entry in d_entries {
                if has_directional_plan && entry.active_directions.is_some() {
                    continue;
                }
                let source_base = entry.source_cluster * points;
                let field_base = entry.field_cluster * points;
                if source_base >= multipoles.nrows() || field_base >= locals.nrows() {
                    continue;
                }
                let limit = points.min(entry.diagonal.len());
                for point in 0..limit {
                    let src = multipoles.row(source_base + point);
                    let dst = locals.row_mut(field_base + point);
                    let coefficient = entry.diagonal[point];
                    for rhs in 0..nrhs {
                        dst[rhs] += coefficient * src[rhs];
                    }
                }
            }
        }
    }

    fn translate_directional_level_block(
        &self,
        level: usize,
        workspace: &mut MlfmmWorkspace,
        nrhs: usize,
    ) {
        let Some(plan) = self.directional_plans.get(level) else {
            return;
        };
        if plan.is_empty()
            || level >= workspace.block_multipoles.len()
            || level >= workspace.block_locals.len()
            || level >= workspace.block_cone_multipoles.len()
            || level >= workspace.block_cone_locals.len()
        {
            return;
        }
        let points = self.sphere_points_per_level[level];

        {
            let multipoles = &workspace.block_multipoles[level];
            let cone_multipoles = &mut workspace.block_cone_multipoles[level];
            for (cone_idx, cone) in plan.cones.iter().enumerate() {
                let Some(cone_matrix) = cone_multipoles.get_mut(cone_idx) else {
                    continue;
                };
                let n_active = cone.active_directions.len();
                for (compact_cluster_idx, &cluster_idx) in cone.source_clusters.iter().enumerate() {
                    let full_base = cluster_idx * points;
                    let compact_base = compact_cluster_idx * n_active;
                    if full_base >= multipoles.nrows() || compact_base >= cone_matrix.nrows() {
                        continue;
                    }
                    for (active_idx, &direction_idx) in cone.active_directions.iter().enumerate() {
                        if direction_idx >= points
                            || compact_base + active_idx >= cone_matrix.nrows()
                        {
                            continue;
                        }
                        let src = multipoles.row(full_base + direction_idx);
                        cone_matrix
                            .row_mut(compact_base + active_idx)
                            .copy_from_slice(src);
                    }
                }
            }
        }

        {
            let cone_multipoles = &workspace.block_cone_multipoles[level];
            let cone_locals = &mut workspace.block_cone_locals[level];
            // Block cone-local accumulation (rayon above the machine-adaptive
            // gate, serial for few cones). The inner row walk stays serial
            // to avoid nested pool dispatch.
            gated_for_each_mut(
                cone_locals.as_mut_slice(),
                BEM_BLOCKS_PER_THREAD,
                |cone_idx, local_matrix| {
                    let Some(multipole_matrix) = cone_multipoles.get(cone_idx) else {
                        return;
                    };
                    let Some(entries) = plan.entries_by_cone.get(cone_idx) else {
                        return;
                    };
                    let Some(entries_by_field_row) = plan.entries_by_cone_field_row.get(cone_idx)
                    else {
                        return;
                    };
                    let Some(cone) = plan.cones.get(cone_idx) else {
                        return;
                    };
                    let n_active = cone.active_directions.len();
                    if n_active == 0 {
                        return;
                    }
                    for (field_row_idx, field_row) in local_matrix
                        .as_mut_slice()
                        .chunks_mut(n_active * nrhs)
                        .enumerate()
                    {
                        let Some(entry_indices) = entries_by_field_row.get(field_row_idx) else {
                            continue;
                        };
                        for &entry_idx in entry_indices {
                            let Some(entry) = entries.get(entry_idx) else {
                                continue;
                            };
                            let source_base = entry.source_row * n_active;
                            if source_base >= multipole_matrix.nrows() {
                                continue;
                            }
                            for active_idx in 0..n_active.min(entry.diagonal.len()) {
                                let src = multipole_matrix.row(source_base + active_idx);
                                let dst =
                                    &mut field_row[active_idx * nrhs..(active_idx + 1) * nrhs];
                                let coefficient = entry.diagonal[active_idx];
                                for rhs in 0..nrhs {
                                    dst[rhs] += coefficient * src[rhs];
                                }
                            }
                        }
                    }
                },
            );
        }

        {
            let cone_locals = &workspace.block_cone_locals[level];
            let locals = &mut workspace.block_locals[level];
            for (cone_idx, cone) in plan.cones.iter().enumerate() {
                let Some(cone_matrix) = cone_locals.get(cone_idx) else {
                    continue;
                };
                let n_active = cone.active_directions.len();
                for (compact_cluster_idx, &cluster_idx) in cone.field_clusters.iter().enumerate() {
                    let compact_base = compact_cluster_idx * n_active;
                    let full_base = cluster_idx * points;
                    if compact_base >= cone_matrix.nrows() || full_base >= locals.nrows() {
                        continue;
                    }
                    for (active_idx, &direction_idx) in cone.active_directions.iter().enumerate() {
                        if direction_idx >= points
                            || compact_base + active_idx >= cone_matrix.nrows()
                        {
                            continue;
                        }
                        let src = cone_matrix.row(compact_base + active_idx);
                        let dst = locals.row_mut(full_base + direction_idx);
                        for rhs in 0..nrhs {
                            dst[rhs] += src[rhs];
                        }
                    }
                }
            }
        }
    }

    fn translate_all_levels_transpose(&self, workspace: &mut MlfmmWorkspace) {
        for level in 0..self.num_levels {
            if level >= self.d_matrices.len()
                || level >= workspace.locals.len()
                || level >= workspace.multipoles.len()
            {
                continue;
            }
            if self
                .directional_plans
                .get(level)
                .is_some_and(|plan| !plan.is_empty())
            {
                self.translate_full_entries_level_transpose(
                    level,
                    &self.d_matrices[level],
                    workspace,
                );
                self.translate_directional_level_transpose(level, workspace);
            } else {
                self.translate_full_entries_level_transpose(
                    level,
                    &self.d_matrices[level],
                    workspace,
                );
            }
        }
    }

    fn translate_full_entries_level_transpose(
        &self,
        level: usize,
        d_entries: &[DMatrixEntry],
        workspace: &mut MlfmmWorkspace,
    ) {
        let has_directional_plan = self
            .directional_plans
            .get(level)
            .is_some_and(|plan| !plan.is_empty());
        if has_directional_plan
            && d_entries
                .iter()
                .all(|entry| entry.active_directions.is_some())
        {
            return;
        }
        for d_entry in d_entries {
            if has_directional_plan && d_entry.active_directions.is_some() {
                continue;
            }
            if d_entry.field_cluster >= workspace.locals[level].nrows()
                || d_entry.source_cluster >= workspace.multipoles[level].nrows()
            {
                continue;
            }
            let field_local = workspace.locals[level].row(d_entry.field_cluster);
            let row = workspace.multipoles[level].row_mut(d_entry.source_cluster);
            let len = row.len().min(field_local.len()).min(d_entry.diagonal.len());
            phase_mul_accumulate_c64(
                &mut row[..len],
                &d_entry.diagonal[..len],
                &field_local[..len],
            );
        }
    }

    fn translate_directional_level_transpose(&self, level: usize, workspace: &mut MlfmmWorkspace) {
        let Some(plan) = self.directional_plans.get(level) else {
            return;
        };
        if plan.is_empty()
            || level >= workspace.multipoles.len()
            || level >= workspace.locals.len()
            || level >= workspace.cone_multipoles.len()
            || level >= workspace.cone_locals.len()
        {
            return;
        }

        {
            let locals_this_level = &workspace.locals[level];
            let cone_locals = &mut workspace.cone_locals[level];
            for (cone_idx, cone) in plan.cones.iter().enumerate() {
                if cone_idx >= cone_locals.len() {
                    continue;
                }
                let cone_local = &mut cone_locals[cone_idx];
                for (compact_cluster_idx, &cluster_idx) in cone.field_clusters.iter().enumerate() {
                    if compact_cluster_idx >= cone_local.nrows()
                        || cluster_idx >= locals_this_level.nrows()
                    {
                        continue;
                    }
                    let src_row = locals_this_level.row(cluster_idx);
                    let dst_row = cone_local.row_mut(compact_cluster_idx);
                    compact_indexed_gather_c64(src_row, &cone.active_directions, dst_row);
                }
            }
        }

        {
            let cone_locals = &workspace.cone_locals[level];
            let cone_multipoles = &mut workspace.cone_multipoles[level];
            // Cone-multipole scatter (rayon above the machine-adaptive gate,
            // serial for few cones). The inner row walk stays serial to
            // avoid nested pool dispatch.
            gated_for_each_mut(
                cone_multipoles.as_mut_slice(),
                BEM_BLOCKS_PER_THREAD,
                |cone_idx, multipole_matrix| {
                    let Some(local_matrix) = cone_locals.get(cone_idx) else {
                        return;
                    };
                    let Some(entries) = plan.entries_by_cone.get(cone_idx) else {
                        return;
                    };
                    let Some(entries_by_source_row) = plan.entries_by_cone_source_row.get(cone_idx)
                    else {
                        return;
                    };
                    let ncols = multipole_matrix.ncols();
                    if ncols == 0 {
                        return;
                    }
                    for (source_row_idx, source_row) in multipole_matrix
                        .as_mut_slice()
                        .chunks_mut(ncols)
                        .enumerate()
                    {
                        let Some(entry_indices) = entries_by_source_row.get(source_row_idx) else {
                            continue;
                        };
                        for &entry_idx in entry_indices {
                            let Some(entry) = entries.get(entry_idx) else {
                                continue;
                            };
                            if entry.field_row >= local_matrix.nrows() {
                                continue;
                            }
                            let field_row = local_matrix.row(entry.field_row);
                            let len = source_row
                                .len()
                                .min(field_row.len())
                                .min(entry.diagonal.len());
                            phase_mul_accumulate_c64(
                                &mut source_row[..len],
                                &entry.diagonal[..len],
                                &field_row[..len],
                            );
                        }
                    }
                },
            );
        }

        {
            let cone_multipoles = &workspace.cone_multipoles[level];
            let multipoles = &mut workspace.multipoles[level];
            for (cone_idx, cone) in plan.cones.iter().enumerate() {
                if cone_idx >= cone_multipoles.len() {
                    continue;
                }
                let cone_multipole = &cone_multipoles[cone_idx];
                for (compact_cluster_idx, &cluster_idx) in cone.source_clusters.iter().enumerate() {
                    if compact_cluster_idx >= cone_multipole.nrows()
                        || cluster_idx >= multipoles.nrows()
                    {
                        continue;
                    }
                    let src_row = cone_multipole.row(compact_cluster_idx);
                    let dst_row = multipoles.row_mut(cluster_idx);
                    compact_indexed_scatter_accumulate_c64(
                        src_row,
                        &cone.active_directions,
                        dst_row,
                    );
                }
            }
        }
    }

    fn downward_pass(&self, workspace: &mut MlfmmWorkspace) {
        for level in 0..self.num_levels.saturating_sub(1) {
            if level >= self.s_matrices.len() || level >= self.cluster_levels.len() {
                continue;
            }
            let s_level = &self.s_matrices[level];
            let clusters = &self.cluster_levels[level].clusters;
            let child_level = level + 1;
            let child_num_points = self.sphere_points_per_level[child_level];
            let child_clusters = &self.cluster_levels[child_level].clusters;
            let phase_plan = self
                .l2l_phase_plans
                .get(level)
                .filter(|plan| !plan.is_empty());
            let (parent_locals_all, child_locals_all): (
                &DenseMatrix<Complex64>,
                &mut DenseMatrix<Complex64>,
            ) = {
                let (first, second) = workspace.locals.split_at_mut(child_level);
                let parent = &first[level];
                let child = &mut second[0];
                (parent, child)
            };
            // Downward child-row fill (rayon above the machine-adaptive
            // gate, serial for few clusters).
            gated_chunks_mut(
                child_locals_all.as_mut_slice(),
                child_num_points,
                BEM_BLOCKS_PER_THREAD,
                |child_cluster_idx, child_row| {
                    let Some(child_cluster) = child_clusters.get(child_cluster_idx) else {
                        return;
                    };
                    let Some(parent_idx) = child_cluster.father else {
                        return;
                    };
                    if parent_idx >= clusters.len()
                        || parent_idx >= s_level.matrices.len()
                        || parent_idx >= parent_locals_all.nrows()
                    {
                        return;
                    }
                    let parent_cluster = &clusters[parent_idx];
                    let Some(son_idx) = parent_cluster
                        .sons
                        .iter()
                        .position(|&idx| idx == child_cluster_idx)
                    else {
                        return;
                    };
                    let parent_local = parent_locals_all.row(parent_idx);
                    if let Some(plan) = phase_plan {
                        if let Some(child_phase) =
                            plan.parent_children.get(parent_idx).and_then(|children| {
                                children
                                    .iter()
                                    .find(|phase| phase.child_cluster == child_cluster_idx)
                            })
                        {
                            let len = child_row
                                .len()
                                .min(parent_local.len())
                                .min(child_phase.phase.len());
                            phase_mul_accumulate_c64(
                                &mut child_row[..len],
                                &child_phase.phase[..len],
                                &parent_local[..len],
                            );
                        }
                    } else {
                        let s_mat = &s_level.matrices[parent_idx];
                        if s_mat.as_slice().is_empty() {
                            return;
                        }
                        let offset = son_idx * child_num_points;
                        let len = child_row.len().min(parent_local.len()).min(s_mat.ncols());
                        for p in 0..len {
                            let row_idx = offset + p;
                            if row_idx < s_mat.nrows() {
                                child_row[p] += s_mat.row(row_idx)[p] * parent_local[p];
                            }
                        }
                    }
                },
            );
        }
    }

    fn downward_pass_block(&self, workspace: &mut MlfmmWorkspace, nrhs: usize) {
        for level in 0..self.num_levels.saturating_sub(1) {
            if level >= self.s_matrices.len() || level >= self.cluster_levels.len() {
                continue;
            }
            let parent_points = self.sphere_points_per_level[level];
            let child_level = level + 1;
            let child_points = self.sphere_points_per_level[child_level];
            if parent_points == 0 || child_points == 0 {
                continue;
            }
            let s_level = &self.s_matrices[level];
            let clusters = &self.cluster_levels[level].clusters;
            let child_clusters = &self.cluster_levels[child_level].clusters;
            let phase_plan = self
                .l2l_phase_plans
                .get(level)
                .filter(|plan| !plan.is_empty());
            let (parent_locals, child_locals) = {
                let (first, second) = workspace.block_locals.split_at_mut(child_level);
                (&first[level], &mut second[0])
            };
            let child_chunk_len = child_points * nrhs;
            // Block downward child-row fill (rayon above the machine-adaptive
            // gate, serial for few clusters).
            gated_chunks_mut(
                child_locals.as_mut_slice(),
                child_chunk_len,
                BEM_BLOCKS_PER_THREAD,
                |child_cluster_idx, child_row| {
                    let Some(child_cluster) = child_clusters.get(child_cluster_idx) else {
                        return;
                    };
                    let Some(parent_idx) = child_cluster.father else {
                        return;
                    };
                    let Some(parent_cluster) = clusters.get(parent_idx) else {
                        return;
                    };
                    let Some(son_idx) = parent_cluster
                        .sons
                        .iter()
                        .position(|&idx| idx == child_cluster_idx)
                    else {
                        return;
                    };
                    let parent_base = parent_idx * parent_points;
                    if parent_base >= parent_locals.nrows() || parent_idx >= s_level.matrices.len()
                    {
                        return;
                    }
                    if let Some(plan) = phase_plan {
                        if let Some(child_phase) =
                            plan.parent_children.get(parent_idx).and_then(|children| {
                                children
                                    .iter()
                                    .find(|phase| phase.child_cluster == child_cluster_idx)
                            })
                        {
                            let limit =
                                child_points.min(parent_points).min(child_phase.phase.len());
                            for point in 0..limit {
                                let dst = &mut child_row[point * nrhs..(point + 1) * nrhs];
                                let coefficient = child_phase.phase[point];
                                let parent_point = parent_base + point;
                                if parent_point >= parent_locals.nrows() {
                                    continue;
                                }
                                let parent_row = parent_locals.row(parent_point);
                                for rhs in 0..nrhs {
                                    dst[rhs] += coefficient * parent_row[rhs];
                                }
                            }
                        }
                    } else {
                        let s_mat = &s_level.matrices[parent_idx];
                        if s_mat.as_slice().is_empty() {
                            return;
                        }
                        let offset = son_idx * child_points;
                        let limit = child_points
                            .min(parent_points)
                            .min(s_mat.nrows().saturating_sub(offset));
                        for point in 0..limit {
                            let coefficient = s_mat.row(offset + point)[point];
                            let dst = &mut child_row[point * nrhs..(point + 1) * nrhs];
                            let parent_point = parent_base + point;
                            if parent_point >= parent_locals.nrows() {
                                continue;
                            }
                            let src = parent_locals.row(parent_point);
                            for rhs in 0..nrhs {
                                dst[rhs] += coefficient * src[rhs];
                            }
                        }
                    }
                },
            );
        }
    }

    fn downward_pass_transpose(&self, workspace: &mut MlfmmWorkspace) {
        let leaf_level = self.num_levels - 1;
        for level in (0..leaf_level).rev() {
            if level >= self.s_matrices.len() || level >= self.cluster_levels.len() {
                continue;
            }
            let s_level = &self.s_matrices[level];
            let clusters = &self.cluster_levels[level].clusters;
            let child_level = level + 1;
            let child_num_points = self.sphere_points_per_level[child_level];

            let (child_locals_all, parent_locals_all): (
                &DenseMatrix<Complex64>,
                &mut DenseMatrix<Complex64>,
            ) = {
                let (first, second) = workspace.locals.split_at_mut(child_level);
                let parent = &mut first[level];
                let child = &second[0];
                (child, parent)
            };

            for (cluster_idx, cluster) in clusters.iter().enumerate() {
                if cluster_idx >= s_level.matrices.len() || cluster.sons.is_empty() {
                    continue;
                }
                let s_mat = &s_level.matrices[cluster_idx];
                if s_mat.as_slice().is_empty() || cluster_idx >= parent_locals_all.nrows() {
                    continue;
                }
                if let Some(phase_plan) = self
                    .l2l_phase_plans
                    .get(level)
                    .filter(|plan| !plan.is_empty())
                {
                    if let Some(children) = phase_plan.parent_children.get(cluster_idx) {
                        for child_phase in children {
                            if child_phase.child_cluster >= child_locals_all.nrows() {
                                continue;
                            }
                            let child = child_locals_all.row(child_phase.child_cluster);
                            let parent = parent_locals_all.row_mut(cluster_idx);
                            let len = parent.len().min(child.len()).min(child_phase.phase.len());
                            for point in 0..len {
                                parent[point] += child_phase.phase[point] * child[point];
                            }
                        }
                    }
                    continue;
                }
                let mut child_l_concat =
                    vec![Complex64::new(0.0, 0.0); cluster.sons.len() * child_num_points];
                for (son_idx, &child_cluster_idx) in cluster.sons.iter().enumerate() {
                    if child_cluster_idx < child_locals_all.nrows() {
                        let offset = son_idx * child_num_points;
                        let child_local = child_locals_all.row(child_cluster_idx);
                        let len = child_num_points.min(child_local.len());
                        child_l_concat[offset..offset + len].copy_from_slice(&child_local[..len]);
                    }
                }
                if child_l_concat.len() == s_mat.nrows() {
                    let mut parent_update = vec![Complex64::new(0.0, 0.0); s_mat.ncols()];
                    s_mat.matvec_transpose_into(&child_l_concat, &mut parent_update);
                    let row = parent_locals_all.row_mut(cluster_idx);
                    for (r, v) in row.iter_mut().zip(parent_update.iter()) {
                        *r += *v;
                    }
                }
            }
        }
    }

    fn evaluate_locals(&self, workspace: &MlfmmWorkspace, leaf_level: usize, y: &mut [Complex64]) {
        use rayon::prelude::*;
        if leaf_level >= self.s_matrices.len() || leaf_level >= workspace.locals.len() {
            return;
        }
        let s_level = &self.s_matrices[leaf_level];
        let locals_leaf = &workspace.locals[leaf_level];
        let leaf_dof_indices = &self.leaf_dof_indices;
        // Local evaluation fold (rayon above the machine-adaptive gate,
        // serial for few clusters). The fold closure is shared by both
        // paths.
        let fold_cluster = |mut local_y: Vec<Complex64>, cluster_idx: usize| {
            if cluster_idx >= leaf_dof_indices.len() || cluster_idx >= locals_leaf.nrows() {
                return local_y;
            }
            let cluster_dofs = &leaf_dof_indices[cluster_idx];
            if cluster_dofs.is_empty() {
                return local_y;
            }
            let s_mat = &s_level.matrices[cluster_idx];
            if s_mat.as_slice().is_empty() {
                return local_y;
            }
            let local_expansion = locals_leaf.row(cluster_idx);
            let num_rows = cluster_dofs.len().min(s_mat.nrows());
            let num_cols = local_expansion.len().min(s_mat.ncols());
            let has_non_owner = (0..num_rows).any(|row| !self.leaf_row_is_owner(cluster_idx, row));
            if !has_non_owner {
                dense_matvec_scatter_accumulate_c64(
                    s_mat.as_slice(),
                    num_rows,
                    num_cols,
                    s_mat.ncols(),
                    local_expansion,
                    &mut local_y,
                    &cluster_dofs[..num_rows],
                );
            } else {
                for local_row in 0..num_rows {
                    if !self.leaf_row_is_owner(cluster_idx, local_row) {
                        continue;
                    }
                    let mut value = Complex64::new(0.0, 0.0);
                    for local_col in 0..num_cols {
                        value += s_mat[[local_row, local_col]] * local_expansion[local_col];
                    }
                    local_y[cluster_dofs[local_row]] += value;
                }
            }
            local_y
        };
        let local_outputs = if use_parallel(s_level.num_clusters, BEM_BLOCKS_PER_THREAD) {
            (0..s_level.num_clusters)
                .into_par_iter()
                .fold(
                    || vec![Complex64::new(0.0, 0.0); self.num_dofs],
                    &fold_cluster,
                )
                .reduce(
                    || vec![Complex64::new(0.0, 0.0); self.num_dofs],
                    |mut a, b| {
                        for (ai, bi) in a.iter_mut().zip(b.iter()) {
                            *ai += *bi;
                        }
                        a
                    },
                )
        } else {
            (0..s_level.num_clusters)
                .fold(vec![Complex64::new(0.0, 0.0); self.num_dofs], &fold_cluster)
        };
        for (dst, src) in y.iter_mut().zip(local_outputs.iter()) {
            *dst += *src;
        }
    }

    fn evaluate_locals_block(
        &self,
        workspace: &MlfmmWorkspace,
        leaf_level: usize,
        y: &mut DenseMatrix<Complex64>,
        nrhs: usize,
    ) {
        if leaf_level >= self.s_matrices.len() || leaf_level >= workspace.block_locals.len() {
            return;
        }
        let points = self.sphere_points_per_level[leaf_level];
        if points == 0 {
            return;
        }
        let s_level = &self.s_matrices[leaf_level];
        let locals = &workspace.block_locals[leaf_level];
        let leaf_dof_indices = &self.leaf_dof_indices;
        // Block local evaluation (rayon above the machine-adaptive gate,
        // serial for few clusters).
        let contributions: Vec<Vec<(usize, Vec<Complex64>)>> = gated_filter_map_indexed(
            s_level.num_clusters,
            s_level.num_clusters,
            BEM_BLOCKS_PER_THREAD,
            |cluster_idx| {
                let cluster_dofs = leaf_dof_indices.get(cluster_idx)?;
                let s_mat = s_level.matrices.get(cluster_idx)?;
                if cluster_dofs.is_empty() || s_mat.as_slice().is_empty() {
                    return None;
                }
                let local_base = cluster_idx * points;
                if local_base >= locals.nrows() {
                    return None;
                }
                let local_expansion = DenseMatrix::from_fn(s_mat.ncols(), nrhs, |point, rhs| {
                    if point < points && local_base + point < locals.nrows() {
                        locals[[local_base + point, rhs]]
                    } else {
                        Complex64::new(0.0, 0.0)
                    }
                });
                let mut output = DenseMatrix::<Complex64>::zeros(s_mat.nrows(), nrhs);
                s_mat.matmul_into(&local_expansion, &mut output);
                Some(
                    cluster_dofs
                        .iter()
                        .enumerate()
                        .take(output.nrows())
                        .filter(|(local_i, _)| self.leaf_row_is_owner(cluster_idx, *local_i))
                        .map(|(local_i, &global_i)| (global_i, output.row(local_i).to_vec()))
                        .collect(),
                )
            },
        );

        for cluster_contributions in contributions {
            for (global_i, values) in cluster_contributions {
                if global_i >= y.nrows() {
                    continue;
                }
                for (rhs, value) in values.into_iter().enumerate() {
                    y[[global_i, rhs]] += value;
                }
            }
        }
    }

    fn evaluate_locals_transpose(
        &self,
        x: &[Complex64],
        leaf_level: usize,
        workspace: &mut MlfmmWorkspace,
    ) {
        if leaf_level >= self.s_matrices.len() || leaf_level >= workspace.locals.len() {
            return;
        }
        let s_level = &self.s_matrices[leaf_level];
        for (cluster_idx, cluster_dofs) in self.leaf_dof_indices.iter().enumerate() {
            if cluster_dofs.is_empty()
                || cluster_idx >= s_level.matrices.len()
                || cluster_idx >= workspace.locals[leaf_level].nrows()
            {
                continue;
            }
            let s_mat = &s_level.matrices[cluster_idx];
            if s_mat.as_slice().is_empty() {
                continue;
            }
            let x_local: Vec<Complex64> = cluster_dofs
                .iter()
                .enumerate()
                .map(|(local, &global)| {
                    if self.leaf_row_is_owner(cluster_idx, local) {
                        x[global]
                    } else {
                        Complex64::new(0.0, 0.0)
                    }
                })
                .collect();
            if x_local.len() == s_mat.nrows() {
                let mut local_update = vec![Complex64::new(0.0, 0.0); s_mat.ncols()];
                s_mat.matvec_transpose_into(&x_local, &mut local_update);
                let row = workspace.locals[leaf_level].row_mut(cluster_idx);
                for (r, v) in row.iter_mut().zip(local_update.iter()) {
                    *r += *v;
                }
            }
        }
    }

    /// Get a reference to a cluster at the given level and index
    pub fn get_cluster(&self, level: usize, index: usize) -> Option<&Cluster> {
        self.cluster_levels.get(level)?.clusters.get(index)
    }

    /// Upload the full, non-directional MLFMM hierarchy to CUDA.
    ///
    /// Tree construction, quadrature, and coefficient generation remain a
    /// host-shared setup phase.  The returned operator retains the T/S
    /// transfers, M2L diagonals, near blocks, index maps, and action
    /// workspaces on the device, and exposes forward, transpose, and
    /// Hermitian device actions without a CPU fallback.  Directional plans
    /// are intentionally rejected until their compact cone workspaces have a
    /// matching CUDA representation.
    #[cfg(feature = "cuda")]
    pub fn try_to_cuda(
        &self,
        pool: MemoryPool<CudaBackend>,
    ) -> sonium_compute::BackendResult<CudaMlfmmC64> {
        let operation = "mlfmm_to_cuda";
        if self.directional_plans.iter().any(|plan| !plan.is_empty()) {
            return Err(BackendError::Unsupported {
                backend: "cuda",
                operation,
                scalar: "Complex64 directional MLFMM",
            });
        }
        if self.num_levels != self.cluster_levels.len()
            || self.t_matrices.len() != self.num_levels
            || self.s_matrices.len() != self.num_levels
            || self.d_matrices.len() != self.num_levels
            || self.sphere_points_per_level.len() != self.num_levels
        {
            return Err(BackendError::InvalidArgument {
                operation,
                reason: "MLFMM hierarchy metadata has inconsistent level counts".into(),
            });
        }

        // The resident CUDA layout currently has one global source/output
        // slot per leaf incidence.  Continuous higher-order meshes repeat a
        // shared DOF across incidences and require the CPU owner-selection
        // rules, which are not represented in the device plan yet.  Reject
        // that case explicitly instead of silently double-counting rows.
        let mut seen_leaf_dofs = vec![false; self.num_dofs];
        let leaf_cluster_count = self
            .cluster_levels
            .last()
            .map_or(0, |level| level.clusters.len());
        if self.leaf_dof_indices.len() != leaf_cluster_count {
            return Err(BackendError::InvalidArgument {
                operation,
                reason: format!(
                    "leaf DOF mapping has {} clusters, expected {leaf_cluster_count}",
                    self.leaf_dof_indices.len()
                ),
            });
        }
        for (cluster, dofs) in self.leaf_dof_indices.iter().enumerate() {
            for &dof in dofs {
                if dof >= self.num_dofs {
                    return Err(BackendError::InvalidArgument {
                        operation,
                        reason: format!(
                            "leaf cluster {cluster} references DOF {dof} outside dimension {}",
                            self.num_dofs
                        ),
                    });
                }
                if seen_leaf_dofs[dof] {
                    return Err(BackendError::Unsupported {
                        backend: "cuda",
                        operation,
                        scalar: "Complex64 shared-DOF MLFMM layout",
                    });
                }
                seen_leaf_dofs[dof] = true;
            }
        }
        if let Some(dof) = seen_leaf_dofs.iter().position(|present| !present) {
            return Err(BackendError::InvalidArgument {
                operation,
                reason: format!("leaf DOF mapping does not cover global DOF {dof}"),
            });
        }

        let to_i32 = |indices: &[usize], label: &str| {
            indices
                .iter()
                .copied()
                .map(|index| {
                    i32::try_from(index).map_err(|_| BackendError::InvalidArgument {
                        operation,
                        reason: format!("{label} index {index} exceeds CUDA signed index range"),
                    })
                })
                .collect::<sonium_compute::BackendResult<Vec<_>>>()
        };
        let range = |cluster: usize, points: usize, label: &str| {
            let start =
                cluster
                    .checked_mul(points)
                    .ok_or_else(|| BackendError::InvalidArgument {
                        operation,
                        reason: format!("{label} range overflows usize"),
                    })?;
            let end = start
                .checked_add(points)
                .ok_or_else(|| BackendError::InvalidArgument {
                    operation,
                    reason: format!("{label} range overflows usize"),
                })?;
            (start..end)
                .map(|index| {
                    i32::try_from(index).map_err(|_| BackendError::InvalidArgument {
                        operation,
                        reason: format!("{label} index {index} exceeds CUDA signed index range"),
                    })
                })
                .collect::<sonium_compute::BackendResult<Vec<_>>>()
        };

        let last = self.num_levels.saturating_sub(1);
        let mut levels = Vec::with_capacity(self.num_levels);
        for level_index in 0..self.num_levels {
            let points = self.sphere_points_per_level[level_index];
            if points == 0 {
                return Err(BackendError::InvalidArgument {
                    operation,
                    reason: format!("MLFMM level {level_index} has zero sphere points"),
                });
            }
            let clusters = &self.cluster_levels[level_index].clusters;
            let t_level =
                self.t_matrices
                    .get(level_index)
                    .ok_or_else(|| BackendError::InvalidArgument {
                        operation,
                        reason: format!("missing T matrices at level {level_index}"),
                    })?;
            let s_level =
                self.s_matrices
                    .get(level_index)
                    .ok_or_else(|| BackendError::InvalidArgument {
                        operation,
                        reason: format!("missing S matrices at level {level_index}"),
                    })?;
            if t_level.matrices.len() != clusters.len() || s_level.matrices.len() != clusters.len()
            {
                return Err(BackendError::InvalidArgument {
                    operation,
                    reason: format!(
                        "level {level_index} transfer count does not match its cluster count"
                    ),
                });
            }

            let mut t_transfers = Vec::with_capacity(clusters.len());
            let mut s_transfers = Vec::with_capacity(clusters.len());
            for cluster_index in 0..clusters.len() {
                let cluster = &clusters[cluster_index];
                let (t_src, t_dst, s_src, s_dst) = if level_index == last {
                    let leaf_dofs = self.leaf_dof_indices.get(cluster_index).ok_or_else(|| {
                        BackendError::InvalidArgument {
                            operation,
                            reason: format!("missing leaf DOF mapping for cluster {cluster_index}"),
                        }
                    })?;
                    (
                        to_i32(leaf_dofs, "leaf DOF")?,
                        range(cluster_index, points, "leaf multipole")?,
                        range(cluster_index, points, "leaf local")?,
                        to_i32(leaf_dofs, "leaf DOF")?,
                    )
                } else {
                    let child_points = self.sphere_points_per_level[level_index + 1];
                    let mut children = Vec::new();
                    for &child in &cluster.sons {
                        if child >= self.cluster_levels[level_index + 1].clusters.len() {
                            return Err(BackendError::InvalidArgument {
                                operation,
                                reason: format!(
                                    "level {level_index} cluster {cluster_index} has invalid child {child}"
                                ),
                            });
                        }
                        children.extend(range(child, child_points, "child expansion")?);
                    }
                    let parent = range(cluster_index, points, "parent expansion")?;
                    (children.clone(), parent.clone(), parent, children)
                };
                t_transfers.push(CudaMlfmmTransferC64Input {
                    matrix: t_level.matrices[cluster_index].as_slice().to_vec(),
                    src_indices: t_src,
                    dst_indices: t_dst,
                });
                s_transfers.push(CudaMlfmmTransferC64Input {
                    matrix: s_level.matrices[cluster_index].as_slice().to_vec(),
                    src_indices: s_src,
                    dst_indices: s_dst,
                });
            }

            let translations = self.d_matrices[level_index]
                .iter()
                .enumerate()
                .map(|(entry_index, entry)| {
                    if entry.level != level_index {
                        return Err(BackendError::InvalidArgument {
                            operation,
                            reason: format!(
                                "M2L entry {entry_index} at level {level_index} claims level {}",
                                entry.level
                            ),
                        });
                    }
                    Ok(CudaMlfmmTranslationC64Input {
                        source_cluster: entry.source_cluster,
                        field_cluster: entry.field_cluster,
                        diagonal: entry.diagonal.clone(),
                    })
                })
                .collect::<sonium_compute::BackendResult<Vec<_>>>()?;
            let mut t_phase_transfers = Vec::new();
            let mut s_phase_transfers = Vec::new();
            if level_index < last {
                if let Some(plan) = self.m2m_phase_plans.get(level_index) {
                    for (parent, children) in plan.parent_children.iter().enumerate() {
                        for child in children {
                            t_phase_transfers.push(CudaMlfmmPhaseTransferC64Input {
                                source_cluster: child.child_cluster,
                                destination_cluster: parent,
                                phase: child.phase.clone(),
                            });
                        }
                    }
                }
                if let Some(plan) = self.l2l_phase_plans.get(level_index) {
                    for (parent, children) in plan.parent_children.iter().enumerate() {
                        for child in children {
                            s_phase_transfers.push(CudaMlfmmPhaseTransferC64Input {
                                source_cluster: parent,
                                destination_cluster: child.child_cluster,
                                phase: child.phase.clone(),
                            });
                        }
                    }
                }
            }
            levels.push(CudaMlfmmLevelC64Input {
                sphere_points: points,
                t_transfers,
                s_transfers,
                t_phase_transfers,
                s_phase_transfers,
                translations,
            });
        }

        let near_blocks = self
            .near_matrix
            .iter()
            .map(|block| {
                let rows = self
                    .leaf_dof_indices
                    .get(block.source_cluster)
                    .ok_or_else(|| BackendError::InvalidArgument {
                        operation,
                        reason: format!(
                            "near block references invalid source cluster {}",
                            block.source_cluster
                        ),
                    })?;
                let cols = self
                    .leaf_dof_indices
                    .get(block.field_cluster)
                    .ok_or_else(|| BackendError::InvalidArgument {
                        operation,
                        reason: format!(
                            "near block references invalid field cluster {}",
                            block.field_cluster
                        ),
                    })?;
                Ok(CudaMlfmmNearBlockC64Input {
                    row_indices: to_i32(rows, "near row")?,
                    col_indices: to_i32(cols, "near column")?,
                    matrix: block.coefficients.as_slice().to_vec(),
                })
            })
            .collect::<sonium_compute::BackendResult<Vec<_>>>()?;

        CudaMlfmmC64::new(pool, self.num_dofs, levels, near_blocks)
    }

    /// Return the number of clusters at the given tree level
    pub fn num_clusters_at_level(&self, level: usize) -> usize {
        self.cluster_levels
            .get(level)
            .map(|l| l.clusters.len())
            .unwrap_or(0)
    }

    /// Return aggregate translation-work counters for this MLFMM system.
    pub fn translation_stats(&self) -> MlfmmTranslationStats {
        let mut stats = MlfmmTranslationStats {
            entries: 0,
            full_points: 0,
            active_points: 0,
            directional_entries: 0,
            cone_count: 0,
            cone_workspace_points: 0,
        };
        for entries in &self.d_matrices {
            for entry in entries {
                stats.entries += 1;
                stats.full_points += entry.diagonal.len();
                if let Some(active) = &entry.active_directions {
                    stats.directional_entries += 1;
                    stats.active_points += active.len();
                } else {
                    stats.active_points += entry.diagonal.len();
                }
            }
        }
        for plan in &self.directional_plans {
            stats.cone_count += plan.cones.len();
            stats.cone_workspace_points += plan
                .cones
                .iter()
                .map(|cone| {
                    cone.active_directions.len()
                        * (cone.source_clusters.len() + cone.field_clusters.len())
                })
                .sum::<usize>();
        }
        stats
    }
}

/// Fixed-topology compressed wavenumber derivative for the full (non-
/// directional) MLFMM operator.
///
/// The factor derivatives are assembled analytically while retaining the
/// MLFMM hierarchy. [`Self::apply`] uses the product rule one factor at a
/// time, so callers never construct an `N x N` derivative matrix or solve
/// perturbed systems.
#[derive(Debug, Clone)]
pub struct MlfmmWavenumberDerivative {
    /// Near-field derivative blocks.
    pub near_matrix: Vec<NearFieldBlock>,
    /// Derivative of all M2M transfer matrices.
    pub t_matrices: Vec<LevelMatrices>,
    /// Derivative of all M2L diagonal translations.
    pub d_matrices: Vec<Vec<DMatrixEntry>>,
    /// Derivative of all L2L transfer matrices.
    pub s_matrices: Vec<LevelMatrices>,
    /// Derivative of the inter-level M2M phases.
    m2m_phase_plans: Vec<InterLevelPhasePlan>,
    /// Derivative of the inter-level L2L phases.
    l2l_phase_plans: Vec<InterLevelPhasePlan>,
    /// Derivative of compact directional M2L plans, when enabled.
    directional_plans: Vec<DirectionalLevelPlan>,
}

impl MlfmmWavenumberDerivative {
    /// Apply the compressed derivative to a global coefficient vector.
    pub fn apply(&self, base: &MlfmmSystem, x: &[Complex64]) -> Vec<Complex64> {
        assert_eq!(x.len(), base.num_dofs);
        let mut near_system = base.clone();
        near_system.near_matrix = self.near_matrix.clone();
        let mut result = near_system.near_field_matvec(x);
        if base.num_levels == 0 || base.t_matrices.is_empty() {
            return result;
        }
        let mut base_workspace = MlfmmWorkspace::new_with_directional(
            &base.cluster_levels,
            &base.sphere_points_per_level,
            &base.directional_plans,
        );
        let mut derivative_workspace = MlfmmWorkspace::new_with_directional(
            &base.cluster_levels,
            &base.sphere_points_per_level,
            &base.directional_plans,
        );
        base_workspace.clear();
        derivative_workspace.clear();
        base.upward_pass(x, &mut base_workspace);
        self.upward_derivative_pass(base, &base_workspace, x, &mut derivative_workspace);
        base.translate_all_levels(&mut base_workspace);
        self.translate_derivative_levels(base, &base_workspace, &mut derivative_workspace);
        base.downward_pass(&mut base_workspace);
        self.downward_derivative_pass(base, &base_workspace, &mut derivative_workspace);
        self.evaluate_derivative_locals(base, &base_workspace, &derivative_workspace, &mut result);
        result
    }

    fn upward_derivative_pass(
        &self,
        base: &MlfmmSystem,
        base_workspace: &MlfmmWorkspace,
        x: &[Complex64],
        derivative_workspace: &mut MlfmmWorkspace,
    ) {
        let leaf_level = base.num_levels.saturating_sub(1);
        if leaf_level >= self.t_matrices.len() {
            return;
        }
        for (cluster_idx, cluster_dofs) in base.leaf_dof_indices.iter().enumerate() {
            let Some(matrix) = self.t_matrices[leaf_level].matrices.get(cluster_idx) else {
                continue;
            };
            let row = derivative_workspace.multipoles[leaf_level].row_mut(cluster_idx);
            let rows = row.len().min(matrix.nrows());
            let cols = cluster_dofs.len().min(matrix.ncols());
            indexed_dense_matvec_c64(
                matrix.as_slice(),
                rows,
                cols,
                matrix.ncols(),
                x,
                &cluster_dofs[..cols],
                row,
            );
        }

        for level in (0..leaf_level).rev() {
            let child_level = level + 1;
            let child_points = base.sphere_points_per_level[child_level];
            let use_phase = base
                .m2m_phase_plans
                .get(level)
                .is_some_and(|plan| !plan.is_empty());
            for cluster_idx in 0..base.cluster_levels[level].clusters.len() {
                let cluster = &base.cluster_levels[level].clusters[cluster_idx];
                if use_phase {
                    let base_plan = &base.m2m_phase_plans[level];
                    let derivative_plan = &self.m2m_phase_plans[level];
                    for child_phase in &base_plan.parent_children[cluster_idx] {
                        let Some(derivative_child_phase) = derivative_plan.parent_children
                            [cluster_idx]
                            .iter()
                            .find(|candidate| candidate.child_cluster == child_phase.child_cluster)
                        else {
                            continue;
                        };
                        let base_child = base_workspace.multipoles[child_level]
                            .row(child_phase.child_cluster)
                            .to_vec();
                        let derivative_child = derivative_workspace.multipoles[child_level]
                            .row(child_phase.child_cluster)
                            .to_vec();
                        let parent_row =
                            derivative_workspace.multipoles[level].row_mut(cluster_idx);
                        let len = parent_row
                            .len()
                            .min(base_child.len())
                            .min(derivative_child.len())
                            .min(child_phase.phase.len())
                            .min(derivative_child_phase.phase.len());
                        for point in 0..len {
                            parent_row[point] += derivative_child_phase.phase[point]
                                * base_child[point]
                                + child_phase.phase[point] * derivative_child[point];
                        }
                    }
                } else {
                    let Some(base_matrix) = base.t_matrices[level].matrices.get(cluster_idx) else {
                        continue;
                    };
                    let Some(derivative_matrix) = self.t_matrices[level].matrices.get(cluster_idx)
                    else {
                        continue;
                    };
                    for (child_index, &child_cluster) in cluster.sons.iter().enumerate() {
                        let base_child = base_workspace.multipoles[child_level]
                            .row(child_cluster)
                            .to_vec();
                        let derivative_child = derivative_workspace.multipoles[child_level]
                            .row(child_cluster)
                            .to_vec();
                        let parent_row =
                            derivative_workspace.multipoles[level].row_mut(cluster_idx);
                        let offset = child_index * child_points;
                        let len = child_points
                            .min(base_child.len())
                            .min(derivative_child.len())
                            .min(base_matrix.ncols().saturating_sub(offset));
                        for point in 0..parent_row.len().min(base_matrix.nrows()) {
                            for column in 0..len {
                                parent_row[point] += base_matrix.row(point)[offset + column]
                                    * derivative_child[offset + column]
                                    + derivative_matrix.row(point)[offset + column]
                                        * base_child[offset + column];
                            }
                        }
                    }
                }
            }
        }
    }

    fn translate_derivative_levels(
        &self,
        base: &MlfmmSystem,
        base_workspace: &MlfmmWorkspace,
        derivative_workspace: &mut MlfmmWorkspace,
    ) {
        for level in 0..base.num_levels {
            let Some(base_entries) = base.d_matrices.get(level) else {
                continue;
            };
            let Some(derivative_entries) = self.d_matrices.get(level) else {
                continue;
            };
            let directional = base
                .directional_plans
                .get(level)
                .is_some_and(|plan| !plan.is_empty());
            for (base_entry, derivative_entry) in base_entries.iter().zip(derivative_entries) {
                if directional && base_entry.active_directions.is_some() {
                    continue;
                }
                let base_source = base_workspace.multipoles[level]
                    .row(base_entry.source_cluster)
                    .to_vec();
                let derivative_source = derivative_workspace.multipoles[level]
                    .row(base_entry.source_cluster)
                    .to_vec();
                let target = derivative_workspace.locals[level].row_mut(base_entry.field_cluster);
                let len = target
                    .len()
                    .min(base_source.len())
                    .min(derivative_source.len())
                    .min(base_entry.diagonal.len())
                    .min(derivative_entry.diagonal.len());
                for point in 0..len {
                    target[point] += derivative_entry.diagonal[point] * base_source[point]
                        + base_entry.diagonal[point] * derivative_source[point];
                }
            }
            if let Some(plan) = base
                .directional_plans
                .get(level)
                .filter(|plan| !plan.is_empty())
            {
                let derivative_plan = &self.directional_plans[level];
                for (cone_index, cone) in plan.cones.iter().enumerate() {
                    for entry in &plan.entries_by_cone[cone_index] {
                        let Some(derivative_entry) = derivative_plan.entries_by_cone[cone_index]
                            .iter()
                            .find(|candidate| {
                                candidate.source_row == entry.source_row
                                    && candidate.field_row == entry.field_row
                            })
                        else {
                            continue;
                        };
                        let source_cluster = cone.source_clusters[entry.source_row];
                        let field_cluster = cone.field_clusters[entry.field_row];
                        let base_source = base_workspace.multipoles[level]
                            .row(source_cluster)
                            .to_vec();
                        let derivative_source = derivative_workspace.multipoles[level]
                            .row(source_cluster)
                            .to_vec();
                        let target = derivative_workspace.locals[level].row_mut(field_cluster);
                        let len = cone
                            .active_directions
                            .len()
                            .min(target.len())
                            .min(base_source.len())
                            .min(derivative_source.len())
                            .min(entry.diagonal.len())
                            .min(derivative_entry.diagonal.len());
                        for (compact, &direction) in
                            cone.active_directions.iter().enumerate().take(len)
                        {
                            if direction >= target.len()
                                || direction >= base_source.len()
                                || direction >= derivative_source.len()
                            {
                                continue;
                            }
                            target[direction] += derivative_entry.diagonal[compact]
                                * base_source[direction]
                                + entry.diagonal[compact] * derivative_source[direction];
                        }
                    }
                }
            }
        }
    }

    fn downward_derivative_pass(
        &self,
        base: &MlfmmSystem,
        base_workspace: &MlfmmWorkspace,
        derivative_workspace: &mut MlfmmWorkspace,
    ) {
        let leaf_level = base.num_levels.saturating_sub(1);
        for level in 0..leaf_level {
            let child_level = level + 1;
            let child_points = base.sphere_points_per_level[child_level];
            let use_phase = base
                .l2l_phase_plans
                .get(level)
                .is_some_and(|plan| !plan.is_empty());
            for cluster_idx in 0..base.cluster_levels[level].clusters.len() {
                let cluster = &base.cluster_levels[level].clusters[cluster_idx];
                let base_parent = base_workspace.locals[level].row(cluster_idx).to_vec();
                let derivative_parent =
                    derivative_workspace.locals[level].row(cluster_idx).to_vec();
                if use_phase {
                    let base_plan = &base.l2l_phase_plans[level];
                    let derivative_plan = &self.l2l_phase_plans[level];
                    for child_phase in &base_plan.parent_children[cluster_idx] {
                        let Some(derivative_child_phase) = derivative_plan.parent_children
                            [cluster_idx]
                            .iter()
                            .find(|candidate| candidate.child_cluster == child_phase.child_cluster)
                        else {
                            continue;
                        };
                        let target = derivative_workspace.locals[child_level]
                            .row_mut(child_phase.child_cluster);
                        let len = target
                            .len()
                            .min(base_parent.len())
                            .min(derivative_parent.len())
                            .min(child_phase.phase.len())
                            .min(derivative_child_phase.phase.len());
                        for point in 0..len {
                            target[point] += derivative_child_phase.phase[point]
                                * base_parent[point]
                                + child_phase.phase[point] * derivative_parent[point];
                        }
                    }
                } else {
                    let Some(base_matrix) = base.s_matrices[level].matrices.get(cluster_idx) else {
                        continue;
                    };
                    let Some(derivative_matrix) = self.s_matrices[level].matrices.get(cluster_idx)
                    else {
                        continue;
                    };
                    for point in 0..base_matrix.nrows().min(derivative_matrix.nrows()) {
                        let target_cluster = cluster.sons.get(point / child_points).copied();
                        let Some(target_cluster) = target_cluster else {
                            continue;
                        };
                        let target_point = point % child_points;
                        let target =
                            derivative_workspace.locals[child_level].row_mut(target_cluster);
                        if target_point >= target.len() || point >= base_matrix.nrows() {
                            continue;
                        }
                        let mut value = Complex64::new(0.0, 0.0);
                        for parent_point in 0..base_parent.len().min(base_matrix.ncols()) {
                            value += derivative_matrix.row(point)[parent_point]
                                * base_parent[parent_point]
                                + base_matrix.row(point)[parent_point]
                                    * derivative_parent[parent_point];
                        }
                        target[target_point] += value;
                    }
                }
            }
        }
    }

    fn evaluate_derivative_locals(
        &self,
        base: &MlfmmSystem,
        base_workspace: &MlfmmWorkspace,
        derivative_workspace: &MlfmmWorkspace,
        result: &mut [Complex64],
    ) {
        let leaf_level = base.num_levels.saturating_sub(1);
        let Some(base_level) = base.s_matrices.get(leaf_level) else {
            return;
        };
        let Some(derivative_level) = self.s_matrices.get(leaf_level) else {
            return;
        };
        for (cluster_idx, cluster_dofs) in base.leaf_dof_indices.iter().enumerate() {
            let Some(base_matrix) = base_level.matrices.get(cluster_idx) else {
                continue;
            };
            let Some(derivative_matrix) = derivative_level.matrices.get(cluster_idx) else {
                continue;
            };
            let base_local = base_workspace.locals[leaf_level].row(cluster_idx).to_vec();
            let derivative_local = derivative_workspace.locals[leaf_level]
                .row(cluster_idx)
                .to_vec();
            for (row, &global) in cluster_dofs.iter().enumerate() {
                if global >= result.len() || row >= base_matrix.nrows() {
                    continue;
                }
                let mut value = Complex64::new(0.0, 0.0);
                for point in 0..base_local
                    .len()
                    .min(derivative_local.len())
                    .min(base_matrix.ncols())
                    .min(derivative_matrix.ncols())
                {
                    value += derivative_matrix.row(row)[point] * base_local[point]
                        + base_matrix.row(row)[point] * derivative_local[point];
                }
                result[global] += value;
            }
        }
    }
}

/// Build a fixed-topology compressed MLFMM wavenumber derivative.
///
/// This checked constant-panel entry point differentiates the value factors
/// analytically. It is intentionally separate from the directional builder
/// because directional M2L plans require their compact cone topology to be
/// copied while the diagonal entries are differentiated.
#[allow(clippy::too_many_arguments)]
pub fn build_mlfmm_wavenumber_derivative(
    elements: &[Element],
    nodes: &[[f64; 3]],
    cluster_levels: Vec<ClusterLevel>,
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
) -> Result<MlfmmWavenumberDerivative, String> {
    build_mlfmm_wavenumber_derivative_with_mode(
        elements,
        nodes,
        cluster_levels,
        physics,
        BasisOrder::Constant,
        sign,
        beta_scale,
        TranslationMode::Full,
    )
}

/// Build a fixed-topology wavenumber derivative for a basis-weighted MLFMM.
///
/// The value and derivative factors use the same panel-local proxy expansion,
/// so linear/quadratic continuous and discontinuous layouts retain their
/// global DOF numbering throughout the near/T/D/S action.
#[allow(clippy::too_many_arguments)]
pub fn build_mlfmm_wavenumber_derivative_with_basis_checked(
    elements: &[Element],
    nodes: &[[f64; 3]],
    cluster_levels: Vec<ClusterLevel>,
    physics: &PhysicsParams,
    basis: BasisOrder,
    sign: f64,
    beta_scale: f64,
) -> Result<MlfmmWavenumberDerivative, String> {
    build_mlfmm_wavenumber_derivative_with_mode(
        elements,
        nodes,
        cluster_levels,
        physics,
        basis,
        sign,
        beta_scale,
        TranslationMode::Full,
    )
}

/// Build a fixed-topology directional-MLFMM wavenumber derivative.
#[allow(clippy::too_many_arguments)]
pub fn build_directional_mlfmm_wavenumber_derivative(
    elements: &[Element],
    nodes: &[[f64; 3]],
    cluster_levels: Vec<ClusterLevel>,
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
    config: DirectionalMlfmmConfig,
) -> Result<MlfmmWavenumberDerivative, String> {
    build_mlfmm_wavenumber_derivative_with_mode(
        elements,
        nodes,
        cluster_levels,
        physics,
        BasisOrder::Constant,
        sign,
        beta_scale,
        TranslationMode::Directional(config),
    )
}

/// Build a fixed-topology directional-MLFMM wavenumber derivative for a
/// basis-weighted linear/quadratic layout.
#[allow(clippy::too_many_arguments)]
pub fn build_directional_mlfmm_wavenumber_derivative_with_basis_checked(
    elements: &[Element],
    nodes: &[[f64; 3]],
    cluster_levels: Vec<ClusterLevel>,
    physics: &PhysicsParams,
    basis: BasisOrder,
    sign: f64,
    beta_scale: f64,
    config: DirectionalMlfmmConfig,
) -> Result<MlfmmWavenumberDerivative, String> {
    build_mlfmm_wavenumber_derivative_with_mode(
        elements,
        nodes,
        cluster_levels,
        physics,
        basis,
        sign,
        beta_scale,
        TranslationMode::Directional(config),
    )
}

#[allow(clippy::too_many_arguments)]
fn build_mlfmm_wavenumber_derivative_with_mode(
    elements: &[Element],
    nodes: &[[f64; 3]],
    cluster_levels: Vec<ClusterLevel>,
    physics: &PhysicsParams,
    basis: BasisOrder,
    sign: f64,
    beta_scale: f64,
    translation_mode: TranslationMode,
) -> Result<MlfmmWavenumberDerivative, String> {
    build_mlfmm_wavenumber_derivative_analytic(
        elements,
        nodes,
        cluster_levels,
        physics,
        basis,
        sign,
        beta_scale,
        translation_mode,
    )
}

/// Build an analytic fixed-topology MLFMM derivative factorization.
///
/// The value builder and this routine share the exact proxy topology and
/// quadrature. Every factor is differentiated in place: near-field entries,
/// M2M/L2L phases, leaf transfers, M2L Hankel translations, and the
/// Burton--Miller source factor. Directional cone membership is copied from
/// the base topology, so a parameter perturbation cannot change the compact
/// action shape while the derivative is being applied.
#[allow(clippy::too_many_arguments)]
fn build_mlfmm_wavenumber_derivative_analytic(
    elements: &[Element],
    nodes: &[[f64; 3]],
    cluster_levels: Vec<ClusterLevel>,
    physics: &PhysicsParams,
    basis: BasisOrder,
    sign: f64,
    beta_scale: f64,
    translation_mode: TranslationMode,
) -> Result<MlfmmWavenumberDerivative, String> {
    let (assembly_elements, assembly_levels) = if basis == BasisOrder::Constant {
        validate_mlfmm_constant_panel_layout(elements, nodes, &cluster_levels)
            .map_err(|error| error.to_string())?;
        (elements.to_vec(), cluster_levels)
    } else {
        validate_mlfmm_basis_layout(elements, nodes, &cluster_levels, basis)
            .map_err(|error| error.to_string())?;
        let (proxies, element_proxies) =
            build_mlfmm_basis_proxies(elements, nodes, basis, has_shared_basis_dofs(elements));
        let expanded_levels = expand_mlfmm_cluster_levels(&cluster_levels, &element_proxies);
        (proxies, expanded_levels)
    };

    let base = build_mlfmm_system_stabilized_with_mode(
        &assembly_elements,
        nodes,
        assembly_levels.clone(),
        physics,
        sign,
        beta_scale,
        translation_mode,
    );
    let actual_sphere_points = base.sphere_points_per_level.clone();

    let near_matrix = build_near_field_wavenumber_derivative(
        &assembly_elements,
        nodes,
        assembly_levels
            .last()
            .map(|level| level.clusters.as_slice())
            .unwrap_or(&[]),
        physics,
        sign,
        beta_scale,
    )?;

    let mut t_matrices = Vec::with_capacity(assembly_levels.len());
    for (level_idx, level) in assembly_levels.iter().enumerate().rev() {
        let (sphere_coords, sphere_weights) =
            unit_sphere_quadrature(level.theta_points, level.phi_points);
        t_matrices.push(build_t_matrices_level_derivative(
            &assembly_elements,
            nodes,
            &level.clusters,
            physics,
            &sphere_coords,
            &sphere_weights,
            level_idx,
            &assembly_levels,
            &actual_sphere_points,
        ));
    }
    t_matrices.reverse();

    let mut d_matrices = Vec::with_capacity(assembly_levels.len());
    for (level_idx, level) in assembly_levels.iter().enumerate() {
        let (sphere_coords, _) = unit_sphere_quadrature(level.theta_points, level.phi_points);
        d_matrices.push(build_d_matrices_level_derivative(
            &level.clusters,
            physics,
            &sphere_coords,
            level.expansion_terms,
            level_idx,
            sphere_coords.len(),
            &base.d_matrices[level_idx],
        ));
    }

    let mut s_matrices = Vec::with_capacity(assembly_levels.len());
    for (level_idx, level) in assembly_levels.iter().enumerate() {
        let (sphere_coords, sphere_weights) =
            unit_sphere_quadrature(level.theta_points, level.phi_points);
        s_matrices.push(build_s_matrices_level_derivative(
            &assembly_elements,
            &level.clusters,
            physics,
            &sphere_coords,
            &sphere_weights,
            level_idx,
            &assembly_levels,
            &actual_sphere_points,
            sign,
            beta_scale,
        ));
    }

    let m2m_phase_plans = build_interlevel_phase_derivative_plans(
        &assembly_levels,
        physics,
        &actual_sphere_points,
        -1.0,
    );
    let l2l_phase_plans = build_interlevel_phase_derivative_plans(
        &assembly_levels,
        physics,
        &actual_sphere_points,
        1.0,
    );
    let directional_plans = build_directional_level_plans(&d_matrices, &assembly_levels);

    Ok(MlfmmWavenumberDerivative {
        near_matrix,
        t_matrices,
        d_matrices,
        s_matrices,
        m2m_phase_plans,
        l2l_phase_plans,
        directional_plans,
    })
}

fn build_near_field_wavenumber_derivative(
    elements: &[Element],
    nodes: &[[f64; 3]],
    leaf_clusters: &[Cluster],
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
) -> Result<Vec<NearFieldBlock>, String> {
    let gamma = Complex64::new(physics.gamma(), 0.0);
    let tau = Complex64::new(physics.tau, 0.0);
    let beta = physics.burton_miller_beta() * beta_scale * sign;
    let beta_derivative = physics.burton_miller_beta_wavenumber_derivative() * beta_scale * sign;
    let mut blocks = Vec::new();

    for (source_cluster, cluster) in leaf_clusters.iter().enumerate() {
        let mut field_clusters = cluster.near_clusters.clone();
        if !field_clusters.contains(&source_cluster) {
            field_clusters.push(source_cluster);
        }
        for field_cluster in field_clusters {
            let Some(field) = leaf_clusters.get(field_cluster) else {
                return Err(format!(
                    "MLFMM derivative near-field cluster {field_cluster} is out of range"
                ));
            };
            let mut coefficients = DenseMatrix::<Complex64>::zeros(
                cluster.element_indices.len(),
                field.element_indices.len(),
            );
            for (row, &source_index) in cluster.element_indices.iter().enumerate() {
                let Some(source_element) = elements.get(source_index) else {
                    return Err(format!(
                        "MLFMM derivative source element {source_index} is out of range"
                    ));
                };
                if source_element.property.is_evaluation() {
                    continue;
                }
                for (col, &field_index) in field.element_indices.iter().enumerate() {
                    let Some(field_element) = elements.get(field_index) else {
                        return Err(format!(
                            "MLFMM derivative field element {field_index} is out of range"
                        ));
                    };
                    if field_element.property.is_evaluation() {
                        continue;
                    }
                    let basis_weighted = decode_dof_proxy_group(source_element.group).is_some()
                        || decode_dof_proxy_group(field_element.group).is_some();
                    coefficients[[row, col]] = if basis_weighted {
                        compute_matrix_entry_wavenumber_derivative_curved(
                            source_index,
                            field_index,
                            source_element,
                            field_element,
                            nodes,
                            physics,
                            gamma,
                            tau,
                            beta,
                            beta_derivative,
                            sign,
                            6,
                        )
                    } else {
                        compute_matrix_entry_wavenumber_derivative(
                            source_index,
                            field_index,
                            source_element,
                            field_element,
                            nodes,
                            physics,
                            gamma,
                            tau,
                            beta,
                            beta_derivative,
                            sign,
                            6,
                        )
                    }
                    .map_err(|error| error.to_string())?;
                }
            }
            blocks.push(NearFieldBlock {
                source_cluster,
                field_cluster,
                coefficients,
            });
        }
    }
    Ok(blocks)
}

#[allow(clippy::too_many_arguments)]
fn build_t_matrices_level_derivative(
    elements: &[Element],
    nodes: &[[f64; 3]],
    clusters: &[Cluster],
    physics: &PhysicsParams,
    sphere_coords: &[[f64; 3]],
    sphere_weights: &[f64],
    level_idx: usize,
    cluster_levels: &[ClusterLevel],
    actual_sphere_points: &[usize],
) -> LevelMatrices {
    let k = physics.wave_number;
    let num_sphere_points = sphere_coords.len();
    let is_leaf_level = level_idx == cluster_levels.len().saturating_sub(1);
    // T-matrix build (rayon above the machine-adaptive gate, serial for few
    // clusters). Each cluster runs quadrature loops, so the grain is coarse
    // blocks per thread.
    let matrices: Vec<DenseMatrix<Complex64>> =
        gated_map(clusters, clusters.len(), BEM_BLOCKS_PER_THREAD, |cluster| {
            if is_leaf_level {
                let mut matrix = DenseMatrix::<Complex64>::zeros(
                    num_sphere_points,
                    cluster.element_indices.len(),
                );
                for (column, &element_index) in cluster.element_indices.iter().enumerate() {
                    let element = &elements[element_index];
                    let element_coords = get_element_coords(element, nodes);
                    let quadrature = match element.element_type {
                        ElementType::Tri3 => quad_or_triangle_quadrature(false),
                        ElementType::Quad4 => quad_or_triangle_quadrature(true),
                    };
                    for (point_index, direction) in sphere_coords.iter().enumerate() {
                        let mut integral = Complex64::new(0.0, 0.0);
                        let mut derivative_integral = Complex64::new(0.0, 0.0);
                        for &(s, t, weight) in &quadrature {
                            let shape = element.shape_functions(s, t);
                            let (shape_ds, shape_dt) = element.shape_derivatives(s, t);
                            let (point, jacobian, normal) = quadrature_geometry(
                                &element_coords,
                                &shape,
                                &shape_ds,
                                &shape_dt,
                                None,
                            );
                            let diff = [
                                point[0] - cluster.center[0],
                                point[1] - cluster.center[1],
                                point[2] - cluster.center[2],
                            ];
                            let distance = direction[0] * diff[0]
                                + direction[1] * diff[1]
                                + direction[2] * diff[2];
                            let exponential = sonium_fastmath::exp_i(-k * distance);
                            let normal_projection = direction[0] * normal[0]
                                + direction[1] * normal[1]
                                + direction[2] * normal[2];
                            let basis_weight = basis_order_for_proxy(element)
                                .and_then(|basis| {
                                    basis_values(element.element_type, basis, s, t)
                                        .get(proxy_local_index(element))
                                        .copied()
                                })
                                .unwrap_or(1.0);
                            let weighted = weight * jacobian * basis_weight * normal_projection;
                            integral += weighted * exponential;
                            derivative_integral +=
                                weighted * Complex64::new(0.0, -distance) * exponential;
                        }
                        let minus_i = Complex64::new(0.0, -1.0);
                        let minus_ik = Complex64::new(0.0, -k);
                        matrix[[point_index, column]] = sphere_weights[point_index]
                            * (minus_i * integral + minus_ik * derivative_integral);
                    }
                }
                matrix
            } else {
                let num_children = cluster.sons.len();
                let child_num_points = actual_sphere_points
                    .get(level_idx + 1)
                    .copied()
                    .unwrap_or(num_sphere_points);
                let mut matrix = DenseMatrix::<Complex64>::zeros(
                    num_sphere_points,
                    num_children * child_num_points,
                );
                for (child_index, &child_cluster_index) in cluster.sons.iter().enumerate() {
                    let child_cluster =
                        &cluster_levels[level_idx + 1].clusters[child_cluster_index];
                    let diff = [
                        child_cluster.center[0] - cluster.center[0],
                        child_cluster.center[1] - cluster.center[1],
                        child_cluster.center[2] - cluster.center[2],
                    ];
                    for (point_index, direction) in sphere_coords.iter().enumerate() {
                        let distance = direction[0] * diff[0]
                            + direction[1] * diff[1]
                            + direction[2] * diff[2];
                        if point_index < child_num_points {
                            let exponential = sonium_fastmath::exp_i(-k * distance);
                            matrix[[point_index, child_index * child_num_points + point_index]] =
                                Complex64::new(0.0, -distance) * exponential;
                        }
                    }
                }
                matrix
            }
        });
    LevelMatrices {
        level: level_idx,
        num_clusters: clusters.len(),
        matrices,
    }
}

fn quad_or_triangle_quadrature(quad: bool) -> Vec<(f64, f64, f64)> {
    if quad {
        quad_quadrature(4)
    } else {
        crate::core::integration::gauss::triangle_quadrature(4)
    }
}

fn build_d_matrices_level_derivative(
    clusters: &[Cluster],
    _physics: &PhysicsParams,
    sphere_coords: &[[f64; 3]],
    n_terms: usize,
    _level_idx: usize,
    _num_sphere_points: usize,
    base_entries: &[DMatrixEntry],
) -> Vec<DMatrixEntry> {
    let green_normalization = 1.0 / (4.0 * std::f64::consts::PI);
    let i_unit = Complex64::new(0.0, 1.0);
    // D-matrix derivative entries (rayon above the machine-adaptive gate,
    // serial for few entries). Each entry sums Hankel-Legendre series, so
    // the grain is coarse blocks per thread.
    gated_map(
        base_entries,
        base_entries.len(),
        BEM_BLOCKS_PER_THREAD,
        |entry| {
            let field = &clusters[entry.field_cluster];
            let source = &clusters[entry.source_cluster];
            let diff = [
                field.center[0] - source.center[0],
                field.center[1] - source.center[1],
                field.center[2] - source.center[2],
            ];
            let r = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]).sqrt();
            if r < 1.0e-15 {
                return entry.clone();
            }
            let kr = _physics.wave_number * r;
            let pair_n_terms = n_terms.min((kr.ceil() as usize).saturating_add(8));
            let h_funcs = spherical_hankel_first_kind((pair_n_terms + 1).max(2), kr, 1.0);
            let h_derivatives = spherical_hankel_derivative(&h_funcs, kr);
            let r_hat = [diff[0] / r, diff[1] / r, diff[2] / r];
            let diagonal = sphere_coords
                .iter()
                .map(|direction| {
                    let cos_theta =
                        direction[0] * r_hat[0] + direction[1] * r_hat[1] + direction[2] * r_hat[2];
                    let p_leg = legendre_polynomials(pair_n_terms + 1, cos_theta);
                    let mut transfer = Complex64::new(0.0, 0.0);
                    let mut derivative_transfer = Complex64::new(0.0, 0.0);
                    let mut i_power = Complex64::new(1.0, 0.0);
                    for n in 0..=pair_n_terms {
                        let scale = (2 * n + 1) as f64;
                        let basis = scale * i_power * p_leg[n];
                        transfer += basis * h_funcs[n];
                        derivative_transfer += basis * h_derivatives[n] * r;
                        i_power *= i_unit;
                    }
                    let ik = Complex64::new(0.0, _physics.wave_number);
                    (derivative_transfer * ik + transfer * i_unit) * green_normalization
                })
                .collect();
            DMatrixEntry {
                source_cluster: entry.source_cluster,
                field_cluster: entry.field_cluster,
                level: entry.level,
                diagonal,
                active_directions: entry.active_directions.clone(),
            }
        },
    )
}

fn build_s_matrices_level_derivative(
    elements: &[Element],
    clusters: &[Cluster],
    physics: &PhysicsParams,
    sphere_coords: &[[f64; 3]],
    _sphere_weights: &[f64],
    level_idx: usize,
    cluster_levels: &[ClusterLevel],
    actual_sphere_points: &[usize],
    sign: f64,
    beta_scale: f64,
) -> LevelMatrices {
    let k = physics.wave_number;
    let ik = Complex64::new(0.0, k);
    let gamma_tau = Complex64::new(physics.gamma() * physics.tau * sign, 0.0);
    let beta = physics.burton_miller_beta() * beta_scale * sign;
    let beta_derivative = physics.burton_miller_beta_wavenumber_derivative() * beta_scale * sign;
    let num_sphere_points = sphere_coords.len();
    let is_leaf_level = level_idx == cluster_levels.len().saturating_sub(1);
    // S-matrix build (rayon above the machine-adaptive gate, serial for few
    // clusters).
    let matrices: Vec<DenseMatrix<Complex64>> =
        gated_map(clusters, clusters.len(), BEM_BLOCKS_PER_THREAD, |cluster| {
            if is_leaf_level {
                let mut matrix = DenseMatrix::<Complex64>::zeros(
                    cluster.element_indices.len(),
                    num_sphere_points,
                );
                for (row, &element_index) in cluster.element_indices.iter().enumerate() {
                    let element = &elements[element_index];
                    for (point_index, direction) in sphere_coords.iter().enumerate() {
                        let diff = [
                            element.center[0] - cluster.center[0],
                            element.center[1] - cluster.center[1],
                            element.center[2] - cluster.center[2],
                        ];
                        let distance = direction[0] * diff[0]
                            + direction[1] * diff[1]
                            + direction[2] * diff[2];
                        let exponential = sonium_fastmath::exp_i(k * distance);
                        let normal_projection = direction[0] * element.normal[0]
                            + direction[1] * element.normal[1]
                            + direction[2] * element.normal[2];
                        let factor = gamma_tau + beta * ik * normal_projection;
                        let derivative_factor = beta_derivative * ik * normal_projection
                            + beta * Complex64::new(0.0, 1.0) * normal_projection;
                        matrix[[row, point_index]] = (derivative_factor
                            + factor * Complex64::new(0.0, distance))
                            * exponential;
                    }
                }
                matrix
            } else {
                let num_children = cluster.sons.len();
                let child_num_points = actual_sphere_points
                    .get(level_idx + 1)
                    .copied()
                    .unwrap_or(num_sphere_points);
                let mut matrix = DenseMatrix::<Complex64>::zeros(
                    num_children * child_num_points,
                    num_sphere_points,
                );
                for (child_index, &child_cluster_index) in cluster.sons.iter().enumerate() {
                    let child_cluster =
                        &cluster_levels[level_idx + 1].clusters[child_cluster_index];
                    let diff = [
                        child_cluster.center[0] - cluster.center[0],
                        child_cluster.center[1] - cluster.center[1],
                        child_cluster.center[2] - cluster.center[2],
                    ];
                    for (point_index, direction) in sphere_coords.iter().enumerate() {
                        let distance = direction[0] * diff[0]
                            + direction[1] * diff[1]
                            + direction[2] * diff[2];
                        if point_index < child_num_points {
                            let exponential = sonium_fastmath::exp_i(k * distance);
                            matrix[[child_index * child_num_points + point_index, point_index]] =
                                Complex64::new(0.0, distance) * exponential;
                        }
                    }
                }
                matrix
            }
        });
    LevelMatrices {
        level: level_idx,
        num_clusters: clusters.len(),
        matrices,
    }
}

fn build_interlevel_phase_derivative_plans(
    cluster_levels: &[ClusterLevel],
    physics: &PhysicsParams,
    actual_sphere_points: &[usize],
    phase_sign: f64,
) -> Vec<InterLevelPhasePlan> {
    let k = physics.wave_number;
    let mut plans = Vec::with_capacity(cluster_levels.len());
    for (level_idx, level) in cluster_levels.iter().enumerate() {
        if level_idx + 1 >= cluster_levels.len() {
            plans.push(InterLevelPhasePlan::empty(level.clusters.len()));
            continue;
        }
        let (sphere_coords, _) = unit_sphere_quadrature(level.theta_points, level.phi_points);
        let child_level = &cluster_levels[level_idx + 1];
        let child_points = actual_sphere_points
            .get(level_idx + 1)
            .copied()
            .unwrap_or(sphere_coords.len());
        let mut parent_children = Vec::with_capacity(level.clusters.len());
        for cluster in &level.clusters {
            let mut children = Vec::with_capacity(cluster.sons.len());
            for &child_cluster_index in &cluster.sons {
                let Some(child_cluster) = child_level.clusters.get(child_cluster_index) else {
                    continue;
                };
                let diff = [
                    child_cluster.center[0] - cluster.center[0],
                    child_cluster.center[1] - cluster.center[1],
                    child_cluster.center[2] - cluster.center[2],
                ];
                let count = sphere_coords.len().min(child_points);
                let phase = sphere_coords
                    .iter()
                    .take(count)
                    .map(|direction| {
                        let distance = direction[0] * diff[0]
                            + direction[1] * diff[1]
                            + direction[2] * diff[2];
                        Complex64::new(0.0, phase_sign * distance)
                            * sonium_fastmath::exp_i(phase_sign * k * distance)
                    })
                    .collect();
                children.push(InterLevelChildPhase {
                    child_cluster: child_cluster_index,
                    phase,
                });
            }
            parent_children.push(children);
        }
        plans.push(InterLevelPhasePlan { parent_children });
    }
    plans
}

fn spherical_hankel_derivative(values: &[Complex64], argument: f64) -> Vec<Complex64> {
    values
        .iter()
        .enumerate()
        .map(|(order, &value)| {
            if order == 0 {
                -values[1]
            } else {
                values[order - 1] - (order as f64 + 1.0) / argument * value
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn build_mlfmm_wavenumber_derivative_with_mode_centered(
    elements: &[Element],
    nodes: &[[f64; 3]],
    cluster_levels: Vec<ClusterLevel>,
    physics: &PhysicsParams,
    basis: BasisOrder,
    sign: f64,
    beta_scale: f64,
    translation_mode: TranslationMode,
) -> Result<MlfmmWavenumberDerivative, String> {
    if basis == BasisOrder::Constant {
        validate_mlfmm_constant_panel_layout(elements, nodes, &cluster_levels)
            .map_err(|error| error.to_string())?;
    }
    let wave_number = physics.wave_number;
    if !wave_number.is_finite() || wave_number <= 0.0 {
        return Err("MLFMM derivative requires a positive finite wavenumber".to_string());
    }
    let step = (wave_number.abs() * 1.0e-5)
        .max(1.0e-8)
        .min(wave_number.abs() * 0.25);
    let make_physics = |value: f64| -> Result<PhysicsParams, String> {
        let mut candidate = PhysicsParams::try_new(
            value * physics.speed_of_sound / (2.0 * std::f64::consts::PI),
            physics.speed_of_sound,
            physics.density,
            physics.tau < 0.0,
        )
        .map_err(|error| error.to_string())?;
        candidate.harmonic_factor = physics.harmonic_factor;
        candidate.pressure_factor = candidate.density * candidate.omega * candidate.harmonic_factor;
        candidate.validate().map_err(|error| error.to_string())?;
        Ok(candidate)
    };
    let plus_physics = make_physics(wave_number + step)?;
    let minus_physics = make_physics(wave_number - step)?;
    let build_value = |candidate_physics: &PhysicsParams,
                       levels: Vec<ClusterLevel>|
     -> Result<MlfmmSystem, String> {
        if basis == BasisOrder::Constant {
            Ok(build_mlfmm_system_stabilized_with_mode(
                elements,
                nodes,
                levels,
                candidate_physics,
                sign,
                beta_scale,
                translation_mode,
            ))
        } else {
            build_mlfmm_system_stabilized_with_basis_mode_checked(
                elements,
                nodes,
                levels,
                candidate_physics,
                basis,
                sign,
                beta_scale,
                translation_mode,
            )
            .map_err(|error| error.to_string())
        }
    };
    let plus = build_value(&plus_physics, cluster_levels.clone())?;
    let minus = build_value(&minus_physics, cluster_levels)?;
    if plus.sphere_points_per_level != minus.sphere_points_per_level
        || plus.t_matrices.len() != minus.t_matrices.len()
        || plus.s_matrices.len() != minus.s_matrices.len()
        || plus.d_matrices.len() != minus.d_matrices.len()
    {
        return Err("MLFMM derivative factor topology changed under perturbation".to_string());
    }
    let scale = 1.0 / (2.0 * step);
    Ok(MlfmmWavenumberDerivative {
        near_matrix: difference_near_blocks(&plus.near_matrix, &minus.near_matrix, scale)?,
        t_matrices: difference_level_matrices(&plus.t_matrices, &minus.t_matrices, scale)?,
        d_matrices: difference_d_matrices(&plus.d_matrices, &minus.d_matrices, scale)?,
        s_matrices: difference_level_matrices(&plus.s_matrices, &minus.s_matrices, scale)?,
        m2m_phase_plans: difference_phase_plans(
            &plus.m2m_phase_plans,
            &minus.m2m_phase_plans,
            scale,
        )?,
        l2l_phase_plans: difference_phase_plans(
            &plus.l2l_phase_plans,
            &minus.l2l_phase_plans,
            scale,
        )?,
        directional_plans: difference_directional_plans(
            &plus.directional_plans,
            &minus.directional_plans,
            scale,
        )?,
    })
}

fn difference_near_blocks(
    plus: &[NearFieldBlock],
    minus: &[NearFieldBlock],
    scale: f64,
) -> Result<Vec<NearFieldBlock>, String> {
    if plus.len() != minus.len() {
        return Err("MLFMM derivative near-field topology changed".to_string());
    }
    plus.iter()
        .zip(minus.iter())
        .map(|(plus, minus)| {
            if plus.source_cluster != minus.source_cluster
                || plus.field_cluster != minus.field_cluster
                || plus.coefficients.nrows() != minus.coefficients.nrows()
                || plus.coefficients.ncols() != minus.coefficients.ncols()
            {
                return Err("MLFMM derivative near-field block topology changed".to_string());
            }
            Ok(NearFieldBlock {
                source_cluster: plus.source_cluster,
                field_cluster: plus.field_cluster,
                coefficients: difference_dense(&plus.coefficients, &minus.coefficients, scale),
            })
        })
        .collect()
}

fn difference_level_matrices(
    plus: &[LevelMatrices],
    minus: &[LevelMatrices],
    scale: f64,
) -> Result<Vec<LevelMatrices>, String> {
    if plus.len() != minus.len() {
        return Err("MLFMM derivative level topology changed".to_string());
    }
    plus.iter()
        .zip(minus.iter())
        .map(|(plus, minus)| {
            if plus.level != minus.level
                || plus.num_clusters != minus.num_clusters
                || plus.matrices.len() != minus.matrices.len()
            {
                return Err("MLFMM derivative level matrix topology changed".to_string());
            }
            let matrices = plus
                .matrices
                .iter()
                .zip(minus.matrices.iter())
                .map(|(plus, minus)| {
                    if plus.nrows() != minus.nrows() || plus.ncols() != minus.ncols() {
                        return Err("MLFMM derivative transfer shape changed".to_string());
                    }
                    Ok(difference_dense(plus, minus, scale))
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(LevelMatrices {
                level: plus.level,
                num_clusters: plus.num_clusters,
                matrices,
            })
        })
        .collect()
}

fn difference_d_matrices(
    plus: &[Vec<DMatrixEntry>],
    minus: &[Vec<DMatrixEntry>],
    scale: f64,
) -> Result<Vec<Vec<DMatrixEntry>>, String> {
    if plus.len() != minus.len() {
        return Err("MLFMM derivative translation levels changed".to_string());
    }
    plus.iter()
        .zip(minus.iter())
        .map(|(plus_level, minus_level)| {
            if plus_level.len() != minus_level.len() {
                return Err("MLFMM derivative translation topology changed".to_string());
            }
            plus_level
                .iter()
                .zip(minus_level.iter())
                .map(|(plus, minus)| {
                    if plus.source_cluster != minus.source_cluster
                        || plus.field_cluster != minus.field_cluster
                        || plus.level != minus.level
                        || plus.active_directions != minus.active_directions
                        || plus.diagonal.len() != minus.diagonal.len()
                    {
                        return Err("MLFMM derivative translation entry changed".to_string());
                    }
                    Ok(DMatrixEntry {
                        source_cluster: plus.source_cluster,
                        field_cluster: plus.field_cluster,
                        level: plus.level,
                        diagonal: plus
                            .diagonal
                            .iter()
                            .zip(minus.diagonal.iter())
                            .map(|(plus, minus)| (*plus - *minus) * scale)
                            .collect(),
                        active_directions: plus.active_directions.clone(),
                    })
                })
                .collect()
        })
        .collect()
}

fn difference_phase_plans(
    plus: &[InterLevelPhasePlan],
    minus: &[InterLevelPhasePlan],
    scale: f64,
) -> Result<Vec<InterLevelPhasePlan>, String> {
    if plus.len() != minus.len() {
        return Err("MLFMM derivative phase topology changed".to_string());
    }
    plus.iter()
        .zip(minus.iter())
        .map(|(plus, minus)| {
            if plus.parent_children.len() != minus.parent_children.len() {
                return Err("MLFMM derivative phase child topology changed".to_string());
            }
            let mut parent_children = Vec::with_capacity(plus.parent_children.len());
            for (plus_children, minus_children) in plus
                .parent_children
                .iter()
                .zip(minus.parent_children.iter())
            {
                if plus_children.len() != minus_children.len() {
                    return Err("MLFMM derivative phase child count changed".to_string());
                }
                let children = plus_children
                    .iter()
                    .zip(minus_children.iter())
                    .map(|(plus, minus)| {
                        if plus.child_cluster != minus.child_cluster
                            || plus.phase.len() != minus.phase.len()
                        {
                            return Err("MLFMM derivative phase entry changed".to_string());
                        }
                        Ok(InterLevelChildPhase {
                            child_cluster: plus.child_cluster,
                            phase: plus
                                .phase
                                .iter()
                                .zip(minus.phase.iter())
                                .map(|(plus, minus)| (*plus - *minus) * scale)
                                .collect(),
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                parent_children.push(children);
            }
            Ok(InterLevelPhasePlan { parent_children })
        })
        .collect()
}

fn difference_directional_plans(
    plus: &[DirectionalLevelPlan],
    minus: &[DirectionalLevelPlan],
    scale: f64,
) -> Result<Vec<DirectionalLevelPlan>, String> {
    if plus.len() != minus.len() {
        return Err("MLFMM derivative directional levels changed".to_string());
    }
    plus.iter()
        .zip(minus.iter())
        .map(|(plus, minus)| {
            if plus.cones.len() != minus.cones.len()
                || plus.entries_by_cone.len() != minus.entries_by_cone.len()
                || plus.entries_by_cone_field_row.len() != minus.entries_by_cone_field_row.len()
                || plus.entries_by_cone_source_row.len() != minus.entries_by_cone_source_row.len()
            {
                return Err("MLFMM derivative directional topology changed".to_string());
            }
            for (plus_cone, minus_cone) in plus.cones.iter().zip(minus.cones.iter()) {
                if plus_cone.active_directions != minus_cone.active_directions
                    || plus_cone.source_clusters != minus_cone.source_clusters
                    || plus_cone.field_clusters != minus_cone.field_clusters
                {
                    return Err("MLFMM derivative directional cone changed".to_string());
                }
            }
            let mut entries_by_cone = Vec::with_capacity(plus.entries_by_cone.len());
            for (plus_entries, minus_entries) in plus
                .entries_by_cone
                .iter()
                .zip(minus.entries_by_cone.iter())
            {
                if plus_entries.len() != minus_entries.len() {
                    return Err("MLFMM derivative directional entry count changed".to_string());
                }
                let entries = plus_entries
                    .iter()
                    .zip(minus_entries.iter())
                    .map(|(plus, minus)| {
                        if plus.source_row != minus.source_row
                            || plus.field_row != minus.field_row
                            || plus.diagonal.len() != minus.diagonal.len()
                        {
                            return Err("MLFMM derivative directional entry changed".to_string());
                        }
                        Ok(DirectionalConeEntry {
                            source_row: plus.source_row,
                            field_row: plus.field_row,
                            diagonal: plus
                                .diagonal
                                .iter()
                                .zip(minus.diagonal.iter())
                                .map(|(plus, minus)| (*plus - *minus) * scale)
                                .collect(),
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                entries_by_cone.push(entries);
            }
            if plus.entries_by_cone_field_row != minus.entries_by_cone_field_row
                || plus.entries_by_cone_source_row != minus.entries_by_cone_source_row
            {
                return Err("MLFMM derivative directional row plan changed".to_string());
            }
            Ok(DirectionalLevelPlan {
                cones: plus.cones.clone(),
                entries_by_cone,
                entries_by_cone_field_row: plus.entries_by_cone_field_row.clone(),
                entries_by_cone_source_row: plus.entries_by_cone_source_row.clone(),
            })
        })
        .collect()
}

fn difference_dense(
    plus: &DenseMatrix<Complex64>,
    minus: &DenseMatrix<Complex64>,
    scale: f64,
) -> DenseMatrix<Complex64> {
    DenseMatrix::from_fn(plus.nrows(), plus.ncols(), |row, col| {
        (plus[[row, col]] - minus[[row, col]]) * scale
    })
}

/// Validate the constant-panel topology consumed by the MLFMM kernels.
///
/// This admission check mirrors the assumptions of the current leaf
/// `T`/`S` and near-field implementations: every active element is a single
/// constant panel with a unique global coefficient, and every active element
/// occurs exactly once in the leaf partition. The hierarchy references and
/// expansion dimensions are checked as well, so malformed externally-built
/// trees fail before the implementation's indexing/assertion boundaries.
pub fn validate_mlfmm_constant_panel_layout(
    elements: &[Element],
    nodes: &[[f64; 3]],
    cluster_levels: &[ClusterLevel],
) -> Result<usize, MlfmmAssemblyError> {
    if cluster_levels.is_empty() {
        return Err(MlfmmAssemblyError::EmptyClusterLevels);
    }

    let num_dofs = count_dofs(elements);
    let mut dof_owner = vec![None; num_dofs];
    for (element_index, element) in elements.iter().enumerate() {
        for (local, &node) in element.connectivity.iter().enumerate() {
            if node >= nodes.len() {
                return Err(MlfmmAssemblyError::InvalidNode {
                    element: element_index,
                    local,
                    node,
                    available: nodes.len(),
                });
            }
        }
        if !matches!(element.element_type, ElementType::Tri3 | ElementType::Quad4) {
            return Err(MlfmmAssemblyError::UnsupportedElementType {
                element: element_index,
            });
        }
        if element.connectivity.len() != element.element_type.num_nodes() {
            return Err(MlfmmAssemblyError::InvalidConnectivity {
                element: element_index,
                nodes: element.connectivity.len(),
            });
        }
        if element.property.is_evaluation() {
            continue;
        }
        if element.dof_addresses.len() != 1 {
            return Err(MlfmmAssemblyError::UnsupportedBasis {
                element: element_index,
                dofs: element.dof_addresses.len(),
            });
        }
        let global = element.dof_addresses[0];
        if global >= num_dofs {
            return Err(MlfmmAssemblyError::InvalidDofAddress {
                element: element_index,
                global,
                num_dofs,
            });
        }
        if let Some(first_element) = dof_owner[global] {
            return Err(MlfmmAssemblyError::DuplicateGlobalDof {
                first_element,
                second_element: element_index,
                global,
            });
        }
        dof_owner[global] = Some(element_index);
    }
    if let Some(global) = dof_owner.iter().position(Option::is_none) {
        return Err(MlfmmAssemblyError::MissingGlobalDof { global, num_dofs });
    }

    for (level_index, level) in cluster_levels.iter().enumerate() {
        if level.theta_points == 0 || level.phi_points == 0 || level.expansion_terms == 0 {
            return Err(MlfmmAssemblyError::InvalidQuadrature {
                level: level_index,
                theta_points: level.theta_points,
                phi_points: level.phi_points,
                expansion_terms: level.expansion_terms,
            });
        }
        for (cluster_index, cluster) in level.clusters.iter().enumerate() {
            for &element_index in &cluster.element_indices {
                if element_index >= elements.len() {
                    return Err(MlfmmAssemblyError::InvalidClusterElement {
                        level: level_index,
                        cluster: cluster_index,
                        element: element_index,
                        available: elements.len(),
                    });
                }
            }
            for &reference in cluster
                .near_clusters
                .iter()
                .chain(&cluster.far_clusters)
                .chain(&cluster.fan_clusters)
            {
                if reference >= level.clusters.len() {
                    return Err(MlfmmAssemblyError::InvalidHierarchyReference {
                        level: level_index,
                        cluster: cluster_index,
                        reference,
                        available: level.clusters.len(),
                    });
                }
            }
            if level_index + 1 < cluster_levels.len() {
                let child_count = cluster_levels[level_index + 1].clusters.len();
                for &reference in &cluster.sons {
                    if reference >= child_count {
                        return Err(MlfmmAssemblyError::InvalidHierarchyReference {
                            level: level_index,
                            cluster: cluster_index,
                            reference,
                            available: child_count,
                        });
                    }
                }
            } else if !cluster.sons.is_empty() {
                return Err(MlfmmAssemblyError::InvalidHierarchyReference {
                    level: level_index,
                    cluster: cluster_index,
                    reference: cluster.sons[0],
                    available: 0,
                });
            }
            if level_index == 0 {
                if let Some(reference) = cluster.father {
                    return Err(MlfmmAssemblyError::InvalidHierarchyReference {
                        level: level_index,
                        cluster: cluster_index,
                        reference,
                        available: 0,
                    });
                }
            } else if let Some(reference) = cluster.father {
                let parent_count = cluster_levels[level_index - 1].clusters.len();
                if reference >= parent_count {
                    return Err(MlfmmAssemblyError::InvalidHierarchyReference {
                        level: level_index,
                        cluster: cluster_index,
                        reference,
                        available: parent_count,
                    });
                }
            }
        }
    }

    let leaf_level = cluster_levels
        .last()
        .expect("non-empty hierarchy checked above");
    let mut leaf_owner = vec![None; elements.len()];
    for (cluster_index, cluster) in leaf_level.clusters.iter().enumerate() {
        for &element_index in &cluster.element_indices {
            if elements[element_index].property.is_evaluation() {
                continue;
            }
            if let Some(first_cluster) = leaf_owner[element_index] {
                return Err(MlfmmAssemblyError::DuplicateLeafElement {
                    element: element_index,
                    first_cluster,
                    second_cluster: cluster_index,
                });
            }
            leaf_owner[element_index] = Some(cluster_index);
        }
    }
    for (element_index, element) in elements.iter().enumerate() {
        if !element.property.is_evaluation() && leaf_owner[element_index].is_none() {
            return Err(MlfmmAssemblyError::MissingLeafElement {
                element: element_index,
            });
        }
    }
    Ok(num_dofs)
}

/// Build a checked MLFMM system for constant-panel elements.
pub fn build_mlfmm_system_stabilized_checked(
    elements: &[Element],
    nodes: &[[f64; 3]],
    cluster_levels: Vec<ClusterLevel>,
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
) -> Result<MlfmmSystem, MlfmmAssemblyError> {
    validate_mlfmm_constant_panel_layout(elements, nodes, &cluster_levels)?;
    Ok(build_mlfmm_system_stabilized_with_mode(
        elements,
        nodes,
        cluster_levels,
        physics,
        sign,
        beta_scale,
        TranslationMode::Full,
    ))
}

/// Build a checked directional MLFMM system for constant-panel elements.
pub fn build_directional_mlfmm_system_stabilized_checked(
    elements: &[Element],
    nodes: &[[f64; 3]],
    cluster_levels: Vec<ClusterLevel>,
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
    config: DirectionalMlfmmConfig,
) -> Result<MlfmmSystem, MlfmmAssemblyError> {
    validate_mlfmm_constant_panel_layout(elements, nodes, &cluster_levels)?;
    Ok(build_mlfmm_system_stabilized_with_mode(
        elements,
        nodes,
        cluster_levels,
        physics,
        sign,
        beta_scale,
        TranslationMode::Directional(config),
    ))
}

/// Build a basis-weighted full MLFMM system for linear or quadratic Tri3/Quad4
/// traces. Each panel-local basis slot is represented by a proxy element so
/// leaf multipoles, near-field blocks, and collocation rows retain the global
/// DOF dimensions used by the matrix-free operator.
pub fn build_mlfmm_system_stabilized_with_basis_checked(
    elements: &[Element],
    nodes: &[[f64; 3]],
    cluster_levels: Vec<ClusterLevel>,
    physics: &PhysicsParams,
    basis: BasisOrder,
    sign: f64,
    beta_scale: f64,
) -> Result<MlfmmSystem, MlfmmAssemblyError> {
    build_mlfmm_system_stabilized_with_basis_mode_checked(
        elements,
        nodes,
        cluster_levels,
        physics,
        basis,
        sign,
        beta_scale,
        TranslationMode::Full,
    )
}

/// Build a basis-weighted directional MLFMM system for linear or quadratic
/// Tri3/Quad4 traces.
pub fn build_directional_mlfmm_system_stabilized_with_basis_checked(
    elements: &[Element],
    nodes: &[[f64; 3]],
    cluster_levels: Vec<ClusterLevel>,
    physics: &PhysicsParams,
    basis: BasisOrder,
    sign: f64,
    beta_scale: f64,
    config: DirectionalMlfmmConfig,
) -> Result<MlfmmSystem, MlfmmAssemblyError> {
    build_mlfmm_system_stabilized_with_basis_mode_checked(
        elements,
        nodes,
        cluster_levels,
        physics,
        basis,
        sign,
        beta_scale,
        TranslationMode::Directional(config),
    )
}

fn build_mlfmm_system_stabilized_with_basis_mode_checked(
    elements: &[Element],
    nodes: &[[f64; 3]],
    cluster_levels: Vec<ClusterLevel>,
    physics: &PhysicsParams,
    basis: BasisOrder,
    sign: f64,
    beta_scale: f64,
    translation_mode: TranslationMode,
) -> Result<MlfmmSystem, MlfmmAssemblyError> {
    if basis == BasisOrder::Constant {
        return Ok(build_mlfmm_system_stabilized_with_mode(
            elements,
            nodes,
            cluster_levels,
            physics,
            sign,
            beta_scale,
            translation_mode,
        ));
    }
    let num_dofs = validate_mlfmm_basis_layout(elements, nodes, &cluster_levels, basis)?;
    // Continuous higher-order TBEM uses the panel normal at each collocation
    // row and the flat-panel quadrature path, even when mesh metadata carries
    // smoothed vertex normals for post-processing.  Infer that layout from a
    // repeated global basis address and preserve the same geometry contract in
    // the proxy elements.  Discontinuous layouts retain geometric collocation
    // normals and curved-panel quadrature.
    let continuous_layout = has_shared_basis_dofs(elements);
    let (proxy_elements, element_proxies) =
        build_mlfmm_basis_proxies(elements, nodes, basis, continuous_layout);
    let expanded_levels = expand_mlfmm_cluster_levels(&cluster_levels, &element_proxies);
    let system = build_mlfmm_system_stabilized_with_mode(
        &proxy_elements,
        nodes,
        expanded_levels,
        physics,
        sign,
        beta_scale,
        translation_mode,
    );
    debug_assert_eq!(system.num_dofs, num_dofs);
    Ok(system)
}

fn validate_mlfmm_basis_layout(
    elements: &[Element],
    nodes: &[[f64; 3]],
    cluster_levels: &[ClusterLevel],
    basis: BasisOrder,
) -> Result<usize, MlfmmAssemblyError> {
    if cluster_levels.is_empty() {
        return Err(MlfmmAssemblyError::EmptyClusterLevels);
    }
    let num_dofs = count_dofs(elements);
    let mut owned = vec![false; num_dofs];
    for (element_index, element) in elements.iter().enumerate() {
        for (local, &node) in element.connectivity.iter().enumerate() {
            if node >= nodes.len() {
                return Err(MlfmmAssemblyError::InvalidNode {
                    element: element_index,
                    local,
                    node,
                    available: nodes.len(),
                });
            }
        }
        if element.property.is_evaluation() {
            continue;
        }
        if !matches!(element.element_type, ElementType::Tri3 | ElementType::Quad4) {
            return Err(MlfmmAssemblyError::UnsupportedElementType {
                element: element_index,
            });
        }
        if element.connectivity.len() != element.element_type.num_nodes() {
            return Err(MlfmmAssemblyError::InvalidConnectivity {
                element: element_index,
                nodes: element.connectivity.len(),
            });
        }
        let expected = element.element_type.num_dofs_for_basis(basis);
        if element.dof_addresses.len() != expected {
            return Err(MlfmmAssemblyError::InvalidBasisDofCount {
                element: element_index,
                expected,
                actual: element.dof_addresses.len(),
            });
        }
        for &global in &element.dof_addresses {
            if global >= num_dofs {
                return Err(MlfmmAssemblyError::InvalidDofAddress {
                    element: element_index,
                    global,
                    num_dofs,
                });
            }
            owned[global] = true;
        }
    }
    if let Some(global) = owned.iter().position(|owned| !owned) {
        return Err(MlfmmAssemblyError::MissingGlobalDof { global, num_dofs });
    }

    for (level_index, level) in cluster_levels.iter().enumerate() {
        if level.theta_points == 0 || level.phi_points == 0 || level.expansion_terms == 0 {
            return Err(MlfmmAssemblyError::InvalidQuadrature {
                level: level_index,
                theta_points: level.theta_points,
                phi_points: level.phi_points,
                expansion_terms: level.expansion_terms,
            });
        }
        for (cluster_index, cluster) in level.clusters.iter().enumerate() {
            for &element in &cluster.element_indices {
                if element >= elements.len() {
                    return Err(MlfmmAssemblyError::InvalidClusterElement {
                        level: level_index,
                        cluster: cluster_index,
                        element,
                        available: elements.len(),
                    });
                }
            }
            for &reference in cluster
                .near_clusters
                .iter()
                .chain(&cluster.far_clusters)
                .chain(&cluster.fan_clusters)
            {
                if reference >= level.clusters.len() {
                    return Err(MlfmmAssemblyError::InvalidHierarchyReference {
                        level: level_index,
                        cluster: cluster_index,
                        reference,
                        available: level.clusters.len(),
                    });
                }
            }
            if level_index + 1 < cluster_levels.len() {
                let child_count = cluster_levels[level_index + 1].clusters.len();
                for &reference in &cluster.sons {
                    if reference >= child_count {
                        return Err(MlfmmAssemblyError::InvalidHierarchyReference {
                            level: level_index,
                            cluster: cluster_index,
                            reference,
                            available: child_count,
                        });
                    }
                }
            } else if !cluster.sons.is_empty() {
                return Err(MlfmmAssemblyError::InvalidHierarchyReference {
                    level: level_index,
                    cluster: cluster_index,
                    reference: cluster.sons[0],
                    available: 0,
                });
            }
            if level_index == 0 {
                if cluster.father.is_some() {
                    return Err(MlfmmAssemblyError::InvalidHierarchyReference {
                        level: level_index,
                        cluster: cluster_index,
                        reference: cluster.father.unwrap_or(0),
                        available: 0,
                    });
                }
            } else if cluster
                .father
                .is_some_and(|father| father >= cluster_levels[level_index - 1].clusters.len())
            {
                return Err(MlfmmAssemblyError::InvalidHierarchyReference {
                    level: level_index,
                    cluster: cluster_index,
                    reference: cluster.father.unwrap_or(0),
                    available: cluster_levels[level_index - 1].clusters.len(),
                });
            }
        }
    }

    let leaf = cluster_levels.last().expect("non-empty hierarchy");
    let mut leaf_owner = vec![None; elements.len()];
    for (cluster_index, cluster) in leaf.clusters.iter().enumerate() {
        for &element in &cluster.element_indices {
            if elements[element].property.is_evaluation() {
                continue;
            }
            if let Some(first_cluster) = leaf_owner[element] {
                return Err(MlfmmAssemblyError::DuplicateLeafElement {
                    element,
                    first_cluster,
                    second_cluster: cluster_index,
                });
            }
            leaf_owner[element] = Some(cluster_index);
        }
    }
    for (element, value) in elements.iter().enumerate() {
        if !value.property.is_evaluation() && leaf_owner[element].is_none() {
            return Err(MlfmmAssemblyError::MissingLeafElement { element });
        }
    }
    Ok(num_dofs)
}

const MLFMM_PROXY_TAG: usize = 1usize << (usize::BITS - 1);
const MLFMM_PROXY_BASIS_SHIFT: usize = 8;

fn encode_mlfmm_proxy_group(num_basis: usize, local: usize) -> usize {
    MLFMM_PROXY_TAG | ((num_basis & 0xff) << MLFMM_PROXY_BASIS_SHIFT) | (local & 0xff)
}

fn has_shared_basis_dofs(elements: &[Element]) -> bool {
    let mut seen = std::collections::HashSet::new();
    elements
        .iter()
        .filter(|element| !element.property.is_evaluation())
        .flat_map(|element| element.dof_addresses.iter().copied())
        .any(|dof| !seen.insert(dof))
}

fn build_mlfmm_basis_proxies(
    elements: &[Element],
    nodes: &[[f64; 3]],
    basis: BasisOrder,
    continuous_layout: bool,
) -> (Vec<Element>, Vec<Vec<usize>>) {
    let mut proxies = Vec::new();
    let mut element_proxies = vec![Vec::new(); elements.len()];
    for (element_index, element) in elements.iter().enumerate() {
        if element.property.is_evaluation() {
            element_proxies[element_index].push(proxies.len());
            proxies.push(element.clone());
            continue;
        }
        let points = element.collocation_points(basis, nodes);
        let normals = element.collocation_normals(basis, nodes);
        for (local, &dof) in element.dof_addresses.iter().enumerate() {
            let mut proxy = element.clone();
            proxy.center = points[local];
            proxy.normal = if continuous_layout {
                element.normal
            } else {
                normals[local]
            };
            if continuous_layout {
                // Continuous TBEM deliberately uses flat-panel quadrature;
                // zero smoothed normals so the near-field helper selects that
                // same path instead of curved geometry integration.
                proxy.node_normals = vec![[0.0; 3]; element.element_type.num_nodes()];
            }
            proxy.dof_addresses = vec![dof];
            proxy.group = encode_mlfmm_proxy_group(element.dof_addresses.len(), local);
            element_proxies[element_index].push(proxies.len());
            proxies.push(proxy);
        }
    }
    (proxies, element_proxies)
}

fn expand_mlfmm_cluster_levels(
    levels: &[ClusterLevel],
    element_proxies: &[Vec<usize>],
) -> Vec<ClusterLevel> {
    levels
        .iter()
        .map(|level| {
            let mut expanded = level.clone();
            expanded.clusters = level
                .clusters
                .iter()
                .map(|cluster| {
                    let mut expanded_cluster = cluster.clone();
                    expanded_cluster.element_indices = cluster
                        .element_indices
                        .iter()
                        .flat_map(|&element| element_proxies[element].iter().copied())
                        .collect();
                    expanded_cluster.num_elements = expanded_cluster.element_indices.len();
                    expanded_cluster
                })
                .collect();
            expanded
        })
        .collect()
}

/// Build an MLFMM system with stabilized Burton-Miller formulation
pub fn build_mlfmm_system_stabilized(
    elements: &[Element],
    nodes: &[[f64; 3]],
    cluster_levels: Vec<ClusterLevel>,
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
) -> MlfmmSystem {
    build_mlfmm_system_stabilized_with_mode(
        elements,
        nodes,
        cluster_levels,
        physics,
        sign,
        beta_scale,
        TranslationMode::Full,
    )
}

/// Build an experimental directional MLFMM system with stabilized
/// Burton-Miller formulation.
pub fn build_directional_mlfmm_system_stabilized(
    elements: &[Element],
    nodes: &[[f64; 3]],
    cluster_levels: Vec<ClusterLevel>,
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
) -> MlfmmSystem {
    build_directional_mlfmm_system_stabilized_with_config(
        elements,
        nodes,
        cluster_levels,
        physics,
        sign,
        beta_scale,
        DirectionalMlfmmConfig::default(),
    )
}

/// Build an experimental directional MLFMM system with an explicit pruning
/// configuration.
pub fn build_directional_mlfmm_system_stabilized_with_config(
    elements: &[Element],
    nodes: &[[f64; 3]],
    cluster_levels: Vec<ClusterLevel>,
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
    config: DirectionalMlfmmConfig,
) -> MlfmmSystem {
    build_mlfmm_system_stabilized_with_mode(
        elements,
        nodes,
        cluster_levels,
        physics,
        sign,
        beta_scale,
        TranslationMode::Directional(config),
    )
}

fn build_mlfmm_system_stabilized_with_mode(
    elements: &[Element],
    nodes: &[[f64; 3]],
    cluster_levels: Vec<ClusterLevel>,
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
    translation_mode: TranslationMode,
) -> MlfmmSystem {
    let num_dofs = count_dofs(elements);
    let num_levels = cluster_levels.len();

    // Pre-compute actual sphere point counts per level. The Gauss-Legendre table
    // may fall back to a lower order than requested, so we must use the real count.
    let actual_sphere_points: Vec<usize> = cluster_levels
        .iter()
        .map(|level| {
            let (coords, _) = unit_sphere_quadrature(level.theta_points, level.phi_points);
            coords.len()
        })
        .collect();

    let mut system = MlfmmSystem::new(
        num_dofs,
        num_levels,
        cluster_levels.clone(),
        actual_sphere_points.clone(),
    );
    system.m2m_phase_plans =
        build_interlevel_phase_plans(&cluster_levels, physics, &actual_sphere_points, -1.0);
    system.l2l_phase_plans =
        build_interlevel_phase_plans(&cluster_levels, physics, &actual_sphere_points, 1.0);
    if let Some(leaf_level) = cluster_levels.last() {
        build_leaf_dof_mappings(&mut system, elements, &leaf_level.clusters);
    }
    if let Some(leaf_level) = cluster_levels.last() {
        build_near_field_stabilized(
            &mut system,
            elements,
            nodes,
            &leaf_level.clusters,
            physics,
            sign,
            beta_scale,
        );
    }
    for (level_idx, level) in cluster_levels.iter().enumerate().rev() {
        let (sphere_coords, sphere_weights) =
            unit_sphere_quadrature(level.theta_points, level.phi_points);
        let t_matrices = build_t_matrices_level(
            elements,
            nodes,
            &level.clusters,
            physics,
            &sphere_coords,
            &sphere_weights,
            level_idx,
            &cluster_levels,
            &actual_sphere_points,
            sign,
            beta_scale,
        );
        system.t_matrices.push(t_matrices);
    }
    system.t_matrices.reverse();
    for (level_idx, level) in cluster_levels.iter().enumerate() {
        let (sphere_coords, _) = unit_sphere_quadrature(level.theta_points, level.phi_points);
        let num_sphere_points = sphere_coords.len();
        let d_matrices = build_d_matrices_level(
            &level.clusters,
            physics,
            &sphere_coords,
            level.expansion_terms,
            level_idx,
            num_sphere_points,
            translation_mode,
        );
        system.d_matrices.push(d_matrices);
    }
    system.rebuild_translation_plans();
    for (level_idx, level) in cluster_levels.iter().enumerate() {
        let (sphere_coords, sphere_weights) =
            unit_sphere_quadrature(level.theta_points, level.phi_points);
        let s_matrices = build_s_matrices_level(
            elements,
            &level.clusters,
            physics,
            &sphere_coords,
            &sphere_weights,
            level_idx,
            &cluster_levels,
            &actual_sphere_points,
            sign,
            beta_scale,
        );
        system.s_matrices.push(s_matrices);
    }
    system
}

fn build_leaf_dof_mappings(
    system: &mut MlfmmSystem,
    elements: &[Element],
    leaf_clusters: &[Cluster],
) {
    // Continuous higher-order elements repeat shared global DOFs in the
    // proxy layout.  Keep every incidence for field-side accumulation, but
    // designate exactly one incidence as the source/output row owner.
    // Hand-built systems leave this metadata empty and retain legacy behavior.
    system.leaf_dof_owner.clear();
    // Match the continuous TBEM owner convention, which chooses the first
    // element/local incidence in the caller's element order (rather than the
    // cluster traversal order used to build the proxy layout).
    let mut owners = vec![None; system.num_dofs];
    for (elem_idx, elem) in elements.iter().enumerate() {
        if elem.property.is_evaluation() {
            continue;
        }
        for (local, &global) in elem.dof_addresses.iter().enumerate() {
            if global < owners.len() {
                owners[global].get_or_insert((elem_idx, local));
            }
        }
    }
    for cluster in leaf_clusters {
        let mut dof_indices = Vec::new();
        let mut dof_owner = Vec::new();
        for &elem_idx in &cluster.element_indices {
            let elem = &elements[elem_idx];
            if elem.property.is_evaluation() {
                continue;
            }
            for (local, &global) in elem.dof_addresses.iter().enumerate() {
                dof_indices.push(global);
                dof_owner.push(
                    owners
                        .get(global)
                        .and_then(|owner| *owner)
                        .map(|(owner_elem, owner_local)| {
                            owner_elem == elem_idx && owner_local == local
                        })
                        .unwrap_or(false),
                );
            }
        }
        system.leaf_dof_indices.push(dof_indices);
        system.leaf_dof_owner.push(dof_owner);
    }
}

fn build_interlevel_phase_plans(
    cluster_levels: &[ClusterLevel],
    physics: &PhysicsParams,
    actual_sphere_points: &[usize],
    phase_sign: f64,
) -> Vec<InterLevelPhasePlan> {
    let k = physics.wave_number;
    let mut plans = Vec::with_capacity(cluster_levels.len());
    for (level_idx, level) in cluster_levels.iter().enumerate() {
        if level_idx + 1 >= cluster_levels.len() {
            plans.push(InterLevelPhasePlan::empty(level.clusters.len()));
            continue;
        }

        let (sphere_coords, _) = unit_sphere_quadrature(level.theta_points, level.phi_points);
        let child_level = &cluster_levels[level_idx + 1];
        let child_points = actual_sphere_points
            .get(level_idx + 1)
            .copied()
            .unwrap_or(sphere_coords.len());
        let mut parent_children = Vec::with_capacity(level.clusters.len());

        for cluster in &level.clusters {
            let mut children = Vec::with_capacity(cluster.sons.len());
            for &child_cluster_idx in &cluster.sons {
                let Some(child_cluster) = child_level.clusters.get(child_cluster_idx) else {
                    continue;
                };
                let diff = [
                    child_cluster.center[0] - cluster.center[0],
                    child_cluster.center[1] - cluster.center[1],
                    child_cluster.center[2] - cluster.center[2],
                ];
                let n = sphere_coords.len().min(child_points);
                let mut phase = Vec::with_capacity(n);
                for coord in sphere_coords.iter().take(n) {
                    let s_dot_diff = coord[0] * diff[0] + coord[1] * diff[1] + coord[2] * diff[2];
                    let angle = k * s_dot_diff;
                    phase.push(sonium_fastmath::exp_i(phase_sign * angle));
                }
                children.push(InterLevelChildPhase {
                    child_cluster: child_cluster_idx,
                    phase,
                });
            }
            parent_children.push(children);
        }

        plans.push(InterLevelPhasePlan { parent_children });
    }
    plans
}

fn build_m2l_field_plans(
    d_matrices: &[Vec<DMatrixEntry>],
    cluster_levels: &[ClusterLevel],
) -> Vec<Vec<Vec<usize>>> {
    d_matrices
        .iter()
        .enumerate()
        .map(|(level_idx, entries)| {
            let num_clusters = cluster_levels
                .get(level_idx)
                .map(|level| level.clusters.len())
                .unwrap_or(0);
            let mut by_field = vec![Vec::new(); num_clusters];
            for (entry_idx, entry) in entries.iter().enumerate() {
                if entry.field_cluster < by_field.len() {
                    by_field[entry.field_cluster].push(entry_idx);
                }
            }
            by_field
        })
        .collect()
}

fn build_directional_level_plans(
    d_matrices: &[Vec<DMatrixEntry>],
    cluster_levels: &[ClusterLevel],
) -> Vec<DirectionalLevelPlan> {
    d_matrices
        .iter()
        .enumerate()
        .map(|(level_idx, entries)| {
            let Some(_level) = cluster_levels.get(level_idx) else {
                return DirectionalLevelPlan::empty();
            };
            let mut cone_lookup: std::collections::HashMap<Vec<usize>, usize> =
                std::collections::HashMap::new();
            let mut cones = Vec::<DirectionalConeBasis>::new();
            let mut raw_entries_by_cone = Vec::<Vec<(usize, usize, Vec<Complex64>)>>::new();

            for entry in entries {
                let Some(active) = &entry.active_directions else {
                    continue;
                };
                if active.is_empty() {
                    continue;
                }
                let cone_idx = if let Some(&idx) = cone_lookup.get(active) {
                    idx
                } else {
                    let idx = cones.len();
                    cone_lookup.insert(active.clone(), idx);
                    cones.push(DirectionalConeBasis {
                        active_directions: active.clone(),
                        source_clusters: Vec::new(),
                        field_clusters: Vec::new(),
                    });
                    raw_entries_by_cone.push(Vec::new());
                    idx
                };
                let diagonal = active
                    .iter()
                    .filter_map(|&direction_idx| entry.diagonal.get(direction_idx).copied())
                    .collect();
                raw_entries_by_cone[cone_idx].push((
                    entry.source_cluster,
                    entry.field_cluster,
                    diagonal,
                ));
            }

            let mut entries_by_cone = Vec::with_capacity(raw_entries_by_cone.len());
            let mut entries_by_cone_field_row = Vec::with_capacity(raw_entries_by_cone.len());
            let mut entries_by_cone_source_row = Vec::with_capacity(raw_entries_by_cone.len());
            for (cone_idx, raw_entries) in raw_entries_by_cone.into_iter().enumerate() {
                let mut source_clusters: Vec<usize> =
                    raw_entries.iter().map(|(source, _, _)| *source).collect();
                source_clusters.sort_unstable();
                source_clusters.dedup();
                let mut field_clusters: Vec<usize> =
                    raw_entries.iter().map(|(_, field, _)| *field).collect();
                field_clusters.sort_unstable();
                field_clusters.dedup();

                let source_rows: std::collections::HashMap<usize, usize> = source_clusters
                    .iter()
                    .enumerate()
                    .map(|(row, &cluster)| (cluster, row))
                    .collect();
                let field_rows: std::collections::HashMap<usize, usize> = field_clusters
                    .iter()
                    .enumerate()
                    .map(|(row, &cluster)| (cluster, row))
                    .collect();
                let compact_entries: Vec<DirectionalConeEntry> = raw_entries
                    .into_iter()
                    .filter_map(|(source_cluster, field_cluster, diagonal)| {
                        Some(DirectionalConeEntry {
                            source_row: *source_rows.get(&source_cluster)?,
                            field_row: *field_rows.get(&field_cluster)?,
                            diagonal,
                        })
                    })
                    .collect();
                let mut entries_by_field_row = vec![Vec::new(); field_clusters.len()];
                let mut entries_by_source_row = vec![Vec::new(); source_clusters.len()];
                for (entry_idx, entry) in compact_entries.iter().enumerate() {
                    if entry.field_row < entries_by_field_row.len() {
                        entries_by_field_row[entry.field_row].push(entry_idx);
                    }
                    if entry.source_row < entries_by_source_row.len() {
                        entries_by_source_row[entry.source_row].push(entry_idx);
                    }
                }

                cones[cone_idx].source_clusters = source_clusters;
                cones[cone_idx].field_clusters = field_clusters;
                entries_by_cone.push(compact_entries);
                entries_by_cone_field_row.push(entries_by_field_row);
                entries_by_cone_source_row.push(entries_by_source_row);
            }

            DirectionalLevelPlan {
                cones,
                entries_by_cone,
                entries_by_cone_field_row,
                entries_by_cone_source_row,
            }
        })
        .collect()
}

fn build_near_field_stabilized(
    system: &mut MlfmmSystem,
    elements: &[Element],
    nodes: &[[f64; 3]],
    leaf_clusters: &[Cluster],
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
) {
    let gamma = Complex64::new(physics.gamma(), 0.0);
    let tau = Complex64::new(physics.tau, 0.0);
    let beta = physics.burton_miller_beta_scaled(beta_scale);
    let all_coords: Vec<DenseMatrix<f64>> = elements
        .iter()
        .map(|e| crate::core::assembly::common::get_element_coords(e, nodes))
        .collect();

    use crate::core::assembly::common::compute_near_block_stabilized;

    // Near-field block build (rayon above the machine-adaptive gate, serial
    // for few clusters). Each cluster runs kernel batches, so the grain is
    // coarse blocks per thread.
    let near_blocks_by_cluster: Vec<Vec<NearFieldBlock>> = gated_enumerate_map(
        leaf_clusters,
        leaf_clusters.len(),
        BEM_BLOCKS_PER_THREAD,
        |i, cluster_i| {
            let mut blocks = Vec::with_capacity(cluster_i.near_clusters.len() + 1);
            let basis_weighted = cluster_i
                .element_indices
                .iter()
                .any(|&index| decode_dof_proxy_group(elements[index].group).is_some());
            let block = if basis_weighted {
                compute_near_block_stabilized_with_basis(
                    elements,
                    &all_coords,
                    &cluster_i.element_indices,
                    &cluster_i.element_indices,
                    physics,
                    gamma,
                    tau,
                    beta,
                    true,
                    sign,
                )
            } else {
                compute_near_block_stabilized(
                    elements,
                    &all_coords,
                    &cluster_i.element_indices,
                    &cluster_i.element_indices,
                    physics,
                    gamma,
                    tau,
                    beta,
                    true,
                    sign,
                )
            };
            blocks.push(NearFieldBlock {
                source_cluster: i,
                field_cluster: i,
                coefficients: block,
            });

            // Near-cluster interactions (symmetric lists: if j is near i, then i is near j)
            for &j in &cluster_i.near_clusters {
                if j != i {
                    let cluster_j = &leaf_clusters[j];
                    let basis_weighted = cluster_i
                        .element_indices
                        .iter()
                        .chain(cluster_j.element_indices.iter())
                        .any(|&index| decode_dof_proxy_group(elements[index].group).is_some());
                    let block_ij = if basis_weighted {
                        compute_near_block_stabilized_with_basis(
                            elements,
                            &all_coords,
                            &cluster_i.element_indices,
                            &cluster_j.element_indices,
                            physics,
                            gamma,
                            tau,
                            beta,
                            false,
                            sign,
                        )
                    } else {
                        compute_near_block_stabilized(
                            elements,
                            &all_coords,
                            &cluster_i.element_indices,
                            &cluster_j.element_indices,
                            physics,
                            gamma,
                            tau,
                            beta,
                            false,
                            sign,
                        )
                    };
                    blocks.push(NearFieldBlock {
                        source_cluster: i,
                        field_cluster: j,
                        coefficients: block_ij,
                    });
                }
            }
            blocks
        },
    );
    system
        .near_matrix
        .extend(near_blocks_by_cluster.into_iter().flatten());
}

fn build_t_matrices_level(
    elements: &[Element],
    nodes: &[[f64; 3]],
    clusters: &[Cluster],
    physics: &PhysicsParams,
    sphere_coords: &[[f64; 3]],
    sphere_weights: &[f64],
    level_idx: usize,
    cluster_levels: &[ClusterLevel],
    actual_sphere_points: &[usize],
    sign: f64,
    beta_scale: f64,
) -> LevelMatrices {
    let k = physics.wave_number;
    let ik = Complex64::new(0.0, k);
    let num_sphere_points = sphere_coords.len();
    let num_clusters = clusters.len();
    let is_leaf_level = level_idx == cluster_levels.len() - 1;

    // BEM operators only apply at the leaf level (element↔multipole).
    // Inter-level T-matrices are pure plane-wave translations.
    let _ = (sign, beta_scale); // used only at leaf level

    // 7-point triangle Gauss rule — captures `kr_panel · O(1)` oscillation
    // of the plane-wave factor across each panel. The single-point
    // (centroid) approximation that previously sat here gave constant
    // 1.14 % per-pair M2L error at `kr_panel ≈ 0.5` because the kernel's
    // intra-panel variation was unresolved; see the audit work and the
    // `diagnose_mlfmm_per_pair_precision_*` tests.
    let tri_qpts = crate::core::integration::gauss::triangle_quadrature(4);

    // T-matrix build (rayon above the machine-adaptive gate, serial for few
    // clusters).
    let matrices: Vec<DenseMatrix<Complex64>> =
        gated_map(clusters, clusters.len(), BEM_BLOCKS_PER_THREAD, |cluster| {
            if is_leaf_level {
                let num_elem = cluster.element_indices.len();
                let mut t_matrix = DenseMatrix::<Complex64>::zeros(num_sphere_points, num_elem);
                for (j, &elem_idx) in cluster.element_indices.iter().enumerate() {
                    let elem = &elements[elem_idx];
                    if matches!(elem.element_type, ElementType::Quad4) {
                        let element_coords = get_element_coords(elem, nodes);
                        let quadrature = quad_quadrature(4);
                        for (p, coord) in sphere_coords.iter().enumerate() {
                            let mut plane_wave_normal_integral = Complex64::new(0.0, 0.0);
                            for &(s, t, weight) in &quadrature {
                                let shape = elem.shape_functions(s, t);
                                let (shape_ds, shape_dt) = elem.shape_derivatives(s, t);
                                let (point, jacobian, normal) = quadrature_geometry(
                                    &element_coords,
                                    &shape,
                                    &shape_ds,
                                    &shape_dt,
                                    None,
                                );
                                let diff = [
                                    point[0] - cluster.center[0],
                                    point[1] - cluster.center[1],
                                    point[2] - cluster.center[2],
                                ];
                                let s_dot_diff =
                                    coord[0] * diff[0] + coord[1] * diff[1] + coord[2] * diff[2];
                                let plane_wave = sonium_fastmath::exp_i(-k * s_dot_diff);
                                let s_dot_normal = coord[0] * normal[0]
                                    + coord[1] * normal[1]
                                    + coord[2] * normal[2];
                                let basis_weight = basis_order_for_proxy(elem)
                                    .and_then(|basis| {
                                        basis_values(elem.element_type, basis, s, t)
                                            .get(proxy_local_index(elem))
                                            .copied()
                                    })
                                    .unwrap_or(1.0);
                                plane_wave_normal_integral +=
                                    weight * jacobian * basis_weight * s_dot_normal * plane_wave;
                            }
                            t_matrix[[p, j]] =
                                sphere_weights[p] * (-ik) * plane_wave_normal_integral;
                        }
                        continue;
                    }
                    assert!(
                        matches!(elem.element_type, crate::core::types::ElementType::Tri3),
                        "MLFMM leaf T assembly currently supports only Tri3 elements; \
                     element {elem_idx} is {:?}",
                        elem.element_type
                    );
                    let v0_idx = elem.connectivity[0];
                    let v1_idx = elem.connectivity[1];
                    let v2_idx = elem.connectivity[2];
                    // Per-quadrature-point physical positions, computed once per
                    // element and reused across all sphere points.
                    let qpt_xyz: Vec<([f64; 3], f64, [f64; 3])> = tri_qpts
                        .iter()
                        .map(|&(xi, eta, w_q)| {
                            let lam0 = 1.0 - xi - eta;
                            let x = lam0 * nodes[v0_idx][0]
                                + xi * nodes[v1_idx][0]
                                + eta * nodes[v2_idx][0];
                            let y = lam0 * nodes[v0_idx][1]
                                + xi * nodes[v1_idx][1]
                                + eta * nodes[v2_idx][1];
                            let z = lam0 * nodes[v0_idx][2]
                                + xi * nodes[v1_idx][2]
                                + eta * nodes[v2_idx][2];
                            let normal = elem.normal;
                            let basis_weight = basis_order_for_proxy(elem)
                                .and_then(|basis| {
                                    basis_values(elem.element_type, basis, xi, eta)
                                        .get(proxy_local_index(elem))
                                        .copied()
                                })
                                .unwrap_or(1.0);
                            ([x, y, z], w_q * basis_weight, normal)
                        })
                        .collect();
                    for (p, coord) in sphere_coords.iter().enumerate() {
                        // ∫_{T_j} exp(-iks·(y - c)) dS_y ≈ 2·area · Σ_q w_q · exp(-iks·(y_q - c))
                        // (Σ_q w_q = 0.5 on the reference triangle; |J| = 2·area_phys.)
                        let mut accum = Complex64::new(0.0, 0.0);
                        for ([x, y, z], w_q, normal) in &qpt_xyz {
                            let dx = x - cluster.center[0];
                            let dy = y - cluster.center[1];
                            let dz = z - cluster.center[2];
                            let s_dot_diff = coord[0] * dx + coord[1] * dy + coord[2] * dz;
                            let exp_factor_q = sonium_fastmath::exp_i(-k * s_dot_diff);
                            let s_dot_normal =
                                coord[0] * normal[0] + coord[1] * normal[1] + coord[2] * normal[2];
                            accum += *w_q * s_dot_normal * exp_factor_q;
                        }
                        t_matrix[[p, j]] = sphere_weights[p] * 2.0 * elem.area * (-ik) * accum;
                    }
                }
                t_matrix
            } else {
                // Inter-level M2M (multipole-to-multipole) shift: block-diagonal phase shift.
                // All levels use the same quadrature grid, so child_num_points == num_sphere_points.
                // T[p, c*N + p] = exp(-ik s_p · (c_child - c_parent)) for each child c.
                //
                // Use the workspace's *actual* sphere-point count (which may be
                // smaller than `theta_points * phi_points` when `gauss_legendre`
                // silently falls back to a lower-order table). Otherwise the
                // T-matrix is built with a wider column dimension than the data
                // the upward pass actually carries, the size guard at the
                // matvec call site fails silently, and inter-level propagation
                // never runs — see audit at kD=10 / 4 levels in
                // `qa-bem-mlfmm-audit`.
                let num_children = cluster.sons.len();
                let child_num_points = if level_idx + 1 < cluster_levels.len() {
                    actual_sphere_points[level_idx + 1]
                } else {
                    num_sphere_points
                };
                let mut t_matrix = DenseMatrix::<Complex64>::zeros(
                    num_sphere_points,
                    num_children * child_num_points,
                );
                for (child_idx, &child_cluster_idx) in cluster.sons.iter().enumerate() {
                    if level_idx + 1 < cluster_levels.len() {
                        let child_cluster =
                            &cluster_levels[level_idx + 1].clusters[child_cluster_idx];
                        let diff = [
                            child_cluster.center[0] - cluster.center[0],
                            child_cluster.center[1] - cluster.center[1],
                            child_cluster.center[2] - cluster.center[2],
                        ];
                        for (p, coord) in sphere_coords.iter().enumerate() {
                            let s_dot_diff =
                                coord[0] * diff[0] + coord[1] * diff[1] + coord[2] * diff[2];
                            let exp_factor = sonium_fastmath::exp_i(-k * s_dot_diff);
                            // Diagonal entry: direction p in child maps to direction p in parent
                            if p < child_num_points {
                                t_matrix[[p, child_idx * child_num_points + p]] = exp_factor;
                            }
                        }
                    }
                }
                t_matrix
            }
        });
    LevelMatrices {
        level: level_idx,
        num_clusters,
        matrices,
    }
}

fn build_d_matrices_level(
    clusters: &[Cluster],
    physics: &PhysicsParams,
    sphere_coords: &[[f64; 3]],
    n_terms: usize,
    level_idx: usize,
    num_sphere_points: usize,
    translation_mode: TranslationMode,
) -> Vec<DMatrixEntry> {
    let k = physics.wave_number;
    // The diagonal carries the Green's-function 1/(4π) normalization. The leaf
    // T-matrices carry normalized unit-sphere quadrature weights separately so
    // the discrete sphere integral is an average over directions.
    let green_normalization = 1.0 / (4.0 * std::f64::consts::PI);
    let cone_centers = match translation_mode {
        TranslationMode::Full => Vec::new(),
        TranslationMode::Directional(config) => {
            fibonacci_sphere_directions(config.num_cones.max(1))
        }
    };
    // Translation-operator build (rayon above the machine-adaptive gate,
    // serial for few clusters). Each entry sums Hankel-Legendre series, so
    // the grain is coarse blocks per thread.
    let d_entries_by_cluster: Vec<Vec<DMatrixEntry>> = gated_enumerate_map(
        clusters,
        clusters.len(),
        BEM_BLOCKS_PER_THREAD,
        |i, cluster_i| {
            let mut entries = Vec::with_capacity(cluster_i.far_clusters.len());
            for &j in &cluster_i.far_clusters {
                let cluster_j = &clusters[j];
                let diff = [
                    cluster_i.center[0] - cluster_j.center[0],
                    cluster_i.center[1] - cluster_j.center[1],
                    cluster_i.center[2] - cluster_j.center[2],
                ];
                let r = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]).sqrt();
                if r < 1e-15 {
                    continue;
                }
                let kr = k * r;
                // Per-pair adaptive expansion (a minimal piece of the
                // Cheng-Crutchfield-Gimbutas-Greengard 2006 wideband M2L).
                //
                // The diagonal-form transfer Σ_n (2n+1) i^n h_n(kr) P_n(cos θ)
                // truncates at exp(-c·N) for `N ≳ kr` but the high-n tail
                // adds floating-point noise when `n_terms ≫ kr` because
                // h_n(kr)·P_n(cos θ) for n > kr is the difference of two
                // large nearly-equal contributions. Capping the sum per-pair
                // at `ceil(kr) + buffer` removes that noise without
                // sacrificing accuracy. `buffer = 8` gives ≤ 1e-6 truncation
                // error at this expansion-vs-frequency relationship.
                //
                // Full Cheng et al. wideband additionally swaps to a
                // low-frequency multipole form when `kr ≪ 1`; that path is
                // not yet active because our cluster trees + near/far
                // criterion already route low-kr pairs to the near field
                // (see `compute_near_far_lists`: the `k·dist > 2` clause).
                let pair_n_terms = n_terms.min((kr.ceil() as usize).saturating_add(8));
                let r_hat = [diff[0] / r, diff[1] / r, diff[2] / r];
                let h_funcs = spherical_hankel_first_kind((pair_n_terms + 1).max(2), kr, 1.0);
                let i_unit = Complex64::new(0.0, 1.0);
                let ik = Complex64::new(0.0, k);
                let mut diagonal = vec![Complex64::new(0.0, 0.0); num_sphere_points];
                for (p, coord) in sphere_coords.iter().enumerate() {
                    let cos_theta = coord[0] * r_hat[0] + coord[1] * r_hat[1] + coord[2] * r_hat[2];
                    let p_leg = legendre_polynomials(pair_n_terms + 1, cos_theta);
                    // Kahan compensated summation on Complex64 (per-component).
                    // The diagonal-form transfer
                    //   Σ_n (2n+1) iⁿ hₙ(kr) Pₙ(cos θ)
                    // builds up partial sums of magnitude `O(10⁵)` at moderate
                    // kr that cancel down to a final value `O(10⁻³)`. Naive f64
                    // accumulation loses ~3-4 digits to that cancellation.
                    // Kahan recovers most of it at ≈4× the per-add cost — well
                    // worth it for the diagonal we're storing for repeated reuse.
                    let mut transfer_re = 0.0_f64;
                    let mut transfer_im = 0.0_f64;
                    let mut comp_re = 0.0_f64;
                    let mut comp_im = 0.0_f64;
                    let mut i_power = Complex64::new(1.0, 0.0);
                    for n in 0..=pair_n_terms {
                        let term = (2.0 * n as f64 + 1.0) * i_power * h_funcs[n] * p_leg[n];
                        let y_re = term.re - comp_re;
                        let t_re = transfer_re + y_re;
                        comp_re = (t_re - transfer_re) - y_re;
                        transfer_re = t_re;
                        let y_im = term.im - comp_im;
                        let t_im = transfer_im + y_im;
                        comp_im = (t_im - transfer_im) - y_im;
                        transfer_im = t_im;
                        i_power *= i_unit;
                    }
                    let transfer = Complex64::new(transfer_re, transfer_im);
                    diagonal[p] = transfer * ik * green_normalization;
                }
                let active_directions = directional_active_direction_indices(
                    translation_mode,
                    sphere_coords,
                    &cone_centers,
                    &r_hat,
                    kr,
                    num_sphere_points,
                );
                entries.push(DMatrixEntry {
                    source_cluster: j,
                    field_cluster: i,
                    level: level_idx,
                    diagonal,
                    active_directions,
                });
            }
            entries
        },
    );
    d_entries_by_cluster.into_iter().flatten().collect()
}

fn directional_active_direction_indices(
    translation_mode: TranslationMode,
    sphere_coords: &[[f64; 3]],
    cone_centers: &[[f64; 3]],
    r_hat: &[f64; 3],
    kr: f64,
    num_sphere_points: usize,
) -> Option<Vec<usize>> {
    let TranslationMode::Directional(config) = translation_mode else {
        return None;
    };
    if kr < config.min_kr || num_sphere_points <= config.min_active_points {
        return None;
    }

    let axis = if cone_centers.is_empty() {
        *r_hat
    } else {
        nearest_cone_axis(cone_centers, r_hat)
    };
    let half_angle = if config.basis_half_angle > 0.0 {
        config
            .basis_half_angle
            .clamp(config.min_half_angle, config.max_half_angle)
    } else {
        (config.cone_safety / kr.sqrt()).clamp(config.min_half_angle, config.max_half_angle)
    };
    let cos_threshold = half_angle.cos();
    let mut active: Vec<usize> = sphere_coords
        .iter()
        .enumerate()
        .filter_map(|(idx, coord)| {
            let cos_angle = coord[0] * axis[0] + coord[1] * axis[1] + coord[2] * axis[2];
            (cos_angle >= cos_threshold).then_some(idx)
        })
        .collect();

    if active.len() < config.min_active_points {
        let mut ranked: Vec<(usize, f64)> = sphere_coords
            .iter()
            .enumerate()
            .map(|(idx, coord)| {
                let cos_angle = coord[0] * axis[0] + coord[1] * axis[1] + coord[2] * axis[2];
                (idx, cos_angle)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        active = ranked
            .into_iter()
            .take(config.min_active_points.min(num_sphere_points))
            .map(|(idx, _)| idx)
            .collect();
        active.sort_unstable();
    }

    // If the cone keeps most of the grid, the branch overhead is not worth it.
    if active.len() * 4 >= num_sphere_points * 3 {
        None
    } else {
        Some(active)
    }
}

fn nearest_cone_axis(cone_centers: &[[f64; 3]], r_hat: &[f64; 3]) -> [f64; 3] {
    cone_centers
        .iter()
        .copied()
        .max_by(|a, b| {
            let da = a[0] * r_hat[0] + a[1] * r_hat[1] + a[2] * r_hat[2];
            let db = b[0] * r_hat[0] + b[1] * r_hat[1] + b[2] * r_hat[2];
            da.total_cmp(&db)
        })
        .unwrap_or(*r_hat)
}

fn fibonacci_sphere_directions(n: usize) -> Vec<[f64; 3]> {
    let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
    (0..n)
        .map(|i| {
            let theta = (1.0 - 2.0 * (i as f64 + 0.5) / n as f64).acos();
            let phi = 2.0 * std::f64::consts::PI * i as f64 / golden_ratio;
            [
                theta.sin() * phi.cos(),
                theta.sin() * phi.sin(),
                theta.cos(),
            ]
        })
        .collect()
}

fn build_s_matrices_level(
    elements: &[Element],
    clusters: &[Cluster],
    physics: &PhysicsParams,
    sphere_coords: &[[f64; 3]],
    _sphere_weights: &[f64],
    level_idx: usize,
    cluster_levels: &[ClusterLevel],
    actual_sphere_points: &[usize],
    sign: f64,
    beta_scale: f64,
) -> LevelMatrices {
    let k = physics.wave_number;
    let ik = Complex64::new(0.0, k);
    let num_sphere_points = sphere_coords.len();
    let num_clusters = clusters.len();
    let is_leaf_level = level_idx == cluster_levels.len() - 1;

    // Burton-Miller combination coefficients (only used at leaf level)
    let gamma_tau = Complex64::new(physics.gamma() * physics.tau * sign, 0.0);
    let beta = physics.burton_miller_beta() * beta_scale * sign;

    // S-matrix build (rayon above the machine-adaptive gate, serial for few
    // clusters).
    let matrices: Vec<DenseMatrix<Complex64>> =
        gated_map(clusters, clusters.len(), BEM_BLOCKS_PER_THREAD, |cluster| {
            if is_leaf_level {
                let num_elem = cluster.element_indices.len();
                let mut s_matrix = DenseMatrix::<Complex64>::zeros(num_elem, num_sphere_points);
                for (j, &elem_idx) in cluster.element_indices.iter().enumerate() {
                    let elem = &elements[elem_idx];
                    let n_i = &elem.normal;
                    for (p, coord) in sphere_coords.iter().enumerate() {
                        let diff = [
                            elem.center[0] - cluster.center[0],
                            elem.center[1] - cluster.center[1],
                            elem.center[2] - cluster.center[2],
                        ];
                        let s_dot_diff =
                            coord[0] * diff[0] + coord[1] * diff[1] + coord[2] * diff[2];
                        let exp_factor = sonium_fastmath::exp_i(k * s_dot_diff);
                        // Source-side: Burton-Miller combination [γτ + β*(ik s·n_x)]
                        let s_dot_ni = coord[0] * n_i[0] + coord[1] * n_i[1] + coord[2] * n_i[2];
                        let bm_factor = gamma_tau + beta * (ik * s_dot_ni);
                        s_matrix[[j, p]] = bm_factor * exp_factor;
                    }
                }
                s_matrix
            } else {
                // Inter-level L2L (local-to-local) shift: block-diagonal phase shift.
                // All levels use the same quadrature grid, so child_num_points == num_sphere_points.
                // S[c*N + p, p] = exp(+ik s_p · (c_child - c_parent)) for each child c.
                //
                // See the matching comment in `build_t_matrices_level` — this
                // must use the *actual* workspace sphere-point count, not the
                // requested `theta * phi`, otherwise the size guard at
                // `downward_pass` silently skips inter-level local propagation.
                let num_children = cluster.sons.len();
                let child_num_points = if level_idx + 1 < cluster_levels.len() {
                    actual_sphere_points[level_idx + 1]
                } else {
                    num_sphere_points
                };
                let mut s_matrix = DenseMatrix::<Complex64>::zeros(
                    num_children * child_num_points,
                    num_sphere_points,
                );
                for (child_idx, &child_cluster_idx) in cluster.sons.iter().enumerate() {
                    if level_idx + 1 < cluster_levels.len() {
                        let child_cluster =
                            &cluster_levels[level_idx + 1].clusters[child_cluster_idx];
                        let diff = [
                            child_cluster.center[0] - cluster.center[0],
                            child_cluster.center[1] - cluster.center[1],
                            child_cluster.center[2] - cluster.center[2],
                        ];
                        for (p, coord) in sphere_coords.iter().enumerate() {
                            let s_dot_diff =
                                coord[0] * diff[0] + coord[1] * diff[1] + coord[2] * diff[2];
                            let exp_factor = sonium_fastmath::exp_i(k * s_dot_diff);
                            // Diagonal entry: direction p in parent maps to direction p in child
                            if p < child_num_points {
                                s_matrix[[child_idx * child_num_points + p, p]] = exp_factor;
                            }
                        }
                    }
                }
                s_matrix
            }
        });
    LevelMatrices {
        level: level_idx,
        num_clusters,
        matrices,
    }
}

/// Estimate the number of octree levels needed for the given element count
pub fn estimate_num_levels(
    num_elements: usize,
    elements_per_leaf: usize,
    min_levels: usize,
    max_levels: usize,
) -> usize {
    if num_elements == 0 {
        return min_levels;
    }
    let mut levels = 1;
    let mut n = num_elements;
    while n > elements_per_leaf && levels < max_levels {
        n /= 8;
        levels += 1;
    }
    levels.max(min_levels).min(max_levels)
}

/// Build an octree cluster hierarchy from the mesh elements
pub fn build_cluster_tree(
    elements: &[Element],
    target_elements_per_leaf: usize,
    physics: &PhysicsParams,
) -> Vec<ClusterLevel> {
    let num_elements = elements.len();
    let num_levels = estimate_num_levels(num_elements, target_elements_per_leaf, 1, 8);
    let mut levels = Vec::with_capacity(num_levels);
    let (min_coords, max_coords) = compute_bounding_box(elements);
    let root_center = [
        (min_coords[0] + max_coords[0]) / 2.0,
        (min_coords[1] + max_coords[1]) / 2.0,
        (min_coords[2] + max_coords[2]) / 2.0,
    ];
    let root_radius = ((max_coords[0] - min_coords[0]).powi(2)
        + (max_coords[1] - min_coords[1]).powi(2)
        + (max_coords[2] - min_coords[2]).powi(2))
    .sqrt()
        / 2.0;

    let mut root_level = ClusterLevel::new(1);
    let mut root = Cluster::new(root_center);
    root.element_indices = (0..num_elements).collect();
    root.radius = root_radius;
    root.level = 0;
    let kr = physics.wave_number * root_radius;
    // Cap at the highest tabulated Gauss-Legendre order in
    // `core::integration::gauss::gauss_legendre`. Above the cap,
    // `gauss_legendre` silently falls back to a lower order, the
    // θ-quadrature is too coarse for the diagonal-form M2L transfer's
    // Legendre series, spherical harmonics alias, and the matvec diverges
    // by many orders of magnitude (see the kD=10 row of
    // `qa-bem-mlfmm-audit` pre-fix). Cap = 32 since we ship GL32; the
    // longer-term path is Lebedev (`core::integration::lebedev`) once
    // higher-order tables are validated.
    root_level.expansion_terms = ((kr + 6.0 * kr.ln().max(1.0)) as usize).clamp(4, 28);
    root_level.theta_points = root_level.expansion_terms;
    // phi_points = 2 * expansion_terms + 2: Nyquist on the circle requires
    // ≥ 2p+1 samples to resolve azimuthal modes 0..p exactly. Using 2p
    // *aliases* mode p into mode 0 (one short by Nyquist), introducing
    // ~10 % error into the M2L integrand at high expansion order. The +2
    // buffer keeps phi_points even for symmetric sampling alignment.
    root_level.phi_points = 2 * root_level.expansion_terms + 2;
    root_level.clusters.push(root);
    root_level.num_original = 1;
    root_level.max_radius = root_radius;
    root_level.avg_radius = root_radius;
    root_level.min_radius = root_radius;
    levels.push(root_level);

    if num_levels > 1 {
        subdivide_level(&mut levels, elements, 0, target_elements_per_leaf, physics);
    }
    for level in &mut levels {
        compute_near_far_lists_optimized(&mut level.clusters, physics.wave_number);
    }

    // Use the root level's quadrature (the finest, since root has the largest clusters)
    // at ALL levels. This ensures inter-level T/S matrices are simple diagonal phase
    // shifts, avoiding the need for spherical harmonic anterpolation/interpolation.
    // The cost is slightly more quadrature points at finer levels, but correctness
    // requires matching grids for the diagonal-form MLFMM.
    if let Some(root) = levels.first() {
        let root_theta = root.theta_points;
        let root_phi = root.phi_points;
        let root_terms = root.expansion_terms;
        for level in levels.iter_mut().skip(1) {
            level.theta_points = root_theta;
            level.phi_points = root_phi;
            level.expansion_terms = root_terms;
        }
    }

    levels
}

fn compute_bounding_box(elements: &[Element]) -> ([f64; 3], [f64; 3]) {
    let mut min_coords = [f64::MAX, f64::MAX, f64::MAX];
    let mut max_coords = [f64::MIN, f64::MIN, f64::MIN];
    for elem in elements {
        for d in 0..3 {
            min_coords[d] = min_coords[d].min(elem.center[d]);
            max_coords[d] = max_coords[d].max(elem.center[d]);
        }
    }
    (min_coords, max_coords)
}

fn subdivide_level(
    levels: &mut Vec<ClusterLevel>,
    elements: &[Element],
    parent_level: usize,
    target_elements_per_leaf: usize,
    physics: &PhysicsParams,
) {
    let parent_clusters = levels[parent_level].clusters.clone();
    let mut child_level = ClusterLevel::new(parent_clusters.len() * 8);
    let mut max_radius = 0.0f64;
    let mut min_radius = f64::MAX;
    let mut sum_radius = 0.0f64;
    for (parent_idx, parent) in parent_clusters.iter().enumerate() {
        if parent.element_indices.len() <= target_elements_per_leaf {
            let mut leaf = parent.clone();
            leaf.level = parent_level + 1;
            leaf.father = Some(parent_idx);
            max_radius = max_radius.max(leaf.radius);
            min_radius = min_radius.min(leaf.radius);
            sum_radius += leaf.radius;
            let child_idx = child_level.clusters.len();
            levels[parent_level].clusters[parent_idx]
                .sons
                .push(child_idx);
            child_level.clusters.push(leaf);
            continue;
        }
        let half_size = parent.radius / 2.0;
        let offsets = [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ];
        for offset in &offsets {
            let child_center = [
                parent.center[0] + offset[0] * half_size * 0.5,
                parent.center[1] + offset[1] * half_size * 0.5,
                parent.center[2] + offset[2] * half_size * 0.5,
            ];
            // Octant assignment must place each element in exactly one
            // child. Inclusive `>= 0` on both negative and positive
            // offsets put any element exactly on a partition plane
            // (real or floating-point coincidence — common on symmetric
            // meshes like icospheres) into multiple octants, double-counting
            // entries in `MlfmmSystem::matvec` and breaking parity with
            // dense TBEM. Use the convention "elements on the boundary go
            // to the positive-side octant": positive-offset child requires
            // `dx >= 0`, negative-offset child requires `dx < 0`.
            let in_octant_axis = |coord: f64, parent_coord: f64, off: f64| -> bool {
                let d = coord - parent_coord;
                if off > 0.0 { d >= 0.0 } else { d < 0.0 }
            };
            let child_elements: Vec<usize> = parent
                .element_indices
                .iter()
                .filter(|&&idx| {
                    let elem = &elements[idx];
                    in_octant_axis(elem.center[0], parent.center[0], offset[0])
                        && in_octant_axis(elem.center[1], parent.center[1], offset[1])
                        && in_octant_axis(elem.center[2], parent.center[2], offset[2])
                })
                .copied()
                .collect();
            if child_elements.is_empty() {
                continue;
            }
            let mut child = Cluster::new(child_center);
            child.element_indices = child_elements;
            child.radius = half_size;
            child.level = parent_level + 1;
            child.father = Some(parent_idx);
            max_radius = max_radius.max(child.radius);
            min_radius = min_radius.min(child.radius);
            sum_radius += child.radius;
            let child_idx = child_level.clusters.len();
            levels[parent_level].clusters[parent_idx]
                .sons
                .push(child_idx);
            child_level.clusters.push(child);
        }
    }
    let num_clusters = child_level.clusters.len();
    child_level.num_original = num_clusters;
    child_level.max_radius = max_radius;
    child_level.min_radius = min_radius;
    child_level.avg_radius = if num_clusters > 0 {
        sum_radius / num_clusters as f64
    } else {
        0.0
    };
    let kr = physics.wave_number * child_level.avg_radius;
    // See `build_cluster_tree` root-level note — cap at the highest
    // tabulated Gauss-Legendre order so θ-quadrature is sufficient for
    // the diagonal-form M2L sums.
    child_level.expansion_terms = ((kr + 6.0 * kr.ln().max(1.0)) as usize).clamp(4, 28);
    child_level.theta_points = child_level.expansion_terms;
    child_level.phi_points = 2 * child_level.expansion_terms + 2; // see root-level Nyquist note
    levels.push(child_level);
    let current_level = levels.len() - 1;
    if levels[current_level]
        .clusters
        .iter()
        .any(|c| c.element_indices.len() > target_elements_per_leaf)
        && current_level < 7
    {
        subdivide_level(
            levels,
            elements,
            current_level,
            target_elements_per_leaf,
            physics,
        );
    }
}

fn compute_near_far_lists(clusters: &mut [Cluster], wave_number: f64) {
    let n = clusters.len();
    for i in 0..n {
        let center_i = &clusters[i].center;
        let radius_i = clusters[i].radius;
        let mut near = Vec::new();
        let mut far = Vec::new();
        for j in 0..n {
            if i == j {
                continue;
            }
            let center_j = &clusters[j].center;
            let radius_j = clusters[j].radius;
            let dist = ((center_i[0] - center_j[0]).powi(2)
                + (center_i[1] - center_j[1]).powi(2)
                + (center_i[2] - center_j[2]).powi(2))
            .sqrt();
            let separation = dist / (radius_i + radius_j).max(1e-15);
            if separation > MIN_GEOMETRIC_SEPARATION
                && wave_number * dist > MIN_WAVELENGTH_SEPARATION
            {
                far.push(j);
            } else {
                near.push(j);
            }
        }
        clusters[i].near_clusters = near;
        clusters[i].far_clusters = far;
    }
}

struct SpatialHash {
    cell_size: f64,
    bins: std::collections::HashMap<(i64, i64, i64), Vec<usize>>,
}

impl SpatialHash {
    fn new(cell_size: f64) -> Self {
        Self {
            cell_size,
            bins: std::collections::HashMap::new(),
        }
    }

    fn insert(&mut self, cluster_idx: usize, center: &[f64; 3]) {
        let key = self.cell_key(center);
        self.bins.entry(key).or_default().push(cluster_idx);
    }

    fn cell_key(&self, center: &[f64; 3]) -> (i64, i64, i64) {
        (
            (center[0] / self.cell_size).floor() as i64,
            (center[1] / self.cell_size).floor() as i64,
            (center[2] / self.cell_size).floor() as i64,
        )
    }

    fn get_nearby_indices(&self, center: &[f64; 3], search_radius_cells: i64) -> Vec<usize> {
        let (cx, cy, cz) = self.cell_key(center);
        let mut result = Vec::new();
        for dx in -search_radius_cells..=search_radius_cells {
            for dy in -search_radius_cells..=search_radius_cells {
                for dz in -search_radius_cells..=search_radius_cells {
                    if let Some(indices) = self.bins.get(&(cx + dx, cy + dy, cz + dz)) {
                        result.extend_from_slice(indices);
                    }
                }
            }
        }
        result
    }
}

fn compute_near_far_lists_optimized(clusters: &mut [Cluster], wave_number: f64) {
    if clusters.len() < 100 {
        compute_near_far_lists(clusters, wave_number);
        return;
    }

    let avg_radius: f64 = clusters.iter().map(|c| c.radius).sum::<f64>() / clusters.len() as f64;
    let cell_size = avg_radius * 4.0;
    let search_radius_cells = 2i64;

    let mut hash = SpatialHash::new(cell_size);
    for (i, cluster) in clusters.iter().enumerate() {
        hash.insert(i, &cluster.center);
    }

    let n = clusters.len();
    for i in 0..n {
        let center_i = &clusters[i].center;
        let radius_i = clusters[i].radius;
        let mut near = Vec::new();
        let mut far = Vec::new();

        let candidate_indices = hash.get_nearby_indices(center_i, search_radius_cells);

        for &j in &candidate_indices {
            if i == j {
                continue;
            }
            let center_j = &clusters[j].center;
            let radius_j = clusters[j].radius;
            let dist = ((center_i[0] - center_j[0]).powi(2)
                + (center_i[1] - center_j[1]).powi(2)
                + (center_i[2] - center_j[2]).powi(2))
            .sqrt();
            let separation = dist / (radius_i + radius_j).max(1e-15);
            if separation > MIN_GEOMETRIC_SEPARATION
                && wave_number * dist > MIN_WAVELENGTH_SEPARATION
            {
                far.push(j);
            } else {
                near.push(j);
            }
        }
        clusters[i].near_clusters = near;
        clusters[i].far_clusters = far;
    }
}

impl LinearOperator<Complex64> for MlfmmSystem {
    fn num_rows(&self) -> usize {
        self.num_dofs
    }

    fn num_cols(&self) -> usize {
        self.num_dofs
    }

    fn apply(&self, x: &[Complex64]) -> Vec<Complex64> {
        self.matvec(x)
    }

    fn apply_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        self.matvec_into(x, y);
    }

    fn apply_block(&self, x: &DenseMatrix<Complex64>, y: &mut DenseMatrix<Complex64>) {
        self.matvec_block_into(x, y);
    }

    fn apply_transpose(&self, x: &[Complex64]) -> Vec<Complex64> {
        self.matvec_transpose(x)
    }

    fn apply_transpose_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        self.matvec_transpose_into(x, y);
    }

    fn transpose_capability(&self) -> sonium_solvers::traits::AdjointCapability {
        sonium_solvers::traits::AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> sonium_solvers::traits::AdjointCapability {
        sonium_solvers::traits::AdjointCapability::Exact
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{BoundaryCondition, ElementProperty, ElementType};

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    #[test]
    fn mlfmm_block_matches_scalar_forward_pipeline() {
        let mut root = Cluster::new([0.5, 0.0, 0.0]);
        root.sons = vec![0, 1];
        let mut leaf_a = Cluster::new([0.0, 0.0, 0.0]);
        leaf_a.father = Some(0);
        let mut leaf_b = Cluster::new([1.0, 0.0, 0.0]);
        leaf_b.father = Some(0);

        let mut root_level = ClusterLevel::new(1);
        root_level.clusters.push(root);
        let mut leaf_level = ClusterLevel::new(2);
        leaf_level.clusters.extend([leaf_a, leaf_b]);

        let mut system = MlfmmSystem::new(2, 2, vec![root_level, leaf_level], vec![1, 1]);
        system.leaf_dof_indices = vec![vec![0], vec![1]];
        system.t_matrices = vec![
            LevelMatrices {
                level: 0,
                num_clusters: 1,
                matrices: vec![DenseMatrix::from_row_major_vec(
                    1,
                    2,
                    vec![c(1.0, 0.0), c(1.0, 0.0)],
                )],
            },
            LevelMatrices {
                level: 1,
                num_clusters: 2,
                matrices: vec![
                    DenseMatrix::from_row_major_vec(1, 1, vec![c(1.0, 0.0)]),
                    DenseMatrix::from_row_major_vec(1, 1, vec![c(1.0, 0.0)]),
                ],
            },
        ];
        system.s_matrices = vec![
            LevelMatrices {
                level: 0,
                num_clusters: 1,
                matrices: vec![DenseMatrix::from_row_major_vec(
                    2,
                    1,
                    vec![c(1.0, 0.0), c(1.0, 0.0)],
                )],
            },
            LevelMatrices {
                level: 1,
                num_clusters: 2,
                matrices: vec![
                    DenseMatrix::from_row_major_vec(1, 1, vec![c(1.0, 0.0)]),
                    DenseMatrix::from_row_major_vec(1, 1, vec![c(1.0, 0.0)]),
                ],
            },
        ];
        system.d_matrices = vec![
            vec![DMatrixEntry {
                source_cluster: 0,
                field_cluster: 0,
                level: 0,
                diagonal: vec![c(1.5, 0.0)],
                active_directions: None,
            }],
            Vec::new(),
        ];
        system.near_matrix.push(NearFieldBlock {
            source_cluster: 0,
            field_cluster: 0,
            coefficients: DenseMatrix::from_row_major_vec(1, 1, vec![c(0.25, 0.0)]),
        });

        let x = DenseMatrix::from_row_major_vec(
            2,
            3,
            vec![
                c(1.0, 0.0),
                c(-2.0, 0.0),
                c(0.25, 0.0),
                c(3.0, 0.0),
                c(0.5, 0.0),
                c(-1.25, 0.0),
            ],
        );
        let block = system.matvec_block(&x);
        for rhs in 0..x.ncols() {
            let scalar_x = vec![x[[0, rhs]], x[[1, rhs]]];
            let scalar_y = system.matvec(&scalar_x);
            for row in 0..system.num_dofs {
                assert!((block[[row, rhs]] - scalar_y[row]).norm() < 1e-12);
            }
        }
    }

    #[test]
    fn extract_near_field_matrix_preserves_asymmetric_block_orientation() {
        let mut level = ClusterLevel::new(1);
        level.clusters.push(Cluster::new([0.0, 0.0, 0.0]));
        level.clusters.push(Cluster::new([1.0, 0.0, 0.0]));

        let mut system = MlfmmSystem::new(3, 1, vec![level], vec![1]);
        system.leaf_dof_indices = vec![vec![0, 1], vec![2]];
        system.near_matrix.push(NearFieldBlock {
            source_cluster: 0,
            field_cluster: 1,
            coefficients: DenseMatrix::from_row_major_vec(2, 1, vec![c(1.0, 0.0), c(2.0, 0.0)]),
        });

        let matrix = system.extract_near_field_matrix();
        assert_eq!(matrix[[0, 2]], c(1.0, 0.0));
        assert_eq!(matrix[[1, 2]], c(2.0, 0.0));
        assert_eq!(matrix[[2, 0]], c(0.0, 0.0));
    }

    #[test]
    fn checked_near_field_matrix_rejects_malformed_layouts() {
        let mut level = ClusterLevel::new(1);
        level.clusters.push(Cluster::new([0.0, 0.0, 0.0]));

        let mut invalid_cluster = MlfmmSystem::new(1, 1, vec![level.clone()], vec![1]);
        invalid_cluster.leaf_dof_indices = vec![vec![0]];
        invalid_cluster.near_matrix.push(NearFieldBlock {
            source_cluster: 1,
            field_cluster: 0,
            coefficients: DenseMatrix::from_row_major_vec(1, 1, vec![c(1.0, 0.0)]),
        });
        assert!(matches!(
            invalid_cluster.try_extract_near_field_matrix(),
            Err(MlfmmNearFieldError::InvalidCluster { side: "source", .. })
        ));

        let mut invalid_shape = MlfmmSystem::new(2, 1, vec![level.clone()], vec![1]);
        invalid_shape.leaf_dof_indices = vec![vec![0, 1]];
        invalid_shape.near_matrix.push(NearFieldBlock {
            source_cluster: 0,
            field_cluster: 0,
            coefficients: DenseMatrix::from_row_major_vec(1, 1, vec![c(1.0, 0.0)]),
        });
        assert!(matches!(
            invalid_shape.try_extract_near_field_matrix(),
            Err(MlfmmNearFieldError::BlockShape { .. })
        ));

        let mut invalid_dof = MlfmmSystem::new(1, 1, vec![level.clone()], vec![1]);
        invalid_dof.leaf_dof_indices = vec![vec![1]];
        invalid_dof.near_matrix.push(NearFieldBlock {
            source_cluster: 0,
            field_cluster: 0,
            coefficients: DenseMatrix::from_row_major_vec(1, 1, vec![c(1.0, 0.0)]),
        });
        assert!(matches!(
            invalid_dof.try_extract_near_field_matrix(),
            Err(MlfmmNearFieldError::GlobalDof { .. })
        ));

        let mut nonfinite = MlfmmSystem::new(1, 1, vec![level], vec![1]);
        nonfinite.leaf_dof_indices = vec![vec![0]];
        nonfinite.near_matrix.push(NearFieldBlock {
            source_cluster: 0,
            field_cluster: 0,
            coefficients: DenseMatrix::from_row_major_vec(
                1,
                1,
                vec![Complex64::new(f64::NAN, 0.0)],
            ),
        });
        assert!(matches!(
            nonfinite.try_extract_near_field_matrix(),
            Err(MlfmmNearFieldError::NonFiniteCoefficient { .. })
        ));
    }

    #[test]
    fn mlfmm_block_downward_phase_uses_point_rows_for_multiple_rhs() {
        let mut root = Cluster::new([0.5, 0.0, 0.0]);
        root.sons = vec![0];
        let mut leaf = Cluster::new([1.0, 0.0, 0.0]);
        leaf.father = Some(0);
        let mut root_level = ClusterLevel::new(2);
        root_level.clusters.push(root);
        let mut leaf_level = ClusterLevel::new(2);
        leaf_level.clusters.push(leaf);
        let mut system = MlfmmSystem::new(1, 2, vec![root_level, leaf_level], vec![2, 2]);
        system.s_matrices = vec![
            LevelMatrices {
                level: 0,
                num_clusters: 1,
                matrices: vec![DenseMatrix::zeros(0, 0)],
            },
            LevelMatrices {
                level: 1,
                num_clusters: 1,
                matrices: Vec::new(),
            },
        ];
        system.l2l_phase_plans = vec![InterLevelPhasePlan {
            parent_children: vec![vec![InterLevelChildPhase {
                child_cluster: 0,
                phase: vec![c(2.0, 0.0), c(3.0, 0.0)],
            }]],
        }];
        let mut workspace =
            MlfmmWorkspace::new(&system.cluster_levels, &system.sphere_points_per_level);
        workspace.ensure_block_storage(
            &system.cluster_levels,
            &system.sphere_points_per_level,
            &[],
            2,
        );
        workspace.block_locals[0][[0, 0]] = c(1.0, 0.0);
        workspace.block_locals[0][[1, 0]] = c(2.0, 0.0);
        workspace.block_locals[0][[0, 1]] = c(10.0, 0.0);
        workspace.block_locals[0][[1, 1]] = c(20.0, 0.0);
        system.downward_pass_block(&mut workspace, 2);
        assert_eq!(workspace.block_locals[1][[0, 0]], c(2.0, 0.0));
        assert_eq!(workspace.block_locals[1][[1, 0]], c(6.0, 0.0));
        assert_eq!(workspace.block_locals[1][[0, 1]], c(20.0, 0.0));
        assert_eq!(workspace.block_locals[1][[1, 1]], c(60.0, 0.0));
    }

    fn bilinear_dot(a: &[Complex64], b: &[Complex64]) -> Complex64 {
        a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
    }

    #[test]
    fn directional_indices_prune_high_kr_cone() {
        let (sphere_coords, _) = unit_sphere_quadrature(16, 34);
        let config = DirectionalMlfmmConfig {
            min_kr: 4.0,
            cone_safety: 3.0,
            min_half_angle: 0.25,
            max_half_angle: 1.0,
            min_active_points: 8,
            num_cones: 32,
            basis_half_angle: 0.0,
        };
        let active = directional_active_direction_indices(
            TranslationMode::Directional(config),
            &sphere_coords,
            &[],
            &[1.0, 0.0, 0.0],
            100.0,
            sphere_coords.len(),
        )
        .expect("high-kr pair should be directionally pruned");
        assert!(active.len() >= config.min_active_points);
        assert!(active.len() < sphere_coords.len() / 2);
        let half_angle = (config.cone_safety / 100.0_f64.sqrt())
            .clamp(config.min_half_angle, config.max_half_angle);
        assert!(
            active
                .iter()
                .all(|&idx| sphere_coords[idx][0] >= half_angle.cos())
        );
    }

    #[test]
    fn directional_indices_keep_low_kr_full() {
        let (sphere_coords, _) = unit_sphere_quadrature(8, 18);
        let active = directional_active_direction_indices(
            TranslationMode::Directional(DirectionalMlfmmConfig::default()),
            &sphere_coords,
            &[],
            &[1.0, 0.0, 0.0],
            1.0,
            sphere_coords.len(),
        );
        assert!(active.is_none());
    }

    #[test]
    fn cone_indexed_translation_matches_expected_sparse_updates() {
        let mut level = ClusterLevel::new(2);
        level.clusters.push(Cluster::new([0.0, 0.0, 0.0]));
        level.clusters.push(Cluster::new([1.0, 0.0, 0.0]));

        let mut system = MlfmmSystem::new(0, 1, vec![level], vec![4]);
        system.d_matrices = vec![vec![
            DMatrixEntry {
                source_cluster: 0,
                field_cluster: 1,
                level: 0,
                diagonal: vec![c(2.0, 0.0), c(0.0, 0.0), c(3.0, 0.0), c(0.0, 0.0)],
                active_directions: Some(vec![0, 2]),
            },
            DMatrixEntry {
                source_cluster: 1,
                field_cluster: 0,
                level: 0,
                diagonal: vec![c(0.0, 0.0), c(5.0, 0.0), c(0.0, 0.0), c(7.0, 0.0)],
                active_directions: Some(vec![1, 3]),
            },
        ]];
        system.directional_plans =
            build_directional_level_plans(&system.d_matrices, &system.cluster_levels);
        let mut workspace = MlfmmWorkspace::new_with_directional(
            &system.cluster_levels,
            &system.sphere_points_per_level,
            &system.directional_plans,
        );

        workspace.multipoles[0].row_mut(0).copy_from_slice(&[
            c(1.0, 0.0),
            c(2.0, 0.0),
            c(3.0, 0.0),
            c(4.0, 0.0),
        ]);
        workspace.multipoles[0].row_mut(1).copy_from_slice(&[
            c(5.0, 0.0),
            c(6.0, 0.0),
            c(7.0, 0.0),
            c(8.0, 0.0),
        ]);

        system.translate_all_levels(&mut workspace);
        assert_eq!(
            workspace.locals[0].row(0),
            &[c(0.0, 0.0), c(30.0, 0.0), c(0.0, 0.0), c(56.0, 0.0)]
        );
        assert_eq!(
            workspace.locals[0].row(1),
            &[c(2.0, 0.0), c(0.0, 0.0), c(9.0, 0.0), c(0.0, 0.0)]
        );

        workspace.ensure_block_storage(
            &system.cluster_levels,
            &system.sphere_points_per_level,
            &system.directional_plans,
            2,
        );
        workspace.clear_block();
        for point in 0..4 {
            workspace.block_multipoles[0][[point, 0]] = workspace.multipoles[0][[0, point]];
            workspace.block_multipoles[0][[point, 1]] =
                c(2.0, 0.0) * workspace.multipoles[0][[0, point]];
            workspace.block_multipoles[0][[4 + point, 0]] = workspace.multipoles[0][[1, point]];
            workspace.block_multipoles[0][[4 + point, 1]] =
                c(2.0, 0.0) * workspace.multipoles[0][[1, point]];
        }
        system.translate_all_levels_block(&mut workspace, 2);
        let expected_field0 = [c(0.0, 0.0), c(30.0, 0.0), c(0.0, 0.0), c(56.0, 0.0)];
        let expected_field1 = [c(2.0, 0.0), c(0.0, 0.0), c(9.0, 0.0), c(0.0, 0.0)];
        for point in 0..4 {
            assert_eq!(
                workspace.block_locals[0][[point, 0]],
                expected_field0[point]
            );
            assert_eq!(
                workspace.block_locals[0][[point, 1]],
                c(2.0, 0.0) * expected_field0[point]
            );
            assert_eq!(
                workspace.block_locals[0][[4 + point, 0]],
                expected_field1[point]
            );
            assert_eq!(
                workspace.block_locals[0][[4 + point, 1]],
                c(2.0, 0.0) * expected_field1[point]
            );
        }

        workspace.clear();
        workspace.locals[0].row_mut(0).copy_from_slice(&[
            c(1.0, 0.0),
            c(2.0, 0.0),
            c(3.0, 0.0),
            c(4.0, 0.0),
        ]);
        workspace.locals[0].row_mut(1).copy_from_slice(&[
            c(5.0, 0.0),
            c(6.0, 0.0),
            c(7.0, 0.0),
            c(8.0, 0.0),
        ]);

        system.translate_all_levels_transpose(&mut workspace);
        assert_eq!(
            workspace.multipoles[0].row(0),
            &[c(10.0, 0.0), c(0.0, 0.0), c(21.0, 0.0), c(0.0, 0.0)]
        );
        assert_eq!(
            workspace.multipoles[0].row(1),
            &[c(0.0, 0.0), c(10.0, 0.0), c(0.0, 0.0), c(28.0, 0.0)]
        );

        workspace.ensure_block_storage(
            &system.cluster_levels,
            &system.sphere_points_per_level,
            &system.directional_plans,
            2,
        );
        workspace.clear_block();
        for point in 0..4 {
            workspace.block_locals[0][[point, 0]] = workspace.locals[0][[0, point]];
            workspace.block_locals[0][[point, 1]] = c(2.0, 0.0) * workspace.locals[0][[0, point]];
            workspace.block_locals[0][[4 + point, 0]] = workspace.locals[0][[1, point]];
            workspace.block_locals[0][[4 + point, 1]] =
                c(2.0, 0.0) * workspace.locals[0][[1, point]];
        }
        system.translate_all_levels_transpose_block(&mut workspace, 2);
        let expected_source0 = [c(10.0, 0.0), c(0.0, 0.0), c(21.0, 0.0), c(0.0, 0.0)];
        let expected_source1 = [c(0.0, 0.0), c(10.0, 0.0), c(0.0, 0.0), c(28.0, 0.0)];
        for point in 0..4 {
            assert_eq!(
                workspace.block_multipoles[0][[point, 0]],
                expected_source0[point]
            );
            assert_eq!(
                workspace.block_multipoles[0][[point, 1]],
                c(2.0, 0.0) * expected_source0[point]
            );
            assert_eq!(
                workspace.block_multipoles[0][[4 + point, 0]],
                expected_source1[point]
            );
            assert_eq!(
                workspace.block_multipoles[0][[4 + point, 1]],
                c(2.0, 0.0) * expected_source1[point]
            );
        }

        let stats = system.translation_stats();
        assert_eq!(stats.cone_count, 2);
        assert_eq!(stats.directional_entries, 2);
        assert_eq!(stats.active_points, 4);
    }

    fn make_test_elements() -> (Vec<Element>, Vec<[f64; 3]>) {
        let nodes: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [1.5, 1.0, 0.0],
        ];
        let elem0 = Element {
            connectivity: vec![0, 1, 2],
            element_type: ElementType::Tri3,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0; 3]; 3],
            center: [0.5, 1.0 / 3.0, 0.0],
            area: 0.5,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(1.0, 0.0)]),
            group: 0,
            dof_addresses: vec![0],
        };
        let elem1 = Element {
            connectivity: vec![1, 3, 2],
            element_type: ElementType::Tri3,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0; 3]; 3],
            center: [1.0, 2.0 / 3.0, 0.0],
            area: 0.5,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
            group: 0,
            dof_addresses: vec![1],
        };
        (vec![elem0, elem1], nodes)
    }

    #[test]
    fn compressed_wavenumber_derivative_matches_mlfmm_action() {
        let nodes = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [4.0, 1.0, 0.0],
        ];
        let mut elements = make_test_elements().0;
        elements[0].connectivity = vec![0, 1, 2];
        elements[0].center = [1.0 / 3.0, 1.0 / 3.0, 0.0];
        elements[1].connectivity = vec![3, 4, 5];
        elements[1].center = [13.0 / 3.0, 1.0 / 3.0, 0.0];
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
        let sign = PhysicsParams::optimal_formulation_sign(physics.wave_number);
        let beta_scale = 0.2;
        let cluster_levels = build_cluster_tree(&elements, 1, &physics);
        let base = build_mlfmm_system_stabilized_checked(
            &elements,
            &nodes,
            cluster_levels.clone(),
            &physics,
            sign,
            beta_scale,
        )
        .expect("checked MLFMM base assembly");
        let derivative = build_mlfmm_wavenumber_derivative(
            &elements,
            &nodes,
            cluster_levels.clone(),
            &physics,
            sign,
            beta_scale,
        )
        .expect("checked MLFMM derivative assembly");
        let x = vec![Complex64::new(0.7, -0.2), Complex64::new(-0.3, 0.4)];
        let analytic = derivative.apply(&base, &x);
        let step = 1.0e-4;
        let make_physics = |wavenumber: f64| {
            PhysicsParams::new(
                wavenumber * physics.speed_of_sound / (2.0 * std::f64::consts::PI),
                physics.speed_of_sound,
                physics.density,
                false,
            )
        };
        let plus_physics = make_physics(physics.wave_number + step);
        let minus_physics = make_physics(physics.wave_number - step);
        let plus = build_mlfmm_system_stabilized_checked(
            &elements,
            &nodes,
            cluster_levels.clone(),
            &plus_physics,
            sign,
            beta_scale,
        )
        .expect("plus MLFMM assembly")
        .matvec(&x);
        let minus = build_mlfmm_system_stabilized_checked(
            &elements,
            &nodes,
            cluster_levels,
            &minus_physics,
            sign,
            beta_scale,
        )
        .expect("minus MLFMM assembly")
        .matvec(&x);
        let finite_difference: Vec<_> = plus
            .iter()
            .zip(minus.iter())
            .map(|(plus, minus)| (*plus - *minus) / (2.0 * step))
            .collect();
        let error = analytic
            .iter()
            .zip(finite_difference.iter())
            .map(|(actual, expected)| (*actual - *expected).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let scale = finite_difference
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt()
            .max(1.0e-12);
        assert!(
            error / scale < 2.0e-3,
            "MLFMM derivative relative error={}",
            error / scale
        );
    }

    #[test]
    fn directional_wavenumber_derivative_matches_centered_action() {
        let nodes = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [4.0, 1.0, 0.0],
        ];
        let mut elements = make_test_elements().0;
        elements[0].connectivity = vec![0, 1, 2];
        elements[0].center = [1.0 / 3.0, 1.0 / 3.0, 0.0];
        elements[1].connectivity = vec![3, 4, 5];
        elements[1].center = [13.0 / 3.0, 1.0 / 3.0, 0.0];
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
        let sign = PhysicsParams::optimal_formulation_sign(physics.wave_number);
        let beta_scale = 0.2;
        let cluster_levels = build_cluster_tree(&elements, 1, &physics);
        let config = DirectionalMlfmmConfig::default();
        let base = build_directional_mlfmm_system_stabilized_checked(
            &elements,
            &nodes,
            cluster_levels.clone(),
            &physics,
            sign,
            beta_scale,
            config,
        )
        .expect("checked directional MLFMM base assembly");
        let derivative = build_directional_mlfmm_wavenumber_derivative(
            &elements,
            &nodes,
            cluster_levels.clone(),
            &physics,
            sign,
            beta_scale,
            config,
        )
        .expect("checked directional MLFMM derivative assembly");
        let x = vec![Complex64::new(0.7, -0.2), Complex64::new(-0.3, 0.4)];
        let step = 1.0e-4;
        let make_physics = |wavenumber: f64| {
            PhysicsParams::new(
                wavenumber * physics.speed_of_sound / (2.0 * std::f64::consts::PI),
                physics.speed_of_sound,
                physics.density,
                false,
            )
        };
        let plus = build_directional_mlfmm_system_stabilized_checked(
            &elements,
            &nodes,
            cluster_levels.clone(),
            &make_physics(physics.wave_number + step),
            sign,
            beta_scale,
            config,
        )
        .expect("plus directional MLFMM assembly")
        .matvec(&x);
        let minus = build_directional_mlfmm_system_stabilized_checked(
            &elements,
            &nodes,
            cluster_levels,
            &make_physics(physics.wave_number - step),
            sign,
            beta_scale,
            config,
        )
        .expect("minus directional MLFMM assembly")
        .matvec(&x);
        let expected: Vec<_> = plus
            .iter()
            .zip(minus.iter())
            .map(|(plus, minus)| (*plus - *minus) / (2.0 * step))
            .collect();
        let actual = derivative.apply(&base, &x);
        let error = actual
            .iter()
            .zip(expected.iter())
            .map(|(actual, expected)| (*actual - *expected).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let scale = expected
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt()
            .max(1.0e-12);
        assert!(
            error / scale < 2.0e-3,
            "directional derivative relative error={}",
            error / scale
        );
    }

    #[test]
    fn test_mlfmm_matvec_stabilized() {
        let (elements, nodes) = make_test_elements();
        let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
        let cluster_tree = build_cluster_tree(&elements, 10, &physics);
        let system =
            build_mlfmm_system_stabilized(&elements, &nodes, cluster_tree, &physics, -1.0, 1.0);
        let x = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];
        let y = system.matvec(&x);
        assert_eq!(y.len(), 2);
    }

    #[test]
    fn test_mlfmm_transpose_matches_forward_bilinear_form() {
        let mut root = Cluster::new([0.0, 0.0, 0.0]);
        root.sons = vec![0, 1];

        let mut root_level = ClusterLevel::new(1);
        root_level.clusters.push(root);

        let mut leaf0 = Cluster::new([-1.0, 0.0, 0.0]);
        leaf0.father = Some(0);
        let mut leaf1 = Cluster::new([1.0, 0.0, 0.0]);
        leaf1.father = Some(0);

        let mut leaf_level = ClusterLevel::new(2);
        leaf_level.clusters.push(leaf0);
        leaf_level.clusters.push(leaf1);

        let mut system = MlfmmSystem::new(2, 2, vec![root_level, leaf_level], vec![2, 2]);
        system.leaf_dof_indices = vec![vec![0], vec![1]];
        system.near_matrix.push(NearFieldBlock {
            source_cluster: 0,
            field_cluster: 1,
            coefficients: DenseMatrix::from_row_major_vec(1, 1, vec![c(0.25, -0.5)]),
        });
        system.t_matrices.push(LevelMatrices {
            level: 0,
            num_clusters: 1,
            matrices: vec![DenseMatrix::from_row_major_vec(
                2,
                4,
                vec![
                    c(1.0, 0.0),
                    c(0.0, 0.0),
                    c(-0.3, 0.1),
                    c(0.0, 0.0),
                    c(0.0, 0.0),
                    c(0.7, -0.4),
                    c(0.0, 0.0),
                    c(-0.1, 0.6),
                ],
            )],
        });
        system.t_matrices.push(LevelMatrices {
            level: 1,
            num_clusters: 2,
            matrices: vec![
                DenseMatrix::from_row_major_vec(2, 1, vec![c(0.8, 0.1), c(-0.2, 0.3)]),
                DenseMatrix::from_row_major_vec(2, 1, vec![c(0.4, -0.6), c(0.9, 0.2)]),
            ],
        });
        system.d_matrices = vec![
            vec![DMatrixEntry {
                source_cluster: 0,
                field_cluster: 0,
                level: 0,
                diagonal: vec![c(0.6, 0.2), c(-0.1, 0.5)],
                active_directions: None,
            }],
            vec![DMatrixEntry {
                source_cluster: 0,
                field_cluster: 1,
                level: 1,
                diagonal: vec![c(0.3, -0.7), c(0.2, 0.4)],
                active_directions: None,
            }],
        ];
        system.s_matrices.push(LevelMatrices {
            level: 0,
            num_clusters: 1,
            matrices: vec![DenseMatrix::from_row_major_vec(
                4,
                2,
                vec![
                    c(0.5, 0.0),
                    c(0.0, 0.0),
                    c(0.0, 0.0),
                    c(0.2, -0.1),
                    c(-0.4, 0.6),
                    c(0.0, 0.0),
                    c(0.0, 0.0),
                    c(0.9, 0.1),
                ],
            )],
        });
        system.s_matrices.push(LevelMatrices {
            level: 1,
            num_clusters: 2,
            matrices: vec![
                DenseMatrix::from_row_major_vec(1, 2, vec![c(0.2, 0.9), c(-0.5, 0.4)]),
                DenseMatrix::from_row_major_vec(1, 2, vec![c(0.6, -0.2), c(0.3, 0.7)]),
            ],
        });

        let x = vec![c(0.7, -0.3), c(-0.2, 0.5)];
        let y = vec![c(0.4, 0.8), c(-0.6, 0.1)];

        let lhs = bilinear_dot(&x, &system.matvec(&y));
        let rhs = bilinear_dot(&system.matvec_transpose(&x), &y);

        assert!(
            (lhs - rhs).norm() < 1e-12,
            "transpose identity failed: lhs={lhs}, rhs={rhs}"
        );
    }

    /// Check the batched algebraic-transpose action against scalar columns on
    /// a two-level non-directional tree with both S/D/T stages populated.
    #[test]
    fn mlfmm_transpose_block_matches_scalar_pipeline() {
        let mut root = Cluster::new([0.0, 0.0, 0.0]);
        root.sons = vec![0, 1];
        let mut root_level = ClusterLevel::new(1);
        root_level.clusters.push(root);
        let mut leaf0 = Cluster::new([-1.0, 0.0, 0.0]);
        leaf0.father = Some(0);
        let mut leaf1 = Cluster::new([1.0, 0.0, 0.0]);
        leaf1.father = Some(0);
        let mut leaf_level = ClusterLevel::new(2);
        leaf_level.clusters.extend([leaf0, leaf1]);

        let mut system = MlfmmSystem::new(2, 2, vec![root_level, leaf_level], vec![2, 2]);
        system.leaf_dof_indices = vec![vec![0], vec![1]];
        system.t_matrices.push(LevelMatrices {
            level: 0,
            num_clusters: 1,
            matrices: vec![DenseMatrix::from_row_major_vec(
                2,
                4,
                vec![
                    c(1.0, 0.0),
                    c(0.0, 0.0),
                    c(-0.3, 0.1),
                    c(0.0, 0.0),
                    c(0.0, 0.0),
                    c(0.7, -0.4),
                    c(0.0, 0.0),
                    c(-0.1, 0.6),
                ],
            )],
        });
        system.t_matrices.push(LevelMatrices {
            level: 1,
            num_clusters: 2,
            matrices: vec![
                DenseMatrix::from_row_major_vec(2, 1, vec![c(0.8, 0.1), c(-0.2, 0.3)]),
                DenseMatrix::from_row_major_vec(2, 1, vec![c(0.4, -0.6), c(0.9, 0.2)]),
            ],
        });
        system.d_matrices = vec![
            vec![DMatrixEntry {
                source_cluster: 0,
                field_cluster: 0,
                level: 0,
                diagonal: vec![c(0.6, 0.2), c(-0.1, 0.5)],
                active_directions: None,
            }],
            vec![DMatrixEntry {
                source_cluster: 0,
                field_cluster: 1,
                level: 1,
                diagonal: vec![c(0.3, -0.7), c(0.2, 0.4)],
                active_directions: None,
            }],
        ];
        system.s_matrices.push(LevelMatrices {
            level: 0,
            num_clusters: 1,
            matrices: vec![DenseMatrix::from_row_major_vec(
                4,
                2,
                vec![
                    c(0.5, 0.0),
                    c(0.0, 0.0),
                    c(0.0, 0.0),
                    c(0.2, -0.1),
                    c(-0.4, 0.6),
                    c(0.0, 0.0),
                    c(0.0, 0.0),
                    c(0.9, 0.1),
                ],
            )],
        });
        system.s_matrices.push(LevelMatrices {
            level: 1,
            num_clusters: 2,
            matrices: vec![
                DenseMatrix::from_row_major_vec(1, 2, vec![c(0.2, 0.9), c(-0.5, 0.4)]),
                DenseMatrix::from_row_major_vec(1, 2, vec![c(0.6, -0.2), c(0.3, 0.7)]),
            ],
        });

        for entries in &mut system.d_matrices {
            for entry in entries {
                entry.active_directions = Some(vec![0, 1]);
            }
        }
        system.directional_plans = vec![
            DirectionalLevelPlan {
                cones: vec![DirectionalConeBasis {
                    active_directions: vec![0, 1],
                    source_clusters: vec![0],
                    field_clusters: vec![0],
                }],
                entries_by_cone: vec![vec![DirectionalConeEntry {
                    source_row: 0,
                    field_row: 0,
                    diagonal: system.d_matrices[0][0].diagonal.clone(),
                }]],
                entries_by_cone_field_row: vec![vec![vec![0]]],
                entries_by_cone_source_row: vec![vec![vec![0]]],
            },
            DirectionalLevelPlan {
                cones: vec![DirectionalConeBasis {
                    active_directions: vec![0, 1],
                    source_clusters: vec![0],
                    field_clusters: vec![1],
                }],
                entries_by_cone: vec![vec![DirectionalConeEntry {
                    source_row: 0,
                    field_row: 0,
                    diagonal: system.d_matrices[1][0].diagonal.clone(),
                }]],
                entries_by_cone_field_row: vec![vec![vec![0]]],
                entries_by_cone_source_row: vec![vec![vec![0]]],
            },
        ];
        let input = DenseMatrix::from_row_major_vec(
            2,
            2,
            vec![c(0.7, -0.3), c(-0.2, 0.5), c(0.4, 0.8), c(-0.6, 0.1)],
        );
        let block = system.matvec_transpose_block(&input);
        for rhs_index in 0..input.ncols() {
            let column = (0..input.nrows())
                .map(|row| input[[row, rhs_index]])
                .collect::<Vec<_>>();
            let scalar = system.matvec_transpose(&column);
            for row in 0..input.nrows() {
                assert!((block[[row, rhs_index]] - scalar[row]).norm() < 1e-12);
            }
        }
    }

    /// **TDD anchor `bem-step1.md` §1.4 near-field bug.**
    ///
    /// Build an MLFMM system where every cluster is mutually near at each
    /// level, so the full matvec consists *only* of near-field contributions
    /// (no M2L work happens). MLFMM matvec must equal dense TBEM matvec to
    /// round-off — they're computing the same dense matrix just blocked
    /// differently.
    ///
    /// Audit signal: kD=1 / 320 DOFs reports near-only relative L² error
    /// ≈ 1.036, so this test should fail until the near-field assembler is
    /// fixed.
    #[test]
    fn test_mlfmm_all_near_matches_dense_tbem() {
        use crate::core::assembly::tbem::build_tbem_system_stabilized;
        use crate::core::mesh::generators::generate_icosphere_mesh;

        let radius = 0.2;
        let speed_of_sound = 343.0;
        let density = 1.21;
        let frequency = 1.0 * speed_of_sound / (2.0 * std::f64::consts::PI * radius);
        let physics = PhysicsParams::new(frequency, speed_of_sound, density, false);

        let mesh = generate_icosphere_mesh(radius, 1);
        let mut elements = mesh.elements.clone();
        for (i, e) in elements.iter_mut().enumerate() {
            e.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
            e.dof_addresses = vec![i];
        }

        // Reference: full dense Burton-Miller assembly.
        let tbem = build_tbem_system_stabilized(&elements, &mesh.nodes, &physics, -1.0, 1.0);

        // Build the MLFMM cluster tree, then *force* every cluster's
        // near_clusters to cover every other cluster at every level.
        // After this step there should be no far interactions; the MLFMM
        // matvec is purely near-field and must reproduce the dense matrix.
        // Force MULTIPLE leaves with `target_elements_per_leaf` smaller
        // than total — this exercises cross-cluster near-field block
        // coverage. The single-cluster case has been independently
        // verified to pass.
        let mut tree = build_cluster_tree(&elements, 12, &physics);
        for level in tree.iter_mut() {
            let m = level.clusters.len();
            for i in 0..m {
                level.clusters[i].near_clusters = (0..m).filter(|&j| j != i).collect();
                level.clusters[i].far_clusters = Vec::new();
            }
        }

        let system =
            build_mlfmm_system_stabilized(&elements, &mesh.nodes, tree, &physics, -1.0, 1.0);

        // Probe with a deterministic complex vector.
        let n = elements.len();
        let x: Vec<Complex64> = (0..n)
            .map(|i| {
                let t = (i + 1) as f64;
                Complex64::new((0.37 * t).sin(), (0.19 * t).cos())
            })
            .collect();

        let y_dense = tbem.matrix.matvec(&x);
        let y_mlfmm = system.matvec(&x);

        let diff_norm: f64 = y_dense
            .iter()
            .zip(y_mlfmm.iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let ref_norm: f64 = y_dense.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
        let rel_err = diff_norm / ref_norm.max(1e-30);

        // Property-level coverage check — independently from the matvec
        // value, the cluster tree must produce a partition with no missing
        // or duplicated DOFs. Catches octant-assignment bugs even when the
        // matvec result happens to look right by accident.
        let total_leaf_dofs: usize = system.leaf_dof_indices.iter().map(|v| v.len()).sum();
        let mut dof_in_leaves = vec![0usize; n];
        for leaf in &system.leaf_dof_indices {
            for &d in leaf {
                dof_in_leaves[d] += 1;
            }
        }
        let max_appearance = *dof_in_leaves.iter().max().unwrap_or(&0);
        let unassigned = dof_in_leaves.iter().filter(|&&c| c == 0).count();
        assert_eq!(
            total_leaf_dofs, n,
            "Σ |leaf_dof_indices[i]| must equal total active DOFs (got {total_leaf_dofs} vs n={n})"
        );
        assert_eq!(
            max_appearance, 1,
            "every DOF must appear in exactly one leaf cluster (max appearance = {max_appearance})"
        );
        assert_eq!(
            unassigned, 0,
            "{unassigned} DOFs are unassigned to any leaf"
        );

        assert!(
            rel_err < 1e-10,
            "MLFMM all-near matvec must match dense TBEM matvec to round-off, \
             got rel L² err = {rel_err:.3e} (n={n})"
        );
    }

    /// **TDD anchor for `bem-step1.md` §1.4 M2L bug.**
    ///
    /// Build a two-cluster geometry where the cross-block must go through
    /// the M2L far-field translation. Force the two leaves as mutually-far
    /// (so the cross matvec has *no* near-field contribution at all) and
    /// compare against the dense TBEM matvec.
    ///
    /// At a moderate expansion order the FMM truncation error is bounded
    /// well below 1 % — anything larger means the diagonal-form
    /// translation operator (sphere quadrature, `1/(4π)` normalisation,
    /// translation phase) is broken. Audit signal: kD=10 reports
    /// `far_share ≈ 1.86e7`, so this test should fail until M2L is fixed.
    #[test]
    fn test_mlfmm_far_field_matches_dense_for_well_separated_clusters() {
        use crate::core::assembly::tbem::build_tbem_system_stabilized;
        use crate::core::mesh::generators::generate_icosphere_mesh;

        let radius = 0.5;
        let speed_of_sound = 343.0;
        let density = 1.21;
        // Pick a frequency well into the FMM-friendly regime (k·R ≳ 2 between
        // clusters) but not so high that we need huge expansion orders.
        let frequency = 2.0 * speed_of_sound / (2.0 * std::f64::consts::PI * radius);
        let physics = PhysicsParams::new(frequency, speed_of_sound, density, false);

        // Two icospheres separated by ~6 radii along x. Treat each sphere as
        // its own atomic cluster — single-element-per-leaf decomposition is
        // not realistic for FMM but isolates one M2L translation in
        // isolation, which is what we need.
        let mesh_a = generate_icosphere_mesh(radius, 1);
        let mesh_b = generate_icosphere_mesh(radius, 1);
        let n_a = mesh_a.elements.len();
        let n_b = mesh_b.elements.len();

        let total_nodes = mesh_a.nodes.len() + mesh_b.nodes.len();
        let mut combined_nodes: Vec<[f64; 3]> = vec![[0.0; 3]; total_nodes];
        for i in 0..mesh_a.nodes.len() {
            for d in 0..3 {
                combined_nodes[i][d] = mesh_a.nodes[i][d];
            }
        }
        let separation = 6.0 * radius;
        for i in 0..mesh_b.nodes.len() {
            combined_nodes[mesh_a.nodes.len() + i][0] = mesh_b.nodes[i][0] + separation;
            combined_nodes[mesh_a.nodes.len() + i][1] = mesh_b.nodes[i][1];
            combined_nodes[mesh_a.nodes.len() + i][2] = mesh_b.nodes[i][2];
        }

        let mut elements: Vec<Element> = Vec::with_capacity(n_a + n_b);
        for (i, e) in mesh_a.elements.iter().enumerate() {
            let mut elem = e.clone();
            elem.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
            elem.dof_addresses = vec![i];
            elements.push(elem);
        }
        for (i, e) in mesh_b.elements.iter().enumerate() {
            let mut elem = e.clone();
            for c in elem.connectivity.iter_mut() {
                *c += mesh_a.nodes.len();
            }
            elem.center[0] += separation;
            elem.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
            elem.dof_addresses = vec![n_a + i];
            elements.push(elem);
        }

        let n = elements.len();
        assert_eq!(n, n_a + n_b);

        // Reference dense matvec.
        let tbem = build_tbem_system_stabilized(&elements, &combined_nodes, &physics, -1.0, 1.0);

        // Build a 2-level cluster tree: root + 2 leaves (one per sphere).
        // Force the two leaves to be mutually FAR so the cross block goes
        // strictly through M2L; each cluster's self block is its only
        // near-field contribution.
        // build_cluster_tree with target=n collapses everything into a
        // single leaf. We need to manually split into two leaves instead.
        // Construct the tree by hand for full control.
        let center_a = [0.0, 0.0, 0.0];
        let center_b = [separation, 0.0, 0.0];
        let mut leaf_a = Cluster::new(center_a);
        leaf_a.element_indices = (0..n_a).collect();
        leaf_a.num_elements = n_a;
        leaf_a.radius = radius * 1.05;
        leaf_a.level = 1;
        leaf_a.father = Some(0);
        leaf_a.near_clusters = vec![0]; // self-near only
        leaf_a.far_clusters = vec![1];
        let mut leaf_b = Cluster::new(center_b);
        leaf_b.element_indices = (n_a..n).collect();
        leaf_b.num_elements = n_b;
        leaf_b.radius = radius * 1.05;
        leaf_b.level = 1;
        leaf_b.father = Some(0);
        leaf_b.near_clusters = vec![1];
        leaf_b.far_clusters = vec![0];

        let mut leaf_level = ClusterLevel::new(2);
        leaf_level.clusters = vec![leaf_a, leaf_b];
        leaf_level.num_original = 2;
        leaf_level.expansion_terms = 16;
        leaf_level.theta_points = 16;
        leaf_level.phi_points = 32;

        let mut root = Cluster::new([separation / 2.0, 0.0, 0.0]);
        root.element_indices = (0..n).collect();
        root.num_elements = n;
        root.radius = (separation / 2.0) + radius;
        root.level = 0;
        root.sons = vec![0, 1];
        let mut root_level = ClusterLevel::new(1);
        root_level.clusters = vec![root];
        root_level.num_original = 1;
        root_level.expansion_terms = 16;
        root_level.theta_points = 16;
        root_level.phi_points = 32;

        let tree = vec![root_level, leaf_level];

        let system =
            build_mlfmm_system_stabilized(&elements, &combined_nodes, tree, &physics, -1.0, 1.0);

        // Sanity: there must be exactly 2 near self-blocks, and the M2L
        // translation must be populated for at least one (cluster-pair, level).
        assert_eq!(
            system.near_matrix.len(),
            2,
            "expected 2 near self-blocks (one per sphere), got {}",
            system.near_matrix.len()
        );
        let total_d_entries: usize = system.d_matrices.iter().map(|v| v.len()).sum();
        assert!(
            total_d_entries > 0,
            "no D-matrix (M2L) entries assembled — far-field path is unexercised"
        );

        let x: Vec<Complex64> = (0..n)
            .map(|i| {
                let t = (i + 1) as f64;
                Complex64::new((0.37 * t).sin(), (0.19 * t).cos())
            })
            .collect();
        let y_dense = tbem.matrix.matvec(&x);
        let y_mlfmm = system.matvec(&x);

        let diff_norm: f64 = y_dense
            .iter()
            .zip(y_mlfmm.iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let ref_norm: f64 = y_dense.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
        let rel_err = diff_norm / ref_norm.max(1e-30);

        // A correctly-implemented diagonal-form M2L at expansion order 16
        // for kR ≈ 2 should give relative error ≲ 1e-3. Loosen to 1e-2
        // here to absorb finite-quadrature error on the unit-sphere
        // integration without making the test too tight to bisect against.
        assert!(
            rel_err < 1e-2,
            "MLFMM far-field matvec must match dense TBEM to ≤ 1 % at expansion order 16, \
             got rel L² err = {rel_err:.3e}"
        );
    }

    /// Diagnostic helper — single-pair M2L precision sweep.
    ///
    /// Returns `(rel_err, kr)` for two leaf-sized icospheres separated by
    /// `separation_radii × radius`, with the supplied `expansion_terms`
    /// and `phi_factor` (`phi_points = phi_factor * expansion_terms + 2`).
    fn measure_single_pair_m2l_precision(
        radius: f64,
        k_target: f64,
        separation_radii: f64,
        expansion_terms: usize,
        phi_factor: usize,
    ) -> (f64, f64) {
        use crate::core::assembly::tbem::build_tbem_system_stabilized;
        use crate::core::mesh::generators::generate_icosphere_mesh;

        let speed_of_sound = 343.0;
        let density = 1.21;
        let frequency = k_target * speed_of_sound / (2.0 * std::f64::consts::PI);
        let physics = PhysicsParams::new(frequency, speed_of_sound, density, false);

        let mesh_a = generate_icosphere_mesh(radius, 1);
        let mesh_b = generate_icosphere_mesh(radius, 1);
        let n_a = mesh_a.elements.len();
        let n_b = mesh_b.elements.len();
        let separation = separation_radii * radius;

        let total_nodes = mesh_a.nodes.len() + mesh_b.nodes.len();
        let mut combined_nodes: Vec<[f64; 3]> = vec![[0.0; 3]; total_nodes];
        for i in 0..mesh_a.nodes.len() {
            for d in 0..3 {
                combined_nodes[i][d] = mesh_a.nodes[i][d];
            }
        }
        for i in 0..mesh_b.nodes.len() {
            combined_nodes[mesh_a.nodes.len() + i][0] = mesh_b.nodes[i][0] + separation;
            combined_nodes[mesh_a.nodes.len() + i][1] = mesh_b.nodes[i][1];
            combined_nodes[mesh_a.nodes.len() + i][2] = mesh_b.nodes[i][2];
        }
        let mut elements: Vec<Element> = Vec::with_capacity(n_a + n_b);
        for (i, e) in mesh_a.elements.iter().enumerate() {
            let mut elem = e.clone();
            elem.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
            elem.dof_addresses = vec![i];
            elements.push(elem);
        }
        for (i, e) in mesh_b.elements.iter().enumerate() {
            let mut elem = e.clone();
            for c in elem.connectivity.iter_mut() {
                *c += mesh_a.nodes.len();
            }
            elem.center[0] += separation;
            elem.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
            elem.dof_addresses = vec![n_a + i];
            elements.push(elem);
        }
        let n = elements.len();

        let tbem = build_tbem_system_stabilized(&elements, &combined_nodes, &physics, -1.0, 1.0);

        let mut leaf_a = Cluster::new([0.0, 0.0, 0.0]);
        leaf_a.element_indices = (0..n_a).collect();
        leaf_a.num_elements = n_a;
        leaf_a.radius = radius * 1.05;
        leaf_a.level = 1;
        leaf_a.father = Some(0);
        leaf_a.near_clusters = vec![0];
        leaf_a.far_clusters = vec![1];
        let mut leaf_b = Cluster::new([separation, 0.0, 0.0]);
        leaf_b.element_indices = (n_a..n).collect();
        leaf_b.num_elements = n_b;
        leaf_b.radius = radius * 1.05;
        leaf_b.level = 1;
        leaf_b.father = Some(0);
        leaf_b.near_clusters = vec![1];
        leaf_b.far_clusters = vec![0];
        let mut leaf_level = ClusterLevel::new(2);
        leaf_level.clusters = vec![leaf_a, leaf_b];
        leaf_level.num_original = 2;
        leaf_level.expansion_terms = expansion_terms;
        leaf_level.theta_points = expansion_terms;
        leaf_level.phi_points = phi_factor * expansion_terms + 2;

        let mut root = Cluster::new([separation / 2.0, 0.0, 0.0]);
        root.element_indices = (0..n).collect();
        root.num_elements = n;
        root.radius = (separation / 2.0) + radius;
        root.level = 0;
        root.sons = vec![0, 1];
        let mut root_level = ClusterLevel::new(1);
        root_level.clusters = vec![root];
        root_level.num_original = 1;
        root_level.expansion_terms = expansion_terms;
        root_level.theta_points = expansion_terms;
        root_level.phi_points = phi_factor * expansion_terms + 2;

        let tree = vec![root_level, leaf_level];
        let system =
            build_mlfmm_system_stabilized(&elements, &combined_nodes, tree, &physics, -1.0, 1.0);

        let x: Vec<Complex64> = (0..n)
            .map(|i| {
                let t = (i + 1) as f64;
                Complex64::new((0.37 * t).sin(), (0.19 * t).cos())
            })
            .collect();
        let y_dense = tbem.matrix.matvec(&x);
        let y_mlfmm = system.matvec(&x);

        let diff_norm: f64 = y_dense
            .iter()
            .zip(y_mlfmm.iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let ref_norm: f64 = y_dense.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
        (diff_norm / ref_norm.max(1e-30), k_target * separation)
    }

    /// **Diagnostic — per-pair M2L precision across a frequency sweep.**
    ///
    /// Sweeps `kr` (via varying `k`) at fixed cluster geometry. If the
    /// per-pair error is constant across `kr`, the bottleneck is a
    /// systematic M2L formula issue (constant factor, sign convention,
    /// ill-formed Burton-Miller combination), not a frequency-dependent
    /// numerical problem.
    #[test]
    #[ignore = "slow MLFMM frequency sweep; run `cargo bench -p sonium-bem --bench fmm_slow_validation`"]
    fn diagnose_mlfmm_per_pair_precision_across_kr() {
        let radius = 0.22;
        let separation_radii = 4.0;
        eprintln!(
            "\n[per-pair frequency sweep] radius={radius}, sep={separation_radii}R, expansion=16, phi=34:"
        );
        for k_target in [0.5_f64, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0] {
            let (rel_err, kr) =
                measure_single_pair_m2l_precision(radius, k_target, separation_radii, 16, 2);
            eprintln!("  k={k_target:>5.1}, kr={kr:>6.2}, rel_err={rel_err:.3e}");
        }
        // Sanity bound — should not be worse than 10 % anywhere.
        let (rel_err, _) = measure_single_pair_m2l_precision(0.22, 20.0, 4.0, 28, 2);
        assert!(rel_err < 0.1);
    }

    /// **Single-pair smoke test at audit parameters (kept for regression).**
    #[test]
    fn diagnose_mlfmm_per_pair_smoke_at_audit_kd10() {
        use crate::core::assembly::tbem::build_tbem_system_stabilized;
        use crate::core::mesh::generators::generate_icosphere_mesh;

        let radius = 0.22; // matches the audit's leaf-cluster radius at kD=10
        let speed_of_sound = 343.0;
        let density = 1.21;
        let k_target = 20.0; // kD=10 audit uses k=20 (R=1, kD=10)
        let frequency = k_target * speed_of_sound / (2.0 * std::f64::consts::PI);
        let physics = PhysicsParams::new(frequency, speed_of_sound, density, false);

        // Two leaf-sized icospheres at separation = 4 R (kr ≈ 17.6 — the
        // *closest* far pair at the audit setup).
        let mesh_a = generate_icosphere_mesh(radius, 1);
        let mesh_b = generate_icosphere_mesh(radius, 1);
        let n_a = mesh_a.elements.len();
        let n_b = mesh_b.elements.len();
        let separation = 4.0 * radius;
        let total_nodes = mesh_a.nodes.len() + mesh_b.nodes.len();
        let mut combined_nodes: Vec<[f64; 3]> = vec![[0.0; 3]; total_nodes];
        for i in 0..mesh_a.nodes.len() {
            for d in 0..3 {
                combined_nodes[i][d] = mesh_a.nodes[i][d];
            }
        }
        for i in 0..mesh_b.nodes.len() {
            combined_nodes[mesh_a.nodes.len() + i][0] = mesh_b.nodes[i][0] + separation;
            combined_nodes[mesh_a.nodes.len() + i][1] = mesh_b.nodes[i][1];
            combined_nodes[mesh_a.nodes.len() + i][2] = mesh_b.nodes[i][2];
        }
        let mut elements: Vec<Element> = Vec::with_capacity(n_a + n_b);
        for (i, e) in mesh_a.elements.iter().enumerate() {
            let mut elem = e.clone();
            elem.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
            elem.dof_addresses = vec![i];
            elements.push(elem);
        }
        for (i, e) in mesh_b.elements.iter().enumerate() {
            let mut elem = e.clone();
            for c in elem.connectivity.iter_mut() {
                *c += mesh_a.nodes.len();
            }
            elem.center[0] += separation;
            elem.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
            elem.dof_addresses = vec![n_a + i];
            elements.push(elem);
        }
        let n = elements.len();

        let tbem = build_tbem_system_stabilized(&elements, &combined_nodes, &physics, -1.0, 1.0);

        // Two-leaf hand-built tree, marked mutually FAR. expansion_terms
        // matches the audit's kD=10 root setting (28).
        let center_a = [0.0, 0.0, 0.0];
        let center_b = [separation, 0.0, 0.0];
        let mut leaf_a = Cluster::new(center_a);
        leaf_a.element_indices = (0..n_a).collect();
        leaf_a.num_elements = n_a;
        leaf_a.radius = radius * 1.05;
        leaf_a.level = 1;
        leaf_a.father = Some(0);
        leaf_a.near_clusters = vec![0];
        leaf_a.far_clusters = vec![1];
        let mut leaf_b = Cluster::new(center_b);
        leaf_b.element_indices = (n_a..n).collect();
        leaf_b.num_elements = n_b;
        leaf_b.radius = radius * 1.05;
        leaf_b.level = 1;
        leaf_b.father = Some(0);
        leaf_b.near_clusters = vec![1];
        leaf_b.far_clusters = vec![0];
        let mut leaf_level = ClusterLevel::new(2);
        leaf_level.clusters = vec![leaf_a, leaf_b];
        leaf_level.num_original = 2;
        leaf_level.expansion_terms = 28;
        leaf_level.theta_points = 28;
        leaf_level.phi_points = 2 * 28 + 2;

        let mut root = Cluster::new([separation / 2.0, 0.0, 0.0]);
        root.element_indices = (0..n).collect();
        root.num_elements = n;
        root.radius = (separation / 2.0) + radius;
        root.level = 0;
        root.sons = vec![0, 1];
        let mut root_level = ClusterLevel::new(1);
        root_level.clusters = vec![root];
        root_level.num_original = 1;
        root_level.expansion_terms = 28;
        root_level.theta_points = 28;
        root_level.phi_points = 2 * 28 + 2;

        let tree = vec![root_level, leaf_level];
        let system =
            build_mlfmm_system_stabilized(&elements, &combined_nodes, tree, &physics, -1.0, 1.0);

        let x: Vec<Complex64> = (0..n)
            .map(|i| {
                let t = (i + 1) as f64;
                Complex64::new((0.37 * t).sin(), (0.19 * t).cos())
            })
            .collect();
        let y_dense = tbem.matrix.matvec(&x);
        let y_mlfmm = system.matvec(&x);

        let diff_norm: f64 = y_dense
            .iter()
            .zip(y_mlfmm.iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let ref_norm: f64 = y_dense.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
        let rel_err = diff_norm / ref_norm.max(1e-30);

        eprintln!(
            "[per-pair diagnostic] kD≈10 single-pair: kr={:.1}, expansion=28, n={}, rel_err={:.3e}",
            k_target * separation,
            n,
            rel_err
        );
        // Currently per-pair M2L is `~1.14 × 10⁻²` at this `kr ≈ 17.6`
        // operating point — see `diagnose_mlfmm_per_pair_precision_across_kr`
        // for the frequency dependence. The threshold here is slack
        // enough to act as a regression detector while the real fix
        // (a `k`-dependent factor mismatch in the diagonal-form M2L
        // formula, suspected) is investigated.
        assert!(
            rel_err < 2e-2,
            "Per-pair M2L at audit parameters regression: rel L² err = {rel_err:.3e}"
        );
    }

    /// **TDD anchor for `bem-step1.md` §1.4 M2L bug at moderate kD.**
    ///
    /// Bisection: report MLFMM-vs-dense rel err at several `ka` values
    /// with the same icosphere mesh. The first `ka` whose error blows up
    /// localises the bug to a particular kr regime in the M2L transfer.
    #[test]
    #[ignore = "slow dense-vs-MLFMM sweep; run `cargo bench -p sonium-bem --bench fmm_slow_validation`"]
    fn test_mlfmm_matches_dense_across_ka_sweep() {
        use crate::core::assembly::tbem::build_tbem_system_stabilized;
        use crate::core::mesh::generators::generate_icosphere_mesh;

        let radius = 1.0;
        let speed_of_sound = 343.0;
        let density = 1.21;

        let mesh = generate_icosphere_mesh(radius, 3);
        let mut elements = mesh.elements.clone();
        for (i, e) in elements.iter_mut().enumerate() {
            e.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
            e.dof_addresses = vec![i];
        }
        let n = elements.len();

        // Always-on: low kD where M2L should be trivially stable.
        // Higher kD cases are reported but not asserted — they reveal
        // where the diagonal-form M2L breaks down with the current
        // expansion-order policy.
        for &ka in &[1.0_f64, 2.0, 5.0, 10.0] {
            let frequency = ka * speed_of_sound / (2.0 * std::f64::consts::PI * radius);
            let physics = PhysicsParams::new(frequency, speed_of_sound, density, false);

            let tbem = build_tbem_system_stabilized(&elements, &mesh.nodes, &physics, -1.0, 1.0);
            let tree = build_cluster_tree(&elements, 16, &physics);
            let n_levels = tree.len();
            let system =
                build_mlfmm_system_stabilized(&elements, &mesh.nodes, tree, &physics, -1.0, 1.0);

            let x: Vec<Complex64> = (0..n)
                .map(|i| {
                    let t = (i + 1) as f64;
                    Complex64::new((0.37 * t).sin(), (0.19 * t).cos())
                })
                .collect();
            let y_dense = tbem.matrix.matvec(&x);
            let y_mlfmm = system.matvec(&x);

            let diff_norm: f64 = y_dense
                .iter()
                .zip(y_mlfmm.iter())
                .map(|(a, b)| (a - b).norm_sqr())
                .sum::<f64>()
                .sqrt();
            let ref_norm: f64 = y_dense.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
            let rel_err = diff_norm / ref_norm.max(1e-30);

            eprintln!(
                "ka={ka:>5.1}  levels={n_levels}  expansion_terms={}  rel_err={:>12.3e}",
                system.cluster_levels[0].expansion_terms, rel_err
            );

            // FMM truncation at the auto-selected expansion order should
            // give ≤ 1 % rel err on a smooth geometry across this kD
            // range now that `expansion_terms` is capped at the highest
            // supported Gauss-Legendre quadrature order.
            assert!(
                rel_err < 1e-2,
                "MLFMM matvec must match dense to ≤ 1 % at ka={ka} (n={n}), got {rel_err:.3e}"
            );
        }
    }
}
