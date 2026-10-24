//! Auxiliary aggregation must preserve every coarse degree of freedom.

// Rust guideline compliant 2026-02-21
use super::*;
use crate::mesh::unit_cube_hexahedra;
use sonium_solvers::{AmgConfig, AmgPreconditioner, AmgSmoother};

#[test]
fn shifted_lor_aggregation_has_no_empty_coarse_columns() {
    let mesh = unit_cube_hexahedra(2);
    let space = H1Space::new(PolynomialDegree::P2);
    let auxiliary =
        prepare_hex_lor_shifted_laplacian(&mesh, &space, Complex64::new(25.0, 25.0), &[]).unwrap();
    let config = AmgConfig {
        coarse_size: 16,
        smoother: AmgSmoother::Jacobi,
        jacobi_weight: 0.6,
        ..AmgConfig::for_aggregation()
    };
    let hierarchy = AmgPreconditioner::try_from_csr(&auxiliary, config).unwrap();
    eprintln!("aggregation levels {:?}", hierarchy.level_dims());
    for level in 0..hierarchy.num_levels() - 1 {
        let p = hierarchy.level_prolongation(level).unwrap();
        let mut norms = vec![0.0; p.num_cols];
        for (&column, value) in p.col_indices.iter().zip(&p.values) {
            norms[column] += value.norm_sqr();
        }
        eprintln!("level {level} coarse column norms {norms:?}");
        assert!(
            norms.iter().all(|&norm| norm > 1e-24),
            "empty column at level {level}"
        );
    }
}
