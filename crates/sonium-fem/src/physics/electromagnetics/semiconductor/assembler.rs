//! Semiconductor drift-diffusion assembler helpers.

use std::collections::HashMap;

use crate::assembly::TripletMatrix;
use crate::assembly::prebaked::PrebakedAssembler;
use crate::basis::PolynomialDegree;
use crate::mesh::Mesh;
use crate::quadrature;
use crate::spaces::H1Space;
use sonium_solvers::CsrMatrix;

use super::{
    CarrierDiffusionForm, CarrierDriftForm, CarrierMaterial, CarrierRecombinationForm,
    DopingChargeForm, DopingProfile,
};

/// Components:
/// 0 = carrier diffusion, 1 = carrier drift, 2 = recombination.
pub type SemiconductorAssembler = PrebakedAssembler<3>;

/// Build semiconductor drift-diffusion components without boundary contributions.
pub fn build_semiconductor_assembler(
    mesh: &Mesh,
    degree: PolynomialDegree,
    carrier: CarrierMaterial,
    electric_field: [f64; 3],
) -> SemiconductorAssembler {
    build_semiconductor_assembler_with_boundaries(mesh, degree, &[], carrier, electric_field)
}

/// Build semiconductor drift-diffusion components with optional boundary mass terms.
pub fn build_semiconductor_assembler_with_boundaries(
    mesh: &Mesh,
    degree: PolynomialDegree,
    boundary_tags: &[i32],
    carrier: CarrierMaterial,
    electric_field: [f64; 3],
) -> SemiconductorAssembler {
    let space = H1Space::new(degree);
    let quad_stiffness = quadrature::required_order_for_stiffness(degree.degree());
    // Drift is test-value times trial-gradient: degree 2p-1, one more
    // than the diffusion (gradient-gradient) stiffness integrand.
    let quad_drift = quadrature::required_order_for_drift(degree.degree());
    let quad_mass = quadrature::required_order_for_mass(degree.degree());

    let diffusion = crate::assembly::assemble_bilinear(
        &CarrierDiffusionForm::new(&carrier),
        &space,
        &space,
        mesh,
        quad_stiffness,
    );
    let drift = crate::assembly::assemble_bilinear(
        &CarrierDriftForm {
            electric_field,
            drift_coefficient: carrier.drift_coefficient(),
        },
        &space,
        &space,
        mesh,
        quad_drift,
    );
    let recombination = crate::assembly::assemble_bilinear(
        &CarrierRecombinationForm {
            lifetime: carrier.lifetime,
        },
        &space,
        &space,
        mesh,
        quad_mass,
    );

    let mut boundary_triplets: Vec<(usize, TripletMatrix<f64>)> = Vec::new();
    for &tag in boundary_tags {
        let b_mass = crate::assembly::assemble_boundary_mass_generic(mesh, tag);
        if b_mass.nnz() > 0 {
            boundary_triplets.push((tag as usize, b_mass));
        }
    }

    PrebakedAssembler::from_triplets([diffusion, drift, recombination], &boundary_triplets)
}

/// Assemble a doping-charge RHS vector for Poisson coupling.
pub fn assemble_doping_charge_rhs(
    mesh: &Mesh,
    degree: PolynomialDegree,
    profile: DopingProfile,
) -> Vec<f64> {
    let space = H1Space::new(degree);
    let quad_mass = quadrature::required_order_for_mass(degree.degree());
    crate::assembly::assemble_linear(&DopingChargeForm { profile }, &space, mesh, quad_mass)
}

/// Extension methods for semiconductor matrix assembly.
pub trait SemiconductorAssemblerExt {
    /// Assemble `A = a0 * K_diff + a1 * K_drift + a2 * M_recomb`.
    fn assemble_carrier_operator(
        &self,
        diffusion_scale: f64,
        drift_scale: f64,
        recombination_scale: f64,
        boundary_coeffs: &HashMap<usize, f64>,
    ) -> CsrMatrix<f64>;
}

impl SemiconductorAssemblerExt for SemiconductorAssembler {
    fn assemble_carrier_operator(
        &self,
        diffusion_scale: f64,
        drift_scale: f64,
        recombination_scale: f64,
        boundary_coeffs: &HashMap<usize, f64>,
    ) -> CsrMatrix<f64> {
        let mut system = self.assemble_real(&[diffusion_scale, drift_scale, recombination_scale]);
        for (&tag, &coeff) in boundary_coeffs {
            if let Some(boundary_values) = self.boundary_components.get(&tag) {
                for (a, &b) in system.values.iter_mut().zip(boundary_values.iter()) {
                    *a += coeff * b;
                }
            }
        }
        system
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::unit_square_triangles;

    #[test]
    fn build_semiconductor_assembler_smoke() {
        let mesh = unit_square_triangles(2);
        let carrier = CarrierMaterial {
            mobility: 0.14,
            lifetime: 1.0e-6,
            temperature: 300.0,
            kind: super::super::CarrierKind::Electron,
        };
        let assembler =
            build_semiconductor_assembler(&mesh, PolynomialDegree::P1, carrier, [1.0, 0.0, 0.0]);
        assert!(assembler.nnz() > 0);
        assert_eq!(assembler.num_rows, mesh.num_nodes());

        let rhs = assemble_doping_charge_rhs(
            &mesh,
            PolynomialDegree::P1,
            DopingProfile {
                donors: 1.0e21,
                acceptors: 2.0e20,
            },
        );
        assert_eq!(rhs.len(), mesh.num_nodes());
    }

    #[test]
    fn p2_drift_component_matches_high_order_reference() {
        // The drift integrand (test value x trial gradient) is degree
        // 2p-1 = 3 at P2; assembling it with the stiffness order 2p-2 = 2
        // under-integrates it.
        let mesh = unit_square_triangles(2);
        let carrier = CarrierMaterial {
            mobility: 0.14,
            lifetime: 1.0e-6,
            temperature: 300.0,
            kind: super::super::CarrierKind::Electron,
        };
        let field = [1.3, -0.7, 0.0];
        let assembler = build_semiconductor_assembler(&mesh, PolynomialDegree::P2, carrier, field);
        let drift = assembler.assemble_real(&[0.0, 1.0, 0.0]);

        let space = H1Space::new(PolynomialDegree::P2);
        let reference = crate::assembly::assemble_bilinear(
            &CarrierDriftForm {
                electric_field: field,
                drift_coefficient: carrier.drift_coefficient(),
            },
            &space,
            &space,
            &mesh,
            6,
        )
        .to_csr();

        assert_eq!(drift.num_rows, reference.num_rows);
        assert_eq!(drift.col_indices, reference.col_indices);
        for (index, (actual, expected)) in
            drift.values.iter().zip(reference.values.iter()).enumerate()
        {
            assert!(
                (actual - expected).abs() < 1e-10,
                "drift entry {index}: {actual:e} != {expected:e}"
            );
        }
    }
}
