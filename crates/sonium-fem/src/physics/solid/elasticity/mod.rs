//! Linear elasticity physics module
//!
//! Solves the static equilibrium equation `div(sigma) + f = 0` where
//! `sigma = C : epsilon(u)` is the Cauchy stress and `epsilon = sym(grad u)` is
//! the infinitesimal strain tensor.
//!
//! # Usage
//!
//! ```ignore
//! use sonium_fem::physics::solid::elasticity::{ElasticityForm, IsotropicMaterial, PlaneAssumption};
//! use sonium_fem::spaces::VectorH1Space;
//! use sonium_fem::basis::PolynomialDegree;
//!
//! let material = IsotropicMaterial::steel();
//! let form = ElasticityForm::new_2d(material, PlaneAssumption::PlaneStress);
//! let space = VectorH1Space::new(PolynomialDegree::P1, 2);
//! let stiffness = sonium_fem::assembly::assemble_bilinear(&form, &space, &space, &mesh, 1);
//! ```

mod forms;
mod materials;
pub mod torsion;
pub mod viscoelastic_freq;

pub use forms::*;
pub use materials::*;
pub use torsion::*;
pub use viscoelastic_freq::{
    ComplexModulusModel, ConstantLossModulus, ModulusPoint, PronyBranch, PronyComplexModulus,
    TabulatedComplexModulus, ViscoelasticError,
};
