//! Magnetostatics for loudspeaker motor design
//!
//! Reduced scalar magnetic potential formulation:
//!
//! ```text
//! div(mu * grad(phi)) = div(B_r)
//! B = -mu * grad(phi) + B_r
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use sonium_fem::physics::electromagnetics::magnetostatics::*;
//! use sonium_fem::spaces::H1Space;
//! use sonium_fem::assembly::assemble_bilinear;
//!
//! // Define regions
//! let air = MagneticMaterial::air();
//! let iron = MagneticMaterial::iron(5000.0);
//! let magnet = MagneticMaterial::neodymium(1.3, [0.0, 0.0, 1.0]);
//!
//! // Assemble per-region stiffness (permeability-weighted Laplacian)
//! let form_air = PermeabilityForm::new(&air);
//! let form_iron = PermeabilityForm::new(&iron);
//! let k_air = assemble_bilinear(&form_air, &space, &space, &mesh_air, 1);
//!
//! // Assemble magnet source term
//! let source = MagnetSourceForm::new(&magnet);
//! let rhs = sonium_fem::assembly::assemble_linear(&source, &space, &mesh_magnet, 1);
//!
//! // Solve for phi, then compute B and BL(x)
//! ```

mod forms;
mod materials;
mod nonlinear;

pub use forms::*;
pub use materials::*;
pub use nonlinear::*;

#[cfg(feature = "cuda")]
mod device_field;
#[cfg(feature = "cuda")]
#[doc(inline)]
pub use device_field::CudaMagneticCentroidField;
