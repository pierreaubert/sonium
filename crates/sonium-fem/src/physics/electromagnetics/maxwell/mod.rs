//! Maxwell electromagnetics physics module
//!
//! Solves the time-harmonic Maxwell curl-curl equation using H(curl)-conforming
//! Nédélec edge elements:
//!
//! ```text
//! curl((1/mu) curl E) - omega^2 eps E = -i omega J
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use sonium_fem::physics::electromagnetics::maxwell::{MaxwellForm, CurlCurlForm, EdgeMassForm, EmMaterial};
//! use sonium_fem::spaces::HcurlSpace;
//!
//! let material = EmMaterial::vacuum();
//! let form = MaxwellForm::new(&material, omega);
//! let space = HcurlSpace::new();
//! let system = sonium_fem::assembly::assemble_bilinear(&form, &space, &space, &mesh, 2);
//! ```

mod forms;

#[cfg(feature = "cuda")]
pub mod device;

pub use forms::*;
