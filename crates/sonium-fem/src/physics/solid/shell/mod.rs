//! Flat triangular shell elements
//!
//! Combines CST membrane (in-plane) with plate bending (out-of-plane)
//! for thin shell analysis. 6 DOFs per node: `(u, v, w, theta_x, theta_y, theta_z)`.
//!
//! Used for loudspeaker cone, surround, and dust cap modeling.

pub mod crash;
pub mod dkt;
pub mod elastoplastic;
mod element;
pub mod explicit;
pub mod layered;
pub mod mitc4;
pub mod nonlinear;
pub mod nonlinear_quad;
pub mod probe;
pub mod revolution;
mod section;
pub mod thickness;

pub use elastoplastic::ElastoplasticShellForm;
pub use element::*;
pub use layered::*;
pub use mitc4::*;
pub use nonlinear::CorotationalShellForm;
pub use nonlinear_quad::CorotationalQuadShellForm;
pub use revolution::*;
pub use section::*;
pub use thickness::*;
