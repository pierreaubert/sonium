//! Heat transfer physics module
//!
//! Supports conduction, convection, radiation, and advection-diffusion:
//!
//! ```text
//! rho*c * dT/dt + rho*c * u.grad(T) = div(k * grad(T)) + Q
//! ```
//!
//! Boundary conditions:
//! - Prescribed temperature (Dirichlet)
//! - Heat flux (Neumann): `q = q_prescribed`
//! - Convection (Robin): `q = h * (T - T_inf)`
//! - Radiation (nonlinear): `q = eps * sigma * (T^4 - T_inf^4)`

mod forms;
mod materials;
pub mod time_stepping;

pub use forms::*;
pub use materials::*;
pub use time_stepping::*;

#[cfg(feature = "cuda")]
mod device_time;
#[cfg(feature = "cuda")]
pub use device_time::CudaThermalTimeStepper;
