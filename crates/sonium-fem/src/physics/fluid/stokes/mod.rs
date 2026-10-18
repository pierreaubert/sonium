//! Stokes flow physics module
//!
//! Solves the steady-state Stokes equations for incompressible viscous flow:
//!
//! ```text
//!   -mu * laplacian(u) + grad(p) = f    (momentum)
//!                        div(u)  = 0    (incompressibility)
//! ```
//!
//! Weak form (saddle-point):
//!
//! ```text
//! [A   B ] [u]   [f]
//! [B^T 0 ] [p] = [0]
//!
//! where A: mu * integral(grad(u) : grad(v))  (viscous, on VectorH1)
//!       B: -integral(p * div(v))              (pressure-velocity coupling)
//! ```
//!
//! Uses Taylor-Hood P2-P1 elements (inf-sup stable, no stabilization needed).

pub mod drag_shape_opt;
pub mod forms;
pub mod materials;
pub mod navier_stokes;
pub mod stabilisation;
pub mod time_stepping;
pub mod traction;
pub mod turbulence;
pub mod workspace;
