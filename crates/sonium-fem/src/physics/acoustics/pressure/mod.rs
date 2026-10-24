//! Pressure acoustics physics module
//!
//! Full-featured acoustic simulation supporting:
//!
//! - **Frequency domain**: Helmholtz equation `∇²p + k²p = f` with efficient
//!   frequency-sweep assembly via [`PrebakedAssembler`](crate::assembly::PrebakedAssembler).
//! - **Time domain**: Scalar wave equation `(1/c²)p̈ − ∇²p = f` with implicit
//!   Newmark-β and explicit leapfrog integrators.
//! - **Viscothermal losses**: Equivalent fluid model and thermal BLI for
//!   microacoustic analysis.
//! - **Boundary conditions**: Wall, impedance (porous layers), waveguide ports
//!   with multimode expansion, radiation (Sommerfeld, BGT), Floquet periodic.
//! - **Sources**: Normal velocity, acceleration, displacement, pressure
//!   (Dirichlet), and monopole point sources.
//! - **Post-processing**: Particle velocity, acoustic intensity, SPL,
//!   transmission loss, absorption coefficient.
//!
//! # Usage
//!
//! ```ignore
//! use sonium_fem::physics::acoustics::pressure::{
//!     HelmholtzForm, build_acoustics_assembler, AcousticsAssemblerExt,
//! };
//! use sonium_fem::basis::PolynomialDegree;
//! use num_complex::Complex64;
//!
//! let mesh = sonium_fem::mesh::unit_square_triangles(10);
//! let assembler = build_acoustics_assembler(&mesh, PolynomialDegree::P1);
//! let k = Complex64::new(5.0, 0.0);
//! let system = assembler.assemble_helmholtz(k, &Default::default());
//! ```

// Core Helmholtz assembly
mod assembler;
pub mod device_boundary;
pub mod device_hex;
#[cfg(feature = "cuda")]
pub mod device_hex_solve;
pub mod device_tet;
#[cfg(feature = "cuda")]
pub mod device_tet_solve;
mod forms;

// Boundary conditions
pub mod impedance;
pub mod port;
pub mod radiation;
pub mod resolved;

// Axisymmetric formulation
pub mod axisymmetric;

// Source terms
pub mod sources;

// Viscothermal losses
mod viscothermal;

// Time-domain solvers
pub mod time_domain;

// Post-processing
pub mod postprocess;

// Sound power and noise metrics
pub mod sound_metrics;

// Room acoustics (ray tracing, diffusion, metrics)
pub mod room;

pub use assembler::*;
pub use forms::*;
pub use viscothermal::*;

// Helmholtz-specific solvers
pub mod neural_multigrid;
pub mod schwarz_pml;
pub mod waveholtz;
