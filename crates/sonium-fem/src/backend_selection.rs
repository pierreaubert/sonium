//! Backend and preconditioner selection policy for Helmholtz sweeps.
//!
//! This module owns the reusable string-to-strategy contracts extracted from
//! the room runner (`fem_runner::fem_preconditioner_kind`, `fem_runner::gpu`):
//! which shifted-Laplacian preconditioner a CPU solve builds, which
//! GPU-resident smoother a device sweep requests, and which precision mode a
//! sweep runs in — plus the per-backend resolution rules (`auto` means AMG on
//! CUDA but Chebyshev-8 on Metal, which has no AMG yet).
//!
//! Assembly, iteration, device residency, and run orchestration stay with the
//! callers. These types only decide *what* to build; a wrong string fails at
//! parse time, never halfway through a sweep.

use std::fmt;
use std::str::FromStr;

/// FEM preconditioner strategy for the shifted-Laplacian system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FemPreconditionerKind {
    /// Global shifted-Laplacian ILUT. Strong convergence, mostly serial apply.
    Ilut,
    /// Block Jacobi with independent local ILUT solves. Parallel apply scales
    /// with the number of partitions; iteration count is typically a small
    /// constant factor above global ILUT, which is more than recovered by the
    /// per-iteration speed-up.
    #[default]
    BlockJacobi,
    /// Overlapping additive Schwarz with local ILU solves.
    AdditiveSchwarz,
    /// Optimized Schwarz with absorbing Robin transmission on artificial interfaces.
    Oras,
    /// Optimized Schwarz with a second-order HABC-style interface correction.
    Habc,
    /// Double-sweep multiplicative Schwarz over ordered coordinate layers
    /// with absorbing Robin transmission (sweeping-style preconditioner
    /// for high-wavenumber Helmholtz).
    SweepingSchwarz,
    /// ILU(0) triangular solve with level scheduling.
    IluColoring,
    /// Approximate ILU(0) triangular solve using fixed-point sweeps.
    IluFixedPoint,
    /// Algebraic multigrid V-cycle configured for parallel smoothing/coarsening.
    Amg,
}
impl FemPreconditionerKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ilut => "ilut",
            Self::BlockJacobi => "block-jacobi",
            Self::AdditiveSchwarz => "additive-schwarz",
            Self::Oras => "oras",
            Self::Habc => "habc",
            Self::SweepingSchwarz => "sweeping-schwarz",
            Self::IluColoring => "ilu-coloring",
            Self::IluFixedPoint => "ilu-fixedpoint",
            Self::Amg => "amg",
        }
    }

    pub fn valid_values() -> &'static str {
        "ilut, block-jacobi, additive-schwarz, oras, habc, sweeping-schwarz, ilu-coloring, ilu-fixedpoint, amg"
    }
}

impl fmt::Display for FemPreconditionerKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for FemPreconditionerKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let normalized = value.trim().to_ascii_lowercase().replace('_', "-");
        match normalized.as_str() {
            "ilut" => Ok(Self::Ilut),
            "block-jacobi" | "blockjacobi" | "bj" => Ok(Self::BlockJacobi),
            "additive-schwarz" | "schwarz" | "as" => Ok(Self::AdditiveSchwarz),
            "oras" | "optimized-schwarz" | "optimised-schwarz" | "optimized-ras"
            | "optimised-ras" => Ok(Self::Oras),
            "habc" | "habc-schwarz" | "second-order-schwarz" => Ok(Self::Habc),
            "sweeping-schwarz" | "sweeping" | "sweep" => Ok(Self::SweepingSchwarz),
            "ilu-coloring" | "ilucoloring" | "coloring" => Ok(Self::IluColoring),
            "ilu-fixedpoint" | "ilu-fixed-point" | "ilufixedpoint" | "fixedpoint"
            | "fixed-point" => Ok(Self::IluFixedPoint),
            "amg" => Ok(Self::Amg),
            _ => Err(format!(
                "unknown FEM preconditioner '{value}'. Valid values: {}",
                Self::valid_values()
            )),
        }
    }
}
/// `Chebyshev { degree }` applies a Chebyshev polynomial to `D^-1 P`.
/// It is the GPU-resident smoother used as the first AMG-compatible
/// stepping stone: stronger than damped Jacobi for elliptic/shifted
/// Laplacian spectra without serial triangular solves or host round trips.
///
/// `Amg` builds a full AMG hierarchy for P and uploads all levels to
/// the backend. The apply path uses backend-resident Jacobi/Chebyshev
/// smoothing and coarse-grid cycles, so Krylov iterations stay on CUDA.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuPrecond {
    Auto,
    PointJacobi,
    PolynomialJacobi { degree: usize },
    Chebyshev { degree: usize },
    Amg,
}

impl GpuPrecond {
    pub fn parse(spec: &str) -> Result<Self, String> {
        let spec = spec.trim().to_ascii_lowercase().replace('_', "-");
        if let Some(rest) = spec.strip_prefix("poly-jacobi") {
            let degree = if rest.is_empty() {
                4
            } else if let Some(d_str) = rest.strip_prefix('-') {
                d_str
                    .parse::<usize>()
                    .map_err(|_| format!("invalid degree in '{spec}'; expected `poly-jacobi-N`"))?
            } else {
                return Err(format!("unrecognised gpu precond '{spec}'"));
            };
            if degree == 0 {
                Ok(Self::PointJacobi)
            } else {
                Ok(Self::PolynomialJacobi { degree })
            }
        } else if let Some(rest) = spec.strip_prefix("chebyshev") {
            let degree = if rest.is_empty() {
                8
            } else if let Some(d_str) = rest.strip_prefix('-') {
                d_str
                    .parse::<usize>()
                    .map_err(|_| format!("invalid degree in '{spec}'; expected `chebyshev-N`"))?
            } else {
                return Err(format!("unrecognised gpu precond '{spec}'"));
            };
            if degree == 0 {
                Ok(Self::PointJacobi)
            } else {
                Ok(Self::Chebyshev { degree })
            }
        } else if matches!(
            spec.as_str(),
            "point-jacobi" | "pointjacobi" | "diagonal" | "jacobi"
        ) {
            Ok(Self::PointJacobi)
        } else if matches!(spec.as_str(), "auto" | "default") {
            Ok(Self::Auto)
        } else if matches!(spec.as_str(), "amg" | "gpu-amg" | "backend-amg") {
            Ok(Self::Amg)
        } else {
            Err(format!(
                "unknown gpu precond '{spec}'. Valid: auto, point-jacobi, poly-jacobi[-N], chebyshev[-N], amg"
            ))
        }
    }

    pub fn resolve_for_cuda(self) -> Self {
        match self {
            Self::Auto => Self::Amg,
            other => other,
        }
    }

    pub fn resolve_for_metal(self) -> Result<Self, String> {
        match self {
            Self::Auto => Ok(Self::Chebyshev { degree: 8 }),
            Self::Amg => Err(
                "Metal FEM prototype does not support AMG yet; use --gpu-precond auto, chebyshev[-N], poly-jacobi[-N], or point-jacobi"
                    .to_string(),
            ),
            other => Ok(other),
        }
    }

    pub fn label(self) -> String {
        match self {
            Self::Auto => "auto".to_string(),
            Self::PointJacobi => "PointJacobi(P)".to_string(),
            Self::PolynomialJacobi { degree } => {
                format!("PolyJacobi(P,deg={degree})")
            }
            Self::Chebyshev { degree } => {
                format!("Chebyshev(P,deg={degree},ratio={CHEBYSHEV_EIGENVALUE_RATIO:.0})")
            }
            Self::Amg => "AMG(P,Chebyshev smoother)".to_string(),
        }
    }
}

/// PETSc/HYPRE-style lower-bound ratio for the Chebyshev smoother:
/// λ_min = λ_max / ratio. This is deliberately conservative for
/// shifted-Laplacian spectra. Shared with device runners, which must use
/// the same ratio the policy labels advertise.
pub const CHEBYSHEV_EIGENVALUE_RATIO: f32 = 30.0;

/// GPU precision strategy.
///
/// On workstation cards with a brutal f64:f32 throughput ratio the
/// mixed-precision IR mode still pays the f64 penalty on the outer FGMRES;
/// pure-f32 mode drops the outer refinement and runs Complex32 GMRES with an
/// accuracy floor (~1e-7 relative) below mesh discretisation error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuPrecision {
    /// Complex32 throughout. Best for RTX / non-datacenter cards.
    F32Pure,
    /// Complex64 outer FGMRES, Complex32 inner GMRES (Carson-Higham
    /// IR). Accuracy floor at f64 level (`outer_tolerance = 1e-6`).
    /// Best on H100/A100 where f64 throughput is full-rate.
    F64Outer,
}

impl GpuPrecision {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f32" | "f32-pure" | "single" | "fp32" => Ok(Self::F32Pure),
            "f64-outer" | "ir" | "mixed" | "f64" => Ok(Self::F64Outer),
            other => Err(format!(
                "unknown gpu precision '{other}'. Valid: f32, f64-outer"
            )),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::F32Pure => "f32-pure",
            Self::F64Outer => "f64-outer (IR)",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preconditioner_kind_parses_all_aliases() {
        let cases = [
            ("ilut", FemPreconditionerKind::Ilut),
            ("block-jacobi", FemPreconditionerKind::BlockJacobi),
            ("blockjacobi", FemPreconditionerKind::BlockJacobi),
            ("bj", FemPreconditionerKind::BlockJacobi),
            ("additive-schwarz", FemPreconditionerKind::AdditiveSchwarz),
            ("schwarz", FemPreconditionerKind::AdditiveSchwarz),
            ("as", FemPreconditionerKind::AdditiveSchwarz),
            ("oras", FemPreconditionerKind::Oras),
            ("optimized-schwarz", FemPreconditionerKind::Oras),
            ("optimised-schwarz", FemPreconditionerKind::Oras),
            ("habc", FemPreconditionerKind::Habc),
            ("habc-schwarz", FemPreconditionerKind::Habc),
            ("second-order-schwarz", FemPreconditionerKind::Habc),
            ("sweeping-schwarz", FemPreconditionerKind::SweepingSchwarz),
            ("sweeping", FemPreconditionerKind::SweepingSchwarz),
            ("sweep", FemPreconditionerKind::SweepingSchwarz),
            ("ilu-coloring", FemPreconditionerKind::IluColoring),
            ("coloring", FemPreconditionerKind::IluColoring),
            ("ilu-fixedpoint", FemPreconditionerKind::IluFixedPoint),
            ("ilu-fixed-point", FemPreconditionerKind::IluFixedPoint),
            ("fixedpoint", FemPreconditionerKind::IluFixedPoint),
            ("fixed-point", FemPreconditionerKind::IluFixedPoint),
            ("amg", FemPreconditionerKind::Amg),
        ];
        for (input, expected) in cases {
            assert_eq!(input.parse(), Ok(expected), "input {input}");
            assert_eq!(format!("{expected}"), expected.as_str());
        }
        // Case, whitespace, and underscore normalization.
        assert_eq!(
            "  BLOCK_JACOBI ".parse(),
            Ok(FemPreconditionerKind::BlockJacobi)
        );
        assert!("multigrid".parse::<FemPreconditionerKind>().is_err());
        assert_eq!(
            FemPreconditionerKind::default(),
            FemPreconditionerKind::BlockJacobi
        );
        assert!(FemPreconditionerKind::valid_values().contains("block-jacobi"));
    }

    #[test]
    fn gpu_precond_parses_and_resolves_per_backend() {
        assert_eq!(GpuPrecond::parse("auto"), Ok(GpuPrecond::Auto));
        assert_eq!(
            GpuPrecond::parse("point-jacobi"),
            Ok(GpuPrecond::PointJacobi)
        );
        assert_eq!(GpuPrecond::parse("jacobi"), Ok(GpuPrecond::PointJacobi));
        assert_eq!(
            GpuPrecond::parse("poly-jacobi"),
            Ok(GpuPrecond::PolynomialJacobi { degree: 4 })
        );
        assert_eq!(
            GpuPrecond::parse("poly-jacobi-6"),
            Ok(GpuPrecond::PolynomialJacobi { degree: 6 })
        );
        assert_eq!(
            GpuPrecond::parse("chebyshev"),
            Ok(GpuPrecond::Chebyshev { degree: 8 })
        );
        assert_eq!(
            GpuPrecond::parse("chebyshev-12"),
            Ok(GpuPrecond::Chebyshev { degree: 12 })
        );
        assert_eq!(GpuPrecond::parse("amg"), Ok(GpuPrecond::Amg));
        assert_eq!(GpuPrecond::parse("gpu-amg"), Ok(GpuPrecond::Amg));
        assert!(GpuPrecond::parse("chebyshev-fast").is_err());
        assert!(
            GpuPrecond::parse("poly-jacobi-0")
                .is_ok_and(|precond| { matches!(precond, GpuPrecond::PointJacobi) })
        );

        // CUDA: auto means AMG; explicit choices are untouched.
        assert_eq!(GpuPrecond::Auto.resolve_for_cuda(), GpuPrecond::Amg);
        assert_eq!(
            GpuPrecond::Chebyshev { degree: 12 }.resolve_for_cuda(),
            GpuPrecond::Chebyshev { degree: 12 }
        );
        // Metal: auto means Chebyshev-8; AMG is rejected; others pass through.
        assert_eq!(
            GpuPrecond::Auto.resolve_for_metal(),
            Ok(GpuPrecond::Chebyshev { degree: 8 })
        );
        assert!(GpuPrecond::Amg.resolve_for_metal().is_err());
        assert_eq!(
            GpuPrecond::PointJacobi.resolve_for_metal(),
            Ok(GpuPrecond::PointJacobi)
        );
        // Labels advertise the shared ratio and degree.
        assert!(
            GpuPrecond::Chebyshev { degree: 8 }
                .label()
                .contains(&format!("{:.0}", CHEBYSHEV_EIGENVALUE_RATIO))
        );
        assert!(
            GpuPrecond::PolynomialJacobi { degree: 6 }
                .label()
                .contains("deg=6")
        );
    }
}

/// Working precision for the per-Picard-iteration linear solve.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum LinearSolverKind {
    /// Standard f64 GMRES via `gmres_backend_preconditioned`. Default.
    #[default]
    F64,
    /// Mixed-precision FGMRES-IR (Carson-Higham 2018) — outer FGMRES in
    /// f64, inner Krylov + preconditioner in f32. Strict superset of
    /// f64 accuracy provided `κ(A) ≲ 10¹⁴`. The Metal-friendly path
    /// since Apple Silicon Metal has no `double` support.
    /// Currently uses [`BackendPreconditionerKind::PolynomialJacobi`]
    /// for the inner solve regardless of the configured kind — an f32
    /// Chebyshev mirror is a Stage 4 follow-up.
    MixedF32F64,
}

/// CPU preconditioner for the per-Picard-iteration linear solve.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum LinearSolverCpuPreconditionerKind {
    /// Legacy monolithic ILU + GMRES path.
    #[default]
    Ilu,
    /// Block Jacobi/Schur preconditioner:
    /// diagonal velocity-block inverse plus pressure mass Schur
    /// approximation `S ≈ M_p / mu_eff`.
    JacobiSchur,
    /// Upper-triangular saddle-point preconditioner with ILUT on both the
    /// velocity block and the external pressure-mass Schur approximation.
    /// Intended for large 3D Taylor-Hood systems where diagonal scaling
    /// leaves an impractically large Krylov count.
    IlutSchur,
    /// Upper-triangular saddle-point preconditioner with ILU(0) on the
    /// velocity block and ILUT on the external Schur approximation.
    /// Middle ground between JacobiSchur (cheap setup, many iterations)
    /// and IlutSchur (expensive setup, few iterations).
    IluSchur,
}

/// Selects which GPU-friendly preconditioner the backend GMRES path
/// uses. Both variants are pure SpMV + BLAS-1 — no triangular solve,
/// no host round-trip per apply.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum BackendPreconditionerKind {
    /// Truncated Neumann series of `D⁻¹A` ("polynomial Jacobi"). Robust
    /// for indefinite saddle-point systems with appropriate damping.
    /// Configured by `linear_solver_polynomial_degree` and
    /// `linear_solver_polynomial_damping`.
    #[default]
    PolynomialJacobi,
    /// Chebyshev polynomial of `D⁻¹A` on `[λ_max/ratio, λ_max]`.
    /// Optimal min-max damping for SPD or near-SPD systems. `λ_max`
    /// is estimated automatically by 20 power iterations at construction.
    /// Configured by `linear_solver_chebyshev_degree` and
    /// `linear_solver_chebyshev_eigenvalue_ratio`.
    Chebyshev,
    /// Upper-triangular saddle-point preconditioner using diagonal
    /// velocity Jacobi and pressure-mass Schur diagonal inverse.
    JacobiSchur,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum BackendPressureSchurPreconditionerKind {
    #[default]
    Polynomial,
    Chebyshev,
    Amg,
}

/// Backend selection for the per-Picard-iteration linear solve.
///
/// Direct solver (`use_direct_solver = true`) ignores this — the sparse
/// LU is CPU-only. The choice only affects the iterative GMRES path.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum LinearSolverBackend {
    /// Best available backend at runtime. Selection rule, evaluated on
    /// the first solve call:
    ///
    /// - macOS with the `metal` feature compiled in → [`Self::Metal`]
    ///   (provided `MetalBackend::new()` succeeds).
    /// - Linux/Windows with `cuda` compiled in → [`Self::Cuda`]
    ///   (provided `CudaBackend::new_default()` succeeds).
    /// - Any other configuration → [`Self::Cpu`] (legacy ILU + GMRES).
    ///
    /// This is the default so that `cargo run --features metal` on macOS
    /// "just works" without further CLI plumbing, and the same source
    /// builds on Linux+CUDA with the analogous behaviour.
    #[default]
    Auto,
    /// CPU path: in-house BLAS-1 + ILU preconditioner via
    /// `sonium_solvers::gmres_preconditioned_with_guess`.
    Cpu,
    /// NVIDIA CUDA via `sonium_compute::cuda::CudaBackend`. Requires the
    /// `cuda` feature (cuBLAS / cuSPARSE via cudarc) on `sonium-compute`,
    /// plus a CUDA driver at runtime. Falls back to `Cpu` with a warning
    /// when the feature is not enabled.
    Cuda,
    /// Apple Metal via `sonium_compute::metal::MetalBackend`. Requires
    /// the `metal` feature; falls back to `Cpu` with a warning when not
    /// built.
    Metal,
}

impl LinearSolverBackend {
    /// Resolve [`Self::Auto`] to a concrete backend by inspecting the
    /// host OS and compiled features.
    ///
    /// Selection order:
    /// 1. macOS + `metal` feature  → [`Self::Metal`]
    /// 2. macOS + `cuda` (no metal) → [`Self::Cuda`] (exotic)
    /// 3. non-macOS + `cuda` feature → [`Self::Cuda`]
    /// 4. otherwise → [`Self::Cpu`]
    ///
    /// Pure cfg dispatch — no runtime device probing here. If the chosen
    /// backend's `new()` later fails, the dispatcher falls back to CPU
    /// with a warning at solve time.
    pub(crate) fn resolve_auto() -> Self {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            Self::Metal
        }
        #[cfg(all(feature = "cuda", target_os = "macos", not(feature = "metal")))]
        {
            Self::Cuda
        }
        #[cfg(all(feature = "cuda", not(target_os = "macos")))]
        {
            Self::Cuda
        }
        #[cfg(not(any(all(feature = "metal", target_os = "macos"), feature = "cuda")))]
        {
            Self::Cpu
        }
    }
}

#[cfg(test)]
mod linear_solver_selection_tests {
    use super::*;

    #[test]
    fn auto_resolves_to_a_concrete_backend() {
        assert_ne!(
            LinearSolverBackend::resolve_auto(),
            LinearSolverBackend::Auto
        );
    }

    #[test]
    fn selection_defaults_match_legacy_config() {
        assert_eq!(LinearSolverKind::default(), LinearSolverKind::F64);
        assert_eq!(
            LinearSolverCpuPreconditionerKind::default(),
            LinearSolverCpuPreconditionerKind::Ilu
        );
        assert_eq!(
            BackendPreconditionerKind::default(),
            BackendPreconditionerKind::PolynomialJacobi
        );
        assert_eq!(
            BackendPressureSchurPreconditionerKind::default(),
            BackendPressureSchurPreconditionerKind::Polynomial
        );
        assert_eq!(LinearSolverBackend::default(), LinearSolverBackend::Auto);
    }
}

#[cfg(test)]
#[test]
fn gpu_precision_parses_and_labels() {
    for input in ["f32", "f32-pure", "single", "fp32"] {
        assert_eq!(GpuPrecision::parse(input), Ok(GpuPrecision::F32Pure));
    }
    for input in ["f64-outer", "ir", "mixed", "f64"] {
        assert_eq!(GpuPrecision::parse(input), Ok(GpuPrecision::F64Outer));
    }
    assert!(GpuPrecision::parse("f16").is_err());
    assert_eq!(GpuPrecision::F32Pure.label(), "f32-pure");
    assert_eq!(GpuPrecision::F64Outer.label(), "f64-outer (IR)");
}
