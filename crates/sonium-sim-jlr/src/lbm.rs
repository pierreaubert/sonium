//! D3Q19 lattice Boltzmann core with BGK collision and Smagorinsky LES.
//!
//! Contains the single-grid solver used by every JLR arm: second-order
//! equilibrium (paper Eq. 2.4), Bhatnagar–Gross–Krook collision
//! (Eq. 2.2–2.3), halfway bounce-back walls, optional Smagorinsky
//! sub-grid viscosity (Eq. 2.6), Guo body forcing, equilibrium velocity
//! inlets, and pressure outlets. All quantities are lattice units
//! (Δx = Δt = 1, cs² = 1/3) inside this module; physical conversion lives
//! in [`crate::case`]. The hot path is the fused collide pass followed by
//! the pull-streaming pass, both rayon-parallel over cells.

// Rust guideline compliant 2026-02-21

use rayon::prelude::*;

/// Number of discrete velocities in the D3Q19 model.
pub const Q: usize = 19;

/// Lattice speed of sound squared, cs² = 1/3 for D3Q19.
pub const CS2: f64 = 1.0 / 3.0;

/// Discrete velocity x-components, rest particle last.
pub const CX: [i32; Q] = [1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0, 0];
/// Discrete velocity y-components, rest particle last.
pub const CY: [i32; Q] = [0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0];
/// Discrete velocity z-components, rest particle last.
pub const CZ: [i32; Q] = [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1, 0];
/// Quadrature weights: 1/3 rest, 1/18 axial, 1/36 diagonal.
pub const W: [f64; Q] = [
    1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 3.0,
];
/// Opposite direction index for each velocity.
pub const OPP: [usize; Q] = [
    1, 0, 3, 2, 5, 4, 9, 8, 7, 6, 13, 12, 11, 10, 17, 16, 15, 14, 18,
];

/// Role of one lattice cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Cell {
    /// Ordinary fluid cell.
    Fluid = 0,
    /// Solid cell, halfway bounce-back wall.
    Solid = 1,
    /// Equilibrium velocity inlet cell.
    Inlet = 2,
    /// Pressure outlet cell at reference density.
    Outlet = 3,
}

/// Static configuration of one lattice.
#[derive(Debug, Clone)]
pub struct LatticeConfig {
    /// Cells along x.
    pub nx: usize,
    /// Cells along y.
    pub ny: usize,
    /// Cells along z.
    pub nz: usize,
    /// Molecular relaxation time τ (paper: 0.500005 on the base grid).
    pub tau: f64,
    /// Smagorinsky constant Cs, `None` disables the LES model.
    pub smagorinsky: Option<f64>,
    /// Guo body force density in lattice units, usually gravity or a
    /// pressure-gradient surrogate for periodic verification cases.
    pub force: [f64; 3],
    /// Periodic wrap per axis; non-periodic edges act as walls.
    pub periodic: [bool; 3],
    /// Inlet velocity in lattice units, scaled by the step's ramp factor.
    pub inlet_velocity: [f64; 3],
    /// Regularised collision (Latt–Chopard): project the
    /// non-equilibrium part onto its second-order Hermite basis before
    /// relaxing. Filters the ghost modes that make SRT-BGK unstable
    /// at tau close to 0.5; off by default, on for JLR production.
    pub regularized: bool,
    /// Central-moment (cascaded) collision: relax second-order central
    /// moments with the viscous (LES-augmented) rate, the trace with
    /// `bulk_tau`, and all higher central moments fully to equilibrium
    /// every step. This is the collision family production LBM codes
    /// use at tau close to 0.5; measured necessary here (SRT and
    /// regularised BGK both blow up in the JLR at the paper's tau).
    pub central_moment: bool,
    /// Relaxation time for the trace (bulk) central moment.
    pub bulk_tau: f64,
    /// Relaxation rate for higher-order (ghost) central moments.
    pub ghost_rate: f64,
    /// Direction from an outlet cell toward the domain interior, used by
    /// the zero-gradient pressure outlet.
    pub outlet_normal: [i32; 3],
    /// Imposed outlet velocity in lattice units (fixed-flow outlet:
    /// the experiment's pump imposes the same Q at both ends). Used by
    /// the non-equilibrium-extrapolation outlet with local density.
    pub outlet_velocity: [f64; 3],
}

/// Molecular lattice viscosity for a relaxation time, ν = cs²(τ − ½).
///
/// # Examples
/// ```rust
/// use sonium_sim_jlr::lbm::viscosity_lattice;
/// assert!((viscosity_lattice(1.0) - 1.0 / 6.0).abs() < 1e-15);
/// ```
#[must_use]
pub fn viscosity_lattice(tau: f64) -> f64 {
    CS2 * (tau - 0.5)
}

/// Relaxation time for a lattice viscosity, τ = ν/cs² + ½.
///
/// # Panics
/// Panics if `nu` is not positive and finite.
#[must_use]
pub fn tau_for_viscosity(nu: f64) -> f64 {
    assert!(nu.is_finite() && nu > 0.0, "viscosity must be positive, got {nu}");
    nu / CS2 + 0.5
}

/// Second-order equilibrium distribution for one direction.
///
/// # Examples
/// ```rust
/// use sonium_sim_jlr::lbm::equilibrium;
/// assert!((equilibrium(18, 1.0, [0.0; 3]) - 1.0 / 3.0).abs() < 1e-15);
/// ```
#[must_use]
pub fn equilibrium(i: usize, rho: f64, u: [f64; 3]) -> f64 {
    let c = [CX[i] as f64, CY[i] as f64, CZ[i] as f64];
    let cu = c[0] * u[0] + c[1] * u[1] + c[2] * u[2];
    let u2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    W[i] * rho * (1.0 + cu / CS2 + cu * cu / (2.0 * CS2 * CS2) - u2 / (2.0 * CS2))
}

/// Moment exponents (a, b, c) of the D3Q19 moment basis.
///
/// The basis is every independent monomial the lattice supports:
/// density, momentum, the six second-order moments, the seven
/// third-order moments, and the three fourth-order pair moments
/// ((1,1,1) is aliased on D3Q19 and cannot be in the basis).
/// Invertibility is not asserted from literature; it is
/// verified numerically by [`moment_inverse`] and a round-trip test.
pub const MOMENT_EXP: [(u32, u32, u32); Q] = [
    (0, 0, 0),
    (1, 0, 0), (0, 1, 0), (0, 0, 1),
    (2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1),
    (2, 1, 0), (2, 0, 1), (1, 2, 0), (0, 2, 1), (1, 0, 2), (0, 1, 2),
    (2, 2, 0), (0, 2, 2), (2, 0, 2),
];

/// Raw moment matrix M[k][i] = cx^a · cy^b · cz^c for basis moment k.
#[must_use]
pub fn moment_matrix() -> [[f64; Q]; Q] {
    let mut m = [[0.0; Q]; Q];
    for (k, &(a, b, c)) in MOMENT_EXP.iter().enumerate() {
        for i in 0..Q {
            m[k][i] = (CX[i] as f64).powi(a as i32)
                * (CY[i] as f64).powi(b as i32)
                * (CZ[i] as f64).powi(c as i32);
        }
    }
    m
}

/// Inverse of [`moment_matrix`], by Gauss–Jordan with partial pivoting.
///
/// # Panics
/// Panics if the basis is singular (a programming error in MOMENT_EXP).
#[must_use]
pub fn moment_inverse() -> [[f64; Q]; Q] {
    let m = moment_matrix();
    let mut aug = [[0.0; 2 * Q]; Q];
    for r in 0..Q {
        for c in 0..Q {
            aug[r][c] = m[r][c];
        }
        aug[r][Q + r] = 1.0;
    }
    for col in 0..Q {
        let piv = (col..Q)
            .max_by(|&a, &b| aug[a][col].abs().partial_cmp(&aug[b][col].abs()).unwrap())
            .unwrap();
        assert!(aug[piv][col].abs() > 1e-12, "moment basis is singular at column {col}");
        aug.swap(col, piv);
        let d = aug[col][col];
        for v in &mut aug[col] {
            *v /= d;
        }
        let pivot_row = aug[col];
        for (r, row) in aug.iter_mut().enumerate() {
            if r != col && row[col] != 0.0 {
                let f = row[col];
                for (c, v) in row.iter_mut().enumerate() {
                    *v -= f * pivot_row[c];
                }
            }
        }
    }
    let mut inv = [[0.0; Q]; Q];
    for r in 0..Q {
        for c in 0..Q {
            inv[r][c] = aug[r][Q + c];
        }
    }
    inv
}

/// Index of a moment exponent in [`MOMENT_EXP`], if present.
#[must_use]
pub fn moment_index(a: u32, b: u32, c: u32) -> Option<usize> {
    MOMENT_EXP.iter().position(|&e| e == (a, b, c))
}

fn binom(n: u32, k: u32) -> f64 {
    if k > n {
        return 0.0;
    }
    let mut r = 1.0;
    for i in 0..k {
        r *= (n - i) as f64 / (i + 1) as f64;
    }
    r
}

impl LatticeConfig {
    /// Builds a laminar, non-periodic configuration with no forcing.
    ///
    /// # Examples
    /// ```rust
    /// use sonium_sim_jlr::lbm::LatticeConfig;
    /// let cfg = LatticeConfig::new(4, 4, 4, 1.0);
    /// assert_eq!(cfg.tau, 1.0);
    /// ```
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize, tau: f64) -> Self {
        Self {
            nx,
            ny,
            nz,
            tau,
            smagorinsky: None,
            force: [0.0; 3],
            periodic: [false; 3],
            inlet_velocity: [0.0; 3],
            outlet_velocity: [0.0; 3],
            regularized: false,
            central_moment: false,
            bulk_tau: 1.0,
            ghost_rate: 1.0,
            outlet_normal: [0, 0, 1],
        }
    }
}

/// One D3Q19 lattice with two distribution buffers and a collided buffer.
#[derive(Debug)]
pub struct Lattice {
    cfg: LatticeConfig,
    cells: Vec<Cell>,
    wall_velocity: Vec<[f64; 3]>,
    inlet_velocity_cells: Vec<[f64; 3]>,
    f: Vec<f64>,
    f_col: Vec<f64>,
    f_next: Vec<f64>,
    steps: u64,
}

impl Lattice {
    /// Builds a lattice initialised at rest at density 1.
    ///
    /// # Panics
    /// Panics on zero dimensions, a cell mask of the wrong length, or a
    /// relaxation time at or below the stability limit τ = ½.
    pub fn new(cfg: LatticeConfig, cells: Vec<Cell>) -> Self {
        assert!(cfg.nx > 0 && cfg.ny > 0 && cfg.nz > 0, "grid dimensions must be positive");
        assert_eq!(cells.len(), cfg.nx * cfg.ny * cfg.nz, "cell mask length mismatch");
        assert!(cfg.tau > 0.5, "tau must exceed 0.5, got {}", cfg.tau);
        let n = cells.len();
        let inlet_default = cfg.inlet_velocity;
        let mut lat = Self {
            cfg,
            cells,
            wall_velocity: vec![[0.0; 3]; n],
            inlet_velocity_cells: vec![inlet_default; n],
            f: vec![0.0; n * Q],
            f_col: vec![0.0; n * Q],
            f_next: vec![0.0; n * Q],
            steps: 0,
        };
        for idx in 0..n {
            for i in 0..Q {
                lat.f[idx * Q + i] = equilibrium(i, 1.0, [0.0; 3]);
            }
        }
        lat
    }

    /// Returns the role of one cell.
    ///
    /// # Panics
    /// Panics if `idx` is out of range.
    #[must_use]
    pub fn cell(&self, idx: usize) -> Cell {
        self.cells[idx]
    }

    /// Returns the static configuration.
    #[must_use]
    pub fn config(&self) -> &LatticeConfig {
        &self.cfg
    }

    /// Returns the number of completed steps.
    #[must_use]
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Returns the total cell count.
    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.cfg.nx * self.cfg.ny * self.cfg.nz
    }

    /// Linear index of a cell coordinate.
    #[must_use]
    pub fn index(&self, x: usize, y: usize, z: usize) -> usize {
        x + self.cfg.nx * (y + self.cfg.ny * z)
    }

    /// Sets the imposed velocity of one inlet cell (profile support).
    ///
    /// # Panics
    /// Panics if `idx` is out of range.
    pub fn set_inlet_velocity(&mut self, idx: usize, u: [f64; 3]) {
        self.inlet_velocity_cells[idx] = u;
    }

    /// Sets the velocity of a moving-wall (solid) cell for bounce-back.
    ///
    /// # Panics
    /// Panics if `idx` is out of range.
    pub fn set_wall_velocity(&mut self, idx: usize, u: [f64; 3]) {
        self.wall_velocity[idx] = u;
    }

    /// Overwrites one cell with the equilibrium at the given state.
    ///
    /// # Panics
    /// Panics if `idx` is out of range.
    pub fn set_equilibrium(&mut self, idx: usize, rho: f64, u: [f64; 3]) {
        for i in 0..Q {
            self.f[idx * Q + i] = equilibrium(i, rho, u);
        }
    }

    /// Macroscopic density and velocity of one cell (Guo half-force).
    ///
    /// # Panics
    /// Panics if `idx` is out of range.
    #[must_use]
    pub fn macroscopic(&self, idx: usize) -> (f64, [f64; 3]) {
        let mut rho = 0.0;
        let mut mom = [0.0; 3];
        for i in 0..Q {
            let fi = self.f[idx * Q + i];
            rho += fi;
            mom[0] += CX[i] as f64 * fi;
            mom[1] += CY[i] as f64 * fi;
            mom[2] += CZ[i] as f64 * fi;
        }
        let inv = 1.0 / rho;
        (
            rho,
            [
                (mom[0] + 0.5 * self.cfg.force[0]) * inv,
                (mom[1] + 0.5 * self.cfg.force[1]) * inv,
                (mom[2] + 0.5 * self.cfg.force[2]) * inv,
            ],
        )
    }

    /// Adds the current axial velocity, scaled, into an accumulator.
    ///
    /// Parallel over cells; solid cells contribute zero. Used for the
    /// running time average in case runs.
    ///
    /// # Panics
    /// Panics if the accumulator length differs from the cell count.
    pub fn accumulate_uz(&self, acc: &mut [f64], scale: f64) {
        assert_eq!(acc.len(), self.cell_count(), "accumulator length mismatch");
        let cells = &self.cells;
        acc.par_chunks_mut(4096)
            .enumerate()
            .for_each(|(chunk, slice)| {
                for (off, a) in slice.iter_mut().enumerate() {
                    let idx = chunk * 4096 + off;
                    if cells[idx] == Cell::Solid {
                        continue;
                    }
                    let mut rho = 0.0;
                    let mut mz = 0.0;
                    for (i, cz) in CZ.iter().enumerate() {
                        let fi = self.f[idx * Q + i];
                        rho += fi;
                        mz += *cz as f64 * fi;
                    }
                    *a += mz / rho * scale;
                }
            });
    }

    /// Advances one lattice step with the inlet ramp factor in `[0, 1]`.
    pub fn step(&mut self, ramp: f64) {
        self.collide();
        self.stream(ramp);
        std::mem::swap(&mut self.f, &mut self.f_next);
        self.steps += 1;
    }

    fn collide(&mut self) {
        if self.cfg.central_moment {
            self.collide_central();
            return;
        }
        let tau0 = self.cfg.tau;
        let force = self.cfg.force;
        let cs = self.cfg.smagorinsky;
        let regularized = self.cfg.regularized;
        let cells = &self.cells;
        self.f_col
            .par_chunks_mut(Q)
            .zip(self.f.par_chunks(Q))
            .enumerate()
            .for_each(|(idx, (dst, src))| {
                if cells[idx] == Cell::Solid {
                    dst.copy_from_slice(src);
                    return;
                }
                let mut rho = 0.0;
                let mut mom = [0.0; 3];
                for i in 0..Q {
                    rho += src[i];
                    mom[0] += CX[i] as f64 * src[i];
                    mom[1] += CY[i] as f64 * src[i];
                    mom[2] += CZ[i] as f64 * src[i];
                }
                let inv_rho = 1.0 / rho;
                let u = [
                    (mom[0] + 0.5 * force[0]) * inv_rho,
                    (mom[1] + 0.5 * force[1]) * inv_rho,
                    (mom[2] + 0.5 * force[2]) * inv_rho,
                ];
                let mut feq = [0.0; Q];
                for (i, f) in feq.iter_mut().enumerate() {
                    *f = equilibrium(i, rho, u);
                }
                // Non-equilibrium second moment: Smagorinsky input and,
                // for the regularised collision, the Hermite projection
                // basis (Latt & Chopard 2006).
                let mut pij = [[0.0; 3]; 3];
                if cs.is_some() || regularized {
                    for i in 0..Q {
                        let d = src[i] - feq[i];
                        let cvec = [CX[i] as f64, CY[i] as f64, CZ[i] as f64];
                        for (a, row) in pij.iter_mut().enumerate() {
                            for (b, p) in row.iter_mut().enumerate() {
                                *p += cvec[a] * cvec[b] * d;
                            }
                        }
                    }
                }
                // Smagorinsky τ_eff from the non-equilibrium second moment
                // (Kruger et al., The Lattice Boltzmann Method, ch. 9):
                // τ_eff solves τ(τ−τ0) = Cs²·sqrt(2 Π:Π) / (2 ρ cs⁴).
                let tau_eff = match cs {
                    Some(c) if c > 0.0 => {
                        let mut pp = 0.0;
                        for row in &pij {
                            for p in row {
                                pp += p * p;
                            }
                        }
                        let q = (2.0 * pp).sqrt();
                        let rhs = c * c * q / (2.0 * rho * CS2 * CS2);
                        0.5 * (tau0 + (tau0 * tau0 + 4.0 * rhs).sqrt())
                    }
                    _ => tau0,
                };
                let omega = 1.0 / tau_eff;
                let guo_pref = 1.0 - 0.5 * omega;
                for (i, d) in dst.iter_mut().enumerate() {
                    let cvec = [CX[i] as f64, CY[i] as f64, CZ[i] as f64];
                    let cu = cvec[0] * u[0] + cvec[1] * u[1] + cvec[2] * u[2];
                    let mut guo = 0.0;
                    for a in 0..3 {
                        guo += (cvec[a] - u[a]) / CS2 * force[a]
                            + cu * cvec[a] / (CS2 * CS2) * force[a];
                    }
                    // Regularised source: feq plus the second-order
                    // Hermite reconstruction of the non-equilibrium part.
                    let src_i = if regularized {
                        let mut q_contract = 0.0;
                        for (a, row) in pij.iter().enumerate() {
                            for (b, p) in row.iter().enumerate() {
                                let q_ab = cvec[a] * cvec[b] - if a == b { CS2 } else { 0.0 };
                                q_contract += q_ab * p;
                            }
                        }
                        feq[i] + W[i] / (2.0 * CS2 * CS2) * q_contract
                    } else {
                        src[i]
                    };
                    *d = src_i - omega * (src_i - feq[i]) + guo_pref * W[i] * guo;
                }
            });
    }

    /// Central-moment collision: relax in the co-moving frame.
    ///
    /// Second-order deviatoric moments relax with the LES-augmented
    /// viscous rate, the trace with the bulk rate, and every
    /// higher-order moment with the ghost rate (default 1: fully
    /// equilibrated). Mass and momentum moments are untouched, so
    /// conservation is exact. The moment-basis inverse is recomputed
    /// per call site from [`moment_inverse`]; it is a 19×19 constant,
    /// so the cost is negligible next to the grid loop.
    fn collide_central(&mut self) {
        let tau0 = self.cfg.tau;
        let force = self.cfg.force;
        let cs = self.cfg.smagorinsky;
        let omega_bulk = 1.0 / self.cfg.bulk_tau;
        let ghost = self.cfg.ghost_rate;
        let cells = &self.cells;
        let inv = moment_inverse();
        self.f_col
            .par_chunks_mut(Q)
            .zip(self.f.par_chunks(Q))
            .enumerate()
            .for_each(|(idx, (dst, src))| {
                if cells[idx] == Cell::Solid {
                    dst.copy_from_slice(src);
                    return;
                }
                let mut rho = 0.0;
                let mut mom = [0.0; 3];
                for i in 0..Q {
                    rho += src[i];
                    mom[0] += CX[i] as f64 * src[i];
                    mom[1] += CY[i] as f64 * src[i];
                    mom[2] += CZ[i] as f64 * src[i];
                }
                let inv_rho = 1.0 / rho;
                let u = [
                    (mom[0] + 0.5 * force[0]) * inv_rho,
                    (mom[1] + 0.5 * force[1]) * inv_rho,
                    (mom[2] + 0.5 * force[2]) * inv_rho,
                ];
                // Central moments of the incoming populations.
                let mut kap = [0.0; Q];
                for i in 0..Q {
                    let d = [
                        CX[i] as f64 - u[0],
                        CY[i] as f64 - u[1],
                        CZ[i] as f64 - u[2],
                    ];
                    for (k, kap_k) in kap.iter_mut().enumerate() {
                        let (a, b, c) = MOMENT_EXP[k];
                        *kap_k += src[i]
                            * d[0].powi(a as i32)
                            * d[1].powi(b as i32)
                            * d[2].powi(c as i32);
                    }
                }
                let keq = |k: usize| -> f64 {
                    match MOMENT_EXP[k] {
                        (0, 0, 0) => rho,
                        (2, 0, 0) | (0, 2, 0) | (0, 0, 2) => rho * CS2,
                        (2, 2, 0) | (0, 2, 2) | (2, 0, 2) => rho * CS2 * CS2,
                        _ => 0.0,
                    }
                };
                // LES tau_eff from the second-order central moments:
                // Pi_ab = kap_ab - kap_eq_ab (cross terms have eq 0).
                let tau_eff = match cs {
                    Some(c) if c > 0.0 => {
                        let ixx = moment_index(2, 0, 0).unwrap();
                        let iyy = moment_index(0, 2, 0).unwrap();
                        let izz = moment_index(0, 0, 2).unwrap();
                        let ixy = moment_index(1, 1, 0).unwrap();
                        let ixz = moment_index(1, 0, 1).unwrap();
                        let iyz = moment_index(0, 1, 1).unwrap();
                        let pij = [
                            [kap[ixx] - rho * CS2, kap[ixy], kap[ixz]],
                            [kap[ixy], kap[iyy] - rho * CS2, kap[iyz]],
                            [kap[ixz], kap[iyz], kap[izz] - rho * CS2],
                        ];
                        let mut pp = 0.0;
                        for row in &pij {
                            for p in row {
                                pp += p * p;
                            }
                        }
                        let q = (2.0 * pp).sqrt();
                        let rhs = c * c * q / (2.0 * rho * CS2 * CS2);
                        0.5 * (tau0 + (tau0 * tau0 + 4.0 * rhs).sqrt())
                    }
                    _ => tau0,
                };
                let omega_nu = 1.0 / tau_eff;
                // Relax: order 0/1 untouched; order 2 deviatoric with
                // omega_nu and trace with omega_bulk; order >= 3 ghost.
                let ixx = moment_index(2, 0, 0).unwrap();
                let iyy = moment_index(0, 2, 0).unwrap();
                let izz = moment_index(0, 0, 2).unwrap();
                let mean = (kap[ixx] + kap[iyy] + kap[izz]) / 3.0;
                let mean_eq = rho * CS2;
                let mean_new = mean - omega_bulk * (mean - mean_eq);
                let mut kap_new = kap;
                for (k, knew) in kap_new.iter_mut().enumerate() {
                    let order = MOMENT_EXP[k].0 + MOMENT_EXP[k].1 + MOMENT_EXP[k].2;
                    if order >= 3 {
                        *knew = kap[k] - ghost * (kap[k] - keq(k));
                    } else if order == 2 && k != ixx && k != iyy && k != izz {
                        *knew = kap[k] - omega_nu * kap[k];
                    }
                }
                for &k in &[ixx, iyy, izz] {
                    let dev = kap[k] - mean;
                    kap_new[k] = mean_new + dev - omega_nu * dev;
                }
                // Fourth-order moments take their cumulant target
                // (cascaded collision): C220 -> 0 means
                // k220 = (k200*k020 + 2*k110^2)/rho from the relaxed
                // second moments, not a fixed Maxwell value. Relaxing
                // them to rho*cs^4 in shear injects O(u^2) errors at
                // exactly the jet shear layers (measured blow-ups).
                let ixy = moment_index(1, 1, 0).unwrap();
                let ixz = moment_index(1, 0, 1).unwrap();
                let iyz = moment_index(0, 1, 1).unwrap();
                let i220 = moment_index(2, 2, 0).unwrap();
                let i022 = moment_index(0, 2, 2).unwrap();
                let i202 = moment_index(2, 0, 2).unwrap();
                kap_new[i220] =
                    (kap_new[ixx] * kap_new[iyy] + 2.0 * kap_new[ixy] * kap_new[ixy]) / rho;
                kap_new[i202] =
                    (kap_new[ixx] * kap_new[izz] + 2.0 * kap_new[ixz] * kap_new[ixz]) / rho;
                kap_new[i022] =
                    (kap_new[iyy] * kap_new[izz] + 2.0 * kap_new[iyz] * kap_new[iyz]) / rho;
                // Central -> raw moments (binomial shift by u), then
                // raw moments -> populations via the basis inverse.
                let mut raw = [0.0; Q];
                for (k, raw_k) in raw.iter_mut().enumerate() {
                    let (a, b, c) = MOMENT_EXP[k];
                    let mut sum = 0.0;
                    for al in 0..=a {
                        for be in 0..=b {
                            for ga in 0..=c {
                                if let Some(sk) = moment_index(al, be, ga) {
                                    sum += binom(a, al)
                                        * binom(b, be)
                                        * binom(c, ga)
                                        * u[0].powi((a - al) as i32)
                                        * u[1].powi((b - be) as i32)
                                        * u[2].powi((c - ga) as i32)
                                        * kap_new[sk];
                                }
                            }
                        }
                    }
                    *raw_k = sum;
                }
                // Body force in moment space: the post-collision first
                // raw moments must carry the full per-step momentum
                // increment (mom + F), matching the Guo BGK path where
                // relaxation plus the Guo term sum to the same. Adding
                // the BGK population term here would under-force by
                // (1 - 1/(2 tau)) - measured as a 0.37x Poiseuille
                // centreline before this fix.
                raw[moment_index(1, 0, 0).unwrap()] += force[0];
                raw[moment_index(0, 1, 0).unwrap()] += force[1];
                raw[moment_index(0, 0, 1).unwrap()] += force[2];
                for (i, d) in dst.iter_mut().enumerate() {
                    let mut fi = 0.0;
                    for (k, raw_k) in raw.iter().enumerate() {
                        fi += inv[i][k] * raw_k;
                    }
                    *d = fi;
                }
            });
    }

    fn stream(&mut self, ramp: f64) {
        let (nx, ny, nz) = (self.cfg.nx, self.cfg.ny, self.cfg.nz);
        let periodic = self.cfg.periodic;
        let cells = &self.cells;
        let wall_velocity = &self.wall_velocity;
        let f_col = &self.f_col;
        let outlet_normal = self.cfg.outlet_normal;
        let outlet_velocity = self.cfg.outlet_velocity;
        let inlet_velocity_cells = &self.inlet_velocity_cells;
        self.f_next
            .par_chunks_mut(Q)
            .enumerate()
            .for_each(|(idx, dst)| {
                let cell = cells[idx];
                if cell == Cell::Solid {
                    dst.copy_from_slice(&f_col[idx * Q..idx * Q + Q]);
                    return;
                }
                if cell == Cell::Inlet {
                    let base = inlet_velocity_cells[idx];
                    let u_in = [base[0] * ramp, base[1] * ramp, base[2] * ramp];
                    for (i, d) in dst.iter_mut().enumerate() {
                        *d = equilibrium(i, 1.0, u_in);
                    }
                    return;
                }
                let x = (idx % nx) as i32;
                let y = ((idx / nx) % ny) as i32;
                let z = (idx / (nx * ny)) as i32;
                if cell == Cell::Outlet {
                    // Zero-gradient outlet: copy the collided state of the
                    // nearest interior cell along the outlet normal.
                    let on = outlet_normal;
                    let ix = (x + on[0]).clamp(0, nx as i32 - 1) as usize;
                    let iy = (y + on[1]).clamp(0, ny as i32 - 1) as usize;
                    let iz = (z + on[2]).clamp(0, nz as i32 - 1) as usize;
                    let src = ix + nx * (iy + ny * iz);
                    dst.copy_from_slice(&f_col[src * Q..src * Q + Q]);
                    return;
                }
                let mut rho_self = 0.0;
                for i in 0..Q {
                    rho_self += f_col[idx * Q + i];
                }
                for i in 0..Q {
                    let mut sx = x - CX[i];
                    let mut sy = y - CY[i];
                    let mut sz = z - CZ[i];
                    let mut outside = false;
                    if periodic[0] {
                        sx = sx.rem_euclid(nx as i32);
                    } else if sx < 0 || sx >= nx as i32 {
                        outside = true;
                    }
                    if periodic[1] {
                        sy = sy.rem_euclid(ny as i32);
                    } else if sy < 0 || sy >= ny as i32 {
                        outside = true;
                    }
                    if periodic[2] {
                        sz = sz.rem_euclid(nz as i32);
                    } else if sz < 0 || sz >= nz as i32 {
                        outside = true;
                    }
                    if outside {
                        dst[i] = bounced(f_col, idx, i, wall_velocity);
                        continue;
                    }
                    let sidx = (sx as usize) + nx * (sy as usize + ny * sz as usize);
                    if cells[sidx] == Cell::Inlet {
                        // Non-equilibrium extrapolation inlet (Guo et al.
                        // 2002): equilibrium at the imposed velocity plus
                        // this cell's own non-equilibrium part. A full
                        // equilibrium overwrite of the inlet plane pumps
                        // acoustic energy (measured: feed-pipe blow-up).
                        let ub = inlet_velocity_cells[sidx];
                        let u_in = [ub[0] * ramp, ub[1] * ramp, ub[2] * ramp];
                        let mut rho_d = 0.0;
                        let mut mom_d = [0.0; 3];
                        for j in 0..Q {
                            let fj = f_col[idx * Q + j];
                            rho_d += fj;
                            mom_d[0] += CX[j] as f64 * fj;
                            mom_d[1] += CY[j] as f64 * fj;
                            mom_d[2] += CZ[j] as f64 * fj;
                        }
                        let u_d = [mom_d[0] / rho_d, mom_d[1] / rho_d, mom_d[2] / rho_d];
                        // Local density in the boundary equilibrium:
                        // fixing rho = 1 here makes the inlet an
                        // unlimited mass source under jet suction and
                        // the feed pipe runs away (measured 16x the
                        // imposed velocity before blow-up).
                        dst[i] = equilibrium(i, rho_d, u_in)
                            + (f_col[idx * Q + i] - equilibrium(i, rho_d, u_d));
                    } else if cells[sidx] == Cell::Outlet {
                        // Fixed-flow outlet, non-equilibrium
                        // extrapolation with local density and the
                        // imposed withdrawal velocity (see cuda.rs
                        // twin). A zero-gradient copy drains the domain
                        // at the circulation rate instead of the pump
                        // rate (measured: mean density 1.0 -> 0.54 in
                        // 30k steps); a fixed-density form blew up at
                        // the patch edge.
                        let mut rho_d = 0.0;
                        let mut mom_d = [0.0; 3];
                        for j in 0..Q {
                            let fj = f_col[idx * Q + j];
                            rho_d += fj;
                            mom_d[0] += CX[j] as f64 * fj;
                            mom_d[1] += CY[j] as f64 * fj;
                            mom_d[2] += CZ[j] as f64 * fj;
                        }
                        let u_d = [mom_d[0] / rho_d, mom_d[1] / rho_d, mom_d[2] / rho_d];
                        dst[i] = equilibrium(i, rho_d, outlet_velocity)
                            + (f_col[idx * Q + i] - equilibrium(i, rho_d, u_d));
                    } else if cells[sidx] == Cell::Solid {
                        // Halfway bounce-back with moving-wall momentum term.
                        let uw = wall_velocity[sidx];
                        let base = f_col[idx * Q + OPP[i]];
                        let corr = 2.0 * W[i] * rho_self * (CX[i] as f64 * uw[0]
                            + CY[i] as f64 * uw[1]
                            + CZ[i] as f64 * uw[2])
                            / CS2;
                        dst[i] = base + corr;
                    } else {
                        dst[i] = f_col[sidx * Q + i];
                    }
                }
            });
    }
}

fn bounced(f_col: &[f64], idx: usize, i: usize, _wall: &[[f64; 3]]) -> f64 {
    f_col[idx * Q + OPP[i]]
}
