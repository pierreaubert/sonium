//! Two-level static grid refinement with one-way macroscopic nesting.
//!
//! Contains the refinement driver used for the paper's level-II region
//! (Δx and Δt halved in the draft-tube inlet region). The fine lattice
//! performs two substeps per coarse step; coupling is one-way and
//! macroscopic: fine halo cells are reset each coarse step to the
//! equilibrium of the nearest coarse cell, and the coarse solution is
//! never overwritten by the patch (a two-way equilibrium overwrite acts
//! as a momentum plug across the patch, which a channel test showed
//! suppresses the through-flow). This is a documented deviation from
//! M-Star's undocumented interface rescaling.

// Rust guideline compliant 2026-02-21

use crate::lbm::{Cell, Lattice};

/// A coarse lattice with one embedded fine patch at half spacing.
#[derive(Debug)]
pub struct RefinedSolver {
    coarse: Lattice,
    fine: Lattice,
    /// Fine-patch origin in coarse-cell coordinates.
    origin: [usize; 3],
    /// Fine-patch extent in coarse cells.
    extent: [usize; 3],
}

impl RefinedSolver {
    /// Builds the coupled pair; fine dimensions must be twice `extent`.
    ///
    /// # Panics
    /// Panics if the fine grid does not match twice the patch extent or
    /// the patch leaves the coarse grid.
    pub fn new(coarse: Lattice, fine: Lattice, origin: [usize; 3], extent: [usize; 3]) -> Self {
        let fc = fine.config();
        assert_eq!(
            (fc.nx, fc.ny, fc.nz),
            (2 * extent[0], 2 * extent[1], 2 * extent[2]),
            "fine grid must be twice the patch extent"
        );
        let cc = coarse.config();
        assert!(
            origin[0] + extent[0] <= cc.nx
                && origin[1] + extent[1] <= cc.ny
                && origin[2] + extent[2] <= cc.nz,
            "fine patch leaves the coarse grid"
        );
        Self { coarse, fine, origin, extent }
    }

    /// Returns the coarse lattice.
    #[must_use]
    pub fn coarse(&self) -> &Lattice {
        &self.coarse
    }

    /// Returns the fine lattice.
    #[must_use]
    pub fn fine(&self) -> &Lattice {
        &self.fine
    }

    /// Advances one coarse step (two fine substeps).
    pub fn step(&mut self, ramp: f64) {
        self.impose_fine_halo();
        self.fine.step(ramp);
        self.impose_fine_halo();
        self.fine.step(ramp);
        self.coarse.step(ramp);
    }

    /// Macroscopic state inside the patch, from the fine lattice.
    ///
    /// Maps a coarse-cell coordinate inside the patch to the average of
    /// its 2×2×2 fine block.
    ///
    /// # Panics
    /// Panics if the coordinate lies outside the patch.
    #[must_use]
    pub fn patch_macroscopic(&self, cx: usize, cy: usize, cz: usize) -> (f64, [f64; 3]) {
        assert!(
            cx >= self.origin[0] && cx < self.origin[0] + self.extent[0]
                && cy >= self.origin[1] && cy < self.origin[1] + self.extent[1]
                && cz >= self.origin[2] && cz < self.origin[2] + self.extent[2],
            "coordinate outside the fine patch"
        );
        let (lx, ly, lz) = (cx - self.origin[0], cy - self.origin[1], cz - self.origin[2]);
        let mut rho = 0.0;
        let mut mom = [0.0; 3];
        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    let (r, u) = self
                        .fine
                        .macroscopic(self.fine.index(2 * lx + dx, 2 * ly + dy, 2 * lz + dz));
                    rho += r;
                    for a in 0..3 {
                        mom[a] += r * u[a];
                    }
                }
            }
        }
        rho /= 8.0;
        (rho, [mom[0] / (8.0 * rho), mom[1] / (8.0 * rho), mom[2] / (8.0 * rho)])
    }

    fn impose_fine_halo(&mut self) {
        let (fx, fy, fz) = {
            let c = self.fine.config();
            (c.nx, c.ny, c.nz)
        };
        let periodic = self.fine.config().periodic;
        for z in 0..fz {
            for y in 0..fy {
                for x in 0..fx {
                    let on_halo = (!periodic[0] && (x == 0 || x == fx - 1))
                        || (!periodic[1] && (y == 0 || y == fy - 1))
                        || (!periodic[2] && (z == 0 || z == fz - 1));
                    if !on_halo {
                        continue;
                    }
                    let cx = self.origin[0] + x / 2;
                    let cy = self.origin[1] + y / 2;
                    let cz = self.origin[2] + z / 2;
                    let cidx = self.coarse.index(cx, cy, cz);
                    let (rho, u) = self.coarse.macroscopic(cidx);
                    let fidx = self.fine.index(x, y, z);
                    self.fine.set_equilibrium(fidx, rho, u);
                }
            }
        }
    }
}

/// Builds a wall-bounded channel cell mask for refinement tests.
///
/// Fluid everywhere except the two z boundary layers, which are solid.
#[must_use]
pub fn channel_cells(nx: usize, ny: usize, nz: usize) -> Vec<Cell> {
    let mut cells = vec![Cell::Fluid; nx * ny * nz];
    for z in 0..nz {
        if z == 0 || z == nz - 1 {
            for y in 0..ny {
                for x in 0..nx {
                    cells[x + nx * (y + ny * z)] = Cell::Solid;
                }
            }
        }
    }
    cells
}
