//! Analytic JLR geometry voxelised onto the Cartesian lattice.
//!
//! Contains the signed-distance classification of the jet loop reactor
//! from the paper's Table A.1 (reactor Ø165 mm × 1640 mm, draft tube
//! 100 mm ID / 5 mm wall × 1200 mm, baffle plate Ø145 × 5 mm, conical
//! nozzle 36→12 mm), the draft-tube support struts, and the head-space
//! shortening of 54 mm applied by the authors' simulation. Dimensions
//! marked derived in `fixtures/geometry_table_a1.csv` are configurable
//! here and are not paper values. Coordinates are millimetres with
//! z = 0 at the reactor bottom inner face.

// Rust guideline compliant 2026-02-21

use crate::lbm::Cell;

/// JLR dimensions in millimetres, defaults from paper Table A.1.
#[derive(Debug, Clone)]
pub struct JlrGeometry {
    /// Reactor inner diameter.
    pub reactor_diameter_mm: f64,
    /// Reactor total height (physical, before head-space shortening).
    pub reactor_height_mm: f64,
    /// Head-space shortening applied in the simulation domain.
    pub headspace_shortening_mm: f64,
    /// Draft tube inner diameter.
    pub draft_tube_diameter_mm: f64,
    /// Draft tube wall thickness.
    pub draft_tube_wall_mm: f64,
    /// Draft tube length.
    pub draft_tube_length_mm: f64,
    /// Bottom space height (z of the baffle plate underside region).
    pub bottom_space_mm: f64,
    /// Bottom redirection height (baffle plate to draft tube bottom).
    pub bottom_redirection_mm: f64,
    /// Baffle plate diameter.
    pub baffle_diameter_mm: f64,
    /// Baffle plate thickness.
    pub baffle_thickness_mm: f64,
    /// Nozzle outlet diameter.
    pub nozzle_outlet_diameter_mm: f64,
    /// Nozzle inlet diameter.
    pub nozzle_inlet_diameter_mm: f64,
    /// Nozzle cone length (derived interpretation, see provenance).
    pub nozzle_cone_length_mm: f64,
    /// Vertical gap between nozzle outlet and draft tube top (derived).
    pub nozzle_gap_mm: f64,
    /// Measuring height from the reactor bottom (paper hMP).
    pub measuring_height_mm: f64,
    /// Support strut radial thickness (derived assumption).
    pub strut_thickness_mm: f64,
    /// Support strut vertical extent (derived assumption).
    pub strut_height_mm: f64,
    /// Azimuth rotation of the whole internals, degrees (paper's 45° study).
    pub rotation_deg: f64,
}

impl Default for JlrGeometry {
    fn default() -> Self {
        Self {
            reactor_diameter_mm: 165.0,
            reactor_height_mm: 1640.0,
            headspace_shortening_mm: 54.0,
            draft_tube_diameter_mm: 100.0,
            draft_tube_wall_mm: 5.0,
            draft_tube_length_mm: 1200.0,
            bottom_space_mm: 157.0,
            bottom_redirection_mm: 115.0,
            baffle_diameter_mm: 145.0,
            baffle_thickness_mm: 5.0,
            nozzle_outlet_diameter_mm: 12.0,
            nozzle_inlet_diameter_mm: 36.0,
            nozzle_cone_length_mm: 73.0,
            nozzle_gap_mm: 22.0,
            measuring_height_mm: 920.0,
            strut_thickness_mm: 5.0,
            strut_height_mm: 40.0,
            rotation_deg: 0.0,
        }
    }
}

impl JlrGeometry {
    /// Simulated domain height after head-space shortening.
    #[must_use]
    pub fn domain_height_mm(&self) -> f64 {
        self.reactor_height_mm - self.headspace_shortening_mm
    }

    /// z of the draft tube bottom face.
    #[must_use]
    pub fn draft_tube_bottom_mm(&self) -> f64 {
        self.bottom_space_mm + self.bottom_redirection_mm
    }

    /// z of the draft tube top face.
    #[must_use]
    pub fn draft_tube_top_mm(&self) -> f64 {
        self.draft_tube_bottom_mm() + self.draft_tube_length_mm
    }

    /// z of the nozzle outlet plane.
    #[must_use]
    pub fn nozzle_outlet_z_mm(&self) -> f64 {
        self.draft_tube_top_mm() + self.nozzle_gap_mm
    }

    /// Analytic annular-gap cross-section, π/4·(dR² − dDT,outer²).
    ///
    /// # Examples
    /// ```rust
    /// use sonium_sim_jlr::geometry::JlrGeometry;
    /// let a = JlrGeometry::default().annulus_area_mm2();
    /// assert!((a - 11879.1).abs() < 1.0);
    /// ```
    #[must_use]
    pub fn annulus_area_mm2(&self) -> f64 {
        let outer = self.draft_tube_diameter_mm + 2.0 * self.draft_tube_wall_mm;
        std::f64::consts::PI / 4.0
            * (self.reactor_diameter_mm * self.reactor_diameter_mm - outer * outer)
    }

    /// Classifies one point (mm, reactor axis at x = y = 0).
    ///
    /// The vessel wall, draft tube wall, baffle plate, nozzle cone wall,
    /// support struts, top head plate outside the feed pipe, and the
    /// bottom face outside the outlet patch are solid. The feed pipe
    /// (r < nozzle inlet radius at the domain top) is the inlet and the
    /// bottom patch under the baffle plate is the outlet.
    #[must_use]
    pub fn classify(&self, x_mm: f64, y_mm: f64, z_mm: f64) -> Cell {
        let h = self.domain_height_mm();
        if !(0.0..=h).contains(&z_mm) {
            return Cell::Solid;
        }
        // Rotate the sampling point by -rotation so internals appear rotated.
        let rot = -self.rotation_deg.to_radians();
        let (xr, yr) = (
            x_mm * rot.cos() - y_mm * rot.sin(),
            x_mm * rot.sin() + y_mm * rot.cos(),
        );
        let r = (xr * xr + yr * yr).sqrt();
        let r_reactor = self.reactor_diameter_mm / 2.0;
        if r > r_reactor {
            return Cell::Solid;
        }
        let dt_in = self.draft_tube_diameter_mm / 2.0;
        let dt_out = dt_in + self.draft_tube_wall_mm;
        let z_dt0 = self.draft_tube_bottom_mm();
        let z_dt1 = self.draft_tube_top_mm();
        // Top boundary: feed pipe inlet, elsewhere head plate.
        if z_mm >= h - 1e-9 {
            return if r <= self.nozzle_inlet_diameter_mm / 2.0 {
                Cell::Inlet
            } else {
                Cell::Solid
            };
        }
        // Nozzle cone wall between outlet and inlet radii: a thin sheet
        // just outside the cone surface (fluid inside, head space
        // outside). A band centred on the surface would choke the
        // 6 mm outlet radius to a pinhole.
        let z_noz = self.nozzle_outlet_z_mm();
        if z_mm > z_noz && z_mm <= z_noz + self.nozzle_cone_length_mm {
            let frac = (z_mm - z_noz) / self.nozzle_cone_length_mm;
            let r_cone = self.nozzle_outlet_diameter_mm / 2.0
                + frac * (self.nozzle_inlet_diameter_mm - self.nozzle_outlet_diameter_mm) / 2.0;
            if r > r_cone && r <= r_cone + 1.5 {
                return Cell::Solid;
            }
            if r > r_cone && z_mm > z_dt1 + self.nozzle_cone_length_mm {
                return Cell::Solid;
            }
        }
        if z_mm > z_noz + self.nozzle_cone_length_mm && r > self.nozzle_inlet_diameter_mm / 2.0 {
            return Cell::Solid;
        }
        // Draft tube wall.
        if (z_dt0..=z_dt1).contains(&z_mm) && r > dt_in && r <= dt_out {
            return Cell::Solid;
        }
        // Support ring at the draft tube bottom (derived: 10 mm tall band).
        if (z_dt0 - 10.0..z_dt0).contains(&z_mm) && (r - dt_out).abs() < 5.0 {
            return Cell::Solid;
        }
        // Support struts: 2 at the top (180° apart), 4 at the bottom (90°).
        let theta = yr.atan2(xr).to_degrees().rem_euclid(360.0);
        if r > dt_out && r <= r_reactor {
            let near = |angles: &[f64]| {
                angles.iter().any(|a| {
                    let d = (theta - a).abs().min(360.0 - (theta - a).abs());
                    // angular half-width from strut thickness at this radius
                    let half = (self.strut_thickness_mm / 2.0 / r.max(1.0)).to_degrees();
                    d < half
                })
            };
            if (z_dt1 - self.strut_height_mm..=z_dt1).contains(&z_mm) && near(&[0.0, 180.0]) {
                return Cell::Solid;
            }
            if (z_dt0..=z_dt0 + self.strut_height_mm).contains(&z_mm)
                && near(&[0.0, 90.0, 180.0, 270.0])
            {
                return Cell::Solid;
            }
        }
        // Baffle plate.
        if (self.bottom_space_mm..=self.bottom_space_mm + self.baffle_thickness_mm)
            .contains(&z_mm)
            && r <= self.baffle_diameter_mm / 2.0
        {
            return Cell::Solid;
        }
        // Bottom face: outlet patch below the baffle plate.
        if z_mm <= 1e-9 {
            return if r <= self.baffle_diameter_mm / 2.0 {
                Cell::Outlet
            } else {
                Cell::Solid
            };
        }
        Cell::Fluid
    }

    /// Builds the cell mask for a grid with spacing `dx_mm`.
    ///
    /// The grid is centred on the reactor axis and spans the reactor
    /// diameter in x/y. Returns the mask plus grid dimensions.
    #[must_use]
    pub fn voxelise(&self, dx_mm: f64) -> (Vec<Cell>, usize, usize, usize) {
        let d = self.reactor_diameter_mm;
        let nx = (d / dx_mm).ceil() as usize + 2;
        let ny = nx;
        let nz = (self.domain_height_mm() / dx_mm).ceil() as usize + 1;
        let mut cells = vec![Cell::Solid; nx * ny * nz];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let xm = (x as f64 + 0.5) * dx_mm - nx as f64 * dx_mm / 2.0;
                    let ym = (y as f64 + 0.5) * dx_mm - ny as f64 * dx_mm / 2.0;
                    // Clamp the top layer onto the domain lid so the
                    // inlet plane is actually represented by nodes.
                    let zm = (z as f64 * dx_mm).min(self.domain_height_mm());
                    let idx = x + nx * (y + ny * z);
                    cells[idx] = self.classify(xm, ym, zm);
                }
            }
        }
        (cells, nx, ny, nz)
    }

    /// Effective fluid area of the voxelised annulus at the measuring
    /// height, in mm², counting fluid cells in the annulus band.
    ///
    /// # Panics
    /// Panics if the mask length does not match the grid dimensions.
    #[must_use]
    pub fn voxel_annulus_area_mm2(
        &self,
        cells: &[Cell],
        nx: usize,
        ny: usize,
        dx_mm: f64,
    ) -> f64 {
        let nz = cells.len() / (nx * ny);
        let z = ((self.measuring_height_mm / dx_mm).round() as usize).min(nz - 1);
        let dt_out = (self.draft_tube_diameter_mm + 2.0 * self.draft_tube_wall_mm) / 2.0;
        let r_reactor = self.reactor_diameter_mm / 2.0;
        let mut count = 0usize;
        for y in 0..ny {
            for x in 0..nx {
                let xm = (x as f64 + 0.5) * dx_mm - nx as f64 * dx_mm / 2.0;
                let ym = (y as f64 + 0.5) * dx_mm - ny as f64 * dx_mm / 2.0;
                let r = (xm * xm + ym * ym).sqrt();
                if r > dt_out && r <= r_reactor && cells[x + nx * (y + ny * z)] != Cell::Solid {
                    count += 1;
                }
            }
        }
        count as f64 * dx_mm * dx_mm
    }
}
