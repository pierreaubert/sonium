//! PORT-OPTIM website gallery emitter.
//!
//! Renders the frozen bass-reflex port-shape optimisation campaign as
//! deterministic website assets: SVG figures, authoritative CSV siblings,
//! a checksummed `gallery.json` manifest, and a provenance record. Reads
//! only committed Step-0/7/8/9 fixtures; no solves run, so output is fast
//! and bit-stable for a fixed source tree.
//!
//! Usage:
//!
//! ```sh
//! cargo run -p sonium-sim-speaker-modeling --example port_optim_gallery -- <out-dir>
//! ```
//!
//! # Errors
//!
//! Returns an error for a missing output directory argument, unreadable or
//! inconsistent fixtures, or any asset write/validation failure.

use sonium_xem::io::{GalleryRenderer, validate_gallery_manifest};
use std::error::Error;
use std::path::{Path, PathBuf};

const SCHEMA: &str = "sonium-port-optim-gallery/v1";
const BG: &str = "#10131a";
const AXIS: &str = "#c7d0df";
const CYAN: &str = "#30c9ff";
const GREY: &str = "#8a93a5";
const GREEN: &str = "#3ddc84";
const RED: &str = "#ff6b6b";

fn main() -> Result<(), Box<dyn Error>> {
    let out_dir = std::env::args()
        .nth(1)
        .ok_or("usage: port_optim_gallery <out-dir>")?;
    run(&PathBuf::from(out_dir))
}

fn run(out_dir: &Path) -> Result<(), Box<dyn Error>> {
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");

    let winner_csv = std::fs::read_to_string(fixtures.join("port_optim_step7/winner_profile.csv"))?;
    let profile = parse_profile_csv(&winner_csv)?;
    let r7 = read_json(&fixtures.join("port_optim_step7/step7_report.json"))?;
    let r8 = read_json(&fixtures.join("port_optim_step8/step8_report.json"))?;
    let r9 = read_json(&fixtures.join("port_optim_step9/step9_report.json"))?;

    let case = obj(&r7, "case")?;
    let r0 = num(case, "port_radius_r0_m")?;
    let duct_l = num(case, "port_length_l_m")?;
    let rho = num(case, "density_kg_per_m3")?;
    let sound_speed = num(case, "sound_speed_m_per_s")?;
    let f_target = num(&r7, "f_target_hz")?;
    let winner_u = arr(&r7, "winner_u")?;
    // Reference volume velocity for the equal-airflow flow figure: the
    // Tier-0 RMS port flow of the baseline finalist at f_target.
    let q_ref = {
        let finalists = r7
            .get("finalists")
            .and_then(serde_json::Value::as_array)
            .ok_or("step7 report lacks finalists")?;
        let baseline = finalists.first().ok_or("step7 report has no finalist")?;
        num(obj(baseline, "tier0")?, "q_rms_m3s")?
    };

    let mut renderer = GalleryRenderer::new(out_dir)?;

    // Source CSVs first: the manifest records a checksum of each source.
    std::fs::write(out_dir.join("winner_profile.csv"), &winner_csv)?;
    let airflow_csv = comparison_csv(&r7)?;
    std::fs::write(out_dir.join("comparison_equal_airflow.csv"), &airflow_csv)?;
    let noise_csv = noise_csv(&r8)?;
    std::fs::write(out_dir.join("noise_floor.csv"), &noise_csv)?;
    let convergence_csv = convergence_csv(&r9)?;
    std::fs::write(out_dir.join("convergence.csv"), &convergence_csv)?;

    let profile_svg = profile_svg(&profile, r0, duct_l);
    std::fs::write(out_dir.join("profile_shape.svg"), &profile_svg)?;
    renderer.existing_figure(
        "profile_shape.svg",
        "svg",
        "winner_profile.csv",
        format!(
            "Winner meridian r(x) vs straight {} mm bore (N=5 spline, u=[{:.4}, {:.4}])",
            r0 * 1000.0,
            winner_u[0],
            winner_u[1]
        ),
    )?;

    // 3D airflow picture: straight bore (left) vs winner (right),
    // coloured by RMS particle velocity from the 1D duct model below.
    let straight_flow = duct_flow(&|_| r0, duct_l, f_target, q_ref, rho, sound_speed);
    let winner_flow = duct_flow(
        &|x| profile_radius(&profile, x),
        duct_l,
        f_target,
        q_ref,
        rho,
        sound_speed,
    );
    std::fs::write(
        out_dir.join("flow_distribution.csv"),
        flow_csv(&straight_flow, &winner_flow),
    )?;
    std::fs::write(
        out_dir.join("flow3d.png"),
        render_flow3d_png(&straight_flow, &winner_flow),
    )?;
    let v_straight_max = straight_flow
        .velocity
        .iter()
        .copied()
        .fold(0.0f64, f64::max);
    let v_winner_max = winner_flow.velocity.iter().copied().fold(0.0f64, f64::max);
    renderer.existing_figure(
        "flow3d.png",
        "png",
        "flow_distribution.csv",
        format!(
            "Duct airflow at {f_target:.1} Hz and equal volume velocity {q_ref:.2e} m3/s (1D plane-wave model): straight bore (left) vs optimised winner (right), colour is RMS particle velocity on a shared scale; peak {v_straight_max:.4} vs {v_winner_max:.4} m/s ({:+.1}% at the winner throat)",
            100.0 * (v_winner_max / v_straight_max - 1.0)
        ),
    )?;

    let (comparison_svg, comparison_caption) = comparison_svg(&r7)?;
    std::fs::write(
        out_dir.join("comparison_equal_airflow.svg"),
        &comparison_svg,
    )?;
    renderer.existing_figure(
        "comparison_equal_airflow.svg",
        "svg",
        "comparison_equal_airflow.csv",
        comparison_caption,
    )?;

    let (noise_svg, noise_caption) = noise_svg(&r8)?;
    std::fs::write(out_dir.join("noise_floor.svg"), &noise_svg)?;
    renderer.existing_figure("noise_floor.svg", "svg", "noise_floor.csv", noise_caption)?;

    let (convergence_svg, convergence_caption) = convergence_svg(&r9)?;
    std::fs::write(out_dir.join("convergence.svg"), &convergence_svg)?;
    renderer.existing_figure(
        "convergence.svg",
        "svg",
        "convergence.csv",
        convergence_caption,
    )?;

    let manifest_path = renderer.finish(
        "Bass-reflex port shape optimisation (inward-only spline)",
        vec![
            "fixed 1920x1080 viewport and deterministic SVG rendering".into(),
            "CSV assets are authoritative frozen campaign evidence".into(),
            "figures render archived Step-7/8/9 reports; no solves run".into(),
            "flow3d.png/flow_distribution.csv derive from a lossless 1D plane-wave duct model at f_target (ka ~ 0.04, open-end correction 0.61 r) fed with the frozen winner profile; no new campaign solve".into(),
        ],
    )?;
    validate_gallery_manifest(&manifest_path)?;

    write_provenance(out_dir, &r7, f_target)?;
    println!("port-optim gallery written to {}", out_dir.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// Fixture access
// ---------------------------------------------------------------------------

fn read_json(path: &Path) -> Result<serde_json::Value, Box<dyn Error>> {
    Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
}

fn obj<'a>(
    value: &'a serde_json::Value,
    key: &str,
) -> Result<&'a serde_json::Value, Box<dyn Error>> {
    value
        .get(key)
        .filter(|v| v.is_object())
        .ok_or_else(|| format!("fixture object missing: {key}").into())
}

fn arr(value: &serde_json::Value, key: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    let items = value
        .get(key)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("fixture array missing: {key}"))?;
    items
        .iter()
        .map(|v| {
            v.as_f64()
                .ok_or_else(|| format!("fixture array non-numeric: {key}").into())
        })
        .collect()
}

fn num(value: &serde_json::Value, key: &str) -> Result<f64, Box<dyn Error>> {
    value
        .get(key)
        .and_then(serde_json::Value::as_f64)
        .filter(|v| v.is_finite())
        .ok_or_else(|| format!("fixture number missing: {key}").into())
}

fn str_field(value: &serde_json::Value, key: &str) -> Result<String, Box<dyn Error>> {
    value
        .get(key)
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| format!("fixture string missing: {key}").into())
}

fn parse_profile_csv(text: &str) -> Result<Vec<(f64, f64)>, Box<dyn Error>> {
    let mut stations = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if index == 0 {
            if line.trim() != "x_m,r_m" {
                return Err("winner_profile.csv has an unexpected header".into());
            }
            continue;
        }
        if line.trim().is_empty() {
            continue;
        }
        let (x, r) = line
            .split_once(',')
            .ok_or("winner_profile.csv has a malformed row")?;
        let (x, r): (f64, f64) = (x.trim().parse()?, r.trim().parse()?);
        if !x.is_finite() || !r.is_finite() || r <= 0.0 {
            return Err("winner_profile.csv has a non-physical station".into());
        }
        stations.push((x, r));
    }
    if stations.len() < 2 {
        return Err("winner_profile.csv has too few stations".into());
    }
    Ok(stations)
}

// ---------------------------------------------------------------------------
// Derived CSVs (authoritative figure sources)
// ---------------------------------------------------------------------------

fn comparison_csv(r7: &serde_json::Value) -> Result<String, Box<dyn Error>> {
    let levels = r7
        .get("equal_airflow")
        .and_then(serde_json::Value::as_array)
        .ok_or("step7 report lacks equal_airflow")?;
    let mut csv = String::from("v_straight_v,v_winner_v,match_residual_pct,d_thd_rel,d_noise_db\n");
    for level in levels {
        csv.push_str(&format!(
            "{:?},{:?},{:?},{:?},{:?}\n",
            num(level, "v_straight_v")?,
            num(level, "v_winner_v")?,
            num(level, "match_residual_pct")?,
            num(level, "d_thd_rel")?,
            num(level, "d_noise_db")?,
        ));
    }
    Ok(csv)
}

fn noise_csv(r8: &serde_json::Value) -> Result<String, Box<dyn Error>> {
    let levels = r8
        .get("noise")
        .and_then(serde_json::Value::as_array)
        .ok_or("step8 report lacks noise")?;
    let mut csv = String::from("voltage_peak_v,n_snapshots,straight_db,winner_db,delta_db\n");
    for level in levels {
        let straight = num(obj(level, "straight")?, "integrated_noise_floor_db")?;
        let winner = num(obj(level, "winner")?, "integrated_noise_floor_db")?;
        csv.push_str(&format!(
            "{:?},{:?},{:?},{:?},{:?}\n",
            num(level, "voltage_peak_v")?,
            level
                .get("n_snapshots")
                .and_then(serde_json::Value::as_u64)
                .ok_or("step8 level lacks n_snapshots")?,
            straight,
            winner,
            winner - straight,
        ));
    }
    Ok(csv)
}

fn convergence_csv(r9: &serde_json::Value) -> Result<String, Box<dyn Error>> {
    let mut csv = String::from(
        "ladder,mesh_size_m,samples_per_cycle,noise_proxy,thd_fraction,port_flow_rms_m3s\n",
    );
    for (ladder, key) in [
        ("spatial", "spatial_levels"),
        ("temporal", "temporal_levels"),
    ] {
        let levels = r9
            .get(key)
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| format!("step9 report lacks {key}"))?;
        for level in levels {
            csv.push_str(&format!(
                "{ladder},{:?},{:?},{:?},{:?},{:?}\n",
                num(level, "mesh_size_m")?,
                level
                    .get("samples_per_cycle")
                    .and_then(serde_json::Value::as_u64)
                    .ok_or("step9 level lacks samples_per_cycle")?,
                num(level, "noise_proxy")?,
                num(level, "thd_fraction")?,
                num(level, "port_flow_rms_m3s")?,
            ));
        }
    }
    Ok(csv)
}

// ---------------------------------------------------------------------------
// Duct flow (lossless 1D plane-wave / Webster model) and 3D rendering
// ---------------------------------------------------------------------------

/// Radius of the winner profile at `x` (linear between stations).
fn profile_radius(profile: &[(f64, f64)], x: f64) -> f64 {
    if x <= profile[0].0 {
        return profile[0].1;
    }
    for pair in profile.windows(2) {
        let (x0, r0) = pair[0];
        let (x1, r1) = pair[1];
        if x <= x1 {
            return r0 + (r1 - r0) * (x - x0) / (x1 - x0);
        }
    }
    profile[profile.len() - 1].1
}

/// Acoustic state along one duct at the reference volume velocity.
struct DuctFlow {
    x: Vec<f64>,
    radius: Vec<f64>,
    /// RMS particle velocity magnitude per station (m/s).
    velocity: Vec<f64>,
    /// RMS acoustic pressure magnitude per station (Pa).
    pressure: Vec<f64>,
}

use num_complex::Complex64;

/// Propagate the plane-wave state from a pressure node placed one
/// open-end correction (0.61 r) past the outlet back to the inlet,
/// segment by segment, then scale so the inlet volume velocity equals
/// `q_ref`. Plane-wave validity: ka ~ 0.04 at the target frequency,
/// first higher duct mode near 4 kHz.
fn duct_flow(
    radius_at: &dyn Fn(f64) -> f64,
    duct_l: f64,
    frequency: f64,
    q_ref: f64,
    rho: f64,
    sound_speed: f64,
) -> DuctFlow {
    let segments = 256usize;
    let dx = duct_l / segments as f64;
    let k = 2.0 * std::f64::consts::PI * frequency / sound_speed;
    let end_correction = 0.61 * radius_at(duct_l);
    // Station areas/radii on [0, L].
    let mut radii = Vec::with_capacity(segments + 1);
    for i in 0..=segments {
        radii.push(radius_at(dx * i as f64));
    }
    // Backward propagation; segment list includes the end-correction
    // extension as one extra straight segment past the outlet.
    let mut pressure = vec![Complex64::new(0.0, 0.0); segments + 1];
    let mut volume_velocity = vec![Complex64::new(0.0, 0.0); segments + 1];
    let (mut p, mut u) = (Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0));
    let i_unit = Complex64::new(0.0, 1.0);
    let propagate = |p: Complex64, u: Complex64, area: f64, length: f64| {
        let z = rho * sound_speed / area;
        let (c, s) = ((k * length).cos(), (k * length).sin());
        // Inverse of [[c, i Z s], [i s / Z, c]] (unit determinant).
        let p_in = p * c - i_unit * (z * s) * u;
        let u_in = -i_unit * (s / z) * p + u * c;
        (p_in, u_in)
    };
    let end_area = std::f64::consts::PI * radii[segments].powi(2);
    (p, u) = propagate(p, u, end_area, end_correction);
    pressure[segments] = p;
    volume_velocity[segments] = u;
    for i in (0..segments).rev() {
        let r_mid = 0.5 * (radii[i] + radii[i + 1]);
        let area = std::f64::consts::PI * r_mid.powi(2);
        (p, u) = propagate(p, u, area, dx);
        pressure[i] = p;
        volume_velocity[i] = u;
    }
    // Scale so U(0) is exactly q_ref (real).
    let u0 = volume_velocity[0];
    let factor = q_ref * u0.conj() / u0.norm_sqr();
    let mut velocity = Vec::with_capacity(segments + 1);
    let mut pressures = Vec::with_capacity(segments + 1);
    for i in 0..=segments {
        let area = std::f64::consts::PI * radii[i].powi(2);
        velocity.push((volume_velocity[i] * factor).norm() / area);
        pressures.push((pressure[i] * factor).norm());
    }
    DuctFlow {
        x: (0..=segments).map(|i| dx * i as f64).collect(),
        radius: radii,
        velocity,
        pressure: pressures,
    }
}

/// Authoritative CSV behind the 3D airflow figure.
fn flow_csv(straight: &DuctFlow, winner: &DuctFlow) -> String {
    let mut csv =
        String::from("x_m,r_straight_m,r_winner_m,v_straight_m_s,v_winner_m_s,p_straight_pa,p_winner_pa\n");
    for i in 0..straight.x.len() {
        csv.push_str(&format!(
            "{:?},{:?},{:?},{:?},{:?},{:?},{:?}\n",
            straight.x[i],
            straight.radius[i],
            winner.radius[i],
            straight.velocity[i],
            winner.velocity[i],
            straight.pressure[i],
            winner.pressure[i],
        ));
    }
    csv
}

/// Gallery colour map (blue -> cyan -> green -> yellow -> red).
fn gallery_colour(t: f64) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    let channel = |value: f64| (value.clamp(0.0, 1.0) * 255.0).round() as u8;
    [
        channel(1.5 * t - 0.5),
        channel(1.5 - (2.0 * t - 1.0).abs()),
        channel(1.0 - 1.5 * t),
    ]
}

/// View-space triangle for the software rasterizer (depth into `y`).
struct Triangle {
    points: [[f64; 3]; 3],
    colour: [u8; 3],
}

/// Build the coloured surface of revolution for one duct flow.
fn duct_triangles(flow: &DuctFlow, t_lo: f64, t_hi: f64) -> Vec<Triangle> {
    let axial = 128usize;
    let angular = 40usize;
    let stations = flow.x.len();
    let vertex = |i: usize, j: usize| -> [f64; 3] {
        let theta = 2.0 * std::f64::consts::PI * j as f64 / angular as f64;
        [
            flow.x[i] - 0.5 * flow.x[stations - 1],
            flow.radius[i] * theta.cos(),
            flow.radius[i] * theta.sin(),
        ]
    };
    // Rotate: azimuth about Y, then elevation about X. Screen is
    // (x, z) with depth in y after both rotations.
    let view = |p: [f64; 3]| -> [f64; 3] {
        let (az, el) = (0.55f64, -0.85f64);
        let (x, y, z) = (
            p[0] * az.cos() + p[2] * az.sin(),
            p[1],
            -p[0] * az.sin() + p[2] * az.cos(),
        );
        [x, y * el.cos() - z * el.sin(), y * el.sin() + z * el.cos()]
    };
    let light = {
        let raw = [-0.35f64, 0.75, 0.55];
        let norm = (raw[0].powi(2) + raw[1].powi(2) + raw[2].powi(2)).sqrt();
        [raw[0] / norm, raw[1] / norm, raw[2] / norm]
    };
    let shade = |points: &[[f64; 3]; 3], base: [u8; 3]| -> [u8; 3] {
        let e1 = [
            points[1][0] - points[0][0],
            points[1][1] - points[0][1],
            points[1][2] - points[0][2],
        ];
        let e2 = [
            points[2][0] - points[0][0],
            points[2][1] - points[0][1],
            points[2][2] - points[0][2],
        ];
        let n = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        let norm = (n[0].powi(2) + n[1].powi(2) + n[2].powi(2)).sqrt().max(1e-30);
        let facing = (n[0] * light[0] + n[1] * light[1] + n[2] * light[2]) / norm;
        let factor = 0.32 + 0.68 * facing.abs().min(1.0);
        [
            (f64::from(base[0]) * factor) as u8,
            (f64::from(base[1]) * factor) as u8,
            (f64::from(base[2]) * factor) as u8,
        ]
    };
    let mut triangles = Vec::new();
    let push = |raw: [[f64; 3]; 3], t: f64, triangles: &mut Vec<Triangle>| {
        let points = [view(raw[0]), view(raw[1]), view(raw[2])];
        let colour = shade(&points, gallery_colour((t - t_lo) / (t_hi - t_lo)));
        triangles.push(Triangle { points, colour });
    };
    for a in 0..axial {
        let i0 = a * (stations - 1) / axial;
        let i1 = (a + 1) * (stations - 1) / axial;
        let t = 0.5 * (flow.velocity[i0] + flow.velocity[i1]);
        for j in 0..angular {
            let (v00, v10) = (vertex(i0, j), vertex(i1, j));
            let (v01, v11) = (vertex(i0, j + 1), vertex(i1, j + 1));
            push([v00, v10, v11], t, &mut triangles);
            push([v00, v11, v01], t, &mut triangles);
        }
        // End-cap fans close the tube at both ends.
        for (i, centre_t) in [(i0, flow.velocity[i0]), (i1, flow.velocity[i1])] {
            if (i == i0 && a != 0) || (i == i1 && a != axial - 1) {
                continue;
            }
            let centre = [flow.x[i] - 0.5 * flow.x[stations - 1], 0.0, 0.0];
            for j in 0..angular {
                push(
                    [centre, vertex(i, j + 1), vertex(i, j)],
                    centre_t,
                    &mut triangles,
                );
            }
        }
    }
    triangles
}

/// Side-by-side 3D airflow picture: straight bore (left), winner
/// (right), shared velocity colour scale, dark gallery theme.
fn render_flow3d_png(straight: &DuctFlow, winner: &DuctFlow) -> Vec<u8> {
    let (width, height) = (1920usize, 1080usize);
    let mut pixels = vec![[16u8, 19, 26]; width * height];
    let mut zbuffer = vec![f64::NEG_INFINITY; width * height];
    let v_lo = straight
        .velocity
        .iter()
        .chain(winner.velocity.iter())
        .copied()
        .fold(f64::INFINITY, f64::min);
    let v_hi = straight
        .velocity
        .iter()
        .chain(winner.velocity.iter())
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    // Fit both ducts with one world scale: bounding box of the view
    // vertices of the straight duct (winner differs by ~1%).
    let all = duct_triangles(straight, v_lo, v_hi);
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for triangle in &all {
        for point in &triangle.points {
            lo[0] = lo[0].min(point[0]);
            hi[0] = hi[0].max(point[0]);
            lo[1] = lo[1].min(point[2]);
            hi[1] = hi[1].max(point[2]);
        }
    }
    let scale = (820.0 / (hi[0] - lo[0])).min(680.0 / (hi[1] - lo[1]));
    let centre = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0];
    let winner_triangles = duct_triangles(winner, v_lo, v_hi);
    for (panel, triangles) in [(470.0f64, &all), (1400.0f64, &winner_triangles)] {
        for triangle in triangles.iter() {
            let screen: [[f64; 3]; 3] = [
                [
                    panel + (triangle.points[0][0] - centre[0]) * scale,
                    540.0 - (triangle.points[0][2] - centre[1]) * scale,
                    triangle.points[0][1],
                ],
                [
                    panel + (triangle.points[1][0] - centre[0]) * scale,
                    540.0 - (triangle.points[1][2] - centre[1]) * scale,
                    triangle.points[1][1],
                ],
                [
                    panel + (triangle.points[2][0] - centre[0]) * scale,
                    540.0 - (triangle.points[2][2] - centre[1]) * scale,
                    triangle.points[2][1],
                ],
            ];
            rasterize(&mut pixels, &mut zbuffer, width, height, screen, triangle.colour);
        }
    }
    // Shared colour bar at the right edge (no text; the caption and
    // CSV carry the numbers).
    for y in 240..840usize {
        let t = 1.0 - (y - 240) as f64 / 600.0;
        let colour = gallery_colour(t);
        for x in 1842..1872usize {
            pixels[y * width + x] = colour;
        }
    }
    png_bytes(width, height, &pixels)
}

/// Fill one projected triangle with a z-buffer (larger depth wins).
fn rasterize(
    pixels: &mut [[u8; 3]],
    zbuffer: &mut [f64],
    width: usize,
    height: usize,
    tri: [[f64; 3]; 3],
    colour: [u8; 3],
) {
    let (x0, x1) = (
        tri.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min),
        tri.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max),
    );
    let (y0, y1) = (
        tri.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min),
        tri.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max),
    );
    let area = (tri[1][0] - tri[0][0]) * (tri[2][1] - tri[0][1])
        - (tri[2][0] - tri[0][0]) * (tri[1][1] - tri[0][1]);
    if area.abs() < 1e-12 {
        return;
    }
    let xa = (x0.floor() as i32).max(0);
    let xb = (x1.ceil() as i32).min(width as i32 - 1);
    let ya = (y0.floor() as i32).max(0);
    let yb = (y1.ceil() as i32).min(height as i32 - 1);
    for y in ya..=yb {
        for x in xa..=xb {
            let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
            let w0 = ((tri[1][0] - px) * (tri[2][1] - py) - (tri[2][0] - px) * (tri[1][1] - py))
                / area;
            let w1 = ((tri[2][0] - px) * (tri[0][1] - py) - (tri[0][0] - px) * (tri[2][1] - py))
                / area;
            let w2 = 1.0 - w0 - w1;
            if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                continue;
            }
            let depth = w0 * tri[0][2] + w1 * tri[1][2] + w2 * tri[2][2];
            let index = y as usize * width + x as usize;
            if depth > zbuffer[index] {
                zbuffer[index] = depth;
                pixels[index] = colour;
            }
        }
    }
}

/// Encode RGB pixels as PNG (zlib-compressed scanlines, filter 0).
fn png_bytes(width: usize, height: usize, pixels: &[[u8; 3]]) -> Vec<u8> {
    use flate2::Compression;
    use flate2::write::ZlibEncoder;
    use std::io::Write;
    let mut raw = Vec::with_capacity((width * 3 + 1) * height);
    for y in 0..height {
        raw.push(0);
        for x in 0..width {
            raw.extend_from_slice(&pixels[y * width + x]);
        }
    }
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::best());
    encoder
        .write_all(&raw)
        .expect("in-memory zlib write cannot fail");
    let zlib = encoder.finish().expect("in-memory zlib finish cannot fail");
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&(width as u32).to_be_bytes());
    ihdr.extend_from_slice(&(height as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    png_chunk(&mut png, b"IHDR", &ihdr);
    png_chunk(&mut png, b"IDAT", &zlib);
    png_chunk(&mut png, b"IEND", &[]);
    png
}

fn png_chunk(png: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    png.extend_from_slice(&(data.len() as u32).to_be_bytes());
    png.extend_from_slice(kind);
    png.extend_from_slice(data);
    let mut crc_input = kind.to_vec();
    crc_input.extend_from_slice(data);
    png.extend_from_slice(&crc32(&crc_input).to_be_bytes());
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

// ---------------------------------------------------------------------------
// SVG helpers (same dark 1920x1080 theme as the shared line renderer)
// ---------------------------------------------------------------------------

fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

struct Canvas {
    body: String,
}

impl Canvas {
    fn new(title: &str) -> Self {
        Self {
            body: format!(
                "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1920\" height=\"1080\" viewBox=\"0 0 1920 1080\">\
                 <rect width=\"100%\" height=\"100%\" fill=\"{BG}\"/>\
                 <text x=\"120\" y=\"90\" fill=\"white\" font-size=\"42\" font-family=\"sans-serif\">{}</text>",
                escape(title)
            ),
        }
    }

    fn axes(&mut self, x0: f64, y0: f64, x1: f64, y1: f64) {
        self.body.push_str(&format!(
            "<path d=\"M{x0:.2} {y0:.2}V{y1:.2}H{x1:.2}\" stroke=\"{AXIS}\" \
             stroke-width=\"3\" fill=\"none\"/>"
        ));
    }

    fn polyline(&mut self, points: &[(f64, f64)], color: &str, width: f64, dash: Option<&str>) {
        let mut data = String::new();
        for (x, y) in points {
            data.push_str(&format!("{x:.2},{y:.2} "));
        }
        let dash = dash.map_or(String::new(), |d| format!(" stroke-dasharray=\"{d}\""));
        self.body.push_str(&format!(
            "<polyline points=\"{data}\" fill=\"none\" stroke=\"{color}\" \
             stroke-width=\"{width:.1}\"{dash}/>"
        ));
    }

    fn bar(&mut self, x: f64, y_top: f64, w: f64, h: f64, color: &str) {
        self.body.push_str(&format!(
            "<rect x=\"{x:.2}\" y=\"{y_top:.2}\" width=\"{w:.2}\" height=\"{h:.2}\" \
             fill=\"{color}\" stroke=\"none\"/>"
        ));
    }

    fn hline(&mut self, x0: f64, x1: f64, y: f64, color: &str, dash: &str, width: f64) {
        self.body.push_str(&format!(
            "<line x1=\"{x0:.2}\" y1=\"{y:.2}\" x2=\"{x1:.2}\" y2=\"{y:.2}\" \
             stroke=\"{color}\" stroke-width=\"{width:.1}\" stroke-dasharray=\"{dash}\"/>"
        ));
    }

    fn text(&mut self, x: f64, y: f64, size: u32, color: &str, anchor: &str, content: &str) {
        self.body.push_str(&format!(
            "<text x=\"{x:.2}\" y=\"{y:.2}\" fill=\"{color}\" font-size=\"{size}\" \
             font-family=\"sans-serif\" text-anchor=\"{anchor}\">{}</text>",
            escape(content)
        ));
    }

    fn finish(mut self) -> String {
        self.body.push_str("</svg>");
        self.body
    }
}

// ---------------------------------------------------------------------------
// Figures
// ---------------------------------------------------------------------------

/// Winner meridian overlaid on the straight bore, radius in mm.
fn profile_svg(profile: &[(f64, f64)], r0: f64, duct_l: f64) -> String {
    let mut canvas = Canvas::new("Port profile: straight bore vs optimised winner r(x)");
    let (x0, y0, x1, y1) = (120.0, 930.0, 1800.0, 150.0);
    canvas.axes(x0, y0, x1, y1);
    let (mut lo, mut hi) = (r0 * 1000.0, r0 * 1000.0);
    for &(_, r) in profile {
        lo = lo.min(r * 1000.0);
        hi = hi.max(r * 1000.0);
    }
    let pad = (hi - lo).max(0.05) * 0.25 + 0.05;
    lo -= pad;
    hi += pad;
    let sx = |x: f64| x0 + (x1 - x0) * x / duct_l;
    let sy = |r_mm: f64| y0 + (y1 - y0) * (r_mm - lo) / (hi - lo);
    canvas.polyline(
        &[(sx(0.0), sy(r0 * 1000.0)), (sx(duct_l), sy(r0 * 1000.0))],
        GREY,
        4.0,
        Some("18 12"),
    );
    let winner: Vec<(f64, f64)> = profile
        .iter()
        .map(|(x, r)| (sx(*x), sy(r * 1000.0)))
        .collect();
    canvas.polyline(&winner, CYAN, 5.0, None);
    let throat = profile
        .iter()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .copied()
        .unwrap_or((0.0, r0));
    canvas.text(
        sx(throat.0),
        sy(throat.1 * 1000.0) - 24.0,
        28,
        CYAN,
        "middle",
        &format!(
            "throat {:.2} mm ({:.1}% dip)",
            throat.1 * 1000.0,
            100.0 * (r0 - throat.1) / r0
        ),
    );
    canvas.text(1500.0, 230.0, 30, GREY, "start", "–– – straight bore");
    canvas.text(1500.0, 275.0, 30, CYAN, "start", "—— winner spline");
    canvas.text(
        120.0,
        990.0,
        24,
        AXIS,
        "start",
        "x: 0–0.1 m (inlet → outlet)",
    );
    canvas.text(
        1800.0,
        990.0,
        24,
        AXIS,
        "end",
        &format!("r: {lo:.1}–{hi:.1} mm"),
    );
    canvas.finish()
}

/// Equal-airflow deltas: THD relative change and noise-floor delta per drive.
fn comparison_svg(r7: &serde_json::Value) -> Result<(String, String), Box<dyn Error>> {
    let levels = r7
        .get("equal_airflow")
        .and_then(serde_json::Value::as_array)
        .ok_or("step7 report lacks equal_airflow")?;
    let mut rows = Vec::new();
    for level in levels {
        rows.push((
            num(level, "v_straight_v")?,
            100.0 * num(level, "d_thd_rel")?,
            num(level, "d_noise_db")?,
        ));
    }
    let mut canvas = Canvas::new("Winner vs straight at equal airflow (matched drive)");
    for (panel, &(v, thd, noise)) in rows.iter().enumerate() {
        let top = 170.0 + panel as f64 * 400.0;
        let mid = top + 150.0;
        canvas.text(
            120.0,
            top + 10.0,
            30,
            AXIS,
            "start",
            &format!("straight drive {v} V"),
        );
        canvas.axes(120.0, mid, 1800.0, mid);
        for (index, &(label, value, unit, good_when_negative)) in
            [("THD", thd, "%", true), ("noise", noise, "dB", true)]
                .iter()
                .enumerate()
        {
            let span = value.abs().max(1.0) * 1.35;
            let cx = 560.0 + index as f64 * 760.0;
            let half_len = 620.0 * value.abs() / span;
            let color = if (value < 0.0) == good_when_negative {
                GREEN
            } else {
                RED
            };
            let (bx, w) = if value < 0.0 {
                (cx - half_len, half_len)
            } else {
                (cx, half_len)
            };
            canvas.bar(bx, mid - 34.0, w.max(2.0), 68.0, color);
            canvas.hline(cx - 640.0, cx + 640.0, mid, AXIS, "4 6", 2.0);
            canvas.text(cx, mid - 52.0, 28, AXIS, "middle", label);
            canvas.text(
                cx,
                mid + 92.0,
                30,
                color,
                "middle",
                &format!("{value:+.2} {unit}"),
            );
        }
    }
    canvas.text(
        120.0,
        990.0,
        24,
        AXIS,
        "start",
        "green = winner better · red = winner worse · null result: noise rises at both drives",
    );
    let caption = format!(
        "Equal-airflow deltas across {} drive levels: THD relative change and integrated-noise-floor delta (winner − straight)",
        rows.len()
    );
    Ok((canvas.finish(), caption))
}

/// Integrated FW–H/Lighthill noise floor at 1 m on-axis per drive level.
fn noise_svg(r8: &serde_json::Value) -> Result<(String, String), Box<dyn Error>> {
    let levels = r8
        .get("noise")
        .and_then(serde_json::Value::as_array)
        .ok_or("step8 report lacks noise")?;
    let mut rows = Vec::new();
    for level in levels {
        rows.push((
            num(level, "voltage_peak_v")?,
            num(obj(level, "straight")?, "integrated_noise_floor_db")?,
            num(obj(level, "winner")?, "integrated_noise_floor_db")?,
        ));
    }
    let floor = rows
        .iter()
        .flat_map(|&(_, s, w)| [s, w])
        .fold(f64::INFINITY, f64::min)
        - 4.0;
    let ceil = rows
        .iter()
        .flat_map(|&(_, s, w)| [s, w])
        .fold(f64::NEG_INFINITY, f64::max)
        + 4.0;
    let mut canvas = Canvas::new("Integrated noise floor at 1 m on-axis (dB re 20 µPa)");
    let (x0, y0, x1, y1) = (160.0, 900.0, 1800.0, 180.0);
    canvas.axes(x0, y0, x1, y1);
    let sy = |db: f64| y0 + (y1 - y0) * (db - floor) / (ceil - floor);
    let slot = (x1 - x0) / rows.len() as f64;
    for (index, &(v, straight, winner)) in rows.iter().enumerate() {
        let cx = x0 + slot * (index as f64 + 0.5);
        canvas.bar(cx - 150.0, sy(straight), 130.0, y0 - sy(straight), GREY);
        canvas.bar(cx + 20.0, sy(winner), 130.0, y0 - sy(winner), CYAN);
        canvas.text(
            cx - 85.0,
            sy(straight) - 14.0,
            26,
            GREY,
            "middle",
            &format!("{straight:.1}"),
        );
        canvas.text(
            cx + 85.0,
            sy(winner) - 14.0,
            26,
            CYAN,
            "middle",
            &format!("{winner:.1}"),
        );
        let delta = winner - straight;
        canvas.text(
            cx,
            y0 + 44.0,
            28,
            if delta < 0.0 { GREEN } else { RED },
            "middle",
            &format!("{v} V  ({delta:+.2} dB)"),
        );
    }
    canvas.text(220.0, 230.0, 30, GREY, "start", "▪ straight");
    canvas.text(220.0, 275.0, 30, CYAN, "start", "▪ winner");
    canvas.text(
        160.0,
        990.0,
        24,
        AXIS,
        "start",
        "lower is quieter · 96 tail snapshots per level",
    );
    let caption = format!(
        "Straight vs winner integrated noise floor at 1 m on-axis across {} drive levels",
        rows.len()
    );
    Ok((canvas.finish(), caption))
}

/// Finest-pair refinement changes against the 2% promotion bar.
fn convergence_svg(r9: &serde_json::Value) -> Result<(String, String), Box<dyn Error>> {
    let promotion = obj(r9, "promotion")?;
    let spatial_max = 100.0 * num(promotion, "max_spatial_observable_relative_change")?;
    let temporal_max = 100.0 * num(promotion, "max_temporal_observable_relative_change")?;
    let spatial_proxy = finest_proxy_change(r9, "spatial_levels")?;
    let temporal_proxy = finest_proxy_change(r9, "temporal_levels")?;
    let bars = [
        ("spatial max", spatial_max),
        ("temporal max", temporal_max),
        ("noise proxy, spatial", spatial_proxy),
        ("noise proxy, temporal", temporal_proxy),
    ];
    let top = bars.iter().map(|&(_, v)| v).fold(2.0, f64::max) * 1.25;
    let mut canvas = Canvas::new("Refinement changes vs the 2% promotion bar (winner, Vmax)");
    let (x0, y0, x1, y1) = (160.0, 900.0, 1800.0, 180.0);
    canvas.axes(x0, y0, x1, y1);
    let sy = |pct: f64| y0 + (y1 - y0) * pct / top;
    canvas.hline(x0, x1, sy(2.0), RED, "16 10", 3.0);
    canvas.text(x0 + 10.0, sy(2.0) - 14.0, 28, RED, "start", "2% bar");
    let slot = (x1 - x0) / bars.len() as f64;
    for (index, &(label, value)) in bars.iter().enumerate() {
        let cx = x0 + slot * (index as f64 + 0.5);
        let color = if value < 2.0 { GREEN } else { RED };
        canvas.bar(cx - 110.0, sy(value), 220.0, y0 - sy(value), color);
        canvas.text(
            cx,
            sy(value) - 14.0,
            30,
            color,
            "middle",
            &format!("{value:.2}%"),
        );
        canvas.text(cx, y0 + 44.0, 26, AXIS, "middle", label);
    }
    canvas.text(
        160.0,
        990.0,
        24,
        AXIS,
        "start",
        "spatial [0.016, 0.012, 0.01] m · temporal [252, 504] samples/cycle · THD by 1e-4 floor",
    );
    Ok((
        canvas.finish(),
        "Winner finest-pair refinement changes against the 2% promotion bar".into(),
    ))
}

fn finest_proxy_change(r9: &serde_json::Value, key: &str) -> Result<f64, Box<dyn Error>> {
    let levels = r9
        .get(key)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("step9 report lacks {key}"))?;
    if levels.len() < 2 {
        return Err(format!("step9 ladder {key} has too few levels").into());
    }
    let coarse = num(&levels[levels.len() - 2], "noise_proxy")?;
    let fine = num(&levels[levels.len() - 1], "noise_proxy")?;
    Ok(100.0 * (fine - coarse).abs() / coarse.abs().max(f64::MIN_POSITIVE))
}

// ---------------------------------------------------------------------------
// Provenance
// ---------------------------------------------------------------------------

fn write_provenance(
    out_dir: &Path,
    r7: &serde_json::Value,
    f_target: f64,
) -> Result<(), Box<dyn Error>> {
    let mut entries: Vec<(String, serde_json::Value)> = Vec::new();
    let mut names: Vec<String> = std::fs::read_dir(out_dir)?
        .filter_map(Result::ok)
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter(|name| name != "gallery-provenance.json")
        .collect();
    names.sort();
    for name in &names {
        let data = std::fs::read(out_dir.join(name))?;
        entries.push((
            name.clone(),
            serde_json::json!(format!("fnv1a64:{:016x}", fnv1a64(&data))),
        ));
    }
    let git_sha = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|sha| sha.trim().to_owned())
        .filter(|sha| !sha.is_empty())
        .unwrap_or_else(|| "unavailable".to_owned());
    let case = obj(r7, "case")?;
    let criterion_1 = obj(r7, "criterion_1")?;
    let criterion_2 = obj(r7, "criterion_2")?;
    let manifest = serde_json::json!({
        "schema": SCHEMA,
        "git_sha": git_sha,
        "case": {
            "port_radius_r0_m": num(case, "port_radius_r0_m")?,
            "port_length_l_m": num(case, "port_length_l_m")?,
            "f_target_hz": f_target,
            "n_control_points": r7.get("n_control_points").and_then(serde_json::Value::as_u64).ok_or("step7 report lacks n_control_points")?,
        },
        "winner_u": arr(r7, "winner_u")?,
        "verdicts": {
            "criterion_1": str_field(criterion_1, "outcome")?,
            "criterion_2": str_field(criterion_2, "outcome")?,
        },
        "units": {"length": "m", "radius_plot": "mm", "spl": "dB re 20 uPa"},
        "provenance": "frozen PORT-OPTIM campaign evidence; SVG plots are derived from accompanying CSV data; winner_profile.csv is a byte copy of the Step-7 archive",
        "checksums": entries.into_iter().collect::<serde_json::Map<String, serde_json::Value>>(),
    });
    // The checksum map covers every sibling asset; the provenance file itself
    // is excluded since embedding its own hash is not a fixed point.
    std::fs::write(
        out_dir.join("gallery-provenance.json"),
        serde_json::to_vec_pretty(&manifest).map_err(std::io::Error::other)?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::run;

    /// The emitter renders the full asset set twice, byte-identically.
    #[test]
    fn gallery_emission_is_deterministic() {
        let root =
            std::env::temp_dir().join(format!("port-optim-gallery-test-{}", std::process::id()));
        let first = root.join("a");
        let second = root.join("b");
        run(&first).expect("first emission must succeed");
        run(&second).expect("second emission must succeed");
        // gallery.json embeds the source-tree git SHA, which is stable within
        // one test process; every asset (including it) must match exactly.
        let mut names: Vec<String> = std::fs::read_dir(&first)
            .expect("gallery dir must list")
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().into_string().expect("utf8 names"))
            .collect();
        names.sort();
        assert_eq!(
            names,
            [
                "comparison_equal_airflow.csv",
                "comparison_equal_airflow.svg",
                "convergence.csv",
                "convergence.svg",
                "flow3d.png",
                "flow_distribution.csv",
                "gallery-provenance.json",
                "gallery.json",
                "noise_floor.csv",
                "noise_floor.svg",
                "profile_shape.svg",
                "winner_profile.csv",
            ]
        );
        for name in &names {
            let (a, b) = (
                std::fs::read(first.join(name)).expect("read first"),
                std::fs::read(second.join(name)).expect("read second"),
            );
            assert_eq!(a, b, "asset {name} must be deterministic");
        }
        let manifest: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(first.join("gallery.json")).expect("read manifest"),
        )
        .expect("manifest must parse");
        assert_eq!(manifest["figures"].as_array().map(Vec::len), Some(5));
        std::fs::remove_dir_all(&root).ok();
    }
}
