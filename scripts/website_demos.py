#!/usr/bin/env python3
"""Generate the website simulation demo movies end to end.

Stages per demo: run the Rust demo driver (`qa-website-demos`, VTU
frame series), convert frames to vertex-coloured PLY (pyvista, IO
only), render a PNG sequence with Blender, and encode WebM/MP4 plus a
poster with ffmpeg. Afterwards the website data files (demo catalogue
and OpenCourant benchmark table) are regenerated.

Usage:
    python3 scripts/website_demos.py all
    python3 scripts/website_demos.py taylor-green crush
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SONIUM_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = SONIUM_ROOT.parent
DEFAULT_WORK_DIR = WORKSPACE_ROOT / "tmp" / "website-demos"
DEFAULT_OUT_DIR = SONIUM_ROOT.parent / "website" / "public" / "demos6"
DEFAULT_SITE_DATA = SONIUM_ROOT.parent / "website" / "src" / "data"

FPS = 12


def gallery_colour(t: np.ndarray) -> np.ndarray:
    """Mirror sonium_xem gallery `colour()`: blue->cyan->green->yellow->red."""
    t = np.clip(t, 0.0, 1.0)
    r = np.clip(1.5 * t - 0.5, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(2.0 * t - 1.0), 0.0, 1.0)
    b = np.clip(1.0 - 1.5 * t, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def write_ply(path: Path, points: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> None:
    points = np.ascontiguousarray(points, dtype=np.float32)
    colors = np.ascontiguousarray(colors, dtype=np.uint8)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\nend_header\n"
    )
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        for point, color in zip(points, colors):
            fh.write(struct.pack("<fffBBB", *point, *color))
        for face in faces:
            fh.write(struct.pack("<Biii", 3, *face))


@dataclass(frozen=True)
class DemoSpec:
    slug: str
    title: str
    capability: str
    blurb: str
    field_mode: str  # "vorticity" or a VTU point-field name
    direction: str
    material: str = "emission"
    distance_factor: float = 3.0
    poster_at: float = 0.6
    signed_field: bool = False
    volume_slices: bool = False
    rotor_geometry: bool = False


DEMOS: dict[str, DemoSpec] = {
    "taylor-green": DemoSpec(
        slug="taylor-green",
        title="Taylor-Green vortex decay",
        capability="Incompressible Navier-Stokes (BDF2, block Schur preconditioner)",
        blurb=(
            "A periodic vortex lattice decays under viscosity. Colour is "
            "vorticity magnitude; the solver is the same BDF2 Navier-Stokes "
            "path gated against the analytic Taylor-Green solution."
        ),
        field_mode="vorticity",
        direction="0.06,-0.10,1",
        distance_factor=3.7,
    ),
    "arch": DemoSpec(
        slug="arch",
        title="Arch snap-through",
        capability="Corotational shell elements (geometric nonlinearity, explicit dynamics)",
        blurb=(
            "A shallow steel arch is driven down at the crown until it "
            "snaps through to the inverted equilibrium. Colour is "
            "membrane strain intensity."
        ),
        field_mode="strain",
        direction="0.85,-1,0.42",
        material="shaded",
        distance_factor=3.0,
        poster_at=0.55,
    ),
    "room-wave": DemoSpec(
        slug="room-wave",
        title="Room impulse response",
        capability="Acoustic wave equation (P1 FEM, central differences, rigid walls)",
        blurb=(
            "A pressure pulse expands in a 3.6 x 2.8 x 2.4 m room and "
            "reflects off the rigid walls. Three mid-planes show the "
            "signed pressure; the field is a P1 finite-element wave "
            "solution at c = 343 m/s."
        ),
        field_mode="pressure",
        signed_field=True,
        volume_slices=True,
        # Near-face-on: the xy mid-plane reads as a rectangle with the
        # pulse rings; the equal-angle view collapses all three planes
        # into an illegible star.
        direction="0.3,-0.45,1.0",
        distance_factor=3.0,
        poster_at=0.5,
    ),
    "plate-modes": DemoSpec(
        slug="plate-modes",
        title="Plate vibration modes",
        capability="Generalized eigenproblem Kv=w2Mv (LOBPCG, shell elements)",
        blurb=(
            "The first four eigenmodes of a clamped steel plate, computed "
            "from the assembled shell stiffness and lumped mass matrices "
            "and animated over one period each."
        ),
        field_mode="disp",
        signed_field=True,
        direction="0.7,-1,0.55",
        material="shaded",
        distance_factor=3.0,
    ),
    "necking": DemoSpec(
        slug="necking",
        title="Necking bar",
        capability="Finite-strain J2 plasticity (Newton equilibrium, Voce hardening)",
        blurb=(
            "A steel bar pulled in tension necks down at a slight "
            "imperfection once the Considere strain is passed. Colour is "
            "axial logarithmic strain; the profile is the computed "
            "cross-section, revolved for display."
        ),
        field_mode="strain",
        direction="0.25,-1,0.35",
        material="shaded",
        distance_factor=3.9,
        poster_at=0.97,
    ),
    "propeller-noise": DemoSpec(
        slug="propeller-noise",
        title="Propeller tonal noise",
        capability="Aeroacoustics (compact loading dipoles, linear wave FEM, sponge free-field)",
        blurb=(
            "Three blades at 60 rev/s radiate at the 180 Hz blade-passage "
            "frequency. Each blade is a compact axial loading dipole; a P1 "
            "wave solve radiates the tonal field into a sponge-terminated "
            "free field. Colour is signed acoustic pressure on three "
            "mid-planes; the grey hub, blades and shaft show the rotor "
            "itself, windowed clear of the planes."
        ),
        field_mode="pressure",
        signed_field=True,
        volume_slices=True,
        rotor_geometry=True,
        direction="0.3,-0.45,1.0",
        distance_factor=3.0,
        poster_at=0.6,
    ),
    "thermal-spreading": DemoSpec(
        slug="thermal-spreading",
        title="Finned heatsink transient",
        capability="Heat conduction + natural convection + radiation (P1 FEM, backward Euler)",
        blurb=(
            "A chip under the base holds 95 C while the finned aluminium "
            "heatsink (k = 167 W/mK) warms over 120 s and loses heat from "
            "every exposed surface by natural convection (h = 8 W/m2K) "
            "and radiation (emissivity 0.85). Colour is temperature; the "
            "solve is sonium-fem's thermal conduction forms with "
            "backward-Euler stepping and per-step Robin coefficients."
        ),
        field_mode="temperature",
        direction="0.6,-1,0.75",
        distance_factor=3.0,
        poster_at=0.85,
    ),
    "melting-plate": DemoSpec(
        slug="melting-plate",
        title="Melting plate",
        capability="Phase-change heat conduction (apparent heat capacity, P1 FEM, backward Euler)",
        blurb=(
            "A 300 mm square aluminium-alloy plate with a 90 mm central "
            "hole is heated on the hole rim to 950 C. Past the 580-650 C "
            "melting range the alloy turns liquid and a melt front grows "
            "outward from the hole; latent heat (380 kJ/kg) is absorbed "
            "in the mushy zone while exposed faces lose heat by natural "
            "convection and T^4 radiation. Colour is liquid fraction: "
            "blue solid, red fully molten. Thermal-only model: the melt "
            "is tracked, but does not flow or slump."
        ),
        field_mode="liquid_fraction",
        direction="0.45,-0.7,1",
        distance_factor=3.0,
        poster_at=0.85,
    ),
    "sloshing-tank": DemoSpec(
        slug="sloshing-tank",
        title="Sloshing water tank",
        capability="Two-phase Navier-Stokes + Cahn-Hilliard phase field (BDF2, segregated coupling)",
        blurb=(
            "A 2 m tank filled with water to 0.20 m is shaken sideways "
            "at its first sloshing resonance (0.34 Hz, 100 mm amplitude) "
            "in the tank frame: fixed walls, oscillating horizontal "
            "gravity, phase-weighted body force, and a Cahn-Hilliard "
            "interface advected by the flow, with element-local "
            "viscosity interpolated from the phase field and "
            "phase-weighted gravity (inertia keeps the reference "
            "water density - full air/water inertia contrast is "
            "unstable at the contact line in this coupling). "
            "Colour is water volume fraction. Capillarity on the "
            "momentum is neglected (Bond number ~5e5 at this scale); "
            "a driven-amplitude sweep accompanies the film."
        ),
        field_mode="water_fraction",
        direction="0.04,-0.06,1",
        distance_factor=3.4,
        poster_at=0.82,
    ),
    "sloshing-turbulent": DemoSpec(
        slug="sloshing-turbulent",
        title="Sloshing water tank (turbulent)",
        capability="Two-phase Navier-Stokes + Cahn-Hilliard with WALE LES, CSF surface tension, compressible air, subgrid bubble parcels (BDF2, adaptive dt)",
        blurb=(
            "The sloshing tank driven past breaking onset: 250 mm "
            "amplitude at the first sloshing resonance. The wave "
            "steepens, plunges, and entrains air. WALE LES eddy "
            "viscosity replaces the resolved demo's interface-band "
            "viscosity; surface tension acts on the momentum; the "
            "air is weakly compressible so roof impact cushions "
            "instead of spiking; subgrid bubble parcels track "
            "entrained air below the interface scale. Colour is "
            "water volume fraction. Companion to the non-breaking "
            "sloshing-tank demo at 100 mm."
        ),
        field_mode="water_fraction",
        direction="0.04,-0.06,1",
        distance_factor=3.4,
        poster_at=0.82,
    ),
    "cylinder-wake": DemoSpec(
        slug="cylinder-wake",
        title="Cylinder vortex shedding",
        capability="Incompressible Navier-Stokes (BDF2, block Schur preconditioner)",
        blurb=(
            "Flow at Re = 100 past an immersed cylinder sheds a Karman "
            "vortex street. Colour is vorticity magnitude; the unsteady "
            "lift driving the street is the classic Aeolian dipole source. "
            "Same BDF2 Navier-Stokes path as the Taylor-Green gate."
        ),
        field_mode="vorticity",
        direction="0.04,-0.06,1",
        distance_factor=3.4,
        poster_at=0.7,
    ),
    "cylinder-mach1": DemoSpec(
        slug="cylinder-mach1",
        title="Cylinder at Mach 1",
        capability="Compressible Navier-Stokes (explicit P1-DG, SSPRK3, ideal gas)",
        blurb=(
            "Sonic freestream past a cylinder, solved with the explicit "
            "P1-DG compressible solver in conservative variables. A bow "
            "shock stands ahead of the nose, the flow stays subsonic in "
            "the pocket in front, then accelerates to locally supersonic "
            "speed at the shoulders. Colour is local Mach number. The "
            "incompressible cylinder demo cannot represent any of this: "
            "it has no density, energy, or sound speed."
        ),
        field_mode="mach",
        direction="0.04,-0.06,1",
        distance_factor=3.4,
        poster_at=0.7,
    ),
    "naca0012": DemoSpec(
        slug="naca0012",
        title="NACA 0012 at Mach 0.8",
        capability="Compressible Navier-Stokes (explicit P2-DG, HLLC, ideal gas)",
        blurb=(
            "Transonic flow over the classical NACA 0012 airfoil at "
            "Mach 0.8 and 1.25 degrees incidence (the AGARD inviscid "
            "benchmark), solved with the explicit P2-DG compressible "
            "solver. A supersonic pocket forms over the upper surface "
            "and ends in a shock near mid-chord. Colour is local Mach "
            "number."
        ),
        field_mode="mach",
        direction="0.04,-0.06,1",
        distance_factor=3.4,
        poster_at=0.7,
    ),
    "crush": DemoSpec(
        slug="crush",
        title="Thin-walled tube crush",
        capability="Q4 crash shells (Belytschko-Tsay, hourglass control), J2 plasticity, driven end-shortening",
        blurb=(
            "An aluminium tube is crushed against a rigid wall by "
            "prescribed end-shortening and folds progressively. Colour "
            "is membrane strain intensity; plastic hinges form where "
            "the folds localise."
        ),
        field_mode="strain",
        direction="1,-1.35,0.5",
        material="shaded",
        distance_factor=3.2,
        poster_at=0.9,
    ),
}


def run(command: list[str], label: str, cwd: Path | None = None) -> None:
    print(f"+ {label}", flush=True)
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=cwd)
    if completed.returncode != 0:
        sys.stdout.write(completed.stdout)
        raise SystemExit(f"stage failed: {label}")


def simulate(spec: DemoSpec, work: Path) -> Path:
    out = work / spec.slug / "vtu"
    out.mkdir(parents=True, exist_ok=True)
    # Drop stale frames: the sim overwrites in place, so a shorter rerun
    # would otherwise leave a tail of previous-run frames behind.
    for stale in out.glob("frame_*.vtu"):
        stale.unlink()
    (out / "manifest.json").unlink(missing_ok=True)
    cargo = os.environ.get("SONIUM_CARGO") or os.environ.get("CARGO", "cargo")
    run(
        [
            cargo,
            "run",
            "--release",
            "-q",
            "-p",
            "sonium-qa",
            "--features",
            "qa-binaries",
            "--bin",
            "qa-website-demos",
            "--",
            spec.slug,
            str(out),
        ],
        f"simulate {spec.slug}",
        cwd=SONIUM_ROOT,
    )
    return out


# Rotor constants mirror `propeller_noise` in website_demos.rs: 3 blades
# at 60 rev/s, tip radius 0.55 m, hub at the domain centre.
ROTOR_OMEGA = 2.0 * math.pi * 60.0
ROTOR_BLADES = 3
ROTOR_RADIUS = 0.55


def _append_rotor(points, faces, colours, centre, angle):
    """Append grey shaft/hub/blade geometry (rotated to `angle`) to a frame."""
    import pyvista as pv

    parts = [
        pv.Cylinder(center=tuple(centre), direction=(0, 0, 1), radius=0.035, height=2.4, resolution=24),
        pv.Cylinder(center=tuple(centre), direction=(0, 0, 1), radius=0.12, height=0.16, resolution=32),
    ]
    for blade in range(ROTOR_BLADES):
        box = pv.Box(bounds=(0.10, ROTOR_RADIUS, -0.07, 0.07, -0.012, 0.012))
        theta = angle + 2.0 * math.pi * blade / ROTOR_BLADES
        parts.append(box.rotate_z(math.degrees(theta), point=(0.0, 0.0, 0.0)).translate(tuple(centre)))
    rotor = parts[0].merge(parts[1:]).triangulate()
    rotor_points = np.asarray(rotor.points)
    rotor_faces = np.asarray(rotor.regular_faces)
    rotor_colours = np.full((len(rotor_points), 3), 150, dtype=np.uint8)
    offset = len(points)
    points = np.vstack([points, rotor_points])
    faces = np.vstack([faces, rotor_faces + offset]) if len(faces) else rotor_faces + offset
    colours = np.vstack([colours, rotor_colours])
    return points, faces, colours


def convert_slices(spec: DemoSpec, vtu_dir: Path, work: Path, manifest: dict) -> Path:
    """Volume VTU frames -> three orthogonal mid-plane slices as PLYs."""
    import pyvista as pv

    ply_dir = work / spec.slug / "ply"
    ply_dir.mkdir(parents=True, exist_ok=True)
    for stale in ply_dir.glob("frame_*.ply"):
        stale.unlink()
    meshes = [pv.read(vtu_dir / name) for name in manifest["frames"]]
    # RMS-referenced scale: the initial spike decays into a reverberant
    # field ~30x weaker, so peak-referencing would wash out the shell.
    rms = [
        float((mesh.point_data[spec.field_mode] ** 2).mean() ** 0.5) for mesh in meshes[5:]
    ]
    scale = 5.0 * float(np.median(rms)) if rms else 1.0
    for index, mesh in enumerate(meshes):
        center = mesh.center
        parts = []
        for normal in ("x", "y", "z"):
            sliced = mesh.slice(normal=normal, origin=center)
            if sliced.n_points:
                parts.append(sliced.triangulate())
        combined = parts[0].merge(parts[1:])
        points = np.asarray(combined.points)
        faces = np.asarray(combined.regular_faces)
        values = np.asarray(combined.point_data[spec.field_mode], dtype=float)
        colours = (gallery_colour(0.5 + 0.5 * values / scale) * 255).astype(np.uint8)
        if spec.rotor_geometry:
            # Window the planes clear of the rotor disk so the appended
            # geometry is not hidden behind (or z-fighting) the slices.
            centre = np.asarray(center, dtype=float)
            centroids = points[faces].mean(axis=1)
            keep = np.linalg.norm(centroids - centre, axis=1) > ROTOR_RADIUS + 0.13
            faces = faces[keep]
            points, faces, colours = _append_rotor(
                points, faces, colours, centre, ROTOR_OMEGA * manifest["times"][index]
            )
        write_ply(ply_dir / f"frame_{index:04d}.ply", points, faces, colours)
    return ply_dir


def nodal_vorticity_magnitude(surface) -> np.ndarray:
    """|dv/dx - du/dy| at surface nodes, area-weighted from P1 triangles.

    Deterministic replacement for pyvista's `compute_derivative`, which
    intermittently returns a zero field on part of the mesh.
    """
    points = np.asarray(surface.points, dtype=float)[:, :2]
    velocity = np.asarray(surface.point_data["velocity"], dtype=float)[:, :2]
    faces = np.asarray(surface.regular_faces)
    p0 = points[faces[:, 0]]
    e1 = points[faces[:, 1]] - p0
    e2 = points[faces[:, 2]] - p0
    det = e1[:, 0] * e2[:, 1] - e2[:, 0] * e1[:, 1]
    valid = np.abs(det) > 0.0
    det_safe = np.where(valid, det, 1.0)
    du = velocity[faces[:, 1]] - velocity[faces[:, 0]]
    dv = velocity[faces[:, 2]] - velocity[faces[:, 0]]

    def grad(component_diff_1: np.ndarray, component_diff_2: np.ndarray) -> tuple:
        gx = (e2[:, 1] * component_diff_1 - e1[:, 1] * component_diff_2) / det_safe
        gy = (-e2[:, 0] * component_diff_1 + e1[:, 0] * component_diff_2) / det_safe
        return gx, gy

    _, du_dy = grad(du[:, 0], dv[:, 0])
    dv_dx, _ = grad(du[:, 1], dv[:, 1])
    cell_vorticity = np.where(valid, dv_dx - du_dy, 0.0)
    weights = 0.5 * np.abs(det)
    numerator = np.zeros(len(points))
    denominator = np.zeros(len(points))
    for corner in range(3):
        np.add.at(numerator, faces[:, corner], weights * cell_vorticity)
        np.add.at(denominator, faces[:, corner], weights)
    nodal = np.where(denominator > 0.0, numerator / np.maximum(denominator, 1e-300), 0.0)
    return np.abs(nodal)


def convert(spec: DemoSpec, vtu_dir: Path, work: Path) -> Path:
    """VTU frames -> vertex-coloured PLY frames (fixed colour range)."""
    import pyvista as pv

    manifest = json.loads((vtu_dir / "manifest.json").read_text(encoding="utf-8"))
    if spec.volume_slices:
        return convert_slices(spec, vtu_dir, work, manifest)
    ply_dir = work / spec.slug / "ply"
    ply_dir.mkdir(parents=True, exist_ok=True)

    meshes = []
    fields: list[np.ndarray] = []
    for name in manifest["frames"]:
        mesh = pv.read(vtu_dir / name)
        if spec.field_mode == "vorticity":
            surface0 = mesh.extract_surface().triangulate()
            field_values = nodal_vorticity_magnitude(surface0)
        else:
            field_values = np.asarray(mesh.point_data[spec.field_mode], dtype=float)
        meshes.append(mesh)
        fields.append(field_values)
    scale = max(float(np.max(values)) for values in fields) or 1.0

    for stale in ply_dir.glob("frame_*.ply"):
        stale.unlink()
    for index, (mesh, values) in enumerate(zip(meshes, fields)):
        surface = mesh.extract_surface().triangulate()
        points = np.asarray(surface.points)
        faces = np.asarray(surface.regular_faces)
        if spec.field_mode == "vorticity":
            surface_values = nodal_vorticity_magnitude(surface)
        else:
            surface_values = np.asarray(surface.point_data[spec.field_mode], dtype=float)
        if spec.signed_field:
            colours = (gallery_colour(0.5 + 0.5 * surface_values / scale) * 255).astype(np.uint8)
        else:
            colours = (gallery_colour(surface_values / scale) * 255).astype(np.uint8)
        write_ply(ply_dir / f"frame_{index:04d}.ply", points, faces, colours)
    return ply_dir


def render(spec: DemoSpec, ply_dir: Path, work: Path) -> Path:
    png_dir = work / spec.slug / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    for stale in png_dir.glob("frame_*.png"):
        stale.unlink()
    args = [
        sys.executable,
        str(SONIUM_ROOT / "scripts" / "blender_movie_render.py"),
        "--frames-dir",
        str(ply_dir),
        "--out-dir",
        str(png_dir),
        "--direction",
        spec.direction,
        "--distance-factor",
        str(spec.distance_factor),
        "--material",
        spec.material,
    ]
    run(args, f"render {spec.slug}")
    return png_dir


def encode(spec: DemoSpec, png_dir: Path, out_dir: Path) -> None:
    target = out_dir / spec.slug
    target.mkdir(parents=True, exist_ok=True)
    frames = sorted(png_dir.glob("frame_*.png"))
    if not frames:
        raise SystemExit(f"no rendered frames for {spec.slug}")
    poster_index = min(len(frames) - 1, max(0, int(len(frames) * spec.poster_at)))
    run(
        ["cp", str(frames[poster_index]), str(target / "poster.png")],
        f"poster {spec.slug}",
    )
    run(
        [
            "ffmpeg", "-y", "-framerate", str(FPS), "-i", str(png_dir / "frame_%04d.png"),
            "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "30", "-pix_fmt", "yuv420p",
            str(target / "movie.webm"),
        ],
        f"encode webm {spec.slug}",
    )
    run(
        [
            "ffmpeg", "-y", "-framerate", str(FPS), "-i", str(png_dir / "frame_%04d.png"),
            "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(target / "movie.mp4"),
        ],
        f"encode mp4 {spec.slug}",
    )


def write_perf_data(site_data: Path) -> None:
    """Regenerate oc-benchmarks.json via the single-writer perf data script."""
    run(
        [
            sys.executable,
            str(WORKSPACE_ROOT / "website" / "scripts" / "generate_perf_data.py"),
            "--oc-only",
            "--site-data",
            str(site_data),
        ],
        "perf data (oc-benchmarks.json)",
    )


def write_site_data(site_data: Path, specs: list[DemoSpec]) -> None:
    site_data.mkdir(parents=True, exist_ok=True)
    # Merge into any existing catalogue so regenerating one demo never
    # drops the other five from the site; order follows DEMOS.
    existing: dict[str, dict] = {}
    catalogue_path = site_data / "demos6.json"
    if catalogue_path.exists():
        try:
            existing = {entry["slug"]: entry for entry in json.loads(catalogue_path.read_text(encoding="utf-8"))}
        except (json.JSONDecodeError, KeyError, TypeError):
            existing = {}
    for spec in specs:
        existing[spec.slug] = {
            "slug": spec.slug,
            "title": spec.title,
            "capability": spec.capability,
            "blurb": spec.blurb,
            "movie_webm": f"/demos6/{spec.slug}/movie.webm",
            "movie_mp4": f"/demos6/{spec.slug}/movie.mp4",
            "poster": f"/demos6/{spec.slug}/poster.png",
        }
    catalogue = [existing[slug] for slug in DEMOS if slug in existing]
    catalogue_path.write_text(json.dumps(catalogue, indent=2) + "\n", encoding="utf-8")
    write_perf_data(site_data)
    print(f"site data written to {site_data}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demos", nargs="+", help="demo slugs or 'all'")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--site-data", type=Path, default=DEFAULT_SITE_DATA)
    args = parser.parse_args(argv[1:])

    slugs = list(DEMOS) if args.demos == ["all"] else args.demos
    specs: list[DemoSpec] = []
    for slug in slugs:
        spec = DEMOS.get(slug)
        if spec is None:
            print(f"unknown demo {slug!r}; known: {sorted(DEMOS)}", file=sys.stderr)
            return 2
        specs.append(spec)

    for spec in specs:
        vtu_dir = simulate(spec, args.work_dir)
        ply_dir = convert(spec, vtu_dir, args.work_dir)
        png_dir = render(spec, ply_dir, args.work_dir)
        encode(spec, png_dir, args.out_dir)
    write_site_data(args.site_data, specs)
    print("website demos done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
