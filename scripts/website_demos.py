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
        write_ply(ply_dir / f"frame_{index:04d}.ply", points, faces, colours)
    return ply_dir


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
            derived = mesh.compute_derivative(scalars="velocity", vorticity=True)
            values = np.asarray(derived.point_data["vorticity"])[:, 2]
            field_values = np.abs(values)
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
            derived = surface.compute_derivative(scalars="velocity", vorticity=True)
            surface_values = np.abs(np.asarray(derived.point_data["vorticity"])[:, 2])
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
