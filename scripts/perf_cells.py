#!/usr/bin/env python3
"""Per-platform, per-cell performance records.

Generates `perf/<platform>-<cell>.md` from the committed benchmark
harnesses. Run on each platform, then commit the results:

    just perf-m1     # Apple M1 Mac (perf-m2, perf-m4, ... likewise)
    just perf-x86    # Linux x86-64
    just perf-cuda   # Linux x86-64 with NVIDIA GPU

Cells: blas, lapack, sparse-matrix, dense-solvers, sparse-solvers.

Harness mapping:
  blas           cargo bench -p sonium-compute blas{1,2,3}_compare
                 (vendor arms: Accelerate on macOS, OpenBLAS with
                 bench-openblas on Linux, cuBLAS with cuda)
  lapack         direct-solver-perf dense section (LU/getrf path)
  sparse-matrix  roofline_preflight SpMV + iterative-run effective BW
  dense-solvers  direct-solver-perf dense section (end-to-end)
  sparse-solvers direct-solver-perf sparse section + iterative JSON summary

Environment:
  PERF_QUICK=1            bounded run (quick criterion, small sizes)
  PERF_CRITERION_ARGS     extra args after `cargo bench ... --` (default:
                          `-- --quick` in quick mode, else none)
  PERF_CRITERION_FILTER   positional criterion name filter (limits benches)
  PERF_OUT_DIR            output dir (default: perf/)
  PERF_KEEP_WORK=1        keep raw harness outputs under <out>/.work/
  DIRECT_PERF_* / ITERATIVE_PERF_*  passthrough to the existing scripts
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform as py_platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PERF_DIR = REPO_ROOT / "perf"
TARGET_DIR = REPO_ROOT / "target"

# Apple Silicon chips with committed records. Adding a new chip is a
# one-line change here plus a `just perf-m<N>` recipe (and a row in
# perf/README.md); auto-detection parses any `Apple M<N>` brand string,
# so an unlisted future chip still generates correctly titled files.
APPLE_PLATFORMS = ("m1", "m2", "m3", "m4", "m5")
PLATFORMS = APPLE_PLATFORMS + ("x86", "cuda")
CELLS = ("blas", "lapack", "sparse-matrix", "dense-solvers", "sparse-solvers")

PLATFORM_TITLE = {
    **{chip: f"Apple {chip.upper()} (macOS, aarch64)"
       for chip in APPLE_PLATFORMS},
    "x86": "x86-64 (Linux)",
    "cuda": "CUDA (NVIDIA GPU, Linux)",
}
CELL_TITLE = {
    "blas": "Dense BLAS",
    "lapack": "Dense LAPACK",
    "sparse-matrix": "Sparse matrix",
    "dense-solvers": "Dense solvers",
    "sparse-solvers": "Sparse solvers + preconditioners",
}
# Preferred vendor arm id per platform for the criterion blas benches.
VENDOR_ARM = {**{chip: "accelerate" for chip in APPLE_PLATFORMS},
              "x86": "openblas", "cuda": "cublas"}


def sh(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=REPO_ROOT, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          **kwargs)


def git_sha() -> str:
    r = sh(["git", "rev-parse", "--short=12", "HEAD"])
    sha = r.stdout.strip() if r.returncode == 0 else "unknown"
    r = sh(["git", "status", "--porcelain"])
    if r.returncode == 0 and r.stdout.strip():
        sha += "-dirty"
    return sha


def rustc_version() -> str:
    r = sh(["rustc", "--version"])
    return r.stdout.strip() if r.returncode == 0 else "unknown"


def host_line() -> str:
    uname = py_platform.uname()
    mem = "unknown"
    try:
        kb = int(Path("/proc/meminfo").read_text().split("MemTotal:")[1]
                 .split()[0])
        mem = f"{kb / 1024 / 1024:.0f} GB"
    except Exception:
        pass
    if mem == "unknown" and py_platform.system() == "Darwin":
        r = sh(["/usr/sbin/sysctl", "-n", "hw.memsize"])
        if r.returncode == 0 and r.stdout.strip().isdigit():
            mem = f"{int(r.stdout.strip()) / 1024**3:.0f} GB"
    return (f"{uname.system} {uname.release} {uname.machine} · "
            f"{os.cpu_count()} logical CPUs · mem {mem}")


def gpu_line() -> str:
    if shutil.which("nvidia-smi") is None:
        return "no nvidia-smi"
    r = sh(["nvidia-smi", "-L"])
    if r.returncode != 0:
        return "nvidia-smi failed"
    return "; ".join(l.strip() for l in r.stdout.splitlines() if l.strip())


def detect_platform() -> str:
    uname = py_platform.uname()
    if uname.system == "Darwin" and uname.machine == "arm64":
        return apple_chip_id() or "m4"
    return "x86"


def apple_chip_id() -> str | None:
    """Platform id for this Mac's chip (`Apple M4 Pro` -> `m4`).

    Parses any `Apple M<N>` brand string, so future chips work before
    they are added to `APPLE_PLATFORMS`; returns None off Apple Silicon.
    """
    r = sh(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"])
    if r.returncode != 0:
        return None
    brand = r.stdout.strip().lower()
    if not brand.startswith("apple m"):
        return None
    digits = ""
    for ch in brand[len("apple m"):]:
        if not ch.isdigit():
            break
        digits += ch
    return f"m{digits}" if digits else None


def platform_title(platform: str) -> str:
    return PLATFORM_TITLE.get(
        platform, f"Apple {platform.upper()} (macOS, aarch64)")


def header(platform: str, cell: str, commands: list[str], quick: bool,
           status: str) -> str:
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds")
    lines = [
        f"# {platform_title(platform)} — {CELL_TITLE[cell]}",
        "",
        f"_Generated {now} from commit `{git_sha()}`._",
        "",
        f"- status: {status}",
        f"- host: {host_line()}",
        f"- rustc: {rustc_version()}",
    ]
    if platform == "cuda":
        lines.append(f"- gpu: {gpu_line()}")
    if quick:
        lines.append("- mode: QUICK (bounded run; not comparable to full runs)")
    lines += ["", "## Commands", ""]
    lines += [f"```text\n{c}\n```" for c in commands]
    lines += [""]
    return "\n".join(lines)


def stub(platform: str, cell: str, reason: str, quick: bool) -> str:
    body = header(platform, cell, [], quick,
                  f"not measured ({reason})")
    body += ("## Results\n\n"
             f"Not measured on this run: {reason}.\n")
    return body


# --- criterion parsing -------------------------------------------------

def criterion_medians() -> dict[tuple[str, ...], float]:
    """Map criterion benchmark path -> median ns from estimates.json."""
    out: dict[tuple[str, ...], float] = {}
    for path in TARGET_DIR.joinpath("criterion").rglob("new/estimates.json"):
        try:
            data = json.loads(path.read_text())
            median_ns = float(data["median"]["point_estimate"])
        except (OSError, ValueError, KeyError):
            continue
        rel = path.parent.parent.relative_to(
            TARGET_DIR / "criterion").parts
        out[rel] = median_ns
    return out


def fmt_time(ns: float) -> str:
    if ns < 1e3:
        return f"{ns:.1f} ns"
    if ns < 1e6:
        return f"{ns / 1e3:.2f} µs"
    if ns < 1e9:
        return f"{ns / 1e6:.3f} ms"
    return f"{ns / 1e9:.3f} s"


def sort_key(label: str):
    nums = [int(t) for t in label.replace("x", " ").split() if t.isdigit()]
    return (0, nums, "") if nums else (1, [], label)


def ratio_cell(sonium_ns: float | None, vendor_ns: float | None) -> str:
    if sonium_ns is None or vendor_ns is None or vendor_ns <= 0:
        return "—"
    return f"{sonium_ns / vendor_ns:.2f}×"


# --- collectors --------------------------------------------------------

class Work:
    def __init__(self, keep: bool, out_dir: Path):
        self.keep = keep
        self.out_dir = out_dir
        if keep:
            self.root = out_dir / ".work"
            self.root.mkdir(parents=True, exist_ok=True)
        else:
            self.root = Path(tempfile.mkdtemp(prefix="perf-cells-"))

    def path(self, name: str) -> Path:
        return self.root / name

    def __del__(self):
        if not self.keep and self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)


def run_cargo_bench(benches: list[str], features: list[str], work: Work,
                    tag: str, quick: bool,
                    package: str = "sonium-compute"
                    ) -> tuple[bool, str, str]:
    cargo = os.environ.get("SONIUM_CARGO", "cargo")
    cmd = [cargo, "bench", "-p", package]
    for b in benches:
        cmd += ["--bench", b]
    if features:
        cmd += ["--features", ",".join(features)]
    extra = os.environ.get("PERF_CRITERION_ARGS")
    if extra:
        cmd += extra.split()
    elif quick:
        cmd += ["--", "--quick"]
    filt = os.environ.get("PERF_CRITERION_FILTER")
    if filt:
        if "--" not in cmd:
            cmd.append("--")
        cmd.append(filt)
    cmd_str = " ".join(cmd)
    print(f"+ {cmd_str}", flush=True)
    r = sh(cmd)
    log = work.path(f"{tag}.log")
    log.write_text(f"$ {cmd_str}\n\n--- stdout ---\n{r.stdout}\n"
                   f"--- stderr ---\n{r.stderr}\n")
    ok = r.returncode == 0
    if not ok:
        print(f"FAILED ({r.returncode}): {cmd_str}\n"
              f"see {log}", file=sys.stderr)
    return ok, cmd_str, r.stderr


def collect_blas(platform: str, work: Work, quick: bool):
    features: list[str] = []
    if platform == "x86":
        features = ["bench-openblas"]
    elif platform == "cuda":
        features = ["cuda"]
    ok, cmd, _ = run_cargo_bench(
        ["blas1_compare", "blas2_compare", "blas3_compare"],
        features, work, f"{platform}-blas", quick)
    return ok, [cmd], criterion_medians() if ok else {}


def render_blas(platform: str, medians: dict, commands: list[str],
                quick: bool) -> str:
    vendor = VENDOR_ARM[platform]
    # group -> label -> variant -> ns
    groups: dict[str, dict[str, dict[str, float]]] = {}
    for parts, ns in medians.items():
        if len(parts) < 3:
            continue
        group, variant = parts[0], parts[1]
        label = "/".join(parts[2:])
        groups.setdefault(group, {}).setdefault(label, {})[variant] = ns
    body = header(platform, "blas", commands, quick,
                  "measured" if groups else
                  "not measured (no criterion estimates found)")
    body += "## Results\n\n"
    if not groups:
        body += "No criterion estimates were produced by this run.\n"
        return body
    body += (f"Median of criterion samples; gap = sonium ÷ {vendor} "
             f"(1.00× is parity).\n\n")
    for group in sorted(groups):
        body += f"### `{group}`\n\n"
        body += (f"| shape | sonium | {vendor} | gap |\n"
                 f"|---|---|---|---|\n")
        for label in sorted(groups[group], key=sort_key):
            arms = groups[group][label]
            s = arms.get("sonium")
            v = arms.get(vendor)
            # Fall back to any other vendor arm present.
            if v is None:
                for alt in ("accelerate", "openblas", "cublas",
                            "metal", "oxiblas"):
                    if alt in arms:
                        v = arms[alt]
                        break
            body += (f"| {label} | "
                     f"{fmt_time(s) if s else '—'} | "
                     f"{fmt_time(v) if v else '—'} | "
                     f"{ratio_cell(s, v)} |\n")
        body += "\n"
    return body


def collect_direct(work: Work, tag: str, quick: bool):
    out = work.path(f"{tag}-direct.md")
    env = dict(os.environ)
    if quick:
        env.setdefault("DIRECT_PERF_DENSE_SIZES", "64,128")
        env.setdefault("DIRECT_PERF_SPARSE_GRIDS", "8,12")
        env.setdefault("DIRECT_PERF_REPEATS", "1")
    cmd = ["bash", "scripts/direct_solver_perf.sh", str(out)]
    cmd_str = " ".join(cmd)
    print(f"+ {cmd_str}", flush=True)
    r = subprocess.run(cmd, cwd=REPO_ROOT, text=True, env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log = work.path(f"{tag}-direct.log")
    log.write_text(f"$ {cmd_str}\n\n{r.stdout}\n")
    if r.returncode != 0 or not out.exists():
        print(f"FAILED ({r.returncode}): {cmd_str}\nsee {log}",
              file=sys.stderr)
        return False, [cmd_str], ""
    return True, [cmd_str], out.read_text()


def split_direct_sections(report: str) -> tuple[str, str]:
    """Return (dense-LU section, sparse-LU section) bodies."""
    dense, sparse = "", ""
    current: list[str] | None = None
    for line in report.splitlines(keepends=True):
        if line.startswith("## Dense LU"):
            current = []
            continue
        if line.startswith("## Sparse LU"):
            dense = "".join(current or [])
            current = []
            continue
        if line.startswith("## ") and current is not None:
            sparse = "".join(current)
            current = None
            continue
        if current is not None:
            current.append(line)
    if current is not None and not sparse and dense:
        sparse = "".join(current)
    return dense.strip(), sparse.strip()


def collect_lapack_bench(platform: str, work: Work, quick: bool):
    features: list[str] = []
    if platform == "x86":
        features = ["bench-openblas"]
    elif platform == "cuda":
        features = ["cuda"]
    ok, cmd, _ = run_cargo_bench(
        ["lapack_compare"], features, work, f"{platform}-lapack", quick,
        package="sonium-solvers-direct")
    medians = {}
    if ok:
        all_medians = criterion_medians()
        lapack_groups = ("potrf_f64", "potrf_f32", "getrf_f64",
                         "getrf_f32", "geqrf_f64")
        medians = {p: ns for p, ns in all_medians.items()
                   if p and p[0] in lapack_groups}
    return ok, [cmd], medians


def render_lapack(platform: str, dense_section: str, commands: list[str],
                  quick: bool, bench_medians: dict | None = None,
                  bench_cmds: list[str] | None = None) -> str:
    body = header(platform, "lapack", commands + (bench_cmds or []),
                  quick, "measured")
    body += "## Results\n\n"
    if bench_medians:
        vendor = "cusolver" if platform == "cuda" else VENDOR_ARM[platform]
        groups: dict[str, dict[str, dict[str, float]]] = {}
        for parts, ns in bench_medians.items():
            if len(parts) < 3:
                continue
            group, variant = parts[0], parts[1]
            label = "/".join(parts[2:])
            groups.setdefault(group, {}).setdefault(label, {})[variant] = ns
        body += ("### Factorization kernels (`lapack_compare`)\n\n"
                 f"Median of criterion samples; gap = sonium ÷ {vendor}.\n\n")
        for group in sorted(groups):
            body += f"#### `{group}`\n\n"
            body += (f"| size | sonium | {vendor} | gap |\n"
                     f"|---|---|---|---|\n")
            for label in sorted(groups[group], key=sort_key):
                arms = groups[group][label]
                s = arms.get("sonium")
                v = arms.get(vendor)
                if v is None:
                    for alt in ("accelerate", "openblas", "cusolver"):
                        if alt in arms:
                            v = arms[alt]
                            break
                body += (f"| {label} | "
                         f"{fmt_time(s) if s else '—'} | "
                         f"{fmt_time(v) if v else '—'} | "
                         f"{ratio_cell(s, v)} |\n")
            body += "\n"
    body += ("### Dense LU via `direct-solver-perf`\n\n"
             "One factorization + one solve per timing; "
             "end-to-end reading lives in "
             f"`{platform}-dense-solvers.md`.\n\n")
    body += dense_section + "\n"
    return body


def render_dense_solvers(platform: str, dense_section: str,
                         commands: list[str], quick: bool) -> str:
    body = header(platform, "dense-solvers", commands, quick, "measured")
    body += ("## Results\n\n"
             "End-to-end dense LU: one factorization + one right-hand-side "
             "solve per timing.\n\n")
    body += dense_section + "\n"
    return body


def collect_roofline(platform: str, work: Work, quick: bool):
    features = ["cuda"] if platform == "cuda" else []
    ok, cmd, stderr = run_cargo_bench(
        ["roofline_preflight"], features, work,
        f"{platform}-roofline", quick)
    return ok, [cmd], stderr, criterion_medians() if ok else {}


def render_sparse_matrix(platform: str, medians: dict, roofline_stderr: str,
                         iter_summary: str, commands: list[str],
                         quick: bool) -> str:
    body = header(platform, "sparse-matrix", commands, quick, "measured")
    body += "## Results\n\n"
    # Parse sonium_roofline_preflight bytes= lines: op -> bytes.
    op_bytes: dict[str, int] = {}
    for line in roofline_stderr.splitlines():
        if "sonium_roofline_preflight" not in line:
            continue
        parts = dict(kv.split("=", 1) for kv in line.split()
                     if "=" in kv and kv.startswith(
                         ("backend=", "op=", "bytes=")))
        try:
            op_bytes[parts["op"]] = int(parts["bytes"])
        except (KeyError, ValueError):
            continue
    spmv_rows = [(p, ns) for p, ns in medians.items()
                 if "spmv" in "/".join(p).lower()]
    if spmv_rows:
        body += ("### CSR SpMV (roofline preflight)\n\n"
                 "| benchmark | median | bytes | eff. BW |\n"
                 "|---|---|---|---|\n")
        for parts, ns in sorted(spmv_rows):
            name = "/".join(parts)
            nbytes = op_bytes.get("csr_spmv")
            bw = (f"{nbytes / (ns / 1e9) / 1e9:.1f} GB/s"
                  if nbytes else "—")
            body += (f"| `{name}` | {fmt_time(ns)} | "
                     f"{nbytes if nbytes else '—'} | {bw} |\n")
        body += "\n"
    else:
        body += "No SpMV criterion estimates were produced by this run.\n\n"
    if iter_summary:
        body += ("### Effective SpMV bandwidth inside iterative solves\n\n"
                 + iter_summary + "\n")
    return body


def collect_iterative(work: Work, tag: str, quick: bool):
    out = work.path(f"{tag}-iterative.json")
    env = dict(os.environ)
    if quick:
        env.setdefault("ITERATIVE_PERF_QUICK", "1")
    cmd = ["bash", "scripts/iterative_solver_perf.sh", str(out)]
    cmd_str = " ".join(cmd)
    print(f"+ {cmd_str}", flush=True)
    r = subprocess.run(cmd, cwd=REPO_ROOT, text=True, env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log = work.path(f"{tag}-iterative.log")
    log.write_text(f"$ {cmd_str}\n\n{r.stdout}\n")
    if r.returncode != 0 or not out.exists():
        print(f"FAILED ({r.returncode}): {cmd_str}\nsee {log}",
              file=sys.stderr)
        return False, [cmd_str], {}
    return True, [cmd_str], json.loads(out.read_text())


def iterative_scaling_table(data: dict, solver: str, pre: str,
                            matrix: str) -> str:
    rows = [r for r in data.get("runs", [])
            if r.get("engine", "sonium") == "sonium"
            and r.get("solver") == solver
            and r.get("preconditioner") == pre
            and r.get("matrix") == matrix
            and r.get("scalar") == "f64"
            and r.get("rhs_count") == 1]
    if not rows:
        return ""
    out = ("| n | solve | iters | residual | eff. SpMV BW |\n"
           "|---|---|---|---|---|\n")
    for r in sorted(rows, key=lambda r: r["n"]):
        bw = r.get("estimated_spmv_gb_s")
        out += (f"| {r['n']} | {r['solve_ms']:.2f} ms | "
                f"{r['iterations']} | {r['residual']:.1e} | "
                f"{bw:.2f} GB/s |\n")
    return out


def render_sparse_solvers(platform: str, sparse_section: str,
                          iter_data: dict, commands: list[str],
                          quick: bool) -> str:
    body = header(platform, "sparse-solvers", commands, quick, "measured")
    body += "## Results\n\n"
    if sparse_section:
        body += ("### Sparse direct (LU)\n\n" + sparse_section + "\n\n")
    if iter_data:
        runs = iter_data.get("runs", [])
        pairs = sorted({(r.get("solver"), r.get("preconditioner"))
                        for r in runs})
        matrices = sorted({r.get("matrix") for r in runs})
        body += "### Iterative + preconditioners\n\n"
        body += (f"{len(runs)} runs · pairs: "
                 + ", ".join(f"{s}+{p}" for s, p in pairs) + "\n\n"
                 f"matrices: {', '.join(m for m in matrices if m)}\n\n")
        table = iterative_scaling_table(iter_data, "cg", "amg",
                                        "poisson_2d")
        if table:
            body += ("#### CG + AMG on Poisson 2-D (f64)\n\n" + table + "\n")
        petsc_table = iterative_petsc_table(iter_data)
        if petsc_table:
            body += ("#### Sonium vs PETSc KSP (f64, rhs=1)\n\n"
                     + petsc_table + "\n"
                     + "_Caveats: GMRES restart lengths differ between the "
                     "harnesses (sonium runs longer cycles; PETSc defaults "
                     "to restart 30), so iteration counts are not directly "
                     "comparable. PETSc AMG is HYPRE BoomerAMG used as a "
                     "solver; sonium AMG is its own smoothed-aggregation "
                     "preconditioner. Rows where either side hit `max_iter` "
                     "compare a converged solve against a truncated one._\n")
    return body


def iterative_petsc_table(data: dict) -> str:
    runs = data.get("runs", [])
    sonium = {(r.get("solver"), r.get("preconditioner"), r.get("matrix"),
               r.get("scalar"), r.get("n"), r.get("rhs_count")): r
              for r in runs if r.get("engine") == "sonium"}
    petsc = {(r.get("solver"), r.get("preconditioner"), r.get("matrix"),
              r.get("scalar"), r.get("n"), r.get("rhs_count")): r
             for r in runs if r.get("engine") == "petsc"}
    keys = sorted(set(sonium) & set(petsc),
                  key=lambda k: (k[2] or "", k[0] or "", k[1] or "", k[4] or 0))
    # Keep the record readable: largest n per (solver, precond, matrix).
    best: dict[tuple, tuple] = {}
    for key in keys:
        short = key[:4]
        if short not in best or (key[4] or 0) > (best[short][4] or 0):
            best[short] = key
    if not best:
        return ""
    out = ("| solver+precond | matrix | n | sonium | petsc | gap | "
           "iters s/p | conv s/p |\n"
           "|---|---|---|---|---|---|---|---|\n")
    for short in sorted(best):
        key = best[short]
        s, p = sonium[key], petsc[key]
        gap = (s["solve_ms"] / p["solve_ms"]
               if p["solve_ms"] > 0 else float("nan"))
        out += (f"| {s['solver']}+{s['preconditioner']} | {s['matrix']} | "
                f"{s['n']} | {s['solve_ms']:.2f} ms | "
                f"{p['solve_ms']:.2f} ms | {gap:.2f}× | "
                f"{s['iterations']}/{p['iterations']} | "
                f"{'✓' if s['converged'] else '✗'}/"
                f"{'✓' if p['converged'] else '✗'} |\n")
    validation = data.get("validation", {})
    if validation.get("total_comparisons"):
        out += (f"\nHarness validation: {validation.get('passed', 0)} passed, "
                f"{validation.get('failed', 0)} failed out of "
                f"{validation.get('total_comparisons', 0)} comparisons.\n")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate perf/<platform>-<cell>.md records.")
    parser.add_argument("platform", nargs="?", choices=PLATFORMS,
                        help="default: auto-detect (chip id such as m1 "
                             "on Apple Silicon, else x86)")
    parser.add_argument("--cells", default=",".join(CELLS),
                        help="comma-separated subset of cells")
    parser.add_argument("--out-dir", default=str(PERF_DIR))
    parser.add_argument("--quick", action="store_true",
                        help="bounded run (recorded in the header)")
    parser.add_argument("--skip-run", action="store_true",
                        help="write not-measured stubs without running "
                             "harnesses")
    args = parser.parse_args()

    platform = args.platform or detect_platform()
    if platform == "cuda" and shutil.which("nvidia-smi") is None \
            and not args.skip_run:
        print("cuda platform requires nvidia-smi / an NVIDIA GPU",
              file=sys.stderr)
        return 2
    cells = [c.strip() for c in args.cells.split(",")]
    unknown = [c for c in cells if c not in CELLS]
    if unknown:
        print(f"unknown cells: {unknown}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    quick = args.quick or os.environ.get("PERF_QUICK") == "1"
    work = Work(keep=os.environ.get("PERF_KEEP_WORK") == "1",
                out_dir=out_dir)

    failures: list[str] = []

    def write(cell: str, content: str):
        path = out_dir / f"{platform}-{cell}.md"
        path.write_text(content)
        print(f"wrote {path}")

    # Collect shared artifacts once.
    direct_ok, direct_cmds, direct_report = False, [], ""
    iter_ok, iter_cmds, iter_data = False, [], {}
    if not args.skip_run:
        if {"lapack", "dense-solvers", "sparse-solvers"} & set(cells):
            direct_ok, direct_cmds, direct_report = collect_direct(
                work, platform, quick)
        if {"sparse-matrix", "sparse-solvers"} & set(cells):
            iter_ok, iter_cmds, iter_data = collect_iterative(
                work, platform, quick)

    for cell in cells:
        if args.skip_run:
            write(cell, stub(platform, cell, "run skipped (--skip-run)",
                             quick))
            continue
        if cell == "blas":
            ok, cmds, medians = collect_blas(platform, work, quick)
            write(cell, render_blas(platform, medians, cmds, quick)
                  if ok else stub(platform, cell,
                                  "blas benches failed", quick))
            if not ok:
                failures.append(cell)
        elif cell == "lapack":
            bench_ok, bench_cmds, bench_medians = collect_lapack_bench(
                platform, work, quick)
            if direct_ok:
                dense, _ = split_direct_sections(direct_report)
                write(cell, render_lapack(platform, dense, direct_cmds,
                                          quick, bench_medians, bench_cmds))
            elif bench_ok and bench_medians:
                write(cell, render_lapack(platform, "(direct-solver-perf "
                                          "run failed; kernel table only)",
                                          [], quick, bench_medians,
                                          bench_cmds))
            else:
                write(cell, stub(platform, cell,
                                 "direct-solver-perf and lapack_compare "
                                 "failed", quick))
                failures.append(cell)
        elif cell == "dense-solvers":
            if direct_ok:
                dense, _ = split_direct_sections(direct_report)
                write(cell, render_dense_solvers(platform, dense,
                                                 direct_cmds, quick))
            else:
                write(cell, stub(platform, cell,
                                 "direct-solver-perf failed", quick))
                failures.append(cell)
        elif cell == "sparse-matrix":
            ok, cmds, stderr, medians = collect_roofline(
                platform, work, quick)
            eff = ""
            if iter_ok and iter_data:
                eff = iterative_scaling_table(
                    iter_data, "cg", "amg", "poisson_2d")
            if ok:
                write(cell, render_sparse_matrix(
                    platform, medians, stderr, eff,
                    cmds + iter_cmds, quick))
            else:
                write(cell, stub(platform, cell,
                                 "roofline benches failed", quick))
                failures.append(cell)
        elif cell == "sparse-solvers":
            if direct_ok or iter_ok:
                _, sparse = split_direct_sections(direct_report)
                write(cell, render_sparse_solvers(
                    platform, sparse, iter_data if iter_ok else {},
                    direct_cmds + iter_cmds, quick))
            else:
                write(cell, stub(platform, cell,
                                 "direct + iterative runs failed", quick))
                failures.append(cell)

    if failures:
        print(f"cells with failures: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
