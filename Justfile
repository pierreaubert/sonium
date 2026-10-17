# Sonium — Multi-Physics simlator

# Use `mbx` (cargo wrapper) when installed, plain `cargo` otherwise.
cargo := `if command -v mbx >/dev/null 2>&1; then echo mbx; else echo cargo; fi`

# --- Build ---

default:
	just --list

check:
    {{cargo}} check

# Check every native package in the workspace, including the QA harness.
check-all:
    {{cargo}} check --workspace

clippy:
    PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 {{cargo}} clippy --workspace -- -D warnings -A clippy::needless_range_loop -A clippy::too_many_arguments -A clippy::type_complexity -A clippy::large_enum_variant

dev:
    {{cargo}} build --workspace

build:
    {{cargo}} build --release

dev-crate crate:
    {{cargo}} build -p {{crate}}

build-crate crate:
    {{cargo}} build -p {{crate}} --release

# --- Tests ---

# Run all tests across native crates
[group('test')]
test:
    {{cargo}} test --release --features="cli"

[group('test')]
test-all:
    {{cargo}} test --workspace --all-targets

# Broad test coverage. Two knobs control core use: `threads` sets nextest
# scheduler slots (default: all CPUs); RAYON/OpenBLAS env caps the inner
# thread pool each test process spawns, so 128 concurrent tests don't each
# grab 128 threads. Heavies wanting more inner threads: run with e.g.
# RAYON_NUM_THREADS=64 just ntest.
[group('test')]
ntest threads="num-cpus":
    RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-8}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}" {{cargo}} nextest run --release --no-fail-fast --workspace --all --features="cli" --test-threads {{threads}}

# Run tests for a specific crate
[group('test')]
test-crate crate:
    {{cargo}} test -p {{crate}} --release

# --- Applications ---

[group('run')]
run-speaker +args:
    {{cargo}} run --release -p sonium-sim-speaker --bin speaker-sim -- {{args}}

[group('run')]
run-room +args:
    {{cargo}} run --release -p sonium-sim-room --bin generate-roomeq-data -- {{args}}

[group('run')]
run-pcfan +args:
    {{cargo}} run --release -p sonium-sim-pcfan --bin pcfan-opt -- {{args}}

# Hermetic Wolfram-derived solver/eigensystem contracts.
qa-wolfram-confidence:
    {{just_executable()}} --justfile crates/sonium-qa/Justfile --working-directory crates/sonium-qa qa-wolfram-confidence
# Deterministic RoomSim website assets (SVG + source data + manifest).
[group('gallery')]
roomsim-gallery dir:
    {{cargo}} run --release -p sonium-roomsim --example web_gallery -- {{dir}}

# Export the canonical XEM JSON Schema artifacts used by the compatibility gate.
[group('qa')]
xem-schema dir='schemas':
    {{cargo}} run -p sonium-xem --example export_schema -- {{dir}}

# Run the dependency-free Python and Node/TypeScript-compatible XEM readers
# against both published golden envelopes.
[group('qa')]
xem-interop:
    python3 crates/sonium-xem/interop/read_room_config.py crates/sonium-xem/fixtures/room-config-v1.json
    node crates/sonium-xem/interop/read_room_config.ts crates/sonium-xem/fixtures/room-config-v1.json
    python3 crates/sonium-xem/interop/read_multiphysics_problem.py crates/sonium-xem/fixtures/multiphysics-problem-v1.json
    node crates/sonium-xem/interop/read_multiphysics_problem.ts crates/sonium-xem/fixtures/multiphysics-problem-v1.json

# Release-only external-reader smoke. Requires Python's h5py in the runner;
# it verifies the generated SOFA with HDF5 rather than sofa-reader and parses
# VTU/PVD independently with ElementTree. ParaView visual smoke remains a
# separate capability because it is not available on every release runner.
[group('qa')]
xem-external-interop dir:
    {{cargo}} run -p sonium-xem --example generate_external_formats --features io-sofa -- {{dir}}
    python3 crates/sonium-xem/interop/read_external_formats.py {{dir}}

# Generate all six website simulation films end-to-end (simulate via
# qa-website-demos -> VTU->PLY -> Blender PNGs -> WebM/MP4+poster) and
# refresh website/src/data/{demos6,oc-benchmarks}.json (incl. the
# OpenCourant explicit-dynamics rows from perf/oc/*.json). No args = all
# six; pass slugs for a subset, e.g. `just website-demos crush arch`.
# Requires blender, ffmpeg, python3+pyvista. In the enclosing checkout,
# `just website-demos` also regenerates the full website afterwards.
[group('gallery')]
website-demos *args:
    SONIUM_CARGO="{{cargo}}" python3 scripts/website_demos.py {{ if args == "" { "all" } else { args } }}

# Regenerate the pinned website-gallery smoke set without any interactive UI.
# On a runner with pvbatch, the PC-fan test additionally verifies its emitted
# ParaView script renders a nonblank 1920×1080 PNG.
[group('gallery')]
simulation-galleries dir:
    {{cargo}} run --release -p sonium-roomsim --example web_gallery -- {{dir}}/roomsim
    {{cargo}} run --release -p sonium-sim-room --bin generate-roomeq-data -- --solver fast-hybrid --scenario small_stereo_2_0 --min-freq 50 --max-freq 100 --num-points 2 --gallery {{dir}}/sim-room --output-dir {{dir}}/sim-room-data
    {{cargo}} run --release -p sonium-sim-pcfan --bin pcfan-opt -- --solve-only --gallery {{dir}}/pcfan --cfd-cells-x 4 --cfd-cells-y 4 --cfd-cells-z 4 --cfd-picard-iterations 2 --cfd-direct-solver --cfd-linear-iterations 40
    {{cargo}} run --release -p sonium-sim-speaker-modeling --example port_optim_gallery -- {{dir}}/port-optim

# --- Per-platform performance records (perf/*.md) ---
#
# Each platform owner runs its recipe and commits the results:
#   Mac (Apple Silicon)   just perf-m1 / perf-m2 / perf-m4 (per chip)
#   Linux x86-64           just perf-x86
#   Linux + NVIDIA GPU     just perf-cuda
# See perf/README.md for the cell/harness mapping. PERF_QUICK=1 selects a
# bounded run (recorded in each file header; not comparable to full runs).

[group('perf')]
perf-m1:
    SONIUM_CARGO="{{cargo}}" python3 scripts/perf_cells.py m1

[group('perf')]
perf-m2:
    SONIUM_CARGO="{{cargo}}" python3 scripts/perf_cells.py m2

[group('perf')]
perf-m4:
    SONIUM_CARGO="{{cargo}}" python3 scripts/perf_cells.py m4

[group('perf')]
perf-x86:
    SONIUM_CARGO="{{cargo}}" python3 scripts/perf_cells.py x86

[group('perf')]
perf-cuda:
    SONIUM_CARGO="{{cargo}}" python3 scripts/perf_cells.py cuda

# Regenerate one cell: just perf-cell x86 blas
[group('perf')]
perf-cell platform cell:
    SONIUM_CARGO="{{cargo}}" python3 scripts/perf_cells.py {{platform}} --cells {{cell}}

# Regenerate the committed solver reports perf/PERF.md (direct) and
# perf/ITERATIVE_PERF.json (iterative). Full matrices by default;
# DIRECT_PERF_* / ITERATIVE_PERF_* env vars pass through for bounded runs.
[group('perf')]
perf-solvers out='perf':
    bash scripts/direct_solver_perf.sh {{out}}/PERF.md
    bash scripts/iterative_solver_perf.sh {{out}}/ITERATIVE_PERF.json

# --- OpenCourant explicit-dynamics matrix (perf/oc-explicit.md) ---
#
# Each recipe runs qa-oc-bench cells serially into perf/oc/*.json and
# republishes the markdown tables. Needs a vendor install for engine
# cells (OPENCOURANT_EXE / OPENCOURANT_MINIQA + LD_LIBRARY_PATH /
# RAD_CFG_PATH); run on a quiet box. PERF_OC_QUICK=1 selects one
# repeat, no warmup everywhere (smoke); PERF_OC_HERTZ_FULL=1 forces
# the full protocol on hertz_sphere too (slow: ~360 s/repeat native).
# Failing cells never block publish (their rows stay unpublished by
# policy); the recipe still exits nonzero so CI notices.

# Full N-deck matrix (every C3-conformant deck dir), then publish.
[group('perf')]
perf-oc out='perf/oc':
    #!/usr/bin/env bash
    set -uo pipefail
    export TMPDIR="${TMPDIR:-$HOME/src/all_of_sonium/tmp}"
    bash scripts/perf_oc_matrix.sh {{out}}; code=$?
    python3 scripts/perf_oc_publish.py {{out}} perf/oc-explicit.md
    exit $code

# LAW1 elastic decks only (taylor_bar, hertz_sphere, necking_bar,
# rail_crush), then publish.
[group('perf')]
perf-oc-elastic out='perf/oc':
    #!/usr/bin/env bash
    set -uo pipefail
    export TMPDIR="${TMPDIR:-$HOME/src/all_of_sonium/tmp}"
    PERF_OC_DECKS='taylor_bar hertz_sphere necking_bar rail_crush' bash scripts/perf_oc_matrix.sh {{out}}; code=$?
    python3 scripts/perf_oc_publish.py {{out}} perf/oc-explicit.md
    exit $code

# Plastic decks only (taylor_plastic, plasticity_thick_cylinder),
# then publish.
[group('perf')]
perf-oc-plastic out='perf/oc':
    #!/usr/bin/env bash
    set -uo pipefail
    export TMPDIR="${TMPDIR:-$HOME/src/all_of_sonium/tmp}"
    PERF_OC_DECKS='taylor_plastic plasticity_thick_cylinder' bash scripts/perf_oc_matrix.sh {{out}}; code=$?
    python3 scripts/perf_oc_publish.py {{out}} perf/oc-explicit.md
    exit $code

# Single deck (all 8 cells), then publish. Example:
# just perf-oc-deck plasticity_thick_cylinder
[group('perf')]
perf-oc-deck deck out='perf/oc':
    #!/usr/bin/env bash
    set -uo pipefail
    export TMPDIR="${TMPDIR:-$HOME/src/all_of_sonium/tmp}"
    PERF_OC_DECKS='{{deck}}' bash scripts/perf_oc_matrix.sh {{out}}; code=$?
    python3 scripts/perf_oc_publish.py {{out}} perf/oc-explicit.md
    exit $code

# G0 GPU proxy for one deck (no publish: proxy evidence is not bundle
# rows). Builds and runs perf/oc/<deck>_cuda_proxy.cu device-timed at
# short step counts plus an empty-force differential, then archives
# the console (<deck>_cuda_proxy.txt) and the roofline report
# (PERF_OC_GPU_REPORT, when set) next to the bundles. Example:
# just perf-oc-gpu necking_bar
# Needs nvcc and an idle NVIDIA GPU; no sonium.cuda explicit arm
# exists, so decks without a checked-in proxy fail closed.
[group('perf')]
perf-oc-gpu deck out='perf/oc':
    #!/usr/bin/env bash
    set -uo pipefail
    export TMPDIR="${TMPDIR:-$HOME/src/all_of_sonium/tmp}"
    bash scripts/perf_oc_gpu.sh {{deck}} {{out}}; code=$?
    exit $code

# --- Benchmarks ---

# Run every supported native benchmark, then available CUDA benchmark arms.
[group('bench')]
[positional-arguments]
bench *args:
    SONIUM_CARGO="{{cargo}}" python3 scripts/bench_machine.py "$@"

# --- Vendor-comparison benchmarks (one per missing perf cell) ---
#
# Threading follows perf/PERF.md methodology: Sonium at 16 threads, vendor
# tuned and never oversubscribed. Override with RAYON_NUM_THREADS /
# OPENBLAS_NUM_THREADS / VECLIB_MAXIMUM_THREADS.

# Dense LAPACK (potrf/getrf/geqrf) vs Accelerate (macOS) or OpenBLAS (Linux).
[group('bench')]
bench-lapack:
    RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-16}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}" {{cargo}} bench -p sonium-solvers-direct --bench lapack_compare --features bench-openblas

# Dense LAPACK incl. the cuSOLVER getrf arm (one-shot, upload inside timer).
[group('bench')]
bench-lapack-cuda:
    RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-16}" {{cargo}} bench -p sonium-solvers-direct --bench lapack_compare --features cuda

# CPU sparse matrix (CSR SpMV roofline preflight + effective iterative BW).
[group('bench')]
bench-sparse-cpu:
    {{cargo}} bench -p sonium-compute --bench roofline_preflight

# CUDA block-GMRES workstation (cpu vs cuda arms; residual-gated).
[group('bench')]
bench-gmres-cuda:
    {{cargo}} bench -p sonium-compute --bench block_gmres_accelerator --features cuda

# Build the PETSc KSP reference binary (headers via apt download, no sudo).
[group('bench')]
bench-petsc-build:
    scripts/build_petsc_reference.sh

# Iterative solvers vs PETSc KSP (needs bench-petsc-build first).
[group('bench')]
bench-petsc-run out='target/petsc-iterative.json':
    ITERATIVE_PERF_PETSC="$PWD/target/petsc_reference" bash scripts/iterative_solver_perf.sh {{out}}

# FFT (complex/real f32/f64, audio + large sizes) vs rustfft/realfft
# everywhere and Accelerate on macOS. Feeds perf/<platform>-fft.md.
[group('bench')]
bench-fft:
    {{cargo}} bench -p sonium-fft --bench fft_compare

# --- Cross-architecture validation (x86_64 SIMD paths) ---

# Validate that crates with x86_64 SIMD kernels compile cleanly for that
# target and pass numerical parity vs scalar. macOS-only — uses Rosetta 2
# (with AVX2+FMA translation) for execution. Requires
# `rustup target add x86_64-apple-darwin` (already installed on this
# machine).
#
# Today the cross-arch surface is:
#   - sonium-fastmath: AVX2/AVX-512 sincos lanes, MPFR vectors, and full-domain ULP harness
#   - sonium-fft: AVX2 f64 Stockham stages and independent-reference suite
#   - sonium-compute:  bem_cpu::avx2 4-pair regular-pair kernel
#
# Run before merging any change to a `#[cfg(target_arch = "x86_64")]`
# block in either crate.
[group('test')]
check-x86:
    RUSTFLAGS="-C target-feature=+avx2,+fma" \
        {{cargo}} check --target x86_64-apple-darwin \
        -p sonium-fastmath -p sonium-fft -p sonium-compute

# Run the x86_64 SIMD numerical-parity test suites under Rosetta 2.
# Each suite asserts equivalence with the scalar reference within the
# documented ULP / relative-error envelope.
[group('test')]
test-x86:
    RUSTFLAGS="-C target-feature=+avx2,+fma" \
        {{cargo}} test --release --target x86_64-apple-darwin \
        -p sonium-fastmath --lib
    RUSTFLAGS="-C target-feature=+avx2,+fma" \
        {{cargo}} test --release --target x86_64-apple-darwin \
        -p sonium-fastmath --test sincos_ulp
    RUSTFLAGS="-C target-feature=+avx2,+fma" \
        {{cargo}} test --release --target x86_64-apple-darwin \
        -p sonium-fastmath --test mpfr_golden_vectors
    RUSTFLAGS="-C target-feature=+avx2,+fma" \
        {{cargo}} test --release --target x86_64-apple-darwin \
        -p sonium-fastmath --test x86_validation
    RUSTFLAGS="-C target-feature=+avx2,+fma" \
        {{cargo}} test --release --target x86_64-apple-darwin \
        -p sonium-compute --lib bem_cpu
    RUSTFLAGS="-C target-feature=+avx2,+fma" \
        {{cargo}} test --release --target x86_64-apple-darwin \
        -p sonium-compute --lib blas1
    RUSTFLAGS="-C target-feature=+avx2,+fma" \
        {{cargo}} test --release --target x86_64-apple-darwin \
        -p sonium-compute --lib blas2
    RUSTFLAGS="-C target-feature=+avx2,+fma" \
        {{cargo}} test --release --target x86_64-apple-darwin \
        -p sonium-fft --lib
    RUSTFLAGS="-C target-feature=+avx2,+fma" \
        {{cargo}} test --release --target x86_64-apple-darwin \
        -p sonium-fft --test independent_reference

# Compile the scalar and wasm-simd FFT/fastmath surfaces for the browser
# target. Execution belongs in a browser/wasm runner; this gate catches
# target-feature, ABI, and no-std-compatible compilation regressions.
[group('test')]
check-wasm:
    RUSTFLAGS="-C target-feature=+simd128" \
        {{cargo}} check --target wasm32-unknown-unknown \
        -p sonium-fastmath -p sonium-fft

# Full cross-arch validation: check + parity tests on x86_64, wasm SIMD
# compilation, plus the native aarch64 test suite. Use this before bumping
# any of the SIMD kernels.
[group('test')]
validate-simd: check-x86 test-x86 check-wasm
    {{cargo}} test --release -p sonium-fastmath
    {{cargo}} test --release -p sonium-compute --lib bem_cpu

# Run the FastMath public dispatch twice on a native Linux x86_64 runner:
# once as a forced scalar baseline and once with AVX2+FMA required. The two
# processes consume the same checked-in MPFR corpus and fail closed when the
# requested AVX2 hardware is absent. This is separate from the Rosetta parity
# recipe above, which is not performance or native-runtime evidence.
[group('test')]
test-fastmath-native-x86:
    SONIUM_FASTMATH_FORCE_SCALAR=1 \
        {{cargo}} test --release -p sonium-fastmath --test mpfr_golden_vectors
    SONIUM_FASTMATH_FORCE_SCALAR=1 \
        {{cargo}} test --release -p sonium-fastmath --test x86_validation
    SONIUM_FASTMATH_REQUIRE_AVX2=1 SONIUM_FASTMATH_FORCE_AVX2=1 \
        {{cargo}} test --release -p sonium-fastmath --test mpfr_golden_vectors
    SONIUM_FASTMATH_REQUIRE_AVX2=1 SONIUM_FASTMATH_FORCE_AVX2=1 \
        {{cargo}} test --release -p sonium-fastmath --test x86_validation

# Run the compute native-x86 evidence producer in both effective dispatch
# modes. The native invocation is fail-closed when AVX2/FMA are absent; the
# scalar invocation remains a real public-dispatch baseline on the same host.
# Keep Cargo and evidence roots repo-local so concurrent review lanes never
# contend for the workspace target lock or mix artifacts.
[group('test')]
test-compute-native-x86:
    #!/usr/bin/env bash
    set -euo pipefail
    repo_root="$PWD"
    test "$(uname -m)" = x86_64 || { echo "native compute evidence requires x86_64" >&2; exit 2; }
    grep -qw avx2 /proc/cpuinfo || { echo "native compute evidence requires AVX2" >&2; exit 2; }
    grep -qw fma /proc/cpuinfo || { echo "native compute evidence requires FMA" >&2; exit 2; }
    mkdir -p "$repo_root/target/sonium-unix-c5/native" "$repo_root/target/sonium-unix-c5/scalar"
    SONIUM_REQUIRE_AVX2_FMA=1 \
        SONIUM_OUTPUT_DIR="$repo_root/target/sonium-unix-c5/native" \
        SONIUM_QA_THREAD_POLICY=single \
        SONIUM_QA_AFFINITY_POLICY=unbound \
        SONIUM_QA_NUMA_POLICY=default \
    SONIUM_QA_DETERMINISTIC_MODE=enabled \
        CARGO_TARGET_DIR=.target-unix-c5 \
        {{cargo}} test --release -p sonium-compute --test native_x86_evidence -- --nocapture --test-threads=1
    SONIUM_COMPUTE_FORCE_SCALAR=1 \
        SONIUM_OUTPUT_DIR="$repo_root/target/sonium-unix-c5/scalar" \
        SONIUM_QA_THREAD_POLICY=single \
        SONIUM_QA_AFFINITY_POLICY=unbound \
        SONIUM_QA_NUMA_POLICY=default \
        SONIUM_QA_DETERMINISTIC_MODE=enabled \
        CARGO_TARGET_DIR=.target-unix-c5 \
        {{cargo}} test --release -p sonium-compute --test native_x86_evidence -- --nocapture --test-threads=1
    SONIUM_C5_ROOT="$repo_root" python3 -c 'import json, os; from pathlib import Path; root=Path(os.environ["SONIUM_C5_ROOT"]); n=json.loads((root/"target/sonium-unix-c5/native/sonium_native_x86_evidence.json").read_text()); s=json.loads((root/"target/sonium-unix-c5/scalar/sonium_native_x86_evidence.json").read_text()); assert n["status"] == "pass" and n["selected_kernel"] == "avx2-fma"; assert s["status"] == "scalar-baseline" and s["selected_kernel"] == "scalar-fallback"; assert n["cases_failed"] == 0 and s["cases_failed"] == 0; assert len(n["observables"]) == len(s["observables"]) == 1634; error=max(abs(a-b) for a,b in zip(n["observables"], s["observables"])); assert error <= 5e-5, f"native/scalar observable error {error} exceeds 5e-5"; print(json.dumps({"observables":len(n["observables"]),"max_abs_error":error}))'

# Run the portable FEM reference/contract producer on native Linux x86_64.
# This lane deliberately records FEM's actual portable-scalar dispatch and
# does not pretend to provide the user-waived Windows/Arm64 comparison.
[group('test')]
test-fem-native-x86:
    #!/usr/bin/env bash
    set -euo pipefail
    repo_root="$PWD"
    test "$(uname -s)" = Linux || { echo "FEM4 evidence requires Linux" >&2; exit 2; }
    test "$(uname -m)" = x86_64 || { echo "FEM4 evidence requires x86_64" >&2; exit 2; }
    artifact="$repo_root/target/sonium-unix-fem4/native-x86_64-fem4-v1.json"
    mkdir -p "$(dirname "$artifact")"
    SONIUM_FEM_DETERMINISTIC=1 \
        RAYON_NUM_THREADS=1 \
        SONIUM_FEM_OUTPUT_DIR="$(dirname "$artifact")" \
        CARGO_TARGET_DIR="$repo_root/.target-unix-fem4" \
        {{cargo}} test --release -p sonium-fem --test native_x86_fem4_evidence -- --nocapture --test-threads=1
    (cd "$(dirname "$artifact")" && sha256sum "$(basename "$artifact")" > checksums.sha256 && sha256sum -c checksums.sha256)
    python3 "$repo_root/crates/sonium-fem/benchmarks/check_native_x86_fem4.py" "$artifact"

# Run the complete propeller application/reference suite on native Linux
# x86_64 and publish deterministic PP1 evidence for single and configured
# Rayon.  CUDA/Metal and the user-waived Windows/Arm64 comparisons remain
# typed non-capabilities; this recipe never relabels them as CPU results.
[group('test')]
test-propellers-native-x86:
    exec bash "{{justfile_directory()}}/scripts/run_unix_pp1_native_x86.sh"

# Emit the complete propeller CUDA/Metal capability boundary plus the bounded
# production-connected CUDA panel-pair cell.  The runner keeps all other
# application stages and Metal explicitly unsupported; GPU inventory alone is
# never promoted to propeller execution.
[group('test')]
test-propellers-cuda-capability:
    exec bash "{{justfile_directory()}}/scripts/run_unix_pp2_cuda.sh"

# Emit the bounded native PP3 fidelity/convergence ladder. This producer
# labels analytic, caller/synthetic, and reduced-physics evidence separately;
# it never promotes missing measured data or resolved CFD/FW-H evidence.
[group('test')]
test-propellers-fidelity-native-x86:
    #!/usr/bin/env bash
    set -euo pipefail
    repo_root="$PWD"
    test "$(uname -s)" = Linux || { echo "PP3 evidence requires Linux" >&2; exit 2; }
    test "$(uname -m)" = x86_64 || { echo "PP3 evidence requires x86_64" >&2; exit 2; }
    output="$repo_root/target/sonium-unix-pp3"
    mkdir -p "$output"
    SONIUM_OUTPUT_DIR="$output" \
        CARGO_TARGET_DIR="$repo_root/.target-unix-pp3" \
        {{cargo}} test --release -p sonium-sim-propellers \
        --test pp3_fidelity_evidence -- --nocapture --test-threads=1
    python3 "$repo_root/crates/sonium-sim-propellers/benchmarks/check_pp3_fidelity.py" \
        "$output/native-x86-pp3-fidelity-v1.json"

# Regenerate and validate the clean headless propeller release bundle on
# native Linux/x86. The CLI gallery and the smaller library-ledger gallery are
# both checked; QA-manifest fan-in remains an explicit blocked join until
# PP1--PP3 are complete and sonium-qa owns the aggregate. The integration
# evidence test runs deterministic child-process interruption and renderer
# capability faults under the repository-local PP4 fault target.
[group('test')]
test-propellers-release-native-x86:
    #!/usr/bin/env bash
    set -euo pipefail
    repo_root="$PWD"
    test "$(uname -s)" = Linux || { echo "PP4 evidence requires Linux" >&2; exit 2; }
    test "$(uname -m)" = x86_64 || { echo "PP4 evidence requires x86_64" >&2; exit 2; }
    output="$repo_root/target/sonium-unix-pp4"
    faults="$repo_root/.target-unix-pp4-faults/pp4-faults"
    rm -rf "$output"
    rm -rf "$faults"
    mkdir -p "$output/runtime"
    RAYON_NUM_THREADS=1 CARGO_TARGET_DIR="$repo_root/.target-unix-pp4-faults" \
        {{cargo}} run --release -p sonium-sim-propellers --bin propeller-3d-opt -- \
        --iterations 1 --gallery "$output/cli-gallery" > "$output/cli.log"
    SONIUM_OUTPUT_DIR="$output" SONIUM_PP4_FAULTS_DIR="$faults" RAYON_NUM_THREADS=1 \
        CARGO_TARGET_DIR="$repo_root/.target-unix-pp4-faults" \
        {{cargo}} test --release -p sonium-sim-propellers --test pp4_release_evidence \
        -- --nocapture --test-threads=1
    python3 "$repo_root/crates/sonium-sim-propellers/benchmarks/check_pp4_release.py" \
        "$output/native-x86-pp4-release-v1.json"

# Run the registered propeller QA shard and archive each structured result.
# This reads validation-manifest.toml without editing it; the workspace-wide
# qa-release-critical aggregate remains a sonium-qa-owned, explicitly gated
# join because PP1--PP3 are not yet complete.
[group('qa')]
test-propellers-qa-native-x86:
    #!/usr/bin/env bash
    set -euo pipefail
    repo_root="$PWD"
    test "$(uname -s)" = Linux || { echo "PP4 QA requires Linux" >&2; exit 2; }
    test "$(uname -m)" = x86_64 || { echo "PP4 QA requires x86_64" >&2; exit 2; }
    output="$repo_root/target/sonium-unix-pp4/qa"
    mkdir -p "$repo_root/target/sonium-unix-pp4/runtime"
    SONIUM_PP4_QA_DIR="$output" SONIUM_QA_RESULT_DIR="$output/cases" \
        SONIUM_OUTPUT_DIR="$output/artifacts" RAYON_NUM_THREADS=1 \
        CARGO_TARGET_DIR="$repo_root/.target-unix-pp4" \
        python3 "$repo_root/crates/sonium-sim-propellers/benchmarks/run_pp4_qa.py"
    python3 "$repo_root/crates/sonium-sim-propellers/benchmarks/check_pp4_qa.py" \
        "$output/qa-propeller-fan-in.json"

# AddressSanitizer coverage for the native AVX2 boundaries. LeakSanitizer is
# disabled because this workspace runs under a traced container where LSAN
# cannot initialize; ASan itself remains enabled for the SIMD loads/stores and
# public tail paths.
[group('test')]
test-fastmath-sanitized:
    RUSTFLAGS="-Zsanitizer=address" ASAN_OPTIONS=detect_leaks=0 \
        {{cargo}} +nightly test -p sonium-fastmath --target x86_64-unknown-linux-gnu \
        --test mpfr_golden_vectors
    RUSTFLAGS="-Zsanitizer=address" ASAN_OPTIONS=detect_leaks=0 \
        {{cargo}} +nightly test -p sonium-fastmath --target x86_64-unknown-linux-gnu \
        --test x86_validation

# Capture the two native Linux x86_64 accuracy profiles as separate JSON
# artifacts. The output directory is under `target/` so generated evidence is
# never mistaken for a checked-in oracle or a result from another machine.
[group('benchmark')]
fastmath-accuracy-native-x86:
    mkdir -p target/sonium-fastmath-native-x86
    SONIUM_FASTMATH_FORCE_SCALAR=1 \
        {{cargo}} run --release -p sonium-fastmath --bin fastmath_accuracy \
        > target/sonium-fastmath-native-x86/accuracy-scalar.json
    SONIUM_FASTMATH_REQUIRE_AVX2=1 SONIUM_FASTMATH_FORCE_AVX2=1 \
        {{cargo}} run --release -p sonium-fastmath --bin fastmath_accuracy \
        > target/sonium-fastmath-native-x86/accuracy-avx2.json

# Produce and gate the regime-separated FastMath performance artifact on a
# native Linux x86_64 AVX2/FMA runner. The scalar run is retained beside the
# AVX2 run for a same-input reference; the gate compares only the native run
# with the pinned checked-in baseline and fails closed on CPU/corpus drift.
[group('benchmark')]
fastmath-performance-native-x86:
    #!/usr/bin/env bash
    set -euo pipefail
    repo_root="$PWD"
    test "$(uname -m)" = x86_64 || { echo "FastMath F3 requires x86_64" >&2; exit 2; }
    grep -qw avx2 /proc/cpuinfo || { echo "FastMath F3 requires AVX2" >&2; exit 2; }
    grep -qw fma /proc/cpuinfo || { echo "FastMath F3 requires FMA" >&2; exit 2; }
    mkdir -p "$repo_root/target/sonium-unix-f3"
    SONIUM_FASTMATH_REQUIRE_AVX2=1 SONIUM_FASTMATH_FORCE_AVX2=1 \
        CARGO_TARGET_DIR="$repo_root/.target-unix-f3" \
        {{cargo}} run --release -p sonium-fastmath --bin fastmath_benchmark \
        > "$repo_root/target/sonium-unix-f3/native-avx2.json"
    SONIUM_FASTMATH_FORCE_SCALAR=1 \
        CARGO_TARGET_DIR="$repo_root/.target-unix-f3" \
        {{cargo}} run --release -p sonium-fastmath --bin fastmath_benchmark \
        > "$repo_root/target/sonium-unix-f3/scalar.json"
    python3 "$repo_root/crates/sonium-fastmath/benchmarks/check_x86_baseline.py" \
        "$repo_root/crates/sonium-fastmath/benchmarks/x86_64-baseline.json" \
        "$repo_root/target/sonium-unix-f3/native-avx2.json" \
        --output "$repo_root/target/sonium-unix-f3/statistical-gate.json"

# Execute the AVX-512 implementation on a runner that advertises AVX-512F.
# The environment variable makes absence of the required hardware a hard
# failure, which is appropriate for the dedicated native AVX-512 lane and
# intentionally separate from the portable validation-simd recipe above.
[group('test')]
test-avx512:
    SONIUM_FASTMATH_REQUIRE_AVX512=1 \
        {{cargo}} test --release -p sonium-fastmath --test x86_validation
    SONIUM_FASTMATH_REQUIRE_AVX512=1 \
        {{cargo}} test --release -p sonium-fastmath --test mpfr_golden_vectors

# Emit the versioned, machine-readable accuracy and performance evidence.
[group('benchmark')]
fastmath-evidence:
    {{cargo}} run --release -p sonium-fastmath --bin fastmath_accuracy
    {{cargo}} run --release -p sonium-fastmath --bin fastmath_benchmark


[group('qa')]
qa: qa-full

# OpenCourant-parity QA subset (release): engine-free native/golden legs,
# sonium-qa lib suites (default + funct-python), and the 15 live-engine legs
# (they skip loudly when OPENCOURANT_EXE is unset). Fails on any FAILED leg
# or suite; skips never fail. Needs TMPDIR on project scratch (defaulted).
[group('qa')]
qa-opencourant:
    #!/usr/bin/env bash
    set -uo pipefail
    export TMPDIR="${TMPDIR:-$HOME/src/all_of_sonium/tmp}"
    pass=0; fail=0; skip=0; failed_bins=""
    run_bin() {
        local bin="$1"
        local features="${2:-qa-binaries}"
        local out rc
        out=$({{cargo}} run -q --release -p sonium-qa --features "$features" --bin "$bin" 2>&1); rc=$?
        if [ $rc -ne 0 ]; then
            echo "FAIL  $bin (exit $rc)"; echo "$out" | tail -5
            fail=$((fail+1)); failed_bins="$failed_bins $bin"
        elif echo "$out" | grep -qE 'FAILED'; then
            echo "FAIL  $bin (FAILED marker)"
            fail=$((fail+1)); failed_bins="$failed_bins $bin"
        elif echo "$out" | grep -qE 'SUCCESS'; then
            echo "PASS  $bin"; pass=$((pass+1))
        else
            echo "SKIP  $bin"; echo "$out" | tail -2; skip=$((skip+1))
        fi
    }
    run_suite() {
        local desc="$1"; shift
        local out rc
        out=$("$@" 2>&1); rc=$?
        if [ $rc -eq 0 ]; then
            echo "PASS  $desc"; echo "$out" | grep -E 'test result' | tail -1; pass=$((pass+1))
        else
            echo "FAIL  $desc (exit $rc)"
            echo "$out" | grep -E 'test result|FAILED|failures:' | head -5
            fail=$((fail+1)); failed_bins="$failed_bins $desc"
        fi
    }
    if [ -z "${OPENCOURANT_EXE:-}${RADISS_EXE:-}" ]; then
        echo "note: OPENCOURANT_EXE unset - the 15 engine legs will skip (exit 0)."
    fi
    echo "== native/golden legs (engine-free) =="
    for b in qa-fem-explicit-taylor-plastic qa-fem-shell-explicit qa-fem-foam-native qa-fem-rupture-native qa-fem-sph qa-fem-ale qa-funct-python; do
        run_bin "$b"
    done
    echo "== sonium-qa lib =="
    run_suite "sonium-qa lib" {{cargo}} test -q --release -p sonium-qa --lib
    echo "== funct-python feature =="
    if {{cargo}} build -q --release -p sonium-qa --features qa-binaries,funct-python --bin qa-funct-python 2>/dev/null; then
        run_bin qa-funct-python qa-binaries,funct-python
        run_suite "sonium-qa lib [funct-python]" {{cargo}} test -q --release -p sonium-qa --features funct-python --lib
    else
        echo "SKIP  qa-funct-python[funct-python] (feature build failed: no Python dev libs?)"; skip=$((skip+1))
    fi
    echo "== live-engine legs =="
    for b in plasticity-thick-cylinder cantilever hertz-sphere beam-3d-cantilever beam-torsion shell-scordelis-lo shell-clamped-plate taylor-bar rail-crush bending-3pt necking-bar beam-curved-arch beam-warping shell-twisted-beam shell-oht-elastic; do
        run_bin "qa-fem-opencourant-$b"
    done
    echo "----"
    echo "opencourant subset: $pass passed, $fail failed, $skip skipped"
    if [ $fail -ne 0 ]; then echo "failed:$failed_bins"; exit 1; fi

# Run the canonical Wolfram reproduction of the release QA manifest.
# Set WOLFRAMSCRIPT when the executable is not on PATH.
[group('qa')]
qa-wolfram-manifest:
    #!/usr/bin/env bash
    set -euo pipefail
    wolfram="${WOLFRAMSCRIPT:-wolframscript}"
    command -v "$wolfram" >/dev/null 2>&1 || { echo "WolframScript not found: $wolfram (set WOLFRAMSCRIPT)" >&2; exit 127; }
    WOLFRAMSCRIPT="$wolfram" {{cargo}} test -p sonium-qa --release --test wolfram_sonium_qa -- --nocapture

# Run every Rust/Wolfram comparison integration test. Each test invokes its
# matching .wls reference and checks the emitted JSON/numerical values.
[group('qa')]
qa-rust-wolfram:
    #!/usr/bin/env bash
    set -euo pipefail
    wolfram="${WOLFRAMSCRIPT:-wolframscript}"
    command -v "$wolfram" >/dev/null 2>&1 || { echo "WolframScript not found: $wolfram (set WOLFRAMSCRIPT)" >&2; exit 127; }
    for test_file in crates/sonium-qa/tests/wolfram_*.rs; do
        test_name="${test_file##*/}"
        test_name="${test_name%.rs}"
        [[ "$test_name" == "wolfram_sonium_qa" ]] && continue
        echo "== $test_name =="
        if [[ "$test_name" == "wolfram_sim_crates" ]]; then
            WOLFRAMSCRIPT="$wolfram" {{cargo}} test -p sonium-qa --release \
                --features sim-wolfram --test "$test_name" -- --nocapture
        else
            WOLFRAMSCRIPT="$wolfram" {{cargo}} test -p sonium-qa --release \
                --test "$test_name" -- --nocapture
        fi
    done

# Run both directions: Wolfram validates the canonical Rust QA manifest, then
# Rust runs the individual Wolfram reference scripts and checks their results.
[group('qa')]
qa-wolfram: qa-wolfram-manifest qa-rust-wolfram

# Regenerate the audio Wolfram goldens (validation-manifest-audio.toml).
# Needs an activated engine: WOLFRAMSCRIPT=wolframscript just qa-goldens-audio.
[group('qa')]
qa-goldens-audio:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p crates/sonium-qa/wolfram/goldens
    engine="${WOLFRAMSCRIPT:-wolframscript}"
    kernel_default="/Applications/Wolfram Engine.app/Contents/Resources/Wolfram Player.app/Contents/MacOS/WolframKernel"
    if [ -z "${WolframKernel:-}" ] && [ -f "$kernel_default" ]; then export WolframKernel="$kernel_default"; fi
    for script in biquad_response capture_array capture_tdoa comb_first_dip fft_peak kautz_correction morlet_spots schroeder_t60 svf_response test_functions third_octave_spl waterfall_spots; do
        echo "--- $script"
        "$engine" -file "crates/sonium-qa/wolfram/$script.wls" | tail -n 1 > "crates/sonium-qa/wolfram/goldens/$script.json"
    done
    {{cargo}} test -p sonium-qa --release --test qa_audio_contract

[group('qa')]
qa-manifest:
    #!/usr/bin/env python3
    import json, pathlib, subprocess, tomllib
    root = pathlib.Path("{{justfile_directory()}}")
    subprocess.run(["python3", str(root / "scripts/generate_qa_inventory.py")], cwd=root, check=True)
    manifest = tomllib.loads((root / "crates/sonium-qa/validation-manifest.toml").read_text())
    cargo = tomllib.loads((root / "crates/sonium-qa/Cargo.toml").read_text())
    bins = {target["name"] for target in cargo["bin"]}
    required = {"id", "owner", "crate", "bin", "entry_point", "method", "scalar", "dimension", "reference", "reference_checksum", "norm", "tolerance", "tier", "timeout_seconds", "required_hardware", "artifacts"}
    seen = set()
    for case in manifest.get("case", []):
        assert not (missing := required - case.keys()), f"{case.get('id', '<missing id>')}: missing {sorted(missing)}"
        assert case["id"] not in seen, f"duplicate case id {case['id']}"
        assert case["bin"] in bins, f"{case['id']}: unknown Cargo binary {case['bin']}"
        if case.get("tier") == "accelerator-gate" or case.get("strict_result_contract"):
            strict = {"evidence_class", "thresholds_required", "runtime_required", "hardware_provenance_required", "artifact_provenance_required"}
            assert not (missing := strict - case.keys()), f"{case['id']}: missing strict accelerator metadata {sorted(missing)}"
            assert case["evidence_class"] in {"fault-injection", "numerical-parity", "workload-correctness", "performance"}, f"{case['id']}: unknown evidence class"
            assert all(isinstance(case[key], bool) and case[key] for key in strict - {"evidence_class"}), f"{case['id']}: accelerator evidence requirements must be true"
        if case["bin"] in {"qa-wave-1d-analytical-consistency", "qa-wave-cylinder-2d-analytical-consistency"}:
            assert case["tier"] == "reference-only", f"{case['id']}: analytical-only wave case cannot claim solver validation"
        seen.add(case["id"])
    assert seen, "validation manifest must contain a release-critical case"
    print(json.dumps({"schema": "sonium-qa-validation-manifest/v1", "cases": manifest["case"]}, indent=2))

[group('qa')]
qa-policy-test:
    PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts -p 'test_qa_result_policy.py'

qa-inventory:
    python3 scripts/generate_qa_inventory.py

# Guard the grouped next-review routing map against orphaned or duplicated docs.
review-index:
    python3 scripts/check_review_index.py

# Execute every non-reference manifest case through the same timeout/marker
# runner used by the normal QA tiers. This makes release evidence an explicit
# opt-in list rather than an accidental consequence of Cargo target discovery.
[group('qa')]
qa-release-critical:
    #!/usr/bin/env python3
    import concurrent.futures
    import os
    import pathlib, subprocess, sys, tomllib
    root = pathlib.Path("{{justfile_directory()}}")
    subprocess.run(["just", "qa-manifest"], cwd=root, check=True)
    manifest = tomllib.loads((root / "crates/sonium-qa/validation-manifest.toml").read_text())
    cases = [case for case in manifest["case"] if case["tier"] not in {"reference-only", "accelerator-gate"}]
    parallelism = int(os.environ.get("SONIUM_QA_PARALLELISM", "4"))
    if parallelism < 1:
        raise SystemExit("SONIUM_QA_PARALLELISM must be at least 1")

    def run_case(case):
        features = case.get("features", [])
        if features:
            command = [
                "just", "_qa-run-features", case["bin"], str(case["timeout_seconds"]),
                ",".join(features), case["id"],
            ]
        else:
            command = ["just", "_qa-run", case["bin"], str(case["timeout_seconds"]), case["id"]]
        case_env = os.environ.copy()
        if case["bin"] == "qa-room-cross-backend-correlation":
            # RoomSim's nested FEM/Rayon path can exceed the default Rust
            # worker stack when several QA processes run concurrently. Keep
            # this resource policy child-local and preserve explicit caller
            # overrides for both stack size and Rayon width.
            case_env.setdefault("RUST_MIN_STACK", "33554432")
            case_env.setdefault("RAYON_NUM_THREADS", "2")
        result = subprocess.run(
            command,
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=case_env,
        )
        return result.returncode, result.stdout

    results = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, min(parallelism, len(cases)))
    ) as pool:
        pending = {pool.submit(run_case, case): case for case in cases}
        for future in concurrent.futures.as_completed(pending):
            case = pending[future]
            results[case["id"]] = future.result()

    failures = []
    for case in cases:
        returncode, output = results[case["id"]]
        print(f"== {case['id']} ==")
        print(output, end="")
        if returncode:
            failures.append(case["id"])
    if failures:
        print(f"release-critical failures: {', '.join(failures)}", file=sys.stderr)
        sys.exit(1)
    print("SUCCESS")

# Execute one labelled accelerator shard and create its immutable fan-in
# record.  CUDA and native-x86 jobs call this independently; no worker edits
# the shared manifest, inventory, or release index while a shard is running.
[group('qa')]
qa-release-accelerators hardware="":
    #!/usr/bin/env python3
    import datetime, hashlib, json, os, pathlib, subprocess, sys, tomllib
    root = pathlib.Path("{{justfile_directory()}}")
    hardware = "{{hardware}}".strip()
    if hardware not in {"nvidia-cuda", "native-x86_64"}:
        raise SystemExit("qa-release-accelerators requires hardware=nvidia-cuda or hardware=native-x86_64")
    subprocess.run(["just", "qa-manifest"], cwd=root, check=True)
    manifest_path = root / "crates/sonium-qa/validation-manifest.toml"
    manifest = tomllib.loads(manifest_path.read_text())
    cases = [
        case for case in manifest.get("case", [])
        if case.get("tier") == "accelerator-gate" and case.get("required_hardware") == hardware
    ]
    if not cases:
        raise SystemExit(f"no accelerator cases registered for {hardware}")
    output = pathlib.Path(os.environ.get("SONIUM_QA_ACCELERATOR_EVIDENCE_DIR", root / "target" / "sonium-qa-accelerators" / hardware))
    case_dir = output / "cases"
    artifact_dir = output / "artifacts"
    if hardware == "nvidia-cuda":
        for legacy_name in (
            "bem_cuda_hmatrix_fmm_report.json",
            "bem_cuda_matrix_report.json",
            "compute_fem_cuda_adjoint.json",
        ):
            legacy_path = output / legacy_name
            if legacy_path.is_file():
                legacy_path.unlink()
    case_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["SONIUM_QA_RESULT_DIR"] = str(case_dir)
    env["SONIUM_OUTPUT_DIR"] = str(artifact_dir)
    env["SONIUM_QA_ACCELERATOR_LANE"] = hardware
    expected_backend = "cuda" if hardware == "nvidia-cuda" else "cpu-native-x86_64"
    env["SONIUM_REQUIRE_BACKEND"] = expected_backend
    sys.path.insert(0, str(root / "scripts"))
    from qa_result_policy import validate_result
    results = []
    failures = []
    for case in cases:
        features = case.get("features", [])
        if features:
            command = ["just", "_qa-run-features", case["bin"], str(case["timeout_seconds"]), ",".join(features), case["id"]]
        else:
            command = ["just", "_qa-run", case["bin"], str(case["timeout_seconds"]), case["id"]]
        completed = subprocess.run(command, cwd=root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(completed.stdout, end="")
        path = case_dir / f"{case['id']}.json"
        try:
            result = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            result = {"case_id": case["id"], "status": "failed", "message": str(error)}
        policy_errors = validate_result(case, result, artifact_root=artifact_dir)
        if completed.returncode or policy_errors:
            failures.append({"id": case["id"], "returncode": completed.returncode, "errors": policy_errors})
        results.append(result)
    compute_summaries = {}
    if hardware == "nvidia-cuda":
        summarizer = root / "crates" / "sonium-compute" / "benchmarks" / "summarize_cuda_evidence.py"
        for artifact_name, summary_name in (
            ("bem_cuda_matrix_report.json", "cuda-bem-summary.json"),
            ("compute_fem_cuda_adjoint.json", "cuda-fem-adjoint-summary.json"),
        ):
            artifact_path = artifact_dir / artifact_name
            if not artifact_path.is_file():
                failures.append({
                    "id": f"compute-evidence:{artifact_name}",
                    "returncode": 1,
                    "errors": ["required raw CUDA compute artifact is missing"],
                })
                continue
            summarized = subprocess.run(
                [sys.executable, str(summarizer), str(artifact_path)],
                cwd=root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            print(summarized.stdout, end="")
            if summarized.returncode:
                failures.append({
                    "id": f"compute-evidence:{artifact_name}",
                    "returncode": summarized.returncode,
                    "errors": ["CUDA compute evidence summarizer rejected the artifact"],
                })
                continue
            try:
                summary = json.loads(summarized.stdout)
            except json.JSONDecodeError as error:
                failures.append({
                    "id": f"compute-evidence:{artifact_name}",
                    "returncode": 1,
                    "errors": [f"summarizer emitted invalid JSON: {error}"],
                })
                continue
            summary_path = artifact_dir / summary_name
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            compute_summaries[artifact_name] = summary
    def digest(path):
        value = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(chunk)
        return value.hexdigest()
    counts = {name: sum((result.get("runtime") or {}).get("counts", {}).get(name, 0) for result in results) for name in ("executed", "passed", "failed", "unsupported")}
    record = {
        "schema": "sonium-qa-accelerator-evidence/v1",
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "hardware": hardware,
        "required_backend": expected_backend,
        "required_case_ids": [case["id"] for case in cases],
        "coverage": counts,
        "cases": results,
        "failures": failures,
        "manifest_sha256": digest(manifest_path),
        "environment": {
            "platform": sys.platform,
            "python": sys.version.split()[0],
            "cpu_model": os.environ.get("SONIUM_QA_CPU_MODEL"),
            "microcode": os.environ.get("SONIUM_QA_MICROCODE"),
            "os": os.environ.get("SONIUM_QA_OS", sys.platform),
            "compiler": os.environ.get("SONIUM_QA_COMPILER"),
            "device": os.environ.get("SONIUM_QA_CUDA_DEVICE_NAME"),
            "driver": os.environ.get("SONIUM_QA_CUDA_DRIVER_VERSION"),
            "toolkit": os.environ.get("SONIUM_QA_CUDA_TOOLKIT_VERSION"),
            "peak_flops": os.environ.get("SONIUM_QA_CUDA_PEAK_FLOPS"),
            "memory_bandwidth": os.environ.get("SONIUM_QA_CUDA_MEMORY_BANDWIDTH"),
            "target_features": os.environ.get("SONIUM_QA_TARGET_FEATURES"),
            "runtime_dispatch": os.environ.get("SONIUM_QA_RUNTIME_DISPATCH"),
            "thread_policy": os.environ.get("SONIUM_QA_THREAD_POLICY"),
            "affinity_policy": os.environ.get("SONIUM_QA_AFFINITY_POLICY"),
            "numa_policy": os.environ.get("SONIUM_QA_NUMA_POLICY"),
            "deterministic_mode": os.environ.get("SONIUM_QA_DETERMINISTIC_MODE"),
        },
        "compute_cuda_summaries": compute_summaries,
    }
    evidence = output / "accelerator-evidence.json"
    evidence.write_text(json.dumps(record, indent=2) + "\n")
    checksums = {"accelerator-evidence.json": digest(evidence), "validation-manifest.toml": digest(manifest_path)}
    for path in sorted(case_dir.glob("*.json")):
        checksums[str(path.relative_to(output))] = digest(path)
    for path in sorted(artifact_dir.rglob("*")):
        if path.is_file():
            checksums[str(path.relative_to(output))] = digest(path)
    (output / "checksums.sha256").write_text("".join(f"{value}  {name}\n" for name, value in sorted(checksums.items())))
    if (
        failures
        or len(results) != len(cases)
        or any(result.get("status") != "passed" for result in results)
        or counts["failed"]
        or counts["unsupported"]
    ):
        print(f"accelerator shard {hardware} failed: {failures}", file=sys.stderr)
        sys.exit(1)
    print(f"SUCCESS: accelerator shard {hardware} evidence written to {evidence}")

# Stable native Linux/NVIDIA QA1 lane.  This consumes the existing immutable
# CUDA shard, runs operation/solver/physical producers, and records required
# missing-device and forced-fallback probes under repository-local paths.
[group('qa')]
qa1-cuda:
    #!/usr/bin/env bash
    set -euo pipefail
    exec python3 "{{justfile_directory()}}/scripts/run_unix_qa1_cuda.py"

# Validate both immutable accelerator shards together before the release
# evidence and workspace quality gates run.
[group('qa')]
qa-release-aggregate:
    python3 scripts/validate_accelerator_fan_in.py

# Archive a machine-readable release-gate result alongside the manifest. The
# wrapped gate remains authoritative; this recipe only records its outcome.
[group('qa')]
qa-release-evidence:
    #!/usr/bin/env python3
    import datetime, hashlib, json, os, pathlib, platform, shutil, subprocess, sys, tomllib
    from xml.sax.saxutils import escape as xml_escape
    root = pathlib.Path("{{justfile_directory()}}")
    sys.path.insert(0, str(root / "scripts"))
    from qa_result_policy import validate_result
    output = pathlib.Path(os.environ.get("SONIUM_QA_EVIDENCE_DIR", root / "target" / "sonium-qa-evidence"))
    output.mkdir(parents=True, exist_ok=True)
    case_dir = output / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = output / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["SONIUM_QA_RESULT_DIR"] = str(case_dir)
    # Keep result-declared artifacts beside the structured evidence.  Result
    # paths are deliberately relative (validated by QaResult), so this also
    # prevents a release bundle from silently referring to a transient temp
    # directory that disappears after the run.
    env["SONIUM_OUTPUT_DIR"] = str(artifact_dir)
    run = subprocess.run(["just", "qa-release-critical"], cwd=root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    manifest_path = root / "crates/sonium-qa/validation-manifest.toml"
    manifest = tomllib.loads(manifest_path.read_text())
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.strip() or "unavailable"
    case_results = []
    for case in sorted(case_dir.glob("*.json")):
        try:
            case_results.append(json.loads(case.read_text()))
        except (OSError, json.JSONDecodeError) as error:
            case_results.append({"case_id": case.stem, "status": "failed", "message": str(error)})
    required_ids = {
        case["id"]
        for case in manifest.get("case", [])
        if case.get("tier") not in {"reference-only", "accelerator-gate"}
    }
    declared_ids = {case["id"] for case in manifest.get("case", [])}
    observed_ids = {result.get("case_id") for result in case_results}
    missing_ids = sorted(required_ids - observed_ids)
    undeclared_ids = sorted(
        case_id for case_id in observed_ids - declared_ids if case_id
    )
    nonpassing_required_ids = sorted(
        result.get("case_id")
        for result in case_results
        if result.get("case_id") in required_ids and result.get("status") != "passed"
    )
    threshold_failures = []
    for result in case_results:
        thresholds = result.get("thresholds") or {}
        metrics = result.get("metrics") or {}
        directions = result.get("threshold_directions") or {}
        for name, threshold in thresholds.items():
            metric = metrics.get(name)
            direction = directions.get(name)
            satisfied = (
                metric is not None
                and direction in {"minimum", "maximum"}
                and ((direction == "minimum" and metric >= threshold)
                     or (direction == "maximum" and metric <= threshold))
            )
            if not satisfied:
                threshold_failures.append({
                    "case_id": result.get("case_id"),
                    "metric": name,
                    "value": metric,
                    "threshold": threshold,
                    "direction": direction,
                })
    evidence_errors = []
    case_by_id = {case["id"]: case for case in manifest.get("case", [])}
    for case in manifest.get("case", []):
        result = next(
            (candidate for candidate in case_results if candidate.get("case_id") == case["id"]),
            None,
        )
        if result is None or result.get("status") != "passed":
            continue
        policy_errors = validate_result(
            case,
            result,
            artifact_root=artifact_dir,
        )
        evidence_errors.extend(
            {"case_id": case["id"], "field": "result_policy", "expected": "valid", "actual": error}
            for error in policy_errors
        )
        evidence = result.get("evidence") or {}
        expected_evidence = {
            "reference": case.get("reference"),
            "reference_checksum": case.get("reference_checksum"),
            "dimension": case.get("dimension"),
            "formulation": case.get("formulation") or case.get("method"),
            "tolerance": case.get("tolerance"),
        }
        if "{backend}" in (expected_evidence.get("formulation") or ""):
            expected_evidence["formulation"] = expected_evidence["formulation"].replace(
                "{backend}", result.get("actual_backend") or result.get("requested_backend") or "unknown"
            )
        for field, expected in expected_evidence.items():
            if expected is not None and evidence.get(field) != expected:
                evidence_errors.append({
                    "case_id": case["id"],
                    "field": field,
                    "expected": expected,
                    "actual": evidence.get(field),
                })
    checksum_errors = []
    artifact_errors = []

    def sha256_file(path):
        digest = hashlib.sha256()
        try:
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as error:
            checksum_errors.append(f"{path}: {error}")
            return None
        return digest.hexdigest()

    # Hash the immutable inputs and every structured case record that will be
    # embedded in the release bundle. Human stdout is intentionally excluded:
    # the JSON records are the authoritative, machine-readable evidence.
    artifact_checksums = {"validation-manifest.toml": sha256_file(manifest_path)}
    for case_file in sorted(case_dir.glob("*.json")):
        digest = sha256_file(case_file)
        if digest is not None:
            artifact_checksums[str(case_file.relative_to(output))] = digest
    artifact_root = artifact_dir.resolve()
    for result in case_results:
        case_id = result.get("case_id", "unknown")
        for artifact in result.get("artifacts") or []:
            if not isinstance(artifact, str) or not artifact.strip():
                artifact_errors.append(f"{case_id}: invalid artifact path {artifact!r}")
                continue
            relative = pathlib.PurePath(artifact)
            if relative.is_absolute() or ".." in relative.parts:
                artifact_errors.append(f"{case_id}: artifact path escapes bundle: {artifact!r}")
                continue
            source = (artifact_dir / relative).resolve()
            try:
                bundle_path = source.relative_to(artifact_root)
            except ValueError:
                artifact_errors.append(f"{case_id}: artifact path escapes bundle: {artifact!r}")
                continue
            if not source.is_file():
                artifact_errors.append(f"{case_id}: declared artifact is missing: {artifact!r}")
                continue
            digest = sha256_file(source)
            if digest is not None:
                artifact_checksums[str(pathlib.Path("artifacts") / bundle_path)] = digest
                case = case_by_id.get(case_id, {})
                if case.get("tier") == "accelerator-gate" or case.get("strict_result_contract"):
                    declared = (result.get("evidence") or {}).get("artifact_checksums", {}).get(artifact)
                    provenance = (result.get("evidence") or {}).get("artifact_provenance", {}).get(artifact, {})
                    if declared != f"sha256:{digest}":
                        artifact_errors.append(f"{case_id}: declared checksum does not match artifact {artifact!r}")
                    if provenance.get("checksum") != declared:
                        artifact_errors.append(f"{case_id}: artifact provenance checksum does not match {artifact!r}")
    accelerator_source = root / "target" / "sonium-qa-accelerators"
    accelerator_bundle = output / "accelerators"
    accelerator_checksums = {}
    if os.environ.get("SONIUM_REQUIRE_ACCELERATOR_EVIDENCE") == "1" and not accelerator_source.is_dir():
        artifact_errors.append("required accelerator shard evidence is missing")
    if accelerator_source.is_dir():
        for source in sorted(accelerator_source.rglob("*")):
            if not source.is_file():
                continue
            relative = source.relative_to(accelerator_source)
            destination = accelerator_bundle / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            digest = sha256_file(destination)
            if digest is not None:
                accelerator_checksums[str(pathlib.Path("accelerators") / relative)] = digest
    artifact_checksums.update(accelerator_checksums)
    platform_sources = {
        "avx512": root / "target" / "sonium-avx512-evidence",
        "windows-x86": root / "target" / "sonium-windows-x86-evidence",
    }
    platform_bundle = output / "platform"
    platform_checksums = {}
    platform_evidence = []
    for label, source_root in platform_sources.items():
        if os.environ.get("SONIUM_REQUIRE_ACCELERATOR_EVIDENCE") == "1" and not source_root.is_dir():
            artifact_errors.append(f"required {label} platform evidence is missing")
        if not source_root.is_dir():
            continue
        for source in sorted(source_root.rglob("*")):
            if not source.is_file():
                continue
            relative = source.relative_to(source_root)
            destination = platform_bundle / label / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            digest = sha256_file(destination)
            if digest is not None:
                path = str(pathlib.Path("platform") / label / relative)
                platform_checksums[path] = digest
                platform_evidence.append(path)
    artifact_checksums.update(platform_checksums)
    gate_returncode = run.returncode or int(
        bool(missing_ids)
        or bool(undeclared_ids)
        or bool(nonpassing_required_ids)
        or bool(threshold_failures)
        or bool(evidence_errors)
        or bool(checksum_errors)
        or bool(artifact_errors)
    )
    record = {
        "schema": "sonium-qa-release-evidence/v1",
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "git_sha": sha,
        "status": "passed" if gate_returncode == 0 else "failed",
        "environment": {"platform": platform.platform(), "python": platform.python_version(), "rayon_threads": os.environ.get("RAYON_NUM_THREADS", "runtime-default"), "cpu_model": os.environ.get("SONIUM_QA_CPU_MODEL"), "microcode": os.environ.get("SONIUM_QA_MICROCODE"), "os": os.environ.get("SONIUM_QA_OS", sys.platform), "compiler": os.environ.get("SONIUM_QA_COMPILER"), "target_features": os.environ.get("SONIUM_QA_TARGET_FEATURES"), "runtime_dispatch": os.environ.get("SONIUM_QA_RUNTIME_DISPATCH"), "thread_policy": os.environ.get("SONIUM_QA_THREAD_POLICY"), "affinity_policy": os.environ.get("SONIUM_QA_AFFINITY_POLICY"), "numa_policy": os.environ.get("SONIUM_QA_NUMA_POLICY"), "deterministic_mode": os.environ.get("SONIUM_QA_DETERMINISTIC_MODE")},
        "coverage": {name: sum((result.get("runtime") or {}).get("counts", {}).get(name, 0) for result in case_results) for name in ("executed", "passed", "failed", "unsupported")},
        "cases": [{"id": case["id"], "bin": case["bin"], "tier": case["tier"], "required_hardware": case["required_hardware"], "result": next((result for result in case_results if result.get("case_id") == case["id"]), None)} for case in manifest.get("case", [])],
        "case_results": case_results,
        "missing_required_cases": missing_ids,
        "undeclared_case_results": undeclared_ids,
        "nonpassing_required_cases": nonpassing_required_ids,
        "threshold_failures": threshold_failures,
        "evidence_errors": evidence_errors,
        "checksum_algorithm": "sha256",
        "artifact_checksums": artifact_checksums,
        "checksum_errors": checksum_errors,
        "artifact_errors": artifact_errors,
        "accelerator_shards": sorted(accelerator_checksums),
        "platform_evidence": sorted(platform_evidence),
        "gate_output": run.stdout,
    }
    evidence_path = output / "release-evidence.json"
    evidence_path.write_text(json.dumps(record, indent=2) + "\n")
    def xml_attribute(value):
        return xml_escape(str(value), {'"': '&quot;', "'": '&apos;'})

    def xml_cdata(value):
        return str(value).replace("]]>", "]]]]><![CDATA[>")

    junit_cases = []
    for result in case_results:
        name = xml_attribute(result.get("case_id", "unknown"))
        status = result.get("status")
        if status == "passed":
            body = ""
        elif status == "skipped":
            reason = xml_attribute(result.get("skip_reason") or "unspecified")
            body = f"<skipped message=\"{reason}\"/>"
        else:
            message = result.get("message") or result.get("stdout_tail") or "QA case failed"
            body = f"<failure message=\"QA case failed\"><![CDATA[{xml_cdata(message)}]]></failure>"
        junit_cases.append(f'<testcase classname="sonium.qa" name="{name}">{body}</testcase>')
    if not junit_cases:
        junit_cases.append(f'<testcase classname="sonium.qa" name="release-critical"><failure message="no structured QA results"/></testcase>')
    failures = (
        sum(1 for result in case_results if result.get("status") != "passed")
        + len(missing_ids)
        + len(undeclared_ids)
    )
    junit = f'<?xml version="1.0" encoding="UTF-8"?>\n<testsuite name="sonium-qa-release-critical" tests="{len(junit_cases)}" failures="{failures}">' + "".join(junit_cases) + "</testsuite>\n"
    junit_path = output / "release-evidence.junit.xml"
    junit_path.write_text(junit)
    bundle_checksums = {
        **artifact_checksums,
        "release-evidence.json": sha256_file(evidence_path),
        "release-evidence.junit.xml": sha256_file(junit_path),
    }
    (output / "checksums.sha256").write_text(
        "".join(
            f"{digest}  {path}\n"
            for path, digest in sorted(bundle_checksums.items())
            if digest is not None
        )
    )
    print(f"release evidence written to {evidence_path}")
    print(run.stdout, end="")
    sys.exit(gate_returncode)

# Compare two macOS evidence bundles. A regression fails only when it exceeds
# 10% and the empirical intervals are disjoint.
[group('qa')]
qa-compute-evidence-compare baseline current:
    python3 {{justfile_directory()}}/crates/sonium-qa/compare_compute_evidence.py {{baseline}} {{current}}

# Cross-target compile gate for the portable CPU backend.  This is compile-only
# and never launches a simulator or browser workload.
[group('qa')]
qa-compute-wasm-check:
    {{cargo}} check -p sonium-compute --features wasm --target wasm32-unknown-unknown --offline -j 1

# Render the declared release-evidence inventory for README/release notes.q
[group('qa')]
qa-manifest-table:
    #!/usr/bin/env python3
    import pathlib, tomllib
    root = pathlib.Path("{{justfile_directory()}}")
    cases = tomllib.loads((root / "crates/sonium-qa/validation-manifest.toml").read_text()).get("case", [])
    print("| ID | Owner | Binary | Tier | Reference |")
    print("|---|---|---|---|---|")
    for case in cases:
        print(f"| `{case['id']}` | {case['owner']} | `{case['bin']}` | {case['tier']} | {case['reference']} |")

# Stable native-CUDA validation for the SP1/SP2 preconditioner evidence.
[group('qa')]
qa-solvers-preconditioners-cuda:
    exec bash "{{justfile_directory()}}/scripts/run_unix_sp2_cuda.sh"

# Stable native-CUDA validation for SI3 iterative throughput evidence.
[group('qa')]
qa-solvers-iterative-cuda-throughput:
    exec bash "{{justfile_directory()}}/scripts/run_unix_si3_cuda.sh"

# Stable native-CUDA validation for the S1 facade support matrix.
[group('qa')]
qa-solvers-facade-support-matrix:
    #!/usr/bin/env bash
    set -euo pipefail
    output="${SONIUM_OUTPUT_DIR:-target/sonium-unix-s1}"
    target="${CARGO_TARGET_DIR:-.target-unix-s1}"
    CARGO_TARGET_DIR="$target" SONIUM_OUTPUT_DIR="$output" SONIUM_REQUIRE_CUDA=1 {{cargo}} run -p sonium-qa --features 'qa-binaries cuda' --bin qa-solvers-facade-support-matrix

# Stable native-CUDA validation for FFT2. The required unit suite covers both
# cuFFT complex precisions, regular/strided/batched views, identity laws,
# malformed buffers, injected resource failures, and terminal device loss.
# The benchmark then emits a machine-readable row matrix and warm resource
# counters. Unsupported high-level/layout/cancellation cells stay explicit;
# no row is relabelled as CUDA when it stages through the host.
[group('qa')]
qa-fft2-cuda:
    #!/usr/bin/env bash
    set -euo pipefail
    repo_root="{{justfile_directory()}}"
    "$repo_root/scripts/run_unix_fft2_cuda.sh"

# Stable native-CUDA validation for FEM2's bounded device-resident solve and
# constraint-transfer seam.  The producer/checker pair records the supported
# H1/Tri3/P1 cell, typed malformed-CSR admission, and a required-CUDA forced
# fallback probe; unsupported FEM spaces and constraints remain explicit.
[group('qa')]
qa-fem2-cuda:
    #!/usr/bin/env bash
    set -euo pipefail
    repo_root="{{justfile_directory()}}"
    "$repo_root/scripts/run_unix_fem2_cuda.sh"

# Stable bounded PC-fan CUDA forward/residency contract.  The producer runs a
# real CPU steady/BDF baseline, probes the requested CUDA linear-solve path,
# and records host staging plus typed unsupported failure classes.  It never
# promotes the current host-assembled FEM bridge to a device-resident pass.
[group('qa')]
qa-pcfan-forward-cuda:
    #!/usr/bin/env bash
    set -euo pipefail
    repo_root="$PWD"
    test "$(uname -s)" = Linux || { echo "PF1 evidence requires Linux" >&2; exit 2; }
    output="${SONIUM_OUTPUT_DIR:-$repo_root/target/sonium-unix-pf1}"
    target="${CARGO_TARGET_DIR:-$repo_root/.target-unix-pf1}"
    mkdir -p "$output"
    env_args=(
        "SONIUM_OUTPUT_DIR=$output"
        "CARGO_TARGET_DIR=$target"
        "SONIUM_REQUIRE_CUDA=1"
    )
    if command -v nvidia-smi >/dev/null 2>&1; then
        env_args+=(
            "SONIUM_QA_CUDA_DEVICE_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
            "SONIUM_QA_CUDA_DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n '1p')"
            "SONIUM_QA_CUDA_TOOLKIT_VERSION=${CUDA_VERSION:-not-reported}"
        )
    fi
    env "${env_args[@]}" {{cargo}} run --release -p sonium-qa --features 'qa-binaries cuda' --bin qa-pcfan-forward-cuda-contract

# Stable PC-fan CUDA adjoint/parity capability evidence.  The producer runs
# the production CPU/CUDA parity collector/evaluator on identical bounded
# fixtures, then requires device residency at the checked adjoint handoff so
# the current host-assembled linear-solve bridge is recorded as unsupported
# instead of being mislabeled as a CUDA flow/acoustic adjoint.
[group('qa')]
qa-pcfan-adjoint-cuda:
    #!/usr/bin/env bash
    set -euo pipefail
    repo_root="$PWD"
    test "$(uname -s)" = Linux || { echo "PF2 evidence requires Linux" >&2; exit 2; }
    output="${SONIUM_OUTPUT_DIR:-$repo_root/target/sonium-unix-pf2}"
    target="${CARGO_TARGET_DIR:-$repo_root/.target-unix-pf2}"
    mkdir -p "$output"
    env_args=(
        "SONIUM_OUTPUT_DIR=$output"
        "CARGO_TARGET_DIR=$target"
        "SONIUM_REQUIRE_CUDA=1"
    )
    if command -v nvidia-smi >/dev/null 2>&1; then
        env_args+=(
            "SONIUM_QA_CUDA_DEVICE_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
            "SONIUM_QA_CUDA_DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n '1p')"
            "SONIUM_QA_CUDA_TOOLKIT_VERSION=${CUDA_VERSION:-not-reported}"
        )
    fi
    env "${env_args[@]}" {{cargo}} run --release -p sonium-qa --features 'qa-binaries cuda' --bin qa-pcfan-adjoint-cuda-contract

# Produce and validate the deterministic native Linux/x86 PC-fan PF3 product
# reference. The producer owns CPU steady/transient fields, conservation
# ledgers, reduced objective/gradient finite differences, and explicit
# Arm64/Windows waivers; no CUDA result is inferred for this CPU-only unit.
[group('test')]
test-pcfan-native-x86-pf3:
    exec bash "{{justfile_directory()}}/scripts/run_unix_pf3_native_x86.sh"

# Stable native-x86 facade conformance validation. The two process invocations
# use identical fixtures and compare their machine-readable outputs below;
# AVX2/FMA absence is a hard unsupported result, never a scalar result labeled
# native. Arm64/Windows comparisons are explicitly waived for this Unix lane.
[group('qa')]
qa-solvers-facade-native-x86:
    #!/usr/bin/env bash
    set -euo pipefail
    repo_root="$PWD"
    test "$(uname -s)" = Linux || { echo "S3 evidence requires Linux" >&2; exit 2; }
    test "$(uname -m)" = x86_64 || { echo "S3 evidence requires native x86_64" >&2; exit 2; }
    grep -qw avx2 /proc/cpuinfo || { echo "S3 evidence requires AVX2" >&2; exit 2; }
    grep -qw fma /proc/cpuinfo || { echo "S3 evidence requires FMA" >&2; exit 2; }
    native="$repo_root/target/sonium-unix-s3/native"
    scalar="$repo_root/target/sonium-unix-s3/scalar"
    mkdir -p "$native" "$scalar"
    env -u SONIUM_SPARSE_FORCE_SCALAR -u SONIUM_COMPUTE_FORCE_SCALAR \
        SONIUM_S3_DISPATCH_MODE=native-avx2-fma \
        SONIUM_OUTPUT_DIR="$native" CARGO_TARGET_DIR="$repo_root/.target-unix-s3" \
        {{cargo}} run --release -p sonium-qa --features qa-binaries --bin qa-solvers-facade-native-x86
    SONIUM_S3_DISPATCH_MODE=scalar SONIUM_SPARSE_FORCE_SCALAR=1 SONIUM_COMPUTE_FORCE_SCALAR=1 \
        SONIUM_OUTPUT_DIR="$scalar" CARGO_TARGET_DIR="$repo_root/.target-unix-s3" \
        {{cargo}} run --release -p sonium-qa --features qa-binaries --bin qa-solvers-facade-native-x86
    comparison="$repo_root/target/sonium-unix-s3/native-vs-scalar-comparison-v1.json"
    python3 "$repo_root/crates/sonium-qa/benchmarks/check_solver_native_x86.py" \
        "$native/solvers_facade_native_x86.json" "$scalar/solvers_facade_native_x86.json" \
        | tee "$comparison"
    (cd "$repo_root/target/sonium-unix-s3" && \
        sha256sum native/solvers_facade_native_x86.json \
        scalar/solvers_facade_native_x86.json \
        native-vs-scalar-comparison-v1.json > checksums.sha256 && \
        sha256sum -c checksums.sha256)

# Fast deterministic validation tier for CI and local pre-merge checks.
[group('qa')]
qa-smoke:
    just _qa-run qa-compute-properties 120
    just _qa-run qa-compute-operation-matrix 120
    just _qa-run qa-compute-block-workspace 120
    just _qa-run qa-compute-block-krylov-workspace 120
    just _qa-run qa-compute-gmres-workspace 120
    just _qa-run qa-compute-scheduler-contract 120
    just _qa-run qa-compute-transfer-stats 120
    just _qa-run qa-compute-nd-view-copy-contract 120
    just _qa-run qa-fft-family-contracts 120
    just _qa-run qa-compute-csr-prebake-contract 120
    just _qa-run qa-xem-artifact-bundle 120
    just _qa-run qa-xem-consumer-adapters 120
    just _qa-run qa-compute-backend-parity 120
    just _qa-run qa-compute-workloads 120
    just _qa-run qa-compute-bem-adjoint 120
    just _qa-run qa-compute-performance-contract 120
    just _qa-run qa-compute-bem-kernel 120
    just _qa-run qa-compute-blas 120
    just _qa-run qa-fem-validation-matrix 120
    just _qa-run qa-fem-element-contracts 120
    just _qa-run qa-fem-operator-invariants 120
    just _qa-run qa-fem-nonlinear-tangent 120
    just _qa-run qa-structural-beam-balance 120
    just _qa-run qa-structural-beam-modal 120
    just _qa-run qa-pcfan-cfd-invariants 180
    just _qa-run qa-mesher-box-quality 120
    just _qa-run qa-mesher-pathological 120
    just _qa-run qa-bem-basic 180
    just _qa-run qa-fem-analytical-validation 240
    just _qa-run qa-solvers-iterative 180
    just _qa-run qa-solvers-optimization 180
    just _qa-run qa-solvers-direct 120
    just _qa-run qa-solvers-preconditioner-contracts 120
    just _qa-run qa-solvers-robustness 120
    just _qa-run qa-solvers-saddle-point-contract 120
    just _qa-run qa-solvers-convergence-spectrum 120
    {{cargo}} test -p sonium-fem basis::shape::tests::test_evaluate_shape_p2_tetrahedron_partition_of_unity -- --nocapture
    {{cargo}} test -p sonium-fem spaces::h1::tests -- --nocapture

# Unit tests embedded in QA executables. These are opt-in so source-contract
# tests do not compile the 279-binary QA inventory.
[group('qa')]
qa-bin-tests:
    {{cargo}} test -p sonium-qa --features qa-bin-tests --bins

[group('qa')]
qa-bin-tests-legacy:
    {{cargo}} test -p sonium-qa --features qa-bin-tests,legacy-sims --bins

# Full-production regressions intentionally excluded from ordinary `cargo
# test`. These preserve the original validation assertions while keeping the
# default lane bounded in time and memory.
[group('qa')]
qa-nightly-regressions:
    {{cargo}} test -p sonium-fem pressure_driven_channel_reaches_poiseuille -- --ignored
    {{cargo}} test -p sonium-qa --test wolfram_bem_spheres burton_miller_stabilizes_rigid_sphere_at_second_irregular_frequency -- --ignored
    {{cargo}} test -p sonium-qa --test wolfram_bem_spheres burton_miller_resonance_brackets_track_mie_reference -- --ignored
    {{cargo}} test -p sonium-sim-speaker production_reference_path_retains_rom_evidence_for_supported_fixture -- --ignored
    {{cargo}} test -p sonium-sim-speaker test_simulation_hmatrix_matches_dense_stabilized_operator -- --ignored
    {{cargo}} test -p sonium-sim-speaker product_structural_feedback_is_exchanged_and_retained -- --ignored
    {{cargo}} test -p sonium-sim-speaker typed_driver_is_back_loaded_and_reports_electrical_impedance -- --ignored
    {{cargo}} test -p sonium-sim-speaker-modeling body_fitted_ported_product_monolithic_closes_port_pressure_continuity -- --ignored
    {{cargo}} test -p sonium-sim-speaker-modeling tiny_drive_matches_linear_lumped_model -- --ignored
    {{cargo}} test -p sonium-sim-speaker-modeling coupled_energy_ledger_is_aligned_finite_and_closes_interfaces -- --ignored
    {{cargo}} test -p sonium-sim-speaker-modeling coupled_energy_closure_is_stable_under_spatial_refinement -- --ignored
    {{cargo}} test -p sonium-sim-speaker-modeling body_fitted_ported_mesh_retains_mouth_and_tube_wall_patches -- --ignored

# Mandatory physics regression gate for pull requests touching numerical code.
# Covers analytical observables, dense/hierarchical/FMM parity, FEM convergence,
# element-order support, and the CPU reference used by accelerator parity jobs.
[group('qa')]
qa-physics-gate:
    just qa-release-critical
    just _qa-run qa-bem-mie-ci 300
    just _qa-run qa-bem-sota-validation 300
    just _qa-run qa-fem-analytical-validation 420
    {{cargo}} test -p sonium-bem public_builder_matches_explicit_and_dense_across_ka_transition -- --nocapture
    {{cargo}} test -p sonium-fem basis::shape::tests::test_evaluate_shape_p2_tetrahedron_partition_of_unity -- --nocapture
    {{cargo}} test -p sonium-fem spaces::h1::tests -- --nocapture
    just _qa-run qa-fem-cfd-backend-parity 300

# Deeper deterministic physics validation tier. This is intended for nightly CI
# or explicit release validation, not every cargo test run.
[group('qa')]
qa-deep:
    just qa-smoke
    just _qa-run qa-bem-mie-validation 300
    just _qa-run qa-bem-mie-ci 300
    just _qa-run qa-bem-mesh-convergence 300
    just _qa-run qa-bem-material-impedance-validation 120
    just _qa-run qa-bem-sota-validation 300
    just _qa-run qa-bem-mlfmm-audit 180
    just _qa-run qa-bem-directional-mlfmm 180
    just _qa-run qa-bem-mesh2hrtf-parity 300
    just _qa-run qa-bem-numcalc-parity 300
    just _qa-run-features qa-bem-freq-sweep-recycling 300 legacy-sims
    just _qa-run qa-bem-node-scaling 300
    just _qa-run qa-room-modal-validation 120
    just _qa-run qa-room-shoebox-fem-benchmark 120
    just _qa-run qa-speaker-piston-directivity-benchmark 120
    just _qa-run qa-speaker-vibroacoustic-validation 120
    just _qa-run qa-speaker-fem-bem-coupling 300
    just _qa-run qa-speaker-acoustic-rom 300
    just _qa-run qa-speaker-projected-reanalysis 120
    just _qa-run qa-fem-waveholtz-convergence 300
    just _qa-run qa-fem-schwarz-pml-validation 300

# Native room backend evidence gates. Accelerator recipes are fail-closed when
# the corresponding SONIUM_SIM_ROOM_REQUIRE_* variable is set.
[group('qa')]
test-room-x86:
    SONIUM_SIM_ROOM_REQUIRE_NATIVE_X86=1 {{cargo}} test --release -p sonium-sim-room --test native_x86_reference -- --nocapture

[group('qa')]
test-room-x86-cross-arch:
    SONIUM_SIM_ROOM_REQUIRE_NATIVE_X86=1 SONIUM_SIM_ROOM_REQUIRE_CROSS_ARCH_REFERENCE=1 {{cargo}} test --release -p sonium-sim-room --test native_x86_reference -- --nocapture

# Produce the bounded native Linux/x86 SR3 room reference in both effective
# dispatch modes.  The native run is fail-closed on AVX2/FMA, while the scalar
# run is an explicitly labelled same-host baseline.  Arm64 is user-waived for
# this lane; the artifact records that waiver and never turns it into parity.
[group('test')]
test-room-native-x86-sr3:
    #!/usr/bin/env bash
    set -euo pipefail
    repo_root="$PWD"
    test "$(uname -s)" = Linux || { echo "SR3 evidence requires Linux" >&2; exit 2; }
    test "$(uname -m)" = x86_64 || { echo "SR3 evidence requires native x86_64" >&2; exit 2; }
    grep -qw avx2 /proc/cpuinfo || { echo "SR3 native evidence requires AVX2" >&2; exit 2; }
    grep -qw fma /proc/cpuinfo || { echo "SR3 native evidence requires FMA" >&2; exit 2; }
    native="$repo_root/target/sonium-unix-sr3/native"
    scalar="$repo_root/target/sonium-unix-sr3/scalar"
    mkdir -p "$native" "$scalar"
    SONIUM_SIM_ROOM_REQUIRE_NATIVE_X86=1 \
        SONIUM_FASTMATH_REQUIRE_AVX2=1 SONIUM_FASTMATH_FORCE_AVX2=1 \
        SONIUM_SIM_ROOM_WRITE_REFERENCE_ARTIFACT="$native/native-x86-sr3-v2.json" \
        CARGO_TARGET_DIR="$repo_root/.target-unix-sr3" \
        {{cargo}} test --release -p sonium-sim-room --test native_x86_reference -- --nocapture --test-threads=1
    SONIUM_SIM_ROOM_REQUIRE_NATIVE_X86=1 SONIUM_FASTMATH_FORCE_SCALAR=1 \
        SONIUM_SIM_ROOM_WRITE_REFERENCE_ARTIFACT="$scalar/scalar-sr3-v2.json" \
        CARGO_TARGET_DIR="$repo_root/.target-unix-sr3" \
        {{cargo}} test --release -p sonium-sim-room --test native_x86_reference -- --nocapture --test-threads=1
    python3 "$repo_root/crates/sonium-sim-room/benchmarks/check_native_x86_sr3.py" \
        "$native/native-x86-sr3-v2.json" "$scalar/scalar-sr3-v2.json"

# Produce and validate the bounded native Linux/x86 SM3 modeling reference.
# The stable runner executes DriverModel, axisymmetric, periodic-sector, and
# FEM--BEM replay producers, then checks the versioned CPU/f64 artifact.  The
# user-waived Arm64/Windows lane is recorded as a waiver, never as parity.
[group('test')]
test-speaker-modeling-native-x86-sm3:
    exec bash "{{justfile_directory()}}/scripts/run_unix_sm3_native_x86.sh"

# Produce and validate the deterministic native Linux/x86 SK3 speaker golden.
# The producer is fail-closed unless AVX2/FMA are available, records scalar,
# SIMD, and vendor provenance, and emits the schema-v3 full/projected
# DriverModel artifact with a checksum sidecar. Arm64/Windows comparison is
# explicitly waived for this Unix lane; this recipe never relabels a fallback
# as native evidence.
[group('qa')]
test-speaker-native-x86-sk3:
    #!/usr/bin/env bash
    set -euo pipefail
    repo_root="$PWD"
    test "$(uname -s)" = Linux || { echo "SK3 evidence requires Linux" >&2; exit 2; }
    test "$(uname -m)" = x86_64 || { echo "SK3 evidence requires native x86_64" >&2; exit 2; }
    grep -qw avx2 /proc/cpuinfo || { echo "SK3 native evidence requires AVX2" >&2; exit 2; }
    grep -qw fma /proc/cpuinfo || { echo "SK3 native evidence requires FMA" >&2; exit 2; }
    output="$repo_root/target/sonium-unix-sk3"
    mkdir -p "$output"
    SONIUM_OUTPUT_DIR="$output" \
        CARGO_TARGET_DIR="$repo_root/.target-unix-sk3" \
        {{cargo}} run --release -p sonium-qa --features qa-binaries \
        --bin qa-speaker-native-x86-golden
    artifact="$output/speaker_native_x86_golden.json"
    checksum="$artifact.sha256"
    test -s "$artifact" && test -s "$checksum"
    (cd "$output" && sha256sum -c "$(basename "$checksum")")
    python3 "$repo_root/crates/sonium-qa/benchmarks/check_speaker_native_x86_sk3.py" "$artifact"
    test "$(jq -r .schema_version "$artifact")" = 3
    test "$(jq -r .target_arch "$artifact")" = x86_64
    test "$(jq -r '.directivity | length' "$artifact")" = 2
    test "$(jq -r '.driver_model_reference.schema_version' "$artifact")" = 2
    test "$(jq -r '.projected_driver_model_reference.schema_version' "$artifact")" = 2
    jq -e '(.dispatch | contains(["scalar=f64"])) and (.dispatch | contains(["avx2=true"])) and (.dispatch | contains(["fma=true"])) and any(.dispatch[]; startswith("vendor=") and (.[7:] | length > 0) and ((.[7:] | ascii_downcase) != "unknown"))' "$artifact" >/dev/null

# Run the bounded SK2 speaker CUDA workload and retain each product surface in
# its own immutable evidence directory.  The three producers use the real
# CUDA speaker entry points; SONIUM_REQUIRE_CUDA and the explicit backend
# requirement make missing hardware or an accidental fallback a hard failure.
# The runner does not synthesize CUDA artifacts and records exact requested /
# actual provenance in every JSON report.  SM2/SR2 and the shared QA manifest
# are intentionally outside this recipe.
[group('qa')]
test-speaker-cuda-sk2:
    scripts/run_unix_sk2_cuda.sh

# Reconcile the stable native Linux/x86 producer artifacts into one immutable
# QA2 fan-in record.  The FFT producer is run here because its native dispatch
# artifact is inexpensive; all other lanes are consumed from their completed
# producer roots.  Missing or failed required lanes make the recipe fail after
# writing the report, so a partial portfolio can never be promoted as a pass.
# AVX-512 is recorded as unsupported when absent; Arm64/Windows are user-waived
# and Rosetta is excluded from this native evidence class.
[group('qa')]
qa-native-x86-fanin:
    #!/usr/bin/env bash
    set -euo pipefail
    repo_root="$PWD"
    test "$(uname -s)" = Linux || { echo "QA2 requires Linux" >&2; exit 2; }
    test "$(uname -m)" = x86_64 || { echo "QA2 requires native x86_64" >&2; exit 2; }
    grep -qw avx2 /proc/cpuinfo || { echo "QA2 requires AVX2" >&2; exit 2; }
    grep -qw fma /proc/cpuinfo || { echo "QA2 requires FMA" >&2; exit 2; }
    output="$repo_root/target/sonium-unix-qa2"
    mkdir -p "$output"
    RAYON_NUM_THREADS=1 \
        SONIUM_EVIDENCE_OUTPUT="$output/fft-x86-dispatch-v1.json" \
        CARGO_TARGET_DIR="$repo_root/.target-unix-qa2" \
        {{cargo}} run --release -p sonium-fft --bin fft_x86_evidence
    SONIUM_QA_THREAD_POLICY=single \
        SONIUM_QA_AFFINITY_POLICY=unbound \
        SONIUM_QA_NUMA_POLICY=default \
        SONIUM_QA_DETERMINISTIC_MODE=enabled \
        python3 "$repo_root/scripts/check_native_x86_qa2.py" "$output"

[group('qa')]
test-room-cuda:
    SONIUM_SIM_ROOM_REQUIRE_CUDA=1 {{cargo}} test --release -p sonium-sim-room --features cuda --test backend_parity_runtime -- --nocapture

# Stable SR1 production room-FEM CUDA evidence. Each invocation uses the same
# bounded production frequency fixture and a repo-local Cargo/evidence root;
# the first two runs emit structured f64-outer and block-RHS artifacts, while
# the cancellation contract is run as a required typed-error test. No CPU
# fallback is accepted as CUDA evidence.
[group('qa')]
qa-room-sr1-cuda:
    bash scripts/run_unix_sr1_cuda.sh

# Stable SR2 production CPU/CUDA physical-parity and failure-matrix evidence.
# The delegated script sets repo-local Cargo/TMP/runtime paths before Cargo is
# started, avoiding Just's own temporary-script directory and preventing
# shared /tmp or /run/user state from contaminating this lane.
[group('qa')]
qa-room-sr2-cuda:
    bash scripts/run_unix_sr2_cuda.sh

[group('qa')]
test-room-metal:
    SONIUM_SIM_ROOM_REQUIRE_METAL=1 {{cargo}} test --release -p sonium-sim-room --features metal --test backend_parity_runtime -- --nocapture

# Expensive application-level validation tier for room/speaker pipelines.
[group('qa')]
qa-expensive:
    {{cargo}} test -p sonium-sim-room --release --test pipeline_test -- --nocapture
    {{cargo}} test -p sonium-sim-speaker --release simulation::tests::test_run_simulation_single_freq -- --nocapture
    just _qa-run qa-pipeline-physics-validation 300
    just _qa-run qa-speaker-rank-preservation 900

# Full standalone QA binary sweep. The manifest is the source of truth so new
# BLR, room, speaker, pipeline, and solver targets cannot be silently omitted.
[group('qa')]
qa-full: _qa-run-all

[group('qa')]
_qa-run-all:
    #!/usr/bin/env python3
    import pathlib
    import subprocess
    import sys
    import tomllib
    import os

    root = pathlib.Path("{{justfile_directory()}}")
    qa_justfile = root / "Justfile"
    manifest = tomllib.loads((root / "crates/sonium-qa/Cargo.toml").read_text())
    failures = []
    passes = []
    base_env = os.environ.copy()
    base_env.setdefault("SONIUM_QA_THREAD_POLICY", "single")
    base_env.setdefault("SONIUM_QA_AFFINITY_POLICY", "unbound")
    base_env.setdefault("SONIUM_QA_NUMA_POLICY", "default")
    base_env.setdefault("SONIUM_QA_DETERMINISTIC_MODE", "enabled")
    targets = sorted(manifest["bin"], key=lambda target: target["name"])
    for target in targets:
        name = target["name"]
        if name == "qa-solvers-facade-native-x86":
            if subprocess.run(["just", "--justfile", str(qa_justfile), name], cwd=root, env=base_env).returncode == 0:
                passes.append(name)
            else:
                failures.append(name)
            continue
        timeout = (
            "1200"
            if name == "qa-speaker-port-nonlinear"
            else "900"
            if name.startswith("qa-bem-")
            else "300"
        )
        features = target.get("required-features", [])
        just_prefix = ["just", "--justfile", str(qa_justfile)]
        if features:
            cmd = just_prefix + ["_qa-run-features", name, timeout, ",".join(features)]
        else:
            cmd = just_prefix + ["_qa-run", name, timeout]
        case_env = base_env.copy()
        if name == "qa-solvers-external-differential-reference":
            tool_dir = root / "target/sonium-qa-tools"
            tool_dir.mkdir(parents=True, exist_ok=True)
            petsc_reference = tool_dir / "petsc_reference"
            source = root / "crates/sonium-solvers/benches/ffi/petsc_reference.c"
            flags = subprocess.run(
                ["pkg-config", "--cflags", "--libs", "PETSc"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            compile_result = subprocess.run(
                ["mpicc", str(source), "-o", str(petsc_reference), *flags.stdout.split()],
                cwd=root,
            ) if flags.returncode == 0 else flags
            if compile_result.returncode != 0:
                failures.append(name)
                continue
            case_env["SONIUM_PETSC_REFERENCE"] = str(petsc_reference)
            case_env["SONIUM_REQUIRE_PETSC"] = "1"
        if subprocess.run(cmd, cwd=root, env=case_env).returncode == 0:
            passes.append(name)
        else:
            failures.append(name)

    print(f"\nqa-full summary: {len(passes)} passed, {len(failures)} failed")
    for name in failures:
        print(f"  FAIL: {name}")
    if failures:
        sys.exit(1)

[group('qa')]
qa-solvers:
    #!/usr/bin/env bash
    set -uo pipefail
    failures=()
    passes=()
    for t in {{justfile_directory()}}/crates/sonium-qa/bin/solvers_*.rs; do
        tname="qa-$(basename "${t}" .rs | tr '_' '-')"
        if just _qa-run "${tname}" 300; then
            passes+=("${tname}")
        else
            failures+=("${tname}")
        fi
    done
    echo
    echo "qa-solvers summary: ${#passes[@]} passed, ${#failures[@]} failed"
    if [ "${#failures[@]}" -gt 0 ]; then
        for f in "${failures[@]}"; do echo "  FAIL: ${f}"; done
        exit 1
    fi

[group('qa')]
qa-fem:
    #!/usr/bin/env bash
    set -uo pipefail
    failures=()
    passes=()
    for t in {{justfile_directory()}}/crates/sonium-qa/bin/fem_*.rs; do
        tname="qa-$(basename "${t}" .rs | tr '_' '-')"
        if just _qa-run "${tname}" 300; then
            passes+=("${tname}")
        else
            failures+=("${tname}")
        fi
    done
    echo
    echo "qa-fem summary: ${#passes[@]} passed, ${#failures[@]} failed"
    if [ "${#failures[@]}" -gt 0 ]; then
        for f in "${failures[@]}"; do echo "  FAIL: ${f}"; done
        exit 1
    fi

[group('qa')]
qa-bem:
    #!/usr/bin/env bash
    set -uo pipefail
    failures=()
    passes=()
    # 900s per-binary budget: mesh-convergence / mesh-refinement /
    # optimal-beta sweep across many subdivision×ka combinations and
    # genuinely need more than the qa-deep tier's 300s.
    for t in {{justfile_directory()}}/crates/sonium-qa/bin/bem_*.rs; do
        tname="qa-$(basename "${t}" .rs | tr '_' '-')"
        if [ "${tname}" = "qa-bem-freq-sweep-recycling" ]; then
            run_cmd=(just _qa-run-features "${tname}" 900 legacy-sims)
        else
            run_cmd=(just _qa-run "${tname}" 900)
        fi
        if "${run_cmd[@]}"; then
            passes+=("${tname}")
        else
            failures+=("${tname}")
        fi
    done
    echo
    echo "============================================================"
    echo "qa-bem summary: ${#passes[@]} passed, ${#failures[@]} failed"
    echo "============================================================"
    if [ "${#failures[@]}" -gt 0 ]; then
        for f in "${failures[@]}"; do echo "  FAIL: ${f}"; done
        exit 1
    fi

[group('qa')]
_qa-run bin timeout_seconds="180" expected_case_id="":
    #!/usr/bin/env python3
    import json
    import os
    import pathlib
    import subprocess
    import sys
    import tempfile
    import platform
    import time

    bin_name = "{{bin}}"
    timeout_seconds = int("{{timeout_seconds}}")
    expected_case_id = "{{expected_case_id}}"
    cmd = [
        "{{cargo}}", "run", "-p", "sonium-qa", "--release",
        "--features", "qa-binaries", "--bin", bin_name,
    ]
    env = os.environ.copy()
    env.setdefault("SONIUM_OUTPUT_DIR", os.path.join(tempfile.gettempdir(), "sonium-qa"))
    result_dir = pathlib.Path(env.get("SONIUM_QA_RESULT_DIR", os.path.join("target", "sonium-qa-results")))
    result_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    case_id = expected_case_id or f"legacy.{bin_name}"
    case_metadata = {
        "input_checksum": None,
        "reference": None,
        "reference_checksum": None,
        "dimension": None,
        "formulation": None,
        "tolerance": None,
    }
    case_policy = {}
    backend_outcomes_required = False
    manifest_path = pathlib.Path("crates/sonium-qa/validation-manifest.toml")
    if manifest_path.exists():
        import tomllib
        for case in tomllib.loads(manifest_path.read_text()).get("case", []):
            matches_case = case.get("id") == expected_case_id if expected_case_id else case.get("bin") == bin_name
            if matches_case:
                if not expected_case_id:
                    case_id = case["id"]
                case_metadata = {
                    "input_checksum": case.get("input_checksum"),
                    "reference": case.get("reference"),
                    "reference_checksum": case.get("reference_checksum"),
                    "dimension": case.get("dimension"),
                    "formulation": case.get("formulation") or case.get("method"),
                    "tolerance": case.get("tolerance"),
                }
                case_policy = case
                backend_outcomes_required = bool(case.get("backend_outcomes_required", False))
                break

    def select_structured_record(output):
        for line in output.splitlines():
            prefixed = line.startswith("QA_RESULT:")
            payload = line.split(":", 1)[1].strip() if prefixed else line.strip()
            if prefixed or payload.startswith("{"):
                try:
                    candidate = json.loads(payload)
                except json.JSONDecodeError:
                    if not prefixed:
                        continue
                    return {"status": "failed", "message": "invalid QA_RESULT JSON"}
                if not isinstance(candidate, dict) or not {
                    "schema_version", "case_id", "status"
                }.issubset(candidate):
                    continue
                if expected_case_id and candidate.get("case_id") != expected_case_id:
                    continue
                return candidate
        return None

    def emit(status, returncode, output, *, timed_out=False):
        requested_backend = env.get("SONIUM_REQUIRE_BACKEND") or env.get("SONIUM_BACKEND")
        runner_message = output[-4000:].strip() or f"runner exited with status {returncode}"
        metadata = {
            key: value for key, value in case_metadata.items() if value is not None
        }
        record = {
            "schema_version": 1,
            "case_id": case_id,
            "status": status,
            "backend_outcomes": [],
            "metrics": {},
            "thresholds": {},
            "elapsed_ms": round((time.monotonic() - started) * 1000),
            "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.strip() or "unavailable",
            "environment": {"platform": platform.platform(), "python": platform.python_version()},
            "artifacts": [],
            "evidence": {
                **metadata,
                "timings_ms": {},
                "artifact_checksums": {},
            },
            "runtime": {
                "failure_stage": "execution",
                "context_reuse": "not_applicable",
                "counts": {"executed": 0, "passed": 0, "failed": 0, "unsupported": 0},
                "counters": {"allocations": 0, "h2d_transfers": 0, "d2h_transfers": 0, "d2d_transfers": 0, "synchronizations": 0, "launches": 0},
                "memory": {"peak_bytes": 0, "live_bytes": 0},
                "hardware": {},
            },
        }
        if requested_backend:
            record["requested_backend"] = requested_backend
        seed = env.get("SONIUM_QA_SEED")
        if seed:
            record["seed"] = seed
        if status == "skipped":
            record["skip_reason"] = "runner skipped the case"
        elif status == "failed":
            record["message"] = runner_message
        embedded = select_structured_record(output)
        if embedded is not None:
            base_evidence = record["evidence"]
            record.update(embedded)
            record["evidence"] = {
                **base_evidence,
                **{
                    key: value
                    for key, value in (embedded.get("evidence") or {}).items()
                    if value is not None
                },
            }
            # The wrapper's decision is authoritative. A binary may emit a
            # syntactically passing record and still fail a required backend,
            # timeout, or marker check; never let that embedded status
            # overwrite the runner failure in the archived evidence.
            record["status"] = (
                "failed" if status == "failed" else embedded.get("status", status)
            )
            if embedded.get("git_sha") in (None, "", "unavailable"):
                record["git_sha"] = subprocess.run(["git", "rev-parse", "HEAD"], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.strip() or "unavailable"
            record["environment"] = {
                "platform": platform.platform(),
                "python": platform.python_version(),
                **(embedded.get("environment") or {}),
            }
            record["elapsed_ms"] = round((time.monotonic() - started) * 1000)
            if record.get("status") == "failed":
                if not isinstance(record.get("message"), str) or not record["message"].strip():
                    record["message"] = runner_message
                runtime = record.get("runtime")
                if isinstance(runtime, dict) and runtime.get("failure_stage") is None:
                    runtime["failure_stage"] = "validation"
            elif record.get("status") == "skipped" and (
                not isinstance(record.get("skip_reason"), str)
                or not record["skip_reason"].strip()
            ):
                record["skip_reason"] = "runner skipped the case"
        allowed_keys = {
            "schema_version", "case_id", "status", "skip_reason",
            "requested_backend", "actual_backend", "backend_outcomes",
            "metrics", "thresholds", "threshold_directions", "seed",
            "elapsed_ms", "git_sha", "environment", "artifacts", "evidence",
            "runtime", "message",
        }
        record = {key: value for key, value in record.items() if key in allowed_keys}
        for optional in ("skip_reason", "requested_backend", "actual_backend", "seed", "message"):
            if record.get(optional) is None:
                record.pop(optional, None)
        path = result_dir / f"{expected_case_id or bin_name}.json"
        path.write_text(json.dumps(record, indent=2) + "\n")
        print("SONIUM_QA_RESULT: " + json.dumps(record, sort_keys=True))

    print(f"== {bin_name} ({timeout_seconds}s timeout) ==")
    try:
        result = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        print(output, end="")
        emit("failed", 124, output, timed_out=True)
        print(f"\nFAILED: {bin_name} timed out after {timeout_seconds}s", file=sys.stderr)
        sys.exit(124)

    print(result.stdout, end="")

    if result.returncode != 0:
        emit("failed", result.returncode, result.stdout)
        print(f"\nFAILED: {bin_name} exited with status {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)
    if "FAILED" in result.stdout:
        emit("failed", 1, result.stdout)
        print(f"\nFAILED: {bin_name} reported FAILED", file=sys.stderr)
        sys.exit(1)
    def structured_status(output):
        record = select_structured_record(output)
        return record.get("status") if record is not None else None

    def structured_record(output):
        return select_structured_record(output)

    def thresholds_satisfied(record):
        if not record or not record.get("thresholds"):
            return True
        metrics = record.get("metrics") or {}
        directions = record.get("threshold_directions") or {}
        for name, threshold in (record.get("thresholds") or {}).items():
            metric = metrics.get(name)
            direction = directions.get(name)
            if metric is None or direction not in {"minimum", "maximum"}:
                return False
            if direction == "minimum" and metric < threshold:
                return False
            if direction == "maximum" and metric > threshold:
                return False
        return True

    def backend_outcomes_valid(record):
        outcomes = record.get("backend_outcomes") or []
        if not isinstance(outcomes, list):
            return False
        names = set()
        for outcome in outcomes:
            if not isinstance(outcome, dict):
                return False
            backend = outcome.get("backend")
            status = outcome.get("status")
            reason = outcome.get("reason")
            if not isinstance(backend, str) or not backend.strip() or backend in names:
                return False
            names.add(backend)
            if status not in {"passed", "failed", "skipped"}:
                return False
            if status == "passed" and reason not in (None, ""):
                return False
            if status in {"failed", "skipped"} and (
                not isinstance(reason, str) or not reason.strip()
            ):
                return False
        return True

    if structured_status(result.stdout) == "skipped":
        emit("skipped", 0, result.stdout)
        print(f"\nFAILED: {bin_name} reported a required skip", file=sys.stderr)
        sys.exit(1)
    if structured_status(result.stdout) == "failed":
        emit("failed", 1, result.stdout)
        print(f"\nFAILED: {bin_name} emitted a failed QA result", file=sys.stderr)
        sys.exit(1)
    embedded_status = structured_status(result.stdout)
    if embedded_status is not None and embedded_status != "passed":
        emit("failed", 1, result.stdout)
        print(
            f"\nFAILED: {bin_name} emitted unsupported QA result status {embedded_status!r}",
            file=sys.stderr,
        )
        sys.exit(1)
    structured = structured_record(result.stdout)
    if expected_case_id and structured is None:
        emit("failed", 1, result.stdout)
        print(
            f"\nFAILED: {bin_name} did not emit QA_RESULT for {expected_case_id!r}",
            file=sys.stderr,
        )
        sys.exit(1)
    if structured is not None:
        sys.path.insert(0, str(pathlib.Path("scripts")))
        from qa_result_policy import validate_result
        policy_errors = validate_result(
            case_policy or {"id": structured.get("case_id")},
            structured,
            artifact_root=pathlib.Path(env["SONIUM_OUTPUT_DIR"]),
        )
        if policy_errors:
            emit("failed", 1, result.stdout)
            print(
                f"\nFAILED: {bin_name} violated QA result policy: {'; '.join(policy_errors)}",
                file=sys.stderr,
            )
            sys.exit(1)
    if structured and not backend_outcomes_valid(structured):
        emit("failed", 1, result.stdout)
        print(f"\nFAILED: {bin_name} emitted invalid backend outcomes", file=sys.stderr)
        sys.exit(1)
    if backend_outcomes_required and (
        structured is None or not structured.get("backend_outcomes")
    ):
        emit("failed", 1, result.stdout)
        print(
            f"\nFAILED: {bin_name} did not emit required backend outcomes",
            file=sys.stderr,
        )
        sys.exit(1)
    if structured and not thresholds_satisfied(structured):
        emit("failed", 1, result.stdout)
        print(f"\nFAILED: {bin_name} emitted an unsatisfied or ambiguous QA threshold", file=sys.stderr)
        sys.exit(1)
    required_backend = env.get("SONIUM_REQUIRE_BACKEND")
    if required_backend:
        record = structured_record(result.stdout)
        actual_backend = record.get("actual_backend") if record else None
        if record is None or record.get("status") != "passed":
            emit("failed", 1, result.stdout)
            print(
                f"\nFAILED: {bin_name} required backend {required_backend!r} but emitted no passing structured result",
                file=sys.stderr,
            )
            sys.exit(1)
        if actual_backend != required_backend:
            emit("failed", 1, result.stdout)
            print(
                f"\nFAILED: {bin_name} used backend {actual_backend!r}, required {required_backend!r}",
                file=sys.stderr,
            )
            sys.exit(1)
    if "SUCCESS" not in result.stdout and embedded_status != "passed":
        emit("failed", 1, result.stdout)
        print(f"\nFAILED: {bin_name} did not report SUCCESS", file=sys.stderr)
        sys.exit(1)
    emit("passed", 0, result.stdout)

[group('qa')]
_qa-run-features bin timeout_seconds features expected_case_id="":
    #!/usr/bin/env python3
    import json
    import os
    import pathlib
    import subprocess
    import sys
    import tempfile
    import platform
    import time

    bin_name = "{{bin}}"
    timeout_seconds = int("{{timeout_seconds}}")
    feature_values = list(dict.fromkeys(
        [value.strip() for value in "{{features}}".split(",") if value.strip()]
        + ["qa-binaries"]
    ))
    features = ",".join(feature_values)
    expected_case_id = "{{expected_case_id}}"
    cmd = [
        "{{cargo}}", "run", "-p", "sonium-qa", "--release",
        "--features", features,
        "--bin", bin_name,
    ]
    env = os.environ.copy()
    env.setdefault("SONIUM_OUTPUT_DIR", os.path.join(tempfile.gettempdir(), "sonium-qa"))
    if "cuda" in feature_values:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if query.returncode == 0 and query.stdout.strip():
            device, _, driver = query.stdout.splitlines()[0].partition(",")
            env.setdefault("SONIUM_QA_CUDA_DEVICE_NAME", device.strip())
            env.setdefault("SONIUM_QA_CUDA_DRIVER_VERSION", driver.strip())
        env.setdefault("SONIUM_QA_CUDA_TOOLKIT_VERSION", env.get("CUDA_VERSION", "not-reported"))
        if "RTX PRO 6000 Blackwell" in env.get("SONIUM_QA_CUDA_DEVICE_NAME", ""):
            env.setdefault("SONIUM_QA_CUDA_PEAK_FLOPS", "125000000000000")
            env.setdefault("SONIUM_QA_CUDA_MEMORY_BANDWIDTH", "1792000000000")
    result_dir = pathlib.Path(env.get("SONIUM_QA_RESULT_DIR", os.path.join("target", "sonium-qa-results")))
    result_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    case_id = expected_case_id or f"legacy.{bin_name}"
    case_metadata = {
        "input_checksum": None,
        "reference": None,
        "reference_checksum": None,
        "dimension": None,
        "formulation": None,
        "tolerance": None,
    }
    case_policy = {}
    backend_outcomes_required = False
    manifest_path = pathlib.Path("crates/sonium-qa/validation-manifest.toml")
    if manifest_path.exists():
        import tomllib
        for case in tomllib.loads(manifest_path.read_text()).get("case", []):
            matches_case = case.get("id") == expected_case_id if expected_case_id else case.get("bin") == bin_name
            if matches_case:
                if not expected_case_id:
                    case_id = case["id"]
                case_metadata = {
                    "input_checksum": case.get("input_checksum"),
                    "reference": case.get("reference"),
                    "reference_checksum": case.get("reference_checksum"),
                    "dimension": case.get("dimension"),
                    "formulation": case.get("formulation") or case.get("method"),
                    "tolerance": case.get("tolerance"),
                }
                case_policy = case
                backend_outcomes_required = bool(case.get("backend_outcomes_required", False))
                break

    def select_structured_record(output):
        for line in output.splitlines():
            prefixed = line.startswith("QA_RESULT:")
            payload = line.split(":", 1)[1].strip() if prefixed else line.strip()
            if prefixed or payload.startswith("{"):
                try:
                    candidate = json.loads(payload)
                except json.JSONDecodeError:
                    if not prefixed:
                        continue
                    return {"status": "failed", "message": "invalid QA_RESULT JSON"}
                if not isinstance(candidate, dict) or not {
                    "schema_version", "case_id", "status"
                }.issubset(candidate):
                    continue
                if expected_case_id and candidate.get("case_id") != expected_case_id:
                    continue
                return candidate
        return None

    def emit(status, returncode, output, *, timed_out=False):
        requested_backend = env.get("SONIUM_REQUIRE_BACKEND") or env.get("SONIUM_BACKEND")
        runner_message = output[-4000:].strip() or f"runner exited with status {returncode}"
        metadata = {
            key: value for key, value in case_metadata.items() if value is not None
        }
        record = {
            "schema_version": 1,
            "case_id": case_id,
            "status": status,
            "backend_outcomes": [],
            "metrics": {},
            "thresholds": {},
            "elapsed_ms": round((time.monotonic() - started) * 1000),
            "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.strip() or "unavailable",
            "environment": {"platform": platform.platform(), "python": platform.python_version(), "features": features},
            "artifacts": [],
            "evidence": {
                **metadata,
                "timings_ms": {},
                "artifact_checksums": {},
            },
            "runtime": {
                "failure_stage": "execution",
                "context_reuse": "not_applicable",
                "counts": {"executed": 0, "passed": 0, "failed": 0, "unsupported": 0},
                "counters": {"allocations": 0, "h2d_transfers": 0, "d2h_transfers": 0, "d2d_transfers": 0, "synchronizations": 0, "launches": 0},
                "memory": {"peak_bytes": 0, "live_bytes": 0},
                "hardware": {},
            },
        }
        if requested_backend:
            record["requested_backend"] = requested_backend
        seed = env.get("SONIUM_QA_SEED")
        if seed:
            record["seed"] = seed
        if status == "skipped":
            record["skip_reason"] = "runner skipped the case"
        elif status == "failed":
            record["message"] = runner_message
        embedded = select_structured_record(output)
        if embedded is not None:
            base_evidence = record["evidence"]
            record.update(embedded)
            record["evidence"] = {
                **base_evidence,
                **{
                    key: value
                    for key, value in (embedded.get("evidence") or {}).items()
                    if value is not None
                },
            }
            record["status"] = (
                "failed" if status == "failed" else embedded.get("status", status)
            )
            if embedded.get("git_sha") in (None, "", "unavailable"):
                record["git_sha"] = subprocess.run(["git", "rev-parse", "HEAD"], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.strip() or "unavailable"
            record["environment"] = {
                "platform": platform.platform(),
                "python": platform.python_version(),
                **(embedded.get("environment") or {}),
            }
            record["elapsed_ms"] = round((time.monotonic() - started) * 1000)
            if record.get("status") == "failed":
                if not isinstance(record.get("message"), str) or not record["message"].strip():
                    record["message"] = runner_message
                runtime = record.get("runtime")
                if isinstance(runtime, dict) and runtime.get("failure_stage") is None:
                    runtime["failure_stage"] = "validation"
            elif record.get("status") == "skipped" and (
                not isinstance(record.get("skip_reason"), str)
                or not record["skip_reason"].strip()
            ):
                record["skip_reason"] = "runner skipped the case"
        allowed_keys = {
            "schema_version", "case_id", "status", "skip_reason",
            "requested_backend", "actual_backend", "backend_outcomes",
            "metrics", "thresholds", "threshold_directions", "seed",
            "elapsed_ms", "git_sha", "environment", "artifacts", "evidence",
            "runtime", "message",
        }
        record = {key: value for key, value in record.items() if key in allowed_keys}
        for optional in ("skip_reason", "requested_backend", "actual_backend", "seed", "message"):
            if record.get(optional) is None:
                record.pop(optional, None)
        path = result_dir / f"{expected_case_id or bin_name}.json"
        path.write_text(json.dumps(record, indent=2) + "\n")
        print("SONIUM_QA_RESULT: " + json.dumps(record, sort_keys=True))

    print(f"== {bin_name} ({timeout_seconds}s timeout, features={features}) ==")
    try:
        result = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        print(output, end="")
        emit("failed", 124, output, timed_out=True)
        print(f"\nFAILED: {bin_name} timed out after {timeout_seconds}s", file=sys.stderr)
        sys.exit(124)

    print(result.stdout, end="")

    if result.returncode != 0:
        emit("failed", result.returncode, result.stdout)
        print(f"\nFAILED: {bin_name} exited with status {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)
    if "FAILED" in result.stdout:
        emit("failed", 1, result.stdout)
        print(f"\nFAILED: {bin_name} reported FAILED", file=sys.stderr)
        sys.exit(1)
    def structured_status(output):
        record = select_structured_record(output)
        return record.get("status") if record is not None else None

    def structured_record(output):
        return select_structured_record(output)

    def thresholds_satisfied(record):
        if not record or not record.get("thresholds"):
            return True
        metrics = record.get("metrics") or {}
        directions = record.get("threshold_directions") or {}
        for name, threshold in (record.get("thresholds") or {}).items():
            metric = metrics.get(name)
            direction = directions.get(name)
            if metric is None or direction not in {"minimum", "maximum"}:
                return False
            if direction == "minimum" and metric < threshold:
                return False
            if direction == "maximum" and metric > threshold:
                return False
        return True

    def backend_outcomes_valid(record):
        outcomes = record.get("backend_outcomes") or []
        if not isinstance(outcomes, list):
            return False
        names = set()
        for outcome in outcomes:
            if not isinstance(outcome, dict):
                return False
            backend = outcome.get("backend")
            status = outcome.get("status")
            reason = outcome.get("reason")
            if not isinstance(backend, str) or not backend.strip() or backend in names:
                return False
            names.add(backend)
            if status not in {"passed", "failed", "skipped"}:
                return False
            if status == "passed" and reason not in (None, ""):
                return False
            if status in {"failed", "skipped"} and (
                not isinstance(reason, str) or not reason.strip()
            ):
                return False
        return True

    if structured_status(result.stdout) == "skipped":
        emit("skipped", 0, result.stdout)
        print(f"\nFAILED: {bin_name} reported a required skip", file=sys.stderr)
        sys.exit(1)
    if structured_status(result.stdout) == "failed":
        emit("failed", 1, result.stdout)
        print(f"\nFAILED: {bin_name} emitted a failed QA result", file=sys.stderr)
        sys.exit(1)
    embedded_status = structured_status(result.stdout)
    if embedded_status is not None and embedded_status != "passed":
        emit("failed", 1, result.stdout)
        print(
            f"\nFAILED: {bin_name} emitted unsupported QA result status {embedded_status!r}",
            file=sys.stderr,
        )
        sys.exit(1)
    structured = structured_record(result.stdout)
    if expected_case_id and structured is None:
        emit("failed", 1, result.stdout)
        print(
            f"\nFAILED: {bin_name} did not emit QA_RESULT for {expected_case_id!r}",
            file=sys.stderr,
        )
        sys.exit(1)
    if structured is not None:
        sys.path.insert(0, str(pathlib.Path("scripts")))
        from qa_result_policy import validate_result
        policy_errors = validate_result(
            case_policy or {"id": structured.get("case_id")},
            structured,
            artifact_root=pathlib.Path(env["SONIUM_OUTPUT_DIR"]),
        )
        if policy_errors:
            emit("failed", 1, result.stdout)
            print(
                f"\nFAILED: {bin_name} violated QA result policy: {'; '.join(policy_errors)}",
                file=sys.stderr,
            )
            sys.exit(1)
    if structured and not backend_outcomes_valid(structured):
        emit("failed", 1, result.stdout)
        print(f"\nFAILED: {bin_name} emitted invalid backend outcomes", file=sys.stderr)
        sys.exit(1)
    if backend_outcomes_required and (
        structured is None or not structured.get("backend_outcomes")
    ):
        emit("failed", 1, result.stdout)
        print(
            f"\nFAILED: {bin_name} did not emit required backend outcomes",
            file=sys.stderr,
        )
        sys.exit(1)
    if structured and not thresholds_satisfied(structured):
        emit("failed", 1, result.stdout)
        print(f"\nFAILED: {bin_name} emitted an unsatisfied or ambiguous QA threshold", file=sys.stderr)
        sys.exit(1)
    required_backend = env.get("SONIUM_REQUIRE_BACKEND")
    if required_backend:
        record = structured_record(result.stdout)
        actual_backend = record.get("actual_backend") if record else None
        if record is None or record.get("status") != "passed":
            emit("failed", 1, result.stdout)
            print(
                f"\nFAILED: {bin_name} required backend {required_backend!r} but emitted no passing structured result",
                file=sys.stderr,
            )
            sys.exit(1)
        if actual_backend != required_backend:
            emit("failed", 1, result.stdout)
            print(
                f"\nFAILED: {bin_name} used backend {actual_backend!r}, required {required_backend!r}",
                file=sys.stderr,
            )
            sys.exit(1)
    if "SUCCESS" not in result.stdout and embedded_status != "passed":
        emit("failed", 1, result.stdout)
        print(f"\nFAILED: {bin_name} did not report SUCCESS", file=sys.stderr)
        sys.exit(1)
    emit("passed", 0, result.stdout)
