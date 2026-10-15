#!/usr/bin/env bash
# Runs the OpenCourant explicit-dynamics perf matrix via qa-oc-bench.
#
# Usage: scripts/perf_oc_matrix.sh [out_dir]
#
# One qa-oc-bench invocation per (deck, side, flavor, threads) cell,
# serially on purpose: engine runs race under concurrent load (vendor
# OpenMP nondeterminism observed), and wall times need a quiet box.
# Cells are additive: only the selected decks are overwritten, other
# bundles in out_dir are left alone.
#
# Env:
#   PERF_OC_DECKS      space-separated deck ids (default: every deck
#                      dir under crates/sonium-qa/decks holding the C3
#                      trio `<id>_0000.rad`, `<id>_0001.rad`,
#                      `probes.json`; dirs without the trio are listed
#                      and skipped, loudly).
#   PERF_OC_QUICK=1    one repeat, no warmup everywhere (CI/smoke).
#   PERF_OC_HERTZ_FULL=1
#                      full protocol for hertz_sphere too (default is
#                      quick: the sonium arm needs ~360 s/repeat).
#   CARGO              cargo binary (default: cargo).
#   OPENCOURANT_EXE / OPENCOURANT_MINIQA (+ LD_LIBRARY_PATH,
#   RAD_CFG_PATH) select the vendor install for engine cells.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
output="${1:-${repo_root}/perf/oc}"
cargo_bin="${CARGO:-cargo}"
quick_all="${PERF_OC_QUICK:-0}"
hertz_full="${PERF_OC_HERTZ_FULL:-0}"

# Default deck set: every deck dir holding the C3 trio, so units
# adding arms only drop files in place. Non-conforming dirs are
# skipped loudly (never silently) to keep the default honest.
discover_decks() {
    local dir id found=""
    for dir in "${repo_root}/crates/sonium-qa/decks/"*/; do
        [ -d "${dir}" ] || continue
        id="$(basename "${dir}")"
        if [ -f "${dir}${id}_0000.rad" ] && [ -f "${dir}${id}_0001.rad" ] \
            && [ -f "${dir}probes.json" ]; then
            found="${found} ${id}"
        else
            echo "skip ${id}: no C3 trio (${id}_0000.rad + ${id}_0001.rad + probes.json)" >&2
        fi
    done
    echo "${found}"
}

if [ -n "${PERF_OC_DECKS:-}" ]; then
    decks="${PERF_OC_DECKS}"
else
    decks="$(discover_decks)"
fi
if [ -z "${decks// /}" ]; then
    echo "no decks resolve (empty PERF_OC_DECKS and no C3 trio under crates/sonium-qa/decks)" >&2
    exit 1
fi
echo "decks:${decks}"

mkdir -p "${output}"
cd "${repo_root}"

failed_cells=""
run_cell() {
    local deck="$1" side="$2" flavor="$3" threads="$4" file="$5" quick="$6"
    local args=(--deck "${deck}" --side "${side}" --threads "${threads}"
        --emit-bundle "${output}/${file}")
    if [ -n "${flavor}" ]; then
        args+=(--flavor "${flavor}")
    fi
    if [ "${side}" = "engine" ] && [ "${flavor}" = "mpi" ]; then
        args+=(--mpi-ranks "${threads}")
    fi
    if [ "${quick}" = "1" ]; then
        args+=(--quick)
    fi
    echo "== ${file}"
    # A failing cell must not abort the matrix: its bundle still lands
    # (correctness_passed=false) and the publisher skips it by policy,
    # so the failure is recorded here and reported at the end.
    if ! "${cargo_bin}" run -q --release -p sonium-qa \
        --features qa-binaries --bin qa-oc-bench -- "${args[@]}"; then
        echo "FAILED CELL ${file}"
        failed_cells="${failed_cells} ${file}"
    fi
}

for deck in ${decks}; do
    quick="${quick_all}"
    if [ "${deck}" = "hertz_sphere" ] && [ "${hertz_full}" != "1" ]; then
        quick=1
    fi
    run_cell "${deck}" sonium "" 1 "${deck}_sonium_native_t1.json" "${quick}"
    run_cell "${deck}" sonium "" 16 "${deck}_sonium_native_t16.json" "${quick}"
    run_cell "${deck}" engine serial 1 "${deck}_engine_serial_t1.json" "${quick}"
    run_cell "${deck}" engine smp 1 "${deck}_engine_smp_t1.json" "${quick}"
    run_cell "${deck}" engine smp 4 "${deck}_engine_smp_t4.json" "${quick}"
    run_cell "${deck}" engine smp 16 "${deck}_engine_smp_t16.json" "${quick}"
    run_cell "${deck}" engine mpi 2 "${deck}_engine_mpi_r2.json" "${quick}"
    run_cell "${deck}" engine mpi 4 "${deck}_engine_mpi_r4.json" "${quick}"
done
if [ -n "${failed_cells}" ]; then
    echo "matrix finished with failing cells:${failed_cells}"
    exit 1
fi
