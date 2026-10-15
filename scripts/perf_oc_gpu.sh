#!/usr/bin/env bash
# Runs one deck's Phase G0 CUDA proxy and archives the evidence.
#
# Usage: scripts/perf_oc_gpu.sh <deck> [out_dir]
#
# G0 proxy-first flow (oc-opt methodology): before any full CUDA port,
# a deck-scoped persistent-kernel proxy at exact deck topology runs
# device-timed at short step counts (proven-linear projection, never a
# full-N shared-GPU hog), plus an empty-force differential build. No
# sonium.cuda explicit arm exists, so this script only ever builds the
# checked-in proxy source with nvcc; it never assumes a native CUDA arm.
#
# Inputs (checked in under perf/oc):
#   <deck>_cuda_proxy.cu      the G0 proxy (required; fail closed)
#   <deck>_cuda_machine.cu    deck machine profile, else the shared
#                             cuda_machine.cu, else skipped with a note
# Outputs (next to the bundles in out_dir):
#   <deck>_cuda_proxy.txt     console archive (machine, N=1, N=steps,
#                             empty-force differential, optional ncu)
#   <deck>_cuda_ncu.txt       ncu SpeedOfLight capture (only with
#                             PERF_OC_GPU_NCU=1 and permitted counters)
#   <deck>_cuda_roofline.md   archived from PERF_OC_GPU_REPORT when set;
#                             otherwise the deck unit authors it at that
#                             path (this script never fabricates it)
#
# Env:
#   PERF_OC_GPU_STEPS    timed proxy steps (default 2000; short-run
#                        projection per proxy discipline)
#   PERF_OC_GPU_SMOKE=1  build + N=1 fidelity only (no timed runs)
#   PERF_OC_GPU_FORCE=1  skip the GPU-idle gate
#   PERF_OC_GPU_REPORT   roofline report file to archive
#   PERF_OC_GPU_NCU=1    attempt an ncu SpeedOfLight capture (best
#                        effort: root-blocked on this box, device-timed
#                        walls stand either way)
#   PERF_OC_GPU_NCU_STEPS
#                        proxy steps under ncu (default 50)
#   NVCC               nvcc binary (default: nvcc)
#   NVCC_ARCH          nvcc -arch value (default: native)
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
deck="${1:-}"
output="${2:-${repo_root}/perf/oc}"
if [ "$#" -gt 2 ]; then
    echo "perf-oc-gpu takes <deck> [out_dir] only; steps come from PERF_OC_GPU_STEPS, ignoring nothing (got extra '${3}')" >&2
    exit 1
fi
steps="${PERF_OC_GPU_STEPS:-2000}"
smoke="${PERF_OC_GPU_SMOKE:-0}"
force="${PERF_OC_GPU_FORCE:-0}"
report="${PERF_OC_GPU_REPORT:-}"
ncu="${PERF_OC_GPU_NCU:-0}"
ncu_steps="${PERF_OC_GPU_NCU_STEPS:-50}"
nvcc_bin="${NVCC:-nvcc}"
nvcc_arch="${NVCC_ARCH:-native}"

case "${deck}" in
    '' | *'/'* | *'..'*)
        echo "perf-oc-gpu needs a plain deck id, got '${deck}'" >&2
        exit 1
        ;;
esac
case "${steps}" in
    '' | *[!0-9]* | 0)
        echo "PERF_OC_GPU_STEPS must be a positive integer, got '${steps}'" >&2
        exit 1
        ;;
esac

proxy_src="${repo_root}/perf/oc/${deck}_cuda_proxy.cu"
if [ ! -f "${proxy_src}" ]; then
    echo "no G0 proxy for deck ${deck}: ${proxy_src} missing (the deck unit authors it)" >&2
    exit 1
fi
machine_src=""
for candidate in "${repo_root}/perf/oc/${deck}_cuda_machine.cu" \
    "${repo_root}/perf/oc/cuda_machine.cu"; do
    if [ -f "${candidate}" ]; then
        machine_src="${candidate}"
        break
    fi
done

if ! command -v "${nvcc_bin}" >/dev/null 2>&1; then
    echo "nvcc not found (${nvcc_bin}); G0 proxy needs a CUDA toolkit" >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found; G0 proxy needs an NVIDIA GPU" >&2
    exit 1
fi
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)"
if [ -z "${gpu_name}" ]; then
    echo "no NVIDIA GPU visible to nvidia-smi" >&2
    exit 1
fi
if [ "${force}" != "1" ]; then
    busy="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
        | awk '$1 > 5 { print $1 }' | head -n 1)"
    others="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | grep -E '[0-9]' | head -n 1 || true)"
    if [ -n "${busy}" ] || [ -n "${others}" ]; then
        echo "GPU busy (util ${busy:-0}%+, compute pid ${others:-none}); refusing timed proxy run (PERF_OC_GPU_FORCE=1 overrides)" >&2
        exit 1
    fi
fi

mkdir -p "${output}"
scratch_base="${TMPDIR:-$HOME/src/all_of_sonium/tmp}"
work="$(mktemp -d "${scratch_base}/perf_oc_gpu_${deck}_XXXXXX")"
console="${output}/${deck}_cuda_proxy.txt"
{
    echo "deck: ${deck}"
    echo "proxy: ${proxy_src}"
    echo "nvcc: $("${nvcc_bin}" --version | tail -n 1)"
    echo "gpu: ${gpu_name}"
    echo "steps: ${steps}"
} >"${console}"

echo "== build ${deck} proxy"
"${nvcc_bin}" -O2 -arch="${nvcc_arch}" -o "${work}/proxy" "${proxy_src}"
"${nvcc_bin}" -O2 -arch="${nvcc_arch}" -DEMPTY_FORCE -o "${work}/proxy_empty" "${proxy_src}"

if [ -n "${machine_src}" ]; then
    echo "== machine profile ${machine_src##*/}"
    "${nvcc_bin}" -O2 -arch="${nvcc_arch}" -o "${work}/machine" "${machine_src}"
    {
        echo "=== machine_profile ==="
        "${work}/machine"
    } >>"${console}" 2>&1
else
    echo "note: no deck or shared machine profile source; Unit 5 profile applies" | tee -a "${console}"
fi

echo "== proxy N=1 (fidelity smoke)"
{
    echo "=== proxy N=1 ==="
    "${work}/proxy" 1
} >>"${console}" 2>&1

if [ "${smoke}" != "1" ]; then
    echo "== proxy N=${steps} (device-timed)"
    {
        echo "=== proxy N=${steps} ==="
        "${work}/proxy" "${steps}"
        echo "=== proxy EMPTY N=${steps} ==="
        "${work}/proxy_empty" "${steps}"
    } >>"${console}" 2>&1
fi

if [ "${ncu}" = "1" ]; then
    echo "== ncu SpeedOfLight (best effort)"
    if ! command -v ncu >/dev/null 2>&1; then
        echo "ncu requested but not on PATH; skipped" | tee -a "${console}"
    elif ncu --section SpeedOfLight "${work}/proxy" "${ncu_steps}" >"${work}/ncu.txt" 2>&1; then
        {
            echo "=== ncu SpeedOfLight N=${ncu_steps} ==="
            cat "${work}/ncu.txt"
        } >>"${console}" 2>&1
        cp "${work}/ncu.txt" "${output}/${deck}_cuda_ncu.txt"
    else
        {
            echo "=== ncu SpeedOfLight N=${ncu_steps} (FAILED) ==="
            echo "ncu capture failed (this box needs root NVreg_RestrictProfilingToAdminHost=0); device-timed walls stand"
            tail -n 20 "${work}/ncu.txt"
        } | tee -a "${console}"
    fi
fi

roofline="${output}/${deck}_cuda_roofline.md"
if [ -n "${report}" ]; then
    if [ ! -f "${report}" ]; then
        echo "PERF_OC_GPU_REPORT points at missing file ${report}" >&2
        exit 1
    fi
    cp "${report}" "${roofline}"
    echo "archived roofline report ${roofline}"
else
    echo "roofline report slot: ${roofline} (authored by the deck unit)"
fi
echo "console archive: ${console}"
