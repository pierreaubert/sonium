// Machine profile probe for the RTX PRO 6000 Blackwell box (oc-opt Phase G).
//
// Unit 5 deck instance of the shared machine probe (see
// sonium/perf/oc-gpu-machine.md): identical measurements to
// necking_bar_cuda_machine.cu, re-run to confirm the ceilings hold
// for the ns_lid_cavity round. Measures the roofline denominators:
//   1. device properties (SMs, clocks, L2, theoretical peaks),
//   2. empty-kernel launch overhead (the host-stepped floor),
//   3. small-transfer latency (4 KB regime) + large pinned BW,
//   4. achievable DRAM STREAM BW + achievable FP64 throughput.
//
// Build: nvcc -O2 -o machine_profile ns_lid_cavity_cuda_machine.cu
// Run:   ./machine_profile   (seconds; no ncu needed)

#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(call)                                                             \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,                 \
                   cudaGetErrorString(err));                                    \
            return 1;                                                           \
        }                                                                       \
    } while (0)

__global__ void empty_kernel() {}

__global__ void stream_triad(const double* __restrict__ a, double* __restrict__ b,
                             double s, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (; i < n; i += stride) b[i] = a[i] * s + 1.0;
}

// Unrolled FP64 MAD chain per thread; each iteration retires 16 MADs = 32 FLOP.
__global__ void fp64_mads(double* out, long iters) {
    double a0 = 1.0001, a1 = 1.0002, a2 = 1.0003, a3 = 1.0004;
    double b0 = 2.0001, b1 = 2.0002, b2 = 2.0003, b3 = 2.0004;
    for (long k = 0; k < iters; ++k) {
        a0 = a0 * b0 + a1; a1 = a1 * b1 + a2;
        a2 = a2 * b2 + a3; a3 = a3 * b3 + a0;
        b0 = b0 * a0 + b1; b1 = b1 * a1 + b2;
        b2 = b2 * a2 + b3; b3 = b3 * a3 + b0;
        a0 = a0 * b1 + a2; a1 = a1 * b2 + a3;
        a2 = a2 * b3 + a0; a3 = a3 * b0 + a1;
        b0 = b0 * a2 + b3; b1 = b1 * a3 + b0;
        b2 = b2 * a0 + b1; b3 = b3 * a1 + b2;
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = a0 + a1 + a2 + a3;
}

static double ms(cudaEvent_t a, cudaEvent_t b) {
    float v = 0;
    cudaEventElapsedTime(&v, a, b);
    return (double)v;
}

int main() {
    int dev = 0;
    CHECK(cudaSetDevice(dev));
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, dev));
    printf("device: %s\n", p.name);
    printf("compute capability: %d.%d\n", p.major, p.minor);
    printf("SMs: %d, max threads/SM: %d, shared/SM: %.0f KB\n", p.multiProcessorCount,
           p.maxThreadsPerMultiProcessor, p.sharedMemPerMultiprocessor / 1024.0);
    printf("L2 cache: %.1f MB\n", p.l2CacheSize / 1048576.0);
    printf("global mem: %.1f GB, bus width: %d bit\n",
           p.totalGlobalMem / 1073741824.0, p.memoryBusWidth);
    printf("(clocks via nvidia-smi; legacy clockRate fields removed in CUDA 13)\n");

    // Warmup to boost clocks.
    empty_kernel<<<1, 1>>>();
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));

    // 1. Launch overhead: 2000 empty launches, device-timed.
    CHECK(cudaEventRecord(t0));
    for (int i = 0; i < 2000; ++i) empty_kernel<<<1, 1>>>();
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    printf("empty-launch: %.3f us/launch (2000 launches, device time)\n",
           ms(t0, t1) * 1000.0 / 2000.0);

    // Host-side launch+sync roundtrip (what a host-stepped loop pays per step).
    CHECK(cudaEventRecord(t0));
    for (int i = 0; i < 500; ++i) {
        empty_kernel<<<1, 1>>>();
        CHECK(cudaDeviceSynchronize());
    }
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    printf("launch+sync roundtrip: %.3f us/step (host-driven floor)\n",
           ms(t0, t1) * 1000.0 / 500.0);

    // 2. Small-transfer latency (4 KB regime; the proxy itself is persistent
    // and device-resident, so this only sizes a hypothetical host-stepped loop).
    double *d_small = nullptr;
    double *h_small = nullptr;
    CHECK(cudaMalloc(&d_small, 4096));
    CHECK(cudaMallocHost(&h_small, 4096));
    CHECK(cudaEventRecord(t0));
    for (int i = 0; i < 1000; ++i) {
        CHECK(cudaMemcpy(d_small, h_small, 4096, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(h_small, d_small, 4096, cudaMemcpyDeviceToHost));
    }
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    printf("4KB H2D+D2H roundtrip: %.3f us (pinned)\n", ms(t0, t1) * 1000.0 / 1000.0);

    // Large pinned bandwidth (machine-profile reference number).
    const size_t big = 256ull << 20;
    double *d_big = nullptr;
    double *h_big = nullptr;
    CHECK(cudaMalloc(&d_big, big));
    CHECK(cudaMallocHost(&h_big, big));
    CHECK(cudaEventRecord(t0));
    CHECK(cudaMemcpy(d_big, h_big, big, cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    printf("H2D pinned BW: %.1f GB/s (256 MB)\n", big / (ms(t0, t1) / 1000.0) / 1e9);
    CHECK(cudaEventRecord(t0));
    CHECK(cudaMemcpy(h_big, d_big, big, cudaMemcpyDeviceToHost));
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    printf("D2H pinned BW: %.1f GB/s (256 MB)\n", big / (ms(t0, t1) / 1000.0) / 1e9);

    // 3. Achievable DRAM STREAM (triad: 2 reads + 1 write per element).
    const size_t n = 64ull << 20; // 64M doubles = 512 MB per array
    double *d_a = nullptr, *d_b = nullptr;
    CHECK(cudaMalloc(&d_a, n * sizeof(double)));
    CHECK(cudaMalloc(&d_b, n * sizeof(double)));
    int blocks = p.multiProcessorCount * 32;
    int threads = 256;
    stream_triad<<<blocks, threads>>>(d_a, d_b, 2.0, n);
    CHECK(cudaEventRecord(t0));
    for (int r = 0; r < 10; ++r) stream_triad<<<blocks, threads>>>(d_a, d_b, 2.0, n);
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    double bytes = 10.0 * n * 3 * sizeof(double);
    printf("STREAM triad: %.0f GB/s (achieved DRAM ceiling)\n",
           bytes / (ms(t0, t1) / 1000.0) / 1e9);

    // 4. Achievable FP64 (proxy and native are both double precision).
    double* d_out = nullptr;
    CHECK(cudaMalloc(&d_out, blocks * threads * sizeof(double)));
    const long iters = 200000;
    fp64_mads<<<blocks, threads>>>(d_out, iters);
    CHECK(cudaEventRecord(t0));
    fp64_mads<<<blocks, threads>>>(d_out, iters);
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    double flops = (double)blocks * threads * iters * 32.0;
    printf("FP64 MAD: %.1f GFLOP/s (achieved compute ceiling)\n",
           flops / (ms(t0, t1) / 1000.0) / 1e9);

    printf("done\n");
    return 0;
}
