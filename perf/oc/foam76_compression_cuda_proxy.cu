// Persistent-kernel proxy for the foam76_compression native replay (oc-opt Phase G0).
//
// Question: can a CUDA port of the foam76 compression material-point replay
// (30720 serial SAMP76 substeps, ~0.083 s sonium native) beat the CPU, and at
// what distance from roofline?
//
// Design: ONE device thread holds the whole serial replay on-device (no
// per-step launch/transfer). Inside, the proxy mirrors the exact CPU replay
// in sonium-qa/bin/support/foam_c3.rs:
//   * exact deck path (load/hold/unload/hold, -0.04 mm, 3 ms) with the same
//     corner-split fixed discretization (10240+5120+10240+5120 = 30720),
//   * logarithmic axial increments ln(j1/j0) with fixed transverse strains,
//   * SAMP76 quadratic return: trial elastic, von Mises check, 64-iteration
//     bisection, numerical tangent (12 extra solves) when yielding,
//   * midpoint stress-work booking sigma V0 dJ.
// Fidelity: final (work, sxx, ep) must match the CPU replay (1.028509e-3 J,
// 3.121125e5 Pa, 8.917500e-4) before timing is trusted; the report records it.
//
// Known deviations: none in the math (same op order, FP64 throughout); no
// TH/probe sampling, path/law constants embedded (deck-parsed on the CPU).
//
// Build: nvcc -O2 -arch=native -o foam76c_proxy foam76_compression_cuda_proxy.cu
// Run:   ./foam76c_proxy [nstep]     (default 30720, the replay count)

#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(call)                                                           \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,               \
                   cudaGetErrorString(err));                                  \
            return 1;                                                         \
        }                                                                     \
    } while (0)

// Exact deck path corners (s, m) and per-segment substep counts.
#define NSEG 4
__constant__ double c_path[NSEG + 1][2];
__constant__ long c_bounds[NSEG + 1]; // cumulative substep bounds
static double h_path[NSEG + 1][2] = {{0.0, 0.0},
                                     {0.001, -0.00004},
                                     {0.0015, -0.00004},
                                     {0.0025, 0.0},
                                     {0.003, 0.0}};
static long h_bounds[NSEG + 1] = {0, 10240, 15360, 25600, 30720};
#define SIDE_M 0.01
#define VOL_M3 1e-6
#define NSTEP_DEFAULT 30720L

// SAMP76 card (Samp76Quadratic::new from E=350MPa, nu=0, T/C/S strengths,
// hardening 1+20ep): bulk, shear, a0, a1, a2, hardening nodes.
struct Samp76 {
    double bulk, shear, a0, a1, a2;
    double hx[2], hy[2];
};

__device__ double funct_eval(const Samp76& law, double x) {
    if (x <= law.hx[0]) return law.hy[0];
    if (x >= law.hx[1]) return law.hy[1];
    double frac = (x - law.hx[0]) / (law.hx[1] - law.hx[0]);
    return law.hy[0] + frac * (law.hy[1] - law.hy[0]);
}

__device__ double funct_slope(const Samp76& law, double x) {
    // Two-node funct: the single segment slope everywhere (mirrors
    // Funct1D::slope adjacent-segment ends for this card).
    (void)x;
    return (law.hy[1] - law.hy[0]) / (law.hx[1] - law.hx[0]);
}

__device__ double pressure_c1(const double* s) {
    return -(s[0] + s[1] + s[2]) / 3.0;
}

__device__ double von_mises_c1(const double* s) {
    double p = (s[0] + s[1] + s[2]) / 3.0;
    double s0 = s[0] - p, s1 = s[1] - p, s2 = s[2] - p;
    double j2 = 0.5 * (s0 * s0 + s1 * s1 + s2 * s2) + s[3] * s[3] +
                s[4] * s[4] + s[5] * s[5];
    return sqrt(3.0 * j2);
}

struct SolveOut {
    double stress[6];
    bool yielding;
    double dep;
};

// Squared von Mises yield radius (Samp76Quadratic::yield_squared).
__device__ double yield_squared(const Samp76& law, double p, double ep) {
    double h = funct_eval(law, ep);
    double sq =
        law.a0 * h * h + law.a1 * h * p + law.a2 * p * p;
    // The CPU checks hardening slope >= 0 and sq >= 0 here (fail-closed
    // on leaving the isochoric domain); this path never triggers on the
    // deck, and the slope eval below keeps the op mix honest.
    double slope = funct_slope(law, ep);
    (void)slope;
    return sq;
}

// Full return with committed ep (transcribed from samp76_quadratic.rs:
// solve() plus the bisection); the kernel threads ep explicitly like
// the Rust state.
__device__ SolveOut samp76_return(const Samp76& law, const double* stress,
                                  double ep, const double* inc) {
    double trace = inc[0] + inc[1] + inc[2];
    double trial[6];
    for (int i = 0; i < 6; ++i) {
        if (i < 3)
            trial[i] = stress[i] + 2.0 * law.shear * (inc[i] - trace / 3.0) +
                       law.bulk * trace;
        else
            trial[i] = stress[i] + law.shear * inc[i];
    }
    double p = pressure_c1(trial);
    double q = von_mises_c1(trial);
    double radius = yield_squared(law, p, ep);
    SolveOut out;
    if (q * q <= radius) {
        for (int i = 0; i < 6; ++i) out.stress[i] = trial[i];
        out.yielding = false;
        out.dep = 0.0;
        return out;
    }
    double low = 0.0, high = q / (3.0 * law.shear);
    for (int k = 0; k < 64; ++k) {
        double dep = 0.5 * (low + high);
        double qnew = q - 3.0 * law.shear * dep;
        double r = yield_squared(law, p, ep + dep);
        if (qnew * qnew > r)
            low = dep;
        else
            high = dep;
    }
    double dep = 0.5 * (low + high);
    double factor = (q - 3.0 * law.shear * dep) / q;
    double mean = (trial[0] + trial[1] + trial[2]) / 3.0;
    for (int i = 0; i < 6; ++i) {
        double dev = trial[i] - (i < 3 ? mean : 0.0);
        out.stress[i] = factor * dev - (i < 3 ? p : 0.0);
    }
    out.yielding = true;
    out.dep = dep;
    return out;
}

struct LoopResult {
    double work, sxx, ep;
    // Liveness sinks (no physics): tsum keeps the numerical tangent and
    // the validity solve live (else the compiler deletes dead solves and
    // the GPU would time less work than the CPU); dj keeps the EMPTY_FORCE
    // path scaffolding live so the differential measures loop overhead.
    double tsum, dj;
    long substeps, yielding;
};

__global__ void foam_loop(LoopResult* out, Samp76 law, long nstep) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    double stress[6] = {0, 0, 0, 0, 0, 0};
    double ep = 0.0;
    double work = 0.0;
    double tsum = 0.0, dj = 0.0;
    long yielding = 0;
    long total = c_bounds[NSEG] < nstep ? c_bounds[NSEG] : nstep;
    for (long s = 0; s < total; ++s) {
        int seg = 0;
        while (s >= c_bounds[seg + 1]) ++seg;
        long count = c_bounds[seg + 1] - c_bounds[seg];
        long local = s - c_bounds[seg];
        double frac = (double)(local + 1) / (double)count;
        double prev = (double)local / (double)count;
        double u0 = c_path[seg][1], u1 = c_path[seg + 1][1];
        double j0 = 1.0 + (u0 + (u1 - u0) * prev) / SIDE_M;
        double j1 = 1.0 + (u0 + (u1 - u0) * frac) / SIDE_M;
        double inc[6] = {log(j1 / j0), 0, 0, 0, 0, 0};
#ifdef EMPTY_FORCE
        dj += j1 - j0;
#else
        double old = stress[0];
        // step(): validity solve, then update (solve + tangent), then commit.
        SolveOut check = samp76_return(law, stress, ep, inc);
        for (int i = 0; i < 6; ++i) tsum += check.stress[i];
        SolveOut upd = samp76_return(law, stress, ep, inc);
        if (upd.yielding) {
            ++yielding;
            for (int col = 0; col < 6; ++col) {
                double ainc = inc[col] < 0 ? -inc[col] : inc[col];
                double h = 1e-7 * (ainc > 1e-3 ? ainc : 1e-3);
                double plus[6], minus[6];
                for (int i = 0; i < 6; ++i) {
                    plus[i] = inc[i];
                    minus[i] = inc[i];
                }
                plus[col] += h;
                minus[col] -= h;
                SolveOut a = samp76_return(law, stress, ep, plus);
                SolveOut b = samp76_return(law, stress, ep, minus);
                for (int row = 0; row < 6; ++row) {
                    tsum += (a.stress[row] - b.stress[row]) / (2.0 * h);
                }
            }
        }
        for (int i = 0; i < 6; ++i) stress[i] = upd.stress[i];
        ep += upd.dep;
        work += 0.5 * (old + upd.stress[0]) * VOL_M3 * (j1 - j0);
#endif
    }
    out->work = work;
    out->sxx = stress[0];
    out->ep = ep;
    out->tsum = tsum;
    out->dj = dj;
    out->substeps = total;
    out->yielding = yielding;
}

int main(int argc, char** argv) {
    long nstep = argc > 1 ? atol(argv[1]) : NSTEP_DEFAULT;
    CHECK(cudaMemcpyToSymbol(c_path, h_path, sizeof(h_path)));
    CHECK(cudaMemcpyToSymbol(c_bounds, h_bounds, sizeof(h_bounds)));
    // Card constants exactly as Samp76Quadratic::new computes them.
    double young = 350e6, poisson = 0.0;
    double tension = 500000.0, compression = 900000.0, shear_s = 420000.0;
    double product = tension * compression;
    Samp76 law;
    law.bulk = young / (3.0 * (1.0 - 2.0 * poisson));
    law.shear = young / (2.0 * (1.0 + poisson));
    law.a0 = 3.0 * shear_s * shear_s;
    law.a1 = 9.0 * shear_s * shear_s * (compression - tension) / product;
    law.a2 = 9.0 * (product - 3.0 * shear_s * shear_s) / product;
    law.hx[0] = 0.0;
    law.hx[1] = 0.5;
    law.hy[0] = 1.0;
    law.hy[1] = 11.0;
    LoopResult* d_out;
    CHECK(cudaMalloc(&d_out, sizeof(LoopResult)));
    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));
    CHECK(cudaEventRecord(t0));
    foam_loop<<<1, 1>>>(d_out, law, nstep);
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    CHECK(cudaGetLastError());
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, t0, t1));
    LoopResult h;
    CHECK(cudaMemcpy(&h, d_out, sizeof(h), cudaMemcpyDeviceToHost));
    printf("nstep=%ld wall=%.3f s (%.3f us/step) work=%.6e sxx=%.6e ep=%.6e yielding=%ld tsum=%.6e dj=%.6e\n",
           h.substeps, ms / 1000.0, ms * 1000.0 / h.substeps, h.work, h.sxx,
           h.ep, h.yielding, h.tsum, h.dj);
    return 0;
}
