// Persistent-kernel proxy for the foam33_cube native replay (oc-opt Phase G0).
//
// Question: can a CUDA port of the foam33 material-point replay (30720
// serial LAW33 substeps, ~0.045 s sonium native) beat the CPU, and at
// what distance from roofline?
//
// Design: ONE device thread holds the whole serial replay on-device (no
// per-step launch/transfer). Inside, the proxy mirrors the exact CPU replay
// in sonium-qa/bin/support/foam_c3.rs:
//   * exact deck path (compression/hold/unload/hold, -0.5 mm, 3 ms) with
//     the same corner-split fixed discretization (30720 substeps),
//   * logarithmic axial increments ln(j1/j0) with geometrical volume j1,
//   * LAW33 plain card: trapped-air pressure, principal cap via 12-sweep
//     Jacobi eigensolve, numerical tangent (12 extra solves) when yielding,
//   * midpoint stress-work booking sigma V0 dJ.
// Fidelity: final (work, sxx) must match the CPU replay (5.326268e-2 J,
// 5.0e5 Pa) before timing is trusted; the report records it. The ep tally
// is informational only (the engine PLAS variable stays zero on this
// fixture, so no engine-comparable ep exists).
//
// Known deviations: none in the math (same op order, FP64 throughout); no
// TH/probe sampling, path/law constants embedded (deck-parsed on the CPU).
//
// Build: nvcc -O2 -arch=native -o foam33_proxy foam33_cube_cuda_proxy.cu
// Run:   ./foam33_proxy [nstep]     (default 30720, the replay count)

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
                                     {0.001, -0.0005},
                                     {0.0015, -0.0005},
                                     {0.0025, 0.0},
                                     {0.003, 0.0}};
static long h_bounds[NSEG + 1] = {0, 10240, 15360, 25600, 30720};
#define SIDE_M 0.01
#define VOL_M3 1e-6
#define NSTEP_DEFAULT 30720L

// Plain LAW33 card (FoamPlas33::new: E, A, B, C, P0, phi, gamma0).
struct Law33 {
    double young, a, b, c, p0, phi, gamma0;
};

struct AirCap {
    double air, cap;
};

__device__ AirCap pressure_and_cap(const Law33& law, double j) {
    double gamma = j - 1.0 + law.gamma0;
    AirCap out;
    double air = -law.p0 * gamma / (1.0 + gamma - law.phi);
    out.air = air > 0.0 ? air : 0.0;
    double cap = law.a + law.b * (1.0 + law.c * gamma);
    out.cap = cap < 0 ? -cap : cap;
    return out;
}

// Symmetric 3x3 Jacobi eigensolve, 12 sweeps (jacobi_eigen_c1).
__device__ void jacobi_eigen(const double* s, double* values,
                             double vectors[3][3]) {
    double a[3][3] = {{s[0], s[3], s[5]},
                      {s[3], s[1], s[4]},
                      {s[5], s[4], s[2]}};
    double v[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    const int pairs[3][2] = {{0, 1}, {0, 2}, {1, 2}};
    for (int sweep = 0; sweep < 12; ++sweep) {
        for (int t = 0; t < 3; ++t) {
            int p = pairs[t][0], q = pairs[t][1];
            double apq = a[p][q];
            if (apq == 0.0) continue;
            double phi = 0.5 * atan2(2.0 * apq, a[q][q] - a[p][p]);
            double c = cos(phi), sn = sin(phi);
            for (int row = 0; row < 3; ++row) {
                double akp = a[row][p], akq = a[row][q];
                a[row][p] = c * akp - sn * akq;
                a[row][q] = sn * akp + c * akq;
            }
            for (int k = 0; k < 3; ++k) {
                double old_p = a[p][k], old_q = a[q][k];
                a[p][k] = c * old_p - sn * old_q;
                a[q][k] = sn * old_p + c * old_q;
            }
            for (int row = 0; row < 3; ++row) {
                double vkp = v[row][p], vkq = v[row][q];
                v[row][p] = c * vkp - sn * vkq;
                v[row][q] = sn * vkp + c * vkq;
            }
        }
    }
    values[0] = a[0][0];
    values[1] = a[1][1];
    values[2] = a[2][2];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) vectors[i][j] = v[i][j];
}

__device__ double voigt_equivalent(const double* d) {
    double mean = (d[0] + d[1] + d[2]) / 3.0;
    double d0 = d[0] - mean, d1 = d[1] - mean, d2 = d[2] - mean;
    double shear = 0.5 * (d[3] * d[3] + d[4] * d[4] + d[5] * d[5]);
    return sqrt((2.0 / 3.0) * (d0 * d0 + d1 * d1 + d2 * d2 + shear));
}

struct SolveOut {
    double stress[6];
    bool yielding;
    double dep;
};

// LAW33 solve (transcribed from foam_plas33.rs).
__device__ SolveOut law33_solve(const Law33& law, const double* stress,
                                const double* strain, double j) {
    AirCap ac = pressure_and_cap(law, j);
    double trial[6];
    for (int i = 0; i < 6; ++i) {
        trial[i] = stress[i] +
                   law.young * strain[i] * (i < 3 ? 1.0 : 0.5) +
                   (i < 3 ? ac.air : 0.0);
    }
    double values[3], vectors[3][3];
    jacobi_eigen(trial, values, vectors);
    double capped[3];
    bool yielding = false;
    for (int k = 0; k < 3; ++k) {
        double v = values[k];
        double c = v;
        if (c < -ac.cap) c = -ac.cap;
        if (c > ac.cap) c = ac.cap;
        capped[k] = c;
        if (v != c) yielding = true;
    }
    double tensor[9];
    for (int i = 0; i < 9; ++i) {
        int row = i / 3, col = i % 3;
        tensor[i] = vectors[row][0] * capped[0] * vectors[col][0] +
                    vectors[row][1] * capped[1] * vectors[col][1] +
                    vectors[row][2] * capped[2] * vectors[col][2];
    }
    double skeleton[6] = {tensor[0], tensor[4], tensor[8],
                          tensor[1], tensor[5], tensor[2]};
    SolveOut out;
    double rejected[6];
    for (int i = 0; i < 6; ++i) {
        out.stress[i] = skeleton[i] - (i < 3 ? ac.air : 0.0);
        rejected[i] =
            (trial[i] - skeleton[i]) / law.young * (i < 3 ? 1.0 : 2.0);
    }
    out.yielding = yielding;
    out.dep = voigt_equivalent(rejected);
    return out;
}

struct LoopResult {
    double work, sxx, ep;
    // Liveness sinks (no physics): tsum keeps the numerical tangent live
    // (else the compiler deletes dead solves and the GPU would time less
    // work than the CPU); dj keeps the EMPTY_FORCE scaffolding live.
    double tsum, dj;
    long substeps, yielding;
};

__global__ void foam_loop(LoopResult* out, Law33 law, long nstep) {
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
        double strain[6] = {log(j1 / j0), 0, 0, 0, 0, 0};
#ifdef EMPTY_FORCE
        dj += j1 - j0;
#else
        double old = stress[0];
        // step(): cap check, then update (solve + tangent), then commit.
        AirCap ac = pressure_and_cap(law, j1);
        (void)ac;
        SolveOut upd = law33_solve(law, stress, strain, j1);
        if (upd.yielding) {
            ++yielding;
            for (int col = 0; col < 6; ++col) {
                double as = strain[col] < 0 ? -strain[col] : strain[col];
                double h = 1e-7 * (as > 1e-3 ? as : 1e-3);
                double plus[6], minus[6];
                for (int i = 0; i < 6; ++i) {
                    plus[i] = strain[i];
                    minus[i] = strain[i];
                }
                plus[col] += h;
                minus[col] -= h;
                SolveOut a = law33_solve(law, stress, plus, j1);
                SolveOut b = law33_solve(law, stress, minus, j1);
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
    Law33 law;
    law.young = 350e6;
    law.a = -177000.0;
    law.b = 677000.0;
    law.c = -3.34;
    law.p0 = 1e5;
    law.phi = 0.1;
    law.gamma0 = 0.0;
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
