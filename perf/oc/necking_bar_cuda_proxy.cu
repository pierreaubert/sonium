// Persistent-kernel proxy for the necking_bar native explicit step (oc-opt Phase G0).
//
// Question: can a CUDA port of the necking_bar time loop (99 nodes, 40 hex8
// bricks, 1,305,578 steps of NeoHookean finite-strain + velocity-Verlet) beat
// the 56.8 s native CPU run, and at what distance from roofline?
//
// Design: ONE persistent kernel holds the whole time loop on-device (no
// per-step launch/transfer: machine_profile shows the host-stepped floor
// alone would consume most of the 43.5 us/step budget). Inside, the proxy
// mirrors the real port a Phase G1 would write:
//   * exact 10x2x2 necking mesh topology (11x3x3 = 99 nodes, 40 bricks),
//   * 8-color element ordering (5 disjoint elements per color) so parallel
//     force scatter is race-free, with __syncthreads between colors,
//   * per Gauss point: gather 8 nodes, F = x*dN, det, log, NeoHookean
//     stress, scatter (same op mix as eval_law1_internal_finite),
//   * velocity-Verlet update over 99 nodes + tip cloads + root pinning,
//   * fixed dt with a scalar cap check (matches the observed regime where
//     the native driver rides dt_max; a real port's dt reduction would add
//     cost, so this is optimistic for the GPU).
//
// Known deviations (all documented in the report): synthetic material
// constants, no TH/probe sampling, atomic-free scatter via coloring (what a
// real port would do), FP64 throughout like the native code.
//
// FLOP estimate: ~430/Gauss-point x 320 points + ~2k update ~= 140 kFLOP
// per iteration; the roofline numerator uses ncu-measured FP64 ops, not
// this estimate.
//
// Build: nvcc -O2 -arch=native -o necking_proxy necking_proxy.cu
// Run:   ./necking_proxy [nstep]        (default 1305578, the sonium count)
// ncu:   ncu --section SpeedOfLight ./necking_proxy 50000

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

// Exact necking lattice: 11 x 3 x 3 nodes, 10 x 2 x 2 bricks.
#define NX 11
#define NY 3
#define NZ 3
#define NN (NX * NY * NZ) // 99
#define EX 10
#define EY 2
#define EZ 2
#define NE (EX * EY * EZ) // 40
#define NCOLOR 8
#define PER_COLOR (NE / NCOLOR) // 5

#define NID(i, j, k) ((i) * NY * NZ + (j) * NZ + (k))

// Host-side mesh + coloring (built once, copied to constant memory).
static int h_elem[NE][8];
static int h_color_elem[NCOLOR][PER_COLOR];

// Standard hex8 2-point Gauss rule: pts +-1/sqrt(3), dN/dxi = s*0.125 table.
__constant__ double c_dN[8][8][3];
__constant__ int c_elem[NE][8];
__constant__ int c_color[NCOLOR][PER_COLOR];

static double h_dN[8][8][3];

__global__ void __launch_bounds__(128) necking_loop(double* x, double* v, double* f,
                                                   const double* m, long nstep,
                                                   double dt, double mu, double lambda,
                                                   double cload) {
    int tid = threadIdx.x;
    // Per-iteration persistent loop.
    for (long s = 0; s < nstep; ++s) {
        // Phase 0: zero forces cooperatively.
        for (int n = tid; n < NN; n += blockDim.x) {
            f[3 * n] = 0.0;
            f[3 * n + 1] = 0.0;
            f[3 * n + 2] = 0.0;
        }
        __syncthreads();
        // Phase 1: element forces, 8 sequential colors x 5 parallel elements.
        for (int c = 0; c < NCOLOR; ++c) {
#ifdef EMPTY_FORCE
            if (false) {
                int e = 0;
                (void)e;
#else
            if (tid < PER_COLOR) {
                int e = c_color[c][tid];
#endif
                double xe[8][3];
#pragma unroll
                for (int a = 0; a < 8; ++a) {
                    int n = c_elem[e][a];
                    xe[a][0] = x[3 * n];
                    xe[a][1] = x[3 * n + 1];
                    xe[a][2] = x[3 * n + 2];
                }
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    // F = xe^T * dN/dX: cube gradients scaled by 2/dim
                    // (elements are 10 x 5 x 5 mm), so F ~= I at rest.
                    double F[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
                    for (int a = 0; a < 8; ++a)
                        for (int i = 0; i < 3; ++i)
                            for (int j = 0; j < 3; ++j)
                                F[i][j] += xe[a][i] * c_dN[q][a][j];
                    for (int i = 0; i < 3; ++i) {
                        F[i][0] *= 200.0;
                        F[i][1] *= 400.0;
                        F[i][2] *= 400.0;
                    }
                    double det = F[0][0] * (F[1][1] * F[2][2] - F[1][2] * F[2][1]) -
                                 F[0][1] * (F[1][0] * F[2][2] - F[1][2] * F[2][0]) +
                                 F[0][2] * (F[1][0] * F[2][1] - F[1][1] * F[2][0]);
                    double inv = 1.0 / det;
                    double Finv[3][3];
                    Finv[0][0] = (F[1][1] * F[2][2] - F[1][2] * F[2][1]) * inv;
                    Finv[0][1] = (F[0][2] * F[2][1] - F[0][1] * F[2][2]) * inv;
                    Finv[0][2] = (F[0][1] * F[1][2] - F[0][2] * F[1][1]) * inv;
                    Finv[1][0] = (F[1][2] * F[2][0] - F[1][0] * F[2][2]) * inv;
                    Finv[1][1] = (F[0][0] * F[2][2] - F[0][2] * F[2][0]) * inv;
                    Finv[1][2] = (F[0][2] * F[1][0] - F[0][0] * F[1][2]) * inv;
                    Finv[2][0] = (F[1][0] * F[2][1] - F[1][1] * F[2][0]) * inv;
                    Finv[2][1] = (F[0][1] * F[2][0] - F[0][0] * F[2][1]) * inv;
                    Finv[2][2] = (F[0][0] * F[1][1] - F[0][1] * F[1][0]) * inv;
                    double logJ = log(det);
                    double g = lambda * logJ * inv;
                    double P[3][3];
                    for (int i = 0; i < 3; ++i)
                        for (int j = 0; j < 3; ++j)
                            P[i][j] = mu * (F[i][j] - Finv[j][i]) + g * Finv[j][i];
                    // Physical scatter: dN/dX with detJ0 = 2.5e-7/8.
                    double wdet = det * 3.125e-8;
                    for (int a = 0; a < 8; ++a) {
                        int n = c_elem[e][a];
                        double fa0 = P[0][0] * c_dN[q][a][0] * 200.0 +
                                     P[0][1] * c_dN[q][a][1] * 400.0 +
                                     P[0][2] * c_dN[q][a][2] * 400.0;
                        double fa1 = P[1][0] * c_dN[q][a][0] * 200.0 +
                                     P[1][1] * c_dN[q][a][1] * 400.0 +
                                     P[1][2] * c_dN[q][a][2] * 400.0;
                        double fa2 = P[2][0] * c_dN[q][a][0] * 200.0 +
                                     P[2][1] * c_dN[q][a][1] * 400.0 +
                                     P[2][2] * c_dN[q][a][2] * 400.0;
                        // Minus sign: the native assembles forces -= residual,
                        // so the stress divergence is restoring, not driving.
                        f[3 * n] -= fa0 * wdet;
                        f[3 * n + 1] -= fa1 * wdet;
                        f[3 * n + 2] -= fa2 * wdet;
                    }
                }
            }
            __syncthreads();
        }
        // Phase 2: cloads (tip face i=NX-1) + Verlet update + root pinning.
        // Quasi-static ramp like the deck (0 -> 1 over 0.9 s).
        double shape = s * dt / 0.9;
        if (shape > 1.0) shape = 1.0;
        for (int n = tid; n < NN; n += blockDim.x) {
            int i = n / (NY * NZ);
            double fx = f[3 * n], fy = f[3 * n + 1], fz = f[3 * n + 2];
            if (i == NX - 1) fx += cload * shape;
            if (i == 0) {
                fx = 0.0;
                fy = 0.0;
                fz = 0.0;
            }
            double invm = 1.0 / m[n];
            v[3 * n] += dt * fx * invm;
            v[3 * n + 1] += dt * fy * invm;
            v[3 * n + 2] += dt * fz * invm;
            x[3 * n] += dt * v[3 * n];
            x[3 * n + 1] += dt * v[3 * n + 1];
            x[3 * n + 2] += dt * v[3 * n + 2];
        }
        __syncthreads();
    }
}

int main(int argc, char** argv) {
    long nstep = argc > 1 ? atol(argv[1]) : 1305578L;
    // Build the 10x2x2 lattice mesh and its 8-coloring on the host.
    int e = 0;
    int count[NCOLOR] = {0};
    for (int ei = 0; ei < EX; ++ei)
        for (int ej = 0; ej < EY; ++ej)
            for (int ek = 0; ek < EZ; ++ek) {
                int n = 0;
                for (int di = 0; di < 2; ++di)
                    for (int dj = 0; dj < 2; ++dj)
                        for (int dk = 0; dk < 2; ++dk)
                            h_elem[e][n++] = NID(ei + di, ej + dj, ek + dk);
                int c = (ei % 2) * 4 + (ej % 2) * 2 + (ek % 2);
                h_color_elem[c][count[c]++] = e;
                ++e;
            }
    for (int c = 0; c < NCOLOR; ++c)
        if (count[c] != PER_COLOR) {
            printf("coloring error: color %d has %d elements\n", c, count[c]);
            return 1;
        }
    // True hex8 dN/dxi table: 0.125*sa_j*prod_{k!=j}(1+sa_k*sq_k/sqrt3).
    for (int q = 0; q < 8; ++q)
        for (int a = 0; a < 8; ++a)
            for (int j = 0; j < 3; ++j) {
                int sa_j = (a >> (2 - j)) & 1 ? 1 : -1;
                double prod = 1.0;
                for (int k = 0; k < 3; ++k) {
                    if (k == j) continue;
                    int sa_k = (a >> (2 - k)) & 1 ? 1 : -1;
                    int sq_k = (q >> (2 - k)) & 1 ? 1 : -1;
                    prod *= 1.0 + sa_k * sq_k * 0.5773502691896258;
                }
                h_dN[q][a][j] = 0.125 * sa_j * prod;
            }
    CHECK(cudaMemcpyToSymbol(c_dN, h_dN, sizeof(h_dN)));
    CHECK(cudaMemcpyToSymbol(c_elem, h_elem, sizeof(h_elem)));
    CHECK(cudaMemcpyToSymbol(c_color, h_color_elem, sizeof(h_color_elem)));

    // State: lattice coords (0.1 x 0.01 x 0.01 bar), unit-ish masses.
    static double h_x[NN * 3], h_v[NN * 3], h_f[NN * 3], h_m[NN];
    for (int i = 0; i < NX; ++i)
        for (int j = 0; j < NY; ++j)
            for (int k = 0; k < NZ; ++k) {
                int n = NID(i, j, k);
                h_x[3 * n] = 0.01 * i;
                h_x[3 * n + 1] = 0.005 * j - 0.005;
                h_x[3 * n + 2] = 0.005 * k - 0.005;
                h_m[n] = 7800.0 * 2.5e-7; // rho * element volume (nodal lump)
            }
    double *d_x, *d_v, *d_f, *d_m;
    CHECK(cudaMalloc(&d_x, sizeof(h_x)));
    CHECK(cudaMalloc(&d_v, sizeof(h_v)));
    CHECK(cudaMalloc(&d_f, sizeof(h_f)));
    CHECK(cudaMalloc(&d_m, sizeof(h_m)));
    CHECK(cudaMemcpy(d_x, h_x, sizeof(h_x), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_v, h_v, sizeof(h_v), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_m, h_m, sizeof(h_m), cudaMemcpyHostToDevice));

    // Fixed dt at 0.8x the native mean step (stop/steps): the native
    // adaptive driver shrinks dt on energy growth, a stabilizer this
    // fixed-step proxy lacks (at 1.0x mean the top mesh mode sits just
    // past the Verlet limit). Timing per step is dt-independent.
    double dt = 0.8 * 7.66e-7;
    double mu = 76.923e9, lambda = 115.385e9; // E=200GPa, nu=0.3
    double cload = 2222.22;

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));
    CHECK(cudaEventRecord(t0));
    necking_loop<<<1, 128>>>(d_x, d_v, d_f, d_m, nstep, dt, mu, lambda, cload);
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    CHECK(cudaGetLastError());
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CHECK(cudaMemcpy(h_x, d_x, sizeof(h_x), cudaMemcpyDeviceToHost));
    static double h_flast[NN * 3];
    CHECK(cudaMemcpy(h_flast, d_f, sizeof(h_flast), cudaMemcpyDeviceToHost));
    double fmax = 0.0;
    for (int i = 0; i < NN * 3; ++i) {
        double a = h_flast[i] < 0 ? -h_flast[i] : h_flast[i];
        if (a > fmax) fmax = a;
    }
    int tip = NID(NX - 1, 1, 1);
    printf("nstep=%ld wall=%.3f s (%.3f us/step) tip_x=%.6e max|f|=%.3e\n", nstep,
           ms / 1000.0, ms * 1000.0 / nstep, h_x[3 * tip], fmax);
    return 0;
}
