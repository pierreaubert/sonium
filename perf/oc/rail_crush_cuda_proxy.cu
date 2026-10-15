// Persistent-kernel proxy for the rail_crush native explicit step (oc-opt Phase G0).
//
// Question: can a CUDA port of the rail_crush time loop (99 nodes, 40 hex8
// bricks, 6528 steps of NeoHookean finite-strain + exact-multiplier rigid
// walls) beat the 0.303 s native CPU run, and at what distance from roofline?
//
// Design: ONE persistent kernel holds the whole time loop on-device (no
// per-step launch/transfer: the machine profile shows the host-stepped floor
// alone would consume most of the 46.4 us/step budget). Inside, the proxy
// mirrors the real port a Phase G1 would write:
//   * exact 10x2x2 rail mesh topology (11x3x3 = 99 nodes, 40 bricks,
//     0.2 x 0.05 x 0.05 m column; same counts as necking, rail dimensions),
//   * 8-color element ordering (5 disjoint elements per color) so parallel
//     force scatter is race-free, with __syncthreads between colors,
//   * per Gauss point: gather 8 nodes, F = I + u*dN, det, log, NeoHookean
//     stress P = mu*F + (lambda*lnJ - mu)*F^-T, scatter with forces -=
//     residual (same op mix and sign convention as q1_element_residual),
//   * position-Verlet stepping exactly like ExplicitStepDriver::step:
//     drift x += dt*v + dt^2/2*a, forces at (x_new, v_half, t_new),
//     v = v_half + dt/2*a_new (one force evaluation per step),
//   * two exact-multiplier rigid walls (no penalty, no stiffness): fixed
//     anvil at x = 0 (+x normal, root-half slaves), crusher at
//     x = 0.2 - 0.1*t (-x normal, tip-half slaves), persistent
//     engagement + one inelastic adoption per transition, mirroring
//     rigid_wall_multiplier and the native adoption block,
//   * fixed dt at 0.8x the native mean step (the native adaptive
//     driver's unlimited shrinkage is a stabilizer a fixed-step proxy
//     lacks); timing per step is dt-independent.
//
// Known deviations (all documented in the report): no TH/probe sampling
// beyond wall-force sums, atomic-free scatter via coloring (what a real
// port would do), serial wall loop on thread 0 (a real port would
// parallelize it; conservative for the GPU), no energy-ledger or
// adoption-work bookkeeping, FP64 throughout like the native code.
//
// FLOP estimate: ~430/Gauss-point x 320 points + ~2k Verlet update +
// ~5k wall multipliers ~= 150 kFLOP per iteration (+-30%); the roofline
// numerator is this estimate, not ncu counters (root-blocked on this box).
//
// Build: nvcc -O2 -arch=native -o rail_proxy rail_crush_cuda_proxy.cu
// Run:   ./rail_proxy [nstep]        (default 8160, the proxy stop count)

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

// Exact rail lattice: 11 x 3 x 3 nodes, 10 x 2 x 2 bricks.
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
#define NSLV0 54 // anvil slaves: axial planes 0..5 (deck group 1)
#define NSLV1 45 // crusher slaves: axial planes 6..10 (deck group 2)

#define NID(i, j, k) ((i) * NY * NZ + (j) * NZ + (k))

// Host-side mesh + coloring (built once, copied to constant memory).
static int h_elem[NE][8];
static int h_color_elem[NCOLOR][PER_COLOR];
static int h_slv0[NSLV0];
static int h_slv1[NSLV1];

// Rail element size: 20 x 25 x 25 mm. F scaling is 2/dim per axis;
// detJ0 = element volume / 8 = 1.5625e-6.
#define FSCX 100.0
#define FSCY 80.0
#define FSCZ 80.0
#define WDET0 1.5625e-6

// Standard hex8 2-point Gauss rule: pts +-1/sqrt(3), dN/dxi = s*0.125 table.
__constant__ double c_dN[8][8][3];
__constant__ int c_elem[NE][8];
__constant__ int c_color[NCOLOR][PER_COLOR];
__constant__ int c_slv0[NSLV0];
__constant__ int c_slv1[NSLV1];
__constant__ double c_x0[NN * 3];

static double h_dN[8][8][3];

__global__ void __launch_bounds__(128) rail_loop(
    double* x, double* v, double* vh, double* fi, double* fe, double* a,
    const double* m, unsigned char* eng0, unsigned char* eng1, double* sums,
    double* final0, double* final1, long nstep, double dt, double mu,
    double lambda) {
    int tid = threadIdx.x;
    double dt2 = 0.5 * dt * dt;
    double hdt = 0.5 * dt;
    for (long s = 0; s < nstep; ++s) {
        double t = (s + 1) * dt; // forces live at end-of-step time
        // Phase 0: drift + half-kick, cooperatively.
        for (int n = tid; n < NN; n += blockDim.x) {
            double vx = v[3 * n], vy = v[3 * n + 1], vz = v[3 * n + 2];
            double ax = a[3 * n], ay = a[3 * n + 1], az = a[3 * n + 2];
            vh[3 * n] = vx + hdt * ax;
            vh[3 * n + 1] = vy + hdt * ay;
            vh[3 * n + 2] = vz + hdt * az;
            x[3 * n] += dt * vx + dt2 * ax;
            x[3 * n + 1] += dt * vy + dt2 * ay;
            x[3 * n + 2] += dt * vz + dt2 * az;
        }
        __syncthreads();
        // Phase 1: zero internal + external forces cooperatively.
        for (int n = tid; n < NN; n += blockDim.x) {
            fi[3 * n] = 0.0;
            fi[3 * n + 1] = 0.0;
            fi[3 * n + 2] = 0.0;
            fe[3 * n] = 0.0;
            fe[3 * n + 1] = 0.0;
            fe[3 * n + 2] = 0.0;
        }
        __syncthreads();
        // Phase 2: element forces, 8 sequential colors x 5 parallel elements.
        for (int c = 0; c < NCOLOR; ++c) {
#ifdef EMPTY_FORCE
            if (false) {
                int e = 0;
                (void)e;
#else
            if (tid < PER_COLOR) {
                int e = c_color[c][tid];
#endif
                double ue[8][3];
#pragma unroll
                for (int aa = 0; aa < 8; ++aa) {
                    int n = c_elem[e][aa];
                    ue[aa][0] = x[3 * n] - c_x0[3 * n];
                    ue[aa][1] = x[3 * n + 1] - c_x0[3 * n + 1];
                    ue[aa][2] = x[3 * n + 2] - c_x0[3 * n + 2];
                }
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    // F = I + u*dN/dX with the rail axis scalings, exactly
                    // like q1_element_residual (u-form gives bit-exact
                    // zero bulk at rest; F-from-xe leaves mu*eps residue
                    // that engages the touching anvil one step early).
                    double F[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
                    const double fsc[3] = {FSCX, FSCY, FSCZ};
                    for (int aa = 0; aa < 8; ++aa)
                        for (int i = 0; i < 3; ++i)
                            for (int j = 0; j < 3; ++j)
                                F[i][j] += ue[aa][i] * c_dN[q][aa][j] * fsc[j];
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
                    // P = mu*F + (lambda*lnJ - mu)*F^-T, like stress_into.
                    double coeff = lambda * logJ - mu;
                    double P[3][3];
                    for (int i = 0; i < 3; ++i)
                        for (int j = 0; j < 3; ++j)
                            P[i][j] = mu * F[i][j] + coeff * Finv[j][i];
                    // Physical scatter: dN/dX with detJ0, Gauss weight 1.
                    double wdet = det * WDET0;
                    for (int aa = 0; aa < 8; ++aa) {
                        int n = c_elem[e][aa];
                        double fa0 = P[0][0] * c_dN[q][aa][0] * FSCX +
                                     P[0][1] * c_dN[q][aa][1] * FSCY +
                                     P[0][2] * c_dN[q][aa][2] * FSCZ;
                        double fa1 = P[1][0] * c_dN[q][aa][0] * FSCX +
                                     P[1][1] * c_dN[q][aa][1] * FSCY +
                                     P[1][2] * c_dN[q][aa][2] * FSCZ;
                        double fa2 = P[2][0] * c_dN[q][aa][0] * FSCX +
                                     P[2][1] * c_dN[q][aa][1] * FSCY +
                                     P[2][2] * c_dN[q][aa][2] * FSCZ;
                        // Minus sign: the native assembles forces -= residual,
                        // so the stress divergence is restoring, not driving.
                        fi[3 * n] -= fa0 * wdet;
                        fi[3 * n + 1] -= fa1 * wdet;
                        fi[3 * n + 2] -= fa2 * wdet;
                    }
                }
            }
            __syncthreads();
        }
        // Phase 3: exact-multiplier walls on thread 0 (serial wall loop;
        // a G1 port would parallelize this over slaves, so the proxy wall
        // cost is conservative). EMPTY_FORCE keeps the wall checks on zero
        // bulk, so the differential isolates the bulk force math.
        int trans0[NSLV0];
        int trans1[NSLV1];
        int ntrans0 = 0, ntrans1 = 0;
        if (tid == 0) {
            // Wall 0: fixed anvil, point x = 0, normal +x, velocity 0.
            // Wall 1: crusher, point x = 0.2 - 0.1*t, normal -x,
            // velocity -0.1 m/s. Both frictionless, zero acceleration.
            double p1 = 0.2 - 0.1 * t;
            double ow0[3] = {0, 0, 0};
            double ow1[3] = {0, 0, 0};
            for (int slot = 0; slot < NSLV0; ++slot) {
                int n = c_slv0[slot];
                double gap = x[3 * n]; // (x - 0) . +x
                double bulk = fi[3 * n] + fe[3 * n];
                double mult = -bulk; // m*a_wall - bulk, a_wall = 0
                double rel = vh[3 * n];
                if (!eng0[slot]) {
                    if (gap > 0.0 || (mult <= 0.0 && rel >= 0.0)) continue;
                    eng0[slot] = 1;
                    trans0[ntrans0++] = slot;
                } else if (mult <= 0.0 && rel >= 0.0) {
                    eng0[slot] = 0;
                    continue;
                } else if (gap > 0.0 && rel >= 0.0) {
                    eng0[slot] = 0;
                    continue;
                }
                double applied = mult > 0.0 ? mult : 0.0;
                fe[3 * n] += applied;
                ow0[0] -= applied;
            }
            for (int slot = 0; slot < NSLV1; ++slot) {
                int n = c_slv1[slot];
                double gap = -(x[3 * n] - p1); // (x - p) . -x
                double bulk = -(fi[3 * n] + fe[3 * n]);
                double mult = -bulk;
                double rel = -(vh[3 * n] + 0.1);
                if (!eng1[slot]) {
                    if (gap > 0.0 || (mult <= 0.0 && rel >= 0.0)) continue;
                    eng1[slot] = 1;
                    trans1[ntrans1++] = slot;
                } else if (mult <= 0.0 && rel >= 0.0) {
                    eng1[slot] = 0;
                    continue;
                } else if (gap > 0.0 && rel >= 0.0) {
                    eng1[slot] = 0;
                    continue;
                }
                double applied = mult > 0.0 ? mult : 0.0;
                fe[3 * n] -= applied;
                ow1[0] += applied;
            }
            final0[0] = ow0[0];
            final0[1] = ow0[1];
            final0[2] = ow0[2];
            final1[0] = ow1[0];
            final1[1] = ow1[1];
            final1[2] = ow1[2];
            if (s >= nstep / 2) {
                sums[0] += ow0[0];
                sums[1] += ow1[0];
                sums[2] += 1.0;
            }
        }
        __syncthreads();
        // Phase 4: full-step velocity + acceleration, cooperatively.
        for (int n = tid; n < NN; n += blockDim.x) {
            double invm = 1.0 / m[n];
            double anx = (fi[3 * n] + fe[3 * n]) * invm;
            double any = (fi[3 * n + 1] + fe[3 * n + 1]) * invm;
            double anz = (fi[3 * n + 2] + fe[3 * n + 2]) * invm;
            v[3 * n] = vh[3 * n] + hdt * anx;
            v[3 * n + 1] = vh[3 * n + 1] + hdt * any;
            v[3 * n + 2] = vh[3 * n + 2] + hdt * anz;
            a[3 * n] = anx;
            a[3 * n + 1] = any;
            a[3 * n + 2] = anz;
        }
        __syncthreads();
        // Phase 5: adoption of this step's fresh engagements (one
        // inelastic reset per transition, like the native block).
        if (tid == 0) {
            for (int k = 0; k < ntrans0; ++k) {
                int slot = trans0[k];
                if (!eng0[slot]) continue;
                int n = c_slv0[slot];
                double rel = v[3 * n]; // anvil velocity 0, normal +x
                if (rel < 0.0) v[3 * n] -= rel;
            }
            for (int k = 0; k < ntrans1; ++k) {
                int slot = trans1[k];
                if (!eng1[slot]) continue;
                int n = c_slv1[slot];
                double rel = -(v[3 * n] + 0.1); // crusher -0.1, normal -x
                if (rel < 0.0) v[3 * n] += rel; // v -= rel*n, n = -x
            }
        }
        __syncthreads();
    }
}

int main(int argc, char** argv) {
    long nstep = argc > 1 ? atol(argv[1]) : 8160L;
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
    // Wall slaves from the deck groups: planes 0..5 anvil, 6..10 crusher.
    int n0 = 0, n1 = 0;
    for (int i = 0; i < NX; ++i)
        for (int j = 0; j < NY; ++j)
            for (int k = 0; k < NZ; ++k) {
                if (i <= 5)
                    h_slv0[n0++] = NID(i, j, k);
                else
                    h_slv1[n1++] = NID(i, j, k);
            }
    if (n0 != NSLV0 || n1 != NSLV1) {
        printf("slave error: %d + %d, want %d + %d\n", n0, n1, NSLV0, NSLV1);
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
    CHECK(cudaMemcpyToSymbol(c_slv0, h_slv0, sizeof(h_slv0)));
    CHECK(cudaMemcpyToSymbol(c_slv1, h_slv1, sizeof(h_slv1)));

    // State: rail lattice coords (0.2 x 0.05 x 0.05 m column), lumped
    // masses like the native arm (rho * vol / 8 per brick-node).
    static double h_x[NN * 3], h_v[NN * 3], h_a[NN * 3], h_m[NN];
    for (int i = 0; i < NX; ++i)
        for (int j = 0; j < NY; ++j)
            for (int k = 0; k < NZ; ++k) {
                int n = NID(i, j, k);
                h_x[3 * n] = 0.02 * i;
                h_x[3 * n + 1] = 0.025 * j - 0.025;
                h_x[3 * n + 2] = 0.025 * k - 0.025;
                h_m[n] = 0.0;
            }
    CHECK(cudaMemcpyToSymbol(c_x0, h_x, sizeof(h_x)));
    const double vol = 0.02 * 0.025 * 0.025;
    for (int ee = 0; ee < NE; ++ee)
        for (int a = 0; a < 8; ++a) h_m[h_elem[ee][a]] += 7800.0 * vol / 8.0;
    double *d_x, *d_v, *d_vh, *d_fi, *d_fe, *d_a, *d_m;
    unsigned char *d_eng0, *d_eng1;
    double *d_sums, *d_final0, *d_final1;
    CHECK(cudaMalloc(&d_x, sizeof(h_x)));
    CHECK(cudaMalloc(&d_v, sizeof(h_v)));
    CHECK(cudaMalloc(&d_vh, sizeof(h_x)));
    CHECK(cudaMalloc(&d_fi, sizeof(h_x)));
    CHECK(cudaMalloc(&d_fe, sizeof(h_x)));
    CHECK(cudaMalloc(&d_a, sizeof(h_a)));
    CHECK(cudaMalloc(&d_m, sizeof(h_m)));
    CHECK(cudaMalloc(&d_eng0, NSLV0));
    CHECK(cudaMalloc(&d_eng1, NSLV1));
    CHECK(cudaMalloc(&d_sums, 3 * sizeof(double)));
    CHECK(cudaMalloc(&d_final0, 3 * sizeof(double)));
    CHECK(cudaMalloc(&d_final1, 3 * sizeof(double)));
    CHECK(cudaMemcpy(d_x, h_x, sizeof(h_x), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_v, h_v, sizeof(h_v), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_a, h_a, sizeof(h_a), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_m, h_m, sizeof(h_m), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_eng0, 0, NSLV0));
    CHECK(cudaMemset(d_eng1, 0, NSLV1));
    CHECK(cudaMemset(d_sums, 0, 3 * sizeof(double)));

    // Fixed dt at 0.8x the native mean step (stop/steps): the native
    // adaptive driver shrinks dt on energy growth, a stabilizer this
    // fixed-step proxy lacks. Timing per step is dt-independent.
    double dt = 0.8 * (0.02 / 6528.0);
    double mu = 76.92307692307693e9, lambda = 115.38461538461539e9; // E=200GPa, nu=0.3

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));
    CHECK(cudaEventRecord(t0));
    rail_loop<<<1, 128>>>(d_x, d_v, d_vh, d_fi, d_fe, d_a, d_m, d_eng0, d_eng1,
                          d_sums, d_final0, d_final1, nstep, dt, mu, lambda);
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    CHECK(cudaGetLastError());
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CHECK(cudaMemcpy(h_x, d_x, sizeof(h_x), cudaMemcpyDeviceToHost));
    static double h_filast[NN * 3];
    CHECK(cudaMemcpy(h_filast, d_fi, sizeof(h_filast), cudaMemcpyDeviceToHost));
    double fmax = 0.0;
    for (int i = 0; i < NN * 3; ++i) {
        double a = h_filast[i] < 0 ? -h_filast[i] : h_filast[i];
        if (a > fmax) fmax = a;
    }
    static unsigned char h_eng0[NSLV0], h_eng1[NSLV1];
    CHECK(cudaMemcpy(h_eng0, d_eng0, NSLV0, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_eng1, d_eng1, NSLV1, cudaMemcpyDeviceToHost));
    int e0 = 0, e1 = 0;
    for (int i = 0; i < NSLV0; ++i) e0 += h_eng0[i] ? 1 : 0;
    for (int i = 0; i < NSLV1; ++i) e1 += h_eng1[i] ? 1 : 0;
    double h_sums[3], h_final0[3], h_final1[3];
    CHECK(cudaMemcpy(h_sums, d_sums, sizeof(h_sums), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_final0, d_final0, sizeof(h_final0), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_final1, d_final1, sizeof(h_final1), cudaMemcpyDeviceToHost));
    int tip = NID(NX - 1, 1, 1);
    double mean1 = h_sums[2] > 0 ? h_sums[1] / h_sums[2] : 0.0;
    printf("nstep=%ld wall=%.3f s (%.3f us/step) tip_dx=%.6e\n", nstep, ms / 1000.0,
           ms * 1000.0 / nstep, h_x[3 * tip] - 0.2);
    printf("fnx_crush_final=%.6e fnx_crush_mean2nd=%.6e fnx_anvil_final=%.6e\n",
           h_final1[0], mean1, h_final0[0]);
    printf("engaged=%d+%d max|fi|=%.3e\n", e0, e1, fmax);
    return 0;
}
