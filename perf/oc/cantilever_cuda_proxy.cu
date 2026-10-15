// Persistent-kernel proxy for the cantilever native explicit step (oc-opt Phase G0).
//
// Question: can a CUDA port of the cantilever time loop (99 nodes, 40 hex8
// bricks, 261,116 steps of EAS-linear stiffness + velocity-Verlet) beat the
// ~3.1 s native CPU run, and at what distance from roofline?
//
// Design: ONE persistent kernel holds the whole time loop on-device (no
// per-step launch/transfer: the necking machine profile shows the
// host-stepped floor alone would eat the ~12 us/step budget). Inside, the
// proxy mirrors the real port a Phase G1 would write:
//   * exact 10x2x2 cantilever mesh topology (11x3x3 = 99 nodes, 40 bricks
//     of 0.1 x 0.025 x 0.025 m),
//   * one EAS-condensed 24x24 element stiffness (built on the host with the
//     same incompatible-modes condensation as sonium-fem eas_linear_stiffness;
//     all deck bricks are congruent), applied as a dgemv per element,
//   * 8-color element ordering (5 disjoint elements per color) so parallel
//     force scatter is race-free, with __syncthreads between colors,
//   * velocity-Verlet update over 99 nodes + tip cloads (Z, 11.111 N/node
//     x 9 nodes = 100 N with the deck FUNCT ramp shape(t) = min(t, 1)) +
//     root pinning (zero force rows, like the native arm),
//   * fixed dt at 0.8x the native mean step (3.83 us -> 3.06 us); the
//     native adaptive driver rides dt_max = 4.26 us, so this is stable
//     and conservative (a real port's dt logic would add cost).
//
// Known deviations (all documented in the report): no TH/probe sampling,
// atomic-free scatter via coloring (what a real port would do), FP64
// throughout like the native code.
//
// FLOP estimate: ~2300/element dgemv x 40 + ~2k update ~= 94 kFLOP/step.
//
// Fidelity anchors: N=1 forces vanish (no prestress); the rigid-shift null
// check (ke_check) verifies the host stiffness; the full-N tip displacement
// must land within ~2% of the CPU sonium final (3.124e-4 m) and the EAS
// static replica (3.136e-4 m).
//
// Build: nvcc -O2 -arch=native -o cantilever_proxy cantilever_cuda_proxy.cu
// Run:   ./cantilever_proxy [nstep]   (default 326371, t = 1.0 s at fixed dt)

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

// Exact cantilever lattice: 11 x 3 x 3 nodes, 10 x 2 x 2 bricks.
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

static int h_elem[NE][8];
static int h_color_elem[NCOLOR][PER_COLOR];

// Slot signs in di + 2*dj + 4*dk order.
static const int S8[8][3] = {{-1, -1, -1}, {1, -1, -1}, {-1, 1, -1}, {1, 1, -1},
                              {-1, -1, 1},  {1, -1, 1},  {-1, 1, 1},  {1, 1, 1}};
// Q1/E9 incompatible modes as 3x3 row-major (mirrors eas.rs).
static void enhancement_modes(double xi, double eta, double zeta, double m[9][9]) {
    for (int k = 0; k < 9; ++k)
        for (int i = 0; i < 9; ++i) m[k][i] = 0.0;
    m[0][0] = xi;
    m[1][4] = eta;
    m[2][8] = zeta;
    m[3][1] = xi;
    m[3][3] = xi;
    m[4][1] = eta;
    m[4][3] = eta;
    m[5][5] = eta;
    m[5][7] = eta;
    m[6][5] = zeta;
    m[6][7] = zeta;
    m[7][2] = xi;
    m[7][6] = xi;
    m[8][2] = zeta;
    m[8][6] = zeta;
}

static void mat3_mul(const double a[9], const double b[9], double out[9]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            out[i * 3 + j] =
                a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
}

static double det3(const double m[9]) {
    return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) +
           m[2] * (m[3] * m[7] - m[4] * m[6]);
}

static void inv3(const double m[9], double inv[9]) {
    double idet = 1.0 / det3(m);
    inv[0] = (m[4] * m[8] - m[5] * m[7]) * idet;
    inv[1] = (m[2] * m[7] - m[1] * m[8]) * idet;
    inv[2] = (m[1] * m[5] - m[2] * m[4]) * idet;
    inv[3] = (m[5] * m[6] - m[3] * m[8]) * idet;
    inv[4] = (m[0] * m[8] - m[2] * m[6]) * idet;
    inv[5] = (m[2] * m[3] - m[0] * m[5]) * idet;
    inv[6] = (m[3] * m[7] - m[4] * m[6]) * idet;
    inv[7] = (m[1] * m[6] - m[0] * m[7]) * idet;
    inv[8] = (m[0] * m[4] - m[1] * m[3]) * idet;
}

// In-place 9x9 solve with partial pivoting.
static void solve9(double a[81], double b[9]) {
    for (int c = 0; c < 9; ++c) {
        int piv = c;
        double best = a[c * 9 + c] < 0 ? -a[c * 9 + c] : a[c * 9 + c];
        for (int r = c + 1; r < 9; ++r) {
            double v = a[r * 9 + c] < 0 ? -a[r * 9 + c] : a[r * 9 + c];
            if (v > best) {
                best = v;
                piv = r;
            }
        }
        for (int k = c; k < 9; ++k) {
            double t = a[c * 9 + k];
            a[c * 9 + k] = a[piv * 9 + k];
            a[piv * 9 + k] = t;
        }
        {
            double t = b[c];
            b[c] = b[piv];
            b[piv] = t;
        }
        double d = a[c * 9 + c];
        for (int r = c + 1; r < 9; ++r) {
            double f = a[r * 9 + c] / d;
            a[r * 9 + c] = 0.0;
            for (int k = c + 1; k < 9; ++k) a[r * 9 + k] -= f * a[c * 9 + k];
            b[r] -= f * b[c];
        }
    }
    for (int r = 8; r >= 0; --r) {
        double s = b[r];
        for (int k = r + 1; k < 9; ++k) s -= a[r * 9 + k] * b[k];
        b[r] = s / a[r * 9 + r];
    }
}

// Condensed EAS stiffness for one axis-aligned brick, mirroring
// sonium-fem eas_linear_stiffness (Voigt B, push-forward G, D solve).
static void eas_ke(double lambda, double mu, double hx, double hy, double hz,
                   double ke[576]) {
    double X[8][3];
    for (int a = 0; a < 8; ++a) {
        X[a][0] = (S8[a][0] + 1) * 0.5 * hx;
        X[a][1] = (S8[a][1] + 1) * 0.5 * hy;
        X[a][2] = (S8[a][2] + 1) * 0.5 * hz;
    }
    // Centroid map.
    double dc[8][3];
    for (int a = 0; a < 8; ++a)
        for (int j = 0; j < 3; ++j) {
            double p = 1.0;
            for (int k = 0; k < 3; ++k) {
                if (k == j) continue;
                p *= 1.0;
            }
            dc[a][j] = 0.125 * S8[a][j] * p;
        }
    double j0[9] = {0};
    for (int a = 0; a < 8; ++a)
        for (int i = 0; i < 3; ++i)
            for (int c = 0; c < 3; ++c) j0[i * 3 + c] += X[a][i] * dc[a][c];
    double j0det = det3(j0), j0inv[9];
    inv3(j0, j0inv);
    double kuu[576] = {0}, kua[24][9] = {{0}}, d[81] = {0};
    double lame2 = lambda + 2.0 * mu;
    const double g1 = 0.5773502691896258;
    for (int q = 0; q < 8; ++q) {
        double xi[3] = {(q & 1) ? g1 : -g1, (q & 2) ? g1 : -g1, (q & 4) ? g1 : -g1};
        double dn[8][3];
        for (int a = 0; a < 8; ++a)
            for (int j = 0; j < 3; ++j) {
                double p = 1.0;
                for (int k = 0; k < 3; ++k) {
                    if (k == j) continue;
                    p *= 1.0 + S8[a][k] * xi[k];
                }
                dn[a][j] = 0.125 * S8[a][j] * p;
            }
        double jac[9] = {0};
        for (int a = 0; a < 8; ++a)
            for (int i = 0; i < 3; ++i)
                for (int c = 0; c < 3; ++c) jac[i * 3 + c] += X[a][i] * dn[a][c];
        double det = det3(jac), inv[9];
        inv3(jac, inv);
        double grad[8][3];
        for (int a = 0; a < 8; ++a)
            for (int i = 0; i < 3; ++i)
                grad[a][i] = inv[i] * dn[a][0] + inv[3 + i] * dn[a][1] + inv[6 + i] * dn[a][2];
        double b[6][24] = {{0}};
        for (int a = 0; a < 8; ++a) {
            b[0][3 * a] = grad[a][0];
            b[1][3 * a + 1] = grad[a][1];
            b[2][3 * a + 2] = grad[a][2];
            b[3][3 * a] = grad[a][1];
            b[3][3 * a + 1] = grad[a][0];
            b[4][3 * a + 1] = grad[a][2];
            b[4][3 * a + 2] = grad[a][1];
            b[5][3 * a] = grad[a][2];
            b[5][3 * a + 2] = grad[a][0];
        }
        double m[9][9];
        enhancement_modes(xi[0], xi[1], xi[2], m);
        double scale = j0det / det, g[6][9];
        for (int k = 0; k < 9; ++k) {
            double pushed[9];
            mat3_mul(m[k], j0inv, pushed);
            g[0][k] = scale * pushed[0];
            g[1][k] = scale * pushed[4];
            g[2][k] = scale * pushed[8];
            g[3][k] = scale * (pushed[1] + pushed[3]);
            g[4][k] = scale * (pushed[5] + pushed[7]);
            g[5][k] = scale * (pushed[2] + pushed[6]);
        }
        for (int r = 0; r < 24; ++r) {
            double cb[6] = {
                lame2 * b[0][r] + lambda * (b[1][r] + b[2][r]),
                lame2 * b[1][r] + lambda * (b[0][r] + b[2][r]),
                lame2 * b[2][r] + lambda * (b[0][r] + b[1][r]),
                mu * b[3][r],
                mu * b[4][r],
                mu * b[5][r],
            };
            for (int c = 0; c < 24; ++c)
                kuu[r * 24 + c] += det * (b[0][c] * cb[0] + b[1][c] * cb[1] + b[2][c] * cb[2] +
                                          b[3][c] * cb[3] + b[4][c] * cb[4] + b[5][c] * cb[5]);
            for (int k = 0; k < 9; ++k)
                kua[r][k] += det * (g[0][k] * cb[0] + g[1][k] * cb[1] + g[2][k] * cb[2] +
                                    g[3][k] * cb[3] + g[4][k] * cb[4] + g[5][k] * cb[5]);
        }
        for (int k = 0; k < 9; ++k) {
            double cg[6] = {
                lame2 * g[0][k] + lambda * (g[1][k] + g[2][k]),
                lame2 * g[1][k] + lambda * (g[0][k] + g[2][k]),
                lame2 * g[2][k] + lambda * (g[0][k] + g[1][k]),
                mu * g[3][k],
                mu * g[4][k],
                mu * g[5][k],
            };
            for (int l = 0; l < 9; ++l)
                d[k * 9 + l] += det * (g[0][l] * cg[0] + g[1][l] * cg[1] + g[2][l] * cg[2] +
                                       g[3][l] * cg[3] + g[4][l] * cg[4] + g[5][l] * cg[5]);
        }
    }
    for (int c = 0; c < 24; ++c) {
        double rhs[9], mat[81];
        for (int k = 0; k < 9; ++k) rhs[k] = kua[c][k];
        for (int i = 0; i < 81; ++i) mat[i] = d[i];
        solve9(mat, rhs);
        for (int r = 0; r < 24; ++r) {
            double corr = 0.0;
            for (int k = 0; k < 9; ++k) corr += kua[r][k] * rhs[k];
            kuu[r * 24 + c] -= corr;
        }
    }
    for (int i = 0; i < 576; ++i) ke[i] = kuu[i];
}

__constant__ double c_ke[576];
__constant__ double c_x0[NN * 3];
__constant__ int c_elem[NE][8];
__constant__ int c_color[NCOLOR][PER_COLOR];

__global__ void __launch_bounds__(128) cantilever_loop(double* x, double* v, double* f,
                                                       const double* m, long nstep,
                                                       double dt, double cload) {
    int tid = threadIdx.x;
    for (long s = 0; s < nstep; ++s) {
        for (int n = tid; n < NN; n += blockDim.x) {
            f[3 * n] = 0.0;
            f[3 * n + 1] = 0.0;
            f[3 * n + 2] = 0.0;
        }
        __syncthreads();
        for (int c = 0; c < NCOLOR; ++c) {
#ifdef EMPTY_FORCE
            if (false) {
                int e = 0;
                (void)e;
#else
            if (tid < PER_COLOR) {
                int e = c_color[c][tid];
#endif
                double ue[24];
#pragma unroll
                for (int a = 0; a < 8; ++a) {
                    int n = c_elem[e][a];
                    ue[3 * a] = x[3 * n] - c_x0[3 * n];
                    ue[3 * a + 1] = x[3 * n + 1] - c_x0[3 * n + 1];
                    ue[3 * a + 2] = x[3 * n + 2] - c_x0[3 * n + 2];
                }
                double fe[24] = {0};
                for (int r = 0; r < 24; ++r) {
                    double s = 0.0;
#pragma unroll
                    for (int k = 0; k < 24; ++k) s += c_ke[r * 24 + k] * ue[k];
                    fe[r] = s;
                }
#pragma unroll
                for (int a = 0; a < 8; ++a) {
                    int n = c_elem[e][a];
                    // Minus sign: the native assembles forces -= residual.
                    f[3 * n] -= fe[3 * a];
                    f[3 * n + 1] -= fe[3 * a + 1];
                    f[3 * n + 2] -= fe[3 * a + 2];
                }
            }
            __syncthreads();
        }
        // Deck FUNCT ramp shape(t) = min(t, 1); tip cload in Z; root pinned.
        double shape = (s + 1) * dt;
        if (shape > 1.0) shape = 1.0;
        for (int n = tid; n < NN; n += blockDim.x) {
            int i = n / (NY * NZ);
            double fx = f[3 * n], fy = f[3 * n + 1], fz = f[3 * n + 2];
            if (i == NX - 1) fz += cload * shape;
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
    double dt = 0.8 * (1.0 / 261116.0);
    long nstep = argc > 1 ? atol(argv[1]) : (long)(1.0 / dt);
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
    // NOTE: host slot order here is di + 2*dj + 4*dk while the mesh loop
    // above fills slots in di-outer order; the gather below reindexes so
    // device slots match the S8 convention used by eas_ke.
    static int h_slot[NE][8];
    for (int ee = 0; ee < NE; ++ee)
        for (int a = 0; a < 8; ++a) {
            int di = a / 4, dj = (a / 2) % 2, dk = a % 2;
            h_slot[ee][di + 2 * dj + 4 * dk] = h_elem[ee][a];
        }
    static double h_ke[576];
    double young = 200.0e9, nu = 0.3;
    double lambda = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    double mu = young / (2.0 * (1.0 + nu));
    eas_ke(lambda, mu, 0.1, 0.025, 0.025, h_ke);
    // Stiffness self-checks: checksum + rigid-shift null residual.
    double kesum = 0.0, kemax = 0.0;
    for (int i = 0; i < 576; ++i) {
        kesum += h_ke[i];
        double a = h_ke[i] < 0 ? -h_ke[i] : h_ke[i];
        if (a > kemax) kemax = a;
    }
    double nullmax = 0.0;
    for (int r = 0; r < 24; ++r) {
        double s = 0.0;
        for (int k = 0; k < 24; ++k) s += h_ke[r * 24 + k];
        if (s < 0) s = -s;
        if (s > nullmax) nullmax = s;
    }
    printf("ke_check: sum=%.6e max=%.6e null=%.3e ke0=%.6e ke100=%.6e ke575=%.6e\n", kesum,
           kemax, nullmax, h_ke[0], h_ke[100], h_ke[575]);
    CHECK(cudaMemcpyToSymbol(c_ke, h_ke, sizeof(h_ke)));
    CHECK(cudaMemcpyToSymbol(c_elem, h_slot, sizeof(h_slot)));
    CHECK(cudaMemcpyToSymbol(c_color, h_color_elem, sizeof(h_color_elem)));

    static double h_x[NN * 3], h_v[NN * 3], h_f[NN * 3], h_m[NN];
    for (int i = 0; i < NX; ++i)
        for (int j = 0; j < NY; ++j)
            for (int k = 0; k < NZ; ++k) {
                int n = NID(i, j, k);
                h_x[3 * n] = 0.1 * i;
                h_x[3 * n + 1] = 0.025 * j - 0.025;
                h_x[3 * n + 2] = 0.025 * k - 0.025;
                h_m[n] = 0.0;
            }
    double ve = 0.1 * 0.025 * 0.025;
    for (int ee = 0; ee < NE; ++ee)
        for (int a = 0; a < 8; ++a) h_m[h_elem[ee][a]] += 7800.0 * ve / 8.0;
    CHECK(cudaMemcpyToSymbol(c_x0, h_x, sizeof(h_x)));
    double *d_x, *d_v, *d_f, *d_m;
    CHECK(cudaMalloc(&d_x, sizeof(h_x)));
    CHECK(cudaMalloc(&d_v, sizeof(h_v)));
    CHECK(cudaMalloc(&d_f, sizeof(h_f)));
    CHECK(cudaMalloc(&d_m, sizeof(h_m)));
    CHECK(cudaMemcpy(d_x, h_x, sizeof(h_x), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_v, h_v, sizeof(h_v), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_m, h_m, sizeof(h_m), cudaMemcpyHostToDevice));

    double cload = 100.0 / 9.0;
    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));
    CHECK(cudaEventRecord(t0));
    cantilever_loop<<<1, 128>>>(d_x, d_v, d_f, d_m, nstep, dt, cload);
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
    double tip_dz = h_x[3 * tip + 2] - 0.0;
    printf("nstep=%ld wall=%.3f s (%.3f us/step) tip_dz=%.6e max|f|=%.3e\n", nstep,
           ms / 1000.0, ms * 1000.0 / nstep, tip_dz, fmax);
    return 0;
}
