// Persistent-kernel proxy for the hertz_sphere native explicit step (oc-opt Phase G0).
//
// Native arm: 486 nodes (485 structural + 1 driver), 304 LAW1 hex8 bricks,
// PRECOMPUTED linear stiffness per element (dgemv force, no per-step
// Jacobians), 2 rigid planes (fixed bottom z=0, top moving down at 1e-4 m/s)
// with exact-multiplier contact, velocity-Verlet, 105056 steps over 0.1 s
// (9.2 s wall, 87.6 us/step on 1 CPU thread).
//
// Proxy mirrors a real port: deck parsed at runtime (nodes/bricks/mats/
// groups/walls), linear K per element built on host, greedy element coloring
// for race-free parallel scatter, persistent device time loop, FP64.
// Contact models the multiplier as velocity+position projection on the
// geometric engagement sets (documented deviation; engagement is nearly
// static at 0.1% strain). Fixed dt = stop/native-cycles (native adaptive
// driver rides the bulk CFL; timing per step is dt-independent).
//
// Fidelity gates: N=1 exact-zero, DZ/247 ~= -1e-5 at full N (probe tol 10%),
// bit-identical repeats, linear short->long scaling.
//
// Build: nvcc -O2 -arch=native -o hertz_proxy hertz_proxy.cu
// Run:   ./hertz_proxy [nstep]   (default 105056; needs deck path below)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CHECK(call)                                                             \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,                 \
                   cudaGetErrorString(err));                                    \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define MAXN 512
#define MAXE 320
#define MAXC 16

static double h_X0[MAXN][3], h_m[MAXN];
static int h_elem[MAXE][8], h_nelem = 0, h_nnode = 0;
static int h_slave_lo[MAXN], h_slave_hi[MAXN], h_nlo = 0, h_nhi = 0;
static int h_color[MAXE], h_ncolor = 0, h_cstart[MAXC + 1], h_cperm[MAXE];
static double h_K[MAXE][24][24];
static double h_E = 2e11, h_nu = 0.3, h_rho = 7800.0;

// ---- minimal .rad reader (nodes, bricks, LAW1, groups) ----
static char* fline(char* buf, int n, FILE* f) {
    if (!fgets(buf, n, f)) return nullptr;
    return buf;
}
static void parse_deck(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) {
        printf("cannot open %s\n", path);
        exit(1);
    }
    char buf[1024];
    int mode = 0; // 1 nodes, 2 bricks, 3 mat, 4 group
    int group_id = 0, mat_line = 0;
    int ids[MAXN], nids = 0;
    while (fline(buf, sizeof buf, f)) {
        if (buf[0] == '/') {
            // flush pending group
            if (mode == 4 && nids > 0) {
                for (int i = 0; i < nids; ++i) {
                    if (group_id == 1) h_slave_lo[h_nlo++] = ids[i] - 1;
                    if (group_id == 2) h_slave_hi[h_nhi++] = ids[i] - 1;
                }
                nids = 0;
            }
            mode = 0;
            if (strncmp(buf, "/BRICK", 6) == 0) mode = 2;
            else if (strncmp(buf, "/MAT/LAW1", 9) == 0) {
                mode = 3;
                mat_line = 0;
                // title line follows; skip via mat_line counting below
            } else if (strncmp(buf, "/GRNOD", 6) == 0) {
                mode = 4;
                // id is 3rd slash field: /GRNOD/NODE/<id>/1
                const char* p = buf;
                int slash = 0;
                group_id = 0;
                while (*p) {
                    if (*p == '/') {
                        ++slash;
                        if (slash == 3) group_id = atoi(p + 1);
                    }
                    ++p;
                }
                nids = 0;
                // title line follows; mark by consuming next line
                fline(buf, sizeof buf, f);
                continue;
            } else if (strncmp(buf, "/NODE", 5) == 0) {
                mode = 1;
                continue;
            }
            // node block has no header in this deck: detect by content below
            if (mode == 0) continue;
            // MAT title line
            if (mode == 3) {
                fline(buf, sizeof buf, f);
                continue;
            }
            continue;
        }
        if (buf[0] == '#' || buf[0] == '\n') continue;
        if (mode == 2) {
            int eid, n[8];
            if (sscanf(buf, "%d %d %d %d %d %d %d %d %d", &eid, &n[0], &n[1], &n[2],
                       &n[3], &n[4], &n[5], &n[6], &n[7]) == 9) {
                for (int a = 0; a < 8; ++a) h_elem[h_nelem][a] = n[a] - 1;
                ++h_nelem;
            }
        } else if (mode == 3) {
            if (mat_line == 0) {
                h_rho = atof(buf);
            } else if (mat_line == 1) {
                sscanf(buf, "%lf %lf", &h_E, &h_nu);
                mode = 0;
            }
            ++mat_line;
        } else if (mode == 4) {
            // free list of node ids, possibly several lines
            const char* p = buf;
            while (*p) {
                while (*p == ' ' || *p == '\t') ++p;
                if (*p == '\n' || *p == 0) break;
                ids[nids++] = atoi(p);
                while (*p && *p != ' ' && *p != '\t' && *p != '\n') ++p;
            }
        } else {
            // node line? id + 3 floats (deck has no /NODE header)
            int id;
            double x, y, z;
            if (sscanf(buf, "%d %lf %lf %lf", &id, &x, &y, &z) == 4 && id >= 1 &&
                id <= MAXN) {
                h_X0[id - 1][0] = x;
                h_X0[id - 1][1] = y;
                h_X0[id - 1][2] = z;
                if (id > h_nnode) h_nnode = id;
            }
        }
    }
    if (mode == 4 && nids > 0) {
        for (int i = 0; i < nids; ++i) {
            if (group_id == 1) h_slave_lo[h_nlo++] = ids[i] - 1;
            if (group_id == 2) h_slave_hi[h_nhi++] = ids[i] - 1;
        }
    }
    fclose(f);
    printf("deck: nodes=%d elems=%d E=%.3e nu=%.3f rho=%.1f slaves lo=%d hi=%d\n",
           h_nnode, h_nelem, h_E, h_nu, h_rho, h_nlo, h_nhi);
}

// ---- reference geometry: lumped masses + linear hex K per element ----
static double det3(double A[3][3]) {
    return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
           A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
           A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
}
static void build_model() {
    const double g = 0.5773502691896258;
    double gp[2] = {-g, g};
    // node ordering in deck: need sa signs per local node; deck order is
    // arbitrary, so derive dN from the reference-cube map via the same
    // convention for all elements (topology-consistent: scatter/gather use
    // the same local ordering, so K is self-consistent).
    // sa[a][j] for standard hex8 local ordering.
    int sa[8][3] = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
                    {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1}};
    for (int n = 0; n < h_nnode; ++n) h_m[n] = 0.0;
    for (int e = 0; e < h_nelem; ++e) {
        double Xe[8][3];
        for (int a = 0; a < 8; ++a)
            for (int i = 0; i < 3; ++i) Xe[a][i] = h_X0[h_elem[e][a]][i];
        double Ke[24][24] = {{0}};
        double Ve = 0.0;
        for (int qx = 0; qx < 2; ++qx)
            for (int qy = 0; qy < 2; ++qy)
                for (int qz = 0; qz < 2; ++qz) {
                    double xi[3] = {gp[qx], gp[qy], gp[qz]};
                    double dN[8][3], J[3][3] = {{0}};
                    for (int a = 0; a < 8; ++a) {
                        double f1 = 1 + sa[a][0] * xi[0], f2 = 1 + sa[a][1] * xi[1],
                               f3 = 1 + sa[a][2] * xi[2];
                        dN[a][0] = 0.125 * sa[a][0] * f2 * f3;
                        dN[a][1] = 0.125 * sa[a][1] * f1 * f3;
                        dN[a][2] = 0.125 * sa[a][2] * f1 * f2;
                        for (int i = 0; i < 3; ++i)
                            for (int j = 0; j < 3; ++j) J[i][j] += Xe[a][i] * dN[a][j];
                    }
                    double detJ = det3(J);
                    if (detJ <= 0) detJ = -detJ; // orientation-agnostic volume
                    Ve += detJ; // weights are 1
                    // invJ
                    double invJ[3][3];
                    invJ[0][0] = (J[1][1] * J[2][2] - J[1][2] * J[2][1]) / detJ;
                    invJ[0][1] = (J[0][2] * J[2][1] - J[0][1] * J[2][2]) / detJ;
                    invJ[0][2] = (J[0][1] * J[1][2] - J[0][2] * J[1][1]) / detJ;
                    invJ[1][0] = (J[1][2] * J[2][0] - J[1][0] * J[2][2]) / detJ;
                    invJ[1][1] = (J[0][0] * J[2][2] - J[0][2] * J[2][0]) / detJ;
                    invJ[1][2] = (J[0][2] * J[1][0] - J[0][0] * J[1][2]) / detJ;
                    invJ[2][0] = (J[1][0] * J[2][1] - J[1][1] * J[2][0]) / detJ;
                    invJ[2][1] = (J[0][1] * J[2][0] - J[0][0] * J[2][1]) / detJ;
                    invJ[2][2] = (J[0][0] * J[1][1] - J[0][1] * J[1][0]) / detJ;
                    double dX[8][3];
                    for (int a = 0; a < 8; ++a)
                        for (int i = 0; i < 3; ++i)
                            dX[a][i] = dN[a][0] * invJ[0][i] + dN[a][1] * invJ[1][i] +
                                       dN[a][2] * invJ[2][i];
                    // B (6x24) implicit in Ke += w*detJ * B^T C B
                    double lam = h_E * h_nu / ((1 + h_nu) * (1 - 2 * h_nu));
                    double mu = h_E / (2 * (1 + h_nu));
                    // C*eps contraction done per (a,i),(b,j) entry:
                    for (int a = 0; a < 8; ++a)
                        for (int i = 0; i < 3; ++i) {
                            // strain mode of unit disp (a,i): eps_kl
                            double ea[3][3] = {{0}};
                            ea[0][0] = (i == 0) ? dX[a][0] : 0;
                            ea[1][1] = (i == 1) ? dX[a][1] : 0;
                            ea[2][2] = (i == 2) ? dX[a][2] : 0;
                            double s01 = ((i == 0) ? dX[a][1] : 0) + ((i == 1) ? dX[a][0] : 0);
                            double s02 = ((i == 0) ? dX[a][2] : 0) + ((i == 2) ? dX[a][0] : 0);
                            double s12 = ((i == 1) ? dX[a][2] : 0) + ((i == 2) ? dX[a][1] : 0);
                            ea[0][1] = ea[1][0] = 0.5 * s01;
                            ea[0][2] = ea[2][0] = 0.5 * s02;
                            ea[1][2] = ea[2][1] = 0.5 * s12;
                            double tr = ea[0][0] + ea[1][1] + ea[2][2];
                            double sg[3][3];
                            for (int k = 0; k < 3; ++k)
                                for (int l = 0; l < 3; ++l)
                                    sg[k][l] = lam * tr * (k == l) + 2 * mu * ea[k][l];
                            for (int b = 0; b < 8; ++b)
                                for (int j = 0; j < 3; ++j) {
                                    double eb[3][3] = {{0}};
                                    eb[0][0] = (j == 0) ? dX[b][0] : 0;
                                    eb[1][1] = (j == 1) ? dX[b][1] : 0;
                                    eb[2][2] = (j == 2) ? dX[b][2] : 0;
                                    double t01 = ((j == 0) ? dX[b][1] : 0) + ((j == 1) ? dX[b][0] : 0);
                                    double t02 = ((j == 0) ? dX[b][2] : 0) + ((j == 2) ? dX[b][0] : 0);
                                    double t12 = ((j == 1) ? dX[b][2] : 0) + ((j == 2) ? dX[b][1] : 0);
                                    eb[0][1] = eb[1][0] = 0.5 * t01;
                                    eb[0][2] = eb[2][0] = 0.5 * t02;
                                    eb[1][2] = eb[2][1] = 0.5 * t12;
                                    double e = 0;
                                    for (int k = 0; k < 3; ++k)
                                        for (int l = 0; l < 3; ++l) e += sg[k][l] * eb[k][l];
                                    Ke[8 * i + a][8 * j + b] += e * detJ;
                                }
                        }
                }
        for (int a = 0; a < 8; ++a) h_m[h_elem[e][a]] += h_rho * Ve / 8.0;
        for (int i = 0; i < 24; ++i)
            for (int j = 0; j < 24; ++j) h_K[e][i][j] = Ke[i][j];
    }
    // greedy coloring over node-sharing adjacency
    for (int e = 0; e < h_nelem; ++e) h_color[e] = -1;
    h_ncolor = 0;
    for (int e = 0; e < h_nelem; ++e) {
        bool used[MAXC] = {false};
        for (int o = 0; o < e; ++o) {
            bool share = false;
            for (int a = 0; a < 8 && !share; ++a)
                for (int b = 0; b < 8; ++b)
                    if (h_elem[e][a] == h_elem[o][b]) {
                        share = true;
                        break;
                    }
            if (share && h_color[o] >= 0) used[h_color[o]] = true;
        }
        int c = 0;
        while (c < MAXC && used[c]) ++c;
        if (c >= MAXC) {
            printf("coloring overflow\n");
            exit(1);
        }
        h_color[e] = c;
        if (c + 1 > h_ncolor) h_ncolor = c + 1;
    }
    h_cstart[0] = 0;
    for (int c = 0; c < h_ncolor; ++c) {
        int n = 0;
        for (int e = 0; e < h_nelem; ++e)
            if (h_color[e] == c) h_cperm[h_cstart[c] + n++] = e;
        h_cstart[c + 1] = h_cstart[c] + n;
    }
    printf("colors=%d masses: m[0]=%.3e\n", h_ncolor, h_m[0]);
}

// ---- device ----
__constant__ int c_elem[MAXE][8]; // 10 KB, fits; K (1.4 MB) is global

__global__ void __launch_bounds__(256)
hertz_loop(double* x, double* v, double* f, const double* m, const double* X0,
           const double* K, const int* cperm, const int* cstart, int ncolor,
           const int* slo, int nlo, const int* shi, int nhi, long nstep,
           double dt, double vtop, double ztop0, int nnode) {
    int tid = threadIdx.x;
    for (long s = 0; s < nstep; ++s) {
        double t = s * dt;
        double ztop = ztop0 + vtop * t;
        for (int n = tid; n < nnode; n += blockDim.x) {
            f[3 * n] = 0.0;
            f[3 * n + 1] = 0.0;
            f[3 * n + 2] = 0.0;
        }
        __syncthreads();
        for (int c = 0; c < ncolor; ++c) {
#ifdef EMPTY_FORCE
            if (false) {
#else
            int beg = cstart[c], end = cstart[c + 1];
            for (int k = beg + tid; k < end; k += blockDim.x) {
#endif
#ifndef EMPTY_FORCE
                int e = cperm[k];
                double ue[24];
                for (int a = 0; a < 8; ++a) {
                    int n = c_elem[e][a];
                    ue[a] = x[3 * n] - X0[3 * n];
                    ue[8 + a] = x[3 * n + 1] - X0[3 * n + 1];
                    ue[16 + a] = x[3 * n + 2] - X0[3 * n + 2];
                }
                double fe[24] = {0};
                for (int i = 0; i < 24; ++i) {
                    double ssum = 0;
                    for (int j = 0; j < 24; ++j)
                        ssum += K[((size_t)e * 24 + i) * 24 + j] * ue[j];
                    fe[i] = ssum;
                }
                for (int a = 0; a < 8; ++a) {
                    int n = c_elem[e][a];
                    f[3 * n] -= fe[a];
                    f[3 * n + 1] -= fe[8 + a];
                    f[3 * n + 2] -= fe[16 + a];
                }
#endif
            }
            __syncthreads();
        }
        for (int n = tid; n < nnode; n += blockDim.x) {
            double invm = 1.0 / m[n];
            v[3 * n] += dt * f[3 * n] * invm;
            v[3 * n + 1] += dt * f[3 * n + 1] * invm;
            v[3 * n + 2] += dt * f[3 * n + 2] * invm;
            x[3 * n] += dt * v[3 * n];
            x[3 * n + 1] += dt * v[3 * n + 1];
            x[3 * n + 2] += dt * v[3 * n + 2];
        }
        __syncthreads();
        // walls: bottom z>=0 fixed; top z<=ztop(t) moving at vtop
        for (int k = tid; k < nlo; k += blockDim.x) {
            int n = slo[k];
            if (x[3 * n + 2] < 0.0) {
                x[3 * n + 2] = 0.0;
                if (v[3 * n + 2] < 0.0) v[3 * n + 2] = 0.0;
            }
        }
        for (int k = tid; k < nhi; k += blockDim.x) {
            int n = shi[k];
            if (x[3 * n + 2] > ztop) {
                x[3 * n + 2] = ztop;
                if (v[3 * n + 2] > vtop) v[3 * n + 2] = vtop;
            }
        }
        __syncthreads();
    }
}

int main(int argc, char** argv) {
    long nstep = argc > 1 ? atol(argv[1]) : 105056L;
    const char* deck = argc > 2 ? argv[2]
        : "/home/pierre/src/all_of_sonium/sonium/crates/sonium-qa/decks/"
          "hertz_sphere/hertz_sphere_0000.rad";
    parse_deck(deck);
    int nn = h_nnode - 1; // exclude driver node 486 (1-based) from dynamics
    build_model();
    size_t n3 = (size_t)nn * 3 * sizeof(double);
    double *d_x, *d_v, *d_f, *d_m, *d_X0, *d_K;
    int *d_perm, *d_cstart, *d_slo, *d_shi;
    CHECK(cudaMalloc(&d_K, sizeof(h_K)));
    CHECK(cudaMemcpy(d_K, h_K, sizeof(h_K), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&d_x, n3));
    CHECK(cudaMalloc(&d_v, n3));
    CHECK(cudaMalloc(&d_f, n3));
    CHECK(cudaMalloc(&d_m, (size_t)nn * sizeof(double)));
    CHECK(cudaMalloc(&d_X0, n3));
    CHECK(cudaMalloc(&d_perm, sizeof(h_cperm)));
    CHECK(cudaMalloc(&d_cstart, sizeof(h_cstart)));
    CHECK(cudaMalloc(&d_slo, sizeof(h_slave_lo)));
    CHECK(cudaMalloc(&d_shi, sizeof(h_slave_hi)));
    CHECK(cudaMemcpy(d_x, h_X0, n3, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_X0, h_X0, n3, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_v, 0, n3));
    CHECK(cudaMemcpy(d_m, h_m, (size_t)nn * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_perm, h_cperm, sizeof(h_cperm), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_cstart, h_cstart, sizeof(h_cstart), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_slo, h_slave_lo, sizeof(h_slave_lo), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_shi, h_slave_hi, sizeof(h_slave_hi), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(c_elem, h_elem, sizeof(h_elem)));
    double dt = 0.1 / 105056.0;
    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));
    CHECK(cudaEventRecord(t0));
    hertz_loop<<<1, 256>>>(d_x, d_v, d_f, d_m, d_X0, d_K, d_perm, d_cstart,
                           h_ncolor, d_slo, h_nlo, d_shi, h_nhi, nstep, dt, -1e-4,
                           0.1, nn);
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    CHECK(cudaGetLastError());
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, t0, t1));
    static double h_x[MAXN * 3], h_flast[MAXN * 3];
    CHECK(cudaMemcpy(h_x, d_x, n3, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_flast, d_f, n3, cudaMemcpyDeviceToHost));
    double fmax = 0.0;
    for (int i = 0; i < nn * 3; ++i) {
        double a = h_flast[i] < 0 ? -h_flast[i] : h_flast[i];
        if (a == a && a > fmax) fmax = a;
    }
    printf("nstep=%ld wall=%.3f s (%.3f us/step) DZ247=%.6e max|f|=%.3e\n", nstep,
           ms / 1000.0, ms * 1000.0 / nstep, h_x[3 * 246 + 2] - h_X0[246][2], fmax);
    return 0;
}

