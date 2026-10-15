// Persistent-kernel GMRES(30)+ILU(0) proxy for the ns_lid_cavity native
// steady-Picard linear solve (oc-opt Phase G0).
//
// Question: can a CUDA port of the cavity's per-Picard linear solve (12,675
// P1-P1 dofs, GMRES(30)+ILU(0) to 1e-8, ~0.67 ms/iteration native CPU) beat
// the 15.8 s native 1-thread run, and at what distance from roofline?
//
// Design: the host builds the EXACT first-Picard (Oseen-about-BC-state)
// system analytically — closed-form P1 stencils on the 65x65 structured
// triangulation plus the native 3-point quadrature for the u-dependent
// terms. Native Picard-1 linearizes about the BC-imposed iterate (lid
// nodes at (1,0), interior 0), so convection, SUPG, LSIC and the
// SUPG/PSPG coupling blocks are live in lid-band elements (with per-QP
// tau), on top of the [mu*K, B; B^T, -PSPG] Stokes base, with symmetric
// Dirichlet elimination (lid + walls, corners with the lid) and the
// center pressure pin, all mirrored from the native assembly including
// sign conventions and structural zeros (entry-verified against a native
// MatrixMarket dump). The host then computes ILU(0) (same pivot rule)
// plus triangular level schedules once (documented optimistic-for-GPU:
// native refactors per Picard on CPU). ONE cooperative persistent kernel
// runs the GMRES(30) loop on-device: SpMV + level-wavefront ILU pair +
// batched CGS with DGKS reorthogonalisation + Givens, grid.sync between
// phases (what a real port would do).
//
// Known deviations (all documented in the report): analytic stencils vs
// quadrature assembly (1e-16-level entry differences, verified by nnz and
// residual match); host-side ILU setup; no Picard outer loop (the proxy
// times one linear solve; the full-deck projection multiplies by the
// native 12,240 total GMRES iterations); FP64 throughout like native.
//
// FLOP estimate per GMRES iteration: SpMV 2*nnz + ILU pair
// 2*(nnz_L+nnz_U+n) + CGS ~(8*restart+12)*n + Givens O(restart^2); the
// roofline numerator uses these closed-form counts, not ncu.
//
// Build: nvcc -O2 -arch=native -o cavity_proxy ns_lid_cavity_cuda_proxy.cu
// Run:   ./cavity_proxy [nstep]        (default 12240, the native total)
// ncu:   ncu --section SpeedOfLight ./cavity_proxy 50 (root-blocked here)

#include <algorithm>
#include <cmath>
#include <cooperative_groups.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

namespace cg = cooperative_groups;

#define CHECK(call)                                                             \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,                 \
                   cudaGetErrorString(err));                                    \
            return 1;                                                           \
        }                                                                       \
    } while (0)

// Exact cavity discretization: 64x64 cells, 65x65 P1 nodes, node-major
// velocity (ux,uy per node), node-ordered pressure.
#define NM 64
#define NP1 65
#define NNODES (NP1 * NP1) // 4225
#define NV (2 * NNODES)    // 8450
#define NPG NNODES         // 4225
#define N (NV + NPG)       // 12675
#define RESTART 30
#define NBASIS (RESTART + 1)

#define NODE(i, j) ((j) * NP1 + (i))
#define UX(a) (2 * (a))
#define UY(a) (2 * (a) + 1)
#define PR(a) (NV + (a))

static const double H = 1.0 / NM; // 1/64, exact in binary
static const double MU = 0.01;    // 1/Re, Re = 100
static const double RHO = 1.0;
static const double NU = 0.01;

struct Coo {
    int row, col;
    double val;
};

// Stabilisation tau: (4u^2/h^2 + 36nu^2/h^4)^-1/2 with h =
// sqrt(2*det_j), det_j = H^2. Same formula and op order as native
// (verified: u = 0 gives 8.138020833333e-3 exactly).
static double stab_tau(double u_mag) {
    double h = sqrt(2.0 * H * H);
    double inv_tau_sq =
        4.0 * u_mag * u_mag / (h * h) + 36.0 * NU * NU / (h * h * h * h);
    return 1.0 / sqrt(inv_tau_sq);
}

// Linearized-about state: native Picard-1 sees the BC-imposed iterate
// (lid nodes at (1,0), everything else 0), NOT zeros. Nodal x-velocity.
static double uk_nodal(int node) {
    int j = node / NP1;
    return (j == NM) ? 1.0 : 0.0;
}

// Assemble the first-Picard (Oseen-about-BC-state) COO list. Viscous
// and Galerkin B/B^T use closed forms (entry-verified against native);
// convection, SUPG, LSIC, SUPG-pressure, PSPG-convection and PSPG run the
// native 3-point quadrature per element against the BC-imposed iterate
// (lid (1,0), else 0), with per-QP tau. T1 = [n00,n10,n11],
// T2 = [n00,n11,n01] per cell; integer gradients times 1/H. Mirrors
// native triplet pushes (including structural zeros) and mono signs
// (convection/SUPG/LSIC/SUPG-p added, PSPG-convection and PSPG negated).
static void assemble_picard1_coo(std::vector<Coo> &coo) {
    int di[2][3] = {{0, 1, 1}, {0, 1, 0}};
    int dj[2][3] = {{0, 0, 1}, {0, 1, 1}};
    double gx[2][3] = {{-1, 1, 0}, {0, 1, -1}};
    double gy[2][3] = {{0, -1, 1}, {-1, 0, 1}};
    double KK[2][3][3] = {
        {{0.5, -0.5, 0.0}, {-0.5, 1.0, -0.5}, {0.0, -0.5, 0.5}},
        {{0.5, 0.0, -0.5}, {0.0, 0.5, -0.5}, {-0.5, -0.5, 1.0}},
    };
    // Native triangle order-2 rule: ref points, P1 values, weight 1/6.
    double qxi[3] = {1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0};
    double qeta[3] = {1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0};
    double qw = 1.0 / 6.0;
    double det = H * H;
    double h_stab = sqrt(2.0 * H * H);
    for (int j = 0; j < NM; ++j) {
        for (int i = 0; i < NM; ++i) {
            for (int t = 0; t < 2; ++t) {
                int g[3];
                for (int a = 0; a < 3; ++a) g[a] = NODE(i + di[t][a], j + dj[t][a]);
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        // Viscous mu*K per component (block diagonal).
                        coo.push_back({UX(g[a]), UX(g[b]), MU * KK[t][a][b]});
                        coo.push_back({UY(g[a]), UY(g[b]), MU * KK[t][a][b]});
                        // B10 (pressure row): -|T|*grad/3 = -H*g/6.
                        coo.push_back({PR(g[a]), UX(g[b]), -H * gx[t][b] / 6.0});
                        coo.push_back({PR(g[a]), UY(g[b]), -H * gy[t][b] / 6.0});
                        // B01 (velocity row, transpose values).
                        coo.push_back({UX(g[b]), PR(g[a]), -H * gx[t][b] / 6.0});
                        coo.push_back({UY(g[b]), PR(g[a]), -H * gy[t][b] / 6.0});
                    }
                }
                // U-dependent locals via quadrature (nonzero only where the
                // BC-imposed iterate is nonzero: lid-band elements).
                double ux[3] = {uk_nodal(g[0]), uk_nodal(g[1]), uk_nodal(g[2])};
                double C[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
                double S[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
                double L[6][6] = {{0}};
                double SP[6][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0},
                                   {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
                double PC[3][6] = {{0}};
                double P[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
                for (int q = 0; q < 3; ++q) {
                    double phi[3] = {1.0 - qxi[q] - qeta[q], qxi[q], qeta[q]};
                    double u_q = phi[0] * ux[0] + phi[1] * ux[1] + phi[2] * ux[2];
                    double umag = sqrt(u_q * u_q);
                    double tau_q = stab_tau(umag);
                    double wdet = qw * det;
                    double tls = tau_q * umag * h_stab * 0.5;
                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            double adv_b = u_q * gx[t][b] / H; // u·∇φ_b (uy=0)
                            C[a][b] += RHO * phi[a] * adv_b * wdet;
                            double adv_a = u_q * gx[t][a] / H;
                            S[a][b] += tau_q * adv_a * adv_b * wdet * RHO;
                            P[a][b] += (tau_q / RHO) *
                                       (gx[t][a] * gx[t][b] + gy[t][a] * gy[t][b]) /
                                       (H * H) * wdet;
                        }
                        for (int ci = 0; ci < 2; ++ci) {
                            double ga = (ci == 0) ? gx[t][a] : gy[t][a];
                            double adv_a = u_q * gx[t][a] / H;
                            for (int b = 0; b < 3; ++b) {
                                double gb = (ci == 0) ? gx[t][b] : gy[t][b];
                                SP[2 * a + ci][b] += tau_q * adv_a * (gb / H) * wdet;
                                for (int cj = 0; cj < 2; ++cj) {
                                    double gbj =
                                        (cj == 0) ? gx[t][b] : gy[t][b];
                                    L[2 * a + ci][2 * b + cj] +=
                                        RHO * tls * (ga / H) * (gbj / H) * wdet;
                                }
                            }
                        }
                    }
                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            double adv_b = u_q * gx[t][b] / H;
                            for (int cj = 0; cj < 2; ++cj) {
                                double ga = (cj == 0) ? gx[t][a] : gy[t][a];
                                PC[a][2 * b + cj] +=
                                    tau_q * (ga / H) * adv_b * wdet;
                            }
                        }
                    }
                }
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        coo.push_back({UX(g[a]), UX(g[b]), C[a][b]});
                        coo.push_back({UY(g[a]), UY(g[b]), C[a][b]});
                        coo.push_back({UX(g[a]), UX(g[b]), S[a][b]});
                        coo.push_back({UY(g[a]), UY(g[b]), S[a][b]});
                        coo.push_back({PR(g[a]), PR(g[b]), -P[a][b]});
                        for (int ci = 0; ci < 2; ++ci)
                            for (int cj = 0; cj < 2; ++cj)
                                coo.push_back({2 * g[a] + ci, 2 * g[b] + cj,
                                               L[2 * a + ci][2 * b + cj]});
                        for (int ci = 0; ci < 2; ++ci) {
                            coo.push_back(
                                {2 * g[a] + ci, PR(g[b]), SP[2 * a + ci][b]});
                            coo.push_back(
                                {PR(g[a]), 2 * g[b] + ci, -PC[a][2 * b + ci]});
                        }
                    }
                }
            }
        }
    }
}

// Sort by (row, col), merge duplicates by summation, keep explicit zeros:
// the native to_csr policy.
static void coo_to_csr(const std::vector<Coo> &coo, std::vector<int> &row_ptr,
                       std::vector<int> &cols, std::vector<double> &vals) {
    std::vector<Coo> sorted = coo;
    std::sort(sorted.begin(), sorted.end(), [](const Coo &a, const Coo &b) {
        if (a.row != b.row) return a.row < b.row;
        return a.col < b.col;
    });
    row_ptr.assign(N + 1, 0);
    cols.clear();
    vals.clear();
    cols.reserve(sorted.size());
    vals.reserve(sorted.size());
    int cur_row = -1;
    int cur_col = -1;
    double acc = 0.0;
    bool open = false;
    auto flush = [&]() {
        if (open) {
            cols.push_back(cur_col);
            vals.push_back(acc);
            row_ptr[cur_row + 1]++;
            open = false;
        }
    };
    for (const auto &e : sorted) {
        if ((int)e.row != cur_row) {
            flush();
            cur_row = e.row;
        }
        if (open && e.col == cur_col) {
            acc += e.val;
        } else {
            flush();
            cur_col = e.col;
            acc = e.val;
            open = true;
        }
    }
    flush();
    for (int i = 0; i < N; ++i) row_ptr[i + 1] += row_ptr[i];
}

// Symmetric Dirichlet elimination, mirroring apply_dirichlet_to_csr: BC
// rows become identity, free rows lift BC columns and zero them in place
// (pattern kept, including the resulting zeros).
static double apply_dirichlet(std::vector<int> &row_ptr, std::vector<int> &cols,
                              std::vector<double> &vals, std::vector<double> &rhs) {
    std::vector<char> is_bc(N, 0);
    std::vector<double> bc(N, 0.0);
    for (int j = 0; j < NP1; ++j) {
        for (int i = 0; i < NP1; ++i) {
            if (i == 0 || i == NM || j == 0 || j == NM) {
                int a = NODE(i, j);
                is_bc[UX(a)] = 1;
                is_bc[UY(a)] = 1;
                bc[UX(a)] = (j == NM) ? 1.0 : 0.0; // lid, corners move along
                bc[UY(a)] = 0.0;
            }
        }
    }
    int pin = PR(NODE(NM / 2, NM / 2)); // center pressure gauge
    is_bc[pin] = 1;
    bc[pin] = 0.0;
    rhs.assign(N, 0.0);
    for (int row = 0; row < N; ++row) {
        if (is_bc[row]) {
            for (int k = row_ptr[row]; k < row_ptr[row + 1]; ++k)
                vals[k] = (cols[k] == row) ? 1.0 : 0.0;
            rhs[row] = bc[row];
        } else {
            for (int k = row_ptr[row]; k < row_ptr[row + 1]; ++k) {
                int c = cols[k];
                if (is_bc[c]) {
                    rhs[row] -= vals[k] * bc[c];
                    vals[k] = 0.0;
                }
            }
        }
    }
    double norm = 0.0;
    for (int i = 0; i < N; ++i) norm += rhs[i] * rhs[i];
    return sqrt(norm);
}

// ILU(0) factorization mirroring the native row-wise algorithm: sorted
// rows, elimination over k < i in order, pivot rule (skip + zero the
// multiplier when |piv| < 1e-2 * max(||row_k||, 1e-30)), updates over
// j > k in pattern order. Splits into strict-lower L (unit diagonal
// implicit), strict-upper U and the diagonal D.
static int ilu0_factorize(const std::vector<int> &row_ptr,
                          const std::vector<int> &cols,
                          const std::vector<double> &vals,
                          std::vector<int> &l_ptr, std::vector<int> &l_cols,
                          std::vector<double> &l_vals, std::vector<int> &u_ptr,
                          std::vector<int> &u_cols, std::vector<double> &u_vals,
                          std::vector<double> &diag) {
    std::vector<double> v = vals;
    std::vector<int> diag_idx(N, -1);
    for (int i = 0; i < N; ++i)
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k)
            if (cols[k] == i) {
                diag_idx[i] = k;
                break;
            }
    std::vector<double> row_norm(N, 0.0);
    for (int i = 0; i < N; ++i) {
        double s = 0.0;
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) s += v[k] * v[k];
        row_norm[i] = sqrt(s);
    }
    int skipped = 0;
    std::vector<int> pos(N, -1); // column -> index scratch per row k
    for (int i = 0; i < N; ++i) {
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
            int c = cols[k];
            if (c >= i) break;
            int ukk = diag_idx[c];
            if (ukk < 0) {
                v[k] = 0.0;
                skipped++;
                continue;
            }
            double piv = v[ukk];
            double tol = 1e-2 * fmax(row_norm[c], 1e-30);
            if (fabs(piv) < tol) {
                v[k] = 0.0;
                skipped++;
                continue;
            }
            double lik = v[k] / piv;
            v[k] = lik;
            for (int q = row_ptr[c]; q < row_ptr[c + 1]; ++q) pos[cols[q]] = q;
            for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
                int d = cols[j];
                if (d <= c) continue;
                int q = pos[d];
                if (q >= 0) v[j] -= lik * v[q];
            }
            for (int q = row_ptr[c]; q < row_ptr[c + 1]; ++q) pos[cols[q]] = -1;
        }
    }
    l_ptr.assign(N + 1, 0);
    u_ptr.assign(N + 1, 0);
    l_cols.clear();
    l_vals.clear();
    u_cols.clear();
    u_vals.clear();
    diag.assign(N, 1.0);
    for (int i = 0; i < N; ++i) {
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
            int c = cols[k];
            if (c < i) {
                l_cols.push_back(c);
                l_vals.push_back(v[k]);
            } else if (c == i) {
                diag[i] = v[k];
            } else {
                u_cols.push_back(c);
                u_vals.push_back(v[k]);
            }
        }
        l_ptr[i + 1] = (int)l_cols.size();
        u_ptr[i + 1] = (int)u_cols.size();
    }
    return skipped;
}

// Level schedule over a triangular pattern: rows in a level share no
// dependency. Forward levels sweep low->high, backward high->low.
static void level_schedule(const std::vector<int> &ptr,
                           const std::vector<int> &cols, bool forward,
                           std::vector<int> &lvl_ptr, std::vector<int> &lvl_rows) {
    std::vector<int> lvl(N, 0);
    if (forward) {
        for (int i = 0; i < N; ++i) {
            int m = 0;
            for (int k = ptr[i]; k < ptr[i + 1]; ++k) {
                int c = cols[k];
                if (c < i && lvl[c] + 1 > m) m = lvl[c] + 1;
            }
            lvl[i] = m;
        }
    } else {
        for (int i = N - 1; i >= 0; --i) {
            int m = 0;
            for (int k = ptr[i]; k < ptr[i + 1]; ++k) {
                int c = cols[k];
                if (c > i && lvl[c] + 1 > m) m = lvl[c] + 1;
            }
            lvl[i] = m;
        }
    }
    int nlvl = 0;
    for (int i = 0; i < N; ++i) nlvl = (lvl[i] + 1 > nlvl) ? lvl[i] + 1 : nlvl;
    lvl_ptr.assign(nlvl + 1, 0);
    for (int i = 0; i < N; ++i) lvl_ptr[lvl[i] + 1]++;
    for (int l = 0; l < nlvl; ++l) lvl_ptr[l + 1] += lvl_ptr[l];
    lvl_rows.assign(N, 0);
    std::vector<int> cur = lvl_ptr;
    for (int i = 0; i < N; ++i) lvl_rows[cur[lvl[i]]++] = i;
}

// Host CSR SpMV and triangular solves for the N=1 replica check.
static void host_spmv(const std::vector<int> &ptr, const std::vector<int> &cols,
                      const std::vector<double> &vals, const double *x, double *y) {
    for (int i = 0; i < N; ++i) {
        double s = 0.0;
        for (int k = ptr[i]; k < ptr[i + 1]; ++k) s += vals[k] * x[cols[k]];
        y[i] = s;
    }
}

static void host_tri_fwd(const std::vector<int> &ptr, const std::vector<int> &cols,
                         const std::vector<double> &vals, const double *r,
                         double *y) {
    for (int i = 0; i < N; ++i) {
        double s = r[i];
        for (int k = ptr[i]; k < ptr[i + 1]; ++k) s -= vals[k] * y[cols[k]];
        y[i] = s; // unit diagonal
    }
}

static void host_tri_bwd(const std::vector<int> &ptr, const std::vector<int> &cols,
                         const std::vector<double> &vals, const double *diag,
                         const double *y, double *x) {
    for (int i = N - 1; i >= 0; --i) {
        double s = y[i];
        for (int k = ptr[i]; k < ptr[i + 1]; ++k) s -= vals[k] * x[cols[k]];
        x[i] = s / diag[i];
    }
}

static double host_dot(const double *a, const double *b) {
    double s = 0.0;
    for (int i = 0; i < N; ++i) s += a[i] * b[i];
    return s;
}

// ---- device: cooperative persistent GMRES(30) ----

#define TB 128 // threads per block

// Block reduction over row sh_row of the shared tile; all threads in the
// block must call. Returns the sum to every thread.
__device__ double block_reduce(double v, double *sh_tile, int sh_row) {
    int tid = threadIdx.x;
    double *row = sh_tile + (size_t)sh_row * TB;
    row[tid] = v;
    __syncthreads();
    for (int s = TB / 2; s > 0; s >>= 1) {
        if (tid < s) row[tid] += row[tid + s];
        __syncthreads();
    }
    return row[0];
}

// Grid-wide dot(a, b) -> out[0]. Two grid syncs.
__device__ void grid_dot(const double *a, const double *b, double *partials,
                         double *out, cg::grid_group &grid, double *sh_tile) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    double s = 0.0;
    for (int i = tidx; i < N; i += stride) s += a[i] * b[i];
    s = block_reduce(s, sh_tile, 0);
    if (threadIdx.x == 0) partials[blockIdx.x] = s;
    grid.sync();
    if (blockIdx.x == 0) {
        double t = 0.0;
        for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) t += partials[i];
        t = block_reduce(t, sh_tile, 0);
        if (threadIdx.x == 0) out[0] = t;
    }
    grid.sync();
}

__device__ void grid_nrm2(const double *a, double *partials, double *out,
                          cg::grid_group &grid, double *sh_tile) {
    grid_dot(a, a, partials, out, grid, sh_tile);
    if (blockIdx.x == 0 && threadIdx.x == 0) out[0] = sqrt(out[0]);
    grid.sync();
}

// Batched dots coeffs[j] = <w, V_j> for j = 0..k in ONE pass over the
// rows (mirrors the native batched CGS). Two grid syncs.
__device__ void batched_dots(const double *V, const double *w, int k,
                             double *coeffs, double *partials,
                             cg::grid_group &grid, double *sh_tile) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int tid = threadIdx.x;
    double c[RESTART];
    for (int j = 0; j <= k; ++j) c[j] = 0.0;
    for (int i = tidx; i < N; i += stride) {
        double wi = w[i];
        for (int j = 0; j <= k; ++j) c[j] += wi * V[(size_t)j * N + i];
    }
    for (int j = 0; j <= k; ++j) sh_tile[(size_t)j * TB + tid] = c[j];
    __syncthreads();
    for (int s = TB / 2; s > 0; s >>= 1) {
        if (tid < s)
            for (int j = 0; j <= k; ++j)
                sh_tile[(size_t)j * TB + tid] += sh_tile[(size_t)j * TB + tid + s];
        __syncthreads();
    }
    if (tid == 0)
        for (int j = 0; j <= k; ++j)
            partials[(size_t)blockIdx.x * RESTART + j] =
                sh_tile[(size_t)j * TB];
    grid.sync();
    if (blockIdx.x == 0) {
        for (int j = 0; j <= k; ++j) {
            double t = 0.0;
            for (int i = tid; i < gridDim.x; i += TB)
                t += partials[(size_t)i * RESTART + j];
            // Reduce across this block via row j.
            double *row = sh_tile + (size_t)j * TB;
            row[tid] = t;
            __syncthreads();
            for (int s = TB / 2; s > 0; s >>= 1) {
                if (tid < s) row[tid] += row[tid + s];
                __syncthreads();
            }
            if (tid == 0) coeffs[j] = row[0];
            __syncthreads();
        }
    }
    grid.sync();
}

__global__ void __launch_bounds__(TB)
    cavity_loop(const int *A_ptr, const int *A_col, const double *A_val,
                const int *L_ptr, const int *L_col, const double *L_val,
                const int *U_ptr, const int *U_col, const double *U_val,
                const double *D, const int *Llvl_ptr, const int *Llvl_rows,
                int L_nlvl, const int *Ulvl_ptr, const int *Ulvl_rows, int U_nlvl,
                const double *b, double *x, double *V, double *w, double *op,
                double *H, double *cs, double *sn, double *g, double *tmp,
                double *res_hist, int *meta, double *scal, double *partials,
                long nstep) {
    // meta: [tol_hit, reorths, breakdown]. scal: [beta, pbnorm, tbnorm].
    cg::grid_group grid = cg::this_grid();
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    __shared__ double sh[32][TB];

    // INIT: x = 0, op = b, w = M^-1 op, beta/v0/g.
    for (int i = tidx; i < N; i += stride) x[i] = 0.0;
    grid.sync();
    for (int i = tidx; i < N; i += stride) op[i] = b[i];
    grid.sync();
    grid_nrm2(b, partials, scal + 2, grid, &sh[0][0]); // tbnorm
#ifndef EMPTY_FORCE
    for (int l = 0; l < L_nlvl; ++l) { // forward: unit diagonal
        for (int q = Llvl_ptr[l] + tidx; q < Llvl_ptr[l + 1]; q += stride) {
            int i = Llvl_rows[q];
            double s = op[i];
            for (int k = L_ptr[i]; k < L_ptr[i + 1]; ++k) s -= L_val[k] * w[L_col[k]];
            w[i] = s;
        }
        grid.sync();
    }
    for (int l = 0; l < U_nlvl; ++l) { // backward with diagonal divide
        for (int q = Ulvl_ptr[l] + tidx; q < Ulvl_ptr[l + 1]; q += stride) {
            int i = Ulvl_rows[q];
            double s = w[i];
            for (int k = U_ptr[i]; k < U_ptr[i + 1]; ++k) s -= U_val[k] * w[U_col[k]];
            w[i] = s / D[i];
        }
        grid.sync();
    }
#else
    for (int i = tidx; i < N; i += stride) w[i] = b[i]; // finite stale state
    grid.sync();
#endif
    grid_nrm2(w, partials, scal, grid, &sh[0][0]); // beta
    if (bid == 0 && tid == 0) {
        scal[1] = scal[0]; // pbnorm = ||M^-1 b||
        for (int i = 0; i < (RESTART + 1) * RESTART; ++i) H[i] = 0.0;
        for (int i = 0; i <= RESTART; ++i) g[i] = 0.0;
        g[0] = scal[0];
        meta[0] = -1;
        meta[1] = 0;
        meta[2] = 0;
    }
    grid.sync();
    // NaN-safe normalize (EMPTY mode can reach beta = 0; zeros stay finite).
    double b0 = scal[0];
    double inv_beta = (b0 > 0.0) ? 1.0 / b0 : 0.0;
    for (int i = tidx; i < N; i += stride) V[i] = w[i] * inv_beta;
    grid.sync();

    for (long total = 0; total < nstep; ++total) {
        int j = (int)(total % RESTART);
        if (total > 0 && j == 0) {
            // Restart: least-squares update, fresh residual/beta/v0.
            if (bid == 0 && tid == 0) {
                double y[RESTART];
                for (int i = RESTART - 1; i >= 0; --i) {
                    double s = g[i];
                    for (int k = i + 1; k < RESTART; ++k) s -= H[i * RESTART + k] * y[k];
                    y[i] = (fabs(H[i * RESTART + i]) > 1e-30) ? s / H[i * RESTART + i] : 0.0;
                    tmp[i] = y[i];
                }
            }
            grid.sync();
            for (int i = tidx; i < N; i += stride) {
                double s = 0.0;
                for (int k = 0; k < RESTART; ++k) s += V[(size_t)k * N + i] * tmp[k];
                x[i] += s;
            }
            grid.sync();
#ifndef EMPTY_FORCE
            for (int i = tidx; i < N; i += stride) { // op = b - A x
                double s = 0.0;
                for (int k = A_ptr[i]; k < A_ptr[i + 1]; ++k) s += A_val[k] * x[A_col[k]];
                op[i] = b[i] - s;
            }
            grid.sync();
            for (int l = 0; l < L_nlvl; ++l) {
                for (int q = Llvl_ptr[l] + tidx; q < Llvl_ptr[l + 1]; q += stride) {
                    int i = Llvl_rows[q];
                    double s = op[i];
                    for (int k = L_ptr[i]; k < L_ptr[i + 1]; ++k)
                        s -= L_val[k] * w[L_col[k]];
                    w[i] = s;
                }
                grid.sync();
            }
            for (int l = 0; l < U_nlvl; ++l) {
                for (int q = Ulvl_ptr[l] + tidx; q < Ulvl_ptr[l + 1]; q += stride) {
                    int i = Ulvl_rows[q];
                    double s = w[i];
                    for (int k = U_ptr[i]; k < U_ptr[i + 1]; ++k)
                        s -= U_val[k] * w[U_col[k]];
                    w[i] = s / D[i];
                }
                grid.sync();
            }
#endif
            grid_nrm2(w, partials, scal, grid, &sh[0][0]);
            if (bid == 0 && tid == 0) {
                for (int i = 0; i < (RESTART + 1) * RESTART; ++i) H[i] = 0.0;
                for (int i = 0; i <= RESTART; ++i) g[i] = 0.0;
                g[0] = scal[0];
            }
            grid.sync();
            double br = scal[0];
            double ib = (br > 0.0) ? 1.0 / br : 0.0;
            for (int i = tidx; i < N; i += stride) V[i] = w[i] * ib;
            grid.sync();
        }
        // w = M^-1 A v_j.
#ifndef EMPTY_FORCE
        for (int i = tidx; i < N; i += stride) {
            double s = 0.0;
            for (int k = A_ptr[i]; k < A_ptr[i + 1]; ++k)
                s += A_val[k] * V[(size_t)j * N + A_col[k]];
            op[i] = s;
        }
        grid.sync();
        for (int l = 0; l < L_nlvl; ++l) {
            for (int q = Llvl_ptr[l] + tidx; q < Llvl_ptr[l + 1]; q += stride) {
                int i = Llvl_rows[q];
                double s = op[i];
                for (int k = L_ptr[i]; k < L_ptr[i + 1]; ++k)
                    s -= L_val[k] * w[L_col[k]];
                w[i] = s;
            }
            grid.sync();
        }
        for (int l = 0; l < U_nlvl; ++l) {
            for (int q = Ulvl_ptr[l] + tidx; q < Ulvl_ptr[l + 1]; q += stride) {
                int i = Ulvl_rows[q];
                double s = w[i];
                for (int k = U_ptr[i]; k < U_ptr[i + 1]; ++k)
                    s -= U_val[k] * w[U_col[k]];
                w[i] = s / D[i];
            }
            grid.sync();
        }
#endif
        grid.sync();
        // Batched CGS with DGKS reorthogonalisation.
        grid_nrm2(w, partials, tmp + 30, grid, &sh[0][0]); // nb in tmp[30]
        batched_dots(V, w, j, tmp, partials, grid, &sh[0][0]);
        if (bid == 0 && tid == 0)
            for (int i = 0; i <= j; ++i) H[i * RESTART + j] = tmp[i];
        grid.sync();
        for (int i = tidx; i < N; i += stride) {
            double s = 0.0;
            for (int k = 0; k <= j; ++k) s += V[(size_t)k * N + i] * tmp[k];
            w[i] -= s;
        }
        grid.sync();
        grid_nrm2(w, partials, tmp + 31, grid, &sh[0][0]); // n in tmp[31]
        double nb = tmp[30], wn = tmp[31];
        if (wn < 0.707 * nb) {
            batched_dots(V, w, j, tmp + 32, partials, grid, &sh[0][0]);
            if (bid == 0 && tid == 0) {
                for (int i = 0; i <= j; ++i) H[i * RESTART + j] += tmp[32 + i];
                meta[1]++;
            }
            grid.sync();
            for (int i = tidx; i < N; i += stride) {
                double s = 0.0;
                for (int k = 0; k <= j; ++k) s += V[(size_t)k * N + i] * tmp[32 + k];
                w[i] -= s;
            }
            grid.sync();
            grid_nrm2(w, partials, tmp + 31, grid, &sh[0][0]);
            wn = tmp[31];
        }
        if (bid == 0 && tid == 0) H[(j + 1) * RESTART + j] = wn;
        grid.sync();
        // Normalize (breakdown-guarded) + Givens + residual estimate.
        if (wn < 1e-20) {
            if (bid == 0 && tid == 0) meta[2] = 1;
            for (int i = tidx; i < N; i += stride)
                V[(size_t)(j + 1) * N + i] = 0.0;
            grid.sync();
        } else {
            double inv_n = 1.0 / wn;
            for (int i = tidx; i < N; i += stride)
                V[(size_t)(j + 1) * N + i] = w[i] * inv_n;
            grid.sync();
        }
        if (bid == 0 && tid == 0) {
            for (int i = 0; i < j; ++i) {
                double hij = H[i * RESTART + j], hi1j = H[(i + 1) * RESTART + j];
                H[i * RESTART + j] = cs[i] * hij + sn[i] * hi1j;
                H[(i + 1) * RESTART + j] = -sn[i] * hij + cs[i] * hi1j;
            }
            double a = H[j * RESTART + j], b_ = H[(j + 1) * RESTART + j];
            double c, s;
            if (fabs(b_) < 1e-30) {
                c = 1.0;
                s = 0.0;
            } else if (fabs(a) < 1e-30) {
                c = 0.0;
                s = 1.0;
            } else {
                double r = sqrt(a * a + b_ * b_);
                c = a / r;
                s = b_ / r;
            }
            cs[j] = c;
            sn[j] = s;
            H[j * RESTART + j] = c * a + s * b_;
            H[(j + 1) * RESTART + j] = 0.0;
            double gj = g[j], gj1 = g[j + 1];
            g[j] = c * gj + s * gj1;
            g[j + 1] = -s * gj + c * gj1;
            double est = fabs(g[j + 1]) / scal[1];
            res_hist[total] = est;
            if (meta[0] < 0 && est < 1e-8) meta[0] = (int)total + 1;
        }
        grid.sync();
    }
    // Final least-squares update (mirrors the native exit update).
    if (bid == 0 && tid == 0) {
        int k = (int)((nstep - 1) % RESTART) + 1;
        meta[3] = k;
        for (int i = k - 1; i >= 0; --i) {
            double s = g[i];
            for (int q = i + 1; q < k; ++q) s -= H[i * RESTART + q] * tmp[q];
            tmp[i] = (fabs(H[i * RESTART + i]) > 1e-30) ? s / H[i * RESTART + i] : 0.0;
        }
    }
    grid.sync();
    int kfin = meta[3];
    for (int i = tidx; i < N; i += stride) {
        double s = 0.0;
        for (int k = 0; k < kfin; ++k) s += V[(size_t)k * N + i] * tmp[k];
        x[i] += s;
    }
    grid.sync();
}

int main(int argc, char **argv) {
    long nstep = argc > 1 ? atol(argv[1]) : 12240L; // native total GMRES iters
    if (nstep < 1 || nstep > 200000L) {
        printf("nstep must be in [1, 200000], got %ld\n", nstep);
        return 1;
    }
    double tau0 = stab_tau(0.0);
    std::vector<Coo> coo;
    coo.reserve(9000 * 81);
    assemble_picard1_coo(coo);
    std::vector<int> row_ptr, cols, l_ptr, l_cols, u_ptr, u_cols;
    std::vector<double> vals, l_vals, u_vals, diag, rhs;
    coo_to_csr(coo, row_ptr, cols, vals);
    double rhs_norm = apply_dirichlet(row_ptr, cols, vals, rhs);
    size_t nnz = vals.size();
    int skipped = ilu0_factorize(row_ptr, cols, vals, l_ptr, l_cols, l_vals,
                                 u_ptr, u_cols, u_vals, diag);
    std::vector<int> l_lvl_ptr, l_lvl_rows, u_lvl_ptr, u_lvl_rows;
    level_schedule(l_ptr, l_cols, true, l_lvl_ptr, l_lvl_rows);
    level_schedule(u_ptr, u_cols, false, u_lvl_ptr, u_lvl_rows);
    int l_nlvl = (int)l_lvl_ptr.size() - 1, u_nlvl = (int)u_lvl_ptr.size() - 1;
    size_t lwmax = 0, uwmax = 0;
    for (int l = 0; l < l_nlvl; ++l)
        lwmax = std::max(lwmax, (size_t)(l_lvl_ptr[l + 1] - l_lvl_ptr[l]));
    for (int l = 0; l < u_nlvl; ++l)
        uwmax = std::max(uwmax, (size_t)(u_lvl_ptr[l + 1] - u_lvl_ptr[l]));
    printf("proxy: n=%d nnz=%zu rhs_norm=%.12e tau0=%.6e ilu_skipped=%d "
           "L_nnz=%zu U_nnz=%zu L_levels=%d(wmax=%zu) U_levels=%d(wmax=%zu)\n",
           N, nnz, rhs_norm, tau0, skipped, l_vals.size(), u_vals.size(), l_nlvl,
           lwmax, u_nlvl, uwmax);

    // Host N=1 replica: first GMRES iteration in serial fp, for the
    // device-math check (device res_hist[0] must match ~1e-12).
    std::vector<double> hv0(N), hop(N), hw(N), hx(N, 0.0);
    host_tri_fwd(l_ptr, l_cols, l_vals, rhs.data(), hop.data());
    host_tri_bwd(u_ptr, u_cols, u_vals, diag.data(), hop.data(), hw.data());
    double hbeta = sqrt(host_dot(hw.data(), hw.data()));
    for (int i = 0; i < N; ++i) hv0[i] = hw[i] / hbeta;
    host_spmv(row_ptr, cols, vals, hv0.data(), hop.data());
    host_tri_fwd(l_ptr, l_cols, l_vals, hop.data(), hw.data());
    host_tri_bwd(u_ptr, u_cols, u_vals, diag.data(), hw.data(), hop.data());
    double hnb = sqrt(host_dot(hop.data(), hop.data()));
    double hc0 = host_dot(hop.data(), hv0.data());
    for (int i = 0; i < N; ++i) hop[i] -= hc0 * hv0[i];
    double hn = sqrt(host_dot(hop.data(), hop.data()));
    if (hn < 0.707 * hnb) {
        double s0 = host_dot(hop.data(), hv0.data());
        hc0 += s0;
        for (int i = 0; i < N; ++i) hop[i] -= s0 * hv0[i];
        hn = sqrt(host_dot(hop.data(), hop.data()));
    }
    double hr = sqrt(hc0 * hc0 + hn * hn);
    double host_est1 = (hn / hr) * (hbeta / hbeta); // = hn/hr; pbnorm is beta
    printf("host_N1: beta=%.12e est1=%.12e (device must match)\n", hbeta, host_est1);

    CHECK(cudaSetDevice(0));
    int coop = 0;
    CHECK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, 0));
    if (!coop) {
        printf("cooperative launch unsupported; fail closed\n");
        return 1;
    }
    int max_blocks = 0, sm_count = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, cavity_loop,
                                                        TB, 0));
    CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
    int coop_max = max_blocks * sm_count;
    int nblocks = 128;
    if (nblocks > coop_max) nblocks = coop_max;
    if (nblocks < 1) {
        printf("no cooperative blocks available; fail closed\n");
        return 1;
    }
    printf("launch: blocks=%d threads=%d (coop_max=%d)\n", nblocks, TB, coop_max);

    // Upload operator, factors, schedules, rhs.
    int *d_Ap = nullptr, *d_Ac = nullptr, *d_Lp = nullptr, *d_Lc = nullptr;
    int *d_Up = nullptr, *d_Uc = nullptr, *d_Llp = nullptr, *d_Llr = nullptr;
    int *d_Ulp = nullptr, *d_Ulr = nullptr, *d_meta = nullptr;
    double *d_Av = nullptr, *d_Lv = nullptr, *d_Uv = nullptr, *d_D = nullptr;
    double *d_b = nullptr, *d_x = nullptr, *d_V = nullptr, *d_w = nullptr;
    double *d_op = nullptr, *d_H = nullptr, *d_cs = nullptr, *d_sn = nullptr;
    double *d_g = nullptr, *d_tmp = nullptr, *d_hist = nullptr;
    double *d_scal = nullptr, *d_part = nullptr;
    CHECK(cudaMalloc(&d_Ap, (N + 1) * sizeof(int)));
    CHECK(cudaMalloc(&d_Ac, nnz * sizeof(int)));
    CHECK(cudaMalloc(&d_Av, nnz * sizeof(double)));
    CHECK(cudaMalloc(&d_Lp, (N + 1) * sizeof(int)));
    CHECK(cudaMalloc(&d_Lc, l_vals.size() * sizeof(int)));
    CHECK(cudaMalloc(&d_Lv, l_vals.size() * sizeof(double)));
    CHECK(cudaMalloc(&d_Up, (N + 1) * sizeof(int)));
    CHECK(cudaMalloc(&d_Uc, u_vals.size() * sizeof(int)));
    CHECK(cudaMalloc(&d_Uv, u_vals.size() * sizeof(double)));
    CHECK(cudaMalloc(&d_D, N * sizeof(double)));
    CHECK(cudaMalloc(&d_Llp, (l_nlvl + 1) * sizeof(int)));
    CHECK(cudaMalloc(&d_Llr, N * sizeof(int)));
    CHECK(cudaMalloc(&d_Ulp, (u_nlvl + 1) * sizeof(int)));
    CHECK(cudaMalloc(&d_Ulr, N * sizeof(int)));
    CHECK(cudaMalloc(&d_b, N * sizeof(double)));
    CHECK(cudaMalloc(&d_x, N * sizeof(double)));
    CHECK(cudaMalloc(&d_V, (size_t)NBASIS * N * sizeof(double)));
    CHECK(cudaMalloc(&d_w, N * sizeof(double)));
    CHECK(cudaMalloc(&d_op, N * sizeof(double)));
    CHECK(cudaMalloc(&d_H, (size_t)(RESTART + 1) * RESTART * sizeof(double)));
    CHECK(cudaMalloc(&d_cs, RESTART * sizeof(double)));
    CHECK(cudaMalloc(&d_sn, RESTART * sizeof(double)));
    CHECK(cudaMalloc(&d_g, NBASIS * sizeof(double)));
    CHECK(cudaMalloc(&d_tmp, 64 * sizeof(double)));
    CHECK(cudaMalloc(&d_hist, (size_t)nstep * sizeof(double)));
    CHECK(cudaMalloc(&d_meta, 4 * sizeof(int)));
    CHECK(cudaMalloc(&d_scal, 3 * sizeof(double)));
    CHECK(cudaMalloc(&d_part, (size_t)nblocks * 32 * sizeof(double)));
    CHECK(cudaMemcpy(d_Ap, row_ptr.data(), (N + 1) * sizeof(int),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_Ac, cols.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_Av, vals.data(), nnz * sizeof(double),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_Lp, l_ptr.data(), (N + 1) * sizeof(int),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_Lc, l_cols.data(), l_vals.size() * sizeof(int),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_Lv, l_vals.data(), l_vals.size() * sizeof(double),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_Up, u_ptr.data(), (N + 1) * sizeof(int),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_Uc, u_cols.data(), u_vals.size() * sizeof(int),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_Uv, u_vals.data(), u_vals.size() * sizeof(double),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_D, diag.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_Llp, l_lvl_ptr.data(), (l_nlvl + 1) * sizeof(int),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_Llr, l_lvl_rows.data(), N * sizeof(int),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_Ulp, u_lvl_ptr.data(), (u_nlvl + 1) * sizeof(int),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_Ulr, u_lvl_rows.data(), N * sizeof(int),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, rhs.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    void *args[] = {&d_Ap, &d_Ac, &d_Av, &d_Lp, &d_Lc, &d_Lv, &d_Up,
                    &d_Uc, &d_Uv, &d_D,  &d_Llp, &d_Llr, &l_nlvl, &d_Ulp,
                    &d_Ulr, &u_nlvl, &d_b, &d_x, &d_V, &d_w, &d_op, &d_H,
                    &d_cs, &d_sn, &d_g, &d_tmp, &d_hist, &d_meta, &d_scal,
                    &d_part, &nstep};
    // Warmup (untimed) then the device-timed run.
    CHECK(cudaLaunchCooperativeKernel((void *)cavity_loop, nblocks, TB, args, 0,
                                      nullptr));
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));
    CHECK(cudaEventRecord(t0));
    CHECK(cudaLaunchCooperativeKernel((void *)cavity_loop, nblocks, TB, args, 0,
                                      nullptr));
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    CHECK(cudaGetLastError());
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, t0, t1));
    std::vector<double> hx_out(N), hist((size_t)nstep);
    int meta[4] = {-9, -9, -9, -9};
    double scal[3] = {0, 0, 0};
    CHECK(cudaMemcpy(hx_out.data(), d_x, N * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hist.data(), d_hist, (size_t)nstep * sizeof(double),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(meta, d_meta, 4 * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(scal, d_scal, 3 * sizeof(double), cudaMemcpyDeviceToHost));
    // Bit-identical repeat (untimed) for the determinism check.
    CHECK(cudaLaunchCooperativeKernel((void *)cavity_loop, nblocks, TB, args, 0,
                                      nullptr));
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    std::vector<double> hx_rep(N), hist_rep((size_t)nstep);
    CHECK(cudaMemcpy(hx_rep.data(), d_x, N * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hist_rep.data(), d_hist, (size_t)nstep * sizeof(double),
                     cudaMemcpyDeviceToHost));
    bool identical = hx_out == hx_rep && hist == hist_rep;

    // True residual + centerline on the host.
    std::vector<double> hres(N);
    host_spmv(row_ptr, cols, vals, hx_out.data(), hres.data());
    double rr = 0.0;
    for (int i = 0; i < N; ++i) {
        double d = rhs[i] - hres[i];
        rr += d * d;
    }
    double true_res = sqrt(rr) / rhs_norm;
    auto at = [&](long k) -> double {
        if (k < 1 || k > nstep) return -1.0;
        return hist[(size_t)k - 1];
    };
    printf("nstep=%ld wall=%.3f s (%.3f us/step) res0=%.6e res1=%.6e "
           "res30=%.6e res330=%.6e resN=%.6e true_res=%.6e tol_hit=%d "
           "reorths=%d breakdown=%d identical=%d\n",
           nstep, ms / 1000.0, ms * 1000.0 / nstep, 1.0, at(1), at(30), at(330),
           at(nstep), true_res, meta[0], meta[1], meta[2], identical ? 1 : 0);
    printf("centerline y,u,p:");
    for (int s = 0; s <= 8; ++s) {
        int a = NODE(NM / 2, s * 8);
        printf(" %.6f,%.6e,%.6e", s * 0.125, hx_out[UX(a)], hx_out[PR(a)]);
    }
    printf("\n");
    return 0;
}
