// shell_twisted_beam G0 CUDA proxy (Unit 4).
//
// Question: what does one native explicit T3-shell step cost on this
// GPU at the exact twisted deck size (231 nodes, 192 quads -> 384
// CST/DKT triangles), and is a G1 port credible?
//
// Design: persistent 1-block/256-thread kernel; host builds the
// deck-faithful T3 constants (frames, CST B, centroid DKT B, lumped
// masses) from the embedded deck mesh, splits quads along the 0-2
// diagonal exactly like the native bench arm, and greedy-colors
// triangles node-disjoint for race-free scatter. Each step replicates
// the native velocity-Verlet sequence (half-kick with F(t), drift,
// half-kick with the same F(t) and recomputed internal forces) with
// the deck 20 ms ramp / 60 ms stop and the 4.448 N tip load on node
// 228. Deck material: E = 1.999479615022e11 Pa, nu = 0.22,
// rho = 7800 kg/m^3, t = 8.128e-3 m; root deck nodes 1..7 pinned on
// all 6 DOFs. FP64 throughout.
//
// Mesh tables below are generated from
// crates/sonium-qa/decks/shell_twisted_beam/shell_twisted_beam_0000.rad
// (/NODE 231 contiguous 1..231, /SHELL 192, reindexed 0-based) by:
//
//   python3 -c " ..."  # see unit4 G0 notes: parse /NODE xyz and
//   # /SHELL quads, print MESH_X (%.6e) and MESH_Q; 6 decimals keep
//   # geometry error ~1e-9 relative (nm on 0.3 m), far below any probe.
//
// Validation (fidelity before timing): N=1 must show zero prestress
// (no NaN, tip ~0); the full 300000-step run must reproduce the
// native bench tip hold mean (t >= 32 ms) 1.350630e-4 m within a few
// % (same linear physics, same dt family, independent assembly).
// Native reference: tmp/u4-twisted.log (sonium-only quick, correct).
//
// Time model: measured device wall for N steps; per-step cost from a
// 2000-step run and a full-N run (linearity check); EMPTY_FORCE build
// (-DEMPTY_FORCE) isolates loop/update overhead from force assembly.
//
// FLOP/step estimate (host-counted, FP64): per triangle CST ~100
// (B 3x6, C 3x3, B^T scatter) + DKT ~140 (B 3x9, Db 3x3, B^T) +
// frame maps ~110 + drilling 3  => ~350; x384 tris x2 assemblies
// (Verlet needs forces twice) = 269 kFLOP + update 1386 DOF x ~12
// = 17 kFLOP  => ~286 kFLOP/step. Bytes/step (DRAM, cold): read
// tri constants ~384 x (9+18+27+1+3) x 8 B = 178 kB + u/m 2 x 11 kB,
// write f 11 kB + hist 8 B  => ~200 kB/step => ~1.4 FLOP/byte:
// memory-bound on paper; the roofline report measures the truth.
//
// Build: nvcc -O2 -arch=native -o proxy shell_twisted_beam_cuda_proxy.cu
//        nvcc -O2 -arch=native -DEMPTY_FORCE -o proxy_empty shell_twisted_beam_cuda_proxy.cu
// Run: ./proxy [nstep]   (default 300000 = 60 ms at dt 2.0e-7 s)

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(call)                                                                          \
    do {                                                                                     \
        cudaError_t err = (call);                                                            \
        if (err != cudaSuccess) {                                                            \
            fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1);                                                                         \
        }                                                                                    \
    } while (0)

#define NN 231
#define NQ 192
#define NTRI (NQ * 2)
#define NDOF (NN * 6)
#define TIP_NODE 227 // deck node 228, 0-based
#define TIP_DOF (TIP_NODE * 6 + 2)
#define DT 2.0e-7
#define RAMP_T 0.02
#define HOLD_T 0.032
#define CLOAD_F 4.448221615260
#define YOUNGS 1.999479615022e11
#define POISSON 0.22
#define DENSITY 7800.0
#define THICK 8.128e-3
// Native bench tip hold mean (t >= 32 ms), tmp/u4-twisted.log.
#define NATIVE_TIP_MEAN 1.350630e-4

static const double MESH_X[NN][3] = {
{0.000000e+00,-1.397000e-02,-0.000000e+00},
{0.000000e+00,-9.313333e-03,-0.000000e+00},
{0.000000e+00,-4.656667e-03,-0.000000e+00},
{0.000000e+00,0.000000e+00,0.000000e+00},
{0.000000e+00,4.656667e-03,0.000000e+00},
{0.000000e+00,9.313333e-03,0.000000e+00},
{0.000000e+00,1.397000e-02,0.000000e+00},
{9.525000e-03,-1.395317e-02,-6.854754e-04},
{9.525000e-03,-9.302115e-03,-4.569836e-04},
{9.525000e-03,-4.651058e-03,-2.284918e-04},
{9.525000e-03,0.000000e+00,0.000000e+00},
{9.525000e-03,4.651058e-03,2.284918e-04},
{9.525000e-03,9.302115e-03,4.569836e-04},
{9.525000e-03,1.395317e-02,6.854754e-04},
{1.905000e-02,-1.390273e-02,-1.369299e-03},
{1.905000e-02,-9.268487e-03,-9.128663e-04},
{1.905000e-02,-4.634244e-03,-4.564332e-04},
{1.905000e-02,0.000000e+00,0.000000e+00},
{1.905000e-02,4.634244e-03,4.564332e-04},
{1.905000e-02,9.268487e-03,9.128663e-04},
{1.905000e-02,1.390273e-02,1.369299e-03},
{2.857500e-02,-1.381880e-02,-2.049825e-03},
{2.857500e-02,-9.212531e-03,-1.366550e-03},
{2.857500e-02,-4.606265e-03,-6.832749e-04},
{2.857500e-02,0.000000e+00,0.000000e+00},
{2.857500e-02,4.606265e-03,6.832749e-04},
{2.857500e-02,9.212531e-03,1.366550e-03},
{2.857500e-02,1.381880e-02,2.049825e-03},
{3.810000e-02,-1.370157e-02,-2.725412e-03},
{3.810000e-02,-9.134380e-03,-1.816941e-03},
{3.810000e-02,-4.567190e-03,-9.084706e-04},
{3.810000e-02,0.000000e+00,0.000000e+00},
{3.810000e-02,4.567190e-03,9.084706e-04},
{3.810000e-02,9.134380e-03,1.816941e-03},
{3.810000e-02,1.370157e-02,2.725412e-03},
{4.762500e-02,-1.355134e-02,-3.394433e-03},
{4.762500e-02,-9.034224e-03,-2.262955e-03},
{4.762500e-02,-4.517112e-03,-1.131478e-03},
{4.762500e-02,0.000000e+00,0.000000e+00},
{4.762500e-02,4.517112e-03,1.131478e-03},
{4.762500e-02,9.034224e-03,2.262955e-03},
{4.762500e-02,1.355134e-02,3.394433e-03},
{5.715000e-02,-1.336846e-02,-4.055277e-03},
{5.715000e-02,-8.912304e-03,-2.703518e-03},
{5.715000e-02,-4.456152e-03,-1.351759e-03},
{5.715000e-02,0.000000e+00,0.000000e+00},
{5.715000e-02,4.456152e-03,1.351759e-03},
{5.715000e-02,8.912304e-03,2.703518e-03},
{5.715000e-02,1.336846e-02,4.055277e-03},
{6.667500e-02,-1.315337e-02,-4.706351e-03},
{6.667500e-02,-8.768914e-03,-3.137568e-03},
{6.667500e-02,-4.384457e-03,-1.568784e-03},
{6.667500e-02,0.000000e+00,0.000000e+00},
{6.667500e-02,4.384457e-03,1.568784e-03},
{6.667500e-02,8.768914e-03,3.137568e-03},
{6.667500e-02,1.315337e-02,4.706351e-03},
{7.620000e-02,-1.290660e-02,-5.346088e-03},
{7.620000e-02,-8.604398e-03,-3.564058e-03},
{7.620000e-02,-4.302199e-03,-1.782029e-03},
{7.620000e-02,0.000000e+00,0.000000e+00},
{7.620000e-02,4.302199e-03,1.782029e-03},
{7.620000e-02,8.604398e-03,3.564058e-03},
{7.620000e-02,1.290660e-02,5.346088e-03},
{8.572500e-02,-1.262873e-02,-5.972945e-03},
{8.572500e-02,-8.419154e-03,-3.981963e-03},
{8.572500e-02,-4.209577e-03,-1.990982e-03},
{8.572500e-02,0.000000e+00,0.000000e+00},
{8.572500e-02,4.209577e-03,1.990982e-03},
{8.572500e-02,8.419154e-03,3.981963e-03},
{8.572500e-02,1.262873e-02,5.972945e-03},
{9.525000e-02,-1.232044e-02,-6.585412e-03},
{9.525000e-02,-8.213627e-03,-4.390275e-03},
{9.525000e-02,-4.106813e-03,-2.195137e-03},
{9.525000e-02,0.000000e+00,0.000000e+00},
{9.525000e-02,4.106813e-03,2.195137e-03},
{9.525000e-02,8.213627e-03,4.390275e-03},
{9.525000e-02,1.232044e-02,6.585412e-03},
{1.047750e-01,-1.198247e-02,-7.182015e-03},
{1.047750e-01,-7.988312e-03,-4.788010e-03},
{1.047750e-01,-3.994156e-03,-2.394005e-03},
{1.047750e-01,0.000000e+00,0.000000e+00},
{1.047750e-01,3.994156e-03,2.394005e-03},
{1.047750e-01,7.988312e-03,4.788010e-03},
{1.047750e-01,1.198247e-02,7.182015e-03},
{1.143000e-01,-1.161563e-02,-7.761316e-03},
{1.143000e-01,-7.743754e-03,-5.174211e-03},
{1.143000e-01,-3.871877e-03,-2.587105e-03},
{1.143000e-01,0.000000e+00,0.000000e+00},
{1.143000e-01,3.871877e-03,2.587105e-03},
{1.143000e-01,7.743754e-03,5.174211e-03},
{1.143000e-01,1.161563e-02,7.761316e-03},
{1.238250e-01,-1.122081e-02,-8.321919e-03},
{1.238250e-01,-7.480539e-03,-5.547946e-03},
{1.238250e-01,-3.740270e-03,-2.773973e-03},
{1.238250e-01,0.000000e+00,0.000000e+00},
{1.238250e-01,3.740270e-03,2.773973e-03},
{1.238250e-01,7.480539e-03,5.547946e-03},
{1.238250e-01,1.122081e-02,8.321919e-03},
{1.333500e-01,-1.079896e-02,-8.862474e-03},
{1.333500e-01,-7.199304e-03,-5.908316e-03},
{1.333500e-01,-3.599652e-03,-2.954158e-03},
{1.333500e-01,0.000000e+00,0.000000e+00},
{1.333500e-01,3.599652e-03,2.954158e-03},
{1.333500e-01,7.199304e-03,5.908316e-03},
{1.333500e-01,1.079896e-02,8.862474e-03},
{1.428750e-01,-1.035109e-02,-9.381679e-03},
{1.428750e-01,-6.900725e-03,-6.254452e-03},
{1.428750e-01,-3.450362e-03,-3.127226e-03},
{1.428750e-01,0.000000e+00,0.000000e+00},
{1.428750e-01,3.450362e-03,3.127226e-03},
{1.428750e-01,6.900725e-03,6.254452e-03},
{1.428750e-01,1.035109e-02,9.381679e-03},
{1.524000e-01,-9.878282e-03,-9.878282e-03},
{1.524000e-01,-6.585521e-03,-6.585521e-03},
{1.524000e-01,-3.292761e-03,-3.292761e-03},
{1.524000e-01,0.000000e+00,0.000000e+00},
{1.524000e-01,3.292761e-03,3.292761e-03},
{1.524000e-01,6.585521e-03,6.585521e-03},
{1.524000e-01,9.878282e-03,9.878282e-03},
{1.619250e-01,-9.381679e-03,-1.035109e-02},
{1.619250e-01,-6.254452e-03,-6.900725e-03},
{1.619250e-01,-3.127226e-03,-3.450362e-03},
{1.619250e-01,0.000000e+00,0.000000e+00},
{1.619250e-01,3.127226e-03,3.450362e-03},
{1.619250e-01,6.254452e-03,6.900725e-03},
{1.619250e-01,9.381679e-03,1.035109e-02},
{1.714500e-01,-8.862474e-03,-1.079896e-02},
{1.714500e-01,-5.908316e-03,-7.199304e-03},
{1.714500e-01,-2.954158e-03,-3.599652e-03},
{1.714500e-01,0.000000e+00,0.000000e+00},
{1.714500e-01,2.954158e-03,3.599652e-03},
{1.714500e-01,5.908316e-03,7.199304e-03},
{1.714500e-01,8.862474e-03,1.079896e-02},
{1.809750e-01,-8.321919e-03,-1.122081e-02},
{1.809750e-01,-5.547946e-03,-7.480539e-03},
{1.809750e-01,-2.773973e-03,-3.740270e-03},
{1.809750e-01,0.000000e+00,0.000000e+00},
{1.809750e-01,2.773973e-03,3.740270e-03},
{1.809750e-01,5.547946e-03,7.480539e-03},
{1.809750e-01,8.321919e-03,1.122081e-02},
{1.905000e-01,-7.761316e-03,-1.161563e-02},
{1.905000e-01,-5.174211e-03,-7.743754e-03},
{1.905000e-01,-2.587105e-03,-3.871877e-03},
{1.905000e-01,0.000000e+00,0.000000e+00},
{1.905000e-01,2.587105e-03,3.871877e-03},
{1.905000e-01,5.174211e-03,7.743754e-03},
{1.905000e-01,7.761316e-03,1.161563e-02},
{2.000250e-01,-7.182015e-03,-1.198247e-02},
{2.000250e-01,-4.788010e-03,-7.988312e-03},
{2.000250e-01,-2.394005e-03,-3.994156e-03},
{2.000250e-01,0.000000e+00,0.000000e+00},
{2.000250e-01,2.394005e-03,3.994156e-03},
{2.000250e-01,4.788010e-03,7.988312e-03},
{2.000250e-01,7.182015e-03,1.198247e-02},
{2.095500e-01,-6.585412e-03,-1.232044e-02},
{2.095500e-01,-4.390275e-03,-8.213627e-03},
{2.095500e-01,-2.195137e-03,-4.106813e-03},
{2.095500e-01,0.000000e+00,0.000000e+00},
{2.095500e-01,2.195137e-03,4.106813e-03},
{2.095500e-01,4.390275e-03,8.213627e-03},
{2.095500e-01,6.585412e-03,1.232044e-02},
{2.190750e-01,-5.972945e-03,-1.262873e-02},
{2.190750e-01,-3.981963e-03,-8.419154e-03},
{2.190750e-01,-1.990982e-03,-4.209577e-03},
{2.190750e-01,0.000000e+00,0.000000e+00},
{2.190750e-01,1.990982e-03,4.209577e-03},
{2.190750e-01,3.981963e-03,8.419154e-03},
{2.190750e-01,5.972945e-03,1.262873e-02},
{2.286000e-01,-5.346088e-03,-1.290660e-02},
{2.286000e-01,-3.564058e-03,-8.604398e-03},
{2.286000e-01,-1.782029e-03,-4.302199e-03},
{2.286000e-01,0.000000e+00,0.000000e+00},
{2.286000e-01,1.782029e-03,4.302199e-03},
{2.286000e-01,3.564058e-03,8.604398e-03},
{2.286000e-01,5.346088e-03,1.290660e-02},
{2.381250e-01,-4.706351e-03,-1.315337e-02},
{2.381250e-01,-3.137568e-03,-8.768914e-03},
{2.381250e-01,-1.568784e-03,-4.384457e-03},
{2.381250e-01,0.000000e+00,0.000000e+00},
{2.381250e-01,1.568784e-03,4.384457e-03},
{2.381250e-01,3.137568e-03,8.768914e-03},
{2.381250e-01,4.706351e-03,1.315337e-02},
{2.476500e-01,-4.055277e-03,-1.336846e-02},
{2.476500e-01,-2.703518e-03,-8.912304e-03},
{2.476500e-01,-1.351759e-03,-4.456152e-03},
{2.476500e-01,0.000000e+00,0.000000e+00},
{2.476500e-01,1.351759e-03,4.456152e-03},
{2.476500e-01,2.703518e-03,8.912304e-03},
{2.476500e-01,4.055277e-03,1.336846e-02},
{2.571750e-01,-3.394433e-03,-1.355134e-02},
{2.571750e-01,-2.262955e-03,-9.034224e-03},
{2.571750e-01,-1.131478e-03,-4.517112e-03},
{2.571750e-01,0.000000e+00,0.000000e+00},
{2.571750e-01,1.131478e-03,4.517112e-03},
{2.571750e-01,2.262955e-03,9.034224e-03},
{2.571750e-01,3.394433e-03,1.355134e-02},
{2.667000e-01,-2.725412e-03,-1.370157e-02},
{2.667000e-01,-1.816941e-03,-9.134380e-03},
{2.667000e-01,-9.084706e-04,-4.567190e-03},
{2.667000e-01,0.000000e+00,0.000000e+00},
{2.667000e-01,9.084706e-04,4.567190e-03},
{2.667000e-01,1.816941e-03,9.134380e-03},
{2.667000e-01,2.725412e-03,1.370157e-02},
{2.762250e-01,-2.049825e-03,-1.381880e-02},
{2.762250e-01,-1.366550e-03,-9.212531e-03},
{2.762250e-01,-6.832749e-04,-4.606265e-03},
{2.762250e-01,0.000000e+00,0.000000e+00},
{2.762250e-01,6.832749e-04,4.606265e-03},
{2.762250e-01,1.366550e-03,9.212531e-03},
{2.762250e-01,2.049825e-03,1.381880e-02},
{2.857500e-01,-1.369299e-03,-1.390273e-02},
{2.857500e-01,-9.128663e-04,-9.268487e-03},
{2.857500e-01,-4.564332e-04,-4.634244e-03},
{2.857500e-01,0.000000e+00,0.000000e+00},
{2.857500e-01,4.564332e-04,4.634244e-03},
{2.857500e-01,9.128663e-04,9.268487e-03},
{2.857500e-01,1.369299e-03,1.390273e-02},
{2.952750e-01,-6.854754e-04,-1.395317e-02},
{2.952750e-01,-4.569836e-04,-9.302115e-03},
{2.952750e-01,-2.284918e-04,-4.651058e-03},
{2.952750e-01,0.000000e+00,0.000000e+00},
{2.952750e-01,2.284918e-04,4.651058e-03},
{2.952750e-01,4.569836e-04,9.302115e-03},
{2.952750e-01,6.854754e-04,1.395317e-02},
{3.048000e-01,-8.554158e-19,-1.397000e-02},
{3.048000e-01,-5.702772e-19,-9.313333e-03},
{3.048000e-01,-2.851386e-19,-4.656667e-03},
{3.048000e-01,0.000000e+00,0.000000e+00},
{3.048000e-01,2.851386e-19,4.656667e-03},
{3.048000e-01,5.702772e-19,9.313333e-03},
{3.048000e-01,8.554158e-19,1.397000e-02},
};

static const int MESH_Q[NQ][4] = {
{0,7,8,1},{1,8,9,2},{2,9,10,3},{3,10,11,4},{4,11,12,5},{5,12,13,6},
{7,14,15,8},{8,15,16,9},{9,16,17,10},{10,17,18,11},{11,18,19,12},{12,19,20,13},
{14,21,22,15},{15,22,23,16},{16,23,24,17},{17,24,25,18},{18,25,26,19},{19,26,27,20},
{21,28,29,22},{22,29,30,23},{23,30,31,24},{24,31,32,25},{25,32,33,26},{26,33,34,27},
{28,35,36,29},{29,36,37,30},{30,37,38,31},{31,38,39,32},{32,39,40,33},{33,40,41,34},
{35,42,43,36},{36,43,44,37},{37,44,45,38},{38,45,46,39},{39,46,47,40},{40,47,48,41},
{42,49,50,43},{43,50,51,44},{44,51,52,45},{45,52,53,46},{46,53,54,47},{47,54,55,48},
{49,56,57,50},{50,57,58,51},{51,58,59,52},{52,59,60,53},{53,60,61,54},{54,61,62,55},
{56,63,64,57},{57,64,65,58},{58,65,66,59},{59,66,67,60},{60,67,68,61},{61,68,69,62},
{63,70,71,64},{64,71,72,65},{65,72,73,66},{66,73,74,67},{67,74,75,68},{68,75,76,69},
{70,77,78,71},{71,78,79,72},{72,79,80,73},{73,80,81,74},{74,81,82,75},{75,82,83,76},
{77,84,85,78},{78,85,86,79},{79,86,87,80},{80,87,88,81},{81,88,89,82},{82,89,90,83},
{84,91,92,85},{85,92,93,86},{86,93,94,87},{87,94,95,88},{88,95,96,89},{89,96,97,90},
{91,98,99,92},{92,99,100,93},{93,100,101,94},{94,101,102,95},{95,102,103,96},{96,103,104,97},
{98,105,106,99},{99,106,107,100},{100,107,108,101},{101,108,109,102},{102,109,110,103},{103,110,111,104},
{105,112,113,106},{106,113,114,107},{107,114,115,108},{108,115,116,109},{109,116,117,110},{110,117,118,111},
{112,119,120,113},{113,120,121,114},{114,121,122,115},{115,122,123,116},{116,123,124,117},{117,124,125,118},
{119,126,127,120},{120,127,128,121},{121,128,129,122},{122,129,130,123},{123,130,131,124},{124,131,132,125},
{126,133,134,127},{127,134,135,128},{128,135,136,129},{129,136,137,130},{130,137,138,131},{131,138,139,132},
{133,140,141,134},{134,141,142,135},{135,142,143,136},{136,143,144,137},{137,144,145,138},{138,145,146,139},
{140,147,148,141},{141,148,149,142},{142,149,150,143},{143,150,151,144},{144,151,152,145},{145,152,153,146},
{147,154,155,148},{148,155,156,149},{149,156,157,150},{150,157,158,151},{151,158,159,152},{152,159,160,153},
{154,161,162,155},{155,162,163,156},{156,163,164,157},{157,164,165,158},{158,165,166,159},{159,166,167,160},
{161,168,169,162},{162,169,170,163},{163,170,171,164},{164,171,172,165},{165,172,173,166},{166,173,174,167},
{168,175,176,169},{169,176,177,170},{170,177,178,171},{171,178,179,172},{172,179,180,173},{173,180,181,174},
{175,182,183,176},{176,183,184,177},{177,184,185,178},{178,185,186,179},{179,186,187,180},{180,187,188,181},
{182,189,190,183},{183,190,191,184},{184,191,192,185},{185,192,193,186},{186,193,194,187},{187,194,195,188},
{189,196,197,190},{190,197,198,191},{191,198,199,192},{192,199,200,193},{193,200,201,194},{194,201,202,195},
{196,203,204,197},{197,204,205,198},{198,205,206,199},{199,206,207,200},{200,207,208,201},{201,208,209,202},
{203,210,211,204},{204,211,212,205},{205,212,213,206},{206,213,214,207},{207,214,215,208},{208,215,216,209},
{210,217,218,211},{211,218,219,212},{212,219,220,213},{213,220,221,214},{214,221,222,215},{215,222,223,216},
{217,224,225,218},{218,225,226,219},{219,226,227,220},{220,227,228,221},{221,228,229,222},{222,229,230,223},
};

// ---- Host-side T3 constant builder (CST + centroid DKT) ----
// Transcribed from sonium-fem shell element.rs / dkt.rs so the proxy
// assembles the same linear map as the native bench arm; validated by
// the tip hold-mean agreement, not by construction alone.

typedef struct {
    int nodes[3];
    double rot[9]; // local-to-global, row-major (native `rotation`)
    double bc[18]; // CST B, 3x6 row-major
    double bd[27]; // DKT B at centroid, 3x9 row-major
    double area;
    double drill; // fictitious drilling stiffness (native 1e-6 rule)
} TriConst;

static void build_frame(const double p0[3], const double p1[3], const double p2[3],
                        double rot[9], double lc[3][2]) {
    double v01[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
    double v02[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
    double l01 = sqrt(v01[0] * v01[0] + v01[1] * v01[1] + v01[2] * v01[2]);
    if (!(l01 > 1e-30)) { fprintf(stderr, "proxy: degenerate tri edge\n"); exit(1); }
    double ex[3] = {v01[0] / l01, v01[1] / l01, v01[2] / l01};
    double n[3] = {v01[1] * v02[2] - v01[2] * v02[1], v01[2] * v02[0] - v01[0] * v02[2],
                   v01[0] * v02[1] - v01[1] * v02[0]};
    double ln = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (!(ln > 1e-30)) { fprintf(stderr, "proxy: degenerate tri area\n"); exit(1); }
    double ez[3] = {n[0] / ln, n[1] / ln, n[2] / ln};
    double ey[3] = {ez[1] * ex[2] - ez[2] * ex[1], ez[2] * ex[0] - ez[0] * ex[2],
                    ez[0] * ex[1] - ez[1] * ex[0]};
    rot[0] = ex[0]; rot[1] = ey[0]; rot[2] = ez[0];
    rot[3] = ex[1]; rot[4] = ey[1]; rot[5] = ez[1];
    rot[6] = ex[2]; rot[7] = ey[2]; rot[8] = ez[2];
    const double *pts[3] = {p0, p1, p2};
    for (int i = 0; i < 3; i++) {
        double d[3] = {pts[i][0] - p0[0], pts[i][1] - p0[1], pts[i][2] - p0[2]};
        lc[i][0] = d[0] * ex[0] + d[1] * ex[1] + d[2] * ex[2];
        lc[i][1] = d[0] * ey[0] + d[1] * ey[1] + d[2] * ey[2];
    }
}

static void build_cst_b(const double lc[3][2], double area, double b[18]) {
    double x1 = lc[0][0], y1 = lc[0][1], x2 = lc[1][0], y2 = lc[1][1], x3 = lc[2][0], y3 = lc[2][1];
    double inv = 1.0 / (2.0 * area);
    double y23 = y2 - y3, y31 = y3 - y1, y12 = y1 - y2;
    double x32 = x3 - x2, x13 = x1 - x3, x21 = x2 - x1;
    double m[18] = {y23, 0, y31, 0, y12, 0, 0, x32, 0, x13, 0, x21,
                    x32, y23, x13, y31, x21, y12};
    for (int i = 0; i < 18; i++) b[i] = m[i] * inv;
}

// DKT B at the centroid (xi = eta = 1/3), Batoz 1980 via dkt.rs.
static void build_dkt_b(const double lc[3][2], double area, double b[27]) {
    double x1 = lc[0][0], y1 = lc[0][1], x2 = lc[1][0], y2 = lc[1][1], x3 = lc[2][0], y3 = lc[2][1];
    double x12 = x1 - x2, x23 = x2 - x3, x31 = x3 - x1;
    double y12 = y1 - y2, y23 = y2 - y3, y31 = y3 - y1;
    double l23 = x23 * x23 + y23 * y23, l31 = x31 * x31 + y31 * y31, l12 = x12 * x12 + y12 * y12;
    double a4 = -x23 / l23, a5 = -x31 / l31, a6 = -x12 / l12;
    double b4 = 0.75 * x23 * y23 / l23, b5 = 0.75 * x31 * y31 / l31, b6 = 0.75 * x12 * y12 / l12;
    double c4 = (0.25 * x23 * x23 - 0.5 * y23 * y23) / l23;
    double c5 = (0.25 * x31 * x31 - 0.5 * y31 * y31) / l31;
    double c6 = (0.25 * x12 * x12 - 0.5 * y12 * y12) / l12;
    double d4 = -y23 / l23, d5 = -y31 / l31, d6 = -y12 / l12;
    double e4 = (0.25 * y23 * y23 - 0.5 * x23 * x23) / l23;
    double e5 = (0.25 * y31 * y31 - 0.5 * x31 * x31) / l31;
    double e6 = (0.25 * y12 * y12 - 0.5 * x12 * x12) / l12;
    double xi = 1.0 / 3.0, eta = 1.0 / 3.0, zeta = 1.0 - xi - eta;
    double dndxi[6] = {1 - 4 * zeta, 4 * xi - 1, 0, 4 * eta, -4 * eta, 4 * (zeta - xi)};
    double dndeta[6] = {1 - 4 * zeta, 0, 4 * eta - 1, 4 * xi, 4 * (zeta - eta), -4 * xi};
    // eval_h for dN/dxi and dN/deta; DOF order per triple [w, tx, ty].
    double hx_xi[9], hy_xi[9], hx_eta[9], hy_eta[9];
    for (int pass = 0; pass < 2; pass++) {
        double *n = pass ? dndeta : dndxi;
        double *hx = pass ? hx_eta : hx_xi, *hy = pass ? hy_eta : hy_xi;
        hx[0] = 1.5 * (a6 * n[5] - a5 * n[4]); hx[1] = b5 * n[4] + b6 * n[5];
        hx[2] = n[0] - c5 * n[4] - c6 * n[5];
        hx[3] = 1.5 * (a4 * n[3] - a6 * n[5]); hx[4] = b6 * n[5] + b4 * n[3];
        hx[5] = n[1] - c6 * n[5] - c4 * n[3];
        hx[6] = 1.5 * (a5 * n[4] - a4 * n[3]); hx[7] = b4 * n[3] + b5 * n[4];
        hx[8] = n[2] - c4 * n[3] - c5 * n[4];
        hy[0] = 1.5 * (d6 * n[5] - d5 * n[4]); hy[1] = -n[0] + e5 * n[4] + e6 * n[5];
        hy[2] = -(b5 * n[4] + b6 * n[5]);
        hy[3] = 1.5 * (d4 * n[3] - d6 * n[5]); hy[4] = -n[1] + e6 * n[5] + e4 * n[3];
        hy[5] = -(b6 * n[5] + b4 * n[3]);
        hy[6] = 1.5 * (d5 * n[4] - d4 * n[3]); hy[7] = -n[2] + e4 * n[3] + e5 * n[4];
        hy[8] = -(b4 * n[3] + b5 * n[4]);
    }
    double inv2a = 1.0 / (2.0 * area);
    for (int j = 0; j < 9; j++) {
        b[j] = (y31 * hx_xi[j] + y12 * hx_eta[j]) * inv2a; // dHx/dx
        b[9 + j] = (x31 * hy_xi[j] + x12 * hy_eta[j]) * -inv2a; // dHy/dy
        b[18 + j] = ((y31 * hy_xi[j] + y12 * hy_eta[j]) - (x31 * hx_xi[j] + x12 * hx_eta[j])) * inv2a;
    }
}

// Greedy node-disjoint coloring of the split triangles.
static int paint_tris(const int conn[NTRI][3], int color_of[NTRI], int color_start[NTRI + 1],
                    int color_tris[NTRI]) {
    int ncolors = 0;
    for (int t = 0; t < NTRI; t++) color_of[t] = -1;
    for (int t = 0; t < NTRI; t++) {
        int used[64];
        for (int c = 0; c < 64; c++) used[c] = 0;
        for (int s = 0; s < t; s++) {
            int share = 0;
            for (int a = 0; a < 3 && !share; a++)
                for (int b = 0; b < 3 && !share; b++)
                    if (conn[t][a] == conn[s][b]) share = 1;
            if (share && color_of[s] < 64) used[color_of[s]] = 1;
        }
        int c = 0;
        while (c < 64 && used[c]) c++;
        if (c == 64) { fprintf(stderr, "proxy: color overflow\n"); exit(1); }
        color_of[t] = c;
        if (c >= ncolors) ncolors = c + 1;
    }
    int counts[64] = {0};
    for (int t = 0; t < NTRI; t++) counts[color_of[t]]++;
    color_start[0] = 0;
    for (int c = 0; c < ncolors; c++) color_start[c + 1] = color_start[c] + counts[c];
    int cursor[64];
    for (int c = 0; c < ncolors; c++) cursor[c] = color_start[c];
    for (int t = 0; t < NTRI; t++) color_tris[cursor[color_of[t]]++] = t;
    return ncolors;
}

// ---- Device explicit loop (1 block x 256 threads, persistent) ----

__device__ static void tri_force_one(const TriConst *tri, const double *u, double *f,
                                     const double *cm, const double *db) {
    double g[18];
    for (int k = 0; k < 3; k++)
        for (int j = 0; j < 6; j++) g[6 * k + j] = u[6 * tri->nodes[k] + j];
    // Global -> local (R^T per translation/rotation triple).
    double l[18];
    for (int k = 0; k < 3; k++)
        for (int q = 0; q < 2; q++)
            for (int a = 0; a < 3; a++)
                l[6 * k + 3 * q + a] = tri->rot[a] * g[6 * k + 3 * q] +
                                       tri->rot[3 + a] * g[6 * k + 3 * q + 1] +
                                       tri->rot[6 + a] * g[6 * k + 3 * q + 2];
    // CST membrane: e = Bc m, N = Cm e, fm = Bc^T N A.
    double m6[6] = {l[0], l[1], l[6], l[7], l[12], l[13]};
    double e[3] = {0, 0, 0};
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 6; c++) e[r] += tri->bc[r * 6 + c] * m6[c];
    double rn[3] = {0, 0, 0};
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) rn[r] += cm[r * 3 + c] * e[c];
    double fm[6] = {0, 0, 0, 0, 0, 0};
    for (int r = 0; r < 6; r++)
        for (int c = 0; c < 3; c++) fm[r] += tri->bc[c * 6 + r] * rn[c] * tri->area;
    // DKT bending: k = Bd b, M = Db k, fb = Bd^T M A.
    double b9[9] = {l[2], l[3], l[4], l[8], l[9], l[10], l[14], l[15], l[16]};
    double k[3] = {0, 0, 0};
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 9; c++) k[r] += tri->bd[r * 9 + c] * b9[c];
    double rm[3] = {0, 0, 0};
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) rm[r] += db[r * 3 + c] * k[c];
    double fb[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int r = 0; r < 9; r++)
        for (int c = 0; c < 3; c++) fb[r] += tri->bd[c * 9 + r] * rm[c] * tri->area;
    // Assemble local f18 (membrane + bending + drilling slots).
    double fl[18] = {0};
    const int mem[6] = {0, 1, 6, 7, 12, 13};
    const int ben[9] = {2, 3, 4, 8, 9, 10, 14, 15, 16};
    for (int i = 0; i < 6; i++) fl[mem[i]] = fm[i];
    for (int i = 0; i < 9; i++) fl[ben[i]] = fb[i];
    fl[5] = tri->drill * l[5];
    fl[11] = tri->drill * l[11];
    fl[17] = tri->drill * l[17];
    // Scatter R fl to the global force (caller guarantees disjointness).
    for (int k = 0; k < 3; k++)
        for (int q = 0; q < 2; q++)
            for (int a = 0; a < 3; a++)
                f[6 * tri->nodes[k] + 3 * q + a] += tri->rot[3 * a] * fl[6 * k + 3 * q] +
                                                    tri->rot[3 * a + 1] * fl[6 * k + 3 * q + 1] +
                                                    tri->rot[3 * a + 2] * fl[6 * k + 3 * q + 2];
}

__global__ void twisted_loop(const TriConst *tris, const int *color_start, const int *color_tris,
                             int ncolors, const double *mass, const int *pinned, const double *cm,
                             const double *db, double *u, double *v, double *f, double *hist,
                             int nstep) {
    int tid = (int)threadIdx.x;
    for (int d = tid; d < NDOF; d += (int)blockDim.x) {
        u[d] = 0.0;
        v[d] = 0.0;
        f[d] = 0.0;
    }
    __syncthreads();
    for (int step = 0; step < nstep; step++) {
        double t = step * DT;
        double ramp = (t < RAMP_T) ? t / RAMP_T : 1.0;
        double fext = CLOAD_F * ramp;
        for (int kick = 0; kick < 2; kick++) {
            for (int d = tid; d < NDOF; d += (int)blockDim.x) f[d] = 0.0;
            __syncthreads();
#ifndef EMPTY_FORCE
            for (int c = 0; c < ncolors; c++) {
                int begin = color_start[c], end = color_start[c + 1];
                for (int idx = begin + tid; idx < end; idx += (int)blockDim.x)
                    tri_force_one(&tris[color_tris[idx]], u, f, cm, db);
                __syncthreads();
            }
#else
            __syncthreads();
#endif
            for (int d = tid; d < NDOF; d += (int)blockDim.x) {
                if (pinned[d]) continue;
                double ext = (d == TIP_DOF) ? fext : 0.0;
                v[d] += 0.5 * DT * (ext - f[d]) / mass[d];
                if (kick == 0) u[d] += DT * v[d];
            }
            __syncthreads();
        }
        if (tid == 0) hist[step] = u[TIP_DOF];
    }
}

int main(int argc, char **argv) {
    int nstep = 300000; // 60 ms at DT
    if (argc > 1) nstep = atoi(argv[1]);
    if (nstep < 1) { fprintf(stderr, "proxy: nstep >= 1\n"); return 1; }

    // Split quads along 0-2 (native arm convention) and build constants.
    static int conn[NTRI][3];
    static TriConst tris[NTRI];
    double cm0 = YOUNGS * THICK / (1.0 - POISSON * POISSON);
    double db0 = YOUNGS * THICK * THICK * THICK / (12.0 * (1.0 - POISSON * POISSON));
    for (int q = 0; q < NQ; q++) {
        const int *quad = MESH_Q[q];
        const int split[2][3] = {{quad[0], quad[1], quad[2]}, {quad[0], quad[2], quad[3]}};
        for (int s = 0; s < 2; s++) {
            int t = 2 * q + s;
            for (int k = 0; k < 3; k++) {
                conn[t][k] = split[s][k];
                tris[t].nodes[k] = split[s][k];
            }
            const double *p0 = MESH_X[conn[t][0]];
            const double *p1 = MESH_X[conn[t][1]];
            const double *p2 = MESH_X[conn[t][2]];
            double lc[3][2];
            build_frame(p0, p1, p2, tris[t].rot, lc);
            double area2 =
                (lc[1][0] - lc[0][0]) * (lc[2][1] - lc[0][1]) - (lc[2][0] - lc[0][0]) * (lc[1][1] - lc[0][1]);
            tris[t].area = fabs(area2) / 2.0;
            if (!(tris[t].area > 0.0)) { fprintf(stderr, "proxy: zero-area tri %d\n", t); return 1; }
            build_cst_b(lc, tris[t].area, tris[t].bc);
            build_dkt_b(lc, tris[t].area, tris[t].bd);
            tris[t].drill = cm0 * tris[t].area * 1e-6;
        }
    }
    // Lumped masses (native explicit rule, translations + rotary).
    static double mass[NDOF] = {0};
    for (int t = 0; t < NTRI; t++) {
        double nodal = DENSITY * THICK * tris[t].area / 3.0;
        double rotary = nodal * THICK * THICK / 12.0;
        for (int k = 0; k < 3; k++)
            for (int a = 0; a < 3; a++) {
                mass[6 * tris[t].nodes[k] + a] += nodal;
                mass[6 * tris[t].nodes[k] + 3 + a] += rotary;
            }
    }
    for (int d = 0; d < NDOF; d++)
        if (!(mass[d] > 0.0)) { fprintf(stderr, "proxy: non-positive mass %d\n", d); return 1; }
    // Root deck nodes 1..7 (idx 0..6) pinned on all 6 DOFs.
    static int pinned[NDOF] = {0};
    for (int n = 0; n < 7; n++)
        for (int a = 0; a < 6; a++) pinned[6 * n + a] = 1;
    // Coloring.
    static int color_of[NTRI], color_start[NTRI + 1], color_tris[NTRI];
    int ncolors = paint_tris(conn, color_of, color_start, color_tris);
    // Constitutive matrices.
    double cm[9] = {cm0, cm0 * POISSON, 0, cm0 * POISSON, cm0, 0, 0, 0, cm0 * (1 - POISSON) / 2};
    double db[9] = {db0, db0 * POISSON, 0, db0 * POISSON, db0, 0, 0, 0, db0 * (1 - POISSON) / 2};

    TriConst *d_tris;
    int *d_cstart, *d_ctris, *d_pinned;
    double *d_mass, *d_cm, *d_db, *d_u, *d_v, *d_f, *d_hist;
    CHECK(cudaMalloc(&d_tris, sizeof(tris)));
    CHECK(cudaMalloc(&d_cstart, (ncolors + 1) * sizeof(int)));
    CHECK(cudaMalloc(&d_ctris, sizeof(color_tris)));
    CHECK(cudaMalloc(&d_pinned, sizeof(pinned)));
    CHECK(cudaMalloc(&d_mass, sizeof(mass)));
    CHECK(cudaMalloc(&d_cm, sizeof(cm)));
    CHECK(cudaMalloc(&d_db, sizeof(db)));
    CHECK(cudaMalloc(&d_u, NDOF * sizeof(double)));
    CHECK(cudaMalloc(&d_v, NDOF * sizeof(double)));
    CHECK(cudaMalloc(&d_f, NDOF * sizeof(double)));
    CHECK(cudaMalloc(&d_hist, (size_t)nstep * sizeof(double)));
    CHECK(cudaMemcpy(d_tris, tris, sizeof(tris), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_cstart, color_start, (ncolors + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_ctris, color_tris, sizeof(color_tris), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_pinned, pinned, sizeof(pinned), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_mass, mass, sizeof(mass), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_cm, cm, sizeof(cm), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_db, db, sizeof(db), cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));
    CHECK(cudaEventRecord(t0));
    twisted_loop<<<1, 256>>>(d_tris, d_cstart, d_ctris, ncolors, d_mass, d_pinned, d_cm, d_db, d_u,
                             d_v, d_f, d_hist, nstep);
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    float wall_ms = 0;
    CHECK(cudaEventElapsedTime(&wall_ms, t0, t1));
    CHECK(cudaGetLastError());

    double *hist = (double *)malloc((size_t)nstep * sizeof(double));
    double fhost[NDOF];
    CHECK(cudaMemcpy(hist, d_hist, (size_t)nstep * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(fhost, d_f, sizeof(fhost), cudaMemcpyDeviceToHost));
    double maxf = 0;
    for (int d = 0; d < NDOF; d++)
        if (fabs(fhost[d]) > maxf) maxf = fabs(fhost[d]);
    double sum = 0;
    long count = 0;
    for (int s = 0; s < nstep; s++) {
        double t = (s + 1) * DT;
        if (t >= HOLD_T) {
            sum += hist[s];
            count++;
        }
    }
    printf("twisted proxy: nstep=%d dt=%.3e colors=%d\n", nstep, DT, ncolors);
    printf("wall_ms=%.3f us_per_step=%.4f\n", wall_ms, 1000.0 * wall_ms / nstep);
    printf("tip_final=%.6e tip_holdmean=%s (t>=32ms, %ld samples)\n", hist[nstep - 1],
           count ? "" : "n/a (short run)", count);
    if (count) {
        double mean = sum / count;
        printf("tip_holdmean=%.6e native=%.6e reldiff=%.3e\n", mean, NATIVE_TIP_MEAN,
               fabs(mean - NATIVE_TIP_MEAN) / fabs(NATIVE_TIP_MEAN));
    }
    printf("max_abs_f=%.6e\n", maxf);
    free(hist);
    return 0;
}


