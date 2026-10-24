#ifndef TET_NODES
#define TET_NODES 4
#endif
#if TET_NODES == 10
#define TET_GEOMETRY 11
#else
#define TET_GEOMETRY 13
#endif
#if TET_NODES == 10
// Exact normalized P2 mass matrix from barycentric polynomial moments.
__device__ __constant__ double p2_mass[10][10] = {
    {36.0/2520.0, 6.0/2520.0, 6.0/2520.0, 6.0/2520.0, -24.0/2520.0, -24.0/2520.0, -24.0/2520.0, -36.0/2520.0, -36.0/2520.0, -36.0/2520.0},
    {6.0/2520.0, 36.0/2520.0, 6.0/2520.0, 6.0/2520.0, -24.0/2520.0, -36.0/2520.0, -36.0/2520.0, -24.0/2520.0, -24.0/2520.0, -36.0/2520.0},
    {6.0/2520.0, 6.0/2520.0, 36.0/2520.0, 6.0/2520.0, -36.0/2520.0, -24.0/2520.0, -36.0/2520.0, -24.0/2520.0, -36.0/2520.0, -24.0/2520.0},
    {6.0/2520.0, 6.0/2520.0, 6.0/2520.0, 36.0/2520.0, -36.0/2520.0, -36.0/2520.0, -24.0/2520.0, -36.0/2520.0, -24.0/2520.0, -24.0/2520.0},
    {-24.0/2520.0, -24.0/2520.0, -36.0/2520.0, -36.0/2520.0, 192.0/2520.0, 96.0/2520.0, 96.0/2520.0, 96.0/2520.0, 96.0/2520.0, 48.0/2520.0},
    {-24.0/2520.0, -36.0/2520.0, -24.0/2520.0, -36.0/2520.0, 96.0/2520.0, 192.0/2520.0, 96.0/2520.0, 96.0/2520.0, 48.0/2520.0, 96.0/2520.0},
    {-24.0/2520.0, -36.0/2520.0, -36.0/2520.0, -24.0/2520.0, 96.0/2520.0, 96.0/2520.0, 192.0/2520.0, 48.0/2520.0, 96.0/2520.0, 96.0/2520.0},
    {-36.0/2520.0, -24.0/2520.0, -24.0/2520.0, -36.0/2520.0, 96.0/2520.0, 96.0/2520.0, 48.0/2520.0, 192.0/2520.0, 96.0/2520.0, 96.0/2520.0},
    {-36.0/2520.0, -24.0/2520.0, -36.0/2520.0, -24.0/2520.0, 96.0/2520.0, 48.0/2520.0, 96.0/2520.0, 96.0/2520.0, 192.0/2520.0, 96.0/2520.0},
    {-36.0/2520.0, -36.0/2520.0, -24.0/2520.0, -24.0/2520.0, 48.0/2520.0, 96.0/2520.0, 96.0/2520.0, 96.0/2520.0, 96.0/2520.0, 192.0/2520.0},
};
__device__ __constant__ unsigned p2_edges[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
__device__ double p2_dot(unsigned i, unsigned j, const double* g) {
    // Packed upper triangle of the four barycentric-gradient Gram matrix.
    unsigned lo=i<j ? i : j, hi=i<j ? j : i;
    return g[1+lo*(7-lo)/2+hi];
}
// Twenty times the normalized integral of grad(N_i).grad(N_j).
// E[lambda_a lambda_b]=(1+delta_ab)/20. Expanding the sparse
// vertex/edge gradients eliminates twelve complex gradient accumulators.
__device__ double p2_stiffness(unsigned i, unsigned j, const double* g) {
    if (i < 4 && j < 4) return (i==j ? 12.0 : -4.0)*p2_dot(i,j,g);
    if (i >= 4 && j < 4) { unsigned swap=i; i=j; j=swap; }
    if (i < 4) {
        unsigned u=p2_edges[j-4][0], v=p2_edges[j-4][1];
        return (i==u ? 12.0 : -4.0)*p2_dot(i,v,g)
             + (i==v ? 12.0 : -4.0)*p2_dot(i,u,g);
    }
    unsigned l=p2_edges[i-4][0], r=p2_edges[i-4][1];
    unsigned u=p2_edges[j-4][0], v=p2_edges[j-4][1];
    return 16.0*((1.0+(l==u))*p2_dot(r,v,g)+(1.0+(l==v))*p2_dot(r,u,g)
               +(1.0+(r==u))*p2_dot(l,v,g)+(1.0+(r==v))*p2_dot(l,u,g));
}
#endif
// Affine P1/P2 tetrahedral mass/diffusion action. Node-owned accumulation
// avoids atomics and intermediate element vectors. Complex numbers are
// stored as adjacent real/imag doubles, matching num_complex::Complex64.
struct C64 { double re, im; };
__device__ C64 add(C64 a, C64 b) { return {a.re+b.re,a.im+b.im}; }
__device__ C64 scale(C64 a, double b) { return {a.re*b,a.im*b}; }
__device__ C64 mul(C64 a, C64 b) { return {a.re*b.re-a.im*b.im,a.re*b.im+a.im*b.re}; }
extern "C" __global__ void sonium_tet_c64(
    const unsigned int* nodes, const double* geometry,
    const unsigned int* offsets, const unsigned int* slots,
    const C64* coefficients, const C64* x, C64* y,
    unsigned int n, unsigned int total, unsigned int adjoint,
    const C64* b, const C64* inverse_diagonal, unsigned int mode, double omega,
    const unsigned int* dirichlet, double dr, double di, double mr, double mi) {
    unsigned long long thread = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (thread >= total) return;
    unsigned int tid = (unsigned int)thread;
    unsigned int node = tid % n;
    unsigned int base = tid - node;
    C64 output = {0.0,0.0};
    if (dirichlet[node]) {
        if (mode == 0 || mode == 3) y[tid] = x[tid];
        else {
            C64 residual = add(b[tid], scale(x[tid], -1.0));
            y[tid] = mode == 1 ? residual : add(x[tid], scale(mul(inverse_diagonal[node], residual), omega));
        }
        return;
    }
    // Lifting gathers prescribed entries; other actions gather free entries.
    C64 self_value = mode == 3 ? C64{0.0,0.0} : x[tid];
    for (unsigned int q=offsets[node]; q<offsets[node+1]; ++q) {
        unsigned int slot=slots[q], e=slot/TET_NODES, local=slot%TET_NODES;
        // 64-bit offsets avoid overflowing element geometry offsets.
        const double* g=geometry+(unsigned long long)e*TET_GEOMETRY;
        const unsigned int* ids=nodes+(unsigned long long)e*TET_NODES;
#if TET_NODES == 10
        C64 stiffness={0.0,0.0}, mass_action={0.0,0.0};
        for (unsigned j=0;j<10;++j) {
            bool selected=(dirichlet[ids[j]] != 0)==(mode==3);
            C64 value=selected ? x[base+ids[j]] : C64{0.0,0.0};
            mass_action=add(mass_action,scale(value,p2_mass[local][j]));
            stiffness=add(stiffness,scale(value,p2_stiffness(local,j,g)));
        }
        C64 diffusion=mul(coefficients[2*(unsigned long long)e],C64{dr,di}), mass=mul(coefficients[2*(unsigned long long)e+1],C64{mr,mi});
        if(adjoint) {diffusion.im=-diffusion.im;mass.im=-mass.im;}
        output=add(output,add(mul(diffusion,scale(stiffness,g[0]/20.0)),mul(mass,scale(mass_action,g[0]))));
#else
        C64 sum={0.0,0.0}, grad[3]={{0.0,0.0},{0.0,0.0},{0.0,0.0}};
        for (unsigned int j=0;j<4;++j) {
            bool selected = (dirichlet[ids[j]] != 0) == (mode == 3);
            C64 value=selected ? x[base+ids[j]] : C64{0.0,0.0};
            sum=add(sum,value);
            for (unsigned int d=0;d<3;++d) grad[d]=add(grad[d],scale(value,g[1+3*j+d]));
        }
        C64 stiffness={0.0,0.0};
        for (unsigned int d=0;d<3;++d) stiffness=add(stiffness,scale(grad[d],g[1+3*local+d]));
        C64 diffusion=mul(coefficients[2*(unsigned long long)e],C64{dr,di}), mass=mul(coefficients[2*(unsigned long long)e+1],C64{mr,mi});
        if (adjoint) { diffusion.im=-diffusion.im; mass.im=-mass.im; }
        // Consistent tetrahedral P1 mass: M_ij = V*(1+delta_ij)/20.
        output=add(output,add(mul(diffusion,scale(stiffness,g[0])),mul(mass,scale(add(sum,self_value),g[0]/20.0))));
#endif
    }
    // mode 0: constrained action; 1: residual; 2: weighted Jacobi; 3: RHS lifting.
    // Inputs remain immutable: neighbors always read the previous iterate.
    if (mode != 0) {
        output=add(b[tid],scale(output,-1.0));
        if (mode == 2) output=add(x[tid],scale(mul(inverse_diagonal[node],output),omega));
    }
    y[tid]=output;
}

// One owner per CSR entry: deterministic element order, no atomics.
extern "C" __global__ void sonium_tet_csr_c64(
    const double* geometry, const C64* coefficients, const unsigned* dirichlet,
    const unsigned* rows, const int* columns, const unsigned* offsets,
    const unsigned* contributions, C64* values, unsigned nnz,
    double dr, double di, double mr, double mi) {
    unsigned long long thread=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(thread>=nnz) return;
    unsigned entry=(unsigned)thread, row=rows[entry], col=(unsigned)columns[entry];
    if(dirichlet[row] || dirichlet[col]) {values[entry]={row==col ? 1.0 : 0.0,0.0};return;}
    C64 value={0.0,0.0};
    for(unsigned p=offsets[entry];p<offsets[entry+1];++p) {
        unsigned slot=contributions[p], e=slot/(TET_NODES*TET_NODES), i=slot/TET_NODES%TET_NODES, j=slot%TET_NODES;
        const double* g=geometry+(unsigned long long)e*TET_GEOMETRY;
        double stiffness=0.0, mass;
#if TET_NODES == 10
        stiffness=p2_stiffness(i,j,g)/20.0;
        mass=p2_mass[i][j];
#else
        for(unsigned d=0;d<3;++d) stiffness+=g[1+3*i+d]*g[1+3*j+d];
        mass=i==j ? 0.1 : 0.05;
#endif
        C64 diffusion=mul(coefficients[2*(unsigned long long)e],C64{dr,di});
        C64 reaction=mul(coefficients[2*(unsigned long long)e+1],C64{mr,mi});
        value=add(value,add(scale(diffusion,g[0]*stiffness),scale(reaction,g[0]*mass)));
    }
    values[entry]=value;
}
