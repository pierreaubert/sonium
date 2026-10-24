// Hex partial assembly: four tensor contractions, quadrature physics, transpose.
struct C64 {double re,im;};
__device__ C64 add(C64 a,C64 b){return {a.re+b.re,a.im+b.im};}
__device__ C64 scale(C64 a,double b){return {a.re*b,a.im*b};}
__device__ C64 mul(C64 a,C64 b){return {a.re*b.re-a.im*b.im,a.re*b.im+a.im*b.re};}
// Absolute material updates from immutable weighted reference coefficients.
extern "C" __global__ void sonium_hex_material_c64(
    const C64* reference,const C64* diffusion,const C64* mass,C64* data,unsigned count) {
    unsigned point=blockIdx.x*blockDim.x+threadIdx.x;
    if(point>=count)return;
    unsigned long long offset=(unsigned long long)point*7;
    data[offset]=mul(reference[offset],mass[point]);
    C64 d=diffusion[point];
    for(unsigned i=1;i<7;++i)data[offset+i]=mul(reference[offset+i],d);
}
template<bool SCALED>
__device__ __forceinline__ void hex_partial(const unsigned* nodes,const double* basis,
    const double* derivative,const C64* data,const C64* x,C64* element,
    unsigned n,unsigned q,unsigned dimension,unsigned elements,unsigned adjoint,const unsigned* constrained,unsigned lifting,
    double diffusion_re,double diffusion_im,double mass_re,double mass_im) {
    // Maximum 8^3 tensor entries, 48 KiB total shared storage.
    __shared__ C64 a[512],b[512],field[4][512];
    unsigned job=blockIdx.x,e=job%elements,column=job/elements;
    unsigned local=n*n*n,quadrature=q*q*q;
    for(unsigned c=0;c<4;++c){
        const double* tx=c==1?derivative:basis;
        const double* ty=c==2?derivative:basis;
        const double* tz=c==3?derivative:basis;
        for(unsigned v=threadIdx.x;v<q*n*n;v+=blockDim.x){
            unsigned i=v%q,j=v/q%n,k=v/(q*n);C64 sum={};
            for(unsigned r=0;r<n;++r){
                unsigned node=nodes[(unsigned long long)e*local+r+n*(j+n*k)];
                if((constrained[node]!=0)==(lifting!=0))
                    sum=add(sum,scale(x[(unsigned long long)column*dimension+node],tx[i*n+r]));
            }
            a[v]=sum;
        }
        __syncthreads();
        for(unsigned v=threadIdx.x;v<q*q*n;v+=blockDim.x){
            unsigned i=v%q,j=v/q%q,k=v/(q*q);C64 sum={};
            for(unsigned r=0;r<n;++r)sum=add(sum,scale(a[i+q*(r+n*k)],ty[j*n+r]));
            b[v]=sum;
        }
        __syncthreads();
        for(unsigned v=threadIdx.x;v<quadrature;v+=blockDim.x){
            unsigned i=v%q,j=v/q%q,k=v/(q*q);C64 sum={};
            for(unsigned r=0;r<n;++r)sum=add(sum,scale(b[i+q*(j+q*r)],tz[k*n+r]));
            field[c][v]=sum;
        }
        __syncthreads();
    }
    for(unsigned v=threadIdx.x;v<quadrature;v+=blockDim.x){
        C64 d[7];for(unsigned i=0;i<7;++i){
            d[i]=data[((unsigned long long)e*quadrature+v)*7+i];
            if(SCALED){
                C64 factor=i==0?C64{mass_re,mass_im}:C64{diffusion_re,diffusion_im};
                if(factor.re!=1.0 || factor.im!=0.0)d[i]=mul(d[i],factor);
            }
            if(adjoint)d[i].im=-d[i].im;
        }
        C64 u=field[0][v],gx=field[1][v],gy=field[2][v],gz=field[3][v];
        field[0][v]=mul(d[0],u);
        field[1][v]=add(add(mul(d[1],gx),mul(d[4],gy)),mul(d[5],gz));
        field[2][v]=add(add(mul(d[4],gx),mul(d[2],gy)),mul(d[6],gz));
        field[3][v]=add(add(mul(d[5],gx),mul(d[6],gy)),mul(d[3],gz));
    }
    __syncthreads();
    for(unsigned c=0;c<4;++c){
        const double* tx=c==1?derivative:basis;
        const double* ty=c==2?derivative:basis;
        const double* tz=c==3?derivative:basis;
        for(unsigned v=threadIdx.x;v<q*q*n;v+=blockDim.x){
            unsigned i=v%q,j=v/q%q,k=v/(q*q);C64 sum={};
            for(unsigned r=0;r<q;++r)sum=add(sum,scale(field[c][i+q*(j+q*r)],tz[r*n+k]));
            a[v]=sum;
        }
        __syncthreads();
        for(unsigned v=threadIdx.x;v<q*n*n;v+=blockDim.x){
            unsigned i=v%q,j=v/q%n,k=v/(q*n);C64 sum={};
            for(unsigned r=0;r<q;++r)sum=add(sum,scale(a[i+q*(r+q*k)],ty[r*n+j]));
            b[v]=sum;
        }
        __syncthreads();
        for(unsigned v=threadIdx.x;v<local;v+=blockDim.x){
            unsigned i=v%n,j=v/n%n,k=v/(n*n);C64 sum={};
            for(unsigned r=0;r<q;++r)sum=add(sum,scale(b[r+q*(j+n*k)],tx[r*n+i]));
            unsigned long long slot=(unsigned long long)job*local+v;
            element[slot]=c==0?sum:add(element[slot],sum);
        }
        __syncthreads();
    }
}
// Keep the original argument list and arithmetic for ordinary fixed operators.
extern "C" __global__ void sonium_hex_partial_c64(const unsigned* nodes,const double* basis,
    const double* derivative,const C64* data,const C64* x,C64* element,
    unsigned n,unsigned q,unsigned dimension,unsigned elements,unsigned adjoint,const unsigned* constrained,unsigned lifting){
    hex_partial<false>(nodes,basis,derivative,data,x,element,n,q,dimension,elements,adjoint,constrained,lifting,1.0,0.0,1.0,0.0);
}
extern "C" __global__ void sonium_hex_partial_scaled_c64(const unsigned* nodes,const double* basis,
    const double* derivative,const C64* data,const C64* x,C64* element,
    unsigned n,unsigned q,unsigned dimension,unsigned elements,unsigned adjoint,const unsigned* constrained,unsigned lifting,
    double diffusion_re,double diffusion_im,double mass_re,double mass_im){
    hex_partial<true>(nodes,basis,derivative,data,x,element,n,q,dimension,elements,adjoint,constrained,lifting,diffusion_re,diffusion_im,mass_re,mass_im);
}
extern "C" __global__ void sonium_hex_scatter_c64(const unsigned* offsets,const unsigned* slots,
    const C64* element,C64* y,unsigned dimension,unsigned element_size,unsigned total,const unsigned* constrained,const C64* x,const C64* rhs,unsigned mode,const C64* inverse,double omega){
    unsigned long long thread=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(thread>=total)return;
    unsigned node=thread%dimension,column=thread/dimension;C64 sum={};
    for(unsigned j=offsets[node];j<offsets[node+1];++j)sum=add(sum,element[(unsigned long long)column*element_size+slots[j]]);
    if(mode==1){
        y[thread]=constrained[node]?x[thread]:add(rhs[thread],scale(sum,-1.0));
        return;
    }
    C64 ax=constrained[node]?x[thread]:sum;
    if(mode==0)y[thread]=ax;
    else if(mode==2)y[thread]=add(rhs[thread],scale(ax,-1.0));
    else if(omega==0.0)y[thread]=x[thread];
    else y[thread]=add(x[thread],scale(mul(inverse[node],add(rhs[thread],scale(ax,-1.0))),omega));
}
