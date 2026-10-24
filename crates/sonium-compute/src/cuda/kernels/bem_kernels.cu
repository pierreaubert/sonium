struct Complex32 {
    float re;
    float im;
};

struct SoniumBemRegularPairIn {
    float source_point[3];
    float _pad0;
    float source_normal[3];
    float _pad1;
    float v0[3];
    float _pad2;
    float v1[3];
    float _pad3;
    float v2[3];
    float _pad4;
};

struct SoniumBemRegularPairOut {
    Complex32 g;
    Complex32 h;
    Complex32 ht;
    Complex32 e;
};

__device__ __forceinline__ float3 load3(const float v[3]) {
    return make_float3(v[0], v[1], v[2]);
}

__device__ __forceinline__ Complex32 cmul(Complex32 a, Complex32 b) {
    return Complex32{a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

extern "C" __global__ void sonium_bem_regular_pair_c32(
    const SoniumBemRegularPairIn* pairs,
    SoniumBemRegularPairOut* out,
    float k,
    float harmonic_factor,
    unsigned long long n_pairs
) {
    const unsigned long long gid =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= n_pairs) {
        return;
    }

    const SoniumBemRegularPairIn p = pairs[gid];
    const float3 src = load3(p.source_point);
    const float3 n_x = load3(p.source_normal);
    const float3 v0 = load3(p.v0);
    const float3 v1 = load3(p.v1);
    const float3 v2 = load3(p.v2);

    const float k2 = k * k;
    const float wavruim = harmonic_factor * k;

    const float3 e1 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
    const float3 e2 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
    const float3 raw_n = make_float3(
        e1.y * e2.z - e1.z * e2.y,
        e1.z * e2.x - e1.x * e2.z,
        e1.x * e2.y - e1.y * e2.x
    );
    const float jac = sqrtf(raw_n.x * raw_n.x + raw_n.y * raw_n.y + raw_n.z * raw_n.z);
    const float inv_jac = (jac > 1.0e-15f) ? (1.0f / jac) : 0.0f;
    const float3 n_y = make_float3(raw_n.x * inv_jac, raw_n.y * inv_jac, raw_n.z * inv_jac);

    Complex32 acc_g = {0.0f, 0.0f};
    Complex32 acc_h = {0.0f, 0.0f};
    Complex32 acc_ht = {0.0f, 0.0f};
    Complex32 acc_e = {0.0f, 0.0f};

    const float xis[4] = {0.333333333333333f, 0.6f, 0.2f, 0.2f};
    const float etas[4] = {0.333333333333333f, 0.2f, 0.6f, 0.2f};
    const float weights[4] = {-0.281250f, 0.260416666666667f, 0.260416666666667f, 0.260416666666667f};

    for (unsigned int q = 0; q < 4; ++q) {
        const float xi = xis[q];
        const float eta = etas[q];
        const float n0 = 1.0f - xi - eta;
        const float3 crd = make_float3(
            n0 * v0.x + xi * v1.x + eta * v2.x,
            n0 * v0.y + xi * v1.y + eta * v2.y,
            n0 * v0.z + xi * v1.z + eta * v2.z
        );
        const float wga = weights[q] * jac;

        const float3 diff = make_float3(crd.x - src.x, crd.y - src.y, crd.z - src.z);
        const float r = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
        if (r < 1.0e-15f) {
            continue;
        }
        const float inv_r = 1.0f / r;
        const float3 ur = make_float3(diff.x * inv_r, diff.y * inv_r, diff.z * inv_r);

        const float re1 = wavruim * r;
        const float re2 = wga / (12.56637061435917295385f * r);
        Complex32 g = {cosf(re1) * re2, sinf(re1) * re2};
        Complex32 z1 = {-inv_r, wavruim};
        Complex32 hb = cmul(g, z1);

        const float r1h = ur.x * n_y.x + ur.y * n_y.y + ur.z * n_y.z;
        const float r2h = -(ur.x * n_x.x + ur.y * n_x.y + ur.z * n_x.z);
        Complex32 h = {hb.re * r1h, hb.im * r1h};
        Complex32 ht = {hb.re * r2h, hb.im * r2h};

        const float rq = r1h * r2h;
        const float nx_ny = n_x.x * n_y.x + n_x.y * n_y.y + n_x.z * n_y.z;
        const float dq = r * r;
        Complex32 e_fac = {
            (3.0f / dq - k2) * rq + nx_ny / dq,
            -wavruim / r * (3.0f * rq + nx_ny)
        };
        Complex32 e = cmul(g, e_fac);

        acc_g.re += g.re;
        acc_g.im += g.im;
        acc_h.re += h.re;
        acc_h.im += h.im;
        acc_ht.re += ht.re;
        acc_ht.im += ht.im;
        acc_e.re += e.re;
        acc_e.im += e.im;
    }

    SoniumBemRegularPairOut result;
    result.g = acc_g;
    result.h = acc_h;
    result.ht = acc_ht;
    result.e = acc_e;
    out[gid] = result;
}

// The f64 ABI is intentionally kept explicit.  It is the reference-precision
// path used by BEM assembly; no f32 geometry or arithmetic is introduced
// before the host-side result is published.
struct Complex64 {
    double re;
    double im;
};

struct SoniumBemRegularPairInF64 {
    double source_point[3];
    double _pad0;
    double source_normal[3];
    double _pad1;
    double v0[3];
    double _pad2;
    double v1[3];
    double _pad3;
    double v2[3];
    double _pad4;
};

struct SoniumBemRegularPairOutF64 {
    Complex64 g;
    Complex64 h;
    Complex64 ht;
    Complex64 e;
};

// Match sonium-bem's order-2 triangle rule for the far-panel c64 path.
// `regular_integration` selects this 4-point rule when the source-to-panel
// ratio is at least QUASI_SINGULAR_THRESHOLD. Quasi-singular and singular
// charts use the separate quadrature-pair kernel, so keeping this fixed rule
// here makes the native value and analytic d/dk channels compare like-for-like
// without a host integration fallback.
__device__ __constant__ static const double SONIUM_TRI4_XI[4] = {
    0.333333333333333, 0.6, 0.2, 0.2
};
__device__ __constant__ static const double SONIUM_TRI4_ETA[4] = {
    0.333333333333333, 0.2, 0.6, 0.2
};
__device__ __constant__ static const double SONIUM_TRI4_WEIGHTS[4] = {
    -0.281250, 0.260416666666667, 0.260416666666667, 0.260416666666667
};

__device__ __forceinline__ double3 load3_f64(const double v[3]) {
    return make_double3(v[0], v[1], v[2]);
}

__device__ __forceinline__ Complex64 cmul_f64(Complex64 a, Complex64 b) {
    return Complex64{a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

extern "C" __global__ void sonium_bem_regular_pair_c64(
    const SoniumBemRegularPairInF64* pairs,
    SoniumBemRegularPairOutF64* out,
    double k,
    double harmonic_factor,
    unsigned long long n_pairs,
    unsigned long long pair_offset,
    unsigned long long pair_stride,
    unsigned long long out_offset,
    unsigned long long out_stride
) {
    const unsigned long long gid =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= n_pairs) {
        return;
    }

    const SoniumBemRegularPairInF64 p = pairs[pair_offset + gid * pair_stride];
    const double3 src = load3_f64(p.source_point);
    const double3 n_x = load3_f64(p.source_normal);
    const double3 v0 = load3_f64(p.v0);
    const double3 v1 = load3_f64(p.v1);
    const double3 v2 = load3_f64(p.v2);
    const double k2 = k * k;
    const double wavruim = harmonic_factor * k;

    const double3 e1 = make_double3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
    const double3 e2 = make_double3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
    const double3 raw_n = make_double3(
        e1.y * e2.z - e1.z * e2.y,
        e1.z * e2.x - e1.x * e2.z,
        e1.x * e2.y - e1.y * e2.x
    );
    const double jac = sqrt(raw_n.x * raw_n.x + raw_n.y * raw_n.y + raw_n.z * raw_n.z);
    const double inv_jac = (jac > 1.0e-15) ? (1.0 / jac) : 0.0;
    const double3 n_y = make_double3(raw_n.x * inv_jac, raw_n.y * inv_jac, raw_n.z * inv_jac);

    Complex64 acc_g = {0.0, 0.0};
    Complex64 acc_h = {0.0, 0.0};
    Complex64 acc_ht = {0.0, 0.0};
    Complex64 acc_e = {0.0, 0.0};
    for (unsigned int q = 0; q < 4; ++q) {
        const double xi = SONIUM_TRI4_XI[q];
        const double eta = SONIUM_TRI4_ETA[q];
        const double n0 = 1.0 - xi - eta;
        const double3 crd = make_double3(
            n0 * v0.x + xi * v1.x + eta * v2.x,
            n0 * v0.y + xi * v1.y + eta * v2.y,
            n0 * v0.z + xi * v1.z + eta * v2.z
        );
        const double wga = SONIUM_TRI4_WEIGHTS[q] * jac;
        const double3 diff = make_double3(crd.x - src.x, crd.y - src.y, crd.z - src.z);
        const double r = sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
        if (r < 1.0e-15) {
            continue;
        }
        const double inv_r = 1.0 / r;
        const double3 ur = make_double3(diff.x * inv_r, diff.y * inv_r, diff.z * inv_r);
        const double re1 = wavruim * r;
        const double re2 = wga / (12.56637061435917295385 * r);
        Complex64 g = {cos(re1) * re2, sin(re1) * re2};
        Complex64 z1 = {-inv_r, wavruim};
        Complex64 hb = cmul_f64(g, z1);
        const double r1h = ur.x * n_y.x + ur.y * n_y.y + ur.z * n_y.z;
        const double r2h = -(ur.x * n_x.x + ur.y * n_x.y + ur.z * n_x.z);
        Complex64 h = {hb.re * r1h, hb.im * r1h};
        Complex64 ht = {hb.re * r2h, hb.im * r2h};
        const double rq = r1h * r2h;
        const double nx_ny = n_x.x * n_y.x + n_x.y * n_y.y + n_x.z * n_y.z;
        const double dq = r * r;
        Complex64 e_fac = {
            (3.0 / dq - k2) * rq + nx_ny / dq,
            -wavruim / r * (3.0 * rq + nx_ny)
        };
        Complex64 e = cmul_f64(g, e_fac);
        acc_g.re += g.re;
        acc_g.im += g.im;
        acc_h.re += h.re;
        acc_h.im += h.im;
        acc_ht.re += ht.re;
        acc_ht.im += ht.im;
        acc_e.re += e.re;
        acc_e.im += e.im;
    }
    SoniumBemRegularPairOutF64 result;
    result.g = acc_g;
    result.h = acc_h;
    result.ht = acc_ht;
    result.e = acc_e;
    out[out_offset + gid * out_stride] = result;
}

// Static real-valued Laplace pair kernel.  This is a separate scalar ABI,
// not a Complex64 projection: every accumulator and published channel is a
// CUDA double.  The production CUDA adapter uses it only for its explicit
// zero-frequency/Laplace seam.
struct SoniumBemRegularPairOutRealF64 {
    double g;
    double h;
    double ht;
    double e;
};

extern "C" __global__ void sonium_bem_regular_pair_real_f64(
    const SoniumBemRegularPairInF64* pairs,
    SoniumBemRegularPairOutRealF64* out,
    unsigned long long n_pairs
) {
    const unsigned long long gid =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= n_pairs) {
        return;
    }

    const SoniumBemRegularPairInF64 p = pairs[gid];
    const double3 src = load3_f64(p.source_point);
    const double3 n_x = load3_f64(p.source_normal);
    const double3 v0 = load3_f64(p.v0);
    const double3 v1 = load3_f64(p.v1);
    const double3 v2 = load3_f64(p.v2);
    const double3 e1 = make_double3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
    const double3 e2 = make_double3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
    const double3 raw_n = make_double3(
        e1.y * e2.z - e1.z * e2.y,
        e1.z * e2.x - e1.x * e2.z,
        e1.x * e2.y - e1.y * e2.x
    );
    const double jac = sqrt(raw_n.x * raw_n.x + raw_n.y * raw_n.y + raw_n.z * raw_n.z);
    const double inv_jac = (jac > 1.0e-15) ? (1.0 / jac) : 0.0;
    const double3 n_y = make_double3(raw_n.x * inv_jac, raw_n.y * inv_jac, raw_n.z * inv_jac);

    double acc_g = 0.0;
    double acc_h = 0.0;
    double acc_ht = 0.0;
    double acc_e = 0.0;
    for (unsigned int q = 0; q < 4; ++q) {
        const double xi = SONIUM_TRI4_XI[q];
        const double eta = SONIUM_TRI4_ETA[q];
        const double n0 = 1.0 - xi - eta;
        const double3 crd = make_double3(
            n0 * v0.x + xi * v1.x + eta * v2.x,
            n0 * v0.y + xi * v1.y + eta * v2.y,
            n0 * v0.z + xi * v1.z + eta * v2.z
        );
        const double wga = SONIUM_TRI4_WEIGHTS[q] * jac;
        const double3 diff = make_double3(crd.x - src.x, crd.y - src.y, crd.z - src.z);
        const double r = sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
        if (r < 1.0e-15) {
            continue;
        }
        const double inv_r = 1.0 / r;
        const double3 ur = make_double3(diff.x * inv_r, diff.y * inv_r, diff.z * inv_r);
        const double green = wga / (12.56637061435917295385 * r);
        const double r1h = ur.x * n_y.x + ur.y * n_y.y + ur.z * n_y.z;
        const double r2h = -(ur.x * n_x.x + ur.y * n_x.y + ur.z * n_x.z);
        const double derivative = -inv_r;
        const double rq = r1h * r2h;
        const double nx_ny = n_x.x * n_y.x + n_x.y * n_y.y + n_x.z * n_y.z;
        const double inv_r2 = inv_r * inv_r;
        acc_g += green;
        acc_h += green * derivative * r1h;
        acc_ht += green * derivative * r2h;
        acc_e += green * (3.0 * inv_r2 * rq + inv_r2 * nx_ny);
    }
    out[gid] = SoniumBemRegularPairOutRealF64{acc_g, acc_h, acc_ht, acc_e};
}

// Analytic d/dk of the four constant-panel pair channels.  The geometry is
// identical to `sonium_bem_regular_pair_c64`; only the phase and the explicit
// k-dependent Helmholtz factors are differentiated on device.  Publishing
// the derivative through the existing output ABI keeps the host-side TBEM
// assembly layout unchanged while avoiding a CPU finite-difference fallback.
extern "C" __global__ void sonium_bem_regular_pair_c64_derivative(
    const SoniumBemRegularPairInF64* pairs,
    SoniumBemRegularPairOutF64* out,
    double k,
    double harmonic_factor,
    unsigned long long n_pairs,
    unsigned long long pair_offset,
    unsigned long long pair_stride,
    unsigned long long out_offset,
    unsigned long long out_stride
) {
    const unsigned long long gid =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= n_pairs) {
        return;
    }

    const SoniumBemRegularPairInF64 p = pairs[pair_offset + gid * pair_stride];
    const double3 src = load3_f64(p.source_point);
    const double3 n_x = load3_f64(p.source_normal);
    const double3 v0 = load3_f64(p.v0);
    const double3 v1 = load3_f64(p.v1);
    const double3 v2 = load3_f64(p.v2);
    const double k2 = k * k;
    const double wavruim = harmonic_factor * k;

    const double3 e1 = make_double3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
    const double3 e2 = make_double3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
    const double3 raw_n = make_double3(
        e1.y * e2.z - e1.z * e2.y,
        e1.z * e2.x - e1.x * e2.z,
        e1.x * e2.y - e1.y * e2.x
    );
    const double jac = sqrt(raw_n.x * raw_n.x + raw_n.y * raw_n.y + raw_n.z * raw_n.z);
    const double inv_jac = (jac > 1.0e-15) ? (1.0 / jac) : 0.0;
    const double3 n_y = make_double3(raw_n.x * inv_jac, raw_n.y * inv_jac, raw_n.z * inv_jac);

    Complex64 acc_g = {0.0, 0.0};
    Complex64 acc_h = {0.0, 0.0};
    Complex64 acc_ht = {0.0, 0.0};
    Complex64 acc_e = {0.0, 0.0};
    for (unsigned int q = 0; q < 4; ++q) {
        const double xi = SONIUM_TRI4_XI[q];
        const double eta = SONIUM_TRI4_ETA[q];
        const double n0 = 1.0 - xi - eta;
        const double3 crd = make_double3(
            n0 * v0.x + xi * v1.x + eta * v2.x,
            n0 * v0.y + xi * v1.y + eta * v2.y,
            n0 * v0.z + xi * v1.z + eta * v2.z
        );
        const double wga = SONIUM_TRI4_WEIGHTS[q] * jac;
        const double3 diff = make_double3(crd.x - src.x, crd.y - src.y, crd.z - src.z);
        const double r = sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
        if (r < 1.0e-15) {
            continue;
        }
        const double inv_r = 1.0 / r;
        const double3 ur = make_double3(diff.x * inv_r, diff.y * inv_r, diff.z * inv_r);
        const double phase = wavruim * r;
        const double green_scale = wga / (12.56637061435917295385 * r);
        const Complex64 g = {cos(phase) * green_scale, sin(phase) * green_scale};
        const Complex64 dgdk = cmul_f64(g, Complex64{0.0, harmonic_factor * r});
        const Complex64 z1 = {-inv_r, wavruim};
        const Complex64 dz1dk = {0.0, harmonic_factor};
        const Complex64 dhbdk = {
            cmul_f64(dgdk, z1).re + cmul_f64(g, dz1dk).re,
            cmul_f64(dgdk, z1).im + cmul_f64(g, dz1dk).im
        };
        const double r1h = ur.x * n_y.x + ur.y * n_y.y + ur.z * n_y.z;
        const double r2h = -(ur.x * n_x.x + ur.y * n_x.y + ur.z * n_x.z);
        const Complex64 dhdk = {dhbdk.re * r1h, dhbdk.im * r1h};
        const Complex64 dhtdk = {dhbdk.re * r2h, dhbdk.im * r2h};
        const double rq = r1h * r2h;
        const double nx_ny = n_x.x * n_y.x + n_x.y * n_y.y + n_x.z * n_y.z;
        const double dq = r * r;
        const Complex64 e_fac = {
            (3.0 / dq - k2) * rq + nx_ny / dq,
            -wavruim / r * (3.0 * rq + nx_ny)
        };
        const Complex64 de_facdk = {
            -2.0 * k * rq,
            -harmonic_factor / r * (3.0 * rq + nx_ny)
        };
        const Complex64 dedk = {
            cmul_f64(dgdk, e_fac).re + cmul_f64(g, de_facdk).re,
            cmul_f64(dgdk, e_fac).im + cmul_f64(g, de_facdk).im
        };
        acc_g.re += dgdk.re;
        acc_g.im += dgdk.im;
        acc_h.re += dhdk.re;
        acc_h.im += dhdk.im;
        acc_ht.re += dhtdk.re;
        acc_ht.im += dhtdk.im;
        acc_e.re += dedk.re;
        acc_e.im += dedk.im;
    }
    out[out_offset + gid * out_stride] = SoniumBemRegularPairOutF64{acc_g, acc_h, acc_ht, acc_e};
}

// Constant-basis exterior field evaluation.  This uses the same order-3
// seven-point triangle rule and Kirchhoff--Helmholtz signs as
// sonium-bem's CPU reference.  Geometry and Cauchy traces are uploaded once
// by CudaBemFieldOperatorC64; only evaluation points and the result vary per
// device action.
struct SoniumBemFieldElementC64 {
    double v0[3];
    double v1[3];
    double v2[3];
    Complex64 pressure;
    Complex64 velocity;
};

struct SoniumBemFieldPointC64 {
    double position[3];
};

__device__ __forceinline__ Complex64 cadd_f64(Complex64 a, Complex64 b) {
    return Complex64{a.re + b.re, a.im + b.im};
}

__device__ __forceinline__ Complex64 cscale_f64(Complex64 a, double factor) {
    return Complex64{a.re * factor, a.im * factor};
}

// Analytic derivative of the regular constant-panel pair channels with
// respect to a source/collocation point coordinate.  The field-panel
// geometry, Jacobian, normals, and quadrature rule remain fixed.  This is a
// genuine device-side geometry-translation derivative and is intentionally a
// separate entry point from the d/dk kernel; callers must not label a CPU
// finite difference as CUDA work.
__device__ __forceinline__ void sonium_bem_regular_pair_c64_source_derivative_impl(
    const SoniumBemRegularPairInF64* pairs,
    SoniumBemRegularPairOutF64* out,
    double k,
    double harmonic_factor,
    unsigned long long n_pairs,
    unsigned long long pair_offset,
    unsigned long long pair_stride,
    unsigned long long out_offset,
    unsigned long long out_stride,
    unsigned int source_axis_index,
    unsigned long long gid
) {
    const SoniumBemRegularPairInF64 p = pairs[pair_offset + gid * pair_stride];
    const double3 source = load3_f64(p.source_point);
    const double3 source_normal = load3_f64(p.source_normal);
    const double3 v0 = load3_f64(p.v0);
    const double3 v1 = load3_f64(p.v1);
    const double3 v2 = load3_f64(p.v2);
    const double3 edge_1 = make_double3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
    const double3 edge_2 = make_double3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
    const double3 raw_normal = make_double3(
        edge_1.y * edge_2.z - edge_1.z * edge_2.y,
        edge_1.z * edge_2.x - edge_1.x * edge_2.z,
        edge_1.x * edge_2.y - edge_1.y * edge_2.x
    );
    const double jacobian = sqrt(
        raw_normal.x * raw_normal.x +
        raw_normal.y * raw_normal.y +
        raw_normal.z * raw_normal.z
    );
    const double inv_jacobian = (jacobian > 1.0e-15) ? (1.0 / jacobian) : 0.0;
    const double3 field_normal = make_double3(
        raw_normal.x * inv_jacobian,
        raw_normal.y * inv_jacobian,
        raw_normal.z * inv_jacobian
    );
    const double wavruim = harmonic_factor * k;
    const double k2 = k * k;

    Complex64 acc_g = {0.0, 0.0};
    Complex64 acc_h = {0.0, 0.0};
    Complex64 acc_ht = {0.0, 0.0};
    Complex64 acc_e = {0.0, 0.0};
    for (unsigned int q = 0; q < 4; ++q) {
        const double xi = SONIUM_TRI4_XI[q];
        const double eta = SONIUM_TRI4_ETA[q];
        const double n0 = 1.0 - xi - eta;
        const double3 position = make_double3(
            n0 * v0.x + xi * v1.x + eta * v2.x,
            n0 * v0.y + xi * v1.y + eta * v2.y,
            n0 * v0.z + xi * v1.z + eta * v2.z
        );
        const double integration_weight = SONIUM_TRI4_WEIGHTS[q] * jacobian;
        const double3 diff = make_double3(
            position.x - source.x,
            position.y - source.y,
            position.z - source.z
        );
        const double distance = sqrt(
            diff.x * diff.x + diff.y * diff.y + diff.z * diff.z
        );
        if (distance < 1.0e-15) {
            continue;
        }
        const double inv_distance = 1.0 / distance;
        const double3 unit_r = make_double3(
            diff.x * inv_distance,
            diff.y * inv_distance,
            diff.z * inv_distance
        );
        const double source_axis = source_axis_index == 0
            ? unit_r.x
            : (source_axis_index == 1 ? unit_r.y : unit_r.z);
        const double3 d_unit_r = source_axis_index == 0
            ? make_double3(
                (unit_r.x * source_axis - 1.0) * inv_distance,
                unit_r.y * source_axis * inv_distance,
                unit_r.z * source_axis * inv_distance
            )
            : (source_axis_index == 1
                ? make_double3(
                    unit_r.x * source_axis * inv_distance,
                    (unit_r.y * source_axis - 1.0) * inv_distance,
                    unit_r.z * source_axis * inv_distance
                )
                : make_double3(
                    unit_r.x * source_axis * inv_distance,
                    unit_r.y * source_axis * inv_distance,
                    (unit_r.z * source_axis - 1.0) * inv_distance
                ));
        const double d_inv_distance = source_axis * inv_distance * inv_distance;
        const double inv_distance2 = inv_distance * inv_distance;
        const double d_inv_distance2 =
            2.0 * source_axis * inv_distance * inv_distance2;
        const double phase = wavruim * distance;
        const double green_scale =
            integration_weight / (12.56637061435917295385 * distance);
        const Complex64 green = {
            cos(phase) * green_scale,
            sin(phase) * green_scale
        };
        const Complex64 z1 = {-inv_distance, wavruim};
        const Complex64 dgreen = cmul_f64(
            green,
            Complex64{-source_axis * z1.re, -source_axis * z1.im}
        );
        const Complex64 dz1 = {-d_inv_distance, 0.0};
        const Complex64 hb = cmul_f64(green, z1);
        const Complex64 dhb = {
            cmul_f64(dgreen, z1).re + cmul_f64(green, dz1).re,
            cmul_f64(dgreen, z1).im + cmul_f64(green, dz1).im
        };
        const double r_dot_ny =
            unit_r.x * field_normal.x +
            unit_r.y * field_normal.y +
            unit_r.z * field_normal.z;
        const double r_dot_nx = -(
            unit_r.x * source_normal.x +
            unit_r.y * source_normal.y +
            unit_r.z * source_normal.z
        );
        const double d_r_dot_ny =
            d_unit_r.x * field_normal.x +
            d_unit_r.y * field_normal.y +
            d_unit_r.z * field_normal.z;
        const double d_r_dot_nx = -(
            d_unit_r.x * source_normal.x +
            d_unit_r.y * source_normal.y +
            d_unit_r.z * source_normal.z
        );
        const Complex64 dh = {
            dhb.re * r_dot_ny + hb.re * d_r_dot_ny,
            dhb.im * r_dot_ny + hb.im * d_r_dot_ny
        };
        const Complex64 dht = {
            dhb.re * r_dot_nx + hb.re * d_r_dot_nx,
            dhb.im * r_dot_nx + hb.im * d_r_dot_nx
        };
        const double rq = r_dot_ny * r_dot_nx;
        const double d_rq = d_r_dot_ny * r_dot_nx + r_dot_ny * d_r_dot_nx;
        const double normal_dot =
            source_normal.x * field_normal.x +
            source_normal.y * field_normal.y +
            source_normal.z * field_normal.z;
        const double a = (3.0 * inv_distance2 - k2) * rq + inv_distance2 * normal_dot;
        const double d_a =
            (3.0 * d_inv_distance2) * rq +
            (3.0 * inv_distance2 - k2) * d_rq +
            d_inv_distance2 * normal_dot;
        const double b = -wavruim * inv_distance * (3.0 * rq + normal_dot);
        const double d_b = -wavruim * (
            d_inv_distance * (3.0 * rq + normal_dot) +
            inv_distance * 3.0 * d_rq
        );
        const Complex64 d_e_factor = {d_a, d_b};
        const Complex64 de = {
            cmul_f64(dgreen, Complex64{a, b}).re + cmul_f64(green, d_e_factor).re,
            cmul_f64(dgreen, Complex64{a, b}).im + cmul_f64(green, d_e_factor).im
        };
        acc_g = cadd_f64(acc_g, dgreen);
        acc_h = cadd_f64(acc_h, dh);
        acc_ht = cadd_f64(acc_ht, dht);
        acc_e = cadd_f64(acc_e, de);
    }
    out[out_offset + gid * out_stride] =
        SoniumBemRegularPairOutF64{acc_g, acc_h, acc_ht, acc_e};
}

extern "C" __global__ void sonium_bem_regular_pair_c64_source_x_derivative(
    const SoniumBemRegularPairInF64* pairs,
    SoniumBemRegularPairOutF64* out,
    double k,
    double harmonic_factor,
    unsigned long long n_pairs,
    unsigned long long pair_offset,
    unsigned long long pair_stride,
    unsigned long long out_offset,
    unsigned long long out_stride
) {
    const unsigned long long gid =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= n_pairs) {
        return;
    }
    sonium_bem_regular_pair_c64_source_derivative_impl(
        pairs, out, k, harmonic_factor, n_pairs, pair_offset, pair_stride,
        out_offset, out_stride, 0, gid
    );
}

extern "C" __global__ void sonium_bem_regular_pair_c64_source_y_derivative(
    const SoniumBemRegularPairInF64* pairs,
    SoniumBemRegularPairOutF64* out,
    double k,
    double harmonic_factor,
    unsigned long long n_pairs,
    unsigned long long pair_offset,
    unsigned long long pair_stride,
    unsigned long long out_offset,
    unsigned long long out_stride
) {
    const unsigned long long gid =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= n_pairs) {
        return;
    }
    sonium_bem_regular_pair_c64_source_derivative_impl(
        pairs, out, k, harmonic_factor, n_pairs, pair_offset, pair_stride,
        out_offset, out_stride, 1, gid
    );
}

extern "C" __global__ void sonium_bem_field_c64(
    const SoniumBemFieldElementC64* elements,
    const Complex64* pressure_traces,
    const Complex64* velocity_traces,
    const SoniumBemFieldPointC64* points,
    Complex64* out,
    double wave_number,
    double harmonic_factor,
    double pressure_factor,
    unsigned long long n_elements,
    unsigned long long n_points,
    unsigned long long point_offset,
    unsigned long long point_stride,
    unsigned long long out_offset,
    unsigned long long out_stride
) {
    const unsigned long long gid =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= n_points) {
        return;
    }

    const double3 x = load3_f64(points[point_offset + gid * point_stride].position);
    const double wavruim = wave_number * harmonic_factor;
    Complex64 result = {0.0, 0.0};

    // These are GAUCORWEI_TR7 weights after the CPU rule's 0.5 area scale.
    const double xis[7] = {
        0.333333333333333, 0.797426985353087, 0.101286507323456,
        0.101286507323456, 0.470142064105115, 0.059715871789770,
        0.470142064105115
    };
    const double etas[7] = {
        0.333333333333333, 0.101286507323456, 0.797426985353087,
        0.101286507323456, 0.059715871789770, 0.470142064105115,
        0.470142064105115
    };
    const double weights[7] = {
        0.1125, 0.0629695902724135, 0.0629695902724135,
        0.0629695902724135, 0.066197076394253, 0.066197076394253,
        0.066197076394253
    };

    for (unsigned long long element_index = 0; element_index < n_elements; ++element_index) {
        const SoniumBemFieldElementC64 e = elements[element_index];
        const double3 v0 = load3_f64(e.v0);
        const double3 v1 = load3_f64(e.v1);
        const double3 v2 = load3_f64(e.v2);
        const double3 edge1 = make_double3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
        const double3 edge2 = make_double3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
        const double3 raw_normal = make_double3(
            edge1.y * edge2.z - edge1.z * edge2.y,
            edge1.z * edge2.x - edge1.x * edge2.z,
            edge1.x * edge2.y - edge1.y * edge2.x
        );
        const double jacobian = sqrt(
            raw_normal.x * raw_normal.x +
            raw_normal.y * raw_normal.y +
            raw_normal.z * raw_normal.z
        );
        if (jacobian < 1.0e-15) {
            continue;
        }
        const double inv_jacobian = 1.0 / jacobian;
        const double3 normal = make_double3(
            raw_normal.x * inv_jacobian,
            raw_normal.y * inv_jacobian,
            raw_normal.z * inv_jacobian
        );

        for (unsigned int q = 0; q < 7; ++q) {
            const double xi = xis[q];
            const double eta = etas[q];
            const double n0 = 1.0 - xi - eta;
            const double3 y = make_double3(
                n0 * v0.x + xi * v1.x + eta * v2.x,
                n0 * v0.y + xi * v1.y + eta * v2.y,
                n0 * v0.z + xi * v1.z + eta * v2.z
            );
            const double3 r_vec = make_double3(y.x - x.x, y.y - x.y, y.z - x.z);
            const double r = sqrt(
                r_vec.x * r_vec.x + r_vec.y * r_vec.y + r_vec.z * r_vec.z
            );
            if (r < 1.0e-15) {
                continue;
            }
            const double inv_r = 1.0 / r;
            const double drdn_y = (
                r_vec.x * normal.x + r_vec.y * normal.y + r_vec.z * normal.z
            ) * inv_r;
            const double surface_weight = jacobian * weights[q];
            const double kr = wavruim * r;
            const double green_scale = surface_weight / (12.56637061435917295385 * r);
            const Complex64 green = {cos(kr) * green_scale, sin(kr) * green_scale};
            const Complex64 derivative = cmul_f64(green, Complex64{-inv_r, wavruim});
            const Complex64 pressure = pressure_traces ? pressure_traces[element_index] : e.pressure;
            const Complex64 velocity = velocity_traces ? velocity_traces[element_index] : e.velocity;
            result = cadd_f64(
                result,
                cscale_f64(cmul_f64(pressure, derivative), drdn_y)
            );

            if (velocity.re * velocity.re + velocity.im * velocity.im > 1.0e-30) {
                const Complex64 iw_rho = {0.0, pressure_factor};
                result = cadd_f64(
                    result,
                    cscale_f64(cmul_f64(cmul_f64(velocity, iw_rho), green), -1.0)
                );
            }
        }
    }
    out[out_offset + gid * out_stride] = result;
}

// General Complex64 pair integration over host-selected quadrature charts.
// The chart geometry is data only: all G/H/H^T/E arithmetic, including the
// regularised edge hypersingular term, is evaluated and accumulated here.
struct SoniumBemQuadratureSampleC64 {
    double position[3];
    double field_normal[3];
    double edge_direction[3];
    double weight;
    double basis_weight;
    unsigned kind;
    unsigned _pad;
};

struct SoniumBemQuadraturePairC64 {
    double source_point[3];
    double source_normal[3];
    unsigned long long sample_offset;
    unsigned long long sample_count;
};

__device__ __forceinline__ Complex64 csub_f64(Complex64 a, Complex64 b) {
    return Complex64{a.re - b.re, a.im - b.im};
}

extern "C" __global__ void sonium_bem_quadrature_pair_c64(
    const SoniumBemQuadraturePairC64* pairs,
    const SoniumBemQuadratureSampleC64* samples,
    SoniumBemRegularPairOutF64* out,
    double wave_number,
    double harmonic_factor,
    unsigned long long n_samples,
    unsigned long long n_pairs,
    unsigned long long pair_offset,
    unsigned long long pair_stride,
    unsigned long long out_offset,
    unsigned long long out_stride
) {
    const unsigned long long gid =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= n_pairs) {
        return;
    }

    const SoniumBemQuadraturePairC64 pair = pairs[pair_offset + gid * pair_stride];
    const double3 source = load3_f64(pair.source_point);
    const double3 source_normal = load3_f64(pair.source_normal);
    Complex64 acc_g = {0.0, 0.0};
    Complex64 acc_h = {0.0, 0.0};
    Complex64 acc_ht = {0.0, 0.0};
    Complex64 acc_e = {0.0, 0.0};

    const unsigned long long end = pair.sample_offset + pair.sample_count;
    if (pair.sample_offset > n_samples || end < pair.sample_offset || end > n_samples) {
        out[out_offset + gid * out_stride] =
            SoniumBemRegularPairOutF64{acc_g, acc_h, acc_ht, acc_e};
        return;
    }

    const double k2 = wave_number * wave_number;
    const double wavruim = harmonic_factor * wave_number;
    for (unsigned long long sample_index = pair.sample_offset;
         sample_index < end;
         ++sample_index) {
        const SoniumBemQuadratureSampleC64 sample = samples[sample_index];
        const double3 position = load3_f64(sample.position);
        const double3 diff = make_double3(
            position.x - source.x,
            position.y - source.y,
            position.z - source.z
        );
        const double r = sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
        if (r < 1.0e-15) {
            continue;
        }
        const double inv_r = 1.0 / r;
        const double3 unit_r = make_double3(diff.x * inv_r, diff.y * inv_r, diff.z * inv_r);
        const double scaled_weight = sample.weight * sample.basis_weight;
        const double phase = wavruim * r;
        const double base_scale = 1.0 / (12.56637061435917295385 * r);

        if (sample.kind == 2u) {
            const double3 edge = load3_f64(sample.edge_direction);
            const double green_scale = base_scale;
            const Complex64 green = {cos(phase) * green_scale, sin(phase) * green_scale};
            const Complex64 gradient_factor = cmul_f64(green, Complex64{-inv_r, wavruim});
            const double3 gradient = make_double3(
                gradient_factor.re * unit_r.x,
                gradient_factor.re * unit_r.y,
                gradient_factor.re * unit_r.z
            );
            const double3 cross_gradient = make_double3(
                gradient.y * edge.z - gradient.z * edge.y,
                gradient.z * edge.x - gradient.x * edge.z,
                gradient.x * edge.y - gradient.y * edge.x
            );
            const double edge_dot =
                cross_gradient.x * source_normal.x +
                cross_gradient.y * source_normal.y +
                cross_gradient.z * source_normal.z;
            // The imaginary component has the same scalar edge factor.
            const Complex64 edge_factor = cmul_f64(green, Complex64{-inv_r, wavruim});
            const double edge_re =
                (edge_factor.re * unit_r.y * edge.z - edge_factor.re * unit_r.z * edge.y) * source_normal.x +
                (edge_factor.re * unit_r.z * edge.x - edge_factor.re * unit_r.x * edge.z) * source_normal.y +
                (edge_factor.re * unit_r.x * edge.y - edge_factor.re * unit_r.y * edge.x) * source_normal.z;
            const double edge_im =
                (edge_factor.im * unit_r.y * edge.z - edge_factor.im * unit_r.z * edge.y) * source_normal.x +
                (edge_factor.im * unit_r.z * edge.x - edge_factor.im * unit_r.x * edge.z) * source_normal.y +
                (edge_factor.im * unit_r.x * edge.y - edge_factor.im * unit_r.y * edge.x) * source_normal.z;
            // `edge_dot` is retained as a readable real-path check; the
            // component-wise expression publishes both complex components.
            (void)edge_dot;
            acc_e.re += edge_re * scaled_weight;
            acc_e.im += edge_im * scaled_weight;
            continue;
        }

        const double3 field_normal = load3_f64(sample.field_normal);
        const double r_dot_ny =
            unit_r.x * field_normal.x + unit_r.y * field_normal.y + unit_r.z * field_normal.z;
        const double r_dot_nx = -(
            unit_r.x * source_normal.x + unit_r.y * source_normal.y + unit_r.z * source_normal.z
        );
        const double normal_dot =
            source_normal.x * field_normal.x +
            source_normal.y * field_normal.y +
            source_normal.z * field_normal.z;
        const Complex64 green = {
            cos(phase) * base_scale * scaled_weight,
            sin(phase) * base_scale * scaled_weight
        };
        const Complex64 kernel_factor = cmul_f64(green, Complex64{-inv_r, wavruim});
        const Complex64 h = cscale_f64(kernel_factor, r_dot_ny);
        const Complex64 ht = cscale_f64(kernel_factor, r_dot_nx);
        acc_g = cadd_f64(acc_g, green);
        acc_h = cadd_f64(acc_h, h);
        acc_ht = cadd_f64(acc_ht, ht);

        if (sample.kind == 0u) {
            const double rq = r_dot_ny * r_dot_nx;
            const double r2 = r * r;
            const Complex64 e_factor = {
                (3.0 / r2 - k2) * rq + normal_dot / r2,
                -wavruim / r * (3.0 * rq + normal_dot)
            };
            acc_e = cadd_f64(acc_e, cmul_f64(green, e_factor));
        } else if (sample.kind == 1u) {
            acc_e = cadd_f64(acc_e, cscale_f64(green, k2 * normal_dot));
        }
    }
    out[out_offset + gid * out_stride] =
        SoniumBemRegularPairOutF64{acc_g, acc_h, acc_ht, acc_e};
}

// Analytic d/dk of the chart-driven pair kernel.  This mirrors every
// regularised path in `sonium_bem_quadrature_pair_c64`, including the
// hypersingular edge and singular bulk contributions.
extern "C" __global__ void sonium_bem_quadrature_pair_c64_derivative(
    const SoniumBemQuadraturePairC64* pairs,
    const SoniumBemQuadratureSampleC64* samples,
    SoniumBemRegularPairOutF64* out,
    double wave_number,
    double harmonic_factor,
    unsigned long long n_samples,
    unsigned long long n_pairs,
    unsigned long long pair_offset,
    unsigned long long pair_stride,
    unsigned long long out_offset,
    unsigned long long out_stride
) {
    const unsigned long long gid =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= n_pairs) {
        return;
    }

    const SoniumBemQuadraturePairC64 pair = pairs[pair_offset + gid * pair_stride];
    const double3 source = load3_f64(pair.source_point);
    const double3 source_normal = load3_f64(pair.source_normal);
    Complex64 acc_g = {0.0, 0.0};
    Complex64 acc_h = {0.0, 0.0};
    Complex64 acc_ht = {0.0, 0.0};
    Complex64 acc_e = {0.0, 0.0};
    const unsigned long long end = pair.sample_offset + pair.sample_count;
    if (pair.sample_offset > n_samples || end < pair.sample_offset || end > n_samples) {
        out[out_offset + gid * out_stride] =
            SoniumBemRegularPairOutF64{acc_g, acc_h, acc_ht, acc_e};
        return;
    }

    const double k2 = wave_number * wave_number;
    const double wavruim = harmonic_factor * wave_number;
    for (unsigned long long sample_index = pair.sample_offset;
         sample_index < end;
         ++sample_index) {
        const SoniumBemQuadratureSampleC64 sample = samples[sample_index];
        const double3 position = load3_f64(sample.position);
        const double3 diff = make_double3(
            position.x - source.x,
            position.y - source.y,
            position.z - source.z
        );
        const double r = sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
        if (r < 1.0e-15) {
            continue;
        }
        const double inv_r = 1.0 / r;
        const double3 unit_r = make_double3(diff.x * inv_r, diff.y * inv_r, diff.z * inv_r);
        const double scaled_weight = sample.weight * sample.basis_weight;
        const double phase = wavruim * r;
        const double base_scale = 1.0 / (12.56637061435917295385 * r);
        const Complex64 green = {
            cos(phase) * base_scale * scaled_weight,
            sin(phase) * base_scale * scaled_weight
        };
        const Complex64 dgreen = cmul_f64(green, Complex64{0.0, harmonic_factor * r});
        const Complex64 z1 = {-inv_r, wavruim};
        const Complex64 dz1 = {0.0, harmonic_factor};

        if (sample.kind == 2u) {
            const double3 edge = load3_f64(sample.edge_direction);
            const double edge_x = unit_r.y * edge.z - unit_r.z * edge.y;
            const double edge_y = unit_r.z * edge.x - unit_r.x * edge.z;
            const double edge_z = unit_r.x * edge.y - unit_r.y * edge.x;
            const double edge_scale =
                edge_x * source_normal.x + edge_y * source_normal.y + edge_z * source_normal.z;
            const Complex64 edge_factor = cmul_f64(green, z1);
            const Complex64 dedge_factor = {
                cmul_f64(dgreen, z1).re + cmul_f64(green, dz1).re,
                cmul_f64(dgreen, z1).im + cmul_f64(green, dz1).im
            };
            acc_e.re += dedge_factor.re * edge_scale;
            acc_e.im += dedge_factor.im * edge_scale;
            continue;
        }

        const double3 field_normal = load3_f64(sample.field_normal);
        const double r_dot_ny =
            unit_r.x * field_normal.x + unit_r.y * field_normal.y + unit_r.z * field_normal.z;
        const double r_dot_nx = -(
            unit_r.x * source_normal.x + unit_r.y * source_normal.y + unit_r.z * source_normal.z
        );
        const double normal_dot =
            source_normal.x * field_normal.x +
            source_normal.y * field_normal.y +
            source_normal.z * field_normal.z;
        const Complex64 dkernel_factor = {
            cmul_f64(dgreen, z1).re + cmul_f64(green, dz1).re,
            cmul_f64(dgreen, z1).im + cmul_f64(green, dz1).im
        };
        acc_g.re += dgreen.re;
        acc_g.im += dgreen.im;
        acc_h.re += dkernel_factor.re * r_dot_ny;
        acc_h.im += dkernel_factor.im * r_dot_ny;
        acc_ht.re += dkernel_factor.re * r_dot_nx;
        acc_ht.im += dkernel_factor.im * r_dot_nx;

        if (sample.kind == 0u) {
            const double rq = r_dot_ny * r_dot_nx;
            const double r2 = r * r;
            const Complex64 e_factor = {
                (3.0 / r2 - k2) * rq + normal_dot / r2,
                -wavruim / r * (3.0 * rq + normal_dot)
            };
            const Complex64 de_factor = {
                -2.0 * wave_number * rq,
                -harmonic_factor / r * (3.0 * rq + normal_dot)
            };
            const Complex64 dvalue = {
                cmul_f64(dgreen, e_factor).re + cmul_f64(green, de_factor).re,
                cmul_f64(dgreen, e_factor).im + cmul_f64(green, de_factor).im
            };
            acc_e.re += dvalue.re;
            acc_e.im += dvalue.im;
        } else if (sample.kind == 1u) {
            acc_e.re += (dgreen.re * k2 + green.re * 2.0 * wave_number) * normal_dot;
            acc_e.im += (dgreen.im * k2 + green.im * 2.0 * wave_number) * normal_dot;
        }
    }
    out[out_offset + gid * out_stride] =
        SoniumBemRegularPairOutF64{acc_g, acc_h, acc_ht, acc_e};
}

// One row per thread gives a deterministic RHS accumulation. Matrix entries
// are column-major, so adjacent row threads write adjacent locations.
extern "C" __global__ void sonium_bem_dense_system_c64(
    const SoniumBemRegularPairOutF64* regular,
    const SoniumBemRegularPairOutF64* quadrature,
    const int* routes, const Complex64* coefficients,
    Complex64* matrix, Complex64* rhs,
    unsigned long long n, unsigned long long nr, unsigned long long nq
) {
    const unsigned long long row = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    const double invalid = __longlong_as_double(0x7ff8000000000000ULL);
    Complex64 sum = coefficients[row * 10 + 9];
    for (unsigned long long col = 0; col < n; ++col) {
        const int route = routes[row + col * n];
        if (route == (-2147483647 - 1)) {
            matrix[row + col * n] = Complex64{0.0, 0.0};
            continue;
        }
        const bool is_regular = route < 0;
        const unsigned long long index = is_regular ? (unsigned long long)(-(long long)route - 1) : (unsigned long long)route;
        if (index >= (is_regular ? nr : nq)) {
            matrix[row + col * n] = Complex64{invalid, invalid};
            sum = Complex64{invalid, invalid};
            continue;
        }
        const SoniumBemRegularPairOutF64 p = is_regular ? regular[index] : quadrature[index];
        const Complex64 channels[4] = {p.g, p.h, p.ht, p.e};
        Complex64 a = {0.0, 0.0};
        for (int k = 0; k < 4; ++k) {
            const Complex64 w = coefficients[col * 10 + k];
            const Complex64 v = coefficients[col * 10 + 4 + k];
            const Complex64 x = channels[k];
            a.re += x.re * w.re - x.im * w.im;
            a.im += x.re * w.im + x.im * w.re;
            sum.re += x.re * v.re - x.im * v.im;
            sum.im += x.re * v.im + x.im * v.re;
        }
        if (row == col) {
            a.re += coefficients[col * 10 + 8].re;
            a.im += coefficients[col * 10 + 8].im;
        }
        matrix[row + col * n] = a;
    }
    rhs[row] = sum;
}

// Six doubles per point (position, normal) and source (coordinates, complex
// amplitude, kind). One thread per collocation row keeps source order stable.
extern "C" __global__ void sonium_bem_incident_rhs_c64(
    const double* points, const double* sources, Complex64* rhs,
    unsigned long long n, unsigned long long source_count,
    double k, double harmonic, double gamma, double beta_re, double beta_im
) {
    const unsigned long long row = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    const double* x = points + row * 6;
    Complex64 p = {0.0, 0.0}, dp = {0.0, 0.0};
    for (unsigned long long j = 0; j < source_count; ++j) {
        const double* s = sources + j * 6;
        double phase, scale = 1.0, dr = 0.0, di;
        if (s[5] == 0.0) {
            phase = harmonic * k * (s[0]*x[0] + s[1]*x[1] + s[2]*x[2]);
            di = harmonic * k * (s[0]*x[3] + s[1]*x[4] + s[2]*x[5]);
        } else if (s[5] == 1.0) {
            const double dx = x[0]-s[0], dy = x[1]-s[1], dz = x[2]-s[2];
            const double r = sqrt(dx*dx + dy*dy + dz*dz);
            // Match IncidentField's existing near-source convention.
            if (r <= 1.0e-10) continue;
            phase = harmonic * k * r;
            scale = 1.0 / (12.56637061435917295385 * r);
            const double drdn = (dx*x[3] + dy*x[4] + dz*x[5]) / r;
            dr = -drdn / r;
            di = harmonic * k * drdn;
        } else {
            const double invalid = __longlong_as_double(0x7ff8000000000000ULL);
            rhs[row] = Complex64{invalid, invalid};
            return;
        }
        double sn, cs;
        sincos(phase, &sn, &cs);
        const Complex64 value = {(s[3]*cs - s[4]*sn)*scale, (s[3]*sn + s[4]*cs)*scale};
        p.re += value.re;
        p.im += value.im;
        dp.re += dr*value.re - di*value.im;
        dp.im += dr*value.im + di*value.re;
    }
    rhs[row] = Complex64{-(gamma*p.re + beta_re*dp.re - beta_im*dp.im),
                         -(gamma*p.im + beta_re*dp.im + beta_im*dp.re)};
}
