// Native CUDA kernels for element-local tensor contractions and segmented
// assembly reductions. Each output owns one deterministic accumulation lane.

extern "C" __global__ void sonium_segmented_sum_f32(
    const float* values, const int* offsets, float* output, unsigned segments) {
    unsigned segment = blockIdx.x * blockDim.x + threadIdx.x;
    if (segment >= segments) return;
    int start = offsets[segment];
    int end = offsets[segment + 1];
    float sum = 0.0f;
    for (int i = start; i < end; ++i) sum += values[i];
    output[segment] = sum;
}

extern "C" __global__ void sonium_segmented_sum_f64(
    const double* values, const int* offsets, double* output, unsigned segments) {
    unsigned segment = blockIdx.x * blockDim.x + threadIdx.x;
    if (segment >= segments) return;
    int start = offsets[segment];
    int end = offsets[segment + 1];
    double sum = 0.0;
    for (int i = start; i < end; ++i) sum += values[i];
    output[segment] = sum;
}

extern "C" __global__ void sonium_segmented_sum_c32(
    const float2* values, const int* offsets, float2* output, unsigned segments) {
    unsigned segment = blockIdx.x * blockDim.x + threadIdx.x;
    if (segment >= segments) return;
    int start = offsets[segment];
    int end = offsets[segment + 1];
    float2 sum = make_float2(0.0f, 0.0f);
    for (int i = start; i < end; ++i) {
        sum.x += values[i].x;
        sum.y += values[i].y;
    }
    output[segment] = sum;
}

extern "C" __global__ void sonium_segmented_sum_c64(
    const double2* values, const int* offsets, double2* output, unsigned segments) {
    unsigned segment = blockIdx.x * blockDim.x + threadIdx.x;
    if (segment >= segments) return;
    int start = offsets[segment];
    int end = offsets[segment + 1];
    double2 sum = make_double2(0.0, 0.0);
    for (int i = start; i < end; ++i) {
        sum.x += values[i].x;
        sum.y += values[i].y;
    }
    output[segment] = sum;
}

extern "C" __global__ void sonium_batched_local_contract_f32(
    const float* lhs, const float* rhs, float* output,
    unsigned batches, unsigned rows, unsigned inner, unsigned cols) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = batches * rows * cols;
    if (index >= total) return;
    unsigned col = index % cols;
    unsigned row = (index / cols) % rows;
    unsigned batch = index / (rows * cols);
    unsigned lhs_base = (batch * rows + row) * inner;
    unsigned rhs_base = batch * inner * cols + col;
    float sum = 0.0f;
    for (unsigned k = 0; k < inner; ++k)
        sum += lhs[lhs_base + k] * rhs[rhs_base + k * cols];
    output[index] = sum;
}

extern "C" __global__ void sonium_batched_local_contract_f64(
    const double* lhs, const double* rhs, double* output,
    unsigned batches, unsigned rows, unsigned inner, unsigned cols) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = batches * rows * cols;
    if (index >= total) return;
    unsigned col = index % cols;
    unsigned row = (index / cols) % rows;
    unsigned batch = index / (rows * cols);
    unsigned lhs_base = (batch * rows + row) * inner;
    unsigned rhs_base = batch * inner * cols + col;
    double sum = 0.0;
    for (unsigned k = 0; k < inner; ++k)
        sum += lhs[lhs_base + k] * rhs[rhs_base + k * cols];
    output[index] = sum;
}

extern "C" __global__ void sonium_batched_local_contract_c32(
    const float2* lhs, const float2* rhs, float2* output,
    unsigned batches, unsigned rows, unsigned inner, unsigned cols) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = batches * rows * cols;
    if (index >= total) return;
    unsigned col = index % cols;
    unsigned row = (index / cols) % rows;
    unsigned batch = index / (rows * cols);
    unsigned lhs_base = (batch * rows + row) * inner;
    unsigned rhs_base = batch * inner * cols + col;
    float2 sum = make_float2(0.0f, 0.0f);
    for (unsigned k = 0; k < inner; ++k) {
        float2 a = lhs[lhs_base + k];
        float2 b = rhs[rhs_base + k * cols];
        sum.x += a.x * b.x - a.y * b.y;
        sum.y += a.x * b.y + a.y * b.x;
    }
    output[index] = sum;
}

// One Gauss-Seidel SOR sweep over a single color subset: thread `t` owns
// row `i = color_rows[t]` (same-color rows share no edges, so updates are
// independent). Mirrors the host gs_color_sweep_host semantics: missing
// diagonal defaults to 1, updates skipped when |a_ii| <= 1e-15.
extern "C" __global__ void sonium_gs_color_sweep_f32(
    const int* row_ptr, const int* col_idx, const float* values,
    const int* color_rows, const float* b, float* x, float omega, unsigned count) {
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    int i = color_rows[t];
    float diag = 1.0f;
    float sum = b[i];
    for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
        int j = col_idx[k];
        if (j == i) diag = values[k];
        else sum -= values[k] * x[j];
    }
    if (diag > 1e-15f || diag < -1e-15f) {
        float new_val = sum / diag;
        x[i] = (1.0f - omega) * x[i] + omega * new_val;
    }
}

extern "C" __global__ void sonium_gs_color_sweep_f64(
    const int* row_ptr, const int* col_idx, const double* values,
    const int* color_rows, const double* b, double* x, double omega, unsigned count) {
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    int i = color_rows[t];
    double diag = 1.0;
    double sum = b[i];
    for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
        int j = col_idx[k];
        if (j == i) diag = values[k];
        else sum -= values[k] * x[j];
    }
    if (diag > 1e-15 || diag < -1e-15) {
        double new_val = sum / diag;
        x[i] = (1.0 - omega) * x[i] + omega * new_val;
    }
}

extern "C" __global__ void sonium_gs_color_sweep_c32(
    const int* row_ptr, const int* col_idx, const float2* values,
    const int* color_rows, const float2* b, float2* x, float omega, unsigned count) {
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    int i = color_rows[t];
    float2 diag = make_float2(1.0f, 0.0f);
    float2 sum = b[i];
    for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
        int j = col_idx[k];
        if (j == i) {
            diag = values[k];
        } else {
            float2 a = values[k];
            float2 xj = x[j];
            sum.x -= a.x * xj.x - a.y * xj.y;
            sum.y -= a.x * xj.y + a.y * xj.x;
        }
    }
    float mag2 = diag.x * diag.x + diag.y * diag.y;
    if (mag2 > 1e-30f) {
        float2 n;
        n.x = (sum.x * diag.x + sum.y * diag.y) / mag2;
        n.y = (sum.y * diag.x - sum.x * diag.y) / mag2;
        float2 old = x[i];
        x[i].x = (1.0f - omega) * old.x + omega * n.x;
        x[i].y = (1.0f - omega) * old.y + omega * n.y;
    }
}

extern "C" __global__ void sonium_gs_color_sweep_c64(
    const int* row_ptr, const int* col_idx, const double2* values,
    const int* color_rows, const double2* b, double2* x, double omega, unsigned count) {
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    int i = color_rows[t];
    double2 diag = make_double2(1.0, 0.0);
    double2 sum = b[i];
    for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
        int j = col_idx[k];
        if (j == i) {
            diag = values[k];
        } else {
            double2 a = values[k];
            double2 xj = x[j];
            sum.x -= a.x * xj.x - a.y * xj.y;
            sum.y -= a.x * xj.y + a.y * xj.x;
        }
    }
    double mag2 = diag.x * diag.x + diag.y * diag.y;
    if (mag2 > 1e-30) {
        double2 n;
        n.x = (sum.x * diag.x + sum.y * diag.y) / mag2;
        n.y = (sum.y * diag.x - sum.x * diag.y) / mag2;
        double2 old = x[i];
        x[i].x = (1.0 - omega) * old.x + omega * n.x;
        x[i].y = (1.0 - omega) * old.y + omega * n.y;
    }
}

extern "C" __global__ void sonium_batched_local_contract_c64(
    const double2* lhs, const double2* rhs, double2* output,
    unsigned batches, unsigned rows, unsigned inner, unsigned cols) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = batches * rows * cols;
    if (index >= total) return;
    unsigned col = index % cols;
    unsigned row = (index / cols) % rows;
    unsigned batch = index / (rows * cols);
    unsigned lhs_base = (batch * rows + row) * inner;
    unsigned rhs_base = batch * inner * cols + col;
    double2 sum = make_double2(0.0, 0.0);
    for (unsigned k = 0; k < inner; ++k) {
        double2 a = lhs[lhs_base + k];
        double2 b = rhs[rhs_base + k * cols];
        sum.x += a.x * b.x - a.y * b.y;
        sum.y += a.x * b.y + a.y * b.x;
    }
    output[index] = sum;
}

extern "C" __global__ void sonium_residual_update_f32(
    const float* rhs, float* action, unsigned int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) action[i] = rhs[i] - action[i];
}

extern "C" __global__ void sonium_residual_update_f64(
    const double* rhs, double* action, unsigned int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) action[i] = rhs[i] - action[i];
}

extern "C" __global__ void sonium_residual_update_c32(
    const float2* rhs, float2* action, unsigned int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) action[i] = make_float2(rhs[i].x - action[i].x, rhs[i].y - action[i].y);
}

extern "C" __global__ void sonium_residual_update_c64(
    const double2* rhs, double2* action, unsigned int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) action[i] = make_double2(rhs[i].x - action[i].x, rhs[i].y - action[i].y);
}

extern "C" __global__ void sonium_jacobi_update_f32(
    const float* inverse, const float* b, const float* ax,
    float* x, double omega, unsigned int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float w = (float)omega;
    x[i] += w * (inverse[i] * (b[i] - ax[i]));
}

extern "C" __global__ void sonium_jacobi_update_f64(
    const double* inverse, const double* b, const double* ax,
    double* x, double omega, unsigned int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double w = (double)omega;
    x[i] += w * (inverse[i] * (b[i] - ax[i]));
}

extern "C" __global__ void sonium_jacobi_update_c32(
    const float2* inverse, const float2* b, const float2* ax,
    float2* x, double omega, unsigned int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float w = (float)omega;
    const float re = b[i].x - ax[i].x, im = b[i].y - ax[i].y;
    const float2 d = inverse[i];
    x[i].x += w * (d.x * re - d.y * im);
    x[i].y += w * (d.x * im + d.y * re);
}

extern "C" __global__ void sonium_jacobi_update_c64(
    const double2* inverse, const double2* b, const double2* ax,
    double2* x, double omega, unsigned int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double w = (double)omega;
    const double re = b[i].x - ax[i].x, im = b[i].y - ax[i].y;
    const double2 d = inverse[i];
    x[i].x += w * (d.x * re - d.y * im);
    x[i].y += w * (d.x * im + d.y * re);
}

extern "C" __global__ void sonium_diagonal_scale_f32(
    const float* diagonal, const float* b, float* x,
    double omega, unsigned int rows, unsigned int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned int row = i < rows ? i : i % rows;
    const float w = (float)omega;
    x[i] = w * (diagonal[row] * b[i]);
}

extern "C" __global__ void sonium_diagonal_scale_f64(
    const double* diagonal, const double* b, double* x,
    double omega, unsigned int rows, unsigned int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned int row = i < rows ? i : i % rows;
    const double w = (double)omega;
    x[i] = w * (diagonal[row] * b[i]);
}

extern "C" __global__ void sonium_diagonal_scale_c32(
    const float2* diagonal, const float2* b, float2* x,
    double omega, unsigned int rows, unsigned int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned int row = i < rows ? i : i % rows;
    const float w = (float)omega;
    const float2 d = diagonal[row], v = b[i];
    x[i] = make_float2(w * (d.x * v.x - d.y * v.y), w * (d.x * v.y + d.y * v.x));
}

extern "C" __global__ void sonium_diagonal_scale_c64(
    const double2* diagonal, const double2* b, double2* x,
    double omega, unsigned int rows, unsigned int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned int row = i < rows ? i : i % rows;
    const double w = (double)omega;
    const double2 d = diagonal[row], v = b[i];
    x[i] = make_double2(w * (d.x * v.x - d.y * v.y), w * (d.x * v.y + d.y * v.x));
}

// One thread owns both recurrence outputs; beta=0 never reads stale direction.
extern "C" __global__ void sonium_chebyshev_update_f32(
    const float* inverse, const float* b, const float* ax,
    float* direction, float* x, double alpha, double beta, unsigned int rows, unsigned int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned int row = i < rows ? i : i % rows;
    const float a = (float)alpha, c = (float)beta;
    float next = a * (inverse[row] * (b[i] - ax[i]));
    if (c != 0) next += c * direction[i];
    direction[i] = next;
    x[i] += next;
}

// One thread owns both recurrence outputs; beta=0 never reads stale direction.
extern "C" __global__ void sonium_chebyshev_update_f64(
    const double* inverse, const double* b, const double* ax,
    double* direction, double* x, double alpha, double beta, unsigned int rows, unsigned int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned int row = i < rows ? i : i % rows;
    const double a = (double)alpha, c = (double)beta;
    double next = a * (inverse[row] * (b[i] - ax[i]));
    if (c != 0) next += c * direction[i];
    direction[i] = next;
    x[i] += next;
}

// One thread owns both recurrence outputs; beta=0 never reads stale direction.
extern "C" __global__ void sonium_chebyshev_update_c32(
    const float2* inverse, const float2* b, const float2* ax,
    float2* direction, float2* x, double alpha, double beta, unsigned int rows, unsigned int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned int row = i < rows ? i : i % rows;
    const float a = (float)alpha, c = (float)beta;
    const float re = b[i].x - ax[i].x, im = b[i].y - ax[i].y;
    const float2 d = inverse[row];
    float2 next = make_float2(a * (d.x * re - d.y * im), a * (d.x * im + d.y * re));
    if (c != 0) { next.x += c * direction[i].x; next.y += c * direction[i].y; }
    direction[i] = next;
    x[i].x += next.x;
    x[i].y += next.y;
}

// One thread owns both recurrence outputs; beta=0 never reads stale direction.
extern "C" __global__ void sonium_chebyshev_update_c64(
    const double2* inverse, const double2* b, const double2* ax,
    double2* direction, double2* x, double alpha, double beta, unsigned int rows, unsigned int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned int row = i < rows ? i : i % rows;
    const double a = (double)alpha, c = (double)beta;
    const double re = b[i].x - ax[i].x, im = b[i].y - ax[i].y;
    const double2 d = inverse[row];
    double2 next = make_double2(a * (d.x * re - d.y * im), a * (d.x * im + d.y * re));
    if (c != 0) { next.x += c * direction[i].x; next.y += c * direction[i].y; }
    direction[i] = next;
    x[i].x += next.x;
    x[i].y += next.y;
}

// Extract and invert a dense diagonal without staging matrix values on the host.
extern "C" __global__ void sonium_inverse_diagonal_f32(
    const float* matrix, float* inverse, unsigned int n, unsigned long long ld) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float d = matrix[(unsigned long long)i * (ld + 1)];
    inverse[i] = isfinite(d) && fabs(d) > 1.1920928955078125e-7 ? (float)1 / d : (float)nan("");
}
extern "C" __global__ void sonium_inverse_diagonal_f64(
    const double* matrix, double* inverse, unsigned int n, unsigned long long ld) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double d = matrix[(unsigned long long)i * (ld + 1)];
    inverse[i] = isfinite(d) && fabs(d) > 2.220446049250313e-16 ? (double)1 / d : (double)nan("");
}
extern "C" __global__ void sonium_inverse_diagonal_c32(
    const float2* matrix, float2* inverse, unsigned int n, unsigned long long ld) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float2 d = matrix[(unsigned long long)i * (ld + 1)];
    if (!isfinite(d.x) || !isfinite(d.y) || hypot(d.x, d.y) <= 1.1920928955078125e-7) {
        inverse[i] = make_float2(nan(""), nan("")); return;
    }
    // Scaling avoids squaring the original components (overflow for large d).
    float scale = fmax(fabs(d.x), fabs(d.y));
    float re = d.x / scale, im = d.y / scale;
    float norm = re * re + im * im;
    inverse[i] = make_float2((re / norm) / scale, (-im / norm) / scale);
}
extern "C" __global__ void sonium_inverse_diagonal_c64(
    const double2* matrix, double2* inverse, unsigned int n, unsigned long long ld) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double2 d = matrix[(unsigned long long)i * (ld + 1)];
    if (!isfinite(d.x) || !isfinite(d.y) || hypot(d.x, d.y) <= 2.220446049250313e-16) {
        inverse[i] = make_double2(nan(""), nan("")); return;
    }
    // Scaling avoids squaring the original components (overflow for large d).
    double scale = fmax(fabs(d.x), fabs(d.y));
    double re = d.x / scale, im = d.y / scale;
    double norm = re * re + im * im;
    inverse[i] = make_double2((re / norm) / scale, (-im / norm) / scale);
}

// One block owns the complete small-level iterate; barriers preserve Jacobi.
__device__ float sj_zero(float) { return 0; }
__device__ float sj_nan(float) { return (float)nan(""); }
__device__ float sj_add(float a, float b) { return a+b; }
__device__ float sj_sub(float a, float b) { return a-b; }
__device__ float sj_mul(float a, float b) { return a*b; }
__device__ float sj_scale(float a, double w) { return a*(float)w; }
__device__ double sj_zero(double) { return 0; }
__device__ double sj_nan(double) { return (double)nan(""); }
__device__ double sj_add(double a, double b) { return a+b; }
__device__ double sj_sub(double a, double b) { return a-b; }
__device__ double sj_mul(double a, double b) { return a*b; }
__device__ double sj_scale(double a, double w) { return a*(double)w; }
__device__ float2 sj_zero(float2) { return make_float2(0,0); }
__device__ float2 sj_nan(float2) { return make_float2(nan(""),nan("")); }
__device__ float2 sj_add(float2 a, float2 b) { return make_float2(a.x+b.x,a.y+b.y); }
__device__ float2 sj_sub(float2 a, float2 b) { return make_float2(a.x-b.x,a.y-b.y); }
__device__ float2 sj_mul(float2 a, float2 b) { return make_float2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x); }
__device__ float2 sj_scale(float2 a, double w) { return make_float2(a.x*(float)w,a.y*(float)w); }
__device__ double2 sj_zero(double2) { return make_double2(0,0); }
__device__ double2 sj_nan(double2) { return make_double2(nan(""),nan("")); }
__device__ double2 sj_add(double2 a, double2 b) { return make_double2(a.x+b.x,a.y+b.y); }
__device__ double2 sj_sub(double2 a, double2 b) { return make_double2(a.x-b.x,a.y-b.y); }
__device__ double2 sj_mul(double2 a, double2 b) { return make_double2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x); }
__device__ double2 sj_scale(double2 a, double w) { return make_double2(a.x*(double)w,a.y*(double)w); }

// Four adjacent lanes cooperate on each row; every launched warp participates.
__device__ float sj_down(float v, int delta) { return __shfl_down_sync(0xffffffff, v, delta, 4); }
__device__ double sj_down(double v, int delta) { return __shfl_down_sync(0xffffffff, v, delta, 4); }
__device__ float2 sj_down(float2 v, int delta) {
    return make_float2(sj_down(v.x, delta), sj_down(v.y, delta));
}
__device__ double2 sj_down(double2 v, int delta) {
    return make_double2(sj_down(v.x, delta), sj_down(v.y, delta));
}

template<class T> __device__ void small_jacobi(
    const int* rows, const int* cols, const T* values, const T* inverse,
    const T* b, T* x, unsigned n, unsigned nnz, unsigned sweeps, double omega) {
    __shared__ T old[256];
    __shared__ int bad;
    unsigned row = threadIdx.x / 4, lane = threadIdx.x % 4;
    if (threadIdx.x == 0) bad = (rows[0] != 0 || rows[n] != (int)nnz);
    __syncthreads();
    int begin = 0, end = 0;
    if (row < n) {
        begin = rows[row]; end = rows[row+1];
        if (begin < 0 || end < begin || (unsigned)end > nnz) {
            if (lane == 0) atomicOr(&bad, 1);
        } else for (unsigned j=(unsigned)begin+lane; j<(unsigned)end; j+=4)
            if (cols[j] < 0 || (unsigned)cols[j] >= n) atomicOr(&bad, 1);
        if (lane == 0) old[row] = x[row];
    }
    __syncthreads();
    if (bad) {
        if (row < n && lane == 0) x[row] = sj_nan(old[row]);
        return;
    }
    T rhs{}, diagonal{};
    if (row < n && lane == 0) { rhs = b[row]; diagonal = inverse[row]; }
    for (unsigned sweep=0; sweep<sweeps; ++sweep) {
        T sum{}, next{};
        if (row < n) {
            for (unsigned j=(unsigned)begin+lane; j<(unsigned)end; j+=4)
                sum = sj_add(sum, sj_mul(values[j],old[cols[j]]));
        }
        // All lanes execute the shuffles, including padding rows in the last warp.
        sum = sj_add(sum, sj_down(sum, 2));
        sum = sj_add(sum, sj_down(sum, 1));
        if (row < n && lane == 0)
            next = sj_add(old[row], sj_scale(sj_mul(diagonal,sj_sub(rhs,sum)),omega));
        __syncthreads(); // All reads use the previous simultaneous iterate.
        if (row < n && lane == 0) old[row] = next;
        __syncthreads(); // Publish the new iterate to every row before reading it.
    }
    if (row < n && lane == 0) x[row] = old[row];
}
extern "C" __global__ void sonium_small_jacobi_f32(
    const int* rows, const int* cols, const float* values, const float* inverse,
    const float* b, float* x, unsigned n, unsigned nnz, unsigned sweeps, double omega) {
    small_jacobi(rows, cols, values, inverse, b, x, n, nnz, sweeps, omega);
}
extern "C" __global__ void sonium_small_jacobi_f64(
    const int* rows, const int* cols, const double* values, const double* inverse,
    const double* b, double* x, unsigned n, unsigned nnz, unsigned sweeps, double omega) {
    small_jacobi(rows, cols, values, inverse, b, x, n, nnz, sweeps, omega);
}
extern "C" __global__ void sonium_small_jacobi_c32(
    const int* rows, const int* cols, const float2* values, const float2* inverse,
    const float2* b, float2* x, unsigned n, unsigned nnz, unsigned sweeps, double omega) {
    small_jacobi(rows, cols, values, inverse, b, x, n, nnz, sweeps, omega);
}
extern "C" __global__ void sonium_small_jacobi_c64(
    const int* rows, const int* cols, const double2* values, const double2* inverse,
    const double2* b, double2* x, unsigned n, unsigned nnz, unsigned sweeps, double omega) {
    small_jacobi(rows, cols, values, inverse, b, x, n, nnz, sweeps, omega);
}

// One pass over two fixed-pattern numeric bases; no global intermediate.
extern "C" __global__ void sonium_combination_f32(
    const float* x, const float* y, float* output, unsigned n,
    double ar, double ai, double br, double bi) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = (float)ar, b = (float)br;
    output[i] = a*x[i] + b*y[i];
}
extern "C" __global__ void sonium_combination_f64(
    const double* x, const double* y, double* output, unsigned n,
    double ar, double ai, double br, double bi) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double a = (double)ar, b = (double)br;
    output[i] = a*x[i] + b*y[i];
}
extern "C" __global__ void sonium_combination_c32(
    const float2* x, const float2* y, float2* output, unsigned n,
    double ar, double ai, double br, double bi) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = (float)ar, b = (float)br;
    float c = (float)ai, d = (float)bi;
    float2 u = x[i], v = y[i];
    output[i] = make_float2((a*u.x-c*u.y)+(b*v.x-d*v.y),
                              (a*u.y+c*u.x)+(b*v.y+d*v.x));
}
extern "C" __global__ void sonium_combination_c64(
    const double2* x, const double2* y, double2* output, unsigned n,
    double ar, double ai, double br, double bi) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double a = (double)ar, b = (double)br;
    double c = (double)ai, d = (double)bi;
    double2 u = x[i], v = y[i];
    output[i] = make_double2((a*u.x-c*u.y)+(b*v.x-d*v.y),
                              (a*u.y+c*u.x)+(b*v.y+d*v.x));
}

// One simultaneous sweep across any number of rows. Input/output never alias.
template<class T> __device__ void csr_jacobi_step(
    const int* rows, const int* cols, const T* values, const T* inverse,
    const T* b, const T* x, T* output, unsigned n, unsigned nnz, double omega) {
    unsigned row = blockIdx.x * 64 + threadIdx.x / 4, lane = threadIdx.x % 4;
    T sum{};
    int bad = 0;
    if (row < n) {
        int begin = rows[row], end = rows[row+1];
        bad = rows[0] != 0 || rows[n] != (int)nnz || begin < 0
            || end < begin || (unsigned)end > nnz;
        if (!bad) {
            for (unsigned j=(unsigned)begin+lane; j<(unsigned)end; j+=4) {
                int col = cols[j];
                if (col < 0 || (unsigned)col >= n) bad = 1;
                else sum = sj_add(sum, sj_mul(values[j], x[col]));
            }
        }
    }
    // Full warps participate even for padding rows in the final block.
    sum = sj_add(sum, sj_down(sum, 2));
    sum = sj_add(sum, sj_down(sum, 1));
    bad |= __shfl_down_sync(0xffffffff, bad, 2, 4);
    bad |= __shfl_down_sync(0xffffffff, bad, 1, 4);
    if (row < n && lane == 0) {
        output[row] = bad ? sj_nan(sum)
            : sj_add(x[row], sj_scale(sj_mul(inverse[row], sj_sub(b[row], sum)), omega));
    }
}
extern "C" __global__ void sonium_csr_jacobi_f32(
    const int* rows, const int* cols, const float* values, const float* inverse,
    const float* b, const float* x, float* output, unsigned n, unsigned nnz, double omega) {
    csr_jacobi_step(rows, cols, values, inverse, b, x, output, n, nnz, omega);
}
extern "C" __global__ void sonium_csr_jacobi_f64(
    const int* rows, const int* cols, const double* values, const double* inverse,
    const double* b, const double* x, double* output, unsigned n, unsigned nnz, double omega) {
    csr_jacobi_step(rows, cols, values, inverse, b, x, output, n, nnz, omega);
}
extern "C" __global__ void sonium_csr_jacobi_c32(
    const int* rows, const int* cols, const float2* values, const float2* inverse,
    const float2* b, const float2* x, float2* output, unsigned n, unsigned nnz, double omega) {
    csr_jacobi_step(rows, cols, values, inverse, b, x, output, n, nnz, omega);
}
extern "C" __global__ void sonium_csr_jacobi_c64(
    const int* rows, const int* cols, const double2* values, const double2* inverse,
    const double2* b, const double2* x, double2* output, unsigned n, unsigned nnz, double omega) {
    csr_jacobi_step(rows, cols, values, inverse, b, x, output, n, nnz, omega);
}
