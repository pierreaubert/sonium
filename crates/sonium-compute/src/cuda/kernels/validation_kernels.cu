// Device-side validation for native compute inputs. Kernels write a single
// status word and never modify input arrays or operation destinations.

__device__ bool sonium_finite_component(const unsigned char* address,
                                        unsigned long long width) {
    if (width == 4) {
        const float value = *reinterpret_cast<const float*>(address);
        return isfinite(value);
    }
    if (width == 8) {
        const double value = *reinterpret_cast<const double*>(address);
        return isfinite(value);
    }
    return false;
}

extern "C" __global__ void sonium_validate_finite(
    const unsigned char* values,
    unsigned long long count,
    unsigned long long value_size,
    unsigned long long real_size,
    int* status) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const unsigned char* value = values + index * value_size;
    if (!sonium_finite_component(value, real_size)
        || (value_size != real_size
            && !sonium_finite_component(value + real_size, real_size))) {
        atomicExch(status, 1);
    }
}

extern "C" __global__ void sonium_validate_gather4(
    const unsigned char* values,
    const int* indices,
    const unsigned char* weights,
    unsigned long long value_count,
    unsigned long long index_count,
    unsigned long long value_size,
    unsigned long long weight_size,
    unsigned long long real_size,
    int* status) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < index_count) {
        const int value_index = indices[index];
        if (value_index < 0 || static_cast<unsigned long long>(value_index) >= value_count) {
            atomicMax(status, 1);
        }
        if (!sonium_finite_component(weights + index * weight_size, real_size)) {
            atomicMax(status, 2);
        }
    }
    if (index < value_count) {
        const unsigned char* value = values + index * value_size;
        if (!sonium_finite_component(value, real_size)
            || (value_size != real_size
                && !sonium_finite_component(value + real_size, real_size))) {
            atomicMax(status, 2);
        }
    }
}

extern "C" __global__ void sonium_validate_segmented_offsets(
    const int* offsets,
    unsigned long long value_count,
    unsigned long long offset_count,
    int* status) {
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= offset_count) return;
    const int current = offsets[index];
    if (current < 0 || static_cast<unsigned long long>(current) > value_count
        || (index > 0 && current < offsets[index - 1])) {
        atomicExch(status, 1);
    }
}
