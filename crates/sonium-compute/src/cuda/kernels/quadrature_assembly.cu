// One thread owns each output; fixed contribution order avoids floating atomics.
extern "C" __global__ void sonium_quadrature_assemble_c64(
    const unsigned *offsets, const unsigned *samples, const double *weights,
    const double2 *coefficients, double2 *output, unsigned count) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= count) return;
    double2 sum = make_double2(0.0, 0.0);
    for (unsigned j = offsets[row]; j < offsets[row + 1]; ++j) {
        double2 value = coefficients[samples[j]];
        sum.x += weights[j] * value.x;
        sum.y += weights[j] * value.y;
    }
    output[row] = sum;
}
