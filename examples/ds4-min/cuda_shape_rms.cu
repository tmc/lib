// CUDA math functions are NVRTC builtins; no host math header is needed.

// Correctness-first dynamic-width RMS. One block owns one row; a later
// parallel reduction must earn its own numerical gate.
extern "C" __global__ void ds4_shape_rms(const float *x, const float *weight,
                                        float *out, int width, int rows,
                                        float epsilon) {
    if (threadIdx.x != 0 || blockIdx.x >= rows || width <= 0) return;
    int row = blockIdx.x;
    const float *src = x + (unsigned long long)row * width;
    float *dst = out + (unsigned long long)row * width;
    float sum = 0.0f;
    for (int i = 0; i < width; i++) sum += src[i] * src[i];
    float scale = rsqrtf(sum / width + epsilon);
    for (int i = 0; i < width; i++) dst[i] = src[i] * scale * weight[i];
}
