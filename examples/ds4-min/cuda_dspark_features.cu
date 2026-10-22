// Selected post-FFN target HC means: ds4.c:29222 and hc_weighted_sum_kernel.
// Lane-major HC, deterministic four-lane order. No target math is changed.
extern "C" __global__ void ds4_dspark_capture_feature(
    const float *hc, float *features, unsigned int *status, unsigned int slot) {
    unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= 4096) return;
    float mean = 0.0f;
    for (unsigned int lane = 0; lane < 4; lane++) {
        mean += hc[lane * 4096 + d] * 0.25f;
    }
    features[slot * 4096 + d] = mean;
    if (!isfinite(mean)) atomicOr(status, 1u);
}
