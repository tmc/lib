// Standalone NVRTC source: no host system headers are required for integers.
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
static_assert(sizeof(uint32_t) == 4 && sizeof(uint64_t) == 8, "integer widths");
static_assert(uint32_t(-1) > 0 && uint64_t(-1) > 0, "integer signedness");

// ds4_cuda.cu:11940 at 11e8cad, entry name only changed.
extern "C" __global__ void ds4_shape_output_hc_weights(
        float *out,
        const float *pre,
        const float *scale,
        const float *base,
        uint32_t n_hc,
        uint32_t n_tokens,
        float epsv) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = n_tokens * n_hc;
    if (gid >= n) return;
    uint32_t h = gid % n_hc;
    float z = pre[gid] * scale[0] + base[h];
    out[gid] = 1.0f / (1.0f + expf(-z)) + epsv;
}
