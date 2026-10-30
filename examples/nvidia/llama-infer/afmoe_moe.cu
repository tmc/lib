// Scalar-token AFMoE baseline. BF16 weights are read directly from the original
// shards. Activation buffers are float32 storage for BF16-rounded values.
// Launch projection blocks with exactly128 threads. This source is independent
// of the Llama kernels; no existing fused MoE semantics are assumed.

__device__ __forceinline__ float afmoe_moe_widen(unsigned short x) {
    return __uint_as_float((unsigned int)x << 16);
}

__device__ __forceinline__ float afmoe_moe_round(float x) {
    unsigned int u = __float_as_uint(x);
    if ((u & 0x7f800000u) == 0x7f800000u) {
        if (u & 0x007fffffu) u |= 0x00400000u;
        return __uint_as_float(u & 0xffff0000u);
    }
    u += 0x7fffu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xffff0000u);
}

// Fixed128-thread reduction. Distinct rounding/reduction order from a BLAS
// implementation is an explicit native-fidelity question, not host proof.
__device__ __forceinline__ float afmoe_moe_dot(
    const float* x, const unsigned short* w, int n, float* scratch
) {
    int t = threadIdx.x;
    float sum = 0.0f;
    for (int j = t; j < n; j += 128) {
        sum = __fadd_rn(sum, __fmul_rn(x[j], afmoe_moe_widen(w[j])));
    }
    scratch[t] = sum;
    __syncthreads();
    for (int offset = 64; offset > 0; offset >>= 1) {
        if (t < offset) scratch[t] = __fadd_rn(scratch[t], scratch[t + offset]);
        __syncthreads();
    }
    return scratch[0];
}

extern "C" __global__ void AFMoERouterLinearBF16(
    const float* x, const unsigned short* w, float* logits,
    int hidden, int experts
) {
    int e = blockIdx.x;
    if (blockDim.x != 128 || e >= experts) return;
    __shared__ float scratch[128];
    float sum = afmoe_moe_dot(x, w + (unsigned long long)e * hidden, hidden, scratch);
    if (threadIdx.x == 0) logits[e] = afmoe_moe_round(sum);
}

// Bias affects selection, not normalized weights. Launch one thread/block.
// Exact selection-score ties choose the smaller expert ID; torch.topk does not
// specify this tie policy. Model-specific dimensions are validated by the host.
extern "C" __global__ void AFMoERoute(
    const float* logits, const float* bias, int* ids, float* weights, int* fault,
    int experts, int top_k, int normalize, float epsilon, float scale
) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || experts < 1 || experts > 128 ||
        top_k < 1 || top_k > experts) return;
    float scores[128];
    float selection[128];
    bool selected[128];
    for (int e = 0; e < experts; e++) {
        if (!isfinite(logits[e]) || (bias && !isfinite(bias[e]))) {
            atomicOr(fault, 1);
            return;
        }
        scores[e] = 1.0f / (1.0f + expf(-logits[e]));
        selection[e] = __fadd_rn(scores[e], bias ? bias[e] : 0.0f);
        selected[e] = false;
    }
    float sum = 0.0f;
    for (int k = 0; k < top_k; k++) {
        int best = -1;
        for (int e = 0; e < experts; e++) {
            if (!selected[e] && (best < 0 || selection[e] > selection[best])) best = e;
        }
        ids[k] = best;
        weights[k] = scores[best];
        selected[best] = true;
        sum = __fadd_rn(sum, scores[best]);
    }
    float denominator = __fadd_rn(sum, epsilon);
    for (int k = 0; k < top_k; k++) {
        float weight = normalize ? weights[k] / denominator : weights[k];
        weights[k] = __fmul_rn(weight, scale);
    }
}

__device__ __forceinline__ float afmoe_moe_swiglu(float gate, float up) {
    // Each linear returns BF16. SiLU is one BF16-producing operation, followed
    // by a separate BF16-producing multiply, as in the pinned model source.
    gate = afmoe_moe_round(gate);
    up = afmoe_moe_round(up);
    float silu = afmoe_moe_round(gate / (1.0f + expf(-gate)));
    return afmoe_moe_round(__fmul_rn(silu, up));
}

extern "C" __global__ void AFMoEMLPGateUpBF16(
    const float* x, const unsigned short* gate, const unsigned short* up,
    float* activation, int hidden, int intermediate
) {
    int row = blockIdx.x;
    if (blockDim.x != 128 || row >= intermediate) return;
    __shared__ float scratch[128];
    float g = afmoe_moe_dot(x, gate + (unsigned long long)row * hidden, hidden, scratch);
    // All threads retain g before reusing shared reduction storage.
    __syncthreads();
    float u = afmoe_moe_dot(x, up + (unsigned long long)row * hidden, hidden, scratch);
    if (threadIdx.x == 0) activation[row] = afmoe_moe_swiglu(g, u);
}

extern "C" __global__ void AFMoEMLPDownBF16(
    const float* activation, const unsigned short* down, float* output,
    int intermediate, int hidden
) {
    int row = blockIdx.x;
    if (blockDim.x != 128 || row >= hidden) return;
    __shared__ float scratch[128];
    float sum = afmoe_moe_dot(activation, down + (unsigned long long)row * intermediate,
                             intermediate, scratch);
    if (threadIdx.x == 0) output[row] = afmoe_moe_round(sum);
}

// Pointer tables contain128 device addresses in expert-ID order, with no
// contiguous full-model repacking. grid.y is the selected-route index.
extern "C" __global__ void AFMoESelectedGateUpBF16(
    const float* x, const unsigned long long* gate_table,
    const unsigned long long* up_table, const int* ids, float* activation,
    int hidden, int intermediate, int top_k, int* fault
) {
    int row = blockIdx.x, route = blockIdx.y;
    if (blockDim.x != 128 || row >= intermediate || route >= top_k) return;
    int expert = ids[route];
    if (expert < 0 || expert >= 128) {
        if (threadIdx.x == 0) atomicOr(fault, 2);
        return;
    }
    const unsigned short* gate = reinterpret_cast<const unsigned short*>(gate_table[expert]);
    const unsigned short* up = reinterpret_cast<const unsigned short*>(up_table[expert]);
    __shared__ float scratch[128];
    float g = afmoe_moe_dot(x, gate + (unsigned long long)row * hidden, hidden, scratch);
    __syncthreads();
    float u = afmoe_moe_dot(x, up + (unsigned long long)row * hidden, hidden, scratch);
    if (threadIdx.x == 0) {
        activation[(unsigned long long)route * intermediate + row] = afmoe_moe_swiglu(g, u);
    }
}

extern "C" __global__ void AFMoESelectedDownBF16(
    const float* activation, const unsigned long long* down_table,
    const int* ids, float* output, int intermediate, int hidden, int top_k, int* fault
) {
    int row = blockIdx.x, route = blockIdx.y;
    if (blockDim.x != 128 || row >= hidden || route >= top_k) return;
    if (ids[route] < 0 || ids[route] >= 128) {
        if (threadIdx.x == 0) atomicOr(fault, 2);
        return;
    }
    const unsigned short* down = reinterpret_cast<const unsigned short*>(down_table[ids[route]]);
    __shared__ float scratch[128];
    float sum = afmoe_moe_dot(activation + (unsigned long long)route * intermediate,
                             down + (unsigned long long)row * intermediate,
                             intermediate, scratch);
    if (threadIdx.x == 0) output[(unsigned long long)route * hidden + row] = afmoe_moe_round(sum);
}

// wide=0 rounds every expert-ID-ordered addition to BF16; wide=1 accumulates
// the already BF16-weighted contributions in FP32 and rounds the final sum.
// Neither is asserted to be the universal scatter_add policy of all backends.
extern "C" __global__ void AFMoECombine(
    const float* shared, const float* outputs, const int* ids,
    const float* weights, float* output, int hidden, int top_k, int wide, int* fault
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= hidden || top_k < 1 || top_k > 128) return;
    float sum = shared[j];
    if (!isfinite(sum)) { atomicOr(fault, 4); return; }
    int previous = -1;
    for (int k = 0; k < top_k; k++) {
        int best = -1;
        for (int r = 0; r < top_k; r++) {
            if (ids[r] < 0 || ids[r] >= 128 || !isfinite(weights[r])) {
                atomicOr(fault, 2);
                return;
            }
            if (ids[r] > previous && (best < 0 || ids[r] < ids[best])) best = r;
        }
        if (best < 0) { atomicOr(fault, 2); return; }
        float contribution = afmoe_moe_round(__fmul_rn(outputs[(unsigned long long)best * hidden + j], weights[best]));
        sum = __fadd_rn(sum, contribution);
        if (!wide) sum = afmoe_moe_round(sum);
        if (!isfinite(sum)) { atomicOr(fault, 4); return; }
        previous = ids[best];
    }
    output[j] = afmoe_moe_round(sum);
}

// Every composite MoE emission starts with this kernel on the same ordered
// execution stream. The caller must inspect fault after synchronization.
extern "C" __global__ void AFMoEClearFault(int* fault) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *fault = 0;
}
