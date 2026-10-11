// expression: "(fused_rmsnorm_gemv a weight constant{1e-05} matrix)"
// fingerprint: "(fused_rmsnorm_gemv a{string:a} weight{string:weight} constant{float32:1e-05} matrix{string:matrix})"
// cost model: "AstSizeCost", cost: 5
// rule suites: []
// extraction stats: work=0 nodes=0 exhausted=false

typedef unsigned char uint8_t;
template <typename T, int N> struct tvec {
    T values[N];
    __device__ T& operator[](int i) { return values[i]; }
    __device__ const T& operator[](int i) const { return values[i]; }
};

// e-class 0: "a", from "input"
__device__ auto eg_c0_a(const tvec<float, 12>& input, const float* __restrict__ params, uint8_t* __restrict__ fwd_ctx = nullptr) -> tvec<float, 3> {
    tvec<float, 3> output;
    for (int i = 0; i < 3; ++i) {
        output[i] = input[0 + i];
    }
    return output;
}

// e-class 5: "weight", from "input"
__device__ auto eg_c5_weight(const tvec<float, 12>& input, const float* __restrict__ params, uint8_t* __restrict__ fwd_ctx = nullptr) -> tvec<float, 3> {
    tvec<float, 3> output;
    for (int i = 0; i < 3; ++i) {
        output[i] = input[9 + i];
    }
    return output;
}

// e-class 7: "constant", from "input"
__device__ auto eg_c7_constant(const tvec<float, 12>& input, const float* __restrict__ params, uint8_t* __restrict__ fwd_ctx = nullptr) -> tvec<float, 1> {
    tvec<float, 1> output;
    output[0] = 1e-05f;
    return output;
}

// e-class 6: "matrix", from "input"
__device__ auto eg_c6_matrix(const tvec<float, 12>& input, const float* __restrict__ params, uint8_t* __restrict__ fwd_ctx = nullptr) -> tvec<float, 6> {
    tvec<float, 6> output;
    for (int i = 0; i < 6; ++i) {
        output[i] = input[3 + i];
    }
    return output;
}

// e-class 8: "fused_rmsnorm_gemv", from "input"
__device__ auto eg_c8_fused_rmsnorm_gemv(const tvec<float, 12>& input, const float* __restrict__ params, uint8_t* __restrict__ fwd_ctx = nullptr) -> tvec<float, 2> {
    auto arg0 = eg_c0_a(input, params, fwd_ctx);
    auto arg1 = eg_c5_weight(input, params, fwd_ctx);
    auto arg2 = eg_c7_constant(input, params, fwd_ctx);
    auto arg3 = eg_c6_matrix(input, params, fwd_ctx);
    tvec<float, 2> output;
    float sum = 0.0f;
    for (int i = 0; i < 3; ++i) {
        sum += arg0[i] * arg0[i];
    }
    float scale = 1.0f / sqrtf(sum / 3.0f + arg2[0]);
    for (int row = 0; row < 2; ++row) {
        float dot = 0.0f;
        for (int i = 0; i < 3; ++i) {
            float normalized = (arg0[i] * scale) * arg1[i];
            dot += arg3[row * 3 + i] * normalized;
        }
        output[row] = dot;
    }
    return output;
}

extern "C" __global__ void chain(const float* a, const float* matrix, const float* weight, float* eg_output) {
    if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0 || threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0) {
        return;
    }
    tvec<float, 12> eg_input;
    for (int i = 0; i < 3; ++i) {
        eg_input[0 + i] = a[i];
    }
    for (int i = 0; i < 6; ++i) {
        eg_input[3 + i] = matrix[i];
    }
    for (int i = 0; i < 3; ++i) {
        eg_input[9 + i] = weight[i];
    }
    auto eg_result = eg_c8_fused_rmsnorm_gemv(eg_input, nullptr, nullptr);
    for (int i = 0; i < 2; ++i) {
        eg_output[i] = eg_result[i];
    }
}
