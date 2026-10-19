// Norm-folded and epilogue-fused decode kernels (Slice A2, 5 kernels/layer).
// This file is compiled after forward.cu and forward_bf16.cu in the same NVRTC
// translation unit and uses their warp_sum, rows_vectorizable, block_reduce_sum,
// and bf16_to_f32 device helpers.

// rows_vectorizable(row_len, row) reports whether both the row stride and the
// specific row offset are 4-element aligned for float4/ushort4 loads.
__device__ __forceinline__ bool rows_vectorizable(int row_len, int row) {
    return rows_vectorizable(row_len) && (((row * row_len) & 3) == 0);
}

// 1. FusedRMSNormQKVRoPEStore folds pre-attention RMSNorm(x, rms_att_w, eps)
// into Q, K, and V linear projections (+ optional bq, bk, bv biases), applies
// split-half RoPE at pos (rope_theta, head_dim), writes q_out[0..q_dim), and
// stores rotated K and V directly into k_cache[pos * kv_dim ..] and
// v_cache[pos * kv_dim ..].
//
// Grid mapping:
//   qPairs  = q_heads * (head_dim / 2)
//   kvPairs = kv_heads * (head_dim / 2)
//   gridDim.x = qPairs + kvPairs
//   - blockIdx.x < qPairs: computes Q pair (row0, row1 = row0 + head_dim/2),
//     applies split-half RoPE, writes q_out[row0] and q_out[row1].
//   - blockIdx.x >= qPairs: computes K pair and V pair (row0, row1), applies
//     split-half RoPE to K, and writes k_cache and v_cache at pos * kv_dim.
extern "C" __global__ void FusedRMSNormQKVRoPEStore(
    const float* __restrict__ x,
    const float* __restrict__ rms_att_w,
    const float* __restrict__ wq,
    const float* __restrict__ wk,
    const float* __restrict__ wv,
    const float* __restrict__ bq,
    const float* __restrict__ bk,
    const float* __restrict__ bv,
    float* __restrict__ q_out,
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    int in_dim,
    int q_heads,
    int kv_heads,
    int head_dim,
    int pos,
    float eps,
    float rope_theta
) {
    if (head_dim <= 0 || (head_dim & 1) != 0) return;
    int half_dim = head_dim / 2;
    int q_pairs = q_heads * half_dim;
    int kv_pairs = kv_heads * half_dim;
    int pair_idx = blockIdx.x;
    if (pair_idx >= q_pairs + kv_pairs) return;

    int tid = threadIdx.x;
    __shared__ float s_sq[32];
    __shared__ float s_r0[32];
    __shared__ float s_r1[32];
    __shared__ float s_r2[32];
    __shared__ float s_r3[32];
    __shared__ float s_inv_rms;

    int sq_dim4 = rows_vectorizable(in_dim) ? in_dim / 4 : 0;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float sum_sq = 0.0f;
    for (int i = tid; i < sq_dim4; i += blockDim.x) {
        float4 xv = x4[i];
        sum_sq += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
    }
    for (int i = sq_dim4 * 4 + tid; i < in_dim; i += blockDim.x) {
        float xv = x[i];
        sum_sq += xv * xv;
    }
    sum_sq = block_reduce_sum(sum_sq, s_sq);
    if (tid == 0) {
        s_inv_rms = rsqrtf(sum_sq / float(in_dim) + eps);
    }
    __syncthreads();
    float inv_rms = s_inv_rms;

    const float4* g4 = rms_att_w ? reinterpret_cast<const float4*>(rms_att_w) : nullptr;

    if (pair_idx < q_pairs) {
        int h = pair_idx / half_dim;
        int d = pair_idx % half_dim;
        int row0 = h * head_dim + d;
        int row1 = row0 + half_dim;

        const float* wq0 = wq + (size_t)row0 * in_dim;
        const float* wq1 = wq + (size_t)row1 * in_dim;
        const float4* wq0_4 = reinterpret_cast<const float4*>(wq0);
        const float4* wq1_4 = reinterpret_cast<const float4*>(wq1);
        int in_dim4 = (rows_vectorizable(in_dim, row0) && rows_vectorizable(in_dim, row1)) ? in_dim / 4 : 0;

        float sum_q0 = 0.0f;
        float sum_q1 = 0.0f;
        for (int i = tid; i < in_dim4; i += blockDim.x) {
            float4 xv = x4[i];
            float4 gv = g4 ? g4[i] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            float4 nx = make_float4(
                xv.x * inv_rms * gv.x,
                xv.y * inv_rms * gv.y,
                xv.z * inv_rms * gv.z,
                xv.w * inv_rms * gv.w
            );
            float4 w0 = wq0_4[i];
            float4 w1 = wq1_4[i];
            sum_q0 += nx.x * w0.x + nx.y * w0.y + nx.z * w0.z + nx.w * w0.w;
            sum_q1 += nx.x * w1.x + nx.y * w1.y + nx.z * w1.z + nx.w * w1.w;
        }
        for (int i = in_dim4 * 4 + tid; i < in_dim; i += blockDim.x) {
            float gv = rms_att_w ? rms_att_w[i] : 1.0f;
            float nx = x[i] * inv_rms * gv;
            sum_q0 += nx * wq0[i];
            sum_q1 += nx * wq1[i];
        }
        sum_q0 = block_reduce_sum(sum_q0, s_r0);
        sum_q1 = block_reduce_sum(sum_q1, s_r1);
        if (tid == 0) {
            float q0 = sum_q0 + (bq ? bq[row0] : 0.0f);
            float q1 = sum_q1 + (bq ? bq[row1] : 0.0f);
            float freq = rope_inv_freq(d, head_dim, rope_theta, 0.0f, 0.0f, 0.0f, 0);
            float val = float(pos) * freq;
            float cos_val = cosf(val);
            float sin_val = sinf(val);
            q_out[row0] = q0 * cos_val - q1 * sin_val;
            q_out[row1] = q0 * sin_val + q1 * cos_val;
        }
        return;
    }

    int kv_pair = pair_idx - q_pairs;
    int h = kv_pair / half_dim;
    int d = kv_pair % half_dim;
    int row0 = h * head_dim + d;
    int row1 = row0 + half_dim;

    const float* wk0 = wk + (size_t)row0 * in_dim;
    const float* wk1 = wk + (size_t)row1 * in_dim;
    const float* wv0 = wv + (size_t)row0 * in_dim;
    const float* wv1 = wv + (size_t)row1 * in_dim;
    const float4* wk0_4 = reinterpret_cast<const float4*>(wk0);
    const float4* wk1_4 = reinterpret_cast<const float4*>(wk1);
    const float4* wv0_4 = reinterpret_cast<const float4*>(wv0);
    const float4* wv1_4 = reinterpret_cast<const float4*>(wv1);
    int in_dim4 = (rows_vectorizable(in_dim, row0) && rows_vectorizable(in_dim, row1)) ? in_dim / 4 : 0;

    float sum_k0 = 0.0f;
    float sum_k1 = 0.0f;
    float sum_v0 = 0.0f;
    float sum_v1 = 0.0f;
    for (int i = tid; i < in_dim4; i += blockDim.x) {
        float4 xv = x4[i];
        float4 gv = g4 ? g4[i] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        float4 nx = make_float4(
            xv.x * inv_rms * gv.x,
            xv.y * inv_rms * gv.y,
            xv.z * inv_rms * gv.z,
            xv.w * inv_rms * gv.w
        );
        float4 wk_0 = wk0_4[i];
        float4 wk_1 = wk1_4[i];
        float4 wv_0 = wv0_4[i];
        float4 wv_1 = wv1_4[i];
        sum_k0 += nx.x * wk_0.x + nx.y * wk_0.y + nx.z * wk_0.z + nx.w * wk_0.w;
        sum_k1 += nx.x * wk_1.x + nx.y * wk_1.y + nx.z * wk_1.z + nx.w * wk_1.w;
        sum_v0 += nx.x * wv_0.x + nx.y * wv_0.y + nx.z * wv_0.z + nx.w * wv_0.w;
        sum_v1 += nx.x * wv_1.x + nx.y * wv_1.y + nx.z * wv_1.z + nx.w * wv_1.w;
    }
    for (int i = in_dim4 * 4 + tid; i < in_dim; i += blockDim.x) {
        float gv = rms_att_w ? rms_att_w[i] : 1.0f;
        float nx = x[i] * inv_rms * gv;
        sum_k0 += nx * wk0[i];
        sum_k1 += nx * wk1[i];
        sum_v0 += nx * wv0[i];
        sum_v1 += nx * wv1[i];
    }
    sum_k0 = block_reduce_sum(sum_k0, s_r0);
    sum_k1 = block_reduce_sum(sum_k1, s_r1);
    sum_v0 = block_reduce_sum(sum_v0, s_r2);
    sum_v1 = block_reduce_sum(sum_v1, s_r3);
    if (tid == 0) {
        float k0 = sum_k0 + (bk ? bk[row0] : 0.0f);
        float k1 = sum_k1 + (bk ? bk[row1] : 0.0f);
        float v0 = sum_v0 + (bv ? bv[row0] : 0.0f);
        float v1 = sum_v1 + (bv ? bv[row1] : 0.0f);
        float freq = rope_inv_freq(d, head_dim, rope_theta, 0.0f, 0.0f, 0.0f, 0);
        float val = float(pos) * freq;
        float cos_val = cosf(val);
        float sin_val = sinf(val);
        int kv_dim = kv_heads * head_dim;
        size_t base = (size_t)pos * kv_dim;
        k_cache[base + row0] = k0 * cos_val - k1 * sin_val;
        k_cache[base + row1] = k0 * sin_val + k1 * cos_val;
        v_cache[base + row0] = v0;
        v_cache[base + row1] = v1;
    }
}

// FusedRMSNormQKVRoPEStoreBF16 is FusedRMSNormQKVRoPEStore with bfloat16
// projection weights. It keeps the exact thread partition, vector-path gate,
// and expression order of FusedRMSNormQKVRoPEStore so that on widened BF16
// weights the two produce bit-identical outputs.
extern "C" __global__ void FusedRMSNormQKVRoPEStoreBF16(
    const float* __restrict__ x,
    const float* __restrict__ rms_att_w,
    const unsigned short* __restrict__ wq,
    const unsigned short* __restrict__ wk,
    const unsigned short* __restrict__ wv,
    const float* __restrict__ bq,
    const float* __restrict__ bk,
    const float* __restrict__ bv,
    float* __restrict__ q_out,
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    int in_dim,
    int q_heads,
    int kv_heads,
    int head_dim,
    int pos,
    float eps,
    float rope_theta
) {
    if (head_dim <= 0 || (head_dim & 1) != 0) return;
    int half_dim = head_dim / 2;
    int q_pairs = q_heads * half_dim;
    int kv_pairs = kv_heads * half_dim;
    int pair_idx = blockIdx.x;
    if (pair_idx >= q_pairs + kv_pairs) return;

    int tid = threadIdx.x;
    __shared__ float s_sq[32];
    __shared__ float s_r0[32];
    __shared__ float s_r1[32];
    __shared__ float s_r2[32];
    __shared__ float s_r3[32];
    __shared__ float s_inv_rms;

    int sq_dim4 = rows_vectorizable(in_dim) ? in_dim / 4 : 0;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float sum_sq = 0.0f;
    for (int i = tid; i < sq_dim4; i += blockDim.x) {
        float4 xv = x4[i];
        sum_sq += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
    }
    for (int i = sq_dim4 * 4 + tid; i < in_dim; i += blockDim.x) {
        float xv = x[i];
        sum_sq += xv * xv;
    }
    sum_sq = block_reduce_sum(sum_sq, s_sq);
    if (tid == 0) {
        s_inv_rms = rsqrtf(sum_sq / float(in_dim) + eps);
    }
    __syncthreads();
    float inv_rms = s_inv_rms;

    const float4* g4 = rms_att_w ? reinterpret_cast<const float4*>(rms_att_w) : nullptr;

    if (pair_idx < q_pairs) {
        int h = pair_idx / half_dim;
        int d = pair_idx % half_dim;
        int row0 = h * head_dim + d;
        int row1 = row0 + half_dim;

        const unsigned short* wq0 = wq + (size_t)row0 * in_dim;
        const unsigned short* wq1 = wq + (size_t)row1 * in_dim;
        const ushort4* wq0_4 = reinterpret_cast<const ushort4*>(wq0);
        const ushort4* wq1_4 = reinterpret_cast<const ushort4*>(wq1);
        int in_dim4 = (rows_vectorizable(in_dim, row0) && rows_vectorizable(in_dim, row1)) ? in_dim / 4 : 0;

        float sum_q0 = 0.0f;
        float sum_q1 = 0.0f;
        for (int i = tid; i < in_dim4; i += blockDim.x) {
            float4 xv = x4[i];
            float4 gv = g4 ? g4[i] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            float4 nx = make_float4(
                xv.x * inv_rms * gv.x,
                xv.y * inv_rms * gv.y,
                xv.z * inv_rms * gv.z,
                xv.w * inv_rms * gv.w
            );
            ushort4 uq0 = wq0_4[i];
            ushort4 uq1 = wq1_4[i];
            float4 w0 = make_float4(bf16_to_f32(uq0.x), bf16_to_f32(uq0.y), bf16_to_f32(uq0.z), bf16_to_f32(uq0.w));
            float4 w1 = make_float4(bf16_to_f32(uq1.x), bf16_to_f32(uq1.y), bf16_to_f32(uq1.z), bf16_to_f32(uq1.w));
            sum_q0 += nx.x * w0.x + nx.y * w0.y + nx.z * w0.z + nx.w * w0.w;
            sum_q1 += nx.x * w1.x + nx.y * w1.y + nx.z * w1.z + nx.w * w1.w;
        }
        for (int i = in_dim4 * 4 + tid; i < in_dim; i += blockDim.x) {
            float gv = rms_att_w ? rms_att_w[i] : 1.0f;
            float nx = x[i] * inv_rms * gv;
            sum_q0 += nx * bf16_to_f32(wq0[i]);
            sum_q1 += nx * bf16_to_f32(wq1[i]);
        }
        sum_q0 = block_reduce_sum(sum_q0, s_r0);
        sum_q1 = block_reduce_sum(sum_q1, s_r1);
        if (tid == 0) {
            float q0 = sum_q0 + (bq ? bq[row0] : 0.0f);
            float q1 = sum_q1 + (bq ? bq[row1] : 0.0f);
            float freq = rope_inv_freq(d, head_dim, rope_theta, 0.0f, 0.0f, 0.0f, 0);
            float val = float(pos) * freq;
            float cos_val = cosf(val);
            float sin_val = sinf(val);
            q_out[row0] = q0 * cos_val - q1 * sin_val;
            q_out[row1] = q0 * sin_val + q1 * cos_val;
        }
        return;
    }

    int kv_pair = pair_idx - q_pairs;
    int h = kv_pair / half_dim;
    int d = kv_pair % half_dim;
    int row0 = h * head_dim + d;
    int row1 = row0 + half_dim;

    const unsigned short* wk0 = wk + (size_t)row0 * in_dim;
    const unsigned short* wk1 = wk + (size_t)row1 * in_dim;
    const unsigned short* wv0 = wv + (size_t)row0 * in_dim;
    const unsigned short* wv1 = wv + (size_t)row1 * in_dim;
    const ushort4* wk0_4 = reinterpret_cast<const ushort4*>(wk0);
    const ushort4* wk1_4 = reinterpret_cast<const ushort4*>(wk1);
    const ushort4* wv0_4 = reinterpret_cast<const ushort4*>(wv0);
    const ushort4* wv1_4 = reinterpret_cast<const ushort4*>(wv1);
    int in_dim4 = (rows_vectorizable(in_dim, row0) && rows_vectorizable(in_dim, row1)) ? in_dim / 4 : 0;

    float sum_k0 = 0.0f;
    float sum_k1 = 0.0f;
    float sum_v0 = 0.0f;
    float sum_v1 = 0.0f;
    for (int i = tid; i < in_dim4; i += blockDim.x) {
        float4 xv = x4[i];
        float4 gv = g4 ? g4[i] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        float4 nx = make_float4(
            xv.x * inv_rms * gv.x,
            xv.y * inv_rms * gv.y,
            xv.z * inv_rms * gv.z,
            xv.w * inv_rms * gv.w
        );
        ushort4 uk0 = wk0_4[i];
        ushort4 uk1 = wk1_4[i];
        ushort4 uv0 = wv0_4[i];
        ushort4 uv1 = wv1_4[i];
        float4 wk_0 = make_float4(bf16_to_f32(uk0.x), bf16_to_f32(uk0.y), bf16_to_f32(uk0.z), bf16_to_f32(uk0.w));
        float4 wk_1 = make_float4(bf16_to_f32(uk1.x), bf16_to_f32(uk1.y), bf16_to_f32(uk1.z), bf16_to_f32(uk1.w));
        float4 wv_0 = make_float4(bf16_to_f32(uv0.x), bf16_to_f32(uv0.y), bf16_to_f32(uv0.z), bf16_to_f32(uv0.w));
        float4 wv_1 = make_float4(bf16_to_f32(uv1.x), bf16_to_f32(uv1.y), bf16_to_f32(uv1.z), bf16_to_f32(uv1.w));
        sum_k0 += nx.x * wk_0.x + nx.y * wk_0.y + nx.z * wk_0.z + nx.w * wk_0.w;
        sum_k1 += nx.x * wk_1.x + nx.y * wk_1.y + nx.z * wk_1.z + nx.w * wk_1.w;
        sum_v0 += nx.x * wv_0.x + nx.y * wv_0.y + nx.z * wv_0.z + nx.w * wv_0.w;
        sum_v1 += nx.x * wv_1.x + nx.y * wv_1.y + nx.z * wv_1.z + nx.w * wv_1.w;
    }
    for (int i = in_dim4 * 4 + tid; i < in_dim; i += blockDim.x) {
        float gv = rms_att_w ? rms_att_w[i] : 1.0f;
        float nx = x[i] * inv_rms * gv;
        sum_k0 += nx * bf16_to_f32(wk0[i]);
        sum_k1 += nx * bf16_to_f32(wk1[i]);
        sum_v0 += nx * bf16_to_f32(wv0[i]);
        sum_v1 += nx * bf16_to_f32(wv1[i]);
    }
    sum_k0 = block_reduce_sum(sum_k0, s_r0);
    sum_k1 = block_reduce_sum(sum_k1, s_r1);
    sum_v0 = block_reduce_sum(sum_v0, s_r2);
    sum_v1 = block_reduce_sum(sum_v1, s_r3);
    if (tid == 0) {
        float k0 = sum_k0 + (bk ? bk[row0] : 0.0f);
        float k1 = sum_k1 + (bk ? bk[row1] : 0.0f);
        float v0 = sum_v0 + (bv ? bv[row0] : 0.0f);
        float v1 = sum_v1 + (bv ? bv[row1] : 0.0f);
        float freq = rope_inv_freq(d, head_dim, rope_theta, 0.0f, 0.0f, 0.0f, 0);
        float val = float(pos) * freq;
        float cos_val = cosf(val);
        float sin_val = sinf(val);
        int kv_dim = kv_heads * head_dim;
        size_t base = (size_t)pos * kv_dim;
        k_cache[base + row0] = k0 * cos_val - k1 * sin_val;
        k_cache[base + row1] = k0 * sin_val + k1 * cos_val;
        v_cache[base + row0] = v0;
        v_cache[base + row1] = v1;
    }
}

// 2. FusedOutProjResidual computes out[row] = residual[row] + dot(w_o[row], attn_out)
// with one CTA per output row in [0, out_dim).
extern "C" __global__ void FusedOutProjResidual(
    const float* __restrict__ attn_out,
    const float* residual,
    const float* __restrict__ w_o,
    float* out,
    int in_dim,
    int out_dim
) {
    int row = blockIdx.x;
    if (row >= out_dim) return;

    int tid = threadIdx.x;
    __shared__ float s_sum[32];

    const float* w_row = w_o + (size_t)row * in_dim;
    const float4* w_row4 = reinterpret_cast<const float4*>(w_row);
    const float4* a4 = reinterpret_cast<const float4*>(attn_out);
    int in_dim4 = rows_vectorizable(in_dim, row) ? in_dim / 4 : 0;

    float sum = 0.0f;
    for (int i = tid; i < in_dim4; i += blockDim.x) {
        float4 av = a4[i];
        float4 wv = w_row4[i];
        sum += av.x * wv.x + av.y * wv.y + av.z * wv.z + av.w * wv.w;
    }
    for (int i = in_dim4 * 4 + tid; i < in_dim; i += blockDim.x) {
        sum += attn_out[i] * w_row[i];
    }
    sum = block_reduce_sum(sum, s_sum);
    if (tid == 0) {
        out[row] = residual[row] + sum;
    }
}

// FusedOutProjResidualBF16 is FusedOutProjResidual with bfloat16 output
// projection weights.
extern "C" __global__ void FusedOutProjResidualBF16(
    const float* __restrict__ attn_out,
    const float* residual,
    const unsigned short* __restrict__ w_o,
    float* out,
    int in_dim,
    int out_dim
) {
    int row = blockIdx.x;
    if (row >= out_dim) return;

    int tid = threadIdx.x;
    __shared__ float s_sum[32];

    const unsigned short* w_row = w_o + (size_t)row * in_dim;
    const ushort4* w_row4 = reinterpret_cast<const ushort4*>(w_row);
    const float4* a4 = reinterpret_cast<const float4*>(attn_out);
    int in_dim4 = rows_vectorizable(in_dim, row) ? in_dim / 4 : 0;

    float sum = 0.0f;
    for (int i = tid; i < in_dim4; i += blockDim.x) {
        float4 av = a4[i];
        ushort4 uo = w_row4[i];
        float4 wv = make_float4(bf16_to_f32(uo.x), bf16_to_f32(uo.y), bf16_to_f32(uo.z), bf16_to_f32(uo.w));
        sum += av.x * wv.x + av.y * wv.y + av.z * wv.z + av.w * wv.w;
    }
    for (int i = in_dim4 * 4 + tid; i < in_dim; i += blockDim.x) {
        sum += attn_out[i] * bf16_to_f32(w_row[i]);
    }
    sum = block_reduce_sum(sum, s_sum);
    if (tid == 0) {
        out[row] = residual[row] + sum;
    }
}

// 3. FusedRMSNormGateUpSwiGLU folds post-attention RMSNorm(x, rms_ffn_w, eps)
// into simultaneous gate and up projections and applies SwiGLU in one kernel,
// with one CTA per intermediate row in [0, inter_dim).
extern "C" __global__ void FusedRMSNormGateUpSwiGLU(
    const float* __restrict__ x,
    const float* __restrict__ rms_ffn_w,
    const float* __restrict__ w_gate,
    const float* __restrict__ w_up,
    float* __restrict__ out_hb,
    int hidden_dim,
    int inter_dim,
    float eps
) {
    int row = blockIdx.x;
    if (row >= inter_dim) return;

    int tid = threadIdx.x;
    __shared__ float s_sq[32];
    __shared__ float s_gate[32];
    __shared__ float s_up[32];
    __shared__ float s_inv_rms;

    int sq_dim4 = rows_vectorizable(hidden_dim) ? hidden_dim / 4 : 0;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float sum_sq = 0.0f;
    for (int i = tid; i < sq_dim4; i += blockDim.x) {
        float4 xv = x4[i];
        sum_sq += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
    }
    for (int i = sq_dim4 * 4 + tid; i < hidden_dim; i += blockDim.x) {
        float xv = x[i];
        sum_sq += xv * xv;
    }
    sum_sq = block_reduce_sum(sum_sq, s_sq);
    if (tid == 0) {
        s_inv_rms = rsqrtf(sum_sq / float(hidden_dim) + eps);
    }
    __syncthreads();
    float inv_rms = s_inv_rms;

    const float* wg_row = w_gate + (size_t)row * hidden_dim;
    const float* wu_row = w_up + (size_t)row * hidden_dim;
    const float4* g4 = rms_ffn_w ? reinterpret_cast<const float4*>(rms_ffn_w) : nullptr;
    const float4* wg4 = reinterpret_cast<const float4*>(wg_row);
    const float4* wu4 = reinterpret_cast<const float4*>(wu_row);
    int hidden_dim4 = rows_vectorizable(hidden_dim, row) ? hidden_dim / 4 : 0;

    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (int i = tid; i < hidden_dim4; i += blockDim.x) {
        float4 xv = x4[i];
        float4 gv = g4 ? g4[i] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        float4 nx = make_float4(
            xv.x * inv_rms * gv.x,
            xv.y * inv_rms * gv.y,
            xv.z * inv_rms * gv.z,
            xv.w * inv_rms * gv.w
        );
        float4 wg = wg4[i];
        float4 wu = wu4[i];
        gate_sum += nx.x * wg.x + nx.y * wg.y + nx.z * wg.z + nx.w * wg.w;
        up_sum += nx.x * wu.x + nx.y * wu.y + nx.z * wu.z + nx.w * wu.w;
    }
    for (int i = hidden_dim4 * 4 + tid; i < hidden_dim; i += blockDim.x) {
        float gv = rms_ffn_w ? rms_ffn_w[i] : 1.0f;
        float nx = x[i] * inv_rms * gv;
        gate_sum += nx * wg_row[i];
        up_sum += nx * wu_row[i];
    }
    gate_sum = block_reduce_sum(gate_sum, s_gate);
    up_sum = block_reduce_sum(up_sum, s_up);
    if (tid == 0) {
        out_hb[row] = (gate_sum / (1.0f + expf(-gate_sum))) * up_sum;
    }
}

// FusedRMSNormGateUpSwiGLUBF16 is FusedRMSNormGateUpSwiGLU with bfloat16 gate
// and up projection weights.
extern "C" __global__ void FusedRMSNormGateUpSwiGLUBF16(
    const float* __restrict__ x,
    const float* __restrict__ rms_ffn_w,
    const unsigned short* __restrict__ w_gate,
    const unsigned short* __restrict__ w_up,
    float* __restrict__ out_hb,
    int hidden_dim,
    int inter_dim,
    float eps
) {
    int row = blockIdx.x;
    if (row >= inter_dim) return;

    int tid = threadIdx.x;
    __shared__ float s_sq[32];
    __shared__ float s_gate[32];
    __shared__ float s_up[32];
    __shared__ float s_inv_rms;

    int sq_dim4 = rows_vectorizable(hidden_dim) ? hidden_dim / 4 : 0;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float sum_sq = 0.0f;
    for (int i = tid; i < sq_dim4; i += blockDim.x) {
        float4 xv = x4[i];
        sum_sq += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
    }
    for (int i = sq_dim4 * 4 + tid; i < hidden_dim; i += blockDim.x) {
        float xv = x[i];
        sum_sq += xv * xv;
    }
    sum_sq = block_reduce_sum(sum_sq, s_sq);
    if (tid == 0) {
        s_inv_rms = rsqrtf(sum_sq / float(hidden_dim) + eps);
    }
    __syncthreads();
    float inv_rms = s_inv_rms;

    const unsigned short* wg_row = w_gate + (size_t)row * hidden_dim;
    const unsigned short* wu_row = w_up + (size_t)row * hidden_dim;
    const float4* g4 = rms_ffn_w ? reinterpret_cast<const float4*>(rms_ffn_w) : nullptr;
    const ushort4* wg4 = reinterpret_cast<const ushort4*>(wg_row);
    const ushort4* wu4 = reinterpret_cast<const ushort4*>(wu_row);
    int hidden_dim4 = rows_vectorizable(hidden_dim, row) ? hidden_dim / 4 : 0;

    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (int i = tid; i < hidden_dim4; i += blockDim.x) {
        float4 xv = x4[i];
        float4 gv = g4 ? g4[i] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        float4 nx = make_float4(
            xv.x * inv_rms * gv.x,
            xv.y * inv_rms * gv.y,
            xv.z * inv_rms * gv.z,
            xv.w * inv_rms * gv.w
        );
        ushort4 ugate = wg4[i];
        ushort4 uup = wu4[i];
        float4 wg = make_float4(bf16_to_f32(ugate.x), bf16_to_f32(ugate.y), bf16_to_f32(ugate.z), bf16_to_f32(ugate.w));
        float4 wu = make_float4(bf16_to_f32(uup.x), bf16_to_f32(uup.y), bf16_to_f32(uup.z), bf16_to_f32(uup.w));
        gate_sum += nx.x * wg.x + nx.y * wg.y + nx.z * wg.z + nx.w * wg.w;
        up_sum += nx.x * wu.x + nx.y * wu.y + nx.z * wu.z + nx.w * wu.w;
    }
    for (int i = hidden_dim4 * 4 + tid; i < hidden_dim; i += blockDim.x) {
        float gv = rms_ffn_w ? rms_ffn_w[i] : 1.0f;
        float nx = x[i] * inv_rms * gv;
        gate_sum += nx * bf16_to_f32(wg_row[i]);
        up_sum += nx * bf16_to_f32(wu_row[i]);
    }
    gate_sum = block_reduce_sum(gate_sum, s_gate);
    up_sum = block_reduce_sum(up_sum, s_up);
    if (tid == 0) {
        out_hb[row] = (gate_sum / (1.0f + expf(-gate_sum))) * up_sum;
    }
}

// 4. FusedDownProjResidual computes out[row] = residual[row] + dot(w_down[row], hb)
// with one CTA per hidden row in [0, hidden_dim).
extern "C" __global__ void FusedDownProjResidual(
    const float* __restrict__ hb,
    const float* residual,
    const float* __restrict__ w_down,
    float* out,
    int inter_dim,
    int hidden_dim
) {
    int row = blockIdx.x;
    if (row >= hidden_dim) return;

    int tid = threadIdx.x;
    __shared__ float s_sum[32];

    const float* w_row = w_down + (size_t)row * inter_dim;
    const float4* w_row4 = reinterpret_cast<const float4*>(w_row);
    const float4* hb4 = reinterpret_cast<const float4*>(hb);
    int inter_dim4 = rows_vectorizable(inter_dim, row) ? inter_dim / 4 : 0;

    float sum = 0.0f;
    for (int i = tid; i < inter_dim4; i += blockDim.x) {
        float4 hv = hb4[i];
        float4 wv = w_row4[i];
        sum += hv.x * wv.x + hv.y * wv.y + hv.z * wv.z + hv.w * wv.w;
    }
    for (int i = inter_dim4 * 4 + tid; i < inter_dim; i += blockDim.x) {
        sum += hb[i] * w_row[i];
    }
    sum = block_reduce_sum(sum, s_sum);
    if (tid == 0) {
        out[row] = residual[row] + sum;
    }
}

// FusedDownProjResidualBF16 is FusedDownProjResidual with bfloat16 down
// projection weights.
extern "C" __global__ void FusedDownProjResidualBF16(
    const float* __restrict__ hb,
    const float* residual,
    const unsigned short* __restrict__ w_down,
    float* out,
    int inter_dim,
    int hidden_dim
) {
    int row = blockIdx.x;
    if (row >= hidden_dim) return;

    int tid = threadIdx.x;
    __shared__ float s_sum[32];

    const unsigned short* w_row = w_down + (size_t)row * inter_dim;
    const ushort4* w_row4 = reinterpret_cast<const ushort4*>(w_row);
    const float4* hb4 = reinterpret_cast<const float4*>(hb);
    int inter_dim4 = rows_vectorizable(inter_dim, row) ? inter_dim / 4 : 0;

    float sum = 0.0f;
    for (int i = tid; i < inter_dim4; i += blockDim.x) {
        float4 hv = hb4[i];
        ushort4 udown = w_row4[i];
        float4 wv = make_float4(bf16_to_f32(udown.x), bf16_to_f32(udown.y), bf16_to_f32(udown.z), bf16_to_f32(udown.w));
        sum += hv.x * wv.x + hv.y * wv.y + hv.z * wv.z + hv.w * wv.w;
    }
    for (int i = inter_dim4 * 4 + tid; i < inter_dim; i += blockDim.x) {
        sum += hb[i] * bf16_to_f32(w_row[i]);
    }
    sum = block_reduce_sum(sum, s_sum);
    if (tid == 0) {
        out[row] = residual[row] + sum;
    }
}
