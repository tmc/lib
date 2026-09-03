#include <metal_stdlib>
using namespace metal;

// TailCandidate is stored as two floats so Go can allocate the scratch with
// the existing typed Metal allocator. The index is exactly representable for
// the supported vocabulary sizes.
static inline bool tail_better(float value, uint index, float best, uint best_index) {
    return !isnan(value) && (value > best || (value == best && index < best_index));
}

kernel void RMSNormScale(
    device const float* x [[buffer(0)]], device float* scale [[buffer(1)]],
    constant uint& rows [[buffer(2)]], constant uint& dim [[buffer(3)]],
    constant float& eps [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    if (row >= rows) return;
    float sum = 0.0f;
    for (uint i = tid; i < dim; i += 128) sum += x[row * dim + i] * x[row * dim + i];
    sum = simd_sum(sum);
    threadgroup float partial[4];
    if (lane == 0) partial[simd] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = (simd == 0 && lane < 4) ? partial[lane] : 0.0f;
    if (simd == 0) total = simd_sum(total);
    if (tid == 0) scale[row] = rsqrt(total / float(dim) + eps);
}

template<typename T>
static inline void lm_head_tile_argmax(
    device const float* x, device const T* norm_weight, device const T* lm_head,
    device const float* scale, device float2* candidates, uint dim, uint vocab,
    uint tile_count, uint group, uint tid, threadgroup float* values,
    threadgroup uint* indices) {
    uint row = group / tile_count;
    uint token = (group % tile_count) * 256 + tid;
    float value = -INFINITY;
    if (token < vocab) {
        value = 0.0f;
        uint xbase = row * dim;
        uint wbase = token * dim;
        float s = scale[row];
        for (uint i = 0; i < dim; i++)
            value += (x[xbase+i] * s * float(norm_weight[i])) * float(lm_head[wbase+i]);
    }
    values[tid] = isnan(value) ? -INFINITY : value;
    indices[tid] = token;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride && tail_better(values[tid+stride], indices[tid+stride], values[tid], indices[tid])) {
            values[tid] = values[tid+stride];
            indices[tid] = indices[tid+stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) candidates[group] = float2(values[0], float(indices[0]));
}

kernel void LMHeadTileArgmax(
    device const float* x [[buffer(0)]], device const float* nw [[buffer(1)]],
    device const float* head [[buffer(2)]], device const float* scale [[buffer(3)]],
    device float2* out [[buffer(4)]], constant uint& dim [[buffer(5)]],
    constant uint& vocab [[buffer(6)]], constant uint& tiles [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float values[256];
    threadgroup uint indices[256];
    lm_head_tile_argmax(x, nw, head, scale, out, dim, vocab, tiles, group, tid, values, indices);
}

kernel void LMHeadTileArgmaxH(
    device const float* x [[buffer(0)]], device const half* nw [[buffer(1)]],
    device const half* head [[buffer(2)]], device const float* scale [[buffer(3)]],
    device float2* out [[buffer(4)]], constant uint& dim [[buffer(5)]],
    constant uint& vocab [[buffer(6)]], constant uint& tiles [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float values[256];
    threadgroup uint indices[256];
    lm_head_tile_argmax(x, nw, head, scale, out, dim, vocab, tiles, group, tid, values, indices);
}

kernel void ArgmaxTiles(
    device const float2* in [[buffer(0)]], device int* tokens [[buffer(1)]],
    constant uint& rows [[buffer(2)]], constant uint& tiles [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    if (row >= rows) return;
    float best = -INFINITY;
    uint index = 0;
    for (uint i = tid; i < tiles; i += 128) {
        float2 candidate = in[row * tiles + i];
        uint candidate_index = uint(candidate.y);
        if (tail_better(candidate.x, candidate_index, best, index)) {
            best = candidate.x;
            index = candidate_index;
        }
    }
    threadgroup float values[128];
    threadgroup uint indices[128];
    values[tid] = best;
    indices[tid] = index;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 64; stride != 0; stride >>= 1) {
        if (tid < stride && tail_better(values[tid+stride], indices[tid+stride], values[tid], indices[tid])) {
            values[tid] = values[tid+stride];
            indices[tid] = indices[tid+stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) tokens[row] = int(indices[0]);
}

#ifndef METAL_SIMD_SUM_DEFINED
#define METAL_SIMD_SUM_DEFINED
inline float simd_sum(float val) {
    #pragma unroll
    for (ushort offset = 16; offset > 0; offset /= 2) {
        val += simd_shuffle_down(val, offset);
    }
    return val;
}
#endif

// 1H. Embedding lookup over half weights
kernel void EmbeddingLookupH(
    device const half* embed_table     [[buffer(0)]],
    device float* x_out                [[buffer(1)]],
    constant uint& token_id            [[buffer(2)]],
    constant uint& dim                 [[buffer(3)]],
    uint tid                           [[thread_position_in_grid]]
) {
    if (tid < dim) {
        x_out[tid] = float(embed_table[token_id * dim + tid]);
    }
}

// 2H. Fast RMSNorm with half norm weights
kernel void ComputeRMSNormH(
    device const float* x              [[buffer(0)]],
    device const half* weight          [[buffer(1)]],
    device float* out_normed           [[buffer(2)]],
    constant uint& dim                 [[buffer(3)]],
    constant float& eps                [[buffer(4)]],
    uint tid                           [[thread_position_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    float local_sq = 0.0f;
    for (uint i = tid; i < dim; i += 128) {
        float v = x[i];
        local_sq += v * v;
    }
    local_sq = simd_sum(local_sq);
    threadgroup float s_sum[4];
    if (simd_lane == 0) s_sum[simd_id] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sq = (simd_id == 0 && simd_lane < 4) ? s_sum[simd_lane] : 0.0f;
    if (simd_id == 0) total_sq = simd_sum(total_sq);

    threadgroup float s_rms = 0.0f;
    if (tid == 0) {
        s_rms = rsqrt(total_sq / float(dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms = s_rms;
    for (uint i = tid; i < dim; i += 128) {
        out_normed[i] = x[i] * rms * float(weight[i]);
    }
}

// 3H. Fast GEMV over half weights, float accumulation
kernel void FastGEMVH(
    device const float* x              [[buffer(0)]],
    device const half* W               [[buffer(1)]],
    device float* y                    [[buffer(2)]],
    constant uint& in_dim              [[buffer(3)]],
    constant uint& out_dim             [[buffer(4)]],
    uint row                           [[threadgroup_position_in_grid]],
    uint tid                           [[thread_position_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    if (row < out_dim) {
        device const half* w_row = W + row * in_dim;
        uint in_dim4 = in_dim / 4;
        device const half4* w_row4 = (device const half4*)w_row;

        float acc = 0.0f;
        for (uint i = tid; i < in_dim4; i += 128) {
            half4 w4 = w_row4[i];
            uint base = i * 4;
            acc += x[base+0] * float(w4[0]) +
                   x[base+1] * float(w4[1]) +
                   x[base+2] * float(w4[2]) +
                   x[base+3] * float(w4[3]);
        }
        for (uint i = in_dim4 * 4 + tid; i < in_dim; i += 128) {
            acc += x[i] * float(w_row[i]);
        }
        acc = simd_sum(acc);
        threadgroup float s_sum[4];
        if (simd_lane == 0) s_sum[simd_id] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float block_sum = (simd_id == 0 && simd_lane < 4) ? s_sum[simd_lane] : 0.0f;
        if (simd_id == 0) block_sum = simd_sum(block_sum);
        if (tid == 0) y[row] = block_sum;
    }
}

// 3bH. Fast GEMV with Residual Accumulation over half weights
kernel void FastGEMVResidualH(
    device const float* x              [[buffer(0)]],
    device const half* W               [[buffer(1)]],
    device float* y_inout              [[buffer(2)]],
    constant uint& in_dim              [[buffer(3)]],
    constant uint& out_dim             [[buffer(4)]],
    uint row                           [[threadgroup_position_in_grid]],
    uint tid                           [[thread_position_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    if (row < out_dim) {
        device const half* w_row = W + row * in_dim;
        uint in_dim4 = in_dim / 4;
        device const half4* w_row4 = (device const half4*)w_row;

        float acc = 0.0f;
        for (uint i = tid; i < in_dim4; i += 128) {
            half4 w4 = w_row4[i];
            uint base = i * 4;
            acc += x[base+0] * float(w4[0]) +
                   x[base+1] * float(w4[1]) +
                   x[base+2] * float(w4[2]) +
                   x[base+3] * float(w4[3]);
        }
        for (uint i = in_dim4 * 4 + tid; i < in_dim; i += 128) {
            acc += x[i] * float(w_row[i]);
        }
        acc = simd_sum(acc);
        threadgroup float s_sum[4];
        if (simd_lane == 0) s_sum[simd_id] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float block_sum = (simd_id == 0 && simd_lane < 4) ? s_sum[simd_lane] : 0.0f;
        if (simd_id == 0) block_sum = simd_sum(block_sum);
        if (tid == 0) y_inout[row] += block_sum;
    }
}

// 4H. 64-thread variants over half weights
kernel void FastGEMV64H(
    device const float* x              [[buffer(0)]],
    device const half* W               [[buffer(1)]],
    device float* y                    [[buffer(2)]],
    constant uint& in_dim              [[buffer(3)]],
    constant uint& out_dim             [[buffer(4)]],
    uint row                           [[threadgroup_position_in_grid]],
    uint tid                           [[thread_position_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    if (row < out_dim) {
        device const half* w_row = W + row * in_dim;
        uint in_dim4 = in_dim / 4;
        device const half4* w_row4 = (device const half4*)w_row;

        float acc = 0.0f;
        for (uint i = tid; i < in_dim4; i += 64) {
            half4 w4 = w_row4[i];
            uint base = i * 4;
            acc += x[base+0] * float(w4[0]) +
                   x[base+1] * float(w4[1]) +
                   x[base+2] * float(w4[2]) +
                   x[base+3] * float(w4[3]);
        }
        for (uint i = in_dim4 * 4 + tid; i < in_dim; i += 64) {
            acc += x[i] * float(w_row[i]);
        }
        acc = simd_sum(acc);
        threadgroup float s_sum[2];
        if (simd_lane == 0) s_sum[simd_id] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float block_sum = (simd_id == 0 && simd_lane < 2) ? s_sum[simd_lane] : 0.0f;
        if (simd_id == 0) block_sum = simd_sum(block_sum);
        if (tid == 0) y[row] = block_sum;
    }
}

kernel void FastGEMVResidual64H(
    device const float* x              [[buffer(0)]],
    device const half* W               [[buffer(1)]],
    device float* y_inout              [[buffer(2)]],
    constant uint& in_dim              [[buffer(3)]],
    constant uint& out_dim             [[buffer(4)]],
    uint row                           [[threadgroup_position_in_grid]],
    uint tid                           [[thread_position_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    if (row < out_dim) {
        device const half* w_row = W + row * in_dim;
        uint in_dim4 = in_dim / 4;
        device const half4* w_row4 = (device const half4*)w_row;

        float acc = 0.0f;
        for (uint i = tid; i < in_dim4; i += 64) {
            half4 w4 = w_row4[i];
            uint base = i * 4;
            acc += x[base+0] * float(w4[0]) +
                   x[base+1] * float(w4[1]) +
                   x[base+2] * float(w4[2]) +
                   x[base+3] * float(w4[3]);
        }
        for (uint i = in_dim4 * 4 + tid; i < in_dim; i += 64) {
            acc += x[i] * float(w_row[i]);
        }
        acc = simd_sum(acc);
        threadgroup float s_sum[2];
        if (simd_lane == 0) s_sum[simd_id] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float block_sum = (simd_id == 0 && simd_lane < 2) ? s_sum[simd_lane] : 0.0f;
        if (simd_id == 0) block_sum = simd_sum(block_sum);
        if (tid == 0) y_inout[row] += block_sum;
    }
}

// 5H. Batched embedding lookup over half weights
kernel void BatchedEmbeddingLookupH(
    device const half* embed_table     [[buffer(0)]],
    device const int* tokens           [[buffer(1)]],
    device float* x_out                [[buffer(2)]],
    constant uint& seq_len             [[buffer(3)]],
    constant uint& dim                 [[buffer(4)]],
    uint gid                           [[thread_position_in_grid]]
) {
    uint total = seq_len * dim;
    if (gid < total) {
        uint token_idx = gid / dim;
        uint dim_idx = gid % dim;
        int tok = tokens[token_idx];
        x_out[gid] = float(embed_table[tok * dim + dim_idx]);
    }
}

// 6H. Batched RMSNorm over half norm weights
kernel void BatchedRMSNormH(
    device const float* x_in           [[buffer(0)]],
    device const half* weight          [[buffer(1)]],
    device float* x_out                [[buffer(2)]],
    constant uint& seq_len             [[buffer(3)]],
    constant uint& dim                 [[buffer(4)]],
    constant float& eps                [[buffer(5)]],
    uint token_idx                     [[threadgroup_position_in_grid]],
    uint tid                           [[thread_position_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    if (token_idx >= seq_len) return;
    device const float* x_row = x_in + token_idx * dim;
    device float* out_row = x_out + token_idx * dim;

    uint dim4 = dim / 4;
    device const float4* x4 = (device const float4*)x_row;

    float local_sq = 0.0f;
    for (uint i = tid; i < dim4; i += 128) {
        float4 v = x4[i];
        local_sq += dot(v, v);
    }
    for (uint i = dim4 * 4 + tid; i < dim; i += 128) {
        float v = x_row[i];
        local_sq += v * v;
    }
    local_sq = simd_sum(local_sq);
    threadgroup float s_sum[4];
    if (simd_lane == 0) s_sum[simd_id] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sq = (simd_id == 0 && simd_lane < 4) ? s_sum[simd_lane] : 0.0f;
    if (simd_id == 0) total_sq = simd_sum(total_sq);

    threadgroup float s_rms = 0.0f;
    if (tid == 0) {
        s_rms = rsqrt(total_sq / float(dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms = s_rms;
    for (uint i = tid; i < dim; i += 128) {
        out_row[i] = x_row[i] * rms * float(weight[i]);
    }
}

// 7H. Batched GEMM over half weights
kernel void BatchedGEMMH(
    device const float* X              [[buffer(0)]],
    device const half* W               [[buffer(1)]],
    device float* Y                    [[buffer(2)]],
    constant uint& seq_len             [[buffer(3)]],
    constant uint& in_dim              [[buffer(4)]],
    constant uint& out_dim             [[buffer(5)]],
    uint group_idx                     [[threadgroup_position_in_grid]],
    uint tid                           [[thread_position_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    uint token_idx = group_idx / out_dim;
    uint out_col = group_idx % out_dim;
    if (token_idx < seq_len && out_col < out_dim) {
        device const float* x_row = X + token_idx * in_dim;
        device const half* w_row = W + out_col * in_dim;
        uint in_dim4 = in_dim / 4;
        device const half4* w_row4 = (device const half4*)w_row;

        float acc = 0.0f;
        for (uint i = tid; i < in_dim4; i += 128) {
            half4 w4 = w_row4[i];
            uint base = i * 4;
            acc += x_row[base+0] * float(w4[0]) +
                   x_row[base+1] * float(w4[1]) +
                   x_row[base+2] * float(w4[2]) +
                   x_row[base+3] * float(w4[3]);
        }
        for (uint i = in_dim4 * 4 + tid; i < in_dim; i += 128) {
            acc += x_row[i] * float(w_row[i]);
        }
        acc = simd_sum(acc);
        threadgroup float s_sum[4];
        if (simd_lane == 0) s_sum[simd_id] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float block_sum = (simd_id == 0 && simd_lane < 4) ? s_sum[simd_lane] : 0.0f;
        if (simd_id == 0) block_sum = simd_sum(block_sum);
        if (tid == 0) {
            Y[token_idx * out_dim + out_col] = block_sum;
        }
    }
}

// 8H. Batched GEMM Residual over half weights
kernel void BatchedGEMMResidualH(
    device const float* X              [[buffer(0)]],
    device const half* W               [[buffer(1)]],
    device float* Y                    [[buffer(2)]],
    constant uint& seq_len             [[buffer(3)]],
    constant uint& in_dim              [[buffer(4)]],
    constant uint& out_dim             [[buffer(5)]],
    uint group_idx                     [[threadgroup_position_in_grid]],
    uint tid                           [[thread_position_in_threadgroup]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]]
) {
    uint token_idx = group_idx / out_dim;
    uint out_col = group_idx % out_dim;
    if (token_idx < seq_len && out_col < out_dim) {
        device const float* x_row = X + token_idx * in_dim;
        device const half* w_row = W + out_col * in_dim;
        uint in_dim4 = in_dim / 4;
        device const half4* w_row4 = (device const half4*)w_row;

        float acc = 0.0f;
        for (uint i = tid; i < in_dim4; i += 128) {
            half4 w4 = w_row4[i];
            uint base = i * 4;
            acc += x_row[base+0] * float(w4[0]) +
                   x_row[base+1] * float(w4[1]) +
                   x_row[base+2] * float(w4[2]) +
                   x_row[base+3] * float(w4[3]);
        }
        for (uint i = in_dim4 * 4 + tid; i < in_dim; i += 128) {
            acc += x_row[i] * float(w_row[i]);
        }
        acc = simd_sum(acc);
        threadgroup float s_sum[4];
        if (simd_lane == 0) s_sum[simd_id] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float block_sum = (simd_id == 0 && simd_lane < 4) ? s_sum[simd_lane] : 0.0f;
        if (simd_id == 0) block_sum = simd_sum(block_sum);
        if (tid == 0) {
            Y[token_idx * out_dim + out_col] += block_sum;
        }
    }
}

kernel void BatchedQKVProjectionH(
    device const float* X [[buffer(0)]], device const half* WQ [[buffer(1)]],
    device const half* WK [[buffer(2)]], device const half* WV [[buffer(3)]],
    device float* Q [[buffer(4)]], device float* K [[buffer(5)]], device float* V [[buffer(6)]],
    constant uint& rows [[buffer(7)]], constant uint& in_dim [[buffer(8)]],
    constant uint& q_dim [[buffer(9)]], constant uint& kv_dim [[buffer(10)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
    uint span=q_dim+2*kv_dim, row=group/span, col=group%span; if(row>=rows) return;
    device const half* weight; device float* output; uint out_dim,out_col;
    if(col<q_dim){weight=WQ;output=Q;out_dim=q_dim;out_col=col;}
    else if(col<q_dim+kv_dim){weight=WK;output=K;out_dim=kv_dim;out_col=col-q_dim;}
    else{weight=WV;output=V;out_dim=kv_dim;out_col=col-q_dim-kv_dim;}
    device const float* x=X+row*in_dim; device const half* w=weight+out_col*in_dim;
    uint n4=in_dim/4; device const half4* w4=(device const half4*)w; float acc=0;
    for(uint i=tid;i<n4;i+=128){half4 h=w4[i];uint b=i*4;acc+=x[b]*float(h[0])+x[b+1]*float(h[1])+x[b+2]*float(h[2])+x[b+3]*float(h[3]);}
    for(uint i=n4*4+tid;i<in_dim;i+=128)acc+=x[i]*float(w[i]);
    acc=simd_sum(acc);threadgroup float sums[4];if(lane==0)sums[simd]=acc;threadgroup_barrier(mem_flags::mem_threadgroup);
    float total=(simd==0&&lane<4)?sums[lane]:0;if(simd==0)total=simd_sum(total);if(tid==0)output[row*out_dim+out_col]=total;
}

kernel void BatchedGateUpProjectionH(
    device const float* X [[buffer(0)]], device const half* WG [[buffer(1)]], device const half* WU [[buffer(2)]],
    device float* G [[buffer(3)]], device float* U [[buffer(4)]], constant uint& rows [[buffer(5)]],
    constant uint& in_dim [[buffer(6)]], constant uint& out_dim [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
    uint span=2*out_dim,row=group/span,col=group%span;if(row>=rows)return;bool up=col>=out_dim;uint out_col=up?col-out_dim:col;
    device const half* w=(up?WU:WG)+out_col*in_dim;device const float* x=X+row*in_dim;uint n4=in_dim/4;device const half4* w4=(device const half4*)w;float acc=0;
    for(uint i=tid;i<n4;i+=128){half4 h=w4[i];uint b=i*4;acc+=x[b]*float(h[0])+x[b+1]*float(h[1])+x[b+2]*float(h[2])+x[b+3]*float(h[3]);}
    for(uint i=n4*4+tid;i<in_dim;i+=128)acc+=x[i]*float(w[i]);acc=simd_sum(acc);threadgroup float sums[4];if(lane==0)sums[simd]=acc;threadgroup_barrier(mem_flags::mem_threadgroup);
    float total=(simd==0&&lane<4)?sums[lane]:0;if(simd==0)total=simd_sum(total);if(tid==0)(up?U:G)[row*out_dim+out_col]=total;
}
