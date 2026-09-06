#include <metal_stdlib>
using namespace metal;

constant constexpr uint PGQA_THREADS = 128;
constant constexpr uint PGQA_TILE = 256;
constant constexpr uint PGQA_MAX_DIM = 256;
constant constexpr uint PGQA_MAX_RATIO = 8;
constant constexpr float PGQA_NEG_INF = -1e30f;

inline float pgqa_sum(float v) {
    for (ushort off = 16; off > 0; off >>= 1) v += simd_shuffle_down(v, off);
    return v;
}

inline float pgqa_max(float v) {
    for (ushort off = 16; off > 0; off >>= 1) v = max(v, simd_shuffle_down(v, off));
    return v;
}

// PersistentGQADecode is an experimental correctness kernel. The dispatch is
// [num_kv_heads, batch, 1], with exactly PGQA_THREADS threads per group.
kernel void PersistentGQADecode(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    device const uint* positions [[buffer(4)]],
    constant uint& max_seq_len [[buffer(5)]],
    constant uint& num_q_heads [[buffer(6)]],
    constant uint& num_kv_heads [[buffer(7)]],
    constant uint& head_dim [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
    const uint kv_head = group % num_kv_heads;
    const uint row = group / num_kv_heads;
    const uint ratio = num_q_heads / num_kv_heads;
    if (kv_head >= num_kv_heads || head_dim == 0 || head_dim > PGQA_MAX_DIM ||
        ratio == 0 || ratio > PGQA_MAX_RATIO || ratio * num_kv_heads != num_q_heads) return;
    const uint seq_len = min(positions[row] + 1, max_seq_len);
    const uint kv_dim = num_kv_heads * head_dim;
    const uint q_row = row * num_q_heads * head_dim;
    const uint kv_row = row * max_seq_len * kv_dim;
    threadgroup float scores[PGQA_MAX_RATIO][PGQA_TILE];
    threadgroup float acc[PGQA_MAX_RATIO][PGQA_MAX_DIM];
    threadgroup float red[4];
    threadgroup float run_max[PGQA_MAX_RATIO];
    threadgroup float run_sum[PGQA_MAX_RATIO];
    for (uint qh = 0; qh < ratio; qh++) {
        for (uint d = tid; d < head_dim; d += PGQA_THREADS) acc[qh][d] = 0;
        if (tid == 0) { run_max[qh] = PGQA_NEG_INF; run_sum[qh] = 0; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint base = 0; base < seq_len; base += PGQA_TILE) {
        const uint tile = min(PGQA_TILE, seq_len - base);
        for (uint j = tid; j < tile; j += PGQA_THREADS) {
            device const float* kp = k + kv_row + ((base+j)*num_kv_heads+kv_head)*head_dim;
            float dots[PGQA_MAX_RATIO];
            for (uint qh = 0; qh < ratio; qh++) dots[qh] = 0;
            for (uint d = 0; d < head_dim; d++) {
                const float kval = kp[d];
                for (uint qh = 0; qh < ratio; qh++) {
                    const uint head = kv_head*ratio+qh;
                    dots[qh] += q[q_row+head*head_dim+d]*kval;
                }
            }
            for (uint qh = 0; qh < ratio; qh++) scores[qh][j] = dots[qh]*scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint qh = 0; qh < ratio; qh++) {
            float local_max = PGQA_NEG_INF;
            for (uint j = tid; j < tile; j += PGQA_THREADS) local_max = max(local_max, scores[qh][j]);
            local_max = pgqa_max(local_max);
            if ((tid & 31) == 0) red[tid >> 5] = local_max;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float tile_max = max(max(red[0], red[1]), max(red[2], red[3]));
            const float next_max = max(run_max[qh], tile_max);
            const float correction = run_max[qh] <= PGQA_NEG_INF ? 0 : exp(run_max[qh]-next_max);
            float local_sum = 0;
            for (uint j = tid; j < tile; j += PGQA_THREADS) {
                scores[qh][j] = exp(scores[qh][j]-next_max);
                local_sum += scores[qh][j];
            }
            local_sum = pgqa_sum(local_sum);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if ((tid & 31) == 0) red[tid >> 5] = local_sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float tile_sum = red[0]+red[1]+red[2]+red[3];
            for (uint d = tid; d < head_dim; d += PGQA_THREADS) {
                float weighted = 0;
                for (uint j = 0; j < tile; j++)
                    weighted += scores[qh][j] * v[kv_row+((base+j)*num_kv_heads+kv_head)*head_dim+d];
                acc[qh][d] = acc[qh][d]*correction+weighted;
            }
            if (tid == 0) { run_max[qh]=next_max; run_sum[qh]=run_sum[qh]*correction+tile_sum; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint qh = 0; qh < ratio; qh++) {
        const uint head = kv_head*ratio+qh;
        for (uint d = tid; d < head_dim; d += PGQA_THREADS)
            out[q_row+head*head_dim+d] = run_sum[qh] > 0 ? acc[qh][d]/run_sum[qh] : 0;
    }
}

struct PGQABlock {
    uint start;
    uint end;
    ulong k_address;
    ulong v_address;
};

struct PGQARow {
    uint prefix_tokens;
    uint suffix_tokens;
    uint suffix_offset;
    uint block_offset;
    uint block_count;
};

// PersistentGQABlockDecode has the production block-table ABI but remains an
// experimental kernel. The encoder must declare every indirect block resource.
kernel void PersistentGQABlockDecode(
    device const float* q [[buffer(0)]],
    device const float* suffix_k [[buffer(1)]],
    device const float* suffix_v [[buffer(2)]],
    device float* out [[buffer(3)]],
    device const PGQABlock* blocks [[buffer(4)]],
    device const PGQARow* rows [[buffer(5)]],
    constant uint& num_q_heads [[buffer(6)]],
    constant uint& num_kv_heads [[buffer(7)]],
    constant uint& head_dim [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
    const uint kv_head = group % num_kv_heads;
    const uint row = group / num_kv_heads;
    const uint ratio = num_q_heads / num_kv_heads;
    if (head_dim == 0 || head_dim > PGQA_MAX_DIM || ratio == 0 ||
        ratio > PGQA_MAX_RATIO || ratio*num_kv_heads != num_q_heads) return;
    const PGQARow p = rows[row];
    const uint total = p.prefix_tokens+p.suffix_tokens;
    const uint kv_dim = num_kv_heads*head_dim;
    const uint q_row = row*num_q_heads*head_dim;
    threadgroup float scores[PGQA_MAX_RATIO][PGQA_TILE];
    threadgroup float acc[PGQA_MAX_RATIO][PGQA_MAX_DIM];
    threadgroup float red[4];
    threadgroup float run_max[PGQA_MAX_RATIO];
    threadgroup float run_sum[PGQA_MAX_RATIO];
    for (uint qh=0; qh<ratio; qh++) {
        for (uint d=tid; d<head_dim; d+=PGQA_THREADS) acc[qh][d]=0;
        if (tid==0) { run_max[qh]=PGQA_NEG_INF; run_sum[qh]=0; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint base=0; base<total; base+=PGQA_TILE) {
        const uint tile=min(PGQA_TILE,total-base);
        for (uint j=tid; j<tile; j+=PGQA_THREADS) {
            const uint token=base+j;
            device const float* kp=suffix_k+p.suffix_offset*kv_dim;
            uint local=token-p.prefix_tokens;
            if (token<p.prefix_tokens) {
                for (uint bi=0; bi<p.block_count; bi++) {
                    const PGQABlock block=blocks[p.block_offset+bi];
                    if (token<block.start || token>=block.end) continue;
                    kp=reinterpret_cast<device const float*>(block.k_address);
                    local=token-block.start;
                    break;
                }
            }
            kp+=(local*num_kv_heads+kv_head)*head_dim;
            float dots[PGQA_MAX_RATIO];
            for (uint qh=0; qh<ratio; qh++) dots[qh]=0;
            for (uint d=0; d<head_dim; d++) {
                const float kval=kp[d];
                for (uint qh=0; qh<ratio; qh++) dots[qh]+=q[q_row+(kv_head*ratio+qh)*head_dim+d]*kval;
            }
            for (uint qh=0; qh<ratio; qh++) scores[qh][j]=dots[qh]*scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint qh=0; qh<ratio; qh++) {
            float local_max=PGQA_NEG_INF;
            for (uint j=tid; j<tile; j+=PGQA_THREADS) local_max=max(local_max,scores[qh][j]);
            local_max=pgqa_max(local_max);
            if ((tid&31)==0) red[tid>>5]=local_max;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float tile_max=max(max(red[0],red[1]),max(red[2],red[3]));
            const float next_max=max(run_max[qh],tile_max);
            const float correction=run_max[qh]<=PGQA_NEG_INF?0:exp(run_max[qh]-next_max);
            float local_sum=0;
            for (uint j=tid; j<tile; j+=PGQA_THREADS) { scores[qh][j]=exp(scores[qh][j]-next_max); local_sum+=scores[qh][j]; }
            local_sum=pgqa_sum(local_sum);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if ((tid&31)==0) red[tid>>5]=local_sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float tile_sum=red[0]+red[1]+red[2]+red[3];
            for (uint d=tid; d<head_dim; d+=PGQA_THREADS) {
                float weighted=0;
                for (uint j=0; j<tile; j++) {
                    const uint token=base+j;
                    device const float* vp=suffix_v+p.suffix_offset*kv_dim;
                    uint local=token-p.prefix_tokens;
                    if (token<p.prefix_tokens) {
                        for (uint bi=0; bi<p.block_count; bi++) {
                            const PGQABlock block=blocks[p.block_offset+bi];
                            if (token<block.start || token>=block.end) continue;
                            vp=reinterpret_cast<device const float*>(block.v_address);
                            local=token-block.start;
                            break;
                        }
                    }
                    weighted+=scores[qh][j]*vp[(local*num_kv_heads+kv_head)*head_dim+d];
                }
                acc[qh][d]=acc[qh][d]*correction+weighted;
            }
            if (tid==0) { run_max[qh]=next_max; run_sum[qh]=run_sum[qh]*correction+tile_sum; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint qh=0; qh<ratio; qh++) {
        const uint head=kv_head*ratio+qh;
        for (uint d=tid; d<head_dim; d+=PGQA_THREADS)
            out[q_row+head*head_dim+d]=run_sum[qh]>0?acc[qh][d]/run_sum[qh]:0;
    }
}
