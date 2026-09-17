#include <cuda_fp16.h>

#ifndef DS4_ATTENTION_HMMA_CU
#define DS4_ATTENTION_HMMA_CU

#ifndef INFINITY
#define INFINITY (__int_as_float(0x7f800000))
#endif

/* -------------------------------------------------------------------------
 * DS4_ATTN_TOKENTILE: token-tile HMMA indexed attention.
 * Ported from antirez/ds4 with fixed STAGE_ROWS=32.
 * Four-token tiles reduce the selected-row union while eight heads keep the
 * 32-row MMA shape full.
 */

static constexpr uint32_t kTTTileTokens = 4u;
static constexpr uint32_t kTTG = 8u;
static constexpr uint32_t kTTM = 32u;
static constexpr uint32_t kTTStageRows = 32u;
static constexpr uint32_t kTTRawWindow = 128u;
static constexpr uint32_t kTTHeadDim = 512u;
static constexpr uint32_t kTTWarps = 16u;
static constexpr uint32_t kTTThreads = 512u;
static constexpr uint32_t kTTScoreKQuarters = 4u;
static constexpr uint32_t kTTScoreKSliceDim = kTTHeadDim / kTTScoreKQuarters;
static constexpr uint32_t kTTScoreKStepsPerQuarter = kTTScoreKSliceDim / 16u;
static constexpr uint32_t kTTRecordRingPlanes = 4u;
static constexpr uint32_t kTTProbStride = 40u;
static constexpr uint32_t kTTRingChunkBytes = 16u;
static constexpr uint32_t kTTRingChunksPerRow =
    (kTTHeadDim * sizeof(half)) / kTTRingChunkBytes;
static constexpr size_t kTTSmemHardCap = 90ull * 1024ull;

static_assert(kTTM == kTTTileTokens * kTTG, "token-tile M must be 4 tokens x G8");
static_assert(kTTProbStride == kTTStageRows + 8u, "token-tile prob stride changed");
static_assert(kTTScoreKSliceDim % 16u == 0, "token-tile score K split changed");
static_assert(kTTRingChunksPerRow == 64u, "token-tile KV ring expects 64 chunks");
static_assert(sizeof(int2) == 8u, "token-tile union record must remain 8 bytes");

template <uint32_t TT_STAGE_ROWS>
struct tt_TokentileLayout {
    static constexpr uint32_t prob_stride = TT_STAGE_ROWS + 8u;
    static constexpr uint32_t ring_plane_bytes =
        TT_STAGE_ROWS * kTTRingChunksPerRow * kTTRingChunkBytes;
    static constexpr uint32_t ring_plane_elems = ring_plane_bytes / sizeof(half);
};

__device__ static float tt_warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ static float tt_warp_max_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ __forceinline__ uint32_t tt_lane_id(void) {
    return threadIdx.x & 31u;
}

__device__ __forceinline__ uint32_t tt_warp_id(void) {
    return threadIdx.x >> 5u;
}

__device__ __forceinline__ int tt_mma_c_i(uint32_t lane, int l) {
    return ((l >> 1) << 3) + (int)(lane >> 2);
}

__device__ __forceinline__ int tt_mma_c_j(uint32_t lane, int l) {
    return (int)((lane & 3u) << 1) + (l & 1);
}

__device__ __forceinline__ unsigned tt_smem_addr(const void *p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ uint32_t tt_ring_off_bytes(uint32_t row, uint32_t c) {
    return (row * kTTRingChunksPerRow + (c ^ (row & 7u))) * kTTRingChunkBytes;
}

__device__ __forceinline__ void tt_ldmatrix_x4_addr(uint32_t (&r)[4], unsigned a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
#else
    (void)a;
    r[0] = r[1] = r[2] = r[3] = 0;
#endif
}

__device__ __forceinline__ void tt_ldmatrix_x2_addr(uint32_t (&r)[2], unsigned a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(a));
#else
    (void)a;
    r[0] = r[1] = 0;
#endif
}

__device__ __forceinline__ void tt_ldmatrix_x2_trans_addr(uint32_t (&r)[2], unsigned a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.b16 {%0, %1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(a));
#else
    (void)a;
    r[0] = r[1] = 0;
#endif
}

__device__ __forceinline__ void tt_mma_m16n8k16_f16_f32(
        float *d,
        const uint32_t (&a)[4],
        const uint32_t (&b)[2]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint32_t d0 = __float_as_uint(d[0]);
    uint32_t d1 = __float_as_uint(d[1]);
    uint32_t d2 = __float_as_uint(d[2]);
    uint32_t d3 = __float_as_uint(d[3]);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    d[0] = __uint_as_float(d0);
    d[1] = __uint_as_float(d1);
    d[2] = __uint_as_float(d2);
    d[3] = __uint_as_float(d3);
#else
    (void)a;
    (void)b;
#endif
}

__device__ __forceinline__ unsigned char *tt_align16(unsigned char *p) {
    size_t x = reinterpret_cast<size_t>(p);
    x = (x + 15u) & ~size_t(15u);
    return reinterpret_cast<unsigned char *>(x);
}

__device__ __forceinline__ void tt_zero_16B(void *dst) {
    *reinterpret_cast<int4 *>(dst) = make_int4(0, 0, 0, 0);
}

__device__ __forceinline__ void tt_zero_8B(void *dst) {
    *reinterpret_cast<int2 *>(dst) = make_int2(0, 0);
}

__device__ __forceinline__ void tt_cp_async_16B(void *dst, const void *src, bool pred) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (pred) {
        const unsigned smem = tt_smem_addr(dst);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                     :: "r"(smem), "l"(src));
    } else {
        tt_zero_16B(dst);
    }
#else
    (void)src;
    (void)pred;
    tt_zero_16B(dst);
#endif
}

__device__ __forceinline__ void tt_cp_async_8B(void *dst, const void *src, bool pred) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (pred) {
        const unsigned smem = tt_smem_addr(dst);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;"
                     :: "r"(smem), "l"(src));
    } else {
        tt_zero_8B(dst);
    }
#else
    (void)src;
    (void)pred;
    tt_zero_8B(dst);
#endif
}

__device__ __forceinline__ void tt_cp_async_commit(void) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
#endif
}

template <int KeepGroups>
__device__ __forceinline__ void tt_cp_async_wait_group(void) {
    static_assert(KeepGroups >= 0 && KeepGroups <= 7, "bad cp.async wait_group depth");
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;" :: "n"(KeepGroups));
#endif
}

__device__ __forceinline__ uint32_t tt_score_partial_slot(
        uint32_t kq,
        uint32_t m,
        uint32_t r) {
    return (kq + r + m) & 3u;
}

template <uint32_t TT_STAGE_ROWS>
__device__ __forceinline__ void tt_store_score_partial(
        float4 * __restrict__ partials,
        uint32_t kq,
        uint32_t m,
        uint32_t r,
        float v) {
    float *dst = &partials[m * TT_STAGE_ROWS + r].x;
    dst[tt_score_partial_slot(kq, m, r)] = v;
}

__device__ __forceinline__ float4 tt_load_score_partial_record(
        const float4 * __restrict__ p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    float x, y, z, w;
    asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(x), "=f"(y), "=f"(z), "=f"(w)
                 : "r"(tt_smem_addr(p)));
    return make_float4(x, y, z, w);
#else
    return *p;
#endif
}

__device__ __forceinline__ void tt_issue_cp_async_row(
        half * __restrict__ dst,
        uint32_t rr,
        uint32_t lane16,
        const half * __restrict__ src,
        bool live) {
    const char *src_b = reinterpret_cast<const char *>(src);
    char *dst_b = reinterpret_cast<char *>(dst);

#pragma unroll
    for (uint32_t i = 0; i < 4u; ++i) {
        const uint32_t chunk = lane16 + i * 16u;
        char *db = dst_b + tt_ring_off_bytes(rr, chunk);
        tt_cp_async_16B(db, live ? static_cast<const void *>(src_b + chunk * 16u)
                                 : static_cast<const void *>(db),
                        live);
    }
}

__device__ __forceinline__ uint32_t tt_stage_raw_rows(
        uint32_t row0,
        uint32_t nr,
        uint32_t raw_union_count) {
    uint32_t raw_rows = 0;
    if (row0 < raw_union_count) {
        const uint32_t raw_left = raw_union_count - row0;
        raw_rows = raw_left < nr ? raw_left : nr;
    }
    return raw_rows;
}

template <uint32_t TT_STAGE_ROWS>
__device__ __forceinline__ void tt_issue_record_stage_cp_async(
        int2 * __restrict__ rec_plane,
        uint32_t row0,
        uint32_t nr,
        uint32_t raw_union_count,
        const int2 * __restrict__ union_records_tile) {
    static_assert(TT_STAGE_ROWS == 32u, "token-tile record issue is fixed at R32");
    const uint32_t raw_rows = tt_stage_raw_rows(row0, nr, raw_union_count);
    const uint32_t lane = tt_lane_id();
    const uint32_t warp = tt_warp_id();
    const uint32_t lane16 = lane & 15u;
    const uint32_t rr = warp * 2u + (lane >> 4u);
    const bool active = rr < TT_STAGE_ROWS;
    const bool comp_live = active && rr >= raw_rows && rr < nr;
    if (comp_live && lane16 == 0u) {
        int2 *dst = rec_plane + rr;
        const uint32_t ci = row0 + rr - raw_union_count;
        tt_cp_async_8B(dst, union_records_tile + ci, true);
    }
}

template <uint32_t TT_STAGE_ROWS, bool USE_SMEM_RECORDS>
__device__ __forceinline__ void tt_issue_kv_stage_cp_async(
        half * __restrict__ dst,
        const int2 * __restrict__ rec_plane,
        uint32_t row0,
        uint32_t nr,
        uint32_t raw_union_count,
        const int2 * __restrict__ union_records_tile,
        uint32_t tile_base,
        const half * __restrict__ raw_kv,
        const half * __restrict__ comp_kv,
        uint32_t tid) {
    constexpr uint32_t kCp16PerRow = (kTTHeadDim * sizeof(half)) / 16u;
    static_assert(kCp16PerRow == 64u, "expected 64 cp.async chunks per f16 KV row");
    static_assert(TT_STAGE_ROWS == 32u, "token-tile KV issue is fixed at R32");
    (void)tid;
    const uint32_t raw_rows = tt_stage_raw_rows(row0, nr, raw_union_count);
    const uint32_t lane = tt_lane_id();
    const uint32_t warp = tt_warp_id();
    const uint32_t lane16 = lane & 15u;
    const uint32_t rr = warp * 2u + (lane >> 4u);
    const bool active = rr < TT_STAGE_ROWS;

    if (active && rr < raw_rows) {
        const uint32_t sr = row0 + rr;
        const half *src = raw_kv + (uint64_t)(tile_base + sr) * kTTHeadDim;
        tt_issue_cp_async_row(dst, rr, lane16, src, true);
    }

    if (active && rr >= raw_rows && rr < nr) {
        uint32_t comp_id = 0u;
        if (USE_SMEM_RECORDS) {
            comp_id = (uint32_t)rec_plane[rr].x;
        } else {
            const uint32_t ci = row0 + rr - raw_union_count;
            comp_id = (uint32_t)union_records_tile[ci].x;
        }
        const half *src = comp_kv + (uint64_t)comp_id * kTTHeadDim;
        tt_issue_cp_async_row(dst, rr, lane16, src, true);
    }

    if (active && rr >= nr) {
        tt_issue_cp_async_row(dst, rr, lane16, NULL, false);
    }
}

__device__ __forceinline__ void tt_load_score_q_frag(
        uint32_t (&q_frag)[kTTScoreKStepsPerQuarter][4],
        const float * __restrict__ q,
        uint32_t q_stride,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t tile_base,
        uint32_t head_base) {
    constexpr uint32_t kMtiles = kTTM / 16u;
    constexpr uint32_t kScoreWarps = kMtiles * kTTScoreKQuarters;
    const uint32_t warp = tt_warp_id();
    if (warp >= kScoreWarps) {
        return;
    }

    const uint32_t mtile = warp >> 2u;
    const uint32_t kq = warp & 3u;
    const uint32_t k_base = kq * kTTScoreKSliceDim;
    const uint32_t lane = tt_lane_id();
    const uint32_t a_group = lane >> 2u;
    const uint32_t a_col_pair = (lane & 3u) << 1u;
#pragma unroll
    for (uint32_t kt = 0; kt < kTTScoreKStepsPerQuarter; ++kt) {
        const uint32_t k0 = k_base + kt * 16u;
#pragma unroll
        for (uint32_t r = 0; r < 4u; ++r) {
            const uint32_t m = mtile * 16u + a_group + ((r & 1u) ? 8u : 0u);
            const uint32_t tok = m / kTTG;
            const uint32_t h = m - tok * kTTG;
            const uint32_t gt = tile_base + tok;
            const uint32_t gh = head_base + h;
            const uint32_t d = k0 + a_col_pair + ((r & 2u) ? 8u : 0u);
            float x0 = 0.0f;
            float x1 = 0.0f;
            if (gt < n_tokens && gh < n_head) {
                const float *q_row = q + (uint64_t)gt * q_stride + (uint64_t)gh * kTTHeadDim;
                x0 = q_row[d];
                x1 = q_row[d + 1u];
            }
            const half2 packed = __floats2half2_rn(x0, x1);
            q_frag[kt][r] =
                (uint32_t)__half_as_ushort(__low2half(packed)) |
                ((uint32_t)__half_as_ushort(__high2half(packed)) << 16);
        }
    }
}

template <uint32_t TT_STAGE_ROWS>
__device__ __forceinline__ void tt_hmma_score_stage(
        float4 * __restrict__ partial_scores,
        const uint32_t (&q_frag)[kTTScoreKStepsPerQuarter][4],
        const half * __restrict__ kv_cur,
        uint32_t nr,
        float score_scale) {
    constexpr uint32_t kMtiles = kTTM / 16u;
    constexpr uint32_t kScoreWarps = kMtiles * kTTScoreKQuarters;
    constexpr uint32_t kNtiles = TT_STAGE_ROWS / 8u;
    const uint32_t warp = tt_warp_id();
    if (warp >= kScoreWarps) {
        return;
    }

    const uint32_t mtile = warp >> 2u;
    const uint32_t kq = warp & 3u;
    const uint32_t lane = tt_lane_id();
    const unsigned kv_smem = tt_smem_addr(kv_cur);
    const uint32_t score_row_lane = lane & 7u;
    const uint32_t score_chunk_lane = (lane >> 3u) & 1u;
    const uint32_t score_chunk_base = kq * (kTTScoreKSliceDim / 8u) + score_chunk_lane;
#pragma unroll
    for (uint32_t ntile = 0; ntile < kNtiles; ++ntile) {
        const uint32_t row_base = ntile * 8u;
        if (row_base < nr) {
            float s_frag[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const uint32_t score_row = row_base + score_row_lane;
#pragma unroll
            for (uint32_t kt = 0; kt < kTTScoreKStepsPerQuarter; ++kt) {
                uint32_t b[2];
                tt_ldmatrix_x2_addr(
                    b,
                    kv_smem + tt_ring_off_bytes(score_row, score_chunk_base + kt * 2u));
                tt_mma_m16n8k16_f16_f32(s_frag, q_frag[kt], b);
            }
            const uint32_t m0 = mtile * 16u + (lane >> 2u);
            const uint32_t r0 = row_base + ((lane & 3u) << 1u);
            const uint32_t r1 = r0 + 1u;
            const uint32_t m1 = m0 + 8u;
            if (r0 < nr) {
                tt_store_score_partial<TT_STAGE_ROWS>(
                    partial_scores, kq, m0, r0, s_frag[0] * score_scale);
                tt_store_score_partial<TT_STAGE_ROWS>(
                    partial_scores, kq, m1, r0, s_frag[2] * score_scale);
            }
            if (r1 < nr) {
                tt_store_score_partial<TT_STAGE_ROWS>(
                    partial_scores, kq, m0, r1, s_frag[1] * score_scale);
                tt_store_score_partial<TT_STAGE_ROWS>(
                    partial_scores, kq, m1, r1, s_frag[3] * score_scale);
            }
        }
    }
}

template <uint32_t TT_STAGE_ROWS>
__device__ __forceinline__ void tt_softmax_stage(
        half * __restrict__ probs,
        float * __restrict__ stage_rescale,
        float * __restrict__ max_s,
        float * __restrict__ sum_s,
        const float4 * __restrict__ scores,
        const int2 * __restrict__ records,
        uint32_t row0,
        uint32_t nr,
        uint32_t raw_union_count,
        uint32_t tile_count,
        uint32_t tile_base,
        uint32_t raw_row_min) {
    constexpr uint32_t kMPerWarp = kTTM / kTTWarps;
    constexpr uint32_t kProbStride = tt_TokentileLayout<TT_STAGE_ROWS>::prob_stride;
    static_assert(kTTM % kTTWarps == 0u, "softmax maps integral m rows per warp");
    const uint32_t lane = tt_lane_id();
    const uint32_t warp = tt_warp_id();
    const uint32_t sr = row0 + lane;
    const bool lane_live = lane < TT_STAGE_ROWS && lane < nr;
    const bool raw_slot = lane_live && sr < raw_union_count;
    uint16_t comp_mask = 0u;
    if (lane_live && !raw_slot) {
        comp_mask = (uint16_t)records[lane].y;
    }
    const uint32_t prob_lane_base = warp * kProbStride + lane;

#pragma unroll
    for (uint32_t mi = 0; mi < kMPerWarp; ++mi) {
        const uint32_t m = warp + mi * kTTWarps;
        const uint32_t tok = m / kTTG;
        const bool valid_token = tok < tile_count;
        const uint32_t score_idx = m * TT_STAGE_ROWS + lane;
        const uint32_t prob_idx = prob_lane_base + mi * kTTWarps * kProbStride;
        float score = -INFINITY;
        if (lane_live && valid_token) {
            const bool selected = raw_slot
                ? (((uint32_t)(sr - tok) < kTTRawWindow) && (tile_base + sr >= raw_row_min))
                : ((comp_mask & (uint16_t)(1u << tok)) != 0u);
            if (selected) {
                const float4 parts = tt_load_score_partial_record(scores + score_idx);
                const float s01 = parts.x + parts.y;
                const float s23 = parts.z + parts.w;
                score = s01 + s23;
            }
        }

        const float stage_m = tt_warp_max_f32(score);
        const float old_m = max_s[m];
        const float new_m = fmaxf(old_m, stage_m);
        float old_scale = 1.0f;
        if (new_m != -INFINITY) {
            old_scale = (old_m == -INFINITY) ? 0.0f : expf(old_m - new_m);
        }
        const float row_scale = (score == -INFINITY || new_m == -INFINITY)
            ? 0.0f
            : expf(score - new_m);
        const float stage_sum = tt_warp_sum_f32(row_scale);

        if (lane < TT_STAGE_ROWS) {
            probs[prob_idx] = __float2half(lane < nr ? row_scale : 0.0f);
        }
        if (lane == 0u) {
            max_s[m] = new_m;
            sum_s[m] = sum_s[m] * old_scale + stage_sum;
            stage_rescale[m] = old_scale;
        }
    }
}

template <uint32_t TT_STAGE_ROWS>
__device__ __forceinline__ void tt_pv_mma_stage(
        float (&o_acc)[2u * kTTTileTokens * kTTG],
        const half * __restrict__ probs,
        const float * __restrict__ stage_rescale,
        const half * __restrict__ kv_cur) {
    constexpr uint32_t kMtiles = kTTM / 16u;
    constexpr uint32_t kPvWarpBase = 8u;
    constexpr uint32_t kPvNTiles = 8u;
    const uint32_t lane = tt_lane_id();
    const uint32_t warp = tt_warp_id();
    if (warp < kPvWarpBase) {
        return;
    }
    const uint32_t pv_warp = warp - kPvWarpBase;

#pragma unroll
    for (uint32_t mtile = 0; mtile < kMtiles; ++mtile) {
        const float *scale0 = stage_rescale + mtile * 16u + (lane >> 2u);
        const float *scale1 = scale0 + 8u;
        const float rs0 = *scale0;
        const float rs1 = *scale1;
#pragma unroll
        for (uint32_t ntile = 0; ntile < kPvNTiles; ++ntile) {
            const uint32_t idx = ((mtile * kPvNTiles + ntile) << 2);
            o_acc[idx + 0u] *= rs0;
            o_acc[idx + 1u] *= rs0;
            o_acc[idx + 2u] *= rs1;
            o_acc[idx + 3u] *= rs1;
        }
    }

    constexpr uint32_t kProbStride = tt_TokentileLayout<TT_STAGE_ROWS>::prob_stride;
    constexpr unsigned kPvAStepBytes = 16u * sizeof(half);
    constexpr unsigned kPvMtileBytes = 16u * kProbStride * sizeof(half);
    const unsigned probs_lane_base =
        tt_smem_addr(probs) +
        (unsigned)(((lane & 15u) * (kProbStride / 2u) +
                    (lane >> 4u) * 4u) * sizeof(uint32_t));
    const unsigned kv_smem = tt_smem_addr(kv_cur);
    const uint32_t pv_row_lane = lane & 15u;
    const uint32_t pv_chunk_base = pv_warp * kPvNTiles;
#pragma unroll
    for (uint32_t kt = 0; kt < TT_STAGE_ROWS / 16u; ++kt) {
        const unsigned probs_kt_base = probs_lane_base + (unsigned)(kt * kPvAStepBytes);
        const uint32_t pv_row = kt * 16u + pv_row_lane;
#pragma unroll
        for (uint32_t mtile = 0; mtile < kMtiles; ++mtile) {
            uint32_t a[4];
            tt_ldmatrix_x4_addr(a, probs_kt_base + (unsigned)(mtile * kPvMtileBytes));
#pragma unroll
            for (uint32_t ntile = 0; ntile < kPvNTiles; ++ntile) {
                uint32_t b[2];
                const uint32_t idx = ((mtile * kPvNTiles + ntile) << 2);
                tt_ldmatrix_x2_trans_addr(
                    b,
                    kv_smem + tt_ring_off_bytes(pv_row, pv_chunk_base + ntile));
                tt_mma_m16n8k16_f16_f32(o_acc + idx, a, b);
            }
        }
    }
}

__device__ __forceinline__ void tt_pv_mma_epilogue(
        const float (&o_acc)[2u * kTTTileTokens * kTTG],
        const float * __restrict__ final_scale,
        float * __restrict__ heads,
        uint32_t base_position,
        uint32_t heads_stride,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t tile_base,
        uint32_t head_base) {
    constexpr uint32_t kMtiles = kTTM / 16u;
    constexpr uint32_t kPvWarpBase = 8u;
    constexpr uint32_t kPvNTiles = 8u;
    const uint32_t lane = tt_lane_id();
    const uint32_t warp = tt_warp_id();
    if (warp < kPvWarpBase) {
        return;
    }
    const uint32_t pv_warp = warp - kPvWarpBase;

#pragma unroll
    for (uint32_t mtile = 0; mtile < kMtiles; ++mtile) {
#pragma unroll
        for (uint32_t ntile = 0; ntile < kPvNTiles; ++ntile) {
            if (pv_warp == 7u) {
                // Warp 7 covers dimensions 448..511, where inverse RoPE rotation applies.
                // Pair 1: l = 0 (even), l = 1 (odd)
                const uint32_t idx0 = ((mtile * kPvNTiles + ntile) << 2) + 0u;
                const uint32_t idx1 = idx0 + 1u;
                const uint32_t m0 = mtile * 16u + (uint32_t)tt_mma_c_i(lane, 0);
                const uint32_t tok0 = m0 / kTTG;
                const uint32_t h0 = m0 - tok0 * kTTG;
                const uint32_t gt0 = tile_base + tok0;
                const uint32_t gh0 = head_base + h0;
                const uint32_t r0 = ntile * 8u + (uint32_t)tt_mma_c_j(lane, 0);
                const uint32_t d0 = 448u + r0;

                if (gt0 < n_tokens && gh0 < n_head) {
                    const float val0 = o_acc[idx0] * final_scale[m0];
                    const float val1 = o_acc[idx1] * final_scale[m0];
                    const float theta0 = ds4_compressed_theta(base_position + gt0, r0);
                    const float c0 = cosf(theta0);
                    const float s0 = -sinf(theta0);
                    float *dst0 = heads + (uint64_t)gt0 * heads_stride + (uint64_t)gh0 * kTTHeadDim + d0;
                    dst0[0] = val0 * c0 - val1 * s0;
                    dst0[1] = val0 * s0 + val1 * c0;
                }

                // Pair 2: l = 2 (even), l = 3 (odd)
                const uint32_t idx2 = idx0 + 2u;
                const uint32_t idx3 = idx0 + 3u;
                const uint32_t m2 = mtile * 16u + (uint32_t)tt_mma_c_i(lane, 2);
                const uint32_t tok2 = m2 / kTTG;
                const uint32_t h2 = m2 - tok2 * kTTG;
                const uint32_t gt2 = tile_base + tok2;
                const uint32_t gh2 = head_base + h2;
                const uint32_t r2 = r0;
                const uint32_t d2 = 448u + r2;

                if (gt2 < n_tokens && gh2 < n_head) {
                    const float val2 = o_acc[idx2] * final_scale[m2];
                    const float val3 = o_acc[idx3] * final_scale[m2];
                    const float theta2 = ds4_compressed_theta(base_position + gt2, r2);
                    const float c2 = cosf(theta2);
                    const float s2 = -sinf(theta2);
                    float *dst2 = heads + (uint64_t)gt2 * heads_stride + (uint64_t)gh2 * kTTHeadDim + d2;
                    dst2[0] = val2 * c2 - val3 * s2;
                    dst2[1] = val2 * s2 + val3 * c2;
                }
            } else {
                // Dimensions 0..447: unrotated
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const uint32_t idx = ((mtile * kPvNTiles + ntile) << 2) + (uint32_t)l;
                    const uint32_t m = mtile * 16u + (uint32_t)tt_mma_c_i(lane, l);
                    const uint32_t tok = m / kTTG;
                    const uint32_t h = m - tok * kTTG;
                    const uint32_t gt = tile_base + tok;
                    const uint32_t gh = head_base + h;
                    const uint32_t d =
                        pv_warp * 64u + ntile * 8u + (uint32_t)tt_mma_c_j(lane, l);
                    if (gt < n_tokens && gh < n_head) {
                        heads[(uint64_t)gt * heads_stride + (uint64_t)gh * kTTHeadDim + d] =
                            o_acc[idx] * final_scale[m];
                    }
                }
            }
        }
    }
}

template <uint32_t TT_STAGE_ROWS, uint32_t TT_G>
struct tt_TokentileSmemBudget {
    static constexpr uint32_t M = kTTTileTokens * TT_G;
    static constexpr uint32_t prob_stride = tt_TokentileLayout<TT_STAGE_ROWS>::prob_stride;
    static constexpr size_t q_bytes = 0;
    static constexpr size_t ring_bytes = 2ull * tt_TokentileLayout<TT_STAGE_ROWS>::ring_plane_bytes;
    static constexpr size_t p_bytes = 2ull * M * prob_stride * sizeof(half);
    static constexpr size_t partial_records = (size_t)M * TT_STAGE_ROWS;
    static constexpr size_t partial_bytes = partial_records * sizeof(float4);
    static constexpr size_t stats_bytes = 4ull * M * sizeof(float);
    static constexpr size_t record_bytes =
        (size_t)kTTRecordRingPlanes * TT_STAGE_ROWS * sizeof(int2);
    static constexpr size_t total =
        (ring_bytes + p_bytes + partial_bytes + stats_bytes + record_bytes + 15u) & ~size_t(15u);
};

extern "C" __global__ __launch_bounds__(512, 1) void ds4_attention_tokentile_dense_build(
        int2 *records,
        uint32_t *counts,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t ratio,
        uint32_t n_comp,
        uint32_t rec_stride) {
    __shared__ uint32_t visible_s[kTTTileTokens];
    const uint32_t tid = threadIdx.x;
    const uint32_t tile_base = blockIdx.x * kTTTileTokens;
    if (tile_base >= n_tokens) {
        if (tid == 0u) counts[blockIdx.x] = 0u;
        return;
    }
    const uint32_t tile_count =
        n_tokens - tile_base < kTTTileTokens
            ? n_tokens - tile_base : kTTTileTokens;
    if (tid < kTTTileTokens) {
        uint32_t visible = 0u;
        if (tid < tile_count && n_comp != 0u && ratio != 0u) {
            visible = (pos0 + tile_base + tid + 1u) / ratio;
            if (visible > n_comp) visible = n_comp;
        }
        visible_s[tid] = visible;
    }
    __syncthreads();

    uint32_t vmax = 0u;
    for (uint32_t i = 0; i < tile_count; i++) {
        if (visible_s[i] > vmax) vmax = visible_s[i];
    }
    for (uint32_t c = tid; c < vmax; c += blockDim.x) {
        uint32_t mask = 0u;
        for (uint32_t i = 0; i < tile_count; i++) {
            if (c < visible_s[i]) mask |= (1u << i);
        }
        records[(uint64_t)blockIdx.x * rec_stride + c] =
            make_int2((int)c, (int)mask);
    }
    if (tid == 0u) counts[blockIdx.x] = vmax;
}

extern "C" __global__ __launch_bounds__(512, 1) void ds4_attention_tokentile_union_build(
        int2 *records,
        uint32_t *counts,
        const int32_t *topk,
        const int32_t *positions,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t ratio,
        uint32_t n_comp,
        uint32_t rec_stride) {
    extern __shared__ uint32_t bitmap[];
    __shared__ uint32_t scan[513];
    __shared__ uint32_t running_s;

    const uint32_t tid = threadIdx.x;
    const uint32_t tile_base = blockIdx.x * kTTTileTokens;
    if (tile_base >= n_tokens) {
        if (tid == 0u) counts[blockIdx.x] = 0u;
        return;
    }
    const uint32_t tile_count =
        n_tokens - tile_base < kTTTileTokens ? n_tokens - tile_base : kTTTileTokens;
    const uint32_t bitmap_words = (n_comp + 1u) >> 1u;

    for (uint32_t w = tid; w < bitmap_words; w += blockDim.x) {
        bitmap[w] = 0u;
    }
    if (tid == 0u) running_s = 0u;
    __syncthreads();

    const uint32_t total_slots = tile_count * top_k;
    for (uint32_t idx = tid; idx < total_slots; idx += blockDim.x) {
        const uint32_t tok = idx / top_k;
        const uint32_t i = idx - tok * top_k;
        const uint32_t t = tile_base + tok;
        const uint32_t qpos = positions ? (uint32_t)positions[t] : pos0 + t;
        uint32_t visible = ratio ? (positions ? qpos / ratio : (qpos + 1u) / ratio) : n_comp;
        if (visible > n_comp) visible = n_comp;
        const int32_t c = topk[(uint64_t)t * top_k + i];
        if (c >= 0 && (uint32_t)c < visible) {
            const uint32_t cu = (uint32_t)c;
            const uint32_t bits = ((uint32_t)(1u << tok)) << ((cu & 1u) * 16u);
            atomicOr(&bitmap[cu >> 1u], bits);
        }
    }
    __syncthreads();

    for (uint32_t base = 0u; base < n_comp; base += blockDim.x) {
        const uint32_t id = base + tid;
        uint32_t mask = 0u;
        if (id < n_comp) {
            const uint32_t word = bitmap[id >> 1u];
            mask = (id & 1u) ? (word >> 16u) : (word & 0xffffu);
        }
        const uint32_t pred = mask != 0u ? 1u : 0u;
        if (tid == 0u) scan[0] = 0u;
        scan[tid + 1u] = pred;
        __syncthreads();
        for (uint32_t off = 1u; off < blockDim.x; off <<= 1u) {
            uint32_t v = 0u;
            if (tid >= off) v = scan[tid + 1u - off];
            __syncthreads();
            scan[tid + 1u] += v;
            __syncthreads();
        }
        const uint32_t rank = scan[tid];
        const uint32_t total = scan[blockDim.x];
        const uint32_t running = running_s;
        if (pred) {
            records[(uint64_t)blockIdx.x * rec_stride + running + rank] =
                make_int2((int)id, (int)mask);
        }
        __syncthreads();
        if (tid == 0u) running_s = running + total;
        __syncthreads();
    }

    if (tid == 0u) counts[blockIdx.x] = running_s;
}

extern "C" __global__ __launch_bounds__(256, 1) void ds4_attention_tokentile_raw_mirror(
        half *dst,
        const float *raw,
        const float *out,
        uint32_t base_position,
        uint32_t n_tokens,
        uint32_t raw_cap,
        uint32_t raw_row_min) {
    const uint32_t row = blockIdx.x;
    const uint32_t d0 = threadIdx.x << 1u;
    if (d0 >= 512u) return;

    half *dst_row = dst + (uint64_t)row * 512u;
    if (row < raw_row_min) {
        dst_row[d0] = __float2half(0.0f);
        dst_row[d0 + 1u] = __float2half(0.0f);
        return;
    }

    const int64_t p = (int64_t)base_position - 127 + (int64_t)row;
    if (p < 0) {
        dst_row[d0] = __float2half(0.0f);
        dst_row[d0 + 1u] = __float2half(0.0f);
        return;
    }

    const float *src;
    if ((uint32_t)p < base_position) {
        const uint32_t slot = (uint32_t)(p % raw_cap);
        src = raw + (uint64_t)slot * 512u;
    } else {
        const uint32_t t = (uint32_t)p - base_position;
        src = out + (uint64_t)t * 66048u + 32768u;
    }

    dst_row[d0]      = __float2half(src[d0]);
    dst_row[d0 + 1u] = __float2half(src[d0 + 1u]);
}

extern "C" __global__ __launch_bounds__(256, 1) void ds4_attention_tokentile_comp_mirror(
        half *dst,
        const float *comp_kv,
        uint32_t n_comp) {
    const uint32_t row = blockIdx.x;
    const uint32_t c4 = threadIdx.x;
    if (row >= n_comp || c4 >= 128u) return;
    const float4 v = ((const float4 *)(comp_kv + (uint64_t)row * 512u))[c4];
    half *out = dst + (uint64_t)row * 512u + (c4 << 2u);
    out[0] = __float2half(v.x);
    out[1] = __float2half(v.y);
    out[2] = __float2half(v.z);
    out[3] = __float2half(v.w);
}

extern "C" __global__ __launch_bounds__(512, 1) void ds4_attention_tokentile_hmma(
        float *heads,
        const float *sinks,
        const float *q,
        const half *raw_kv,
        const half *comp_kv,
        const int2 *union_records,
        const uint32_t *union_counts,
        uint32_t base_position,
        uint32_t q_stride,
        uint32_t heads_stride,
        uint32_t rec_stride,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t raw_row_min) {
    constexpr uint32_t kKvElems = tt_TokentileLayout<kTTStageRows>::ring_plane_elems;
    constexpr uint32_t kProbStride = tt_TokentileLayout<kTTStageRows>::prob_stride;
    const uint32_t tid = threadIdx.x;
    const uint32_t tile_idx = blockIdx.x;
    const uint32_t head_group = blockIdx.y;
    const uint32_t tile_base = tile_idx * kTTTileTokens;
    const uint32_t head_base = head_group * kTTG;
    if (tile_base >= n_tokens) {
        return;
    }
    const uint32_t tile_count =
        n_tokens - tile_base < kTTTileTokens ? n_tokens - tile_base : kTTTileTokens;
    const uint32_t raw_union_count = tile_count + kTTRawWindow - 1u;
    const uint32_t comp_union_count = union_counts[tile_idx];
    const uint32_t n_score = raw_union_count + comp_union_count;
    const uint64_t union_tile_off = (uint64_t)tile_idx * rec_stride;
    const int2 * __restrict__ union_records_tile = union_records + union_tile_off;
    const float score_scale = rsqrtf((float)kTTHeadDim);

    extern __shared__ unsigned char smem[];
    unsigned char *p = tt_align16(smem);
    half *kv_h = reinterpret_cast<half *>(p);
    p = tt_align16(p + 2u * tt_TokentileLayout<kTTStageRows>::ring_plane_bytes);
    half *probs = reinterpret_cast<half *>(p);
    p = tt_align16(p + 2u * kTTM * kProbStride * sizeof(half));
    float4 *score_scratch = reinterpret_cast<float4 *>(p);
    p = tt_align16(p + kTTM * kTTStageRows * sizeof(float4));
    float *max_s = reinterpret_cast<float *>(p);
    p = tt_align16(p + kTTM * sizeof(float));
    float *sum_s = reinterpret_cast<float *>(p);
    p = tt_align16(p + kTTM * sizeof(float));
    float *stage_rescale = reinterpret_cast<float *>(p);
    p = tt_align16(p + kTTM * sizeof(float));
    float *final_scale = reinterpret_cast<float *>(p);
    p = tt_align16(p + kTTM * sizeof(float));
    int2 *rec_ring = reinterpret_cast<int2 *>(p);

    for (uint32_t m = tid; m < kTTM; m += blockDim.x) {
        max_s[m] = -INFINITY;
        sum_s[m] = 0.0f;
        stage_rescale[m] = 1.0f;
        final_scale[m] = 0.0f;
    }
    __syncthreads();

    union tt_TokentileRoleRegs {
        uint32_t score_q_frag[kTTScoreKStepsPerQuarter][4];
        float o_acc[2u * kTTTileTokens * kTTG];
    };
    tt_TokentileRoleRegs role_regs;
    tt_load_score_q_frag(
        role_regs.score_q_frag, q, q_stride, n_tokens, n_head, tile_base, head_base);
    if (tt_warp_id() >= 8u) {
#pragma unroll
        for (uint32_t i = 0; i < 2u * kTTM; ++i) {
            role_regs.o_acc[i] = 0.0f;
        }
    }

    uint32_t cur = 0u;
    uint32_t free = 1u;
    if (n_score != 0u) {
        const uint32_t nr0 = n_score < kTTStageRows ? n_score : kTTStageRows;
        tt_issue_kv_stage_cp_async<kTTStageRows, false>(
            kv_h + cur * kKvElems,
            rec_ring,
            0u,
            nr0,
            raw_union_count,
            union_records_tile,
            tile_base,
            raw_kv,
            comp_kv,
            tid);
        tt_issue_record_stage_cp_async<kTTStageRows>(
            rec_ring,
            0u,
            nr0,
            raw_union_count,
            union_records_tile);
        if (kTTStageRows < n_score) {
            const uint32_t nr1 =
                n_score - kTTStageRows < kTTStageRows ? n_score - kTTStageRows : kTTStageRows;
            tt_issue_record_stage_cp_async<kTTStageRows>(
                rec_ring + kTTStageRows,
                kTTStageRows,
                nr1,
                raw_union_count,
                union_records_tile);
        }
        tt_cp_async_commit();
        tt_cp_async_wait_group<0>();
    }
    __syncthreads();

    uint32_t prob_cur = 0u;
    for (uint32_t row0 = 0u; row0 < n_score; row0 += kTTStageRows) {
        const uint32_t nr = n_score - row0 < kTTStageRows ? n_score - row0 : kTTStageRows;
        half *kv_cur = kv_h + cur * kKvElems;
        half *kv_free = kv_h + free * kKvElems;

        tt_hmma_score_stage<kTTStageRows>(
            score_scratch, role_regs.score_q_frag, kv_cur, nr, score_scale);
        if (row0 != 0u) {
            const half *kv_prev = kv_h + (cur ^ 1u) * kKvElems;
            const half *probs_prev = probs + (prob_cur ^ 1u) * kTTM * kProbStride;
            tt_pv_mma_stage<kTTStageRows>(
                role_regs.o_acc, probs_prev, stage_rescale, kv_prev);
        }
        __syncthreads();

        const uint32_t next_row0 = row0 + kTTStageRows;
        const bool has_next = next_row0 < n_score;
        if (has_next) {
            const uint32_t next_nr =
                n_score - next_row0 < kTTStageRows ? n_score - next_row0 : kTTStageRows;
            tt_issue_kv_stage_cp_async<kTTStageRows, true>(
                kv_free,
                rec_ring + (((row0 / kTTStageRows) + 1u) & 3u) * kTTStageRows,
                next_row0,
                next_nr,
                raw_union_count,
                union_records_tile,
                tile_base,
                raw_kv,
                comp_kv,
                tid);
            const uint32_t prefetch_row0 = next_row0 + kTTStageRows;
            if (prefetch_row0 < n_score) {
                const uint32_t prefetch_nr =
                    n_score - prefetch_row0 < kTTStageRows
                        ? n_score - prefetch_row0
                        : kTTStageRows;
                tt_issue_record_stage_cp_async<kTTStageRows>(
                    rec_ring + (((row0 / kTTStageRows) + 2u) & 3u) * kTTStageRows,
                    prefetch_row0,
                    prefetch_nr,
                    raw_union_count,
                    union_records_tile);
            }
            tt_cp_async_commit();
        }

        half *probs_cur = probs + prob_cur * kTTM * kProbStride;
        tt_softmax_stage<kTTStageRows>(
            probs_cur,
            stage_rescale,
            max_s,
            sum_s,
            score_scratch,
            rec_ring + ((row0 / kTTStageRows) & 3u) * kTTStageRows,
            row0,
            nr,
            raw_union_count,
            tile_count,
            tile_base,
            raw_row_min);
        tt_cp_async_wait_group<0>();
        __syncthreads();
        cur ^= 1u;
        free ^= 1u;
        prob_cur ^= 1u;
    }

    if (n_score != 0u) {
        const half *kv_prev = kv_h + (cur ^ 1u) * kKvElems;
        const half *probs_prev = probs + (prob_cur ^ 1u) * kTTM * kProbStride;
        tt_pv_mma_stage<kTTStageRows>(
            role_regs.o_acc, probs_prev, stage_rescale, kv_prev);
    }
    __syncthreads();

    for (uint32_t m = tid; m < kTTM; m += blockDim.x) {
        const uint32_t tok = m / kTTG;
        const uint32_t h = m - tok * kTTG;
        const uint32_t gt = tile_base + tok;
        const uint32_t gh = head_base + h;
        if (gt < n_tokens && gh < n_head) {
            const float sink = sinks[gh];
            const float old_m = max_s[m];
            const float new_m = fmaxf(old_m, sink);
            const float old_scale = old_m == -INFINITY ? 0.0f : expf(old_m - new_m);
            const float sink_scale = expf(sink - new_m);
            const float den = sum_s[m] * old_scale + sink_scale;
            final_scale[m] = den == 0.0f ? 0.0f : old_scale / den;
        } else {
            final_scale[m] = 0.0f;
        }
    }
    __syncthreads();

    tt_pv_mma_epilogue(
        role_regs.o_acc, final_scale, heads, base_position, heads_stride, n_tokens, n_head, tile_base, head_base);
}

#endif // DS4_ATTENTION_HMMA_CU
