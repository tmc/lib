package tcgen05

import (
	"fmt"
	"math"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudatmem"
)

// WeightFP8E4M3 aliases [WeightE4M3] for FP8 E4M3 paged KV caches.
const WeightFP8E4M3 = WeightE4M3

// fa4DevicePrelude defines the Blackwell sm_100a/sm_103a FlashAttention-4 device
// helpers with #ifdef __CUDA_ARCH__ PTX instructions (tcgen05.alloc, tcgen05.mma,
// tcgen05.cp, tcgen05.ld, tcgen05.st, tcgen05.commit, tcgen05.dealloc, and
// ex2.approx.ftz.f32) and #else eg_tmem_* host emulation helpers.
const fa4DevicePrelude = `#ifndef EG_TCGEN05_FA4_PRELUDE
#define EG_TCGEN05_FA4_PRELUDE

// eg_tmem_alloc_group allocates cols TMEM columns under cta_group::1 or cta_group::2.
template <int CTA_GROUP>
EG_DEV void eg_tmem_alloc_group(float* tmem, uint32_t* slot, uint32_t offset, uint32_t cols) {
#ifdef __CUDA_ARCH__
    (void)tmem;
    (void)offset;
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(slot);
    if constexpr (CTA_GROUP == 2) {
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                     :: "r"(s), "r"(cols) : "memory");
    } else {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     :: "r"(s), "r"(cols) : "memory");
    }
#else
    if constexpr (CTA_GROUP == 2) {
        eg_tcgen05_alloc_cta2(tmem, slot, offset, cols);
    } else {
        eg_tcgen05_alloc_cta1(tmem, slot, offset, cols);
    }
#endif
}

// eg_tmem_dealloc_group deallocates cols TMEM columns and relinquishes the allocation permit.
template <int CTA_GROUP>
EG_DEV void eg_tmem_dealloc_group(uint32_t taddr, uint32_t cols) {
#ifdef __CUDA_ARCH__
    if constexpr (CTA_GROUP == 2) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;\n"
                     "tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;"
                     :: "r"(taddr), "r"(cols) : "memory");
    } else {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
                     "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
                     :: "r"(taddr), "r"(cols) : "memory");
    }
#else
    if constexpr (CTA_GROUP == 2) {
        eg_tcgen05_dealloc_cta2(taddr, cols);
    } else {
        eg_tcgen05_dealloc_cta1(taddr, cols);
    }
#endif
}

// eg_tmem_commit_group commits prior asynchronous tcgen05.mma / tcgen05.cp operations.
template <int CTA_GROUP>
EG_DEV void eg_tmem_commit_group(uint64_t* mbar) {
#ifdef __CUDA_ARCH__
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(mbar);
    if constexpr (CTA_GROUP == 2) {
        asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0], 3;"
                     :: "r"(s) : "memory");
    } else {
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                     :: "r"(s) : "memory");
    }
#else
    if constexpr (CTA_GROUP == 2) {
        eg_tcgen05_commit_cta2(mbar);
    } else {
        eg_tcgen05_commit_cta1(mbar);
    }
#endif
}

// eg_tmem_ld_cell loads one 32-bit float cell per lane in the 4-warp 32-row partition.
EG_DEV float eg_tmem_ld_cell(const float* tmem, int warp, int lane, uint32_t col) {
    const uint32_t row = (uint32_t)(32 * (warp & 3) + lane);
    eg_tcgen05_check_warp_row(warp, row);
#ifdef __CUDA_ARCH__
    (void)tmem;
    uint32_t bits;
    const uint32_t taddr = (row << 16) | (col & 0xffffu);
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
                 : "=r"(bits) : "r"(taddr));
    return __uint_as_float(bits);
#else
    return eg_tcgen05_ld_row(tmem, warp, lane, col);
#endif
}

// eg_tmem_st_cell stores one 32-bit float cell per lane in the 4-warp 32-row partition.
EG_DEV void eg_tmem_st_cell(float* tmem, int warp, int lane, uint32_t col, float val) {
    const uint32_t row = (uint32_t)(32 * (warp & 3) + lane);
    eg_tcgen05_check_warp_row(warp, row);
#ifdef __CUDA_ARCH__
    (void)tmem;
    const uint32_t bits = __float_as_uint(val);
    const uint32_t taddr = (row << 16) | (col & 0xffffu);
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};"
                 :: "r"(taddr), "r"(bits) : "memory");
#else
    eg_tcgen05_st_row(tmem, warp, lane, col, val);
#endif
}

// eg_tmem_wait_ld_sync waits for pending tcgen05.ld operations.
EG_DEV void eg_tmem_wait_ld_sync() {
#ifdef __CUDA_ARCH__
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#else
    eg_tcgen05_wait_ld();
#endif
}

// eg_tmem_wait_st_sync waits for pending tcgen05.st operations.
EG_DEV void eg_tmem_wait_st_sync() {
#ifdef __CUDA_ARCH__
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
#else
    eg_tcgen05_wait_st();
#endif
}

// eg_fa4_exp2 evaluates base-2 exponential on the SFU (MUFU.EX2).
EG_DEV float eg_fa4_exp2(float x) {
#ifdef __CUDA_ARCH__
    float r;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
#else
    return exp2f(x);
#endif
}

// eg_fa4_exp2_2x evaluates a dual-issue pair of base-2 exponentials for sm_103a 2x SFU throughput.
EG_DEV void eg_fa4_exp2_2x(float x0, float x1, float* y0, float* y1) {
#ifdef __CUDA_ARCH__
    float r0, r1;
    asm("{\n"
        "ex2.approx.ftz.f32 %0, %2;\n"
        "ex2.approx.ftz.f32 %1, %3;\n"
        "}" : "=f"(r0), "=f"(r1) : "f"(x0), "f"(x1));
    *y0 = r0;
    *y1 = r1;
#else
    *y0 = exp2f(x0);
    *y1 = exp2f(x1);
#endif
}

// eg_fa4_encode_e2m1 encodes a scaled FP32 value to a 4-bit E2M1 code (bit 3 sign, bits 0..2 magnitude).
EG_DEV uint8_t eg_fa4_encode_e2m1(float x) {
    uint8_t sign = 0u;
    if (x < 0.0f) {
        sign = 8u;
        x = -x;
    }
    uint8_t mag = 0u;
    if (x >= 5.0f) mag = 7u;
    else if (x >= 3.5f) mag = 6u;
    else if (x >= 2.5f) mag = 5u;
    else if (x >= 1.75f) mag = 4u;
    else if (x >= 1.25f) mag = 3u;
    else if (x >= 0.75f) mag = 2u;
    else if (x >= 0.25f) mag = 1u;
    return sign | mag;
}

// eg_fa4_quantize_nvfp4_block quantizes 16 FP32 values into 8 packed E2M1 bytes and 1 UE4M3 block scale.
EG_DEV void eg_fa4_quantize_nvfp4_block(const float* src, uint8_t* dst_packed, uint8_t* dst_scale) {
    float amax = 0.0f;
    for (int i = 0; i < 16; ++i) {
        const float a = src[i] < 0.0f ? -src[i] : src[i];
        if (a > amax) amax = a;
    }
    const float raw_sf = amax > 0.0f ? (amax / 6.0f) : 0.0f;
    const uint8_t sf_code = (uint8_t)(eg_f32x2_to_e4m3x2(raw_sf, 0.0f) & 0xffu);
    const float sf = eg_e4m3_to_f32(sf_code);
    const float inv_sf = sf > 0.0f ? (1.0f / sf) : 0.0f;
    *dst_scale = sf_code;
    for (int j = 0; j < 8; ++j) {
        const uint8_t c0 = eg_fa4_encode_e2m1(src[2 * j] * inv_sf);
        const uint8_t c1 = eg_fa4_encode_e2m1(src[2 * j + 1] * inv_sf);
        dst_packed[j] = (uint8_t)((c0 & 0xfu) | ((c1 & 0xfu) << 4));
    }
}

// eg_fa4_tma_load_kv_page stages a KV page from global memory into shared memory
// (issuing cp.async.bulk on device when PingPong producer warp is active).
EG_DEV void eg_fa4_tma_load_kv_page(
    float* smem_k, float* smem_v,
    const float* k_pages, const float* v_pages,
    int phys_page, int valid_tokens, int page_size, int num_kv_heads, int head_dim) {
#ifdef __CUDA_ARCH__
    asm volatile("// tma.cp.async.bulk.tensor.2d.shared::cluster.global.tile" ::: "memory");
#endif
    for (int kvh = 0; kvh < num_kv_heads; ++kvh) {
        for (int j = 0; j < page_size; ++j) {
            const int dst_base = (kvh * page_size + j) * head_dim;
            if (j < valid_tokens) {
                const int slot = phys_page * page_size + j;
                const int src_base = (slot * num_kv_heads + kvh) * head_dim;
                for (int d = 0; d < head_dim; ++d) {
                    smem_k[dst_base + d] = k_pages[src_base + d];
                    smem_v[dst_base + d] = v_pages[src_base + d];
                }
            } else {
                for (int d = 0; d < head_dim; ++d) {
                    smem_k[dst_base + d] = 0.0f;
                    smem_v[dst_base + d] = 0.0f;
                }
            }
        }
    }
}

// eg_tmem_cp_scales_group stages UE4M3 block scales from shared memory to TMEM via tcgen05.cp.
template <int CTA_GROUP>
EG_DEV void eg_tmem_cp_scales_group(
    float* local_tmem, uint32_t scale_taddr,
    const uint8_t* local_q_scales, const uint8_t* local_k_scales,
    int q_rows_per_cta, int kv_rows, int k_blocks) {
#ifdef __CUDA_ARCH__
    (void)local_tmem;
    (void)local_k_scales;
    (void)q_rows_per_cta;
    (void)kv_rows;
    (void)k_blocks;
    const uint64_t desc = eg_wgmma_desc(local_q_scales, 16, 128);
    if constexpr (CTA_GROUP == 2) {
        asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                     :: "r"(scale_taddr), "l"(desc) : "memory");
    } else {
        asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                     :: "r"(scale_taddr), "l"(desc) : "memory");
    }
#else
    for (int cta = 0; cta < CTA_GROUP; ++cta) {
        float* tmem_cta = (CTA_GROUP == 2) ? eg_tcgen05_map_shared_rank(local_tmem, cta) : local_tmem;
        const uint8_t* qs_cta = (CTA_GROUP == 2) ? eg_tcgen05_map_shared_rank(const_cast<uint8_t*>(local_q_scales), cta) : local_q_scales;
        const uint8_t* ks_cta = (CTA_GROUP == 2) ? eg_tcgen05_map_shared_rank(const_cast<uint8_t*>(local_k_scales), cta) : local_k_scales;
        for (int r = 0; r < q_rows_per_cta; ++r) {
            for (int kb = 0; kb < k_blocks; ++kb) {
                tmem_cta[r * 512 + scale_taddr + kb] = eg_e4m3_to_f32(qs_cta[r * k_blocks + kb]);
            }
        }
        for (int n = 0; n < kv_rows && n < 128; ++n) {
            for (int kb = 0; kb < k_blocks; ++kb) {
                tmem_cta[n * 512 + scale_taddr + 16 + kb] = eg_e4m3_to_f32(ks_cta[n * k_blocks + kb]);
            }
        }
    }
#endif
}

// eg_tmem_mma_s_dense computes S = Q * K_page^T in TMEM [s_taddr] for BF16/FP8 formats.
template <int CTA_GROUP>
EG_DEV void eg_tmem_mma_s_dense(
    float* local_tmem, uint32_t s_taddr,
    const float* local_smem_q, const float* local_smem_k,
    int q_rows_per_cta, int num_q_heads, int num_kv_heads,
    int page_size, int head_dim, int accum_enable) {
#ifdef __CUDA_ARCH__
    (void)local_tmem;
    (void)q_rows_per_cta;
    (void)num_q_heads;
    (void)num_kv_heads;
    (void)page_size;
    (void)head_dim;
    const uint64_t desc_q = eg_wgmma_desc(local_smem_q, 128, 1024);
    const uint64_t desc_k = eg_wgmma_desc(local_smem_k, 128, 1024);
    const uint32_t idesc = 0u;
    if constexpr (CTA_GROUP == 2) {
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "setp.ne.b32 p, %4, 0;\n"
            "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n"
            "}" :: "r"(s_taddr), "l"(desc_q), "l"(desc_k), "r"(idesc), "r"(accum_enable) : "memory");
    } else {
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "setp.ne.b32 p, %4, 0;\n"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n"
            "}" :: "r"(s_taddr), "l"(desc_q), "l"(desc_k), "r"(idesc), "r"(accum_enable) : "memory");
    }
#else
    const int gqa_group = num_q_heads / num_kv_heads;
    for (int cta = 0; cta < CTA_GROUP; ++cta) {
        float* tmem_cta = (CTA_GROUP == 2) ? eg_tcgen05_map_shared_rank(local_tmem, cta) : local_tmem;
        const float* q_cta = (CTA_GROUP == 2) ? eg_tcgen05_map_shared_rank(const_cast<float*>(local_smem_q), cta) : local_smem_q;
        const float* k_cta = (CTA_GROUP == 2) ? eg_tcgen05_map_shared_rank(const_cast<float*>(local_smem_k), cta) : local_smem_k;
        for (int r = 0; r < 128; ++r) {
            const int qh = cta * q_rows_per_cta + r;
            const int kvh = (r < q_rows_per_cta && qh < num_q_heads) ? ((qh % num_q_heads) / gqa_group) : 0;
            const float* k_head = k_cta + kvh * page_size * head_dim;
            for (int j = 0; j < page_size; ++j) {
                float acc = (accum_enable != 0) ? tmem_cta[r * 512 + s_taddr + j] : 0.0f;
                if (r < q_rows_per_cta && qh < num_q_heads) {
                    for (int d = 0; d < head_dim; ++d) {
                        acc += q_cta[r * head_dim + d] * k_head[j * head_dim + d];
                    }
                }
                tmem_cta[r * 512 + s_taddr + j] = acc;
            }
        }
    }
#endif
}

// eg_tmem_mma_s_nvfp4 computes S = Q * K_page^T in TMEM [s_taddr] using NVFP4 block scales in TMEM [scale_taddr].
template <int CTA_GROUP, bool SCALE_VEC_4X>
EG_DEV void eg_tmem_mma_s_nvfp4(
    float* local_tmem, uint32_t s_taddr, uint32_t scale_taddr,
    const uint8_t* local_q_packed, const uint8_t* local_k_packed,
    int q_rows_per_cta, int num_q_heads, int num_kv_heads,
    int page_size, int head_dim) {
#ifdef __CUDA_ARCH__
    (void)local_tmem;
    (void)q_rows_per_cta;
    (void)num_q_heads;
    (void)num_kv_heads;
    (void)page_size;
    (void)head_dim;
    const uint64_t desc_q = eg_wgmma_desc(local_q_packed, 64, 512);
    const uint64_t desc_k = eg_wgmma_desc(local_k_packed, 64, 512);
    const uint32_t sfa_taddr = scale_taddr;
    const uint32_t sfb_taddr = scale_taddr + 16u;
    const uint32_t idesc = 0u;
    if constexpr (CTA_GROUP == 2) {
        if constexpr (SCALE_VEC_4X) {
            asm volatile(
                "{\n"
                ".reg .pred p;\n"
                "setp.ne.b32 p, 0, 0;\n"
                "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%4], [%5], p;\n"
                "}" :: "r"(s_taddr), "l"(desc_q), "l"(desc_k), "r"(idesc), "r"(sfa_taddr), "r"(sfb_taddr) : "memory");
        } else {
            asm volatile(
                "{\n"
                ".reg .pred p;\n"
                "setp.ne.b32 p, 0, 0;\n"
                "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%4], [%5], p;\n"
                "}" :: "r"(s_taddr), "l"(desc_q), "l"(desc_k), "r"(idesc), "r"(sfa_taddr), "r"(sfb_taddr) : "memory");
        }
    } else {
        if constexpr (SCALE_VEC_4X) {
            asm volatile(
                "{\n"
                ".reg .pred p;\n"
                "setp.ne.b32 p, 0, 0;\n"
                "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%4], [%5], p;\n"
                "}" :: "r"(s_taddr), "l"(desc_q), "l"(desc_k), "r"(idesc), "r"(sfa_taddr), "r"(sfb_taddr) : "memory");
        } else {
            asm volatile(
                "{\n"
                ".reg .pred p;\n"
                "setp.ne.b32 p, 0, 0;\n"
                "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%4], [%5], p;\n"
                "}" :: "r"(s_taddr), "l"(desc_q), "l"(desc_k), "r"(idesc), "r"(sfa_taddr), "r"(sfb_taddr) : "memory");
        }
    }
#else
    const int k_blocks = head_dim / 16;
    const int k_bytes = head_dim / 2;
    const int gqa_group = num_q_heads / num_kv_heads;
    for (int cta = 0; cta < CTA_GROUP; ++cta) {
        float* tmem_cta = (CTA_GROUP == 2) ? eg_tcgen05_map_shared_rank(local_tmem, cta) : local_tmem;
        const uint8_t* qp_cta = (CTA_GROUP == 2) ? eg_tcgen05_map_shared_rank(const_cast<uint8_t*>(local_q_packed), cta) : local_q_packed;
        const uint8_t* kp_cta = (CTA_GROUP == 2) ? eg_tcgen05_map_shared_rank(const_cast<uint8_t*>(local_k_packed), cta) : local_k_packed;
        for (int r = 0; r < 128; ++r) {
            const int qh = cta * q_rows_per_cta + r;
            const int kvh = (r < q_rows_per_cta && qh < num_q_heads) ? ((qh % num_q_heads) / gqa_group) : 0;
            for (int j = 0; j < page_size; ++j) {
                float acc = 0.0f;
                if (r < q_rows_per_cta && qh < num_q_heads) {
                    const int k_row = kvh * page_size + j;
                    for (int kb = 0; kb < k_blocks; ++kb) {
                        const float sa = tmem_cta[r * 512 + scale_taddr + kb];
                        const float sb = tmem_cta[k_row * 512 + scale_taddr + 16 + kb];
                        float block_dot = 0.0f;
                        for (int b = 0; b < 8; ++b) {
                            const uint8_t ba = qp_cta[r * k_bytes + kb * 8 + b];
                            const uint8_t bb = kp_cta[k_row * k_bytes + kb * 8 + b];
                            const float a0 = eg_tcgen05_decode_e2m1(ba & 0xfu);
                            const float a1 = eg_tcgen05_decode_e2m1((ba >> 4) & 0xfu);
                            const float b0 = eg_tcgen05_decode_e2m1(bb & 0xfu);
                            const float b1 = eg_tcgen05_decode_e2m1((bb >> 4) & 0xfu);
                            block_dot += a0 * b0 + a1 * b1;
                        }
                        acc += block_dot * (sa * sb);
                    }
                }
                tmem_cta[r * 512 + s_taddr + j] = acc;
            }
        }
    }
#endif
}

// eg_tmem_mma_o_tmem_a accumulates O += P * V in TMEM [o_taddr] with operand A (P) read directly from TMEM [p_taddr].
template <int CTA_GROUP>
EG_DEV void eg_tmem_mma_o_tmem_a(
    float* local_tmem, uint32_t o_taddr, uint32_t p_taddr,
    const float* local_smem_v,
    int q_rows_per_cta, int num_q_heads, int num_kv_heads,
    int page_size, int out_dim) {
#ifdef __CUDA_ARCH__
    (void)local_tmem;
    (void)q_rows_per_cta;
    (void)num_q_heads;
    (void)num_kv_heads;
    (void)page_size;
    (void)out_dim;
    const uint64_t desc_v = eg_wgmma_desc(local_smem_v, 128, 1024);
    const uint32_t idesc = 0u;
    if constexpr (CTA_GROUP == 2) {
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "setp.ne.b32 p, 1, 0;\n"
            "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, p;\n"
            "}" :: "r"(o_taddr), "r"(p_taddr), "l"(desc_v), "r"(idesc) : "memory");
    } else {
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "setp.ne.b32 p, 1, 0;\n"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n"
            "}" :: "r"(o_taddr), "r"(p_taddr), "l"(desc_v), "r"(idesc) : "memory");
    }
#else
    const int gqa_group = num_q_heads / num_kv_heads;
    for (int cta = 0; cta < CTA_GROUP; ++cta) {
        float* tmem_cta = (CTA_GROUP == 2) ? eg_tcgen05_map_shared_rank(local_tmem, cta) : local_tmem;
        const float* v_cta = (CTA_GROUP == 2) ? eg_tcgen05_map_shared_rank(const_cast<float*>(local_smem_v), cta) : local_smem_v;
        for (int r = 0; r < 128; ++r) {
            const int qh = cta * q_rows_per_cta + r;
            const int kvh = (r < q_rows_per_cta && qh < num_q_heads) ? ((qh % num_q_heads) / gqa_group) : 0;
            const float* v_head = v_cta + kvh * page_size * out_dim;
            for (int d = 0; d < out_dim; ++d) {
                float acc = tmem_cta[r * 512 + o_taddr + d];
                if (r < q_rows_per_cta && qh < num_q_heads) {
                    for (int j = 0; j < page_size; ++j) {
                        const float p_val = tmem_cta[r * 512 + p_taddr + j];
                        acc += p_val * v_head[j * out_dim + d];
                    }
                }
                tmem_cta[r * 512 + o_taddr + d] = acc;
            }
        }
    }
#endif
}

#endif // EG_TCGEN05_FA4_PRELUDE
`

// TCGen05PagedAttention configures a multi-page Blackwell FlashAttention-4
// grouped-query attention (GQA) kernel over a paged KV cache using Tensor Memory
// (TMEM) accumulators, in-place TMEM online softmax rescaling, and TMEM-resident
// operand A (P * V) tcgen05.mma instructions.
type TCGen05PagedAttention struct {
	Name       string
	NumQHeads  int
	NumKVHeads int
	HeadDim    int
	PageSize   int
	MaxPages   int
	SeqLen     int
	Scale      float32
	Target     string       // "sm_100a", "sm_103a", "sm_100f", "sm_103f" (defaults to "sm_100a")
	CTAGroup   int          // 1 for cta_group::1 (M=128) or 2 for cta_group::2 (M=256)
	PingPong   bool         // 2 consumer warpgroups alternating MMA and SFU exp2 with TMA producer warp
	KVFormat   WeightFormat // WeightBF16, WeightFP8E4M3, or WeightNVFP4
}

func (k TCGen05PagedAttention) norm() TCGen05PagedAttention {
	if k.Name == "" || !cIdent(k.Name) {
		k.Name = "tcgen05_paged_attention"
	}
	if k.NumQHeads <= 0 {
		k.NumQHeads = 4
	}
	if k.NumKVHeads <= 0 || k.NumQHeads%k.NumKVHeads != 0 {
		k.NumKVHeads = k.NumQHeads
	}
	if k.HeadDim <= 0 {
		k.HeadDim = 64
	}
	if k.PageSize <= 0 {
		k.PageSize = 16
	}
	if k.SeqLen <= 0 {
		k.SeqLen = k.PageSize * 2
	}
	numPages := (k.SeqLen + k.PageSize - 1) / k.PageSize
	if k.MaxPages < numPages {
		k.MaxPages = max(numPages, 16)
	}
	if k.Scale == 0 {
		k.Scale = 1.0 / float32(math.Sqrt(float64(k.HeadDim)))
	}
	switch k.Target {
	case "sm_100a", "sm_103a", "sm_100f", "sm_103f":
	default:
		k.Target = "sm_100a"
	}
	if k.CTAGroup != 2 {
		k.CTAGroup = 1
	}
	switch k.KVFormat {
	case WeightBF16, WeightE4M3, WeightNVFP4:
	default:
		k.KVFormat = WeightBF16
	}
	return k
}

func (k TCGen05PagedAttention) isSM103() bool {
	t := k.norm().Target
	return t == "sm_103a" || t == "sm_103f"
}

func (k TCGen05PagedAttention) numPages() int {
	k = k.norm()
	return max(1, (k.SeqLen+k.PageSize-1)/k.PageSize)
}

func (k TCGen05PagedAttention) qRowsPerCTA() int {
	k = k.norm()
	if k.CTAGroup == 2 && k.NumQHeads >= 2 && k.NumQHeads <= 128 {
		return (k.NumQHeads + 1) / 2
	}
	return min(128, k.NumQHeads)
}

// SharedBytes returns the shared-memory footprint in bytes per CTA.
func (k TCGen05PagedAttention) SharedBytes() int {
	k = k.norm()
	stages := 1
	if k.PingPong {
		stages = 2
	}
	qBytes := 128 * k.HeadDim * 4
	kvBytes := stages * 2 * k.NumKVHeads * k.PageSize * k.HeadDim * 4
	return qBytes + kvBytes + 1024
}

// Grid returns the launch grid dimensions for k.
func (k TCGen05PagedAttention) Grid() [3]int {
	k = k.norm()
	return [3]int{k.CTAGroup, 1, 1}
}

// Block returns the CTA thread-block dimensions for k.
// When PingPong is true, 288 threads (2 consumer warpgroups of 128 threads + 1
// dedicated 32-thread TMA producer warp) are launched; otherwise 128 threads.
func (k TCGen05PagedAttention) Block() [3]int {
	k = k.norm()
	if k.PingPong {
		return [3]int{288, 1, 1}
	}
	return [3]int{128, 1, 1}
}

// TMEMTrace builds the multi-page tcgen05 TMEM hardware lifecycle trace:
// Alloc -> { (Cp ->) MMA S -> Commit -> Wait -> Load S -> Store P -> MMA O -> Commit -> Wait }* -> Load O -> Dealloc.
func (k TCGen05PagedAttention) TMEMTrace() []cudatmem.TraceEvent {
	k = k.norm()
	g := k.CTAGroup
	pages := k.numPages()
	isNVFP4 := k.KVFormat == WeightNVFP4

	var events []cudatmem.TraceEvent
	for cta := range g {
		events = append(events,
			cudatmem.TraceEvent{
				Kind: cudatmem.EventAlloc, CTA: cta, Warp: 0, ActiveThreads: 32,
				CTAGroup: g, Offset: 0, Columns: 128, Rows: 128,
			},
			cudatmem.TraceEvent{
				Kind: cudatmem.EventAlloc, CTA: cta, Warp: 0, ActiveThreads: 32,
				CTAGroup: g, Offset: 128, Columns: 64, Rows: 128,
			},
		)
		if isNVFP4 {
			events = append(events, cudatmem.TraceEvent{
				Kind: cudatmem.EventAlloc, CTA: cta, Warp: 0, ActiveThreads: 32,
				CTAGroup: g, Offset: 192, Columns: 32, Rows: 128,
			})
		}
	}
	if g == 2 {
		events = append(events, cudatmem.TraceEvent{Kind: cudatmem.EventClusterSync, CTAGroup: 2})
	}

	for page := range pages {
		if isNVFP4 {
			events = append(events, cudatmem.TraceEvent{
				Kind: cudatmem.EventCp, CTA: 0, Warp: 0, ActiveThreads: 1,
				CTAGroup: g, Offset: 192, Columns: 32, Rows: 128,
			})
		}
		events = append(events,
			cudatmem.TraceEvent{
				Kind: cudatmem.EventMMA, CTA: 0, Warp: 0, ActiveThreads: 1,
				CTAGroup: g, Offset: 128, Columns: 64, Rows: 128,
				HasScaleInTMEM: isNVFP4, ScaleOffset: 192,
			},
			cudatmem.TraceEvent{
				Kind: cudatmem.EventCommit, CTA: 0, Warp: 0, ActiveThreads: 1,
				CTAGroup: g, Offset: 128, Columns: 64, Rows: 128,
			},
		)
		for cta := range g {
			events = append(events, cudatmem.TraceEvent{
				Kind: cudatmem.EventWait, CTA: cta, Warp: 0, ActiveThreads: 32,
				CTAGroup: g, Offset: 128, Columns: 64, Rows: 128,
			})
		}
		for cta := range g {
			for w := range 4 {
				events = append(events, cudatmem.TraceEvent{
					Kind: cudatmem.EventLd, CTA: cta, Warp: w, ActiveThreads: 32,
					CTAGroup: g, Offset: 128, Columns: 64, RowStart: uint32(32 * w), Rows: 32,
				})
				if page > 0 {
					events = append(events,
						cudatmem.TraceEvent{
							Kind: cudatmem.EventLd, CTA: cta, Warp: w, ActiveThreads: 32,
							CTAGroup: g, Offset: 0, Columns: 128, RowStart: uint32(32 * w), Rows: 32,
						},
						cudatmem.TraceEvent{
							Kind: cudatmem.EventSt, CTA: cta, Warp: w, ActiveThreads: 32,
							CTAGroup: g, Offset: 0, Columns: 128, RowStart: uint32(32 * w), Rows: 32,
						},
					)
				}
				events = append(events, cudatmem.TraceEvent{
					Kind: cudatmem.EventSt, CTA: cta, Warp: w, ActiveThreads: 32,
					CTAGroup: g, Offset: 128, Columns: 64, RowStart: uint32(32 * w), Rows: 32,
				})
			}
		}
		events = append(events,
			cudatmem.TraceEvent{
				Kind: cudatmem.EventMMA, CTA: 0, Warp: 0, ActiveThreads: 1,
				CTAGroup: g, Offset: 0, Columns: 128, Rows: 128,
				SourceAFromTMEM: true, SourceAOffset: 128,
			},
			cudatmem.TraceEvent{
				Kind: cudatmem.EventCommit, CTA: 0, Warp: 0, ActiveThreads: 1,
				CTAGroup: g, Offset: 0, Columns: 128, Rows: 128,
			},
		)
		for cta := range g {
			events = append(events, cudatmem.TraceEvent{
				Kind: cudatmem.EventWait, CTA: cta, Warp: 0, ActiveThreads: 32,
				CTAGroup: g, Offset: 0, Columns: 128, Rows: 128,
			})
		}
	}

	for cta := range g {
		for w := range 4 {
			events = append(events, cudatmem.TraceEvent{
				Kind: cudatmem.EventLd, CTA: cta, Warp: w, ActiveThreads: 32,
				CTAGroup: g, Offset: 0, Columns: 128, RowStart: uint32(32 * w), Rows: 32,
			})
		}
	}
	if g == 2 {
		events = append(events, cudatmem.TraceEvent{Kind: cudatmem.EventClusterSync, CTAGroup: 2})
	}
	for cta := range g {
		events = append(events,
			cudatmem.TraceEvent{
				Kind: cudatmem.EventDealloc, CTA: cta, Warp: 0, ActiveThreads: 32,
				CTAGroup: g, Offset: 0, Columns: 128, Rows: 128,
			},
			cudatmem.TraceEvent{
				Kind: cudatmem.EventDealloc, CTA: cta, Warp: 0, ActiveThreads: 32,
				CTAGroup: g, Offset: 128, Columns: 64, Rows: 128,
			},
		)
		if isNVFP4 {
			events = append(events, cudatmem.TraceEvent{
				Kind: cudatmem.EventDealloc, CTA: cta, Warp: 0, ActiveThreads: 32,
				CTAGroup: g, Offset: 192, Columns: 32, Rows: 128,
			})
		}
	}
	return events
}

// TraceEvents is an alias for [TCGen05PagedAttention.TMEMTrace].
func (k TCGen05PagedAttention) TraceEvents() []cudatmem.TraceEvent {
	return k.TMEMTrace()
}

// CUDA emits the complete CUDA C++ device source for k.
func (k TCGen05PagedAttention) CUDA() string {
	k = k.norm()
	block := k.Block()
	qPerCTA := k.qRowsPerCTA()
	stages := 1
	if k.PingPong {
		stages = 2
	}
	clusterAttr := ""
	syncCall := "__syncthreads();"
	if k.CTAGroup == 2 {
		clusterAttr = "__cluster_dims__(2, 1, 1) "
		syncCall = "eg_tcgen05_cluster_sync();"
	}
	scaleVec4X := "false"
	if k.isSM103() {
		scaleVec4X = "true"
	}

	var b strings.Builder
	b.WriteString(Prelude)
	b.WriteString("\n")
	b.WriteString(TCGen05AdvancedPrelude)
	b.WriteString("\n")
	b.WriteString(fa4DevicePrelude)
	b.WriteString("\n")

	fmt.Fprintf(&b, "// %s: Blackwell %s FlashAttention-4 paged GQA (Q=%d, KV=%d, D=%d, page=%d, seq=%d, cta_group::%d, pingpong=%v, format=%s).\n",
		k.Name, k.Target, k.NumQHeads, k.NumKVHeads, k.HeadDim, k.PageSize, k.SeqLen, k.CTAGroup, k.PingPong, k.KVFormat)
	if k.KVFormat == WeightNVFP4 {
		if k.isSM103() {
			fmt.Fprintf(&b, "// PTX MMA instruction: tcgen05.mma.cta_group::%d.kind::mxf4nvf4.block_scale.scale_vec::4X with tcgen05.cp.cta_group::%d.32x128b\n",
				k.CTAGroup, k.CTAGroup)
		} else {
			fmt.Fprintf(&b, "// PTX MMA instruction: tcgen05.mma.cta_group::%d.kind::mxf4nvf4.block_scale with tcgen05.cp.cta_group::%d.32x128b\n",
				k.CTAGroup, k.CTAGroup)
		}
	} else {
		fmt.Fprintf(&b, "// PTX MMA instruction: tcgen05.mma.cta_group::%d.kind::f16\n", k.CTAGroup)
	}

	fmt.Fprintf(&b, `__global__ void %s__launch_bounds__(%d, 1) %s(
    const float* __restrict__ q,
    const float* __restrict__ k_pages,
    const float* __restrict__ v_pages,
    const int32_t* __restrict__ page_table,
    float* __restrict__ out) {
    constexpr int NUM_Q_HEADS = %d;
    constexpr int NUM_KV_HEADS = %d;
    constexpr int HEAD_DIM = %d;
    constexpr int PAGE_SIZE = %d;
    constexpr int SEQ_LEN = %d;
    constexpr int NUM_PAGES = %d;
    constexpr int CTA_GROUP = %d;
    constexpr int Q_ROWS_PER_CTA = %d;
    constexpr int STAGES = %d;
    constexpr float SCALE_LOG2E = %s;
    const float NEG_INF = -__uint_as_float(0x7f800000u);

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wg = warp >> 2; // 0: consumer WG0, 1: consumer WG1, 2: TMA producer warp
    const int rank = (CTA_GROUP == 2) ? eg_tcgen05_cluster_rank() : 0;

    EG_TCGEN05_SHARED(float, smem_q, [128 * HEAD_DIM]);
    EG_TCGEN05_SHARED(float, smem_k, [STAGES * NUM_KV_HEADS * PAGE_SIZE * HEAD_DIM]);
    EG_TCGEN05_SHARED(float, smem_v, [STAGES * NUM_KV_HEADS * PAGE_SIZE * HEAD_DIM]);
    EG_TCGEN05_SHARED(float, s_m, [128]);
    EG_TCGEN05_SHARED(float, s_l, [128]);
    EG_TCGEN05_SHARED(float, tmem, [128 * 512]);
    EG_TCGEN05_SHARED(uint32_t, slot_o, [1]);
    EG_TCGEN05_SHARED(uint32_t, slot_s, [1]);
    EG_TCGEN05_SHARED(uint32_t, slot_scale, [1]);
    EG_TCGEN05_SHARED(uint64_t, mbar, [1]);
`, clusterAttr, block[0], k.Name,
		k.NumQHeads, k.NumKVHeads, k.HeadDim, k.PageSize, k.SeqLen, k.numPages(),
		k.CTAGroup, qPerCTA, stages, floatLiteral(k.Scale*1.4426950408889634))

	if k.KVFormat == WeightNVFP4 {
		b.WriteString(`    constexpr int K_BYTES = HEAD_DIM / 2;
    constexpr int K_BLOCKS = HEAD_DIM / 16;
    constexpr int KV_ROWS = NUM_KV_HEADS * PAGE_SIZE;
    EG_TCGEN05_SHARED(uint8_t, smem_q_packed, [128 * K_BYTES]);
    EG_TCGEN05_SHARED(uint8_t, smem_q_scales, [128 * K_BLOCKS]);
    EG_TCGEN05_SHARED(uint8_t, smem_k_packed, [KV_ROWS * K_BYTES]);
    EG_TCGEN05_SHARED(uint8_t, smem_k_scales, [KV_ROWS * K_BLOCKS]);
    EG_TCGEN05_SHARED(uint8_t, smem_v_packed, [KV_ROWS * K_BYTES]);
    EG_TCGEN05_SHARED(uint8_t, smem_v_scales, [KV_ROWS * K_BLOCKS]);
`)
	}

	b.WriteString(`    // 1. Allocate TMEM slices in non-increasing column order: O (128 cols), S/P (64 cols)`)
	if k.KVFormat == WeightNVFP4 {
		b.WriteString(`, scales (32 cols)`)
	}
	b.WriteString(".\n")
	b.WriteString(`    if (warp == 0) {
        eg_tmem_alloc_group<CTA_GROUP>(tmem, &slot_o[0], 0u, 128u);
        eg_tmem_alloc_group<CTA_GROUP>(tmem, &slot_s[0], 128u, 64u);
`)
	if k.KVFormat == WeightNVFP4 {
		b.WriteString(`        eg_tmem_alloc_group<CTA_GROUP>(tmem, &slot_scale[0], 192u, 32u);
`)
	}
	b.WriteString(`    }
    __syncthreads();
    const uint32_t tmem_o = slot_o[0];
    const uint32_t tmem_s = slot_s[0];
`)
	if k.KVFormat == WeightNVFP4 {
		b.WriteString("    const uint32_t tmem_scale = slot_scale[0];\n")
	}

	// Stage Q rows and initialize online softmax state s_m, s_l.
	b.WriteString(`    for (int i = tid; i < 128; i += blockDim.x) {
        s_m[i] = NEG_INF;
        s_l[i] = 0.0f;
    }
    for (int i = tid; i < 128 * HEAD_DIM; i += blockDim.x) {
        const int r = i / HEAD_DIM, d = i % HEAD_DIM;
        const int qh = rank * Q_ROWS_PER_CTA + r;
        float qv = (r < Q_ROWS_PER_CTA && qh < NUM_Q_HEADS) ? q[qh * HEAD_DIM + d] : 0.0f;
`)
	if k.KVFormat == WeightE4M3 {
		b.WriteString("        qv = eg_e4m3_to_f32((uint8_t)(eg_f32x2_to_e4m3x2(qv, 0.0f) & 0xffu));\n")
	}
	b.WriteString(`        smem_q[i] = qv;
    }
    __syncthreads();
`)
	if k.KVFormat == WeightNVFP4 {
		b.WriteString(`    if (tid == 0) {
        for (int r = 0; r < 128; ++r) {
            for (int kb = 0; kb < K_BLOCKS; ++kb) {
                eg_fa4_quantize_nvfp4_block(&smem_q[r * HEAD_DIM + kb * 16],
                                            &smem_q_packed[r * K_BYTES + kb * 8],
                                            &smem_q_scales[r * K_BLOCKS + kb]);
            }
        }
    }
    __syncthreads();
`)
	}

	// If PingPong is enabled, prefetch page 0 via the dedicated TMA producer warp (wg == 2).
	if k.PingPong {
		b.WriteString(`    // Producer warp (wg == 2, warp == 8) prefetches KV page 0 into stage 0.
    if (wg == 2 && lane == 0) {
        const int phys0 = page_table[0];
        const int valid0 = SEQ_LEN < PAGE_SIZE ? SEQ_LEN : PAGE_SIZE;
        eg_fa4_tma_load_kv_page(smem_k, smem_v, k_pages, v_pages, phys0, valid0, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM);
    }
`)
	}
	fmt.Fprintf(&b, "    %s\n", syncCall)

	// Main multi-page loop.
	b.WriteString(`    for (int page = 0; page < NUM_PAGES; ++page) {
        const int stage = page % STAGES;
        float* const cur_k = smem_k + stage * NUM_KV_HEADS * PAGE_SIZE * HEAD_DIM;
        float* const cur_v = smem_v + stage * NUM_KV_HEADS * PAGE_SIZE * HEAD_DIM;
        const int valid_tokens = (SEQ_LEN - page * PAGE_SIZE < PAGE_SIZE) ? (SEQ_LEN - page * PAGE_SIZE) : PAGE_SIZE;
`)
	if k.PingPong {
		b.WriteString(`        // Dedicated producer warp (wg == 2) overlaps TMA prefetch of page + 1 into (page + 1) % STAGES.
        if (wg == 2 && lane == 0 && page + 1 < NUM_PAGES) {
            const int next_stage = (page + 1) % STAGES;
            const int next_phys = page_table[page + 1];
            const int next_rem = SEQ_LEN - (page + 1) * PAGE_SIZE;
            const int next_valid = next_rem < PAGE_SIZE ? next_rem : PAGE_SIZE;
            eg_fa4_tma_load_kv_page(
                smem_k + next_stage * NUM_KV_HEADS * PAGE_SIZE * HEAD_DIM,
                smem_v + next_stage * NUM_KV_HEADS * PAGE_SIZE * HEAD_DIM,
                k_pages, v_pages, next_phys, next_valid, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM);
        }
`)
	} else {
		b.WriteString(`        if (tid == 0) {
            const int phys_page = page_table[page];
            eg_fa4_tma_load_kv_page(cur_k, cur_v, k_pages, v_pages, phys_page, valid_tokens, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM);
        }
`)
	}
	fmt.Fprintf(&b, "        %s\n", syncCall)

	if k.KVFormat == WeightE4M3 {
		b.WriteString(`        for (int i = tid; i < NUM_KV_HEADS * PAGE_SIZE * HEAD_DIM; i += blockDim.x) {
            cur_k[i] = eg_e4m3_to_f32((uint8_t)(eg_f32x2_to_e4m3x2(cur_k[i], 0.0f) & 0xffu));
            cur_v[i] = eg_e4m3_to_f32((uint8_t)(eg_f32x2_to_e4m3x2(cur_v[i], 0.0f) & 0xffu));
        }
`)
		fmt.Fprintf(&b, "        %s\n", syncCall)
	} else if k.KVFormat == WeightNVFP4 {
		b.WriteString(`        if (tid == 0) {
            for (int r = 0; r < KV_ROWS; ++r) {
                for (int kb = 0; kb < K_BLOCKS; ++kb) {
                    eg_fa4_quantize_nvfp4_block(&cur_k[r * HEAD_DIM + kb * 16],
                                                &smem_k_packed[r * K_BYTES + kb * 8],
                                                &smem_k_scales[r * K_BLOCKS + kb]);
                    eg_fa4_quantize_nvfp4_block(&cur_v[r * HEAD_DIM + kb * 16],
                                                &smem_v_packed[r * K_BYTES + kb * 8],
                                                &smem_v_scales[r * K_BLOCKS + kb]);
                    const float vsf = eg_e4m3_to_f32(smem_v_scales[r * K_BLOCKS + kb]);
                    for (int b = 0; b < 8; ++b) {
                        const uint8_t vb = smem_v_packed[r * K_BYTES + kb * 8 + b];
                        cur_v[r * HEAD_DIM + kb * 16 + 2 * b] = eg_tcgen05_decode_e2m1(vb & 0xfu) * vsf;
                        cur_v[r * HEAD_DIM + kb * 16 + 2 * b + 1] = eg_tcgen05_decode_e2m1((vb >> 4) & 0xfu) * vsf;
                    }
                }
            }
        }
`)
		fmt.Fprintf(&b, "        %s\n", syncCall)
	}

	// Step A: Issue MMA S = Q * K_page^T into TMEM [tmem_s].
	b.WriteString(`        // Step A: Elected leader issues tcgen05.mma for S = Q * K_page^T into TMEM [tmem_s].
        if (rank == 0 && warp == 0 && eg_elect_one_sync()) {
`)
	if k.KVFormat == WeightNVFP4 {
		fmt.Fprintf(&b, `            eg_tmem_cp_scales_group<CTA_GROUP>(tmem, tmem_scale, smem_q_scales, smem_k_scales, 128, KV_ROWS, K_BLOCKS);
            eg_tmem_mma_s_nvfp4<CTA_GROUP, %s>(tmem, tmem_s, tmem_scale, smem_q_packed, smem_k_packed, Q_ROWS_PER_CTA, NUM_Q_HEADS, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM);
            eg_tmem_commit_group<CTA_GROUP>(&mbar[0]);
`, scaleVec4X)
	} else {
		b.WriteString(`            eg_tmem_mma_s_dense<CTA_GROUP>(tmem, tmem_s, smem_q, cur_k, Q_ROWS_PER_CTA, NUM_Q_HEADS, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, 0);
            eg_tmem_commit_group<CTA_GROUP>(&mbar[0]);
`)
	}
	b.WriteString("        }\n")
	fmt.Fprintf(&b, "        %s\n", syncCall)

	// Step B: Online softmax in TMEM.
	if k.PingPong {
		b.WriteString(`        // Step B: Ping-pong consumer warpgroups (wg 0 on even pages, wg 1 on odd pages)
        // execute 4-warp 32-row partitioned TMEM online softmax with SFU exp2.
        const int active_sfu_wg = page & 1;
        if (wg == active_sfu_wg) {
`)
	} else {
		b.WriteString(`        // Step B: Consumer warpgroup (wg == 0) executes 4-warp 32-row partitioned TMEM online softmax.
        if (wg == 0) {
`)
	}
	b.WriteString(`            const int row = 32 * (warp & 3) + lane;
            float s_row[PAGE_SIZE];
            float m_page = NEG_INF;
            for (int j = 0; j < PAGE_SIZE; ++j) {
                const float raw_s = eg_tmem_ld_cell(tmem, warp, lane, tmem_s + (uint32_t)j);
                const float scaled_s = (j < valid_tokens) ? (raw_s * SCALE_LOG2E) : NEG_INF;
                s_row[j] = scaled_s;
                m_page = fmaxf(m_page, scaled_s);
            }
            eg_tmem_wait_ld_sync();

            const float m_old = s_m[row];
            const float m_new = fmaxf(m_old, m_page);
            const float base = (m_new == NEG_INF) ? 0.0f : m_new;
            const float alpha = (m_old == NEG_INF) ? 0.0f : eg_fa4_exp2(m_old - base);

            if (page > 0) {
                for (int d = 0; d < HEAD_DIM; ++d) {
                    const float o_prev = eg_tmem_ld_cell(tmem, warp, lane, tmem_o + (uint32_t)d);
                    eg_tmem_st_cell(tmem, warp, lane, tmem_o + (uint32_t)d, o_prev * alpha);
                }
            }

            float l_page = 0.0f;
`)
	if k.isSM103() {
		b.WriteString(`            // sm_103a 2x SFU schedule: unroll exp2 in pairs across key columns.
            int j = 0;
            for (; j + 1 < PAGE_SIZE; j += 2) {
                float p0 = 0.0f, p1 = 0.0f;
                eg_fa4_exp2_2x(s_row[j] - base, s_row[j + 1] - base, &p0, &p1);
                if (j >= valid_tokens) p0 = 0.0f;
                if (j + 1 >= valid_tokens) p1 = 0.0f;
                l_page += p0 + p1;
                eg_tmem_st_cell(tmem, warp, lane, tmem_s + (uint32_t)j, p0);
                eg_tmem_st_cell(tmem, warp, lane, tmem_s + (uint32_t)(j + 1), p1);
            }
            for (; j < PAGE_SIZE; ++j) {
                const float p = (j < valid_tokens) ? eg_fa4_exp2(s_row[j] - base) : 0.0f;
                l_page += p;
                eg_tmem_st_cell(tmem, warp, lane, tmem_s + (uint32_t)j, p);
            }
`)
	} else {
		b.WriteString(`            for (int j = 0; j < PAGE_SIZE; ++j) {
                const float p = (j < valid_tokens) ? eg_fa4_exp2(s_row[j] - base) : 0.0f;
                l_page += p;
                eg_tmem_st_cell(tmem, warp, lane, tmem_s + (uint32_t)j, p);
            }
`)
	}
	b.WriteString(`            eg_tmem_wait_st_sync();
            s_m[row] = m_new;
            s_l[row] = s_l[row] * alpha + l_page;
        }
`)
	fmt.Fprintf(&b, "        %s\n", syncCall)

	// Step C: Second MMA O += P * V_page with operand A read directly from TMEM [tmem_s].
	b.WriteString(`        // Step C: Elected leader issues tcgen05.mma for O += P * V_page reading P from TMEM [tmem_s].
        if (rank == 0 && warp == 0 && eg_elect_one_sync()) {
            eg_tmem_mma_o_tmem_a<CTA_GROUP>(tmem, tmem_o, tmem_s, cur_v, Q_ROWS_PER_CTA, NUM_Q_HEADS, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM);
            eg_tmem_commit_group<CTA_GROUP>(&mbar[0]);
        }
`)
	fmt.Fprintf(&b, "        %s\n    }\n\n", syncCall)

	// Step D: Epilogue load O from TMEM [tmem_o], normalize by 1/s_l[row], and deallocate.
	b.WriteString(`    // Epilogue: 4-warp 32-row partitioned tcgen05.ld from TMEM [tmem_o] and normalize by 1 / l_i.
    if (wg == 0) {
        const int row = 32 * (warp & 3) + lane;
        const int qh = rank * Q_ROWS_PER_CTA + row;
        const float l_val = s_l[row];
        const float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
        for (int d = 0; d < HEAD_DIM; ++d) {
            const float o_val = eg_tmem_ld_cell(tmem, warp, lane, tmem_o + (uint32_t)d);
            if (row < Q_ROWS_PER_CTA && qh < NUM_Q_HEADS) {
                out[qh * HEAD_DIM + d] = o_val * inv_l;
            }
        }
        eg_tmem_wait_ld_sync();
    }
`)
	fmt.Fprintf(&b, "    %s\n", syncCall)
	b.WriteString(`    if (warp == 0) {
        eg_tmem_dealloc_group<CTA_GROUP>(tmem_o, 128u);
        eg_tmem_dealloc_group<CTA_GROUP>(tmem_s, 64u);
`)
	if k.KVFormat == WeightNVFP4 {
		b.WriteString(`        eg_tmem_dealloc_group<CTA_GROUP>(tmem_scale, 32u);
`)
	}
	b.WriteString(`    }
}
`)
	if k.CTAGroup == 2 {
		fmt.Fprintf(&b, "EG_REGISTER_CLUSTER2(%s)\n", k.Name)
	}
	return b.String()
}

// Launcher returns the C++ host launcher function for k.
func (k TCGen05PagedAttention) Launcher() string {
	k = k.norm()
	grid := k.Grid()
	block := k.Block()
	var b strings.Builder
	fmt.Fprintf(&b, `inline void %s_launch(
    const float* q,
    const float* k_pages,
    const float* v_pages,
    const int32_t* page_table,
    float* out,
    cudaStream_t stream = 0) {
    const dim3 grid(%d, %d, %d), block(%d, %d, %d);
    EG_LAUNCH(%s, grid, block, 0, stream, q, k_pages, v_pages, page_table, out);
}
`, k.Name, grid[0], grid[1], grid[2], block[0], block[1], block[2], k.Name)
	return b.String()
}

// Source returns the complete runnable/emulatable CUDA C++ translation unit
// (including device prelude, kernel, and host launcher) for k.
func (k TCGen05PagedAttention) Source() string {
	return k.CUDA() + "\n" + k.Launcher()
}

// Reference computes the exact float64 multi-page grouped-query online-softmax
// attention reference over kPages and vPages indexed by pageTable.
func (k TCGen05PagedAttention) Reference(q []float32, kPages, vPages []float32, pageTable []int32, seqLen int) []float32 {
	k = k.norm()
	if seqLen <= 0 {
		seqLen = k.SeqLen
	}
	groupSize := max(1, k.NumQHeads/k.NumKVHeads)
	numRows := k.NumQHeads
	if len(q) >= k.HeadDim && len(q)%k.HeadDim == 0 {
		numRows = len(q) / k.HeadDim
	}
	out := make([]float32, numRows*k.HeadDim)
	numPages := (seqLen + k.PageSize - 1) / k.PageSize

	for r := range numRows {
		qh := r % k.NumQHeads
		kvh := qh / groupSize
		m := math.Inf(-1)
		l := 0.0
		acc := make([]float64, k.HeadDim)

		for p := range numPages {
			physPage := int(pageTable[p])
			valid := min(k.PageSize, seqLen-p*k.PageSize)
			scores := make([]float64, valid)
			mPage := m
			for j := range valid {
				slot := physPage*k.PageSize + j
				kvBase := (slot*k.NumKVHeads + kvh) * k.HeadDim
				var dot float64
				for d := range k.HeadDim {
					dot += float64(q[r*k.HeadDim+d]) * float64(kPages[kvBase+d])
				}
				s := dot * float64(k.Scale)
				scores[j] = s
				if s > mPage {
					mPage = s
				}
			}
			alpha := 0.0
			if !math.IsInf(m, -1) {
				alpha = math.Exp(m - mPage)
			}
			l *= alpha
			for d := range k.HeadDim {
				acc[d] *= alpha
			}
			for j := range valid {
				prob := math.Exp(scores[j] - mPage)
				l += prob
				slot := physPage*k.PageSize + j
				kvBase := (slot*k.NumKVHeads + kvh) * k.HeadDim
				for d := range k.HeadDim {
					acc[d] += prob * float64(vPages[kvBase+d])
				}
			}
			m = mPage
		}
		if l > 0 {
			for d := range k.HeadDim {
				out[r*k.HeadDim+d] = float32(acc[d] / l)
			}
		}
	}
	return out
}

// TCGen05MLAPagedAttention configures a Blackwell Multi-head Latent Attention
// (MLA) kernel with matrix absorption over a paged compressed KV cache (C_KV
// and decoupled K_rope) using tcgen05 TMEM accumulators and TMEM-resident
// operand A (O_latent = P * C_KV).
type TCGen05MLAPagedAttention struct {
	Name       string
	NumHeads   int
	KVLoRARank int
	RoPEDim    int
	VHeadDim   int
	PageSize   int
	MaxPages   int
	SeqLen     int
	Scale      float32
	Target     string // "sm_100a", "sm_103a", "sm_100f", "sm_103f" (defaults to "sm_100a")
	CTAGroup   int    // 1 or 2
	PingPong   bool
}

func (k TCGen05MLAPagedAttention) norm() TCGen05MLAPagedAttention {
	if k.Name == "" || !cIdent(k.Name) {
		k.Name = "tcgen05_mla_paged_attention"
	}
	if k.NumHeads <= 0 {
		k.NumHeads = 4
	}
	if k.KVLoRARank <= 0 {
		k.KVLoRARank = 16
	}
	if k.RoPEDim <= 0 {
		k.RoPEDim = 8
	}
	if k.VHeadDim <= 0 {
		k.VHeadDim = k.KVLoRARank
	}
	if k.PageSize <= 0 {
		k.PageSize = 16
	}
	if k.SeqLen <= 0 {
		k.SeqLen = k.PageSize * 2
	}
	numPages := (k.SeqLen + k.PageSize - 1) / k.PageSize
	if k.MaxPages < numPages {
		k.MaxPages = max(numPages, 16)
	}
	if k.Scale == 0 {
		k.Scale = 1.0 / float32(math.Sqrt(float64(k.KVLoRARank+k.RoPEDim)))
	}
	switch k.Target {
	case "sm_100a", "sm_103a", "sm_100f", "sm_103f":
	default:
		k.Target = "sm_100a"
	}
	if k.CTAGroup != 2 {
		k.CTAGroup = 1
	}
	return k
}

func (k TCGen05MLAPagedAttention) isSM103() bool {
	t := k.norm().Target
	return t == "sm_103a" || t == "sm_103f"
}

func (k TCGen05MLAPagedAttention) numPages() int {
	k = k.norm()
	return max(1, (k.SeqLen+k.PageSize-1)/k.PageSize)
}

func (k TCGen05MLAPagedAttention) headsPerCTA() int {
	k = k.norm()
	if k.CTAGroup == 2 && k.NumHeads >= 2 && k.NumHeads <= 128 {
		return (k.NumHeads + 1) / 2
	}
	return min(128, k.NumHeads)
}

// SharedBytes returns the shared-memory footprint in bytes per CTA.
func (k TCGen05MLAPagedAttention) SharedBytes() int {
	k = k.norm()
	stages := 1
	if k.PingPong {
		stages = 2
	}
	qBytes := 128 * (k.KVLoRARank + k.RoPEDim) * 4
	kvBytes := stages * k.PageSize * (k.KVLoRARank + k.RoPEDim) * 4
	return qBytes + kvBytes + 1024
}

// Grid returns the launch grid dimensions for k.
func (k TCGen05MLAPagedAttention) Grid() [3]int {
	k = k.norm()
	return [3]int{k.CTAGroup, 1, 1}
}

// Block returns the CTA thread-block dimensions for k.
func (k TCGen05MLAPagedAttention) Block() [3]int {
	k = k.norm()
	if k.PingPong {
		return [3]int{288, 1, 1}
	}
	return [3]int{128, 1, 1}
}

// TMEMTrace builds the multi-page tcgen05 TMEM hardware lifecycle trace for
// absorbed MLA:
// Alloc -> { MMA S_absorbed -> MMA S_rope -> Commit -> Wait -> Load S -> Store P -> MMA O_latent -> Commit -> Wait }* -> Load O_latent -> Dealloc.
func (k TCGen05MLAPagedAttention) TMEMTrace() []cudatmem.TraceEvent {
	k = k.norm()
	g := k.CTAGroup
	pages := k.numPages()

	var events []cudatmem.TraceEvent
	for cta := range g {
		events = append(events,
			cudatmem.TraceEvent{
				Kind: cudatmem.EventAlloc, CTA: cta, Warp: 0, ActiveThreads: 32,
				CTAGroup: g, Offset: 0, Columns: 128, Rows: 128,
			},
			cudatmem.TraceEvent{
				Kind: cudatmem.EventAlloc, CTA: cta, Warp: 0, ActiveThreads: 32,
				CTAGroup: g, Offset: 128, Columns: 64, Rows: 128,
			},
		)
	}
	if g == 2 {
		events = append(events, cudatmem.TraceEvent{Kind: cudatmem.EventClusterSync, CTAGroup: 2})
	}

	for page := range pages {
		// Score S = Q_absorbed * C_KV^T + Q_rope * K_rope^T in TMEM [Offset: 128].
		events = append(events,
			cudatmem.TraceEvent{
				Kind: cudatmem.EventMMA, CTA: 0, Warp: 0, ActiveThreads: 1,
				CTAGroup: g, Offset: 128, Columns: 64, Rows: 128,
			},
			cudatmem.TraceEvent{
				Kind: cudatmem.EventMMA, CTA: 0, Warp: 0, ActiveThreads: 1,
				CTAGroup: g, Offset: 128, Columns: 64, Rows: 128,
			},
			cudatmem.TraceEvent{
				Kind: cudatmem.EventCommit, CTA: 0, Warp: 0, ActiveThreads: 1,
				CTAGroup: g, Offset: 128, Columns: 64, Rows: 128,
			},
		)
		for cta := range g {
			events = append(events, cudatmem.TraceEvent{
				Kind: cudatmem.EventWait, CTA: cta, Warp: 0, ActiveThreads: 32,
				CTAGroup: g, Offset: 128, Columns: 64, Rows: 128,
			})
		}
		for cta := range g {
			for w := range 4 {
				events = append(events, cudatmem.TraceEvent{
					Kind: cudatmem.EventLd, CTA: cta, Warp: w, ActiveThreads: 32,
					CTAGroup: g, Offset: 128, Columns: 64, RowStart: uint32(32 * w), Rows: 32,
				})
				if page > 0 {
					events = append(events,
						cudatmem.TraceEvent{
							Kind: cudatmem.EventLd, CTA: cta, Warp: w, ActiveThreads: 32,
							CTAGroup: g, Offset: 0, Columns: 128, RowStart: uint32(32 * w), Rows: 32,
						},
						cudatmem.TraceEvent{
							Kind: cudatmem.EventSt, CTA: cta, Warp: w, ActiveThreads: 32,
							CTAGroup: g, Offset: 0, Columns: 128, RowStart: uint32(32 * w), Rows: 32,
						},
					)
				}
				events = append(events, cudatmem.TraceEvent{
					Kind: cudatmem.EventSt, CTA: cta, Warp: w, ActiveThreads: 32,
					CTAGroup: g, Offset: 128, Columns: 64, RowStart: uint32(32 * w), Rows: 32,
				})
			}
		}
		// Latent output accumulation O_latent += P * C_KV in TMEM [Offset: 0] reading P from [Offset: 128].
		events = append(events,
			cudatmem.TraceEvent{
				Kind: cudatmem.EventMMA, CTA: 0, Warp: 0, ActiveThreads: 1,
				CTAGroup: g, Offset: 0, Columns: 128, Rows: 128,
				SourceAFromTMEM: true, SourceAOffset: 128,
			},
			cudatmem.TraceEvent{
				Kind: cudatmem.EventCommit, CTA: 0, Warp: 0, ActiveThreads: 1,
				CTAGroup: g, Offset: 0, Columns: 128, Rows: 128,
			},
		)
		for cta := range g {
			events = append(events, cudatmem.TraceEvent{
				Kind: cudatmem.EventWait, CTA: cta, Warp: 0, ActiveThreads: 32,
				CTAGroup: g, Offset: 0, Columns: 128, Rows: 128,
			})
		}
	}

	for cta := range g {
		for w := range 4 {
			events = append(events, cudatmem.TraceEvent{
				Kind: cudatmem.EventLd, CTA: cta, Warp: w, ActiveThreads: 32,
				CTAGroup: g, Offset: 0, Columns: 128, RowStart: uint32(32 * w), Rows: 32,
			})
		}
	}
	if g == 2 {
		events = append(events, cudatmem.TraceEvent{Kind: cudatmem.EventClusterSync, CTAGroup: 2})
	}
	for cta := range g {
		events = append(events,
			cudatmem.TraceEvent{
				Kind: cudatmem.EventDealloc, CTA: cta, Warp: 0, ActiveThreads: 32,
				CTAGroup: g, Offset: 0, Columns: 128, Rows: 128,
			},
			cudatmem.TraceEvent{
				Kind: cudatmem.EventDealloc, CTA: cta, Warp: 0, ActiveThreads: 32,
				CTAGroup: g, Offset: 128, Columns: 64, Rows: 128,
			},
		)
	}
	return events
}

// TraceEvents is an alias for [TCGen05MLAPagedAttention.TMEMTrace].
func (k TCGen05MLAPagedAttention) TraceEvents() []cudatmem.TraceEvent {
	return k.TMEMTrace()
}

// CUDA emits the complete CUDA C++ device source for absorbed MLA on Blackwell tcgen05.
func (k TCGen05MLAPagedAttention) CUDA() string {
	k = k.norm()
	block := k.Block()
	headsPerCTA := k.headsPerCTA()
	stages := 1
	if k.PingPong {
		stages = 2
	}
	clusterAttr := ""
	syncCall := "__syncthreads();"
	if k.CTAGroup == 2 {
		clusterAttr = "__cluster_dims__(2, 1, 1) "
		syncCall = "eg_tcgen05_cluster_sync();"
	}

	var b strings.Builder
	b.WriteString(Prelude)
	b.WriteString("\n")
	b.WriteString(TCGen05AdvancedPrelude)
	b.WriteString("\n")
	b.WriteString(fa4DevicePrelude)
	b.WriteString("\n")

	fmt.Fprintf(&b, "// %s: Blackwell %s absorbed Multi-head Latent Attention (H=%d, dc=%d, dr=%d, page=%d, seq=%d, cta_group::%d, pingpong=%v).\n",
		k.Name, k.Target, k.NumHeads, k.KVLoRARank, k.RoPEDim, k.PageSize, k.SeqLen, k.CTAGroup, k.PingPong)
	fmt.Fprintf(&b, `__global__ void %s__launch_bounds__(%d, 1) %s(
    const float* __restrict__ q_absorbed,
    const float* __restrict__ q_rope,
    const float* __restrict__ c_kv_pages,
    const float* __restrict__ k_rope_pages,
    const int32_t* __restrict__ page_table,
    float* __restrict__ out) {
    constexpr int NUM_HEADS = %d;
    constexpr int KV_LORA_RANK = %d;
    constexpr int ROPE_DIM = %d;
    constexpr int PAGE_SIZE = %d;
    constexpr int SEQ_LEN = %d;
    constexpr int NUM_PAGES = %d;
    constexpr int CTA_GROUP = %d;
    constexpr int HEADS_PER_CTA = %d;
    constexpr int STAGES = %d;
    constexpr float SCALE_LOG2E = %s;
    const float NEG_INF = -__uint_as_float(0x7f800000u);

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wg = warp >> 2;
    const int rank = (CTA_GROUP == 2) ? eg_tcgen05_cluster_rank() : 0;

    EG_TCGEN05_SHARED(float, smem_q_abs, [128 * KV_LORA_RANK]);
    EG_TCGEN05_SHARED(float, smem_q_rope, [128 * ROPE_DIM]);
    EG_TCGEN05_SHARED(float, smem_c_kv, [STAGES * PAGE_SIZE * KV_LORA_RANK]);
    EG_TCGEN05_SHARED(float, smem_k_rope, [STAGES * PAGE_SIZE * ROPE_DIM]);
    EG_TCGEN05_SHARED(float, s_m, [128]);
    EG_TCGEN05_SHARED(float, s_l, [128]);
    EG_TCGEN05_SHARED(float, tmem, [128 * 512]);
    EG_TCGEN05_SHARED(uint32_t, slot_o, [1]);
    EG_TCGEN05_SHARED(uint32_t, slot_s, [1]);
    EG_TCGEN05_SHARED(uint64_t, mbar, [1]);

    // 1. Allocate TMEM slices: O_latent (128 cols at offset 0) and S/P (64 cols at offset 128).
    if (warp == 0) {
        eg_tmem_alloc_group<CTA_GROUP>(tmem, &slot_o[0], 0u, 128u);
        eg_tmem_alloc_group<CTA_GROUP>(tmem, &slot_s[0], 128u, 64u);
    }
    __syncthreads();
    const uint32_t tmem_o = slot_o[0];
    const uint32_t tmem_s = slot_s[0];

    // 2. Stage absorbed query Q_absorbed [H, dc] and decoupled RoPE query Q_rope [H, dr].
    for (int i = tid; i < 128; i += blockDim.x) {
        s_m[i] = NEG_INF;
        s_l[i] = 0.0f;
    }
    for (int i = tid; i < 128 * KV_LORA_RANK; i += blockDim.x) {
        const int r = i / KV_LORA_RANK, d = i %% KV_LORA_RANK;
        const int h = rank * HEADS_PER_CTA + r;
        smem_q_abs[i] = (r < HEADS_PER_CTA && h < NUM_HEADS) ? q_absorbed[h * KV_LORA_RANK + d] : 0.0f;
    }
    for (int i = tid; i < 128 * ROPE_DIM; i += blockDim.x) {
        const int r = i / ROPE_DIM, d = i %% ROPE_DIM;
        const int h = rank * HEADS_PER_CTA + r;
        smem_q_rope[i] = (r < HEADS_PER_CTA && h < NUM_HEADS) ? q_rope[h * ROPE_DIM + d] : 0.0f;
    }
    __syncthreads();
`, clusterAttr, block[0], k.Name,
		k.NumHeads, k.KVLoRARank, k.RoPEDim, k.PageSize, k.SeqLen, k.numPages(),
		k.CTAGroup, headsPerCTA, stages, floatLiteral(k.Scale*1.4426950408889634))

	if k.PingPong {
		b.WriteString(`    if (wg == 2 && lane == 0) {
        const int phys0 = page_table[0];
        const int valid0 = SEQ_LEN < PAGE_SIZE ? SEQ_LEN : PAGE_SIZE;
        eg_fa4_tma_load_kv_page(smem_c_kv, smem_c_kv, c_kv_pages, c_kv_pages, phys0, valid0, PAGE_SIZE, 1, KV_LORA_RANK);
        eg_fa4_tma_load_kv_page(smem_k_rope, smem_k_rope, k_rope_pages, k_rope_pages, phys0, valid0, PAGE_SIZE, 1, ROPE_DIM);
    }
`)
	}
	fmt.Fprintf(&b, "    %s\n", syncCall)

	b.WriteString(`    for (int page = 0; page < NUM_PAGES; ++page) {
        const int stage = page % STAGES;
        float* const cur_c_kv = smem_c_kv + stage * PAGE_SIZE * KV_LORA_RANK;
        float* const cur_k_rope = smem_k_rope + stage * PAGE_SIZE * ROPE_DIM;
        const int valid_tokens = (SEQ_LEN - page * PAGE_SIZE < PAGE_SIZE) ? (SEQ_LEN - page * PAGE_SIZE) : PAGE_SIZE;
`)
	if k.PingPong {
		b.WriteString(`        if (wg == 2 && lane == 0 && page + 1 < NUM_PAGES) {
            const int next_stage = (page + 1) % STAGES;
            const int next_phys = page_table[page + 1];
            const int next_rem = SEQ_LEN - (page + 1) * PAGE_SIZE;
            const int next_valid = next_rem < PAGE_SIZE ? next_rem : PAGE_SIZE;
            float* const next_c = smem_c_kv + next_stage * PAGE_SIZE * KV_LORA_RANK;
            float* const next_r = smem_k_rope + next_stage * PAGE_SIZE * ROPE_DIM;
            eg_fa4_tma_load_kv_page(next_c, next_c, c_kv_pages, c_kv_pages, next_phys, next_valid, PAGE_SIZE, 1, KV_LORA_RANK);
            eg_fa4_tma_load_kv_page(next_r, next_r, k_rope_pages, k_rope_pages, next_phys, next_valid, PAGE_SIZE, 1, ROPE_DIM);
        }
`)
	} else {
		b.WriteString(`        if (tid == 0) {
            const int phys_page = page_table[page];
            eg_fa4_tma_load_kv_page(cur_c_kv, cur_c_kv, c_kv_pages, c_kv_pages, phys_page, valid_tokens, PAGE_SIZE, 1, KV_LORA_RANK);
            eg_fa4_tma_load_kv_page(cur_k_rope, cur_k_rope, k_rope_pages, k_rope_pages, phys_page, valid_tokens, PAGE_SIZE, 1, ROPE_DIM);
        }
`)
	}
	fmt.Fprintf(&b, "        %s\n", syncCall)

	b.WriteString(`        // Step A: Compute S = Q_absorbed * C_KV^T + Q_rope * K_rope^T in TMEM [tmem_s].
        if (rank == 0 && warp == 0 && eg_elect_one_sync()) {
            eg_tmem_mma_s_dense<CTA_GROUP>(tmem, tmem_s, smem_q_abs, cur_c_kv, HEADS_PER_CTA, NUM_HEADS, 1, PAGE_SIZE, KV_LORA_RANK, 0);
            eg_tmem_mma_s_dense<CTA_GROUP>(tmem, tmem_s, smem_q_rope, cur_k_rope, HEADS_PER_CTA, NUM_HEADS, 1, PAGE_SIZE, ROPE_DIM, 1);
            eg_tmem_commit_group<CTA_GROUP>(&mbar[0]);
        }
`)
	fmt.Fprintf(&b, "        %s\n", syncCall)

	if k.PingPong {
		b.WriteString(`        const int active_sfu_wg = page & 1;
        if (wg == active_sfu_wg) {
`)
	} else {
		b.WriteString(`        if (wg == 0) {
`)
	}
	b.WriteString(`            const int row = 32 * (warp & 3) + lane;
            float s_row[PAGE_SIZE];
            float m_page = NEG_INF;
            for (int j = 0; j < PAGE_SIZE; ++j) {
                const float raw_s = eg_tmem_ld_cell(tmem, warp, lane, tmem_s + (uint32_t)j);
                const float scaled_s = (j < valid_tokens) ? (raw_s * SCALE_LOG2E) : NEG_INF;
                s_row[j] = scaled_s;
                m_page = fmaxf(m_page, scaled_s);
            }
            eg_tmem_wait_ld_sync();

            const float m_old = s_m[row];
            const float m_new = fmaxf(m_old, m_page);
            const float base = (m_new == NEG_INF) ? 0.0f : m_new;
            const float alpha = (m_old == NEG_INF) ? 0.0f : eg_fa4_exp2(m_old - base);

            if (page > 0) {
                for (int d = 0; d < KV_LORA_RANK; ++d) {
                    const float o_prev = eg_tmem_ld_cell(tmem, warp, lane, tmem_o + (uint32_t)d);
                    eg_tmem_st_cell(tmem, warp, lane, tmem_o + (uint32_t)d, o_prev * alpha);
                }
            }

            float l_page = 0.0f;
`)
	if k.isSM103() {
		b.WriteString(`            // sm_103a 2x SFU schedule: dual-issue exp2 across page tokens.
            int j = 0;
            for (; j + 1 < PAGE_SIZE; j += 2) {
                float p0 = 0.0f, p1 = 0.0f;
                eg_fa4_exp2_2x(s_row[j] - base, s_row[j + 1] - base, &p0, &p1);
                if (j >= valid_tokens) p0 = 0.0f;
                if (j + 1 >= valid_tokens) p1 = 0.0f;
                l_page += p0 + p1;
                eg_tmem_st_cell(tmem, warp, lane, tmem_s + (uint32_t)j, p0);
                eg_tmem_st_cell(tmem, warp, lane, tmem_s + (uint32_t)(j + 1), p1);
            }
            for (; j < PAGE_SIZE; ++j) {
                const float p = (j < valid_tokens) ? eg_fa4_exp2(s_row[j] - base) : 0.0f;
                l_page += p;
                eg_tmem_st_cell(tmem, warp, lane, tmem_s + (uint32_t)j, p);
            }
`)
	} else {
		b.WriteString(`            for (int j = 0; j < PAGE_SIZE; ++j) {
                const float p = (j < valid_tokens) ? eg_fa4_exp2(s_row[j] - base) : 0.0f;
                l_page += p;
                eg_tmem_st_cell(tmem, warp, lane, tmem_s + (uint32_t)j, p);
            }
`)
	}
	b.WriteString(`            eg_tmem_wait_st_sync();
            s_m[row] = m_new;
            s_l[row] = s_l[row] * alpha + l_page;
        }
`)
	fmt.Fprintf(&b, "        %s\n", syncCall)

	b.WriteString(`        // Step C: Accumulate O_latent += P * C_KV in TMEM [tmem_o] with TMEM-resident operand A [tmem_s].
        if (rank == 0 && warp == 0 && eg_elect_one_sync()) {
            eg_tmem_mma_o_tmem_a<CTA_GROUP>(tmem, tmem_o, tmem_s, cur_c_kv, HEADS_PER_CTA, NUM_HEADS, 1, PAGE_SIZE, KV_LORA_RANK);
            eg_tmem_commit_group<CTA_GROUP>(&mbar[0]);
        }
`)
	fmt.Fprintf(&b, "        %s\n    }\n\n", syncCall)

	b.WriteString(`    // Epilogue: 4-warp 32-row partitioned tcgen05.ld of O_latent from TMEM [tmem_o].
    if (wg == 0) {
        const int row = 32 * (warp & 3) + lane;
        const int h = rank * HEADS_PER_CTA + row;
        const float l_val = s_l[row];
        const float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
        for (int d = 0; d < KV_LORA_RANK; ++d) {
            const float o_val = eg_tmem_ld_cell(tmem, warp, lane, tmem_o + (uint32_t)d);
            if (row < HEADS_PER_CTA && h < NUM_HEADS) {
                out[h * KV_LORA_RANK + d] = o_val * inv_l;
            }
        }
        eg_tmem_wait_ld_sync();
    }
`)
	fmt.Fprintf(&b, "    %s\n", syncCall)
	b.WriteString(`    if (warp == 0) {
        eg_tmem_dealloc_group<CTA_GROUP>(tmem_o, 128u);
        eg_tmem_dealloc_group<CTA_GROUP>(tmem_s, 64u);
    }
}
`)
	if k.CTAGroup == 2 {
		fmt.Fprintf(&b, "EG_REGISTER_CLUSTER2(%s)\n", k.Name)
	}
	return b.String()
}

// Launcher returns the C++ host launcher function for k.
func (k TCGen05MLAPagedAttention) Launcher() string {
	k = k.norm()
	grid := k.Grid()
	block := k.Block()
	var b strings.Builder
	fmt.Fprintf(&b, `inline void %s_launch(
    const float* q_absorbed,
    const float* q_rope,
    const float* c_kv_pages,
    const float* k_rope_pages,
    const int32_t* page_table,
    float* out,
    cudaStream_t stream = 0) {
    const dim3 grid(%d, %d, %d), block(%d, %d, %d);
    EG_LAUNCH(%s, grid, block, 0, stream, q_absorbed, q_rope, c_kv_pages, k_rope_pages, page_table, out);
}
`, k.Name, grid[0], grid[1], grid[2], block[0], block[1], block[2], k.Name)
	return b.String()
}

// Source returns the complete runnable/emulatable CUDA C++ translation unit
// (including device prelude, kernel, and host launcher) for k.
func (k TCGen05MLAPagedAttention) Source() string {
	return k.CUDA() + "\n" + k.Launcher()
}

// Reference computes the exact float64 multi-page absorbed Multi-head Latent
// Attention reference output O_latent [NumHeads, KVLoRARank].
func (k TCGen05MLAPagedAttention) Reference(qAbsorbed, qRope, cKVPages, kRopePages []float32, pageTable []int32, seqLen int) []float32 {
	k = k.norm()
	if seqLen <= 0 {
		seqLen = k.SeqLen
	}
	out := make([]float32, k.NumHeads*k.KVLoRARank)
	numPages := (seqLen + k.PageSize - 1) / k.PageSize

	for h := range k.NumHeads {
		m := math.Inf(-1)
		l := 0.0
		acc := make([]float64, k.KVLoRARank)

		for p := range numPages {
			physPage := int(pageTable[p])
			valid := min(k.PageSize, seqLen-p*k.PageSize)
			scores := make([]float64, valid)
			mPage := m

			for j := range valid {
				slot := physPage*k.PageSize + j
				cBase := slot * k.KVLoRARank
				rBase := slot * k.RoPEDim
				var dotAbs, dotRope float64
				for d := range k.KVLoRARank {
					dotAbs += float64(qAbsorbed[h*k.KVLoRARank+d]) * float64(cKVPages[cBase+d])
				}
				for d := range k.RoPEDim {
					dotRope += float64(qRope[h*k.RoPEDim+d]) * float64(kRopePages[rBase+d])
				}
				s := (dotAbs + dotRope) * float64(k.Scale)
				scores[j] = s
				if s > mPage {
					mPage = s
				}
			}

			alpha := 0.0
			if !math.IsInf(m, -1) {
				alpha = math.Exp(m - mPage)
			}
			l *= alpha
			for d := range k.KVLoRARank {
				acc[d] *= alpha
			}
			for j := range valid {
				prob := math.Exp(scores[j] - mPage)
				l += prob
				slot := physPage*k.PageSize + j
				cBase := slot * k.KVLoRARank
				for d := range k.KVLoRARank {
					acc[d] += prob * float64(cKVPages[cBase+d])
				}
			}
			m = mPage
		}

		if l > 0 {
			for d := range k.KVLoRARank {
				out[h*k.KVLoRARank+d] = float32(acc[d] / l)
			}
		}
	}
	return out
}
