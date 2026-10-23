#ifndef EG_PRELUDE
#define EG_PRELUDE
#if !defined(EG_HOST_EMU)
#if defined(__CUDACC_RTC__)
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef int int32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
#else
#include <stdint.h>
#endif
#endif

#define EG_DEV static __device__ __forceinline__

// EG_LAUNCH launches kernel; host launchers use it so the emulator can run them.
#if defined(EG_HOST_EMU)
#define EG_LAUNCH(kernel, grid, block, smem, stream, ...) eg_emu_launch(grid, block, smem, kernel, __VA_ARGS__)
#else
#define EG_LAUNCH(kernel, grid, block, smem, stream, ...) kernel<<<grid, block, smem, stream>>>(__VA_ARGS__)
#endif

struct alignas(16) eg_u32x4 { uint32_t x, y, z, w; };

EG_DEV float eg_bf16_to_f32(uint32_t b) { return __uint_as_float((b & 0xffffu) << 16); }
EG_DEV float eg_bf16_hi_to_f32(uint32_t b) { return __uint_as_float(b & 0xffff0000u); }

// eg_f32_to_bf16 rounds to nearest even (cvt.rn.bf16.f32).
EG_DEV uint16_t eg_f32_to_bf16(float f) {
#if defined(EG_HOST_EMU)
    return eg_emu_f32_to_bf16(f);
#else
    uint16_t r;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(f));
    return r;
#endif
}

// eg_pack_bf16x2 packs lo into bits 0-15 and hi into bits 16-31.
EG_DEV uint32_t eg_pack_bf16x2(float lo, float hi) {
#if defined(EG_HOST_EMU)
    return eg_emu_f32_to_bf16(lo) | ((uint32_t)eg_emu_f32_to_bf16(hi) << 16);
#else
    uint32_t r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
#endif
}

// eg_e4m3_to_f32 decodes float8_e4m3fn bits; 0x7f and 0xff are NaN.
EG_DEV float eg_e4m3_to_f32(uint32_t b) {
    const uint32_t e = (b >> 3) & 15u, m = b & 7u;
    float v;
    if (e == 0) v = (float)m * (1.0f / 512.0f);
    else if (e == 15 && m == 7) v = __uint_as_float(0x7fc00000u);
    else v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return (b & 0x80u) ? -v : v;
}

// eg_f32x2_to_e4m3x2 rounds to nearest even, saturating to ±448
// (cvt.rn.satfinite.e4m3x2.f32); lo lands in bits 0-7.
EG_DEV uint16_t eg_f32x2_to_e4m3x2(float lo, float hi) {
#if defined(EG_HOST_EMU)
    return eg_emu_f32_to_e4m3(lo) | ((uint16_t)eg_emu_f32_to_e4m3(hi) << 8);
#else
    uint16_t r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
    return r;
#endif
}

EG_DEV uint32_t eg_prmt(uint32_t a, uint32_t b, uint32_t sel) {
#if defined(EG_HOST_EMU)
    return eg_emu_prmt(a, b, sel);
#else
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
#endif
}

// eg_e2m1x4_to_bf16x4 decodes the four E2M1 codes in the low 16 bits of q
// (code i in bits 4i..4i+3) to bfloat16: out[0] holds codes 0 and 1, out[1]
// codes 2 and 3. Each magnitude's two bytes come from a byte-permute table
// lookup and the sign is bit 3 of the code.
EG_DEV void eg_e2m1x4_to_bf16x4(uint32_t q, uint32_t out[2]) {
    // bf16 bytes of 0, 0.5, 1, 1.5, 2, 3, 4, 6.
    const uint32_t lo_a = 0xc0800000u, lo_b = 0xc0804000u;
    const uint32_t hi_a = 0x3f3f3f00u, hi_b = 0x40404040u;
    const uint32_t idx = q & 0x7777u;
    const uint32_t lo = eg_prmt(lo_a, lo_b, idx), hi = eg_prmt(hi_a, hi_b, idx);
    out[0] = eg_prmt(lo, hi, 0x5140u) | ((q & 0x8u) << 12) | ((q & 0x80u) << 24);
    out[1] = eg_prmt(lo, hi, 0x7362u) | ((q & 0x800u) << 4) | ((q & 0x8000u) << 16);
}

EG_DEV float eg_warp_sum(float v) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

EG_DEV float eg_warp_max(float v) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m));
    return v;
}

// eg_ldg_stream loads 16 bytes that are read once (weights), bypassing L1.
EG_DEV eg_u32x4 eg_ldg_stream(const void* p) {
#if defined(EG_HOST_EMU)
    eg_u32x4 r;
    memcpy(&r, p, 16);
    return r;
#else
    eg_u32x4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
#endif
}

EG_DEV eg_u32x4 eg_ldg16(const void* p) { return *reinterpret_cast<const eg_u32x4*>(p); }

// eg_ld_cg loads through L2 only, to read another block's fenced writes.
EG_DEV float eg_ld_cg(const float* p) {
#if defined(EG_HOST_EMU)
    return *(volatile const float*)p;
#else
    float r;
    asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(r) : "l"(p));
    return r;
#endif
}

// eg_cp_async16 copies 16 bytes global to shared without staging in registers.
EG_DEV void eg_cp_async16(void* smem, const void* gmem) {
#if defined(EG_HOST_EMU)
    memcpy(smem, gmem, 16);
#else
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(gmem));
#endif
}

// eg_cp_async4 copies 4 bytes global to shared (through L1).
EG_DEV void eg_cp_async4(void* smem, const void* gmem) {
#if defined(EG_HOST_EMU)
    memcpy(smem, gmem, 4);
#else
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(s), "l"(gmem));
#endif
}

EG_DEV void eg_cp_async_commit() {
#if !defined(EG_HOST_EMU)
    asm volatile("cp.async.commit_group;");
#endif
}

// eg_cp_async_wait waits until at most n committed groups are pending.
template <int n> EG_DEV void eg_cp_async_wait() {
#if !defined(EG_HOST_EMU)
    asm volatile("cp.async.wait_group %0;" :: "n"(n));
#endif
}

EG_DEV void eg_ldmatrix_x4(uint32_t r[4], const void* row) {
#if defined(EG_HOST_EMU)
    eg_emu_ldmatrix_x4(r, row, false);
#else
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(row);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
#endif
}

EG_DEV void eg_ldmatrix_x4_trans(uint32_t r[4], const void* row) {
#if defined(EG_HOST_EMU)
    eg_emu_ldmatrix_x4(r, row, true);
#else
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(row);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
#endif
}

// eg_mma_bf16 is d = a·b + c for one m16n8k16 tile, BF16 inputs, FP32 accumulate.
EG_DEV void eg_mma_bf16(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
#if defined(EG_HOST_EMU)
    eg_emu_mma_bf16_16816(d, a, b, c);
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#endif
}

// Programmatic dependent launch: wait for the previous kernel's writes, and
// let the next kernel start its preamble.
EG_DEV void eg_pdl_wait() {
#if !defined(EG_HOST_EMU)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

EG_DEV void eg_pdl_launch_dependents() {
#if !defined(EG_HOST_EMU)
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

#if defined(EG_HOST_EMU)
#define EG_DYNAMIC_SMEM(T, name) T* name = reinterpret_cast<T*>(eg_emu_dynamic_smem())
#else
#define EG_DYNAMIC_SMEM(T, name) extern __shared__ __align__(16) unsigned char eg_smem_raw[]; T* name = reinterpret_cast<T*>(eg_smem_raw)
#endif

struct alignas(8) eg_u32x2 { uint32_t x, y; };

// eg_load converts V contiguous elements at p (aligned to their width) to FP32.
template <int V> EG_DEV void eg_load(const float* p, float* o) {
    if constexpr (V % 4 == 0) {
#pragma unroll
        for (int i = 0; i < V; i += 4) {
            const eg_u32x4 u = *reinterpret_cast<const eg_u32x4*>(p + i);
            o[i] = __uint_as_float(u.x); o[i + 1] = __uint_as_float(u.y);
            o[i + 2] = __uint_as_float(u.z); o[i + 3] = __uint_as_float(u.w);
        }
    } else {
#pragma unroll
        for (int i = 0; i < V; i += 2) {
            const eg_u32x2 u = *reinterpret_cast<const eg_u32x2*>(p + i);
            o[i] = __uint_as_float(u.x); o[i + 1] = __uint_as_float(u.y);
        }
    }
}

template <int V> EG_DEV void eg_load(const uint16_t* p, float* o) {
    uint32_t w[V / 2];
    if constexpr (V % 8 == 0) {
#pragma unroll
        for (int i = 0; i < V / 2; i += 4) {
            const eg_u32x4 u = *reinterpret_cast<const eg_u32x4*>(p + 2 * i);
            w[i] = u.x; w[i + 1] = u.y; w[i + 2] = u.z; w[i + 3] = u.w;
        }
    } else {
#pragma unroll
        for (int i = 0; i < V / 2; ++i) w[i] = reinterpret_cast<const uint32_t*>(p)[i];
    }
#pragma unroll
    for (int i = 0; i < V / 2; ++i) { o[2 * i] = eg_bf16_to_f32(w[i]); o[2 * i + 1] = eg_bf16_hi_to_f32(w[i]); }
}

template <int V> EG_DEV void eg_load(const uint8_t* p, float* o) {
#pragma unroll
    for (int i = 0; i < V; i += 2) {
        const uint32_t u = reinterpret_cast<const uint16_t*>(p)[i / 2];
        o[i] = eg_e4m3_to_f32(u & 0xffu); o[i + 1] = eg_e4m3_to_f32(u >> 8);
    }
}

// eg_store rounds V FP32 values to the element type at p.
template <int V> EG_DEV void eg_store(float* p, const float* v) {
    if constexpr (V % 4 == 0) {
#pragma unroll
        for (int i = 0; i < V; i += 4)
            *reinterpret_cast<eg_u32x4*>(p + i) = eg_u32x4{__float_as_uint(v[i]), __float_as_uint(v[i + 1]), __float_as_uint(v[i + 2]), __float_as_uint(v[i + 3])};
    } else {
#pragma unroll
        for (int i = 0; i < V; i += 2) *reinterpret_cast<eg_u32x2*>(p + i) = eg_u32x2{__float_as_uint(v[i]), __float_as_uint(v[i + 1])};
    }
}

template <int V> EG_DEV void eg_store(uint16_t* p, const float* v) {
    if constexpr (V % 8 == 0) {
#pragma unroll
        for (int i = 0; i < V; i += 8)
            *reinterpret_cast<eg_u32x4*>(p + i) = eg_u32x4{eg_pack_bf16x2(v[i], v[i + 1]), eg_pack_bf16x2(v[i + 2], v[i + 3]),
                                                           eg_pack_bf16x2(v[i + 4], v[i + 5]), eg_pack_bf16x2(v[i + 6], v[i + 7])};
    } else {
#pragma unroll
        for (int i = 0; i < V; i += 2) reinterpret_cast<uint32_t*>(p)[i / 2] = eg_pack_bf16x2(v[i], v[i + 1]);
    }
}

template <int V> EG_DEV void eg_store(uint8_t* p, const float* v) {
#pragma unroll
    for (int i = 0; i < V; i += 2) reinterpret_cast<uint16_t*>(p)[i / 2] = eg_f32x2_to_e4m3x2(v[i], v[i + 1]);
}

// eg_block_combine combines one warp-reduced value per warp (OP 0: sum,
// 1: max) in warp order; every thread gets the result. red holds 32 floats.
template <int OP> EG_DEV float eg_block_combine(float v, float* red) {
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float s = red[0];
    for (int i = 1; i < (int)(blockDim.x >> 5); ++i) s = OP ? fmaxf(s, red[i]) : s + red[i];
    __syncthreads();
    return s;
}

EG_DEV float eg_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
EG_DEV float eg_silu(float x) { return x / (1.0f + expf(-x)); }
#endif // EG_PRELUDE

// moe8_attn_sliding: paged FP8 GQA attention, 16/2 heads, 1 query tokens, window 2048, page 1, 256-key splits.
__global__ void __launch_bounds__(128) moe8_attn_sliding(
    const uint8_t* __restrict__ q,
    const uint8_t* __restrict__ k_cache,
    const uint8_t* __restrict__ v_cache,
    const int32_t* __restrict__ page_table,
    int max_pages,
    const int32_t* __restrict__ seqlens,
    const float* __restrict__ k_scale,
    const float* __restrict__ v_scale,
    float* __restrict__ out,
    unsigned char* __restrict__ ws) {
    constexpr int HQ = 16, HKV = 2, G = 8, L = 1, ROWS = 8, MROWS = 16, MT = 1, KW = 4;
    constexpr int WINDOW = 2048, PAGE = 1, CHUNK = 256, LD = 72, PIECES = KW * 64, NP = (PIECES + 127) / 128;
    const float scale = 0.6841258f * k_scale[0] * 1.4426950408889634f; // log2 domain
    const int split = blockIdx.x, kvh = blockIdx.y, b = blockIdx.z;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int mt = warp % MT, kw = warp / MT;
    const int seqlen = seqlens[b], p0 = seqlen - L;
    const int kbegin = WINDOW ? max(0, p0 - WINDOW + 1) : 0;
    const int nsplit = (seqlen - kbegin + CHUNK - 1) / CHUNK;
    if (split >= nsplit) return;
    const int c0 = kbegin + split * CHUNK, c1 = min(seqlen, c0 + CHUNK);
    const float NEG_INF = -__uint_as_float(0x7f800000u);

    alignas(16) __shared__ uint16_t sq[MROWS * LD];
    alignas(16) __shared__ uint16_t sk[KW * 16 * LD];
    alignas(16) __shared__ uint16_t sv[KW * 16 * LD];
    __shared__ float red_m[KW * MROWS], red_l[KW * MROWS];
    __shared__ float red_o[KW * MROWS * 64];
    __shared__ int last;

    // Query rows r = j*G + g (token j, head kvh*G + g) as BF16, padded with zeros.
    for (int i = tid; i < MROWS * 8; i += 128) {
        const int r = i / 8, d = (i % 8) * 8;
        eg_u32x4 w = {0, 0, 0, 0};
        if (r < ROWS) {
            float f[8];
            eg_load<8>(q + ((size_t)(b * L + r / G) * HQ + kvh * G + r % G) * 64 + d, f);
            w = eg_u32x4{eg_pack_bf16x2(f[0], f[1]), eg_pack_bf16x2(f[2], f[3]),
                         eg_pack_bf16x2(f[4], f[5]), eg_pack_bf16x2(f[6], f[7])};
        }
        *reinterpret_cast<eg_u32x4*>(&sq[r * LD + d]) = w;
    }

    // Each thread moves NP 16-byte pieces (16 keys x 4 pieces per KW group).
    eg_u32x4 kreg[NP], vreg[NP];
    auto fetch = [&](int kb) {
#pragma unroll
        for (int n = 0; n < NP; ++n) {
            const int piece = tid + 128 * n, kp = kb + piece / 4;
            kreg[n] = vreg[n] = eg_u32x4{0, 0, 0, 0};
            if (piece < PIECES && kp < c1) {
                const size_t row = (size_t)page_table[(size_t)b * max_pages + kp / PAGE] * PAGE + kp % PAGE;
                const size_t at = (row * HKV + kvh) * 64 + (piece % 4) * 16;
                kreg[n] = eg_ldg16(k_cache + at);
                vreg[n] = eg_ldg16(v_cache + at);
            }
        }
    };
    auto widen = [&](uint16_t* dst, eg_u32x4 u) {
        const uint32_t w[4] = {u.x, u.y, u.z, u.w};
        uint32_t o[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            o[2 * i] = eg_pack_bf16x2(eg_e4m3_to_f32(w[i] & 0xffu), eg_e4m3_to_f32((w[i] >> 8) & 0xffu));
            o[2 * i + 1] = eg_pack_bf16x2(eg_e4m3_to_f32((w[i] >> 16) & 0xffu), eg_e4m3_to_f32(w[i] >> 24));
        }
        reinterpret_cast<eg_u32x4*>(dst)[0] = eg_u32x4{o[0], o[1], o[2], o[3]};
        reinterpret_cast<eg_u32x4*>(dst)[1] = eg_u32x4{o[4], o[5], o[6], o[7]};
    };
    fetch(c0);
    __syncthreads();

    uint32_t qa[4][4];
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) eg_ldmatrix_x4(qa[ks], &sq[(mt * 16 + lane % 16) * LD + ks * 16 + (lane / 16) * 8]);

    // This lane's two rows (g and g+8 of its tile): positions and first keys.
    int pos[2], lo[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int r = mt * 16 + lane / 4 + 8 * h;
        pos[h] = p0 + min(r / G, L - 1);
        lo[h] = WINDOW ? max(0, pos[h] - WINDOW + 1) : 0;
    }
    float m[2] = {NEG_INF, NEG_INF}, l[2] = {0.0f, 0.0f}, o[8][4] = {};

    for (int kb = c0; kb < c1; kb += 16 * KW) {
        __syncthreads();
#pragma unroll
        for (int n = 0; n < NP; ++n) {
            const int piece = tid + 128 * n;
            if (piece < PIECES) {
                widen(&sk[(piece / 4) * LD + (piece % 4) * 16], kreg[n]);
                widen(&sv[(piece / 4) * LD + (piece % 4) * 16], vreg[n]);
            }
        }
        __syncthreads();
        if (kb + 16 * KW < c1) fetch(kb + 16 * KW);
        const int key0 = kb + kw * 16;
        if (key0 >= c1) continue;
        float s[2][4] = {};
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            uint32_t kf[4];
            eg_ldmatrix_x4(kf, &sk[(kw * 16 + lane % 8 + 8 * (lane / 16)) * LD + ks * 16 + 8 * ((lane / 8) % 2)]);
            eg_mma_bf16(s[0], qa[ks], &kf[0], s[0]);
            eg_mma_bf16(s[1], qa[ks], &kf[2], s[1]);
        }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float mx = NEG_INF;
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int key = key0 + nt * 8 + 2 * (lane % 4) + e;
                    float& x = s[nt][2 * h + e];
                    x = key < c1 && key >= lo[h] && key <= pos[h] ? x * scale : NEG_INF;
                    mx = fmaxf(mx, x);
                }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            const float mnew = fmaxf(m[h], mx), base = mnew == NEG_INF ? 0.0f : mnew;
            const float corr = exp2f(m[h] - base);
            m[h] = mnew;
            l[h] *= corr;
#pragma unroll
            for (int nd = 0; nd < 8; ++nd) { o[nd][2 * h] *= corr; o[nd][2 * h + 1] *= corr; }
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    float& x = s[nt][2 * h + e];
                    x = exp2f(x - base);
                    l[h] += x;
                }
        }
        const uint32_t pa[4] = {eg_pack_bf16x2(s[0][0], s[0][1]), eg_pack_bf16x2(s[0][2], s[0][3]),
                                eg_pack_bf16x2(s[1][0], s[1][1]), eg_pack_bf16x2(s[1][2], s[1][3])};
#pragma unroll
        for (int nd = 0; nd < 4; ++nd) {
            uint32_t vf[4];
            eg_ldmatrix_x4_trans(vf, &sv[(kw * 16 + lane % 8 + 8 * ((lane / 8) % 2)) * LD + nd * 16 + 8 * (lane / 16)]);
            eg_mma_bf16(o[2 * nd], pa, &vf[0], o[2 * nd]);
            eg_mma_bf16(o[2 * nd + 1], pa, &vf[2], o[2 * nd + 1]);
        }
    }

    // Combine the KW key groups of each row tile in shared memory.
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        l[h] += __shfl_xor_sync(0xffffffffu, l[h], 1);
        l[h] += __shfl_xor_sync(0xffffffffu, l[h], 2);
        const int r = mt * 16 + lane / 4 + 8 * h;
        if (lane % 4 == 0) { red_m[kw * MROWS + r] = m[h]; red_l[kw * MROWS + r] = l[h]; }
#pragma unroll
        for (int nd = 0; nd < 8; ++nd)
#pragma unroll
            for (int e = 0; e < 2; ++e)
                red_o[(kw * MROWS + r) * 64 + nd * 8 + 2 * (lane % 4) + e] = o[nd][2 * h + e];
    }
    __syncthreads();
    const float vs = v_scale[0];
    unsigned char* const counters = ws;
    float* const parts = reinterpret_cast<float*>(ws + 4 * (size_t)gridDim.z * HKV);
    float* const mine = parts + (((size_t)b * HKV + kvh) * gridDim.x + split) * ROWS * 66;
    for (int i = tid; i < ROWS * 64 && nsplit > 1; i += 128) {
        const int r = i / 64, d = i % 64;
        float mx = NEG_INF;
#pragma unroll
        for (int k = 0; k < KW; ++k) mx = fmaxf(mx, red_m[k * MROWS + r]);
        const float base = mx == NEG_INF ? 0.0f : mx;
        float sum = 0.0f, acc = 0.0f;
#pragma unroll
        for (int k = 0; k < KW; ++k) {
            const float w = exp2f(red_m[k * MROWS + r] - base);
            sum += w * red_l[k * MROWS + r];
            acc += w * red_o[(k * MROWS + r) * 64 + d];
        }
        mine[r * 66 + 2 + d] = acc;
        if (d == 0) { mine[r * 66] = mx; mine[r * 66 + 1] = sum; }
    }
    auto put = [&](size_t i, float v) { out[i] = v; };
    // The finished rows go to fin (the first KW slot of red_o, which each
    // thread reads only at the index it then writes).
    float* const fin = red_o;
    if (nsplit == 1) {
        for (int i = tid; i < ROWS * 64; i += 128) {
            const int r = i / 64, d = i % 64;
            float mx = NEG_INF;
#pragma unroll
            for (int k = 0; k < KW; ++k) mx = fmaxf(mx, red_m[k * MROWS + r]);
            float sum = 0.0f, acc = 0.0f;
#pragma unroll
            for (int k = 0; k < KW; ++k) {
                const float w = exp2f(red_m[k * MROWS + r] - mx);
                sum += w * red_l[k * MROWS + r];
                acc += w * red_o[(k * MROWS + r) * 64 + d];
            }
            fin[r * 64 + d] = acc / sum * vs;
        }
    } else {
        __threadfence();
        __syncthreads();
        int* const counter = reinterpret_cast<int*>(counters) + b * HKV + kvh;
        if (tid == 0) last = atomicAdd(counter, 1) == nsplit - 1;
        __syncthreads();
        if (!last) return;
        __threadfence();
        const float* const all = parts + ((size_t)b * HKV + kvh) * gridDim.x * ROWS * 66;
        for (int i = tid; i < ROWS * 64; i += 128) {
            const int r = i / 64, d = i % 64;
            float mx = NEG_INF;
            for (int s = 0; s < nsplit; ++s) mx = fmaxf(mx, eg_ld_cg(all + (size_t)s * ROWS * 66 + r * 66));
            float sum = 0.0f, acc = 0.0f;
            for (int s = 0; s < nsplit; ++s) {
                const float* p = all + (size_t)s * ROWS * 66 + r * 66;
                const float w = exp2f(eg_ld_cg(p) - mx);
                sum += w * eg_ld_cg(p + 1);
                acc += w * eg_ld_cg(p + 2 + d);
            }
            fin[r * 64 + d] = acc / sum * vs;
        }
        if (tid == 0) *counter = 0;
    }
    __syncthreads();
    for (int i = tid; i < ROWS * 64; i += 128) {
        const int r = i / 64, d = i % 64;
        put(((size_t)(b * L + r / G) * HQ + kvh * G + r % G) * 64 + d, fin[i]);
    }
}

inline void moe8_attn_sliding_launch(int batch, int max_keys, const uint8_t* __restrict__ q, const uint8_t* __restrict__ k_cache, const uint8_t* __restrict__ v_cache, const int32_t* __restrict__ page_table, int max_pages, const int32_t* __restrict__ seqlens, const float* __restrict__ k_scale, const float* __restrict__ v_scale, float* __restrict__ out, unsigned char* __restrict__ ws, cudaStream_t stream) {
    const dim3 grid((max_keys + 255) / 256, 2, batch), block(128);
    if (grid.x > 0 && batch > 0) EG_LAUNCH(moe8_attn_sliding, grid, block, 0, stream, q, k_cache, v_cache, page_table, max_pages, seqlens, k_scale, v_scale, out, ws);
}
