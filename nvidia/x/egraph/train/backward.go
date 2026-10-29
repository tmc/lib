package train

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// A DType is the element type of a tensor in memory.
// Values in registers are always FP32; a DType says how they are stored.
type DType uint8

const (
	F32  DType = iota + 1
	BF16       // bfloat16, round to nearest even
	E4M3       // float8_e4m3fn, saturating to ±448
	E2M1       // float4_e2m1fn, two codes per byte, element 2i in the low nibble
	U8
	I32
	I64
	// F16 is IEEE binary16 (half precision), rounded to nearest even.
	F16

	// FP16 and FP32 are aliases for [F16] and [F32].
	FP16 = F16
	FP32 = F32
)

var dtypeNames = [...]string{
	F32:  "f32",
	BF16: "bf16",
	E4M3: "e4m3",
	E2M1: "e2m1",
	U8:   "u8",
	I32:  "i32",
	I64:  "i64",
	F16:  "f16",
}

func (d DType) String() string {
	if int(d) < len(dtypeNames) && dtypeNames[d] != "" {
		return dtypeNames[d]
	}
	return fmt.Sprintf("DType(%d)", uint8(d))
}

// BF16Bits returns the bfloat16 encoding of f, rounded to nearest even.
func BF16Bits(f float32) uint16 {
	u := math.Float32bits(f)
	if u&0x7fffffff > 0x7f800000 {
		return 0x7fc0 | uint16(u>>16)&0x8000
	}
	u += 0x7fff + (u>>16)&1
	return uint16(u >> 16)
}

// FromBF16Bits decodes a bfloat16 encoding.
func FromBF16Bits(b uint16) float32 { return math.Float32frombits(uint32(b) << 16) }

// F16Bits rounds f to IEEE binary16 bits, nearest even.
func F16Bits(f float32) uint16 {
	u := math.Float32bits(f)
	s, a := uint16(u>>16)&0x8000, u&0x7fffffff
	switch {
	case a > 0x7f800000:
		return s | 0x7e00
	case a >= 0x477ff000:
		return s | 0x7c00
	case a < 0x38800000:
		return s | uint16(math.RoundToEven(float64(math.Float32frombits(a))*(1<<24)))
	}
	r := ((a>>23)-112)<<10 | (a>>13)&0x3ff
	if rem := a & 0x1fff; rem > 0x1000 || rem == 0x1000 && r&1 == 1 {
		r++
	}
	return s | uint16(r)
}

// FromF16Bits decodes an IEEE binary16 encoding.
func FromF16Bits(h uint16) float32 {
	s := uint32(h&0x8000) << 16
	e := uint32(h>>10) & 31
	m := uint32(h & 0x3ff)
	if e == 0 {
		v := float32(m) * (1.0 / 16777216.0)
		if s != 0 {
			return -v
		}
		return v
	}
	if e == 31 {
		return math.Float32frombits(s | 0x7f800000 | (m << 13))
	}
	return math.Float32frombits(s | ((e + 112) << 23) | (m << 13))
}

// Prelude is the device prelude every emitted kernel starts with.
const Prelude = `#ifndef EG_PRELUDE
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

// eg_pack_f16x2 rounds lo and hi to F16 (nearest even) into bits 0-15 and 16-31.
EG_DEV uint32_t eg_pack_f16x2(float lo, float hi) {
#if defined(EG_HOST_EMU)
    return eg_emu_f32_to_f16(lo) | ((uint32_t)eg_emu_f32_to_f16(hi) << 16);
#else
    uint32_t r;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
#endif
}

// eg_e4m3x2_to_f16x2 widens the two E4M3 bytes in bits 0-15 of x to F16
// (exactly; cvt.rn.f16x2.e4m3x2, sm_89+), the low byte into bits 0-15.
EG_DEV uint32_t eg_e4m3x2_to_f16x2(uint32_t x) {
#if defined(EG_HOST_EMU)
    return eg_emu_f32_to_f16(eg_emu_e4m3(x & 0xffu)) | ((uint32_t)eg_emu_f32_to_f16(eg_emu_e4m3((x >> 8) & 0xffu)) << 16);
#else
    uint32_t r;
    asm("{\n.reg .b16 lo;\ncvt.u16.u32 lo, %1;\ncvt.rn.f16x2.e4m3x2 %0, lo;\n}" : "=r"(r) : "r"(x));
    return r;
#endif
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
    return __ldcg(p);
#endif
}

// eg_ld_cg2 loads two floats (8-byte aligned) through L2 only.
EG_DEV float2 eg_ld_cg2(const float* p) {
#if defined(EG_HOST_EMU)
    const volatile float* q = p;
    return float2{q[0], q[1]};
#else
    return __ldcg(reinterpret_cast<const float2*>(p));
#endif
}

// eg_cp_async16 copies 16 bytes global to shared without staging in registers.
EG_DEV void eg_cp_async16(void* smem, const void* gmem) {
#if defined(EG_HOST_EMU)
    eg_emu_cp_async(smem, gmem, 16);
#else
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(gmem));
#endif
}

// eg_cp_async4 copies 4 bytes global to shared (through L1).
EG_DEV void eg_cp_async4(void* smem, const void* gmem) {
#if defined(EG_HOST_EMU)
    eg_emu_cp_async(smem, gmem, 4);
#else
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(s), "l"(gmem));
#endif
}

EG_DEV void eg_cp_async_commit() {
#if defined(EG_HOST_EMU)
    eg_emu_cp_async_commit();
#else
    asm volatile("cp.async.commit_group;");
#endif
}

// eg_cp_async_wait waits until at most n committed groups are pending.
template <int n> EG_DEV void eg_cp_async_wait() {
#if defined(EG_HOST_EMU)
    eg_emu_cp_async_wait(n);
#else
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

// eg_mma_f16 is d = a·b + c for one m16n8k16 tile, F16 inputs, FP32 accumulate.
EG_DEV void eg_mma_f16(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
#if defined(EG_HOST_EMU)
    eg_emu_mma_f16_16816(d, a, b, c);
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#endif
}

// eg_wgmma_desc returns a wgmma shared-memory matrix descriptor (no
// swizzle) for the matrix at p with leading and stride byte offsets.
EG_DEV uint64_t eg_wgmma_desc(const void* p, uint32_t lbo, uint32_t sbo) {
#if defined(EG_HOST_EMU)
    const uint64_t a = (uint64_t)((const unsigned char*)p - eg_emu_dynamic_smem());
#else
    const uint64_t a = (uint32_t)__cvta_generic_to_shared(p);
#endif
    return ((a & 0x3ffffu) >> 4) | (uint64_t)((lbo & 0x3ffffu) >> 4) << 16 | (uint64_t)((sbo & 0x3ffffu) >> 4) << 32;
}

// eg_fence_proxy_async orders this thread's generic-proxy shared memory
// writes (st.shared, cp.async) before later async-proxy reads (wgmma).
EG_DEV void eg_fence_proxy_async() {
#if !defined(EG_HOST_EMU)
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
}

EG_DEV void eg_wgmma_fence() {
#if defined(EG_HOST_EMU)
    eg_emu_wgmma_fence();
#else
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#endif
}

EG_DEV void eg_wgmma_commit() {
#if defined(EG_HOST_EMU)
    eg_emu_wgmma_commit();
#else
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
#endif
}

// eg_wgmma_wait waits until at most n wgmma groups are pending.
template <int n> EG_DEV void eg_wgmma_wait() {
#if defined(EG_HOST_EMU)
    eg_emu_wgmma_wait(n);
#else
    asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(n) : "memory");
#endif
}

// eg_mma_e4m3 is d = a·b + c for one m16n8k32 tile, E4M3 inputs (four per
// register, k ascending from the low byte), FP32 accumulate (sm_89+).
EG_DEV void eg_mma_e4m3(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
#if defined(EG_HOST_EMU)
    eg_emu_mma_e4m3_16832(d, a, b, c);
#else
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
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
`

// backwardPrelude extends [Prelude] with type-dispatched FP32 conversions
// (eg_to_f32 and eg_from_f32) over __nv_bfloat16, __half, and float.
const backwardPrelude = `#ifndef EG_BACKWARD_PRELUDE
#define EG_BACKWARD_PRELUDE

#if defined(EG_HOST_EMU)
struct __half { uint16_t b; };
#else
#if !defined(__CUDA_BF16_TYPES_EXIST__)
struct __nv_bfloat16 { uint16_t b; };
#endif
#if !defined(__CUDA_FP16_TYPES_EXIST__)
struct __half { uint16_t b; };
#endif
#endif

EG_DEV float eg_f16_to_f32(uint32_t h) {
    const uint32_t s = (h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 31u, m = h & 0x3ffu;
    if (e == 0) {
        const float v = (float)m * (1.0f / 16777216.0f);
        return s ? -v : v;
    }
    if (e == 31) return __uint_as_float(s | 0x7f800000u | (m << 13));
    return __uint_as_float(s | ((e + 112u) << 23) | (m << 13));
}

EG_DEV uint16_t eg_f32_to_f16(float f) {
    return (uint16_t)eg_pack_f16x2(f, 0.0f);
}

EG_DEV float eg_to_f32(float v) { return v; }
EG_DEV float eg_to_f32(uint16_t v) { return eg_bf16_to_f32(v); }
EG_DEV float eg_to_f32(__nv_bfloat16 v) {
    uint16_t u;
    __builtin_memcpy(&u, &v, 2);
    return eg_bf16_to_f32(u);
}
EG_DEV float eg_to_f32(__half v) {
    uint16_t u;
    __builtin_memcpy(&u, &v, 2);
    return eg_f16_to_f32(u);
}

template <typename T> EG_DEV T eg_from_f32(float v);
template <> EG_DEV float eg_from_f32<float>(float v) { return v; }
template <> EG_DEV uint16_t eg_from_f32<uint16_t>(float v) { return eg_f32_to_bf16(v); }
template <> EG_DEV __nv_bfloat16 eg_from_f32<__nv_bfloat16>(float v) {
    const uint16_t u = eg_f32_to_bf16(v);
    __nv_bfloat16 r;
    __builtin_memcpy(&r, &u, 2);
    return r;
}
template <> EG_DEV __half eg_from_f32<__half>(float v) {
    const uint16_t u = eg_f32_to_f16(v);
    __half r;
    __builtin_memcpy(&r, &u, 2);
    return r;
}

#endif // EG_BACKWARD_PRELUDE
`

func cIdent(s string) bool {
	if s == "" {
		return false
	}
	for i, c := range s {
		if !(c == '_' || 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z' || i > 0 && '0' <= c && c <= '9') {
			return false
		}
	}
	return true
}

func floatLiteral(v float32) string {
	s := strconv.FormatFloat(float64(v), 'g', -1, 32)
	if !strings.ContainsAny(s, ".e") {
		s += ".0"
	}
	return s + "f"
}

func launcher(name string, params []string, grid func(*strings.Builder)) string {
	names := make([]string, len(params))
	for i, p := range params {
		f := strings.Fields(p)
		names[i] = f[len(f)-1]
	}
	var b strings.Builder
	fmt.Fprintf(&b, "inline void %s_launch(%s, cudaStream_t stream) {\n", name, strings.Join(params, ", "))
	grid(&b)
	fmt.Fprintf(&b, "    if (grid.x > 0) EG_LAUNCH(%s, grid, block, 0, stream, %s);\n}\n", name, strings.Join(names, ", "))
	return b.String()
}

func bwdCType(d DType) (string, error) {
	switch d {
	case BF16:
		return "__nv_bfloat16", nil
	case FP16:
		return "__half", nil
	case FP32:
		return "float", nil
	default:
		return "", fmt.Errorf("unsupported dtype %v (want BF16, FP16, or FP32)", d)
	}
}

func normThreads(threads int) (int, error) {
	if threads == 0 {
		return 256, nil
	}
	if threads < 32 || threads > 1024 || threads%32 != 0 {
		return 0, fmt.Errorf("threads %d must be a multiple of 32 in 32..1024", threads)
	}
	return threads, nil
}

// RMSNormBackward emits a fused CUDA C++ backward kernel for root-mean-square
// layer normalization:
//
//	y[r, i] = x[r, i] * rsqrt(mean_j(x[r, j]^2) + Eps) * w[i]
//
// One CTA per row r = blockIdx.x computes dX[r, i] and atomically accumulates
// the weight gradient into dW_accum[i] in FP32.
type RMSNormBackward struct {
	Name    string
	Cols    int
	Eps     float32
	DType   DType // BF16, FP16, FP32
	Threads int
}

func (k RMSNormBackward) norm() (RMSNormBackward, string, error) {
	if k.Name == "" {
		k.Name = "rmsnorm_bwd"
	}
	if !cIdent(k.Name) {
		return k, "", fmt.Errorf("rmsnorm backward: bad name %q", k.Name)
	}
	if k.Cols < 1 {
		return k, "", fmt.Errorf("rmsnorm backward %s: cols %d must be positive", k.Name, k.Cols)
	}
	if k.Eps < 0 {
		return k, "", fmt.Errorf("rmsnorm backward %s: negative eps %g", k.Name, k.Eps)
	}
	if k.Eps == 0 {
		k.Eps = 1e-5
	}
	if k.DType == 0 {
		k.DType = BF16
	}
	ct, err := bwdCType(k.DType)
	if err != nil {
		return k, "", fmt.Errorf("rmsnorm backward %s: %w", k.Name, err)
	}
	k.Threads, err = normThreads(k.Threads)
	if err != nil {
		return k, "", fmt.Errorf("rmsnorm backward %s: %w", k.Name, err)
	}
	return k, ct, nil
}

func (k RMSNormBackward) params(ct string) []string {
	return []string{
		ct + "* __restrict__ dX",
		"float* __restrict__ dW_accum",
		"const " + ct + "* __restrict__ dY",
		"const " + ct + "* __restrict__ X",
		"const " + ct + "* __restrict__ W",
		"int rows",
	}
}

// Grid returns the launch grid and block for rows rows.
func (k RMSNormBackward) Grid(rows int) (grid, block [3]int) {
	threads := k.Threads
	if threads == 0 {
		threads = 256
	}
	return [3]int{rows, 1, 1}, [3]int{threads, 1, 1}
}

// CUDA returns the kernel source (with [backwardPrelude], without [Prelude]).
func (k RMSNormBackward) CUDA() (string, error) {
	k, ct, err := k.norm()
	if err != nil {
		return "", err
	}
	var b strings.Builder
	b.WriteString(backwardPrelude)
	b.WriteString("\n")
	fmt.Fprintf(&b, "// %s: fused RMSNorm backward over rows of %d values.\n", k.Name, k.Cols)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(%d) %s(\n    %s) {\n",
		k.Threads, k.Name, strings.Join(k.params(ct), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int COLS = %d, THREADS = %d;\n", k.Cols, k.Threads)
	fmt.Fprintf(&b, "    constexpr float EPS = %s, INV_COLS = 1.0f / (float)COLS;\n", floatLiteral(k.Eps))
	fmt.Fprintf(&b, "    using T = %s;\n", ct)
	b.WriteString(`    const int r = blockIdx.x;
    if (r >= rows) return;
    __shared__ float eg_red[32];
    const T* x_row = X + (size_t)r * COLS;
    const T* dy_row = dY + (size_t)r * COLS;
    T* dx_row = dX + (size_t)r * COLS;
    float ss = 0.0f, dot = 0.0f;
    for (int i = threadIdx.x; i < COLS; i += THREADS) {
        const float x = eg_to_f32(x_row[i]);
        const float dy = eg_to_f32(dy_row[i]);
        const float w = eg_to_f32(W[i]);
        ss += x * x;
        dot += dy * w * x;
    }
    ss = eg_block_combine<0>(eg_warp_sum(ss), eg_red);
    dot = eg_block_combine<0>(eg_warp_sum(dot), eg_red);
    const float inv_rms = rsqrtf(ss * INV_COLS + EPS);
    const float c = (dot * INV_COLS) * (inv_rms * inv_rms * inv_rms);
    for (int i = threadIdx.x; i < COLS; i += THREADS) {
        const float x = eg_to_f32(x_row[i]);
        const float dy = eg_to_f32(dy_row[i]);
        const float w = eg_to_f32(W[i]);
        dx_row[i] = eg_from_f32<T>(inv_rms * (dy * w) - x * c);
        atomicAdd(&dW_accum[i], dy * x * inv_rms);
    }
}
`)
	return b.String(), nil
}

// Launcher returns void Name_launch(<params...>, cudaStream_t stream).
func (k RMSNormBackward) Launcher() (string, error) {
	k, ct, err := k.norm()
	if err != nil {
		return "", err
	}
	return launcher(k.Name, k.params(ct), func(b *strings.Builder) {
		fmt.Fprintf(b, "    const dim3 grid(rows > 0 ? rows : 0), block(%d);\n", k.Threads)
	}), nil
}

// Emit returns the complete CUDA C++ translation unit ([Prelude], kernel, and launcher).
func (k RMSNormBackward) Emit() (string, error) {
	src, err := k.CUDA()
	if err != nil {
		return "", err
	}
	l, err := k.Launcher()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src + "\n" + l, nil
}

// SwiGLUBackward emits a fused CUDA C++ backward kernel for SwiGLU:
//
//	y[r, i] = silu(g[r, i]) * u[r, i],  silu(g) = g * sigmoid(g)
//
// where GateUp has 2*Cols columns per row (g in 0..Cols-1, u in Cols..2*Cols-1).
type SwiGLUBackward struct {
	Name    string
	Cols    int // output hidden cols (GateUp has 2*Cols)
	DType   DType
	Threads int
}

func (k SwiGLUBackward) norm() (SwiGLUBackward, string, error) {
	if k.Name == "" {
		k.Name = "swiglu_bwd"
	}
	if !cIdent(k.Name) {
		return k, "", fmt.Errorf("swiglu backward: bad name %q", k.Name)
	}
	if k.Cols < 1 {
		return k, "", fmt.Errorf("swiglu backward %s: cols %d must be positive", k.Name, k.Cols)
	}
	if k.DType == 0 {
		k.DType = BF16
	}
	ct, err := bwdCType(k.DType)
	if err != nil {
		return k, "", fmt.Errorf("swiglu backward %s: %w", k.Name, err)
	}
	k.Threads, err = normThreads(k.Threads)
	if err != nil {
		return k, "", fmt.Errorf("swiglu backward %s: %w", k.Name, err)
	}
	return k, ct, nil
}

func (k SwiGLUBackward) params(ct string) []string {
	return []string{
		ct + "* __restrict__ dGateUp",
		"const " + ct + "* __restrict__ dY",
		"const " + ct + "* __restrict__ GateUp",
		"int rows",
	}
}

// Grid returns the launch grid and block for rows rows.
func (k SwiGLUBackward) Grid(rows int) (grid, block [3]int) {
	threads := k.Threads
	if threads == 0 {
		threads = 256
	}
	return [3]int{rows, 1, 1}, [3]int{threads, 1, 1}
}

// CUDA returns the kernel source (with [backwardPrelude], without [Prelude]).
func (k SwiGLUBackward) CUDA() (string, error) {
	k, ct, err := k.norm()
	if err != nil {
		return "", err
	}
	var b strings.Builder
	b.WriteString(backwardPrelude)
	b.WriteString("\n")
	fmt.Fprintf(&b, "// %s: fused SwiGLU backward over rows of %d values (GateUp width %d).\n", k.Name, k.Cols, 2*k.Cols)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(%d) %s(\n    %s) {\n",
		k.Threads, k.Name, strings.Join(k.params(ct), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int COLS = %d, THREADS = %d;\n", k.Cols, k.Threads)
	fmt.Fprintf(&b, "    using T = %s;\n", ct)
	b.WriteString(`    const int r = blockIdx.x;
    if (r >= rows) return;
    const T* gu_row = GateUp + (size_t)r * (2 * COLS);
    const T* dy_row = dY + (size_t)r * COLS;
    T* dgu_row = dGateUp + (size_t)r * (2 * COLS);
    for (int i = threadIdx.x; i < COLS; i += THREADS) {
        const float dy = eg_to_f32(dy_row[i]);
        const float g = eg_to_f32(gu_row[i]);
        const float u = eg_to_f32(gu_row[COLS + i]);
        const float sig = 1.0f / (1.0f + expf(-g));
        const float silu = g * sig;
        const float dsilu = sig * (1.0f + g * (1.0f - sig));
        dgu_row[i] = eg_from_f32<T>(dy * u * dsilu);
        dgu_row[COLS + i] = eg_from_f32<T>(dy * silu);
    }
}
`)
	return b.String(), nil
}

// Launcher returns void Name_launch(<params...>, cudaStream_t stream).
func (k SwiGLUBackward) Launcher() (string, error) {
	k, ct, err := k.norm()
	if err != nil {
		return "", err
	}
	return launcher(k.Name, k.params(ct), func(b *strings.Builder) {
		fmt.Fprintf(b, "    const dim3 grid(rows > 0 ? rows : 0), block(%d);\n", k.Threads)
	}), nil
}

// Emit returns the complete CUDA C++ translation unit ([Prelude], kernel, and launcher).
func (k SwiGLUBackward) Emit() (string, error) {
	src, err := k.CUDA()
	if err != nil {
		return "", err
	}
	l, err := k.Launcher()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src + "\n" + l, nil
}

// CrossEntropySoftcapBackward emits a fused CUDA C++ forward-loss + backward
// kernel for cross-entropy with optional logit softcapping:
//
//	z_cap = Softcap * tanh(z / Softcap)   (when Softcap > 0, else z)
//	loss[r] = -log(softmax(z_cap[r])[targets[r]])
//	dLogits[r, i] = (softmax(z_cap[r])[i] - 1[i == targets[r]]) * dLoss[r] * dz_cap
//
// Targets outside 0..Vocab-1 are treated as masked/ignored tokens with zero
// loss and zero logit gradients.
type CrossEntropySoftcapBackward struct {
	Name    string
	Vocab   int
	Softcap float32 // if > 0, applies cap * tanh(z / cap) before softmax
	DType   DType
	Threads int
}

func (k CrossEntropySoftcapBackward) norm() (CrossEntropySoftcapBackward, string, error) {
	if k.Name == "" {
		k.Name = "cross_entropy_softcap_bwd"
	}
	if !cIdent(k.Name) {
		return k, "", fmt.Errorf("cross entropy backward: bad name %q", k.Name)
	}
	if k.Vocab < 1 {
		return k, "", fmt.Errorf("cross entropy backward %s: vocab %d must be positive", k.Name, k.Vocab)
	}
	if k.DType == 0 {
		k.DType = BF16
	}
	ct, err := bwdCType(k.DType)
	if err != nil {
		return k, "", fmt.Errorf("cross entropy backward %s: %w", k.Name, err)
	}
	k.Threads, err = normThreads(k.Threads)
	if err != nil {
		return k, "", fmt.Errorf("cross entropy backward %s: %w", k.Name, err)
	}
	return k, ct, nil
}

func (k CrossEntropySoftcapBackward) params(ct string) []string {
	return []string{
		ct + "* __restrict__ dLogits",
		"float* __restrict__ losses",
		"const float* __restrict__ dLoss",
		"const " + ct + "* __restrict__ logits",
		"const int* __restrict__ targets",
		"int rows",
	}
}

// Grid returns the launch grid and block for rows rows.
func (k CrossEntropySoftcapBackward) Grid(rows int) (grid, block [3]int) {
	threads := k.Threads
	if threads == 0 {
		threads = 256
	}
	return [3]int{rows, 1, 1}, [3]int{threads, 1, 1}
}

// CUDA returns the kernel source (with [backwardPrelude], without [Prelude]).
func (k CrossEntropySoftcapBackward) CUDA() (string, error) {
	k, ct, err := k.norm()
	if err != nil {
		return "", err
	}
	var b strings.Builder
	b.WriteString(backwardPrelude)
	b.WriteString("\n")
	fmt.Fprintf(&b, "// %s: fused cross-entropy + softcap backward over vocab %d.\n", k.Name, k.Vocab)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(%d) %s(\n    %s) {\n",
		k.Threads, k.Name, strings.Join(k.params(ct), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int VOCAB = %d, THREADS = %d;\n", k.Vocab, k.Threads)
	fmt.Fprintf(&b, "    constexpr float CAP = %s;\n", floatLiteral(k.Softcap))
	fmt.Fprintf(&b, "    using T = %s;\n", ct)
	b.WriteString(`    const int r = blockIdx.x;
    if (r >= rows) return;
    __shared__ float eg_red[32];
    const T* z_row = logits + (size_t)r * VOCAB;
    T* dz_row = dLogits + (size_t)r * VOCAB;
    const int target = targets[r];
    if (target < 0 || target >= VOCAB) {
        if (threadIdx.x == 0) losses[r] = 0.0f;
        for (int i = threadIdx.x; i < VOCAB; i += THREADS) {
            dz_row[i] = eg_from_f32<T>(0.0f);
        }
        return;
    }
    auto cap_val = [&](float z) -> float {
        return CAP > 0.0f ? CAP * tanhf(z / CAP) : z;
    };
    float row_max = -__uint_as_float(0x7f800000u);
    for (int i = threadIdx.x; i < VOCAB; i += THREADS) {
        const float z_cap = cap_val(eg_to_f32(z_row[i]));
        row_max = fmaxf(row_max, z_cap);
    }
    row_max = eg_block_combine<1>(eg_warp_max(row_max), eg_red);
    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < VOCAB; i += THREADS) {
        const float z_cap = cap_val(eg_to_f32(z_row[i]));
        sum_exp += expf(z_cap - row_max);
    }
    sum_exp = eg_block_combine<0>(eg_warp_sum(sum_exp), eg_red);
    if (threadIdx.x == 0) {
        const float z_t = cap_val(eg_to_f32(z_row[target]));
        losses[r] = -(z_t - row_max - logf(sum_exp));
    }
    const float inv_sum = 1.0f / sum_exp;
    const float dloss = dLoss[r];
    for (int i = threadIdx.x; i < VOCAB; i += THREADS) {
        const float z = eg_to_f32(z_row[i]);
        float z_cap = z;
        float dz_cap = 1.0f;
        if (CAP > 0.0f) {
            z_cap = CAP * tanhf(z / CAP);
            const float u = z_cap / CAP;
            dz_cap = 1.0f - u * u;
        }
        const float prob_i = expf(z_cap - row_max) * inv_sum;
        const float grad = ((prob_i - (i == target ? 1.0f : 0.0f)) * dloss) * dz_cap;
        dz_row[i] = eg_from_f32<T>(grad);
    }
}
`)
	return b.String(), nil
}

// Launcher returns void Name_launch(<params...>, cudaStream_t stream).
func (k CrossEntropySoftcapBackward) Launcher() (string, error) {
	k, ct, err := k.norm()
	if err != nil {
		return "", err
	}
	return launcher(k.Name, k.params(ct), func(b *strings.Builder) {
		fmt.Fprintf(b, "    const dim3 grid(rows > 0 ? rows : 0), block(%d);\n", k.Threads)
	}), nil
}

// Emit returns the complete CUDA C++ translation unit ([Prelude], kernel, and launcher).
func (k CrossEntropySoftcapBackward) Emit() (string, error) {
	src, err := k.CUDA()
	if err != nil {
		return "", err
	}
	l, err := k.Launcher()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src + "\n" + l, nil
}
