package tcgen05

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// Prelude is the self-contained device prelude every emitted tcgen05 or CLC
// kernel begins with.
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

// eg_e4m3x2_to_f16x2 widens the two E4M3 bytes in bits 0-15 of x to F16.
EG_DEV uint32_t eg_e4m3x2_to_f16x2(uint32_t x) {
#if defined(EG_HOST_EMU)
    return eg_emu_f32_to_f16(eg_emu_e4m3(x & 0xffu)) | ((uint32_t)eg_emu_f32_to_f16(eg_emu_e4m3((x >> 8) & 0xffu)) << 16);
#else
    uint32_t r;
    asm("{\n.reg .b16 lo;\ncvt.u16.u32 lo, %1;\ncvt.rn.f16x2.e4m3x2 %0, lo;\n}" : "=r"(r) : "r"(x));
    return r;
#endif
}

// eg_f32x2_to_e4m3x2 rounds to nearest even, saturating to ±448.
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
// codes 2 and 3.
EG_DEV void eg_e2m1x4_to_bf16x4(uint32_t q, uint32_t out[2]) {
    const uint32_t lo_a = 0xc0800000u, lo_b = 0xc0804000u;
    const uint32_t hi_a = 0x3f3f3f00u, hi_b = 0x40404040u;
    const uint32_t idx = q & 0x7777u;
    const uint32_t lo = eg_prmt(lo_a, lo_b, idx), hi = eg_prmt(hi_a, hi_b, idx);
    out[0] = eg_prmt(lo, hi, 0x5140u) | ((q & 0x8u) << 12) | ((q & 0x80u) << 24);
    out[1] = eg_prmt(lo, hi, 0x7362u) | ((q & 0x800u) << 4) | ((q & 0x8000u) << 16);
}

// eg_wgmma_desc returns a shared-memory matrix descriptor (no swizzle) for the
// matrix at p with leading and stride byte offsets.
EG_DEV uint64_t eg_wgmma_desc(const void* p, uint32_t lbo, uint32_t sbo) {
#if defined(EG_HOST_EMU)
    const uint64_t a = (uint64_t)((const unsigned char*)p - eg_emu_dynamic_smem());
#else
    const uint64_t a = (uint32_t)__cvta_generic_to_shared(p);
#endif
    return ((a & 0x3ffffu) >> 4) | (uint64_t)((lbo & 0x3ffffu) >> 4) << 16 | (uint64_t)((sbo & 0x3ffffu) >> 4) << 32;
}

// eg_fence_proxy_async orders generic-proxy shared-memory writes before async-proxy reads.
EG_DEV void eg_fence_proxy_async() {
#if !defined(EG_HOST_EMU)
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
}

#if defined(EG_HOST_EMU)
#define EG_DYNAMIC_SMEM(T, name) T* name = reinterpret_cast<T*>(eg_emu_dynamic_smem())
#else
#define EG_DYNAMIC_SMEM(T, name) extern __shared__ __align__(16) unsigned char eg_smem_raw[]; T* name = reinterpret_cast<T*>(eg_smem_raw)
#endif

struct alignas(8) eg_u32x2 { uint32_t x, y; };

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

EG_DEV float eg_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
EG_DEV float eg_silu(float x) { return x / (1.0f + expf(-x)); }
#endif // EG_PRELUDE
`

// A DType is the element type of a tensor in memory.
type DType uint8

const (
	F32  DType = iota + 1
	BF16       // bfloat16, round to nearest even
	E4M3       // float8_e4m3fn, saturating to ±448
	E2M1       // float4_e2m1fn, two codes per byte, element 2i in the low nibble
	U8
	I32
	I64
)

var dtypeNames = [...]string{F32: "f32", BF16: "bf16", E4M3: "e4m3", E2M1: "e2m1", U8: "u8", I32: "i32", I64: "i64"}

func (d DType) String() string {
	if int(d) < len(dtypeNames) && dtypeNames[d] != "" {
		return dtypeNames[d]
	}
	return fmt.Sprintf("DType(%d)", uint8(d))
}

// ParseDType returns the DType named s, as printed by [DType.String].
func ParseDType(s string) (DType, error) {
	for d, name := range dtypeNames {
		if name != "" && name == s {
			return DType(d), nil
		}
	}
	return 0, fmt.Errorf("unknown dtype %q", s)
}

// Bits reports the storage size of one element in bits.
func (d DType) Bits() int {
	switch d {
	case F32, I32:
		return 32
	case BF16:
		return 16
	case E4M3, U8:
		return 8
	case E2M1:
		return 4
	case I64:
		return 64
	}
	return 0
}

func (d DType) cType() string {
	switch d {
	case F32:
		return "float"
	case BF16:
		return "uint16_t"
	case E4M3, E2M1, U8:
		return "uint8_t"
	case I32:
		return "int32_t"
	case I64:
		return "int64_t"
	}
	return ""
}

// Round returns f rounded to d, as the matching CUDA conversion does.
func (d DType) Round(f float32) float32 {
	switch d {
	case BF16:
		return FromBF16Bits(BF16Bits(f))
	case E4M3:
		return FromE4M3Bits(E4M3Bits(f))
	}
	return f
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

// SplitBF16 splits f into hi + lo, both bfloat16, with hi = bf16(f) and lo = bf16(f - hi).
func SplitBF16(f float32) (hi, lo uint16) {
	hi = BF16Bits(f)
	lo = BF16Bits(f - FromBF16Bits(hi))
	return hi, lo
}

// E4M3Bits encodes f as float8_e4m3fn, rounding to nearest even and saturating finite overflow to ±448.
func E4M3Bits(f float32) uint8 {
	bits := math.Float32bits(f)
	sign := uint8(bits>>24) & 0x80
	a := bits & 0x7fffffff
	if a > 0x7f800000 {
		return sign | 0x7f
	}
	if a >= 0x43e00000 {
		return sign | 0x7e
	}
	if a < 0x3c800000 {
		m := uint8(math.RoundToEven(float64(math.Float32frombits(a)) * 512))
		return sign | m
	}
	e := int(a>>23) - 127 + 7
	m := (a >> 20) & 7
	rem := a & 0xfffff
	if rem > 0x80000 || rem == 0x80000 && m&1 == 1 {
		m++
		if m == 8 {
			m = 0
			e++
		}
	}
	if e > 15 || e == 15 && m == 7 {
		return sign | 0x7e
	}
	return sign | uint8(e<<3) | uint8(m)
}

// FromE4M3Bits decodes a float8_e4m3fn encoding.
func FromE4M3Bits(b uint8) float32 {
	e, m := int(b>>3)&15, int(b&7)
	var v float32
	switch {
	case e == 15 && m == 7:
		v = float32(math.NaN())
	case e == 0:
		v = float32(m) / 512
	default:
		v = float32(math.Ldexp(1+float64(m)/8, e-7))
	}
	if b&0x80 != 0 {
		return -v
	}
	return v
}

var e2m1Grid = [8]float32{0, 0.5, 1, 1.5, 2, 3, 4, 6}

// FromE2M1 decodes a 4-bit E2M1 code; bit 3 is the sign.
func FromE2M1(code uint8) float32 {
	v := e2m1Grid[code&7]
	if code&8 != 0 {
		return -v
	}
	return v
}

// NVFP4Block is the number of values sharing one E4M3 block scale.
const NVFP4Block = 16

// DequantNVFP4 decodes one NVFP4 row of k values.
func DequantNVFP4(dst []float32, packed, scales []uint8, global float32) error {
	k := len(dst)
	if k%NVFP4Block != 0 || len(packed) != k/2 || len(scales) != k/NVFP4Block {
		return fmt.Errorf("dequant nvfp4: %d values need %d packed bytes and %d scales, have %d and %d",
			k, k/2, k/NVFP4Block, len(packed), len(scales))
	}
	for i := range dst {
		code := packed[i/2] >> (4 * (i & 1)) & 15
		dst[i] = FromE2M1(code) * (FromE4M3Bits(scales[i/NVFP4Block]) * global)
	}
	return nil
}

// A WeightFormat specifies how a GEMM weight matrix W [N, K] is stored.
type WeightFormat uint8

const (
	WeightBF16 WeightFormat = iota + 1
	WeightBF16HiLo
	WeightNVFP4
	WeightE4M3
)

func (f WeightFormat) String() string {
	switch f {
	case WeightBF16:
		return "bf16"
	case WeightBF16HiLo:
		return "bf16hilo"
	case WeightNVFP4:
		return "nvfp4"
	case WeightE4M3:
		return "e4m3"
	}
	return fmt.Sprintf("WeightFormat(%d)", uint8(f))
}

// GEMMWeights holds W in one of the formats of [WeightFormat].
type GEMMWeights struct {
	BF16       []uint16
	Lo         []uint16
	Packed     []uint8
	Scales     []uint8
	GlobalScal float32
	E4M3       []uint8
}

// cIdent reports whether s is a valid C identifier that cannot collide with eg_ names.
func cIdent(s string) bool {
	if s == "" || len(s) > 64 || strings.HasPrefix(s, "eg_") || strings.HasPrefix(s, "__") {
		return false
	}
	for i, r := range s {
		if !(r == '_' || r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || i > 0 && r >= '0' && r <= '9') {
			return false
		}
	}
	return true
}

func floatLiteral(v float32) string {
	s := strconv.FormatFloat(float64(v), 'g', -1, 32)
	if !strings.ContainsAny(s, ".eE") {
		s += ".0"
	}
	return s + "f"
}

// TileEpilogue selects the elementwise epilogue applied to TCGen05GEMM accumulators.
type TileEpilogue uint8

const (
	TileLinear TileEpilogue = iota // identity: y = acc
	TileSiLU                       // SiLU: y = acc / (1 + exp(-acc))
	TileGLU                        // SwiGLU over paired (gate, up) columns: y = silu(gate) * up
)

func (e TileEpilogue) String() string {
	switch e {
	case TileLinear:
		return "linear"
	case TileSiLU:
		return "silu"
	case TileGLU:
		return "glu"
	}
	return fmt.Sprintf("TileEpilogue(%d)", uint8(e))
}

func (e TileEpilogue) valid() bool {
	return e == TileLinear || e == TileSiLU || e == TileGLU
}

// TCGen05GEMM configures a Blackwell sm_100a/sm_103a Tensor Memory (TMEM) GEMM kernel
// using tcgen05.alloc, tcgen05.mma.cta_group::1, tcgen05.ld, and tcgen05.dealloc.
type TCGen05GEMM struct {
	Name     string
	N, K     int
	BM, BN   int          // e.g., BM=128, BN=128 or 256 (accumulators live in TMEM, not registers!)
	BK       int          // reduction tile step (default 64)
	Weight   WeightFormat // WeightBF16, WeightE4M3, WeightNVFP4
	Out      DType
	Epilogue TileEpilogue // TileLinear, TileSiLU, TileGLU
}

const tcgen05DeviceHelpers = `// Blackwell sm_100a Tensor Memory (TMEM) and tcgen05 helpers.
EG_DEV void eg_tmem_alloc(uint32_t* smem_addr_out, uint32_t num_cols) {
#if defined(EG_EMU) || defined(EG_HOST_EMU)
    if (threadIdx.x == 0) eg_emu_tmem_alloc(smem_addr_out, num_cols);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    if ((threadIdx.x >> 5) == 0) {
        const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem_addr_out);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(s), "r"(num_cols) : "memory");
    }
#endif
}

EG_DEV void eg_tmem_dealloc(uint32_t tmem_addr, uint32_t num_cols) {
#if defined(EG_EMU) || defined(EG_HOST_EMU)
    if (threadIdx.x == 0) eg_emu_tmem_dealloc(tmem_addr, num_cols);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    if ((threadIdx.x >> 5) == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr), "r"(num_cols) : "memory");
    }
#endif
}

// eg_tcgen05_mma_bf16, eg_tcgen05_mma_e4m3, and eg_tcgen05_mma_nvfp4 issue one
// tcgen05.mma with the instruction descriptor idesc, which the emitting Go code
// encodes for the instruction's kind and shape.
EG_DEV void eg_tcgen05_mma_bf16(uint32_t tmem_d, const void* smem_a, const void* smem_b, int M, int N, int K, uint32_t idesc, int accum_enable) {
#if defined(EG_EMU) || defined(EG_HOST_EMU)
    (void)idesc;
    if (threadIdx.x == 0) eg_emu_tcgen05_mma_bf16(tmem_d, smem_a, smem_b, M, N, K, accum_enable);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    if (threadIdx.x == 0) {
        const uint64_t da = eg_wgmma_desc(smem_a, (uint32_t)(K * 2), (uint32_t)(8 * K * 2));
        const uint64_t db = eg_wgmma_desc(smem_b, (uint32_t)(K * 2), (uint32_t)(8 * K * 2));
        asm volatile(
            "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
            :: "r"(tmem_d), "l"(da), "l"(db), "r"(idesc), "r"(accum_enable) : "memory");
    }
#endif
}

EG_DEV void eg_tcgen05_mma_e4m3(uint32_t tmem_d, const void* smem_a, const void* smem_b, int M, int N, int K, uint32_t idesc, int accum_enable) {
#if defined(EG_EMU) || defined(EG_HOST_EMU)
    (void)idesc;
    if (threadIdx.x == 0) eg_emu_tcgen05_mma_e4m3(tmem_d, smem_a, smem_b, M, N, K, accum_enable);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    if (threadIdx.x == 0) {
        const uint64_t da = eg_wgmma_desc(smem_a, (uint32_t)K, (uint32_t)(8 * K));
        const uint64_t db = eg_wgmma_desc(smem_b, (uint32_t)K, (uint32_t)(8 * K));
        asm volatile(
            "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
            "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n}\n"
            :: "r"(tmem_d), "l"(da), "l"(db), "r"(idesc), "r"(accum_enable) : "memory");
    }
#endif
}

EG_DEV void eg_tcgen05_mma_nvfp4(
    uint32_t tmem_d,
    const uint8_t* smem_a_fp4, const uint8_t* smem_a_scale,
    const uint8_t* smem_b_fp4, const uint8_t* smem_b_scale,
    int M, int N, int K, uint32_t idesc, int accum_enable) {
#if defined(EG_EMU) || defined(EG_HOST_EMU)
    (void)idesc;
    if (threadIdx.x == 0) {
        eg_emu_tcgen05_mma_nvfp4(tmem_d, smem_a_fp4, smem_a_scale, smem_b_fp4, smem_b_scale, M, N, K, accum_enable);
    }
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    if (threadIdx.x == 0) {
        const uint64_t da = eg_wgmma_desc(smem_a_fp4, (uint32_t)(K / 2), (uint32_t)(8 * (K / 2)));
        const uint64_t db = eg_wgmma_desc(smem_b_fp4, (uint32_t)(K / 2), (uint32_t)(8 * (K / 2)));
        const uint32_t sfa = (uint32_t)__cvta_generic_to_shared(smem_a_scale);
        const uint32_t sfb = (uint32_t)__cvta_generic_to_shared(smem_b_scale);
        asm volatile(
            "{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%4], [%5], p;\n}\n"
            :: "r"(tmem_d), "l"(da), "l"(db), "r"(idesc), "r"(sfa), "r"(sfb), "r"(accum_enable) : "memory");
    }
#endif
}

EG_DEV void eg_tcgen05_commit(uint64_t* mbar) {
#if defined(EG_EMU) || defined(EG_HOST_EMU)
    (void)mbar;
    if (threadIdx.x == 0) eg_emu_tcgen05_commit();
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    if (threadIdx.x == 0) {
        const uint32_t s = (uint32_t)__cvta_generic_to_shared(mbar);
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(s) : "memory");
    }
#endif
}

EG_DEV void eg_tcgen05_wait() {
#if defined(EG_EMU) || defined(EG_HOST_EMU)
    eg_emu_tcgen05_wait();
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#endif
}

EG_DEV void eg_tcgen05_ld_f32(float* reg_out, uint32_t tmem_addr, int row, int col_offset, int count) {
#if defined(EG_EMU) || defined(EG_HOST_EMU)
    eg_emu_tcgen05_ld_f32(reg_out, tmem_addr, row, col_offset, count);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    for (int c = 0; c < count; c += 4) {
        const uint32_t addr = ((uint32_t)(row & 0x7f) << 16) | ((tmem_addr + (uint32_t)col_offset + (uint32_t)c) & 0xffffu);
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];"
            : "=f"(reg_out[c]), "=f"(reg_out[c + 1]), "=f"(reg_out[c + 2]), "=f"(reg_out[c + 3])
            : "r"(addr) : "memory");
    }
#endif
}

`

func (g TCGen05GEMM) norm() (TCGen05GEMM, error) {
	if g.Name == "" {
		g.Name = "tcgen05_gemm"
	}
	if !cIdent(g.Name) {
		return g, fmt.Errorf("tcgen05 gemm: bad name %q", g.Name)
	}
	if g.BM == 0 {
		g.BM = 128
	}
	if g.BN == 0 {
		g.BN = 128
	}
	if g.BK == 0 {
		g.BK = 64
	}
	if g.Out == 0 {
		g.Out = F32
	}
	if g.BM != 64 && g.BM != 128 {
		return g, fmt.Errorf("tcgen05 gemm %s: BM %d must be 64 or 128", g.Name, g.BM)
	}
	if g.BN < 32 || g.BN > 512 || g.BN&(g.BN-1) != 0 {
		return g, fmt.Errorf("tcgen05 gemm %s: BN %d must be a power of two in [32, 512]", g.Name, g.BN)
	}
	if g.BK < 16 || g.BK%16 != 0 {
		return g, fmt.Errorf("tcgen05 gemm %s: BK %d must be a positive multiple of 16", g.Name, g.BK)
	}
	if g.N < g.BN || g.N%g.BN != 0 {
		return g, fmt.Errorf("tcgen05 gemm %s: N %d must be a positive multiple of BN %d", g.Name, g.N, g.BN)
	}
	if g.K < g.BK || g.K%g.BK != 0 {
		return g, fmt.Errorf("tcgen05 gemm %s: K %d must be a positive multiple of BK %d", g.Name, g.K, g.BK)
	}
	switch g.Weight {
	case WeightBF16, WeightE4M3, WeightNVFP4:
	default:
		return g, fmt.Errorf("tcgen05 gemm %s: unsupported weight format %v", g.Name, g.Weight)
	}
	if g.Out != F32 && g.Out != BF16 {
		return g, fmt.Errorf("tcgen05 gemm %s: out must be F32 or BF16, got %v", g.Name, g.Out)
	}
	if !g.Epilogue.valid() {
		return g, fmt.Errorf("tcgen05 gemm %s: invalid epilogue %v", g.Name, g.Epilogue)
	}
	if g.Epilogue == TileGLU && (g.BN < 64 || (g.N/2)%(g.BN/2) != 0) {
		return g, fmt.Errorf("tcgen05 gemm %s: GLU requires BN >= 64 and N/2 divisible by BN/2", g.Name)
	}
	return g, nil
}

// SharedBytes returns the dynamic shared memory bytes per CTA needed by g.
func (g TCGen05GEMM) SharedBytes() int {
	g, err := g.norm()
	if err != nil {
		return 0
	}
	switch g.Weight {
	case WeightBF16:
		return (g.BM + g.BN) * g.BK * 2
	case WeightE4M3:
		return (g.BM + g.BN) * g.BK
	case WeightNVFP4:
		return (g.BM+g.BN)*(g.BK/2) + (g.BM+g.BN)*(g.BK/16)
	}
	return 0
}

// Grid returns the launch grid and block dimensions for m activation rows.
func (g TCGen05GEMM) Grid(m int) (grid, block [3]int) {
	g, err := g.norm()
	if err != nil {
		return
	}
	return [3]int{g.N / g.BN, (m + g.BM - 1) / g.BM, 1}, [3]int{128, 1, 1}
}

func (g TCGen05GEMM) params() []string {
	switch g.Weight {
	case WeightNVFP4:
		return []string{
			"int m_total",
			"const uint8_t* __restrict__ a_fp4",
			"const uint8_t* __restrict__ a_scale",
			"const uint8_t* __restrict__ b_fp4",
			"const uint8_t* __restrict__ b_scale",
			g.Out.cType() + "* __restrict__ out",
		}
	case WeightE4M3:
		return []string{
			"int m_total",
			"const uint8_t* __restrict__ x",
			"const uint8_t* __restrict__ w",
			g.Out.cType() + "* __restrict__ out",
		}
	default:
		return []string{
			"int m_total",
			"const uint16_t* __restrict__ x",
			"const uint16_t* __restrict__ w",
			g.Out.cType() + "* __restrict__ out",
		}
	}
}

// instDesc returns the instruction descriptor of the BM x BN tcgen05.mma g
// issues: .kind::f16 (BF16, K=16), .kind::f8f6f4 (E4M3, K=32), or
// .kind::mxf4nvf4 (E2M1 with UE4M3 scales, K=64), accumulating in F32.
func (g TCGen05GEMM) instDesc() InstDesc {
	switch g.Weight {
	case WeightE4M3:
		return InstDesc{Kind: KindF8F6F4, DType: ElemF32, AType: ElemE4M3, BType: ElemE4M3, M: g.BM, N: g.BN, K: 32}
	case WeightNVFP4:
		return InstDesc{Kind: KindMXF4NVF4, AType: ElemE2M1, BType: ElemE2M1, M: g.BM, N: g.BN, K: 64, ScaleType: ScaleUE4M3}
	}
	return InstDesc{Kind: KindF16, DType: ElemF32, AType: ElemBF16, BType: ElemBF16, M: g.BM, N: g.BN, K: 16}
}

// CUDA returns the kernel source without [Prelude].
func (g TCGen05GEMM) CUDA() (string, error) {
	g, err := g.norm()
	if err != nil {
		return "", err
	}
	idesc, err := g.instDesc().Encode()
	if err != nil {
		return "", fmt.Errorf("tcgen05 gemm %s: %w", g.Name, err)
	}
	var b strings.Builder
	b.WriteString(tcgen05DeviceHelpers)
	fmt.Fprintf(&b, "// %s: Blackwell sm_100a TMEM GEMM (%v, epilogue=%s, %dx%dx%d tile).\n",
		g.Name, g.Weight, g.Epilogue, g.BM, g.BN, g.BK)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(128) %s(\n    %s) {\n",
		g.Name, strings.Join(g.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int N = %d, K = %d, BM = %d, BN = %d, BK = %d, KT = K / BK;\n",
		g.N, g.K, g.BM, g.BN, g.BK)
	fmt.Fprintf(&b, "    constexpr uint32_t IDESC = 0x%08xu;\n", idesc)
	if g.Epilogue == TileGLU {
		b.WriteString("    constexpr int NOUT = N / 2;\n")
	} else {
		b.WriteString("    constexpr int NOUT = N;\n")
	}
	b.WriteString(`    const int tid = threadIdx.x;
    const int bm0 = blockIdx.y * BM;
    const int bn0 = blockIdx.x * BN;
    if (bm0 >= m_total) return;
    __shared__ uint32_t s_tmem_addr;
    __shared__ uint64_t s_tmem_mbar;
    eg_tmem_alloc(&s_tmem_addr, BN);
    __syncthreads();
    const uint32_t tmem_d = s_tmem_addr;
    EG_DYNAMIC_SMEM(unsigned char, smem);
`)
	if g.Epilogue == TileGLU {
		b.WriteString(`    auto wrow = [&](int j) {
        const int c = blockIdx.x * (BN / 2) + j % (BN / 2);
        return j < BN / 2 ? c : NOUT + c;
    };
`)
	} else {
		b.WriteString("    auto wrow = [&](int j) { return bn0 + j; };\n")
	}

	switch g.Weight {
	case WeightBF16:
		b.WriteString(`    uint16_t* const sa = reinterpret_cast<uint16_t*>(smem);
    uint16_t* const sb = sa + BM * BK;
    for (int kt = 0; kt < KT; ++kt) {
        for (int i = tid; i < BM * BK; i += 128) {
            const int r = i / BK, k = i % BK;
            const int gm = bm0 + r;
            sa[i] = (gm < m_total) ? x[(size_t)gm * K + kt * BK + k] : (uint16_t)0u;
        }
        for (int i = tid; i < BN * BK; i += 128) {
            const int r = i / BK, k = i % BK;
            const int gn = wrow(r);
            sb[i] = w[(size_t)gn * K + kt * BK + k];
        }
        eg_fence_proxy_async();
        __syncthreads();
        eg_tcgen05_mma_bf16(tmem_d, sa, sb, BM, BN, BK, IDESC, kt > 0 ? 1 : 0);
        __syncthreads();
    }
`)
	case WeightE4M3:
		b.WriteString(`    uint8_t* const sa = reinterpret_cast<uint8_t*>(smem);
    uint8_t* const sb = sa + BM * BK;
    for (int kt = 0; kt < KT; ++kt) {
        for (int i = tid; i < BM * BK; i += 128) {
            const int r = i / BK, k = i % BK;
            const int gm = bm0 + r;
            sa[i] = (gm < m_total) ? x[(size_t)gm * K + kt * BK + k] : (uint8_t)0u;
        }
        for (int i = tid; i < BN * BK; i += 128) {
            const int r = i / BK, k = i % BK;
            const int gn = wrow(r);
            sb[i] = w[(size_t)gn * K + kt * BK + k];
        }
        eg_fence_proxy_async();
        __syncthreads();
        eg_tcgen05_mma_e4m3(tmem_d, sa, sb, BM, BN, BK, IDESC, kt > 0 ? 1 : 0);
        __syncthreads();
    }
`)
	case WeightNVFP4:
		b.WriteString(`    constexpr int BK_FP4 = BK / 2, BK_SF = BK / 16;
    constexpr int K_FP4 = K / 2, K_SF = K / 16;
    uint8_t* const sa_fp4 = reinterpret_cast<uint8_t*>(smem);
    uint8_t* const sa_sf = sa_fp4 + BM * BK_FP4;
    uint8_t* const sb_fp4 = sa_sf + BM * BK_SF;
    uint8_t* const sb_sf = sb_fp4 + BN * BK_FP4;
    for (int kt = 0; kt < KT; ++kt) {
        for (int i = tid; i < BM * BK_FP4; i += 128) {
            const int r = i / BK_FP4, k = i % BK_FP4;
            const int gm = bm0 + r;
            sa_fp4[i] = (gm < m_total) ? a_fp4[(size_t)gm * K_FP4 + kt * BK_FP4 + k] : (uint8_t)0u;
        }
        for (int i = tid; i < BM * BK_SF; i += 128) {
            const int r = i / BK_SF, k = i % BK_SF;
            const int gm = bm0 + r;
            sa_sf[i] = (gm < m_total) ? a_scale[(size_t)gm * K_SF + kt * BK_SF + k] : (uint8_t)0u;
        }
        for (int i = tid; i < BN * BK_FP4; i += 128) {
            const int r = i / BK_FP4, k = i % BK_FP4;
            const int gn = wrow(r);
            sb_fp4[i] = b_fp4[(size_t)gn * K_FP4 + kt * BK_FP4 + k];
        }
        for (int i = tid; i < BN * BK_SF; i += 128) {
            const int r = i / BK_SF, k = i % BK_SF;
            const int gn = wrow(r);
            sb_sf[i] = b_scale[(size_t)gn * K_SF + kt * BK_SF + k];
        }
        eg_fence_proxy_async();
        __syncthreads();
        eg_tcgen05_mma_nvfp4(tmem_d, sa_fp4, sa_sf, sb_fp4, sb_sf, BM, BN, BK, IDESC, kt > 0 ? 1 : 0);
        __syncthreads();
    }
`)
	}

	b.WriteString(`    eg_tcgen05_commit(&s_tmem_mbar);
    __syncthreads();
    eg_tcgen05_wait();
    __syncthreads();
    if (tid < BM && bm0 + tid < m_total) {
        const int gm = bm0 + tid;
`)
	switch g.Epilogue {
	case TileGLU:
		b.WriteString(`        const int bn_out = blockIdx.x * (BN / 2);
        for (int c = 0; c < BN / 2; c += 4) {
            float gate[4], up[4], yv[4];
            eg_tcgen05_ld_f32(gate, tmem_d, tid, c, 4);
            eg_tcgen05_ld_f32(up, tmem_d, tid, (BN / 2) + c, 4);
            for (int j = 0; j < 4; ++j) {
                yv[j] = eg_silu(gate[j]) * up[j];
            }
            eg_store<4>(out + (size_t)gm * NOUT + bn_out + c, yv);
        }
`)
	case TileSiLU:
		b.WriteString(`        for (int c = 0; c < BN; c += 4) {
            float acc[4], yv[4];
            eg_tcgen05_ld_f32(acc, tmem_d, tid, c, 4);
            for (int j = 0; j < 4; ++j) {
                yv[j] = eg_silu(acc[j]);
            }
            eg_store<4>(out + (size_t)gm * NOUT + bn0 + c, yv);
        }
`)
	default: // TileLinear
		b.WriteString(`        for (int c = 0; c < BN; c += 4) {
            float acc[4];
            eg_tcgen05_ld_f32(acc, tmem_d, tid, c, 4);
            eg_store<4>(out + (size_t)gm * NOUT + bn0 + c, acc);
        }
`)
	}
	b.WriteString(`    }
    __syncthreads();
    eg_tmem_dealloc(tmem_d, BN);
}
`)
	return b.String(), nil
}

// Launcher returns the host launch wrapper for g.
func (g TCGen05GEMM) Launcher() (string, error) {
	g, err := g.norm()
	if err != nil {
		return "", err
	}
	params := g.params()[1:]
	names := []string{"m"}
	for _, p := range params {
		f := strings.Fields(p)
		names = append(names, f[len(f)-1])
	}
	var b strings.Builder
	fmt.Fprintf(&b, "inline void %s_launch(int m, %s, cudaStream_t stream) {\n", g.Name, strings.Join(params, ", "))
	fmt.Fprintf(&b, "    constexpr int smem = %d;\n", g.SharedBytes())
	fmt.Fprintf(&b, "    static const cudaError_t attr = cudaFuncSetAttribute(%s, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);\n    (void)attr;\n", g.Name)
	fmt.Fprintf(&b, "    const dim3 grid(%d, (m + %d) / %d, 1), block(128);\n", g.N/g.BN, g.BM-1, g.BM)
	fmt.Fprintf(&b, "    if (grid.y > 0) EG_LAUNCH(%s, grid, block, smem, stream, %s);\n}\n", g.Name, strings.Join(names, ", "))
	return b.String(), nil
}

// Source returns [Prelude], the Blackwell tcgen05 kernel, and its launcher.
func (g TCGen05GEMM) Source() (string, error) {
	src, err := g.CUDA()
	if err != nil {
		return "", err
	}
	l, err := g.Launcher()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src + "\n" + l, nil
}

func evalTileEpilogue(e TileEpilogue, gate, up float32) float32 {
	switch e {
	case TileSiLU:
		return gate / (1 + float32(math.Exp(float64(-gate))))
	case TileGLU:
		s := gate / (1 + float32(math.Exp(float64(-gate))))
		return s * up
	default:
		return gate
	}
}

// Reference computes the float64 reference output for WeightBF16 or WeightE4M3
// with FP32 epilogue and Out rounding.
func (g TCGen05GEMM) Reference(m int, x []uint16, w GEMMWeights) ([]float32, error) {
	g, err := g.norm()
	if err != nil {
		return nil, err
	}
	if m < 1 || len(x) != m*g.K {
		return nil, fmt.Errorf("tcgen05 gemm %s reference: bad x length %d for m=%d K=%d", g.Name, len(x), m, g.K)
	}
	nout := g.N
	if g.Epilogue == TileGLU {
		nout = g.N / 2
	}
	dots := make([]float32, m*g.N)
	for n := 0; n < g.N; n++ {
		for r := 0; r < m; r++ {
			var sum float64
			for k := 0; k < g.K; k++ {
				var av, bv float64
				switch g.Weight {
				case WeightBF16:
					av = float64(FromBF16Bits(x[r*g.K+k]))
					bv = float64(FromBF16Bits(w.BF16[n*g.K+k]))
				case WeightE4M3:
					av = float64(FromE4M3Bits(uint8(x[r*g.K+k])))
					bv = float64(FromE4M3Bits(w.E4M3[n*g.K+k]))
				default:
					return nil, fmt.Errorf("tcgen05 gemm %s reference: use ReferenceNVFP4 for WeightNVFP4", g.Name)
				}
				sum += av * bv
			}
			dots[r*g.N+n] = float32(sum)
		}
	}
	out := make([]float32, m*nout)
	for r := 0; r < m; r++ {
		for c := 0; c < nout; c++ {
			var y float32
			if g.Epilogue == TileGLU {
				y = evalTileEpilogue(TileGLU, dots[r*g.N+c], dots[r*g.N+nout+c])
			} else {
				y = evalTileEpilogue(g.Epilogue, dots[r*g.N+c], 0)
			}
			out[r*nout+c] = g.Out.Round(y)
		}
	}
	return out, nil
}

// ReferenceNVFP4 computes the float64 reference output for WeightNVFP4 with
// block-scaled E2M1 operands (aFP4 [m, K/2], aScale [m, K/16], bFP4 [N, K/2],
// bScale [N, K/16]), FP32 epilogue, and Out rounding.
func (g TCGen05GEMM) ReferenceNVFP4(m int, aFP4, aScale, bFP4, bScale []uint8) ([]float32, error) {
	g, err := g.norm()
	if err != nil {
		return nil, err
	}
	if g.Weight != WeightNVFP4 {
		return nil, fmt.Errorf("tcgen05 gemm %s: ReferenceNVFP4 requires WeightNVFP4", g.Name)
	}
	kFP4, kSF := g.K/2, g.K/16
	if len(aFP4) != m*kFP4 || len(aScale) != m*kSF || len(bFP4) != g.N*kFP4 || len(bScale) != g.N*kSF {
		return nil, fmt.Errorf("tcgen05 gemm %s: bad nvfp4 slice lengths", g.Name)
	}
	nout := g.N
	if g.Epilogue == TileGLU {
		nout = g.N / 2
	}
	dots := make([]float32, m*g.N)
	for n := 0; n < g.N; n++ {
		for r := 0; r < m; r++ {
			var sum float64
			for k := 0; k < g.K; k++ {
				ab := aFP4[r*kFP4+k/2]
				bb := bFP4[n*kFP4+k/2]
				ac := (ab >> (4 * (k & 1))) & 0xf
				bc := (bb >> (4 * (k & 1))) & 0xf
				asf := FromE4M3Bits(aScale[r*kSF+k/16])
				bsf := FromE4M3Bits(bScale[n*kSF+k/16])
				av := float64(FromE2M1(ac) * asf)
				bv := float64(FromE2M1(bc) * bsf)
				sum += av * bv
			}
			dots[r*g.N+n] = float32(sum)
		}
	}
	out := make([]float32, m*nout)
	for r := 0; r < m; r++ {
		for c := 0; c < nout; c++ {
			var y float32
			if g.Epilogue == TileGLU {
				y = evalTileEpilogue(TileGLU, dots[r*g.N+c], dots[r*g.N+nout+c])
			} else {
				y = evalTileEpilogue(g.Epilogue, dots[r*g.N+c], 0)
			}
			out[r*nout+c] = g.Out.Round(y)
		}
	}
	return out, nil
}
