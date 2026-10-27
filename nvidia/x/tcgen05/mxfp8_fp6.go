package tcgen05

import (
	"fmt"
	"math"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudatmem"
)

// MXBlock is the standard 32-element block size for OCP microscaling formats
// (MXFP8, MXFP6, and MXFP4) scaled by a shared UE8M0 exponent.
const MXBlock = 32

// UE8M0Bias is the unsigned exponent bias (127) for 8-bit OCP UE8M0 scales.
const UE8M0Bias = 127

// UE8M0NaN is the reserved NaN code (0xFF) in the OCP UE8M0 scale encoding.
const UE8M0NaN uint8 = 0xFF

// EncodeUE8M0 encodes a positive scale factor into an 8-bit OCP UE8M0 biased
// exponent (bias = 127), returning the smallest power-of-two 2^(code-127) >= scale
// (or 127 for 1.0, 0 for scale <= 2^-127, 254 for overflow, and 0xFF for NaN).
func EncodeUE8M0(scale float32) uint8 {
	f := float64(scale)
	if math.IsNaN(f) {
		return UE8M0NaN
	}
	if f <= 0 {
		return 0
	}
	if math.IsInf(f, 1) {
		return 254
	}
	frac, exp := math.Frexp(f)
	e := exp
	if frac == 0.5 {
		e = exp - 1
	}
	code := e + UE8M0Bias
	if code <= 0 {
		return 0
	}
	if code >= 254 {
		return 254
	}
	return uint8(code)
}

// DecodeUE8M0 decodes an 8-bit OCP UE8M0 scale code to float32 (2^(code-127)
// for code in [0..254], or NaN for 0xFF).
func DecodeUE8M0(code uint8) float32 {
	if code == UE8M0NaN {
		return float32(math.NaN())
	}
	return float32(math.Ldexp(1.0, int(code)-UE8M0Bias))
}

// QuantizeMXFP8E4M3 quantizes a slice of float32 values (length a multiple of 32)
// into 32-element OCP MXFP8 blocks using E4M3 elements and UE8M0 block scales.
func QuantizeMXFP8E4M3(src []float32) (codes []uint8, scales []uint8) {
	nBlocks := (len(src) + MXBlock - 1) / MXBlock
	codes = make([]uint8, len(src))
	scales = make([]uint8, nBlocks)
	for b := 0; b < nBlocks; b++ {
		start := b * MXBlock
		end := start + MXBlock
		if end > len(src) {
			end = len(src)
		}
		block := src[start:end]
		var amax float32
		for _, x := range block {
			ax := float32(math.Abs(float64(x)))
			if ax > amax {
				amax = ax
			}
		}
		var sc uint8 = UE8M0Bias
		if amax > 0 {
			sc = EncodeUE8M0(amax / 448.0)
		}
		scales[b] = sc
		scale := DecodeUE8M0(sc)
		for i, x := range block {
			codes[start+i] = E4M3Bits(x / scale)
		}
	}
	return codes, scales
}

// DequantizeMXFP8E4M3 reconstructs float32 values from E4M3 codes and 32-element
// UE8M0 block scales.
func DequantizeMXFP8E4M3(codes []uint8, scales []uint8) []float32 {
	dst := make([]float32, len(codes))
	for i, c := range codes {
		scale := DecodeUE8M0(scales[i/MXBlock])
		dst[i] = FromE4M3Bits(c) * scale
	}
	return dst
}

// EncodeE2M1 encodes v into a 4-bit OCP E2M1 code in [0..15] (bit 3 is sign,
// bits 2..0 select from {0, 0.5, 1, 1.5, 2, 3, 4, 6} by round-to-nearest-even).
func EncodeE2M1(v float32) uint8 {
	var sign uint8
	if math.Signbit(float64(v)) {
		sign = 0x8
		v = -v
	}
	best := uint8(0)
	bestDist := float32(math.Abs(float64(v - e2m1Grid[0])))
	for i := uint8(1); i < 8; i++ {
		d := float32(math.Abs(float64(v - e2m1Grid[i])))
		if d < bestDist || (d == bestDist && (i&1) == 0) {
			bestDist = d
			best = i
		}
	}
	return sign | best
}

// QuantizeMXFP4E2M1 quantizes a slice of float32 values (length a multiple of 32)
// into 32-element OCP MXFP4 blocks (2 E2M1 nibbles per byte, 1 UE8M0 scale per 32 elements).
func QuantizeMXFP4E2M1(src []float32) (packed []uint8, scales []uint8) {
	nBlocks := len(src) / MXBlock
	packed = make([]uint8, len(src)/2)
	scales = make([]uint8, nBlocks)
	for b := 0; b < nBlocks; b++ {
		block := src[b*MXBlock : (b+1)*MXBlock]
		var amax float32
		for _, x := range block {
			ax := float32(math.Abs(float64(x)))
			if ax > amax {
				amax = ax
			}
		}
		var sc uint8 = UE8M0Bias
		if amax > 0 {
			sc = EncodeUE8M0(amax / 6.0)
		}
		scales[b] = sc
		scale := DecodeUE8M0(sc)
		for j := 0; j < MXBlock/2; j++ {
			c0 := EncodeE2M1(block[2*j+0] / scale)
			c1 := EncodeE2M1(block[2*j+1] / scale)
			packed[b*(MXBlock/2)+j] = (c0 & 0xf) | ((c1 & 0xf) << 4)
		}
	}
	return packed, scales
}

// DequantizeMXFP4E2M1 reconstructs float32 values from packed E2M1 bytes and
// 32-element UE8M0 block scales.
func DequantizeMXFP4E2M1(packed []uint8, scales []uint8) []float32 {
	dst := make([]float32, len(packed)*2)
	for i, b := range packed {
		scale := DecodeUE8M0(scales[(2*i)/MXBlock])
		dst[2*i+0] = FromE2M1(b&0xf) * scale
		dst[2*i+1] = FromE2M1((b>>4)&0xf) * scale
	}
	return dst
}

// DecodeFP6E3M2 decodes a 6-bit OCP FP6 E3M2 code in [0..63] (1 sign bit,
// 3 exponent bits with bias 3, 2 mantissa bits, max magnitude 28.0) to float32.
func DecodeFP6E3M2(code uint8) float32 {
	code &= 0x3f
	sign := (code >> 5) & 1
	exp := int((code >> 2) & 0x7)
	mant := int(code & 0x3)
	var v float32
	if exp == 0 {
		v = float32(mant) * 0.0625
	} else {
		v = float32(math.Ldexp(1.0+float64(mant)*0.25, exp-3))
	}
	if sign != 0 {
		return -v
	}
	return v
}

// EncodeFP6E3M2 encodes v into a 6-bit OCP FP6 E3M2 code in [0..63] using
// round-to-nearest-even and saturating outside [-28.0, 28.0].
func EncodeFP6E3M2(v float32) uint8 {
	if math.IsNaN(float64(v)) {
		return 0x1f
	}
	var sign uint8
	if math.Signbit(float64(v)) {
		sign = 0x20
		v = -v
	}
	best := uint8(0)
	bestDist := math.Abs(float64(v - DecodeFP6E3M2(0)))
	for c := uint8(1); c < 32; c++ {
		d := math.Abs(float64(v - DecodeFP6E3M2(c)))
		if d < bestDist || (d == bestDist && (c&1) == 0) {
			bestDist = d
			best = c
		}
	}
	return sign | best
}

// DecodeFP6E2M3 decodes a 6-bit OCP FP6 E2M3 code in [0..63] (1 sign bit,
// 2 exponent bits with bias 1, 3 mantissa bits, max magnitude 7.5) to float32.
func DecodeFP6E2M3(code uint8) float32 {
	code &= 0x3f
	sign := (code >> 5) & 1
	exp := int((code >> 3) & 0x3)
	mant := int(code & 0x7)
	var v float32
	if exp == 0 {
		v = float32(mant) * 0.125
	} else {
		v = float32(math.Ldexp(1.0+float64(mant)*0.125, exp-1))
	}
	if sign != 0 {
		return -v
	}
	return v
}

// EncodeFP6E2M3 encodes v into a 6-bit OCP FP6 E2M3 code in [0..63] using
// round-to-nearest-even and saturating outside [-7.5, 7.5].
func EncodeFP6E2M3(v float32) uint8 {
	if math.IsNaN(float64(v)) {
		return 0x1f
	}
	var sign uint8
	if math.Signbit(float64(v)) {
		sign = 0x20
		v = -v
	}
	best := uint8(0)
	bestDist := math.Abs(float64(v - DecodeFP6E2M3(0)))
	for c := uint8(1); c < 32; c++ {
		d := math.Abs(float64(v - DecodeFP6E2M3(c)))
		if d < bestDist || (d == bestDist && (c&1) == 0) {
			bestDist = d
			best = c
		}
	}
	return sign | best
}

// PackFP6x4 packs four 6-bit FP6 codes (c0, c1, c2, c3 in [0..63]) into 3
// contiguous bytes (24 bits):
//
//	b[0] = c0 | (c1 << 6)
//	b[1] = (c1 >> 2) | (c2 << 4)
//	b[2] = (c2 >> 4) | (c3 << 2)
func PackFP6x4(c0, c1, c2, c3 uint8) [3]uint8 {
	c0 &= 0x3f
	c1 &= 0x3f
	c2 &= 0x3f
	c3 &= 0x3f
	return [3]uint8{
		c0 | (c1 << 6),
		(c1 >> 2) | (c2 << 4),
		(c2 >> 4) | (c3 << 2),
	}
}

// UnpackFP6x4 unpacks 3 contiguous bytes (24 bits) into four 6-bit FP6 codes
// in [0..63].
func UnpackFP6x4(b [3]uint8) [4]uint8 {
	return [4]uint8{
		b[0] & 0x3f,
		((b[0] >> 6) | (b[1] << 2)) & 0x3f,
		((b[1] >> 4) | (b[2] << 4)) & 0x3f,
		(b[2] >> 2) & 0x3f,
	}
}

// TCGen05MXFP8GEMM emits a Blackwell sm_100a/sm_103a OCP microscaling GEMM
// using tcgen05.cp.cta_group::1.32x128b.warpx4 to stage 32-element UE8M0 block
// scales into TMEM and tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X
// for either pure MXFP8 (E4M3 x E4M3) or mixed W4A8 (MXFP8 E4M3 activations x
// MXFP4 E2M1 weights), with optional tcgen05.shift.cta_group::1.down TMEM row
// shifting.
type TCGen05MXFP8GEMM struct {
	Name         string
	M, N, K      int  // M=128, N=128, K multiple of 32 in 32..256 (default 64)
	MixedW4A8    bool // when false: pure MXFP8 (E4M3 x E4M3); when true: W4A8 (MXFP8 E4M3 x MXFP4 E2M1)
	UseShiftDown bool // when true: emits tcgen05.shift.cta_group::1.down on TMEM accumulator tile
}

func (k TCGen05MXFP8GEMM) norm() (TCGen05MXFP8GEMM, error) {
	if k.Name == "" {
		switch {
		case k.MixedW4A8 && k.UseShiftDown:
			k.Name = "tcgen05_w4a8_shift_gemm"
		case k.MixedW4A8:
			k.Name = "tcgen05_w4a8_gemm"
		case k.UseShiftDown:
			k.Name = "tcgen05_mxfp8_shift_gemm"
		default:
			k.Name = "tcgen05_mxfp8_gemm"
		}
	}
	if !cIdent(k.Name) {
		return k, fmt.Errorf("tcgen05 mxfp8 gemm: bad kernel name %q", k.Name)
	}
	if k.M == 0 {
		k.M = 128
	}
	if k.N == 0 {
		k.N = 128
	}
	if k.K == 0 {
		k.K = 64
	}
	if k.M != 128 || k.N != 128 || k.K < 32 || k.K > 256 || k.K%32 != 0 {
		return k, fmt.Errorf("tcgen05 mxfp8 gemm %s: invalid shape M=%d N=%d K=%d (want M=128, N=128, K in 32..256 multiple of 32)", k.Name, k.M, k.N, k.K)
	}
	return k, nil
}

// InstDescriptor packs the 32-bit tcgen05.mma instruction descriptor (idesc)
// for kind::mxf8f6f4.block_scale.scale_vec::1X.
func (k TCGen05MXFP8GEMM) InstDescriptor() (uint32, error) {
	k, err := k.norm()
	if err != nil {
		return 0, err
	}
	bType := E4M3
	if k.MixedW4A8 {
		bType = E2M1
	}
	return InstDesc{
		DType:    F32,
		AType:    E4M3,
		BType:    bType,
		M:        uint16(k.M),
		N:        uint16(k.N),
		ScaleVec: ScaleVec1X,
	}.Encode(), nil
}

// Grid returns the single-CTA grid and 128-thread (4-warp) block dimensions.
func (k TCGen05MXFP8GEMM) Grid() (grid, block [3]int) {
	return [3]int{1, 1, 1}, [3]int{128, 1, 1}
}

// TraceEvents returns the formal sm_100a/sm_103a tcgen05 hardware protocol
// event trace for verification with cudatmem.VerifyTrace.
func (k TCGen05MXFP8GEMM) TraceEvents() []cudatmem.TraceEvent {
	events := []cudatmem.TraceEvent{
		{Kind: cudatmem.EventAlloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128},
		{Kind: cudatmem.EventAlloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 128, Columns: 32, Rows: 128},
		{Kind: cudatmem.EventCp, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1, Offset: 128, Columns: 32, Rows: 128},
		{Kind: cudatmem.EventMMA, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128, HasScaleInTMEM: true, ScaleOffset: 128},
		{Kind: cudatmem.EventCommit, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128},
		{Kind: cudatmem.EventWait, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128},
	}
	if k.UseShiftDown {
		events = append(events,
			cudatmem.TraceEvent{Kind: cudatmem.EventShift, CTA: 0, Warp: 0, ActiveThreads: 1, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128},
			cudatmem.TraceEvent{Kind: cudatmem.EventWait, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128},
		)
	}
	for w := range 4 {
		events = append(events, cudatmem.TraceEvent{
			Kind:          cudatmem.EventLd,
			CTA:           0,
			Warp:          w,
			ActiveThreads: 32,
			CTAGroup:      1,
			Offset:        0,
			Columns:       128,
			RowStart:      uint32(32 * w),
			Rows:          32,
		})
	}
	events = append(events,
		cudatmem.TraceEvent{Kind: cudatmem.EventDealloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128},
		cudatmem.TraceEvent{Kind: cudatmem.EventDealloc, CTA: 0, Warp: 0, ActiveThreads: 32, CTAGroup: 1, Offset: 128, Columns: 32, Rows: 128},
	)
	return events
}

// CUDA emits the complete CUDA C++ source (without Prelude) for k.
func (k TCGen05MXFP8GEMM) CUDA() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	idesc, err := k.InstDescriptor()
	if err != nil {
		return "", err
	}
	bRowBytes := k.K
	if k.MixedW4A8 {
		bRowBytes = k.K / 2
	}

	var b strings.Builder
	b.WriteString(TCGen05AdvancedPrelude)
	b.WriteString(`
EG_DEV float eg_ue8m0_to_f32(uint8_t code) {
    if (code == 0xffu) return __uint_as_float(0x7fc00000u);
    if (code == 0u) return __uint_as_float(0x00400000u);
    return __uint_as_float((uint32_t)code << 23);
}

EG_DEV void eg_tcgen05_cp_ue8m0_scales_32x128b(
    float* tmem, uint32_t scale_taddr,
    const uint8_t* smem_a_scales, const uint8_t* smem_b_scales,
    int m_rows, int n_cols, int k_blocks) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !defined(EG_EMU)
    (void)tmem;
    (void)smem_b_scales;
    (void)m_rows;
    (void)n_cols;
    (void)k_blocks;
    const uint64_t desc = eg_wgmma_desc(smem_a_scales, 16, 128);
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                 :: "r"(scale_taddr), "l"(desc) : "memory");
#else
    for (int r = 0; r < m_rows; ++r) {
        for (int kb = 0; kb < k_blocks; ++kb) {
            tmem[r * 512 + scale_taddr + kb] = eg_ue8m0_to_f32(smem_a_scales[r * k_blocks + kb]);
        }
    }
    for (int n = 0; n < n_cols; ++n) {
        for (int kb = 0; kb < k_blocks; ++kb) {
            tmem[n * 512 + scale_taddr + 16 + kb] = eg_ue8m0_to_f32(smem_b_scales[n * k_blocks + kb]);
        }
    }
#endif
}

EG_DEV void eg_tcgen05_shift_down_cta1(float* tmem, uint32_t d_taddr, int m_rows, int n_cols) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !defined(EG_EMU)
    (void)tmem;
    (void)m_rows;
    (void)n_cols;
    asm volatile("tcgen05.shift.cta_group::1.down [%0];" :: "r"(d_taddr) : "memory");
#else
    for (int n = 0; n < n_cols; ++n) {
        for (int r = m_rows - 1; r >= 1; --r) {
            tmem[r * 512 + d_taddr + n] = tmem[(r - 1) * 512 + d_taddr + n];
        }
        tmem[0 * 512 + d_taddr + n] = 0.0f;
    }
#endif
}
`)

	fmt.Fprintf(&b, "// %s: OCP microscaling GEMM (M=%d, N=%d, K=%d, mixed_w4a8=%v, shift_down=%v).\n",
		k.Name, k.M, k.N, k.K, k.MixedW4A8, k.UseShiftDown)
	fmt.Fprintf(&b, `__global__ void __launch_bounds__(128) %s(
    const uint8_t* __restrict__ a_data,
    const uint8_t* __restrict__ a_scales,
    const uint8_t* __restrict__ b_data,
    const uint8_t* __restrict__ b_scales,
    float* __restrict__ out) {
    constexpr int M = %d, N = %d, K = %d;
    constexpr int B_ROW_BYTES = %d, K_BLOCKS = K / 32;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

    EG_TCGEN05_SHARED(uint8_t, smem_a, [M * K]);
    EG_TCGEN05_SHARED(uint8_t, smem_b, [N * B_ROW_BYTES]);
    EG_TCGEN05_SHARED(uint8_t, smem_a_scales, [M * K_BLOCKS]);
    EG_TCGEN05_SHARED(uint8_t, smem_b_scales, [N * K_BLOCKS]);
    EG_TCGEN05_TMEM_BUF(tmem);
    EG_TCGEN05_SHARED(uint32_t, slot_d, [1]);
    EG_TCGEN05_SHARED(uint32_t, slot_s, [1]);
    EG_TCGEN05_SHARED(uint64_t, mbar, [1]);

    // 1. Allocate 128 accumulator columns (offset 0) and 32 scale columns (offset 128).
    if (warp == 0) {
        eg_tcgen05_alloc_cta1(tmem, &slot_d[0], 0u, 128u);
        eg_tcgen05_alloc_cta1(tmem, &slot_s[0], 128u, 32u);
    }
    __syncthreads();
    const uint32_t d_taddr = slot_d[0];
    const uint32_t s_taddr = slot_s[0];

    // 2. Stage operands and 32-element UE8M0 block scales in shared memory.
    for (int i = tid; i < M * K; i += 128) smem_a[i] = a_data[i];
    for (int i = tid; i < N * B_ROW_BYTES; i += 128) smem_b[i] = b_data[i];
    for (int i = tid; i < M * K_BLOCKS; i += 128) smem_a_scales[i] = a_scales[i];
    for (int i = tid; i < N * K_BLOCKS; i += 128) smem_b_scales[i] = b_scales[i];
    __syncthreads();

    // 3. Single elected thread stages UE8M0 scales via tcgen05.cp.cta_group::1.32x128b.warpx4
    //    and issues tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.
    if (warp == 0 && eg_elect_one_sync()) {
        eg_tcgen05_cp_ue8m0_scales_32x128b(tmem, s_taddr, smem_a_scales, smem_b_scales, M, N, K_BLOCKS);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !defined(EG_EMU)
        const uint64_t desc_a = eg_wgmma_desc(smem_a, 32, 256);
        const uint64_t desc_b = eg_wgmma_desc(smem_b, 32, 256);
        const uint32_t sfa_taddr = s_taddr;
        const uint32_t sfb_taddr = s_taddr + 16u;
        const uint32_t idesc = 0x%08xu;
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "setp.ne.b32 p, 1, 0;\n"
            "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X [%%0], %%1, %%2, %%3, [%%4], [%%5], p;\n"
            "}" :: "r"(d_taddr), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(sfa_taddr), "r"(sfb_taddr) : "memory");
#else
`, k.Name, k.M, k.N, k.K, bRowBytes, idesc)

	if k.MixedW4A8 {
		b.WriteString(`        for (int r = 0; r < M; ++r) {
            for (int n = 0; n < N; ++n) {
                float acc = tmem[r * 512 + d_taddr + n];
                for (int kb = 0; kb < K_BLOCKS; ++kb) {
                    const float sa = tmem[r * 512 + s_taddr + kb];
                    const float sb = tmem[n * 512 + s_taddr + 16 + kb];
                    float block_dot = 0.0f;
                    for (int j = 0; j < 16; ++j) {
                        const uint8_t bb = smem_b[n * B_ROW_BYTES + kb * 16 + j];
                        const float a0 = eg_e4m3_to_f32(smem_a[r * K + kb * 32 + 2 * j + 0]);
                        const float a1 = eg_e4m3_to_f32(smem_a[r * K + kb * 32 + 2 * j + 1]);
                        const float b0 = eg_tcgen05_decode_e2m1(bb & 0xfu);
                        const float b1 = eg_tcgen05_decode_e2m1((bb >> 4) & 0xfu);
                        block_dot += a0 * b0 + a1 * b1;
                    }
                    acc += block_dot * (sa * sb);
                }
                tmem[r * 512 + d_taddr + n] = acc;
            }
        }
`)
	} else {
		b.WriteString(`        for (int r = 0; r < M; ++r) {
            for (int n = 0; n < N; ++n) {
                float acc = tmem[r * 512 + d_taddr + n];
                for (int kb = 0; kb < K_BLOCKS; ++kb) {
                    const float sa = tmem[r * 512 + s_taddr + kb];
                    const float sb = tmem[n * 512 + s_taddr + 16 + kb];
                    float block_dot = 0.0f;
                    for (int j = 0; j < 32; ++j) {
                        const float av = eg_e4m3_to_f32(smem_a[r * K + kb * 32 + j]);
                        const float bv = eg_e4m3_to_f32(smem_b[n * B_ROW_BYTES + kb * 32 + j]);
                        block_dot += av * bv;
                    }
                    acc += block_dot * (sa * sb);
                }
                tmem[r * 512 + d_taddr + n] = acc;
            }
        }
`)
	}

	b.WriteString(`#endif
        eg_tcgen05_commit_cta1(&mbar[0]);
    }
    eg_tcgen05_wait_ld();
    __syncthreads();
`)

	if k.UseShiftDown {
		b.WriteString(`
    // 3b. Single elected thread shifts TMEM accumulator rows down via tcgen05.shift.cta_group::1.down.
    if (warp == 0 && eg_elect_one_sync()) {
        eg_tcgen05_shift_down_cta1(tmem, d_taddr, M, N);
    }
    eg_tcgen05_wait_ld();
    __syncthreads();
`)
	}

	b.WriteString(`
    // 4. 4-warp 32-row partitioned tcgen05.ld epilogue.
    const int row = 32 * warp + lane;
    for (int col = 0; col < N; ++col) {
        out[row * N + col] = eg_tcgen05_ld_row(tmem, warp, lane, d_taddr + (uint32_t)col);
    }
    eg_tcgen05_wait_ld();
    __syncthreads();

    // 5. Release TMEM allocations.
    if (warp == 0) {
        eg_tcgen05_dealloc_cta1(d_taddr, 128u);
        eg_tcgen05_dealloc_cta1(s_taddr, 32u);
    }
}
`)
	return b.String(), nil
}

// CUDAFile emits the self-contained CUDA C++ translation unit (including
// Prelude) for k, ready for cudahostemu.Run or nvcc.
func (k TCGen05MXFP8GEMM) CUDAFile() (string, error) {
	src, err := k.CUDA()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src, nil
}

// Reference computes the host reference output [M, N] for k given E4M3
// activations aData [M, K] with UE8M0 scales aScales [M, K/32] and weights
// bData ([N, K] E4M3 when MixedW4A8 is false, or [N, K/2] packed E2M1 when
// MixedW4A8 is true) with UE8M0 scales bScales [N, K/32].
func (k TCGen05MXFP8GEMM) Reference(aData, aScales, bData, bScales []uint8) ([]float32, error) {
	k, err := k.norm()
	if err != nil {
		return nil, err
	}
	kBlocks := k.K / MXBlock
	bRowBytes := k.K
	if k.MixedW4A8 {
		bRowBytes = k.K / 2
	}
	if len(aData) != k.M*k.K {
		return nil, fmt.Errorf("tcgen05 mxfp8 gemm %s reference: len(aData)=%d, want %d", k.Name, len(aData), k.M*k.K)
	}
	if len(aScales) != k.M*kBlocks {
		return nil, fmt.Errorf("tcgen05 mxfp8 gemm %s reference: len(aScales)=%d, want %d", k.Name, len(aScales), k.M*kBlocks)
	}
	if len(bData) != k.N*bRowBytes {
		return nil, fmt.Errorf("tcgen05 mxfp8 gemm %s reference: len(bData)=%d, want %d", k.Name, len(bData), k.N*bRowBytes)
	}
	if len(bScales) != k.N*kBlocks {
		return nil, fmt.Errorf("tcgen05 mxfp8 gemm %s reference: len(bScales)=%d, want %d", k.Name, len(bScales), k.N*kBlocks)
	}

	out := make([]float32, k.M*k.N)
	for r := 0; r < k.M; r++ {
		for n := 0; n < k.N; n++ {
			var acc float32
			for kb := 0; kb < kBlocks; kb++ {
				sa := DecodeUE8M0(aScales[r*kBlocks+kb])
				sb := DecodeUE8M0(bScales[n*kBlocks+kb])
				var blockDot float32
				if k.MixedW4A8 {
					for j := 0; j < 16; j++ {
						bb := bData[n*bRowBytes+kb*16+j]
						a0 := FromE4M3Bits(aData[r*k.K+kb*32+2*j+0])
						a1 := FromE4M3Bits(aData[r*k.K+kb*32+2*j+1])
						b0 := FromE2M1(bb & 0xf)
						b1 := FromE2M1((bb >> 4) & 0xf)
						blockDot += a0*b0 + a1*b1
					}
				} else {
					for j := 0; j < 32; j++ {
						av := FromE4M3Bits(aData[r*k.K+kb*32+j])
						bv := FromE4M3Bits(bData[n*bRowBytes+kb*32+j])
						blockDot += av * bv
					}
				}
				acc += blockDot * (sa * sb)
			}
			out[r*k.N+n] = acc
		}
	}

	if k.UseShiftDown {
		for n := 0; n < k.N; n++ {
			for r := k.M - 1; r >= 1; r-- {
				out[r*k.N+n] = out[(r-1)*k.N+n]
			}
			out[0*k.N+n] = 0
		}
	}
	return out, nil
}
