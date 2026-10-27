package train

import (
	"fmt"
	"math"
	"strings"
)

var e2m1Magnitudes = [8]float32{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}

// DecodeE2M1 decodes a 4-bit float4_e2m1fn code (bits [2:0] magnitude index in
// [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], bit 3 sign) to float32.
func DecodeE2M1(code uint8) float32 {
	v := e2m1Magnitudes[code&7]
	if code&8 != 0 {
		return -v
	}
	return v
}

// EncodeE2M1 quantizes val to a 4-bit float4_e2m1fn code using deterministic
// round-to-nearest-even (ties to lower magnitude index, saturating to ±6.0).
func EncodeE2M1(val float32) uint8 {
	if math.IsNaN(float64(val)) {
		return 0
	}
	var sign uint8
	a := val
	if a < 0 {
		sign = 8
		a = -a
	}
	var mag uint8
	switch {
	case a > 5.0:
		mag = 7
	case a > 3.5:
		mag = 6
	case a > 2.5:
		mag = 5
	case a > 1.75:
		mag = 4
	case a > 1.25:
		mag = 3
	case a > 0.75:
		mag = 2
	case a > 0.25:
		mag = 1
	default:
		return 0
	}
	return sign | mag
}

// StochasticRoundE2M1 performs unbiased hardware-equivalent stochastic rounding
// of val onto the E2M1 grid [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] using 16-bit
// uniform random entropy randBits (u = float32(randBits) / 65536.0), matching
// Blackwell's cvt.rs.satfinite.e2m1x2.f32 semantics.
//
// For a magnitude |val| in [q_lo, q_hi], StochasticRoundE2M1 rounds up to q_hi
// with probability p = (|val| - q_lo) / (q_hi - q_lo) and down to q_lo with
// probability 1 - p, so E[DecodeE2M1(StochasticRoundE2M1(val, r))] = val for all
// val in [-6, 6]. Values outside [-6, 6] saturate to ±6.0.
func StochasticRoundE2M1(val float32, randBits uint16) uint8 {
	if math.IsNaN(float64(val)) {
		return 0
	}
	var sign uint8
	a := val
	if a < 0 {
		sign = 8
		a = -a
	}
	if a <= 0 {
		return 0
	}
	if a >= 6.0 {
		return sign | 7
	}
	idx := 0
	switch {
	case a >= 4.0:
		idx = 6
	case a >= 3.0:
		idx = 5
	case a >= 2.0:
		idx = 4
	case a >= 1.5:
		idx = 3
	case a >= 1.0:
		idx = 2
	case a >= 0.5:
		idx = 1
	}
	qLo := e2m1Magnitudes[idx]
	qHi := e2m1Magnitudes[idx+1]
	prob := (a - qLo) / (qHi - qLo)
	u := float32(randBits) * (1.0 / 65536.0)
	mag := uint8(idx)
	if u < prob {
		mag = uint8(idx + 1)
	}
	if mag == 0 {
		return 0
	}
	return sign | mag
}

// E4M3Bits encodes f as float8_e4m3fn, rounding to nearest even and saturating
// finite overflow to ±448.
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

// FromE4M3Bits decodes a float8_e4m3fn byte to float32.
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

// EncodeUE4M3 encodes a non-negative NVFP4 block scale s = amax / 6.0 into a
// positive unsigned-E4M3 byte, clamping underflow/zero to 0x01 (2^-9 = 1/512)
// so downstream division by the decoded scale never divides by zero.
func EncodeUE4M3(s float32) uint8 {
	if !(s > 0) {
		return 1
	}
	b := E4M3Bits(s) & 0x7f
	if b == 0 {
		return 1
	}
	if b == 0x7f {
		return 0x7e
	}
	return b
}

// DecodeUE4M3 decodes an unsigned-E4M3 block scale byte to a positive float32.
func DecodeUE4M3(b uint8) float32 {
	s := FromE4M3Bits(b & 0x7f)
	if !(s > 0) {
		return 1.0 / 512.0
	}
	return s
}

func splitMix32(x uint32) uint32 {
	x += 0x9e3779b9
	x = (x ^ (x >> 16)) * 0x85ebca6b
	x = (x ^ (x >> 13)) * 0xc2b2ae35
	return x ^ (x >> 16)
}

// StochasticRoundNVFP4 quantizes a flat slice of FP32 values (whose length is a
// positive multiple of 16) into block-scaled NVFP4 (packed E2M1 nibbles and
// per-16-element UE4M3 block scales) using SplitMix32 stochastic rounding seeded
// by seed.
func StochasticRoundNVFP4(vals []float32, seed uint32) (packed, scales []uint8, err error) {
	if len(vals) == 0 || len(vals)%16 != 0 {
		return nil, nil, fmt.Errorf("stochastic round nvfp4: length %d must be a positive multiple of 16", len(vals))
	}
	numBlocks := len(vals) / 16
	packed = make([]uint8, len(vals)/2)
	scales = make([]uint8, numBlocks)
	for b := range numBlocks {
		base := b * 16
		var amax float32
		for t := range 16 {
			if a := float32(math.Abs(float64(vals[base+t]))); a > amax {
				amax = a
			}
		}
		scaleU8 := EncodeUE4M3(amax * (1.0 / 6.0))
		scales[b] = scaleU8
		invS := 1.0 / DecodeUE4M3(scaleU8)
		for i := range 8 {
			pairIdx := uint32(b*8 + i)
			rbits := splitMix32(seed ^ pairIdx)
			q0 := StochasticRoundE2M1(vals[base+2*i]*invS, uint16(rbits))
			q1 := StochasticRoundE2M1(vals[base+2*i+1]*invS, uint16(rbits>>16))
			packed[b*8+i] = q0 | (q1 << 4)
		}
	}
	return packed, scales, nil
}

// QuantizeGradDualAxisNVFP4 quantizes an upstream gradient matrix dY of shape
// [M, N] (where M and N are positive multiples of 16) along both reduction axes
// using Blackwell stochastic rounding seeded by seed:
//   - Row-wise [M, N/2] packed E2M1 codes and [M, N/16] UE4M3 scales for dGrad
//     (dX = dY * W, reducing along N).
//   - Transposed column-wise [N, M/2] packed E2M1 codes and [N, M/16] UE4M3
//     scales for wGrad (dW = dY^T * X, reducing along M).
func QuantizeGradDualAxisNVFP4(dY []float32, M, N int, seed uint32) (rowPacked, rowScales, colPacked, colScales []uint8, err error) {
	return quantizeGradDualAxisNVFP4DType(dY, M, N, FP32, seed)
}

func quantizeGradDualAxisNVFP4DType(dY []float32, M, N int, dt DType, seed uint32) (rowPacked, rowScales, colPacked, colScales []uint8, err error) {
	if M < 16 || M%16 != 0 {
		return nil, nil, nil, nil, fmt.Errorf("quantize grad dual axis nvfp4: M %d must be a positive multiple of 16", M)
	}
	if N < 16 || N%16 != 0 {
		return nil, nil, nil, nil, fmt.Errorf("quantize grad dual axis nvfp4: N %d must be a positive multiple of 16", N)
	}
	if len(dY) != M*N {
		return nil, nil, nil, nil, fmt.Errorf("quantize grad dual axis nvfp4: len(dY) = %d, want %d (%dx%d)", len(dY), M*N, M, N)
	}
	roundIn := func(v float32) float32 {
		switch dt {
		case BF16:
			return FromBF16Bits(BF16Bits(v))
		case FP16:
			return FromF16Bits(F16Bits(v))
		default:
			return v
		}
	}

	nBytes := N / 2
	nBlocks := N / 16
	mBytes := M / 2
	mBlocks := M / 16

	rowPacked = make([]uint8, M*nBytes)
	rowScales = make([]uint8, M*nBlocks)
	colPacked = make([]uint8, N*mBytes)
	colScales = make([]uint8, N*mBlocks)

	// Row-wise quantization [M, N] for dGrad (reduces along N).
	for m := range M {
		for nb := range nBlocks {
			base := m*N + nb*16
			var vals [16]float32
			var amax float32
			for t := range 16 {
				v := roundIn(dY[base+t])
				vals[t] = v
				if a := float32(math.Abs(float64(v))); a > amax {
					amax = a
				}
			}
			scaleU8 := EncodeUE4M3(amax * (1.0 / 6.0))
			rowScales[m*nBlocks+nb] = scaleU8
			invS := 1.0 / DecodeUE4M3(scaleU8)
			for i := range 8 {
				pairIdx := uint32(m*nBytes + nb*8 + i)
				rbits := splitMix32(seed ^ pairIdx)
				q0 := StochasticRoundE2M1(vals[2*i]*invS, uint16(rbits))
				q1 := StochasticRoundE2M1(vals[2*i+1]*invS, uint16(rbits>>16))
				rowPacked[m*nBytes+nb*8+i] = q0 | (q1 << 4)
			}
		}
	}

	// Transposed column-wise quantization [N, M] for wGrad (reduces along M).
	colSeed := seed ^ 0x6a09e667
	for n := range N {
		for mb := range mBlocks {
			var vals [16]float32
			var amax float32
			for t := range 16 {
				v := roundIn(dY[(mb*16+t)*N+n])
				vals[t] = v
				if a := float32(math.Abs(float64(v))); a > amax {
					amax = a
				}
			}
			scaleU8 := EncodeUE4M3(amax * (1.0 / 6.0))
			colScales[n*mBlocks+mb] = scaleU8
			invS := 1.0 / DecodeUE4M3(scaleU8)
			for i := range 8 {
				pairIdx := uint32(n*mBytes + mb*8 + i)
				rbits := splitMix32(colSeed ^ pairIdx)
				q0 := StochasticRoundE2M1(vals[2*i]*invS, uint16(rbits))
				q1 := StochasticRoundE2M1(vals[2*i+1]*invS, uint16(rbits>>16))
				colPacked[n*mBytes+mb*8+i] = q0 | (q1 << 4)
			}
		}
	}

	return rowPacked, rowScales, colPacked, colScales, nil
}

const blackwellBwdPrelude = `#ifndef EG_BLACKWELL_BWD_PRELUDE
#define EG_BLACKWELL_BWD_PRELUDE

EG_DEV uint32_t eg_splitmix32(uint32_t x) {
    x += 0x9e3779b9u;
    x = (x ^ (x >> 16)) * 0x85ebca6bu;
    x = (x ^ (x >> 13)) * 0xc2b2ae35u;
    return x ^ (x >> 16);
}

EG_DEV uint8_t eg_f32_to_ue4m3(float s) {
    if (!(s > 0.0f)) return 1u;
    uint8_t b = (uint8_t)(eg_f32x2_to_e4m3x2(s, 0.0f) & 0x7fu);
    if (b == 0u) return 1u;
    if (b == 0x7fu) return 0x7eu;
    return b;
}

EG_DEV float eg_ue4m3_to_f32(uint8_t b) {
    float s = eg_e4m3_to_f32((uint32_t)(b & 0x7fu));
    return (s > 0.0f) ? s : (1.0f / 512.0f);
}

EG_DEV uint8_t eg_f32_to_e2m1_rn(float x) {
    if (!(x == x)) return 0u;
    uint8_t sign = 0u;
    if (x < 0.0f) {
        sign = 8u;
        x = -x;
    }
    uint8_t mag = 0u;
    if (x > 5.0f) mag = 7u;
    else if (x > 3.5f) mag = 6u;
    else if (x > 2.5f) mag = 5u;
    else if (x > 1.75f) mag = 4u;
    else if (x > 1.25f) mag = 3u;
    else if (x > 0.75f) mag = 2u;
    else if (x > 0.25f) mag = 1u;
    else return 0u;
    return sign | mag;
}

EG_DEV uint8_t eg_stochastic_round_e2m1(float val, uint16_t rand_bits) {
    if (!(val == val)) return 0u;
    uint8_t sign = 0u;
    float a = val;
    if (a < 0.0f) {
        sign = 8u;
        a = -a;
    }
    if (a <= 0.0f) return 0u;
    if (a >= 6.0f) return sign | 7u;
    const float g0 = 0.0f, g1 = 0.5f, g2 = 1.0f, g3 = 1.5f, g4 = 2.0f, g5 = 3.0f, g6 = 4.0f, g7 = 6.0f;
    int idx = 0;
    float q_lo = g0, q_hi = g1;
    if (a >= g6) { idx = 6; q_lo = g6; q_hi = g7; }
    else if (a >= g5) { idx = 5; q_lo = g5; q_hi = g6; }
    else if (a >= g4) { idx = 4; q_lo = g4; q_hi = g5; }
    else if (a >= g3) { idx = 3; q_lo = g3; q_hi = g4; }
    else if (a >= g2) { idx = 2; q_lo = g2; q_hi = g3; }
    else if (a >= g1) { idx = 1; q_lo = g1; q_hi = g2; }
    const float prob = (a - q_lo) / (q_hi - q_lo);
    const float u = (float)rand_bits * (1.0f / 65536.0f);
    const uint8_t mag = (uint8_t)((u < prob) ? (idx + 1) : idx);
    return (mag == 0u) ? 0u : (uint8_t)(sign | mag);
}

// eg_cvt_rs_e2m1x2_f32 stochastically rounds (v0, v1) to two packed E2M1 codes
// using 32-bit entropy rbits (cvt.rs.satfinite.e2m1x2.f32 on sm_100a+).
EG_DEV uint8_t eg_cvt_rs_e2m1x2_f32(float v0, float v1, uint32_t rbits) {
#if !defined(EG_HOST_EMU) && !defined(EG_EMU) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    uint16_t r;
    asm("{\n.reg .b8 b;\ncvt.rs.satfinite.e2m1x2.f32 b, %1, %2, %3;\ncvt.u16.u8 %0, b;\n}"
        : "=h"(r) : "f"(v1), "f"(v0), "r"(rbits));
    return (uint8_t)(r & 0xffu);
#else
    const uint8_t q0 = eg_stochastic_round_e2m1(v0, (uint16_t)(rbits & 0xffffu));
    const uint8_t q1 = eg_stochastic_round_e2m1(v1, (uint16_t)(rbits >> 16));
    return (uint8_t)(q0 | (q1 << 4));
#endif
}

// eg_cvt_rs_e4m3x2_f32 stochastically rounds (v0, v1) to two packed E4M3 bytes
// using 32-bit entropy rbits (cvt.rs.satfinite.e4m3x2.f32 on sm_100a+).
EG_DEV uint16_t eg_cvt_rs_e4m3x2_f32(float v0, float v1, uint32_t rbits) {
#if !defined(EG_HOST_EMU) && !defined(EG_EMU) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    uint16_t r;
    asm("cvt.rs.satfinite.e4m3x2.f32 %0, %1, %2, %3;"
        : "=h"(r) : "f"(v1), "f"(v0), "r"(rbits));
    return r;
#else
    (void)rbits;
    return eg_f32x2_to_e4m3x2(v0, v1);
#endif
}

EG_DEV float eg_decode_e2m1(uint8_t code) {
    static constexpr float grid[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = grid[code & 7u];
    return (code & 8u) ? -v : v;
}

#if defined(EG_HOST_EMU) && !defined(EG_EMU)
inline float* eg_bwd_host_tmem() {
    return static_cast<float*>(eg_emu_cur_block->get("eg_bwd_tmem", 128 * 512 * sizeof(float), 0));
}
#endif

EG_DEV void eg_bwd_tmem_alloc(uint32_t* smem_addr_out, uint32_t num_cols) {
#if defined(EG_EMU)
    if (threadIdx.x == 0) eg_emu_tmem_alloc(smem_addr_out, num_cols);
#elif defined(EG_HOST_EMU)
    (void)num_cols;
    if (threadIdx.x == 0) *smem_addr_out = 0u;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    if ((threadIdx.x >> 5) == 0) {
        const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem_addr_out);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     :: "r"(s), "r"(num_cols) : "memory");
    }
#endif
}

EG_DEV void eg_bwd_tmem_dealloc(uint32_t tmem_addr, uint32_t num_cols) {
#if defined(EG_EMU)
    if (threadIdx.x == 0) eg_emu_tmem_dealloc(tmem_addr, num_cols);
#elif defined(EG_HOST_EMU)
    (void)tmem_addr;
    (void)num_cols;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    if ((threadIdx.x >> 5) == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :: "r"(tmem_addr), "r"(num_cols) : "memory");
    }
#endif
}

EG_DEV void eg_bwd_tcgen05_mma_f32(
    uint32_t tmem_d,
    const float* smem_a,
    const float* smem_b,
    int M, int N, int K, int accum_enable) {
#if defined(EG_EMU)
    if (threadIdx.x == 0) {
        eg_emu_tmem_cta* cta = eg_emu_get_tmem_cta();
        std::lock_guard<std::mutex> lock(cta->mu);
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                const uint32_t col = tmem_d + (uint32_t)n;
                float acc = accum_enable ? cta->staged[m][col] : 0.0f;
                for (int k = 0; k < K; ++k) {
                    acc += smem_a[m * K + k] * smem_b[n * K + k];
                }
                cta->staged[m][col] = acc;
                cta->live_bits[m][col] = eg_emu_tmem_nan_bits;
                cta->col_state[col] = eg_emu_tmem_col_uncommitted;
            }
        }
    }
#elif defined(EG_HOST_EMU)
    if (threadIdx.x == 0) {
        float* tmem = eg_bwd_host_tmem();
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                const uint32_t col = tmem_d + (uint32_t)n;
                float acc = accum_enable ? tmem[m * 512 + col] : 0.0f;
                for (int k = 0; k < K; ++k) {
                    acc += smem_a[m * K + k] * smem_b[n * K + k];
                }
                tmem[m * 512 + col] = acc;
            }
        }
    }
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    if (threadIdx.x == 0) {
        const uint64_t da = eg_wgmma_desc(smem_a, (uint32_t)(K * 4), (uint32_t)(8 * K * 4));
        const uint64_t db = eg_wgmma_desc(smem_b, (uint32_t)(K * 4), (uint32_t)(8 * K * 4));
        const uint32_t idesc = ((uint32_t)M << 24) | ((uint32_t)N << 16) | (uint32_t)K;
        asm volatile(
            "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
            "tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, p;\n}\n"
            :: "r"(tmem_d), "l"(da), "l"(db), "r"(idesc), "r"(accum_enable) : "memory");
    }
#endif
}

EG_DEV void eg_bwd_tcgen05_mma_nvfp4(
    uint32_t tmem_d,
    const uint8_t* smem_a_fp4, const uint8_t* smem_a_scale,
    const uint8_t* smem_b_fp4, const uint8_t* smem_b_scale,
    int M, int N, int K, int accum_enable) {
#if defined(EG_EMU)
    if (threadIdx.x == 0) {
        eg_emu_tcgen05_mma_nvfp4(tmem_d, smem_a_fp4, smem_a_scale, smem_b_fp4, smem_b_scale, M, N, K, accum_enable);
    }
#elif defined(EG_HOST_EMU)
    if (threadIdx.x == 0) {
        float* tmem = eg_bwd_host_tmem();
        const int k_packed = K / 2;
        const int k_scales = K / 16;
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                const uint32_t col = tmem_d + (uint32_t)n;
                float acc = accum_enable ? tmem[m * 512 + col] : 0.0f;
                for (int k = 0; k < K; ++k) {
                    const uint8_t a_byte = smem_a_fp4[m * k_packed + (k >> 1)];
                    const uint8_t b_byte = smem_b_fp4[n * k_packed + (k >> 1)];
                    const uint8_t a_code = (a_byte >> (4 * (k & 1))) & 0xfu;
                    const uint8_t b_code = (b_byte >> (4 * (k & 1))) & 0xfu;
                    const float a_sf = smem_a_scale ? eg_ue4m3_to_f32(smem_a_scale[m * k_scales + (k >> 4)]) : 1.0f;
                    const float b_sf = smem_b_scale ? eg_ue4m3_to_f32(smem_b_scale[n * k_scales + (k >> 4)]) : 1.0f;
                    acc += (eg_decode_e2m1(a_code) * a_sf) * (eg_decode_e2m1(b_code) * b_sf);
                }
                tmem[m * 512 + col] = acc;
            }
        }
    }
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    if (threadIdx.x == 0) {
        const uint64_t da = eg_wgmma_desc(smem_a_fp4, (uint32_t)(K / 2), (uint32_t)(8 * (K / 2)));
        const uint64_t db = eg_wgmma_desc(smem_b_fp4, (uint32_t)(K / 2), (uint32_t)(8 * (K / 2)));
        const uint32_t sfa = (uint32_t)__cvta_generic_to_shared(smem_a_scale);
        const uint32_t sfb = (uint32_t)__cvta_generic_to_shared(smem_b_scale);
        const uint32_t idesc = ((uint32_t)M << 24) | ((uint32_t)N << 16) | (uint32_t)K;
        asm volatile(
            "{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%4], [%5], p;\n}\n"
            :: "r"(tmem_d), "l"(da), "l"(db), "r"(idesc), "r"(sfa), "r"(sfb), "r"(accum_enable) : "memory");
    }
#endif
}

EG_DEV void eg_bwd_tcgen05_commit(uint64_t* mbar) {
#if defined(EG_EMU)
    (void)mbar;
    if (threadIdx.x == 0) eg_emu_tcgen05_commit();
#elif defined(EG_HOST_EMU)
    (void)mbar;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    if (threadIdx.x == 0) {
        const uint32_t s = (uint32_t)__cvta_generic_to_shared(mbar);
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                     :: "r"(s) : "memory");
    }
#endif
}

EG_DEV void eg_bwd_tcgen05_wait() {
#if defined(EG_EMU)
    eg_emu_tcgen05_wait();
#elif defined(EG_HOST_EMU)
    // Synchronized by caller __syncthreads().
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#endif
}

EG_DEV void eg_bwd_tcgen05_ld_f32(float* reg_out, uint32_t tmem_addr, int row, int col_offset, int count) {
#if defined(EG_EMU)
    eg_emu_tcgen05_ld_f32(reg_out, tmem_addr, row, col_offset, count);
#elif defined(EG_HOST_EMU)
    const float* tmem = eg_bwd_host_tmem();
    const uint32_t start_col = tmem_addr + (uint32_t)col_offset;
    for (int i = 0; i < count; ++i) {
        reg_out[i] = tmem[row * 512 + start_col + (uint32_t)i];
    }
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

#endif // EG_BLACKWELL_BWD_PRELUDE
`

// DualAxisGradQuantizer emits a fused Blackwell CUDA C++ kernel that quantizes
// an upstream gradient matrix dY [M, N] into NVFP4 along both reduction axes
// in a single shared-memory-staged pass using hardware stochastic rounding
// (cvt.rs.satfinite.e2m1x2.f32):
//   - Row-quantized dY_row_packed [M, N/2] and dY_row_scales [M, N/16] for dGrad
//     (dX = dY * W, reducing along N).
//   - Transposed column-quantized dY_col_packed [N, M/2] and dY_col_scales [N, M/16]
//     for wGrad (dW = dY^T * X, reducing along M).
type DualAxisGradQuantizer struct {
	Name  string
	M, N  int   // rows (tokens/batch) and columns (out features); multiples of 16
	DType DType // input dtype: FP32 (default), BF16, or FP16
	Seed  uint32
}

func (k DualAxisGradQuantizer) norm() (DualAxisGradQuantizer, string, error) {
	if k.Name == "" {
		k.Name = "dual_axis_grad_quant_nvfp4"
	}
	if !cIdent(k.Name) {
		return k, "", fmt.Errorf("dual axis grad quantizer: bad name %q", k.Name)
	}
	if k.M < 16 || k.M%16 != 0 {
		return k, "", fmt.Errorf("dual axis grad quantizer %s: M %d must be a positive multiple of 16", k.Name, k.M)
	}
	if k.N < 16 || k.N%16 != 0 {
		return k, "", fmt.Errorf("dual axis grad quantizer %s: N %d must be a positive multiple of 16", k.Name, k.N)
	}
	if k.DType == 0 {
		k.DType = FP32
	}
	ct, err := bwdCType(k.DType)
	if err != nil {
		return k, "", fmt.Errorf("dual axis grad quantizer %s: %w", k.Name, err)
	}
	return k, ct, nil
}

func (k DualAxisGradQuantizer) params(ct string) []string {
	return []string{
		"uint8_t* __restrict__ dY_row_packed",
		"uint8_t* __restrict__ dY_row_scales",
		"uint8_t* __restrict__ dY_col_packed",
		"uint8_t* __restrict__ dY_col_scales",
		"const " + ct + "* __restrict__ dY",
		"uint32_t seed",
	}
}

// Grid returns the launch grid and block dimensions for k.
func (k DualAxisGradQuantizer) Grid() (grid, block [3]int) {
	k, _, err := k.norm()
	if err != nil {
		return
	}
	return [3]int{k.N / 16, k.M / 16, 1}, [3]int{32, 1, 1}
}

// CUDA returns the device kernel source (with [backwardPrelude] and
// [blackwellBwdPrelude], without [Prelude]).
func (k DualAxisGradQuantizer) CUDA() (string, error) {
	k, ct, err := k.norm()
	if err != nil {
		return "", err
	}
	var b strings.Builder
	b.WriteString(backwardPrelude)
	b.WriteString("\n")
	b.WriteString(blackwellBwdPrelude)
	b.WriteString("\n")
	fmt.Fprintf(&b, "// %s: fused dual-axis NVFP4 stochastic-rounding gradient quantizer (%dx%d, DType=%v).\n",
		k.Name, k.M, k.N, k.DType)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(32) %s(\n    %s) {\n",
		k.Name, strings.Join(k.params(ct), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int M = %d, N = %d;\n", k.M, k.N)
	b.WriteString(`    constexpr int N_BYTES = N / 2, N_BLOCKS = N / 16;
    constexpr int M_BYTES = M / 2, M_BLOCKS = M / 16;
    const int nb = blockIdx.x;
    const int mb = blockIdx.y;
    const int tid = threadIdx.x;
    if (mb >= M_BLOCKS || nb >= N_BLOCKS) return;

    __shared__ float s_tile[16][17];
    for (int idx = tid; idx < 256; idx += 32) {
        const int r = idx >> 4;
        const int c = idx & 15;
        s_tile[r][c] = eg_to_f32(dY[(size_t)(mb * 16 + r) * N + (nb * 16 + c)]);
    }
    __syncthreads();

    // Warp half 0 (tid 0..15): row-wise quantization of row m = mb*16 + tid along N.
    if (tid < 16) {
        const int r = tid;
        const int m = mb * 16 + r;
        float amax = 0.0f;
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            amax = fmaxf(amax, fabsf(s_tile[r][c]));
        }
        const uint8_t scale_u8 = eg_f32_to_ue4m3(amax * (1.0f / 6.0f));
        dY_row_scales[(size_t)m * N_BLOCKS + nb] = scale_u8;
        const float inv_s = 1.0f / eg_ue4m3_to_f32(scale_u8);
        uint8_t* const dst = dY_row_packed + (size_t)m * N_BYTES + nb * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t pair_idx = (uint32_t)(m * N_BYTES + nb * 8 + i);
            const uint32_t rbits = eg_splitmix32(seed ^ pair_idx);
            dst[i] = eg_cvt_rs_e2m1x2_f32(s_tile[r][2 * i] * inv_s, s_tile[r][2 * i + 1] * inv_s, rbits);
        }
    } else {
        // Warp half 1 (tid 16..31): transposed column-wise quantization of col n = nb*16 + (tid-16) along M.
        const int c = tid - 16;
        const int n = nb * 16 + c;
        float amax = 0.0f;
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            amax = fmaxf(amax, fabsf(s_tile[r][c]));
        }
        const uint8_t scale_u8 = eg_f32_to_ue4m3(amax * (1.0f / 6.0f));
        dY_col_scales[(size_t)n * M_BLOCKS + mb] = scale_u8;
        const float inv_s = 1.0f / eg_ue4m3_to_f32(scale_u8);
        uint8_t* const dst = dY_col_packed + (size_t)n * M_BYTES + mb * 8;
        const uint32_t col_seed = seed ^ 0x6a09e667u;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t pair_idx = (uint32_t)(n * M_BYTES + mb * 8 + i);
            const uint32_t rbits = eg_splitmix32(col_seed ^ pair_idx);
            dst[i] = eg_cvt_rs_e2m1x2_f32(s_tile[2 * i][c] * inv_s, s_tile[2 * i + 1][c] * inv_s, rbits);
        }
    }
}
`)
	return b.String(), nil
}

// Launcher returns the host launch wrapper for k:
//
//	void Name_launch(dY_row_packed, dY_row_scales, dY_col_packed, dY_col_scales, dY, seed, stream)
func (k DualAxisGradQuantizer) Launcher() (string, error) {
	k, ct, err := k.norm()
	if err != nil {
		return "", err
	}
	return launcher(k.Name, k.params(ct), func(b *strings.Builder) {
		fmt.Fprintf(b, "    const dim3 grid(%d, %d, 1), block(32, 1, 1);\n", k.N/16, k.M/16)
	}), nil
}

// Source returns the complete CUDA C++ translation unit ([Prelude], kernel, and launcher).
func (k DualAxisGradQuantizer) Source() (string, error) {
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

// Emit is an alias for [DualAxisGradQuantizer.Source].
func (k DualAxisGradQuantizer) Emit() (string, error) {
	return k.Source()
}

// Reference executes the exact CPU reference for [DualAxisGradQuantizer],
// returning both the row-wise quantized buffers (rowPacked [M, N/2], rowScales [M, N/16])
// and the transposed column-wise quantized buffers (colPacked [N, M/2], colScales [N, M/16]).
func (k DualAxisGradQuantizer) Reference(dY []float32) (rowPacked, rowScales, colPacked, colScales []uint8, err error) {
	k, _, err = k.norm()
	if err != nil {
		return nil, nil, nil, nil, err
	}
	return quantizeGradDualAxisNVFP4DType(dY, k.M, k.N, k.DType, k.Seed)
}

// TCGen05LinearBackward emits a Blackwell sm_100a/sm_103a Tensor Memory (TMEM)
// linear backward kernel computing both activation gradients (dGrad) and weight
// gradients (wGrad):
//
//	dX[m, k] = sum_{n=0}^{N-1} dY[m, n] * W[n, k]    (shape [M, K], dX = dY * W)
//	dW[n, k] = sum_{m=0}^{M-1} dY[m, n] * X[m, k]    (shape [N, K], dW = dY^T * X)
//
// When QuantizedBlockScaled is true, dY is stochastically rounded along both N
// (for dGrad) and M (for wGrad) to NVFP4 with UE4M3 block-16 scales via
// cvt.rs.satfinite.e2m1x2.f32 and accumulated in TMEM via
// tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale. When false, dX and dW
// are accumulated in TMEM in FP32.
type TCGen05LinearBackward struct {
	Name                 string
	M, N, K              int  // M batch/tokens, N out features, K in features (multiples of 16)
	QuantizedBlockScaled bool // true: stochastic-rounded NVFP4 + UE4M3 block-scaled tcgen05.mma; false: FP32 TMEM MMA
	Seed                 uint32
}

func (g TCGen05LinearBackward) norm() (TCGen05LinearBackward, error) {
	if g.Name == "" {
		g.Name = "tcgen05_linear_bwd"
	}
	if !cIdent(g.Name) {
		return g, fmt.Errorf("tcgen05 linear backward: bad name %q", g.Name)
	}
	if g.M < 16 || g.M%16 != 0 {
		return g, fmt.Errorf("tcgen05 linear backward %s: M %d must be a positive multiple of 16", g.Name, g.M)
	}
	if g.N < 16 || g.N%16 != 0 {
		return g, fmt.Errorf("tcgen05 linear backward %s: N %d must be a positive multiple of 16", g.Name, g.N)
	}
	if g.K < 16 || g.K%16 != 0 {
		return g, fmt.Errorf("tcgen05 linear backward %s: K %d must be a positive multiple of 16", g.Name, g.K)
	}
	return g, nil
}

func (g TCGen05LinearBackward) params() []string {
	return []string{
		"float* __restrict__ dX",
		"float* __restrict__ dW",
		"const float* __restrict__ X",
		"const float* __restrict__ W",
		"const float* __restrict__ dY",
	}
}

// Grid returns the launch grid and block dimensions for g.
// The y-grid spans M/16 CTAs for dGrad followed by N/16 CTAs for wGrad.
func (g TCGen05LinearBackward) Grid() (grid, block [3]int) {
	g, err := g.norm()
	if err != nil {
		return
	}
	return [3]int{g.K / 16, g.M/16 + g.N/16, 1}, [3]int{32, 1, 1}
}

// CUDA returns the kernel source without [Prelude].
func (g TCGen05LinearBackward) CUDA() (string, error) {
	g, err := g.norm()
	if err != nil {
		return "", err
	}
	var b strings.Builder
	b.WriteString(backwardPrelude)
	b.WriteString("\n")
	b.WriteString(blackwellBwdPrelude)
	b.WriteString("\n")
	fmt.Fprintf(&b, "// %s: Blackwell tcgen05 linear backward dGrad + wGrad (M=%d, N=%d, K=%d, QuantizedBlockScaled=%v).\n",
		g.Name, g.M, g.N, g.K, g.QuantizedBlockScaled)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(32) %s(\n    %s) {\n",
		g.Name, strings.Join(g.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int M = %d, N = %d, K = %d;\n", g.M, g.N, g.K)
	fmt.Fprintf(&b, "    constexpr uint32_t SEED = %du;\n", g.Seed)
	b.WriteString(`    constexpr int M_TILES = M / 16, N_TILES = N / 16, K_TILES = K / 16;
    const int kb = blockIdx.x;
    const int tile_y = blockIdx.y;
    const int tid = threadIdx.x;
    if (kb >= K_TILES || tile_y >= M_TILES + N_TILES) return;

    __shared__ uint32_t s_tmem_addr;
    __shared__ uint64_t s_tmem_mbar;
    eg_bwd_tmem_alloc(&s_tmem_addr, 32u);
    __syncthreads();
    const uint32_t tmem_d = s_tmem_addr;
`)

	if !g.QuantizedBlockScaled {
		b.WriteString(`    __shared__ float s_a[16 * 16];
    __shared__ float s_b[16 * 16];
    if (tile_y < M_TILES) {
        // dGrad tile: dX[m0..m0+15, k0..k0+15] = sum_nb dY[m0..m0+15, nb*16..nb*16+15] * W[nb*16..nb*16+15, k0..k0+15]
        const int mb = tile_y;
        for (int nb = 0; nb < N_TILES; ++nb) {
            for (int idx = tid; idx < 256; idx += 32) {
                const int r = idx >> 4;
                const int c = idx & 15;
                s_a[r * 16 + c] = dY[(size_t)(mb * 16 + r) * N + (nb * 16 + c)];
                s_b[r * 16 + c] = W[(size_t)(nb * 16 + c) * K + (kb * 16 + r)];
            }
            eg_fence_proxy_async();
            __syncthreads();
            eg_bwd_tcgen05_mma_f32(tmem_d, s_a, s_b, 16, 16, 16, nb > 0 ? 1 : 0);
            __syncthreads();
        }
        eg_bwd_tcgen05_commit(&s_tmem_mbar);
        __syncthreads();
        eg_bwd_tcgen05_wait();
        __syncthreads();
        if (tid < 16) {
            float acc[16];
            eg_bwd_tcgen05_ld_f32(acc, tmem_d, tid, 0, 16);
            float* const dst = dX + (size_t)(mb * 16 + tid) * K + kb * 16;
            for (int c = 0; c < 16; ++c) dst[c] = acc[c];
        }
    } else {
        // wGrad tile: dW[n0..n0+15, k0..k0+15] = sum_mb dY^T[n0..n0+15, mb*16..mb*16+15] * X[mb*16..mb*16+15, k0..k0+15]
        const int nb = tile_y - M_TILES;
        for (int mb = 0; mb < M_TILES; ++mb) {
            for (int idx = tid; idx < 256; idx += 32) {
                const int r = idx >> 4;
                const int c = idx & 15;
                s_a[r * 16 + c] = dY[(size_t)(mb * 16 + c) * N + (nb * 16 + r)];
                s_b[r * 16 + c] = X[(size_t)(mb * 16 + c) * K + (kb * 16 + r)];
            }
            eg_fence_proxy_async();
            __syncthreads();
            eg_bwd_tcgen05_mma_f32(tmem_d, s_a, s_b, 16, 16, 16, mb > 0 ? 1 : 0);
            __syncthreads();
        }
        eg_bwd_tcgen05_commit(&s_tmem_mbar);
        __syncthreads();
        eg_bwd_tcgen05_wait();
        __syncthreads();
        if (tid < 16) {
            float acc[16];
            eg_bwd_tcgen05_ld_f32(acc, tmem_d, tid, 0, 16);
            float* const dst = dW + (size_t)(nb * 16 + tid) * K + kb * 16;
            for (int c = 0; c < 16; ++c) dst[c] = acc[c];
        }
    }
`)
	} else {
		b.WriteString(`    __shared__ float s_raw_a[16][17];
    __shared__ float s_raw_b[16][17];
    __shared__ uint8_t s_a_fp4[16 * 8];
    __shared__ uint8_t s_a_sf[16];
    __shared__ uint8_t s_b_fp4[16 * 8];
    __shared__ uint8_t s_b_sf[16];
    constexpr int N_BYTES = N / 2, M_BYTES = M / 2;

    if (tile_y < M_TILES) {
        // dGrad tile with row-wise stochastic-rounded NVFP4 dY and transposed NVFP4 W.
        const int mb = tile_y;
        for (int nb = 0; nb < N_TILES; ++nb) {
            for (int idx = tid; idx < 256; idx += 32) {
                const int r = idx >> 4;
                const int c = idx & 15;
                s_raw_a[r][c] = dY[(size_t)(mb * 16 + r) * N + (nb * 16 + c)];
                s_raw_b[r][c] = W[(size_t)(nb * 16 + c) * K + (kb * 16 + r)];
            }
            __syncthreads();
            if (tid < 16) {
                const int r = tid;
                const int m = mb * 16 + r;
                float amax_a = 0.0f;
                for (int c = 0; c < 16; ++c) amax_a = fmaxf(amax_a, fabsf(s_raw_a[r][c]));
                const uint8_t sf_a = eg_f32_to_ue4m3(amax_a * (1.0f / 6.0f));
                s_a_sf[r] = sf_a;
                const float inv_sa = 1.0f / eg_ue4m3_to_f32(sf_a);
                for (int i = 0; i < 8; ++i) {
                    const uint32_t pair_idx = (uint32_t)(m * N_BYTES + nb * 8 + i);
                    const uint32_t rbits = eg_splitmix32(SEED ^ pair_idx);
                    s_a_fp4[r * 8 + i] = eg_cvt_rs_e2m1x2_f32(
                        s_raw_a[r][2 * i] * inv_sa, s_raw_a[r][2 * i + 1] * inv_sa, rbits);
                }
            } else {
                const int r = tid - 16;
                float amax_b = 0.0f;
                for (int c = 0; c < 16; ++c) amax_b = fmaxf(amax_b, fabsf(s_raw_b[r][c]));
                const uint8_t sf_b = eg_f32_to_ue4m3(amax_b * (1.0f / 6.0f));
                s_b_sf[r] = sf_b;
                const float inv_sb = 1.0f / eg_ue4m3_to_f32(sf_b);
                for (int i = 0; i < 8; ++i) {
                    const uint8_t q0 = eg_f32_to_e2m1_rn(s_raw_b[r][2 * i] * inv_sb);
                    const uint8_t q1 = eg_f32_to_e2m1_rn(s_raw_b[r][2 * i + 1] * inv_sb);
                    s_b_fp4[r * 8 + i] = (uint8_t)(q0 | (q1 << 4));
                }
            }
            eg_fence_proxy_async();
            __syncthreads();
            eg_bwd_tcgen05_mma_nvfp4(tmem_d, s_a_fp4, s_a_sf, s_b_fp4, s_b_sf, 16, 16, 16, nb > 0 ? 1 : 0);
            __syncthreads();
        }
        eg_bwd_tcgen05_commit(&s_tmem_mbar);
        __syncthreads();
        eg_bwd_tcgen05_wait();
        __syncthreads();
        if (tid < 16) {
            float acc[16];
            eg_bwd_tcgen05_ld_f32(acc, tmem_d, tid, 0, 16);
            float* const dst = dX + (size_t)(mb * 16 + tid) * K + kb * 16;
            for (int c = 0; c < 16; ++c) dst[c] = acc[c];
        }
    } else {
        // wGrad tile with column-wise stochastic-rounded NVFP4 dY^T and transposed NVFP4 X.
        const int nb = tile_y - M_TILES;
        const uint32_t col_seed = SEED ^ 0x6a09e667u;
        for (int mb = 0; mb < M_TILES; ++mb) {
            for (int idx = tid; idx < 256; idx += 32) {
                const int r = idx >> 4;
                const int c = idx & 15;
                s_raw_a[r][c] = dY[(size_t)(mb * 16 + c) * N + (nb * 16 + r)];
                s_raw_b[r][c] = X[(size_t)(mb * 16 + c) * K + (kb * 16 + r)];
            }
            __syncthreads();
            if (tid < 16) {
                const int r = tid;
                const int n = nb * 16 + r;
                float amax_a = 0.0f;
                for (int c = 0; c < 16; ++c) amax_a = fmaxf(amax_a, fabsf(s_raw_a[r][c]));
                const uint8_t sf_a = eg_f32_to_ue4m3(amax_a * (1.0f / 6.0f));
                s_a_sf[r] = sf_a;
                const float inv_sa = 1.0f / eg_ue4m3_to_f32(sf_a);
                for (int i = 0; i < 8; ++i) {
                    const uint32_t pair_idx = (uint32_t)(n * M_BYTES + mb * 8 + i);
                    const uint32_t rbits = eg_splitmix32(col_seed ^ pair_idx);
                    s_a_fp4[r * 8 + i] = eg_cvt_rs_e2m1x2_f32(
                        s_raw_a[r][2 * i] * inv_sa, s_raw_a[r][2 * i + 1] * inv_sa, rbits);
                }
            } else {
                const int r = tid - 16;
                float amax_b = 0.0f;
                for (int c = 0; c < 16; ++c) amax_b = fmaxf(amax_b, fabsf(s_raw_b[r][c]));
                const uint8_t sf_b = eg_f32_to_ue4m3(amax_b * (1.0f / 6.0f));
                s_b_sf[r] = sf_b;
                const float inv_sb = 1.0f / eg_ue4m3_to_f32(sf_b);
                for (int i = 0; i < 8; ++i) {
                    const uint8_t q0 = eg_f32_to_e2m1_rn(s_raw_b[r][2 * i] * inv_sb);
                    const uint8_t q1 = eg_f32_to_e2m1_rn(s_raw_b[r][2 * i + 1] * inv_sb);
                    s_b_fp4[r * 8 + i] = (uint8_t)(q0 | (q1 << 4));
                }
            }
            eg_fence_proxy_async();
            __syncthreads();
            eg_bwd_tcgen05_mma_nvfp4(tmem_d, s_a_fp4, s_a_sf, s_b_fp4, s_b_sf, 16, 16, 16, mb > 0 ? 1 : 0);
            __syncthreads();
        }
        eg_bwd_tcgen05_commit(&s_tmem_mbar);
        __syncthreads();
        eg_bwd_tcgen05_wait();
        __syncthreads();
        if (tid < 16) {
            float acc[16];
            eg_bwd_tcgen05_ld_f32(acc, tmem_d, tid, 0, 16);
            float* const dst = dW + (size_t)(nb * 16 + tid) * K + kb * 16;
            for (int c = 0; c < 16; ++c) dst[c] = acc[c];
        }
    }
`)
	}

	b.WriteString(`    __syncthreads();
    eg_bwd_tmem_dealloc(tmem_d, 32u);
}
`)
	return b.String(), nil
}

// Launcher returns the host launch wrapper for g:
//
//	void Name_launch(dX, dW, X, W, dY, stream)
func (g TCGen05LinearBackward) Launcher() (string, error) {
	g, err := g.norm()
	if err != nil {
		return "", err
	}
	return launcher(g.Name, g.params(), func(b *strings.Builder) {
		fmt.Fprintf(b, "    const dim3 grid(%d, %d, 1), block(32, 1, 1);\n", g.K/16, g.M/16+g.N/16)
	}), nil
}

// Source returns the complete CUDA C++ translation unit ([Prelude], kernel, and launcher).
func (g TCGen05LinearBackward) Source() (string, error) {
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

// Emit is an alias for [TCGen05LinearBackward.Source].
func (g TCGen05LinearBackward) Emit() (string, error) {
	return g.Source()
}

// Reference computes the CPU reference for [TCGen05LinearBackward] over X [M, K],
// W [N, K], and dY [M, N], returning dX [M, K] and dW [N, K].
func (g TCGen05LinearBackward) Reference(x, w, dY []float32) (dX, dW []float32, err error) {
	g, err = g.norm()
	if err != nil {
		return nil, nil, err
	}
	if len(x) != g.M*g.K {
		return nil, nil, fmt.Errorf("tcgen05 linear backward %s reference: len(x) = %d, want %d", g.Name, len(x), g.M*g.K)
	}
	if len(w) != g.N*g.K {
		return nil, nil, fmt.Errorf("tcgen05 linear backward %s reference: len(w) = %d, want %d", g.Name, len(w), g.N*g.K)
	}
	if len(dY) != g.M*g.N {
		return nil, nil, fmt.Errorf("tcgen05 linear backward %s reference: len(dY) = %d, want %d", g.Name, len(dY), g.M*g.N)
	}

	dX = make([]float32, g.M*g.K)
	dW = make([]float32, g.N*g.K)

	if !g.QuantizedBlockScaled {
		for m := range g.M {
			for k := range g.K {
				var acc float32
				for n := range g.N {
					acc += dY[m*g.N+n] * w[n*g.K+k]
				}
				dX[m*g.K+k] = acc
			}
		}
		for n := range g.N {
			for k := range g.K {
				var acc float32
				for m := range g.M {
					acc += dY[m*g.N+n] * x[m*g.K+k]
				}
				dW[n*g.K+k] = acc
			}
		}
		return dX, dW, nil
	}

	rowPacked, rowScales, colPacked, colScales, err := QuantizeGradDualAxisNVFP4(dY, g.M, g.N, g.Seed)
	if err != nil {
		return nil, nil, err
	}

	// Quantize W^T [K, N] along N in 16-element blocks for dGrad.
	nBlocks := g.N / 16
	nBytes := g.N / 2
	wtDeq := make([]float32, g.K*g.N)
	for k := range g.K {
		for nb := range nBlocks {
			var amax float32
			for t := range 16 {
				v := w[(nb*16+t)*g.K+k]
				if a := float32(math.Abs(float64(v))); a > amax {
					amax = a
				}
			}
			sf := DecodeUE4M3(EncodeUE4M3(amax * (1.0 / 6.0)))
			invS := 1.0 / sf
			for t := range 16 {
				v := w[(nb*16+t)*g.K+k]
				q := EncodeE2M1(v * invS)
				wtDeq[k*g.N+nb*16+t] = DecodeE2M1(q) * sf
			}
		}
	}

	// Quantize X^T [K, M] along M in 16-element blocks for wGrad.
	mBlocks := g.M / 16
	mBytes := g.M / 2
	xtDeq := make([]float32, g.K*g.M)
	for k := range g.K {
		for mb := range mBlocks {
			var amax float32
			for t := range 16 {
				v := x[(mb*16+t)*g.K+k]
				if a := float32(math.Abs(float64(v))); a > amax {
					amax = a
				}
			}
			sf := DecodeUE4M3(EncodeUE4M3(amax * (1.0 / 6.0)))
			invS := 1.0 / sf
			for t := range 16 {
				v := x[(mb*16+t)*g.K+k]
				q := EncodeE2M1(v * invS)
				xtDeq[k*g.M+mb*16+t] = DecodeE2M1(q) * sf
			}
		}
	}

	// Compute dX [M, K] from row-quantized dY and column-quantized W.
	for m := range g.M {
		for k := range g.K {
			var acc float32
			for n := range g.N {
				byteVal := rowPacked[m*nBytes+(n>>1)]
				code := (byteVal >> (4 * (n & 1))) & 0xf
				sf := DecodeUE4M3(rowScales[m*nBlocks+(n>>4)])
				dyVal := DecodeE2M1(code) * sf
				acc += dyVal * wtDeq[k*g.N+n]
			}
			dX[m*g.K+k] = acc
		}
	}

	// Compute dW [N, K] from column-quantized dY^T and column-quantized X.
	for n := range g.N {
		for k := range g.K {
			var acc float32
			for m := range g.M {
				byteVal := colPacked[n*mBytes+(m>>1)]
				code := (byteVal >> (4 * (m & 1))) & 0xf
				sf := DecodeUE4M3(colScales[n*mBlocks+(m>>4)])
				dyVal := DecodeE2M1(code) * sf
				acc += dyVal * xtDeq[k*g.M+m]
			}
			dW[n*g.K+k] = acc
		}
	}

	return dX, dW, nil
}
