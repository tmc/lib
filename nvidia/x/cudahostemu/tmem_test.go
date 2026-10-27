package cudahostemu

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"strings"
	"testing"
)

func f32ToBF16Bits(f float32) uint16 {
	u := math.Float32bits(f)
	if u&0x7fffffff > 0x7f800000 {
		return 0x7fc0 | uint16(u>>16)&0x8000
	}
	u += 0x7fff + (u>>16)&1
	return uint16(u >> 16)
}

func bf16BitsToF32(b uint16) float32 {
	return math.Float32frombits(uint32(b) << 16)
}

func f32ToE4M3Bits(f float32) uint8 {
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
	if rem > 0x80000 || (rem == 0x80000 && m&1 == 1) {
		m++
		if m == 8 {
			m = 0
			e++
		}
	}
	if e > 15 || (e == 15 && m == 7) {
		return sign | 0x7e
	}
	return sign | uint8(e<<3) | uint8(m)
}

func e4m3BitsToF32(b uint8) float32 {
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

var tmemE2M1Grid = [8]float32{0, 0.5, 1, 1.5, 2, 3, 4, 6}

func e2m1CodeToF32(code uint8) float32 {
	v := tmemE2M1Grid[code&7]
	if code&8 != 0 {
		return -v
	}
	return v
}

func packU16Slice(v []uint16) []byte {
	b := make([]byte, 2*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint16(b[2*i:], x)
	}
	return b
}

func unpackF32Slice(b []byte) []float32 {
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

func TestTMEMHeaderNonEmpty(t *testing.T) {
	hdr := TMEMHeader()
	for _, sym := range []string{
		"eg_emu_tmem_alloc",
		"eg_emu_tmem_dealloc",
		"eg_emu_tcgen05_mma_bf16",
		"eg_emu_tcgen05_mma_e4m3",
		"eg_emu_tcgen05_mma_nvfp4",
		"eg_emu_tcgen05_commit",
		"eg_emu_tcgen05_wait",
		"eg_emu_tcgen05_ld_f32",
		"eg_emu_tmem_error_count",
	} {
		if !strings.Contains(hdr, sym) {
			t.Errorf("TMEMHeader() missing %q", sym)
		}
	}
}

func TestTMEMPositiveMMATypes(t *testing.T) {
	ctx := context.Background()
	const M, N, K = 128, 64, 32

	t.Run("bf16", func(t *testing.T) {
		a := make([]uint16, M*K)
		b := make([]uint16, N*K)
		for i := range a {
			a[i] = f32ToBF16Bits(float32((i%11)-5) * 0.25)
		}
		for i := range b {
			b[i] = f32ToBF16Bits(float32((i%7)-3) * 0.5)
		}
		want := make([]float64, M*N)
		for m := 0; m < M; m++ {
			for n := 0; n < N; n++ {
				var sum float64
				for k := 0; k < K; k++ {
					sum += float64(bf16BitsToF32(a[m*K+k])) * float64(bf16BitsToF32(b[n*K+k]))
				}
				want[m*N+n] = sum
			}
		}

		const src = `
__global__ void k_tmem_bf16(const uint16_t* a, const uint16_t* b, float* out) {
    __shared__ uint16_t sa[128 * 16];
    __shared__ uint16_t sb[64 * 16];
    __shared__ uint32_t tmem_d;
    const int tid = threadIdx.x;
    if (tid == 0) {
        eg_emu_tmem_alloc(&tmem_d, 64);
    }
    __syncthreads();
    for (int kt = 0; kt < 2; ++kt) {
        for (int i = tid; i < 128 * 16; i += 128) {
            int r = i / 16, c = i % 16;
            sa[i] = a[r * 32 + kt * 16 + c];
        }
        for (int i = tid; i < 64 * 16; i += 128) {
            int r = i / 16, c = i % 16;
            sb[i] = b[r * 32 + kt * 16 + c];
        }
        __syncthreads();
        if (tid == 0) {
            eg_emu_tcgen05_mma_bf16(tmem_d, sa, sb, 128, 64, 16, kt > 0 ? 1 : 0);
        }
        __syncthreads();
    }
    if (tid == 0) {
        eg_emu_tcgen05_commit();
    }
    __syncthreads();
    eg_emu_tcgen05_wait();
    __syncthreads();
    for (int c = 0; c < 64; c += 4) {
        float regs[4];
        eg_emu_tcgen05_ld_f32(regs, tmem_d, tid, c, 4);
        for (int j = 0; j < 4; ++j) {
            out[tid * 64 + c + j] = regs[j];
        }
    }
    __syncthreads();
    if (tid == 0) {
        eg_emu_tmem_dealloc(tmem_d, 64);
    }
}
`
		outBytes := Poison(4 * M * N)
		err := RunWithTMEM(ctx, src, Launch{
			Grid:  Dim3{X: 1},
			Block: Dim3{X: 128},
		},
			BufferArg("a", "uint16_t", packU16Slice(a)),
			BufferArg("b", "uint16_t", packU16Slice(b)),
			BufferArg("out", "float", outBytes),
		)
		if errors.Is(err, ErrNoCompiler) {
			t.Skip(err)
		}
		if err != nil {
			t.Fatalf("RunWithTMEM bf16: %v", err)
		}
		got := unpackF32Slice(outBytes)
		for i, g := range got {
			if math.Abs(float64(g)-want[i]) > 1e-5 {
				t.Fatalf("bf16 out[%d] = %g, want %g", i, g, want[i])
			}
		}
	})

	t.Run("e4m3", func(t *testing.T) {
		a := make([]uint8, M*K)
		b := make([]uint8, N*K)
		for i := range a {
			a[i] = f32ToE4M3Bits(float32((i%9)-4) * 0.25)
		}
		for i := range b {
			b[i] = f32ToE4M3Bits(float32((i%5)-2) * 0.5)
		}
		want := make([]float64, M*N)
		for m := 0; m < M; m++ {
			for n := 0; n < N; n++ {
				var sum float64
				for k := 0; k < K; k++ {
					sum += float64(e4m3BitsToF32(a[m*K+k])) * float64(e4m3BitsToF32(b[n*K+k]))
				}
				want[m*N+n] = sum
			}
		}

		const src = `
__global__ void k_tmem_e4m3(const uint8_t* a, const uint8_t* b, float* out) {
    __shared__ uint8_t sa[128 * 32];
    __shared__ uint8_t sb[64 * 32];
    __shared__ uint32_t tmem_d;
    const int tid = threadIdx.x;
    if (tid == 0) {
        eg_emu_tmem_alloc(&tmem_d, 64);
    }
    __syncthreads();
    for (int i = tid; i < 128 * 32; i += 128) sa[i] = a[i];
    for (int i = tid; i < 64 * 32; i += 128) sb[i] = b[i];
    __syncthreads();
    if (tid == 0) {
        eg_emu_tcgen05_mma_e4m3(tmem_d, sa, sb, 128, 64, 32, 0);
        eg_emu_tcgen05_commit();
    }
    __syncthreads();
    eg_emu_tcgen05_wait();
    __syncthreads();
    float row_vals[64];
    eg_emu_tcgen05_ld_f32(row_vals, tmem_d, tid, 0, 64);
    for (int c = 0; c < 64; ++c) {
        out[tid * 64 + c] = row_vals[c];
    }
    __syncthreads();
    if (tid == 0) {
        eg_emu_tmem_dealloc(tmem_d, 64);
    }
}
`
		outBytes := Poison(4 * M * N)
		err := RunWithTMEM(ctx, src, Launch{
			Grid:  Dim3{X: 1},
			Block: Dim3{X: 128},
		},
			BufferArg("a", "uint8_t", a),
			BufferArg("b", "uint8_t", b),
			BufferArg("out", "float", outBytes),
		)
		if errors.Is(err, ErrNoCompiler) {
			t.Skip(err)
		}
		if err != nil {
			t.Fatalf("RunWithTMEM e4m3: %v", err)
		}
		got := unpackF32Slice(outBytes)
		for i, g := range got {
			if math.Abs(float64(g)-want[i]) > 1e-5 {
				t.Fatalf("e4m3 out[%d] = %g, want %g", i, g, want[i])
			}
		}
	})

	t.Run("nvfp4", func(t *testing.T) {
		aFP4 := make([]uint8, M*(K/2))
		aScale := make([]uint8, M*(K/16))
		bFP4 := make([]uint8, N*(K/2))
		bScale := make([]uint8, N*(K/16))
		for i := range aFP4 {
			lo := uint8((i * 3) & 0xf)
			hi := uint8((i*5 + 1) & 0xf)
			aFP4[i] = lo | (hi << 4)
		}
		for i := range aScale {
			aScale[i] = f32ToE4M3Bits(0.5 + float32(i%4)*0.25)
		}
		for i := range bFP4 {
			lo := uint8((i*7 + 2) & 0xf)
			hi := uint8((i*11 + 3) & 0xf)
			bFP4[i] = lo | (hi << 4)
		}
		for i := range bScale {
			bScale[i] = f32ToE4M3Bits(0.25 + float32(i%3)*0.5)
		}

		want := make([]float64, M*N)
		for m := 0; m < M; m++ {
			for n := 0; n < N; n++ {
				var sum float64
				for k := 0; k < K; k++ {
					ab := aFP4[m*(K/2)+k/2]
					bb := bFP4[n*(K/2)+k/2]
					ac := (ab >> (4 * (k & 1))) & 0xf
					bc := (bb >> (4 * (k & 1))) & 0xf
					asf := e4m3BitsToF32(aScale[m*(K/16)+k/16])
					bsf := e4m3BitsToF32(bScale[n*(K/16)+k/16])
					av := float64(e2m1CodeToF32(ac) * asf)
					bv := float64(e2m1CodeToF32(bc) * bsf)
					sum += av * bv
				}
				want[m*N+n] = sum
			}
		}

		const src = `
__global__ void k_tmem_nvfp4(
    const uint8_t* a_fp4, const uint8_t* a_sf,
    const uint8_t* b_fp4, const uint8_t* b_sf,
    float* out) {
    __shared__ uint8_t sa_fp4[128 * 16];
    __shared__ uint8_t sa_sf[128 * 2];
    __shared__ uint8_t sb_fp4[64 * 16];
    __shared__ uint8_t sb_sf[64 * 2];
    __shared__ uint32_t tmem_d;
    const int tid = threadIdx.x;
    if (tid == 0) {
        eg_emu_tmem_alloc(&tmem_d, 64);
    }
    __syncthreads();
    for (int i = tid; i < 128 * 16; i += 128) sa_fp4[i] = a_fp4[i];
    for (int i = tid; i < 128 * 2; i += 128) sa_sf[i] = a_sf[i];
    for (int i = tid; i < 64 * 16; i += 128) sb_fp4[i] = b_fp4[i];
    for (int i = tid; i < 64 * 2; i += 128) sb_sf[i] = b_sf[i];
    __syncthreads();
    if (tid == 0) {
        eg_emu_tcgen05_mma_nvfp4(tmem_d, sa_fp4, sa_sf, sb_fp4, sb_sf, 128, 64, 32, 0);
        eg_emu_tcgen05_commit();
    }
    __syncthreads();
    eg_emu_tcgen05_wait();
    __syncthreads();
    for (int c = 0; c < 64; c += 4) {
        float regs[4];
        eg_emu_tcgen05_ld_f32(regs, tmem_d, tid, c, 4);
        for (int j = 0; j < 4; ++j) {
            out[tid * 64 + c + j] = regs[j];
        }
    }
    __syncthreads();
    if (tid == 0) {
        eg_emu_tmem_dealloc(tmem_d, 64);
    }
}
`
		outBytes := Poison(4 * M * N)
		err := RunWithTMEM(ctx, src, Launch{
			Grid:  Dim3{X: 1},
			Block: Dim3{X: 128},
		},
			BufferArg("a_fp4", "uint8_t", aFP4),
			BufferArg("a_sf", "uint8_t", aScale),
			BufferArg("b_fp4", "uint8_t", bFP4),
			BufferArg("b_sf", "uint8_t", bScale),
			BufferArg("out", "float", outBytes),
		)
		if errors.Is(err, ErrNoCompiler) {
			t.Skip(err)
		}
		if err != nil {
			t.Fatalf("RunWithTMEM nvfp4: %v", err)
		}
		got := unpackF32Slice(outBytes)
		for i, g := range got {
			if math.Abs(float64(g)-want[i]) > 1e-4 {
				t.Fatalf("nvfp4 out[%d] = %g, want %g", i, g, want[i])
			}
		}
	})
}

func TestTMEMAdversarialHazardsAndLeaks(t *testing.T) {
	ctx := context.Background()

	t.Run("missing_wait_poisons_nan_and_errors", func(t *testing.T) {
		const src = `
__global__ void k_missing_wait(float* out) {
    __shared__ uint16_t sa[128 * 16];
    __shared__ uint16_t sb[32 * 16];
    __shared__ uint32_t tmem_d;
    const int tid = threadIdx.x;
    for (int i = tid; i < 128 * 16; i += 128) sa[i] = 0x3f80u; // 1.0 in bf16
    for (int i = tid; i < 32 * 16; i += 128) sb[i] = 0x3f80u;
    if (tid == 0) {
        eg_emu_tmem_alloc(&tmem_d, 32);
    }
    __syncthreads();
    if (tid == 0) {
        eg_emu_tcgen05_mma_bf16(tmem_d, sa, sb, 128, 32, 16, 0);
        eg_emu_tcgen05_commit();
        // Deliberately omit eg_emu_tcgen05_wait()!
    }
    __syncthreads();
    float vals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    eg_emu_tcgen05_ld_f32(vals, tmem_d, tid, 0, 4);
    for (int j = 0; j < 4; ++j) {
        out[tid * 4 + j] = vals[j];
    }
    __syncthreads();
    if (tid == 0) {
        eg_emu_tmem_dealloc(tmem_d, 32);
    }
}
`
		outBytes := make([]byte, 4*128*4)
		err := RunWithTMEM(ctx, src, Launch{
			Grid:  Dim3{X: 1},
			Block: Dim3{X: 128},
		}, BufferArg("out", "float", outBytes))
		if errors.Is(err, ErrNoCompiler) {
			t.Skip(err)
		}
		if err == nil {
			t.Fatal("expected TMEM error when reading before eg_emu_tcgen05_wait(), got nil")
		}
		if !strings.Contains(err.Error(), "before eg_emu_tcgen05_wait") {
			t.Fatalf("expected missing wait error detail, got: %v", err)
		}
		got := unpackF32Slice(outBytes)
		for i, v := range got {
			if !math.IsNaN(float64(v)) {
				t.Fatalf("out[%d] = %g, want NaN when reading TMEM before wait", i, v)
			}
		}
	})

	t.Run("missing_dealloc_reports_leak_error", func(t *testing.T) {
		const src = `
__global__ void k_leak_tmem(float* out) {
    __shared__ uint16_t sa[128 * 16];
    __shared__ uint16_t sb[64 * 16];
    __shared__ uint32_t tmem_d;
    const int tid = threadIdx.x;
    for (int i = tid; i < 128 * 16; i += 128) sa[i] = 0x3f80u;
    for (int i = tid; i < 64 * 16; i += 128) sb[i] = 0x3f80u;
    if (tid == 0) {
        eg_emu_tmem_alloc(&tmem_d, 64);
    }
    __syncthreads();
    if (tid == 0) {
        eg_emu_tcgen05_mma_bf16(tmem_d, sa, sb, 128, 64, 16, 0);
        eg_emu_tcgen05_commit();
    }
    __syncthreads();
    eg_emu_tcgen05_wait();
    __syncthreads();
    float v = 0.0f;
    eg_emu_tcgen05_ld_f32(&v, tmem_d, tid, 0, 1);
    out[tid] = v;
    // Deliberately omit eg_emu_tmem_dealloc(tmem_d, 64)!
}
`
		outBytes := make([]byte, 4*128)
		err := RunWithTMEM(ctx, src, Launch{
			Grid:  Dim3{X: 1},
			Block: Dim3{X: 128},
		}, BufferArg("out", "float", outBytes))
		if errors.Is(err, ErrNoCompiler) {
			t.Skip(err)
		}
		if err == nil {
			t.Fatal("expected TMEM leak error when omitting eg_emu_tmem_dealloc, got nil")
		}
		if !strings.Contains(err.Error(), "leaked") {
			t.Fatalf("expected TMEM leak error detail, got: %v", err)
		}
	})

	t.Run("invalid_num_cols_reports_error", func(t *testing.T) {
		const src = `
__global__ void k_bad_cols(uint32_t* out) {
    uint32_t addr = 0;
    if (threadIdx.x == 0) {
        eg_emu_tmem_alloc(&addr, 48); // not a power of two in [32, 512]
        out[0] = addr;
    }
}
`
		outBytes := make([]byte, 4)
		err := RunWithTMEM(ctx, src, Launch{
			Grid:  Dim3{X: 1},
			Block: Dim3{X: 32},
		}, BufferArg("out", "uint32_t", outBytes))
		if errors.Is(err, ErrNoCompiler) {
			t.Skip(err)
		}
		if err == nil {
			t.Fatal("expected TMEM allocation error for num_cols=48, got nil")
		}
		if !strings.Contains(err.Error(), "invalid num_cols") {
			t.Fatalf("expected invalid num_cols error detail, got: %v", err)
		}
	})
}
