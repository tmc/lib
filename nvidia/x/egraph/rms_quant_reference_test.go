package egraph

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// rmsQuantQ81Block mirrors the 36-byte block_q8_1 layout emitted by GenerateCUDA.
type rmsQuantQ81Block struct {
	D   uint16
	Sum uint16
	QS  [32]int8
}

// rmsQuantQ81UnfusedReference computes the unfused specification of
// rms_quant_q8_1 in float64:
//  1. Unweighted RMSNorm scale over x: norm = 1 / sqrt(mean(x[i]^2) + eps)
//  2. Elementwise affine scaling: y[i] = x[i] * weight[i] * norm
//  3. Per-32-element block Q8_1 quantization of y
func rmsQuantQ81UnfusedReference(x, weight []float32, eps float32) []rmsQuantQ81Block {
	cols := len(x)
	if len(weight) != cols {
		panic("rmsQuantQ81UnfusedReference: x and weight length mismatch")
	}
	if cols == 0 {
		return nil
	}

	var sq float64
	for i := 0; i < cols; i++ {
		xv := float64(x[i])
		sq += xv * xv
	}
	norm := 1.0 / math.Sqrt(sq/float64(cols)+float64(eps))

	y := make([]float64, cols)
	for i := 0; i < cols; i++ {
		y[i] = float64(x[i]) * float64(weight[i]) * norm
	}

	nBlocks := (cols + 31) / 32
	out := make([]rmsQuantQ81Block, nBlocks)
	for b := 0; b < nBlocks; b++ {
		begin := b * 32
		end := begin + 32
		if end > cols {
			end = cols
		}
		var maxv, sumv float64
		for i := begin; i < end; i++ {
			v := y[i]
			if abs := math.Abs(v); abs > maxv {
				maxv = abs
			}
			sumv += v
		}
		d := maxv / 127.0
		out[b].D = rmsQuantFloat32ToHalf(float32(d))
		out[b].Sum = rmsQuantFloat32ToHalf(float32(sumv))
		if d != 0 {
			for i := begin; i < end; i++ {
				q := math.Round(y[i] / d)
				if q > 127 {
					q = 127
				} else if q < -127 {
					q = -127
				}
				out[b].QS[i-begin] = int8(q)
			}
		}
	}
	return out
}

func rmsQuantFloat32ToHalf(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16((b >> 16) & 0x8000)
	exp := int32((b>>23)&0xff) - 127 + 15
	mant := b & 0x7fffff
	if exp >= 0x1f {
		if (b>>23)&0xff == 0xff && mant != 0 {
			return sign | 0x7e00
		}
		return sign | 0x7c00
	}
	if exp <= 0 {
		if exp < -10 {
			return sign
		}
		mant |= 0x800000
		shift := uint32(14 - exp)
		h := sign | uint16(mant>>shift)
		halfWay := uint32(1) << (shift - 1)
		mask := (uint32(1) << shift) - 1
		rest := mant & mask
		if rest > halfWay || (rest == halfWay && (h&1) != 0) {
			h++
		}
		return h
	}
	h := sign | (uint16(exp) << 10) | uint16(mant>>13)
	rest := mant & 0x1fff
	if rest > 0x1000 || (rest == 0x1000 && (h&1) != 0) {
		h++
	}
	return h
}

func rmsQuantHalfToFloat32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h & 0x3ff)
	switch {
	case exp == 0 && mant == 0:
		return math.Float32frombits(sign)
	case exp == 0:
		v := float32(mant) * float32(math.Pow(2, -24))
		if sign != 0 {
			return -v
		}
		return v
	case exp == 0x1f:
		return math.Float32frombits(sign | 0x7f800000 | (mant << 13))
	default:
		return math.Float32frombits(sign | ((exp - 15 + 127) << 23) | (mant << 13))
	}
}

const rmsQuantHostShimHeader = `
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EGRAPH_COMMON_DEFINED
#define __global__
#define __device__
#define __forceinline__ inline

struct { int x; } blockIdx = {0}, threadIdx = {0};

struct block_q8_1 {
    uint16_t d;
    uint16_t sum;
    int8_t qs[32];
};
static_assert(sizeof(block_q8_1) == 36, "Q8_1 activation layout");

static inline float rsqrtf(float x) {
    return 1.0f / sqrtf(x);
}

static inline uint16_t float_to_half(float f) {
    uint32_t b = 0;
    memcpy(&b, &f, sizeof(b));
    uint16_t sign = (uint16_t)((b >> 16) & 0x8000u);
    int32_t exp = (int32_t)((b >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = b & 0x7fffffu;
    if (exp >= 0x1f) {
        if (((b >> 23) & 0xffu) == 0xffu && mant != 0) {
            return (uint16_t)(sign | 0x7e00u);
        }
        return (uint16_t)(sign | 0x7c00u);
    }
    if (exp <= 0) {
        if (exp < -10) {
            return sign;
        }
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint16_t h = (uint16_t)(sign | (mant >> shift));
        uint32_t half_way = 1u << (shift - 1);
        uint32_t mask = (1u << shift) - 1u;
        uint32_t rest = mant & mask;
        if (rest > half_way || (rest == half_way && (h & 1u))) {
            h++;
        }
        return h;
    }
    uint16_t h = (uint16_t)(sign | ((uint16_t)exp << 10) | (uint16_t)(mant >> 13));
    uint32_t rest = mant & 0x1fffu;
    if (rest > 0x1000u || (rest == 0x1000u && (h & 1u))) {
        h++;
    }
    return h;
}
`

const rmsQuantHostDriverMain = `
int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 2;
    }
    FILE* fin = fopen(argv[1], "rb");
    if (!fin) {
        perror("fopen input");
        return 2;
    }
    int32_t cols = 0;
    float eps = 0.0f;
    if (fread(&cols, sizeof(int32_t), 1, fin) != 1 ||
        fread(&eps, sizeof(float), 1, fin) != 1) {
        fprintf(stderr, "short header read\n");
        fclose(fin);
        return 2;
    }
    float* x = (float*)calloc((size_t)cols, sizeof(float));
    float* weight = (float*)calloc((size_t)cols, sizeof(float));
    int nblocks = (cols + 31) / 32;
    block_q8_1* out = (block_q8_1*)calloc((size_t)(nblocks > 0 ? nblocks : 1), sizeof(block_q8_1));
    if (!x || !weight || !out) {
        fprintf(stderr, "alloc failed\n");
        fclose(fin);
        return 2;
    }
    if ((cols > 0 && fread(x, sizeof(float), (size_t)cols, fin) != (size_t)cols) ||
        (cols > 0 && fread(weight, sizeof(float), (size_t)cols, fin) != (size_t)cols)) {
        fprintf(stderr, "short payload read\n");
        fclose(fin);
        return 2;
    }
    fclose(fin);

    rms_quant_q8_1_kernel(x, weight, out, (int)cols, eps);

    FILE* fout = fopen(argv[2], "wb");
    if (!fout) {
        perror("fopen output");
        return 2;
    }
    if (nblocks > 0 && fwrite(out, sizeof(block_q8_1), (size_t)nblocks, fout) != (size_t)nblocks) {
        fprintf(stderr, "short output write\n");
        fclose(fout);
        return 2;
    }
    fclose(fout);
    free(x);
    free(weight);
    free(out);
    return 0;
}
`

func compileEmittedRMSQuantRunner(t *testing.T, kernelSrc string) string {
	t.Helper()
	var cxx string
	for _, cand := range []string{"clang++", "c++", "g++", "cc"} {
		if p, err := exec.LookPath(cand); err == nil {
			cxx = p
			break
		}
	}
	if cxx == "" {
		t.Skip("no host C++ compiler found in PATH")
	}

	dir := t.TempDir()
	srcPath := filepath.Join(dir, "rms_quant_q8_1_host.cpp")
	binPath := filepath.Join(dir, "rms_quant_q8_1_host")

	fullSrc := rmsQuantHostShimHeader + "\n" + kernelSrc + "\n" + rmsQuantHostDriverMain
	if err := os.WriteFile(srcPath, []byte(fullSrc), 0o600); err != nil {
		t.Fatalf("write host source: %v", err)
	}

	cmd := exec.Command(cxx, "-std=c++11", "-O2", srcPath, "-lm", "-o", binPath)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("compile emitted kernel with %s: %v\n%s\nsource:\n%s", cxx, err, out, fullSrc)
	}
	return binPath
}

func runEmittedRMSQuantKernel(t *testing.T, binPath string, x, weight []float32, eps float32) []rmsQuantQ81Block {
	t.Helper()
	if len(x) != len(weight) {
		t.Fatalf("len(x)=%d != len(weight)=%d", len(x), len(weight))
	}
	cols := len(x)
	dir := t.TempDir()
	inPath := filepath.Join(dir, "in.bin")
	outPath := filepath.Join(dir, "out.bin")

	buf := make([]byte, 8+cols*4*2)
	binary.LittleEndian.PutUint32(buf[0:4], uint32(int32(cols)))
	binary.LittleEndian.PutUint32(buf[4:8], math.Float32bits(eps))
	off := 8
	for _, v := range x {
		binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(v))
		off += 4
	}
	for _, v := range weight {
		binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(v))
		off += 4
	}
	if err := os.WriteFile(inPath, buf, 0o600); err != nil {
		t.Fatalf("write input: %v", err)
	}

	cmd := exec.Command(binPath, inPath, outPath)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("run emitted kernel binary: %v\n%s", err, out)
	}

	raw, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}
	nBlocks := (cols + 31) / 32
	if len(raw) != nBlocks*36 {
		t.Fatalf("output byte length = %d, want %d (%d blocks * 36 bytes)", len(raw), nBlocks*36, nBlocks)
	}
	blocks := make([]rmsQuantQ81Block, nBlocks)
	for b := 0; b < nBlocks; b++ {
		base := b * 36
		blocks[b].D = binary.LittleEndian.Uint16(raw[base : base+2])
		blocks[b].Sum = binary.LittleEndian.Uint16(raw[base+2 : base+4])
		for j := 0; j < 32; j++ {
			blocks[b].QS[j] = int8(raw[base+4+j])
		}
	}
	return blocks
}

func compareRMSQuantBlocks(t *testing.T, got, want []rmsQuantQ81Block, cols int) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("got %d blocks, want %d", len(got), len(want))
	}
	for b := range want {
		gotD := rmsQuantHalfToFloat32(got[b].D)
		wantD := rmsQuantHalfToFloat32(want[b].D)
		dTol := 1e-3 * math.Max(1e-4, math.Abs(float64(wantD)))
		if diff := math.Abs(float64(gotD - wantD)); diff > dTol {
			t.Errorf("block[%d].d = %#04x (%g), want %#04x (%g) (diff=%g > tol=%g)",
				b, got[b].D, gotD, want[b].D, wantD, diff, dTol)
		}

		gotSum := rmsQuantHalfToFloat32(got[b].Sum)
		wantSum := rmsQuantHalfToFloat32(want[b].Sum)
		sumTol := 2e-3 * math.Max(1.0, math.Abs(float64(wantSum)))
		if diff := math.Abs(float64(gotSum - wantSum)); diff > sumTol {
			t.Errorf("block[%d].sum = %#04x (%g), want %#04x (%g) (diff=%g > tol=%g)",
				b, got[b].Sum, gotSum, want[b].Sum, wantSum, diff, sumTol)
		}

		begin := b * 32
		end := begin + 32
		if end > cols {
			end = cols
		}
		active := end - begin
		for j := 0; j < active; j++ {
			diff := int(got[b].QS[j]) - int(want[b].QS[j])
			if diff < -1 || diff > 1 {
				t.Errorf("block[%d].qs[%d] = %d, want %d (±1)", b, j, got[b].QS[j], want[b].QS[j])
			}
		}
		for j := active; j < 32; j++ {
			if got[b].QS[j] != 0 {
				t.Errorf("tail block[%d].qs[%d] = %d, want 0", b, j, got[b].QS[j])
			}
		}
	}
}

// TestRMSQuantQ81EmittedKernelMatchesUnfusedReference compiles and executes the
// C source emitted by generateRMSQuantQ81Kernel (and GenerateCUDA for
// "rms_quant_q8_1") and compares the resulting block_q8_1 output against an
// independent unfused float64 RMSNorm + affine weight + Q8_1 quantizer.
func TestRMSQuantQ81EmittedKernelMatchesUnfusedReference(t *testing.T) {
	opts := CodegenOptions{KernelName: "rms_quant_q8_1_kernel", Precision: "q8_1"}
	kernelSrc := generateRMSQuantQ81Kernel(opts)

	fullCUDA, err := GenerateCUDA(&Expr{Op: "rms_quant_q8_1"}, opts)
	if err != nil {
		t.Fatalf("GenerateCUDA(rms_quant_q8_1): %v", err)
	}
	if !strings.Contains(fullCUDA, kernelSrc) {
		t.Fatal("GenerateCUDA(rms_quant_q8_1) output does not contain generateRMSQuantQ81Kernel body")
	}

	binPath := compileEmittedRMSQuantRunner(t, fullCUDA)

	t.Run("non_unit_alternating_weights", func(t *testing.T) {
		// 32 elements with x[i] = 1.0 and weight[i] alternating 1.0 and 2.0.
		// True unweighted RMS of x is 1.0 (norm = 1.0), so weighted values
		// alternate 1.0 and 2.0: maxv = 2.0, d = 2.0/127.0, sum = 48.0.
		// A buggy kernel that accumulates (x[i]*weight[i])^2 computes
		// mean square = (16*1 + 16*4)/32 = 2.5 and scales by 1/sqrt(2.5) ≈ 0.6325.
		const cols = 32
		x := make([]float32, cols)
		weight := make([]float32, cols)
		for i := 0; i < cols; i++ {
			x[i] = 1.0
			if i%2 == 0 {
				weight[i] = 1.0
			} else {
				weight[i] = 2.0
			}
		}
		got := runEmittedRMSQuantKernel(t, binPath, x, weight, 0.0)
		want := rmsQuantQ81UnfusedReference(x, weight, 0.0)
		compareRMSQuantBlocks(t, got, want, cols)
	})

	t.Run("positive_eps", func(t *testing.T) {
		// Small x with non-trivial eps = 0.75 and non-unit weights so both eps
		// and unweighted sq reduction materially affect the output scale.
		// With x[i] = 0.5, mean(x^2) = 0.25, mean(x^2) + eps = 1.0 -> norm = 1.0.
		const cols = 32
		const eps = float32(0.75)
		x := make([]float32, cols)
		weight := make([]float32, cols)
		for i := 0; i < cols; i++ {
			x[i] = 0.5
			if i%2 == 0 {
				weight[i] = 3.0
			} else {
				weight[i] = 1.5
			}
		}
		got := runEmittedRMSQuantKernel(t, binPath, x, weight, eps)
		want := rmsQuantQ81UnfusedReference(x, weight, eps)
		compareRMSQuantBlocks(t, got, want, cols)
	})

	t.Run("signed_x_and_signed_weight", func(t *testing.T) {
		const cols = 64
		const eps = float32(1e-5)
		x := make([]float32, cols)
		weight := make([]float32, cols)
		for i := 0; i < cols; i++ {
			x[i] = float32((i*7)%13-6) * 0.35
			weight[i] = float32((i*5)%11-5) * 0.6
		}
		got := runEmittedRMSQuantKernel(t, binPath, x, weight, eps)
		want := rmsQuantQ81UnfusedReference(x, weight, eps)
		compareRMSQuantBlocks(t, got, want, cols)
	})

	for _, cols := range []int{45, 50} {
		t.Run(fmt.Sprintf("tail_block_cols_%d", cols), func(t *testing.T) {
			const eps = float32(1e-4)
			x := make([]float32, cols)
			weight := make([]float32, cols)
			for i := 0; i < cols; i++ {
				x[i] = float32((i*11)%17-8) * 0.25
				weight[i] = float32((i*3)%7-3)*0.75 + 0.5
			}
			got := runEmittedRMSQuantKernel(t, binPath, x, weight, eps)
			want := rmsQuantQ81UnfusedReference(x, weight, eps)
			compareRMSQuantBlocks(t, got, want, cols)
		})
	}

	t.Run("weighted_square_mutant_rejected", func(t *testing.T) {
		mutated := strings.Replace(fullCUDA, "float v = x[i];\n        sq += v * v;", "float v = x[i] * weight[i];\n        sq += v * v;", 1)
		if mutated == fullCUDA {
			t.Fatal("failed to construct weighted-square mutant from emitted kernel")
		}
		mutBin := compileEmittedRMSQuantRunner(t, mutated)
		const cols = 32
		x := make([]float32, cols)
		weight := make([]float32, cols)
		for i := 0; i < cols; i++ {
			x[i] = 1.0
			if i%2 == 0 {
				weight[i] = 1.0
			} else {
				weight[i] = 2.0
			}
		}
		gotMut := runEmittedRMSQuantKernel(t, mutBin, x, weight, 0.0)
		want := rmsQuantQ81UnfusedReference(x, weight, 0.0)
		gotSum := rmsQuantHalfToFloat32(gotMut[0].Sum)
		wantSum := rmsQuantHalfToFloat32(want[0].Sum)
		if math.Abs(float64(gotSum-wantSum)) < 1.0 {
			t.Fatalf("weighted-square mutant sum=%g unexpectedly matched unfused reference sum=%g", gotSum, wantSum)
		}
	})
}
