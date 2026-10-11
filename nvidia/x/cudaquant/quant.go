// SPDX-License-Identifier: Apache-2.0

package cudaquant

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// QuantFormat identifies a weight quantization format.
type QuantFormat string

const (
	FormatQ4_0 QuantFormat = "Q4_0" // 4-bit symmetric with float16 scale per 32 elements
	FormatQ8_0 QuantFormat = "Q8_0" // 8-bit symmetric with float16 scale per 32 elements
)

// BlockQ4_0 represents a 32-element Q4_0 quantized block (18 bytes total).
type BlockQ4_0 struct {
	Scale  uint16   // float16 scale factor
	Quants [16]byte // 32 4-bit nibbles (low/high)
}

// BlockQ8_0 represents a 32-element Q8_0 quantized block (34 bytes total).
type BlockQ8_0 struct {
	Scale  uint16   // float16 scale factor
	Quants [32]int8 // 32 8-bit integers
}

// BlockQ8_1 stores a normalized 32-element row block with fp16 d and s.
type BlockQ8_1 struct {
	D      uint16
	S      uint16
	Quants [32]int8
}

// RMSNormQ8_1 computes row-wise RMS normalization and Q8_1 quantization.
// When weighted is true, weight must contain one value per column and is
// applied after normalization. Quantization rounds to nearest integer and
// saturates to [-127,127].
func RMSNormQ8_1(input, weight []float32, rows, cols int, epsilon float32, weighted bool) ([]BlockQ8_1, error) {
	if rows <= 0 || cols <= 0 || rows > math.MaxInt/cols || cols%32 != 0 || epsilon < 0 || math.IsNaN(float64(epsilon)) || math.IsInf(float64(epsilon), 0) {
		return nil, fmt.Errorf("cudaquant: invalid RMSNorm dimensions or epsilon")
	}
	if len(input) != rows*cols {
		return nil, fmt.Errorf("cudaquant: input length mismatch")
	}
	if weighted && len(weight) != cols {
		return nil, fmt.Errorf("cudaquant: weight length mismatch")
	}
	for _, v := range input {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, fmt.Errorf("cudaquant: nonfinite input")
		}
	}
	if weighted {
		for _, v := range weight {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return nil, fmt.Errorf("cudaquant: nonfinite weight")
			}
		}
	}
	out := make([]BlockQ8_1, rows*cols/32)
	for r := 0; r < rows; r++ {
		var ss float64
		for c := 0; c < cols; c++ {
			v := input[r*cols+c]
			ss += float64(v) * float64(v)
		}
		denom := ss/float64(cols) + float64(epsilon)
		if denom == 0 {
			return nil, fmt.Errorf("cudaquant: zero normalization denominator")
		}
		inv := float32(1 / math.Sqrt(denom))
		for b := 0; b < cols/32; b++ {
			var max float32
			vals := [32]float32{}
			for i := 0; i < 32; i++ {
				v := input[r*cols+b*32+i] * inv
				if weighted {
					v *= weight[b*32+i]
				}
				if math.IsInf(float64(v), 0) || math.IsNaN(float64(v)) {
					return nil, fmt.Errorf("cudaquant: normalized value is nonfinite")
				}
				vals[i] = v
				if math.Abs(float64(v)) > float64(max) {
					max = float32(math.Abs(float64(v)))
				}
			}
			qscale := max / 127
			q := &out[r*cols/32+b]
			q.D = float32ToFloat16(qscale)
			if qscale == 0 {
				continue
			}
			sum := 0
			for i, v := range vals {
				qv := int(math.Round(float64(v / qscale)))
				if qv > 127 {
					qv = 127
				}
				if qv < -127 {
					qv = -127
				}
				q.Quants[i] = int8(qv)
				sum += qv
			}
			q.S = float32ToFloat16(float32(sum) * qscale)
		}
	}
	return out, nil
}

// RMSNormQ8_1Source emits the fused CUDA kernel for the declared fixture.
// It includes cuda_fp16.h; NVRTC supplies size_t and the math functions.
func RMSNormQ8_1Source() string {
	return `#include <cuda_fp16.h>
struct Q8_1 { __half d, s; signed char q[32]; };
static_assert(sizeof(Q8_1) == 36, "q8_1 layout");
// Bounded correctness implementation: one serial thread per row. Input and
// weights must be finite, cols positive and divisible by 32, epsilon nonnegative.
extern "C" __global__ void rms_quant_q8_1(const float* in, const float* weight,
    Q8_1* out, unsigned rows, unsigned cols, float eps, unsigned weighted) {
  unsigned r = blockIdx.x;
  if (r >= rows || threadIdx.x != 0) return;
  double ss = 0;
  for (unsigned c=0; c<cols; ++c) { double v=in[(size_t)r*cols+c]; ss += v*v; }
  float inv = (float)(1.0 / sqrt(ss/cols + (double)eps));
  for (unsigned b=0; b<cols/32; ++b) {
    float values[32], maximum=0;
    for (unsigned i=0; i<32; ++i) {
      unsigned c=b*32+i;
      float v=in[(size_t)r*cols+c]*inv;
      if (weighted) v*=weight[c];
      values[i]=v; maximum=fmaxf(maximum,fabsf(v));
    }
    float d=maximum/127.0f;
    Q8_1* dst=out+(size_t)r*(cols/32)+b;
    dst->d=__float2half_rn(d);
    int sum=0;
    for (unsigned i=0; i<32; ++i) {
      int q=d == 0 ? 0 : (int)roundf(values[i]/d);
      q=q>127 ? 127 : (q < -127 ? -127 : q);
      dst->q[i]=(signed char)q; sum+=q;
    }
    dst->s=__float2half_rn((float)sum*d);
  }
}`
}

// RMSNormQ8_1Artifact compiles the generated kernel for a physical CUDA target
// (for example, sm_121a), preserving its fixed ABI and compiled resources.
// It requires NVRTC and CUDA headers; it does not load or launch the kernel.
func RMSNormQ8_1Artifact(target string) (*cudakernel.Artifact, error) {
	return normQuantArtifact(target, func(src string, opts ...nvrtcx.Option) (normQuantModule, error) {
		return nvrtcx.Compile(src, opts...)
	})
}

type normQuantModule interface {
	ToArtifact(string, cudakernel.Schema, string) (*cudakernel.Artifact, error)
}

func normQuantArtifact(target string, compile func(string, ...nvrtcx.Option) (normQuantModule, error)) (*cudakernel.Artifact, error) {
	artifacts, err := normQuantArtifacts(target, RMSNormQ8_1Source(), []string{"rms_quant_q8_1"}, compile)
	if err != nil {
		return nil, err
	}
	return artifacts[0], nil
}

func normQuantArtifacts(target, source string, entries []string, compile func(string, ...nvrtcx.Option) (normQuantModule, error)) ([]*cudakernel.Artifact, error) {
	if !strings.HasPrefix(target, "sm_") {
		return nil, fmt.Errorf("cudaquant: expected physical target, got %q", target)
	}
	major, minor, suffix, err := cudakernel.ParseArch(target)
	if err != nil {
		return nil, fmt.Errorf("cudaquant: target: %w", err)
	}
	if (suffix != "" && suffix != "a" && suffix != "f") || target != fmt.Sprintf("sm_%d%d%s", major, minor, suffix) {
		return nil, fmt.Errorf("cudaquant: invalid physical target %q", target)
	}
	schema, err := cudakernel.NewSchema(
		cudakernel.Param{Name: "in", Type: "ptr", IsPointer: true}, cudakernel.Param{Name: "weight", Type: "ptr", IsPointer: true}, cudakernel.Param{Name: "out", Type: "ptr", IsPointer: true},
		cudakernel.Param{Name: "rows", Type: "u32"}, cudakernel.Param{Name: "cols", Type: "u32"}, cudakernel.Param{Name: "eps", Type: "f32"}, cudakernel.Param{Name: "weighted", Type: "u32"})
	if err != nil {
		return nil, err
	}
	opts := []nvrtcx.Option{nvrtcx.WithPhysicalArchTarget(target), nvrtcx.WithStandard("c++17"), nvrtcx.WithToolkitIncludes(), nvrtcx.WithOutput(nvrtcx.OutputCUBIN)}
	flags, err := nvrtcx.CompileFlags(opts...)
	if err != nil {
		return nil, fmt.Errorf("cudaquant: compiler options: %w", err)
	}
	cm, err := compile(source, opts...)
	if err != nil {
		return nil, fmt.Errorf("cudaquant: compile norm quant: %w", err)
	}
	if cm == nil {
		return nil, fmt.Errorf("cudaquant: compile norm quant returned no module")
	}
	var artifacts []*cudakernel.Artifact
	for _, entry := range entries {
		artifact, err := cm.ToArtifact(entry, schema, target)
		if err != nil {
			return nil, fmt.Errorf("cudaquant: %s artifact: %w", entry, err)
		}
		if artifact == nil {
			return nil, fmt.Errorf("cudaquant: %s returned no artifact", entry)
		}
		artifact.CompilerOpts = append([]string(nil), flags...)
		artifacts = append(artifacts, artifact)
	}
	return artifacts, nil
}

// DequantizeQ4_0 unpacks Q4_0 blocks into a float32 slice.
func DequantizeQ4_0(data []byte, numElements int) ([]float32, error) {
	want, err := encodedLen(numElements, 18)
	if err != nil {
		return nil, err
	}
	numBlocks := numElements / 32
	if len(data) != want {
		return nil, fmt.Errorf("cudaquant: buffer length %d, want exactly %d for %d Q4_0 blocks", len(data), want, numBlocks)
	}

	out := make([]float32, numElements)
	for b := 0; b < numBlocks; b++ {
		offset := b * 18
		scaleBits := binary.LittleEndian.Uint16(data[offset : offset+2])
		scale := float16ToFloat32(scaleBits)
		if !finite(scale) {
			return nil, fmt.Errorf("cudaquant: non-finite Q4_0 scale in block %d", b)
		}

		for i := 0; i < 16; i++ {
			byteVal := data[offset+2+i]
			q0 := int8(byteVal&0x0F) - 8
			q1 := int8(byteVal>>4) - 8

			out[b*32+i] = float32(q0) * scale
			out[b*32+16+i] = float32(q1) * scale
		}
	}
	return out, nil
}

// DequantizeQ8_0 unpacks Q8_0 blocks into a float32 slice.
func DequantizeQ8_0(data []byte, numElements int) ([]float32, error) {
	want, err := encodedLen(numElements, 34)
	if err != nil {
		return nil, err
	}
	numBlocks := numElements / 32
	if len(data) != want {
		return nil, fmt.Errorf("cudaquant: buffer length %d, want exactly %d for %d Q8_0 blocks", len(data), want, numBlocks)
	}

	out := make([]float32, numElements)
	for b := 0; b < numBlocks; b++ {
		offset := b * 34
		scaleBits := binary.LittleEndian.Uint16(data[offset : offset+2])
		scale := float16ToFloat32(scaleBits)
		if !finite(scale) {
			return nil, fmt.Errorf("cudaquant: non-finite Q8_0 scale in block %d", b)
		}

		for i := 0; i < 32; i++ {
			q := int8(data[offset+2+i])
			out[b*32+i] = float32(q) * scale
		}
	}
	return out, nil
}

func encodedLen(numElements, blockBytes int) (int, error) {
	if numElements < 0 {
		return 0, fmt.Errorf("cudaquant: negative element count %d", numElements)
	}
	if numElements%32 != 0 {
		return 0, fmt.Errorf("cudaquant: %d elements is not a multiple of the 32-element block", numElements)
	}
	blocks := numElements / 32
	if blocks > math.MaxInt/blockBytes {
		return 0, fmt.Errorf("cudaquant: encoded length overflows int")
	}
	return blocks * blockBytes, nil
}

func finite(v float32) bool {
	return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0)
}

func float16ToFloat32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h&0x7C00) >> 10
	mant := uint32(h & 0x03FF)

	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		for (mant & 0x0400) == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x03FF
	} else if exp == 0x1F {
		return math.Float32frombits(sign | 0x7F800000 | (mant << 13))
	}

	exp = exp + (127 - 15)
	mant = mant << 13
	return math.Float32frombits(sign | (exp << 23) | mant)
}

func float32ToFloat16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits>>23)&0xff) - 127 + 15
	mant := bits & 0x7fffff
	if exp <= 0 {
		if exp < -10 {
			return sign
		}
		mant |= 0x800000
		shift := uint(14 - exp)
		q := mant >> shift
		rem := mant & ((uint32(1) << shift) - 1)
		half := uint32(1) << (shift - 1)
		if rem > half || rem == half && q&1 != 0 {
			q++
		}
		return sign | uint16(q)
	}
	if exp >= 31 {
		return sign | 0x7c00
	}
	q := uint32(exp<<10) | mant>>13
	rem := mant & 0x1fff
	if rem > 0x1000 || rem == 0x1000 && q&1 != 0 {
		q++
	}
	return sign | uint16(q)
}
