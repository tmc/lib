//go:build darwin && arm64

package main

import (
	"context"
	"fmt"
	"math"
	"math/bits"
	"strings"
	"testing"

	"github.com/tmc/apple/objectivec"
	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// Registered before device execution. Quantized outputs must satisfy this
// fixture gate; a quantization-bin change is a failure, not a tolerance update.
const metalIndexerRegressionATol = 1e-4
const metalIndexerRegressionRTol = 1e-4

func TestMetalIndexerFinishRegression(t *testing.T) {
	dev, err := libmetal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	start := strings.Index(metalPinnedMSL, "kernel void compressor_finish_indexer(")
	if start < 0 {
		t.Fatal("indexer kernel missing")
	}
	end := strings.Index(metalPinnedMSL[start:], "kernel void rope_attention_mixed(") + start
	if start < 0 || end < start {
		t.Fatal("indexer kernel boundaries missing")
	}
	kernel := metalPinnedMSL[start:end]
	clear := "  if(simdgroup==0)absbuf[lane]=0.0f;\n  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
	declaration := "  threadgroup float *vals=scratch,*absbuf=scratch+128;\n"
	poison := "  absbuf[tid]=1024.0f;\n  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
	ropeStart := strings.Index(kernel, "  if(tid<32){")
	if ropeStart < 0 {
		t.Fatal("indexer rotation missing")
	}
	ropeEnd := strings.Index(kernel[ropeStart:], "  for(uint stride=1;") + ropeStart
	if strings.Count(kernel, clear) != 1 || strings.Count(kernel, declaration) != 1 || ropeStart < 0 || ropeEnd < ropeStart {
		t.Fatal("indexer mutation anchors changed")
	}
	poisoned := strings.Replace(kernel, declaration, declaration+poison, 1)
	for _, tc := range []struct {
		name, source string
		reject       bool
	}{
		{"production", kernel, false},
		{"poisoned scratch", poisoned, false},
		{"old unwritten reduction lanes", strings.Replace(poisoned, clear, "", 1), true},
		{"old omitted rotation", kernel[:ropeStart] + kernel[ropeEnd:], true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			source := metalPinnedMSL[:start] + tc.source + metalPinnedMSL[end:]
			library, err := dev.CompileLibrary(source)
			if err != nil {
				t.Fatal(err)
			}
			defer objectivec.ObjectFromID(library.GetID()).Release()
			pipeline, err := dev.NewComputePipeline(library, "compressor_finish_indexer")
			if err != nil {
				t.Fatal(err)
			}
			defer objectivec.ObjectFromID(pipeline.GetID()).Release()
			for _, position := range []int{0, 4, 2048} {
				if tc.name == "old omitted rotation" && position == 0 {
					continue
				}
				t.Run(fmt.Sprintf("position_%d", position), func(t *testing.T) {
					pooled, weight := make([]float32, 128), make([]float32, 128)
					for i := range pooled {
						pooled[i] = float32((i*19)%67-33) / 32
						weight[i] = 1 + float32(i%7-3)/16
					}
					want := metalIndexerRegressionReference(pooled, weight, position)
					p, err := metalmem.AllocFromSlice(dev, pooled)
					if err != nil {
						t.Fatal(err)
					}
					defer p.Free()
					w, err := metalmem.AllocFromSlice(dev, weight)
					if err != nil {
						t.Fatal(err)
					}
					defer w.Free()
					cache, err := metalmem.Alloc[float32](dev, 3*128)
					if err != nil {
						t.Fatal(err)
					}
					defer cache.Free()
					for i := range cache.HostSlice() {
						cache.HostSlice()[i] = float32(math.NaN())
					}
					cb := dev.Queue().CommandBuffer()
					enc := cb.ComputeCommandEncoder()
					enc.SetComputePipelineState(pipeline)
					for i, b := range []metalmem.Slice[float32]{p, w, cache} {
						enc.SetBufferWithOffsetAtIndex(b.Buffer(), b.Offset(), uint(i))
					}
					libmetal.SetScalar(enc, uint32(1), 3)
					libmetal.SetScalar(enc, uint32(position), 4)
					enc.SetThreadgroupMemoryLengthAtIndex(256*4, 0)
					enc.DispatchThreadgroupsThreadsPerThreadgroup(libmetal.Threads(1), libmetal.Threads(128))
					enc.EndEncoding()
					cb.Commit()
					if err := libmetal.WaitCommandBuffer(context.Background(), cb); err != nil {
						t.Fatal(err)
					}
					err = metalIndexerRegressionClose(cache.HostSlice()[128:256], want)
					if tc.reject {
						if err == nil {
							t.Fatal("old behavior passed the fixture")
						}
						return
					}
					if err != nil {
						t.Fatal(err)
					}
					if err := metalIndexerRegressionClose(p.HostSlice(), want); err != nil {
						t.Fatalf("pooled: %v", err)
					}
					for i, v := range cache.HostSlice() {
						if (i < 128 || i >= 256) && !math.IsNaN(float64(v)) {
							t.Fatalf("wrote outside cache row at %d", i)
						}
					}
				})
			}
		})
	}
}

func metalIndexerRegressionClose(got, want []float32) error {
	if len(got) != len(want) {
		return fmt.Errorf("length %d, want %d", len(got), len(want))
	}
	for i, g := range got {
		x, y := float64(g), float64(want[i])
		if math.IsNaN(x) || math.IsInf(x, 0) || math.IsNaN(y) || math.IsInf(y, 0) {
			return fmt.Errorf("non-finite element %d", i)
		}
		if math.Abs(x-y) > metalIndexerRegressionATol+metalIndexerRegressionRTol*math.Abs(y) {
			return fmt.Errorf("element %d: got %g, want %g", i, x, y)
		}
	}
	return nil
}

// Independent scalar RMS, analytic YaRN, and a direct Hadamard matrix product.
// The device butterfly and reduction scratch are deliberately not reproduced.
func metalIndexerRegressionReference(pooled, weight []float32, position int) []float32 {
	values := make([]float64, 128)
	ss := 0.0
	for _, x := range pooled {
		ss += float64(x) * float64(x)
	}
	for i, x := range pooled {
		values[i] = float64(x) / math.Sqrt(ss/128+1e-6) * float64(weight[i])
	}
	low := math.Max(0, math.Floor(64*math.Log(65536/(32*2*math.Pi))/(2*math.Log(160000))))
	high := math.Min(63, math.Ceil(64*math.Log(65536/(2*math.Pi))/(2*math.Log(160000))))
	for r := 0; r < 64; r += 2 {
		ramp := 1 - min(1, max(0, (float64(r/2)-low)/math.Max(.001, high-low)))
		theta := float64(position) * math.Pow(160000, -float64(r)/64) * (.0625*(1-ramp) + ramp)
		s, c := math.Sincos(theta)
		a, b := values[64+r], values[65+r]
		values[64+r], values[65+r] = a*c-b*s, a*s+b*c
	}
	transformed := make([]float64, 128)
	for i := range transformed {
		for j, v := range values {
			if bits.OnesCount(uint(i&j))%2 == 0 {
				transformed[i] += v
			} else {
				transformed[i] -= v
			}
		}
		transformed[i] /= math.Sqrt(128)
	}
	out := make([]float32, 128)
	levels := [...]float64{0, .5, 1, 1.5, 2, 3, 4, 6}
	for start := 0; start < 128; start += 32 {
		amax := 7.052966104933725e-38
		for _, v := range transformed[start : start+32] {
			amax = math.Max(amax, math.Abs(v))
		}
		scale := math.Exp2(math.Ceil(math.Log2(amax / 6)))
		for i := start; i < start+32; i++ {
			value := math.Min(math.Abs(transformed[i])/scale, 6)
			best := 0
			for j := 1; j < len(levels); j++ {
				d, b := math.Abs(value-levels[j]), math.Abs(value-levels[best])
				if d < b || d == b && j%2 == 0 && best%2 != 0 {
					best = j
				}
			}
			out[i] = float32(math.Copysign(levels[best]*scale, transformed[i]))
		}
	}
	return out
}

func TestMetalIndexerRegressionReference(t *testing.T) {
	pooled, weight := make([]float32, 128), make([]float32, 128)
	for i := range pooled {
		pooled[i] = float32((i*19)%67-33) / 32
		weight[i] = 1 + float32(i%7-3)/16
	}
	for _, position := range []int{0, 4, 2048} {
		var ss float64
		for _, v := range pooled {
			ss += float64(v) * float64(v)
		}
		normalized := make([]float32, 128)
		for i, v := range pooled {
			normalized[i] = v * float32(1/math.Sqrt(ss/128+1e-6)) * weight[i]
		}
		old := append([]float32(nil), normalized...)
		ropeCompressedTail(normalized, position, false)
		indexerQAT(normalized)
		want := metalIndexerRegressionReference(pooled, weight, position)
		if err := metalIndexerRegressionClose(normalized, want); err != nil {
			t.Fatalf("position %d: scalar oracle versus existing CPU path: %v", position, err)
		}
		if position != 0 {
			indexerQAT(old)
			if metalIndexerRegressionClose(old, want) == nil {
				t.Fatalf("position %d: missing rotation was not detected", position)
			}
		}
	}
}
