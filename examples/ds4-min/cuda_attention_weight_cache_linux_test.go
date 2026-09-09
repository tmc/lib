//go:build linux

package main

import (
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

const cudaAttentionWeightCacheNew = `  float denom=expf(sinks[head]-maximum);
  // Retain each softmax weight. The value loop consumes the same float that
  // the denominator did instead of evaluating expf again for every value
  // dimension.
  for(unsigned int t=0;t<n_raw;t++){
    const unsigned int i=head*total+t;const float weight=expf(scores[i]-maximum);scores[i]=weight;denom+=weight;
  }
  for(unsigned int cidx=0;cidx<n_comp;cidx++)if(allowed[cidx]){
    const unsigned int i=head*total+n_raw+cidx;const float weight=expf(scores[i]-maximum);scores[i]=weight;denom+=weight;
  }
  for(unsigned int i=0;i<512;i++){
    float value=0.0f;
    for(unsigned int t=0;t<n_raw;t++){const unsigned int row=(first+t)%raw_cap;value+=scores[head*total+t]*raw[row*512+i];}
    for(unsigned int cidx=0;cidx<n_comp;cidx++)if(allowed[cidx])value+=scores[head*total+n_raw+cidx]*comp[cidx*512+i];`

const cudaAttentionWeightCacheOld = `  float denom=expf(sinks[head]-maximum);
  for(unsigned int t=0;t<n_raw;t++)denom+=expf(scores[head*total+t]-maximum);
  for(unsigned int cidx=0;cidx<n_comp;cidx++)if(allowed[cidx])denom+=expf(scores[head*total+n_raw+cidx]-maximum);
  for(unsigned int i=0;i<512;i++){
    float value=0.0f;
    for(unsigned int t=0;t<n_raw;t++){const unsigned int row=(first+t)%raw_cap;value+=expf(scores[head*total+t]-maximum)*raw[row*512+i];}
    for(unsigned int cidx=0;cidx<n_comp;cidx++)if(allowed[cidx])value+=expf(scores[head*total+n_raw+cidx]-maximum)*comp[cidx*512+i];`

func cudaAttentionKernel(t *testing.T, r *cudaFoundation, source, name string) driver.Function {
	t.Helper()
	program, err := nvrtcx.Compile(source, nvrtcx.WithName(name+".cu"), nvrtcx.WithArch(r.major, r.minor))
	if err != nil {
		t.Fatal(err)
	}
	module, err := program.Load(r.device)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := module.Unload(); err != nil {
			t.Error(err)
		}
	})
	fn, err := module.GetFunction("ds4_attention_mixed")
	if err != nil {
		t.Fatal(err)
	}
	return fn
}

func TestCUDAAttentionWeightCacheBits(t *testing.T) {
	r := cudaSynthFoundation(t)
	if err := r.device.WithContext(r.loadMathKernels); err != nil {
		t.Fatal(err)
	}
	candidate := r.mixedAttentionFn
	if strings.Count(cudaMathSource, cudaAttentionWeightCacheNew) != 1 {
		t.Fatal("attention weight-cache source anchor changed")
	}
	baselineSource := strings.Replace(cudaMathSource, cudaAttentionWeightCacheNew, cudaAttentionWeightCacheOld, 1)
	baseline := cudaAttentionKernel(t, r, baselineSource, "attention_weight_baseline")
	mutantSource := strings.Replace(cudaMathSource, "scores[i]=weight;denom+=weight;", "scores[i]=weight+0.125f;denom+=weight;", 2)
	if mutantSource == cudaMathSource {
		t.Fatal("attention weight mutation anchor changed")
	}
	mutant := cudaAttentionKernel(t, r, mutantSource, "attention_weight_mutant")

	const capacity = 128
	q, kv := cudaAttentionQualificationInputs(129)
	raw := make([]float32, capacity*512)
	for position := 2; position <= 128; position++ {
		_, row := cudaAttentionQualificationInputs(position)
		cudaCompressedRotateReference(row, position)
		cudaCompressedFP8Reference(row)
		for i := range row {
			row[i] = cudaAttentionQualificationRound(row[i], 10, -14)
		}
		copy(raw[(position%capacity)*512:], row)
	}
	comp := make([]float32, 3*512)
	for i := range comp {
		comp[i] = float32((i*7)%43-21) / 16
	}
	sinks := make([]float32, 64)
	for h := range sinks {
		sinks[h] = float32(h%7-3) / 4
	}
	sink := cudaCompressedTensor(t, r, "sinks", sinks)
	dq, dk := cudaTestAlloc(t, r, len(q)), cudaTestAlloc(t, r, len(kv))
	dr, dc := cudaTestAlloc(t, r, len(raw)), cudaTestAlloc(t, r, len(comp))
	ds, out := cudaTestAlloc(t, r, 64*(capacity+3)), cudaTestAlloc(t, r, 32768+512+32768)
	allowed := cudaCompressedMask(t, r, 3)
	cudaCompressedUpload(t, dq, q)
	cudaCompressedUpload(t, dk, kv)
	cudaCompressedUpload(t, dc, comp)

	run := func(fn driver.Function, n int, mask []uint32) ([]float32, []float32) {
		t.Helper()
		if err := allowed.CopyFromHost(mask); err != nil {
			t.Fatal(err)
		}
		original := r.mixedAttentionFn
		r.mixedAttentionFn = fn
		defer func() { r.mixedAttentionFn = original }()
		cudaCompressedUpload(t, dr, raw)
		poison := make([]float32, out.Len())
		for i := range poison {
			poison[i] = float32(math.NaN())
		}
		cudaCompressedUpload(t, out, poison)
		cudaQualificationMeasure(t, r, func() error {
			return r.encodeMixedAttention(dq, dk, sink, dr, dc, n, allowed, ds, out, 129, 2)
		})
		return cudaTestRead(t, out), cudaTestRead(t, dr)
	}

	requireFinite := func(name string, values []float32) {
		t.Helper()
		for i, value := range values {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("%s[%d] is non-finite", name, i)
			}
		}
	}
	for _, tc := range []struct {
		name string
		n    int
		mask []uint32
	}{
		{name: "masked", n: 3, mask: []uint32{1, 0, 1}},
		{name: "no compressed after reuse", n: 0, mask: []uint32{1, 0, 0}},
		{name: "all compressed masked", n: 3, mask: []uint32{0, 0, 0}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			wantOut, wantRaw := run(baseline, tc.n, tc.mask)
			gotOut, gotRaw := run(candidate, tc.n, tc.mask)
			requireFinite("baseline output", wantOut)
			requireFinite("candidate output", gotOut)
			requireFinite("baseline cache", wantRaw)
			requireFinite("candidate cache", gotRaw)
			if err := exactFloat32Bits(gotOut, wantOut); err != nil {
				t.Fatalf("weight cache changed output bits: %v", err)
			}
			if err := exactFloat32Bits(gotRaw, wantRaw); err != nil {
				t.Fatalf("weight cache changed cache bits: %v", err)
			}
		})
	}
	wantOut, _ := run(baseline, 3, []uint32{1, 0, 1})
	mutantOut, _ := run(mutant, 3, []uint32{1, 0, 1})
	requireFinite("mutant output", mutantOut)
	if exactFloat32Bits(mutantOut, wantOut) == nil {
		t.Fatal("weight-cache mutation passed exact output gate")
	}
}
