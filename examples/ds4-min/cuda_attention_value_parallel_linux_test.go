//go:build linux

package main

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// Frozen memoized kernel from weight-cache-v3, before value parallelization.
const cudaAttentionValueReference = `extern "C" __global__ void ds4_attention_mixed(const float *q,const float *kv,const float *sinks,float *raw,const float *comp,const unsigned int *allowed,float *scores,float *out,unsigned int position,unsigned int layer,unsigned int raw_cap,unsigned int n_comp) {
 const unsigned int head=threadIdx.x;if(blockIdx.x || head>=64)return;
  const unsigned int qoff=head*512;
  for(unsigned int i=0;i<512;i++)out[qoff+i]=q[qoff+i];
  if(head==0)for(unsigned int i=0;i<512;i++)out[32768+i]=kv[i];
  for(unsigned int r=0;r<64;r+=2){
    const unsigned int qi=qoff+448+r;float theta=ds4_compressed_theta(position,r),c=cosf(theta),s=sinf(theta),a=out[qi],b=out[qi+1];
    out[qi]=a*c-b*s;out[qi+1]=a*s+b*c;
    if(head==0){unsigned int ki=32768+448+r;a=out[ki];b=out[ki+1];out[ki]=a*c-b*s;out[ki+1]=a*s+b*c;}
  }
  __syncthreads();
  if(head==0){
    for(unsigned int off=0;off<448;off+=64){
      float amax=1.0e-4f;for(unsigned int i=0;i<64;i++)amax=fmaxf(amax,fabsf(out[32768+off+i]));
      const float scale=exp2f(ceilf(log2f(amax/448.0f)));
      for(unsigned int i=0;i<64;i++)out[32768+off+i]=ds4_e4m3fn_round(fminf(fmaxf(out[32768+off+i]/scale,-448.0f),448.0f))*scale;
    }
    const unsigned int row=position%raw_cap;
    for(unsigned int i=0;i<512;i++)raw[row*512+i]=ds4_half_round(out[32768+i]);
  }
  __syncthreads();
  const unsigned int n_raw=min(position+1,raw_cap),first=position+1-n_raw,total=n_raw+n_comp;
  float maximum=sinks[head];
  for(unsigned int t=0;t<n_raw;t++){
    const unsigned int row=(first+t)%raw_cap;float score=0.0f;
    // Both dot operands go through f16 as the oracle does; see ds4_attention_raw.
    for(unsigned int i=0;i<512;i++)score+=ds4_half_round(out[qoff+i])*raw[row*512+i];
    score*=0.04419417382415922f;scores[head*total+t]=score;maximum=fmaxf(maximum,score);
  }
  for(unsigned int cidx=0;cidx<n_comp;cidx++){
    float score=-1.0e30f;
    if(allowed[cidx]){score=0.0f;for(unsigned int i=0;i<512;i++)score+=ds4_half_round(out[qoff+i])*comp[cidx*512+i];score*=0.04419417382415922f;maximum=fmaxf(maximum,score);}
    scores[head*total+n_raw+cidx]=score;
  }
  float denom=expf(sinks[head]-maximum);
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
    for(unsigned int cidx=0;cidx<n_comp;cidx++)if(allowed[cidx])value+=scores[head*total+n_raw+cidx]*comp[cidx*512+i];
    out[33280+qoff+i]=value/denom;
  }
  for(unsigned int r=0;r<64;r+=2){
    const unsigned int oi=33280+qoff+448+r;float theta=ds4_compressed_theta(position,r),c=cosf(theta),s=-sinf(theta),a=out[oi],b=out[oi+1];
    out[oi]=a*c-b*s;out[oi+1]=a*s+b*c;
  }
  (void)layer;
}`

func TestCUDAAttentionValueParallelBits(t *testing.T) {
	r := cudaSynthFoundation(t)
	r.disableMixedAttentionFused = true
	if err := r.device.WithContext(r.loadMathKernels); err != nil {
		t.Fatal(err)
	}
	// Only dependencies come from current source. The old kernel body is literal
	// evidence retained above, never reconstructed by reversing the candidate.
	start := strings.Index(cudaMathSource, `extern "C" __global__ void ds4_attention_mixed(`)
	if start < 0 {
		t.Fatal("reference kernel boundary missing")
	}
	end := strings.Index(cudaMathSource[start:], "\n}")
	if end < 0 {
		t.Fatal("reference kernel end missing")
	}
	referenceSource := cudaMathSource[:start] + cudaAttentionValueReference + cudaMathSource[start+end+2:]
	reference := cudaAttentionKernel(t, r, referenceSource, "attention_value_reference")
	type kernels struct{ prepare, values driver.Function }
	compile := func(source, name string) kernels {
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
		prepare, err := module.GetFunction("ds4_attention_mixed_prepare")
		if err != nil {
			t.Fatal(err)
		}
		values, err := module.GetFunction("ds4_attention_mixed_values")
		if err != nil {
			t.Fatal(err)
		}
		return kernels{prepare, values}
	}
	candidate := kernels{r.mixedAttentionPrepareFn, r.mixedAttentionValuesFn}
	mutants := []struct {
		name, old, new string
		functions      kernels
	}{
		{name: "index", old: "i=2*pair,qoff=head*512", new: "i=2*pair,qoff=((head+1)%64)*512"},
		{name: "denominator", old: "const float denom=denominators[head];", new: "const float denom=denominators[head]+(head==63?0.125f:0.0f);"},
		{name: "rotation", old: "theta=ds4_compressed_theta(position,i-448),c=cosf(theta),s=-sinf(theta)", new: "theta=ds4_compressed_theta(position,i-448),c=cosf(theta),s=sinf(theta)"},
	}
	for i := range mutants {
		m := &mutants[i]
		if strings.Count(cudaMathSource, m.old) != 1 {
			t.Fatalf("%s mutation anchor changed", m.name)
		}
		m.functions = compile(strings.Replace(cudaMathSource, m.old, m.new, 1), "attention_value_mutant_"+m.name)
	}
	const capacity = 128
	const maxComp = 513
	raw := make([]float32, capacity*512)
	for i := range raw {
		raw[i] = float32((i*13+i/512*7)%97-48) / 64
	}
	comp := make([]float32, maxComp*512)
	for i := range comp {
		comp[i] = float32((i*7+i/512*11)%89-44) / 64
	}
	sinks := make([]float32, 64)
	for i := range sinks {
		sinks[i] = float32(i%7-3) / 4
	}
	sink := cudaCompressedTensor(t, r, "sinks", sinks)
	dq, dk := cudaTestAlloc(t, r, 64*512), cudaTestAlloc(t, r, 512)
	dr, dc := cudaTestAlloc(t, r, len(raw)), cudaTestAlloc(t, r, len(comp))
	scores := cudaTestAlloc(t, r, 64*(capacity+maxComp))
	denominators := cudaTestAlloc(t, r, 64)
	out := cudaTestAlloc(t, r, 32768+512+32768)
	allowed := cudaCompressedMask(t, r, maxComp)
	cudaCompressedUpload(t, dc, comp)
	finite := func(t *testing.T, name string, values []float32) {
		t.Helper()
		for i, v := range values {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("%s[%d] non-finite", name, i)
			}
		}
	}
	run := func(t *testing.T, fn kernels, position, n int, maskAll bool, old bool) ([]float32, []float32) {
		t.Helper()
		q, kv := cudaAttentionQualificationInputs(position)
		// Perturb each head and coordinate to make mapping mutants observable.
		for i := range q {
			q[i] += float32((i*17+i/512*5)%31-15) / 128
		}
		cudaCompressedUpload(t, dq, q)
		cudaCompressedUpload(t, dk, kv)
		cudaCompressedUpload(t, dr, raw)
		mask := make([]uint32, maxComp)
		for i := 0; i < n; i++ {
			if !maskAll && i%2 == 0 {
				mask[i] = 1
			}
		}
		if err := allowed.CopyFromHost(mask); err != nil {
			t.Fatal(err)
		}
		poison := make([]float32, out.Len())
		for i := range poison {
			poison[i] = float32(math.NaN())
		}
		cudaCompressedUpload(t, out, poison)
		dp := make([]float32, 64)
		for i := range dp {
			dp[i] = float32(math.NaN())
		}
		cudaCompressedUpload(t, denominators, dp)
		// scores is intentionally reused, including from n=513 to n=0.
		savedOld, savedPrepare, savedValues := r.mixedAttentionFn, r.mixedAttentionPrepareFn, r.mixedAttentionValuesFn
		defer func() {
			r.mixedAttentionFn, r.mixedAttentionPrepareFn, r.mixedAttentionValuesFn = savedOld, savedPrepare, savedValues
		}()
		if old {
			r.mixedAttentionFn = reference
		} else {
			r.mixedAttentionPrepareFn, r.mixedAttentionValuesFn = fn.prepare, fn.values
		}
		cudaQualificationMeasure(t, r, func() error {
			if old {
				return r.encodeMixedAttention(dq, dk, sink, dr, dc, n, allowed, scores, out, position, 2)
			}
			return r.encodeMixedAttentionParallel(dq, dk, sink, dr, dc, n, allowed, scores, denominators, out, position, 2)
		})
		got, cache := cudaTestRead(t, out), cudaTestRead(t, dr)
		finite(t, "output", got)
		finite(t, "raw cache", cache)
		if !old {
			d := cudaTestRead(t, denominators)
			finite(t, "denominators", d)
			for i, v := range d {
				if v <= 0 {
					t.Fatalf("denominator[%d]=%g", i, v)
				}
			}
		}
		return got, cache
	}
	for _, position := range []int{127, 128, 129} {
		for _, tc := range []struct {
			n      int
			masked bool
		}{{1, false}, {512, false}, {513, false}, {0, false}, {513, true}} {
			t.Run(fmt.Sprintf("position%d/comp%d/masked%t", position, tc.n, tc.masked), func(t *testing.T) {
				want, wantRaw := run(t, candidate, position, tc.n, tc.masked, true)
				got, gotRaw := run(t, candidate, position, tc.n, tc.masked, false)
				if err := exactFloat32Bits(got, want); err != nil {
					t.Fatalf("output bits: %v", err)
				}
				if err := exactFloat32Bits(gotRaw, wantRaw); err != nil {
					t.Fatalf("raw cache bits: %v", err)
				}
			})
		}
	}
	want, wantRaw := run(t, candidate, 129, 513, false, true)
	for _, m := range mutants {
		t.Run("mutation/"+m.name, func(t *testing.T) {
			got, gotRaw := run(t, m.functions, 129, 513, false, false)
			if err := exactFloat32Bits(gotRaw, wantRaw); err != nil {
				t.Fatalf("mutation changed preparation/cache: %v", err)
			}
			if exactFloat32Bits(got, want) == nil {
				t.Fatal("candidate-only mutation survived exact output comparison")
			}
		})
	}
}
