//go:build linux

package main

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// Frozen preparation from 884fb2c, with only a function rename, an extra
// output pointer, and a copy of the raw scores before softmax. All arithmetic
// remains literal. The unchanged values kernel completes this reference arm.
const cudaAttentionScoreReference = `extern "C" __global__ void ds4_attention_mixed_prepare_reference(const float *q,const float *kv,const float *sinks,float *raw,const float *comp,const unsigned int *allowed,float *scores,float *denominators,float *out,float *unscaled,unsigned int position,unsigned int layer,unsigned int raw_cap,unsigned int n_comp) {
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
  for(unsigned int t=0;t<total;t++)unscaled[head*total+t]=scores[head*total+t];
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
  denominators[head]=denom;
}`

type cudaAttentionScoreResult struct {
	output, cache, scores, weights, denominators []float32
}

func compareCUDAAttentionScores(got, want cudaAttentionScoreResult) error {
	for _, v := range []struct {
		name      string
		got, want []float32
	}{
		{"output", got.output, want.output}, {"cache", got.cache, want.cache},
		{"scores", got.scores, want.scores}, {"weights", got.weights, want.weights},
		{"denominators", got.denominators, want.denominators},
	} {
		if len(v.got) == 0 || len(v.want) == 0 {
			return fmt.Errorf("%s: empty", v.name)
		}
		for arm, values := range [][]float32{v.got, v.want} {
			for i, x := range values {
				if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
					return fmt.Errorf("%s arm %d at %d: non-finite", v.name, arm, i)
				}
			}
		}
		if err := exactFloat32Bits(v.got, v.want); err != nil {
			return fmt.Errorf("%s bits: %w", v.name, err)
		}
	}
	for _, v := range got.denominators {
		if v <= 0 {
			return fmt.Errorf("denominator: non-positive")
		}
	}
	return nil
}

func TestCUDAAttentionScoreComparison(t *testing.T) {
	makeResult := func() cudaAttentionScoreResult {
		return cudaAttentionScoreResult{[]float32{1}, []float32{2}, []float32{3}, []float32{4}, []float32{5}}
	}
	if err := compareCUDAAttentionScores(makeResult(), makeResult()); err != nil {
		t.Fatal(err)
	}
	field := func(r *cudaAttentionScoreResult, name string) *[]float32 {
		switch name {
		case "output":
			return &r.output
		case "cache":
			return &r.cache
		case "scores":
			return &r.scores
		case "weights":
			return &r.weights
		default:
			return &r.denominators
		}
	}
	for _, name := range []string{"output", "cache", "scores", "weights", "denominators"} {
		t.Run(name, func(t *testing.T) {
			for _, arm := range []string{"candidate", "reference"} {
				for _, kind := range []string{"ulp", "nan", "inf", "empty", "length"} {
					t.Run(arm+"/"+kind, func(t *testing.T) {
						got, want := makeResult(), makeResult()
						f := field(&got, name)
						if arm == "reference" {
							f = field(&want, name)
						}
						switch kind {
						case "ulp":
							(*f)[0] = math.Float32frombits(math.Float32bits((*f)[0]) + 1)
						case "nan":
							(*f)[0] = float32(math.NaN())
						case "inf":
							(*f)[0] = float32(math.Inf(1))
						case "empty":
							*f = nil
						case "length":
							*f = append(*f, 0)
						}
						if compareCUDAAttentionScores(got, want) == nil {
							t.Fatal("invalid intermediate accepted")
						}
					})
				}
			}
		})
	}
	for _, name := range []string{"output", "cache", "scores", "weights", "denominators"} {
		got, want := makeResult(), makeResult()
		*field(&got, name), *field(&want, name) = nil, nil
		if compareCUDAAttentionScores(got, want) == nil {
			t.Fatal("identical empty intermediates accepted")
		}
		for _, bad := range []float32{float32(math.NaN()), float32(math.Inf(1))} {
			got, want := makeResult(), makeResult()
			(*field(&got, name))[0], (*field(&want, name))[0] = bad, bad
			if compareCUDAAttentionScores(got, want) == nil {
				t.Fatal("identical non-finite intermediates accepted")
			}
		}
	}
	got, want := makeResult(), makeResult()
	got.denominators[0] = 0
	want.denominators[0] = 0
	if compareCUDAAttentionScores(got, want) == nil {
		t.Fatal("zero denominator accepted")
	}
	got, want = makeResult(), makeResult()
	got.scores[0] = 0
	want.scores[0] = math.Float32frombits(1 << 31)
	if compareCUDAAttentionScores(got, want) == nil {
		t.Fatal("signed zero accepted")
	}
}

func TestCUDAAttentionScoreParallelBits(t *testing.T) {
	r := cudaSynthFoundation(t)
	if err := r.device.WithContext(r.loadMathKernels); err != nil {
		t.Fatal(err)
	}
	compile := func(source, name, symbol string) driver.Function {
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
		fn, err := module.GetFunction(symbol)
		if err != nil {
			t.Fatal(err)
		}
		return fn
	}
	reference := compile(cudaMathSource+cudaAttentionScoreReference, "attention_score_reference", "ds4_attention_mixed_prepare_reference")
	start := strings.Index(cudaMathSource, `extern "C" __global__ void ds4_attention_mixed_scores(`)
	if start < 0 {
		t.Fatal("score kernel missing")
	}
	end := strings.Index(cudaMathSource[start:], "\n}")
	if end < 0 {
		t.Fatal("score kernel end missing")
	}
	body := cudaMathSource[start : start+end+2]
	const anchor = "const unsigned int qoff=head*512;"
	if strings.Count(body, anchor) != 1 {
		t.Fatal("dot-index mutation anchor changed")
	}
	mutantBody := strings.Replace(body, anchor, "const unsigned int qoff=((head+1)%64)*512;", 1)
	mutant := compile(cudaMathSource[:start]+mutantBody+cudaMathSource[start+end+2:], "attention_score_mutant", "ds4_attention_mixed_scores")
	const capacity, maxComp = 128, 513
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
	unscaled := cudaTestAlloc(t, r, scores.Len())
	denominators := cudaTestAlloc(t, r, 64)
	out := cudaTestAlloc(t, r, 32768+512+32768)
	allowed := cudaCompressedMask(t, r, maxComp)
	cudaCompressedUpload(t, dc, comp)
	run := func(t *testing.T, position, n int, maskAll, old bool, scoreFn driver.Function) cudaAttentionScoreResult {
		t.Helper()
		q, kv := cudaAttentionQualificationInputs(position)
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
		// Scores and unscaled storage are deliberately reused across changing extents.
		total := min(position+1, capacity) + n
		var rawScores []float32
		cudaQualificationMeasure(t, r, func() error {
			if old {
				return cudalaunch.Kernel(reference).Grid(1).Block(64).On(r.device.DefaultStream()).Launch(dq, dk, sink.device, dr, dc, allowed, scores, denominators, out, unscaled, uint32(position), uint32(2), uint32(capacity), uint32(n))
			}
			if err := cudalaunch.Kernel(r.mixedAttentionPrepareFn).Grid(1).Block(64).On(r.device.DefaultStream()).Launch(dq, dk, sink.device, dr, dc, allowed, scores, denominators, out, uint32(position), uint32(2), uint32(capacity), uint32(n)); err != nil {
				return err
			}
			return cudalaunch.Kernel(scoreFn).Grid(64, uint32((total+63)/64)).Block(64).On(r.device.DefaultStream()).Launch(dr, dc, allowed, scores, out, uint32(position), uint32(capacity), uint32(n))
		})
		if old {
			rawScores = cudaTestRead(t, unscaled)[:64*total]
		} else {
			rawScores = cudaTestRead(t, scores)[:64*total]
		}
		// This diagnostic boundary exposes raw scores; it is not a performance run.
		cudaQualificationMeasure(t, r, func() error {
			if !old {
				if err := cudalaunch.Kernel(r.mixedAttentionNormalizeFn).Grid(1).Block(64).On(r.device.DefaultStream()).Launch(sink.device, allowed, scores, denominators, uint32(position), uint32(capacity), uint32(n)); err != nil {
					return err
				}
			}
			return cudalaunch.Kernel(r.mixedAttentionValuesFn).Grid(64).Block(256).On(r.device.DefaultStream()).Launch(dr, dc, allowed, scores, denominators, out, uint32(position), uint32(capacity), uint32(n))
		})
		return cudaAttentionScoreResult{cudaTestRead(t, out), cudaTestRead(t, dr), rawScores, cudaTestRead(t, scores)[:64*total], cudaTestRead(t, denominators)}
	}
	for _, position := range []int{127, 128, 129} {
		for _, tc := range []struct {
			n      int
			masked bool
		}{{1, false}, {512, false}, {513, false}, {0, false}, {513, true}} {
			t.Run(fmt.Sprintf("position%d/comp%d/masked%t", position, tc.n, tc.masked), func(t *testing.T) {
				want := run(t, position, tc.n, tc.masked, true, r.mixedAttentionScoresFn)
				got := run(t, position, tc.n, tc.masked, false, r.mixedAttentionScoresFn)
				if err := exactFloat32Bits(got.scores, want.scores); err != nil {
					t.Fatalf("scores bits: %v", err)
				}
				if err := compareCUDAAttentionScores(got, want); err != nil {
					t.Fatal(err)
				}
			})
		}
	}
	t.Run("mutation/dot-index", func(t *testing.T) {
		want := run(t, 129, 513, false, true, r.mixedAttentionScoresFn)
		got := run(t, 129, 513, false, false, mutant)
		if err := exactFloat32Bits(got.cache, want.cache); err != nil {
			t.Fatalf("dot mutation changed cache: %v", err)
		}
		if err := compareCUDAAttentionScores(got, got); err != nil {
			t.Fatalf("mutant invalid instead of discriminating: %v", err)
		}
		if exactFloat32Bits(got.scores, want.scores) == nil {
			t.Fatal("dot-index mutation survived raw score comparison")
		}
		if compareCUDAAttentionScores(got, want) == nil {
			t.Fatal("dot-index mutation survived comparison")
		}
	})
}
