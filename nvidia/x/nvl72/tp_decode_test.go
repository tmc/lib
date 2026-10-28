package nvl72

import (
	"fmt"
	"math"
	"testing"
)

func TestAllReduceUs(t *testing.T) {
	f := NVLinkFabric{Name: "test", LinkGBps: 1000, HopUs: 1, NVLS: true} // 1 MB/µs
	const s = 1e6
	for _, tt := range []struct {
		algo  AllReduceAlgo
		ranks int
		want  float64
	}{
		{AllReduceOneShot, 1, 0},
		{AllReduceOneShot, 4, 1 + 3},
		{AllReduceTwoShot, 4, 2 + 2*0.75},
		{AllReduceRing, 4, 6 + 2*0.75},
		{AllReduceNVLS, 4, 2 + 1},
		{AllReduceRing, 8, 14 + 2*7.0/8},
	} {
		got, err := f.AllReduceUs(tt.algo, tt.ranks, s)
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(got-tt.want) > 1e-9 {
			t.Errorf("%v over %d ranks = %v µs, want %v", tt.algo, tt.ranks, got, tt.want)
		}
	}
	f.NVLS = false
	if _, err := f.AllReduceUs(AllReduceNVLS, 2, s); err == nil {
		t.Error("NVLS without NVLS succeeded")
	}
	if _, err := GB10().AllReduceUs(AllReduceOneShot, 2, s); err == nil {
		t.Error("all-reduce on a fabric without NVLink succeeded")
	}
}

// TestBestAllReduce checks the tuner's choice at the latency- and
// bandwidth-bound ends.
func TestBestAllReduce(t *testing.T) {
	f := GB300NVLink5()
	if a, _, err := f.BestAllReduce(8, 8192); err != nil || a != AllReduceOneShot {
		t.Errorf("8 KiB over 8 ranks: %v, %v; want one-shot", a, err)
	}
	if a, _, err := f.BestAllReduce(8, 256<<20); err != nil || a != AllReduceNVLS {
		t.Errorf("256 MiB over 8 ranks: %v, %v; want nvls", a, err)
	}
	f.NVLS = false
	if a, _, err := f.BestAllReduce(8, 256<<20); err != nil || a != AllReduceTwoShot {
		t.Errorf("256 MiB over 8 ranks without nvls: %v, %v; want two-shot", a, err)
	}
}

func TestLlamaParams(t *testing.T) {
	for _, tt := range []struct {
		m    DenseLlama
		want float64 // billions
	}{
		{Llama3_8B, 8.03},
		{Llama3_70B, 70.55},
	} {
		if got := tt.m.Params() / 1e9; math.Abs(got-tt.want) > 0.01 {
			t.Errorf("%s: %.3fB parameters, want %.2fB", tt.m.Name, got, tt.want)
		}
	}
}

// TestGB10Calibration ties the memory term to the one device this
// repository measures: Llama-3-8B BF16 decode at TP=1 on GB10 runs at
// about 63 ms per token.
func TestGB10Calibration(t *testing.T) {
	e, err := EstimateTPDecode(GB10(), TPDecodeConfig{Model: Llama3_8B, TP: 1, Context: 32})
	if err != nil {
		t.Fatal(err)
	}
	if ms := e.StepUs / 1e3; ms < 58 || ms > 68 {
		t.Errorf("GB10 Llama-3-8B step %.1f ms, want 63 ± 5", ms)
	}
}

func TestEstimateTPDecode(t *testing.T) {
	f := GB300NVLink5()
	var prev TPDecodeEstimate
	for i, tp := range []int{1, 2, 4, 8} {
		e, err := EstimateTPDecode(f, TPDecodeConfig{Model: Llama3_70B, TP: tp, Context: 4096})
		if err != nil {
			t.Fatal(err)
		}
		if tp == 1 && (e.AllReduceUs != 0 || e.AllReduceAlgo != 0) {
			t.Errorf("tp 1 pays %v µs of all-reduce", e.AllReduceUs)
		}
		if i > 0 {
			if r := prev.WeightUs / e.WeightUs; math.Abs(r-2) > 0.01 {
				t.Errorf("tp %d: weight time ratio %v, want 2", tp, r)
			}
			if e.AllReduceUs <= prev.AllReduceUs {
				t.Errorf("tp %d: all-reduce time %v not above tp %d's %v", tp, e.AllReduceUs, tp/2, prev.AllReduceUs)
			}
		}
		fused, err := EstimateTPDecode(f, TPDecodeConfig{Model: Llama3_70B, TP: tp, Context: 4096, FuseAllReduce: true, Megakernel: true})
		if err != nil {
			t.Fatal(err)
		}
		if fused.StepUs > e.StepUs || fused.GapUs != 0 {
			t.Errorf("tp %d: fused megakernel step %v µs (gaps %v), unfused %v", tp, fused.StepUs, fused.GapUs, e.StepUs)
		}
		prev = e
	}
	// KV heads stop sharding beyond 8 ranks.
	a, _ := EstimateTPDecode(f, TPDecodeConfig{Model: Llama3_8B, TP: 8, Context: 8192})
	b, _ := EstimateTPDecode(f, TPDecodeConfig{Model: DenseLlama{Name: "16kv", Layers: 32, Hidden: 4096, Heads: 32, KVHeads: 16, HeadDim: 128, Intermediate: 14336, Vocab: 128256}, TP: 8, Context: 8192})
	if a.KVUs <= 0 || math.Abs(b.KVUs/a.KVUs-2) > 1e-9 {
		t.Errorf("kv time with 16 kv heads / 8 kv heads = %v, want 2", b.KVUs/a.KVUs)
	}
	for _, c := range []TPDecodeConfig{
		{Model: Llama3_8B, TP: 3},
		{Model: Llama3_8B, TP: 0},
		{Model: Llama3_8B, TP: 2, Context: -1},
	} {
		if _, err := EstimateTPDecode(f, c); err == nil {
			t.Errorf("%+v: no error", c)
		}
	}
}

// ExampleEstimateTPDecode projects batch-1 decode on GB300 NVL72 for
// Llama 3 at a 4096-token context. The figures are modelled, not
// measured.
func ExampleEstimateTPDecode() {
	f := GB300NVLink5()
	for _, m := range []DenseLlama{Llama3_8B, Llama3_70B} {
		for _, tp := range []int{1, 2, 4, 8} {
			e, err := EstimateTPDecode(f, TPDecodeConfig{Model: m, TP: tp, Context: 4096})
			if err != nil {
				fmt.Println(err)
				return
			}
			fe, _ := EstimateTPDecode(f, TPDecodeConfig{Model: m, TP: tp, Context: 4096, FuseAllReduce: true, Megakernel: true})
			fmt.Printf("%-11s tp=%d weights %6.3f ms  all-reduce %5.3f ms  step %6.3f ms  fused+mega %6.3f ms\n",
				m.Name, tp, e.WeightUs/1e3, e.AllReduceUs/1e3, e.StepUs/1e3, fe.StepUs/1e3)
		}
	}
	// Output:
	// Llama-3-8B  tp=1 weights  2.345 ms  all-reduce 0.000 ms  step  2.719 ms  fused+mega  2.429 ms
	// Llama-3-8B  tp=2 weights  1.173 ms  all-reduce 0.196 ms  step  1.765 ms  fused+mega  1.280 ms
	// Llama-3-8B  tp=4 weights  0.586 ms  all-reduce 0.197 ms  step  1.159 ms  fused+mega  0.674 ms
	// Llama-3-8B  tp=8 weights  0.293 ms  all-reduce 0.199 ms  step  0.858 ms  fused+mega  0.373 ms
	// Llama-3-70B tp=1 weights 21.720 ms  all-reduce 0.000 ms  step 22.651 ms  fused+mega 21.929 ms
	// Llama-3-70B tp=2 weights 10.860 ms  all-reduce 0.486 ms  step 12.334 ms  fused+mega 11.129 ms
	// Llama-3-70B tp=4 weights  5.430 ms  all-reduce 0.492 ms  step  6.857 ms  fused+mega  5.652 ms
	// Llama-3-70B tp=8 weights  2.715 ms  all-reduce 0.504 ms  step  4.128 ms  fused+mega  2.923 ms
}
