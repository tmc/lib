//go:build linux && !tpu

package main

// Qualification for LLAMA_INFER_FP8_KV=1. Both device tests need a CUDA
// device and LIB_CUDA_HARDWARE_TEST=1 and have never been executed by their
// author. A skip is not a pass.

import (
	"context"
	"math"
	"math/rand/v2"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudamem"
)

var fp8TestConfig = Config{
	Name:       "FP8KVTestModel",
	Dim:        256,
	HiddenDim:  512,
	NumLayers:  2,
	NumHeads:   4,
	NumKVHeads: 2,
	HeadDim:    64,
	VocabSize:  256,
	MaxSeqLen:  64,
	Eps:        1e-5,
	RopeTheta:  10000.0,
	Precision:  "fp32",
	TestMode:   true,
}

// TestFP8KVKernelMatchesHost stores random rows through the FP8 store kernel,
// dequantizes them, and compares every element with the host codec: they must
// agree bit for bit, since both perform the same integer operations.
func TestFP8KVKernelMatchesHost(t *testing.T) {
	requireCUDAHardware(t)
	cfg := fp8TestConfig
	t.Setenv("LLAMA_INFER_CUDA_GRAPH", "")
	t.Setenv("LLAMA_INFER_PAGED_KV", "")
	t.Setenv("LLAMA_INFER_FP8_KV", "1")
	t.Setenv("LLAMA_INFER_FP8_KV_SCALE", "0.5")
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatalf("fp8 engine: %v", err)
	}
	rt := eng.fp8
	if rt == nil || eng.fp8Main == nil || eng.kvCache != nil {
		t.Fatal("fp8 engine did not replace the dense cache")
	}
	cache, err := rt.newCache()
	if err != nil {
		t.Fatal(err)
	}
	defer cache.Close()

	kvDim := cfg.NumKVHeads * cfg.HeadDim
	const rows = 5
	rng := rand.New(rand.NewPCG(7, 11))
	// Magnitudes across the whole E4M3 range and past it, so subnormals,
	// normals and saturation are all exercised, in both signs.
	sample := func() float32 {
		return float32(math.Copysign(math.Exp(rng.Float64()*16-10), rng.Float64()-0.5))
	}
	k := make([]float32, rows*kvDim)
	v := make([]float32, rows*kvDim)
	for i := range k {
		k[i], v[i] = sample(), sample()
	}
	k[0], v[0], k[1], v[1] = 0, float32(math.Copysign(0, -1)), 1000, -1000
	devK, err := cudamem.AllocTracked[float32](eng.dev, eng.memory, rows*kvDim)
	if err != nil {
		t.Fatal(err)
	}
	defer devK.Free()
	devV, err := cudamem.AllocTracked[float32](eng.dev, eng.memory, rows*kvDim)
	if err != nil {
		t.Fatal(err)
	}
	defer devV.Free()
	if err := devK.CopyFromHost(k); err != nil {
		t.Fatal(err)
	}
	if err := devV.CopyFromHost(v); err != nil {
		t.Fatal(err)
	}
	for row := 0; row < rows; row++ {
		if err := rt.storeRow(eng.stream, nil, cache, 1, devK.Slice(row*kvDim, (row+1)*kvDim), devV.Slice(row*kvDim, (row+1)*kvDim), row); err != nil {
			t.Fatalf("store row %d: %v", row, err)
		}
	}
	if err := rt.dequantize(eng.stream, nil, cache, 1, rows); err != nil {
		t.Fatal(err)
	}
	if err := eng.stream.Synchronize(context.Background()); err != nil {
		t.Fatal(err)
	}
	gotK := make([]float32, rows*kvDim)
	gotV := make([]float32, rows*kvDim)
	if err := rt.kScratch.Slice(0, len(gotK)).CopyToHost(gotK); err != nil {
		t.Fatal(err)
	}
	if err := rt.vScratch.Slice(0, len(gotV)).CopyToHost(gotV); err != nil {
		t.Fatal(err)
	}
	for i := range k {
		if want := fp8KVRoundTrip(k[i], rt.scale); math.Float32bits(gotK[i]) != math.Float32bits(want) {
			t.Errorf("K element %d: stored %g, read %g, host codec says %g", i, k[i], gotK[i], want)
		}
		if want := fp8KVRoundTrip(v[i], rt.scale); math.Float32bits(gotV[i]) != math.Float32bits(want) {
			t.Errorf("V element %d: stored %g, read %g, host codec says %g", i, v[i], gotV[i], want)
		}
	}
}

// TestFP8KVEngineTracksDense compares an FP8 engine with a dense engine on
// the single-stream path and on the batch path. E4M3 keeps three mantissa
// bits, so the bound is loose: the relative L2 distance between the two
// engines' final logits must stay under 0.1. Token agreement is reported, not
// required. A real model's tolerance is unmeasured; this pins that the path
// computes attention over what it stored rather than over garbage.
func TestFP8KVEngineTracksDense(t *testing.T) {
	requireCUDAHardware(t)
	cfg := fp8TestConfig
	t.Setenv("LLAMA_INFER_CUDA_GRAPH", "")
	t.Setenv("LLAMA_INFER_PAGED_KV", "")
	t.Setenv("LLAMA_INFER_FP8_KV", "")
	dense, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatalf("dense engine: %v", err)
	}
	t.Setenv("LLAMA_INFER_FP8_KV", "1")
	t.Setenv("LLAMA_INFER_FP8_KV_SCALE", "")
	fp8, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatalf("fp8 engine: %v", err)
	}
	if fp8 == dense || fp8.fp8 == nil || fp8.fp8Main == nil || fp8.kvCache != nil || dense.fp8 != nil {
		t.Fatal("engine cache did not separate the fp8 and dense engines")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	relL2 := func(t *testing.T, what string, got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s: %d logits, want %d", what, len(got), len(want))
		}
		var num, den float64
		for i := range got {
			if math.IsNaN(float64(got[i])) {
				t.Fatalf("%s: logit %d is NaN", what, i)
			}
			d := float64(got[i]) - float64(want[i])
			num += d * d
			den += float64(want[i]) * float64(want[i])
		}
		rel := math.Sqrt(num / math.Max(den, 1e-30))
		t.Logf("%s: relative L2 distance %.4f", what, rel)
		if rel > 0.1 {
			t.Fatalf("%s: relative L2 distance %.4f exceeds 0.1", what, rel)
		}
	}
	agreement := func(t *testing.T, what string, got, want []int32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s: %d tokens, want %d", what, len(got), len(want))
		}
		same := 0
		for i := range got {
			if got[i] == want[i] {
				same++
			}
		}
		t.Logf("%s: %d of %d tokens agree (fp8 %v, dense %v)", what, same, len(got), got, want)
	}

	const prompt = "fp8 kv"
	wantTokens, err := dense.GenerateTokenIDs(ctx, cfg, prompt, 8)
	if err != nil {
		t.Fatalf("dense generate: %v", err)
	}
	wantLogits, err := dense.ReadLogits()
	if err != nil {
		t.Fatal(err)
	}
	gotTokens, err := fp8.GenerateTokenIDs(ctx, cfg, prompt, 8)
	if err != nil {
		t.Fatalf("fp8 generate: %v", err)
	}
	gotLogits, err := fp8.ReadLogits()
	if err != nil {
		t.Fatal(err)
	}
	agreement(t, "single-stream", gotTokens, wantTokens)
	relL2(t, "single-stream logits", gotLogits, wantLogits)

	prompts := [][]int32{{10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170}, {60, 70, 80}}
	run := func(t *testing.T, eng *ModelEngine) ([]int32, []float32) {
		t.Helper()
		sp, err := newGPUBatchScratchpad(eng.dev, eng.memory, &eng.mCfg, len(prompts))
		if err != nil {
			t.Fatal(err)
		}
		defer sp.Free()
		caches := make([]*SequenceKVCache, len(prompts))
		tokens := make([]int32, len(prompts))
		positions := make([]int, len(prompts))
		var out []int32
		for i, p := range prompts {
			c, err := eng.NewSequenceKV(cfg)
			if err != nil {
				t.Fatal(err)
			}
			defer c.Free()
			next, _, err := gpuPrefillPassKVMeasured(eng, cfg, c, p)
			if err != nil {
				t.Fatalf("prefill sequence %d: %v", i, err)
			}
			caches[i], tokens[i], positions[i] = c, next, len(p)
			out = append(out, next)
		}
		for step := 0; step < 4; step++ {
			if err := eng.bindKVBatch(sp, caches, positions); err != nil {
				t.Fatalf("step %d bind: %v", step, err)
			}
			next, _, err := GPUForwardBatchKV(ctx, eng, sp, caches, tokens, positions)
			if err != nil {
				t.Fatalf("step %d: %v", step, err)
			}
			out = append(out, next...)
			for i := range tokens {
				tokens[i] = next[i]
				positions[i]++
			}
		}
		logits := make([]float32, len(prompts)*cfg.VocabSize)
		if err := sp.Logits.Slice(0, len(logits)).CopyToHost(logits); err != nil {
			t.Fatal(err)
		}
		return out, logits
	}
	wantTokens, wantLogits = run(t, dense)
	gotTokens, gotLogits = run(t, fp8)
	agreement(t, "batch", gotTokens, wantTokens)
	relL2(t, "batch logits", gotLogits, wantLogits)
}

// TestFP8KVRefusals pins the exclusions with the other compile-time switches.
// No device is needed: the refusal happens before one is opened.
func TestFP8KVRefusals(t *testing.T) {
	cfg := Config{Name: "FP8KVRefusal", Dim: 8, HiddenDim: 16, NumLayers: 1, NumHeads: 1, NumKVHeads: 1, HeadDim: 8, VocabSize: 8, MaxSeqLen: 8, TestMode: true}
	for _, test := range []struct{ other, want string }{
		{"LLAMA_INFER_PAGED_KV", "float32"},
		{"LLAMA_INFER_CUDA_GRAPH", "graph-captured"},
	} {
		t.Run(test.other, func(t *testing.T) {
			t.Setenv("LLAMA_INFER_PAGED_KV", "")
			t.Setenv("LLAMA_INFER_CUDA_GRAPH", "")
			t.Setenv("LLAMA_INFER_FP8_KV", "1")
			t.Setenv(test.other, "1")
			_, err := GetOrCompileEngine(cfg)
			if err == nil || !strings.Contains(err.Error(), "exclusive") || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("GetOrCompileEngine with LLAMA_INFER_FP8_KV=1 and %s=1: err = %v, want the exclusivity refusal", test.other, err)
			}
		})
	}
}
