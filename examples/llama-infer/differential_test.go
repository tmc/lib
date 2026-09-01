package main

import (
	"math"
	"testing"
)

func TestDifferentialForwardPass(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	mCfg := ModelConfig{
		HiddenSize:        cfg.Dim,
		IntermediateSize:  cfg.HiddenDim,
		NumHiddenLayers:   cfg.NumLayers,
		NumAttentionHeads: cfg.NumHeads,
		NumKeyValueHeads:  cfg.NumKVHeads,
		VocabSize:         cfg.VocabSize,
		RmsNormEps:        cfg.Eps,
		RopeTheta:         cfg.RopeTheta,
		MaxSeqLen:         cfg.MaxSeqLen,
	}

	weights := GenerateDeterministicWeights(&mCfg)
	headDim := mCfg.HeadDim()
	kvDim := mCfg.NumKeyValueHeads * headDim
	cpuKV := NewKVCacheState(mCfg.NumHiddenLayers, mCfg.MaxSeqLen, kvDim)

	// Step 0 CPU Reference
	cpuLogits0 := Forward(weights, &mCfg, cpuKV, 42, 0)
	cpuTok0 := sampleArgmax(cpuLogits0)

	// Step 0 GPU Engine
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatalf("GetOrCompileEngine failed: %v", err)
	}

	gpuTok0 := GPUForwardPass(eng, cfg, 42, 0)
	gpuLogits0, err := eng.ReadLogits()
	if err != nil {
		t.Fatalf("ReadLogits failed: %v", err)
	}

	if len(gpuLogits0) != len(cpuLogits0) {
		t.Fatalf("logits length mismatch: GPU %d, CPU %d", len(gpuLogits0), len(cpuLogits0))
	}

	var maxAbsDiff float32
	var maxRelDiff float32
	for i := range cpuLogits0 {
		diff := float32(math.Abs(float64(gpuLogits0[i] - cpuLogits0[i])))
		if diff > maxAbsDiff {
			maxAbsDiff = diff
		}
		rel := diff / (float32(math.Abs(float64(cpuLogits0[i]))) + 1e-6)
		if rel > maxRelDiff {
			maxRelDiff = rel
		}
	}
	t.Logf("Step 0 Logit Diff: MaxAbs=%f, MaxRel=%f (CPU tok: %d, GPU tok: %d)", maxAbsDiff, maxRelDiff, cpuTok0, gpuTok0)
	if maxAbsDiff > 1e-4 {
		t.Fatalf("Step 0 GPU logits exceed absolute tolerance 1e-4 (maxAbs = %f)", maxAbsDiff)
	}
	if maxRelDiff > 1e-3 {
		t.Fatalf("Step 0 GPU logits exceed relative tolerance 1e-3 (maxRel = %f)", maxRelDiff)
	}

	// Step 1 CPU Reference
	cpuLogits1 := Forward(weights, &mCfg, cpuKV, cpuTok0, 1)
	cpuTok1 := sampleArgmax(cpuLogits1)

	// Step 1 GPU Engine
	gpuTok1 := GPUForwardPass(eng, cfg, gpuTok0, 1)
	gpuLogits1, err := eng.ReadLogits()
	if err != nil {
		t.Fatalf("ReadLogits step 1 failed: %v", err)
	}

	var maxAbsDiff1 float32
	var maxRelDiff1 float32
	for i := range cpuLogits1 {
		diff := float32(math.Abs(float64(gpuLogits1[i] - cpuLogits1[i])))
		if diff > maxAbsDiff1 {
			maxAbsDiff1 = diff
		}
		rel := diff / (float32(math.Abs(float64(cpuLogits1[i]))) + 1e-6)
		if rel > maxRelDiff1 {
			maxRelDiff1 = rel
		}
	}
	t.Logf("Step 1 Logit Diff: MaxAbs=%f, MaxRel=%f (CPU tok: %d, GPU tok: %d)", maxAbsDiff1, maxRelDiff1, cpuTok1, gpuTok1)
	if maxAbsDiff1 > 1e-4 {
		t.Fatalf("Step 1 GPU logits exceed absolute tolerance 1e-4 (maxAbs = %f)", maxAbsDiff1)
	}
	if maxRelDiff1 > 1e-3 {
		t.Fatalf("Step 1 GPU logits exceed relative tolerance 1e-3 (maxRel = %f)", maxRelDiff1)
	}
}

func TestWeightMutationAffectsOutput(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatalf("GetOrCompileEngine failed: %v", err)
	}

	// 1. Run fixed input and weights
	tok0 := GPUForwardPass(eng, cfg, 10, 0)
	logits0, err := eng.ReadLogits()
	if err != nil {
		t.Fatalf("ReadLogits initial failed: %v", err)
	}
	if len(logits0) == 0 {
		t.Fatal("no logits read from GPU")
	}

	// 2. Mutate one uploaded weight tensor element (token 10 embedding)
	restore, err := eng.MutateWeightForTest(-1, 10*cfg.Dim, 5.0)
	if err != nil {
		t.Fatalf("MutateWeightForTest failed: %v", err)
	}

	// 3. Rerun identical input and state
	tokMutated := GPUForwardPass(eng, cfg, 10, 0)
	logitsMutated, err := eng.ReadLogits()
	if err != nil {
		t.Fatalf("ReadLogits mutated failed: %v", err)
	}

	_ = tok0
	_ = tokMutated

	// 4. Require logits to change
	diffFound := false
	for i := range logits0 {
		if math.Abs(float64(logits0[i]-logitsMutated[i])) > 1e-4 {
			diffFound = true
			break
		}
	}
	if !diffFound {
		t.Fatal("GPU engine output did not change when weight tensor was mutated!")
	}

	// 5. Restore the tensor and require the original result within tolerance
	restore()
	_ = GPUForwardPass(eng, cfg, 10, 0)
	logitsRestored, err := eng.ReadLogits()
	if err != nil {
		t.Fatalf("ReadLogits restored failed: %v", err)
	}

	for i := range logits0 {
		diff := float32(math.Abs(float64(logits0[i] - logitsRestored[i])))
		if diff > 1e-5 {
			t.Fatalf("restored logits diverge from initial at index %d (diff=%f > 1e-5)", i, diff)
		}
	}
}

func TestKVCacheIsolation(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatalf("GetOrCompileEngine failed: %v", err)
	}

	// Clean baseline run
	eng.ResetKV()
	tok0 := GPUForwardPass(eng, cfg, 15, 0)
	logits0, err := eng.ReadLogits()
	if err != nil {
		t.Fatalf("ReadLogits step 0 failed: %v", err)
	}

	tok1 := GPUForwardPass(eng, cfg, tok0, 1)
	logits1, err := eng.ReadLogits()
	if err != nil {
		t.Fatalf("ReadLogits step 1 failed: %v", err)
	}

	// Reset KV and run again to verify clean reproducibility
	eng.ResetKV()
	tok0Repeat := GPUForwardPass(eng, cfg, 15, 0)
	logits0Repeat, err := eng.ReadLogits()
	if err != nil {
		t.Fatalf("ReadLogits repeat step 0 failed: %v", err)
	}

	tok1Repeat := GPUForwardPass(eng, cfg, tok0Repeat, 1)
	logits1Repeat, err := eng.ReadLogits()
	if err != nil {
		t.Fatalf("ReadLogits repeat step 1 failed: %v", err)
	}

	if tok0 != tok0Repeat || tok1 != tok1Repeat {
		t.Fatalf("token mismatch across reset: (%d,%d) vs (%d,%d)", tok0, tok1, tok0Repeat, tok1Repeat)
	}

	for i := range logits0 {
		if math.Abs(float64(logits0[i]-logits0Repeat[i])) > 1e-5 {
			t.Fatalf("step 0 logits mismatch after reset at index %d", i)
		}
		if math.Abs(float64(logits1[i]-logits1Repeat[i])) > 1e-5 {
			t.Fatalf("step 1 logits mismatch after reset at index %d", i)
		}
	}
}

func TestFusedVsUnfusedEquivalence(t *testing.T) {
	cfgUnfused := DefaultTinyLlamaConfig()
	cfgUnfused.FuseEGraph = false

	cfgFused := DefaultTinyLlamaConfig()
	cfgFused.FuseEGraph = true

	engUnfused, err := GetOrCompileEngine(cfgUnfused)
	if err != nil {
		t.Fatalf("GetOrCompileEngine unfused failed: %v", err)
	}

	engFused, err := GetOrCompileEngine(cfgFused)
	if err != nil {
		t.Fatalf("GetOrCompileEngine fused failed: %v", err)
	}

	tokU := GPUForwardPass(engUnfused, cfgUnfused, 24, 0)
	logitsU, err := engUnfused.ReadLogits()
	if err != nil {
		t.Fatalf("ReadLogits unfused failed: %v", err)
	}

	tokF := GPUForwardPass(engFused, cfgFused, 24, 0)
	logitsF, err := engFused.ReadLogits()
	if err != nil {
		t.Fatalf("ReadLogits fused failed: %v", err)
	}

	if len(logitsU) == 0 || len(logitsF) == 0 {
		t.Fatal("empty logits returned from GPU")
	}
	if len(logitsU) != len(logitsF) {
		t.Fatalf("logits length mismatch: unfused %d vs fused %d", len(logitsU), len(logitsF))
	}

	var maxDiff float32
	for i := range logitsU {
		diff := float32(math.Abs(float64(logitsU[i] - logitsF[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	t.Logf("Fused vs Unfused Max Diff: %f (Unfused tok: %d, Fused tok: %d)", maxDiff, tokU, tokF)
	if maxDiff > 1e-3 {
		t.Fatalf("fused vs unfused max difference %f exceeded tolerance 1e-3", maxDiff)
	}
	if tokU != tokF {
		t.Fatalf("fused token %d disagrees with unfused token %d", tokF, tokU)
	}
}
