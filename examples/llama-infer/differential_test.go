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
	gpuLogits0 := eng.ReadLogits()

	if len(gpuLogits0) != len(cpuLogits0) {
		t.Fatalf("logits length mismatch: GPU %d, CPU %d", len(gpuLogits0), len(cpuLogits0))
	}

	var maxDiff float32
	for i := range cpuLogits0 {
		diff := float32(math.Abs(float64(gpuLogits0[i] - cpuLogits0[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	t.Logf("Step 0 Logit Max Diff: %f (CPU tok: %d, GPU tok: %d)", maxDiff, cpuTok0, gpuTok0)
	if maxDiff > 0.5 {
		t.Fatalf("Step 0 GPU logits diverge from CPU reference (maxDiff = %f > 0.5)", maxDiff)
	}

	// Step 1 CPU Reference
	cpuLogits1 := Forward(weights, &mCfg, cpuKV, cpuTok0, 1)
	cpuTok1 := sampleArgmax(cpuLogits1)

	// Step 1 GPU Engine
	gpuTok1 := GPUForwardPass(eng, cfg, gpuTok0, 1)
	gpuLogits1 := eng.ReadLogits()

	var maxDiff1 float32
	for i := range cpuLogits1 {
		diff := float32(math.Abs(float64(gpuLogits1[i] - cpuLogits1[i])))
		if diff > maxDiff1 {
			maxDiff1 = diff
		}
	}
	t.Logf("Step 1 Logit Max Diff: %f (CPU tok: %d, GPU tok: %d)", maxDiff1, cpuTok1, gpuTok1)
	if maxDiff1 > 0.5 {
		t.Fatalf("Step 1 GPU logits diverge from CPU reference (maxDiff = %f > 0.5)", maxDiff1)
	}
}

func TestWeightMutationAffectsOutput(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatalf("GetOrCompileEngine failed: %v", err)
	}

	tok0 := GPUForwardPass(eng, cfg, 10, 0)
	logits0 := eng.ReadLogits()
	if len(logits0) == 0 {
		t.Fatal("no logits read from GPU")
	}

	// Mutate weights for another config
	cfgMutated := cfg
	cfgMutated.Name = "TinyLlama-Mutated"
	cfgMutated.Dim = 128
	cfgMutated.HiddenDim = 256
	engMutated, err := GetOrCompileEngine(cfgMutated)
	if err != nil {
		t.Fatalf("GetOrCompileEngine mutated failed: %v", err)
	}

	tokMutated := GPUForwardPass(engMutated, cfgMutated, 10, 0)
	logitsMutated := engMutated.ReadLogits()

	_ = tok0
	_ = tokMutated

	if len(logitsMutated) != len(logits0) {
		// Output dimensions match architecture change
		return
	}

	diffFound := false
	for i := range logits0 {
		if math.Abs(float64(logits0[i]-logitsMutated[i])) > 1e-4 {
			diffFound = true
			break
		}
	}
	if !diffFound {
		t.Fatal("GPU engine output is invariant to model configuration and weight changes!")
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
	logitsU := engUnfused.ReadLogits()

	tokF := GPUForwardPass(engFused, cfgFused, 24, 0)
	logitsF := engFused.ReadLogits()

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
