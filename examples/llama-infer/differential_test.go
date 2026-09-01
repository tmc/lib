//go:build darwin

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

func TestFP16VsFP32Equivalence(t *testing.T) {
	cfgF32 := DefaultTinyLlamaConfig()
	cfgF32.Precision = "f32"

	cfgF16 := DefaultTinyLlamaConfig()
	cfgF16.Precision = "f16"

	engF32, err := GetOrCompileEngine(cfgF32)
	if err != nil {
		t.Fatalf("GetOrCompileEngine F32 failed: %v", err)
	}
	if engF32.IsF16() {
		t.Fatal("expected engF32.IsF16() to be false")
	}

	engF16, err := GetOrCompileEngine(cfgF16)
	if err != nil {
		t.Fatalf("GetOrCompileEngine F16 failed: %v", err)
	}
	if !engF16.IsF16() {
		t.Fatal("expected engF16.IsF16() to be true")
	}

	// Step 0 Forward pass comparison
	tok32_0 := GPUForwardPass(engF32, cfgF32, 42, 0)
	logits32_0, err := engF32.ReadLogits()
	if err != nil {
		t.Fatalf("ReadLogits F32 step 0 failed: %v", err)
	}

	tok16_0 := GPUForwardPass(engF16, cfgF16, 42, 0)
	logits16_0, err := engF16.ReadLogits()
	if err != nil {
		t.Fatalf("ReadLogits F16 step 0 failed: %v", err)
	}

	if len(logits32_0) != len(logits16_0) {
		t.Fatalf("logits length mismatch: F32 %d vs F16 %d", len(logits32_0), len(logits16_0))
	}

	var maxDiff0 float32
	for i := range logits32_0 {
		diff := float32(math.Abs(float64(logits32_0[i] - logits16_0[i])))
		if diff > maxDiff0 {
			maxDiff0 = diff
		}
	}
	t.Logf("FP32 vs FP16 Step 0 Max Logit Diff: %f (F32 tok: %d, F16 tok: %d)", maxDiff0, tok32_0, tok16_0)
	if maxDiff0 > 0.05 {
		t.Fatalf("FP32 vs FP16 step 0 logit diff %f exceeds tolerance 0.05", maxDiff0)
	}
	if tok32_0 != tok16_0 {
		t.Fatalf("step 0 token mismatch: F32 %d vs F16 %d", tok32_0, tok16_0)
	}

	// Step 1 Forward pass comparison
	tok32_1 := GPUForwardPass(engF32, cfgF32, tok32_0, 1)
	logits32_1, err := engF32.ReadLogits()
	if err != nil {
		t.Fatalf("ReadLogits F32 step 1 failed: %v", err)
	}

	tok16_1 := GPUForwardPass(engF16, cfgF16, tok16_0, 1)
	logits16_1, err := engF16.ReadLogits()
	if err != nil {
		t.Fatalf("ReadLogits F16 step 1 failed: %v", err)
	}

	var maxDiff1 float32
	for i := range logits32_1 {
		diff := float32(math.Abs(float64(logits32_1[i] - logits16_1[i])))
		if diff > maxDiff1 {
			maxDiff1 = diff
		}
	}
	t.Logf("FP32 vs FP16 Step 1 Max Logit Diff: %f (F32 tok: %d, F16 tok: %d)", maxDiff1, tok32_1, tok16_1)
	if maxDiff1 > 0.05 {
		t.Fatalf("FP32 vs FP16 step 1 logit diff %f exceeds tolerance 0.05", maxDiff1)
	}
	if tok32_1 != tok16_1 {
		t.Fatalf("step 1 token mismatch: F32 %d vs F16 %d", tok32_1, tok16_1)
	}
}

func TestBatchedVsSequentialPrefillEquivalence(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	engSeq, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatalf("GetOrCompileEngine failed: %v", err)
	}

	prompt := []int32{12, 45, 99, 102}

	// 1. Sequential Prefill
	engSeq.ResetKV()
	tokSeq := GPUPrefillPass(engSeq, cfg, prompt)
	logitsSeq, err := engSeq.ReadLogits()
	if err != nil {
		t.Fatalf("ReadLogits sequential failed: %v", err)
	}

	// 2. True Batched Matrix-Matrix Prefill
	engBatch, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatalf("GetOrCompileEngine failed: %v", err)
	}
	engBatch.ResetKV()
	tokBatch, err := GPUBatchedPrefillPass(engBatch, cfg, prompt)
	if err != nil {
		t.Fatalf("GPUBatchedPrefillPass failed: %v", err)
	}
	logitsBatch, err := engBatch.ReadLogits()
	if err != nil {
		t.Fatalf("ReadLogits batched failed: %v", err)
	}

	var maxDiff float32
	for i := range logitsSeq {
		diff := float32(math.Abs(float64(logitsSeq[i] - logitsBatch[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	t.Logf("Sequential vs Batched Prefill Max Diff: %f (Seq tok: %d, Batch tok: %d)", maxDiff, tokSeq, tokBatch)
	if maxDiff > 0.01 {
		t.Fatalf("Sequential vs Batched Prefill logit diff %f exceeds tolerance 0.01", maxDiff)
	}
	if tokSeq != tokBatch {
		t.Fatalf("prefill token mismatch: Sequential %d vs Batched %d", tokSeq, tokBatch)
	}
}

func TestAuthoritativePlanDispatch(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatalf("GetOrCompileEngine failed: %v", err)
	}

	if eng.Plan == nil {
		t.Fatalf("expected non-nil ExecutionPlan on engine")
	}
	if len(eng.Plan.Ops) == 0 {
		t.Fatalf("expected ExecutionPlan to contain planned ops, got 0")
	}

	eng.ResetKV()
	tok0 := GPUForwardPass(eng, cfg, 42, 0)
	logits0, err := eng.ReadLogits()
	if err != nil {
		t.Fatalf("ReadLogits failed: %v", err)
	}

	if tok0 != 100 {
		t.Errorf("expected token 100 from plan-driven forward pass, got %d", tok0)
	}
	if len(logits0) != cfg.VocabSize {
		t.Errorf("expected logits size %d, got %d", cfg.VocabSize, len(logits0))
	}
}

func TestMultiSequenceBatchedDecodeEquivalence(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatalf("GetOrCompileEngine failed: %v", err)
	}

	skvSingle0, _ := eng.NewSequenceKV(cfg)
	defer skvSingle0.Free()
	skvSingle1, _ := eng.NewSequenceKV(cfg)
	defer skvSingle1.Free()

	skvBatch0, _ := eng.NewSequenceKV(cfg)
	defer skvBatch0.Free()
	skvBatch1, _ := eng.NewSequenceKV(cfg)
	defer skvBatch1.Free()

	// Prefill sequences
	_, _ = GPUPrefillPassKV(eng, cfg, skvSingle0, []int32{12, 45})
	_, _ = GPUPrefillPassKV(eng, cfg, skvSingle1, []int32{88, 102})
	_, _ = GPUPrefillPassKV(eng, cfg, skvBatch0, []int32{12, 45})
	_, _ = GPUPrefillPassKV(eng, cfg, skvBatch1, []int32{88, 102})

	// 1. Single forward pass for each sequence independently
	tokSingle0, err := GPUForwardPassKV(eng, cfg, skvSingle0, 99, 2)
	if err != nil {
		t.Fatalf("Forward single seq 0 failed: %v", err)
	}
	tokSingle1, err := GPUForwardPassKV(eng, cfg, skvSingle1, 201, 2)
	if err != nil {
		t.Fatalf("Forward single seq 1 failed: %v", err)
	}

	// 2. Batch decode step for both sequences concurrently
	batchTokens := []int32{99, 201}
	batchPositions := []int{2, 2}
	batchKVs := []*SequenceKVCache{skvBatch0, skvBatch1}

	batchOut, err := GPUBatchDecodeStep(eng, cfg, batchKVs, batchTokens, batchPositions)
	if err != nil {
		t.Fatalf("GPUBatchDecodeStep failed: %v", err)
	}

	if len(batchOut) != 2 {
		t.Fatalf("expected 2 batch outputs, got %d", len(batchOut))
	}
	t.Logf("Seq 0: Single=%d, Batch=%d | Seq 1: Single=%d, Batch=%d", tokSingle0, batchOut[0], tokSingle1, batchOut[1])
	if batchOut[0] != tokSingle0 {
		t.Fatalf("Seq 0 token mismatch: Single %d vs Batch %d", tokSingle0, batchOut[0])
	}
	if batchOut[1] != tokSingle1 {
		t.Fatalf("Seq 1 token mismatch: Single %d vs Batch %d", tokSingle1, batchOut[1])
	}
}

func TestRadixPrefixReuseIntegration(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatalf("GetOrCompileEngine failed: %v", err)
	}

	systemPrompt := []int32{1, 50, 60, 70, 80}
	userPrompt1 := append(append([]int32(nil), systemPrompt...), 91, 92)
	userPrompt2 := append(append([]int32(nil), systemPrompt...), 101, 102)

	eng.prefixes = newPrefixCache(64<<20, 16)
	eng.ResetKV()

	// 1. First request with userPrompt1
	_ = GPUPrefillPass(eng, cfg, userPrompt1)
	snap1 := eng.capturePrefix(len(userPrompt1), 0, cfg)
	eng.prefixes.store(userPrompt1, snap1)

	m1 := eng.prefixes.metrics()
	if m1.Entries != 1 {
		t.Fatalf("expected 1 cache entry, got %d", m1.Entries)
	}

	// 2. Second request with userPrompt2 sharing the 5-token system prompt prefix
	snapMatched, found := eng.prefixes.load(userPrompt2)
	if !found || snapMatched == nil {
		t.Fatalf("expected radix prefix match for shared system prompt")
	}

	m2 := eng.prefixes.metrics()
	if m2.Hits < 1 {
		t.Errorf("expected at least 1 prefix cache hit, got %d", m2.Hits)
	}
	if m2.SavedTokens < int64(len(systemPrompt)) {
		t.Errorf("expected at least %d saved tokens, got %d", len(systemPrompt), m2.SavedTokens)
	}
	t.Logf("Radix Prefix Reuse: Hits=%d, SavedTokens=%d", m2.Hits, m2.SavedTokens)
}
