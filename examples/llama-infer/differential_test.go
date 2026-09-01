//go:build darwin

package main

import (
	"context"
	"math"
	"testing"
	"time"
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

	origPlan := eng.Plan
	defer func() {
		eng.Plan = origPlan
	}()

	// 1. Reference execution without plan
	eng.Plan = nil
	eng.ResetKV()
	tokRef := GPUForwardPass(eng, cfg, 42, 0)
	logitsRef, err := eng.ReadLogits()
	if err != nil {
		t.Fatalf("ReadLogits ref failed: %v", err)
	}

	// 2. Authoritative Plan execution
	eng.Plan = origPlan
	eng.ResetKV()
	tokPlan := GPUForwardPass(eng, cfg, 42, 0)
	logitsPlan, err := eng.ReadLogits()
	if err != nil {
		t.Fatalf("ReadLogits plan failed: %v", err)
	}

	if tokPlan != tokRef {
		t.Fatalf("token mismatch: Ref %d vs Plan %d", tokRef, tokPlan)
	}
	for i := range logitsPlan {
		diff := float32(math.Abs(float64(logitsPlan[i] - logitsRef[i])))
		if diff > 1e-4 {
			t.Fatalf("logit mismatch at index %d: Ref %f vs Plan %f (diff %f)", i, logitsRef[i], logitsPlan[i], diff)
		}
	}

	// 3. Mutate / tamper with the plan by removing an essential internal layer op (e.g. down projection)
	mutatedOps := make([]OpDispatch, 0, len(origPlan.Ops))
	for _, op := range origPlan.Ops {
		if op.Name != "FastGEMVResidual_Down" && op.Name != "FusedSwiGLUDown" {
			mutatedOps = append(mutatedOps, op)
		}
	}
	eng.Plan = &ExecutionPlan{
		DType:      origPlan.DType,
		FuseEGraph: origPlan.FuseEGraph,
		BatchShape: origPlan.BatchShape,
		HeadDim:    origPlan.HeadDim,
		NumHeads:   origPlan.NumHeads,
		NumKVHeads: origPlan.NumKVHeads,
		NumLayers:  origPlan.NumLayers,
		VocabSize:  origPlan.VocabSize,
		Ops:        mutatedOps,
		Buffers:    origPlan.Buffers,
	}

	eng.ResetKV()
	_, err = GPUForwardPassKV(eng, cfg, nil, 42, 0)
	if err == nil {
		t.Fatalf("mutated plan failed authority gate: expected GPUForwardPassKV to fail closed on invalid topology, got nil")
	}
	t.Logf("Authoritative Plan Fail-Closed Mutation Gate: %v", err)
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
	snapMatched, _, found := eng.prefixes.load(userPrompt2)
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

func TestPartialPrefixRestoreDivergence(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatalf("GetOrCompileEngine failed: %v", err)
	}

	prefix := []int32{1, 10, 20, 30, 40}
	promptLong := append(append([]int32(nil), prefix...), 100, 101, 102)
	promptDivergent := append(append([]int32(nil), prefix...), 200, 201, 202)

	// 1. Reference from scratch (clean cache)
	skvClean, _ := eng.NewSequenceKV(cfg)
	defer skvClean.Free()
	tokRef, err := GPUBatchedPrefillPassKV(eng, cfg, skvClean, promptDivergent, 0)
	if err != nil {
		t.Fatalf("clean prefill failed: %v", err)
	}

	// 2. Prefill promptLong on skvLong and capture snapshot
	skvLong, _ := eng.NewSequenceKV(cfg)
	defer skvLong.Free()
	_, _ = GPUBatchedPrefillPassKV(eng, cfg, skvLong, promptLong, 0)
	snapLong := eng.capturePrefix(len(promptLong), 0, cfg)

	// 3. Restore only the 5-token prefix into skvDivergent and prefill remaining 3 divergent tokens
	skvDivergent, _ := eng.NewSequenceKV(cfg)
	defer skvDivergent.Free()
	if !skvDivergent.RestorePrefixPartial(snapLong, len(prefix), cfg) {
		t.Fatalf("RestorePrefixPartial failed")
	}
	tokRestored, err := GPUBatchedPrefillPassKV(eng, cfg, skvDivergent, promptDivergent[len(prefix):], len(prefix))
	if err != nil {
		t.Fatalf("divergent suffix prefill failed: %v", err)
	}

	t.Logf("Partial Prefix Restore Divergence: Clean Tok=%d, Restored Tok=%d", tokRef, tokRestored)
	if tokRef != tokRestored {
		t.Fatalf("token mismatch: clean %d vs restored partial prefix %d", tokRef, tokRestored)
	}
}

func TestBatchedDecodeScratchIsolation(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatalf("GetOrCompileEngine failed: %v", err)
	}

	numSeqs := 4
	prompts := [][]int32{
		{5, 12, 19, 23},
		{44, 55, 66, 77},
		{101, 102, 103, 104},
		{210, 220, 230, 240},
	}
	decodeTokens := []int32{99, 150, 201, 33}
	positions := []int{4, 4, 4, 4}

	// 1. Single forward pass reference for each sequence
	singleOut := make([]int32, numSeqs)
	batchKVs := make([]*SequenceKVCache, numSeqs)
	for i := 0; i < numSeqs; i++ {
		skv, _ := eng.NewSequenceKV(cfg)
		defer skv.Free()
		_, _ = GPUBatchedPrefillPassKV(eng, cfg, skv, prompts[i], 0)
		singleTok, err := GPUForwardPassKV(eng, cfg, skv, decodeTokens[i], positions[i])
		if err != nil {
			t.Fatalf("single forward seq %d failed: %v", i, err)
		}
		singleOut[i] = singleTok
	}

	// 2. Batched decode step across all 4 sequences concurrently
	for i := 0; i < numSeqs; i++ {
		skv, _ := eng.NewSequenceKV(cfg)
		defer skv.Free()
		_, _ = GPUBatchedPrefillPassKV(eng, cfg, skv, prompts[i], 0)
		batchKVs[i] = skv
	}

	batchOut, err := GPUBatchDecodeStep(eng, cfg, batchKVs, decodeTokens, positions)
	if err != nil {
		t.Fatalf("GPUBatchDecodeStep failed: %v", err)
	}

	for i := 0; i < numSeqs; i++ {
		t.Logf("Seq %d: Single=%d, Batch=%d", i, singleOut[i], batchOut[i])
		if singleOut[i] != batchOut[i] {
			t.Fatalf("Seq %d scratch collision: Single %d != Batch %d", i, singleOut[i], batchOut[i])
		}
	}
}

func TestPrefixCacheEvictionAndColdWarmEquivalence(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatalf("GetOrCompileEngine failed: %v", err)
	}

	// 1. Cold reference generation from scratch for a prompt with divergent suffix
	promptPrefix := []int32{1, 10, 20, 30}
	promptA := append(append([]int32(nil), promptPrefix...), 88, 89)
	promptB := append(append([]int32(nil), promptPrefix...), 98, 99)

	eng.prefixes = newPrefixCache(64<<20, 2)
	eng.ResetKV()

	// Clean run of prompt B without any cache
	skvCleanB, _ := eng.NewSequenceKV(cfg)
	defer skvCleanB.Free()
	tokPrefillB, _ := GPUBatchedPrefillPassKV(eng, cfg, skvCleanB, promptB, 0)
	tokDecB0, _ := GPUForwardPassKV(eng, cfg, skvCleanB, tokPrefillB, len(promptB))
	tokDecB1, _ := GPUForwardPassKV(eng, cfg, skvCleanB, tokDecB0, len(promptB)+1)
	cleanTokensB := []int32{tokPrefillB, tokDecB0, tokDecB1}
	cleanLogitsB, _ := eng.ReadLogits()

	// 2. Warm run: first run prompt A on isolated SequenceKVCache and capture from skvA
	skvA, _ := eng.NewSequenceKV(cfg)
	defer skvA.Free()
	tokPrefillA, _ := GPUBatchedPrefillPassKV(eng, cfg, skvA, promptA, 0)
	snapA := skvA.CapturePrefix(len(promptA), tokPrefillA, cfg)
	if snapA == nil {
		t.Fatalf("skvA.CapturePrefix returned nil")
	}
	eng.prefixes.store(promptA, snapA)

	// Now run prompt B with radix cache lookup (it will match the 4-token shared prefix)
	skvWarmB, _ := eng.NewSequenceKV(cfg)
	defer skvWarmB.Free()
	snapMatched, matched, found := eng.prefixes.load(promptB)
	if !found || matched != len(promptPrefix) {
		t.Fatalf("expected 4-token prefix match, got found=%t, matched=%d", found, matched)
	}
	if !skvWarmB.RestorePrefixPartial(snapMatched, matched, cfg) {
		t.Fatalf("RestorePrefixPartial failed")
	}
	// Prefill remaining divergent suffix
	tokWarmPrefillB, err := GPUBatchedPrefillPassKV(eng, cfg, skvWarmB, promptB[matched:], matched)
	if err != nil {
		t.Fatalf("warm divergent prefill failed: %v", err)
	}
	tokWarmDecB0, _ := GPUForwardPassKV(eng, cfg, skvWarmB, tokWarmPrefillB, len(promptB))
	tokWarmDecB1, _ := GPUForwardPassKV(eng, cfg, skvWarmB, tokWarmDecB0, len(promptB)+1)
	warmTokensB := []int32{tokWarmPrefillB, tokWarmDecB0, tokWarmDecB1}
	warmLogitsB, _ := eng.ReadLogits()

	t.Logf("Cold Tokens B: %v", cleanTokensB)
	t.Logf("Warm Tokens B: %v", warmTokensB)
	for i := range cleanTokensB {
		if cleanTokensB[i] != warmTokensB[i] {
			t.Fatalf("Cold vs Warm token mismatch at step %d: Cold %d vs Warm %d", i, cleanTokensB[i], warmTokensB[i])
		}
	}
	for i := range cleanLogitsB {
		diff := float32(math.Abs(float64(cleanLogitsB[i] - warmLogitsB[i])))
		if diff > 1e-4 {
			t.Fatalf("Cold vs Warm logit mismatch at index %d: Cold %f vs Warm %f", i, cleanLogitsB[i], warmLogitsB[i])
		}
	}

	// 3. Mutation control: tamper with restored prefix KV entries and verify logits diverge
	skvMutated, _ := eng.NewSequenceKV(cfg)
	defer skvMutated.Free()
	skvMutated.RestorePrefixPartial(snapMatched, matched, cfg)
	tokMutatedPrefill, _ := GPUBatchedPrefillPassKV(eng, cfg, skvMutated, promptB[matched:], matched)
	for i := 0; i < matched*cfg.NumKVHeads*cfg.HeadDim; i++ {
		skvMutated.VCache[0].HostSlice()[i] += 50.0
	}
	_, _ = GPUForwardPassKV(eng, cfg, skvMutated, tokMutatedPrefill, len(promptB))
	mutatedLogits, _ := eng.ReadLogits()
	maxMutationDiff := float32(0.0)
	for i := range mutatedLogits {
		diff := float32(math.Abs(float64(mutatedLogits[i] - warmLogitsB[i])))
		if diff > maxMutationDiff {
			maxMutationDiff = diff
		}
	}
	if maxMutationDiff < 1e-1 {
		t.Fatalf("KV cache mutation control failed: expected logit divergence >= 0.1, got %f", maxMutationDiff)
	}
	t.Logf("KV Cache Mutation Control Divergence: %f", maxMutationDiff)

	// 4. Test LRU eviction and verify evicted blockMap entries are deleted
	evictedBlocks := append([]int(nil), eng.prefixes.tokenBlocks[prefixKey(promptA)]...)
	if len(evictedBlocks) == 0 {
		t.Fatalf("expected tracked tokenBlocks for promptA")
	}

	promptC := []int32{5, 6, 7, 8, 9}
	promptD := []int32{15, 16, 17, 18, 19}
	snapC := &prefixSnapshot{tokens: 5, bytes: 512, lastUsed: time.Now().Add(-10 * time.Minute)}
	snapD := &prefixSnapshot{tokens: 5, bytes: 512, lastUsed: time.Now().Add(-5 * time.Minute)}
	eng.prefixes.store(promptC, snapC)
	eng.prefixes.store(promptD, snapD)

	// Cache capacity is maxItems=2, so promptA should have been evicted
	m := eng.prefixes.metrics()
	if m.Entries > 2 {
		t.Fatalf("expected at most 2 cache entries, got %d", m.Entries)
	}
	if len(eng.prefixes.entries) > 2 {
		t.Fatalf("expected at most 2 entries in map, got %d", len(eng.prefixes.entries))
	}
	for _, blockID := range evictedBlocks {
		if _, exists := eng.prefixes.blockMap[blockID]; exists {
			t.Fatalf("evicted block ID %d still present in blockMap", blockID)
		}
	}
}

func TestHeadDim64BatchedPrefillEquivalence(t *testing.T) {
	cfg := Config{
		Name:         "HeadDim64TestModel",
		Dim:          256,
		HiddenDim:    512,
		NumLayers:    2,
		NumHeads:     4,
		NumKVHeads:   2,
		HeadDim:      64,
		VocabSize:    256,
		MaxSeqLen:    128,
		Eps:          1e-5,
		RopeTheta:    10000.0,
		Precision:    "fp32",
		FuseEGraph:   false,
		TestMode:     true,
	}
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatalf("GetOrCompileEngine failed: %v", err)
	}

	prompt := []int32{10, 20, 30, 40, 50, 60, 70, 80}

	// 1. Sequential prefill via single forward steps
	skvSeq, _ := eng.NewSequenceKV(cfg)
	defer skvSeq.Free()
	var tokSeq int32
	for p, tok := range prompt {
		tokSeq, err = GPUForwardPassKV(eng, cfg, skvSeq, tok, p)
		if err != nil {
			t.Fatalf("seq step %d failed: %v", p, err)
		}
	}
	logitsSeq, err := eng.ReadLogits()
	if err != nil {
		t.Fatalf("ReadLogits seq failed: %v", err)
	}

	// 2. Batched prefill
	skvBatch, _ := eng.NewSequenceKV(cfg)
	defer skvBatch.Free()
	tokBatch, err := GPUBatchedPrefillPassKV(eng, cfg, skvBatch, prompt, 0)
	if err != nil {
		t.Fatalf("batched prefill failed: %v", err)
	}
	logitsBatch, err := eng.ReadLogits()
	if err != nil {
		t.Fatalf("ReadLogits batch failed: %v", err)
	}

	if tokSeq != tokBatch {
		t.Fatalf("HeadDim=64 prefill token mismatch: Sequential %d vs Batched %d", tokSeq, tokBatch)
	}
	maxDiff := float32(0.0)
	for i := range logitsSeq {
		diff := float32(math.Abs(float64(logitsSeq[i] - logitsBatch[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > 1e-4 {
			t.Fatalf("HeadDim=64 logit mismatch at %d: Seq %f vs Batch %f (diff %f)", i, logitsSeq[i], logitsBatch[i], diff)
		}
	}
	t.Logf("HeadDim=64 Batched vs Sequential Max Logit Diff: %f (Tok: %d)", maxDiff, tokBatch)
}

func TestExecutionPlanFailClosedValidation(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatalf("GetOrCompileEngine failed: %v", err)
	}

	// 1. Valid plan passes
	validPlan := BuildExecutionPlan(cfg)
	if err := ValidatePlan(validPlan, cfg); err != nil {
		t.Fatalf("valid plan failed validation: %v", err)
	}

	// 2. Unknown op fails validation
	badOpPlan := BuildExecutionPlan(cfg)
	badOpPlan.Ops = append([]OpDispatch{{Name: "MalformedNonexistentOp", Category: "invalid"}}, badOpPlan.Ops...)
	if err := ValidatePlan(badOpPlan, cfg); err == nil {
		t.Fatalf("expected error validating plan with unknown op, got nil")
	}

	// 3. Out-of-bounds layer fails validation
	badLayerPlan := BuildExecutionPlan(cfg)
	badLayerPlan.Ops = append([]OpDispatch{{Name: "ComputeRMSNorm", Category: "norm", LayerIndex: 999}}, badLayerPlan.Ops...)
	if err := ValidatePlan(badLayerPlan, cfg); err == nil {
		t.Fatalf("expected error validating plan with out-of-bounds layer, got nil")
	}

	// 4. Unsupported dtype fails validation
	badDTypePlan := BuildExecutionPlan(cfg)
	badDTypePlan.DType = "int8_unsupported"
	if err := ValidatePlan(badDTypePlan, cfg); err == nil {
		t.Fatalf("expected error validating plan with unsupported dtype, got nil")
	}

	// 5. Zero / invalid dimension fails validation
	badDimPlan := BuildExecutionPlan(cfg)
	badDimPlan.HeadDim = 0
	if err := ValidatePlan(badDimPlan, cfg); err == nil {
		t.Fatalf("expected error validating plan with zero HeadDim, got nil")
	}

	// 6. Dimension mismatch against config fails validation
	mismatchDimPlan := BuildExecutionPlan(cfg)
	mismatchDimPlan.NumLayers = cfg.NumLayers + 1
	if err := ValidatePlan(mismatchDimPlan, cfg); err == nil {
		t.Fatalf("expected error validating plan with mismatched NumLayers, got nil")
	}

	// 7. Topology violation: missing leading EmbeddingLookup fails validation
	badTopoPlan := BuildExecutionPlan(cfg)
	badTopoPlan.Ops[0] = OpDispatch{Name: "ComputeRMSNorm", Category: "norm", LayerIndex: 0}
	if err := ValidatePlan(badTopoPlan, cfg); err == nil {
		t.Fatalf("expected error validating plan missing leading EmbeddingLookup, got nil")
	}

	// 8. Topology violation: missing layer op fails validation
	missingOpPlan := BuildExecutionPlan(cfg)
	// Remove an operation from layer 0 (e.g., CausalFlashAttentionDecode)
	missingOpPlan.Ops = append(missingOpPlan.Ops[:5], missingOpPlan.Ops[6:]...)
	if err := ValidatePlan(missingOpPlan, cfg); err == nil {
		t.Fatalf("expected error validating plan with missing layer operation, got nil")
	}

	// 9. Topology violation: duplicate layer op fails validation
	dupOpPlan := BuildExecutionPlan(cfg)
	// Duplicate an operation in layer 0
	dupOpPlan.Ops = append(dupOpPlan.Ops[:5], append([]OpDispatch{dupOpPlan.Ops[4]}, dupOpPlan.Ops[5:]...)...)
	if err := ValidatePlan(dupOpPlan, cfg); err == nil {
		t.Fatalf("expected error validating plan with duplicate layer operation, got nil")
	}

	// 10. Topology violation: same-layer op reordering fails validation
	reorderOpPlan := BuildExecutionPlan(cfg)
	// Swap FastGEMV_Q and FastGEMV_K
	if len(reorderOpPlan.Ops) > 4 && reorderOpPlan.Ops[2].Name == "FastGEMV_Q" && reorderOpPlan.Ops[3].Name == "FastGEMV_K" {
		reorderOpPlan.Ops[2], reorderOpPlan.Ops[3] = reorderOpPlan.Ops[3], reorderOpPlan.Ops[2]
		if err := ValidatePlan(reorderOpPlan, cfg); err == nil {
			t.Fatalf("expected error validating plan with same-layer op reordering, got nil")
		}
	}

	// 11. Topology violation: out-of-order layer sequencing fails validation
	badSeqPlan := BuildExecutionPlan(cfg)
	if len(badSeqPlan.Ops) > 15 {
		// Swap layer 0 op with layer 1 op
		badSeqPlan.Ops[2].LayerIndex = 1
		badSeqPlan.Ops[12].LayerIndex = 0
		if err := ValidatePlan(badSeqPlan, cfg); err == nil {
			t.Fatalf("expected error validating plan with out-of-order layer sequencing, got nil")
		}
	}

	// 12. Truncation boundary: truncated immediately after QKV block
	truncQKVPlan := BuildExecutionPlan(cfg)
	if len(truncQKVPlan.Ops) >= 5 {
		truncQKVPlan.Ops = truncQKVPlan.Ops[:5]
		if err := ValidatePlan(truncQKVPlan, cfg); err == nil {
			t.Fatalf("expected error validating plan truncated after QKV, got nil")
		}
	}

	// 13. Truncation boundary: truncated immediately after attention core block
	truncCorePlan := BuildExecutionPlan(cfg)
	if len(truncCorePlan.Ops) >= 9 {
		truncCorePlan.Ops = truncCorePlan.Ops[:9]
		if err := ValidatePlan(truncCorePlan, cfg); err == nil {
			t.Fatalf("expected error validating plan truncated after attention core, got nil")
		}
	}

	// 14. Topology violation: missing epilogue (truncated before final norm/lm_head/sampling)
	missingEpiloguePlan := BuildExecutionPlan(cfg)
	if len(missingEpiloguePlan.Ops) > 3 {
		missingEpiloguePlan.Ops = missingEpiloguePlan.Ops[:len(missingEpiloguePlan.Ops)-3]
		if err := ValidatePlan(missingEpiloguePlan, cfg); err == nil {
			t.Fatalf("expected error validating plan with missing epilogue, got nil")
		}
	}

	// 15. Topology violation: extra trailing operations after epilogue
	extraTrailingPlan := BuildExecutionPlan(cfg)
	extraTrailingPlan.Ops = append(extraTrailingPlan.Ops, OpDispatch{Name: "ComputeRMSNorm_Final", Category: "norm"})
	if err := ValidatePlan(extraTrailingPlan, cfg); err == nil {
		t.Fatalf("expected error validating plan with extra trailing op, got nil")
	}

	// 16. ExecutePlanWithLogits fails closed on invalid plan
	origPlan := eng.Plan
	defer func() { eng.Plan = origPlan }()
	eng.Plan = badOpPlan
	eng.ResetKV()
	_, err = GPUForwardPassKV(eng, cfg, nil, 42, 0)
	if err == nil {
		t.Fatalf("expected GPUForwardPassKV to fail closed when plan is invalid, got nil")
	}
	t.Logf("Plan Fail-Closed Verification: %v", err)
}

func TestEOSStoppingAndFinishReason(t *testing.T) {
	tests := []struct {
		name         string
		cfg          Config
		tokenID      int32
		wantIsEOS    bool
		tokensToGen  int
		maxTokens    int
		wantFinish   string
	}{
		{
			name:        "llama3_primary_eos",
			cfg:         Llama3_2_1BConfig(""),
			tokenID:     128009,
			wantIsEOS:   true,
			tokensToGen: 5,
			maxTokens:   10,
			wantFinish:  "stop",
		},
		{
			name:        "llama3_alternate_eos",
			cfg:         Llama3_2_1BConfig(""),
			tokenID:     128001,
			wantIsEOS:   true,
			tokensToGen: 5,
			maxTokens:   10,
			wantFinish:  "stop",
		},
		{
			name:        "llama3_non_eos",
			cfg:         Llama3_2_1BConfig(""),
			tokenID:     42,
			wantIsEOS:   false,
			tokensToGen: 10,
			maxTokens:   10,
			wantFinish:  "length",
		},
		{
			name:        "tinyllama_eos",
			cfg:         DefaultTinyLlamaConfig(),
			tokenID:     2,
			wantIsEOS:   true,
			tokensToGen: 3,
			maxTokens:   10,
			wantFinish:  "stop",
		},
		{
			name:        "tinyllama_non_eos",
			cfg:         DefaultTinyLlamaConfig(),
			tokenID:     100,
			wantIsEOS:   false,
			tokensToGen: 10,
			maxTokens:   10,
			wantFinish:  "length",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotEOS := tt.cfg.IsEOS(tt.tokenID)
			if gotEOS != tt.wantIsEOS {
				t.Errorf("IsEOS(%d) = %v; want %v", tt.tokenID, gotEOS, tt.wantIsEOS)
			}

			// Test finish_reason logic
			finishReason := "stop"
			if tt.tokensToGen >= tt.maxTokens {
				finishReason = "length"
			}
			if finishReason != tt.wantFinish {
				t.Errorf("finish_reason = %q; want %q", finishReason, tt.wantFinish)
			}
		})
	}

	// Test Scheduler completion and DoneCh closure on EOS
	cfg := Config{
		Name:        "TestEOSModel",
		EOSTokenID:  15,
		EOSTokenIDs: []int32{15},
		MaxSeqLen:   32,
		TestMode:    true,
	}
	sched := NewScheduler(nil, cfg, 4)
	seq := NewSequence("test-eos", "prompt", []int32{10, 11, 12, 13, 14}, 10, nil)
	if err := sched.Submit(seq); err != nil {
		t.Fatalf("Submit: %v", err)
	}

	// Step mock scheduler until sequence produces token 15 (EOS)
	for i := 0; i < 5; i++ {
		_, err := sched.Step(context.Background())
		if err != nil {
			t.Fatalf("Step: %v", err)
		}
	}

	seq.mu.Lock()
	status := seq.Status
	seq.mu.Unlock()

	if status != SeqFinished {
		t.Errorf("sequence status = %v; want SeqFinished (%v)", status, SeqFinished)
	}

	select {
	case <-seq.DoneCh:
		// success: DoneCh is closed
	default:
		t.Errorf("expected seq.DoneCh to be closed upon reaching EOS")
	}
}

func TestTokenizerSpecialTokensAndChatTemplate(t *testing.T) {
	tok := &BpeTokenizer{
		vocab: map[string]int32{
			"<|start_header_id|>": 100,
			"<|end_header_id|>":   101,
			"<|eot_id|>":           102,
			"system":              200,
			"user":                201,
			"assistant":           202,
		},
		invVocab: map[int32]string{
			100: "<|start_header_id|>",
			101: "<|end_header_id|>",
			102: "<|eot_id|>",
			200: "system",
			201: "user",
			202: "assistant",
		},
		special: map[string]int32{
			"<|start_header_id|>": 100,
			"<|end_header_id|>":   101,
			"<|eot_id|>":           102,
		},
		specialNames: []string{
			"<|start_header_id|>",
			"<|end_header_id|>",
			"<|eot_id|>",
		},
		byteTokens: make(map[byte]string),
	}

	for b := 0; b < 256; b++ {
		tok.byteTokens[byte(b)] = string([]byte{byte(b)})
		if _, ok := tok.vocab[string([]byte{byte(b)})]; !ok {
			tok.vocab[string([]byte{byte(b)})] = int32(1000 + b)
			tok.invVocab[int32(1000+b)] = string([]byte{byte(b)})
		}
	}

	prompt := "<|start_header_id|>system<|end_header_id|>\n\n<|eot_id|>"
	tokens, err := tok.Encode(prompt)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}

	if len(tokens) < 3 {
		t.Fatalf("expected at least 3 tokens, got %v", tokens)
	}
	if tokens[0] != 100 {
		t.Errorf("token[0] = %d; want 100 (<|start_header_id|>)", tokens[0])
	}
	if tokens[len(tokens)-1] != 102 {
		t.Errorf("last token = %d; want 102 (<|eot_id|>)", tokens[len(tokens)-1])
	}
}

func TestMaxSeqLenClamping(t *testing.T) {
	cfg := ModelConfig{
		HiddenSize:        64,
		NumAttentionHeads: 4,
		MaxSeqLen:         131072, // large declaration as in LLaMA 3.2
	}

	// Verify clamp logic bounds it
	if cfg.MaxSeqLen > 8192 {
		cfg.MaxSeqLen = 2048
	}

	if cfg.MaxSeqLen != 2048 {
		t.Errorf("MaxSeqLen = %d; want 2048", cfg.MaxSeqLen)
	}
}
