//go:build darwin

package main

import (
	"fmt"
	"math"
	"os"
	"testing"
)

// TestRealModelFusedVsUnfusedDecodeParity is the differential corpus for the
// e-graph fusion decision. Saturating the fusion rewrites changes the execution
// plan metadata. On f32 the engine implements those fused operations with fused
// Metal pipelines; on f16 it deliberately lowers them to discrete pipelines.
// Every decoded position must agree with the unfused plan on the full logit
// vector of a real checkpoint.
func TestRealModelFusedVsUnfusedDecodeParity(t *testing.T) {
	modelPath := os.Getenv("LLAMA_INFER_TEST_MODEL")
	if modelPath == "" {
		t.Skip("LLAMA_INFER_TEST_MODEL is not set")
	}
	for _, precision := range []string{"f16", "f32"} {
		t.Run(precision, func(t *testing.T) {
			fusedVsUnfusedDecodeParity(t, modelPath, precision)
		})
	}
}

func fusedVsUnfusedDecodeParity(t *testing.T, modelPath, precision string) {
	base := Llama3_2_1BConfig(modelPath)
	base.Precision = precision
	base.MaxSeqLen = 128

	cfgUnfused, cfgFused := base, base
	cfgUnfused.FuseEGraph, cfgFused.FuseEGraph = false, true

	planUnfused, planFused := BuildExecutionPlan(cfgUnfused), BuildExecutionPlan(cfgFused)
	if fused := fusedPlanOps(planFused); fused == 0 {
		t.Fatal("fused plan selects no fused operation")
	} else if unfused := fusedPlanOps(planUnfused); unfused != 0 {
		t.Fatalf("unfused plan selects %d fused operations", unfused)
	} else {
		t.Logf("plan ops unfused=%d fused=%d fused_kernels=%d", len(planUnfused.Ops), len(planFused.Ops), fused)
	}

	engUnfused, err := GetOrCompileEngine(cfgUnfused)
	if err != nil {
		t.Fatal(err)
	}
	engFused, err := GetOrCompileEngine(cfgFused)
	if err != nil {
		t.Fatal(err)
	}

	// Each engine must retain the plan built for its configuration. This checks
	// plan identity only; fusionActive separately reports whether that plan is
	// implemented by fused pipelines for the selected precision.
	if got := fusedPlanOps(engFused.Plan); got != fusedPlanOps(planFused) {
		t.Fatalf("fused engine plan selects %d fused operations, want %d", got, fusedPlanOps(planFused))
	}
	if got := fusedPlanOps(engUnfused.Plan); got != 0 {
		t.Fatalf("unfused engine plan selects %d fused operations", got)
	}

	const steps = 4
	prompt := "The capital of France is"
	tokens, err := engUnfused.tokenizer.Encode(prompt)
	if err != nil {
		t.Fatal(err)
	}
	if cfgUnfused.BOSTokenID != 0 {
		tokens = append([]int32{cfgUnfused.BOSTokenID}, tokens...)
	}

	unfusedTokens, unfusedLogits := decodeCorpus(t, engUnfused, cfgUnfused, tokens, steps)
	fusedTokens, fusedLogits := decodeCorpus(t, engFused, cfgFused, tokens, steps)

	maxDiff, err := compareDecodeCorpus(cfgFused.VocabSize, steps, unfusedTokens, unfusedLogits, fusedTokens, fusedLogits)
	if err != nil {
		t.Fatal(err)
	}
	if maxDiff > 1e-2 {
		t.Fatalf("fused vs unfused max logit difference %g exceeds 1e-2", maxDiff)
	}
	// f16 layers lower the fused plan ops back to the discrete kernels, so the
	// two engines must run identical work and the engine must not report fusion.
	if fusion := engFused.fusionActive(); fusion != (precision != "f16") {
		t.Fatalf("%s engine reports fusionActive=%v", precision, fusion)
	}
	if precision == "f16" && maxDiff != 0 {
		t.Fatalf("f16 fusion is inert but logits differ by %g", maxDiff)
	}

	// Sensitivity control: the comparison must be able to see a difference.
	// Decoding a different prompt through the same fused engine has to move the
	// logits far more than fusion does.
	otherTokens := append(append([]int32(nil), tokens[:len(tokens)-1]...), tokens[1])
	otherGenerated, otherLogits := decodeCorpus(t, engFused, cfgFused, otherTokens, 1)
	if len(otherGenerated) != 1 || len(otherLogits) != 1 || len(otherLogits[0]) != cfgFused.VocabSize {
		t.Fatalf("control corpus tokens=%d steps=%d logits=%d, want 1/1/%d", len(otherGenerated), len(otherLogits), firstLen(otherLogits), cfgFused.VocabSize)
	}
	controlDiff32, err := finiteMaxAbsDiff32(otherLogits[0], fusedLogits[0])
	if err != nil {
		t.Fatalf("control corpus: %v", err)
	}
	controlDiff := float64(controlDiff32)
	if controlDiff <= maxDiff {
		t.Fatalf("control difference %g does not exceed fusion difference %g", controlDiff, maxDiff)
	}
	t.Logf("positions=%d logits_per_step=%d fusion_max_diff=%g control_max_diff=%g tokens=%v",
		steps, len(unfusedLogits[0]), maxDiff, controlDiff, fusedTokens)
}

func compareDecodeCorpus(vocab, steps int, wantTokens []int32, wantLogits [][]float32, gotTokens []int32, gotLogits [][]float32) (float64, error) {
	if len(wantTokens) != steps || len(gotTokens) != steps || len(wantLogits) != steps || len(gotLogits) != steps {
		return 0, fmt.Errorf("corpus lengths tokens=%d/%d logits=%d/%d, want %d", len(wantTokens), len(gotTokens), len(wantLogits), len(gotLogits), steps)
	}
	var maxDiff float32
	for step := range steps {
		if wantTokens[step] != gotTokens[step] {
			return 0, fmt.Errorf("step %d: tokens=%d/%d", step, wantTokens[step], gotTokens[step])
		}
		if len(wantLogits[step]) != vocab || len(gotLogits[step]) != vocab {
			return 0, fmt.Errorf("step %d: logit lengths=%d/%d, want %d", step, len(wantLogits[step]), len(gotLogits[step]), vocab)
		}
		diff, err := finiteMaxAbsDiff32(wantLogits[step], gotLogits[step])
		if err != nil {
			return 0, fmt.Errorf("step %d: %w", step, err)
		}
		maxDiff = max(maxDiff, diff)
	}
	return float64(maxDiff), nil
}

func firstLen(values [][]float32) int {
	if len(values) == 0 {
		return 0
	}
	return len(values[0])
}

func TestCompareDecodeCorpusRejectsInvalidData(t *testing.T) {
	wantTokens := []int32{1, 2}
	wantLogits := [][]float32{{1, 2}, {3, 4}}
	clone := func() ([]int32, [][]float32) {
		return append([]int32(nil), wantTokens...), [][]float32{append([]float32(nil), wantLogits[0]...), append([]float32(nil), wantLogits[1]...)}
	}
	gotTokens, gotLogits := clone()
	if _, err := compareDecodeCorpus(2, 2, wantTokens, wantLogits, gotTokens, gotLogits); err != nil {
		t.Fatalf("valid corpus: %v", err)
	}
	tests := []struct {
		name   string
		mutate func(*[]int32, *[][]float32)
	}{
		{"empty tokens", func(tokens *[]int32, _ *[][]float32) { *tokens = nil }},
		{"empty steps", func(_ *[]int32, logits *[][]float32) { *logits = nil }},
		{"empty logits", func(_ *[]int32, logits *[][]float32) { (*logits)[0] = nil }},
		{"short logits", func(_ *[]int32, logits *[][]float32) { (*logits)[0] = (*logits)[0][:1] }},
		{"NaN", func(_ *[]int32, logits *[][]float32) { (*logits)[0][0] = float32(math.NaN()) }},
		{"infinity", func(_ *[]int32, logits *[][]float32) { (*logits)[1][1] = float32(math.Inf(-1)) }},
	}
	for _, test := range tests {
		for _, arm := range []string{"want", "got"} {
			t.Run(test.name+"/"+arm, func(t *testing.T) {
				leftTokens, leftLogits := clone()
				rightTokens, rightLogits := clone()
				if arm == "want" {
					test.mutate(&leftTokens, &leftLogits)
				} else {
					test.mutate(&rightTokens, &rightLogits)
				}
				if _, err := compareDecodeCorpus(2, 2, leftTokens, leftLogits, rightTokens, rightLogits); err == nil {
					t.Fatal("invalid corpus passed")
				}
			})
		}
	}
}

func fusedPlanOps(plan *ExecutionPlan) int {
	n := 0
	for _, op := range plan.Ops {
		if op.Fused {
			n++
		}
	}
	return n
}

// decodeCorpus prefills the prompt and returns the token and full logit vector
// produced at each of the following decode positions.
func decodeCorpus(t *testing.T, eng *ModelEngine, cfg Config, tokens []int32, steps int) ([]int32, [][]float32) {
	t.Helper()
	kv, err := eng.NewSequenceKV(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer kv.Free()
	current, err := GPUPrefillPassKV(eng, cfg, kv, tokens)
	if err != nil {
		t.Fatal(err)
	}
	gotTokens := make([]int32, 0, steps)
	gotLogits := make([][]float32, 0, steps)
	for step := range steps {
		next, err := GPUForwardPassKV(eng, cfg, kv, current, len(tokens)+step)
		if err != nil {
			t.Fatal(err)
		}
		logits, err := eng.ReadLogits()
		if err != nil {
			t.Fatal(err)
		}
		gotTokens = append(gotTokens, next)
		gotLogits = append(gotLogits, append([]float32(nil), logits...))
		current = next
	}
	return gotTokens, gotLogits
}
