//go:build darwin

package main

import (
	"context"
	"os"
	"reflect"
	"testing"
)

// TestRealModelGenerateColdWarmPrefixParity proves the retained owner of the
// snapshot prefix cache end to end. ModelEngine.Generate is the only production
// caller left; a warm generation must restore the prompt KV instead of
// prefilling it and must still emit the cold token sequence.
func TestRealModelGenerateColdWarmPrefixParity(t *testing.T) {
	modelPath := os.Getenv("LLAMA_INFER_TEST_MODEL")
	if modelPath == "" {
		t.Skip("LLAMA_INFER_TEST_MODEL is not set")
	}
	cfg := Llama3_2_1BConfig(modelPath)
	cfg.Precision = "f16"
	cfg.MaxSeqLen = 128
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	previous := eng.prefixes
	eng.prefixes = newPrefixCache(64<<20, 8)
	t.Cleanup(func() { eng.prefixes = previous })

	const prompt = "The capital of France is"
	promptTokens, err := eng.tokenizer.Encode(prompt)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.BOSTokenID != 0 {
		promptTokens = append([]int32{cfg.BOSTokenID}, promptTokens...)
	}

	beforeCold, _, _, _, _ := eng.metrics.Snapshot()
	cold, err := eng.GenerateTokenIDs(context.Background(), cfg, prompt, 4)
	if err != nil {
		t.Fatal(err)
	}
	afterCold, _, _, _, _ := eng.metrics.Snapshot()
	coldMetrics := eng.prefixes.metrics()
	if coldMetrics.Entries != 1 || coldMetrics.Hits != 0 || coldMetrics.Misses != 1 || coldMetrics.Bytes <= 0 {
		t.Fatalf("cold prefix metrics = %+v, want one stored miss", coldMetrics)
	}

	warm, err := eng.GenerateTokenIDs(context.Background(), cfg, prompt, 4)
	if err != nil {
		t.Fatal(err)
	}
	afterWarm, _, _, _, _ := eng.metrics.Snapshot()
	warmMetrics := eng.prefixes.metrics()
	if warmMetrics.Hits != 1 || warmMetrics.SavedTokens != int64(len(promptTokens)) {
		t.Fatalf("warm prefix metrics = %+v, want one hit saving %d tokens", warmMetrics, len(promptTokens))
	}
	if !reflect.DeepEqual(cold, warm) {
		t.Fatalf("warm tokens = %v, cold tokens = %v", warm, cold)
	}
	coldForwards := len(afterCold) - len(beforeCold)
	warmForwards := len(afterWarm) - len(afterCold)
	if warmForwards >= coldForwards {
		t.Fatalf("warm ran %d forwards, cold ran %d; prompt prefill was not restored", warmForwards, coldForwards)
	}
	// Negative control: a corrupted snapshot must reach the decode, so parity
	// above cannot come from ignoring the restored KV.
	snapshot := eng.prefixes.entries[prefixKey(promptTokens)]
	if snapshot == nil {
		t.Fatal("stored snapshot is missing")
	}
	for i := range snapshot.v[0] {
		snapshot.v[0][i] += 50
	}
	corrupted, err := eng.GenerateTokenIDs(context.Background(), cfg, prompt, 4)
	if err != nil {
		t.Fatal(err)
	}
	if reflect.DeepEqual(cold, corrupted) {
		t.Fatalf("corrupted snapshot still emitted %v; the restore is not observed", corrupted)
	}
	t.Logf("prompt_tokens=%d tokens=%v corrupted=%v cold_forwards=%d warm_forwards=%d cold=%+v warm=%+v",
		len(promptTokens), cold, corrupted, coldForwards, warmForwards, coldMetrics, warmMetrics)
}
