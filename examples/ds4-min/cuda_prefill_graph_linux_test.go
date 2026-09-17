//go:build linux

package main

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"testing"
	"time"
)

// TestCUDAPrefillCUDAGraph tests instantiated CUDA graph capture and replay for prefill batch execution.
// It verifies that prefill with CUDA graphs produces exact canonical continuation tokens
// (15342, 201), valid finite logits across all 43 layers, preserves layer compressed counts,
// and produces the canonical Target #3 logit digest 51c9b4ced3bc92c098b184981f6fcf5175b6ad5aed066aa5abc6fd77ed49ab24.
func TestCUDAPrefillCUDAGraph(t *testing.T) {
	gate := os.Getenv("DS4_CUDA_PREFILL_DENSE_D2R_GATE")
	if gate == "" {
		t.Skip("DS4_CUDA_PREFILL_DENSE_D2R_GATE is not set")
	}
	if gate != "1" || os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("DS4_CUDA_PREFILL_DENSE_D2R_GATE=1 and a GPU lock are required")
	}
	model, input := os.Getenv("DS4_MODEL"), os.Getenv("DS4_CUDA_PREFILL_TOKENS")
	if model == "" || input == "" {
		t.Fatal("DS4_MODEL and DS4_CUDA_PREFILL_TOKENS are required")
	}
	data, err := os.ReadFile(input)
	if err != nil {
		t.Fatal(err)
	}
	tokens, err := parsePrefillTokens(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(tokens) != 2048 {
		t.Fatalf("input contains %d tokens, want 2048", len(tokens))
	}
	r, err := newCUDARuntime(model, 4096)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := r.Close(); err != nil {
			t.Error(err)
		}
	}()

	r.enableDenseD2R = true
	r.enableD2RExperts = true
	r.enableAttentionHMMA = true
	if err := r.Reset(); err != nil {
		t.Fatal(err)
	}

	start := time.Now()
	if err := r.EvalBatch(tokens[:2048], 0); err != nil {
		t.Fatalf("CUDA graph prefill 2048: %v", err)
	}
	elapsed := time.Since(start)

	// Verify that CUDA graphs were captured for all layers.
	if r.batch == nil {
		t.Fatal("expected batch resources to be allocated")
	}
	for i := 0; i < cudaLayers; i++ {
		if r.batch.graphs[i].qkv == nil {
			t.Fatalf("layer %d qkv batch graph was not instantiated", i)
		}
		if r.batch.graphs[i].ffn == nil {
			t.Fatalf("layer %d ffn batch graph was not instantiated", i)
		}
	}

	logits := r.Logits()
	if len(logits) != 129280 {
		t.Fatalf("logits length %d, want 129280", len(logits))
	}
	raw := make([]byte, len(logits)*4)
	for i, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("non-finite logit %d", i)
		}
		binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(v))
	}
	digest := fmt.Sprintf("%x", sha256.Sum256(raw))
	const wantDigest = "51c9b4ced3bc92c098b184981f6fcf5175b6ad5aed066aa5abc6fd77ed49ab24"
	if digest != wantDigest {
		t.Fatalf("logit digest: got %s, want %s", digest, wantDigest)
	}

	// Verify continuation tokens: 15342 then 201.
	tok0 := greedyToken(logits)
	if tok0 != 15342 {
		t.Fatalf("continuation token 0: got %d, want 15342", tok0)
	}
	if err := r.Eval(tok0, 2048); err != nil {
		t.Fatalf("eval continuation token 0: %v", err)
	}
	tok1 := greedyToken(r.Logits())
	if tok1 != 201 {
		t.Fatalf("continuation token 1: got %d, want 201", tok1)
	}

	// Verify compressed counts across all 43 layers.
	for l := 2; l < len(r.state.layers); l++ {
		nComp := r.state.layers[l].attn.n
		if nComp <= 0 {
			t.Fatalf("layer %d compressed count = %d, want > 0", l, nComp)
		}
	}

	toksPerSec := float64(len(tokens)) / elapsed.Seconds()
	t.Logf("Prefill CUDA graph verified: elapsed=%v, throughput=%.2f tok/s, digest=%s, tokens=[%d, %d]",
		elapsed, toksPerSec, digest, tok0, tok1)
}
