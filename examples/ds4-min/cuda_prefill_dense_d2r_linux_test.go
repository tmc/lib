//go:build linux

package main

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"testing"
)

// TestCUDAPrefillDenseD2R tests the fused Direct-to-Register (D2R) dense Q8_0 Tensor Core GEMMs.
// It verifies that dense D2R produces exact canonical continuation tokens
// (15342, 201), valid finite logits across all 43 layers, and preserves layer compressed counts.
func TestCUDAPrefillDenseD2R(t *testing.T) {
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
	// Select the arm through the user's door; see cudaShipDenseD2R.
	t.Setenv("DS4_CUDA_DENSE_D2R", "1")
	t.Setenv("DS4_CUDA_MOE_D2R", "1")
	t.Setenv("DS4_CUDA_ATTENTION_HMMA", "1")
	r, err := newCUDARuntime(model, 4096)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := r.Close(); err != nil {
			t.Error(err)
		}
	}()

	if err := r.Reset(); err != nil {
		t.Fatal(err)
	}
	if err := r.EvalBatch(tokens[:2048], 0); err != nil {
		t.Fatalf("Dense D2R prefill 2048: %v", err)
	}

	logits := r.Logits()
	if len(logits) != 129280 {
		t.Fatalf("logits length %d, want 129280", len(logits))
	}
	raw := make([]byte, len(logits)*4)
	if e := nonFiniteExtent(logits); e.bad() {
		t.Fatalf("non-finite logits %v", e)
	}
	for i, v := range logits {
		binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(v))
	}
	digest := fmt.Sprintf("%x", sha256.Sum256(raw))

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

	t.Logf("Dense D2R prefill verified: digest=%s, tokens=[%d, %d]", digest, tok0, tok1)
}
