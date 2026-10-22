//go:build linux

package main

import (
	"math"
	"os"
	"testing"
)

// TestCUDARawAttentionGroupEquivalence checks that evaluating layers 0 and 1 a
// group at a time agrees with the token-at-a-time raw attention it replaces.
// The two are not bit-identical -- the tokentile path accumulates a whole tile
// of tokens against a half-precision mirror of the raw window, where the raw
// kernel walks one token's rows in float -- so the gate is a relative bound,
// and the control below is a different prompt, which must not clear it.
func TestCUDARawAttentionGroupEquivalence(t *testing.T) {
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
	const width = 512
	if len(tokens) < 2*width {
		t.Fatalf("input holds %d tokens, want at least %d", len(tokens), 2*width)
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
	prefill := func(grouped bool, in []int) []float32 {
		t.Helper()
		r.disableRawAttentionBatch = !grouped
		if err := r.Reset(); err != nil {
			t.Fatal(err)
		}
		if err := r.EvalBatch(in, 0); err != nil {
			t.Fatalf("prefill (grouped=%v): %v", grouped, err)
		}
		logits := r.Logits()
		out := make([]float32, len(logits))
		copy(out, logits)
		return out
	}

	want := prefill(false, tokens[:width])
	scale := 0.0
	for _, v := range want {
		scale = math.Max(scale, math.Abs(float64(v)))
	}
	if scale == 0 {
		t.Fatal("reference logits are all zero")
	}
	deviation := func(got []float32) float64 {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("logit count %d, want %d", len(got), len(want))
		}
		if e := nonFiniteExtent(got); e.bad() {
			t.Fatalf("non-finite logits %v", e)
		}
		worst := 0.0
		for i, v := range got {
			worst = math.Max(worst, math.Abs(float64(v-want[i])))
		}
		return worst / scale
	}

	// 0.25 is set from the family this deviation belongs to rather than from
	// nothing. Measured over these 512 tokens against the token-at-a-time arm:
	// a repeat of the same arm is exactly 0, turning the tokentile path off for
	// layers 2..42 is 0.066, the query and key/value GEMM that shipped ahead of
	// this is 0.081, this change is 0.163, and a different prompt is 0.978.
	const tol = 0.25
	if d := deviation(prefill(true, tokens[:width])); d > tol {
		t.Fatalf("grouped raw attention deviates by %g of the logit scale, want at most %g", d, tol)
	} else {
		t.Logf("deviation %g of scale %g", d, scale)
	}

	// The control: a different prompt through the same grouped path has to
	// exceed the same bound, or the bound is one nothing can fail.
	if d := deviation(prefill(true, tokens[width:2*width])); d <= tol {
		t.Fatalf("a different prompt passed the bound at %g: the bound cannot fail", d)
	}
}

// TestCUDARawAttentionGroupedRoPE checks the dense RoPE boundary directly.
// Layers zero and one use dense base-10000 RoPE even when the four-token
// attention route is enabled. The test compares the committed raw K rows
// against the scalar route at both a zero and a nonzero start position.
func TestCUDARawAttentionGroupedRoPE(t *testing.T) {
	if os.Getenv("DS4_CUDA_RAW_ROPE_GATE") != "1" || os.Getenv("GPU_LOCK_HELD") == "" {
		t.Skip("DS4_CUDA_RAW_ROPE_GATE=1 and a GPU lock are required")
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
	if len(tokens) < 2048 {
		t.Fatalf("input holds %d tokens, want at least 2048", len(tokens))
	}
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

	readRaw := func(grouped bool, start int) []float32 {
		t.Helper()
		r.disableRawAttentionBatch = !grouped
		if err := r.Reset(); err != nil {
			t.Fatal(err)
		}
		if err := r.EvalBatch(tokens[:2048], start); err != nil {
			t.Fatalf("grouped=%v start=%d: %v", grouped, start, err)
		}
		row := make([]float32, r.state.layers[0].rawKV.Len())
		if err := r.state.layers[0].rawKV.CopyToHost(row); err != nil {
			t.Fatal(err)
		}
		return row
	}
	for _, start := range []int{0, 256} {
		want := readRaw(false, start)
		got := readRaw(true, start)
		worst := 0.0
		for i, v := range want {
			d := math.Abs(float64(v - got[i]))
			if d > worst {
				worst = d
			}
		}
		if worst > 2e-3 {
			t.Fatalf("start=%d grouped raw K differs by %g, want <= 0.002", start, worst)
		}
		t.Logf("start=%d grouped raw K max abs=%g", start, worst)
	}
}
