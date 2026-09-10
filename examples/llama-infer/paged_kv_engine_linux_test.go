//go:build linux && !tpu

package main

// Engine-level qualification for LLAMA_INFER_PAGED_KV=1.
//
// TestPagedKVMatchesDense qualifies pagedKVRuntime below the engine. This test
// is the one the wiring commits to: a dense and a paged engine compiled from
// the same deterministic weights must produce the same tokens and logits on
// the single-stream path and on the batch path that serving and prefill use.
//
// It needs a CUDA device and LIB_CUDA_HARDWARE_TEST=1, like the other hardware
// tests here, and has never been executed by its author. A skip is not a pass.

import (
	"context"
	"math"
	"strings"
	"testing"
	"time"
)

func TestPagedKVEngineMatchesDense(t *testing.T) {
	requireCUDAHardware(t)
	cfg := Config{
		Name:       "PagedKVTestModel",
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
	t.Setenv("LLAMA_INFER_CUDA_GRAPH", "")
	t.Setenv("LLAMA_INFER_PAGED_KV", "")
	dense, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatalf("dense engine: %v", err)
	}
	t.Setenv("LLAMA_INFER_PAGED_KV", "1")
	t.Setenv("LLAMA_INFER_PAGED_KV_SEQS", "4")
	paged, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatalf("paged engine: %v", err)
	}
	if paged == dense {
		t.Fatal("the engine cache returned the dense engine for the paged switch")
	}
	if paged.paged == nil || paged.pagedMain == nil || paged.kvCache != nil {
		t.Fatal("paged engine did not replace the dense cache with the block pool")
	}
	if dense.paged != nil || dense.kvCache == nil {
		t.Fatal("dense engine carries paged state")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	closeLogits := func(t *testing.T, what string, got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s: %d logits, want %d", what, len(got), len(want))
		}
		worst := 0.0
		for i := range got {
			d := math.Abs(float64(got[i]) - float64(want[i]))
			if math.IsNaN(d) || d > 1e-3+1e-3*math.Abs(float64(want[i])) {
				t.Fatalf("%s: logit %d is %g, want %g", what, i, got[i], want[i])
			}
			worst = max(worst, d)
		}
		t.Logf("%s: %d logits, max abs difference %g", what, len(got), worst)
	}
	sameTokens := func(t *testing.T, what string, got, want []int32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s: %d tokens, want %d", what, len(got), len(want))
		}
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("%s: token %d is %d, want %d (got %v, want %v)", what, i, got[i], want[i], got, want)
			}
		}
	}

	// Single-stream: batched prefill on the engine's own sequence, then the
	// decode loop, twice over so the paged reset path runs too.
	const prompt = "paged kv"
	for round := 0; round < 2; round++ {
		wantTokens, err := dense.GenerateTokenIDs(ctx, cfg, prompt, 8)
		if err != nil {
			t.Fatalf("dense generate: %v", err)
		}
		wantLogits, err := dense.ReadLogits()
		if err != nil {
			t.Fatal(err)
		}
		if err := paged.ResetKV(); err != nil {
			t.Fatalf("paged reset: %v", err)
		}
		gotTokens, err := paged.GenerateTokenIDs(ctx, cfg, prompt, 8)
		if err != nil {
			t.Fatalf("paged generate: %v", err)
		}
		gotLogits, err := paged.ReadLogits()
		if err != nil {
			t.Fatal(err)
		}
		sameTokens(t, "single-stream tokens", gotTokens, wantTokens)
		closeLogits(t, "single-stream logits", gotLogits, wantLogits)
		if err := paged.paged.CheckStatus(); err != nil {
			t.Fatalf("single-stream round %d: %v", round, err)
		}
	}

	// Batch: two sequences of different lengths prefilled separately, then
	// decoded together for a few steps, exactly as the serving backend does.
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
			if err := eng.bindPagedBatch(sp, caches, positions); err != nil {
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
	wantTokens, wantLogits := run(t, dense)
	gotTokens, gotLogits := run(t, paged)
	sameTokens(t, "batch tokens", gotTokens, wantTokens)
	closeLogits(t, "batch logits", gotLogits, wantLogits)
	if err := paged.paged.CheckStatus(); err != nil {
		t.Fatalf("batch: %v", err)
	}

	// The pool must be whole again: every serving sequence freed its blocks,
	// and only the single-stream sequence still holds some.
	if free, total := paged.paged.cache.NumFreeBlocks(), paged.paged.cfg.TotalBlocks; free >= total || free < total-paged.paged.cfg.MaxBlocksPerSeq {
		t.Fatalf("pool has %d of %d blocks free after the sequences ended", free, total)
	}
}

// TestPagedKVRefusesGraph pins the exclusion of the two compile-time switches.
// It needs no device: the refusal happens before one is opened.
func TestPagedKVRefusesGraph(t *testing.T) {
	t.Setenv("LLAMA_INFER_PAGED_KV", "1")
	t.Setenv("LLAMA_INFER_CUDA_GRAPH", "1")
	cfg := Config{Name: "PagedKVGraphRefusal", Dim: 8, HiddenDim: 16, NumLayers: 1, NumHeads: 1, NumKVHeads: 1, HeadDim: 8, VocabSize: 8, MaxSeqLen: 8, TestMode: true}
	_, err := GetOrCompileEngine(cfg)
	if err == nil || !strings.Contains(err.Error(), "exclusive") {
		t.Fatalf("GetOrCompileEngine with both switches: err = %v, want the exclusivity refusal", err)
	}
}
