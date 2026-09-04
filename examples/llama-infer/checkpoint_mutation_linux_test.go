//go:build linux

package main

import (
	"math"
	"os"
	"testing"
)

func TestRealCheckpointCUDAKVMutation(t *testing.T) {
	path := os.Getenv("LLAMA_MODEL_PATH")
	if path == "" {
		t.Skip("LLAMA_MODEL_PATH is empty")
	}
	cfg := Llama3_2_1BConfig(path)
	cfg.MaxSeqLen = 8
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if eng.CheckpointReceipt().Verdict != "accepted" {
		t.Fatal("checkpoint receipt was not accepted")
	}
	cfg = cudaServingConfig(eng, cfg)
	baseline, _ := eng.memory.Snapshot()

	run := func(mutate bool) []float32 {
		cache, err := eng.NewSequenceKV(cfg)
		if err != nil {
			t.Fatal(err)
		}
		defer cache.Free()
		current, err := GPUPrefillPassKV(eng, cfg, cache, []int32{41, 42})
		if err != nil {
			t.Fatal(err)
		}
		if mutate {
			layer, err := cache.Cache.Layer(0)
			if err != nil {
				t.Fatal(err)
			}
			kvDim := cfg.NumKVHeads * cfg.HeadDim
			key := make([]float32, kvDim)
			populated := layer.K.Slice(0, kvDim)
			if err := populated.CopyToHost(key); err != nil {
				t.Fatal(err)
			}
			for i := range key {
				key[i] += 100
			}
			if err := populated.CopyFromHost(key); err != nil {
				t.Fatal(err)
			}
		}
		if _, err := GPUForwardPassKV(eng, cfg, cache, current, 2); err != nil {
			t.Fatal(err)
		}
		logits, err := eng.ReadLogits()
		if err != nil {
			t.Fatal(err)
		}
		if len(logits) != cfg.VocabSize {
			t.Fatalf("logits = %d elements, want %d", len(logits), cfg.VocabSize)
		}
		return logits
	}

	before := run(false)
	control := run(false)
	after := run(true)
	if maxAbsDifference(before, control) != 0 {
		t.Fatal("unmodified control changed full logits")
	}
	maxAbs := maxAbsDifference(before, after)
	if maxAbs == 0 {
		t.Fatal("populated KV mutation did not affect full logits")
	}
	if current, _ := eng.memory.Snapshot(); current != baseline {
		t.Fatalf("allocation current=%d want baseline=%d", current, baseline)
	}
	t.Logf("mutation=layer0/K/token0/all_coordinates delta=100 full_logits=%d max_abs=%g allocation_baseline=%d", len(after), maxAbs, baseline)
}

func maxAbsDifference(a, b []float32) float64 {
	var maximum float64
	for i := range a {
		maximum = max(maximum, math.Abs(float64(a[i]-b[i])))
	}
	return maximum
}
