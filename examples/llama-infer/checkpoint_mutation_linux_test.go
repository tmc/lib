//go:build linux

package main

import (
	"math"
	"os"
	"testing"
)

func TestRealCheckpointCUDAMutation(t *testing.T) {
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
	if err := eng.ResetKV(); err != nil {
		t.Fatal(err)
	}
	if _, err = GPUForwardPassKV(eng, cfg, nil, 41, 0); err != nil {
		t.Fatal(err)
	}
	_, err = GPUForwardPassKV(eng, cfg, nil, 42, 1)
	if err != nil {
		t.Fatal(err)
	}
	before, err := eng.ReadLogits()
	if err != nil {
		t.Fatal(err)
	}
	restore, err := eng.MutateWeightForTest(0, 0, 10)
	if err != nil {
		t.Fatal(err)
	}
	defer restore()
	if err := eng.ResetKV(); err != nil {
		t.Fatal(err)
	}
	if _, err = GPUForwardPassKV(eng, cfg, nil, 41, 0); err != nil {
		t.Fatal(err)
	}
	_, err = GPUForwardPassKV(eng, cfg, nil, 42, 1)
	if err != nil {
		t.Fatal(err)
	}
	after, err := eng.ReadLogits()
	if err != nil {
		t.Fatal(err)
	}
	var maxAbs float64
	for i := range before {
		maxAbs = max(maxAbs, math.Abs(float64(before[i]-after[i])))
	}
	if maxAbs == 0 {
		t.Fatal("weight mutation did not affect logits")
	}
	t.Logf("mutation_max_abs=%g", maxAbs)
}
