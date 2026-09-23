//go:build darwin

package main

import (
	"encoding/json"
	"math"
	"os"
	"strconv"
	"testing"
)

func TestRealCheckpointMetalDefaultReceipt(t *testing.T) {
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
	receipt := eng.CheckpointReceipt()
	if receipt.Verdict != "accepted" || receipt.TargetType != "F16" || len(receipt.Tensors) != 147 {
		t.Fatalf("receipt=%+v", receipt)
	}
	b, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("metal_checkpoint_receipt=%s", b)
}

func TestRealCheckpointMetalMatchesCPU(t *testing.T) {
	path := os.Getenv("LLAMA_MODEL_PATH")
	if path == "" {
		t.Skip("LLAMA_MODEL_PATH is empty")
	}
	weights, modelConfig, err := LoadModel(path)
	if err != nil {
		t.Fatal(err)
	}
	cfg := Llama3_2_1BConfig(path)
	cfg.Precision = "f32"
	cfg.MaxSeqLen = 8
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	cpuKV := NewKVCacheState(modelConfig.NumHiddenLayers, cfg.MaxSeqLen, modelConfig.NumKeyValueHeads*modelConfig.HeadDim())
	eng.ResetKV()
	for position, token := range []int32{41, 42} {
		want := Forward(weights, modelConfig, cpuKV, token, position)
		if _, err := GPUForwardPassKV(eng, cfg, nil, token, position); err != nil {
			t.Fatal(err)
		}
		got, err := eng.ReadLogits()
		if err != nil {
			t.Fatal(err)
		}
		var maxAbs float64
		for i := range want {
			maxAbs = max(maxAbs, math.Abs(float64(want[i]-got[i])))
		}
		if maxAbs > 1e-3 {
			t.Fatalf("position %d full-logit max absolute difference %g exceeds 1e-3", position, maxAbs)
		}
		t.Logf("position=%d full_logits=%d max_abs=%g", position, len(got), maxAbs)
	}
}

func TestRealCheckpointMetalMutation(t *testing.T) {
	path := os.Getenv("LLAMA_MODEL_PATH")
	if path == "" {
		t.Skip("LLAMA_MODEL_PATH is empty")
	}
	cfg := Llama3_2_1BConfig(path)
	cfg.Precision = "f32"
	cfg.MaxSeqLen = 8
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if eng.CheckpointReceipt().Verdict != "accepted" {
		t.Fatal("checkpoint receipt was not accepted")
	}
	eng.ResetKV()
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
	for layer := range cfg.NumLayers {
		t.Run("layer="+strconv.Itoa(layer), func(t *testing.T) {
			restore, err := eng.MutateWeightForTest(layer, 0, 10)
			if err != nil {
				t.Fatal(err)
			}
			defer restore()
			eng.ResetKV()
			if _, err = GPUForwardPassKV(eng, cfg, nil, 41, 0); err != nil {
				t.Fatal(err)
			}
			if _, err = GPUForwardPassKV(eng, cfg, nil, 42, 1); err != nil {
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
				t.Fatal("query weight mutation did not affect logits")
			}
			t.Logf("layer=%d query_weight_mutation_max_abs=%g", layer, maxAbs)
		})
	}
}
