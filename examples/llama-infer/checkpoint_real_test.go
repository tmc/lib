package main

import (
	"encoding/json"
	"os"
	"testing"
)

func TestRealCheckpointConsumption(t *testing.T) {
	path := os.Getenv("LLAMA_MODEL_PATH")
	if path == "" {
		t.Skip("LLAMA_MODEL_PATH is empty")
	}
	weights, _, err := LoadModel(path)
	if err != nil {
		t.Fatal(err)
	}
	receipt := weights.Consumption
	if receipt.Verdict != "accepted" || len(receipt.Tensors) != 147 {
		t.Fatalf("verdict=%q tensors=%d", receipt.Verdict, len(receipt.Tensors))
	}
	counts := make(map[string]int)
	for _, tensor := range receipt.Tensors {
		counts[tensor.Action]++
	}
	if counts["converted"] != 146 || counts["tied"] != 1 {
		t.Fatalf("actions=%v", counts)
	}
	b, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("checkpoint_receipt=%s", b)
	t.Logf("checkpoint_tensors=%d actions=%v", len(receipt.Tensors), counts)
}
