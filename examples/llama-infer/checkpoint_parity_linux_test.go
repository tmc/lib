//go:build linux && !tpu

package main

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

const cudaParityCheckpointRevision = "5a8abab4a5d6f164389b1079fb721cfab8d7126c"

func requireCUDAParityCheckpoint(t *testing.T, path string, cfg *ModelConfig) {
	t.Helper()
	if err := validateCUDAParityCheckpoint(path, cfg); err != nil {
		t.Fatal(err)
	}
}

func validateCUDAParityCheckpoint(path string, cfg *ModelConfig) error {
	resolved, err := filepath.EvalSymlinks(path)
	if err != nil {
		return fmt.Errorf("resolve checkpoint path: %w", err)
	}
	if got := filepath.Base(filepath.Clean(resolved)); got != cudaParityCheckpointRevision {
		return fmt.Errorf("checkpoint revision = %q, want %q", got, cudaParityCheckpointRevision)
	}
	if cfg == nil {
		return fmt.Errorf("checkpoint architecture: nil config")
	}
	if cfg.ModelType != "llama" || cfg.HiddenSize != 2048 || cfg.IntermediateSize != 8192 ||
		cfg.NumHiddenLayers != 16 || cfg.NumAttentionHeads != 32 || cfg.NumKeyValueHeads != 8 ||
		cfg.HeadDim() != 64 || cfg.VocabSize != 128256 {
		return fmt.Errorf("checkpoint architecture = type %q dim %d hidden %d layers %d heads %d kv_heads %d head_dim %d vocab %d, want pinned Llama-3.2-1B-Instruct architecture",
			cfg.ModelType, cfg.HiddenSize, cfg.IntermediateSize, cfg.NumHiddenLayers,
			cfg.NumAttentionHeads, cfg.NumKeyValueHeads, cfg.HeadDim(), cfg.VocabSize)
	}
	return nil
}

func TestValidateCUDAParityCheckpoint(t *testing.T) {
	root := t.TempDir()
	pinned := filepath.Join(root, cudaParityCheckpointRevision)
	if err := os.Mkdir(pinned, 0o755); err != nil {
		t.Fatal(err)
	}
	cfg := &ModelConfig{
		ModelType:         "llama",
		HiddenSize:        2048,
		IntermediateSize:  8192,
		NumHiddenLayers:   16,
		NumAttentionHeads: 32,
		NumKeyValueHeads:  8,
		VocabSize:         128256,
	}
	if err := validateCUDAParityCheckpoint(pinned, cfg); err != nil {
		t.Fatalf("pinned checkpoint rejected: %v", err)
	}
	if err := validateCUDAParityCheckpoint(root, cfg); err == nil {
		t.Fatal("wrong revision accepted")
	}
	wrong := *cfg
	wrong.NumKeyValueHeads = 32
	if err := validateCUDAParityCheckpoint(pinned, &wrong); err == nil {
		t.Fatal("wrong architecture accepted")
	}
}

// TestRealCheckpointCUDAMatchesCPU is the CUDA counterpart of
// TestRealCheckpointMetalMatchesCPU. It compares every output logit from the
// dense CUDA forward path with a CPU reference that keeps its own KV history.
// This is indirect evidence for dense KV correctness, not an element-wise KV
// comparison.
func TestRealCheckpointCUDAMatchesCPU(t *testing.T) {
	path := os.Getenv("LLAMA_MODEL_PATH")
	if path == "" {
		t.Skip("LLAMA_MODEL_PATH is empty")
	}
	weights, modelConfig, err := LoadModel(path)
	if err != nil {
		t.Fatal(err)
	}
	requireCUDAParityCheckpoint(t, path, modelConfig)
	cfg := Llama3_2_1BConfig(path)
	cfg.Precision = "f32"
	cfg.MaxSeqLen = 8
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if got := eng.CheckpointReceipt().Verdict; got != "accepted" {
		t.Fatalf("checkpoint receipt verdict = %q", got)
	}

	cpuKV := NewKVCacheState(modelConfig.NumHiddenLayers, cfg.MaxSeqLen, modelConfig.NumKeyValueHeads*modelConfig.HeadDim())
	if err := eng.ResetKV(); err != nil {
		t.Fatal(err)
	}
	// Several positions: a KV cache that is written or read at the wrong
	// offset agrees at position 0 and diverges afterwards.
	for position, token := range []int32{41, 42, 43, 44} {
		want := Forward(weights, modelConfig, cpuKV, token, position)
		if _, err := GPUForwardPassKV(eng, cfg, nil, token, position); err != nil {
			t.Fatal(err)
		}
		got, err := eng.ReadLogits()
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != len(want) {
			t.Fatalf("position %d: %d logits, want %d", position, len(got), len(want))
		}
		maxAbs := maxAbsDifference(want, got)
		if maxAbs > 1e-3 {
			t.Fatalf("position %d full-logit max absolute difference %g exceeds 1e-3", position, maxAbs)
		}
		t.Logf("position=%d full_logits=%d max_abs=%g", position, len(got), maxAbs)
	}
}

// TestRealCheckpointCUDAKVDiverges is the null control for the logit parity test: a
// CPU reference whose KV state is discarded between positions must not match
// the GPU, so the comparison above is capable of failing.
func TestRealCheckpointCUDAKVDiverges(t *testing.T) {
	path := os.Getenv("LLAMA_MODEL_PATH")
	if path == "" {
		t.Skip("LLAMA_MODEL_PATH is empty")
	}
	weights, modelConfig, err := LoadModel(path)
	if err != nil {
		t.Fatal(err)
	}
	requireCUDAParityCheckpoint(t, path, modelConfig)
	cfg := Llama3_2_1BConfig(path)
	cfg.Precision = "f32"
	cfg.MaxSeqLen = 8
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if got := eng.CheckpointReceipt().Verdict; got != "accepted" {
		t.Fatalf("checkpoint receipt verdict = %q", got)
	}
	if err := eng.ResetKV(); err != nil {
		t.Fatal(err)
	}
	kvDim := modelConfig.NumKeyValueHeads * modelConfig.HeadDim()
	var maxAbs float64
	for position, token := range []int32{41, 42, 43, 44} {
		// A fresh CPU cache each step: history is lost.
		cpuKV := NewKVCacheState(modelConfig.NumHiddenLayers, cfg.MaxSeqLen, kvDim)
		want := Forward(weights, modelConfig, cpuKV, token, position)
		if _, err := GPUForwardPassKV(eng, cfg, nil, token, position); err != nil {
			t.Fatal(err)
		}
		got, err := eng.ReadLogits()
		if err != nil {
			t.Fatal(err)
		}
		maxAbs = math.Max(maxAbs, maxAbsDifference(want, got))
		_ = position
	}
	if maxAbs <= 1e-3 {
		t.Fatalf("history-free reference still matched the GPU within %g; the parity test cannot fail", maxAbs)
	}
	t.Logf("history-free reference diverges by max_abs=%g", maxAbs)
}
