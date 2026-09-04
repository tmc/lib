//go:build linux

package main

import (
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"testing"
)

const cudaAttentionCheckpointRevision = "5a8abab4a5d6f164389b1079fb721cfab8d7126c"

var cudaAttentionCheckpointFiles = []checkpointFileIdentity{
	{"config.json", "dfb67fd8afe73a1c75245824ef9d64a6ba8983025447e3bf76aa1ea57ee46152"},
	{"tokenizer.json", "6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b"},
	{"model.safetensors", "1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f"},
}

type checkpointFileIdentity struct {
	name   string
	sha256 string
}

func requireCUDAAttentionCheckpoint(t *testing.T, path string, cfg *ModelConfig) {
	t.Helper()
	if err := validateCUDAAttentionCheckpoint(path, cfg, cudaAttentionCheckpointFiles); err != nil {
		t.Fatal(err)
	}
}

func validateCUDAAttentionCheckpoint(path string, cfg *ModelConfig, files []checkpointFileIdentity) error {
	resolved, err := filepath.EvalSymlinks(path)
	if err != nil {
		return fmt.Errorf("resolve checkpoint path: %w", err)
	}
	if got := filepath.Base(filepath.Clean(resolved)); got != cudaAttentionCheckpointRevision {
		return fmt.Errorf("checkpoint revision = %q, want %q", got, cudaAttentionCheckpointRevision)
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
	for _, file := range files {
		got, err := sha256File(filepath.Join(resolved, file.name))
		if err != nil {
			return fmt.Errorf("checkpoint %s: %w", file.name, err)
		}
		if got != file.sha256 {
			return fmt.Errorf("checkpoint %s sha256 = %s, want %s", file.name, got, file.sha256)
		}
	}
	return nil
}

func sha256File(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return fmt.Sprintf("%x", h.Sum(nil)), nil
}

func pinnedCUDAAttentionConfig() *ModelConfig {
	return &ModelConfig{
		ModelType:         "llama",
		HiddenSize:        2048,
		IntermediateSize:  8192,
		NumHiddenLayers:   16,
		NumAttentionHeads: 32,
		NumKeyValueHeads:  8,
		VocabSize:         128256,
	}
}

func TestValidateCUDAAttentionCheckpointRejectsMutatedContents(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, cudaAttentionCheckpointRevision)
	if err := os.Mkdir(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	contents := map[string][]byte{
		"config.json":       []byte("config"),
		"tokenizer.json":    []byte("tokenizer"),
		"model.safetensors": []byte("weights"),
	}
	files := make([]checkpointFileIdentity, 0, len(contents))
	for name, data := range contents {
		if err := os.WriteFile(filepath.Join(dir, name), data, 0o644); err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(data)
		files = append(files, checkpointFileIdentity{name, fmt.Sprintf("%x", sum)})
	}
	if err := validateCUDAAttentionCheckpoint(dir, pinnedCUDAAttentionConfig(), files); err != nil {
		t.Fatalf("matching contents rejected: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), []byte("mutated"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := validateCUDAAttentionCheckpoint(dir, pinnedCUDAAttentionConfig(), files); err == nil {
		t.Fatal("mutated weights accepted")
	}
}
