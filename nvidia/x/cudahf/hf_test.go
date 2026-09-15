package cudahf_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahf"
)

func TestResolveIn(t *testing.T) {
	tmpDir := t.TempDir()
	repoID := "meta-llama/Llama-3.2-1B-Instruct"

	snapshotPath := filepath.Join(tmpDir, "models--meta-llama--Llama-3.2-1B-Instruct", "snapshots", "commit123")
	if err := os.MkdirAll(snapshotPath, 0755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}

	_ = os.WriteFile(filepath.Join(snapshotPath, "config.json"), []byte("{}"), 0644)
	_ = os.WriteFile(filepath.Join(snapshotPath, "tokenizer.json"), []byte("{}"), 0644)
	_ = os.WriteFile(filepath.Join(snapshotPath, "model.safetensors"), []byte("data"), 0644)

	mf, err := cudahf.ResolveIn(tmpDir, repoID)
	if err != nil {
		t.Fatalf("ResolveIn failed: %v", err)
	}

	if mf.ConfigPath == "" {
		t.Error("expected ConfigPath to be resolved")
	}
	if mf.TokenizerPath == "" {
		t.Error("expected TokenizerPath to be resolved")
	}
	if len(mf.WeightsPaths) != 1 {
		t.Errorf("expected 1 weights path, got %d", len(mf.WeightsPaths))
	}
}
