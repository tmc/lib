//go:build darwin

package metalsafetensors

import (
	"os"
	"path/filepath"
	"testing"
)

func TestOpenSafetensorsLlama3_2(t *testing.T) {
	home, _ := os.UserHomeDir()
	pattern := filepath.Join(home, ".cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/*/model.safetensors")
	matches, _ := filepath.Glob(pattern)
	if len(matches) == 0 {
		t.Skip("local LLaMA 3.2 1B safetensors not found in cache")
	}

	w, err := Open(matches[0])
	if err != nil {
		t.Fatalf("Open safetensors failed: %v", err)
	}
	defer w.Close()

	if len(w.Tensors) == 0 {
		t.Fatalf("no tensors parsed from header")
	}

	t.Logf("Successfully parsed %d tensors from LLaMA 3.2 1B safetensors:", len(w.Tensors))
	checkTensors := []string{
		"model.embed_tokens.weight",
		"model.layers.0.self_attn.q_proj.weight",
		"model.layers.0.mlp.gate_proj.weight",
		"model.norm.weight",
		"lm_head.weight",
	}

	for _, name := range checkTensors {
		if info, ok := w.Tensors[name]; ok {
			t.Logf("  Found tensor: %-40s Shape=%v DType=%s (Size: %d bytes)",
				name, info.Shape, info.DType, info.OffsetEnd-info.OffsetStart)
		}
	}

	// Verify decoding first embedding row
	embed, err := w.TensorFloat32("model.embed_tokens.weight")
	if err != nil {
		t.Fatalf("TensorFloat32 failed: %v", err)
	}

	t.Logf("Decoded embed_tokens (%d floats). First 4 values: %v", len(embed), embed[:4])
	if embed[0] == 0 && embed[1] == 0 && embed[2] == 0 {
		t.Errorf("unexpected zero weights in embed_tokens")
	}
}
