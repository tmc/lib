//go:build ds4cpumodel && ds4cpuactualmodel

package dsv4cpu

import (
	"os"
	"testing"
)

// This admission stage binds the actual pinned GGUF; it does not run forward.
func TestFlashRealModelBinding(t *testing.T) {
	path := os.Getenv("DS4_CPU_MODEL")
	if path == "" {
		t.Fatal("DS4_CPU_MODEL is required for actual-model binding")
	}
	const digest = "ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0"
	b, err := openFlashBackend(path, digest)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := b.Close(); err != nil {
			t.Errorf("close model: %v", err)
		}
	}()
	s, err := b.NewState(256)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := s.Close(); err != nil {
			t.Errorf("close state: %v", err)
		}
	}()
	if len(b.model.layers) != 43 || len(b.Logits(s)) != 129280 {
		t.Fatal("real binding or request shape differs")
	}
	if err := b.Close(); err == nil {
		t.Fatal("model closed with live State")
	}
	t.Log("actual immutable GGUF bound: 43 layers, context256; no forward execution")
}
