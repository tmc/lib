//go:build ds4cpumodel && ds4cpuactualmodel

package dsv4cpu

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestCPUForwardOutputExclusive(t *testing.T) {
	path := filepath.Join(t.TempDir(), "logits.f32")
	want := []byte{1, 2, 3, 4}
	if err := writeCPUForwardOutput(path, want); err != nil {
		t.Fatal(err)
	}
	if err := writeCPUForwardOutput(path, []byte{9}); !os.IsExist(err) {
		t.Fatal("existing receipt not refused", err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("prior receipt changed", got)
	}
}

func TestCPUForwardReferenceIdentity(t *testing.T) {
	for _, raw := range [][]byte{nil, make([]byte, 129280*4)} {
		if err := checkCPUForwardReference(raw); err == nil {
			t.Fatal("wrong reference identity accepted")
		}
	}
}
