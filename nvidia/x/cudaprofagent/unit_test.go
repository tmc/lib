package cudaprofagent_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprofagent"
)

func TestWriteUnit(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("CUDAPROF_TRANSPORT", dir)
	published, err := cudaprofagent.WriteUnit("fused_gate", []byte(`{"Name":"fused_gate"}`))
	if err != nil || !published {
		t.Fatalf("WriteUnit = %v, %v", published, err)
	}
	data, err := os.ReadFile(filepath.Join(dir, "units", "fused_gate.json"))
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != `{"Name":"fused_gate"}` {
		t.Errorf("published %q", data)
	}
	// Republishing replaces, and leaves no temporary behind: the capture
	// harvests this directory by listing it.
	if _, err := cudaprofagent.WriteUnit("fused_gate", []byte(`{"Name":"fused_gate","Source":"x"}`)); err != nil {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(filepath.Join(dir, "units"))
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 {
		t.Errorf("units directory holds %d files, want 1", len(entries))
	}
}

// TestWriteUnitWithoutCapture is the case every target hits when it is run
// normally: no capture, nothing written, no error to handle.
func TestWriteUnitWithoutCapture(t *testing.T) {
	t.Setenv("CUDAPROF_TRANSPORT", "")
	published, err := cudaprofagent.WriteUnit("fused_gate", []byte(`{}`))
	if published || err != nil {
		t.Fatalf("WriteUnit = %v, %v, want false, nil", published, err)
	}
}

// TestWriteUnitRejectsPath keeps a generated name from naming a file outside
// the capture.
func TestWriteUnitRejectsPath(t *testing.T) {
	t.Setenv("CUDAPROF_TRANSPORT", t.TempDir())
	for _, name := range []string{"", "../escape", "sub/unit"} {
		if _, err := cudaprofagent.WriteUnit(name, []byte(`{}`)); err == nil {
			t.Errorf("WriteUnit(%q) was accepted", name)
		}
	}
}
