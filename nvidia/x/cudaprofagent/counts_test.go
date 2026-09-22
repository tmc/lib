package cudaprofagent_test

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprofagent"
)

func TestWriteCounts(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("CUDAPROF_TRANSPORT", dir)
	image := []byte("CUBIN")
	counts := []cudaprofagent.Count{{Function: "fused_gate", Offset: 0x10, Value: 4096}}
	published, err := cudaprofagent.WriteCounts("fused_gate", image, "smsp__sass_inst_executed", counts)
	if err != nil || !published {
		t.Fatalf("WriteCounts = %v, %v", published, err)
	}
	data, err := os.ReadFile(filepath.Join(dir, "counts", "fused_gate.json"))
	if err != nil {
		t.Fatal(err)
	}
	var got cudaprofagent.Counts
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(image)
	if got.Module != hex.EncodeToString(sum[:]) {
		t.Errorf("module digest = %q", got.Module)
	}
	if got.Metric != "smsp__sass_inst_executed" || len(got.Counts) != 1 || got.Counts[0].Value != 4096 {
		t.Errorf("published %+v", got)
	}
}

// Counts that name no image bind to no module, so they are refused rather
// than published as counts of something unidentified.
func TestWriteCountsNeedsAnImage(t *testing.T) {
	t.Setenv("CUDAPROF_TRANSPORT", t.TempDir())
	if _, err := cudaprofagent.WriteCounts("fused_gate", nil, "m", nil); err == nil {
		t.Error("no error for counts with no module image")
	}
}

// The case every target hits when it is run normally: no capture, nothing
// written, no error to handle.
func TestWriteCountsWithoutCapture(t *testing.T) {
	t.Setenv("CUDAPROF_TRANSPORT", "")
	published, err := cudaprofagent.WriteCounts("fused_gate", []byte("CUBIN"), "m", nil)
	if published || err != nil {
		t.Fatalf("WriteCounts = %v, %v", published, err)
	}
}

func TestWriteCountsRejectsPath(t *testing.T) {
	t.Setenv("CUDAPROF_TRANSPORT", t.TempDir())
	for _, name := range []string{"", "../escape", "sub/gate"} {
		if _, err := cudaprofagent.WriteCounts(name, []byte("CUBIN"), "m", nil); err == nil {
			t.Errorf("no error for name %q", name)
		}
	}
}
