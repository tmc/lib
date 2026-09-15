package main

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// TestLoadDisassemblyNamesTheWayToGetSASS keeps the refusal actionable.
// cudaprof capture embeds journals and handshakes only, so a capture carries
// SASS only when the workload wrote it and "cudaprof -pack" embedded it.
func TestLoadDisassemblyNamesTheWayToGetSASS(t *testing.T) {
	path := writeGapCapture(t, filepath.Join(t.TempDir(), "nosass.pprof"), pacedJournal(8_000, 1_000), completeStatus())
	_, err := loadDisassembly(path, cudacapture.Limits{})
	if err == nil {
		t.Fatal("loadDisassembly found SASS in a capture that has none")
	}
	for _, want := range []string{"cudaprof -pack", "kernels.sass", "cudaprof diff base.sass target.sass"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("refusal does not mention %q: %v", want, err)
		}
	}
}
