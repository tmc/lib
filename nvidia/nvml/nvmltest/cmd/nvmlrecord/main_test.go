package main

import (
	"path/filepath"
	"testing"
)

func TestAdmissionRefusalBeforeNVML(t *testing.T) {
	for _, tt := range []struct{ pid, source string }{
		{"", "123"}, {"0", "123"}, {"not pid", "123"}, {"-1", "123"}, {"1", ""}, {"1", "not hex"}, {"1", "12"},
	} {
		t.Setenv("GPU_LOCK_HELD", tt.pid)
		if err := run(0, tt.source, filepath.Join(t.TempDir(), "receipt.json")); err == nil {
			t.Fatal("accepted invalid admission")
		}
	}
}
