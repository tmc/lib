package main

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// TestRunNCCL runs runNCCL against fake child probes.
func TestRunNCCL(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("needs a shell script")
	}
	tests := []struct {
		name    string
		script  string
		results int
		err     string
	}{
		{"ok", `echo '[{"op":"x","ranks":1,"bytes":1024}]'`, 2, ""},
		{"exit", `echo boom >&2; exit 3`, 0, "boom"},
		{"hang", `exec sleep 60`, 0, "timed out after 2s"},
		{"json", `echo nope`, 0, "invalid character"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			exe := filepath.Join(t.TempDir(), "probe")
			if err := os.WriteFile(exe, []byte("#!/bin/sh\n"+tt.script+"\n"), 0o755); err != nil {
				t.Fatal(err)
			}
			defer func(f func() (string, error)) { executable = f }(executable)
			executable = func() (string, error) { return exe, nil }
			rs, err := runNCCL(context.Background(), 1, 1<<10, 2*time.Second)
			if len(rs) != tt.results {
				t.Errorf("got %d results, want %d", len(rs), tt.results)
			}
			switch {
			case tt.err == "" && err != nil:
				t.Errorf("err = %v, want nil", err)
			case tt.err != "" && (err == nil || !strings.Contains(err.Error(), tt.err)):
				t.Errorf("err = %v, want %q", err, tt.err)
			}
		})
	}
}
