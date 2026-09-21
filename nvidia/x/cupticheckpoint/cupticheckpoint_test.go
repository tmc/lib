package cupticheckpoint

import (
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
)

func TestVersionQualification(t *testing.T) {
	tests := []struct {
		name    string
		version uint32
		want    bool
	}{
		{name: "CUDA 12.6 API version", version: 24, want: true},
		{name: "CUDA 12.8 API version", version: 26, want: true},
		{name: "CUDA 12.9 API version", version: 28, want: true},
		{name: "CUDA 13.0 API version", version: 130001, want: true},
		{name: "CUDA 13.1 API version", version: 130101, want: true},
		{name: "CUDA 13.2 header version", version: 130200, want: true},
		{name: "CUDA 13.2 runtime version", version: 130202, want: true},
		{name: "CUDA 13.3 API version", version: 130301, want: true},
		{name: "CUDA 13.4 API version", version: 130400, want: true},
		{name: "zero version", version: 0, want: false},
		{name: "unmeasured 130100", version: 130100, want: false},
		{name: "unmeasured 130201", version: 130201, want: false},
		{name: "unmeasured 130500", version: 130500, want: false},
		{name: "unmeasured CUDA 14", version: 140000, want: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := qualifiedVersion(tt.version); got != tt.want {
				t.Errorf("qualifiedVersion(%d) = %v, want %v", tt.version, got, tt.want)
			}
		})
	}
}

func TestOptimizations(t *testing.T) {
	if got, want := OptNone.String(), "None"; got != want {
		t.Errorf("OptNone.String() = %q, want %q", got, want)
	}
	if got, want := OptTransfer.String(), "Transfer"; got != want {
		t.Errorf("OptTransfer.String() = %q, want %q", got, want)
	}
	if got, want := Optimizations(99).String(), "Optimizations(99)"; got != want {
		t.Errorf("Optimizations(99).String() = %q, want %q", got, want)
	}
}

func TestFreedState(t *testing.T) {
	c := &Checkpoint{freed: true}
	if err := c.Restore(); err == nil {
		t.Error("Restore on freed checkpoint expected error, got nil")
	}
	if err := c.Free(); err != nil {
		t.Errorf("idempotent Free returned error: %v", err)
	}
}

// TestGeneratedVersionIsQualified pins the qualification table to the version
// the Go types were generated from. cupti.ApiVersion is that version, and
// the structs this package passes to CUPTI are its shapes.
//
// The table holds version literals, so nothing otherwise connects it to
// [cupti.ApiVersion] -- a regeneration moves the constant and leaves every
// literal, and neither the compiler nor any other test notices. Whatever else
// the table admits, it must admit the version whose struct shapes these types
// actually are.
func TestGeneratedVersionIsQualified(t *testing.T) {
	if !qualifiedVersion(cupti.ApiVersion) {
		t.Errorf("qualifiedVersion(cupti.ApiVersion = %d) = false, want true", cupti.ApiVersion)
	}
}
