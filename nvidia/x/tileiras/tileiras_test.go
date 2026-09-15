package tileiras_test

import (
	"context"
	"errors"
	"os/exec"
	"testing"

	"github.com/tmc/lib/nvidia/x/tileiras"
)

func TestIsBytecode(t *testing.T) {
	tests := []struct {
		name string
		data []byte
		want bool
	}{
		{
			name: "valid magic",
			data: []byte{0x7F, 'T', 'i', 'l', 'e', 'I', 'R', 0x00, 0x0D, 0x04},
			want: true,
		},
		{
			name: "invalid prefix",
			data: []byte("entry @kernel() {}"),
			want: false,
		},
		{
			name: "short",
			data: []byte{0x7F, 'T'},
			want: false,
		},
		{
			name: "empty",
			data: nil,
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tileiras.IsBytecode(tt.data); got != tt.want {
				t.Errorf("IsBytecode() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestAssembleEmptyInput(t *testing.T) {
	ctx := context.Background()
	_, err := tileiras.Assemble(ctx, nil)
	if err == nil {
		t.Error("expected error on nil input")
	}
}

func TestAssembleBinaryMissing(t *testing.T) {
	ctx := context.Background()
	dummyBytecode := []byte{0x7F, 'T', 'i', 'l', 'e', 'I', 'R', 0x00, 0x0D, 0x04}
	_, err := tileiras.Assemble(ctx, dummyBytecode, tileiras.WithPath("/nonexistent/path/to/tileiras"))
	if err == nil {
		t.Error("expected error when tileiras binary is missing")
	}
}

// TestAssembleGPUNameSuffix checks that Assemble rejects a suffixed --gpu-name
// before invoking the assembler.
//
// The accept and reject rows were measured against
// /usr/local/cuda-13.4/bin/tileiras on a GB10 host, assembling a genuine 13.4
// .tilebc so that the architecture name was the only variable:
//
//	sm_90 sm_100 sm_121      rc=0
//	sm_90a sm_100a sm_121a   rc=1  Cannot find option named '...'!
//	sm_121f compute_90       rc=1  Cannot find option named '...'!
//
// --gpu-name is a closed enum; tileiras --help lists sm_80, sm_86, sm_87,
// sm_88, sm_89, sm_90, sm_100, sm_103, sm_107, sm_110, sm_120 and sm_121, with
// no suffixed spelling at any architecture. The guard rejects only the suffix
// shape rather than enumerating that list, so an architecture added by a later
// toolkit reaches the assembler and gets its own error. compute_90 is the
// deliberate gap: the tool rejects it, the guard does not, because the guard
// makes no claim about non-"sm_" spellings.
func TestAssembleGPUNameSuffix(t *testing.T) {
	tests := []struct {
		name    string
		gpuName string
		want    bool // want ErrArchSuffix
	}{
		{name: "measured reject", gpuName: "sm_121a", want: true},
		{name: "measured accept", gpuName: "sm_121", want: false},
		{name: "family suffix", gpuName: "sm_121f", want: true},
		{name: "hopper suffix", gpuName: "sm_90a", want: true},
		{name: "plain", gpuName: "sm_90", want: false},
		{name: "not an sm name", gpuName: "compute_90", want: false},
	}

	ctx := context.Background()
	dummyBytecode := []byte{0x7F, 'T', 'i', 'l', 'e', 'I', 'R', 0x00, 0x0D, 0x04}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// WithPath names a binary that does not exist, so a name that passes
			// the guard fails later at exec: the guard must fire before that.
			_, err := tileiras.Assemble(ctx, dummyBytecode,
				tileiras.WithGPUName(tt.gpuName),
				tileiras.WithPath("/nonexistent/path/to/tileiras"))
			if got := errors.Is(err, tileiras.ErrArchSuffix); got != tt.want {
				t.Errorf("Assemble(%q) err = %v, errors.Is(ErrArchSuffix) = %v, want %v", tt.gpuName, err, got, tt.want)
			}
		})
	}
}

func TestAssembleIfBinaryPresent(t *testing.T) {
	if _, err := exec.LookPath("tileiras"); err != nil {
		t.Skip("tileiras executable not in PATH")
	}

	ctx := context.Background()
	dummyBytecode := []byte{0x7F, 'T', 'i', 'l', 'e', 'I', 'R', 0x00, 0x0D, 0x04}
	_, _ = tileiras.Assemble(ctx, dummyBytecode, tileiras.WithGPUName("sm_90"))
}
