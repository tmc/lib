package tileiras_test

import (
	"context"
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

func TestAssembleIfBinaryPresent(t *testing.T) {
	if _, err := exec.LookPath("tileiras"); err != nil {
		t.Skip("tileiras executable not in PATH")
	}

	ctx := context.Background()
	dummyBytecode := []byte{0x7F, 'T', 'i', 'l', 'e', 'I', 'R', 0x00, 0x0D, 0x04}
	_, _ = tileiras.Assemble(ctx, dummyBytecode, tileiras.WithGPUName("sm_90"))
}
