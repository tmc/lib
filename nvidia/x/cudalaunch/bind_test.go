package cudalaunch_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

func TestBindInvalidTarget(t *testing.T) {
	type kernels struct {
		Embed driver.Function
	}
	type tagged struct {
		Count int `cuda:"Count"`
	}
	type hidden struct {
		embed driver.Function
	}

	tests := []struct {
		name    string
		target  any
		wantErr string
	}{
		{"nil", nil, "non-nil pointer"},
		{"not a pointer", kernels{}, "non-nil pointer"},
		{"nil pointer", (*kernels)(nil), "non-nil pointer"},
		{"pointer to non-struct", new(int), "pointer to struct"},
		{"tagged non-function field", &tagged{}, "want driver.Function"},
		{"unexported field", &hidden{}, "is unexported"},
		{"no kernel fields", &struct{ N int }{}, "no driver.Function fields"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := cudalaunch.Bind(0, tt.target)
			if err == nil {
				t.Fatalf("Bind(0, %T) = nil, want error", tt.target)
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("Bind(0, %T) = %v, want error containing %q", tt.target, err, tt.wantErr)
			}
		})
	}
}

func TestBindNamesFailingKernel(t *testing.T) {
	// Without a loaded module the first lookup fails; the error must name the
	// kernel and the field so the caller knows which one to look at.
	var k struct {
		Embed driver.Function `cuda:"EmbeddingLookup"`
		Norm  driver.Function `cuda:"RMSNorm"`
		Skip  driver.Function `cuda:"-"`
	}
	err := cudalaunch.Bind(0, &k)
	if err == nil {
		t.Skip("module 0 resolved a function; a CUDA driver appears to be present")
	}
	if !strings.Contains(err.Error(), `"EmbeddingLookup"`) || !strings.Contains(err.Error(), "Embed") {
		t.Errorf("Bind = %v, want error naming kernel EmbeddingLookup and field Embed", err)
	}
	if strings.Contains(err.Error(), "RMSNorm") {
		t.Errorf("Bind = %v, want only the first failing kernel reported", err)
	}
}
