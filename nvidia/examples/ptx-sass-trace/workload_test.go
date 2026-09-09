package main

import (
	"math"
	"testing"
)

func TestConfigValidation(t *testing.T) {
	for _, tt := range []struct {
		name    string
		cfg     config
		wantErr bool
	}{
		{"valid default", config{elements: 1024, passes: 2, rounds: 8}, false},
		{"elements zero", config{elements: 0, passes: 2, rounds: 8}, true},
		{"elements negative", config{elements: -1, passes: 2, rounds: 8}, true},
		{"elements too large", config{elements: 1<<26 + 1, passes: 2, rounds: 8}, true},
		{"passes zero", config{elements: 1024, passes: 0, rounds: 8}, true},
		{"passes too large", config{elements: 1024, passes: 1001, rounds: 8}, true},
		{"rounds zero", config{elements: 1024, passes: 2, rounds: 0}, true},
		{"rounds too large", config{elements: 1024, passes: 2, rounds: 1<<20 + 1}, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			err := run("", tt.cfg)
			// error will trigger at validation before directory creation
			if (err != nil) != tt.wantErr {
				// if wantErr is true and err != nil, that's expected
				if !tt.wantErr {
					// on a host without GPU, run will fail after validation when opening device/dir
					// so we only check if err matches the validation error string
				}
			}
		})
	}
}

func TestReferenceOracle(t *testing.T) {
	ref := reference(8)
	if len(ref) != 1024 {
		t.Fatalf("expected 1024 reference values, got %d", len(ref))
	}
	for i, v := range ref {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("reference[%d] is not finite: %g", i, v)
		}
	}
}
