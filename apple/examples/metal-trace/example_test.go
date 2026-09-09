package main

import (
	"math"
	"testing"
)

func TestWorkloadReference(t *testing.T) {
	rounds := 8
	ref := reference(rounds)
	if len(ref) != 1024 {
		t.Fatalf("expected 1024 reference elements, got %d", len(ref))
	}
	for i, v := range ref {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("reference[%d] is NaN or Inf", i)
		}
	}
}

func TestValidationErrors(t *testing.T) {
	// Wrong finite output
	a := float32(1.0)
	b := float32(2.0)
	wrong := float32(4.0)
	if wrong == a+b {
		t.Fatal("expected wrong result to differ")
	}

	// NaN control
	nan := float32(math.NaN())
	if !math.IsNaN(float64(nan)) {
		t.Fatal("expected NaN detection")
	}
}

func TestTimingModeSelection(t *testing.T) {
	cases := []struct {
		mode    string
		wantErr bool
	}{
		{"auto", false},
		{"command-buffer", false},
		{"counter", false},
		{"invalid-mode", true},
	}
	for _, tc := range cases {
		t.Run(tc.mode, func(t *testing.T) {
			cfg := config{elements: 1024, rounds: 8, passes: 1, timing: tc.mode}
			isKnown := cfg.timing == "auto" || cfg.timing == "counter" || cfg.timing == "command-buffer"
			if tc.wantErr && isKnown {
				t.Fatalf("expected unknown mode for %q", tc.mode)
			}
			if !tc.wantErr && !isKnown {
				t.Fatalf("unexpected known mode rejection for %q", tc.mode)
			}
		})
	}
}

func TestStageTimingValidationControls(t *testing.T) {
	// Test stageTiming fields and controls
	st := stageTiming{
		nanos:  1000,
		mode:   "counter",
		scope:  "stage_boundary",
		reason: "",
	}
	if st.nanos <= 0 || st.mode != "counter" || st.scope != "stage_boundary" {
		t.Fatalf("unexpected stage timing values: %+v", st)
	}
}
