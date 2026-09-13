package main

import (
	"strings"
	"testing"
)

// TestInteriorCalibratorStride pins the marker inventory the recipe
// produces, because the auditor on the other end compares the markers a
// capture carries against the recipe that was executed. A stride that
// silently marked the wrong ordinals would pass every clock check and
// still describe the wrong points in the run.
func TestInteriorCalibratorStride(t *testing.T) {
	tests := []struct {
		name    string
		batches int
		every   int
		want    []string
	}{
		{
			// The production shape: 2048 tokens at
			// cudaPrefillBatchWidth 8 is 256 batches.
			name: "production", batches: 256, every: 64,
			want: []string{"batch-64", "batch-128", "batch-192", "batch-256"},
		},
		{"no markers when disabled", 256, 0, nil},
		{"stride longer than the run", 10, 64, nil},
		{"every batch", 3, 1, []string{"batch-1", "batch-2", "batch-3"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &interiorCalibrator{runID: "test-run", every: tt.every}
			prev := interiorCalib
			interiorCalib = c
			defer func() { interiorCalib = prev }()
			for i := 0; i < tt.batches; i++ {
				beginTraced(levelToken, "batch", nil).end()
			}
			got := c.calibrations()
			if len(got) != len(tt.want) {
				t.Fatalf("took %d markers, want %d: %v", len(got), len(tt.want), labelsOf(got))
			}
			for i, w := range tt.want {
				if got[i].Label != w {
					t.Errorf("marker %d label = %q, want %q", i, got[i].Label, w)
				}
			}
		})
	}
}

func labelsOf(cs []calibration) []string {
	out := make([]string, len(cs))
	for i, c := range cs {
		out[i] = c.Label
	}
	return out
}

// TestInteriorCalibratorOnlyCountsBatches guards the hook itself: the
// stride must see batch spans and nothing else, or the ordinals in the
// labels name something the capture does not contain.
func TestInteriorCalibratorOnlyCountsBatches(t *testing.T) {
	c := &interiorCalibrator{runID: "test-run", every: 2}
	prev := interiorCalib
	interiorCalib = c
	defer func() { interiorCalib = prev }()
	beginTraced(levelToken, "token", nil).end()
	beginTraced(levelPhase, "prefill", nil).end()
	beginTraced(levelToken, "batch", nil).end()
	beginTraced(levelToken, "batch", nil).end()
	got := c.calibrations()
	if len(got) != 1 || got[0].Label != "batch-2" {
		t.Fatalf("markers = %v, want [batch-2]", labelsOf(got))
	}
}

// TestInteriorMarkersAreUniqueAndValid holds the two properties the
// offline auditor requires of the receipt: every label distinct, including
// against the boundary brackets, and every bracket ordered.
func TestInteriorMarkersAreUniqueAndValid(t *testing.T) {
	c := &interiorCalibrator{runID: "test-run", every: 64}
	prev := interiorCalib
	interiorCalib = c
	defer func() { interiorCalib = prev }()
	all := []calibration{calibrate("test-run", "start")}
	for i := 0; i < 256; i++ {
		beginTraced(levelToken, "batch", nil).end()
	}
	all = append(all, c.calibrations()...)
	all = append(all, calibrate("test-run", "end"))

	if len(all) != 6 {
		t.Fatalf("receipt holds %d brackets, want 6: %v", len(all), labelsOf(all))
	}
	seenLabel := map[string]bool{}
	seenName := map[string]bool{}
	for _, cal := range all {
		if seenLabel[cal.Label] {
			t.Errorf("duplicate label %q", cal.Label)
		}
		seenLabel[cal.Label] = true
		if seenName[cal.Name] {
			t.Errorf("duplicate NVTX name %q", cal.Name)
		}
		seenName[cal.Name] = true
		if !cal.valid() {
			t.Errorf("bracket %q runs backwards: %d..%d", cal.Label, cal.Before, cal.After)
		}
		if !strings.Contains(cal.Name, "test-run") {
			t.Errorf("name %q does not carry the run ID", cal.Name)
		}
	}
	// The receipt must be writable as it stands; writeCalibrations
	// refuses a bracket it cannot defend.
	if err := writeCalibrations(&strings.Builder{}, all); err != nil {
		t.Errorf("writeCalibrations: %v", err)
	}

	// The markers must be interior, not clustered at the boundary the
	// start and end brackets already cover.
	if all[1].Label != "batch-64" || all[len(all)-2].Label != "batch-256" {
		t.Errorf("marker order = %v, want the interior ordinals between start and end", labelsOf(all))
	}
}
