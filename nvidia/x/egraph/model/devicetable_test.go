package model

import "testing"

// TestTargetMemoryGBps pins the bandwidth the multi-arch latency estimate
// divides by for each target.
func TestTargetMemoryGBps(t *testing.T) {
	tests := []struct {
		target string
		want   float64
	}{
		{"sm_90a", 4800},
		{"sm_100a", 8000},
		{"sm_100f", 8000},
		{"sm_103a", 8000},
		{"sm_107a", 22000},
		{"sm_120a", 1792},
		{"sm_121a", 273},
	}
	for _, tt := range tests {
		spec, err := LookupArchTarget(tt.target)
		if err != nil {
			t.Fatal(err)
		}
		if got := targetMemoryGBps(spec); got != tt.want {
			t.Errorf("targetMemoryGBps(%s) = %v, want %v", tt.target, got, tt.want)
		}
	}
}

// TestClassifyHasTMA pins the one capability flag TestClassifyComputeCapability
// does not check: every supported architecture has TMA.
func TestClassifyHasTMA(t *testing.T) {
	for _, cc := range [][2]int{{9, 0}, {10, 0}, {10, 3}, {10, 7}, {12, 0}, {12, 1}} {
		spec, err := ClassifyComputeCapability(cc[0], cc[1])
		if err != nil {
			t.Fatal(err)
		}
		if !spec.HasTMA {
			t.Errorf("%s: HasTMA = false, want true", spec.Target)
		}
	}
}
