package main

import "testing"

func TestIsSpillOpcode(t *testing.T) {
	for _, tt := range []struct {
		in, wantType string
		want         bool
	}{
		{"STS [R1], R2;", "STS", true}, {"@P0 LDS R2, [R1];", "LDS", true}, {"LDG.E R2, [R1];", "", false},
	} {
		got, typ := isSpillOpcode(tt.in)
		if got != tt.want || typ != tt.wantType {
			t.Errorf("isSpillOpcode(%q) = %t, %q; want %t, %q", tt.in, got, typ, tt.want, tt.wantType)
		}
	}
}

func TestShortStallReason(t *testing.T) {
	for _, tt := range []struct{ in, want string }{
		{"stall_scoreboard", "scoreboard"}, {"memory_dependency", "mem"}, {"unknown_long_reason", "unknown_long"},
	} {
		if got := shortStallReason(tt.in); got != tt.want {
			t.Errorf("shortStallReason(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

func TestComputeStallDelta(t *testing.T) {
	base := &DiffInstruction{StallPct: 10, StallBreakdown: map[string]float64{"stall_mem": 10}}
	cand := &DiffInstruction{StallPct: 25, StallBreakdown: map[string]float64{"stall_mem": 25}}
	delta, reason, _ := computeStallDelta(base, cand)
	if delta != 15 || reason != "mem" {
		t.Fatalf("computeStallDelta = %v, %q; want 15, mem", delta, reason)
	}
}
