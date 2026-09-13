package main

import "testing"

import "github.com/tmc/lib/nvidia/x/cudasass"

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
	tests := []struct {
		name                    string
		base, cand              *DiffInstruction
		wantDelta               float64
		wantReason, wantSummary string
	}{
		{"both", &DiffInstruction{StallPct: 10, StallBreakdown: map[string]float64{"stall_mem": 10}}, &DiffInstruction{StallPct: 25, StallBreakdown: map[string]float64{"stall_mem": 25}}, 15, "mem", "mem +15.0%"},
		{"tie", &DiffInstruction{StallPct: 10, StallBreakdown: map[string]float64{"stall_tex": 5, "stall_mem": 5}}, &DiffInstruction{StallPct: 20, StallBreakdown: map[string]float64{"stall_tex": 10, "stall_mem": 10}}, 10, "mem", "mem +10.0%"},
		{"base_nil", nil, &DiffInstruction{StallPct: 5, StallBreakdown: map[string]float64{"stall_math": 5}}, 5, "math", "math +5.0%"},
		{"cand_nil", &DiffInstruction{StallPct: 5, StallBreakdown: map[string]float64{"stall_math": 5}}, nil, -5, "math", "math -5.0%"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			delta, reason, summary := computeStallDelta(tt.base, tt.cand)
			if delta != tt.wantDelta || reason != tt.wantReason || summary != tt.wantSummary {
				t.Fatalf("got %v, %q, %q; want %v, %q, %q", delta, reason, summary, tt.wantDelta, tt.wantReason, tt.wantSummary)
			}
		})
	}
	if delta, reason, summary := computeStallDelta(nil, nil); delta != 0 || reason != "" || summary != "" {
		t.Fatalf("nil inputs = %v, %q, %q", delta, reason, summary)
	}
}

func TestAlignInstructions(t *testing.T) {
	fd := cudasass.FunctionDiff{Name: "kernel", Changes: []cudasass.InstructionDiff{
		{Op: cudasass.DiffMatch, Base: &cudasass.Instruction{Opcode: "LDG.E"}, Target: &cudasass.Instruction{Opcode: "LDG.E"}},
		{Op: cudasass.DiffDelete, Base: &cudasass.Instruction{Opcode: "STS"}},
		{Op: cudasass.DiffAdd, Target: &cudasass.Instruction{Opcode: "LDS"}},
	}}
	rows := alignInstructions(fd, nil, nil, nil, nil)
	if len(rows) != 2 || rows[0].Type != DiffRowMatch || rows[1].Type != DiffRowModify || !rows[1].HasSpill {
		t.Fatalf("alignInstructions = %#v", rows)
	}
}
