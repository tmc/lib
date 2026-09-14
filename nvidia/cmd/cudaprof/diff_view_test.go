package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

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

func TestRenderDiffHTML(t *testing.T) {
	data := &DiffViewData{
		Title:         "baseline vs candidate",
		BaseCapture:   "base.pprof",
		TargetCapture: "candidate.pprof",
		Kernels:       []string{"kernel"},
		ActiveKernel:  "kernel",
		ActiveRows: []DiffRow{{Num: 1, Type: DiffRowModify,
			Base: &DiffInstruction{Opcode: "LDG.E"}, Candidate: &DiffInstruction{Opcode: "LDG.E.128"}}},
	}
	var buf bytes.Buffer
	if err := renderDiffHTML(&buf, data); err != nil {
		t.Fatalf("renderDiffHTML: %v", err)
	}
	for _, want := range []string{"<!DOCTYPE html>", "baseline vs candidate", "base.pprof", "LDG.E.128"} {
		if !strings.Contains(buf.String(), want) {
			t.Errorf("rendered HTML missing %q", want)
		}
	}
}

func TestConvertInstruction(t *testing.T) {
	if got := convertInstruction(nil, "", nil, nil); got != nil {
		t.Fatalf("convertInstruction(nil) = %#v, want nil", got)
	}
	got := convertInstruction(&cudasass.Instruction{
		Address: 0x20, Predicate: "@P0", Opcode: "LDG.E.128", Operands: "R1, [R2]",
	}, "kernel", nil, nil)
	if got.Offset != "0020" || got.Text != "@P0 LDG.E.128 R1, [R2]" || got.SemanticTag != "memory" {
		t.Fatalf("convertInstruction = %#v", got)
	}
	got = convertInstruction(&cudasass.Instruction{Opcode: "STS"}, "", nil, nil)
	if !got.IsSpill || got.SpillType != "STS" {
		t.Fatalf("spill conversion = %#v", got)
	}
}

func TestComputeTopStallShifts(t *testing.T) {
	// Equal deltas: the order must come from the reason name, not from map
	// iteration, or the same pair of profiles diffs differently each run.
	base := map[string]int64{"stall_mem": 10, "stall_tex": 10}
	target := map[string]int64{"stall_mem": 15, "stall_tex": 15}
	for range 20 {
		got := computeTopStallShifts(base, target)
		if len(got) != 2 || got[0].Reason != "stall_mem" || got[0].Delta != 5 {
			t.Fatalf("computeTopStallShifts = %#v", got)
		}
	}
	if got := computeTopStallShifts(nil, nil); got != nil {
		t.Fatalf("empty shifts = %#v, want nil", got)
	}
}

func TestBuildKernelSummaryEmptyCaptures(t *testing.T) {
	sum := buildKernelSummary(cudasass.FunctionDiff{Name: "kernel"}, nil, nil)
	if sum.BaseInstructions != 0 || sum.TargetInstructions != 0 || sum.TopStallShifts != nil {
		t.Fatalf("buildKernelSummary empty = %#v", sum)
	}
	if profileOrNil(nil) != nil {
		t.Fatal("profileOrNil(nil) returned a profile")
	}
}
