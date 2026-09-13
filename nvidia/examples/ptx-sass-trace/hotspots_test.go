package main

import (
	"bytes"
	"context"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof"
)

func TestInstructionCost(t *testing.T) {
	tests := []struct {
		opcode    string
		minWeight int64
		wantStall string
	}{
		{"LDG.E.CONSTANT", 50, "stall_scoreboard"},
		{"STG.E", 50, "stall_scoreboard"},
		{"LDS", 15, "stall_scoreboard"},
		{"STS", 15, "stall_scoreboard"},
		{"SHFL.BFLY", 20, "stall_wait"},
		{"BAR.SYNC", 20, "stall_wait"},
		{"IDP4A.S32.S32", 15, "stall_math"},
		{"HMMA.16816", 15, "stall_math"},
		{"MUFU.RSQ", 10, "stall_math"},
		{"FFMA", 4, "stall_math"},
		{"IMAD.MOV.U32", 4, "stall_math"},
		{"BRA", 2, "stall_branch"},
		{"NOP", 1, "stall_math"},
	}

	for _, tt := range tests {
		t.Run(tt.opcode, func(t *testing.T) {
			weight, stall := instructionCost(tt.opcode)
			if weight < tt.minWeight {
				t.Errorf("opcode %q: weight %d < min %d", tt.opcode, weight, tt.minWeight)
			}
			if stall != tt.wantStall {
				t.Errorf("opcode %q: stall %q, want %q", tt.opcode, stall, tt.wantStall)
			}
		})
	}
}

func TestRecordInstructionHotspots(t *testing.T) {
	var buf bytes.Buffer
	prof := cudaprof.NewSession(&buf)
	if err := prof.SetSampleTypes(sampleTypesWithHotspots(false, true)...); err != nil {
		t.Fatalf("SetSampleTypes: %v", err)
	}

	ptxLines := []uint64{0, 0, 10, 20}
	rec := newLaunchRecorder(prof, ptxLines)

	k := &cudacapture.Kernel{
		Name:       "test_kernel",
		PTXLine:    2,
		SourceLine: 40,
		Instructions: []cudacapture.Instruction{
			{Offset: 0x00, SASSLine: 10, PTXLine: 2, SourceLine: 41, Text: "/*0000*/ LDG.E R2, [R0];"},
			{Offset: 0x10, SASSLine: 11, PTXLine: 2, SourceLine: 42, Text: "/*0010*/ FFMA R3, R2, R1, R3;"},
			{Offset: 0x20, SASSLine: 12, PTXLine: 3, SourceLine: 43, Text: "/*0020*/ STG.E [R0], R3;"},
		},
	}

	attributed := recordInstructionHotspots(context.Background(), rec, k, 5, 1000000, nil)
	if attributed != 3 {
		t.Errorf("attributed %d instructions, want 3", attributed)
	}

	if err := prof.Close(); err != nil {
		t.Fatalf("Close profile: %v", err)
	}

	p, err := profile.ParseData(buf.Bytes())
	if err != nil {
		t.Fatalf("ParseData: %v", err)
	}

	if len(p.Sample) != 3 {
		t.Fatalf("expected 3 samples, got %d", len(p.Sample))
	}

	// Verify that each sample contains SASS, PTX, TileIR, and Source frames
	for i, s := range p.Sample {
		if len(s.Location) != 4 {
			t.Errorf("sample %d: expected 4 locations (SASS, PTX, TileIR, Source), got %d", i, len(s.Location))
		}
	}
}
