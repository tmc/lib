package main

import (
	"bytes"
	"context"
	"slices"
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

	// Two ten-byte lines, so PTX line 2 is ten bytes into the text and line 3
	// is the sentinel past its end.
	rec := newLaunchRecorder(prof, cudacapture.NewPTXAddresses("// aaaaaa\n// bbbbbb\n"))

	// Offset is nonzero because that is the case the addresses have to survive:
	// cudacapture enforces Instructions[0].Offset == 0, so a kernel-relative
	// address is indistinguishable from a correct one on the first kernel in
	// the CUBIN and wrong on every other.
	k := &cudacapture.Kernel{
		Name:       "test_kernel",
		Offset:     0xd900,
		PTXLine:    2,
		SourceLine: 40,
		Instructions: []cudacapture.Instruction{
			{Offset: 0x00, SASSLine: 10, PTXLine: 2, SourceLine: 41, Text: "/*0000*/ LDG.E R2, [R0];"},
			{Offset: 0x10, SASSLine: 11, PTXLine: 2, SourceLine: 42, Text: "/*0010*/ FFMA R3, R2, R1, R3;"},
			{Offset: 0x20, SASSLine: 12, PTXLine: 3, SourceLine: 43, Text: "/*0020*/ STG.E [R0], R3;"},
		},
	}

	attributed := recordInstructionHotspots(context.Background(), rec, k, 5, nil)
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

	// A modeled run must not write a sample type a measurement writes. pprof
	// carries no lineage beyond these names, so a reader who sees pc_samples
	// has no way to learn nothing was sampled.
	var names []string
	for _, st := range p.SampleType {
		names = append(names, st.Type)
	}
	for _, measured := range []string{
		"pc_samples", "inst_executed", "stall_mem", "stall_wait",
		"stall_scoreboard", "stall_math", "stall_branch",
	} {
		if slices.Contains(names, measured) {
			t.Errorf("modeled profile declares measured sample type %q; types = %v", measured, names)
		}
	}
	costIndex := slices.Index(names, "modeled_cost")
	if costIndex < 0 {
		t.Fatalf("modeled profile has no modeled_cost sample type; types = %v", names)
	}

	// The weights are the opcode table's, so they are the assertion: LDG and
	// STG cost 60, FFMA costs 4. Reading them back proves the model's output
	// reached the profile as a model rather than as a counter.
	want := []int64{60, 4, 60}
	for i, s := range p.Sample {
		if got := s.Value[costIndex]; got != want[i] {
			t.Errorf("sample %d: modeled_cost = %d, want %d", i, got, want[i])
		}
	}

	// Excluding the measured type names is not enough on its own: dividing the
	// launch duration by weight and writing it to gpu_time passes every check
	// above, and gpu_time means a measured CUDA event interval in the other two
	// modes. A modeled row must leave every measured value at zero and let the
	// caller's launch-level row carry the real one.
	for _, measured := range []string{"gpu_time", "samples"} {
		i := slices.Index(names, measured)
		if i < 0 {
			continue
		}
		for j, s := range p.Sample {
			if s.Value[i] != 0 {
				t.Errorf("modeled sample %d has nonzero %s = %d; only the launch row may carry it", j, measured, s.Value[i])
			}
		}
	}

	// pprof addresses are DefaultMappingStart plus a CUBIN file offset, which
	// is the kernel's offset plus the instruction's offset within it. Dropping
	// the kernel term puts every kernel's entry on one address and makes their
	// ranges alias; nothing in a line-oriented view shows it, and
	// -granularity=addresses, peek and weblist are all keyed on it.
	wantAddr := []uint64{
		cudaprof.DefaultMappingStart + 0xd900 + 0x00,
		cudaprof.DefaultMappingStart + 0xd900 + 0x10,
		cudaprof.DefaultMappingStart + 0xd900 + 0x20,
	}
	for i, s := range p.Sample {
		leaf := s.Location[0]
		if leaf.Address != wantAddr[i] {
			t.Errorf("sample %d: SASS address = %#x, want %#x", i, leaf.Address, wantAddr[i])
		}
	}
}
