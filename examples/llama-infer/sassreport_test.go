package main

import (
	"bytes"
	"strings"
	"testing"
)

// fixtureSASS is in the text form nvdisasm emits. It is a fixture for the
// analysis in this file; the disassembler's own parsing is covered by
// x/cudasass against retained nvdisasm output.
const fixtureSASS = `
		.headerflags	@"EF_CUDA_TEXMODE_UNIFIED"
		.text.LinearGEMV:
        /*0000*/                   LDC R1, c[0x0][0x37c] ;
        /*0010*/                   S2R R5, SR_TID.X ;
        /*0020*/                   LDCU.64 UR4, c[0x0][0x358] ;
        /*0030*/                   IMAD.WIDE.U32 R2, R5, 0x4, R2 ;
        /*0040*/                   LDG.E.CONSTANT.F32 R17, desc[UR4][R2.64] ;
        /*0050*/                   FFMA R0, R17, R17, RZ ;
        /*0060*/                   STG.E desc[UR4][R2.64], R0 ;
        /*0070*/                   EXIT ;
		.text.LinearGEMVF16:
        /*0000*/                   LDC R1, c[0x0][0x37c] ;
        /*0010*/                   LDG.E.U16 R4, [R2.64] ;
        /*0020*/                   F2F.F32.F16 R4, R4 ;
        /*0030*/                   FFMA R4, R4, R4, RZ ;
        /*0040*/                   EXIT ;
`

func TestSASSKernelStats(t *testing.T) {
	stats, err := sassKernelStats(fixtureSASS)
	if err != nil {
		t.Fatal(err)
	}
	if len(stats) != 2 {
		t.Fatalf("got %d functions, want 2: %+v", len(stats), stats)
	}
	byName := map[string]sassKernelStat{}
	for _, s := range stats {
		byName[s.Name] = s
	}
	gemv, ok := byName["LinearGEMV"]
	if !ok {
		t.Fatalf("LinearGEMV missing: %+v", stats)
	}
	if gemv.Instructions != 8 {
		t.Errorf("LinearGEMV instructions = %d, want 8", gemv.Instructions)
	}
	// R17 is the highest general-purpose register mentioned, so at least 18
	// were allocated. UR4 is a uniform register and RZ is not an allocation.
	if gemv.MinRegisters != 18 {
		t.Errorf("LinearGEMV MinRegisters = %d, want 18", gemv.MinRegisters)
	}
	// Modifiers are stripped, so the two loads group under LDG and LDC.
	if gemv.Opcodes["LDG"] != 1 || gemv.Opcodes["LDC"] != 1 {
		t.Errorf("LinearGEMV opcode histogram = %v, want one LDG and one LDC", gemv.Opcodes)
	}
	if f16 := byName["LinearGEMVF16"]; f16.MinRegisters != 5 {
		t.Errorf("LinearGEMVF16 MinRegisters = %d, want 5", f16.MinRegisters)
	}
}

func TestMaxRegisterIndex(t *testing.T) {
	tests := []struct {
		name     string
		operands string
		want     int
	}{
		{"empty", "", -1},
		{"single", "R4, R5, R0", 5},
		{"zero register is not an allocation", "R0, RZ, RZ", 0},
		{"uniform registers are a separate file", "UR4, UR16", -1},
		{"special registers do not count", "R5, SR_TID.X", 5},
		{"inside a memory operand", "desc[UR4][R12.64]", 12},
		{"predicated negation", "!R7, R3", 7},
		{"no registers at all", "c[0x0][0x37c], 0x4", -1},
		{"two digits", "R7, R128", 128},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := maxRegisterIndex(tt.operands); got != tt.want {
				t.Errorf("maxRegisterIndex(%q) = %d, want %d", tt.operands, got, tt.want)
			}
		})
	}
}

func TestBaseOpcode(t *testing.T) {
	for _, tt := range []struct{ in, want string }{
		{"LDG.E.CONSTANT.F32", "LDG"},
		{"FFMA", "FFMA"},
		{"IMAD.WIDE.U32", "IMAD"},
	} {
		if got := baseOpcode(tt.in); got != tt.want {
			t.Errorf("baseOpcode(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

func TestWriteSASSReport(t *testing.T) {
	var buf bytes.Buffer
	if err := writeSASSReport(&buf, fixtureSASS); err != nil {
		t.Fatal(err)
	}
	out := buf.String()
	for _, want := range []string{"REGISTERS>=", "LinearGEMV", "LinearGEMVF16"} {
		if !strings.Contains(out, want) {
			t.Errorf("report is missing %q:\n%s", want, out)
		}
	}
}
