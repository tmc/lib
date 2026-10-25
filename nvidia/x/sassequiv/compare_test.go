package sassequiv

import (
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

// sass returns a cuobjdump listing of one function, f, whose
// instructions are separated by newlines and numbered from 0.
func sass(body string) string {
	var b strings.Builder
	b.WriteString("\tcode for sm_90a\n\t\tFunction : f\n")
	pc := 0
	for _, line := range strings.Split(strings.TrimSpace(body), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, ".L_") {
			fmt.Fprintf(&b, "%s:\n", line)
			continue
		}
		fmt.Fprintf(&b, "        /*%04x*/                   %s ;\n", pc, line)
		pc += 0x10
	}
	return b.String()
}

const prologue = `
LDC R1, c[0x0][0x28]
S2R R0, SR_TID.X
LDC.64 R2, c[0x0][0x210]
LDC.64 R4, c[0x0][0x218]
IMAD.WIDE.U32 R2, R0, 0x4, R2
IMAD.WIDE.U32 R4, R0, 0x4, R4
`

// loop sums a[i*32] for i < c[0x0][0x220] into R8, then stores it.
const loop = prologue + `
LDC R6, c[0x0][0x220]
MOV R8, RZ
ISETP.GE.AND P0, PT, RZ, R6, PT
@P0 BRA 0x110
LDG.E R7, desc[UR4][R2.64]
IADD3 R6, R6, -0x1, RZ
IADD3 R2, P1, R2, 0x80, RZ
IADD3.X R3, RZ, R3, RZ, P1, !PT
FADD R8, R8, R7
ISETP.NE.AND P0, PT, R6, RZ, PT
@P0 BRA 0xa0
STG.E desc[UR4][R4.64], R8
EXIT
BRA 0x100
`

func TestCompare(t *testing.T) {
	tests := []struct {
		name          string
		before, after string
		want          Verdict
		reason        string // in the result, if set
	}{
		{"identical", prologue + `
LDG.E R6, desc[UR4][R2.64]
FADD R6, R6, 1
STG.E desc[UR4][R4.64], R6
EXIT`, prologue + `
LDG.E R6, desc[UR4][R2.64]
FADD R6, R6, 1
STG.E desc[UR4][R4.64], R6
EXIT`, Equal, ""},
		{"renamed and rescheduled", prologue + `
LDG.E R6, desc[UR4][R2.64]
LDG.E R7, desc[UR4][R2.64+0x4]
FADD R6, R6, R7
FMUL R7, R7, 2
STG.E desc[UR4][R4.64], R6
STG.E desc[UR4][R4.64+0x4], R7
EXIT`, prologue + `
LDG.E R9, desc[UR4][R2.64+0x4]
LDG.E R10, desc[UR4][R2.64]
FMUL R11, R9, 2
FADD R10, R9, R10
STG.E desc[UR4][R4.64], R10
STG.E desc[UR4][R4.64+0x4], R11
EXIT`, Equal, ""},
		{"stores reordered", prologue + `
STG.E desc[UR4][R4.64], RZ
STG.E desc[UR4][R4.64+0x4], RZ
EXIT`, prologue + `
STG.E desc[UR4][R4.64+0x4], RZ
STG.E desc[UR4][R4.64], RZ
EXIT`, Unknown, ""},
		{"spilled", prologue + `
LDG.E R6, desc[UR4][R2.64]
FMUL R6, R6, R6
STG.E desc[UR4][R4.64], R6
EXIT`, prologue + `
LDG.E R6, desc[UR4][R2.64]
STL [R1+0x8], R6
MOV R6, RZ
LDL R7, [R1+0x8]
FMUL R6, R7, R7
STG.E desc[UR4][R4.64], R6
EXIT`, Equal, ""},
		{"zero register and zero immediate", prologue + `
MOV R6, RZ
STG.E desc[UR4][R4.64], R6
EXIT`, prologue + `
MOV R6, 0x0
STG.E desc[UR4][R4.64], R6
EXIT`, Equal, ""},
		{"HFMA2.MMA writes a as the high half", prologue + `
MOV R6, 0x39000000
STG.E desc[UR4][R4.64], R6
EXIT`, prologue + `
HFMA2.MMA R6, -RZ, RZ, 0.625, 0
STG.E desc[UR4][R4.64], R6
EXIT`, Equal, ""},
		{"HFMA2.MMA halves exchanged", prologue + `
MOV R6, 0x39000000
STG.E desc[UR4][R4.64], R6
EXIT`, prologue + `
HFMA2.MMA R6, -RZ, RZ, 0, 0.625
STG.E desc[UR4][R4.64], R6
EXIT`, Different, ""},
		{"fadd operands swapped", prologue + `
LDG.E R6, desc[UR4][R2.64]
LDG.E R7, desc[UR4][R2.64+0x4]
FADD R6, R6, R7
STG.E desc[UR4][R4.64], R6
EXIT`, prologue + `
LDG.E R6, desc[UR4][R2.64]
LDG.E R7, desc[UR4][R2.64+0x4]
FADD R6, R7, R6
STG.E desc[UR4][R4.64], R6
EXIT`, Equal, ""},
		{"ffma operands exchanged", prologue + `
LDG.E R6, desc[UR4][R2.64]
LDG.E R7, desc[UR4][R2.64+0x4]
FFMA R6, R6, R7, R7
STG.E desc[UR4][R4.64], R6
EXIT`, prologue + `
LDG.E R6, desc[UR4][R2.64]
LDG.E R7, desc[UR4][R2.64+0x4]
FFMA R6, R7, R7, R6
STG.E desc[UR4][R4.64], R6
EXIT`, Different, ""},
		{"fp op changed", prologue + `
LDG.E R6, desc[UR4][R2.64]
FADD R6, R6, R6
STG.E desc[UR4][R4.64], R6
EXIT`, prologue + `
LDG.E R6, desc[UR4][R2.64]
FMUL R6, R6, R6
STG.E desc[UR4][R4.64], R6
EXIT`, Different, ""},
		{"load addresses swapped", prologue + `
LDG.E R6, desc[UR4][R2.64]
LDG.E R7, desc[UR4][R2.64+0x4]
FFMA R6, R6, 2, R7
STG.E desc[UR4][R4.64], R6
EXIT`, prologue + `
LDG.E R6, desc[UR4][R2.64+0x4]
LDG.E R7, desc[UR4][R2.64]
FFMA R6, R6, 2, R7
STG.E desc[UR4][R4.64], R6
EXIT`, Different, ""},
		{"load offset changed", prologue + `
LDG.E R6, desc[UR4][R2.64+0x4]
STG.E desc[UR4][R4.64], R6
EXIT`, prologue + `
LDG.E R6, desc[UR4][R2.64+0x8]
STG.E desc[UR4][R4.64], R6
EXIT`, Different, ""},
		{"index computed with LEA instead of IMAD.WIDE", `
LDC R1, c[0x0][0x28]
S2R R0, SR_TID.X
LDC.64 R2, c[0x0][0x210]
IMAD.WIDE.U32 R2, R0, 0x4, R2
LDG.E R6, desc[UR4][R2.64]
STG.E desc[UR4][R2.64], R6
EXIT`, `
LDC R1, c[0x0][0x28]
S2R R0, SR_TID.X
LDC.64 R4, c[0x0][0x210]
LEA R2, P0, R0, R4, 0x2
LEA.HI.X R3, R0, R5, RZ, 0x2, P0
LDG.E R6, desc[UR4][R2.64]
STG.E desc[UR4][R2.64], R6
EXIT`, Equal, ""},
		{"loop renamed", loop, strings.NewReplacer("R8", "R9", "R7", "R10").Replace(loop), Equal, ""},
		{"loop recurrence changed", loop, strings.Replace(loop, "FADD R8, R8, R7", "FMUL R8, R8, R7", 1), Different, ""},
		{"loop stride changed", loop, strings.Replace(loop, "R2, 0x80, RZ", "R2, 0x100, RZ", 1), Different, ""},
		{"loop unrolled", loop, prologue + `
LDC R6, c[0x0][0x220]
MOV R8, RZ
ISETP.GE.AND P0, PT, RZ, R6, PT
@P0 BRA 0x150
LDG.E R7, desc[UR4][R2.64]
ISETP.GT.AND P1, PT, R6, 0x1, PT
@P1 LDG.E R9, desc[UR4][R2.64+0x80]
IADD3 R6, R6, -0x2, RZ
IADD3 R2, P2, R2, 0x100, RZ
IADD3.X R3, RZ, R3, RZ, P2, !PT
FADD R8, R8, R7
@P1 FADD R8, R8, R9
ISETP.GT.AND P0, PT, R6, RZ, PT
@P0 BRA 0xa0
STG.E desc[UR4][R4.64], R8
EXIT
BRA 0x160`, Unknown, ""},
		{"call with registers renamed", prologue + `
LDG.E R6, desc[UR4][R2.64]
CALL.REL.NOINC 0xa0
STG.E desc[UR4][R4.64], R6
EXIT
FMUL R6, R6, R6
RET.REL.NODEC R10 0x0`, prologue + `
LDG.E R7, desc[UR4][R2.64]
MOV R6, R7
CALL.REL.NOINC 0xb0
STG.E desc[UR4][R4.64], R6
EXIT
FMUL R6, R6, R6
RET.REL.NODEC R10 0x0`, Equal, ""},
		{"call same layout", prologue + `
LDG.E R6, desc[UR4][R2.64]
CALL.REL.NOINC 0xa0
STG.E desc[UR4][R4.64], R6
EXIT
FMUL R6, R6, R6
RET.REL.NODEC R10 0x0`, prologue + `
LDG.E R6, desc[UR4][R2.64]
CALL.REL.NOINC 0xa0
STG.E desc[UR4][R4.64], R6
EXIT
FMUL R6, R6, R6
RET.REL.NODEC R10 0x0`, Equal, ""},
		{"subroutine changed", prologue + `
LDG.E R6, desc[UR4][R2.64]
CALL.REL.NOINC 0xa0
STG.E desc[UR4][R4.64], R6
EXIT
FMUL R6, R6, R6
RET.REL.NODEC R10 0x0`, prologue + `
LDG.E R6, desc[UR4][R2.64]
CALL.REL.NOINC 0xa0
STG.E desc[UR4][R4.64], R6
EXIT
FADD R6, R6, R6
RET.REL.NODEC R10 0x0`, Different, ""},
		{"predicated write into registers holding different values", prologue + `
LDG.E R6, desc[UR4][R2.64]
LDG.E R7, desc[UR4][R2.64+0x4]
ISETP.GT.AND P0, PT, R0, 0x3, PT
FADD R8, R7, R6
STG.E desc[UR4][R4.64+0x4], R8
@P0 FMUL R8, R6, 2
@P0 STG.E desc[UR4][R4.64], R8
EXIT`, prologue + `
LDG.E R6, desc[UR4][R2.64]
LDG.E R7, desc[UR4][R2.64+0x4]
ISETP.GT.AND P0, PT, R0, 0x3, PT
FADD R8, R7, R6
STG.E desc[UR4][R4.64+0x4], R8
@P0 FMUL R7, R6, 2
@P0 STG.E desc[UR4][R4.64], R7
EXIT`, Equal, ""},
		{"comparison mirrored", prologue + `
LDC R6, c[0x0][0x220]
ISETP.LT.AND P0, PT, R0, R6, PT
@P0 EXIT
STG.E desc[UR4][R4.64], RZ
EXIT`, prologue + `
LDC R6, c[0x0][0x220]
ISETP.GT.AND P0, PT, R6, R0, PT
@P0 EXIT
STG.E desc[UR4][R4.64], RZ
EXIT`, Equal, ""},
		{"special register read again under a predicate", prologue + `
ISETP.GT.AND P0, PT, R0, 0x3, PT
I2FP.F32.U32 R6, R0
STG.E desc[UR4][R4.64], R6
EXIT`, prologue + `
ISETP.GT.AND P0, PT, R0, 0x3, PT
@P0 S2R R0, SR_TID.X
I2FP.F32.U32 R6, R0
STG.E desc[UR4][R4.64], R6
EXIT`, Equal, ""},
		{"operation moved across a merge", prologue + `
LDG.E R6, desc[UR4][R2.64]
ISETP.GT.AND P0, PT, R0, 0x3, PT
@P0 BRA 0xa0
FMUL R6, R6, 0.5
FADD R6, R6, 1
STG.E desc[UR4][R4.64], R6
EXIT`, prologue + `
LDG.E R6, desc[UR4][R2.64]
ISETP.GT.AND P0, PT, R0, 0x3, PT
FADD R7, R6, 1
@P0 BRA 0xb0
FFMA R7, R6, 0.5, 1
STG.E desc[UR4][R4.64], R7
EXIT`, Unknown, "values merge at different blocks"},
		{"loop pointer offset folded into its initial value",
			strings.NewReplacer(
				"LDC R6, c[0x0][0x220]", "LDC R6, c[0x0][0x220]\nIADD3 R2, P1, R2, 0x0, RZ\nIADD3.X R3, RZ, R3, RZ, P1, !PT",
				"LDG.E R7, desc[UR4][R2.64]", "LDG.E R7, desc[UR4][R2.64+0x4]",
				"BRA 0x110", "BRA 0x130",
				"BRA 0xa0", "BRA 0xc0",
				"BRA 0x100", "BRA 0x120",
			).Replace(loop),
			strings.NewReplacer(
				"LDC R6, c[0x0][0x220]", "LDC R6, c[0x0][0x220]\nIADD3 R2, P1, R2, 0x4, RZ\nIADD3.X R3, RZ, R3, RZ, P1, !PT",
				"BRA 0x110", "BRA 0x130",
				"BRA 0xa0", "BRA 0xc0",
				"BRA 0x100", "BRA 0x120",
			).Replace(loop), Unknown, ""},
		{"loop invariant read again in the body",
			strings.NewReplacer(
				"FADD R8, R8, R7", "NOP\nI2FP.F32.U32 R9, R0\nFADD R8, R8, R9",
				"BRA 0x110", "BRA 0x130", "BRA 0x100", "BRA 0x120",
			).Replace(loop),
			strings.NewReplacer(
				"FADD R8, R8, R7", "S2R R9, SR_TID.X\nI2FP.F32.U32 R9, R9\nFADD R8, R8, R9",
				"BRA 0x110", "BRA 0x130", "BRA 0x100", "BRA 0x120",
			).Replace(loop), Equal, ""},
		{"unmodeled opcode", prologue + `
BPT.TRAP 0x1
EXIT`, prologue + `
BPT.TRAP 0x1
EXIT`, Unknown, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			before, after := parse(t, sass(tt.before)), parse(t, sass(tt.after))
			r := Compare(before.Functions[0], after.Functions[0])
			if r.Verdict != tt.want || !strings.Contains(r.Reason, tt.reason) {
				t.Errorf("got %v, want %v %q", r, tt.want, tt.reason)
			}
		})
	}
}

func TestSubroutineEdges(t *testing.T) {
	d := parse(t, sass(prologue+`
CALL.REL.NOINC 0x90
CALL.REL.NOINC 0x90
EXIT
FMUL R6, R6, R6
RET.REL.NODEC R10 0x0`))
	cfg, err := cudasass.NewCFG(d.Functions[0])
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, e := range subroutineEdges(cfg) {
		from := cfg.Blocks[e.From].Instructions[0].Address
		to := cfg.Blocks[e.To].Instructions[0].Address
		got = append(got, fmt.Sprintf("%#x->%#x", from, to))
	}
	// Each call enters the subroutine, and its return reaches the
	// instruction after either call; no call falls through.
	want := []string{"0x0->0x90", "0x70->0x90", "0x90->0x70", "0x90->0x80"}
	if !slices.Equal(got, want) {
		t.Errorf("edges = %v, want %v", got, want)
	}
}
