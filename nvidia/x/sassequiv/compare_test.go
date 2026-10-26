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
EXIT`, Equal, ""},
		{"shared stores reordered", prologue + `
STS [R0+0x400], R2
STS [R0+0x404], R3
EXIT`, prologue + `
STS [R0+0x404], R3
STS [R0+0x400], R2
EXIT`, Equal, ""},
		{"stores reordered with their values exchanged", prologue + `
LDG.E R8, desc[UR4][R2.64]
LDG.E R9, desc[UR4][R2.64+0x4]
STG.E desc[UR4][R4.64], R8
STG.E desc[UR4][R4.64+0x4], R9
EXIT`, prologue + `
LDG.E R8, desc[UR4][R2.64]
LDG.E R9, desc[UR4][R2.64+0x4]
STG.E desc[UR4][R4.64+0x4], R8
STG.E desc[UR4][R4.64], R9
EXIT`, Different, ""},
		{"stores a byte short of disjoint reordered", prologue + `
STG.E desc[UR4][R4.64], RZ
STG.E desc[UR4][R4.64+0x3], RZ
EXIT`, prologue + `
STG.E desc[UR4][R4.64+0x3], RZ
STG.E desc[UR4][R4.64], RZ
EXIT`, Unknown, ""},
		{"guarded store reordered", prologue + `
ISETP.GE.AND P0, PT, R0, 0x10, PT
@P0 STG.E desc[UR4][R4.64+0x4], RZ
STG.E desc[UR4][R4.64], RZ
EXIT`, prologue + `
ISETP.GE.AND P0, PT, R0, 0x10, PT
STG.E desc[UR4][R4.64], RZ
@P0 STG.E desc[UR4][R4.64+0x4], RZ
EXIT`, Equal, ""},
		{"guarded store reordered without its guard", prologue + `
ISETP.GE.AND P0, PT, R0, 0x10, PT
@P0 STG.E desc[UR4][R4.64+0x4], RZ
STG.E desc[UR4][R4.64], RZ
EXIT`, prologue + `
ISETP.GE.AND P0, PT, R0, 0x10, PT
STG.E desc[UR4][R4.64], RZ
STG.E desc[UR4][R4.64+0x4], RZ
EXIT`, Unknown, ""},
		{"shared stores off two bases reordered", prologue + `
STS [R0+R3+0x4], RZ
STS [R0], RZ
EXIT`, prologue + `
STS [R0], RZ
STS [R0+R3+0x4], RZ
EXIT`, Unknown, ""},
		{"stores under two descriptors reordered", prologue + `
STG.E desc[UR4][R4.64+0x4], RZ
STG.E desc[UR6][R4.64], RZ
EXIT`, prologue + `
STG.E desc[UR6][R4.64], RZ
STG.E desc[UR4][R4.64+0x4], RZ
EXIT`, Unknown, ""},
		{"overlapping stores reordered", prologue + `
STG.E.64 desc[UR4][R4.64], R2
STG.E.64 desc[UR4][R4.64+0x4], R6
EXIT`, prologue + `
STG.E.64 desc[UR4][R4.64+0x4], R6
STG.E.64 desc[UR4][R4.64], R2
EXIT`, Unknown, ""},
		{"strong stores reordered", prologue + `
STG.E.STRONG.GPU desc[UR4][R4.64], RZ
STG.E.STRONG.GPU desc[UR4][R4.64+0x4], RZ
EXIT`, prologue + `
STG.E.STRONG.GPU desc[UR4][R4.64+0x4], RZ
STG.E.STRONG.GPU desc[UR4][R4.64], RZ
EXIT`, Unknown, ""},
		{"stores off two bases reordered", prologue + `
STG.E desc[UR4][R4.64], RZ
STG.E desc[UR4][R2.64+0x4], RZ
EXIT`, prologue + `
STG.E desc[UR4][R2.64+0x4], RZ
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
		{"register never written", prologue + `
LDG.E R6, desc[UR4][R2.64]
ISETP.GT.AND P0, PT, R0, 0x3, PT
@P0 MOV R9, R6
STG.E desc[UR4][R4.64], R9
EXIT`, prologue + `
LDG.E R6, desc[UR4][R2.64]
ISETP.GT.AND P0, PT, R0, 0x3, PT
@P0 MOV R10, R6
STG.E desc[UR4][R4.64], R10
EXIT`, Unknown, "no path writes"},
		{"register never written, other value", prologue + `
LDG.E R6, desc[UR4][R2.64]
ISETP.GT.AND P0, PT, R0, 0x3, PT
@P0 MOV R9, R6
STG.E desc[UR4][R4.64], R9
EXIT`, prologue + `
LDG.E R6, desc[UR4][R2.64+0x4]
ISETP.GT.AND P0, PT, R0, 0x3, PT
@P0 MOV R10, R6
STG.E desc[UR4][R4.64], R10
EXIT`, Different, ""},
		{"stale value on the edge that skips its use", prologue + `
LDG.E R6, desc[UR4][R2.64]
ISETP.GT.AND P0, PT, R0, 0x3, PT
MOV R9, R0
@!P0 BRA 0xb0
FMUL R9, R6, 2
@P0 FADD R6, R6, R9
STG.E desc[UR4][R4.64], R6
EXIT`, prologue + `
LDG.E R6, desc[UR4][R2.64]
ISETP.GT.AND P0, PT, R0, 0x3, PT
IADD3 R9, R0, 0x1, RZ
@!P0 BRA 0xb0
FMUL R9, R6, 2
@P0 FADD R6, R6, R9
STG.E desc[UR4][R4.64], R6
EXIT`, Equal, ""},
		{"stale value used", prologue + `
LDG.E R6, desc[UR4][R2.64]
ISETP.GT.AND P0, PT, R0, 0x3, PT
MOV R9, R0
@!P0 BRA 0xb0
FMUL R9, R6, 2
@!P0 FADD R6, R6, R9
STG.E desc[UR4][R4.64], R6
EXIT`, prologue + `
LDG.E R6, desc[UR4][R2.64]
ISETP.GT.AND P0, PT, R0, 0x3, PT
IADD3 R9, R0, 0x1, RZ
@!P0 BRA 0xb0
FMUL R9, R6, 2
@!P0 FADD R6, R6, R9
STG.E desc[UR4][R4.64], R6
EXIT`, Unknown, ""},
		{"guard that changes between iterations", prologue + `
LDC R6, c[0x0][0x220]
MOV R8, RZ
MOV R9, R0
LDG.E R7, desc[UR4][R2.64]
IADD3 R6, R6, -0x1, RZ
ISETP.NE.AND P0, PT, R6, RZ, PT
@!P0 FADD R8, R8, R9
FMUL R9, R7, 2
@P0 BRA 0xa0
STG.E desc[UR4][R4.64], R8
EXIT`, prologue + `
LDC R6, c[0x0][0x220]
MOV R8, RZ
MOV R9, R0
LDG.E R7, desc[UR4][R2.64]
IADD3 R6, R6, -0x1, RZ
ISETP.NE.AND P0, PT, R6, RZ, PT
@!P0 FADD R8, R8, R9
FMUL R9, R7, 3
@P0 BRA 0xa0
STG.E desc[UR4][R4.64], R8
EXIT`, Different, ""},
		{"predicate saved and restored", prologue + `
ISETP.GT.AND P0, PT, R0, 0x3, PT
P2R R9, PR, RZ, 0x1
ISETP.GT.AND P0, PT, R0, 0x7, PT
ISETP.NE.AND P1, PT, R9, RZ, PT
LDG.E R6, desc[UR4][R2.64]
@P1 FADD R6, R6, 1
STG.E desc[UR4][R4.64], R6
EXIT`, prologue + `
ISETP.GT.AND P1, PT, R0, 0x3, PT
LDG.E R6, desc[UR4][R2.64]
@P1 FADD R6, R6, 1
STG.E desc[UR4][R4.64], R6
EXIT`, Equal, ""},
		{"predicate restored from another bit", prologue + `
ISETP.GT.AND P0, PT, R0, 0x3, PT
ISETP.GT.AND P1, PT, R0, 0x5, PT
P2R R9, PR, RZ, 0x2
ISETP.GT.AND P0, PT, R0, 0x7, PT
ISETP.NE.AND P1, PT, R9, RZ, PT
LDG.E R6, desc[UR4][R2.64]
@P1 FADD R6, R6, 1
STG.E desc[UR4][R4.64], R6
EXIT`, prologue + `
ISETP.GT.AND P1, PT, R0, 0x3, PT
LDG.E R6, desc[UR4][R2.64]
@P1 FADD R6, R6, 1
STG.E desc[UR4][R4.64], R6
EXIT`, Unknown, ""},
		{"high word added two ways", prologue + `
IADD3 R2, P1, R2, 0x4, RZ
IADD3.X R3, RZ, R3, RZ, P1, !PT
LDG.E R6, desc[UR4][R2.64]
STG.E desc[UR4][R4.64], R6
EXIT`, prologue + `
IADD3 R2, P1, R2, 0x4, RZ
IMAD.X R3, RZ, RZ, R3, P1
LDG.E R6, desc[UR4][R2.64]
STG.E desc[UR4][R4.64], R6
EXIT`, Equal, ""},
		{"high word from another register", prologue + `
IADD3 R2, P1, R2, 0x4, RZ
IADD3.X R3, RZ, R3, RZ, P1, !PT
LDG.E R6, desc[UR4][R2.64]
STG.E desc[UR4][R4.64], R6
EXIT`, prologue + `
IADD3 R2, P1, R2, 0x4, RZ
IMAD.X R3, RZ, RZ, R5, P1
LDG.E R6, desc[UR4][R2.64]
STG.E desc[UR4][R4.64], R6
EXIT`, Unknown, ""},
		{"high half of a word permuted two ways", prologue + `
LDG.E R6, desc[UR4][R2.64]
PRMT R7, R6, 0x7632, R6
IMAD.U32 R7, R7, 0x10000, RZ
FADD R7, R7, 1
STG.E desc[UR4][R4.64], R7
EXIT`, prologue + `
LDG.E R6, desc[UR4][R2.64]
PRMT R7, RZ, 0x76, R6
IMAD.U32 R7, R7, 0x10000, RZ
FADD R7, R7, 1
STG.E desc[UR4][R4.64], R7
EXIT`, Equal, ""},
		{"low half permuted instead", prologue + `
LDG.E R6, desc[UR4][R2.64]
PRMT R7, R6, 0x7632, R6
IMAD.U32 R7, R7, 0x10000, RZ
FADD R7, R7, 1
STG.E desc[UR4][R4.64], R7
EXIT`, prologue + `
LDG.E R6, desc[UR4][R2.64]
PRMT R7, RZ, 0x54, R6
IMAD.U32 R7, R7, 0x10000, RZ
FADD R7, R7, 1
STG.E desc[UR4][R4.64], R7
EXIT`, Unknown, ""},
		{"constant kept across calls", prologue + `
LDG.E R6, desc[UR4][R2.64]
CALL.REL.NOINC 0x120
MOV R9, 0x3e99999a
FMUL R6, R6, R9
CALL.REL.NOINC 0x120
MOV R9, 0x3e99999a
FMUL R6, R6, R9
FMUL R9, R6, R6
CALL.REL.NOINC 0x120
FADD R6, R6, R9
STG.E desc[UR4][R4.64], R6
EXIT
FADD R6, R6, 1
RET.REL.NODEC R10 0x0`, prologue + `
LDG.E R6, desc[UR4][R2.64]
MOV R9, 0x3e99999a
CALL.REL.NOINC 0x110
FMUL R6, R6, R9
CALL.REL.NOINC 0x110
FMUL R6, R6, R9
FMUL R9, R6, R6
CALL.REL.NOINC 0x110
FADD R6, R6, R9
STG.E desc[UR4][R4.64], R6
EXIT
FADD R6, R6, 1
RET.REL.NODEC R10 0x0`, Equal, ""},
		{"constant clobbered by the subroutine", prologue + `
LDG.E R6, desc[UR4][R2.64]
CALL.REL.NOINC 0x120
MOV R9, 0x3e99999a
FMUL R6, R6, R9
CALL.REL.NOINC 0x120
MOV R9, 0x3e99999a
FMUL R6, R6, R9
FMUL R9, R6, R6
CALL.REL.NOINC 0x120
FADD R6, R6, R9
STG.E desc[UR4][R4.64], R6
EXIT
FADD R6, R6, 1
RET.REL.NODEC R10 0x0`, prologue + `
LDG.E R6, desc[UR4][R2.64]
MOV R9, 0x3e99999a
CALL.REL.NOINC 0x110
FMUL R6, R6, R9
CALL.REL.NOINC 0x110
FMUL R6, R6, R9
FMUL R9, R6, R6
CALL.REL.NOINC 0x110
FADD R6, R6, R9
STG.E desc[UR4][R4.64], R6
EXIT
FADD R9, R6, 1
RET.REL.NODEC R10 0x0`, Different, ""},
		{"moves after a return", prologue + `
LDG.E R6, desc[UR4][R2.64]
ISETP.GT.AND P0, PT, R0, 0x3, PT
@P0 BRA 0xa0
CALL.REL.NOINC 0xc0
STG.E desc[UR4][R4.64], R6
EXIT
FADD R6, R6, 1
RET.REL.NODEC R10 0x0`, prologue + `
LDG.E R7, desc[UR4][R2.64]
MOV R6, R7
ISETP.GT.AND P0, PT, R0, 0x3, PT
@P0 BRA 0xc0
CALL.REL.NOINC 0xe0
MOV R6, R7
STG.E desc[UR4][R4.64], R6
EXIT
FADD R7, R7, 1
RET.REL.NODEC R10 0x0`, Equal, ""},
		{"branch with a predicate operand", prologue + `
LDG.E R6, desc[UR4][R2.64]
ISETP.GT.AND P2, PT, R0, 0x3, PT
ISETP.GT.AND P3, PT, R0, 0x1, PT
@P2 FADD R6, R6, 1
@!P2 BRA !P3, 0xc0
STG.E desc[UR4][R4.64], R6
EXIT`, prologue + `
LDG.E R7, desc[UR4][R2.64]
ISETP.GT.AND P2, PT, R0, 0x3, PT
ISETP.GT.AND P3, PT, R0, 0x1, PT
@!P2 BRA !P3, 0xc0
FADD R6, R7, 1
STG.E desc[UR4][R4.64], R6
EXIT`, Unknown, ""},
		{"taken branch with a predicate operand", prologue + `
LDG.E R6, desc[UR4][R2.64]
ISETP.GT.AND P2, PT, R0, 0x3, PT
ISETP.GT.AND P3, PT, R0, 0x1, PT
@P2 FADD R6, R6, 1
@P2 BRA P3, 0xc0
EXIT
STG.E desc[UR4][R4.64], R6
EXIT`, prologue + `
LDG.E R7, desc[UR4][R2.64]
ISETP.GT.AND P2, PT, R0, 0x3, PT
ISETP.GT.AND P3, PT, R0, 0x1, PT
@P2 BRA P3, 0xb0
EXIT
FADD R6, R7, 1
STG.E desc[UR4][R4.64], R6
EXIT`, Equal, ""},
		{"halves of two loads", prologue + `
LDG.E.64 R6, desc[UR4][R2.64]
LDG.E.64 R8, desc[UR4][R2.64+0x100]
STG.E.64 desc[UR4][R4.64], R6
EXIT`, prologue + `
LDG.E.64 R6, desc[UR4][R2.64]
LDG.E.64 R8, desc[UR4][R2.64+0x100]
MOV R10, R6
MOV R11, R9
STG.E.64 desc[UR4][R4.64], R10
EXIT`, Unknown, ""},
		{"sign extended index scaled two ways", prologue + `
IMAD.WIDE R6, R0, 0x4, R2
LDG.E R8, desc[UR4][R6.64]
STG.E desc[UR4][R4.64], R8
EXIT`, prologue + `
SHF.R.S32.HI R9, RZ, 0x1f, R0
LEA R6, P0, R0, R2, 0x2
LEA.HI.X R7, R0, R3, R9, 0x2, P0
LDG.E R8, desc[UR4][R6.64]
STG.E desc[UR4][R4.64], R8
EXIT`, Equal, ""},
		{"index extended with the wrong word", prologue + `
IMAD.WIDE R6, R0, 0x4, R2
LDG.E R8, desc[UR4][R6.64]
STG.E desc[UR4][R4.64], R8
EXIT`, prologue + `
SHF.R.S32.HI R9, RZ, 0x1e, R0
LEA R6, P0, R0, R2, 0x2
LEA.HI.X R7, R0, R3, R9, 0x2, P0
LDG.E R8, desc[UR4][R6.64]
STG.E desc[UR4][R4.64], R8
EXIT`, Unknown, ""},
		{"register after a 64-bit conversion source", prologue + `
LDG.E.64 R8, desc[UR4][R2.64]
LDG.E R6, desc[UR4][R2.64+0x100]
FMUL R7, R6, 2
I2F.S64 R6, R8
FADD R7, R7, R6
STG.E desc[UR4][R4.64], R7
EXIT`, prologue + `
LDG.E.64 R8, desc[UR4][R2.64]
LDG.E R6, desc[UR4][R2.64+0x100]
FMUL R7, R6, 3
I2F.S64 R6, R8
FADD R7, R7, R6
STG.E desc[UR4][R4.64], R7
EXIT`, Different, ""},
		{"high word of a 64-bit conversion source", prologue + `
LDG.E.64 R8, desc[UR4][R2.64]
I2F.S64 R6, R8
STG.E desc[UR4][R4.64], R6
EXIT`, prologue + `
LDG.E.64 R8, desc[UR4][R2.64]
MOV R9, RZ
I2F.S64 R6, R8
STG.E desc[UR4][R4.64], R6
EXIT`, Unknown, ""},
		{"generic address in another pair", prologue + `
MOV R16, R2
MOV R17, R3
LD.E R6, [R16]
STG.E desc[UR4][R4.64], R6
EXIT`, prologue + `
MOV R10, R2
MOV R11, R3
LD.E R6, [R10]
STG.E desc[UR4][R4.64], R6
EXIT`, Equal, ""},
		{"generic address with another high word", prologue + `
MOV R16, R2
MOV R17, R3
LD.E R6, [R16]
STG.E desc[UR4][R4.64], R6
EXIT`, prologue + `
MOV R16, R2
MOV R17, R5
LD.E R6, [R16]
STG.E desc[UR4][R4.64], R6
EXIT`, Unknown, ""},
		{"register undefined on one path", prologue + `
LDG.E R6, desc[UR4][R2.64]
ISETP.GT.AND P0, PT, R0, 0x7, PT
@P0 MOV R7, R6
@!P0 MOV R7, R6
FADD R8, R7, 1
STG.E desc[UR4][R4.64], R8
EXIT`, prologue + `
LDG.E R6, desc[UR4][R2.64]
ISETP.GT.AND P0, PT, R0, 0x7, PT
@P0 MOV R7, R6
FADD R8, R7, 1
STG.E desc[UR4][R4.64], R8
EXIT`, Unknown, ""},
		{"same register read undefined", prologue + `
LOP3.LUT R9, R9, 0xfffffffd, RZ, 0xc0, !PT
STG.E desc[UR4][R4.64], R9
EXIT`, prologue + `
LOP3.LUT R9, R9, 0xfffffffd, RZ, 0xc0, !PT
STG.E desc[UR4][R4.64], R9
EXIT`, Equal, ""},
		{"another register read undefined", prologue + `
LOP3.LUT R9, R9, 0xfffffffd, RZ, 0xc0, !PT
STG.E desc[UR4][R4.64], R9
EXIT`, prologue + `
LOP3.LUT R9, R10, 0xfffffffd, RZ, 0xc0, !PT
STG.E desc[UR4][R4.64], R9
EXIT`, Unknown, ""},
		{"uniform constant loaded either way", prologue + `
ULDC.64 UR6, c[0x0][0x220]
MOV R6, UR7
STG.E desc[UR4][R4.64], R6
EXIT`, prologue + `
LDCU.64 UR6, c[0x0][0x220]
MOV R6, UR7
STG.E desc[UR4][R4.64], R6
EXIT`, Equal, ""},
		{"negated predicate written under the same guard", prologue + `
LDG.E R6, desc[UR4][R2.64]
ISETP.GT.AND P1, PT, R0, 0x7, PT
MOV R7, 0x5
@!P1 ISETP.GE.AND P3, PT, R6, 0x20, PT
@!P1 SEL R7, R6, 0xffffffff, !P3
STG.E desc[UR4][R4.64], R7
EXIT`, prologue + `
LDG.E R6, desc[UR4][R2.64]
ISETP.GT.AND P1, PT, R0, 0x7, PT
MOV R7, 0x5
@!P1 ISETP.GE.AND P4, PT, R6, 0x20, PT
@!P1 SEL R7, R6, 0xffffffff, !P4
STG.E desc[UR4][R4.64], R7
EXIT`, Equal, ""},
		{"two exits in one block", prologue + `
LDG.E R6, desc[UR4][R2.64]
ISETP.GE.AND P0, PT, R6, RZ, PT
@!P0 EXIT
STG.E desc[UR4][R4.64], R6
EXIT`, prologue + `
LDG.E R6, desc[UR4][R2.64+0x4]
ISETP.GE.AND P0, PT, R6, RZ, PT
@!P0 EXIT
LDG.E R6, desc[UR4][R2.64]
STG.E desc[UR4][R4.64], R6
EXIT`, Different, ""},
		{"mask before or after a shift", prologue + `
LOP3.LUT R7, R0, 0x1f, RZ, 0xc0, !PT
IMAD.SHL.U32 R7, R7, 0x2, RZ
STG.E desc[UR4][R4.64], R7
EXIT`, prologue + `
IMAD.SHL.U32 R7, R0, 0x2, RZ
LOP3.LUT R7, R7, 0x3e, RZ, 0xe2, !PT
STG.E desc[UR4][R4.64], R7
EXIT`, Equal, ""},
		{"mask before or after a shift, one bit more", prologue + `
LOP3.LUT R7, R0, 0x1f, RZ, 0xc0, !PT
IMAD.SHL.U32 R7, R7, 0x2, RZ
STG.E desc[UR4][R4.64], R7
EXIT`, prologue + `
IMAD.SHL.U32 R7, R0, 0x2, RZ
LOP3.LUT R7, R7, 0x7e, RZ, 0xe2, !PT
STG.E desc[UR4][R4.64], R7
EXIT`, Unknown, ""},
		{"mask of the complement", prologue + `
LOP3.LUT R7, R0, 0x1f, RZ, 0xc0, !PT
STG.E desc[UR4][R4.64], R7
EXIT`, prologue + `
LOP3.LUT R7, R0, 0x1f, RZ, 0x0c, !PT
STG.E desc[UR4][R4.64], R7
EXIT`, Unknown, ""},
		{"mask of a mask", prologue + `
LOP3.LUT R7, R0, 0xf0, RZ, 0xc0, !PT
LOP3.LUT R7, R7, 0x3c, RZ, 0xc0, !PT
STG.E desc[UR4][R4.64], R7
EXIT`, prologue + `
LOP3.LUT R7, R0, 0x30, RZ, 0xc0, !PT
STG.E desc[UR4][R4.64], R7
EXIT`, Equal, ""},
		{"mask written with the operands swapped", prologue + `
LOP3.LUT R7, R0, 0x1f, RZ, 0xc0, !PT
STG.E desc[UR4][R4.64], R7
EXIT`, prologue + `
LOP3.LUT R7, RZ, R0, 0x1f, 0x88, !PT
STG.E desc[UR4][R4.64], R7
EXIT`, Equal, ""},
		{"low bits cleared after a shift", prologue + `
SHF.R.U32.HI R7, RZ, 0x5, R0
IMAD.SHL.U32 R7, R7, 0x10, RZ
STG.E desc[UR4][R4.64], R7
EXIT`, prologue + `
SHF.R.U32.HI R7, RZ, 0x1, R0
LOP3.LUT R7, R7, 0x7ffffff0, RZ, 0xc0, !PT
STG.E desc[UR4][R4.64], R7
EXIT`, Equal, ""},
		{"low bits cleared after a shift, one bit more", prologue + `
SHF.R.U32.HI R7, RZ, 0x5, R0
IMAD.SHL.U32 R7, R7, 0x10, RZ
STG.E desc[UR4][R4.64], R7
EXIT`, prologue + `
SHF.R.U32.HI R7, RZ, 0x1, R0
LOP3.LUT R7, R7, 0x7fffffe0, RZ, 0xc0, !PT
STG.E desc[UR4][R4.64], R7
EXIT`, Unknown, ""},
		{"low bits cleared", prologue + `
LOP3.LUT R7, R0, 0xfffffff0, RZ, 0xc0, !PT
STG.E desc[UR4][R4.64], R7
EXIT`, prologue + `
SHF.R.U32.HI R7, RZ, 0x4, R0
IMAD.SHL.U32 R7, R7, 0x10, RZ
STG.E desc[UR4][R4.64], R7
EXIT`, Equal, ""},
		{"mask keeping every bit a shift leaves", prologue + `
SHF.R.U32.HI R7, RZ, 0x4, R0
LOP3.LUT R7, R7, 0xfffffff, RZ, 0xc0, !PT
STG.E desc[UR4][R4.64], R7
EXIT`, prologue + `
SHF.R.U32.HI R7, RZ, 0x4, R0
STG.E desc[UR4][R4.64], R7
EXIT`, Equal, ""},
		{"shift of a shift", prologue + `
SHF.R.U32.HI R7, RZ, 0x2, R0
SHF.R.U32.HI R7, RZ, 0x3, R7
STG.E desc[UR4][R4.64], R7
EXIT`, prologue + `
SHF.R.U32.HI R7, RZ, 0x5, R0
STG.E desc[UR4][R4.64], R7
EXIT`, Equal, ""},
		{"shift of a shift, one more", prologue + `
SHF.R.U32.HI R7, RZ, 0x2, R0
SHF.R.U32.HI R7, RZ, 0x3, R7
STG.E desc[UR4][R4.64], R7
EXIT`, prologue + `
SHF.R.U32.HI R7, RZ, 0x6, R0
STG.E desc[UR4][R4.64], R7
EXIT`, Unknown, ""},
		{"negation as a move", prologue + `
IMAD.MOV R7, RZ, RZ, -R0
STG.E desc[UR4][R4.64], R7
EXIT`, prologue + `
IADD3 R7, -R0, RZ, RZ
STG.E desc[UR4][R4.64], R7
EXIT`, Equal, ""},
		{"low bits cleared after a left shift", prologue + `
SHF.L.U32 R7, R0, 0x2, RZ
LOP3.LUT R7, R7, 0xffffff80, RZ, 0xc0, !PT
STG.E desc[UR4][R4.64], R7
EXIT`, prologue + `
SHF.R.U32.HI R7, RZ, 0x5, R0
IMAD.SHL.U32 R7, R7, 0x80, RZ
STG.E desc[UR4][R4.64], R7
EXIT`, Equal, ""},
		{"low bits cleared after a left shift, one bit more", prologue + `
SHF.L.U32 R7, R0, 0x2, RZ
LOP3.LUT R7, R7, 0xffffff00, RZ, 0xc0, !PT
STG.E desc[UR4][R4.64], R7
EXIT`, prologue + `
SHF.R.U32.HI R7, RZ, 0x5, R0
IMAD.SHL.U32 R7, R7, 0x80, RZ
STG.E desc[UR4][R4.64], R7
EXIT`, Unknown, ""},
		{"mask bits a shift discards", prologue + `
LOP3.LUT R7, R0, 0x8000001f, RZ, 0xc0, !PT
IMAD.SHL.U32 R7, R7, 0x2, RZ
STG.E desc[UR4][R4.64], R7
EXIT`, prologue + `
LOP3.LUT R7, R0, 0x1f, RZ, 0xc0, !PT
IMAD.SHL.U32 R7, R7, 0x2, RZ
STG.E desc[UR4][R4.64], R7
EXIT`, Equal, ""},
		{"mask bits a shift keeps", prologue + `
LOP3.LUT R7, R0, 0x4000001f, RZ, 0xc0, !PT
IMAD.SHL.U32 R7, R7, 0x2, RZ
STG.E desc[UR4][R4.64], R7
EXIT`, prologue + `
LOP3.LUT R7, R0, 0x1f, RZ, 0xc0, !PT
IMAD.SHL.U32 R7, R7, 0x2, RZ
STG.E desc[UR4][R4.64], R7
EXIT`, Unknown, ""},
		{"low word of a wide multiply-add", prologue + `
IMAD.WIDE R8, R0, 0x2000, R2
MOV R7, R8
STG.E desc[UR4][R4.64], R7
EXIT`, prologue + `
IMAD R7, R0, 0x2000, R2
STG.E desc[UR4][R4.64], R7
EXIT`, Equal, ""},
		{"low word of a wide multiply-add, constant first", prologue + `
IMAD.WIDE.U32 R8, 0x2000, R0, R2
MOV R7, R8
STG.E desc[UR4][R4.64], R7
EXIT`, prologue + `
IMAD R7, R0, 0x2000, R2
STG.E desc[UR4][R4.64], R7
EXIT`, Equal, ""},
		{"low word of a wide multiply-add, other constant", prologue + `
IMAD.WIDE R8, R0, 0x1000, R2
MOV R7, R8
STG.E desc[UR4][R4.64], R7
EXIT`, prologue + `
IMAD R7, R0, 0x2000, R2
STG.E desc[UR4][R4.64], R7
EXIT`, Unknown, ""},
		{"high word of a wide multiply-add", prologue + `
IMAD.WIDE R8, R0, 0x2000, R2
MOV R7, R9
STG.E desc[UR4][R4.64], R7
EXIT`, prologue + `
IMAD R7, R0, 0x2000, R2
STG.E desc[UR4][R4.64], R7
EXIT`, Unknown, ""},
		{"sum that also writes a carry", prologue + `
LEA R7, P0, R0, R2, 0x1
STG.E desc[UR4][R4.64], R7
EXIT`, prologue + `
IMAD R7, R0, 0x2, R2
STG.E desc[UR4][R4.64], R7
EXIT`, Equal, ""},
		{"sum that also writes a carry, other shift", prologue + `
LEA R7, P0, R0, R2, 0x2
STG.E desc[UR4][R4.64], R7
EXIT`, prologue + `
IMAD R7, R0, 0x2, R2
STG.E desc[UR4][R4.64], R7
EXIT`, Unknown, ""},
		{"carry of a sum", prologue + `
LEA R6, P0, R0, R2, 0x1
SEL R7, RZ, 0x1, !P0
STG.E desc[UR4][R4.64], R7
EXIT`, prologue + `
IMAD R6, R0, 0x2, R2
ISETP.LT.U32.AND P0, PT, R6, R2, PT
SEL R7, RZ, 0x1, !P0
STG.E desc[UR4][R4.64], R7
EXIT`, Unknown, ""},
		{"sign extended index scaled with LEA.HI.X.SX32", prologue + `
IMAD.WIDE R6, R0, 0x4, R2
LDG.E R8, desc[UR4][R6.64]
STG.E desc[UR4][R4.64], R8
EXIT`, prologue + `
LEA R6, P0, R0, R2, 0x2
LEA.HI.X.SX32 R7, R0, R3, 0x2, P0
LDG.E R8, desc[UR4][R6.64]
STG.E desc[UR4][R4.64], R8
EXIT`, Equal, ""},
		{"low word changed after its carry", prologue + `
IMAD.WIDE.U32 R6, R0, 0x4, R2
LDG.E R8, desc[UR4][R6.64]
STG.E desc[UR4][R4.64], R8
EXIT`, prologue + `
LEA R6, P0, R0, R2, 0x2
IADD3 R6, R6, 0x4, RZ
LEA.HI.X R7, R0, R3, RZ, 0x2, P0
LDG.E R8, desc[UR4][R6.64]
STG.E desc[UR4][R4.64], R8
EXIT`, Unknown, ""},
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

func TestInline(t *testing.T) {
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
	blocks, edges := inline(cfg)
	var got []string
	for _, e := range edges {
		got = append(got, fmt.Sprintf("%d:%#x->%d:%#x", e.From, blocks[e.From].Start, e.To, blocks[e.To].Start))
	}
	// Each call enters its own copy of the subroutine, whose return
	// reaches the instruction after that call; no call falls through.
	want := []string{"0:0x0->1:0x90", "1:0x90->2:0x70", "2:0x70->3:0x90", "3:0x90->4:0x80"}
	if !slices.Equal(got, want) {
		t.Errorf("edges = %v, want %v", got, want)
	}
}

func TestInlineUnknown(t *testing.T) {
	tests := []struct {
		name, body, why string
	}{
		{"recursion", `
CALL.REL.NOINC 0x80
EXIT
CALL.REL.NOINC 0x80
RET.REL.NODEC R10 0x0`, "recursive call"},
		{"guarded call", `
ISETP.GT.AND P0, PT, R0, 0x3, PT
@P0 CALL.REL.NOINC 0x90
EXIT
RET.REL.NODEC R10 0x0`, "guarded call"},
		{"return from the kernel", `
RET.REL.NODEC R10 0x0`, "outside a call"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := parse(t, sass(prologue+tt.body))
			r := Compare(d.Functions[0], d.Functions[0])
			if r.Verdict != Unknown || !strings.Contains(r.Reason, tt.why) {
				t.Errorf("got %v, want unknown %q", r, tt.why)
			}
		})
	}
}
