package sassequiv

import (
	"slices"
	"strconv"
	"strings"
	"testing"
)

// defsUses runs one instruction and reports the locations it writes and
// reads. Every location a lone instruction reads is unwritten, so its
// uses are the undefined values and memory the run creates.
func defsUses(t *testing.T, line string) (defs, uses []string, why string) {
	t.Helper()
	d := parse(t, sass(line+"\nEXIT"))
	g := newGraph()
	fx := build(g, 0, d.Functions[0])
	if fx.why != "" {
		return nil, nil, fx.why
	}
	for k := range fx.outs[0] {
		defs = append(defs, k)
	}
	for _, n := range g.nodes {
		switch {
		case strings.HasPrefix(n.label, "undef:"):
			uses = append(uses, strings.TrimPrefix(n.label, "undef:"))
		case n.label == "in:MEM":
			uses = append(uses, "MEM")
		}
	}
	slices.Sort(defs)
	slices.Sort(uses)
	return defs, slices.Compact(uses), ""
}

// TestDefsUses pins the registers each form in the corpus writes and
// reads. The lines are cuobjdump output from the corpus; the widths are
// the instruction set's. EXIT reads memory, so every use list has MEM.
func TestDefsUses(t *testing.T) {
	preds := "P0 P1 P2 P3 P4 P5 P6"
	tests := []struct {
		line       string
		defs, uses string
	}{
		// Conversions: a lone type names the side that is not the
		// default, F32 for a float and S32 for an integer.
		{"I2F.S64 R14, R18", "R14", "R18 R19"},
		{"I2F.U64 R11, R32", "R11", "R32 R33"},
		{"I2F.U64.RP R16, R6", "R16", "R6 R7"},
		{"I2F.F64.S64 R10, R10", "R10 R11", "R10 R11"},
		{"I2F.S16 R12, R12", "R12", "R12"},
		{"I2F.U32.RP R7, R0", "R7", "R0"},
		{"I2FP.F32.S32 R13, R14", "R13", "R14"},
		{"I2FP.F32.U32 R0, R100", "R0", "R100"},
		{"F2I.NTZ R14, R10", "R14", "R10"},
		{"F2I.U64.TRUNC R12, R12", "R12 R13", "R12"},
		{"F2I.FTZ.U32.TRUNC.NTZ R13, R12", "R13", "R12"},
		{"F2F.F32.F64 R10, R10", "R10", "R10 R11"},
		{"F2FP.BF16.F32.PACK_AB R0, R0, R3", "R0", "R0 R3"},
		{"F2FP.F16.E4M3.UNPACK_B R10, R10", "R10", "R10"},

		// Wide arithmetic.
		{"IMAD.WIDE R10, R0, 0x4, R10", "R10 R11", "R0 R10 R11"},
		{"IMAD.WIDE.U32 R10, R10, UR7, R8", "R10 R11", "R10 R8 R9 UR7"},
		{"UIMAD.WIDE UR12, UR19, 0x210, UR8", "UR12 UR13", "UR19 UR8 UR9"},
		{"UIMAD.WIDE.U32 UR10, UR11, UR10, URZ", "UR10 UR11", "UR10 UR11"},
		{"IMAD.HI R0, R111, -0x6db6db6d, R110", "R0", "R110 R111"},
		{"DMUL R10, R10, UR6", "R10 R11", "R10 R11 UR6 UR7"},
		{"DFMA R2, R8, R2, 1", "R2 R3", "R2 R3 R8 R9"},
		{"IADD3.X R0, R3, R9, RZ, P0, !PT", "R0", "P0 R3 R9"},
		{"LEA.HI.X R0, R10, R2, R73, 0x6, P1", "R0", "P1 R10 R2 R73"},
		{"LEA.HI.X.SX32 R0, R0, R3, 0x1, P0", "R0", "P0 R0 R3"},

		// Funnel shifts and byte permutes read two 32-bit halves.
		{"SHF.R.U64 R11, R16, 0x4, R17", "R11", "R16 R17"},
		{"SHF.L.U64.HI R0, R0, 0x2, R5", "R0", "R0 R5"},
		{"SHF.R.S32.HI R0, RZ, 0x1f, R104", "R0", "R104"},
		{"SHF.L.U32.HI R11, R20, 0x2, R15", "R11", "R15 R20"},
		{"USHF.L.U64.HI UR22, UR6, 0x3, UR7", "UR22", "UR6 UR7"},
		{"PRMT R0, R0, 0x7362, R81", "R0", "R0 R81"},

		// Matrix multiplies: m16n8k16 takes A in 4 registers, B in 2
		// and C in 4; a 64x128 HGMMA has 64 accumulators and two
		// 64-bit descriptors.
		{"HMMA.16816.F32.BF16 R100, R112, R146, R100", "R100 R101 R102 R103", "R100 R101 R102 R103 R112 R113 R114 R115 R146 R147"},
		{"HMMA.16816.F32 R20, R44, R10, R20", "R20 R21 R22 R23", "R10 R11 R20 R21 R22 R23 R44 R45 R46 R47"},
		{"HGMMA.64x128x16.F32.BF16 R24, gdesc[UR12], R24", regs("R", 24, 64), regs("R", 24, 64) + " MEM UR12 UR13 UR14 UR15"},

		// Loads and stores: desc names a uniform pair, .64 a register
		// pair, and an .E access without a descriptor a pair too.
		{"LDG.E.64 R10, desc[UR8][R2.64+0x18]", "R10 R11", "R2 R3 UR8 UR9"},
		{"LDG.E.128 R12, desc[UR8][R12.64]", "R12 R13 R14 R15", "R12 R13 UR8 UR9"},
		{"STG.E.64 desc[UR20][R34.64+0x18], R36", "MEM", "R34 R35 R36 R37 UR20 UR21"},
		{"STG.E.128 desc[UR10][R32.64], R20", "MEM", "R20 R21 R22 R23 R32 R33 UR10 UR11"},
		{"LDS.64 R10, [R62+0x1450]", "R10 R11", "R62"},
		{"LDS.128 R12, [R15]", "R12 R13 R14 R15", "R15"},
		{"STS.64 [R34], R6", "MEM", "R34 R6 R7"},
		{"STS.128 [R0], R4", "MEM", "R0 R4 R5 R6 R7"},
		{"LDSM.16.M88.4 R100, [R116+0x1200]", "R100 R101 R102 R103", "R116"},
		{"LD.E R17, [R16]", "R17", "R16 R17"},
		{"ATOMG.E.CAS.STRONG.GPU PT, RZ, [R14], R16, R17", "MEM", "R14 R15 R16 R17"},
		{"ATOMG.E.ADD.64.STRONG.GPU PT, R14, desc[UR24][R16.64], R18", "MEM R14 R15", "R16 R17 R18 R19 UR24 UR25"},
		{"REDG.E.ADD.STRONG.GPU desc[UR24][R14.64], R17", "MEM", "R14 R15 R17 UR24 UR25"},
		{"LDGSTS.E.BYPASS.128 [R103], desc[UR10][R14.64]", "MEM", "R103 R14 R15 UR10 UR11"},

		// Stack slots are locations of their own.
		{"STL [R1], R16", "L0x0", "R16"},
		{"LDL.LU R0, [R1+0x4]", "R0", "L0x4"},

		// Moves and constants.
		{"MOV.64 R12, UR18", "R12 R13", "UR18 UR19"},
		{"LDC.64 R10, c[0x0][0x210]", "R10 R11", ""},
		{"ULDC.64 UR10, c[0x0][0x208]", "UR10 UR11", ""},
		{"LDCU.64 UR10, c[0x0][0x390]", "UR10 UR11", ""},
		{"CS2R R10, SRZ", "R10 R11", ""},
		{"SHFL.BFLY P0, R10, R4, R12, R13", "P0 R10", "R12 R13 R4"},

		// P2R reads the predicates its mask names. R2P names every
		// predicate through PR, and its mask selects among them later.
		{"P2R R0, PR, RZ, 0x1", "R0", "P0"},
		{"P2R R16, PR, RZ, 0x40", "R16", "P6"},
		{"R2P PR, R15, 0x7e", preds, preds + " R15"},
	}
	for _, tt := range tests {
		t.Run(tt.line, func(t *testing.T) {
			defs, uses, why := defsUses(t, tt.line)
			if why != "" {
				t.Fatalf("unknown: %s", why)
			}
			want := strings.Fields(tt.uses + " MEM")
			slices.Sort(want)
			want = slices.Compact(want)
			wantDefs := strings.Fields(tt.defs)
			slices.Sort(wantDefs)
			if !slices.Equal(defs, wantDefs) {
				t.Errorf("defs = %v, want %v", defs, wantDefs)
			}
			if !slices.Equal(uses, want) {
				t.Errorf("uses = %v, want %v", uses, want)
			}
		})
	}
}

// regs returns n consecutive registers starting at prefix+first.
func regs(prefix string, first, n int) string {
	var s []string
	for k := range n {
		s = append(s, prefix+strconv.Itoa(first+k))
	}
	return strings.Join(s, " ")
}
