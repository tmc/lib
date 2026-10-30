package bitopt

import (
	"errors"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/sassequiv"
)

func TestSASSInstFormatting(t *testing.T) {
	tests := []struct {
		name string
		inst Inst
		want string
	}{
		{
			name: "OpAndImm",
			inst: Inst{Op: OpAndImm, Dst: "R8", Src0: "R6", Imm: 0x81c081c0},
			want: "LOP3.LUT R8, R6, 0x81c081c0, RZ, 0xc0, !PT",
		},
		{
			name: "OpOrImm",
			inst: Inst{Op: OpOrImm, Dst: "R8", Src0: "R6", Imm: 0x7700},
			want: "LOP3.LUT R8, R6, 0x7700, RZ, 0xfc, !PT",
		},
		{
			name: "OpXorImm",
			inst: Inst{Op: OpXorImm, Dst: "R8", Src0: "R6", Imm: 0x80008000},
			want: "LOP3.LUT R8, R6, 0x80008000, RZ, 0x3c, !PT",
		},
		{
			name: "OpShlImm",
			inst: Inst{Op: OpShlImm, Dst: "R8", Src0: "R6", Imm: 4},
			want: "SHF.L.U32 R8, R6, 0x4, RZ",
		},
		{
			name: "OpShrImm",
			inst: Inst{Op: OpShrImm, Dst: "R8", Src0: "R6", Imm: 2},
			want: "SHF.R.U32.HI R8, RZ, 0x2, R6",
		},
		{
			name: "OpAddImm",
			inst: Inst{Op: OpAddImm, Dst: "R8", Src0: "R6", Imm: 0x7700},
			want: "IADD3 R8, R6, 0x7700, RZ",
		},
		{
			name: "OpPRMT",
			inst: Inst{Op: OpPRMT, Dst: "R8", Src0: "R6", Src1: "R7", Imm: 0x4163},
			want: "PRMT R8, R6, 0x4163, R7",
		},
		{
			name: "OpLOP3",
			inst: Inst{Op: OpLOP3, Dst: "R8", Src0: "R8", Src1: "R9", Src2: "RZ", LUT: 0xfc},
			want: "LOP3.LUT R8, R8, R9, RZ, 0xfc, !PT",
		},
		{
			name: "OpSHFL",
			inst: Inst{Op: OpSHFL, Dst: "R8", Src0: "R6", Src1: "R6", Imm: 10},
			want: "SHF.L.U32 R8, R6, 0xa, R6",
		},
		{
			name: "OpSHFR",
			inst: Inst{Op: OpSHFR, Dst: "R8", Src0: "R6", Src1: "R7", Imm: 6},
			want: "SHF.R.U32.HI R8, R6, 0x6, R7",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.inst.SASS()
			if err != nil {
				t.Fatalf("SASS() error: %v", err)
			}
			if got != tt.want {
				t.Errorf("SASS() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestSASSListingAndProveSASSEquiv(t *testing.T) {
	p1 := NVFP4Program1Word()
	listing1 := p1.SASSListing()
	if listing1 == "" {
		t.Fatal("NVFP4Program1Word().SASSListing() returned empty string")
	}
	for _, wantSub := range []string{
		"LDG.E R6, desc[UR4][R2.64]",
		"LOP3.LUT",
		"SHF.L.U32",
		"SHF.R.U32.HI",
		"STG.E desc[UR4][R4.64]",
		"EXIT",
	} {
		if !strings.Contains(listing1, wantSub) {
			t.Errorf("1-word SASSListing missing %q:\n%s", wantSub, listing1)
		}
	}

	p2 := NVFP4Program2Word()
	listing2 := p2.SASSListing()
	if !strings.Contains(listing2, "LDG.E R7, desc[UR4][R2.64+0x4]") || !strings.Contains(listing2, "PRMT") {
		t.Errorf("2-word SASSListing missing 2nd word load or PRMT:\n%s", listing2)
	}

	spec1 := E2M1ToBF16Spec(1)
	p1PreMasked := cloneProgram(p1)
	// Add a redundant pre-shift mask on move 1 (Shl by 4) and reversed part order on output 3.
	p1PreMasked.Moves[1].Sel = 0x081c081c
	p1PreMasked.Parts[3][0], p1PreMasked.Parts[3][1] = p1PreMasked.Parts[3][1], p1PreMasked.Parts[3][0]

	p1WrongShift := cloneProgram(p1)
	p1WrongShift.Moves[1].N = 5 // shifts by 5 instead of 4

	p1WrongMask := cloneProgram(p1)
	p1WrongMask.Parts[0][0].Mask = 0x80c080c0

	tests := []struct {
		name   string
		a, b   *Program
		want   sassequiv.Verdict
		wantEq bool
		verify bool
	}{
		{
			name:   "1-word self equivalence",
			a:      &p1,
			b:      &p1,
			want:   sassequiv.Equal,
			wantEq: true,
			verify: true,
		},
		{
			name:   "1-word pre-masked shift and reordered parts",
			a:      &p1,
			b:      &p1PreMasked,
			want:   sassequiv.Equal,
			wantEq: true,
			verify: true,
		},
		{
			name:   "1-word wrong shift amount",
			a:      &p1,
			b:      &p1WrongShift,
			want:   sassequiv.Different,
			wantEq: false,
			verify: false,
		},
		{
			name:   "1-word wrong output mask",
			a:      &p1,
			b:      &p1WrongMask,
			want:   sassequiv.Different,
			wantEq: false,
			verify: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if gotEq := Equivalent(tt.a, tt.b, spec1); gotEq != tt.wantEq {
				t.Errorf("Equivalent() = %v, want %v", gotEq, tt.wantEq)
			}
			res, err := ProveSASSEquiv(tt.a, tt.b)
			if err != nil {
				t.Fatalf("ProveSASSEquiv() error: %v", err)
			}
			if res.Verdict != tt.want {
				t.Errorf("ProveSASSEquiv() verdict = %v (%s), want %v", res.Verdict, res.Reason, tt.want)
			}
			err = VerifySASSEquiv(tt.a, tt.b, spec1)
			if tt.verify && err != nil {
				t.Errorf("VerifySASSEquiv() unexpected error: %v", err)
			}
			if !tt.verify && err == nil {
				t.Errorf("VerifySASSEquiv() succeeded for inequivalent programs")
			}
		})
	}
}

func TestSuperoptimizeSASS(t *testing.T) {
	spec := Spec{
		Words:   1,
		Outputs: []uint32{0x81C081C0},
	}
	// Verbose 3-instruction SASS sequence:
	//   1. LOP3.LUT Rpre, R6, 0x081c081c, RZ, 0xc0, !PT  (pre-mask via Sel)
	//   2. SHF.L.U32 Rd, Rpre, 0x4, RZ                   (shift left by 4)
	//   3. LOP3.LUT Rout, Rd, 0x81c081c0, RZ, 0xc0, !PT  (output mask)
	// plus an unused dead shift move so Cost() is 3 and Insts() is 4.
	verbose := &Program{
		Problem: spec,
		Moves: []Move{
			{Op: Shl, A: 0, B: 0, N: 4, Sel: 0x081c081c},
			{Op: Shr, A: 0, B: 0, N: 2},
		},
		Parts: [][]Part{
			{{Move: 0, Mask: 0x81C081C0}},
		},
	}
	beforeInsts, _, err := verbose.Insts(nil)
	if err != nil {
		t.Fatalf("verbose.Insts: %v", err)
	}
	if len(beforeInsts) < 3 {
		t.Fatalf("len(beforeInsts) = %d, want >= 3", len(beforeInsts))
	}

	opt, res, err := SuperoptimizeSASS(verbose, spec, 2)
	if err != nil {
		t.Fatalf("SuperoptimizeSASS: %v", err)
	}
	if res.Verdict != sassequiv.Equal {
		t.Fatalf("SuperoptimizeSASS verdict = %v (%s), want Equal", res.Verdict, res.Reason)
	}
	if opt.Cost() != 2 {
		t.Fatalf("opt.Cost() = %d, want 2", opt.Cost())
	}
	afterInsts, _, err := opt.Insts(nil)
	if err != nil {
		t.Fatalf("opt.Insts: %v", err)
	}
	if len(afterInsts) != 2 {
		t.Fatalf("len(afterInsts) = %d, want 2", len(afterInsts))
	}
	if err := VerifySASSEquiv(verbose, opt, spec); err != nil {
		t.Fatalf("VerifySASSEquiv(verbose, opt): %v", err)
	}

	if _, _, err := SuperoptimizeSASS(verbose, spec, 1); !errors.Is(err, ErrNoProgram) {
		t.Fatalf("SuperoptimizeSASS(maxInsts=1) = %v, want ErrNoProgram", err)
	}
}
