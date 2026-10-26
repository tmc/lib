package hopper

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/egraph/bitopt"
	"github.com/tmc/lib/nvidia/x/sassequiv"
)

// VerifyDequantProgramsSASS verifies the synthesized 1-word and 2-word NVFP4
// dequantization bit-routing programs ([bitopt.NVFP4Program1Word] and
// [bitopt.NVFP4Program2Word]) at both the bit-routing and symbolic SASS levels.
func VerifyDequantProgramsSASS() error {
	spec1 := bitopt.E2M1ToBF16Spec(1)
	p1 := bitopt.NVFP4Program1Word()
	if err := bitopt.Check(spec1, p1); err != nil {
		return fmt.Errorf("verify dequant9 1-word spec: %w", err)
	}
	p1Verbose := bitopt.NVFP4Program1Word()
	p1Verbose.Moves[1].Sel = 0x081c081c // redundant pre-shift mask before << 4
	p1Verbose.Moves[3].Sel = 0x02030203 // redundant pre-shift mask before << 6
	opt1, res1, err := bitopt.SuperoptimizeSASS(&p1Verbose, spec1, 9)
	if err != nil {
		return fmt.Errorf("superoptimize dequant9 1-word SASS: %w", err)
	}
	if res1.Verdict != sassequiv.Equal {
		return fmt.Errorf("dequant9 1-word SASS verdict %v: %s", res1.Verdict, res1.Reason)
	}
	if err := bitopt.VerifySASSEquiv(&p1, opt1, spec1); err != nil {
		return fmt.Errorf("verify dequant9 1-word SASS equivalence: %w", err)
	}

	sass1Default, err := p1.SASS([2]string{}, nil)
	if err != nil {
		return fmt.Errorf("lower dequant9 1-word default SASS: %w", err)
	}
	sass1Renamed, err := p1.SASS([2]string{"R4", "R5"}, []string{"R12"})
	if err != nil {
		return fmt.Errorf("lower dequant9 1-word renamed SASS: %w", err)
	}
	cmp1, err := bitopt.CompareSASS(sass1Default, sass1Renamed)
	if err != nil {
		return fmt.Errorf("compare dequant9 1-word renamed SASS: %w", err)
	}
	if cmp1.Verdict != sassequiv.Equal || len(cmp1.Assumes) > 0 {
		return fmt.Errorf("dequant9 1-word renamed SASS comparison: %v", cmp1)
	}

	spec2 := bitopt.E2M1ToBF16Spec(2)
	p2 := bitopt.NVFP4Program2Word()
	if err := bitopt.Check(spec2, p2); err != nil {
		return fmt.Errorf("verify dequant 2-word spec: %w", err)
	}
	p2Verbose := bitopt.NVFP4Program2Word()
	p2Verbose.Moves[0].Sel = 0x20702070 // redundant pre-shift mask before << 2
	p2Verbose.Moves[3].Sel = 0x081c081c // redundant pre-shift mask before << 4
	opt2, res2, err := bitopt.SuperoptimizeSASS(&p2Verbose, spec2, 16)
	if err != nil {
		return fmt.Errorf("superoptimize dequant 2-word SASS: %w", err)
	}
	if res2.Verdict != sassequiv.Equal {
		return fmt.Errorf("dequant 2-word SASS verdict %v: %s", res2.Verdict, res2.Reason)
	}
	if err := bitopt.VerifySASSEquiv(&p2, opt2, spec2); err != nil {
		return fmt.Errorf("verify dequant 2-word SASS equivalence: %w", err)
	}

	sass2Default, err := p2.SASS([2]string{}, nil)
	if err != nil {
		return fmt.Errorf("lower dequant 2-word default SASS: %w", err)
	}
	sass2Renamed, err := p2.SASS([2]string{"R4", "R5"}, []string{"R14", "R15"})
	if err != nil {
		return fmt.Errorf("lower dequant 2-word renamed SASS: %w", err)
	}
	cmp2, err := bitopt.CompareSASS(sass2Default, sass2Renamed)
	if err != nil {
		return fmt.Errorf("compare dequant 2-word renamed SASS: %w", err)
	}
	if cmp2.Verdict != sassequiv.Equal || len(cmp2.Assumes) > 0 {
		return fmt.Errorf("dequant 2-word renamed SASS comparison: %v", cmp2)
	}

	return nil
}

// ProveScaleConversionSASS proves SASS-level symbolic equivalence between the
// mask-before-shift and shift-before-mask formulations of positive-normal E4M3
// to BF16 scale conversion ([ScaleE4M3ToBF16Fast]: ((b & 0x7f) << 4) + 0x7700
// vs. ((b << 4) & 0x7f0) + 0x7700), and checks all 119 positive normal E4M3
// encodings for bit-exactness.
func ProveScaleConversionSASS() (sassequiv.Result, error) {
	maskThenShift := []bitopt.Inst{
		{Op: bitopt.OpAndImm, Dst: "R7", Src0: "R6", Imm: 0x7f},
		{Op: bitopt.OpShlImm, Dst: "R8", Src0: "R7", Imm: 4},
		{Op: bitopt.OpAddImm, Dst: "R9", Src0: "R8", Imm: 0x7700},
	}
	shiftThenMask := []bitopt.Inst{
		{Op: bitopt.OpShlImm, Dst: "R10", Src0: "R6", Imm: 4},
		{Op: bitopt.OpAndImm, Dst: "R11", Src0: "R10", Imm: 0x7f0},
		{Op: bitopt.OpAddImm, Dst: "R12", Src0: "R11", Imm: 0x7700},
	}

	sassA, err := bitopt.FormatSASSListing("scale_e4m3_to_bf16", [2]string{"R4", "R5"}, []string{"R6"}, maskThenShift, []string{"R9"})
	if err != nil {
		return sassequiv.Result{}, fmt.Errorf("format mask-then-shift SASS: %w", err)
	}
	sassB, err := bitopt.FormatSASSListing("scale_e4m3_to_bf16", [2]string{"R4", "R5"}, []string{"R6"}, shiftThenMask, []string{"R12"})
	if err != nil {
		return sassequiv.Result{}, fmt.Errorf("format shift-then-mask SASS: %w", err)
	}

	res, err := bitopt.CompareSASS(sassA, sassB)
	if err != nil {
		return sassequiv.Result{}, err
	}
	if res.Verdict != sassequiv.Equal {
		return res, fmt.Errorf("scale conversion SASS verdict %v: %s", res.Verdict, res.Reason)
	}
	if len(res.Assumes) > 0 {
		return res, fmt.Errorf("scale conversion SASS assumes uninitialized %v", res.Assumes)
	}

	for b := range 256 {
		eb := uint8(b)
		if !IsPositiveNormalE4M3(eb) {
			continue
		}
		gotA := uint16(((uint32(eb) & 0x7f) << 4) + 0x7700)
		gotB := uint16(((uint32(eb) << 4) & 0x7f0) + 0x7700)
		want := ScaleE4M3ToBF16(eb)
		if gotA != want || gotB != want || ScaleE4M3ToBF16Fast(eb) != want {
			return res, fmt.Errorf("scale byte %#02x: got (%#04x, %#04x), want %#04x", eb, gotA, gotB, want)
		}
	}
	return res, nil
}
