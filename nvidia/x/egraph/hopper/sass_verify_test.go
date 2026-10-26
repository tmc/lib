package hopper

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph/bitopt"
	"github.com/tmc/lib/nvidia/x/sassequiv"
)

func TestVerifyDequantProgramsSASS(t *testing.T) {
	if err := VerifyDequantProgramsSASS(); err != nil {
		t.Fatalf("VerifyDequantProgramsSASS: %v", err)
	}
}

func TestProveScaleConversionSASS(t *testing.T) {
	res, err := ProveScaleConversionSASS()
	if err != nil {
		t.Fatalf("ProveScaleConversionSASS: %v", err)
	}
	if res.Verdict != sassequiv.Equal {
		t.Fatalf("ProveScaleConversionSASS verdict = %v (%s), want Equal", res.Verdict, res.Reason)
	}
	if len(res.Assumes) > 0 {
		t.Fatalf("ProveScaleConversionSASS assumes = %v, want empty", res.Assumes)
	}

	// Negative control: wrong rebias constant (0x7800 instead of 0x7700) must compare Different.
	good := []bitopt.Inst{
		{Op: bitopt.OpAndImm, Dst: "R7", Src0: "R6", Imm: 0x7f},
		{Op: bitopt.OpShlImm, Dst: "R8", Src0: "R7", Imm: 4},
		{Op: bitopt.OpAddImm, Dst: "R9", Src0: "R8", Imm: 0x7700},
	}
	badRebias := []bitopt.Inst{
		{Op: bitopt.OpShlImm, Dst: "R10", Src0: "R6", Imm: 4},
		{Op: bitopt.OpAndImm, Dst: "R11", Src0: "R10", Imm: 0x7f0},
		{Op: bitopt.OpAddImm, Dst: "R12", Src0: "R11", Imm: 0x7800},
	}
	sassGood, err := bitopt.FormatSASSListing("f", [2]string{"R4", "R5"}, []string{"R6"}, good, []string{"R9"})
	if err != nil {
		t.Fatal(err)
	}
	sassBad, err := bitopt.FormatSASSListing("f", [2]string{"R4", "R5"}, []string{"R6"}, badRebias, []string{"R12"})
	if err != nil {
		t.Fatal(err)
	}
	neg, err := bitopt.CompareSASS(sassGood, sassBad)
	if err != nil {
		t.Fatal(err)
	}
	if neg.Verdict != sassequiv.Different {
		t.Fatalf("wrong rebias constant verdict = %v (%s), want Different", neg.Verdict, neg.Reason)
	}
}
