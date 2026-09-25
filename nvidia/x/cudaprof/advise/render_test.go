package advise

import (
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// stallCapNoSASS builds a PC-sampled capture with no disassembly and no opcode
// label, so nothing in the capture can name the stalling instruction. The
// advisor must say so rather than rendering an empty opcode into the sentence.
func stallCapNoSASS() *cudacapture.Capture {
	return &cudacapture.Capture{
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{
				{Type: "stall_scoreboard", Unit: "count"},
				{Type: "stall_mem", Unit: "count"},
				{Type: "stall_branch", Unit: "count"},
				{Type: "stall_wait", Unit: "count"},
			},
			Location: []*profile.Location{{ID: 1, Address: 0x4874d0}},
			Sample: []*profile.Sample{
				{
					Location: []*profile.Location{{ID: 1, Address: 0x4874d0}},
					Value:    []int64{900, 40, 30, 30},
					Label: map[string][]string{
						"kernel": {"ds4_repack_iq2_xxs_aligned"},
					},
				},
			},
		},
	}
}

// TestFindingNamesItsInstruction is the B5 control. A finding whose opcode is
// unavailable used to render "of stalls on  at PC 0x4874d0" -- two spaces where
// an instruction belongs. The failure mode is the rendered string, so this
// asserts on the string.
func TestFindingNamesItsInstruction(t *testing.T) {
	rep, err := Analyze(stallCapNoSASS())
	if err != nil {
		t.Fatalf("Analyze failed: %v", err)
	}

	var stall *Item
	for i := range rep.Items {
		if strings.Contains(rep.Items[i].Finding, "at PC 0x") {
			stall = &rep.Items[i]
			break
		}
	}
	if stall == nil {
		t.Fatal("expected a PC-located stall finding, found none; control cannot fire")
	}

	if strings.Contains(stall.Finding, "  ") {
		t.Errorf("finding contains a double space where an instruction belongs: %q", stall.Finding)
	}
	if strings.Contains(stall.Finding, " on  at ") || strings.Contains(stall.Finding, " on at ") {
		t.Errorf("finding has an empty opcode slot: %q", stall.Finding)
	}
	if !strings.Contains(stall.Finding, "an unnamed instruction") {
		t.Errorf("finding must say the instruction is unnamed, got: %q", stall.Finding)
	}
}

// TestFindingNamesKnownOpcode is the other half of the control: when the
// opcode IS recoverable the real name must appear, not the placeholder. Without
// this half the fix could pass by always printing "an unnamed instruction".
func TestFindingNamesKnownOpcode(t *testing.T) {
	const sass = `
	.section	.text.stally,"ax",@progbits
stally:
.text.stally:
        /*0000*/                   MOV R1, c[0x0][0x28] ;
        /*0010*/                   LDG.E.128 R0, [R2] ;
        /*0020*/                   EXIT ;
`
	cap := &cudacapture.Capture{
		Code:  cudacapture.CodeFiles{SASS: "kernels.sass"},
		Files: map[string][]byte{"kernels.sass": []byte(sass)},
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{
				{Type: "stall_scoreboard", Unit: "count"},
				{Type: "stall_mem", Unit: "count"},
			},
			Location: []*profile.Location{{ID: 1, Address: 0x0010}},
			Sample: []*profile.Sample{
				{
					Location: []*profile.Location{{ID: 1, Address: 0x0010}},
					Value:    []int64{900, 100},
					Label:    map[string][]string{"kernel": {"stally"}},
				},
			},
		},
	}

	rep, err := Analyze(cap)
	if err != nil {
		t.Fatalf("Analyze failed: %v", err)
	}
	var found bool
	for _, it := range rep.Items {
		if strings.Contains(it.Finding, "at PC 0x") && strings.Contains(it.Finding, "LDG") {
			found = true
		}
		if strings.Contains(it.Finding, "at PC 0x") && strings.Contains(it.Finding, "an unnamed instruction") {
			t.Errorf("opcode was recoverable from SASS but finding says unnamed: %q", it.Finding)
		}
	}
	if !found {
		t.Error("expected a stall finding naming the LDG instruction; control cannot fire")
	}
}
