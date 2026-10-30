package cudasass_test

import (
	"fmt"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

func TestParseFlaglessDirectiveRegression(t *testing.T) {
	tests := []struct {
		name      string
		directive string
		want      string
	}{
		{"mangled", ".section .text._Z1kPf", "_Z1kPf"},
		{"plain", ".section .text.compute", "compute"},
		{"literal comment marker", `.section .text.f<"//">()`, `f<"//">()`},
		{"comment with flags", `.section .text.k // ,"ax",@progbits`, "k"},
		{"comment with open literal", `.section .text.k // "`, "k"},
		{"demangled", "\t.section\t.text." + overloadF, overloadF},
		{"qualified", "\t.section\t.text." + qualified, qualified},
		{"template", "\t.section\t.text.k<1, 2>(int)", "k<1, 2>(int)"},
		{"unmatched closer", "\t.section\t.text.operator>(a, b)", "operator>(a, b)"},
		{"life-range columns", "\t.section\t.text." + overloadF + "        // |       |        |", overloadF},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			instr := "        /*0000*/                   EXIT ;\n"
			for _, listing := range []string{
				tt.directive + "\n" + instr,
				tt.directive + "\n.text." + tt.want + ":\n" + instr,
			} {
				d, err := cudasass.ParseString(listing)
				if err != nil {
					t.Fatalf("ParseString(%q): %v", listing, err)
				}
				var got []string
				for _, fn := range d.Functions {
					got = append(got, fmt.Sprintf("%s/%d/%s", fn.Name, len(fn.Instructions), fn.ELFSection))
				}
				if want := []string{tt.want + "/1/.text." + tt.want}; !slices.Equal(got, want) {
					t.Errorf("ParseString(%q): name/instructions/section = %q, want %q", listing, got, want)
				}
				if n := len(d.Sections()); n != 1 {
					t.Errorf("ParseString(%q): %d sections, want 1", listing, n)
				}
			}
		})
	}
}
