package advise

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

func TestElectedBulkRejectsDegenerateAndUnsupportedCFG(t *testing.T) {
	needle := "PLOP3.LUT P2, PT, PT, PT, PT, 0x8, 0x0"
	for _, opcode := range []string{"JMX", "JMXU", "BRX", "BRXU"} {
		t.Run(opcode, func(t *testing.T) {
			d, err := cudasass.ParseString(strings.ReplaceAll(electedBulkWitness, needle, opcode+" R30"))
			if err != nil {
				t.Fatal(err)
			}
			cfg, err := cudasass.NewCFG(d.Functions[0])
			if err != nil {
				t.Fatal(err)
			}
			if !cfg.IsDegenerate {
				t.Fatal("negative fixture did not establish indirect control")
			}
			if got := len(electedBulkLoops(d.Functions[0])); got != 0 {
				t.Fatalf("accepted %d loops through indirect transfer", got)
			}
		})
	}
	for _, tt := range []struct{ name, text string }{
		{"unknown body opcode", strings.ReplaceAll(electedBulkWitness, needle, "UNSUPPORTED_CONTROL R30")},
		{"unresolved direct target", strings.ReplaceAll(electedBulkWitness, needle, "BRA 0xdeadbeef")},
		{"indirect outside loop", electedBulkWitness + "\n/*9da0*/ JMX R30 ;\n"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			d, err := cudasass.ParseString(tt.text)
			if err != nil {
				t.Fatal(err)
			}
			if got := len(electedBulkLoops(d.Functions[0])); got != 0 {
				t.Fatalf("accepted %d unsupported control loops", got)
			}
		})
	}
	if got := electedBulkLoops(nil); len(got) != 0 {
		t.Fatal("accepted missing CFG")
	}
}
