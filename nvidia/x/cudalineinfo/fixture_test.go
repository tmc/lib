package cudalineinfo

import (
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

// fixtureUnit is the fused_gate kernel of testdata: x.Silu().Mul(w) over two
// inputs declared on kernels.go lines 17 and 18 and combined on line 19. The
// line map is the one cudafuse recorded when the fixtures were generated.
func fixtureUnit() Unit {
	src, err := os.ReadFile("testdata/fused_gate.cu")
	if err != nil {
		panic(err)
	}
	at := func(line int) cudair.SourceLocation {
		return cudair.SourceLocation{File: "kernels.go", Line: line, Function: "main.main"}
	}
	return Unit{
		Name:   "fused_gate",
		Source: stripDirectives(string(src)),
		LineMap: map[int]cudair.SourceLocation{
			4: at(17), 5: at(18), 6: at(19),
			10: at(17), 11: at(18), 12: at(19), 13: at(19), 14: at(19),
		},
	}
}

func loadFixture(t *testing.T, suffix string) *Attribution {
	t.Helper()
	ptx, err := os.ReadFile("testdata/fused_gate" + suffix + ".ptx")
	if err != nil {
		t.Fatal(err)
	}
	sass, err := os.ReadFile("testdata/fused_gate" + suffix + ".sass")
	if err != nil {
		t.Fatal(err)
	}
	dis, err := cudasass.ParseString(string(sass))
	if err != nil {
		t.Fatal(err)
	}
	return Attribute(fixtureUnit(), string(ptx), dis)
}
