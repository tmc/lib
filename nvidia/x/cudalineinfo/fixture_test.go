package cudalineinfo

import (
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

// The fixtures are the fused_gate and fused_layered kernels of the
// source-wiring example, exported by testdata/gen.go. gate builds both
// inputs and the multiply in one function; layered builds its operations
// inside two helpers, so its line map is the one that carries callers worth
// rolling up.
const (
	kernelsFile = "x/examples/source-wiring/kernels.go"
	mainFile    = "x/examples/source-wiring/main.go"
)

// fixtureUnit reads a unit of testdata. Its functions and callers are the
// ones cudafuse recorded while the example ran, not values a test invented.
func fixtureUnit(t *testing.T, name string) Unit {
	t.Helper()
	data, err := os.ReadFile("testdata/" + name + ".json")
	if err != nil {
		t.Fatal(err)
	}
	var u Unit
	if err := json.Unmarshal(data, &u); err != nil {
		t.Fatal(err)
	}
	return u
}

// loadFixture attributes a unit's artifacts. suffix selects a build other
// than the default one, such as "-fast".
func loadFixture(t *testing.T, name, suffix string) *Attribution {
	t.Helper()
	ptx, err := os.ReadFile("testdata/" + name + suffix + ".ptx")
	if err != nil {
		t.Fatal(err)
	}
	sass, err := os.ReadFile("testdata/" + name + suffix + ".sass")
	if err != nil {
		t.Fatal(err)
	}
	dis, err := cudasass.ParseString(string(sass))
	if err != nil {
		t.Fatal(err)
	}
	return Attribute(fixtureUnit(t, name), string(ptx), dis)
}

// offsetOn returns the offset of the first instruction of function that the
// attribution files under site. Tests name an instruction by the line it
// belongs to, so regenerating the fixtures does not move it.
func offsetOn(t *testing.T, a *Attribution, function string, site Site) uint64 {
	t.Helper()
	for _, in := range a.Disassembly.Function(function).Instructions {
		if s, ok := a.SiteAt(function, in.Address); ok && s == site {
			return in.Address
		}
	}
	t.Fatalf("%s has no instruction on %v", function, site)
	return 0
}

// subroutine returns the name and entry offset of the subroutine of
// function whose symbol ends in suffix. The compiler numbers its internal
// symbols, so the full name depends on what else the kernel calls.
func subroutine(t *testing.T, a *Attribution, function, suffix string) (string, uint64) {
	t.Helper()
	for _, s := range a.Disassembly.Function(function).Symbols {
		if strings.HasSuffix(s.Name, suffix) {
			return s.Name, s.Address
		}
	}
	t.Fatalf("%s has no subroutine ending in %s", function, suffix)
	return "", 0
}

// divSlowpath and sqrtSlowpath end the names of the subroutines a kernel's
// IEEE division and square root fall back to.
const (
	divSlowpath  = "_$__cuda_sm3x_div_rn_noftz_f32_slowpath"
	sqrtSlowpath = "_$__cuda_sm20_sqrt_rn_f32_slowpath"
)
