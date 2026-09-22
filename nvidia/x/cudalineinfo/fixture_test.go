package cudalineinfo

import (
	"encoding/json"
	"os"
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
