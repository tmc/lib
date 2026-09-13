package cudalowering

import (
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

// The fixtures in testdata are byte-exact output of CUDA 13.0 nvcc, ptxas,
// nvdisasm and cu++filt over testdata/lowering.cu; see testdata/gen.sh. Neither
// side of a comparison is written here, so a test that expects a divergence is
// a claim about what ptxas did, not about what this package was told.

func readPTX(t *testing.T, name string) *cudaptx.Module {
	t.Helper()
	f, err := os.Open("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	m, err := cudaptx.Parse(f)
	if err != nil {
		t.Fatalf("parse %s: %v", name, err)
	}
	return m
}

func readSASS(t *testing.T, name string) *cudasass.Disassembly {
	t.Helper()
	f, err := os.Open("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	d, err := cudasass.Parse(f)
	if err != nil {
		t.Fatalf("parse %s: %v", name, err)
	}
	return d
}

// divergences reports the String form of every divergence, for comparison with
// a literal list. Order follows kernel order in the PTX, which is stable.
func divergences(r *Report) []string {
	out := make([]string, len(r.Divergences))
	for i, d := range r.Divergences {
		out[i] = d.String()
	}
	return out
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
