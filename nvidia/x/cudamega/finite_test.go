package cudamega_test

import (
	"github.com/tmc/lib/nvidia/x/cudamega"
	"strings"
	"testing"
)

func TestFiniteWorkerSource(t *testing.T) {
	s := cudamega.FiniteWorkerSource()
	for _, x := range []string{"finite_worker", "abort", "__syncthreads", "unsigned long long epoch"} {
		if !strings.Contains(s, x) {
			t.Fatalf("missing %q", x)
		}
	}
}
func TestOrdinaryReference(t *testing.T) {
	if got := cudamega.OrdinaryReference([]uint32{1, 2}); got[0] != 2 || got[1] != 3 {
		t.Fatal(got)
	}
}
