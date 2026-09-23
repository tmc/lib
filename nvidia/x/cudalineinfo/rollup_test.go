package cudalineinfo

import (
	"bytes"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
)

// TestRollup charges the instructions of the lines inside the two helpers to
// the calls that reached them and to the frames above those, and leaves the
// input lines of the kernel body alone. The callers are the ones the
// compiler path recorded in the fixture, not values set here.
func TestRollup(t *testing.T) {
	a := loadFixture(t, "fused_layered", "")
	got := map[string]Rollup{}
	for _, r := range a.Rollup() {
		got[r.Site.String()] = r
	}
	var (
		x          = len(a.SASS[Site{File: kernelsFile, Line: 38}])
		w          = len(a.SASS[Site{File: kernelsFile, Line: 39}])
		activate   = len(a.SASS[Site{File: kernelsFile, Line: 45}])
		sqrt       = len(a.SASS[Site{File: kernelsFile, Line: 49}])
		div        = len(a.SASS[Site{File: kernelsFile, Line: 50}])
		attributed = a.Attributed
	)
	for _, tt := range []struct {
		site      string
		function  string
		flat, cum int
	}{
		{"kernels.go:38", "main.layered", x, x},
		{"kernels.go:39", "main.layered", w, w},
		{"kernels.go:40", "main.layered", 0, activate},
		{"kernels.go:41", "main.layered", 0, sqrt + div},
		{"kernels.go:45", "main.activate", activate, activate},
		{"kernels.go:49", "main.normalize", sqrt, sqrt},
		{"kernels.go:50", "main.normalize", div, div},
		{"main.go:47", "main.attribute", 0, attributed},
		{"main.go:37", "main.run", 0, attributed},
		{"main.go:30", "main.main", 0, attributed},
	} {
		r, ok := got[tt.site]
		if !ok {
			t.Errorf("no rollup for %s", tt.site)
			continue
		}
		if r.Function != tt.function || r.Flat != tt.flat || r.Cum != tt.cum {
			t.Errorf("%s = %s flat %d cum %d, want %s flat %d cum %d", tt.site, r.Function, r.Flat, r.Cum, tt.function, tt.flat, tt.cum)
		}
	}
	if len(got) != 10 {
		t.Errorf("got %d rollups, want 10", len(got))
	}

	var buf bytes.Buffer
	if err := WriteRollup(&buf, a); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), "kernels.go:40  main.layered") {
		t.Errorf("rollup table lacks the caller row:\n%s", buf.String())
	}
}

// TestRollupRecursion counts a frame that appears twice in one stack once.
// The callers are set here because no fixture has a recursive stack: the
// example's kernels are built by straight-line calls.
func TestRollupRecursion(t *testing.T) {
	a := loadFixture(t, "fused_gate", "")
	self := cudair.Frame{File: kernelsFile, Line: 14, Function: "main.gate"}
	for line, loc := range a.Unit.LineMap {
		if loc.Line == 14 {
			loc.Callers = []cudair.Frame{self, self}
			a.Unit.LineMap[line] = loc
		}
	}
	for _, r := range a.Rollup() {
		if r.Site.Line == 14 && r.Cum != r.Flat {
			t.Errorf("recursive frame counted twice: flat %d cum %d", r.Flat, r.Cum)
		}
	}
}

func TestShortFunction(t *testing.T) {
	for _, tt := range []struct{ in, want string }{
		{"main.main", "main.main"},
		{"github.com/tmc/lib/nvidia/x/cudatensor.(*Tensor).Silu", "cudatensor.(*Tensor).Silu"},
		{"", ""},
	} {
		if got := shortFunction(tt.in); got != tt.want {
			t.Errorf("shortFunction(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}
