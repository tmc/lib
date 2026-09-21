package cudalineinfo

import (
	"bytes"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
)

// TestRollup charges the silu line's instructions to the helper that built
// it and to the caller of that helper, and leaves the input lines alone.
func TestRollup(t *testing.T) {
	a := loadFixture(t, "")
	helper := []cudair.Frame{
		{File: "kernels.go", Line: 40, Function: "main.gate"},
		{File: "main.go", Line: 9, Function: "main.main"},
	}
	for line, loc := range a.Unit.LineMap {
		if loc.Line == 19 {
			loc.Function = "main.silu"
			loc.Callers = helper
			a.Unit.LineMap[line] = loc
		}
	}

	got := map[string]Rollup{}
	for _, r := range a.Rollup() {
		got[r.Site.String()] = r
	}
	silu := len(a.SASS[Site{File: "kernels.go", Line: 19}])
	for _, tt := range []struct {
		site      string
		function  string
		flat, cum int
	}{
		{"kernels.go:17", "main.main", 5, 5},
		{"kernels.go:18", "main.main", 3, 3},
		{"kernels.go:19", "main.silu", silu, silu},
		{"kernels.go:40", "main.gate", 0, silu},
		{"main.go:9", "main.main", 0, silu},
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
	if len(got) != 5 {
		t.Errorf("got %d rollups, want 5", len(got))
	}

	var buf bytes.Buffer
	if err := WriteRollup(&buf, a); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), "kernels.go:40  main.gate") {
		t.Errorf("rollup table lacks the caller row:\n%s", buf.String())
	}
}

// TestRollupRecursion counts a frame that appears twice in one stack once.
func TestRollupRecursion(t *testing.T) {
	a := loadFixture(t, "")
	self := cudair.Frame{File: "kernels.go", Line: 19, Function: "main.main"}
	for line, loc := range a.Unit.LineMap {
		if loc.Line == 19 {
			loc.Callers = []cudair.Frame{self, self}
			a.Unit.LineMap[line] = loc
		}
	}
	for _, r := range a.Rollup() {
		if r.Site.Line == 19 && r.Cum != r.Flat {
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
