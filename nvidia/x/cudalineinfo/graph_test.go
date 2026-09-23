package cudalineinfo

import (
	"bytes"
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

// The layered fixture builds its lines inside two helpers, so its call
// graph reaches the kernel through the stack the compiler recorded. Its
// division and square root each fall back to a subroutine.
func TestEdges(t *testing.T) {
	a := loadFixture(t, "fused_layered", "")
	divSlowpath, _ := subroutine(t, a, "fused_layered", divSlowpath)
	sqrtSlowpath, _ := subroutine(t, a, "fused_layered", sqrtSlowpath)
	got := map[Edge]bool{}
	for _, e := range a.Edges() {
		got[e] = true
	}
	for _, want := range []Edge{
		{"main.main", "main.run"},
		{"main.run", "main.attribute"},
		{"main.attribute", "main.layered"},
		{"main.layered", "main.activate"},
		{"main.activate", "kernels.go:45"},
		{"main.layered", "kernels.go:38"},
		{"kernels.go:45", divSlowpath},
		{"kernels.go:45", "fused_layered"},
		{"kernels.go:38", "fused_layered"},
	} {
		if !got[want] {
			t.Errorf("missing edge %s -> %s", want.From, want.To)
		}
	}
	for e := range got {
		if e.From == "kernels.go:38" && (e.To == divSlowpath || e.To == sqrtSlowpath) {
			t.Error("the input line calls a slowpath")
		}
		if e.From == "fused_layered" && (e.To == divSlowpath || e.To == sqrtSlowpath) {
			t.Error("the kernel calls a slowpath from an instruction naming no line")
		}
		if e.From == divSlowpath || e.From == sqrtSlowpath {
			t.Errorf("a slowpath's RET became an edge to %s", e.To)
		}
	}

	var buf bytes.Buffer
	if err := WriteDigraph(&buf, a); err != nil {
		t.Fatal(err)
	}
	for _, line := range strings.Split(strings.TrimSpace(buf.String()), "\n") {
		if strings.HasPrefix(line, "kernels.go:45 ") && line != "kernels.go:45 "+divSlowpath+" fused_layered" {
			t.Errorf("digraph line for the activate site = %q", line)
		}
	}
	buf.Reset()
	if err := WriteDot(&buf, a); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), `"kernels.go:45" -> "`+divSlowpath+`";`) || !strings.Contains(buf.String(), fmt.Sprintf(`"kernels.go:45" [label="kernels.go:45\n%d SASS"];`, len(a.SASS[Site{kernelsFile, 45}]))) {
		t.Errorf("dot output:\n%s", buf.String())
	}
}

func TestCallTarget(t *testing.T) {
	for _, tt := range []struct{ opcode, operands, want string }{
		{"CALL.REL.NOINC", "`(" + divSlowpath + ")", divSlowpath},
		{"RET.REL.NODEC", "R14 `(fused_layered)", ""},
		{"BRA", "`(.L_x_1)", ""},
		{"CALL.REL.NOINC", "`(.L_x_1)", ""},
		{"FMUL", "R3, R0", ""},
	} {
		in := cudasass.Instruction{Opcode: tt.opcode, Operands: tt.operands}
		if got := callTarget(in); got != tt.want {
			t.Errorf("callTarget(%s %s) = %q, want %q", tt.opcode, tt.operands, got, tt.want)
		}
	}
}

func TestParseUnit(t *testing.T) {
	u := fixtureUnit(t, "fused_gate")
	got := ParseUnit(u.Name, u.SourceWithLineDirectives())
	if got.Source != u.Source {
		t.Errorf("source did not round-trip:\n%s\nwant:\n%s", got.Source, u.Source)
	}
	if len(got.LineMap) != len(u.LineMap) {
		t.Fatalf("got %d mapped lines, want %d: %v", len(got.LineMap), len(u.LineMap), got.LineMap)
	}
	for n, loc := range u.LineMap {
		g := got.LineMap[n]
		if g.File != loc.File || g.Line != loc.Line {
			t.Errorf("line %d = %s:%d, want %s:%d", n, g.File, g.Line, loc.File, loc.Line)
		}
	}
	if len(ParseUnit("k", "no directives\nat all").LineMap) != 0 {
		t.Error("mapped lines without directives")
	}
}
