package cudalineinfo

import (
	"bytes"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

const slowpath = "$__internal_0_$__cuda_sm3x_div_rn_noftz_f32_slowpath"

func TestEdges(t *testing.T) {
	a := loadFixture(t, "")
	for line, loc := range a.Unit.LineMap {
		if loc.Line == 19 {
			loc.Function = "main.silu"
			loc.Callers = []cudair.Frame{
				{File: "kernels.go", Line: 40, Function: "main.gate"},
				{File: "main.go", Line: 9, Function: "main.main"},
			}
			a.Unit.LineMap[line] = loc
		}
	}
	got := map[Edge]bool{}
	for _, e := range a.Edges() {
		got[e] = true
	}
	for _, want := range []Edge{
		{"main.main", "main.gate"},
		{"main.gate", "main.silu"},
		{"main.silu", "kernels.go:19"},
		{"main.main", "kernels.go:17"},
		{"kernels.go:19", "fused_gate"},
		{"kernels.go:19", slowpath},
		{"kernels.go:17", "fused_gate"},
	} {
		if !got[want] {
			t.Errorf("missing edge %s -> %s", want.From, want.To)
		}
	}
	for e := range got {
		if e.From == "kernels.go:17" && e.To == slowpath {
			t.Error("the input line calls the slowpath")
		}
		if e.From == "fused_gate" && e.To == slowpath {
			t.Error("the kernel calls the slowpath from an instruction naming no line")
		}
		if e.From == slowpath {
			t.Errorf("the slowpath's RET became an edge to %s", e.To)
		}
	}

	var buf bytes.Buffer
	if err := WriteDigraph(&buf, a); err != nil {
		t.Fatal(err)
	}
	for _, line := range strings.Split(strings.TrimSpace(buf.String()), "\n") {
		if strings.HasPrefix(line, "kernels.go:19 ") && line != "kernels.go:19 "+slowpath+" fused_gate" {
			t.Errorf("digraph line for the silu site = %q", line)
		}
	}
	buf.Reset()
	if err := WriteDot(&buf, a); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), `"kernels.go:19" -> "`+slowpath+`";`) || !strings.Contains(buf.String(), "118 SASS") {
		t.Errorf("dot output:\n%s", buf.String())
	}
}

func TestCallTarget(t *testing.T) {
	for _, tt := range []struct{ opcode, operands, want string }{
		{"CALL.REL.NOINC", "`($__internal_0_$__cuda_sm3x_div_rn_noftz_f32_slowpath)", "$__internal_0_$__cuda_sm3x_div_rn_noftz_f32_slowpath"},
		{"RET.REL.NODEC", "R14 `(fused_gate)", ""},
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
	u := fixtureUnit()
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
