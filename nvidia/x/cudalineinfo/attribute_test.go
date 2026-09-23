package cudalineinfo

import (
	"bytes"
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
)

func TestAttribute(t *testing.T) {
	a := loadFixture(t, "fused_gate", "")
	if a.Instructions != 464 || a.Attributed != 162 {
		t.Errorf("attributed %d of %d, want 162 of 464", a.Attributed, a.Instructions)
	}
	tests := []struct {
		site      Site
		cuda      []int
		ptx, sass int
	}{
		{Site{kernelsFile, 12}, []int{4, 11}, 5, 14},
		{Site{kernelsFile, 13}, []int{5, 12}, 5, 12},
		{Site{kernelsFile, 14}, []int{13}, 5, 117},
		{Site{kernelsFile, 15}, []int{6, 14, 15}, 5, 19},
	}
	rows := a.Rows()
	if len(rows) != len(tests) {
		t.Fatalf("got %d rows, want %d", len(rows), len(tests))
	}
	for i, tt := range tests {
		r := rows[i]
		if r.Site != tt.site || r.PTX != tt.ptx || len(r.SASS) != tt.sass || !equalInts(r.CUDA, tt.cuda) {
			t.Errorf("row %d = %v cuda=%v ptx=%d sass=%d, want %v cuda=%v ptx=%d sass=%d",
				i, r.Site, r.CUDA, r.PTX, len(r.SASS), tt.site, tt.cuda, tt.ptx, tt.sass)
		}
	}

	// The remainder must account for every instruction not attributed, and
	// the division slowpath must be counted as a subroutine rather than as
	// generated code, which is what it would look like from the line
	// directives alone: its instructions inherit the last one before its
	// label. The 64-bit division is the one the loop's trip count needs.
	div, _ := subroutine(t, a, "fused_gate", divSlowpath)
	divU64, _ := subroutine(t, a, "fused_gate", "_$__cuda_sm20_div_u64")
	want := Remainder{Generated: 125, Subroutines: map[string]int{div: 115, divU64: 62}}
	if a.Remainder.Total() != a.Instructions-a.Attributed {
		t.Errorf("remainder totals %d, want %d", a.Remainder.Total(), a.Instructions-a.Attributed)
	}
	if a.Remainder.String() != want.String() {
		t.Errorf("remainder = %s, want %s", a.Remainder, want)
	}
}

// TestAttributeSilu checks the instructions themselves, not only the counts.
// Silu is x / (1 + expf(-x)), so the line that holds it owns the
// exponential, the reciprocal, and the call into the division slowpath,
// while the multiply and the store belong to the line below it.
func TestAttributeSilu(t *testing.T) {
	a := loadFixture(t, "fused_gate", "")
	ops := func(line int) string {
		var out []string
		for _, in := range a.SASS[Site{kernelsFile, line}] {
			out = append(out, in.Opcode)
		}
		return strings.Join(out, " ")
	}
	for _, tt := range []struct {
		line int
		want []string
	}{
		{14, []string{"MUFU.EX2", "MUFU.RCP", "FCHK", "CALL.REL.NOINC"}},
		{15, []string{"FMUL", "STG.E"}},
	} {
		joined := ops(tt.line)
		for _, want := range tt.want {
			if !strings.Contains(joined, want) {
				t.Errorf("line %d lacks %s; has %s", tt.line, want, joined)
			}
		}
	}
}

func TestSiteAt(t *testing.T) {
	a := loadFixture(t, "fused_gate", "")
	o := offsets(t, a)
	tests := []struct {
		offset uint64
		want   Site
		ok     bool
	}{
		{o.prologue, Site{}, false}, // names the generated file
		{o.gate, Site{kernelsFile, 12}, true},
		{o.silu, Site{kernelsFile, 14}, true},
		{o.slowpath, Site{}, false},
		{0xfff0, Site{}, false}, // no such instruction
	}
	for _, tt := range tests {
		got, ok := a.SiteAt("fused_gate", tt.offset)
		if ok != tt.ok || got != tt.want {
			t.Errorf("SiteAt(%#x) = %v, %v; want %v, %v", tt.offset, got, ok, tt.want, tt.ok)
		}
	}
	if _, ok := a.SiteAt("no_such_kernel", 0); ok {
		t.Error("SiteAt found a site in a kernel that does not exist")
	}
}

func TestAttributeWithoutArtifacts(t *testing.T) {
	a := Attribute(fixtureUnit(t, "fused_gate"), "", nil)
	if len(a.CUDA) != 4 || len(a.PTXLines) != 0 || len(a.SASS) != 0 || a.Instructions != 0 {
		t.Errorf("attribution from the unit alone = cuda %d ptx %d sass %d insts %d", len(a.CUDA), len(a.PTXLines), len(a.SASS), a.Instructions)
	}
}

func TestPTXSites(t *testing.T) {
	const ptx = `
	.loc	1 3 0
	.loc	2 17 9
	.loc	2 17 9
	.loc	1 0 5
	.loc	3 5 0
	.file	1 "k.cu"
	.file	2 "kernels.go"
`
	got := ptxSites(ptx)
	want := map[Site][]int{{"k.cu", 3}: {2}, {"kernels.go", 17}: {3, 4}}
	if len(got) != len(want) {
		t.Fatalf("ptxSites = %v, want %v", got, want)
	}
	for s, lines := range want {
		if !equalInts(got[s], lines) {
			t.Errorf("ptxSites[%v] = %v, want %v", s, got[s], lines)
		}
	}
}

func TestWriteTable(t *testing.T) {
	a := loadFixture(t, "fused_gate", "")
	var buf bytes.Buffer
	if err := WriteTable(&buf, a); err != nil {
		t.Fatal(err)
	}
	out := buf.String()
	for _, want := range []string{
		fmt.Sprintf("fused_gate: %d of %d SASS instructions name a program line", a.Attributed, a.Instructions),
		"kernels.go:14",
		"remainder: " + a.Remainder.String(),
	} {
		if !strings.Contains(out, want) {
			t.Errorf("table lacks %q:\n%s", want, out)
		}
	}
}

func TestUnitSourceWithLineDirectives(t *testing.T) {
	u := Unit{
		Name:   "k",
		Source: "a\nb\nc\nd",
		LineMap: map[int]cudair.SourceLocation{
			2: {File: "m.go", Line: 10},
			3: {File: "m.go", Line: 11},
		},
	}
	want := "a\n#line 10 \"m.go\"\nb\n#line 11 \"m.go\"\nc\n#line 4 \"k.cu\"\nd"
	if got := u.SourceWithLineDirectives(); got != want {
		t.Errorf("got:\n%s\nwant:\n%s", got, want)
	}
	if got := (Unit{Name: "k", Source: "a\nb"}).SourceWithLineDirectives(); got != "a\nb" {
		t.Errorf("unit without a line map changed its source: %q", got)
	}
}

func TestUnitLocation(t *testing.T) {
	u := fixtureUnit(t, "fused_gate")
	loc, ok := u.Location(Site{kernelsFile, 14})
	if !ok || loc.Function != "main.gate" || loc.Line != 14 {
		t.Errorf("Location = %+v, %v", loc, ok)
	}
	if _, ok := u.Location(Site{kernelsFile, 1}); ok {
		t.Error("found a location for a site no line maps to")
	}
}

func TestIsUserFile(t *testing.T) {
	u := Unit{Name: "k"}
	for _, tt := range []struct {
		file string
		want bool
	}{
		{"kernels.go", true},
		{"/abs/path/kernels.go", true},
		{"model.py", true},
		{"k.cu", false},
		{"/abs/k.cu", false},
		{"cuda_fp16.hpp", false},
		{"sm_30_intrinsics.h", false},
		{".nv_debug_ptx_txt", false},
		{"", false},
	} {
		if got := u.ProgramFile(tt.file); got != tt.want {
			t.Errorf("ProgramFile(%q) = %v, want %v", tt.file, got, tt.want)
		}
	}
}

func equalInts(a, b []int) bool {
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
