package cudalineinfo

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

// unitDecoder checks ParseUnit under name against the package's own printer,
// SourceWithLineDirectives. Only Source and LineMap survive the round trip,
// as ParseUnit documents.
func unitDecoder(name string) cudatest.Decoder[Unit] {
	return cudatest.Decoder[Unit]{
		Decode: func(b []byte) (Unit, error) { return ParseUnit(name, string(b)), nil },
		Encode: func(u Unit) ([]byte, error) { return []byte(u.SourceWithLineDirectives()), nil },
		Equal: func(a, b Unit) bool {
			return a.Name == b.Name && a.Source == b.Source && reflect.DeepEqual(a.LineMap, b.LineMap)
		},
	}
}

// lineinfoTriples are the fixtures compiled together: a unit's annotated
// source, its PTX and its disassembly.
var lineinfoTriples = []struct{ cu, ptx, sass string }{
	{"testdata/forward.cu", "testdata/forward.ptx", "testdata/forward.sass"},
	{"testdata/fused_gate.cu", "testdata/fused_gate.ptx", "testdata/fused_gate.sass"},
	{"testdata/fused_gate.cu", "testdata/fused_gate-fast.ptx", "testdata/fused_gate-fast.sass"},
	{"testdata/fused_layered.cu", "testdata/fused_layered.ptx", "testdata/fused_layered.sass"},
}

func unitName(path string) string {
	return strings.TrimSuffix(filepath.Base(path), ".cu")
}

func FuzzLineInfoParseUnitRoundTrips(f *testing.F) {
	seeds := cudatest.ReadSeeds(f, "testdata/*.cu")
	for _, p := range cudatest.SortedPaths(seeds) {
		f.Add(unitName(p), seeds[p])
	}
	f.Fuzz(func(t *testing.T, name string, annotated []byte) {
		unitDecoder(name).Check(t, annotated)
	})
}

func TestLineInfoParseUnitRegressions(t *testing.T) {
	tests := []struct {
		name, annotated string
		want            map[int]string // generated line to "file:line"
	}{
		// strings.Fields split a quoted name at its space, so the
		// directive was not recognized and stayed in Source.
		{"space in file name", "#line 7 \"my dir/a.go\"\nx\ny", map[int]string{1: "my dir/a.go:7", 2: "my dir/a.go:8"}},
		{"tab separated", "#line\t7\t\"a.go\"\nx", map[int]string{1: "a.go:7"}},
		{"trailing text", "#line 7 \"a.go\" junk\nx", map[int]string{}},
		{"unterminated name", "#line 7 \"a.go\nx", map[int]string{}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			u, _ := unitDecoder("k").Check(t, []byte(tt.annotated))
			got := map[int]string{}
			for n, loc := range u.LineMap {
				got[n] = fmt.Sprintf("%s:%d", loc.File, loc.Line)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("LineMap = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestLineInfoParseUnitMutantControls(t *testing.T) {
	seed, err := os.ReadFile("testdata/fused_gate.cu")
	if err != nil {
		t.Fatal(err)
	}
	d := unitDecoder("fused_gate")
	tests := []struct {
		name   string
		anchor string // text in the seed
		offset int    // byte of anchor to mutate
		b      byte
	}{
		{"line number", `#line 13 "x/cudalineinfo`, 7, '4'},
		{"file name", `#line 13 "x/cudalineinfo`, 11, 'y'},
		{"source", "__global__ void fused_gate(", 16, 'h'},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			i := bytes.Index(seed, []byte(tt.anchor))
			if i < 0 {
				t.Fatalf("seed has no %q", tt.anchor)
			}
			d.CheckMutant(t, seed, i+tt.offset, tt.b)
		})
	}
}

// FuzzLineInfoParseUnitAttributionsResolve attributes arbitrary tiers and
// checks the PTX tier against an independent parser: every site cudaptx
// attributes an instruction to must be a site PTXLines lists. The converse is
// not required, since a .loc that no instruction follows names a site no
// instruction has.
func FuzzLineInfoParseUnitAttributionsResolve(f *testing.F) {
	for _, tr := range lineinfoTriples {
		var b [3][]byte
		for i, p := range []string{tr.cu, tr.ptx, tr.sass} {
			data, err := os.ReadFile(p)
			if err != nil {
				f.Fatal(err)
			}
			b[i] = data
		}
		f.Add(unitName(tr.cu), b[0], b[1], b[2])
	}
	f.Fuzz(func(t *testing.T, name string, cu, ptx, sass []byte) {
		u := ParseUnit(name, string(cu))
		dis, err := cudasass.Parse(bytes.NewReader(sass))
		if err != nil {
			dis = nil
		}
		a := Attribute(u, string(ptx), dis)
		if err := ptxSitesAgree(a); err != nil {
			t.Error(err)
		}
	})
}

// ptxSitesAgree reports a site cudaptx attributes an instruction of a.PTX to
// that a.PTXLines does not list.
func ptxSitesAgree(a *Attribution) error {
	m, err := cudaptx.ParseString(a.PTX)
	if err != nil {
		return nil
	}
	for _, k := range m.Kernels {
		for _, in := range k.Instructions {
			s := Site{File: in.Loc.File, Line: in.Loc.Line}
			if s.Line <= 0 || !a.Unit.ProgramFile(s.File) {
				continue
			}
			if len(a.PTXLines[s]) == 0 {
				return fmt.Errorf("cudaptx places %s line %d (%q) at %s; PTXLines does not list the site", k.Name, in.Line, in.Raw, s)
			}
		}
	}
	return nil
}

// TestPTXSitesAgreeFails shows the differential rejects an attribution whose
// PTX tier lost a site.
func TestPTXSitesAgreeFails(t *testing.T) {
	ptx := ".visible .entry k()\n{\n.loc 1 3 1\nret;\n}\n.file 1 \"a.go\"\n"
	a := Attribute(Unit{Name: "k"}, ptx, nil)
	if err := ptxSitesAgree(a); err != nil {
		t.Fatalf("valid attribution: %v", err)
	}
	a.PTXLines = map[Site][]int{}
	if ptxSitesAgree(a) == nil {
		t.Error("accepted an attribution with an empty PTX tier")
	}
}

// TestPTXSitesRegressions holds PTX on which ptxSites disagreed with cudaptx
// when it read directives with strings.Fields: it split names at spaces, read
// directives inside block comments, matched indices as text, and took "3," for
// a line number. The comment marker case guards the comment stripping.
func TestPTXSitesRegressions(t *testing.T) {
	const body = ".visible .entry k()\n{\n.loc 1 3 1\nret;\n}\n"
	tests := []struct {
		name, ptx string
		want      Site
	}{
		{"space in file name", body + ".file 1 \"my dir/a.go\"\n", Site{"my dir/a.go", 3}},
		{"commented file", body + ".file 1 \"a.go\"\n// .file 1 \"b.go\"\n", Site{"a.go", 3}},
		{"block commented file", body + ".file 1 \"a.go\"\n/*\n.file 1 \"b.go\"\n*/\n", Site{"a.go", 3}},
		{"leading zero index", body + ".file 01 \"a.go\"\n", Site{"a.go", 3}},
		{"comment marker in file name", body + ".file 1 \"x//a.go\"\n", Site{"x//a.go", 3}},
		{"comma after line", ".visible .entry k()\n{\n.loc 1 3, inlined_at 1 9 2\nret;\n}\n.file 1 \"a.go\"\n", Site{"a.go", 3}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := Attribute(Unit{Name: "k"}, tt.ptx, nil)
			if err := ptxSitesAgree(a); err != nil {
				t.Error(err)
			}
			if got := a.PTXLines[tt.want]; !reflect.DeepEqual(got, []int{3}) {
				t.Errorf("PTXLines[%v] = %v, want [3]; all %v", tt.want, got, a.PTXLines)
			}
		})
	}
}

func TestSeedCorpusCoversEveryRecordedFixture(t *testing.T) {
	var seeded []string
	for _, tr := range lineinfoTriples {
		seeded = append(seeded, tr.cu, tr.ptx, tr.sass)
	}
	excluded := map[string]string{
		"testdata/fused_gate.json":    "the Unit fused_gate.cu was printed from; its source is fuzzed as fused_gate.cu",
		"testdata/fused_layered.json": "the Unit fused_layered.cu was printed from; its source is fuzzed as fused_layered.cu",
		"testdata/gen.go":             "the fixture generator, not an input",
		"testdata/wiring/kernels.go":  "Go source of the program the units are lowered from",
		"testdata/wiring/main.go":     "Go source of the program the units are lowered from",
	}
	for k, v := range excluded {
		delete(excluded, k)
		excluded[filepath.FromSlash(k)] = v
	}
	cudatest.CheckSeedCoverage(t, "testdata", seeded, excluded)
}
