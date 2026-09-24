package cudasass

import (
	"bytes"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatest"
)

// sassSeeds are the nvdisasm listings Parse is fuzzed from.
var sassSeeds = []string{"testdata/*.sass", "testdata/roundtrip/*.sass"}

// cubinSeeds are the hex-encoded CUDA ELF images ReadToolchains is fuzzed from.
var cubinSeeds = []string{"testdata/*.cubin.hex", "testdata/roundtrip/*.cubin.hex"}

// sassDecoder checks Parse. Parse has no printer of a whole Disassembly, but
// Section.Text is the package's own rendering of one kernel's region, and it
// is documented to re-parse through ParseString; sectionsReparse holds it to
// that.
var sassDecoder = cudatest.Decoder[*Disassembly]{
	Decode: func(b []byte) (*Disassembly, error) { return Parse(bytes.NewReader(b)) },
	Equal: func(a, b *Disassembly) bool {
		return reflect.DeepEqual(a.Functions, b.Functions) && reflect.DeepEqual(a.Sections(), b.Sections())
	},
	Invariants: []func(*Disassembly) error{sectionsReparse},
}

// sectionsReparse checks that each Section's Text parses back to that one
// section, and that the instructions it yields are a contiguous run of the
// instructions the whole listing gave its function. A run, not the whole list:
// one kernel may have several regions, and Function merges them.
func sectionsReparse(d *Disassembly) error {
	for i, s := range d.Sections() {
		r, err := Parse(bytes.NewReader(s.Text))
		if err != nil {
			return fmt.Errorf("section %d (%s): re-parse: %v", i, s.Name, err)
		}
		got := r.Sections()
		if len(got) != 1 || got[0].Name != s.Name || !bytes.Equal(got[0].Text, s.Text) {
			return fmt.Errorf("section %d (%s): re-parses as %d sections %v", i, s.Name, len(got), sectionNames(got))
		}
		fn, rf := d.Function(s.Name), r.Function(s.Name)
		if fn == nil || rf == nil {
			return fmt.Errorf("section %d (%s): function missing (listing %v, re-parse %v)", i, s.Name, fn != nil, rf != nil)
		}
		if !containsRun(fn.Instructions, rf.Instructions) {
			return fmt.Errorf("section %d (%s): re-parse gives %d instructions that are not a run of the function's %d",
				i, s.Name, len(rf.Instructions), len(fn.Instructions))
		}
	}
	return nil
}

func sectionNames(ss []Section) []string {
	var names []string
	for _, s := range ss {
		names = append(names, s.Name)
	}
	return names
}

// containsRun reports whether run occurs contiguously in all.
func containsRun(all, run []Instruction) bool {
	for i := 0; i+len(run) <= len(all); i++ {
		if reflect.DeepEqual(all[i:i+len(run)], run) {
			return true
		}
	}
	return false
}

func FuzzSASSParseNeverPanics(f *testing.F) {
	seeds := cudatest.ReadSeeds(f, sassSeeds...)
	for _, p := range cudatest.SortedPaths(seeds) {
		f.Add(seeds[p])
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		sassDecoder.Check(t, data)
	})
}

// TestSASSSectionsReparseFails shows the section invariant rejects a
// disassembly whose Text does not hold its instructions.
func TestSASSSectionsReparseFails(t *testing.T) {
	d, err := ParseString(".text.k:\n        /*0000*/                   EXIT ;\n")
	if err != nil {
		t.Fatal(err)
	}
	if err := sectionsReparse(d); err != nil {
		t.Fatalf("valid listing: %v", err)
	}
	d.sections[0].Text = []byte(".text.k:\n        /*0010*/                   EXIT ;\n")
	if sectionsReparse(d) == nil {
		t.Error("accepted a section whose Text names a different instruction")
	}
	d.sections[0].Text = []byte(".text.j:\n")
	if sectionsReparse(d) == nil {
		t.Error("accepted a section whose Text names a different kernel")
	}
}

// TestSASSParseMutantControls mutates bytes of a fixture that the parse
// retains and requires each mutant to decode to a different listing or be
// rejected.
func TestSASSParseMutantControls(t *testing.T) {
	seed, err := os.ReadFile("testdata/sm121a-encoding.sass")
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name   string
		anchor string // text in the seed
		offset int    // byte of anchor to mutate
		b      byte
	}{
		{"address", "/*0010*/", 5, '2'},
		{"opcode", "S2R R0", 1, '3'},
		{"operand", "IMAD R9, R9, UR4, R0", 6, '8'},
		{"encoding low word", "/* 0x0000df00ff017b82 */", 10, '1'},
		{"encoding high word", "/* 0x000e2e0000002100 */", 8, 'f'},
		{"predicate", "@P0 EXIT", 2, '1'},
		{"kernel name", ".global         addk", 19, 'j'},
		{"banner name", "//--------------------- .text.addk", 33, 'j'},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			i := bytes.Index(seed, []byte(tt.anchor))
			if i < 0 {
				t.Fatalf("seed has no %q", tt.anchor)
			}
			sassDecoder.CheckMutant(t, seed, i+tt.offset, tt.b)
		})
	}
}

// FuzzSASSParseLifeRangeCountNeverPanics checks ParseLifeRangeCount on
// arbitrary text. Its one invariant a caller can rely on is a non-nil map on
// a nil error; a negative pressure cell is accepted today (see the lane
// handoff), so no sign invariant is asserted.
func FuzzSASSParseLifeRangeCountNeverPanics(f *testing.F) {
	f.Add([]byte(sampleLifeRangeText))
	seeds := cudatest.ReadSeeds(f, sassSeeds...)
	for _, p := range cudatest.SortedPaths(seeds) {
		f.Add(seeds[p])
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		m, err := ParseLifeRangeCount(bytes.NewReader(data))
		if err == nil && m == nil {
			t.Errorf("parse %q: nil error with a nil map", data)
		}
	})
}

// FuzzSASSReadToolchainsNeverPanics checks ReadToolchains on arbitrary bytes.
// An image with no stamp is documented to yield no elements and no error, so
// the zero-value check of cudatest.Decoder does not apply; each element it
// does return must carry a count.
func FuzzSASSReadToolchainsNeverPanics(f *testing.F) {
	seeds := cudatest.ReadSeeds(f, cubinSeeds...)
	for _, p := range cudatest.SortedPaths(seeds) {
		b, err := hex.DecodeString(strings.Join(strings.Fields(string(seeds[p])), ""))
		if err != nil {
			f.Fatalf("decode %s: %v", p, err)
		}
		f.Add(b)
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		ts, err := ReadToolchains(bytes.NewReader(data))
		if err != nil {
			return
		}
		for _, tc := range ts {
			if tc.Count < 1 || tc.Tool == "" {
				t.Errorf("toolchain %+v: empty tool or no count", tc)
			}
		}
	})
}

func TestSeedCorpusCoversEveryRecordedFixture(t *testing.T) {
	var seeded []string
	for _, pats := range [][]string{sassSeeds, cubinSeeds, jsonSeeds} {
		seeded = append(seeded, cudatest.SortedPaths(cudatest.ReadSeeds(t, pats...))...)
	}
	excluded := map[string]string{
		"testdata/README.md":                            "describes the fixtures",
		"testdata/roundtrip/README.md":                  "describes the round-trip fixtures",
		"testdata/roundtrip/.gitignore":                 "keeps the raw cubin out of the repository",
		"testdata/roundtrip/kernel.cu":                  "CUDA source of the round-trip fixtures, not a disassembly",
		"testdata/roundtrip/sm121a-roundtrip.bbcfg.dot": "seed of FuzzDOTGraphs in dot_test.go",
	}
	for k, v := range excluded {
		delete(excluded, k)
		excluded[filepath.FromSlash(k)] = v
	}
	cudatest.CheckSeedCoverage(t, "testdata", seeded, excluded)
}
