package cudaptx_test

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

// The fuzz target lives in the external test package: cudatest reaches
// cudaptx through cuda and cudalaunch, so an internal test importing it would
// be an import cycle.

// ptxSeeds are the fixtures the fuzz target starts from.
var ptxSeeds = []string{"testdata/*.ptx"}

// ptxDecoder checks Parse. cudaptx has no printer, so there is no Encode and
// no round trip: a printer written only for this test would be checked
// against nothing but its author's reading of the parser.
var ptxDecoder = cudatest.Decoder[*cudaptx.Module]{
	Decode:     cudaptx.ParseBytes,
	Equal:      moduleEqual,
	Invariants: []func(*cudaptx.Module) error{opcodesAreOpcodes, fileNamesAreUnquoted, branchesResolve},
}

// branchesResolve checks the direct-branch contract of Parse. Other branch
// forms are not modelled; in particular this says nothing about brx.idx tables.
func branchesResolve(m *cudaptx.Module) error {
	for _, k := range m.Kernels {
		labels := make(map[string]int, len(k.Labels))
		for _, label := range k.Labels {
			if _, ok := labels[label.Name]; ok {
				return fmt.Errorf("%s: duplicate label %q", k.Name, label.Name)
			}
			if label.Instruction < 0 || label.Instruction > len(k.Instructions) {
				return fmt.Errorf("%s: label %q has invalid instruction index %d", k.Name, label.Name, label.Instruction)
			}
			labels[label.Name] = label.Instruction
		}
		for _, inst := range k.Instructions {
			if inst.Opcode != "bra" && inst.Opcode != "bra.uni" {
				continue
			}
			i, ok := labels[inst.Operands]
			if !ok || i == len(k.Instructions) {
				return fmt.Errorf("%s line %d: %s target %q does not resolve to an instruction", k.Name, inst.Line, inst.Opcode, inst.Operands)
			}
		}
	}
	return nil
}

// moduleEqual compares the exported contents of two modules; the lookup map
// Module.Kernel builds lazily is not part of the value.
func moduleEqual(a, b *cudaptx.Module) bool {
	return fmt.Sprintf("%+v %+v", *a, kernels(a)) == fmt.Sprintf("%+v %+v", *b, kernels(b))
}

func kernels(m *cudaptx.Module) []cudaptx.Kernel {
	var ks []cudaptx.Kernel
	for _, k := range m.Kernels {
		ks = append(ks, *k)
	}
	return ks
}

// opcodesAreOpcodes checks that every instruction the parse reports is an
// instruction: a directive, a label or a bare guard is not.
func opcodesAreOpcodes(m *cudaptx.Module) error {
	for _, k := range m.Kernels {
		for _, in := range k.Instructions {
			switch {
			case in.Opcode == "":
				return fmt.Errorf("%s line %d: empty opcode in %q", k.Name, in.Line, in.Raw)
			case strings.ContainsAny(in.Opcode, " \t\r\n"):
				return fmt.Errorf("%s line %d: opcode %q holds whitespace", k.Name, in.Line, in.Opcode)
			case strings.HasPrefix(in.Opcode, "."), strings.HasPrefix(in.Opcode, "@"):
				return fmt.Errorf("%s line %d: %q is not an instruction", k.Name, in.Line, in.Opcode)
			}
		}
	}
	return nil
}

// fileNamesAreUnquoted checks that a .file name is the string the directive
// quotes, not a piece of the directive: a name holding a quote was cut short.
func fileNamesAreUnquoted(m *cudaptx.Module) error {
	for i, name := range m.Files {
		if strings.Contains(name, `"`) {
			return fmt.Errorf(".file %d: name %q holds a quote", i, name)
		}
	}
	return nil
}

func FuzzPTXParseNeverPanics(f *testing.F) {
	seeds := cudatest.ReadSeeds(f, ptxSeeds...)
	for _, p := range cudatest.SortedPaths(seeds) {
		f.Add(seeds[p])
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		ptxDecoder.Check(t, data)
	})
}

// TestPTXParseInvariantsFail shows that each invariant rejects a module that
// violates it, so a passing fuzz run is not a run of checks that cannot fail.
func TestPTXParseInvariantsFail(t *testing.T) {
	quoted := &cudaptx.Module{Files: map[int]string{1: `"/a`}}
	if fileNamesAreUnquoted(quoted) == nil {
		t.Error("fileNamesAreUnquoted accepted a quoted name")
	}
	for _, op := range []string{"", ".reg", "@%p1", "ld global"} {
		m := &cudaptx.Module{Kernels: []*cudaptx.Kernel{{Name: "k", Instructions: []cudaptx.Instruction{{Opcode: op}}}}}
		if opcodesAreOpcodes(m) == nil {
			t.Errorf("opcodesAreOpcodes accepted opcode %q", op)
		}
	}
}

// TestPTXParseRegressions holds inputs that violated an invariant before the
// parser was fixed.
func TestPTXParseRegressions(t *testing.T) {
	tests := []struct {
		name, ptx string
	}{
		// A "//" inside the quoted name was read as a comment, and the
		// name became `"/a`.
		{"comment marker in file name", ".visible .entry k()\n{\n.loc 1 3 1\nret;\n}\n.file 1 \"/a//b.cu\"\n"},
		{"block comment marker in file name", ".file 1 \"/a/*b.cu\"\n.visible .entry k()\n{\nret;\n}\n"},
		{"unterminated file name", ".file 1 \"/a/b.cu\n"},
		// A guard followed by a directive was reported as an instruction
		// with opcode ".reg".
		{"guarded directive", ".visible .entry k()\n{\n@%p1 .reg .b32 %r;\nret;\n}\n"},
		{"second guard", ".visible .entry k()\n{\n@%p1 @%p2 ret;\n}\n"},
		// A CRLF file leaves a carriage return inside the line, and the
		// opcode took it and the operands with it.
		{"carriage return after opcode", ".visible .entry k()\n{\nld.global.u32\r%r1, [x];\n}\n"},
		{"carriage return after guard", ".visible .entry k()\n{\n@%p1\rret;\n}\n"},
		{"carriage return before semicolon", ".visible .entry k()\n{\nret\r;\n}\n"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ptxDecoder.Check(t, []byte(tt.ptx))
		})
	}
}

func TestPTXFileNameKeepsCommentMarkers(t *testing.T) {
	m, err := cudaptx.ParseString(".visible .entry k()\n{\n.loc 1 3 1\nret;\n}\n.file 1 \"/a//b.cu\" // trailing\n")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := m.Files[1], "/a//b.cu"; got != want {
		t.Errorf("Files[1] = %q, want %q", got, want)
	}
	if got, want := m.Kernels[0].Instructions[0].Loc.File, "/a//b.cu"; got != want {
		t.Errorf("Loc.File = %q, want %q", got, want)
	}
}

// TestPTXMutantControls mutates bytes of a fixture that the parse retains and
// requires each mutant to decode to a different module or be rejected.
func TestPTXMutantControls(t *testing.T) {
	seed, err := os.ReadFile("testdata/lexer.sm90.ptx")
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name   string
		anchor string // text in the seed
		offset int    // byte of anchor to mutate
		b      byte
	}{
		{"loc line", ".loc\t1 6 3, function_name", 7, '7'},
		{"loc file index", ".loc\t1 6 3, function_name", 5, '2'},
		{"inlined_at line", "inlined_at 1 15 5", 13, '6'},
		{"maxntid", ".maxntid 256, 1, 1", 9, '3'},
		{"minnctapersm", ".minnctapersm 4", 14, '5'},
		{"file name", `.file	1 "/tmp/cudaptx-fixtures/lexer.cu"`, 10, 'u'},
		{"opcode", "ld.param.u64", 3, 'q'},
		{"guard", "@%p1", 3, '2'},
		{"kernel name", ".entry ", 7, 'Z'},
		{"version", ".version 9.0", 11, '1'},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			i := bytes.Index(seed, []byte(tt.anchor))
			if i < 0 {
				t.Fatalf("seed has no %q", tt.anchor)
			}
			ptxDecoder.CheckMutant(t, seed, i+tt.offset, tt.b)
		})
	}
}

func TestSeedCorpusCoversEveryRecordedFixture(t *testing.T) {
	seeded := cudatest.SortedPaths(cudatest.ReadSeeds(t, ptxSeeds...))
	excluded := map[string]string{}
	for _, f := range []string{"gen.sh", "README", "lexer.cu", "minimal.cu", "vector.cu"} {
		excluded[filepath.Join("testdata", f)] = "the generator and CUDA sources of the .ptx fixtures, not PTX"
	}
	cudatest.CheckSeedCoverage(t, "testdata", seeded, excluded)
}
