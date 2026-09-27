package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"golang.org/x/tools/go/analysis/analysistest"

	"github.com/tmc/lib/nvidia/x/cudakernel"
)

// TestAnalyzers runs each check over the fixture package named after it,
// which holds both the cases that must be reported and the ones that must not.
func TestAnalyzers(t *testing.T) {
	for _, a := range analyzers {
		t.Run(a.Name, func(t *testing.T) {
			analysistest.Run(t, analysistest.TestData(), a, a.Name)
		})
	}
}

func TestSetterName(t *testing.T) {
	tests := []struct {
		name string
		want bool
	}{
		{"SetAttribute", true},
		{"FuncSetAttribute", true},
		{"CtxSetLimit", true},
		{"SetCacheConfig", true},
		{"StreamSetAttribute", true},
		{"Set", false},
		{"Settle", false},
		{"Setup", false},
		{"PrimaryCtxReset", false},
		{"ResetPersistingL2Cache", false},
		{"Offset", false},
		{"GetSettings", false},
	}
	for _, tt := range tests {
		if got := setterRE.MatchString(tt.name); got != tt.want {
			t.Errorf("setterRE.MatchString(%q) = %v, want %v", tt.name, got, tt.want)
		}
	}
}

// TestCleanPackageProducesNoDiagnostics runs every check over packages that
// state their layouts and lifetimes correctly. The fixtures carry no want
// comments, so any diagnostic fails the test.
func TestCleanPackageProducesNoDiagnostics(t *testing.T) {
	analysistest.Run(t, analysistest.TestData(), cargalignAnalyzer, "cargalignclean")
	analysistest.Run(t, analysistest.TestData(), unpinnedAnalyzer, "unpinnedclean")
	analysistest.Run(t, analysistest.TestData(), leakAnalyzer, "leakclean")
	analysistest.Run(t, analysistest.TestData(), errCompareAnalyzer, "errcompareclean")
}

// TestUnsupportedLayoutsReported checks the -unsupported flag names layouts
// that have no compile-time offsets. The silent half is cargalignclean, whose
// runtimeBuilt assembles parameters the same way and draws nothing.
func TestUnsupportedLayoutsReported(t *testing.T) {
	defer func(v bool) { reportUnsupported = v }(reportUnsupported)
	reportUnsupported = true
	analysistest.Run(t, analysistest.TestData(), cargalignAnalyzer, "cargalignunsupported")
}

// TestMisaligned12ByteStructRefusedByBothAuthorities is the gate: a twelve-byte
// parameter block holding a device pointer at offset 4 is refused when a Schema
// is constructed and refused by the analyzer at build time, and the offset the
// two name is the same one.
func TestMisaligned12ByteStructRefusedByBothAuthorities(t *testing.T) {
	params := []cudakernel.Param{
		{Name: "n", Type: "u32", ByteWidth: 4, Alignment: 4, Offset: 0},
		{Name: "p", Type: "ptr", ByteWidth: 8, Alignment: 8, IsPointer: true, Offset: 4},
	}
	written := cudakernel.Schema{Params: params, TotalParamBytes: 12}
	if err := written.Validate(); err == nil {
		t.Error("cudakernel.Schema accepted a 12-byte block with a device pointer at offset 4")
	} else {
		t.Logf("construction refused it: %v", err)
	}

	computed, err := cudakernel.NewSchema(params...)
	if err != nil {
		t.Fatalf("cudakernel.NewSchema: %v", err)
	}
	want := computed.Params[1].Offset

	// The analyzer refuses the same layout written in source; its message is
	// checked exactly by TestCargalign.
	analysistest.Run(t, analysistest.TestData(), cargalignAnalyzer, "cargalign")
	got := placedOffsets(t, "cargalign/cargalign.go")
	if len(got) == 0 {
		t.Fatal("no offset asserted of the analyzer in testdata/src/cargalign")
	}
	if got[0] != want {
		t.Errorf("analyzer places the device pointer at offset %d, cudakernel.Schema at %d", got[0], want)
	}
	if total := computed.TotalParamBytes; total == written.TotalParamBytes {
		t.Errorf("the written block and cudakernel.Schema agree on %d bytes; the fixture is no longer misaligned", total)
	}
}

// TestAnalyzerOffsetMovesWithSchema is the control that tells a correct
// analyzer from one that agrees by computing offsets of its own. The two
// fixture packages hold the same source; only the authority differs. If the
// analyzer did its own arithmetic the reported offsets would not move.
func TestAnalyzerOffsetMovesWithSchema(t *testing.T) {
	const (
		authority = "cargalignauthority/block.go"
		perturbed = "cargalignperturbed/block.go"
	)
	if a, p := sourceWithoutWants(t, authority), sourceWithoutWants(t, perturbed); a != p {
		t.Fatal("the two fixtures differ in more than their expected offsets; the control proves nothing")
	}

	analysistest.Run(t, analysistest.TestData(), cargalignAnalyzer, "cargalignauthority")

	defer func(v func(...cudakernel.Param) (cudakernel.Schema, error)) { newSchema = v }(newSchema)
	newSchema = func(params ...cudakernel.Param) (cudakernel.Schema, error) {
		widened := make([]cudakernel.Param, len(params))
		for i, p := range params {
			p.Alignment = 16
			widened[i] = p
		}
		return cudakernel.NewSchema(widened...)
	}
	analysistest.Run(t, analysistest.TestData(), cargalignAnalyzer, "cargalignperturbed")

	// Each Run above binds the analyzer's output to its fixture's offsets
	// exactly, so the two arms differing is the offset having moved.
	before, after := placedOffsets(t, authority), placedOffsets(t, perturbed)
	if len(before) == 0 || len(after) == 0 {
		t.Fatal("no offsets asserted of the analyzer in the control fixtures")
	}
	if before[0] == after[0] {
		t.Fatalf("both arms assert offset %d, so the control cannot see a stuck analyzer", before[0])
	}
	t.Logf("the authority moved the reported offset from %d to %d", before[0], after[0])
}

// TestEveryDiagnosticHasAFiringCase holds the checks to the condition the
// command was deleted for. Every message the source can report must be made
// to fire by a fixture.
func TestEveryDiagnosticHasAFiringCase(t *testing.T) {
	wants := allWants(t)
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, file := range files {
		if strings.HasSuffix(file, "_test.go") {
			continue
		}
		for _, format := range reportFormats(t, file) {
			literal := longestLiteral(format)
			if literal == "" {
				t.Errorf("%s: diagnostic %q has no literal text to match", file, format)
				continue
			}
			if !strings.Contains(wants, literal) {
				t.Errorf("%s: diagnostic %q is never made to fire by a fixture", file, format)
			}
		}
	}
}

var placedRE = regexp.MustCompile(`(?:places it at offset|computes) (\d+)`)

// placedOffsets returns the offsets a fixture asserts of the analyzer, in the
// order they appear.
func placedOffsets(t *testing.T, fixture string) []int {
	t.Helper()
	var out []int
	for _, m := range placedRE.FindAllStringSubmatch(read(t, "testdata/src/"+fixture), -1) {
		n, err := strconv.Atoi(m[1])
		if err != nil {
			t.Fatalf("%s: %v", fixture, err)
		}
		out = append(out, n)
	}
	return out
}

// sourceWithoutWants returns a fixture with its expectations removed, so that
// two fixtures can be compared for being the same program.
func sourceWithoutWants(t *testing.T, fixture string) string {
	t.Helper()
	var b strings.Builder
	for _, line := range strings.Split(read(t, "testdata/src/"+fixture), "\n") {
		if i := strings.Index(line, "// want "); i >= 0 {
			line = line[:i]
		}
		b.WriteString(strings.TrimRight(line, " \t"))
		b.WriteByte('\n')
	}
	return b.String()
}

// allWants returns every expectation in testdata, as one string to search.
// Backslashes are removed, so that a pattern escaping a metacharacter still
// contains the literal text it matches.
func allWants(t *testing.T) string {
	t.Helper()
	var b strings.Builder
	root := "testdata/src"
	entries, err := os.ReadDir(root)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		files, err := os.ReadDir(root + "/" + e.Name())
		if err != nil {
			continue
		}
		for _, f := range files {
			if strings.HasSuffix(f.Name(), ".go") {
				b.WriteString(read(t, root+"/"+e.Name()+"/"+f.Name()))
			}
		}
	}
	return strings.ReplaceAll(b.String(), `\`, "")
}

// reportFormats returns the format strings of every diagnostic a source file
// can report.
func reportFormats(t *testing.T, file string) []string {
	t.Helper()
	f, err := parser.ParseFile(token.NewFileSet(), file, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var out []string
	ast.Inspect(f, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok || len(call.Args) < 2 {
			return true
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok || sel.Sel.Name != "Reportf" {
			return true
		}
		lit, ok := call.Args[1].(*ast.BasicLit)
		if !ok || lit.Kind != token.STRING {
			return true
		}
		s, err := strconv.Unquote(lit.Value)
		if err != nil {
			t.Fatalf("%s: %v", file, err)
		}
		out = append(out, s)
		return true
	})
	return out
}

var verbRE = regexp.MustCompile(`%[-+# 0-9.]*[a-zA-Z]`)

// longestLiteral returns the longest run of a format string that a
// diagnostic prints verbatim, which is what a fixture has to contain.
func longestLiteral(format string) string {
	longest := ""
	for _, part := range verbRE.Split(format, -1) {
		if len(part) > len(longest) {
			longest = part
		}
	}
	return longest
}

func read(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}
