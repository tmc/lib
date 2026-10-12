package cudacluster

import (
	"errors"
	"go/ast"
	"go/printer"
	"regexp"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaptx"
)

// The geometry emitters must spell their special registers the way the
// generated PTX ISA table does. The literal pins elsewhere fix the strings;
// these tests tie them to cudaptx, so a spelling the table does not know fails
// even if both the emitter and a literal pin are changed together.

// specialRegister splits the source operand of a "mov.u32 dst, src;" read into
// the register name and its axis suffix, if any.
func specialRegister(t *testing.T, ptx, dst string) (reg, axis string) {
	t.Helper()
	prefix := "mov.u32 " + dst + ", "
	if !strings.HasPrefix(ptx, prefix) || !strings.HasSuffix(ptx, ";") {
		t.Fatalf("%q is not a mov.u32 read into %s", ptx, dst)
	}
	src := strings.TrimSuffix(strings.TrimPrefix(ptx, prefix), ";")
	reg, axis, _ = strings.Cut(src, ".")
	return reg, axis
}

func TestEmittedSpecialRegistersAreInTheISATable(t *testing.T) {
	type emitter func(string, Dim) (string, error)
	scalar := func(f func(string) string) emitter {
		return func(dst string, _ Dim) (string, error) { return f(dst), nil }
	}
	tests := []struct {
		name     string
		emit     emitter
		dim      Dim
		wantAxis string
	}{
		{"EmitBlockRank", scalar(EmitBlockRank), DimX, ""},
		{"EmitClusterSize", scalar(EmitClusterSize), DimX, ""},
		{"EmitClusterCTAID.x", EmitClusterCTAID, DimX, "x"},
		{"EmitClusterCTAID.y", EmitClusterCTAID, DimY, "y"},
		{"EmitClusterCTAID.z", EmitClusterCTAID, DimZ, "z"},
		{"EmitClusterNCTAID.x", EmitClusterNCTAID, DimX, "x"},
		{"EmitClusterNCTAID.y", EmitClusterNCTAID, DimY, "y"},
		{"EmitClusterNCTAID.z", EmitClusterNCTAID, DimZ, "z"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ptx, err := tc.emit("%r1", tc.dim)
			if err != nil {
				t.Fatal(err)
			}
			reg, axis := specialRegister(t, ptx, "%r1")
			if !cudaptx.IsSpecialRegister(reg) {
				t.Errorf("%s emits %q, which cudaptx does not list as a special register", tc.name, reg)
			}
			if axis != tc.wantAxis {
				t.Errorf("%s emits axis %q, want %q", tc.name, axis, tc.wantAxis)
			}
		})
	}
}

// asmTemplate matches the template string of a GNU inline asm statement.
var asmTemplate = regexp.MustCompile(`asm(?:\s+volatile)?\s*\(\s*"((?:[^"\\]|\\.)*)"`)

// escapedRegister matches a special register in an inline asm template, where
// a literal percent sign is written %%.
var escapedRegister = regexp.MustCompile(`%%([a-z][a-z0-9_]*)(?:\.([xyz]))?`)

func TestRankGeometrySourceSpecialRegistersAreInTheISATable(t *testing.T) {
	src := rankGeometrySource(4)
	var got []string
	for _, m := range asmTemplate.FindAllStringSubmatch(src, -1) {
		for _, r := range escapedRegister.FindAllStringSubmatch(m[1], -1) {
			reg := "%" + r[1]
			if !cudaptx.IsSpecialRegister(reg) {
				t.Errorf("rank geometry fixture reads %q, which cudaptx does not list as a special register", reg)
			}
			if r[2] != "" {
				reg += "." + r[2]
			}
			got = append(got, reg)
		}
	}
	slices.Sort(got)
	want := []string{
		"%cluster_ctaid.x",
		"%cluster_ctaid.y",
		"%cluster_ctarank",
		"%cluster_nctaid.x",
		"%cluster_nctaid.y",
		"%cluster_nctarank",
	}
	if !slices.Equal(got, want) {
		t.Errorf("rank geometry fixture reads %q, want %q", got, want)
	}
}

// TestISATableRejectsUnderscorelessSpelling keeps the cross-checks above from
// passing vacuously: the table must refuse the spelling an earlier draft used.
func TestISATableRejectsUnderscorelessSpelling(t *testing.T) {
	for _, reg := range []string{"%clusterctaid", "%clusternctaid", "%clusterctarank", "%clusternctarank", "%cluster_ctaid.x"} {
		if cudaptx.IsSpecialRegister(reg) {
			t.Errorf("cudaptx.IsSpecialRegister(%q) = true, want false", reg)
		}
	}
	if !cudaptx.IsSpecialRegister("%cluster_ctaid") {
		t.Errorf("cudaptx.IsSpecialRegister(%q) = false, want true", "%cluster_ctaid")
	}
}

// TestQualificationClosedGateIsNotASkip keeps the native qualification test
// from passing a tagged run with the native gate closed. Its file must build
// only with the native tag, which is what makes its literal true correct. Its
// first statement must report the closed gate as a vacuous native run, which
// fails wrapping [ErrHardwarePending]; its body must not call a skip itself;
// and its next statement must still refuse an open gate without an inherited
// GPU lock. This reads only the test's own body: a skip inside a function it
// calls, such as cudatest.RequireDevice, is not visible here.
func TestQualificationClosedGateIsNotASkip(t *testing.T) {
	const name = "TestClusterQualification_RankAndLocalWindowControl"
	fset, file := parseTestFile(t, "qualification_native_test.go")
	if expr := fileBuildConstraint(t, file); expr == nil || expr.String() != "native" {
		t.Fatalf("qualification_native_test.go build constraint = %v, want %q", expr, "native")
	}
	var fn *ast.FuncDecl
	for _, d := range file.Decls {
		if f, ok := d.(*ast.FuncDecl); ok && f.Name.Name == name {
			fn = f
		}
	}
	if fn == nil || len(fn.Body.List) < 2 {
		t.Fatalf("%s not found or too short", name)
	}
	src := func(n ast.Node) string {
		var b strings.Builder
		if err := printer.Fprint(&b, fset, n); err != nil {
			t.Fatal(err)
		}
		return b.String()
	}

	ast.Inspect(fn.Body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		switch s := src(call.Fun); s {
		case "t.Skip", "t.Skipf", "t.SkipNow", "skipClosedGate":
			t.Errorf("%s calls %s at %s; a closed gate in a native build must fail", name, s, fset.Position(call.Pos()))
		}
		return true
	})

	closed, ok := fn.Body.List[0].(*ast.IfStmt)
	if !ok || closed.Init == nil || closed.Else != nil {
		t.Fatalf("%s first statement = %s, want the closed-gate check", name, src(fn.Body.List[0]))
	}
	if got, want := src(closed.Init), "err := requireNativeLock(os.Getenv)"; got != want {
		t.Errorf("closed-gate check init = %q, want %q", got, want)
	}
	if got, want := src(closed.Cond), "errors.Is(err, errNativeGateClosed)"; got != want {
		t.Errorf("closed-gate check condition = %q, want %q", got, want)
	}
	if len(closed.Body.List) != 1 || src(closed.Body.List[0]) != "reportNativeGate(t, true, err)" {
		t.Errorf("closed-gate branch = %s, want reportNativeGate(t, true, err)", src(closed.Body))
	}

	open, ok := fn.Body.List[1].(*ast.IfStmt)
	if !ok || open.Init == nil || src(open.Init) != "err := requireNativeLock(os.Getenv)" || src(open.Cond) != "err != nil" ||
		len(open.Body.List) != 1 || src(open.Body.List[0]) != "t.Fatal(err)" {
		t.Errorf("%s second statement = %s, want an open gate without an inherited lock to fail", name, src(fn.Body.List[1]))
	}

	// The branch above, run against a closed gate, fails with the typed
	// pending error and does not skip.
	err := requireNativeLock(func(string) string { return "" })
	if !errors.Is(err, errNativeGateClosed) || !errors.Is(err, ErrHardwarePending) {
		t.Fatalf("closed gate = %v, want errNativeGateClosed wrapping ErrHardwarePending", err)
	}
	var r recordedGate
	reportNativeGate(&r, true, err)
	if !slices.Equal(r.calls, []string{"fatal"}) {
		t.Fatalf("closed gate in a native build reported %q, want [fatal]", r.calls)
	}
	if !strings.Contains(r.text, errVacuousNativeRun.Error()) || !strings.Contains(r.text, "native gate is closed") {
		t.Errorf("closed gate in a native build reported %q, want the vacuous-run and closed-gate errors", r.text)
	}
}
