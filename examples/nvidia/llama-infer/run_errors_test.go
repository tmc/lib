package main

import (
	"errors"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

// fakeCloser counts its closes and returns err from each.
type fakeCloser struct {
	err    error
	closes int
}

func (c *fakeCloser) Close() error {
	c.closes++
	return c.err
}

// TestCloseInto is a behaviour test of closeInto, which run defers for its
// trace file, profile file and profile session, and printKernelDisassembly
// defers for its device: it closes once and joins the close error into the
// result without losing the error already there.
func TestCloseInto(t *testing.T) {
	errPrior := errors.New("prior failure")
	for _, tt := range []struct {
		name     string
		prior    error
		closeErr error
		want     []error // errors the result must wrap
		wantText string  // text the result must contain
	}{
		{"positive control: close succeeds", nil, nil, nil, ""},
		{"positive control: close succeeds after a failure", errPrior, nil, []error{errPrior}, ""},
		{"close fails", nil, errFakeClose, []error{errFakeClose}, "close trace file: fake close failed"},
		{"close fails after a failure", errPrior, errFakeClose, []error{errPrior, errFakeClose}, "close trace file: fake close failed"},
		{"close fails with releases held", errReleasesHeld, errFakeClose, []error{errReleasesHeld, errFakeClose}, "close trace file"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			c := &fakeCloser{err: tt.closeErr}
			err := tt.prior
			closeInto(&err, c, "close trace file")
			if c.closes != 1 {
				t.Errorf("closed %d times, want 1", c.closes)
			}
			if tt.closeErr == nil && err != tt.prior {
				t.Errorf("err = %v, want %v unchanged", err, tt.prior)
			}
			if tt.closeErr != nil && err == nil {
				t.Errorf("err = nil, want an error wrapping %v", tt.closeErr)
			}
			for _, e := range tt.want {
				if !errors.Is(err, e) {
					t.Errorf("error %v does not wrap %v", err, e)
				}
			}
			if tt.wantText != "" && (err == nil || !strings.Contains(err.Error(), tt.wantText)) {
				t.Errorf("error %v does not contain %q", err, tt.wantText)
			}
		})
	}
}

// TestPrintSASSDisassembly is a CPU behaviour test of printSASSDisassembly:
// it prints the requested kernel, all kernels, or a notice when the kernel
// is not in forward.sass, and returns an error wrapping cudasass.ParseString's
// error instead of printing it and returning nil when forwardSASS cannot be
// parsed.
func TestPrintSASSDisassembly(t *testing.T) {
	origSASS := forwardSASS
	stdout := os.Stdout
	t.Cleanup(func() {
		forwardSASS = origSASS
		os.Stdout = stdout
	})

	capture := func(t *testing.T, kernel string) (string, error) {
		t.Helper()
		path := filepath.Join(t.TempDir(), "stdout")
		f, err := os.Create(path)
		if err != nil {
			t.Fatal(err)
		}
		os.Stdout = f
		callErr := printSASSDisassembly(kernel)
		os.Stdout = stdout
		if err := f.Close(); err != nil {
			t.Fatal(err)
		}
		out, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return string(out), callErr
	}

	out, err := capture(t, "RMSNorm")
	if err != nil {
		t.Fatalf("RMSNorm: %v", err)
	}
	if !strings.Contains(out, "--- Kernel: RMSNorm ---") || strings.Contains(out, "--- Kernel: EmbeddingLookup ---") || strings.Contains(out, "[SASS Notice]") {
		t.Errorf("RMSNorm output = %q, want RMSNorm kernel only", out)
	}

	out, err = capture(t, "all")
	if err != nil {
		t.Fatalf("all: %v", err)
	}
	if !strings.Contains(out, "--- Kernel: RMSNorm ---") || !strings.Contains(out, "--- Kernel: EmbeddingLookup ---") {
		t.Errorf("all output missing expected kernels")
	}

	out, err = capture(t, "NoSuchKernel")
	if err != nil {
		t.Fatalf("NoSuchKernel: %v", err)
	}
	if !strings.Contains(out, `[SASS Notice] "NoSuchKernel" is not among the kernels in forward.sass`) {
		t.Errorf("NoSuchKernel output = %q, want notice", out)
	}

	for _, tt := range []struct {
		name     string
		sass     string
		kernel   string
		wantIs   error
		wantText string
	}{
		{
			name:     "non-text section instruction",
			sass:     "\t.section\t.nv.capmerc.text.k,\"\",@progbits\n /*0000*/ EXIT ;\n",
			kernel:   "RMSNorm",
			wantIs:   cudasass.ErrNotTextSection,
			wantText: "parse SASS disassembly:",
		},
		{
			name:     "instruction outside section",
			sass:     " /*0000*/ NOP ;\n",
			kernel:   "all",
			wantText: "parse SASS disassembly: cudasass: instruction at 0x0000 outside section",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			forwardSASS = tt.sass
			out, err := capture(t, tt.kernel)
			if err == nil || !strings.Contains(err.Error(), tt.wantText) {
				t.Errorf("err = %v, want error containing %q", err, tt.wantText)
			}
			if tt.wantIs != nil && !errors.Is(err, tt.wantIs) {
				t.Errorf("err = %v, does not wrap %v", err, tt.wantIs)
			}
			if out != "" {
				t.Errorf("output = %q, want empty on parse error", out)
			}
		})
	}
}

// findFunc returns the function named name in f that has no receiver.
func findFunc(f *ast.File, name string) *ast.FuncDecl {
	for _, decl := range f.Decls {
		if fd, ok := decl.(*ast.FuncDecl); ok && fd.Recv == nil && fd.Name.Name == name {
			return fd
		}
	}
	return nil
}

// nodeText returns n formatted.
func nodeText(n ast.Node) string {
	var b strings.Builder
	if err := format.Node(&b, token.NewFileSet(), n); err != nil {
		return fmt.Sprintf("<%v>", err)
	}
	return b.String()
}

// statementAfter returns the statement that follows, in the same list, the
// assignment in body whose value formats as call, and whether exactly one
// such assignment was found.
func statementAfter(body *ast.BlockStmt, call string) (next ast.Stmt, ok bool) {
	n := 0
	ast.Inspect(body, func(node ast.Node) bool {
		var list []ast.Stmt
		switch s := node.(type) {
		case *ast.FuncLit:
			return false
		case *ast.BlockStmt:
			list = s.List
		case *ast.CaseClause:
			list = s.Body
		default:
			return true
		}
		for i, s := range list {
			as, ok := s.(*ast.AssignStmt)
			if !ok || len(as.Rhs) != 1 || nodeText(as.Rhs[0]) != call {
				continue
			}
			n++
			if i+1 < len(list) {
				next = list[i+1]
			}
		}
		return true
	})
	return next, n == 1
}

// runWiringProblems is a source-shape check of src, the text of main.go,
// for the error paths that no CPU test can reach: run returns the errors of
// dev.Name and dev.ComputeCapability; run defers closeInto into its result
// rerr for the trace file, the profile file and then the profile session,
// so the session flushes before its file closes; printKernelDisassembly
// defers closeInto into its result rerr for the device, reports archOr's
// error and returns printSASSDisassembly's result; and runGPUInference's
// show closure returns printToken's result.
func runWiringProblems(src []byte) ([]string, error) {
	f, err := parser.ParseFile(token.NewFileSet(), "main.go", src, 0)
	if err != nil {
		return nil, err
	}
	var bad []string
	run := findFunc(f, "run")
	if run == nil {
		return []string{"run not found"}, nil
	}
	if r := run.Type.Results; r == nil || len(r.List) != 1 || len(r.List[0].Names) != 1 || r.List[0].Names[0].Name != "rerr" {
		bad = append(bad, "run's result is not named rerr")
	}

	// run returns the device query errors.
	for _, call := range []string{"dev.Name()", "dev.ComputeCapability()"} {
		next, ok := statementAfter(run.Body, call)
		if !ok {
			bad = append(bad, fmt.Sprintf("run: want one assignment from %s", call))
			continue
		}
		is, ok := next.(*ast.IfStmt)
		if !ok || is.Init != nil || nodeText(is.Cond) != "err != nil" || !returnsErr(is.Body, "err", 1) {
			bad = append(bad, fmt.Sprintf("run: %s is not followed by if err != nil returning err", call))
		}
	}

	// run defers closeInto for its outputs, in this order.
	var deferred []string
	ast.Inspect(run.Body, func(n ast.Node) bool {
		switch s := n.(type) {
		case *ast.FuncLit:
			return false
		case *ast.DeferStmt:
			if id, ok := s.Call.Fun.(*ast.Ident); ok && id.Name == "closeInto" {
				deferred = append(deferred, nodeText(s.Call))
			}
		}
		return true
	})
	wantDeferred := []string{
		`closeInto(&rerr, f, "close trace file")`,
		`closeInto(&rerr, f, "close profile file")`,
		`closeInto(&rerr, profSession, "close profile session")`,
	}
	if !slices.Equal(deferred, wantDeferred) {
		bad = append(bad, fmt.Sprintf("run defers %q, want %q", deferred, wantDeferred))
	}

	// printKernelDisassembly defers closeInto into rerr for the device,
	// reports archOr's error, and returns printSASSDisassembly's result.
	pkd := findFunc(f, "printKernelDisassembly")
	if pkd == nil {
		return append(bad, "printKernelDisassembly not found"), nil
	}
	if r := pkd.Type.Results; r == nil || len(r.List) != 1 || len(r.List[0].Names) != 1 || r.List[0].Names[0].Name != "rerr" {
		bad = append(bad, "printKernelDisassembly's result is not named rerr")
	}
	var pkdDefers []string
	devClosed := 0
	sassReturns := 0
	const archCall = "archMajor, archMinor, err = archOr(dev.ComputeCapability, archMajor, archMinor)"
	reports := 0
	ast.Inspect(pkd.Body, func(n ast.Node) bool {
		switch s := n.(type) {
		case *ast.DeferStmt:
			pkdDefers = append(pkdDefers, nodeText(s.Call))
			return true
		case *ast.IfStmt:
			if s.Init != nil && nodeText(s.Init) == "dev, err := cuda.NewDevice(0)" &&
				s.Else == nil && nodeText(s.Cond) == "err == nil" &&
				len(s.Body.List) > 0 && nodeText(s.Body.List[0]) == `defer closeInto(&rerr, dev, "close device")` {
				devClosed++
			}
			if s.Init == nil && s.Else == nil && nodeText(s.Cond) == "showSASS" &&
				len(s.Body.List) > 0 && nodeText(s.Body.List[len(s.Body.List)-1]) == "return printSASSDisassembly(targetKernel)" {
				sassReturns++
			}
			if s.Init == nil || nodeText(s.Init) != archCall {
				return true
			}
			if nodeText(s.Cond) != "err != nil" || len(s.Body.List) != 1 {
				return true
			}
			es, ok := s.Body.List[0].(*ast.ExprStmt)
			call, cok := ast.Unparen(es.X).(*ast.CallExpr)
			if ok && cok && len(call.Args) > 1 && nodeText(call.Fun) == "fmt.Fprintf" && nodeText(call.Args[0]) == "os.Stderr" &&
				slices.ContainsFunc(call.Args[1:], func(e ast.Expr) bool { return mentions(e, "err") }) {
				reports++
			}
		}
		return true
	})
	wantPKDDefers := []string{`closeInto(&rerr, dev, "close device")`}
	if devClosed != 1 || !slices.Equal(pkdDefers, wantPKDDefers) {
		bad = append(bad, fmt.Sprintf("printKernelDisassembly: %d device close branches, defers %q; want 1 and %q", devClosed, pkdDefers, wantPKDDefers))
	}
	if reports != 1 {
		bad = append(bad, fmt.Sprintf("printKernelDisassembly: %d statements %q reporting err to standard error, want 1", reports, "if "+archCall+"; err != nil"))
	}
	if sassReturns != 1 {
		bad = append(bad, fmt.Sprintf("printKernelDisassembly: %d showSASS branches ending with return printSASSDisassembly(targetKernel), want 1", sassReturns))
	}

	// runGPUInference's show closure returns printToken's result.
	gpu := findFunc(f, "runGPUInference")
	if gpu == nil {
		return append(bad, "runGPUInference not found"), nil
	}
	var shows []string
	ast.Inspect(gpu.Body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
			return true
		}
		if id, ok := as.Lhs[0].(*ast.Ident); ok && id.Name == "show" {
			shows = append(shows, nodeText(as.Rhs[0]))
		}
		return true
	})
	wantShow := []string{"func(tok int32) error {\n\treturn printToken(tokenizer, tok)\n}"}
	if !slices.Equal(shows, wantShow) {
		bad = append(bad, fmt.Sprintf("runGPUInference assigns show %q, want %q", shows, wantShow))
	}
	return bad, nil
}

// TestRunWiring is a source-shape check, not a behaviour test; see
// runWiringProblems. TestCloseInto and TestPrintSASSDisassembly test the
// behaviour of closeInto and printSASSDisassembly.
func TestRunWiring(t *testing.T) {
	src, err := os.ReadFile("main.go")
	if err != nil {
		t.Fatal(err)
	}
	bad, err := runWiringProblems(src)
	if err != nil {
		t.Fatal(err)
	}
	for _, b := range bad {
		t.Error(b)
	}
}

// TestRunWiringControls checks that runWiringProblems flags each rewrite of
// main.go that drops or misroutes one of the errors it checks.
func TestRunWiringControls(t *testing.T) {
	src, err := os.ReadFile("main.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct{ name, from, to string }{
		{"show prints and returns nil",
			"return printToken(tokenizer, tok)\n\t}",
			"printToken(tokenizer, tok)\n\t\treturn nil\n\t}"},
		{"show drops the error by assignment",
			"return printToken(tokenizer, tok)\n\t}",
			"err := printToken(tokenizer, tok)\n\t\t_ = err\n\t\treturn nil\n\t}"},
		{"device name error printed",
			"return fmt.Errorf(\"device name: %w\", err)",
			"fmt.Fprintln(os.Stderr, err)"},
		{"device name error replaced",
			"return fmt.Errorf(\"device name: %w\", err)",
			"name = \"unknown\""},
		{"device name error ignored",
			"name, err := dev.Name()\n\t\tif err != nil {",
			"name, err := dev.Name()\n\t\tif false {"},
		{"compute capability error printed",
			"return fmt.Errorf(\"device compute capability: %w\", err)",
			"fmt.Fprintln(os.Stderr, err)"},
		{"disasm result not named rerr",
			"func printKernelDisassembly(targetKernel, mode string) (rerr error) {",
			"func printKernelDisassembly(targetKernel, mode string) error {"},
		{"disasm device close only printed",
			"defer closeInto(&rerr, dev, \"close device\")",
			"defer func() {\n\t\t\tif err := dev.Close(); err != nil {\n\t\t\t\tfmt.Fprintf(os.Stderr, \"close device: %v\\n\", err)\n\t\t\t}\n\t\t}()"},
		{"disasm device close into another error",
			"defer closeInto(&rerr, dev, \"close device\")",
			"defer closeInto(&err, dev, \"close device\")"},
		{"archOr error not reported",
			"fmt.Fprintf(os.Stderr, \"compute capability: %v; compiling for sm_%d%d\\n\", err, archMajor, archMinor)",
			"fmt.Fprintf(os.Stderr, \"compiling for sm_%d%d\\n\", archMajor, archMinor)"},
		{"archOr error not tested",
			"archMinor); err != nil {",
			"archMinor); false {"},
		{"disasm SASS error dropped",
			"return printSASSDisassembly(targetKernel)",
			"printSASSDisassembly(targetKernel)"},
		{"disasm SASS error blanked",
			"return printSASSDisassembly(targetKernel)",
			"_ = printSASSDisassembly(targetKernel)"},
		{"disasm SASS error only printed",
			"return printSASSDisassembly(targetKernel)",
			"if err := printSASSDisassembly(targetKernel); err != nil {\n\t\t\tfmt.Fprintln(os.Stderr, err)\n\t\t}"},
		{"disasm SASS wrong kernel argument",
			"return printSASSDisassembly(targetKernel)",
			"return printSASSDisassembly(\"all\")"},
		{"trace file close only printed",
			"defer closeInto(&rerr, f, \"close trace file\")",
			"defer func() {\n\t\t\tif err := f.Close(); err != nil {\n\t\t\t\tfmt.Fprintf(os.Stderr, \"close trace file: %v\\n\", err)\n\t\t\t}\n\t\t}()"},
		{"profile session close into another error",
			"defer closeInto(&rerr, profSession, \"close profile session\")",
			"defer closeInto(&err, profSession, \"close profile session\")"},
		{"profile file closed before its session",
			"defer closeInto(&rerr, f, \"close profile file\")\n\t\tprofSession = cudaprof.NewSession(f)\n\t\t// Registered after the file's, so the session flushes into the\n\t\t// file before the file closes.\n\t\tdefer closeInto(&rerr, profSession, \"close profile session\")",
			"profSession = cudaprof.NewSession(f)\n\t\tdefer closeInto(&rerr, profSession, \"close profile session\")\n\t\tdefer closeInto(&rerr, f, \"close profile file\")"},
	} {
		if n := strings.Count(string(src), tt.from); n != 1 {
			t.Errorf("%s: main.go contains %q %d times, want once", tt.name, tt.from, n)
			continue
		}
		bad, err := runWiringProblems([]byte(strings.Replace(string(src), tt.from, tt.to, 1)))
		if err != nil {
			t.Errorf("%s: %v", tt.name, err)
			continue
		}
		if len(bad) == 0 {
			t.Errorf("%s: not flagged", tt.name)
		}
	}
}

// bareErrorCalls parses the files in srcs, which form one package, and
// returns the number of calls in them of the package's own functions whose
// last result is an error, and the deferred, go and expression statements
// among those calls, which drop the error. The check is syntactic: it sees
// only calls of package-level functions by their plain name, not methods,
// function values or calls into other packages.
func bareErrorCalls(srcs map[string][]byte) (calls int, bad []string, err error) {
	fset := token.NewFileSet()
	var files []*ast.File
	for _, name := range slices.Sorted(maps.Keys(srcs)) {
		f, err := parser.ParseFile(fset, name, srcs[name], 0)
		if err != nil {
			return 0, nil, err
		}
		files = append(files, f)
	}
	returnsError := make(map[string]bool)
	for _, f := range files {
		for _, decl := range f.Decls {
			fd, ok := decl.(*ast.FuncDecl)
			if !ok || fd.Recv != nil || fd.Type.Results == nil {
				continue
			}
			r := fd.Type.Results.List
			if id, ok := r[len(r)-1].Type.(*ast.Ident); ok && id.Name == "error" {
				returnsError[fd.Name.Name] = true
			}
		}
	}
	isErrorCall := func(e ast.Expr) bool {
		call, ok := ast.Unparen(e).(*ast.CallExpr)
		if !ok {
			return false
		}
		id, ok := call.Fun.(*ast.Ident)
		return ok && returnsError[id.Name]
	}
	for _, f := range files {
		ast.Inspect(f, func(n ast.Node) bool {
			var e ast.Expr
			switch s := n.(type) {
			case *ast.CallExpr:
				if isErrorCall(s) {
					calls++
				}
				return true
			case *ast.ExprStmt:
				e = s.X
			case *ast.DeferStmt:
				e = s.Call
			case *ast.GoStmt:
				e = s.Call
			default:
				return true
			}
			if isErrorCall(e) {
				bad = append(bad, fmt.Sprintf("%s: %s", fset.Position(n.Pos()), nodeText(n)))
			}
			return true
		})
	}
	return calls, bad, nil
}

// TestNoBareErrorCalls is a source check of the package's non-test files,
// whatever their build constraints, for a statement that calls one of the
// package's own error-returning functions and drops its error. gen.go is a
// separate program and is checked on its own.
func TestNoBareErrorCalls(t *testing.T) {
	paths, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	pkg := make(map[string][]byte)
	for _, p := range paths {
		if strings.HasSuffix(p, "_test.go") {
			continue
		}
		src, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		if p == "gen.go" {
			_, bad, err := bareErrorCalls(map[string][]byte{p: src})
			if err != nil {
				t.Fatal(err)
			}
			for _, b := range bad {
				t.Errorf("error dropped: %s", b)
			}
			continue
		}
		pkg[p] = src
	}
	if pkg["main.go"] == nil {
		t.Fatal("main.go not read")
	}
	calls, bad, err := bareErrorCalls(pkg)
	if err != nil {
		t.Fatal(err)
	}
	// main.go calls printToken twice, among others; fewer calls means the
	// check looked at nothing.
	if calls < 2 {
		t.Errorf("%d calls of error-returning package functions found, want at least 2", calls)
	}
	for _, b := range bad {
		t.Errorf("error dropped: %s", b)
	}
}

// TestBareErrorCallsControls checks bareErrorCalls on small sources: forms
// that keep the error, and must-fail forms that each drop it.
func TestBareErrorCallsControls(t *testing.T) {
	const decls = `
func g() error { return nil }
func k() (int, error) { return 0, nil }
func h() {}
type T struct{}
func (T) g() error { return nil }
`
	for _, tt := range []struct {
		name  string
		body  string
		calls int
		bad   bool
	}{
		{"positive control: checked", `
		if err := g(); err != nil {
			return err
		}`, 1, false},
		{"positive control: returned", `
		return g()`, 1, false},
		{"positive control: returned from a literal", `
		show := func(tok int32) error { return g() }
		_ = show`, 1, false},
		{"positive control: no error result", `
		h()`, 0, false},
		{"positive control: method, not seen", `
		var x T
		x.g()`, 0, false},
		{"bare call", `
		g()`, 1, true},
		{"bare call with two results", `
		k()`, 1, true},
		{"bare call in a literal", `
		show := func(tok int32) error {
			g()
			return nil
		}
		_ = show`, 1, true},
		{"parenthesized", `
		(g())`, 1, true},
		{"deferred", `
		defer g()`, 1, true},
		{"go statement", `
		go g()`, 1, true},
	} {
		src := "package p\n" + decls + "func f() error {\n" + tt.body + "\nreturn nil\n}\n"
		calls, bad, err := bareErrorCalls(map[string][]byte{"control.go": []byte(src)})
		if err != nil {
			t.Fatalf("%s: %v", tt.name, err)
		}
		if calls != tt.calls {
			t.Errorf("%s: %d calls found, want %d", tt.name, calls, tt.calls)
		}
		if got := len(bad) > 0; got != tt.bad {
			t.Errorf("%s: flagged %v, want %v (%v)", tt.name, got, tt.bad, bad)
		}
	}
}
