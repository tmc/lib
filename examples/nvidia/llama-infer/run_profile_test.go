package main

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cuda"
)

// TestRunProfile is a behaviour test: run with -profile on the CPU
// reference executor returns nil and leaves a pprof profile in the file.
func TestRunProfile(t *testing.T) {
	// Hide models under the user's home so run uses the generated model.
	t.Setenv("HOME", t.TempDir())
	if p := findDefaultModelPath(); p != "" {
		t.Skipf("model found at %s; this test needs the generated model", p)
	}
	if dev, err := cuda.NewDevice(0); err == nil {
		if err := dev.Close(); err != nil {
			t.Errorf("close device: %v", err)
		}
		t.Skip("CUDA device present; this test needs the CPU reference executor")
	}

	devNull, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	stdout := os.Stdout
	os.Stdout = devNull
	path := filepath.Join(t.TempDir(), "llama.pprof")
	oldProfile, oldSteps := *profilePath, *stepsFlag
	*profilePath, *stepsFlag = path, 1
	t.Cleanup(func() {
		*profilePath, *stepsFlag = oldProfile, oldSteps
		os.Stdout = stdout
		if err := devNull.Close(); err != nil {
			t.Error(err)
		}
	})

	if err := run(); err != nil {
		t.Fatalf("run: %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := profile.ParseData(data); err != nil {
		t.Errorf("parse profile %s: %v", path, err)
	}
}

// isStdoutAssign reports whether stmt is "os.Stdout = rhs".
func isStdoutAssign(stmt ast.Stmt, rhs string) bool {
	as, ok := stmt.(*ast.AssignStmt)
	if !ok || as.Tok != token.ASSIGN || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
		return false
	}
	return nodeText(as.Lhs[0]) == "os.Stdout" && nodeText(as.Rhs[0]) == rhs
}

// cleanupRestoresStdout reports whether stmt is a t.Cleanup call whose
// function literal contains a top-level "os.Stdout = stdout" statement.
func cleanupRestoresStdout(stmt ast.Stmt) bool {
	es, ok := stmt.(*ast.ExprStmt)
	if !ok {
		return false
	}
	call, ok := es.X.(*ast.CallExpr)
	if !ok || len(call.Args) != 1 {
		return false
	}
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok || sel.Sel.Name != "Cleanup" {
		return false
	}
	lit, ok := call.Args[0].(*ast.FuncLit)
	if !ok || lit.Body == nil {
		return false
	}
	return slices.ContainsFunc(lit.Body.List, func(s ast.Stmt) bool {
		return isStdoutAssign(s, "stdout")
	})
}

// unredirectedInferenceCalls parses src and counts calls to run or
// runCPUInference, which stream decoded tokens to os.Stdout (including raw
// single-byte tokens under NewByteTokenizer that make test output binary).
// It reports any function that calls or references either without first
// redirecting os.Stdout to devNull and registering a t.Cleanup that
// restores os.Stdout.
func unredirectedInferenceCalls(name string, src []byte) (calls int, bad []string, err error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, name, src, 0)
	if err != nil {
		return 0, nil, err
	}
	isTarget := func(name string) bool {
		return name == "run" || name == "runCPUInference"
	}
	for _, decl := range f.Decls {
		fd, ok := decl.(*ast.FuncDecl)
		if !ok || fd.Body == nil {
			continue
		}
		redirected, restored := false, false
		locals := make(map[string]bool)
		if fd.Type.Params != nil {
			for _, field := range fd.Type.Params.List {
				for _, name := range field.Names {
					locals[name.Name] = true
				}
			}
		}
		for _, stmt := range fd.Body.List {
			if isStdoutAssign(stmt, "devNull") {
				redirected = true
			}
			if cleanupRestoresStdout(stmt) {
				restored = true
			}
			var stack []ast.Node
			ast.Inspect(stmt, func(n ast.Node) bool {
				if n == nil {
					stack = stack[:len(stack)-1]
					return true
				}
				switch d := n.(type) {
				case *ast.ValueSpec:
					for _, name := range d.Names {
						locals[name.Name] = true
					}
				case *ast.AssignStmt:
					if d.Tok == token.DEFINE {
						for _, lhs := range d.Lhs {
							if id, ok := lhs.(*ast.Ident); ok {
								locals[id.Name] = true
							}
						}
					}
				}
				id, ok := n.(*ast.Ident)
				if !ok || !isTarget(id.Name) || locals[id.Name] {
					stack = append(stack, n)
					return true
				}
				if len(stack) > 0 {
					if sel, ok := stack[len(stack)-1].(*ast.SelectorExpr); ok && sel.Sel == id {
						return false
					}
				}
				calls++
				var call *ast.CallExpr
				if len(stack) > 0 {
					call, _ = stack[len(stack)-1].(*ast.CallExpr)
				}
				if call == nil || call.Fun != id {
					bad = append(bad, fmt.Sprintf("%s: %s: %s referenced without calling it", fset.Position(id.Pos()), fd.Name.Name, id.Name))
					return false
				}
				if !redirected || !restored {
					bad = append(bad, fmt.Sprintf("%s: %s: %s called without os.Stdout = devNull and t.Cleanup restoration", fset.Position(call.Pos()), fd.Name.Name, id.Name))
				}
				return false
			})
		}
	}
	return calls, bad, nil
}

// TestInferenceTestsSilenceStdout checks that every test calling run or
// runCPUInference redirects os.Stdout to os.DevNull first, so byte-level
// fallback tokens never write non-UTF-8 bytes into go test -v output.
func TestInferenceTestsSilenceStdout(t *testing.T) {
	paths, err := filepath.Glob("*_test.go")
	if err != nil {
		t.Fatal(err)
	}
	perFile := make(map[string]int)
	var bad []string
	for _, p := range paths {
		src, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		n, b, err := unredirectedInferenceCalls(p, src)
		if err != nil {
			t.Fatal(err)
		}
		if n > 0 {
			perFile[p] = n
		}
		bad = append(bad, b...)
	}
	wantFiles := map[string]int{
		"dropped_errors_test.go": 1,
		"fallbacks_test.go":      3,
		"run_profile_test.go":    1,
	}
	for file, want := range wantFiles {
		if perFile[file] != want {
			t.Errorf("%s: %d inference calls found, want %d", file, perFile[file], want)
		}
	}
	for _, b := range bad {
		t.Error(b)
	}
}

// TestUnredirectedInferenceCallsControls checks unredirectedInferenceCalls on
// positive and must-fail forms, and on a rewrite of run_profile_test.go.
func TestUnredirectedInferenceCallsControls(t *testing.T) {
	for _, tt := range []struct {
		name  string
		body  string
		calls int
		bad   bool
	}{
		{"positive control: redirected and restored", `
		os.Stdout = devNull
		t.Cleanup(func() { os.Stdout = stdout })
		_ = run()`, 1, false},
		{"positive control: local run variable", `
		run := func() {}
		run()`, 0, false},
		{"positive control: method named run", `
		x.run()`, 0, false},
		{"unredirected run", `
		_ = run()`, 1, true},
		{"unredirected runCPUInference", `
		_ = runCPUInference(w, cfg, tok, nil, nil, nil)`, 1, true},
		{"redirected without cleanup restore", `
		os.Stdout = devNull
		_ = run()`, 1, true},
		{"redirected after run", `
		_ = run()
		os.Stdout = devNull
		t.Cleanup(func() { os.Stdout = stdout })`, 1, true},
		{"referenced as value", `
		os.Stdout = devNull
		t.Cleanup(func() { os.Stdout = stdout })
		f := run
		_ = f`, 1, true},
	} {
		src := "package main\nfunc TestControl(t *testing.T) {\n" + tt.body + "\n}\n"
		calls, bad, err := unredirectedInferenceCalls("control_test.go", []byte(src))
		if err != nil {
			t.Fatalf("%s: %v", tt.name, err)
		}
		if calls != tt.calls {
			t.Errorf("%s: calls = %d, want %d", tt.name, calls, tt.calls)
		}
		if got := len(bad) > 0; got != tt.bad {
			t.Errorf("%s: flagged = %v, want %v (%v)", tt.name, got, tt.bad, bad)
		}
	}

	src, err := os.ReadFile("run_profile_test.go")
	if err != nil {
		t.Fatal(err)
	}
	mutated := strings.Replace(string(src), "\tos.Stdout = devNull\n", "", 1)
	if mutated == string(src) {
		t.Fatal("rewrite did not apply to run_profile_test.go")
	}
	if _, bad, err := unredirectedInferenceCalls("run_profile_test.go", []byte(mutated)); err != nil || len(bad) == 0 {
		t.Errorf("run_profile_test.go without os.Stdout = devNull: %v, %v; want flagged", bad, err)
	}
}
