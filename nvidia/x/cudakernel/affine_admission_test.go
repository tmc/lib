package cudakernel_test

import (
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalock"
)

func TestAffineNativeAdmissionRefusesUnverifiedOwner(t *testing.T) {
	// A missing parent makes even a positive PID refuse before opening or
	// locking a namespace guard. No ambient GPU lock directory is consulted.
	dir := filepath.Join(t.TempDir(), "absent-parent", "lock")
	for _, tc := range []struct {
		name string
		held string
	}{
		{"absent", ""},
		{"presence flag", "1"},
		{"malformed pid", "yes"},
		{"zero pid", "0"},
		{"negative pid", "-1"},
		{"live pid without owner", strconv.Itoa(os.Getpid())},
	} {
		t.Run(tc.name, func(t *testing.T) {
			env := func(key string) string {
				switch key {
				case "GPU_LOCK_HELD":
					return tc.held
				case "GPU_LOCKDIR":
					return dir
				}
				return ""
			}
			if err := affineNativeAdmission(env); !errors.Is(err, cudalock.ErrNotInherited) {
				t.Fatalf("admission = %v, want ErrNotInherited", err)
			}
		})
	}
	if err := affineNativeAdmission(nil); !errors.Is(err, cudalock.ErrNotInherited) {
		t.Fatalf("nil environment admission = %v, want ErrNotInherited", err)
	}
	if _, err := os.Stat(filepath.Dir(dir)); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("absent namespace parent = %v, want not exist", err)
	}
}

// The native test is intentionally not run by host controls. Check its actual
// top-level route so deleting or moving the guard cannot leave these refusal
// tests green while device setup is unguarded.
func TestAffineNativeAdmissionGuardsActualDeviceRoute(t *testing.T) {
	f, err := parser.ParseFile(token.NewFileSet(), "affine_test.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var native *ast.FuncDecl
	for _, decl := range f.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Name.Name == "TestNativeAffineArtifact" {
			native = fn
		}
	}
	if native == nil {
		t.Fatal("native affine test missing")
	}
	var guard *ast.IfStmt
	for _, stmt := range native.Body.List {
		check, ok := stmt.(*ast.IfStmt)
		if !ok {
			continue
		}
		assign, ok := check.Init.(*ast.AssignStmt)
		if !ok || len(assign.Lhs) != 1 || len(assign.Rhs) != 1 {
			continue
		}
		call, ok := assign.Rhs[0].(*ast.CallExpr)
		if !ok || !affineIdent(call.Fun, "affineNativeAdmission") {
			continue
		}
		if guard != nil {
			t.Fatal("duplicate native affine admission")
		}
		guard = check
		if len(call.Args) != 1 || !affineSelector(call.Args[0], "os", "Getenv") {
			t.Fatal("native affine admission does not use actual environment")
		}
		cond, ok := check.Cond.(*ast.BinaryExpr)
		if !ok || cond.Op != token.NEQ || !affineIdent(cond.X, "err") || !affineIdent(cond.Y, "nil") || !affineIdent(assign.Lhs[0], "err") {
			t.Fatal("native affine admission does not refuse its error")
		}
		if check.Else != nil || len(check.Body.List) != 1 {
			t.Fatal("native affine refusal must terminate unconditionally")
		}
		expr, ok := check.Body.List[0].(*ast.ExprStmt)
		if !ok {
			t.Fatal("native affine refusal is not fatal")
		}
		fatal, ok := expr.X.(*ast.CallExpr)
		if !ok || !affineSelector(fatal.Fun, "t", "Fatalf") {
			t.Fatal("native affine refusal is not fatal")
		}
	}
	if guard == nil {
		t.Fatal("native affine admission missing")
	}
	devices := 0
	ast.Inspect(native.Body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok || !affineSelector(call.Fun, "cuda", "NewDevice") {
			return true
		}
		devices++
		if call.Pos() <= guard.End() {
			t.Error("device setup precedes canonical inherited admission")
		}
		return true
	})
	if devices != 1 {
		t.Fatalf("native device setup calls = %d, want 1", devices)
	}
}

func affineIdent(expr ast.Expr, name string) bool {
	x, ok := expr.(*ast.Ident)
	return ok && x.Name == name
}

func affineSelector(expr ast.Expr, pkg, name string) bool {
	x, ok := expr.(*ast.SelectorExpr)
	return ok && affineIdent(x.X, pkg) && x.Sel.Name == name
}
