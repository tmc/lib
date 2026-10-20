package cudalayout

import (
	"errors"
	"fmt"
	"go/ast"
	goparser "go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudalock"
)

// layoutNativeAdmission checks custody before the fixture can discover tools
// or run a compiler or executable. It does not acquire a lock or admit a device.
func layoutNativeAdmission(env func(string) string) (bool, error) {
	if env == nil {
		return false, fmt.Errorf("%w: no environment", cudalock.ErrNotInherited)
	}
	if env("CUDA_LAYOUT_NATIVE") != "1" {
		return false, nil
	}
	if env("GPU_LOCK_DISABLED") != "" || env("CUDATEST_NO_LOCK") != "" {
		return false, fmt.Errorf("%w: native lock bypass is set", cudalock.ErrNotInherited)
	}
	if _, err := cudalock.Inherited(env); err != nil {
		return false, err
	}
	return true, nil
}

func TestNativeAdmissionRefusesUnverifiedCustody(t *testing.T) {
	root := t.TempDir()
	forged := filepath.Join(root, "forged")
	if err := os.Mkdir(forged, 0700); err != nil {
		t.Fatal(err)
	}
	owner := cudalock.Owner{PID: os.Getpid(), Host: "invalid-native-fixture-host", Label: "not-a-holder", Epoch: time.Now().Unix(), At: time.Now()}
	if err := os.WriteFile(filepath.Join(forged, "owner"), []byte(owner.String()), 0600); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ name, held, dir, bypass string }{
		{"missing", "", root, ""},
		{"malformed", "not-a-pid", root, ""},
		{"hand-set-one", "1", root, ""},
		{"no-owner", strconv.Itoa(os.Getpid()), filepath.Join(root, "absent"), ""},
		{"forged-owner", strconv.Itoa(os.Getpid()), forged, ""},
		{"disabled", strconv.Itoa(os.Getpid()), forged, "GPU_LOCK_DISABLED"},
		{"test-bypass", strconv.Itoa(os.Getpid()), forged, "CUDATEST_NO_LOCK"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			values := map[string]string{"CUDA_LAYOUT_NATIVE": "1", "GPU_LOCK_HELD": tc.held, "GPU_LOCKDIR": tc.dir}
			if tc.bypass != "" {
				values[tc.bypass] = "1"
			}
			enabled, err := layoutNativeAdmission(func(key string) string { return values[key] })
			if enabled || !errors.Is(err, cudalock.ErrNotInherited) {
				t.Fatalf("admission = %v, %v; want false, ErrNotInherited", enabled, err)
			}
		})
	}
	if enabled, err := layoutNativeAdmission(nil); enabled || !errors.Is(err, cudalock.ErrNotInherited) {
		t.Fatalf("nil environment = %v, %v", enabled, err)
	}
}

func TestNativeAdmissionDisabled(t *testing.T) {
	for _, value := range []string{"", "0", "true"} {
		calls := 0
		enabled, err := layoutNativeAdmission(func(key string) string {
			calls++
			if key != "CUDA_LAYOUT_NATIVE" {
				t.Fatalf("disabled fixture inspected %s", key)
			}
			return value
		})
		if enabled || err != nil || calls != 1 {
			t.Fatalf("admission(%q) = %v, %v, calls %d", value, enabled, err, calls)
		}
	}
}

// This parses the fixture without invoking its body, compiler discovery, or GPU
// code. Direct admission controls alone would miss a call-site bypass.
func TestNativeAdmissionPrecedesTools(t *testing.T) {
	tree, err := goparser.ParseFile(token.NewFileSet(), "native_test.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	funcs := make(map[string]*ast.FuncDecl)
	for _, d := range tree.Decls {
		if f, ok := d.(*ast.FuncDecl); ok {
			funcs[f.Name.Name] = f
		}
	}
	fixture := funcs["TestNativeLayoutFixture"]
	if fixture == nil || fixture.Body == nil {
		t.Fatal("native fixture missing")
	}
	gate := fixture

	index := 0

	if len(gate.Body.List) < index+3 {
		t.Fatal("admission prefix missing")
	}
	assignment, ok := gate.Body.List[index].(*ast.AssignStmt)
	if !ok || len(assignment.Rhs) != 1 || len(assignment.Lhs) != 2 || assignment.Tok != token.DEFINE {
		t.Fatal("admission must be the first operation")
	}
	isIdent := func(expr ast.Expr, name string) bool {
		id, ok := expr.(*ast.Ident)
		return ok && id.Name == name
	}
	if !isIdent(assignment.Lhs[0], "enabled") || !isIdent(assignment.Lhs[1], "err") {
		t.Fatal("admission results must be checked")
	}
	callGate, ok := assignment.Rhs[0].(*ast.CallExpr)
	if !ok || len(callGate.Args) != 1 {
		t.Fatal("admission call missing")
	}
	getenv, ok := callGate.Args[0].(*ast.SelectorExpr)
	if !ok || !isIdent(getenv.X, "os") || getenv.Sel.Name != "Getenv" {
		t.Fatal("admission must inspect the actual environment")
	}
	nameGate, ok := callGate.Fun.(*ast.Ident)
	if !ok || nameGate.Name != "layoutNativeAdmission" {
		t.Fatal("canonical admission helper missing")
	}
	for i, method := range []string{"Fatal", "Skip"} {
		branch, ok := gate.Body.List[index+1+i].(*ast.IfStmt)
		if !ok || len(branch.Body.List) != 1 {
			t.Fatal("admission failure must stop the fixture")
		}
		if branch.Init != nil || branch.Else != nil {
			t.Fatal("admission branch must be unconditional apart from its result")
		}
		if i == 0 {
			cond, ok := branch.Cond.(*ast.BinaryExpr)
			if !ok || cond.Op != token.NEQ || !isIdent(cond.X, "err") || !isIdent(cond.Y, "nil") {
				t.Fatal("admission error must be refused")
			}
		} else {
			cond, ok := branch.Cond.(*ast.UnaryExpr)
			if !ok || cond.Op != token.NOT || !isIdent(cond.X, "enabled") {
				t.Fatal("disabled admission must skip")
			}
		}
		stmt, ok := branch.Body.List[0].(*ast.ExprStmt)
		if !ok {
			t.Fatal("admission failure does not stop")
		}
		stop, ok := stmt.X.(*ast.CallExpr)
		if !ok {
			t.Fatal("admission stop call missing")
		}
		sel, ok := stop.Fun.(*ast.SelectorExpr)
		if !ok || !isIdent(sel.X, "t") || sel.Sel.Name != method {
			t.Fatalf("admission branch must %s", method)
		}
		if i == 0 && (len(stop.Args) != 1 || !isIdent(stop.Args[0], "err")) {
			t.Fatal("admission error must be reported")
		}
	}
}
