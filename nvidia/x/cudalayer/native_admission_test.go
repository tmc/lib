package cudalayer_test

import (
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalock"
)

// Check the source rather than invoking a device helper: even a broken guard
// must not make these host controls attempt to load a vendor library.
func TestMLPNativeAdmissionSource(t *testing.T) {
	for _, row := range []struct {
		file  string
		calls int
	}{
		{"fused_mlp_native_test.go", 1},
		{"mlp_backward_native_test.go", 2},
	} {
		t.Run(row.file, func(t *testing.T) {
			body, err := os.ReadFile(row.file)
			if err != nil {
				t.Fatal(err)
			}
			if err := checkMLPNativeAdmission(body, row.calls); err != nil {
				t.Fatal(err)
			}
			guard := "if _, err := cudalock.Inherited(os.Getenv); err != nil {\n\t\tt.Skipf(\"SKIPPED, NOT VERIFIED: inherited GPU lock required: %v\", err)\n\t}"
			for _, changed := range []string{
				strings.Replace(string(body), guard+"\n\tdev := cudatest.RequireDevice(t)", "dev := cudatest.RequireDevice(t)\n\t"+guard, 1),
				strings.Replace(string(body), guard, "", 1),
			} {
				if changed == string(body) {
					t.Fatal("guard-order mutation did not match")
				}
				if err := checkMLPNativeAdmission([]byte(changed), row.calls); err == nil {
					t.Fatal("missing or late guard accepted")
				}
			}
			for _, mutation := range []struct {
				name string
				old  string
				new  string
			}{
				{"presence only", "_, err := cudalock.Inherited(os.Getenv); err != nil", "err := os.Getenv(\"GPU_LOCK_HELD\"); err == \"\""},
				{"wrong environment", "cudalock.Inherited(os.Getenv)", "cudalock.Inherited(nil)"},
				{"inverted refusal", "err != nil", "err == nil"},
				{"nonterminal refusal", "t.Skipf(\"SKIPPED, NOT VERIFIED: inherited", "t.Logf(\"SKIPPED, NOT VERIFIED: inherited"},
			} {
				t.Run(mutation.name, func(t *testing.T) {
					if !strings.Contains(string(body), mutation.old) {
						t.Fatalf("mutation did not match %q", mutation.old)
					}
					changed := strings.Replace(string(body), mutation.old, mutation.new, 1)
					if err := checkMLPNativeAdmission([]byte(changed), row.calls); err == nil {
						t.Fatal("broken admission accepted")
					}
				})
			}
		})
	}
}

func checkMLPNativeAdmission(body []byte, want int) error {
	if !strings.HasPrefix(string(body), "//go:build native\n") {
		return errors.New("native opt-in removed")
	}
	f, err := parser.ParseFile(token.NewFileSet(), "native.go", body, 0)
	if err != nil {
		return err
	}
	calls := 0
	var problem error
	ast.Inspect(f, func(n ast.Node) bool {
		block, ok := n.(*ast.BlockStmt)
		if !ok {
			return true
		}
		for i, statement := range block.List {
			assignment, ok := statement.(*ast.AssignStmt)
			if !ok || len(assignment.Rhs) != 1 {
				continue
			}
			call, ok := assignment.Rhs[0].(*ast.CallExpr)
			if !ok || !mlpSelector(call.Fun, "cudatest", "RequireDevice") {
				continue
			}
			calls++
			if i == 0 || !mlpInheritedRefusal(block.List[i-1]) {
				problem = fmt.Errorf("device acquisition %d lacks immediately preceding inherited-owner refusal", calls)
			}
		}
		return true
	})
	if problem != nil {
		return problem
	}
	if calls != want {
		return fmt.Errorf("device acquisition count %d, want %d", calls, want)
	}
	return nil
}

func mlpInheritedRefusal(statement ast.Stmt) bool {
	guard, ok := statement.(*ast.IfStmt)
	if !ok || guard.Else != nil {
		return false
	}
	init, ok := guard.Init.(*ast.AssignStmt)
	if !ok || init.Tok != token.DEFINE || len(init.Lhs) != 2 || len(init.Rhs) != 1 {
		return false
	}
	blank, ok := init.Lhs[0].(*ast.Ident)
	if !ok || blank.Name != "_" {
		return false
	}
	errName, ok := init.Lhs[1].(*ast.Ident)
	if !ok || errName.Name != "err" {
		return false
	}
	call, ok := init.Rhs[0].(*ast.CallExpr)
	if !ok || !mlpSelector(call.Fun, "cudalock", "Inherited") || len(call.Args) != 1 || !mlpSelector(call.Args[0], "os", "Getenv") {
		return false
	}
	condition, ok := guard.Cond.(*ast.BinaryExpr)
	if !ok || condition.Op != token.NEQ {
		return false
	}
	left, ok := condition.X.(*ast.Ident)
	if !ok || left.Name != errName.Name {
		return false
	}
	right, ok := condition.Y.(*ast.Ident)
	if !ok || right.Name != "nil" || len(guard.Body.List) != 1 {
		return false
	}
	refusal, ok := guard.Body.List[0].(*ast.ExprStmt)
	if !ok {
		return false
	}
	stop, ok := refusal.X.(*ast.CallExpr)
	return ok && mlpSelector(stop.Fun, "t", "Skipf")
}

func mlpSelector(expr ast.Expr, object, method string) bool {
	selector, ok := expr.(*ast.SelectorExpr)
	if !ok || selector.Sel.Name != method {
		return false
	}
	name, ok := selector.X.(*ast.Ident)
	return ok && name.Name == object
}

func TestMLPNativeAdmissionRefusesUnverifiedOwner(t *testing.T) {
	for _, held := range []string{"", "1", "yes", "0", "-1", "9999999999999999999999"} {
		t.Run(held, func(t *testing.T) {
			values := map[string]string{
				"GPU_LOCK_HELD": held,
				"GPU_LOCKDIR":   filepath.Join(t.TempDir(), "absent"),
			}
			_, err := cudalock.Inherited(func(key string) string { return values[key] })
			if !errors.Is(err, cudalock.ErrNotInherited) {
				t.Fatalf("unverified owner %q: %v", held, err)
			}
		})
	}
}
