package main

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strings"
	"testing"
)

// Scalar addresses hidden in uintptr must be pinned across the FFI call.
// KeepAlive alone proves reachability, not immobility of Go stack storage.
func visionScalarPins(source []byte) error {
	f, err := parser.ParseFile(token.NewFileSet(), "cuda_flashvision_linux.go", source, 0)
	if err != nil {
		return err
	}
	var body *ast.BlockStmt
	for _, decl := range f.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Name.Name == "flashVisionMatmul" {
			body = fn.Body
		}
	}
	if body == nil {
		return fmt.Errorf("matmul implementation absent")
	}
	pinned := map[string]bool{}
	unpin := false
	gemm := false
	for _, stmt := range body.List {
		if d, ok := stmt.(*ast.DeferStmt); ok {
			if s, ok := d.Call.Fun.(*ast.SelectorExpr); ok && s.Sel.Name == "Unpin" {
				if id, ok := s.X.(*ast.Ident); ok && id.Name == "pin" {
					unpin = true
				}
			}
		}
		var call *ast.CallExpr
		switch s := stmt.(type) {
		case *ast.ExprStmt:
			call, _ = s.X.(*ast.CallExpr)
		case *ast.AssignStmt:
			if len(s.Rhs) == 1 {
				call, _ = s.Rhs[0].(*ast.CallExpr)
			}
		}
		if call == nil {
			continue
		}
		s, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			continue
		}
		id, ok := s.X.(*ast.Ident)
		if !ok {
			continue
		}
		if id.Name == "pin" && s.Sel.Name == "Pin" && len(call.Args) == 1 {
			if u, ok := call.Args[0].(*ast.UnaryExpr); ok && u.Op == token.AND {
				if v, ok := u.X.(*ast.Ident); ok {
					pinned[v.Name] = true
				}
			}
		}
		if s.Sel.Name == "GemmEx" {
			gemm = true
			if !pinned["alpha"] || !pinned["beta"] || !unpin {
				return fmt.Errorf("GemmEx scalars are not pinned with deferred release")
			}
		}
	}
	if !gemm {
		return fmt.Errorf("GemmEx call absent")
	}
	return nil
}

func TestFlashVisionScalarLifetime(t *testing.T) {
	source, err := os.ReadFile("cuda_flashvision_linux.go")
	if err != nil {
		t.Fatal(err)
	}
	if err := visionScalarPins(source); err != nil {
		t.Fatal(err)
	}
	for _, line := range []string{"pin.Pin(&alpha)", "pin.Pin(&beta)", "defer pin.Unpin()"} {
		if strings.Count(string(source), line) != 1 {
			t.Fatalf("nonunique ownership site %s", line)
		}
		mutant := strings.Replace(string(source), line, "", 1)
		if err := visionScalarPins([]byte(mutant)); err == nil {
			t.Fatalf("unpinned ownership mutant survived: %s", line)
		} else {
			t.Logf("actual ownership source-site removal %q rejected: %v", line, err)
		}
	}
}
