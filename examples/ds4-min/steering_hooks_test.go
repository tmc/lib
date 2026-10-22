package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strings"
	"testing"
)

// This is a host source-placement check, not a device numerical gate.
func batchSteeringHooks(source []byte) (attention, ffn bool, err error) {
	f, err := parser.ParseFile(token.NewFileSet(), "cuda_session_linux.go", source, 0)
	if err != nil {
		return false, false, err
	}
	ast.Inspect(f, func(node ast.Node) bool {
		call, ok := node.(*ast.CallExpr)
		if !ok || len(call.Args) != 4 {
			return true
		}
		fn, ok := call.Fun.(*ast.SelectorExpr)
		if !ok || fn.Sel.Name != "projectRows" {
			return true
		}
		row, ok := call.Args[0].(*ast.CallExpr)
		if !ok {
			return true
		}
		slice, ok := row.Fun.(*ast.SelectorExpr)
		if !ok || slice.Sel.Name != "Slice" {
			return true
		}
		buffer, ok := slice.X.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		count, ok := call.Args[2].(*ast.Ident)
		if !ok || count.Name != "n" {
			return true
		}
		scale, ok := call.Args[3].(*ast.SelectorExpr)
		if !ok {
			return true
		}
		if buffer.Sel.Name == "attnOut" && scale.Sel.Name == "attention" {
			attention = true
		}
		if buffer.Sel.Name == "ffnOut" && scale.Sel.Name == "ffn" {
			ffn = true
		}
		return true
	})
	return attention, ffn, nil
}

func TestSteeringBatchHookSourceMutations(t *testing.T) {
	source, err := os.ReadFile("cuda_session_linux.go")
	if err != nil {
		t.Fatal(err)
	}
	attention, ffn, err := batchSteeringHooks(source)
	if err != nil || !attention || !ffn {
		t.Fatalf("batch attention %v FFN %v error %v", attention, ffn, err)
	}
	for _, buffer := range []string{"attnOut", "ffnOut"} {
		old := "projectRows(d." + buffer
		mutant := strings.Replace(string(source), old, "removedProjection(d."+buffer, 1)
		a, f, err := batchSteeringHooks([]byte(mutant))
		if err != nil {
			t.Fatal(err)
		}
		if a && f {
			t.Fatalf("missing %s hook survived source gate", buffer)
		}
	}
}
