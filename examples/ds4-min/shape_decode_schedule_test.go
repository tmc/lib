package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"slices"
	"testing"
)

// This checks source wiring, not executed CUDA arithmetic. The corresponding
// inverse kernel body is independently tied to the pinned RoPE fixture.
func TestShapeDecodeAttentionOrder(t *testing.T) {
	path := "cuda_shape_decode_linux.go"
	if p := os.Getenv("DS4_SHAPE_DECODE_TEST_SOURCE"); p != "" {
		path = p
	}
	f, err := parser.ParseFile(token.NewFileSet(), path, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var calls []string
	ast.Inspect(f, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		id, ok := call.Fun.(*ast.Ident)
		if ok && slices.Contains([]string{"encodeShapeAttentionExternal", "encodeShapeAttentionInverseRoPE", "encodeShapeAttentionOutput"}, id.Name) {
			calls = append(calls, id.Name)
		}
		return true
	})
	want := []string{"encodeShapeAttentionExternal", "encodeShapeAttentionInverseRoPE", "encodeShapeAttentionOutput"}
	if !slices.Equal(calls, want) {
		t.Fatalf("DECODE_ATTENTION_ORDER got=%v want=%v", calls, want)
	}
}

func TestShapeInverseRoPEArgument(t *testing.T) {
	path := "cuda_shape_rope_linux.go"
	if p := os.Getenv("DS4_SHAPE_ROPE_TEST_SOURCE"); p != "" {
		path = p
	}
	f, err := parser.ParseFile(token.NewFileSet(), path, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	launches := 0
	for _, decl := range f.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Name.Name != "encodeShapeAttentionInverseRoPE" {
			continue
		}
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			sel, ok := call.Fun.(*ast.SelectorExpr)
			if !ok || sel.Sel.Name != "Launch" {
				return true
			}
			launches++
			if len(call.Args) != 15 {
				t.Fatal("INVERSE_ROPE_ARGUMENT_COUNT")
			}
			inverse, ok := call.Args[8].(*ast.CallExpr)
			if !ok || len(inverse.Args) != 1 {
				t.Fatal("INVERSE_ROPE_DIRECTION")
			}
			id, ok := inverse.Fun.(*ast.Ident)
			if !ok || id.Name != "int32" {
				t.Fatal("INVERSE_ROPE_DIRECTION_TYPE")
			}
			value, ok := inverse.Args[0].(*ast.BasicLit)
			if !ok || value.Kind != token.INT || value.Value != "1" {
				t.Fatal("INVERSE_ROPE_DIRECTION")
			}
			return true
		})
	}
	if launches != 1 {
		t.Fatalf("INVERSE_ROPE_LAUNCH_COUNT got=%d", launches)
	}
}
