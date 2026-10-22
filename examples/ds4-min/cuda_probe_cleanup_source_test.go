package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"testing"
)

func TestCUDAProbeCleanupWiring(t *testing.T) {
	path := os.Getenv("DS4_PROBE_SOURCE")
	if path == "" {
		path = "cuda_runtime_linux.go"
	}
	file, err := parser.ParseFile(token.NewFileSet(), path, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	found := 0
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || (fn.Name.Name != "probe" && fn.Name.Name != "probeModel" && fn.Name.Name != "q80Row") {
			continue
		}
		found++
		deferred := 0
		ast.Inspect(fn.Body, func(node ast.Node) bool {
			deferNode, ok := node.(*ast.DeferStmt)
			if !ok {
				return true
			}
			ast.Inspect(deferNode, func(n ast.Node) bool {
				call, ok := n.(*ast.CallExpr)
				if !ok {
					return true
				}
				id, ok := call.Fun.(*ast.Ident)
				if ok && id.Name == "finishBackendCleanup" {
					deferred++
				}
				return true
			})
			return false
		})
		if deferred != 1 {
			t.Fatalf("%s retained deferred cleanup count %d", fn.Name.Name, deferred)
		}
	}
	if found != 3 {
		t.Fatalf("probe route count %d", found)
	}
}
