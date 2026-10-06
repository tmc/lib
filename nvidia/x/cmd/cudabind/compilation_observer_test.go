package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"strings"
	"testing"
)

func TestNativeWitnessObserverUsesCompiledInput(t *testing.T) {
	m := &module{kernels: []*kernel{{name: "sentinel", method: "sentinel", params: []param{{cName: "value", name: "value", typ: "uint32", cType: "unsigned int", width: 4, abiAlignment: 4}}}}}
	cfg := &config{typeName: "fixture", pkg: "testfixture", output: "fixture.go"}
	b, err := generateTest(cfg, m, "compileFixture", "newFixture")
	if err != nil {
		t.Fatal(err)
	}
	s := string(b)
	if strings.Count(s, "var fixtureWitnessCompilationObserver ") != 1 {
		t.Fatal("observer declaration missing or repeated")
	}
	f, err := parser.ParseFile(token.NewFileSet(), "fixture_test.go", b, 0)
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	ast.Inspect(f, func(node ast.Node) bool {
		call, ok := node.(*ast.CallExpr)
		if !ok {
			return true
		}
		id, ok := call.Fun.(*ast.Ident)
		if !ok || id.Name != "observer" {
			return true
		}
		n++
		if len(call.Args) != 7 {
			t.Fatal("observer argument count")
		}
		if kind, ok := call.Args[6].(*ast.Ident); !ok || kind.Name != "output" {
			t.Fatal("observer replaced selected load output")
		}
		cm, ok := call.Args[5].(*ast.Ident)
		if !ok || cm.Name != "cm" {
			t.Fatal("observer replaced actual compilation")
		}
		device, ok := call.Args[1].(*ast.Ident)
		if !ok || device.Name != "dev" {
			t.Fatal("observer replaced actual device")
		}
		function, ok := call.Args[2].(*ast.CallExpr)
		if !ok {
			t.Fatal("observer replaced actual function")
		}
		selector, ok := function.Fun.(*ast.SelectorExpr)
		if !ok || selector.Sel.Name != "Raw" {
			t.Fatal("observer replaced loaded function")
		}
		return true
	})
	if n != 2 {
		t.Fatalf("observer calls %d, want layout plus parameter", n)
	}
	for _, want := range []string{"observer != nil", "return loader(t, dev, cm)", "witness observer requires explicit module loader", "m, output, err := fixtureLoadWitnessModule(t, dev, cm)", "k.Raw(), fixtureLayoutSource(), nil, cm, output", "kernel.Raw(), fixtureLayoutSource()+parameterSource, nil, cm, output"} {
		if !strings.Contains(s, want) {
			t.Fatalf("missing emitted contract %q", want)
		}
	}
}

func TestNativeWitnessHooksOnlyInTestOutput(t *testing.T) {
	m := &module{kernels: []*kernel{{name: "sentinel", method: "sentinel", params: []param{{cName: "value", name: "value", typ: "uint32", cType: "unsigned int", width: 4, abiAlignment: 4}}}}}
	cfg := &config{typeName: "fixture", pkg: "testfixture", output: "fixture.go", file: "fixture.cu"}
	b, _, err := generate(cfg, m)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(b), "WitnessModuleLoader") || strings.Contains(string(b), "WitnessCompilationObserver") {
		t.Fatal("qualification hooks leaked into production binding")
	}
}

func TestNativeWitnessCleanupPrecedesAllocationError(t *testing.T) {
	m := &module{kernels: []*kernel{{name: "sentinel", method: "sentinel", params: []param{{cName: "value", name: "value", typ: "uint32", cType: "unsigned int", width: 4, abiAlignment: 4}}}}}
	cfg := &config{typeName: "fixture", pkg: "testfixture", output: "fixture.go"}
	b, err := generateTest(cfg, m, "compileFixture", "newFixture")
	if err != nil {
		t.Fatal(err)
	}
	f, err := parser.ParseFile(token.NewFileSet(), "fixture_test.go", b, 0)
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	ast.Inspect(f, func(node ast.Node) bool {
		block, ok := node.(*ast.BlockStmt)
		if !ok {
			return true
		}
		for i, stmt := range block.List {
			assign, ok := stmt.(*ast.AssignStmt)
			if !ok || len(assign.Rhs) != 1 {
				continue
			}
			call, ok := assign.Rhs[0].(*ast.CallExpr)
			if !ok {
				continue
			}
			index, ok := call.Fun.(*ast.IndexExpr)
			if !ok {
				continue
			}
			selector, ok := index.X.(*ast.SelectorExpr)
			if !ok || selector.Sel.Name != "Alloc" {
				continue
			}
			n++
			if i+2 >= len(block.List) {
				t.Fatal("allocation lacks cleanup/error handling")
			}
			switch next := block.List[i+1].(type) {
			case *ast.DeferStmt:
			case *ast.ExprStmt:
				call, ok := next.X.(*ast.CallExpr)
				if !ok {
					t.Fatal("allocation cleanup call missing")
				}
				sel, ok := call.Fun.(*ast.SelectorExpr)
				if !ok || sel.Sel.Name != "Cleanup" {
					t.Fatal("allocation owner not registered")
				}
			default:
				t.Fatalf("partial allocation owner not adopted before error branch: next %T", next)
			}
			if _, ok := block.List[i+2].(*ast.IfStmt); !ok {
				t.Fatal("allocation error handling missing after owner adoption")
			}
		}
		return true
	})
	if n != 4 {
		t.Fatalf("checked %d allocation paths, want layout dummy pointer, layout reader, layout test and parameter", n)
	}
}
