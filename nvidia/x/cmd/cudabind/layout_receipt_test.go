package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"strconv"
	"testing"
)

func TestGeneratedLayoutReceiptUsesActualReadback(t *testing.T) {
	m := &module{kernels: []*kernel{{name: "value", method: "value", params: []param{{cName: "value", name: "value", typ: "uint32", cType: "unsigned int", width: 4, abiAlignment: 4}}}}}
	cfg := &config{typeName: "fixture", pkg: "testfixture", output: "fixture.go"}
	body, err := generateTest(cfg, m, "compileFixture", "newFixture")
	if err != nil {
		t.Fatal(err)
	}
	file, err := parser.ParseFile(token.NewFileSet(), "fixture_test.go", body, 0)
	if err != nil {
		t.Fatal(err)
	}
	calls := make(map[string]int)
	for _, decl := range file.Decls {
		function, ok := decl.(*ast.FuncDecl)
		if !ok {
			continue
		}
		ast.Inspect(function, func(node ast.Node) bool {
			call, ok := node.(*ast.CallExpr)
			if !ok {
				return true
			}
			id, ok := call.Fun.(*ast.Ident)
			if ok && id.Name == "fixtureLogLayout" {
				if len(call.Args) != 4 {
					t.Fatal("layout receipt arguments")
				}
				for i, want := range map[int]string{0: "t", 2: "got", 3: "want"} {
					arg, ok := call.Args[i].(*ast.Ident)
					if !ok || arg.Name != want {
						t.Fatal("layout receipt replaces actual readback or Go oracle")
					}
				}
				kind, ok := call.Args[1].(*ast.BasicLit)
				if !ok || kind.Kind != token.STRING {
					t.Fatal("layout receipt kind")
				}
				name, err := strconv.Unquote(kind.Value)
				if err != nil {
					t.Fatal(err)
				}
				calls[name]++
			}
			if function.Name.Name == "fixtureLogLayout" {
				selector, ok := call.Fun.(*ast.SelectorExpr)
				if !ok || selector.Sel.Name != "Marshal" {
					return true
				}
				if len(call.Args) != 1 {
					t.Fatal("layout receipt marshal arguments")
				}
				value, ok := call.Args[0].(*ast.CompositeLit)
				if !ok || len(value.Elts) != 4 {
					t.Fatal("layout receipt fields")
				}
				for i, want := range []string{"kind", "labels", "got", "expected"} {
					field, ok := value.Elts[i].(*ast.Ident)
					if !ok || field.Name != want {
						t.Fatal("layout serializer replaces observed values")
					}
				}
			}
			return true
		})
	}
	if calls["generated-layout"] != 1 || calls["generated-layout-prerequisite"] != 1 || len(calls) != 2 {
		t.Fatalf("layout receipt calls = %v", calls)
	}
}
