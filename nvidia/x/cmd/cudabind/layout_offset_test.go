package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"strconv"
	"strings"
	"testing"
)

func TestGeneratedLayoutOffsetsMeasureGuardedObjects(t *testing.T) {
	m := &module{records: []*record{
		{cName: "Point", goName: "point", size: 16, align: 8, fields: []field{{cName: "x", goName: "X", typ: "float32", size: 4}, {cName: "y", goName: "Y", typ: "float64", off: 8, size: 8}}},
		{cName: "Pair", goName: "pair", size: 32, align: 16, fields: []field{{cName: "a", goName: "A", typ: "int32", size: 4}, {cName: "b", goName: "B", typ: "[4]uint32", off: 16, size: 16}}},
		{cName: "Opaque", goName: "opaque", size: 8, align: 1, opaque: true},
		{cName: "OnlyPadding", goName: "onlyPadding", size: 8, align: 1, fields: []field{{goName: "_", typ: "[8]byte", size: 8}}},
	}}
	cfg := &config{typeName: "fixture", pkg: "testfixture", output: "fixture.go"}
	body, err := generateTest(cfg, m, "compileFixture", "newFixture")
	if err != nil {
		t.Fatal(err)
	}
	file, err := parser.ParseFile(token.NewFileSet(), "fixture_test.go", body, 0)
	if err != nil {
		t.Fatal(err)
	}
	var kernel string
	for _, decl := range file.Decls {
		f, ok := decl.(*ast.FuncDecl)
		if !ok || f.Name.Name != "fixtureLayoutSource" {
			continue
		}
		if len(f.Body.List) != 1 {
			t.Fatal("layout source body")
		}
		ret, ok := f.Body.List[0].(*ast.ReturnStmt)
		if !ok || len(ret.Results) != 1 {
			t.Fatal("layout source return")
		}
		literal, ok := ret.Results[0].(*ast.BasicLit)
		if !ok {
			t.Fatal("layout source literal")
		}
		kernel, err = strconv.Unquote(literal.Value)
		if err != nil {
			t.Fatal(err)
		}
	}
	if kernel == "" {
		t.Fatal("layout source missing")
	}
	for i, name := range []string{"Point", "Pair"} {
		guard := "static_assert(__is_trivially_constructible(" + name + ") && __is_trivially_copyable(" + name + ") && __is_standard_layout(" + name + "), \"cudabind layout requires trivial default construction and standard layout\");"
		object := "cudabind_layout_object_" + strconv.Itoa(i)
		declaration := name + " " + object + "{};"
		if strings.Count(kernel, guard) != 1 || strings.Count(kernel, declaration) != 1 || strings.Index(kernel, guard) > strings.Index(kernel, declaration) {
			t.Fatalf("%s measurement lacks prior record guard", name)
		}
		fields := []string{"x", "y"}
		if name == "Pair" {
			fields = []string{"a", "b"}
		}
		for _, field := range fields {
			expr := "(reinterpret_cast<unsigned long long>(&" + object + "." + field + ") - reinterpret_cast<unsigned long long>(&" + object + "))"
			if strings.Count(kernel, expr) != 1 {
				t.Fatalf("%s.%s does not measure its own object addresses", name, field)
			}
		}
	}
	if strings.Count(kernel, "static_assert(") != 2 || strings.Count(kernel, "{};") != 2 {
		t.Fatal("sizeof-only record acquired an object or constructor guard")
	}
	if strings.Contains(kernel, "__builtin_offsetof") || strings.Contains(kernel, "#include") || strings.Contains(kernel, "(&Opaque") || strings.Contains(kernel, "(&OnlyPadding") {
		t.Fatal("unexpected offset intrinsic, header or sizeof-only object")
	}
}
