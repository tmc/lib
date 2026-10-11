package cudamega_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strings"
	"testing"
)

func TestFinitePublicABI(t *testing.T) {
	paths, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range paths {
		if strings.HasSuffix(path, "_test.go") {
			continue
		}
		file, err := parser.ParseFile(token.NewFileSet(), path, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, name := range legacyDeclarations(file) {
			t.Errorf("%s declares obsolete ring-queue ABI %s", path, name)
		}
	}
}

func legacyDeclarations(file *ast.File) []string {
	var names []string
	check := func(name string) {
		switch name {
		case "Task", "Dispatcher", "NewDispatcher", "MegaKernelTemplate":
			names = append(names, name)
		}
	}
	for _, decl := range file.Decls {
		switch decl := decl.(type) {
		case *ast.FuncDecl:
			if decl.Recv == nil {
				check(decl.Name.Name)
			}
		case *ast.GenDecl:
			for _, spec := range decl.Specs {
				switch spec := spec.(type) {
				case *ast.TypeSpec:
					check(spec.Name.Name)
				case *ast.ValueSpec:
					for _, name := range spec.Names {
						check(name.Name)
					}
				}
			}
		}
	}
	return names
}

func TestLegacyDeclarationsRefused(t *testing.T) {
	tests := []struct {
		name, source, want string
	}{
		{"finite program", "type Program struct{}", ""},
		{"task", "type Task struct{ Arg0 uint64 }", "Task"},
		{"dispatcher", "type Dispatcher struct{}", "Dispatcher"},
		{"constructor", "func NewDispatcher() {}", "NewDispatcher"},
		{"template", "const MegaKernelTemplate = `empty loop`", "MegaKernelTemplate"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			file, err := parser.ParseFile(token.NewFileSet(), "control.go", "package cudamega\n"+tt.source, 0)
			if err != nil {
				t.Fatal(err)
			}
			if got := strings.Join(legacyDeclarations(file), ","); got != tt.want {
				t.Fatalf("legacy declarations = %q, want %q", got, tt.want)
			}
		})
	}
}
