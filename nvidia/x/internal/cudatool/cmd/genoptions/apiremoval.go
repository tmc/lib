package main

import (
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"sort"
	"strings"
)

// allowAPIRemoval is declared here rather than with the other flags in main
// because the write path is the only code that reads it, and a reader asking
// what the flag permits should find the check in the same file.
var allowAPIRemoval = flag.Bool("allow-api-removal", false,
	"write the output even if it removes exported declarations the current file has")

// writeGenerated writes src to file, refusing first if that would remove an
// exported declaration the file already has.
//
// The unit is the exported Go API rather than a count of options or of parsed
// lines. A count cannot tell a removal from a rename, and a rename breaks every
// caller while leaving the count intact.
//
// The comparison is against the one file being replaced, not against the
// directory. genoptions writes a single file into directories whose other
// files belong to gen-nvidia, so a whole-directory comparison would report
// every declaration it does not write as removed. A declaration that moved
// between the two generators' files would make this refuse when it should not;
// that has not happened, and a refusal is recoverable with -allow-api-removal
// while a silent removal is not.
func writeGenerated(file string, src []byte) error {
	old, err := os.ReadFile(file)
	switch {
	case os.IsNotExist(err):
		return os.WriteFile(file, src, 0o644)
	case err != nil:
		return fmt.Errorf("read %s: %w", file, err)
	}
	before, err := exportedAPI(file, old)
	if err != nil {
		return err
	}
	after, err := exportedAPI(file, src)
	if err != nil {
		return err
	}
	var removed []string
	for name := range before {
		if !after[name] {
			removed = append(removed, name)
		}
	}
	if len(removed) > 0 && !*allowAPIRemoval {
		sort.Strings(removed)
		return fmt.Errorf("%s: regenerating removes %d exported declaration(s): %s\n"+
			"\tnothing was written. If the removal is intended, re-run with -allow-api-removal",
			file, len(removed), strings.Join(removed, ", "))
	}
	return os.WriteFile(file, src, 0o644)
}

// exportedAPI returns the exported declarations in the Go source src. Methods
// are named "Receiver.Method"; everything else by its identifier.
func exportedAPI(name string, src []byte) (map[string]bool, error) {
	f, err := parser.ParseFile(token.NewFileSet(), name, src, parser.SkipObjectResolution)
	if err != nil {
		return nil, fmt.Errorf("parse %s: %w", name, err)
	}
	names := make(map[string]bool)
	for _, decl := range f.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			if !d.Name.IsExported() {
				continue
			}
			if d.Recv == nil || len(d.Recv.List) == 0 {
				names[d.Name.Name] = true
				continue
			}
			if recv := receiverName(d.Recv.List[0].Type); recv != "" {
				names[recv+"."+d.Name.Name] = true
			}
		case *ast.GenDecl:
			for _, spec := range d.Specs {
				switch s := spec.(type) {
				case *ast.TypeSpec:
					if s.Name.IsExported() {
						names[s.Name.Name] = true
					}
				case *ast.ValueSpec:
					for _, id := range s.Names {
						if id.IsExported() {
							names[id.Name] = true
						}
					}
				}
			}
		}
	}
	return names, nil
}

func receiverName(expr ast.Expr) string {
	switch t := expr.(type) {
	case *ast.StarExpr:
		return receiverName(t.X)
	case *ast.IndexExpr:
		return receiverName(t.X)
	case *ast.IndexListExpr:
		return receiverName(t.X)
	case *ast.Ident:
		if t.IsExported() {
			return t.Name
		}
	}
	return ""
}
