package main

import (
	"fmt"
	"go/ast"
	"go/constant"
	"go/parser"
	"go/token"
	"go/types"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"golang.org/x/tools/go/packages"
)

// stubRoot holds the declaration-only stand-ins the fixtures type-check
// against in GOPATH mode.
const stubRoot = "testdata/src"

// TestStubsMatchRealPackages keeps the stand-ins honest. A fixture proves an
// analyzer fires only against the signatures it was written for; once the
// real API moves on, a stub that still declares the old one lets every test
// pass while the analyzer checks nothing. Every exported declaration of a
// stub must therefore exist in the real package with the same signature,
// parameter names included, since the analyzers match on them.
func TestStubsMatchRealPackages(t *testing.T) {
	paths := stubPackages(t)
	if len(paths) == 0 {
		t.Fatalf("no stub packages under %s", stubRoot)
	}

	gopath, err := filepath.Abs("testdata")
	if err != nil {
		t.Fatal(err)
	}
	stubs := load(t, &packages.Config{
		Mode: packages.NeedSyntax | packages.NeedTypesInfo,
		Dir:  stubRoot,
		Env:  append(os.Environ(), "GOPATH="+gopath, "GO111MODULE=off", "GOWORK=off", "GOPROXY=off"),
	}, paths)
	for _, p := range stubs {
		for _, e := range p.Errors {
			t.Errorf("stub %s: %v", p.PkgPath, e)
		}
	}
	if t.Failed() {
		t.FailNow()
	}
	// The bindings target Linux, and some, such as cufile, build only there.
	real := load(t, &packages.Config{
		Fset: token.NewFileSet(),
		Env:  append(os.Environ(), "GOOS=linux"),
	}, paths)

	for _, path := range paths {
		s, r := stubs[path], real[path]
		if r == nil || len(r.Errors) > 0 {
			t.Errorf("%s: stub has no real package", path)
			continue
		}
		for _, msg := range stubDrift(s.Types, r.Types) {
			t.Errorf("%s.%s", s.Types.Name(), msg)
		}
		for _, msg := range docDrift(s, r) {
			t.Errorf("%s.%s", s.Types.Name(), msg)
		}
	}
}

// stubPackages returns the import path of every stub package.
func stubPackages(t *testing.T) []string {
	t.Helper()
	var paths []string
	root := filepath.Join(stubRoot, strings.TrimSuffix(nvidiaPrefix, "/"))
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(path, ".go") {
			return err
		}
		rel, err := filepath.Rel(stubRoot, filepath.Dir(path))
		if err != nil {
			return err
		}
		if p := filepath.ToSlash(rel); !slices.Contains(paths, p) {
			paths = append(paths, p)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return paths
}

func load(t *testing.T, cfg *packages.Config, paths []string) map[string]*packages.Package {
	t.Helper()
	cfg.Mode |= packages.NeedName | packages.NeedTypes
	pkgs, err := packages.Load(cfg, paths...)
	if err != nil {
		t.Fatalf("loading %s: %v", strings.Join(paths, " "), err)
	}
	out := make(map[string]*packages.Package)
	for _, p := range pkgs {
		out[p.PkgPath] = p
	}
	return out
}

// stubDrift returns a line for each exported declaration of stub that real
// lacks or declares differently.
func stubDrift(stub, real *types.Package) []string {
	var out []string
	for _, name := range stub.Scope().Names() {
		s := stub.Scope().Lookup(name)
		if !s.Exported() {
			continue
		}
		r := real.Scope().Lookup(name)
		if r == nil {
			out = append(out, name+": not in the real package")
			continue
		}
		if msg := declDrift(s, r); msg != "" {
			out = append(out, name+": "+msg)
		}
		sn, ok := s.Type().(*types.Named)
		if _, isType := s.(*types.TypeName); !isType || !ok {
			continue
		}
		rn, _ := r.Type().(*types.Named)
		for m := range sn.Methods() {
			if !m.Exported() {
				continue
			}
			var rm *types.Func
			if rn != nil {
				for c := range rn.Methods() {
					if c.Name() == m.Name() {
						rm = c
					}
				}
			}
			switch {
			case rm == nil:
				out = append(out, name+"."+m.Name()+": not in the real package")
			case isPtrRecv(m) != isPtrRecv(rm):
				out = append(out, fmt.Sprintf("%s.%s: pointer receiver is %v, real is %v", name, m.Name(), isPtrRecv(m), isPtrRecv(rm)))
			default:
				if st, rt := typeString(m.Type()), typeString(rm.Type()); st != rt {
					out = append(out, fmt.Sprintf("%s.%s: stub\n\t%s\nreal\n\t%s", name, m.Name(), st, rt))
				}
			}
		}
	}
	return out
}

// declDrift describes how the declaration s differs from r, or returns "".
func declDrift(s, r types.Object) string {
	switch s := s.(type) {
	case *types.Func:
		if _, ok := r.(*types.Func); !ok {
			return fmt.Sprintf("func in the stub, %T in the real package", r)
		}
		if st, rt := typeString(s.Type()), typeString(r.Type()); st != rt {
			return fmt.Sprintf("stub\n\t%s\nreal\n\t%s", st, rt)
		}
	case *types.Const:
		r, ok := r.(*types.Const)
		if !ok {
			return fmt.Sprintf("const in the stub, %T in the real package", r)
		}
		if st, rt := typeString(s.Type()), typeString(r.Type()); st != rt {
			return fmt.Sprintf("type %s, real %s", st, rt)
		}
		if !constant.Compare(s.Val(), token.EQL, r.Val()) {
			return fmt.Sprintf("value %s, real %s", s.Val(), r.Val())
		}
	case *types.Var:
		if _, ok := r.(*types.Var); !ok {
			return fmt.Sprintf("var in the stub, %T in the real package", r)
		}
		if st, rt := typeString(s.Type()), typeString(r.Type()); st != rt {
			return fmt.Sprintf("type %s, real %s", st, rt)
		}
	case *types.TypeName:
		r, ok := r.(*types.TypeName)
		if !ok {
			return fmt.Sprintf("type in the stub, %T in the real package", r)
		}
		if s.IsAlias() != r.IsAlias() {
			return fmt.Sprintf("alias is %v, real is %v", s.IsAlias(), r.IsAlias())
		}
		return underlyingDrift(s.Type().Underlying(), r.Type().Underlying())
	}
	return ""
}

// underlyingDrift compares the underlying types of a stub type and its real
// counterpart. A stub struct may declare a subset of the real fields.
func underlyingDrift(s, r types.Type) string {
	ss, sok := s.(*types.Struct)
	rs, rok := r.(*types.Struct)
	switch {
	case sok && rok:
		for sf := range ss.Fields() {
			var rf *types.Var
			for f := range rs.Fields() {
				if f.Name() == sf.Name() {
					rf = f
				}
			}
			if rf == nil {
				return "field " + sf.Name() + " not in the real struct"
			}
			if st, rt := typeString(sf.Type()), typeString(rf.Type()); st != rt {
				return fmt.Sprintf("field %s is %s, real %s", sf.Name(), st, rt)
			}
		}
		return ""
	case sok != rok:
		return fmt.Sprintf("underlying type %s, real %s", typeString(s), typeString(r))
	}
	if _, ok := s.(*types.Interface); ok {
		if _, ok := r.(*types.Interface); ok {
			return ""
		}
	}
	if st, rt := typeString(s), typeString(r); st != rt {
		return fmt.Sprintf("underlying type %s, real %s", st, rt)
	}
	return ""
}

// docDrift returns a line for each function of stub whose doc comment is not
// part of the real one. legacycupti reads doc comments, so a stub's must be
// the real text, though it may carry only the paragraphs a check reads.
func docDrift(stub, real *packages.Package) []string {
	var out []string
	for _, f := range stub.Syntax {
		for _, decl := range f.Decls {
			fd, ok := decl.(*ast.FuncDecl)
			if !ok || fd.Doc == nil {
				continue
			}
			s, _ := stub.TypesInfo.Defs[fd.Name].(*types.Func)
			if s == nil {
				continue
			}
			r := realFunc(real.Types, s)
			if r == nil {
				continue // reported by stubDrift
			}
			doc := realDoc(real.Fset, r)
			if !strings.Contains(squash(doc), squash(fd.Doc.Text())) {
				out = append(out, fmt.Sprintf("%s: stub doc comment\n\t%s\nis not in the real one\n\t%s", s.Name(), fd.Doc.Text(), doc))
			}
		}
	}
	return out
}

// realFunc returns the function or method of real that s stands in for.
func realFunc(real *types.Package, s *types.Func) *types.Func {
	recv := recvNamed(s)
	if recv == nil {
		f, _ := real.Scope().Lookup(s.Name()).(*types.Func)
		return f
	}
	tn, _ := real.Scope().Lookup(recv.Obj().Name()).(*types.TypeName)
	if tn == nil {
		return nil
	}
	n, _ := tn.Type().(*types.Named)
	if n == nil {
		return nil
	}
	for m := range n.Methods() {
		if m.Name() == s.Name() {
			return m
		}
	}
	return nil
}

// realDoc returns the doc comment of fn, read from the source file its
// export data names.
func realDoc(fset *token.FileSet, fn *types.Func) string {
	pos := fset.Position(fn.Pos())
	fset = token.NewFileSet()
	f, err := parser.ParseFile(fset, pos.Filename, nil, parser.ParseComments)
	if err != nil {
		return ""
	}
	for _, decl := range f.Decls {
		fd, ok := decl.(*ast.FuncDecl)
		if ok && fd.Doc != nil && fset.Position(fd.Name.Pos()).Line == pos.Line {
			return fd.Doc.Text()
		}
	}
	return ""
}

func squash(s string) string { return strings.Join(strings.Fields(s), " ") }

func isPtrRecv(m *types.Func) bool {
	_, ok := m.Type().(*types.Signature).Recv().Type().(*types.Pointer)
	return ok
}

// typeString writes t with packages qualified by their full path, so that a
// stub and a real package with the same path print the same.
func typeString(t types.Type) string {
	return types.TypeString(t, func(p *types.Package) string { return p.Path() })
}
