package nvidia_test

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

var wrapsLine = regexp.MustCompile(`^Wraps (\w+)(?:, bound as (\w+))?\.$`)

// unlinkedCNames reports doc comments on exported declarations in the
// generated files of dir that name a C function the package wraps without
// linking it. Such prose points readers at a symbol they cannot call from Go
// when the Go symbol that wraps it is one link away.
func unlinkedCNames(dir string) ([]string, error) {
	files, err := filepath.Glob(filepath.Join(dir, "*.gen.go"))
	if err != nil {
		return nil, err
	}
	fset := token.NewFileSet()
	var parsed []*ast.File
	wrapped := make(map[string]bool)
	for _, name := range files {
		f, err := parser.ParseFile(fset, name, nil, parser.ParseComments)
		if err != nil {
			return nil, err
		}
		parsed = append(parsed, f)
		for _, cg := range f.Comments {
			for _, line := range strings.Split(cg.Text(), "\n") {
				if m := wrapsLine.FindStringSubmatch(line); m != nil {
					wrapped[m[1]] = true
				}
			}
		}
	}
	if len(wrapped) == 0 {
		return nil, nil
	}
	var names []string
	for name := range wrapped {
		names = append(names, regexp.QuoteMeta(name))
	}
	sort.Slice(names, func(i, j int) bool { return len(names[i]) > len(names[j]) })
	// A name inside a doc link or reached through a selector is not prose.
	cname := regexp.MustCompile(`(^|[^\w\[.])(` + strings.Join(names, "|") + `)\b`)

	var bad []string
	check := func(decl string, cg *ast.CommentGroup) {
		if cg == nil || !ast.IsExported(decl) {
			return
		}
		for _, line := range strings.Split(cg.Text(), "\n") {
			if wrapsLine.MatchString(line) {
				continue
			}
			for _, m := range cname.FindAllStringSubmatch(line, -1) {
				if m[2] == decl {
					continue // the Go name happens to be the C name
				}
				pos := fset.Position(cg.Pos())
				bad = append(bad, fmt.Sprintf("%s:%d: %s names %s", filepath.Base(pos.Filename), pos.Line, decl, m[2]))
			}
		}
	}
	for _, f := range parsed {
		for _, d := range f.Decls {
			switch d := d.(type) {
			case *ast.FuncDecl:
				if strings.HasPrefix(d.Name.Name, "Has") {
					continue // "HasX reports whether cuX resolved" names the symbol on purpose
				}
				check(d.Name.Name, d.Doc)
			case *ast.GenDecl:
				for _, s := range d.Specs {
					switch s := s.(type) {
					case *ast.TypeSpec:
						doc := s.Doc
						if doc == nil && len(d.Specs) == 1 {
							doc = d.Doc
						}
						check(s.Name.Name, doc)
						if st, ok := s.Type.(*ast.StructType); ok {
							for _, field := range st.Fields.List {
								for _, n := range field.Names {
									check(n.Name, field.Doc)
									check(n.Name, field.Comment)
								}
							}
						}
					case *ast.ValueSpec:
						for _, n := range s.Names {
							check(n.Name, s.Doc)
							check(n.Name, s.Comment)
						}
					}
				}
			}
		}
	}
	return bad, nil
}

func TestDocLinksWrappedFunctions(t *testing.T) {
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if !e.IsDir() || e.Name() == "x" || e.Name() == "internal" {
			continue
		}
		bad, err := unlinkedCNames(e.Name())
		if err != nil {
			t.Fatalf("%s: %v", e.Name(), err)
		}
		for _, b := range bad {
			t.Errorf("%s/%s", e.Name(), b)
		}
	}
}

func TestUnlinkedCNames(t *testing.T) {
	dir := t.TempDir()
	src := `package p

// Frob does it. See cuFrob, [Frob], and [Thing.cuFrob].
//
// Wraps cuFrob.
func Frob() {}

// HasFrob reports whether cuFrob resolved.
func HasFrob() bool { return true }

// cuFrob in an unexported doc is not godoc.
func frob() {}

// Thing is a thing.
type Thing struct {
	// N counts calls to cuFrob.
	N int
}
`
	if err := os.WriteFile(filepath.Join(dir, "p.gen.go"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	bad, err := unlinkedCNames(dir)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"p.gen.go:3: Frob names cuFrob", "p.gen.go:16: N names cuFrob"}
	if strings.Join(bad, "\n") != strings.Join(want, "\n") {
		t.Errorf("unlinkedCNames = %q, want %q", bad, want)
	}
}
