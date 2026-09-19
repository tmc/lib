package cudaroofline

import (
	"bytes"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"os"
	"sort"
	"strings"
	"testing"
)

func TestRecordRoundTrips(t *testing.T) {
	m, js := goldenJudgments(t)
	want := NewRecord(m, js)

	var buf bytes.Buffer
	if err := want.Encode(&buf); err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if !bytes.HasSuffix(buf.Bytes(), []byte("\n")) {
		t.Error("encoded record does not end in a newline")
	}
	got, err := DecodeRecord(&buf)
	if err != nil {
		t.Fatalf("DecodeRecord: %v", err)
	}
	if got.Device != want.Device {
		t.Errorf("device:\n got %+v\nwant %+v", got.Device, want.Device)
	}
	if len(got.Triples) != len(want.Triples) {
		t.Fatalf("got %d triples, want %d", len(got.Triples), len(want.Triples))
	}
	for i := range want.Triples {
		if got.Triples[i] != want.Triples[i] {
			t.Errorf("triple %d:\n got %+v\nwant %+v", i, got.Triples[i], want.Triples[i])
		}
	}
}

// TestRecordVersionRefusedWhenUnknown: a consumer that reads the fields it
// recognizes out of a record it does not understand is the way a floor quietly
// changes meaning across a module boundary.
func TestRecordVersionRefusedWhenUnknown(t *testing.T) {
	const future = `{"version":99,"device":{"architecture":"sm_121","residency":"device"},"triples":[]}`
	if _, err := DecodeRecord(strings.NewReader(future)); err == nil {
		t.Error("a record from a future version was accepted")
	} else if !strings.Contains(err.Error(), "version") {
		t.Errorf("error does not mention the version: %v", err)
	}
}

func TestRecordRejectsUnnamedDeviceAndUnknownVerdict(t *testing.T) {
	tests := []struct {
		name, json, want string
	}{
		{
			"no device",
			`{"version":1,"device":{"architecture":"","residency":""},"triples":[{"name":"x","basis":"memory_only","verdict":"accepted"}]}`,
			"names no device",
		},
		{
			"no triples",
			`{"version":1,"device":{"architecture":"sm_121","residency":"device"},"triples":[]}`,
			"no triples",
		},
		{
			"unknown verdict",
			`{"version":1,"device":{"architecture":"sm_121","residency":"device"},"triples":[{"name":"x","basis":"memory_only","verdict":"probably_fine"}]}`,
			"unknown verdict",
		},
		{
			"unknown basis",
			`{"version":1,"device":{"architecture":"sm_121","residency":"device"},"triples":[{"name":"x","basis":"vibes","verdict":"accepted"}]}`,
			"unknown floor basis",
		},
		{
			"unknown field",
			`{"version":1,"device":{"architecture":"sm_121","residency":"device"},"triples":[],"floor_override_ns":0}`,
			"unknown field",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := DecodeRecord(strings.NewReader(tt.json))
			if err == nil {
				t.Fatal("accepted")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error %q does not contain %q", err, tt.want)
			}
		})
	}
}

// TestNoOTLPDependencyInModule: the transport half of the draft belongs to
// another module. An OTLP or gRPC import here would put a vendor-neutral wire
// format's dependencies inside the NVIDIA bindings module, which has a
// three-dependency budget, and would make the encoding this package defines
// unreadable by anything that does not share them.
func TestNoOTLPDependencyInModule(t *testing.T) {
	for _, f := range parsePackage(t) {
		for _, imp := range f.Imports {
			path := strings.Trim(imp.Path.Value, `"`)
			for _, banned := range []string{"otlp", "opentelemetry", "grpc", "prometheus", "gpufleet"} {
				if strings.Contains(strings.ToLower(path), banned) {
					t.Errorf("%s imports %q", f.Name.Name, path)
				}
			}
		}
	}
}

// TestExportedSurfaceMatchesGoldenBaseline records the exported surface so an
// addition is a deliberate line in a diff. Two consumers in other modules read
// the encoding this package defines and cannot be recompiled against a change
// here, so a silently renamed field is a break they discover at runtime.
func TestExportedSurfaceMatchesGoldenBaseline(t *testing.T) {
	const baseline = "testdata/exported-surface.txt"
	got := strings.Join(exportedSurface(t), "\n") + "\n"

	if *update {
		if err := os.WriteFile(baseline, []byte(got), 0o644); err != nil {
			t.Fatalf("writing %s: %v", baseline, err)
		}
		t.Logf("wrote %s", baseline)
		return
	}
	want, err := os.ReadFile(baseline)
	if err != nil {
		t.Fatalf("reading %s: %v (regenerate with -update)", baseline, err)
	}
	if got != string(want) {
		t.Errorf("exported surface changed; review the diff and regenerate with -update:\n%s",
			surfaceDiff(strings.Split(string(want), "\n"), strings.Split(got, "\n")))
	}
}

func parsePackage(t *testing.T) []*ast.File {
	t.Helper()
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, ".", func(fi os.FileInfo) bool {
		return !strings.HasSuffix(fi.Name(), "_test.go")
	}, parser.ParseComments)
	if err != nil {
		t.Fatalf("parsing the package: %v", err)
	}
	pkg, ok := pkgs["cudaroofline"]
	if !ok {
		t.Fatal("package cudaroofline not found in the working directory")
	}
	names := make([]string, 0, len(pkg.Files))
	for name := range pkg.Files {
		names = append(names, name)
	}
	sort.Strings(names)
	files := make([]*ast.File, 0, len(names))
	for _, n := range names {
		files = append(files, pkg.Files[n])
	}
	return files
}

// exportedSurface lists every exported declaration, one per line, sorted. Struct
// fields are listed with their types because a consumer decoding this package's
// record depends on them.
func exportedSurface(t *testing.T) []string {
	t.Helper()
	fset := token.NewFileSet()
	var lines []string
	for _, f := range parsePackage(t) {
		for _, d := range f.Decls {
			switch d := d.(type) {
			case *ast.FuncDecl:
				if !d.Name.IsExported() {
					continue
				}
				recv := ""
				if d.Recv != nil && len(d.Recv.List) == 1 {
					recv = "(" + typeString(fset, d.Recv.List[0].Type) + ") "
					if !isExportedReceiver(d.Recv.List[0].Type) {
						continue
					}
				}
				lines = append(lines, "func "+recv+d.Name.Name+typeString(fset, d.Type)[4:])
			case *ast.GenDecl:
				for _, spec := range d.Specs {
					switch s := spec.(type) {
					case *ast.TypeSpec:
						if !s.Name.IsExported() {
							continue
						}
						lines = append(lines, "type "+s.Name.Name+" "+typeKind(s.Type))
						if st, ok := s.Type.(*ast.StructType); ok {
							for _, fld := range st.Fields.List {
								for _, n := range fld.Names {
									if n.IsExported() {
										lines = append(lines, "  field "+s.Name.Name+"."+n.Name+" "+typeString(fset, fld.Type))
									}
								}
							}
						}
					case *ast.ValueSpec:
						for _, n := range s.Names {
							if n.IsExported() {
								lines = append(lines, declKind(d.Tok)+" "+n.Name)
							}
						}
					}
				}
			}
		}
	}
	sort.Strings(lines)
	return lines
}

func isExportedReceiver(e ast.Expr) bool {
	if st, ok := e.(*ast.StarExpr); ok {
		e = st.X
	}
	id, ok := e.(*ast.Ident)
	return ok && id.IsExported()
}

func typeKind(e ast.Expr) string {
	switch e.(type) {
	case *ast.StructType:
		return "struct"
	case *ast.InterfaceType:
		return "interface"
	}
	return typeString(token.NewFileSet(), e)
}

func declKind(tok token.Token) string {
	if tok == token.CONST {
		return "const"
	}
	return "var"
}

func typeString(fset *token.FileSet, e ast.Expr) string {
	var buf bytes.Buffer
	if err := printer.Fprint(&buf, fset, e); err != nil {
		return "?"
	}
	return strings.Join(strings.Fields(buf.String()), " ")
}

func surfaceDiff(want, got []string) string {
	in := func(s string, list []string) bool {
		for _, v := range list {
			if v == s {
				return true
			}
		}
		return false
	}
	var b strings.Builder
	for _, w := range want {
		if w != "" && !in(w, got) {
			b.WriteString("- " + w + "\n")
		}
	}
	for _, g := range got {
		if g != "" && !in(g, want) {
			b.WriteString("+ " + g + "\n")
		}
	}
	return b.String()
}
