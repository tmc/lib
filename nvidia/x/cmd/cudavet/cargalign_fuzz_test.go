package main

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudakernel"
	"golang.org/x/tools/go/analysis"
)

// The declaration-only fixture supplies Go type identity. All device offsets
// and rejection decisions below come from the actual cudakernel.NewSchema.
type cargalignImporter struct{ pkg *types.Package }

func (i cargalignImporter) Import(path string) (*types.Package, error) {
	if path != cudakernelPath {
		return nil, fmt.Errorf("unexpected import %q", path)
	}
	return i.pkg, nil
}
func cargalignTypes(t testing.TB) *types.Package {
	t.Helper()
	body, err := os.ReadFile("testdata/src/github.com/tmc/lib/nvidia/x/cudakernel/abi.go")
	if err != nil {
		t.Fatal(err)
	}
	fs := token.NewFileSet()
	f, err := parser.ParseFile(fs, "abi.go", body, 0)
	if err != nil {
		t.Fatal(err)
	}
	p, err := new(types.Config).Check(cudakernelPath, fs, []*ast.File{f}, nil)
	if err != nil {
		t.Fatal(err)
	}
	return p
}
func cargalignPass(t testing.TB, pkg *types.Package, source string) (*analysis.Pass, *[]analysis.Diagnostic) {
	t.Helper()
	fs := token.NewFileSet()
	file, err := parser.ParseFile(fs, "layout.go", source, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	info := &types.Info{Types: map[ast.Expr]types.TypeAndValue{}, Defs: map[*ast.Ident]types.Object{}, Uses: map[*ast.Ident]types.Object{}, Selections: map[*ast.SelectorExpr]*types.Selection{}}
	sizes := types.SizesFor("gc", "amd64")
	conf := types.Config{Importer: cargalignImporter{pkg}, Sizes: sizes}
	checked, err := conf.Check("fixture", fs, []*ast.File{file}, info)
	if err != nil {
		t.Fatal(err)
	}
	var diagnostics []analysis.Diagnostic
	return &analysis.Pass{Analyzer: cargalignAnalyzer, Fset: fs, Files: []*ast.File{file}, Pkg: checked, TypesInfo: info, TypesSizes: sizes, Report: func(d analysis.Diagnostic) { diagnostics = append(diagnostics, d) }}, &diagnostics
}

func FuzzCargalignLayout(f *testing.F) {
	// Empty, twelve-byte pointer gap, implicit zero raw offsets, bank tail,
	// dynamic total/spread, positional and marked-struct cases.
	for _, seed := range [][]byte{{0, 0}, {0, 2, 0, 0, 0, 0, 1, 1, 0, 0, 4, 1}, {1, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0}, {1, 1, 0, 0, 0, 0, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 1, 0}, {6, 3, 0}, {6, 2, 0, 0, 0, 0, 0, 1}} {
		f.Add(seed)
	}
	for _, mode := range []byte{0, 1, 2, 3, 4, 5, 6} {
		seed := make([]byte, 32)
		seed[0] = mode
		seed[1] = 1
		seed[29] = 1
		seed[31] = 4
		if mode == 0 {
			seed[5] = 4
			seed[6] = 1
		}
		f.Add(seed)
	}
	for _, alias := range []byte{2, 4} {
		seed := make([]byte, 32)
		seed[0] = 6
		seed[1] = 1
		seed[29] = alias
		f.Add(seed)
	}
	pkg := cargalignTypes(f)
	saved := reportUnsupported
	reportUnsupported = true
	defer func() { reportUnsupported = saved }()
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 32 {
			return
		}
		at := func(i int) byte {
			if i < len(data) {
				return data[i]
			}
			return 0
		}
		mode, count := int(at(0)%7), int(at(1)%5)
		paramType, schemaType := "cudakernel.Param", "cudakernel.Schema"
		if at(29)&1 != 0 {
			paramType, schemaType = "ParamAlias", "SchemaAlias"
		}
		widths := []int{0, -1, 1, 2, 3, 4, 8, 16}
		aligns := []int{0, -1, 1, 2, 3, 4, 8, 16}
		names := []string{"u32", "ptr", "u8", "u64"}
		goTypes := []string{"uint32", "uintptr", "uint8", "uint64"}
		params := make([]cudakernel.Param, count)
		offsets := make([]bool, count)
		var literals, fields []string
		for i := range params {
			k := int(at(2+i*5) % 4)
			p := cudakernel.Param{Name: fmt.Sprintf("p%d", i), Type: names[k], IsPointer: k == 1, ByteWidth: widths[int(at(3+i*5))%len(widths)], Alignment: aligns[int(at(4+i*5))%len(aligns)]}
			offsets[i] = at(6+i*5)&1 != 0
			p.Offset = int(at(5+i*5) % 33)
			if !offsets[i] {
				p.Offset = 0
			}
			text := fmt.Sprintf("{Name:%q,Type:%q,IsPointer:%t,ByteWidth:%d,Alignment:%d", p.Name, p.Type, p.IsPointer, p.ByteWidth, p.Alignment)
			if offsets[i] {
				text += fmt.Sprintf(",Offset:%d", p.Offset)
			}
			literals = append(literals, text+"}")
			fields = append(fields, p.Name+" "+goTypes[k])
			if mode == 6 {
				p.ByteWidth, p.Alignment, p.Offset = 0, 0, 0
			}
			params[i] = p
		}
		head := "package fixture\nimport \"github.com/tmc/lib/nvidia/x/cudakernel\"\ntype ParamAlias = cudakernel.Param\ntype SchemaAlias = cudakernel.Schema\n"
		var body string
		total := int(at(31) % 49)
		implicitTotal := at(30)&1 != 0
		switch mode {
		case 0:
			var args []string
			for _, s := range literals {
				args = append(args, paramType+s)
			}
			body = "var _,_ = cudakernel.NewSchema(" + strings.Join(args, ",") + ")"
		case 1:
			body = "var _ = " + schemaType + "{Params:[]" + paramType + "{" + strings.Join(literals, ",") + "}"
			if !implicitTotal {
				body += fmt.Sprintf(",TotalParamBytes:%d", total)
			} else {
				total = 0
			}
			body += "}"
		case 2:
			body = "var total int\nvar _ = " + schemaType + "{Params:[]" + paramType + "{},TotalParamBytes:total}"
		case 3:
			body = "var params []" + paramType + "\nvar _,_ = cudakernel.NewSchema(params...)"
		case 4:
			body = "var _ = " + schemaType + "{[]" + paramType + "{},0}"
		case 5:
			body = "var params []" + paramType + "\nvar _ = " + schemaType + "{Params:params}"
		case 6:
			body = "var _ cudakernel.Schema\n//cuda:kernelparams\ntype Args struct{" + strings.Join(fields, ";") + "}"
			if at(29)&2 != 0 {
				body = "type Small struct{" + strings.Join(fields, ";") + "}\n//cuda:kernelparams\ntype Args = Small"
			}
			if at(29)&4 != 0 {
				body = "//cuda:kernelparams\ntype Args = struct{" + strings.Join(fields, ";") + "}"
			}
		}
		pass, diagnostics := cargalignPass(t, pkg, head+body)
		if _, err := runCargalign(pass); err != nil {
			t.Fatal(err)
		}
		var want []string
		if mode >= 2 && mode <= 5 {
			reasons := map[int]string{2: "total parameter bytes are not constant", 3: "parameters are not written as literals", 4: "schema fields are not keyed literals", 5: "parameters are not written as constant literals"}
			want = []string{"kernel parameter layout not checked: " + reasons[mode]}
		} else {
			schema, err := cudakernel.NewSchema(params...)
			if err != nil {
				want = append(want, fmt.Sprintf("cudakernel.NewSchema rejects these parameters: %v", err))
			} else if mode == 6 {
				typ := pass.Pkg.Scope().Lookup("Args").Type()
				st := typ.Underlying().(*types.Struct)
				goFields := make([]*types.Var, st.NumFields())
				for i := range goFields {
					goFields[i] = st.Field(i)
				}
				goOffsets := pass.TypesSizes.Offsetsof(goFields)
				for i, p := range params {
					if goOffsets[i] != int64(schema.Params[i].Offset) {
						want = append(want, fmt.Sprintf("kernel parameter %s is at Go offset %d; cudakernel.Schema places it at offset %d", p.Name, goOffsets[i], schema.Params[i].Offset))
					}
				}
				if size := pass.TypesSizes.Sizeof(typ); size != int64(schema.TotalParamBytes) {
					want = append(want, fmt.Sprintf("kernel parameter block Args is %d bytes in Go; cudakernel.Schema computes %d total parameter bytes", size, schema.TotalParamBytes))
				}
			} else {
				for i, p := range params {
					if (mode == 1 || offsets[i]) && p.Offset != schema.Params[i].Offset {
						want = append(want, fmt.Sprintf("parameter %s declares offset %d; cudakernel.Schema places it at offset %d", p.Name, p.Offset, schema.Params[i].Offset))
					}
				}
				if mode == 1 && total != schema.TotalParamBytes {
					want = append(want, fmt.Sprintf("layout declares %d total parameter bytes; cudakernel.Schema computes %d", total, schema.TotalParamBytes))
				}
			}
		}
		var got []string
		for _, d := range *diagnostics {
			got = append(got, d.Message)
		}
		if strings.Join(got, "\n") != strings.Join(want, "\n") {
			t.Fatalf("source=%s\ngot=%q\nwant=%q", head+body, got, want)
		}
	})
}
