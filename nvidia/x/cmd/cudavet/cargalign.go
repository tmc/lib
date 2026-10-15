package main

import (
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
	"strings"

	"golang.org/x/tools/go/analysis"

	"github.com/tmc/lib/nvidia/x/cudakernel"
)

// cudakernelPath is the import path of the package that owns the parameter
// layout model. A composite literal is a kernel parameter declaration only if
// its type comes from here.
const cudakernelPath = "github.com/tmc/lib/nvidia/x/cudakernel"

// kernelParamsMarker opts a Go struct type into the parameter block check.
// A struct carrying it is laid out by the Go compiler but read by the device
// as a kernel parameter bank, so the two layouts have to agree. This is the
// shape of the bug where a 12-byte struct crosses a foreign-function bridge
// as 8 bytes and the tail is silently zero.
const kernelParamsMarker = "cuda:kernelparams"

// newSchema is the single authority on kernel parameter alignment. cargalign
// performs no alignment arithmetic; every offset and size it reports is read
// out of the Schema this returns. Tests replace it to show that: under a
// perturbed authority the reported offsets move, which an analyzer consulting
// its own arithmetic would not do.
var newSchema = cudakernel.NewSchema

// reportUnsupported makes cargalign name the parameter layouts it cannot
// check instead of passing over them. Layouts assembled at run time out of
// non-constant values have no offsets at compile time, so they are outside
// the check rather than clean.
var reportUnsupported bool

var cargalignAnalyzer = &analysis.Analyzer{
	Name: "cargalign",
	Doc: `check kernel parameter layouts against cudakernel.Schema

cargalign compares every kernel parameter layout written in source against
the layout cudakernel.NewSchema computes for the same parameters, and reports
the ones that disagree. It checks three constructs:

	cudakernel.NewSchema(...) calls with constant Param literals,
	cudakernel.Schema composite literals built without NewSchema,
	struct types marked //cuda:kernelparams, whose Go layout must match.

Parameter lists assembled from non-constant values are outside the check;
-cargalign.unsupported reports them rather than leaving them silent.`,
	Run: runCargalign,
}

func init() {
	cargalignAnalyzer.Flags.BoolVar(&reportUnsupported, "unsupported", false,
		"report parameter layouts that cannot be checked at compile time")
}

func runCargalign(pass *analysis.Pass) (any, error) {
	for _, file := range pass.Files {
		ast.Inspect(file, func(n ast.Node) bool {
			switch n := n.(type) {
			case *ast.CallExpr:
				checkNewSchemaCall(pass, n)
			case *ast.CompositeLit:
				checkSchemaLit(pass, n)
			case *ast.GenDecl:
				checkParamStructs(pass, n)
			}
			return true
		})
	}
	return nil, nil
}

// checkNewSchemaCall checks a cudakernel.NewSchema call whose parameters are
// written out as constant literals. A declared Offset that the authority does
// not agree with is reported even though NewSchema recomputes it: the value is
// a second, wrong statement of the same fact.
func checkNewSchemaCall(pass *analysis.Pass, call *ast.CallExpr) {
	if !isCudakernelFunc(pass, call.Fun, "NewSchema") {
		return
	}
	args := call.Args
	if call.Ellipsis.IsValid() {
		if len(args) != 1 {
			return
		}
		lit, ok := args[0].(*ast.CompositeLit)
		if !ok {
			reportUncheckable(pass, call.Pos(), "parameters are not written as literals")
			return
		}
		args = lit.Elts
	}
	params, offsets, ok := paramsFromExprs(pass, args)
	if !ok {
		reportUncheckable(pass, call.Pos(), "parameters are not written as constant literals")
		return
	}
	checkLayout(pass, call.Pos(), params, offsets, nil, 0)
}

// checkSchemaLit checks a cudakernel.Schema assembled as a composite literal,
// which is the one way to reach the device with offsets the authority never
// computed.
func checkSchemaLit(pass *analysis.Pass, lit *ast.CompositeLit) {
	if !isCudakernelType(pass.TypesInfo.TypeOf(lit), "Schema") {
		return
	}
	var (
		paramsLit *ast.CompositeLit
		totalExpr ast.Expr
		total     int
	)
	for _, elt := range lit.Elts {
		kv, ok := elt.(*ast.KeyValueExpr)
		if !ok {
			return
		}
		key, ok := kv.Key.(*ast.Ident)
		if !ok {
			return
		}
		switch key.Name {
		case "Params":
			if paramsLit, ok = kv.Value.(*ast.CompositeLit); !ok {
				reportUncheckable(pass, lit.Pos(), "parameters are not written as constant literals")
				return
			}
		case "TotalParamBytes":
			v, ok := constInt(pass, kv.Value)
			if !ok {
				return
			}
			totalExpr, total = kv.Value, v
		}
	}
	if paramsLit == nil {
		return
	}
	params, offsets, ok := paramsFromExprs(pass, paramsLit.Elts)
	if !ok {
		reportUncheckable(pass, lit.Pos(), "parameters are not written as constant literals")
		return
	}
	// A raw Schema stores an omitted Offset as zero. Unlike NewSchema,
	// it never computes that field, so check it at the parameter literal.
	for i, off := range offsets {
		if off == nil {
			offsets[i] = paramsLit.Elts[i]
		}
	}
	checkLayout(pass, lit.Pos(), params, offsets, totalExpr, total)
}

// checkLayout reports where a written layout disagrees with the authority.
// offsets holds the expression each parameter's Offset was written as, the
// parameter literal for an implicit zero in a raw Schema, or nil for an
// offset NewSchema computes. totalExpr is the written TotalParamBytes, or nil.
func checkLayout(pass *analysis.Pass, pos token.Pos, params []cudakernel.Param, offsets []ast.Expr, totalExpr ast.Expr, total int) {
	computed, err := newSchema(params...)
	if err != nil {
		pass.Reportf(pos, "cudakernel.NewSchema rejects these parameters: %v", err)
		return
	}
	for i, off := range offsets {
		if off == nil {
			continue
		}
		if want := computed.Params[i].Offset; params[i].Offset != want {
			pass.Reportf(off.Pos(), "parameter %s declares offset %d; cudakernel.Schema places it at offset %d",
				params[i].Name, params[i].Offset, want)
		}
	}
	if totalExpr != nil && total != computed.TotalParamBytes {
		pass.Reportf(totalExpr.Pos(), "layout declares %d total parameter bytes; cudakernel.Schema computes %d",
			total, computed.TotalParamBytes)
	}
}

// checkParamStructs checks struct types marked //cuda:kernelparams. The Go
// compiler lays such a struct out for the host; the device reads it as a
// parameter bank laid out by the authority. Where the two differ the bytes on
// the device are not the bytes the program wrote.
func checkParamStructs(pass *analysis.Pass, decl *ast.GenDecl) {
	if decl.Tok != token.TYPE {
		return
	}
	for _, spec := range decl.Specs {
		spec, ok := spec.(*ast.TypeSpec)
		if !ok || !hasMarker(decl.Doc, spec.Doc) {
			continue
		}
		named, ok := pass.TypesInfo.TypeOf(spec.Name).(*types.Named)
		if !ok {
			continue
		}
		st, ok := named.Underlying().(*types.Struct)
		if !ok {
			pass.Reportf(spec.Pos(), "%s is marked %s but is not a struct type", spec.Name.Name, kernelParamsMarker)
			continue
		}
		checkParamStruct(pass, spec, named, st)
	}
}

func checkParamStruct(pass *analysis.Pass, spec *ast.TypeSpec, named *types.Named, st *types.Struct) {
	params := make([]cudakernel.Param, 0, st.NumFields())
	fields := make([]*types.Var, 0, st.NumFields())
	supported := true
	for i := range st.NumFields() {
		f := st.Field(i)
		p, ok := paramForGoType(f.Name(), f.Type())
		if !ok {
			pass.Reportf(f.Pos(), "kernel parameter %s has Go type %s, which has no device parameter type",
				f.Name(), f.Type())
			supported = false
			continue
		}
		params = append(params, p)
		fields = append(fields, f)
	}
	if !supported || len(params) == 0 {
		return
	}
	computed, err := newSchema(params...)
	if err != nil {
		pass.Reportf(spec.Pos(), "cudakernel.NewSchema rejects these parameters: %v", err)
		return
	}
	goOffsets := pass.TypesSizes.Offsetsof(fields)
	for i, f := range fields {
		if want := int64(computed.Params[i].Offset); goOffsets[i] != want {
			pass.Reportf(f.Pos(), "kernel parameter %s is at Go offset %d; cudakernel.Schema places it at offset %d",
				f.Name(), goOffsets[i], want)
		}
	}
	if size := pass.TypesSizes.Sizeof(named); size != int64(computed.TotalParamBytes) {
		pass.Reportf(spec.Pos(), "kernel parameter block %s is %d bytes in Go; cudakernel.Schema computes %d total parameter bytes",
			spec.Name.Name, size, computed.TotalParamBytes)
	}
}

// paramForGoType returns the kernel parameter a Go field of type t stands for.
// Types whose width is not fixed by the language, such as int and uint, have
// no device parameter type: the device ABI is fixed-width.
func paramForGoType(name string, t types.Type) (cudakernel.Param, bool) {
	if _, ok := t.Underlying().(*types.Pointer); ok {
		return cudakernel.Param{Name: name, Type: "ptr", IsPointer: true}, true
	}
	basic, ok := t.Underlying().(*types.Basic)
	if !ok {
		return cudakernel.Param{}, false
	}
	switch basic.Kind() {
	case types.UnsafePointer, types.Uintptr:
		return cudakernel.Param{Name: name, Type: "ptr", IsPointer: true}, true
	}
	ptx, ok := map[types.BasicKind]string{
		types.Bool:    "bool",
		types.Uint8:   "u8",
		types.Int8:    "s8",
		types.Uint16:  "u16",
		types.Int16:   "s16",
		types.Uint32:  "u32",
		types.Int32:   "s32",
		types.Uint64:  "u64",
		types.Int64:   "s64",
		types.Float32: "f32",
		types.Float64: "f64",
	}[basic.Kind()]
	if !ok {
		return cudakernel.Param{}, false
	}
	return cudakernel.Param{Name: name, Type: ptx}, true
}

// paramsFromExprs reads a list of cudakernel.Param literals. It reports ok
// only when every parameter is a keyed literal of constants, since a layout
// with a value the analyzer cannot see has no compile-time offsets.
func paramsFromExprs(pass *analysis.Pass, exprs []ast.Expr) (params []cudakernel.Param, offsets []ast.Expr, ok bool) {
	if len(exprs) == 0 {
		return nil, nil, false
	}
	for _, e := range exprs {
		lit, ok := e.(*ast.CompositeLit)
		if !ok {
			return nil, nil, false
		}
		p, off, ok := paramFromLit(pass, lit)
		if !ok {
			return nil, nil, false
		}
		params = append(params, p)
		offsets = append(offsets, off)
	}
	return params, offsets, true
}

func paramFromLit(pass *analysis.Pass, lit *ast.CompositeLit) (p cudakernel.Param, offset ast.Expr, ok bool) {
	if !isCudakernelType(pass.TypesInfo.TypeOf(lit), "Param") {
		return p, nil, false
	}
	for _, elt := range lit.Elts {
		kv, ok := elt.(*ast.KeyValueExpr)
		if !ok {
			return p, nil, false
		}
		key, ok := kv.Key.(*ast.Ident)
		if !ok {
			return p, nil, false
		}
		tv, ok := pass.TypesInfo.Types[kv.Value]
		if !ok || tv.Value == nil {
			return p, nil, false
		}
		switch key.Name {
		case "Name":
			p.Name = constant.StringVal(tv.Value)
		case "Type":
			p.Type = constant.StringVal(tv.Value)
		case "ByteWidth":
			if p.ByteWidth, ok = constInt(pass, kv.Value); !ok {
				return p, nil, false
			}
		case "Alignment":
			if p.Alignment, ok = constInt(pass, kv.Value); !ok {
				return p, nil, false
			}
		case "IsPointer":
			p.IsPointer = constant.BoolVal(tv.Value)
		case "Offset":
			if p.Offset, ok = constInt(pass, kv.Value); !ok {
				return p, nil, false
			}
			offset = kv.Value
		default:
			return p, nil, false
		}
	}
	return p, offset, true
}

func constInt(pass *analysis.Pass, e ast.Expr) (int, bool) {
	tv, ok := pass.TypesInfo.Types[e]
	if !ok || tv.Value == nil {
		return 0, false
	}
	v, exact := constant.Int64Val(constant.ToInt(tv.Value))
	if !exact {
		return 0, false
	}
	return int(v), true
}

func reportUncheckable(pass *analysis.Pass, pos token.Pos, why string) {
	if reportUnsupported {
		pass.Reportf(pos, "kernel parameter layout not checked: %s", why)
	}
}

func isCudakernelFunc(pass *analysis.Pass, fun ast.Expr, name string) bool {
	sel, ok := fun.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	fn, ok := pass.TypesInfo.Uses[sel.Sel].(*types.Func)
	return ok && fn.Name() == name && fn.Pkg() != nil && fn.Pkg().Path() == cudakernelPath
}

func isCudakernelType(t types.Type, name string) bool {
	named, ok := t.(*types.Named)
	if !ok {
		return false
	}
	obj := named.Obj()
	return obj.Name() == name && obj.Pkg() != nil && obj.Pkg().Path() == cudakernelPath
}

func hasMarker(groups ...*ast.CommentGroup) bool {
	for _, g := range groups {
		if g == nil {
			continue
		}
		for _, c := range g.List {
			if strings.TrimPrefix(c.Text, "//") == kernelParamsMarker {
				return true
			}
		}
	}
	return false
}
