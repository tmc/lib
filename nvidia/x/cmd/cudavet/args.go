package main

import (
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
	"slices"
	"strings"

	"golang.org/x/tools/go/analysis"
)

// staticLen returns the length of a slice expression when the function
// fixes it: a composite literal without keys, make with a constant length,
// or a variable assigned exactly once from one of those.
func staticLen(info *types.Info, defs map[*types.Var][]ast.Expr, e ast.Expr) (int, bool) {
	e = ast.Unparen(e)
	switch e := e.(type) {
	case *ast.CompositeLit:
		if _, ok := info.TypeOf(e).Underlying().(*types.Slice); !ok {
			return 0, false
		}
		for _, elt := range e.Elts {
			if _, ok := elt.(*ast.KeyValueExpr); ok {
				return 0, false
			}
		}
		return len(e.Elts), true
	case *ast.CallExpr:
		if id, ok := e.Fun.(*ast.Ident); ok && id.Name == "make" && len(e.Args) >= 2 {
			if n, ok := intConst(info, e.Args[1]); ok {
				return int(n), true
			}
		}
	case *ast.Ident:
		v, _ := info.Uses[e].(*types.Var)
		return varLen(info, defs, v)
	}
	return 0, false
}

// varLen is staticLen for the slice held in v.
func varLen(info *types.Info, defs map[*types.Var][]ast.Expr, v *types.Var) (int, bool) {
	if d := defs[v]; len(d) == 1 && d[0] != nil {
		return staticLen(info, defs, d[0])
	}
	return 0, false
}

var modeCountAnalyzer = &analysis.Analyzer{
	Name: "modecount",
	Doc: `check that tensor mode slices match their descriptors

A tensor descriptor is created with one extent per mode, and each operation
that takes it takes a mode slice that must be as long. The C API has no query
for the count, so the bindings cannot check it. modecount follows a descriptor
back to the extent slice it was created from and reports a mode slice of a
different length, when both lengths are fixed in the function.`,
	Run: runModeCount,
}

func runModeCount(pass *analysis.Pass) (any, error) {
	info := pass.TypesInfo
	eachFunc(pass, func(body *ast.BlockStmt) {
		defs := assignments(info, body)
		modes := make(map[*types.Var]int) // descriptor -> extents
		for _, call := range calls(body) {
			fn := callee(info, call)
			if fn == nil || !inNvidia(fn.Pkg()) {
				continue
			}
			if fn.Name() == "CreateTensorDescriptor" && len(call.Args) > 0 {
				n, ok := staticLen(info, defs, call.Args[0])
				if !ok {
					continue
				}
				if v := assignedFrom(info, body, call); v != nil {
					modes[v] = n
				}
				continue
			}
			sig := fn.Type().(*types.Signature)
			for i := 0; i+1 < sig.Params().Len() && i+1 < len(call.Args); i++ {
				if !isNamed(sig.Params().At(i).Type(), fn.Pkg().Name(), "TensorDescriptor") {
					continue
				}
				if _, ok := sig.Params().At(i + 1).Type().(*types.Slice); !ok {
					continue
				}
				want, ok := modes[objOf(info, call.Args[i])]
				if !ok {
					continue
				}
				got, ok := staticLen(info, defs, call.Args[i+1])
				if ok && got != want {
					pass.Reportf(call.Args[i+1].Pos(), "%s has %d modes, but %s was created with %d extents",
						render(pass.Fset, call.Args[i+1]), got, types.ExprString(call.Args[i]), want)
				}
			}
		}
	})
	return nil, nil
}

// assignedFrom returns the variable the first result of call is assigned
// to, or nil.
func assignedFrom(info *types.Info, body *ast.BlockStmt, call *ast.CallExpr) *types.Var {
	var v *types.Var
	ast.Inspect(body, func(n ast.Node) bool {
		if a, ok := n.(*ast.AssignStmt); ok && len(a.Rhs) == 1 && a.Rhs[0] == ast.Expr(call) {
			v = objOf(info, a.Lhs[0])
		}
		return v == nil
	})
	return v
}

var firstElemAnalyzer = &analysis.Analyzer{
	Name: "firstelem",
	Doc: `check &s[0] arguments of bindings

Passing &s[0] panics when s is empty, and a count passed beside it must count
s. firstelem reports &s[0] passed to an nvidia binding when the function
neither fixes the length of s nor tests it, directly or by passing s to a
function of the package that tests it, and a count parameter directly
before the pointer that is the length of a different slice. A count after
its own pointer, as in &ids[0], len(ids), &vals[0], and a slice made with
the counted length, directly or by a function of the package that returns
make(_, len(p)) of its parameter p, are not mismatches. Test files are
checked only for mismatches.`,
	Run: runFirstElem,
}

func runFirstElem(pass *analysis.Pass) (any, error) {
	info := pass.TypesInfo
	decls := funcDecls(pass)
	// Closures see the variables of the function around them, and the tests
	// on them, so look at whole outer functions.
	eachOuterFunc(pass, func(body *ast.BlockStmt) {
		defs := assignments(info, body)
		inTest := strings.HasSuffix(pass.Fset.Position(body.Pos()).Filename, "_test.go")
		var all []*ast.CallExpr
		ast.Inspect(body, func(n ast.Node) bool {
			if c, ok := n.(*ast.CallExpr); ok {
				all = append(all, c)
			}
			return true
		})
		for _, call := range all {
			fn := callee(info, call)
			if fn == nil || !isBinding(fn.Pkg()) {
				continue
			}
			for i, arg := range call.Args {
				s := firstElemOf(info, arg)
				if s == nil {
					continue
				}
				if i > 0 {
					l := lenOf(info, call.Args[i-1])
					countAfter := i > 1 && firstElemOf(info, call.Args[i-2]) == l
					if l != nil && l != s && !countAfter && madeWithLen(info, decls, defs, s) != l {
						pass.Reportf(call.Args[i-1].Pos(), "count is len(%s), but the buffer after it is %s", l.Name(), s.Name())
					}
				}
				if inTest {
					continue
				}
				if !nonEmpty(info, decls, body, defs, s, call.Pos()) {
					pass.Reportf(arg.Pos(), "&%s[0] panics if %s is empty; test len(%s) first", s.Name(), s.Name(), s.Name())
				}
			}
		}
	})
	return nil, nil
}

// firstElemOf returns s for an argument &s[0], possibly converted.
func firstElemOf(info *types.Info, e ast.Expr) *types.Var {
	u, ok := unconvert(info, e).(*ast.UnaryExpr)
	if !ok || u.Op != token.AND {
		return nil
	}
	ix, ok := ast.Unparen(u.X).(*ast.IndexExpr)
	if !ok {
		return nil
	}
	if n, ok := intConst(info, ix.Index); !ok || n != 0 {
		return nil
	}
	v := objOf(info, ix.X)
	if v == nil {
		return nil
	}
	if _, ok := v.Type().Underlying().(*types.Slice); !ok {
		return nil
	}
	return v
}

// lenOf returns s for a count argument len(s), possibly converted.
func lenOf(info *types.Info, e ast.Expr) *types.Var {
	call, ok := unconvert(info, e).(*ast.CallExpr)
	if !ok || len(call.Args) != 1 {
		return nil
	}
	if id, ok := call.Fun.(*ast.Ident); !ok || id.Name != "len" {
		return nil
	}
	return objOf(info, call.Args[0])
}

// lenTested reports whether body compares len(s) before pos.
func lenTested(info *types.Info, body *ast.BlockStmt, s *types.Var, pos token.Pos) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		b, ok := n.(*ast.BinaryExpr)
		if ok && b.Pos() < pos && b.Op.Precedence() == token.EQL.Precedence() && (lenOf(info, b.X) == s || lenOf(info, b.Y) == s) {
			found = true
		}
		return !found
	})
	return found
}

// nonEmpty reports whether s is known to be non-empty at pos: it has a
// positive static length, its length is tested or an element read before pos,
// here or in a function of the package it is passed to, it comes from the
// same call as a slice whose length is tested, or it is made with a length
// that was tested, the length of a non-empty slice, or a product of the
// lengths of slices that were tested.
func nonEmpty(info *types.Info, decls map[*types.Func]*ast.FuncDecl, body *ast.BlockStmt, defs map[*types.Var][]ast.Expr, s *types.Var, pos token.Pos) bool {
	for range 4 { // follow make(_, len(l)) a few steps
		if n, ok := varLen(info, defs, s); ok && n > 0 || grown(info, defs, s) {
			return true
		}
		if lenTested(info, body, s, pos) || indexed(info, body, s, pos) || lenTestedBy(info, decls, body, s, pos) {
			return true
		}
		for _, t := range companions(info, body, s) {
			if lenTested(info, body, t, pos) {
				return true
			}
		}
		if n := madeWithVar(info, defs, s); n != nil && tested(info, body, n, pos) {
			return true
		}
		if ls := madeWithLens(info, defs, s); len(ls) > 1 && !slices.ContainsFunc(ls, func(l *types.Var) bool {
			return !lenTested(info, body, l, pos) && !indexed(info, body, l, pos)
		}) {
			return true
		}
		if s = madeWithLen(info, decls, defs, s); s == nil {
			return false
		}
	}
	return false
}

// funcDecls maps each function declared in the package to its declaration.
func funcDecls(pass *analysis.Pass) map[*types.Func]*ast.FuncDecl {
	decls := make(map[*types.Func]*ast.FuncDecl)
	for _, file := range srcFiles(pass) {
		for _, d := range file.Decls {
			if fd, ok := d.(*ast.FuncDecl); ok && fd.Body != nil {
				if fn, ok := pass.TypesInfo.Defs[fd.Name].(*types.Func); ok {
					decls[fn] = fd
				}
			}
		}
	}
	return decls
}

// params returns the parameters of fd in order, not counting the receiver.
func params(info *types.Info, fd *ast.FuncDecl) []*types.Var {
	var out []*types.Var
	for _, f := range fd.Type.Params.List {
		for _, name := range f.Names {
			v, _ := info.Defs[name].(*types.Var)
			out = append(out, v)
		}
		if len(f.Names) == 0 {
			out = append(out, nil)
		}
	}
	return out
}

// lenTestedBy reports whether s is passed, before pos, to a function of the
// package that tests the length of the parameter it arrives in, as a helper
// that validates its arguments does.
func lenTestedBy(info *types.Info, decls map[*types.Func]*ast.FuncDecl, body *ast.BlockStmt, s *types.Var, pos token.Pos) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if found || !ok || call.Pos() >= pos {
			return !found
		}
		fd := decls[callee(info, call)]
		if fd == nil {
			return true
		}
		ps := params(info, fd)
		for i, arg := range call.Args {
			if i < len(ps) && ps[i] != nil && objOf(info, arg) == s && lenTested(info, fd.Body, ps[i], fd.Body.End()) {
				found = true
			}
		}
		return !found
	})
	return found
}

// sameLen returns the parameter of fd whose length is the length of every
// slice fd returns first, each made with make(_, len(p)), or nil.
func sameLen(info *types.Info, fd *ast.FuncDecl) *types.Var {
	defs := assignments(info, fd.Body)
	var p *types.Var
	ok := true
	inspectShallow(fd.Body, func(n ast.Node) bool {
		ret, isRet := n.(*ast.ReturnStmt)
		if !isRet || !ok {
			return ok
		}
		var l *types.Var
		if len(ret.Results) > 0 {
			e := ast.Unparen(ret.Results[0])
			if v := objOf(info, e); v != nil {
				l = madeWithLen(info, nil, defs, v)
			} else if call, isCall := e.(*ast.CallExpr); isCall && isBuiltin(info, call, "make") && len(call.Args) >= 2 {
				l = lenOf(info, call.Args[1])
			}
		}
		if l == nil || p != nil && l != p {
			ok = false
		}
		p = l
		return ok
	})
	if !ok || p == nil || !slices.Contains(params(info, fd), p) {
		return nil
	}
	return p
}

// madeWithLens returns the slices l1, l2, ... when s is assigned once, from
// make(_, len(l1)*len(l2)*...), and nil otherwise.
func madeWithLens(info *types.Info, defs map[*types.Var][]ast.Expr, s *types.Var) []*types.Var {
	d := defs[s]
	if len(d) != 1 {
		return nil
	}
	call, ok := d[0].(*ast.CallExpr)
	if !ok || !isBuiltin(info, call, "make") || len(call.Args) < 2 {
		return nil
	}
	var out []*types.Var
	var factors func(e ast.Expr) bool
	factors = func(e ast.Expr) bool {
		e = unconvert(info, e)
		if b, ok := e.(*ast.BinaryExpr); ok && b.Op == token.MUL {
			return factors(b.X) && factors(b.Y)
		}
		l := lenOf(info, e)
		out = append(out, l)
		return l != nil
	}
	if !factors(call.Args[1]) {
		return nil
	}
	return out
}

// companions returns the other slices assigned with s from one call, as in
// opts, vals := o.DriverOptions(), whose lengths agree.
func companions(info *types.Info, body *ast.BlockStmt, s *types.Var) []*types.Var {
	var out []*types.Var
	ast.Inspect(body, func(n ast.Node) bool {
		a, ok := n.(*ast.AssignStmt)
		if !ok || len(a.Rhs) != 1 || len(a.Lhs) < 2 || !slices.ContainsFunc(a.Lhs, func(e ast.Expr) bool { return objOf(info, e) == s }) {
			return true
		}
		for _, e := range a.Lhs {
			if v := objOf(info, e); v != nil && v != s {
				if _, ok := v.Type().Underlying().(*types.Slice); ok {
					out = append(out, v)
				}
			}
		}
		return true
	})
	return out
}

// madeWithVar returns n when s is assigned once, from make(_, n).
func madeWithVar(info *types.Info, defs map[*types.Var][]ast.Expr, s *types.Var) *types.Var {
	d := defs[s]
	if len(d) != 1 {
		return nil
	}
	call, ok := d[0].(*ast.CallExpr)
	if !ok || !isBuiltin(info, call, "make") || len(call.Args) < 2 {
		return nil
	}
	return objOf(info, unconvert(info, call.Args[1]))
}

// tested reports whether v is compared in a condition before pos.
func tested(info *types.Info, body *ast.BlockStmt, v *types.Var, pos token.Pos) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		b, ok := n.(*ast.BinaryExpr)
		if ok && b.Pos() < pos && b.Op.Precedence() == token.EQL.Precedence() &&
			(objOf(info, unconvert(info, b.X)) == v || objOf(info, unconvert(info, b.Y)) == v) {
			found = true
		}
		return !found
	})
	return found
}

// grown reports whether s is first given a positive static length or an
// append of explicit elements, and later only appended to.
func grown(info *types.Info, defs map[*types.Var][]ast.Expr, s *types.Var) bool {
	d := defs[s]
	if len(d) == 0 || d[0] == nil {
		return false
	}
	if n, ok := staticLen(info, defs, d[0]); !(ok && n > 0 || appendsElems(info, d[0])) {
		return false
	}
	for _, e := range d[1:] {
		call, ok := e.(*ast.CallExpr)
		if !ok || !isBuiltin(info, call, "append") || objOf(info, call.Args[0]) != s {
			return false
		}
	}
	return true
}

// appendsElems reports whether e is append(x, elems...) with at least one
// element written out.
func appendsElems(info *types.Info, e ast.Expr) bool {
	call, ok := e.(*ast.CallExpr)
	return ok && isBuiltin(info, call, "append") && len(call.Args) > 1 && !call.Ellipsis.IsValid()
}

// isBuiltin reports whether call calls the builtin name.
func isBuiltin(info *types.Info, call *ast.CallExpr, name string) bool {
	id, ok := ast.Unparen(call.Fun).(*ast.Ident)
	return ok && info.Uses[id] == types.Universe.Lookup(name)
}

// madeWithLen returns l when every value assigned to s is make(_, len(l)),
// or a call of a function in decls that makes its result that way from a
// parameter given l, and nil otherwise.
func madeWithLen(info *types.Info, decls map[*types.Func]*ast.FuncDecl, defs map[*types.Var][]ast.Expr, s *types.Var) *types.Var {
	var l *types.Var
	for _, e := range defs[s] {
		call, ok := e.(*ast.CallExpr)
		if !ok {
			return nil
		}
		var v *types.Var
		if fd := decls[callee(info, call)]; fd != nil {
			if p := sameLen(info, fd); p != nil {
				if i := slices.Index(params(info, fd), p); i < len(call.Args) {
					v = objOf(info, call.Args[i])
				}
			}
		} else if isBuiltin(info, call, "make") && len(call.Args) >= 2 {
			v = lenOf(info, call.Args[1])
		}
		if v == nil || l != nil && v != l {
			return nil
		}
		l = v
	}
	return l
}

// indexed reports whether body indexes s with a constant before pos, which
// would already have panicked were s empty.
func indexed(info *types.Info, body *ast.BlockStmt, s *types.Var, pos token.Pos) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		ix, ok := n.(*ast.IndexExpr)
		if ok && ix.End() < pos && objOf(info, ix.X) == s {
			if _, ok := intConst(info, ix.Index); ok {
				found = true
			}
		}
		return !found
	})
	return found
}

var scalarTypeAnalyzer = &analysis.Analyzer{
	Name: "scalartype",
	Doc: `check that host scaling factors have the kind the operation reads

A Scalar made by Float32Scalar, Float64Scalar and the like carries its kind,
and the library reads alpha and beta as the type the operation computes in. A
float64 read as a float32 is silent garbage, not an error. Where the type is
in a descriptor the binding cannot check it, and where it is a parameter the
binding checks only at run time. scalartype follows a cuTENSOR plan back
through CreatePlan and the operation descriptor to the compute descriptor, a
cuBLASLt Matmul to the scale type of its MatmulDesc, and reads the computeType
argument of cuBLAS and cuSPARSE calls, and reports a Scalar made with a
constructor of another kind.`,
	Run: runScalarType,
}

// Kinds of host Scalar, named by the Go type the constructor takes.
var (
	real32  = []string{"float32", "complex64"}
	real64  = []string{"float64", "complex128"}
	half    = []string{"float16"}
	integer = []string{"int32"}
)

// cutensorScalar gives the Scalar kinds each cuTENSOR compute descriptor
// reads. Reduced-precision descriptors take float scalars.
var cutensorScalar = map[string][]string{
	"ComputeDesc16F": real32, "ComputeDesc16BF": real32, "ComputeDescTF32": real32,
	"ComputeDesc3XTF32": real32, "ComputeDesc32F": real32, "ComputeDesc64F": real64,
}

// cublasScalar gives the Scalar kinds each cuBLAS compute type reads, as the
// table of cublasGemmEx lists them.
var cublasScalar = map[string][]string{
	"Compute16F": half, "Compute16FPedantic": half,
	"Compute32F": real32, "Compute32FPedantic": real32, "Compute32FFast16F": real32,
	"Compute32FFast16BF": real32, "Compute32FFastTF32": real32, "Compute32FEmulated16bfx9": real32,
	"Compute64F": real64, "Compute64FPedantic": real64, "Compute64FEmulatedFixedpoint": real64,
	"Compute32I": integer, "Compute32IPedantic": integer,
}

// dataTypeScalar gives the Scalar kind of a cudaDataType used as a compute
// or scale type.
var dataTypeScalar = map[string][]string{
	"R16F": half, "R32F": {"float32"}, "R64F": {"float64"},
	"C32F": {"complex64"}, "C64F": {"complex128"}, "R32I": integer,
}

// scalarKinds gives the kind of the Scalar each constructor makes.
var scalarKinds = map[string]string{
	"Float16BitsScalar": "float16",
	"Float32Scalar":     "float32",
	"Float64Scalar":     "float64",
	"Complex64Scalar":   "complex64",
	"Complex128Scalar":  "complex128",
	"Int32Scalar":       "int32",
}

func runScalarType(pass *analysis.Pass) (any, error) {
	info := pass.TypesInfo
	eachFunc(pass, func(body *ast.BlockStmt) {
		source := soleSource(info, body)
		for _, call := range calls(body) {
			fn := callee(info, call)
			if fn == nil || !inNvidia(fn.Pkg()) || len(call.Args) == 0 {
				continue
			}
			var compute string
			var want []string
			switch pkg := fn.Pkg().Name(); {
			case pkg == "cutensor" && (fn.Name() == "Contract" || fn.Name() == "ContractTrinary" || strings.HasPrefix(fn.Name(), "Elementwise") || fn.Name() == "Reduce" || fn.Name() == "Permute"):
				compute = cutensorCompute(info, source, call.Args[0])
				want = cutensorScalar[compute]
			case pkg == "cublaslt" && fn.Name() == "Matmul":
				// NewMatmulDesc(computeType, scaleType)
				if c := source(call.Args[0]); c != nil && calleeIn(info, c, "cublaslt") == "NewMatmulDesc" && len(c.Args) == 2 {
					compute = constName(info, c.Args[1])
					want = dataTypeScalar[compute]
				}
			case pkg == "cublas" || pkg == "cusparse":
				sig := fn.Type().(*types.Signature)
				for i := range min(sig.Params().Len(), len(call.Args)) {
					p := sig.Params().At(i)
					if p.Name() != "computeType" {
						continue
					}
					compute = constName(info, call.Args[i])
					if isNamed(p.Type(), pkg, "ComputeType") {
						want = cublasScalar[compute]
					} else {
						want = dataTypeScalar[compute]
					}
				}
			}
			if want == nil {
				continue
			}
			sig := fn.Type().(*types.Signature)
			for i := range min(sig.Params().Len(), len(call.Args)) {
				if !isNamed(sig.Params().At(i).Type(), fn.Pkg().Name(), "Scalar") {
					continue
				}
				arg := call.Args[i]
				if kind := scalarKind(info, fn.Pkg(), arg, source); kind != "" && !slices.Contains(want, kind) {
					pass.Reportf(arg.Pos(), "%s is a %s Scalar, but %s takes %s", render(pass.Fset, arg), kind, compute, strings.Join(want, " or "))
				}
			}
		}
	})
	return nil, nil
}

// soleSource returns a function that gives the call whose first result is
// assigned to e, when e is a variable assigned only there in body.
func soleSource(info *types.Info, body *ast.BlockStmt) func(e ast.Expr) *ast.CallExpr {
	stmts := assignStmts(body)
	return func(e ast.Expr) *ast.CallExpr {
		v := objOf(info, e)
		if v == nil {
			return nil
		}
		var c *ast.CallExpr
		n := 0
		for _, stmt := range stmts {
			for i, l := range stmt.Lhs {
				if objOf(info, l) != v {
					continue
				}
				n++
				switch {
				case len(stmt.Rhs) == len(stmt.Lhs):
					c, _ = stmt.Rhs[i].(*ast.CallExpr)
				case i == 0 && len(stmt.Rhs) == 1:
					c, _ = stmt.Rhs[0].(*ast.CallExpr)
				}
			}
		}
		if n != 1 {
			return nil
		}
		return c
	}
}

func assignStmts(body *ast.BlockStmt) []*ast.AssignStmt {
	var out []*ast.AssignStmt
	ast.Inspect(body, func(n ast.Node) bool {
		if a, ok := n.(*ast.AssignStmt); ok {
			out = append(out, a)
		}
		return true
	})
	return out
}

// cutensorCompute follows a plan back to the compute descriptor its
// operation was created with, and returns the descriptor's constructor name.
func cutensorCompute(info *types.Info, source func(ast.Expr) *ast.CallExpr, plan ast.Expr) string {
	c := source(plan)
	if c == nil || calleeIn(info, c, "cutensor") != "CreatePlan" || len(c.Args) == 0 {
		return ""
	}
	op := source(c.Args[0])
	if op == nil || !strings.HasPrefix(calleeIn(info, op, "cutensor"), "Create") || len(op.Args) == 0 {
		return ""
	}
	desc := source(op.Args[len(op.Args)-1])
	if desc == nil {
		return ""
	}
	return calleeIn(info, desc, "cutensor")
}

// scalarKind returns the kind of a host Scalar of package pkg made by a
// constructor, directly or through a variable assigned once, or "" if the
// kind is not known.
func scalarKind(info *types.Info, pkg *types.Package, e ast.Expr, source func(ast.Expr) *ast.CallExpr) string {
	c, ok := ast.Unparen(e).(*ast.CallExpr)
	if !ok {
		c = source(e)
	}
	if c == nil {
		return ""
	}
	fn := callee(info, c)
	if fn == nil || fn.Pkg() != pkg {
		return ""
	}
	return scalarKinds[fn.Name()]
}

// constName returns the name of the constant e refers to, or "".
func constName(info *types.Info, e ast.Expr) string {
	var id *ast.Ident
	switch e := ast.Unparen(e).(type) {
	case *ast.Ident:
		id = e
	case *ast.SelectorExpr:
		id = e.Sel
	default:
		return ""
	}
	if c, ok := info.Uses[id].(*types.Const); ok {
		return c.Name()
	}
	return ""
}

var rawEnumAnalyzer = &analysis.Analyzer{
	Name: "rawenum",
	Doc: `report numeric literals of binding enum types that have names

A literal such as cusolver.Operation(0), or const opN cusolver.Operation = 0,
often works around a constant the binding once lacked. rawenum reports a
conversion or typed constant of an enum type of an nvidia package when that
package now declares a constant of the type with the same value, and names the
constant. An untyped literal argument, such as a 0 for no flags, is not
reported. cufft.Direction is left to fftdirection.`,
	Run: runRawEnum,
}

func runRawEnum(pass *analysis.Pass) (any, error) {
	eachEnumLiteral(pass, func(e ast.Expr, n *types.Named, name string, explicit bool) {
		if !explicit || n.Obj().Name() == "Direction" && inPkg(n.Obj().Pkg(), "cufft") {
			return
		}
		pass.Reportf(e.Pos(), "literal %s of type %s.%s; use %s.%s", types.ExprString(e), n.Obj().Pkg().Name(), n.Obj().Name(), n.Obj().Pkg().Name(), name)
	})
	return nil, nil
}

var fftDirectionAnalyzer = &analysis.Analyzer{
	Name: "fftdirection",
	Doc: `report cuFFT directions written as numbers

cufft.Forward is -1 and cufft.Inverse is 1; the sign convention is easy to
flip when written as a number. fftdirection reports a numeric direction.`,
	Run: runFFTDirection,
}

func runFFTDirection(pass *analysis.Pass) (any, error) {
	eachEnumLiteral(pass, func(e ast.Expr, n *types.Named, name string, explicit bool) {
		if n.Obj().Name() == "Direction" && inPkg(n.Obj().Pkg(), "cufft") {
			pass.Reportf(e.Pos(), "direction %s written as a number; use cufft.%s", types.ExprString(e), name)
		}
	})
	return nil, nil
}

// eachEnumLiteral calls f for each numeric literal, outside the declaring
// package, whose type is a named integer type of an nvidia package that
// declares a constant of that type and value, with the constant's name.
// explicit reports whether the type is written out, in a conversion or a
// typed declaration, rather than given by context.
func eachEnumLiteral(pass *analysis.Pass, f func(e ast.Expr, n *types.Named, name string, explicit bool)) {
	info := pass.TypesInfo
	seen := make(map[token.Pos]bool)
	explicit := make(map[token.Pos]bool)
	for _, file := range srcFiles(pass) {
		ast.Inspect(file, func(node ast.Node) bool {
			var e ast.Expr
			var t types.Type
			switch node := node.(type) {
			case *ast.ValueSpec:
				if node.Type != nil {
					for _, v := range node.Values {
						explicit[ast.Unparen(v).Pos()] = true
					}
				}
			case *ast.BasicLit:
				e, t = node, info.TypeOf(node)
			case *ast.UnaryExpr:
				if lit, ok := node.X.(*ast.BasicLit); ok && node.Op == token.SUB {
					e, t = node, info.TypeOf(node)
					seen[lit.Pos()] = true // typed too, with the opposite sign
				}
			case *ast.CallExpr:
				// unsafe.Sizeof(T(0)) wants a value of the type, not a name.
				if sel, ok := node.Fun.(*ast.SelectorExpr); ok {
					if _, ok := info.Uses[sel.Sel].(*types.Builtin); ok {
						return false
					}
				}
				// T(-1): the operand may keep its untyped type.
				if tv, ok := info.Types[node.Fun]; ok && tv.IsType() && len(node.Args) == 1 {
					arg := ast.Unparen(node.Args[0])
					if u, ok := arg.(*ast.UnaryExpr); ok {
						arg = u.X
					}
					if _, ok := arg.(*ast.BasicLit); ok {
						e, t = node.Args[0], tv.Type
						explicit[e.Pos()] = true
					}
				}
			}
			if e == nil || seen[e.Pos()] {
				return true
			}
			n, ok := types.Unalias(t).(*types.Named)
			if !ok || !inNvidia(n.Obj().Pkg()) || n.Obj().Pkg() == pass.Pkg {
				return true
			}
			if b, ok := n.Underlying().(*types.Basic); !ok || b.Info()&types.IsInteger == 0 {
				return true
			}
			val := info.Types[e].Value
			if val == nil {
				return true
			}
			if name := enumName(n, val); name != "" {
				seen[e.Pos()] = true
				f(e, n, name, explicit[e.Pos()])
			}
			return true
		})
	}
}

// enumName returns the name of the constant of type n with value val in n's
// package, or "".
func enumName(n *types.Named, val constant.Value) string {
	scope := n.Obj().Pkg().Scope()
	for _, name := range scope.Names() {
		c, ok := scope.Lookup(name).(*types.Const)
		if ok && c.Exported() && types.Identical(c.Type(), n) && constant.Compare(c.Val(), token.EQL, val) {
			return name
		}
	}
	return ""
}

var stringNULAnalyzer = &analysis.Analyzer{
	Name: "stringnul",
	Doc: `check that constant strings passed to bindings hold no NUL byte

A binding passes a string to C as a NUL-terminated copy, and rejects one that
contains a NUL at run time. stringnul reports a constant string with a NUL
passed to a string parameter of a binding. Packages under x/ take Go strings
and are not reported.`,
	Run: runStringNUL,
}

func runStringNUL(pass *analysis.Pass) (any, error) {
	info := pass.TypesInfo
	eachFunc(pass, func(body *ast.BlockStmt) {
		for _, call := range calls(body) {
			fn := callee(info, call)
			if fn == nil || !isBinding(fn.Pkg()) {
				continue
			}
			sig := fn.Type().(*types.Signature)
			for i, arg := range call.Args {
				var p types.Type
				switch {
				case sig.Variadic() && i >= sig.Params().Len()-1:
					p = sig.Params().At(sig.Params().Len() - 1).Type().(*types.Slice).Elem()
				case i < sig.Params().Len():
					p = sig.Params().At(i).Type()
				}
				if b, ok := p.(*types.Basic); !ok || b.Kind() != types.String {
					continue
				}
				if s, ok := constString(info, arg); ok && strings.Contains(s, "\x00") {
					pass.Reportf(arg.Pos(), "%s argument contains a NUL byte; %s rejects it at run time", fn.Name(), fn.Name())
				}
			}
		}
	})
	return nil, nil
}

// jitCall names the driver calls that fill JIT log buffers given as options.
var jitCall = map[string]bool{"ModuleLoadDataEx": true, "AddData": true, "AddFile": true, "Complete": true, "LinkCreate": true}

// jitBuffers returns the byte slices of body whose address is taken with
// &b[0], the way a log buffer is handed to the JIT as an option value.
func jitBuffers(info *types.Info, body *ast.BlockStmt) map[*types.Var]bool {
	bufs := make(map[*types.Var]bool)
	ast.Inspect(body, func(n ast.Node) bool {
		if v := firstElemOf(info, asExpr(n)); v != nil {
			if s, ok := v.Type().Underlying().(*types.Slice); ok {
				if b, ok := s.Elem().(*types.Basic); ok && b.Kind() == types.Byte {
					bufs[v] = true
				}
			}
		}
		return true
	})
	return bufs
}

func asExpr(n ast.Node) ast.Expr {
	e, _ := n.(ast.Expr)
	return e
}

var jitLogAnalyzer = &analysis.Analyzer{
	Name: "jitlog",
	Doc: `check that JIT log buffers are read only as far as they were filled

The JIT writes its info and error logs into buffers given as options and
reports how much it wrote; the rest of the buffer is whatever it held. jitlog
reports string(buf) of a whole log buffer after a ModuleLoadDataEx or linker
call, instead of the filled prefix.`,
	Run: runJITLog,
}

func runJITLog(pass *analysis.Pass) (any, error) {
	info := pass.TypesInfo
	eachFunc(pass, func(body *ast.BlockStmt) {
		bufs := jitBuffers(info, body)
		if len(bufs) == 0 {
			return
		}
		var jit token.Pos
		for _, call := range calls(body) {
			if fn := callee(info, call); fn != nil && inPkg(fn.Pkg(), "driver") && jitCall[fn.Name()] {
				if jit == token.NoPos {
					jit = call.End()
				}
				continue
			}
			tv, ok := info.Types[call.Fun]
			if !ok || !tv.IsType() || len(call.Args) != 1 || jit == token.NoPos || call.Pos() < jit {
				continue
			}
			if b, ok := tv.Type.Underlying().(*types.Basic); !ok || b.Kind() != types.String {
				continue
			}
			if v := objOf(info, call.Args[0]); bufs[v] {
				pass.Reportf(call.Pos(), "string(%s) takes the whole log buffer; only the prefix the JIT reports filling holds the log", v.Name())
			}
		}
	})
	return nil, nil
}

var linkOptionsAnalyzer = &analysis.Analyzer{
	Name: "linkoptions",
	Doc: `check that LinkCreate's option buffers outlive the link

The linker keeps the options given to LinkCreate and writes its logs and
timings into their buffers as late as LinkComplete (cuda.h:7819). linkoptions
reports a LinkCreate given buffers local to the function when the link state
leaves the function, returned or stored, so that the buffers die first.`,
	Run: runLinkOptions,
}

func runLinkOptions(pass *analysis.Pass) (any, error) {
	info := pass.TypesInfo
	eachFunc(pass, func(body *ast.BlockStmt) {
		bufs := jitBuffers(info, body)
		local := false
		for v := range bufs {
			if declaredIn(v, body) {
				local = true
			}
		}
		if !local {
			return
		}
		inspectShallow(body, func(n ast.Node) bool {
			a, ok := n.(*ast.AssignStmt)
			if !ok || len(a.Rhs) != 1 {
				return true
			}
			call, ok := a.Rhs[0].(*ast.CallExpr)
			if !ok || calleeIn(info, call, "driver") != "LinkCreate" {
				return true
			}
			l := objOf(info, a.Lhs[0])
			if l != nil && linkEscapes(info, body, l) {
				pass.Reportf(call.Pos(), "LinkCreate is given option buffers local to this function, but %s outlives it; the linker writes them until LinkComplete", l.Name())
			}
			return true
		})
	})
	return nil, nil
}

// linkEscapes reports whether the link state l leaves body: returned, or
// assigned somewhere other than a local variable.
func linkEscapes(info *types.Info, body *ast.BlockStmt, l *types.Var) bool {
	found := false
	withStack(body, func(n ast.Node, stack []ast.Node) {
		id, ok := n.(*ast.Ident)
		if !ok || info.Uses[id] != l || len(stack) == 0 {
			return
		}
		switch p := stack[len(stack)-1].(type) {
		case *ast.ReturnStmt, *ast.CompositeLit, *ast.KeyValueExpr:
			found = true
		case *ast.AssignStmt:
			for i, r := range p.Rhs {
				if r == ast.Expr(id) && i < len(p.Lhs) {
					if _, ok := p.Lhs[i].(*ast.Ident); !ok {
						found = true
					}
				}
			}
		}
	})
	return found
}
