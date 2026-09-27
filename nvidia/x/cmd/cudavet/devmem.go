package main

import (
	"go/ast"
	"go/token"
	"go/types"
	"regexp"
	"strings"

	"golang.org/x/tools/go/analysis"
)

var deviceDerefAnalyzer = &analysis.Analyzer{
	Name: "devicederef",
	Doc: `check that device addresses are never made into Go pointers

A DevicePtr is an address in device memory. Made into a Go pointer, it may be
scanned by the garbage collector, and dereferencing it on the host faults.
devicederef reports unsafe.Pointer applied to a DevicePtr or to a uintptr
derived from one, which vet's unsafeptr check catches only in its first form,
and unsafe.Add(nil, d), which vet does not catch at all.`,
	Run: runDeviceDeref,
}

func runDeviceDeref(pass *analysis.Pass) (any, error) {
	info := pass.TypesInfo
	eachFunc(pass, func(body *ast.BlockStmt) {
		derived := devicePtrVars(info, body)
		inspectShallow(body, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			switch {
			case isUnsafe(info, call.Fun, "Pointer") && len(call.Args) == 1:
				if fromDevicePtr(info, derived, call.Args[0]) {
					pass.Reportf(call.Pos(), "unsafe.Pointer of a device address makes it a Go pointer; keep it a DevicePtr")
				}
			case isUnsafe(info, call.Fun, "Add") && len(call.Args) == 2:
				if isNil(info, call.Args[0]) && fromDevicePtr(info, derived, call.Args[1]) {
					pass.Reportf(call.Pos(), "unsafe.Add(nil, d) of a device address makes it a Go pointer; keep it a DevicePtr")
				}
			}
			return true
		})
	})
	return nil, nil
}

// isUnsafe reports whether fun names unsafe.name, either the builtin
// function or the conversion to unsafe.Pointer.
func isUnsafe(info *types.Info, fun ast.Expr, name string) bool {
	sel, ok := ast.Unparen(fun).(*ast.SelectorExpr)
	if !ok || sel.Sel.Name != name {
		return false
	}
	pkg, ok := sel.X.(*ast.Ident)
	if !ok {
		return false
	}
	pn, ok := info.Uses[pkg].(*types.PkgName)
	return ok && pn.Imported().Path() == "unsafe"
}

// devicePtrVars returns the variables of body assigned only values derived
// from a DevicePtr: uintptr(d), and sums and differences involving one.
func devicePtrVars(info *types.Info, body *ast.BlockStmt) map[*types.Var]bool {
	derived := make(map[*types.Var]bool)
	for changed := true; changed; {
		changed = false
		inspectShallow(body, func(n ast.Node) bool {
			assign, ok := n.(*ast.AssignStmt)
			if !ok || len(assign.Lhs) != len(assign.Rhs) {
				return true
			}
			for i, l := range assign.Lhs {
				v := objOf(info, l)
				if v != nil && !derived[v] && !isDevicePtr(v.Type()) && fromDevicePtr(info, derived, assign.Rhs[i]) {
					derived[v] = true
					changed = true
				}
			}
			return true
		})
	}
	return derived
}

// fromDevicePtr reports whether e is a DevicePtr, or a value computed from
// one, possibly through a variable in derived.
func fromDevicePtr(info *types.Info, derived map[*types.Var]bool, e ast.Expr) bool {
	e = unconvert(info, e)
	if isDevicePtr(info.TypeOf(e)) {
		return true
	}
	switch e := e.(type) {
	case *ast.Ident:
		v, _ := info.Uses[e].(*types.Var)
		return derived[v]
	case *ast.BinaryExpr:
		if e.Op == token.ADD || e.Op == token.SUB {
			return fromDevicePtr(info, derived, e.X) || fromDevicePtr(info, derived, e.Y)
		}
	}
	return false
}

var devptrUintptrAnalyzer = &analysis.Analyzer{
	Name: "devptruintptr",
	Doc: `report device pointers passed to bindings as uintptr

A binding that types device memory, one that declares a DevicePtr or names
one in an exported signature, should take every device buffer as a DevicePtr.
A uintptr parameter there means the generator did not classify the buffer as
device memory, and the caller has to convert, losing the type. devptruintptr
reports uintptr(d), d a DevicePtr, passed to a uintptr parameter of such a
binding, naming the parameter so that the classification can be fixed in the
generator. A binding that never names a DevicePtr takes all device memory as
uintptr by policy, and calls to it are not reported.`,
	Run: runDevptrUintptr,
}

func runDevptrUintptr(pass *analysis.Pass) (any, error) {
	info := pass.TypesInfo
	typed := make(map[*types.Package]bool)
	eachFunc(pass, func(body *ast.BlockStmt) {
		for _, call := range calls(body) {
			fn := callee(info, call)
			if fn == nil || !isBinding(fn.Pkg()) || fn.Pkg() == pass.Pkg || !fn.Exported() {
				continue
			}
			t, ok := typed[fn.Pkg()]
			if !ok {
				t = typesDeviceMemory(fn.Pkg())
				typed[fn.Pkg()] = t
			}
			if !t {
				continue
			}
			sig := fn.Type().(*types.Signature)
			for i, arg := range call.Args {
				if i >= sig.Params().Len() || sig.Variadic() && i >= sig.Params().Len()-1 {
					break
				}
				p := sig.Params().At(i)
				if !types.Identical(p.Type(), types.Typ[types.Uintptr]) {
					continue
				}
				if conv, ok := ast.Unparen(arg).(*ast.CallExpr); ok && len(conv.Args) == 1 && isDevicePtr(info.TypeOf(conv.Args[0])) {
					pass.Reportf(arg.Pos(), "device pointer passed to %s.%s as uintptr; the binding does not type %s as device memory",
						fn.Pkg().Name(), fn.Name(), p.Name())
				}
			}
		}
	})
	return nil, nil
}

// typesDeviceMemory reports whether pkg declares a DevicePtr or names one in
// the parameters or results of an exported function or method. A binding that
// does neither passes all device memory as uintptr.
func typesDeviceMemory(pkg *types.Package) bool {
	scope := pkg.Scope()
	for _, name := range scope.Names() {
		obj := scope.Lookup(name)
		if !obj.Exported() {
			continue
		}
		switch obj := obj.(type) {
		case *types.Func:
			if namesDevicePtr(obj.Type().(*types.Signature)) {
				return true
			}
		case *types.TypeName:
			if isDevicePtr(obj.Type()) {
				return true
			}
			if n, ok := types.Unalias(obj.Type()).(*types.Named); ok {
				for m := range n.Methods() {
					if m.Exported() && namesDevicePtr(m.Type().(*types.Signature)) {
						return true
					}
				}
			}
		}
	}
	return false
}

// namesDevicePtr reports whether a parameter or result of sig is a DevicePtr.
func namesDevicePtr(sig *types.Signature) bool {
	for _, tup := range []*types.Tuple{sig.Params(), sig.Results()} {
		for v := range tup.Variables() {
			if isDevicePtr(v.Type()) {
				return true
			}
		}
	}
	return false
}

var memallocPtrAnalyzer = &analysis.Analyzer{
	Name: "memallocptr",
	Doc: `check that types placed in device memory contain no Go pointers

The garbage collector does not scan device memory, so a Go pointer stored
there dangles, and a copy back to the host fills pointer fields with whatever
the device held. memallocptr reports MemAlloc[T], its managed and stream
forms, and the typed host copies, instantiated with a T that contains
pointers, slices, maps, channels, strings, interfaces or functions.`,
	Run: runMemallocPtr,
}

// typedDeviceCall names the generic calls whose type argument is laid out in
// device memory.
var typedDeviceCall = map[string]bool{
	"MemAlloc": true, "MemAllocManaged": true, "MemAllocAsync": true,
	"CopyFromHost": true, "CopyToHost": true, "CopyFromHostAsync": true, "CopyToHostAsync": true,
}

func runMemallocPtr(pass *analysis.Pass) (any, error) {
	info := pass.TypesInfo
	for id, inst := range info.Instances {
		fn, ok := info.Uses[id].(*types.Func)
		if !ok || !inNvidia(fn.Pkg()) || !typedDeviceCall[fn.Name()] || inst.TypeArgs.Len() == 0 || inGenerated(pass, id.Pos()) {
			continue
		}
		t := inst.TypeArgs.At(0)
		if why := goPointer(t, nil); why != "" {
			pass.Reportf(id.Pos(), "%s[%s]: %s holds %s, which the garbage collector cannot follow into device memory",
				fn.Name(), types.TypeString(t, types.RelativeTo(pass.Pkg)), types.TypeString(t, types.RelativeTo(pass.Pkg)), why)
		}
	}
	return nil, nil
}

// goPointer describes the first Go pointer t contains, or returns "".
func goPointer(t types.Type, seen map[types.Type]bool) string {
	if seen[t] {
		return ""
	}
	if seen == nil {
		seen = make(map[types.Type]bool)
	}
	seen[t] = true
	switch u := t.Underlying().(type) {
	case *types.Basic:
		switch u.Kind() {
		case types.String:
			return "a string"
		case types.UnsafePointer:
			return "an unsafe.Pointer"
		}
	case *types.Pointer:
		return "a pointer"
	case *types.Slice:
		return "a slice"
	case *types.Map:
		return "a map"
	case *types.Chan:
		return "a channel"
	case *types.Signature:
		return "a func"
	case *types.Interface:
		if _, ok := t.(*types.TypeParam); ok {
			return ""
		}
		return "an interface"
	case *types.Array:
		return goPointer(u.Elem(), seen)
	case *types.Struct:
		for f := range u.Fields() {
			if why := goPointer(f.Type(), seen); why != "" {
				return why + " in field " + f.Name()
			}
		}
	}
	return ""
}

var useAfterDestroyAnalyzer = &analysis.Analyzer{
	Name: "useafterdestroy",
	Doc: `check that handles are not used after they are released

A handle released with Close, Destroy, Free or Unload names nothing afterwards; in C,
using it is a use after free. useafterdestroy reports a use of a handle that
follows its release in the same block, before the variable is assigned again.
Comparing the handle with a constant, as in h != 0 after a Destroy that zeroes
it, is not a use. A release whose error is tested so that the code goes on
only when it failed, as in

	if err := h.Free(); !errors.Is(err, ErrInUse) {
		t.Fatal(err)
	}

leaves the handle live. A handle that records its own release, such as a
*cuda.Stream or a cudamem.Slice, whose copies share their allocation, may be
released twice, or used where the use is expected to fail: that tests the
record.`,
	Run: runUseAfterDestroy,
}

func runUseAfterDestroy(pass *analysis.Pass) (any, error) {
	info := pass.TypesInfo
	inspectBlocks(pass, func(list []ast.Stmt) {
		for i, stmt := range list {
			v, name := releaseStmt(info, stmt)
			if v == nil {
				continue
			}
			if s, ok := stmt.(*ast.IfStmt); ok && releaseFailed(info, s) {
				continue
			}
			for _, next := range list[i+1:] {
				if assigns(info, next, v) {
					break
				}
				var use *ast.Ident
				inspectShallow(next, func(n ast.Node) bool {
					if valueOnly(info, n, v) {
						return false
					}
					if id, ok := n.(*ast.Ident); ok && use == nil && info.Uses[id] == v {
						use = id
					}
					return use == nil
				})
				if use != nil {
					if !deliberateUse(info, list[:i], next, v, use) {
						pass.Reportf(use.Pos(), "%s is used after %s.%s", v.Name(), v.Name(), name)
					}
					break
				}
				if _, ok := next.(*ast.ReturnStmt); ok {
					break
				}
			}
		}
	})
	return nil, nil
}

// inspectBlocks calls f with the statement list of every block and case
// clause in the package.
func inspectBlocks(pass *analysis.Pass, f func([]ast.Stmt)) {
	for _, file := range srcFiles(pass) {
		ast.Inspect(file, func(n ast.Node) bool {
			switch n := n.(type) {
			case *ast.BlockStmt:
				f(n.List)
			case *ast.CaseClause:
				f(n.Body)
			case *ast.CommClause:
				f(n.Body)
			}
			return true
		})
	}
}

// releaseStmt returns the handle variable a statement releases, not in a
// defer, and the name of the release method.
func releaseStmt(info *types.Info, stmt ast.Stmt) (*types.Var, string) {
	var e ast.Expr
	switch s := stmt.(type) {
	case *ast.ExprStmt:
		e = s.X
	case *ast.AssignStmt:
		if len(s.Rhs) == 1 {
			e = s.Rhs[0]
		}
	case *ast.IfStmt:
		if a, ok := s.Init.(*ast.AssignStmt); ok && len(a.Rhs) == 1 {
			e = a.Rhs[0]
		}
	}
	call, ok := ast.Unparen(e).(*ast.CallExpr)
	if !ok {
		return nil, ""
	}
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok || !isRelease(sel.Sel.Name) {
		return nil, ""
	}
	v := objOf(info, sel.X)
	if v == nil || destructor(v.Type()) == "" {
		return nil, ""
	}
	return v, sel.Sel.Name
}

// valueOnly reports whether n reads the Go value of the handle v without
// handing it to anything that could use it: a comparison with a constant, as
// in h != 0, or a call that only formats its arguments, as in
// t.Errorf("%#x", uintptr(h)). Those read the variable, not the handle.
func valueOnly(info *types.Info, n ast.Node, v *types.Var) bool {
	switch n := n.(type) {
	case *ast.BinaryExpr:
		return (n.Op == token.EQL || n.Op == token.NEQ) &&
			(objOf(info, unconvert(info, n.X)) == v && isConst(info, n.Y) || objOf(info, unconvert(info, n.Y)) == v && isConst(info, n.X))
	case *ast.CallExpr:
		fn := callee(info, n)
		if fn == nil || fn.Pkg() == nil || !formats(fn) {
			return false
		}
		for _, arg := range n.Args {
			if !isConst(info, arg) && objOf(info, unconvert(info, arg)) != v && uses(info, arg, v) {
				return false
			}
		}
		return true
	}
	return false
}

// formats reports whether fn is a print or format function of fmt, log or
// testing, which read their arguments only to print them.
func formats(fn *types.Func) bool {
	switch fn.Pkg().Path() {
	case "fmt", "log", "testing":
	default:
		return false
	}
	switch strings.TrimSuffix(strings.TrimSuffix(fn.Name(), "f"), "ln") {
	case "Print", "Sprint", "Fprint", "Error", "Log", "Fatal", "Skip", "Panic", "Append":
		return true
	}
	return false
}

// isConst reports whether e is a constant or nil.
func isConst(info *types.Info, e ast.Expr) bool {
	tv, ok := info.Types[e]
	return ok && (tv.Value != nil || tv.IsNil())
}

// releaseFailed reports whether the code after s, an if statement whose init
// releases a handle into err, runs only when the release failed: the body
// always leaves, and the condition expects the release to fail.
func releaseFailed(info *types.Info, s *ast.IfStmt) bool {
	if len(s.Body.List) == 0 || !expectsFailure(info, s) {
		return false
	}
	switch last := s.Body.List[len(s.Body.List)-1].(type) {
	case *ast.ReturnStmt:
		return true
	case *ast.ExprStmt:
		c, ok := last.X.(*ast.CallExpr)
		return ok && noReturn(info, c)
	}
	return false
}

// expectsFailure reports whether the init of s assigns an error and one
// alternative of its condition holds when that error is not the expected
// failure: err == nil, err != target or !errors.Is(err, target), with target
// not nil.
func expectsFailure(info *types.Info, s *ast.IfStmt) bool {
	init, ok := s.Init.(*ast.AssignStmt)
	if !ok || len(init.Lhs) == 0 {
		return false
	}
	err := objOf(info, init.Lhs[len(init.Lhs)-1])
	if err == nil || !isError(err.Type()) {
		return false
	}
	var alts []ast.Expr
	var split func(e ast.Expr)
	split = func(e ast.Expr) {
		if b, ok := ast.Unparen(e).(*ast.BinaryExpr); ok && b.Op == token.LOR {
			split(b.X)
			split(b.Y)
			return
		}
		alts = append(alts, ast.Unparen(e))
	}
	split(s.Cond)
	for _, e := range alts {
		switch e := e.(type) {
		case *ast.BinaryExpr:
			other := e.Y
			if objOf(info, e.Y) == err {
				other = e.X
			} else if objOf(info, e.X) != err {
				continue
			}
			if e.Op == token.EQL && isNil(info, other) || e.Op == token.NEQ && !isNil(info, other) {
				return true
			}
		case *ast.UnaryExpr:
			c, ok := ast.Unparen(e.X).(*ast.CallExpr)
			if e.Op == token.NOT && ok && isPkgFunc(info, c, "errors", "Is") &&
				len(c.Args) == 2 && objOf(info, c.Args[0]) == err && !isNil(info, c.Args[1]) {
				return true
			}
		}
	}
	return false
}

// deliberateUse reports whether use, in stmt, of a released handle v is a
// test of the release rather than a mistake. It applies only to a handle
// that records its release and rejects later calls (see recordsRelease): a
// second release, a call whose error an if expects to be a failure, or a
// handle never set at all, declared nil in before.
func deliberateUse(info *types.Info, before []ast.Stmt, stmt ast.Stmt, v *types.Var, use *ast.Ident) bool {
	if !recordsRelease(v.Type()) {
		return false
	}
	if declaredNil(info, before, v) {
		return true
	}
	found := false
	ast.Inspect(stmt, func(n ast.Node) bool {
		s, ok := n.(ast.Stmt)
		if !ok || found || use.Pos() < s.Pos() || use.Pos() >= s.End() {
			return !found
		}
		if w, _ := releaseStmt(info, s); w == v {
			found = true
		}
		if is, ok := s.(*ast.IfStmt); ok && is.Init != nil && use.Pos() < is.Init.End() && expectsFailure(info, is) {
			found = true
		}
		return !found
	})
	return found
}

// recordsRelease reports whether a handle of type t is a Go value that can
// record its release: a pointer, or a struct holding a pointer to a struct its
// copies share, as cudamem.Slice holds its allocation. A bare driver handle
// such as driver.DevicePtr records nothing; releasing it twice reaches C twice.
func recordsRelease(t types.Type) bool {
	switch u := t.Underlying().(type) {
	case *types.Pointer:
		return true
	case *types.Struct:
		for f := range u.Fields() {
			if p, ok := f.Type().Underlying().(*types.Pointer); ok {
				if _, ok := p.Elem().Underlying().(*types.Struct); ok {
					return true
				}
			}
		}
	}
	return false
}

// declaredNil reports whether stmts declare v with no value and never assign
// it.
func declaredNil(info *types.Info, stmts []ast.Stmt, v *types.Var) bool {
	declared := false
	for _, s := range stmts {
		if assigns(info, s, v) {
			return false
		}
		d, ok := s.(*ast.DeclStmt)
		if !ok {
			continue
		}
		g, ok := d.Decl.(*ast.GenDecl)
		if !ok {
			continue
		}
		for _, spec := range g.Specs {
			if vs, ok := spec.(*ast.ValueSpec); ok && len(vs.Values) == 0 {
				for _, name := range vs.Names {
					if info.Defs[name] == v {
						declared = true
					}
				}
			}
		}
	}
	return declared
}

// assigns reports whether stmt assigns v a new value.
func assigns(info *types.Info, stmt ast.Stmt, v *types.Var) bool {
	a, ok := stmt.(*ast.AssignStmt)
	if !ok {
		return false
	}
	for _, l := range a.Lhs {
		if objOf(info, l) == v {
			return true
		}
	}
	return false
}

var sizeUnitsAnalyzer = &analysis.Analyzer{
	Name: "sizeunits",
	Doc: `check workspace sizes from legacy cuSOLVER helpers are counted in elements

The legacy cuSOLVER *_bufferSize helpers, such as DgetrfBufferSize, return a
count of elements of the routine's type, not bytes; the X and IRS helpers
return bytes. sizeunits reports such a count used as a byte count
(MemAllocBytes, MemAlloc[byte]) or as a count of a different element type.
The first letter of the routine gives its type: S float32, D float64,
C complex64, Z complex128.`,
	Run: runSizeUnits,
}

var legacyBufferSizeRE = regexp.MustCompile(`^([SDCZ])[a-z][a-z0-9]*BufferSize$`)

var legacyElem = map[string]string{"S": "float32", "D": "float64", "C": "complex64", "Z": "complex128"}

func runSizeUnits(pass *analysis.Pass) (any, error) {
	info := pass.TypesInfo
	eachFunc(pass, func(body *ast.BlockStmt) {
		counts := make(map[*types.Var]string) // variable -> routine
		inspectShallow(body, func(n ast.Node) bool {
			assign, ok := n.(*ast.AssignStmt)
			if !ok || len(assign.Rhs) != 1 {
				return true
			}
			call, ok := assign.Rhs[0].(*ast.CallExpr)
			if !ok {
				return true
			}
			if fn := calleeIn(info, call, "cusolver"); legacyBufferSizeRE.MatchString(fn) {
				if v := objOf(info, assign.Lhs[0]); v != nil {
					counts[v] = fn
				}
			}
			return true
		})
		if len(counts) == 0 {
			return
		}
		for _, call := range calls(body) {
			fn := callee(info, call)
			if fn == nil || !inPkg(fn.Pkg(), "driver") || len(call.Args) == 0 {
				continue
			}
			v := objOf(info, unconvert(info, call.Args[0]))
			routine, ok := counts[v]
			if !ok {
				continue
			}
			want := legacyElem[legacyBufferSizeRE.FindStringSubmatch(routine)[1]]
			var got string
			switch fn.Name() {
			case "MemAllocBytes", "MemAllocManagedBytes":
				got = "byte"
			case "MemAlloc", "MemAllocManaged":
				got = typeArg(info, call)
			default:
				continue
			}
			if got != "" && got != want {
				pass.Reportf(call.Args[0].Pos(), "%s from %s counts %s elements, not %ss; allocate MemAlloc[%s](%s)",
					v.Name(), routine, want, got, want, v.Name())
			}
		}
	})
	return nil, nil
}

// typeArg returns the type argument of a call of an instantiated generic
// function, as it would be written, or "".
func typeArg(info *types.Info, call *ast.CallExpr) string {
	fun := ast.Unparen(call.Fun)
	if ix, ok := fun.(*ast.IndexExpr); ok {
		fun = ix.X
	}
	var id *ast.Ident
	switch f := fun.(type) {
	case *ast.Ident:
		id = f
	case *ast.SelectorExpr:
		id = f.Sel
	}
	inst, ok := info.Instances[id]
	if !ok || inst.TypeArgs.Len() == 0 {
		return ""
	}
	t := inst.TypeArgs.At(0)
	if b, ok := t.(*types.Basic); ok && b.Kind() == types.Uint8 {
		return "byte"
	}
	return t.String()
}

var sizeTruncAnalyzer = &analysis.Analyzer{
	Name: "sizetrunc",
	Doc: `check that size_t workspace sizes are bounds-checked before narrowing

A workspace size returned as a uint64 or uint, such as cuTENSOR's
EstimateWorkspaceSize, can exceed what an int holds on the host or what a
narrower parameter accepts. sizetrunc reports a conversion of such a size to
a signed or narrower integer type in a function that never compares it first.`,
	Run: runSizeTrunc,
}

func runSizeTrunc(pass *analysis.Pass) (any, error) {
	info := pass.TypesInfo
	eachFunc(pass, func(body *ast.BlockStmt) {
		sizes := make(map[*types.Var]string)
		inspectShallow(body, func(n ast.Node) bool {
			assign, ok := n.(*ast.AssignStmt)
			if !ok || len(assign.Rhs) != 1 {
				return true
			}
			call, ok := assign.Rhs[0].(*ast.CallExpr)
			if !ok {
				return true
			}
			fn := callee(info, call)
			if fn == nil || !inNvidia(fn.Pkg()) || !(strings.Contains(fn.Name(), "WorkspaceSize") || strings.Contains(fn.Name(), "BufferSize")) {
				return true
			}
			for _, l := range assign.Lhs {
				if v := objOf(info, l); v != nil && isWideUnsigned(v.Type()) {
					sizes[v] = fn.Name()
				}
			}
			return true
		})
		if len(sizes) == 0 {
			return
		}
		checked := make(map[*types.Var]token.Pos)
		inspectShallow(body, func(n ast.Node) bool {
			switch n := n.(type) {
			case *ast.BinaryExpr:
				switch n.Op {
				case token.LSS, token.GTR, token.LEQ, token.GEQ:
					for _, e := range []ast.Expr{n.X, n.Y} {
						if v := objOf(info, e); v != nil {
							if _, ok := sizes[v]; ok && checked[v] == token.NoPos {
								checked[v] = n.Pos()
							}
						}
					}
				}
			case *ast.CallExpr:
				tv, ok := info.Types[n.Fun]
				if !ok || !tv.IsType() || len(n.Args) != 1 {
					return true
				}
				v := objOf(info, n.Args[0])
				from, ok := sizes[v]
				if !ok || !narrows(tv.Type) {
					return true
				}
				if c := checked[v]; c == token.NoPos || c > n.Pos() {
					pass.Reportf(n.Pos(), "%s from %s is converted to %s without a bounds check; a size_t that does not fit is silently truncated",
						v.Name(), from, tv.Type)
				}
			}
			return true
		})
	})
	return nil, nil
}

func isWideUnsigned(t types.Type) bool {
	b, ok := t.Underlying().(*types.Basic)
	return ok && (b.Kind() == types.Uint64 || b.Kind() == types.Uint || b.Kind() == types.Uintptr)
}

// narrows reports whether converting a wide unsigned value to t can lose
// it: t is signed, or narrower than 64 bits.
func narrows(to types.Type) bool {
	b, ok := to.Underlying().(*types.Basic)
	if !ok || b.Info()&types.IsInteger == 0 {
		return false
	}
	switch b.Kind() {
	case types.Uint64, types.Uint, types.Uintptr:
		return false
	}
	return true
}
