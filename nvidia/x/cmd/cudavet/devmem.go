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

A binding parameter that takes a device buffer should be a DevicePtr. One typed
uintptr means the generator did not classify the buffer as device memory, and
the caller has to convert, losing the type. devptruintptr reports uintptr(d),
d a DevicePtr, passed to a uintptr parameter of an nvidia binding, naming the
parameter so that the classification can be fixed in the generator.`,
	Run: runDevptrUintptr,
}

func runDevptrUintptr(pass *analysis.Pass) (any, error) {
	info := pass.TypesInfo
	eachFunc(pass, func(body *ast.BlockStmt) {
		for _, call := range calls(body) {
			fn := callee(info, call)
			if fn == nil || !isBinding(fn.Pkg()) || fn.Pkg() == pass.Pkg || !fn.Exported() {
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

A handle released with Destroy, Free or Unload names nothing afterwards; in C,
using it is a use after free. useafterdestroy reports a use of a handle that
follows its release in the same block, before the variable is assigned again.`,
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
			for _, next := range list[i+1:] {
				if assigns(info, next, v) {
					break
				}
				var use *ast.Ident
				inspectShallow(next, func(n ast.Node) bool {
					if id, ok := n.(*ast.Ident); ok && use == nil && info.Uses[id] == v {
						use = id
					}
					return use == nil
				})
				if use != nil {
					pass.Reportf(use.Pos(), "%s is used after %s.%s", v.Name(), v.Name(), name)
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
