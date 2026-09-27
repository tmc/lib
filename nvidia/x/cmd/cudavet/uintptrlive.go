package main

import (
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/analysis"
)

var uintptrLiveAnalyzer = &analysis.Analyzer{
	Name: "uintptrlive",
	Doc: `check that Go memory stored as a uintptr is kept alive

The garbage collector does not trace a uintptr. After
p.KernelParams = uintptr(unsafe.Pointer(&s[0])), s is dead by liveness
analysis while the driver still reads through the stored address.
//go:uintptrescapes does not help: it covers only a conversion written in the
argument list of the marked call, not a number passed on through a variable
or a struct.

uintptrlive reports uintptr(unsafe.Pointer(p)), p the address of a local
variable's memory (&x, &x.f, &s[i], unsafe.SliceData(s),
unsafe.StringData(s)), when all of these hold:

  - the value is stored, not passed directly as a call argument: in a field
    or element of a local variable, in a composite literal that is not
    stored outside the function, or in a local variable that is then passed
    as a uintptr argument, stored in a field or element, or returned;
  - no runtime.KeepAlive of the referent, or of a local variable holding it
    as a Go pointer, is deferred or follows the last use of the holder that
    could hand the address on: passing it or its field's address, or
    calling its method, but not reading or writing one of its fields;
  - no runtime.Pinner pins the referent.

The KeepAlive and Pin calls may be anywhere in the outermost enclosing
function, including its function literals.

A value only compared, printed or converted back to unsafe.Pointer is not
reported. Neither is memory the function does not own: a package-level
variable, memory reached through a pointer's field (s.buf in a method of
*S), or a value stored in memory reached from a parameter, receiver or
global, which outlives the call and whose referent its owner must keep
alive. KeepAlive of the uintptr's holder does not count, since a uintptr
keeps nothing alive.`,
	Run: runUintptrLive,
}

func runUintptrLive(pass *analysis.Pass) (any, error) {
	info := pass.TypesInfo
	eachOuterFunc(pass, func(outer *ast.BlockStmt) {
		withStackAll(outer, func(n ast.Node, stack []ast.Node) {
			conv, ok := n.(*ast.CallExpr)
			if !ok || !isUintptrConv(info, conv) {
				return
			}
			root, ok := goReferent(info, conv.Args[0])
			if !ok || root != nil && isGlobal(root) {
				return
			}
			inner := outer
			for _, s := range stack {
				if lit, ok := s.(*ast.FuncLit); ok {
					inner = lit.Body
				}
			}
			holder, field, end, ok := storedIn(info, conv, stack)
			if !ok || holder != nil && !declaredIn(holder, inner) {
				return
			}
			if !field && !passedOn(info, outer, holder) {
				return
			}
			if root != nil {
				keep := owners(info, outer, root)
				if keptAlive(info, outer, keep, handedOn(info, outer, holder, end)) || pinnedVar(info, outer, root) {
					return
				}
			}
			what := "Go memory"
			if root != nil {
				what = root.Name()
			}
			pass.Reportf(conv.Pos(), "%s stored as a uintptr is not traced by the garbage collector; defer runtime.KeepAlive(%s) or pin it", what, what)
		})
	})
	return nil, nil
}

// isUintptrConv reports whether call converts a value to uintptr.
func isUintptrConv(info *types.Info, call *ast.CallExpr) bool {
	return len(call.Args) == 1 && isConv(info, call) && types.Identical(info.TypeOf(call), types.Typ[types.Uintptr])
}

// isConv reports whether call is a type conversion.
func isConv(info *types.Info, call *ast.CallExpr) bool {
	tv, ok := info.Types[call.Fun]
	return ok && tv.IsType()
}

// isIntConv reports whether call is a conversion to an integer type, which
// keeps a uintptr a number.
func isIntConv(info *types.Info, call *ast.CallExpr) bool {
	if !isConv(info, call) {
		return false
	}
	b, ok := info.TypeOf(call).Underlying().(*types.Basic)
	return ok && b.Info()&types.IsInteger != 0
}

// goReferent reports whether e is unsafe.Pointer applied to the address of
// Go memory owned by the function, and returns the variable that holds the
// memory. The variable is nil for the address of a composite literal, which
// nothing else holds.
func goReferent(info *types.Info, e ast.Expr) (*types.Var, bool) {
	call, ok := ast.Unparen(e).(*ast.CallExpr)
	if !ok || len(call.Args) != 1 || !isUnsafe(info, call.Fun, "Pointer") {
		return nil, false
	}
	var x ast.Expr
	switch p := ast.Unparen(call.Args[0]).(type) {
	case *ast.UnaryExpr:
		if p.Op != token.AND {
			return nil, false
		}
		x = p.X
	case *ast.CallExpr:
		if len(p.Args) != 1 || !isUnsafe(info, p.Fun, "SliceData") && !isUnsafe(info, p.Fun, "StringData") {
			return nil, false
		}
		x = p.Args[0]
	default:
		return nil, false
	}
	if throughPointer(info, x) {
		return nil, false
	}
	return rootVar(info, x), true
}

// rootVar returns the variable at the base of e: s in &s[i].f, x in x.f[1:],
// or nil if there is none.
func rootVar(info *types.Info, e ast.Expr) *types.Var {
	for {
		switch x := unconvert(info, e).(type) {
		case *ast.Ident:
			v, _ := info.ObjectOf(x).(*types.Var)
			return v
		case *ast.SelectorExpr:
			if v, ok := info.Uses[x.Sel].(*types.Var); ok && !v.IsField() {
				return v // pkg.V
			}
			e = x.X
		case *ast.IndexExpr:
			e = x.X
		case *ast.SliceExpr:
			e = x.X
		case *ast.StarExpr:
			e = x.X
		case *ast.UnaryExpr:
			if x.Op != token.AND {
				return nil
			}
			e = x.X
		case *ast.CallExpr:
			if len(x.Args) != 1 || !isUnsafe(info, x.Fun, "Pointer") && !isUnsafe(info, x.Fun, "SliceData") && !isUnsafe(info, x.Fun, "StringData") {
				return nil
			}
			e = x.Args[0]
		default:
			return nil
		}
	}
}

// throughPointer reports whether e reaches its memory through a field of a
// pointer, as s.buf[0] does with s a *S: memory owned by an object whose
// lifetime the function cannot see.
func throughPointer(info *types.Info, e ast.Expr) bool {
	for {
		switch x := ast.Unparen(e).(type) {
		case *ast.SelectorExpr:
			sel, ok := info.Selections[x]
			if !ok {
				return false
			}
			if sel.Kind() == types.FieldVal && sel.Indirect() {
				return true
			}
			e = x.X
		case *ast.IndexExpr:
			e = x.X
		case *ast.SliceExpr:
			e = x.X
		default:
			return false
		}
	}
}

func isGlobal(v *types.Var) bool {
	return v.Pkg() != nil && v.Parent() == v.Pkg().Scope()
}

// storedIn reports whether the value of conv, possibly through arithmetic
// and integer conversions, is stored rather than passed as a call argument.
// It returns the variable holding it, or nil for a composite literal held by
// no variable; whether the value is in a field, element or composite literal
// of that variable rather than the variable itself; and the end of the
// enclosing statement.
func storedIn(info *types.Info, conv *ast.CallExpr, stack []ast.Node) (holder *types.Var, field bool, end token.Pos, ok bool) {
	for i := len(stack) - 1; i >= 0; i-- {
		if s, isStmt := stack[i].(ast.Stmt); isStmt {
			end = s.End()
			break
		}
	}
	child := ast.Node(conv)
	for i := len(stack) - 1; i >= 0; i-- {
		switch p := stack[i].(type) {
		case *ast.ParenExpr, *ast.BinaryExpr:
			child = p
			continue
		case *ast.CallExpr:
			if isIntConv(info, p) {
				child = p
				continue
			}
			return nil, false, end, false // a call argument or a pointer again
		case *ast.KeyValueExpr:
			if p.Value != child {
				return nil, false, end, false
			}
			child = p
			continue
		case *ast.CompositeLit:
			return literalHolder(info, stack[:i]), true, end, true
		case *ast.AssignStmt:
			if l := lhsOf(p, child); l != nil {
				if id, isIdent := ast.Unparen(l).(*ast.Ident); isIdent {
					v, _ := info.ObjectOf(id).(*types.Var)
					return v, false, end, v != nil
				}
				v := rootVar(info, l)
				return v, true, end, v != nil
			}
		case *ast.ValueSpec:
			if v := specVar(info, p, child); v != nil {
				return v, false, end, true
			}
		}
		return nil, false, end, false
	}
	return nil, false, end, false
}

// lhsOf returns the left-hand side assign stores rhs in, or nil.
func lhsOf(assign *ast.AssignStmt, rhs ast.Node) ast.Expr {
	if len(assign.Lhs) != len(assign.Rhs) {
		return nil
	}
	for i, r := range assign.Rhs {
		if r == rhs {
			return assign.Lhs[i]
		}
	}
	return nil
}

// specVar returns the variable spec declares with value v, or nil.
func specVar(info *types.Info, spec *ast.ValueSpec, v ast.Node) *types.Var {
	if len(spec.Names) != len(spec.Values) {
		return nil
	}
	for i, r := range spec.Values {
		if r == v {
			w, _ := info.Defs[spec.Names[i]].(*types.Var)
			return w
		}
	}
	return nil
}

// literalHolder returns the variable a composite literal, possibly nested or
// with its address taken, is assigned to, given the literal's ancestors.
func literalHolder(info *types.Info, stack []ast.Node) *types.Var {
	var child ast.Node
	for i := len(stack) - 1; i >= 0; i-- {
		switch p := stack[i].(type) {
		case *ast.UnaryExpr, *ast.ParenExpr, *ast.KeyValueExpr, *ast.CompositeLit:
			child = p
			continue
		case *ast.AssignStmt:
			if l := lhsOf(p, child); l != nil {
				return rootVar(info, l)
			}
		case *ast.ValueSpec:
			return specVar(info, p, child)
		}
		return nil
	}
	return nil
}

// passedOn reports whether the uintptr in the local variable v, or in a
// variable computed from it, leaves the function's arithmetic: passed as a
// uintptr argument, stored in a field or element, or returned.
func passedOn(info *types.Info, body *ast.BlockStmt, v *types.Var) bool {
	seen := map[*types.Var]bool{v: true}
	work := []*types.Var{v}
	for len(work) > 0 {
		v := work[0]
		work = work[1:]
		found := false
		withStackAll(body, func(n ast.Node, stack []ast.Node) {
			id, ok := n.(*ast.Ident)
			if found || !ok || info.Uses[id] != v {
				return
			}
			child := ast.Node(id)
			for i := len(stack) - 1; i >= 0; i-- {
				switch p := stack[i].(type) {
				case *ast.BinaryExpr:
					if isComparison(p.Op) {
						return
					}
					child = p
					continue
				case *ast.ParenExpr, *ast.KeyValueExpr:
					child = p
					continue
				case *ast.CallExpr:
					if isIntConv(info, p) {
						child = p
						continue
					}
					found = !isConv(info, p) && uintptrParam(info, p, child)
				case *ast.CompositeLit, *ast.ReturnStmt, *ast.SendStmt:
					found = true
				case *ast.AssignStmt:
					if l := lhsOf(p, child); l != nil {
						switch w := objOf(info, l); {
						case w == nil:
							found = true // a field or element
						case !seen[w]:
							seen[w] = true
							work = append(work, w)
						}
					}
				case *ast.ValueSpec:
					if w := specVar(info, p, child); w != nil && !seen[w] {
						seen[w] = true
						work = append(work, w)
					}
				}
				return
			}
		})
		if found {
			return true
		}
	}
	return false
}

func isComparison(op token.Token) bool {
	switch op {
	case token.EQL, token.NEQ, token.LSS, token.LEQ, token.GTR, token.GEQ:
		return true
	}
	return false
}

// uintptrParam reports whether arg is passed to a parameter of call typed
// uintptr, including a variadic ...uintptr.
func uintptrParam(info *types.Info, call *ast.CallExpr, arg ast.Node) bool {
	sig, ok := info.TypeOf(call.Fun).Underlying().(*types.Signature)
	if !ok {
		return false
	}
	for i, a := range call.Args {
		if a != arg {
			continue
		}
		var t types.Type
		switch n := sig.Params().Len(); {
		case sig.Variadic() && i >= n-1:
			t = sig.Params().At(n - 1).Type().(*types.Slice).Elem()
		case i < n:
			t = sig.Params().At(i).Type()
		default:
			return false
		}
		b, ok := t.Underlying().(*types.Basic)
		return ok && b.Kind() == types.Uintptr
	}
	return false
}

// handedOn returns the position of the last use of holder in body that could
// hand the stored address on, and at least end. Reading or writing a field
// of the holder cannot; passing the holder, calling its method or taking a
// field's address can.
func handedOn(info *types.Info, body *ast.BlockStmt, holder *types.Var, end token.Pos) token.Pos {
	last := end
	if holder == nil {
		return last
	}
	withStackAll(body, func(n ast.Node, stack []ast.Node) {
		id, ok := n.(*ast.Ident)
		if !ok || info.Uses[id] != holder || id.Pos() <= last {
			return
		}
		if len(stack) >= 2 {
			if sel, ok := stack[len(stack)-1].(*ast.SelectorExpr); ok && sel.X == id {
				switch p := stack[len(stack)-2].(type) {
				case *ast.UnaryExpr:
					if p.Op == token.AND {
						last = id.Pos()
					}
				case *ast.CallExpr:
					if p.Fun == sel && isMethod(info, sel) {
						last = id.Pos()
					}
				}
				return
			}
		}
		last = id.Pos()
	})
	return last
}

func isMethod(info *types.Info, sel *ast.SelectorExpr) bool {
	s, ok := info.Selections[sel]
	return ok && s.Kind() == types.MethodVal
}

// owners returns root and the local variables of body assigned a value that
// holds root as a Go pointer, outside any uintptr conversion.
func owners(info *types.Info, body *ast.BlockStmt, root *types.Var) map[*types.Var]bool {
	keep := map[*types.Var]bool{root: true}
	for v, rhs := range assignments(info, body) {
		if v == nil || v == root || goPointer(v.Type(), nil) == "" {
			continue
		}
		for _, r := range rhs {
			if r != nil && holdsTraced(info, r, root) {
				keep[v] = true
			}
		}
	}
	return keep
}

// holdsTraced reports whether e refers to v outside a conversion to uintptr.
func holdsTraced(info *types.Info, e ast.Expr, v *types.Var) bool {
	found := false
	ast.Inspect(e, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.CallExpr:
			if isUintptrConv(info, n) {
				return false
			}
		case *ast.Ident:
			if info.Uses[n] == v {
				found = true
			}
		}
		return !found
	})
	return found
}

// keptAlive reports whether body calls runtime.KeepAlive on one of vars in a
// defer, or after last.
func keptAlive(info *types.Info, body *ast.BlockStmt, vars map[*types.Var]bool, last token.Pos) bool {
	found := false
	withStackAll(body, func(n ast.Node, stack []ast.Node) {
		call, ok := n.(*ast.CallExpr)
		if found || !ok || len(call.Args) != 1 || !isPkgFunc(info, call, "runtime", "KeepAlive") || !vars[rootVar(info, call.Args[0])] {
			return
		}
		if call.Pos() > last {
			found = true
			return
		}
		for _, s := range stack {
			if _, ok := s.(*ast.DeferStmt); ok {
				found = true
			}
		}
	})
	return found
}

// pinnedVar reports whether body pins v with a runtime.Pinner.
func pinnedVar(info *types.Info, body *ast.BlockStmt, v *types.Var) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if ok && len(call.Args) == 1 && isPkgFunc(info, call, "runtime", "Pin") && rootVar(info, call.Args[0]) == v {
			found = true
		}
		return !found
	})
	return found
}
