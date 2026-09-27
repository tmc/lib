package main

import (
	"go/ast"
	"go/token"
	"go/types"
	"slices"
	"strings"

	"golang.org/x/tools/go/analysis"
)

var closeCopyAnalyzer = &analysis.Analyzer{
	Name: "closecopy",
	Doc: `check that a handle is not released through one copy and used through another

The Close of a generated handle, like driver.Stream's, has a pointer receiver
and sets the handle it is called on to zero, which makes a second Close a
no-op. Only that variable is zeroed: a copy made before, by h2 := h, a struct
copy, a range value or a call that takes the handle by value, still holds the
released handle, and closing it releases it a second time. closecopy follows
the copies within one function and reports a handle released through one of
them and then released or used through another. A copy that is the only one
released, a handle assigned or addressed after it is copied, and releases on
exclusive branches, or on a path that leaves before the other runs, are not
reported.

A handle type is a value type of an nvidia package whose release method,
Close, Destroy, Free or Unload, has a pointer receiver.`,
	Run: runCloseCopy,
}

// A ccPath names a place that holds a handle: a variable and a chain of
// field selections and indexes below it, as in s.list*[] for s.list[i]. A
// step through a pointer, slice or map is marked with *: the place it leads
// to is shared, not copied, when the variable is. A copy passed to a
// function has no variable; its suffix names the call.
type ccPath struct {
	root   *types.Var
	suffix string
}

// under reports whether p is m or a place inside m, and returns the rest.
func (p ccPath) under(m ccPath) (string, bool) {
	if p.root != m.root || !strings.HasPrefix(p.suffix, m.suffix) {
		return "", false
	}
	rest := p.suffix[len(m.suffix):]
	if rest != "" && rest[0] != '.' && rest[0] != '[' && rest[0] != '*' {
		return "", false
	}
	return rest, true
}

func overlaps(p, q ccPath) bool {
	_, a := p.under(q)
	_, b := q.under(p)
	return a || b
}

// A ccEvent is a release of a handle or a use of one.
type ccEvent struct {
	path    ccPath
	release bool
	desc    string     // what the report calls it: h, h.Close, closeAll(h)
	node    ast.Node   // the call or the expression
	stack   []ast.Node // the ancestors of node
	at      token.Pos  // when it runs: deferred calls run after the body
	deferAt token.Pos  // the defer statement, or NoPos
}

type ccCopy struct {
	dst, src ccPath
	at       token.Pos
}

type ccKill struct {
	path ccPath
	at   token.Pos
}

func runCloseCopy(pass *analysis.Pass) (any, error) {
	closers := paramClosers(pass)
	eachFunc(pass, func(body *ast.BlockStmt) {
		checkCloseCopy(pass, body, closers)
	})
	return nil, nil
}

// isCloseHandle reports whether t is a handle type whose release zeroes
// only its receiver.
func isCloseHandle(t types.Type) bool {
	named, ok := types.Unalias(t).(*types.Named)
	if !ok {
		return false
	}
	m, _, _ := types.LookupFieldOrMethod(types.NewPointer(named), false, named.Obj().Pkg(), destructor(named))
	fn, ok := m.(*types.Func)
	return ok && isPtrRecvFunc(fn)
}

func isPtrRecvFunc(fn *types.Func) bool {
	_, ok := fn.Type().(*types.Signature).Recv().Type().(*types.Pointer)
	return ok
}

// paramClosers returns, for each function of the package, which of its
// parameters it releases: parameters of a handle type, taken by value.
func paramClosers(pass *analysis.Pass) map[*types.Func][]bool {
	info := pass.TypesInfo
	out := make(map[*types.Func][]bool)
	for _, f := range srcFiles(pass) {
		for _, decl := range f.Decls {
			fd, ok := decl.(*ast.FuncDecl)
			if !ok || fd.Body == nil {
				continue
			}
			fn, _ := info.Defs[fd.Name].(*types.Func)
			if fn == nil {
				continue
			}
			params := fn.Type().(*types.Signature).Params()
			closes := make([]bool, params.Len())
			found := false
			for i := range params.Len() {
				p := params.At(i)
				ast.Inspect(fd.Body, func(n ast.Node) bool {
					if path, _, ok := releaseCall(info, n); ok && path == (ccPath{root: p}) {
						closes[i], found = true, true
					}
					return !closes[i]
				})
			}
			if found {
				out[fn] = closes
			}
		}
	}
	return out
}

// releaseCall reports whether n is a call of a release method on a handle,
// and returns where the handle is held.
func releaseCall(info *types.Info, n ast.Node) (ccPath, string, bool) {
	call, ok := n.(*ast.CallExpr)
	if !ok {
		return ccPath{}, "", false
	}
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok || !isRelease(sel.Sel.Name) {
		return ccPath{}, "", false
	}
	// A package's function has no receiver; the operand is checked first.
	if !isCloseHandle(info.TypeOf(sel.X)) {
		return ccPath{}, "", false
	}
	p, ok := pathOf(info, sel.X)
	return p, sel.Sel.Name, ok
}

// pathOf returns the place e names: a variable, a field of one, or an
// element of one.
func pathOf(info *types.Info, e ast.Expr) (ccPath, bool) {
	switch e := ast.Unparen(e).(type) {
	case *ast.Ident:
		v, ok := info.ObjectOf(e).(*types.Var)
		return ccPath{root: v}, ok
	case *ast.SelectorExpr:
		p, ok := pathOf(info, e.X)
		if !ok {
			return p, false
		}
		if info.Selections[e].Indirect() {
			p.suffix += "*"
		}
		p.suffix += "." + e.Sel.Name
		return p, true
	case *ast.IndexExpr:
		p, ok := pathOf(info, e.X)
		if _, array := info.TypeOf(e.X).Underlying().(*types.Array); !array {
			p.suffix += "*"
		}
		p.suffix += "[]"
		return p, ok
	}
	return ccPath{}, false
}

func checkCloseCopy(pass *analysis.Pass, body *ast.BlockStmt, closers map[*types.Func][]bool) {
	info := pass.TypesInfo
	var (
		copies []ccCopy
		events []ccEvent
		kills  []ccKill
	)
	addCopy := func(dst, src ast.Expr, at token.Pos) {
		d, ok1 := pathOf(info, dst)
		s, ok2 := pathOf(info, src)
		if ok1 && ok2 {
			copies = append(copies, ccCopy{d, s, at})
		}
	}
	kill := func(e ast.Expr, at token.Pos) {
		if p, ok := pathOf(info, e); ok {
			kills = append(kills, ccKill{p, at})
		}
	}
	end := body.End()
	var stack []ast.Node
	ast.Inspect(body, func(n ast.Node) bool {
		if n == nil {
			stack = stack[:len(stack)-1]
			return true
		}
		// Of the function literals only a deferred one runs in this
		// function's frame at a known time.
		if lit, ok := n.(*ast.FuncLit); ok && deferOf(append(stack, lit)) == nil {
			return false
		}
		at, deferAt := n.Pos(), token.NoPos
		if d := deferOf(stack); d != nil {
			// Deferred calls run after the body, last deferred first,
			// and each in its own order.
			deferAt = d.Pos()
			at = end + (end-d.Pos())*(end-body.Pos()+1) + (n.Pos() - d.Pos())
		}
		switch n := n.(type) {
		case *ast.AssignStmt:
			for _, l := range n.Lhs {
				if n.Tok != token.DEFINE {
					kill(l, n.Pos())
				}
			}
			if len(n.Lhs) == len(n.Rhs) {
				for i := range n.Lhs {
					addCopy(n.Lhs[i], n.Rhs[i], n.Pos())
				}
			}
		case *ast.ValueSpec:
			if len(n.Names) == len(n.Values) {
				for i := range n.Names {
					addCopy(n.Names[i], n.Values[i], n.Pos())
				}
			}
		case *ast.UnaryExpr:
			if n.Op == token.AND {
				kill(n.X, n.Pos())
			}
		case *ast.RangeStmt:
			if n.Value != nil {
				s, ok1 := pathOf(info, n.X)
				d, ok2 := pathOf(info, n.Value)
				if ok1 && ok2 {
					if _, array := info.TypeOf(n.X).Underlying().(*types.Array); !array {
						s.suffix += "*"
					}
					s.suffix += "[]"
					copies = append(copies, ccCopy{d, s, n.Pos()})
				}
			}
		case *ast.CallExpr:
			if p, name, ok := releaseCall(info, n); ok {
				sel := n.Fun.(*ast.SelectorExpr)
				events = append(events, ccEvent{p, true, render(pass.Fset, sel.X) + "." + name, n, slices.Clone(stack), at, deferAt})
				break
			}
			fn := callee(info, n)
			for i, closes := range closers[fn] {
				if !closes {
					continue
				}
				s, ok := pathOf(info, n.Args[i])
				if !ok {
					continue
				}
				// The parameter is a copy with no name in this function.
				d := ccPath{suffix: "call " + pass.Fset.Position(n.Args[i].Pos()).String()}
				copies = append(copies, ccCopy{d, s, n.Pos()})
				events = append(events, ccEvent{d, true, fn.Name() + "(" + render(pass.Fset, n.Args[i]) + ")", n, slices.Clone(stack), at + (n.End() - n.Pos()), deferAt})
			}
		case *ast.Ident, *ast.SelectorExpr, *ast.IndexExpr:
			e := n.(ast.Expr)
			p, ok := pathOf(info, e)
			if ok && handleUse(info, e, stack) {
				events = append(events, ccEvent{p, false, render(pass.Fset, e), e, slices.Clone(stack), at, deferAt})
			}
		}
		stack = append(stack, n)
		return true
	})
	if len(copies) == 0 {
		return
	}

	// Group the places that hold the same handle.
	var groups [][]ccPath
	joined := make(map[ccPath]token.Pos)
	find := func(p ccPath) int {
		for i, g := range groups {
			if slices.Contains(g, p) {
				return i
			}
		}
		return -1
	}
	for _, c := range copies {
		for _, p := range []ccPath{c.dst, c.src} {
			if _, ok := joined[p]; !ok {
				joined[p] = c.at
			}
		}
		// A destination already in a group is assigned again, which ends
		// what the analysis knows of that group.
		i, j := find(c.dst), find(c.src)
		switch {
		case i < 0 && j < 0:
			groups = append(groups, []ccPath{c.dst, c.src})
		case i < 0:
			groups[j] = append(groups[j], c.dst)
		}
	}
	// killed reports whether a place of g is assigned or addressed after it
	// joined g and before at. The places copied from it since then may hold
	// another handle.
	killed := func(g []ccPath, at token.Pos) bool {
		for _, m := range g {
			for _, k := range kills {
				if overlaps(k.path, m) && joined[m] < k.at && k.at < at {
					return true
				}
			}
		}
		return false
	}
	// A ccHit is an event on a place of a group: the member the place is
	// in, and the rest of the place below it.
	type ccHit struct {
		ccEvent
		member ccPath
		rest   string
	}
	reported := make(map[ast.Node]bool)
	for _, g := range groups {
		var hits []ccHit
		for _, e := range events {
			for _, m := range g {
				if rest, ok := e.path.under(m); ok {
					hits = append(hits, ccHit{e, m, rest})
					break
				}
			}
		}
		for _, a := range hits {
			// A place reached through a pointer is shared by the copies.
			if !a.release || a.at <= joined[a.member] || strings.Contains(a.rest, "*") {
				continue
			}
			for _, b := range hits {
				if b.member == a.member || b.rest != a.rest || b.at <= a.at || reported[b.node] {
					continue
				}
				if b.at <= joined[b.member] || killed(g, b.at) || exclusive(pass.TypesInfo, a.ccEvent, b.ccEvent) {
					continue
				}
				reported[b.node] = true
				line := pass.Fset.Position(a.node.Pos()).Line
				if b.release {
					pass.Reportf(b.node.Pos(), "%s releases a handle that %s released on line %d; each release zeroes only its own copy", b.desc, a.desc, line)
				} else {
					pass.Reportf(b.node.Pos(), "%s is used after %s released it on line %d; each release zeroes only its own copy", b.desc, a.desc, line)
				}
			}
		}
	}
}

// deferOf returns the defer statement among the ancestors, or nil.
func deferOf(stack []ast.Node) *ast.DeferStmt {
	for i := len(stack) - 1; i >= 0; i-- {
		if d, ok := stack[i].(*ast.DeferStmt); ok {
			return d
		}
	}
	return nil
}

// handleUse reports whether the place e, with ancestors stack, is read: not
// assigned, discarded or ranged over, not the receiver of a release, and not
// only compared with a constant or printed.
func handleUse(info *types.Info, e ast.Expr, stack []ast.Node) bool {
	switch p := stack[len(stack)-1].(type) {
	case *ast.SelectorExpr:
		if _, _, ok := releaseCall(info, stack[len(stack)-2]); ok {
			return false
		}
	case *ast.AssignStmt:
		if slices.Contains(p.Lhs, e) || allBlank(p.Lhs) {
			return false
		}
	case *ast.RangeStmt:
		return false
	case *ast.BinaryExpr:
		if (p.Op == token.EQL || p.Op == token.NEQ) && (isConst(info, p.X) || isConst(info, p.Y)) {
			return false
		}
	}
	// A handle printed, or converted to print it, is not used.
	for i := len(stack) - 1; i >= 0; i-- {
		call, ok := stack[i].(*ast.CallExpr)
		if !ok {
			break
		}
		if tv, ok := info.Types[call.Fun]; ok && tv.IsType() {
			continue
		}
		fn := callee(info, call)
		return fn == nil || fn.Pkg() == nil || !formats(fn)
	}
	return true
}

// exclusive reports whether the release a and the later event b cannot both
// run: they lie on different branches of an if or a switch, or the path
// from a leaves the function or the loop before b.
func exclusive(info *types.Info, a, b ccEvent) bool {
	sa := append(slices.Clone(a.stack), a.node)
	sb := append(slices.Clone(b.stack), b.node)
	k := 0
	for k < len(sa) && k < len(sb) && sa[k] == sb[k] {
		k++
	}
	switch anc := sa[k-1].(type) {
	case *ast.IfStmt:
		inBranch := func(n ast.Node) bool { return n == anc.Body || n == anc.Else }
		if inBranch(sa[k]) && inBranch(sb[k]) {
			return true
		}
	case *ast.BlockStmt:
		if k >= 2 {
			switch sa[k-2].(type) {
			case *ast.SwitchStmt, *ast.TypeSwitchStmt, *ast.SelectStmt:
				return true
			}
		}
	}
	// A deferred b registered before a runs on every path out of a.
	if b.deferAt.IsValid() && b.deferAt < a.node.Pos() {
		return false
	}
	for j := len(sa) - 2; j >= k; j-- {
		var list []ast.Stmt
		switch s := sa[j].(type) {
		case *ast.BlockStmt:
			list = s.List
		case *ast.CaseClause:
			list = s.Body
		case *ast.CommClause:
			list = s.Body
		default:
			continue
		}
		for _, s := range list {
			if leaves(info, s) {
				return true
			}
		}
	}
	return false
}

// leaves reports whether s ends the enclosing block's path: a return, a
// branch, or a call that does not return.
func leaves(info *types.Info, s ast.Stmt) bool {
	switch s := s.(type) {
	case *ast.ReturnStmt, *ast.BranchStmt:
		return true
	case *ast.ExprStmt:
		call, ok := s.X.(*ast.CallExpr)
		return ok && noReturn(info, call)
	}
	return false
}
