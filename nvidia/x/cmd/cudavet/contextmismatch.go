package main

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"maps"
	"reflect"
	"slices"
	"strings"

	"golang.org/x/tools/go/analysis"
)

// ctxEffectFact marks a function that may return with a different context
// current than it was called with, or that enables peer access, directly or
// through a call to another function that does.
type ctxEffectFact struct {
	Switches    bool
	EnablesPeer bool
}

func (*ctxEffectFact) AFact() {}

func (f *ctxEffectFact) String() string {
	var s []string
	if f.Switches {
		s = append(s, "switches")
	}
	if f.EnablesPeer {
		s = append(s, "enablespeer")
	}
	return strings.Join(s, ",")
}

// ctxEffects holds the effects on the current context of the functions
// visible from a package.
type ctxEffects map[*types.Func]ctxEffectFact

// ctxEffectsAnalyzer computes ctxEffectFact for the functions of each
// package, so that a helper that makes a context current ends what
// contextmismatch knows about the current context at its call sites.
var ctxEffectsAnalyzer = &analysis.Analyzer{
	Name:       "ctxeffects",
	Doc:        "find the functions that change the current context or enable peer access",
	Run:        runCtxEffects,
	FactTypes:  []analysis.Fact{new(ctxEffectFact)},
	ResultType: reflect.TypeFor[ctxEffects](),
}

func runCtxEffects(pass *analysis.Pass) (any, error) {
	eff := make(ctxEffects)
	for _, f := range pass.AllObjectFacts() {
		if fn, ok := f.Object.(*types.Func); ok {
			eff[fn] = *f.Fact.(*ctxEffectFact)
		}
	}
	decls := make(map[*types.Func]*ast.FuncDecl)
	for _, file := range srcFiles(pass) {
		for _, decl := range file.Decls {
			if fd, ok := decl.(*ast.FuncDecl); ok && fd.Body != nil {
				if fn, ok := pass.TypesInfo.Defs[fd.Name].(*types.Func); ok {
					decls[fn] = fd
				}
			}
		}
	}
	// Effects only grow, so a chain of local helpers reaches a fixed point.
	for changed := true; changed; {
		changed = false
		for fn, fd := range decls {
			if e := eff.of(pass.TypesInfo, fd.Body); e != eff[fn] {
				eff[fn] = e
				changed = true
			}
		}
	}
	for fn := range decls {
		if e := eff[fn]; e.Switches || e.EnablesPeer {
			pass.ExportObjectFact(fn, &e)
		}
	}
	return eff, nil
}

// of returns the effects of body. A body switches if it makes a context
// current, pops or destroys one, or calls a function that switches, and has
// no deferred call that restores the context. Closures run by Context.Run or
// Device.WithContext do not count: those restore the context themselves.
func (eff ctxEffects) of(info *types.Info, body ast.Node) ctxEffectFact {
	var e ctxEffectFact
	switches, restores := false, false
	withStackAll(body, func(n ast.Node, stack []ast.Node) {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return
		}
		fn := callee(info, call)
		if fn == nil {
			return
		}
		if inNvidia(fn.Pkg()) && strings.HasSuffix(fn.Name(), "EnablePeerAccess") || eff[fn].EnablesPeer {
			e.EnablesPeer = true
		}
		if !changesCurrent(fn) && !eff[fn].Switches {
			return
		}
		inRun, inDefer := false, false
		for i, p := range stack {
			switch p := p.(type) {
			case *ast.DeferStmt:
				inDefer = true
			case *ast.FuncLit:
				if i > 0 && runsInContext(info, stack[i-1], p) {
					inRun = true
				}
			}
		}
		switch {
		case inRun:
		case inDefer:
			restores = true
		default:
			switches = true
		}
	})
	e.Switches = switches && !restores
	return e
}

// changesCurrent reports whether fn changes the calling thread's current
// context: it makes one current, pops one, or destroys one.
func changesCurrent(fn *types.Func) bool {
	return makesCurrent(fn) ||
		fn.Name() == "CtxPopCurrent" && inPkg(fn.Pkg(), "driver") && recvNamed(fn) == nil ||
		methodOf(fn, "driver", "Context", "Close")
}

// runsInContext reports whether lit is the closure that parent, a call of
// x/cuda's Context.Run or Device.WithContext, runs with a context current.
func runsInContext(info *types.Info, parent ast.Node, lit *ast.FuncLit) bool {
	call, ok := parent.(*ast.CallExpr)
	return ok && len(call.Args) == 1 && call.Args[0] == ast.Expr(lit) && isRunIn(callee(info, call))
}

// isRunIn reports whether fn is x/cuda's Context.Run or Device.WithContext.
func isRunIn(fn *types.Func) bool {
	return methodOf(fn, "x/cuda", "Context", "Run") || methodOf(fn, "x/cuda", "Device", "WithContext")
}

var contextMismatchAnalyzer = &analysis.Analyzer{
	Name: "contextmismatch",
	Doc: `check that device memory is used under the context it was allocated in

Memory from MemAlloc belongs to the context current when it was allocated.
Device work in another context, a kernel launch or a library call, cannot
reach it unless that context has peer access to the first: the kernel faults
on an illegal address. contextmismatch reports, within one function, a
DevicePtr allocated while one context is known to be current and passed to a
kernel launch or a library call while a different one is known to be current.

A context is known to be current after SetCurrent, PushCurrent or NewContext,
until CtxPopCurrent, a merge of paths that disagree, or a call to a function
that changes the current context; and inside the closure given to x/cuda's
Context.Run or Device.WithContext. Contexts are told apart by the variable
that holds them. Primary contexts are the same context for the same device,
so they are told apart by the device variable, or by its ordinal when that
is a constant, and a context from NewContext is never a primary one.

Copies, frees and pointer queries are not reported: under unified
addressing, which every 64-bit platform CUDA supports provides, the driver
finds the allocation from its address. Memsets and managed or stream-ordered
allocations are not reported either. A function that enables peer access,
directly or through a helper, is not checked; access enabled elsewhere is
invisible to the check.`,
	Run:      runContextMismatch,
	Requires: []*analysis.Analyzer{ctxEffectsAnalyzer},
}

func runContextMismatch(pass *analysis.Pass) (any, error) {
	eff := pass.ResultOf[ctxEffectsAnalyzer].(ctxEffects)
	info := pass.TypesInfo
	reported := make(map[token.Pos]bool)
	eachOuterFunc(pass, func(body *ast.BlockStmt) {
		if eff.of(info, body).EnablesPeer || hasGoto(body) {
			return
		}
		w := &ctxWalker{
			pass:     pass,
			info:     info,
			eff:      eff,
			res:      newCtxResolver(info, body),
			inline:   make(map[*ast.FuncLit]bool),
			reported: reported,
		}
		withStackAll(body, func(n ast.Node, stack []ast.Node) {
			if lit, ok := n.(*ast.FuncLit); ok && len(stack) > 0 && runsInContext(info, stack[len(stack)-1], lit) {
				w.inline[lit] = true
			}
		})
		w.untracked = w.untrackable(body)
		w.stmts(body.List, newCtxState())
		// Every other closure runs later, or on another goroutine, with no
		// context known to be current when it starts.
		ast.Inspect(body, func(n ast.Node) bool {
			if lit, ok := n.(*ast.FuncLit); ok && !w.inline[lit] {
				w.stmts(lit.Body.List, newCtxState())
			}
			return true
		})
	})
	return nil, nil
}

// hasGoto reports whether body holds a goto, whose flow the check does not
// follow.
func hasGoto(body *ast.BlockStmt) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		if b, ok := n.(*ast.BranchStmt); ok && b.Tok == token.GOTO {
			found = true
		}
		return !found
	})
	return found
}

type ctxKind int

const (
	ctxUnknown ctxKind = iota
	ctxNew             // made by NewContext and held in v
	ctxPrimary         // the primary context of device v, or of ordinal
	ctxOpaque          // held in v, a parameter or global of unknown origin
)

// A ctxKey names a context. Two keys that are equal name the same context;
// distinctCtx says when two keys certainly name different ones.
type ctxKey struct {
	kind       ctxKind
	v          types.Object
	ordinal    int64
	hasOrdinal bool
}

func (k ctxKey) known() bool { return k.kind != ctxUnknown }

func (k ctxKey) String() string {
	switch {
	case k.kind == ctxPrimary && k.hasOrdinal:
		return fmt.Sprintf("the primary context of device %d", k.ordinal)
	case k.kind == ctxPrimary:
		return "the primary context of " + k.v.Name()
	}
	return k.v.Name()
}

// distinctCtx reports whether a and b certainly name different contexts.
func distinctCtx(a, b ctxKey) bool {
	switch {
	case !a.known() || !b.known():
		return false
	case a.kind == ctxOpaque || b.kind == ctxOpaque:
		// A context of unknown origin may be any other of the same type.
		return a.kind == b.kind && a.v != b.v && types.Identical(a.v.Type(), b.v.Type())
	case a.kind != b.kind:
		return true // NewContext never returns a primary context
	case a.hasOrdinal && b.hasOrdinal:
		return a.ordinal != b.ordinal
	case a.hasOrdinal != b.hasOrdinal:
		return false
	}
	return a.v != b.v
}

// ctxResolver names the context a variable holds, from every value the
// function assigns to it.
type ctxResolver struct {
	info    *types.Info
	body    *ast.BlockStmt
	sources map[*types.Var][]ast.Expr // nil for a value of unknown origin
}

func newCtxResolver(info *types.Info, body *ast.BlockStmt) *ctxResolver {
	r := &ctxResolver{info: info, body: body, sources: make(map[*types.Var][]ast.Expr)}
	add := func(l ast.Expr, v ast.Expr) {
		if x := objOf(info, l); x != nil {
			r.sources[x] = append(r.sources[x], v)
		}
	}
	ast.Inspect(body, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.AssignStmt:
			for i, l := range n.Lhs {
				var v ast.Expr
				switch {
				case n.Tok != token.DEFINE && n.Tok != token.ASSIGN:
				case len(n.Lhs) == len(n.Rhs):
					v = n.Rhs[i]
				case i == 0 && len(n.Rhs) == 1:
					v = n.Rhs[0] // the first result of a call
				}
				add(l, v)
			}
		case *ast.ValueSpec:
			for i, name := range n.Names {
				switch {
				case len(n.Values) == len(n.Names):
					add(name, n.Values[i])
				case i == 0 && len(n.Values) == 1:
					add(name, n.Values[0])
				case len(n.Values) > 0:
					add(name, nil)
				}
			}
		case *ast.RangeStmt:
			add(n.Key, nil)
			if n.Value != nil {
				add(n.Value, nil)
			}
		case *ast.UnaryExpr:
			if n.Op == token.AND {
				add(n.X, nil)
			}
		}
		return true
	})
	return r
}

// context returns the key of the context e holds: a driver.Context or an
// x/cuda *Context.
func (r *ctxResolver) context(e ast.Expr) ctxKey {
	v := objOf(r.info, e)
	if v == nil {
		return ctxKey{}
	}
	return r.contextVar(v, 0)
}

// contextVar returns the key of the context variable v holds.
func (r *ctxResolver) contextVar(v *types.Var, depth int) ctxKey {
	if depth > 4 {
		return ctxKey{}
	}
	if !declaredIn(v, r.body) {
		return ctxKey{kind: ctxOpaque, v: v}
	}
	srcs := r.sources[v]
	if len(srcs) == 0 {
		return ctxKey{}
	}
	var k ctxKey
	for i, src := range srcs {
		sk := r.contextSource(src, v, depth)
		if !sk.known() || i > 0 && sk != k {
			return ctxKey{}
		}
		k = sk
	}
	return k
}

// contextSource returns the key of the context src, assigned to v, yields.
func (r *ctxResolver) contextSource(src ast.Expr, v *types.Var, depth int) ctxKey {
	switch src := ast.Unparen(src).(type) {
	case *ast.CallExpr:
		fn := callee(r.info, src)
		if fn == nil {
			return ctxKey{}
		}
		if (fn.Name() == "NewContext" || fn.Name() == "NewContextParams") && inPkg(fn.Pkg(), "driver") && recvNamed(fn) == nil {
			return ctxKey{kind: ctxNew, v: v}
		}
		if methodOf(fn, "driver", "Device", "PrimaryCtxRetain") || methodOf(fn, "x/cuda", "Device", "PrimaryContext") {
			if sel, ok := ast.Unparen(src.Fun).(*ast.SelectorExpr); ok {
				return r.device(sel.X)
			}
		}
	case *ast.Ident:
		if w, ok := r.info.Uses[src].(*types.Var); ok {
			return r.contextVar(w, depth+1)
		}
	}
	return ctxKey{}
}

// device returns the key of the primary context of the device e holds: a
// driver.Device or an x/cuda *Device. A local device variable is known only
// by a constant ordinal.
func (r *ctxResolver) device(e ast.Expr) ctxKey {
	if n, ok := intConst(r.info, unconvert(r.info, e)); ok {
		return ctxKey{kind: ctxPrimary, ordinal: n, hasOrdinal: true}
	}
	v := objOf(r.info, e)
	if v == nil {
		return ctxKey{}
	}
	if !declaredIn(v, r.body) {
		return ctxKey{kind: ctxPrimary, v: v}
	}
	srcs := r.sources[v]
	var ords []int64
	for _, src := range srcs {
		if n, ok := r.ordinal(src); ok {
			ords = append(ords, n)
		}
	}
	if len(ords) == 0 || len(ords) < len(srcs) || slices.ContainsFunc(ords, func(n int64) bool { return n != ords[0] }) {
		return ctxKey{}
	}
	return ctxKey{kind: ctxPrimary, ordinal: ords[0], hasOrdinal: true}
}

// ordinal returns the constant ordinal of a device obtained by
// driver.DeviceGet, cuda.NewDevice or a conversion to driver.Device.
func (r *ctxResolver) ordinal(src ast.Expr) (int64, bool) {
	if src == nil {
		return 0, false
	}
	if n, ok := intConst(r.info, unconvert(r.info, src)); ok {
		return n, true
	}
	call, ok := ast.Unparen(src).(*ast.CallExpr)
	if !ok || len(call.Args) != 1 {
		return 0, false
	}
	fn := callee(r.info, call)
	if fn == nil || recvNamed(fn) != nil ||
		!(fn.Name() == "DeviceGet" && inPkg(fn.Pkg(), "driver") || fn.Name() == "NewDevice" && inPkg(fn.Pkg(), "x/cuda")) {
		return 0, false
	}
	return intConst(r.info, call.Args[0])
}

// ctxState is what the check knows at a point in a function: the current
// context, the contexts PushCurrent saved below it, and the context each
// device pointer was allocated in.
type ctxState struct {
	dead   bool // no path reaches here
	cur    ctxKey
	stack  []ctxKey // what PushCurrent saved; below it is unknown
	allocs map[*types.Var]ctxKey
}

func newCtxState() *ctxState {
	return &ctxState{allocs: make(map[*types.Var]ctxKey)}
}

func (s *ctxState) clone() *ctxState {
	t := *s
	t.stack = slices.Clone(s.stack)
	t.allocs = maps.Clone(s.allocs)
	return &t
}

func (s *ctxState) equal(t *ctxState) bool {
	return s.dead == t.dead && s.cur == t.cur &&
		slices.Equal(s.stack, t.stack) && maps.Equal(s.allocs, t.allocs)
}

// lose forgets the current context and the stack below it.
func (s *ctxState) lose() {
	s.cur, s.stack = ctxKey{}, nil
}

func (s *ctxState) push(k ctxKey) {
	s.stack = append(s.stack, s.cur)
	s.cur = k
}

func (s *ctxState) pop() {
	if len(s.stack) == 0 {
		s.cur = ctxKey{}
		return
	}
	s.cur = s.stack[len(s.stack)-1]
	s.stack = s.stack[:len(s.stack)-1]
}

// joinCtx returns what is known where the paths reaching a and b merge.
func joinCtx(a, b *ctxState) *ctxState {
	switch {
	case a.dead:
		return b.clone()
	case b.dead:
		return a.clone()
	}
	j := a.clone()
	if a.cur != b.cur {
		j.cur = ctxKey{}
	}
	if !slices.Equal(a.stack, b.stack) {
		j.stack = nil
	}
	for v, k := range a.allocs {
		if bk, ok := b.allocs[v]; !ok || bk != k {
			delete(j.allocs, v)
		}
	}
	return j
}

// ctxWalker follows one function in source order.
type ctxWalker struct {
	pass      *analysis.Pass
	info      *types.Info
	eff       ctxEffects
	res       *ctxResolver
	inline    map[*ast.FuncLit]bool // closures run in place by Run or WithContext
	untracked map[*types.Var]bool
	reported  map[token.Pos]bool
	exits     []*ctxState // states at the returns of the closure being walked
}

// untrackable returns the device pointer variables of body whose value may
// change where the walk cannot see: through their address, or in a closure
// that runs later than their declaration.
func (w *ctxWalker) untrackable(body *ast.BlockStmt) map[*types.Var]bool {
	out := make(map[*types.Var]bool)
	withStackAll(body, func(n ast.Node, stack []ast.Node) {
		var targets []ast.Expr
		later := true
		switch n := n.(type) {
		case *ast.UnaryExpr:
			if n.Op != token.AND {
				return
			}
			if p, ok := stack[len(stack)-1].(*ast.CallExpr); ok && isUnsafe(w.info, p.Fun, "Pointer") {
				return // a kernel parameter: read, not written
			}
			targets, later = []ast.Expr{n.X}, false
		case *ast.AssignStmt:
			targets = n.Lhs
		case *ast.RangeStmt:
			targets = []ast.Expr{n.Key, n.Value}
		}
		for _, t := range targets {
			v := objOf(w.info, t)
			if v == nil || !isDevicePtr(v.Type()) {
				continue
			}
			if !later || runsLater(w.inline, stack, v) {
				out[v] = true
			}
		}
	})
	return out
}

// runsLater reports whether a node with ancestors stack lies in a closure,
// other than one run in place, that does not declare v.
func runsLater(inline map[*ast.FuncLit]bool, stack []ast.Node, v *types.Var) bool {
	for _, n := range stack {
		if lit, ok := n.(*ast.FuncLit); ok && !inline[lit] && !(lit.Pos() <= v.Pos() && v.Pos() < lit.End()) {
			return true
		}
	}
	return false
}

func (w *ctxWalker) stmts(list []ast.Stmt, s *ctxState) {
	for _, st := range list {
		w.stmt(st, s)
	}
}

func (w *ctxWalker) stmt(st ast.Stmt, s *ctxState) {
	if st == nil || s.dead {
		return
	}
	switch st := st.(type) {
	case *ast.ExprStmt:
		w.expr(st.X, s)
	case *ast.AssignStmt:
		w.assign(st.Lhs, st.Rhs, st.Tok, s)
	case *ast.DeclStmt:
		if gd, ok := st.Decl.(*ast.GenDecl); ok && gd.Tok == token.VAR {
			for _, spec := range gd.Specs {
				vs := spec.(*ast.ValueSpec)
				lhs := make([]ast.Expr, len(vs.Names))
				for i, name := range vs.Names {
					lhs[i] = name
				}
				w.assign(lhs, vs.Values, token.DEFINE, s)
			}
		}
	case *ast.IncDecStmt:
		w.expr(st.X, s)
	case *ast.SendStmt:
		w.expr(st.Chan, s)
		w.expr(st.Value, s)
	case *ast.GoStmt, *ast.DeferStmt:
		// Runs later, or on another thread.
	case *ast.ReturnStmt:
		for _, r := range st.Results {
			w.expr(r, s)
		}
		w.exits = append(w.exits, s.clone())
		s.dead = true
	case *ast.BranchStmt:
		// The enclosing loop or switch forgets what its body changed.
		s.dead = true
	case *ast.BlockStmt:
		w.stmts(st.List, s)
	case *ast.LabeledStmt:
		w.stmt(st.Stmt, s)
	case *ast.IfStmt:
		w.stmt(st.Init, s)
		w.expr(st.Cond, s)
		then, els := s.clone(), s.clone()
		w.stmts(st.Body.List, then)
		w.stmt(st.Else, els)
		*s = *joinCtx(then, els)
	case *ast.ForStmt:
		w.stmt(st.Init, s)
		w.loop(st, st.Cond, st.Body, st.Post, s)
	case *ast.RangeStmt:
		w.expr(st.X, s)
		w.loop(st, nil, st.Body, nil, s)
	case *ast.SwitchStmt:
		w.stmt(st.Init, s)
		w.expr(st.Tag, s)
		w.clauses(st, st.Body, s)
	case *ast.TypeSwitchStmt:
		w.stmt(st.Init, s)
		w.stmt(st.Assign, s)
		w.clauses(st, st.Body, s)
	case *ast.SelectStmt:
		w.clauses(st, st.Body, s)
	}
}

// clauses walks the clauses of a switch or select from s and merges them.
func (w *ctxWalker) clauses(st ast.Stmt, body *ast.BlockStmt, s *ctxState) {
	out := &ctxState{dead: true}
	exhaustive := false
	for _, c := range body.List {
		cs := s.clone()
		switch c := c.(type) {
		case *ast.CaseClause:
			for _, e := range c.List {
				w.expr(e, cs)
			}
			w.stmts(c.Body, cs)
			exhaustive = exhaustive || c.List == nil
		case *ast.CommClause:
			w.stmt(c.Comm, cs)
			w.stmts(c.Body, cs)
			exhaustive = true
		}
		out = joinCtx(out, cs)
	}
	if !exhaustive {
		out = joinCtx(out, s)
	}
	if hasBranch(body) {
		out = s.clone()
		w.forget(out, st)
	}
	*s = *out
}

// loop walks a for or range statement. Its body is walked from what holds
// on every iteration, found by merging until nothing changes, and the loop
// ends in that state too. A body that breaks or continues leaves both the
// current context and the pointers it assigns unknown.
func (w *ctxWalker) loop(st ast.Stmt, cond ast.Expr, body *ast.BlockStmt, post ast.Stmt, s *ctxState) {
	iterate := func(entry *ctxState) *ctxState {
		c := entry.clone()
		w.expr(cond, c)
		w.stmts(body.List, c)
		w.stmt(post, c)
		return c
	}
	entry := s.clone()
	if hasBranch(body) {
		w.forget(entry, st)
	}
	// A merge only forgets, so this ends.
	for {
		next := joinCtx(entry, iterate(entry))
		if next.equal(entry) {
			break
		}
		entry = next
	}
	*s = *entry
}

// forget clears the current context and the allocations of the pointers
// assigned within n.
func (w *ctxWalker) forget(s *ctxState, n ast.Node) {
	s.lose()
	ast.Inspect(n, func(n ast.Node) bool {
		var lhs []ast.Expr
		switch n := n.(type) {
		case *ast.AssignStmt:
			lhs = n.Lhs
		case *ast.ValueSpec:
			for _, name := range n.Names {
				lhs = append(lhs, name)
			}
		case *ast.RangeStmt:
			lhs = []ast.Expr{n.Key, n.Value}
		}
		for _, l := range lhs {
			if v := objOf(w.info, l); v != nil {
				delete(s.allocs, v)
			}
		}
		return true
	})
}

// hasBranch reports whether body holds a break, continue or fallthrough
// outside a closure.
func hasBranch(body ast.Node) bool {
	found := false
	inspectShallow(body, func(n ast.Node) bool {
		if b, ok := n.(*ast.BranchStmt); ok && b.Tok != token.GOTO {
			found = true
		}
		return !found
	})
	return found
}

func (w *ctxWalker) assign(lhs, rhs []ast.Expr, tok token.Token, s *ctxState) {
	var first *types.Var
	if len(lhs) > 0 {
		first = objOf(w.info, lhs[0])
	}
	if len(rhs) == 1 {
		if call, ok := ast.Unparen(rhs[0]).(*ast.CallExpr); ok {
			w.call(call, s, first)
		} else {
			w.expr(rhs[0], s)
		}
	} else {
		for _, r := range rhs {
			w.expr(r, s)
		}
	}
	for i, l := range lhs {
		if _, ok := l.(*ast.Ident); !ok {
			w.expr(l, s)
		}
		v := objOf(w.info, l)
		if v == nil || !isDevicePtr(v.Type()) {
			continue
		}
		if tok != token.DEFINE && tok != token.ASSIGN {
			continue // arithmetic: the same allocation
		}
		var r ast.Expr
		switch {
		case len(lhs) == len(rhs):
			r = rhs[i]
		case i == 0 && len(rhs) == 1:
			r = rhs[0]
		}
		w.setAlloc(v, r, s)
	}
}

// setAlloc records the context the value r assigned to v was allocated in:
// the current one for an allocation, and that of the one pointer it is
// derived from for arithmetic.
func (w *ctxWalker) setAlloc(v *types.Var, r ast.Expr, s *ctxState) {
	var k ctxKey
	switch {
	case w.untracked[v] || r == nil:
	case w.isAlloc(r):
		k = s.cur
	default:
		k = w.derived(r, s)
	}
	delete(s.allocs, v)
	if k.known() {
		s.allocs[v] = k
	}
}

// isAlloc reports whether e allocates device memory in the current context.
func (w *ctxWalker) isAlloc(e ast.Expr) bool {
	call, ok := ast.Unparen(e).(*ast.CallExpr)
	if !ok {
		return false
	}
	fn := callee(w.info, call)
	if fn == nil || !inPkg(fn.Pkg(), "driver") {
		return false
	}
	switch fn.Name() {
	case "MemAlloc", "MemAllocBytes", "MemAllocPitch":
		return true
	}
	return false
}

// derived returns the context of the one tracked pointer e is computed
// from by conversion and arithmetic, if e calls nothing else.
func (w *ctxWalker) derived(e ast.Expr, s *ctxState) ctxKey {
	var from []*types.Var
	pure := true
	ast.Inspect(e, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.CallExpr:
			if tv, ok := w.info.Types[n.Fun]; !ok || !tv.IsType() {
				pure = false
			}
		case *ast.Ident:
			if v, ok := w.info.Uses[n].(*types.Var); ok && isDevicePtr(v.Type()) {
				from = append(from, v)
			}
		}
		return pure
	})
	if !pure || len(from) != 1 {
		return ctxKey{}
	}
	return s.allocs[from[0]]
}

// expr walks the calls in e in the order they are made.
func (w *ctxWalker) expr(e ast.Expr, s *ctxState) {
	if e == nil {
		return
	}
	ast.Inspect(e, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.FuncLit:
			return false
		case *ast.CallExpr:
			w.call(n, s, nil)
			return false
		}
		return true
	})
}

// call walks call, whose first result is assigned to lhs if it is not nil.
func (w *ctxWalker) call(call *ast.CallExpr, s *ctxState, lhs *types.Var) {
	if sel, ok := ast.Unparen(call.Fun).(*ast.SelectorExpr); ok {
		w.expr(sel.X, s)
	}
	for _, a := range call.Args {
		w.expr(a, s)
	}
	fn := callee(w.info, call)
	if fn == nil {
		w.dynamicCall(call, s)
		return
	}
	var recv ast.Expr
	if sel, ok := ast.Unparen(call.Fun).(*ast.SelectorExpr); ok && recvNamed(fn) != nil {
		recv = sel.X
	}
	switch {
	case isRunIn(fn):
		w.runIn(call, fn, recv, s)
		return
	case fn.Name() == "SetCurrent" && makesCurrent(fn):
		s.cur = w.res.context(recv)
	case fn.Name() == "PushCurrent" && makesCurrent(fn):
		s.push(w.res.context(recv))
	case makesCurrent(fn): // NewContext pushes the context it makes
		k := ctxKey{}
		if lhs != nil {
			k = w.res.contextVar(lhs, 0)
		}
		s.push(k)
	case fn.Name() == "CtxPopCurrent" && changesCurrent(fn):
		s.pop()
	case changesCurrent(fn), w.eff[fn].Switches:
		s.lose()
	}
	w.use(call, fn, recv, s)
}

// dynamicCall handles a call of a function value: a local closure known by
// its body, or anything else, which may change the current context.
func (w *ctxWalker) dynamicCall(call *ast.CallExpr, s *ctxState) {
	if tv, ok := w.info.Types[call.Fun]; ok && (tv.IsType() || tv.IsBuiltin()) {
		return
	}
	v := objOf(w.info, call.Fun)
	if v == nil || !declaredIn(v, w.res.body) {
		s.lose()
		return
	}
	for _, src := range w.res.sources[v] {
		lit, ok := ast.Unparen(src).(*ast.FuncLit)
		if !ok || w.eff.of(w.info, lit.Body).Switches {
			s.lose()
			return
		}
	}
}

// runIn walks the closure given to Context.Run or Device.WithContext with
// its context current. Run restores the context current before it.
func (w *ctxWalker) runIn(call *ast.CallExpr, fn *types.Func, recv ast.Expr, s *ctxState) {
	lit, ok := call.Args[0].(*ast.FuncLit)
	if !ok || recv == nil {
		return
	}
	in := s.clone()
	if fn.Name() == "Run" {
		in.cur = w.res.context(recv)
	} else {
		in.cur = w.res.device(recv)
	}
	exits := w.exits
	w.exits = nil
	w.stmts(lit.Body.List, in)
	out := in
	for _, e := range w.exits {
		out = joinCtx(out, e)
	}
	w.exits = exits
	if out.dead {
		return
	}
	s.allocs = out.allocs
	if !slices.Equal(out.stack, s.stack) {
		s.stack = nil
	}
}

// use reports the tracked pointers call hands to device work while a
// context other than theirs is current.
func (w *ctxWalker) use(call *ast.CallExpr, fn *types.Func, recv ast.Expr, s *ctxState) {
	if !isBinding(fn.Pkg()) {
		return
	}
	if inPkg(fn.Pkg(), "driver") && !isLaunch(fn) {
		return // copies, frees and queries find the allocation by address
	}
	seen := make(map[*types.Var]bool) // report each pointer once per call
	check := func(e ast.Expr, at token.Pos) {
		ast.Inspect(e, func(n ast.Node) bool {
			id, ok := n.(*ast.Ident)
			if !ok {
				return true
			}
			v, _ := w.info.Uses[id].(*types.Var)
			if v == nil {
				return true
			}
			pos := at
			if pos == token.NoPos {
				pos = id.Pos()
			}
			if k, ok := s.allocs[v]; ok && distinctCtx(k, s.cur) && !seen[v] && !w.reported[pos] {
				seen[v], w.reported[pos] = true, true
				w.pass.Reportf(pos, "%s was allocated while %s was current, but %s uses it while %s is current; device memory belongs to the context it was allocated in",
					v.Name(), k, fn.Name(), s.cur)
			}
			return true
		})
	}
	args := call.Args
	if recv != nil {
		args = append([]ast.Expr{recv}, args...)
	}
	for _, a := range args {
		check(a, token.NoPos)
		// A kernel parameter slice built from &p reads p at the launch.
		v := objOf(w.info, a)
		if v == nil || !declaredIn(v, w.res.body) {
			continue
		}
		for _, src := range w.res.sources[v] {
			if cl, ok := ast.Unparen(src).(*ast.CompositeLit); ok {
				check(cl, a.Pos())
			}
		}
	}
}
