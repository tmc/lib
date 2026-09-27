package main

import (
	"go/ast"
	"go/token"
	"go/types"
	"reflect"
	"strings"

	"golang.org/x/tools/go/analysis"
)

// syncsFact marks a function that synchronizes with the device, directly or
// through a call to another function that does, so that a helper such as
// wait(s) counts as a synchronization at its call sites in other packages.
type syncsFact struct{}

func (*syncsFact) AFact()         {}
func (*syncsFact) String() string { return "syncs" }

// syncsAnalyzer computes syncsFact for the functions of each package and
// hands the checks that follow asynchronous work the set of synchronizing
// functions visible from the package.
var syncsAnalyzer = &analysis.Analyzer{
	Name:       "syncs",
	Doc:        "find the functions that synchronize with the device",
	Run:        runSyncs,
	FactTypes:  []analysis.Fact{new(syncsFact)},
	ResultType: reflect.TypeFor[syncSet](),
}

// syncSet holds the functions known to synchronize, beyond the driver's own.
type syncSet map[*types.Func]bool

// isSync reports whether call waits for the device: a Synchronize method or
// CtxSynchronize of an nvidia package, or a function known to call one.
func (s syncSet) isSync(info *types.Info, call *ast.CallExpr) bool {
	fn := callee(info, call)
	if fn == nil {
		return false
	}
	if inNvidia(fn.Pkg()) && (strings.HasPrefix(fn.Name(), "Synchronize") || fn.Name() == "CtxSynchronize") {
		return true
	}
	return s[fn]
}

func runSyncs(pass *analysis.Pass) (any, error) {
	set := make(syncSet)
	for _, f := range pass.AllObjectFacts() {
		if fn, ok := f.Object.(*types.Func); ok {
			set[fn] = true
		}
	}
	// Iterate to a fixed point so that a chain of local helpers resolves.
	for changed := true; changed; {
		changed = false
		for _, file := range pass.Files {
			for _, decl := range file.Decls {
				fd, ok := decl.(*ast.FuncDecl)
				if !ok || fd.Body == nil {
					continue
				}
				fn, _ := pass.TypesInfo.Defs[fd.Name].(*types.Func)
				if fn == nil || set[fn] {
					continue
				}
				for _, c := range calls(fd.Body) {
					if set.isSync(pass.TypesInfo, c) {
						set[fn] = true
						pass.ExportObjectFact(fn, new(syncsFact))
						changed = true
						break
					}
				}
			}
		}
	}
	return set, nil
}

// asyncCopy describes a call that copies between host and device memory
// after it returns.
type asyncCopy struct {
	call   *ast.CallExpr
	name   string
	host   *types.Var // the host buffer
	stream ast.Expr
	toHost bool
}

// asyncCopies returns the asynchronous host copies in body whose host buffer
// is a variable, in source order.
func asyncCopies(info *types.Info, body ast.Node) []asyncCopy {
	var out []asyncCopy
	for _, call := range calls(body) {
		fn := callee(info, call)
		if fn == nil || !inNvidia(fn.Pkg()) || !asyncHostCopy[fn.Name()] || len(call.Args) < 2 {
			continue
		}
		c := asyncCopy{call: call, name: fn.Name(), stream: call.Args[len(call.Args)-1], toHost: strings.HasPrefix(fn.Name(), "CopyToHost")}
		for _, arg := range call.Args[:len(call.Args)-1] {
			if v := hostBuffer(info, arg); v != nil {
				c.host = v
				break
			}
		}
		if c.host != nil {
			out = append(out, c)
		}
	}
	return out
}

// hostBuffer returns the variable a host-buffer argument names: s itself, or
// s in unsafe.Pointer(&s[i]).
func hostBuffer(info *types.Info, e ast.Expr) *types.Var {
	e = unconvert(info, e)
	if u, ok := e.(*ast.UnaryExpr); ok && u.Op == token.AND {
		if ix, ok := ast.Unparen(u.X).(*ast.IndexExpr); ok {
			e = ix.X
		} else {
			e = u.X
		}
	}
	v := objOf(info, e)
	if v == nil {
		return nil
	}
	switch v.Type().Underlying().(type) {
	case *types.Slice, *types.Array:
		return v
	}
	return nil
}

// firstSync returns the position of the first synchronizing call in body
// after pos, or token.NoPos.
func firstSync(pass *analysis.Pass, set syncSet, body ast.Node, pos token.Pos) token.Pos {
	for _, c := range calls(body) {
		if c.Pos() > pos && set.isSync(pass.TypesInfo, c) {
			return c.Pos()
		}
	}
	return token.NoPos
}

var asyncHostAnalyzer = &analysis.Analyzer{
	Name: "asynchost",
	Doc: `check that host buffers outlive the asynchronous copies using them

An asynchronous copy reads or writes its host buffer after the call returns,
until the stream reaches it. asynchost reports a host buffer that is written
or resliced after being handed to an asynchronous copy and before a
synchronizing call, and a buffer local to the function that is left to die
with no synchronizing call after the copy. unpinned covers pinning; this
covers the use before synchronization.`,
	Run:      runAsyncHost,
	Requires: []*analysis.Analyzer{syncsAnalyzer},
}

func runAsyncHost(pass *analysis.Pass) (any, error) {
	set := pass.ResultOf[syncsAnalyzer].(syncSet)
	eachFunc(pass, func(body *ast.BlockStmt) {
		for _, c := range asyncCopies(pass.TypesInfo, body) {
			sync := firstSync(pass, set, body, c.call.End())
			if w := firstWrite(pass.TypesInfo, body, c.host, c.call.End(), sync); w != nil {
				pass.Reportf(w.Pos(), "%s is written while %s may still be using it; synchronize the stream first", c.host.Name(), c.name)
				continue
			}
			if sync == token.NoPos && declaredIn(c.host, body) && !escapes(pass.TypesInfo, body, c.host) {
				pass.Reportf(c.call.Pos(), "%s dies when the function returns but %s may still be using it; synchronize the stream before returning", c.host.Name(), c.name)
			}
		}
	})
	return nil, nil
}

// firstWrite returns the first write to v in body between from and to (to
// may be token.NoPos, meaning the end): an assignment to v or to an element
// of it, an increment of an element, or a copy into it.
func firstWrite(info *types.Info, body ast.Node, v *types.Var, from, to token.Pos) ast.Node {
	var found ast.Node
	inspectShallow(body, func(n ast.Node) bool {
		if found != nil || n == nil || n.Pos() <= from || (to != token.NoPos && n.Pos() >= to) {
			return found == nil
		}
		switch n := n.(type) {
		case *ast.AssignStmt:
			for _, lhs := range n.Lhs {
				if writesTo(info, lhs, v) {
					found = n
				}
			}
		case *ast.IncDecStmt:
			if writesTo(info, n.X, v) {
				found = n
			}
		case *ast.CallExpr:
			if id, ok := n.Fun.(*ast.Ident); ok && id.Name == "copy" && len(n.Args) == 2 && writesTo(info, n.Args[0], v) {
				found = n
			}
		}
		return found == nil
	})
	return found
}

// writesTo reports whether assigning to lhs writes v or an element of it.
func writesTo(info *types.Info, lhs ast.Expr, v *types.Var) bool {
	for {
		lhs = ast.Unparen(lhs)
		switch e := lhs.(type) {
		case *ast.IndexExpr:
			lhs = e.X
		case *ast.SliceExpr:
			lhs = e.X
		default:
			return objOf(info, lhs) == v
		}
	}
}

var readBeforeSyncAnalyzer = &analysis.Analyzer{
	Name: "readbeforesync",
	Doc: `check that host results of asynchronous copies are read after a synchronization

A copy to the host on a stream other than the default one returns before the
data arrives. readbeforesync reports a read of the destination buffer with no
Stream.Synchronize, Event.Synchronize, CtxSynchronize, or call to a function
that synchronizes, in between. A copy on the default stream is not reported.`,
	Run:      runReadBeforeSync,
	Requires: []*analysis.Analyzer{syncsAnalyzer},
}

func runReadBeforeSync(pass *analysis.Pass) (any, error) {
	set := pass.ResultOf[syncsAnalyzer].(syncSet)
	eachFunc(pass, func(body *ast.BlockStmt) {
		for _, c := range asyncCopies(pass.TypesInfo, body) {
			if !c.toHost || isDefaultStream(pass.TypesInfo, c.stream) {
				continue
			}
			sync := firstSync(pass, set, body, c.call.End())
			if r := firstRead(pass.TypesInfo, body, c.host, c.call.End(), sync); r != nil {
				pass.Reportf(r.Pos(), "%s is read before the stream is synchronized; %s may not have finished", c.host.Name(), c.name)
			}
		}
	})
	return nil, nil
}

// isDefaultStream reports whether a stream argument is the constant zero.
func isDefaultStream(info *types.Info, e ast.Expr) bool {
	v, ok := intConst(info, e)
	return ok && v == 0
}

// firstRead returns the first use of v's contents in body between from and
// to: any mention other than an assignment target, a len or cap, or an
// address handed to another call.
func firstRead(info *types.Info, body ast.Node, v *types.Var, from, to token.Pos) ast.Node {
	var found ast.Node
	withStack(body, func(n ast.Node, stack []ast.Node) {
		if found != nil || n.Pos() <= from || (to != token.NoPos && n.Pos() >= to) {
			return
		}
		id, ok := n.(*ast.Ident)
		if !ok || info.ObjectOf(id) != v || len(stack) == 0 {
			return
		}
		// Climb over index and slice expressions to the use of the value.
		i := len(stack) - 1
		var child ast.Node = id
	climb:
		for ; i >= 0; i-- {
			switch stack[i].(type) {
			case *ast.IndexExpr, *ast.SliceExpr, *ast.ParenExpr:
			default:
				break climb
			}
			child = stack[i]
		}
		if i < 0 {
			found = id
			return
		}
		switch p := stack[i].(type) {
		case *ast.AssignStmt:
			for _, l := range p.Lhs {
				if l == child {
					return
				}
			}
		case *ast.UnaryExpr:
			if p.Op == token.AND {
				return
			}
		case *ast.CallExpr:
			if f, ok := p.Fun.(*ast.Ident); ok && (f.Name == "len" || f.Name == "cap") {
				return
			}
		}
		found = id
	})
	return found
}

// isLaunch reports whether fn launches device work whose failure is reported
// later: a Launch method of an nvidia package.
func isLaunch(fn *types.Func) bool {
	return fn != nil && inNvidia(fn.Pkg()) && recvNamed(fn) != nil && strings.HasPrefix(fn.Name(), "Launch") && fn.Name() != "LaunchHostFunc"
}

// syncCopy names the copies to the host that wait for the default stream,
// and so report an earlier launch's failure as their own.
var syncCopy = map[string]bool{"CopyToHost": true, "CopyToHostPtr": true}

var launchErrAnalyzer = &analysis.Analyzer{
	Name: "launcherr",
	Doc: `check that a launch's failure is looked for where it is reported

A kernel launch returns only whether it was enqueued; a fault in the kernel is
reported by a later call, usually the next one that synchronizes. launcherr
reports a launch whose first synchronizing call afterwards discards its error,
so that a nil error from the launch is all the program checks before reading
results.`,
	Run:      runLaunchErr,
	Requires: []*analysis.Analyzer{syncsAnalyzer},
}

func runLaunchErr(pass *analysis.Pass) (any, error) {
	set := pass.ResultOf[syncsAnalyzer].(syncSet)
	eachFunc(pass, func(body *ast.BlockStmt) {
		type syncCall struct {
			call      *ast.CallExpr
			name      string
			discarded bool
		}
		var syncs []syncCall
		var launches []*ast.CallExpr
		withStack(body, func(n ast.Node, stack []ast.Node) {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return
			}
			fn := callee(pass.TypesInfo, call)
			switch {
			case isLaunch(fn):
				launches = append(launches, call)
			case set.isSync(pass.TypesInfo, call) || (fn != nil && inNvidia(fn.Pkg()) && syncCopy[fn.Name()]):
				if len(stack) > 0 {
					if _, ok := stack[len(stack)-1].(*ast.DeferStmt); ok {
						return
					}
				}
				// A helper that returns no error has handled it itself.
				discarded := resultDiscarded(stack, call) && fn.Type().(*types.Signature).Results().Len() > 0
				syncs = append(syncs, syncCall{call, fn.Name(), discarded})
			}
		})
		for _, l := range launches {
			for _, s := range syncs {
				if s.call.Pos() <= l.End() {
					continue
				}
				if s.discarded {
					pass.Reportf(l.Pos(), "%s reports only that the launch was enqueued, and the error of %s, the first call to wait for it, is discarded; check that error before reading results",
						callee(pass.TypesInfo, l).Name(), s.name)
				}
				break
			}
		}
	})
	return nil, nil
}

var streamMismatchAnalyzer = &analysis.Analyzer{
	Name: "streammismatch",
	Doc: `check that a library handle's stream waits for its inputs

A library handle bound to stream S with SetStream runs its work on S. A device
buffer last written by an asynchronous copy on another stream T may not be
ready when that work starts. streammismatch reports such a buffer passed to a
method of the handle with no S.WaitEvent, no synchronization of T and no
context synchronization in between.`,
	Run:      runStreamMismatch,
	Requires: []*analysis.Analyzer{syncsAnalyzer},
}

func runStreamMismatch(pass *analysis.Pass) (any, error) {
	set := pass.ResultOf[syncsAnalyzer].(syncSet)
	info := pass.TypesInfo
	eachFunc(pass, func(body *ast.BlockStmt) {
		bound := make(map[*types.Var]*types.Var) // handle -> stream
		type write struct {
			pos    token.Pos
			stream *types.Var
		}
		written := make(map[*types.Var]write) // device buffer -> last write
		var waits []*ast.CallExpr
		for _, call := range calls(body) {
			fn := callee(info, call)
			if fn == nil || !inNvidia(fn.Pkg()) {
				continue
			}
			sel, ok := ast.Unparen(call.Fun).(*ast.SelectorExpr)
			if !ok {
				continue
			}
			recv := objOf(info, sel.X)
			switch {
			case fn.Name() == "SetStream" && recv != nil && len(call.Args) == 1:
				if s := objOf(info, call.Args[0]); s != nil {
					bound[recv] = s
				}
				continue
			case fn.Name() == "WaitEvent" || set.isSync(info, call):
				waits = append(waits, call)
				continue
			case recv != nil && isDevicePtr(recv.Type()) && writesDevice(fn.Name()) && len(call.Args) > 0:
				if s := objOf(info, call.Args[len(call.Args)-1]); s != nil {
					written[recv] = write{call.Pos(), s}
				}
				continue
			}
			s, ok := bound[recv]
			if !ok {
				continue
			}
			for _, arg := range call.Args {
				d := objOf(info, arg)
				w, ok := written[d]
				if !ok || w.stream == s || waited(info, waits, w.pos, call.Pos(), s, w.stream) {
					continue
				}
				pass.Reportf(arg.Pos(), "%s was last written on stream %s, but %s runs on stream %s; record an event on %s and have %s wait for it",
					d.Name(), w.stream.Name(), recv.Name(), s.Name(), w.stream.Name(), s.Name())
			}
		}
	})
	return nil, nil
}

// writesDevice reports whether a DevicePtr method of this name writes the
// device buffer asynchronously, on the stream it takes last.
func writesDevice(name string) bool {
	name = strings.TrimSuffix(name, "Ptr")
	return strings.HasSuffix(name, "Async") && (strings.HasPrefix(name, "CopyFrom") || strings.HasPrefix(name, "Memset"))
}

// waited reports whether one of waits, between from and to, makes s wait
// for writer: s.WaitEvent, writer.Synchronize, or a context synchronization.
func waited(info *types.Info, waits []*ast.CallExpr, from, to token.Pos, s, writer *types.Var) bool {
	for _, w := range waits {
		if !between(w.Pos(), from, to) {
			continue
		}
		sel, ok := ast.Unparen(w.Fun).(*ast.SelectorExpr)
		if !ok {
			return true // a function call: CtxSynchronize or a helper
		}
		switch objOf(info, sel.X) {
		case s:
			if sel.Sel.Name == "WaitEvent" {
				return true
			}
		case writer:
			return true
		case nil:
			return true // driver.CtxSynchronize
		}
	}
	return false
}

var queryDoneAnalyzer = &analysis.Analyzer{
	Name: "querydone",
	Doc: `check that the answer of Event.Query and Stream.Query is used

The cuda package's Query returns whether the work is done and an error. A nil
error does not mean done: it means the query worked. querydone reports a Query
whose done result is discarded. A Query of a handle the function closes,
outside a defer, before the Query or while the Query runs in a goroutine, is
left alone: it tests the closed handle, and only the error means anything
there.

The driver's Query returns only an error: nil when the work is done,
ErrNotReady when it is not, and another error when the query failed.
querydone reports such a Query called as a bare statement, which throws the
answer away, and a loop that runs while the Query's error is non-nil and
never mentions ErrNotReady: a failure, which Query keeps returning, spins it
forever.`,
	Run: runQueryDone,
}

func runQueryDone(pass *analysis.Pass) (any, error) {
	info := pass.TypesInfo
	cleanup := cleanupBodies(pass)
	eachOuterFunc(pass, func(body *ast.BlockStmt) {
		closes := make(map[*types.Var][]token.Pos)
		withStackAll(body, func(n ast.Node, stack []ast.Node) {
			call, ok := n.(*ast.CallExpr)
			if !ok || deferred(cleanup, stack) {
				return
			}
			if sel, ok := call.Fun.(*ast.SelectorExpr); ok && isRelease(sel.Sel.Name) {
				if v := objOf(info, sel.X); v != nil {
					closes[v] = append(closes[v], call.Pos())
				}
			}
		})
		withStackAll(body, func(n ast.Node, stack []ast.Node) {
			call, ok := n.(*ast.CallExpr)
			if !ok || len(stack) == 0 {
				return
			}
			fn := callee(info, call)
			if fn == nil || fn.Name() != "Query" || !inNvidia(fn.Pkg()) || recvNamed(fn) == nil {
				return
			}
			if errQuery(info, call) != "" {
				if _, bare := stack[len(stack)-1].(*ast.ExprStmt); bare {
					pass.Reportf(call.Pos(), "the result of %s.Query is discarded; it is nil when the work is done and ErrNotReady when it is not", recvNamed(fn).Obj().Name())
				}
				return
			}
			res := fn.Type().(*types.Signature).Results()
			if res.Len() != 2 || !types.Identical(res.At(0).Type(), types.Typ[types.Bool]) {
				return
			}
			discarded := false
			switch p := stack[len(stack)-1].(type) {
			case *ast.ExprStmt:
				discarded = true
			case *ast.AssignStmt:
				if id, ok := p.Lhs[0].(*ast.Ident); ok && id.Name == "_" {
					discarded = true
				}
			}
			if !discarded {
				return
			}
			if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
				for _, pos := range closes[objOf(info, sel.X)] {
					if pos < call.Pos() || inGoroutine(stack) {
						return
					}
				}
			}
			pass.Reportf(call.Pos(), "the done result of %s.Query is discarded; a nil error does not mean the work is done", recvNamed(fn).Obj().Name())
		})
		ast.Inspect(body, func(n ast.Node) bool {
			loop, ok := n.(*ast.ForStmt)
			if !ok {
				return true
			}
			if name := spinQuery(info, loop); name != "" && !mentionsNotReady(info, loop) {
				pass.Reportf(loop.Cond.Pos(), "the loop waits while %s.Query returns any error; only ErrNotReady means not done, and a failure spins it forever", name)
			}
			return true
		})
	})
	return nil, nil
}

// errQuery returns the receiver type name of call if it calls a Query method
// of a binding that returns only an error, as the driver's does, or "".
func errQuery(info *types.Info, call *ast.CallExpr) string {
	fn := callee(info, call)
	if fn == nil || fn.Name() != "Query" || !inNvidia(fn.Pkg()) || recvNamed(fn) == nil {
		return ""
	}
	res := fn.Type().(*types.Signature).Results()
	if res.Len() != 1 || !isError(res.At(0).Type()) {
		return ""
	}
	return recvNamed(fn).Obj().Name()
}

// spinQuery returns the receiver type name of the error-only Query whose
// result the condition of loop tests against nil, directly or through a
// variable the loop assigns from the Query, or "".
func spinQuery(info *types.Info, loop *ast.ForStmt) string {
	cond, ok := ast.Unparen(loop.Cond).(*ast.BinaryExpr)
	if !ok || cond.Op != token.NEQ {
		return ""
	}
	x := cond.X
	if isNil(info, x) {
		x = cond.Y
	} else if !isNil(info, cond.Y) {
		return ""
	}
	if c, ok := ast.Unparen(x).(*ast.CallExpr); ok {
		return errQuery(info, c)
	}
	v := objOf(info, x)
	if v == nil {
		return ""
	}
	name := ""
	ast.Inspect(loop, func(n ast.Node) bool {
		a, ok := n.(*ast.AssignStmt)
		if !ok || len(a.Lhs) != len(a.Rhs) {
			return true
		}
		for i, l := range a.Lhs {
			if c, ok := ast.Unparen(a.Rhs[i]).(*ast.CallExpr); ok && objOf(info, l) == v {
				if q := errQuery(info, c); q != "" {
					name = q
				}
			}
		}
		return true
	})
	return name
}

// mentionsNotReady reports whether n refers to a binding's ErrNotReady.
func mentionsNotReady(info *types.Info, n ast.Node) bool {
	found := false
	ast.Inspect(n, func(n ast.Node) bool {
		if id, ok := n.(*ast.Ident); ok && id.Name == "ErrNotReady" {
			if obj := info.Uses[id]; obj != nil && inNvidia(obj.Pkg()) {
				found = true
			}
		}
		return !found
	})
	return found
}

// deferred reports whether a node with ancestors stack runs as a cleanup: it
// lies in a defer statement or in a cleanup body.
func deferred(cleanup map[*ast.BlockStmt]bool, stack []ast.Node) bool {
	for _, n := range stack {
		switch n := n.(type) {
		case *ast.DeferStmt:
			return true
		case *ast.BlockStmt:
			if cleanup[n] {
				return true
			}
		}
	}
	return false
}

// inGoroutine reports whether the stack of ancestors passes through a
// function literal started by a go statement.
func inGoroutine(stack []ast.Node) bool {
	for i, n := range stack {
		if _, ok := n.(*ast.GoStmt); ok && i+2 < len(stack) {
			if _, ok := stack[i+2].(*ast.FuncLit); ok {
				return true
			}
		}
	}
	return false
}
