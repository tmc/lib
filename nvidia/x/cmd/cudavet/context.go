package main

import (
	"go/ast"
	"go/token"
	"go/types"
	"strings"

	"golang.org/x/tools/go/analysis"
)

// makesCurrent reports whether call makes a context current on the calling
// thread: NewContext, NewContextParams, Context.SetCurrent or
// Context.PushCurrent.
func makesCurrent(fn *types.Func) bool {
	if fn == nil || !inPkg(fn.Pkg(), "driver") {
		return false
	}
	switch fn.Name() {
	case "NewContext", "NewContextParams":
		return recvNamed(fn) == nil
	case "SetCurrent", "PushCurrent":
		return methodOf(fn, "driver", "Context", fn.Name())
	}
	return false
}

// needsContext reports whether fn uses the calling thread's current context.
// Every call into an nvidia binding does, except the ones that set up the
// library, enumerate devices, or make a context current.
func needsContext(fn *types.Func) bool {
	if fn == nil || !inNvidia(fn.Pkg()) || strings.HasPrefix(fn.Pkg().Path(), nvidiaPrefix+"x/") {
		return false
	}
	if makesCurrent(fn) {
		return false
	}
	if r := recvNamed(fn); r != nil && r.Obj().Name() == "Device" {
		return false
	}
	switch name := fn.Name(); {
	case name == "Init", name == "Load", name == "Loaded", name == "Close", name == "Path",
		strings.HasPrefix(name, "DeviceGet"), strings.HasPrefix(name, "Has"),
		strings.HasPrefix(name, "GetError"), name == "DriverGetVersion", name == "GetVersion",
		name == "Error", name == "String", name == "Unwrap":
		return false
	}
	return true
}

var contextThreadAnalyzer = &analysis.Analyzer{
	Name: "contextthread",
	Doc: `check that a goroutine holds its OS thread while it uses a current context

The current context belongs to the OS thread, and the Go scheduler moves
goroutines between threads. contextthread reports a context made current
(NewContext, Context.SetCurrent, Context.PushCurrent) in a function that has
not called runtime.LockOSThread first, and a goroutine started after a
context was made current that makes driver calls without making a context
current itself: it runs on another thread, without the context.`,
	Run: runContextThread,
}

func runContextThread(pass *analysis.Pass) (any, error) {
	info := pass.TypesInfo

	// A function that makes a context current without locking its thread is
	// safe when every caller has locked it. Find those functions, following
	// calls within the package to a fixed point.
	decls := make(map[*types.Func]*ast.FuncDecl)
	for _, file := range srcFiles(pass) {
		for _, decl := range file.Decls {
			if fd, ok := decl.(*ast.FuncDecl); ok && fd.Body != nil {
				if fn, ok := info.Defs[fd.Name].(*types.Func); ok {
					decls[fn] = fd
				}
			}
		}
	}
	unlocked := make(map[*types.Func]*ast.CallExpr)
	for changed := true; changed; {
		changed = false
		for fn, fd := range decls {
			if unlocked[fn] != nil {
				continue
			}
			if c := unlockedCurrent(info, fd.Body, unlocked); c != nil {
				unlocked[fn] = c
				changed = true
			}
		}
	}
	for fn, c := range unlocked {
		if lockedByCallers(info, srcFiles(pass), fn) {
			continue
		}
		name := callee(info, c).Name()
		pass.Reportf(c.Pos(), "%s makes a context current on a thread the goroutine may leave; call runtime.LockOSThread first", name)
	}
	// A goroutine starts on a thread of its own, locked or not.
	for _, file := range srcFiles(pass) {
		ast.Inspect(file, func(n ast.Node) bool {
			if g, ok := n.(*ast.GoStmt); ok {
				if lit, ok := g.Call.Fun.(*ast.FuncLit); ok {
					if c := unlockedCurrent(info, lit.Body, unlocked); c != nil {
						pass.Reportf(c.Pos(), "%s makes a context current on a thread the goroutine may leave; call runtime.LockOSThread first", callee(info, c).Name())
					}
				}
			}
			return true
		})
	}

	eachFunc(pass, func(body *ast.BlockStmt) {
		var current token.Pos
		for _, call := range calls(body) {
			if makesCurrent(callee(info, call)) {
				current = call.Pos()
				break
			}
		}
		if current == token.NoPos {
			return
		}
		inspectShallow(body, func(n ast.Node) bool {
			g, ok := n.(*ast.GoStmt)
			if !ok || g.Pos() < current {
				return true
			}
			lit, ok := g.Call.Fun.(*ast.FuncLit)
			if !ok {
				return true
			}
			var use *types.Func
			for _, c := range calls(lit.Body) {
				fn := callee(info, c)
				if makesCurrent(fn) {
					return true
				}
				if use == nil && needsContext(fn) {
					use = fn
				}
			}
			if use != nil {
				pass.Reportf(g.Pos(), "goroutine calls %s without the context made current before it started; it runs on another thread, so make the context current inside it", use.Name())
			}
			return true
		})
	})
	return nil, nil
}

// unlockedCurrent returns the first call in body that makes a context
// current, directly or through a function in unlocked, before any
// runtime.LockOSThread. Closures other than goroutines run on the thread of
// the function around them, and are followed.
func unlockedCurrent(info *types.Info, body *ast.BlockStmt, unlocked map[*types.Func]*ast.CallExpr) *ast.CallExpr {
	locked := false
	var found *ast.CallExpr
	ast.Inspect(body, func(n ast.Node) bool {
		if found != nil || locked {
			return false
		}
		switch n := n.(type) {
		case *ast.GoStmt:
			return false
		case *ast.CallExpr:
			if isPkgFunc(info, n, "runtime", "LockOSThread") {
				locked = true
				return false
			}
			if fn := callee(info, n); makesCurrent(fn) || unlocked[fn] != nil {
				found = n
			}
		}
		return found == nil
	})
	return found
}

// lockedByCallers reports whether fn can rely on its callers to have locked
// the thread: it is unexported, it has callers, and every reference to it in
// the package is a direct call outside a go statement. Callers that did not
// lock are reported in their own right.
func lockedByCallers(info *types.Info, files []*ast.File, fn *types.Func) bool {
	if fn.Exported() {
		return false
	}
	ok, refs := true, 0
	for _, file := range files {
		withStackAll(file, func(n ast.Node, stack []ast.Node) {
			id, isID := n.(*ast.Ident)
			if !isID || info.Uses[id] != fn || len(stack) == 0 {
				return
			}
			refs++
			var ref ast.Node = id
			if sel, isSel := stack[len(stack)-1].(*ast.SelectorExpr); isSel && sel.Sel == id {
				ref = sel
				stack = stack[:len(stack)-1]
			}
			call, isCall := stack[len(stack)-1].(*ast.CallExpr)
			if !isCall || call.Fun != ref {
				ok = false // a func value: called from anywhere
				return
			}
			if len(stack) > 1 {
				if _, isGo := stack[len(stack)-2].(*ast.GoStmt); isGo {
					ok = false
				}
			}
		})
	}
	return ok && refs > 0
}

var unlockDestroyAnalyzer = &analysis.Analyzer{
	Name: "unlockdestroy",
	Doc: `check that a context is destroyed on the thread it was current on

Deferred calls run last-in first-out, so a defer runtime.UnlockOSThread written
after defer ctx.Close() runs first, and Close then runs on whatever thread
the goroutine lands on. unlockdestroy reports that order, an UnlockOSThread
followed by a Close of a context, and a context destroyed from a goroutine
other than the one that made it.`,
	Run: runUnlockDestroy,
}

// isContextRelease reports whether fn destroys or releases a context.
func isContextRelease(fn *types.Func) bool {
	return methodOf(fn, "driver", "Context", "Close") || methodOf(fn, "driver", "Device", "PrimaryCtxRelease")
}

func runUnlockDestroy(pass *analysis.Pass) (any, error) {
	info := pass.TypesInfo
	eachFunc(pass, func(body *ast.BlockStmt) {
		var deferredRelease, unlocked token.Pos
		var releaseName string
		withStack(body, func(n ast.Node, stack []ast.Node) {
			call, ok := n.(*ast.CallExpr)
			if !ok || len(stack) == 0 {
				return
			}
			_, deferred := stack[len(stack)-1].(*ast.DeferStmt)
			fn := callee(info, call)
			switch {
			case isContextRelease(fn) && deferred:
				if deferredRelease == token.NoPos {
					deferredRelease, releaseName = call.Pos(), fn.Name()
				}
			case isContextRelease(fn):
				if unlocked != token.NoPos {
					pass.Reportf(call.Pos(), "%s runs after runtime.UnlockOSThread, on whatever thread the goroutine is on", fn.Name())
				}
			case isPkgFunc(info, call, "runtime", "UnlockOSThread"):
				if deferred && deferredRelease != token.NoPos {
					pass.Reportf(call.Pos(), "this deferred UnlockOSThread runs before the deferred %s above it; defer it first", releaseName)
				} else if !deferred && unlocked == token.NoPos {
					unlocked = call.Pos()
				}
			}
		})
		// A context destroyed from a goroutine that did not make it.
		inspectShallow(body, func(n ast.Node) bool {
			g, ok := n.(*ast.GoStmt)
			if !ok {
				return true
			}
			lit, ok := g.Call.Fun.(*ast.FuncLit)
			if !ok {
				return true
			}
			for _, c := range calls(lit.Body) {
				if !isContextRelease(callee(info, c)) {
					continue
				}
				sel, ok := c.Fun.(*ast.SelectorExpr)
				if !ok {
					continue
				}
				if v := objOf(info, sel.X); v != nil && !(lit.Pos() < v.Pos() && v.Pos() < lit.End()) {
					pass.Reportf(c.Pos(), "%s.%s runs in a new goroutine, on a different thread from the one %s is current on", v.Name(), sel.Sel.Name, v.Name())
				}
			}
			return true
		})
	})
	return nil, nil
}

var graphShareAnalyzer = &analysis.Analyzer{
	Name: "graphshare",
	Doc: `check that a CUDA graph is not used from two goroutines at once

Graph objects are not thread safe: NVIDIA requires even cuGraphClone and
cuGraphInstantiate to be serialized with other calls on the same graph.
graphshare reports a Graph or GraphExec captured by a go statement's function
literal while the starting goroutine goes on using it, with no WaitGroup.Wait
or channel receive in between and no mutex taken in the goroutine.`,
	Run: runGraphShare,
}

func isGraph(t types.Type) bool {
	return isNamed(t, "driver", "Graph") || isNamed(t, "driver", "GraphExec")
}

func runGraphShare(pass *analysis.Pass) (any, error) {
	info := pass.TypesInfo
	eachFunc(pass, func(body *ast.BlockStmt) {
		inspectShallow(body, func(n ast.Node) bool {
			g, ok := n.(*ast.GoStmt)
			if !ok {
				return true
			}
			lit, ok := g.Call.Fun.(*ast.FuncLit)
			if !ok || locks(info, lit.Body) {
				return true
			}
			shared := make(map[*types.Var]bool)
			ast.Inspect(lit.Body, func(n ast.Node) bool {
				if id, ok := n.(*ast.Ident); ok {
					if v, ok := info.Uses[id].(*types.Var); ok && isGraph(v.Type()) && !(lit.Pos() < v.Pos() && v.Pos() < lit.End()) {
						shared[v] = true
					}
				}
				return true
			})
			if len(shared) == 0 {
				return true
			}
			handoff := handoffAfter(info, body, g.End())
			reported := make(map[*types.Var]bool)
			inspectShallow(body, func(n ast.Node) bool {
				id, ok := n.(*ast.Ident)
				if !ok || id.Pos() < g.End() || (handoff != token.NoPos && id.Pos() > handoff) {
					return true
				}
				v, _ := info.Uses[id].(*types.Var)
				if shared[v] && !reported[v] {
					reported[v] = true
					pass.Reportf(id.Pos(), "%s is used here while the goroutine started at line %d may be using it; calls on a graph must be serialized",
						v.Name(), pass.Fset.Position(g.Pos()).Line)
				}
				return true
			})
			return true
		})
	})
	return nil, nil
}

// locks reports whether body takes a sync.Mutex or sync.RWMutex.
func locks(info *types.Info, body ast.Node) bool {
	for _, c := range calls(body) {
		fn := callee(info, c)
		if fn != nil && fn.Pkg() != nil && fn.Pkg().Path() == "sync" && (fn.Name() == "Lock" || fn.Name() == "RLock") {
			return true
		}
	}
	return false
}

// handoffAfter returns the position of the first WaitGroup.Wait or channel
// receive in body after pos, or token.NoPos.
func handoffAfter(info *types.Info, body ast.Node, pos token.Pos) token.Pos {
	var at token.Pos
	inspectShallow(body, func(n ast.Node) bool {
		if at != token.NoPos || n == nil || n.Pos() < pos {
			return at == token.NoPos
		}
		switch n := n.(type) {
		case *ast.UnaryExpr:
			if n.Op == token.ARROW {
				at = n.Pos()
			}
		case *ast.CallExpr:
			if fn := callee(info, n); fn != nil && fn.Pkg() != nil && fn.Pkg().Path() == "sync" && fn.Name() == "Wait" {
				at = n.Pos()
			}
		}
		return at == token.NoPos
	})
	return at
}
