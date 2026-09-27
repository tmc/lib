package main

import (
	"go/ast"
	"go/token"
	"go/types"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/cfg"
)

var captureSyncAnalyzer = &analysis.Analyzer{
	Name: "capturesync",
	Doc: `check that nothing synchronizes with a stream while it is being captured

Between Stream.BeginCapture and Stream.EndCapture, and inside the function
given to cudagraph.Capture, work on the stream is recorded, not run. A call
that waits for that work cannot be recorded: it fails and invalidates the
capture, or, for cuBLAS, forces a synchronization. capturesync reports, on a
path from the start of a capture to its end in one function:

  - Synchronize of the capturing stream;
  - CtxSynchronize or a Context Synchronize, which waits for every stream of
    the context, the capturing one included;
  - Synchronize of an event recorded on the capturing stream after the
    capture began;
  - a synchronous copy or memset of the driver, which runs on the legacy
    stream, when the capture mode is Global or ThreadLocal, which prohibit
    such calls; in Relaxed or an unknown mode it is legal on a non-blocking
    stream and is not reported;
  - a cuBLAS routine that writes its result to host memory, such as Sdot or
    Isamax, on a handle bound by SetStream to the capturing stream. A result
    returned as a Go value or through a Go pointer is host memory unless the
    handle is known to be in device pointer mode; a result through an
    unsafe.Pointer, as in DotEx, is reported only when the handle is known to
    be in host pointer mode.

A stream is known by the variable naming it, through the Raw method of an
x/cuda Stream. A capture whose stream has no EndCapture in the function is
ended elsewhere and is not followed. A synchronization of another stream,
or through a helper function, is not reported.`,
	Run: runCaptureSync,
}

// A capture is a stretch of a function while a stream is being captured.
type capture struct {
	stream *types.Var
	mode   string // name of the StreamCaptureMode constant, or ""
	calls  []*ast.CallExpr
}

func runCaptureSync(pass *analysis.Pass) (any, error) {
	info := pass.TypesInfo
	eachFunc(pass, func(body *ast.BlockStmt) {
		caps := append(streamCaptures(info, body), graphCaptures(info, body)...)
		if len(caps) == 0 {
			return
		}
		all := calls(body)
		events := modeEvents(info, body, all, soleSource(info, body))
		blocks, loops := blocksAndLoops(body)
		mode := func(h *types.Var, pos token.Pos) pointerMode {
			if events[h] == nil {
				return modeUnknown
			}
			return modeAt(events[h], pos, blocks, loops)
		}
		binds := streamBindings(info, body)
		reported := make(map[token.Pos]bool)
		for _, c := range caps {
			for _, call := range c.calls {
				if !reported[call.Pos()] {
					reported[call.Pos()] = reportCaptureSync(pass, c, call, binds, mode)
				}
			}
		}
	})
	return nil, nil
}

// reportCaptureSync reports call if it must not be made during capture c,
// and reports whether it did.
func reportCaptureSync(pass *analysis.Pass, c capture, call *ast.CallExpr, binds []streamBinding, mode func(*types.Var, token.Pos) pointerMode) bool {
	info := pass.TypesInfo
	fn := callee(info, call)
	if fn == nil || !inNvidia(fn.Pkg()) {
		return false
	}
	s := c.stream.Name()
	recv := receiverVar(info, call)
	switch {
	case methodOf(fn, "driver", "Stream", "Synchronize") || methodOf(fn, "x/cuda", "Stream", "Synchronize"):
		if streamVar(info, receiverExpr(call)) != c.stream {
			return false
		}
		pass.Reportf(call.Pos(), "%s.Synchronize waits for stream %s while it is being captured, which invalidates the capture", s, s)
	case inPkg(fn.Pkg(), "driver") && fn.Name() == "CtxSynchronize" && recvNamed(fn) == nil,
		methodOf(fn, "driver", "Context", "SynchronizeV2"),
		methodOf(fn, "x/cuda", "Context", "Synchronize"):
		pass.Reportf(call.Pos(), "%s waits for every stream of the context while %s is being captured, which invalidates the capture", fn.Name(), s)
	case methodOf(fn, "driver", "Event", "Synchronize") || methodOf(fn, "x/cuda", "Event", "Synchronize"):
		if recv == nil || !recordedOn(info, c, recv, call.Pos()) {
			return false
		}
		pass.Reportf(call.Pos(), "%s.Synchronize waits for an event recorded on %s while it is being captured, which invalidates the capture", recv.Name(), s)
	case syncTransfer(fn):
		if c.mode != "StreamCaptureModeGlobal" && c.mode != "StreamCaptureModeThreadLocal" {
			return false
		}
		pass.Reportf(call.Pos(), "%s is synchronous and runs on the legacy stream, which %s capture mode prohibits while %s is being captured; use the Async form on %s",
			fn.Name(), strings.TrimPrefix(c.mode, "StreamCaptureMode"), s, s)
	case inPkg(fn.Pkg(), "cublas") && hostOutput(fn.Name()) && recv != nil:
		if boundTo(binds, recv, call.Pos()) != c.stream {
			return false
		}
		m := mode(recv, call.Pos())
		if !(goResult(fn) && m != modeDevice) && m != modeHost {
			return false
		}
		pass.Reportf(call.Pos(), "%s writes its result to host memory, which forces a synchronization while %s is being captured; set device pointer mode and write the result to device memory", fn.Name(), s)
	default:
		return false
	}
	return true
}

// streamCaptures returns the captures begun in body by BeginCapture or
// BeginCaptureToGraph, each with the calls on some path from the begin call
// to the EndCapture of the same stream.
func streamCaptures(info *types.Info, body *ast.BlockStmt) []capture {
	ended := make(map[*types.Var]bool)
	for _, call := range calls(body) {
		if isCaptureCall(info, call, "EndCapture") {
			ended[streamVar(info, receiverExpr(call))] = true
		}
	}
	var out []capture
	g := cfg.New(body, func(call *ast.CallExpr) bool { return !noReturn(info, call) })
	for _, b := range g.Blocks {
		for i, n := range b.Nodes {
			for _, call := range calls(n) {
				var modeArg ast.Expr
				switch {
				case isCaptureCall(info, call, "BeginCapture") && len(call.Args) == 1:
					modeArg = call.Args[0]
				case isCaptureCall(info, call, "BeginCaptureToGraph") && len(call.Args) > 0:
					modeArg = call.Args[len(call.Args)-1]
				default:
					continue
				}
				s := streamVar(info, receiverExpr(call))
				if s == nil || !ended[s] {
					continue
				}
				c := capture{stream: s, mode: constName(info, modeArg)}
				c.calls = capturedCalls(info, b, i+1, s, make(map[*cfg.Block]bool))
				out = append(out, c)
			}
		}
	}
	return out
}

// capturedCalls returns the calls made on the paths from node i of block b
// up to an EndCapture of s. Deferred calls and goroutines are left out.
func capturedCalls(info *types.Info, b *cfg.Block, i int, s *types.Var, seen map[*cfg.Block]bool) []*ast.CallExpr {
	var out []*ast.CallExpr
	for _, n := range b.Nodes[i:] {
		switch n.(type) {
		case *ast.DeferStmt, *ast.GoStmt:
			continue
		}
		for _, call := range calls(n) {
			if isCaptureCall(info, call, "EndCapture") && streamVar(info, receiverExpr(call)) == s {
				return out
			}
		}
		out = append(out, calls(n)...)
	}
	for _, succ := range b.Succs {
		if !seen[succ] {
			seen[succ] = true
			out = append(out, capturedCalls(info, succ, 0, s, seen)...)
		}
	}
	return out
}

// graphCaptures returns the captures made by cudagraph.Capture(s, fn) in
// body where fn is a function literal: all its calls are captured on s, in
// Global mode.
func graphCaptures(info *types.Info, body *ast.BlockStmt) []capture {
	var out []capture
	for _, call := range calls(body) {
		if !isPkgFunc(info, call, nvidiaPrefix+"x/cudagraph", "Capture") || len(call.Args) != 2 {
			continue
		}
		lit, ok := ast.Unparen(call.Args[1]).(*ast.FuncLit)
		s := streamVar(info, call.Args[0])
		if !ok || s == nil {
			continue
		}
		var in []*ast.CallExpr
		withStack(lit.Body, func(n ast.Node, stack []ast.Node) {
			if c, ok := n.(*ast.CallExpr); ok && !inGoStmt(stack) {
				in = append(in, c)
			}
		})
		out = append(out, capture{stream: s, mode: "StreamCaptureModeGlobal", calls: in})
	}
	return out
}

// inGoStmt reports whether the stack of ancestors passes through a go
// statement.
func inGoStmt(stack []ast.Node) bool {
	for _, n := range stack {
		if _, ok := n.(*ast.GoStmt); ok {
			return true
		}
	}
	return false
}

// isCaptureCall reports whether call is the driver Stream method name.
func isCaptureCall(info *types.Info, call *ast.CallExpr, name string) bool {
	return methodOf(callee(info, call), "driver", "Stream", name)
}

// receiverExpr returns the receiver of a method call, or nil.
func receiverExpr(call *ast.CallExpr) ast.Expr {
	if sel, ok := ast.Unparen(call.Fun).(*ast.SelectorExpr); ok {
		return sel.X
	}
	return nil
}

// streamVar returns the variable a stream expression names, seeing through
// the Raw method of an x/cuda Stream, or nil.
func streamVar(info *types.Info, e ast.Expr) *types.Var {
	if e == nil {
		return nil
	}
	e = unconvert(info, e)
	if call, ok := e.(*ast.CallExpr); ok && len(call.Args) == 0 && methodOf(callee(info, call), "x/cuda", "Stream", "Raw") {
		e = receiverExpr(call)
	}
	return objOf(info, e)
}

// syncTransfer reports whether fn is a synchronous copy or memset of the driver:
// a function or DevicePtr or Array method named Copy, Memcpy or MemsetD
// without Async.
func syncTransfer(fn *types.Func) bool {
	if !inPkg(fn.Pkg(), "driver") || strings.Contains(fn.Name(), "Async") {
		return false
	}
	if r := recvNamed(fn); r != nil && r.Obj().Name() != "DevicePtr" && r.Obj().Name() != "Array" {
		return false
	}
	name := fn.Name()
	return strings.HasPrefix(name, "Copy") || strings.HasPrefix(name, "Memcpy") || strings.HasPrefix(name, "MemsetD")
}

// hostOutput reports whether the cuBLAS routine name returns a result, which
// in host pointer mode lands in host memory: a dot product, norm, sum,
// index of the extreme element, or rotation.
func hostOutput(name string) bool {
	name = strings.ToLower(name)
	for _, s := range []string{"dot", "nrm2", "asum", "amax", "amin", "rotg", "rotmg"} {
		if strings.Contains(name, s) {
			return true
		}
	}
	return false
}

// goResult reports whether fn hands its result back in Go memory: as a
// result other than the error, or through a Go pointer parameter.
func goResult(fn *types.Func) bool {
	sig := fn.Type().(*types.Signature)
	if sig.Results().Len() > 1 {
		return true
	}
	for p := range sig.Params().Variables() {
		if _, ok := p.Type().Underlying().(*types.Pointer); ok {
			return true
		}
	}
	return false
}

// A streamBinding is a SetStream call binding a handle to a stream.
type streamBinding struct {
	pos    token.Pos
	handle *types.Var
	stream *types.Var
}

// streamBindings returns the SetStream calls of the bindings in body,
// function literals included, in source order.
func streamBindings(info *types.Info, body *ast.BlockStmt) []streamBinding {
	var out []streamBinding
	ast.Inspect(body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok || len(call.Args) != 1 {
			return true
		}
		fn := callee(info, call)
		if fn == nil || fn.Name() != "SetStream" || !isBinding(fn.Pkg()) {
			return true
		}
		if h := receiverVar(info, call); h != nil {
			out = append(out, streamBinding{call.Pos(), h, streamVar(info, call.Args[0])})
		}
		return true
	})
	return out
}

// boundTo returns the stream handle h was last bound to before pos, or nil.
func boundTo(binds []streamBinding, h *types.Var, pos token.Pos) *types.Var {
	var s *types.Var
	for _, b := range binds {
		if b.pos < pos && b.handle == h {
			s = b.stream
		}
	}
	return s
}

// recordedOn reports whether event e is recorded on the stream of capture c
// by one of its calls before pos.
func recordedOn(info *types.Info, c capture, e *types.Var, pos token.Pos) bool {
	for _, call := range c.calls {
		fn := callee(info, call)
		if call.Pos() >= pos || len(call.Args) == 0 || receiverVar(info, call) != e {
			continue
		}
		if methodOf(fn, "driver", "Event", "Record") || methodOf(fn, "driver", "Event", "RecordWithFlags") || methodOf(fn, "x/cuda", "Event", "Record") {
			if streamVar(info, call.Args[0]) == c.stream {
				return true
			}
		}
	}
	return false
}
