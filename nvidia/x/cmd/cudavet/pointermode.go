package main

import (
	"go/ast"
	"go/token"
	"go/types"
	"slices"

	"golang.org/x/tools/go/analysis"
)

var pointerModeAnalyzer = &analysis.Analyzer{
	Name: "pointermode",
	Doc: `check that scalars match the pointer mode they are read under

A Scalar made by DeviceScalar holds a device address, which the library
reads as an address only in device pointer mode; a host Scalar holds a
value, which it reads as a value only in host pointer mode. Read the other
way, an address is used as a number or a number as an address, and no call
checks it. pointermode follows a cuBLAS or cuSPARSE handle created by Create
in the function through its SetPointerMode calls, and a cuBLASLt MatmulDesc
or MatrixTransformDesc created in the function, whose pointer mode is host
until its pointer-mode attribute is set, and reports a Scalar of the other
kind passed to a call on it. A handle or descriptor that comes from
elsewhere, is passed to a function outside the bindings, or whose mode is
set to a value that is not a constant, is in an unknown mode and is not
checked. A mode set in a nested block or later in an enclosing loop leaves
the mode unknown where the block does not reach.`,
	Run: runPointerMode,
}

// A pointerMode is the pointer mode of a handle or descriptor at a point in
// a function.
type pointerMode int

const (
	modeUnknown pointerMode = iota
	modeHost
	modeDevice
)

// A modeEvent is a point where a holder's pointer mode becomes mode.
type modeEvent struct {
	pos  token.Pos
	mode pointerMode
}

// modeHolders gives, for each package, the constructors of the values that
// carry a pointer mode, which all start in host mode.
var modeHolders = map[string][]string{
	"cublas":   {"Create"},
	"cusparse": {"Create"},
	"cublaslt": {"NewMatmulDesc", "NewMatrixTransformDesc"},
}

// ltPointerModeAttrs are the cuBLASLt descriptor attributes that set the
// pointer mode. Their value is behind a uintptr, so setting one makes the
// mode unknown.
var ltPointerModeAttrs = map[string]bool{
	"MatmulDescPointerMode":          true,
	"MatrixTransformDescPointerMode": true,
}

func runPointerMode(pass *analysis.Pass) (any, error) {
	info := pass.TypesInfo
	eachFunc(pass, func(body *ast.BlockStmt) {
		source := soleSource(info, body)
		all := calls(body)

		// Find the holders created here and each change to their mode.
		events := make(map[*types.Var][]modeEvent)
		for _, call := range all {
			fn := callee(info, call)
			if fn == nil || !isBinding(fn.Pkg()) {
				continue
			}
			ctors, ok := modeHolders[fn.Pkg().Name()]
			if !ok || recvNamed(fn) != nil || !slices.Contains(ctors, fn.Name()) {
				continue
			}
			id := assignedIdent(body, call)
			if id == nil || source(id) != call {
				continue
			}
			v := objOf(info, id)
			if v == nil || !declaredIn(v, body) || addressedOrCaptured(info, body, v) {
				continue
			}
			events[v] = []modeEvent{{call.Pos(), modeHost}}
		}
		if len(events) == 0 {
			return
		}
		for _, call := range all {
			fn := callee(info, call)
			if recv := receiverVar(info, call); recv != nil && events[recv] != nil && fn != nil && isBinding(fn.Pkg()) {
				if m, ok := modeSet(info, fn, call); ok {
					events[recv] = append(events[recv], modeEvent{call.Pos(), m})
				}
			}
			if fn != nil && isBinding(fn.Pkg()) {
				continue
			}
			// Any other function may change the mode of a holder it is given.
			for _, arg := range call.Args {
				if v := objOf(info, arg); v != nil && events[v] != nil {
					events[v] = append(events[v], modeEvent{call.Pos(), modeUnknown})
				}
			}
		}

		blocks, loops := blocksAndLoops(body)
		for _, call := range all {
			fn := callee(info, call)
			if fn == nil || !isBinding(fn.Pkg()) {
				continue
			}
			var holders []ast.Expr
			if sel, ok := ast.Unparen(call.Fun).(*ast.SelectorExpr); ok && recvNamed(fn) != nil {
				holders = append(holders, sel.X)
			}
			holders = append(holders, call.Args...)
			for _, h := range holders {
				v := objOf(info, h)
				if v == nil || events[v] == nil {
					continue
				}
				mode := modeAt(events[v], call.Pos(), blocks, loops)
				if mode == modeUnknown {
					continue
				}
				checkScalars(pass, fn, call, source, v, mode)
			}
		}
	})
	return nil, nil
}

// assignedIdent returns the variable the first result of call is assigned
// to, or nil.
func assignedIdent(body *ast.BlockStmt, call *ast.CallExpr) *ast.Ident {
	var id *ast.Ident
	ast.Inspect(body, func(n ast.Node) bool {
		if a, ok := n.(*ast.AssignStmt); ok && len(a.Rhs) == 1 && a.Rhs[0] == ast.Expr(call) {
			id, _ = a.Lhs[0].(*ast.Ident)
		}
		return id == nil
	})
	return id
}

// checkScalars reports each Scalar argument of call whose kind disagrees
// with mode, the pointer mode of holder v.
func checkScalars(pass *analysis.Pass, fn *types.Func, call *ast.CallExpr, source func(ast.Expr) *ast.CallExpr, v *types.Var, mode pointerMode) {
	info := pass.TypesInfo
	sig := fn.Type().(*types.Signature)
	for i := range min(sig.Params().Len(), len(call.Args)) {
		if !isNamed(sig.Params().At(i).Type(), fn.Pkg().Name(), "Scalar") {
			continue
		}
		arg := call.Args[i]
		switch {
		case mode == modeHost && isDeviceScalar(info, fn.Pkg(), arg, source):
			pass.Reportf(arg.Pos(), "%s is a device Scalar, but %s is in host pointer mode", render(pass.Fset, arg), v.Name())
		case mode == modeDevice:
			if kind := scalarKind(info, fn.Pkg(), arg, source); kind != "" {
				pass.Reportf(arg.Pos(), "%s is a host %s Scalar, but %s is in device pointer mode", render(pass.Fset, arg), kind, v.Name())
			}
		}
	}
}

// isDeviceScalar reports whether e is a Scalar of package pkg made by
// DeviceScalar, directly or through a variable assigned once.
func isDeviceScalar(info *types.Info, pkg *types.Package, e ast.Expr, source func(ast.Expr) *ast.CallExpr) bool {
	c, ok := ast.Unparen(e).(*ast.CallExpr)
	if !ok {
		c = source(e)
	}
	if c == nil {
		return false
	}
	fn := callee(info, c)
	return fn != nil && fn.Pkg() == pkg && fn.Name() == "DeviceScalar"
}

// modeSet returns the mode a call of fn sets on its receiver, and whether
// it sets one.
func modeSet(info *types.Info, fn *types.Func, call *ast.CallExpr) (pointerMode, bool) {
	if len(call.Args) == 0 {
		return 0, false
	}
	switch fn.Name() {
	case "SetPointerMode":
		switch constName(info, call.Args[0]) {
		case "PointerModeHost":
			return modeHost, true
		case "PointerModeDevice":
			return modeDevice, true
		}
		return modeUnknown, true
	case "SetAttribute":
		if fn.Pkg().Name() != "cublaslt" {
			return 0, false
		}
		if name := constName(info, call.Args[0]); name != "" && !ltPointerModeAttrs[name] {
			return 0, false
		}
		return modeUnknown, true
	}
	return 0, false
}

// modeAt returns the mode at pos given the events of one holder: the mode
// of the last event before pos, when the block holding that event also
// holds pos and no event in a loop around pos follows it.
func modeAt(events []modeEvent, pos token.Pos, blocks, loops []ast.Node) pointerMode {
	var last *modeEvent
	for i := range events {
		e := &events[i]
		if e.pos < pos {
			last = e
			continue
		}
		for _, l := range loops {
			if l.Pos() < pos && pos < l.End() && e.pos < l.End() {
				return modeUnknown
			}
		}
	}
	if last == nil {
		return modeUnknown
	}
	b := innermost(blocks, last.pos)
	if b == nil || pos >= b.End() {
		return modeUnknown
	}
	return last.mode
}

// blocksAndLoops returns the blocks and case clauses of body and its loops,
// outside nested function literals.
func blocksAndLoops(body *ast.BlockStmt) (blocks, loops []ast.Node) {
	inspectShallow(body, func(n ast.Node) bool {
		switch n.(type) {
		case *ast.BlockStmt, *ast.CaseClause, *ast.CommClause:
			blocks = append(blocks, n)
		case *ast.ForStmt, *ast.RangeStmt:
			loops = append(loops, n)
		}
		return true
	})
	return blocks, loops
}

// innermost returns the smallest of nodes that holds pos.
func innermost(nodes []ast.Node, pos token.Pos) ast.Node {
	var in ast.Node
	for _, n := range nodes {
		if n.Pos() <= pos && pos < n.End() && (in == nil || n.End()-n.Pos() < in.End()-in.Pos()) {
			in = n
		}
	}
	return in
}

// receiverVar returns the variable a method call is made on, or nil.
func receiverVar(info *types.Info, call *ast.CallExpr) *types.Var {
	sel, ok := ast.Unparen(call.Fun).(*ast.SelectorExpr)
	if !ok {
		return nil
	}
	if s := info.Selections[sel]; s == nil || s.Kind() != types.MethodVal {
		return nil
	}
	return objOf(info, sel.X)
}

// addressedOrCaptured reports whether body takes the address of v or refers
// to it from a function literal, so that the mode can change out of sight.
func addressedOrCaptured(info *types.Info, body *ast.BlockStmt, v *types.Var) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.UnaryExpr:
			if n.Op == token.AND && objOf(info, n.X) == v {
				found = true
			}
		case *ast.FuncLit:
			if uses(info, n, v) {
				found = true
			}
		}
		return !found
	})
	return found
}
