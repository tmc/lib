package main

import (
	"go/ast"
	"go/token"
	"go/types"
	"regexp"

	"golang.org/x/tools/go/analysis"
)

var archAnalyzer = &analysis.Analyzer{
	Name: "arch",
	Doc: `report GPU architectures fixed in compiler options

A compile option such as --gpu-architecture=sm_90 fails, or falls back to a
slow JIT, on a GPU of another architecture. arch reports a constant
architecture option passed to an nvidia binding, such as nvrtc's
Program.Compile or nvJitLink's Create. Derive it from
Device.ComputeCapability instead.`,
	Run: runArch,
}

var archRE = regexp.MustCompile(`^(?:-arch[= ]?|--gpu-architecture[= ]?)?((?:sm|compute|lto)_\d+[af]?)$`)

func runArch(pass *analysis.Pass) (any, error) {
	info := pass.TypesInfo
	eachFunc(pass, func(body *ast.BlockStmt) {
		for _, call := range calls(body) {
			fn := callee(info, call)
			if fn == nil || !inNvidia(fn.Pkg()) {
				continue
			}
			for i, arg := range call.Args {
				s, ok := constString(info, arg)
				if !ok {
					continue
				}
				m := archRE.FindStringSubmatch(s)
				if m == nil {
					continue
				}
				// A bare sm_90 counts only after an -arch option.
				if s == m[1] {
					prev, ok := "", i > 0
					if ok {
						prev, ok = constString(info, call.Args[i-1])
					}
					if !ok || (prev != "-arch" && prev != "--gpu-architecture") {
						continue
					}
				}
				pass.Reportf(arg.Pos(), "%s fixes the GPU architecture at %s; derive it from Device.ComputeCapability", fn.Name(), m[1])
			}
		}
	})
	return nil, nil
}

var gridTruncAnalyzer = &analysis.Analyzer{
	Name: "gridtrunc",
	Doc: `check that grid sizes round up

A grid of n/block blocks covers n threads only when block divides n; otherwise
the last partial block is silently skipped. gridtrunc reports a grid
dimension given to driver.Dims, a Dim3 literal or LaunchGrid, directly or
through a local variable, as a truncating division. Use driver.GridFor(n,
block) or (n+block-1)/block.`,
	Run: runGridTrunc,
}

func runGridTrunc(pass *analysis.Pass) (any, error) {
	info := pass.TypesInfo
	eachFunc(pass, func(body *ast.BlockStmt) {
		defs := assignments(info, body)
		check := func(e ast.Expr) {
			q := truncatingDiv(info, defs, e)
			if q == nil {
				return
			}
			pass.Reportf(e.Pos(), "grid dimension %s truncates, skipping the last partial block; use driver.GridFor or round up", types.ExprString(q))
		}
		inspectShallow(body, func(n ast.Node) bool {
			switch n := n.(type) {
			case *ast.CallExpr:
				switch calleeIn(info, n, "driver") {
				case "Dims", "LaunchGrid", "LaunchGridAsync":
					if !isGridArg(info, body, n) {
						return true
					}
					for _, a := range n.Args {
						check(a)
					}
				}
			case *ast.CompositeLit:
				if !isNamed(info.TypeOf(n), "driver", "Dim3") || !isGridArg(info, body, n) {
					return true
				}
				for _, e := range n.Elts {
					if kv, ok := e.(*ast.KeyValueExpr); ok {
						e = kv.Value
					}
					check(e)
				}
			}
			return true
		})
	})
	return nil, nil
}

// isGridArg reports whether e, a Dim3 or a LaunchGrid call, gives a grid
// rather than a block: it is not passed or assigned as a block.
func isGridArg(info *types.Info, body *ast.BlockStmt, e ast.Expr) bool {
	if call, ok := e.(*ast.CallExpr); ok && calleeIn(info, call, "driver") != "Dims" {
		return true // LaunchGrid
	}
	block := false
	withStack(body, func(n ast.Node, stack []ast.Node) {
		if n != e || len(stack) == 0 {
			return
		}
		switch p := stack[len(stack)-1].(type) {
		case *ast.KeyValueExpr:
			if id, ok := p.Key.(*ast.Ident); ok && id.Name == "BlockDim" {
				block = true
			}
		case *ast.CallExpr:
			fn := callee(info, p)
			if fn == nil {
				return
			}
			sig := fn.Type().(*types.Signature)
			for i, a := range p.Args {
				if a == e && i < sig.Params().Len() && sig.Params().At(i).Name() == "blockDim" {
					block = true
				}
			}
		case *ast.AssignStmt:
			for i, r := range p.Rhs {
				if r == e && i < len(p.Lhs) {
					if id, ok := p.Lhs[i].(*ast.Ident); ok && isBlockName(id.Name) {
						block = true
					}
				}
			}
		}
	})
	return !block
}

var blockNameRE = regexp.MustCompile(`(?i)^(block|threads)`)

func isBlockName(name string) bool { return blockNameRE.MatchString(name) }

// truncatingDiv returns the division e computes, directly or through a
// variable assigned once, if it truncates: a / b where a is not itself a
// round-up sum and the quotient is not exact.
func truncatingDiv(info *types.Info, defs map[*types.Var][]ast.Expr, e ast.Expr) ast.Expr {
	e = unconvert(info, e)
	if id, ok := e.(*ast.Ident); ok {
		v, _ := info.Uses[id].(*types.Var)
		if d := defs[v]; len(d) == 1 && d[0] != nil {
			e = unconvert(info, d[0])
		}
	}
	b, ok := e.(*ast.BinaryExpr)
	if !ok || b.Op != token.QUO {
		return nil
	}
	if _, ok := intConst(info, b); ok {
		return nil // constant: the author can see whether it divides
	}
	if sum, ok := unconvert(info, b.X).(*ast.BinaryExpr); ok && (sum.Op == token.ADD || sum.Op == token.SUB) {
		return nil
	}
	if t, ok := info.TypeOf(b).Underlying().(*types.Basic); !ok || t.Info()&types.IsInteger == 0 {
		return nil
	}
	return b
}

var blockLimitAnalyzer = &analysis.Analyzer{
	Name: "blocklimit",
	Doc: `check constant launch shapes against the hardware limits

A block holds at most 1024 threads, and a kernel gets at most 48 KiB of
dynamic shared memory unless FuncAttributeMaxDynamicSharedSizeBytes is raised
first. blocklimit reports a constant block larger than 1024 threads, and a
constant dynamic shared memory size over 48 KiB in a function that sets no
function attribute.`,
	Run: runBlockLimit,
}

const (
	maxBlockThreads     = 1024
	defaultDynamicShmem = 48 << 10
)

func runBlockLimit(pass *analysis.Pass) (any, error) {
	info := pass.TypesInfo
	eachFunc(pass, func(body *ast.BlockStmt) {
		setsAttr := false
		for _, c := range calls(body) {
			if fn := callee(info, c); fn != nil && inPkg(fn.Pkg(), "driver") && (fn.Name() == "SetAttribute" || fn.Name() == "FuncSetAttribute") {
				setsAttr = true
			}
		}
		checkBlock := func(e ast.Expr) {
			if n, ok := dimProduct(info, e); ok && n > maxBlockThreads {
				pass.Reportf(e.Pos(), "block of %d threads exceeds the limit of %d", n, maxBlockThreads)
			}
		}
		checkShmem := func(e ast.Expr) {
			if n, ok := intConst(info, e); ok && n > defaultDynamicShmem && !setsAttr {
				pass.Reportf(e.Pos(), "%d bytes of dynamic shared memory exceeds the default 48 KiB; raise FuncAttributeMaxDynamicSharedSizeBytes first", n)
			}
		}
		inspectShallow(body, func(n ast.Node) bool {
			switch n := n.(type) {
			case *ast.CallExpr:
				fn := callee(info, n)
				if fn == nil || !inPkg(fn.Pkg(), "driver") {
					return true
				}
				sig := fn.Type().(*types.Signature)
				for i, a := range n.Args {
					if i >= sig.Params().Len() {
						break
					}
					switch sig.Params().At(i).Name() {
					case "blockDim":
						checkBlock(a)
					case "sharedMemBytes":
						checkShmem(a)
					}
				}
			case *ast.CompositeLit:
				if !isNamed(info.TypeOf(n), "driver", "LaunchConfig") {
					return true
				}
				for _, e := range n.Elts {
					kv, ok := e.(*ast.KeyValueExpr)
					if !ok {
						continue
					}
					switch kv.Key.(*ast.Ident).Name {
					case "BlockDim":
						checkBlock(kv.Value)
					case "SharedMemBytes":
						checkShmem(kv.Value)
					}
				}
			}
			return true
		})
	})
	return nil, nil
}

// dimProduct returns the thread count of a constant block shape given as
// driver.Dims or a Dim3 literal.
func dimProduct(info *types.Info, e ast.Expr) (int64, bool) {
	var dims []ast.Expr
	switch e := ast.Unparen(e).(type) {
	case *ast.CallExpr:
		if calleeIn(info, e, "driver") != "Dims" {
			return 0, false
		}
		dims = e.Args
	case *ast.CompositeLit:
		if !isNamed(info.TypeOf(e), "driver", "Dim3") {
			return 0, false
		}
		for _, elt := range e.Elts {
			if kv, ok := elt.(*ast.KeyValueExpr); ok {
				elt = kv.Value
			}
			dims = append(dims, elt)
		}
	default:
		return 0, false
	}
	n := int64(1)
	for _, d := range dims {
		v, ok := intConst(info, d)
		if !ok {
			return 0, false
		}
		n *= v
	}
	return n, len(dims) > 0
}
