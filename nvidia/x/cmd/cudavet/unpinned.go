package main

import (
	"go/ast"
	"go/types"

	"golang.org/x/tools/go/analysis"
)

// asyncHostCopy names the driver calls that read or write host memory after
// they return: the slice forms and the unsafe.Pointer forms of the
// asynchronous copies between host and device memory or arrays.
var asyncHostCopy = map[string]bool{
	"CopyFromHostAsync":    true,
	"CopyToHostAsync":      true,
	"CopyFromHostAsyncPtr": true,
	"CopyToHostAsyncPtr":   true,
}

var unpinnedAnalyzer = &analysis.Analyzer{
	Name: "unpinned",
	Doc: `check that Go memory handed to asynchronous DMA is pinned

An asynchronous copy reads or writes the host buffer after the call returns,
so the garbage collector must not move it in between. Converting the address
of Go memory to unsafe.Pointer does not stop that, and neither does passing a
slice: only a runtime.Pinner held until the stream completes does. unpinned
reports such a conversion, or a slice, passed to an asynchronous copy of a
generated binding from a function that pins nothing. A slice over host memory
from the driver, made with unsafe.Slice, needs no pinning and is not reported
when the call makes it. The copies of the packages under x/, such as
cudamem's, pin the buffer themselves and are not reported.`,
	Run: runUnpinned,
}

func runUnpinned(pass *analysis.Pass) (any, error) {
	for _, file := range pass.Files {
		ast.Inspect(file, func(n ast.Node) bool {
			fn, ok := n.(*ast.FuncDecl)
			if !ok || fn.Body == nil {
				return true
			}
			if !pins(pass, fn.Body) {
				reportUnpinned(pass, fn.Body)
			}
			return true
		})
	}
	return nil, nil
}

func reportUnpinned(pass *analysis.Pass, body *ast.BlockStmt) {
	ast.Inspect(body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		if fn := callee(pass.TypesInfo, call); !asyncHostCopy[sel.Sel.Name] || fn == nil || !isBinding(fn.Pkg()) {
			return true
		}
		for _, arg := range call.Args {
			switch {
			case isGoAddress(pass, arg):
				pass.Reportf(arg.Pos(), "%s takes the address of Go memory; pin it with a runtime.Pinner held until the stream completes",
					sel.Sel.Name)
			case isGoSlice(pass, arg):
				pass.Reportf(arg.Pos(), "%s takes a Go slice; pin its first element with a runtime.Pinner held until the stream completes",
					sel.Sel.Name)
			}
		}
		return true
	})
}

// isGoAddress reports whether e converts the address of a Go variable to
// unsafe.Pointer, the shape that hands collector-owned memory to the driver.
func isGoAddress(pass *analysis.Pass, e ast.Expr) bool {
	call, ok := e.(*ast.CallExpr)
	if !ok || len(call.Args) != 1 {
		return false
	}
	id, ok := call.Fun.(*ast.SelectorExpr)
	if !ok || id.Sel.Name != "Pointer" {
		return false
	}
	pkg, ok := id.X.(*ast.Ident)
	if !ok || pkg.Name != "unsafe" {
		return false
	}
	unary, ok := call.Args[0].(*ast.UnaryExpr)
	return ok && unary.Op.String() == "&"
}

// isGoSlice reports whether e is a slice other than one made by unsafe.Slice,
// the way host memory the driver allocated is viewed as a slice.
func isGoSlice(pass *analysis.Pass, e ast.Expr) bool {
	if _, ok := pass.TypesInfo.TypeOf(e).Underlying().(*types.Slice); !ok {
		return false
	}
	if call, ok := e.(*ast.CallExpr); ok {
		if sel, ok := call.Fun.(*ast.SelectorExpr); ok && sel.Sel.Name == "Slice" {
			if pkg, ok := sel.X.(*ast.Ident); ok && pkg.Name == "unsafe" {
				return false
			}
		}
	}
	return true
}

// pins reports whether the function pins anything at all. The check is
// deliberately coarse: whether the pinned object is the one being copied is
// beyond what the analyzer can see, so a function that pins is left alone.
func pins(pass *analysis.Pass, body *ast.BlockStmt) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		fn, ok := pass.TypesInfo.Uses[sel.Sel].(*types.Func)
		if ok && fn.Name() == "Pin" && fn.Pkg() != nil && fn.Pkg().Path() == "runtime" {
			found = true
		}
		return !found
	})
	return found
}
