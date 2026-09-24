package main

import (
	"go/ast"
	"go/types"

	"golang.org/x/tools/go/analysis"
)

// asyncHostPointer names the driver calls that read or write host memory
// after they return, and which argument is the host pointer.
var asyncHostPointer = map[string]int{
	"CopyFromHostAsync": 0,
	"CopyToHostAsync":   0,
	"MemcpyAtoHAsync":   0,
	"MemcpyHtoAAsync":   1,
}

var unpinnedAnalyzer = &analysis.Analyzer{
	Name: "unpinned",
	Doc: `check that Go memory handed to asynchronous DMA is pinned

An asynchronous copy reads or writes the host buffer after the call returns,
so the garbage collector must not move it in between. Converting the address
of Go memory to unsafe.Pointer does not stop that: only a runtime.Pinner held
until the stream completes does. unpinned reports such a conversion passed as
the host pointer of an asynchronous copy from a function that pins nothing.`,
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
		i, ok := asyncHostPointer[sel.Sel.Name]
		if !ok || i >= len(call.Args) {
			return true
		}
		if arg := call.Args[i]; isGoAddress(pass, arg) {
			pass.Reportf(arg.Pos(), "%s takes the address of Go memory; pin it with a runtime.Pinner held until the stream completes",
				sel.Sel.Name)
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
