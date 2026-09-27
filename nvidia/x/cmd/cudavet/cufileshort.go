package main

import (
	"go/ast"

	"golang.org/x/tools/go/analysis"
)

var cufileShortAnalyzer = &analysis.Analyzer{
	Name: "cufileshort",
	Doc: `check that the byte count of a cuFile read or write is not discarded (advisory)

cufile's Handle.Read and Handle.Write return the number of bytes transferred.
NVIDIA documents it as the size that was successfully read or written and
does not promise it is the size asked for, so a short transfer is reported
only by the count. cufileshort reports a Read or Write whose count is
discarded: called as a statement, or with the count assigned to _. A
transfer of a constant zero bytes, as a test of the error path makes, cannot
be short and is not reported.

The check is advisory: the documentation neither promises a short transfer
nor rules one out, and a read that ends at end of file is short by design,
which a program that knows its file sizes may ignore. ReadAsync and
WriteAsync deliver the count through their bytes pointer after the stream
runs, and a batch through its IOEvents, so neither is checked.`,
	Run: runCufileShort,
}

func runCufileShort(pass *analysis.Pass) (any, error) {
	info := pass.TypesInfo
	eachFunc(pass, func(body *ast.BlockStmt) {
		withStack(body, func(n ast.Node, stack []ast.Node) {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return
			}
			fn := callee(info, call)
			if !methodOf(fn, "cufile", "Handle", "Read") && !methodOf(fn, "cufile", "Handle", "Write") {
				return
			}
			// A transfer of no bytes cannot be short.
			if n, ok := intConst(info, call.Args[1]); ok && n == 0 {
				return
			}
			if countDiscarded(stack[len(stack)-1]) {
				pass.Reportf(call.Pos(), "the byte count of Handle.%s is discarded; a short transfer is reported only by the count", fn.Name())
			}
		})
	})
	return nil, nil
}

// countDiscarded reports whether the first result of a call whose parent is
// parent is thrown away.
func countDiscarded(parent ast.Node) bool {
	switch p := parent.(type) {
	case *ast.ExprStmt:
		return true
	case *ast.AssignStmt:
		// A call that is a direct child of an assignment is its only value.
		id, ok := p.Lhs[0].(*ast.Ident)
		return ok && id.Name == "_"
	}
	return false
}
