package main

import (
	"go/ast"
	"go/types"
	"strings"

	"golang.org/x/tools/go/analysis"
)

// driverPath is the import path of the package that owns the driver handles
// whose lifetime this check follows.
const driverPath = "github.com/tmc/lib/nvidia/driver"

var leakAnalyzer = &analysis.Analyzer{
	Name: "leak",
	Doc: `check that created driver streams and events are destroyed

A stream or event is a driver resource, not Go memory: nothing collects it.
leak reports a local variable assigned a freshly created driver.Stream or
driver.Event whose every use is a method call on the handle, with no Destroy
among them. A handle that is returned, stored in a struct or passed on has an
owner elsewhere and is left alone.`,
	Run: runLeak,
}

func runLeak(pass *analysis.Pass) (any, error) {
	for _, file := range pass.Files {
		ast.Inspect(file, func(n ast.Node) bool {
			fn, ok := n.(*ast.FuncDecl)
			if !ok || fn.Body == nil {
				return true
			}
			checkLeaks(pass, fn.Body)
			return true
		})
	}
	return nil, nil
}

func checkLeaks(pass *analysis.Pass, body *ast.BlockStmt) {
	ast.Inspect(body, func(n ast.Node) bool {
		assign, ok := n.(*ast.AssignStmt)
		if !ok || len(assign.Rhs) != 1 {
			return true
		}
		call, ok := assign.Rhs[0].(*ast.CallExpr)
		if !ok || !createsHandle(pass, call) {
			return true
		}
		name, ok := assign.Lhs[0].(*ast.Ident)
		if !ok || name.Name == "_" {
			return true
		}
		if !released(body, name.Name, assign) {
			pass.Reportf(name.Pos(), "%s is created and never destroyed; defer %s.Destroy()", name.Name, name.Name)
		}
		return true
	})
}

// createsHandle reports whether call constructs a driver stream or event.
// Accessors that return a handle owned by something else, such as the event
// of a graph node, are not constructors and are not followed.
func createsHandle(pass *analysis.Pass, call *ast.CallExpr) bool {
	var name string
	switch fun := call.Fun.(type) {
	case *ast.Ident:
		name = fun.Name
	case *ast.SelectorExpr:
		name = fun.Sel.Name
	default:
		return false
	}
	if !strings.HasPrefix(name, "New") && !strings.Contains(name, "Create") {
		return false
	}
	sig, ok := pass.TypesInfo.TypeOf(call.Fun).(*types.Signature)
	if !ok || sig.Results().Len() == 0 {
		return false
	}
	return isDriverHandle(sig.Results().At(0).Type())
}

func isDriverHandle(t types.Type) bool {
	named, ok := t.(*types.Named)
	if !ok || named.Obj().Pkg() == nil || named.Obj().Pkg().Path() != driverPath {
		return false
	}
	return named.Obj().Name() == "Stream" || named.Obj().Name() == "Event"
}

// released reports whether name's lifetime ends somewhere the analyzer can
// account for. A handle that is returned, stored in a struct or passed on has
// been handed to an owner the analyzer cannot follow, and is not reported.
// Only a variable whose every remaining use is a method call on the handle
// itself, and never Destroy, is a leak.
func released(body *ast.BlockStmt, name string, def *ast.AssignStmt) bool {
	var stack []ast.Node
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		if n == nil {
			stack = stack[:len(stack)-1]
			return true
		}
		defer func() { stack = append(stack, n) }()
		id, ok := n.(*ast.Ident)
		if !ok || id.Name != name {
			return true
		}
		switch parent := stack[len(stack)-1].(type) {
		case *ast.SelectorExpr:
			// A method call on the handle. Only Destroy ends its life.
			if parent.X == id && parent.Sel.Name == "Destroy" {
				found = true
			}
		case *ast.AssignStmt:
			if parent == def || contains(parent.Lhs, id) {
				return true // the definition, or a later assignment to it
			}
			if allBlank(parent.Lhs) {
				return true // discarded, not handed anywhere
			}
			found = true
		default:
			// Returned, stored in a literal, passed as an argument: the
			// handle now belongs to something else.
			found = true
		}
		return true
	})
	return found
}

func contains(exprs []ast.Expr, id *ast.Ident) bool {
	for _, e := range exprs {
		if e == ast.Expr(id) {
			return true
		}
	}
	return false
}

func allBlank(exprs []ast.Expr) bool {
	for _, e := range exprs {
		if id, ok := e.(*ast.Ident); !ok || id.Name != "_" {
			return false
		}
	}
	return true
}
