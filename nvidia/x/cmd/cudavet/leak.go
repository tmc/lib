package main

import (
	"go/ast"
	"go/types"
	"regexp"
	"strings"

	"golang.org/x/tools/go/analysis"
)

var leakAnalyzer = &analysis.Analyzer{
	Name: "leak",
	Doc: `check that created driver and library handles are released

A stream, context, allocation, module, plan or library handle is a resource
the driver or library owns, not Go memory: nothing collects it. leak reports
a local variable assigned a freshly created handle whose every use is a method
call on the handle, with no release among them. A handle that is returned,
stored in a struct or passed on has an owner elsewhere and is left alone.

The pairs are not listed by hand. A handle type is any type of an nvidia
package with a destructor method, one that takes nothing and returns only an
error and is named Destroy, Free or Unload, or begins with Destroy. That is
the method set the generator emits, so a new binding is covered when it is
generated. A constructor is a function or method returning such a type whose
name says it makes one: New, Create, MemAlloc, ModuleLoad, Plan, Clone,
Instantiate or CommInit.`,
	Run: runLeak,
}

// constructorRE matches the names of the calls that make a handle. Accessors
// that return a handle owned by something else, such as the event of a graph
// node or the retained primary context, do not match.
var constructorRE = regexp.MustCompile(`^(New|Create|MemAlloc|ModuleLoad|Plan|Clone|Instantiate|CommInit)|Create`)

func runLeak(pass *analysis.Pass) (any, error) {
	eachFunc(pass, func(body *ast.BlockStmt) { checkLeaks(pass, body) })
	return nil, nil
}

func checkLeaks(pass *analysis.Pass, body *ast.BlockStmt) {
	inspectShallow(body, func(n ast.Node) bool {
		assign, ok := n.(*ast.AssignStmt)
		if !ok || len(assign.Rhs) != 1 {
			return true
		}
		call, ok := assign.Rhs[0].(*ast.CallExpr)
		if !ok {
			return true
		}
		destroy := createsHandle(pass, call)
		if destroy == "" {
			return true
		}
		name, ok := assign.Lhs[0].(*ast.Ident)
		if !ok || name.Name == "_" {
			return true
		}
		// A variable of an enclosing function is released there, if at all.
		if obj := pass.TypesInfo.ObjectOf(name); obj == nil || obj.Pos() < body.Pos() || obj.Pos() > body.End() {
			return true
		}
		if !released(body, name.Name, assign) {
			pass.Reportf(name.Pos(), "%s is created and never released; defer %s.%s()", name.Name, name.Name, destroy)
		}
		return true
	})
}

// createsHandle reports the destructor of the handle call constructs, or ""
// if call is not a constructor.
func createsHandle(pass *analysis.Pass, call *ast.CallExpr) string {
	fn := callee(pass.TypesInfo, call)
	if fn == nil || !inNvidia(fn.Pkg()) || !constructorRE.MatchString(fn.Name()) {
		return ""
	}
	sig := fn.Type().(*types.Signature)
	if sig.Results().Len() == 0 {
		return ""
	}
	return destructor(sig.Results().At(0).Type())
}

// destructor returns the name of the method that releases a handle of type
// t, or "" if t is not a handle type of an nvidia package.
func destructor(t types.Type) string {
	named, ok := types.Unalias(t).(*types.Named)
	if !ok || !inNvidia(named.Obj().Pkg()) {
		return ""
	}
	// Free is preferred over FreeAsync, and Destroy over DestroyX.
	best := ""
	for i := range named.NumMethods() {
		m := named.Method(i)
		if !m.Exported() || !isDestructorName(m.Name()) {
			continue
		}
		sig := m.Type().(*types.Signature)
		if sig.Params().Len() != 0 || sig.Results().Len() != 1 || !isError(sig.Results().At(0).Type()) {
			continue
		}
		if best == "" || len(m.Name()) < len(best) {
			best = m.Name()
		}
	}
	return best
}

func isDestructorName(name string) bool {
	return name == "Free" || name == "Unload" || strings.HasPrefix(name, "Destroy")
}

// isRelease reports whether a method named name ends a handle's life. It
// admits the forms that take arguments, such as FreeAsync.
func isRelease(name string) bool {
	return isDestructorName(name) || strings.HasPrefix(name, "Free")
}

func isError(t types.Type) bool {
	return types.Identical(t, types.Universe.Lookup("error").Type())
}

// released reports whether name's lifetime ends somewhere the analyzer can
// account for. A handle that is returned, stored in a struct or passed on has
// been handed to an owner the analyzer cannot follow, and is not reported.
// Only a variable whose every remaining use is a method call on the handle
// itself, and never a release, is a leak.
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
			// A method call on the handle. Only a release ends its life.
			if parent.X == id && isRelease(parent.Sel.Name) {
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
