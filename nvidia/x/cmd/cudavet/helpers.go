package main

import (
	"go/ast"
	"go/constant"
	"go/printer"
	"go/token"
	"go/types"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/types/typeutil"
)

// nvidiaPrefix is the import path prefix of the generated bindings and the
// packages built on them.
const nvidiaPrefix = "github.com/tmc/lib/nvidia/"

// inNvidia reports whether pkg is one of the nvidia packages.
func inNvidia(pkg *types.Package) bool {
	return pkg != nil && strings.HasPrefix(pkg.Path(), nvidiaPrefix)
}

// isBinding reports whether pkg is a generated binding of a C library, as
// opposed to a package under x/ built on the bindings.
func isBinding(pkg *types.Package) bool {
	return inNvidia(pkg) && !strings.HasPrefix(pkg.Path(), nvidiaPrefix+"x/")
}

// inPkg reports whether pkg is the nvidia package with the given path below
// nvidiaPrefix, such as "driver" or "cusolver".
func inPkg(pkg *types.Package, name string) bool {
	return pkg != nil && pkg.Path() == nvidiaPrefix+name
}

// callee returns the function or method call invokes, or nil. A call of an
// instantiated generic function returns the generic function.
func callee(info *types.Info, call *ast.CallExpr) *types.Func {
	fn, _ := typeutil.Callee(info, call).(*types.Func)
	if fn != nil {
		fn = fn.Origin()
	}
	return fn
}

// calleeIn returns the name of the function call invokes if it belongs to
// the nvidia package name, and "" otherwise.
func calleeIn(info *types.Info, call *ast.CallExpr, name string) string {
	fn := callee(info, call)
	if fn == nil || !inPkg(fn.Pkg(), name) {
		return ""
	}
	return fn.Name()
}

// recvNamed returns the named receiver type of a method, or nil.
func recvNamed(fn *types.Func) *types.Named {
	sig, ok := fn.Type().(*types.Signature)
	if !ok || sig.Recv() == nil {
		return nil
	}
	t := sig.Recv().Type()
	if p, ok := t.(*types.Pointer); ok {
		t = p.Elem()
	}
	n, _ := t.(*types.Named)
	return n
}

// methodOf reports whether fn is the method name of type typ in the nvidia
// package pkg.
func methodOf(fn *types.Func, pkg, typ, name string) bool {
	if fn == nil || fn.Name() != name {
		return false
	}
	n := recvNamed(fn)
	return n != nil && n.Obj().Name() == typ && inPkg(n.Obj().Pkg(), pkg)
}

// isNamed reports whether t is the named type typ of the nvidia package pkg.
func isNamed(t types.Type, pkg, typ string) bool {
	n, ok := types.Unalias(t).(*types.Named)
	return ok && n.Obj().Name() == typ && inPkg(n.Obj().Pkg(), pkg)
}

// isDevicePtr reports whether t is a DevicePtr of some nvidia package.
func isDevicePtr(t types.Type) bool {
	n, ok := types.Unalias(t).(*types.Named)
	return ok && n.Obj().Name() == "DevicePtr" && inNvidia(n.Obj().Pkg())
}

// srcFiles returns the files of the package that were not generated. The
// checks look at code calling the bindings; generated code is checked by the
// generator's own tests.
func srcFiles(pass *analysis.Pass) []*ast.File {
	var out []*ast.File
	for _, f := range pass.Files {
		if !ast.IsGenerated(f) {
			out = append(out, f)
		}
	}
	return out
}

// inGenerated reports whether pos lies in a generated file of the package.
func inGenerated(pass *analysis.Pass, pos token.Pos) bool {
	for _, f := range pass.Files {
		if f.FileStart <= pos && pos <= f.FileEnd {
			return ast.IsGenerated(f)
		}
	}
	return false
}

// eachFunc calls f with the body of every function declaration and function
// literal in the package. Checks that follow one function at a time use
// inspectShallow over the body so that a nested literal is its own function.
func eachFunc(pass *analysis.Pass, f func(body *ast.BlockStmt)) {
	for _, file := range srcFiles(pass) {
		ast.Inspect(file, func(n ast.Node) bool {
			switch n := n.(type) {
			case *ast.FuncDecl:
				if n.Body != nil {
					f(n.Body)
				}
			case *ast.FuncLit:
				f(n.Body)
			}
			return true
		})
	}
}

// eachOuterFunc calls f with the body of each function that is not inside
// another. The body includes the function literals within it, which share its
// variables.
func eachOuterFunc(pass *analysis.Pass, f func(body *ast.BlockStmt)) {
	for _, file := range srcFiles(pass) {
		ast.Inspect(file, func(n ast.Node) bool {
			switch n := n.(type) {
			case *ast.FuncDecl:
				if n.Body != nil {
					f(n.Body)
				}
				return false
			case *ast.FuncLit:
				f(n.Body)
				return false
			}
			return true
		})
	}
}

// inspectShallow is ast.Inspect that does not descend into function literals.
func inspectShallow(root ast.Node, f func(ast.Node) bool) {
	ast.Inspect(root, func(n ast.Node) bool {
		if _, ok := n.(*ast.FuncLit); ok && n != root {
			return false
		}
		return f(n)
	})
}

// calls returns the calls in body, outside nested function literals, in
// source order.
func calls(body ast.Node) []*ast.CallExpr {
	var out []*ast.CallExpr
	inspectShallow(body, func(n ast.Node) bool {
		if c, ok := n.(*ast.CallExpr); ok {
			out = append(out, c)
		}
		return true
	})
	return out
}

// unconvert strips parentheses and type conversions, so that uint(n) and n
// name the same quantity.
func unconvert(info *types.Info, e ast.Expr) ast.Expr {
	for {
		e = ast.Unparen(e)
		call, ok := e.(*ast.CallExpr)
		if !ok || len(call.Args) != 1 {
			return e
		}
		tv, ok := info.Types[call.Fun]
		if !ok || !tv.IsType() {
			return e
		}
		e = call.Args[0]
	}
}

// objOf returns the variable e names, or nil.
func objOf(info *types.Info, e ast.Expr) *types.Var {
	id, ok := ast.Unparen(e).(*ast.Ident)
	if !ok {
		return nil
	}
	v, _ := info.ObjectOf(id).(*types.Var)
	return v
}

// uses reports whether n refers to v anywhere.
func uses(info *types.Info, n ast.Node, v *types.Var) bool {
	found := false
	ast.Inspect(n, func(n ast.Node) bool {
		if id, ok := n.(*ast.Ident); ok && info.ObjectOf(id) == v {
			found = true
		}
		return !found
	})
	return found
}

// intConst returns the value of a constant integer expression.
func intConst(info *types.Info, e ast.Expr) (int64, bool) {
	tv, ok := info.Types[e]
	if !ok || tv.Value == nil || tv.Value.Kind() != constant.Int {
		return 0, false
	}
	return constant.Int64Val(tv.Value)
}

// constString returns the value of a constant string expression.
func constString(info *types.Info, e ast.Expr) (string, bool) {
	tv, ok := info.Types[e]
	if !ok || tv.Value == nil || tv.Value.Kind() != constant.String {
		return "", false
	}
	return constant.StringVal(tv.Value), true
}

// isPkgFunc reports whether call invokes the function name of package path.
func isPkgFunc(info *types.Info, call *ast.CallExpr, path, name string) bool {
	fn := callee(info, call)
	return fn != nil && fn.Pkg() != nil && fn.Pkg().Path() == path && fn.Name() == name
}

// resultDiscarded reports whether the error a call returns is thrown away:
// the call is an expression statement, or every result is assigned to _.
func resultDiscarded(stack []ast.Node, call *ast.CallExpr) bool {
	if len(stack) == 0 {
		return false
	}
	switch p := stack[len(stack)-1].(type) {
	case *ast.ExprStmt:
		return true
	case *ast.AssignStmt:
		return len(p.Rhs) == 1 && p.Rhs[0] == ast.Expr(call) && allBlank(p.Lhs)
	}
	return false
}

// withStack walks root outside nested function literals, handing f each node
// with the stack of its ancestors.
func withStack(root ast.Node, f func(n ast.Node, stack []ast.Node)) {
	var stack []ast.Node
	ast.Inspect(root, func(n ast.Node) bool {
		if n == nil {
			stack = stack[:len(stack)-1]
			return true
		}
		if _, ok := n.(*ast.FuncLit); ok && n != root {
			return false
		}
		f(n, stack)
		stack = append(stack, n)
		return true
	})
}

// render returns the source form of e in full; types.ExprString elides the
// elements of composite literals.
func render(fset *token.FileSet, e ast.Expr) string {
	var b strings.Builder
	if err := printer.Fprint(&b, fset, e); err != nil {
		return types.ExprString(e)
	}
	return b.String()
}

// withStackAll is withStack descending into function literals too.
func withStackAll(root ast.Node, f func(n ast.Node, stack []ast.Node)) {
	var stack []ast.Node
	ast.Inspect(root, func(n ast.Node) bool {
		if n == nil {
			stack = stack[:len(stack)-1]
			return true
		}
		f(n, stack)
		stack = append(stack, n)
		return true
	})
}

// between reports whether pos lies strictly between a and b.
func between(pos, a, b token.Pos) bool { return a < pos && pos < b }

// noReturn reports whether call never returns: panic, os.Exit, log.Fatal
// and friends.
func noReturn(info *types.Info, call *ast.CallExpr) bool {
	if id, ok := call.Fun.(*ast.Ident); ok && id.Name == "panic" {
		return info.Uses[id] == types.Universe.Lookup("panic")
	}
	fn := callee(info, call)
	if fn == nil || fn.Pkg() == nil {
		return false
	}
	switch fn.Pkg().Path() + "." + fn.Name() {
	case "os.Exit", "log.Fatal", "log.Fatalf", "log.Fatalln", "log.Panic", "log.Panicf":
		return true
	}
	return fn.Pkg().Path() == "testing" && (fn.Name() == "Fatal" || fn.Name() == "Fatalf" || fn.Name() == "FailNow")
}

func isNil(info *types.Info, e ast.Expr) bool {
	e = unconvert(info, e)
	id, ok := e.(*ast.Ident)
	return ok && info.Uses[id] == types.Universe.Lookup("nil")
}

// declaredIn reports whether v is declared inside body, not as a parameter.
func declaredIn(v *types.Var, body *ast.BlockStmt) bool {
	return body.Pos() < v.Pos() && v.Pos() < body.End()
}

// escapes reports whether v leaves body: returned, stored, sent, or captured
// by a function literal.
func escapes(info *types.Info, body *ast.BlockStmt, v *types.Var) bool {
	found := false
	withStack(body, func(n ast.Node, stack []ast.Node) {
		id, ok := n.(*ast.Ident)
		if !ok || info.ObjectOf(id) != v || len(stack) == 0 {
			return
		}
		switch p := stack[len(stack)-1].(type) {
		case *ast.ReturnStmt, *ast.CompositeLit, *ast.KeyValueExpr, *ast.SendStmt:
			found = true
		case *ast.AssignStmt:
			for _, r := range p.Rhs {
				if r == ast.Expr(id) {
					found = true
				}
			}
		}
	})
	ast.Inspect(body, func(n ast.Node) bool {
		if lit, ok := n.(*ast.FuncLit); ok && uses(info, lit, v) {
			found = true
		}
		return !found
	})
	return found
}

// assignments returns every value assigned to each local variable of body,
// including by declaration. A variable assigned from a multi-value call is
// given a nil value, so that it has no static length.
func assignments(info *types.Info, body *ast.BlockStmt) map[*types.Var][]ast.Expr {
	defs := make(map[*types.Var][]ast.Expr)
	ast.Inspect(body, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.AssignStmt:
			for i, l := range n.Lhs {
				v := objOf(info, l)
				if v == nil {
					continue
				}
				var r ast.Expr
				if len(n.Lhs) == len(n.Rhs) && n.Tok != token.ADD_ASSIGN {
					r = n.Rhs[i]
				}
				defs[v] = append(defs[v], r)
			}
		case *ast.ValueSpec:
			for i, name := range n.Names {
				v, _ := info.Defs[name].(*types.Var)
				var r ast.Expr
				if len(n.Values) == len(n.Names) {
					r = n.Values[i]
				}
				defs[v] = append(defs[v], r)
			}
		case *ast.UnaryExpr:
			if n.Op == token.AND {
				if v := objOf(info, n.X); v != nil {
					defs[v] = append(defs[v], nil) // address taken: anything goes
				}
			}
		}
		return true
	})
	return defs
}
