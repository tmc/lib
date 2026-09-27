package main

import (
	"go/ast"
	"go/token"
	"go/types"
	"regexp"
	"strings"

	"golang.org/x/tools/go/analysis"
)

var errCompareAnalyzer = &analysis.Analyzer{
	Name: "errcompare",
	Doc: `check that binding errors are matched with errors.Is

A binding that fails returns a *CallError wrapping the library's result code,
so err == driver.ErrInvalidValue is never true. errcompare reports == and !=
comparisons, and switch cases, of an error against a result constant of a
package that declares CallError.`,
	Run: runErrCompare,
}

// isResultCode reports whether t is an error type of an nvidia package whose
// failures are returned wrapped in a *CallError.
func isResultCode(t types.Type) bool {
	n, ok := types.Unalias(t).(*types.Named)
	if !ok || !inNvidia(n.Obj().Pkg()) || n.Obj().Pkg().Scope().Lookup("CallError") == nil {
		return false
	}
	return types.Implements(n, types.Universe.Lookup("error").Type().Underlying().(*types.Interface))
}

func isErrorIface(t types.Type) bool {
	return t != nil && types.Identical(t, types.Universe.Lookup("error").Type())
}

func runErrCompare(pass *analysis.Pass) (any, error) {
	info := pass.TypesInfo
	report := func(e ast.Expr) {
		pass.Reportf(e.Pos(), "a failed call returns a *CallError, which never equals %s; use errors.Is", types.ExprString(e))
	}
	for _, file := range srcFiles(pass) {
		ast.Inspect(file, func(n ast.Node) bool {
			switch n := n.(type) {
			case *ast.BinaryExpr:
				if n.Op != token.EQL && n.Op != token.NEQ {
					return true
				}
				switch {
				case isErrorIface(info.TypeOf(n.X)) && isResultCode(info.TypeOf(n.Y)):
					report(n.Y)
				case isErrorIface(info.TypeOf(n.Y)) && isResultCode(info.TypeOf(n.X)):
					report(n.X)
				}
			case *ast.SwitchStmt:
				if n.Tag == nil || !isErrorIface(info.TypeOf(n.Tag)) {
					return true
				}
				for _, s := range n.Body.List {
					for _, e := range s.(*ast.CaseClause).List {
						if isResultCode(info.TypeOf(e)) {
							report(e)
						}
					}
				}
			}
			return true
		})
	}
	return nil, nil
}

var uncheckedSyncAnalyzer = &analysis.Analyzer{
	Name: "uncheckedsync",
	Doc: `check that errors of synchronizing and release calls are not discarded

A failure of an earlier kernel launch or asynchronous copy is reported by a
later call, usually the next one that synchronizes; a release may report it
too. uncheckedsync reports a Synchronize or CtxSynchronize called as a bare
statement, dropping its error, and a release (Close, Destroy, Free, Unload)
called the same way outside a defer. An explicit _ = x.Close() is taken as
deliberate. A release on an error path, followed by a return of a non-nil
error or a call such as t.Fatal, is left alone: the first error is the one
to report.`,
	Run: runUncheckedSync,
}

// eachBareCall calls f for every call of an nvidia function made as a bare
// statement, with the stack of the call's ancestors. An assignment such as
// _ = x.Close() discards the error on purpose and is not a bare call.
func eachBareCall(pass *analysis.Pass, f func(call *ast.CallExpr, fn *types.Func, stack []ast.Node)) {
	info := pass.TypesInfo
	eachFunc(pass, func(body *ast.BlockStmt) {
		withStack(body, func(n ast.Node, stack []ast.Node) {
			call, ok := n.(*ast.CallExpr)
			if !ok || len(stack) == 0 {
				return
			}
			if _, bare := stack[len(stack)-1].(*ast.ExprStmt); !bare {
				return
			}
			fn := callee(info, call)
			if fn == nil || !inNvidia(fn.Pkg()) {
				return
			}
			f(call, fn, stack)
		})
	})
}

func runUncheckedSync(pass *analysis.Pass) (any, error) {
	info := pass.TypesInfo
	eachBareCall(pass, func(call *ast.CallExpr, fn *types.Func, stack []ast.Node) {
		switch {
		case strings.HasPrefix(fn.Name(), "Synchronize") || fn.Name() == "CtxSynchronize":
			pass.Reportf(call.Pos(), "the error of %s is discarded; it is where failures of earlier asynchronous work are reported", fn.Name())
		case recvNamed(fn) != nil && isRelease(fn.Name()) && destructor(recvNamed(fn)) != "":
			if !onErrorPath(info, stack) {
				pass.Reportf(call.Pos(), "the error of %s is discarded; it can report a failure of earlier asynchronous work", fn.Name())
			}
		}
	})
	return nil, nil
}

var setterErrAnalyzer = &analysis.Analyzer{
	Name: "settererr",
	Doc: `check that errors of setters are not discarded

A setter that rejects its value returns an error and keeps the old value, and
reading the setting back returns the old value too, so a program that drops
the error runs on the old setting without a sign. settererr reports a call of
a generated binding's function or method returning an error, called as a bare
statement, whose name contains Set followed by an upper-case letter: SetLimit,
CtxSetLimit, FuncSetAttribute, SetCacheConfig. Settle and Setup are not
setters, nor is Reset. Packages under x/, whose Set methods are often Go-side
validators, are not checked. An explicit _ = f.SetAttribute(a, v) is taken as
deliberate.`,
	Run: runSetterErr,
}

var setterRE = regexp.MustCompile(`Set[A-Z]`)

func runSetterErr(pass *analysis.Pass) (any, error) {
	eachBareCall(pass, func(call *ast.CallExpr, fn *types.Func, stack []ast.Node) {
		res := fn.Type().(*types.Signature).Results()
		if !isBinding(fn.Pkg()) || !setterRE.MatchString(fn.Name()) || res.Len() == 0 || !isErrorIface(res.At(res.Len()-1).Type()) {
			return
		}
		pass.Reportf(call.Pos(), "the error of %s is discarded; a rejected value leaves the old setting in place", fn.Name())
	})
	return nil, nil
}

// onErrorPath reports whether the statement at the top of stack is followed,
// in its block, by a return of a non-nil error or a call that does not
// return, such as t.Fatal.
func onErrorPath(info *types.Info, stack []ast.Node) bool {
	if len(stack) < 2 {
		return false
	}
	stmt, ok := stack[len(stack)-1].(ast.Stmt)
	if !ok {
		return false
	}
	var list []ast.Stmt
	switch b := stack[len(stack)-2].(type) {
	case *ast.BlockStmt:
		list = b.List
	case *ast.CaseClause:
		list = b.Body
	default:
		return false
	}
	for i, s := range list {
		if s != stmt {
			continue
		}
		for _, next := range list[i+1:] {
			if es, ok := next.(*ast.ExprStmt); ok {
				if c, ok := es.X.(*ast.CallExpr); ok && noReturn(info, c) {
					return true
				}
			}
			ret, ok := next.(*ast.ReturnStmt)
			if !ok {
				continue
			}
			if len(ret.Results) == 0 {
				return false
			}
			id, ok := ret.Results[len(ret.Results)-1].(*ast.Ident)
			return !ok || id.Name != "nil"
		}
	}
	return false
}

var devInfoAnalyzer = &analysis.Analyzer{
	Name: "devinfo",
	Doc: `check that cuSOLVER's devInfo result is copied back and checked

A cuSOLVER routine reports a bad parameter (info = -i) or a singular pivot
(info > 0 from getrf) in a device buffer, not in its error. devinfo reports a
devInfo or info buffer passed to cuSOLVER that the function never copies back
to the host or hands to anything outside cuSOLVER.`,
	Run: runDevInfo,
}

func runDevInfo(pass *analysis.Pass) (any, error) {
	info := pass.TypesInfo
	eachFunc(pass, func(body *ast.BlockStmt) {
		for _, call := range calls(body) {
			fn := callee(info, call)
			if fn == nil || !inPkg(fn.Pkg(), "cusolver") {
				continue
			}
			sig := fn.Type().(*types.Signature)
			for i := range sig.Params().Len() {
				p := sig.Params().At(i)
				if (p.Name() != "devInfo" && p.Name() != "info") || !isDevicePtr(p.Type()) || i >= len(call.Args) {
					continue
				}
				v := objOf(info, call.Args[i])
				if v != nil && !infoRead(info, body, v, call) {
					pass.Reportf(call.Args[i].Pos(), "%s from %s is never copied back and checked; a negative value is a bad parameter and a positive one a failure such as a singular pivot",
						v.Name(), fn.Name())
				}
			}
		}
	})
	return nil, nil
}

// infoRead reports whether the info buffer v is read after call: copied to
// the host, or passed to something other than cuSOLVER or a release.
func infoRead(info *types.Info, body *ast.BlockStmt, v *types.Var, after *ast.CallExpr) bool {
	for _, c := range calls(body) {
		if c.Pos() <= after.Pos() {
			continue
		}
		fn := callee(info, c)
		if sel, ok := ast.Unparen(c.Fun).(*ast.SelectorExpr); ok && objOf(info, sel.X) == v {
			if fn != nil && strings.HasPrefix(fn.Name(), "CopyToHost") {
				return true
			}
			continue
		}
		if fn != nil && inPkg(fn.Pkg(), "cusolver") {
			continue
		}
		for _, a := range c.Args {
			if uses(info, a, v) {
				return true
			}
		}
	}
	// A deferred or closure read, or a return of the buffer, also counts.
	return escapes(info, body, v)
}

var zeroVersionAnalyzer = &analysis.Analyzer{
	Name: "zeroversion",
	Doc: `check that a GetVersion with no error result is not trusted blindly

A GetVersion that returns only a number returns 0 when the library is not
loaded. zeroversion reports such a version compared or printed, where the 0
reads as a version, before the function has compared it with 0. Bindings
that return (version, error) are not reported.`,
	Run: runZeroVersion,
}

func runZeroVersion(pass *analysis.Pass) (any, error) {
	info := pass.TypesInfo
	eachFunc(pass, func(body *ast.BlockStmt) {
		versions := make(map[*types.Var]*types.Func)
		checked := make(map[*types.Var]bool) // compared with 0 already
		isVersion := func(e ast.Expr) *types.Func {
			e = unconvert(info, e)
			if call, ok := e.(*ast.CallExpr); ok {
				if fn := callee(info, call); fn != nil && fn.Name() == "GetVersion" && inNvidia(fn.Pkg()) && fn.Type().(*types.Signature).Results().Len() == 1 {
					return fn
				}
				return nil
			}
			if v := objOf(info, e); !checked[v] {
				return versions[v]
			}
			return nil
		}
		withStack(body, func(n ast.Node, stack []ast.Node) {
			switch n := n.(type) {
			case *ast.AssignStmt:
				if len(n.Lhs) == len(n.Rhs) {
					for i, r := range n.Rhs {
						if fn := isVersion(r); fn != nil {
							if v := objOf(info, n.Lhs[i]); v != nil {
								versions[v] = fn
							}
						}
					}
				}
			case *ast.BinaryExpr:
				switch n.Op {
				case token.EQL, token.NEQ, token.LSS, token.GTR, token.LEQ, token.GEQ:
					for _, e := range []ast.Expr{n.X, n.Y} {
						if fn := isVersion(e); fn != nil {
							if z, ok := intConst(info, otherSide(n, e)); ok && z == 0 {
								if v := objOf(info, e); v != nil {
									checked[v] = true
								}
								continue // the not-loaded check itself
							}
							pass.Reportf(e.Pos(), "%s.GetVersion returns 0 when the library is not loaded; check for 0 before comparing it as a version", fn.Pkg().Name())
						}
					}
				}
			case *ast.CallExpr:
				fn := callee(info, n)
				if fn == nil || fn.Pkg() == nil || (fn.Pkg().Path() != "fmt" && fn.Pkg().Path() != "log") {
					return
				}
				for _, a := range n.Args {
					if v := isVersion(a); v != nil {
						pass.Reportf(a.Pos(), "%s.GetVersion returns 0 when the library is not loaded; a printed 0 is not a version", v.Pkg().Name())
					}
				}
			}
		})
	})
	return nil, nil
}

func otherSide(b *ast.BinaryExpr, e ast.Expr) ast.Expr {
	if b.X == e {
		return b.Y
	}
	return b.X
}

// legacyFact records that a function always fails since CUDA 13.0, and what
// replaces it, as its doc comment says.
type legacyFact struct{ Use string }

func (*legacyFact) AFact()           {}
func (f *legacyFact) String() string { return "legacy: " + f.Use }

var legacyCuptiAnalyzer = &analysis.Analyzer{
	Name: "legacycupti",
	Doc: `report calls to CUPTI functions that always fail since CUDA 13.0

The legacy Event and Metric APIs return ErrLegacyProfilerNotSupported on every
call since CUDA 13.0. legacycupti reads that from each function's doc comment,
records it as a fact, and reports calls in other packages with the
replacement the doc names.`,
	Run:       runLegacyCupti,
	FactTypes: []analysis.Fact{new(legacyFact)},
}

var legacyDocRE = regexp.MustCompile(`always returns ErrLegacyProfilerNotSupported; (use .*?, starting with \w+)`)

func runLegacyCupti(pass *analysis.Pass) (any, error) {
	info := pass.TypesInfo
	for _, file := range pass.Files {
		for _, decl := range file.Decls {
			fd, ok := decl.(*ast.FuncDecl)
			if !ok || fd.Doc == nil {
				continue
			}
			doc := strings.Join(strings.Fields(fd.Doc.Text()), " ")
			if m := legacyDocRE.FindStringSubmatch(doc); m != nil {
				if fn, ok := info.Defs[fd.Name].(*types.Func); ok {
					pass.ExportObjectFact(fn, &legacyFact{Use: m[1]})
				}
			}
		}
	}
	eachFunc(pass, func(body *ast.BlockStmt) {
		for _, call := range calls(body) {
			fn := callee(info, call)
			var f legacyFact
			if fn == nil || fn.Pkg() == pass.Pkg || !pass.ImportObjectFact(fn, &f) {
				continue
			}
			pass.Reportf(call.Pos(), "%s always returns ErrLegacyProfilerNotSupported since CUDA 13.0; %s", fn.Name(), f.Use)
		}
	})
	return nil, nil
}
