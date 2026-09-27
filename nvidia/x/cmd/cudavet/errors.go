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
package that declares CallError.

A cuFile call that fails with -1 returns an *ErrnoError wrapping the errno,
so err == syscall.ENOENT and err == fs.ErrNotExist are never true either. In
a package that imports cufile, errcompare also reports comparisons of an error
against a syscall.Errno or an io/fs sentinel.`,
	Run: runErrCompare,
}

// isErrnoSentinel reports whether e is a syscall.Errno or one of the io/fs
// sentinels an Errno matches through errors.Is, also spelled os.ErrNotExist
// and the like.
func isErrnoSentinel(info *types.Info, e ast.Expr) bool {
	if t, ok := types.Unalias(info.TypeOf(e)).(*types.Named); ok && t.Obj().Pkg() != nil && t.Obj().Pkg().Path() == "syscall" && t.Obj().Name() == "Errno" {
		return true
	}
	var id *ast.Ident
	switch e := ast.Unparen(e).(type) {
	case *ast.Ident:
		id = e
	case *ast.SelectorExpr:
		id = e.Sel
	default:
		return false
	}
	v, ok := info.Uses[id].(*types.Var)
	if !ok || v.Pkg() == nil {
		return false
	}
	switch v.Pkg().Path() {
	case "io/fs":
		return strings.HasPrefix(v.Name(), "Err")
	case "os":
		switch v.Name() {
		case "ErrInvalid", "ErrPermission", "ErrExist", "ErrNotExist", "ErrClosed":
			return true
		}
	}
	return false
}

// importsCufile reports whether pkg imports the cufile binding.
func importsCufile(pkg *types.Package) bool {
	for _, p := range pkg.Imports() {
		if inPkg(p, "cufile") {
			return true
		}
	}
	return false
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
	cufile := importsCufile(pass.Pkg)
	// check reports e, compared with an error, if it cannot equal one.
	check := func(e ast.Expr) {
		switch {
		case isResultCode(info.TypeOf(e)):
			report(e)
		case cufile && isErrnoSentinel(info, e):
			pass.Reportf(e.Pos(), "a failed cuFile call returns an *ErrnoError, which never equals %s; use errors.Is", types.ExprString(e))
		}
	}
	for _, file := range srcFiles(pass) {
		ast.Inspect(file, func(n ast.Node) bool {
			switch n := n.(type) {
			case *ast.BinaryExpr:
				if n.Op != token.EQL && n.Op != token.NEQ {
					return true
				}
				if isErrorIface(info.TypeOf(n.X)) {
					check(n.Y)
				}
				if isErrorIface(info.TypeOf(n.Y)) {
					check(n.X)
				}
			case *ast.SwitchStmt:
				if n.Tag == nil || !isErrorIface(info.TypeOf(n.Tag)) {
					return true
				}
				for _, s := range n.Body.List {
					for _, e := range s.(*ast.CaseClause).List {
						check(e)
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
called the same way outside a defer. A function literal that is deferred or
registered with testing's Cleanup counts as a defer. An explicit
_ = x.Close() is taken as deliberate. A release on an error path, followed
by a return of a non-nil error or a call such as t.Fatal, is left alone: the
first error is the one to report.`,
	Run: runUncheckedSync,
}

// eachBareCall calls f for every call of an nvidia function made as a bare
// statement, with the body of the enclosing function and the stack of the
// call's ancestors. An assignment such as _ = x.Close() discards the error on
// purpose and is not a bare call.
func eachBareCall(pass *analysis.Pass, f func(call *ast.CallExpr, fn *types.Func, body *ast.BlockStmt, stack []ast.Node)) {
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
			f(call, fn, body, stack)
		})
	})
}

func runUncheckedSync(pass *analysis.Pass) (any, error) {
	info := pass.TypesInfo
	cleanup := cleanupBodies(pass)
	eachBareCall(pass, func(call *ast.CallExpr, fn *types.Func, body *ast.BlockStmt, stack []ast.Node) {
		switch {
		case strings.HasPrefix(fn.Name(), "Synchronize") || fn.Name() == "CtxSynchronize":
			pass.Reportf(call.Pos(), "the error of %s is discarded; it is where failures of earlier asynchronous work are reported", fn.Name())
		case recvNamed(fn) != nil && isRelease(fn.Name()) && destructor(recvNamed(fn)) != "":
			if !cleanup[body] && !onErrorPath(info, stack) {
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
	eachBareCall(pass, func(call *ast.CallExpr, fn *types.Func, _ *ast.BlockStmt, _ []ast.Node) {
		res := fn.Type().(*types.Signature).Results()
		if !isBinding(fn.Pkg()) || !setterRE.MatchString(fn.Name()) || res.Len() == 0 || !isErrorIface(res.At(res.Len()-1).Type()) {
			return
		}
		pass.Reportf(call.Pos(), "the error of %s is discarded; a rejected value leaves the old setting in place", fn.Name())
	})
	return nil, nil
}

// cleanupBodies returns the bodies of the function literals that run as
// cleanups: deferred, or passed to the Cleanup method of a testing.T, B or F.
func cleanupBodies(pass *analysis.Pass) map[*ast.BlockStmt]bool {
	bodies := make(map[*ast.BlockStmt]bool)
	for _, file := range srcFiles(pass) {
		ast.Inspect(file, func(n ast.Node) bool {
			switch n := n.(type) {
			case *ast.DeferStmt:
				if lit, ok := n.Call.Fun.(*ast.FuncLit); ok {
					bodies[lit.Body] = true
				}
			case *ast.CallExpr:
				if len(n.Args) == 1 && isPkgFunc(pass.TypesInfo, n, "testing", "Cleanup") {
					if lit, ok := n.Args[0].(*ast.FuncLit); ok {
						bodies[lit.Body] = true
					}
				}
			}
			return true
		})
	}
	return bodies
}

// onErrorPath reports whether the statement at the top of stack is followed,
// in its block, by a return of a non-nil error or a call that does not
// return, such as t.Fatal. When the block is the body of an if statement and
// ends first, the statements after the if statement are looked at in turn.
func onErrorPath(info *types.Info, stack []ast.Node) bool {
	for len(stack) >= 2 {
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
				switch last := ret.Results[len(ret.Results)-1].(type) {
				case *ast.Ident:
					return last.Name != "nil"
				case *ast.CallExpr:
					// return s.Synchronize() returns the result of
					// more work, not an error already in hand.
					return isPkgFunc(info, last, "errors", "New") || isPkgFunc(info, last, "fmt", "Errorf")
				}
				return true
			}
		}
		// The block ran out. Carry on after an if statement around it;
		// anything else, such as a loop body, may go around again.
		if len(stack) < 3 {
			return false
		}
		if _, ok := stack[len(stack)-3].(*ast.IfStmt); !ok {
			return false
		}
		stack = stack[:len(stack)-2]
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
	Doc: `check that a version is not used when its error is discarded

A GetVersion, ApiVersion or other version query returns (version, error), and
the version is 0 or empty when the library is not loaded or the call fails.
zeroversion reports a version query whose error is assigned to _ and whose
version is then used, where the zero reads as a version.`,
	Run: runZeroVersion,
}

func runZeroVersion(pass *analysis.Pass) (any, error) {
	info := pass.TypesInfo
	eachFunc(pass, func(body *ast.BlockStmt) {
		inspectShallow(body, func(n ast.Node) bool {
			assign, ok := n.(*ast.AssignStmt)
			if !ok || len(assign.Lhs) != 2 || len(assign.Rhs) != 1 {
				return true
			}
			call, ok := ast.Unparen(assign.Rhs[0]).(*ast.CallExpr)
			if !ok {
				return true
			}
			fn := callee(info, call)
			if fn == nil || !inNvidia(fn.Pkg()) || !isVersionQuery(fn) {
				return true
			}
			if blank, ok := assign.Lhs[1].(*ast.Ident); !ok || blank.Name != "_" {
				return true
			}
			id, ok := assign.Lhs[0].(*ast.Ident)
			if !ok {
				return true
			}
			v := objOf(info, id)
			if v == nil {
				return true
			}
			var use *ast.Ident
			ast.Inspect(body, func(n ast.Node) bool {
				if x, ok := n.(*ast.Ident); ok && use == nil && x.Pos() > assign.End() && info.Uses[x] == v {
					use = x
				}
				return use == nil
			})
			if use != nil {
				pass.Reportf(assign.Lhs[1].Pos(), "the error of %s is discarded, but %s is used; when the call fails it is the zero value, not a version", fn.Name(), id.Name)
			}
			return true
		})
	})
	return nil, nil
}

// isVersionQuery reports whether fn is named for a version and returns a
// basic-typed version and an error.
func isVersionQuery(fn *types.Func) bool {
	if !strings.Contains(fn.Name(), "Version") {
		return false
	}
	res := fn.Type().(*types.Signature).Results()
	if res.Len() != 2 || !isError(res.At(1).Type()) {
		return false
	}
	_, ok := res.At(0).Type().Underlying().(*types.Basic)
	return ok
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
			// Doc links such as [RangeProfilerEnable] read as plain names.
			doc := strings.Join(strings.Fields(fd.Doc.Text()), " ")
			doc = strings.NewReplacer("[", "", "]", "").Replace(doc)
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
