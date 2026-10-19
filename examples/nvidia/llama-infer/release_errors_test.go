package main

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"slices"
	"strings"
	"testing"
)

// releaseNames are the methods whose error a cleanup path must report:
// the exported ones and the private release methods behind them
// (stepTimer.close, GPUModules.close, GPUScratchpad.release,
// deviceWeight.release, GPUModelWeights.free, and an unload method). The
// check matches names, not types, so a method by one of these names that
// returns no error is matched too and must be allowed below. The release
// functions those private methods take as arguments are called as
// function values, which errorFuncCalls finds instead.
var releaseNames = map[string]bool{
	"Close": true, "Free": true, "Destroy": true, "Synchronize": true, "CtxSynchronize": true,
	"Reset": true, "close": true, "release": true, "free": true, "unload": true,
}

// discardAllowed counts, by file and statement text, the statements that
// drop a release error on purpose.
var discardAllowed = map[string]int{
	// The L3 must-fail control in TestReleaseErrorOwner stands for a
	// caller that drops the close error; dropping it is what it tests.
	"module_ownership_test.go: c.Close()": 1,
	// The S3 and W1 must-fail controls in TestScratchpadRelease and
	// TestGPUModelWeightsFreeVectors stand for a Free that drops the
	// error of its private release; dropping it is what they test.
	"weights_bf16_test.go: sp.release(f32, i32)": 1,
	"weights_bf16_test.go: w.free(vec)":          1,
}

// discard is a statement that drops the error of a release call.
type discard struct {
	key string // file and statement text, as in discardAllowed
	pos string
}

// isReleaseCall reports whether e, ignoring parentheses, calls a method
// named in releaseNames or is one of the function value calls in byValue.
func isReleaseCall(e ast.Expr, byValue map[*ast.CallExpr]bool) bool {
	call, ok := ast.Unparen(e).(*ast.CallExpr)
	if !ok {
		return false
	}
	if byValue[call] {
		return true
	}
	sel, ok := call.Fun.(*ast.SelectorExpr)
	return ok && releaseNames[sel.Sel.Name]
}

// returnsError reports whether typ is a function type whose last result
// is error, or names such a type in funcTypes.
func returnsError(typ ast.Expr, funcTypes map[string]bool) bool {
	if id, ok := typ.(*ast.Ident); ok {
		return funcTypes[id.Name]
	}
	ft, ok := typ.(*ast.FuncType)
	if !ok || ft.Results == nil || len(ft.Results.List) == 0 {
		return false
	}
	id, ok := ft.Results.List[len(ft.Results.List)-1].Type.(*ast.Ident)
	return ok && id.Name == "error"
}

// errorFuncCalls returns the calls in f whose callee is a bare identifier
// or struct field selector naming a function value that returns an error:
// a parameter of an enclosing function or function literal, a struct field
// declared in f with such a type, or a variable the enclosing function
// declares with such a type or assigns from a literal or another such
// function value. The release functions passed to GPUScratchpad.release,
// GPUModelWeights.free, GPUModules.close and stepTimer.build are called
// this way. Types are read from the syntax, not type-checked: a named
// function type counts only if f declares it at top level, a struct field
// counts by its name across f, and a local counts throughout the function
// that declares or assigns it, so the result errs toward including a call.
func errorFuncCalls(f *ast.File) map[*ast.CallExpr]bool {
	funcTypes := make(map[string]bool)
	for _, d := range f.Decls {
		if gd, ok := d.(*ast.GenDecl); ok && gd.Tok == token.TYPE {
			for _, s := range gd.Specs {
				if ts := s.(*ast.TypeSpec); returnsError(ts.Type, nil) {
					funcTypes[ts.Name.Name] = true
				}
			}
		}
	}
	funcFields := make(map[string]bool)
	ast.Inspect(f, func(n ast.Node) bool {
		st, ok := n.(*ast.StructType)
		if !ok {
			return true
		}
		for _, field := range st.Fields.List {
			if returnsError(field.Type, funcTypes) {
				for _, id := range field.Names {
					funcFields[id.Name] = true
				}
			}
		}
		return true
	})
	calls := make(map[*ast.CallExpr]bool)
	var walk func(n ast.Node, names map[string]bool)
	walk = func(n ast.Node, names map[string]bool) {
		ast.Inspect(n, func(n ast.Node) bool {
			var typ *ast.FuncType
			var body *ast.BlockStmt
			switch n := n.(type) {
			case *ast.CallExpr:
				switch fn := n.Fun.(type) {
				case *ast.Ident:
					if names[fn.Name] {
						calls[n] = true
					}
				case *ast.SelectorExpr:
					if funcFields[fn.Sel.Name] {
						calls[n] = true
					}
				}
				return true
			case *ast.FuncDecl:
				typ, body = n.Type, n.Body
			case *ast.FuncLit:
				typ, body = n.Type, n.Body
			default:
				return true
			}
			if body == nil {
				return false
			}
			inner := make(map[string]bool)
			for k, v := range names {
				inner[k] = v
			}
			for _, p := range typ.Params.List {
				for _, id := range p.Names {
					inner[id.Name] = returnsError(p.Type, funcTypes)
				}
			}
			isErrorValue := func(e ast.Expr) bool {
				switch e := e.(type) {
				case *ast.FuncLit:
					return returnsError(e.Type, funcTypes)
				case *ast.Ident:
					return inner[e.Name]
				case *ast.SelectorExpr:
					return funcFields[e.Sel.Name]
				}
				return false
			}
			ast.Inspect(body, func(n ast.Node) bool {
				switch s := n.(type) {
				case *ast.FuncLit:
					return false
				case *ast.AssignStmt:
					if (s.Tok != token.DEFINE && s.Tok != token.ASSIGN) || len(s.Lhs) != len(s.Rhs) {
						break
					}
					for i, l := range s.Lhs {
						if id, ok := l.(*ast.Ident); ok && isErrorValue(s.Rhs[i]) {
							inner[id.Name] = true
						}
					}
				case *ast.ValueSpec:
					for i, id := range s.Names {
						if s.Type != nil && returnsError(s.Type, funcTypes) || i < len(s.Values) && isErrorValue(s.Values[i]) {
							inner[id.Name] = true
						}
					}
				}
				return true
			})
			walk(body, inner)
			return false
		})
	}
	walk(f, nil)
	return calls
}

// discardedReleases parses src and returns the number of calls to a
// method named in releaseNames, the number of calls of an error-returning
// function value (see errorFuncCalls), and the statements that drop the
// error of either: a deferred or go statement calling it, an expression
// statement calling it, an assignment of its result to blanks only, an if
// statement whose initializer assigns its error to a variable the
// condition does not test, and an assignment of its error to a variable
// that a later statement in the same list assigns again before anything
// reads it (see overwrites).
func discardedReleases(name string, src []byte) (calls, valueCalls int, bad []discard, err error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, name, src, 0)
	if err != nil {
		return 0, 0, nil, err
	}
	byValue := errorFuncCalls(f)
	isRelease := func(e ast.Expr) bool { return isReleaseCall(e, byValue) }
	flag := func(s ast.Stmt) {
		text := string(src[fset.Position(s.Pos()).Offset:fset.Position(s.End()).Offset])
		bad = append(bad, discard{key: name + ": " + text, pos: fset.Position(s.Pos()).String()})
	}
	flagOverwritten := func(list []ast.Stmt) {
		for i, s := range list {
			as, ok := s.(*ast.AssignStmt)
			if !ok || len(as.Rhs) != 1 || !isRelease(as.Rhs[0]) {
				continue
			}
			if v, ok := as.Lhs[len(as.Lhs)-1].(*ast.Ident); ok && v.Name != "_" && overwrites(list[i+1:], v.Name) {
				flag(s)
			}
		}
	}
	ast.Inspect(f, func(n ast.Node) bool {
		switch s := n.(type) {
		case *ast.CallExpr:
			if byValue[s] {
				valueCalls++
			} else if isRelease(s) {
				calls++
			}
		case *ast.DeferStmt:
			if isRelease(s.Call) {
				flag(s)
			}
		case *ast.GoStmt:
			if isRelease(s.Call) {
				flag(s)
			}
		case *ast.ExprStmt:
			if isRelease(s.X) {
				flag(s)
			}
		case *ast.AssignStmt:
			if !slices.ContainsFunc(s.Lhs, func(e ast.Expr) bool {
				id, ok := e.(*ast.Ident)
				return !ok || id.Name != "_"
			}) && slices.ContainsFunc(s.Rhs, isRelease) {
				flag(s)
			}
		case *ast.BlockStmt:
			flagOverwritten(s.List)
		case *ast.CaseClause:
			flagOverwritten(s.Body)
		case *ast.CommClause:
			flagOverwritten(s.Body)
		case *ast.IfStmt:
			init, ok := s.Init.(*ast.AssignStmt)
			if !ok || len(init.Rhs) != 1 || !isRelease(init.Rhs[0]) {
				break
			}
			if v, ok := init.Lhs[len(init.Lhs)-1].(*ast.Ident); ok && v.Name != "_" && !mentions(s.Cond, v.Name) {
				flag(s)
			}
		}
		return true
	})
	return calls, valueCalls, bad, nil
}

// overwrites reports whether the first statement in list that mentions
// name assigns to it without reading it. A statement that mentions name in
// any other way, including a nested block that assigns it on only some
// paths, counts as a read, so overwrites errs toward false.
func overwrites(list []ast.Stmt, name string) bool {
	for _, s := range list {
		if !mentions(s, name) {
			continue
		}
		as, ok := s.(*ast.AssignStmt)
		if !ok || slices.ContainsFunc(as.Rhs, func(e ast.Expr) bool { return mentions(e, name) }) {
			return false
		}
		return slices.ContainsFunc(as.Lhs, func(e ast.Expr) bool {
			id, ok := e.(*ast.Ident)
			return ok && id.Name == name
		})
	}
	return false
}

// unallowed returns the discards allowed does not account for, and the
// allowed entries that no discard used up.
func unallowed(discards []discard, allowed map[string]int) (bad, stale []string) {
	left := make(map[string]int)
	for k, n := range allowed {
		left[k] = n
	}
	for _, d := range discards {
		if left[d.key] > 0 {
			left[d.key]--
			continue
		}
		bad = append(bad, fmt.Sprintf("%s: %s", d.pos, d.key))
	}
	for k, n := range left {
		if n > 0 {
			stale = append(stale, fmt.Sprintf("%s: %d allowed discards unused", k, n))
		}
	}
	slices.Sort(stale)
	return bad, stale
}

// TestReleaseErrorsReported checks every .go file in the package, whatever
// its build constraints, for a statement that drops the error of a call to
// a method named in releaseNames or of an error-returning function value.
// It is a source-shape check, not a behaviour test.
func TestReleaseErrorsReported(t *testing.T) {
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	var discards []discard
	perFile := make(map[string]int)
	perFileValue := make(map[string]int)
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".go") {
			continue
		}
		src, err := os.ReadFile(e.Name())
		if err != nil {
			t.Fatal(err)
		}
		calls, valueCalls, bad, err := discardedReleases(e.Name(), src)
		if err != nil {
			t.Fatal(err)
		}
		perFile[e.Name()] = calls
		perFileValue[e.Name()] = valueCalls
		discards = append(discards, bad...)
	}
	// The files holding the cleanup paths must have been read and must
	// contain release calls, or a clean result means nothing was looked at.
	for _, name := range []string{
		"main.go", "steptime.go", "model.go", "gpu_model.go", "decode_graph.go", "gen.go",
		"profile_cuda_test.go", "fused_parity_cuda_test.go", "bf16_parity_cuda_test.go", "graph_parity_cuda_test.go",
	} {
		if perFile[name] == 0 {
			t.Errorf("%s: no release calls found", name)
		}
	}
	// The release functions that GPUScratchpad.release,
	// GPUModelWeights.free, GPUModules.close, DecodeGraph.close and
	// stepTimer.build take must have been seen as function value calls.
	for _, name := range []string{"gpu_model.go", "gpu_kernels.go", "decode_graph.go", "steptime.go"} {
		if perFileValue[name] == 0 {
			t.Errorf("%s: no function value calls found", name)
		}
	}
	bad, stale := unallowed(discards, discardAllowed)
	for _, b := range bad {
		t.Errorf("release error dropped: %s", b)
	}
	for _, s := range stale {
		t.Errorf("stale allowance: %s", s)
	}
}

// TestDiscardedReleasesControls checks discardedReleases on small sources:
// forms that report the error, and must-fail forms that each drop it.
func TestDiscardedReleasesControls(t *testing.T) {
	for _, tt := range []struct {
		name  string
		body  string
		calls int
		bad   bool
	}{
		{"positive control: deferred and checked", `
		defer func() {
			if err := m.Close(); err != nil {
				fmt.Fprintln(os.Stderr, err)
			}
		}()`, 1, false},
		{"positive control: t.Cleanup reporting", `
		t.Cleanup(func() {
			if err := m.Close(); err != nil {
				t.Error(err)
			}
		})`, 1, false},
		{"positive control: joined", `
		return errors.Join(err, st.close())`, 1, false},
		{"positive control: assigned", `
		err = s.Free()`, 1, false},
		{"positive control: tested in the condition", `
		if s.Free() == nil {
			*s = x{}
		}`, 1, false},
		{"positive control: builtin close", `
		close(ch)`, 0, false},
		{"positive control: other method", `
		defer f.Sync()`, 0, false},
		{"positive control: assigned, then joined", `
		err = s.Free()
		err = errors.Join(err, m.Close())`, 2, false},
		{"positive control: assigned, tested, then reassigned", `
		err := s.Free()
		if err != nil {
			return err
		}
		err = m.Close()
		return err`, 2, false},
		{"positive control: assigned, read in a nested block, then reassigned", `
		err := s.Free()
		if x {
			log(err)
		}
		err = g()
		return err`, 1, false},
		{"positive control: assigned to another variable", `
		cerr := s.Free()
		err = g()
		return errors.Join(err, cerr)`, 1, false},
		{"bare defer", `
		defer m.Close()`, 1, true},
		{"expression statement", `
		f.Close()`, 1, true},
		{"parenthesized expression statement", `
		(f.Close())`, 1, true},
		{"blank assignment", `
		_ = st.stream.Synchronize(ctx)`, 1, true},
		{"blank assignment in a deferred literal", `
		defer func() { _ = sp.Free() }()`, 1, true},
		{"dropped in t.Cleanup", `
		t.Cleanup(func() { attnMod.Close() })`, 1, true},
		{"deferred lowercase close", `
		defer timer.close()`, 1, true},
		{"package function", `
		driver.CtxSynchronize()`, 1, true},
		{"go statement", `
		go s.Free()`, 1, true},
		{"deferred private release", `
		defer sp.release(f32, i32)`, 1, true},
		{"private free dropped", `
		w.free(vec)`, 1, true},
		{"unload dropped", `
		f.unload(&m)`, 1, true},
		{"Reset dropped", `
		kv.Reset()`, 1, true},
		{"assigned, then overwritten", `
		err = s.Free()
		err = g()
		return err`, 1, true},
		{"declared, then overwritten after other statements", `
		err := sp.release(f32, i32)
		n := 1
		use(n)
		n, err = g()
		return err`, 1, true},
		{"overwritten in a case clause", `
		switch {
		case x:
			err = s.Free()
			err = g()
		}`, 1, true},
		{"overwritten in a select clause", `
		select {
		case <-ch:
			err = timer.close()
			err = g()
		}`, 1, true},
		{"if initializer not tested", `
		if waitErr := stream.Synchronize(ctx); err == nil {
			err = waitErr
		}`, 1, true},
	} {
		src := "package p\nfunc f() error {\n" + tt.body + "\nreturn nil\n}\n"
		calls, valueCalls, bad, err := discardedReleases("control.go", []byte(src))
		if err != nil {
			t.Fatalf("%s: %v", tt.name, err)
		}
		if calls != tt.calls || valueCalls != 0 {
			t.Errorf("%s: %d release calls and %d function value calls found, want %d and 0", tt.name, calls, valueCalls, tt.calls)
		}
		if got := len(bad) > 0; got != tt.bad {
			t.Errorf("%s: flagged %v, want %v (%v)", tt.name, got, tt.bad, bad)
		}
	}
}

// TestDiscardedFuncValuesControls checks discardedReleases on calls of
// function values, inside a function whose parameters include
// error-returning release functions and a struct holding them: forms that
// report the error or call a function value that returns none, and
// must-fail forms that each drop an error.
func TestDiscardedFuncValuesControls(t *testing.T) {
	for _, tt := range []struct {
		name   string
		body   string
		values int
		bad    bool
	}{
		{"positive control: appended", `
		errs = append(errs, f32(s))`, 1, false},
		{"positive control: tested", `
		if err := release(); err != nil {
			return err
		}`, 1, false},
		{"positive control: joined", `
		return errors.Join(vec(v), closer())`, 2, false},
		{"positive control: local alias tested", `
		g := vec
		if err := g(v); err != nil {
			return err
		}`, 1, false},
		{"positive control: struct field tested", `
		if err := h.fn(v); err != nil {
			return err
		}`, 1, false},
		{"positive control: no error result", `
		count()`, 0, false},
		{"positive control: named type with no error result", `
		n()`, 0, false},
		{"positive control: local literal with no error result", `
		fn := func() int { return 0 }
		fn()`, 0, false},
		{"positive control: local alias with no error result", `
		g := count
		g()`, 0, false},
		{"positive control: struct field with no error result", `
		h.count()
		h.n()`, 0, false},
		{"positive control: aliased struct field with no error result", `
		g := h.count
		g()`, 0, false},
		{"positive control: shadowed by a literal parameter", `
		use(func(f32 func()) { f32() })`, 0, false},
		{"positive control: package function", `
		g()`, 0, false},
		{"expression statement", `
		f32(&sp.Q)`, 1, true},
		{"blank assignment", `
		_ = vec(v)`, 1, true},
		{"bare defer", `
		defer release()`, 1, true},
		{"go statement", `
		go setup()`, 1, true},
		{"two results blanked", `
		_, _ = next(1)`, 1, true},
		{"two results dropped", `
		next(1)`, 1, true},
		{"named type", `
		closer()`, 1, true},
		{"in a deferred literal", `
		defer func() { setup() }()`, 1, true},
		{"local literal", `
		undo := func() error { return nil }
		undo()`, 1, true},
		{"local variable", `
		var undo func() error
		undo()`, 1, true},
		{"local alias expression statement", `
		g := vec
		g(v)`, 1, true},
		{"local alias blank assignment", `
		var g = f32
		_ = g(&sp.Q)`, 1, true},
		{"local alias assigned after declaration", `
		var g otherFunc
		g = vec
		g(v)`, 1, true},
		{"chained local alias", `
		g := vec
		h2 := g
		defer h2(v)`, 1, true},
		{"local literal assigned after declaration", `
		var g otherFunc
		g = func() error { return nil }
		g()`, 1, true},
		{"struct field expression statement", `
		h.fn(v)`, 1, true},
		{"struct field with named type", `
		_ = h.closer()`, 1, true},
		{"struct field deferred", `
		defer h.fn(v)`, 1, true},
		{"inline struct field", `
		s := struct{ vec func(*S) error }{vec}
		s.vec(v)`, 1, true},
		{"local alias of struct field", `
		g := h.fn
		g(v)`, 1, true},
		{"var alias of struct field", `
		var g = h.closer
		g()`, 1, true},
		{"literal parameter", `
		use(func(free func() error) { free() })`, 1, true},
		{"if initializer not tested", `
		if ferr := f32(s); err == nil {
			err = ferr
		}`, 1, true},
		{"assigned, then overwritten", `
		err = vec(v)
		err = g()
		return err`, 1, true},
	} {
		src := "package p\ntype releaseFunc func() error\ntype countFunc func() int\n" +
			"type holder struct { fn func(*S) error; closer releaseFunc; count func() int; n countFunc }\n" +
			"func f(f32, vec func(*S) error, release, setup func() error, next func(int32) (int32, error), count func() int, closer releaseFunc, n countFunc, h holder) error {\n" +
			tt.body + "\nreturn nil\n}\n"
		calls, values, bad, err := discardedReleases("control.go", []byte(src))
		if err != nil {
			t.Fatalf("%s: %v", tt.name, err)
		}
		if calls != 0 || values != tt.values {
			t.Errorf("%s: %d release calls and %d function value calls found, want 0 and %d", tt.name, calls, values, tt.values)
		}
		if got := len(bad) > 0; got != tt.bad {
			t.Errorf("%s: flagged %v, want %v (%v)", tt.name, got, tt.bad, bad)
		}
	}
}

// TestUnallowed checks that an allowance covers exactly as many discards as
// it counts, and that one left unused is reported.
func TestUnallowed(t *testing.T) {
	d := discard{key: "a.go: c.Close()", pos: "a.go:1:1"}
	for _, tt := range []struct {
		name       string
		discards   []discard
		allowed    map[string]int
		bad, stale int
	}{
		{"positive control: allowance used up", []discard{d}, map[string]int{d.key: 1}, 0, 0},
		{"not allowed", []discard{d}, nil, 1, 0},
		{"one more than allowed", []discard{d, d}, map[string]int{d.key: 1}, 1, 0},
		{"allowance unused", nil, map[string]int{d.key: 1}, 0, 1},
		{"other text not covered", []discard{{key: "a.go: c.Free()", pos: "a.go:2:1"}}, map[string]int{d.key: 1}, 1, 1},
	} {
		bad, stale := unallowed(tt.discards, tt.allowed)
		if len(bad) != tt.bad || len(stale) != tt.stale {
			t.Errorf("%s: bad %q, stale %q; want %d, %d", tt.name, bad, stale, tt.bad, tt.stale)
		}
	}
}
