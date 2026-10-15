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
// deviceWeight.release, GPUModelWeights.free, and the unload functions
// GPUModules.close takes). The check matches names, not types, so a
// method by one of these names that returns no error is matched too and
// must be allowed below.
var releaseNames = map[string]bool{
	"Close": true, "Free": true, "Synchronize": true, "CtxSynchronize": true,
	"Reset": true, "close": true, "release": true, "free": true, "unload": true,
}

// discardAllowed counts, by file and statement text, the statements that
// drop a release error on purpose.
var discardAllowed = map[string]int{
	// The L3 must-fail control in TestReleaseErrorOwner stands for a
	// caller that drops the close error; dropping it is what it tests.
	"module_ownership_test.go: c.Close()": 1,
	// cudakv.Cache.Reset returns no error; it only rewinds the cache
	// length. It is matched by name only.
	"main.go: kvCache.Reset()": 1,
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
// named in releaseNames.
func isReleaseCall(e ast.Expr) bool {
	call, ok := ast.Unparen(e).(*ast.CallExpr)
	if !ok {
		return false
	}
	sel, ok := call.Fun.(*ast.SelectorExpr)
	return ok && releaseNames[sel.Sel.Name]
}

// discardedReleases parses src and returns the number of release calls in
// it and the statements that drop a release error: a deferred or go
// statement calling it, an expression statement calling it, an assignment
// of its result to blanks only, an if statement whose initializer assigns
// its error to a variable the condition does not test, and an assignment
// of its error to a variable that a later statement in the same list
// assigns again before anything reads it (see overwrites).
func discardedReleases(name string, src []byte) (calls int, bad []discard, err error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, name, src, 0)
	if err != nil {
		return 0, nil, err
	}
	flag := func(s ast.Stmt) {
		text := string(src[fset.Position(s.Pos()).Offset:fset.Position(s.End()).Offset])
		bad = append(bad, discard{key: name + ": " + text, pos: fset.Position(s.Pos()).String()})
	}
	flagOverwritten := func(list []ast.Stmt) {
		for i, s := range list {
			as, ok := s.(*ast.AssignStmt)
			if !ok || len(as.Rhs) != 1 || !isReleaseCall(as.Rhs[0]) {
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
			if isReleaseCall(s) {
				calls++
			}
		case *ast.DeferStmt:
			if isReleaseCall(s.Call) {
				flag(s)
			}
		case *ast.GoStmt:
			if isReleaseCall(s.Call) {
				flag(s)
			}
		case *ast.ExprStmt:
			if isReleaseCall(s.X) {
				flag(s)
			}
		case *ast.AssignStmt:
			if !slices.ContainsFunc(s.Lhs, func(e ast.Expr) bool {
				id, ok := e.(*ast.Ident)
				return !ok || id.Name != "_"
			}) && slices.ContainsFunc(s.Rhs, isReleaseCall) {
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
			if !ok || len(init.Rhs) != 1 || !isReleaseCall(init.Rhs[0]) {
				break
			}
			if v, ok := init.Lhs[len(init.Lhs)-1].(*ast.Ident); ok && v.Name != "_" && !mentions(s.Cond, v.Name) {
				flag(s)
			}
		}
		return true
	})
	return calls, bad, nil
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
// a method named in releaseNames.
func TestReleaseErrorsReported(t *testing.T) {
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	var discards []discard
	perFile := make(map[string]int)
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".go") {
			continue
		}
		src, err := os.ReadFile(e.Name())
		if err != nil {
			t.Fatal(err)
		}
		calls, bad, err := discardedReleases(e.Name(), src)
		if err != nil {
			t.Fatal(err)
		}
		perFile[e.Name()] = calls
		discards = append(discards, bad...)
	}
	// The files holding the cleanup paths must have been read and must
	// contain release calls, or a clean result means nothing was looked at.
	for _, name := range []string{
		"main.go", "steptime.go", "model.go", "gpu_model.go", "gen.go",
		"profile_cuda_test.go", "fused_parity_cuda_test.go", "bf16_parity_cuda_test.go",
	} {
		if perFile[name] == 0 {
			t.Errorf("%s: no release calls found", name)
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
		calls, bad, err := discardedReleases("control.go", []byte(src))
		if err != nil {
			t.Fatalf("%s: %v", tt.name, err)
		}
		if calls != tt.calls {
			t.Errorf("%s: %d release calls found, want %d", tt.name, calls, tt.calls)
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
