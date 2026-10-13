package main

import (
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

var (
	errFakeLookup = errors.New("fake lookup failed")
	errFakeUnload = errors.New("fake unload failed")
)

// fakeModule is a module handle with fake lookup and unload. Lookups fail
// from the failAt'th on (0 means never); the first unloadFails unloads fail.
// A successful unload does not clear the handle, so the caller must.
type fakeModule struct {
	failAt      int
	lookups     int
	unloadFails int
	unloads     int
	handles     []driver.Module // the handle each unload was given
}

func (f *fakeModule) lookup(name string) (driver.Function, error) {
	f.lookups++
	if f.failAt > 0 && f.lookups >= f.failAt {
		return 0, errFakeLookup
	}
	return driver.Function(f.lookups), nil
}

func (f *fakeModule) unload(m *driver.Module) error {
	f.unloads++
	f.handles = append(f.handles, *m)
	if f.unloadFails > 0 {
		f.unloadFails--
		return errFakeUnload
	}
	return nil
}

type lookupFunc func(driver.Module, func(string) (driver.Function, error), func(*driver.Module) error) (*GPUModules, error)

// moduleFailures returns how lookup departs from: on success, every kernel
// found and nothing unloaded; on a failed lookup, the module unloaded and
// the lookup error returned; and if that unload fails too, both errors and
// a GPUModules still owning the module, which a later Close releases once.
func moduleFailures(lookup lookupFunc) []string {
	const loaded = driver.Module(42)
	var bad []string

	f := &fakeModule{}
	gm, err := lookup(loaded, f.lookup, f.unload)
	if err != nil || gm == nil || gm.mod != loaded || gm.embedBF16Fn == 0 || f.unloads != 0 {
		bad = append(bad, fmt.Sprintf("success: %v, %v, %d unloads", gm, err, f.unloads))
	}

	f = &fakeModule{failAt: 3}
	gm, err = lookup(loaded, f.lookup, f.unload)
	if gm != nil || !errors.Is(err, errFakeLookup) || errors.Is(err, errFakeUnload) || f.unloads != 1 {
		bad = append(bad, fmt.Sprintf("lookup fails: %v, %v, %d unloads", gm, err, f.unloads))
	}

	f = &fakeModule{failAt: 3, unloadFails: 1}
	gm, err = lookup(loaded, f.lookup, f.unload)
	if !errors.Is(err, errFakeLookup) || !errors.Is(err, errFakeUnload) {
		bad = append(bad, fmt.Sprintf("lookup and unload fail: error %v, want both", err))
	}
	if gm == nil || gm.mod != loaded {
		return append(bad, fmt.Sprintf("lookup and unload fail: %v, want the module kept", gm))
	}
	if err := gm.close(f.unload); err != nil || gm.mod != 0 {
		bad = append(bad, fmt.Sprintf("retry: %v, module %d", err, gm.mod))
	}
	if err := gm.close(f.unload); err != nil || f.unloads != 2 {
		bad = append(bad, fmt.Sprintf("repeat: %v, %d unloads, want 2", err, f.unloads))
	}
	if !slices.Equal(f.handles, []driver.Module{loaded, loaded}) {
		bad = append(bad, fmt.Sprintf("unloaded handles %v, want [%d %d]", f.handles, loaded, loaded))
	}
	return bad
}

func TestLookupModules(t *testing.T) {
	for _, f := range moduleFailures(lookupModules) {
		t.Error(f)
	}
	// Must-fail control: the behaviour this change replaced, which returned
	// nil whether or not the unload succeeded.
	old := func(loaded driver.Module, lookup func(string) (driver.Function, error), unload func(*driver.Module) error) (*GPUModules, error) {
		gm, err := lookupModules(loaded, lookup, unload)
		if err != nil {
			return nil, err
		}
		return gm, nil
	}
	if len(moduleFailures(old)) == 0 {
		t.Error("must-fail control: module dropped with a failed unload, not detected")
	}
}

func TestModulesCloseRetains(t *testing.T) {
	f := &fakeModule{unloadFails: 1}
	gm := &GPUModules{mod: 42}
	if err := gm.close(f.unload); !errors.Is(err, errFakeUnload) || gm.mod != 42 {
		t.Errorf("failed close: %v, module %d; want the error and the module kept", err, gm.mod)
	}
	if err := gm.close(f.unload); err != nil || gm.mod != 0 {
		t.Errorf("retry: %v, module %d", err, gm.mod)
	}
	if err := gm.close(f.unload); err != nil || f.unloads != 2 {
		t.Errorf("repeat: %v, %d unloads, want 2", err, f.unloads)
	}
	if !slices.Equal(f.handles, []driver.Module{42, 42}) {
		t.Errorf("unloaded handles %v, want [42 42]", f.handles)
	}
	var zero GPUModules
	if err := zero.Close(); err != nil {
		t.Errorf("zero modules: %v", err)
	}
}

// ownerReturning names the constructors this package calls that can
// return an owner, to release, together with an error.
var ownerReturning = map[string]bool{
	"uploadModel": true, "UploadModelToGPU": true, "NewGPUScratchpad": true,
	"CompileGPUModules": true, "compileGPUModules": true,
	"cudakv.New": true, "cudamem.Alloc": true,
}

// calleeName returns the name of the function call calls, as pkg.Name for
// a qualified name, ignoring type arguments.
func calleeName(call *ast.CallExpr) string {
	fun := call.Fun
	if ix, ok := fun.(*ast.IndexExpr); ok {
		fun = ix.X
	}
	switch f := fun.(type) {
	case *ast.Ident:
		return f.Name
	case *ast.SelectorExpr:
		if x, ok := f.X.(*ast.Ident); ok {
			return x.Name + "." + f.Sel.Name
		}
	}
	return ""
}

// releasesFirst checks every call in src to an ownerReturning constructor:
// a statement releasing the owner (calling its Free or Close, directly or
// in a deferred or registered function) must come after the call and
// before the first statement that examines the error.
func releasesFirst(name, src string) (calls int, bad []string, err error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, name, src, 0)
	if err != nil {
		return 0, nil, err
	}
	ast.Inspect(f, func(n ast.Node) bool {
		block, ok := n.(*ast.BlockStmt)
		if !ok {
			return true
		}
		for i, stmt := range block.List {
			as, ok := stmt.(*ast.AssignStmt)
			if !ok || len(as.Lhs) != 2 || len(as.Rhs) != 1 {
				continue
			}
			call, ok := as.Rhs[0].(*ast.CallExpr)
			if !ok || !ownerReturning[calleeName(call)] {
				continue
			}
			calls++
			owner, _ := as.Lhs[0].(*ast.Ident)
			errv, _ := as.Lhs[1].(*ast.Ident)
			if owner == nil || owner.Name == "_" || errv == nil {
				bad = append(bad, fmt.Sprintf("%v: %s owner or error discarded", fset.Position(as.Pos()), calleeName(call)))
				continue
			}
			released := false
			for _, next := range block.List[i+1:] {
				// A registered release may mention a shadowing err of its
				// own, so it is recognized before the error check is.
				if registersRelease(next, owner.Name) {
					released = true
					break
				}
				if mentions(next, errv.Name) {
					break
				}
			}
			if !released {
				bad = append(bad, fmt.Sprintf("%v: %s owner %s not released before %s is examined", fset.Position(as.Pos()), calleeName(call), owner.Name, errv.Name))
			}
		}
		return true
	})
	return calls, bad, nil
}

// registersRelease reports whether stmt arranges for owner to be released
// on every path out of the function: a defer or t.Cleanup whose call
// releases it, or "if owner != nil" whose body is such a statement.
func registersRelease(stmt ast.Stmt, owner string) bool {
	switch s := stmt.(type) {
	case *ast.DeferStmt:
		return callReleases(s.Call, owner)
	case *ast.ExprStmt:
		call, ok := s.X.(*ast.CallExpr)
		if !ok || len(call.Args) != 1 {
			return false
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok || sel.Sel.Name != "Cleanup" {
			return false
		}
		lit, ok := call.Args[0].(*ast.FuncLit)
		return ok && bodyReleases(lit.Body, owner)
	case *ast.IfStmt:
		if s.Init != nil || s.Else != nil || len(s.Body.List) != 1 {
			return false
		}
		cond, ok := s.Cond.(*ast.BinaryExpr)
		if !ok || cond.Op != token.NEQ {
			return false
		}
		x, xok := cond.X.(*ast.Ident)
		y, yok := cond.Y.(*ast.Ident)
		return xok && yok && x.Name == owner && y.Name == "nil" && registersRelease(s.Body.List[0], owner)
	}
	return false
}

// callReleases reports whether call is owner.Free() or owner.Close(), or a
// call of a function literal whose body releases owner.
func callReleases(call *ast.CallExpr, owner string) bool {
	if lit, ok := call.Fun.(*ast.FuncLit); ok {
		return bodyReleases(lit.Body, owner)
	}
	return isRelease(call, owner)
}

// bodyReleases reports whether a top-level statement of body calls
// owner.Free or owner.Close, directly or as the call in an if statement's
// initializer, as in "if err := owner.Free(); err != nil".
func bodyReleases(body *ast.BlockStmt, owner string) bool {
	for _, stmt := range body.List {
		switch s := stmt.(type) {
		case *ast.ExprStmt:
			if call, ok := s.X.(*ast.CallExpr); ok && isRelease(call, owner) {
				return true
			}
		case *ast.AssignStmt:
			if call, ok := s.Rhs[0].(*ast.CallExpr); ok && len(s.Rhs) == 1 && isRelease(call, owner) {
				return true
			}
		case *ast.IfStmt:
			if init, ok := s.Init.(*ast.AssignStmt); ok && len(init.Rhs) == 1 {
				if call, ok := init.Rhs[0].(*ast.CallExpr); ok && isRelease(call, owner) {
					return true
				}
			}
		}
	}
	return false
}

func isRelease(call *ast.CallExpr, owner string) bool {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok || (sel.Sel.Name != "Free" && sel.Sel.Name != "Close") {
		return false
	}
	id, ok := sel.X.(*ast.Ident)
	return ok && id.Name == owner
}

// mentions reports whether n refers to the identifier name.
func mentions(n ast.Node, name string) bool {
	found := false
	ast.Inspect(n, func(n ast.Node) bool {
		if id, ok := n.(*ast.Ident); ok && id.Name == name {
			found = true
		}
		return !found
	})
	return found
}

// TestCallersReleaseReturnedOwners checks every call to a constructor that
// can return an owner with an error in main.go and the three CUDA test files,
// which hold all such calls in this package.
func TestCallersReleaseReturnedOwners(t *testing.T) {
	for _, tt := range []struct {
		file  string
		calls int
	}{
		{"main.go", 4},
		{"fused_parity_cuda_test.go", 6},
		{"bf16_parity_cuda_test.go", 6},
		{"profile_cuda_test.go", 2},
	} {
		src, err := os.ReadFile(tt.file)
		if err != nil {
			t.Fatal(err)
		}
		calls, bad, err := releasesFirst(tt.file, string(src))
		if err != nil {
			t.Fatal(err)
		}
		if calls != tt.calls {
			t.Errorf("%s: %d constructor calls, want %d", tt.file, calls, tt.calls)
		}
		for _, b := range bad {
			t.Error(b)
		}
	}
	for _, tt := range []struct {
		name, body string
		bad        bool
	}{
		{"positive control: release registered first", `
		kv, e := cudakv.New(dev, 1, 1, 1, 1)
		if kv != nil {
			defer kv.Close()
		}
		if e != nil {
			return e
		}`, false},
		{"error examined first", `
		kv, err := cudakv.New(dev, 1, 1, 1, 1)
		if err != nil {
			return err
		}
		defer kv.Close()`, true},
		{"generic allocation, error first", `
		s, err := cudamem.Alloc[int32](dev, 1)
		if err != nil {
			return err
		}
		defer s.Free()`, true},
		{"releases another owner", `
		gm, err := CompileGPUModules(dev)
		defer kv.Close()
		if err != nil {
			return err
		}`, true},
		{"owner discarded", `
		_, err := NewGPUScratchpad(dev, cfg)
		return err`, true},
		{"release only in the else branch", `
		kv, err := cudakv.New(dev, 1, 1, 1, 1)
		if err != nil {
			return err
		} else {
			defer kv.Close()
		}`, true},
		{"release only on success", `
		kv, err := cudakv.New(dev, 1, 1, 1, 1)
		if err == nil {
			defer kv.Close()
		}`, true},
		{"release in a closure never called", `
		kv, err := cudakv.New(dev, 1, 1, 1, 1)
		release := func() { kv.Close() }
		if err != nil {
			return err
		}
		release()`, true},
		{"method value, not a call", `
		kv, err := cudakv.New(dev, 1, 1, 1, 1)
		_ = kv.Close
		if err != nil {
			return err
		}`, true},
		{"release under if false", `
		kv, err := cudakv.New(dev, 1, 1, 1, 1)
		if false {
			defer kv.Close()
		}
		if err != nil {
			return err
		}`, true},
		{"positive control: t.Cleanup reporting the error", `
		s, err := cudamem.Alloc[int32](dev, 1)
		t.Cleanup(func() {
			if err := s.Free(); err != nil {
				t.Error(err)
			}
		})
		if err != nil {
			t.Fatal(err)
		}`, false},
	} {
		src := "package p\nfunc f() error {\n" + tt.body + "\nreturn nil\n}\n"
		calls, bad, err := releasesFirst("control.go", src)
		if err != nil || calls != 1 || (len(bad) > 0) != tt.bad {
			t.Errorf("%s: %d calls, %q, %v; want flagged %v", tt.name, calls, bad, err, tt.bad)
		}
	}
}

// heldCleanupError is the error a failed load returns while still holding
// a module, like nvrtcx's: Close retries the unload, failing the first
// fails times.
type heldCleanupError struct {
	fails  int
	closes int
}

func (e *heldCleanupError) Error() string { return "load failed, module retained" }

func (e *heldCleanupError) Close() error {
	e.closes++
	if e.fails > 0 {
		e.fails--
		return errFakeUnload
	}
	return nil
}

// releaseErrorFailures returns how release departs from: nothing closed for a
// nil or unrelated error, which is returned as is; a retained module
// closed once through a wrapped error, and the error returned as is; and
// if that close fails, the original error still carrying its Closer,
// joined with the close error.
func releaseErrorFailures(release func(error) error) []string {
	var bad []string
	if err := release(nil); err != nil {
		bad = append(bad, fmt.Sprintf("nil: %v", err))
	}
	if err := release(errFakeLookup); err != errFakeLookup {
		bad = append(bad, fmt.Sprintf("unrelated: %v", err))
	}

	held := &heldCleanupError{}
	wrapped := fmt.Errorf("load: %w", held)
	if err := release(wrapped); err != wrapped || held.closes != 1 {
		bad = append(bad, fmt.Sprintf("close succeeds: %v, %d closes", err, held.closes))
	}

	held = &heldCleanupError{fails: 1}
	wrapped = fmt.Errorf("load: %w", held)
	err := release(wrapped)
	var c *heldCleanupError
	if !errors.Is(err, wrapped) || !errors.As(err, &c) || c != held {
		bad = append(bad, fmt.Sprintf("close fails: %v loses the error carrying the module", err))
	}
	if !errors.Is(err, errFakeUnload) || !strings.Contains(fmt.Sprint(err), "release resources held by failed call") {
		bad = append(bad, fmt.Sprintf("close fails: %v does not report the failed release", err))
	}
	if held.closes != 1 {
		bad = append(bad, fmt.Sprintf("close fails: %d closes, want 1", held.closes))
	}
	return bad
}

func TestReleaseErrorOwner(t *testing.T) {
	for _, f := range releaseErrorFailures(releaseErrorOwner) {
		t.Error(f)
	}
	for _, m := range []struct {
		name    string
		release func(error) error
	}{
		{"L1 nothing closed", func(err error) error { return err }},
		{"L2 replaced by the close error", func(err error) error {
			var c io.Closer
			if errors.As(err, &c) {
				if cerr := c.Close(); cerr != nil {
					return cerr
				}
			}
			return err
		}},
		{"L3 close error dropped", func(err error) error {
			var c io.Closer
			if errors.As(err, &c) {
				c.Close()
			}
			return err
		}},
	} {
		if len(releaseErrorFailures(m.release)) == 0 {
			t.Errorf("must-fail control %s: not detected", m.name)
		}
	}
}

// TestCallErrorsReleased checks that every module compile and load in
// the program passes its error through releaseErrorOwner. gen.go, which
// builds alone, is checked by TestGenReleasesCompileError.
func TestCallErrorsReleased(t *testing.T) {
	for _, tt := range []struct {
		file  string
		loads int
	}{
		{"main.go", 5},
		{"fused_parity_cuda_test.go", 4},
		{"bf16_parity_cuda_test.go", 4},
		{"gpu_kernels.go", 2},
	} {
		src, err := os.ReadFile(tt.file)
		if err != nil {
			t.Fatal(err)
		}
		loads, bad, err := callErrorsReleased(tt.file, string(src))
		if err != nil {
			t.Fatal(err)
		}
		if loads != tt.loads {
			t.Errorf("%s: %d loads, want %d", tt.file, loads, tt.loads)
		}
		for _, b := range bad {
			t.Error(b)
		}
	}
	for _, tt := range []struct {
		name, body string
		bad        bool
	}{
		{"positive control", "if err != nil {\nreturn releaseErrorOwner(err)\n}", false},
		{"error returned unreleased", "if err != nil {\nreturn err\n}", true},
		{"helper given another error", "if err != nil {\nreleaseErrorOwner(nil)\nreturn err\n}", true},
		{"helper only on success", "if err == nil {\nreleaseErrorOwner(err)\n}", true},
	} {
		src := "package p\nfunc f() error {\nm, err := c.Load(dev)\n" + tt.body + "\nreturn nil\n}\n"
		loads, bad, err := callErrorsReleased("control.go", src)
		if err != nil || loads != 1 || (len(bad) > 0) != tt.bad {
			t.Errorf("%s: %d loads, %q, %v; want flagged %v", tt.name, loads, bad, err, tt.bad)
		}
	}
}

// callErrorsReleased checks each "x, err := y.Load(...)" in src: the next
// statement must be "if err != nil" whose body calls releaseErrorOwner(err).
func callErrorsReleased(name, src string) (loads int, bad []string, err error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, name, src, 0)
	if err != nil {
		return 0, nil, err
	}
	ast.Inspect(f, func(n ast.Node) bool {
		block, ok := n.(*ast.BlockStmt)
		if !ok {
			return true
		}
		for i, stmt := range block.List {
			as, ok := stmt.(*ast.AssignStmt)
			if !ok || len(as.Lhs) != 2 || len(as.Rhs) != 1 {
				continue
			}
			call, ok := as.Rhs[0].(*ast.CallExpr)
			if !ok {
				continue
			}
			if sel, ok := call.Fun.(*ast.SelectorExpr); !ok || (sel.Sel.Name != "Load" && sel.Sel.Name != "Compile") {
				continue
			}
			loads++
			errv, _ := as.Lhs[1].(*ast.Ident)
			ok = errv != nil && i+1 < len(block.List) && releasesCallError(block.List[i+1], errv.Name)
			if !ok {
				bad = append(bad, fmt.Sprintf("%v: call error not passed to releaseErrorOwner", fset.Position(as.Pos())))
			}
		}
		return true
	})
	return loads, bad, nil
}

// releasesCallError reports whether stmt is "if errv != nil { ... }" whose
// body calls releaseErrorOwner(errv).
func releasesCallError(stmt ast.Stmt, errv string) bool {
	is, ok := stmt.(*ast.IfStmt)
	if !ok || is.Init != nil {
		return false
	}
	cond, ok := is.Cond.(*ast.BinaryExpr)
	if !ok || cond.Op != token.NEQ {
		return false
	}
	x, xok := cond.X.(*ast.Ident)
	y, yok := cond.Y.(*ast.Ident)
	if !xok || !yok || x.Name != errv || y.Name != "nil" {
		return false
	}
	found := false
	ast.Inspect(is.Body, func(n ast.Node) bool {
		if c, isCall := n.(*ast.CallExpr); isCall && calleeName(c) == "releaseErrorOwner" && len(c.Args) == 1 {
			if id, isIdent := c.Args[0].(*ast.Ident); isIdent && id.Name == errv {
				found = true
			}
		}
		return !found
	})
	return found
}

// TestGenReleasesCompileError checks gen.go, which builds alone with
// "go run gen.go" and so cannot call releaseErrorOwner: its Compile error
// branch must find an io.Closer with errors.As, close it, and join a
// failure to the error.
func TestGenReleasesCompileError(t *testing.T) {
	src, err := os.ReadFile("gen.go")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(string(src), "//go:build ignore\n") {
		t.Fatal("gen.go is no longer a standalone build; this check assumes it is")
	}
	if bad, err := genReleaseFailures(string(src)); err != nil || len(bad) > 0 {
		t.Errorf("gen.go: %q, %v", bad, err)
	}

	const ok = `if err != nil {
		var c io.Closer
		if errors.As(err, &c) {
			if cerr := c.Close(); cerr != nil {
				err = errors.Join(err, cerr)
			}
		}
		return err
	}`
	for _, tt := range []struct {
		name, branch string
		bad          bool
	}{
		{"positive control", ok, false},
		{"branch taken on success", strings.Replace(ok, "err != nil {", "err == nil {", 1), true},
		{"closes another value", strings.Replace(ok, "c.Close()", "os.Stdout.Close()", 1), true},
		{"As on another error", strings.Replace(ok, "errors.As(err, &c)", "errors.As(other, &c)", 1), true},
		{"retry error dropped", strings.Replace(ok, "err = errors.Join(err, cerr)", "_ = cerr", 1), true},
		{"calls the program's helper", "if err != nil {\nreturn releaseErrorOwner(err)\n}", true},
	} {
		src := "package p\nfunc f() error {\nmod, err := nvrtcx.Compile(src)\n" + tt.branch + "\n_ = mod\nreturn nil\n}\n"
		bad, err := genReleaseFailures(src)
		if err != nil || (len(bad) > 0) != tt.bad {
			t.Errorf("%s: %q, %v; want flagged %v", tt.name, bad, err, tt.bad)
		}
	}
}

// genReleaseFailures returns how src departs from: no use of
// releaseErrorOwner, and exactly one nvrtcx.Compile, followed by
// "if err != nil" whose body passes err to errors.As with &c, calls
// c.Close(), and joins the Close error into err.
func genReleaseFailures(src string) ([]string, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "gen.go", src, 0)
	if err != nil {
		return nil, err
	}
	var bad []string
	if mentions(f, "releaseErrorOwner") {
		bad = append(bad, "calls releaseErrorOwner, which its standalone build cannot see")
	}
	compiles, released := 0, 0
	ast.Inspect(f, func(n ast.Node) bool {
		block, ok := n.(*ast.BlockStmt)
		if !ok {
			return true
		}
		for i, stmt := range block.List {
			as, ok := stmt.(*ast.AssignStmt)
			if !ok || len(as.Lhs) != 2 || len(as.Rhs) != 1 {
				continue
			}
			call, ok := as.Rhs[0].(*ast.CallExpr)
			if !ok || calleeName(call) != "nvrtcx.Compile" {
				continue
			}
			compiles++
			errv, _ := as.Lhs[1].(*ast.Ident)
			if errv != nil && i+1 < len(block.List) && genBranchReleases(block.List[i+1], errv.Name) {
				released++
			}
		}
		return true
	})
	if compiles != 1 || released != 1 {
		bad = append(bad, fmt.Sprintf("%d compiles, %d release their error; want 1 and 1", compiles, released))
	}
	return bad, nil
}

// genBranchReleases reports whether stmt is "if errv != nil" whose body
// calls errors.As(errv, &c), calls c.Close(), and calls errors.Join with
// errv.
func genBranchReleases(stmt ast.Stmt, errv string) bool {
	is, ok := stmt.(*ast.IfStmt)
	if !ok || is.Init != nil {
		return false
	}
	cond, ok := is.Cond.(*ast.BinaryExpr)
	if !ok || cond.Op != token.NEQ {
		return false
	}
	if x, ok := cond.X.(*ast.Ident); !ok || x.Name != errv {
		return false
	}
	if y, ok := cond.Y.(*ast.Ident); !ok || y.Name != "nil" {
		return false
	}
	target := ""
	ast.Inspect(is.Body, func(n ast.Node) bool {
		c, ok := n.(*ast.CallExpr)
		if !ok || calleeName(c) != "errors.As" || len(c.Args) != 2 {
			return true
		}
		e, eok := c.Args[0].(*ast.Ident)
		u, uok := c.Args[1].(*ast.UnaryExpr)
		if eok && uok && e.Name == errv && u.Op == token.AND {
			if id, ok := u.X.(*ast.Ident); ok {
				target = id.Name
			}
		}
		return true
	})
	if target == "" {
		return false
	}
	closes, joins := false, false
	ast.Inspect(is.Body, func(n ast.Node) bool {
		c, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		if isRelease(c, target) {
			closes = true
		}
		if calleeName(c) == "errors.Join" && len(c.Args) > 0 && mentions(c.Args[0], errv) {
			joins = true
		}
		return true
	})
	return closes && joins
}
