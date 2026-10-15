package main

import (
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"maps"
	"os"
	"path/filepath"
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
// return an owner, to release, together with an error. The value reports
// whether the owner is a pointer, nil when nothing is held: Free on a nil
// *GPUScratchpad or *GPUModelWeights and Close on a nil *GPUModules
// panic, so its release must be guarded by "if owner != nil".
var ownerReturning = map[string]bool{
	"uploadModel": true, "UploadModelToGPU": true, "NewGPUScratchpad": true,
	"CompileGPUModules": true, "compileGPUModules": true,
	"cudakv.New": true, "cudamem.Alloc": false,
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

// releasesFirst counts the calls in src to an ownerReturning constructor,
// in any form, and checks each "owner, err := ctor(...)" or "owner, err =
// ctor(...)" statement: owner and err must be named, and the next
// statement must register a release of owner (see registersRelease), so
// that an owner returned with an error is released before anything else
// runs. Calls in other forms, such as "return ctor(...)" or "var owner, err
// = ctor(...)", are counted but not checked.
func releasesFirst(name, src string) (n callCount, bad []string, err error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, name, src, 0)
	if err != nil {
		return callCount{}, nil, err
	}
	ast.Inspect(f, func(node ast.Node) bool {
		if call, ok := node.(*ast.CallExpr); ok {
			if _, ok := ownerReturning[calleeName(call)]; ok {
				n.calls++
			}
		}
		block, ok := node.(*ast.BlockStmt)
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
			ctor := calleeName(call)
			nilable, ok := ownerReturning[ctor]
			if !ok {
				continue
			}
			n.checked++
			pos := fset.Position(as.Pos())
			owner, _ := as.Lhs[0].(*ast.Ident)
			errv, _ := as.Lhs[1].(*ast.Ident)
			if owner == nil || owner.Name == "_" || errv == nil || errv.Name == "_" {
				bad = append(bad, fmt.Sprintf("%v: %s owner or error discarded", pos, ctor))
				continue
			}
			if i+1 >= len(block.List) || !registersRelease(block.List[i+1], owner.Name, nilable) {
				guard := ""
				if nilable {
					guard = fmt.Sprintf(" under \"if %s != nil\"", owner.Name)
				}
				bad = append(bad, fmt.Sprintf("%v: %s owner %s not released%s in the next statement", pos, ctor, owner.Name, guard))
			}
		}
		return true
	})
	return n, bad, nil
}

// registersRelease reports whether stmt arranges for owner to be released
// on every path out of the function: a defer or t.Cleanup whose call
// releases it, or, if nilable, only "if owner != nil" whose body is
// such a statement.
func registersRelease(stmt ast.Stmt, owner string, nilable bool) bool {
	if nilable {
		s, ok := stmt.(*ast.IfStmt)
		if !ok || s.Init != nil || s.Else != nil || len(s.Body.List) != 1 {
			return false
		}
		cond, ok := s.Cond.(*ast.BinaryExpr)
		if !ok || cond.Op != token.NEQ {
			return false
		}
		x, xok := cond.X.(*ast.Ident)
		y, yok := cond.Y.(*ast.Ident)
		if !xok || !yok || x.Name != owner || y.Name != "nil" {
			return false
		}
		stmt = s.Body.List[0]
	}
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

// A callCount counts the calls a source check watches in one file: all of
// them, in any form, and those in the one form it inspects. Rewriting an
// inspected call into another form changes checked, not calls.
type callCount struct{ calls, checked int }

// scanDir runs check on every .go file in dir except those named in skip,
// whatever their build constraints, and returns the count check reports
// for each file, omitting files with no calls, and the problems it reports.
func scanDir(dir string, check func(name, src string) (callCount, []string, error), skip ...string) (map[string]callCount, []string, error) {
	paths, err := filepath.Glob(filepath.Join(dir, "*.go"))
	if err != nil {
		return nil, nil, err
	}
	counts := make(map[string]callCount)
	var bad []string
	for _, p := range paths {
		name := filepath.Base(p)
		if slices.Contains(skip, name) {
			continue
		}
		src, err := os.ReadFile(p)
		if err != nil {
			return nil, nil, err
		}
		n, b, err := check(name, string(src))
		if err != nil {
			return nil, nil, err
		}
		if n.calls > 0 {
			counts[name] = n
		}
		bad = append(bad, b...)
	}
	return counts, bad, nil
}

// countDiffs returns, sorted by file, each file whose counts in got differ
// from its counts in want, a missing file counting zero.
func countDiffs(got, want map[string]callCount) []string {
	names := slices.Concat(slices.Collect(maps.Keys(got)), slices.Collect(maps.Keys(want)))
	slices.Sort(names)
	var diffs []string
	for _, name := range slices.Compact(names) {
		if got[name] != want[name] {
			g, w := got[name], want[name]
			diffs = append(diffs, fmt.Sprintf("%s: %d calls, %d checked; want %d, %d", name, g.calls, g.checked, w.calls, w.checked))
		}
	}
	return diffs
}

// packageWith returns a temporary directory holding a copy of every .go
// file in the package and a file name with contents src.
func packageWith(t *testing.T, name, src string) string {
	t.Helper()
	dir := t.TempDir()
	paths, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range paths {
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, p), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(dir, name), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// isExtra reports whether a problem is in the file packageWith adds.
func isExtra(problem string) bool { return strings.HasPrefix(problem, "extra.go:") }

// rewriteDiffs returns the count differences from want that scanDir with
// check reports for a copy of the package in which the first from in file
// is replaced by to.
func rewriteDiffs(t *testing.T, check func(name, src string) (callCount, []string, error), want map[string]callCount, file, from, to string, skip ...string) []string {
	t.Helper()
	b, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	src := strings.Replace(string(b), from, to, 1)
	if src == string(b) {
		t.Fatalf("control does not apply: %s has no %q", file, from)
	}
	got, _, err := scanDir(packageWith(t, file, src), check, skip...)
	if err != nil {
		t.Fatal(err)
	}
	return countDiffs(got, want)
}

// ownerCalls counts, by file, the calls to an ownerReturning constructor
// in the package, and those releasesFirst inspects. A call added in any
// file, in any form, or rewritten out of the inspected form, changes a
// count, and must be reviewed before the count is updated here.
var ownerCalls = map[string]callCount{
	"main.go":                   {4, 4},
	"fused_parity_cuda_test.go": {6, 6},
	"bf16_parity_cuda_test.go":  {6, 6},
	"profile_cuda_test.go":      {2, 2},
	// The constructors themselves: "return ctor(...)", and allocations
	// assigned straight into the owner they return, which this check
	// counts but does not inspect.
	"gpu_kernels.go":  {1, 0},
	"gpu_model.go":    {5, 0},
	"weights_bf16.go": {2, 0},
}

// TestCallersReleaseReturnedOwners checks every call to a constructor that
// can return an owner with an error, in every .go file of the package.
func TestCallersReleaseReturnedOwners(t *testing.T) {
	got, bad, err := scanDir(".", releasesFirst)
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range countDiffs(got, ownerCalls) {
		t.Error(d)
	}
	for _, b := range bad {
		t.Error(b)
	}

	// Must-fail controls: a call in a file the count does not list is
	// caught, by its count if it releases its owner and by the check too
	// if it does not; and a checked call in a listed file rewritten into a
	// form the check does not inspect is caught by its file's count.
	for _, tt := range []struct {
		name, body string
		bad        bool
	}{
		{"released", "kv, err := cudakv.New(dev, 1, 1, 1, 1)\nif kv != nil {\ndefer kv.Close()\n}\nreturn err", false},
		{"error first", "kv, err := cudakv.New(dev, 1, 1, 1, 1)\nif err != nil {\nreturn err\n}\nreturn kv.Close()", true},
		{"returned", "return NewGPUScratchpad(dev, cfg)", false},
	} {
		dir := packageWith(t, "extra.go", "package main\nfunc f() error {\n"+tt.body+"\n}\n")
		got, bad, err := scanDir(dir, releasesFirst)
		if err != nil {
			t.Fatalf("unlisted file, %s: %v", tt.name, err)
		}
		if d := countDiffs(got, ownerCalls); !slices.ContainsFunc(d, isExtra) {
			t.Errorf("unlisted file, %s: count differences %q, want extra.go's", tt.name, d)
		}
		if flagged := slices.ContainsFunc(bad, isExtra); flagged != tt.bad {
			t.Errorf("unlisted file, %s: %q, want extra.go flagged %v", tt.name, bad, tt.bad)
		}
	}
	d := rewriteDiffs(t, releasesFirst, ownerCalls, "main.go", "scratch, err := NewGPUScratchpad(", "var scratch, err = NewGPUScratchpad(")
	if !slices.ContainsFunc(d, func(s string) bool { return strings.HasPrefix(s, "main.go:") }) {
		t.Errorf("checked call rewritten as a var declaration: count differences %q, want main.go's", d)
	}

	for _, tt := range []struct {
		name, body string
		bad        bool
	}{
		{"positive control: release registered first, any error name", `
		kv, e := cudakv.New(dev, 1, 1, 1, 1)
		if kv != nil {
			defer kv.Close()
		}
		if e != nil {
			return e
		}`, false},
		{"positive control: deferred literal reporting the error", `
		sp, err := NewGPUScratchpad(dev, cfg)
		if sp != nil {
			defer func() {
				if err := sp.Free(); err != nil {
					log(err)
				}
			}()
		}
		if err != nil {
			return err
		}`, false},
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
		{"guard releases another owner", `
		sp, err := NewGPUScratchpad(dev, cfg)
		if sp != nil {
			defer gw.Free()
		}`, true},
		{"guard without a release", `
		sp, err := NewGPUScratchpad(dev, cfg)
		if sp != nil {
			_ = sp
		}`, true},
		{"owner discarded", `
		_, err := NewGPUScratchpad(dev, cfg)
		return err`, true},
		{"error discarded", `
		sp, _ := NewGPUScratchpad(dev, cfg)
		if sp != nil {
			defer sp.Free()
		}`, true},
		{"assignment, not declaration", `
		gw, err = uploadModel(dev, w, cfg, dtype)
		return err`, true},
		{"nilable owner released unguarded", `
		sp, err := NewGPUScratchpad(dev, cfg)
		defer func() { _ = sp.Free() }()
		if err != nil {
			return err
		}`, true},
		{"statement before the release", `
		kv, err := cudakv.New(dev, 1, 1, 1, 1)
		if done {
			return nil
		}
		if kv != nil {
			defer kv.Close()
		}`, true},
		{"guarded release not deferred", `
		sp, err := NewGPUScratchpad(dev, cfg)
		if sp != nil {
			sp.Free()
		}`, true},
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
		sp, err := NewGPUScratchpad(dev, cfg)
		if sp != nil {
			defer func() { _ = sp.Free }()
		}`, true},
		{"release under if false", `
		kv, err := cudakv.New(dev, 1, 1, 1, 1)
		if kv != nil {
			if false {
				defer kv.Close()
			}
		}`, true},
	} {
		src := "package p\nfunc f() error {\n" + tt.body + "\nreturn nil\n}\n"
		n, bad, err := releasesFirst("control.go", src)
		if err != nil || n != (callCount{1, 1}) || (len(bad) > 0) != tt.bad {
			t.Errorf("%s: %+v calls, %q, %v; want 1 checked, flagged %v", tt.name, n, bad, err, tt.bad)
		}
	}
	// Forms the check does not inspect count as calls but not as checked.
	for _, body := range []string{
		"var sp, err = NewGPUScratchpad(dev, cfg)",
		"if sp, err = NewGPUScratchpad(dev, cfg); err != nil {\nreturn err\n}",
		"return NewGPUScratchpad(dev, cfg)",
	} {
		src := "package p\nfunc f() error {\n" + body + "\nreturn nil\n}\n"
		if n, _, err := releasesFirst("control.go", src); err != nil || n != (callCount{1, 0}) {
			t.Errorf("%q: %+v calls, %v; want 1 call, 0 checked", body, n, err)
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

// loadCalls counts, by file, the module compiles and loads in the
// package, and those callErrorsReleased inspects, except gen.go's, which
// TestGenReleasesCompileError counts and checks: gen.go builds alone and
// cannot call releaseErrorOwner.
var loadCalls = map[string]callCount{
	"main.go":                   {5, 5},
	"fused_parity_cuda_test.go": {4, 4},
	"bf16_parity_cuda_test.go":  {4, 4},
	"gpu_kernels.go":            {2, 2},
}

// TestCallErrorsReleased checks that every module compile and load in the
// package passes its error through releaseErrorOwner.
func TestCallErrorsReleased(t *testing.T) {
	got, bad, err := scanDir(".", callErrorsReleased, "gen.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range countDiffs(got, loadCalls) {
		t.Error(d)
	}
	for _, b := range bad {
		t.Error(b)
	}

	// Must-fail controls: gen.go is skipped by name only; a load in a file
	// the count does not list is caught, by its count whatever its form and
	// by the check too if it returns the error unreleased; and a checked
	// load in a listed file rewritten into a form the check does not
	// inspect is caught by its file's count.
	if _, bad, err := scanDir(".", callErrorsReleased); err != nil || len(bad) == 0 {
		t.Errorf("gen.go not skipped: %q, %v; want its compile flagged", bad, err)
	}
	for _, tt := range []struct {
		name, body string
		bad        bool
	}{
		{"released", "m, err := c.Load(dev)\nif err != nil {\nreturn releaseErrorOwner(err)\n}\n_ = m\nreturn nil", false},
		{"error returned unreleased", "m, err := c.Load(dev)\nif err != nil {\nreturn err\n}\n_ = m\nreturn nil", true},
		{"returned", "return c.Load(dev)", false},
	} {
		dir := packageWith(t, "extra.go", "package main\nfunc f() error {\n"+tt.body+"\n}\n")
		got, bad, err := scanDir(dir, callErrorsReleased, "gen.go")
		if err != nil {
			t.Fatalf("unlisted file, %s: %v", tt.name, err)
		}
		if d := countDiffs(got, loadCalls); !slices.ContainsFunc(d, isExtra) {
			t.Errorf("unlisted file, %s: count differences %q, want extra.go's", tt.name, d)
		}
		if flagged := slices.ContainsFunc(bad, isExtra); flagged != tt.bad {
			t.Errorf("unlisted file, %s: %q, want extra.go flagged %v", tt.name, bad, tt.bad)
		}
	}
	d := rewriteDiffs(t, callErrorsReleased, loadCalls, "main.go", "attnMod, err := attnModCompiled.Load(", "var attnMod, err = attnModCompiled.Load(", "gen.go")
	if !slices.ContainsFunc(d, func(s string) bool { return strings.HasPrefix(s, "main.go:") }) {
		t.Errorf("checked load rewritten as a var declaration: count differences %q, want main.go's", d)
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
		n, bad, err := callErrorsReleased("control.go", src)
		if err != nil || n != (callCount{1, 1}) || (len(bad) > 0) != tt.bad {
			t.Errorf("%s: %+v loads, %q, %v; want 1 checked, flagged %v", tt.name, n, bad, err, tt.bad)
		}
	}
	// Forms the check does not inspect count as calls but not as checked.
	for _, body := range []string{
		"var m, err = c.Load(dev)",
		"if m, err = c.Load(dev); err != nil {\nreturn err\n}",
		"return c.Load(dev)",
	} {
		src := "package p\nfunc f() error {\n" + body + "\nreturn nil\n}\n"
		if n, _, err := callErrorsReleased("control.go", src); err != nil || n != (callCount{1, 0}) {
			t.Errorf("%q: %+v loads, %v; want 1 call, 0 checked", body, n, err)
		}
	}
}

// isLoad reports whether call is a method or qualified call named Load or
// Compile.
func isLoad(call *ast.CallExpr) bool {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	return ok && (sel.Sel.Name == "Load" || sel.Sel.Name == "Compile")
}

// callErrorsReleased counts the Load and Compile calls in src, in any form,
// and checks each "x, err := y.Load(...)" or "x, err = y.Load(...)": the
// next statement must be "if err != nil" whose body calls
// releaseErrorOwner(err). Calls in other forms are counted but not checked.
func callErrorsReleased(name, src string) (n callCount, bad []string, err error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, name, src, 0)
	if err != nil {
		return callCount{}, nil, err
	}
	ast.Inspect(f, func(node ast.Node) bool {
		if call, ok := node.(*ast.CallExpr); ok && isLoad(call) {
			n.calls++
		}
		block, ok := node.(*ast.BlockStmt)
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
			if !isLoad(call) {
				continue
			}
			n.checked++
			errv, _ := as.Lhs[1].(*ast.Ident)
			ok = errv != nil && i+1 < len(block.List) && releasesCallError(block.List[i+1], errv.Name)
			if !ok {
				bad = append(bad, fmt.Sprintf("%v: call error not passed to releaseErrorOwner", fset.Position(as.Pos())))
			}
		}
		return true
	})
	return n, bad, nil
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
		{"second compile, if initializer", ok + "\nif _, err := nvrtcx.Compile(src); err != nil {\nreturn err\n}", true},
		{"compile returned", ok + "\nreturn nvrtcx.Compile(src)", true},
		{"load, var declaration", ok + "\nvar m, lerr = mod.Load(dev)", true},
	} {
		src := "package p\nfunc f() error {\nmod, err := nvrtcx.Compile(src)\n" + tt.branch + "\n_ = mod\nreturn nil\n}\n"
		bad, err := genReleaseFailures(src)
		if err != nil || (len(bad) > 0) != tt.bad {
			t.Errorf("%s: %q, %v; want flagged %v", tt.name, bad, err, tt.bad)
		}
	}
}

// genReleaseFailures returns how src departs from: no use of
// releaseErrorOwner, and exactly one Load or Compile call in any form,
// which is "x, err := nvrtcx.Compile(...)" followed by "if err != nil"
// whose body passes err to errors.As with &c, calls c.Close(), and joins
// the Close error into err.
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
	calls, compiles, released := 0, 0, 0
	ast.Inspect(f, func(n ast.Node) bool {
		if call, ok := n.(*ast.CallExpr); ok && isLoad(call) {
			calls++
		}
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
	if calls != 1 || compiles != 1 || released != 1 {
		bad = append(bad, fmt.Sprintf("%d loads and compiles, %d checked compiles, %d release their error; want 1, 1 and 1", calls, compiles, released))
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
