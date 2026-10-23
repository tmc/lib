package cublasltx_test

import (
	"context"
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"runtime"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

// Keep unresolved test owners reachable until process exit. A stream that
// could not be closed may still run work that reads the engines and buffers
// it was given, so those are never released after its failure is reported.
var retainedTestOwners struct {
	sync.Mutex
	owners []*testOwner
}

// A testOwner releases what a test acquired. Streams close first, and the
// engines and buffers their work reads are released only after every stream
// has closed. Register each result before checking the constructor's error,
// because a constructor can return a live owner together with an error.
type testOwner struct {
	mu      sync.Mutex
	streams []func() error
	deps    []func() error
}

// newTestOwner returns an owner that tb disposes of during cleanup.
func newTestOwner(tb testing.TB) *testOwner {
	o := new(testOwner)
	tb.Cleanup(func() { o.dispose(tb) })
	return o
}

// requireDevice acquires the GPU lease and then CUDA device 0, or skips when
// this machine has no CUDA driver. Both are held by the returned owner: the
// lease is its bottom entry and the device the next, so the device closes
// only after every later registration has been released, and the lease is
// released last. A failed release anywhere above keeps the lease held.
//
// Only the driver library is opened before the lease is held; cuInit and
// every other driver call come after. Unlike cudatest.Require, it does not
// scan past a device 0 that fails to open, and it always takes the lease.
func requireDevice(tb testing.TB) (*cuda.Device, *testOwner) {
	tb.Helper()
	var dev *cuda.Device
	owner := ownDevice(tb, loadDriver, acquireLease, func() func() error {
		dev = cudatest.RequireDevice(tb)
		return dev.Close
	})
	return dev, owner
}

// loadDriver opens the CUDA driver library without initializing it.
func loadDriver() error {
	return driver.Load("")
}

// acquireLease acquires the canonical GPU lease labeled with the test name.
// On failure it returns the cleanup carried by the error, if any.
func acquireLease(tb testing.TB) (func() error, error) {
	lease, err := cudalock.Acquire(context.Background(), cudalock.WithLabel(tb.Name()))
	if err != nil {
		var pending io.Closer
		if errors.As(err, &pending) {
			return pending.Close, err
		}
		return nil, err
	}
	return lease.Release, nil
}

// ownDevice skips tb if load fails, as cudatest.RequireDevice would, without
// taking the lease. Otherwise it returns a new test owner whose bottom entry
// releases the lease from acquire and whose next entry closes the device that
// open returns. open runs only once the lease is held.
func ownDevice(tb testing.TB, load func() error, acquire func(testing.TB) (func() error, error), open func() func() error) *testOwner {
	tb.Helper()
	if err := load(); err != nil {
		tb.Skipf("SKIPPED, NOT VERIFIED: the CUDA driver could not be loaded: %v", err)
	}
	owner := newTestOwner(tb)
	release, err := acquire(tb)
	if release != nil {
		owner.add(release)
	}
	if err != nil {
		tb.Fatalf("acquire GPU lease: %v", err)
	}
	owner.add(open())
	return owner
}

// stream registers s, if it is not nil, to be closed before any dependency.
func (o *testOwner) stream(s *cuda.Stream) {
	if s != nil {
		o.addStream(s.Close)
	}
}

// addStream registers fn to run before any dependency is released.
func (o *testOwner) addStream(fn func() error) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.streams = append(o.streams, fn)
}

// add registers release to run after every stream has closed.
func (o *testOwner) add(release func() error) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.deps = append(o.deps, release)
}

// Close closes the streams and then releases the dependencies, each in
// reverse order of registration. It stops at the first failure and keeps that
// entry and every entry not yet reached for a later Close. Entries that
// succeeded are never run again.
func (o *testOwner) Close() error {
	o.mu.Lock()
	defer o.mu.Unlock()
	if err := closeReverse(&o.streams); err != nil {
		return fmt.Errorf("close stream: %w", err)
	}
	if err := closeReverse(&o.deps); err != nil {
		return fmt.Errorf("release dependency: %w", err)
	}
	return nil
}

func closeReverse(list *[]func() error) error {
	for n := len(*list); n > 0; n-- {
		if err := (*list)[n-1](); err != nil {
			return err
		}
		*list = (*list)[:n-1]
	}
	return nil
}

// dispose closes o and, if that fails, reports the failure and retains o
// until process exit.
func (o *testOwner) dispose(tb testing.TB) {
	if err := o.Close(); err != nil {
		retainedTestOwners.Lock()
		retainedTestOwners.owners = append(retainedTestOwners.owners, o)
		retainedTestOwners.Unlock()
		tb.Errorf("cleanup: %v (remaining owners retained)", err)
	}
}

// recorder returns cleanup functions that log their names to calls and fail
// with the error stored in fail[name], if any.
func recorder(calls *[]string, fail map[string]error) func(string) func() error {
	return func(name string) func() error {
		return func() error {
			*calls = append(*calls, name)
			return fail[name]
		}
	}
}

func TestTestOwnerOrder(t *testing.T) {
	var calls []string
	entry := recorder(&calls, nil)
	o := new(testOwner)
	o.add(entry("engine"))
	o.addStream(entry("stream1"))
	o.add(entry("a"))
	o.addStream(entry("stream2"))
	o.add(entry("b"))
	o.stream(nil)
	for range 2 {
		if err := o.Close(); err != nil {
			t.Fatal(err)
		}
	}
	want := []string{"stream2", "stream1", "b", "a", "engine"}
	if !slices.Equal(calls, want) {
		t.Fatalf("calls = %v, want %v", calls, want)
	}
}

func TestTestOwnerFailedClose(t *testing.T) {
	blocked := errors.New("injected close failure")
	tests := []struct {
		name      string
		fail      string
		wantErr   string
		wantFirst []string
		wantRetry []string
	}{
		{
			name:      "stream",
			fail:      "stream2",
			wantErr:   "close stream: injected close failure",
			wantFirst: []string{"stream2"},
			wantRetry: []string{"stream2", "stream1", "b", "a"},
		},
		{
			name:      "first stream",
			fail:      "stream1",
			wantErr:   "close stream: injected close failure",
			wantFirst: []string{"stream2", "stream1"},
			wantRetry: []string{"stream1", "b", "a"},
		},
		{
			name:      "dependency",
			fail:      "b",
			wantErr:   "release dependency: injected close failure",
			wantFirst: []string{"stream2", "stream1", "b"},
			wantRetry: []string{"b", "a"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var calls []string
			fail := map[string]error{tt.fail: blocked}
			entry := recorder(&calls, fail)
			o := new(testOwner)
			o.addStream(entry("stream1"))
			o.add(entry("a"))
			o.addStream(entry("stream2"))
			o.add(entry("b"))

			err := o.Close()
			if !errors.Is(err, blocked) || err.Error() != tt.wantErr {
				t.Fatalf("Close = %v, want %q wrapping %v", err, tt.wantErr, blocked)
			}
			if !slices.Equal(calls, tt.wantFirst) {
				t.Fatalf("calls = %v, want %v", calls, tt.wantFirst)
			}

			// An explicit retry resumes at the failed entry and never
			// repeats one that already closed.
			calls = nil
			delete(fail, tt.fail)
			for range 2 {
				if err := o.Close(); err != nil {
					t.Fatal(err)
				}
			}
			if !slices.Equal(calls, tt.wantRetry) {
				t.Fatalf("retry calls = %v, want %v", calls, tt.wantRetry)
			}
		})
	}
}

// fakeTB records failures, skips and cleanups instead of acting on them.
// Fatalf and Skipf stop the calling goroutine, as they do on a real
// testing.TB, so callers that may reach them should go through run.
type fakeTB struct {
	testing.TB
	errs     []string
	skips    []string
	cleanups []func()
}

func (f *fakeTB) Helper()           {}
func (f *fakeTB) Name() string      { return "TestFake" }
func (f *fakeTB) Cleanup(fn func()) { f.cleanups = append(f.cleanups, fn) }
func (f *fakeTB) Errorf(format string, args ...any) {
	f.errs = append(f.errs, fmt.Sprintf(format, args...))
}
func (f *fakeTB) Fatalf(format string, args ...any) {
	f.Errorf(format, args...)
	runtime.Goexit()
}
func (f *fakeTB) Skipf(format string, args ...any) {
	f.skips = append(f.skips, fmt.Sprintf(format, args...))
	runtime.Goexit()
}

// loaded is a driver load that succeeds.
func loaded() error { return nil }

// run calls fn on its own goroutine, as testing does, and waits for it.
func (f *fakeTB) run(fn func()) {
	done := make(chan struct{})
	go func() {
		defer close(done)
		fn()
	}()
	<-done
}

// runCleanups runs the registered cleanups in reverse order.
func (f *fakeTB) runCleanups() {
	for i := len(f.cleanups) - 1; i >= 0; i-- {
		f.cleanups[i]()
	}
	f.cleanups = nil
}

func TestTestOwnerDisposeRetains(t *testing.T) {
	blocked := errors.New("injected close failure")
	var calls []string
	fail := map[string]error{"stream": blocked}
	entry := recorder(&calls, fail)
	o := new(testOwner)
	o.addStream(entry("stream"))
	o.add(entry("buffer"))

	r := &fakeTB{TB: t}
	o.dispose(r)
	want := "cleanup: close stream: injected close failure (remaining owners retained)"
	if !slices.Equal(r.errs, []string{want}) {
		t.Fatalf("errors = %q, want %q", r.errs, want)
	}
	if !slices.Equal(calls, []string{"stream"}) {
		t.Fatalf("calls = %v, buffer must stay live behind the stream", calls)
	}
	retainedTestOwners.Lock()
	retained := slices.Contains(retainedTestOwners.owners, o)
	retainedTestOwners.Unlock()
	if !retained {
		t.Fatal("unresolved owner was not retained")
	}

	calls = nil
	delete(fail, "stream")
	if err := o.Close(); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(calls, []string{"stream", "buffer"}) {
		t.Fatalf("retry calls = %v", calls)
	}

	// A clean dispose reports nothing and retains nothing.
	calls = nil
	clean := new(testOwner)
	clean.addStream(entry("stream"))
	clean.dispose(r)
	retainedTestOwners.Lock()
	retained = slices.Contains(retainedTestOwners.owners, clean)
	retainedTestOwners.Unlock()
	if len(r.errs) != 1 || retained || !slices.Equal(calls, []string{"stream"}) {
		t.Fatalf("clean dispose: errors=%q retained=%v calls=%v", r.errs, retained, calls)
	}
}

// TestDeviceOwnerLeaseCustody drives ownDevice, as requireDevice uses it,
// with a fake lease and device and checks that a failed release anywhere
// above the lease keeps it held until an explicit retry succeeds.
func TestDeviceOwnerLeaseCustody(t *testing.T) {
	blocked := errors.New("injected close failure")
	all := []string{"stream", "engine", "device", "lease"}
	for i, fail := range append([]string{""}, all...) {
		name := fail
		if name == "" {
			name = "none"
		}
		t.Run(name, func(t *testing.T) {
			var calls []string
			failures := map[string]error{}
			if fail != "" {
				failures[fail] = blocked
			}
			entry := recorder(&calls, failures)
			f := &fakeTB{TB: t}
			var o *testOwner
			acquired := false
			f.run(func() {
				o = ownDevice(f, loaded, func(testing.TB) (func() error, error) {
					acquired = true
					return entry("lease"), nil
				}, func() func() error {
					if !acquired {
						t.Error("device opened before the lease was acquired")
					}
					return entry("device")
				})
				o.addStream(entry("stream"))
				o.add(entry("engine"))
			})
			f.runCleanups()

			wantFirst, wantRetry := all, []string(nil)
			if fail != "" {
				wantFirst, wantRetry = all[:i], all[i-1:]
			}
			if !slices.Equal(calls, wantFirst) {
				t.Fatalf("calls = %v, want %v", calls, wantFirst)
			}
			retainedTestOwners.Lock()
			retained := slices.Contains(retainedTestOwners.owners, o)
			retainedTestOwners.Unlock()
			if fail == "" {
				if len(f.errs) != 0 || retained {
					t.Fatalf("clean release: errors=%q retained=%v", f.errs, retained)
				}
				calls = nil
				if err := o.Close(); err != nil || len(calls) != 0 {
					t.Fatalf("second Close = %v, calls %v, want nothing released twice", err, calls)
				}
				return
			}
			if len(f.errs) != 1 || !strings.Contains(f.errs[0], blocked.Error()) || !retained {
				t.Fatalf("failed %s: errors=%q retained=%v", fail, f.errs, retained)
			}

			calls = nil
			delete(failures, fail)
			for range 2 {
				if err := o.Close(); err != nil {
					t.Fatal(err)
				}
			}
			if !slices.Equal(calls, wantRetry) {
				t.Fatalf("retry calls = %v, want %v", calls, wantRetry)
			}
		})
	}
}

// TestDeviceOwnerLeaseFailure checks that a failed lease acquisition stops
// the test before any device is opened and keeps the cleanup the failure
// carries.
func TestDeviceOwnerLeaseFailure(t *testing.T) {
	refused := errors.New("injected lease failure")
	for _, tt := range []struct {
		name    string
		pending bool
	}{
		{"no cleanup", false},
		{"pending cleanup", true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var calls []string
			entry := recorder(&calls, nil)
			f := &fakeTB{TB: t}
			opened, returned := false, false
			f.run(func() {
				ownDevice(f, loaded, func(testing.TB) (func() error, error) {
					if tt.pending {
						return entry("pending"), refused
					}
					return nil, refused
				}, func() func() error {
					opened = true
					return entry("device")
				})
				returned = true
			})
			want := "acquire GPU lease: " + refused.Error()
			if opened || returned || !slices.Equal(f.errs, []string{want}) {
				t.Fatalf("opened=%v returned=%v errors=%q, want only %q", opened, returned, f.errs, want)
			}
			f.runCleanups()
			var wantCalls []string
			if tt.pending {
				wantCalls = []string{"pending"}
			}
			if !slices.Equal(calls, wantCalls) {
				t.Fatalf("calls = %v, want %v", calls, wantCalls)
			}
		})
	}
}

// TestDeviceOwnerDriverAbsent checks that when the driver library cannot be
// loaded the test skips with cudatest.RequireDevice's message before the
// lease is acquired or any device is opened.
func TestDeviceOwnerDriverAbsent(t *testing.T) {
	absent := errors.New("injected load failure")
	f := &fakeTB{TB: t}
	acquired, opened, returned := false, false, false
	f.run(func() {
		ownDevice(f, func() error { return absent }, func(testing.TB) (func() error, error) {
			acquired = true
			return func() error { return nil }, nil
		}, func() func() error {
			opened = true
			return func() error { return nil }
		})
		returned = true
	})
	want := "SKIPPED, NOT VERIFIED: the CUDA driver could not be loaded: " + absent.Error()
	if acquired || opened || returned || len(f.errs) != 0 || !slices.Equal(f.skips, []string{want}) {
		t.Fatalf("acquired=%v opened=%v returned=%v errors=%q skips=%q, want only skip %q",
			acquired, opened, returned, f.errs, f.skips, want)
	}
	if len(f.cleanups) != 0 {
		t.Fatalf("%d cleanups registered before the driver loaded", len(f.cleanups))
	}
}

// TestAcquisitionsRegisteredBeforeErrorCheck checks that in every function
// that creates a stream, builds an epilogue fixture, acquires a device or uses
// a test owner, each
// stream, engine and buffer constructor result is handed to the test owner by
// the statement right after the call, before its error is examined. Streams
// must be registered with stream and everything else with add, so that
// streams close before the engines and buffers their work reads.
func TestAcquisitionsRegisteredBeforeErrorCheck(t *testing.T) {
	constructors := []string{"NewStream", "NewEngine", "WithWorkspace", "Alloc"}
	fset := token.NewFileSet()
	streams := 0
	var checked []string
	for _, name := range []string{"cublasltx_test.go", "epilogue_test.go"} {
		f, err := parser.ParseFile(fset, name, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, decl := range f.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Body == nil || !needsOwner(fn.Body) {
				continue
			}
			checked = append(checked, fn.Name.Name)
			ast.Inspect(fn.Body, func(n ast.Node) bool {
				var list []ast.Stmt
				switch n := n.(type) {
				case *ast.BlockStmt:
					list = n.List
				case *ast.CaseClause:
					list = n.Body
				default:
					return true
				}
				for i, stmt := range list {
					as, ok := stmt.(*ast.AssignStmt)
					if !ok || len(as.Rhs) != 1 || len(as.Lhs) != 2 {
						continue
					}
					call, ok := as.Rhs[0].(*ast.CallExpr)
					if !ok || !slices.Contains(constructors, calleeName(call)) {
						continue
					}
					v, ok := as.Lhs[0].(*ast.Ident)
					if !ok || v.Name == "_" {
						t.Errorf("%s: %s result is discarded", fset.Position(as.Pos()), calleeName(call))
						continue
					}
					method := "add"
					if calleeName(call) == "NewStream" {
						method = "stream"
						streams++
					}
					if i+1 >= len(list) || !registers(list[i+1], method, v.Name) {
						t.Errorf("%s: %s result %s is not registered with owner.%s before its error is checked",
							fset.Position(as.Pos()), calleeName(call), v.Name, method)
					}
				}
				return true
			})
		}
	}
	// 13 streams in cublasltx_test.go and 2 in epilogue_test.go.
	if streams != 15 {
		t.Errorf("found %d stream acquisitions, want 15", streams)
	}
	// These acquire engines or buffers without creating a stream.
	for _, name := range []string{"TestMatMulEpilogueGELUAux", "TestWithWorkspaceBytesZero", "TestWithWorkspaceCustom"} {
		if !slices.Contains(checked, name) {
			t.Errorf("%s is not checked", name)
		}
	}
}

// needsOwner reports whether body creates a stream, builds an epilogue
// fixture, acquires a device or uses a test owner.
func needsOwner(body ast.Node) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.CallExpr:
			switch calleeName(n) {
			case "NewStream", "newEpilogueFixture", "newTestOwner", "requireDevice":
				found = true
			}
		case *ast.SelectorExpr:
			if n.Sel.Name == "owner" {
				found = true
			}
		}
		return !found
	})
	return found
}

// calleeName returns the name of the function or method call invokes,
// ignoring any explicit type arguments.
func calleeName(call *ast.CallExpr) string {
	fun := call.Fun
	switch f := fun.(type) {
	case *ast.IndexExpr:
		fun = f.X
	case *ast.IndexListExpr:
		fun = f.X
	}
	switch f := fun.(type) {
	case *ast.Ident:
		return f.Name
	case *ast.SelectorExpr:
		return f.Sel.Name
	}
	return ""
}

// registers reports whether stmt registers v with a test owner's method.
// A stream is passed to stream itself; anything else is passed to add as
// one of v's methods.
func registers(stmt ast.Stmt, method, v string) bool {
	es, ok := stmt.(*ast.ExprStmt)
	if !ok {
		return false
	}
	call, ok := es.X.(*ast.CallExpr)
	if !ok || len(call.Args) != 1 || calleeName(call) != method {
		return false
	}
	switch arg := call.Args[0].(type) {
	case *ast.Ident:
		return method == "stream" && arg.Name == v
	case *ast.SelectorExpr:
		x, ok := arg.X.(*ast.Ident)
		return method == "add" && ok && x.Name == v
	}
	return false
}

// TestDeviceAdmissionWiring checks that the converted tests take their lease
// and device from requireDevice instead of cudatest.Require, and that
// requireDevice loads the driver library, owns the canonical lease and only
// then admits device 0 and registers it.
func TestDeviceAdmissionWiring(t *testing.T) {
	fset := token.NewFileSet()
	parse := func(name string) *ast.File {
		f, err := parser.ParseFile(fset, name, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		return f
	}
	noLockOptOut := func(n ast.Node) {
		ast.Inspect(n, func(n ast.Node) bool {
			if lit, ok := n.(*ast.BasicLit); ok && (strings.Contains(lit.Value, "GPU_LOCK_DISABLED") || strings.Contains(lit.Value, "CUDATEST_NO_LOCK")) {
				t.Errorf("%s: lock opt-out %s", fset.Position(lit.Pos()), lit.Value)
			}
			return true
		})
	}

	uses := 0
	for _, name := range []string{"cublasltx_test.go", "epilogue_test.go"} {
		f := parse(name)
		noLockOptOut(f)
		ast.Inspect(f, func(n ast.Node) bool {
			switch n := n.(type) {
			case *ast.SelectorExpr:
				x, ok := n.X.(*ast.Ident)
				if ok && (x.Name == "cudatest" || x.Name == "cuda" && n.Sel.Name == "NewDevice") {
					t.Errorf("%s: %s.%s is used instead of requireDevice", fset.Position(n.Pos()), x.Name, n.Sel.Name)
				}
			case *ast.CallExpr:
				if calleeName(n) == "requireDevice" {
					uses++
				}
			}
			return true
		})
	}
	// 14 tests in cublasltx_test.go, and newEpilogueFixture and
	// TestMatMulEpilogueRowMajorBias in epilogue_test.go.
	if uses != 16 {
		t.Errorf("found %d requireDevice calls, want 16", uses)
	}

	funcs := make(map[string]*ast.FuncDecl)
	for _, decl := range parse("constructor_client_custody_test.go").Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok {
			funcs[fn.Name.Name] = fn
		}
	}
	for _, name := range []string{"requireDevice", "loadDriver", "acquireLease", "ownDevice"} {
		if funcs[name] == nil {
			t.Fatalf("%s is missing", name)
		}
		noLockOptOut(funcs[name])
	}
	if req := funcs["requireDevice"]; !hasCall(req, "cudatest", "RequireDevice", "") || hasCall(req, "cudatest", "Require", "") ||
		!hasCall(req, "", "ownDevice", "loadDriver") || !hasCall(req, "", "ownDevice", "acquireLease") {
		t.Error("requireDevice does not admit through ownDevice, loadDriver, acquireLease and cudatest.RequireDevice alone")
	}
	if load := funcs["loadDriver"]; !hasCall(load, "driver", "Load", "") || countCalls(load) != 1 {
		t.Error("loadDriver does more than driver.Load")
	}
	if !hasCall(funcs["acquireLease"], "cudalock", "Acquire", "") || !hasCall(funcs["acquireLease"], "cudalock", "WithLabel", "") {
		t.Error("acquireLease does not take the labeled canonical cudalock lease")
	}

	// ownDevice must load the driver and skip if that fails, then acquire
	// and register the lease, then stop if acquisition failed, and only then
	// open the device and register it. Nothing else may run before the
	// lease is acquired.
	steps := []string{"load", "Skipf", "acquire", "add(release)", "Fatalf", "add(open())"}
	pos := make(map[string]token.Pos)
	var early []string
	ast.Inspect(funcs["ownDevice"].Body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		step := calleeName(call)
		if step == "add" && len(call.Args) == 1 {
			switch arg := call.Args[0].(type) {
			case *ast.Ident:
				step = "add(" + arg.Name + ")"
			case *ast.CallExpr:
				step = "add(" + calleeName(arg) + "())"
			}
		}
		if _, seen := pos[step]; !seen {
			pos[step] = call.Pos()
		}
		if !slices.Contains([]string{"Helper", "load", "Skipf", "newTestOwner", "acquire"}, step) {
			early = append(early, step)
		}
		return true
	})
	for i, step := range steps {
		if pos[step] == 0 {
			t.Errorf("ownDevice: missing %s", step)
		} else if i > 0 && pos[steps[i-1]] >= pos[step] {
			t.Errorf("ownDevice: %s at %s does not follow %s at %s",
				step, fset.Position(pos[step]), steps[i-1], fset.Position(pos[steps[i-1]]))
		}
	}
	for _, step := range early {
		if pos[step] < pos["acquire"] {
			t.Errorf("ownDevice: %s runs before the lease is acquired", step)
		}
	}
}

// countCalls returns the number of calls in n.
func countCalls(n ast.Node) int {
	count := 0
	ast.Inspect(n, func(n ast.Node) bool {
		if _, ok := n.(*ast.CallExpr); ok {
			count++
		}
		return true
	})
	return count
}

// hasCall reports whether n calls pkg.name, or name when pkg is empty, and,
// if arg is not empty, passes the identifier arg.
func hasCall(n ast.Node, pkg, name, arg string) bool {
	found := false
	ast.Inspect(n, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok || found {
			return !found
		}
		switch fun := call.Fun.(type) {
		case *ast.Ident:
			if pkg != "" || fun.Name != name {
				return true
			}
		case *ast.SelectorExpr:
			x, ok := fun.X.(*ast.Ident)
			if !ok || x.Name != pkg || fun.Sel.Name != name {
				return true
			}
		default:
			return true
		}
		if arg == "" {
			found = true
		}
		for _, a := range call.Args {
			if id, ok := a.(*ast.Ident); ok && id.Name == arg {
				found = true
			}
		}
		return !found
	})
	return found
}
