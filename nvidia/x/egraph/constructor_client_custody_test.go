package egraph_test

import (
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"maps"
	"reflect"
	"runtime"
	"slices"
	"strings"
	"sync"
	"testing"
)

// nativeOwners releases a test's native resources in reverse acquisition
// order. An entry may depend on any entry acquired before it: a stream on the
// buffers and module its kernels use, and all of them on the device. A failed
// release therefore keeps that entry and every earlier one. Close stops at the
// first failure, and a later close retries from that entry without repeating
// any entry that was already released.
type nativeOwners struct {
	closers []func() error
}

// retainedNativeOwners keeps chains whose cleanup failed reachable for the
// life of the test process.
var retainedNativeOwners struct {
	sync.Mutex
	chains []*nativeOwners
}

// newNativeOwners returns an empty chain that t closes during cleanup.
// A chain that cannot be released fails t and is retained.
func newNativeOwners(t testing.TB) *nativeOwners {
	o := new(nativeOwners)
	t.Cleanup(func() {
		if err := o.close(); err != nil {
			retainedNativeOwners.Lock()
			retainedNativeOwners.chains = append(retainedNativeOwners.chains, o)
			retainedNativeOwners.Unlock()
			t.Errorf("native owners retained for the process lifetime: %v", err)
		}
	})
	return o
}

// add records the release function of a resource just acquired.
// Call it before checking the acquisition error so a partial owner is kept.
func (o *nativeOwners) add(release func() error) {
	o.closers = append(o.closers, release)
}

// addError records the owner retained by err, if err carries an io.Closer.
func (o *nativeOwners) addError(err error) {
	var c io.Closer
	if errors.As(err, &c) {
		o.add(c.Close)
	}
}

// child returns a chain for resources that t acquires on top of o's,
// typically in a subtest. The cleanup of t releases it. If that fails,
// o retries the child before releasing anything the child depends on.
// A chain is not safe for concurrent use, so subtests sharing o must not
// run in parallel.
func (o *nativeOwners) child(t testing.TB) *nativeOwners {
	c := newNativeOwners(t)
	o.add(c.close)
	return c
}

func (o *nativeOwners) close() error {
	for n := len(o.closers); n > 0; n-- {
		if err := o.closers[n-1](); err != nil {
			return fmt.Errorf("release native owner %d: %w", n, err)
		}
		o.closers = o.closers[:n-1]
	}
	return nil
}

// ownerProbe records release calls and fails the named entries.
type ownerProbe struct {
	calls    []string
	fail     map[string]error
	released map[string]int // successful releases
}

func (p *ownerProbe) closer(name string) func() error {
	return func() error {
		p.calls = append(p.calls, name)
		if err := p.fail[name]; err != nil {
			return err
		}
		if p.released == nil {
			p.released = make(map[string]int)
		}
		p.released[name]++
		return nil
	}
}

func TestNativeOwnersClose(t *testing.T) {
	blocked := errors.New("blocked")
	tests := []struct {
		name      string
		fail      string
		wantCalls []string
		wantErr   error
		retry     []string
	}{
		{
			name:      "order",
			wantCalls: []string{"stream", "buffer", "module", "device"},
		},
		{
			name:      "stream failure keeps dependencies",
			fail:      "stream",
			wantCalls: []string{"stream"},
			wantErr:   blocked,
			retry:     []string{"stream", "buffer", "module", "device"},
		},
		{
			name:      "buffer failure keeps module and device",
			fail:      "buffer",
			wantCalls: []string{"stream", "buffer"},
			wantErr:   blocked,
			retry:     []string{"buffer", "module", "device"},
		},
		{
			name:      "device failure",
			fail:      "device",
			wantCalls: []string{"stream", "buffer", "module", "device"},
			wantErr:   blocked,
			retry:     []string{"device"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &ownerProbe{fail: map[string]error{}}
			if tt.fail != "" {
				p.fail[tt.fail] = blocked
			}
			var o nativeOwners
			for _, name := range []string{"device", "module", "buffer", "stream"} {
				o.add(p.closer(name))
			}
			err := o.close()
			if !errors.Is(err, tt.wantErr) || (tt.wantErr == nil) != (err == nil) {
				t.Fatalf("close() = %v, want %v", err, tt.wantErr)
			}
			if !reflect.DeepEqual(p.calls, tt.wantCalls) {
				t.Fatalf("calls = %v, want %v", p.calls, tt.wantCalls)
			}
			// A retry resumes at the failed entry and never repeats a released one.
			p.calls, p.fail = nil, nil
			if err := o.close(); err != nil {
				t.Fatalf("retry close() = %v", err)
			}
			if !reflect.DeepEqual(p.calls, tt.retry) {
				t.Fatalf("retry calls = %v, want %v", p.calls, tt.retry)
			}
			p.calls = nil
			if err := o.close(); err != nil || p.calls != nil {
				t.Fatalf("third close() = %v, calls %v; want no-op", err, p.calls)
			}
		})
	}
}

func TestNativeOwnersChild(t *testing.T) {
	blocked := errors.New("blocked")
	p := &ownerProbe{fail: map[string]error{"stream": blocked}}
	var parent nativeOwners
	parent.add(p.closer("device"))

	tb := &cleanupTB{TB: t}
	sub := parent.child(tb)
	sub.add(p.closer("module"))
	sub.add(p.closer("stream"))
	tb.cleanup()
	if len(tb.errs) != 1 || !strings.Contains(tb.errs[0], "blocked") {
		t.Fatalf("subtest errors = %q, want one blocked failure", tb.errs)
	}
	if want := []string{"stream"}; !reflect.DeepEqual(p.calls, want) {
		t.Fatalf("subtest calls = %v, want %v", p.calls, want)
	}

	// The parent must not release the device under a child that is still held.
	p.calls = nil
	if err := parent.close(); !errors.Is(err, blocked) {
		t.Fatalf("parent close() = %v, want %v", err, blocked)
	}
	if want := []string{"stream"}; !reflect.DeepEqual(p.calls, want) {
		t.Fatalf("parent calls = %v, want %v", p.calls, want)
	}

	p.calls, p.fail = nil, nil
	if err := parent.close(); err != nil {
		t.Fatalf("parent retry close() = %v", err)
	}
	if want := []string{"stream", "module", "device"}; !reflect.DeepEqual(p.calls, want) {
		t.Fatalf("parent retry calls = %v, want %v", p.calls, want)
	}
}

func TestNativeOwnersCleanupRetains(t *testing.T) {
	blocked := errors.New("blocked")
	p := &ownerProbe{fail: map[string]error{"stream": blocked}}
	tb := &cleanupTB{TB: t}
	o := newNativeOwners(tb)
	o.add(p.closer("device"))
	o.add(p.closer("stream"))
	tb.cleanup()
	if len(tb.errs) != 1 || !strings.Contains(tb.errs[0], "release native owner 2: blocked") {
		t.Fatalf("errors = %q, want stream failure", tb.errs)
	}
	retainedNativeOwners.Lock()
	retained := slices.Contains(retainedNativeOwners.chains, o)
	retainedNativeOwners.Unlock()
	if !retained {
		t.Fatal("failed chain was not retained")
	}
	if want := []string{"stream"}; !reflect.DeepEqual(p.calls, want) {
		t.Fatalf("calls = %v, want %v", p.calls, want)
	}

	p.calls, p.fail = nil, nil
	tb = &cleanupTB{TB: t}
	o = newNativeOwners(tb)
	o.add(p.closer("device"))
	tb.cleanup()
	if tb.errs != nil || !reflect.DeepEqual(p.calls, []string{"device"}) {
		t.Fatalf("errors = %q, calls = %v; want clean release", tb.errs, p.calls)
	}
}

func TestNativeOwnersAddError(t *testing.T) {
	cause := errors.New("load")
	closes := 0
	owner := &closerError{err: cause, close: func() error { closes++; return nil }}
	var o nativeOwners
	o.addError(nil)
	o.addError(cause)
	o.addError(fmt.Errorf("compile: %w", owner))
	if len(o.closers) != 1 {
		t.Fatalf("recorded %d owners, want 1", len(o.closers))
	}
	if err := o.close(); err != nil || closes != 1 {
		t.Fatalf("close() = %v, closes = %d; want nil, 1", err, closes)
	}
}

func TestNativeOwnersLease(t *testing.T) {
	blocked := errors.New("blocked")
	all := []string{"stream", "buffer", "module", "device", "lease"}
	for _, fail := range all {
		t.Run(fail, func(t *testing.T) {
			p := &ownerProbe{fail: map[string]error{fail: blocked}}
			tb := &cleanupTB{TB: t}
			o := leaseOwners(tb, func() (func() error, error) { return p.closer("lease"), nil })
			for _, name := range []string{"device", "module", "buffer", "stream"} {
				o.add(p.closer(name))
			}
			tb.cleanup()
			want := all[:slices.Index(all, fail)+1]
			if !reflect.DeepEqual(p.calls, want) {
				t.Fatalf("calls = %v, want %v", p.calls, want)
			}
			if len(tb.errs) != 1 || !strings.Contains(tb.errs[0], "blocked") {
				t.Fatalf("errors = %q, want one blocked failure", tb.errs)
			}
			// The lease stays held until every dependent has been released.
			p.calls, p.fail = nil, nil
			if err := o.close(); err != nil {
				t.Fatalf("retry close() = %v", err)
			}
			if want := all[len(want)-1:]; !reflect.DeepEqual(p.calls, want) {
				t.Fatalf("retry calls = %v, want %v", p.calls, want)
			}
		})
	}
}

func TestNativeOwnersLeaseChild(t *testing.T) {
	blocked := errors.New("blocked")
	p := &ownerProbe{fail: map[string]error{"stream": blocked}}
	tb := &cleanupTB{TB: t}
	o := leaseOwners(tb, func() (func() error, error) { return p.closer("lease"), nil })
	o.add(p.closer("device"))
	sub := o.child(tb)
	sub.add(p.closer("buffer"))
	sub.add(p.closer("stream"))
	tb.cleanup()
	if want := []string{"stream", "stream"}; !reflect.DeepEqual(p.calls, want) {
		t.Fatalf("calls = %v, want %v (child, then parent retry, no device or lease)", p.calls, want)
	}
	if len(tb.errs) != 2 {
		t.Fatalf("errors = %q, want child and parent failures", tb.errs)
	}
}

func TestNativeOwnersLeaseAcquireError(t *testing.T) {
	cause := errors.New("cancel")
	closes := 0
	owner := &closerError{err: cause, close: func() error { closes++; return nil }}
	tb := &cleanupTB{TB: t}
	done := make(chan bool)
	go func() {
		defer close(done)
		leaseOwners(tb, func() (func() error, error) { return nil, owner })
		done <- true
	}()
	if <-done {
		t.Fatal("leaseOwners returned after an acquisition error")
	}
	tb.cleanup()
	if closes != 1 {
		t.Fatalf("acquisition cleanup ran %d times, want 1", closes)
	}
	if len(tb.errs) != 1 || !strings.Contains(tb.errs[0], "acquire GPU lease: cancel") {
		t.Fatalf("errors = %q, want acquisition failure", tb.errs)
	}
}

// fakeFixture runs ownedDevice with fake steps on its own goroutine, as
// requireNativeDevice runs it with the real ones. It reports the chain, how
// many times the lease was acquired, and whether ownedDevice returned.
func fakeFixture(tb testing.TB, p *ownerProbe, loadErr error, require func(testing.TB) string) (*nativeOwners, int, bool) {
	var owners *nativeOwners
	acquired := 0
	returned := runTB(func() {
		_, owners = ownedDevice(tb,
			func() error { return loadErr },
			func() (func() error, error) {
				acquired++
				return p.closer("lease"), nil
			},
			require,
			func(dev string) error { return p.closer(dev)() },
		)
	})
	return owners, acquired, returned
}

func fakeDevice(testing.TB) string { return "device" }

// runTB runs f on a new goroutine, as testing runs a test, and reports
// whether f returned rather than stopping with runtime.Goexit.
func runTB(f func()) bool {
	done := make(chan bool)
	go func() {
		defer close(done)
		f()
		done <- true
	}()
	return <-done
}

// TestNativeOwnersFixture drives the fixture sequence of the value tests
// with fake steps: lease, device, then a module, two buffers and a stream,
// either on the fixture chain or, as in u2, on a subtest's child chain.
func TestNativeOwnersFixture(t *testing.T) {
	blocked := errors.New("blocked")
	order := []string{"stream", "buffer2", "buffer1", "module", "device", "lease"}
	tests := []struct {
		name  string
		fail  string // entry whose first release fails
		child bool
		want  []string // releases attempted by cleanup
	}{
		{name: "success", want: order},
		{name: "success child", child: true, want: order},
		{name: "stream", fail: "stream", want: []string{"stream"}},
		{name: "child stream", fail: "stream", child: true, want: []string{"stream", "stream"}},
		{name: "device", fail: "device", want: order[:5]},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &ownerProbe{fail: map[string]error{}}
			if tt.fail != "" {
				p.fail[tt.fail] = blocked
			}
			tb := &cleanupTB{TB: t}
			owners, acquired, ok := fakeFixture(tb, p, nil, fakeDevice)
			if !ok || acquired != 1 {
				t.Fatalf("ownedDevice returned %v after %d acquisitions, want true, 1", ok, acquired)
			}
			sub, subTB := owners, tb
			if tt.child {
				subTB = &cleanupTB{TB: t}
				sub = owners.child(subTB)
			}
			for _, name := range []string{"module", "buffer1", "buffer2", "stream"} {
				sub.add(p.closer(name))
			}
			if tt.child {
				subTB.cleanup()
			}
			tb.cleanup()
			if !reflect.DeepEqual(p.calls, tt.want) {
				t.Fatalf("cleanup calls = %v, want %v", p.calls, tt.want)
			}
			once := make(map[string]int)
			for _, name := range order {
				once[name] = 1
			}
			if tt.fail == "" {
				if tb.errs != nil || subTB.errs != nil {
					t.Fatalf("errors = %q, %q; want none", tb.errs, subTB.errs)
				}
				if !reflect.DeepEqual(p.released, once) {
					t.Fatalf("released = %v, want each exactly once", p.released)
				}
			} else {
				if p.released["device"] != 0 || p.released["lease"] != 0 {
					t.Fatalf("released = %v, want device and lease held", p.released)
				}
				if len(tb.errs) != 1 {
					t.Fatalf("errors = %q, want one cleanup failure", tb.errs)
				}
				// An explicit retry resumes at the failed entry.
				p.calls, p.fail = nil, nil
				if err := owners.close(); err != nil {
					t.Fatalf("retry close() = %v", err)
				}
				if want := order[slices.Index(order, tt.fail):]; !reflect.DeepEqual(p.calls, want) {
					t.Fatalf("retry calls = %v, want %v", p.calls, want)
				}
				if !reflect.DeepEqual(p.released, once) {
					t.Fatalf("released after retry = %v, want each exactly once", p.released)
				}
			}
			p.calls = nil
			tb.cleanup()
			subTB.cleanup()
			if p.calls != nil {
				t.Fatalf("second cleanup released %v, want nothing", p.calls)
			}
		})
	}
}

// TestNativeOwnersFixtureRequireStops checks that a skip or failure while
// requiring the device releases the lease exactly once and no device.
func TestNativeOwnersFixtureRequireStops(t *testing.T) {
	for _, stop := range []string{"skip", "fatal"} {
		t.Run(stop, func(t *testing.T) {
			p := &ownerProbe{}
			tb := &cleanupTB{TB: t}
			_, acquired, ok := fakeFixture(tb, p, nil, func(tb testing.TB) string {
				if stop == "skip" {
					tb.Skipf("no device")
				}
				tb.Fatalf("cuInit failed")
				return "device"
			})
			if ok || acquired != 1 {
				t.Fatalf("ownedDevice returned %v after %d acquisitions, want false, 1", ok, acquired)
			}
			tb.cleanup()
			tb.cleanup()
			if want := []string{"lease"}; !reflect.DeepEqual(p.calls, want) || p.released["lease"] != 1 {
				t.Fatalf("calls = %v, released = %v; want the lease once", p.calls, p.released)
			}
			want, got := []string{"cuInit failed"}, tb.errs
			if stop == "skip" {
				want, got = []string{"no device"}, tb.skips
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("reported %q, want %q", got, want)
			}
		})
	}
}

// TestNativeOwnersFixtureLoadSkips checks that a machine without the driver
// library skips before the lease is acquired.
func TestNativeOwnersFixtureLoadSkips(t *testing.T) {
	p := &ownerProbe{}
	tb := &cleanupTB{TB: t}
	_, acquired, ok := fakeFixture(tb, p, errors.New("no libcuda"), fakeDevice)
	if ok || acquired != 0 || len(tb.cleanups) != 0 {
		t.Fatalf("returned %v, acquired %d, %d cleanups; want false, 0, 0", ok, acquired, len(tb.cleanups))
	}
	want := []string{"SKIPPED, NOT VERIFIED: the CUDA driver could not be loaded: no libcuda"}
	if !reflect.DeepEqual(tb.skips, want) || tb.errs != nil || p.calls != nil {
		t.Fatalf("skips = %q, errors = %q, calls = %v; want driver skip only", tb.skips, tb.errs, p.calls)
	}
}

// TestNativeOwnerCallSites checks the source of the CUDA value tests and their
// fixture. The fixture must load only the driver library before it acquires an
// owned GPU lease, hold that lease below the device, and the tests must obtain
// both from it rather than from cudatest.Require. Each test
// must register its stream, allocation, module, and owner-carrying errors
// before the error checks, on the one chain that owns its buffers, create the
// stream after those buffers, unload the module in the device context, and
// release nothing through defer.
func TestNativeOwnerCallSites(t *testing.T) {
	files := []string{
		"composite_cuda_test.go",
		"fp8_cuda_test.go",
		"half_cuda_test.go",
		"iq2xxs_cuda_test.go",
		"q2k_cuda_test.go",
		"rmsnorm_cuda_test.go",
		"swiglu_cuda_test.go",
		"u2_cuda_test.go",
		"constructor_client_native_test.go",
	}
	counts := make(map[string]int)
	for _, name := range files {
		f, err := parser.ParseFile(token.NewFileSet(), name, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		ast.Inspect(f, func(n ast.Node) bool {
			switch n := n.(type) {
			case *ast.BasicLit:
				if strings.Contains(n.Value, "GPU_LOCK_DISABLED") || strings.Contains(n.Value, "CUDATEST_NO_LOCK") {
					t.Errorf("%s: disables the GPU lock", name)
				}
			case *ast.CallExpr:
				switch exprString(n.Fun) {
				case "cudatest.Require":
					t.Errorf("%s: cudatest.Require releases its lease outside the owner chain", name)
				case "requireNativeDevice":
					counts["requireNativeDevice"]++
				}
			case *ast.DeferStmt:
				if sel, ok := n.Call.Fun.(*ast.SelectorExpr); ok && (sel.Sel.Name == "Close" || sel.Sel.Name == "Free" || sel.Sel.Name == "Release") {
					t.Errorf("%s: defer %s releases outside the owner chain", name, exprString(sel))
				}
				if id, ok := n.Call.Fun.(*ast.Ident); ok && id.Name == "free" {
					t.Errorf("%s: defer free releases outside the owner chain", name)
				}
			case *ast.FuncDecl:
				switch n.Name.Name {
				case "requireNativeDevice":
					counts["fixture"]++
					checkDeviceFixture(t, name, n.Body.List)
				case "ownedDevice":
					counts["owned"]++
					checkOwnedDevice(t, name, n.Body.List)
				case "leaseOwners":
					counts["lease"]++
					checkLeaseFixture(t, name, n.Body.List)
				}
			case *ast.BlockStmt:
				checkOwnerBlock(t, name, n.List, counts)
			}
			return true
		})
	}
	want := map[string]int{
		"NewStream": 10, "Alloc": 1, "Compile": 10, "Load": 10,
		"requireNativeDevice": 10, "fixture": 1, "owned": 1, "lease": 1,
	}
	if !reflect.DeepEqual(counts, want) {
		t.Errorf("acquisitions = %v, want %v", counts, want)
	}
}

// checkDeviceFixture checks that requireNativeDevice runs ownedDevice with
// the real steps: driver.Load, a lease whose release is the Release of the
// handle cudalock.Acquire returned, cudatest.RequireDevice, and Device.Close.
func checkDeviceFixture(t *testing.T, name string, list []ast.Stmt) {
	var call *ast.CallExpr
	for _, s := range list {
		if ret, ok := s.(*ast.ReturnStmt); ok && len(ret.Results) == 1 {
			if c, ok := ret.Results[0].(*ast.CallExpr); ok && exprString(c.Fun) == "ownedDevice" {
				call = c
			}
		}
	}
	if call == nil || len(call.Args) != 5 {
		t.Errorf("%s: requireNativeDevice does not return ownedDevice(t, load, acquire, require, closeDevice)", name)
		return
	}
	if !returnsOnly(call.Args[1], "driver.Load") {
		t.Errorf("%s: fixture load step is not driver.Load", name)
	}
	if !releasesAcquiredHandle(call.Args[2]) {
		t.Errorf("%s: fixture lease release is not the Release of the cudalock.Acquire handle", name)
	}
	if exprString(call.Args[3]) != "cudatest.RequireDevice" {
		t.Errorf("%s: fixture require step is not cudatest.RequireDevice", name)
	}
	if !isDeviceClose(call.Args[4]) {
		t.Errorf("%s: fixture device release is not (*cuda.Device).Close", name)
	}
}

// returnsOnly reports whether e is `func() error { return fn(...) }`.
func returnsOnly(e ast.Expr, fn string) bool {
	lit, ok := e.(*ast.FuncLit)
	if !ok || len(lit.Body.List) != 1 {
		return false
	}
	ret, ok := lit.Body.List[0].(*ast.ReturnStmt)
	if !ok || len(ret.Results) != 1 {
		return false
	}
	call, ok := ret.Results[0].(*ast.CallExpr)
	return ok && exprString(call.Fun) == fn
}

// releasesAcquiredHandle reports whether e is a function literal that binds
// h from cudalock.Acquire and finally returns h.Release, nil.
func releasesAcquiredHandle(e ast.Expr) bool {
	lit, ok := e.(*ast.FuncLit)
	if !ok || len(lit.Body.List) == 0 {
		return false
	}
	list := lit.Body.List
	as, ok := list[0].(*ast.AssignStmt)
	if !ok || len(as.Lhs) != 2 || len(as.Rhs) != 1 || !containsCall(as.Rhs[0], "cudalock.Acquire") {
		return false
	}
	h := exprString(as.Lhs[0])
	ret, ok := list[len(list)-1].(*ast.ReturnStmt)
	return ok && h != "" && len(ret.Results) == 2 && exprString(ret.Results[0]) == h+".Release" && exprString(ret.Results[1]) == "nil"
}

// isDeviceClose reports whether e is the method expression (*cuda.Device).Close.
func isDeviceClose(e ast.Expr) bool {
	sel, ok := e.(*ast.SelectorExpr)
	if !ok || sel.Sel.Name != "Close" {
		return false
	}
	paren, ok := sel.X.(*ast.ParenExpr)
	if !ok {
		return false
	}
	star, ok := paren.X.(*ast.StarExpr)
	return ok && exprString(star.X) == "cuda.Device"
}

// checkOwnedDevice checks that ownedDevice loads the driver library, and
// skips if that fails, before it acquires the lease; acquires the lease
// before it requires the device; and adds the device release to the lease
// chain immediately.
func checkOwnedDevice(t *testing.T, name string, list []ast.Stmt) {
	load, lease, device := -1, -1, -1
	for i, s := range list {
		if is, ok := s.(*ast.IfStmt); ok && is.Init != nil && containsCall(is.Init, "load") && containsCall(is.Body, "t.Skipf") {
			load = i
		}
		as, ok := s.(*ast.AssignStmt)
		if !ok || len(as.Rhs) != 1 {
			continue
		}
		call, ok := as.Rhs[0].(*ast.CallExpr)
		if !ok {
			continue
		}
		switch exprString(call.Fun) {
		case "leaseOwners":
			lease = i
			if len(call.Args) != 2 || exprString(call.Args[1]) != "acquire" {
				t.Errorf("%s: lease chain is not built from the acquire step", name)
			}
		case "require":
			device = i
			chain := ""
			if lease >= 0 {
				chain = exprString(list[lease].(*ast.AssignStmt).Lhs[0])
			}
			if i+1 >= len(list) || !isDeviceRelease(list[i+1], chain, exprString(as.Lhs[0])) {
				t.Errorf("%s: device is not added to the lease chain immediately", name)
			}
		}
	}
	if load < 0 || lease < 0 || device < 0 || load > lease || lease > device {
		t.Errorf("%s: fixture order is load (statement %d), lease (%d), device (%d); want load, lease, device", name, load, lease, device)
	}
}

// isDeviceRelease reports whether s is
// `chain.add(func() error { return closeDevice(dev) })`.
func isDeviceRelease(s ast.Stmt, chain, dev string) bool {
	es, ok := s.(*ast.ExprStmt)
	if !ok {
		return false
	}
	call, ok := es.X.(*ast.CallExpr)
	if !ok || exprString(call.Fun) != chain+".add" || len(call.Args) != 1 || !returnsOnly(call.Args[0], "closeDevice") {
		return false
	}
	inner := call.Args[0].(*ast.FuncLit).Body.List[0].(*ast.ReturnStmt).Results[0].(*ast.CallExpr)
	return len(inner.Args) == 1 && exprString(inner.Args[0]) == dev
}

// checkLeaseFixture checks that leaseOwners creates the chain before it
// acquires, keeps an owner-carrying acquisition error, and adds the release
// after the error check.
func checkLeaseFixture(t *testing.T, name string, list []ast.Stmt) {
	chain := -1
	for i, s := range list {
		as, ok := s.(*ast.AssignStmt)
		if !ok || len(as.Rhs) != 1 {
			continue
		}
		call, ok := as.Rhs[0].(*ast.CallExpr)
		if !ok {
			continue
		}
		switch exprString(call.Fun) {
		case "newNativeOwners":
			chain = i
		case "acquire":
			if chain < 0 {
				t.Errorf("%s: lease acquired before its chain exists", name)
				continue
			}
			owners := exprString(list[chain].(*ast.AssignStmt).Lhs[0])
			release := exprString(as.Lhs[0])
			if i+3 >= len(list) || !isErrorCapture(list[i+1]) || !isAddOn(list[i+3], owners, release) {
				t.Errorf("%s: lease release is not added to the chain", name)
			}
			return
		}
	}
	t.Errorf("%s: no lease acquisition", name)
}

// checkOwnerBlock checks the acquisitions in one statement list and
// counts them by method in counts.
func checkOwnerBlock(t *testing.T, name string, list []ast.Stmt, counts map[string]int) {
	lastSlice := -1
	streams := false
	for i, s := range list {
		if containsCall(s, "deviceSlice") {
			lastSlice = i
		}
		if as, ok := s.(*ast.AssignStmt); ok && len(as.Rhs) == 1 && isCall(as.Rhs[0], "NewStream") {
			streams = true
		}
	}
	if streams {
		checkOneChain(t, name, list)
	}
	for i, s := range list {
		as, ok := s.(*ast.AssignStmt)
		if !ok || len(as.Rhs) != 1 {
			continue
		}
		switch {
		case isCall(as.Rhs[0], "NewStream"):
			counts["NewStream"]++
			if i < lastSlice {
				t.Errorf("%s: stream created before the buffers its kernels use", name)
			}
			if i+1 >= len(list) || !isStreamCapture(list[i+1]) {
				t.Errorf("%s: stream is not captured before its error check", name)
			}
		case isCall(as.Rhs[0], "Alloc"):
			counts["Alloc"]++
			if i+1 >= len(list) || !isAdd(list[i+1], exprString(as.Lhs[0])+".Free") {
				t.Errorf("%s: allocation is not captured before its error check", name)
			}
		case isCall(as.Rhs[0], "Compile"):
			counts["Compile"]++
			if i+1 >= len(list) || !isErrorCapture(list[i+1]) {
				t.Errorf("%s: owner in nvrtcx.Compile error is not captured before its error check", name)
			}
		case isCall(as.Rhs[0], "Load"):
			counts["Load"]++
			if i+1 >= len(list) || !isErrorCapture(list[i+1]) {
				t.Errorf("%s: owner in compiled.Load error is not captured before its error check", name)
			}
			if i+3 >= len(list) || !isModuleRelease(list[i+3], exprString(as.Lhs[0])) {
				t.Errorf("%s: module is not unloaded with dev.WithContext(%s.Close) after its error check", name, exprString(as.Lhs[0]))
			}
		}
	}
}

// checkOneChain checks that a block creating a stream registers every owner
// on one chain, the chain passed to deviceSlice, and that this chain comes
// from requireNativeDevice or child in the same block.
func checkOneChain(t *testing.T, name string, list []ast.Stmt) {
	chains := make(map[string]bool)
	defined := ""
	for _, s := range list {
		if as, ok := s.(*ast.AssignStmt); ok && len(as.Rhs) == 1 {
			if call, ok := as.Rhs[0].(*ast.CallExpr); ok {
				switch {
				case exprString(call.Fun) == "requireNativeDevice" && len(as.Lhs) == 2:
					defined = exprString(as.Lhs[1])
				case strings.HasSuffix(exprString(call.Fun), ".child") && len(as.Lhs) == 1:
					defined = exprString(as.Lhs[0])
				}
			}
		}
		ast.Inspect(s, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			fun := exprString(call.Fun)
			switch {
			case fun == "deviceSlice" && len(call.Args) == 4:
				chains[exprString(call.Args[1])] = true
			case strings.HasSuffix(fun, ".add"), strings.HasSuffix(fun, ".addError"):
				chains[fun[:strings.LastIndex(fun, ".")]] = true
			}
			return true
		})
	}
	if len(chains) != 1 || !chains[defined] {
		t.Errorf("%s: owners registered on chains %v, want only the block's chain %q", name, slices.Sorted(maps.Keys(chains)), defined)
	}
}

// isModuleRelease reports whether s is
// `owners.add(func() error { return dev.WithContext(mod.Close) })`.
func isModuleRelease(s ast.Stmt, mod string) bool {
	es, ok := s.(*ast.ExprStmt)
	if !ok {
		return false
	}
	call, ok := es.X.(*ast.CallExpr)
	if !ok || !strings.HasSuffix(exprString(call.Fun), ".add") || len(call.Args) != 1 {
		return false
	}
	lit, ok := call.Args[0].(*ast.FuncLit)
	if !ok || len(lit.Body.List) != 1 {
		return false
	}
	ret, ok := lit.Body.List[0].(*ast.ReturnStmt)
	if !ok || len(ret.Results) != 1 {
		return false
	}
	run, ok := ret.Results[0].(*ast.CallExpr)
	return ok && exprString(run.Fun) == "dev.WithContext" && len(run.Args) == 1 && exprString(run.Args[0]) == mod+".Close"
}

// isErrorCapture reports whether s is `owners.addError(err)`.
func isErrorCapture(s ast.Stmt) bool {
	es, ok := s.(*ast.ExprStmt)
	if !ok {
		return false
	}
	call, ok := es.X.(*ast.CallExpr)
	return ok && strings.HasSuffix(exprString(call.Fun), ".addError") && len(call.Args) == 1 && exprString(call.Args[0]) == "err"
}

// isStreamCapture reports whether s is `if stream != nil { owners.add(stream.Close) }`.
func isStreamCapture(s ast.Stmt) bool {
	is, ok := s.(*ast.IfStmt)
	if !ok || is.Init != nil || is.Else != nil || len(is.Body.List) != 1 {
		return false
	}
	cond, ok := is.Cond.(*ast.BinaryExpr)
	if !ok || cond.Op != token.NEQ || exprString(cond.X) != "stream" || exprString(cond.Y) != "nil" {
		return false
	}
	return isAdd(is.Body.List[0], "stream.Close")
}

// isAdd reports whether s is `owners.add(release)` on any chain.
func isAdd(s ast.Stmt, release string) bool {
	es, ok := s.(*ast.ExprStmt)
	if !ok {
		return false
	}
	call, ok := es.X.(*ast.CallExpr)
	return ok && strings.HasSuffix(exprString(call.Fun), ".add") && len(call.Args) == 1 && exprString(call.Args[0]) == release
}

// isAddOn reports whether s is `chain.add(release)`.
func isAddOn(s ast.Stmt, chain, release string) bool {
	es, ok := s.(*ast.ExprStmt)
	if !ok {
		return false
	}
	call, ok := es.X.(*ast.CallExpr)
	return ok && exprString(call.Fun) == chain+".add" && len(call.Args) == 1 && exprString(call.Args[0]) == release
}

func isCall(e ast.Expr, method string) bool {
	call, ok := e.(*ast.CallExpr)
	if !ok {
		return false
	}
	fun := call.Fun
	if ix, ok := fun.(*ast.IndexExpr); ok {
		fun = ix.X
	}
	return strings.HasSuffix(exprString(fun), "."+method)
}

func containsCall(n ast.Node, fn string) bool {
	found := false
	ast.Inspect(n, func(n ast.Node) bool {
		if call, ok := n.(*ast.CallExpr); ok {
			if id, ok := call.Fun.(*ast.IndexExpr); ok && exprString(id.X) == fn {
				found = true
			}
			if exprString(call.Fun) == fn {
				found = true
			}
		}
		return !found
	})
	return found
}

// exprString renders an identifier or selector chain such as owners.add.
func exprString(e ast.Expr) string {
	switch e := e.(type) {
	case *ast.Ident:
		return e.Name
	case *ast.SelectorExpr:
		return exprString(e.X) + "." + e.Sel.Name
	}
	return ""
}

// cleanupTB collects cleanups and errors so tests can observe a failing
// cleanup without failing themselves.
type cleanupTB struct {
	testing.TB
	cleanups []func()
	errs     []string
	skips    []string
}

func (tb *cleanupTB) Cleanup(f func()) { tb.cleanups = append(tb.cleanups, f) }

func (tb *cleanupTB) Errorf(format string, args ...any) {
	tb.errs = append(tb.errs, fmt.Sprintf(format, args...))
}

// Fatalf records the failure and stops the calling goroutine, as t.Fatalf does.
func (tb *cleanupTB) Fatalf(format string, args ...any) {
	tb.Errorf(format, args...)
	runtime.Goexit()
}

// Skipf records the skip and stops the calling goroutine, as t.Skipf does.
func (tb *cleanupTB) Skipf(format string, args ...any) {
	tb.skips = append(tb.skips, fmt.Sprintf(format, args...))
	runtime.Goexit()
}

func (tb *cleanupTB) cleanup() {
	for i := len(tb.cleanups) - 1; i >= 0; i-- {
		tb.cleanups[i]()
	}
}

// closerError is an error that retains an owner, like nvrtcx load failures.
type closerError struct {
	err   error
	close func() error
}

func (e *closerError) Error() string { return e.err.Error() }
func (e *closerError) Unwrap() error { return e.err }
func (e *closerError) Close() error  { return e.close() }
