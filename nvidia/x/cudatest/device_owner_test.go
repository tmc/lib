package cudatest

import (
	"context"
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cuda"
)

// fakeBackend records the operations that admission and custody perform.
// Its devices and streams are inert values identified by pointer; nothing
// calls into a driver.
type fakeBackend struct {
	count   int
	reject  map[int]bool     // ordinals the requirement rejects
	fail    map[string]error // events that fail
	log     []string
	devices map[*cuda.Device]int
	streams map[*cuda.Stream]string
	hooks   map[string]func() // run when an event is logged
}

func newFakeBackend(count int) *fakeBackend {
	return &fakeBackend{
		count:   count,
		reject:  make(map[int]bool),
		fail:    make(map[string]error),
		devices: make(map[*cuda.Device]int),
		streams: make(map[*cuda.Stream]string),
		hooks:   make(map[string]func()),
	}
}

func (f *fakeBackend) call(event string) error {
	f.log = append(f.log, event)
	if hook := f.hooks[event]; hook != nil {
		hook()
	}
	return f.fail[event]
}

func (f *fakeBackend) ops() *operations {
	return &operations{
		load: func() error { return f.call("load") },
		acquire: func(ctx context.Context, label string) (func() error, error) {
			if err := f.call("acquire " + label); err != nil {
				return nil, err
			}
			return func() error { return f.call("release lease") }, nil
		},
		init: func() error { return f.call("init") },
		count: func() (int, error) {
			return f.count, f.call("count")
		},
		open: func(i int) (*cuda.Device, error) {
			if err := f.call(fmt.Sprintf("open %d", i)); err != nil {
				return nil, err
			}
			d := new(cuda.Device)
			f.devices[d] = i
			return d, nil
		},
		closeDevice: func(d *cuda.Device) error {
			return f.call(fmt.Sprintf("close device %d", f.devices[d]))
		},
		closeStream: func(s *cuda.Stream) error {
			return f.call("close stream " + f.streams[s])
		},
	}
}

// require returns a Requirement that records its evaluation and rejects the
// ordinals in f.reject.
func (f *fakeBackend) require() Requirement {
	return func(d *cuda.Device) error {
		i := f.devices[d]
		if err := f.call(fmt.Sprintf("require %d", i)); err != nil {
			return err
		}
		if f.reject[i] {
			return fmt.Errorf("device %d rejected", i)
		}
		return nil
	}
}

func (f *fakeBackend) stream(name string) *cuda.Stream {
	s := new(cuda.Stream)
	f.streams[s] = name
	return s
}

func (f *fakeBackend) release(name string) func() error {
	return func() error { return f.call("release " + name) }
}

// admitted runs admission against a backend with one acceptable device.
func admitted(t *testing.T, f *fakeBackend) *DeviceOwner {
	t.Helper()
	dev, o, err := admit(context.Background(), "T", f.ops(), []Requirement{f.require()})
	if err != nil || dev == nil {
		t.Fatalf("admit = %v, %v; want device", dev, err)
	}
	f.log = nil
	return o
}

func checkLog(t *testing.T, f *fakeBackend, want ...string) {
	t.Helper()
	if !slices.Equal(f.log, want) {
		t.Errorf("events:\n\t%s\nwant:\n\t%s", strings.Join(f.log, "\n\t"), strings.Join(want, "\n\t"))
	}
	f.log = nil
}

func isRetained(o *DeviceOwner) bool {
	retained.mu.Lock()
	defer retained.mu.Unlock()
	return slices.Contains(retained.owners, o)
}

func TestDeviceOwnerAdmitOrder(t *testing.T) {
	f := newFakeBackend(3)
	f.reject[0] = true
	dev, o, err := admit(context.Background(), "T", f.ops(), []Requirement{f.require()})
	if err != nil {
		t.Fatal(err)
	}
	if got := f.devices[dev]; got != 1 {
		t.Errorf("admitted device %d, want 1", got)
	}
	checkLog(t, f,
		"load", "acquire T", "init", "count",
		"open 0", "require 0", "close device 0",
		"open 1", "require 1")

	o.Defer(f.release("buffer"))
	o.DeferStream(f.stream("s"))
	if err := o.Close(); err != nil {
		t.Fatal(err)
	}
	checkLog(t, f, "close stream s", "release buffer", "close device 1", "release lease")
	if err := o.Close(); err != nil {
		t.Fatal(err)
	}
	if err := o.cleanup(); err != nil {
		t.Fatal(err)
	}
	checkLog(t, f)
}

func TestDeviceOwnerAdmitOutcomes(t *testing.T) {
	errFail := errors.New("injected")
	tests := []struct {
		name     string
		count    int
		reject   []int
		fail     string // event that fails
		skip     bool
		admitLog []string
		closeLog []string
	}{
		{
			name:     "driver absent",
			count:    1,
			fail:     "load",
			skip:     true,
			admitLog: []string{"load"},
		},
		{
			name:     "acquire",
			count:    1,
			fail:     "acquire T",
			admitLog: []string{"load", "acquire T"},
		},
		{
			name:     "init",
			count:    1,
			fail:     "init",
			admitLog: []string{"load", "acquire T", "init"},
			closeLog: []string{"release lease"},
		},
		{
			name:     "count",
			count:    1,
			fail:     "count",
			admitLog: []string{"load", "acquire T", "init", "count"},
			closeLog: []string{"release lease"},
		},
		{
			name:     "no device",
			count:    0,
			skip:     true,
			admitLog: []string{"load", "acquire T", "init", "count"},
			closeLog: []string{"release lease"},
		},
		{
			name:     "open",
			count:    2,
			fail:     "open 0",
			admitLog: []string{"load", "acquire T", "init", "count", "open 0"},
			closeLog: []string{"release lease"},
		},
		{
			name:     "open after rejection",
			count:    2,
			reject:   []int{0},
			fail:     "open 1",
			admitLog: []string{"load", "acquire T", "init", "count", "open 0", "require 0", "close device 0", "open 1"},
			closeLog: []string{"release lease"},
		},
		{
			name:     "all rejected",
			count:    2,
			reject:   []int{0, 1},
			skip:     true,
			admitLog: []string{"load", "acquire T", "init", "count", "open 0", "require 0", "close device 0", "open 1", "require 1", "close device 1"},
			closeLog: []string{"release lease"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newFakeBackend(tt.count)
			for _, i := range tt.reject {
				f.reject[i] = true
			}
			if tt.fail != "" {
				f.fail[tt.fail] = errFail
			}
			dev, o, err := admit(context.Background(), "T", f.ops(), []Requirement{f.require()})
			if dev != nil || o == nil || err == nil {
				t.Fatalf("admit = %v, %v, %v; want nil device, owner, error", dev, o, err)
			}
			if skip := errors.Is(err, errNotVerified); skip != tt.skip {
				t.Errorf("admit error %q: skip = %v, want %v", err, skip, tt.skip)
			}
			if tt.fail != "" && !errors.Is(err, errFail) {
				t.Errorf("admit error %q does not wrap the injected failure", err)
			}
			checkLog(t, f, tt.admitLog...)
			if err := o.Close(); err != nil {
				t.Fatal(err)
			}
			checkLog(t, f, tt.closeLog...)
		})
	}
}

func TestDeviceOwnerRejectedCloseFailure(t *testing.T) {
	errClose := errors.New("close failed")
	f := newFakeBackend(2)
	f.reject[0] = true
	f.reject[1] = true
	f.fail["close device 0"] = errClose
	_, o, err := admit(context.Background(), "T", f.ops(), []Requirement{f.require()})
	if !errors.Is(err, errClose) || errors.Is(err, errNotVerified) {
		t.Fatalf("admit error = %v, want failure wrapping %v", err, errClose)
	}
	checkLog(t, f, "load", "acquire T", "init", "count", "open 0", "require 0", "close device 0")

	if err := o.Close(); !errors.Is(err, errClose) {
		t.Fatalf("Close = %v, want %v", err, errClose)
	}
	checkLog(t, f, "close device 0")

	delete(f.fail, "close device 0")
	if err := o.Close(); err != nil {
		t.Fatal(err)
	}
	checkLog(t, f, "close device 0", "release lease")
}

// leaseCleanupError mimics an Acquire error that carries cleanup custody.
type leaseCleanupError struct {
	err   error
	close func() error
}

func (e *leaseCleanupError) Error() string { return e.err.Error() }
func (e *leaseCleanupError) Unwrap() error { return e.err }
func (e *leaseCleanupError) Close() error  { return e.close() }

func TestDeviceOwnerAcquireErrorCloser(t *testing.T) {
	f := newFakeBackend(1)
	ops := f.ops()
	ops.acquire = func(ctx context.Context, label string) (func() error, error) {
		f.log = append(f.log, "acquire "+label)
		err := &leaseCleanupError{
			err:   errors.Join(fmt.Errorf("cudalock: %w", context.Canceled), errors.New("cleanup failed")),
			close: f.release("cleanup lease"),
		}
		return nil, fmt.Errorf("wrapped: %w", err)
	}
	_, o, err := admit(context.Background(), "T", ops, nil)
	if !errors.Is(err, context.Canceled) || errors.Is(err, errNotVerified) {
		t.Fatalf("admit error = %v, want failure wrapping %v", err, context.Canceled)
	}
	checkLog(t, f, "load", "acquire T")
	if err := o.Close(); err != nil {
		t.Fatal(err)
	}
	checkLog(t, f, "release cleanup lease")
}

func TestDeviceOwnerCanceled(t *testing.T) {
	t.Run("nil", func(t *testing.T) {
		f := newFakeBackend(1)
		var ctx context.Context
		_, o, err := admit(ctx, "T", f.ops(), nil)
		if err == nil || errors.Is(err, errNotVerified) {
			t.Fatalf("admit error = %v, want failure", err)
		}
		checkLog(t, f)
		if err := o.Close(); err != nil {
			t.Fatal(err)
		}
		checkLog(t, f)
	})
	t.Run("before load", func(t *testing.T) {
		cause := errors.New("cause")
		ctx, cancel := context.WithCancelCause(context.Background())
		cancel(cause)
		f := newFakeBackend(1)
		_, _, err := admit(ctx, "T", f.ops(), nil)
		if !errors.Is(err, context.Canceled) || !errors.Is(err, cause) || errors.Is(err, errNotVerified) {
			t.Fatalf("admit error = %v, want failure wrapping %v and %v", err, context.Canceled, cause)
		}
		checkLog(t, f)
	})
	t.Run("during acquire", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		f := newFakeBackend(1)
		ops := f.ops()
		ops.acquire = func(ctx context.Context, label string) (func() error, error) {
			f.log = append(f.log, "acquire "+label)
			cancel()
			return nil, fmt.Errorf("cudalock: %w", ctx.Err())
		}
		_, o, err := admit(ctx, "T", ops, nil)
		if !errors.Is(err, context.Canceled) || errors.Is(err, errNotVerified) {
			t.Fatalf("admit error = %v, want failure wrapping %v", err, context.Canceled)
		}
		checkLog(t, f, "load", "acquire T")
		if err := o.Close(); err != nil {
			t.Fatal(err)
		}
		checkLog(t, f)
	})
	t.Run("during acquire with cause", func(t *testing.T) {
		cause := errors.New("cause")
		ctx, cancel := context.WithCancelCause(context.Background())
		defer cancel(nil)
		f := newFakeBackend(1)
		ops := f.ops()
		ops.acquire = func(ctx context.Context, label string) (func() error, error) {
			f.log = append(f.log, "acquire "+label)
			cancel(cause)
			return nil, fmt.Errorf("cudalock: %w", ctx.Err())
		}
		_, _, err := admit(ctx, "T", ops, nil)
		if !errors.Is(err, context.Canceled) || !errors.Is(err, cause) || errors.Is(err, errNotVerified) {
			t.Fatalf("admit error = %v, want failure wrapping %v and %v", err, context.Canceled, cause)
		}
		checkLog(t, f, "load", "acquire T")
	})
	t.Run("between devices", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		f := newFakeBackend(2)
		reject := func(d *cuda.Device) error {
			cancel()
			return f.require()(d)
		}
		f.reject[0] = true
		_, o, err := admit(ctx, "T", f.ops(), []Requirement{reject})
		if !errors.Is(err, context.Canceled) || errors.Is(err, errNotVerified) {
			t.Fatalf("admit error = %v, want failure wrapping %v", err, context.Canceled)
		}
		checkLog(t, f, "load", "acquire T", "init", "count", "open 0", "require 0", "close device 0")
		if err := o.Close(); err != nil {
			t.Fatal(err)
		}
		checkLog(t, f, "release lease")
	})
}

// TestDeviceOwnerCanceledStage cancels the admission context with a cause
// during each stage after the lease. Admission must fail with both the
// context error and the cause, never succeed or skip, and the owner must still
// release what admission retained.
func TestDeviceOwnerCanceledStage(t *testing.T) {
	errFail := errors.New("injected")
	tests := []struct {
		name     string
		count    int
		reject   []int
		cancelOn string // event during which ctx is canceled
		fail     string // event that also fails
		admitLog []string
		closeLog []string
	}{
		{
			name:     "init",
			count:    1,
			cancelOn: "init",
			admitLog: []string{"load", "acquire T", "init"},
			closeLog: []string{"release lease"},
		},
		{
			name:     "init fails",
			count:    1,
			cancelOn: "init",
			fail:     "init",
			admitLog: []string{"load", "acquire T", "init"},
			closeLog: []string{"release lease"},
		},
		{
			name:     "count with no devices",
			count:    0,
			cancelOn: "count",
			admitLog: []string{"load", "acquire T", "init", "count"},
			closeLog: []string{"release lease"},
		},
		{
			name:     "count fails",
			count:    1,
			cancelOn: "count",
			fail:     "count",
			admitLog: []string{"load", "acquire T", "init", "count"},
			closeLog: []string{"release lease"},
		},
		{
			name:     "open",
			count:    1,
			cancelOn: "open 0",
			admitLog: []string{"load", "acquire T", "init", "count", "open 0"},
			closeLog: []string{"close device 0", "release lease"},
		},
		{
			name:     "open fails",
			count:    1,
			cancelOn: "open 0",
			fail:     "open 0",
			admitLog: []string{"load", "acquire T", "init", "count", "open 0"},
			closeLog: []string{"release lease"},
		},
		{
			name:     "matching requirement",
			count:    1,
			cancelOn: "require 0",
			admitLog: []string{"load", "acquire T", "init", "count", "open 0", "require 0"},
			closeLog: []string{"close device 0", "release lease"},
		},
		{
			name:     "last rejecting requirement",
			count:    1,
			reject:   []int{0},
			cancelOn: "require 0",
			admitLog: []string{"load", "acquire T", "init", "count", "open 0", "require 0", "close device 0"},
			closeLog: []string{"release lease"},
		},
		{
			name:     "last rejected close fails",
			count:    1,
			reject:   []int{0},
			cancelOn: "require 0",
			fail:     "close device 0",
			admitLog: []string{"load", "acquire T", "init", "count", "open 0", "require 0", "close device 0"},
			closeLog: []string{"close device 0", "release lease"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cause := errors.New("cause")
			ctx, cancel := context.WithCancelCause(context.Background())
			defer cancel(nil)
			f := newFakeBackend(tt.count)
			for _, i := range tt.reject {
				f.reject[i] = true
			}
			f.hooks[tt.cancelOn] = func() { cancel(cause) }
			if tt.fail != "" {
				f.fail[tt.fail] = errFail
			}
			dev, o, err := admit(ctx, "T", f.ops(), []Requirement{f.require()})
			if dev != nil || o == nil {
				t.Fatalf("admit = %v, %v, %v; want nil device and owner", dev, o, err)
			}
			if !errors.Is(err, context.Canceled) || !errors.Is(err, cause) || errors.Is(err, errNotVerified) {
				t.Fatalf("admit error = %v, want failure wrapping %v and %v", err, context.Canceled, cause)
			}
			if tt.fail != "" && !errors.Is(err, errFail) {
				t.Errorf("admit error %q does not wrap the injected failure", err)
			}
			checkLog(t, f, tt.admitLog...)

			delete(f.fail, tt.fail)
			if err := o.Close(); err != nil {
				t.Fatal(err)
			}
			checkLog(t, f, tt.closeLog...)
		})
	}
}

func TestDeviceOwnerStreamsFirst(t *testing.T) {
	f := newFakeBackend(1)
	o := admitted(t, f)
	o.Defer(f.release("buffer a"))
	o.DeferStream(f.stream("s1"))
	o.Defer(f.release("buffer b"))
	o.DeferStream(f.stream("s2"))
	o.Defer(f.release("graph"))
	o.Defer(nil)
	o.DeferStream(nil)
	if err := o.Close(); err != nil {
		t.Fatal(err)
	}
	checkLog(t, f,
		"close stream s2", "close stream s1",
		"release graph", "release buffer b", "release buffer a",
		"close device 0", "release lease")
}

func TestDeviceOwnerRetention(t *testing.T) {
	all := []string{
		"close stream s2", "close stream s1",
		"release buffer b", "release buffer a",
		"close device 0", "release lease",
	}
	for i, failed := range all {
		t.Run(failed, func(t *testing.T) {
			errFail := errors.New("injected")
			f := newFakeBackend(1)
			o := admitted(t, f)
			o.Defer(f.release("buffer a"))
			o.DeferStream(f.stream("s1"))
			o.Defer(f.release("buffer b"))
			o.DeferStream(f.stream("s2"))
			f.fail[failed] = errFail
			if err := o.Close(); !errors.Is(err, errFail) {
				t.Fatalf("Close = %v, want %v", err, errFail)
			}
			checkLog(t, f, all[:i+1]...)

			if err := o.cleanup(); !errors.Is(err, errFail) {
				t.Fatalf("cleanup = %v, want %v", err, errFail)
			}
			checkLog(t, f)
			if !isRetained(o) {
				t.Error("owner not retained after failed Close")
			}
		})
	}
}

func TestDeviceOwnerRetry(t *testing.T) {
	errFail := errors.New("injected")
	f := newFakeBackend(1)
	o := admitted(t, f)
	o.DeferStream(f.stream("s1"))
	o.Defer(f.release("buffer"))
	o.DeferStream(f.stream("s2"))
	f.fail["close stream s1"] = errFail
	if err := o.Close(); !errors.Is(err, errFail) {
		t.Fatalf("Close = %v, want %v", err, errFail)
	}
	checkLog(t, f, "close stream s2", "close stream s1")

	f.fail["close device 0"] = errFail
	delete(f.fail, "close stream s1")
	if err := o.Close(); !errors.Is(err, errFail) {
		t.Fatalf("Close = %v, want %v", err, errFail)
	}
	checkLog(t, f, "close stream s1", "release buffer", "close device 0")

	delete(f.fail, "close device 0")
	if err := o.Close(); err != nil {
		t.Fatal(err)
	}
	checkLog(t, f, "close device 0", "release lease")
	if err := o.Close(); err != nil {
		t.Fatal(err)
	}
	if err := o.cleanup(); err != nil {
		t.Fatal(err)
	}
	checkLog(t, f)
}

func TestDeviceOwnerNoAutomaticRetry(t *testing.T) {
	errFail := errors.New("injected")
	f := newFakeBackend(1)
	o := admitted(t, f)
	o.Defer(f.release("graph"))
	f.fail["release graph"] = errFail
	if err := o.Close(); !errors.Is(err, errFail) {
		t.Fatalf("Close = %v, want %v", err, errFail)
	}
	checkLog(t, f, "release graph")

	// Even with the failure gone, the automatic cleanup reports the failed
	// explicit Close instead of retrying it.
	delete(f.fail, "release graph")
	if err := o.cleanup(); !errors.Is(err, errFail) {
		t.Fatalf("cleanup = %v, want %v", err, errFail)
	}
	checkLog(t, f)
	if !isRetained(o) {
		t.Error("owner not retained after failed Close")
	}

	// A later explicit Close resumes.
	if err := o.Close(); err != nil {
		t.Fatal(err)
	}
	checkLog(t, f, "release graph", "close device 0", "release lease")
}

func TestDeviceOwnerAutomaticFailure(t *testing.T) {
	errFail := errors.New("injected")
	f := newFakeBackend(1)
	o := admitted(t, f)
	f.fail["close device 0"] = errFail
	if err := o.cleanup(); !errors.Is(err, errFail) {
		t.Fatalf("cleanup = %v, want %v", err, errFail)
	}
	checkLog(t, f, "close device 0")
	if !isRetained(o) {
		t.Error("owner not retained after failed cleanup")
	}
}

// TestDeviceOwnerSource pins the order and calls of the real admission path,
// which the fake backend cannot observe.
func TestDeviceOwnerSource(t *testing.T) {
	src := parseSource(t, "device_owner.go")
	funcs := src.funcs()

	// RequireOwned admits with the real operations and t.Name(), and
	// registers the owner before interpreting the outcome.
	ro := funcs["RequireOwned"]
	if ro == nil {
		t.Fatal("RequireOwned not found")
	}
	if !slices.Equal(callNames(ro.Body), []string{"t.Helper", "admit", "t.Name", "o.register", "errors.Is", "t.Skip", "t.Fatal"}) {
		t.Errorf("RequireOwned calls %v", callNames(ro.Body))
	}
	if call := findCall(ro.Body, "admit"); call == nil || src.text(call) != "admit(ctx, t.Name(), &driverOperations, reqs)" {
		t.Error("RequireOwned does not admit with ctx, t.Name(), the driver operations and reqs")
	}

	// The real operations load the driver and take the canonical lease.
	var ops *ast.CompositeLit
	for _, d := range src.file.Decls {
		if g, ok := d.(*ast.GenDecl); ok && g.Tok == token.VAR {
			for _, s := range g.Specs {
				if v := s.(*ast.ValueSpec); v.Names[0].Name == "driverOperations" {
					ops = v.Values[0].(*ast.CompositeLit)
				}
			}
		}
	}
	if ops == nil {
		t.Fatal("driverOperations not found")
	}
	fields := make(map[string]string)
	for _, e := range ops.Elts {
		kv := e.(*ast.KeyValueExpr)
		fields[kv.Key.(*ast.Ident).Name] = src.text(kv.Value)
	}
	want := map[string]string{
		"load": `func() error { return driver.Load("") }`,
		"acquire": "func(ctx context.Context, label string) (func() error, error) {\n" +
			"\t\th, err := cudalock.Acquire(ctx, cudalock.WithLabel(label))\n" +
			"\t\tif err != nil {\n" +
			"\t\t\treturn nil, err\n" +
			"\t\t}\n" +
			"\t\treturn h.Release, nil\n" +
			"\t}",
		"init":        "cuda.Init",
		"count":       "cuda.DeviceCount",
		"open":        "cuda.NewDevice",
		"closeDevice": "(*cuda.Device).Close",
		"closeStream": "(*cuda.Stream).Close",
	}
	if len(fields) != len(want) {
		t.Errorf("driverOperations has fields %v, want %d", fields, len(want))
	}
	for field, text := range want {
		if fields[field] != text {
			t.Errorf("driverOperations.%s = %q, want %q", field, fields[field], text)
		}
	}

	// Admission loads, leases, initializes, counts and opens in order, and
	// records an opened device before evaluating or closing it.
	admitFn := funcs["admit"]
	if admitFn == nil {
		t.Fatal("admit not found")
	}
	var order []string
	for _, name := range callNames(admitFn.Body) {
		switch name {
		case "ops.load", "ops.acquire", "ops.init", "ops.count", "ops.open", "satisfies", "ops.closeDevice":
			order = append(order, name)
		}
	}
	if !slices.Equal(order, []string{"ops.load", "ops.acquire", "ops.init", "ops.count", "ops.open", "satisfies", "ops.closeDevice"}) {
		t.Errorf("admit calls operations in order %v", order)
	}
	body := src.text(admitFn.Body)
	if !strings.Contains(body, "o.lease = c.Close") || !strings.Contains(body, "o.lease = release") {
		t.Error("admit does not record the lease or an acquire error's closer")
	}
	if i, j := strings.Index(body, "o.dev = dev"), strings.Index(body, "satisfies(dev, reqs)"); i < 0 || i > j {
		t.Error("admit evaluates a device before recording it")
	}

	// The new path never falls back to the unowned helpers or lock bypasses.
	for _, name := range callNames(src.file) {
		switch name {
		case "Require", "RequireDevice", "os.Getenv", "os.LookupEnv":
			t.Errorf("device_owner.go calls %s", name)
		}
	}
}

// TestDeviceOwnerSourceCaller pins TestRequireFilters to the owned path.
func TestDeviceOwnerSourceCaller(t *testing.T) {
	src := parseSource(t, "cudatest_test.go")
	fn := src.funcs()["TestRequireFilters"]
	if fn == nil {
		t.Fatal("TestRequireFilters not found")
	}
	calls := callNames(fn.Body)
	const owned = "cudatest.RequireOwned(admission, t, cudatest.MinCompute(wantMajor, wantMinor))"
	if call := findCall(fn.Body, "cudatest.RequireOwned"); call == nil || src.text(call) != owned {
		t.Errorf("TestRequireFilters does not call %s", owned)
	}
	for _, name := range calls {
		switch name {
		case "cudatest.Require", "cudatest.RequireDevice", "dev.Close", "t.Cleanup":
			t.Errorf("TestRequireFilters calls %s", name)
		}
	}
	if !strings.Contains(src.text(fn.Body), "context.WithTimeout(t.Context(), 30*time.Second)") {
		t.Error("TestRequireFilters does not bound admission with a 30-second context")
	}
}

// goSource is a parsed Go file together with its bytes, so guards check the
// file as it is on disk.
type goSource struct {
	fset *token.FileSet
	file *ast.File
	src  []byte
}

func parseSource(t *testing.T, name string) *goSource {
	t.Helper()
	src, err := os.ReadFile(name)
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, name, src, 0)
	if err != nil {
		t.Fatal(err)
	}
	return &goSource{fset, file, src}
}

// text returns the source text of n.
func (s *goSource) text(n ast.Node) string {
	return string(s.src[s.fset.Position(n.Pos()).Offset:s.fset.Position(n.End()).Offset])
}

func (s *goSource) funcs() map[string]*ast.FuncDecl {
	m := make(map[string]*ast.FuncDecl)
	for _, d := range s.file.Decls {
		if fn, ok := d.(*ast.FuncDecl); ok && fn.Recv == nil {
			m[fn.Name.Name] = fn
		}
	}
	return m
}

// callNames returns the names of the functions called in n, in source order.
func callNames(n ast.Node) []string {
	var names []string
	ast.Inspect(n, func(n ast.Node) bool {
		if call, ok := n.(*ast.CallExpr); ok {
			names = append(names, funcName(call.Fun))
		}
		return true
	})
	return names
}

func funcName(fun ast.Expr) string {
	switch f := fun.(type) {
	case *ast.Ident:
		return f.Name
	case *ast.SelectorExpr:
		if x, ok := f.X.(*ast.Ident); ok {
			return x.Name + "." + f.Sel.Name
		}
		return "?." + f.Sel.Name
	}
	return "?"
}

func findCall(n ast.Node, name string) *ast.CallExpr {
	var found *ast.CallExpr
	ast.Inspect(n, func(n ast.Node) bool {
		if call, ok := n.(*ast.CallExpr); ok && found == nil && funcName(call.Fun) == name {
			found = call
		}
		return found == nil
	})
	return found
}

const publicEnv = "CUDATEST_REQUIRE_OWNED_PUBLIC"

// runPublic reruns test in a subprocess with publicEnv set and with driver
// bindings disabled, so the subprocess can never open a vendor library.
func runPublic(t *testing.T, test string) (string, error) {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^"+test+"$", "-test.v", "-test.count=1", "-test.timeout=1m")
	cmd.Env = append(os.Environ(), publicEnv+"=1", "CODEGEN_SKIP_BINDINGS=1")
	cmd.WaitDelay = 10 * time.Second
	out, err := cmd.CombinedOutput()
	if ctx.Err() != nil {
		t.Fatalf("subprocess %s: %v\n%s", test, ctx.Err(), out)
	}
	return string(out), err
}

func TestRequireOwnedPublicCanceled(t *testing.T) {
	if os.Getenv(publicEnv) != "" {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		RequireOwned(ctx, t)
		t.Error("RequireOwned returned with a canceled context")
		return
	}
	out, err := runPublic(t, "TestRequireOwnedPublicCanceled")
	// Loading the disabled bindings would skip. A failure naming the
	// cancellation shows that admission stopped before driver.Load.
	if err == nil || !strings.Contains(out, "--- FAIL: TestRequireOwnedPublicCanceled") ||
		!strings.Contains(out, "admission canceled before driver load: context canceled") ||
		strings.Contains(out, "SKIPPED") || strings.Contains(out, "RequireOwned returned") {
		t.Errorf("subprocess err = %v, output:\n%s", err, out)
	}
}

func TestRequireOwnedPublicSkip(t *testing.T) {
	if os.Getenv(publicEnv) != "" {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		RequireOwned(ctx, t)
		t.Error("RequireOwned returned with driver bindings disabled")
		return
	}
	out, err := runPublic(t, "TestRequireOwnedPublicSkip")
	if err != nil || !strings.Contains(out, "--- SKIP: TestRequireOwnedPublicSkip") ||
		!strings.Contains(out, "SKIPPED, NOT VERIFIED: the CUDA driver could not be loaded: driver: bindings disabled via CODEGEN_SKIP_BINDINGS") ||
		strings.Contains(out, "--- PASS") || strings.Contains(out, "--- FAIL") {
		t.Errorf("subprocess err = %v, output:\n%s", err, out)
	}
}
