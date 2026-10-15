package cuda

import (
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

// These callbacks model only the existing Context.Run call boundary. They do
// not bind a context or claim that any CUDA operation succeeded on a device.
type ownerRunPhase struct{ entry, restore error }

func ownerRun(phases []ownerRunPhase, calls *int) func(func() error) error {
	return func(fn func() error) error {
		i := *calls
		*calls = i + 1
		var phase ownerRunPhase
		if i < len(phases) {
			phase = phases[i]
		}
		if phase.entry != nil {
			return phase.entry
		}
		return errors.Join(fn(), phase.restore)
	}
}

func TestContextOwnerStreamClose(t *testing.T) {
	entryErr, syncErr, destroyErr, restoreErr := errors.New("entry"), errors.New("sync"), errors.New("destroy"), errors.New("restore")
	tests := []struct {
		name                 string
		phases               []ownerRunPhase
		syncErr, destroyErr  error
		want                 []error
		calls, sync, destroy int
		closed               bool
	}{
		{name: "success", calls: 2, sync: 1, destroy: 1, closed: true},
		{name: "sync entry", phases: []ownerRunPhase{{entry: entryErr}}, want: []error{entryErr}, calls: 1},
		{name: "sync failure", syncErr: syncErr, want: []error{syncErr}, calls: 1, sync: 1},
		{name: "sync restore", phases: []ownerRunPhase{{restore: restoreErr}}, want: []error{restoreErr}, calls: 1, sync: 1},
		{name: "sync and restore", phases: []ownerRunPhase{{restore: restoreErr}}, syncErr: syncErr, want: []error{syncErr, restoreErr}, calls: 1, sync: 1},
		{name: "destroy entry", phases: []ownerRunPhase{{}, {entry: entryErr}}, want: []error{entryErr}, calls: 2, sync: 1},
		{name: "destroy failure", destroyErr: destroyErr, want: []error{destroyErr}, calls: 2, sync: 1, destroy: 1},
		{name: "destroy and restore", phases: []ownerRunPhase{{}, {restore: restoreErr}}, destroyErr: destroyErr, want: []error{destroyErr, restoreErr}, calls: 2, sync: 1, destroy: 1},
		{name: "destroyed then restore", phases: []ownerRunPhase{{}, {restore: restoreErr}}, want: []error{restoreErr}, calls: 2, sync: 1, destroy: 1, closed: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &Stream{raw: driver.Stream(41)}
			calls, syncCalls, destroyCalls, cleanups := 0, 0, 0, 0
			s.AddCleanup(func() {
				cleanups++
				if !s.lifeMu.TryLock() {
					t.Error("cleanup ran under lifecycle lock")
				} else {
					s.lifeMu.Unlock()
				}
				if !s.mu.TryLock() {
					t.Error("cleanup ran under cleanup-list lock")
				} else {
					s.mu.Unlock()
				}
			})
			synchronize := func() error {
				syncCalls++
				if s.raw == 0 {
					t.Error("synchronized the legacy zero stream after destruction")
				}
				return tt.syncErr
			}
			destroy := func() error {
				destroyCalls++
				if s.raw == 0 {
					t.Error("destroy callback received a zero owner")
				}
				if tt.destroyErr == nil {
					s.raw = 0 // The actual generated pointer Close has this contract.
				}
				return tt.destroyErr
			}
			err := s.closeWith(ownerRun(tt.phases, &calls), synchronize, destroy)
			if (err == nil) != (len(tt.want) == 0) {
				t.Fatalf("Close error = %v, want causes %v", err, tt.want)
			}
			for _, want := range tt.want {
				if !errors.Is(err, want) {
					t.Errorf("Close error %v lost %v", err, want)
				}
			}
			if calls != tt.calls || syncCalls != tt.sync || destroyCalls != tt.destroy || s.destroyed != tt.closed || (s.raw == 0) != tt.closed {
				t.Fatalf("calls=%d sync=%d destroy=%d destroyed=%v raw=%v", calls, syncCalls, destroyCalls, s.destroyed, s.raw)
			}
			wantCleanups := 0
			if tt.closed {
				wantCleanups = 1
			}
			if cleanups != wantCleanups {
				t.Fatalf("cleanup count=%d, want %d", cleanups, wantCleanups)
			}
			if !tt.closed {
				if err := s.closeWith(func(fn func() error) error { return fn() }, func() error { return nil }, func() error { s.raw = 0; return nil }); err != nil {
					t.Fatal("retry:", err)
				}
			}
			if err := s.Close(); err != nil {
				t.Fatal("terminal public Close retried context or operation:", err)
			}
			called := false
			if err := s.WithHandle(func(driver.Stream) error { called = true; return nil }); err == nil || called {
				t.Fatal("destroyed wrapper admitted a raw handle callback")
			}
			if _, err := s.Query(); err == nil {
				t.Fatal("destroyed wrapper admitted Query")
			}
			if err := s.Synchronize(nil); err == nil {
				t.Fatal("destroyed wrapper admitted Synchronize")
			}
			s.AddCleanup(func() { cleanups++ })
			if cleanups != 2 || len(s.cleanups) != 0 {
				t.Fatalf("terminal cleanup count=%d retained=%d", cleanups, len(s.cleanups))
			}
		})
	}
}

func TestContextOwnerEventClose(t *testing.T) {
	entryErr, destroyErr, restoreErr := errors.New("entry"), errors.New("destroy"), errors.New("restore")
	tests := []struct {
		name       string
		phase      ownerRunPhase
		destroyErr error
		want       []error
		destroys   int
		closed     bool
	}{
		{name: "success", destroys: 1, closed: true},
		{name: "entry", phase: ownerRunPhase{entry: entryErr}, want: []error{entryErr}},
		{name: "destroy", destroyErr: destroyErr, want: []error{destroyErr}, destroys: 1},
		{name: "destroy and restore", phase: ownerRunPhase{restore: restoreErr}, destroyErr: destroyErr, want: []error{destroyErr, restoreErr}, destroys: 1},
		{name: "destroyed then restore", phase: ownerRunPhase{restore: restoreErr}, want: []error{restoreErr}, destroys: 1, closed: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := &Event{raw: driver.Event(43)}
			calls, destroys := 0, 0
			err := e.closeWith(ownerRun([]ownerRunPhase{tt.phase}, &calls), func() error {
				destroys++
				if tt.destroyErr == nil {
					e.raw = 0
				}
				return tt.destroyErr
			})
			if (err == nil) != (len(tt.want) == 0) {
				t.Fatalf("Close error=%v, want %v", err, tt.want)
			}
			for _, want := range tt.want {
				if !errors.Is(err, want) {
					t.Errorf("Close error %v lost %v", err, want)
				}
			}
			if calls != 1 || destroys != tt.destroys || e.destroyed != tt.closed || (e.raw == 0) != tt.closed {
				t.Fatalf("calls=%d destroys=%d closed=%v raw=%v", calls, destroys, e.destroyed, e.raw)
			}
			if !tt.closed {
				if err := e.closeWith(func(fn func() error) error { return fn() }, func() error { e.raw = 0; return nil }); err != nil {
					t.Fatal("retry:", err)
				}
			}
			if err := e.Close(); err != nil {
				t.Fatal("terminal public Close retried context:", err)
			}
			if _, err := e.Query(); err == nil {
				t.Fatal("destroyed event admitted Query")
			}
			if err := e.Synchronize(nil); err == nil {
				t.Fatal("destroyed event admitted Synchronize")
			}
			if err := e.Record(&Stream{}); err == nil {
				t.Fatal("destroyed event admitted Record")
			}
			if _, err := e.ElapsedSince(e); err == nil {
				t.Fatal("destroyed event admitted ElapsedSince")
			}
		})
	}
}

func TestContextOwnerDefaultStreamClose(t *testing.T) {
	restoreErr := errors.New("restore")
	s := (&Device{}).DefaultStream()
	calls, syncs, destroys, cleanups := 0, 0, 0, 0
	s.AddCleanup(func() { cleanups++ })
	synchronize := func() error { syncs++; return nil }
	destroy := func() error { destroys++; return nil }
	if err := s.closeWith(ownerRun([]ownerRunPhase{{restore: restoreErr}}, &calls), synchronize, destroy); !errors.Is(err, restoreErr) {
		t.Fatalf("restore error=%v", err)
	}
	if cleanups != 0 || s.destroyed {
		t.Fatal("failed default drain discarded callbacks or default stream")
	}
	if err := s.closeWith(func(fn func() error) error { return fn() }, synchronize, destroy); err != nil {
		t.Fatal(err)
	}
	if s.destroyed || cleanups != 1 || destroys != 0 || syncs != 2 || s.Device().DefaultStream() != s {
		t.Fatal("default stream owner/drain mismatch")
	}
	s.AddCleanup(func() { cleanups++ })
	if err := s.closeWith(func(fn func() error) error { return fn() }, synchronize, destroy); err != nil {
		t.Fatal(err)
	}
	if cleanups != 2 || destroys != 0 || s.destroyed {
		t.Fatal("default stream cannot be reused for cleanup")
	}
}

func TestContextOwnerConcurrentCloseAndCleanup(t *testing.T) {
	s := &Stream{raw: driver.Stream(47)}
	var runs, syncs, destroys, cleanups atomic.Int32
	started, proceed := make(chan struct{}), make(chan struct{})
	s.AddCleanup(func() { cleanups.Add(1) })
	restoreErr := errors.New("restore")
	run := func(fn func() error) error {
		n := runs.Add(1)
		err := fn()
		if n == 2 {
			return errors.Join(err, restoreErr)
		}
		return err
	}
	synchronize := func() error { syncs.Add(1); close(started); <-proceed; return nil }
	destroy := func() error { destroys.Add(1); s.raw = 0; return nil }
	results := make(chan error, 2)
	go func() { results <- s.closeWith(run, synchronize, destroy) }()
	<-started
	go func() { results <- s.closeWith(run, synchronize, destroy) }()
	var wg sync.WaitGroup
	wg.Go(func() { s.AddCleanup(func() { cleanups.Add(1) }) })
	close(proceed)
	a, b := <-results, <-results
	wg.Wait()
	if !((a == nil && errors.Is(b, restoreErr)) || (b == nil && errors.Is(a, restoreErr))) {
		t.Fatalf("concurrent errors=%v / %v", a, b)
	}
	if runs.Load() != 2 || syncs.Load() != 1 || destroys.Load() != 1 || cleanups.Load() != 2 {
		t.Fatalf("runs=%d sync=%d destroy=%d cleanup=%d", runs.Load(), syncs.Load(), destroys.Load(), cleanups.Load())
	}
}

// The runtime controls call the actual closeWith state machines. These source
// checks bind the public entry points to the per-call context and real raw
// operations, so passing a different callback route cannot hide behind them.
func TestContextOwnerPublicCloseRoute(t *testing.T) {
	if driver.Loaded() {
		t.Fatal("host controls require an unloaded driver")
	}
	t.Cleanup(func() {
		if driver.Loaded() {
			t.Error("host controls loaded a driver")
		}
	})
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("missing source location")
	}
	for _, tt := range []struct {
		file, receiver string
		operations     []string
	}{
		{"stream.go", "s", []string{"Synchronize", "Close"}},
		{"event.go", "e", []string{"Close"}},
	} {
		f, err := parser.ParseFile(token.NewFileSet(), filepath.Join(filepath.Dir(filename), tt.file), nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		var method *ast.FuncDecl
		for _, d := range f.Decls {
			if fn, ok := d.(*ast.FuncDecl); ok && fn.Name.Name == "Close" {
				method = fn
			}
		}
		if method == nil || len(method.Body.List) != 1 {
			t.Fatalf("%s public Close bypassed its state helper", tt.file)
		}
		ret, ok := method.Body.List[0].(*ast.ReturnStmt)
		if !ok || len(ret.Results) != 1 {
			t.Fatal("public Close must return helper result")
		}
		call, ok := ret.Results[0].(*ast.CallExpr)
		if !ok || !ownerSelector(call.Fun, tt.receiver, "closeWith") || len(call.Args) != len(tt.operations)+1 {
			t.Fatalf("%s missing actual state helper", tt.file)
		}
		if !ownerSelector(call.Args[0], tt.receiver, "device", "WithContext") {
			t.Fatalf("%s omitted actual context runner", tt.file)
		}
		for i, operation := range tt.operations {
			fn, ok := call.Args[i+1].(*ast.FuncLit)
			if !ok || len(fn.Body.List) != 1 {
				t.Fatal("raw operation must be evaluated inside lifecycle callback")
			}
			ret, ok := fn.Body.List[0].(*ast.ReturnStmt)
			if !ok || len(ret.Results) != 1 {
				t.Fatal("raw callback must return its operation error")
			}
			operationCall, ok := ret.Results[0].(*ast.CallExpr)
			if !ok || len(operationCall.Args) != 0 || !ownerSelector(operationCall.Fun, tt.receiver, "raw", operation) {
				t.Fatalf("%s replaced actual raw %s operation", tt.file, operation)
			}
		}
	}
}

func ownerSelector(expr ast.Expr, root string, fields ...string) bool {
	for i := len(fields) - 1; i >= 0; i-- {
		s, ok := expr.(*ast.SelectorExpr)
		if !ok || s.Sel.Name != fields[i] {
			return false
		}
		expr = s.X
	}
	id, ok := expr.(*ast.Ident)
	return ok && id.Name == root
}
