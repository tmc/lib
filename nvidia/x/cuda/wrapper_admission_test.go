package cuda

import (
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
)

func wrapperAdmissionUnloaded(t *testing.T) {
	t.Helper()
	if driver.Loaded() {
		t.Fatal("host admission test requires an unloaded driver")
	}
	t.Cleanup(func() {
		if driver.Loaded() {
			t.Error("host admission test loaded the driver")
		}
	})
}

func TestWrapperAdmissionStream(t *testing.T) {
	wrapperAdmissionUnloaded(t)
	callbackErr := errors.New("callback")
	for _, tt := range []struct {
		name string
		s    *Stream
		want driver.Stream
		call bool
	}{
		{"nil", nil, 0, false},
		{"zero", &Stream{}, 0, false},
		{"zero with device", &Stream{device: &Device{}}, 0, false},
		{"default", (&Device{}).DefaultStream(), 0, true},
		{"live", &Stream{raw: driver.Stream(17)}, 17, true},
		{"retired nonzero", &Stream{raw: driver.Stream(17), destroyed: true}, 0, false},
		{"retired default", &Stream{isDefault: true, destroyed: true}, 0, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			calls := 0
			err := tt.s.WithHandle(func(raw driver.Stream) error {
				calls++
				if raw != tt.want {
					t.Errorf("handle = %v, want %v", raw, tt.want)
				}
				return callbackErr
			})
			if (calls == 1) != tt.call || calls > 1 || err == nil || errors.Is(err, callbackErr) != tt.call {
				t.Fatalf("calls=%d error=%v, want callback=%v", calls, err, tt.call)
			}
			if err := tt.s.WithHandle(nil); err == nil {
				t.Fatal("nil callback admitted")
			}
		})
	}
}

// The callbacks replace only context entry and the raw destroy call in the real
// private close state machines; no device operation is modeled as successful.
func TestWrapperAdmissionStreamClose(t *testing.T) {
	wrapperAdmissionUnloaded(t)
	s := &Stream{raw: driver.Stream(19)}
	entered, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	releaseCall := func() { once.Do(func() { close(release) }) }
	t.Cleanup(releaseCall)
	callDone := make(chan error, 1)
	go func() {
		callDone <- s.WithHandle(func(raw driver.Stream) error {
			if raw != 19 {
				return errors.New("wrong live handle")
			}
			close(entered)
			<-release
			return nil
		})
	}()
	wrapperAdmissionReceive(t, entered)
	if s.lifeMu.TryLock() {
		s.lifeMu.Unlock()
		t.Fatal("WithHandle did not exclude Close")
	}
	closeDone := make(chan error, 1)
	go func() {
		closeDone <- s.closeWith(func(fn func() error) error { return fn() },
			func() error { return nil }, func() error { s.raw = 0; return nil })
	}()
	releaseCall()
	if err := wrapperAdmissionError(t, callDone); err != nil {
		t.Fatal(err)
	}
	if err := wrapperAdmissionError(t, closeDone); err != nil {
		t.Fatal(err)
	}
	if err := s.WithHandle(func(driver.Stream) error { t.Error("retired callback ran"); return nil }); err == nil {
		t.Fatal("retired stream admitted")
	}
}

func TestWrapperAdmissionStreamRetainedClose(t *testing.T) {
	wrapperAdmissionUnloaded(t)
	refused := errors.New("destroy refused")
	for _, tt := range []struct {
		name       string
		s          *Stream
		want       driver.Stream
		destroyErr error
	}{
		{"default drain", (&Device{}).DefaultStream(), 0, nil},
		{"failed destruction", &Stream{raw: driver.Stream(21)}, 21, refused},
	} {
		t.Run(tt.name, func(t *testing.T) {
			destroys := 0
			err := tt.s.closeWith(func(fn func() error) error { return fn() },
				func() error { return nil }, func() error { destroys++; return tt.destroyErr })
			if !errors.Is(err, tt.destroyErr) || (destroys == 0) != tt.s.isDefault {
				t.Fatalf("close=%v destroys=%d", err, destroys)
			}
			calls := 0
			if err := tt.s.WithHandle(func(raw driver.Stream) error {
				calls++
				if raw != tt.want {
					t.Errorf("retained handle=%v, want %v", raw, tt.want)
				}
				return nil
			}); err != nil || calls != 1 {
				t.Fatalf("retained callback calls=%d error=%v", calls, err)
			}
		})
	}
}

func TestWrapperAdmissionElapsed(t *testing.T) {
	wrapperAdmissionUnloaded(t)
	e := &Event{raw: driver.Event(23), device: &Device{}}
	if _, err := e.ElapsedSince(nil); err == nil {
		t.Fatal("nil start admitted")
	}
	for _, start := range []*Event{e, {raw: driver.Event(23), device: &Device{}}, {raw: driver.Event(29)}} {
		if d, err := e.ElapsedSince(start); d != 0 || err == nil || !strings.Contains(err.Error(), "nil or closed device") {
			t.Fatalf("unusable device: duration=%v error=%v", d, err)
		}
	}
	if err := e.closeWith(func(fn func() error) error { return fn() }, func() error { e.raw = 0; return nil }); err != nil {
		t.Fatal(err)
	}
	for _, pair := range [][2]*Event{{e, e}, {e, {raw: driver.Event(31)}}, {{raw: driver.Event(31)}, e}} {
		if d, err := pair[0].ElapsedSince(pair[1]); d != 0 || err == nil || !strings.Contains(err.Error(), "event is destroyed") {
			t.Fatalf("retired pair: duration=%v error=%v", d, err)
		}
	}
}

func TestWrapperAdmissionElapsedContextBoundary(t *testing.T) {
	wrapperAdmissionUnloaded(t)
	for _, alias := range []bool{false, true} {
		t.Run(map[bool]string{false: "distinct", true: "alias"}[alias], func(t *testing.T) {
			dev := &Device{}
			e := &Event{raw: driver.Event(37), device: dev}
			start := &Event{raw: driver.Event(41), device: dev}
			if alias {
				start = e
			}
			dev.mu.Lock()
			var once sync.Once
			release := func() { once.Do(dev.mu.Unlock) }
			t.Cleanup(release)
			done := make(chan error, 1)
			go func() { _, err := e.ElapsedSince(start); done <- err }()
			wrapperAdmissionWait(t, func() bool {
				if e.mu.TryLock() {
					e.mu.Unlock()
					return false
				}
				if start != e && start.mu.TryLock() {
					start.mu.Unlock()
					return false
				}
				return true
			})
			wrapperAdmissionWait(t, func() bool {
				if !eventPairMu.TryLock() {
					return false
				}
				eventPairMu.Unlock()
				return true
			})
			// Close's writer cannot run until the blocked context call returns.
			closed := make(chan error, 1)
			go func() {
				closed <- start.closeWith(func(fn func() error) error { return fn() }, func() error { start.raw = 0; return nil })
			}()
			release()
			if err := wrapperAdmissionError(t, done); err == nil || !strings.Contains(err.Error(), "nil or closed device") {
				t.Fatalf("context refusal = %v", err)
			}
			if err := wrapperAdmissionError(t, closed); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestWrapperAdmissionElapsedConcurrentClose(t *testing.T) {
	wrapperAdmissionUnloaded(t)
	for round := range 64 {
		a := &Event{raw: driver.Event(round + 43), device: &Device{}}
		b := &Event{raw: driver.Event(round + 107), device: &Device{}}
		ready := make(chan struct{})
		var wg sync.WaitGroup
		for _, pair := range [][2]*Event{{a, b}, {b, a}, {a, a}, {b, b}} {
			wg.Go(func() {
				<-ready
				for range 128 {
					if d, err := pair[0].ElapsedSince(pair[1]); d != 0 || err == nil {
						t.Errorf("unusable or retired pair: duration=%v error=%v", d, err)
					}
				}
			})
		}
		for _, e := range []*Event{a, b} {
			// Failed destruction retains the live event, allowing real writer
			// pressure before the final successful transition in closeWith.
			wg.Go(func() {
				<-ready
				refused := errors.New("destroy refused")
				for range 32 {
					if err := e.closeWith(func(fn func() error) error { return fn() }, func() error { runtime.Gosched(); return refused }); !errors.Is(err, refused) {
						t.Errorf("failed close = %v", err)
					}
				}
				if err := e.closeWith(func(fn func() error) error { return fn() }, func() error { e.raw = 0; return nil }); err != nil {
					t.Error(err)
				}
			})
		}
		close(ready)
		done := make(chan struct{})
		go func() { wg.Wait(); close(done) }()
		wrapperAdmissionReceive(t, done)
	}
}

// Bind the host-only controls to the actual native-facing method: every raw
// read follows pair acquisition and the gate is released before context entry.
// Successful driver elapsed-time values remain outside this host-only test.
func TestWrapperAdmissionElapsedSourceBoundary(t *testing.T) {
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("missing source location")
	}
	f, err := parser.ParseFile(token.NewFileSet(), filepath.Join(filepath.Dir(filename), "event.go"), nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var method *ast.FuncDecl
	for _, d := range f.Decls {
		if fn, ok := d.(*ast.FuncDecl); ok && fn.Name.Name == "ElapsedSince" {
			method = fn
		}
	}
	if method == nil {
		t.Fatal("missing actual ElapsedSince method")
	}
	var gateLock, endLock, startLock, gateUnlock, contextCall token.Pos
	ast.Inspect(method.Body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		for _, target := range []struct {
			path []string
			pos  *token.Pos
		}{
			{[]string{"eventPairMu", "Lock"}, &gateLock},
			{[]string{"e", "mu", "RLock"}, &endLock},
			{[]string{"start", "mu", "RLock"}, &startLock},
			{[]string{"eventPairMu", "Unlock"}, &gateUnlock},
			{[]string{"e", "device", "WithContext"}, &contextCall},
		} {
			if wrapperAdmissionSelector(call.Fun, target.path) {
				*target.pos = call.Pos()
			}
		}
		return true
	})
	if !(gateLock > 0 && gateLock < endLock && endLock < startLock && startLock < gateUnlock && gateUnlock < contextCall) {
		t.Fatalf("missing ordered pair acquisition and release: %v %v %v %v %v", gateLock, endLock, startLock, gateUnlock, contextCall)
	}
	reads := 0
	ast.Inspect(method.Body, func(n ast.Node) bool {
		if s, ok := n.(*ast.SelectorExpr); ok && s.Sel.Name == "raw" {
			reads++
			if s.Pos() < gateUnlock {
				t.Error("raw event handle read before pair acquisition completed")
			}
		}
		return true
	})
	if reads != 2 {
		t.Fatalf("native elapsed call raw reads=%d, want 2", reads)
	}
}

func wrapperAdmissionSelector(e ast.Expr, path []string) bool {
	for i := len(path) - 1; i > 0; i-- {
		s, ok := e.(*ast.SelectorExpr)
		if !ok || s.Sel.Name != path[i] {
			return false
		}
		e = s.X
	}
	id, ok := e.(*ast.Ident)
	return ok && id.Name == path[0]
}

func wrapperAdmissionWait(t *testing.T, ready func() bool) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for !ready() {
		if time.Now().After(deadline) {
			t.Fatal("timed out waiting for lock boundary")
		}
		runtime.Gosched()
	}
}

func wrapperAdmissionReceive(t *testing.T, done <-chan struct{}) {
	t.Helper()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("host operation did not finish")
	}
}

func wrapperAdmissionError(t *testing.T, done <-chan error) error {
	t.Helper()
	select {
	case err := <-done:
		return err
	case <-time.After(3 * time.Second):
		t.Fatal("host operation did not finish")
		return nil
	}
}
