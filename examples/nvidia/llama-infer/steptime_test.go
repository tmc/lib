package main

import (
	"context"
	"errors"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
)

// closeLog records the calls closeStepTimer makes, in order, and returns
// the error fail assigns to each.
type closeLog struct {
	calls []string
	fail  map[string]error
}

func (l *closeLog) call(name string) error {
	l.calls = append(l.calls, name)
	return l.fail[name]
}

type fakeStream struct{ log *closeLog }

func (s *fakeStream) Synchronize(context.Context) error { return s.log.call("sync") }
func (s *fakeStream) Close() error                      { return s.log.call("stream") }

type fakeEvent struct {
	name string
	log  *closeLog
}

// errNilEventClosed is returned by Close on a nil *fakeEvent, which
// closeStepTimer must skip.
var errNilEventClosed = errors.New("nil event closed")

func (e *fakeEvent) Close() error {
	if e == nil {
		return errNilEventClosed
	}
	return e.log.call(e.name)
}

// TestCloseStepTimer checks that closeStepTimer drains the stream before
// closing anything; that a failed drain closes nothing, keeps everything and
// matches errUndrained; and that otherwise it closes every held event and
// then the stream once each even after earlier closes fail, forgets each one
// it closed, keeps each one that failed, and joins every failure in order
// with its identity kept.
func TestCloseStepTimer(t *testing.T) {
	errSync := errors.New("sync failed")
	errAnchor := errors.New("anchor failed")
	errStart := errors.New("start failed")
	errEnd := errors.New("end failed")
	errStream := errors.New("stream failed")
	for _, tt := range []struct {
		name      string
		noStream  bool
		events    []string // "" is a nil event
		fail      map[string]error
		wantCalls []string
		wantErr   string
		wantIs    []error
		wantHeld  []string // what is still held afterwards, stream last
	}{
		{
			name:      "all succeed",
			events:    []string{"anchor", "start", "end"},
			wantCalls: []string{"sync", "anchor", "start", "end", "stream"},
		},
		{
			name:      "sync fails",
			events:    []string{"anchor", "start", "end"},
			fail:      map[string]error{"sync": errSync, "anchor": errAnchor, "stream": errStream},
			wantCalls: []string{"sync"},
			wantErr:   "synchronize stream: stream not drained: sync failed",
			wantIs:    []error{errSync, errUndrained},
			wantHeld:  []string{"anchor", "start", "end", "stream"},
		},
		{
			name:      "every close fails",
			events:    []string{"anchor", "start", "end"},
			fail:      map[string]error{"anchor": errAnchor, "start": errStart, "end": errEnd, "stream": errStream},
			wantCalls: []string{"sync", "anchor", "start", "end", "stream"},
			wantErr: "close event: anchor failed\n" +
				"close event: start failed\n" +
				"close event: end failed\n" +
				"close stream: stream failed",
			wantIs:   []error{errAnchor, errStart, errEnd, errStream},
			wantHeld: []string{"anchor", "start", "end", "stream"},
		},
		{
			name:      "first event fails",
			events:    []string{"anchor", "start", "end"},
			fail:      map[string]error{"anchor": errAnchor},
			wantCalls: []string{"sync", "anchor", "start", "end", "stream"},
			wantErr:   "close event: anchor failed",
			wantIs:    []error{errAnchor},
			wantHeld:  []string{"anchor"},
		},
		{
			name:      "last event and stream fail",
			events:    []string{"anchor", "start", "end"},
			fail:      map[string]error{"end": errEnd, "stream": errStream},
			wantCalls: []string{"sync", "anchor", "start", "end", "stream"},
			wantErr:   "close event: end failed\nclose stream: stream failed",
			wantIs:    []error{errEnd, errStream},
			wantHeld:  []string{"end", "stream"},
		},
		{
			name:      "nil events skipped",
			events:    []string{"anchor", "", ""},
			fail:      map[string]error{"anchor": errAnchor},
			wantCalls: []string{"sync", "anchor", "stream"},
			wantErr:   "close event: anchor failed",
			wantIs:    []error{errAnchor},
			wantHeld:  []string{"anchor"},
		},
		{
			name:      "nil stream",
			noStream:  true,
			events:    []string{"anchor", "start", "end"},
			wantCalls: []string{"anchor", "start", "end"},
		},
		{
			name:     "nothing built",
			noStream: true,
			events:   []string{"", "", ""},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			log := &closeLog{fail: tt.fail}
			stream := &fakeStream{log: log}
			if tt.noStream {
				stream = nil
			}
			var events []*fakeEvent
			for _, name := range tt.events {
				var e *fakeEvent
				if name != "" {
					e = &fakeEvent{name: name, log: log}
				}
				events = append(events, e)
			}
			slots := make([]**fakeEvent, len(events))
			for i := range events {
				slots[i] = &events[i]
			}
			var drained bool
			err := closeStepTimer(&stream, &drained, slots)
			if !slices.Equal(log.calls, tt.wantCalls) {
				t.Errorf("calls = %q, want %q", log.calls, tt.wantCalls)
			}
			if got := heldNames(stream, events); !slices.Equal(got, tt.wantHeld) {
				t.Errorf("held after close = %q, want %q", got, tt.wantHeld)
			}
			checkErr(t, err, tt.wantErr, tt.wantIs)
			if want := !tt.noStream && tt.fail["sync"] == nil; drained != want {
				t.Errorf("drained = %v, want %v", drained, want)
			}
			if tt.wantErr != "" && strings.HasPrefix(tt.wantErr, "close ") && errors.Is(err, errUndrained) {
				t.Errorf("errors.Is(err, errUndrained) = true after the stream drained")
			}

			// A second close retries only what is still held, waiting on
			// the stream again only if it has not drained. With every
			// failure cleared it must release all of it.
			wasDrained, streamHeld := drained, stream != nil
			log.calls, log.fail = nil, nil
			if err := closeStepTimer(&stream, &drained, slots); err != nil {
				t.Fatalf("retry: %v", err)
			}
			var want []string
			if !wasDrained && streamHeld {
				want = []string{"sync"}
			}
			want = append(want, tt.wantHeld...)
			if !slices.Equal(log.calls, want) {
				t.Errorf("retry calls = %q, want %q", log.calls, want)
			}
			if got := heldNames(stream, events); got != nil {
				t.Errorf("held after retry = %q, want nothing", got)
			}
		})
	}
}

// destroyingStream is a stream whose first Close destroys it and then
// fails, as cuda.Stream.Close does when the stream is destroyed but
// restoring the previous context fails. Once destroyed, Synchronize fails
// and Close succeeds without doing anything.
type destroyingStream struct {
	log       *closeLog
	destroyed bool
}

var errDestroyed = errors.New("stream is destroyed")

func (s *destroyingStream) Synchronize(context.Context) error {
	s.log.calls = append(s.log.calls, "sync")
	if s.destroyed {
		return errDestroyed
	}
	return nil
}

func (s *destroyingStream) Close() error {
	s.log.calls = append(s.log.calls, "stream")
	if s.destroyed {
		return nil
	}
	s.destroyed = true
	return errors.New("restore context failed")
}

// TestCloseStepTimerRetryAfterDestroy checks that a close retried after
// the stream's Close destroyed it but failed does not wait on the destroyed
// stream again, so it is not reported as undrained, and that it still
// closes an event whose first close failed. The control shows that waiting
// again, as close would if it did not record the drain, reports
// errUndrained and never retries the event.
func TestCloseStepTimerRetryAfterDestroy(t *testing.T) {
	for _, control := range []bool{false, true} {
		log := &closeLog{fail: map[string]error{"start": errors.New("start failed")}}
		stream := &destroyingStream{log: log}
		events := []*fakeEvent{{"anchor", log}, {"start", log}, {"end", log}}
		slots := []**fakeEvent{&events[0], &events[1], &events[2]}
		var drained bool
		if err := closeStepTimer(&stream, &drained, slots); err == nil {
			t.Fatalf("control=%v: first close: err = nil, want the event and stream failures", control)
		}
		if got, want := heldNames2(stream, events), []string{"start", "stream"}; !slices.Equal(got, want) {
			t.Fatalf("control=%v: held after first close = %q, want %q", control, got, want)
		}
		if control {
			drained = false
		}
		log.calls, log.fail = nil, nil
		err := closeStepTimer(&stream, &drained, slots)
		if control {
			if !errors.Is(err, errUndrained) || !slices.Equal(log.calls, []string{"sync"}) || events[1] == nil {
				t.Errorf("control: retry err = %v, calls %q, start held = %v; want errUndrained after one sync with start held", err, log.calls, events[1] != nil)
			}
			continue
		}
		if err != nil {
			t.Errorf("retry: %v", err)
		}
		if want := []string{"start", "stream"}; !slices.Equal(log.calls, want) {
			t.Errorf("retry calls = %q, want %q", log.calls, want)
		}
		if got := heldNames2(stream, events); got != nil {
			t.Errorf("held after retry = %q, want nothing", got)
		}
	}
}

// heldNames2 is heldNames for a destroyingStream.
func heldNames2(stream *destroyingStream, events []*fakeEvent) []string {
	var held []string
	for _, e := range events {
		if e != nil {
			held = append(held, e.name)
		}
	}
	if stream != nil {
		held = append(held, "stream")
	}
	return held
}

// TestMeasureAfterClose checks that measure enqueues nothing once close
// has drained the timer's stream.
func TestMeasureAfterClose(t *testing.T) {
	st := &stepTimer{drained: true}
	called := false
	err := st.measure(context.Background(), "step", nil, func(*cuda.Stream) error {
		called = true
		return nil
	})
	if !errors.Is(err, errTimerClosed) || called {
		t.Errorf("measure after close: err = %v, fn called = %v; want errTimerClosed, not called", err, called)
	}
}

// heldNames names the events that are still set, then "stream" if the
// stream is.
func heldNames(stream *fakeStream, events []*fakeEvent) []string {
	var held []string
	for _, e := range events {
		if e != nil {
			held = append(held, e.name)
		}
	}
	if stream != nil {
		held = append(held, "stream")
	}
	return held
}

// checkErr checks err against the wanted message and identities; an empty
// want means no error.
func checkErr(t *testing.T, err error, want string, is []error) {
	t.Helper()
	if want == "" {
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		return
	}
	if err == nil {
		t.Fatalf("err = nil, want %q", want)
	}
	if err.Error() != want {
		t.Errorf("err = %q, want %q", err, want)
	}
	for _, w := range is {
		if !errors.Is(err, w) {
			t.Errorf("errors.Is(err, %v) = false", w)
		}
	}
}

// TestStepTimerCloseZero checks that close on a timer with nothing built
// makes no calls through nil pointers and reports no error.
func TestStepTimerCloseZero(t *testing.T) {
	if err := (&stepTimer{}).close(); err != nil {
		t.Fatalf("close: %v", err)
	}
}

// TestStepTimerBuild checks that build keeps a stream or event returned
// along with an error, releases on every failure, returns no timer when the
// release succeeds, and returns the partly built timer with the build and
// release errors joined when it fails. The cuda values are only compared,
// never used.
func TestStepTimerBuild(t *testing.T) {
	errStream := errors.New("stream failed")
	errEvent := errors.New("event failed")
	errSetup := errors.New("setup failed")
	errRelease := errors.New("release failed")
	for _, tt := range []struct {
		name       string
		streamErr  error
		streamNil  bool // newStream returns no stream with its error
		eventFail  int  // 1-based newEvent call that fails; 0 for none
		eventNil   bool // the failing newEvent returns no event
		setupErr   error
		releaseErr error
		wantEvents int // events newEvent was asked for
		wantSetup  bool
		wantSeen   []string // fields set when release ran, stream first
		wantOwner  bool
		wantErr    string
		wantIs     []error
	}{
		{name: "success", wantEvents: 3, wantSetup: true, wantOwner: true},
		{
			name: "stream fails empty", streamErr: errStream, streamNil: true,
			wantSeen: []string{}, wantErr: "new stream: stream failed", wantIs: []error{errStream},
		},
		{
			name: "stream fails with a stream", streamErr: errStream,
			wantSeen: []string{"stream"}, wantErr: "new stream: stream failed", wantIs: []error{errStream},
		},
		{
			name: "second event fails with an event", eventFail: 2, wantEvents: 2,
			wantSeen: []string{"stream", "anchor", "start"}, wantErr: "new event: event failed", wantIs: []error{errEvent},
		},
		{
			name: "last event fails empty", eventFail: 3, eventNil: true, wantEvents: 3,
			wantSeen: []string{"stream", "anchor", "start"}, wantErr: "new event: event failed", wantIs: []error{errEvent},
		},
		{
			name: "setup fails", setupErr: errSetup, wantEvents: 3, wantSetup: true,
			wantSeen: []string{"stream", "anchor", "start", "end"}, wantErr: "setup failed", wantIs: []error{errSetup},
		},
		{
			name: "event fails and release fails", eventFail: 1, wantEvents: 1, releaseErr: errRelease,
			wantSeen: []string{"stream", "anchor"}, wantOwner: true,
			wantErr: "new event: event failed\nclose step timer: release failed", wantIs: []error{errEvent, errRelease},
		},
		{
			name: "setup fails and release fails undrained", setupErr: errSetup, wantEvents: 3, wantSetup: true,
			releaseErr: fmt.Errorf("synchronize stream: %w: %w", errUndrained, errRelease),
			wantSeen:   []string{"stream", "anchor", "start", "end"}, wantOwner: true,
			wantErr: "setup failed\nclose step timer: synchronize stream: stream not drained: release failed",
			wantIs:  []error{errSetup, errRelease, errUndrained},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			st := &stepTimer{source: "probe"}
			events := 0
			setup := false
			var seen []string
			newStream := func() (*cuda.Stream, error) {
				if tt.streamNil {
					return nil, tt.streamErr
				}
				return new(cuda.Stream), tt.streamErr
			}
			newEvent := func() (*cuda.Event, error) {
				events++
				if events != tt.eventFail {
					return new(cuda.Event), nil
				}
				if tt.eventNil {
					return nil, errEvent
				}
				return new(cuda.Event), errEvent
			}
			release := func() error {
				seen = []string{}
				for _, f := range []struct {
					name string
					set  bool
				}{{"stream", st.stream != nil}, {"anchor", st.anchor != nil}, {"start", st.start != nil}, {"end", st.end != nil}} {
					if f.set {
						seen = append(seen, f.name)
					}
				}
				return tt.releaseErr
			}
			got, err := st.build(newStream, newEvent, func() error { setup = true; return tt.setupErr }, release)
			if events != tt.wantEvents {
				t.Errorf("newEvent called %d times, want %d", events, tt.wantEvents)
			}
			if setup != tt.wantSetup {
				t.Errorf("setup called = %v, want %v", setup, tt.wantSetup)
			}
			if !slices.Equal(seen, tt.wantSeen) || (seen == nil) != (tt.wantSeen == nil) {
				t.Errorf("release saw %q, want %q (nil: release not called)", seen, tt.wantSeen)
			}
			if tt.wantOwner && got != st {
				t.Errorf("build returned %p, want the timer %p", got, st)
			}
			if !tt.wantOwner && got != nil {
				t.Errorf("build returned a timer %p, want nil", got)
			}
			checkErr(t, err, tt.wantErr, tt.wantIs)
		})
	}
}

// TestTeardown checks the deferred cleanup order runGPUInference and the
// profile test use: releases run last registered first, each failure is
// recorded in order with its identity, a failure that does not match
// errUndrained lets the earlier releases run, and one that does stops them
// so their resources are kept.
func TestTeardown(t *testing.T) {
	errFree := errors.New("free failed")
	errSync := errors.New("sync failed")
	undrained := fmt.Errorf("synchronize stream: %w: %w", errUndrained, errSync)
	for _, tt := range []struct {
		name      string
		fail      map[string]error
		wantCalls []string
		wantErr   string
		wantIs    []error
		wantHeld  bool
	}{
		{name: "all succeed", wantCalls: []string{"timer", "scratch", "weights", "modules"}},
		{
			name:      "dependent free fails",
			fail:      map[string]error{"scratch": errFree},
			wantCalls: []string{"timer", "scratch", "weights", "modules"},
			wantErr:   "scratch: free failed",
			wantIs:    []error{errFree},
		},
		{
			name:      "timer close fails drained",
			fail:      map[string]error{"timer": errFree, "weights": errFree},
			wantCalls: []string{"timer", "scratch", "weights", "modules"},
			wantErr:   "timer: free failed\nweights: free failed",
			wantIs:    []error{errFree},
		},
		{
			name:      "timer undrained",
			fail:      map[string]error{"timer": undrained, "scratch": errFree},
			wantCalls: []string{"timer"},
			wantErr:   "timer: synchronize stream: stream not drained: sync failed\nreleases held for undrained work",
			wantIs:    []error{errSync, errUndrained, errReleasesHeld},
			wantHeld:  true,
		},
		{
			name:      "dependent undrained",
			fail:      map[string]error{"weights": undrained},
			wantCalls: []string{"timer", "scratch", "weights"},
			wantErr:   "weights: synchronize stream: stream not drained: sync failed\nreleases held for undrained work",
			wantIs:    []error{errUndrained, errReleasesHeld},
			wantHeld:  true,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var calls []string
			release := func(name string) error {
				calls = append(calls, name)
				return tt.fail[name]
			}
			var td *teardown
			var err error
			func() {
				td = new(teardown)
				defer func() { err = td.err() }()
				for _, name := range []string{"modules", "weights", "scratch", "timer"} {
					defer func() {
						if td.held() {
							return
						}
						if rerr := release(name); rerr != nil {
							td.add(fmt.Errorf("%s: %w", name, rerr))
						}
					}()
				}
			}()
			if !slices.Equal(calls, tt.wantCalls) {
				t.Errorf("calls = %q, want %q", calls, tt.wantCalls)
			}
			if td.held() != tt.wantHeld {
				t.Errorf("held = %v, want %v", td.held(), tt.wantHeld)
			}
			checkErr(t, err, tt.wantErr, tt.wantIs)
			if !tt.wantHeld && errors.Is(err, errReleasesHeld) {
				t.Errorf("errors.Is(err, errReleasesHeld) = true with nothing held")
			}
		})
	}
}

// TestStepTimerSource checks, from source only, that newStepTimer builds
// with the device's constructors and the timer's own setup and close, and
// that close is closeStepTimer on the timer's stream and events. The
// behaviour of build, close and closeStepTimer is tested above; no CPU test
// can build a real timer.
func TestStepTimerSource(t *testing.T) {
	f := parseFile(t, "steptime.go")
	for _, tt := range []struct{ recv, name, want string }{
		{"", "newStepTimer", "st.build(dev.NewStream, dev.NewEvent, st.setup, st.close)"},
		{"*stepTimer", "close", "closeStepTimer(&st.stream, &st.drained, []**cuda.Event{&st.anchor, &st.start, &st.end})"},
	} {
		got := "not found"
		for _, decl := range f.Decls {
			fd, ok := decl.(*ast.FuncDecl)
			if !ok || fd.Name.Name != tt.name || (fd.Recv == nil) != (tt.recv == "") {
				continue
			}
			if fd.Recv != nil && types.ExprString(fd.Recv.List[0].Type) != tt.recv {
				continue
			}
			got = "no final return"
			if n := len(fd.Body.List); n > 0 {
				if r, ok := fd.Body.List[n-1].(*ast.ReturnStmt); ok && len(r.Results) == 1 {
					var b strings.Builder
					if err := format.Node(&b, token.NewFileSet(), r.Results[0]); err != nil {
						t.Fatal(err)
					}
					got = b.String()
				}
			}
		}
		if got != tt.want {
			t.Errorf("%s returns %s, want %s", tt.name, got, tt.want)
		}
	}
}

// TestTeardownWiring checks, from source only, the shape TestTeardown
// tests the behaviour of. In runGPUInference and the profile test, every
// deferred release is a function literal; the step timer's is registered
// last, so it runs first, and before newStepTimer's error is checked; and
// every other one first returns when the timer's stream did not drain, then
// records its release error. run keeps the device when runGPUInference
// held releases, and runGPUInference and run return their release errors.
func TestTeardownWiring(t *testing.T) {
	for _, tt := range []struct {
		file, fn  string
		guard     string   // the statement each guarded release starts with
		record    string   // how a guarded release records its error
		unguarded []string // releases allowed to run regardless
		last      string   // the release registered last, if it must be
		guarded   int
	}{
		{"main.go", "runGPUInference", "if td.held() {\n\treturn\n}", "td.add(", []string{"timer"}, "timer", 6},
		{"main.go", "run", "if errors.Is(rerr, errReleasesHeld) {\n\treturn\n}", "rerr = errors.Join(rerr, ", []string{"f", "profSession"}, "", 1},
		{"profile_cuda_test.go", "TestStepTimerExcludesPostEnqueueHostDelay", "if undrained {\n\treturn\n}", "t.Error", []string{"timer"}, "timer", 3},
	} {
		guarded, last, bad := teardownGuards(t, tt.file, tt.fn, tt.guard, tt.record, tt.unguarded)
		for _, b := range bad {
			t.Errorf("%s: %s: %s", tt.file, tt.fn, b)
		}
		if guarded != tt.guarded {
			t.Errorf("%s: %s: %d guarded releases, want %d", tt.file, tt.fn, guarded, tt.guarded)
		}
		if tt.last != "" && last != tt.last {
			t.Errorf("%s: %s: last registered release is of %s, want %s", tt.file, tt.fn, last, tt.last)
		}
	}
	for _, c := range []struct{ file, want string }{
		// runGPUInference reports its release errors in its result.
		{"main.go", "td := new(teardown)\n\tdefer func() {\n\t\trerr = errors.Join(rerr, td.err())\n\t}()\n"},
		// The step timer's release is registered before its error is
		// checked, and records into td.
		{"main.go", "timer, err = newStepTimer(dev, prof, tracer, forwardCU)\n\t\tif timer != nil {\n\t\t\tdefer func() {\n\t\t\t\tif cerr := timer.close(); cerr != nil {\n\t\t\t\t\ttd.add(fmt.Errorf(\"close step timer: %w\", cerr))\n"},
		// The profile test holds its buffers when the timer is undrained.
		{"profile_cuda_test.go", "timer, err := newStepTimer(dev, nil, nil, \"probe\")\n\tif timer != nil {\n\t\tdefer func() {\n\t\t\tif err := timer.close(); err != nil {\n\t\t\t\tt.Errorf(\"close step timer: %v\", err)\n\t\t\t\tundrained = errors.Is(err, errUndrained)\n"},
	} {
		src, err := os.ReadFile(c.file)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(src), c.want) {
			t.Errorf("%s does not contain %q", c.file, c.want)
		}
	}
}

// teardownGuards inspects the deferred calls in fn of file that release an
// owner. It counts those that start with guard and then release the owner
// in an if statement whose body records the error with a call starting with
// record, and returns the owner of the last one registered. It reports a
// releasing defer that is not a function literal, and one that is not
// guarded so, unless it releases one of unguarded.
func teardownGuards(t *testing.T, file, fn, guard, record string, unguarded []string) (n int, last string, bad []string) {
	t.Helper()
	f := parseFile(t, file)
	var body *ast.BlockStmt
	for _, decl := range f.Decls {
		if fd, ok := decl.(*ast.FuncDecl); ok && fd.Name.Name == fn {
			body = fd.Body
		}
	}
	if body == nil {
		return 0, "", []string{"not found"}
	}
	text := func(n ast.Node) string {
		var b strings.Builder
		if err := format.Node(&b, token.NewFileSet(), n); err != nil {
			t.Fatal(err)
		}
		return b.String()
	}
	released := func(n ast.Node) string {
		owner := ""
		ast.Inspect(n, func(m ast.Node) bool {
			if call, ok := m.(*ast.CallExpr); ok {
				if sel, ok := call.Fun.(*ast.SelectorExpr); ok && (sel.Sel.Name == "Close" || sel.Sel.Name == "Free" || sel.Sel.Name == "close") {
					if id, ok := sel.X.(*ast.Ident); ok {
						owner = id.Name
					}
				}
			}
			return owner == ""
		})
		return owner
	}
	ast.Inspect(body, func(node ast.Node) bool {
		d, ok := node.(*ast.DeferStmt)
		if !ok {
			return true
		}
		owner := released(d.Call)
		if owner == "" {
			return true
		}
		last = owner
		lit, ok := d.Call.Fun.(*ast.FuncLit)
		if !ok {
			bad = append(bad, fmt.Sprintf("release of %s is not a function literal", owner))
			return true
		}
		if slices.Contains(unguarded, owner) {
			return true
		}
		list := lit.Body.List
		if len(list) != 2 || text(list[0]) != guard {
			bad = append(bad, fmt.Sprintf("release of %s does not start with %q", owner, guard))
			return true
		}
		is, ok := list[1].(*ast.IfStmt)
		init, iok := is.Init.(*ast.AssignStmt)
		if !ok || !iok || len(init.Lhs) != 1 || released(init) != owner || is.Else != nil || len(is.Body.List) != 1 {
			bad = append(bad, fmt.Sprintf("release of %s is not a single if releasing it", owner))
			return true
		}
		v, ok := init.Lhs[0].(*ast.Ident)
		rec := text(is.Body.List[0])
		if !ok || !strings.HasPrefix(rec, record) || !mentions(is.Body.List[0], v.Name) {
			bad = append(bad, fmt.Sprintf("release of %s handles its error with %q, want a call starting %q with %s", owner, rec, record, init.Lhs[0]))
			return true
		}
		n++
		return true
	})
	return n, last, bad
}

// parseFile parses name in the package directory.
func parseFile(t *testing.T, name string) *ast.File {
	t.Helper()
	f, err := parser.ParseFile(token.NewFileSet(), name, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	return f
}
