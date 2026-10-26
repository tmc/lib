//go:build unix

package cudahotswap

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudalock"
)

// waitHolder is another process's owner record. Its At is two hours before
// waitEpoch, so its age cannot pass for a request's wait.
var (
	waitHolder = cudalock.Owner{PID: 4242, Host: "gpuhost", Label: "other-job", Epoch: 1767323045, At: time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)}
	waitEpoch  = time.Date(2026, 1, 2, 5, 4, 5, 0, time.UTC)
)

// waitClock is a controlled stand-in for the host monotonic clock.
type waitClock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *waitClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *waitClock) Advance(d time.Duration) {
	c.mu.Lock()
	c.now = c.now.Add(d)
	c.mu.Unlock()
}

// blockedAcquire is one raw acquisition held until release or cancellation.
type blockedAcquire struct {
	name    string
	release chan struct{}
}

// waitHarness drives public requests into blocked raw acquisition beneath the
// service's real publication wrapper. Owner observation reads only dir.
type waitHarness struct {
	s        *Service
	dir      string
	clock    *waitClock
	block    atomic.Bool
	entered  chan *blockedAcquire
	calls    atomic.Int64
	returned atomic.Int64
	granted  atomic.Int64
}

func newWaitHarness(t *testing.T, l *serviceLoader) *waitHarness {
	t.Helper()
	s, err := NewService(l)
	if err != nil {
		t.Fatal(err)
	}
	h := &waitHarness{s: s, dir: t.TempDir(), clock: &waitClock{now: waitEpoch}, entered: make(chan *blockedAcquire, 8)}
	s.waits.now = h.clock.Now
	s.waits.observe = func() (cudalock.Owner, string) { return observeOwner(h.dir) }
	s.rawAcquire = func(ctx context.Context, name string) (io.Closer, error) {
		h.calls.Add(1)
		defer h.returned.Add(1)
		if h.block.Load() {
			b := &blockedAcquire{name: name, release: make(chan struct{})}
			h.entered <- b
			select {
			case <-b.release:
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
		h.granted.Add(1)
		return new(serviceCloser), nil
	}
	return h
}

func (h *waitHarness) next(t *testing.T) *blockedAcquire {
	t.Helper()
	select {
	case b := <-h.entered:
		return b
	case <-time.After(5 * time.Second):
		t.Fatal("request did not enter acquisition")
		return nil
	}
}

func (h *waitHarness) writeOwner(t *testing.T, data string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(h.dir, "owner"), []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
}

func receive[T any](t *testing.T, c <-chan T) T {
	t.Helper()
	select {
	case v := <-c:
		return v
	case <-time.After(5 * time.Second):
		t.Fatal("request did not finish")
		var zero T
		return zero
	}
}

func sameOwner(a, b cudalock.Owner) bool {
	return a.PID == b.PID && a.Host == b.Host && a.Label == b.Label && a.Epoch == b.Epoch && a.At.Equal(b.At)
}

// startWaitRequest blocks a public Register or Push of "kernel" in acquisition.
// For push it first registers the original and configures an application check.
func startWaitRequest(t *testing.T, ctx context.Context, h *waitHarness, l *serviceLoader, op string) (*blockedAcquire, <-chan error) {
	t.Helper()
	s := h.s
	if op == "push" {
		registerService(t, s)
		if _, err := s.SetCheck("kernel", CheckConfig{Checker: func(context.Context, State) error { return nil }}, "check"); err != nil {
			t.Fatal(err)
		}
		l.kernel = &serviceKernel{raw: 2}
	}
	h.block.Store(true)
	done := make(chan error, 1)
	go func() {
		var err error
		if op == "register" {
			_, err = s.Register(ctx, "kernel", Source{Code: "source", Entry: "entry"}, "original")
		} else {
			_, err = s.Push(ctx, "kernel", Source{Code: "candidate", Entry: "entry"}, "candidate")
		}
		done <- err
	}()
	return h.next(t), done
}

func TestServiceWaitingReportsHolderAndWait(t *testing.T) {
	for _, tt := range []struct {
		op      string
		request uint64
		loads   int
	}{
		{"register", 1, 0},
		{"push", 2, 1},
	} {
		t.Run(tt.op, func(t *testing.T) {
			l := &serviceLoader{kernel: &serviceKernel{raw: 1}}
			h := newWaitHarness(t, l)
			s := h.s
			h.writeOwner(t, waitHolder.String())
			b, done := startWaitRequest(t, t.Context(), h, l, tt.op)
			h.clock.Advance(1500 * time.Millisecond)

			waits := s.Waiting()
			if len(waits) != 1 {
				t.Fatalf("Waiting() = %+v, want one blocked request", waits)
			}
			w := waits[0]
			if w.Request != tt.request || w.Name != "kernel" || w.Label != "cudahotswap:kernel" || b.name != "kernel" {
				t.Fatalf("request = %d %q %q, want %d kernel cudahotswap:kernel", w.Request, w.Name, w.Label, tt.request)
			}
			if !sameOwner(w.Holder, waitHolder) || w.HolderUnavailable != "" {
				t.Fatalf("holder = %+v (%q), want observed %+v", w.Holder, w.HolderUnavailable, waitHolder)
			}
			if w.Waited != 1500*time.Millisecond {
				t.Fatalf("Waited = %v, want 1.5s since this request entered acquisition", w.Waited)
			}
			if age := h.clock.Now().Sub(w.Holder.At); age != 2*time.Hour+1500*time.Millisecond || w.Waited == age {
				t.Fatalf("Waited = %v conflated with holder age %v", w.Waited, age)
			}
			if l.loads != tt.loads {
				t.Fatalf("loads = %d before acquisition, want %d", l.loads, tt.loads)
			}
			if tt.op == "register" {
				if len(s.List()) != 0 {
					t.Fatal("pending first Register published a kernel")
				}
				if _, err := s.Status("kernel"); err == nil {
					t.Fatal("pending first Register has a status")
				}
			} else if state, err := s.Status("kernel"); err != nil || state.Generation != 1 || state.Version != 1 {
				t.Fatalf("pending Push changed status: %+v, %v", state, err)
			}

			h.clock.Advance(500 * time.Millisecond)
			if waits := s.Waiting(); len(waits) != 1 || waits[0].Request != tt.request || waits[0].Waited != 2*time.Second {
				t.Fatalf("second Waiting() = %+v, want request %d at 2s", waits, tt.request)
			}

			close(b.release)
			if err := receive(t, done); err != nil {
				t.Fatal(err)
			}
			if waits := s.Waiting(); waits != nil {
				t.Fatalf("acquired request still waiting: %+v", waits)
			}
			if l.loads != tt.loads+1 {
				t.Fatalf("loads = %d after acquisition, want %d", l.loads, tt.loads+1)
			}
			h.block.Store(false)
			if err := s.Close(context.Background()); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestServiceWaitingCancelRetiresRequest(t *testing.T) {
	for _, op := range []string{"register", "push"} {
		t.Run(op, func(t *testing.T) {
			l := &serviceLoader{kernel: &serviceKernel{raw: 1}}
			h := newWaitHarness(t, l)
			s := h.s
			h.writeOwner(t, waitHolder.String())
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			_, done := startWaitRequest(t, ctx, h, l, op)
			granted, loads := h.granted.Load(), l.loads
			if waits := s.Waiting(); len(waits) != 1 {
				t.Fatalf("Waiting() = %+v, want one blocked request", waits)
			}

			cancel()
			if err := receive(t, done); !errors.Is(err, context.Canceled) {
				t.Fatalf("canceled request error = %v, want context.Canceled", err)
			}
			if waits := s.Waiting(); waits != nil {
				t.Fatalf("canceled request still waiting: %+v", waits)
			}
			if h.calls.Load() != h.returned.Load() || h.granted.Load() != granted || l.loads != loads {
				t.Fatal("canceled request left or completed an acquisition")
			}
			if len(s.cleanup) != 0 {
				t.Fatalf("canceled request retained %d cleanup owners", len(s.cleanup))
			}
			if op == "register" {
				if len(s.List()) != 0 {
					t.Fatal("canceled first Register published a kernel")
				}
			} else if state, err := s.Status("kernel"); err != nil || state.Generation != 1 || state.Version != 1 {
				t.Fatalf("canceled Push changed status: %+v, %v", state, err)
			}
			if data, err := os.ReadFile(filepath.Join(h.dir, "owner")); err != nil || string(data) != waitHolder.String() {
				t.Fatalf("observation changed the owner file: %q, %v", data, err)
			}
			h.block.Store(false)
			if err := s.Close(context.Background()); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestServiceWaitingUnavailableHolder(t *testing.T) {
	l := &serviceLoader{kernel: &serviceKernel{raw: 1}}
	h := newWaitHarness(t, l)
	s := h.s
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	_, done := startWaitRequest(t, ctx, h, l, "register")
	h.clock.Advance(time.Second)

	for _, tt := range []struct {
		name  string
		setup func(t *testing.T, dir string)
		want  string
	}{
		{"missing", func(*testing.T, string) {}, "owner file missing"},
		{"lock-dir-missing", nil, "owner file missing"},
		{"empty", ownerData(""), "owner file malformed"},
		{"malformed", ownerData("pid=abc host=gpuhost"), "owner file malformed"},
		{"no-pid", ownerData("host=gpuhost label=other-job"), "owner file malformed"},
		{"oversized", ownerData(waitHolder.String() + strings.Repeat(" ", maxOwnerBytes)), "owner file unreadable"},
		{"nonregular", func(t *testing.T, dir string) {
			if err := os.Mkdir(filepath.Join(dir, "owner"), 0o755); err != nil {
				t.Fatal(err)
			}
		}, "owner file unreadable"},
		{"symlink", func(t *testing.T, dir string) {
			target := filepath.Join(dir, "target")
			if err := os.WriteFile(target, []byte(waitHolder.String()), 0o644); err != nil {
				t.Fatal(err)
			}
			if err := os.Symlink(target, filepath.Join(dir, "owner")); err != nil {
				t.Fatal(err)
			}
		}, "owner file unreadable"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			h.dir = t.TempDir()
			if tt.setup == nil {
				h.dir = filepath.Join(h.dir, "absent")
			} else {
				tt.setup(t, h.dir)
			}
			waits := s.Waiting()
			if len(waits) != 1 || waits[0].Name != "kernel" || waits[0].Waited != time.Second {
				t.Fatalf("Waiting() = %+v, want one request waiting 1s", waits)
			}
			if w := waits[0]; w.Holder != (cudalock.Owner{}) || !strings.HasPrefix(w.HolderUnavailable, tt.want) {
				t.Fatalf("holder = %+v (%q), want zero owner and %q", w.Holder, w.HolderUnavailable, tt.want)
			}
		})
	}

	cancel()
	if err := receive(t, done); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	h.block.Store(false)
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func ownerData(data string) func(*testing.T, string) {
	return func(t *testing.T, dir string) {
		if err := os.WriteFile(filepath.Join(dir, "owner"), []byte(data), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

func TestServiceWaitingConcurrentRequests(t *testing.T) {
	l := &serviceLoader{kernel: &serviceKernel{raw: 1}}
	h := newWaitHarness(t, l)
	s := h.s
	registerService(t, s)
	pin, err := s.Pin("kernel")
	if err != nil {
		t.Fatal(err)
	}
	h.writeOwner(t, waitHolder.String())
	h.block.Store(true)
	ctx := t.Context()

	type launchResult struct {
		pending *PendingLaunch
		err     error
	}
	launches := make(chan launchResult, 2)
	released := make(chan error, 1)
	launch := func() {
		p, err := s.Launch(ctx, "kernel", cudalaunch.Config{})
		launches <- launchResult{p, err}
	}
	// Enter one at a time so serials and waits are deterministic.
	var blocked []*blockedAcquire
	for _, start := range []func(){launch, launch, func() { released <- pin.Release(ctx) }} {
		go start()
		blocked = append(blocked, h.next(t))
		h.clock.Advance(time.Second)
	}

	type want struct {
		request uint64
		name    string
		waited  time.Duration
	}
	check := func(wants ...want) {
		t.Helper()
		waits := s.Waiting()
		if len(waits) != len(wants) {
			t.Fatalf("Waiting() = %+v, want %d requests", waits, len(wants))
		}
		for i, w := range wants {
			got := waits[i]
			if got.Request != w.request || got.Name != w.name || got.Label != lockLabel(w.name) || got.Waited != w.waited {
				t.Fatalf("Waiting()[%d] = %d %q %q %v, want %d %q %v", i, got.Request, got.Name, got.Label, got.Waited, w.request, w.name, w.waited)
			}
			if !sameOwner(got.Holder, waitHolder) || got.HolderUnavailable != "" {
				t.Fatalf("Waiting()[%d] holder = %+v (%q)", i, got.Holder, got.HolderUnavailable)
			}
		}
	}
	all := []want{{2, "kernel", 3 * time.Second}, {3, "kernel", 2 * time.Second}, {4, "release", time.Second}}
	check(all...)

	// A caller's copy is detached from service state.
	waits := s.Waiting()
	waits[0].Name = "mutated"
	waits[0].Holder.Label = "mutated"
	waits[1].Waited = 0
	waits[2].Request = 99
	check(all...)

	// Retiring one same-name request leaves the other one's identity and wait.
	close(blocked[1].release)
	first := receive(t, launches)
	if first.err != nil {
		t.Fatal(first.err)
	}
	check(all[0], all[2])

	close(blocked[0].release)
	close(blocked[2].release)
	second := receive(t, launches)
	if second.err != nil {
		t.Fatal(second.err)
	}
	if err := receive(t, released); err != nil {
		t.Fatal(err)
	}
	if waits := s.Waiting(); waits != nil {
		t.Fatalf("acquired requests still waiting: %+v", waits)
	}
	h.block.Store(false)
	for _, r := range []launchResult{first, second} {
		if err := r.pending.Wait(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if h.calls.Load() != h.returned.Load() {
		t.Fatal("acquisition left running")
	}
}

func TestServiceAcquirePreservesRawResult(t *testing.T) {
	sentinel := errors.New("partial acquisition")
	for _, tt := range []struct {
		name  string
		lease io.Closer
		err   error
		panic bool
	}{
		{"lease", new(serviceCloser), nil, false},
		{"error", nil, sentinel, false},
		{"partial", new(serviceCloser), sentinel, false},
		{"panic", nil, nil, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			s, err := NewService(&serviceLoader{})
			if err != nil {
				t.Fatal(err)
			}
			s.waits.observe = func() (cudalock.Owner, string) { return cudalock.Owner{}, "not observed" }
			type key struct{}
			ctx := context.WithValue(t.Context(), key{}, tt.name)
			var got context.Context
			var during []LockWait
			s.rawAcquire = func(ctx context.Context, name string) (io.Closer, error) {
				got = ctx
				during = s.Waiting()
				if tt.panic {
					panic(sentinel)
				}
				return tt.lease, tt.err
			}
			var lease io.Closer
			recovered := func() (r any) {
				defer func() { r = recover() }()
				lease, err = s.acquire(ctx, "kernel")
				return nil
			}()
			if tt.panic && recovered != sentinel || !tt.panic && recovered != nil {
				t.Fatalf("recovered %v", recovered)
			}
			if lease != tt.lease || err != tt.err || got != ctx {
				t.Fatalf("acquire = %v, %v with %v; want raw %v, %v with request context", lease, err, got, tt.lease, tt.err)
			}
			if len(during) != 1 || during[0].Name != "kernel" || during[0].HolderUnavailable != "not observed" {
				t.Fatalf("Waiting() during acquisition = %+v", during)
			}
			if waits := s.Waiting(); waits != nil {
				t.Fatalf("finished acquisition still waiting: %+v", waits)
			}
		})
	}
}
