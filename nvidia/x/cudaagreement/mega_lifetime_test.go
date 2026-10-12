package cudaagreement

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func TestMegaLifetimeEnqueueRetention(t *testing.T) {
	for _, fail := range []bool{false, true} {
		t.Run(map[bool]string{false: "queued", true: "enqueue error"}[fail], func(t *testing.T) {
			var events []string
			joinErr := context.DeadlineExceeded
			p := &megaInvocation{join: func(context.Context) error {
				events = append(events, "join")
				return joinErr
			}}
			e := &MegaExecutor{pending: p}
			for _, name := range []string{"input", "output", "scratch", "completed", "abort", "order"} {
				if err := p.borrow(func() (func(), error) {
					events = append(events, "borrow "+name)
					return func() { events = append(events, "release "+name) }, nil
				}); err != nil {
					t.Fatal(err)
				}
			}
			err := p.enqueue(func() error {
				events = append(events, "enqueue")
				if fail {
					return errors.New("attempted enqueue failed")
				}
				return nil
			})
			if (err != nil) != fail {
				t.Fatal(err)
			}
			e.rollbackSubmission()
			if len(p.releases) != 6 || e.pending != p {
				t.Fatal("attempted submission released ownership")
			}
			if _, err := e.Execute(&cuda.Stream{}, nil, nil); !errors.Is(err, cudamem.ErrInUse) {
				t.Fatalf("reuse: %v", err)
			}
			if _, err := e.buffer("scratch", 1); !errors.Is(err, cudamem.ErrInUse) {
				t.Fatalf("resize: %v", err)
			}
			if err := e.orderBuffer(1); !errors.Is(err, cudamem.ErrInUse) {
				t.Fatalf("order resize: %v", err)
			}
			if err := e.Drain(nil); !errors.Is(err, context.DeadlineExceeded) || len(p.releases) != 6 {
				t.Fatalf("failed drain released: %v", err)
			}
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			before := len(events)
			if err := e.Drain(ctx); !errors.Is(err, context.Canceled) || len(events) != before {
				t.Fatalf("cancelled drain changed ownership: %v", err)
			}
			joinErr = nil
			if err := e.Drain(nil); err != nil {
				t.Fatal(err)
			}
			want := []string{"borrow input", "borrow output", "borrow scratch", "borrow completed", "borrow abort", "borrow order", "enqueue", "join", "join", "release order", "release abort", "release completed", "release scratch", "release output", "release input"}
			if !reflect.DeepEqual(events, want) || e.pending != nil {
				t.Fatalf("events: %q", events)
			}
			if err := e.Drain(nil); err != nil || !reflect.DeepEqual(events, want) {
				t.Fatal("repeated drain changed released ownership", err)
			}
		})
	}
}

func TestMegaLifetimeBeforeEnqueueRollback(t *testing.T) {
	released := 0
	e := &MegaExecutor{pending: &megaInvocation{}}
	if err := e.pending.borrow(func() (func(), error) {
		return func() { released++ }, nil
	}); err != nil {
		t.Fatal(err)
	}
	want := errors.New("borrow refused")
	if err := e.pending.borrow(func() (func(), error) { return nil, want }); !errors.Is(err, want) {
		t.Fatal(err)
	}
	e.rollbackSubmission()
	e.rollbackSubmission()
	if released != 1 || e.pending != nil {
		t.Fatal("pre-enqueue rollback did not release exactly once")
	}
}

func TestMegaLifetimeFinalCopiesAndClose(t *testing.T) {
	var events []string
	joinErr := errors.New("join failed")
	p := &megaInvocation{join: func(context.Context) error {
		events = append(events, "join")
		return joinErr
	}}
	e := &MegaExecutor{pending: p, scratch: map[string]cudamem.Slice[uint32]{"scratch": {}}}
	if err := p.borrow(func() (func(), error) {
		return func() { events = append(events, "release") }, nil
	}); err != nil {
		t.Fatal(err)
	}
	if err := p.enqueue(func() error { events = append(events, "worker"); return nil }); err != nil {
		t.Fatal(err)
	}
	// The worker join used for inspecting publication is not the final drain.
	joinErr = nil
	if err := p.join(nil); err != nil {
		t.Fatal(err)
	}
	if len(p.releases) != 1 {
		t.Fatal("worker join released storage before output copy")
	}
	copyErr := errors.New("copy enqueue failed")
	if err := p.enqueue(func() error { events = append(events, "copy"); return copyErr }); !errors.Is(err, copyErr) {
		t.Fatal(err)
	}
	joinErr = errors.New("final join failed")
	if err := e.Close(); err == nil || e.pending != p || len(e.scratch) != 1 || len(p.releases) != 1 {
		t.Fatal("Close released storage after a failed join", err)
	}
	joinErr = nil
	if err := e.Close(); err != nil || e.pending != nil || len(e.scratch) != 0 {
		t.Fatal("Close retry did not drain and free", err)
	}
	want := []string{"worker", "join", "copy", "join", "join", "release"}
	if !reflect.DeepEqual(events, want) {
		t.Fatalf("events: %q", events)
	}
}

func TestMegaLifetimeConcurrentDrain(t *testing.T) {
	joins, releases := 0, 0
	e := &MegaExecutor{pending: &megaInvocation{attempted: true, join: func(context.Context) error {
		joins++
		return nil
	}, releases: []func(){func() { releases++ }}}}
	var wg sync.WaitGroup
	for range 16 {
		wg.Go(func() {
			if err := e.Drain(nil); err != nil {
				t.Error(err)
			}
		})
	}
	wg.Wait()
	if joins != 1 || releases != 1 || e.pending != nil {
		t.Fatalf("joins %d, releases %d, pending %v", joins, releases, e.pending)
	}
}
