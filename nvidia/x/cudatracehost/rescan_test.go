package cudatracehost

import (
	"context"
	"errors"
	"testing"
	"time"
)

func rescanFixture() *threadRescanner {
	return &threadRescanner{initial: map[uint32]uint64{1: 10}, known: map[uint32]uint64{1: 10}, limit: 2, probes: 2, now: func() int64 { return 1 }, process: func() error { return nil }, discover: func() ([]ThreadSnapshot, error) { return []ThreadSnapshot{{1, 10}, {2, 20}}, nil }, generation: func(int) (uint64, error) { return 20, nil }, attach: func(_ []ThreadSnapshot, v func() error) error {
		if err := v(); err != nil {
			return err
		}
		return v()
	}}
}

func TestRescanAttachAndBounds(t *testing.T) {
	s := rescanFixture()
	calls := 0
	s.attach = func(ts []ThreadSnapshot, v func() error) error {
		calls++
		if len(ts) != 1 || ts[0].TID != 2 {
			t.Fatal(ts)
		}
		if err := v(); err != nil {
			return err
		}
		return v()
	}
	r, err := s.step(context.Background())
	if err != nil || !r.AttachmentAttempted || r.Status != "generation_verified_attachment" {
		t.Fatalf("%+v %v", r, err)
	}
	if _, err = s.step(context.Background()); err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Fatal(calls)
	}
	if _, err = s.step(context.Background()); err == nil {
		t.Fatal("budget admitted")
	}
	if _, err = s.step(context.Background()); err == nil {
		t.Fatal("failure retried")
	}
	if len(s.history) != 3 {
		t.Fatal(len(s.history))
	}
	copy := s.snapshot()
	copy[0].Threads[0].Generation = 99
	if s.history[0].Threads[0].Generation != 20 {
		t.Fatal("aliased history")
	}
}

func TestRescanRefusals(t *testing.T) {
	for _, name := range []string{"recycle", "restart", "before", "after", "capacity", "duplicate", "cancel"} {
		t.Run(name, func(t *testing.T) {
			s := rescanFixture()
			attached := false
			switch name {
			case "recycle":
				s.known[2] = 21
			case "restart":
				s.process = func() error { return errors.New("restart") }
			case "before":
				s.generation = func(int) (uint64, error) { return 21, nil }
			case "after":
				s.attach = func(_ []ThreadSnapshot, v func() error) error {
					attached = true
					s.generation = func(int) (uint64, error) { return 21, nil }
					return v()
				}
			case "capacity":
				s.probes = 128
				s.known[3] = 30
			case "duplicate":
				s.discover = func() ([]ThreadSnapshot, error) { return []ThreadSnapshot{{2, 20}, {2, 20}}, nil }
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if name == "cancel" {
				cancel()
			}
			r, err := s.step(ctx)
			if err == nil {
				t.Fatal("contradiction admitted")
			}
			if name == "after" && (!attached || !r.AttachmentAttempted || r.BlindInterval != "attachment_and_rollback_unknown") {
				t.Fatal(r)
			}
			if _, err = s.step(context.Background()); err == nil {
				t.Fatal("retried")
			}
		})
	}
}

func TestRescanWorkerCancelBlockedDelivery(t *testing.T) {
	c := &Collector{rescanner: rescanFixture(), events: make(chan transportEvent)}
	started := make(chan struct{})
	c.rescanner.attach = func(_ []ThreadSnapshot, v func() error) error { close(started); return v() }
	c.startRescans(context.Background(), time.Millisecond)
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("worker did not attach")
	}
	done := make(chan struct{})
	go func() { c.stopRescans(); close(done) }()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("blocked shutdown")
	}
}

func TestRescanConfiguration(t *testing.T) {
	for _, cfg := range []Config{{ThreadRescanInterval: time.Millisecond, MaxThreadRescans: 1}, {ThreadRescanInterval: time.Second}, {MaxThreadRescans: 129}, {ThreadRescanInterval: -1}} {
		if validateRescan(cfg, "bpf") == nil {
			t.Fatal(cfg)
		}
	}
	if err := validateRescan(Config{ThreadRescanInterval: time.Second, MaxThreadRescans: 2}, "bpf"); err != nil {
		t.Fatal(err)
	}
	if validateRescan(Config{ThreadRescanInterval: time.Second, MaxThreadRescans: 2}, "tracefs") == nil {
		t.Fatal("tracefs rescan admitted")
	}
}

func TestRescanWorkerBudget(t *testing.T) {
	c := &Collector{rescanner: rescanFixture(), events: make(chan transportEvent, 4)}
	c.rescanner.limit = 1
	c.startRescans(context.Background(), time.Millisecond)
	defer c.stopRescans()
	for i := 0; i < 2; i++ {
		select {
		case e := <-c.events:
			if e.coverage == nil || (i == 1) != (e.err != nil) {
				t.Fatalf("event %d: %+v", i, e)
			}
		case <-time.After(time.Second):
			t.Fatal("missing bounded completion")
		}
	}
	c.rescanWait.Wait()
	if len(c.rescanner.snapshot()) != 2 {
		t.Fatal("unbounded history")
	}
}

func TestRescanConcurrentStopClose(t *testing.T) {
	c := &Collector{rescanner: rescanFixture(), events: make(chan transportEvent), done: make(chan struct{})}
	started := make(chan struct{})
	c.rescanner.attach = func(_ []ThreadSnapshot, v func() error) error { close(started); return v() }
	c.startRescans(context.Background(), time.Millisecond)
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("worker did not run")
	}
	complete := make(chan struct{}, 2)
	go func() { c.StopAndStats(); complete <- struct{}{} }()
	go func() { c.Close(); complete <- struct{}{} }()
	for i := 0; i < 2; i++ {
		select {
		case <-complete:
		case <-time.After(time.Second):
			t.Fatal("lifecycle deadlock")
		}
	}
}
