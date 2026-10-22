//go:build linux

package dsv4expertcache

import (
	"errors"
	"reflect"
	"testing"
)

type fakeCUDAUpload struct {
	buf      []byte
	live     map[string]bool
	calls    map[string]int
	fail     map[string]int
	pending  func()
	retained int
	outputs  []record
	events   []*fakeCUDAEvent
}

func fakeUpload() *fakeCUDAUpload {
	return &fakeCUDAUpload{live: make(map[string]bool), calls: make(map[string]int), fail: make(map[string]int)}
}
func (b *fakeCUDAUpload) call(name string) error {
	b.calls[name]++
	if b.fail[name] > 0 {
		b.fail[name]--
		return errors.New(name)
	}
	return nil
}
func (b *fakeCUDAUpload) allocate(slots int, strides [3]int) error {
	for _, name := range []string{"stream", "host", "slab0", "slab1", "slab2"} {
		if err := b.call("alloc-" + name); err != nil {
			return err
		}
		b.live[name] = true
	}
	b.buf = make([]byte, max(strides[0], strides[1], strides[2]))
	return nil
}
func (b *fakeCUDAUpload) stage() []byte { return b.buf }
func (b *fakeCUDAUpload) newEvent() (cudaUploadEvent, error) {
	if err := b.call("event"); err != nil {
		return nil, err
	}
	e := &fakeCUDAEvent{b: b, live: true}
	b.events = append(b.events, e)
	if err := b.call("event-context"); err != nil {
		return e, err
	}
	return e, nil
}
func (b *fakeCUDAUpload) copy(component, slot int, src []byte) error {
	if b.pending != nil {
		return errors.New("staging reused before completion")
	}
	b.pending = func() { b.outputs = append(b.outputs, record{slot, component, string(src)}) }
	b.retained++
	return b.call("copy")
}
func (b *fakeCUDAUpload) finish() {
	if b.pending != nil {
		b.pending()
		b.pending = nil
	}
}
func (b *fakeCUDAUpload) drain() error {
	if !b.live["stream"] {
		return nil
	}
	if err := b.call("drain"); err != nil {
		return err
	}
	b.finish()
	b.retained = 0
	return nil
}
func (b *fakeCUDAUpload) free(name string) error {
	if !b.live[name] {
		return nil
	}
	if b.pending != nil {
		return errors.New("free with pending upload")
	}
	if err := b.call("free-" + name); err != nil {
		return err
	}
	b.live[name] = false
	return nil
}
func (b *fakeCUDAUpload) freeSlab(i int) error { return b.free([]string{"slab0", "slab1", "slab2"}[i]) }
func (b *fakeCUDAUpload) freeStage() error     { return b.free("host") }
func (b *fakeCUDAUpload) destroyStream() error { return b.free("stream") }

type fakeCUDAEvent struct {
	b              *fakeCUDAUpload
	live, recorded bool
}

func (e *fakeCUDAEvent) record() error {
	if err := e.b.call("record"); err != nil {
		return err
	}
	e.recorded = true
	return nil
}
func (e *fakeCUDAEvent) wait() error {
	if !e.live || !e.recorded {
		return errors.New("wait on invalid event")
	}
	if err := e.b.call("wait"); err != nil {
		return err
	}
	e.b.finish()
	return nil
}
func (e *fakeCUDAEvent) destroy() error {
	if !e.live {
		return errors.New("event destroyed twice")
	}
	if err := e.b.call("destroy"); err != nil {
		return err
	}
	e.live = false
	return nil
}
func makeUpload(t *testing.T, b *fakeCUDAUpload) *CUDAUpload {
	t.Helper()
	u, err := newCUDAUpload(b, 2, [3]int{2, 2, 2}, 12)
	if err != nil {
		t.Fatal(err)
	}
	return u
}

func TestCUDAUploadStageLifetime(t *testing.T) {
	b := fakeUpload()
	u := makeUpload(t, b)
	c := cache(t)
	l, _ := c.Acquire([]Key{key(1), key(0), key(1)})
	if err := u.Stage(source(t), l, false); err != nil {
		t.Fatal(err)
	}
	want := []record{{0, 0, "cd"}, {0, 1, "gh"}, {0, 2, "kl"}, {1, 0, "ab"}, {1, 1, "ef"}, {1, 2, "ij"}}
	if !reflect.DeepEqual(b.outputs, want) {
		t.Fatalf("uploads = %v, want %v", b.outputs, want)
	}
	if b.calls["destroy"] != 0 {
		t.Fatal("Wait destroyed lease-owned events")
	}
	if u.Close() == nil {
		t.Fatal("Close accepted active consumer lease")
	}
	if err := l.Release(&pendingEvent{t: &transfer{}, complete: true}); err != nil {
		t.Fatal(err)
	}
	if err := u.Close(); err != nil {
		t.Fatal(err)
	}
	if b.calls["destroy"] != 6 {
		t.Fatalf("destroy count %d", b.calls["destroy"])
	}
	before := b.calls["drain"]
	if err := u.Close(); err != nil || b.calls["drain"] != before {
		t.Fatal("Close not idempotent")
	}
}

func TestCUDAUploadCompletionRetries(t *testing.T) {
	b := fakeUpload()
	u := makeUpload(t, b)
	done, err := u.Submit(Upload{Slot: 0}, 0, b.buf)
	if err != nil {
		t.Fatal(err)
	}
	b.fail["wait"] = 1
	if done.Wait() == nil {
		t.Fatal("wait failure hidden")
	}
	if err := done.Wait(); err != nil {
		t.Fatal(err)
	}
	before := b.calls["wait"]
	if err := done.Wait(); err != nil || b.calls["wait"] != before {
		t.Fatal("successful completion did not remain successful")
	}
	if !b.events[0].live {
		t.Fatal("Wait destroyed event")
	}
	b.fail["destroy"] = 1
	if u.Close() == nil || len(u.events) != 1 {
		t.Fatal("failed event destroy lost owner")
	}
	if err := u.Close(); err != nil || len(u.events) != 0 {
		t.Fatal("event destroy retry failed")
	}
}

func TestCUDAUploadSubmissionFallback(t *testing.T) {
	for _, fault := range []string{"event", "copy", "record"} {
		t.Run(fault, func(t *testing.T) {
			b := fakeUpload()
			u := makeUpload(t, b)
			b.fail[fault] = 1
			if fault != "event" {
				b.fail["destroy"] = 1
			}
			done, err := u.Submit(Upload{Slot: 0}, 0, b.buf)
			if err == nil || done == nil {
				t.Fatal("failure lacks retained completion")
			}
			b.fail["drain"] = 1
			if done.Wait() == nil {
				t.Fatal("failed fallback drain hidden")
			}
			if err := done.Wait(); err != nil || b.pending != nil {
				t.Fatal("fallback did not drain work")
			}
			if err := u.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestCUDAUploadCloseRetries(t *testing.T) {
	for _, fault := range []string{"drain", "free-slab0", "free-slab1", "free-slab2", "free-host", "free-stream"} {
		t.Run(fault, func(t *testing.T) {
			b := fakeUpload()
			u := makeUpload(t, b)
			b.fail[fault] = 1
			if u.Close() == nil || u.closed {
				t.Fatal("failed cleanup lost owner")
			}
			if fault == "drain" && b.calls["free-host"] != 0 {
				t.Fatal("freed storage before drain")
			}
			if err := u.Close(); err != nil {
				t.Fatal(err)
			}
			for name, live := range b.live {
				if live {
					t.Fatalf("%s still live", name)
				}
			}
			for _, name := range []string{"slab0", "slab1", "slab2", "host", "stream"} {
				want := 1
				if fault == "free-"+name {
					want = 2
				}
				if b.calls["free-"+name] != want {
					t.Fatalf("%s free count %d, want %d", name, b.calls["free-"+name], want)
				}
			}
		})
	}
}

func TestCUDAUploadPartialAllocation(t *testing.T) {
	for _, fault := range []string{"alloc-stream", "alloc-host", "alloc-slab0", "alloc-slab1", "alloc-slab2"} {
		t.Run(fault, func(t *testing.T) {
			b := fakeUpload()
			b.fail[fault] = 1
			u, err := newCUDAUpload(b, 2, [3]int{2, 2, 2}, 12)
			if err == nil || u != nil {
				t.Fatal("allocation failure not cleaned")
			}
			for name, live := range b.live {
				if live {
					t.Fatalf("leaked %s", name)
				}
			}
		})
	}
	b := fakeUpload()
	b.fail["alloc-slab1"] = 1
	b.fail["free-host"] = 1
	u, err := newCUDAUpload(b, 2, [3]int{2, 2, 2}, 12)
	if err == nil || u == nil || !b.live["host"] {
		t.Fatal("failed constructor cleanup lost owner")
	}
	if err := u.Close(); err != nil || b.live["host"] {
		t.Fatal("constructor cleanup retry failed")
	}
}

func TestCUDAUploadPreflight(t *testing.T) {
	b := fakeUpload()
	u := makeUpload(t, b)
	l, _ := cache(t).Acquire([]Key{key(0)})
	if u.Stage(source(t), l, true) == nil || l.done || b.calls["event"] != 0 {
		t.Fatal("graph preflight mutated lease")
	}
	if err := l.Abort(); err != nil {
		t.Fatal(err)
	}
	if _, err := u.Submit(Upload{Slot: 0}, 0, make([]byte, 2)); err == nil {
		t.Fatal("foreign staging accepted")
	}
	if err := u.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestCUDAUploadFailedStageCannotReuse(t *testing.T) {
	b := fakeUpload()
	u := makeUpload(t, b)
	l, _ := cache(t).Acquire([]Key{key(0)})
	b.fail["wait"] = 2 // The initial wait and Abort both fail.
	if u.Stage(source(t), l, false) == nil || b.pending == nil {
		t.Fatal("failed stage did not leave pending work")
	}
	before := append([]byte(nil), b.buf...)
	next, _ := cache(t).Acquire([]Key{key(1)})
	if u.Stage(source(t), next, false) == nil || !reflect.DeepEqual(before, b.buf) {
		t.Fatal("failed adapter reused pending stage")
	}
	if err := next.Abort(); err != nil {
		t.Fatal(err)
	}
	if err := u.Close(); err != nil || b.pending != nil {
		t.Fatal("Close did not drain failed stage")
	}
}

func TestCUDAUploadEndedLeaseReclaimsEvents(t *testing.T) {
	b := fakeUpload()
	u := makeUpload(t, b)
	c := cache(t)
	l, _ := c.Acquire([]Key{key(0)})
	if err := u.Stage(source(t), l, false); err != nil {
		t.Fatal(err)
	}
	if err := l.Release(&pendingEvent{t: &transfer{}, complete: true}); err != nil {
		t.Fatal(err)
	}
	next, _ := c.Acquire([]Key{key(1)})
	b.fail["destroy"] = 1
	if u.Stage(source(t), next, false) == nil || b.calls["copy"] != 3 || len(u.events) != 1 {
		t.Fatal("failed event cleanup did not retain owner and block submission")
	}
	if err := u.Stage(source(t), next, false); err != nil {
		t.Fatal(err)
	}
	if len(u.events) != 3 {
		t.Fatalf("retained %d events, want one triplet", len(u.events))
	}
	if err := next.Release(&pendingEvent{t: &transfer{}, complete: true}); err != nil {
		t.Fatal(err)
	}
	if err := u.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestCUDAUploadEventCreatedWithError(t *testing.T) {
	b := fakeUpload()
	u := makeUpload(t, b)
	b.fail["event-context"] = 1
	done, err := u.Submit(Upload{Slot: 0}, 0, b.buf)
	if err == nil || done == nil || len(u.events) != 1 {
		t.Fatal("creation-success/context-error lost event owner")
	}
	if b.calls["copy"] != 0 || b.calls["record"] != 0 {
		t.Fatal("failed creation submitted work")
	}
	if err := done.Wait(); err != nil || b.calls["wait"] != 0 {
		t.Fatal("wait used unrecorded event")
	}
	b.fail["destroy"] = 1
	if u.Close() == nil || len(u.events) != 1 {
		t.Fatal("failed destroy lost event")
	}
	if err := u.Close(); err != nil || len(u.events) != 0 {
		t.Fatal("retained event not released")
	}
}

func TestCUDAUploadEndedLeaseDrainsRetentions(t *testing.T) {
	b := fakeUpload()
	u := makeUpload(t, b)
	c := cache(t)
	l, _ := c.Acquire([]Key{key(0)})
	if err := u.Stage(source(t), l, false); err != nil {
		t.Fatal(err)
	}
	if b.retained != 3 {
		t.Fatal("event wait incorrectly cleared stream retention")
	}
	if err := l.Release(&pendingEvent{t: &transfer{}, complete: true}); err != nil {
		t.Fatal(err)
	}
	next, _ := c.Acquire([]Key{key(1)})
	b.fail["drain"] = 1
	if u.Stage(source(t), next, false) == nil || b.calls["copy"] != 3 || b.retained != 3 || len(u.events) != 3 {
		t.Fatal("failed drain did not retain transfers and block staging")
	}
	if err := u.Stage(source(t), next, false); err != nil {
		t.Fatal(err)
	}
	if b.retained != 3 {
		t.Fatalf("retained %d copies, want one triplet", b.retained)
	}
	if err := next.Release(&pendingEvent{t: &transfer{}, complete: true}); err != nil {
		t.Fatal(err)
	}
	if err := u.Close(); err != nil || b.retained != 0 {
		t.Fatal("Close retained completed transfers")
	}
}

func TestCUDAUploadManyLeasesBounded(t *testing.T) {
	b := fakeUpload()
	u, err := newCUDAUpload(b, 1, [3]int{2, 2, 2}, 6)
	if err != nil {
		t.Fatal(err)
	}
	c, err := New(identity(), 1, 6, 6)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 64; i++ {
		l, err := c.Acquire([]Key{key(i % 2)})
		if err != nil {
			t.Fatal(err)
		}
		if err := u.Stage(source(t), l, false); err != nil {
			t.Fatal(err)
		}
		if b.retained != 3 || len(u.events) != 3 {
			t.Fatalf("lease %d retained copies/events %d/%d, want 3/3", i, b.retained, len(u.events))
		}
		if err := l.Release(&pendingEvent{t: &transfer{}, complete: true}); err != nil {
			t.Fatal(err)
		}
	}
	if b.calls["copy"] != 192 {
		t.Fatalf("copy count %d, want 192", b.calls["copy"])
	}
	if err := u.Close(); err != nil || b.retained != 0 || len(u.events) != 0 {
		t.Fatal("final cleanup retained uploads")
	}
}
