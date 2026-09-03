package cudakvlease

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/tmc/lib/lm/kvcache"
)

type fakePages struct {
	mu     sync.Mutex
	free   []int
	tables map[int64][]int
	closed bool
}

func newFakePages(n int) *fakePages {
	p := &fakePages{tables: make(map[int64][]int)}
	for i := range n {
		p.free = append(p.free, i)
	}
	return p
}
func (p *fakePages) AllocSequence(id int64, tokens int) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed || tokens <= 0 || len(p.free) == 0 {
		return errors.New("no page")
	}
	n := len(p.free) - 1
	p.tables[id] = []int{p.free[n]}
	p.free = p.free[:n]
	return nil
}
func (p *fakePages) FreeSequence(id int64) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.free = append(p.free, p.tables[id]...)
	delete(p.tables, id)
}
func (p *fakePages) BlockTable(id int64) []int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]int(nil), p.tables[id]...)
}
func (p *fakePages) NumFreeBlocks() int { p.mu.Lock(); defer p.mu.Unlock(); return len(p.free) }
func (p *fakePages) Close() error       { p.mu.Lock(); defer p.mu.Unlock(); p.closed = true; return nil }

type fakeCompletion struct{ err error }

func (f *fakeCompletion) Synchronize(context.Context) error { return f.err }

type contextCompletion struct{}

func (contextCompletion) Synchronize(ctx context.Context) error { return ctx.Err() }

func testConfig(n int) Config {
	return Config{NumLayers: 2, KVHeads: 4, HeadDim: 64, BlockSize: 16, TotalPages: n, DType: "float32", Domain: "gpu0"}
}
func allocate(t *testing.T, a *Adapter, owner string) Allocation {
	t.Helper()
	allocation, err := a.Allocate(owner, kvcache.Range{Start: 0, End: 16})
	if err != nil {
		t.Fatalf("Allocate: %v", err)
	}
	return allocation
}

func TestForcedSlotReuseRejectsABA(t *testing.T) {
	a, err := New(newFakePages(1), testConfig(1))
	if err != nil {
		t.Fatal(err)
	}
	first := allocate(t, a, "first")
	firstID, firstGeneration := first.ID()
	use, err := a.Bind(firstID, firstGeneration)
	if err != nil {
		t.Fatal(err)
	}
	firstSlot, err := use.Slot()
	if err != nil {
		t.Fatal(err)
	}
	if err := use.Complete(context.Background(), &fakeCompletion{}); err != nil {
		t.Fatal(err)
	}
	if _, err := use.Slot(); !errors.Is(err, ErrStale) {
		t.Fatalf("Slot after Complete: %v, want ErrStale", err)
	}
	if err := a.Retire(context.Background(), first, &fakeCompletion{}); err != nil {
		t.Fatal(err)
	}
	second := allocate(t, a, "second")
	secondID, secondGeneration := second.ID()
	use, err = a.Bind(secondID, secondGeneration)
	if err != nil {
		t.Fatal(err)
	}
	secondSlot, err := use.Slot()
	if err != nil {
		t.Fatal(err)
	}
	if secondSlot != firstSlot || secondID != firstID || secondGeneration == firstGeneration {
		t.Fatalf("reuse = slot %d id %d generation %d; want slot %d id %d with new generation", secondSlot, secondID, secondGeneration, firstSlot, firstID)
	}
	if _, err := a.Bind(firstID, firstGeneration); !errors.Is(err, ErrStale) {
		t.Fatalf("Bind retired generation: %v, want ErrStale", err)
	}
	if err := use.Complete(context.Background(), &fakeCompletion{}); err != nil {
		t.Fatal(err)
	}
	if err := a.Retire(context.Background(), second, &fakeCompletion{}); err != nil {
		t.Fatal(err)
	}
}

func TestDelayedConsumerPreventsReuse(t *testing.T) {
	a, _ := New(newFakePages(1), testConfig(1))
	allocation := allocate(t, a, "request")
	id, generation := allocation.ID()
	use, err := a.Bind(id, generation)
	if err != nil {
		t.Fatal(err)
	}
	if err := a.Retire(context.Background(), allocation, &fakeCompletion{}); !errors.Is(err, kvcache.ErrBusy) {
		t.Fatalf("Retire with consumer: %v, want ErrBusy", err)
	}
	if _, err := a.Allocate("other", kvcache.Range{Start: 0, End: 1}); err == nil {
		t.Fatal("Allocate succeeded while consumer held only page")
	}
	if err := use.Complete(context.Background(), &fakeCompletion{}); err != nil {
		t.Fatal(err)
	}
	if err := a.Retire(context.Background(), allocation, &fakeCompletion{}); err != nil {
		t.Fatal(err)
	}
}

func TestCompletionFailurePinsPage(t *testing.T) {
	a, _ := New(newFakePages(1), testConfig(1))
	allocation := allocate(t, a, "request")
	id, generation := allocation.ID()
	use, _ := a.Bind(id, generation)
	want := errors.New("device lost")
	if err := use.Complete(context.Background(), &fakeCompletion{err: want}); !errors.Is(err, want) {
		t.Fatalf("Complete: %v, want device error", err)
	}
	if err := a.Retire(context.Background(), allocation, &fakeCompletion{}); !errors.Is(err, kvcache.ErrBusy) {
		t.Fatalf("Retire after event failure: %v, want ErrBusy", err)
	}
	if err := use.Complete(context.Background(), &fakeCompletion{}); err != nil {
		t.Fatal(err)
	}
	if err := a.Retire(context.Background(), allocation, &fakeCompletion{}); err != nil {
		t.Fatal(err)
	}
}

func TestCancellationPinsPage(t *testing.T) {
	a, _ := New(newFakePages(1), testConfig(1))
	allocation := allocate(t, a, "request")
	id, generation := allocation.ID()
	use, _ := a.Bind(id, generation)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := use.Complete(ctx, contextCompletion{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("Complete: %v, want context.Canceled", err)
	}
	if err := a.Retire(context.Background(), allocation, &fakeCompletion{}); !errors.Is(err, kvcache.ErrBusy) {
		t.Fatalf("Retire after cancellation: %v, want ErrBusy", err)
	}
	if err := use.Complete(context.Background(), &fakeCompletion{}); err != nil {
		t.Fatal(err)
	}
	if err := a.Retire(context.Background(), allocation, &fakeCompletion{}); err != nil {
		t.Fatal(err)
	}
}

func TestRetirementFailurePinsPage(t *testing.T) {
	a, _ := New(newFakePages(1), testConfig(1))
	allocation := allocate(t, a, "request")
	want := errors.New("event query failed")
	if err := a.Retire(context.Background(), allocation, &fakeCompletion{err: want}); !errors.Is(err, want) {
		t.Fatalf("Retire: %v, want event error", err)
	}
	if got := a.NumFreePages(); got != 0 {
		t.Fatalf("NumFreePages = %d, want 0", got)
	}
	if err := a.Retire(context.Background(), allocation, &fakeCompletion{}); err != nil {
		t.Fatal(err)
	}
}

func TestExhaustionAndClose(t *testing.T) {
	a, _ := New(newFakePages(1), testConfig(1))
	allocation := allocate(t, a, "request")
	if _, err := a.Allocate("other", kvcache.Range{Start: 0, End: 1}); err == nil {
		t.Fatal("second Allocate succeeded")
	}
	if err := a.Close(); !errors.Is(err, kvcache.ErrBusy) {
		t.Fatalf("Close live adapter: %v, want ErrBusy", err)
	}
	if err := a.Retire(context.Background(), allocation, &fakeCompletion{}); err != nil {
		t.Fatal(err)
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := a.Allocate("late", kvcache.Range{Start: 0, End: 1}); !errors.Is(err, ErrClosed) {
		t.Fatalf("Allocate after Close: %v, want ErrClosed", err)
	}
}

func TestConcurrentBindComplete(t *testing.T) {
	a, _ := New(newFakePages(1), testConfig(1))
	allocation := allocate(t, a, "request")
	id, generation := allocation.ID()
	const consumers = 32
	uses := make([]*Use, consumers)
	for i := range uses {
		var err error
		uses[i], err = a.Bind(id, generation)
		if err != nil {
			t.Fatal(err)
		}
	}
	var wg sync.WaitGroup
	for _, use := range uses {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := use.Complete(context.Background(), &fakeCompletion{}); err != nil {
				t.Errorf("Complete: %v", err)
			}
		}()
	}
	wg.Wait()
	if err := a.Retire(context.Background(), allocation, &fakeCompletion{}); err != nil {
		t.Fatal(err)
	}
}
