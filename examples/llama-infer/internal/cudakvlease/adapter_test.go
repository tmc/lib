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

type nthCompletion struct {
	count int
	fail  int
}

func (c *nthCompletion) Synchronize(context.Context) error {
	c.count++
	if c.count == c.fail {
		return errors.New("injected completion failure")
	}
	return nil
}

type fakeHostBuffer struct {
	data  []float32
	freed *int
}

func (b *fakeHostBuffer) Slice() []float32 { return b.data }
func (b *fakeHostBuffer) Free() error {
	*b.freed++
	b.data = nil
	return nil
}

type fakeTierPages struct {
	*fakePages
	values       int
	layers       int
	data         map[[2]int][2][]float32
	failToHost   int
	failToDevice int
	hostFrees    int
}

func newFakeTierPages(pages, layers, values int) *fakeTierPages {
	return &fakeTierPages{
		fakePages:    newFakePages(pages),
		values:       values,
		layers:       layers,
		data:         make(map[[2]int][2][]float32),
		failToHost:   -1,
		failToDevice: -1,
	}
}

func (p *fakeTierPages) AllocHost(n int) (hostBuffer, error) {
	return &fakeHostBuffer{data: make([]float32, n), freed: &p.hostFrees}, nil
}

func (p *fakeTierPages) CopyPageToHost(layer, slot int, k, v []float32) error {
	if layer == p.failToHost {
		return errors.New("injected device-to-host failure")
	}
	pair := p.data[[2]int{layer, slot}]
	copy(k, pair[0])
	copy(v, pair[1])
	return nil
}

func (p *fakeTierPages) CopyHostToPage(layer, slot int, k, v []float32) error {
	if layer == p.failToDevice {
		return errors.New("injected host-to-device failure")
	}
	p.data[[2]int{layer, slot}] = [2][]float32{append([]float32(nil), k...), append([]float32(nil), v...)}
	return nil
}

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

func TestOffloadRestoreRoundTripAndDeviceBound(t *testing.T) {
	cfg := Config{NumLayers: 2, KVHeads: 1, HeadDim: 2, BlockSize: 2, TotalPages: 1, DType: "float32", Domain: "gpu0"}
	pages := newFakeTierPages(1, cfg.NumLayers, cfg.BlockSize*cfg.KVHeads*cfg.HeadDim)
	a, err := New(pages, cfg)
	if err != nil {
		t.Fatal(err)
	}
	first, err := a.Allocate("first", kvcache.Range{Start: 0, End: 2})
	if err != nil {
		t.Fatal(err)
	}
	id, generation := first.ID()
	use, err := a.Bind(id, generation)
	if err != nil {
		t.Fatal(err)
	}
	slot, _ := use.Slot()
	for layer := 0; layer < cfg.NumLayers; layer++ {
		pages.data[[2]int{layer, slot}] = [2][]float32{
			{float32(layer + 1), 2, 3, 4},
			{-float32(layer + 1), -2, -3, -4},
		}
	}
	want := make(map[[2]int][2][]float32)
	for key, pair := range pages.data {
		want[key] = [2][]float32{append([]float32(nil), pair[0]...), append([]float32(nil), pair[1]...)}
	}
	if err := use.Complete(context.Background(), &fakeCompletion{}); err != nil {
		t.Fatal(err)
	}
	if err := a.Offload(context.Background(), first, &fakeCompletion{}); err != nil {
		t.Fatal(err)
	}
	if _, err := a.Bind(id, generation); !errors.Is(err, ErrNotDevice) {
		t.Fatalf("Bind host block: %v, want ErrNotDevice", err)
	}
	stats := a.Stats()
	if stats.DevicePages != 0 || stats.HostBlocks != 1 || stats.DeviceBytes != 0 || stats.PinnedHostBytes != 64 {
		t.Fatalf("offloaded stats = %+v", stats)
	}
	second, err := a.Allocate("second", kvcache.Range{Start: 2, End: 4})
	if err != nil {
		t.Fatalf("Allocate with logical block offloaded: %v", err)
	}
	stats = a.Stats()
	if stats.DevicePages != 1 || stats.DeviceBytes != 64 {
		t.Fatalf("device bound stats = %+v", stats)
	}
	if err := a.Restore(context.Background(), first); err == nil {
		t.Fatal("Restore succeeded without a free device page")
	}
	if err := a.Retire(context.Background(), second, &fakeCompletion{}); err != nil {
		t.Fatal(err)
	}
	if err := a.Restore(context.Background(), first); err != nil {
		t.Fatal(err)
	}
	use, err = a.Bind(id, generation)
	if err != nil {
		t.Fatal(err)
	}
	restoredSlot, _ := use.Slot()
	for layer := 0; layer < cfg.NumLayers; layer++ {
		got := pages.data[[2]int{layer, restoredSlot}]
		original := want[[2]int{layer, slot}]
		for i := range original[0] {
			if got[0][i] != original[0][i] || got[1][i] != original[1][i] {
				t.Fatalf("layer %d value %d = %g/%g, want %g/%g", layer, i, got[0][i], got[1][i], original[0][i], original[1][i])
			}
		}
	}
	if err := use.Complete(context.Background(), &fakeCompletion{}); err != nil {
		t.Fatal(err)
	}
	if err := a.Retire(context.Background(), first, &fakeCompletion{}); err != nil {
		t.Fatal(err)
	}
	if err := a.Restore(context.Background(), first); !errors.Is(err, kvcache.ErrStale) {
		t.Fatalf("Restore retired generation: %v, want ErrStale", err)
	}
}

func TestOffloadFailureKeepsDeviceAuthoritative(t *testing.T) {
	cfg := Config{NumLayers: 2, KVHeads: 1, HeadDim: 2, BlockSize: 2, TotalPages: 1, DType: "float32", Domain: "gpu0"}
	pages := newFakeTierPages(1, cfg.NumLayers, cfg.BlockSize*cfg.KVHeads*cfg.HeadDim)
	pages.failToHost = 1
	a, _ := New(pages, cfg)
	allocation, _ := a.Allocate("request", kvcache.Range{Start: 0, End: 2})
	if err := a.Offload(context.Background(), allocation, &fakeCompletion{}); err == nil {
		t.Fatal("Offload succeeded after injected partial copy failure")
	}
	if pages.hostFrees != 1 {
		t.Fatalf("host cleanup count = %d, want 1", pages.hostFrees)
	}
	residency, err := a.Residency(allocation)
	if err != nil || residency.Tier != "device" {
		t.Fatalf("Residency = %+v, %v; want device", residency, err)
	}
	id, generation := allocation.ID()
	use, err := a.Bind(id, generation)
	if err != nil {
		t.Fatalf("Bind after failed offload: %v", err)
	}
	if err := use.Complete(context.Background(), &fakeCompletion{}); err != nil {
		t.Fatal(err)
	}
	if err := a.Retire(context.Background(), allocation, &fakeCompletion{}); err != nil {
		t.Fatal(err)
	}
}

func TestRestoreFailureKeepsHostAuthoritative(t *testing.T) {
	cfg := Config{NumLayers: 2, KVHeads: 1, HeadDim: 2, BlockSize: 2, TotalPages: 1, DType: "float32", Domain: "gpu0"}
	pages := newFakeTierPages(1, cfg.NumLayers, cfg.BlockSize*cfg.KVHeads*cfg.HeadDim)
	a, _ := New(pages, cfg)
	allocation, _ := a.Allocate("request", kvcache.Range{Start: 0, End: 2})
	if err := a.Offload(context.Background(), allocation, &fakeCompletion{}); err != nil {
		t.Fatal(err)
	}
	pages.failToDevice = 1
	if err := a.Restore(context.Background(), allocation); err == nil {
		t.Fatal("Restore succeeded after injected partial copy failure")
	}
	residency, err := a.Residency(allocation)
	if err != nil || residency.Tier != "pinned-host" {
		t.Fatalf("Residency = %+v, %v; want pinned-host", residency, err)
	}
	if stats := a.Stats(); stats.DevicePages != 0 || stats.HostBlocks != 1 {
		t.Fatalf("failed restore stats = %+v", stats)
	}
	pages.failToDevice = -1
	if err := a.Restore(context.Background(), allocation); err != nil {
		t.Fatalf("retry Restore: %v", err)
	}
	if err := a.Retire(context.Background(), allocation, &fakeCompletion{}); err != nil {
		t.Fatal(err)
	}
}

func TestOffloadRequiresSoleLeaseAndCompletion(t *testing.T) {
	cfg := Config{NumLayers: 1, KVHeads: 1, HeadDim: 2, BlockSize: 2, TotalPages: 1, DType: "float32", Domain: "gpu0"}
	pages := newFakeTierPages(1, cfg.NumLayers, cfg.BlockSize*cfg.KVHeads*cfg.HeadDim)
	a, _ := New(pages, cfg)
	allocation, _ := a.Allocate("request", kvcache.Range{Start: 0, End: 2})
	id, generation := allocation.ID()
	use, _ := a.Bind(id, generation)
	if err := a.Offload(context.Background(), allocation, &fakeCompletion{}); !errors.Is(err, kvcache.ErrBusy) {
		t.Fatalf("Offload with consumer: %v, want ErrBusy", err)
	}
	if err := use.Complete(context.Background(), &fakeCompletion{}); err != nil {
		t.Fatal(err)
	}
	want := errors.New("completion failed")
	if err := a.Offload(context.Background(), allocation, &fakeCompletion{err: want}); !errors.Is(err, want) {
		t.Fatalf("Offload completion failure: %v, want injected error", err)
	}
	if stats := a.Stats(); stats.DevicePages != 1 || stats.HostBlocks != 0 {
		t.Fatalf("completion failure stats = %+v", stats)
	}
	if err := a.Offload(context.Background(), allocation, &fakeCompletion{}); err != nil {
		t.Fatalf("retry Offload: %v", err)
	}
	if err := a.Retire(context.Background(), allocation, &fakeCompletion{}); err != nil {
		t.Fatal(err)
	}
	if pages.hostFrees != 1 || a.NumFreePages() != 1 {
		t.Fatalf("host retirement: frees=%d free_pages=%d, want 1/1", pages.hostFrees, a.NumFreePages())
	}
}

func TestReserveVictimUsesCompletedOrderAndExcludesConsumers(t *testing.T) {
	cfg := Config{NumLayers: 1, KVHeads: 1, HeadDim: 2, BlockSize: 16, TotalPages: 3, DType: "float32", Domain: "gpu0"}
	a, err := New(newFakeTierPages(3, cfg.NumLayers, cfg.BlockSize*cfg.KVHeads*cfg.HeadDim), cfg)
	if err != nil {
		t.Fatal(err)
	}
	allocations := []Allocation{
		allocate(t, a, "first"),
		allocate(t, a, "second"),
		allocate(t, a, "third"),
	}
	// Completion order, rather than bind or allocation order, is authoritative.
	for _, i := range []int{1, 0, 2} {
		id, generation := allocations[i].ID()
		use, err := a.Bind(id, generation)
		if err != nil {
			t.Fatal(err)
		}
		if err := use.Complete(context.Background(), &fakeCompletion{}); err != nil {
			t.Fatal(err)
		}
	}
	reservation, err := a.ReserveVictim()
	if err != nil {
		t.Fatal(err)
	}
	id, generation := allocations[1].ID()
	if _, err := a.Bind(id, generation); !errors.Is(err, kvcache.ErrBusy) {
		t.Fatalf("Bind reserved victim: %v, want ErrBusy", err)
	}
	if err := reservation.Abort(); err != nil {
		t.Fatal(err)
	}
	use, err := a.Bind(id, generation)
	if err != nil {
		t.Fatalf("Bind after abort: %v", err)
	}
	if err := use.Complete(context.Background(), &fakeCompletion{}); err != nil {
		t.Fatal(err)
	}

	reservation, err = a.ReserveVictim(allocations[0], allocations[1])
	if err != nil {
		t.Fatal(err)
	}
	if err := reservation.Offload(context.Background(), &fakeCompletion{}); err != nil {
		t.Fatal(err)
	}
	residency, err := a.Residency(allocations[2])
	if err != nil || residency.Tier != "pinned-host" {
		t.Fatalf("excluded victim residency = %+v, %v; want pinned-host", residency, err)
	}
	if err := reservation.Abort(); !errors.Is(err, ErrStale) {
		t.Fatalf("Abort completed reservation: %v, want ErrStale", err)
	}
}

func TestReserveVictimSkipsLiveUseAndCancellationPreservesPage(t *testing.T) {
	cfg := Config{NumLayers: 1, KVHeads: 1, HeadDim: 2, BlockSize: 16, TotalPages: 2, DType: "float32", Domain: "gpu0"}
	a, _ := New(newFakeTierPages(2, cfg.NumLayers, cfg.BlockSize*cfg.KVHeads*cfg.HeadDim), cfg)
	first := allocate(t, a, "first")
	second := allocate(t, a, "second")
	id, generation := first.ID()
	use, err := a.Bind(id, generation)
	if err != nil {
		t.Fatal(err)
	}
	reservation, err := a.ReserveVictim()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := reservation.Offload(ctx, &fakeCompletion{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("Offload canceled reservation: %v, want context.Canceled", err)
	}
	residency, err := a.Residency(second)
	if err != nil || residency.Tier != "device" {
		t.Fatalf("canceled victim residency = %+v, %v; want device", residency, err)
	}
	if err := use.Complete(context.Background(), &fakeCompletion{}); err != nil {
		t.Fatal(err)
	}
}

func TestReserveVictimsIsAtomicAndRollsBackPartialOffload(t *testing.T) {
	cfg := Config{NumLayers: 1, KVHeads: 1, HeadDim: 2, BlockSize: 16, TotalPages: 2, DType: "float32", Domain: "gpu0"}
	a, _ := New(newFakeTierPages(2, cfg.NumLayers, cfg.BlockSize*cfg.KVHeads*cfg.HeadDim), cfg)
	first := allocate(t, a, "first")
	second := allocate(t, a, "second")
	if _, err := a.ReserveVictims(3); !errors.Is(err, kvcache.ErrExhausted) {
		t.Fatalf("ReserveVictims(3): %v, want ErrExhausted", err)
	}
	for _, allocation := range []Allocation{first, second} {
		id, generation := allocation.ID()
		use, err := a.Bind(id, generation)
		if err != nil {
			t.Fatalf("Bind after failed set reservation: %v", err)
		}
		if err := use.Complete(context.Background(), &fakeCompletion{}); err != nil {
			t.Fatal(err)
		}
	}
	set, err := a.ReserveVictims(2)
	if err != nil {
		t.Fatal(err)
	}
	if err := set.Offload(context.Background(), &nthCompletion{fail: 2}); err == nil {
		t.Fatal("multi-page offload succeeded after second completion failed")
	}
	for _, allocation := range []Allocation{first, second} {
		residency, err := a.Residency(allocation)
		if err != nil || residency.Tier != "device" {
			t.Fatalf("rollback residency = %+v, %v; want device", residency, err)
		}
	}
	if stats := a.Stats(); stats.DevicePages != 2 || stats.HostBlocks != 0 {
		t.Fatalf("rollback stats = %+v, want two device pages", stats)
	}
}
