package cudabpf_test

import (
	"context"
	"errors"
	"math"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudabpf"
)

func TestRegistrationAndCallbackProofRequired(t *testing.T) {
	for _, name := range []string{"registration", "callback"} {
		t.Run(name, func(t *testing.T) {
			b := patchedBTF()
			if name == "registration" {
				b.tableErr = errors.New("permission denied")
			} else {
				b.memberErr = errors.New("wrong callback prototype")
			}
			l := newFakeLoader(b)
			br := cudabpf.NewBridge(l, nil, cudabpf.Config{})
			if _, err := br.Attach(context.Background(), memObject(), memSpec()); err == nil {
				t.Fatal("unverified table loaded")
			}
			if l.loads != 0 {
				t.Fatal("load attempted before proof")
			}
		})
	}
}

func TestAllLoadedObjectsReleased(t *testing.T) {
	l := newFakeLoader(patchedBTF())
	br := cudabpf.NewBridge(l, nil, cudabpf.Config{})
	if _, err := br.Attach(context.Background(), memObject(), memSpec()); err != nil {
		t.Fatal(err)
	}
	if _, err := br.Attach(context.Background(), schedObject(), schedSpec()); err != nil {
		t.Fatal(err)
	}
	if err := br.Detach(cudabpf.MemOpsTable); err != nil {
		t.Fatal(err)
	}
	if err := br.Close(); err != nil {
		t.Fatal(err)
	}
	if l.loads != 2 || l.closes != 2 || l.detaches != 2 {
		t.Fatalf("loads=%d closes=%d detaches=%d", l.loads, l.closes, l.detaches)
	}
}

type blockingLoader struct {
	*fakeLoader
	entered, proceed chan struct{}
}

func (l *blockingLoader) Load(ctx context.Context, s cudabpf.ObjectSpec) (cudabpf.Object, error) {
	close(l.entered)
	<-l.proceed
	return l.fakeLoader.Load(ctx, s)
}

func TestCloseWaitsForPendingAttach(t *testing.T) {
	l := &blockingLoader{newFakeLoader(patchedBTF()), make(chan struct{}), make(chan struct{})}
	br := cudabpf.NewBridge(l, nil, cudabpf.Config{})
	attached := make(chan error, 1)
	closed := make(chan error, 1)
	go func() { _, err := br.Attach(context.Background(), memObject(), memSpec()); attached <- err }()
	<-l.entered
	go func() { closed <- br.Close() }()
	close(l.proceed)
	if err := <-attached; err != nil {
		t.Fatal(err)
	}
	if err := <-closed; err != nil {
		t.Fatal(err)
	}
	if l.attaches != l.detaches || l.loads != l.closes {
		t.Fatal("pending attachment leaked across Close")
	}
	if _, err := br.Attach(context.Background(), memObject(), memSpec()); !errors.Is(err, cudabpf.ErrDisabled) {
		t.Fatalf("closed bridge: %v", err)
	}
}

func TestBoundedOutcomes(t *testing.T) {
	br := cudabpf.NewBridge(nil, nil, cudabpf.Config{MaxOutcomes: 3})
	for i := 0; i < 100; i++ {
		br.OnBlockAccess(nil)
	}
	if got := len(br.Outcomes()); got != 3 {
		t.Fatalf("retained %d outcomes", got)
	}
}

func TestBudgetCannotRefillOrOverflow(t *testing.T) {
	b := cudabpf.NewBudget(math.MaxInt)
	if err := b.Step(math.MaxInt); err != nil {
		t.Fatal(err)
	}
	if err := b.Step(1); !errors.Is(err, cudabpf.ErrBudgetExceeded) {
		t.Fatalf("overflow admitted: %v", err)
	}
	if err := b.Step(0); !errors.Is(err, cudabpf.ErrBudgetExceeded) {
		t.Fatal("exhaustion not sticky")
	}
	b.Reset()
	if err := b.Step(-1); err == nil {
		t.Fatal("negative charge admitted")
	}
	if err := b.Step(1); !errors.Is(err, cudabpf.ErrBudgetExceeded) {
		t.Fatal("invalid charge refilled budget")
	}
}

func TestPrefetchSubmitCloseRace(t *testing.T) {
	for i := 0; i < 50; i++ {
		w := cudabpf.NewPrefetchWorker(&fakeKfuncs{}, 2, nil)
		w.Start()
		w.Start()
		var wg sync.WaitGroup
		wg.Add(2)
		go func() {
			defer wg.Done()
			for id := uint64(1); id < 20; id++ {
				w.Submit(cudabpf.PrefetchRequest{ID: id, Region: cudabpf.Region{Outer: 1}, Decision: &cudabpf.PrefetchDecision{}})
			}
		}()
		go func() { defer wg.Done(); w.Close() }()
		wg.Wait()
		w.Start()
		w.Close()
		if len(w.Results()) > 2 {
			t.Fatal("unbounded completion history")
		}
		if w.Cancel(9999) {
			t.Fatal("unknown ID cancelled")
		}
	}
}

func TestPrefetchDuplicateAndInvalidRequest(t *testing.T) {
	w := cudabpf.NewPrefetchWorker(nil, 2, nil)
	defer w.Close()
	r := cudabpf.PrefetchRequest{ID: 1, Region: cudabpf.Region{Outer: 1}, Decision: &cudabpf.PrefetchDecision{}}
	if !w.Submit(r) {
		t.Fatal("valid request refused")
	}
	if w.Submit(r) {
		t.Fatal("duplicate pending ID admitted")
	}
	r.ID = 2
	r.Decision = nil
	if w.Submit(r) {
		t.Fatal("nil decision admitted")
	}
}

func TestDiscoveryReportDoesNotExposeCachedState(t *testing.T) {
	br := cudabpf.NewBridge(newFakeLoader(patchedBTF()), nil, cudabpf.Config{})
	r, err := br.Discover(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	r.Hooks[0].Reason = "changed"
	r.Capabilities.StructOpsTables[0] = "changed"
	cached := br.Report()
	if cached.Hooks[0].Reason == "changed" || cached.Capabilities.StructOpsTables[0] == "changed" {
		t.Fatal("caller mutated cache")
	}
	cached.Hooks[0].Reason = "second change"
	if br.Report().Hooks[0].Reason == "second change" {
		t.Fatal("Report exposes cache")
	}
}

func TestWrongReturnedTypeIdentityRejected(t *testing.T) {
	b := patchedBTF()
	typ := b.types["uvm_bpf_prefetch_decision_t"]
	typ.Name = "unrelated_type"
	b.types["uvm_bpf_prefetch_decision_t"] = typ
	if _, err := cudabpf.CheckLayouts(b); err == nil {
		t.Fatal("wrong type identity accepted")
	}
	k := b.kfuncs["bpf_gpu_request_reorder"]
	k.Name = "unrelated_function"
	b.kfuncs["bpf_gpu_request_reorder"] = k
	b.types["uvm_bpf_prefetch_decision_t"] = cudabpf.InventoryLayouts()[0]
	r, err := cudabpf.Discover(context.Background(), newFakeLoader(b))
	if err != nil {
		t.Fatal(err)
	}
	status, _ := r.Kfunc("bpf_gpu_request_reorder")
	if status.Available {
		t.Fatal("wrong kfunc identity accepted")
	}
}

type blockedPrefetch struct {
	*fakeKfuncs
	entered, proceed chan struct{}
}

func (k *blockedPrefetch) SetPrefetchRegion(d *cudabpf.PrefetchDecision, first, outer uint64) error {
	close(k.entered)
	<-k.proceed
	return k.fakeKfuncs.SetPrefetchRegion(d, first, outer)
}

func TestPrefetchClaimCannotBeCancelledOrDuplicated(t *testing.T) {
	k := &blockedPrefetch{&fakeKfuncs{}, make(chan struct{}), make(chan struct{})}
	w := cudabpf.NewPrefetchWorker(k, 2, nil)
	w.Start()
	r := cudabpf.PrefetchRequest{ID: 1, Region: cudabpf.Region{Outer: 1}, Decision: &cudabpf.PrefetchDecision{}}
	if !w.Submit(r) {
		t.Fatal("submit")
	}
	<-k.entered
	if w.Cancel(1) {
		t.Error("in-flight cancellation reported success")
	}
	if w.Submit(r) {
		t.Error("in-flight request ID reused")
	}
	close(k.proceed)
	w.Close()
}

type failingCloseObject struct {
	cudabpf.Object
	failures int
}

func (o *failingCloseObject) Close() error {
	if o.failures > 0 {
		o.failures--
		return errors.New("temporary close failure")
	}
	return o.Object.Close()
}

type closeFailureLoader struct{ *fakeLoader }

func (l closeFailureLoader) Load(ctx context.Context, s cudabpf.ObjectSpec) (cudabpf.Object, error) {
	o, err := l.fakeLoader.Load(ctx, s)
	if err != nil {
		return nil, err
	}
	return &failingCloseObject{Object: o, failures: 1}, nil
}

func TestFailedCleanupRetainsObjectOwnership(t *testing.T) {
	l := closeFailureLoader{newFakeLoader(patchedBTF())}
	br := cudabpf.NewBridge(l, nil, cudabpf.Config{})
	if _, err := br.Attach(context.Background(), memObject(), memSpec()); err != nil {
		t.Fatal(err)
	}
	if err := br.Detach(cudabpf.MemOpsTable); err == nil {
		t.Fatal("close failure hidden")
	}
	if _, err := br.Attach(context.Background(), memObject(), memSpec()); err == nil {
		t.Fatal("pending object overwritten")
	}
	if err := br.Close(); err != nil {
		t.Fatal(err)
	}
	if l.loads != l.closes {
		t.Fatal("object ownership lost")
	}
}

func TestReattachRejectsWrongGeneration(t *testing.T) {
	l := newFakeLoader(patchedBTF())
	br := cudabpf.NewBridge(l, nil, cudabpf.Config{})
	if _, err := br.Reattach(context.Background(), memObject(), memSpec(), 999); err == nil {
		t.Fatal("wrong generation accepted")
	}
	if l.attaches != l.detaches || l.loads != l.closes {
		t.Fatal("wrong-generation cleanup leaked")
	}
}

func TestCloseWithoutStartReportsQueuedCancellation(t *testing.T) {
	w := cudabpf.NewPrefetchWorker(nil, 2, nil)
	for id := uint64(1); id <= 2; id++ {
		if !w.Submit(cudabpf.PrefetchRequest{ID: id, Region: cudabpf.Region{Outer: 1}, Decision: &cudabpf.PrefetchDecision{}}) {
			t.Fatal("submit")
		}
	}
	w.Close()
	if len(w.Results()) != 2 {
		t.Fatal("queued requests silently discarded")
	}
	for _, r := range w.Results() {
		if !r.Cancelled || r.Issued {
			t.Fatal("closed queue result", r)
		}
	}
}
