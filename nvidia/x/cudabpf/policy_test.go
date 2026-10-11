package cudabpf_test

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudabpf"
)

func attached(t *testing.T, k cudabpf.Kfuncs, cfg cudabpf.Config) (*cudabpf.Bridge, *fakeLoader) {
	t.Helper()
	l := newFakeLoader(patchedBTF())
	br := cudabpf.NewBridge(l, k, cfg)
	if _, err := br.Attach(context.Background(), memObject(), memSpec()); err != nil {
		t.Fatalf("Attach: %v", err)
	}
	return br, l
}

// TestProgramLoadAndDetachLifetime walks the whole lifetime: load, attach,
// detach, detach again, close. Detach must be idempotent and must always
// succeed, because it is the mechanism the disable switch and the error paths
// both rely on.
func TestProgramLoadAndDetachLifetime(t *testing.T) {
	br, l := attached(t, &fakeKfuncs{}, cudabpf.Config{})
	if l.loads != 1 || l.attaches != 1 {
		t.Fatalf("loads=%d attaches=%d, want 1 and 1", l.loads, l.attaches)
	}
	if !l.registered[cudabpf.MemOpsTable] {
		t.Fatal("table not registered after Attach")
	}
	if err := br.Detach(cudabpf.MemOpsTable); err != nil {
		t.Fatalf("Detach: %v", err)
	}
	if l.registered[cudabpf.MemOpsTable] {
		t.Fatal("table still registered after Detach")
	}
	if err := br.Detach(cudabpf.MemOpsTable); err != nil {
		t.Fatalf("second Detach: %v", err)
	}
	if err := br.Detach("nv_gpu_sched_ops"); err != nil {
		t.Fatalf("detaching what was never attached: %v", err)
	}
	if err := br.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

// TestStructOpsAttachIdempotence covers both senses. Attaching twice from one
// bridge registers once and returns the same link. Two bridges racing for the
// same table is a different case: the kernel arbitrates it, and the loser
// must see the error rather than believe it attached.
func TestStructOpsAttachIdempotence(t *testing.T) {
	l := newFakeLoader(patchedBTF())
	br := cudabpf.NewBridge(l, &fakeKfuncs{}, cudabpf.Config{})
	ctx := context.Background()

	first, err := br.Attach(ctx, memObject(), memSpec())
	if err != nil {
		t.Fatalf("first Attach: %v", err)
	}
	second, err := br.Attach(ctx, memObject(), memSpec())
	if err != nil {
		t.Fatalf("second Attach: %v", err)
	}
	if first != second {
		t.Error("a repeated Attach returned a different link")
	}
	if l.attaches != 1 {
		t.Errorf("registered %d times, want 1", l.attaches)
	}

	other := cudabpf.NewBridge(l, &fakeKfuncs{}, cudabpf.Config{})
	if _, err := other.Attach(ctx, memObject(), memSpec()); err == nil {
		t.Fatal("a second process attached the same table")
	}
}

// TestReattachAfterGenerationChange covers a driver that re-registered its
// tables underneath a live link. The link is stale, the policy is not.
func TestReattachAfterGenerationChange(t *testing.T) {
	l := newFakeLoader(patchedBTF())
	br := cudabpf.NewBridge(l, &fakeKfuncs{}, cudabpf.Config{})
	ctx := context.Background()

	link, err := br.Attach(ctx, memObject(), memSpec())
	if err != nil {
		t.Fatalf("Attach: %v", err)
	}
	if got, err := br.Reattach(ctx, memObject(), memSpec(), link.Generation()); err != nil || got != link {
		t.Fatalf("Reattach at the same generation returned (%v, %v), want the existing link", got, err)
	}
	l.generation = 2
	fresh, err := br.Reattach(ctx, memObject(), memSpec(), 2)
	if err != nil {
		t.Fatalf("Reattach after a generation change: %v", err)
	}
	if fresh.Generation() != 2 {
		t.Errorf("new link generation = %d, want 2", fresh.Generation())
	}
	if l.detaches != 1 {
		t.Errorf("detached %d times during reattach, want 1", l.detaches)
	}
}

// TestReorderExpressedAsDestinationAndPosition checks the shape of the
// interface, not just that a call was made. Head and tail are values written
// into the request and actuated by one kfunc; they are not separate calls,
// and there is no way to express a rank between the two ends.
func TestReorderExpressedAsDestinationAndPosition(t *testing.T) {
	k := &fakeKfuncs{}
	p := &staticPolicy{reorder: reorderOpinion{cudabpf.Reorder{
		Destination: cudabpf.DestinationUnused, Position: cudabpf.PositionHead}, true}}
	br, _ := attached(t, k, cudabpf.Config{})
	br.SetMemoryPolicy(p)

	d := &cudabpf.PMMDecision{Observed: cudabpf.PMMSnapshot{RootID: 7}}
	out := br.OnBlockAccess(d)
	if !out.Submitted {
		t.Fatalf("not submitted: %+v", out)
	}
	if len(k.reorders) != 1 {
		t.Fatalf("made %d kfunc calls, want 1", len(k.reorders))
	}
	if k.reorders[0].dst != cudabpf.DestinationUnused || k.reorders[0].pos != cudabpf.PositionHead {
		t.Errorf("submitted %v/%v", k.reorders[0].dst, k.reorders[0].pos)
	}
	if d.Request.Destination != uint64(cudabpf.DestinationUnused) || d.Request.Position != uint64(cudabpf.PositionHead) {
		t.Errorf("request fields = %d/%d, want %d/%d",
			d.Request.Destination, d.Request.Position,
			cudabpf.DestinationUnused, cudabpf.PositionHead)
	}

	// Only the two ends are addressable. An arbitrary rank is not a
	// value this interface has, and a policy that invents one is
	// refused rather than rounded to an end.
	for _, bad := range []cudabpf.Reorder{
		{Destination: cudabpf.DestinationUsed, Position: cudabpf.Position(3)},
		{Destination: cudabpf.Destination(9), Position: cudabpf.PositionTail},
		{},
	} {
		if err := bad.Valid(); err == nil {
			t.Errorf("%v was accepted", bad)
		}
	}
}

// TestEvictionListReorderRespectsDriverFallback covers the boundary that
// matters most: the driver keeps final authority. A submitted request the
// driver does not record leaves the driver's own ordering in place, and the
// policy layer reports that rather than reporting success.
func TestEvictionListReorderRespectsDriverFallback(t *testing.T) {
	k := &fakeKfuncs{ignore: true} // the driver declines to record anything
	br, _ := attached(t, k, cudabpf.Config{})
	br.SetMemoryPolicy(&staticPolicy{reorder: reorderOpinion{
		cudabpf.Reorder{Destination: cudabpf.DestinationUsed, Position: cudabpf.PositionTail}, true}})

	d := &cudabpf.PMMDecision{}
	out := br.OnBlockAccess(d)
	if !out.Submitted {
		t.Fatal("the request was never submitted")
	}
	if out.DriverAttempted {
		t.Error("DriverAttempted is true though the driver recorded nothing")
	}
	if d.Request.Attempted != 0 {
		t.Error("the policy layer wrote the driver's own attempted flag")
	}

	// And with no policy at all, the driver's behavior is what runs.
	plain, _ := attached(t, &fakeKfuncs{}, cudabpf.Config{})
	out = plain.OnBlockAccess(&cudabpf.PMMDecision{})
	if out.HadOpinion || out.Submitted {
		t.Errorf("a bridge with no policy submitted something: %+v", out)
	}
}

// TestDriverConflictFlagObservedOnContendedReorder asserts on the driver's
// own record of a collision, not on a Go-side flag. attempted and conflict
// are written by the driver into the request; they are the only evidence the
// policy path has, and a Go boolean set next to the call would prove nothing.
func TestDriverConflictFlagObservedOnContendedReorder(t *testing.T) {
	k := &fakeKfuncs{conflict: true}
	br, _ := attached(t, k, cudabpf.Config{})
	br.SetMemoryPolicy(&staticPolicy{reorder: reorderOpinion{
		cudabpf.Reorder{Destination: cudabpf.DestinationUnused, Position: cudabpf.PositionTail}, true}})

	d := &cudabpf.PMMDecision{}
	out := br.OnBlockAccess(d)
	if !out.DriverConflict {
		t.Fatal("a contended reorder did not report the driver's conflict flag")
	}
	if d.Request.Conflict == 0 {
		t.Fatal("the driver's conflict flag was not read from the context")
	}
	if !out.Fallback {
		t.Error("a conflicting request did not leave the driver's ordering as the outcome")
	}
	if !strings.Contains(out.Reason, "conflict") {
		t.Errorf("reason = %q", out.Reason)
	}
}

// TestHandlerErrorFallsBackToDriverDefault covers every way a handler can
// fail: it returns an error, it returns something invalid, or the kfunc call
// itself fails. All three must land on the driver's default with a recorded
// reason, and none may leave a placement undefined.
func TestHandlerErrorFallsBackToDriverDefault(t *testing.T) {
	tests := []struct {
		name   string
		policy *staticPolicy
		kfuncs *fakeKfuncs
		reason string
	}{
		{
			name:   "handler error",
			policy: &staticPolicy{err: errors.New("handler blew up")},
			kfuncs: &fakeKfuncs{},
			reason: "handler blew up",
		},
		{
			name: "invalid decision",
			policy: &staticPolicy{reorder: reorderOpinion{
				cudabpf.Reorder{Destination: cudabpf.Destination(42), Position: cudabpf.PositionHead}, true}},
			kfuncs: &fakeKfuncs{},
			reason: "invalid reorder destination",
		},
		{
			name: "kfunc failed",
			policy: &staticPolicy{reorder: reorderOpinion{
				cudabpf.Reorder{Destination: cudabpf.DestinationUsed, Position: cudabpf.PositionHead}, true}},
			kfuncs: &fakeKfuncs{reorderErr: errors.New("EPERM")},
			reason: "EPERM",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			br, _ := attached(t, tt.kfuncs, cudabpf.Config{})
			br.SetMemoryPolicy(tt.policy)
			d := &cudabpf.PMMDecision{}
			out := br.OnBlockAccess(d)
			if !out.Fallback {
				t.Fatalf("no fallback: %+v", out)
			}
			if !strings.Contains(out.Reason, tt.reason) {
				t.Errorf("reason = %q, want it to contain %q", out.Reason, tt.reason)
			}
			if out.Submitted {
				t.Error("reported as submitted after a failure")
			}
			if d.Request.Destination != 0 || d.Request.Position != 0 {
				t.Errorf("a failed handler left a placement behind: %+v", d.Request)
			}
		})
	}
}

// TestHandlerBudgetExceededDisablesHook covers the timeout control. A handler
// that cannot finish inside a fault path must stop being called, not be
// called again and fail again; and the hook it disables must be the one that
// blew the budget.
func TestHandlerBudgetExceededDisablesHook(t *testing.T) {
	k := &fakeKfuncs{}
	br, _ := attached(t, k, cudabpf.Config{Budget: 4})
	br.SetMemoryPolicy(&staticPolicy{steps: 10, reorder: reorderOpinion{
		cudabpf.Reorder{Destination: cudabpf.DestinationUsed, Position: cudabpf.PositionHead}, true}})

	out := br.OnBlockAccess(&cudabpf.PMMDecision{})
	if !out.Fallback || !strings.Contains(out.Reason, "budget") {
		t.Fatalf("first invocation: %+v", out)
	}
	reason, disabled := br.HookDisabled("gpu_block_access")
	if !disabled {
		t.Fatal("the hook was not disabled after blowing its budget")
	}
	if !strings.Contains(reason, "budget") {
		t.Errorf("disable reason = %q", reason)
	}

	out = br.OnBlockAccess(&cudabpf.PMMDecision{})
	if out.Consulted {
		t.Error("the disabled hook was consulted again")
	}
	if n, _, _ := k.counts(); n != 0 {
		t.Errorf("%d kfunc calls from a disabled hook", n)
	}

	// The sibling hook is untouched: disabling is per-hook.
	if _, off := br.HookDisabled("gpu_block_activate"); off {
		t.Error("blowing one hook's budget disabled another")
	}

	// And a budget refills per invocation, so a policy that fits keeps
	// working; otherwise this test would pass for the wrong reason.
	fresh, _ := attached(t, &fakeKfuncs{}, cudabpf.Config{Budget: 4})
	fresh.SetMemoryPolicy(&staticPolicy{steps: 2, reorder: reorderOpinion{
		cudabpf.Reorder{Destination: cudabpf.DestinationUsed, Position: cudabpf.PositionHead}, true}})
	for i := 0; i < 3; i++ {
		if out := fresh.OnBlockAccess(&cudabpf.PMMDecision{}); !out.Submitted {
			t.Fatalf("invocation %d within budget did not submit: %+v", i, out)
		}
	}
}

// TestDisableSwitchRestoresDefaultPolicy covers the required per-hook and
// per-process switches. Attachment is opt-in and reversible, and turning it
// off must restore the driver's behavior immediately without a reload.
func TestDisableSwitchRestoresDefaultPolicy(t *testing.T) {
	k := &fakeKfuncs{}
	br, _ := attached(t, k, cudabpf.Config{})
	br.SetMemoryPolicy(&staticPolicy{reorder: reorderOpinion{
		cudabpf.Reorder{Destination: cudabpf.DestinationUsed, Position: cudabpf.PositionHead}, true}})

	if out := br.OnBlockAccess(&cudabpf.PMMDecision{}); !out.Submitted {
		t.Fatalf("baseline invocation did not submit: %+v", out)
	}

	br.DisableHook("gpu_block_access", "operator request")
	out := br.OnBlockAccess(&cudabpf.PMMDecision{})
	if !out.Fallback || out.Reason != "operator request" {
		t.Errorf("per-hook disable: %+v", out)
	}
	if out := br.OnBlockActivate(&cudabpf.PMMDecision{}); !out.Submitted {
		t.Errorf("a per-hook disable stopped another hook: %+v", out)
	}
	br.EnableHook("gpu_block_access")
	if out := br.OnBlockAccess(&cudabpf.PMMDecision{}); !out.Submitted {
		t.Errorf("re-enable did not restore the hook: %+v", out)
	}

	br.Disable()
	for _, out := range []cudabpf.Outcome{
		br.OnBlockAccess(&cudabpf.PMMDecision{}),
		br.OnBlockActivate(&cudabpf.PMMDecision{}),
	} {
		if !out.Fallback || out.Reason != "bridge disabled" {
			t.Errorf("process disable: %+v", out)
		}
	}
	br.Enable()
	if out := br.OnBlockAccess(&cudabpf.PMMDecision{}); !out.Submitted {
		t.Errorf("process enable did not restore policy: %+v", out)
	}

	// Close is the irreversible one, and it detaches.
	if err := br.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if out := br.OnBlockAccess(&cudabpf.PMMDecision{}); !out.Fallback {
		t.Errorf("policy still ran after Close: %+v", out)
	}
}

// TestUnmappedAddressHandledWithoutFault covers unmapped and partially mapped
// ranges. A handler reached with nothing to decide about must return without
// dereferencing anything, and a partial range must be refused rather than
// clamped into a plausible one.
func TestUnmappedAddressHandledWithoutFault(t *testing.T) {
	k := &fakeKfuncs{}
	br, _ := attached(t, k, cudabpf.Config{})
	br.SetMemoryPolicy(&cudabpf.HolderPolicy{
		State: mapState{1: 2},
		Block: func(s cudabpf.PMMSnapshot) (int, bool) {
			if s.RootID == 0 {
				return 0, false // not KV cache: no opinion
			}
			return int(s.RootID), true
		},
	})

	// A nil context, an unmapped chunk, and a chunk that is not KV cache.
	for _, d := range []*cudabpf.PMMDecision{
		nil,
		{},
		{Observed: cudabpf.PMMSnapshot{RootID: 0}, RootChunk: 0},
	} {
		out := br.OnBlockAccess(d)
		if out.Submitted {
			t.Errorf("submitted a decision for %+v", d)
		}
	}
	if n, _, _ := k.counts(); n != 0 {
		t.Errorf("%d kfunc calls for unmapped input", n)
	}

	// A partially mapped range reaches the policy as a region that does
	// not close. It is refused, not clamped.
	for _, r := range []cudabpf.Region{{First: 10, Outer: 10}, {First: 10, Outer: 4}} {
		if err := r.Valid(); err == nil {
			t.Errorf("region %s was accepted", r)
		}
	}
	w := cudabpf.NewPrefetchWorker(k, 4, nil)
	if w.Submit(cudabpf.PrefetchRequest{ID: 1, Region: cudabpf.Region{First: 8, Outer: 8}}) {
		t.Error("an empty prefetch region was queued")
	}
}

// TestPrefetchIssuanceNonBlocking covers the rule that expensive work leaves
// the fault path. Submit never blocks: a full queue is reported to the caller
// rather than absorbed, and the caller can tell the difference.
func TestPrefetchIssuanceNonBlocking(t *testing.T) {
	k := &fakeKfuncs{}
	w := cudabpf.NewPrefetchWorker(k, 2, nil) // not started: nothing drains
	defer w.Close()

	done := make(chan bool, 1)
	go func() {
		accepted := 0
		for i := 0; i < 8; i++ {
			if w.Submit(cudabpf.PrefetchRequest{
				ID: uint64(i + 1), Decision: &cudabpf.PrefetchDecision{}, Region: cudabpf.Region{First: uint64(i) * 16, Outer: uint64(i)*16 + 16},
			}) {
				accepted++
			}
		}
		done <- accepted == 2
	}()
	select {
	case ok := <-done:
		if !ok {
			t.Error("a queue of depth 2 accepted a number of requests other than 2")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Submit blocked on a full queue")
	}
	if _, n, _ := k.counts(); n != 0 {
		t.Errorf("%d prefetches issued from the submitting goroutine", n)
	}
}

// TestPrefetchCancellationAfterRequest covers cancellation, including the
// case it cannot honor. A request already issued cannot be recalled, and
// saying otherwise would make a cancelled prefetch indistinguishable from one
// that never ran.
func TestPrefetchCancellationAfterRequest(t *testing.T) {
	k := &fakeKfuncs{}
	w := cudabpf.NewPrefetchWorker(k, 4, nil)

	region := cudabpf.Region{First: 0, Outer: 64}
	if !w.Submit(cudabpf.PrefetchRequest{ID: 1, Region: region, Decision: &cudabpf.PrefetchDecision{}}) {
		t.Fatal("Submit refused a valid request")
	}
	if !w.Cancel(1) {
		t.Fatal("Cancel refused a queued request")
	}
	w.Start()
	w.Close()

	results := w.Results()
	if len(results) != 1 {
		t.Fatalf("got %d results, want 1", len(results))
	}
	if !results[0].Cancelled || results[0].Issued {
		t.Errorf("result = %+v, want cancelled and not issued", results[0])
	}
	if _, n, _ := k.counts(); n != 0 {
		t.Errorf("%d prefetches issued despite cancellation", n)
	}
	if results[0].Latency != cudabpf.Unmeasured {
		t.Errorf("latency = %v with no clock, want Unmeasured", results[0].Latency)
	}
	if w.Cancel(1) {
		t.Error("cancelling an already-finished request reported success")
	}

	// The control: an uncancelled request is issued.
	k2 := &fakeKfuncs{}
	w2 := cudabpf.NewPrefetchWorker(k2, 4, nil)
	w2.Start()
	if !w2.Submit(cudabpf.PrefetchRequest{ID: 2, Region: region, Decision: &cudabpf.PrefetchDecision{}}) {
		t.Fatal("Submit refused")
	}
	w2.Close()
	res := w2.Results()
	if len(res) != 1 || !res[0].Issued {
		t.Fatalf("uncancelled request: %+v", res)
	}
	if _, n, _ := k2.counts(); n != 1 {
		t.Errorf("%d prefetches issued, want 1", n)
	}
}

// TestDetachDuringInFlightFault covers the negative control of the same name:
// detaching while a handler is running must not lose the handler's outcome
// and must not leave the placement undefined.
func TestDetachDuringInFlightFault(t *testing.T) {
	k := &fakeKfuncs{}
	br, _ := attached(t, k, cudabpf.Config{})

	release := make(chan struct{})
	br.SetMemoryPolicy(&blockingPolicy{release: release})

	type result struct{ out cudabpf.Outcome }
	ch := make(chan result, 1)
	go func() { ch <- result{br.OnBlockAccess(&cudabpf.PMMDecision{})} }()

	if err := br.Detach(cudabpf.MemOpsTable); err != nil {
		t.Fatalf("Detach during a fault: %v", err)
	}
	close(release)

	select {
	case r := <-ch:
		if r.out.Fallback && r.out.Reason == "" {
			t.Error("the in-flight handler fell back with no reason")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("the in-flight handler never returned after a detach")
	}
	// After the detach, later invocations see no attachment but still
	// produce a defined outcome.
	if out := br.OnBlockAccess(&cudabpf.PMMDecision{}); out.Hook == "" {
		t.Error("an invocation after detach produced no outcome")
	}
}

type blockingPolicy struct{ release chan struct{} }

func (p *blockingPolicy) BlockAccess(_ *cudabpf.PMMDecision, b *cudabpf.Budget) (cudabpf.Reorder, bool, error) {
	<-p.release
	return cudabpf.Reorder{Destination: cudabpf.DestinationUsed, Position: cudabpf.PositionHead}, true, nil
}

func (p *blockingPolicy) BlockActivate(d *cudabpf.PMMDecision, b *cudabpf.Budget) (cudabpf.Reorder, bool, error) {
	return p.BlockAccess(d, b)
}

func (p *blockingPolicy) Prefetch(*cudabpf.PrefetchDecision, *cudabpf.Budget) (cudabpf.Region, bool, error) {
	return cudabpf.Region{}, false, nil
}

type mapState map[int]int

func (m mapState) Holders(block int) (int, error) {
	n, ok := m[block]
	if !ok {
		return 0, errors.New("unknown block")
	}
	return n, nil
}

// TestObservationIsBoundedAndOutOfLine covers step 7's constraint: a fault
// handler does no logging beyond a fixed-size counter or ring slot.
func TestObservationIsBoundedAndOutOfLine(t *testing.T) {
	var c cudabpf.Counters
	br, _ := attached(t, &fakeKfuncs{}, cudabpf.Config{Observer: &c})
	br.SetMemoryPolicy(&staticPolicy{reorder: reorderOpinion{
		cudabpf.Reorder{Destination: cudabpf.DestinationUsed, Position: cudabpf.PositionHead}, true}})

	for i := 0; i < 100; i++ {
		br.OnBlockAccess(&cudabpf.PMMDecision{})
	}
	if c.Consulted != 100 || c.Submitted != 100 {
		t.Errorf("counters = %s", c)
	}
	if c.Fallbacks != 0 {
		t.Errorf("unexpected fallbacks: %s", c)
	}
	// The handler time is Unmeasured, not zero, when no clock was given.
	for _, o := range br.Outcomes() {
		if o.HandlerTime != cudabpf.Unmeasured {
			t.Fatalf("handler time = %v with no clock, want Unmeasured", o.HandlerTime)
		}
	}
	// And the non-additive timing fields default to unmeasured too.
	tm := cudabpf.UnmeasuredTimings()
	if tm.FaultHandling != cudabpf.Unmeasured || tm.PrefetchWorker != cudabpf.Unmeasured ||
		tm.Transfer != cudabpf.Unmeasured || tm.GPUCompute != cudabpf.Unmeasured {
		t.Errorf("UnmeasuredTimings = %+v", tm)
	}
}
