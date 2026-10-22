package cudamegacpu

import (
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

type heldEntry struct {
	cudamega.Backend
	w                *worker
	entered, release chan struct{}
}

func (h *heldEntry) Begin(p *cudamega.Program) error {
	if err := h.Backend.Begin(p); err != nil {
		return err
	}
	if h.w.cta == 1 && h.w.tid == 0 {
		close(h.entered)
		<-h.release
	}
	return nil
}

// A late scanner owns a poisoned consumer slot. No producer is allowed to
// publish before that scanner joins entry, even though its own slot is clean.
func TestEntryBeforeAnyPublication(t *testing.T) {
	for _, initial := range []uint32{0, 1, 17} {
		t.Run(string(rune('a'+initial)), func(t *testing.T) {
			g, err := cudataskgraph.New([]cudataskgraph.Task{
				{ID: "producer", Scope: cudataskgraph.ScopeDevice},
				{ID: "consumer", Rank: 1, Deps: []string{"producer"}, Scope: cudataskgraph.ScopeDevice},
			})
			if err != nil {
				t.Fatal(err)
			}
			p, err := cudamega.NewProgram(g, map[string]cudamega.Operation{"producer": {Kind: cudamega.Add, Scalar: 3}, "consumer": {Kind: cudamega.Multiply, Scalar: 2}}, cudamega.Grid{CTAs: 2, ThreadsPerCTA: 32}, cudamega.SM80)
			if err != nil {
				t.Fatal(err)
			}
			e, err := New(cudamega.Admission{ResidentCTAs: 2, WorkerThreads: 32, SharedBytes: 1})
			if err != nil {
				t.Fatal(err)
			}
			entered, release := make(chan struct{}), make(chan struct{})
			var once sync.Once
			var state *run
			e.wrap = func(b cudamega.Backend) cudamega.Backend {
				w := b.(*worker)
				once.Do(func() { state = w.r; state.completed[1].Store(initial) })
				return &heldEntry{Backend: b, w: w, entered: entered, release: release}
			}
			type outcome struct {
				result Result
				err    error
			}
			done := make(chan outcome, 1)
			go func() { r, err := e.Run(p, []uint32{4, 5, 6}, 1, 0); done <- outcome{r, err} }()
			select {
			case <-entered:
			case <-time.After(5 * time.Second):
				t.Fatal("late worker not entered")
			}
			deadline := time.Now().Add(5 * time.Second)
			for state.arrivals.Load() == 0 && time.Now().Before(deadline) {
				runtime.Gosched()
			}
			arrivals, tickets, completed := state.arrivals.Load(), state.ticket.Load(), state.completed[0].Load()
			close(release)
			var out outcome
			select {
			case out = <-done:
			case <-time.After(5 * time.Second):
				t.Fatal("entry did not terminate")
			}
			if arrivals != 1 || tickets != 0 || completed != 0 {
				t.Fatalf("before all entries: arrivals=%d tickets=%d completed=%d", arrivals, tickets, completed)
			}
			if out.err != nil {
				t.Fatal(out.err)
			}
			if initial != 0 {
				if out.result.Abort != cudamega.AbortStaleCompletion || state.ticket.Load() != 0 || state.completed[0].Load() != 0 {
					t.Fatalf("poison published: %+v", out.result)
				}
				for _, x := range out.result.Scratch {
					if x != 0 {
						t.Fatal("poison wrote scratch")
					}
				}
			} else if out.result.Abort != 0 || state.ticket.Load() != 4 {
				t.Fatalf("clean entry: %+v", out.result)
			}
		})
	}
}

func TestEntryPreservesHostAbort(t *testing.T) {
	g, _ := cudataskgraph.New([]cudataskgraph.Task{{ID: "a", Scope: cudataskgraph.ScopeDevice}})
	p, err := cudamega.NewProgram(g, map[string]cudamega.Operation{"a": {Kind: cudamega.Add}}, cudamega.Grid{CTAs: 1, ThreadsPerCTA: 32}, cudamega.SM80)
	if err != nil {
		t.Fatal(err)
	}
	e, _ := New(cudamega.Admission{ResidentCTAs: 1, WorkerThreads: 32, SharedBytes: 1})
	r, err := e.Run(p, []uint32{1}, 1, cudamega.AbortHostRequest)
	if err != nil || r.Abort != cudamega.AbortHostRequest || r.StartOrder[0] != 0 {
		t.Fatalf("host abort: %+v %v", r, err)
	}
}

func TestEntryMissingWorkerIsBounded(t *testing.T) {
	g, _ := cudataskgraph.New([]cudataskgraph.Task{{ID: "a", Scope: cudataskgraph.ScopeDevice}})
	p, err := cudamega.NewProgram(g, map[string]cudamega.Operation{"a": {Kind: cudamega.Add}}, cudamega.Grid{CTAs: 2, ThreadsPerCTA: 32, SpinLimit: 1}, cudamega.SM80)
	if err != nil {
		t.Fatal(err)
	}
	r := &run{p: p, steps: p.Steps(), ctas: 2, completed: make([]atomic.Uint32, 1), cta: make([]ctaState, 2)}
	// Drive just the leader's entry primitive: the missing second CTA cannot
	// publish an arrival. A one-participant test barrier avoids starting lanes.
	r.cta[0].barrier = newBarrier(1)
	w := &worker{r: r}
	if err := w.Entry(); err != nil {
		t.Fatal(err)
	}
	if r.abort.Load() != cudamega.AbortSpinExhausted || r.arrivals.Load() != 1 || r.released.Load() != 0 || r.ticket.Load() != 0 || !w.stopped {
		t.Fatalf("unbounded/accepted missing worker: abort=%d arrivals=%d release=%d", r.abort.Load(), r.arrivals.Load(), r.released.Load())
	}
}
