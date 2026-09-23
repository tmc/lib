package cudamegacpu

import (
	"sync/atomic"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

// TestDependencyWaitIsLoadBearing checks the property the completion word
// exists for: no worker begins a task before every worker has published that
// task's dependency.
//
// It exists because comparing the run against the ordinary-launch reference
// does not check that property and cannot be made to. The emitted worker
// gives element e to the same thread in every task -- the stride is the whole
// grid's thread count and does not change from task to task -- so every
// element a consumer reads was written by the very thread reading it. The
// values therefore agree whether or not the cross-worker wait happened, and a
// run with the wait deleted produces the correct answer. That is a property
// of the elementwise fixture, not a defect, and it is exactly why the wait
// needs a test that looks at the ordering rather than at the numbers.
//
// The skew is what makes the check fail when the wait is gone rather than
// merely usually pass. One worker is held back before publishing task zero,
// long enough that a worker which did not wait would already be computing
// task one.
func TestDependencyWaitIsLoadBearing(t *testing.T) {
	g, err := cudataskgraph.New([]cudataskgraph.Task{
		{ID: "seed", Rank: 0, Bytes: 16, Scope: cudataskgraph.ScopeDevice},
		{ID: "scale", Deps: []string{"seed"}, Rank: 1, Bytes: 16, Scope: cudataskgraph.ScopeDevice},
	})
	if err != nil {
		t.Fatal(err)
	}
	ops := map[string]cudamega.Operation{
		"seed":  {Kind: cudamega.Add, Scalar: 3},
		"scale": {Kind: cudamega.Multiply, Scalar: 2},
	}
	const ctas, threads = 2, 32
	p, err := cudamega.NewProgram(g, ops, cudamega.Grid{CTAs: ctas, ThreadsPerCTA: threads}, cudamega.SM80)
	if err != nil {
		t.Fatal(err)
	}
	e, err := New(cudamega.Admission{ResidentCTAs: ctas, WorkerThreads: threads, SharedBytes: 64})
	if err != nil {
		t.Fatal(err)
	}
	var early atomic.Int32
	e.wrap = func(b cudamega.Backend) cudamega.Backend { return &watching{worker: b.(*worker), early: &early} }

	in := make([]uint32, 32)
	for i := range in {
		in[i] = uint32(i)
	}
	if _, err := e.Run(p, in, 1, cudamega.AbortNone); err != nil {
		t.Fatal(err)
	}
	if n := early.Load(); n != 0 {
		t.Fatalf("%d workers began a task before every worker had published its dependency", n)
	}
}

// watching delegates every primitive, holds one worker back before it
// publishes task zero, and counts the workers that begin a dependent task
// before its dependency is fully published.
type watching struct {
	*worker
	early *atomic.Int32
}

func (w *watching) Compute(i, src int, op cudamega.Operation) error {
	if src >= 0 && w.r.completed[src].Load() < w.r.ctas {
		w.early.Add(1)
	}
	return w.worker.Compute(i, src, op)
}

func (w *watching) Publish(i int) error {
	if i == 0 && w.cta == 0 && w.tid == 0 {
		time.Sleep(20 * time.Millisecond)
	}
	return w.worker.Publish(i)
}

// TestBarrierReleasesEveryThreadOnce checks the reusable barrier on its own.
// A barrier that released a thread early, or that lost one across phases,
// would make every test above either flaky or deadlocked, which are the two
// failures hardest to attribute to their cause.
func TestBarrierReleasesEveryThreadOnce(t *testing.T) {
	const n = 8
	b := newBarrier(n)
	var passed atomic.Int32
	done := make(chan struct{})
	for i := 0; i < n; i++ {
		go func() {
			for r := 0; r < 100; r++ {
				b.wait()
				passed.Add(1)
			}
			done <- struct{}{}
		}()
	}
	for i := 0; i < n; i++ {
		<-done
	}
	if got, want := passed.Load(), int32(n*100); got != want {
		t.Fatalf("barrier passed %d times, want %d", got, want)
	}
}

// TestCrossWorkerResultIsLoadBearing is the equivalence gate made
// deterministic.
//
// The cross-worker demonstrator makes "matches the ordinary-launch reference"
// able to see a missing dependency wait, which the earlier fixtures could not:
// there the element a consumer read was written by the thread reading it, so
// deleting the wait left the comparison green. But able to see is not the same
// as certain to see. With the wait deleted and nothing skewing the workers,
// the producer usually still runs first and the comparison still passes: over
// five runs of the unskewed test in cudamegacpu_test with the wait mutated
// away, one came back green.
//
// So the skew is part of the gate rather than a convenience. One worker is
// held back before publishing, long enough that a consumer which did not wait
// reads the scratch word before the producer wrote it, and the comparison then
// fails every time rather than usually.
//
// The consumers here read only elements the other half of the grid owns, so
// what they read early is the zero the scratch buffer was allocated with, not
// a value that happens to be right.
func TestCrossWorkerResultIsLoadBearing(t *testing.T) {
	g, err := cudataskgraph.New([]cudataskgraph.Task{
		{ID: "lower", Rank: 0, Bytes: 16, Scope: cudataskgraph.ScopeDevice},
		{ID: "upper", Rank: 0, Bytes: 16, Scope: cudataskgraph.ScopeDevice},
		{ID: "cross", Deps: []string{"upper"}, Rank: 1, Bytes: 16, Scope: cudataskgraph.ScopeDevice},
		{ID: "swap", Deps: []string{"lower"}, Rank: 1, Bytes: 16, Scope: cudataskgraph.ScopeSystem},
	})
	if err != nil {
		t.Fatal(err)
	}
	ops := map[string]cudamega.Operation{
		"lower": {Kind: cudamega.Add, Scalar: 3},
		"upper": {Kind: cudamega.Add, Scalar: 5},
		"cross": {Kind: cudamega.Multiply, Scalar: 2},
		"swap":  {Kind: cudamega.Add, Scalar: 7},
	}
	const ctas, threads = 4, 32
	p, err := cudamega.NewProgram(g, ops, cudamega.Grid{CTAs: ctas, ThreadsPerCTA: threads, Policy: cudamega.CountedPolicy}, cudamega.SM121a)
	if err != nil {
		t.Fatal(err)
	}
	steps := p.Steps()
	for _, s := range steps {
		for _, d := range s.Deps {
			c, q := s.Owner, steps[d].Owner
			if c.Lo < q.Hi && q.Lo < c.Hi {
				t.Fatalf("task %q owns %s and its dependency %q owns %s; the ranges overlap and the comparison below is blind again",
					s.Task.ID, c, steps[d].Task.ID, q)
			}
		}
	}
	e, err := New(cudamega.Admission{ResidentCTAs: ctas, WorkerThreads: threads, SharedBytes: 64})
	if err != nil {
		t.Fatal(err)
	}
	e.wrap = func(b cudamega.Backend) cudamega.Backend { return &skewed{worker: b.(*worker)} }

	in := make([]uint32, 64)
	for i := range in {
		in[i] = uint32(i)
	}
	got, err := e.Run(p, in, 1, cudamega.AbortNone)
	if err != nil {
		t.Fatal(err)
	}
	if got.Abort != cudamega.AbortNone {
		t.Fatalf("run aborted with code %d", got.Abort)
	}
	want, err := Reference(p, in)
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if got.Scratch[i] != want[i] {
			t.Fatalf("element %d of task %d: skewed run %d, reference %d",
				i%len(in), i/len(in), got.Scratch[i], want[i])
		}
	}
}

// skewed holds one producer back so that a consumer which took its task
// without waiting reads the scratch buffer before anything wrote it.
//
// The delay is before the producer computes, not before it publishes. A delay
// placed at the publication is no skew at all: the producer has already
// written its elements by then, so a consumer that did not wait still reads
// the right values and the comparison passes. That version of this wrapper
// was written first and let the mutated wait through.
//
// Only "upper" is held. Delaying both producers delays the whole grid equally
// and re-synchronizes it, which is the same absence of skew by a different
// route; holding one lets the half of the grid that owns "lower" and "cross"
// run ahead into a task whose input has not been written.
func (w *skewed) Compute(i, src int, op cudamega.Operation) error {
	if i == 1 && w.tid == 0 {
		time.Sleep(20 * time.Millisecond)
	}
	return w.worker.Compute(i, src, op)
}

type skewed struct {
	*worker
}
