package cudamega_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

func admission() cudamega.Admission {
	return cudamega.Admission{ResidentCTAs: 4, WorkerThreads: 128, SharedBytes: 1024}
}

// TestFiniteSchedulerProgress checks the progress argument against the
// schedule it describes: the grid comes from the compiled worker's admission,
// every dependency runs in an earlier wave than its consumer, and the barrier
// phase alternates across waves rather than resetting.
func TestFiniteSchedulerProgress(t *testing.T) {
	g, err := cudataskgraph.New([]cudataskgraph.Task{
		{ID: "a", Rank: 0, Bytes: 64, Scope: cudataskgraph.ScopeDevice},
		{ID: "b", Rank: 0, Bytes: 64, Scope: cudataskgraph.ScopeDevice},
		{ID: "c", Deps: []string{"a", "b"}, Rank: 1, Bytes: 64, Scope: cudataskgraph.ScopeDevice},
		{ID: "d", Deps: []string{"c"}, Rank: 2, Bytes: 64, Scope: cudataskgraph.ScopeSystem},
	})
	if err != nil {
		t.Fatal(err)
	}
	p, err := cudamega.Progress(g, admission())
	if err != nil {
		t.Fatal(err)
	}
	if p.Grid.CTAs != admission().ResidentCTAs || p.Grid.ThreadsPerCTA != admission().WorkerThreads {
		t.Fatalf("grid %+v is not the compiled worker's admitted residency", p.Grid)
	}
	if len(p.Waves) != 4 {
		t.Fatalf("got %d waves for 4 tasks", len(p.Waves))
	}
	at := map[string]int{}
	for i, w := range p.Waves {
		if i > 0 && w.Rank < p.Waves[i-1].Rank {
			t.Fatalf("wave %d has rank %d after rank %d", i, w.Rank, p.Waves[i-1].Rank)
		}
		if want := uint32(i & 1); w.Parity != want {
			t.Fatalf("wave %d arrives on parity %d, want %d", i, w.Parity, want)
		}
		at[w.Task] = i
	}
	for _, task := range g.Tasks() {
		for _, d := range task.Deps {
			if at[d] >= at[task.ID] {
				t.Fatalf("dependency %q runs in wave %d, consumer %q in wave %d", d, at[d], task.ID, at[task.ID])
			}
		}
	}
	if len(p.Invariants) == 0 {
		t.Fatal("progress argument states no invariants")
	}
	if _, err := cudamega.Progress(g, cudamega.Admission{}); err == nil {
		t.Fatal("derived a grid from an empty admission")
	}
}

// TestFanInResourcePressure forces the consumer-first, resource-saturating
// case: the tasks are presented consumer first, and their shared-memory
// requirements sum to four times the compiled worker's admission.
//
// The admission check is against the widest single task, not the sum, because
// the generated worker runs one task at a time and a task's shared window is
// gone when it ends. Summing would reject a schedule that fits, which is the
// failure that looks like caution.
func TestFanInResourcePressure(t *testing.T) {
	tasks := []cudataskgraph.Task{
		{ID: "sink", Deps: []string{"p0", "p1", "p2", "p3"}, Rank: 1, Bytes: 1024, Scope: cudataskgraph.ScopeDevice},
		{ID: "p0", Rank: 0, Bytes: 1024, Scope: cudataskgraph.ScopeDevice},
		{ID: "p1", Rank: 0, Bytes: 1024, Scope: cudataskgraph.ScopeDevice},
		{ID: "p2", Rank: 0, Bytes: 1024, Scope: cudataskgraph.ScopeDevice},
		{ID: "p3", Rank: 0, Bytes: 1024, Scope: cudataskgraph.ScopeDevice},
	}
	g, err := cudataskgraph.New(tasks)
	if err != nil {
		t.Fatal(err)
	}
	p, err := cudamega.Progress(g, admission())
	if err != nil {
		t.Fatalf("rejected a fan-in whose widest task fits the admission: %v", err)
	}
	if p.Waves[len(p.Waves)-1].Task != "sink" {
		t.Fatalf("consumer-first input produced a consumer-first schedule: %+v", p.Waves)
	}
	if got := p.RankWidths; len(got) != 2 || got[0] != 4 || got[1] != 1 {
		t.Fatalf("rank widths %v do not describe a 4-to-1 fan-in", got)
	}

	// One task over the admission is the case that must be refused, and the
	// refusal must name that task: a message naming the sum would describe a
	// limit this scheduler does not enforce.
	tasks[3].Bytes = 1025
	over, err := cudataskgraph.New(tasks)
	if err != nil {
		t.Fatal(err)
	}
	_, err = cudamega.Progress(over, admission())
	if err == nil {
		t.Fatal("admitted a task wider than the compiled worker's shared memory")
	}
	if !strings.Contains(err.Error(), `"p2"`) {
		t.Fatalf("refusal does not name the oversized task: %v", err)
	}
}

// TestCTARankAssignmentValidation covers contracts 1 and 2 at the boundary
// each is enforced: rank precedence when the graph is built, and the static
// grid when it is derived.
func TestCTARankAssignmentValidation(t *testing.T) {
	// R_p < R_c is strict: an equal rank is a dependency the rank order
	// cannot schedule, and the persistent grid has no later pass to put it in.
	if _, err := cudataskgraph.New([]cudataskgraph.Task{
		{ID: "p", Rank: 1, Scope: cudataskgraph.ScopeDevice},
		{ID: "c", Deps: []string{"p"}, Rank: 1, Scope: cudataskgraph.ScopeDevice},
	}); err == nil {
		t.Fatal("accepted a dependency at the consumer's own rank")
	}
	g, err := cudataskgraph.New([]cudataskgraph.Task{
		{ID: "p", Rank: 0, Bytes: 8, Scope: cudataskgraph.ScopeDevice},
		{ID: "c", Deps: []string{"p"}, Rank: 1, Bytes: 8, Scope: cudataskgraph.ScopeDevice},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		a    cudamega.Admission
		want uint32
	}{
		{"one resident CTA", cudamega.Admission{ResidentCTAs: 1, WorkerThreads: 64, SharedBytes: 64}, 1},
		{"more CTAs than tasks", cudamega.Admission{ResidentCTAs: 16, WorkerThreads: 64, SharedBytes: 64}, 16},
	} {
		grid, err := cudamega.CTAGrid(tc.a, g)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if grid.CTAs != tc.want {
			t.Fatalf("%s: grid has %d CTAs, want %d", tc.name, grid.CTAs, tc.want)
		}
	}
	for _, bad := range []cudamega.Admission{
		{ResidentCTAs: 0, WorkerThreads: 64, SharedBytes: 64},
		{ResidentCTAs: 1, WorkerThreads: 0, SharedBytes: 64},
		{ResidentCTAs: 1, WorkerThreads: 2048, SharedBytes: 64},
		{ResidentCTAs: 1, WorkerThreads: 64, SharedBytes: 0},
	} {
		if _, err := cudamega.CTAGrid(bad, g); err == nil {
			t.Fatalf("derived a grid from admission %+v", bad)
		}
	}
	// A grid the generated source was not written for is refused by the
	// generator, because the emitted worker checks gridDim against a
	// compiled-in constant.
	src, err := cudamega.ProgramSource(g, map[string]cudamega.Operation{
		"p": {Kind: cudamega.Add, Scalar: 1},
		"c": {Kind: cudamega.Multiply, Scalar: 2},
	}, cudamega.Grid{CTAs: 3, ThreadsPerCTA: 64})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(src, "constexpr unsigned grid_ctas = 3;") {
		t.Fatal("generated source does not compile in the grid it was generated for")
	}
	if !strings.Contains(src, "gridDim.x != grid_ctas") {
		t.Fatal("generated worker does not refuse a geometry other than its own")
	}
}

// TestAbortPreservesOwnership is the cancel-while-owned control: aborting a
// launch must not be a licence to free what the worker may still be storing
// into.
func TestAbortPreservesOwnership(t *testing.T) {
	g, err := cudataskgraph.New([]cudataskgraph.Task{
		{ID: "p", Rank: 0, Bytes: 8, Scope: cudataskgraph.ScopeDevice},
		{ID: "c", Deps: []string{"p"}, Rank: 1, Bytes: 8, Scope: cudataskgraph.ScopeDevice},
	})
	if err != nil {
		t.Fatal(err)
	}
	e, err := cudamega.NewExecutor(g, admission(), 7)
	if err != nil {
		t.Fatal(err)
	}
	if got := e.Epoch().Retained(); len(got) != 2 {
		t.Fatalf("launch claims %v, want both task outputs", got)
	}
	retained := e.Abort()
	if len(retained) != 2 || retained[0] != "c" || retained[1] != "p" {
		t.Fatalf("abort retained %v, want both task outputs", retained)
	}
	if !e.Epoch().Aborted() {
		t.Fatal("abort did not mark the epoch")
	}
	// The refusal must name the outstanding abort. If it ever starts saying
	// the storage was never claimed, the ownership record has been lost and
	// this test must fail even though a release is still refused.
	err = e.Release("p")
	if err == nil {
		t.Fatal("released output storage while an abort was outstanding")
	}
	if !strings.Contains(err.Error(), "abort is outstanding") {
		t.Fatalf("refusal does not name the outstanding abort: %v", err)
	}
	if strings.Contains(err.Error(), "not claimed") {
		t.Fatalf("refusal claims the storage was never owned: %v", err)
	}
	e.Drain()
	if err := e.Release("p"); err != nil {
		t.Fatalf("refused release after the launch was drained: %v", err)
	}
	if got := e.Epoch().Retained(); len(got) != 1 || got[0] != "c" {
		t.Fatalf("after releasing p, retained %v", got)
	}
	if err := e.Release("p"); err == nil {
		t.Fatal("released the same output storage twice")
	}
	if _, err := cudamega.NewExecutor(g, admission(), 0); err == nil {
		t.Fatal("launched epoch zero, which is the unpublished value")
	}
}
