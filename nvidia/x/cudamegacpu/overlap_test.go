package cudamegacpu_test

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/cudamegacpu"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

// crossGraph is the S1 demonstrator, restated from cudamega's golden fixture
// for the same reason the two graphs above are restated: a copy that drifted
// would fail cudamega's own golden test first.
//
// Its point is the one property the other two fixtures do not have. Under
// [cudamega.CountedPolicy] "lower" and "cross" are dealt to the first half of
// the grid and "upper" and "swap" to the second, while "cross" depends on
// "upper" and "swap" on "lower". Every element each consumer reads was
// therefore written by a worker other than the one reading it, which is what
// makes comparing the result against the ordinary-launch reference a test of
// the dependency wait rather than of arithmetic.
func crossGraph(t *testing.T) (cudataskgraph.Graph, map[string]cudamega.Operation) {
	t.Helper()
	g, err := cudataskgraph.New([]cudataskgraph.Task{
		{ID: "lower", Rank: 0, Bytes: 16, Scope: cudataskgraph.ScopeDevice},
		{ID: "upper", Rank: 0, Bytes: 16, Scope: cudataskgraph.ScopeDevice},
		{ID: "cross", Deps: []string{"upper"}, Rank: 1, Bytes: 16, Scope: cudataskgraph.ScopeDevice},
		{ID: "swap", Deps: []string{"lower"}, Rank: 1, Bytes: 16, Scope: cudataskgraph.ScopeSystem},
	})
	if err != nil {
		t.Fatal(err)
	}
	return g, map[string]cudamega.Operation{
		"lower": {Kind: cudamega.Add, Scalar: 3},
		"upper": {Kind: cudamega.Add, Scalar: 5},
		"cross": {Kind: cudamega.Multiply, Scalar: 2},
		"swap":  {Kind: cudamega.Add, Scalar: 7},
	}
}

// chainGraph is the control for the interleaving observation: four tasks in a
// line, so that a task cannot start anywhere in the grid until every worker
// has published the one before it.
func chainGraph(t *testing.T) (cudataskgraph.Graph, map[string]cudamega.Operation) {
	t.Helper()
	var tasks []cudataskgraph.Task
	ops := map[string]cudamega.Operation{}
	for i := 0; i < 4; i++ {
		id := fmt.Sprintf("t%d", i)
		task := cudataskgraph.Task{ID: id, Rank: i, Bytes: 16, Scope: cudataskgraph.ScopeDevice}
		if i > 0 {
			task.Deps = []string{fmt.Sprintf("t%d", i-1)}
		}
		tasks = append(tasks, task)
		ops[id] = cudamega.Operation{Kind: cudamega.Add, Scalar: uint32(i + 1)}
	}
	g, err := cudataskgraph.New(tasks)
	if err != nil {
		t.Fatal(err)
	}
	return g, ops
}

func buildPolicy(t *testing.T, g cudataskgraph.Graph, ops map[string]cudamega.Operation, ctas, threads uint32, policy cudamega.Policy) (*cudamega.Program, *cudamegacpu.Executor) {
	t.Helper()
	p, err := cudamega.NewProgram(g, ops, cudamega.Grid{CTAs: ctas, ThreadsPerCTA: threads, Policy: policy}, cudamega.SM121a)
	if err != nil {
		t.Fatal(err)
	}
	e, err := cudamegacpu.New(cudamega.Admission{ResidentCTAs: ctas, WorkerThreads: threads, SharedBytes: 64})
	if err != nil {
		t.Fatal(err)
	}
	return p, e
}

// TestCrossWorkerDependencyMatchesReference is the equivalence gate the S0.5
// finding said the earlier fixtures could not support.
//
// There, the emitted worker gave element e to the same (cta, tid) in every
// task, so the element a consumer read was written by the thread reading it
// and the comparison stayed green with the wait deleted. Here the owner
// ranges of each edge's two ends are disjoint, which the subtest asserts
// before comparing anything: if that assertion is ever removed or the policy
// changed, the comparison below goes back to being vacuous and this test says
// so rather than passing quietly.
func TestCrossWorkerDependencyMatchesReference(t *testing.T) {
	g, ops := crossGraph(t)
	in := input()
	for _, ctas := range []uint32{2, 4, 8} {
		t.Run(fmt.Sprint(ctas, " workers"), func(t *testing.T) {
			p, e := buildPolicy(t, g, ops, ctas, 32, cudamega.CountedPolicy)
			steps := p.Steps()
			edges := 0
			for _, s := range steps {
				for _, d := range s.Deps {
					edges++
					c, q := s.Owner, steps[d].Owner
					if c.Lo < q.Hi && q.Lo < c.Hi {
						t.Fatalf("task %q owns %s and its dependency %q owns %s; the ranges overlap, so the comparison below cannot see the wait",
							s.Task.ID, c, steps[d].Task.ID, q)
					}
				}
			}
			if edges == 0 {
				t.Fatal("no dependency edge: nothing for the wait to be load-bearing about")
			}
			got, err := e.Run(p, in, 1, cudamega.AbortNone)
			if err != nil {
				t.Fatal(err)
			}
			if got.Abort != cudamega.AbortNone {
				t.Fatalf("run aborted with code %d", got.Abort)
			}
			want, err := cudamegacpu.Reference(p, in)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got.Scratch, want) {
				t.Fatalf("persistent and ordinary-launch reference disagree\n got %v\nwant %v", got.Scratch, want)
			}
		})
	}
}

// startOrder is one run's start-order buffer read as a list of (ticket, task,
// cta), sorted by ticket, with the tasks no worker started left out.
type startEvent struct {
	ticket uint64
	task   int
	cta    uint32
}

func events(t *testing.T, order []uint64, tasks int, ctas uint32) []startEvent {
	t.Helper()
	if want := 1 + tasks*int(ctas); len(order) != want {
		t.Fatalf("start-order buffer has %d words, want %d", len(order), want)
	}
	var out []startEvent
	seen := map[uint64]bool{}
	for c := uint32(0); c < ctas; c++ {
		for i := 0; i < tasks; i++ {
			tk := order[1+int(c)*tasks+i]
			if tk == 0 {
				continue
			}
			if tk > order[0] || seen[tk] {
				t.Fatalf("task %d on worker %d holds ticket %d, which is out of range or reused", i, c, tk)
			}
			seen[tk] = true
			out = append(out, startEvent{tk, i, c})
		}
	}
	if uint64(len(out)) != order[0] {
		t.Fatalf("%d tasks were started but the ticket counter reads %d", len(out), order[0])
	}
	for i := range out {
		for j := i; j > 0 && out[j].ticket < out[j-1].ticket; j-- {
			out[j], out[j-1] = out[j-1], out[j]
		}
	}
	return out
}

// TestStartOrderRecordsWhichWorkerStartedWhat is the deterministic half of the
// S1 overlap gate.
//
// Nothing recorded task starts before this buffer, so neither the overlap
// claim nor its falsifier could be run at all. What the buffer makes checkable
// without depending on timing is ownership as the run actually performed it:
// under [cudamega.CountedPolicy] a task is started by exactly the workers its
// owner range names and by no others, and the two tasks of a rank are started
// by disjoint sets of workers -- which is what "two tasks are in flight at
// once" means structurally. Under [cudamega.WavePolicy] every worker starts
// every task, and the same assertion fails, which is this gate's control.
func TestStartOrderRecordsWhichWorkerStartedWhat(t *testing.T) {
	g, ops := crossGraph(t)
	in := input()
	for _, tc := range []struct {
		name   string
		policy cudamega.Policy
	}{
		{"counted", cudamega.CountedPolicy},
		{"wave", cudamega.WavePolicy},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const ctas = 4
			p, e := buildPolicy(t, g, ops, ctas, 32, tc.policy)
			got, err := e.Run(p, in, 1, cudamega.AbortNone)
			if err != nil {
				t.Fatal(err)
			}
			steps := p.Steps()
			evs := events(t, got.StartOrder, len(steps), ctas)

			started := make([]map[uint32]bool, len(steps))
			for i := range started {
				started[i] = map[uint32]bool{}
			}
			for _, ev := range evs {
				started[ev.task][ev.cta] = true
			}
			for i, s := range steps {
				want := map[uint32]bool{}
				for c := s.Owner.Lo; c < s.Owner.Hi; c++ {
					want[c] = true
				}
				if !reflect.DeepEqual(started[i], want) {
					t.Errorf("task %q owns %s but was started by workers %v", s.Task.ID, s.Owner, keys(started[i]))
				}
			}
			// The two tasks of each rank: disjoint workers under counted,
			// identical workers under wave.
			overlap := len(intersect(started[0], started[1])) > 0
			if tc.policy == cudamega.CountedPolicy && overlap {
				t.Errorf("counted policy started %q and %q on a common worker; they are meant to run side by side", steps[0].Task.ID, steps[1].Task.ID)
			}
			if tc.policy == cudamega.WavePolicy && !overlap {
				t.Errorf("wave policy did not start %q and %q on a common worker; under a wave every worker performs every task", steps[0].Task.ID, steps[1].Task.ID)
			}
		})
	}
}

func keys(m map[uint32]bool) []uint32 {
	var out []uint32
	for k := range m {
		out = append(out, k)
	}
	return out
}

func intersect(a, b map[uint32]bool) []uint32 {
	var out []uint32
	for k := range a {
		if b[k] {
			out = append(out, k)
		}
	}
	return out
}

// grouped reports whether the run's starts are grouped by task: every task's
// starts contiguous, and the tasks in index order.
func grouped(evs []startEvent) bool {
	last := -1
	seen := map[int]bool{}
	for _, ev := range evs {
		if ev.task == last {
			continue
		}
		if seen[ev.task] {
			return false
		}
		seen[ev.task] = true
		last = ev.task
	}
	return true
}

// TestStartOrderInterleavesUnderCountedPolicy is the timing half of the S1
// overlap gate, and it is reported with a denominator because it is an
// observation and not a theorem.
//
// The chain control is a theorem: with every task depending on the one before
// it, no worker can start task i+1 until every worker has published task i, so
// the starts are grouped by task in every run. Zero ungrouped runs out of N
// there is a fact about the program.
//
// The counted arm is not. Whether two independent tasks running on disjoint
// workers actually have their starts interleave is a race, and a run in which
// one rank's starts happened to be contiguous is a legal schedule rather than
// a defect. So the assertion is that interleaving is observed at all -- which
// is what distinguishes a policy that can overlap tasks from one that cannot
// -- and the count is logged so a run that observed it once in N is not read
// as the same evidence as one that observed it every time.
//
// This is narrower than "the counted worker's start order interleaves and the
// wave worker's does not", which is not true as stated: under a wave, two
// independent tasks of one rank also overlap, because each worker reaches the
// second without waiting for any other worker to finish the first. Policy is
// not what decides that; graph shape is. What the policy decides is which
// workers a task runs on, which is the deterministic half above.
func TestStartOrderInterleavesUnderCountedPolicy(t *testing.T) {
	const trials, ctas = 64, 4
	in := input()

	cross, crossOps := crossGraph(t)
	interleaved := 0
	for n := 0; n < trials; n++ {
		p, e := buildPolicy(t, cross, crossOps, ctas, 32, cudamega.CountedPolicy)
		got, err := e.Run(p, in, 1, cudamega.AbortNone)
		if err != nil {
			t.Fatal(err)
		}
		if !grouped(events(t, got.StartOrder, len(p.Steps()), ctas)) {
			interleaved++
		}
	}
	t.Logf("counted policy: %d of %d runs started tasks out of task order", interleaved, trials)
	if interleaved == 0 {
		t.Errorf("no run out of %d interleaved two tasks' starts; the counted policy is not overlapping anything", trials)
	}

	chain, chainOps := chainGraph(t)
	for n := 0; n < trials; n++ {
		p, e := buildPolicy(t, chain, chainOps, ctas, 32, cudamega.WavePolicy)
		got, err := e.Run(p, in, 1, cudamega.AbortNone)
		if err != nil {
			t.Fatal(err)
		}
		evs := events(t, got.StartOrder, len(p.Steps()), ctas)
		if !grouped(evs) {
			t.Fatalf("a chained wave started tasks out of task order, which no worker can do before its dependency is published: %v", evs)
		}
	}
	t.Logf("chained wave: 0 of %d runs started tasks out of task order", trials)
}
