package cudamegacpu_test

import (
	"fmt"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/cudamegacpu"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

// The demonstrator, restated here.
//
// These are the two fixtures cudamega's goldens are generated from: a
// one-CTA graph whose dependency edge is carried at CTA scope, and a
// multi-CTA graph carrying device and system scope plus one isolated
// CTA-scope task. They are restated rather than imported because they live in
// cudamega's external test package, and a copy that drifted from the original
// would fail cudamega's own golden test first.
func ctaGraph(t *testing.T) (cudataskgraph.Graph, map[string]cudamega.Operation) {
	t.Helper()
	g, err := cudataskgraph.New([]cudataskgraph.Task{
		{ID: "add3", Rank: 0, Bytes: 16, Scope: cudataskgraph.ScopeCTA},
		{ID: "multiply2", Deps: []string{"add3"}, Rank: 1, Bytes: 16, Scope: cudataskgraph.ScopeCTA},
	})
	if err != nil {
		t.Fatal(err)
	}
	return g, map[string]cudamega.Operation{
		"add3":      {Kind: cudamega.Add, Scalar: 3},
		"multiply2": {Kind: cudamega.Multiply, Scalar: 2},
	}
}

func deviceGraph(t *testing.T) (cudataskgraph.Graph, map[string]cudamega.Operation) {
	t.Helper()
	g, err := cudataskgraph.New([]cudataskgraph.Task{
		{ID: "local", Rank: 0, Bytes: 16, Scope: cudataskgraph.ScopeCTA},
		{ID: "seed", Rank: 0, Bytes: 16, Scope: cudataskgraph.ScopeDevice},
		{ID: "scale", Deps: []string{"seed"}, Rank: 1, Bytes: 16, Scope: cudataskgraph.ScopeDevice},
		{ID: "offset", Deps: []string{"scale"}, Rank: 2, Bytes: 16, Scope: cudataskgraph.ScopeSystem},
	})
	if err != nil {
		t.Fatal(err)
	}
	return g, map[string]cudamega.Operation{
		"local":  {Kind: cudamega.Add, Scalar: 11},
		"seed":   {Kind: cudamega.Add, Scalar: 3},
		"scale":  {Kind: cudamega.Multiply, Scalar: 2},
		"offset": {Kind: cudamega.Add, Scalar: 7},
	}
}

// input is 200 elements including the values that wrap under both
// operations.
//
// The count is deliberately not a multiple of any grid width the tests use,
// and is larger than the widest of them. At 32 elements and 32 threads per
// worker every thread owns exactly one element whatever the grid, so a
// backend that dropped the worker index out of its element stride computed
// the right answer anyway: every worker wrote every element, with the same
// value. A mapping the comparison cannot see is a mapping the comparison is
// not checking.
func input() []uint32 {
	in := make([]uint32, 200)
	for i := range in {
		in[i] = uint32(i)
	}
	in[198] = 0xffffffff
	in[199] = 0xfffffffd
	return in
}

// build returns a program and an executor admitting exactly its grid.
func build(t *testing.T, g cudataskgraph.Graph, ops map[string]cudamega.Operation, ctas, threads uint32) (*cudamega.Program, *cudamegacpu.Executor) {
	t.Helper()
	// The target is still an input with no device in sight, which is the
	// point of taking it at construction: the CPU backend runs a program
	// built for sm_121a without ever being one.
	p, err := cudamega.NewProgram(g, ops, cudamega.Grid{CTAs: ctas, ThreadsPerCTA: threads}, cudamega.SM121a)
	if err != nil {
		t.Fatal(err)
	}
	e, err := cudamegacpu.New(cudamega.Admission{ResidentCTAs: ctas, WorkerThreads: threads, SharedBytes: 64})
	if err != nil {
		t.Fatal(err)
	}
	return p, e
}

// TestCPUBackendMatchesReference runs both demonstrator graphs at several
// widths and requires the result to equal the ordinary-launch reference.
//
// The widths are the control that this schedules rather than happens to work:
// one worker, as many workers as tasks, and more workers than tasks. A model
// correct only at one width has not been shown to schedule anything, and a
// model whose extra workers double-counted or skipped elements shows up here
// and nowhere else.
func TestCPUBackendMatchesReference(t *testing.T) {
	in := input()
	cta, ctaOps := ctaGraph(t)
	dev, devOps := deviceGraph(t)
	for _, tc := range []struct {
		name    string
		graph   cudataskgraph.Graph
		ops     map[string]cudamega.Operation
		ctas    uint32
		threads uint32
	}{
		// The CTA-scope graph is a one-worker graph by construction: an
		// mbarrier orders one CTA, so cudamega refuses either end of its
		// dependency edge in a wider grid.
		{"cta-1cta", cta, ctaOps, 1, 32},
		{"cta-1cta wide threads", cta, ctaOps, 1, 128},
		{"device 1 worker", dev, devOps, 1, 32},
		{"device 2 workers", dev, devOps, 2, 32},
		{"device 4 workers", dev, devOps, 4, 32},
		{"device 8 workers, 4 tasks", dev, devOps, 8, 32},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, e := build(t, tc.graph, tc.ops, tc.ctas, tc.threads)
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

// TestReferenceIsSensitiveToDependencyOrder is the mutant control for the
// comparison above.
//
// Two arms that agree have shown nothing unless disagreement was possible.
// Removing the dependency edge changes what the consumer reads -- the program
// input instead of the producer's output -- so both arms must move, and move
// to the same new answer. A comparison that passed before and after this
// change would be a comparison sensitive to the inputs and blind to the
// ordering, which is the only thing this model is about.
func TestReferenceIsSensitiveToDependencyOrder(t *testing.T) {
	in := input()
	g, ops := deviceGraph(t)
	p, e := build(t, g, ops, 2, 32)
	linked, err := e.Run(p, in, 1, cudamega.AbortNone)
	if err != nil {
		t.Fatal(err)
	}

	// The same tasks with "scale"'s dependency removed. Its rank is kept so
	// the graph is otherwise identical.
	cut, err := cudataskgraph.New([]cudataskgraph.Task{
		{ID: "local", Rank: 0, Bytes: 16, Scope: cudataskgraph.ScopeCTA},
		{ID: "seed", Rank: 0, Bytes: 16, Scope: cudataskgraph.ScopeDevice},
		{ID: "scale", Rank: 1, Bytes: 16, Scope: cudataskgraph.ScopeDevice},
		{ID: "offset", Deps: []string{"scale"}, Rank: 2, Bytes: 16, Scope: cudataskgraph.ScopeSystem},
	})
	if err != nil {
		t.Fatal(err)
	}
	q, e2 := build(t, cut, ops, 2, 32)
	unlinked, err := e2.Run(q, in, 1, cudamega.AbortNone)
	if err != nil {
		t.Fatal(err)
	}
	if reflect.DeepEqual(linked.Scratch, unlinked.Scratch) {
		t.Fatal("removing a dependency edge changed nothing; the comparison cannot see ordering")
	}
	want, err := cudamegacpu.Reference(q, in)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(unlinked.Scratch, want) {
		t.Fatalf("with the edge cut the backend and the reference disagree\n got %v\nwant %v", unlinked.Scratch, want)
	}
}

// TestCPUBackendRefusals checks the refusals the emitted worker's entry makes
// and the ones binding makes, on the backend that can be made to make them
// without a device.
func TestCPUBackendRefusals(t *testing.T) {
	in := input()
	g, ops := deviceGraph(t)

	t.Run("geometry", func(t *testing.T) {
		// A program generated for two workers, run by an executor of four.
		// Binding admits it -- four workers can host two -- and the worker
		// entry refuses it, which is the CPU counterpart of the emitted
		// gridDim check.
		p, _ := build(t, g, ops, 2, 32)
		e, err := cudamegacpu.New(cudamega.Admission{ResidentCTAs: 4, WorkerThreads: 32, SharedBytes: 64})
		if err != nil {
			t.Fatal(err)
		}
		_, err = e.Run(p, in, 1, cudamega.AbortNone)
		if err == nil || !strings.Contains(err.Error(), fmt.Sprintf("abort code %d", cudamega.AbortBadLaunch)) {
			t.Fatalf("ran under a geometry the program was not generated for: %v", err)
		}
	})

	t.Run("epoch zero", func(t *testing.T) {
		p, e := build(t, g, ops, 2, 32)
		_, err := e.Run(p, in, 0, cudamega.AbortNone)
		if err == nil || !strings.Contains(err.Error(), fmt.Sprintf("abort code %d", cudamega.AbortBadExtent)) {
			t.Fatalf("ran at the unpublished epoch: %v", err)
		}
	})

	t.Run("narrower executor", func(t *testing.T) {
		p, _ := build(t, g, ops, 4, 32)
		e, err := cudamegacpu.New(cudamega.Admission{ResidentCTAs: 2, WorkerThreads: 32, SharedBytes: 64})
		if err != nil {
			t.Fatal(err)
		}
		_, err = e.Run(p, in, 1, cudamega.AbortNone)
		if err == nil || !strings.Contains(err.Error(), "admits 2 workers, the program's grid is 4") {
			t.Fatalf("bound a program wider than the backend admits: %v", err)
		}
	})
}

// TestAbortedRunWritesNothing is the CPU counterpart of the aborted-launch
// check the native harness makes: a run that starts aborted must leave the
// output storage alone.
func TestAbortedRunWritesNothing(t *testing.T) {
	in := input()
	g, ops := deviceGraph(t)
	p, e := build(t, g, ops, 2, 32)
	got, err := e.Run(p, in, 1, cudamega.AbortHostRequest)
	if err != nil {
		t.Fatal(err)
	}
	if got.Abort != cudamega.AbortHostRequest {
		t.Fatalf("abort code %d, want %d", got.Abort, cudamega.AbortHostRequest)
	}
	for i, v := range got.Scratch {
		if v != 0 {
			t.Fatalf("an aborted run wrote %d at word %d", v, i)
		}
	}
	// The control: the same program not aborted writes something, so the
	// check above is not passing because nothing ever writes.
	live, err := e.Run(p, in, 1, cudamega.AbortNone)
	if err != nil {
		t.Fatal(err)
	}
	nonzero := false
	for _, v := range live.Scratch {
		if v != 0 {
			nonzero = true
			break
		}
	}
	if !nonzero {
		t.Fatal("an unaborted run also wrote nothing; the abort check is vacuous")
	}
}

// TestEveryPrimitiveHasTwoImplementations is the falsifier, stated as a test.
//
// A primitive only one backend implements means the boundary is in the wrong
// place: it is a CUDA idiom wearing an interface. The check is that both
// backends are cudamega.Backend -- which the compiler settles -- and that
// each one, driven over a program using every scope this package supports and
// every shape of task, completes without reporting any primitive unsupported.
// A backend that satisfied the interface by returning an error from one
// method would pass the compiler and fail here.
func TestEveryPrimitiveHasTwoImplementations(t *testing.T) {
	in := input()
	g, ops := deviceGraph(t)
	p, e := build(t, g, ops, 2, 32)

	// The rendering backend.
	src, err := p.Source()
	if err != nil {
		t.Fatalf("the CUDA backend could not render a primitive: %v", err)
	}
	if !strings.Contains(src, "dag_worker") {
		t.Fatal("the CUDA backend rendered no worker")
	}

	// The performing backend, over the same program.
	if _, err := e.Run(p, in, 1, cudamega.AbortNone); err != nil {
		t.Fatalf("the CPU backend could not perform a primitive: %v", err)
	}

	// And the interface itself has no method beyond the ones the driver
	// invokes, which cudamega's TestDriverExercisesEveryPrimitive settles.
	// Restated here as a count so that a method added to Backend without a
	// driver call fails in both packages rather than only in one.
	iface := reflect.TypeOf((*cudamega.Backend)(nil)).Elem()
	var names []string
	for i := 0; i < iface.NumMethod(); i++ {
		names = append(names, iface.Method(i).Name)
	}
	sort.Strings(names)
	want := []string{"Barrier", "Begin", "BeginSelection", "BeginTask", "Compute", "End", "EndSelection", "EndTask", "Fence", "ForwardProgress", "NextRound", "Publish", "Ready", "Residency", "Take"}
	if !reflect.DeepEqual(names, want) {
		t.Fatalf("cudamega.Backend has methods %v, want %v; a new primitive needs an implementation in every backend and a call in the driver", names, want)
	}
}

// TestForwardProgressGuaranteeIsDeclared records what this backend claims and
// why it is true.
func TestForwardProgressGuaranteeIsDeclared(t *testing.T) {
	g, ops := deviceGraph(t)
	p, e := build(t, g, ops, 2, 32)
	if _, err := e.Run(p, input(), 1, cudamega.AbortNone); err != nil {
		t.Fatal(err)
	}
	if got := e.Admission().ResidentCTAs; got != 2 {
		t.Fatalf("residency query reports %d workers, want the 2 configured", got)
	}
}

func Example() {
	g, _ := cudataskgraph.New([]cudataskgraph.Task{
		{ID: "add1", Rank: 0, Bytes: 4, Scope: cudataskgraph.ScopeDevice},
		{ID: "double", Deps: []string{"add1"}, Rank: 1, Bytes: 4, Scope: cudataskgraph.ScopeDevice},
	})
	ops := map[string]cudamega.Operation{
		"add1":   {Kind: cudamega.Add, Scalar: 1},
		"double": {Kind: cudamega.Multiply, Scalar: 2},
	}
	p, err := cudamega.NewProgram(g, ops, cudamega.Grid{CTAs: 2, ThreadsPerCTA: 32}, cudamega.SM80)
	if err != nil {
		fmt.Println(err)
		return
	}
	e, err := cudamegacpu.New(cudamega.Admission{ResidentCTAs: 2, WorkerThreads: 32, SharedBytes: 64})
	if err != nil {
		fmt.Println(err)
		return
	}
	r, err := e.Run(p, []uint32{1, 2, 3, 4}, 1, cudamega.AbortNone)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(r.Scratch)
	// Output: [2 3 4 5 4 6 8 10]
}
