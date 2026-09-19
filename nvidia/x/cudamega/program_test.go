package cudamega_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

func programGraph(t *testing.T, bytes uint64) cudataskgraph.Graph {
	t.Helper()
	g, err := cudataskgraph.New([]cudataskgraph.Task{
		{ID: "add3", Rank: 0, Bytes: bytes, Scope: cudataskgraph.ScopeCTA},
		{ID: "multiply2", Deps: []string{"add3"}, Rank: 1, Bytes: bytes, Scope: cudataskgraph.ScopeCTA},
	})
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func programOps() map[string]cudamega.Operation {
	return map[string]cudamega.Operation{
		"add3":      {Kind: cudamega.Add, Scalar: 3},
		"multiply2": {Kind: cudamega.Multiply, Scalar: 2},
	}
}

func TestProgramSourceFiniteNumericalFixture(t *testing.T) {
	g := programGraph(t, 16)
	s, err := cudamega.ProgramSource(g, programOps(), cudamega.Grid{CTAs: 1, ThreadsPerCTA: 128})
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"dag_worker", "ordinary_task", "x + 3u", "x * 2u"} {
		if !strings.Contains(s, want) {
			t.Fatalf("generated source lacks %q", want)
		}
	}
	// The host oracle matches the native fixture's two operations, including
	// sentinel values and uint32 wraparound.
	input := []uint32{0, 1, 0xffffffff}
	want := []uint32{6, 8, 4}
	got := make([]uint32, len(input))
	for i, v := range input {
		got[i] = (v + 3) * 2
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("index %d: got %d want %d", i, got[i], want[i])
		}
	}
}

func TestSchedulerResourceAndCloneContracts(t *testing.T) {
	g := programGraph(t, 32)
	s, err := cudamega.NewScheduler(g, cudamega.Admission{ResidentCTAs: 1, WorkerThreads: 128, SharedBytes: 64})
	if err != nil {
		t.Fatal(err)
	}
	if len(s.Graph().Tasks()) != 2 {
		t.Fatal("scheduler lost tasks")
	}
	waves, err := s.Assign()
	if err != nil {
		t.Fatal(err)
	}
	if len(waves) != 2 || waves[0].Task != "add3" || waves[1].Task != "multiply2" {
		t.Fatalf("assignment is not the unrolled task order: %+v", waves)
	}
	if _, err := cudamega.NewScheduler(g, cudamega.Admission{ResidentCTAs: 1, WorkerThreads: 128, SharedBytes: 16}); err == nil {
		t.Fatal("accepted task beyond shared-memory admission")
	}
}

func TestProgramSourceFailsClosed(t *testing.T) {
	g := programGraph(t, 1)
	ops := programOps()
	grid := cudamega.Grid{CTAs: 1, ThreadsPerCTA: 128}
	cluster, err := cudataskgraph.New([]cudataskgraph.Task{{ID: "x", Rank: 0, Scope: cudataskgraph.ScopeCluster}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := cudamega.ProgramSource(cluster, map[string]cudamega.Operation{"x": {Kind: cudamega.Add, Scalar: 1}}, grid); err == nil {
		t.Fatal("accepted unsupported scope")
	}
	if _, err := cudamega.ProgramSource(g, map[string]cudamega.Operation{"add3": ops["add3"]}, grid); err == nil {
		t.Fatal("accepted incomplete operation set")
	}
	if _, err := cudamega.ProgramSource(g, ops, cudamega.Grid{CTAs: 1, ThreadsPerCTA: 100}); err == nil {
		t.Fatal("accepted a block that is not a whole number of warps")
	}
}

// TestCTAScopeRefusedOnBothEndsOfACrossCTAEdge fixes the asymmetry that the
// first version of the check had: it refused a CTA-scope consumer and
// accepted the CTA-scope producer feeding it, even though the producer's
// mbarrier release is exactly as unable to reach another CTA as the
// consumer's wait is. The rejection must also name the side it found, because
// a message that always says "depends on" would describe the producer case
// wrongly while still failing the test that only counts errors.
func TestCTAScopeRefusedOnBothEndsOfACrossCTAEdge(t *testing.T) {
	g, err := cudataskgraph.New([]cudataskgraph.Task{
		{ID: "produce", Rank: 0, Scope: cudataskgraph.ScopeCTA},
		{ID: "consume", Deps: []string{"produce"}, Rank: 1, Scope: cudataskgraph.ScopeDevice},
	})
	if err != nil {
		t.Fatal(err)
	}
	ops := map[string]cudamega.Operation{
		"produce": {Kind: cudamega.Add, Scalar: 1},
		"consume": {Kind: cudamega.Multiply, Scalar: 2},
	}
	if _, err := cudamega.ProgramSource(g, ops, cudamega.Grid{CTAs: 1, ThreadsPerCTA: 32}); err != nil {
		t.Fatalf("one CTA: a CTA-scope edge is orderable and must be accepted: %v", err)
	}
	_, err = cudamega.ProgramSource(g, ops, cudamega.Grid{CTAs: 2, ThreadsPerCTA: 32})
	if err == nil {
		t.Fatal("two CTAs: accepted a CTA-scope producer whose consumer runs in another CTA")
	}
	if !strings.Contains(err.Error(), `"produce" is depended on by another task`) {
		t.Fatalf("rejection does not name the producer side: %v", err)
	}
	if strings.Contains(err.Error(), "depends on another task") {
		t.Fatalf("rejection names the consumer side, which is not the task at fault: %v", err)
	}
}

// TestDependencySpinIsBounded checks that a dependency wait cannot spin
// forever.
//
// The bound has to be in the kernel. A harness-side watchdog cannot stop a CTA
// that will not yield: killing the host process leaves the grid resident, and
// on a device that also drives the display that costs the desktop. So the test
// asserts the bound is inside the poll loop -- between the `while` and the
// brace that closes it -- rather than merely somewhere in the file, and that it
// raises a code distinguishable from a clean finish and from every other abort.
func TestDependencySpinIsBounded(t *testing.T) {
	f := deviceFixture(t)
	src, err := cudamega.ProgramSource(f.graph, f.ops, f.grid)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(src, "\n")

	var loops int
	for i, ln := range lines {
		if !strings.Contains(ln, "while (((volatile unsigned*)completed)") {
			continue
		}
		loops++
		var body []string
		for _, in := range lines[i+1:] {
			if strings.TrimSpace(in) == "}" {
				break
			}
			body = append(body, in)
		}
		joined := strings.Join(body, "\n")
		if !strings.Contains(joined, "spin_limit") {
			t.Errorf("poll loop at line %d has no iteration bound in its body:\n%s", i+1, joined)
			continue
		}
		if !strings.Contains(joined, fmt.Sprintf("%du", cudamega.AbortSpinExhausted)) {
			t.Errorf("poll loop at line %d bounds the spin but does not raise cudamega.AbortSpinExhausted:\n%s", i+1, joined)
		}
	}
	if loops == 0 {
		t.Fatal("no dependency poll loop in the device fixture; the bound has nothing to guard")
	}

	// A code the host cannot tell apart from a clean finish is not a
	// distinguishable status.
	for name, code := range map[string]int{
		"cudamega.AbortNone":        cudamega.AbortNone,
		"cudamega.AbortHostRequest": cudamega.AbortHostRequest,
		"cudamega.AbortBadLaunch":   cudamega.AbortBadLaunch,
		"cudamega.AbortBadExtent":   cudamega.AbortBadExtent,
	} {
		if code == cudamega.AbortSpinExhausted {
			t.Errorf("cudamega.AbortSpinExhausted collides with %s (%d)", name, code)
		}
	}
}

// TestIndependentTaskHasNoSpinToBound is the control for
// [TestDependencySpinIsBounded]: a task with no dependency must emit no poll
// loop, so a future change that made the bound unconditional -- costing every
// task a counter it never reads -- shows up as a failure here rather than as
// nothing at all.
func TestIndependentTaskHasNoSpinToBound(t *testing.T) {
	g, err := cudataskgraph.New([]cudataskgraph.Task{
		{ID: "solo", Rank: 0, Bytes: 16, Scope: cudataskgraph.ScopeDevice},
	})
	if err != nil {
		t.Fatal(err)
	}
	ops := map[string]cudamega.Operation{"solo": {Kind: cudamega.Add, Scalar: 1}}
	src, err := cudamega.ProgramSource(g, ops, cudamega.Grid{CTAs: 2, ThreadsPerCTA: 128})
	if err != nil {
		t.Fatal(err)
	}
	for _, unwanted := range []string{"spins", "spin_limit"} {
		if strings.Contains(src, unwanted) {
			t.Errorf("a dependency-free task emitted %q, which nothing in it can reach:\n%s", unwanted, src)
		}
	}
}
