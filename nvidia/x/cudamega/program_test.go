package cudamega_test

import (
	"fmt"
	"slices"
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
	s, err := cudamega.ProgramSource(g, programOps(), cudamega.Grid{CTAs: 1, ThreadsPerCTA: 128}, goldenTarget)
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
	if _, err := cudamega.ProgramSource(cluster, map[string]cudamega.Operation{"x": {Kind: cudamega.Add, Scalar: 1}}, grid, goldenTarget); err == nil {
		t.Fatal("accepted unsupported scope")
	}
	if _, err := cudamega.ProgramSource(g, map[string]cudamega.Operation{"add3": ops["add3"]}, grid, goldenTarget); err == nil {
		t.Fatal("accepted incomplete operation set")
	}
	if _, err := cudamega.ProgramSource(g, ops, cudamega.Grid{CTAs: 1, ThreadsPerCTA: 100}, goldenTarget); err == nil {
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
	local, err := cudataskgraph.New([]cudataskgraph.Task{
		{ID: "produce", Rank: 0, Scope: cudataskgraph.ScopeCTA},
		{ID: "consume", Deps: []string{"produce"}, Rank: 1, Scope: cudataskgraph.ScopeCTA},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := cudamega.ProgramSource(local, ops, cudamega.Grid{CTAs: 1, ThreadsPerCTA: 32}, goldenTarget); err != nil {
		t.Fatalf("one CTA: a CTA-to-CTA edge must be accepted: %v", err)
	}
	_, err = cudamega.ProgramSource(g, ops, cudamega.Grid{CTAs: 2, ThreadsPerCTA: 32}, goldenTarget)
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

// blockAt returns the lines of the brace-delimited block opened on the line at
// index i, not including the line that opens it or the line that closes it.
//
// It counts braces rather than looking for a closing line at a fixed indent
// because the blocks the tests below examine contain nested blocks: a
// selection loop contains a task, and a task contains its element loop.
func blockAt(lines []string, i int) []string {
	depth := 0
	for k := i; k < len(lines); k++ {
		depth += strings.Count(lines[k], "{") - strings.Count(lines[k], "}")
		if depth == 0 {
			return lines[i+1 : k]
		}
	}
	return nil
}

// TestDependencySpinIsBounded checks that a dependency wait cannot spin
// forever.
//
// The bound has to be in the kernel. A harness-side watchdog cannot stop a CTA
// that will not yield: killing the host process leaves the grid resident, and
// on a device that also drives the display that costs the desktop. So the test
// asserts the bound is inside the loop a waiting CTA actually revisits --
// between the `while` and the brace that closes it -- rather than merely
// somewhere in the file, and that it raises a code distinguishable from a
// clean finish and from every other abort.
//
// The loop it asserts on is the selection loop, not a per-dependency poll
// loop: since ready-only selection there is no poll loop to bound, because a
// CTA that finds a dependency unmet does not sit on it. It reads the
// completion word once, declines to take the task, and comes back around. The
// bound moved onto the round counter with the waiting, and this test moved
// with it. The assertion is stronger for it -- one bound now covers every way
// a worker can fail to make progress, not only the one shape of wait -- and
// [TestBoundedLoopIsReachedOnEveryRound] holds it to being reached
// unconditionally rather than on some path through the body.
func TestDependencySpinIsBounded(t *testing.T) {
	f := deviceFixture(t)
	src, err := cudamega.ProgramSource(f.graph, f.ops, f.grid, f.target)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(src, "\n")

	var loops int
	for i, ln := range lines {
		if !strings.Contains(ln, "while (remaining)") {
			continue
		}
		loops++
		joined := strings.Join(blockAt(lines, i), "\n")
		if !strings.Contains(joined, "spin_limit") {
			t.Errorf("selection loop at line %d has no iteration bound in its body:\n%s", i+1, joined)
			continue
		}
		// The read the loop's progress depends on must be an acquiring load. A
		// read that observed the count without acquiring would let the reads
		// that follow it be satisfied from lines written before the producer's
		// stores, which is the defect the volatile form had.
		if !strings.Contains(joined, "ld.acquire.") {
			t.Errorf("selection loop at line %d reads the completion word without acquiring it:\n%s", i+1, joined)
		}
		if strings.Contains(joined, "volatile unsigned*)completed") {
			t.Errorf("selection loop at line %d still reads the completion word through a volatile cast:\n%s", i+1, joined)
		}
		if !strings.Contains(joined, fmt.Sprintf("%du", cudamega.AbortSpinExhausted)) {
			t.Errorf("selection loop at line %d bounds the spin but does not raise cudamega.AbortSpinExhausted:\n%s", i+1, joined)
		}
	}
	if loops != 1 {
		t.Fatalf("the device fixture emits %d bounded loops; a worker has exactly one loop to bound", loops)
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

// TestBoundedLoopIsReachedOnEveryRound checks that the bound is on the loop
// rather than inside a branch of it.
//
// A counter incremented under `if (owns_3 && !done_3)` bounds the rounds in
// which that CTA had that task left, which for a CTA waiting on a dependency
// it does not own is no rounds at all. The increment and the test of it must
// both sit directly in the loop body, at the depth the loop opens.
func TestBoundedLoopIsReachedOnEveryRound(t *testing.T) {
	f := deviceFixture(t)
	src, err := cudamega.ProgramSource(f.graph, f.ops, f.grid, f.target)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(src, "\n")
	i := slices.IndexFunc(lines, func(ln string) bool { return strings.Contains(ln, "while (remaining)") })
	if i < 0 {
		t.Fatal("no selection loop in the device fixture")
	}
	body := blockAt(lines, i)

	depth := 0
	var top []string
	for _, ln := range body {
		if depth == 0 {
			top = append(top, ln)
		}
		depth += strings.Count(ln, "{") - strings.Count(ln, "}")
	}
	joined := strings.Join(top, "\n")
	for _, want := range []string{"++spins;", "spins > spin_limit"} {
		if !strings.Contains(joined, want) {
			t.Errorf("%q is not at the top level of the selection loop, so a round that takes no task does not reach it:\n%s", want, joined)
		}
	}
}

// TestTakingATaskIsNotWaitingForIt is the S1.2 gate on emitted text.
//
// Pop-then-block is the failure it rules out: a CTA that takes a task and then
// waits inside it holds that task while it waits, and when every resident CTA
// holds a consumer there is no CTA left to run the producer. The shape that
// cannot do that is the one asserted here -- a task's body, from the brace
// that admits the CTA to the brace that retires the task, contains no wait at
// all. All the waiting is outside it, in the round the CTA comes back around
// on, where holding nothing is what lets the producer run.
//
// The element loop is the one loop a task body may contain, and it is bounded
// by n.
func TestTakingATaskIsNotWaitingForIt(t *testing.T) {
	for _, f := range goldenFixtures(t) {
		t.Run(f.name, func(t *testing.T) {
			src, err := cudamega.ProgramSource(f.graph, f.ops, f.grid, f.target)
			if err != nil {
				t.Fatal(err)
			}
			lines := strings.Split(src, "\n")
			bodies := 0
			for i, ln := range lines {
				trimmed := strings.TrimSpace(ln)
				taken := strings.HasPrefix(trimmed, "if (take_") ||
					strings.HasPrefix(trimmed, "{ // no dependency")
				if !taken {
					continue
				}
				bodies++
				for k, in := range blockAt(lines, i) {
					switch {
					case strings.Contains(in, "while"):
						t.Errorf("%s: line %d, inside a taken task, is a wait: %s", f.name, i+k+2, strings.TrimSpace(in))
					case strings.Contains(in, "ld.acquire."):
						t.Errorf("%s: line %d, inside a taken task, reads a dependency: %s", f.name, i+k+2, strings.TrimSpace(in))
					case strings.Contains(in, "spin"):
						t.Errorf("%s: line %d, inside a taken task, counts a spin: %s", f.name, i+k+2, strings.TrimSpace(in))
					case strings.Contains(in, "for ") && !strings.Contains(in, "for (size_t e ="):
						t.Errorf("%s: line %d, inside a taken task, is a loop that is not the element loop: %s", f.name, i+k+2, strings.TrimSpace(in))
					}
				}
			}
			if want := len(f.graph.Tasks()); bodies != want {
				t.Fatalf("%s: found %d task bodies for %d tasks; the assertion above ran over the wrong text", f.name, bodies, want)
			}
		})
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
	src, err := cudamega.ProgramSource(g, ops, cudamega.Grid{CTAs: 2, ThreadsPerCTA: 128}, goldenTarget)
	if err != nil {
		t.Fatal(err)
	}
	for _, unwanted := range []string{"spins", "spin_limit"} {
		if strings.Contains(src, unwanted) {
			t.Errorf("a dependency-free task emitted %q, which nothing in it can reach:\n%s", unwanted, src)
		}
	}
}
