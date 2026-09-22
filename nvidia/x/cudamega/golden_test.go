package cudamega_test

import (
	"flag"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

// goldenArch is the arch-specific target the goldens under testdata/golden
// were generated for. It is recorded here, not inferred: an artifact
// generated for one target says nothing about another, and sm_121 is the
// family rather than this target.
const goldenArch = "sm_121a"

// goldenTarget is goldenArch as the emitter's own type. The two spellings are
// kept apart deliberately: goldenArch is what the toolchain is invoked with
// and what the artifacts announce, goldenTarget is what the program is built
// for, and a test that used one value for both could not notice them
// disagreeing.
const goldenTarget = cudamega.Target(goldenArch)

// ptxTargets are the architectures a PTX golden is committed for. The emitted
// text is the same for all three -- see TestEmittedTextIsTargetInvariant --
// so only the PTX is per-target.
var ptxTargets = []cudamega.Target{cudamega.SM80, cudamega.SM90, cudamega.SM121a}

var updateGolden = flag.Bool("update-golden", false, "rewrite the generated .cu goldens under testdata/golden")

// A programFixture is one graph, its operations, and the grid its source was
// generated for. The grid is part of the fixture because the generated source
// refuses at runtime to run under any other geometry.
type programFixture struct {
	name  string
	graph cudataskgraph.Graph
	ops   map[string]cudamega.Operation
	grid  cudamega.Grid
	// target is part of the fixture for the same reason the grid is: a
	// program is built for one architecture, and an artifact generated for
	// one target is no evidence about another.
	target cudamega.Target
}

// ctaFixture exercises the CTA synchronization domain: a dependency edge
// carried by the shared mbarrier, which is only orderable because the grid is
// one CTA.
func ctaFixture(t *testing.T) programFixture {
	t.Helper()
	g, err := cudataskgraph.New([]cudataskgraph.Task{
		{ID: "add3", Rank: 0, Bytes: 16, Scope: cudataskgraph.ScopeCTA},
		{ID: "multiply2", Deps: []string{"add3"}, Rank: 1, Bytes: 16, Scope: cudataskgraph.ScopeCTA},
	})
	if err != nil {
		t.Fatal(err)
	}
	return programFixture{
		name:  "cta-1cta",
		graph: g,
		ops: map[string]cudamega.Operation{
			"add3":      {Kind: cudamega.Add, Scalar: 3},
			"multiply2": {Kind: cudamega.Multiply, Scalar: 2},
		},
		grid:   cudamega.Grid{CTAs: 1, ThreadsPerCTA: 128},
		target: goldenTarget,
	}
}

// deviceFixture exercises the device and system domains across two CTAs, plus
// one isolated CTA-scope task. The isolated task is what makes the CTA
// arrival appear in a multi-CTA artifact at all: a CTA-scope task on either
// end of a dependency edge is refused in this grid.
func deviceFixture(t *testing.T) programFixture {
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
	return programFixture{
		name:  "device-2cta",
		graph: g,
		ops: map[string]cudamega.Operation{
			"local":  {Kind: cudamega.Add, Scalar: 11},
			"seed":   {Kind: cudamega.Add, Scalar: 3},
			"scale":  {Kind: cudamega.Multiply, Scalar: 2},
			"offset": {Kind: cudamega.Add, Scalar: 7},
		},
		grid:   cudamega.Grid{CTAs: 2, ThreadsPerCTA: 128},
		target: goldenTarget,
	}
}

// crossFixture is the S1 demonstrator: every element each consumer reads was
// written by the other worker.
//
// It exists because "matches the ordinary-launch reference" is blind to the
// dependency wait in the other two fixtures. There the emitted worker gives
// element e to the same (cta, tid) in every task, so the element a consumer
// reads was written by the thread reading it, and deleting the wait leaves
// the comparison green. Here the grid is partitioned by rank -- "lower" and
// "cross" on CTA 0, "upper" and "swap" on CTA 1 -- and the two rank-1 tasks
// depend across the partition: "cross" reads what CTA 1 wrote and "swap"
// reads what CTA 0 wrote. The owner ranges are disjoint from their
// dependencies' for every element, not for some of them, which
// [TestDemonstratorDependsAcrossWorkers] states as an assertion rather than as
// this comment.
//
// The scopes are deliberately mixed. "swap" is at system scope over a
// device-scope producer, so its edge and its producer's release are ordered
// at system scope by the max(producer, consumer) rule and by nothing the
// consumer declares alone.
func crossFixture(t *testing.T) programFixture {
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
	return programFixture{
		name:  "cross-2cta",
		graph: g,
		ops: map[string]cudamega.Operation{
			"lower": {Kind: cudamega.Add, Scalar: 3},
			"upper": {Kind: cudamega.Add, Scalar: 5},
			"cross": {Kind: cudamega.Multiply, Scalar: 2},
			"swap":  {Kind: cudamega.Add, Scalar: 7},
		},
		grid:   cudamega.Grid{CTAs: 2, ThreadsPerCTA: 128, Policy: cudamega.CountedPolicy},
		target: goldenTarget,
	}
}

// skewFixture is crossFixture with the deal deliberately unbalanced, so that
// a consumer reaches its edge before its producer has written anything.
//
// Rank 0 is three tasks over four CTAs, which CountedPolicy splits as [0,1),
// [1,2) and [2,4): "pair" has two owners and each of them does half as many
// elements as "base"'s single owner does. CTAs 2 and 3 therefore finish rank 0
// in half the time CTA 0 takes, and "tail" -- which they own at rank 1 --
// arrives at its edge with "base" only half written. crossFixture's deal is
// balanced, so there its consumer reaches the edge exactly when its producer
// leaves it; here the consumer is there first by construction.
//
// Both rank-1 owner ranges are disjoint from their dependencies', as in
// crossFixture, so the comparison against the ordinary-launch reference is
// not blind for either edge.
//
// It is a witness and not a gate. On sm_121a, with the emitted dependency
// wait made a no-op, this fixture agreed with the reference in 3 of 3 runs at
// n = 1048576 -- the same count crossFixture returns, 0 of 5. The skew makes
// the consumer reach its edge first, but it does not make the race reachable:
// producer and consumer sweep elements in the same increasing order at
// comparable rates, so the read index trails the write index for the whole
// task and a consumer that starts late cannot overtake. What deterministically
// catches a wait that does not wait is the CPU backend's skewed worker, which
// delays the producer's Compute rather than the consumer's start, and what
// catches a wait counted against the wrong population is an unsatisfiable
// wait: emitting "published < grid_ctas" aborts crossFixture on the device
// with code 4.
func skewFixture(t *testing.T) programFixture {
	t.Helper()
	g, err := cudataskgraph.New([]cudataskgraph.Task{
		{ID: "base", Rank: 0, Bytes: 16, Scope: cudataskgraph.ScopeDevice},
		{ID: "extra", Rank: 0, Bytes: 16, Scope: cudataskgraph.ScopeDevice},
		{ID: "pair", Rank: 0, Bytes: 16, Scope: cudataskgraph.ScopeDevice},
		{ID: "mirror", Deps: []string{"pair"}, Rank: 1, Bytes: 16, Scope: cudataskgraph.ScopeDevice},
		{ID: "tail", Deps: []string{"base"}, Rank: 1, Bytes: 16, Scope: cudataskgraph.ScopeDevice},
	})
	if err != nil {
		t.Fatal(err)
	}
	return programFixture{
		name:  "skew-4cta",
		graph: g,
		ops: map[string]cudamega.Operation{
			"base":   {Kind: cudamega.Add, Scalar: 3},
			"extra":  {Kind: cudamega.Add, Scalar: 5},
			"pair":   {Kind: cudamega.Add, Scalar: 7},
			"mirror": {Kind: cudamega.Multiply, Scalar: 2},
			"tail":   {Kind: cudamega.Multiply, Scalar: 3},
		},
		grid:   cudamega.Grid{CTAs: 4, ThreadsPerCTA: 128, Policy: cudamega.CountedPolicy},
		target: goldenTarget,
	}
}

func goldenFixtures(t *testing.T) []programFixture {
	return []programFixture{ctaFixture(t), deviceFixture(t), crossFixture(t)}
}

// TestDemonstratorDependsAcrossWorkers is the precondition every other gate
// over crossFixture rests on.
//
// A fixture whose consumer and producer share an owner range gives element e
// to the same CTA in both tasks, and then no comparison of the result -- with
// the ordinary-launch reference or with anything else -- can tell a run that
// waited from a run that did not. This test states the property the fixture
// must have for those comparisons to mean anything: every consumer's owner
// range is disjoint from its dependency's, so every element it reads crossed
// a worker boundary.
func TestDemonstratorDependsAcrossWorkers(t *testing.T) {
	f := crossFixture(t)
	p, err := cudamega.NewProgram(f.graph, f.ops, f.grid, f.target)
	if err != nil {
		t.Fatal(err)
	}
	steps := p.Steps()
	edges := 0
	for _, s := range steps {
		for _, d := range s.Deps {
			edges++
			c, q := s.Owner, steps[d].Owner
			if c.Lo < q.Hi && q.Lo < c.Hi {
				t.Errorf("task %q owns %s and its dependency %q owns %s; the ranges overlap, so some element is read by the CTA that wrote it",
					s.Task.ID, c, steps[d].Task.ID, q)
			}
		}
	}
	if edges == 0 {
		t.Fatal("the demonstrator has no dependency edge; there is nothing for a wait to be load-bearing about")
	}
}

// TestGoldenSourceMatchesGenerator keeps the committed .cu in step with the
// generator. Without it the PTX golden would be evidence about a source file
// nothing in this package still emits.
func TestGoldenSourceMatchesGenerator(t *testing.T) {
	for _, f := range goldenFixtures(t) {
		src, err := cudamega.ProgramSource(f.graph, f.ops, f.grid, f.target)
		if err != nil {
			t.Fatalf("%s: %v", f.name, err)
		}
		path := filepath.Join("testdata", "golden", f.name+".cu")
		if *updateGolden {
			if err := os.WriteFile(path, []byte(src), 0o644); err != nil {
				t.Fatal(err)
			}
			continue
		}
		want, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if string(want) != src {
			t.Fatalf("%s: generator and committed golden disagree; regenerate with -update-golden and rebuild the PTX", path)
		}
	}
}
