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

func goldenFixtures(t *testing.T) []programFixture {
	return []programFixture{ctaFixture(t), deviceFixture(t)}
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
