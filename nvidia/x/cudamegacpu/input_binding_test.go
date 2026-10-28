package cudamegacpu_test

import (
	"fmt"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/cudamegacpu"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

type orderedPair struct{}

func (orderedPair) Inputs() int { return 2 }
func (orderedPair) Source(inputs []string, output string) (string, error) {
	return fmt.Sprintf("%s = %s*7u + %s;", output, inputs[0], inputs[1]), nil
}
func (orderedPair) Evaluate(inputs []uint32) uint32 { return inputs[0]*7 + inputs[1] }

func mixedProgram(t *testing.T) *cudamega.Program {
	t.Helper()
	g, err := cudataskgraph.New([]cudataskgraph.Task{
		{ID: "a", Extent: 5, Scope: cudataskgraph.ScopeDevice},
		{ID: "b", Extent: 3, Scope: cudataskgraph.ScopeDevice},
		{ID: "c", Rank: 1, Deps: []string{"a", "b"}, Inputs: []string{"b", "a"}, Extent: 3, Scope: cudataskgraph.ScopeDevice},
		{ID: "d", Rank: 2, Deps: []string{"c"}, Inputs: []string{"c"}, Extent: 2, Scope: cudataskgraph.ScopeDevice},
		{ID: "e", Rank: 2, Deps: []string{"a", "b"}, Inputs: []string{"a"}, Extent: 4, Scope: cudataskgraph.ScopeDevice},
	})
	if err != nil {
		t.Fatal(err)
	}
	p, err := cudamega.NewProgram(g, map[string]cudamega.Operation{
		"a": {Kind: cudamega.Add, Scalar: 1},
		"b": {Kind: cudamega.Multiply, Scalar: 3},
		"c": {Body: orderedPair{}},
		"d": {Kind: cudamega.Multiply, Scalar: 2},
		"e": {Kind: cudamega.Add, Scalar: 10},
	}, cudamega.Grid{CTAs: 2, ThreadsPerCTA: 32, MaxElements: 5, Policy: cudamega.CountedPolicy}, cudamega.SM80)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestMixedInputExecution(t *testing.T) {
	p := mixedProgram(t)
	input := []uint32{1, 2, 3, 4, 5}
	// Independent arithmetic, including the second producer, operand order,
	// each ragged tail and an ordering-only edge to b for e.
	want := []uint32{2, 3, 4, 5, 6, 3, 6, 9, 23, 45, 67, 46, 90, 12, 13, 14, 15}
	layout, err := p.Storage(5)
	if err != nil || !slices.Equal(layout, []cudamega.TaskStorage{{Offset: 0, Elements: 5}, {Offset: 5, Elements: 3}, {Offset: 8, Elements: 3}, {Offset: 11, Elements: 2}, {Offset: 13, Elements: 4}}) {
		t.Fatalf("layout = %v, %v", layout, err)
	}
	inv, err := p.Invocation(5)
	if err != nil || inv.ScratchWords != len(want) {
		t.Fatalf("invocation = %+v, %v", inv, err)
	}
	ref, err := cudamegacpu.Reference(p, input)
	if err != nil || !slices.Equal(ref, want) {
		t.Fatalf("ordinary = %v, %v; want %v", ref, err, want)
	}
	e, err := cudamegacpu.New(cudamega.Admission{ResidentCTAs: 2, WorkerThreads: 32, SharedBytes: 1})
	if err != nil {
		t.Fatal(err)
	}
	got, err := e.Run(p, input, 1, 0)
	if err != nil || got.Abort != 0 || !slices.Equal(got.Scratch, want) {
		t.Fatalf("persistent = %+v, %v; want %v", got, err, want)
	}
	if _, err := e.Run(p, input[:4], 1, 0); err == nil {
		t.Fatal("short external input reached execution")
	}
	if _, err := e.Run(p, append(input, 6), 1, 0); err == nil {
		t.Fatal("extent above declared maximum reached execution")
	}
	// Layout accessors do not expose mutable program state.
	layout[0].Elements = 1
	gotLayout, _ := p.Storage(5)
	if gotLayout[0].Elements != 5 {
		t.Fatal("caller changed the stored layout")
	}
}
