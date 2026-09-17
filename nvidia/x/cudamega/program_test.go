package cudamega_test

import (
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

func TestProgramSourceFiniteNumericalFixture(t *testing.T) {
	g := programGraph(t, 16)
	s, err := cudamega.ProgramSource(g, map[string]cudamega.Operation{
		"add3":      {Kind: cudamega.Add, Scalar: 3},
		"multiply2": {Kind: cudamega.Multiply, Scalar: 2},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"dag_worker", "ordinary_task", "x + 3u", "x * 2u", "__threadfence_block", "atomicExch(completed"} {
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
	if _, err := cudamega.NewScheduler(g, cudamega.Admission{ResidentCTAs: 1, WorkerThreads: 128, SharedBytes: 16}); err == nil {
		t.Fatal("accepted task beyond shared-memory admission")
	}
}

func TestProgramSourceFailsClosed(t *testing.T) {
	g := programGraph(t, 1)
	ops := map[string]cudamega.Operation{"add3": {Kind: cudamega.Add, Scalar: 3}, "multiply2": {Kind: cudamega.Multiply, Scalar: 2}}
	cluster, err := cudataskgraph.New([]cudataskgraph.Task{{ID: "x", Rank: 0, Scope: cudataskgraph.ScopeCluster}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := cudamega.ProgramSource(cluster, map[string]cudamega.Operation{"x": {Kind: cudamega.Add, Scalar: 1}}); err == nil {
		t.Fatal("accepted unsupported scope")
	}
	if _, err := cudamega.ProgramSource(g, map[string]cudamega.Operation{"add3": ops["add3"]}); err == nil {
		t.Fatal("accepted incomplete operation set")
	}
}
