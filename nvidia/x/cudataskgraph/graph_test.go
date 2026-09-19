package cudataskgraph

import (
	"strings"
	"testing"
)

func graph(t *testing.T) Graph {
	g, e := New([]Task{{ID: "p", Rank: 0, Scope: ScopeCTA}, {ID: "c", Deps: []string{"p"}, Rank: 1, Scope: ScopeCTA}})
	if e != nil {
		t.Fatal(e)
	}
	return g
}
func TestTaskGraphCycleRejected(t *testing.T) {
	if _, e := New([]Task{{ID: "a", Deps: []string{"b"}, Rank: 1, Scope: ScopeCTA}, {ID: "b", Deps: []string{"a"}, Rank: 0, Scope: ScopeCTA}}); e == nil {
		t.Fatal("accepted cycle")
	}
}
func TestTaskDependencyPublication(t *testing.T) {
	g := graph(t)
	e := NewEpoch(1)
	if e.Publish("c", g) == nil {
		t.Fatal("published before dependency")
	}
	if e.Publish("p", g) != nil || e.Publish("c", g) != nil {
		t.Fatal(e)
	}
}
func TestTaskEpochReuse(t *testing.T) {
	g := graph(t)
	e := NewEpoch(1)
	e.Publish("p", g)
	e.Abort()
	if e.Publish("c", g) == nil {
		t.Fatal("published aborted epoch")
	}
	e = NewEpoch(2)
	if e.Publish("p", g) != nil {
		t.Fatal("new epoch not reusable")
	}
}

func TestGraphCloneIsolationAndIdentity(t *testing.T) {
	g := graph(t)
	clone := g.Clone()
	if clone.Identity() != g.Identity() {
		t.Fatal("clone identity differs")
	}
	clone.tasks["c"].Deps[0] = "changed"
	if g.tasks["c"].Deps[0] != "p" {
		t.Fatal("clone shares dependency storage")
	}
	clone.tasks["c"] = Task{ID: "c", Rank: 1, Scope: ScopeCTA}
	if clone.Identity() == g.Identity() {
		t.Fatal("identity ignores topology")
	}
}

func TestStaleCompletionRejected(t *testing.T) {
	g := graph(t)
	e := NewEpoch(7)
	if e.PublishEpoch(6, "p", g) == nil {
		t.Fatal("accepted stale completion")
	}
	if e.PublishEpoch(7, "p", g) != nil {
		t.Fatal("rejected current completion")
	}
}

// TestDelayedProducerBlocksConsumerPublication is the delayed-producer
// control. The consumer's completion must be refused for the reason that
// matters -- its dependency has not published -- and not for either of the
// two reasons that would also produce a refusal here: an unknown task or an
// aborted epoch. A test that only counted the error would pass if the
// dependency check were deleted and the task lookup happened to fail.
func TestDelayedProducerBlocksConsumerPublication(t *testing.T) {
	g := graph(t)
	e := NewEpoch(3)
	err := e.Publish("c", g)
	if err == nil {
		t.Fatal("published a consumer whose producer has not run")
	}
	if !strings.Contains(err.Error(), `dependency "p" is unpublished`) {
		t.Fatalf("refusal does not name the unpublished dependency: %v", err)
	}
	for _, wrong := range []string{"unknown task", "epoch is aborted", "stale completion"} {
		if strings.Contains(err.Error(), wrong) {
			t.Fatalf("refusal is attributed to %q, not to the delayed producer: %v", wrong, err)
		}
	}
	if err := e.Publish("p", g); err != nil {
		t.Fatal(err)
	}
	if err := e.Publish("c", g); err != nil {
		t.Fatalf("consumer still refused after its producer published: %v", err)
	}
	if e.Publish("c", g) == nil {
		t.Fatal("published the same completion twice in one epoch")
	}
}

// TestOutputStorageOwnershipAcrossAbort states the ownership rule on its own,
// without an executor: claim, abort, refuse, drain, release.
func TestOutputStorageOwnershipAcrossAbort(t *testing.T) {
	g := graph(t)
	e := NewEpoch(1)
	if err := e.Claim("p", g); err != nil {
		t.Fatal(err)
	}
	if err := e.Claim("p", g); err == nil {
		t.Fatal("claimed the same output storage twice")
	}
	if err := e.Claim("absent", g); err == nil {
		t.Fatal("claimed storage for a task not in the graph")
	}
	if got := e.Retained(); len(got) != 1 || got[0] != "p" {
		t.Fatalf("retained %v", got)
	}
	if err := e.Release("p"); err != nil {
		t.Fatalf("refused a release with no abort outstanding: %v", err)
	}
	if err := e.Claim("p", g); err != nil {
		t.Fatal(err)
	}
	e.Abort()
	if err := e.Release("p"); err == nil {
		t.Fatal("released output storage while an abort was outstanding")
	}
	if e.Drained() {
		t.Fatal("epoch reports drained before the host joined the launch")
	}
	e.Drain()
	if err := e.Release("p"); err != nil {
		t.Fatalf("refused a release after the launch was drained: %v", err)
	}
	if got := e.Retained(); len(got) != 0 {
		t.Fatalf("retained %v after release", got)
	}
}
