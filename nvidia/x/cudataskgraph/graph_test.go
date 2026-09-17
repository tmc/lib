package cudataskgraph

import "testing"

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
