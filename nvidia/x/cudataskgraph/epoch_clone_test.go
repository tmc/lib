package cudataskgraph_test

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

func TestEpochCloneIsolation(t *testing.T) {
	g, err := cudataskgraph.New([]cudataskgraph.Task{{ID: "p", Scope: cudataskgraph.ScopeDevice}, {ID: "q", Rank: 1, Deps: []string{"p"}, Scope: cudataskgraph.ScopeDevice}})
	if err != nil {
		t.Fatal(err)
	}
	e := cudataskgraph.NewEpoch(7)
	if err := e.Publish("p", g); err != nil {
		t.Fatal(err)
	}
	if err := e.Claim("p", g); err != nil {
		t.Fatal(err)
	}
	clone := e.Clone()
	if clone.Number != 7 || !clone.Published("p") || !reflect.DeepEqual(clone.Retained(), []string{"p"}) {
		t.Fatal("clone lost epoch state")
	}
	if err := clone.Publish("q", g); err != nil {
		t.Fatal(err)
	}
	if err := clone.Release("p"); err != nil {
		t.Fatalf("standalone clone release: %v", err)
	}
	if err := clone.Claim("q", g); err != nil {
		t.Fatal(err)
	}
	clone.Number = 8
	clone.Abort()
	clone.Drain()
	if e.Number != 7 || e.Published("q") || e.Aborted() || e.Drained() || !reflect.DeepEqual(e.Retained(), []string{"p"}) {
		t.Fatal("clone mutation changed source epoch")
	}
	later := clone.Clone()
	if later.Number != 8 || !later.Aborted() || !later.Drained() || !later.Published("q") || !reflect.DeepEqual(later.Retained(), []string{"q"}) {
		t.Fatal("clone lost later lifecycle state")
	}
	if err := e.Release("p"); err != nil {
		t.Fatalf("standalone source release changed: %v", err)
	}
	if !reflect.DeepEqual(later.Retained(), []string{"q"}) {
		t.Fatal("source mutation changed clone")
	}
}

func TestEpochCloneZeroAndNil(t *testing.T) {
	var nilEpoch *cudataskgraph.Epoch
	if nilEpoch.Clone() != nil {
		t.Fatal("nil clone is nonnil")
	}
	var zero cudataskgraph.Epoch
	clone := zero.Clone()
	if clone == nil || clone.Number != 0 || clone.Aborted() || clone.Drained() || len(clone.Retained()) != 0 {
		t.Fatal("zero epoch clone differs")
	}
}

func ExampleEpoch_Clone() {
	e := cudataskgraph.NewEpoch(1)
	clone := e.Clone()
	clone.Abort()
	fmt.Println(e.Aborted(), clone.Aborted())
	// Output: false true
}
