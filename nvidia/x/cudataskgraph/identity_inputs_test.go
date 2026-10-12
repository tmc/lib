package cudataskgraph_test

import (
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

// fanIn returns two sources and one consumer of both, with the consumer's
// data inputs and extent supplied by the caller. Every graph it returns has
// the same tasks, ranks, bytes, scopes and dependency edges.
func fanIn(t *testing.T, inputs []string, extent uint64) cudataskgraph.Graph {
	t.Helper()
	g, err := cudataskgraph.New([]cudataskgraph.Task{
		{ID: "a", Rank: 0, Bytes: 16, Scope: cudataskgraph.ScopeDevice, Extent: 64},
		{ID: "b", Rank: 0, Bytes: 16, Scope: cudataskgraph.ScopeDevice, Extent: 64},
		{ID: "c", Deps: []string{"a", "b"}, Inputs: inputs, Rank: 1, Bytes: 16, Scope: cudataskgraph.ScopeDevice, Extent: extent},
	})
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func TestIdentityCoversInputsAndExtent(t *testing.T) {
	tests := []struct {
		name   string
		inputs []string
		extent uint64
	}{
		{"undeclared", nil, 0},
		{"first", []string{"a"}, 0},
		{"second", []string{"b"}, 0},
		{"both", []string{"a", "b"}, 0},
		{"both-swapped", []string{"b", "a"}, 0},
		{"extent-64", nil, 64},
		{"extent-128", nil, 128},
		{"both-extent-64", []string{"a", "b"}, 64},
	}
	seen := map[string]string{}
	for _, tt := range tests {
		id := fanIn(t, tt.inputs, tt.extent).Identity()
		if prev, ok := seen[id]; ok {
			t.Errorf("%s and %s have the same identity %s", prev, tt.name, id)
		}
		seen[id] = tt.name
		if again := fanIn(t, tt.inputs, tt.extent).Identity(); again != id {
			t.Errorf("%s: identity is not stable: %s then %s", tt.name, id, again)
		}
	}
}

// TestIdentityFramingIsUnambiguous builds pairs of different graphs whose
// fields, written one after another with a separator, are the same bytes: a
// task ID that contains the separator absorbs what would otherwise be a
// second task's fields. An identity must tell each pair apart.
//
// The pairs differ in which fields follow the ID, so that a separator
// encoding is caught whatever fields it writes: the first lines up with ID,
// rank, bytes and scope; the second with an extent written after the scope.
func TestIdentityFramingIsUnambiguous(t *testing.T) {
	two, err := cudataskgraph.New([]cudataskgraph.Task{
		{ID: "a", Rank: 0, Scope: cudataskgraph.ScopeSystem},
		{ID: "b", Rank: 1, Scope: cudataskgraph.ScopeSystem},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{
		"a\x000\x000\x004\x00b",
		"a\x000\x000\x004\x000\x00b",
	} {
		one, err := cudataskgraph.New([]cudataskgraph.Task{
			{ID: id, Rank: 1, Scope: cudataskgraph.ScopeSystem},
		})
		if err != nil {
			t.Fatal(err)
		}
		if two.Identity() == one.Identity() {
			t.Errorf("a two-task graph and the one-task graph %q share identity %s", id, one.Identity())
		}
	}
}

// TestIdentityLengthPrefixes builds pairs of different graphs whose strings,
// written one after another, are the same bytes. The first pair moves a
// string boundary inside one list, which only a length before each string
// tells apart; the second moves a boundary between the Deps and Inputs
// lists, which only a length before each list tells apart.
func TestIdentityLengthPrefixes(t *testing.T) {
	dev := cudataskgraph.ScopeDevice
	sources := []cudataskgraph.Task{
		{ID: "a", Scope: dev},
		{ID: "ab", Scope: dev},
		{ID: "b", Scope: dev},
		{ID: "bc", Scope: dev},
		{ID: "c", Scope: dev},
	}
	tests := []struct {
		name       string
		xDeps, xIn []string
		yDeps, yIn []string
	}{
		{"string-boundary", []string{"ab", "c"}, nil, []string{"a", "bc"}, nil},
		{"list-boundary", []string{"a", "b"}, []string{"b"}, []string{"a", "b", "b"}, nil},
	}
	build := func(deps, inputs []string) cudataskgraph.Graph {
		t.Helper()
		g, err := cudataskgraph.New(append(slices.Clone(sources),
			cudataskgraph.Task{ID: "z", Deps: deps, Inputs: inputs, Rank: 1, Scope: dev}))
		if err != nil {
			t.Fatal(err)
		}
		return g
	}
	for _, tt := range tests {
		x, y := build(tt.xDeps, tt.xIn), build(tt.yDeps, tt.yIn)
		if x.Identity() == y.Identity() {
			t.Errorf("%s: deps %q inputs %q and deps %q inputs %q share identity %s",
				tt.name, tt.xDeps, tt.xIn, tt.yDeps, tt.yIn, x.Identity())
		}
	}
}

func TestNewValidatesInputs(t *testing.T) {
	tests := []struct {
		name   string
		inputs []string
		want   string // substring of the error; empty means accepted
	}{
		{"none", nil, ""},
		{"one", []string{"b"}, ""},
		{"two", []string{"b", "a"}, ""},
		{"not-a-dependency", []string{"z"}, `input "z" that is not a dependency`},
		{"self", []string{"c"}, `input "c" that is not a dependency`},
		{"duplicate", []string{"a", "a"}, `input "a" twice`},
		{"empty-name", []string{""}, `input "" that is not a dependency`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := cudataskgraph.New([]cudataskgraph.Task{
				{ID: "a", Rank: 0, Scope: cudataskgraph.ScopeDevice},
				{ID: "b", Rank: 0, Scope: cudataskgraph.ScopeDevice},
				{ID: "c", Deps: []string{"a", "b"}, Inputs: tt.inputs, Rank: 1, Scope: cudataskgraph.ScopeDevice},
			})
			switch {
			case tt.want == "" && err != nil:
				t.Fatalf("New: %v", err)
			case tt.want != "" && err == nil:
				t.Fatalf("New accepted inputs %q", tt.inputs)
			case tt.want != "" && !strings.Contains(err.Error(), tt.want):
				t.Fatalf("New: %v, want error containing %q", err, tt.want)
			}
		})
	}
}

func TestInputsAreCopied(t *testing.T) {
	inputs := []string{"a", "b"}
	g, err := cudataskgraph.New([]cudataskgraph.Task{
		{ID: "a", Rank: 0, Scope: cudataskgraph.ScopeDevice},
		{ID: "b", Rank: 0, Scope: cudataskgraph.ScopeDevice},
		{ID: "c", Deps: []string{"a", "b"}, Inputs: inputs, Rank: 1, Scope: cudataskgraph.ScopeDevice, Extent: 32},
	})
	if err != nil {
		t.Fatal(err)
	}
	id := g.Identity()
	inputs[0] = "b"
	g.Tasks()[2].Inputs[0] = "b"
	clone := g.Clone()
	clone.Tasks()[2].Inputs[1] = "a"
	c := g.Tasks()[2]
	if !slices.Equal(c.Inputs, []string{"a", "b"}) || c.Extent != 32 {
		t.Fatalf("task c = inputs %q extent %d, want [a b] 32", c.Inputs, c.Extent)
	}
	if g.Identity() != id || clone.Identity() != id {
		t.Fatal("mutating a returned or supplied input slice changed the identity")
	}
}

func ExampleGraph_Identity() {
	build := func(inputs []string, extent uint64) string {
		g, err := cudataskgraph.New([]cudataskgraph.Task{
			{ID: "x", Rank: 0, Scope: cudataskgraph.ScopeDevice},
			{ID: "y", Rank: 0, Scope: cudataskgraph.ScopeDevice},
			{ID: "sum", Deps: []string{"x", "y"}, Inputs: inputs, Rank: 1, Scope: cudataskgraph.ScopeDevice, Extent: extent},
		})
		if err != nil {
			return err.Error()
		}
		return g.Identity()
	}
	first := build([]string{"x"}, 256)
	fmt.Println(first == build([]string{"x", "y"}, 256))
	fmt.Println(first == build([]string{"x"}, 512))
	fmt.Println(first == build([]string{"x"}, 256))
	// Output:
	// false
	// false
	// true
}
