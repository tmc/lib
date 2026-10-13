package cudataskgraph_test

import (
	"encoding/binary"
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

// TestIdentityDepLengthAfterFirst moves a string boundary between the
// second and third dependencies of one task, after a first dependency that
// is the same in both graphs: deps [d ab c] and [d a bc]. Only a length
// before the second or third dependency tells them apart; a length before
// the first alone does not.
func TestIdentityDepLengthAfterFirst(t *testing.T) {
	dev := cudataskgraph.ScopeDevice
	sources := []cudataskgraph.Task{
		{ID: "a", Scope: dev},
		{ID: "ab", Scope: dev},
		{ID: "bc", Scope: dev},
		{ID: "c", Scope: dev},
		{ID: "d", Scope: dev},
	}
	build := func(deps []string) cudataskgraph.Graph {
		t.Helper()
		g, err := cudataskgraph.New(append(slices.Clone(sources),
			cudataskgraph.Task{ID: "z", Deps: deps, Rank: 1, Scope: dev}))
		if err != nil {
			t.Fatal(err)
		}
		return g
	}
	xDeps, yDeps := []string{"d", "ab", "c"}, []string{"d", "a", "bc"}
	x, y := build(xDeps), build(yDeps)
	if x.Identity() == y.Identity() {
		t.Errorf("deps %q and %q share identity %s", xDeps, yDeps, x.Identity())
	}
}

// be returns v as the eight big-endian bytes Identity writes for a length
// or a number.
func be(v uint64) string {
	return string(binary.BigEndian.AppendUint64(nil, v))
}

// TestIdentityIDLengthAtIndex3 builds two graphs that differ in the ID of
// the task at index 3 in task order, "k" in one and "k" followed by be(1)
// in the other. Written without that one ID's length, the two graphs are
// the same bytes: the first k task's rank, bytes, scope, extent and
// dependency count line up with the second's ID suffix, rank, bytes, scope
// and extent, and the two dependency lists cover the rest.
//
// The three tasks before it are the same in both graphs, so the pair is
// also the same bytes when the IDs at index 1 and 3 both lose their
// lengths. The pair says nothing about an ID at any other index.
func TestIdentityIDLengthAtIndex3(t *testing.T) {
	dev := cudataskgraph.ScopeDevice
	b8, b112 := be(8), be(112)
	d2 := strings.Repeat(b8+b112, 7)
	sources := []cudataskgraph.Task{
		{ID: b8, Scope: dev},
		{ID: d2, Scope: dev},
		{ID: b112, Scope: dev},
	}
	build := func(k cudataskgraph.Task) cudataskgraph.Graph {
		t.Helper()
		g, err := cudataskgraph.New(append(slices.Clone(sources), k))
		if err != nil {
			t.Fatal(err)
		}
		return g
	}
	x := build(cudataskgraph.Task{ID: "k", Deps: []string{b8, d2},
		Rank: 1, Bytes: 1, Scope: dev, Extent: uint64(dev)})
	y := build(cudataskgraph.Task{ID: "k" + be(1), Deps: slices.Repeat([]string{b112}, 8),
		Rank: 1, Bytes: uint64(dev), Scope: dev, Extent: 2})
	for _, g := range []cudataskgraph.Graph{x, y} {
		var order []string
		for _, task := range g.Tasks() {
			order = append(order, task.ID)
		}
		if len(order) != 4 || order[0] != b8 || order[1] != d2 || order[2] != b112 || order[3][0] != 'k' {
			t.Fatalf("task order %q, want sources %q %q %q then the k task", order, b8, d2, b112)
		}
	}
	if x.Identity() == y.Identity() {
		t.Errorf("k tasks %q and %q at index 3 share identity %s", x.Tasks()[3].ID, y.Tasks()[3].ID, x.Identity())
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
