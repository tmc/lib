package cudataskgraph_test

import (
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

// TestIdentityDependencyCountBoundaries distinguishes dependency counts from
// adjacent input-list fields even when IDs contain encoded lengths.
func TestIdentityDependencyCountBoundaries(t *testing.T) {
	dev := cudataskgraph.ScopeDevice
	n7, n8, n24 := be(7), be(8), be(24)
	w := n8 + n8 + n24
	sources := []cudataskgraph.Task{
		{ID: "a", Scope: dev},
		{ID: "ab", Scope: dev},
		{ID: "b", Scope: dev},
		{ID: "bc", Scope: dev},
		{ID: "c", Scope: dev},
		{ID: "d", Scope: dev},
		{ID: "e", Scope: dev},
		{ID: n7, Scope: dev},
		{ID: n8, Scope: dev},
		{ID: n24, Scope: dev},
		{ID: w, Scope: dev},
	}
	seven := []string{"a", "ab", "b", "bc", "c", "d", "e"}
	tests := []struct {
		name       string
		xDeps, xIn []string
		yDeps, yIn []string
	}{
		{"deps-length", append([]string{n7}, seven...), append([]string{n7}, seven...),
			append(append([]string{n7}, seven...), n8), seven},
		{"deps-count", append([]string{w}, seven...), append([]string{w}, seven...),
			append(append([]string{w}, seven...), n24), append([]string{n24}, seven...)},
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

// TestIdentityIDLengthAtSeveralPositions distinguishes IDs from adjacent
// task fields at several positions in task order.
func TestIdentityIDLengthAtSeveralPositions(t *testing.T) {
	dev := cudataskgraph.ScopeDevice
	tests := []struct {
		name          string
		before, after []cudataskgraph.Task
	}{
		{"alone", nil, nil},
		{"after-one", []cudataskgraph.Task{{ID: "a", Scope: dev}}, nil},
		{"after-two", []cudataskgraph.Task{{ID: "a", Scope: dev}, {ID: "b", Scope: dev}}, nil},
		{"before-one", nil, []cudataskgraph.Task{{ID: "r", Rank: 2, Scope: dev}}},
	}
	for _, tt := range tests {
		build := func(p, q cudataskgraph.Task) cudataskgraph.Graph {
			t.Helper()
			tasks := slices.Concat(tt.before, []cudataskgraph.Task{p, q}, tt.after)
			g, err := cudataskgraph.New(tasks)
			if err != nil {
				t.Fatal(err)
			}
			return g
		}
		x := build(
			cudataskgraph.Task{ID: "p", Rank: 0, Bytes: 0, Scope: dev, Extent: 1},
			cudataskgraph.Task{ID: be(0) + "q", Rank: 1, Scope: dev})
		y := build(
			cudataskgraph.Task{ID: "p" + be(0), Rank: 0, Bytes: uint64(dev), Scope: cudataskgraph.ScopeCTA, Extent: 0},
			cudataskgraph.Task{ID: "q", Rank: 1, Scope: dev})
		if x.Identity() == y.Identity() {
			t.Errorf("%s: graphs %q and %q share identity %s", tt.name, framingIDs(x), framingIDs(y), x.Identity())
		}
	}
}

func framingIDs(g cudataskgraph.Graph) []string {
	var r []string
	for _, t := range g.Tasks() {
		r = append(r, t.ID)
	}
	return r
}
