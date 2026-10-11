package cudair_test

import (
	"errors"
	"reflect"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudaref"
)

func TestSnapshotDeepOwnership(t *testing.T) {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{2}, cudair.Float32)
	c := g.AddConstant("c", []int{2}, cudair.Float32, []float32{3, 4})
	a := g.AddBinary("add", x, c)
	out := g.AddReshape(a, []int{1, 2})
	a.Axes, a.Perm = []int{0}, []int{0}
	a.SliceStarts, a.SliceEnds, a.SliceSteps = []int{0}, []int{2}, []int{1}
	a.Strides = []int{1}
	x.SourceLoc.Callers = []cudair.Frame{{Function: "original"}}
	_ = g.SetOutputs(out)
	aID := a.ID
	s, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	check := func(t *testing.T) {
		t.Helper()
		frozen := s.NodeByID(aID)
		if frozen.Axes[0] != 0 || frozen.Perm[0] != 0 || frozen.Strides[0] != 1 || frozen.SliceStarts[0] != 0 || frozen.SliceEnds[0] != 2 || frozen.SliceSteps[0] != 1 {
			t.Fatal("mutable attributes leaked")
		}
		got, err := cudaref.EvaluateSnapshot(s, map[string][]float32{"x": {1, 2}})
		if err != nil || !reflect.DeepEqual(got[out.Name], []float32{4, 6}) {
			t.Fatalf("snapshot changed: %v %v", got, err)
		}
	}
	damage := func(nodes []*cudair.Node) {
		seen := make(map[*cudair.Node]bool)
		var visit func(*cudair.Node)
		visit = func(n *cudair.Node) {
			if n == nil || seen[n] {
				return
			}
			seen[n] = true
			for _, in := range n.Inputs {
				visit(in)
			}
			visit(n.StorageAlias)
			if len(n.Shape) > 0 {
				n.Shape[0] = 99
			}
			if len(n.ConstValues) > 0 {
				n.ConstValues[0] = 99
			}
			for _, attr := range [][]int{n.Strides, n.Axes, n.Perm, n.SliceStarts, n.SliceEnds, n.SliceSteps} {
				if len(attr) > 0 {
					attr[0] = 99
				}
			}
			n.Op = "unknown"
			n.Shape = []int{99}
			n.ID = -1
			n.Strides = []int{99}
			n.ConstValues = []float32{99}
			n.Axes = []int{99}
			n.Perm = []int{99}
			n.SliceStarts = []int{99}
			n.SliceEnds = []int{99}
			n.SliceSteps = []int{99}
			if len(n.SourceLoc.Callers) > 0 {
				n.SourceLoc.Callers[0].Function = "changed"
			}
			n.Inputs = nil
			n.StorageAlias = n
		}
		for _, n := range nodes {
			visit(n)
		}
	}
	getters := []struct {
		name  string
		nodes func() []*cudair.Node
	}{
		{"nodes", s.Nodes}, {"inputs", s.Inputs}, {"outputs", s.Outputs}, {"order", s.Order},
		{"node by ID", func() []*cudair.Node { return []*cudair.Node{s.NodeByID(out.ID)} }},
		{"dependencies", func() []*cudair.Node { return s.Dependencies(out) }},
		{"consumers", func() []*cudair.Node { return s.Consumers(x) }},
		{"graph", func() []*cudair.Node { return s.Nodes()[0].Graph().Nodes() }},
		{"arena", func() []*cudair.Node {
			p := s.ArenaPlan()
			p.TotalBytes = 99
			var nodes []*cudair.Node
			for n := range p.Offsets {
				nodes = append(nodes, n)
				p.Offsets[n] = 99
			}
			return nodes
		}},
	}
	for _, getter := range getters {
		t.Run(getter.name, func(t *testing.T) { damage(getter.nodes()); check(t) })
	}
	// Mutating the source graph must not change already validated metadata.
	damage(g.Nodes())
	check(t)
}

func TestSnapshotMembership(t *testing.T) {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{1}, cudair.Float32)
	y := g.AddUnary("relu", x)
	s, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	foreign := cudair.NewGraph().AddInput("x", []int{1}, cudair.Float32)
	if s.Contains(foreign) || s.Dependencies(foreign) != nil || s.Consumers(foreign) != nil {
		t.Fatal("foreign same-ID node admitted")
	}
	own := s.NodeByID(x.ID)
	if !s.Contains(x) || !s.Contains(own) || len(s.Consumers(own)) != 1 {
		t.Fatal("valid membership rejected")
	}
	viewGraph := own.Graph()
	forged := viewGraph.AddInput("forged", []int{1}, cudair.Float32)
	forged.ID = x.ID
	if s.Contains(forged) {
		t.Fatal("new view-graph node spoofed membership")
	}
	own.ID = y.ID
	if s.Contains(own) || s.Consumers(own) != nil {
		t.Fatal("changed detached ID admitted")
	}
	x.ID = y.ID
	if s.Contains(x) {
		t.Fatal("changed original ID admitted")
	}
	// Foreign aliases are checked before any copy can erase their provenance.
	x.ID = 0
	y.StorageAlias = foreign
	y.Effect = cudair.EffectInPlaceAlias
	if _, err := g.Snapshot(); !errors.Is(err, cudair.ErrMixedGraphOwnership) {
		t.Fatalf("foreign alias admitted: %v", err)
	}
}

func TestSnapshotConcurrentViews(t *testing.T) {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{1}, cudair.Float32)
	g.AddUnary("relu", x)
	s, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 20; j++ {
				n := s.Nodes()
				n[0].Shape[0] = 99
				_ = s.ArenaPlan()
				if s.NodeByID(x.ID).Shape[0] != 1 {
					t.Error("shared mutable view")
				}
			}
		}()
	}
	wg.Wait()
}
