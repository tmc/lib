package cudair_test

import (
	"errors"
	"math"
	"math/big"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

// checkArenaLifetimes derives closed storage lifetimes from public input edges
// and output IDs. It does not use the allocator's freeStep or lastUse maps.
func checkArenaLifetimes(t *testing.T, order, outputs []*cudair.Node, plan *cudair.ArenaPlan) {
	t.Helper()
	nodes := make(map[int]*cudair.Node)
	steps := make(map[int]int)
	for i, n := range order {
		nodes[n.ID], steps[n.ID] = n, i
	}
	root := func(n *cudair.Node) *cudair.Node {
		seen := make(map[int]bool)
		for n.StorageAlias != nil {
			if seen[n.ID] {
				t.Fatal("cycle in public storage aliases")
			}
			seen[n.ID] = true
			n = nodes[n.StorageAlias.ID]
			if n == nil {
				t.Fatal("alias target missing from public order")
			}
		}
		return n
	}
	bytes := func(n *cudair.Node) int {
		value := big.NewInt(int64(n.Type.SizeBytes()))
		for _, d := range n.Shape {
			value.Mul(value, big.NewInt(int64(d)))
		}
		if !value.IsInt64() || value.Sign() < 0 || value.Int64() > int64(math.MaxInt) {
			t.Fatal("fixture size does not fit an int")
		}
		return int(value.Int64())
	}
	offsets := make(map[int]int)
	for n, offset := range plan.Offsets {
		if n == nil || nodes[n.ID] == nil {
			t.Fatal("arena contains a foreign node")
		}
		if _, found := offsets[n.ID]; found {
			t.Fatal("arena contains duplicate node IDs")
		}
		offsets[n.ID] = offset
	}
	until := make(map[int]int)
	for _, n := range order {
		storage := root(n)
		if _, ok := until[storage.ID]; !ok {
			until[storage.ID] = steps[storage.ID]
		}
		for _, input := range n.Inputs {
			id := root(nodes[input.ID]).ID
			if steps[n.ID] > until[id] {
				until[id] = steps[n.ID]
			}
		}
	}
	for _, n := range outputs {
		until[root(nodes[n.ID]).ID] = len(order)
	}
	var allocated []*cudair.Node
	for _, n := range order {
		storage := root(n)
		offset, present := offsets[n.ID]
		if storage.Kind == cudair.KindInput {
			if present {
				t.Fatalf("external node %d has arena offset %d", n.ID, offset)
			}
			continue
		}
		if !present || offset < 0 || offset%256 != 0 || bytes(n) > plan.TotalBytes-offset {
			t.Fatalf("invalid arena entry for node %d: offset=%d present=%v total=%d", n.ID, offset, present, plan.TotalBytes)
		}
		if n.ID != storage.ID {
			base, found := offsets[storage.ID]
			if !found || offset != base {
				t.Fatalf("alias %d does not share storage root %d", n.ID, storage.ID)
			}
		} else {
			allocated = append(allocated, n)
		}
	}
	for i, a := range allocated {
		for _, b := range allocated[i+1:] {
			if steps[a.ID] > until[b.ID] || steps[b.ID] > until[a.ID] || bytes(a) == 0 || bytes(b) == 0 {
				continue
			}
			if offsets[a.ID] < offsets[b.ID]+bytes(b) && offsets[b.ID] < offsets[a.ID]+bytes(a) {
				t.Fatalf("live storage overlaps: nodes %d and %d, steps [%d,%d] and [%d,%d], offsets %d and %d",
					a.ID, b.ID, steps[a.ID], until[a.ID], steps[b.ID], until[b.ID], offsets[a.ID], offsets[b.ID])
			}
		}
	}
}

func TestArenaSequentialLifetimes(t *testing.T) {
	tests := []struct {
		name  string
		build func(*testing.T, *cudair.Graph)
	}{
		{"unary chain", func(t *testing.T, g *cudair.Graph) {
			x := g.AddInput("x", []int{16}, cudair.Float32)
			a := g.AddUnary("neg", x)
			b := g.AddUnary("exp", a)
			g.AddUnary("relu", b)
		}},
		{"readonly transpose explicit outputs", func(t *testing.T, g *cudair.Graph) {
			x := g.AddInput("x", []int{2, 2}, cudair.Float32)
			a := g.AddUnary("neg", x)
			view := g.AddTranspose(a, []int{1, 0})
			c := g.AddUnary("neg", view)
			if err := g.SetOutputs(a, c); err != nil {
				t.Fatal(err)
			}
		}},
		{"transpose after input retirement", func(t *testing.T, g *cudair.Graph) {
			x := g.AddInput("x", []int{2, 2}, cudair.Float32)
			a := g.AddUnary("neg", x)
			view := g.AddTranspose(a, []int{1, 0})
			g.AddUnary("neg", view)
		}},
		{"early intermediate output", func(t *testing.T, g *cudair.Graph) {
			x := g.AddInput("x", []int{16}, cudair.Float32)
			a := g.AddUnary("neg", x)
			b := g.AddUnary("relu", a)
			g.AddUnary("exp", b)
			if err := g.SetOutputs(a, b); err != nil {
				t.Fatal(err)
			}
		}},
		{"alias chain output retains root", func(t *testing.T, g *cudair.Graph) {
			x := g.AddInput("x", []int{16}, cudair.Float32)
			a := g.AddUnary("neg", x)
			v := g.AddReshape(a, []int{4, 4})
			w := g.AddReshape(v, []int{8, 2})
			b := g.AddUnary("exp", a)
			c := g.AddUnary("neg", b)
			if err := g.SetOutputs(w, c); err != nil {
				t.Fatal(err)
			}
		}},
		{"late alias consumer retains root", func(t *testing.T, g *cudair.Graph) {
			x := g.AddInput("x", []int{16}, cudair.Float32)
			a := g.AddUnary("neg", x)
			v := g.AddReshape(a, []int{4, 4})
			b := g.AddUnary("exp", x)
			c := g.AddUnary("neg", b)
			last := g.AddUnary("relu", v)
			if err := g.SetOutputs(c, last); err != nil {
				t.Fatal(err)
			}
		}},
		{"explicit aliases may share root", func(t *testing.T, g *cudair.Graph) {
			x := g.AddInput("x", []int{16}, cudair.Float32)
			a := g.AddUnary("neg", x)
			v := g.AddReshape(a, []int{4, 4})
			if err := g.SetOutputs(a, v); err != nil {
				t.Fatal(err)
			}
		}},
		{"external alias chain", func(t *testing.T, g *cudair.Graph) {
			x := g.AddInput("x", []int{16}, cudair.Float32)
			v := g.AddReshape(x, []int{4, 4})
			w := g.AddReshape(v, []int{8, 2})
			y := g.AddUnary("neg", w)
			if err := g.SetOutputs(w, y); err != nil {
				t.Fatal(err)
			}
		}},
		{"only external alias output", func(t *testing.T, g *cudair.Graph) {
			x := g.AddInput("x", []int{16}, cudair.Float32)
			v := g.AddReshape(x, []int{4, 4})
			g.AddReshape(v, []int{8, 2})
		}},
		{"multiple default outputs", func(t *testing.T, g *cudair.Graph) {
			x := g.AddInput("x", []int{16}, cudair.Float32)
			g.AddUnary("neg", x)
			g.AddUnary("relu", x)
		}},
		{"empty output", func(t *testing.T, g *cudair.Graph) {
			x := g.AddInput("x", []int{0}, cudair.Float32)
			g.AddUnary("neg", x)
		}},
		{"larger aligned buffers", func(t *testing.T, g *cudair.Graph) {
			x := g.AddInput("x", []int{129}, cudair.Float32)
			a := g.AddUnary("neg", x)
			b := g.AddUnary("exp", a)
			g.AddUnary("relu", b)
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			g := cudair.NewGraph()
			test.build(t, g)
			snapshot, err := g.Snapshot()
			if err != nil {
				t.Fatal(err)
			}
			checkArenaLifetimes(t, snapshot.Order(), snapshot.Outputs(), snapshot.ArenaPlan())
			checkArenaLifetimes(t, g.Nodes(), g.Outputs(), cudair.PlanMemory(g))
		})
	}
}

func TestArenaReuseAfterLastConsumer(t *testing.T) {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{1024}, cudair.Float32)
	a := g.AddUnary("relu", x)
	b := g.AddUnary("exp", a)
	c := g.AddUnary("neg", b)
	snapshot, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	plan := snapshot.ArenaPlan()
	offsets := make(map[int]int)
	for n, offset := range plan.Offsets {
		offsets[n.ID] = offset
	}
	aOffset, aOK := offsets[a.ID]
	bOffset, bOK := offsets[b.ID]
	cOffset, cOK := offsets[c.ID]
	if !aOK || !bOK || !cOK || aOffset == bOffset || aOffset != cOffset || plan.TotalBytes != 2*4096 {
		t.Fatalf("expected two alternating buffers: offsets=%v total=%d", offsets, plan.TotalBytes)
	}
}

func TestArenaDistinctOutputsRespectByteLimit(t *testing.T) {
	elements := cudashape.MaxByteLimit / 4
	if elements > uint64(math.MaxInt) {
		return
	}
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{int(elements)}, cudair.Float32)
	g.AddUnary("neg", x)
	g.AddUnary("relu", x)
	if _, err := g.Snapshot(); !errors.Is(err, cudair.ErrArenaOverflow) {
		t.Fatalf("expected arena overflow for distinct maximum-sized outputs, got %v", err)
	}
}
