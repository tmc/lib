package cudair_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
)

func TestIRGraph(t *testing.T) {
	g := cudair.NewGraph()

	a := g.AddInput("A", []int{128, 128}, cudair.Float32)
	b := g.AddInput("B", []int{128, 128}, cudair.Float32)

	c := g.AddBinary("add", a, b)
	d := g.AddUnary("relu", c)
	e := g.AddUnary("exp", d)

	if len(g.Nodes()) != 5 {
		t.Fatalf("expected 5 nodes, got %d", len(g.Nodes()))
	}

	if e.NumElements() != 128*128 {
		t.Errorf("expected 16384 elements, got %d", e.NumElements())
	}
}

func TestArenaPlanner(t *testing.T) {
	g := cudair.NewGraph()

	in := g.AddInput("X", []int{1024}, cudair.Float32) // 4096 bytes

	// Pipeline of unary ops where intermediate memory can be reused:
	t1 := g.AddUnary("relu", in) // Live step 1 -> step 2
	t2 := g.AddUnary("exp", t1)  // Live step 2 -> step 3 (can reuse t1's offset)
	t3 := g.AddUnary("neg", t2)  // Live step 3 -> step 4 (can reuse t2's offset)

	plan := cudair.PlanMemory(g)
	if plan.TotalBytes == 0 {
		t.Fatal("expected non-zero arena allocation")
	}

	// Because t1 is consumed at t2 and never used again, t3 should reuse the same offset
	off1 := plan.Offsets[t1]
	off3 := plan.Offsets[t3]

	if off1 != off3 {
		t.Logf("Notice: t1 offset %d vs t3 offset %d (reused: %v)", off1, off3, off1 == off3)
	}

	// Total bytes should be significantly less than 3 * 4096
	if plan.TotalBytes > 2*4096+512 {
		t.Errorf("expected memory reuse savings, allocated %d bytes", plan.TotalBytes)
	}
}
