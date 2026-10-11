package cudair_test

import (
	"errors"
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

	snap, err := g.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot failed: %v", err)
	}
	if len(snap.Outputs()) != 1 || snap.Outputs()[0].ID != e.ID {
		t.Errorf("expected default terminal output to be node e")
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

func TestGraphOutputs(t *testing.T) {
	// Acceptance test: TestGraphOutputs
	g := cudair.NewGraph()
	x := g.AddInput("X", []int{16, 16}, cudair.Float32)
	y := g.AddUnary("relu", x)
	z := g.AddUnary("exp", y)

	// Explicitly designate y as an output (even though z consumes it)
	if err := g.SetOutputs(y, z); err != nil {
		t.Fatalf("SetOutputs failed: %v", err)
	}

	outs := g.Outputs()
	if len(outs) != 2 || outs[0] != y || outs[1] != z {
		t.Fatalf("Outputs mismatch: got %v", outs)
	}

	snap, err := g.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot failed: %v", err)
	}
	if len(snap.Outputs()) != 2 {
		t.Errorf("expected 2 outputs in snapshot, got %d", len(snap.Outputs()))
	}

	// Negative control: attempt setting output from an alien node
	foreignGraph := cudair.NewGraph()
	foreignNode := foreignGraph.AddInput("Alien", []int{16}, cudair.Float32)

	gInvalid := cudair.NewGraph()
	a := gInvalid.AddInput("A", []int{16}, cudair.Float32)
	_ = gInvalid.AddUnary("relu", a)

	err = gInvalid.SetOutputs(foreignNode)
	if !errors.Is(err, cudair.ErrInvalidOutput) {
		t.Errorf("expected ErrInvalidOutput for foreign node, got %v", err)
	}
}

func TestMixedGraphRejected(t *testing.T) {
	// Acceptance test: TestMixedGraphRejected
	g1 := cudair.NewGraph()
	g2 := cudair.NewGraph()

	n1 := g1.AddInput("A", []int{16, 16}, cudair.Float32)
	n2 := g2.AddInput("B", []int{16, 16}, cudair.Float32)

	// Attempt binary op combining nodes from g1 and g2 in g2
	_, err := g2.AddBinaryChecked("add", n1, n2)
	if !errors.Is(err, cudair.ErrMixedGraphOwnership) {
		t.Fatalf("expected ErrMixedGraphOwnership on AddBinaryChecked, got %v", err)
	}

	// Attempt unary op on alien node
	_, err = g2.AddUnaryChecked("relu", n1)
	if !errors.Is(err, cudair.ErrMixedGraphOwnership) {
		t.Fatalf("expected ErrMixedGraphOwnership on AddUnaryChecked, got %v", err)
	}

	// Even if inserted via unchecked methods, Snapshot must reject mixed graph ownership
	g3 := cudair.NewGraph()
	n3 := g3.AddInput("C", []int{16, 16}, cudair.Float32)
	// Bypass checked builder by wiring foreign node into g1
	g1.AddBinary("mul", n1, n3)
	_, err = g1.Snapshot()
	if !errors.Is(err, cudair.ErrMixedGraphOwnership) {
		t.Fatalf("expected ErrMixedGraphOwnership during Snapshot, got %v", err)
	}
}

func TestAliasEffects(t *testing.T) {
	// Acceptance test: TestAliasEffects
	g := cudair.NewGraph()
	x := g.AddInput("X", []int{4, 8}, cudair.Float32)

	// Reshape creates an in-place storage alias
	r := g.AddReshape(x, []int{8, 4})
	if r.Effect != cudair.EffectInPlaceAlias {
		t.Errorf("expected EffectInPlaceAlias, got %v", r.Effect)
	}
	if r.StorageAlias != x {
		t.Errorf("expected StorageAlias to point to input x")
	}

	// Verify that PlanMemory accounts for alias reuse
	y := g.AddUnary("relu", r)
	_ = g.SetOutputs(y)

	snap, err := g.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot failed: %v", err)
	}

	plan := snap.ArenaPlan()
	// r must share the same offset or not allocate fresh buffer
	if plan.Offsets[r] != plan.Offsets[r.StorageAlias] {
		t.Errorf("expected alias offset to match input offset: r=%d, alias=%d",
			plan.Offsets[r], plan.Offsets[r.StorageAlias])
	}

	// Mutant control: reordered stateful/mutating operation
	t.Run("mutant_reorder_stateful_operation", func(t *testing.T) {
		// A cycle created by attempting to reorder/feedback a consumer into producer
		gCycle := cudair.NewGraph()
		in := gCycle.AddInput("In", []int{16}, cudair.Float32)
		u1 := gCycle.AddUnary("relu", in)
		u2 := gCycle.AddUnary("exp", u1)
		// Introduce artificial feedback cycle: u1.Inputs[0] = u2
		u1.Inputs[0] = u2

		_, err := gCycle.Snapshot()
		if !errors.Is(err, cudair.ErrCycleDetected) {
			t.Fatalf("mutant control failed: expected ErrCycleDetected, got %v", err)
		}
	})
}

func TestAuthoritativeOperators(t *testing.T) {
	g := cudair.NewGraph()
	a := g.AddInput("A", []int{32, 16}, cudair.Float32)
	b := g.AddInput("B", []int{16, 64}, cudair.Float32)

	// Matmul
	mm := g.AddMatMul(a, b) // [32, 64]
	if mm.Shape[0] != 32 || mm.Shape[1] != 64 {
		t.Errorf("matmul shape mismatch: %v", mm.Shape)
	}

	// Transpose
	tr := g.AddTranspose(mm, []int{1, 0}) // [64, 32]
	if tr.Shape[0] != 64 || tr.Shape[1] != 32 {
		t.Errorf("transpose shape mismatch: %v", tr.Shape)
	}

	// Reductions
	rSum := g.AddReduce("sum", tr, 1) // [64]
	if len(rSum.Shape) != 1 || rSum.Shape[0] != 64 {
		t.Errorf("reduce sum shape mismatch: %v", rSum.Shape)
	}

	rMax := g.AddReduce("max", tr, 0) // [32]
	if len(rMax.Shape) != 1 || rMax.Shape[0] != 32 {
		t.Errorf("reduce max shape mismatch: %v", rMax.Shape)
	}

	// Slice
	sl := g.AddSlice(mm, []int{0, 0}, []int{16, 32}, []int{1, 1}) // [16, 32]
	if sl.Shape[0] != 16 || sl.Shape[1] != 32 {
		t.Errorf("slice shape mismatch: %v", sl.Shape)
	}

	// Concat
	cat := g.AddConcat([]*cudair.Node{sl, sl}, 0) // [32, 32]
	if cat.Shape[0] != 32 || cat.Shape[1] != 32 {
		t.Errorf("concat shape mismatch: %v", cat.Shape)
	}

	_ = g.SetOutputs(cat, rSum, rMax)
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot failed: %v", err)
	}
	if len(snap.Outputs()) != 3 {
		t.Errorf("expected 3 outputs, got %d", len(snap.Outputs()))
	}
}

func TestBroadcastDimensionMutant(t *testing.T) {
	// Mutant control: mutated broadcast dimension
	g := cudair.NewGraph()
	a := g.AddInput("A", []int{16, 32}, cudair.Float32)
	b := g.AddInput("B", []int{33, 64}, cudair.Float32) // Mutated inner dimension 33 != 32!

	_, err := g.AddMatMulChecked(a, b)
	if !errors.Is(err, cudair.ErrShapeMismatch) {
		t.Fatalf("mutant control failed: expected ErrShapeMismatch for inner dim 32 vs 33, got %v", err)
	}
}
