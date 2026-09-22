package cudaplan_test

import (
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudaplan"
	"github.com/tmc/lib/nvidia/x/cudaref"
)

func reductionFixture(t *testing.T, target string) (*cudair.Snapshot, *cudaplan.KernelRegistry, string) {
	t.Helper()
	// Reuse the 64-element add artifact for the operation following reduction.
	_, registry, _ := graphFixture(t, target, false)
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{64, 8}, cudair.Float32)
	b := g.AddInput("b", []int{64}, cudair.Float32)
	y := g.AddBinary("add", g.AddReduce("sum", x, 1), b)
	if err := g.SetOutputs(y); err != nil {
		t.Fatal(err)
	}
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	schema, err := cudakernel.NewSchema(cudakernel.Param{Name: "input", Type: "ptr", IsPointer: true}, cudakernel.Param{Name: "output", Type: "ptr", IsPointer: true})
	if err != nil {
		t.Fatal(err)
	}
	// One thread sums one row. Fixed shape and axis are part of this fixture's
	// artifact contract. Dyadic test data keeps the float32 sum exact.
	ptx := fmt.Sprintf(`.version 9.0
.target %s
.address_size 64
.visible .entry sum_rows(.param .u64 input, .param .u64 output) {
 .reg .pred p;
 .reg .b32 row, column;
 .reg .b64 pi, po, offset;
 .reg .f32 value, total;
 mov.u32 row, %%tid.x;
 setp.ge.u32 p, row, 64;
 @p bra done;
 ld.param.u64 pi, [input];
 ld.param.u64 po, [output];
 mul.wide.u32 offset, row, 32;
 add.u64 pi, pi, offset;
 mul.wide.u32 offset, row, 4;
 add.u64 po, po, offset;
 mov.f32 total, 0f00000000;
 mov.u32 column, 0;
loop:
 ld.global.f32 value, [pi];
 add.rn.f32 total, total, value;
 add.u64 pi, pi, 4;
 add.u32 column, column, 1;
 setp.lt.u32 p, column, 8;
 @p bra loop;
 st.global.f32 [po], total;
done:
 ret;
}
`, target)
	art, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(ptx), "sum_rows", schema, target)
	if err != nil {
		t.Fatal(err)
	}
	if err := registry.Register("sum", art); err != nil {
		t.Fatal(err)
	}
	return snap, registry, y.Name
}

func reductionInputs(round int) map[string][]float32 {
	x, b := make([]float32, 64*8), make([]float32, 64)
	for i := range x {
		x[i] = float32(i%23-11+round) / 8
	}
	for i := range b {
		b[i] = float32(i%7-round) / 4
	}
	return map[string][]float32{"x": x, "b": b}
}

func TestReductionReplayFixtureOracle(t *testing.T) {
	for _, target := range []string{"sm_100a", "sm_121a"} {
		snap, registry, output := reductionFixture(t, target)
		if registry.Get("sum") == nil || registry.Get("add") == nil {
			t.Fatal("missing reduction or successor artifact")
		}
		for round := range 3 {
			inputs := reductionInputs(round)
			result, err := cudaref.EvaluateSnapshot(snap, inputs)
			if err != nil {
				t.Fatal(err)
			}
			got := result[output]
			want := make([]float32, 64)
			for row := range want {
				for column := range 8 {
					want[row] += inputs["x"][row*8+column]
				}
				want[row] += inputs["b"][row]
			}
			if !sameReplayOutput(got, want) {
				t.Fatal("row reduction disagrees with cudaref")
			}
			want[0]++
			if sameReplayOutput(got, want) {
				t.Fatal("corrupted reduction output accepted")
			}
		}
	}
}
