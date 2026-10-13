package cudafuse_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudafuse"
	"github.com/tmc/lib/nvidia/x/cudair"
	"slices"
	"testing"
)

func ExampleLazyArray_Cycle() {
	ctx := cudafuse.NewLazyContext()
	a := cudafuse.NewLazyArray(ctx, []int{3}, cudair.Float32)
	b := a.Cycle([]int{2, 3})
	fmt.Println(b.Shape, ctx.Tracker.Snapshot().Allocations)
	// Output: [2 3] 0
}

func TestCycleShapeConstruction(t *testing.T) {
	ctx := cudafuse.NewLazyContext()
	a := cudafuse.NewLazyArray(ctx, []int{3}, cudair.Float32)
	shape := []int{2, 3}
	r := a.Cycle(shape)
	shape[0] = 9
	if r.Node.Err != nil || !slices.Equal(r.Shape, []int{2, 3}) {
		t.Fatal(r.Shape, r.Node.Err)
	}
	r.Shape[0] = 8
	if !slices.Equal(r.Node.Shape, []int{2, 3}) || !slices.Equal(r.Node.Attrs["target_shape"].([]int), []int{2, 3}) {
		t.Fatal("shape alias")
	}
	for _, target := range [][]int{nil, {}, {0}, {0, -1}, {-1}} {
		if got := a.Cycle(target); got.Node.Err == nil {
			t.Fatal("bad target", target)
		}
	}
	if a.Cycle([]int{2}).Node.Err != nil {
		t.Fatal("truncation refused")
	}
	if cudafuse.NewLazyArray(ctx, nil, cudair.Float32).Cycle([]int{4}).Node.Err != nil {
		t.Fatal("scalar source refused")
	}
	if cudafuse.NewLazyArray(ctx, []int{0}, cudair.Float32).Cycle([]int{4}).Node.Err == nil {
		t.Fatal("empty source admitted")
	}
}

func TestGatherReplicateShapeConstruction(t *testing.T) {
	ctx := cudafuse.NewLazyContext()
	a := cudafuse.NewLazyArray(ctx, []int{2, 3}, cudair.Float32)
	i := cudafuse.NewLazyArray(ctx, []int{2, 2}, cudair.Int32)
	if got := a.Gather(i); got.Node.Err != nil || !slices.Equal(got.Shape, []int{4}) {
		t.Fatal(got)
	}
	if got := a.Replicate(2); got.Node.Err != nil || !slices.Equal(got.Shape, []int{12}) {
		t.Fatal(got)
	}
	if a.Gather(a).Node.Err == nil || a.Replicate(0).Node.Err == nil {
		t.Fatal("invalid view admitted")
	}
	if cudafuse.NewLazyArray(ctx, []int{0}, cudair.Float32).Gather(i).Node.Err == nil {
		t.Fatal("nonempty gather of empty source")
	}
}
