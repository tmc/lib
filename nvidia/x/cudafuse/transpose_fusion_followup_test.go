package cudafuse_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudafuse"
	"github.com/tmc/lib/nvidia/x/cudair"
)

func TestTransposeStaysInLazyFusionRegion(t *testing.T) {
	ctx := cudafuse.NewLazyContext()
	x := cudafuse.NewLazyArray(ctx, []int{2, 3}, cudair.Float32)
	y := cudafuse.NewLazyArray(ctx, []int{3, 2}, cudair.Float32)
	root := x.Map("neg").Transpose([]int{1, 0}).Add(y)
	if ctx.Tracker.Snapshot().Allocations != 0 || root.Node.Evaluated {
		t.Fatal("transpose chain materialized during construction")
	}
	plan, err := root.Materialize()
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Regions) != 1 || plan.Regions[0].Boundary != "" || plan.Regions[0].Kernel == nil {
		t.Fatalf("transpose split lazy chain into %d regions, want one compiled region", len(plan.Regions))
	}
	if got := cudafuse.ClassifyOp("transpose"); got != cudafuse.OpFusibleElementwise {
		t.Fatalf("transpose classification = %v, want fusible", got)
	}
}
