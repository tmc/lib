package cudafuse_test

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudafuse"
	"github.com/tmc/lib/nvidia/x/cudair"
)

func ExampleLazyArray_Filter() {
	ctx := cudafuse.NewLazyContext()
	a := cudafuse.NewLazyArray(ctx, []int{0}, cudair.Float32)
	predicate := cudafuse.NewLazyArray(ctx, []int{0}, cudair.Float32)
	filtered := a.Filter(predicate)
	plan, _ := filtered.Materialize()
	fmt.Println(filtered.Shape, plan.AllocatedBytes)
	bindings := map[string]cudafuse.ExecutionBinding{
		a.Node.Name():         {Shape: []int{0}, DType: cudair.Float32},
		predicate.Node.Name(): {Shape: []int{0}, DType: cudair.Float32},
	}
	result, err := plan.Execute(context.Background(), nil, bindings)
	if result != nil {
		defer result.Close()
	}
	fmt.Println(result.Shape(), err)
	// Output:
	// [-1] -1
	// [0] <nil>
}
