package cudafuse_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudafuse"
	"github.com/tmc/lib/nvidia/x/cudair"
)

func ExampleFuseLazy() {
	ctx := cudafuse.NewLazyContext()
	x := cudafuse.NewLazyArray(ctx, []int{2, 3}, cudair.Float32)
	output := x.Transpose([]int{1, 0}).Map("neg")
	kernel, err := cudafuse.FuseLazy([]*cudafuse.LazyNode{x.Node}, []*cudafuse.LazyNode{output.Node}, "transpose_neg")
	fmt.Println(kernel.Name, len(kernel.Inputs), len(kernel.Outputs), err)
	// Output: transpose_neg 1 1 <nil>
}
