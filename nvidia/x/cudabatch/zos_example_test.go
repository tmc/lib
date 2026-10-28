// SPDX-License-Identifier: Apache-2.0

package cudabatch_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudabatch"
)

func ExampleBuildZOSRemapPlan() {
	plan, err := cudabatch.BuildZOSRemapPlan(
		[]int32{0, 4, 7},
		[]int32{3, 2},
		[]int32{2, 3},
		[]int32{10, 20},
	)
	if err != nil {
		panic(err)
	}
	fmt.Printf("remap=%v cuSeqlensQ=%v positions=%v outToIn=%v total=%d\n",
		plan.Remap, plan.CuSeqlensQ, plan.Positions, plan.OutToIn, plan.TotalDraftTokens)
	// Output:
	// remap=[2 5] cuSeqlensQ=[0 2 5] positions=[13 14 22 23 24] outToIn=[0 0 1 1 1] total=5
}
