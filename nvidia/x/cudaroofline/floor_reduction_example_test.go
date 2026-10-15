package cudaroofline_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

func reductionExample() (cudaroofline.GEMMSplitKMovement, error) {
	levels := []cudalayout.StorageLevel{0, 1, 2, 3}
	factors := map[cudalayout.StorageLevel]map[string]int{}
	orders := map[cudalayout.StorageLevel][]string{}
	bypass := cudalayout.NewDatatypeBypass()
	for _, l := range levels {
		factors[l] = map[string]int{"B": 1, "M": 1, "N": 1, "K": 1}
		orders[l] = []string{"B", "M", "N", "K"}
		if l != 3 {
			for r := cudalayout.TensorA; r <= cudalayout.TensorC; r++ {
				bypass.SetBypass(l, r, true)
			}
		}
	}
	factors[2]["K"] = 3
	m := cudalayout.MapSpace{Problem: map[string]int{"B": 1, "M": 1, "N": 1, "K": 3}, Factorization: cudalayout.NewIndexFactorization(levels, factors), Permutation: cudalayout.NewLoopPermutation(orders), Spatial: cudalayout.NewSpatialTiling(map[string]int{"K": 3}, nil), Bypass: bypass}
	return cudaroofline.AnalyticalGEMMSplitK(m, map[cudalayout.TensorRole]int{0: 1, 1: 1, 2: 1}, cudaroofline.GEMMSpatialProfile{})
}

func ExampleAnalyticalGEMMSplitK() {
	r, err := reductionExample()
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(r.Partials, r.GEMM.Movement.DRAM.TotalBytes, r.DRAM.TotalBytes)
	fmt.Println(r.WorkspaceBytes, r.ReductionAdditions, r.ExecutedFLOPs)
	// Output:
	// 3 9 17
	// 5 2 8
}

func ExampleGEMMReductionRound() {
	r, err := reductionExample()
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, round := range r.Rounds {
		fmt.Println(round.InputPartials, round.OutputPartials, round.ReadBytes, round.WriteBytes, round.Additions)
	}
	// Output:
	// 3 2 3 2 1
	// 2 1 2 1 1
}

func ExampleGEMMSplitKMovement() {
	r, err := reductionExample()
	if err != nil {
		fmt.Println(err)
		return
	}
	// GEMM capacity/union covers tagged partials. The tree bypasses mapped
	// intermediate buffers, reserves separate scratch, and writes final C.
	fmt.Println(r.OutputBytes, r.WorkspaceBytes, r.PeakLivePartialBytes, r.MergeRegisterBytes)
	// Output: 1 5 5 2
}
