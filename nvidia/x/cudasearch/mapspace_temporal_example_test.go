package cudasearch_test

import (
	"context"
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/cudasearch"
)

func ExampleMapSpaceSearchResult() {
	problem := map[string]int{"B": 1, "M": 2, "N": 2, "K": 2}
	fact := cudalayout.NewIndexFactorization([]cudalayout.StorageLevel{cudalayout.LevelL2, cudalayout.LevelDRAM}, map[cudalayout.StorageLevel]map[string]int{
		cudalayout.LevelL2: {"B": 1, "M": 1, "N": 1, "K": 1}, cudalayout.LevelDRAM: problem,
	})
	orders := []cudalayout.LoopPermutation{
		cudalayout.NewLoopPermutation(map[cudalayout.StorageLevel][]string{cudalayout.LevelDRAM: {"K", "M", "N"}}),
		cudalayout.NewLoopPermutation(map[cudalayout.StorageLevel][]string{cudalayout.LevelDRAM: {"M", "N", "K"}}),
	}
	sizes := map[cudalayout.TensorRole]int{cudalayout.TensorA: 1, cudalayout.TensorB: 1, cudalayout.TensorC: 1}
	search := cudasearch.NewLinearPrunedSearch([]cudalayout.IndexFactorization{fact}, orders, nil, cudalayout.HardwareConstraints{}, sizes, problem)
	stats, err := search.Search(context.Background())
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(stats.Best.Mapping.Permutation.Orders[cudalayout.LevelDRAM])
	fmt.Println(stats.Best.Movement.DRAM.TotalBytes, stats.BestIntensity)
	// Output:
	// [M N K]
	// 20 0.8
}
