package cudasearch_test

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/cudasearch"
)

func ExampleLinearPrunedSearch_Search_unitFactors() {
	problem := map[string]int{"B": 1, "M": 1, "N": 1, "K": 1}
	fact := cudalayout.NewIndexFactorization(
		[]cudalayout.StorageLevel{cudalayout.LevelRegister, cudalayout.LevelShared, cudalayout.LevelDRAM},
		map[cudalayout.StorageLevel]map[string]int{
			cudalayout.LevelRegister: problem,
			cudalayout.LevelShared:   problem,
			cudalayout.LevelDRAM:     problem,
		})
	dtypes := map[cudalayout.TensorRole]int{cudalayout.TensorA: 4, cudalayout.TensorB: 4, cudalayout.TensorC: 4}
	limits := cudalayout.HardwareConstraints{MaxSharedMemoryPerBlock: 1024, MaxRegistersPerBlock: 1024, MaxThreadsPerBlock: 32}
	perms := []cudalayout.LoopPermutation{{Orders: map[cudalayout.StorageLevel][]string{cudalayout.LevelShared: nil}}}
	search := cudasearch.NewLinearPrunedSearch([]cudalayout.IndexFactorization{fact}, perms, nil, limits, dtypes, problem)
	stats, err := search.Search(context.Background())
	fmt.Println(stats.EvaluatedCandidates, stats.DownstreamSkipped, err)
	// Output: 1 0 <nil>
}
