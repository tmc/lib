package cudasearch_test

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/cudasearch"
)

func ExampleLinearPrunedSearch_Search() {
	levels := []cudalayout.StorageLevel{cudalayout.LevelRegister, cudalayout.LevelShared, cudalayout.LevelDRAM}
	fact := cudalayout.NewIndexFactorization(levels, map[cudalayout.StorageLevel]map[string]int{
		cudalayout.LevelRegister: {"B": 1, "M": 1, "N": 1, "K": 1},
		cudalayout.LevelShared:   {"B": 1, "M": 4, "N": 4, "K": 4},
		cudalayout.LevelDRAM:     {"B": 1, "M": 1, "N": 1, "K": 1},
	})
	search := cudasearch.NewLinearPrunedSearch(
		[]cudalayout.IndexFactorization{fact}, nil, nil,
		cudalayout.HardwareConstraints{MaxSharedMemoryPerBlock: 1024},
		map[cudalayout.TensorRole]int{cudalayout.TensorA: 2, cudalayout.TensorB: 2, cudalayout.TensorC: 4},
		map[string]int{"B": 1, "M": 4, "N": 4, "K": 4},
	)
	stats, err := search.Search(context.Background())
	fmt.Println(err, stats.EvaluatedCandidates)
	// Output: <nil> 1
}
