package cudasearch_test

import (
	"bytes"
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/cudasearch"
	"github.com/tmc/lib/nvidia/x/cudatensor"
)

func ExampleSearchTimeloopGEMM() {
	g := &cudatensor.Graph{
		Nodes: []cudatensor.EinsumNode{{
			ID: "mm", Equation: "mk,kn->mn", Inputs: []string{"", ""},
			InputTensors:  []cudatensor.TensorInfo{{Name: "act"}, {Name: "wt"}},
			OutputTensors: []cudatensor.TensorInfo{{Name: "out"}},
		}},
		Shape: map[string]string{"M": "4", "N": "6", "K": "8"},
	}
	canonical, err := cudatensor.CanonicalizeRanks(g)
	if err != nil {
		fmt.Println(err)
		return
	}
	var workload bytes.Buffer
	if err := cudatensor.ExportTimeloopWorkload(canonical, &workload); err != nil {
		fmt.Println(err)
		return
	}

	levels := []cudalayout.StorageLevel{cudalayout.LevelShared, cudalayout.LevelDRAM}
	tiles := func(shared, dram map[string]int) cudalayout.IndexFactorization {
		return cudalayout.NewIndexFactorization(levels, map[cudalayout.StorageLevel]map[string]int{
			cudalayout.LevelShared: shared,
			cudalayout.LevelDRAM:   dram,
		})
	}
	result, err := cudasearch.SearchTimeloopGEMM(context.Background(), &workload, cudasearch.TimeloopGEMMOptions{
		Einsum:       "mm",
		MaxBytes:     1 << 20,
		ElementBytes: map[string]int{"act": 1, "wt": 2, "out": 4},
		Factorizations: []cudalayout.IndexFactorization{
			tiles(map[string]int{"B": 1, "M": 4, "N": 6, "K": 8}, map[string]int{"B": 1, "M": 1, "N": 1, "K": 1}),
			tiles(map[string]int{"B": 1, "M": 2, "N": 3, "K": 4}, map[string]int{"B": 1, "M": 2, "N": 2, "K": 2}),
		},
		Constraints: cudalayout.HardwareConstraints{MaxSharedMemoryPerBlock: 64},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	stats := result.Stats
	fmt.Println(result.TensorNames[cudalayout.TensorA], result.TensorNames[cudalayout.TensorB], result.TensorNames[cudalayout.TensorC])
	fmt.Println(result.RankNames["A"], result.RankNames["B"], result.RankNames["C"])
	fmt.Println(stats.FactorizationsTested, stats.FactorizationsPruned, stats.EvaluatedCandidates)
	fmt.Println(stats.Best.TotalDRAM().TensorTransfers)
	// Output:
	// act wt out
	// M N K
	// 2 1 1
	// map[A:64 B:192 C:96]
}
