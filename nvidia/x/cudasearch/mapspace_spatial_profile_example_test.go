package cudasearch_test

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/cudaroofline"
	"github.com/tmc/lib/nvidia/x/cudasearch"
)

func ExampleLinearPrunedSearch_Search_spatialProfile() {
	levels := []cudalayout.StorageLevel{cudalayout.LevelRegister, cudalayout.LevelShared, cudalayout.LevelL2, cudalayout.LevelDRAM}
	factors := map[cudalayout.StorageLevel]map[string]int{}
	for _, l := range levels {
		factors[l] = map[string]int{"B": 1, "M": 1, "N": 1, "K": 1}
	}
	factors[cudalayout.LevelL2]["M"] = 2
	factors[cudalayout.LevelDRAM]["N"] = 2
	factors[cudalayout.LevelDRAM]["K"] = 2
	search := cudasearch.NewLinearPrunedSearch([]cudalayout.IndexFactorization{cudalayout.NewIndexFactorization(levels, factors)}, nil, nil, cudalayout.HardwareConstraints{}, map[cudalayout.TensorRole]int{cudalayout.TensorA: 1, cudalayout.TensorB: 1, cudalayout.TensorC: 1}, map[string]int{"B": 1, "M": 2, "N": 2, "K": 2})
	search.SpatialTilings = []cudalayout.SpatialTiling{cudalayout.NewSpatialTiling(map[string]int{"M": 2}, nil), cudalayout.NewSpatialTiling(map[string]int{"N": 2}, nil)}
	search.SpatialProfile = &cudaroofline.GEMMSpatialProfile{}
	stats, err := search.Search(context.Background())
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(stats.EvaluatedCandidates, stats.Best.Mapping.Spatial.CTAMesh["N"], stats.Best.Movement.DRAM.TotalBytes)
	// Output: 2 2 16
}
