package cudaroofline_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

func spatialExample() (cudaroofline.SpatialHierarchyMovement, error) {
	levels := []cudalayout.StorageLevel{cudalayout.LevelRegister, cudalayout.LevelShared, cudalayout.LevelL2, cudalayout.LevelDRAM}
	factors := map[cudalayout.StorageLevel]map[string]int{
		cudalayout.LevelRegister: {"B": 1, "M": 1, "N": 1, "K": 1},
		cudalayout.LevelShared:   {"B": 1, "M": 2, "N": 2, "K": 2},
		cudalayout.LevelL2:       {"B": 1, "M": 2, "N": 1, "K": 1},
		cudalayout.LevelDRAM:     {"B": 1, "M": 1, "N": 1, "K": 1},
	}
	orders := map[cudalayout.StorageLevel][]string{}
	for _, l := range levels {
		orders[l] = []string{"B", "M", "N", "K"}
	}
	mapping := cudalayout.MapSpace{Problem: map[string]int{"B": 1, "M": 4, "N": 2, "K": 2}, Factorization: cudalayout.NewIndexFactorization(levels, factors), Permutation: cudalayout.NewLoopPermutation(orders), Spatial: cudalayout.NewSpatialTiling(map[string]int{"M": 2}, map[string]int{"N": 2})}
	return cudaroofline.AnalyticalGEMMSpatial(mapping, map[cudalayout.TensorRole]int{cudalayout.TensorA: 1, cudalayout.TensorB: 1, cudalayout.TensorC: 1}, cudaroofline.GEMMSpatialProfile{LaneMesh: map[string]int{"M": 2}})
}

func ExampleAnalyticalGEMMSpatial() {
	r, err := spatialExample()
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(r.ComputeInstances, r.Movement.DRAM.TotalBytes)
	fmt.Println(r.Levels[0].Tensors[cudalayout.TensorA].MaxResidentUnionBytes)
	// Output:
	// 8 20
	// 5
}

func ExampleGEMMSpatialProfile() {
	profile := cudaroofline.GEMMSpatialProfile{LaneMesh: map[string]int{"M": 2}}
	fmt.Println(profile.LaneMesh["M"])
	// Output: 2
}

func ExampleSpatialTensorMovement() {
	r, err := spatialExample()
	if err != nil {
		fmt.Println(err)
		return
	}
	a := r.Levels[0].Tensors[cudalayout.TensorA]
	fmt.Println(a.PerInstance.ResidentBytes, a.MaxResidentUnionBytes, a.ParentFanout)
	// Output: 1 5 4
}

func ExampleSpatialHierarchyLevel() {
	r, err := spatialExample()
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(r.Levels[1].Level, r.Levels[1].Instances)
	// Output: Shared 2
}

func ExampleSpatialHierarchyMovement() {
	r, err := spatialExample()
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(r.Movement.Levels[0].Tensors[cudalayout.TensorA].ResidentBytes)
	// Output: 8
}
