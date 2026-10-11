package cudaroofline_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

func exampleHierarchy() (cudaroofline.HierarchyMovement, error) {
	mapping := cudalayout.MapSpace{
		Problem: map[string]int{"B": 1, "M": 2, "N": 2, "K": 2},
		Factorization: cudalayout.NewIndexFactorization(
			[]cudalayout.StorageLevel{cudalayout.LevelL2, cudalayout.LevelDRAM},
			map[cudalayout.StorageLevel]map[string]int{
				cudalayout.LevelL2:   {"B": 1, "M": 1, "N": 1, "K": 1},
				cudalayout.LevelDRAM: {"B": 1, "M": 2, "N": 2, "K": 2},
			}),
		Permutation: cudalayout.NewLoopPermutation(map[cudalayout.StorageLevel][]string{
			cudalayout.LevelDRAM: {"K", "M", "N"},
		}),
	}
	return cudaroofline.AnalyticalGEMMHierarchy(mapping, map[cudalayout.TensorRole]int{
		cudalayout.TensorA: 1, cudalayout.TensorB: 1, cudalayout.TensorC: 1,
	})
}

func ExampleAnalyticalGEMMHierarchy() {
	report, err := exampleHierarchy()
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(report.Levels[0].ResidentBytes, report.DRAM.ReadBytes, report.DRAM.WriteBytes)
	// Output: 3 16 8
}

func ExampleTensorMovement() {
	report, err := exampleHierarchy()
	if err != nil {
		fmt.Println(err)
		return
	}
	tensor := report.Levels[1].Tensors[cudalayout.TensorC]
	fmt.Println(tensor.ResidentBytes, tensor.ReadBytes, tensor.WriteBytes)
	// Output: 4 4 8
}

func ExampleHierarchyLevel() {
	report, err := exampleHierarchy()
	if err != nil {
		fmt.Println(err)
		return
	}
	level := report.Levels[0]
	fmt.Println(level.Level, level.ResidentBytes)
	// Output: L2 3
}

func ExampleHierarchyMovement() {
	report, err := exampleHierarchy()
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("%d levels, %d DRAM bytes, %.3f FLOPs/byte\n", len(report.Levels), report.DRAM.TotalBytes, report.DRAM.OperationalIntensity)
	// Output: 2 levels, 24 DRAM bytes, 0.667 FLOPs/byte
}
