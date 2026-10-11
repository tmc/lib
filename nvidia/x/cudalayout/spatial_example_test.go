package cudalayout_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudalayout"
)

func ExampleSpatialTiling_Validate() {
	f := cudalayout.NewIndexFactorization(
		[]cudalayout.StorageLevel{cudalayout.LevelRegister, cudalayout.LevelShared, cudalayout.LevelDRAM},
		map[cudalayout.StorageLevel]map[string]int{
			cudalayout.LevelRegister: {"M": 1},
			cudalayout.LevelShared:   {"M": 4},
			cudalayout.LevelDRAM:     {"M": 2},
		})
	s := cudalayout.NewSpatialTiling(map[string]int{"M": 2}, map[string]int{"M": 4})
	fmt.Println(s.Validate(f, map[string]int{"M": 8}))
	fmt.Println(s.TotalWarps())
	// Output:
	// <nil>
	// 4
}
