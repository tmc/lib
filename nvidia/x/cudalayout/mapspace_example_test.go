package cudalayout

import "fmt"

func ExampleValidateCapacity() {
	levels := []StorageLevel{LevelRegister, LevelShared, LevelDRAM}
	factors := NewIndexFactorization(levels, map[StorageLevel]map[string]int{
		LevelRegister: {"B": 1, "M": 1, "N": 1, "K": 1},
		LevelShared:   {"B": 1, "M": 4, "N": 4, "K": 4},
		LevelDRAM:     {"B": 1, "M": 4, "N": 4, "K": 4},
	})
	err := ValidateCapacity(factors, NewDatatypeBypass(), map[TensorRole]int{
		TensorA: 2, TensorB: 2, TensorC: 4,
	}, HardwareConstraints{MaxSharedMemoryPerBlock: 1024})
	fmt.Println(err)
	// Output: <nil>
}
