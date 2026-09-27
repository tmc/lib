package cudaroofline_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

func ExampleAnalyticalDataMovement() {
	movement, err := cudaroofline.AnalyticalDataMovement(
		map[string]int{"B": 1, "M": 4, "N": 4, "K": 4},
		map[string]int{"B": 1, "M": 4, "N": 4, "K": 4},
		map[string]int{"A": 2, "B": 2, "C": 4},
		map[string]bool{},
	)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(movement.TotalBytes)
	// Output: 128
}
