package cudaregress_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaregress"
)

func ExampleOutputDigest() {
	digest, err := cudaregress.OutputDigest([]float64{1, 2.5})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(len(digest))
	// Output:
	// 64
}
