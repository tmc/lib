package metalsample_test

import (
	"fmt"

	"github.com/tmc/lib/apple/x/metal/metalsample"
)

func ExampleSampleArgmaxCPU() {
	fmt.Println(metalsample.SampleArgmaxCPU([]float32{-1, 3, 2}))
	// Output: 1
}
