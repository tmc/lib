package cudasample_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudasample"
)

func ExampleSampleArgmaxCPU() {
	logits := []float32{-2, 4, 4, 1}
	fmt.Println(cudasample.SampleArgmaxCPU(logits))
	fmt.Println(cudasample.SampleArgmaxCPU(nil))
	// Output:
	// 1
	// -1
}
