package cudatest_test

import (
	"errors"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudatest"
)

func ExampleCompareCUDAOutputFP32() {
	p := cudatest.CUDAOutputPolicy{Target: "sm_90", DType: "fp32", MathMode: "default-fp32", SourceDigest: strings.Repeat("1", 64), InputDigest: strings.Repeat("2", 64), Elements: 1, Absolute: 0.001, NaN: "reject", Infinity: "reject", Zero: "equal"}
	fmt.Println(cudatest.CompareCUDAOutputFP32([]float32{2}, []float32{2}, p))
	// Output: <nil>
}

func ExampleCUDAOutputPolicy() {
	p := cudatest.CUDAOutputPolicy{Target: "sm_90", DType: "fp32", MathMode: "default-fp32", SourceDigest: strings.Repeat("1", 64), InputDigest: strings.Repeat("2", 64), NaN: "reject", Infinity: "reject", Zero: "equal"}
	fmt.Println(cudatest.CompareCUDAOutputFP32(nil, nil, p))
	// Output: <nil>
}

func ExampleCUDAOutputMismatch_Error() {
	p := cudatest.CUDAOutputPolicy{Target: "sm_90", DType: "fp32", MathMode: "default-fp32", SourceDigest: strings.Repeat("1", 64), InputDigest: strings.Repeat("2", 64), Elements: 1, NaN: "reject", Infinity: "reject", Zero: "equal"}
	err := cudatest.CompareCUDAOutputFP32([]float32{3}, []float32{2}, p)
	var mismatch *cudatest.CUDAOutputMismatch
	fmt.Println(errors.As(err, &mismatch), mismatch.Index, mismatch.ActualBits)
	// Output: true 0 1077936128
}
