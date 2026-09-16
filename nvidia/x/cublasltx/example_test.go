package cublasltx_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/cublaslt"
	"github.com/tmc/lib/nvidia/x/cublasltx"
)

func ExampleMatMulConfig() {
	cfg := cublasltx.MatMulConfig{
		ComputeType: cublaslt.Compute32F,
		ScaleType:   cublaslt.R32F,
		AType:       cublaslt.R16F,
		BType:       cublaslt.R16F,
		DType:       cublaslt.R16F,
		TransA:      true,
		Alpha:       1.0,
		Beta:        0.0,
	}

	fmt.Println("ComputeType:", cfg.ComputeType)
	fmt.Println("TransA:", cfg.TransA)
	fmt.Println("Alpha:", cfg.Alpha)

	// Output:
	// ComputeType: 68
	// TransA: true
	// Alpha: 1
}
