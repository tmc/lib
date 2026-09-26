package cublasltx_test

import (
	"fmt"
	"log"

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

func ExampleWithWorkspaceBytes() {
	opt := cublasltx.WithWorkspaceBytes(0)
	fmt.Println("configured option:", opt != nil)

	// Output:
	// configured option: true
}

func ExampleMatMulConfig_workspace() {
	cfg := cublasltx.MatMulConfig{
		ComputeType:   cublaslt.Compute32F,
		ScaleType:     cublaslt.R32F,
		AType:         cublaslt.R32F,
		BType:         cublaslt.R32F,
		DType:         cublaslt.R32F,
		WorkspaceSize: 1024 * 1024,
	}

	fmt.Println("WorkspaceSize:", cfg.WorkspaceSize)

	// Output:
	// WorkspaceSize: 1048576
}

func ExampleAlgoConfig() {
	// A rule preferring a CUTLASS 128x128x64 kernel.
	want := cublasltx.AlgoConfig{ID: 21, Tile: cublaslt.MatmulTile128x128, Stages: cublaslt.MatmulStages64x3}
	fmt.Println(want.ID, want.Tile, want.Stages)

	// Output:
	// 21 MatmulTile128x128 MatmulStages64x3
}

func ExamplePick() {
	var (
		h    cublaslt.Handle
		desc cublaslt.MatmulDesc
		a, b cublaslt.MatrixLayout
		d    cublaslt.MatrixLayout
		pref cublaslt.MatmulPreference
	)
	cands, err := cublasltx.Heuristic(h, desc, a, b, d, d, pref, 8)
	if err != nil {
		log.Fatal(err)
	}
	i, err := cublasltx.Pick(cands, func(c cublasltx.AlgoConfig) bool {
		return c.ID == 21 && c.Tile == cublaslt.MatmulTile128x128
	})
	if err != nil {
		log.Fatal(err)
	}
	algo := cands[i].Algo
	_ = algo // pass &algo to h.Matmul
}
