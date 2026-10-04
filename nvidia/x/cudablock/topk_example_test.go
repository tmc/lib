package cudablock_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudablock"
)

func ExampleTopKConfig_Extents() {
	c := cudablock.TopKConfig{Rows: 2, Width: 33, RowStride: 40, K: 32}
	e, err := c.Extents()
	fmt.Println(e.Input, e.Results, e.ScratchBytes, err)
	// Output: 73 64 272 <nil>
}

func ExampleCheckTopKStatus() {
	fmt.Println(cudablock.CheckTopKStatus([]int32{0, 0}))
	fmt.Println(cudablock.CheckTopKStatus([]int32{0, int32(cudablock.TopKNaN)}))
	// Output:
	// <nil>
	// cudablock: topk row 1 device status 1
}

func ExampleTopKSchema() {
	s, err := cudablock.TopKSchema()
	fmt.Println(len(s.Params), s.TotalParamBytes, err)
	// Output: 12 80 <nil>
}

func ExampleCompileTopK() {
	// Validate before attempting to compile or use a device.
	_, err := cudablock.CompileTopK(nil, cudablock.TopKConfig{Rows: 1, Width: 33, RowStride: 33, K: 33})
	fmt.Println(err)
	// Output: cudablock: unsupported topk configuration
}

func ExampleTopK_Execute() {
	var selection *cudablock.TopK
	fmt.Println(selection.Execute(nil, cudablock.TopKBuffers{}))
	// Output: cudablock: closed topk
}

func ExampleTopK_Close() {
	var selection *cudablock.TopK
	fmt.Println(selection.Close())
	// Output: <nil>
}

func ExampleTopK_Artifact() {
	var selection *cudablock.TopK
	fmt.Println(selection.Artifact() == nil)
	// Output: true
}
