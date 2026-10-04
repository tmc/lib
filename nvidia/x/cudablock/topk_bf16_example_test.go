package cudablock_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudablock"
)

func ExampleCompileTopKBF16() {
	_, err := cudablock.CompileTopKBF16(nil, cudablock.TopKConfig{})
	fmt.Println(err)
	// Output: cudablock: unsupported topk configuration
}

func ExampleTopKBF16_Artifact() {
	var selection cudablock.TopKBF16
	fmt.Println(selection.Artifact() == nil)
	// Output: true
}

func ExampleTopKBF16_Close() {
	var selection cudablock.TopKBF16
	fmt.Println(selection.Close())
	// Output: <nil>
}

func ExampleTopKBF16_Execute() {
	var selection cudablock.TopKBF16
	fmt.Println(selection.Execute(nil, cudablock.TopKBF16Buffers{}))
	// Output: cudablock: closed topk bf16
}
