//go:build linux

package dsv4expertcache_test

import (
	"fmt"
	"github.com/tmc/lib/examples/ds4-min/dsv4expertcache"
)

func ExampleNewCUDASelections() {
	_, err := dsv4expertcache.NewCUDASelections(nil, nil)
	fmt.Println(err)
	// Output: invalid CUDA selection owner
}
func ExampleCUDASelection_Finish() {
	var selection *dsv4expertcache.CUDASelection
	fmt.Println(selection.Finish(nil))
	// Output: missing CUDA consumer stream
}

func ExampleCUDASelection_CheckStream() {
	var selection *dsv4expertcache.CUDASelection
	fmt.Println(selection.CheckStream(nil))
	// Output: missing CUDA consumer stream
}
