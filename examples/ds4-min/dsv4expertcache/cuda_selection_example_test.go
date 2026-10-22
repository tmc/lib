package dsv4expertcache_test

import (
	"fmt"
	"github.com/tmc/lib/examples/ds4-min/dsv4expertcache"
)

func ExampleCUDASelections_Close() {
	var owner *dsv4expertcache.CUDASelections
	fmt.Println(owner.Close())
	// Output: <nil>
}
func ExampleCUDASelections_Select() {
	var owner *dsv4expertcache.CUDASelections
	_, err := owner.Select(0, []int{1}, false)
	fmt.Println(err)
	// Output: missing CUDA selections
}
func ExampleCUDASelection_Views() {
	var selection *dsv4expertcache.CUDASelection
	_, err := selection.Views(0)
	fmt.Println(err)
	// Output: missing CUDA selection
}
func ExampleCUDAView_Address() {
	var view dsv4expertcache.CUDAView
	_, _, err := view.Address()
	fmt.Println(err)
	// Output: invalid CUDA view
}
