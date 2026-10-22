//go:build darwin && arm64

package dsv4expertcache

import "fmt"

func ExampleNewMetalUpload() {
	// Admission checks run before any device allocation.
	upload, err := NewMetalUpload(nil, 2, [3]int{2, 2, 2}, 32768)
	fmt.Println(upload == nil, err)
	// Output: true missing Metal device
}

func ExampleMetalSelection_Views() {
	var selection *MetalSelection
	_, err := selection.Views(0)
	fmt.Println(err)
	// Output: inactive Metal selection
}

func ExampleMetalCommandCompletion_Wait() {
	completion := MetalCommandCompletion{}
	fmt.Println(completion.Wait())
	// Output: metal: nil command buffer
}
