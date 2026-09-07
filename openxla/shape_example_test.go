package openxla_test

import (
	"fmt"

	"github.com/tmc/lib/openxla"
)

func ExampleBufferWrapper_Shape() {
	// A zero buffer has no native resource to query.
	var buffer openxla.BufferWrapper
	_, _, err := buffer.Shape()
	fmt.Println(err)
	// Output: openxla: buffer is closed
}

func ExampleBufferWrapper_Device() {
	var b openxla.BufferWrapper
	_, err := b.Device()
	fmt.Println(err)
	// Output: openxla: buffer is closed
}
