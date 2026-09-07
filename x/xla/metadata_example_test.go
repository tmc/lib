package xla_test

import (
	"fmt"
	"github.com/tmc/lib/x/xla"
)

func ExampleBuffer_Shape() {
	var b xla.Buffer
	_, _, err := b.Shape()
	fmt.Println(err) // Output: xla: buffer is closed
}
func ExampleBuffer_Device() {
	var b xla.Buffer
	_, err := b.Device()
	fmt.Println(err) // Output: xla: buffer is closed
}
